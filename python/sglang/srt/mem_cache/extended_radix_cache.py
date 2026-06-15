from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    EvictResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.kv_connector import BaseKVConnector, LoadOperation
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams, InsertParams, MatchPrefixParams
if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

logger = logging.getLogger(__name__)


class ExtendedRadixCache(BasePrefixCache):
    """RadixCache decorator with external KV storage connector.

    Wraps any BasePrefixCache implementation (RadixCache, SWARadixCache, etc.)
    and adds external KV storage capabilities via a BaseKVConnector.
    """

    def __init__(
        self,
        params: CacheInitParams,
        connector: Optional[BaseKVConnector] = None,
        inner_cache: Optional[BasePrefixCache] = None,
    ):
        # Use provided inner cache, or create a default RadixCache
        if inner_cache is not None:
            self._inner_radixtree = inner_cache
        else:
            self._inner_radixtree = RadixCache(params)
        self._connector = connector

        self._load_task_id_counter = 0
        self._load_queue: List[LoadOperation] = []
        self._ongoing_load_tasks: Dict[int, List[TreeNode]] = {}
        self._ongoing_store_tasks: Dict[int, TreeNode] = {}

    # -- Forward PrefixCacheTrait properties to inner cache --

    @property
    def req_to_token_pool(self):
        return self._inner_radixtree.req_to_token_pool

    @req_to_token_pool.setter
    def req_to_token_pool(self, value):
        self._inner_radixtree.req_to_token_pool = value

    @property
    def token_to_kv_pool_allocator(self):
        return self._inner_radixtree.token_to_kv_pool_allocator

    @token_to_kv_pool_allocator.setter
    def token_to_kv_pool_allocator(self, value):
        self._inner_radixtree.token_to_kv_pool_allocator = value

    @property
    def page_size(self):
        return self._inner_radixtree.page_size

    @page_size.setter
    def page_size(self, value):
        self._inner_radixtree.page_size = value

    @property
    def disable(self):
        return self._inner_radixtree.disable

    @property
    def device(self):
        return self._inner_radixtree.device

    @property
    def metrics_collector(self):
        return self._inner_radixtree.metrics_collector

    @metrics_collector.setter
    def metrics_collector(self, value):
        self._inner_radixtree.metrics_collector = value

    @property
    def layer_done_counter(self):
        if self._connector is None:
            return None
        return self._connector.layer_done_counter

    # -- Core methods with connector logic --

    def reset(self):
        self._ongoing_store_tasks.clear()
        self._ongoing_load_tasks.clear()
        self._load_queue.clear()

        if self._connector is not None:
            self._connector.reset()

        self._inner_radixtree.reset()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        device_match_result = self._inner_radixtree.match_prefix(params)
        if self._connector is None:
            return device_match_result

        key = params.key
        device_indices: torch.Tensor = device_match_result.device_indices
        last_device_node = device_match_result.last_device_node

        uncached_len = len(key) - device_indices.numel()
        if uncached_len <= 0:
            if params.req is not None:
                params.req.cached_tokens_extended_device = 0
            return device_match_result

        # ``key.token_ids`` may be longer than ``len(key)`` in EAGLE bigram
        # mode (N raw tokens make N-1 bigrams).  The connector consumes
        # token_ids and a parallel mask, so both must have the same length —
        # use ``len(key)`` (the logical bigram count) as the source of truth
        # and slice token_ids to match.
        n = len(key)
        token_ids_for_connector = key.token_ids[:n] if hasattr(key, 'token_ids') else key.token_ids
        token_mask = torch.zeros(n, dtype=torch.bool)
        token_mask[device_indices.numel() :] = True

        new_hit_length = self._connector.get_new_hit_length(
            token_ids=token_ids_for_connector,
            token_mask=token_mask,
            update_state_for_load=params.update_connector_state,
            rid=params.req.rid if params.req is not None else None,
        )

        if params.req is not None:
            params.req.cached_tokens_extended_device = new_hit_length

        return MatchResult(
            device_indices=device_indices,
            last_device_node=last_device_node,
            last_host_node=last_device_node,
            best_match_node=last_device_node,
            host_hit_length=new_hit_length,
        )

    def init_load_back(
        self,
        params: InitLoadBackParams
    ):
        """Reserve GPU slots, build a new tree node, and queue the H2D op.

        Returns (new_indices, last_node) tuple — the schedule_policy caller
        unpacks this directly:
            new_indices, req.last_node = tree_cache.init_load_back(...)
        On any early-exit path we return (empty_tensor, req.last_node) to
        keep that contract intact.
        """
        req = params.req
        mem_quota = params.mem_quota

        # Empty tensor sized to req.prefix_indices' device, so torch.cat works.
        empty_indices = torch.empty(
            (0,), dtype=torch.int64, device=self._inner_radixtree.device
        )

        if self._connector is None:
            return empty_indices, req.last_node

        host_hit_length = req.host_hit_length

        if host_hit_length <= 0 or (
            mem_quota is not None and host_hit_length > mem_quota
        ):
            self._connector.release_load_state(req.rid)
            return empty_indices, req.last_node

        # Allocate GPU slots for host_hit_length tokens. Three allocator regimes:
        #   * page_size == 1: plain RadixCache / SWARadixCache w/ token-level alloc.
        #     `.alloc(N)` works directly.
        #   * page_size  > 1, no SWA: DSv4 paged allocator without window — use
        #     `alloc_extend()` for the whole hit length.
        #   * page_size  > 1, hybrid SWA (DSv4): full layers get the whole hit,
        #     but SWA layers only get the trailing window. We MUST call
        #     `alloc_extend_swa_tail(extend_num_tokens=H, swa_tail_len=window)`
        #     because the SWA pool is sized to ~window-per-request and
        #     never has room for H tokens of SWA data — and FlexKV only stored
        #     `window` tokens of SWA data on the host side anyway.
        allocator = self._inner_radixtree.token_to_kv_pool_allocator
        page_size = getattr(allocator, "page_size", 1) or 1
        # Pull window size from the connector if it exposes one (FlexKV does).
        window_size = getattr(self._connector, "_swa_window_size", 0) or 0
        # Detect hybrid-SWA paged allocator by the alloc_extend_swa_tail method.
        has_swa_tail = hasattr(allocator, "alloc_extend_swa_tail")

        def _swa_tail_len() -> int:
            # SWA slots to allocate/map for the trailing window. Use CEIL to a
            # page so the mapped tail node is >= sliding_window_size tokens.
            # _insert_helper splits the unmapped head into a `swa_tombstone`
            # node; if the mapped tail were < window, inc_lock_ref would not
            # finish locking the window inside the tail and would walk up into
            # the tombstone head, hitting `assert not swa_tombstone`. Cap at
            # host_hit_length (which is already page-aligned) and re-floor to a
            # page as a defensive guard.
            tail = min(
                ((window_size + page_size - 1) // page_size) * page_size,
                host_hit_length,
            )
            tail = (tail // page_size) * page_size
            return tail if tail > 0 else page_size

        def _alloc_paged():
            # prefix_lens is the device-cached length so far. host_hit_length is
            # appended after that.
            gpu_prefix_len = int(req.prefix_indices.numel())
            seq_len = gpu_prefix_len + host_hit_length
            device = self._inner_radixtree.device
            prefix_lens = torch.tensor([gpu_prefix_len], dtype=torch.int64, device=device)
            prefix_lens_cpu = torch.tensor([gpu_prefix_len], dtype=torch.int64, device="cpu")
            seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
            seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64, device="cpu")
            # last_loc: previous token's slot index, or -1 sentinel when empty.
            if gpu_prefix_len > 0:
                last_loc = req.prefix_indices[-1:].to(device=device, dtype=torch.int64)
            else:
                last_loc = torch.tensor([-1], dtype=torch.int64, device=device)
            if has_swa_tail and window_size > 0:
                # Hybrid SWA: only allocate the trailing window of SWA slots
                # regardless of how many full slots we need (see _swa_tail_len
                # for the ceil-to-page rationale vs. inc_lock_ref).
                swa_tail_len = _swa_tail_len()
                return allocator.alloc_extend_swa_tail(
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    host_hit_length,
                    swa_tail_len,
                )
            return allocator.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                host_hit_length,
            )

        def _do_alloc():
            if page_size == 1:
                return allocator.alloc(host_hit_length)
            # Page-aligned paged allocator path
            if host_hit_length % page_size != 0:
                logger.warning(
                    "[FlexKV] host_hit_length=%d not page-aligned (page_size=%d), "
                    "skipping H2D for rid=%s",
                    host_hit_length, page_size, req.rid,
                )
                return None
            return _alloc_paged()

        swa_alloc_need = host_hit_length
        if has_swa_tail and window_size > 0 and page_size > 1:
            swa_alloc_need = _swa_tail_len()

        device_indices = _do_alloc()
        if device_indices is None:
            from sglang.srt.mem_cache.common import evict_from_tree_cache

            evict_from_tree_cache(
                self,
                host_hit_length,
                swa_num_tokens=swa_alloc_need,
            )
            device_indices = _do_alloc()
        if device_indices is None:
            from sglang.srt.mem_cache.common import available_and_evictable_str

            logger.warning(
                "Failed to allocate %d GPU slots for external load (swa_need=%d). %s",
                host_hit_length,
                swa_alloc_need,
                available_and_evictable_str(self),
            )
            self._connector.release_load_state(req.rid)
            return empty_indices, req.last_node

        gpu_cached_len = len(req.prefix_indices)
        # Sanity-check the effective length that ``_inner_radixtree.insert``
        # will keep in the tree.  insert() applies two transforms:
        #   1. EAGLE bigram view: ``len(key) -> len(key) - 1`` when is_eagle.
        #   2. page_aligned trim: ``len(key) -> (len(key) // page_size) * page_size``.
        # If ``effective_len`` is 0, the insert is a no-op and re-match would
        # land on root; bail cleanly to avoid wasting a transfer.
        #
        # We do NOT pre-trim ``device_indices`` to ``effective_len``: the
        # connector already prepared the FlexKV transfer graph with
        # ``host_hit_length`` source blocks, and shrinking the destination
        # mapping would crash with ``src_block_ids.size != dst_block_ids.size``
        # in the worker. Instead we mirror normal sglang prefill: pass the
        # FULL ``device_indices`` to the load op, let ``insert`` do its
        # internal trim, and rely on the trailing slots being soft-locked
        # by ``req.prefix_indices`` (the same way they are after a normal
        # paged prefill — those slots are "allocator-allocated, not
        # tree-tracked" and stay alive for the lifetime of the request).
        is_eagle = bool(getattr(self._inner_radixtree, 'is_eagle', False))
        effective_len = host_hit_length - (1 if is_eagle else 0)
        if page_size > 1:
            # TEMP FIX (999-temp-fix-init-load-back.patch): the original
            # implementation used floor-page-align, which under EAGLE bigram
            # always trims a host_hit_length that's already a page multiple
            # to 0:
            #     hit_length=256, page_size=256, eagle=True
            #     → effective_len = 256 - 1 = 255
            #     → floor(255/256)*256 = 0
            #     → "trims to 0" warning, rolls back alloc, H2D never fires.
            #
            # The store path (cache_finished_req above) page-aligns
            # ``len(token_ids)`` BEFORE applying EAGLE's bigram view, so
            # for hit_length=256 it stores exactly 256 tokens. For load to
            # match store, we want effective_len to round to the same
            # 256-token boundary.
            #
            # Fix: under EAGLE, ceil-page-align AFTER the bigram trim,
            # capped at host_hit_length so we never exceed what FlexKV
            # actually has on the host side.
            if is_eagle and effective_len > 0:
                effective_len = min(
                    ((effective_len + page_size - 1) // page_size) * page_size,
                    host_hit_length,
                )
            else:
                effective_len = (effective_len // page_size) * page_size
        if effective_len <= 0:
            logger.warning(
                "[FlexKV] init_load_back: host_hit_length=%d trims to 0 after "
                "is_eagle=%s + page_align(%d) for rid=%s; rolling back alloc.",
                host_hit_length, is_eagle, page_size, req.rid,
            )
            try:
                allocator.free(device_indices)
            except Exception:
                pass
            self._connector.release_load_state(req.rid)
            return empty_indices, req.last_node

        # IMPORTANT: insert with the FULL key from sequence start, not a slice
        # starting at gpu_cached_len. insert() always navigates from root_node;
        # a relative slice would be treated as starting at position 0 and create
        # a PARALLEL node hanging off root (or the wrong parent), duplicating the
        # loaded slots. The subsequent cache_unfinished_req / cache_finished_req
        # re-inserts the same prefix with the full key and lands the slots at the
        # correct position, leaving two tree nodes that own the same physical KV
        # slots -> tree-internal duplication -> pool leak. Mirror the re-match
        # below (which already uses the full key) and pass the device-cached
        # prefix slots as the head of `value` so insert can dedup against the
        # existing prefix nodes. `prev_prefix_len=gpu_cached_len` makes
        # _insert_helper skip (not free) that already-in-tree prefix region.
        key = RadixKey(
            token_ids=req.fill_ids[: gpu_cached_len + host_hit_length],
            extra_key=req.extra_key,
        )
        # Full value = device-cached prefix slots (currently in req.prefix_indices,
        # before the post-insert cat below) ++ freshly loaded tail slots.
        full_value = torch.cat([req.prefix_indices.to(device_indices), device_indices])

        # Delegate node creation to the inner cache's insert() API. This is
        # important because:
        #   * SWARadixCache uses its own TreeNode subclass with full_lock_ref/
        #     swa_lock_ref/swa_uuid/swa_prev/swa_next fields. Manually creating
        #     a generic radix_cache.TreeNode would crash inc_lock_ref with
        #     AttributeError: 'TreeNode' object has no attribute 'full_lock_ref'.
        #   * insert() correctly updates lru_list, evictable_size_, kv events,
        #     splits parent nodes, etc. — none of which we want to re-implement.
        #
        # After insert, we re-match the same key under root to find the new
        # leaf node we just created (or the merged leaf if the prefix already
        # existed). That node is what we lock and queue for H2D.
        #
        # Tail-only SWA mapping: alloc_extend_swa_tail mapped only the trailing
        # `swa_tail_len` full slots to the SWA pool; the head is unmapped. Pass
        # that boundary as `swa_evicted_seqlen` so _insert_helper splits the
        # head into a `swa_tombstone` node (full KV kept, no SWA) and the tail
        # into a normal node (homogeneously SWA-mapped). This keeps every node
        # SWA-homogeneous so `_swa_slots_in_value == len(value)` holds and the
        # inc/dec lock accounting stays consistent. Boundary is relative to the
        # key start (total_prefix_length is 0 for this freshly inserted prefix).
        # Skip for eagle: the bigram view shifts the key by 1 and would misalign
        # the boundary; eagle+hybrid-SWA is not a supported combination.
        # swa_evicted_seqlen is now ABSOLUTE from sequence start (key starts at 0):
        # the loaded tail spans [gpu_cached_len, gpu_cached_len + effective_len);
        # its head [.., gpu_cached_len + effective_len - swa_tail_len) is unmapped.
        swa_evicted_seqlen = 0
        if has_swa_tail and window_size > 0 and not is_eagle:
            swa_evicted_seqlen = gpu_cached_len + max(
                0, effective_len - _swa_tail_len()
            )
        try:
            self._inner_radixtree.insert(
                InsertParams(
                    key=key,
                    value=full_value,
                    prev_prefix_len=gpu_cached_len,
                    swa_evicted_seqlen=swa_evicted_seqlen,
                )
            )
        except TypeError:
            # Fallback for caches whose insert() takes positional args
            self._inner_radixtree.insert(InsertParams(key=key, value=full_value))

        # Persist the SWA eviction frontier onto the req. The boundary computed
        # above is an INTRINSIC property of this prefix: alloc_extend_swa_tail
        # physically mapped only the trailing `swa_tail_len` slots, so the head
        # has no SWA pool backing — permanently, not just for this one insert.
        # `swa_evicted_seqlen` is already absolute from sequence start. Without
        # persisting it, the NEXT cache_unfinished_req / cache_finished_req for
        # the same req would re-insert the whole prefix with swa_evicted_seqlen=0,
        # recreating a heterogeneous (head-unmapped) non-tombstone node and
        # tripping the SWA-homogeneity invariant. Use max() so we never walk the
        # frontier backwards past an _evict_swa update.
        if swa_evicted_seqlen > 0:
            req.swa_evicted_seqlen = max(req.swa_evicted_seqlen, swa_evicted_seqlen)

        # Re-match the full prefix path so we get the actual leaf node.
        # Use a fresh key starting from root with the same token_ids that
        # are now in the tree, so the match traverses to the new leaf.
        #
        # IMPORTANT: pass the *original* host_hit_length tokens (not
        # effective_len), because match_prefix internally re-applies the
        # same bigram + page_align transform that insert did. Passing only
        # effective_len tokens would shrink AGAIN inside match_prefix
        # (effective_len -> effective_len - 1 (bigram) -> page_align), and
        # match could fall short of the leaf we just created.
        full_key = RadixKey(
            token_ids=req.fill_ids[: gpu_cached_len + host_hit_length],
            extra_key=req.extra_key,
        )
        match_result = self._inner_radixtree.match_prefix(
            MatchPrefixParams(key=full_key)
        )
        new_node = match_result.last_device_node
        if new_node is None or new_node is getattr(self._inner_radixtree, 'root_node', None):
            # TEMP FIX (999-temp-fix-init-load-back.patch, layer 2):
            # When insert() is effectively a no-op because EAGLE bigram +
            # page_align trims all host_hit_length tokens to 0, no new
            # tree node is created and match_prefix naturally returns
            # root.  This is the extreme case of "the trailing partial
            # page is not in the tree but is held alive by
            # req.prefix_indices" (comment below).
            #
            # Instead of rolling back the alloc and aborting H2D, fall
            # back to the existing parent node (req.last_node at
            # gpu_cached_len).  We skip inc_lock_ref because no new node
            # was created; ``device_indices`` are still added to
            # ``prefix_indices`` so model forward sees them all.
            logger.info(
                "[FlexKV] init_load_back: re-match returned root for rid=%s, "
                "host_hit_length=%d (EAGLE bigram+page_align trimmed insert to 0). "
                "Falling back to parent node for H2D.",
                req.rid, host_hit_length,
            )
            new_node = req.last_node
            load_swa_uuid = None
            # Fall through to H2D queuing below.
        else:
            # Normal path: insert created a node, lock it.
            inc_result = self._inner_radixtree.inc_lock_ref(new_node)
            load_swa_uuid = getattr(inc_result, 'swa_uuid_for_lock', None)

        # NOTE: device_indices may be longer than match_result.device_indices
        # by up to one page when is_eagle + page_size > 1 — that's normal
        # sglang behavior (the trailing partial page is not in the tree but
        # is held alive by req.prefix_indices for the request's forward).
        # The full device_indices is passed to the connector load op (so
        # FlexKV can H2D into all allocated slots, matching its src graph)
        # and to req.prefix_indices (so model forward sees them all).
        # ``new_node`` only reflects the in-tree slots, which is correct
        # for the lock_ref protection.

        self._load_queue.append(
            LoadOperation(
                rid=req.rid,
                device_indices=device_indices,
                node=new_node,
            )
        )
        # Track the (node, swa_uuid) pair keyed by op identity so
        # ready_to_load_host_cache can pair load_ops with their swa_uuids
        # when stashing into _ongoing_load_tasks. Using id() of the
        # LoadOperation works because load_ops live until ready_to_load
        # transfers them to _ongoing_load_tasks (then we drop this).
        if not hasattr(self, '_load_queue_swa_uuids'):
            self._load_queue_swa_uuids: Dict[int, Optional[int]] = {}
        self._load_queue_swa_uuids[id(self._load_queue[-1])] = load_swa_uuid

        req.prefix_indices = torch.cat([req.prefix_indices, device_indices])
        req.last_node = new_node
        # Match the (new_indices, last_node) contract that schedule_policy.add_one_req
        # expects.
        return device_indices, new_node

    def ready_to_load_host_cache(self) -> int:
        if self._connector is None or not self._load_queue:
            return -1

        task_id = self._load_task_id_counter
        self._load_task_id_counter += 1

        self._connector.start_load_kv(task_id, self._load_queue)

        # Pair each (node, swa_uuid) so _check_load_completion can pass the
        # right swa_uuid_for_lock to dec_lock_ref. Without the swa_uuid,
        # SWARadixCache.dec_lock_ref walks the SWA chain to root which can
        # underflow swa_lock_ref or hit ``dec_lock_ref on swa_tombstone node``.
        uuid_map = getattr(self, '_load_queue_swa_uuids', {})
        nodes_with_uuid = [
            (op.node, uuid_map.pop(id(op), None)) for op in self._load_queue
        ]
        self._ongoing_load_tasks[task_id] = nodes_with_uuid

        self._load_queue.clear()
        return task_id

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        # Save kv_committed_len before super() pops it (pop_committed_kv_cache
        # sets kv_committed_freed=True and cannot be called again).
        kv_committed_len = req.kv_committed_len

        token_ids = None
        cache_to_connector = False
        if self._connector is not None and is_insert:
            req_id = req.req_pool_idx
            token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
            # Truncate to page boundary (page_align_floor equivalent)
            page_aligned_len = (len(token_ids) // self.page_size) * self.page_size
            token_ids = token_ids[:page_aligned_len]
            if len(token_ids) > 0 and req_id is not None:
                cache_to_connector = True

        # Let the inner radix tree do insert + free duplicates + dec_lock_ref.
        self._inner_radixtree.cache_finished_req(req, is_insert=is_insert, **kwargs)

        if not cache_to_connector:
            return

        # Re-match the tree to get the actual leaf node and its kv_indices
        # AFTER insert.  These kv_indices are the tree node values (not the
        # req_to_token_pool snapshot), so they are protected by lock_ref and
        # won't be freed by the allocator while D2H transfer is in flight.
        radix_key = RadixKey(token_ids, req.extra_key)
        match_result = self._inner_radixtree.match_prefix(
            MatchPrefixParams(key=radix_key)
        )
        new_last_node = match_result.last_device_node
        if new_last_node is None or new_last_node is self._inner_radixtree.root_node:
            return

        kv_indices = match_result.device_indices
        if kv_indices is None or kv_indices.numel() == 0:
            return

        # The radix tree may return fewer indices than tokens passed in. Common
        # causes:
        #   * EAGLE bigram mode: ``len(key) == len(token_ids) - 1``, so the
        #     match anchors on N-1 bigrams and ``device_indices`` is sized to
        #     N-1, then page-aligned (truncating the unmatched tail).
        #   * Page-alignment in ``match_prefix``: SWARadixCache truncates the
        #     key to a page-aligned length before traversal.
        # We cannot store more tokens than we have indices for, so truncate
        # ``token_ids`` to match ``kv_indices`` length.  If after truncation
        # nothing remains, skip the store (typical for very short requests).
        if len(token_ids) > kv_indices.numel():
            token_ids = token_ids[: kv_indices.numel()]
        if len(token_ids) == 0:
            return
        if len(token_ids) != kv_indices.numel():
            logger.warning(
                "[FlexKV] cache_finished_req: irrecoverable length mismatch! "
                "len(token_ids)=%d, kv_indices.numel()=%d, skipping store",
                len(token_ids), kv_indices.numel(),
            )
            return

        # Lock the resolved tree node BEFORE starting async D2H transfer
        # so that evict cannot free these pages while transfer is in flight.
        # On SWARadixCache, inc_lock_ref locks both full_lock_ref and a
        # bounded prefix of swa_lock_ref (up to sliding_window_size from the
        # leaf). It returns ``IncLockRefResult.swa_uuid_for_lock`` which
        # MUST be passed to dec_lock_ref so the SWA-side release walks the
        # exact same range. Plain RadixCache returns the result but ignores
        # the SWA fields, so this also works there.
        inc_result = self._inner_radixtree.inc_lock_ref(new_last_node)
        swa_uuid_for_lock = getattr(inc_result, 'swa_uuid_for_lock', None)

        task_id = self._load_task_id_counter
        self._load_task_id_counter += 1

        self._connector.start_store_kv(
            task_id=task_id,
            token_ids=token_ids,
            kv_indices=kv_indices,
        )

        self._ongoing_store_tasks[task_id] = (new_last_node, swa_uuid_for_lock)

    def evict(self, params: EvictParams) -> EvictResult:
        return self._inner_radixtree.evict(params)

    def check_kv_events(self):
        if self._connector is None:
            return
        self._check_store_completion()
        self._check_load_completion()

    # Alias for compatibility with scheduler (which calls check_hicache_events)
    def check_hicache_events(self):
        self.check_kv_events()

    def prefetch(self, req: Req) -> None:
        if self._connector is None:
            return
        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        self._connector.prefetch(req.rid, token_ids)

    def check_prefetch_progress(self, req_id: str) -> bool:
        if self._connector is None:
            return True
        return self._connector.check_prefetch_progress(req_id)

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        if self._connector is None:
            return 0
        return self._connector.pop_prefetch_loaded_tokens(req_id)

    def release_aborted_request(self, req_id: str) -> None:
        if self._connector is None:
            return
        self._connector.cancel_prefetch(req_id)

    # -- Private helpers --

    def _check_store_completion(self) -> None:
        completed_ids = self._connector.check_completed_store_tasks()
        for task_id in completed_ids:
            entry = self._ongoing_store_tasks.pop(task_id, None)
            if entry is None:
                continue
            # Backward-compat: entry may be a bare node OR (node, swa_uuid)
            if isinstance(entry, tuple):
                node, swa_uuid = entry
            else:
                node, swa_uuid = entry, None
            # On SWARadixCache, dec_lock_ref needs DecLockRefParams to know
            # how far up the SWA chain to walk; on plain RadixCache the
            # extra param is ignored.
            try:
                from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
                params = DecLockRefParams(swa_uuid_for_lock=swa_uuid)
                self._inner_radixtree.dec_lock_ref(node, params)
            except (ImportError, TypeError):
                self._inner_radixtree.dec_lock_ref(node)

    def _check_load_completion(self) -> None:
        completed_ids = self._connector.check_completed_load_tasks()
        for task_id in completed_ids:
            nodes = self._ongoing_load_tasks.pop(task_id, None)
            if nodes is None:
                continue
            for entry in nodes:
                # Backward-compat: entry may be a bare node OR (node, swa_uuid)
                if isinstance(entry, tuple):
                    node, swa_uuid = entry
                else:
                    node, swa_uuid = entry, None
                try:
                    from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
                    params = DecLockRefParams(swa_uuid_for_lock=swa_uuid)
                    self._inner_radixtree.dec_lock_ref(node, params)
                except (ImportError, TypeError):
                    self._inner_radixtree.dec_lock_ref(node)

    # -- Pass-through methods --

    def insert(self, *args, **kwargs):
        return self._inner_radixtree.insert(*args, **kwargs)

    def inc_lock_ref(self, *args, **kwargs):
        return self._inner_radixtree.inc_lock_ref(*args, **kwargs)

    def dec_lock_ref(self, *args, **kwargs):
        return self._inner_radixtree.dec_lock_ref(*args, **kwargs)

    def cache_unfinished_req(self, *args, **kwargs):
        return self._inner_radixtree.cache_unfinished_req(*args, **kwargs)

    def evictable_size(self):
        inner = self._inner_radixtree
        if inner.supports_swa():
            return inner.full_evictable_size()
        return inner.evictable_size()

    def protected_size(self):
        inner = self._inner_radixtree
        if inner.supports_swa():
            return inner.full_protected_size()
        return inner.protected_size()

    def available_and_evictable_str(self) -> str:
        inner = self._inner_radixtree
        if hasattr(inner, "available_and_evictable_str"):
            return inner.available_and_evictable_str()
        return super().available_and_evictable_str()

    # NOTE: BasePrefixCache gives non-raising DEFAULT implementations for the
    # methods below (the size getters return 0, supports_* return False).
    # Because the base class defines them, normal attribute lookup resolves
    # them BEFORE __getattr__ fires, so __getattr__ never delegates them to the
    # inner cache and the inner's real value is silently shadowed by the base
    # default. Every method here is overridden by at least one inner cache we
    # wrap (SWARadixCache / MambaRadixCache / ...), so it MUST be forwarded
    # explicitly.
    #
    # This shadowing previously caused a phantom "pool memory leak detected"
    # crash: under hybrid-SWA, invariant_checker and pool_stats_observer read
    # tree_cache.full_protected_size()/swa_protected_size()/full_evictable_size()
    # /swa_evictable_size(), got the base-class 0 instead of the inner
    # SWARadixCache's locked sizes, and mis-counted the locked pages as leaked.
    def full_evictable_size(self):
        return self._inner_radixtree.full_evictable_size()

    def swa_evictable_size(self):
        return self._inner_radixtree.swa_evictable_size()

    def full_protected_size(self):
        return self._inner_radixtree.full_protected_size()

    def swa_protected_size(self):
        return self._inner_radixtree.swa_protected_size()

    def supports_swa(self) -> bool:
        return self._inner_radixtree.supports_swa()

    def supports_mamba(self) -> bool:
        return self._inner_radixtree.supports_mamba()

    def is_chunk_cache(self) -> bool:
        return self._inner_radixtree.is_chunk_cache()

    def flush_write_through_acks(self) -> None:
        return self._inner_radixtree.flush_write_through_acks()

    def sanity_check(self, *args, **kwargs):
        # An in-flight async store (FlexKV D2H) holds full_lock_ref/swa_lock_ref
        # on the stored leaf AND every ancestor up to root (see
        # SWARadixCache.inc_lock_ref, which locks the [leaf, root) path) until
        # the transfer completes and _check_store_completion runs dec_lock_ref.
        # The scheduler can go idle and run sanity_check while a store is still
        # in flight (is_fully_idle does not account for connector store tasks),
        # so those nodes are legitimately locked at idle. Collect their ids and
        # exempt them from SWARadixCache's "must be unlocked when idle" assert,
        # mirroring how UnifiedRadixCache.sanity_check tolerates
        # ongoing_write_through / ongoing_load_back nodes.
        inner = self._inner_radixtree
        if not self._ongoing_store_tasks or not inner.supports_swa():
            return inner.sanity_check(*args, **kwargs)

        exempt_node_ids: Set[int] = set()
        root_node = getattr(inner, "root_node", None)
        for entry in self._ongoing_store_tasks.values():
            # main2 stores (node, swa_uuid) tuples; tolerate a bare node too.
            node = entry[0] if isinstance(entry, tuple) else entry
            walker = node
            while walker is not None and walker is not root_node:
                exempt_node_ids.add(walker.id)
                walker = walker.parent
        return inner.sanity_check(exempt_node_ids=exempt_node_ids)

    def total_size(self):
        return self._inner_radixtree.total_size()

    def pretty_print(self):
        return self._inner_radixtree.pretty_print()

    def all_values_flatten(self):
        return self._inner_radixtree.all_values_flatten()

    def take_events(self):
        return self._inner_radixtree.take_events()

    def __getattr__(self, name):
        return getattr(self._inner_radixtree, name)
