from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

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
from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams
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

        token_mask = torch.zeros(len(key), dtype=torch.bool)
        token_mask[device_indices.numel() :] = True

        # In bigram mode (EAGLE), key.token_ids holds N+1 raw tokens for N bigrams,
        # while len(key) == N. The connector expects token_ids and token_mask to share
        # the same logical length, so slice to len(key).
        connector_token_ids = key.token_ids[: len(key)]

        new_hit_length = self._connector.get_new_hit_length(
            token_ids=connector_token_ids,
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
            best_match_node=device_match_result.best_match_node,
            host_hit_length=new_hit_length,
            mamba_branching_seqlen=device_match_result.mamba_branching_seqlen,
            cache_protected_len=device_match_result.cache_protected_len,
        )

    def _empty_device_indices(self) -> torch.Tensor:
        allocator = self._inner_radixtree.token_to_kv_pool_allocator
        device = getattr(allocator, "device", None) or self._inner_radixtree.device
        return torch.empty((0,), dtype=torch.int64, device=device)

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        req = params.req
        mem_quota = params.mem_quota

        if self._connector is None:
            return self._empty_device_indices(), req.last_node

        host_hit_length = req.host_hit_length

        if host_hit_length <= 0 or (
            mem_quota is not None and host_hit_length > mem_quota
        ):
            self._connector.release_load_state(req.rid)
            return self._empty_device_indices(), req.last_node

        device_indices = self._inner_radixtree.token_to_kv_pool_allocator.alloc(
            host_hit_length
        )
        if device_indices is None:
            self.evict(EvictParams(num_tokens=host_hit_length))
            device_indices = self._inner_radixtree.token_to_kv_pool_allocator.alloc(
                host_hit_length
            )
        if device_indices is None:
            logger.warning(
                "Failed to allocate %d GPU slots for external load",
                host_hit_length,
            )
            self._connector.release_load_state(req.rid)
            return self._empty_device_indices(), req.last_node

        gpu_cached_len = len(req.prefix_indices)
        key = RadixKey(
            token_ids=req.fill_ids[gpu_cached_len : gpu_cached_len + host_hit_length],
            extra_key=req.extra_key,
        )

        last_node = req.last_node
        new_node = TreeNode()
        new_node.key = key
        new_node.value = device_indices
        new_node.parent = last_node
        last_node.children[self._inner_radixtree.get_child_key_fn(new_node.key)] = (
            new_node
        )
        self._inner_radixtree.evictable_size_ += len(device_indices)
        self._inner_radixtree._record_store_event(new_node)

        _pre_full_lock = getattr(new_node, "full_lock_ref", None)
        _pre_swa_lock = getattr(new_node, "swa_lock_ref", None)
        self._inner_radixtree.inc_lock_ref(new_node)

        self._load_queue.append(
            LoadOperation(
                rid=req.rid,
                device_indices=device_indices,
                node=new_node,
            )
        )

        logger.info(
            "[LOAD-DEBUG init_load_back] rid=%s host_hit_length=%d "
            "new_node.id=%s new_node.full_lock_ref=%s->%s new_node.swa_lock_ref=%s->%s "
            "load_queue_size=%d ongoing_load_tasks_size=%d",
            getattr(req, "rid", None),
            host_hit_length,
            getattr(new_node, "id", None),
            _pre_full_lock,
            getattr(new_node, "full_lock_ref", None),
            _pre_swa_lock,
            getattr(new_node, "swa_lock_ref", None),
            len(self._load_queue),
            len(self._ongoing_load_tasks),
        )

        req.prefix_indices = torch.cat([req.prefix_indices, device_indices])
        req.last_node = new_node
        return device_indices, new_node

    def ready_to_load_host_cache(self) -> int:
        if self._connector is None or not self._load_queue:
            return -1

        task_id = self._load_task_id_counter
        self._load_task_id_counter += 1

        self._connector.start_load_kv(task_id, self._load_queue)

        nodes = [op.node for op in self._load_queue]
        self._ongoing_load_tasks[task_id] = nodes

        logger.info(
            "[LOAD-DEBUG ready_to_load_host_cache] task_id=%d n_ops=%d "
            "node_ids=%s node_full_locks=%s "
            "ongoing_load_tasks_size=%d",
            task_id,
            len(nodes),
            [getattr(n, "id", None) for n in nodes],
            [getattr(n, "full_lock_ref", None) for n in nodes],
            len(self._ongoing_load_tasks),
        )

        self._load_queue.clear()
        return task_id

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        # Save kv_committed_len before super() pops it (pop_committed_kv_cache
        # sets kv_committed_freed=True and cannot be called again).
        kv_committed_len = req.kv_committed_len

        is_eagle = getattr(self._inner_radixtree, "is_eagle", False)

        raw_token_ids = None
        cache_to_connector = False
        if self._connector is not None and is_insert:
            req_id = req.req_pool_idx
            # Keep the FULL committed raw token slice. With EAGLE bigram the
            # inner tree stores keys as bigrams: N bigrams need N+1 raw tokens,
            # so we must NOT page-align in raw-token units here — that drops
            # the boundary token and shortens the bigram count by page_size,
            # which made the re-match below return half the expected length
            # and silently leak the inserted node's slots.
            raw_token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
            if len(raw_token_ids) > 0 and req_id is not None:
                cache_to_connector = True

        # Let the inner radix tree do insert + free duplicates + dec_lock_ref.
        # SWARadixCache / RadixCache.cache_finished_req does not accept **kwargs,
        # so drop unknown keys instead of forwarding them blindly.
        self._inner_radixtree.cache_finished_req(req, is_insert=is_insert)

        if not cache_to_connector:
            return

        # Re-match the tree to get the actual leaf node and its kv_indices
        # AFTER insert.  These kv_indices are the tree node values (not the
        # req_to_token_pool snapshot), so they are protected by lock_ref and
        # won't be freed by the allocator while D2H transfer is in flight.
        #
        # The inner tree inserts under the bigram view when is_eagle=True
        # (see swa_radix_cache.insert / radix_cache.insert). Mirror that here:
        # build the lookup key with is_bigram=is_eagle and let RadixKey's own
        # page_aligned() compute the page-aligned LOGICAL length (bigram count
        # under EAGLE, raw-token count otherwise).
        radix_key = RadixKey(
            raw_token_ids, req.extra_key, is_bigram=is_eagle
        ).page_aligned(self.page_size)
        page_aligned_len = len(radix_key)
        if page_aligned_len == 0:
            return

        match_result = self._inner_radixtree.match_prefix(
            MatchPrefixParams(key=radix_key)
        )
        new_last_node = match_result.last_device_node
        if new_last_node is None or new_last_node is self._inner_radixtree.root_node:
            return

        kv_indices = match_result.device_indices
        if kv_indices is None or kv_indices.numel() == 0:
            return

        if page_aligned_len != kv_indices.numel():
            logger.warning(
                "[FlexKV] cache_finished_req: length mismatch! "
                "page_aligned_len=%d, kv_indices.numel()=%d, "
                "is_eagle=%s, skipping store",
                page_aligned_len, kv_indices.numel(), is_eagle,
            )
            return

        # The connector consumes a flat token_ids list paired 1:1 with
        # kv_indices, so feed it the raw-token prefix that maps to those
        # page_aligned_len logical units. Under EAGLE that prefix is one
        # token longer than the bigram count (the trailing boundary token).
        connector_token_ids = raw_token_ids[: len(radix_key.token_ids)]

        # Lock the resolved tree node BEFORE starting async D2H transfer
        # so that evict cannot free these pages while transfer is in flight.
        _pre_full_lock = getattr(new_last_node, "full_lock_ref", None)
        _pre_swa_lock = getattr(new_last_node, "swa_lock_ref", None)
        _pre_full_evict = getattr(self._inner_radixtree, "full_evictable_size_", getattr(self._inner_radixtree, "evictable_size_", -1))
        _pre_full_prot = getattr(self._inner_radixtree, "full_protected_size_", getattr(self._inner_radixtree, "protected_size_", -1))
        _pre_swa_evict = getattr(self._inner_radixtree, "swa_evictable_size_", -1)
        _pre_swa_prot = getattr(self._inner_radixtree, "swa_protected_size_", -1)
        try:
            _pre_full_avail = self._inner_radixtree.token_to_kv_pool_allocator.full_available_size()
        except Exception:
            try:
                _pre_full_avail = self._inner_radixtree.token_to_kv_pool_allocator.available_size()
            except Exception:
                _pre_full_avail = -1

        # Walk path from new_last_node up to root and capture each node's
        # pre-inc state so we can see if SWARadixCache.inc_lock_ref actually
        # bumps full_protected_size_ for each one.
        _path_pre = []
        _walker = new_last_node
        while _walker is not None and _walker is not getattr(self._inner_radixtree, "root_node", None):
            _path_pre.append((
                getattr(_walker, "id", None),
                (0 if getattr(_walker, "value", None) is None else len(_walker.value)),
                getattr(_walker, "full_lock_ref", None),
                getattr(_walker, "swa_lock_ref", None),
                getattr(_walker, "swa_tombstone", None),
            ))
            _walker = getattr(_walker, "parent", None)

        self._inner_radixtree.inc_lock_ref(new_last_node)

        _post_full_evict = getattr(self._inner_radixtree, "full_evictable_size_", getattr(self._inner_radixtree, "evictable_size_", -1))
        _post_full_prot = getattr(self._inner_radixtree, "full_protected_size_", getattr(self._inner_radixtree, "protected_size_", -1))
        _post_swa_evict = getattr(self._inner_radixtree, "swa_evictable_size_", -1)
        _post_swa_prot = getattr(self._inner_radixtree, "swa_protected_size_", -1)

        _path_post = []
        _walker = new_last_node
        while _walker is not None and _walker is not getattr(self._inner_radixtree, "root_node", None):
            _path_post.append((
                getattr(_walker, "id", None),
                getattr(_walker, "full_lock_ref", None),
                getattr(_walker, "swa_lock_ref", None),
            ))
            _walker = getattr(_walker, "parent", None)

        task_id = self._load_task_id_counter
        self._load_task_id_counter += 1

        logger.info(
            "[STORE-DEBUG cache_finished_req INC_LOCK_DIFF] rid=%s task_id=%d "
            "node.id=%s full_lock=%s->%s swa_lock=%s->%s "
            "full_evict %s->%s (delta=%s) full_prot %s->%s (delta=%s) "
            "swa_evict %s->%s (delta=%s) swa_prot %s->%s (delta=%s) "
            "page_aligned_len=%d kv_indices.numel=%d "
            "path_pre=%s path_post=%s",
            getattr(req, "rid", None),
            task_id,
            getattr(new_last_node, "id", None),
            _pre_full_lock, getattr(new_last_node, "full_lock_ref", None),
            _pre_swa_lock, getattr(new_last_node, "swa_lock_ref", None),
            _pre_full_evict, _post_full_evict,
            (_post_full_evict - _pre_full_evict) if isinstance(_pre_full_evict, int) and isinstance(_post_full_evict, int) else "n/a",
            _pre_full_prot, _post_full_prot,
            (_post_full_prot - _pre_full_prot) if isinstance(_pre_full_prot, int) and isinstance(_post_full_prot, int) else "n/a",
            _pre_swa_evict, _post_swa_evict,
            (_post_swa_evict - _pre_swa_evict) if isinstance(_pre_swa_evict, int) and isinstance(_post_swa_evict, int) else "n/a",
            _pre_swa_prot, _post_swa_prot,
            (_post_swa_prot - _pre_swa_prot) if isinstance(_pre_swa_prot, int) and isinstance(_post_swa_prot, int) else "n/a",
            page_aligned_len,
            kv_indices.numel(),
            _path_pre,
            _path_post,
        )

        self._connector.start_store_kv(
            task_id=task_id,
            token_ids=connector_token_ids,
            kv_indices=kv_indices,
        )

        self._ongoing_store_tasks[task_id] = new_last_node

        logger.info(
            "[STORE-DEBUG cache_finished_req POST-START_STORE] rid=%s task_id=%d "
            "ongoing_store_tasks_size=%d ongoing_task_ids=%s",
            getattr(req, "rid", None),
            task_id,
            len(self._ongoing_store_tasks),
            list(self._ongoing_store_tasks.keys()),
        )

    def evict(self, params: EvictParams) -> EvictResult:
        return self._inner_radixtree.evict(params)

    def check_kv_events(self):
        if self._connector is None:
            return
        # [LEAK-DEBUG] Right before draining store/load completions, compare
        # the getter that invariant_checker reads (self.protected_size() ->
        # inner.protected_size()) against the raw field that inc_lock_ref
        # writes to (inner.full_protected_size_). Each getter wrapped
        # individually so one failure doesn't blank the whole line.
        def _safe(label, fn):
            try:
                return fn()
            except BaseException as _e:
                return f"<{label}-err:{type(_e).__name__}:{_e!r}>"

        _inner = self._inner_radixtree
        # invariant_checker reads full_protected_size()/swa_protected_size()
        # under hybrid-SWA, NOT protected_size() (which raises
        # NotImplementedError on SWARadixCache). Compare those method getters
        # against the raw fields inc_lock_ref writes to, to tell whether the
        # leak is a getter-vs-field divergence (possibility A) or the field
        # being zeroed between inc_lock_ref and the idle check (possibility B).
        _tc_prot = _safe("full_prot()", lambda: _inner.full_protected_size())
        _tc_evict = _safe("full_evict()", lambda: _inner.full_evictable_size())
        _inner_prot_method = _safe("swa_prot()", lambda: _inner.swa_protected_size())
        _inner_evict_method = _safe("swa_evict()", lambda: _inner.swa_evictable_size())
        _full_prot_field = getattr(_inner, "full_protected_size_", "<missing>")
        _full_evict_field = getattr(_inner, "full_evictable_size_", "<missing>")
        _swa_prot_field = getattr(_inner, "swa_protected_size_", "<missing>")
        _swa_evict_field = getattr(_inner, "swa_evictable_size_", "<missing>")
        _interesting = (
            bool(self._ongoing_store_tasks)
            or bool(self._ongoing_load_tasks)
            or isinstance(_full_prot_field, int) and _full_prot_field != 0
            or isinstance(_full_evict_field, int) and _full_evict_field != 0
        )
        if _interesting:
            logger.info(
                "[LEAK-DEBUG check_kv_events PRE-DRAIN] "
                "full_protected_size()=%s full_evictable_size()=%s "
                "swa_protected_size()=%s swa_evictable_size()=%s "
                "inner.full_protected_size_=%s inner.full_evictable_size_=%s "
                "inner.swa_protected_size_=%s inner.swa_evictable_size_=%s "
                "ongoing_store=%s ongoing_load=%s "
                "inner_type=%s",
                _tc_prot, _tc_evict,
                _inner_prot_method, _inner_evict_method,
                _full_prot_field, _full_evict_field,
                _swa_prot_field, _swa_evict_field,
                list(self._ongoing_store_tasks.keys()),
                list(self._ongoing_load_tasks.keys()),
                type(_inner).__name__,
            )
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
        if completed_ids or self._ongoing_store_tasks:
            logger.info(
                "[STORE-DEBUG check_store_completion] completed_ids=%s "
                "ongoing_store_tasks_size=%d ongoing_task_ids=%s",
                completed_ids,
                len(self._ongoing_store_tasks),
                list(self._ongoing_store_tasks.keys()),
            )
        for task_id in completed_ids:
            node = self._ongoing_store_tasks.pop(task_id, None)
            if node is not None:
                try:
                    _full_avail = self._inner_radixtree.token_to_kv_pool_allocator.full_available_size()
                except Exception:
                    try:
                        _full_avail = self._inner_radixtree.token_to_kv_pool_allocator.available_size()
                    except Exception:
                        _full_avail = -1
                _full_evict = getattr(self._inner_radixtree, "full_evictable_size_", getattr(self._inner_radixtree, "evictable_size_", -1))
                _full_prot = getattr(self._inner_radixtree, "full_protected_size_", getattr(self._inner_radixtree, "protected_size_", -1))
                logger.info(
                    "[STORE-DEBUG check_store_completion DEC-LOCK] task_id=%d "
                    "node.id=%s node.full_lock_ref=%s node.swa_lock_ref=%s "
                    "full_avail=%s full_evict=%s full_prot=%s",
                    task_id,
                    getattr(node, "id", None),
                    getattr(node, "full_lock_ref", None),
                    getattr(node, "swa_lock_ref", None),
                    _full_avail,
                    _full_evict,
                    _full_prot,
                )
                self._inner_radixtree.dec_lock_ref(node)
            else:
                logger.info(
                    "[STORE-DEBUG check_store_completion ORPHAN] task_id=%d not in ongoing_store_tasks",
                    task_id,
                )

    def _check_load_completion(self) -> None:
        completed_ids = self._connector.check_completed_load_tasks()
        if completed_ids or self._ongoing_load_tasks:
            logger.info(
                "[LOAD-DEBUG check_load_completion] completed_ids=%s "
                "ongoing_load_tasks_size=%d ongoing_task_ids=%s",
                completed_ids,
                len(self._ongoing_load_tasks),
                list(self._ongoing_load_tasks.keys()),
            )
        for task_id in completed_ids:
            nodes = self._ongoing_load_tasks.pop(task_id, None)
            if nodes is not None:
                for node in nodes:
                    logger.info(
                        "[LOAD-DEBUG check_load_completion DEC-LOCK] task_id=%d "
                        "node.id=%s node.full_lock_ref=%s node.swa_lock_ref=%s",
                        task_id,
                        getattr(node, "id", None),
                        getattr(node, "full_lock_ref", None),
                        getattr(node, "swa_lock_ref", None),
                    )
                    self._inner_radixtree.dec_lock_ref(node)
            else:
                logger.info(
                    "[LOAD-DEBUG check_load_completion ORPHAN] task_id=%d not in ongoing_load_tasks",
                    task_id,
                )

    # -- Pass-through methods --

    def insert(self, key, value=None, **kwargs):
        return self._inner_radixtree.insert(key, value=value, **kwargs)

    def inc_lock_ref(self, node):
        return self._inner_radixtree.inc_lock_ref(node)

    def dec_lock_ref(self, node, params=None):
        if params is None:
            return self._inner_radixtree.dec_lock_ref(node)
        return self._inner_radixtree.dec_lock_ref(node, params)

    def cache_unfinished_req(self, req: Req, chunked: bool = False, **kwargs):
        # Inner caches (RadixCache/SWARadixCache/MambaRadixCache/...) only
        # accept ``chunked=``; drop any extra kwargs from upstream wrappers
        # (e.g. StreamingSession) to keep cross-impl compatibility.
        return self._inner_radixtree.cache_unfinished_req(req, chunked=chunked)

    def evictable_size(self):
        return self._inner_radixtree.evictable_size()

    def protected_size(self):
        return self._inner_radixtree.protected_size()

    # NOTE: BasePrefixCache gives non-raising DEFAULT implementations for the
    # methods below (the size getters return 0, supports_* return False).
    # Because the base class defines them, normal attribute lookup resolves
    # them BEFORE __getattr__ fires, so __getattr__ never delegates them to the
    # inner cache and the inner's real value is silently shadowed by the base
    # default. Every method here is overridden by at least one inner cache we
    # wrap (SWARadixCache / MambaRadixCache / HiMambaRadixCache / ChunkCache),
    # so it MUST be forwarded explicitly.
    #
    # This shadowing previously caused a phantom "pool memory leak detected"
    # crash: under hybrid-SWA, invariant_checker and pool_stats_observer read
    # tree_cache.full_protected_size()/swa_protected_size()/full_evictable_size()
    # /swa_evictable_size(), got the base-class 0 instead of the inner
    # SWARadixCache's locked sizes, and mis-counted the locked pages as leaked.
    #
    # Methods NOT forwarded on purpose: session_held_* / release_session are
    # only overridden by SessionAwareCache (a sibling wrapper, never our
    # inner), and SessionAwareCache uses a different signature (no
    # active_pool_idxs) — forwarding them here would add dead code plus a
    # positional-arg landmine, so we leave them to __getattr__ / base default.
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
        for node in self._ongoing_store_tasks.values():
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
