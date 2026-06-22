from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the hybrid (full and SWA) KV cache.
"""

import heapq
import time
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

import torch
from numpy import float64

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.events import KVCacheEventMixin
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.utils import split_node_hash_value

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

import logging

logger = logging.getLogger(__name__)


class TreeNode:

    counter = 0
    swa_uuid_counter = 1
    last_access_time_counter_float = float64(1.0)

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        # swa_tombstone is used to indicate the kv indices have been freed for swa layers
        self.swa_tombstone = False
        # invariant: for any node, if swa_lock_ref is locked, full_lock_ref must be locked;
        # if full_lock_ref is locked, swa_lock_ref doesn't need to be locked. So,
        # full_lock_ref is always >= swa_lock_ref.
        self.full_lock_ref = 0
        self.swa_lock_ref = 0
        # last access time is only used for sanity check. LRU is maintained by the lru list.
        self.last_access_time = get_last_access_time()

        self.hit_count = 0
        # store the host indices of KV cache
        self.host_value = None
        # store hash values of each page
        self.hash_value: Optional[List[str]] = None

        # for lru list, invariant:
        # 1. prev has greater last_access_time
        # 2. next has smaller last_access_time
        self.prev = None
        self.next = None
        self.swa_prev = None
        self.swa_next = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1
        self.swa_uuid = None

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __lt__(self, other: TreeNode):
        return self.last_access_time < other.last_access_time


def gen_swa_uuid() -> int:
    TreeNode.swa_uuid_counter += 1
    return TreeNode.swa_uuid_counter


def get_last_access_time() -> float64:
    ret = TreeNode.last_access_time_counter_float
    TreeNode.last_access_time_counter_float += 1.0
    return ret


class LRUList:
    def __init__(self, is_swa_list: bool = False):
        self.is_swa_list = is_swa_list
        if self.is_swa_list:
            self.prv = "swa_prev"
            self.nxt = "swa_next"
            self.lock_ref = "swa_lock_ref"
        else:
            self.prv = "prev"
            self.nxt = "next"
            self.lock_ref = "full_lock_ref"
        # Initialize dummy head and tail nodes
        self.head = TreeNode()  # Most recently used side
        self.tail = TreeNode()  # Least recently used side
        setattr(self.head, self.nxt, self.tail)  # self.head.next = self.tail
        setattr(self.tail, self.prv, self.head)  # self.tail.prev = self.head
        self.cache = {}

    def _add_node(self, node):
        """Helper to add node right after head (most recently used)"""
        self._add_node_after(self.head, node)

    def _add_node_after(self, old_node, new_node):
        """Helper to add node right after old_node"""
        setattr(new_node, self.prv, old_node)  # new_node.prev = old_node
        setattr(
            new_node, self.nxt, getattr(old_node, self.nxt)
        )  # new_node.next = old_node.next
        setattr(
            getattr(old_node, self.nxt), self.prv, new_node
        )  # old_node.next.prev = new_node
        setattr(old_node, self.nxt, new_node)  # old_node.next = new_node

    def _remove_node(self, node):
        """Helper to remove node from linked list"""
        setattr(
            getattr(node, self.prv), self.nxt, getattr(node, self.nxt)
        )  # node.prev.next = node.next
        setattr(
            getattr(node, self.nxt), self.prv, getattr(node, self.prv)
        )  # node.next.prev = node.prev
        # Clear self pointers to break reference cycles among evicted nodes.
        setattr(node, self.prv, None)
        setattr(node, self.nxt, None)

    def _get_lru(self) -> Optional[TreeNode]:
        """
        Get the least recently used node
        """
        if len(self.cache) == 0:
            return None
        return getattr(self.tail, self.prv)

    def reset_node_mru(self, node):
        """
        Move a (existing) node to most recently used position
        """
        assert node.id in self.cache, f"Resetting node {node.id=} not in lru list"
        assert (
            not self.is_swa_list or not node.swa_tombstone
        ), f"Resetting swa tombstone node in swa lru list: {node.id=}"
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(self, node, root_node):
        """
        Move an (existing) node and its parents to most recently used position. Child node is
        more recently used than parent node.
        """
        prev_node = self.head
        while node != root_node:
            # for swa lru list, only reset non-tombstone nodes
            if not self.is_swa_list or not node.swa_tombstone:
                assert (
                    node.id in self.cache
                ), f"Resetting node {node.id=} not in lru list when resetting node and parents mru"
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def insert_mru(self, node):
        """
        Insert a (new) node as most recently used
        """
        assert (
            not self.is_swa_list or not node.swa_tombstone
        ), f"Inserting swa tombstone node in swa lru list: {node.id=}"
        assert (
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in lru list, existing node: {self.cache[node.id].id=}"
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: TreeNode):
        """
        Remove node from lru list
        """
        assert node.id in self.cache, f"Removing node {node.id=} not in lru list"
        assert (
            not self.is_swa_list or not node.swa_tombstone
        ), f"Removing swa tombstone node from swa lru list: {node.id=}"
        del self.cache[node.id]
        self._remove_node(node)

    def get_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used node that is not locked
        """
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used leaf node that is not locked
        """
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)

    def get_prev_no_lock(
        self, node: TreeNode, check_id: bool = True
    ) -> Optional[TreeNode]:
        """
        Get the previous (i.e. more recently used) node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no node in the lru list without lock
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: TreeNode, check_id: bool = True):
        """
        Get the previous (i.e. more recently used) leaf node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0 or len(x.children) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no leaf node in the lru list without lock
        if x == self.head:
            return None
        return x

    def in_list(self, node: Optional[TreeNode]):
        """
        Check if the node is in the lru list
        """
        if not node:
            return False
        return node.id in self.cache

    # Note: this is expensive, only use for debug
    def sanity_check_evictable_size(self):
        """
        Check the evictable size (i.e. the size of the nodes that are not locked)
        """
        node = self.get_lru_no_lock()
        evictable_size = 0
        while self.in_list(node):
            evictable_size += len(node.value)
            node = self.get_prev_no_lock(node)
        return evictable_size

    # Note: this is expensive, only use for debug or idle check
    def sanity_check(
        self,
        tree_cache: SWARadixCache,
        exempt_node_ids: Optional[Set[int]] = None,
    ):
        """
        Check if the lru list is valid by rebuilding the lru list from the tree, heapifying it, and
        checking if the lru list is valid.

        exempt_node_ids: ids of nodes that are legitimately locked at idle (e.g. a
        leaf + its ancestors held by an in-flight FlexKV connector store). These
        skip the "must be unlocked when idle" lock-ref asserts; all other checks
        still apply.
        """
        try:
            if self.is_swa_list:
                nodes = tree_cache._collect_nontombstone_nodes()
            else:
                nodes = tree_cache._collect_all_nodes()
            total_nodes = len(nodes)
            total_lru_plus_1 = len(self.cache) + 1
            # heapify based on last_access_time
            heapq.heapify(nodes)
            # the root node is not in the lru list
            assert (
                len(nodes) == len(self.cache) + 1
            ), f"len(nodes): {len(nodes)} != len(self.cache) + 1: {len(self.cache) + 1}"

            x_lru = self._get_lru()
            while len(nodes):
                x = heapq.heappop(nodes)
                if x == tree_cache.root_node:
                    # root node is not in the lru list
                    continue
                assert (
                    x == x_lru
                ), f"Incorrect LRU list, {self.is_swa_list=}, x: {x.id=} != x_lru: {x_lru.id=}"
                if exempt_node_ids is None or x_lru.id not in exempt_node_ids:
                    assert (
                        x_lru.full_lock_ref == 0
                    ), f"x_lru should not be locked when idle, {x_lru.full_lock_ref=}, {x_lru.swa_uuid=}, {x_lru.id=}"
                    assert (
                        x_lru.swa_lock_ref == 0
                    ), f"x_lru should not be locked when idle, {x_lru.swa_lock_ref=}, {x_lru.swa_uuid=}, {x_lru.id=}"
                x_lru = getattr(x, self.prv)

            if self.is_swa_list:
                evictable_size = tree_cache.swa_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()
            else:
                evictable_size = tree_cache.full_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()

            assert (
                evictable_size == lru_list_evictable_size
            ), f"{self.is_swa_list=}, total nodes: {total_nodes}, total lru plus 1: {total_lru_plus_1}, evictable size: {evictable_size} != lru list evictable size: {lru_list_evictable_size}"
        except Exception as e:
            msg = f"SWA Radix tree sanity check failed, ping @hanming-lu: {e}"
            logger.error(msg)
            raise Exception(msg)


class SWARadixCache(KVCacheEventMixin, BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        assert isinstance(params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable = params.disable
        self.is_eagle = params.is_eagle
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.kv_event_queue = []

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()

        self.sliding_window_size = params.sliding_window_size
        self.reset()

    ##### Public API #####

    def supports_swa(self) -> bool:
        assert (
            self.sliding_window_size is not None
        ), "sliding_window_size must be set for SWARadixCache"
        return True

    def _swa_slots_in_value(self, value: torch.Tensor) -> int:
        """SWA-pool slots referenced by full-layer indices in *value*.

        Every non-tombstone node is *homogeneously* SWA-mapped: normal prefill
        maps the whole node, and FlexKV load-back (``alloc_extend_swa_tail``)
        splits its tail-only mapping at the SWA boundary in ``_insert_helper``
        so the unmapped head becomes a separate ``swa_tombstone`` node. Hence
        the SWA slot count is simply ``len(value)``. Reading the live
        full-to-SWA mapping here would be unsafe: ``dec_swa_lock_only`` zeroes
        it mid-lock, making ``inc_lock_ref``/``dec_lock_ref`` disagree.

        Tombstone nodes never reach here (their SWA was already freed and the
        callers take an explicit 0 / skip branch for them).
        """
        return len(value)

    def _swa_evictable_adjust(self, delta: int, tag: str, **ctx) -> None:
        self.swa_evictable_size_ += delta

    def _tagged_free(self, indices: torch.Tensor, tag: str, **ctx) -> None:
        self.token_to_kv_pool_allocator.free(indices)

    def _tagged_free_swa(self, indices: torch.Tensor, tag: str, **ctx) -> int:
        return self.token_to_kv_pool_allocator.free_swa(indices)

    def _mapped_swa_tail_len(self, value: torch.Tensor) -> int:
        """Length of the contiguous SWA-mapped *suffix* of ``value``.

        ``alloc_extend_swa_tail`` maps only a trailing window and ``free_swa``
        zeroes contiguous regions, so the live full->swa mapping of any value is
        always a suffix: ``[0, gap)`` unmapped, ``[gap, end)`` mapped. We read it
        exactly ONCE here, at insert/revive time, to decide where to split the
        tombstone head from the active tail. After the split every non-tombstone
        node is homogeneous, so all later accounting uses the stable
        ``len(value)`` (see ``_swa_slots_in_value``) and never re-reads the
        mutating mapping.

        Non-hybrid allocators (no SWA pool) are always fully mapped -> ``len``.
        """
        allocator = self.token_to_kv_pool_allocator
        mapping = getattr(allocator, "full_to_swa_index_mapping", None)
        if mapping is None or len(value) == 0:
            return len(value)
        mapped = mapping[value] > 0  # bool tensor, True where SWA-mapped
        total = int(mapped.sum().item())
        if total == 0 or total == len(value):
            return total  # fully unmapped / fully mapped fast path
        # Mapped region must be a contiguous suffix; count trailing True.
        rev = mapped.flip(0)
        first_unmapped = (~rev).nonzero()
        tail = (
            int(first_unmapped[0].item())
            if first_unmapped.numel()
            else len(value)
        )
        assert tail == total, (
            f"[swa] non-suffix SWA mapping in revive: {tail=} {total=} "
            f"{len(value)=}"
        )
        return tail

    def _revive_tombstone_tail(
        self, node: TreeNode, new_value: torch.Tensor
    ) -> None:
        """Revive only the SWA-mapped suffix of a tombstone ``node``.

        Caller has already freed the node's stale full slots; ``new_value`` is
        the fresh full KV for this region. We activate (un-tombstone) only the
        contiguous SWA-mapped tail and keep the unmapped head as a
        ``swa_tombstone`` node, so every non-tombstone node stays homogeneous
        and the ``len``-based / ``free_swa``-based accounting cannot diverge.
        """
        node.value = new_value.clone()
        mapped_tail = self._mapped_swa_tail_len(node.value)

        if mapped_tail == 0:
            # Nothing mapped: refreshed full KV only, stay tombstone, no SWA
            # accounting touched.
            return

        if mapped_tail < len(node.value):
            # Split at the (page-aligned) mapping boundary: _split_node makes the
            # unmapped prefix a new head node (inherits swa_tombstone=True) and
            # keeps `node` as the mapped suffix.
            split_at = len(node.value) - mapped_tail
            self._split_node(node.key, node, split_at)

        node.swa_tombstone = False
        self.swa_lru_list.insert_mru(node)
        # node.value is now the homogeneous mapped tail, so len == mapped_tail.
        self._swa_evictable_adjust(
            len(node.value),
            "revive_tombstone_tail",
            node_id=getattr(node, "id", None),
        )

    def reset(self) -> None:
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.hash_value = []
        self.root_node.full_lock_ref = 1
        self.root_node.swa_lock_ref = 1
        self.full_evictable_size_ = 0
        self.swa_evictable_size_ = 0
        self.full_protected_size_ = 0
        self.swa_protected_size_ = 0
        # LRU lists are used to maintain the order of eviction of the nodes in the tree
        self.full_lru_list = LRUList(is_swa_list=False)
        self.swa_lru_list = LRUList(is_swa_list=True)
        self._record_all_cleared_event()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the matching prefix from the radix tree.
        Args:
            params: MatchPrefixParams containing key.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """

        key = self._match_pre_processor(params)
        if key is None:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                best_match_node=self.root_node,
            )

        value, last_node, best_value_len = self._match_prefix_helper(key)
        return self._match_post_processor(params, value, last_node, best_value_len)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        prev_prefix_len = params.prev_prefix_len
        swa_evicted_seqlen = params.swa_evicted_seqlen

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]
        else:
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        prefix_len = self._insert_helper(
            self.root_node, key, value, prev_prefix_len, swa_evicted_seqlen
        )
        return InsertResult(prefix_len=prefix_len)

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        """Cache request when it finishes."""
        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self._tagged_free(
                kv_indices, "cache_finished_req.disable", rid=getattr(req, "rid", None)
            )
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        page_aligned_len = len(radix_key)
        values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)
        old_prefix_len = req.cache_protected_len

        # Radix Cache takes one ref in memory pool
        # Note: the insert function already frees the overlapped kv_indices
        if is_insert:
            self.insert(
                InsertParams(
                    key=radix_key,
                    value=values,
                    prev_prefix_len=old_prefix_len,
                    swa_evicted_seqlen=req.swa_evicted_seqlen,
                )
            )
        else:
            self._tagged_free(
                kv_indices[old_prefix_len:page_aligned_len],
                "cache_finished_req.no_insert.body",
                rid=getattr(req, "rid", None),
                old_prefix_len=old_prefix_len,
                page_aligned_len=page_aligned_len,
            )

        # free the unaligned tail
        self._tagged_free(
            kv_indices[page_aligned_len:],
            "cache_finished_req.unaligned_tail",
            rid=getattr(req, "rid", None),
            page_aligned_len=page_aligned_len,
            kv_committed_len=kv_committed_len,
        )

        # Remove req slot release the cache lock
        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock),
            skip_swa=req.swa_prefix_lock_released,
        )
        req.swa_prefix_lock_released = False

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        """Cache request when it is unfinished."""
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : req.fill_len
            ]

            # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
            req.prefix_indices = kv_indices
            return

        token_ids = req.get_fill_ids()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)
        old_prefix_len = req.cache_protected_len

        # Radix Cache takes one ref in memory pool
        # Note: the insert function already frees the overlapped kv_indices
        #
        # Pass swa_evicted_seqlen so _insert_helper splits this prefix at the
        # SWA eviction frontier: `maybe_evict_swa`/`_evict_swa` already called
        # `free_swa` on req_to_token[*, :swa_evicted_seqlen] for tokens that
        # slid out of the window, zeroing their full->swa mapping. Without this
        # boundary the head [old_prefix_len, swa_evicted_seqlen) would be
        # inserted as a NON-tombstone node whose SWA mapping is already gone,
        # making it heterogeneous (len(value) != mapped slots) and breaking the
        # `_swa_slots_in_value == len(value)` accounting invariant. This mirrors
        # `cache_finished_req`, which already passes swa_evicted_seqlen.
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                prev_prefix_len=old_prefix_len,
                swa_evicted_seqlen=req.swa_evicted_seqlen,
            )
        )
        new_prefix_len = result.prefix_len

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )

        assert old_prefix_len <= len(new_indices), f"{old_prefix_len=}, {new_indices=}"
        assert new_prefix_len <= len(new_indices), f"{new_prefix_len=}, {new_indices=}"
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(old_prefix_len, len(new_indices))),
            new_indices[old_prefix_len:],
        )

        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock),
            skip_swa=req.swa_prefix_lock_released,
        )
        req.swa_prefix_lock_released = False
        result = self.inc_lock_ref(new_last_node)
        swa_uuid_for_lock = result.swa_uuid_for_lock

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node
        req.swa_uuid_for_lock = swa_uuid_for_lock

    def pretty_print(self) -> None:
        self._print_helper(self.root_node, 0)
        total_size, total_swa_size = self._total_size_helper()
        print(f"#full_tokens: {total_size}, #swa_tokens: {total_swa_size}")

    def total_size(self) -> Tuple[int, int]:
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        full_num_tokens = params.num_tokens
        swa_num_tokens = params.swa_num_tokens
        full_num_evicted = 0
        swa_num_evicted = 0
        if full_num_tokens > 0:
            # get the least recently used leaf node that is not locked
            x = self.full_lru_list.get_leaf_lru_no_lock()

            while full_num_evicted < full_num_tokens and self.full_lru_list.in_list(x):
                assert (
                    x != self.root_node
                ), f"root node should not exist in full lru list, {x.id=}"
                assert x.full_lock_ref == 0, f"node is in use, {x.id=}"

                # 1. free node kv indices, evict full and swa tokens
                self._record_remove_event(x)
                swa_slots = (
                    self._swa_slots_in_value(x.value) if not x.swa_tombstone else 0
                )
                self._tagged_free(
                    x.value,
                    "evict.full_lru.leaf",
                    node_id=getattr(x, "id", None),
                    is_tombstone=x.swa_tombstone,
                )
                full_num_evicted += len(x.value)
                # Tombstoned leaves had their SWA freed earlier in `dec_swa_lock_only`
                if not x.swa_tombstone:
                    swa_num_evicted += swa_slots

                # 2. get the next leaf, update the lru lists
                x_next = self.full_lru_list.get_prev_leaf_no_lock(x)
                self.full_lru_list.remove_node(x)
                if not x.swa_tombstone:
                    self.swa_lru_list.remove_node(x)

                # 3. delete the leaf node
                self._delete_leaf(x)

                # 4. Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone
                x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
                full_num_evicted += leaf_full_num_evicted

                # 5. if parent has no more children, it is a leaf. It is possible that this node is lru, so
                # we need to get the first leaf node in the lru list
                if len(x.parent.children) == 0:
                    x_next = self.full_lru_list.get_leaf_lru_no_lock()

                x = x_next

        if swa_num_evicted < swa_num_tokens:
            # get the least recently used node that is not locked, doesn't have to be a leaf
            x = self.swa_lru_list.get_lru_no_lock()

            # evict lru leaf nodes until swa_num_tokens is reached
            while swa_num_evicted < swa_num_tokens and (self.swa_lru_list.in_list(x)):
                assert not x.swa_tombstone, f"duplicate swa tombstone node, {x.id=}"
                assert x != self.root_node, f"root node is not evictable, {x.id=}"
                assert x.swa_lock_ref == 0, f"node is in use by swa kv indices, {x.id=}"

                if len(x.children) > 0:
                    # 1. an internal node, free swa tokens.
                    swa_freed = self._tagged_free_swa(
                        x.value,
                        "evict.swa_lru.internal->tombstone",
                        node_id=getattr(x, "id", None),
                    )
                    swa_num_evicted += swa_freed

                    # 2. get the next node, update the lru lists
                    x_next = self.swa_lru_list.get_prev_no_lock(x)
                    self.swa_lru_list.remove_node(x)

                    # 3. tombstone the node
                    self._tombstone_internal_node(x, swa_slots=swa_freed)
                elif x.full_lock_ref > 0:
                    # Leaf still holds a full-side lock (can happen when the
                    # SWA leaf-lock early-release optimization revived a
                    # tombstoned leaf. Treat it like an internal tombstone.
                    swa_freed = self._tagged_free_swa(
                        x.value,
                        "evict.swa_lru.leaf_full_locked",
                        node_id=getattr(x, "id", None),
                    )
                    swa_num_evicted += swa_freed

                    x_next = self.swa_lru_list.get_prev_no_lock(x)
                    self.swa_lru_list.remove_node(x)

                    self._swa_evictable_adjust(
                        -swa_freed,
                        "evict:leaf_full_locked_tombstone",
                        node_id=getattr(x, "id", None),
                        swa_freed=swa_freed,
                    )
                    x.swa_tombstone = True
                else:
                    assert (
                        x.full_lock_ref == 0
                    ), f"leaf node with full lock must also have swa lock, {x.id=}"
                    # 1. a leaf node, free full and swa tokens
                    self._record_remove_event(x)
                    swa_slots = self._swa_slots_in_value(x.value)
                    self._tagged_free(
                        x.value,
                        "evict.swa_lru.leaf",
                        node_id=getattr(x, "id", None),
                    )
                    full_num_evicted += len(x.value)
                    swa_num_evicted += swa_slots

                    # 2. get the next node, update the lru lists
                    x_next = self.swa_lru_list.get_prev_no_lock(x)
                    self.full_lru_list.remove_node(x)
                    self.swa_lru_list.remove_node(x)

                    # 3. delete the leaf node
                    self._delete_leaf(x)

                    # 4. Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone
                    self._iteratively_delete_tombstone_leaf(x)

                x = x_next

        self.update_eviction_metrics(full_num_evicted + swa_num_evicted, start_time)
        return EvictResult(
            num_tokens_evicted=full_num_evicted, swa_num_tokens_evicted=swa_num_evicted
        )

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        """
        Increment the lock reference count for the node. Returns the swa_uuid_for_lock, which needs
        to be passed to dec_lock_ref.
        It locks the full_lock_ref for nodes between the [last node, root), exclusive.
        It locks the swa_lock_ref for nodes between the [last node, swa_uuid_for_lock], inclusive.
        """
        if self.disable:
            return IncLockRefResult()

        swa_lock_size = 0
        swa_uuid_for_lock = None
        while node != self.root_node:
            # lock full from node to root
            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 0:
                self.full_evictable_size_ -= len(node.value)
                self.full_protected_size_ += len(node.value)
            node.full_lock_ref += 1

            # lock swa if we have not reached the sliding window size.
            # When we reach the sliding window size, we will set the swa_uuid_for_lock.
            # caller needs to pass the swa_uuid_for_lock to dec_lock_ref
            if swa_lock_size < self.sliding_window_size:
                assert (
                    not node.swa_tombstone
                ), f"inc_lock_swa on swa_tombstone node, {node.id=}"
                swa_slots = self._swa_slots_in_value(node.value)
                if node.swa_lock_ref == 0:
                    self._swa_evictable_adjust(
                        -swa_slots,
                        "inc_lock_ref",
                        node_id=getattr(node, "id", None),
                    )
                    self.swa_protected_size_ += swa_slots
                node.swa_lock_ref += 1
                swa_lock_size += swa_slots
                if swa_lock_size >= self.sliding_window_size:
                    if node.swa_uuid is None:
                        node.swa_uuid = gen_swa_uuid()
                    swa_uuid_for_lock = node.swa_uuid
            node = node.parent

        return IncLockRefResult(swa_uuid_for_lock=swa_uuid_for_lock)

    def dec_lock_ref(
        self,
        node: TreeNode,
        params: Optional[DecLockRefParams] = None,
        skip_swa: bool = False,
    ) -> DecLockRefResult:
        """
        Decrement the lock reference count for the node.
        It unlocks the full_lock_ref for nodes between the [last node, root), exclusive.
        It unlocks the swa_lock_ref for nodes between the [last node, swa_uuid_for_lock], inclusive.
        If swa_uuid_for_lock is None, it unlocks to the root, exclusive.

        If skip_swa is True, only the full_lock_ref is decremented; the SWA lock is
        assumed to have been released already (e.g. via `dec_swa_lock_only`).
        """
        swa_uuid_for_lock = params.swa_uuid_for_lock if params is not None else None

        if self.disable:
            return DecLockRefResult()

        dec_lock_swa = not skip_swa
        while node != self.root_node:
            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 1:
                self.full_evictable_size_ += len(node.value)
                self.full_protected_size_ -= len(node.value)
            node.full_lock_ref -= 1

            if dec_lock_swa:
                assert (
                    not node.swa_tombstone
                ), f"dec_lock_ref on swa_tombstone node, {node.id=}"
                assert (
                    node.swa_lock_ref > 0
                ), f"dec_lock_ref on node with {node.swa_lock_ref=}, {node.id=}"

                if node.swa_lock_ref == 1:
                    swa_slots = self._swa_slots_in_value(node.value)
                    self._swa_evictable_adjust(
                        swa_slots,
                        "dec_lock_ref",
                        node_id=getattr(node, "id", None),
                    )
                    self.swa_protected_size_ -= swa_slots
                node.swa_lock_ref -= 1
                if swa_uuid_for_lock and node.swa_uuid == swa_uuid_for_lock:
                    dec_lock_swa = False

            node = node.parent

        return DecLockRefResult()

    def dec_swa_lock_only(
        self, node: TreeNode, swa_uuid_for_lock: Optional[int] = None
    ):
        """
        Decrement only the swa_lock_ref (and swa_protected_size_) along the chain
        [node, swa_uuid_for_lock], inclusive. The full_lock_ref is left untouched
        so the caller's full-cache protection is preserved.

        Used to early-release the SWA portion of a request's tree lock once the
        request's decode position has advanced past the sliding window, so the
        protected window can be reclaimed.

        For internal nodes, the standard protected -> evictable transition is
        applied (node stays in swa_lru_list and may be evicted by SWA LRU later).
        For leaf nodes, since `swa_lru_list` cannot contain a leaf with
        `full_lock_ref > 0` (SWA-eviction would also delete the still-referenced
        leaf), we instead free the SWA pool slots immediately and mark the leaf
        as `swa_tombstone=True`. The full kv stays alive until the full-side
        lock drops; future prefix-matches stop before this tombstoned leaf.

        Caller must ensure this is invoked at most once per (node, swa_uuid_for_lock)
        pair (track via e.g. `Req.swa_prefix_lock_released`). When the request
        finally releases its full lock via `dec_lock_ref`, pass `skip_swa=True`
        to avoid touching SWA state again.
        """
        if self.disable:
            return

        while node != self.root_node:
            assert (
                not node.swa_tombstone
            ), f"dec_swa_lock_only on swa_tombstone node, {node.id=}"
            assert (
                node.swa_lock_ref > 0
            ), f"dec_swa_lock_only on node with {node.swa_lock_ref=}, {node.id=}"

            if node.swa_lock_ref == 1:
                swa_slots = self._swa_slots_in_value(node.value)
                self.swa_protected_size_ -= swa_slots
                if len(node.children) == 0:
                    # Leaf: free SWA pool slots and tombstone, and remove from
                    # swa_lru_list so SWA-eviction won't pick this tombstoned
                    # leaf (which still holds full_lock_ref > 0). The full kv
                    # stays alive until the request releases its full lock.
                    self._tagged_free_swa(
                        node.value,
                        "dec_swa_lock_only.leaf->tombstone",
                        node_id=getattr(node, "id", None),
                    )
                    self.swa_lru_list.remove_node(node)
                    node.swa_tombstone = True
                else:
                    # Internal: standard protected -> evictable.
                    self._swa_evictable_adjust(
                        swa_slots,
                        "dec_swa_lock_only:internal",
                        node_id=getattr(node, "id", None),
                    )
            node.swa_lock_ref -= 1

            if swa_uuid_for_lock and node.swa_uuid == swa_uuid_for_lock:
                break
            node = node.parent

    def sanity_check(self, exempt_node_ids: Optional[Set[int]] = None):
        self.full_lru_list.sanity_check(self, exempt_node_ids)
        self.swa_lru_list.sanity_check(self, exempt_node_ids)

    def evictable_size(self) -> int:
        # Legacy single-pool callers use full-attention evictable only.
        # Hybrid scheduling should call full_evictable_size() / swa_evictable_size().
        return self.full_evictable_size_

    def full_evictable_size(self) -> int:
        return self.full_evictable_size_

    def swa_evictable_size(self) -> int:
        return self.swa_evictable_size_

    def protected_size(self) -> int:
        return self.full_protected_size_

    def full_protected_size(self) -> int:
        # protected size refers to the size of the full cache that is locked
        return self.full_protected_size_

    def swa_protected_size(self) -> int:
        # protected size refers to the size of the swa cache that is locked
        return self.swa_protected_size_

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    def available_and_evictable_str(self) -> str:
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        full_evictable_size = self.full_evictable_size()
        swa_evictable_size = self.swa_evictable_size()
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
            f"Available swa tokens: {swa_available_size + swa_evictable_size} ({swa_available_size=} + {swa_evictable_size=})\n"
            f"Full LRU list evictable size: {self.full_lru_list.sanity_check_evictable_size()}\n"
            f"SWA LRU list evictable size: {self.swa_lru_list.sanity_check_evictable_size()}\n"
        )

    ##### Internal Helper Functions #####

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> Tuple[List[torch.Tensor], TreeNode, int]:
        """
        SWA prefix matching helper. It factors in the sliding window size such that
        the matched node is guaranteed to either 1. connected to root without swa tombstone,
        or 2. the number of matching tokens from the matched node to the last swa tombstone
        node is greater than or equal to the sliding window size.
        """
        node = self.root_node
        child_key = key.child_key(self.page_size)

        value = []
        # for path connected to root without tombstone, always match, so set to inf
        match_len_since_tombstone = float("inf")
        best_value_len = 0
        best_last_node = node
        enable_compact = envs.SGLANG_OPT_SWA_RADIX_CACHE_COMPACT.get()
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]

            if enable_compact:
                self._compact_single_child_chain(child)

            if child.swa_tombstone:
                # update best_value_len and best_last_node if needed
                if match_len_since_tombstone >= self.sliding_window_size:
                    best_value_len = len(value)
                    best_last_node = node
                # reset match_len_since_tombstone if we hit a tombstone node
                match_len_since_tombstone = 0

            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                if not new_node.swa_tombstone:
                    match_len_since_tombstone += len(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                if not child.swa_tombstone:
                    match_len_since_tombstone += len(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)

        # handle best_value_len and best_last_node, for the case that last node is fully matched
        if match_len_since_tombstone >= self.sliding_window_size:
            best_value_len = len(value)
            best_last_node = node

        return value, best_last_node, best_value_len

    def _match_pre_processor(self, params: MatchPrefixParams) -> Optional[RadixKey]:
        """Preprocess the key before matching."""
        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if self.disable or len(key) == 0:
            return None
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return None
        return key

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: List[torch.Tensor],
        last_node: TreeNode,
        best_value_len: int,
    ) -> MatchResult:
        """Post-process the matched result."""
        node_update = last_node
        # update time for matched nodes, and make nodes closer to root to be least recently used
        # this allows swa to evict nodes closer to root first
        self.full_lru_list.reset_node_and_parents_mru(node_update, self.root_node)
        self.swa_lru_list.reset_node_and_parents_mru(node_update, self.root_node)

        # This last_access_time is for sanity check, can be deleted after validation in production
        cur_time = get_last_access_time()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= (
                0.00001  # assuming less than 100000 nodes in a branch of the tree
            )
            node_update = node_update.parent

        value = value[:best_value_len]
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
        )

    def _compact_single_child_chain(self, node: TreeNode) -> None:
        # FIXME(ispobock): drifts retract pool accounting (commit 6348cb506);
        # also overwrites active swa_uuid when window > page_size. Off by
        # default via SGLANG_OPT_SWA_RADIX_CACHE_COMPACT.
        while len(node.children) == 1:
            child = next(iter(node.children.values()))
            if len(child.children) == 0:
                break
            sum_gc_full_lock_ref = sum(
                gc.full_lock_ref for gc in child.children.values()
            )
            if child.full_lock_ref > sum_gc_full_lock_ref:
                break
            if (
                child.swa_tombstone != node.swa_tombstone
                or child.full_lock_ref != node.full_lock_ref
                or child.swa_lock_ref != node.swa_lock_ref
            ):
                break

            # Preserve is_bigram: main #23106 made bigram an O(1) flag on RadixKey;
            # the constructor defaults to False, so concat without explicit flag
            # silently demotes EAGLE/MTP bigram keys → match() returns 0 →
            # _split_node assert.
            node.key = RadixKey(
                node.key.token_ids + child.key.token_ids,
                node.key.extra_key,
                is_bigram=node.key.is_bigram,
            )
            node.value = torch.cat([node.value, child.value])
            node.children = child.children
            for grandchild in node.children.values():
                grandchild.parent = node

            if child.swa_uuid is not None:
                node.swa_uuid = child.swa_uuid

            if node.hash_value is not None and child.hash_value is not None:
                node.hash_value = list(node.hash_value) + list(child.hash_value)
            else:
                node.hash_value = None

            self.full_lru_list.remove_node(child)
            if not child.swa_tombstone:
                self.swa_lru_list.remove_node(child)

    def _maybe_split_leaf_for_swa_lock(self, leaf: TreeNode) -> TreeNode:
        """``inc_lock_ref`` protects ``len(leaf.value)`` SWA tokens for the
        leaf even though SWA only actually needs the last
        ``sliding_window_size`` tokens. With chunked prefill, leaves can be
        thousands of tokens long, which inflates ``swa_protected_size_`` by
        ~``chunked_prefill_size / sliding_window_size`` and causes premature
        SWA pool exhaustion / retract thrashing.
        """
        if (
            leaf is self.root_node
            or leaf.swa_lock_ref > 0
            or leaf.swa_tombstone
            or len(leaf.value) == 0
        ):
            return leaf

        # Smallest page-aligned size that still covers the sliding window.
        tail_size = (
            (self.sliding_window_size + self.page_size - 1)
            // self.page_size
            * self.page_size
        )
        if len(leaf.value) <= tail_size:
            return leaf

        split_at = len(leaf.value) - tail_size

        if split_at <= 0 or split_at >= len(leaf.value):
            return leaf
        if self.page_size > 1 and (
            split_at % self.page_size != 0 or len(leaf.value) % self.page_size != 0
        ):
            return leaf

        self._split_node(leaf.key, leaf, split_at)
        return leaf

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.swa_tombstone = child.swa_tombstone
        new_node.full_lock_ref = child.full_lock_ref
        new_node.swa_lock_ref = child.swa_lock_ref
        new_node.key = child.key[:split_len]
        assert len(new_node.key) > 0, f"new_node.key should not be empty"
        new_node.value = child.value[:split_len].clone()
        # parent inherits the swa_uuid from child for swa lock ref
        new_node.swa_uuid = child.swa_uuid
        child.swa_uuid = None
        # child time should be later than parent's time for swa tombstone
        child.last_access_time = get_last_access_time()

        # remove the child from the lru lists because it is being split
        self.full_lru_list.remove_node(child)
        if not new_node.swa_tombstone:
            self.swa_lru_list.remove_node(child)
        child.parent = new_node
        child.key = child.key[split_len:]
        assert len(child.key) > 0, f"child.key should not be empty"
        child.value = child.value[split_len:].clone()
        new_node.parent.children[key.child_key(self.page_size)] = new_node
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        # insert the new node and child into the lru lists, insert
        # parent first so that parent is after child in the lru list
        self.full_lru_list.insert_mru(new_node)
        self.full_lru_list.insert_mru(child)
        if not new_node.swa_tombstone:
            self.swa_lru_list.insert_mru(new_node)
            self.swa_lru_list.insert_mru(child)
        return new_node

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        update_kv_after_len: int,
        swa_evicted_seqlen: int = 0,
    ) -> int:
        # Update the last access time from root to leaf, so that
        # swa will tombstone the node closer to root first
        node.last_access_time = get_last_access_time()
        if node != self.root_node:
            self.full_lru_list.reset_node_mru(node)
            if not node.swa_tombstone:
                self.swa_lru_list.reset_node_mru(node)
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()
            self.full_lru_list.reset_node_mru(node)
            if not node.swa_tombstone:
                self.swa_lru_list.reset_node_mru(node)
            prefix_len = node.key.match(key, page_size=self.page_size)

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            # if tombstone after update_kv_after_len, update node.value to be the input value.
            # This is needed because it is possible that the last sliding window size tokens
            # contains tombstone. If this is the case and we don't update the kv value, then
            # the prefill prefix matching will stuck.
            if update_kv_after_len < total_prefix_length + prefix_len:
                # For page_size > 1 and chunked prefill case, update_kv_after_len may be not page-aligned due to a trailing partial page
                # (kept in the request but not inserted into the radix tree) appended to prefix_indices.
                if node.swa_tombstone:
                    assert (
                        node.swa_lock_ref == 0
                    ), f"tombstone swa_lock_ref should always be 0, {node.full_lock_ref=}, {node.swa_lock_ref=}, {node.id=}"
                    assert (
                        swa_evicted_seqlen % self.page_size == 0
                    ), f"swa_evicted_seqlen must be page aligned, {swa_evicted_seqlen=}, {self.page_size=}"
                    if swa_evicted_seqlen <= total_prefix_length:
                        # Branch 1: caller believes the whole region is past the
                        # eviction boundary. Under tail-only FlexKV mapping that
                        # belief can be wrong (the value may have only a mapped
                        # suffix), so revive by ACTUAL mapped tail, not by len.
                        # Free the node's stale full slots first.
                        self._tagged_free(
                            node.value[:prefix_len],
                            "insert_helper.tombstone.branch1.free_node_stale",
                            node_id=getattr(node, "id", None),
                            prefix_len=prefix_len,
                            swa_evicted_seqlen=swa_evicted_seqlen,
                            total_prefix_length=total_prefix_length,
                        )
                        self._revive_tombstone_tail(node, value[:prefix_len])
                    elif swa_evicted_seqlen < total_prefix_length + prefix_len:
                        # Branch 2: caller-known boundary lands inside this
                        # region. Free the head's stale full, split at the
                        # boundary, then revive the tail by its actual mapping
                        # (a second split if the mapped suffix is shorter still).
                        start_update_idx = swa_evicted_seqlen - total_prefix_length
                        self._tagged_free(
                            node.value[start_update_idx:prefix_len],
                            "insert_helper.tombstone.branch2.free_node_tail_stale",
                            node_id=getattr(node, "id", None),
                            start_update_idx=start_update_idx,
                            prefix_len=prefix_len,
                        )
                        self._split_node(node.key, node, start_update_idx)
                        # The old (head) node stays swa tombstone; `node` is now
                        # the suffix to revive.
                        self._tagged_free(
                            value[:start_update_idx],
                            "insert_helper.tombstone.branch2.free_input_head",
                            node_id=getattr(node, "id", None),
                            start_update_idx=start_update_idx,
                        )
                        self._revive_tombstone_tail(
                            node, value[start_update_idx:prefix_len]
                        )
                    else:
                        # Branch 3: all swa tokens of value[:prefix_len] are evicted, so we don't need to update the node.
                        self._tagged_free(
                            value[:prefix_len],
                            "insert_helper.tombstone.branch3.free_input",
                            node_id=getattr(node, "id", None),
                            prefix_len=prefix_len,
                        )
                else:
                    # The node is not tombstone, so we don't need to update the node.
                    self._tagged_free(
                        value[:prefix_len],
                        "insert_helper.non_tombstone.free_input_dup",
                        node_id=getattr(node, "id", None),
                        prefix_len=prefix_len,
                    )

            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            # Layout: |--- total_prefix_length ---|--- len(key) ---|
            #         ^                           ^                ^
            #         0              total_prefix_length     total_length
            #
            # Cases based on swa_evicted_seqlen position:
            # 1. swa_evicted_seqlen <= total_prefix_length:
            #    Already handled in the while loop above. All of len(key) is non-tombstone.
            # 2. total_prefix_length < swa_evicted_seqlen < total_length:
            #    Split: [total_prefix_length, swa_evicted_seqlen) as tombstone,
            #           [swa_evicted_seqlen, total_length) as non-tombstone.
            # 3. swa_evicted_seqlen == total_length:
            #    All remaining tokens are evicted. Free value and return without
            #    creating a node (leaf nodes must not be tombstone).
            #    Note: the -page_size fix in _evict_swa prevents this case from
            #    occurring in normal operation. This check is a defensive guard
            #    against unexpected eviction states from other code paths.
            if swa_evicted_seqlen == total_prefix_length + len(key):
                self._tagged_free(
                    value,
                    "insert_helper.tail_all_evicted",
                    swa_evicted_seqlen=swa_evicted_seqlen,
                    total_prefix_length=total_prefix_length,
                    key_len=len(key),
                )
                return total_prefix_length

            if (
                swa_evicted_seqlen > total_prefix_length
                and swa_evicted_seqlen < total_prefix_length + len(key)
            ):
                swa_tombstone_len = swa_evicted_seqlen - total_prefix_length
                node = self._add_new_node(
                    node,
                    key[:swa_tombstone_len],
                    value[:swa_tombstone_len],
                    swa_tombstone=True,
                )
                key = key[swa_tombstone_len:]
                value = value[swa_tombstone_len:]

            new_leaf = self._add_new_node(node, key, value, swa_tombstone=False)

            if envs.SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT.get():
                # Cap the leaf at one (page-aligned) sliding window so a future
                # inc_lock_ref only protects `sliding_window_size` tokens of SWA pool.
                self._maybe_split_leaf_for_swa_lock(new_leaf)

        return total_prefix_length

    def _add_new_node(
        self,
        parent: TreeNode,
        key: RadixKey,
        value: torch.Tensor,
        swa_tombstone: bool = False,
    ) -> TreeNode:
        assert len(key) > 0, f"key should not be empty"
        new_node = TreeNode()
        new_node.parent = parent
        new_node.key = key
        new_node.value = value.clone()
        new_node.swa_tombstone = swa_tombstone
        parent.children[key.child_key(self.page_size)] = new_node
        self.full_lru_list.insert_mru(new_node)
        self.full_evictable_size_ += len(value)
        if not swa_tombstone:
            self.swa_lru_list.insert_mru(new_node)
            swa_slots = self._swa_slots_in_value(value)
            self._swa_evictable_adjust(swa_slots, "add_new_node")
        self._record_store_event(new_node)
        return new_node

    def _iteratively_delete_tombstone_leaf(
        self, node: TreeNode
    ) -> Tuple[TreeNode, int]:
        full_num_evicted = 0
        while node.parent.swa_tombstone and len(node.parent.children) == 0:
            # root node is not evictable
            if node.parent == self.root_node:
                break
            # if locked, means node is in use, skip
            if node.parent.full_lock_ref > 0:
                break
            assert (
                node.parent.swa_lock_ref == 0
            ), f"tombstone swa_lock_ref should always be 0, {node.parent.full_lock_ref=}, {node.parent.swa_lock_ref=}, {node.parent.id=}"
            # delete tombstone node evicts full tokens
            self._record_remove_event(node.parent)
            self._tagged_free(
                node.parent.value,
                "iter_delete_tombstone.parent",
                parent_id=getattr(node.parent, "id", None),
                child_id=getattr(node, "id", None),
                parent_value_len=len(node.parent.value),
            )
            full_num_evicted += len(node.parent.value)
            self.full_lru_list.remove_node(node.parent)
            self._delete_tombstone_leaf(node.parent)
            node = node.parent

        return node, full_num_evicted

    def _delete_leaf(self, node: TreeNode) -> None:
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"
        self.full_evictable_size_ -= len(node.key)
        # Tombstoned leaves were never (re-)added to swa_lru_list and were
        # already removed from swa_evictable_size_ when they were tombstoned.
        if not node.swa_tombstone:
            self._swa_evictable_adjust(
                -self._swa_slots_in_value(node.value),
                "delete_leaf",
                node_id=getattr(node, "id", None),
            )

    def _tombstone_internal_node(
        self, node: TreeNode, *, swa_slots: int | None = None
    ) -> None:
        assert len(node.children) != 0, f"Cannot tombstone a leaf node, {node.id=}"
        node.swa_tombstone = True
        freed = (
            swa_slots
            if swa_slots is not None
            else self._swa_slots_in_value(node.value)
        )
        self._swa_evictable_adjust(
            -freed,
            "tombstone_internal_node",
            node_id=getattr(node, "id", None),
            swa_freed=freed,
        )

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert (
            node.swa_tombstone
        ), f"Deleting a unexpected non-tombstone leaf node, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)

    def _collect_nontombstone_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if not cur_node.swa_tombstone:
                ret_list.append(cur_node)
            stack.extend(cur_node.children.values())

        return ret_list

    def _collect_all_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            ret_list.append(cur_node)
            stack.extend(cur_node.children.values())
        return ret_list

    def _print_helper(self, node: TreeNode, indent: int) -> None:
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                current_node.id,
                len(current_node.key),
                f"fr={current_node.full_lock_ref}",
                f"sr={current_node.swa_lock_ref}",
                f"fll={self.full_lru_list.in_list(current_node)}",
                f"sll={self.swa_lru_list.in_list(current_node)}",
                f"ts={current_node.swa_tombstone}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == child.key.child_key(
                    self.page_size
                ), f"{key=}, {child.key.child_key(self.page_size)=}"

    def _total_size_helper(self) -> Tuple[int, int]:
        total_size = 0
        total_swa_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            if not current_node.swa_tombstone:
                total_swa_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size, total_swa_size
