"""Unit tests for ExtendedRadixCache.init_load_back over SWARadixCache.

These tests run CPU-only and do NOT boot a server. They mock
ReqToTokenPool / SWATokenToKVPoolAllocator with the minimal surface
init_load_back uses, so we can exercise:

  1. The TreeNode class mismatch bug (generic radix_cache.TreeNode
     vs swa_radix_cache.TreeNode with full_lock_ref/swa_lock_ref).
  2. Paged allocation path (page_size=256) without hitting the
     ``assert self.page_size == 1`` in SWATokenToKVPoolAllocator.alloc.
  3. The (new_indices, last_node) tuple contract that schedule_policy
     unpacks.

Run:
    cd /data/git_store2/dpskv4/sglang
    PYTHONPATH=python python3 -m pytest \\
        test/srt/mem_cache/test_extended_radix_cache_swa.py -v
"""
from __future__ import annotations

import os
import sys
import unittest
from array import array
from typing import Optional
from unittest.mock import MagicMock

import torch

# Make the in-repo sglang package importable without installing.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# We import these AFTER tweaking sys.path so dev edits are picked up live.
from sglang.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.extended_radix_cache import ExtendedRadixCache  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache  # noqa: E402


# ----------------------------------------------------------------------------- #
# Minimal mocks                                                                  #
# ----------------------------------------------------------------------------- #

class FakePagedAllocator:
    """Stand-in for SWATokenToKVPoolAllocator that mimics page_size>1 behavior.

    Implements just enough of the interface that ExtendedRadixCache and
    SWARadixCache touch:
      * page_size, device
      * alloc(N): asserts page_size==1, MIRRORS the real bug
      * alloc_extend(prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu,
                     last_loc, extend_num_tokens): hands out a contiguous block
      * alloc_extend_swa_tail(...): tail-only SWA slot allocation (the real
        DSv4 allocator has tiny SWA capacity, so we must use this when window
        is set, NOT alloc_extend).
      * free(indices): mark slots free (returns count)
      * available_size(): unused free slots
    """

    def __init__(
        self,
        total_slots: int = 4096,
        page_size: int = 256,
        # If swa_capacity is set, alloc_extend will refuse N > swa_capacity to
        # mimic the real allocator's full+swa pair where SWA pool is much
        # smaller than the full pool.
        swa_capacity: int = None,
    ):
        self.page_size = page_size
        self.device = torch.device("cpu")
        self._next = 0
        self._total = total_slots
        self.size = total_slots
        self.dtype = torch.int64
        self._swa_capacity = swa_capacity
        # Track the swa_tail path was taken (test hook).
        self.swa_tail_calls = 0
        self.alloc_extend_calls = 0

    def alloc(self, need_size: int):
        assert self.page_size == 1, (
            "FakePagedAllocator: alloc(N) should NOT be called when page_size>1"
        )
        return self._reserve(need_size)

    def alloc_extend(
        self,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
    ):
        self.alloc_extend_calls += 1
        # Mimic the real allocator: if SWA pool is too small, return None.
        if self._swa_capacity is not None and extend_num_tokens > self._swa_capacity:
            return None
        return self._reserve(int(extend_num_tokens))

    def alloc_extend_swa_tail(
        self,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
        swa_tail_len,
    ):
        self.swa_tail_calls += 1
        # Real impl checks both pools; here we just check the SWA cap against
        # the *tail*, not the whole extend.
        if self._swa_capacity is not None and swa_tail_len > self._swa_capacity:
            return None
        return self._reserve(int(extend_num_tokens))

    def _reserve(self, n: int):
        if self._next + n > self._total:
            return None
        out = torch.arange(self._next, self._next + n, dtype=torch.int64)
        self._next += n
        return out

    def free(self, indices):
        # Stub: just count, don't actually reclaim — tests don't run long enough.
        return indices.numel() if hasattr(indices, "numel") else len(indices)

    def available_size(self):
        return self._total - self._next

    def full_available_size(self):
        return self.available_size()

    def swa_available_size(self):
        cap = self._swa_capacity if self._swa_capacity is not None else self._total
        return max(0, cap - min(self._next, cap))

    def backup_state(self):
        return self._next

    def restore_state(self, state):
        self._next = state

    # Some SWA paths peek at allocator.size_used / merge_and_sort_free, etc.
    # Keep them no-op.
    def merge_and_sort_free(self):
        pass


class FakeConnector:
    """Mimics BaseKVConnector surface that init_load_back / match_prefix uses."""

    def __init__(self, swa_window_size: int = 0, layer_done_counter=None):
        self.released = []
        self.start_load_calls = []
        # If a counter is provided, start_load_kv calls update_producer to mimic
        # FlexKV's real counter rotation. This is what the consumer-rotation
        # regression test exercises.
        self.layer_done_counter = layer_done_counter
        # ExtendedRadixCache reads this to decide whether to use the
        # alloc_extend_swa_tail path.
        self._swa_window_size = swa_window_size

    def reset(self): pass
    def release_load_state(self, rid): self.released.append(rid)
    def get_new_hit_length(self, **kw): return 0

    def start_load_kv(self, task_id, ops):
        self.start_load_calls.append((task_id, list(ops)))
        if self.layer_done_counter is not None:
            # Mirror flexkv_connector.start_load_kv: rotates producer slot.
            producer_id = self.layer_done_counter.update_producer()
            self.layer_done_counter.events[producer_id].reset_for_new_transfer()
            self.layer_done_counter.register_task(task_id, producer_id)

    def start_store_kv(self, **kw): pass
    def check_completed_load_tasks(self): return []
    def check_completed_store_tasks(self): return []
    def prefetch(self, *a, **kw): pass
    def check_prefetch_progress(self, rid): return True
    def pop_prefetch_loaded_tokens(self, rid): return 0
    def cancel_prefetch(self, rid): pass


def _make_swa_cache(page_size=256, sliding_window_size=256, total_slots=8192):
    """Build a real SWARadixCache backed by mocked allocator/req-pool."""
    allocator = FakePagedAllocator(total_slots=total_slots, page_size=page_size)

    # Patch SWARadixCache's isinstance check (it asserts the allocator is the
    # production class). Easiest path: monkey-patch the isinstance call by
    # registering FakePagedAllocator as a virtual subclass.
    from sglang.srt.mem_cache import swa_memory_pool as _swa_pool_mod
    _swa_pool_mod.SWATokenToKVPoolAllocator.register(FakePagedAllocator)  # type: ignore[attr-defined]

    req_to_token_pool = MagicMock()
    req_to_token_pool.req_to_token = torch.zeros((128, 1024), dtype=torch.int64)

    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=page_size,
        is_eagle=False,
        sliding_window_size=sliding_window_size,
    )
    swa = SWARadixCache(params)
    return swa, allocator


def _make_req(rid: str, token_ids: list[int], host_hit_length: int,
              prefix_indices: Optional[torch.Tensor] = None,
              last_node=None):
    req = MagicMock()
    req.rid = rid
    req.req_pool_idx = 0
    req.fill_ids = list(token_ids)
    req.origin_input_ids = list(token_ids)
    req.output_ids = []
    req.extra_key = None
    req.host_hit_length = host_hit_length
    req.prefix_indices = prefix_indices if prefix_indices is not None else torch.empty((0,), dtype=torch.int64)
    req.last_node = last_node
    req.cached_tokens_extended_device = 0
    return req


# ----------------------------------------------------------------------------- #
# Tests                                                                          #
# ----------------------------------------------------------------------------- #

class TestExtendedRadixCacheSWAInitLoadBack(unittest.TestCase):
    """Reproduce and verify fixes for the H2D init_load_back crash."""

    def setUp(self):
        self.page_size = 256
        self.window = 256
        self.swa, self.allocator = _make_swa_cache(
            page_size=self.page_size, sliding_window_size=self.window
        )
        self.connector = FakeConnector()
        self.cache = ExtendedRadixCache(
            params=CacheInitParams(
                disable=False,
                req_to_token_pool=self.swa.req_to_token_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=self.page_size,
                is_eagle=False,
                sliding_window_size=self.window,
            ),
            connector=self.connector,
            inner_cache=self.swa,
        )

    def test_paged_allocator_does_not_call_alloc_N(self):
        """Original bug: init_load_back called allocator.alloc(N) which asserts page_size==1.

        Fix should route to alloc_extend on page_size>1 allocators.
        """
        token_ids = list(range(self.page_size * 2))   # 512 tokens, 2 pages
        host_hit_length = self.page_size * 2          # match the full prompt
        req = _make_req(
            rid="rid-paged-1",
            token_ids=token_ids,
            host_hit_length=host_hit_length,
            last_node=self.swa.root_node,
        )
        # Should NOT raise AssertionError("page_size == 1") and NOT raise
        # AttributeError on full_lock_ref.
        new_indices, new_last_node = self.cache.init_load_back(
            InitLoadBackParams(req=req, host_hit_length=host_hit_length,
                               best_match_node=self.swa.root_node)
        )
        self.assertEqual(new_indices.numel(), host_hit_length)
        self.assertIsNotNone(new_last_node)
        self.assertIsNot(new_last_node, self.swa.root_node,
                         "new leaf should not be the root sentinel")
        # The leaf MUST be a SWARadixCache TreeNode (with full_lock_ref).
        self.assertTrue(hasattr(new_last_node, "full_lock_ref"),
                        f"leaf node {type(new_last_node)} lacks full_lock_ref")
        self.assertTrue(hasattr(new_last_node, "swa_lock_ref"))
        # inc_lock_ref ran inside init_load_back, so full_lock_ref >= 1.
        self.assertGreaterEqual(new_last_node.full_lock_ref, 1)

    def test_returns_tuple_for_schedule_policy_unpack(self):
        """schedule_policy.add_one_req does:
            new_indices, req.last_node = tree_cache.init_load_back(...)
        so init_load_back MUST return a 2-tuple (Tensor, TreeNode), not None.
        """
        req = _make_req(
            rid="rid-tuple",
            token_ids=list(range(self.page_size)),
            host_hit_length=self.page_size,
            last_node=self.swa.root_node,
        )
        result = self.cache.init_load_back(
            InitLoadBackParams(req=req, host_hit_length=self.page_size,
                               best_match_node=self.swa.root_node)
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        new_indices, last_node = result
        self.assertIsInstance(new_indices, torch.Tensor)
        # Tuple-unpack should work without raising.
        a, b = result
        self.assertIs(a, new_indices)
        self.assertIs(b, last_node)

    def test_zero_host_hit_returns_root(self):
        """host_hit_length=0 should release_load_state and return (empty, last_node)."""
        req = _make_req(
            rid="rid-zero",
            token_ids=list(range(self.page_size)),
            host_hit_length=0,
            last_node=self.swa.root_node,
        )
        new_indices, last_node = self.cache.init_load_back(
            InitLoadBackParams(req=req, host_hit_length=0,
                               best_match_node=self.swa.root_node)
        )
        self.assertEqual(new_indices.numel(), 0)
        self.assertIs(last_node, self.swa.root_node)
        self.assertIn("rid-zero", self.connector.released)

    def test_lock_ref_bookkeeping_pairs_with_dec_lock(self):
        """After init_load_back, calling dec_lock_ref on the same node must not
        underflow lock counts or hit ``dec_lock_ref on node with full_lock_ref=0``.
        """
        req = _make_req(
            rid="rid-lock",
            token_ids=list(range(self.page_size * 2)),
            host_hit_length=self.page_size * 2,
            last_node=self.swa.root_node,
        )
        new_indices, leaf = self.cache.init_load_back(
            InitLoadBackParams(req=req, host_hit_length=self.page_size * 2,
                               best_match_node=self.swa.root_node)
        )
        # Mirror what _check_load_completion does:
        from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
        # Use the same SWA UUID that inc_lock_ref returned (we don't have it,
        # but we can read it off the leaf).
        result = self.swa.dec_lock_ref(
            leaf,
            DecLockRefParams(swa_uuid_for_lock=getattr(leaf, "swa_uuid", None)),
        )
        # If we got here without assertion errors, lock pairing is sound.
        self.assertIsNotNone(result)

    def test_load_queue_is_populated(self):
        """After init_load_back, ready_to_load_host_cache should hand the op
        off to the connector, not silently no-op.
        """
        req = _make_req(
            rid="rid-queue",
            token_ids=list(range(self.page_size)),
            host_hit_length=self.page_size,
            last_node=self.swa.root_node,
        )
        self.cache.init_load_back(
            InitLoadBackParams(req=req, host_hit_length=self.page_size,
                               best_match_node=self.swa.root_node)
        )
        task_id = self.cache.ready_to_load_host_cache()
        self.assertGreaterEqual(task_id, 0)
        self.assertEqual(len(self.connector.start_load_calls), 1)
        op_task_id, ops = self.connector.start_load_calls[0]
        self.assertEqual(op_task_id, task_id)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].rid, "rid-queue")


class TestExtendedRadixCacheSWATailAllocation(unittest.TestCase):
    """Verify hybrid-SWA H2D uses alloc_extend_swa_tail (trailing-window only).

    Repro: SWA pool capacity is much smaller than full-attn pool. If we naively
    call alloc_extend(N) for an H2D of N tokens, the SWA half of the allocator
    refuses (returns None) and we report "Failed to allocate N GPU slots".

    Fix: when connector exposes a positive swa_window_size and the allocator
    has alloc_extend_swa_tail, route through it with swa_tail_len=window so
    only the window's worth of SWA slots are reserved.
    """

    def setUp(self):
        self.page_size = 256
        self.window = 256
        # SWA pool only has room for ~2 pages — way less than a typical hit.
        # If the fix is wrong (alloc_extend used), allocation fails;
        # if the fix is right (alloc_extend_swa_tail used), it succeeds because
        # only `window` (= 1 page) is needed.
        self.allocator = FakePagedAllocator(
            total_slots=8192,
            page_size=self.page_size,
            swa_capacity=self.page_size * 2,  # 512 token cap
        )
        from sglang.srt.mem_cache import swa_memory_pool as _swa_pool_mod
        _swa_pool_mod.SWATokenToKVPoolAllocator.register(FakePagedAllocator)

        req_to_token_pool = MagicMock()
        req_to_token_pool.req_to_token = torch.zeros((128, 1024), dtype=torch.int64)

        params = CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=self.page_size,
            is_eagle=False,
            sliding_window_size=self.window,
        )
        self.swa = SWARadixCache(params)
        self.connector = FakeConnector(swa_window_size=self.window)
        self.cache = ExtendedRadixCache(
            params=params,
            connector=self.connector,
            inner_cache=self.swa,
        )

    def test_swa_tail_path_is_used_when_window_set(self):
        """With swa_window_size set, allocation of a hit much larger than the
        SWA pool's capacity must succeed via alloc_extend_swa_tail."""
        # Hit is 16 pages (4096 tokens) — far exceeds the 512-token SWA cap.
        host_hit_length = self.page_size * 16
        req = _make_req(
            rid="rid-tail",
            token_ids=list(range(host_hit_length)),
            host_hit_length=host_hit_length,
            last_node=self.swa.root_node,
        )
        new_indices, leaf = self.cache.init_load_back(
            InitLoadBackParams(req=req, host_hit_length=host_hit_length,
                               best_match_node=self.swa.root_node)
        )
        self.assertEqual(
            new_indices.numel(), host_hit_length,
            f"Expected {host_hit_length} GPU slots allocated, got {new_indices.numel()}"
        )
        self.assertIsNot(leaf, self.swa.root_node)
        # alloc_extend_swa_tail must have been used.
        self.assertEqual(self.allocator.swa_tail_calls, 1,
                         "alloc_extend_swa_tail should be called exactly once")
        self.assertEqual(self.allocator.alloc_extend_calls, 0,
                         "alloc_extend should NOT be called when window is set")

    def test_falls_back_to_alloc_extend_when_no_window(self):
        """When connector has window=0 (no SWA), alloc_extend is the right path."""
        self.connector._swa_window_size = 0
        # Use a smaller hit that fits the SWA cap so alloc_extend would succeed.
        host_hit_length = self.page_size * 2
        req = _make_req(
            rid="rid-no-window",
            token_ids=list(range(host_hit_length)),
            host_hit_length=host_hit_length,
            last_node=self.swa.root_node,
        )
        self.cache.init_load_back(
            InitLoadBackParams(req=req, host_hit_length=host_hit_length,
                               best_match_node=self.swa.root_node)
        )
        self.assertEqual(self.allocator.swa_tail_calls, 0)
        self.assertEqual(self.allocator.alloc_extend_calls, 1)


class TestKVConnectorCounterRotation(unittest.TestCase):
    """Regression: with a 3-slot producer ring and no consumer rotation,
    the 4th H2D crashes with ``Producer event should be finished before reuse``.

    Repro mirrors the production crash where ``start_load_kv`` calls
    ``layer_done_counter.update_producer()`` but tp_worker never calls
    ``set_consumer(consumer_index)`` because the kv-connector counter was
    not registered with tp_worker.

    Verifies the registry wiring: once we hand the counter to tp_worker (or
    in this unit-test, manually call set_consumer) so that consumer rotates
    in lockstep with the producer, the loop runs without exhausting the ring.
    """

    def setUp(self):
        from sglang.srt.mem_cache.storage.flexkv.flexkv_comm import (
            FlexKVLayerDoneCounter,
        )
        self.counter = FlexKVLayerDoneCounter(num_layers=4)

        self.page_size = 256
        self.window = 256
        self.allocator = FakePagedAllocator(
            total_slots=8192, page_size=self.page_size,
            swa_capacity=self.page_size * 4,
        )
        from sglang.srt.mem_cache import swa_memory_pool as _swa_pool_mod
        _swa_pool_mod.SWATokenToKVPoolAllocator.register(FakePagedAllocator)

        req_to_token_pool = MagicMock()
        req_to_token_pool.req_to_token = torch.zeros((128, 1024), dtype=torch.int64)

        params = CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=self.page_size,
            is_eagle=False,
            sliding_window_size=self.window,
        )
        self.swa = SWARadixCache(params)
        self.connector = FakeConnector(
            swa_window_size=self.window,
            layer_done_counter=self.counter,
        )
        self.cache = ExtendedRadixCache(
            params=params, connector=self.connector, inner_cache=self.swa,
        )

    def _do_one_h2d(self, idx: int):
        host_hit_length = self.page_size  # 1 page per call to keep allocator happy
        req = _make_req(
            rid=f"rid-{idx}",
            token_ids=list(range(idx * 1024, idx * 1024 + host_hit_length)),
            host_hit_length=host_hit_length,
            last_node=self.swa.root_node,
        )
        self.cache.init_load_back(
            InitLoadBackParams(req=req, host_hit_length=host_hit_length,
                               best_match_node=self.swa.root_node)
        )
        task_id = self.cache.ready_to_load_host_cache()
        return task_id

    def test_ring_exhausts_without_consumer_rotation(self):
        """Without set_consumer, the 4th iteration MUST crash on update_producer.
        This pins the bug from the production log.
        """
        for i in range(3):
            self._do_one_h2d(i)
        with self.assertRaises(AssertionError) as cm:
            self._do_one_h2d(3)
        self.assertIn("finished before reuse", str(cm.exception))

    def test_consumer_rotation_keeps_ring_alive(self):
        """When the consumer rotates in lockstep (mimicking what tp_worker
        does via set_hicache_consumer once the registry wiring is in place),
        we can run many H2D ops without exhausting the ring."""
        for i in range(20):
            task_id = self._do_one_h2d(i)
            # Mimic tp_worker.set_hicache_consumer + kvcache forward consuming
            # all layers.
            self.counter.set_consumer(task_id)
            event = self.counter.events[self.counter.consumer_index]
            # Mark all layers consumed → _finished=True on last layer.
            for layer in range(self.counter.num_layers):
                # Bypass the eventfd_read by directly setting _finished on the
                # last layer; earlier layers don't gate update_producer.
                if layer == self.counter.num_layers - 1:
                    event._finished = True
        # If we got here without AssertionError, rotation works.


class TestSWAUuidPairing(unittest.TestCase):
    """Regression: the load path stored bare nodes in _ongoing_load_tasks
    instead of (node, swa_uuid) pairs. _check_load_completion called
    dec_lock_ref(node, swa_uuid_for_lock=None), which underflowed the SWA
    chain — ``dec_lock_ref on swa_tombstone node`` or ``swa_lock_ref=0``.
    """

    def setUp(self):
        self.page_size = 256
        self.window = 256
        self.swa, self.allocator = _make_swa_cache(
            page_size=self.page_size, sliding_window_size=self.window
        )
        self.connector = FakeConnector(swa_window_size=self.window)
        self.cache = ExtendedRadixCache(
            params=CacheInitParams(
                disable=False,
                req_to_token_pool=self.swa.req_to_token_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=self.page_size,
                is_eagle=False,
                sliding_window_size=self.window,
            ),
            connector=self.connector,
            inner_cache=self.swa,
        )

    def test_load_path_stores_swa_uuid_with_node(self):
        """init_load_back -> ready_to_load_host_cache: _ongoing_load_tasks
        must contain (node, swa_uuid) tuples, not bare nodes.
        """
        host_hit_length = self.page_size * 2
        req = _make_req(
            rid="rid-uuid",
            token_ids=list(range(host_hit_length)),
            host_hit_length=host_hit_length,
            last_node=self.swa.root_node,
        )
        self.cache.init_load_back(
            InitLoadBackParams(req=req, host_hit_length=host_hit_length,
                               best_match_node=self.swa.root_node)
        )
        task_id = self.cache.ready_to_load_host_cache()
        self.assertGreaterEqual(task_id, 0)
        entry = self.cache._ongoing_load_tasks[task_id]
        self.assertIsInstance(entry, list)
        self.assertEqual(len(entry), 1)
        item = entry[0]
        self.assertIsInstance(item, tuple,
                              "Should be (node, swa_uuid) tuple, not bare node")
        self.assertEqual(len(item), 2)
        node, swa_uuid = item
        self.assertTrue(hasattr(node, "full_lock_ref"))
        # On SWA tree with hit_length >= window, swa_uuid should be assigned
        # by inc_lock_ref. (None is also OK for sub-window hits.)
        # Just verify no crash on dec_lock_ref pairing:
        from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
        # Simulate _check_load_completion's dispatch
        self.swa.dec_lock_ref(node, DecLockRefParams(swa_uuid_for_lock=swa_uuid))

    def test_check_load_completion_uses_swa_uuid(self):
        """_check_load_completion must pass swa_uuid_for_lock to dec_lock_ref.

        Concrete failure mode: without this, dec_lock_ref walks SWA chain to
        root and underflows when leaf is a SWA-tombstone or when swa_lock_ref
        is already 0 above the window boundary.

        We simulate the full lifecycle: init_load_back -> ready_to_load -> mark
        connector task completed -> _check_load_completion -> verify the leaf's
        full_lock_ref/swa_lock_ref returned to pre-load values without raising.
        """
        host_hit_length = self.page_size * 4  # 4 pages, exceeds 1-page window
        req = _make_req(
            rid="rid-completion",
            token_ids=list(range(host_hit_length)),
            host_hit_length=host_hit_length,
            last_node=self.swa.root_node,
        )
        # Pre-load lock state on root
        pre_full = self.swa.root_node.full_lock_ref
        pre_swa = self.swa.root_node.swa_lock_ref

        self.cache.init_load_back(
            InitLoadBackParams(req=req, host_hit_length=host_hit_length,
                               best_match_node=self.swa.root_node)
        )
        task_id = self.cache.ready_to_load_host_cache()

        # Make the connector report this task completed.
        original_check = self.connector.check_completed_load_tasks
        self.connector.check_completed_load_tasks = lambda: [task_id]
        # Should not raise.
        self.cache._check_load_completion()
        self.connector.check_completed_load_tasks = original_check

        # Lock counts on root should be back to baseline.
        self.assertEqual(self.swa.root_node.full_lock_ref, pre_full)
        self.assertEqual(self.swa.root_node.swa_lock_ref, pre_swa)


class TestEagleBigramAlignment(unittest.TestCase):
    """EAGLE bigram + page_align silently slices key in SWARadixCache.insert.
    Without trimming device_indices to match, allocated GPU slots leak.

    Regression: with is_eagle=True and host_hit_length=page_size, after
    bigram (-1) and page_align, effective_len=0, so the entire alloc would
    leak.  The fix detects this, frees the slots, and bails cleanly.
    """

    def setUp(self):
        self.page_size = 256
        self.window = 256
        self.allocator = FakePagedAllocator(
            total_slots=8192, page_size=self.page_size,
        )
        from sglang.srt.mem_cache import swa_memory_pool as _swa_pool_mod
        _swa_pool_mod.SWATokenToKVPoolAllocator.register(FakePagedAllocator)

        req_to_token_pool = MagicMock()
        req_to_token_pool.req_to_token = torch.zeros((128, 1024), dtype=torch.int64)

        params = CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=self.page_size,
            is_eagle=True,    # <-- EAGLE on
            sliding_window_size=self.window,
        )
        self.swa = SWARadixCache(params)
        self.connector = FakeConnector(swa_window_size=self.window)
        self.cache = ExtendedRadixCache(
            params=params, connector=self.connector, inner_cache=self.swa,
        )

    def test_eagle_single_page_hit_bails_cleanly(self):
        """host_hit_length=page_size, is_eagle=True:
            bigram: 256 -> 255
            page_align: 255 -> 0
        Should NOT proceed with H2D when nothing would land in the tree.
        """
        host_hit_length = self.page_size  # exactly 1 page
        req = _make_req(
            rid="rid-eagle-tiny",
            token_ids=list(range(host_hit_length + 1)),  # +1 so bigram has data
            host_hit_length=host_hit_length,
            last_node=self.swa.root_node,
        )
        new_indices, last_node = self.cache.init_load_back(
            InitLoadBackParams(req=req, host_hit_length=host_hit_length,
                               best_match_node=self.swa.root_node)
        )
        # effective_len trims to 0 → bail with empty + root.
        self.assertEqual(new_indices.numel(), 0,
                         "single-page EAGLE hit should trim to 0 and bail")
        self.assertIn("rid-eagle-tiny", self.connector.released)

    def test_eagle_multi_page_hit_loads_full_alloc(self):
        """host_hit_length=512, is_eagle=True:
            bigram: 512 -> 511
            page_align: 511 -> 256  (effective_len in tree)

        device_indices passed to load op MUST be the full 512-slot allocation
        (not trimmed to 256). FlexKV's transfer graph has 512/page_size = 2
        source blocks; trimming dst would crash with
        ``src_block_ids.size != dst_block_ids.size`` in flexkv.common.transfer.
        The trailing partial page is soft-locked by req.prefix_indices, exactly
        like a normal paged prefill.
        """
        host_hit_length = self.page_size * 2  # 512
        req = _make_req(
            rid="rid-eagle-multi",
            token_ids=list(range(host_hit_length + 1)),
            host_hit_length=host_hit_length,
            last_node=self.swa.root_node,
        )
        new_indices, last_node = self.cache.init_load_back(
            InitLoadBackParams(req=req, host_hit_length=host_hit_length,
                               best_match_node=self.swa.root_node)
        )
        # Full host_hit_length slots are allocated and handed to the load op
        # (matching the FlexKV src graph size), even though tree only retains
        # effective_len=256 internally.
        self.assertEqual(
            new_indices.numel(), host_hit_length,
            "EAGLE multi-page hit must keep full alloc to match FlexKV src size",
        )
        self.assertIsNot(last_node, self.swa.root_node)
        # The leaf in the tree only protects effective_len indices, not all.
        self.assertEqual(last_node.value.numel(), self.page_size,
                         "leaf value should be effective_len after insert trim")

    def test_load_op_device_indices_match_host_hit_length(self):
        """Concrete prod regression: src_block_ids.size != dst_block_ids.size.

        FlexKV builds its transfer graph based on host_hit_length pages.
        The device_indices we pass into start_load_kv become the dst slot
        mapping. They MUST have exactly host_hit_length entries — the same
        page count as the host side — or flexkv.common.transfer.set_gpu_blocks
        will assert ``op.src_block_ids.size == op.dst_block_ids.size``.
        """
        # Repro the production sizes: hit_length=66048 = 258 pages.
        # Use smaller numbers so the test is fast: 4 pages, EAGLE on.
        host_hit_length = self.page_size * 4
        req = _make_req(
            rid="rid-prod-regression",
            token_ids=list(range(host_hit_length + 1)),
            host_hit_length=host_hit_length,
            last_node=self.swa.root_node,
        )
        self.cache.init_load_back(
            InitLoadBackParams(req=req, host_hit_length=host_hit_length,
                               best_match_node=self.swa.root_node)
        )
        # Inspect the queued LoadOperation directly, BEFORE ready_to_load
        # transfers it. Asserts the exact contract FlexKV cares about.
        self.assertEqual(len(self.cache._load_queue), 1)
        op = self.cache._load_queue[0]
        self.assertEqual(
            op.device_indices.numel(), host_hit_length,
            "LoadOperation.device_indices must equal host_hit_length so FlexKV "
            "src/dst block counts match",
        )


class TestSignalDenseLayersReady(unittest.TestCase):
    """Regression: scheduler hangs after first H2D because FlexKV's
    multi-group C++ worker only fires layerwise eventfds for layers that
    have at least one LayerGroupSpec member. DSv4 dense layers
    (compress_ratio=0, e.g. layers 0/1 in DSv4-Flash) have empty
    layer_members and get NO eventfd. sglang's forward calls
    ``get_swa_key_buffer_radix(0)`` → ``wait_until(0)`` → ``eventfd_read``
    which blocks forever.

    Fix: pre-fire eventfds for those dense layers right after producer
    rotation. Their SWA KV is restored by the SWA fallback path before
    ``kv_manager.launch`` is even called, so the eventfd is purely a
    "ready" signal.

    These tests exercise ``_signal_dense_layers_ready`` against a real
    ``FlexKVLayerDoneCounter`` over a realistic DSv4-Flash layout
    (43 layers, dense=[0, 1]).
    """

    DSV4_FLASH_NUM_LAYERS = 43
    DSV4_FLASH_DENSE_IDS = [0, 1]
    DSV4_FLASH_COMPRESS_RATIOS = [0, 0] + [4, 128] * 20 + [4]

    def setUp(self):
        from sglang.srt.mem_cache.storage.flexkv.flexkv_comm import (
            FlexKVLayerDoneCounter,
        )
        self.counter = FlexKVLayerDoneCounter(
            num_layers=self.DSV4_FLASH_NUM_LAYERS
        )

    def _make_stub_with_dense_ids(self, dense_ids):
        """Bind ``_signal_dense_layers_ready`` from FlexKVConnector onto a
        minimal stub object. Avoids spinning up the full connector."""
        from sglang.srt.mem_cache.storage.flexkv.flexkv_connector import (
            FlexKVConnector,
        )
        stub = type("Stub", (), {})()
        stub._layer_done_counter = self.counter
        stub._dense_layer_local_ids = list(dense_ids)
        stub._signal_dense_layers_ready = (
            FlexKVConnector._signal_dense_layers_ready.__get__(stub)
        )
        return stub

    def test_dsv4_flash_dense_ids_match_config(self):
        """Sanity-check our ratios fixture. DSv4-Flash config.json has
        compress_ratios = [0, 0, 4, 128, ..., 4, 0] = 44 entries (43 main
        layers + 1 nextn). The 43-main-layer slice has dense at [0, 1]."""
        ratios = self.DSV4_FLASH_COMPRESS_RATIOS
        self.assertEqual(len(ratios), self.DSV4_FLASH_NUM_LAYERS)
        dense = [i for i, r in enumerate(ratios) if r == 0]
        self.assertEqual(dense, self.DSV4_FLASH_DENSE_IDS)
        self.assertEqual(ratios.count(4), 21)
        self.assertEqual(ratios.count(128), 20)

    def test_prefire_unblocks_dense_layer_wait(self):
        """After ``_signal_dense_layers_ready``, ``wait_until(layer)`` for
        each dense layer must NOT block. We use SIGALRM as a safety net
        so the test fails (instead of deadlocking) if the prefire didn't
        write the eventfds."""
        import signal

        producer_id = self.counter.update_producer()
        self.counter.events[producer_id].reset_for_new_transfer()
        stub = self._make_stub_with_dense_ids(self.DSV4_FLASH_DENSE_IDS)
        stub._signal_dense_layers_ready(producer_id)

        # Register the task so set_consumer can resolve to the producer slot.
        TASK_ID = 1234
        self.counter._task_to_producer[TASK_ID] = producer_id
        self.counter.set_consumer(TASK_ID)

        def _alarm(signum, frame):
            raise TimeoutError(
                "wait_until blocked beyond 2s — prefire did not signal eventfd"
            )
        prev = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(2)
        try:
            for layer in self.DSV4_FLASH_DENSE_IDS:
                self.counter.wait_until(layer)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev)

    def test_prefire_does_not_signal_managed_layers(self):
        """Layers FlexKV's worker WILL handle (compress_ratio in {4, 128})
        must NOT have their eventfds pre-fired by us — those eventfds are
        the worker's responsibility. Pre-firing them would cause the
        eventfd counter to drift up by 1 per round (we write 1 + worker
        writes 2 per round, sglang reads 1 per round → +2 net per round)."""
        producer_id = self.counter.update_producer()
        self.counter.events[producer_id].reset_for_new_transfer()
        stub = self._make_stub_with_dense_ids(self.DSV4_FLASH_DENSE_IDS)
        stub._signal_dense_layers_ready(producer_id)

        # Eventfds for non-dense layers should still be unsignalled. Reading
        # any of them must block (we use a non-blocking dup to verify).
        import os, fcntl, errno

        for managed_layer in (2, 3, 21, 42):  # sample of c4 / c128 layers
            fd = self.counter.events[producer_id].load_event_fds[managed_layer]
            # Set non-blocking and try to read; should fail with EAGAIN.
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            try:
                with self.assertRaises(BlockingIOError,
                                       msg=f"layer {managed_layer} should NOT be pre-fired"):
                    os.read(fd, 8)
            finally:
                fcntl.fcntl(fd, fcntl.F_SETFL, flags)

    def test_prefire_eventfd_counter_is_balanced_across_rounds(self):
        """With sglang's ``wait_remaining=1`` (one read per layer per
        round), pre-firing 1 per round keeps the eventfd counter at 0
        after every wait. Three rounds: counter must stay bounded."""
        stub = self._make_stub_with_dense_ids(self.DSV4_FLASH_DENSE_IDS)

        TASK_BASE = 9000
        for round_idx in range(3):
            producer_id = self.counter.update_producer()
            self.counter.events[producer_id].reset_for_new_transfer()
            stub._signal_dense_layers_ready(producer_id)
            task_id = TASK_BASE + round_idx
            self.counter._task_to_producer[task_id] = producer_id
            self.counter.set_consumer(task_id)
            for layer in self.DSV4_FLASH_DENSE_IDS:
                self.counter.wait_until(layer)
            # Mark slot finished so update_producer can rotate next round.
            self.counter.events[producer_id]._finished = True

        # After 3 rounds, dense-layer eventfds should be drained (counter == 0).
        # Verify by setting them non-blocking and confirming a read fails.
        import os, fcntl
        last_event = self.counter.events[producer_id]
        for layer in self.DSV4_FLASH_DENSE_IDS:
            fd = last_event.load_event_fds[layer]
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            try:
                with self.assertRaises(BlockingIOError,
                                       msg=f"dense layer {layer} eventfd not drained"):
                    os.read(fd, 8)
            finally:
                fcntl.fcntl(fd, fcntl.F_SETFL, flags)

    def test_no_dense_layers_is_noop(self):
        """If the model has no dense layers (e.g. DSv2/V3 style), the
        signal call must be a no-op — no eventfds touched."""
        producer_id = self.counter.update_producer()
        self.counter.events[producer_id].reset_for_new_transfer()
        stub = self._make_stub_with_dense_ids([])  # no dense layers
        stub._signal_dense_layers_ready(producer_id)

        # All eventfds must still be empty.
        import os, fcntl
        for layer in (0, 1, 2, 42):
            fd = self.counter.events[producer_id].load_event_fds[layer]
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            try:
                with self.assertRaises(BlockingIOError):
                    os.read(fd, 8)
            finally:
                fcntl.fcntl(fd, fcntl.F_SETFL, flags)


class TestExtendedRadixCacheOOMDiagnostics(unittest.TestCase):
    """Regression: OOM paths must not raise via evictable_size / diag strings."""

    def test_wrapper_evictable_and_available_str(self):
        inner, allocator = _make_swa_cache()
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=inner.req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=inner.page_size,
            is_eagle=False,
            sliding_window_size=256,
        )
        cache = ExtendedRadixCache(
            params, connector=FakeConnector(), inner_cache=inner
        )
        self.assertEqual(cache.evictable_size(), inner.full_evictable_size())
        diag = cache.available_and_evictable_str()
        self.assertIn("Available full tokens", diag)
        self.assertIn("Available swa tokens", diag)

        from sglang.srt.mem_cache.common import available_and_evictable_str

        helper_diag = available_and_evictable_str(cache)
        self.assertIn("full_available_size", helper_diag)


if __name__ == "__main__":
    unittest.main(verbosity=2)
