from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    Deque,
    List,
    Optional,
    Tuple,
)

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    PoolStats,
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import (
    ceil_align,
    raise_error_or_warn,
)
from sglang.srt.utils.watchdog import WatchdogRaw

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


logger = logging.getLogger(__name__)

# Number of recent busy-check messages buffered for the level-1 dump-on-leak path.
BUSY_MEM_CHECK_LOG_RING_SIZE = 1000


@dataclass(kw_only=True, slots=True)
class SchedulerInvariantChecker:
    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    disaggregation_mode: DisaggregationMode
    page_size: int
    full_tokens_per_layer: Optional[int]
    swa_tokens_per_layer: Optional[int]
    max_total_num_tokens: int
    server_args: ServerArgs
    tree_cache: BasePrefixCache
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    req_to_token_pool: ReqToTokenPool
    pool_stats_observer: SchedulerPoolStatsObserver
    get_last_batch: Callable
    get_running_batch: Callable
    # Optional: scheduler's waiting queue accessor.  Waiting-queue requests
    # can already hold pre-allocated GPU slots after ``init_load_back`` runs
    # (extended_radix_cache.py appends restored slots to ``req.prefix_indices``
    # while leaving ``req.req_pool_idx`` as None and ``kv_allocated_len == 0``).
    # Without an explicit accounting for those slots the full-pool invariant
    # ``total = available + evictable + protected + session_held + uncached``
    # under-counts by exactly that request's ``hit_length`` — which is what
    # tripped the 2026-07-07 Full Pool Mem Leak crash.  Default is an empty
    # accessor so backends that don't opt in stay behavior-compatible.
    get_waiting_queue: Callable = field(default_factory=lambda: (lambda: []))
    count_req_pool_leak_warnings: int = 0
    count_memory_leak_warnings: int = 0
    recent_busy_msgs: Deque[str] = field(
        default_factory=lambda: deque(maxlen=BUSY_MEM_CHECK_LOG_RING_SIZE)
    )

    @staticmethod
    def _check_pool_invariant(
        pool_name: str,
        available: int,
        evictable: int,
        protected: int,
        session_held: int,
        total: int,
        uncached: int = 0,
    ) -> Tuple[bool, str]:
        """Check: available + evictable + protected + session_held + uncached == total."""
        total_accounted = available + evictable + protected + session_held + uncached
        leak = total_accounted != total
        msg = (
            f"[{pool_name}] {total=}, {available=}, {evictable=}, "
            f"{protected=}, {session_held=}, {uncached=}"
        )
        return leak, msg

    def _check_full_pool(self, ps: PoolStats, uncached: int = 0) -> Tuple[bool, str]:
        if self.is_hybrid_swa:
            protected = self.tree_cache.full_protected_size()
            session_held = self.pool_stats_observer.session_held_full_tokens()
            total = self.full_tokens_per_layer
        elif self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            protected = self.tree_cache.full_protected_size()
            session_held = self.pool_stats_observer.session_held_tokens()
            total = self.token_to_kv_pool_allocator.size
        else:
            protected = self.tree_cache.protected_size()
            session_held = self.pool_stats_observer.session_held_tokens()
            total = self.max_total_num_tokens
        return self._check_pool_invariant(
            "full",
            ps.full_available_size,
            ps.full_evictable_size,
            protected,
            session_held,
            total,
            uncached,
        )

    def _check_swa_pool(self, ps: PoolStats, uncached: int = 0) -> Tuple[bool, str]:
        return self._check_pool_invariant(
            "swa",
            ps.swa_available_size,
            ps.swa_evictable_size,
            self.tree_cache.swa_protected_size(),
            self.pool_stats_observer.session_held_swa_tokens(),
            self.swa_tokens_per_layer,
            uncached,
        )

    def _check_mamba_pool(self, ps: PoolStats) -> Tuple[bool, str]:
        leak, msg = self._check_pool_invariant(
            "mamba",
            ps.mamba_available_size,
            ps.mamba_evictable_size,
            self.tree_cache.mamba_protected_size(),
            self.pool_stats_observer.session_held_mamba_slots(),
            self.req_to_token_pool.mamba_pool.size,
        )
        if leak:
            # Page-level leak diagnosis for mamba
            free_full_pages = set(
                self.token_to_kv_pool_allocator.free_pages.tolist()
                + self.token_to_kv_pool_allocator.release_pages.tolist()
            )
            cached_full_pages = set(self.tree_cache.all_values_flatten().tolist())
            expected_full_pages = set(
                range(1, self.token_to_kv_pool_allocator.size + 1)
            )
            leaked_full_pages = (
                expected_full_pages - free_full_pages - cached_full_pages
            )
            mamba_allocator = self.req_to_token_pool.mamba_allocator
            free_mamba_pages = set(mamba_allocator.free_slots.tolist())
            cached_mamba_pages = set(
                self.tree_cache.all_mamba_values_flatten().tolist()
            )
            expected_mamba_pages = set(range(1, mamba_allocator.size + 1))
            leaked_mamba_pages = (
                expected_mamba_pages - free_mamba_pages - cached_mamba_pages
            )
            msg += (
                f", leaked_full_pages={leaked_full_pages or None}"
                f", leaked_mamba_pages={leaked_mamba_pages or None}"
            )
        return leak, msg

    def _get_total_uncached_sizes(
        self,
    ) -> Tuple[int, int]:
        """Sum uncached tokens for full and SWA pools across all active batches.

        Returns (full_uncached, swa_uncached). For non-SWA models, swa_uncached is 0.

        For full pool: uncached = allocated - cache_protected_len
        For SWA pool:  uncached = allocated - max(cache_protected_len, swa_evicted_seqlen)

        Requests in the *waiting queue* that already went through
        ``init_load_back`` also hold pre-allocated GPU slots (in
        ``req.prefix_indices``) that are NOT accounted for by the batch
        iteration above — those slots have no ``req_pool_idx`` yet
        (only ``alloc_for_extend`` sets that) and ``kv_allocated_len`` is
        still 0.  Without counting them the full-pool invariant fails
        the moment the checker samples between ``init_load_back`` and
        the request's batch entry, which is exactly what caused the
        2026-07-07 leak: three DPs each leaked their SWA-hit request's
        entire ``hit_length`` in one shot.
        """
        # After decode: running_batch IS last_batch (same object), count once.
        # After prefill: they differ, both hold uncached tokens.
        # Use identity (is / is not), not membership or ==: ScheduleBatch's
        # dataclass __eq__ compares tensor fields and raises on ambiguous bools.
        last_batch = self.get_last_batch()
        running_batch = self.get_running_batch()
        batches = [last_batch]
        if (
            running_batch is not None
            and running_batch is not last_batch
            and not running_batch.is_empty()
        ):
            batches.append(running_batch)

        full_uncached = 0
        swa_uncached = 0
        for batch in batches:
            for req in batch.reqs:
                assert req.kv_committed_freed == req.kv_overallocated_freed
                if req.kv_committed_freed or req.req_pool_idx is None:
                    continue

                allocated_len = req.kv_allocated_len
                if self.page_size > 1:
                    allocated_len = ceil_align(allocated_len, self.page_size)
                    assert req.cache_protected_len % self.page_size == 0

                full_uncached += allocated_len - req.cache_protected_len
                if self.is_hybrid_swa:
                    swa_uncached += allocated_len - max(
                        req.cache_protected_len, req.swa_evicted_seqlen
                    )

        # ---- Waiting-queue reqs with pre-allocated slots ------------------
        # A req that hit external cache (flexkv host/SSD hit → hit_length > 0)
        # goes through ``init_load_back`` during admission attempt.  That call
        # reserves ``hit_length`` GPU slots against the KV pool allocator and
        # appends them to ``req.prefix_indices``, but the req may not be
        # admitted in the same iteration (memory quota, chunked-prefill
        # overflow, etc.).  In that window the req stays in ``waiting_queue``
        # holding real GPU slots that no invariant category currently claims:
        #
        #   * not ``available``      — allocator handed them out
        #   * not ``evictable``      — not inserted into radix
        #   * not ``protected``      — same (radix un-tracked)
        #   * not ``session_held``   — no session
        #   * not per-batch uncached — req_pool_idx is None, kv_allocated_len == 0
        #
        # The flexkv path deliberately keeps ``req.cache_protected_len`` at
        # the *pre-restore* prefix length so the extra tokens show up as
        # uncached (see extended_radix_cache.py + the ``_flexkv_uncached_restore``
        # flag).  We surface that here.
        #
        # Wrapped in a broad try/except so a stray attribute miss (mock reqs
        # in tests, third-party queue implementations that hand us plain
        # objects) can never mask a real leak by raising.  We prefer
        # "possibly under-count" to "checker crashes and hides the leak."
        try:
            waiting_reqs = self.get_waiting_queue() or ()
        except Exception:
            waiting_reqs = ()
        for req in waiting_reqs:
            try:
                # Skip reqs that were already committed (defensive; a req
                # that reached commit shouldn't still be in waiting_queue,
                # but the check is cheap and matches the batch-loop contract).
                if getattr(req, "kv_committed_freed", False):
                    continue
                prefix_indices = getattr(req, "prefix_indices", None)
                if prefix_indices is None:
                    continue
                allocated_len = int(prefix_indices.numel())
                if allocated_len <= 0:
                    continue
                cache_protected_len = int(getattr(req, "cache_protected_len", 0))
                if self.page_size > 1:
                    allocated_len = ceil_align(allocated_len, self.page_size)
                    if cache_protected_len % self.page_size != 0:
                        # Contract violation somewhere upstream; skip rather
                        # than assert (checker must be robust for leak-report
                        # purposes even in half-broken states).
                        continue
                delta = allocated_len - cache_protected_len
                if delta <= 0:
                    continue
                full_uncached += delta
                if self.is_hybrid_swa:
                    swa_evicted = int(getattr(req, "swa_evicted_seqlen", 0) or 0)
                    swa_uncached += allocated_len - max(
                        cache_protected_len, swa_evicted
                    )
            except Exception:
                # Same defensive stance as the outer try — a bad req shape
                # must not mask the leak that follows.
                continue

        return full_uncached, swa_uncached

    def self_check_during_busy(self):
        if self.get_last_batch() is None:
            return

        ps = self.pool_stats_observer.get_pool_stats()
        full_uncached, swa_uncached = self._get_total_uncached_sizes()

        full_leak, full_msg = self._check_full_pool(ps, uncached=full_uncached)

        swa_leak, swa_msg = False, ""
        if self.is_hybrid_swa:
            swa_leak, swa_msg = self._check_swa_pool(ps, uncached=swa_uncached)

        level = envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get()
        full_line = f"[Mem Check (BUSY)] {full_msg}"
        swa_line = f"[Mem Check (BUSY)] {swa_msg}" if swa_msg else None

        if level > 1:
            # Verbose: log every iteration.
            logger.info(full_line)
            if swa_line:
                logger.info(swa_line)
        elif level == 1:
            # Quiet: buffer and stay silent; flush the recent ones only on a leak.
            self.recent_busy_msgs.append(full_line)
            if swa_line:
                self.recent_busy_msgs.append(swa_line)
            if full_leak or swa_leak:
                for msg in self.recent_busy_msgs:
                    logger.info(msg)

        assert not full_leak, f"Full Pool Mem Leak Detected! {full_msg}"
        assert not swa_leak, f"SWA Pool Mem Leak Detected! {swa_msg}"

    def _check_req_pool(self):
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        session_req_count = self.pool_stats_observer.session_held_req_count()
        if len(self.req_to_token_pool.free_slots) + session_req_count != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"session_held={session_req_count}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_req_pool_leak_warnings",
                msg,
            )

    def _report_leak(self, pool_name: str, token_msg: str):
        msg = f"{pool_name} memory leak detected! {token_msg}"
        raise_error_or_warn(
            self,
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
            "count_memory_leak_warnings",
            msg,
        )

    def _check_all_pools(
        self, ps: PoolStats, uncached: int = 0
    ) -> Tuple[bool, List[str]]:
        """Check memory invariant across all pools. Returns (has_leak, messages)."""
        has_leak = False
        messages = []

        full_leak, full_msg = self._check_full_pool(ps, uncached=uncached)
        has_leak |= full_leak
        messages.append(full_msg)

        if self.is_hybrid_swa:
            swa_leak, swa_msg = self._check_swa_pool(ps)
            has_leak |= swa_leak
            messages.append(swa_msg)

        if self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            mamba_leak, mamba_msg = self._check_mamba_pool(ps)
            has_leak |= mamba_leak
            messages.append(mamba_msg)

        return has_leak, messages

    def _check_tree_cache(self):
        if (
            self.tree_cache.is_tree_cache()
            and (self.is_hybrid_swa and self.tree_cache.supports_swa())
            or (self.is_hybrid_ssm and self.tree_cache.supports_mamba())
        ):
            self.tree_cache.sanity_check()


def create_scheduler_watchdog(
    scheduler: Scheduler, watchdog_timeout: float, soft: bool = False
) -> WatchdogRaw:
    def dump_info() -> str:
        if scheduler.is_initializing:
            return ""
        _, messages = scheduler.invariant_checker._check_all_pools(
            scheduler.pool_stats_observer.get_pool_stats(),
        )
        return (
            f"{scheduler.cur_batch.batch_size()=}\n"
            f"{scheduler.cur_batch.reqs=}\n" + "\n".join(messages)
        )

    return WatchdogRaw(
        debug_name="Scheduler",
        get_counter=lambda: scheduler.forward_ct,
        is_active=lambda: scheduler.is_initializing or scheduler.cur_batch is not None,
        watchdog_timeout=watchdog_timeout,
        soft=soft,
        dump_info=dump_info,
    )
