"""TTFT-based admission control for PD prefill server.

Estimates the TTFT (Time To First Token) of each incoming request by combining:
  - The request's estimated prefill compute time
    = len(input_ids) * (1 - cache_hit_rate) / throughput
  - The current queue remaining time (sum of remaining times of all in-flight
    requests, subtracting already-elapsed compute time for running requests).

If the estimated TTFT exceeds a configured threshold, the request is rejected
(503) to avoid upstream gateway timeout aborts that waste KV transfers.

All parameters are driven by environment variables (see environ.py) and the
feature is disabled by default.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Tuple

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

# Sentinel value: a threshold >= this is treated as "effectively disabled".
_THRESHOLD_DISABLED = 999998.0

# Fallback throughput when runtime value is unavailable (cold start).
# A conservative small value so estimates are large (pessimistic).
_FALLBACK_THROUGHPUT = 1.0


@dataclass
class InFlightReqTracker:
    """Tracking record for a single in-flight request."""

    rid: str
    est_proc_time: float  # estimated prefill compute time (seconds)
    compute_start_time: float = 0.0  # perf_counter() when compute started, 0=not started
    stage: str = "bootstrap"  # "bootstrap" | "waiting" | "computing" | "transferring"


class TtftLimitProtector:
    """TTFT-based admission control for PD prefill mode.

    Maintains a dict of in-flight requests with their estimated compute times.
    On each new request, estimates its TTFT = queue_remaining + est_proc_time
    and rejects if it exceeds the configured threshold.

    Lifecycle hooks (called by Scheduler):
      - should_admit(num_tokens) -> (bool, float): check before enqueue
      - register(rid, num_tokens): call after admission, before enqueue
      - mark_compute_start(rid): call when request first enters forward
      - mark_transferring(rid): call when prefill compute finishes
      - deregister(rid): call when request completes or is aborted
    """

    def __init__(self, scheduler: "Scheduler"):
        self.scheduler = scheduler
        self.inflight: Dict[str, InFlightReqTracker] = {}

    def _get_throughput(self) -> float:
        """Resolve the prefill throughput to use for estimation."""
        fixed = envs.SGLANG_TTFT_PREFILL_THROUGHPUT.get()
        if fixed > 0:
            return fixed
        # Use runtime measured throughput.
        runtime = getattr(
            self.scheduler.metrics_reporter, "last_input_throughput", 0.0
        )
        if runtime > 0:
            return runtime
        return _FALLBACK_THROUGHPUT

    def estimate_proc_time(self, num_tokens: int) -> float:
        """Estimate the prefill compute time for a request with given token count."""
        hit_rate = envs.SGLANG_TTFT_CACHE_HIT_RATE.get()
        throughput = self._get_throughput()
        # Only un-cached tokens need prefill computation.
        tokens_to_compute = num_tokens * (1.0 - hit_rate)
        if throughput <= 0:
            return float("inf")
        return tokens_to_compute / throughput

    def get_queue_remaining_time(self) -> float:
        """Sum of remaining times across all in-flight requests."""
        now = time.perf_counter()
        total = 0.0
        for tracker in self.inflight.values():
            if tracker.stage in ("bootstrap", "waiting"):
                # Not yet started computing — full cost counts.
                total += tracker.est_proc_time
            elif tracker.stage == "computing":
                # Subtract already-elapsed compute time.
                elapsed = now - tracker.compute_start_time
                total += max(0.0, tracker.est_proc_time - elapsed)
            elif tracker.stage == "transferring":
                # Prefill done, KV transfer in progress.
                # v1: transfer time not counted (conservative — underestimates
                # TTFT, but transfer is usually much smaller than compute).
                pass
        return total

    def should_admit(self, num_tokens: int) -> Tuple[bool, float]:
        """Check whether a new request should be admitted.

        Returns (admit: bool, estimated_ttft: float).
        """
        threshold = envs.SGLANG_TTFT_LIMIT_THRESHOLD.get()
        # Effectively disabled.
        if threshold >= _THRESHOLD_DISABLED:
            return True, 0.0

        est_proc = self.estimate_proc_time(num_tokens)
        queue_remaining = self.get_queue_remaining_time()
        estimated_ttft = queue_remaining + est_proc

        if estimated_ttft > threshold:
            return False, estimated_ttft
        return True, estimated_ttft

    def register(self, rid: str, num_tokens: int) -> None:
        """Register a newly admitted request."""
        est_proc = self.estimate_proc_time(num_tokens)
        self.inflight[rid] = InFlightReqTracker(
            rid=rid, est_proc_time=est_proc, stage="bootstrap"
        )

    def mark_compute_start(self, rid: str) -> None:
        """Mark that a request has started prefill computation (first forward)."""
        tracker = self.inflight.get(rid)
        if tracker is not None and tracker.stage in ("bootstrap", "waiting"):
            tracker.stage = "computing"
            tracker.compute_start_time = time.perf_counter()

    def mark_transferring(self, rid: str) -> None:
        """Mark that a request's prefill compute is done, KV transfer started."""
        tracker = self.inflight.get(rid)
        if tracker is not None:
            tracker.stage = "transferring"

    def deregister(self, rid: str) -> None:
        """Remove a request from tracking (completed or aborted)."""
        self.inflight.pop(rid, None)

    def log_status(self) -> None:
        """Log current protector status (for debugging)."""
        if not self.inflight:
            return
        now = time.perf_counter()
        stages = {"bootstrap": 0, "waiting": 0, "computing": 0, "transferring": 0}
        for tracker in self.inflight.values():
            stages[tracker.stage] = stages.get(tracker.stage, 0) + 1
        remaining = self.get_queue_remaining_time()
        threshold = envs.SGLANG_TTFT_LIMIT_THRESHOLD.get()
        logger.info(
            f"TTFT protector: inflight={len(self.inflight)} "
            f"stages={stages} queue_remaining={remaining:.1f}s "
            f"threshold={threshold:.1f}s"
        )
