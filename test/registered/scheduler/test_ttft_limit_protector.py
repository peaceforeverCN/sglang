"""Unit tests for TtftLimitProtector.

Tests cover:
  - should_admit: threshold-based admission decision
  - register / deregister: inflight tracking lifecycle
  - mark_compute_start: elapsed time subtraction for running requests
  - mark_transferring: stage transition removes compute time from queue
  - get_queue_remaining_time: accumulation across multiple in-flight requests
  - Environment variable defaults: feature disabled by default
  - Edge cases: empty queue, disabled threshold, chunked prefill dedup
"""

import os
import time
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.ttft_limit_protector import (
    InFlightReqTracker,
    TtftLimitProtector,
)
from sglang.test.test_utils import CustomTestCase

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    register_cpu_ci(est_time=2, suite="base-a-test-cpu")
except Exception:
    pass


def _make_protector():
    """Create a TtftLimitProtector with a minimal mocked scheduler."""
    scheduler = MagicMock()
    scheduler.metrics_reporter = MagicMock()
    scheduler.metrics_reporter.last_input_throughput = 0.0
    return TtftLimitProtector(scheduler)


class _EnvOverride:
    """Context manager that temporarily overrides envs values via os.environ.

    Since EnvField reads from os.environ on each .get() call, setting env vars
    is the cleanest way to test different configurations.
    """

    def __init__(self, **overrides):
        self.overrides = overrides
        self._saved = {}

    def __enter__(self):
        for key, val in self.overrides.items():
            env_key = key  # e.g. "SGLANG_TTFT_LIMIT_ENABLED"
            self._saved[env_key] = os.environ.get(env_key)
            os.environ[env_key] = str(val)
        return self

    def __exit__(self, *args):
        for key, old_val in self._saved.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val


class TestTtftLimitProtectorBasics(CustomTestCase):
    """Basic admission logic tests."""

    def test_empty_queue_admits(self):
        """Empty queue: any request should be admitted."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="2.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            admit, est = protector.should_admit(1000)
            self.assertTrue(admit)
            self.assertAlmostEqual(est, 0.2, places=2)

    def test_short_request_admitted(self):
        """Short request on empty queue: est < threshold -> admit."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="2.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            admit, est = protector.should_admit(100)
            self.assertTrue(admit)
            self.assertAlmostEqual(est, 0.02, places=3)

    def test_long_request_rejected(self):
        """Very long request: est > threshold -> reject."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="2.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            # 20000 tokens / 5000 = 4.0s > 2.0s
            admit, est = protector.should_admit(20000)
            self.assertFalse(admit)
            self.assertAlmostEqual(est, 4.0, places=1)

    def test_boundary_exactly_at_threshold(self):
        """Request exactly at threshold: est == threshold -> admit (not >)."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="2.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            # 10000 tokens / 5000 = 2.0s == threshold 2.0 -> admit
            admit, est = protector.should_admit(10000)
            self.assertTrue(admit)
            self.assertAlmostEqual(est, 2.0, places=1)


class TestQueueAccumulation(CustomTestCase):
    """Test that queue_remaining_time correctly accumulates across requests."""

    def test_single_request_in_queue(self):
        """One request registered: queue_remaining = est_proc."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="10.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            protector.register("req1", 5000)  # est_proc = 1.0s
            remaining = protector.get_queue_remaining_time()
            self.assertAlmostEqual(remaining, 1.0, places=1)

    def test_two_requests_accumulate(self):
        """Two requests: queue_remaining = est1 + est2."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="10.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            protector.register("req1", 5000)  # 1.0s
            protector.register("req2", 5000)  # 1.0s
            remaining = protector.get_queue_remaining_time()
            self.assertAlmostEqual(remaining, 2.0, places=1)

    def test_admission_rejected_after_accumulation(self):
        """After 2 reqs (2.0s), a 3rd with est 1.0s -> est_ttft=3.0 >= 3.0 -> admit.
        A 4th with est 1.0s would be 4.0 > 3.0 -> reject."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="3.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            protector.register("req1", 5000)
            protector.register("req2", 5000)
            # 3rd: queue=2.0, est=1.0, ttft=3.0 == threshold 3.0 -> admit
            admit3, est3 = protector.should_admit(5000)
            self.assertTrue(admit3)
            protector.register("req3", 5000)
            # 4th: queue=3.0, est=1.0, ttft=4.0 > 3.0 -> reject
            admit4, est4 = protector.should_admit(5000)
            self.assertFalse(admit4)
            self.assertAlmostEqual(est4, 4.0, places=1)

    def test_short_request_rejected_when_queue_full(self):
        """Queue full of long requests: even a short request gets rejected."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="3.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            # 3 reqs * 5000 tokens = 3.0s queue, threshold=3.0
            protector.register("r1", 5000)
            protector.register("r2", 5000)
            protector.register("r3", 5000)
            # Short request: 16 tokens -> est=0.003s, ttft=3.0+0.003=3.003 > 3.0
            admit, est = protector.should_admit(16)
            self.assertFalse(admit)
            self.assertGreater(est, 3.0)


class TestLifecycleHooks(CustomTestCase):
    """Test mark_compute_start / mark_transferring / deregister."""

    def test_mark_compute_start_reduces_remaining(self):
        """After mark_compute_start, elapsed time is subtracted."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="10.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            protector.register("r1", 5000)  # est=1.0s
            # Before compute: remaining = 1.0
            self.assertAlmostEqual(protector.get_queue_remaining_time(), 1.0, places=1)

            # Mock time to simulate 0.5s elapsed
            with patch(
                "sglang.srt.managers.scheduler_components.ttft_limit_protector.time"
            ) as mock_time:
                # mark_compute_start calls perf_counter once
                # get_queue_remaining_time calls perf_counter once
                mock_time.perf_counter.side_effect = [100.0, 100.5]
                protector.mark_compute_start("r1")
                remaining = protector.get_queue_remaining_time()
            # elapsed = 100.5 - 100.0 = 0.5, remaining = 1.0 - 0.5 = 0.5
            self.assertAlmostEqual(remaining, 0.5, places=1)

    def test_mark_compute_start_dedup(self):
        """mark_compute_start called twice: only first call takes effect."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="10.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            protector.register("r1", 5000)
            with patch(
                "sglang.srt.managers.scheduler_components.ttft_limit_protector.time"
            ) as mock_time:
                mock_time.perf_counter.return_value = 100.0
                protector.mark_compute_start("r1")
                # Should be "computing" now
                self.assertEqual(protector.inflight["r1"].stage, "computing")
                # Call again — should NOT reset (stage already computing)
                mock_time.perf_counter.return_value = 200.0
                protector.mark_compute_start("r1")
                # compute_start_time should still be 100.0, not 200.0
                self.assertEqual(protector.inflight["r1"].compute_start_time, 100.0)

    def test_mark_transferring_removes_from_queue(self):
        """After mark_transferring, request no longer contributes to queue time."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="10.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            protector.register("r1", 5000)  # est=1.0s
            protector.mark_transferring("r1")
            # Transferring requests don't count in queue_remaining
            self.assertAlmostEqual(protector.get_queue_remaining_time(), 0.0, places=2)

    def test_deregister_removes_request(self):
        """deregister completely removes the request from tracking."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="10.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            protector.register("r1", 5000)
            self.assertIn("r1", protector.inflight)
            protector.deregister("r1")
            self.assertNotIn("r1", protector.inflight)
            self.assertEqual(len(protector.inflight), 0)

    def test_deregister_unknown_rid_no_error(self):
        """deregister on non-existent rid: no error (graceful)."""
        with _EnvOverride(SGLANG_TTFT_LIMIT_ENABLED="true"):
            protector = _make_protector()
            protector.deregister("nonexistent")  # should not raise

    def test_full_lifecycle(self):
        """Full lifecycle: register -> compute -> transfer -> deregister."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="10.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            protector.register("r1", 5000)
            self.assertEqual(protector.inflight["r1"].stage, "bootstrap")

            # Simulate compute start
            with patch(
                "sglang.srt.managers.scheduler_components.ttft_limit_protector.time"
            ) as mock_time:
                mock_time.perf_counter.return_value = 100.0
                protector.mark_compute_start("r1")
            self.assertEqual(protector.inflight["r1"].stage, "computing")

            # Simulate transfer
            protector.mark_transferring("r1")
            self.assertEqual(protector.inflight["r1"].stage, "transferring")

            # Simulate done
            protector.deregister("r1")
            self.assertNotIn("r1", protector.inflight)


class TestCacheHitRate(CustomTestCase):
    """Test that cache hit rate reduces estimated compute time."""

    def test_zero_hit_rate(self):
        """hit_rate=0: all tokens need compute."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="10.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            est = protector.estimate_proc_time(5000)
        self.assertAlmostEqual(est, 1.0, places=2)

    def test_full_hit_rate(self):
        """hit_rate=1.0: no tokens need compute -> est=0."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="10.0",
            SGLANG_TTFT_CACHE_HIT_RATE="1.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            est = protector.estimate_proc_time(5000)
        self.assertAlmostEqual(est, 0.0, places=4)

    def test_partial_hit_rate(self):
        """hit_rate=0.9: only 10% tokens need compute."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="10.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.9",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            est = protector.estimate_proc_time(5000)
        # 5000 * 0.1 / 5000 = 0.1s
        self.assertAlmostEqual(est, 0.1, places=2)

    def test_hit_rate_affects_admission(self):
        """Same request: with hit_rate=0 rejected, with hit_rate=0.9 admitted."""
        # 20000 tokens, throughput=5000, threshold=2.0
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="2.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            admit, est = protector.should_admit(20000)
        self.assertFalse(admit)  # 20000/5000 = 4.0 > 2.0

        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="2.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.9",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            admit, est = protector.should_admit(20000)
        self.assertTrue(admit)  # 20000*0.1/5000 = 0.4 < 2.0


class TestDisabledByDefault(CustomTestCase):
    """Test that the feature is disabled by default and behaves correctly."""

    def test_disabled_threshold_admits_everything(self):
        """When threshold >= _THRESHOLD_DISABLED, everything is admitted."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="999999.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            # Even a huge request should be admitted
            admit, est = protector.should_admit(10_000_000)
        self.assertTrue(admit)

    def test_should_admit_when_disabled(self):
        """When threshold is sentinel, should_admit returns True, 0.0."""
        with _EnvOverride(
            SGLANG_TTFT_LIMIT_ENABLED="true",
            SGLANG_TTFT_LIMIT_THRESHOLD="999999.0",
            SGLANG_TTFT_CACHE_HIT_RATE="0.0",
            SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0",
        ):
            protector = _make_protector()
            admit, est = protector.should_admit(5000)
        self.assertTrue(admit)
        self.assertEqual(est, 0.0)


class TestThroughputFallback(CustomTestCase):
    """Test throughput resolution: fixed value vs runtime vs fallback."""

    def test_fixed_throughput_used(self):
        """When SGLANG_TTFT_PREFILL_THROUGHPUT > 0, fixed value is used."""
        with _EnvOverride(SGLANG_TTFT_PREFILL_THROUGHPUT="5000.0"):
            protector = _make_protector()
            tp = protector._get_throughput()
        self.assertEqual(tp, 5000.0)

    def test_runtime_throughput_used(self):
        """When fixed=0, runtime measured throughput is used."""
        scheduler = MagicMock()
        scheduler.metrics_reporter = MagicMock()
        scheduler.metrics_reporter.last_input_throughput = 8000.0
        with _EnvOverride(SGLANG_TTFT_PREFILL_THROUGHPUT="0.0"):
            protector = TtftLimitProtector(scheduler)
            tp = protector._get_throughput()
        self.assertEqual(tp, 8000.0)

    def test_fallback_when_runtime_zero(self):
        """When fixed=0 and runtime=0 (cold start), fallback is used."""
        scheduler = MagicMock()
        scheduler.metrics_reporter = MagicMock()
        scheduler.metrics_reporter.last_input_throughput = 0.0
        with _EnvOverride(SGLANG_TTFT_PREFILL_THROUGHPUT="0.0"):
            protector = TtftLimitProtector(scheduler)
            tp = protector._get_throughput()
        self.assertGreater(tp, 0)  # fallback throughput, not zero


class TestInFlightReqTracker(unittest.TestCase):
    """Test the InFlightReqTracker dataclass."""

    def test_default_values(self):
        tracker = InFlightReqTracker(rid="r1", est_proc_time=1.0)
        self.assertEqual(tracker.rid, "r1")
        self.assertEqual(tracker.est_proc_time, 1.0)
        self.assertEqual(tracker.compute_start_time, 0.0)
        self.assertEqual(tracker.stage, "bootstrap")


if __name__ == "__main__":
    unittest.main()
