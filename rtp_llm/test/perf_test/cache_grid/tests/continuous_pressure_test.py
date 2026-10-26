"""Bounded continuous request supply, without engine imports."""

import json
import os
import struct
import tempfile
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.cache_grid.runner import continuous_pressure as cp
from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
    decode_pressure_response,
    resolve_pressure_config,
    select_targets,
)
from rtp_llm.test.perf_test.cache_grid.runner.pressure_seed_plan import SeedPlan


def config(**changes):
    values = dict(
        seed_timeout_s=30,
        fill_timeout_s=120,
        fill_max_requests=1000,
        fill_no_progress_s=30,
        fill_input_tokens=16,
        gpu_tolerance_blocks=0,
        activity_timeout_s=30,
        drain_timeout_s=60,
        capture_timeout_s=120,
    )
    values.update(changes)
    return resolve_pressure_config(values)


def seed_plan(seed_count=0, input_tokens=16, fill_tokens=16, client_inflight=4):
    return SeedPlan(
        source="manual",
        seed_count=seed_count,
        grid_reuse_tokens=0,
        reusable_tokens=input_tokens - 1,
        commit_tail_tokens=1,
        seed_input_tokens=input_tokens,
        fill_target_tokens=fill_tokens,
        fill_input_tokens=fill_tokens,
        client_inflight=client_inflight,
        formal_config_valid=False,
        nonformal_reasons=("manual_seed_count",),
        groups=(),
    )


class PhaseTest(unittest.TestCase):
    def test_measurement_ends_without_waiting_for_profile(self):
        state = cp.PressurePhases(config(), started_s=0)
        self.assertEqual(state.advance(2, fill_ready=True), "steady")
        self.assertEqual(state.advance(5, steady_covered_s=3), "measurement")
        self.assertEqual(state.measurement_end, 65)
        self.assertEqual(state.advance(65, profile_pending=True), "support")
        self.assertEqual(state.advance(125, profile_pending=True), "drain")
        self.assertEqual(state.failure, "profile_timeout")

    def test_timeouts_and_existing_failure(self):
        state = cp.PressurePhases(config(), started_s=0)
        self.assertEqual(state.advance(120, fill_ready=True), "drain")
        self.assertEqual(state.failure, "pressure_not_reached")
        state = cp.PressurePhases(config(), started_s=0)
        state.advance(1, fill_ready=True)
        self.assertEqual(state.advance(61, steady_covered_s=3), "drain")

    def test_reject_nonfinite_and_negative_arrival_rates(self):
        for rate in (-1, float("inf"), float("nan"), True):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                cp.ArrivalClock(rate, False, 0)

    def test_execution_limits_bound_long_requests(self):
        limits = cp.ExecutionLimits(128, 8, 100, 12, 16)
        self.assertEqual(limits.concurrency_for(33), 2)
        with self.assertRaises(ValueError):
            limits.concurrency_for(128)

    def test_live_pool_admission_keeps_each_owner_geometry(self):
        limits = cp.ExecutionLimits(
            262144,
            1,
            200000,
            400,
            512,
            owner_kv_budgets=(
                ("owner=0 group=1 pool=wide", 2, 65536),
                ("owner=0 group=2 pool=narrow", 400, 512),
            ),
        )
        self.assertIsNone(limits.require_request(102400))
        self.assertEqual(limits.concurrency_for(102400), 1)
        with self.assertRaisesRegex(ValueError, "owner=0 group=1 pool=wide.*KV"):
            limits.require_request(150000)

    def test_tiered_rolling_pressure_settings_are_validated(self):
        try:
            resolved = config(
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.5,
                fill_host_max_ratio=0.8,
            )
        except ValueError as exc:
            self.fail(f"tiered rolling settings were rejected: {exc}")
        self.assertEqual(resolved.family_schedule, "tiered_rolling")
        self.assertEqual(resolved.memory_reuse_ratio, 0.5)
        self.assertEqual(resolved.fill_host_max_ratio, 0.8)
        with self.assertRaisesRegex(ValueError, "50%"):
            config(
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.25,
                fill_host_max_ratio=0.8,
            )

    def test_adaptive_pressure_settings_are_validated(self):
        resolved = config(
            family_schedule="tiered_rolling",
            memory_reuse_ratio=0.5,
            fill_host_max_ratio=0.8,
            rate_control="adaptive_backlog",
            request_rate_min=50,
            request_rate_max=400,
            backlog_low=64,
            backlog_high=192,
            max_outstanding=256,
            cohort_size=32,
        )
        self.assertEqual(resolved.rate_control, "adaptive_backlog")
        self.assertEqual(resolved.request_rate_max, 400)
        self.assertEqual(resolved.max_outstanding, 256)
        for changes in (
            {"request_rate_min": 0},
            {"request_rate_min": 500},
            {"backlog_low": 192},
            {"backlog_high": 256},
            {"cohort_size": 0},
            {"cohort_size": 31},
        ):
            adaptive = dict(
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.5,
                fill_host_max_ratio=0.8,
                rate_control="adaptive_backlog",
                request_rate_min=50,
                request_rate_max=400,
                backlog_low=64,
                backlog_high=192,
                max_outstanding=256,
                cohort_size=32,
            )
            adaptive.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config(**adaptive)


class AdaptiveBacklogControllerTest(unittest.TestCase):
    def test_piecewise_rate_stays_positive_and_decreases_with_backlog(self):
        controller_type = getattr(cp, "AdaptiveBacklogController", None)
        self.assertIsNotNone(
            controller_type, "adaptive backlog controller is not implemented"
        )
        controller = controller_type(
            rate_min=50,
            rate_max=400,
            backlog_low=64,
            backlog_high=192,
        )

        self.assertEqual(controller.rate(0), 400)
        self.assertEqual(controller.rate(64), 400)
        self.assertAlmostEqual(controller.rate(128), 225)
        self.assertEqual(controller.rate(192), 50)
        self.assertEqual(controller.rate(255), 50)

    def test_rejects_invalid_rate_and_watermark_ranges(self):
        controller_type = getattr(cp, "AdaptiveBacklogController", None)
        self.assertIsNotNone(
            controller_type, "adaptive backlog controller is not implemented"
        )
        for kwargs in (
            dict(rate_min=0, rate_max=400, backlog_low=64, backlog_high=192),
            dict(rate_min=500, rate_max=400, backlog_low=64, backlog_high=192),
            dict(rate_min=50, rate_max=400, backlog_low=192, backlog_high=192),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                controller_type(**kwargs)


class TieredRollingPlannerTest(unittest.TestCase):
    def planner(self, seed_count=1000):
        planner_type = getattr(cp, "TieredRollingPlanner", None)
        self.assertIsNotNone(
            planner_type, "tiered rolling family planner is not implemented"
        )
        return planner_type(
            effective_seed_count=seed_count,
            cohort_size=32,
            memory_reuse_ratio=0.5,
            case_indices=(1, 2, 3, 4),
        )

    def test_promotes_each_memory_family_with_the_same_case_next_cohort(self):
        planner = self.planner()

        first = planner.next_cohort()
        second = planner.next_cohort()

        first_device = [item for item in first if item.expected_cache_tier == "device"]
        first_memory = [item for item in first if item.expected_cache_tier == "memory"]
        second_device = [
            item for item in second if item.expected_cache_tier == "device"
        ]
        second_memory = [
            item for item in second if item.expected_cache_tier == "memory"
        ]
        self.assertEqual([item.family for item in first_device], list(range(984, 1000)))
        self.assertEqual([item.family for item in first_memory], list(range(16)))
        self.assertEqual(
            [(item.family, item.case_index) for item in second_device],
            [(item.family, item.case_index) for item in first_memory],
        )
        self.assertEqual([item.family for item in second_memory], list(range(16, 32)))
        self.assertEqual(
            {
                case: sum(item.case_index == case for item in first_memory)
                for case in range(1, 5)
            },
            {1: 4, 2: 4, 3: 4, 4: 4},
        )
        self.assertTrue(all(item.cohort_id == 0 for item in first))
        self.assertTrue(all(item.cohort_id == 1 for item in second))
        self.assertTrue(all(item.promoted_from_cohort == 0 for item in second_device))

    def test_effective_seed_boundary_wraps_without_using_the_initial_hot_tail(self):
        planner = self.planner(seed_count=48)

        first = planner.next_cohort()
        second = planner.next_cohort()
        third = planner.next_cohort()

        self.assertEqual(
            [item.family for item in first if item.expected_cache_tier == "device"],
            list(range(32, 48)),
        )
        self.assertEqual(
            [item.family for item in second if item.expected_cache_tier == "memory"],
            list(range(16, 32)),
        )
        self.assertEqual(
            [item.family for item in third if item.expected_cache_tier == "device"],
            list(range(16, 32)),
        )
        self.assertEqual(
            [item.family for item in third if item.expected_cache_tier == "memory"],
            list(range(16)),
        )
        self.assertTrue(
            all(
                item.family < 32
                for item in second + third
                if item.expected_cache_tier == "memory"
            )
        )

    def test_assignment_is_random_access_and_independent_of_worker_order(self):
        sequential = self.planner(seed_count=48)
        expected = sequential.next_cohort() + sequential.next_cohort()
        random_access = self.planner(seed_count=48)

        observed = {
            sequence: random_access.assignment(sequence)
            for sequence in reversed(range(64))
        }

        self.assertEqual(tuple(observed[index] for index in range(64)), expected)

    def test_freezes_only_the_contiguous_completed_seed_range(self):
        seeds = [[0], [1], [2], None, None]
        self.assertEqual(cp.freeze_seed_families(seeds), 3)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            cp.freeze_seed_families([[0], None, [2]])


class FakeClock:
    def __init__(self):
        self.now = 1.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now = round(self.now + seconds, 8)


class InlineExecutor:
    def __init__(self, **kwargs):
        self.closed = False

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        self.closed = True


class MockTerminalEvidenceTest(unittest.TestCase):
    def test_mock_policy_requires_supported_terminal_counter_semantics(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            PressureCoverage,
        )

        baseline = self.sample(1, 0.5)
        # Old servers have the mock counters but date them by settlement.
        for version in (None, 0, 2):
            with self.subTest(version=version):
                fields = asdict(baseline.mock_memory)
                fields.pop("terminal_outcome_version", None)
                if version is not None:
                    fields["terminal_outcome_version"] = version
                old = replace(baseline, mock_memory=SimpleNamespace(**fields))
                with self.assertRaisesRegex(ValueError, "terminal"):
                    PressureCoverage((old,), config(evidence_policy="mock_memory"))

    def sample(self, sequence, now, *, completed=0, h2d_failed=0, d2h_failed=0):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            GroupSample,
            MockMemorySample,
            PoolSample,
            RankSample,
        )

        return RankSample(
            0,
            "service",
            sequence,
            now,
            now,
            now,
            (
                PoolSample("gpu", 100, 0, 0, 0.8, 0.9),
                PoolSample("host", 100, 0, 0, 0.8, 0.9),
            ),
            (GroupSample(0, "FULL", ("gpu",), "host", 0, 0),),
            mock_memory=MockMemorySample(
                16,
                8,
                completed,
                h2d_failed,
                completed,
                completed,
                d2h_failed,
                terminal_outcome_version=1,
            ),
        )

    def test_failure_envelopes_overlapping_either_window_boundary_reject(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            PressureCoverage,
        )

        # The break caught here is accepting positive transfers while silently
        # discarding a failure because its sampled envelope straddles a boundary.
        for direction in ("h2d", "d2h"):
            for failure_at in (1.5, 2.5):
                with self.subTest(direction=direction, failure_at=failure_at):
                    coverage = PressureCoverage(
                        (self.sample(1, 0.5),), config(evidence_policy="mock_memory")
                    )
                    for sequence, now in enumerate((1.0, 1.5, 2.0, 2.5), 2):
                        failed = int(now >= failure_at)
                        sample = self.sample(
                            sequence,
                            now,
                            completed=4 if now >= 2 else 0,
                            **{f"{direction}_failed": failed},
                        )
                        coverage.add((sample,), now)
                    result = coverage.assess_migration(1.25, 2.25)
                    self.assertFalse(result.valid)
                    self.assertIn(f"failed_{direction}:0:0", result.reasons)

    def test_delayed_settlement_cannot_move_terminal_evidence_across_boundaries(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            PressureCoverage,
        )

        # Counters model physical terminal notifications. The later settlement
        # poll carries unchanged counters and must neither create nor hide work.
        for terminal_at, settlement_at, expected in (
            (1.0, 2.0, False),
            (2.0, 3.0, True),
        ):
            with self.subTest(terminal_at=terminal_at, settlement_at=settlement_at):
                coverage = PressureCoverage(
                    (self.sample(1, 0.0),), config(evidence_policy="mock_memory")
                )
                for sequence, now in enumerate((0.5, 1.0, 1.5, 2.0, 2.5, 3.0), 2):
                    coverage.add(
                        (
                            self.sample(
                                sequence, now, completed=4 if now >= terminal_at else 0
                            ),
                        ),
                        now,
                    )
                result = coverage.assess_migration(1.5, 2.5)
                self.assertEqual(result.valid, expected)
                if expected:
                    self.assertEqual(result.groups[0].d2h_intervals, ((1.5, 2.0),))
                    self.assertEqual(result.groups[0].h2d_intervals, ((1.5, 2.0),))


class LifecycleTest(unittest.TestCase):
    def test_mock_memory_policy_runs_formal_measurement_without_tree_pressure(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            MockMemorySample,
        )

        class Controller:
            def start(self, now):
                self.started = now

            def pending(self, now):
                return False

            def stop(self, reason, now):
                self.stopped = (reason, now)

            def finish(self, coverage):
                return dict(trace_valid=True, pressure_valid=True)

        controller = Controller()

        def mock_only(evidence, sequence, clock):
            sample = evidence.samples[0]
            return cp.PressurePoll(
                (
                    replace(
                        sample,
                        pools=tuple(
                            replace(pool, used=0, pending_release=0, host_resident=0)
                            for pool in sample.pools
                        ),
                        groups=tuple(
                            replace(group, triggers=0, gpu_host_published=0)
                            for group in sample.groups
                        ),
                        counters=(),
                        mock_memory=MockMemorySample(
                            eligible_blocks=sequence * 8,
                            planned_host_blocks=sequence * 4,
                            successful_h2d_blocks=100 + max(0, sequence - 1) * 4,
                            failed_h2d_blocks=7 if sequence > 1 else 0,
                            d2h_submitted_blocks=100 + max(0, sequence - 1) * 4,
                            d2h_completed_blocks=100 + max(0, sequence - 1) * 4,
                            d2h_failed_blocks=9 if sequence > 1 else 0,
                            terminal_outcome_version=1,
                        ),
                    ),
                ),
                evidence.raw,
                evidence.activity,
            )

        result, events, _, session = self.run_session(
            valid=False,
            seed_count=3,
            controller=controller,
            poll_transform=mock_only,
            rate=0,
            cfg=config(
                evidence_policy="mock_memory",
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.5,
                fill_host_max_ratio=0.8,
                rate_control="adaptive_backlog",
                request_rate_min=10,
                request_rate_max=20,
                backlog_low=1,
                backlog_high=2,
                max_outstanding=4,
                cohort_size=2,
                measure_s=1,
                steady_s=0.5,
                sample_period_s=0.1,
            ),
        )

        dispatches = [event for event in events if event["event"] == "dispatch"]
        self.assertIsNone(result["failure"], events)
        self.assertEqual(result["seed"]["completed"], 3)
        self.assertTrue(any(event["event"] == "fill_baseline" for event in events))
        baseline = session.coverage.baselines[0].mock_memory
        self.assertGreater(baseline.successful_h2d_blocks, 100)
        self.assertEqual(baseline.failed_h2d_blocks, 7)
        self.assertEqual(baseline.d2h_failed_blocks, 9)
        self.assertIsNotNone(result["measurement_start"])
        self.assertEqual(controller.started, result["measurement_start"])
        self.assertFalse(any(event["phase"] == "fill" for event in dispatches))
        self.assertTrue(any(event["phase"] == "measurement" for event in dispatches))
        self.assertTrue(result["migration"]["valid"])
        for group in result["migration"]["groups"]:
            self.assertTrue(group["valid"])
            self.assertTrue(
                all(
                    item["before_blocks"] >= baseline.successful_h2d_blocks
                    for item in group["h2d_evidence"]
                )
            )
            self.assertTrue(
                all(
                    item["before_blocks"] >= baseline.d2h_completed_blocks
                    for item in group["d2h_evidence"]
                )
            )
            self.assertTrue(
                all(
                    item["source"] == "mock_memory_successful_h2d_blocks"
                    for item in group["h2d_evidence"]
                )
            )
            self.assertTrue(
                all(
                    item["source"] == "mock_memory_d2h_completed_blocks"
                    for item in group["d2h_evidence"]
                )
            )

    def test_mock_memory_policy_rejects_completion_failures(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            GroupSample,
            MockMemorySample,
            PoolSample,
            PressureCoverage,
            RankSample,
        )

        def sample(sequence, now, *, completed, failed):
            return RankSample(
                0,
                "service",
                sequence,
                now,
                now,
                now,
                (
                    PoolSample("gpu", 100, 0, 0, 0.8, 0.9),
                    PoolSample("host", 100, 0, 0, 0.8, 0.9),
                ),
                (GroupSample(0, "FULL", ("gpu",), "host", 0, 0),),
                counters=(),
                mock_memory=MockMemorySample(
                    eligible_blocks=16,
                    planned_host_blocks=8,
                    successful_h2d_blocks=completed,
                    failed_h2d_blocks=failed,
                    d2h_submitted_blocks=completed,
                    d2h_completed_blocks=completed,
                    d2h_failed_blocks=failed,
                    terminal_outcome_version=1,
                ),
            )

        coverage = PressureCoverage(
            (sample(1, 1.0, completed=0, failed=0),),
            config(evidence_policy="mock_memory"),
        )
        coverage.add((sample(2, 1.5, completed=0, failed=0),), 1.5)
        coverage.add((sample(3, 2.0, completed=4, failed=1),), 2.0)
        assessment = coverage.assess_migration(1.5, 2.0)
        self.assertFalse(assessment.valid)
        self.assertIn("failed_h2d:0:0", assessment.reasons)
        self.assertIn("failed_d2h:0:0", assessment.reasons)

    def test_early_capture_completion_preserves_measurement_and_finalizes_before_finished(
        self,
    ):
        cleanup = []
        original_drain = cp.PressureSession._drain
        original_shutdown = InlineExecutor.shutdown

        def drain(session):
            original_drain(session)
            cleanup.append("drained")

        def shutdown(executor, **kwargs):
            original_shutdown(executor, **kwargs)
            cleanup.append("shutdown")

        class Controller:
            def start(self, now):
                self.started = now

            def pending(self, now):
                return False

            def stop(self, reason, now):
                self.stopped = now

            def consume(self, event):
                if event["event"] == "finished":
                    assert self.finalized

            def finish(self, coverage):
                assert cleanup == ["drained", "shutdown", "shutdown"]
                self.finalized = True
                return dict(trace_valid=True, pressure_valid=False)

        controller = Controller()
        with patch.object(cp.PressureSession, "_drain", drain), patch.object(
            InlineExecutor, "shutdown", shutdown
        ):
            result, events, _, _ = self.run_session(
                controller=controller, cfg=config(measure_s=4, steady_s=1)
            )
        self.assertAlmostEqual(
            result["measurement_end"] - result["measurement_start"], 4
        )
        self.assertFalse(any(e.get("phase") == "support" for e in events))
        self.assertTrue(result["profile"]["trace_valid"])
        self.assertFalse(result["profile"]["pressure_valid"])

    def run_session(
        self,
        *,
        valid=True,
        cfg=None,
        controller=None,
        scheduler="fifo",
        poll_transform=None,
        payload_hook=None,
        seed_count=0,
        rate=3,
        control=None,
        diagnostics_control=None,
        concurrency=4,
        batch_size=1,
        request_observer=None,
        executor_factory=InlineExecutor,
    ):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            GroupSample,
            PoolSample,
            RankSample,
        )

        clock, events, batches = FakeClock(), [], []
        counter = [0]

        def sample(sequence):
            group = GroupSample(
                0, "FULL", ("gpu",), "host", max(0, sequence - 1), max(0, sequence - 1)
            )
            return RankSample(
                0,
                "service",
                sequence,
                clock(),
                clock(),
                clock(),
                (
                    PoolSample("gpu", 100, 90, 0, 0.8, 0.9),
                    PoolSample("host", 100, 30, 0, 0.8, 0.9, 25 if valid else 0),
                ),
                (group,),
            )

        def poll(timeout):
            counter[0] += 1
            evidence = cp.PressurePoll((sample(counter[0]),), {}, ())
            return (
                poll_transform(evidence, counter[0], clock)
                if poll_transform
                else evidence
            )

        def request(payload, register, before):
            identity = before(payload)
            if request_observer:
                request_observer(identity)
            return dict(
                success=True,
                input_len=16,
                output_len=1,
            )

        def payload(kind, sequence, formal_sequence):
            if payload_hook:
                payload_hook(kind, sequence, clock)
            return {"input_ids": [1] * 16, "case": "cold"}

        state = cp.PressureSession(
            cfg or config(measure_s=3, steady_s=1, sample_period_s=0.5),
            session_id="session",
            limits=cp.ExecutionLimits(128, 8, 256, 64, 16),
            concurrency=concurrency,
            scheduler=scheduler,
            batch_size=batch_size,
            max_input_tokens=16,
            seed_plan=seed_plan(seed_count, client_inflight=concurrency),
            startup_targets=select_targets((sample(0),), (0,)),
            make_payload=payload,
            request=request,
            poll=poll,
            sink=events.append,
            set_batch=control or (lambda count, **kwargs: batches.append(count)),
            controller=controller,
            diagnostics_control=diagnostics_control,
            clock=clock,
            sleep=clock.sleep,
            executor_factory=executor_factory,
            rate=rate,
        )
        result = state.run()
        return result, events, batches, state

    def test_diagnostics_enable_and_disable_enclose_baseline_and_work(self):
        controls = []
        polls_after_disable = []

        def poll(evidence, count, clock):
            self.assertEqual(controls[0][0], "enable")
            polls_after_disable.append(controls[-1][0] == "disable")
            return evidence

        result, events, _, _ = self.run_session(
            diagnostics_control=lambda action, phase, timeout: controls.append(
                (action, phase)
            ),
            poll_transform=poll,
        )
        self.assertIsNone(result["failure"])
        self.assertEqual(controls[0], ("enable", ""))
        self.assertEqual(controls[-1], ("disable", ""))
        self.assertIn(("phase", "measurement"), controls)
        self.assertEqual(events[-1]["event"], "finished")
        self.assertTrue(polls_after_disable[-1])

    def test_diagnostics_control_uses_configured_rpc_and_application_timeouts(self):
        rpc_timeouts = []
        application_timeouts = []

        class TrackingFuture:
            def __init__(self, future):
                self.future = future

            def result(self, timeout=None):
                application_timeouts.append(timeout)
                return self.future.result(timeout=timeout)

            def __getattr__(self, name):
                return getattr(self.future, name)

        class TrackingExecutor(InlineExecutor):
            def submit(self, fn, *args):
                return TrackingFuture(super().submit(fn, *args))

        def control(action, phase, timeout):
            rpc_timeouts.append(timeout)
            return {"success": True}

        result, _, _, _ = self.run_session(
            cfg=config(control_timeout_s=7.5),
            diagnostics_control=control,
            executor_factory=TrackingExecutor,
        )

        self.assertIsNone(result["failure"])
        self.assertTrue(rpc_timeouts)
        self.assertEqual(set(rpc_timeouts), {7.5})
        self.assertIn(7.6, application_timeouts)

    def test_partial_enable_and_disable_failure_are_final_failures(self):
        for fail_action in ("enable", "disable", "phase"):
            calls = []

            def control(action, phase, timeout):
                calls.append(action)
                if action == fail_action:
                    raise ValueError("partial rank failure")

            result, events, _, _ = self.run_session(diagnostics_control=control)
            self.assertIsNotNone(result["failure"])
            self.assertEqual(calls[-1], "disable")
            self.assertTrue(any(e["event"] == "error" for e in events))
            if fail_action == "enable":
                self.assertFalse(any(e["event"] == "dispatch" for e in events))

    def test_invalid_outstanding_final_poll_cannot_be_hidden_by_clean_page(self):
        self._assert_cleanup_diagnostic_failure(final_steps=True)

    def test_profile_failure_does_not_hide_invalid_drain_poll(self):
        self._assert_cleanup_diagnostic_failure(final_steps=False)

    def _assert_cleanup_diagnostic_failure(self, *, final_steps):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_results import (
            PressureEvidence,
            PressureWriter,
        )

        _, _, _, session = self.run_session()
        session.state.phase = "drain"
        session.state.failure = None if final_steps else "profile_timeout"
        invalid = cp.PressurePoll(
            (),
            dict(
                diagnostics_error="invalid clock bounds",
                response=dict(ranks=[dict(world_rank=0)]),
            ),
        )
        session.poll_future = Future()
        session.poll_future.set_result(invalid)
        session.poll_executor = InlineExecutor()
        session.poll = lambda timeout: cp.PressurePoll(
            (), dict(response=dict(ranks=[]))
        )
        session.next_poll = float("inf")
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {})
            events = []
            session.sink = PressureEvidence(
                writer, events.append, profile_requested=True
            )
            if final_steps:
                session._final_steps()
            else:
                session._drain()
            session.emit("finished", failure=session.state.failure)
            writer.close()
            summary = json.loads(
                (Path(directory) / "pressure/summary.json").read_text()
            )
            self.assertFalse(summary["validity"]["execution_valid"])
            self.assertTrue(any(event["event"] == "error" for event in events))
            self.assertTrue(
                any(
                    event["event"] == "pressure" and event["raw"] == invalid.raw
                    for event in events
                )
            )
            if not final_steps:
                self.assertEqual(summary["result"]["failure"], "profile_timeout")

    def test_all_cold_fills_then_uninterrupted_formal_schedule(self):
        result, events, batches, state = self.run_session()
        dispatch = [e for e in events if e["event"] == "dispatch"]
        self.assertIsNone(result["failure"])
        self.assertEqual(batches, [])
        self.assertTrue(any(e["phase"] == "fill" for e in dispatch))
        self.assertFalse(any(e["phase"] == "seed" for e in dispatch))
        formal = [e for e in dispatch if e["phase"] in ("steady", "measurement")]
        self.assertEqual(
            [e["sequence"] for e in dispatch], sorted({e["sequence"] for e in dispatch})
        )
        self.assertTrue(any(e["phase"] == "measurement" for e in formal))
        for first, second in zip(formal, formal[1:]):
            self.assertAlmostEqual(
                second["scheduled_offset_s"] - first["scheduled_offset_s"], 1 / 3
            )
        self.assertTrue(
            all(
                e["dispatched_s"] < result["measurement_end"]
                for e in formal
                if e["phase"] == "measurement"
            )
        )
        self.assertGreater(state.last_poll_s, result["measurement_end"])

    def test_insufficient_residency_stops_at_request_budget(self):
        result, events, _, _ = self.run_session(
            valid=False, cfg=config(fill_max_requests=3)
        )
        self.assertEqual(result["failure"], "pressure_not_reached")
        self.assertEqual(
            sum(e["event"] == "dispatch" and e["phase"] == "fill" for e in events), 3
        )

    def test_support_capture_deadline_stops_controller(self):
        class Controller:
            def __init__(self):
                self.stops = []

            def start(self, now):
                self.started = now

            def pending(self, now):
                return True

            def stop(self, reason, now):
                self.stops.append((reason, now))

        controller = Controller()
        result, events, _, _ = self.run_session(
            controller=controller,
            cfg=config(
                measure_s=2,
                steady_s=1,
                capture_timeout_s=3,
                support_s=10,
                sample_period_s=0.5,
            ),
        )
        self.assertEqual(result["failure"], "profile_timeout")
        self.assertTrue(any(e.get("phase") == "support" for e in events))
        self.assertIn("support_pressure", result)
        self.assertAlmostEqual(controller.stops[0][1], controller.started + 3, places=2)

    def test_preload_rejects_fixed_batch_schedulers_before_any_request(self):
        with self.assertRaisesRegex(ValueError, "FIFO-only"):
            self.run_session(scheduler="batch_decode")
        for batch_size in (2, True, 1.0):
            with self.subTest(batch_size=batch_size), self.assertRaisesRegex(
                ValueError, "no fixed batch"
            ):
                self.run_session(batch_size=batch_size)

    def test_fill_time_and_no_progress_budgets(self):
        for settings in (dict(fill_timeout_s=1), dict(fill_no_progress_s=1)):
            with self.subTest(settings=settings):
                result, _, _, state = self.run_session(
                    valid=False, cfg=config(**settings)
                )
                self.assertEqual(result["failure"], "pressure_not_reached")
                self.assertLess(state.clock(), 4)

    def test_seed_transfers_cannot_satisfy_fill_evidence(self):
        def seed_only_transfers(evidence, count, clock):
            sample = evidence.samples[0]
            group = replace(
                sample.groups[0],
                triggers=int(count > 1),
                gpu_host_published=int(count > 1),
            )
            return replace(evidence, samples=(replace(sample, groups=(group,)),))

        result, events, _, session = self.run_session(
            seed_count=1,
            cfg=config(fill_max_requests=2),
            poll_transform=seed_only_transfers,
        )
        self.assertEqual(result["failure"], "pressure_not_reached")
        self.assertEqual(result["seed"]["completed"], 1)
        self.assertFalse(
            any(e.get("phase") in ("steady", "measurement") for e in events)
        )
        self.assertEqual(session.coverage.baselines[0].groups[0].triggers, 1)

    def test_terminal_migration_serializes_window_local_owner_publications(self):
        def transfers(evidence, count, clock):
            sample = evidence.samples[0]
            return replace(
                evidence,
                samples=(
                    replace(
                        sample,
                        counters=(
                            (("published_blocks", 0, "host", 2, 1, 2), count),
                            (("published_blocks", 0, "gpu", 1, 2, 1), count),
                        ),
                    ),
                ),
            )

        result, events, _, session = self.run_session(poll_transform=transfers)
        self.assertIsNone(result["failure"])
        self.assertTrue(result["migration"]["valid"])
        self.assertEqual(result["migration"]["reasons"], ())
        self.assertEqual(len(result["migration"]["groups"]), 1)
        group = result["migration"]["groups"][0]
        self.assertEqual(group["d2h_evidence"][0]["source"], "published_blocks")
        for direction in ("d2h", "h2d"):
            for start, end in group[f"{direction}_intervals"]:
                self.assertGreaterEqual(start, result["measurement_start"])
                self.assertLessEqual(end, result["measurement_end"])
        self.assertGreater(session.last_poll_s, result["measurement_end"])
        self.assertEqual(events[-1]["migration"], result["migration"])
        json.dumps(result, allow_nan=False)

    def test_preload_and_fill_publications_cannot_satisfy_measurement_migration(self):
        def seed_only(evidence, count, clock):
            sample = evidence.samples[0]
            published = int(count > 1)
            return replace(
                evidence,
                samples=(
                    replace(
                        sample,
                        counters=(
                            (("published_blocks", 0, "host", 2, 1, 2), published),
                            (("published_blocks", 0, "gpu", 1, 2, 1), published),
                        ),
                    ),
                ),
            )

        result, _, _, session = self.run_session(seed_count=1, poll_transform=seed_only)
        self.assertIsNone(result["failure"])
        self.assertEqual(
            dict(session.coverage.baselines[0].counters)[
                ("published_blocks", 0, "host", 2, 1, 2)
            ],
            1,
        )
        self.assertTrue(result["pressure"]["valid"])
        self.assertFalse(result["migration"]["valid"])
        self.assertEqual(
            result["migration"]["reasons"], ("missing_d2h:0:0", "missing_h2d_load:0:0")
        )

    def test_support_only_publications_do_not_satisfy_measurement_migration(self):
        phase, published = ["seed"], [0]

        class Controller:
            def start(self, now):
                pass

            def pending(self, now):
                return True

            def stop(self, reason, now):
                pass

        def control(action, current, timeout):
            if action == "phase":
                phase[0] = current

        def support_only(evidence, count, clock):
            if phase[0] == "support":
                published[0] += 1
            sample = evidence.samples[0]
            return replace(
                evidence,
                samples=(
                    replace(
                        sample,
                        counters=(
                            (("published_blocks", 0, "host", 2, 1, 2), published[0]),
                            (("published_blocks", 0, "gpu", 1, 2, 1), published[0]),
                        ),
                    ),
                ),
            )

        result, events, _, _ = self.run_session(
            poll_transform=support_only,
            diagnostics_control=control,
            controller=Controller(),
            cfg=config(
                measure_s=2, steady_s=1, capture_timeout_s=4, sample_period_s=0.5
            ),
        )
        self.assertTrue(any(e.get("phase") == "support" for e in events))
        self.assertGreater(published[0], 0)
        self.assertFalse(result["migration"]["valid"])
        self.assertEqual(result["migration"]["groups"][0]["d2h_intervals"], ())

    def test_seed_count_is_exact_and_baseline_precedes_sends(self):
        result, events, batches, _ = self.run_session(seed_count=7)
        self.assertIsNone(result["failure"])
        self.assertEqual(events[0]["event"], "frozen")
        self.assertEqual(batches, [])
        self.assertEqual(result["seed"]["submitted"], 7)
        self.assertEqual(result["seed"]["completed"], 7)
        self.assertEqual(
            sum(e["event"] == "dispatch" and e["phase"] == "seed" for e in events), 7
        )

    def test_tiered_seed_limit_freezes_the_completed_contiguous_range(self):
        def reach_headroom(evidence, count, clock):
            if count < 2:
                return evidence
            sample = evidence.samples[0]
            pools = tuple(
                replace(pool, used=85) if pool.pool_id == "host" else pool
                for pool in sample.pools
            )
            return replace(evidence, samples=(replace(sample, pools=pools),))

        result, events, _, _ = self.run_session(
            seed_count=20,
            concurrency=4,
            rate=0,
            poll_transform=reach_headroom,
            cfg=config(
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.5,
                fill_host_max_ratio=0.8,
                measure_s=1,
                steady_s=0.5,
                sample_period_s=0.5,
            ),
        )

        self.assertEqual(result["seed"].get("configured_limit"), 20)
        self.assertEqual(result["seed"].get("effective_seed_count"), 8)
        self.assertEqual(result["seed"].get("stop_reason"), "host_max_ratio")
        self.assertEqual(result["seed"]["planned"], 8)
        self.assertEqual(result["seed"]["submitted"], 8)
        self.assertEqual(result["seed"]["completed"], 8)
        self.assertTrue(
            any(event["event"] == "seed_limit" for event in events),
            "the effective seed boundary must be persisted",
        )

    def test_fill_stops_before_consuming_reserved_host_headroom(self):
        def exhaust_headroom(evidence, count, clock):
            # Let fill first satisfy the pressure target, then prove that the
            # reserved host headroom remains a hard stop rather than a success.
            if count < 3:
                return evidence
            sample = evidence.samples[0]
            pools = tuple(
                replace(pool, used=85) if pool.pool_id == "host" else pool
                for pool in sample.pools
            )
            return replace(evidence, samples=(replace(sample, pools=pools),))

        result, events, _, _ = self.run_session(
            valid=True,
            rate=0,
            poll_transform=exhaust_headroom,
            cfg=config(
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.5,
                fill_host_max_ratio=0.8,
                fill_max_requests=100,
                fill_no_progress_s=30,
            ),
        )

        self.assertEqual(result["failure"], "fill_headroom_exhausted")
        fill_dispatches = [
            event
            for event in events
            if event["event"] == "dispatch" and event["phase"] == "fill"
        ]
        self.assertLessEqual(len(fill_dispatches), 4)

    def test_tiered_formal_supply_refills_before_the_whole_cohort_completes(self):
        active = [None]
        formal_dispatches = []
        formal_completions = []
        completions_before_refill = []

        class StaggeredHandle:
            def __init__(self, executor, work, on_done=None):
                self.phase = None
                self.formal_index = None
                self.checks = 0
                active[0] = self
                self.value = work(lambda call: None)
                active[0] = None

            def done(self):
                if self.phase not in ("steady", "measurement", "support"):
                    return True
                self.checks += 1
                lane = self.formal_index % 32
                return self.checks > lane

            def result(self):
                if self.formal_index is not None:
                    formal_completions.append(self.formal_index)
                return self.value

            def cancel(self):
                return True

        def observe(identity):
            handle = active[0]
            handle.phase = identity.rsplit(":", 2)[-2]
            if handle.phase in ("steady", "measurement", "support"):
                handle.formal_index = len(formal_dispatches)
                if handle.formal_index == 32 and len(formal_completions) < 32:
                    completions_before_refill.append(len(formal_completions))
                formal_dispatches.append(identity)

        with patch.object(cp, "CancellableRequestHandle", StaggeredHandle):
            result, _, _, _ = self.run_session(
                concurrency=32,
                rate=0,
                request_observer=observe,
                cfg=config(
                    family_schedule="tiered_rolling",
                    memory_reuse_ratio=0.5,
                    fill_host_max_ratio=0.8,
                    steady_s=0.2,
                    measure_s=0.5,
                    sample_period_s=0.1,
                ),
            )

        self.assertIsNone(result["failure"])
        self.assertGreaterEqual(len(formal_dispatches), 64)
        self.assertEqual(completions_before_refill, [1])

    def test_adaptive_supply_dispatches_without_completions_until_hard_limit(self):
        active = [None]

        class PendingFormalHandle:
            def __init__(self, executor, work, on_done=None):
                self.formal = False
                self.cancelled = False
                active[0] = self
                self.value = work(lambda call: None)
                active[0] = None

            def done(self):
                return not self.formal or self.cancelled

            def result(self):
                return self.value

            def cancel(self):
                self.cancelled = True
                return True

        def observe(identity):
            phase = identity.rsplit(":", 2)[-2]
            if phase in ("steady", "measurement", "support"):
                active[0].formal = True

        with patch.object(cp, "CancellableRequestHandle", PendingFormalHandle):
            result, events, _, _ = self.run_session(
                concurrency=2,
                rate=0,
                request_observer=observe,
                cfg=config(
                    family_schedule="tiered_rolling",
                    memory_reuse_ratio=0.5,
                    fill_host_max_ratio=0.8,
                    rate_control="adaptive_backlog",
                    request_rate_min=2,
                    request_rate_max=8,
                    backlog_low=1,
                    backlog_high=3,
                    max_outstanding=4,
                    cohort_size=2,
                    steady_s=0.1,
                    measure_s=1,
                    sample_period_s=0.1,
                    drain_timeout_s=1,
                ),
            )

        formal = [
            event
            for event in events
            if event["event"] == "dispatch"
            and event["phase"] in ("steady", "measurement", "support")
        ]
        self.assertEqual(result["failure"], "sender_saturated")
        self.assertEqual(len(formal), 4)
        self.assertEqual(
            [event["outstanding_before"] for event in formal], [0, 1, 2, 3]
        )
        self.assertEqual([event["effective_rate"] for event in formal], [8, 8, 5, 2])
        self.assertEqual([event["formal_sequence"] for event in formal], list(range(4)))
        saturation = [event for event in events if event["event"] == "sender_saturated"]
        self.assertEqual(len(saturation), 1)
        self.assertEqual(saturation[0]["outstanding"], 4)
        self.assertEqual(result["supply"]["max_outstanding"], 4)
        self.assertTrue(result["supply"]["saturated"])

    def test_steady_uses_elapsed_coverage_and_times_out(self):
        def alternate(evidence, count, clock):
            if count > 3 and count % 2:
                return cp.PressurePoll((), {"missing": True})
            return evidence

        result, _, _, _ = self.run_session(
            poll_transform=alternate,
            cfg=config(steady_timeout_s=3, steady_s=2, sample_period_s=0.5),
        )
        self.assertEqual(result["failure"], "pressure_not_reached")
        self.assertIsNone(result["measurement_start"])

    def test_measurement_pressure_loss_never_injects_repair_fill(self):
        def lose(evidence, count, clock):
            if clock() >= 4:
                sample = evidence.samples[0]
                evidence = replace(
                    evidence,
                    samples=(
                        replace(
                            sample,
                            pools=(
                                sample.pools[0],
                                replace(sample.pools[1], host_resident=0),
                            ),
                        ),
                    ),
                )
            return evidence

        result, events, _, _ = self.run_session(poll_transform=lose)
        self.assertFalse(result["pressure"]["valid"])
        self.assertTrue(any(e["event"] == "pressure_lost" for e in events))
        formal_start = next(
            e["dispatched_s"]
            for e in events
            if e["event"] == "dispatch" and e["phase"] == "steady"
        )
        self.assertFalse(
            any(
                e["event"] == "dispatch"
                and e["phase"] == "fill"
                and e["dispatched_s"] >= formal_start
                for e in events
            )
        )

    def test_service_restart_ends_epoch(self):
        def restart(evidence, count, clock):
            return (
                replace(
                    evidence, samples=(replace(evidence.samples[0], instance="new"),)
                )
                if count > 5
                else evidence
            )

        result, _, _, _ = self.run_session(poll_transform=restart)
        self.assertEqual(result["failure"], "pressure_epoch_invalid")

    def test_complete_changed_target_inventory_ends_epoch(self):
        def change(evidence, count, clock):
            if count > 5:
                sample = evidence.samples[0]
                return replace(
                    evidence,
                    samples=(
                        replace(
                            sample, groups=(replace(sample.groups[0], group_id=1),)
                        ),
                    ),
                )
            return evidence

        result, _, _, _ = self.run_session(
            poll_transform=change, cfg=config(steady_timeout_s=4)
        )
        self.assertEqual(result["failure"], "pressure_epoch_invalid")

    def test_support_finishes_before_cap(self):
        class Controller:
            def start(self, now):
                self.started = now

            def pending(self, now):
                return now < self.started + 4

            def stop(self, reason, now):
                self.stopped = now

        controller = Controller()
        result, events, _, _ = self.run_session(controller=controller)
        self.assertIsNone(result["failure"])
        self.assertAlmostEqual(controller.stopped, controller.started + 4, places=2)
        self.assertTrue(any(e.get("phase") == "support" for e in events))

    def test_controller_stop_failure_still_returns_terminal_record(self):
        class Controller:
            def start(self, now):
                pass

            def pending(self, now):
                return False

            def stop(self, reason, now):
                raise RuntimeError("stop failed")

        result, events, _, _ = self.run_session(controller=Controller())
        self.assertEqual(result["failure"], "profile_stop_failed")
        self.assertEqual(events[-1]["event"], "finished")

    def test_dispatch_exactly_at_t1_is_support_after_input_generation(self):
        class Controller:
            def start(self, now):
                self.end = now + 3

            def pending(self, now):
                return now < self.end + 1

            def stop(self, reason, now):
                pass

        controller = Controller()
        moved = []

        def delay(kind, sequence, clock):
            if kind == "formal" and hasattr(controller, "end") and not moved:
                clock.now = controller.end
                moved.append(True)

        result, events, _, _ = self.run_session(
            controller=controller, payload_hook=delay
        )
        dispatch = [e for e in events if e["event"] == "dispatch"]
        at_end = [e for e in dispatch if e["dispatched_s"] == result["measurement_end"]]
        self.assertTrue(at_end)
        self.assertTrue(all(e["phase"] == "support" for e in at_end))

    def test_failed_poll_breaks_coverage_and_is_retained(self):
        def fail(evidence, count, clock):
            if 4 <= clock() <= 5:
                raise TimeoutError("poll timed out")
            return evidence

        result, events, _, _ = self.run_session(poll_transform=fail)
        self.assertFalse(result["pressure"]["valid"])
        self.assertTrue(any("error" in e.get("raw", {}) for e in events))

    def test_invalid_baseline_tolerance_sends_nothing_and_cleans_up(self):
        result, events, _, _ = self.run_session(cfg=config(gpu_tolerance_blocks=1000))
        self.assertEqual(result["failure"], "supervisor_failed")
        self.assertFalse(any(e["event"] == "dispatch" for e in events))

    def test_fifo_never_invokes_fixed_batch_control(self):
        result, _, _, _ = self.run_session(
            control=lambda *args, **kwargs: self.fail("FIFO scheduler control")
        )
        self.assertIsNone(result["failure"])

    def test_blocked_rpc_is_cancelled_at_drain_without_boundary_drain(self):
        self.run_blocked_rpc(seed_count=0)

    def test_blocked_seed_rpc_is_cancelled_and_reaped_before_finished(self):
        self.run_blocked_rpc(seed_count=2)

    def run_blocked_rpc(self, *, seed_count):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            GroupSample,
            PoolSample,
            RankSample,
        )

        events, cancels = [], []
        sequence = [0]

        def poll(timeout):
            now = time.monotonic()
            sequence[0] += 1
            group = GroupSample(0, "FULL", ("gpu",), "host", sequence[0], sequence[0])
            sample = RankSample(
                0,
                "service",
                sequence[0],
                now,
                now,
                now,
                (
                    PoolSample("gpu", 100, 90, 0, 0.8, 0.9),
                    PoolSample("host", 100, 30, 0, 0.8, 0.9, 25),
                ),
                (group,),
            )
            return cp.PressurePoll((sample,), {})

        def request(payload, register, before):
            identity = before(payload)
            if ":fill:" not in identity:
                cancelled = Event()

                class Call:
                    def cancel(self):
                        cancels.append(time.monotonic())
                        cancelled.set()
                        return True

                register(Call())
                if not cancelled.wait(2):
                    raise AssertionError("missing bounded cancellation")
            return dict(success=True, input_len=16, output_len=1)

        startup_targets = select_targets(poll(0.01).samples, (0,))
        sequence[0] = 0
        session = cp.PressureSession(
            config(
                seed_timeout_s=0.03,
                sample_period_s=0.01,
                steady_s=0.02,
                measure_s=0.03,
                drain_timeout_s=0.03,
                activity_timeout_s=1,
            ),
            session_id="blocked",
            limits=cp.ExecutionLimits(128, 2, 128, 64, 16),
            concurrency=2,
            scheduler="fifo",
            batch_size=1,
            max_input_tokens=16,
            seed_plan=seed_plan(seed_count, client_inflight=2),
            startup_targets=startup_targets,
            make_payload=lambda kind, i, formal_sequence: {"input_ids": [1] * 16},
            request=request,
            poll=poll,
            sink=events.append,
            set_batch=lambda count: self.fail("FIFO control"),
        )
        began = time.monotonic()
        result = session.run()
        self.assertLess(time.monotonic() - began, 1)
        self.assertEqual(
            result["failure"], "seed_timeout" if seed_count else "drain_timeout"
        )
        self.assertEqual(len(cancels), 2)
        if seed_count:
            self.assertEqual(result["seed"]["submitted"], 2)
            self.assertEqual(result["seed"]["cancelled"], 2)
            self.assertEqual(result["seed"]["completed"], 2)
            self.assertEqual(result["seed"]["pending"], 0)
            self.assertEqual(session.sender.pending, {})
            self.assertFalse(any(e.get("phase") == "fill" for e in events))
        else:
            self.assertTrue(all(t >= result["measurement_end"] for t in cancels))
        self.assertEqual(sum(e["event"] == "cancellation" for e in events), 2)
        self.assertTrue(
            any(e["event"] == "phase" and e["phase"] == "measurement" for e in events)
            == (not seed_count)
        )


class SeedLifecycleTest(unittest.TestCase):
    def run_seed(
        self,
        *,
        mode="rolling",
        count=20,
        cfg=None,
        payload_hook=None,
        request_observer=None,
    ):
        handles, reaped = [], []

        class Handle:
            def __init__(self, executor, work, on_done=None):
                self.family = len(handles)
                self.cancel_calls = 0
                self.result_value = work(lambda call: None)
                handles.append(self)

            def done(self):
                if self.family >= count:
                    return True
                if self.cancel_calls:
                    return mode != "reject_cancel"
                if mode in ("timeout", "reject_cancel", "cancel_raises"):
                    return False
                if mode == "failure":
                    return self.family == 0
                return self.family == 0 or len(handles) > 16

            def result(self):
                reaped.append(self.family)
                if self.family < count and (mode == "failure" or self.cancel_calls):
                    return dict(success=False, input_len=16, output_len=1)
                return self.result_value

            def cancel(self):
                self.cancel_calls += 1
                if mode == "cancel_raises":
                    raise RuntimeError("transport cancellation failed")
                return mode != "reject_cancel"

        def payload(kind, family, clock):
            if kind == "seed" and family == 16 and mode == "rolling":
                self.assertEqual(
                    reaped, [0], "replacement waited for the initial window"
                )
            if payload_hook:
                payload_hook(kind, family, clock)

        with patch.object(cp, "CancellableRequestHandle", Handle):
            result, events, batches, session = LifecycleTest().run_session(
                seed_count=count,
                concurrency=16,
                payload_hook=payload,
                request_observer=request_observer,
                cfg=cfg or config(seed_timeout_s=0.2, drain_timeout_s=0.1),
            )
        return result, events, session, handles

    def test_one_completion_replenishes_before_fifteen_initial_requests_drain(self):
        result, events, session, handles = self.run_seed()
        self.assertIsNone(result["failure"])
        self.assertEqual(session.max_seed_pending, 16)
        self.assertEqual(result["seed"]["planned"], 20)
        self.assertEqual(result["seed"]["submitted"], 20)
        self.assertEqual(result["seed"]["completed"], 20)
        self.assertEqual(result["seed"]["failed"], 0)
        self.assertEqual(result["seed"]["cancelled"], 0)
        self.assertEqual(result["seed"]["pending"], 0)
        self.assertEqual(result["seed"]["max_inflight"], 16)
        self.assertFalse(any(e.get("phase") == "padding" for e in events))
        self.assertEqual(events[-1]["seed"], result["seed"])
        phases = [e["phase"] for e in events if e["event"] == "phase"]
        self.assertEqual(phases, ["seed", "fill", "steady", "measurement", "drain"])

    def test_seed_timeout_cancels_every_accepted_handle_and_never_enters_fill(self):
        result, events, _, handles = self.run_seed(mode="timeout")
        self.assertEqual(result["failure"], "seed_timeout")
        self.assertEqual(result["seed"]["submitted"], 16)
        self.assertEqual(result["seed"]["completed"], 0)
        self.assertEqual(result["seed"]["cancelled"], 16)
        self.assertEqual(result["seed"]["pending"], 0)
        self.assertTrue(all(handle.cancel_calls == 1 for handle in handles))
        self.assertFalse(any(e.get("phase") == "fill" for e in events))
        self.assertIsNone(result["measurement_start"])

    def test_seed_failure_stops_replacements_and_cancels_other_accepted_work(self):
        result, events, _, handles = self.run_seed(mode="failure")
        self.assertEqual(result["failure"], "request_failed")
        self.assertEqual(result["seed"]["submitted"], 16)
        self.assertEqual(result["seed"]["failed"], 1)
        self.assertEqual(result["seed"]["cancelled"], 15)
        self.assertEqual(result["seed"]["pending"], 0)
        self.assertTrue(all(handle.cancel_calls == 1 for handle in handles[1:]))
        self.assertFalse(any(e.get("phase") == "fill" for e in events))

    def test_rejected_or_raising_cancel_is_attempted_once_for_every_seed(self):
        for mode, pending in (("reject_cancel", 16), ("cancel_raises", 0)):
            with self.subTest(mode=mode):
                result, events, _, handles = self.run_seed(mode=mode)
                self.assertEqual(result["failure"], "seed_timeout")
                self.assertEqual(result["seed"]["cancelled"], 16)
                self.assertEqual(result["seed"]["pending"], pending)
                self.assertTrue(all(handle.cancel_calls == 1 for handle in handles))
                self.assertEqual(sum(e["event"] == "cancellation" for e in events), 16)
                self.assertFalse(any(e.get("phase") == "fill" for e in events))

    def test_seed_budget_is_independent_of_fill_budget(self):
        def delay(kind, family, clock):
            if kind == "seed" and family == 0:
                clock.now += 2

        result, events, _, _ = self.run_seed(
            cfg=config(seed_timeout_s=5, fill_timeout_s=1, fill_no_progress_s=1),
            payload_hook=delay,
        )
        self.assertIsNone(result["failure"])
        self.assertGreaterEqual(result["seed"]["elapsed_s"], 2)
        fill = next(e for e in events if e["event"] == "phase" and e["phase"] == "fill")
        self.assertGreaterEqual(fill["at_s"], 3)

    def test_seed_input_preparation_cannot_dispatch_after_seed_deadline(self):
        def delay(kind, family, clock):
            if kind == "seed":
                clock.now += 1

        result, events, _, _ = self.run_seed(payload_hook=delay)
        self.assertEqual(result["failure"], "seed_timeout")
        self.assertFalse(any(e["event"] == "dispatch" for e in events))
        self.assertFalse(any(e.get("phase") == "fill" for e in events))

    def test_seed_dispatch_preparation_cannot_cross_transport_deadline(self):
        original_admission = cp.ExecutionLimits.concurrency_for
        original_sha256 = cp.hashlib.sha256
        original_emit = cp.PressureSession.emit
        for stage in ("admission", "hash", "event"):
            with self.subTest(stage=stage):
                clocks, sends = [], []

                def payload(kind, family, clock):
                    if kind == "seed":
                        self.assertEqual(clock(), 1.0)
                        clocks.append(clock)

                def admission(limits, input_tokens):
                    result = original_admission(limits, input_tokens)
                    if stage == "admission" and clocks:
                        clocks[0].now = 2.0
                    return result

                def sha256(value):
                    result = original_sha256(value)
                    if stage == "hash" and clocks:
                        clocks[0].now = 2.0
                    return result

                def emit(session, event, **values):
                    original_emit(session, event, **values)
                    if stage == "event" and event == "dispatch":
                        clocks[0].now = 2.0

                with patch.object(
                    cp.ExecutionLimits, "concurrency_for", admission
                ), patch.object(cp.hashlib, "sha256", sha256), patch.object(
                    cp.PressureSession, "emit", emit
                ):
                    result, events, session, handles = self.run_seed(
                        count=1, payload_hook=payload, request_observer=sends.append
                    )
                self.assertEqual(sends, [])
                self.assertEqual(result["failure"], "seed_timeout")
                self.assertEqual(result["seed"]["submitted"], 1)
                self.assertEqual(result["seed"]["completed"], 0)
                self.assertEqual(result["seed"]["failed"], 1)
                self.assertEqual(result["seed"]["pending"], 0)
                self.assertEqual(session.sender.pending, {})
                self.assertFalse(any(e.get("phase") == "fill" for e in events))
                if stage == "event":
                    completion = next(e for e in events if e["event"] == "completion")
                    self.assertEqual(completion["result"], {"not_dispatched": True})
                else:
                    self.assertFalse(any(e["event"] == "dispatch" for e in events))


from rtp_llm.test.perf_test.cache_grid.runner.continuous_pressure import (
    ArrivalClock,
    BoundedSender,
    CancellableRequestHandle,
)


class WarmupCutoffTest(unittest.TestCase):
    def run_cutoff(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            GroupSample,
            PoolSample,
            RankSample,
        )

        clock = FakeClock()
        first_sent, crossed = Event(), Event()
        events, dispatched, cancelled, batches = [], [], [], []
        sample_sequence = [0]

        def poll(timeout):
            sample_sequence[0] += 1
            now = clock()
            sample = RankSample(
                0,
                "service",
                sample_sequence[0],
                now,
                now,
                now,
                (
                    PoolSample("gpu", 100, 90, 0, 0.8, 0.9),
                    PoolSample("host", 100, 0, 0, 0.8, 0.9, 0),
                ),
                (GroupSample(0, "FULL", ("gpu",), "host", 0, 0),),
            )
            return cp.PressurePoll((sample,), {})

        def payload(request_kind, job_id, formal_sequence):
            self.assertEqual(request_kind, "fill")
            if job_id == 1:
                if not first_sent.wait(2):
                    raise AssertionError("first request never actually dispatched")
                clock.now = 1.21
                crossed.set()
            return {"input_ids": [job_id + 1] * 16}

        def request(payload, register, before):
            identity = before(payload)
            dispatched.append(identity)
            if len(dispatched) == 1:
                first_sent.set()

            class Call:
                def cancel(self):
                    cancelled.append(identity)
                    return True

            register(Call())
            return dict(success=True, input_len=16, output_len=1)

        def sleep(seconds):
            # Only the second worker advances across the fill deadline. After
            # that, let cleanup progress even if the partial batch is blocked.
            if crossed.is_set():
                clock.now = round(clock.now + 0.01, 8)
            time.sleep(0.001)

        startup_targets = select_targets(poll(0.01).samples, (0,))
        sample_sequence[0] = 0
        session = cp.PressureSession(
            config(fill_timeout_s=0.2, drain_timeout_s=0.2, sample_period_s=0.01),
            session_id="partial",
            limits=cp.ExecutionLimits(128, 2, 128, 64, 16),
            concurrency=2,
            scheduler="fifo",
            batch_size=1,
            max_input_tokens=16,
            seed_plan=seed_plan(client_inflight=2),
            startup_targets=startup_targets,
            make_payload=payload,
            request=request,
            poll=poll,
            sink=events.append,
            set_batch=lambda count, **kwargs: batches.append(count),
            clock=clock,
            sleep=sleep,
        )
        result = session.run()
        self.assertTrue(first_sent.is_set())
        self.assertTrue(crossed.is_set())
        self.assertEqual(result["failure"], "pressure_not_reached")
        return events, dispatched, cancelled, batches, session

    def test_fifo_never_dispatches_warmup_padding(self):
        _, dispatched, cancelled, batches, _ = self.run_cutoff()
        self.assertEqual(dispatched, ["partial:fill:0"])
        self.assertEqual(batches, [])
        self.assertEqual(cancelled, [])


class BoundedSenderTest(unittest.TestCase):
    def test_does_not_construct_or_submit_when_full(self):
        sender = BoundedSender(2)
        first, second, third = Future(), Future(), Future()
        calls = []
        self.assertTrue(sender.submit(0, lambda: first))
        self.assertTrue(sender.submit(1, lambda: second))
        self.assertFalse(sender.submit(2, lambda: calls.append(2) or third))
        self.assertEqual(calls, [])
        first.set_result({"success": True})
        self.assertEqual(sender.collect(), [(0, {"success": True})])
        self.assertTrue(sender.submit(2, lambda: third))
        sender.cancel_all()
        self.assertTrue(second.cancelled())
        self.assertTrue(third.cancelled())

    def test_failure_becomes_record_and_releases_slot(self):
        sender = BoundedSender(1)
        failed = Future()
        failed.set_exception(RuntimeError("broken"))
        sender.submit(1, lambda: failed)
        self.assertEqual(
            sender.collect(),
            [(1, {"success": False, "error": "RuntimeError('broken')"})],
        )
        self.assertEqual(sender.collect(), [])
        self.assertTrue(sender.submit(2, Future))

    def test_invalid_limit_and_duplicate_id(self):
        with self.assertRaises(ValueError):
            BoundedSender(0)
        sender = BoundedSender(1)
        sender.submit(0, Future)
        with self.assertRaises(ValueError):
            sender.submit(0, Future)

    def test_backpressure_does_not_advance_schedule(self):
        clock = ArrivalClock(2, False, 0)
        sender = BoundedSender(1)
        first = Future()
        due = clock.next_offset()
        sender.submit(0, lambda: first)
        due = clock.next_offset()
        self.assertFalse(sender.submit(1, Future))
        self.assertEqual(due, 0.5)
        first.set_result({})
        sender.collect()
        self.assertTrue(sender.submit(1, Future))
        self.assertEqual(clock.next_offset(), 1.0)


class ArrivalClockTest(unittest.TestCase):
    def test_burst_and_constant_offsets(self):
        burst = ArrivalClock(0, False, 0)
        regular = ArrivalClock(2, False, 0)
        self.assertEqual([burst.next_offset() for _ in range(3)], [0, 0, 0])
        self.assertEqual([regular.next_offset() for _ in range(3)], [0, 0.5, 1])

    def test_poisson_replays_seed(self):
        first, second = ArrivalClock(2, True, 9), ArrivalClock(2, True, 9)
        offsets = [first.next_offset() for _ in range(10)]
        self.assertEqual(offsets, [second.next_offset() for _ in range(10)])
        self.assertTrue(all(b > a for a, b in zip(offsets, offsets[1:])))


class CancellableRequestHandleTest(unittest.TestCase):
    def test_completion_wakes_supervisor(self):
        woke = Event()
        with ThreadPoolExecutor(max_workers=1) as executor:
            handle = CancellableRequestHandle(
                executor, lambda register: {}, on_done=woke.set
            )
            self.assertTrue(woke.wait(2))
            self.assertTrue(handle.done())

    def test_cancels_running_rpc(self):
        started, cancelled = Event(), Event()

        class Call:
            def cancel(self):
                cancelled.set()
                return True

        def request(register_call):
            register_call(Call())
            started.set()
            if not cancelled.wait(2):
                raise AssertionError("RPC was not cancelled")
            return {"success": False}

        with ThreadPoolExecutor(max_workers=1) as executor:
            handle = CancellableRequestHandle(executor, request)
            self.assertTrue(started.wait(2))
            self.assertTrue(handle.cancel())
            self.assertTrue(cancelled.is_set())
            self.assertEqual(handle.result(), {"success": False})
            self.assertTrue(handle.done())

    def test_cancel_queued_worker_does_not_start_request(self):
        entered, release = Event(), Event()
        calls = []

        def occupy_worker():
            entered.set()
            release.wait(2)

        with ThreadPoolExecutor(max_workers=1) as executor:
            blocker = executor.submit(occupy_worker)
            self.assertTrue(entered.wait(2))
            handle = CancellableRequestHandle(
                executor, lambda register: calls.append("started") or {}
            )
            self.assertTrue(handle.cancel())
            release.set()
            blocker.result()
            self.assertTrue(handle.done())
            self.assertEqual(calls, [])

    def test_cancel_before_rpc_registration_cancels_late_call(self):
        entered, proceed, cancelled = Event(), Event(), Event()

        class Call:
            def cancel(self):
                cancelled.set()
                return True

        def request(register_call):
            entered.set()
            proceed.wait(2)
            register_call(Call())
            return {}

        with ThreadPoolExecutor(max_workers=1) as executor:
            handle = CancellableRequestHandle(executor, request)
            self.assertTrue(entered.wait(2))
            handle.cancel()
            proceed.set()
            self.assertEqual(handle.result(), {})
            self.assertTrue(cancelled.is_set())


class TransportRegistrationTest(unittest.TestCase):
    def test_empty_or_non_inference_stream_is_not_a_successful_request(self):
        from rtp_llm.dash_sc.proto.predict_v2_pb2 import ModelStreamInferResponse
        from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
            _post_prefill_ids,
        )

        dependencies = {
            "rtp_llm.dash_sc.client": SimpleNamespace(
                build_model_infer_request=lambda **kwargs: kwargs
            ),
            "rtp_llm.dash_sc.codec": SimpleNamespace(
                SamplingParams=lambda **kwargs: kwargs
            ),
        }
        for responses in ([], [ModelStreamInferResponse()]):
            with self.subTest(responses=len(responses)), patch.dict(
                "sys.modules", dependencies
            ):
                result = _post_prefill_ids(
                    SimpleNamespace(ModelStreamInfer=lambda *a, **k: iter(responses)),
                    [1, 2],
                    2,
                    "empty-stream",
                )
            self.assertFalse(result["success"])
            self.assertIn("inference response", result["error"])

    def test_closed_dispatch_propagates_without_sending(self):
        from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
            _post_prefill_ids,
        )

        stub = Mock()
        dependencies = {
            "rtp_llm.dash_sc.client": SimpleNamespace(
                build_model_infer_request=lambda **kwargs: SimpleNamespace(id="pending")
            ),
            "rtp_llm.dash_sc.codec": SimpleNamespace(
                SamplingParams=lambda **kwargs: kwargs
            ),
        }

        def closed():
            raise cp.DispatchClosed()

        with patch.dict("sys.modules", dependencies), self.assertRaises(
            cp.DispatchClosed
        ):
            _post_prefill_ids(stub, [1], 2, "pending", before_dispatch=closed)
        stub.ModelStreamInfer.assert_not_called()

    def test_dispatch_hook_sets_identity_immediately_before_rpc(self):
        from rtp_llm.dash_sc.proto.predict_v2_pb2 import ModelStreamInferResponse
        from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
            _post_prefill_ids,
        )

        order = []
        request = SimpleNamespace(id="old")

        def build(**kwargs):
            order.append("build")
            return request

        def before():
            order.append("dispatch")
            return "session:support:9"

        def invoke(requests, **kwargs):
            self.assertEqual(next(requests).id, "session:support:9")
            order.append("rpc")
            response = ModelStreamInferResponse()
            response.infer_response.parameters["prompt_token_num"].int64_param = 1
            response.infer_response.outputs.add(
                name="generated_ids", datatype="INT32", shape=[1]
            )
            response.infer_response.raw_output_contents.append(struct.pack("<i", 7))
            return iter((response,))

        dependencies = {
            "rtp_llm.dash_sc.client": SimpleNamespace(build_model_infer_request=build),
            "rtp_llm.dash_sc.codec": SimpleNamespace(
                SamplingParams=lambda **kwargs: kwargs
            ),
        }
        with patch.dict("sys.modules", dependencies):
            result = _post_prefill_ids(
                SimpleNamespace(ModelStreamInfer=invoke),
                [1],
                2,
                "old",
                before_dispatch=before,
            )
        self.assertTrue(result["success"])
        self.assertEqual(result["request_id"], "session:support:9")
        self.assertEqual(order, ["build", "dispatch", "rpc"])

    def test_post_prefill_ids_registers_actual_call_before_iteration(self):
        from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
            _post_prefill_ids,
        )

        registered = []

        class Call:
            cancelled = False

            def cancel(self):
                self.cancelled = True
                return True

            def __iter__(self):
                self_test.assertEqual(registered, [self])
                self_test.assertTrue(self.cancelled)
                return iter(())

        self_test = self
        call = Call()
        stub = Mock()
        stub.ModelStreamInfer.return_value = call

        def register_call(value):
            registered.append(value)
            value.cancel()

        # Only wire builders are replaced: importing them otherwise loads the
        # engine extension. Exercise the real adapter and stream lifecycle.
        dependencies = {
            "rtp_llm.dash_sc.client": SimpleNamespace(
                build_model_infer_request=lambda **kwargs: kwargs
            ),
            "rtp_llm.dash_sc.codec": SimpleNamespace(
                SamplingParams=lambda **kwargs: kwargs
            ),
        }
        with patch.dict("sys.modules", dependencies):
            result = _post_prefill_ids(
                stub, [1, 2], 17, "registered", register_call=register_call
            )
        self.assertFalse(result["success"])
        self.assertEqual(stub.ModelStreamInfer.call_args.kwargs["timeout"], 17)


class RunnerIntegrationTest(unittest.TestCase):
    def test_pressure_wire_decodes_mock_completion_and_failure_counters(self):
        clock = FakeClock()
        response = self._startup_pressure(clock, 1, startup=True)
        for rank in response.ranks:
            rank.mock_memory.eligible_blocks = 16
            rank.mock_memory.planned_host_blocks = 8
            rank.mock_memory.successful_h2d_blocks = 7
            rank.mock_memory.failed_h2d_blocks = 1
            rank.mock_memory.d2h_submitted_blocks = 9
            rank.mock_memory.d2h_completed_blocks = 8
            rank.mock_memory.d2h_failed_blocks = 1
            # Literal protobuf tag 8, varint value 1 also checks compatibility
            # when the binding predates the terminal-outcome capability.
            rank.mock_memory.ParseFromString(
                rank.mock_memory.SerializeToString() + b"\x40\x01"
            )

        decoded = decode_pressure_response(response, clock(), clock(), (0, 1))

        for sample in decoded.samples:
            self.assertIsNotNone(sample.mock_memory)
            self.assertEqual(
                asdict(sample.mock_memory),
                {
                    "eligible_blocks": 16,
                    "planned_host_blocks": 8,
                    "successful_h2d_blocks": 7,
                    "failed_h2d_blocks": 1,
                    "d2h_submitted_blocks": 9,
                    "d2h_completed_blocks": 8,
                    "d2h_failed_blocks": 1,
                    "terminal_outcome_version": 1,
                },
            )

    def _startup_runner(self, directory, **changes):
        from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
            CacheGridRunner,
        )

        tokenizer = SimpleNamespace(
            encode=lambda text: [sum(map(ord, word)) for word in text.split()]
        )
        options = dict(
            pressure_mode="continuous",
            pressure_scheduler="fifo",
            pressure_client_concurrency=16,
            pressure_request_rate=4,
            pressure_config=config(
                fill_input_tokens=102400,
                steady_s=0.2,
                measure_s=0.3,
                sample_period_s=0.1,
            ),
            request_transport="dashsc_input_ids",
            expected_block_size=512,
            cache_commit_tail_tokens=4096,
            grid_metadata={"generator": {"cache_alignment": 4096}},
            run_config={
                "full_pressure_startup": dict(
                    dp_size=1,
                    exclusive_service=True,
                    target_ranks=[0, 1],
                    model_tokens=262144,
                    admission_requests=1,
                    batch_tokens=200000,
                    block_tokens=512,
                    test_block_num=0,
                )
            },
        )
        options.update(changes)
        runner = CacheGridRunner(
            123,
            tokenizer,
            [dict(case_id=7, input_len=135168, cache_len=131072, batch_size=1)],
            directory,
            **options,
        )
        self.addCleanup(runner._close_resources)
        runner._pressure_epoch_dir = Path(directory) / "epoch"
        runner._pressure_epoch_dir.mkdir()
        return runner

    def _startup_pressure(self, clock, sequence, *, startup=False, fault=None):
        from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb

        response = pb.CachePressureResponsePB(schema_version=1, requested_ranks=[0, 1])
        owner = response.ranks.add(
            world_rank=0,
            service_instance="owner-instance",
            sequence=sequence,
            sampled_monotonic_ns=int(clock() * 1e9),
        )
        owner.ownership.mode = pb.PressureOwnershipPB.SHARED_TP
        owner.ownership.owner_world_rank.value = 0
        owner.ownership.participant_world_ranks.extend([0, 1])
        owner.ownership.publication_scope = pb.PressureOwnershipPB.ALL_PARTICIPANTS
        owner.pools.add(
            pool_id="gpu",
            pool_name="device",
            tier=1,
            total_blocks=300,
            free_blocks=300 if startup else 30,
            available_blocks=300,
            used_blocks=0 if startup else 270,
            block_size_bytes=4096,
        )
        owner.pools.add(
            pool_id="host",
            pool_name="host",
            tier=2,
            total_blocks=2048,
            free_blocks=1024,
            available_blocks=1024,
            used_blocks=0 if startup else 600,
            published_host_blocks=0 if startup else 600,
            block_size_bytes=4096,
        )
        group = owner.groups.add(
            group_id=0,
            group_type=1,
            device_pool_ids=["gpu"],
            group_tags=["device"],
            block_tokens=512,
            payload_bytes=4096,
            storage_bytes=4096,
            watermark_trigger_count=0 if startup else sequence,
        )
        group.host_pool_id.value = "host"
        group.device_watermark.low_ratio, group.device_watermark.high_ratio = 0.8, 0.9
        group.host_watermark.low_ratio, group.host_watermark.high_ratio = 0.8, 0.9
        owner.transfers.add(
            group_id=0,
            pool_id="host",
            pool_tier=2,
            operation=2,
            source_tier=1,
            target_tier=2,
            published_blocks=0 if startup else sequence,
        )
        owner.activity.running_requests.value = 1
        owner.activity.waiting_requests.value = 0
        owner.step_history_available = True
        peer = response.ranks.add()
        peer.CopyFrom(owner)
        peer.world_rank, peer.service_instance = 1, "participant-instance"
        for pool in peer.pools:
            pool.available_blocks = 1
            pool.used_blocks = pool.published_host_blocks = 0
        if startup:
            if fault == "missing_full":
                owner.groups[0].group_type = peer.groups[0].group_type = 2
            elif fault == "available_overflow":
                owner.pools[0].available_blocks = 301
            elif fault == "missing_owner_pool":
                del owner.pools[0]
            elif fault == "infeasible":
                owner.pools[1].available_blocks = 0
        return response

    def _run_startup_epoch(self, runner, fault=None, submissions=None):
        from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
        from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2_grpc as rpc

        clock, observations, sessions = FakeClock(), [], []
        startup_connect_failures = 1 if fault == "startup_connect_once" else 0
        submissions = [] if submissions is None else submissions
        original = cp.PressureSession

        def poll(request, timeout):
            nonlocal startup_connect_failures
            startup = not request.session_id
            if startup and (
                startup_connect_failures
                or fault
                in (
                    "startup_connect_always",
                    "startup_nontransport",
                    "startup_malformed_connect",
                )
            ):
                if startup_connect_failures:
                    startup_connect_failures -= 1
                response = self._startup_pressure(clock, 1, startup=True)
                response.ranks[1].error = (
                    "pressure diagnostics unavailable"
                    if fault == "startup_nontransport"
                    else "Connect Failed"
                )
                if fault == "startup_malformed_connect":
                    response.schema_version = 2
                return response
            if startup:
                self.assertEqual(observations, [])
                self.assertIsNone(runner._pressure_seed_plan)
                self.assertEqual(runner._continuous_seed_ids, [])
            else:
                self.assertIsNotNone(runner._pressure_seed_plan)
            observations.append("startup" if startup else "poll")
            response = self._startup_pressure(
                clock, len(observations), startup=startup, fault=fault
            )
            if not startup and not any(
                ":fill:" in identity for identity, _ in submissions
            ):
                for rank in response.ranks:
                    rank.pools[0].used_blocks = rank.pools[1].published_host_blocks = 0
                    rank.groups[0].watermark_trigger_count = 0
                    rank.transfers[0].published_blocks = 0
            if not startup and len(observations) == 3:
                for rank in response.ranks:
                    if fault == "baseline_geometry":
                        rank.groups[0].block_tokens = 256
                    elif fault == "baseline_capacity":
                        rank.pools[0].total_blocks = 301
                    elif fault == "baseline_watermark":
                        rank.groups[0].device_watermark.low_ratio = 0.7
                    elif fault == "baseline_host_watermark":
                        rank.groups[0].host_watermark.low_ratio = 0.7
                    elif fault == "baseline_pool_geometry":
                        rank.pools[0].block_size_bytes = 8192
                    elif fault == "baseline_owner":
                        rank.ownership.owner_world_rank.value = 1
                    elif fault == "baseline_participants":
                        del rank.ownership.participant_world_ranks[:]
                        rank.ownership.participant_world_ranks.extend([1, 0])
                    elif fault == "baseline_topology":
                        rank.pools[0].pool_id = "changed-gpu"
                        rank.groups[0].device_pool_ids[0] = "changed-gpu"
                    elif fault == "baseline_dynamic_occupancy":
                        rank.pools[0].free_blocks = 20
                        rank.pools[0].available_blocks = 30
            return response

        def probe():
            self.assertEqual(observations, ["startup"])
            self.assertIsNotNone(runner._pressure_seed_plan)
            self.assertEqual(runner._continuous_seed_ids, [])
            observations.append("probe")

        def control(request, timeout):
            response = pb.PressureSessionControlResponsePB(schema_version=1)
            for rank, instance in ((0, "owner-instance"), (1, "participant-instance")):
                response.ranks.add(
                    world_rank=rank,
                    service_instance=instance,
                    applied_monotonic_ns=int(clock() * 1e9),
                )
            return response

        def post(stub, ids, timeout, request_id, *, register_call, before_dispatch):
            self.assertIs(sessions[0].seed_plan, runner._pressure_seed_plan)
            self.assertEqual(observations[:2], ["startup", "probe"])
            identity = before_dispatch()
            submissions.append((identity, ids))
            return dict(success=True, input_len=len(ids), output_len=1)

        def session(*args, **kwargs):
            instance = original(
                *args,
                **kwargs,
                clock=clock,
                sleep=clock.sleep,
                executor_factory=InlineExecutor,
            )
            sessions.append(instance)
            self.assertIs(instance.startup_targets, kwargs["startup_targets"])
            return instance

        runner._probe_reuse_granularity = probe
        with patch.object(
            rpc,
            "RpcServiceStub",
            return_value=SimpleNamespace(
                GetCachePressure=poll, ControlPressureDiagnostics=control
            ),
        ), patch.object(cp, "PressureSession", session), patch(
            "time.monotonic", clock
        ), patch(
            "time.sleep", clock.sleep
        ), patch(
            "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill_ids",
            side_effect=post,
        ):
            result = runner._run_full_pressure_epoch("epoch")
        return result, submissions, sessions

    def test_direct_runner_rejects_malformed_smoke_marker_before_grid_combination(self):
        for value in (0, "", None, [], {}, 1, "false"):
            for smoke_grid in (False, True):
                with self.subTest(
                    value=value, smoke_grid=smoke_grid
                ), tempfile.TemporaryDirectory() as directory:
                    options = dict(profile={"cache_grid": {"explicit_smoke": value}})
                    if smoke_grid:
                        options["grid_metadata"] = dict(
                            kind="explicit_cache_grid_smoke",
                            generator=dict(cache_alignment=4096),
                        )
                    with self.assertRaisesRegex(ValueError, "explicit_smoke.*boolean"):
                        self._startup_runner(directory, **options)

    def test_explicit_smoke_remains_nonformal_through_runtime_artifacts(self):
        for source in ("profile", "grid"):
            with self.subTest(
                source=source
            ), tempfile.TemporaryDirectory() as directory:
                options = (
                    dict(profile={"cache_grid": {"explicit_smoke": True}})
                    if source == "profile"
                    else dict(
                        grid_metadata={
                            "kind": "explicit_cache_grid_smoke",
                            "generator": {"cache_alignment": 4096},
                        }
                    )
                )
                runner = self._startup_runner(directory, **options)
                self._run_startup_epoch(runner)
                root = runner._pressure_epoch_dir / "pressure"
                for name in ("manifest.json", "summary.json"):
                    artifact = json.loads((root / name).read_text())
                    self.assertTrue(artifact["seed_plan"]["explicit_smoke"])
                    self.assertEqual(
                        artifact["seed_plan"]["nonformal_reasons"], ["explicit_smoke"]
                    )
                    self.assertFalse(artifact["validity"]["formal_config_valid"])
                    self.assertFalse(artifact["validity"]["formal_acceptance_valid"])
                    self.assertTrue(artifact["validity"]["execution_valid"])
                    self.assertEqual(
                        artifact["classification"]["nonformal_reasons"],
                        ["explicit_smoke"],
                    )

    def test_startup_plan_is_frozen_before_requests_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._startup_runner(directory)
            self.assertIsNone(runner.pressure_seed_count)
            result, submissions, sessions = self._run_startup_epoch(runner)
            self.assertEqual(result, [])
            self.assertTrue(submissions)
            manifest = json.loads(
                (runner._pressure_epoch_dir / "pressure/manifest.json").read_text()
            )
            plan = manifest["seed_plan"]
            self.assertEqual(plan["source"], "automatic")
            self.assertEqual(plan["seed_count"], 2)
            self.assertEqual(plan["seed_input_tokens"], 135168)
            self.assertEqual(plan["fill_input_tokens"], 102400)
            self.assertTrue(plan["formal_config_valid"])
            self.assertEqual(len(plan["groups"]), 1)
            group = plan["groups"][0]
            self.assertEqual(
                group["participants"],
                [[0, "owner-instance"], [1, "participant-instance"]],
            )
            self.assertEqual(group["host_pool"]["available_blocks"], 1024)
            lower = (
                min(
                    p["capacity_blocks"] // p["seed_blocks"]
                    for p in group["device_pools"]
                )
                + 1
            )
            upper = (
                min(p["low_blocks"] // p["seed_blocks"] for p in group["device_pools"])
                + group["host_pool"]["available_blocks"]
                // group["host_pool"]["seed_blocks"]
            )
            self.assertEqual((lower, upper), (2, 4))
            self.assertEqual(plan["seed_count"], lower)
            self.assertEqual(manifest["frozen"]["seed_plan"], plan)
            self.assertEqual(
                plan, json.loads(json.dumps(asdict(runner._pressure_seed_plan)))
            )
            raw = json.loads(
                (runner._pressure_epoch_dir / "startup_pressure.json").read_text()
            )
            self.assertEqual(manifest["startup_pressure"], raw)
            self.assertEqual(sessions[0].limits.concurrency_for(135168), 1)
            self.assertEqual(sessions[0].sender.limit, 16)
            seeds = [ids for identity, ids in submissions if ":seed:" in identity]
            self.assertEqual([len(ids) for ids in seeds], [135168, 135168])
            fills = [ids for identity, ids in submissions if ":fill:" in identity]
            self.assertTrue(fills)
            self.assertTrue(all(len(ids) == 102400 for ids in fills))
            formal = [
                ids
                for identity, ids in submissions
                if ":steady:" in identity or ":measurement:" in identity
            ]
            self.assertTrue(formal)
            self.assertTrue(
                all(
                    any(ids[:131072] == seed[:131072] for seed in seeds)
                    for ids in formal
                )
            )
            self.assertTrue(
                all(
                    any(
                        ids[131072] != seed[131072]
                        for seed in seeds
                        if ids[:131072] == seed[:131072]
                    )
                    for ids in formal
                )
            )

    def test_startup_snapshot_retries_transient_rank_connection_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._startup_runner(directory)
            result, submissions, _ = self._run_startup_epoch(
                runner, "startup_connect_once"
            )

            self.assertEqual(result, [])
            self.assertTrue(submissions)
            attempts = json.loads(
                (
                    runner._pressure_epoch_dir / "startup_pressure_attempts.json"
                ).read_text()
            )
            self.assertEqual(len(attempts), 2)
            self.assertEqual(
                attempts[0]["response"]["ranks"][1]["error"], "Connect Failed"
            )
            self.assertNotIn("error", attempts[1]["response"]["ranks"][1])

    def test_startup_snapshot_first_success_records_one_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._startup_runner(directory)
            result, submissions, _ = self._run_startup_epoch(runner)

            self.assertEqual(result, [])
            self.assertTrue(submissions)
            attempts = json.loads(
                (
                    runner._pressure_epoch_dir / "startup_pressure_attempts.json"
                ).read_text()
            )
            self.assertEqual(len(attempts), 1)

    def test_startup_snapshot_persistent_connection_failure_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._startup_runner(directory)
            submissions = []

            with self.assertRaisesRegex(ValueError, "Connect Failed"):
                self._run_startup_epoch(
                    runner, "startup_connect_always", submissions=submissions
                )

            self.assertEqual(submissions, [])
            attempts = json.loads(
                (
                    runner._pressure_epoch_dir / "startup_pressure_attempts.json"
                ).read_text()
            )
            self.assertGreater(len(attempts), 1)
            self.assertLessEqual(len(attempts), 42)

    def test_startup_snapshot_nontransport_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._startup_runner(directory)
            submissions = []

            with self.assertRaisesRegex(ValueError, "unavailable"):
                self._run_startup_epoch(
                    runner, "startup_nontransport", submissions=submissions
                )

            self.assertEqual(submissions, [])
            attempts = json.loads(
                (
                    runner._pressure_epoch_dir / "startup_pressure_attempts.json"
                ).read_text()
            )
            self.assertEqual(len(attempts), 1)

    def test_startup_snapshot_malformed_response_is_not_masked_by_rank_error(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._startup_runner(directory)
            submissions = []

            with self.assertRaisesRegex(ValueError, "unsupported pressure schema"):
                self._run_startup_epoch(
                    runner, "startup_malformed_connect", submissions=submissions
                )

            self.assertEqual(submissions, [])
            attempts = json.loads(
                (
                    runner._pressure_epoch_dir / "startup_pressure_attempts.json"
                ).read_text()
            )
            self.assertEqual(len(attempts), 1)

    def test_manual_startup_plan_records_nonformal_count(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._startup_runner(directory, pressure_seed_count=1)
            self._run_startup_epoch(runner)
            manifest = json.loads(
                (runner._pressure_epoch_dir / "pressure/manifest.json").read_text()
            )
            self.assertEqual(manifest["seed_plan"]["source"], "manual")
            self.assertEqual(manifest["seed_plan"]["seed_count"], 1)
            self.assertFalse(manifest["seed_plan"]["formal_config_valid"])
            self.assertEqual(
                manifest["seed_plan"]["nonformal_reasons"], ["manual_seed_count"]
            )

    def test_small_manual_grid_still_uses_frozen_32k_seed_span(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._startup_runner(directory, pressure_seed_count=1)
            runner.cases[0].update(input_len=20480, cache_len=16384)
            _, submissions, _ = self._run_startup_epoch(runner)
            self.assertEqual(runner._pressure_seed_plan.seed_input_tokens, 36864)
            self.assertEqual(
                runner._pressure_seed_plan.nonformal_reasons,
                ("manual_seed_count", "grid_reuse_below_32768"),
            )
            seeds = [ids for identity, ids in submissions if ":seed:" in identity]
            self.assertEqual([len(ids) for ids in seeds], [36864])
            formal = [
                ids
                for identity, ids in submissions
                if ":steady:" in identity or ":measurement:" in identity
            ]
            self.assertTrue(formal)
            self.assertTrue(
                all(
                    len(ids) == 20480
                    and ids[:16384] == seeds[0][:16384]
                    and ids[16384] != seeds[0][16384]
                    for ids in formal
                )
            )

    def test_aligned_fill_is_never_shortened_to_configured_target(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._startup_runner(
                directory, grid_metadata={"generator": {"cache_alignment": 65536}}
            )
            _, submissions, sessions = self._run_startup_epoch(runner)
            self.assertEqual(runner.pressure_config.fill_input_tokens, 102400)
            self.assertEqual(sessions[0].config.fill_input_tokens, 131072)
            self.assertEqual(sessions[0].seed_plan.fill_input_tokens, 131072)
            fills = [ids for identity, ids in submissions if ":fill:" in identity]
            self.assertTrue(fills)
            self.assertTrue(all(len(ids) == 131072 for ids in fills))

    def test_each_epoch_discards_previous_live_plan_before_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._startup_runner(directory)
            self._run_startup_epoch(runner)
            previous = runner._pressure_seed_plan
            runner._pressure_epoch_dir = Path(directory) / "next-epoch"
            runner._pressure_epoch_dir.mkdir()
            self._run_startup_epoch(runner)
            self.assertIsNot(runner._pressure_seed_plan, previous)

    def test_changed_first_session_inventory_submits_no_pressure_requests(self):
        for fault in (
            "baseline_geometry",
            "baseline_capacity",
            "baseline_watermark",
            "baseline_host_watermark",
            "baseline_pool_geometry",
            "baseline_owner",
            "baseline_participants",
            "baseline_topology",
        ):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                runner = self._startup_runner(directory)
                submissions = []
                with self.assertRaisesRegex(RuntimeError, "full pressure failed"):
                    self._run_startup_epoch(runner, fault, submissions)
                self.assertEqual(len(submissions), 0)
                self.assertEqual(runner._pressure_seed_plan.groups[0].block_tokens, 512)
                root = runner._pressure_epoch_dir / "pressure"
                self.assertFalse((root / "requests.jsonl").exists())
                events = [
                    json.loads(line)
                    for line in (root / "steps.jsonl").read_text().splitlines()
                ]
                self.assertTrue(
                    any(
                        event["event"] == "error"
                        and "discontinuity" in event.get("error", "")
                        for event in events
                    )
                )

    def test_first_session_inventory_allows_dynamic_free_and_available_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._startup_runner(directory)
            _, submissions, _ = self._run_startup_epoch(
                runner, "baseline_dynamic_occupancy"
            )
            self.assertTrue(submissions)

    def test_invalid_startup_evidence_fails_before_any_submission(self):
        for fault, reason in (
            ("missing_full", "FULL"),
            ("missing_owner_pool", "pool"),
            ("available_overflow", "occupancy"),
            ("infeasible", "owner_rank=0.*minimum=2 maximum=0"),
        ):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                runner = self._startup_runner(directory)
                with self.assertRaisesRegex(ValueError, reason):
                    self._run_startup_epoch(runner, fault)
                self.assertEqual(runner._continuous_seed_ids, [])
                self.assertFalse(
                    (runner._pressure_epoch_dir / "pressure/requests.jsonl").exists()
                )
                self.assertTrue(
                    (runner._pressure_epoch_dir / "startup_pressure.json").exists()
                )

    def test_formal_grid_below_32k_fails_before_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._startup_runner(directory)
            runner.cases = [
                dict(case_id=8, input_len=20480, cache_len=16384, batch_size=1)
            ]
            with self.assertRaisesRegex(ValueError, "32768"):
                self._run_startup_epoch(runner)
            self.assertEqual(runner._continuous_seed_ids, [])

    def test_request_shape_admission_names_each_violated_constraint(self):
        for shape, field, value, reason in (
            ("fill", "model_tokens", 102400, "model"),
            ("fill", "batch_tokens", 102400, "batch.*token"),
            ("fill", "available", 200, "owner=0 group=0 pool=gpu.*KV"),
            ("seed", "model_tokens", 135168, "model"),
            ("formal.*case_id=7", "model_tokens", 150000, "model"),
        ):
            with self.subTest(
                shape=shape, field=field
            ), tempfile.TemporaryDirectory() as directory:
                runner = self._startup_runner(directory)
                if field == "available":
                    original = self._startup_pressure

                    def pressure(*args, **kwargs):
                        response = original(*args, **kwargs)
                        response.ranks[0].pools[0].available_blocks = value
                        return response

                    with patch.object(self, "_startup_pressure", side_effect=pressure):
                        with self.assertRaisesRegex(ValueError, shape + ".*" + reason):
                            self._run_startup_epoch(runner)
                else:
                    runner.run_config["full_pressure_startup"][field] = value
                    if shape.startswith("formal"):
                        runner.cases[0]["input_len"] = 180224
                    with self.assertRaisesRegex(ValueError, shape + ".*" + reason):
                        self._run_startup_epoch(runner)
                self.assertEqual(runner._continuous_seed_ids, [])
                self.assertFalse(
                    (runner._pressure_epoch_dir / "pressure/requests.jsonl").exists()
                )

    def test_preload_constructor_requires_fifo_and_positive_integer_smoke_count(self):
        for changes in (
            dict(pressure_scheduler="batch_decode"),
            dict(pressure_batch_size=2),
            dict(pressure_seed_count=0),
            dict(pressure_seed_count=True),
            dict(pressure_seed_count=1.5),
        ):
            with self.subTest(
                changes=changes
            ), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self._startup_runner(directory, **changes)

    def test_latest_epoch_is_atomic_and_retains_preflight_failure(self):
        from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
            CacheGridRunner,
        )

        for fault in ("checkpoint", "rpc"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                self._assert_latest_epoch_preflight_failure(
                    CacheGridRunner, directory, fault
                )

    def _assert_latest_epoch_preflight_failure(self, runner_type, directory, fault):
        from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2_grpc as rpc

        with patch.object(rpc, "RpcServiceStub") as stub:
            runner = object.__new__(runner_type)
            runner.result_dir = Path(directory)
            runner.run_config = {"full_pressure_startup": {"target_ranks": [0]}}
            runner._results = {}
            runner.port, runner.request_timeout = 123, 1
            runner.pressure_config = config()
            runner._save = Mock(
                side_effect=(
                    ValueError("preflight failure") if fault == "checkpoint" else None
                )
            )
            stub.return_value.GetCachePressure.side_effect = ValueError(
                "preflight failure"
            )
            latest = runner.result_dir / "pressure_latest.json"
            latest.write_text('{"epoch_id":"previous"}')
            with patch("os.replace", wraps=os.replace) as replace_file:
                with self.assertRaisesRegex(ValueError, "preflight failure"):
                    runner._run_full_pressure_public()
            pointer = json.loads(latest.read_text())
            self.assertNotEqual(pointer["epoch_id"], "previous")
            self.assertEqual(Path(pointer["directory"]), runner._pressure_epoch_dir)
            self.assertTrue(
                any(call.args[1] == latest for call in replace_file.call_args_list)
            )
            failure = json.loads(
                (runner._pressure_epoch_dir / "preflight_failure.json").read_text()
            )
            self.assertIn("preflight failure", failure["error"])

    def test_nsys_wiring_uses_duration_and_never_starts_kineto(self):
        self._runner_case("nsys_failure")

    def test_continuous_capture_wiring_and_failure_preserve_requests_and_samples(self):
        self._runner_case("profile_failure")

    def test_scheduler_timeout_override_preserves_legacy_default(self):
        from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
            CacheGridRunner,
        )

        runner = CacheGridRunner.__new__(CacheGridRunner)
        runner.port, runner.request_timeout = 123, 100
        runner._http_session = SimpleNamespace(post=Mock())
        with patch(
            "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner.configure_scheduler"
        ) as configure:
            runner._set_prefill_batch_size(4)
            self.assertEqual(configure.call_args.kwargs["timeout"], 60)
            runner._set_prefill_batch_size(4, timeout=0.2)
            self.assertEqual(configure.call_args.kwargs["timeout"], 0.2)

    def test_real_runner_transport_and_protobuf_polling(self):
        self._runner_case()

    def test_transient_missing_or_failed_rank_remains_unknown_and_measurement_continues(
        self,
    ):
        for fault in ("transient_missing", "transient_error"):
            with self.subTest(fault=fault):
                self._runner_case(fault)

    def test_persistent_missing_rank_fails_coverage_and_final_drain(self):
        self._runner_case("persistent_missing")

    def test_poll_instance_corruption_remains_fatal(self):
        self._runner_case("poll_instance")

    def test_missing_or_failed_peer_cannot_mask_available_corruption(self):
        for fault in ("missing_corrupt", "error_corrupt", "errored_foreign_session"):
            with self.subTest(fault=fault):
                self._runner_case(fault)

    def test_empty_rank_transport_error_remains_recoverable(self):
        self._runner_case("empty_transport_error")

    def test_shared_owner_partial_pages_remain_unknown(self):
        for fault in ("transient_missing", "transient_error", "empty_transport_error"):
            with self.subTest(fault=fault):
                self._runner_case(fault, shared=True)

    def test_shared_owner_initial_error_cannot_establish_inventory(self):
        self._runner_case("initial_error", shared=True)

    def test_shared_owner_partial_pages_validate_concrete_corruption(self):
        for fault in (
            "missing_corrupt",
            "error_corrupt",
            "missing_reset",
            "missing_layout",
            "missing_scope",
        ):
            with self.subTest(fault=fault):
                self._runner_case(fault, shared=True)

    def test_measurement_loss_recovers_for_cleanup_but_fails_coverage(self):
        self._runner_case("measurement_loss")

    def test_shared_owner_real_protobuf_preserves_raw_peers_and_manifest_provenance(
        self,
    ):
        self._runner_case(shared=True)

    def test_missing_waiting_wrapper_is_unknown_activity(self):
        self._runner_case("missing_waiting")

    def test_owner_zero_presence_survives_json_and_missing_wrapper_is_rejected(self):
        from google.protobuf.json_format import MessageToDict, ParseDict

        from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb

        message = pb.CachePressureRankPB()
        message.ownership.mode = pb.PressureOwnershipPB.SHARED_TP
        message.ownership.owner_world_rank.value = 0
        encoded = MessageToDict(message, preserving_proto_field_name=True)
        self.assertIn("owner_world_rank", encoded["ownership"])
        restored = ParseDict(encoded, pb.CachePressureRankPB())
        self.assertTrue(restored.ownership.HasField("owner_world_rank"))
        del encoded["ownership"]["owner_world_rank"]
        restored = ParseDict(encoded, pb.CachePressureRankPB())
        self.assertFalse(restored.ownership.HasField("owner_world_rank"))
        self._runner_case("missing_owner", shared=True)

    def test_runner_all_rank_control_and_poll_failures_retain_evidence(self):
        for fault in (
            "missing",
            "unsupported",
            "partial",
            "duplicate",
            "restart",
            "disable",
            "poll_rank",
            "poll_clock",
        ):
            with self.subTest(fault=fault):
                self._runner_case(fault)

    def _runner_case(self, fault=None, shared=False):
        from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
        from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
            CacheGridRunner,
        )

        clock, events, polls, rpc_ids = FakeClock(), [], [], []
        controls = []
        sequence = [0]
        missing_polls = []
        corrupt_polls = []

        def control(request, timeout):
            self.assertLessEqual(timeout, 1)
            controls.append(request)
            response = pb.PressureSessionControlResponsePB(schema_version=1)
            response.ranks.add(
                world_rank=0,
                service_instance="actual-wire",
                applied_monotonic_ns=int(clock() * 1e9),
            )
            if fault != "missing":
                response.ranks.add(
                    world_rank=1,
                    service_instance="peer-wire",
                    applied_monotonic_ns=int(clock() * 1e9),
                )
            if fault == "unsupported":
                response.schema_version = 99
            elif fault == "partial":
                response.ranks[-1].error = "unsupported executor"
            elif fault == "duplicate":
                response.ranks[-1].world_rank = 0
            elif fault == "restart" and request.action != 1:
                response.ranks[-1].service_instance = "new-peer"
            elif fault == "disable" and request.action == 3:
                response.ranks[-1].error = "timeout"
            return response

        def poll(request, timeout):
            self.assertEqual(
                controls[0].action, pb.PressureSessionControlRequestPB.ENABLE
            )
            polls.append(request)
            sequence[0] += 1
            response = pb.CachePressureResponsePB(
                schema_version=1, requested_ranks=[0, 1]
            )
            rank = response.ranks.add(
                world_rank=0,
                service_instance="actual-wire",
                sequence=sequence[0],
                sampled_monotonic_ns=int(clock() * 1e9),
            )
            rank.pools.add(pool_id="gpu", tier=1, total_blocks=100, used_blocks=90)
            rank.pools.add(
                pool_id="host",
                tier=2,
                total_blocks=100,
                used_blocks=30,
                published_host_blocks=25,
            )
            group = rank.groups.add(
                group_id=0,
                group_type=1,
                device_pool_ids=["gpu"],
                watermark_trigger_count=sequence[0],
            )
            group.host_pool_id.value = "host"
            group.device_watermark.low_ratio, group.device_watermark.high_ratio = (
                0.8,
                0.9,
            )
            group.host_watermark.low_ratio, group.host_watermark.high_ratio = 0.8, 0.9
            rank.transfers.add(
                group_id=0,
                pool_id="host",
                pool_tier=2,
                operation=2,
                source_tier=1,
                target_tier=2,
                published_blocks=sequence[0],
            )
            rank.activity.running_requests.value = 1
            rank.activity.waiting_requests.value = 0
            rank.next_step_sequence = sequence[0]
            rank.step_history_available = True
            if rpc_ids:
                begin = int((clock() - 0.1) * 1e9)
                rank.steps.add(
                    session_id=request.session_id,
                    step_id=sequence[0],
                    sequence=sequence[0],
                    completed=True,
                    context_count=1,
                    preparation_begin_ns=begin,
                    forward_enter_ns=begin + 100,
                    forward_return_ns=begin + 200,
                    end_ns=begin + 200,
                    cpu_start_ns=sequence[0] * 100,
                    cpu_end_ns=sequence[0] * 100 + 10,
                    source_request_ids=[rpc_ids[-1]],
                    executed_tokens=16,
                )
                rank.steps.add(
                    session_id=request.session_id,
                    step_id=sequence[0] + 1,
                    sequence=0,
                    completed=False,
                    context_count=1,
                    wall_start_ns=int(clock() * 1e9),
                    source_request_ids=[rpc_ids[-1]],
                )
            peer = response.ranks.add()
            peer.CopyFrom(rank)
            peer.world_rank, peer.service_instance = 1, "peer-wire"
            if shared:
                for item in (rank, peer):
                    item.ownership.mode = pb.PressureOwnershipPB.SHARED_TP
                    item.ownership.owner_world_rank.value = 0
                    item.ownership.participant_world_ranks.extend([0, 1])
                    item.ownership.publication_scope = (
                        pb.PressureOwnershipPB.ALL_PARTICIPANTS
                    )
                    item.groups[0].group_tags.append("gpu-tag")
                    item.groups[0].block_tokens = 512
                    item.groups[0].payload_bytes = item.groups[0].storage_bytes = 4096
                    for pool in item.pools:
                        pool.pool_name = "gpu-tag" if pool.tier == 1 else "host-tag"
                        pool.block_size_bytes = 4096
                peer.groups[0].watermark_trigger_count = 0
                peer.ClearField("transfers")
                for pool in peer.pools:
                    pool.used_blocks = pool.published_host_blocks = 0
            if fault == "missing_waiting":
                rank.activity.ClearField("waiting_requests")
            if fault == "missing_owner":
                rank.ownership.ClearField("owner_world_rank")
            for step in peer.steps:
                del step.source_request_ids[:]
            if fault == "poll_rank":
                del response.ranks[-1]
            if fault == "poll_clock":
                response.coordinator_received_ns = 1
            if fault == "poll_instance":
                peer.service_instance = "restarted-peer"
            measuring = any(control.phase == "measurement" for control in controls)
            if fault == "initial_error":
                peer.error = "initial peer deadline"
            if (
                fault in ("missing_reset", "missing_layout", "missing_scope")
                and measuring
                and not corrupt_polls
            ):
                corrupt_polls.append(len(polls) - 1)
                if fault == "missing_reset":
                    rank.transfers[0].published_blocks = 0
                elif fault == "missing_layout":
                    rank.pools[0].total_blocks += 1
                else:
                    rank.ownership.publication_scope = 1
                del response.ranks[-1]
            if (
                fault in ("missing_corrupt", "error_corrupt")
                and measuring
                and not corrupt_polls
            ):
                corrupt_polls.append(len(polls) - 1)
                rank.pools[0].tier = 99
                if fault == "missing_corrupt":
                    del response.ranks[-1]
                else:
                    peer.error = "peer deadline"
            if fault == "errored_foreign_session" and measuring and not corrupt_polls:
                corrupt_polls.append(len(polls) - 1)
                peer.error = "peer deadline"
                peer.steps.add(session_id="foreign-session", step_id=100)
            if measuring and (
                fault == "persistent_missing"
                or (fault == "measurement_loss" and controls[-1].phase == "measurement")
                or (
                    fault
                    in ("transient_missing", "transient_error", "empty_transport_error")
                    and not missing_polls
                )
            ):
                missing_polls.append(len(polls) - 1)
                if fault == "empty_transport_error":
                    peer.Clear()
                    peer.world_rank, peer.error = 1, "temporary transport error"
                elif fault == "transient_error":
                    peer.error = "temporary peer deadline"
                else:
                    del response.ranks[-1]
            return response

        def infer(requests, **kwargs):
            request = next(requests)
            rpc_ids.append(request.id)
            outputs = [
                SimpleNamespace(name=name, datatype="INT32", shape=[1])
                for name in (
                    "prompt_token_num",
                    "generated_ids",
                    "prompt_cached_token_num",
                )
            ]
            infer = SimpleNamespace(
                parameters={},
                outputs=outputs,
                raw_output_contents=[
                    struct.pack("<i", len(request.ids)),
                    struct.pack("<i", 1),
                    struct.pack("<i", 0),
                ],
            )
            return iter(
                [
                    SimpleNamespace(
                        error_message="",
                        HasField=lambda field: True,
                        infer_response=infer,
                    )
                ]
            )

        runner = CacheGridRunner.__new__(CacheGridRunner)
        runner._grpc_stub = SimpleNamespace(ModelStreamInfer=infer)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        runner.result_dir = Path(directory.name)
        runner.pressure_seed_count = 2
        runner.cache_commit_tail_tokens = 1
        runner.pressure_random_seed = 7
        runner.pressure_client_concurrency = 2
        runner.pressure_scheduler = "fifo"
        runner.pressure_batch_size = 1
        runner.pressure_request_rate = 4
        runner.pressure_arrival = "constant"
        runner.request_timeout = 2
        runner._continuous_prompt_lock = Lock()
        runner._set_prefill_batch_size = lambda count: self.fail("FIFO batch control")
        generated = [0]

        def exact(tokens, nonce):
            generated[0] += 1
            return "", [generated[0]] * tokens

        runner.factory = SimpleNamespace(_exact_text_and_ids=exact)
        original = cp.PressureSession

        def session(*args, **kwargs):
            return original(
                *args,
                **kwargs,
                clock=clock,
                sleep=clock.sleep,
                executor_factory=InlineExecutor,
            )

        dependencies = {
            "rtp_llm.dash_sc.client": SimpleNamespace(
                build_model_infer_request=lambda **kw: SimpleNamespace(
                    id=kw["request_id"], ids=kw["input_ids"]
                )
            ),
            "rtp_llm.dash_sc.codec": SimpleNamespace(
                SamplingParams=lambda **kwargs: kwargs
            ),
        }
        stop_requests = []

        def stop_profile(request, timeout):
            stop_requests.append(request)
            response = pb.StopProfileResponsePB()
            for rank, instance in ((0, "actual-wire"), (1, "peer-wire")):
                page = response.ranks.add(
                    world_rank=rank, service_instance=instance, accepted=True
                )
                page.profile.trace_name = request.trace_name
            return response

        def capture(**kwargs):
            self.assertEqual(kwargs["ranks"], (0, 1))
            if fault == "nsys_failure":
                self.assertEqual(kwargs["duration_s"], 0.1)
                self.assertNotIn("start_capture", kwargs)
                self.assertEqual(kwargs["command"], runner._nsys_command)

            class Adapter:
                def start(self, now):
                    if fault != "nsys_failure":
                        kwargs["start_capture"]()
                        kwargs["read_status"](1)

                def pending(self, now):
                    return False

                def stop(self, reason, now):
                    if fault != "nsys_failure":
                        kwargs["stop_capture"]()

                def finish(self, coverage):
                    return dict(
                        trace_valid=False,
                        pressure_valid=True,
                        reasons=["fixture_invalid_trace"],
                    )

            return Adapter()

        if fault in ("profile_failure", "nsys_failure"):
            runner.profile_only = True
            runner.profile_backend = "nsys" if fault == "nsys_failure" else "kineto"
            runner.profile_trace_timeout = 120
            response = Mock()
            response.json.return_value = {"status": "ok"}
            runner._http_session = SimpleNamespace(post=Mock(return_value=response))
            runner.port = 123
        # Freeze a clean startup inventory before applying this session's fault.
        session_fault = fault
        fault = None
        controls.append(pb.PressureSessionControlRequestPB(action=1))
        try:
            startup_response = poll(
                pb.CachePressureRequestPB(target_ranks=[0, 1]), timeout=1
            )
        finally:
            fault = session_fault
            controls.clear()
            polls.clear()
            sequence[0] = 0
        startup_targets = select_targets(
            decode_pressure_response(
                startup_response, clock(), clock(), (0, 1)
            ).samples,
            (0, 1),
        )
        with patch.dict("sys.modules", dependencies), patch.object(
            cp, "PressureSession", session
        ), patch("time.monotonic", clock), patch(
            "rtp_llm.test.perf_test.cache_grid.runner.pressure_profile."
            + ("NsightCapture" if fault == "nsys_failure" else "KinetoCapture"),
            side_effect=capture,
        ):
            result = runner._run_full_pressure(
                [dict(input_len=16, cache_len=0, batch_size=1)],
                config(
                    steady_s=1,
                    measure_s=4 if fault == "measurement_loss" else 2,
                    sample_period_s=0.5,
                    nsys_duration_s=0.1 if fault == "nsys_failure" else None,
                ),
                cp.ExecutionLimits(128, 2, 128, 64, 16),
                SimpleNamespace(
                    GetCachePressure=poll,
                    ControlPressureDiagnostics=control,
                    StopProfile=stop_profile,
                ),
                (0, 1),
                events.append,
                seed_plan=seed_plan(1, client_inflight=2),
                startup_targets=startup_targets,
                evidence_manifest=dict(
                    control_times=dict(capacity_send_s=1),
                    epoch={"0": "actual-wire", "1": "peer-wire"},
                ),
            )
        self.assertEqual(
            controls[-1].action, pb.PressureSessionControlRequestPB.DISABLE
        )
        if fault == "nsys_failure":
            runner._http_session.post.assert_not_called()
            self.assertFalse(stop_requests)
            self.assertEqual(result["failure"], "profile_invalid")
            self.assertTrue(any(":measurement:" in identity for identity in rpc_ids))
            return
        if fault == "profile_failure":
            self.assertEqual(result["failure"], "profile_invalid")
            ack_request = runner._http_session.post.call_args.kwargs["json"]
            self.assertEqual(ack_request["num_steps"], 20)
            self.assertEqual(ack_request["start_step"], 0)
            self.assertTrue(ack_request["enable_all_rank"])
            self.assertEqual(stop_requests[0].trace_name, ack_request["trace_name"])
            summary = json.loads(
                (runner.result_dir / "pressure/summary.json").read_text()
            )
            self.assertTrue(summary["validity"]["execution_valid"])
            self.assertFalse(summary["validity"]["trace_valid"])
            self.assertTrue(summary["validity"]["pressure_valid"]["profile"])
            self.assertTrue((runner.result_dir / "pressure/requests.jsonl").read_text())
            self.assertTrue(
                (runner.result_dir / "pressure/cache_samples.jsonl").read_text()
            )
            self.assertTrue(any(":measurement:" in identity for identity in rpc_ids))
            return
        if fault in (
            "transient_missing",
            "transient_error",
            "persistent_missing",
            "empty_transport_error",
            "measurement_loss",
        ):
            self.assertTrue(missing_polls)
            summary = json.loads(
                (runner.result_dir / "pressure/summary.json").read_text()
            )
            if fault == "persistent_missing":
                self.assertEqual(result["failure"], "diagnostics_drain_failed")
                self.assertFalse(result["pressure"]["valid"])
                self.assertFalse(summary["validity"]["execution_valid"])
            else:
                self.assertIsNone(result["failure"])
                self.assertEqual(
                    result["measurement_end"] - result["measurement_start"],
                    4 if fault == "measurement_loss" else 2,
                )
                self.assertTrue(summary["validity"]["execution_valid"])
                self.assertFalse(any(e["event"] == "error" for e in events))
                if fault == "measurement_loss":
                    self.assertFalse(result["pressure"]["valid"])
                    self.assertFalse(
                        summary["validity"]["pressure_valid"]["measurement"]
                    )
                    self.assertLess(result["pressure"]["ratio"], 0.95)
                    self.assertGreater(result["pressure"]["longest_invalid_s"], 2)
                    self.assertGreater((1 - result["pressure"]["ratio"]) * 4, 2)
                    self.assertTrue(
                        (runner.result_dir / "pressure/cache_samples.jsonl")
                        .read_text()
                        .strip()
                    )
                    self.assertLessEqual(
                        clock() - result["measurement_end"], config().drain_timeout_s
                    )
            first = missing_polls[0]
            self.assertEqual(
                dict(polls[first].after_step_sequence),
                dict(polls[first + 1].after_step_sequence),
            )
            self.assertTrue(
                any(
                    e["event"] == "pressure" and e["raw"].get("sampling_error")
                    for e in events
                )
            )
            return
        if fault:
            self.assertIsNotNone(result["failure"])
            if fault in ("missing_reset", "missing_layout", "missing_scope"):
                self.assertEqual(result["failure"], "pressure_epoch_invalid")
            evidence = json.loads(
                (runner.result_dir / "pressure/summary.json").read_text()
            )
            self.assertFalse(evidence["validity"]["execution_valid"])
            self.assertTrue(
                any(
                    e["event"] == "control" and e["result"]["response"]["ranks"]
                    for e in events
                )
            )
            if fault in ("poll_rank", "poll_clock"):
                self.assertTrue(
                    any(
                        e["event"] == "pressure" and e["raw"].get("response")
                        for e in events
                    )
                )
            return
        self.assertIsNone(result["failure"])
        if shared:
            manifest = json.loads(
                (runner.result_dir / "pressure/manifest.json").read_text()
            )
            targets = manifest["frozen"]["targets"]
            self.assertEqual([target["owner_rank"] for target in targets], [0, 0])
            self.assertEqual(
                targets[1]["participants"], [[0, "actual-wire"], [1, "peer-wire"]]
            )
            self.assertEqual(targets[1]["publication_scope"], 2)
            self.assertEqual(targets[1]["owner_instance"], "actual-wire")
            raw = manifest["frozen"]["raw_baseline"]["response"]["ranks"][1]
            self.assertTrue(
                all(int(pool.get("used_blocks", 0)) == 0 for pool in raw["pools"])
            )
            self.assertTrue(result["pressure"]["valid"])
        self.assertIn("measurement", [r.phase for r in controls])
        evidence = json.loads((runner.result_dir / "pressure/summary.json").read_text())
        self.assertEqual(evidence["metrics"], [])
        self.assertGreater(evidence["window"]["measurement_requests"], 0)
        self.assertGreater(evidence["baseline_steps"]["ranks"]["0"]["eligible"], 0)
        self.assertGreater(evidence["baseline_steps"]["ranks"]["1"]["eligible"], 0)
        self.assertTrue((runner.result_dir / "pressure/requests.jsonl").exists())
        self.assertTrue(any(":fill:" in identity for identity in rpc_ids))
        self.assertTrue(any(":measurement:" in identity for identity in rpc_ids))
        self.assertTrue(all(request.max_step_records == 256 for request in polls))
        self.assertGreater(polls[-1].after_step_sequence[0], 0)
        self.assertIn(
            "rank_offset_bounds_s",
            next(e["raw"] for e in events if e["event"] == "pressure"),
        )


if __name__ == "__main__":
    unittest.main()
