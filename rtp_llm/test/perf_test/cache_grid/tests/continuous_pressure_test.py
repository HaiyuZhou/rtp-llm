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
    resolve_pressure_config,
)
from rtp_llm.test.perf_test.cache_grid.runner.pressure_seed_plan import SeedPlan


def config(**changes):
    values = dict(
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
        self.assertEqual(cp.padding_needed(6, "batch_decode", 4), 2)
        self.assertEqual(cp.padding_needed(6, "fifo", 4), 0)

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


class LifecycleTest(unittest.TestCase):
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
        fail_padding=False,
        poll_transform=None,
        payload_hook=None,
        seed_count=0,
        rate=3,
        control=None,
        diagnostics_control=None
    ):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            GroupSample,
            PoolSample,
            RankSample,
        )

        clock, events, batches = FakeClock(), [], []
        counter = [0]

        def poll(timeout):
            counter[0] += 1
            group = GroupSample(
                0, "FULL", ("gpu",), "host", int(counter[0] > 1), int(counter[0] > 1)
            )
            sample = RankSample(
                0,
                "service",
                counter[0],
                clock(),
                clock(),
                clock(),
                (
                    PoolSample("gpu", 100, 90, 0, 0.8, 0.9),
                    PoolSample("host", 100, 30, 0, 0.8, 0.9, 25 if valid else 0),
                ),
                (group,),
            )
            evidence = cp.PressurePoll((sample,), {}, ())
            return (
                poll_transform(evidence, counter[0], clock)
                if poll_transform
                else evidence
            )

        def request(payload, register, before):
            identity = before(payload)
            return dict(
                success=not (fail_padding and ":padding:" in identity),
                input_len=16,
                output_len=1,
            )

        def payload(kind, sequence):
            if payload_hook:
                payload_hook(kind, sequence, clock)
            return {"input_ids": [1] * 16, "case": "cold"}

        state = cp.PressureSession(
            cfg or config(measure_s=3, steady_s=1, sample_period_s=0.5),
            session_id="session",
            limits=cp.ExecutionLimits(128, 8, 256, 64, 16),
            concurrency=4,
            scheduler=scheduler,
            batch_size=3 if scheduler == "batch_decode" else 1,
            max_input_tokens=16,
            seed_plan=seed_plan(seed_count),
            make_payload=payload,
            request=request,
            poll=poll,
            sink=events.append,
            set_batch=control or (lambda count, **kwargs: batches.append(count)),
            controller=controller,
            diagnostics_control=diagnostics_control,
            clock=clock,
            sleep=clock.sleep,
            executor_factory=InlineExecutor,
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

    def test_fixed_batch_tail_is_padded_and_failure_recorded(self):
        result, events, batches, _ = self.run_session(
            scheduler="batch_decode", fail_padding=True
        )
        formal = [
            e
            for e in events
            if e["event"] == "dispatch"
            and e["phase"] in ("steady", "measurement", "support", "padding")
        ]
        self.assertEqual(len(formal) % 3, 0)
        self.assertIn(3, batches)
        self.assertTrue(any(e["phase"] == "padding" for e in formal))
        self.assertEqual(result["failure"], "padding_failed")

    def test_fill_time_and_no_progress_budgets(self):
        for settings in (dict(fill_timeout_s=1), dict(fill_no_progress_s=1)):
            with self.subTest(settings=settings):
                result, _, _, state = self.run_session(
                    valid=False, cfg=config(**settings)
                )
                self.assertEqual(result["failure"], "pressure_not_reached")
                self.assertLess(state.clock(), 4)

    def test_seed_batches_are_bounded_and_baseline_precedes_sends(self):
        result, events, batches, _ = self.run_session(
            seed_count=7, scheduler="batch_decode"
        )
        self.assertIsNone(result["failure"])
        self.assertEqual(events[0]["event"], "frozen")
        self.assertEqual(batches[:2], [4, 3])
        self.assertEqual(
            sum(e["event"] == "dispatch" and e["phase"] == "seed" for e in events), 7
        )

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

    def test_padding_preparation_cannot_silently_cross_drain_deadline(self):
        def delay(kind, sequence, clock):
            if kind == "padding":
                clock.now += 100

        result, _, _, _ = self.run_session(scheduler="batch_decode", payload_hook=delay)
        self.assertIsNotNone(result["failure"])

    def test_blocked_scheduler_control_obeys_remaining_fill_budget(self):
        budgets = []

        def control(count, *, timeout):
            budgets.append(timeout)
            raise TimeoutError("bounded scheduler acknowledgement")

        result, events, _, _ = self.run_session(
            scheduler="batch_decode", cfg=config(fill_timeout_s=0.2), control=control
        )
        self.assertEqual(result["failure"], "supervisor_failed")
        self.assertEqual(len(budgets), 1)
        self.assertGreater(budgets[0], 0)
        self.assertLessEqual(budgets[0], 0.2)
        self.assertFalse(any(e["event"] == "dispatch" for e in events))

    def test_blocked_rpc_is_cancelled_at_drain_without_boundary_drain(self):
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

        session = cp.PressureSession(
            config(
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
            seed_plan=seed_plan(client_inflight=2),
            make_payload=lambda kind, i: {"input_ids": [1] * 16},
            request=request,
            poll=poll,
            sink=events.append,
            set_batch=lambda count: self.fail("FIFO control"),
        )
        began = time.monotonic()
        result = session.run()
        self.assertLess(time.monotonic() - began, 1)
        self.assertEqual(result["failure"], "drain_timeout")
        self.assertEqual(len(cancels), 2)
        self.assertTrue(all(t >= result["measurement_end"] for t in cancels))
        self.assertEqual(sum(e["event"] == "cancellation" for e in events), 2)
        self.assertTrue(
            any(e["event"] == "phase" and e["phase"] == "measurement" for e in events)
        )


from rtp_llm.test.perf_test.cache_grid.runner.continuous_pressure import (
    ArrivalClock,
    BoundedSender,
    CancellableRequestHandle,
)


class WarmupCutoffTest(unittest.TestCase):
    def run_cutoff(
        self,
        kind="fill",
        scheduler="batch_decode",
        preparation_fails=False,
        past_drain=False,
    ):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            GroupSample,
            PoolSample,
            RankSample,
        )

        clock = FakeClock()
        first_sent, crossed, batch_complete = Event(), Event(), Event()
        cancellation, second_attempted = Event(), Event()
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

        def payload(request_kind, job_id):
            self.assertEqual(request_kind, kind)
            if job_id == 1:
                if not first_sent.wait(2):
                    raise AssertionError("first request never actually dispatched")
                clock.now = 1.21
                crossed.set()
                if preparation_fails:
                    raise RuntimeError("second preparation failed")
                if past_drain and not cancellation.wait(2):
                    raise AssertionError("drain did not cancel the partial batch")
            return {"input_ids": [job_id + 1] * 16}

        def request(payload, register, before):
            try:
                identity = before(payload)
            finally:
                if payload["input_ids"][0] == 2:
                    second_attempted.set()
            dispatched.append(identity)
            if len(dispatched) == 1:
                first_sent.set()
            else:
                batch_complete.set()

            class Call:
                def cancel(self):
                    cancelled.append(identity)
                    batch_complete.set()
                    cancellation.set()
                    return True

            register(Call())
            if scheduler == "batch_decode" and not batch_complete.wait(2):
                raise AssertionError(
                    "partial scheduler batch was neither completed nor cancelled"
                )
            return dict(success=True, input_len=16, output_len=1)

        def sleep(seconds):
            # Only the second worker advances across the fill deadline. After
            # that, let cleanup progress even if the partial batch is blocked.
            if crossed.is_set():
                clock.now = round(clock.now + 0.01, 8)
            time.sleep(0.001)

        session = cp.PressureSession(
            config(fill_timeout_s=0.2, drain_timeout_s=0.2, sample_period_s=0.01),
            session_id="partial",
            limits=cp.ExecutionLimits(128, 2, 128, 64, 16),
            concurrency=2,
            scheduler=scheduler,
            batch_size=1,
            max_input_tokens=16,
            seed_plan=seed_plan(2 if kind == "seed" else 0, client_inflight=2),
            make_payload=payload,
            request=request,
            poll=poll,
            sink=events.append,
            set_batch=lambda count, **kwargs: batches.append(count),
            clock=clock,
            sleep=sleep,
        )
        result = session.run()
        if past_drain:
            self.assertTrue(second_attempted.wait(2))
        self.assertTrue(first_sent.is_set())
        self.assertTrue(crossed.is_set())
        if preparation_fails:
            self.assertIn(result["failure"], ("pressure_not_reached", "request_failed"))
        else:
            self.assertEqual(result["failure"], "pressure_not_reached")
        return events, dispatched, cancelled, batches, session

    def test_delayed_fill_preparation_completes_actual_partial_batch(self):
        events, dispatched, cancelled, batches, session = self.run_cutoff()
        self.assertEqual(batches, [2])
        self.assertEqual(dispatched, ["partial:fill:0", "partial:padding:1"])
        self.assertEqual(cancelled, [])
        self.assertEqual(session.formal_submitted, 0)
        self.assertEqual(session.padding_submitted, 0)
        padding = next(
            e for e in events if e["event"] == "dispatch" and e["phase"] == "padding"
        )
        self.assertEqual(padding["padding_for"], "fill")

    def test_delayed_seed_preparation_completes_actual_partial_batch(self):
        events, dispatched, cancelled, batches, _ = self.run_cutoff(kind="seed")
        self.assertEqual(batches, [2])
        self.assertEqual(dispatched, ["partial:seed:0", "partial:padding:1"])
        self.assertEqual(cancelled, [])

    def test_fifo_never_dispatches_warmup_padding(self):
        _, dispatched, cancelled, batches, _ = self.run_cutoff(scheduler="fifo")
        self.assertEqual(dispatched, ["partial:fill:0"])
        self.assertEqual(batches, [])
        self.assertEqual(cancelled, [])

    def test_failed_preparation_reports_incomplete_actual_tail(self):
        events, dispatched, cancelled, _, _ = self.run_cutoff(preparation_fails=True)
        self.assertEqual(dispatched, ["partial:fill:0"])
        self.assertEqual(cancelled, ["partial:fill:0"])
        failure = next(
            e
            for e in events
            if e["event"] == "error" and e.get("reason") == "padding_failed"
        )
        self.assertEqual(
            (failure["stage"], failure["submitted"], failure["batch_size"]),
            ("fill", 1, 2),
        )

    def test_warmup_tail_cannot_dispatch_after_drain_deadline(self):
        events, dispatched, cancelled, _, _ = self.run_cutoff(past_drain=True)
        self.assertEqual(dispatched, ["partial:fill:0"])
        self.assertEqual(cancelled, ["partial:fill:0"])
        self.assertTrue(any(e.get("reason") == "padding_failed" for e in events))


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
            return iter(())

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
        self.assertTrue(result["success"])
        self.assertEqual(stub.ModelStreamInfer.call_args.kwargs["timeout"], 17)


class RunnerIntegrationTest(unittest.TestCase):
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

    def _run_startup_epoch(self, runner, fault=None):
        from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
        from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2_grpc as rpc

        clock, observations, submissions, sessions = FakeClock(), [], [], []
        original = cp.PressureSession

        def poll(request, timeout):
            startup = not request.session_id
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
            "rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner._post_prefill_ids",
            side_effect=post,
        ):
            result = runner._run_full_pressure_epoch("epoch")
        return result, submissions, sessions

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
                seed_plan=seed_plan(client_inflight=2),
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
