import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.cache_grid.runner import pressure_results as results
from rtp_llm.test.perf_test.cache_grid.runner.pressure_results import (
    PressureEvidence,
    PressureWriter,
    RequestWindow,
)


class StepClassificationTest(unittest.TestCase):
    def step(self, **changes):
        return dict(
            dict(
                step_id=1,
                complete=True,
                context_batch=1,
                decode_batch=0,
                begin_ns=10,
                end_ns=20,
                forward_enter_ns=12,
                forward_return_ns=18,
                forward_enter_cpu_ns=5,
                forward_return_cpu_ns=8,
                member_phases=["measurement"],
                member_kinds=["formal"],
                session_id="s",
                instance="i",
            ),
            **changes
        )

    def test_gap_requires_adjacent_same_executor_steps(self):
        previous = self.step()
        current = self.step(
            step_id=2,
            begin_ns=30,
            end_ns=40,
            forward_enter_ns=32,
            forward_enter_cpu_ns=11,
        )
        self.assertTrue(results.step_eligible(previous, 10, 20))
        self.assertEqual(
            results.step_gap(previous, current), {"wall_ns": 14, "cpu_ns": 3}
        )
        for changes in (
            dict(step_id=3),
            dict(instance="other"),
            dict(session_id="other"),
            dict(forward_enter_cpu_ns=7),
            dict(forward_enter_ns=17),
        ):
            self.assertIsNone(results.step_gap(previous, dict(current, **changes)))

    def test_exclusion_categories_and_uncertain_boundaries(self):
        for changes, reason in (
            (dict(member_phases=["support"]), "support"),
            (
                dict(
                    member_phases=["measurement", "support"],
                    member_kinds=["formal"] * 2,
                    context_batch=2,
                ),
                "mixed_phase",
            ),
            (dict(member_kinds=["padding"]), "padding"),
            (dict(member_phases=[]), "unknown_membership"),
            (dict(decode_batch=1), "decode"),
            (dict(complete=False), "incomplete"),
            (dict(begin_ns=9), "boundary"),
        ):
            step = self.step(**changes)
            self.assertFalse(results.step_eligible(step, 10, 20))
            self.assertEqual(results.step_exclusion(step, 10, 20), reason)
        self.assertEqual(results.server_window(100, 200, (1.2, 2.8)), (99, 197))

    def test_distribution_is_bounded_and_labels_tail_heuristic(self):
        distribution = results.BoundedDistribution(capacity=16)
        for value in range(999):
            distribution.add(value)
        summary = distribution.summary()
        self.assertEqual(summary["count"], 999)
        self.assertEqual(summary["reservoir_size"], 16)
        self.assertEqual(summary["mean"], 499)
        self.assertEqual(summary["max"], 998)
        self.assertEqual(summary["p99_status"], "insufficient")
        distribution.add(999)
        self.assertEqual(distribution.summary()["p99_status"], "estimated")


class StepEvidenceTest(unittest.TestCase):
    def page(self, rank, ids=(1, 2), lost=False, source=True):
        return dict(
            rank_offset_bounds_s={str(rank): [0, 0]},
            response=dict(
                ranks=[
                    dict(
                        world_rank=rank,
                        service_instance="rank" + str(rank),
                        step_records_lost=lost,
                        step_history_available=True,
                        steps=[
                            dict(
                                session_id="s",
                                step_id=i,
                                sequence=i,
                                completed=True,
                                preparation_begin_ns=i * 100,
                                forward_enter_ns=i * 100 + 10,
                                forward_return_ns=i * 100 + 20,
                                end_ns=i * 100 + 20,
                                cpu_start_ns=i * 10,
                                cpu_end_ns=i * 10 + 5,
                                context_count=1,
                                source_request_ids=(
                                    ["req"] if source and rank == 0 else []
                                ),
                            )
                            for i in ids
                        ],
                    )
                ]
            ),
        )

    def make(self, capacity=4):
        analysis = results.StepAnalysis((0, 1), "s", capacity=capacity)
        analysis.window = (0, 10000)
        analysis.dispatch(dict(request_id="req", phase="measurement", kind="formal"))
        return analysis

    def test_delayed_root_joins_peer_by_id_and_independent_instance(self):
        analysis = self.make()
        analysis.consume(self.page(1))
        self.assertEqual(analysis.summary(final=False)["ranks"]["1"]["eligible"], 0)
        analysis.consume(self.page(0))
        summary = analysis.summary()
        for rank in ("0", "1"):
            self.assertEqual(summary["ranks"][rank]["eligible"], 2)
            self.assertEqual(summary["ranks"][rank]["gap_wall_ns"]["mean"], 90)
            self.assertEqual(summary["ranks"][rank]["gap_cpu_ns"]["mean"], 5)

    def test_loss_breaks_adjacency_unknown_membership_stays_excluded(self):
        analysis = self.make()
        analysis.consume(self.page(0, (1,)))
        analysis.consume(self.page(0, (2,), lost=True))
        analysis.consume(self.page(1, (3,), source=False))
        summary = analysis.summary()
        self.assertEqual(summary["ranks"]["0"]["gap_wall_ns"]["count"], 0)
        self.assertEqual(summary["ranks"]["1"]["exclusions"]["unknown_membership"], 1)

    def test_peer_loss_does_not_bridge_pending_pages(self):
        analysis = self.make()
        analysis.consume(self.page(1, (1,)))
        analysis.consume(self.page(1, (2,), lost=True))
        analysis.consume(self.page(0))
        self.assertEqual(analysis.summary()["ranks"]["1"]["gap_wall_ns"]["count"], 0)

    def test_sequence_gap_breaks_pending_adjacency_without_loss_flag(self):
        analysis = self.make()
        analysis.consume(self.page(1, (1,)))
        page = self.page(0)
        peer = self.page(1, (2,))["response"]["ranks"][0]
        peer["steps"][0]["sequence"] = 3
        page["response"]["ranks"].append(peer)
        page["rank_offset_bounds_s"]["1"] = [0, 0]
        analysis.consume(page)
        self.assertEqual(analysis.summary()["ranks"]["1"]["gap_wall_ns"]["count"], 0)

    def test_active_zero_cursor_does_not_hide_completed_and_restart_excludes(self):
        analysis = self.make()
        page = self.page(0, (1,))
        active = dict(
            page["response"]["ranks"][0]["steps"][0],
            sequence=0,
            step_id=2,
            completed=False,
        )
        page["response"]["ranks"][0]["steps"].append(active)
        analysis.consume(page)
        self.assertEqual(analysis.last_sequence[0], 1)
        analysis.consume(self.page(0, (2,)))
        page = self.page(0, (3,))
        page["response"]["ranks"][0]["service_instance"] = "restart"
        analysis.consume(page)
        summary = analysis.summary()["ranks"]["0"]
        self.assertEqual(summary["eligible"], 2)
        self.assertEqual(summary["exclusions"]["unknown_membership"], 1)

    def test_clock_uncertainty_rejects_boundary_without_subtracting_cpu_from_wall(self):
        analysis = self.make()
        analysis.window = (100, 220)
        page = self.page(0)
        page["rank_offset_bounds_s"] = {"0": [-1e-9, 1e-9]}
        analysis.consume(page)
        self.assertEqual(analysis.summary()["ranks"]["0"]["exclusions"]["boundary"], 2)
        analysis = self.make()
        page = self.page(0)
        page["response"]["ranks"][0]["steps"][1]["cpu_start_ns"] = 1000
        page["response"]["ranks"][0]["steps"][1]["cpu_end_ns"] = 1005
        analysis.consume(page)
        summary = analysis.summary()["ranks"]["0"]
        self.assertEqual(summary["gap_wall_ns"]["mean"], 90)
        self.assertEqual(summary["gap_cpu_ns"]["mean"], 985)

    def test_state_is_bounded_and_raw_precedes_analysis(self):
        analysis = self.make(capacity=2)
        for i in range(10):
            analysis.dispatch(dict(request_id=str(i), phase="support", kind="formal"))
            analysis.consume(self.page(1, (i + 1,)))
        self.assertLessEqual(len(analysis.requests), 2)
        self.assertLessEqual(sum(len(v) for v in analysis.pending.values()), 4)
        self.assertGreater(analysis.summary()["state_evictions"], 0)
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {})
            evidence = PressureEvidence(writer, target_ranks=(0, 1), session_id="s")
            evidence(dict(event="pressure", raw=self.page(0)))
            writer.close()
            raw = json.loads(
                (Path(directory) / "pressure/cache_samples.jsonl").read_text()
            )
            self.assertEqual(
                raw["raw"]["response"]["ranks"][0]["steps"][0]["source_request_ids"],
                ["req"],
            )


class PressureResultsTest(unittest.TestCase):
    def test_profile_first_failure_does_not_mask_execution_evidence(self):
        for terminal in (
            "failed",
            "cancelled",
            "unfinished",
            "clean",
            "error",
            "shape",
        ):
            with self.subTest(
                terminal=terminal
            ), tempfile.TemporaryDirectory() as directory:
                writer = PressureWriter(Path(directory), {})
                sink = PressureEvidence(writer, profile_requested=True)
                request = dict(
                    event="dispatch",
                    job_id=1,
                    request_id="a",
                    phase="support",
                    dispatched_s=71,
                    input_len=16,
                )
                sink(request)
                if terminal in {"failed", "clean", "error", "shape"}:
                    sink(
                        dict(
                            event="completion",
                            job_id=1,
                            completed_s=72,
                            result=dict(
                                success=terminal != "failed",
                                input_len=16,
                                output_len=2 if terminal == "shape" else 1,
                            ),
                        )
                    )
                elif terminal == "cancelled":
                    sink(dict(event="cancellation", job_id=1, at_s=72, accepted=True))
                if terminal == "error":
                    sink(dict(event="error", reason="padding_failed"))
                sink(dict(event="finished", failure="profile_timeout"))
                writer.close()
                summary = json.loads(
                    (Path(directory) / "pressure/summary.json").read_text()
                )
                self.assertEqual(summary["result"]["failure"], "profile_timeout")
                self.assertEqual(
                    summary["validity"]["execution_valid"], terminal == "clean"
                )

    def test_writer_rejects_nonfinite_and_unknown_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {})
            with self.assertRaises(ValueError):
                writer.append("requests", dict(latency=float("nan")))
            with self.assertRaises(ValueError):
                writer.append("other", {})
            writer.close()

    def test_unfinished_cohort_cancel_is_not_server_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {})
            sink = PressureEvidence(writer)
            sink(dict(event="frozen", session_id="s", config=dict(measure_s=60)))
            sink(dict(event="phase", phase="measurement", at_s=10))
            sink(
                dict(
                    event="dispatch",
                    job_id=1,
                    request_id="a",
                    phase="measurement",
                    kind="formal",
                    dispatched_s=11,
                )
            )
            sink(dict(event="cancellation", job_id=1, at_s=72, accepted=True))
            sink(dict(event="finished", failure="drain_timeout"))
            writer.close()
            root = Path(directory) / "pressure"
            row = json.loads((root / "requests.jsonl").read_text())
            self.assertTrue(row["transport_cancel_accepted"])
            self.assertIsNone(row["completed_s"])
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["window"]["measurement_requests"], 1)
            self.assertEqual(summary["window"]["request_throughput"], 0)
            self.assertFalse(summary["validity"]["execution_valid"])

    def test_late_completion_stays_in_cohort_not_throughput(self):
        window = RequestWindow(10, 70)
        window.observe(
            dict(
                phase="measurement",
                kind="formal",
                request_id="a",
                dispatch_s=69,
                completed_s=71,
                success=True,
            )
        )
        window.observe(
            dict(
                phase="support",
                kind="formal",
                request_id="b",
                dispatch_s=70,
                completed_s=71,
                success=True,
            )
        )
        summary = window.summary()
        self.assertEqual(summary["measurement_requests"], 1)
        self.assertEqual(summary["crossing_requests"], 1)
        self.assertEqual(summary["in_window_completions"], 0)
        self.assertEqual(summary["request_throughput"], 0)

    def test_partial_evidence_survives_close_without_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {"session_id": "s"})
            writer.append("cache_samples", {"rank": 0, "sequence": 1})
            writer.close()
            rows = (Path(directory) / "pressure/cache_samples.jsonl").read_text()
            self.assertEqual(json.loads(rows)["sequence"], 1)

    def test_edges_failures_cancellation_and_padding(self):
        window = RequestWindow(10, 70)
        base = dict(
            phase="measurement",
            kind="formal",
            dispatch_s=10,
            completed_s=10,
            success=False,
        )
        window.observe(base)
        window.observe(dict(base, completed_s=None, terminal="cancellation"))
        window.observe(dict(base, phase="padding", padding_for="seed"))
        window.observe(dict(base, phase="padding", padding_for="fill"))
        with self.assertRaises(ValueError):
            window.observe(dict(base, dispatch_s=70))
        self.assertEqual(window.summary()["measurement_requests"], 2)
        self.assertEqual(window.summary()["in_window_completions"], 1)
        self.assertEqual(window.summary()["in_window_successes"], 0)
        self.assertEqual(window.summary()["cancelled_requests"], 1)

    def test_cancellation_then_completion_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {})
            sink = PressureEvidence(writer)
            sink(dict(event="frozen", session_id="s", config=dict(measure_s=60)))
            sink(dict(event="phase", phase="measurement", at_s=10))
            request = dict(
                job_id=1,
                request_id="a",
                phase="measurement",
                dispatched_s=11,
            )
            sink(dict(request, event="dispatch"))
            sink(dict(event="cancellation", job_id=1, at_s=12, accepted=True))
            sink(
                dict(
                    request,
                    event="completion",
                    completed_s=13,
                    result=dict(success=True, reuse_len=4),
                )
            )
            sink(dict(event="finished", failure=None, pressure=dict(valid=True)))
            writer.close()
            root = Path(directory) / "pressure"
            rows = [
                json.loads(s)
                for s in (root / "requests.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["terminal"], "completion")
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["metrics"], [])
            self.assertEqual(summary["window"]["measurement_requests"], 1)
            self.assertEqual(summary["validity"]["trace_valid"], None)
            self.assertEqual(sink.pending, {})

    def test_abort_retains_terminal_cancellation_without_success(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {})
            sink = PressureEvidence(writer)
            sink(
                dict(
                    event="dispatch",
                    job_id=1,
                    request_id="a",
                    phase="steady",
                    kind="formal",
                    dispatched_s=1,
                )
            )
            sink.abort("failed")
            writer.close()
            root = Path(directory) / "pressure"
            row = json.loads((root / "requests.jsonl").read_text())
            self.assertFalse(row["success"])
            self.assertIsNone(row["completed_s"])
            self.assertFalse(
                json.loads((root / "manifest.json").read_text())["complete"]
            )


if __name__ == "__main__":
    unittest.main()
