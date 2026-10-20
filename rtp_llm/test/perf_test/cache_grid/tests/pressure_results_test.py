import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.cache_grid.runner.pressure_results import (
    PressureEvidence,
    PressureWriter,
    RequestWindow,
)


class PressureResultsTest(unittest.TestCase):
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
