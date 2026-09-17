import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm.test.perf_test.batch_replay import main as replay
from rtp_llm.test.perf_test.batch_replay import make_plan
from rtp_llm.test.perf_test.batch_trace_analyze import (
    batch_request_timings,
    compare_kernels,
    correlate_trace,
    execution_timings,
)
from rtp_llm.test.perf_test.batch_trace_analyze import main as analyze
from rtp_llm.test.perf_test.batch_trace_analyze import (
    read_jsonl,
    request_latencies,
    validate_batch,
)


class BatchOfflineToolsTest(unittest.TestCase):
    def test_embedded_request_timing(self):
        row = dict(
            request_id="r1",
            batch_slot=0,
            phase="prefill",
            q_tokens=4,
            kv_tokens_before=0,
            enqueue_time_unix_ns=1000,
            first_scheduled_time_unix_ns=2000,
            ttft_us=None,
        )
        first = dict(schema_version=1, execution_id=1, requests=[row])
        final = dict(first, execution_id=2, requests=[dict(row, ttft_us=3)])
        result = batch_request_timings([first, final, final])[0]
        self.assertEqual(result["engine_first_token_latency_ns"], 3000)
        self.assertEqual(result["initial_queue_latency_ns"], 1000)
        self.assertEqual(result["status"], "observed")
        self.assertIsNone(result["terminal"])
        self.assertIsNone(result["engine_request_latency_ns"])
        self.assertIsNone(
            batch_request_timings([first])[0]["engine_first_token_latency_ns"]
        )
        conflict = dict(final, requests=[dict(row, ttft_us=4)])
        self.assertEqual(
            batch_request_timings([final, conflict])[0]["status"], "partial"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                json.dumps(
                    dict(
                        delivery_policy="best_effort",
                        request_log_format="batch_embedded_v1",
                    )
                )
            )
            (root / "batches.jsonl").write_text(
                json.dumps(first) + "\n" + json.dumps(final) + "\n"
            )
            (root / "request_events.jsonl").write_text("stale legacy data\n")
            analyze(["--record-dir", str(root), "--output", str(root / "report")])
            report = json.loads((root / "report/report.json").read_text())
            self.assertEqual(len(report["requests"]), 1)
            self.assertEqual(
                report["requests"][0]["engine_first_token_latency_ns"], 3000
            )

    def test_external_alog_batch_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(
                session_id="s",
                replica_id="d0",
                dp_rank=0,
                world_rank=0,
                batch_log_location="alog_config",
                delivery_policy="best_effort",
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            batch = dict(
                session_id="s",
                replica_id="d0",
                dp_rank=0,
                schema_version=1,
                execution_id=42,
                requests=[],
            )
            external = root / "configured-batches.jsonl"
            external.write_text(
                json.dumps(dict(batch, session_id="other"))
                + "\n"
                + json.dumps(batch)
                + "\n"
            )
            analyze(
                [
                    "--record-dir",
                    str(root),
                    "--batch-file",
                    str(external),
                    "--output",
                    str(root / "report"),
                ]
            )
            report = json.loads((root / "report/report.json").read_text())
            self.assertEqual(report["batches"], [batch])

    def test_best_effort_missing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = self.events(
                ["enqueue", "first_scheduled", "first_token", "finish"]
            )
            missing = [dict(e, request_id="r2") for e in events if e["event_seq"] != 2]
            (root / "request_events.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in events + missing)
                + '{"truncated":'
            )
            manifest = dict(
                session_id="s",
                replica_id="d0",
                dp_rank=0,
                world_rank=0,
                complete=False,
                delivery_policy="best_effort",
                storage_backend="alog",
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertWarns(UserWarning):
                analyze(["--record-dir", str(root), "--output", str(root / "report")])
            report = json.loads((root / "report/report.json").read_text())
            self.assertFalse(report["recording_complete"])
            self.assertEqual(report["requests"][0]["engine_first_token_latency_ns"], 40)
            self.assertEqual(report["requests"][1]["status"], "partial")
            self.assertIsNone(report["requests"][1]["engine_first_token_latency_ns"])
            with self.assertRaises(ValueError):
                list(read_jsonl(root / "request_events.jsonl"))

    def test_legacy_incomplete_stays_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = self.events(
                ["enqueue", "first_scheduled", "first_token", "finish"]
            )
            (root / "request_events.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in events)
            )
            (root / "manifest.json").write_text('{"complete":false}')
            analyze(["--record-dir", str(root), "--output", str(root / "report")])
            report = json.loads((root / "report/report.json").read_text())
            self.assertIsNone(report["requests"][0]["engine_first_token_latency_ns"])

    def test_partial_utf8_is_missing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batches.jsonl"
            path.write_bytes(b'{}\n{"broken":"\xe4\xb8')
            with self.assertWarns(UserWarning):
                self.assertEqual(list(read_jsonl(path, skip_invalid=True)), [{}])
            with self.assertRaises(ValueError):
                list(read_jsonl(path))

    def events(self, names):
        return [
            dict(
                session_id="s",
                replica_id="d0",
                dp_rank=0,
                request_id="r1",
                owner_instance_id="o",
                clock_id="c",
                event=name,
                event_seq=i,
                timestamp_monotonic_ns=100 + i * 20,
            )
            for i, name in enumerate(names)
        ]

    def test_lifecycle(self):
        events = self.events(["enqueue", "first_scheduled", "first_token", "finish"])
        result = request_latencies(list(reversed(events)))[0]
        self.assertEqual(result["engine_request_latency_ns"], 60)
        self.assertEqual(result["initial_queue_latency_ns"], 20)
        self.assertEqual(result["engine_first_token_latency_ns"], 40)
        self.assertEqual(result["post_first_schedule_latency_ns"], 40)

    def test_cancel_before_schedule(self):
        result = request_latencies(self.events(["enqueue", "cancel"]))[0]
        self.assertEqual(result["status"], "complete")
        self.assertIsNone(result["initial_queue_latency_ns"])
        self.assertIsNone(result["post_first_schedule_latency_ns"])

    def test_missing_and_duplicate(self):
        events = self.events(["enqueue", "first_scheduled", "first_token", "finish"])
        self.assertIsNone(
            request_latencies(events[:1] + events[2:])[0]["engine_request_latency_ns"]
        )
        self.assertEqual(
            request_latencies(events + [events[-1]])[0]["status"], "partial"
        )
        events[-1]["clock_id"] = "other"
        self.assertEqual(request_latencies(events)[0]["status"], "partial")

    def test_gpu_outside_cpu_scope(self):
        trace = {
            "traceEvents": [
                dict(ph="X", name="rtp.execution(id=42)", pid=1, tid=7, ts=10, dur=10),
                dict(
                    ph="X",
                    name="cudaLaunchKernel",
                    cat="cuda_runtime",
                    pid=1,
                    tid=7,
                    ts=12,
                    dur=1,
                    args={"correlation": 99},
                ),
                dict(
                    ph="X",
                    name="kernel",
                    cat="kernel",
                    pid=0,
                    tid=3,
                    ts=100,
                    dur=4,
                    args={"correlation": 99},
                ),
                dict(
                    ph="X",
                    name="unmatched",
                    cat="kernel",
                    pid=0,
                    tid=3,
                    ts=13,
                    dur=1,
                    args={"correlation": 100},
                ),
            ]
        }
        kernels, _ = correlate_trace(trace, {"world_rank": 0})
        self.assertEqual(kernels[0]["execution_id"], 42)
        self.assertEqual(kernels[0]["duration_ns"], 4000)
        self.assertIsNone(kernels[1]["execution_id"])

    def test_overlapping_kernel_times(self):
        kernels = [
            dict(
                execution_id=42,
                world_rank=0,
                device=0,
                start_ns=start,
                duration_ns=duration,
            )
            for start, duration in ((0, 10), (5, 10), (20, 5))
        ]
        result = execution_timings(kernels)[0]
        self.assertEqual(result["kernel_sum_ns"], 25)
        self.assertEqual(result["kernel_union_ns"], 20)
        self.assertEqual(result["gpu_span_ns"], 25)

    def test_compare_does_not_align_by_call_index(self):
        original = [
            dict(execution_id=42, world_rank=0, kernel_name="k", duration_ns=10)
        ]
        replay = [
            dict(execution_id=42, world_rank=0, kernel_name="k", duration_ns=15),
            dict(execution_id=42, world_rank=0, kernel_name="extra", duration_ns=1),
        ]
        result = {r["kernel_name"]: r for r in compare_kernels(original, replay)}
        self.assertEqual(result["k"]["change_percent"], 50)
        self.assertIsNone(result["extra"]["change_percent"])

    def test_large_batch_and_replay(self):
        batch = dict(
            schema_version=1,
            execution_id=42,
            requests=[
                dict(
                    batch_slot=i,
                    phase="decode",
                    q_tokens=1,
                    kv_tokens_before=i + 1,
                    prompt_tokens=1,
                )
                for i in range(100)
            ],
        )
        self.assertEqual(len(validate_batch(batch)), 100)
        self.assertEqual(len(make_plan([batch]).splitlines()), 102)
        batch["requests"][0]["q_tokens"] = 2
        with self.assertRaises(ValueError):
            make_plan([batch])


class ReplayManifestTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.record_dir = self.root / "recordings"
        self.owner_dir = self.record_dir / "owner-input"
        self.owner_dir.mkdir(parents=True)
        self.batches = self.root / "logs" / "batch_schedule.log"
        self.batches.parent.mkdir()
        self.manifest = dict(
            schema_version=1,
            session_id="s",
            replica_id="dp0",
            dp_rank=0,
            world_rank=0,
            owner_instance_id="input",
            delivery_policy="best_effort",
            request_log_format="batch_embedded_v1",
            batch_log_location="alog_config",
        )
        self.write_manifest(self.owner_dir, self.manifest)
        # A TP peer also writes a manifest but owns no batch records.
        self.peer_dir = self.record_dir / "owner-peer"
        self.peer_manifest = dict(self.manifest, world_rank=1, owner_instance_id="peer")
        self.write_manifest(self.peer_dir, self.peer_manifest)
        self.batch = dict(
            self.manifest,
            execution_id=42,
            requests=[
                dict(
                    batch_slot=0,
                    phase="prefill",
                    q_tokens=3,
                    kv_tokens_before=0,
                    prompt_tokens=3,
                )
            ],
        )
        self.batches.write_text(json.dumps(self.batch) + "\n")
        environment = mock.patch.dict("os.environ", {"RTP_LLM_RECORD_DIR": ""})
        environment.start()
        self.addCleanup(environment.stop)

    def write_manifest(self, directory, manifest):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(json.dumps(manifest))

    def run_replay(self, name, *args):
        output = self.root / name
        with contextlib.redirect_stdout(io.StringIO()):
            replay(["--batches", str(self.batches), "--output", str(output), *args])
        return output

    def assert_source(self, output, manifest, batches):
        self.assertEqual(
            json.loads((output / "source_manifest.json").read_text()), manifest
        )
        self.assertEqual(list(read_jsonl(output / "source_batches.jsonl")), batches)
        self.assertEqual((output / "replay.plan").read_text(), make_plan(batches))

    def test_separate_manifest_locations_with_truncated_log(self):
        self.batches.write_bytes(
            b'{"broken":\n'
            + (json.dumps(self.batch) + "\n").encode()
            + b'{"truncated":"\xe4\xb8'
        )
        for name, directory in (("owner", self.owner_dir), ("root", self.record_dir)):
            with self.subTest(location=name), self.assertWarns(UserWarning):
                output = self.run_replay(name, "--record-dir", str(directory))
            self.assert_source(output, self.manifest, [self.batch])
        with mock.patch.dict("os.environ", {"RTP_LLM_RECORD_DIR": str(self.record_dir)}):
            with self.assertWarns(UserWarning):
                output = self.run_replay("environment")
        self.assert_source(output, self.manifest, [self.batch])

    def test_explicit_owner_filters_other_processes(self):
        other_manifest = dict(self.manifest, owner_instance_id="previous")
        self.write_manifest(self.record_dir / "owner-previous", other_manifest)
        other_batch = dict(self.batch, owner_instance_id="previous", execution_id=43)
        self.batches.write_text(
            json.dumps(other_batch) + "\n" + json.dumps(self.batch) + "\n"
        )
        output = self.run_replay("selected-owner", "--record-dir", str(self.owner_dir))
        self.assert_source(output, self.manifest, [self.batch])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_replay("ambiguous", "--record-dir", str(self.record_dir))
        self.assertFalse((self.root / "ambiguous").exists())
        output = self.run_replay(
            "selected-id", "--record-dir", str(self.record_dir), "--execution-ids", "42"
        )
        self.assert_source(output, self.manifest, [self.batch])

    def test_explicit_directory_overrides_adjacent_and_environment(self):
        self.write_manifest(self.batches.parent, self.peer_manifest)
        with mock.patch.dict("os.environ", {"RTP_LLM_RECORD_DIR": str(self.peer_dir)}):
            output = self.run_replay("explicit", "--record-dir", str(self.owner_dir))
        self.assert_source(output, self.manifest, [self.batch])

    def test_strict_manifest_still_rejects_corruption(self):
        self.write_manifest(self.owner_dir, dict(self.manifest, delivery_policy="strict"))
        self.batches.write_text(json.dumps(self.batch) + '\n{"truncated":')
        with self.assertRaisesRegex(ValueError, "invalid JSONL"):
            self.run_replay("strict", "--record-dir", str(self.record_dir))
        self.assertFalse((self.root / "strict").exists())

    def test_missing_or_wrong_owner_fails_before_creating_output(self):
        for name, directory in (
            ("missing", self.root / "missing-root"),
            ("wrong", self.peer_dir),
        ):
            with self.subTest(location=name):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                    SystemExit
                ):
                    self.run_replay(name, "--record-dir", str(directory))
                self.assertFalse((self.root / name).exists())

    def test_adjacent_manifest_and_bare_batches_remain_supported(self):
        output = self.run_replay("bare")
        self.assertFalse((output / "source_manifest.json").exists())
        self.assertEqual((output / "replay.plan").read_text(), make_plan([self.batch]))
        self.write_manifest(self.batches.parent, self.manifest)
        with mock.patch.dict(
            "os.environ", {"RTP_LLM_RECORD_DIR": str(self.root / "absent")}
        ):
            output = self.run_replay("adjacent")
        self.assert_source(output, self.manifest, [self.batch])


if __name__ == "__main__":
    unittest.main()
