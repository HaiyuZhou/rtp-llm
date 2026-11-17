"""Tests for decode profiling_test."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.decode.executor import DecodeCaseExecutor
from rtp_llm.test.perf_test.decode.model import DecodeProfilingConfig
from rtp_llm.test.perf_test.decode.profiling import DecodeProfiler
from rtp_llm.test.perf_test.storage import atomic_write_json
from rtp_llm.test.perf_test.tests.support.decode_profiling import (
    decode_profiling_case_payload,
    decode_profiling_exact_case,
    decode_profiling_measurement,
    decode_profiling_TOPOLOGY,
    decode_profiling_tpot_case,
)


class DecodeProfilingTest(unittest.TestCase):

    def test_tpot_profiles_only_selected_batch_after_search(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.json"
            batch_executor = Mock()
            events = []

            def measure(_case, batch):
                events.append(("measure", batch))
                return decode_profiling_measurement(batch, 8.0 if batch <= 8 else 12.0)

            batch_executor.measure.side_effect = measure

            def profile(_case, batch, **_kwargs):
                events.append(("profile", batch))
                trace.write_text(json.dumps({"traceEvents": []}))

            batch_executor.profile.side_effect = profile
            profiler = DecodeProfiler(
                batch_executor,
                run_root=Path(directory),
                log_path=None,
                sleep=lambda _seconds: None,
                artifact_discoverer=lambda _log, _offset, _name: (str(trace),),
            )
            result = DecodeCaseExecutor(
                batch_executor, topology=decode_profiling_TOPOLOGY, profiler=profiler
            ).execute(decode_profiling_case_payload(decode_profiling_tpot_case()))
            self.assertEqual([("measure", 8), ("measure", 12), ("profile", 8)], events)
            self.assertEqual("ok", result["status"])
            manifest = result["metrics"]["profiling"]
            self.assertEqual("completed", manifest["status"])
            self.assertEqual([128] * 8, manifest["input_lengths"])
            self.assertEqual(8, manifest["batch_size"])
            self.assertEqual(
                [str(Path(directory) / "traces/trace.json")], manifest["artifacts"]
            )
            self.assertIn("decode_in128_out16_bs8_", manifest["trace_name"])
            self.assertEqual(
                manifest,
                json.loads((Path(directory) / manifest["manifest_path"]).read_text()),
            )

    def test_profile_failure_does_not_change_valid_performance_status(self):
        with tempfile.TemporaryDirectory() as directory:
            batch_executor = Mock()
            batch_executor.measure.return_value = decode_profiling_measurement(4)
            batch_executor.profile.side_effect = RuntimeError(
                "profile failed\n" + "x" * 600
            )
            profiler = DecodeProfiler(
                batch_executor, run_root=Path(directory), log_path=None
            )
            result = DecodeCaseExecutor(
                batch_executor, topology=decode_profiling_TOPOLOGY, profiler=profiler
            ).execute(
                decode_profiling_case_payload(decode_profiling_exact_case(profile=True))
            )
            self.assertEqual("request_failed", result["status"])
            self.assertIn("requires shutdown", result["errors"][0]["message"])
            self.assertFalse(profiler.session_reusable)
            self.assertEqual(4, result["metrics"]["global_batch_size"])
            manifest = result["metrics"]["profiling"]
            self.assertEqual("failed", manifest["status"])
            self.assertEqual([], manifest["artifacts"])
            self.assertEqual(500, len(manifest["error"]))
            self.assertNotIn("\n", manifest["error"])
            self.assertEqual(
                manifest,
                json.loads((Path(directory) / manifest["manifest_path"]).read_text()),
            )

        for failure in ("copy", "mkdir"):
            with self.subTest(
                failure=failure
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                trace = root / "source.json"
                batch_executor = Mock()
                batch_executor.measure.return_value = decode_profiling_measurement(4)
                batch_executor.profile.side_effect = lambda *args, **kwargs: (
                    trace.write_text(json.dumps({"traceEvents": []}))
                )
                if failure == "mkdir":
                    (root / "traces").write_text("occupied")

                def persist(path, payload):
                    if path.parent.name == "traces":
                        raise OSError("trace copy disk full")
                    atomic_write_json(path, payload)

                profiler = DecodeProfiler(
                    batch_executor,
                    run_root=root,
                    log_path=None,
                    artifact_discoverer=lambda *_args: (str(trace),),
                )
                executor = DecodeCaseExecutor(
                    batch_executor,
                    topology=decode_profiling_TOPOLOGY,
                    profiler=profiler,
                )
                with patch(
                    "rtp_llm.test.perf_test.decode.profiling.atomic_write_json",
                    side_effect=persist,
                ):
                    result = executor.execute(
                        decode_profiling_case_payload(
                            decode_profiling_exact_case(profile=True)
                        )
                    )
                self.assertEqual("ok", result["status"])
                self.assertTrue(profiler.session_reusable)
                manifest = result["metrics"]["profiling"]
                self.assertEqual("failed", manifest["status"])
                self.assertEqual([], manifest["artifacts"])
                self.assertIn(
                    "disk full" if failure == "copy" else "File exists",
                    manifest["error"],
                )
                self.assertEqual(
                    manifest, json.loads((root / manifest["manifest_path"]).read_text())
                )
                next_result = executor.execute(
                    decode_profiling_case_payload(decode_profiling_exact_case())
                )
                self.assertEqual("ok", next_result["status"])
                self.assertEqual(2, batch_executor.measure.call_count)

    def test_flush_timeout_writes_failed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            profiler = DecodeProfiler(
                Mock(),
                run_root=Path(directory),
                log_path=None,
                monotonic=Mock(side_effect=[0.0, 0.0, 2.0]),
                sleep=lambda _seconds: None,
                artifact_discoverer=lambda _log, _offset, _name: (),
            )
            case = replace(
                decode_profiling_exact_case(batch=1, profile=True),
                profiling=DecodeProfilingConfig(enabled=True, flush_timeout_seconds=1),
            )
            manifest = profiler.profile(case, batch_size=1)
            self.assertEqual("failed", manifest["status"])
            self.assertIn("trace flush timeout", manifest["error"])
            self.assertTrue((Path(directory) / manifest["manifest_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
