import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, call, patch

from rtp_llm.test.perf_test.config_schema import EngineConfig
from rtp_llm.test.perf_test.decode.executor import (
    DecodeBatchExecutor,
    DecodeCapacityError,
    DecodeCaseExecutor,
)
from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    DecodeProfilingConfig,
    ExactBatchPolicy,
    TargetTPOTPolicy,
)
from rtp_llm.test.perf_test.decode.profiling import (
    DecodeProfiler,
    discover_profile_artifacts,
)
from rtp_llm.test.perf_test.plan import case_id

TOPOLOGY = {"tp": 1, "dp": 1, "pp": 1, "ep": 1}


def _exact_case(*, batch=4, profile=False):
    return DecodeCaseSpec(
        workload_kind="grid",
        input_lengths=(128,) * batch,
        output_length=16,
        batch_policy=ExactBatchPolicy(strategy="fixed", size=batch),
        measure_runs=2,
        warmup_runs=1,
        generate_config={"top_p": 0.8, "nested": {"value": 1}},
        profiling=DecodeProfilingConfig(enabled=profile),
    )


def _tpot_case(*, profile=True):
    return replace(
        _exact_case(batch=16, profile=profile),
        batch_policy=TargetTPOTPolicy("target_tpot", 16, 4, 10.0),
    )


def _case_payload(case):
    identity = case.to_identity()
    return {"case_id": case_id("decode", identity), **identity}


def _measurement(batch, tpot=1.0):
    return DecodeBatchMeasurement(
        batch_size=batch,
        input_lengths=(128,) * batch,
        worker_allocations=(),
        measurements=tuple(
            {
                "run": run,
                "throughput_tokens_per_second": 100.0,
                "requests": [
                    {
                        "request_index": index,
                        "success": True,
                        "input_length": 128,
                        "output_length": 16,
                        "latency_ms": {
                            "wait": 1.0,
                            "prefill": 2.0,
                            "decode": 15.0,
                            "tpot": tpot,
                            "total": 17.0,
                        },
                    }
                    for index in range(batch)
                ],
            }
            for run in range(2)
        ),
    )


class DecodeProfilingTest(unittest.TestCase):
    def test_profile_discovers_fresh_complete_trace_without_stdout_save_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "process.log"
            log.write_text("server ready\n")
            case = replace(
                _exact_case(batch=1, profile=True),
                profiling=DecodeProfilingConfig(enabled=True, flush_timeout_seconds=1),
            )
            name = f"decode_in128_out16_bs1_{case_id('decode', case.to_identity()).rsplit('-', 1)[-1]}"
            stale = root / f"{name}_wr0_99.json"
            stale.write_text(json.dumps({"traceEvents": []}))
            fresh = root / f"{name}_wr0_1.json"
            partial = root / f"{name}_wr0_2.json"

            def export_trace(*_args, **_kwargs):
                fresh.write_text(json.dumps({"traceEvents": [{"name": "kernel"}]}))
                partial.write_text('{"traceEvents": [')

            executor = Mock()
            executor.profile.side_effect = export_trace
            with patch.dict(os.environ, {"MAGA_SERVER_WORK_DIR": directory}):
                profiler = DecodeProfiler(
                    executor,
                    run_root=root / "run",
                    log_path=str(log),
                    monotonic=Mock(side_effect=[0.0, 0.0, 2.0]),
                    sleep=lambda _seconds: None,
                )
                manifest = profiler.profile(case, 1)
            self.assertEqual("completed", manifest["status"])
            retained = root / "run/traces" / fresh.name
            self.assertEqual([str(retained.resolve())], manifest["artifacts"])
            self.assertEqual(
                json.loads(fresh.read_text()), json.loads(retained.read_text())
            )
            self.assertIsNone(manifest["error"])

    def test_tpot_profiles_only_selected_batch_after_search(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.json"
            trace.write_text("{}", encoding="utf-8")
            batch_executor = Mock()
            events = []

            def measure(_case, batch):
                events.append(("measure", batch))
                return _measurement(batch, 8.0 if batch <= 8 else 12.0)

            batch_executor.measure.side_effect = measure
            batch_executor.profile.side_effect = (
                lambda _case, batch, **_kw: events.append(("profile", batch))
            )
            profiler = DecodeProfiler(
                batch_executor,
                run_root=Path(directory),
                log_path=None,
                sleep=lambda _seconds: None,
                artifact_discoverer=lambda _log, _offset, _name: (str(trace),),
            )
            result = DecodeCaseExecutor(
                batch_executor, topology=TOPOLOGY, profiler=profiler
            ).execute(_case_payload(_tpot_case()))
            self.assertEqual([("measure", 8), ("measure", 12), ("profile", 8)], events)
            self.assertEqual("ok", result["status"])
            manifest = result["metrics"]["profiling"]
            self.assertEqual("completed", manifest["status"])
            self.assertEqual([128] * 8, manifest["input_lengths"])
            self.assertEqual(8, manifest["batch_size"])
            self.assertEqual([str(trace)], manifest["artifacts"])
            self.assertIn("decode_in128_out16_bs8_", manifest["trace_name"])
            self.assertEqual(
                manifest,
                json.loads((Path(directory) / manifest["manifest_path"]).read_text()),
            )

    def test_profile_failure_does_not_change_valid_performance_status(self):
        with tempfile.TemporaryDirectory() as directory:
            batch_executor = Mock()
            batch_executor.measure.return_value = _measurement(4)
            batch_executor.profile.side_effect = RuntimeError(
                "profile failed\n" + "x" * 600
            )
            profiler = DecodeProfiler(
                batch_executor, run_root=Path(directory), log_path=None
            )
            result = DecodeCaseExecutor(
                batch_executor, topology=TOPOLOGY, profiler=profiler
            ).execute(_case_payload(_exact_case(profile=True)))
            self.assertEqual("ok", result["status"])
            self.assertEqual([], result["errors"])
            manifest = result["metrics"]["profiling"]
            self.assertEqual("failed", manifest["status"])
            self.assertEqual([], manifest["artifacts"])
            self.assertEqual(500, len(manifest["error"]))
            self.assertNotIn("\n", manifest["error"])
            self.assertEqual(
                manifest,
                json.loads((Path(directory) / manifest["manifest_path"]).read_text()),
            )

    def test_disabled_profile_sends_no_profile_request(self):
        batch_executor = Mock()
        batch_executor.measure.return_value = _measurement(1)
        profiler = Mock()
        result = DecodeCaseExecutor(
            batch_executor, topology=TOPOLOGY, profiler=profiler
        ).execute(_case_payload(_exact_case(batch=1)))
        profiler.profile.assert_not_called()
        batch_executor.profile.assert_not_called()
        self.assertEqual("disabled", result["metrics"]["profiling"]["status"])

    def test_failed_profile_request_with_trace_keeps_performance_ok(self):
        for failed_run in (0, 1):
            for failed_request in (0, 1):
                with self.subTest(run=failed_run, request=failed_request):
                    with tempfile.TemporaryDirectory() as directory:
                        trace = Path(directory) / "trace.json"
                        trace.write_text("{}", encoding="utf-8")
                        case = replace(
                            _exact_case(batch=2, profile=True),
                            profiling=DecodeProfilingConfig(enabled=True, runs=2),
                        )
                        successful = _measurement(2).measurements[0]["requests"]
                        failed = _measurement(2).measurements[0]["requests"]
                        failed[failed_request].update(
                            success=False, error="HTTP 503\n" + "x" * 600
                        )
                        profile_rounds = [successful, successful]
                        profile_rounds[failed_run] = failed
                        transport = Mock()
                        transport.run_round.side_effect = [
                            successful
                        ] * 3 + profile_rounds
                        batch_executor = DecodeBatchExecutor(
                            transport,
                            engine=EngineConfig(
                                max_seq_len=512,
                                max_batch_tokens=512,
                                concurrency_limit=2,
                            ),
                            dp_size=1,
                            monotonic=Mock(side_effect=[0.0, 1.0, 2.0, 3.0]),
                        )
                        profiler = DecodeProfiler(
                            batch_executor,
                            run_root=Path(directory),
                            log_path=None,
                            artifact_discoverer=lambda *_args: (str(trace),),
                        )
                        result = DecodeCaseExecutor(
                            batch_executor, topology=TOPOLOGY, profiler=profiler
                        ).execute(_case_payload(case))
                        self.assertEqual("ok", result["status"])
                        self.assertEqual([], result["errors"])
                        self.assertEqual(4, result["metrics"]["successful_requests"])
                        manifest = result["metrics"]["profiling"]
                        self.assertEqual("failed", manifest["status"])
                        self.assertEqual([], manifest["artifacts"])
                        self.assertIn(f"round {failed_run}", manifest["error"])
                        self.assertIn(f"request {failed_request}", manifest["error"])
                        self.assertIn("HTTP 503", manifest["error"])
                        self.assertNotIn("\n", manifest["error"])
                        self.assertLessEqual(len(manifest["error"]), 500)
                        self.assertEqual(
                            manifest,
                            json.loads(
                                (
                                    Path(directory) / manifest["manifest_path"]
                                ).read_text()
                            ),
                        )

    def test_enabled_profile_requires_configured_profiler(self):
        batch_executor = Mock()
        batch_executor.measure.return_value = _measurement(1)
        executor = DecodeCaseExecutor(batch_executor, topology=TOPOLOGY)
        with self.assertRaisesRegex(RuntimeError, "no profiler is configured"):
            executor.execute(_case_payload(_exact_case(batch=1, profile=True)))

    def test_unsuccessful_performance_does_not_profile(self):
        for kind in ("invalid", "request_failed", "target_not_met"):
            with self.subTest(kind=kind):
                batch_executor = Mock()
                measurement = _measurement(1, tpot=20.0)
                case = _exact_case(batch=1, profile=True)
                if kind == "invalid":
                    measurement.measurements[0]["requests"][0]["output_length"] = 1
                elif kind == "request_failed":
                    measurement.measurements[0]["requests"][0]["success"] = False
                else:
                    case = replace(
                        case, batch_policy=TargetTPOTPolicy("target_tpot", 1, 1, 1.0)
                    )
                batch_executor.measure.return_value = measurement
                profiler = Mock()
                result = DecodeCaseExecutor(
                    batch_executor, topology=TOPOLOGY, profiler=profiler
                ).execute(_case_payload(case))
                self.assertNotEqual("ok", result["status"])
                profiler.profile.assert_not_called()

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
                _exact_case(batch=1, profile=True),
                profiling=DecodeProfilingConfig(enabled=True, flush_timeout_seconds=1),
            )
            manifest = profiler.profile(case, batch_size=1)
            self.assertEqual("failed", manifest["status"])
            self.assertIn("trace flush timeout", manifest["error"])
            self.assertTrue((Path(directory) / manifest["manifest_path"]).is_file())

    def test_discovery_error_writes_failed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            profiler = DecodeProfiler(
                Mock(),
                run_root=Path(directory),
                log_path=None,
                artifact_discoverer=Mock(side_effect=OSError("trace read failed")),
            )
            manifest = profiler.profile(_exact_case(batch=1, profile=True), 1)
            self.assertEqual("failed", manifest["status"])
            self.assertIn("trace read failed", manifest["error"])
            self.assertTrue((Path(directory) / manifest["manifest_path"]).is_file())

    def test_polling_discovers_only_traces_after_profile_start(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text("existing log\n", encoding="utf-8")
            discover = Mock(side_effect=[(), ("trace.json",)])
            sleep = Mock()
            profiler = DecodeProfiler(
                Mock(),
                run_root=Path(directory),
                log_path=str(log),
                monotonic=Mock(side_effect=[0.0, 0.0, 0.2]),
                sleep=sleep,
                artifact_discoverer=discover,
            )
            manifest = profiler.profile(_exact_case(batch=1, profile=True), 1)
            self.assertEqual("completed", manifest["status"])
            self.assertEqual(
                [call(str(log), 13, manifest["trace_name"])] * 2,
                discover.call_args_list,
            )
            sleep.assert_called_once_with(0.2)

    def test_manifest_is_atomically_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            profiler = DecodeProfiler(
                Mock(),
                run_root=Path(directory),
                log_path=None,
                artifact_discoverer=lambda *_args: ("trace.json",),
            )
            case = _exact_case(batch=1, profile=True)
            manifest = profiler.profile(case, 1)
            target = Path(directory) / manifest["manifest_path"]
            with patch(
                "rtp_llm.test.perf_test.storage.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    profiler.profile(case, 1)
            self.assertEqual(manifest, json.loads(target.read_text()))
            self.assertEqual([target], list(target.parent.iterdir()))

    def test_discovery_filters_offset_identity_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "selected.json"
            trace.write_text("{}", encoding="utf-8")
            old = root / "selected-old.json"
            old.write_text("{}", encoding="utf-8")
            other = root / "other.json"
            other.write_text("{}", encoding="utf-8")
            log = root / "server.log"
            before = f"profiler trace saved: {old}\n"
            log.write_text(
                before
                + f"info profiler trace saved: {trace}\n"
                + f"profiler trace saved: {trace}\n"
                + f"profiler trace saved: {root / 'selected-missing.json'}\n"
                + f"profiler trace saved: {other}\n",
                encoding="utf-8",
            )
            self.assertEqual(
                (str(trace.resolve()),),
                discover_profile_artifacts(str(log), len(before), "selected"),
            )
            self.assertEqual((), discover_profile_artifacts(None, 0, "selected"))
            self.assertEqual(
                (), discover_profile_artifacts(str(root / "absent.log"), 0, "selected")
            )


class DecodeBatchProfilingTest(unittest.TestCase):
    def test_profile_prepares_selected_batch_and_uses_only_profile_rounds(self):
        transport = Mock()
        transport.run_round.return_value = _measurement(3).measurements[0]["requests"]
        executor = DecodeBatchExecutor(
            transport,
            engine=EngineConfig(
                max_seq_len=512, max_batch_tokens=512, concurrency_limit=2
            ),
            dp_size=2,
        )
        case = replace(
            _exact_case(batch=4, profile=True),
            input_lengths=(100, 200, 300, 400),
            profiling=DecodeProfilingConfig(enabled=True, runs=2),
        )
        before = case.to_identity()
        executor.profile(
            case, 3, trace_name="selected", steps=5, enable_all_ranks=False
        )
        allocations = transport.configure_scheduler.call_args.args[0]
        self.assertEqual(
            [[100, 200], [300]], [worker["input_lengths"] for worker in allocations]
        )
        expected_config = {
            "top_p": 0.8,
            "nested": {"value": 1},
            "gen_timeline": True,
            "profile_step": 5,
            "profile_trace_name": "selected",
        }
        self.assertEqual(
            [
                call.configure_scheduler(allocations),
                call.start_profile(
                    trace_name="selected", steps=5, enable_all_ranks=False
                ),
            ]
            + [
                call.run_round(
                    input_lengths=(100, 200, 300),
                    output_length=16,
                    generate_config=expected_config,
                )
            ]
            * 2,
            transport.method_calls,
        )
        transport.run_round.call_args.kwargs["generate_config"]["nested"]["value"] = 2
        self.assertEqual(before, case.to_identity())

    def test_profile_checks_capacity_before_transport(self):
        for inputs, batch, concurrency, tokens, message in (
            ((128,), 2, 2, 512, "inputs"),
            ((128, 128, 128), 3, 1, 512, "concurrency_limit"),
            ((200, 200, 100), 3, 2, 300, "max_batch_tokens"),
        ):
            with self.subTest(message=message):
                transport = Mock()
                executor = DecodeBatchExecutor(
                    transport,
                    engine=EngineConfig(
                        max_seq_len=512,
                        max_batch_tokens=tokens,
                        concurrency_limit=concurrency,
                    ),
                    dp_size=2,
                )
                case = replace(
                    _exact_case(batch=batch, profile=True), input_lengths=inputs
                )
                with self.assertRaisesRegex(DecodeCapacityError, message):
                    executor.profile(
                        case,
                        batch,
                        trace_name="selected",
                        steps=3,
                        enable_all_ranks=True,
                    )
                self.assertEqual([], transport.method_calls)


if __name__ == "__main__":
    unittest.main()
