import copy
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from rtp_llm.test.perf_test.multi_node.analyzer import batch_analyze
from rtp_llm.test.perf_test.multi_node.analyzer import merge_decode_results as merge


def _decode_case_v2(*, case_id, policy, batch, profile, search):
    inputs = [128] * batch
    return {
        "case_id": case_id,
        "identity": {
            "workload_kind": "grid",
            "input_lengths": [128] * (policy.get("max_size") or batch),
            "output_length": 16,
            "batch_policy": policy,
            "measure_runs": 1,
            "warmup_runs": 0,
            "generate_config": {},
            "profiling": {
                "enabled": profile["status"] != "disabled",
                "runs": 1,
                "steps": 3,
                "enable_all_ranks": True,
                "flush_timeout_seconds": 60,
            },
            "distribution": None,
        },
        "status": "ok",
        "runs": 1,
        "errors": [],
        "metrics": {
            "workload_kind": "grid",
            "input_lengths": inputs,
            "output_length": 16,
            "global_batch_size": batch,
            "topology": {"tp": 1, "dp": 1, "pp": 1, "ep": 1},
            "successful_requests": batch,
            "failed_requests": 0,
            "latency_ms": {
                "wait": 1.0,
                "prefill": 2.0,
                "decode": 8.0,
                "tpot": 1.0,
                "total": 11.0,
            },
            "throughput_tokens_per_second": 1000.0,
            "measurements": [{"run": 0, "worker_allocations": [], "requests": []}],
            "search": search,
            "distribution": None,
            "profiling": profile,
        },
    }


def _decode_envelope_v2():
    return {
        "schema_version": 2,
        "run_id": "decode-run",
        "mode": "decode",
        "backend": "local",
        "status": "completed",
        "config_fingerprint": "a" * 64,
        "workload_fingerprint": "b" * 64,
        "model": {
            "label": "test",
            "type": "registered",
            "checkpoint": "/m",
            "topology": {"tp": 1, "dp": 1, "pp": 1, "ep": 1},
        },
        "summary": {
            "planned_cases": 2,
            "successful_cases": 2,
            "failed_cases": 0,
            "profiling_failed_cases": 1,
        },
        "cases": [
            _decode_case_v2(
                case_id="decode-exact",
                policy={"strategy": "fixed", "size": 4},
                batch=4,
                profile={"status": "disabled", "manifest_path": None, "error": None},
                search=None,
            ),
            _decode_case_v2(
                case_id="decode-tpot",
                policy={
                    "strategy": "target_tpot",
                    "max_size": 16,
                    "candidate_step": 4,
                    "target_tpot_ms": 10.0,
                },
                batch=8,
                profile={
                    "status": "failed",
                    "manifest_path": "profiles/decode-tpot/profile_manifest.json",
                    "error": "trace flush timeout",
                },
                search={
                    "target_tpot_ms": 10.0,
                    "max_size": 16,
                    "candidate_step": 4,
                    "outcome": "found",
                    "best_batch_size": 8,
                    "observations": [{"batch_size": 8, "outcome": "satisfied"}],
                },
            ),
        ],
    }


def _write_collected_run(
    root,
    *,
    input_length=128,
    output_length=64,
    strategy="target_tpot",
    batch_size=8,
    nested=False,
):
    task = root / "Task_dp_size-2"
    run = task / "run" if nested else task
    result_path = run / "results" / "result.json"
    result_path.parent.mkdir(parents=True)
    envelope = _decode_envelope_v2()
    envelope["cases"] = [envelope["cases"][1]]
    envelope["summary"].update(planned_cases=1, successful_cases=1)
    case = envelope["cases"][0]
    case["identity"]["input_lengths"] = [input_length] * 16
    case["identity"]["output_length"] = output_length
    case["identity"]["batch_policy"]["strategy"] = strategy
    case["metrics"].update(
        input_lengths=[input_length] * batch_size,
        output_length=output_length,
        global_batch_size=batch_size,
        successful_requests=batch_size,
    )
    case["metrics"]["search"]["best_batch_size"] = batch_size
    result_path.write_text(json.dumps(envelope), encoding="utf-8")
    return result_path


def _write_profile_manifest(
    root, *, input_length=128, output_length=16, batch=4, nested=False
):
    case_id = f"in{input_length}-out{output_length}-bs{batch}"
    task = root / "Task_dp_size-2"
    run = task / "run" if nested else task
    trace = task / "trace_files" / f"{case_id}.json"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text("{}", encoding="utf-8")
    path = run / "profiles" / case_id / "profile_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "case_id": case_id,
                "dp_size": 99,
                "input_lengths": [input_length] * batch,
                "output_length": output_length,
                "batch_size": batch,
                "trace_name": case_id,
                "artifacts": [str(trace)],
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    return path


class DecodeMultiNodeAnalyzerTest(unittest.TestCase):
    def test_result_merger_reads_standard_decode_envelope(self):
        for nested in (False, True):
            with self.subTest(
                nested=nested
            ), tempfile.TemporaryDirectory() as directory:
                _write_collected_run(Path(directory), nested=nested)
                rows = merge.merge_all_results(merge.scan_result_files(directory))
                self.assertEqual(1, len(rows))
                self.assertEqual({"dp_size": 2}, rows[0][0])
                metric = rows[0][1][0]
                self.assertEqual(64, metric["output_length"])
                self.assertEqual("target_tpot", metric["batch_strategy"])
                self.assertEqual(8, metric["best_batch_size"])
                self.assertEqual(1.0, metric["success_rate"])
                self.assertEqual(1.0, metric["tpot_ms"])
                self.assertEqual("ok", metric["status"])

    def test_merger_keeps_failed_case_and_profile_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_collected_run(Path(directory))
            envelope = json.loads(path.read_text())
            failed = copy.deepcopy(envelope["cases"][0])
            failed.update(
                case_id="failed-case",
                status="target_tpot_not_met",
                metrics=None,
                errors=[{"code": "target_tpot_not_met", "message": "no candidate"}],
            )
            envelope["cases"].append(failed)
            path.write_text(json.dumps(envelope))
            output = io.StringIO()
            with redirect_stdout(output):
                rows = merge.merge_all_results(merge.scan_result_files(directory))
            self.assertEqual(1, len(rows))
            self.assertEqual(1, len(rows[0][1]))
            for diagnostic in (
                "failed-case",
                "target_tpot_not_met",
                "no candidate",
                "decode-tpot",
                "trace flush timeout",
            ):
                self.assertIn(diagnostic, output.getvalue())

    def test_csv_retains_exact_batch_fields_and_request_failure_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_collected_run(Path(directory))
            value = _decode_envelope_v2()
            value["cases"][0]["status"] = "request_failed"
            value["cases"][0]["metrics"].update(
                successful_requests=3, failed_requests=1
            )
            path.write_text(json.dumps(value))
            rows = merge.merge_all_results(merge.scan_result_files(directory))
            self.assertEqual(1, len(rows))
            csv_path = Path(directory) / "merged.csv"
            merge.save_merged_csv(rows, str(csv_path))
            with csv_path.open() as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(
                    [
                        "dp_size",
                        "input_length",
                        "output_length",
                        "batch_strategy",
                        "global_batch_size",
                        "best_batch_size",
                        "success_rate",
                        "wait_ms",
                        "prefill_ms",
                        "decode_ms",
                        "tpot_ms",
                        "total_ms",
                        "throughput_tokens_per_second",
                        "status",
                    ],
                    reader.fieldnames,
                )
                exact, searched = list(reader)
            self.assertEqual("4", exact["best_batch_size"])
            self.assertEqual("0.75", exact["success_rate"])
            self.assertEqual("request_failed", exact["status"])
            self.assertEqual("8", searched["best_batch_size"])

    def test_trace_scan_uses_manifest_and_keeps_output_axis(self):
        for nested in (False, True):
            with self.subTest(
                nested=nested
            ), tempfile.TemporaryDirectory() as directory:
                _write_profile_manifest(
                    Path(directory), output_length=64, nested=nested
                )
                _write_profile_manifest(
                    Path(directory), output_length=16, nested=nested
                )
                configs = batch_analyze.scan_trace_files(directory)
                self.assertEqual(
                    [(128, 16, 4), (128, 64, 4)],
                    [
                        (item.input_length, item.output_length, item.batch_size)
                        for item in configs
                    ],
                )
                self.assertEqual([2, 2], [item.dp_size for item in configs])

    def test_trace_scan_skips_failed_manifests_and_unlisted_traces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_profile_manifest(Path(directory))
            value = json.loads(path.read_text())
            value.update(status="failed", error="flush timeout")
            path.write_text(json.dumps(value))
            trace_dir = path.parents[2] / "trace_files"
            (trace_dir / "normal_profiler_wr0_bs4_seq128_prefill0_1.json").write_text(
                "{}"
            )
            output = io.StringIO()
            with redirect_stdout(output):
                configs = batch_analyze.scan_trace_files(directory)
            self.assertEqual([], configs)
            self.assertIn("failed", output.getvalue())
            self.assertIn("flush timeout", output.getvalue())

    def test_trace_scan_resolves_collected_and_relative_manifest_artifacts(self):
        for relative in (False, True):
            with self.subTest(
                relative=relative
            ), tempfile.TemporaryDirectory() as directory:
                path = _write_profile_manifest(Path(directory), nested=True)
                value = json.loads(path.read_text())
                trace = Path(value["artifacts"][0])
                value["artifacts"] = [
                    (
                        str(Path("../trace_files") / trace.name)
                        if relative
                        else str(Path("/remote/worker/profiles") / trace.name)
                    )
                ]
                path.write_text(json.dumps(value))
                configs = batch_analyze.scan_trace_files(directory)
                self.assertEqual(1, len(configs))
                self.assertEqual(trace, Path(configs[0].trace_file))

    def test_trace_scan_reports_missing_artifacts_and_nonuniform_inputs(self):
        for nonuniform in (False, True):
            with self.subTest(
                nonuniform=nonuniform
            ), tempfile.TemporaryDirectory() as directory:
                path = _write_profile_manifest(Path(directory))
                value = json.loads(path.read_text())
                if nonuniform:
                    value["input_lengths"][1] = 256
                else:
                    Path(value["artifacts"][0]).unlink()
                path.write_text(json.dumps(value))
                output = io.StringIO()
                with redirect_stdout(output):
                    configs = batch_analyze.scan_trace_files(directory)
                self.assertEqual([], configs)
                self.assertIn(str(path), output.getvalue())

    def test_trace_analysis_exports_both_output_lengths_in_every_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for output_length in (16, 64):
                path = _write_profile_manifest(root, output_length=output_length)
                trace = Path(json.loads(path.read_text())["artifacts"][0])
                trace.write_text(
                    json.dumps(
                        {
                            "traceEvents": [
                                {
                                    "cat": "kernel",
                                    "ph": "X",
                                    "name": "gemm",
                                    "ts": 0,
                                    "dur": 2,
                                },
                                {
                                    "cat": "kernel",
                                    "ph": "X",
                                    "name": "gemm",
                                    "ts": 3,
                                    "dur": 2,
                                },
                            ]
                        }
                    )
                )
            configs = batch_analyze.scan_trace_files(directory)
            results = batch_analyze.aggregate_results(configs, min_occurrences=1)
            self.assertEqual([16, 64], [row["output_length"] for row in results])
            self.assertEqual([128, 128], [row["input_length"] for row in results])
            batch_analyze.save_aggregated_csv(results, str(root / "batch.csv"))
            kernels = batch_analyze.filter_kernels_by_module(results, "gemm", "gemm")
            batch_analyze.save_module_results_to_csv("gemm", kernels, directory)
            batch_analyze.generate_combined_summary_table(
                results,
                {"gemm": batch_analyze.ModuleConf(name_pattern="gemm")},
                directory,
            )
            for filename in (
                "batch.csv",
                "gemm_analysis.csv",
                "combined_module_summary.csv",
            ):
                with self.subTest(filename=filename), (
                    root / filename
                ).open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(["16", "64"], [row["output_length"] for row in rows])
                self.assertEqual(["128", "128"], [row["input_length"] for row in rows])


if __name__ == "__main__":
    unittest.main()
