"""Tests for decode metrics_test."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.decode.finalize import finalize_decode_run
from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.results import ResultSchemaError, parse_result_envelope
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.tests.support.results import (
    decode_metrics_case,
    decode_metrics_measurements,
    decode_metrics_selected,
    decode_metrics_source,
    decode_metrics_TOPOLOGY,
)


class DecodeMetricsTest(unittest.TestCase):

    def test_valid_and_partially_failed_batches_preserve_all_metrics(self):
        valid = build_decode_case_result(
            decode_metrics_case(),
            selected=decode_metrics_selected(),
            topology=decode_metrics_TOPOLOGY,
        )
        self.assertEqual("ok", valid["status"])
        self.assertEqual(2, valid["metrics"]["successful_requests"])
        self.assertEqual(0, valid["metrics"]["failed_requests"])
        self.assertEqual(
            {"wait": 1.5, "prefill": 2.5, "decode": 8.5, "tpot": 2.5, "total": 11.5},
            valid["metrics"]["latency_ms"],
        )
        self.assertEqual(640.0, valid["metrics"]["throughput_tokens_per_second"])
        partial = build_decode_case_result(
            decode_metrics_case(),
            selected=decode_metrics_selected(
                decode_metrics_measurements(second_success=False)
            ),
            topology=decode_metrics_TOPOLOGY,
        )
        self.assertEqual("request_failed", partial["status"])
        self.assertEqual(1, partial["metrics"]["successful_requests"])
        self.assertEqual(1, partial["metrics"]["failed_requests"])
        self.assertFalse(
            partial["metrics"]["measurements"][0]["requests"][1]["success"]
        )

    def test_finalize_round_trip_preserves_distribution_and_profile_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            case = decode_metrics_case()
            case["profiling"]["enabled"] = True
            plan = build_execution_plan(
                mode="decode",
                action="run",
                resolved=resolve_config(decode_metrics_source()),
                cases=[
                    {key: value for (key, value) in case.items() if key != "case_id"}
                ],
                result_directory=str(root),
            )
            case["case_id"] = plan.cases[0].case_id
            result = build_decode_case_result(
                case,
                selected=decode_metrics_selected(),
                topology=decode_metrics_TOPOLOGY,
                profiling={
                    "status": "failed",
                    "manifest_path": "profiles/test/profile_manifest.json",
                    "error": "trace flush timeout",
                },
            )
            initialize_run_directory(
                root,
                source_config=decode_metrics_source(),
                plan=plan,
                run_id="decode-run",
                code_version="rev",
            )
            envelope = finalize_decode_run(
                root, plan=plan, run_id="decode-run", case_results=[result]
            )
            parsed = parse_result_envelope(envelope)
            metrics = parsed.cases[0].metrics.to_dict()
            self.assertEqual([64, 128], metrics["input_lengths"])
            self.assertEqual(4, metrics["output_length"])
            self.assertEqual(2, metrics["global_batch_size"])
            self.assertEqual(decode_metrics_TOPOLOGY, metrics["topology"])
            self.assertEqual(
                [64, 128], metrics["distribution"]["actual_sequence_lengths"]
            )
            self.assertNotIn("search", metrics)
            self.assertTrue((root / "results" / "result.json").is_file())
            self.assertTrue((root / "results" / "summary.json").is_file())
            report = root / "reports" / "decode_report.json"
            self.assertEqual("formal", json.loads(report.read_text())["report_kind"])
            self.assertEqual("completed", envelope["status"])
            self.assertEqual(1, envelope["summary"]["profiling_failed_cases"])
            self.assertEqual(
                1,
                json.loads((root / "results" / "summary.json").read_text())[
                    "profiling_failed_cases"
                ],
            )
            paths = [
                root / "results" / "result.json",
                root / "results" / "summary.json",
                report,
                root / "results" / "cases" / f"{case['case_id']}.json",
            ]
            before = {path: path.read_bytes() for path in paths}
            invalid = copy.deepcopy(result)
            invalid["metrics"]["profiling"]["manifest_path"] = None
            with self.assertRaises(ResultSchemaError):
                finalize_decode_run(
                    root, plan=plan, run_id="decode-run", case_results=[invalid]
                )
            self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_metrics_count_every_formal_round_and_preserve_fixed_shape(self):
        case = decode_metrics_case()
        case["measure_runs"] = 2
        measurements = decode_metrics_measurements() + decode_metrics_measurements(
            second_success=False
        )
        measurements[1]["run"] = 1
        selected = decode_metrics_selected(measurements)
        result = build_decode_case_result(
            case, selected=selected, topology=decode_metrics_TOPOLOGY
        )
        self.assertEqual("request_failed", result["status"])
        metrics = result["metrics"]
        self.assertEqual(2, metrics["global_batch_size"])
        self.assertEqual([64, 128], metrics["input_lengths"])
        self.assertEqual(3, metrics["successful_requests"])
        self.assertEqual(1, metrics["failed_requests"])
        self.assertEqual(2, len(metrics["measurements"]))
        self.assertEqual([64, 128], metrics["distribution"]["actual_sequence_lengths"])
        measurements[0]["requests"][0]["latency_ms"]["tpot"] = 999.0
        self.assertEqual(
            2.0, metrics["measurements"][0]["requests"][0]["latency_ms"]["tpot"]
        )


if __name__ == "__main__":
    unittest.main()
