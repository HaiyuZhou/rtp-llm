"""Performance framework behavior and failure regressions."""

import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.plan import case_id
from rtp_llm.test.perf_test.regression.baseline import (
    compare_result_to_baseline,
    write_baseline_candidate,
)
from rtp_llm.test.perf_test.regression.golden import GoldenMismatchError
from rtp_llm.test.perf_test.regression.golden import main as golden_main
from rtp_llm.test.perf_test.results import RESULT_SCHEMA_VERSION


def _set_measured_metrics(result, *, throughput, total_latency):
    metrics = result["cases"][0]["metrics"]
    metrics["throughput_tokens_per_second"] = throughput
    metrics["latency_ms"]["total"] = total_latency
    measurement = metrics["measurements"][0]
    measurement["throughput_tokens_per_second"] = throughput
    for request in measurement["requests"]:
        request["latency_ms"]["total"] = total_latency


class BaselineRegressionTest(unittest.TestCase):

    def test_higher_and_lower_is_better_include_threshold_boundary(self):
        result = _baseline_result()
        key = result["cases"][0]["case_id"]
        _set_measured_metrics(result, throughput=7600.0, total_latency=11.55)
        comparison = compare_result_to_baseline(
            result,
            _baseline(
                _entry(key, "throughput_tokens_per_second", 8000.0, "higher_is_better"),
                _entry(key, "latency_ms.total", 11.0, "lower_is_better"),
            ),
        )
        self.assertTrue(comparison["passed"])
        self.assertEqual("passed", comparison["status"])
        self.assertEqual(
            [7600.0, 11.55], [item["limit"] for item in comparison["comparisons"]]
        )

    def test_partial_result_is_ineligible_even_if_available_metrics_pass(self):
        result = _baseline_result()
        key = result["cases"][0]["case_id"]
        result["status"] = "partial"
        comparison = compare_result_to_baseline(
            result, _baseline(_entry(key, "latency_ms.total", 11.0, "lower_is_better"))
        )
        self.assertFalse(comparison["passed"])
        self.assertEqual("ineligible", comparison["status"])
        self.assertEqual([], comparison["comparisons"])
        self.assertIn("partial", comparison["diagnostics"][0])


def _prefill_case(input_length, cache_length, latency, throughput):
    identity = {
        "batch_size": 1,
        "cache_len": cache_length,
        "compute_len": input_length - cache_length,
        "input_len": input_length,
        "measure_runs": 1,
    }
    return {
        "case_id": case_id("prefill", identity),
        "identity": identity,
        "status": "ok",
        "runs": 1,
        "errors": [],
        "metrics": {
            "input_length": input_length,
            "expected_reuse_length": cache_length,
            "observed_reuse_length": cache_length,
            "compute_length": input_length - cache_length,
            "latency_ms": latency,
            "throughput_tokens_per_second": throughput,
            "validation": {"shape": True, "reuse": True, "timing": True},
            "seed": 7,
            "prefix_policy": {"kind": "shared_seed"},
            "measurements": [{"run": 0}],
        },
    }


def _golden_result():
    cases = [
        _prefill_case(2048, 1024, 20.0, 51200.0),
        _prefill_case(1024, 512, 10.0, 51200.0),
    ]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": "run-001",
        "mode": "prefill",
        "backend": "local",
        "status": "completed",
        "config_fingerprint": "a" * 64,
        "workload_fingerprint": "b" * 64,
        "model": {
            "label": "test-model",
            "type": "registered_model",
            "checkpoint": "/models/test",
            "topology": {"tp": 1, "dp": 1, "pp": 1, "ep": 1},
        },
        "summary": {"planned_cases": 2, "successful_cases": 2, "failed_cases": 0},
        "cases": cases,
    }


class BaselineCandidateTest(unittest.TestCase):

    def test_candidate_write_does_not_overwrite_accepted_baseline_or_golden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reports").mkdir()
            accepted = root / "accepted_baseline.json"
            golden = root / "committed_golden.json"
            accepted.write_bytes(b"accepted-baseline\n")
            golden.write_bytes(b"committed-golden\n")
            path = write_baseline_candidate(root, _golden_result(), threshold=0.05)
            self.assertEqual(root / "reports" / "baseline_candidate.json", path)
            self.assertEqual(b"accepted-baseline\n", accepted.read_bytes())
            self.assertEqual(b"committed-golden\n", golden.read_bytes())
            self.assertEqual(1, json.loads(path.read_text())["schema_version"])


class GoldenTest(unittest.TestCase):

    def test_validation_reports_mismatch_without_rewriting_golden(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "golden.json"
            result_path = Path(directory) / "result.json"
            result_path.write_text(json.dumps(_golden_result()))
            self.assertEqual(
                0,
                golden_main(
                    ["generate", "--result", str(result_path), "--output", str(output)]
                ),
            )
            before = output.read_bytes()
            self.assertEqual(
                0,
                golden_main(
                    ["validate", "--result", str(result_path), "--golden", str(output)]
                ),
            )
            current = _golden_result()
            current["cases"][0]["metrics"]["latency_ms"] = 21.0
            result_path.write_text(json.dumps(current))
            with self.assertRaisesRegex(GoldenMismatchError, "does not match"):
                golden_main(
                    ["validate", "--result", str(result_path), "--golden", str(output)]
                )
            self.assertEqual(before, output.read_bytes())


class PublicPolicyExtractionTest(unittest.TestCase):

    def test_public_baseline_schema_is_available(self):
        try:
            from rtp_llm.test.perf_test.regression.baseline import parse_baseline
        except ModuleNotFoundError:
            self.fail("public baseline parser is missing")
        entries = parse_baseline(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "case_key": "decode-" + "a" * 24,
                        "metric": "latency_ms.total",
                        "value": 10.0,
                        "direction": "lower_is_better",
                        "threshold": 0.05,
                    }
                ],
            }
        )
        self.assertEqual(1, len(entries))
        self.assertEqual(10.0, entries[0].value)

    def test_public_golden_loader_validates_json_object(self):
        try:
            from rtp_llm.test.perf_test.regression.golden import load_result
        except ModuleNotFoundError:
            self.fail("public golden module is missing")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({"run_id": "run-001"}), encoding="utf-8")
            self.assertEqual({"run_id": "run-001"}, load_result(path))
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected an object"):
                load_result(path)


def _baseline_result():
    identity = {
        "workload_kind": "grid",
        "input_lengths": [128] * 4,
        "output_length": 16,
        "batch_policy": {"strategy": "fixed", "size": 4},
        "measure_runs": 1,
        "warmup_runs": 0,
        "generate_config": {},
        "profiling": {
            "enabled": False,
            "runs": 1,
            "steps": 3,
            "enable_all_ranks": True,
            "flush_timeout_seconds": 60,
        },
        "distribution": None,
    }
    latency = {"wait": 1.0, "prefill": 2.0, "decode": 8.0, "tpot": 0.5, "total": 11.0}
    key = case_id("decode", identity)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": "run-001",
        "mode": "decode",
        "backend": "local",
        "status": "completed",
        "config_fingerprint": "a" * 64,
        "workload_fingerprint": "b" * 64,
        "model": {
            "label": "test-model",
            "type": "registered_model",
            "checkpoint": "/models/test",
            "topology": {"tp": 1, "dp": 1, "pp": 1, "ep": 1},
        },
        "summary": {
            "planned_cases": 1,
            "successful_cases": 1,
            "failed_cases": 0,
            "profiling_failed_cases": 0,
        },
        "cases": [
            {
                "case_id": key,
                "identity": identity,
                "status": "ok",
                "runs": 1,
                "errors": [],
                "metrics": {
                    "workload_kind": "grid",
                    "input_lengths": [128] * 4,
                    "output_length": 16,
                    "global_batch_size": 4,
                    "topology": {"tp": 1, "dp": 1, "pp": 1, "ep": 1},
                    "successful_requests": 4,
                    "failed_requests": 0,
                    "latency_ms": latency,
                    "throughput_tokens_per_second": 8000.0,
                    "measurements": [
                        {
                            "run": 0,
                            "throughput_tokens_per_second": 8000.0,
                            "worker_allocations": [],
                            "requests": [
                                {
                                    "request_index": index,
                                    "success": True,
                                    "input_length": 128,
                                    "output_length": 16,
                                    "latency_ms": dict(latency),
                                }
                                for index in range(4)
                            ],
                        }
                    ],
                    "distribution": None,
                    "profiling": {
                        "status": "disabled",
                        "manifest_path": None,
                        "error": None,
                    },
                },
            }
        ],
    }


def _baseline(*entries):
    return {"schema_version": 1, "entries": list(entries)}


def _entry(key, metric, value, direction, threshold=0.05):
    return {
        "case_key": key,
        "metric": metric,
        "value": value,
        "direction": direction,
        "threshold": threshold,
    }


if __name__ == "__main__":
    unittest.main()
