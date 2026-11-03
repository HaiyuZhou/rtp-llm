import copy
import unittest

from rtp_llm.test.perf_test.plan import case_id
from rtp_llm.test.perf_test.regression.baseline import (
    BaselineSchemaError,
    compare_result_to_baseline,
)
from rtp_llm.test.perf_test.results import RESULT_SCHEMA_VERSION


def _result():
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
                    "search": None,
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


def _set_measured_metrics(result, *, throughput, total_latency):
    metrics = result["cases"][0]["metrics"]
    metrics["throughput_tokens_per_second"] = throughput
    metrics["latency_ms"]["total"] = total_latency
    measurement = metrics["measurements"][0]
    measurement["throughput_tokens_per_second"] = throughput
    for request in measurement["requests"]:
        request["latency_ms"]["total"] = total_latency


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


class BaselineRegressionTest(unittest.TestCase):
    def test_higher_and_lower_is_better_include_threshold_boundary(self):
        result = _result()
        key = result["cases"][0]["case_id"]
        _set_measured_metrics(result, throughput=7600.0, total_latency=11.55)

        comparison = compare_result_to_baseline(
            result,
            _baseline(
                _entry(
                    key,
                    "throughput_tokens_per_second",
                    8000.0,
                    "higher_is_better",
                ),
                _entry(key, "latency_ms.total", 11.0, "lower_is_better"),
            ),
        )

        self.assertTrue(comparison["passed"])
        self.assertEqual("passed", comparison["status"])
        self.assertEqual(
            [7600.0, 11.55], [item["limit"] for item in comparison["comparisons"]]
        )

    def test_values_past_threshold_fail_with_actionable_diagnostics(self):
        result = _result()
        key = result["cases"][0]["case_id"]
        _set_measured_metrics(result, throughput=7599.0, total_latency=11.551)

        comparison = compare_result_to_baseline(
            result,
            _baseline(
                _entry(key, "throughput_tokens_per_second", 8000.0, "higher_is_better"),
                _entry(key, "latency_ms.total", 11.0, "lower_is_better"),
            ),
        )

        self.assertFalse(comparison["passed"])
        self.assertEqual("failed", comparison["status"])
        diagnostics = "\n".join(comparison["diagnostics"])
        self.assertIn(key, diagnostics)
        self.assertIn("throughput_tokens_per_second", diagnostics)
        self.assertIn("latency_ms.total", diagnostics)
        self.assertIn("7599", diagnostics)
        self.assertIn("11.551", diagnostics)

    def test_missing_case_or_metric_is_a_gate_failure(self):
        result = _result()
        key = result["cases"][0]["case_id"]

        comparison = compare_result_to_baseline(
            result,
            _baseline(
                _entry(key, "latency_ms.unknown", 1.0, "lower_is_better"),
                _entry(
                    "decode-" + "0" * 24, "latency_ms.total", 1.0, "lower_is_better"
                ),
            ),
        )

        self.assertFalse(comparison["passed"])
        self.assertEqual(
            ["missing_metric", "missing_case"],
            [item["status"] for item in comparison["comparisons"]],
        )
        self.assertIn("latency_ms.unknown", comparison["diagnostics"][0])
        self.assertIn("case key", comparison["diagnostics"][1])

    def test_partial_result_is_ineligible_even_if_available_metrics_pass(self):
        result = _result()
        key = result["cases"][0]["case_id"]
        result["status"] = "partial"

        comparison = compare_result_to_baseline(
            result,
            _baseline(_entry(key, "latency_ms.total", 11.0, "lower_is_better")),
        )

        self.assertFalse(comparison["passed"])
        self.assertEqual("ineligible", comparison["status"])
        self.assertEqual([], comparison["comparisons"])
        self.assertIn("partial", comparison["diagnostics"][0])

    def test_result_case_key_must_match_mode_and_identity(self):
        result = _result()
        stable_key = result["cases"][0]["case_id"]
        result["cases"][0]["case_id"] = "decode-" + "f" * 24

        comparison = compare_result_to_baseline(
            result,
            _baseline(_entry(stable_key, "latency_ms.total", 11.0, "lower_is_better")),
        )

        self.assertFalse(comparison["passed"])
        self.assertEqual("invalid_result", comparison["status"])
        self.assertIn("stable case key mismatch", comparison["diagnostics"][0])

    def test_baseline_schema_rejects_duplicate_or_invalid_entries(self):
        result = _result()
        key = result["cases"][0]["case_id"]
        entry = _entry(key, "latency_ms.total", 11.0, "lower_is_better")
        invalid = [
            (_baseline(entry, copy.deepcopy(entry)), "duplicate"),
            (_baseline({**entry, "direction": "smaller"}), "direction"),
            (_baseline({**entry, "threshold": -0.01}), "threshold"),
            (_baseline({**entry, "metric": "latency_ms..total"}), "metric"),
        ]

        for baseline, diagnostic in invalid:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(BaselineSchemaError, diagnostic):
                    compare_result_to_baseline(result, baseline)

    def test_legacy_compare_result_baseline_is_rejected(self):
        legacy = {
            "decode": {
                "batch_size": 4,
                "input_len": 128,
                "tps": 1000.0,
            }
        }

        with self.assertRaises(BaselineSchemaError):
            compare_result_to_baseline(_result(), legacy)


if __name__ == "__main__":
    unittest.main()
