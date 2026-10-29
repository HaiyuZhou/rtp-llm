import copy
import unittest

from rtp_llm.test.perf_test.results import (
    DecodeMetrics,
    PrefillMetrics,
    ResultSchemaError,
    parse_result_envelope,
)


def _base_result(mode):
    return {
        "schema_version": 1,
        "run_id": "run-001",
        "mode": mode,
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
        "summary": {"planned_cases": 1, "successful_cases": 1, "failed_cases": 0},
        "cases": [],
    }


def _prefill_result():
    value = _base_result("prefill")
    value["cases"] = [
        {
            "case_id": "prefill-case-a",
            "identity": {"input_len": 1024, "cache_len": 512, "batch_size": 1},
            "status": "ok",
            "runs": 3,
            "errors": [],
            "metrics": {
                "input_length": 1024,
                "expected_reuse_length": 512,
                "observed_reuse_length": 512,
                "compute_length": 512,
                "latency_ms": 10.5,
                "throughput_tokens_per_second": 48761.9,
                "validation": {"shape": True, "reuse": True, "timing": True},
                "seed": 7,
                "prefix_policy": {"kind": "shared_seed", "commit_tail_tokens": 64},
                "measurements": [{"run": 0, "latency_ms": 10.5}],
            },
        }
    ]
    return value


def _decode_result():
    value = _base_result("decode")
    value["cases"] = [
        {
            "case_id": "decode-case-a",
            "identity": {"input_len": 128, "output_len": 16, "batch_size": 4},
            "status": "ok",
            "runs": 3,
            "errors": [],
            "metrics": {
                "workload_kind": "grid",
                "input_lengths": [128, 128, 128, 128],
                "output_length": 16,
                "global_batch_size": 4,
                "topology": {"tp": 1, "dp": 2, "pp": 1, "ep": 1},
                "successful_requests": 4,
                "failed_requests": 0,
                "latency_ms": {
                    "wait": 1.0,
                    "prefill": 2.0,
                    "decode": 8.0,
                    "tpot": 0.5,
                    "total": 11.0,
                },
                "throughput_tokens_per_second": 8000.0,
                "measurements": [{"run": 0, "successful_requests": 4}],
                "search": None,
                "distribution": None,
            },
        }
    ]
    return value


class ResultEnvelopeTest(unittest.TestCase):
    def test_prefill_envelope_round_trips_as_golden_schema(self):
        source = _prefill_result()

        envelope = parse_result_envelope(source)

        self.assertIsInstance(envelope.cases[0].metrics, PrefillMetrics)
        self.assertEqual(source, envelope.to_dict())

    def test_decode_envelope_round_trips_as_golden_schema(self):
        source = _decode_result()

        envelope = parse_result_envelope(source)

        self.assertIsInstance(envelope.cases[0].metrics, DecodeMetrics)
        self.assertEqual(16, envelope.cases[0].metrics.output_length)
        self.assertEqual(source, envelope.to_dict())

    def test_every_common_envelope_field_is_required(self):
        for field in _base_result("prefill"):
            with self.subTest(field=field):
                value = _prefill_result()
                value.pop(field)
                with self.assertRaisesRegex(ResultSchemaError, field):
                    parse_result_envelope(value)

    def test_mode_specific_metrics_cannot_be_interpreted_as_shared_fields(self):
        value = _prefill_result()
        case = value["cases"][0]
        case["latency_ms"] = case["metrics"].pop("latency_ms")

        with self.assertRaisesRegex(ResultSchemaError, "cases\[0\].latency_ms"):
            parse_result_envelope(value)

    def test_metrics_are_discriminated_by_envelope_mode(self):
        prefill_with_decode = _prefill_result()
        prefill_with_decode["cases"][0]["metrics"] = _decode_result()["cases"][0][
            "metrics"
        ]
        decode_with_prefill = _decode_result()
        decode_with_prefill["cases"][0]["metrics"] = _prefill_result()["cases"][0][
            "metrics"
        ]

        with self.assertRaisesRegex(ResultSchemaError, "required field is missing"):
            parse_result_envelope(prefill_with_decode)
        with self.assertRaisesRegex(ResultSchemaError, "required field is missing"):
            parse_result_envelope(decode_with_prefill)

    def test_failed_case_may_have_diagnostics_without_metrics(self):
        value = _prefill_result()
        value["status"] = "partial"
        value["summary"] = {
            "planned_cases": 1,
            "successful_cases": 0,
            "failed_cases": 1,
        }
        value["cases"][0].update(
            status="invalid_reuse",
            errors=[{"code": "reuse_mismatch", "message": "expected 512, got 0"}],
            metrics=None,
        )

        envelope = parse_result_envelope(value)

        self.assertIsNone(envelope.cases[0].metrics)
        self.assertEqual(value, envelope.to_dict())

    def test_unknown_fields_and_invalid_counts_are_rejected(self):
        cases = []
        unknown = _decode_result()
        unknown["extra"] = True
        cases.append((unknown, "extra"))
        negative = _decode_result()
        negative["summary"]["failed_cases"] = -1
        cases.append((negative, "summary.failed_cases"))
        mismatched = copy.deepcopy(_decode_result())
        mismatched["cases"][0]["metrics"]["global_batch_size"] = 3
        cases.append((mismatched, "global_batch_size"))
        for value, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(ResultSchemaError, diagnostic):
                    parse_result_envelope(value)

    def test_legacy_cache_grid_result_is_not_accepted_as_envelope(self):
        legacy = {
            "schema_version": 2,
            "mode": "prefix_cache_grid",
            "complete": True,
            "run_config": {"cache_grid_json": "grid.json"},
            "metrics": [],
        }

        with self.assertRaises(ResultSchemaError):
            parse_result_envelope(legacy)


if __name__ == "__main__":
    unittest.main()
