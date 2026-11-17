"""Tests for core results_test."""

import copy
import unittest

from rtp_llm.test.perf_test.results import (
    DecodeMetrics,
    PrefillMetrics,
    ResultSchemaError,
    parse_result_envelope,
)
from rtp_llm.test.perf_test.tests.support.results import (
    result_envelope_decode_result,
    result_envelope_prefill_result,
)


class ResultEnvelopeTest(unittest.TestCase):

    def test_prefill_result_preserves_server_ttft_and_wait(self):
        value = result_envelope_prefill_result()
        metrics = value["cases"][0]["metrics"]
        metrics["ttft_ms"] = 10.5
        metrics["wait_time_ms"] = 1.25
        metrics["measurements"][0].update(ttft_ms=10.5, wait_time_ms=1.25)
        parsed = parse_result_envelope(value)
        self.assertEqual(value, parsed.to_dict())

    @staticmethod
    def _concurrent_prefill_result():
        value = result_envelope_prefill_result()
        case = value["cases"][0]
        case["identity"].update(
            batch_size=4, measure_runs=2, measurement_contract="concurrent_batch_v1"
        )
        case["runs"] = 2
        case["metrics"].update(latency_ms=4.5, throughput_tokens_per_second=153600.0)
        case["metrics"]["measurements"] = [
            {
                "run": run,
                "request_index": member,
                "latency_ms": run * 4 + member + 1,
                "observed_reuse_length": 512,
                "round_elapsed_ms": elapsed,
                "measurement_contract": "concurrent_batch_v1",
            }
            for (run, elapsed) in enumerate((10.0, 20.0))
            for member in range(4)
        ]
        return value

    def test_concurrent_prefill_rejects_missing_entire_round(self):
        value = self._concurrent_prefill_result()
        case = value["cases"][0]
        case["runs"] = 1
        case["metrics"]["measurements"] = case["metrics"]["measurements"][:4]
        case["metrics"].update(latency_ms=2.5, throughput_tokens_per_second=204800.0)
        with self.assertRaisesRegex(ResultSchemaError, "measure_runs"):
            parse_result_envelope(value)

    def test_concurrent_prefill_evidence_rejects_tampering(self):
        mutations = {
            "duplicate": lambda c: c["metrics"]["measurements"][1].update(
                request_index=0
            ),
            "missing": lambda c: c["metrics"]["measurements"].pop(),
            "elapsed_inconsistent": lambda c: c["metrics"]["measurements"][0].update(
                round_elapsed_ms=11
            ),
            "reuse": lambda c: c["metrics"]["measurements"][0].update(
                observed_reuse_length=511
            ),
            "throughput_summary": lambda c: c["metrics"].update(
                throughput_tokens_per_second=204800
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                value = self._concurrent_prefill_result()
                mutate(value["cases"][0])
                with self.assertRaises(ResultSchemaError):
                    parse_result_envelope(value)

    def test_failed_tpot_keeps_only_structured_search_diagnostics(self):
        value = result_envelope_decode_result()
        case = value["cases"][0]
        case["identity"]["batch_policy"] = {
            "strategy": "target_tpot",
            "max_size": 4,
            "candidate_step": 4,
            "target_tpot_ms": 10.0,
        }
        search = {
            "target_tpot_ms": 10.0,
            "max_size": 4,
            "candidate_step": 4,
            "outcome": "no_satisfying_candidate",
            "best_batch_size": None,
            "observations": [{"batch_size": 1, "outcome": "target_exceeded"}],
        }
        case.update(
            status="target_tpot_not_met",
            metrics=None,
            errors=[
                {
                    "code": "target_tpot_not_met",
                    "message": "no candidate",
                    "details": {"search": search},
                }
            ],
        )
        value["status"] = "partial"
        value["summary"].update(successful_cases=0, failed_cases=1)
        self.assertEqual(value, parse_result_envelope(value).to_dict())
        for key, bad in (
            ("observations", [{"batch_size": 5, "outcome": "satisfied"}]),
        ):
            with self.subTest(key=key):
                invalid = copy.deepcopy(value)
                invalid["cases"][0]["errors"][0]["details"]["search"][key] = bad
                with self.assertRaises(ResultSchemaError):
                    parse_result_envelope(invalid)

    def test_prefill_envelope_round_trips_as_golden_schema(self):
        source = result_envelope_prefill_result()
        envelope = parse_result_envelope(source)
        self.assertIsInstance(envelope.cases[0].metrics, PrefillMetrics)
        self.assertEqual(source, envelope.to_dict())

    def test_decode_envelope_round_trips_as_golden_schema(self):
        source = result_envelope_decode_result()
        envelope = parse_result_envelope(source)
        self.assertIsInstance(envelope.cases[0].metrics, DecodeMetrics)
        self.assertEqual(16, envelope.cases[0].metrics.output_length)
        self.assertEqual(source, envelope.to_dict())


if __name__ == "__main__":
    unittest.main()
