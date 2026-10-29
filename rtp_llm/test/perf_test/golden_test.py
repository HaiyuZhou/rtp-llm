import copy
import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.plan import case_id
from rtp_llm.test.perf_test.regression.candidate import (
    build_baseline_candidate,
    write_baseline_candidate,
)
from rtp_llm.test.perf_test.regression.golden import (
    GoldenMismatchError,
    validate_golden,
    write_golden,
)


def _prefill_case(input_length, cache_length, latency, throughput):
    identity = {
        "batch_size": 1,
        "cache_len": cache_length,
        "compute_len": input_length - cache_length,
        "input_len": input_length,
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


def _result():
    cases = [
        _prefill_case(2048, 1024, 20.0, 51200.0),
        _prefill_case(1024, 512, 10.0, 51200.0),
    ]
    return {
        "schema_version": 1,
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
    def test_candidate_is_deterministic_and_uses_mode_metric_directions(self):
        result = _result()
        first = build_baseline_candidate(result, threshold=0.1)
        reversed_result = copy.deepcopy(result)
        reversed_result["cases"].reverse()

        second = build_baseline_candidate(reversed_result, threshold=0.1)

        self.assertEqual(first, second)
        self.assertEqual(4, len(first["entries"]))
        self.assertEqual(
            {
                "latency_ms": "lower_is_better",
                "throughput_tokens_per_second": "higher_is_better",
            },
            {entry["metric"]: entry["direction"] for entry in first["entries"]},
        )

    def test_candidate_write_does_not_overwrite_accepted_baseline_or_golden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reports").mkdir()
            accepted = root / "accepted_baseline.json"
            golden = root / "committed_golden.json"
            accepted.write_bytes(b"accepted-baseline\n")
            golden.write_bytes(b"committed-golden\n")

            path = write_baseline_candidate(root, _result(), threshold=0.05)

            self.assertEqual(root / "reports" / "baseline_candidate.json", path)
            self.assertEqual(b"accepted-baseline\n", accepted.read_bytes())
            self.assertEqual(b"committed-golden\n", golden.read_bytes())
            self.assertEqual(1, json.loads(path.read_text())["schema_version"])


class GoldenTest(unittest.TestCase):
    def test_explicit_generation_is_deterministic_and_validation_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "golden.json"

            write_golden(_result(), output)
            first = output.read_bytes()
            write_golden(copy.deepcopy(_result()), output)
            second = output.read_bytes()
            before_validation = output.read_bytes()

            validate_golden(_result(), output)

            self.assertEqual(first, second)
            self.assertEqual(before_validation, output.read_bytes())

    def test_validation_reports_mismatch_without_rewriting_golden(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "golden.json"
            write_golden(_result(), output)
            before = output.read_bytes()
            current = _result()
            current["cases"][0]["metrics"]["latency_ms"] = 21.0

            with self.assertRaisesRegex(GoldenMismatchError, "does not match"):
                validate_golden(current, output)

            self.assertEqual(before, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
