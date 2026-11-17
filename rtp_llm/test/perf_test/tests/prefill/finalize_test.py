"""Tests for prefill finalize_test."""

import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.prefill.finalize import finalize_prefill_run
from rtp_llm.test.perf_test.results import parse_result_envelope
from rtp_llm.test.perf_test.tests.support.results import prefill_adapter_finalize_plan


class PrefillFinalizeTest(unittest.TestCase):

    def test_writes_valid_common_envelope_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir(parents=True)
            plan = prefill_adapter_finalize_plan(root)
            identity = plan.cases[0].to_dict()["identity"]
            result = {
                "case_id": plan.cases[0].case_id,
                "identity": identity,
                "status": "ok",
                "runs": 1,
                "errors": [],
                "metrics": {
                    "input_length": 512,
                    "expected_reuse_length": 0,
                    "observed_reuse_length": 0,
                    "compute_length": 512,
                    "latency_ms": 10.0,
                    "throughput_tokens_per_second": 51200.0,
                    "validation": {"shape": True, "reuse": True, "timing": True},
                    "seed": 1,
                    "prefix_policy": {"kind": "shared_seed"},
                    "measurements": [{"run": 0, "latency_ms": 10.0}],
                },
            }
            envelope = finalize_prefill_run(
                root, plan=plan, run_id="prefill-run", case_results=[result]
            )
            stream = root / "results" / "cases.jsonl"
            before = stream.read_bytes()
            finalize_prefill_run(
                root, plan=plan, run_id="prefill-run", case_results=[result]
            )
            self.assertEqual(before, stream.read_bytes())
            self.assertEqual(1, len(before.splitlines()))
            parsed = parse_result_envelope(envelope)
            self.assertEqual("prefill", parsed.mode)
            self.assertEqual("completed", parsed.status.value)
            self.assertTrue((root / "results" / "result.json").is_file())
            self.assertTrue((root / "results" / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
