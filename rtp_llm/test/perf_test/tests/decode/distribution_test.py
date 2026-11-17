"""Tests for decode distribution_test."""

import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.tests.support.workload import decode_distribution_source


class DecodeDistributionTest(unittest.TestCase):

    def test_materialized_sequence_lengths_generate_deterministic_actual_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "requests.json").write_text(
                json.dumps({"sequence_lengths": [64, 128, 256, 512]}), encoding="utf-8"
            )
            config_path = root / "perf.json"
            config_path.write_text(
                json.dumps(decode_distribution_source()), encoding="utf-8"
            )
            resolved = load_resolved_config(config_path)
            parsed = parse_config(decode_distribution_source())
            first = plan_decode_cases(
                parsed.workload,
                parsed.engine,
                dp_size=1,
                resolved_workload=resolved.resolved_config["workload"],
            )
            second = plan_decode_cases(
                parsed.workload,
                parsed.engine,
                dp_size=1,
                resolved_workload=resolved.resolved_config["workload"],
            )
            self.assertEqual(first, second)
            self.assertEqual(
                [[64, 128], [64, 128, 256]], [case["input_lengths"] for case in first]
            )
            self.assertEqual([2, 3], [case["batch_policy"]["size"] for case in first])
            self.assertEqual({32}, {case["output_length"] for case in first})


if __name__ == "__main__":
    unittest.main()
