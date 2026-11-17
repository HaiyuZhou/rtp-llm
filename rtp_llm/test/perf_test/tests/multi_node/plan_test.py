"""Tests for multi_node plan_test."""

import json
import tempfile
import unittest
from unittest.mock import patch

from rtp_llm.test.perf_test.config_schema import _parse_execution
from rtp_llm.test.perf_test.multi_node import decode_runtime, prefill_runtime
from rtp_llm.test.perf_test.tests.support.paths import PERF_TEST_ROOT


class ExternalPlanPolicyTest(unittest.TestCase):

    def test_explicit_parser_preserves_default_plan_and_source(self):
        root = PERF_TEST_ROOT
        for mode, runtime in (("prefill", prefill_runtime), ("decode", decode_runtime)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                builder = getattr(runtime, f"build_external_{mode}_plan")
                path = (
                    root
                    / "examples"
                    / (
                        "prefill_local.json"
                        if mode == "prefill"
                        else "decode_grid_local.json"
                    )
                )
                import json

                topology = json.loads(path.read_text())["model"]["topology"]
                (expected, source) = builder(path, directory, topology)
                with patch(
                    "rtp_llm.test.perf_test.config_schema._parse_execution",
                    side_effect=AssertionError("default policy forbidden"),
                ):
                    (actual, actual_source) = builder(
                        path, directory, topology, execution_parser=_parse_execution
                    )
                self.assertEqual(expected.to_dict(), actual.to_dict())
                self.assertEqual(source, actual_source)


if __name__ == "__main__":
    unittest.main()
