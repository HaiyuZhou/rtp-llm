"""Frozen plans bind paths without resolving submit-host inputs again."""

import dataclasses
import unittest
from unittest.mock import patch

from rtp_llm.test.perf_test.cli import parse_cli
from rtp_llm.test.perf_test.entry import _new_plan
from rtp_llm.test.perf_test.execution.distributed.plan import bind_member_plan
from rtp_llm.test.perf_test.tests.support.paths import PERF_TEST_ROOT


class FrozenBindingTest(unittest.TestCase):
    def test_only_result_path_is_bound(self):
        request = parse_cli(
            [
                "--mode=decode",
                "--action=run",
                "--config=" + str(PERF_TEST_ROOT / "examples/decode_grid_local.json"),
            ]
        )
        plan, _ = _new_plan(request)
        with patch(
            "rtp_llm.test.perf_test.config_resolver.load_resolved_config",
            side_effect=AssertionError("must not reload source"),
        ):
            bound = bind_member_plan(plan, result_directory="/remote/run")
        expected = dataclasses.replace(plan, result_directory="/remote/run")
        self.assertEqual(expected, bound)
        self.assertEqual(plan.config_fingerprint, bound.config_fingerprint)
        self.assertEqual(plan.workload_fingerprint, bound.workload_fingerprint)
        self.assertEqual(plan.cases, bound.cases)

    def test_relative_binding_is_rejected(self):
        with self.assertRaises(ValueError):
            bind_member_plan(None, result_directory="relative")


if __name__ == "__main__":
    unittest.main()
