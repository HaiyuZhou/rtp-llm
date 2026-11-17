"""Tests for decode report_test."""

import copy
import unittest

from rtp_llm.test.perf_test.decode.report import generate_decode_report
from rtp_llm.test.perf_test.tests.support.reports import (
    decode_report_exact_cases,
    decode_report_tpot_cases,
)


class DecodeReportTest(unittest.TestCase):

    def test_exact_report_groups_input_output_matrix_by_batch(self):
        cases = decode_report_exact_cases((128, 512), (16, 64), (1, 4))
        before = copy.deepcopy(cases)
        report = generate_decode_report(cases)
        self.assertEqual(2, report["schema_version"])
        self.assertEqual("formal", report["report_kind"])
        self.assertEqual({"planned": 8, "valid": 8, "excluded": 0}, report["counts"])
        self.assertEqual(
            [1, 4], [matrix["batch_size"] for matrix in report["matrices"]]
        )
        first = report["matrices"][0]
        self.assertEqual([128, 512], first["input_lengths"])
        self.assertEqual([16, 64], first["output_lengths"])
        self.assertEqual(4, len(first["cells"]))
        self.assertEqual(1.0, first["cells"][0]["tpot_ms"])
        self.assertEqual(100.0, first["cells"][0]["throughput_tokens_per_second"])
        self.assertEqual(17.0, first["cells"][0]["total_latency_ms"])
        self.assertEqual(1.0, first["cells"][0]["success_rate"])
        self.assertEqual(
            [item["case_id"] for item in cases],
            [item["case_id"] for item in report["observations"]],
        )
        self.assertEqual(before, cases)
        first["cells"][0]["errors"].append({"message": "changed"})
        self.assertEqual([], cases[0]["errors"])

    def test_tpot_report_cells_expose_selected_batch(self):
        report = generate_decode_report(decode_report_tpot_cases((128,), (16, 64)))
        self.assertEqual("formal", report["report_kind"])
        self.assertEqual(1, len(report["matrices"]))
        self.assertEqual(
            [8, 12],
            [cell["best_batch_size"] for cell in report["matrices"][0]["cells"]],
        )
        self.assertEqual(
            ["found", "found"],
            [cell["search_outcome"] for cell in report["matrices"][0]["cells"]],
        )


if __name__ == "__main__":
    unittest.main()
