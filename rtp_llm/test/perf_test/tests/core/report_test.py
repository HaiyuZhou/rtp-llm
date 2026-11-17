"""Performance framework behavior and failure regressions."""

import copy
import csv
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

from rtp_llm.test.perf_test.decode.report import generate_decode_report
from rtp_llm.test.perf_test.execution.engine_session import EngineSession
from rtp_llm.test.perf_test.multi_node.analyzer import merge_decode_results as merge
from rtp_llm.test.perf_test.prefill.report import generate_prefill_report
from rtp_llm.test.perf_test.tests.support.reports import (
    _setup_prefill_report_run,
    decode_multi_node_analyzer_write_collected_run,
    decode_report_exact_cases,
    prefill_report_case_bytes,
)


class PrefillReportTest(unittest.TestCase):

    def test_complete_input_generates_formal_observations_formula_and_chart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            plan = _setup_prefill_report_run(root)
            report = generate_prefill_report(root)
            self.assertEqual("formal", report["report_kind"])
            self.assertEqual(2, report["counts"]["valid"])
            self.assertEqual([], report["excluded"])
            self.assertEqual(
                [case.case_id for case in plan.cases],
                [row["case_id"] for row in report["observations"]],
            )
            self.assertTrue((root / "reports" / "prefill_report.json").is_file())
            self.assertTrue((root / "reports" / "prefill_observations.json").is_file())
            html = (root / "reports" / "prefill_report.html").read_text()
            self.assertIn("formal", html)

    def test_invalid_case_is_excluded_and_report_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            _setup_prefill_report_run(root, statuses=("ok", "invalid_reuse"))
            report = generate_prefill_report(root)
            self.assertEqual("diagnostic", report["report_kind"])
            self.assertEqual(1, report["counts"]["valid"])
            self.assertEqual(1, report["counts"]["excluded"])
            self.assertEqual("invalid_reuse", report["excluded"][0]["status"])

    def test_rerun_never_launches_engine_or_changes_case_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            _setup_prefill_report_run(root)
            before = prefill_report_case_bytes(root)
            with patch.object(
                EngineSession, "start", side_effect=AssertionError("no engine")
            ) as start:
                first = generate_prefill_report(root)
                second = generate_prefill_report(root)
            self.assertEqual(first, second)
            start.assert_not_called()
            self.assertEqual(before, prefill_report_case_bytes(root))


class PrefillChartTest(unittest.TestCase):

    def test_plotly_available_tracks_the_offline_import(self):
        from rtp_llm.test.perf_test.prefill.chart import plotly_available

        with patch.dict(
            "sys.modules",
            {"plotly": None, "plotly.graph_objects": None, "plotly.offline": None},
        ):
            self.assertFalse(plotly_available())
        offline = types.ModuleType("plotly.offline")
        offline.get_plotlyjs = lambda: ""
        package = types.ModuleType("plotly")
        package.offline = offline
        with patch.dict("sys.modules", {"plotly": package, "plotly.offline": offline}):
            self.assertTrue(plotly_available())

    def test_standalone_charts_and_static_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            _setup_prefill_report_run(root, concurrent=True)
            report = generate_prefill_report(root)
            self.assertTrue(report["observations"])
            for row in report["observations"]:
                self.assertGreater(row["latency_ms"], 0)
                self.assertGreater(row["throughput_tokens_per_second"], 0)
            for metric in ("latency", "tpm"):
                document = (root / "reports" / f"{metric}.interactive.html").read_text()
                self.assertIn("<html", document.lower())
                self.assertEqual(
                    "{http://www.w3.org/2000/svg}svg",
                    ElementTree.parse(root / "reports" / f"{metric}.svg").getroot().tag,
                )
            with patch.dict(
                "sys.modules",
                {"plotly": None, "plotly.graph_objects": None, "plotly.offline": None},
            ):
                fallback_report = generate_prefill_report(root)
            self.assertEqual(report["observations"], fallback_report["observations"])
            for metric in ("latency", "tpm"):
                document = (root / "reports" / f"{metric}.interactive.html").read_text()
                self.assertIn("<svg", document)
                self.assertIn("request_index", document)


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
        for row in report["observations"]:
            self.assertNotIn("search", row)
        for matrix in report["matrices"]:
            for cell in matrix["cells"]:
                self.assertNotIn("search", cell)
        self.assertEqual(before, cases)
        first["cells"][0]["errors"].append({"message": "changed"})
        self.assertEqual([], cases[0]["errors"])


class DecodeMultiNodeAnalyzerTest(unittest.TestCase):

    def test_merger_exports_fixed_batch_metrics_to_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decode_multi_node_analyzer_write_collected_run(root)
            rows = merge.merge_all_results(merge.scan_result_files(directory))
            output = root / "decode.csv"
            merge.save_merged_csv(rows, str(output))
            with output.open(newline="") as stream:
                actual = list(csv.DictReader(stream))
            self.assertEqual(
                [
                    {
                        "dp_size": "2",
                        "input_length": "128",
                        "output_length": "64",
                        "batch_strategy": "fixed",
                        "global_batch_size": "8",
                        "success_rate": "1.0",
                        "wait_ms": "1.0",
                        "prefill_ms": "2.0",
                        "decode_ms": "8.0",
                        "tpot_ms": "1.0",
                        "total_ms": "11.0",
                        "throughput_tokens_per_second": "1000.0",
                        "status": "ok",
                    }
                ],
                actual,
            )

    def test_merger_deduplicates_copied_run_identity_but_keeps_distinct_runs(self):
        for same_run in (False, True):
            with self.subTest(
                same_run=same_run
            ), tempfile.TemporaryDirectory() as directory:
                path = decode_multi_node_analyzer_write_collected_run(Path(directory))
                envelope = json.loads(path.read_text())
                if not same_run:
                    envelope["run_id"] = "another-run"
                nested = path.parents[1] / "run" / "results" / "result.json"
                nested.parent.mkdir(parents=True)
                nested.write_text(json.dumps(envelope))
                rows = merge.merge_all_results(merge.scan_result_files(directory))
                self.assertEqual(
                    1 if same_run else 2, sum((len(metrics) for (_, metrics) in rows))
                )


if __name__ == "__main__":
    unittest.main()
