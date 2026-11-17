"""Tests for prefill report_test."""

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

from rtp_llm.test.perf_test.execution.engine_session import EngineSession
from rtp_llm.test.perf_test.prefill.report import generate_prefill_report
from rtp_llm.test.perf_test.tests.support.reports import (
    _setup_prefill_report_run,
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
        with patch.dict(
            "sys.modules",
            {"plotly": package, "plotly.offline": offline},
        ):
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


if __name__ == "__main__":
    unittest.main()
