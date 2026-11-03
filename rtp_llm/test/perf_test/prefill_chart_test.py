import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.prefill.report import generate_prefill_report
from rtp_llm.test.perf_test.prefill_report_test import _setup


class PrefillChartTest(unittest.TestCase):
    def test_standalone_charts_and_static_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            _setup(root, concurrent=True)
            generate_prefill_report(root)
            for metric in ("latency", "tpm"):
                path = root / "reports" / f"{metric}.interactive.html"
                self.assertTrue(path.exists())
                document = path.read_text()
                self.assertNotRegex(document, r"<script[^>]+src=")
                self.assertIn("Batch", document)
                self.assertTrue((root / "reports" / f"{metric}.svg").exists())
            with patch.dict(
                "sys.modules", {"plotly": None, "plotly.graph_objects": None}
            ):
                generate_prefill_report(root)
            document = (root / "reports" / "latency.interactive.html").read_text()
            self.assertIn("<svg", document)
            self.assertIn("default", document)
            self.assertIn("request_index", document)
