import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.generate_prefill_3d_chart import load_rows, render_clean


def _sample_payload():
    return {
        "metrics": [
            {
                "batch_size": 1,
                "input_len": 1024,
                "cache_len_requested": 256,
                "cache_len_observed": [256, 256, 256],
                "measure_runs": 3,
                "success_runs": 3,
                "status": "ok",
                "runs": [
                    {
                        "success": True,
                        "reuse_len": 256,
                        "prefill_time_ms": value,
                    }
                    for value in (9.0, 10.0, 11.0)
                ],
            }
        ]
    }


class GeneratePrefill3dChartTest(unittest.TestCase):
    def test_render_uses_compute_cache_ttft_axis_order(self):
        payload = _sample_payload()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
            self.assertEqual(rows[0]["compute"], 768)
            svg = render_clean(rows, source, 1)
        self.assertIn("compute tokens (X)", svg)
        self.assertIn("cached tokens (Y)", svg)
        self.assertIn("TTFT / prefill RT (Z, ms)", svg)

    def test_default_title_is_dsv4_pro(self):
        payload = _sample_payload()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
            svg = render_clean(rows, source, 1)
        self.assertIn("DeepSeek-V4-Pro Prefill", svg)

    def test_custom_title(self):
        payload = _sample_payload()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
            svg = render_clean(rows, source, 1, title="Custom Model Chart")
        self.assertIn("Custom Model Chart", svg)
        self.assertNotIn("DeepSeek-V4-Pro", svg)

    def test_annotate_cold_threshold(self):
        payload = {
            "metrics": [
                {
                    "batch_size": 1,
                    "input_len": 2048,
                    "cache_len_requested": 0,
                    "cache_len_observed": [0, 0, 0],
                    "measure_runs": 3,
                    "success_runs": 3,
                    "status": "ok",
                    "runs": [
                        {
                            "success": True,
                            "reuse_len": 0,
                            "prefill_time_ms": value,
                        }
                        for value in (50.0, 51.0, 52.0)
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
            svg_high = render_clean(rows, source, 1, annotate_cold_threshold=1024)
            svg_low = render_clean(rows, source, 1, annotate_cold_threshold=4096)
        self.assertIn("cold", svg_high)
        self.assertNotIn("cold", svg_low)


if __name__ == "__main__":
    unittest.main()
