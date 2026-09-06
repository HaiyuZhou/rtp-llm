import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.generate_prefill_3d_chart import load_rows, render_clean


class GeneratePrefill3dChartTest(unittest.TestCase):
    def test_render_uses_compute_cache_ttft_axis_order(self):
        payload = {
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
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
            self.assertEqual(rows[0]["compute"], 768)
            svg = render_clean(rows, source, 1)
        self.assertIn("compute tokens (X)", svg)
        self.assertIn("cached tokens (Y)", svg)
        self.assertIn("TTFT / prefill RT (Z, ms)", svg)


if __name__ == "__main__":
    unittest.main()
