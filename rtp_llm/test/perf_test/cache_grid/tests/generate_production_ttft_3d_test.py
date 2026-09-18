import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.cache_grid.plot.generate_production_ttft_3d import (
    load_traffic_histogram,
    main,
)


def bucket(compute_len, reuse_len, count, p50, p95, p99, mean=None):
    values = [value for value in (p50, p95, p99) if value is not None]
    return {
        "mean_input_len": float(compute_len + reuse_len),
        "mean_reuse_len": float(reuse_len),
        "mean_compute_len": float(compute_len),
        "request_count": count,
        "online_ttft_p50_ms": p50,
        "online_ttft_p95_ms": p95,
        "online_ttft_p99_ms": p99,
        "online_ttft_mean_ms": mean if mean is not None else sum(values) / len(values),
    }


class LoadTrafficHistogramTest(unittest.TestCase):
    def test_returns_buckets(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "histogram.json"
            source.write_text(
                json.dumps({"buckets": [bucket(100, 0, 5, 1.0, 2.0, 3.0)]}),
                encoding="utf-8",
            )
            rows = load_traffic_histogram(source)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["request_count"], 5)


class MainFigureTest(unittest.TestCase):
    @staticmethod
    def run_main(buckets, *extra_args):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "histogram.json"
            source.write_text(json.dumps({"buckets": buckets}), encoding="utf-8")
            output = Path(directory) / "chart.html"
            argv = [
                "chart",
                "--input",
                str(source),
                "--output",
                str(output),
                *extra_args,
            ]
            with patch("sys.argv", argv), patch(
                "plotly.graph_objects.Figure.write_html", autospec=True
            ) as write_html, redirect_stdout(io.StringIO()):
                main()
            return write_html.call_args.args[0]

    def test_z_axis_uses_selected_ttft_metric_with_reversed_colors(self):
        figure = self.run_main(
            [bucket(100, 0, 5, 1.0, 2.0, 3.0), bucket(200, 100, 1, 4.0, 5.0, 6.0)],
            "--z-metric",
            "p95",
        )
        points = figure.data[0]
        self.assertEqual(list(points.z), [2.0, 5.0])
        self.assertEqual(list(points.marker.color), [5, 1])
        self.assertTrue(points.marker.reversescale)
        self.assertIn("请求数", points.hovertemplate)

    def test_mean_metric_reads_mean_field(self):
        figure = self.run_main(
            [bucket(100, 0, 5, 1.0, 2.0, 3.0, mean=9.0)],
            "--z-metric",
            "mean",
        )
        self.assertEqual(list(figure.data[0].z), [9.0])

    def test_skips_buckets_without_metric_value(self):
        figure = self.run_main(
            [bucket(100, 0, 5, 1.0, None, 3.0), bucket(200, 100, 1, 4.0, 5.0, 6.0)],
            "--z-metric",
            "p95",
        )
        self.assertEqual(list(figure.data[0].z), [5.0])
        self.assertEqual(list(figure.data[0].marker.color), [1])

    def test_color_clip_caps_values_at_percentile(self):
        counts = [1, 10, 100, 1000]
        buckets = [
            bucket(100 * index, 0, count, 1.0, 2.0, 3.0)
            for index, count in enumerate(counts, 1)
        ]
        figure = self.run_main(buckets, "--color-max-percentile", "50")
        colors = list(figure.data[0].marker.color)
        # index int(4 * 0.5) - 1 = 1 -> sorted median 10 caps the scale.
        self.assertEqual(colors, [1, 10, 10, 10])

    def test_log_color_transforms_before_clipping(self):
        counts = [1, 10, 100, 1000]
        buckets = [
            bucket(100 * index, 0, count, 1.0, 2.0, 3.0)
            for index, count in enumerate(counts, 1)
        ]
        figure = self.run_main(
            buckets,
            "--log-color",
            "--color-max-percentile",
            "50",
        )
        colors = list(figure.data[0].marker.color)
        self.assertAlmostEqual(colors[0], math.log1p(1))
        self.assertAlmostEqual(colors[1], math.log1p(10))
        self.assertAlmostEqual(colors[2], math.log1p(10))
        self.assertAlmostEqual(colors[3], math.log1p(10))


if __name__ == "__main__":
    unittest.main()
