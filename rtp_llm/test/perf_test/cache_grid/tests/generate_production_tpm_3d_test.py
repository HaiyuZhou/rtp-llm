import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.cache_grid.plot.generate_production_tpm_3d import (
    MS_PER_MINUTE,
    compute_tpm,
    detect_cards,
    main,
)


def bucket(compute_len, reuse_len, count, ttft_mean, ttft_p95=None):
    input_len = compute_len + reuse_len
    return {
        "input_bucket_start": 0,
        "input_bucket_end": 1024,
        "reuse_bucket_start": 0,
        "reuse_bucket_end": 1024,
        "request_count": count,
        "input_token_sum": input_len * count,
        "reuse_token_sum": reuse_len * count,
        "compute_token_sum": compute_len * count,
        "mean_input_len": float(input_len),
        "mean_reuse_len": float(reuse_len),
        "mean_compute_len": float(compute_len),
        "online_ttft_mean_ms": ttft_mean,
        "online_ttft_p50_ms": ttft_mean,
        "online_ttft_p95_ms": ttft_p95 if ttft_p95 is not None else ttft_mean,
        "online_ttft_p99_ms": ttft_p95 if ttft_p95 is not None else ttft_mean,
    }


class DetectCardsTest(unittest.TestCase):
    def test_explicit_override_wins_without_reading_file(self):
        self.assertEqual(detect_cards(Path("/nonexistent/bench.json"), 6), 6)

    def test_reads_run_config_tp_size(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bench.json"
            source.write_text(
                json.dumps({"run_config": {"tp_size": 2}}), encoding="utf-8"
            )
            self.assertEqual(detect_cards(source, None), 2)

    def test_reads_engine_args(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bench.json"
            source.write_text(
                json.dumps(
                    {
                        "run_config": {
                            "engine": {"args": ["--model_type=x", "--tp_size=8"]}
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(detect_cards(source, None), 8)

    def test_profile_and_legacy_topology_use_all_worker_dimensions(self):
        cases = [
            ({"profile": {"engine": {"tp_size": 8, "ep_size": 8}}}, 8),
            (
                {
                    "profile": {"engine": {"world_size": 16}},
                    "run_config": {"engine": {"tp_size": 8}},
                },
                16,
            ),
            (
                {
                    "run_config": {
                        "engine": {
                            "args": ["--tp_size", "2", "--pp_size=2"],
                            "dp_size": 4,
                        }
                    }
                },
                16,
            ),
            ({"run_config": {"tp_size": 2, "dp_size": 4}}, 8),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bench.json"
            for payload, expected in cases:
                with self.subTest(payload=payload):
                    source.write_text(json.dumps(payload))
                    self.assertEqual(detect_cards(source, None), expected)
                    self.assertEqual(detect_cards(source, 3), 3)

    def test_missing_benchmark_defaults_to_one(self):
        self.assertEqual(detect_cards(Path("/nonexistent/bench.json"), None), 1)


class ComputeTpmTest(unittest.TestCase):
    def test_scales_tokens_by_time_and_cards(self):
        self.assertAlmostEqual(
            compute_tpm(1024.0, 1000.0, 8), 1024.0 * MS_PER_MINUTE / 1000.0 / 8
        )
        self.assertAlmostEqual(compute_tpm(1024.0, 1000.0, 8), 7680.0)

    def test_rejects_invalid_inputs(self):
        self.assertIsNone(compute_tpm(1024.0, 0.0, 8))
        self.assertIsNone(compute_tpm(1024.0, -5.0, 8))
        self.assertIsNone(compute_tpm(-1.0, 100.0, 8))
        self.assertIsNone(compute_tpm(1024.0, 100.0, 0))
        self.assertIsNone(compute_tpm(float("nan"), 100.0, 8))
        self.assertIsNone(compute_tpm(1024.0, float("inf"), 8))


class MainSummaryTest(unittest.TestCase):
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
            captured = io.StringIO()
            with patch("sys.argv", argv), patch(
                "plotly.graph_objects.Figure.write_html", autospec=True
            ) as write_html, redirect_stdout(captured):
                main()
            return write_html.call_args.args[0], json.loads(captured.getvalue())

    def test_single_card_tpm_and_weighted_summary(self):
        # Bucket A: 6000 tokens / 600 ms / 4 cards -> 150000 TPM per card.
        # Bucket B: 400 tokens / 200 ms / 4 cards -> 30000 TPM per card.
        buckets = [
            bucket(1000, 5000, 100, ttft_mean=600.0),
            bucket(400, 0, 300, ttft_mean=200.0),
        ]
        figure, summary = self.run_main(buckets, "--cards", "4")

        self.assertEqual(summary["cards"], 4)
        self.assertEqual(summary["points"], 2)
        self.assertEqual(list(figure.data[0].z), [150000.0, 30000.0])
        self.assertTrue(figure.data[0].marker.reversescale)
        self.assertEqual(summary["total_requests"], 400)
        self.assertEqual(summary["total_input_tokens"], 720000)
        self.assertAlmostEqual(summary["request_weighted_single_card_tpm"], 60000.0)
        self.assertAlmostEqual(summary["token_weighted_single_card_tpm"], 130000.0)
        self.assertAlmostEqual(summary["max_single_card_tpm"], 150000.0)
        self.assertEqual(summary["max_tpm_bucket"]["request_count"], 100)

    def test_p95_ttft_metric_changes_throughput(self):
        buckets = [bucket(400, 0, 10, ttft_mean=100.0, ttft_p95=400.0)]
        figure, summary = self.run_main(buckets, "--cards", "1", "--ttft-metric", "p95")

        self.assertEqual(summary["ttft_metric"], "p95")
        self.assertEqual(list(figure.data[0].z), [60000.0])

    def test_auto_detects_cards_from_benchmark_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "bench.json"
            benchmark.write_text(
                json.dumps(
                    {
                        "profile": {
                            "schema_version": 1,
                            "engine": {"tp_size": 8, "ep_size": 8},
                        }
                    }
                ),
                encoding="utf-8",
            )
            source = root / "histogram.json"
            source.write_text(
                json.dumps(
                    {
                        "settings": {"benchmark": str(benchmark)},
                        "buckets": [bucket(400, 0, 10, ttft_mean=100.0)],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "chart.html"
            argv = [
                "chart",
                "--input",
                str(source),
                "--output",
                str(output),
            ]
            captured = io.StringIO()
            with patch("sys.argv", argv), patch(
                "plotly.graph_objects.Figure.write_html", autospec=True
            ) as write_html, redirect_stdout(captured):
                main()
            summary = json.loads(captured.getvalue())
            figure = write_html.call_args.args[0]

        self.assertEqual(summary["cards"], 8)
        # 400 tokens / 100 ms / 8 cards -> 30000 TPM per card.
        self.assertEqual(list(figure.data[0].z), [30000.0])


if __name__ == "__main__":
    unittest.main()
