import csv
import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.cache_grid.plot.generate_prefill_traffic_report import (
    CsvSchemaError,
    Reservoir,
    aggregate_production_csvs,
    load_benchmark,
    map_buckets_to_benchmark,
    render_report,
    summarize,
)

HEADERS = [
    "ds",
    "hh",
    "mm",
    "input_len",
    "reuse_len",
    "first_token_cost_time",
]


def write_csv(path, rows, headers=HEADERS):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def benchmark_metric(input_len=2048, cache_len=1024, times=(10.0, 12.0, 14.0)):
    return {
        "batch_size": 1,
        "input_len": input_len,
        "cache_len_requested": cache_len,
        "cache_len_observed": [cache_len] * len(times),
        "measure_runs": len(times),
        "success_runs": len(times),
        "reuse_exact": True,
        "status": "ok",
        "runs": [
            {
                "success": True,
                "input_len": input_len,
                "reuse_len": cache_len,
                "prefill_time_ms": value,
            }
            for value in times
        ],
    }


class ProductionAggregationTest(unittest.TestCase):
    def test_streams_multiple_files_buckets_and_audits_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            write_csv(
                first,
                [
                    ["20260914", "15", "00", "1023.0", "0.0", "10.0"],
                    ["20260914", "15", "00", "1024.0", "1024.0", "20.0"],
                    ["20260914", "15", "00", "100.0", "200.0", "30.0"],
                    ["20260914", "15", "00", "nan", "0.0", "30.0"],
                ],
            )
            write_csv(
                second,
                [["20260914", "15", "15", "1000.0", "0.0", "30.0"]],
            )
            rows, audit, _ = aggregate_production_csvs(
                [first, second], bucket_tokens=1024, reservoir_size=10, seed=7
            )

        self.assertEqual(audit["rows_total"], 5)
        self.assertEqual(audit["rows_accepted"], 3)
        self.assertEqual(audit["rows_rejected"], 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["input_bucket_start"], 0)
        self.assertEqual(rows[0]["input_bucket_end"], 1024)
        self.assertEqual(rows[0]["request_count"], 2)
        self.assertAlmostEqual(rows[0]["online_ttft_p50_ms"], 20.0)
        self.assertEqual(rows[1]["input_bucket_start"], 1024)
        self.assertEqual(rows[1]["reuse_bucket_start"], 1024)
        self.assertEqual(
            audit["source_files"][str(first)]["rejected_by_reason"],
            {"non_integral_token_length": 1, "reuse_exceeds_input": 1},
        )

    def test_custom_bucket_width_serializes_correct_end(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "traffic.csv"
            write_csv(source, [["20260914", "15", "00", "130.0", "0.0", "10.0"]])
            rows, _, _ = aggregate_production_csvs([source], 128, 10, 1)
        self.assertEqual(rows[0]["input_bucket_start"], 128)
        self.assertEqual(rows[0]["input_bucket_end"], 256)

    def test_rejects_missing_header(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "traffic.csv"
            write_csv(source, [], headers=HEADERS[:-1])
            with self.assertRaisesRegex(CsvSchemaError, "first_token_cost_time"):
                aggregate_production_csvs([source], 1024, 10, 1)

    def test_reservoir_is_deterministic_and_bounded(self):
        first = Reservoir(3, 17, "bucket")
        second = Reservoir(3, 17, "bucket")
        for value in range(100):
            first.add(float(value))
            second.add(float(value))
        self.assertEqual(first.values, second.values)
        self.assertEqual(first.seen, 100)
        self.assertEqual(len(first.values), 3)
        self.assertFalse(first.exact)


class BenchmarkLoadingTest(unittest.TestCase):
    def test_strictly_loads_engine_prefill_and_collapses_duplicates(self):
        valid = benchmark_metric(times=(10.0, 12.0, 14.0))
        duplicate = benchmark_metric(times=(20.0, 30.0, 40.0))
        rejected = benchmark_metric()
        rejected["runs"][1]["reuse_len"] = 0
        payload = {"metrics": [valid, duplicate, rejected]}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "benchmark.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            points, audit = load_benchmark(source, 1)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].compute_len, 1024)
        self.assertEqual(points[0].cache_len, 1024)
        self.assertEqual(points[0].offline_prefill_ms, 21.0)
        self.assertEqual(audit["metrics_rejected"], 1)
        self.assertEqual(audit["rejected_by_reason"], {"run_reuse_mismatch": 1})

    def test_excludes_another_batch_without_counting_rejection(self):
        item = benchmark_metric()
        item["batch_size"] = 2
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "benchmark.json"
            source.write_text(json.dumps({"metrics": [item]}), encoding="utf-8")
            points, audit = load_benchmark(source, 1)
        self.assertEqual(points, [])
        self.assertEqual(audit["metrics_selected_batch"], 0)


class MappingAndSummaryTest(unittest.TestCase):
    @staticmethod
    def row(compute, cache, count=4, input_sum=None):
        input_mean = compute + cache
        return {
            "input_bucket_start": 0,
            "input_bucket_end": 1024,
            "reuse_bucket_start": 0,
            "reuse_bucket_end": 1024,
            "request_count": count,
            "input_token_sum": (
                input_sum if input_sum is not None else input_mean * count
            ),
            "reuse_token_sum": cache * count,
            "compute_token_sum": compute * count,
            "request_share": 0.5,
            "input_token_share": 0.5,
            "mean_input_len": float(input_mean),
            "mean_reuse_len": float(cache),
            "mean_compute_len": float(compute),
            "online_ttft_mean_ms": 120.0,
            "online_ttft_p50_ms": 100.0,
            "online_ttft_p95_ms": 200.0,
            "online_ttft_p99_ms": 300.0,
            "ttft_sample_count": count,
            "ttft_quantiles_exact": True,
        }

    def test_maps_nearby_point_and_leaves_far_bucket_uncovered(self):
        rows = [self.row(1050, 950), self.row(5000, 5000)]
        points = [
            type(
                "Point",
                (),
                {"compute_len": 1000, "cache_len": 1000, "offline_prefill_ms": 80.0},
            )(),
            type(
                "Point",
                (),
                {"compute_len": 1100, "cache_len": 900, "offline_prefill_ms": 90.0},
            )(),
        ]
        mapped = map_buckets_to_benchmark(rows, points, 100)
        self.assertTrue(mapped[0]["covered"])
        self.assertEqual(mapped[0]["matched_compute_len"], 1000)
        self.assertEqual(mapped[0]["matched_cache_len"], 1000)
        self.assertEqual(mapped[0]["online_p95_minus_offline_ms"], 120.0)
        self.assertFalse(mapped[1]["covered"])
        self.assertIsNone(mapped[1]["offline_prefill_ms"])

    def test_summary_separates_global_and_covered_metrics(self):
        points = [
            type(
                "Point",
                (),
                {"compute_len": 1000, "cache_len": 1000, "offline_prefill_ms": 80.0},
            )()
        ]
        rows = map_buckets_to_benchmark(
            [self.row(1000, 1000), self.row(5000, 5000)], points, 100
        )
        audit = {
            "rows_accepted": 8,
            "input_token_sum": sum(row["input_token_sum"] for row in rows),
        }
        global_reservoir = Reservoir(20, 1, "global")
        for value in (10.0, 20.0, 30.0, 40.0):
            global_reservoir.add(value)
        summary = summarize(rows, audit, global_reservoir)
        self.assertEqual(summary["coverage"]["covered_requests"], 4)
        self.assertAlmostEqual(summary["coverage"]["covered_request_share"], 0.5)
        self.assertEqual(summary["global_online_ttft"]["p95_ms"], 38.5)
        self.assertEqual(
            summary["request_weighted_bucket_metrics"]["covered_p95_gap_ms"], 120.0
        )


class RenderReportTest(unittest.TestCase):
    def test_emits_labeled_3d_and_2d_report(self):
        points = [
            type(
                "Point",
                (),
                {
                    "compute_len": 1024,
                    "cache_len": 0,
                    "offline_prefill_ms": 80.0,
                    "input_len": 1024,
                },
            )(),
            type(
                "Point",
                (),
                {
                    "compute_len": 1024,
                    "cache_len": 1024,
                    "offline_prefill_ms": 90.0,
                    "input_len": 2048,
                },
            )(),
            type(
                "Point",
                (),
                {
                    "compute_len": 2048,
                    "cache_len": 0,
                    "offline_prefill_ms": 120.0,
                    "input_len": 2048,
                },
            )(),
        ]
        rows = map_buckets_to_benchmark(
            [MappingAndSummaryTest.row(1024, 0), MappingAndSummaryTest.row(1024, 1024)],
            points,
            128,
        )
        audit = {
            "rows_total": 8,
            "rows_accepted": 8,
            "rows_rejected": 0,
            "input_token_sum": sum(row["input_token_sum"] for row in rows),
            "bucket_count": 2,
            "source_files": {
                "traffic.csv": {
                    "rows_total": 8,
                    "rows_accepted": 8,
                    "rows_rejected": 0,
                    "rejected_by_reason": {},
                }
            },
        }
        reservoir = Reservoir(20, 1, "global")
        for value in (10.0, 20.0, 30.0):
            reservoir.add(value)
        decision = summarize(rows, audit, reservoir)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            render_report(
                output,
                points,
                rows,
                audit,
                {"geometries_accepted": 3},
                decision,
                {"bucket_tokens": 1024},
            )
            report = output.read_text(encoding="utf-8")
        self.assertIn("线上流量与离线 prefill 基准对比", report)
        self.assertIn("线上真实体验", report)
        self.assertIn("线上请求密度（底面）", report)
        self.assertIn("离线引擎 prefill", report)


if __name__ == "__main__":
    unittest.main()
