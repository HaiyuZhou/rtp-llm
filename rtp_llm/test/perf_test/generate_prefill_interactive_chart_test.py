import json
import tempfile
import unittest
import warnings
from pathlib import Path

from rtp_llm.test.perf_test.generate_prefill_interactive_chart import (
    load_rows,
    number,
    observed_cache_len,
    prefill_rt,
)


def _sample_metrics(statuses=None):
    if statuses is None:
        statuses = ["ok"]
    metrics = []
    for idx, status in enumerate(statuses):
        metrics.append(
            {
                "batch_size": 1,
                "input_len": 2048 + idx * 1024,
                "cache_len_requested": 512,
                "cache_len_observed": [512, 512, 512],
                "measure_runs": 3,
                "success_runs": 3,
                "status": status,
                "runs": [
                    {"success": True, "reuse_len": 512, "prefill_time_ms": 10.0 + idx}
                    for _ in range(3)
                ],
            }
        )
    return {"metrics": metrics}


class LoadRowsTest(unittest.TestCase):
    def test_accepts_invalid_reuse_status(self):
        payload = _sample_metrics(["invalid_reuse"])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
        self.assertEqual(len(rows), 1)

    def test_rejects_failed_status(self):
        payload = _sample_metrics(["failed"])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
        self.assertEqual(len(rows), 0)

    def test_deduplicates_by_geometry_median(self):
        payload = {
            "metrics": [
                {
                    "batch_size": 1,
                    "input_len": 2048,
                    "cache_len_requested": 512,
                    "cache_len_observed": [512, 512, 512],
                    "measure_runs": 3,
                    "success_runs": 3,
                    "status": "ok",
                    "runs": [
                        {"success": True, "reuse_len": 512, "prefill_time_ms": rt}
                        for rt in (9.0, 10.0, 11.0)
                    ],
                },
                {
                    "batch_size": 1,
                    "input_len": 2048,
                    "cache_len_requested": 512,
                    "cache_len_observed": [512, 512, 512],
                    "measure_runs": 3,
                    "success_runs": 3,
                    "status": "ok",
                    "runs": [
                        {"success": True, "reuse_len": 512, "prefill_time_ms": rt}
                        for rt in (19.0, 20.0, 21.0)
                    ],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["prefill_rt"], 15.0)
        self.assertEqual(rows[0]["compute_len"], 1536)
        self.assertEqual(rows[0]["cache_len"], 512)

    def test_axes_are_compute_cache_rt(self):
        payload = _sample_metrics(["ok"])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input_len"], 2048)
        self.assertEqual(rows[0]["cache_len"], 512)
        self.assertEqual(rows[0]["compute_len"], 1536)
        self.assertAlmostEqual(rows[0]["prefill_rt"], 10.0)


class NumberTest(unittest.TestCase):
    def test_finite(self):
        self.assertEqual(number(42), 42.0)
        self.assertEqual(number("3.14"), 3.14)

    def test_non_finite(self):
        self.assertIsNone(number(float("nan")))
        self.assertIsNone(number(float("inf")))
        self.assertIsNone(number("abc"))
        self.assertIsNone(number(None))


class ObservedCacheLenTest(unittest.TestCase):
    def test_from_observed_list(self):
        item = {"cache_len_observed": [512, 512, 512]}
        self.assertEqual(observed_cache_len(item), 512.0)

    def test_inconsistent_list(self):
        item = {"cache_len_observed": [512, 256, 512]}
        self.assertIsNone(observed_cache_len(item))

    def test_from_runs(self):
        item = {
            "runs": [
                {"reuse_len": 512},
                {"reuse_len": 512},
                {"reuse_len": 512},
            ]
        }
        self.assertEqual(observed_cache_len(item), 512.0)


class PrefillRtTest(unittest.TestCase):
    def test_from_direct_key(self):
        item = {"prefill_time_ms": 42.0}
        self.assertEqual(prefill_rt(item), 42.0)

    def test_from_runs(self):
        item = {
            "runs": [
                {"success": True, "prefill_time_ms": 9.0},
                {"success": True, "prefill_time_ms": 11.0},
                {"success": True, "prefill_time_ms": 10.0},
            ]
        }
        self.assertEqual(prefill_rt(item), 10.0)


if __name__ == "__main__":
    unittest.main()
