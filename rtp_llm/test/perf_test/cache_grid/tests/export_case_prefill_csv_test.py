import csv
import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.cache_grid.plot.export_case_prefill_csv import (
    build_table,
    discover_sources,
    export_csv,
    parse_case_ids,
)


def _metric(case_id, input_len, reuse_len, values, case_key=None):
    return {
        "case_id": case_id,
        "case_key": case_key or f"case_{case_id}",
        "batch_size": 1,
        "input_len": input_len,
        "cache_len_observed": [reuse_len] * len(values),
        "runs": [
            {
                "success": value is not None,
                "input_len": input_len,
                "reuse_len": reuse_len,
                "prefill_time_ms": value,
            }
            for value in values
        ],
    }


def _write_result(path, started_at, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"started_at": started_at, "metrics": metrics}),
        encoding="utf-8",
    )


class ParseCaseIdsTest(unittest.TestCase):
    def test_accepts_spaces_commas_and_removes_duplicates(self):
        self.assertEqual(parse_case_ids(["7,8", "9", "8"]), [7, 8, 9])

    def test_rejects_invalid_id(self):
        with self.assertRaisesRegex(ValueError, "invalid case ID"):
            parse_case_ids(["abc"])


class ExportCasePrefillCsvTest(unittest.TestCase):
    def test_exports_original_then_chronological_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_result(
                root / "cache_grid_results.json",
                "2026-01-01T00:00:00+0800",
                [
                    _metric(7, 2048, 512, [10.0, 11.0, 12.0]),
                    _metric(8, 4096, 0, [20.0, 21.0, 22.0]),
                ],
            )
            _write_result(
                root
                / "cache_perf_replays"
                / "retest_bbbbbb123456"
                / "cache_grid_results.json",
                "2026-01-03T00:00:00+0800",
                [_metric(7, 2048, 512, [30.0, 31.0, 32.0])],
            )
            _write_result(
                root
                / "cache_perf_replays"
                / "retest_aaaaaa123456"
                / "cache_grid_results.json",
                "2026-01-02T00:00:00+0800",
                [_metric(7, 2048, 512, [40.0, None, 42.0])],
            )
            output = root / "out.csv"

            row_count, warnings = export_csv(root, [7, 8], output)
            with output.open(encoding="utf-8", newline="") as source:
                rows = list(csv.reader(source))

        self.assertEqual(row_count, 2)
        self.assertEqual(len(warnings), 2)
        self.assertEqual(
            rows[0][:9],
            [
                "case_id",
                "case_key",
                "batch_size",
                "input_len",
                "reuse_len",
                "compute_len",
                "ori_run0",
                "ori_run1",
                "ori_run2",
            ],
        )
        self.assertIn("re_aaaaaa_run0", rows[0])
        self.assertLess(
            rows[0].index("re_aaaaaa_run0"),
            rows[0].index("re_bbbbbb_run0"),
        )
        self.assertEqual(rows[1][:6], ["7", "case_7", "1", "2048", "512", "1536"])
        self.assertEqual(rows[1][6:9], ["10.0", "11.0", "12.0"])
        earlier_index = rows[0].index("re_aaaaaa_run0")
        self.assertEqual(
            rows[1][earlier_index : earlier_index + 3], ["40.0", "", "42.0"]
        )

    def test_rejects_replay_hash_prefix_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("retest_abcdef111111", "retest_abcdef222222"):
                _write_result(
                    root / "cache_perf_replays" / name / "cache_grid_results.json",
                    "2026-01-01",
                    [_metric(7, 2048, 512, [10.0])],
                )
            with self.assertRaisesRegex(ValueError, "hash-prefix collision"):
                discover_sources(root, [7])

    def test_rejects_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_result(
                root / "cache_grid_results.json",
                "2026-01-01",
                [_metric(7, 2048, 512, [10.0])],
            )
            _write_result(
                root / "cache_perf_replays" / "retest" / "cache_grid_results.json",
                "2026-01-02",
                [_metric(7, 2048, 1024, [11.0])],
            )
            sources = discover_sources(root, [7])
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                build_table(sources, [7])

    def test_rejects_case_missing_from_every_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_result(root / "cache_grid_results.json", "2026-01-01", [])
            sources = discover_sources(root, [99])
            with self.assertRaisesRegex(ValueError, "was not found"):
                build_table(sources, [99])


if __name__ == "__main__":
    unittest.main()
