"""Tests for multi_node analyzer_test."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.multi_node.analyzer import merge_decode_results as merge
from rtp_llm.test.perf_test.tests.support.reports import (
    decode_multi_node_analyzer_write_collected_run,
)


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
