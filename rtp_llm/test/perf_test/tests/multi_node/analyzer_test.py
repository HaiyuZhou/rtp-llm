"""Tests for multi_node analyzer_test."""

import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.multi_node.analyzer import merge_decode_results as merge
from rtp_llm.test.perf_test.tests.support.reports import (
    decode_multi_node_analyzer_write_collected_run,
)


class DecodeMultiNodeAnalyzerTest(unittest.TestCase):

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
