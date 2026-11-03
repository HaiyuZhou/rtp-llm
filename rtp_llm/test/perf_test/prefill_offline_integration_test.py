"""Public offline command regenerates derived artifacts from formal results."""

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.cli import parse_cli
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.entry import run_request
from rtp_llm.test.perf_test.execution.engine_session import EngineSession
from rtp_llm.test.perf_test.plan import build_execution_plan, thaw
from rtp_llm.test.perf_test.prefill.chart import render_charts
from rtp_llm.test.perf_test.prefill.finalize import finalize_prefill_run
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.storage import RunStore

SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "perf_test"


def _formal_run(root):
    config = {
        "schema_version": 1,
        "model": {"label": "fixture", "type": "registered", "checkpoint": "/m"},
        "engine": {"max_seq_len": 64, "max_batch_tokens": 64, "concurrency_limit": 1},
        "workload": {
            "kind": "cache_grid",
            "max_input_len": 16,
            "cache_alignment": 4,
            "commit_tail_tokens": 4,
            "compute_steps": [4],
            "cache_lengths": [0],
            "batch_sizes": [1],
        },
        "execution": {"backend": "local", "devices": [0]},
        "output": {"result_root": str(root.parent)},
        "regression": {},
        "publish": {},
    }
    plan = build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(config),
        cases=[
            {
                "input_len": 16,
                "cache_len": 0,
                "compute_len": 16,
                "batch_size": 1,
                "measure_runs": 1,
                "measurement_contract": "concurrent_batch_v1",
            }
        ],
        result_directory=str(root),
    )
    initialize_run_directory(
        root, source_config=config, plan=plan, run_id="fixture", code_version="test"
    )
    case = plan.cases[0]
    result = {
        "case_id": case.case_id,
        "identity": thaw(case.identity),
        "status": "ok",
        "runs": 1,
        "errors": [],
        "metrics": {
            "input_length": 16,
            "expected_reuse_length": 0,
            "observed_reuse_length": 0,
            "compute_length": 16,
            "latency_ms": 10.0,
            "throughput_tokens_per_second": 800.0,
            "validation": {"shape": True, "reuse": True, "timing": True},
            "seed": 1,
            "prefix_policy": {"kind": "shared_seed"},
            "measurements": [
                {
                    "run": 0,
                    "request_index": 0,
                    "latency_ms": 10.0,
                    "round_elapsed_ms": 20.0,
                    "observed_reuse_length": 0,
                    "measurement_contract": "concurrent_batch_v1",
                }
            ],
        },
    }
    RunStore(root).write_checkpoint("prefill_progress", {"schema_version": 1})
    finalize_prefill_run(root, plan=plan, run_id="fixture", case_results=[result])


def _formal_bytes(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for tree in ("results", "checkpoints")
        for path in (root / tree).rglob("*.json")
    } | {"execution_plan.json": (root / "execution_plan.json").read_bytes()}


class PrefillOfflineIntegrationTest(unittest.TestCase):
    def test_skip_test_command_regenerates_reports_without_changing_formal_results(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            _formal_run(root)
            before = _formal_bytes(root)
            argv = ["--mode=prefill", "--skip-test", f"--run-dir={root}"]

            with patch.object(
                EngineSession, "start", side_effect=AssertionError("engine launched")
            ) as start:
                first = run_request(parse_cli(argv))
                second = run_request(parse_cli(argv))

            start.assert_not_called()
            self.assertEqual(first, second)
            self.assertEqual(before, _formal_bytes(root))
            self.assertEqual("formal", first["report_kind"])
            self.assertEqual("total_compute_tokens", first["chart"]["x"])
            self.assertEqual("batch_latency_ms", first["chart"]["latency_y"])
            self.assertEqual("tokens_per_minute", first["chart"]["tpm_y"])
            for artifact in (
                "prefill_report.json",
                "latency.interactive.html",
                "tpm.interactive.html",
                "latency.svg",
                "tpm.svg",
                "formula/fit_report.json",
            ):
                self.assertTrue((root / "reports" / artifact).is_file(), artifact)

            completed = subprocess.run(
                [str(SCRIPT), *argv], capture_output=True, text=True
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertEqual("formal", json.loads(completed.stdout)["report_kind"])
            self.assertEqual(before, _formal_bytes(root))

    def test_plotly_multi_batch_selector_visibility(self):
        try:
            import plotly.graph_objects  # noqa: F401
        except ImportError:
            self.skipTest("Plotly is unavailable")
        rows = []
        for batch in (1, 2):
            rows.append(
                {
                    "fit_eligible": True,
                    "grid_name": "grid",
                    "case_id": f"case-{batch}",
                    "batch_size": batch,
                    "measurement_contract": "concurrent_batch_v1",
                    "request_distribution": [
                        {"request_index": index, "compute_length": 16}
                        for index in range(batch)
                    ],
                    "batch_latency_ms": 10.0 * batch,
                    "tokens_per_minute": 1000.0 * batch,
                }
            )
        report = {"observations": rows, "excluded": [], "report_kind": "formal"}
        for metric in ("latency", "tpm"):
            document = render_charts(report)[f"{metric}.interactive.html"]
            match = re.search(r"const figure=(.*?);Plotly.newPlot", document)
            self.assertIsNotNone(match)
            figure = json.loads(match.group(1))
            self.assertEqual(2, len(figure["data"]))
            buttons = figure["layout"]["updatemenus"][0]["buttons"]
            self.assertEqual(
                ["All batches", "Batch 1", "Batch 2"],
                [button["label"] for button in buttons],
            )
            self.assertEqual([True, False], buttons[1]["args"][0]["visible"])
            self.assertEqual([False, True], buttons[2]["args"][0]["visible"])


if __name__ == "__main__":
    unittest.main()
