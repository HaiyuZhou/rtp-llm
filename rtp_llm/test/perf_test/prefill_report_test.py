import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.execution.engine_session import EngineSession
from rtp_llm.test.perf_test.plan import build_execution_plan, thaw
from rtp_llm.test.perf_test.prefill.report import generate_prefill_report
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.storage import RunStore


def _setup(root, statuses=("ok", "ok")):
    config = {
        "schema_version": 1,
        "model": {"label": "test", "type": "registered", "checkpoint": "/m"},
        "engine": {
            "max_seq_len": 4097,
            "max_batch_tokens": 8192,
            "concurrency_limit": 1,
        },
        "workload": {
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [256],
            "cache_lengths": [0],
            "batch_sizes": [1],
        },
        "execution": {"backend": "local", "devices": [0]},
        "output": {},
        "regression": {},
        "publish": {},
    }
    identities = [
        {"input_len": 512, "cache_len": 0, "compute_len": 512, "batch_size": 1},
        {"input_len": 1024, "cache_len": 512, "compute_len": 512, "batch_size": 1},
    ]
    plan = build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(config),
        cases=identities,
        result_directory=str(root),
    )
    initialize_run_directory(
        root, source_config=config, plan=plan, run_id="run", code_version="rev"
    )
    store = RunStore(root)
    for index, (case, status) in enumerate(zip(plan.cases, statuses)):
        identity = thaw(case.identity)
        metrics = None
        errors = [{"code": status, "message": "invalid"}]
        if status == "ok":
            metrics = {
                "input_length": identity["input_len"],
                "expected_reuse_length": identity["cache_len"],
                "observed_reuse_length": identity["cache_len"],
                "compute_length": identity["compute_len"],
                "latency_ms": 10.0 + index,
                "throughput_tokens_per_second": 1000.0,
                "validation": {"shape": True, "reuse": True, "timing": True},
                "seed": 1,
                "prefix_policy": {"kind": "shared_seed"},
                "measurements": [{"run": 0, "latency_ms": 10.0 + index}],
            }
            errors = []
        store.write_case(
            case.case_id,
            {
                "case_id": case.case_id,
                "identity": identity,
                "status": status,
                "runs": 1,
                "errors": errors,
                "metrics": metrics,
            },
        )
    return plan


def _case_bytes(root):
    return {
        path.name: path.read_bytes()
        for path in (root / "results" / "cases").glob("*.json")
    }


class PrefillReportTest(unittest.TestCase):
    def test_complete_input_generates_formal_observations_formula_and_chart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            plan = _setup(root)

            report = generate_prefill_report(root)

            self.assertEqual("formal", report["report_kind"])
            self.assertEqual(2, report["counts"]["valid"])
            self.assertEqual([], report["excluded"])
            self.assertEqual(
                [case.case_id for case in plan.cases],
                [row["case_id"] for row in report["observations"]],
            )
            self.assertTrue((root / "reports" / "prefill_report.json").is_file())
            self.assertTrue((root / "reports" / "prefill_observations.json").is_file())
            html = (root / "reports" / "prefill_report.html").read_text()
            self.assertIn("formal", html)
            self.assertIn("Server first-token latency", html)

    def test_invalid_case_is_excluded_and_report_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            _setup(root, statuses=("ok", "invalid_reuse"))

            report = generate_prefill_report(root)

            self.assertEqual("diagnostic", report["report_kind"])
            self.assertEqual(1, report["counts"]["valid"])
            self.assertEqual(1, report["counts"]["excluded"])
            self.assertEqual("invalid_reuse", report["excluded"][0]["status"])

    def test_rerun_never_launches_engine_or_changes_case_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            _setup(root)
            before = _case_bytes(root)

            with patch.object(
                EngineSession, "start", side_effect=AssertionError("no engine")
            ) as start:
                first = generate_prefill_report(root)
                second = generate_prefill_report(root)

            self.assertEqual(first, second)
            start.assert_not_called()
            self.assertEqual(before, _case_bytes(root))


if __name__ == "__main__":
    unittest.main()
