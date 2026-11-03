import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.cli import parse_cli
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.entry import build_request_plan
from rtp_llm.test.perf_test.execution.engine_session import EngineSession
from rtp_llm.test.perf_test.plan import build_execution_plan, thaw
from rtp_llm.test.perf_test.prefill.report import generate_prefill_report
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.storage import RunStore


def _setup(root, statuses=("ok", "ok"), *, concurrent=False, measure_runs=1):
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
    if concurrent:
        identities = [
            {
                **identity,
                "measure_runs": measure_runs,
                "measurement_contract": "concurrent_batch_v1",
            }
            for identity in identities
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
            if concurrent:
                metrics["throughput_tokens_per_second"] = (
                    identity["compute_len"] * 1000.0 / 20.0
                )
                metrics["measurements"][0].update(
                    {
                        "request_index": 0,
                        "round_elapsed_ms": 20.0,
                        "observed_reuse_length": identity["cache_len"],
                        "measurement_contract": "concurrent_batch_v1",
                    }
                )
                metrics["measurements"] = [
                    {**metrics["measurements"][0], "run": run}
                    for run in range(measure_runs)
                ]
            errors = []
        store.write_case(
            case.case_id,
            {
                "case_id": case.case_id,
                "identity": identity,
                "status": status,
                "runs": measure_runs if concurrent else 1,
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
    def test_report_excludes_tampered_concurrent_round(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            plan = _setup(root, concurrent=True)
            case_path = root / "results" / "cases" / f"{plan.cases[0].case_id}.json"
            stored = json.loads(case_path.read_text())
            stored["metrics"]["measurements"][0]["round_elapsed_ms"] = 40.0
            case_path.write_text(json.dumps(stored))

            report = generate_prefill_report(root)

            self.assertEqual("diagnostic", report["report_kind"])
            self.assertEqual(1, report["counts"]["valid"])
            self.assertEqual(plan.cases[0].case_id, report["excluded"][0]["case_id"])
            self.assertIn("throughput", report["excluded"][0]["reason"])
            self.assertEqual(
                "concurrent_batch_v1", report["observations"][0]["measurement_contract"]
            )
            self.assertIn(
                "concurrent_batch_v1",
                (root / "reports" / "prefill_report.html").read_text(),
            )

    def test_report_excludes_missing_entire_concurrent_round(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            plan = _setup(root, concurrent=True, measure_runs=2)
            self.assertEqual(2, generate_prefill_report(root)["counts"]["valid"])
            case_path = root / "results" / "cases" / f"{plan.cases[0].case_id}.json"
            stored = json.loads(case_path.read_text())
            stored["runs"] = 1
            stored["metrics"]["measurements"] = stored["metrics"]["measurements"][:1]
            # Both rounds have equal timing, so the recomputed aggregates are unchanged.
            case_path.write_text(json.dumps(stored))

            report = generate_prefill_report(root)

            self.assertEqual("diagnostic", report["report_kind"])
            self.assertEqual(1, report["counts"]["valid"])
            self.assertEqual(plan.cases[0].case_id, report["excluded"][0]["case_id"])
            self.assertIn("measure_runs", report["excluded"][0]["reason"])

    def test_report_labels_legacy_without_rewriting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            _setup(root)
            before = _case_bytes(root)
            report = generate_prefill_report(root)
            self.assertEqual(
                {"legacy_latency_derived_v1"},
                {row["measurement_contract"] for row in report["observations"]},
            )
            self.assertIn(
                "legacy_latency_derived_v1",
                (root / "reports" / "prefill_report.html").read_text(),
            )
            self.assertEqual(before, _case_bytes(root))

    def test_legacy_plan_report_still_reads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            _setup(root)
            plan_before = (root / "execution_plan.json").read_bytes()
            cases_before = _case_bytes(root)
            plan, _ = build_request_plan(
                parse_cli(["--mode=prefill", "--action=report", f"--run-dir={root}"])
            )
            self.assertEqual("report", plan.action)
            self.assertTrue(
                all("measurement_contract" not in case.identity for case in plan.cases)
            )
            report = generate_prefill_report(root)
            self.assertEqual(2, report["counts"]["valid"])
            self.assertEqual(plan_before, (root / "execution_plan.json").read_bytes())
            self.assertEqual(cases_before, _case_bytes(root))

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
