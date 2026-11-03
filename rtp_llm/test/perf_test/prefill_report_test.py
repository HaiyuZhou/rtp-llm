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


def _setup(
    root, statuses=("ok", "ok"), *, concurrent=False, measure_runs=1, grid_names=None
):
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
    if grid_names:
        identities = [
            dict(identity, grid_name=name)
            for identity, name in zip(identities, grid_names)
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
    def test_formula_rejection_replaces_stale_accepted_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _setup(root, concurrent=True)
            formula_dir = root / "reports" / "formula"
            formula_dir.mkdir()
            (formula_dir / "model.json").write_text('{"production_acceptance": true}')
            (formula_dir / "PREFILL_TIME_FORMULA.txt").write_text("old accepted")
            report = generate_prefill_report(root)
            self.assertFalse(report["formula_fit"]["production_acceptance"])
            self.assertEqual(
                report["formula_fit"]["reason"], "insufficient_valid_geometries"
            )
            self.assertFalse((formula_dir / "model.json").exists())
            self.assertFalse((formula_dir / "PREFILL_TIME_FORMULA.txt").exists())
            self.assertEqual(
                json.loads((formula_dir / "fit_report.json").read_text()),
                report["formula_fit"],
            )

    def test_unexpected_formula_errors_fail_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _setup(root, concurrent=True)
            with patch(
                "rtp_llm.test.perf_test.prefill.formula.fit_prefill_formula",
                side_effect=RuntimeError("solver broken"),
            ):
                with self.assertRaisesRegex(RuntimeError, "solver broken"):
                    generate_prefill_report(root)

    def test_merged_grid_provenance_survives_offline_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            _setup(root, concurrent=True, grid_names=("cold", "cached"))
            report = generate_prefill_report(root)
            self.assertEqual(
                {"cold", "cached"}, {row["grid_name"] for row in report["observations"]}
            )
            persisted = json.loads(
                (root / "reports" / "prefill_observations.json").read_text()
            )
            self.assertEqual(report["observations"], persisted["observations"])
            self.assertEqual(2, len(report["formula_input"]["rows"]))

    def test_legacy_is_retained_but_excluded_from_fit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            _setup(root)
            report = generate_prefill_report(root)
            self.assertEqual([], report["formula_input"]["rows"])
            self.assertFalse(report["observations"][0]["fit_eligible"])

    def test_grouped_and_homogeneous_round_targets_and_provenance(self):
        from types import SimpleNamespace

        from rtp_llm.test.perf_test.prefill.report import _observation

        for grouped in (False, True):
            contract = (
                "concurrent_grouped_batch_v1" if grouped else "concurrent_batch_v1"
            )
            identity = dict(
                input_len=1024,
                cache_len=512,
                compute_len=512,
                batch_size=2,
                measure_runs=2,
                measurement_contract=contract,
                grid_name="grouped" if grouped else "flat",
            )
            if grouped:
                identity.update(
                    prefix_policy="independent",
                    request_groups=[
                        dict(count=1, input_len=512, cache_len=0),
                        dict(count=1, input_len=1024, cache_len=512),
                    ],
                )
            members = []
            for run, (latencies, elapsed) in enumerate([([1, 9], 20), ([2, 10], 40)]):
                for index, latency in enumerate(latencies):
                    member = dict(
                        run=run,
                        request_index=index,
                        latency_ms=latency,
                        round_elapsed_ms=elapsed,
                        measurement_contract=contract,
                        observed_reuse_length=0 if grouped and index == 0 else 512,
                    )
                    if grouped:
                        member.update(
                            input_length=512 if index == 0 else 1024,
                            expected_reuse_length=0 if index == 0 else 512,
                            compute_length=512,
                            group_index=index,
                        )
                    members.append(member)
            metrics = dict(
                input_length=1536 if grouped else 1024,
                expected_reuse_length=512,
                observed_reuse_length=512,
                compute_length=1024 if grouped else 512,
                latency_ms=9.5 if grouped else 5.5,
                throughput_tokens_per_second=38400,
                validation=dict(shape=True, reuse=True, timing=True),
                measurements=list(reversed(members)),
            )
            planned = SimpleNamespace(case_id="case", identity=identity)
            committed = dict(identity=identity, status="ok", runs=2, metrics=metrics)
            row, exclusion = _observation(planned, committed)
            self.assertIsNone(exclusion)
            self.assertIn("grid_name", row)
            self.assertEqual(identity["grid_name"], row["grid_name"])
            self.assertEqual(9.5, row["batch_latency_ms"])
            self.assertEqual(2304000, row["tokens_per_minute"])
            self.assertEqual(
                [0, 1], [m["request_index"] for m in row["request_distribution"]]
            )
            if grouped:
                self.assertEqual(
                    [512, 1024],
                    [m["input_length"] for m in row["request_distribution"]],
                )
            members[0]["observed_reuse_length"] = 99
            row, exclusion = _observation(planned, committed)
            self.assertIsNone(row)
            self.assertIn("reuse", exclusion["reason"])

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
