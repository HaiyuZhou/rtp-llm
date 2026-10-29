import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.aone.sharding import plan_shards
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.errors import ReportError
from rtp_llm.test.perf_test.execution.aone import AOneExecutor
from rtp_llm.test.perf_test.plan import build_execution_plan, thaw
from rtp_llm.test.perf_test.run_directory import (
    initialize_run_directory,
    inspect_run_directory,
)
from rtp_llm.test.perf_test.state import RunState


def _source_config(shard_count):
    return {
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
            "batch_sizes": [1],
        },
        "execution": {
            "backend": "aone",
            "resource": "daily",
            "gpu_type": "H20",
            "max_parallel": shard_count,
            "shard_count": shard_count,
        },
        "output": {},
        "regression": {},
        "publish": {},
    }


def _setup(root, shard_count):
    source = _source_config(shard_count)
    cases = [
        {
            "input_len": 512,
            "cache_len": 0,
            "compute_len": 512,
            "batch_size": 1,
            "measure_runs": 1,
            "seed": 1,
            "prefix_policy": "shared_seed",
        },
        {
            "input_len": 1024,
            "cache_len": 512,
            "compute_len": 512,
            "batch_size": 1,
            "measure_runs": 1,
            "seed": 1,
            "prefix_policy": "shared_seed",
        },
    ]
    plan = build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(source),
        cases=cases,
        result_directory=str(root),
    )
    initialize_run_directory(
        root, source_config=source, plan=plan, run_id="run-aone", code_version="rev"
    )
    return plan


def _case_result(case):
    identity = thaw(case.identity)
    return {
        "case_id": case.case_id,
        "identity": identity,
        "status": "ok",
        "runs": 1,
        "errors": [],
        "metrics": {
            "input_length": identity["input_len"],
            "expected_reuse_length": identity["cache_len"],
            "observed_reuse_length": identity["cache_len"],
            "compute_length": identity["compute_len"],
            "latency_ms": 10.0,
            "throughput_tokens_per_second": 1000.0,
            "validation": {"shape": True, "reuse": True, "timing": True},
            "seed": 1,
            "prefix_policy": {"kind": "shared_seed"},
            "measurements": [{"run": 0, "latency_ms": 10.0}],
        },
    }


def _remote_results(plan, failed_shard=None):
    shard_plan = plan_shards(plan)
    cases = {case.case_id: case for case in plan.cases}
    payloads = {}
    for shard in shard_plan.shards:
        failed = shard.shard_id == failed_shard
        completed = () if failed else shard.case_ids
        payloads[shard.shard_id] = {
            "schema_version": 1,
            "run_id": "run-aone",
            "shard_id": shard.shard_id,
            "mode": "prefill",
            "status": "failed" if failed else "completed",
            "config_fingerprint": plan.config_fingerprint,
            "workload_fingerprint": plan.workload_fingerprint,
            "owned_case_ids": list(shard.case_ids),
            "completed_case_ids": list(completed),
            "cases": [_case_result(cases[case_id]) for case_id in completed],
            "engine_diagnostics": {},
            "failure": {"type": "EngineError", "message": "remote"} if failed else None,
        }
    return payloads


class AOneOrchestratorTest(unittest.TestCase):
    def test_single_and_multi_shard_complete_all_stages_and_formal_report(self):
        for shard_count in (1, 2):
            with self.subTest(
                shard_count=shard_count
            ), tempfile.TemporaryDirectory() as temp_dir:
                workspace = Path(temp_dir)
                root = workspace / "runs" / f"run-{shard_count}"
                plan = _setup(root, shard_count)
                payloads = _remote_results(plan)
                builder = Mock()
                submitter = Mock(side_effect=lambda _target, shard_id: shard_id)
                result_collector = Mock(
                    side_effect=lambda job, _shard_id: payloads[job]
                )
                executor = AOneExecutor(
                    workspace_root=workspace,
                    builder=builder,
                    submitter=submitter,
                    result_collector=result_collector,
                )

                outcome = executor.execute(plan)

                self.assertEqual(RunState.COMPLETED, outcome.state)
                self.assertEqual(0, outcome.exit_code)
                self.assertEqual("formal", outcome.report["report_kind"])
                builder.assert_called_once()
                self.assertEqual(shard_count, submitter.call_count)
                self.assertEqual(shard_count, result_collector.call_count)
                manifest = inspect_run_directory(root).manifest
                self.assertEqual("completed", manifest["state"])
                self.assertEqual(
                    {
                        "build": "completed",
                        "submit": "completed",
                        "collect": "completed",
                        "report": "completed",
                    },
                    {
                        name: stage["status"]
                        for name, stage in manifest["stages"].items()
                    },
                )

    def test_remote_failure_keeps_valid_cases_and_generates_diagnostic_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            root = workspace / "runs" / "partial"
            plan = _setup(root, 2)
            failed_shard = plan_shards(plan).shards[1].shard_id
            payloads = _remote_results(plan, failed_shard=failed_shard)
            executor = AOneExecutor(
                workspace_root=workspace,
                builder=Mock(),
                submitter=Mock(side_effect=lambda _target, shard_id: shard_id),
                result_collector=Mock(side_effect=lambda job, _shard_id: payloads[job]),
            )

            outcome = executor.execute(plan)

            self.assertEqual(RunState.PARTIAL, outcome.state)
            self.assertEqual(5, outcome.exit_code)
            self.assertEqual("diagnostic", outcome.report["report_kind"])
            self.assertEqual(1, outcome.report["counts"]["valid"])
            manifest = inspect_run_directory(root).manifest
            self.assertEqual("partial", manifest["state"])
            self.assertEqual("partial", manifest["stages"]["collect"]["status"])
            self.assertEqual("completed", manifest["stages"]["report"]["status"])

    def test_report_failure_is_exit_six_and_does_not_remove_collected_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            root = workspace / "runs" / "report-failure"
            plan = _setup(root, 1)
            payloads = _remote_results(plan)
            executor = AOneExecutor(
                workspace_root=workspace,
                builder=Mock(),
                submitter=Mock(side_effect=lambda _target, shard_id: shard_id),
                result_collector=Mock(side_effect=lambda job, _shard_id: payloads[job]),
                reporter=Mock(side_effect=RuntimeError("chart failed")),
            )

            outcome = executor.execute(plan)

            self.assertEqual(RunState.FAILED, outcome.state)
            self.assertEqual(6, outcome.exit_code)
            self.assertIsInstance(outcome.failure, ReportError)
            self.assertEqual(2, len(list((root / "results" / "cases").glob("*.json"))))
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual("failed", manifest["state"])
            self.assertEqual("failed", manifest["stages"]["report"]["status"])


if __name__ == "__main__":
    unittest.main()
