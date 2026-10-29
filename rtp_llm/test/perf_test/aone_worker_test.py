import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.aone.executor import RemoteShardExecutor
from rtp_llm.test.perf_test.aone.package import generate_aone_package
from rtp_llm.test.perf_test.aone.worker import load_remote_inputs
from rtp_llm.test.perf_test.aone.worker import main as worker_main
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.execution.engine_session import EngineDiagnostics
from rtp_llm.test.perf_test.modes.prefill import PrefillAdapter
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.state import RunState


def _plan(root):
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
            "cache_lengths": [0, 512],
            "batch_sizes": [1],
        },
        "execution": {
            "backend": "aone",
            "resource": "daily",
            "gpu_type": "H20",
            "max_parallel": 1,
            "shard_count": 1,
        },
        "output": {},
        "regression": {},
        "publish": {},
    }
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
    return build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(config),
        cases=cases,
        result_directory=str(root / "formal-results"),
    )


class _FakeSession:
    def __init__(self):
        self.enter_calls = 0
        self.exit_calls = 0
        self.diagnostics = EngineDiagnostics(log_path="remote-engine.log")

    def __enter__(self):
        self.enter_calls += 1
        return self

    def __exit__(self, _type, _value, _traceback):
        self.exit_calls += 1


def _result(case):
    return {
        "case_id": case["case_id"],
        "identity": {
            key: case[key]
            for key in ("input_len", "cache_len", "compute_len", "batch_size")
        },
        "status": "ok",
        "runs": 1,
        "errors": [],
        "metrics": {
            "input_length": case["input_len"],
            "expected_reuse_length": case["cache_len"],
            "observed_reuse_length": case["cache_len"],
            "compute_length": case["compute_len"],
            "latency_ms": 10.0,
            "throughput_tokens_per_second": 1000.0,
            "validation": {"shape": True, "reuse": True, "timing": True},
            "seed": 1,
            "prefix_policy": {"kind": "shared_seed"},
            "measurements": [{"run": 0, "latency_ms": 10.0}],
        },
    }


class AOneWorkerTest(unittest.TestCase):
    def test_worker_entry_executes_loaded_shard_into_declared_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _plan(root)
            package = generate_aone_package(
                root / "generated",
                plan,
                run_id="run-worker",
                ownership=[[case.case_id for case in plan.cases]],
            )
            executor = Mock()
            executor.execute.return_value = Mock(exit_code=5)
            factory = Mock(return_value=executor)
            output = root / "remote-output"

            exit_code = worker_main(
                [
                    f"--resolved-config={package.resolved_config}",
                    f"--resolved-workload={package.resolved_workload}",
                    f"--shard-plan={package.shard_plan}",
                    "--shard-id=shard-0000",
                    f"--output-dir={output}",
                ],
                executor_factory=factory,
            )

            self.assertEqual(5, exit_code)
            factory.assert_called_once()
            loaded = executor.execute.call_args.args[0]
            self.assertEqual("run-worker", loaded.run_id)
            self.assertEqual("shard-0000", loaded.shard_id)
            self.assertEqual(output, executor.execute.call_args.args[1])

    def test_multi_case_shard_uses_one_session_and_returns_atomic_claims(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _plan(root)
            package = generate_aone_package(
                root / "generated",
                plan,
                run_id="run-remote",
                ownership=[[case.case_id for case in plan.cases]],
            )
            inputs = load_remote_inputs(
                resolved_config=package.resolved_config,
                resolved_workload=package.resolved_workload,
                shard_plan=package.shard_plan,
                shard_id="shard-0000",
            )
            session = _FakeSession()
            session_factory = Mock(return_value=session)
            runner = Mock()
            runner.run_case.side_effect = _result
            executor = RemoteShardExecutor(
                session_factory=session_factory,
                adapter=PrefillAdapter(lambda _context: runner),
            )
            output = root / "remote-output"

            result = executor.execute(inputs, output)

            self.assertEqual(RunState.COMPLETED, result.state)
            self.assertEqual("run-remote", result.run_id)
            self.assertEqual("shard-0000", result.shard_id)
            self.assertEqual(plan.config_fingerprint, result.config_fingerprint)
            self.assertEqual(plan.workload_fingerprint, result.workload_fingerprint)
            self.assertEqual(inputs.owned_case_ids, result.owned_case_ids)
            self.assertEqual(inputs.owned_case_ids, result.completed_case_ids)
            session_factory.assert_called_once()
            self.assertEqual(1, session.enter_calls)
            self.assertEqual(1, session.exit_calls)
            self.assertEqual(2, runner.run_case.call_count)
            case_files = sorted((output / "results" / "cases").glob("*.json"))
            self.assertEqual(2, len(case_files))
            persisted = json.loads((output / "shard_result.json").read_text())
            self.assertEqual(result.to_dict(), persisted)
            self.assertEqual(
                list(inputs.owned_case_ids), persisted["completed_case_ids"]
            )


if __name__ == "__main__":
    unittest.main()
