import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.modes.base import ModeExecutionContext
from rtp_llm.test.perf_test.modes.prefill import PrefillAdapter
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import RunStore


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
            "cache_lengths": [0],
            "batch_sizes": [1],
        },
        "execution": {"backend": "local", "devices": [0]},
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
        result_directory=str(root),
    )


def _prepare_store(root):
    (root / "checkpoints").mkdir(parents=True)
    (root / "results" / "cases").mkdir(parents=True)
    return RunStore(root)


def _result(case, status):
    return {
        "case_id": case["case_id"],
        "identity": {"input_len": case["input_len"]},
        "status": status,
        "runs": 1,
        "errors": [] if status == "ok" else [{"code": status, "message": "failed"}],
        "metrics": {"latency_ms": 1.0} if status == "ok" else None,
    }


class PrefillAdapterTest(unittest.TestCase):
    def test_failed_case_is_committed_and_independent_case_continues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _plan(root)
            runner = Mock()
            runner.run_case.side_effect = lambda case: _result(
                case, "request_failed" if case["cache_len"] == 0 else "ok"
            )
            adapter = PrefillAdapter(lambda _context: runner)

            outcome = adapter(
                ModeExecutionContext(
                    plan=plan, session=object(), store=_prepare_store(root)
                )
            )

            self.assertEqual(RunState.PARTIAL, outcome.state)
            self.assertEqual(2, runner.run_case.call_count)
            committed = RunStore(root).read_committed_cases()
            self.assertEqual([case.case_id for case in plan.cases], list(committed))
            checkpoint = json.loads(
                (root / "checkpoints" / "prefill_progress.json").read_text()
            )
            self.assertEqual(list(committed), checkpoint["completed_case_ids"])
            self.assertEqual(
                {"request_failed", "ok"}, set(checkpoint["case_statuses"].values())
            )

    def test_unexpected_request_exception_is_recorded_and_execution_continues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _plan(root)
            runner = Mock()
            runner.run_case.side_effect = [
                RuntimeError("transport"),
                _result({"case_id": plan.cases[1].case_id, "input_len": 1024}, "ok"),
            ]
            adapter = PrefillAdapter(lambda _context: runner)

            outcome = adapter(
                ModeExecutionContext(
                    plan=plan, session=object(), store=_prepare_store(root)
                )
            )

            self.assertEqual(RunState.PARTIAL, outcome.state)
            committed = RunStore(root).read_committed_cases()
            self.assertEqual(
                "request_failed", committed[plan.cases[0].case_id]["status"]
            )
            self.assertIn(
                "transport", committed[plan.cases[0].case_id]["errors"][0]["message"]
            )

    def test_interrupt_preserves_only_already_committed_cases_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _plan(root)
            runner = Mock()

            calls = 0

            def execute(case):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt()
                return _result(case, "ok")

            runner.run_case.side_effect = execute
            adapter = PrefillAdapter(lambda _context: runner)
            store = _prepare_store(root)

            with self.assertRaises(KeyboardInterrupt):
                adapter(ModeExecutionContext(plan=plan, session=object(), store=store))

            committed = store.read_committed_cases()
            self.assertEqual([plan.cases[0].case_id], list(committed))
            checkpoint = json.loads(
                (root / "checkpoints" / "prefill_progress.json").read_text()
            )
            self.assertEqual([plan.cases[0].case_id], checkpoint["completed_case_ids"])


if __name__ == "__main__":
    unittest.main()
