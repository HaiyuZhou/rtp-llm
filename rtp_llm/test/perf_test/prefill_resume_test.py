import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.errors import PlanningError, ResourceError
from rtp_llm.test.perf_test.modes.base import ModeExecutionContext
from rtp_llm.test.perf_test.modes.prefill import PrefillAdapter
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.prefill.checkpoint import PrefillResume, resume_fingerprints
from rtp_llm.test.perf_test.run_directory import (
    RunDirectoryLock,
    initialize_run_directory,
)
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import RunStore


def _config(extra_engine_args=None):
    return {
        "schema_version": 1,
        "model": {"label": "test", "type": "registered", "checkpoint": "/m"},
        "engine": {
            "max_seq_len": 4097,
            "max_batch_tokens": 8192,
            "concurrency_limit": 1,
            "args": extra_engine_args or {},
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


CASES = [
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


def _plan(root, action="run", config=None):
    return build_execution_plan(
        mode="prefill",
        action=action,
        resolved=resolve_config(config or _config()),
        cases=CASES,
        result_directory=str(root),
    )


def _case_result(planned, status):
    return {
        "case_id": planned.case_id,
        "identity": {"input_len": planned.identity["input_len"]},
        "status": status,
        "runs": 1,
        "errors": [] if status == "ok" else [{"code": status, "message": "old"}],
        "metrics": {"latency_ms": 1.0} if status == "ok" else None,
    }


def _tree(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _initialized(root):
    run_plan = _plan(root)
    initialize_run_directory(
        root, source_config=_config(), plan=run_plan, run_id="run", code_version="rev"
    )
    store = RunStore(root)
    store.write_case(run_plan.cases[0].case_id, _case_result(run_plan.cases[0], "ok"))
    store.write_case(
        run_plan.cases[1].case_id, _case_result(run_plan.cases[1], "invalid_reuse")
    )
    fingerprints = resume_fingerprints(run_plan)
    store.write_checkpoint(
        "prefill_progress",
        {
            "schema_version": 1,
            **fingerprints,
            "completed_case_ids": [case.case_id for case in run_plan.cases],
            "case_statuses": {
                run_plan.cases[0].case_id: "ok",
                run_plan.cases[1].case_id: "invalid_reuse",
            },
        },
    )
    return run_plan, store


class PrefillResumeTest(unittest.TestCase):
    def test_resume_preserves_ok_bytes_and_reruns_every_other_case(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            run_plan, store = _initialized(root)
            ok_path = root / "results" / "cases" / f"{run_plan.cases[0].case_id}.json"
            ok_before = ok_path.read_bytes()
            runner = Mock()
            runner.run_case.side_effect = lambda case: _case_result(
                next(
                    item for item in run_plan.cases if item.case_id == case["case_id"]
                ),
                "ok",
            )
            resume = PrefillResume(PrefillAdapter(lambda _context: runner))

            outcome = resume(
                ModeExecutionContext(
                    plan=_plan(root, action="resume"), session=object(), store=store
                )
            )

            self.assertEqual(RunState.COMPLETED, outcome.state)
            runner.run_case.assert_called_once()
            self.assertEqual(
                run_plan.cases[1].case_id, runner.run_case.call_args.args[0]["case_id"]
            )
            self.assertEqual(ok_before, ok_path.read_bytes())
            self.assertTrue(
                all(
                    case["status"] == "ok"
                    for case in store.read_committed_cases().values()
                )
            )

    def test_fingerprint_mismatch_causes_no_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            _, store = _initialized(root)
            before = _tree(root)
            resume = PrefillResume(PrefillAdapter(lambda _context: Mock()))
            mismatched = _plan(
                root, action="resume", config=_config({"reuse_cache": 0})
            )

            with self.assertRaisesRegex(PlanningError, "config_fingerprint"):
                resume(
                    ModeExecutionContext(plan=mismatched, session=object(), store=store)
                )

            self.assertEqual(before, _tree(root))

    def test_missing_checkpoint_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            _, store = _initialized(root)
            (root / "checkpoints" / "prefill_progress.json").unlink()
            before = _tree(root)

            with self.assertRaisesRegex(PlanningError, "checkpoint"):
                PrefillResume(PrefillAdapter(lambda _context: Mock()))(
                    ModeExecutionContext(
                        plan=_plan(root, action="resume"), session=object(), store=store
                    )
                )

            self.assertEqual(before, _tree(root))

    def test_legacy_cache_checkpoint_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            _, store = _initialized(root)
            store.write_checkpoint(
                "prefill_progress",
                {
                    "schema_version": 1,
                    "completed_cases": ["legacy-case"],
                    "cache_grid_sha256": "a" * 64,
                },
            )
            before = _tree(root)

            with self.assertRaisesRegex(PlanningError, "config_fingerprint"):
                PrefillResume(PrefillAdapter(lambda _context: Mock()))(
                    ModeExecutionContext(
                        plan=_plan(root, action="resume"),
                        session=object(),
                        store=store,
                    )
                )

            self.assertEqual(before, _tree(root))

    def test_concurrent_resume_is_rejected_without_runner_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            _, store = _initialized(root)
            runner = Mock()
            resume = PrefillResume(PrefillAdapter(lambda _context: runner))

            with RunDirectoryLock(root, action="resume"):
                before = _tree(root)
                with self.assertRaises(ResourceError):
                    resume(
                        ModeExecutionContext(
                            plan=_plan(root, action="resume"),
                            session=object(),
                            store=store,
                        )
                    )
                self.assertEqual(before, _tree(root))
            runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
