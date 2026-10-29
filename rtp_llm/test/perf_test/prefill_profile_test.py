import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.prefill.profile import PrefillProfiler
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.storage import RunStore


def _setup(root):
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
    identity = {
        "input_len": 512,
        "cache_len": 0,
        "compute_len": 512,
        "batch_size": 1,
        "measure_runs": 1,
        "seed": 1,
        "prefix_policy": "shared_seed",
    }
    plan = build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(config),
        cases=[identity],
        result_directory=str(root),
    )
    initialize_run_directory(
        root, source_config=config, plan=plan, run_id="run", code_version="rev"
    )
    store = RunStore(root)
    store.write_checkpoint("prefill_progress", {"formal": True})
    store.write_case(
        plan.cases[0].case_id, {"case_id": plan.cases[0].case_id, "status": "ok"}
    )
    store.write_final_result({"status": "completed"})
    store.write_summary({"successful_cases": 1})
    return plan


def _formal_bytes(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
        and "profiles" not in path.parts
        and path.name != ".writer.lock"
    }


class PrefillProfileTest(unittest.TestCase):
    def test_selected_case_is_reconstructed_into_isolated_profile_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            plan = _setup(root)
            before = _formal_bytes(root)
            runner = Mock()
            runner.run_case.side_effect = lambda case: {
                "case_id": case["case_id"],
                "status": "ok",
                "metrics": {"latency_ms": 1.0},
                "errors": [],
            }
            profiler = PrefillProfiler(lambda profile_dir: runner)

            outcome = profiler.run(
                root, [plan.cases[0].case_id], session_id="profile-001"
            )

            self.assertEqual("completed", outcome["status"])
            reconstructed = runner.run_case.call_args.args[0]
            self.assertEqual(plan.cases[0].case_id, reconstructed["case_id"])
            self.assertEqual(512, reconstructed["input_len"])
            self.assertTrue(
                (root / "profiles" / "profile-001" / "profile_result.json").is_file()
            )
            self.assertEqual(before, _formal_bytes(root))

    def test_failed_profile_keeps_formal_artifacts_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            plan = _setup(root)
            before = _formal_bytes(root)
            runner = Mock()
            runner.run_case.return_value = {
                "case_id": plan.cases[0].case_id,
                "status": "invalid_reuse",
                "metrics": None,
                "errors": [{"code": "invalid_reuse", "message": "mismatch"}],
            }

            outcome = PrefillProfiler(lambda _profile_dir: runner).run(
                root, [plan.cases[0].case_id], session_id="failed"
            )

            self.assertEqual("failed", outcome["status"])
            self.assertEqual(before, _formal_bytes(root))

    def test_invalid_case_id_creates_no_profile_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            _setup(root)
            before = _formal_bytes(root)

            with self.assertRaisesRegex(PlanningError, "unknown profile case"):
                PrefillProfiler(lambda _profile_dir: Mock()).run(
                    root, ["missing"], session_id="invalid"
                )

            self.assertFalse((root / "profiles" / "invalid").exists())
            self.assertEqual(before, _formal_bytes(root))


if __name__ == "__main__":
    unittest.main()
