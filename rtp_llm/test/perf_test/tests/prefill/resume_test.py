"""Tests for prefill resume_test."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.errors import PlanningError, ResourceError
from rtp_llm.test.perf_test.modes import ModeExecutionContext, PrefillAdapter
from rtp_llm.test.perf_test.prefill.case_store import (
    append_prefill_case,
    read_prefill_cases,
)
from rtp_llm.test.perf_test.prefill.checkpoint import PrefillResume
from rtp_llm.test.perf_test.run_directory import RunDirectoryLock
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import RunStore
from rtp_llm.test.perf_test.tests.support.fixtures import (
    prefill_resume_config,
    prefill_resume_plan,
)
from rtp_llm.test.perf_test.tests.support.run_storage import (
    prefill_adapter_plan,
    prefill_adapter_prepare_store,
    prefill_adapter_result,
    prefill_resume_case_result,
    prefill_resume_initialized,
    prefill_resume_tree,
)


class PrefillResumeTest(unittest.TestCase):

    def test_resume_preserves_ok_bytes_and_reruns_every_other_case(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            (run_plan, store) = prefill_resume_initialized(root)
            ok_path = root / "results" / "cases.jsonl"
            ok_before = ok_path.read_bytes()
            runner = Mock()
            runner.run_case.side_effect = lambda case: prefill_resume_case_result(
                next(
                    (item for item in run_plan.cases if item.case_id == case["case_id"])
                ),
                "ok",
            )
            resume = PrefillResume(PrefillAdapter(lambda _context: runner))
            outcome = resume(
                ModeExecutionContext(
                    plan=prefill_resume_plan(root, action="resume"),
                    session=object(),
                    store=store,
                )
            )
            self.assertEqual(RunState.COMPLETED, outcome.state)
            runner.run_case.assert_called_once()
            self.assertEqual(
                run_plan.cases[1].case_id, runner.run_case.call_args.args[0]["case_id"]
            )
            self.assertTrue(ok_path.read_bytes().startswith(ok_before))
            self.assertTrue(
                all(
                    (
                        case["status"] == "ok"
                        for case in store.read_prefill_cases(run_plan).values()
                    )
                )
            )

    def test_fingerprint_mismatch_causes_no_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            (_, store) = prefill_resume_initialized(root)
            before = prefill_resume_tree(root)
            resume = PrefillResume(PrefillAdapter(lambda _context: Mock()))
            mismatched = prefill_resume_plan(
                root, action="resume", config=prefill_resume_config({"reuse_cache": 0})
            )
            with self.assertRaisesRegex(PlanningError, "config_fingerprint"):
                resume(
                    ModeExecutionContext(plan=mismatched, session=object(), store=store)
                )
            self.assertEqual(before, prefill_resume_tree(root))

    def test_concurrent_resume_is_rejected_without_runner_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            (_, store) = prefill_resume_initialized(root)
            runner = Mock()
            resume = PrefillResume(PrefillAdapter(lambda _context: runner))
            with RunDirectoryLock(root, action="resume"):
                before = prefill_resume_tree(root)
                with self.assertRaises(ResourceError):
                    resume(
                        ModeExecutionContext(
                            plan=prefill_resume_plan(root, action="resume"),
                            session=object(),
                            store=store,
                        )
                    )
                self.assertEqual(before, prefill_resume_tree(root))
            runner.assert_not_called()


class PrefillAdapterTest(unittest.TestCase):

    def test_failed_case_is_committed_and_independent_case_continues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = prefill_adapter_plan(root)
            runner = Mock()
            runner.run_case.side_effect = lambda case: prefill_adapter_result(
                case, "request_failed" if case["cache_len"] == 0 else "ok"
            )
            adapter = PrefillAdapter(lambda _context: runner)
            outcome = adapter(
                ModeExecutionContext(
                    plan=plan,
                    session=object(),
                    store=prefill_adapter_prepare_store(root),
                )
            )
            self.assertEqual(RunState.PARTIAL, outcome.state)
            self.assertEqual(2, runner.run_case.call_count)
            committed = RunStore(root).read_prefill_cases(plan)
            self.assertEqual([case.case_id for case in plan.cases], list(committed))
            self.assertEqual(
                2, len((root / "results/cases.jsonl").read_text().splitlines())
            )

    def test_interrupt_preserves_only_already_committed_cases_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = prefill_adapter_plan(root)
            runner = Mock()
            calls = 0

            def execute(case):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt()
                return prefill_adapter_result(case, "ok")

            runner.run_case.side_effect = execute
            adapter = PrefillAdapter(lambda _context: runner)
            store = prefill_adapter_prepare_store(root)
            with self.assertRaises(KeyboardInterrupt):
                adapter(ModeExecutionContext(plan=plan, session=object(), store=store))
            committed = store.read_prefill_cases(plan)
            self.assertEqual([plan.cases[0].case_id], list(committed))


class PrefillCaseStoreTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "results").mkdir()
        self.path = self.root / "results" / "cases.jsonl"
        self.planned = {"one": {"shape": [1, 2]}, "two": {"shape": [3]}}
        self.one = {"case_id": "one", "identity": {"shape": [1, 2]}, "status": "ok"}
        self.two = {"case_id": "two", "identity": {"shape": [3]}, "status": "failed"}

    def test_torn_tail_is_ignored_and_repaired_before_append(self):
        append_prefill_case(self.path, "one", self.one)
        with self.path.open("ab") as stream:
            stream.write(b'{"case_id":"broken')
        self.assertEqual({"one": self.one}, read_prefill_cases(self.path, self.planned))
        append_prefill_case(self.path, "two", self.two)
        self.assertEqual(
            {"one": self.one, "two": self.two},
            read_prefill_cases(self.path, self.planned),
        )
        self.assertEqual(2, len(self.path.read_bytes().splitlines()))


if __name__ == "__main__":
    unittest.main()
