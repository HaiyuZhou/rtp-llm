"""Tests for execution local_test."""

import json
import signal
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.errors import PlanningError, ResourceError
from rtp_llm.test.perf_test.execution.engine_session import (
    EngineInterrupted,
    EngineSession,
)
from rtp_llm.test.perf_test.execution.local import LocalExecutor, UnifiedLocalExecutor
from rtp_llm.test.perf_test.modes import ModeExecutionResult, PrefillAdapter
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.run_directory import (
    RunDirectoryLock,
    initialize_run_directory,
)
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.tests.support.execution import (
    default_executor_decode_source_and_plan,
    default_executor_FailingSession,
    default_executor_measurement,
    default_executor_Session,
    default_executor_source,
    default_executor_Tokenizer,
    engine_session_FakeHandle,
    local_executor_FakeSession,
    local_executor_plan,
)


class DefaultExecutorTest(unittest.TestCase):

    def test_signal_interrupt_keeps_signal_and_closes_real_engine_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            plan = replace(self._resume_plan(root), action="run")
            handle = engine_session_FakeHandle()
            session = EngineSession(
                launcher=lambda: handle,
                readiness_timeout_seconds=30,
                timeline_collector=lambda: (),
                coredump_collector=lambda: {},
            )
            executor = UnifiedLocalExecutor(session_factory=lambda _: session)
            adapter = Mock(side_effect=EngineInterrupted(signal.SIGTERM))
            with patch.object(executor, "_prefill_adapter", return_value=adapter):
                outcome = executor(plan)
            self.assertEqual(RunState.INTERRUPTED, outcome.state)
            self.assertEqual(128 + signal.SIGTERM, outcome.exit_code)
            self.assertEqual(
                signal.SIGTERM,
                json.loads((root / "manifest.json").read_text())["signal_number"],
            )
            self.assertEqual(1, handle.stop_calls)
            with RunDirectoryLock(root, action="verify-released"):
                pass

    def test_invalid_resume_preserves_run_before_loading_or_starting_engine(self):
        for invalid in ("missing", "mismatched"):
            with self.subTest(
                invalid=invalid
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "run"
                plan = self._resume_plan(root)
                checkpoint = root / "checkpoints/prefill_progress.json"
                if invalid == "missing":
                    checkpoint.unlink()
                else:
                    value = json.loads(checkpoint.read_text())
                    value["workload_fingerprint"] = "invalid"
                    checkpoint.write_text(json.dumps(value))
                before = {
                    str(p.relative_to(root)): p.read_bytes()
                    for p in root.rglob("*")
                    if p.is_file() and p.name != ".writer.lock"
                }
                factory = Mock(return_value=default_executor_Session())
                tokenizer = Mock(return_value=default_executor_Tokenizer())
                executor = UnifiedLocalExecutor(
                    session_factory=factory, tokenizer_loader=tokenizer
                )
                with self.assertRaises(PlanningError):
                    executor(plan)
                factory.assert_not_called()
                tokenizer.assert_not_called()
                self.assertEqual(
                    before,
                    {
                        str(p.relative_to(root)): p.read_bytes()
                        for p in root.rglob("*")
                        if p.is_file() and p.name != ".writer.lock"
                    },
                )

    def test_preflight_probe_and_interrupt_leave_terminal_state_and_close_engine(self):
        for stage, error, state in (
            ("preflight", ValueError("invalid preflight"), "failed"),
            ("probe", ValueError("invalid_reuse"), "failed"),
            ("preflight", KeyboardInterrupt(), "interrupted"),
        ):
            with self.subTest(
                stage=stage, state=state
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "run"
                plan = replace(self._resume_plan(root), action="run")
                session = default_executor_Session()
                executor = UnifiedLocalExecutor(session_factory=lambda _: session)
                runner = Mock()
                probe = Mock(return_value={"status": "ok"})
                if stage == "preflight":
                    runner.preflight.side_effect = error
                else:
                    probe.side_effect = error
                adapter = PrefillAdapter(lambda _: runner, probe=probe)
                with patch.object(executor, "_prefill_adapter", return_value=adapter):
                    with self.assertRaises(type(error)) as raised:
                        executor(plan)
                self.assertIs(error, raised.exception)
                self.assertEqual((1, 1), (session.entered, session.exited))
                manifest = json.loads((root / "manifest.json").read_text())
                self.assertEqual(state, manifest["state"])
                self.assertEqual(type(error).__name__, manifest["error"]["type"])
                with RunDirectoryLock(root, action="verify-released"):
                    pass

    def test_report_failure_leaves_failed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            plan = replace(self._resume_plan(root), action="run")
            session = default_executor_Session()
            executor = UnifiedLocalExecutor(
                session_factory=lambda _: session,
                tokenizer_loader=lambda _: default_executor_Tokenizer(),
            )
            with patch(
                "rtp_llm.test.perf_test.execution.local.generate_prefill_report",
                side_effect=OSError("report write failed"),
            ):
                with self.assertRaisesRegex(OSError, "report write failed"):
                    executor(plan)
            self.assertEqual(
                "failed", json.loads((root / "manifest.json").read_text())["state"]
            )
            self.assertEqual((1, 1), (session.entered, session.exited))

    def _resume_plan(self, root):
        source = default_executor_source(root.parent)
        plan = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=resolve_config(source),
            cases=[
                {
                    "measurement_contract": "concurrent_batch_v1",
                    "input_len": 16,
                    "cache_len": 0,
                    "compute_len": 16,
                    "batch_size": 1,
                    "measure_runs": 1,
                    "seed": 1,
                    "prefix_policy": "shared_seed",
                }
            ],
            result_directory=str(root),
        )
        initialize_run_directory(
            root, source_config=source, plan=plan, run_id="resume", code_version="test"
        )
        return replace(plan, action="resume")

    def test_resume_lock_rejection_preserves_artifacts_and_does_not_start_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            plan = self._resume_plan(root)
            session = default_executor_Session()
            executor = UnifiedLocalExecutor(
                session_factory=lambda _plan: session,
                tokenizer_loader=lambda _plan: default_executor_Tokenizer(),
            )
            with RunDirectoryLock(root, action="active-profile"):
                before = {
                    str(path.relative_to(root)): path.read_bytes()
                    for path in root.rglob("*")
                    if path.is_file() and path.name != ".writer.lock"
                }
                with self.assertRaises(ResourceError):
                    executor(plan)
                self.assertEqual(
                    before,
                    {
                        str(path.relative_to(root)): path.read_bytes()
                        for path in root.rglob("*")
                        if path.is_file() and path.name != ".writer.lock"
                    },
                )
                self.assertEqual(0, session.entered)

    def test_resume_holds_lock_through_finalization_and_report(self):
        from rtp_llm.test.perf_test.prefill.finalize import finalize_prefill_run
        from rtp_llm.test.perf_test.prefill.report import generate_prefill_report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            plan = self._resume_plan(root)
            session = default_executor_Session()
            executor = UnifiedLocalExecutor(
                session_factory=lambda _plan: session,
                tokenizer_loader=lambda _plan: default_executor_Tokenizer(),
            )
            visited = []

            def check_lock(stage):
                with self.assertRaises(ResourceError):
                    with RunDirectoryLock(root, action="second-writer"):
                        pass
                visited.append(stage)

            def finalize(*args, **kwargs):
                check_lock("finalize")
                return finalize_prefill_run(*args, **kwargs)

            def report(*args, **kwargs):
                check_lock("report")
                return generate_prefill_report(*args, **kwargs)

            with patch(
                "rtp_llm.test.perf_test.execution.local.finalize_prefill_run", finalize
            ), patch(
                "rtp_llm.test.perf_test.execution.local.generate_prefill_report", report
            ):
                outcome = executor(plan)
            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertEqual(["finalize", "report"], visited)
            self.assertEqual(
                "completed", json.loads((root / "manifest.json").read_text())["state"]
            )
            self.assertTrue((root / "results/result.json").is_file())
            with RunDirectoryLock(root, action="verify-released"):
                pass

    def test_decode_run_uses_one_session_and_unified_case_executor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            (source, plan) = default_executor_decode_source_and_plan(root)
            initialize_run_directory(
                root,
                source_config=source,
                plan=plan,
                run_id="decode-local",
                code_version="test",
            )
            session = default_executor_Session()
            session.log_path = None
            batch_executor = Mock()
            batch_executor.measure.return_value = default_executor_measurement(
                batch=1, inputs=(32,), output=8
            )
            executor = UnifiedLocalExecutor(
                session_factory=Mock(return_value=session),
                tokenizer_loader=lambda _plan: default_executor_Tokenizer(),
                query_builder=lambda _plan, lengths: {
                    length: f"prompt-{length}" for length in lengths
                },
                decode_batch_executor_factory=Mock(return_value=batch_executor),
            )
            outcome = executor(plan)
            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertEqual((1, 1), (session.entered, session.exited))
            self.assertEqual("ok", outcome.case_results[0]["status"])
            self.assertEqual(8, outcome.case_results[0]["metrics"]["output_length"])
            self.assertTrue((root / "reports" / "decode_report.json").is_file())

    def test_prefill_run_uses_one_session_and_finalizes_all_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            source = default_executor_source(root.parent)
            plan = build_execution_plan(
                mode="prefill",
                action="run",
                resolved=resolve_config(source),
                cases=[
                    {
                        "input_len": 16,
                        "cache_len": 0,
                        "compute_len": 16,
                        "batch_size": 1,
                        "measure_runs": 1,
                        "seed": 1,
                        "prefix_policy": "shared_seed",
                    }
                ],
                result_directory=str(root),
            )
            initialize_run_directory(
                root,
                source_config=source,
                plan=plan,
                run_id="run-local",
                code_version="test",
            )
            session = default_executor_Session()
            executor = UnifiedLocalExecutor(
                session_factory=Mock(return_value=session),
                tokenizer_loader=lambda _plan: default_executor_Tokenizer(),
                query_builder=lambda _plan, _lengths: {},
            )
            outcome = executor(plan)
            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertEqual(1, session.entered)
            self.assertEqual(1, session.exited)
            self.assertTrue(
                (root / "checkpoints" / "prefill_cache_probe.json").is_file()
            )
            self.assertTrue((root / "results" / "result.json").is_file())
            self.assertTrue((root / "results" / "summary.json").is_file())
            self.assertTrue((root / "reports" / "prefill_report.json").is_file())
            self.assertTrue((root / "reports" / "latency.interactive.html").is_file())
            self.assertTrue((root / "reports" / "tpm.interactive.html").is_file())
            self.assertTrue(
                (root / "reports" / "formula" / "fit_report.json").is_file()
            )

    def test_engine_failure_preserves_typed_outcome_without_finalizing_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            source = default_executor_source(root.parent)
            plan = build_execution_plan(
                mode="prefill",
                action="run",
                resolved=resolve_config(source),
                cases=[
                    {
                        "input_len": 16,
                        "cache_len": 0,
                        "compute_len": 16,
                        "batch_size": 1,
                        "measure_runs": 1,
                        "seed": 1,
                        "prefix_policy": "shared_seed",
                    }
                ],
                result_directory=str(root),
            )
            initialize_run_directory(
                root,
                source_config=source,
                plan=plan,
                run_id="run-engine-failure",
                code_version="test",
            )
            executor = UnifiedLocalExecutor(
                session_factory=Mock(return_value=default_executor_FailingSession()),
                tokenizer_loader=lambda _plan: default_executor_Tokenizer(),
                query_builder=lambda _plan, _lengths: {},
            )
            outcome = executor(plan)
            self.assertEqual(RunState.FAILED, outcome.state)
            self.assertEqual(4, outcome.exit_code)
            self.assertFalse((root / "results" / "result.json").exists())
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual("failed", manifest["state"])


class LocalExecutorTest(unittest.TestCase):

    def test_prefill_and_decode_adapters_receive_one_session_and_frozen_plan(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                plan = local_executor_plan(mode)
                session = local_executor_FakeSession()
                adapter = Mock(
                    return_value=ModeExecutionResult(
                        state=RunState.COMPLETED,
                        case_results=({"case_id": plan.cases[0].case_id},),
                    )
                )
                other_adapter = Mock()
                adapters = {
                    mode: adapter,
                    "decode" if mode == "prefill" else "prefill": other_adapter,
                }
                session_factory = Mock(return_value=session)
                store = object()
                store_factory = Mock(return_value=store)
                executor = LocalExecutor(
                    session_factory=session_factory,
                    store_factory=store_factory,
                    adapters=adapters,
                )
                with patch(
                    "rtp_llm.test.perf_test.config_resolver.load_resolved_config",
                    side_effect=AssertionError("source config must not be parsed"),
                ) as source_loader:
                    outcome = executor.execute(plan)
                self.assertEqual(0, outcome.exit_code)
                self.assertEqual("process.log", outcome.engine_diagnostics.log_path)
                session_factory.assert_called_once_with(plan)
                store_factory.assert_called_once_with(plan.result_directory)
                adapter.assert_called_once()
                context = adapter.call_args.args[0]
                self.assertIs(plan, context.plan)
                self.assertIs(session, context.session)
                self.assertIs(store, context.store)
                self.assertFalse(hasattr(context, "source_config"))
                self.assertFalse(hasattr(context, "config_path"))
                other_adapter.assert_not_called()
                source_loader.assert_not_called()
                self.assertEqual(1, session.enter_calls)
                self.assertEqual(1, session.exit_calls)


if __name__ == "__main__":
    unittest.main()
