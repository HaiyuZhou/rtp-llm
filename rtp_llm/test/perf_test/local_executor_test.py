import unittest
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.errors import CaseExecutionError, EngineLifecycleError
from rtp_llm.test.perf_test.execution.engine_session import EngineDiagnostics
from rtp_llm.test.perf_test.execution.local import LocalExecutor
from rtp_llm.test.perf_test.modes.base import ModeExecutionResult
from rtp_llm.test.perf_test.orchestrator import Orchestrator
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.state import RunState


def _config(mode):
    workload = {
        "kind": "cache_grid",
        "max_input_len": 2048,
        "cache_alignment": 64,
        "commit_tail_tokens": 64,
        "compute_steps": [128],
        "batch_sizes": [1],
    }
    if mode == "decode":
        workload = {
            "kind": "grid",
            "input_lengths": [128],
            "output_length": 16,
            "batch_sizes": [1],
        }
    return {
        "schema_version": 1,
        "model": {"label": "test", "type": "registered", "checkpoint": "/m"},
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 8,
        },
        "workload": workload,
        "execution": {"backend": "local", "devices": [0]},
        "output": {},
        "regression": {},
        "publish": {},
    }


def _plan(mode):
    identity = {"input_len": 512, "cache_len": 0, "batch_size": 1}
    if mode == "decode":
        identity = {"input_len": 128, "output_len": 16, "batch_size": 1}
    return build_execution_plan(
        mode=mode,
        action="run",
        resolved=resolve_config(_config(mode)),
        cases=[identity],
        result_directory=f"runs/{mode}",
    )


class _FakeSession:
    def __init__(self, enter_error=None):
        self.enter_error = enter_error
        self.enter_calls = 0
        self.exit_calls = 0
        self.diagnostics = EngineDiagnostics(log_path="process.log")

    def __enter__(self):
        self.enter_calls += 1
        if self.enter_error is not None:
            raise self.enter_error
        return self

    def __exit__(self, _type, _value, _traceback):
        self.exit_calls += 1


class LocalExecutorTest(unittest.TestCase):
    def test_prefill_and_decode_adapters_receive_one_session_and_frozen_plan(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                plan = _plan(mode)
                session = _FakeSession()
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

    def test_engine_failure_becomes_typed_failed_outcome(self):
        plan = _plan("prefill")
        failure = EngineLifecycleError("startup")
        session = _FakeSession(enter_error=failure)
        executor = LocalExecutor(
            session_factory=Mock(return_value=session),
            store_factory=Mock(return_value=object()),
            adapters={"prefill": Mock(), "decode": Mock()},
        )

        outcome = executor.execute(plan)

        self.assertEqual(RunState.FAILED, outcome.state)
        self.assertEqual(4, outcome.exit_code)

    def test_partial_adapter_outcome_keeps_case_failure_exit_class(self):
        plan = _plan("decode")
        failure = CaseExecutionError("one request failed")
        adapter = Mock(
            return_value=ModeExecutionResult(
                state=RunState.PARTIAL,
                case_results=({"case_id": plan.cases[0].case_id},),
                failure=failure,
            )
        )
        executor = LocalExecutor(
            session_factory=Mock(return_value=_FakeSession()),
            store_factory=Mock(return_value=object()),
            adapters={"prefill": Mock(), "decode": adapter},
        )

        outcome = executor.execute(plan)

        self.assertEqual(RunState.PARTIAL, outcome.state)
        self.assertEqual(5, outcome.exit_code)

    def test_orchestrator_dispatches_only_by_immutable_plan_backend(self):
        plan = _plan("prefill")
        local = Mock(return_value="local-result")
        aone = Mock(return_value="aone-result")
        orchestrator = Orchestrator({"local": local, "aone": aone})

        result = orchestrator.execute(plan)

        self.assertEqual("local-result", result)
        local.assert_called_once_with(plan)
        aone.assert_not_called()


if __name__ == "__main__":
    unittest.main()
