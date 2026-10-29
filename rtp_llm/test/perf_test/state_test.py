import signal
import unittest

from rtp_llm.test.perf_test.errors import (
    EngineLifecycleError,
    PlanningError,
    PostExecutionError,
)
from rtp_llm.test.perf_test.state import (
    CaseState,
    RunOutcome,
    RunState,
    StateTransitionError,
    transition_case,
    transition_run,
)


class StateTest(unittest.TestCase):
    def test_valid_run_stage_and_terminal_transitions(self):
        chains = [
            ["created", "planned", "running", "reporting", "completed"],
            [
                "created",
                "planned",
                "running",
                "collecting",
                "reporting",
                "completed",
            ],
            ["created", "planned", "running", "partial"],
            ["created", "planned", "running", "failed"],
            ["created", "planned", "running", "interrupted"],
        ]
        for chain in chains:
            with self.subTest(chain=chain):
                state = RunState(chain[0])
                for target in chain[1:]:
                    state = transition_run(state, RunState(target))
                self.assertEqual(RunState(chain[-1]), state)

    def test_terminal_run_states_and_skipped_stages_reject_invalid_transitions(self):
        invalid = [
            (RunState.CREATED, RunState.RUNNING),
            (RunState.PLANNED, RunState.COMPLETED),
            (RunState.COMPLETED, RunState.RUNNING),
            (RunState.PARTIAL, RunState.COMPLETED),
            (RunState.FAILED, RunState.RUNNING),
            (RunState.INTERRUPTED, RunState.RUNNING),
        ]
        for current, target in invalid:
            with self.subTest(current=current, target=target):
                with self.assertRaises(StateTransitionError):
                    transition_run(current, target)

    def test_case_states_cover_success_and_every_correctness_failure(self):
        terminal_states = (
            CaseState.OK,
            CaseState.REQUEST_FAILED,
            CaseState.INVALID_SHAPE,
            CaseState.INVALID_REUSE,
            CaseState.INVALID_TIMING,
        )
        for terminal in terminal_states:
            with self.subTest(terminal=terminal):
                state = transition_case(CaseState.PLANNED, CaseState.RUNNING)
                state = transition_case(state, terminal)
                self.assertEqual(terminal, state)

    def test_resume_can_rerun_failed_or_invalid_cases_but_not_ok_cases(self):
        retryable = (
            CaseState.REQUEST_FAILED,
            CaseState.INVALID_SHAPE,
            CaseState.INVALID_REUSE,
            CaseState.INVALID_TIMING,
        )
        for state in retryable:
            with self.subTest(state=state):
                self.assertEqual(
                    CaseState.RUNNING, transition_case(state, CaseState.RUNNING)
                )
        with self.assertRaises(StateTransitionError):
            transition_case(CaseState.OK, CaseState.RUNNING)

    def test_exit_code_mapping_covers_all_failure_classes(self):
        cases = [
            (RunOutcome(RunState.COMPLETED), 0),
            (RunOutcome(RunState.PARTIAL), 5),
            (RunOutcome(RunState.FAILED), 5),
            (RunOutcome(RunState.FAILED, PlanningError("plan")), 3),
            (RunOutcome(RunState.FAILED, EngineLifecycleError("engine")), 4),
            (RunOutcome(RunState.PARTIAL, PostExecutionError("report")), 6),
            (RunOutcome(RunState.FAILED, PostExecutionError("publisher")), 6),
            (RunOutcome(RunState.INTERRUPTED, signal_number=signal.SIGINT), 130),
            (RunOutcome(RunState.INTERRUPTED, signal_number=signal.SIGTERM), 143),
        ]
        for outcome, expected in cases:
            with self.subTest(outcome=outcome):
                self.assertEqual(expected, outcome.exit_code)

    def test_exit_code_rejects_incoherent_outcomes(self):
        invalid = [
            RunOutcome(RunState.COMPLETED, EngineLifecycleError("unexpected")),
            RunOutcome(RunState.INTERRUPTED),
            RunOutcome(RunState.RUNNING),
        ]
        for outcome in invalid:
            with self.subTest(outcome=outcome):
                with self.assertRaises(ValueError):
                    _ = outcome.exit_code


if __name__ == "__main__":
    unittest.main()
