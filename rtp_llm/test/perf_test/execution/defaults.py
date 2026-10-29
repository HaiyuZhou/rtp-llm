"""Production composition for unified local perf-test execution."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from rtp_llm.test.perf_test.decode.distribution import DistributionDecodeCaseRunner
from rtp_llm.test.perf_test.decode.finalize import finalize_decode_run
from rtp_llm.test.perf_test.decode.fixed import FixedDecodeCaseRunner
from rtp_llm.test.perf_test.decode.http_batch import DecodeHttpBatchRunner
from rtp_llm.test.perf_test.execution.engine_session import EngineSession
from rtp_llm.test.perf_test.execution.local import LocalExecutor
from rtp_llm.test.perf_test.execution.runtime import create_maga_handle
from rtp_llm.test.perf_test.modes.decode import DecodeAdapter
from rtp_llm.test.perf_test.modes.prefill import PrefillAdapter
from rtp_llm.test.perf_test.plan import ExecutionPlan, thaw
from rtp_llm.test.perf_test.prefill.checkpoint import PrefillResume
from rtp_llm.test.perf_test.prefill.finalize import finalize_prefill_run
from rtp_llm.test.perf_test.prefill.profile import PrefillProfiler
from rtp_llm.test.perf_test.prefill.report import generate_prefill_report
from rtp_llm.test.perf_test.prefill.runner import PrefillRunner
from rtp_llm.test.perf_test.run_directory import RunDirectoryLock, inspect_run_directory
from rtp_llm.test.perf_test.storage import RunStore

SessionFactory = Callable[[ExecutionPlan], Any]
TokenizerLoader = Callable[[ExecutionPlan], Any]
QueryBuilder = Callable[[ExecutionPlan, Sequence[int]], Mapping[int, str]]


def _default_session_factory(plan: ExecutionPlan) -> EngineSession:
    return EngineSession(
        launcher=lambda: create_maga_handle(plan),
        readiness_timeout_seconds=600,
        timeline_collector=lambda: (),
        coredump_collector=lambda: {},
    )


def _default_tokenizer_loader(plan: ExecutionPlan) -> Any:
    from rtp_llm.test.perf_test.test_util import _load_tokenizer

    model = thaw(plan.resolved_config)["model"]
    return _load_tokenizer(
        model["tokenizer"],
        checkpoint_path=model["checkpoint"],
        model_type=model["type"],
    )


def _default_query_builder(
    plan: ExecutionPlan, input_lengths: Sequence[int]
) -> Mapping[int, str]:
    from rtp_llm.test.perf_test.test_util import create_query

    model = thaw(plan.resolved_config)["model"]
    return create_query(model["tokenizer"], sorted(set(input_lengths)))


def _write_manifest_state(root: Path, state: str) -> Dict[str, Any]:
    snapshot = inspect_run_directory(root)
    manifest = copy.deepcopy(snapshot.manifest)
    manifest["state"] = state
    RunStore(root).write_manifest(manifest)
    return manifest


class UnifiedLocalExecutor:
    """Compose local actions without leaking config parsing into mode runners."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory = _default_session_factory,
        tokenizer_loader: TokenizerLoader = _default_tokenizer_loader,
        query_builder: QueryBuilder = _default_query_builder,
        selected_case_ids: Sequence[str] = (),
    ):
        self._session_factory = session_factory
        self._tokenizer_loader = tokenizer_loader
        self._query_builder = query_builder
        self._selected_case_ids = tuple(selected_case_ids)

    def _prefill_adapter(self, plan: ExecutionPlan) -> PrefillAdapter:
        tokenizer = self._tokenizer_loader(plan)
        commit_tail = int(thaw(plan.resolved_config)["workload"]["commit_tail_tokens"])
        return PrefillAdapter(
            lambda context: PrefillRunner(
                context.session.client,
                tokenizer,
                commit_tail_tokens=commit_tail,
            )
        )

    def _decode_adapter(self, plan: ExecutionPlan) -> DecodeAdapter:
        config = thaw(plan.resolved_config)
        workload_kind = config["workload"]["kind"]
        topology = config["model"]["topology"]
        input_lengths = []
        for planned_case in plan.cases:
            identity = thaw(planned_case.identity)
            if workload_kind == "distribution":
                input_lengths.extend(identity["input_lengths"])
            else:
                input_lengths.append(identity["input_length"])
        prompts = self._query_builder(plan, input_lengths)

        def runner_factory(context):
            batch_runner = DecodeHttpBatchRunner(
                context.session.client.port,
                prompt_for_length=lambda length: prompts[length],
            )
            runner_type = (
                DistributionDecodeCaseRunner
                if workload_kind == "distribution"
                else FixedDecodeCaseRunner
            )
            return runner_type(batch_runner, topology=topology)

        return DecodeAdapter(runner_factory)

    def _execute_measurement(self, plan: ExecutionPlan):
        if plan.mode == "prefill":
            adapter = self._prefill_adapter(plan)
            if plan.action == "resume":
                adapter = PrefillResume(adapter)
        else:
            adapter = self._decode_adapter(plan)
        return LocalExecutor(
            session_factory=self._session_factory,
            store_factory=RunStore,
            adapters={plan.mode: adapter},
        ).execute(plan)

    def __call__(self, plan: ExecutionPlan):
        root = Path(plan.result_directory)
        if plan.action == "report":
            return generate_prefill_report(root)
        if plan.action == "profile":
            tokenizer = self._tokenizer_loader(plan)
            commit_tail = int(
                thaw(plan.resolved_config)["workload"]["commit_tail_tokens"]
            )
            selected = self._selected_case_ids or tuple(
                case.case_id for case in plan.cases
            )
            session_id = datetime.now(timezone.utc).strftime("profile-%Y%m%dT%H%M%S%fZ")
            with self._session_factory(plan) as session:
                profiler = PrefillProfiler(
                    lambda _profile_dir: PrefillRunner(
                        session.client,
                        tokenizer,
                        commit_tail_tokens=commit_tail,
                    )
                )
                return profiler.run(root, selected, session_id=session_id)

        if plan.action == "run":
            with RunDirectoryLock(root, action=f"{plan.mode}-run"):
                snapshot = inspect_run_directory(root)
                _write_manifest_state(root, "running")
                outcome = self._execute_measurement(plan)
                if plan.mode == "prefill":
                    finalize_prefill_run(
                        root,
                        plan=plan,
                        run_id=snapshot.manifest["run_id"],
                        case_results=outcome.case_results,
                    )
                    generate_prefill_report(root, acquire_lock=False)
                else:
                    finalize_decode_run(
                        root,
                        plan=plan,
                        run_id=snapshot.manifest["run_id"],
                        case_results=outcome.case_results,
                    )
                _write_manifest_state(root, outcome.state.value)
                return outcome

        snapshot = inspect_run_directory(root)
        _write_manifest_state(root, "running")
        outcome = self._execute_measurement(plan)
        finalize_prefill_run(
            root,
            plan=plan,
            run_id=snapshot.manifest["run_id"],
            case_results=outcome.case_results,
        )
        generate_prefill_report(root)
        _write_manifest_state(root, outcome.state.value)
        return outcome


def create_default_local_executor(
    *, selected_case_ids: Sequence[str] = ()
) -> UnifiedLocalExecutor:
    return UnifiedLocalExecutor(selected_case_ids=selected_case_ids)
