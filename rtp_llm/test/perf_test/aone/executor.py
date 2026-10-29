"""Execute one AOne prefill shard through exactly one engine session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

from rtp_llm.test.perf_test.aone.worker import RemoteInputs
from rtp_llm.test.perf_test.execution.engine_session import (
    EngineDiagnostics,
    EngineInterrupted,
)
from rtp_llm.test.perf_test.modes.base import ModeExecutionContext
from rtp_llm.test.perf_test.plan import (
    PLAN_SCHEMA_VERSION,
    AOneResourcePlan,
    ExecutionPlan,
    PlannedCase,
    case_id,
    freeze,
)
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import RunStore, atomic_write_json

SHARD_RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RemoteShardResult:
    run_id: str
    shard_id: str
    state: RunState
    config_fingerprint: str
    workload_fingerprint: str
    owned_case_ids: Tuple[str, ...]
    completed_case_ids: Tuple[str, ...]
    cases: Tuple[Mapping[str, Any], ...]
    engine_diagnostics: EngineDiagnostics
    failure: Optional[Dict[str, str]] = None

    @property
    def exit_code(self) -> int:
        if self.state == RunState.COMPLETED:
            return 0
        if self.state == RunState.INTERRUPTED:
            return 130
        return 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SHARD_RESULT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "shard_id": self.shard_id,
            "mode": "prefill",
            "status": self.state.value,
            "config_fingerprint": self.config_fingerprint,
            "workload_fingerprint": self.workload_fingerprint,
            "owned_case_ids": list(self.owned_case_ids),
            "completed_case_ids": list(self.completed_case_ids),
            "cases": [dict(case) for case in self.cases],
            "engine_diagnostics": self.engine_diagnostics.to_dict(),
            "failure": None if self.failure is None else dict(self.failure),
        }


def _plan_for_shard(inputs: RemoteInputs, output: Path) -> ExecutionPlan:
    execution = inputs.config.get("execution")
    model = inputs.config.get("model")
    if not isinstance(execution, Mapping) or execution.get("backend") != "aone":
        raise ValueError("remote shard requires normalized AOne execution config")
    if not isinstance(model, Mapping) or not isinstance(model.get("topology"), Mapping):
        raise ValueError("remote shard requires normalized model topology")
    topology = model["topology"]
    cards = topology["tp"] * topology["dp"] * topology["pp"]
    resource = AOneResourcePlan(
        backend="aone",
        resource=execution["resource"],
        gpu_type=execution["gpu_type"],
        gpus_per_shard=cards,
        shard_count=execution["shard_count"],
        max_parallel=execution["max_parallel"],
    )
    cases = []
    for item in inputs.owned_cases:
        identity = item.get("identity")
        current_id = item.get("case_id")
        if not isinstance(identity, Mapping) or current_id != case_id(
            "prefill", identity
        ):
            raise ValueError("remote shard case identity does not match its case ID")
        cases.append(PlannedCase(current_id, freeze(identity)))
    return ExecutionPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        mode="prefill",
        action="run",
        backend="aone",
        resolved_config=freeze(inputs.config),
        config_fingerprint=inputs.config_fingerprint,
        workload_fingerprint=inputs.workload_fingerprint,
        cases=tuple(cases),
        resource_plan=resource,
        result_directory=str(output),
    )


def _failure(error: BaseException) -> Dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


class RemoteShardExecutor:
    def __init__(
        self, *, session_factory: Callable[[ExecutionPlan], Any], adapter: Any
    ):
        self._session_factory = session_factory
        self._adapter = adapter

    def execute(
        self, inputs: RemoteInputs, output_directory: Union[str, Path]
    ) -> RemoteShardResult:
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=False)
        (output / "checkpoints").mkdir()
        (output / "results" / "cases").mkdir(parents=True)
        plan = _plan_for_shard(inputs, output)
        store = RunStore(output)
        session = self._session_factory(plan)
        state = RunState.FAILED
        failure: Optional[Dict[str, str]] = None
        try:
            with session as active_session:
                mode_result = self._adapter(
                    ModeExecutionContext(
                        plan=plan,
                        session=active_session,
                        store=store,
                    )
                )
                state = mode_result.state
                if mode_result.failure is not None:
                    failure = _failure(mode_result.failure)
        except EngineInterrupted as error:
            state = RunState.INTERRUPTED
            failure = _failure(error)
        except Exception as error:
            state = RunState.FAILED
            failure = _failure(error)

        committed = store.read_committed_cases()
        completed = tuple(
            case_id for case_id in inputs.owned_case_ids if case_id in committed
        )
        cases = tuple(committed[case_id] for case_id in completed)
        result = RemoteShardResult(
            run_id=inputs.run_id,
            shard_id=inputs.shard_id,
            state=state,
            config_fingerprint=inputs.config_fingerprint,
            workload_fingerprint=inputs.workload_fingerprint,
            owned_case_ids=inputs.owned_case_ids,
            completed_case_ids=completed,
            cases=cases,
            engine_diagnostics=session.diagnostics,
            failure=failure,
        )
        atomic_write_json(output / "shard_result.json", result.to_dict())
        return result
