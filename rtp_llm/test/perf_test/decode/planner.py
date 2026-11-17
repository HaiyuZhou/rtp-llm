"""Planning for fixed-batch decode grids and distribution workloads."""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from rtp_llm.test.perf_test.config_schema import (
    DecodeDistributionWorkloadConfig,
    DecodeGridWorkloadConfig,
    EngineConfig,
)
from rtp_llm.test.perf_test.decode.distribution import parse_distribution_inputs
from rtp_llm.test.perf_test.decode.model import DecodeCaseSpec, ExactBatchPolicy
from rtp_llm.test.perf_test.decode.scheduler import (
    schedule_decode_requests,
    scheduler_local_batch_size,
)
from rtp_llm.test.perf_test.errors import PlanningError

DecodeWorkloadConfig = DecodeGridWorkloadConfig | DecodeDistributionWorkloadConfig


def _validate_capacity(
    inputs: Sequence[int],
    output: int,
    engine: EngineConfig,
    dp_size: int,
) -> None:
    if any(input_length + output > engine.max_seq_len for input_length in inputs):
        raise PlanningError("decode input plus output exceeds engine.max_seq_len")
    schedule = schedule_decode_requests(inputs, dp_size=dp_size)
    scheduler_local_batch_size(
        tuple(worker.local_batch_size for worker in schedule.workers)
    )
    for worker in schedule.workers:
        if worker.local_batch_size > engine.concurrency_limit:
            raise PlanningError(
                "decode worker-local batch exceeds engine.concurrency_limit"
            )
        if sum(worker.input_lengths) > engine.max_batch_tokens:
            raise PlanningError(
                "decode worker-local input exceeds engine.max_batch_tokens"
            )


def _spec(
    workload: DecodeWorkloadConfig,
    inputs: Sequence[int],
    output: int,
    policy: ExactBatchPolicy,
    distribution: Optional[Dict[str, Any]] = None,
) -> DecodeCaseSpec:
    return DecodeCaseSpec(
        workload_kind=workload.kind,
        input_lengths=tuple(inputs),
        output_length=output,
        batch_policy=policy,
        measure_runs=workload.measure_runs,
        warmup_runs=workload.warmup_runs,
        generate_config=copy.deepcopy(workload.generate_config),
        profiling=workload.profiling,
        distribution=distribution,
        batch_unit=workload.batch.unit,
    )


def validate_decode_frontend_lengths(
    cases: Sequence[Mapping[str, Any]], *, dp_size: int, local_dp_size: int
) -> None:
    """Check each executable batch against its frontend-local DP allocations."""
    if local_dp_size <= 1:
        return
    for case in cases:
        spec = DecodeCaseSpec.from_identity(case)
        workers = schedule_decode_requests(spec.input_lengths, dp_size=dp_size).workers
        for offset in range(0, dp_size, local_dp_size):
            lengths = {
                length
                for worker in workers[offset : offset + local_dp_size]
                for length in worker.input_lengths
            }
            if len(lengths) > 1:
                raise PlanningError(
                    "mixed decode input lengths with multiple DP workers on one "
                    "frontend require explicit DP routing; use one DP worker per "
                    "node or uniform input lengths"
                )


def plan_decode_cases(
    workload: DecodeWorkloadConfig,
    engine: EngineConfig,
    *,
    dp_size: int,
    resolved_workload: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, object], ...]:
    """Freeze ordered decode cases and reject known engine capacity violations."""
    if workload.profiling.enabled and dp_size != 1:
        raise PlanningError("decode profiling requires DP=1")
    cases = []
    if isinstance(workload, DecodeGridWorkloadConfig):
        for input_length in workload.input_lengths:
            for output_length in workload.output_lengths:
                for size in workload.batch.sizes:
                    global_size = (
                        size * dp_size if workload.batch.unit == "per_dp_rank" else size
                    )
                    inputs = (input_length,) * global_size
                    _validate_capacity(
                        inputs,
                        output_length,
                        engine,
                        dp_size,
                    )
                    cases.append(
                        _spec(
                            workload,
                            inputs,
                            output_length,
                            ExactBatchPolicy(strategy="fixed", size=global_size),
                        )
                    )
    elif isinstance(workload, DecodeDistributionWorkloadConfig):
        if resolved_workload is None or "distribution" not in resolved_workload:
            raise PlanningError("resolved distribution content is required")
        materialized = parse_distribution_inputs(
            resolved_workload["distribution"], workload.batch
        )
        for inputs, source_kind in materialized:
            _validate_capacity(
                inputs,
                workload.output_length,
                engine,
                dp_size,
            )
            cases.append(
                _spec(
                    workload,
                    inputs,
                    workload.output_length,
                    ExactBatchPolicy(strategy="fixed", size=len(inputs)),
                    distribution={"source_kind": source_kind},
                )
            )
    else:
        raise PlanningError("decode planner requires grid or distribution workload")
    if not cases:
        raise PlanningError("decode workload generated no cases")
    return tuple(spec.to_identity() for spec in cases)
