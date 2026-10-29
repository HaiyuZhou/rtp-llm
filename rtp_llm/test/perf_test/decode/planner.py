"""Planning for fixed input-length by global-batch decode grids."""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Tuple

from rtp_llm.test.perf_test.config_schema import DecodeGridWorkloadConfig, EngineConfig
from rtp_llm.test.perf_test.errors import PlanningError


def plan_decode_grid_cases(
    workload: DecodeGridWorkloadConfig,
    engine: EngineConfig,
    *,
    dp_size: int,
) -> Tuple[Dict[str, object], ...]:
    """Expand one scalar output length across input and global-batch axes."""
    if not isinstance(workload, DecodeGridWorkloadConfig) or workload.kind != "grid":
        raise PlanningError("fixed decode planner requires workload.kind=grid")
    if type(dp_size) is not int or dp_size <= 0:
        raise PlanningError("decode dp_size must be a positive integer")
    cases: List[Dict[str, object]] = []
    for global_batch_size in workload.batch_sizes:
        local_batch_size = math.ceil(global_batch_size / dp_size)
        if local_batch_size > engine.concurrency_limit:
            raise PlanningError(
                "decode worker-local batch exceeds engine.concurrency_limit"
            )
        for input_length in workload.input_lengths:
            if input_length + workload.output_length > engine.max_seq_len:
                raise PlanningError(
                    "decode input plus output length exceeds engine.max_seq_len"
                )
            if local_batch_size * input_length > engine.max_batch_tokens:
                raise PlanningError(
                    "decode worker-local input exceeds engine.max_batch_tokens"
                )
            cases.append(
                {
                    "workload_kind": "grid",
                    "input_length": input_length,
                    "output_length": workload.output_length,
                    "global_batch_size": global_batch_size,
                    "measure_runs": workload.measure_runs,
                    "warmup_runs": workload.warmup_runs,
                    "generate_config": copy.deepcopy(workload.generate_config),
                    "target_tpot_ms": workload.target_tpot_ms,
                }
            )
    if not cases:
        raise PlanningError("fixed decode workload generated no cases")
    return tuple(cases)
