"""Planning and case execution for materialized decode distributions."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Dict, List, Tuple

from rtp_llm.test.perf_test.config_schema import EngineConfig
from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
from rtp_llm.test.perf_test.decode.scheduler import schedule_decode_requests
from rtp_llm.test.perf_test.errors import PlanningError


def _lengths(path: str, value: Any) -> Tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise PlanningError(f"{path} must be a non-empty array")
    parsed = []
    for index, item in enumerate(value):
        if type(item) is not int or item <= 0:
            raise PlanningError(f"{path}[{index}] must be a positive integer")
        parsed.append(item)
    return tuple(parsed)


def _distribution_cases(
    distribution: Any, batch_sizes: Sequence[int]
) -> List[Tuple[Tuple[int, ...], str]]:
    if isinstance(distribution, list):
        sequence_lengths = _lengths("distribution", distribution)
        source_kind = "sequence_lengths"
    elif isinstance(distribution, Mapping) and set(distribution) == {
        "sequence_lengths"
    }:
        sequence_lengths = _lengths(
            "distribution.sequence_lengths", distribution["sequence_lengths"]
        )
        source_kind = "sequence_lengths"
    else:
        sequence_lengths = ()
        source_kind = ""
    if sequence_lengths:
        if len(sequence_lengths) < max(batch_sizes):
            raise PlanningError(
                "distribution does not contain enough sequence lengths for batch_sizes"
            )
        return [
            (sequence_lengths[:batch_size], source_kind) for batch_size in batch_sizes
        ]

    if not isinstance(distribution, Mapping) or set(distribution) != {
        "batch_seq_len_map"
    }:
        raise PlanningError(
            "distribution must contain sequence_lengths or batch_seq_len_map"
        )
    raw_map = distribution["batch_seq_len_map"]
    if not isinstance(raw_map, Mapping) or not raw_map:
        raise PlanningError("distribution.batch_seq_len_map must be a non-empty object")
    parsed: Dict[int, Tuple[int, ...]] = {}
    for key, value in raw_map.items():
        if not isinstance(key, str) or not key.isdecimal() or int(key) <= 0:
            raise PlanningError("distribution batch map keys must be positive integers")
        batch_size = int(key)
        lengths = _lengths(f"distribution.batch_seq_len_map.{key}", value)
        if len(lengths) != batch_size:
            raise PlanningError(
                f"distribution batch {batch_size} does not match its sequence count"
            )
        parsed[batch_size] = lengths
    if set(parsed) != set(batch_sizes):
        raise PlanningError(
            "distribution batch map keys must exactly match workload.batch_sizes"
        )
    return [(parsed[batch_size], "batch_seq_len_map") for batch_size in sorted(parsed)]


def plan_decode_distribution_cases(
    workload: Mapping[str, Any],
    engine: EngineConfig,
    *,
    dp_size: int,
) -> Tuple[Dict[str, object], ...]:
    """Validate materialized distribution content and freeze actual lengths."""
    if workload.get("kind") != "distribution":
        raise PlanningError("distribution planner requires workload.kind=distribution")
    if type(dp_size) is not int or dp_size <= 0:
        raise PlanningError("decode dp_size must be a positive integer")
    batch_sizes = workload.get("batch_sizes")
    if not isinstance(batch_sizes, (list, tuple)) or not batch_sizes:
        raise PlanningError("distribution workload.batch_sizes must be non-empty")
    if any(type(item) is not int or item <= 0 for item in batch_sizes):
        raise PlanningError("distribution batch_sizes must be positive integers")
    output_length = workload.get("output_length")
    if type(output_length) is not int or output_length <= 0:
        raise PlanningError("distribution output_length must be one positive scalar")
    materialized = _distribution_cases(workload.get("distribution"), batch_sizes)
    cases: List[Dict[str, object]] = []
    for input_lengths, source_kind in materialized:
        global_batch_size = len(input_lengths)
        local_batch_size = math.ceil(global_batch_size / dp_size)
        if local_batch_size > engine.concurrency_limit:
            raise PlanningError(
                "distribution worker-local batch exceeds engine.concurrency_limit"
            )
        if any(length + output_length > engine.max_seq_len for length in input_lengths):
            raise PlanningError(
                "distribution input plus output length exceeds engine.max_seq_len"
            )
        schedule = schedule_decode_requests(input_lengths, dp_size=dp_size)
        if any(
            sum(worker.input_lengths) > engine.max_batch_tokens
            for worker in schedule.workers
        ):
            raise PlanningError(
                "distribution worker-local input exceeds engine.max_batch_tokens"
            )
        cases.append(
            {
                "workload_kind": "distribution",
                "input_lengths": input_lengths,
                "output_length": output_length,
                "global_batch_size": global_batch_size,
                "measure_runs": int(workload.get("measure_runs", 3)),
                "warmup_runs": int(workload.get("warmup_runs", 1)),
                "generate_config": copy.deepcopy(workload.get("generate_config", {})),
                "target_tpot_ms": workload.get("target_tpot_ms"),
                "distribution": {"source_kind": source_kind},
            }
        )
    return tuple(cases)


class DistributionDecodeCaseRunner:
    def __init__(
        self,
        batch_runner: Callable[..., Mapping[str, Any]],
        *,
        topology: Mapping[str, int] | None = None,
    ):
        self._batch_runner = batch_runner
        self._topology = dict(topology or {"tp": 1, "dp": 1, "pp": 1, "ep": 1})
        if set(self._topology) != {"tp", "dp", "pp", "ep"} or any(
            type(value) is not int or value <= 0 for value in self._topology.values()
        ):
            raise ValueError("decode topology requires positive tp/dp/pp/ep values")

    def run_case(self, case: Mapping[str, Any]) -> Dict[str, Any]:
        input_lengths = tuple(int(item) for item in case["input_lengths"])
        output_length = case["output_length"]
        if type(output_length) is not int or output_length <= 0:
            raise ValueError("distribution output_length must be one scalar")
        schedule = schedule_decode_requests(input_lengths, dp_size=self._topology["dp"])
        worker_allocations = [worker.to_dict() for worker in schedule.workers]
        raw = self._batch_runner(
            case_id=case["case_id"],
            input_lengths=input_lengths,
            output_length=output_length,
            global_batch_size=int(case["global_batch_size"]),
            worker_allocations=copy.deepcopy(worker_allocations),
            warmup_runs=int(case["warmup_runs"]),
            measure_runs=int(case["measure_runs"]),
            generate_config=copy.deepcopy(case["generate_config"]),
        )
        if not isinstance(raw, Mapping):
            raise ValueError("distribution batch runner must return an object")
        distribution = copy.deepcopy(dict(case.get("distribution", {})))
        distribution["actual_sequence_lengths"] = list(input_lengths)
        return build_decode_case_result(
            case,
            measurements=raw.get("measurements"),
            topology=self._topology,
            search=raw.get("search"),
            distribution=distribution,
        )
