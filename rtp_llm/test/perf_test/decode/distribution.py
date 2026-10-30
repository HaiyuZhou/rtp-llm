"""Materialized decode distribution inputs and the legacy case runner."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any, Dict, Tuple

from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchConfig,
    FixedBatchConfig,
    TargetTPOTConfig,
)
from rtp_llm.test.perf_test.decode.tpot import run_batch_with_optional_tpot_search
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


def parse_distribution_inputs(
    distribution: Any, batch: DecodeBatchConfig
) -> Tuple[Tuple[Tuple[int, ...], str], ...]:
    """Materialize policy-sized input prefixes or exact fixed-batch maps."""
    if isinstance(distribution, list):
        sequence_lengths = _lengths("distribution", distribution)
    elif isinstance(distribution, Mapping) and set(distribution) == {
        "sequence_lengths"
    }:
        sequence_lengths = _lengths(
            "distribution.sequence_lengths", distribution["sequence_lengths"]
        )
    else:
        sequence_lengths = ()
    if sequence_lengths:
        required = (
            max(batch.sizes) if isinstance(batch, FixedBatchConfig) else batch.max_size
        )
        if len(sequence_lengths) < required:
            raise PlanningError("distribution does not contain enough sequence lengths")
        sizes = (
            batch.sizes if isinstance(batch, FixedBatchConfig) else (batch.max_size,)
        )
        return tuple((sequence_lengths[:size], "sequence_lengths") for size in sizes)
    if not isinstance(distribution, Mapping) or set(distribution) != {
        "batch_seq_len_map"
    }:
        raise PlanningError(
            "distribution must contain sequence_lengths or batch_seq_len_map"
        )
    if isinstance(batch, TargetTPOTConfig):
        raise PlanningError("batch_seq_len_map cannot be used with target_tpot")
    raw_map = distribution["batch_seq_len_map"]
    expected_keys = {str(size) for size in batch.sizes}
    if not isinstance(raw_map, Mapping) or set(raw_map) != expected_keys:
        raise PlanningError(
            "distribution batch map keys must exactly match fixed batch sizes"
        )
    result = []
    for size in batch.sizes:
        lengths = _lengths(f"distribution.batch_seq_len_map.{size}", raw_map[str(size)])
        if len(lengths) != size:
            raise PlanningError(
                f"distribution batch {size} does not match its sequence count"
            )
        result.append((lengths, "batch_seq_len_map"))
    return tuple(result)


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
        raw = run_batch_with_optional_tpot_search(
            self._batch_runner,
            case=case,
            topology=self._topology,
            input_lengths=input_lengths,
        )
        distribution = copy.deepcopy(dict(case.get("distribution", {})))
        distribution["actual_sequence_lengths"] = list(input_lengths)
        result = build_decode_case_result(
            case,
            measurements=raw.get("measurements"),
            topology=self._topology,
            search=raw.get("search"),
            distribution=distribution,
        )
        search = raw.get("search")
        if (
            result["status"] == "ok"
            and isinstance(search, Mapping)
            and search.get("best_batch_size") is None
        ):
            result["status"] = "request_failed"
            result["errors"].append(
                {
                    "code": "target_tpot_not_met",
                    "message": "no measured batch satisfied target_tpot_ms",
                }
            )
        return result
