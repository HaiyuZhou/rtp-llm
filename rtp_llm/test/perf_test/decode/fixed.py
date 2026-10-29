"""Execution of one fixed decode grid case through an injected batch runner."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any, Dict

from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
from rtp_llm.test.perf_test.decode.scheduler import schedule_decode_requests


class FixedDecodeCaseRunner:
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
        global_batch_size = int(case["global_batch_size"])
        output_length = case["output_length"]
        if type(output_length) is not int or output_length <= 0:
            raise ValueError("fixed decode case output_length must be one scalar")
        input_lengths = (int(case["input_length"]),) * global_batch_size
        schedule = schedule_decode_requests(input_lengths, dp_size=self._topology["dp"])
        worker_allocations = [worker.to_dict() for worker in schedule.workers]
        raw = self._batch_runner(
            case_id=case["case_id"],
            input_lengths=input_lengths,
            output_length=output_length,
            global_batch_size=global_batch_size,
            worker_allocations=copy.deepcopy(worker_allocations),
            warmup_runs=int(case["warmup_runs"]),
            measure_runs=int(case["measure_runs"]),
            generate_config=copy.deepcopy(case["generate_config"]),
        )
        if not isinstance(raw, Mapping):
            raise ValueError("fixed decode batch runner must return an object")
        return build_decode_case_result(
            case,
            measurements=raw.get("measurements"),
            topology=self._topology,
            search=raw.get("search"),
        )
