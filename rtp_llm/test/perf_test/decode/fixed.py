"""Execution of one fixed decode grid case through an injected batch runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Dict

from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
from rtp_llm.test.perf_test.decode.tpot import run_batch_with_optional_tpot_search


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
        raw = run_batch_with_optional_tpot_search(
            self._batch_runner,
            case=case,
            topology=self._topology,
            input_lengths=input_lengths,
        )
        result = build_decode_case_result(
            case,
            measurements=raw.get("measurements"),
            topology=self._topology,
            search=raw.get("search"),
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
