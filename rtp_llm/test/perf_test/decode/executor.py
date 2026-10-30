"""Transport-independent execution and measurement of a candidate decode batch."""

from __future__ import annotations

import copy
import time
from typing import Callable

from rtp_llm.test.perf_test.config_schema import EngineConfig
from rtp_llm.test.perf_test.decode.model import DecodeBatchMeasurement, DecodeCaseSpec
from rtp_llm.test.perf_test.decode.scheduler import schedule_decode_requests
from rtp_llm.test.perf_test.decode.transport import DecodeHttpTransport
from rtp_llm.test.perf_test.errors import PlanningError


class DecodeCapacityError(PlanningError):
    """A decode candidate exceeds available inputs or per-worker engine bounds."""


class DecodeBatchExecutor:
    def __init__(
        self,
        transport: DecodeHttpTransport,
        *,
        engine: EngineConfig,
        dp_size: int,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self._transport = transport
        self._engine = engine
        self._dp_size = dp_size
        self._monotonic = monotonic

    def measure(self, case: DecodeCaseSpec, batch_size: int) -> DecodeBatchMeasurement:
        inputs = case.input_lengths[:batch_size]
        if len(inputs) != batch_size:
            raise DecodeCapacityError(
                f"decode batch {batch_size} exceeds available inputs "
                f"({len(case.input_lengths)})"
            )
        schedule = schedule_decode_requests(inputs, dp_size=self._dp_size)
        for worker in schedule.workers:
            if worker.local_batch_size > self._engine.concurrency_limit:
                raise DecodeCapacityError(
                    f"decode worker {worker.worker_index} batch "
                    f"{worker.local_batch_size} exceeds concurrency_limit "
                    f"{self._engine.concurrency_limit}"
                )
            if sum(worker.input_lengths) > self._engine.max_batch_tokens:
                raise DecodeCapacityError(
                    f"decode worker {worker.worker_index} input tokens "
                    f"{sum(worker.input_lengths)} exceeds max_batch_tokens "
                    f"{self._engine.max_batch_tokens}"
                )
        self._transport.configure_scheduler(
            tuple(worker.to_dict() for worker in schedule.workers)
        )
        for _ in range(case.warmup_runs):
            self._transport.run_round(
                input_lengths=inputs,
                output_length=case.output_length,
                generate_config=case.generate_config,
            )
        measurements = []
        for run_index in range(case.measure_runs):
            started = self._monotonic()
            requests = self._transport.run_round(
                input_lengths=inputs,
                output_length=case.output_length,
                generate_config=case.generate_config,
            )
            elapsed = self._monotonic() - started
            successful_tokens = sum(
                item["output_length"] for item in requests if item["success"]
            )
            measurements.append(
                {
                    "run": run_index,
                    "throughput_tokens_per_second": (
                        successful_tokens / elapsed if elapsed > 0 else 0.0
                    ),
                    "worker_allocations": [
                        worker.to_dict() for worker in schedule.workers
                    ],
                    "requests": copy.deepcopy(list(requests)),
                }
            )
        return DecodeBatchMeasurement(
            batch_size=batch_size,
            input_lengths=inputs,
            worker_allocations=tuple(worker.to_dict() for worker in schedule.workers),
            measurements=tuple(measurements),
        )
