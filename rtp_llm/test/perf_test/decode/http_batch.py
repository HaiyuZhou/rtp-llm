"""Concurrent HTTP execution for one planned decode batch case."""

from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Mapping, Sequence

import requests


class DecodeHttpBatchRunner:
    """Execute warmup and formal decode rounds against one model server."""

    def __init__(
        self,
        port: int,
        *,
        prompt_for_length: Callable[[int], str],
        post: Callable[..., Any] = requests.post,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        timeout_seconds: float = 300,
        scheduler_attempts: int = 20,
    ):
        self.port = int(port)
        self._prompt_for_length = prompt_for_length
        self._post = post
        self._monotonic = monotonic
        self._sleep = sleep
        self._timeout_seconds = timeout_seconds
        self._scheduler_attempts = scheduler_attempts

    def _configure_scheduler(self, worker_allocations: Sequence[Mapping[str, Any]]):
        local_batch_size = max(
            int(worker["local_batch_size"]) for worker in worker_allocations
        )
        payload = {"batch_size": local_batch_size, "mode": "decode"}
        last_error = "unknown scheduler error"
        for attempt in range(self._scheduler_attempts):
            try:
                response = self._post(
                    f"http://127.0.0.1:{self.port}/update_scheduler_info",
                    json=payload,
                    timeout=60,
                )
                body = response.json() if response.status_code == 200 else {}
                if response.status_code == 200 and body.get("status", "ok") == "ok":
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            except Exception as error:
                last_error = str(error)
            if attempt + 1 < self._scheduler_attempts:
                self._sleep(3)
        raise RuntimeError(
            f"failed to configure decode scheduler after "
            f"{self._scheduler_attempts} attempts: {last_error}"
        )

    def _request(
        self,
        request_index: int,
        input_length: int,
        output_length: int,
        generate_config: Mapping[str, Any],
    ) -> Dict[str, Any]:
        config = {
            "max_new_tokens": output_length,
            "min_new_tokens": output_length,
            "force_sp_accept": True,
            "top_k": 1,
            **copy.deepcopy(dict(generate_config)),
        }
        try:
            response = self._post(
                f"http://127.0.0.1:{self.port}",
                json={
                    "prompt": self._prompt_for_length(input_length),
                    "generate_config": config,
                },
                headers={"X-DashScope-LoggingConsent": "none"},
                timeout=self._timeout_seconds,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:500]}"
                )
            aux = response.json().get("aux_info") or {}
            actual_input = int(aux.get("input_len", 0))
            actual_output = int(aux.get("output_len", 0))
            wait = float(aux.get("wait_time", 0.0))
            first_token = float(aux.get("first_token_cost_time", 0.0))
            cost = float(aux.get("cost_time", 0.0))
            decode = cost - first_token
            return {
                "request_index": request_index,
                "success": True,
                "input_length": actual_input,
                "output_length": actual_output,
                "latency_ms": {
                    "wait": wait,
                    "prefill": first_token - wait,
                    "decode": decode,
                    "tpot": decode / (actual_output - 1) if actual_output > 1 else 0.0,
                    "total": cost - wait,
                },
            }
        except Exception as error:
            return {
                "request_index": request_index,
                "success": False,
                "input_length": input_length,
                "output_length": output_length,
                "error": str(error),
            }

    def _round(
        self,
        input_lengths: Sequence[int],
        output_length: int,
        generate_config: Mapping[str, Any],
    ) -> list[Dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=len(input_lengths)) as executor:
            futures = [
                executor.submit(
                    self._request,
                    request_index,
                    input_length,
                    output_length,
                    generate_config,
                )
                for request_index, input_length in enumerate(input_lengths)
            ]
            return [future.result() for future in futures]

    def __call__(
        self,
        *,
        case_id: str,
        input_lengths: Sequence[int],
        output_length: int,
        global_batch_size: int,
        worker_allocations: Sequence[Mapping[str, Any]],
        warmup_runs: int,
        measure_runs: int,
        generate_config: Mapping[str, Any],
    ) -> Dict[str, Any]:
        del case_id
        lengths = tuple(int(length) for length in input_lengths)
        if len(lengths) != global_batch_size:
            raise ValueError("decode input lengths do not match global batch size")
        if not lengths:
            raise ValueError("decode batch must contain at least one request")
        self._configure_scheduler(worker_allocations)
        for _ in range(warmup_runs):
            self._round(lengths, output_length, generate_config)

        measurements = []
        for run_index in range(measure_runs):
            started = self._monotonic()
            requests_out = self._round(lengths, output_length, generate_config)
            elapsed = self._monotonic() - started
            successful_tokens = sum(
                request["output_length"]
                for request in requests_out
                if request["success"]
            )
            throughput = successful_tokens / elapsed if elapsed > 0 else 0.0
            measurements.append(
                {
                    "run": run_index,
                    "throughput_tokens_per_second": throughput,
                    "worker_allocations": copy.deepcopy(list(worker_allocations)),
                    "requests": requests_out,
                }
            )
        return {"measurements": measurements}
