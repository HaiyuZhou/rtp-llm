"""Cache-aware prefill request execution and correctness classification."""

from __future__ import annotations

import copy
import math
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Tuple

from rtp_llm.test.perf_test.prefill.measurement import PREFILL_MEASUREMENT_CONTRACT
from rtp_llm.test.perf_test.prefill.prompts import PrefixPromptFactory


class PrefillRunner:
    def __init__(
        self,
        client: Any,
        tokenizer: Any,
        *,
        commit_tail_tokens: int,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.monotonic = monotonic
        self.client = client
        self.factory = PrefixPromptFactory(tokenizer)
        self.commit_tail_tokens = commit_tail_tokens

    @staticmethod
    def _identity(case: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: copy.deepcopy(value) for key, value in case.items() if key != "case_id"
        }

    @staticmethod
    def _failure(case: Dict[str, Any], status: str, message: str) -> Dict[str, Any]:
        return {
            "case_id": case["case_id"],
            "identity": PrefillRunner._identity(case),
            "status": status,
            "runs": case["measure_runs"],
            "errors": [{"code": status, "message": message}],
            "metrics": None,
        }

    @staticmethod
    def _validate(
        response: Dict[str, Any], *, input_len: int, expected_reuse: int
    ) -> Tuple[str, str] | None:
        if not isinstance(response, dict) or not response.get("success"):
            return "request_failed", "request did not complete successfully"
        if response.get("input_len") != input_len or response.get("output_len") != 1:
            return "invalid_shape", "response input/output shape does not match plan"
        if response.get("reuse_len") != expected_reuse:
            return (
                "invalid_reuse",
                f"expected reuse {expected_reuse}, got {response.get('reuse_len')}",
            )
        latency = response.get("latency_ms")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
        ):
            return "invalid_timing", "latency_ms must be a finite positive number"
        return None

    def run_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        batch_size = int(case["batch_size"])
        measure_runs = int(case["measure_runs"])
        input_len = int(case["input_len"])
        cache_len = int(case["cache_len"])
        members = [
            self.factory.build_case_prompts(
                f"{case['case_id']}_r{member}", input_len, cache_len, measure_runs
            )
            for member in range(batch_size)
        ]

        if cache_len:
            try:
                self.client.configure_scheduler(1, mode="prefill")
            except Exception as error:
                return self._failure(case, "request_failed", str(error))
            seed_len = cache_len + self.commit_tail_tokens
            for member, prompts in enumerate(members):
                seed_text, seed_ids = self.factory.make_seed_and_ids(
                    f"{case['case_id']}_r{member}",
                    prompts.seed_text,
                    cache_len,
                    self.commit_tail_tokens,
                )
                try:
                    response = self.client.request(
                        phase="seed",
                        case_id=case["case_id"],
                        request_index=member,
                        run_index=None,
                        prompt=seed_text,
                        input_ids=seed_ids,
                        output_length=1,
                    )
                except Exception as error:
                    return self._failure(case, "request_failed", str(error))
                invalid = self._validate(response, input_len=seed_len, expected_reuse=0)
                if invalid is not None:
                    return self._failure(case, *invalid)

        try:
            self.client.configure_scheduler(batch_size, mode="prefill")
        except Exception as error:
            return self._failure(case, "request_failed", str(error))

        measurements: List[Dict[str, Any]] = []
        round_rates: List[float] = []
        compute_length = int(case["compute_len"])
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            for run_index in range(measure_runs):
                release = threading.Event()

                def request_member(member: int) -> Dict[str, Any]:
                    release.wait()
                    prompts = members[member]
                    return self.client.request(
                        phase="measure",
                        case_id=case["case_id"],
                        request_index=member,
                        run_index=run_index,
                        prompt=prompts.run_texts[run_index],
                        input_ids=prompts.run_ids[run_index],
                        output_length=1,
                    )

                futures = [
                    executor.submit(request_member, member)
                    for member in range(batch_size)
                ]
                started = self.monotonic()
                release.set()
                responses = []
                failures = []
                # Drain every submitted member before classifying the round.
                for future in futures:
                    try:
                        responses.append(future.result())
                    except Exception as error:
                        failures.append(str(error))
                elapsed_ms = (self.monotonic() - started) * 1000.0
                if failures:
                    return self._failure(case, "request_failed", "; ".join(failures))
                if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                    return self._failure(
                        case,
                        "invalid_timing",
                        "round elapsed time must be finite and positive",
                    )
                for member, response in enumerate(responses):
                    invalid = self._validate(
                        response, input_len=input_len, expected_reuse=cache_len
                    )
                    if invalid is not None:
                        return self._failure(case, *invalid)
                    measurements.append(
                        {
                            "run": run_index,
                            "request_index": member,
                            "latency_ms": response["latency_ms"],
                            "observed_reuse_length": response["reuse_len"],
                            "round_elapsed_ms": elapsed_ms,
                            "measurement_contract": PREFILL_MEASUREMENT_CONTRACT,
                        }
                    )
                round_rates.append(compute_length * batch_size * 1000.0 / elapsed_ms)

        latencies = [float(item["latency_ms"]) for item in measurements]
        median_latency = statistics.median(latencies)
        throughput = statistics.mean(round_rates)
        return {
            "case_id": case["case_id"],
            "identity": self._identity(case),
            "status": "ok",
            "runs": measure_runs,
            "errors": [],
            "metrics": {
                "input_length": input_len,
                "expected_reuse_length": cache_len,
                "observed_reuse_length": cache_len,
                "compute_length": compute_length,
                "latency_ms": median_latency,
                "throughput_tokens_per_second": throughput,
                "validation": {"shape": True, "reuse": True, "timing": True},
                "seed": int(case["seed"]),
                "prefix_policy": {"kind": case["prefix_policy"]},
                "measurements": measurements,
            },
        }
