"""Fixtures shared by tests split from prefill_profile_test."""

from dataclasses import replace

from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    DecodeProfilingConfig,
    ExactBatchPolicy,
    TargetTPOTPolicy,
)
from rtp_llm.test.perf_test.plan import case_id

decode_profiling_TOPOLOGY = {"tp": 1, "dp": 1, "pp": 1, "ep": 1}


def decode_profiling_exact_case(*, batch=4, profile=False):
    return DecodeCaseSpec(
        workload_kind="grid",
        input_lengths=(128,) * batch,
        output_length=16,
        batch_policy=ExactBatchPolicy(strategy="fixed", size=batch),
        measure_runs=2,
        warmup_runs=1,
        generate_config={"top_p": 0.8, "nested": {"value": 1}},
        profiling=DecodeProfilingConfig(enabled=profile),
    )


def decode_profiling_tpot_case(*, profile=True):
    return replace(
        decode_profiling_exact_case(batch=16, profile=profile),
        batch_policy=TargetTPOTPolicy("target_tpot", 16, 4, 10.0),
    )


def decode_profiling_case_payload(case):
    identity = case.to_identity()
    return {"case_id": case_id("decode", identity), **identity}


def decode_profiling_measurement(batch, tpot=1.0):
    return DecodeBatchMeasurement(
        batch_size=batch,
        input_lengths=(128,) * batch,
        worker_allocations=(),
        measurements=tuple(
            (
                {
                    "run": run,
                    "worker_allocations": [],
                    "throughput_tokens_per_second": 100.0,
                    "requests": [
                        {
                            "request_index": index,
                            "success": True,
                            "input_length": 128,
                            "output_length": 16,
                            "latency_ms": {
                                "wait": 1.0,
                                "prefill": 2.0,
                                "decode": 15.0,
                                "tpot": tpot,
                                "total": 17.0,
                            },
                        }
                        for index in range(batch)
                    ],
                }
                for run in range(2)
            )
        ),
    )
