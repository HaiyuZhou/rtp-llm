"""Fixtures shared by tests split from plan_test."""

import json
import tempfile
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.tests.support.fixtures import (
    fixture_engine,
    fixture_model,
    fixture_workload,
)


def decode_planner_payload(workload):
    return {
        "schema_version": 1,
        "model": fixture_model(label="test-model"),
        "engine": fixture_engine(),
        "workload": workload,
        "execution": {"backend": "local"},
        "output": {},
        "regression": {},
        "publish": {},
    }


def decode_planner_resolved_grid(*, inputs, outputs, sizes):
    payload = decode_planner_payload(
        {
            "kind": "grid",
            "input_lengths": inputs,
            "output_lengths": outputs,
            "batch": {"strategy": "fixed", "sizes": sizes},
        }
    )
    return (parse_config(payload), resolve_config(payload))


def decode_planner_resolved_distribution(*, distribution, max_size=None, sizes=None):
    batch = (
        {"strategy": "fixed", "sizes": sizes}
        if sizes is not None
        else {
            "strategy": "target_tpot",
            "max_size": max_size,
            "candidate_step": 2,
            "target_tpot_ms": 10.0,
        }
    )
    payload = decode_planner_payload(
        {
            "kind": "distribution",
            "distribution_file": "requests.json",
            "output_length": 16,
            "batch": batch,
        }
    )
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "requests.json").write_text(json.dumps(distribution))
        return (parse_config(payload), resolve_config(payload, source_dir=directory))


def prefill_planner_payload():
    return {
        "schema_version": 1,
        "model": fixture_model(),
        "engine": fixture_engine(max_seq_len=4097, concurrency_limit=2),
        "workload": fixture_workload(
            commit_tail_tokens=128,
            compute_steps=[512, 256],
            min_compute_len=256,
            cache_lengths=[0, 512],
            batch_sizes=[1, 2],
            measure_runs=3,
            seed=17,
            max_cases=100,
        ),
        "execution": {"backend": "local"},
        "output": {},
        "regression": {},
        "publish": {},
    }


def prefill_planner_parsed(payload=None):
    config = parse_config(payload or prefill_planner_payload())
    return (config.workload, config.engine)


def plan_config(backend="local"):
    execution = {"backend": "local", "devices": [2, 3]}
    if backend == "aone":
        execution = {
            "backend": "aone",
            "resource": "daily",
            "gpu_type": "H20",
            "max_parallel": 2,
            "shard_count": 2,
        }
    return {
        "schema_version": 1,
        "model": fixture_model(
            label="test-model",
            type="registered_model",
            checkpoint="/models/test",
            topology={"tp": 2, "dp": 1, "pp": 1, "ep": 2},
        ),
        "engine": fixture_engine(),
        "workload": fixture_workload(),
        "execution": execution,
        "output": {},
        "regression": {},
        "publish": {},
    }


def plan_cases():
    return [
        {"input_len": 1024, "cache_len": 512, "batch_size": 1},
        {"batch_size": 1, "cache_len": 0, "input_len": 512},
    ]
