"""Fixtures shared by tests split from entry_test."""

from rtp_llm.test.perf_test.tests.support.fixtures import (
    fixture_engine,
    fixture_model,
    fixture_workload,
)
from rtp_llm.test.perf_test.tests.support.paths import PERF_TEST_ROOT

entry_ROOT = PERF_TEST_ROOT.parents[2]

entry_SCRIPT = entry_ROOT / "tools" / "perf_test"

entry_LOCAL = (
    entry_ROOT / "rtp_llm" / "test" / "perf_test" / "examples" / "prefill_local.json"
)


def dry_run_config(backend):
    execution = {"backend": "local", "devices": [0]}
    return {
        "schema_version": 1,
        "model": fixture_model(
            label="test-model", type="registered_model", checkpoint="/models/test"
        ),
        "engine": fixture_engine(),
        "workload": fixture_workload(),
        "execution": execution,
        "output": {},
        "regression": {},
        "publish": {},
    }


def dry_run_cases():
    return [
        {"input_len": 512, "cache_len": 0, "batch_size": 1},
        {"input_len": 1024, "cache_len": 512, "batch_size": 1},
    ]


public_boundary_ROOT = PERF_TEST_ROOT
