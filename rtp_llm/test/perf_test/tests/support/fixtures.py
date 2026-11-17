"""Performance framework behavior and failure regressions."""

from pathlib import Path


def fixture_execution(**overrides):
    """Fresh minimal execution fixture with case-specific overrides."""
    return {**{"backend": "local", "devices": [0]}, **overrides}


def fixture_workload(**overrides):
    """Fresh minimal workload fixture with case-specific overrides."""
    return {
        **{
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [128],
            "batch_sizes": [1],
        },
        **overrides,
    }


def fixture_model(**overrides):
    """Fresh minimal model fixture with case-specific overrides."""
    return {**{"label": "test", "type": "registered", "checkpoint": "/m"}, **overrides}


def fixture_engine(**overrides):
    """Fresh minimal engine fixture with case-specific overrides."""
    return {
        **{"max_seq_len": 4096, "max_batch_tokens": 8192, "concurrency_limit": 8},
        **overrides,
    }


PERF_TEST_ROOT = Path(__file__).resolve().parents[2]


def config_resolution_base_config():
    return {
        "schema_version": 1,
        "model": fixture_model(
            label="test-model", type="registered_model", checkpoint="/models/test"
        ),
        "engine": fixture_engine(args={"act_type": "BF16", "reuse_cache": 1}),
        "workload": fixture_workload(),
        "execution": {"backend": "local"},
        "output": {},
        "regression": {},
        "publish": {},
    }
