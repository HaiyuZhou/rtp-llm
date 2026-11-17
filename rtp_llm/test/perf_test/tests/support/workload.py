"""Fixtures shared by tests split from workload_test."""

from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases
from rtp_llm.test.perf_test.tests.support.fixtures import (
    config_resolution_base_config,
    fixture_engine,
    fixture_execution,
    fixture_model,
)


class prefill_prompt_WhitespaceTokenizer:

    def encode(self, text):
        return text.split()


def decode_distribution_source(distribution_file="requests.json"):
    return {
        "schema_version": 1,
        "model": fixture_model(),
        "engine": fixture_engine(),
        "workload": {
            "kind": "distribution",
            "distribution_file": distribution_file,
            "output_length": 32,
            "batch": {"strategy": "fixed", "sizes": [2, 3]},
            "measure_runs": 2,
            "warmup_runs": 1,
            "generate_config": {"top_k": 1},
        },
        "execution": fixture_execution(),
        "output": {},
        "regression": {},
        "publish": {},
    }


random_batch_export_MODULE = "rtp_llm.test.perf_test.prefill.generate_random_batch_grid"


def random_batch_export_random_config():
    return {
        "schema_version": 1,
        "model": fixture_model(
            label="random-test",
            type="qwen_2",
            checkpoint="/models/test",
            tokenizer="/models/test",
        ),
        "engine": fixture_engine(
            max_seq_len=4097, max_batch_tokens=16384, concurrency_limit=4
        ),
        "workload": {
            "kind": "cache_grid",
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "measure_runs": 2,
            "workspace_tokens": 16384,
            "random_request_batches": {
                "num_cases": 6,
                "batch_sizes": [2, 4],
                "min_input_tokens": 256,
                "max_input_tokens": 2048,
                "input_alignment": 256,
                "batch_input_limits": {"4": 1024},
            },
        },
        "execution": fixture_execution(),
        "output": {"result_root": "perf_test_results"},
        "regression": {},
        "publish": {},
    }


def random_batch_export_planned(source):
    config = parse_config(source)
    return plan_prefill_cases(
        config.workload, config.engine, dp_size=config.model.topology.dp
    )


traffic_batch_export_MODULE = (
    "rtp_llm.test.perf_test.prefill.generate_traffic_batch_grid"
)


def traffic_batch_config():
    raw = config_resolution_base_config()
    raw["workload"] = {
        "kind": "cache_grid",
        "cache_alignment": 8,
        "commit_tail_tokens": 8,
        "workspace_tokens": 4096,
        "traffic_request_batches": {
            "distribution_file": "distribution.json",
            "num_cases": 3,
            "batch_sizes": [1, 2],
            "min_input_tokens": 100,
        },
    }
    return raw


def traffic_distribution_payload():
    return dict(
        schema_version=1,
        kind="conditional_request_lengths",
        input_bucket_width=100,
        reuse_ratio_bins=2,
        source_rows=30,
        incomplete_rows=0,
        cells=[
            dict(input_bucket=1, reuse_bin=b, count=10, input_min=100, input_max=199)
            for b in (-1, 2, 1)
        ],
    )
