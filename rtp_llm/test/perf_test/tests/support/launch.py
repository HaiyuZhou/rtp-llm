"""Fixtures shared by tests split from runtime_adapter_test."""

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.tests.support.fixtures import (
    fixture_engine,
    fixture_model,
    fixture_workload,
)


def runtime_adapter_plan(backend="local", transport="http_text", topology=None):
    if topology is None:
        topology = {"tp": 2, "dp": 1, "pp": 1, "ep": 2}
    world_size = topology.get("tp", 1) * topology.get("dp", 1) * topology.get("pp", 1)
    source = {
        "schema_version": 1,
        "model": fixture_model(
            label="model",
            type="qwen_2",
            checkpoint="/models/qwen",
            tokenizer="/models/qwen-tokenizer",
            topology=topology,
        ),
        "engine": fixture_engine(
            concurrency_limit=4,
            args={"reuse_cache": 1, "enable_cuda_graph": 0},
            env={"TOKENIZERS_PARALLELISM": "false"},
        ),
        "workload": fixture_workload(
            transport=transport,
            max_input_len=1024,
            compute_steps=[256],
            cache_lengths=[0],
        ),
        "execution": (
            {
                "backend": "local",
                "devices": [3, 5] if world_size == 2 else list(range(world_size)),
            }
            if backend == "local"
            else {
                "backend": "aone",
                "resource": "daily",
                "gpu_type": "H20_CU13",
                "max_parallel": 1,
                "shard_count": 1,
            }
        ),
        "output": {"result_root": "/tmp/results"},
        "regression": {},
        "publish": {},
    }
    return build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(source),
        cases=[
            {"input_len": 1024, "cache_len": 0, "compute_len": 1024, "batch_size": 1}
        ],
        result_directory="/tmp/results/run",
    )


class decode_transport_Response:

    def __init__(self, payload=None, *, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload
