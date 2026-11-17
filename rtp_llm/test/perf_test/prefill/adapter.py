"""Build the shared prefill measurement and cache-probe adapter."""

from typing import Any, Mapping

from rtp_llm.test.perf_test.config_schema import DEFAULT_PREFILL_TRANSPORT
from rtp_llm.test.perf_test.modes import PrefillAdapter
from rtp_llm.test.perf_test.prefill.probe import (
    probe_cache_block,
    resolve_probe_block_size,
)
from rtp_llm.test.perf_test.prefill.runner import PrefillRunner


def create_prefill_adapter(config: Mapping[str, Any], tokenizer: Any) -> PrefillAdapter:
    workload = config["workload"]
    commit_tail = int(workload["commit_tail_tokens"])
    block_size = resolve_probe_block_size(config)
    return PrefillAdapter(
        lambda context: PrefillRunner(
            context.session.client,
            tokenizer,
            commit_tail_tokens=commit_tail,
            max_batch_tokens=config["engine"]["max_batch_tokens"],
            workspace_tokens=workload.get("workspace_tokens"),
            input_ids_only=workload.get("transport", DEFAULT_PREFILL_TRANSPORT)
            == "dashsc_input_ids",
            cache_block_size=block_size,
        ),
        probe=lambda context: probe_cache_block(
            context.session.client, tokenizer, block_size, commit_tail
        ),
    )
