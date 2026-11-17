"""Unified decode planning and execution for an externally owned service."""

import json
import os
from pathlib import Path

from rtp_llm.test.perf_test.config_schema import EngineConfig
from rtp_llm.test.perf_test.decode.executor import (
    DecodeBatchExecutor,
    DecodeCaseExecutor,
)
from rtp_llm.test.perf_test.decode.finalize import finalize_decode_run
from rtp_llm.test.perf_test.decode.profiling import DecodeProfiler
from rtp_llm.test.perf_test.decode.transport import DecodeHttpTransport
from rtp_llm.test.perf_test.modes import DecodeAdapter, ModeExecutionContext
from rtp_llm.test.perf_test.plan import thaw
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.storage import RunStore


def execute_master_decode(
    plan,
    *,
    source_config,
    endpoint,
    prompt_for_length,
    server_log_path,
    profiler_directory,
    transport=None,
    run_context=None,
):
    """Execute and finalize a run while the caller owns server lifetime."""
    root = Path(plan.result_directory)
    run_context = run_context or {}
    run_id = run_context.get("run_id", root.name)
    initialize_run_directory(
        root,
        source_config=source_config,
        plan=plan,
        run_id=run_id,
        code_version=os.environ.get("BUILD_SCM_REVISION", "unknown"),
        repository_revisions=run_context.get("repository_revisions", {}),
    )
    store = RunStore(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["state"] = "running"
    store.write_manifest(manifest)
    try:
        resolved = thaw(plan.resolved_config)
        topology = resolved["model"]["topology"]
        if transport is None:
            transport = DecodeHttpTransport(
                endpoint=endpoint, prompt_for_length=prompt_for_length
            )
        batch_executor = DecodeBatchExecutor(
            transport,
            engine=EngineConfig(**resolved["engine"]),
            dp_size=int(topology["dp"]),
        )
        profiler = DecodeProfiler(
            batch_executor,
            run_root=root,
            log_path=server_log_path,
            trace_directory=Path(profiler_directory),
        )
        case_executor = DecodeCaseExecutor(
            batch_executor, topology=topology, profiler=profiler
        )
        result = DecodeAdapter(lambda _context: case_executor)(
            ModeExecutionContext(plan=plan, session=None, store=store)
        )
        finalize_decode_run(
            root, plan=plan, run_id=run_id, case_results=result.case_results
        )
    except BaseException:
        manifest["state"] = "failed"
        store.write_manifest(manifest)
        raise
    manifest["state"] = result.state.value
    store.write_manifest(manifest)
    return result


def execute_decode_member(
    plan,
    *,
    source_config,
    endpoint,
    tokenizer_path,
    server_log_path,
    profiler_directory,
    world_rank,
    local_world_size,
    member_hosts,
    control_port,
    run_context=None,
    check_cancelled=lambda: None,
):
    """Run the shared per-member decode load, independent of its scheduler."""
    from rtp_llm.test.perf_test.decode.scheduler import requires_distributed_decode
    from rtp_llm.test.perf_test.execution.distributed.decode_sync import (
        ClusterDecodeTransport,
        serve_decode_worker,
    )
    from rtp_llm.test.perf_test.test_util import create_query

    topology = plan.resolved_config["model"]["topology"]
    distributed = requires_distributed_decode(topology, local_world_size)
    if world_rank and not distributed:
        return None
    lengths = sorted(
        {length for case in plan.cases for length in case.identity["input_lengths"]}
    )
    queries = create_query(tokenizer_path=tokenizer_path, input_len_list=lengths)
    local = DecodeHttpTransport(
        endpoint=endpoint, prompt_for_length=lambda length: queries[length]
    )
    if world_rank:
        serve_decode_worker(
            member_hosts[0],
            control_port,
            world_rank=world_rank,
            local_world_size=local_world_size,
            tp_size=int(topology["tp"]),
            dp_size=int(topology["dp"]),
            local_transport=local,
            check_cancelled=check_cancelled,
        )
        return None
    cluster = None
    try:
        check_cancelled()
        if distributed:
            cluster = ClusterDecodeTransport(
                local,
                bind_host="0.0.0.0",
                control_port=control_port,
                local_world_size=local_world_size,
                tp_size=int(topology["tp"]),
                dp_size=int(topology["dp"]),
                member_world_ranks=tuple(
                    index * local_world_size for index in range(len(member_hosts))
                ),
                check_cancelled=check_cancelled,
            )
            cluster.connect_workers()
        return execute_master_decode(
            plan,
            source_config=source_config,
            endpoint=endpoint,
            prompt_for_length=lambda length: queries[length],
            server_log_path=server_log_path,
            profiler_directory=profiler_directory,
            transport=cluster,
            run_context=run_context,
        )
    finally:
        if cluster is not None:
            cluster.close()
