"""Stable callable API for externally managed decode grid servers."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from rtp_llm.test.perf_test.batch_perf_impl import BatchPerfImpl
from rtp_llm.test.perf_test.dataclass import (
    MetricState,
    TableType,
    create_metrics_table,
)


def run_cartesian_grid(
    port: int,
    dp_size: int,
    batch_size_list: List[int],
    input_len_list: List[int],
    input_query_dict: Dict[int, str],
    *,
    is_decode: bool = True,
    dump_json_path: str = ".",
    decode_test_length: int = 10,
    tp_size: int = 1,
    generate_config: Optional[Dict[str, Any]] = None,
    num_measures: int = 3,
    log_path: str = "",
) -> List[MetricState]:
    """Run the stable externally-managed batch-size by input-length grid."""
    generate_config = generate_config or {}
    title = "Decode Result" if is_decode else "Prefill Result"
    warmup_runs = int(os.environ.get("PERF_GRID_WARMUP_RUNS", "1"))
    BatchPerfImpl(
        port,
        dp_size,
        dp_size,
        input_query_dict[input_len_list[0]],
        is_decode,
        1000,
        decode_test_length,
        False,
        generate_config,
        warmup_runs=0,
        measure_runs=warmup_runs,
        profile_runs=0,
        log_path=log_path,
    ).run()

    metrics_list: List[MetricState] = []
    log_flush_offset = 0
    phase = "decode" if is_decode else "prefill"
    total_tests = len(batch_size_list) * len(input_len_list)
    with tqdm(total=total_tests, desc=f"Running {title}", unit="test") as progress:
        for batch_size in batch_size_list:
            for input_len in input_len_list:
                impl = BatchPerfImpl(
                    port,
                    dp_size,
                    batch_size * dp_size,
                    input_query_dict[input_len],
                    is_decode,
                    500,
                    decode_test_length,
                    True,
                    generate_config,
                    f"bs{batch_size}_seq{input_len}_{phase}",
                    log_path=log_path,
                    log_flush_offset=log_flush_offset,
                )
                metric = impl.run(num_measures=num_measures)
                log_flush_offset = impl.log_flush_offset
                metrics_list.append(MetricState(input_len, batch_size, metric))
                progress.update(1)
    table = create_metrics_table(
        TableType.Decode if is_decode else TableType.Prefill,
        metrics_list,
        dump_json_path,
        {"dp_size": dp_size, "tp_size": tp_size},
        title,
        generate_config,
    )
    logging.info("metrics_table: \n%s", table)
    return metrics_list


def run_single(
    port: int,
    dp_size: int,
    batch_size_list: List[int],
    input_len_list: List[int],
    input_query_dict: Dict[int, str],
    is_decode: bool = True,
    dump_json_path: str = ".",
    decode_test_length: int = 20,
    tp_size: int = 1,
    generate_config: Optional[Dict[str, Any]] = None,
):
    """Run a grid against an already managed server.

    Multi-node launchers own server lifecycle and call this stable boundary.
    """
    return run_cartesian_grid(
        port,
        dp_size,
        batch_size_list,
        input_len_list,
        input_query_dict,
        is_decode=is_decode,
        dump_json_path=dump_json_path,
        decode_test_length=decode_test_length,
        tp_size=tp_size,
        generate_config=generate_config,
    )
