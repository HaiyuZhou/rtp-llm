def unified_prefill_ci_test(name, config, gpu_type, gpu_count, data = [], deps = []):
    """Run one strict unified prefill config in an AOne Bazel test action."""
    native.py_test(
        name = name,
        main = "//rtp_llm/test/perf_test:main.py",
        srcs = ["//rtp_llm/test/perf_test:main.py"],
        timeout = "eternal",
        deps = [
            "//rtp_llm/test/perf_test:perf_test_entry_lib",
            "//rtp_llm/test/perf_test:perf_test_runtime_lib",
        ] + deps,
        data = [
            "//rtp_llm:sdk",
            config,
        ] + data,
        args = [
            "--mode=prefill",
            "--action=run",
            "--config=$(location " + config + ")",
        ],
        tags = ["manual", gpu_type],
        exec_properties = {
            "gpu": gpu_type,
            "gpu_count": str(gpu_count),
        },
    )
    return name

def perf_test_aone_shard(name, resolved_config, resolved_workload, shard_plan, shard_id, gpu_type, gpu_count):
    """Define one unified prefill shard using only declared runfiles."""
    native.py_test(
        name = name,
        main = "//rtp_llm/test/perf_test:aone_worker.py",
        srcs = ["//rtp_llm/test/perf_test:aone_worker.py"],
        timeout = "eternal",
        deps = ["//rtp_llm/test/perf_test:perf_test_lib"],
        data = [
            "//rtp_llm:sdk",
            resolved_config,
            resolved_workload,
            shard_plan,
        ],
        args = [
            "--resolved-config=$(location " + resolved_config + ")",
            "--resolved-workload=$(location " + resolved_workload + ")",
            "--shard-plan=$(location " + shard_plan + ")",
            "--shard-id=" + shard_id,
        ],
        tags = ["manual", gpu_type],
        exec_properties = {
            "gpu": gpu_type,
            "gpu_count": str(gpu_count),
        },
        env = {"WORLD_SIZE": str(gpu_count)},
    )

def perf_test_gang_shard(name, resolved_config, resolved_workload, gang_plan, gpu_type, gpus_per_host, machine_count):
    """Define one multi-machine gang run as sharded remote test actions.

    The single target expands to ``machine_count`` test shards and every
    shard requests the full registered per-host GPU count so each member
    occupies a whole machine. No world-size env is set here: the gang
    worker computes it from the gang plan. The submit side runs these
    targets with ``--test_timeout=7000``.
    """
    native.py_test(
        name = name,
        main = "//rtp_llm/test/perf_test:aone_gang_worker.py",
        srcs = ["//rtp_llm/test/perf_test:aone_gang_worker.py"],
        timeout = "eternal",
        shard_count = machine_count,
        deps = ["//rtp_llm/test/perf_test:perf_test_lib"],
        data = [
            "//rtp_llm:sdk",
            resolved_config,
            resolved_workload,
            gang_plan,
        ],
        args = [
            "--resolved-config=$(location " + resolved_config + ")",
            "--resolved-workload=$(location " + resolved_workload + ")",
            "--gang-plan=$(location " + gang_plan + ")",
        ],
        tags = ["manual", gpu_type],
        exec_properties = {
            "gpu": gpu_type,
            "gpu_count": str(gpus_per_host),
        },
        env_inherit = [
            "PERF_TEST_BROKER_URL",
            "PERF_TEST_BROKER_TOKEN",
            "PERF_TEST_BROKER_RUN_ID",
        ],
    )
