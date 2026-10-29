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
