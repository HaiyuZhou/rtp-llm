package(default_visibility = ["//visibility:public"])

py_library(
    name = "perf_test_entry_lib",
    srcs = [
        "aone/__init__.py",
        "aone/sharding.py",
        "capabilities.py",
        "cli.py",
        "config_resolver.py",
        "config_schema.py",
        "decode/__init__.py",
        "decode/distribution.py",
        "decode/metrics.py",
        "decode/planner.py",
        "decode/scheduler.py",
        "dry_run.py",
        "entry.py",
        "errors.py",
        "orchestrator.py",
        "plan.py",
        "prefill/__init__.py",
        "prefill/planner.py",
        "run_directory.py",
        "state.py",
        "storage.py",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "perf_test_runtime_lib",
    srcs = [
        "decode/finalize.py",
        "decode/fixed.py",
        "decode/http_batch.py",
        "execution/defaults.py",
        "execution/engine_session.py",
        "execution/local.py",
        "execution/runtime.py",
        "execution/scheduler.py",
        "modes/base.py",
        "modes/decode.py",
        "modes/prefill.py",
        "prefill/checkpoint.py",
        "prefill/finalize.py",
        "prefill/profile.py",
        "prefill/prompts.py",
        "prefill/report.py",
        "prefill/runner.py",
        "results.py",
        "test_util.py",
    ],
    deps = [
        ":perf_test_entry_lib",
        "//rtp_llm:testlib",
        "//rtp_llm/test/utils:maga_server_manager",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "perf_test_lib",
    srcs = [
        "batch_perf_impl.py",
        "aone/package.py",
        "aone/collector.py",
        "aone/executor.py",
        "aone/sharding.py",
        "aone/worker.py",
        "aone_worker.py",
        "capabilities.py",
        "cli.py",
        "config_resolver.py",
        "config_schema.py",
        "dataclass.py",
        "dataset.py",
        "decode/fixed.py",
        "decode/grid_runner.py",
        "decode/http_batch.py",
        "decode/distribution.py",
        "decode/finalize.py",
        "decode/planner.py",
        "decode/metrics.py",
        "decode/scheduler.py",
        "decode/tpot.py",
        "dry_run.py",
        "entry.py",
        "errors.py",
        "execution/engine_session.py",
        "execution/aone.py",
        "execution/defaults.py",
        "execution/local.py",
        "execution/runtime.py",
        "execution/scheduler.py",
        "engine_paths.py",
        "hub_download.py",
        "offline_bench_test.py",
        "offline_runner.py",
        "orchestrator.py",
        "plan.py",
        "prefill/planner.py",
        "prefill/checkpoint.py",
        "prefill/finalize.py",
        "prefill/prompts.py",
        "prefill/profile.py",
        "prefill/report.py",
        "prefill/runner.py",
        "publish/__init__.py",
        "publish/archive.py",
        "publish/odps.py",
        "publish/policy.py",
        "regression/__init__.py",
        "regression/baseline.py",
        "regression/candidate.py",
        "regression/golden.py",
        "server.py",
        "run_directory.py",
        "storage.py",
        "state.py",
        "results.py",
        "modes/base.py",
        "modes/decode.py",
        "modes/prefill.py",
        "test_util.py",
    ],
    data = glob(["examples/*.json"]),
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/test/utils:maga_server_manager",
        "//rtp_llm/test/utils:device_resource",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "entry_test",
    srcs = ["entry_test.py"],
    data = [
        "//tools:perf_test_script",
        "examples/prefill_aone.json",
        "examples/prefill_local.json",
    ],
    deps = [
        ":perf_test_entry_lib",
    ],
)

py_binary(
    name = "perf_test",
    main = "main.py",
    srcs = ["main.py"],
    deps = [
        ":perf_test_entry_lib",
        ":perf_test_runtime_lib",
    ],
)

py_test(
    name = "regression_publish_docs_test",
    srcs = ["regression_publish_docs_test.py"],
    data = [
        "BUILD",
        "docs/regression_publish.md",
        "examples/prefill_regression_publish.json",
    ],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "publisher_policy_test",
    srcs = ["publisher_policy_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "odps_publisher_test",
    srcs = ["odps_publisher_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "archive_publisher_test",
    srcs = ["archive_publisher_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "baseline_regression_test",
    srcs = ["baseline_regression_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "golden_test",
    srcs = ["golden_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_binary(
    name = "generate_perf_test_golden",
    srcs = ["generate_perf_test_golden.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_binary(
    name = "validate_perf_test_golden",
    srcs = ["validate_perf_test_golden.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "unified_config_test",
    srcs = ["unified_config_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "config_resolution_test",
    srcs = ["config_resolution_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "capabilities_test",
    srcs = ["capabilities_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "plan_test",
    srcs = ["plan_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "cli_test",
    srcs = ["cli_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "dry_run_test",
    srcs = ["dry_run_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "run_directory_test",
    srcs = ["run_directory_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "atomic_storage_test",
    srcs = ["atomic_storage_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "state_test",
    srcs = ["state_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "result_envelope_test",
    srcs = ["result_envelope_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "runtime_adapter_test",
    srcs = ["runtime_adapter_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "decode_http_batch_test",
    srcs = ["decode_http_batch_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "prefill_finalize_test",
    srcs = ["prefill_finalize_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "entry_execution_test",
    srcs = ["entry_execution_test.py"],
    deps = [
        ":perf_test_entry_lib",
    ],
)

py_test(
    name = "default_executor_test",
    srcs = ["default_executor_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "unified_docs_test",
    srcs = ["unified_docs_test.py"],
    data = glob(["docs/*.md"]),
    deps = [
        ":perf_test_entry_lib",
    ],
)

py_test(
    name = "engine_session_test",
    srcs = ["engine_session_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "local_executor_test",
    srcs = ["local_executor_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "prefill_planner_test",
    srcs = ["prefill_planner_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "prefill_prompt_test",
    srcs = ["prefill_prompt_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "prefill_runner_test",
    srcs = ["prefill_runner_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "prefill_adapter_test",
    srcs = ["prefill_adapter_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "prefill_resume_test",
    srcs = ["prefill_resume_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "prefill_profile_test",
    srcs = ["prefill_profile_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "prefill_report_test",
    srcs = ["prefill_report_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "prefill_examples_test",
    srcs = ["prefill_examples_test.py"],
    data = [
        "docs/prefill.md",
        "examples/prefill_local.json",
    ],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "aone_package_test",
    srcs = ["aone_package_test.py"],
    data = ["defs.bzl"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "aone_sharding_test",
    srcs = ["aone_sharding_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "aone_worker_test",
    srcs = ["aone_worker_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "aone_collector_test",
    srcs = ["aone_collector_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "aone_orchestrator_test",
    srcs = ["aone_orchestrator_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "aone_examples_test",
    srcs = ["aone_examples_test.py"],
    data = [
        "docs/aone_prefill.md",
        "examples/prefill_aone.json",
    ],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "decode_grid_api_test",
    srcs = ["decode_grid_api_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "decode_fixed_test",
    srcs = ["decode_fixed_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "decode_distribution_test",
    srcs = ["decode_distribution_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "decode_scheduler_test",
    srcs = ["decode_scheduler_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "decode_tpot_test",
    srcs = ["decode_tpot_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "decode_metrics_test",
    srcs = ["decode_metrics_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "decode_examples_test",
    srcs = ["decode_examples_test.py"],
    data = [
        "docs/decode.md",
        "examples/decode_distribution_local.json",
        "examples/decode_distribution_requests.json",
        "examples/decode_grid_local.json",
        "examples/decode_tpot_local.json",
    ],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "engine_paths_test",
    srcs = ["engine_paths_test.py"],
    deps = [":perf_test_lib"],
)

py_test(
    name = "execution_scheduler_test",
    srcs = ["execution_scheduler_test.py"],
    deps = [":perf_test_lib"],
)

# Offline throughput benchmark (new entry point).
# Manual target: Qwen3.5 weights must be available locally or resolvable by perf path resolution.
py_test(
    name = "offline_bench_test",
    main = "offline_bench_test.py",
    srcs = ["offline_bench_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = [
        "//rtp_llm:sdk",
    ],
    args = [
        "--checkpoint_path", "Qwen/Qwen3.5-35B-A3B",
        "--model_type", "qwen35_moe",
        "--input_len_min", "128",
        "--input_len_max", "512",
        "--output_len_min", "64",
        "--output_len_max", "256",
        "--concurrency_limit", "256",
        "--seq_size_per_block", "1024",
    ],
    tags = ["manual"],
)
