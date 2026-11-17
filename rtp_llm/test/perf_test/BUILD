package(default_visibility = ["//visibility:public"])

load(":aone_defs.bzl", "unified_decode_remote_test")

load("@pip_gpu_cuda13_torch//:requirements.bzl", requirement_cuda13 = "requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_cuda12 = "requirement")
load("@pip_gpu_cuda12_9_torch//:requirements.bzl", requirement_cuda12_9 = "requirement")
load("@pip_cuda12_arm_torch//:requirements.bzl", requirement_cuda12_arm = "requirement")
load("@pip_cuda13_arm_torch//:requirements.bzl", requirement_cuda13_arm = "requirement")
load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_rocm = "requirement")
load("@pip_cpu_torch//:requirements.bzl", requirement_cpu = "requirement")
load("@pip_arm_torch//:requirements.bzl", requirement_arm = "requirement")
load("@pip_ppu_torch//:requirements.bzl", requirement_ppu = "requirement")

config_setting(
    name = "using_cuda12_x86_pre_12_9",
    define_values = {
        "using_cuda12_x86": "true",
        "using_cuda12_9_x86": "false",
    },
)

py_library(
    name = "perf_test_entry_lib",
    srcs = [
        "aone/__init__.py",
        "aone/config_schema.py",
        "aone/plan.py",
        "aone/sharding.py",
        "capabilities.py",
        "cli.py",
        "config_resolver.py",
        "config_schema.py",
        "decode/__init__.py",
        "decode/distribution.py",
        "decode/metrics.py",
        "decode/model.py",
        "decode/planner.py",
        "decode/scheduler.py",
        "entry.py",
        "case_planning.py",
        "errors.py",
        "plan.py",
        "prefill/__init__.py",
        "prefill/block_size.py",
        "prefill/case_store.py",
        "prefill/fingerprints.py",
        "prefill/measurement.py",
        "prefill/planner.py",
        "prefill/prompts.py",
        "prefill/random_batch.py",
        "prefill/traffic_batch.py",
        "prefill/traffic_distribution.py",
        "prefill/generate_traffic_batch_grid.py",
        "prefill/generate_random_batch_grid.py",
        "run_context.py",
        "run_directory.py",
        "state.py",
        "storage.py",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "perf_test_results_lib",
    srcs = ["results.py"],
    deps = [":perf_test_entry_lib"],
)

py_library(
    name = "perf_test_runtime_lib",
    srcs = [
        "decode/executor.py",
        "decode/finalize.py",
        "decode/profiling.py",
        "decode/report.py",
        "decode/strategy.py",
        "decode/transport.py",
        "execution/devices.py",
        "execution/engine_session.py",
        "execution/launch_boundary.py",
        "execution/local.py",
        "execution/runtime.py",
        "execution/scheduler.py",
        "modes.py",
        "multi_node/decode_runtime.py",
        "multi_node/decode_sync.py",
        "multi_node/multi_benchmark.py",
        "multi_node/gang_config.py",
        "multi_node/local_server_runner.py",
        "multi_node/prefill_runtime.py",
        "prefill/adapter.py",
        "prefill/checkpoint.py",
        "prefill/chart.py",
        "prefill/finalize.py",
        "prefill/dashsc_transport.py",
        "prefill/formula.py",
        "prefill/profile.py",
        "prefill/probe.py",
        "prefill/report.py",
        "prefill/restricted_symbolic.py",
        "prefill/runner.py",
        "prefill/trace.py",
        "test_util.py",
    ],
    deps = [
        ":perf_test_entry_lib",
        ":perf_test_results_lib",
        "//rtp_llm:testlib",
        "//rtp_llm/test/utils:maga_server_manager",
    ] + select({
        "@rtp_llm//:using_cuda13_x86": [requirement_cuda13("plotly")],
        "@rtp_llm//:using_cuda13_arm": [requirement_cuda13_arm("plotly")],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

exports_files([
    "BUILD",
    "aone/gang_worker.py",
    "aone/worker.py",
    "aone_defs.bzl",
    "decode_remote_test.py",
    "examples/decode_aone_gang.json",
    "examples/decode_distribution_local.json",
    "examples/decode_distribution_requests.json",
    "examples/decode_grid_local.json",
    "examples/decode_tpot_local.json",
    "examples/prefill_aone.json",
    "examples/prefill_aone_ci.json",
    "examples/prefill_aone_gang.json",
    "examples/prefill_deepseek_v4_flash_daily.json",
    "examples/prefill_deepseek_v4_pro_daily.json",
    "examples/prefill_grouped_local.json",
    "examples/prefill_local.json",
    "examples/prefill_multi_grid_local.json",
    "examples/prefill_random_batch_local.json",
    "examples/prefill_traffic_distribution_local.json",
    "examples/qwen35_prefill_test.json",
    "examples/req_distribution.json",
    "main.py",
    "multi_node/analyzer/configs/module_base.py",
    "multi_node/analyzer/configs/module_conf.py",
    "multi_node/analyzer/configs/qwen_480b_decode-fp8.py",
    "multi_node/analyzer/configs/qwen_480b_prefill_fp8.py",
    "multi_node/gang_config.py",
    "multi_node/local_server_runner.py",
    "multi_node/multi_benchmark_config.yaml",
    "multi_node/multi_local_executor.sh",
    "multi_node/multi_runner.sh",
    "testdata/historical_aone_plan.json",
])

py_library(
    name = "aone_lib",
    srcs = [
        "aone/broker.py",
        "aone/gang_worker.py",
        "aone/collector.py",
        "aone/package.py",
        "aone/worker.py",
        "execution/aone.py",
        "execution/aone_bazel.py",
        "execution/aone_gang.py",
        "hub_download.py",
    ],
    deps = [
        ":perf_test_entry_lib",
        ":perf_test_runtime_lib",
    ],
)

# Optional, portable result policies. Normal perf_test execution does not
# import these libraries or run baseline comparisons and publishers implicitly.
py_library(
    name = "perf_test_regression_lib",
    srcs = [
        "regression/__init__.py",
        "regression/baseline.py",
        "regression/golden.py",
    ],
    deps = [":perf_test_results_lib"],
)

py_library(
    name = "perf_test_publish_policy_lib",
    srcs = [
        "publish/__init__.py",
        "publish/policy.py",
    ],
    deps = [":perf_test_entry_lib"],
)

py_binary(
    name = "generate_perf_test_golden",
    srcs = ["generate_perf_test_golden.py"],
    deps = [":perf_test_regression_lib"],
)

py_binary(
    name = "validate_perf_test_golden",
    srcs = ["validate_perf_test_golden.py"],
    deps = [":perf_test_regression_lib"],
)

# Compatibility aggregate for offline callers and CI helpers.
py_library(
    name = "perf_test_lib",
    srcs = [
        "dataset.py",
        "engine_paths.py",
        "offline_bench_test.py",
        "offline_runner.py",
        "server.py",
    ],
    data = [":perf_test_examples"],
    deps = [
        ":aone_lib",
        ":perf_test_entry_lib",
        ":perf_test_runtime_lib",
        ":perf_test_results_lib",
        "//rtp_llm:testlib",
        "//rtp_llm/test/utils:maga_server_manager",
        "//rtp_llm/test/utils:device_resource",
    ],
)

# Site-CI adapters: deliver pipeline parameters onto a strict config and
# gate collected prefill artifacts. Portable helpers; CI invokes them
# explicitly, they never run as part of a normal perf test.
py_library(
    name = "ci_helpers",
    srcs = [
        "ci_delivery.py",
        "ci_prefill_verify.py",
    ],
    data = [
        "examples/decode_aone_gang.json",
        "examples/prefill_aone_ci.json",
    ],
    deps = [
        ":perf_test_lib",
    ],
)

py_binary(
    name = "ci_delivery",
    srcs = ["ci_delivery.py"],
    deps = [":ci_helpers"],
)

py_binary(
    name = "ci_prefill_verify",
    srcs = ["ci_prefill_verify.py"],
    deps = [":ci_helpers"],
)

py_binary(
    name = "perf_test",
    main = "main.py",
    srcs = ["main.py"],
    deps = [
        ":aone_lib",
        ":perf_test_entry_lib",
        ":perf_test_runtime_lib",
    ],
)

# Offline throughput benchmark (new entry point).
# Manual target: Qwen3.5 weights must be available locally or resolvable by perf path resolution.
py_test(
    name = "offline_bench_test",
    main = "offline_bench_test.py",
    srcs = ["offline_bench_test.py"],
    timeout = "eternal",
    deps = [
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

py_binary(
    name = "generate_random_batch_grid",
    main = "prefill/generate_random_batch_grid.py",
    srcs = ["prefill/generate_random_batch_grid.py"],
    deps = [":perf_test_entry_lib"],
)

py_binary(
    name = "generate_traffic_batch_grid",
    main = "prefill/generate_traffic_batch_grid.py",
    srcs = ["prefill/generate_traffic_batch_grid.py"],
    deps = [":perf_test_entry_lib"],
)

# One test action runs the complete local-backend grid on a worker with two GPUs.
# Adapt the example model/tokenizer paths to files readable on that worker.
unified_decode_remote_test(
    name = "decode_remote_example",
    config = "examples/decode_grid_local.json",
    gpu_type = "H20",
    gpu_count = 2,
)

filegroup(
    name = "perf_test_python_sources",
    srcs = glob(["**/*.py"], exclude = ["tests/**"]),
)

py_library(
    name = "perf_test_analyzer_lib",
    srcs = [
        "multi_node/analyzer/batch_analyze.py",
        "multi_node/analyzer/merge_decode_results.py",
        "multi_node/analyzer/trace_analyze.py",
        "multi_node/analyzer/configs/module_base.py",
        "multi_node/analyzer/configs/module_conf.py",
        "multi_node/analyzer/configs/qwen_480b_decode-fp8.py",
        "multi_node/analyzer/configs/qwen_480b_prefill_fp8.py",
    ],
    deps = [
        ":perf_test_results_lib",
    ],
)

# Preserve existing test labels as suites; aliases do not expand in tests().

test_suite(
    name = "aone_broker_test",
    tests = ["//rtp_llm/test/perf_test/tests:aone_broker_test"],
)

test_suite(
    name = "aone_entry_test",
    tests = ["//rtp_llm/test/perf_test/tests:aone_entry_test"],
)

test_suite(
    name = "aone_gang_test",
    tests = ["//rtp_llm/test/perf_test/tests:aone_gang_test"],
)

test_suite(
    name = "aone_single_test",
    tests = ["//rtp_llm/test/perf_test/tests:aone_single_test"],
)

test_suite(
    name = "aone_worker_test",
    tests = ["//rtp_llm/test/perf_test/tests:aone_worker_test"],
)

test_suite(
    name = "ci_test",
    tests = ["//rtp_llm/test/perf_test/tests:ci_test"],
)

test_suite(
    name = "decode_executor_test",
    tests = ["//rtp_llm/test/perf_test/tests:decode_executor_test"],
)

test_suite(
    name = "decode_multi_node_test",
    tests = ["//rtp_llm/test/perf_test/tests:decode_multi_node_test"],
)

test_suite(
    name = "decode_remote_test_test",
    tests = ["//rtp_llm/test/perf_test/tests:decode_remote_test_test"],
)

test_suite(
    name = "engine_session_test",
    tests = ["//rtp_llm/test/perf_test/tests:engine_session_test"],
)

test_suite(
    name = "entry_test",
    tests = ["//rtp_llm/test/perf_test/tests:entry_test"],
)

test_suite(
    name = "offline_runner_test",
    tests = ["//rtp_llm/test/perf_test/tests:offline_runner_test"],
)

alias(
    name = "perf_test_regression_test_support",
    testonly = True,
    actual = "//rtp_llm/test/perf_test/tests:perf_test_regression_test_support",
)

alias(
    name = "perf_test_support",
    testonly = True,
    actual = "//rtp_llm/test/perf_test/tests:perf_test_support",
)

test_suite(
    name = "plan_test",
    tests = ["//rtp_llm/test/perf_test/tests:plan_test"],
)

test_suite(
    name = "policy_extraction_test",
    tests = ["//rtp_llm/test/perf_test/tests:policy_extraction_test"],
)

test_suite(
    name = "prefill_profile_test",
    tests = ["//rtp_llm/test/perf_test/tests:prefill_profile_test"],
)

test_suite(
    name = "prefill_report_test",
    tests = ["//rtp_llm/test/perf_test/tests:prefill_report_test"],
)

test_suite(
    name = "prefill_resume_test",
    tests = ["//rtp_llm/test/perf_test/tests:prefill_resume_test"],
)

test_suite(
    name = "prefill_runner_test",
    tests = ["//rtp_llm/test/perf_test/tests:prefill_runner_test"],
)

test_suite(
    name = "publication_policy_test",
    tests = ["//rtp_llm/test/perf_test/tests:publication_policy_test"],
)

test_suite(
    name = "regression_test",
    tests = ["//rtp_llm/test/perf_test/tests:regression_test"],
)

test_suite(
    name = "result_envelope_test",
    tests = ["//rtp_llm/test/perf_test/tests:result_envelope_test"],
)

test_suite(
    name = "runtime_adapter_test",
    tests = ["//rtp_llm/test/perf_test/tests:runtime_adapter_test"],
)

test_suite(
    name = "ssh_prefill_test",
    tests = ["//rtp_llm/test/perf_test/tests:ssh_prefill_test"],
)

test_suite(
    name = "unified_config_test",
    tests = ["//rtp_llm/test/perf_test/tests:unified_config_test"],
)

test_suite(
    name = "unit_tests",
    tests = ["//rtp_llm/test/perf_test/tests:unit_tests"],
)

test_suite(
    name = "workload_test",
    tests = ["//rtp_llm/test/perf_test/tests:workload_test"],
)

filegroup(
    name = "perf_test_examples",
    srcs = [
        "examples/decode_aone_gang.json",
        "examples/decode_distribution_local.json",
        "examples/decode_distribution_requests.json",
        "examples/decode_grid_local.json",
        "examples/decode_tpot_local.json",
        "examples/prefill_aone.json",
        "examples/prefill_aone_ci.json",
        "examples/prefill_aone_gang.json",
        "examples/prefill_deepseek_v4_flash_daily.json",
        "examples/prefill_deepseek_v4_pro_daily.json",
        "examples/prefill_grouped_local.json",
        "examples/prefill_local.json",
        "examples/prefill_multi_grid_local.json",
        "examples/prefill_random_batch_local.json",
        "examples/prefill_traffic_distribution_local.json",
        "examples/qwen35_prefill_test.json",
        "examples/req_distribution.json",
    ],
)
