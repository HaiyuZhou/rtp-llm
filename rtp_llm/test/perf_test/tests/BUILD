package(default_visibility = ["//visibility:public"])

py_library(
    name = "perf_test_regression_test_support",
    testonly = True,
    srcs = ["support/regression.py"],
    deps = ["//rtp_llm/test/perf_test:perf_test_regression_lib"],
)

py_library(
    name = "perf_test_support",
    testonly = True,
    srcs = [
        "support/__init__.py",
        "support/fixtures.py",
        "support/paths.py",
        "support/aone.py",
    ],
    deps = ["//rtp_llm/test/perf_test:perf_test_entry_lib"],
)

py_library(
    name = "perf_test_gang_test_support",
    testonly = True,
    srcs = ["support/gang.py"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_results_lib",
    ],
)

py_library(
    name = "perf_test_execution_test_support",
    testonly = True,
    srcs = ["support/execution.py"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
        ":perf_test_support",
    ],
)

py_library(
    name = "perf_test_entry_test_support",
    testonly = True,
    srcs = ["support/entry.py"],
    deps = [":perf_test_support"],
)

py_library(
    name = "perf_test_plan_test_support",
    testonly = True,
    srcs = ["support/plan.py"],
    deps = ["//rtp_llm/test/perf_test:perf_test_entry_lib", ":perf_test_support"],
)

py_library(
    name = "perf_test_decode_profiling_test_support",
    testonly = True,
    srcs = ["support/decode_profiling.py"],
    deps = ["//rtp_llm/test/perf_test:perf_test_entry_lib"],
)

py_library(
    name = "perf_test_reports_test_support",
    testonly = True,
    srcs = ["support/reports.py"],
    deps = ["//rtp_llm/test/perf_test:perf_test_entry_lib", ":perf_test_support"],
)

py_library(
    name = "perf_test_run_storage_test_support",
    testonly = True,
    srcs = ["support/run_storage.py"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
        ":perf_test_support",
    ],
)

py_library(
    name = "perf_test_prefill_test_support",
    testonly = True,
    srcs = ["support/prefill.py"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
        ":perf_test_support",
    ],
)

py_library(
    name = "perf_test_results_test_support",
    testonly = True,
    srcs = ["support/results.py"],
    deps = ["//rtp_llm/test/perf_test:perf_test_entry_lib", ":perf_test_support"],
)

py_library(
    name = "perf_test_launch_test_support",
    testonly = True,
    srcs = ["support/launch.py"],
    deps = ["//rtp_llm/test/perf_test:perf_test_entry_lib", ":perf_test_support"],
)

py_library(
    name = "perf_test_workload_test_support",
    testonly = True,
    srcs = ["support/workload.py"],
    deps = ["//rtp_llm/test/perf_test:perf_test_entry_lib", ":perf_test_support"],
)

py_test(
    name = "aone_broker_test",
    srcs = ["aone/broker_test.py"],
    main = "aone/broker_test.py",
    data = ["//rtp_llm/test/perf_test:aone_defs.bzl"],
    deps = ["//rtp_llm/test/perf_test:aone_lib"],
)

py_test(
    name = "aone_entry_test",
    srcs = ["aone/entry_test.py"],
    main = "aone/entry_test.py",
    deps = [
        "//rtp_llm/test/perf_test:aone_lib",
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
    ],
)

py_test(
    name = "unit_aone_gang_executor_test",
    srcs = ["aone/gang_executor_test.py"],
    main = "aone/gang_executor_test.py",
    data = [],
    deps = [
        "//rtp_llm/test/perf_test:aone_lib",
        ":perf_test_gang_test_support",
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_results_lib",
    ],
)

py_test(
    name = "unit_aone_gang_worker_test",
    srcs = ["aone/gang_worker_test.py"],
    main = "aone/gang_worker_test.py",
    data = [
        "//rtp_llm/test/perf_test:perf_test_examples",
    ],
    deps = [
        "//rtp_llm/test/perf_test:aone_lib",
        ":perf_test_gang_test_support",
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_results_lib",
        ":perf_test_support",
    ],
)

py_test(
    name = "unit_aone_gang_package_test",
    srcs = ["aone/gang_package_test.py"],
    main = "aone/gang_package_test.py",
    data = ["//rtp_llm/test/perf_test:aone_defs.bzl"],
    deps = [
        "//rtp_llm/test/perf_test:aone_lib",
        ":perf_test_gang_test_support",
        ":perf_test_support",
    ],
)


py_test(
    name = "aone_single_test",
    srcs = ["aone/single_test.py"],
    main = "aone/single_test.py",
    data = ["//rtp_llm/test/perf_test:aone_defs.bzl"],
    deps = [
        "//rtp_llm/test/perf_test:aone_lib",
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_results_lib",
        ":perf_test_support",
    ],
)

py_test(
    name = "aone_worker_test",
    srcs = ["aone/worker_test.py"],
    main = "aone/worker_test.py",
    deps = [
        "//rtp_llm/test/perf_test:aone_lib",
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
        ":perf_test_support",
    ],
)

py_test(
    name = "ci_test",
    srcs = ["core/ci_test.py"],
    main = "core/ci_test.py",
    deps = [
        "//rtp_llm/test/perf_test:ci_helpers",
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
        ":perf_test_support",
    ],
)

py_test(
    name = "decode_executor_test",
    srcs = ["decode/executor_test.py"],
    main = "decode/executor_test.py",
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_results_lib",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
        ":perf_test_support",
    ],
)



py_test(
    name = "unit_execution_engine_session_test",
    srcs = ["execution/engine_session_test.py"],
    main = "execution/engine_session_test.py",
    deps = [
        ":perf_test_execution_test_support",
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
    ],
)

py_test(
    name = "unit_execution_local_test",
    srcs = ["execution/local_test.py"],
    main = "execution/local_test.py",
    deps = [
        ":perf_test_execution_test_support",
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
    ],
)

py_test(
    name = "unit_prefill_backend_integration_test",
    srcs = ["prefill/backend_integration_test.py"],
    main = "prefill/backend_integration_test.py",
    deps = [
        ":perf_test_execution_test_support",
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
    ],
)





py_test(
    name = "offline_runner_test",
    srcs = ["execution/offline_runner_test.py"],
    main = "execution/offline_runner_test.py",
    deps = ["//rtp_llm/test/perf_test:perf_test_lib"],
)

py_test(
    name = "unit_core_plan_test",
    srcs = ["core/plan_test.py"],
    main = "core/plan_test.py",
    data = [
        "//rtp_llm/test/perf_test:examples/prefill_local.json",
        "//rtp_llm/test/perf_test:examples/decode_grid_local.json",
    ],
    deps = ["//rtp_llm/test/perf_test:perf_test_entry_lib", ":perf_test_plan_test_support"],
)

py_test(
    name = "unit_aone_plan_test",
    srcs = ["aone/plan_test.py"],
    main = "aone/plan_test.py",
    data = [
        "//rtp_llm/test/perf_test:examples/prefill_local.json",
        "//rtp_llm/test/perf_test:examples/decode_grid_local.json",
    ],
    deps = ["//rtp_llm/test/perf_test:perf_test_entry_lib", ":perf_test_plan_test_support"],
)


py_test(
    name = "unit_prefill_planner_test",
    srcs = ["prefill/planner_test.py"],
    main = "prefill/planner_test.py",
    data = [
        "//rtp_llm/test/perf_test:examples/prefill_local.json",
        "//rtp_llm/test/perf_test:examples/decode_grid_local.json",
    ],
    deps = ["//rtp_llm/test/perf_test:perf_test_entry_lib", ":perf_test_plan_test_support"],
)

py_test(
    name = "unit_decode_planner_test",
    srcs = ["decode/planner_test.py"],
    main = "decode/planner_test.py",
    data = [
        "//rtp_llm/test/perf_test:examples/prefill_local.json",
        "//rtp_llm/test/perf_test:examples/decode_grid_local.json",
    ],
    deps = ["//rtp_llm/test/perf_test:perf_test_entry_lib", ":perf_test_plan_test_support"],
)


py_test(
    name = "policy_extraction_test",
    srcs = ["policies/extraction_test.py"],
    main = "policies/extraction_test.py",
    deps = [
        "//rtp_llm/test/perf_test:perf_test_publish_policy_lib",
        "//rtp_llm/test/perf_test:perf_test_regression_lib",
    ],
)

py_test(
    name = "unit_prefill_profile_test",
    srcs = ["prefill/profile_test.py"],
    main = "prefill/profile_test.py",
    deps = ["//rtp_llm/test/perf_test:perf_test_runtime_lib", ":perf_test_support"],
)

py_test(
    name = "unit_execution_profiling_launch_test",
    srcs = ["execution/profiling_launch_test.py"],
    main = "execution/profiling_launch_test.py",
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
        ":perf_test_support",
    ],
)

py_test(
    name = "unit_decode_profiling_test",
    srcs = ["decode/profiling_test.py"],
    main = "decode/profiling_test.py",
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        ":perf_test_decode_profiling_test_support",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
    ],
)


py_test(
    name = "unit_prefill_report_test",
    srcs = ["prefill/report_test.py"],
    main = "prefill/report_test.py",
    data = [
        "//rtp_llm/test/perf_test:multi_node/analyzer/configs/module_base.py",
        "//rtp_llm/test/perf_test:multi_node/analyzer/configs/module_conf.py",
        "//rtp_llm/test/perf_test:multi_node/analyzer/configs/qwen_480b_decode-fp8.py",
        "//rtp_llm/test/perf_test:multi_node/analyzer/configs/qwen_480b_prefill_fp8.py",
    ],
    deps = [
        ":perf_test_reports_test_support",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
    ],
)

py_test(
    name = "unit_decode_report_test",
    srcs = ["decode/report_test.py"],
    main = "decode/report_test.py",
    data = [
        "//rtp_llm/test/perf_test:multi_node/analyzer/configs/module_base.py",
        "//rtp_llm/test/perf_test:multi_node/analyzer/configs/module_conf.py",
        "//rtp_llm/test/perf_test:multi_node/analyzer/configs/qwen_480b_decode-fp8.py",
        "//rtp_llm/test/perf_test:multi_node/analyzer/configs/qwen_480b_prefill_fp8.py",
    ],
    deps = [
        ":perf_test_reports_test_support",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
    ],
)

py_test(
    name = "unit_multi_node_analyzer_test",
    srcs = ["multi_node/analyzer_test.py"],
    main = "multi_node/analyzer_test.py",
    data = [
        "//rtp_llm/test/perf_test:multi_node/analyzer/configs/module_base.py",
        "//rtp_llm/test/perf_test:multi_node/analyzer/configs/module_conf.py",
        "//rtp_llm/test/perf_test:multi_node/analyzer/configs/qwen_480b_decode-fp8.py",
        "//rtp_llm/test/perf_test:multi_node/analyzer/configs/qwen_480b_prefill_fp8.py",
    ],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_analyzer_lib",
        ":perf_test_reports_test_support",
    ],
)


py_test(
    name = "unit_prefill_resume_test",
    srcs = ["prefill/resume_test.py"],
    main = "prefill/resume_test.py",
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        ":perf_test_run_storage_test_support",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
        ":perf_test_support",
    ],
)

py_test(
    name = "unit_core_storage_test",
    srcs = ["core/storage_test.py"],
    main = "core/storage_test.py",
    deps = ["//rtp_llm/test/perf_test:perf_test_entry_lib"],
)

py_test(
    name = "unit_core_run_directory_test",
    srcs = ["core/run_directory_test.py"],
    main = "core/run_directory_test.py",
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        ":perf_test_run_storage_test_support",
    ],
)


py_test(
    name = "unit_prefill_runner_test",
    srcs = ["prefill/runner_test.py"],
    main = "prefill/runner_test.py",
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        ":perf_test_prefill_test_support",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
        ":perf_test_support",
    ],
)



py_test(
    name = "publication_policy_test",
    srcs = ["policies/publication_test.py"],
    main = "policies/publication_test.py",
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_publish_policy_lib",
    ],
)

py_test(
    name = "regression_test",
    srcs = ["policies/regression_test.py"],
    main = "policies/regression_test.py",
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_regression_lib",
        ":perf_test_regression_test_support",
        "//rtp_llm/test/perf_test:perf_test_results_lib",
    ],
)

py_test(
    name = "unit_core_results_test",
    srcs = ["core/results_test.py"],
    main = "core/results_test.py",
    deps = [
        ":perf_test_results_test_support",
        "//rtp_llm/test/perf_test:perf_test_results_lib",
    ],
)

py_test(
    name = "unit_decode_metrics_test",
    srcs = ["decode/metrics_test.py"],
    main = "decode/metrics_test.py",
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        ":perf_test_results_test_support",
        "//rtp_llm/test/perf_test:perf_test_results_lib",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
    ],
)

py_test(
    name = "unit_prefill_finalize_test",
    srcs = ["prefill/finalize_test.py"],
    main = "prefill/finalize_test.py",
    deps = [
        ":perf_test_results_test_support",
        "//rtp_llm/test/perf_test:perf_test_results_lib",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
    ],
)


py_test(
    name = "unit_execution_runtime_adapter_test",
    srcs = ["execution/runtime_adapter_test.py"],
    main = "execution/runtime_adapter_test.py",
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        ":perf_test_launch_test_support",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
    ],
)

py_test(
    name = "unit_prefill_dashsc_transport_test",
    srcs = ["prefill/dashsc_transport_test.py"],
    main = "prefill/dashsc_transport_test.py",
    deps = ["//rtp_llm/test/perf_test:perf_test_runtime_lib"],
)

py_test(
    name = "unit_decode_http_transport_test",
    srcs = ["decode/http_transport_test.py"],
    main = "decode/http_transport_test.py",
    deps = [
        ":perf_test_launch_test_support",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
    ],
)



py_test(
    name = "unified_config_test",
    srcs = ["core/config_test.py"],
    main = "core/config_test.py",
    deps = ["//rtp_llm/test/perf_test:perf_test_entry_lib", ":perf_test_support"],
)

py_test(
    name = "unit_prefill_workload_test",
    srcs = ["prefill/workload_test.py"],
    main = "prefill/workload_test.py",
    data = [
        "//rtp_llm/test/perf_test:examples/prefill_random_batch_local.json",
        "//rtp_llm/test/perf_test:examples/req_distribution.json",
    ],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        ":perf_test_support",
        ":perf_test_workload_test_support",
    ],
)

py_test(
    name = "unit_decode_distribution_test",
    srcs = ["decode/distribution_test.py"],
    main = "decode/distribution_test.py",
    data = [
        "//rtp_llm/test/perf_test:examples/prefill_random_batch_local.json",
        "//rtp_llm/test/perf_test:examples/req_distribution.json",
    ],
    deps = ["//rtp_llm/test/perf_test:perf_test_entry_lib", ":perf_test_workload_test_support"],
)



py_test(
    name = "unit_prefill_fingerprints_test",
    srcs = ["prefill/fingerprints_test.py"],
    main = "prefill/fingerprints_test.py",
    deps = ["//rtp_llm/test/perf_test:perf_test_entry_lib"],
)

py_test(
    name = "unit_ssh_executor_test",
    srcs = ["ssh/executor_test.py"],
    main = "ssh/executor_test.py",
    data = ["//rtp_llm/test/perf_test:perf_test_examples"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_ssh_lib",
        "//rtp_llm/test/perf_test:aone_lib",
        ":perf_test_support",
        ":perf_test_gang_test_support",
        ":perf_test_results_test_support",
    ],
)

py_test(
    name = "unit_ssh_maintenance_test",
    srcs = ["ssh/maintenance_test.py", "ssh/executor_test.py"],
    main = "ssh/maintenance_test.py",
    data = ["//rtp_llm/test/perf_test:perf_test_examples"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_ssh_lib",
        "//rtp_llm/test/perf_test:aone_lib",
        ":perf_test_support",
        ":perf_test_gang_test_support",
        ":perf_test_results_test_support",
    ],
)

py_test(
    name = "unit_ssh_remote_test",
    srcs = ["ssh/remote_test.py"],
    main = "ssh/remote_test.py",
    data = ["//rtp_llm/test/perf_test:perf_test_examples"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_ssh_lib",
        "//rtp_llm/test/perf_test:aone_lib",
        ":perf_test_support",
        ":perf_test_gang_test_support",
        ":perf_test_results_test_support",
    ],
)

py_test(
    name = "unit_ssh_source_test",
    srcs = ["ssh/source_test.py"],
    main = "ssh/source_test.py",
    data = ["//rtp_llm/test/perf_test:perf_test_examples"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_ssh_lib",
        "//rtp_llm/test/perf_test:aone_lib",
        ":perf_test_support",
        ":perf_test_gang_test_support",
        ":perf_test_results_test_support",
    ],
)

py_test(
    name = "unit_ssh_worker_test",
    srcs = ["ssh/worker_test.py"],
    main = "ssh/worker_test.py",
    data = ["//rtp_llm/test/perf_test:perf_test_examples"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_ssh_lib",
        "//rtp_llm/test/perf_test:aone_lib",
        ":perf_test_support",
        ":perf_test_gang_test_support",
        ":perf_test_results_test_support",
    ],
)

py_test(
    name = "unit_distributed_artifacts_test",
    srcs = ["distributed/artifacts_test.py"],
    main = "distributed/artifacts_test.py",
    data = ["//rtp_llm/test/perf_test:perf_test_examples"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_ssh_lib",
        "//rtp_llm/test/perf_test:aone_lib",
        ":perf_test_support",
        ":perf_test_gang_test_support",
        ":perf_test_results_test_support",
    ],
)

py_test(
    name = "unit_distributed_binding_test",
    srcs = ["distributed/binding_test.py"],
    main = "distributed/binding_test.py",
    data = ["//rtp_llm/test/perf_test:perf_test_examples"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_ssh_lib",
        "//rtp_llm/test/perf_test:aone_lib",
        ":perf_test_support",
        ":perf_test_gang_test_support",
        ":perf_test_results_test_support",
    ],
)

py_test(
    name = "unit_distributed_completion_test",
    srcs = ["distributed/completion_test.py"],
    main = "distributed/completion_test.py",
    data = ["//rtp_llm/test/perf_test:perf_test_examples"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_ssh_lib",
        "//rtp_llm/test/perf_test:aone_lib",
        ":perf_test_support",
        ":perf_test_gang_test_support",
        ":perf_test_results_test_support",
    ],
)

py_test(
    name = "unit_distributed_decode_sync_test",
    srcs = ["distributed/decode_sync_test.py"],
    main = "distributed/decode_sync_test.py",
    data = ["//rtp_llm/test/perf_test:perf_test_examples"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_ssh_lib",
        "//rtp_llm/test/perf_test:aone_lib",
        ":perf_test_support",
        ":perf_test_gang_test_support",
        ":perf_test_results_test_support",
    ],
)

py_test(
    name = "unit_distributed_gang_test",
    srcs = ["distributed/gang_test.py"],
    main = "distributed/gang_test.py",
    data = ["//rtp_llm/test/perf_test:perf_test_examples"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_ssh_lib",
        "//rtp_llm/test/perf_test:aone_lib",
        ":perf_test_support",
        ":perf_test_gang_test_support",
        ":perf_test_results_test_support",
    ],
)


py_test(
    name = "unit_distributed_prefill_runtime_test",
    srcs = ["distributed/prefill_runtime_test.py"],
    main = "distributed/prefill_runtime_test.py",
    data = ["//rtp_llm/test/perf_test:perf_test_examples"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_ssh_lib",
        "//rtp_llm/test/perf_test:aone_lib",
        ":perf_test_support",
        ":perf_test_gang_test_support",
        ":perf_test_results_test_support",
    ],
)

py_test(
    name = "unit_distributed_prefill_session_test",
    srcs = ["distributed/prefill_session_test.py"],
    main = "distributed/prefill_session_test.py",
    data = ["//rtp_llm/test/perf_test:perf_test_examples"],
    deps = [
        ":perf_test_prefill_test_support",
        "//rtp_llm/test/perf_test:perf_test_ssh_lib",
        "//rtp_llm/test/perf_test:aone_lib",
        ":perf_test_support",
        ":perf_test_gang_test_support",
        ":perf_test_results_test_support",
    ],
)

py_test(
    name = "unit_experiments_experiment_test",
    srcs = ["experiments/experiment_test.py"],
    main = "experiments/experiment_test.py",
    data = ["//rtp_llm/test/perf_test:perf_test_examples"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_ssh_lib",
        "//rtp_llm/test/perf_test:aone_lib",
        ":perf_test_support",
        ":perf_test_gang_test_support",
        ":perf_test_results_test_support",
    ],
)

test_suite(
    name = "unit_tests",
    tests = [
        ":unit_core_build_closure_test",
        ":unit_core_entry_test",
        ":decode_remote_test_test",
        ":unit_distributed_member_test",
        ":aone_broker_test",
        ":aone_entry_test",
        ":aone_single_test",
        ":aone_worker_test",
        ":ci_test",
        ":decode_executor_test",
        ":offline_runner_test",
        ":policy_extraction_test",
        ":publication_policy_test",
        ":regression_test",
        ":unified_config_test",
        ":unit_aone_gang_executor_test",
        ":unit_aone_gang_package_test",
        ":unit_aone_gang_worker_test",
        ":unit_aone_plan_test",
        ":unit_core_plan_test",
        ":unit_core_results_test",
        ":unit_core_run_directory_test",
        ":unit_core_storage_test",
        ":unit_decode_distribution_test",
        ":unit_decode_http_transport_test",
        ":unit_decode_metrics_test",
        ":unit_decode_planner_test",
        ":unit_decode_profiling_test",
        ":unit_decode_report_test",
        ":unit_distributed_artifacts_test",
        ":unit_distributed_binding_test",
        ":unit_distributed_completion_test",
        ":unit_distributed_decode_sync_test",
        ":unit_distributed_gang_test",
        ":unit_distributed_prefill_runtime_test",
        ":unit_distributed_prefill_session_test",
        ":unit_execution_engine_session_test",
        ":unit_execution_local_test",
        ":unit_execution_profiling_launch_test",
        ":unit_execution_runtime_adapter_test",
        ":unit_experiments_experiment_test",
        ":unit_multi_node_analyzer_test",
        ":unit_prefill_backend_integration_test",
        ":unit_prefill_dashsc_transport_test",
        ":unit_prefill_finalize_test",
        ":unit_prefill_fingerprints_test",
        ":unit_prefill_planner_test",
        ":unit_prefill_profile_test",
        ":unit_prefill_report_test",
        ":unit_prefill_resume_test",
        ":unit_prefill_runner_test",
        ":unit_prefill_workload_test",
        ":unit_ssh_executor_test",
        ":unit_ssh_maintenance_test",
        ":unit_ssh_remote_test",
        ":unit_ssh_source_test",
        ":unit_ssh_worker_test",
    ],
)

test_suite(
    name = "aone_gang_test",
    tests = [":unit_aone_gang_executor_test", ":unit_aone_gang_worker_test", ":unit_aone_gang_package_test"],
)

py_test(
    name = "unit_distributed_member_test",
    srcs = ["distributed/member_test.py"],
    main = "distributed/member_test.py",
    data = ["//rtp_llm/test/perf_test:perf_test_examples"],
    deps = ["//rtp_llm/test/perf_test:perf_test_runtime_lib", ":perf_test_support"],
)

test_suite(
    name = "engine_session_test",
    tests = [":unit_execution_engine_session_test", ":unit_execution_local_test", ":unit_prefill_backend_integration_test"],
)

test_suite(
    name = "plan_test",
    tests = [":unit_core_plan_test", ":unit_aone_plan_test", ":unit_distributed_binding_test", ":unit_prefill_planner_test", ":unit_decode_planner_test"],
)

test_suite(
    name = "prefill_profile_test",
    tests = [":unit_prefill_profile_test", ":unit_execution_profiling_launch_test", ":unit_decode_profiling_test"],
)

test_suite(
    name = "prefill_report_test",
    tests = [":unit_prefill_report_test", ":unit_decode_report_test", ":unit_multi_node_analyzer_test"],
)

test_suite(
    name = "prefill_resume_test",
    tests = [":unit_prefill_resume_test", ":unit_core_storage_test", ":unit_core_run_directory_test"],
)

test_suite(
    name = "prefill_runner_test",
    tests = [":unit_prefill_runner_test", ":unit_distributed_prefill_session_test"],
)

test_suite(
    name = "result_envelope_test",
    tests = [":unit_core_results_test", ":unit_decode_metrics_test", ":unit_prefill_finalize_test"],
)

test_suite(
    name = "runtime_adapter_test",
    tests = [":unit_execution_runtime_adapter_test", ":unit_prefill_dashsc_transport_test", ":unit_decode_http_transport_test"],
)

test_suite(
    name = "workload_test",
    tests = [":unit_prefill_workload_test", ":unit_decode_distribution_test"],
)

py_test(
    name = "decode_remote_test_test",
    srcs = ["decode/remote_entry_test.py", "//rtp_llm/test/perf_test:decode_remote_test.py"],
    main = "decode/remote_entry_test.py",
    data = ["//rtp_llm/test/perf_test:aone_defs.bzl"],
    deps = [
        ":perf_test_support",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
    ],
)

py_test(
    name = "unit_core_entry_test",
    srcs = ["core/entry_test.py", "//rtp_llm/test/perf_test:main.py"],
    main = "core/entry_test.py",
    data = [
        "//tools:perf_test_script",
        "//rtp_llm/test/perf_test:examples/prefill_local.json",
        "//rtp_llm/test/perf_test:examples/decode_grid_local.json",
    ],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        ":perf_test_entry_test_support",
        "//rtp_llm/test/perf_test:perf_test_runtime_lib",
        ":perf_test_support",
    ],
)

py_test(
    name = "unit_core_build_closure_test",
    srcs = ["core/build_closure_test.py", "//rtp_llm/test/perf_test:main.py"],
    main = "core/build_closure_test.py",
    data = [
        "//rtp_llm/test/perf_test:BUILD",
        "//rtp_llm/test/perf_test:perf_test_python_sources",
    ],
    deps = [":perf_test_entry_test_support", ":perf_test_support"],
)

test_suite(
    name = "entry_test",
    tests = [":unit_core_entry_test", ":unit_core_build_closure_test"],
)
