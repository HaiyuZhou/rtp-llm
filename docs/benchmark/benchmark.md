# RTP-LLM Unified Performance Benchmark

RTP-LLM uses one strict performance-test entry for local prefill and decode:
`tools/perf_test`. New runs accept one versioned JSON config; engine arguments,
environment, topology, workload, output, regression, and publishing policy are
declared in that file rather than forwarded as command-line flags.

## Single-Node Benchmark

Preview cache-aware prefill without starting a model:

```bash
./tools/perf_test --mode=prefill --action=run --config=rtp_llm/test/perf_test/examples/prefill_local.json --dry-run
```

Preview an input-length by global-batch decode grid:

```bash
./tools/perf_test --mode=decode --action=run --config=rtp_llm/test/perf_test/examples/decode_grid_local.json --dry-run
```

Remove `--dry-run` to execute. Prefill supports local run, resume, profile,
and offline report. Decode supports local run only. AOne and the CI macro support
prefill only. Model and tokenizer paths must already be available in the runtime.

See `rtp_llm/test/perf_test/docs/reference.md` for the capability matrix,
run-directory layout, result contract, exit codes, AOne delivery, regression,
and publishers. Mode-specific examples are in the adjacent `docs/` and
`examples/` directories.

## Multi Node Benchmark
We also provide a Python script to enable multi-node benchmark, but since it requires setting up the environment and starting the script on multiple machines, it still involves more steps than single-node testing.

For each machine to be tested, you need to create an environment in which RTP-LLM can run and support passwordless SSH access from the current machine’s port.
### Benchmark Yaml
Configure the parameters with reference to `rtp_llm/test/perf_test/multi_node/multi_benchmark_config.yaml`. Below is the detail explaination of yaml structure.

First part is the experiment name identifier
```yaml
experiment_name: "H20_Node1"
```

Second part is the test configuration and retry settings.
```yaml
test_config:
  num_retry_times: 3
  build_from_scratch: 2
  copy_test_result: true
```

Third part describe the machine list, user name and port to ssh.
```yaml
machine_config:
  ip_lists:
    - "127.0.0.1"
  run_user: "admin"
  ssh_port: 2222
```

Fourth part is for cloning code to local in ssh machine, checkout to the branch you need to test, and compilation options.
```yaml
build_config:
  git_repo_url: "git@gitlab.alibaba-inc.com:foundation_models/RTP-LLM.git"
  git_checkout_ref: "origin/feature/yiyin_fork_rtp_llm"
  # open_source_url: "git@github.com:alibaba/rtp-llm.git"
  open_source_ref: "origin/feature/yiyin_multi_benchmark"
  ft_sub_dir: "rtp_llm_perf_test"
  bazel_build_args: '" --jobs 64 --verbose_failures --config=cuda13 "'
```

Fifth part is the common configuration shared across all benchmarks.
```yaml
common_config:
  warm_up: 1
  act_type: "bf16"
  hack_layer_num: 4
  start_port: 12333
  decode_test_length: 2048
  max_context_batch_size: 1
  reserver_runtime_mem_mb: 0
```

The sixth part describes the benchmark configuration structure. Each benchmark consists of four main components:

**benchmark_name**: A unique identifier for the benchmark test case.

**fixed_config**: Contains static configuration that remain constant across all test iterations. This includes model paths, model type, and other fixed settings.
```yaml
benchmarks:
  - benchmark_name: "Qwen3-Coder-480B-A35B-Instruct"
    fixed_config:
      tokenizer_path: "Qwen/Qwen3-Coder-480B-A35B-Instruct"
      checkpoint_path: "Qwen/Qwen3-Coder-480B-A35B-Instruct"
      model_type: "qwen_3_moe"
      is_decode: true
      weight_type: "fp8"
      use_deepep_moe: true
      use_deepep_low_latency: true
      accl_fp8_cast_level: 1
      accl_low_latency_optimize: 1
      accl_dispatch_num_warp_groups: 4
      accl_combine_num_warp_groups: 4
      enable_merge_w13: true
```

**iterative_search_config**: Defines configuration that will be tested in parallel combinations. All configuration in this section must have lists of equal length, as each index position represents one test configuration. For example, if `dp_size: [1, 4]` and `tp_size: [8, 8]`, the script will test two configurations: `(dp_size=1, tp_size=8)` and `(dp_size=4, tp_size=8)`.
```yaml
    iterative_search_config:
      dp_size: [1, 4]
      tp_size: [8, 8]
      concurrency_limit: [8, 4]
      batch_size_list: ["[8]", "[4]"]
      input_len_list: ["[2048,4096,8192]", "[65536]"]
```

**recursive_search_config**: Defines configuration that will be tested using Cartesian product combinations. Each configuration can have multiple values, and all possible combinations will be tested. For example, with `enable_comm_overlap: [false, true]` and `enable_layer_micro_batch: [0, 2]`, four combinations will be tested: `(false, 0)`, `(false, 2)`, `(true, 0)`, `(true, 2)`.
```yaml
    recursive_search_config:
      enable_comm_overlap: [false, true]
      enable_layer_micro_batch: [0, 2]
```

### Run step
```shell
# in root of RTP-LLM
cd rtp_llm/test/perf_test/multi_node
# Configure multi_benchmark_config.yaml
/opt/conda310/bin/python3 multi_benchmark.py -c multi_benchmark_config.yaml
```
Ultimately, multi-node benchmark dumps trace profiles and logs to local `test_output` directory.

## Result Format
Decode result, where batch size stands for per DP Rank
``` shell
+---------------------------------------------------------------------------------------------+
|                                        Decode Result                                        |
+---------+------------+------------------+--------------+------------------+-----------------+
| Seq Len | Batch Size | Sucess/Total Req | Input/Output | Waiting Time(ms) | Decode Time(ms) |
+---------+------------+------------------+--------------+------------------+-----------------+
| 4096    | 1          | 1/1              | 4096/2048    | 0.00             | 1.75            |
| 4096    | 2          | 2/2              | 4096/2048    | 0.00             | 1.72            |
| 4096    | 4          | 4/4              | 4096/2048    | 0.00             | 1.91            |
| 4096    | 8          | 8/8              | 4096/2048    | 0.00             | 1.93            |
| 4096    | 16         | 16/16            | 4096/2048    | 0.00             | 2.00            |
| 4096    | 32         | 32/32            | 4096/2048    | 0.00             | 2.19            |
| 4096    | 48         | 48/48            | 4096/2048    | 0.00             | 2.43            |
| 4096    | 64         | 64/64            | 4096/2048    | 0.00             | 2.60            |
| 4096    | 80         | 80/80            | 4096/2048    | 0.00             | 2.87            |
+---------+------------+------------------+--------------+------------------+-----------------+
```
Trace files and logs
``` shell
test_output/
├── Experiment_H20_Node1_20251023-181554-251847
│   ├── Benchmark_Qwen3-8B_20251023-184040-471413
│   │   ├── Task_concurrency_limit-4_batch_size_list-[4]_input_len_list-[8192]
│   │   │   ├── clean.log
│   │   │   ├── copy.log
│   │   │   ├── Decode_Result.json
│   │   │   ├── kill.log
│   │   │   ├── process_logs
│   │   │   │   └── process_TEST_OUTPUT_QWEN_3_1_2_0_20251023_184240.log
│   │   │   ├── test.log
│   │   │   └── trace_files
│   │   │       ├── normal_profiler_wr0_b4_s8192_prefill0_1.json
│   │   │       └── normal_profiler_wr1_b0_s0_prefill0_1.json
│   │   └── Task_concurrency_limit-8_batch_size_list-[8]_input_len_list-[2048,4096]
│   │       ├── clean.log
│   │       ├── copy.log
│   │       ├── Decode_Result.json
│   │       ├── kill.log
│   │       ├── process_logs
│   │       │   └── process_TEST_OUTPUT_QWEN_3_1_2_0_20251023_184046.log
│   │       ├── test.log
│   │       └── trace_files
│   │           ├── normal_profiler_wr0_b8_s2048_prefill0_1.json
│   │           ├── normal_profiler_wr0_b8_s4096_prefill0_2.json
│   │           ├── normal_profiler_wr1_b0_s0_prefill0_1.json
│   │           └── normal_profiler_wr1_b0_s0_prefill0_2.json
│   ├── Benchmark_Qwen3-Coder-480B-A35B-Instruct-FP8_20251023-181634-679473
│   │   ├── Task_concurrency_limit-4_batch_size_list-[4]_input_len_list-[8192]_accl_dispatch_num_warp_groups-4_accl_combine_num_warp_groups-4
│   │   │   ├── clean.log
│   │   │   ├── copy.log
│   │   │   ├── Decode_Result.json
│   │   │   ├── kill.log
│   │   │   ├── process_logs
│   │   │   │   └── process_TEST_OUTPUT_QWEN_3_MOE_1_2_0_20251023_183008.log
│   │   │   ├── test.log
│   │   │   └── trace_files
│   │   │       ├── normal_profiler_wr0_b4_s8192_prefill0_1.json
│   │   │       └── normal_profiler_wr1_b0_s0_prefill0_1.json
│   │   ├── Task_concurrency_limit-4_batch_size_list-[4]_input_len_list-[8192]_accl_dispatch_num_warp_groups-4_accl_combine_num_warp_groups-5
│   │   │   ├── clean.log
│   │   │   ├── copy.log
│   │   │   ├── Decode_Result.json
│   │   │   ├── kill.log
│   │   │   ├── process_logs
│   │   │   │   └── process_TEST_OUTPUT_QWEN_3_MOE_1_2_0_20251023_183240.log
│   │   │   ├── test.log
│   │   │   └── trace_files
│   │   │       ├── normal_profiler_wr0_b4_s8192_prefill0_1.json
│   │   │       └── normal_profiler_wr1_b0_s0_prefill0_1.json
│   │   ├── Task_concurrency_limit-4_batch_size_list-[4]_input_len_list-[8192]_accl_dispatch_num_warp_groups-5_accl_combine_num_warp_groups-4
│   │   │   ├── clean.log
│   │   │   ├── copy.log
│   │   │   ├── Decode_Result.json
│   │   │   ├── kill.log
│   │   │   ├── process_logs
│   │   │   │   └── process_TEST_OUTPUT_QWEN_3_MOE_1_2_0_20251023_183518.log
│   │   │   ├── test.log
│   │   │   └── trace_files
│   │   │       ├── normal_profiler_wr0_b4_s8192_prefill0_1.json
│   │   │       └── normal_profiler_wr1_b0_s0_prefill0_1.json
│   │   ├── Task_concurrency_limit-4_batch_size_list-[4]_input_len_list-[8192]_accl_dispatch_num_warp_groups-5_accl_combine_num_warp_groups-5
│   │   │   ├── clean.log
│   │   │   ├── copy.log
│   │   │   ├── Decode_Result.json
│   │   │   ├── kill.log
│   │   │   ├── process_logs
│   │   │   │   └── process_TEST_OUTPUT_QWEN_3_MOE_1_2_0_20251023_183801.log
│   │   │   ├── test.log
│   │   │   └── trace_files
│   │   │       ├── normal_profiler_wr0_b4_s8192_prefill0_1.json
│   │   │       └── normal_profiler_wr1_b0_s0_prefill0_1.json
│   │   ├── Task_concurrency_limit-8_batch_size_list-[8]_input_len_list-[2048,4096]_accl_dispatch_num_warp_groups-4_accl_combine_num_warp_groups-4
│   │   │   ├── clean.log
│   │   │   ├── copy.log
│   │   │   ├── Decode_Result.json
│   │   │   ├── kill.log
│   │   │   ├── process_logs
│   │   │   │   └── process_TEST_OUTPUT_QWEN_3_MOE_1_2_0_20251023_181641.log
│   │   │   ├── test.log
│   │   │   └── trace_files
│   │   │       ├── normal_profiler_wr0_b8_s2048_prefill0_1.json
│   │   │       ├── normal_profiler_wr0_b8_s4096_prefill0_2.json
│   │   │       ├── normal_profiler_wr1_b0_s0_prefill0_1.json
│   │   │       └── normal_profiler_wr1_b0_s0_prefill0_2.json
│   │   ├── Task_concurrency_limit-8_batch_size_list-[8]_input_len_list-[2048,4096]_accl_dispatch_num_warp_groups-4_accl_combine_num_warp_groups-5
│   │   │   ├── clean.log
│   │   │   ├── copy.log
│   │   │   ├── Decode_Result.json
│   │   │   ├── kill.log
│   │   │   ├── process_logs
│   │   │   │   └── process_TEST_OUTPUT_QWEN_3_MOE_1_2_0_20251023_182004.log
│   │   │   ├── test.log
│   │   │   └── trace_files
│   │   │       ├── normal_profiler_wr0_b8_s2048_prefill0_1.json
│   │   │       ├── normal_profiler_wr0_b8_s4096_prefill0_2.json
│   │   │       ├── normal_profiler_wr1_b0_s0_prefill0_1.json
│   │   │       └── normal_profiler_wr1_b0_s0_prefill0_2.json
│   │   ├── Task_concurrency_limit-8_batch_size_list-[8]_input_len_list-[2048,4096]_accl_dispatch_num_warp_groups-5_accl_combine_num_warp_groups-4
│   │   │   ├── clean.log
│   │   │   ├── copy.log
│   │   │   ├── Decode_Result.json
│   │   │   ├── kill.log
│   │   │   ├── process_logs
│   │   │   │   └── process_TEST_OUTPUT_QWEN_3_MOE_1_2_0_20251023_182327.log
│   │   │   ├── test.log
│   │   │   └── trace_files
│   │   │       ├── normal_profiler_wr0_b8_s2048_prefill0_1.json
│   │   │       ├── normal_profiler_wr0_b8_s4096_prefill0_2.json
│   │   │       ├── normal_profiler_wr1_b0_s0_prefill0_1.json
│   │   │       └── normal_profiler_wr1_b0_s0_prefill0_2.json
│   │   └── Task_concurrency_limit-8_batch_size_list-[8]_input_len_list-[2048,4096]_accl_dispatch_num_warp_groups-5_accl_combine_num_warp_groups-5
│   │       ├── clean.log
│   │       ├── copy.log
│   │       ├── Decode_Result.json
│   │       ├── kill.log
│   │       ├── process_logs
│   │       │   └── process_TEST_OUTPUT_QWEN_3_MOE_1_2_0_20251023_182644.log
│   │       ├── test.log
│   │       └── trace_files
│   │           ├── normal_profiler_wr0_b8_s2048_prefill0_1.json
│   │           ├── normal_profiler_wr0_b8_s4096_prefill0_2.json
│   │           ├── normal_profiler_wr1_b0_s0_prefill0_1.json
│   │           └── normal_profiler_wr1_b0_s0_prefill0_2.json
│   ├── build.log
│   ├── clean.log
│   ├── host.log
│   └── kill.log
└── Experiment_H20_Node1_20251023-200243-425716
    ├── Benchmark_Qwen3-8B_20251023-205656-543908
    │   ├── Task_concurrency_limit-4_batch_size_list-[4]_input_len_list-[8192]
    │   │   ├── clean.log
    │   │   ├── copy.log
    │   │   ├── Decode_Result.json
    │   │   ├── kill.log
    │   │   ├── process_logs
    │   │   │   └── process_TEST_OUTPUT_QWEN_3_1_2_0_20251023_205847.log
    │   │   ├── test.log
    │   │   └── trace_files
    │   │       ├── normal_profiler_wr0_b4_s8192_prefill0_1.json
    │   │       └── normal_profiler_wr1_b0_s0_prefill0_1.json
    │   └── Task_concurrency_limit-8_batch_size_list-[8]_input_len_list-[2048,4096]
    │       ├── clean.log
    │       ├── copy.log
    │       ├── Decode_Result.json
    │       ├── kill.log
    │       ├── process_logs
    │       │   └── process_TEST_OUTPUT_QWEN_3_1_2_0_20251023_205700.log
    │       ├── test.log
    │       └── trace_files
    │           ├── normal_profiler_wr0_b8_s2048_prefill0_1.json
    │           ├── normal_profiler_wr0_b8_s4096_prefill0_2.json
    │           ├── normal_profiler_wr1_b0_s0_prefill0_1.json
    │           └── normal_profiler_wr1_b0_s0_prefill0_2.json
    ├── build.log
    ├── clean.log
    ├── host.log
    └── kill.log

41 directories, 116 files
```
