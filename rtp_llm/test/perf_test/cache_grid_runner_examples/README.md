# CacheGridRunner 测试配置与运行说明

本目录保存 `CacheGridRunner` 的可复现测试输入和 DSV4-Pro CP=8 启动示例。
配置文件只描述 workload geometry；模型、并行度和引擎开关由启动命令提供。

## 文件

- `dsv4_pro_prefill_smoke.json`：9 个 case，用于在全量测试前验证 scheduler mode、
  cold/cache-hit 路径和 GPU trace。
- `dsv4_pro_prefill_full_template.json`：紧凑型全量模板。runner 加载时自动展开为
  1,024 个输入长度、15,175 个 geometry。默认每个 geometry 测 3 次，共 45,525 次
  正式请求；其中 14,151 个 geometry 需要先执行 seed。

不要从旧 decode-mode 结果目录恢复。scheduler mode 修复后的 smoke 和全量测试必须分别
使用全新的 `result_dir`。

## JSON 参数

### 通用元数据

| 字段 | 是否参与执行 | 说明 |
|---|---|---|
| `schema_version` | 记录 | 配置格式版本 |
| `kind` | 记录 | workload 类型 |
| `description` | 否 | 人工说明 |
| `generator.name/version` | 记录 | 配置来源和版本 |
| `generator.cache_alignment` | 是 | `--expected_cache_block_size=0` 时，runner 从这里读取预期复用粒度 |
| `summary` | 否 | 预估规模，用于启动前检查成本；runner 仍以实际展开结果为准 |

当前 DSV4-Pro 配置中：

```text
single-rank physical block = seq_size_per_block = 512
CP size                    = 8
CP-visible cache block     = 512 × 8 = 4096
kernel block               = 128
```

因此 `generator.cache_alignment`、`cache_block_size` 和 runner 实际探测到的
`reuse_len` 粒度都必须是 4096。`seq_block_size=256` 同时控制紧凑模板的最小输入和输入
长度采样对齐，不表示 cache 可以按 256 token 复用。

### Smoke 配置

Smoke 文件使用显式 `cases`：

| 字段 | 约束 |
|---|---|
| `case_id` | 唯一整数；也是 `--cache_profile_case_ids` 使用的编号 |
| `batch_size` | 本目录的 DSV4 模板固定为 1；当前 runner 也支持显式 grouped batch case |
| `input_len` | 完整输入 token 数，必须大于 0 |
| `cache_len` | 请求复用的前缀 token 数，必须满足 `0 <= cache_len < input_len` |
| `description` | 人工说明，runner 忽略 |

cache-hit case 还必须满足：

```text
cache_len % 4096 == 0
cache_len + cache_commit_tail_tokens <= input_len
```

需要测试 batch>1 时，可在显式 case 中增加 `request_groups`，各 group 的 `count` 总和必须
等于 `batch_size`；`prefix_policy` 可取 `independent` 或 `shared_by_group`。当前 cache
profiler 只支持本目录这种未分组的 batch=1 case，因此 grouped batch 不得加入
`--cache_profile_case_ids`。

### 全量模板

全量文件不展开 `cases`，而是使用 runner 原生支持的紧凑参数：

| 字段 | 当前值 | 修改效果 |
|---|---:|---|
| `seq_generation.kind` | `linear_with_dense_prefix` | 必须保持该值，这是当前支持的自动展开方式 |
| `seq_generation.count` | 1024 | 输入长度采样数量；减小可缩短测试时间 |
| `seq_generation.max_seq_len` | 1048575 | 最大输入；加上 1 个输出 token 后严格等于 1M |
| `seq_block_size` | 256 | 最小输入和普通输入长度的采样对齐 |
| `cache_block_size` | 4096 | cache 请求对齐及每个 case 至少保留的新计算尾部 |
| `cache_ratios` | 0 到 0.875 | 每个输入的 cache 比例；runner 还会自动加入 near-full case |

修改 `count`、`max_seq_len`、`seq_block_size`、`cache_block_size` 或 `cache_ratios` 后，
必须重新运行下文的预检命令，并同步更新 `summary`。`summary` 写错不会改变实际 case，
但会误导请求量和运行时间评估。

## Runner 已有默认值

以下参数当前可以不写：

| 参数 | 默认值/来源 |
|---|---|
| `--cache_measure_runs` | 3 |
| `--cache_request_timeout` | 7200 秒 |
| `--cache_commit_tail_tokens` | 4096 |
| `--cache_request_transport` | `dashsc_input_ids` |
| `--cache_grpc_port` | 0，即 HTTP port + 8 |
| `--expected_cache_block_size` | 0，然后从 JSON 的 `generator.cache_alignment` 读取 4096 |
| `--cache_checkpoint_every` | 100 |
| `--cache_profile_runs` | 0，默认不采 cache trace |
| `--cache_profile_trace_timeout` | 120 秒 |
| `--dp_size` | 1 |

`--batch_size` 和 `--input_len` 在 cache-grid 分支由 JSON case 决定，也不需要传。
`EngineServer` 会为该入口设置 `REUSE_CACHE=1`；引擎的 `enable_device_cache` 默认开启、
`enable_cuda_graph` 默认关闭，因此命令中也可以省略这三项。

## 必须显式配置

- `--cache_grid_json`：选择 smoke 或全量配置。
- `--result_dir`：每轮首次运行使用全新目录。
- `--partial=2`：默认值 0 不符合 cache-grid 的 prefill-only 约束。
- `--decode_test_length=1`：默认值 10 不符合首 token 测试口径。
- `--concurrency_limit=1`：默认值 64 不符合 batch=1 串行基线。
- 模型路径：`model_type`、`checkpoint_path`、`tokenizer_path`。
- DSV4 拓扑：`tp_size=8`、`ep_size=8`、`world_size=8`、`cp_rotate_method=ALL_GATHER`、
  `prefill_cp_kv_cache_sharded=1`。
- 正式基线形状和精度：`max_seq_len`、`max_batch_tokens_size`、两个 block size、FP8 KV、
  DeepEP、激活类型、权重加载方式和显存预留。

`PERF_PROFILE_RUNS` 属于普通 `GridRunner` 的 profiler；cache-grid 的诊断由
`--cache_profile_runs` 独立控制。建议显式设置 `PERF_PROFILE_RUNS=0`，避免继承 shell 中的旧值。

## 启动前预检

在仓库根目录执行：

```bash
python3 - <<'PY'
import json
from pathlib import Path
from rtp_llm.test.perf_test.batch_decode_test import (
    _load_cache_grid_cases,
    _resolve_cache_block_size,
)

path = Path(
    "rtp_llm/test/perf_test/cache_grid_runner_examples/"
    "dsv4_pro_prefill_full_template.json"
)
payload = json.loads(path.read_text(encoding="utf-8"))
cases = _load_cache_grid_cases(str(path))
block = _resolve_cache_block_size(payload, 0)
seeds = sum(case["cache_len"] > 0 for case in cases)
print(
    {
        "cases": len(cases),
        "inputs": len({case["input_len"] for case in cases}),
        "cache_block": block,
        "measure_requests": len(cases) * 3,
        "seed_requests": seeds,
    }
)
PY
```

当前模板应输出：

```text
{'cases': 15175, 'inputs': 1024, 'cache_block': 4096, 'measure_requests': 45525, 'seed_requests': 14151}
```

## 公共模型和引擎参数

下面两个命令共用同一组参数。当前可读模型路径是
`/data4/nanjun.cp/DeepSeek-V4-Pro`；换模型时只修改 `DSV4_CACHE_MODEL_DIR`。

```bash
cd /data7/zhouhaiyu.zhy/RTP-LLM/github-opensource

export DSV4_CACHE_MODEL_DIR=/data4/nanjun.cp/DeepSeek-V4-Pro
export DSV4_CACHE_CONFIG_DIR="$PWD/rtp_llm/test/perf_test/cache_grid_runner_examples"

DSV4_CACHE_COMMON_ARGS=(
  "--test_arg=--partial=2"
  "--test_arg=--decode_test_length=1"
  "--test_arg=--concurrency_limit=1"
  "--test_arg=--model_type=deepseek_v4"
  "--test_arg=--checkpoint_path=${DSV4_CACHE_MODEL_DIR}"
  "--test_arg=--tokenizer_path=${DSV4_CACHE_MODEL_DIR}"
  "--test_arg=--max_seq_len=1048576"
  "--test_arg=--max_batch_tokens_size=1048576"
  "--test_arg=--tp_size=8"
  "--test_arg=--ep_size=8"
  "--test_arg=--world_size=8"
  "--test_arg=--cp_rotate_method=ALL_GATHER"
  "--test_arg=--prefill_cp_kv_cache_sharded=1"
  "--test_arg=--seq_size_per_block=512"
  "--test_arg=--kernel_seq_size_per_block=128"
  "--test_arg=--fp8_kv_cache=1"
  "--test_arg=--use_deepep_moe=1"
  "--test_arg=--use_deepep_low_latency=0"
  "--test_arg=--act_type=BF16"
  "--test_arg=--load_method=fastsafetensors"
  "--test_arg=--reserver_runtime_mem_mb=81920"
  "--test_env=WORLD_SIZE=8"
  "--test_env=DG_JIT_CPP_STANDARD=20"
  "--test_env=DG_JIT_CACHE_DIR=/data7/zhouhaiyu.zhy/dsv4_perf_work/jit_cache"
  "--test_env=TRITON_CACHE_DIR=/data7/zhouhaiyu.zhy/dsv4_perf_work/triton_cache"
  "--test_env=TILELANG_CACHE_DIR=/data7/zhouhaiyu.zhy/dsv4_perf_work/tilelang_cache"
  "--test_env=DSV4_USE_MEGA_MOE=1"
  "--test_env=DSV4_CHUNK_TOKENS=8192"
  "--test_env=DSV4_PREFILL_CP_OVERLAP=0"
  "--test_env=PERF_PROFILE_RUNS=0"
  "--test_env=TOKENIZERS_PARALLELISM=false"
)
```

不要同时从 profile 注入另一套模型路径或 cache block 参数。最终 argv 中每个关键参数应只
出现一次。

## 第一步：运行 smoke 并验证 trace

```bash
export DSV4_CACHE_GRID_JSON="${DSV4_CACHE_CONFIG_DIR}/dsv4_pro_prefill_smoke.json"
export DSV4_CACHE_RESULT_DIR="/data7/zhouhaiyu.zhy/tmp/dsv4_pro_prefill_smoke_$(date +%Y%m%d_%H%M%S)"

bazelisk test //rtp_llm/test/perf_test:cache_grid_perf_test \
  --config=cuda13 --config=sm10x \
  --test_timeout=345600 \
  --test_output=streamed \
  --nocache_test_results \
  "${DSV4_CACHE_COMMON_ARGS[@]}" \
  "--test_arg=--cache_grid_json=${DSV4_CACHE_GRID_JSON}" \
  "--test_arg=--result_dir=${DSV4_CACHE_RESULT_DIR}" \
  --test_arg=--cache_profile_runs=1 \
  --test_arg=--cache_profile_case_ids \
  --test_arg=0 \
  --test_arg=3 \
  --test_arg=5 \
  --test_arg=--cache_profile_trace_timeout=180
```

验收条件：

1. perf 请求前日志出现 `mode=prefill` 的 scheduler 更新；
2. case 0、3、5 的输入所属 rank trace 分别显示全局计算量 4096、65536、4096，
   并满足 `ctx_batch=1, gen_batch=0`；
3. case 5 三轮 `cache_len_observed` 都是 61440；
4. 所有正式请求 `output_len=1` 且 `status=ok`。

## 第二步：运行全量模板

只有 smoke 验收通过后再运行。全量测试必须换一个新的结果目录，正式性能测量不开 profiler：

```bash
export DSV4_CACHE_GRID_JSON="${DSV4_CACHE_CONFIG_DIR}/dsv4_pro_prefill_full_template.json"
export DSV4_CACHE_RESULT_DIR="/data7/zhouhaiyu.zhy/tmp/dsv4_pro_prefill_full_$(date +%Y%m%d_%H%M%S)"

bazelisk test //rtp_llm/test/perf_test:cache_grid_perf_test \
  --config=cuda13 --config=sm10x \
  --test_timeout=345600 \
  --test_output=streamed \
  --nocache_test_results \
  "${DSV4_CACHE_COMMON_ARGS[@]}" \
  "--test_arg=--cache_grid_json=${DSV4_CACHE_GRID_JSON}" \
  "--test_arg=--result_dir=${DSV4_CACHE_RESULT_DIR}"
```

第一次运行不要传 `--require_cache_resume`。任务中断后，只有在代码、JSON、模型和全部引擎
参数都没有变化时，才用相同命令和相同 `DSV4_CACHE_RESULT_DIR` 追加：

```bash
--test_arg=--require_cache_resume
```

结果目录中的主要文件：

```text
cache_grid_results.json
cache_grid_results.journal.jsonl
cache_grid_progress.json
test_info.json
timelines/                    # 只有 smoke 的 cache profiler 会生成
cache_profiles/               # profiler manifest
```
