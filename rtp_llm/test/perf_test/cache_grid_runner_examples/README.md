# CacheGridRunner 测试配置与运行说明

本目录保存 `CacheGridRunner` 的可复现测试输入和 DSV4-Pro CP=8 启动示例。
配置文件只描述 workload geometry；模型、并行度和引擎开关由启动命令提供。

## 文件

- `dsv4_pro_prefill_smoke.json`：9 个 case，用于在全量测试前验证 scheduler mode、
  cold/cache-hit 路径和 GPU trace。
- `dsv4_pro_prefill_batch_smoke.json`：3 个固定组批 case，覆盖 batch=2/4、独立前缀、
  组内共享前缀和混合 cold/cache-hit。默认 3 轮，共 30 条正式请求、7 条 seed 请求。
- `dsv4_pro_prefill_full_template.json`：紧凑型全量模板。runner 加载时自动展开为
  1,024 个输入长度、44,505 个 geometry。默认每个 geometry 测 3 次，共 133,515 次
  正式请求；其中 43,481 个 geometry 需要先执行 seed。

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
| `batch_size` | 单请求 smoke 和全量模板为 1；batch smoke 使用 2/4 |
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
profiler 只支持未分组的 batch=1 case，因此 grouped batch 不得加入
`--cache_profile_case_ids`。

### Batch 配置

`dsv4_pro_prefill_batch_smoke.json` 中的请求长度单位均为 token：

| case_id | 正式 batch | 请求构成（input_len / cache_len × 数量） | 前缀策略 | seed batch |
|---|---:|---|---|---:|
| 0 | 2 | 8192 / 4096 × 2 | 默认 `independent` | 2 |
| 1 | 4 | 8192 / 4096 × 2，8192 / 0 × 1，16384 / 12288 × 1 | `independent` | 3 |
| 2 | 4 | 与 case 1 相同 | `shared_by_group` | 2 |

- `request_groups[].count`：组内请求数，各组之和必须等于 `batch_size`。
- `request_groups[].input_len/cache_len`：该组每条请求的完整输入长度和缓存前缀长度；
  每组都要满足上述缓存对齐和 commit tail 约束。
- `prefix_policy=independent`：每条请求使用独立前缀。
- `prefix_policy=shared_by_group`：同组共享前缀，不同组独立；每个共享前缀只 seed 一次。

case 0 展示兼容写法：省略 `request_groups`，runner 将顶层长度复制 `batch_size` 次。
case 1/2 使用显式分组，不必填写顶层 `input_len/cache_len`。cold 请求不需要 seed。

执行时先将专用 `BatchDecodeScheduler` 设为 seed 数量 S，并发发送 S 条 seed；
等全部完成后切换为正式 batch B，再逐轮发送完整的 B 条请求。S=0 时跳过 seed。
case 结束恢复 batch=1。该模式要求 **DP=1 且测试服务没有其他请求流量**。
入口会自动提高准入并发数、上下文 batch 上限和 token 容量，确保容纳完整 batch；
仅提高客户端并发数不能保证同一次 forward。

批量结果使用 `execution_mode=scheduler_fixed_batch`，记录 `seed_batch_size`、
`scheduler_batch_size` 和 `runs[].requests[]` 的逐请求校验。
`median_batch_wall_time_ms` 是整批完成耗时的中位数，不含 seed 和调度设置耗时。
case 1/2 每轮预期缓存命中均为 `[4096, 4096, 0, 12288]`。
更多细节见 [批量模式说明](../cache_grid_batches.md)。

### 全量模板

全量文件不展开 `cases`，而是使用 runner 原生支持的紧凑参数：

| 字段 | 当前值 | 修改效果 |
|---|---:|---|
| `seq_generation.kind` | `linear_with_dense_prefix` | 必须保持该值，这是当前支持的自动展开方式 |
| `seq_generation.count` | 1024 | 输入长度采样数量；减小可缩短测试时间 |
| `seq_generation.max_seq_len` | 1048575 | 最大输入；加上 1 个输出 token 后严格等于 1M |
| `seq_block_size` | 256 | 最小输入和普通输入长度的采样对齐 |
| `cache_block_size` | 4096 | cache 请求对齐及每个 case 至少保留的新计算尾部 |
| `cache_ratio_interval` | 0.02 | 自动生成 0、0.02、...、0.98；runner 还会加入 near-full case |

`cache_ratio_interval` 必须是 `(0, 1)` 内的有限数值。runner 按
`0, interval, 2 * interval, ... < 1` 生成均匀比例，最多生成 10,000 个比例；对每个输入长度，
比例仍会按 `cache_block_size` 向下对齐、去重，并自动补一个 near-full case。若需要非均匀
采样，也可以删掉 `cache_ratio_interval`，改用显式 `cache_ratios` 数组；两个字段不能同时配置。

修改 `count`、`max_seq_len`、`seq_block_size`、`cache_block_size`、
`cache_ratio_interval` 或 `cache_ratios` 后，
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

- `--cache_grid_json`：选择单请求 smoke、batch smoke 或全量配置。
- `--result_dir`：每轮首次运行使用全新目录。
- `--partial=2`：默认值 0 不符合 cache-grid 的 prefill-only 约束。
- `--decode_test_length=1`：默认值 10 不符合首 token 测试口径。
- `--concurrency_limit=1`：用于 batch=1 串行基线；batch 配置会自动提高到至少最大 batch size。
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
groups = [
    case.get("request_groups", [{"count": case["batch_size"],
                                 "input_len": case["input_len"],
                                 "cache_len": case["cache_len"]}])
    for case in cases
]
seeds = sum(
    (1 if case.get("prefix_policy") == "shared_by_group" else group["count"])
    for case, case_groups in zip(cases, groups)
    for group in case_groups if group["cache_len"] > 0
)
print(
    {
        "cases": len(cases),
        "inputs": len({g["input_len"] for case_groups in groups for g in case_groups}),
        "cache_block": block,
        "measure_requests": sum(case["batch_size"] for case in cases) * 3,
        "seed_requests": seeds,
    }
)
PY
```

当前模板应输出：

```text
{'cases': 44505, 'inputs': 1024, 'cache_block': 4096, 'measure_requests': 133515, 'seed_requests': 43481}
```

将预检代码中的文件名换成 `dsv4_pro_prefill_batch_smoke.json`，应输出：

```text
{'cases': 3, 'inputs': 2, 'cache_block': 4096, 'measure_requests': 30, 'seed_requests': 7}
```

## 公共模型和引擎参数

下面的单请求、batch 和全量命令共用同一组参数。当前可读模型路径是
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

## 运行 batch smoke

使用上面的 `DSV4_CACHE_COMMON_ARGS`，无需重复指定 `--batch_size`。公共参数中的
`concurrency_limit=1` 会自动提高到 4，`max_context_batch_size` 至少为 4。
不要添加单请求 smoke 的 `--cache_profile_runs` 或 `--cache_profile_case_ids`。

```bash
export DSV4_CACHE_GRID_JSON="${DSV4_CACHE_CONFIG_DIR}/dsv4_pro_prefill_batch_smoke.json"
export DSV4_CACHE_RESULT_DIR="/data7/zhouhaiyu.zhy/tmp/dsv4_pro_prefill_batch_$(date +%Y%m%d_%H%M%S)"

bazelisk test //rtp_llm/test/perf_test:cache_grid_perf_test \
  --config=cuda13 --config=sm10x \
  --test_timeout=345600 \
  --test_output=streamed \
  --nocache_test_results \
  "${DSV4_CACHE_COMMON_ARGS[@]}" \
  "--test_arg=--cache_grid_json=${DSV4_CACHE_GRID_JSON}" \
  "--test_arg=--result_dir=${DSV4_CACHE_RESULT_DIR}"
```

验收时检查 3 个 case 均为 `status=ok`，seed batch 依次为 2、3、2，正式 batch
依次为 2、4、4。服务端 `engine.log` 的 `BatchDecodeScheduler::schedule` 应显示
对应的 `running_streams_.size()`；这比客户端峰值并发数更直接反映实际组批。
如另行采集 forward trace，应在输入所属 rank 检查 `ctx_batch=B, gen_batch=0`。

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

### 中断后续测参数

| 参数 | 作用 | 推荐场景 |
|---|---|---|
| `--require_cache_resume` | 要求 `result_dir/cache_grid_results.json` 必须存在，否则在加载 tokenizer 和模型前报错 | 正常中断后的严格续测 |
| `--allow_resume_mismatch` | 发现旧 checkpoint 与当前参数不一致时只记录 warning 并继续 | 仅限人工审核后的故障恢复，不用于正式数据 |

即使不传 `--require_cache_resume`，只要同一 `result_dir` 中已有合法 checkpoint，runner
也会自动恢复并跳过其中所有 `status=ok` 的 case。推荐在续测时加该参数，是为了防止目录
变量写错或 checkpoint 丢失后悄悄开始一轮新测试。

首次运行两个参数都不要传。中断后的标准做法是保持以下内容完全不变：

- 代码 commit；
- `cache_grid_json` 内容及 SHA256；
- profile 内容及 SHA256（如果使用）；
- 模型路径和模型/权重；
- 测量轮数、request transport、commit tail 和 expected block size；
- 并行度、最大长度、cache、MoE 及其他引擎参数和相关环境变量。

然后重新执行原命令、复用原 `DSV4_CACHE_RESULT_DIR`，仅追加：

```bash
--test_arg=--require_cache_resume
```

恢复校验会比较 `grid_sha256`、`profile_sha256`、`measure_runs`、
`cache_commit_tail_tokens`、`request_transport`、`expected_block_size` 和完整
`run_config_sha256`。任一项不一致时，默认在启动模型前终止，不会覆盖原 checkpoint。

`--allow_resume_mismatch` 不会创建或寻找 checkpoint，也不等于
`--require_cache_resume`。如确实需要强制从一个参数不一致的目录恢复，应同时追加：

```bash
--test_arg=--require_cache_resume \
--test_arg=--allow_resume_mismatch
```

这会复用旧 checkpoint 中已经成功的 case，并用当前配置执行剩余 case，因而可能把不同代码、
模型、block size、transport 或测量口径的数据混入同一结果。正式性能报告禁止这样做。
以下场景必须新建 `result_dir`，不能使用 `--allow_resume_mismatch`：

- scheduler mode 修复前的旧 decode-mode 数据；
- 修改模型、并行度、cache block、grid 或测量轮数；
- 无法解释 `run_config_sha256` 差异；
- 原目录只有 journal、缺少基础 `cache_grid_results.json`。

使用 pipeline 续测时，这两个参数属于 runner 参数，必须放在第二个 `--` 后，不要写成
`--test_arg`：

```bash
bazelisk run //rtp_llm/test/perf_test:run_cache_grid_pipeline -- \
  --cache-grid-json="${DSV4_CACHE_GRID_JSON}" \
  --result-dir="${DSV4_CACHE_PIPELINE_RESULT_DIR}" \
  --batch-size=1 \
  --estimator=median \
  -- \
  ...首次运行的全部 runner/引擎参数... \
  --require_cache_resume
```

`--skip-test` 会完全跳过 runner，只重新拟合和绘图，因此不要与上述两个续测参数组合。
`--cache_profile_only` 会创建隔离的 replay 目录，也不能与 `--require_cache_resume` 同时使用。

结果目录中的主要文件：

```text
cache_grid_results.json
cache_grid_results.journal.jsonl
cache_grid_progress.json
test_info.json
timelines/                    # 只有 smoke 的 cache profiler 会生成
cache_profiles/               # profiler manifest
```

## 一条命令完成全量测试、拟合和绘图

`run_cache_grid_pipeline` 依次执行 cache-grid runner、公式拟合、静态 SVG 和交互式 HTML
绘图。只有 `cache_grid_results.json` 标记为完整时才会进入后处理。

pipeline 的参数分为两段：第一个 `--` 后是 pipeline 自己的参数，第二个 `--` 后是直接转发给
`batch_decode_test.py` 的 runner/引擎参数。这里不能复用前面的
`DSV4_CACHE_COMMON_ARGS`，因为其中的 `--test_arg` 和 `--test_env` 只适用于 `bazelisk test`。

先把原先通过 `--test_env` 传递的环境变量导出到当前 shell：

```bash
export WORLD_SIZE=8
export DG_JIT_CPP_STANDARD=20
export DG_JIT_CACHE_DIR=/data7/zhouhaiyu.zhy/dsv4_perf_work/jit_cache
export TRITON_CACHE_DIR=/data7/zhouhaiyu.zhy/dsv4_perf_work/triton_cache
export TILELANG_CACHE_DIR=/data7/zhouhaiyu.zhy/dsv4_perf_work/tilelang_cache
export DSV4_USE_MEGA_MOE=1
export DSV4_CHUNK_TOKENS=8192
export DSV4_PREFILL_CP_OVERLAP=0
export PERF_PROFILE_RUNS=0
export TOKENIZERS_PARALLELISM=false
```

运行全量 pipeline：

```bash
cd /data7/zhouhaiyu.zhy/RTP-LLM/github-opensource

export DSV4_CACHE_MODEL_DIR=/data4/nanjun.cp/DeepSeek-V4-Pro
export DSV4_CACHE_CONFIG_DIR="$PWD/rtp_llm/test/perf_test/cache_grid_runner_examples"
export DSV4_CACHE_GRID_JSON="${DSV4_CACHE_CONFIG_DIR}/dsv4_pro_prefill_full_template.json"
export DSV4_CACHE_PIPELINE_RESULT_DIR="/data7/zhouhaiyu.zhy/tmp/dsv4_pro_prefill_pipeline_$(date +%Y%m%d_%H%M%S)"

bazelisk run //rtp_llm/test/perf_test:run_cache_grid_pipeline \
  --config=cuda13 --config=sm10x -- \
  --cache-grid-json="${DSV4_CACHE_GRID_JSON}" \
  --result-dir="${DSV4_CACHE_PIPELINE_RESULT_DIR}" \
  --batch-size=1 \
  --estimator=median \
  -- \
  --decode_test_length=1 \
  --concurrency_limit=1 \
  --model_type=deepseek_v4 \
  --checkpoint_path="${DSV4_CACHE_MODEL_DIR}" \
  --tokenizer_path="${DSV4_CACHE_MODEL_DIR}" \
  --max_seq_len=1048576 \
  --max_batch_tokens_size=1048576 \
  --tp_size=8 \
  --ep_size=8 \
  --world_size=8 \
  --cp_rotate_method=ALL_GATHER \
  --prefill_cp_kv_cache_sharded=1 \
  --seq_size_per_block=512 \
  --kernel_seq_size_per_block=128 \
  --fp8_kv_cache=1 \
  --use_deepep_moe=1 \
  --use_deepep_low_latency=0 \
  --act_type=BF16 \
  --load_method=fastsafetensors \
  --reserver_runtime_mem_mb=81920
```

pipeline 自动管理 `--cache_grid_json`、`--partial=2` 和 `--result_dir`，不要在第二个 `--`
后重复传入。`--cache_measure_runs=3`、`--cache_commit_tail_tokens=4096`、
`--cache_request_transport=dashsc_input_ids` 使用 runner 默认值；预期 cache block 会从模板的
`generator.cache_alignment=4096` 读取。

输出除 `cache_grid_results.json` 外，还包括：

```text
formula/deepseek_v4_prefill_formula.txt
formula/fit_report.json
formula/fit_gap.svg
prefill_3d.svg
prefill_cold_miss.svg
prefill_3d.interactive.html
pipeline_summary.json
```

拟合质量门禁未通过时，pipeline 仍会生成图表，并在 `pipeline_summary.json` 中写入
`fit_rejected`，最终命令返回非零状态。不要因为图表已经生成就把公式视为可交付。

若测试已经完整结束，只需要重新拟合或绘图，可复用原结果目录并加 `--skip-test`：

```bash
bazelisk run //rtp_llm/test/perf_test:run_cache_grid_pipeline -- \
  --cache-grid-json="${DSV4_CACHE_GRID_JSON}" \
  --result-dir="${DSV4_CACHE_PIPELINE_RESULT_DIR}" \
  --batch-size=1 \
  --estimator=median \
  --skip-test
```
# 随机混合 batch 的交互图

使用 `../generate_batch_interactive_chart.py` 从已分析的
`interactive_data.json` 生成离线 HTML（依赖 Python `plotly`，不需要 GPU）。
在仓库根目录执行：

```bash
python rtp_llm/test/perf_test/generate_batch_interactive_chart.py \
  --input /path/to/run/report/interactive_data.json \
  --output /path/to/run/report/batch_cache_compute_ttft.html
```

输入是分析后的 JSON，不是 runner 原始 `cache_grid_results.json`：顶层包含
`measurement_contract=batch_input_ids_wall_barrier_to_last_response_ms` 和 `rows`。
每行包含 case_id、batch_size、band（cache/compute 的 low/low、low/high、high/low、high/high）、
cached_tokens/compute_tokens 总量及 mean/min/max、median/min/max_batch_ttft_ms、
p95_request_ttft_ms、formal_rounds_ms，以及 request_distribution
（每请求 `[input_tokens, observed_cache_tokens, compute_tokens]`）。
脚本只呈现输入统计，不重新选择预热轮次或计算测量中位数；不完整数据加 `--partial`。

三种视图均在轴上及图外明确标注含义、单位：

- 默认：X=batch 请求数，Y=每请求平均实际 cache，Z=每请求平均 compute，颜色=整批 TTFT。
- 总量：Y/Z 改为整批 cache/compute token 总量。
- 时延：X=平均 compute，Y=平均实际 cache，Z=整批 TTFT，颜色=batch。

长度单位 Ki tokens=1024 tokens；TTFT 单位 ms，计时到整批最后一条响应，
不是纯 GPU prefill 时间。支持旋转、筛选和点击请求明细，不插值未测量曲面。
页面内嵌 Plotly，无 CDN 依赖。可执行交互逻辑回归检查（需要 Node.js，不代替浏览器渲染验收）：

```bash
node rtp_llm/test/perf_test/generate_batch_interactive_chart_test.js \
  /path/to/run/report/batch_cache_compute_ttft.html \
  /path/to/run/report/interactive_data.json
```
