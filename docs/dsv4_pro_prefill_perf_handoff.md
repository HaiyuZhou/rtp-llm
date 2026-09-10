# DeepSeek-V4-Pro Prefill 性能测试交接

这份手册给接手测试的同事使用。它描述的是当前仓库里的单机 DSV4-Pro prefill 测试，不是线上压测脚本。测试前先确认模型目录、GPU 和代码版本，三者有一个不对就不要启动。

## 0. 最短执行路径

先准备一个**本地、容器内可见**的模型目录，再按下面顺序操作：

```bash
# 1) 确认代码版本（工作树有未提交改动时不要强行切分支）
cd /data0/luoli.hn/work/rtp_llm_4/dsv4-cache-affinity-1OIKUM
git branch --show-current
git status --short

# 2) 确认模型和 8 张 GPU
export MODEL_DIR=/data1/serina.wzq/DeepSeek-V4-Pro
test -r "$MODEL_DIR/config.json" && test -r "$MODEL_DIR/tokenizer_config.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# 3) 先只跑单元测试，再启动手工 perf target
bazelisk test //rtp_llm/test/perf_test:batch_decode_test_test \
  --config=cuda13 --config=sm10x --test_output=errors
# 具体的 DSV4 target 见第 6 节；target 名 dsv4_pro_prefill_handoff
```

`dsv4_pro_prefill_handoff` 是交接手册建议的新 target 名称，仓库现有 BUILD 不会自动生成它。需要先复制现有 DSV4-Pro prefill stanza，并按第 5 节替换参数；不要直接运行旧的 64K profile target 当作 1M 基线。

## 1. 测试口径

当前标准入口是 `rtp_llm/test/perf_test/batch_decode_test.py`，通过 `GridRunner` 逐个执行 `batch_size × input_len`。本手册的基线口径是：

| 项目 | 基线 |
|---|---|
| 模型 | DeepSeek-V4-Pro |
| 阶段 | prefill only，`--partial=2` |
| GPU | 8 卡，CUDA 13.2，SM100/L20D 运行环境 |
| 并行 | `tp=8, dp=1, ep=8, world=8` |
| CP | `ALL_GATHER` |
| KV cache | FP8，复用开启 |
| batch | `1`（本地 context batch 也设为 `1`） |
| 输出 | 每个请求 1 token；prefill 时间就是首 token 前的计算时间 |
| 测量 | 建议每个 geometry 预热 1 次、正式测量 3 次；正式测量不要开启 profiler |
| 1M 边界 | `input_len=1,048,575`、`decode_test_length=1`、`max_seq_len=1,048,576` |

这里的 `input_len` 是完整输入长度，不是新计算 token 数。带前缀缓存的场景需要额外记录 `observed_cache_len`，新计算量是 `input_len - observed_cache_len`。

标准 `GridRunner` 只扫描 batch 和 input length，本身没有 `--cache_len` 参数。因此，不能把普通 grid 的结果当成 cache-hit 结果；cache-hit 测试必须由能执行“seed → hit request → 校验 reuse_len”的专用 runner 完成，并把实际 `reuse_len` 写入结果。

### 1.1 cache-grid prefill mode 修复（2026-09-10）

旧版 cache-grid 分支绕过了标准 `GridRunner` 的 `_set_concurrency()`，服务启动后没有把
scheduler 从默认 decode mode 切到 prefill mode。即使请求携带长输入、只输出 1 token，
GPU 也可能实际执行 `ctx_batch=0, gen_batch=1, tokens=1` 的单 token decode；“输出 1 token”
只定义请求输出长度，不能证明长输入已经走过 prefill。

修复后的 `batch_decode_test.py` 会在 cache block 探测、seed 和正式测量之前调用
`/update_scheduler_info`，固定发送 `batch_size=1, mode=prefill`；接口不可用或响应不符合
预期时直接终止。因此不新增 scheduler CLI 参数，测试仍使用 `--partial=2` 和
`--decode_test_length=1`。但必须按第 7 节先做 GPU trace 验证，不能只根据接口 ACK、请求成功、
`reuse_len` 或时延趋势认定为 prefill。

修复只影响之后新启动的服务，不会修正已经落盘的数据。目录
`/data7/zhouhaiyu.zhy/tmp/dsv4_pro_center_dense_input_20260908` 的时延、吞吐、cache 收益、
三维图和拟合结论不得继续使用；其中输入长度、请求 cache 长度、observed reuse 和请求成功状态
可以作为元数据留档。

## 2. 代码版本

当前代码分成两个仓库：

```text
GitHub inner repo: /data0/luoli.hn/work/rtp_llm_4/dsv4-cache-affinity-1OIKUM
branch: feat/dsv4_on_dev

GitLab outer repo: /data0/luoli.hn/work/rtp_llm_4
branch: develop/wangyin_ds_v4_20260424
```

常用检查：

```bash
cd /data0/luoli.hn/work/rtp_llm_4/dsv4-cache-affinity-1OIKUM
git status --short
git branch --show-current
git rev-parse HEAD
```

不要在两个仓库之间复制源码，也不要用 `git clean -fdx` 清理外层工作区；外层目录可能有其他任务留下的构建缓存和工作树。

如果两个仓库都没有本地改动，切换版本可以这样做：

```bash
cd /data0/luoli.hn/work/rtp_llm_4/dsv4-cache-affinity-1OIKUM
git switch feat/dsv4_on_dev

cd /data0/luoli.hn/work/rtp_llm_4
git switch develop/wangyin_ds_v4_20260424
```

切换前后都记录 `git rev-parse HEAD`。如果 `git status --short` 有输出，先让负责人确认这些改动是否属于本次测试；不要为了切分支删除它们。

## 3. 运行环境

在容器内确认以下命令可用：

```bash
/opt/conda310/bin/python --version       # Python 3.10
/opt/conda310/bin/python - <<'PY'
import torch
print(torch.__version__)
print('cuda_available=', torch.cuda.is_available())
print('device_count=', torch.cuda.device_count())
PY
nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_mode --format=csv
bazelisk --version
/opt/rh/gcc-toolset-12/root/usr/bin/g++ --version
/usr/local/cuda/bin/nvcc --version
```

L20D/CUDA 13 的构建配置是：

```bash
--config=cuda13 --config=sm10x
```

`.bazelrc` 已为 `cuda13` 设置 GCC 12、CUDA 13.2 和 `TF_CUDA_COMPUTE_CAPABILITIES`。如果是从一个没有加载 `.bazelrc` 的 shell 启动，至少补上：

```bash
export DG_JIT_CPP_STANDARD=20
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export CUDAHOSTCXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export NVCC_PREPEND_FLAGS=-ccbin=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export PATH=/opt/rh/gcc-toolset-12/root/usr/bin:/usr/local/cuda-13.2/bin:/usr/local/cuda/bin:/opt/conda310/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/opt/rh/gcc-toolset-12/root/usr/lib64:/usr/local/cuda-13.2/lib64:/usr/local/cuda/lib64:/opt/conda310/lib:/usr/lib64:/usr/local/lib64:/usr/lib:/lib64
```

`cuda13` 分支使用 Triton 3.6；除非目标镜像明确要求，不要从旧 CUDA 12 测试复制 `TRITON_PTXAS_PATH`。

DSV4 的常用运行时开关（都能在仓库的 server args 或 DSV4 attention 代码中找到）如下。它们是实验配置，不要在不同批次之间悄悄改变：

| 环境变量 | 建议值 | 用途 |
|---|---:|---|
| `WORLD_SIZE` | `8` | 8 个 rank 的进程拓扑 |
| `CP_ROTATE_METHOD` | `ALL_GATHER` | CP token 旋转方式 |
| `PREFILL_CP_KV_CACHE_SHARDED` | `1` | CP prefill KV 分片 |
| `FP8_KV_CACHE` | `1` | FP8 KV cache |
| `REUSE_CACHE` | `1` | 允许请求复用前缀 KV |
| `ENABLE_DEVICE_CACHE` | `1` | 启用设备侧 cache |
| `DSV4_USE_MEGA_MOE` | `1` | DSV4 MegaMoE 路径 |
| `DSV4_CHUNK_TOKENS` | `8192` | DSV4 prefill 分块大小 |
| `DSV4_PREFILL_CP_OVERLAP` | `0/1` | 是否启用 CP prefill overlap；基线必须固定 |
| `WARM_UP` | `1` | 开启服务 warm-up |

`MAX_SEQ_LEN`、`MAX_BATCH_TOKENS_SIZE`、`MAX_CONTEXT_BATCH_SIZE`、`FP8_KV_CACHE` 等也有对应 server args。为了避免环境变量和 CLI 产生两套口径，长度、拓扑、batch、block size 优先写在 target 的 `args` 中；缓存和 JIT 开关再通过 `env` 传入。

## 4. 模型和缓存目录

模型和 tokenizer 必须在**运行测试的容器内**可读。推荐把两者设成同一个本地模型目录：

```bash
export MODEL_DIR=/data1/serina.wzq/DeepSeek-V4-Pro
test -r "$MODEL_DIR/config.json"
test -r "$MODEL_DIR/tokenizer_config.json"
find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' | sort | head
```

如果模型目录在宿主机而容器内不存在，先做只读 bind mount 或使用容器已有的 NAS 路径；不要让 `start_server` 在测试过程中临时下载 1M 模型权重。使用 Hub repo id 只有 perf 入口会尝试 ModelScope/HuggingFace 解析，普通 `start_server` 不会自动下载。

JIT/cache 建议放在持久化工作目录：

```bash
export HIPPO_APP_WORKDIR=/ssd/1/dsv4_perf_work
mkdir -p "$HIPPO_APP_WORKDIR"/{jit_cache,triton_cache,tilelang_cache,results,logs}
export DG_JIT_CACHE_DIR=$HIPPO_APP_WORKDIR/jit_cache
export TRITON_CACHE_DIR=$HIPPO_APP_WORKDIR/triton_cache
export TILELANG_CACHE_DIR=$HIPPO_APP_WORKDIR/tilelang_cache
```

远端 JIT/OSS 凭证只能由 secret manager 或当前 shell 注入，不能写进 BUILD、脚本、日志或提交记录。交接文件不包含任何 access key。

如果模型还没有落盘，先由存储或模型平台管理员把权重同步到 `MODEL_DIR`，并在容器内做文件可读性检查。不要在 Bazel 测试过程中执行下载；权重下载失败和服务启动失败要分开处理。

## 5. 推荐的 DSV4-Pro prefill 参数

这些参数是服务参数，不是 `batch_decode_test.py` 自己实现的逻辑；perf 入口会把未消费的参数转发给 `start_server`。

```text
--model_type deepseek_v4
--checkpoint_path $MODEL_DIR
--tokenizer_path $MODEL_DIR
--batch_size 1
--input_len 256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,786432,1048575
--partial 2
--decode_test_length 1
--max_seq_len 1048576
--max_batch_tokens_size 1048576
--max_context_batch_size 1
--concurrency_limit 1
--tp_size 8
--dp_size 1
--ep_size 8
--world_size 8
--cp_rotate_method ALL_GATHER
--seq_size_per_block 512
--kernel_seq_size_per_block 128
--fp8_kv_cache 1
--reuse_cache 1
--enable_device_cache 1
--use_deepep_moe 1
--use_deepep_low_latency 0
--act_type BF16
--load_method fastsafetensors
--enable_cuda_graph 0
--reserver_runtime_mem_mb 81920
```

`decode_test_length` 虽然在 prefill 请求中仍只发 1 个 token，但它会参与服务最大长度计算：

```text
effective_max_seq_len = max(max_seq_len, max(input_len) + decode_test_length)
```

因此 1M 边界使用 `1,048,575 + 1 = 1,048,576`。如果把 `decode_test_length` 写成 2，服务实际会申请到 1,048,577，不应再把它称为严格 1M 配置。

同一个参数不要同时在 CLI 和环境变量里写两份。建议模型、并行度、block size 等放 CLI；缓存开关和 JIT 开关放环境变量。`GridRunner` 在 grid 模式会用 `max(batch_size)` 设置服务并发，`--concurrency_limit` 主要对 distribution 模式生效。

本手册的基线是 batch=1，所以 `--concurrency_limit=1` 是为了保持“一个 geometry 一次 forward”的可比性。如果要测并发吞吐，另建 target，显式记录 `batch_size`、`max_context_batch_size` 和实际 server concurrency；不要把两类结果合并拟合。

### 修复后需要确认的测试参数

| 类别 | 参数 | 要求 |
|---|---|---|
| 保持 | `--partial=2` | cache-grid 仍是 prefill-only；代码据此进入修复后的分支 |
| 保持 | `--decode_test_length=1` | 仍只生成首 token；它不等于 decode 性能测试 |
| 保持 | `--batch_size=1`、`--max_context_batch_size=1`、`--concurrency_limit=1` | 与 scheduler 更新的 `batch_size=1` 一致 |
| 保持 | `--cache_measure_runs=3` | 正式数据每个 geometry 三次，报告使用成功样本的 median TTFT |
| 必改 | `--result_dir=<全新目录>` | 首次修复后运行不得续接旧的 decode-mode 结果 |
| 必改 | `--cache-alignment=4096` | 生成 grid 时使用；当前 CP=8 分片配置下为 `512 × 8` |
| 必改 | `--expected_cache_block_size=4096` | runner 的去重和启动探测使用同一 CP-visible 粒度 |
| 保持 | `--cache_commit_tail_tokens=4096` | cache case 至少留下一个 4096-token 新计算尾部 |
| 验证时临时开启 | `--cache_profile_runs=1`、`--cache_profile_case_ids ...` | 只用于小规模验真；正式全量运行恢复为 0 |
| 正式关闭 | `--cache_profile_runs=0`、`PERF_PROFILE_RUNS=0` | profiler 时延不得进入正式基线 |

这里必须区分三个概念：DSV4 单 rank 物理 block 是 512，kernel block 是 128；启用
`PREFILL_CP_KV_CACHE_SHARDED=1` 且 CP=8 后，cache-grid 的 CP-visible reuse 粒度是 4096。
当前受版本控制的 `dsv4_pro_prefill.json` 仍把 `cache_alignment` 和
`expected_block_size` 写成 512；在该配置被同步修正前，生成 grid 和运行 runner 时必须用
CLI 显式传入 4096，不能只依赖 profile 默认值。

## 6. 可复现的 Bazel 命令

先做无 GPU 的 Python 检查：

```bash
cd /data0/luoli.hn/work/rtp_llm_4/dsv4-cache-affinity-1OIKUM
python3 -m py_compile \
  rtp_llm/test/perf_test/batch_decode_test.py \
  rtp_llm/test/perf_test/batch_perf_impl.py \
  rtp_llm/test/perf_test/grid_runner.py
bazelisk test //rtp_llm/test/perf_test:batch_decode_test_test \
  --config=cuda13 --config=sm10x \
  --test_output=errors
```

正式运行时，建议从 BUILD 中复制 `v4_pro_cp8_ep8_prefill_64k_perf` 成一个新的手工 target，并把第 5 节的参数写成该 target 的唯一 `args`。这样不会遇到重复参数被旧值抢先解析的问题。当前仓库已有的 target 使用的是 6 个示例长度、`decode_test_length=2`、`seq_size_per_block=256`，适合 profile smoke，不是严格 1M handoff 基线。

运行新 target 的命令形状如下：

```bash
export BAZELISK_HOME=/ssd/1/dsv4_perf_work/bazelisk-home
export RESULT_DIR=/ssd/1/dsv4_perf_work/results/prefill_$(date +%Y%m%d_%H%M%S)

bazelisk test //rtp_llm/test/perf_test:dsv4_pro_prefill_handoff \
  --config=cuda13 --config=sm10x \
  --test_timeout=345600 \
  --test_output=streamed \
  --nocache_test_results \
  --test_arg=--result_dir="$RESULT_DIR" \
  --test_env=PERF_GRID_WARMUP_RUNS=1 \
  --test_env=PERF_FORMAL_WARMUP_RUNS=1 \
  --test_env=PERF_MEASURE_RUNS=3 \
  --test_env=PERF_PROFILE_RUNS=0 \
  --test_env=TOKENIZERS_PARALLELISM=false
```

注意：`--test_arg=--checkpoint_path=...` 不要直接追加到已有 target。`batch_decode_test.py` 会把 checkpoint/tokenizer 留在转发参数中，路径提取函数按第一次出现的值解析，重复写法可能仍然使用 BUILD 里的旧路径。需要换模型时，修改新 target 的唯一 `--checkpoint_path` 和 `--tokenizer_path`，再重新 build/test。

启动前把最终 `args` 保存到结果目录旁的 `argv.txt`，并检查以下值只出现一次：`checkpoint_path`、`tokenizer_path`、`max_seq_len`、`max_batch_tokens_size`、`decode_test_length`、`tp_size`、`dp_size`、`ep_size`、`world_size`。这一步能避免“日志里写的是 1M，服务实际申请了 1,048,577”之类的隐性偏差。

## 7. 测量次数和结果文件

`GridRunner` 的每个 case 大致经过以下阶段：

1. 预热：处理 JIT 或首次分配的抖动；
2. 正式测量：由 `PERF_MEASURE_RUNS` 控制，建议 3 次；
3. profiler：默认存在，但会影响 RT。做性能拟合时用 `PERF_PROFILE_RUNS=0` 关闭。

结果目录通常包含：

```text
$RESULT_DIR/
├── Prefill_Result.json       # 逐 case 指标
├── test_info.json             # 本次配置
└── timelines/                 # 只有开启 profiler 时才有

$TEST_UNDECLARED_OUTPUTS_DIR/main_logs/process.log  # 引擎日志
```

标准 grid JSON 的核心字段是 `input_len`、`batch_size`、`success_rate`、`avg_wait_time`、`avg_prefill_time`。cache matrix 应额外保存 `cache_len_requested`、`cache_len_observed`、`runs[]`、`success_runs` 和 `status`。

cache 结果必须以 `cache_len_observed` 为准。请求的 cache 长度可能因物理 block 对齐而被向下取整；请求了正 cache 但实际 `reuse_len=0` 时，不得把它标成 cache hit。

建议每个结果 JSON 顶层同时记录：`model_id`、代码 commit、完整 argv、关键 env 的脱敏 hash、GPU 型号、`max_seq_len`、`max_batch_tokens_size`、测量轮数、grid 文件 SHA。没有这些 provenance，结果只能用于临时排查，不能拿去拟合或横向比较。

### 专用 cache runner 的实际位置

这一点不能靠默认入口推断：标准 `GridRunner` 没有 cache 维度。历史测试曾把临时 runner 放在：

```text
/tmp/cache_grid_runner.remote.py
```

它提供 `PrefixPromptFactory` 和 `CacheGridRunner`，流程是“写入唯一 prefix → 发起带相同 prefix 的 continuation → 读取 `aux_info.reuse_len` → 保存三轮 RT”。配套的临时入口是：

```text
/tmp/batch_decode_test.remote.py
```

其中 `--cache_grid_json` 指向显式的 `input_len × cache_len` case 文件，`--cache_measure_runs=3` 控制每个 geometry 的测量次数，结果文件为：

```text
$RESULT_DIR/cache_grid_results.json
```

需要特别说明：这两个 `/tmp` 文件不是当前 Git 分支里的受版本控制文件，机器重启或清理临时目录后可能消失，因此只能解释历史结果。后续必须使用仓库内受版本控制的
`rtp_llm/test/perf_test/cache_grid_runner.py` 和 `batch_decode_test.py`；不要再复制或运行历史
`/tmp` 入口，否则不会得到本次 scheduler mode 修复。

### cache-grid 异常点的可选 profiler 补采

版本控制内的 `rtp_llm/test/perf_test/cache_grid_runner.py` 支持指定 case 的
Torch/Kineto CPU + CUDA trace。默认 `--cache_profile_runs=0`，不调用 profiler，
也不增加诊断请求。该参数独立于普通 grid 的 `--profile_runs` / `PERF_PROFILE_RUNS`。

在原测试命令上追加以下参数，可针对报告中的 case ID 补采：

```bash
# 保留原命令中的模型、并行、cache 对齐、transport 和 engine_env 配置。
# --cache_grid_json 仍传完整原始 grid，不需要手工裁剪或重新编号。
<原 cache-grid 测试命令> \
  --cache_profile_only \
  --cache_profile_case_ids 14579 14595 \
  --cache_profile_runs 3 \
  --cache_profile_trace_timeout 180
```

- `--cache_profile_case_ids`：显式选择原 grid / 报告的 case ID，必须与正数
  `--cache_profile_runs` 一起使用；未知或已被对齐去重移除的 ID 会在启动模型前报错。
- `--cache_profile_only`：仅补采，跳过正式测量。实际输出到
  `$RESULT_DIR/cache_profile_replays/<唯一 session>/`，保留原报告、test_info 和 checkpoint。
  可以对已经完成全量测试的目录使用。不要同时传 `--require_cache_resume`。
- 不传 `--cache_profile_only`：先完成正常测量/恢复，再补采所选 case；profile
  结果不进入正式 `runs[]`、`median_ttft_ms` 或吞吐计算。
- 当前支持 DP=1，并通过 `/start_profile` 的 `enable_all_rank=true` 采集全部 TP rank。
  每个诊断请求使用 `start_step=0, num_steps=1`，捕获一个实际 forward。

每次诊断使用新的 prompt case 标识，按相同 input/cache geometry 重新生成输入，
先执行未开启 profiler 的 seed，再开启 profiler 并发送 continuation；冷请求不发 seed。
这会更换前缀内容，避免复用正式请求或上次诊断已经写入的后缀缓存，属于同形状补采，
不是原 token 序列的逐字重放。诊断仍检查引擎反馈的 `input_len`、`reuse_len`、`output_len`。
当前补采不增加单独的形状预热，若要分析稳定态，应先按原流程预热并比较多次 trace。

补采产物（相对于实际输出目录）：

```text
cache_profiles/<session>/manifest.json  # case / request ID、输入生成标识、seed、诊断响应、trace 路径
timelines/cache_profile_*_wr0_*.json   # rank 0 的 Kineto trace
timelines/cache_profile_*_wr1_*.json   # 其他 TP rank，以此类推
```

`manifest.json` 中的 `trace_files` 相对于实际输出目录。runner 等待每个 TP rank
写出可解析且含事件的 JSON 后才记录成功；接口失败、命中不符或 trace 超时都会保存失败
记录并终止补采，不会把已经完成的正式测试标为失败。trace 超时可用
`--cache_profile_trace_timeout` 调整，已有文件也会保留。

trace 可以在 Perfetto 等支持 Chrome trace JSON 的工具中查看 CPU/CUDA 时间线。
优先核对 `executor.model_forward(ctx_batch=...,gen_batch=...,tokens=...,max_seq=...)`
确认实际执行模式，再比较每层、kernel、内存拷贝和跨卡等待。修复后的 cache-grid 入口会在
block 探测、seed、正式测量和 profiler 补采之前统一切到 prefill mode；旧提交不会自动切换。
现有采样窗口从模型输入 TP 同步之后开始，不能据此宣称
完整覆盖调度、cache 加载和输入构造。DSV4 prefill fast path 会省略部分细分逻辑标记，
保留层级范围；GPU kernel 时间线仍由 profiler 采集。诊断时延包含 profiler 开销，
不能替代关闭 profiler 的正式性能基线。

### 修复后的最小验真

不要直接启动 1.5 万 case 全量网格。先用一个全新的结果目录和小 grid 验证以下三类 geometry：

| case | `input_len` | requested/observed cache | 期望新计算 token |
|---|---:|---:|---:|
| 短 cold | 4,096 | 0 | 4,096 |
| 长 cold | 65,536 | 0 | 65,536 |
| 长 hit | 65,536 | 61,440 | 4,096 |

小 grid 仍使用 `--partial=2 --decode_test_length=1 --cache_measure_runs=3`，并显式设置
`--cache_commit_tail_tokens=4096 --expected_cache_block_size=4096`。选中这三个 case，临时追加：

```bash
--cache_profile_runs 1 \
--cache_profile_case_ids <short-cold-id> <long-cold-id> <long-hit-id> \
--cache_profile_trace_timeout 180
```

验收必须同时满足：

1. runner 日志在任何 perf 请求前出现
   `scheduler configured before perf requests: payload={'batch_size': 1, 'mode': 'prefill'}`，
   引擎日志出现 scheduler mode 更新为 prefill；
2. 输入所属 rank 的 GPU trace 为 `ctx_batch=1, gen_batch=0`，`tokens` 等于
   `input_len - cache_len_observed`。CP=8 时局部 embedding token 数可分别表现为 512、8192、
   512；其他 rank 在广播前出现零 token 标记不算失败；
3. hit case 三次 `cache_len_observed` 都是 61,440，cold case 都是 0，且每次
   `output_len=1`、请求成功；
4. 在干净 cache 或独立服务上，对同一完整 input IDs 比较 cold 与 hit 的首个生成 token
   （有 logits 时同时比较 logits），结果一致；只比较 `reuse_len` 不足以证明计算正确；
5. profiler 关闭后再取正式 TTFT；不能用“输入越长越慢”或“命中越多越快”代替上述验真。

任一项不满足时停止全量测试。验真通过后换另一个全新 `result_dir` 执行正式网格，并将
`--cache_profile_runs` 和 `PERF_PROFILE_RUNS` 设为 0。

## 8. 运行时监控和停机

启动后至少每 30 秒看一次：

```bash
tail -F "$TEST_UNDECLARED_OUTPUTS_DIR/main_logs/process.log"
nvidia-smi
```

重点看：

- `All backend ranks started` / health ready；
- 8 个 rank 是否都存在；
- GPU 显存、util 是否在变化；
- `OOM`, `Traceback`, `FATAL`, `PROCESS_EXIT`, `GET_HOST_FAILED`；
- 结果 JSON 的 case 数是否增长。

不要因为某一个 case 很慢就重启服务。服务应覆盖整个 grid，case 之间只切换 scheduler 配置。只有明确确认属于自己的测试进程、并且服务已经无法自行退出时，才按进程树做有范围的清理；不要使用无目标的 `kill -9`。

## 9. 拟合和画图

### 9.1 输入校验和拟合

拟合脚本只会使用通过校验的 geometry：三次 measurement 都成功、output length 为 1、prefill RT 有限、observed reuse 在三次中一致。它会把同一物理 geometry 的重复记录合并成 median。

```bash
cd /data0/luoli.hn/work/rtp_llm_4/dsv4-cache-affinity-1OIKUM
FIT=rtp_llm/test/perf_test/deepseek_v4_prefill_formula_fit.py
DATA=/path/to/cache_grid_results.json
OUT=/path/to/formula_output

python3 "$FIT" validate-inputs \
  --inputs "$DATA" \
  --batch-size 1 \
  --report "$OUT/input_validation.json"

python3 "$FIT" fit \
  --inputs "$DATA" \
  --batch-size 1 \
  --objective mae \
  --output-dir "$OUT"
```

输出：

```text
$OUT/input_audit.json
$OUT/fit_report.json
$OUT/deepseek_v4_prefill_formula.txt
$OUT/predictions.csv
```

`--objective mae` 是绝对误差目标。脚本返回码为 0 表示生产门禁通过，3 表示公式已经生成但 MAPE/p95/max 门禁没有通过，2 表示输入无效或样本不足。返回 3 不能当作命令失败，也不能把未通过的公式直接上线。

当前 fitter 导出的公式只使用 `tokens`、`hitCacheTokens` 和 `+ - * / ( )`。公式适用于固定 batch 和实际测量范围，不自动外推 batch scaling。

### 9.2 三维图

```bash
CHART=rtp_llm/test/perf_test/generate_prefill_3d_chart.py
python3 "$CHART" \
  --input "$DATA" \
  --output "$OUT/deepseek_v4_prefill_3d.svg" \
  --batch-size 1
```

当前图的坐标约定固定为：

```text
X = uncached compute tokens = input_len - observed_cache_len
Y = observed cached tokens
Z = measured TTFT / prefill RT (ms)
```

浅灰点是全部可用 geometry；实线是近固定 cache 的中位数趋势，虚线是近固定 compute 的中位数趋势。颜色只区分趋势线，不表示 RT 数值。

## 10. 出问题时先查这几项

| 现象 | 先查什么 |
|---|---|
| 服务启动后立刻退出 | `process.log`、模型目录、`model_type`、CUDA/GCC 版本 |
| 1M case 被拒绝 | `max(input_len)+decode_test_length` 是否超过 `max_seq_len` |
| cache 命中为 0 | seed 是否先成功、prefix 是否完全相同、block 对齐是否改变 observed reuse |
| 结果行数少 | 是否有失败请求、重复物理 geometry、正 cache 实际 reuse=0 |
| 每个 case 很慢 | 是否误开 profiler、是否每个 case 都重启 server、JIT cache 是否持久化 |
| 公式在 FlexLB 解析失败 | 是否混入 `sum`、`max`、`batchSize`、`computeTokens` 或 Python 语法 |

交接时至少提供：代码 commit、模型目录、完整 argv/env、GPU 型号、结果 JSON、`process.log`、拟合报告和 SHA256。不要只交一张截图或一行平均 RT。

## 11. 生成 128-token 对齐的密集 cache grid

不要把 1M 范围内所有 `input_len × cache_len` 的 128-token 组合全部展开；
完整三角网格约有 3355 万个 geometry。仓库提供分层采样生成器，使候选点按
128 token 对齐，同时限制实际 case 数：

```bash
bazelisk run //rtp_llm/test/perf_test:generate_cache_grid -- \
  --min-input-len 256 \
  --max-input-len 1048575 \
  --alignment 128 \
  --cache-alignment 4096 \
  --input-points 1024 \
  --cache-points-per-input 16 \
  --cache-ratio-points 7 \
  --seed 104729 \
  --max-cases 20000 \
  --output /path/to/cache_grid_128.json
```

默认使用固定质数 `104729` 作为随机种子。同一组参数会生成相同的计划；每个
input 同时包含 cold、near-full、按 cache ratio 分层以及按 compute tokens
分层的点。普通点均按 128 对齐，严格 1M 的 `input_len=1048575` 是唯一保留的
非对齐边界例外。

`--cache-alignment` 是 cache 维度的对齐，应等于 cache-grid 可观察的复用粒度：
`seq_size_per_block`（DSV4 未显式传入时默认 256），开启
`PREFILL_CP_KV_CACHE_SHARDED=1` 时再乘以 CP size。它与 `--alignment`
（input 维度）解耦：handoff 基线 `--seq_size_per_block 512` 对应
`--cache-alignment 4096`。cache 长度按 CP-visible block 对齐后，每个点的 requested
与 observed reuse 一致；若 cache 对齐小于物理 block（例如 128 对齐 × 512
block），同桶点测的是同一几何，白付 seed 和测量轮次。`--cache-alignment 0`
（默认）保持旧行为，即跟随 `--alignment`。

如果需要枚举范围内每个 128-token input 候选点，可增加
`--input-mode=stride`。这通常会突破默认 20000 case 保护，生成器会拒绝输出；
只有确认预计请求量后才使用 `--allow-large-grid`。

运行密集计划：

```bash
bazelisk test //rtp_llm/test/perf_test:cache_grid_perf_test \
  --config=cuda13 --config=sm10x \
  --test_timeout=345600 --test_output=streamed --nocache_test_results \
  --test_arg=--cache_grid_json=/path/to/cache_grid_128.json \
  --test_arg=--partial=2 \
  --test_arg=--decode_test_length=1 \
  --test_arg=--cache_measure_runs=3 \
  --test_arg=--cache_commit_tail_tokens=4096 \
  --test_arg=--expected_cache_block_size=4096 \
  --test_arg=--result_dir=/path/to/new_prefill_results
```

`--expected_cache_block_size`（缺省 0 时自动读取计划里的
`generator.cache_alignment`）有两个作用：加载时把落到同一物理 block 桶的
case 去重；服务启动后先做一次探测——写入恰好一个 block 的前缀、发送两
block 的命中请求并校验 `reuse_len` 等于 block size。粒度不一致（比如
`seq_size_per_block` 算错或 CP 分片开关与预期不符）会立即终止测试，避免
整轮 cache-hit 数据全部作废。引擎实际生效的粒度也可以从启动日志确认：
`cache config: ... seq_size_per_block=N`（通用）或
`DSV4 physical block=N, kernel block=M`（DSV4）。

生成计划中的 `generator`、`summary` 和输入文件 SHA256 会写入
`cache_grid_results.json`。采样坐标使用 requested cache；拟合和三维图必须
继续使用引擎报告的 observed cache。三维图坐标为 X=compute tokens、
Y=observed cached tokens、Z=TTFT。

### 一条命令完成测试、拟合和绘图

`run_cache_grid_pipeline` 会先运行 cache-grid 测试，并且只在
`cache_grid_results.json` 标记为完整后依次生成拟合公式、静态 SVG 和可旋转的
HTML 三维图。cache 请求默认使用 `dashsc_input_ids`，直接通过 Dash-SC gRPC
发送已经校验的 INT32 token IDs；需要兼容旧链路时可显式传
`--cache_request_transport=http_prompt`。`--` 后的参数原样转发给 cache runner：

```bash
bazelisk run //rtp_llm/test/perf_test:run_cache_grid_pipeline \
  --config=cuda13 --config=sm10x -- \
  --cache-grid-json=/path/to/cache_grid_128.json \
  --result-dir=/path/to/results \
  --profile=rtp_llm/test/perf_test/profiles/dsv4_pro_prefill.json \
  --estimator=median \
  -- \
  --cache_measure_runs=3 \
  --cache_commit_tail_tokens=4096 \
  --expected_cache_block_size=4096
```

默认产物为：

```text
results/cache_grid_results.json
results/formula/deepseek_v4_prefill_formula.txt
results/formula/fit_report.json
results/formula/fit_gap.svg
results/prefill_3d.svg
results/prefill_cold_miss.svg
results/prefill_3d.interactive.html
results/pipeline_summary.json
```

模型和并行参数可以由 `--profile` 提供，也可以放在第二个 `--` 后传给 runner。
已有完整测试结果需要重新拟合或绘图时，加 `--skip-test`。拟合质量门禁不通过时，
脚本仍会生成 SVG 和 HTML，但最终返回拟合脚本的非零状态，并在
`pipeline_summary.json` 中记录 `fit_rejected`。

## 7. Profile 参数化

所有工具（grid 生成、runner、拟合、图表）都支持 `--profile` 参数，指向一个
版本化的 JSON 文件。DSV4-Pro 的标准 profile 在
`rtp_llm/test/perf_test/profiles/dsv4_pro_prefill.json`。

### 优先级链

每个参数按以下顺序解析：

1. **CLI 显式值**（包括 `0`）
2. **Profile 字段**（`--profile` 指定的 JSON 文件）
3. **输入中嵌入的 profile**（结果 JSON 的 `profile` 键，不传 `--profile` 时自动继承）
4. **Legacy 默认值**（之前的硬编码值）

不传 `--profile` 时，所有输出文件与 profile 化之前的版本字节一致——不会多出
额外的键，也不会改变文件名。

### 使用示例

```bash
PROFILE=rtp_llm/test/perf_test/profiles/dsv4_pro_prefill.json

# 当前 profile 的 cache_alignment 仍是 512，本基线必须用 CLI 覆盖为 4096
python3 generate_cache_grid.py --profile $PROFILE \
  --min-input-len 256 --max-input-len 1048575 --input-points 489 \
  --cache-alignment 4096

# 运行测试（profile 注入引擎参数 + cache grid 参数）
bazelisk test //rtp_llm/test/perf_test:cache_grid_perf_test \
  --test_arg=--profile=$PROFILE \
  --test_arg=--cache_grid_json=grid.json \
  --test_arg=--partial=2 \
  --test_arg=--decode_test_length=1 \
  --test_arg=--expected_cache_block_size=4096

# 拟合公式（profile 提供 token_unit、model_label 等）
python3 deepseek_v4_prefill_formula_fit.py fit \
  --inputs cache_grid_results.json --output-dir formula/ \
  --profile $PROFILE --estimator min

# 异常分析
python3 deepseek_v4_prefill_formula_fit.py analyze-anomalies \
  --inputs cache_grid_results.json --output-dir anomalies/ \
  --profile $PROFILE --estimator min
```

### 断点续测与 Resume 守卫

scheduler mode 修复后的第一次测试必须使用全新的 `--result_dir`，且不要传
`--require_cache_resume`。不得把旧 decode-mode 目录中的
`cache_grid_results.json`、journal 或 checkpoint 复制到新目录。只有已经通过本节验真的
修复后新任务，才可以在代码 commit、grid、profile、模型、引擎参数和运行指纹都一致时续测。

Cache-grid 每完成一个 case 都会同步追加
`cache_grid_results.journal.jsonl`，并原子更新轻量的
`cache_grid_progress.json`；每 100 个 case（可用
`--cache_checkpoint_every` 调整）及正常退出时，将 journal 合并进完整的
`cache_grid_results.json`。重新执行原命令并使用**同一个**
`--result_dir` 时，会自动跳过所有 `status=ok` 的 case；中断时尚未完成或
失败的 case 会从该点重新执行。推荐在续测命令额外加
`--require_cache_resume`，这样结果目录写错时会直接报错，而不会意外开始一轮
新测试：

```bash
# 其余模型、profile、grid 和引擎参数必须与首次运行保持一致
python3 -m rtp_llm.test.perf_test.batch_decode_test \
  ...首次运行的全部参数... \
  --result_dir "$RESULT_DIR" \
  --require_cache_resume
```

checkpoint 顶层的 `progress` 会记录 `completed_cases`、`pending_cases`、
`progress_pct`，以及 `last_completed_case` / `next_pending_case` 的
`input_len` 和 `cache_len`。`test_info.json` 会记录完整 argv、脱敏后的相关环境
变量值、首次启动时间、最近更新时间和 `attempt_count`。

恢复校验在 tokenizer 和模型加载**之前**执行，覆盖 `grid_sha256`、
`profile_sha256`、模型/权重/引擎配置指纹、`measure_runs`、transport、
`cache_commit_tail_tokens` 和 `expected_block_size`。配置不一致时默认中止。
`--allow_resume_mismatch` 仅用于人工确认过的特殊场景，因为它会复用不同配置
产生的成功结果。

### 限制

环境变量（如 `WORLD_SIZE`、`DSV4_USE_MEGA_MOE`）**不会**被 profile 捕获。
它们必须通过 BUILD target 的 env 块或命令行显式设置。Profile 只记录
CLI 可见的引擎参数。
