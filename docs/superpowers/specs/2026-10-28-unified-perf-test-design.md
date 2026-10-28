# Unified Perf Test 设计

日期：2026-10-28
状态：已确认，待实现计划
范围：统一现有 perf test 与 cache perf test 的入口、配置、编排、结果和 CI 集成

## 1. 背景

当前仓库同时存在两套性能测试体系：

- 原 perf test 主要覆盖 decode，包括固定网格、分布 workload 和 TPOT 目标下的 batch 自动搜索。
- cache perf test 主要覆盖可控 KV cache reuse 的 prefill 网格，并提供 resume、profile、report、local/AOne 调度等能力。

两套体系在入口、配置、进程生命周期、结果格式、CI 封装和发布逻辑上互相重叠，但能力边界并不相同。继续在旧实现上增加条件分支会扩大重复代码，并使 prefill/decode 的语义更难约束。

本次重构建立一套新的统一 perf test。它统一控制面，但不强行统一 prefill 和 decode 的 workload 算法：prefill 只保留 cache perf 的能力，decode 只保留原 perf test 的能力。

## 2. 已确认决策

### 2.1 产品边界

- 新的唯一用户入口为 `./tools/perf_test`。
- 旧入口直接删除，不提供兼容脚本、参数别名或结果转换层。
- 使用单个配置文件描述模型、引擎、workload、执行环境和输出。
- `--mode=prefill|decode` 显式区分测试模式。
- prefill 只保留 cache perf 的可控 reuse 网格测试，不保留原来的完整请求普通 prefill 测试。
- decode 只保留原 perf test 的固定网格、分布测试和 TPOT batch 搜索。
- 本阶段不增加 decode 的 input length x output length 二维网格；decode 的 `output_length` 为单值。
- AOne CI 本阶段只支持 prefill；统一入口仍支持本地 decode。
- prefill 保留 `resume`、`profile`、`report`。
- 不兼容旧配置、旧结果目录、旧 checkpoint/resume 数据或旧 baseline 格式。
- 新实现重建契约、编排和结果层，但复用经过验证的 workload 核心算法、指标公式和基础设施能力。

### 2.2 能力矩阵

| Mode | Backend | run | resume | profile | report |
| --- | --- | --- | --- | --- | --- |
| prefill | local | 支持 | 支持 | 支持 | 支持 |
| prefill | aone | 支持 | 不支持 | 不支持 | 收集后离线支持 |
| decode | local | 支持 | 不支持 | 不支持 | run 内生成基础报告 |
| decode | aone | 不支持 | 不支持 | 不支持 | 不支持 |

所有不支持的组合必须在创建结果目录、构建 Bazel target、启动模型或提交任务之前失败，并返回能力校验错误。

## 3. 目标与非目标

### 3.1 目标

1. 用户只学习一个入口、一套配置结构和一套结果契约。
2. local 与 AOne 使用同一个解析、校验和计划生成过程。
3. prefill 与 decode 共享控制面和执行基础设施，同时保持各自 workload 语义独立。
4. 每次运行都能由归一化配置、执行计划和 fingerprint 重现与审计。
5. 中断、部分失败、远程分片和离线报告具有明确、可验证的状态语义。
6. 保留现有 CI 依赖的 baseline regression、golden、OSS 归档和 ODPS 汇总能力。
7. 最终切换时删除旧入口和旧编排代码，避免双轨维护。

### 3.2 非目标

- 不兼容任何旧 CLI 或旧配置字段。
- 不读取旧 perf/cache perf 结果作为新的 resume 输入。
- 不支持 AOne decode。
- 不实现 decode input x output 二维网格。
- 不将 prefill 和 decode 的 case 指标强行压成相同结构。
- 不在配置中实现 shell 展开、环境变量插值、include 或 secret 管理。
- 不在本设计阶段修改现有 OpenSpec 或实现代码。

## 4. 用户接口

### 4.1 命令形态

新入口的基本形式：

```bash
./tools/perf_test \
  --mode=prefill \
  --action=run \
  --config=path/to/perf.json
```

decode 本地运行：

```bash
./tools/perf_test \
  --mode=decode \
  --action=run \
  --config=path/to/perf.json
```

基于既有 prefill 运行目录执行后续动作：

```bash
./tools/perf_test --mode=prefill --action=resume --run-dir=path/to/run
./tools/perf_test --mode=prefill --action=profile --run-dir=path/to/run --cases=10,20
./tools/perf_test --mode=prefill --action=report --run-dir=path/to/run
```

`run` 接受 `--config`，`resume/profile/report` 接受 `--run-dir`。不允许同时传入两个来源，也不允许通过命令行覆盖配置中的模型、workload 或执行参数。

### 4.2 Dry run

`--dry-run` 执行以下步骤：

1. 解析并严格校验配置。
2. 归一化配置并计算 fingerprint。
3. 生成 case、资源计划和 AOne 分片预览。
4. 输出即将执行的命令、Bazel targets 和资源摘要。

Dry run 不创建正式运行目录、不构建 target、不启动模型、不提交远程任务，也不发布结果。

## 5. 总体架构

采用“统一控制面 + mode adapter + execution backend”结构：

```text
tools/perf_test
      |
      v
ConfigResolver -> CapabilityValidator -> ExecutionPlan
                                             |
                       +---------------------+--------------------+
                       |                                          |
                       v                                          v
                LocalExecutor                              AOneExecutor
                       |                                          |
               +-------+-------+                                  |
               |               |                                  |
               v               v                                  v
         PrefillRunner     DecodeRunner                     PrefillRunner
               |               |                                  |
               +---------------+----------------------------------+
                               |
                               v
                   ResultWriter -> Report -> Publishers
```

控制面负责参数、配置、能力、计划、状态和结果契约。mode adapter 负责 workload 生成与执行语义。execution backend 负责模型生命周期、设备和远程调度，不理解具体性能指标。

### 5.1 建议模块布局

```text
rtp_llm/test/perf_test/
  main.py
  config_schema.py
  capabilities.py
  plan.py
  orchestrator.py
  results.py
  execution/
    local.py
    aone.py
    engine_session.py
  modes/
    prefill.py
    decode.py
  prefill/
    grid.py
    planner.py
    runner.py
    checkpoint.py
    profile.py
    report.py
  decode/
    batch.py
    grid_runner.py
    distribution.py
    tpot.py
    report.py
  publish/
    baseline.py
    archive.py
    odps.py
```

目录是所有权边界，不要求第一阶段立即拆到最细文件；但控制面、backend 和 mode adapter 之间的依赖方向必须保持稳定。

### 5.2 ExecutionPlan

配置解析后生成不可变 `ExecutionPlan`，至少包含：

```text
mode
action
backend
resolved_config
config_fingerprint
workload_fingerprint
cases
resource_plan
result_directory
```

runner 只能消费计划，不允许在运行中重新解释源配置或从环境隐式补参数。AOne 分片计划也必须由同一个顶层 `ExecutionPlan` 派生。

## 6. 配置契约

### 6.1 顶层结构

```json
{
  "schema_version": 1,
  "model": {},
  "engine": {},
  "workload": {},
  "execution": {},
  "output": {},
  "regression": {},
  "publish": {}
}
```

除明确声明的扩展 map 外，未知字段一律报错。配置错误必须指出完整字段路径和期望类型。

### 6.2 Model

`model` 描述测试对象和并行拓扑：

```json
{
  "label": "qwen25-7b",
  "type": "qwen_2",
  "checkpoint": "/path/to/model",
  "tokenizer": "/path/to/tokenizer",
  "topology": {
    "tp": 1,
    "dp": 1,
    "pp": 1,
    "ep": 1
  }
}
```

`checkpoint` 和 `tokenizer` 是运行环境必须可访问的路径，不嵌入 AOne 配置包。拓扑字段为正整数，并在计划阶段结合资源数量校验。

### 6.3 Engine

```json
{
  "max_seq_len": 32768,
  "max_batch_tokens": 32768,
  "concurrency_limit": 64,
  "args": {},
  "env": {}
}
```

`args` 和 `env` 是有意保留的扩展 map。它们必须被完整记录到 resolved config 和 manifest；禁止再通过未声明的 CLI 参数隐式透传给模型进程。

### 6.4 Prefill workload

prefill 只有 `cache_grid`：

```json
{
  "kind": "cache_grid",
  "cache_alignment": 64,
  "commit_tail_tokens": 1,
  "max_input_length": 32768,
  "compute_steps": [64, 128, 256],
  "cache_lengths": {
    "strategy": "fixed",
    "values": [0, 1024, 4096]
  },
  "batch_sizes": [1, 2, 4],
  "measure_runs": 3,
  "seed": 1234,
  "min_compute_tokens": 1,
  "max_cases": 1000
}
```

`cache_lengths.strategy` 可为固定值或受约束的随机采样。planner 必须根据 cache alignment、commit tail、最大输入长度、最小 compute token 和 batch 约束生成有效 case，并为每个 case 固化 seed、prefix/reuse 策略及预期 reuse 长度。

prefill 指标所测对象是完整请求中“未命中 cache、需要本轮计算”的部分，而不是恢复原 perf test 的普通完整 prefill 模式。

### 6.5 Decode workload

固定网格：

```json
{
  "kind": "grid",
  "input_lengths": [128, 1024, 4096],
  "output_length": 128,
  "batch_sizes": [1, 2, 4, 8],
  "measure_runs": 3,
  "generate_config": {}
}
```

分布 workload：

```json
{
  "kind": "distribution",
  "distribution_file": "path/to/distribution.json",
  "output_length": 128,
  "batch_sizes": [1, 2, 4],
  "measure_runs": 3,
  "generate_config": {}
}
```

TPOT batch 搜索是在 decode workload 上启用 `target_tpot_ms`。它搜索满足目标 TPOT 的最大可用 batch，用于寻找吞吐与单 token 延迟约束之间的容量边界。搜索轨迹和最终 batch 都必须进入结果，不只保留最终数字。

本阶段 `output_length` 必须是一个正整数。数组或 output range 明确拒绝，以免在尚未裁剪 decode 语义前引入 input x output 二维扩张。

### 6.6 Execution

`execution` 是按 backend 判别的联合类型。

Local 示例：

```json
{
  "backend": "local",
  "devices": [0, 1]
}
```

AOne 示例：

```json
{
  "backend": "aone",
  "resource": "gpu",
  "gpu_type": "H20",
  "max_parallel": 4,
  "shard_count": 4
}
```

decode + aone 在 capability validation 阶段拒绝。

### 6.7 配置归一化和外部文件

解析器完成默认值展开、路径规范化和枚举标准化后，输出 `resolved_config.json`，并对 canonical JSON 计算 SHA-256 `config_fingerprint`。

distribution、显式 grid 等小型输入文件在提交主机读取、校验并内联到 resolved workload。配置不支持 shell、环境变量插值或 include，凭证只能由运行环境注入，不能写入 config 或 manifest。

## 7. 运行目录和可追溯性

每次 `run` 创建独立目录：

```text
<run-dir>/
  source_config.json
  resolved_config.json
  execution_plan.json
  manifest.json
  workload/
  checkpoints/
  results/
    cases/
    result.json
    summary.json
  reports/
  traces/
```

- `source_config.json` 是用户配置的字节级快照。
- `resolved_config.json` 是 runner 的唯一配置来源。
- `execution_plan.json` 固化 case、分片和资源计划。
- `manifest.json` 记录状态、fingerprint、代码版本、模型信息、时间和错误。
- 每个 case 独立原子写入 `results/cases/<case_id>.json`。
- `result.json` 和 `summary.json` 只由已落盘 case 汇总生成。

所有 manifest、checkpoint 和正式结果采用临时文件加原子 rename。一个 run directory 同一时刻只能由一个写操作持有者使用，必须有排他锁；`report` 可在无写任务时读取。

## 8. 执行语义

### 8.1 LocalExecutor

一次完整计划只启动一次模型。`EngineSession` 作为 context manager 负责：

- 组装显式 engine 参数和环境变量。
- 启动服务并等待健康检查。
- 收集 stdout/stderr、timeline 和 coredump 信息。
- 在正常完成、异常或信号中断时关闭服务。
- 将进程级失败转换成统一 run error。

mode runner 不自行启动或销毁模型，只通过 session 提供的客户端执行 case。

### 8.2 PrefillRunner

执行流程：

1. planner 生成有效 cache/compute/batch case。
2. runner 按 case seed 构造共享 prefix 和请求 token。
3. 先建立需要的 cache 状态，再发送测量请求。
4. 校验响应形状、实际 reuse length 和 timing 数据。
5. 记录每次测量和聚合指标。

reuse 校验是正确性条件，不是可忽略的观测项。实际 reuse 与预期不一致时，case 标记 `invalid_reuse`，不得作为有效性能点进入报告。

### 8.3 DecodeRunner

decode adapter 保留三个路径：

- `grid_runner.py`：对 input length x batch size 执行固定网格，output length 为单值。
- `distribution.py`：按照输入长度/请求分布构造 workload。
- `tpot.py`：以目标 TPOT 为约束搜索 batch，并保存完整搜索步骤。

DP > 1 的 workload 分配逻辑属于 decode runner/engine session 协作边界：plan 固化全局请求，runner 按原有算法将请求调度到 DP workers，并在结果中保留全局 batch 和 DP 拓扑，避免把每个 worker batch 误写为总 batch。

## 9. AOne CI

### 9.1 配置传递

AOne 不传递用户机器上的原始配置路径。提交端完成解析和校验后，生成一个临时 Bazel package，包含：

```text
resolved_config.json
resolved_workload.json
shard_plan.json
BUILD.bazel
```

Bazel target 通过 `$(location ...)` 将这些文件作为 runfiles 传给远程 action。远程 runner 只消费 runfiles 内文件，因此不会依赖提交机绝对路径。

模型 checkpoint/tokenizer 不复制进 package，它们必须是 AOne runtime 能访问的存储路径。小型 workload 外部文件在提交端被解析后写入 `resolved_workload.json`。

### 9.2 分片

prefill cases 使用现有经过验证的基于预计成本的 LPT 类分片方法，目标是平衡 shard 总计算量，而不是平均 case 数量。每个 shard 的 action 只启动一次模型，顺序执行所拥有的 cases。

每个远端 shard 返回：

```text
run_id
shard_id
config_fingerprint
workload_fingerprint
owned_case_ids
completed_case_ids
case_results
```

collect 阶段必须拒绝：

- fingerprint 不一致。
- 未知 case 或不属于该 shard 的 case。
- case 重复。
- 声明完成但缺失结果。
- shard 缺失或远程 action 失败。

任何上述问题都不能被静默合并成完整结果。可用 case 可以保留用于诊断，但 run 状态为 `partial` 或 `failed`，退出码非零。

### 9.3 AOne 动作限制

AOne 只执行新的 prefill `run`。`resume` 和 `profile` 不提交 AOne job。收集后的 run directory 可在本地执行离线 `report`。decode + aone 必须在产生任何构建或提交副作用前失败。

## 10. 状态模型

### 10.1 Run 状态

正常主路径：

```text
created -> planned -> running -> collecting -> reporting -> completed
```

`collecting` 只用于远程运行。任意阶段还可进入：

- `partial`：至少一个有效 case，但计划未完整成功。
- `failed`：没有可接受结果或发生 run 级失败。
- `interrupted`：收到中断信号，已完成 case 已原子保存。

report 失败不会破坏已测量 case，但 run 不能标记 `completed`。

### 10.2 Case 状态

```text
planned
running
ok
request_failed
invalid_shape
invalid_reuse
invalid_timing
```

case failure 不应立即丢弃其他独立 case。执行结束后根据完整性决定 `completed`、`partial` 或 `failed`。

## 11. Resume、Profile 和 Report

### 11.1 Resume

resume 仅支持 local prefill：

1. 从 run directory 读取 resolved config、plan、manifest 和 case snapshots。
2. 校验 config、workload、model 和 runner fingerprint。
3. 保留 `ok` case。
4. 重新执行未开始、中断和失败 case。
5. 不允许命令行覆盖原配置。

fingerprint 不匹配直接失败，不能通过“尽力恢复”混合两次不同测试定义。

### 11.2 Profile

profile 仅支持 local prefill。它从既有 run directory 选取 case，复用原始 seed 和请求构造规则，重新验证精确 reuse 后采集 profile。

输出写入：

```text
profiles/<profile_id>/
```

profile 不修改正式 case 结果、checkpoint 或 summary。case 选择可用 `--cases=10,20`；未指定时由配置中的 profile 策略选择。

### 11.3 Report

report 是离线操作，不启动模型。成功 run 结束后自动执行一次，也可从 run directory 单独重跑。

- 完整结果生成正式报告、公式/图表、baseline 比较和 candidate。
- partial 结果只生成带明显标记的诊断报告。
- 只有 shape、reuse、timing 校验均有效的 case 才能进入性能拟合和聚合。
- 报告输出不能回写或改写原始 case 数据。

## 12. 结果契约

### 12.1 公共 envelope

`result.json` 使用版本化公共 envelope：

```json
{
  "schema_version": 1,
  "run_id": "...",
  "mode": "prefill",
  "backend": "local",
  "status": "completed",
  "fingerprints": {},
  "model": {},
  "summary": {},
  "metrics": {}
}
```

公共层只定义身份、状态、fingerprint、模型和摘要。prefill/decode 的详细指标位于各自的 `metrics` schema 中。case 级只强制共享 `case_id`、`status`、`runs` 和 `errors`，不制造含义模糊的公共字段。

### 12.2 Prefill metrics

至少记录：

- batch size 和请求分组。
- input tokens。
- expected/observed reuse tokens。
- compute tokens。
- 每次运行的 server prefill latency 和 TTFT。
- shape/reuse/timing validation。
- input TPM 和 compute TPM。
- seed、prefix policy 和测量次数。

### 12.3 Decode metrics

至少记录：

- workload kind。
- input length、output length、batch size 和 DP 拓扑。
- 成功请求数和失败请求数。
- wait、prefill、decode、TPOT、total latency。
- token throughput。
- TPOT 搜索步骤和最佳 batch。
- distribution 测试中的实际 sequence lengths。

## 13. Baseline、Golden 和发布

旧 `test_entry.py` 的以下能力必须保留，但迁移到显式模块和配置，不保留旧 wrapper：

- baseline regression 比较。
- golden 结果生成。
- OSS 归档上传。
- ODPS 摘要发布。
- 测试环境语义。

### 13.1 Baseline schema

新的 baseline 按稳定 case key 描述指标：

```json
{
  "case_key": "...",
  "metric": "compute_tpm",
  "value": 123.4,
  "direction": "higher_is_better",
  "threshold": 0.05
}
```

report 生成 `baseline_candidate.json`，但不会自动覆盖已提交 baseline。golden 由显式 CI target 生成或验证。

### 13.2 Publishers

- OSS 逻辑迁入 `publish/archive.py`，由 `output`/`publish` 配置决定归档内容。
- ODPS 是可选 publisher，只接收已聚合摘要。
- 认证信息只从运行环境读取，不写入配置、日志或 manifest。
- publisher 可配置 `required`。required publisher 失败会使整体退出非零；非 required 失败记录在 manifest 和 summary 中。
- 原 `test_env` 的隐式行为改成 config 中显式、可审计的 environment 标签。

## 14. 错误分类和退出码

Run 级错误：配置、能力、计划、资源、engine、collect、merge、report。

Case 级错误：请求、shape、reuse、timing、容量限制。

| Exit code | 含义 |
| --- | --- |
| 0 | 计划完整成功，所需报告/发布成功 |
| 2 | 配置或能力组合错误 |
| 3 | 计划或资源校验错误 |
| 4 | 模型/engine 生命周期错误 |
| 5 | case 失败或 partial run |
| 6 | collect、merge、report 或 required publisher 失败 |
| 130 / 143 | 对应中断/终止信号 |

约束：

- 不允许静默跳过 KV cache/reuse 校验。
- reuse mismatch 必须使 case 无效。
- AOne 缺 shard、重复 case 或 fingerprint mismatch 必须使 run 非成功。
- partial 必须返回非零。
- 中断必须保留已完成 case，并将 run 标记为 `interrupted`。

## 15. `batch_decode_test.py` 迁移映射

`batch_decode_test.py` 不整体搬迁到一个新文件，而是按职责拆分：

| 旧职责 | 新位置 | 处理方式 |
| --- | --- | --- |
| `main()` 顶层编排 | `main.py` + `orchestrator.py` | 迁移为统一入口 |
| argparse 和 partial 参数 | `main.py` + `config_schema.py` | 使用新契约重写 |
| 模型启动/关闭 | `execution/engine_session.py` | 抽成共享生命周期 |
| `_run_decode` | `modes/decode.py` | 保留 decode 调度语义 |
| `run_single` | `decode/grid_runner.py` | 保留固定网格执行核心 |
| decode batch 搜索 | `decode/tpot.py` | 保留并输出完整搜索轨迹 |
| distribution workload | `decode/distribution.py` | 保留并契约化输入 |
| cache grid 生成/校验 | `prefill/planner.py` | 采用 cache perf 语义 |
| seed/reuse 构造 | `prefill/runner.py` | 复用并强化校验 |
| resume | `prefill/checkpoint.py` | 按新 fingerprint 重写 |
| materialize/profile | `prefill/profile.py` | 与正式结果隔离 |
| test info/env/fingerprint | `manifest.json` 写入层 | 统一审计信息 |
| timeline/coredump/cleanup | `engine_session.py` | 共享基础设施 |

以下内容删除：

- 旧 `_run_prefill` 普通完整 prefill 路径。
- `--partial`。
- 根据 cache 文件是否存在而隐式选择行为。
- 未声明 engine 参数的自动透传。
- `CacheGridRunner.run_cartesian_grid` 中旧 decode 网格入口；核心循环迁入 `decode/grid_runner.py`。

## 16. 迁移阶段

### Phase 1：统一契约与基础设施

- 建立 config schema、capability matrix、ExecutionPlan 和结果 schema。
- 建立 EngineSession、manifest、原子 case writer 和状态模型。
- 迁移 baseline、golden、OSS、ODPS 能力。
- 此阶段不切换用户入口。

### Phase 2：Prefill

- 迁移 cache grid planner/runner。
- 完成 local run/resume/profile/report。
- 完成 AOne package、sharding、collect 和离线 report。
- 用新结果契约跑通 local/AOne 验证。

### Phase 3：Decode 与原子切换

- 迁移 local decode grid、distribution、TPOT batch search。
- 更新 Bazel targets/macros、CI 配置和文档。
- 完成 release gates 后，一次性将 `./tools/perf_test` 指向新实现并删除所有旧入口和旧编排代码。

阶段划分用于降低实现风险，不代表对用户长期暴露双入口。最终切换必须原子完成。

## 17. 最终删除与保留清单

### 17.1 删除

- `tools/cache_perf`。
- 旧 `tools/perf_test` 实现。
- `batch_decode_test.py`。
- `test_entry.py`。
- `perf_config.py`。
- cache perf 旧 CLI、orchestrator 和 CI planner。
- `CacheGridRunner.run_cartesian_grid` 旧入口。
- `--partial`、`--cache_grid_json` 和隐式 engine 参数透传。
- 旧 `perf_test`、`cache_grid_ci_test` Bazel macros/targets。
- 只验证旧配置、旧结果、旧 resume 或旧 baseline 的测试与文档。

删除前必须以 repository-wide reference scan 证明没有生产/CI 引用残留。

### 17.2 复用或重构

- server/client capability 和健康检查。
- cache prompt、seed、reuse 构造与校验算法。
- decode batch、distribution、TPOT 搜索算法。
- 性能公式、拟合和图表。
- workload sampling。
- timeline、coredump 和日志收集。
- archive、baseline、OSS 和 ODPS 的有效逻辑。

“复用”表示迁入新所有权边界并由新契约测试，不表示继续从旧入口调用。

## 18. 测试策略

### 18.1 Contract tests

- 所有合法 mode/action/backend 组合。
- 所有非法能力组合在副作用前失败。
- unknown field、类型、范围、路径和互斥字段错误。
- config normalization 和 fingerprint 稳定性。
- decode `output_length` 数组明确拒绝。

### 18.2 Planner tests

- prefill alignment、commit tail、最大长度、最小 compute、batch 约束。
- seed 和 case ID 的确定性。
- AOne LPT 分片完整、互斥且负载合理。
- decode grid/distribution case 生成。
- TPOT batch 搜索边界、容量失败和最佳 batch 选择。

### 18.3 Runner tests

使用 fake server 覆盖成功、请求失败、shape mismatch、reuse mismatch、invalid timing 和 engine crash。验证无效 case 不进入报告拟合。

### 18.4 状态与恢复测试

- 在各原子写入点模拟中断。
- resume 只保留 `ok` case。
- fingerprint 不一致拒绝 resume。
- profile 不修改正式结果。
- run directory 排他锁。

### 18.5 AOne tests

- generated Bazel package 通过 runfiles 获取配置。
- 远程不依赖提交机配置路径。
- shard ownership、重复、缺失、未知 case 和 fingerprint 校验。
- partial collect 的结果与退出码。

### 18.6 结果与发布测试

- prefill/decode result schema golden tests。
- 完整与 partial report golden tests。
- baseline direction/threshold 比较。
- golden generation target。
- fake OSS/ODPS 成功、失败和 required 语义。
- 凭证不进入日志和 manifest。

### 18.7 GPU smoke

- local prefill run/resume/profile/report。
- AOne prefill 单 shard 和多 shard。
- local decode grid。
- local decode distribution 和 TPOT 在基础逻辑稳定后加入 smoke。
- 所有 smoke 输出都通过统一 result envelope 校验。

## 19. Release gates

入口切换和旧代码删除前必须同时满足：

1. local prefill 四个 action 全部通过。
2. AOne prefill 单/多 shard 通过，collect 严格校验通过。
3. local decode grid、distribution、TPOT 搜索通过。
4. 新 baseline regression 可在 Bazel/CI 中作为门禁。
5. OSS/ODPS 失败路径和 required 语义通过。
6. repository-wide scan 无旧入口、旧 flags、旧 macros 的有效引用。
7. GPU smoke 结果均符合新 schema。
8. 新运行文档覆盖 local、AOne、resume、profile、report 和常见失败定位。

## 20. OpenSpec 处理

当前 `add-cache-perf-local-aone-runners` change 的方向与本设计冲突：它保留低层旧命令，并且只围绕 cache prefill 扩展。实现本设计时不应继续直接 apply 该 change。

下一步应新建 `unify-perf-test` OpenSpec change，并在 proposal/design 中明确 supersede 旧 change。旧 change 的处置和归档在新 change 建立后进行，避免两份规格同时指导实现。

## 21. 验收标准

- 仓库只有 `./tools/perf_test` 一个用户入口。
- 单个新 schema 配置可运行 local prefill、AOne prefill 和 local decode。
- 非法 mode/action/backend 组合无副作用地失败。
- prefill 只存在可控 reuse 的 cache grid 语义。
- decode 保留固定网格、distribution 和 TPOT batch 搜索，且 output length 为单值。
- prefill resume/profile/report 行为满足本设计的隔离和 fingerprint 约束。
- AOne 配置通过 Bazel runfiles 传递，不依赖提交机路径。
- 每个运行均产出可审计的 resolved config、plan、manifest、case results 和版本化 result envelope。
- baseline/golden/OSS/ODPS 能力已迁移并有失败路径测试。
- 旧入口、旧配置兼容代码和双轨文档全部删除。
