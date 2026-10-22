# 连续 Prefill 满载压力测试与跨 Step CPU Overhead 分析方案

> 本文选用了 C++ cache benchmark 基线，不适用于本次基于 `tools/cache_perf continuous` 的需求。已由 [Cache Perf 连续 Prefill 满载压力方案](../../../../../test/perf_test/cache_grid/docs/continuous_full_pressure_plan.md) 替代，本文仅保留为历史草案，不作为实施依据。

状态：待实现的设计方案。本文中的新增参数、指标和阶段控制尚未实现；现有能力与拟新增能力在下文分别说明。

## 1. 目标与范围

先通过请求把 GPU cache 推到容量水位并溢出至 memory cache，再保持连续 prefill 压力，测量相邻 step 之间的 CPU 开销，并分析这些开销是否阻塞下一批执行。

保留现有单 foreground scheduler、C32 逻辑请求并发、后台 cache task pool、长度和复用率分布，以及 BASE/CONTINUATION 请求依赖。CONTINUATION 表示复用历史前缀的新一次 prefill，不代表 decode。

一个 step 定义为一个非空 READY batch 的 forward；一个请求在一次 prefill 后完成，不进入逐 token decode。持续提交请求即可产生多个 step，无须依赖 chunked prefill。C32 是请求上下文上限，不保证每个 batch 恰好有 32 个请求。

当前模拟器使用缩小 payload 的 block pool 和固定 100ms forward sleep，适合研究 cache/scheduler 的 CPU 成本。真实 prefill 用于验证这些成本与 GPU 计算重叠后，实际暴露了多少执行停顿；两种结果分别报告。

## 2. 现有实现与缺口

- `TreeBenchmarkRunner` 固定 warmup 15 秒，随后检查 device 占用不低于 75%、host 非空、device eviction heap 非空及完成请求数等条件，但 `pressure_ready=false` 不会阻止正式测量。
- warmup 结束会 drain 活动请求和后台任务，再等待 2 秒 attach profiler，正式测试从活动请求队列为空的状态启动。
- `OnlineTreeScheduler` 已有 match、insert、load commit、match-to-ready 等计时，但没有直接统计相邻 forward 之间的完整间隔和实际线程 CPU 时间。
- 当前 device/host pool 各有 32,768 blocks，device watermark 约为 0.8，host watermark 约为 0.9；payload 经过缩放，不能用 pool 占用推断整卡物理显存占用。
- 真实引擎的 `StepWindowProfiler` 已支持跨多个 step 连续采集，已有 scheduler、输入准备、model forward 和输出处理等 Torch profiling 范围。

主要代码位置：

- [TreeBenchmarkRunner.cc](../TreeBenchmarkRunner.cc)
- [OnlineTreeScheduler.cc](../OnlineTreeScheduler.cc)
- [TreeWorkloadGenerator.h](../TreeWorkloadGenerator.h)
- [BenchmarkFixture.cc](../BenchmarkFixture.cc)
- [TorchProfiler.cc](../../../../engine_base/TorchProfiler.cc)
- [NormalExecutor.cc](../../../../normal_engine/NormalExecutor.cc)

## 3. 整体执行流程

```text
setup -> pressure_fill -> steady_warmup -> measured -> drain/finalize
         请求灌注         正式分布稳定运行    连续采样      单独收尾
```

| 阶段 | 工作内容 | 完成条件 |
| --- | --- | --- |
| setup | 创建 cache，生成确定性请求 trace，预分配统计缓冲区 | 初始拓扑和资源检查通过 |
| pressure_fill | 通过低复用请求建立容量压力，驱动 GPU 向 host 降级 | 达到灌注目标，且存在成功降级证据 |
| steady_warmup | 切换到正式请求分布，持续提供压力，提前 attach profiler | 连续观察窗口满足稳定压力条件 |
| measured | 从明确的 step 边界开始连续采样，默认运行 60 秒 | 到达测量截止时间 |
| drain/finalize | 停止 admission，有界完成或清理在途请求和后台任务 | 资源残留检查通过，收尾单独计时 |

warmup 到 measured 只切换测量窗口，保留活动请求、REQUEST refs、依赖状态、trace 游标和后台任务。当前 `runPhase()` 在阶段末尾会 drain，因此需要增加支持连续运行的阶段控制，不能直接重复调用现有接口实现切换。

## 4. 灌注请求与容量压力判定

### 4.1 灌注方式

灌注使用预生成的独立 key 空间、低复用请求和足够的唯一 suffix，走完整生命周期：

```text
match -> allocate/load -> forward -> insert -> release
```

通过正常 watermark 机制完成 GPU 到 host 的降级。完成请求后释放 REQUEST refs，使缓存可被淘汰。初版保留与正式阶段相同的 forward 模拟行为。

灌注按新增唯一 blocks 和实际驻留判断，累计插入量仅作辅助，不能仅凭请求数或累计写入量认定 cache 已满。灌注流量与正式 workload 分别记录 trace hash 和统计。

### 4.2 满载口径

“打满”按配置的 cache 水位定义。当前 device watermark 约为 80%，正常降级会主动释放空间，不能要求占用长期等于 100%。整卡物理显存接近耗尽属于另一种容量配置，不能由当前 scaled pool 推断。

建议采用以下初始判据，具体阈值通过小型 GPU 测试校准后固化到 resolved config：

| 条件 | 判据 |
| --- | --- |
| GPU 驻留 | 各目标 device pool 接近配置 watermark；初始允许低于水位 5 个百分点 |
| Host 驻留 | 各目标 host pool 达到明确目标，初始建议容量的 25% |
| 降级成功 | 已完成的 GPU 到 host 降级 blocks 增加，不能只统计提交或 pending |
| 稳定推进 | 正式请求分布下持续完成请求，无不可恢复失败 |
| 压力持续 | 初始建议连续 3 个 1 秒观察窗口满足条件，窗口内有降级进展 |

memory cache 不要求填满；后续若要研究 host 淘汰，应作为独立场景。

### 4.3 未达标处理

灌注设置最大时间和请求预算，避免无限等待。未达标就输出 `pressure_not_reached`、各 pool 状态及原因，不进入正式测量，随后执行有界清理。

失败结果应区分容量目标不可达、请求供给不足、分配失败、load/evict 失败、trace 耗尽和清理超时。

## 5. 持续满载与阶段边界

`steady_warmup` 恢复正式的长度、复用率和 continuation 分布，让缓存从灌注形态过渡到测试形态。持续维持 backlog，区分 token budget 限制、依赖未就绪、load 未完成和请求供给不足；不能把所有“没有 READY”都归为 cache 慢。

profiler 提前 attach，attach 期间继续运行 warmup。压力达标后在一个 forward 开始边界开启 measured 窗口。deadline 后停止接纳新请求，收尾耗时单独记录；仅统计完整落在窗口中的 step 间隔。

生命周期计数持续累计，跨阶段请求保留归属。通过时间戳或计数基线划分窗口，避免直接清零导致 load 提交/完成等计数失配。窗口内提交数与完成数不要求天然相等，需结合窗口起止的在途状态解释。

measured 全程低频采集压力，报告达标窗口比例和最长失压时间。初始可要求达标比例不低于 95%；不达标仍保留数据，但不作为有效满载结果。压力目标、采样周期、容差和有效性门槛都写入 resolved config。

## 6. 跨 Step Overhead 的测量方法

### 6.1 时间边界

模拟器的核心测量区间为：

```text
forward(i) 实际返回
    -> insert/release
    -> 调度与请求准备
    -> match/allocate/load
    -> readiness 检查与等待
forward(i+1) 开始
```

用单调时钟测墙钟时间，用线程 CPU 时钟测 foreground 线程实际 CPU 消耗。forward sleep 的实际时长单独记录，不以配置的 100ms 代替实际时间。

### 6.2 指标

以下为拟新增或完善的指标，名称在实现时统一到现有输出规范。

| 指标 | 解释 |
| --- | --- |
| `inter_step_wall_ns` | 相邻 forward 之间的完整墙钟间隔 |
| `inter_step_thread_cpu_ns` | 同一区间 foreground 线程实际 CPU 时间 |
| 分段 wall/CPU 时间 | insert/release、refill/依赖处理、match、分配、load commit、其他调度工作 |
| 显式等待 | 无 READY 轮询、allocation retry sleep 等等待的实际时长 |
| 后台 CPU 成本 | load/evict/store 工作线程的执行成本，单独统计 |
| 关联上下文 | step_id、batch 大小、tokens、cache 占用、load/evict 进展、重试和等待原因 |

输出样本数、均值、p50/p95/p99/max，同时按 batch 大小和 token 规模解释差异。保留逐 step 数据，使长尾间隔能够与压力快照和 timeline 对应。

不能以“总耗时减去 step 数乘以 100ms”代表 CPU overhead，也不能把 `wall - thread_cpu` 直接称为锁等待。并行后台 CPU 时间不与前台墙钟间隔相加；父子范围和跨线程范围也不能直接相加。

## 7. Profile 与 Timeline 采集

### 7.1 模拟器

增加阶段、step_id、forward sleep 和上述子阶段范围。Torch timeline 使用 `record_function`；Nsight Systems 需要对应 NVTX 标记，不能假定两者自动互通。

计时样本预分配，压力快照低频采集，结果序列化放到测量窗口之外。避免每 step 打日志或抓取重型快照，并记录统计自身的开销。

模拟器没有真实模型计算 kernel，其 timeline 用于解释 cache/scheduler 开销、异步 load/evict 活动及等待，不能直接据此推导真实模型的 GPU 计算停顿。

### 7.2 真实 Prefill

在灌注达标并保持流量后，使用现有控制接口连续采集，例如：

```bash
curl -X POST http://HOST:PORT/start_profile \
  -H 'Content-Type: application/json' \
  -d '{
    "trace_name": "prefill_full_cache",
    "start_step": 0,
    "num_steps": 20
  }'
```

采集要求：

- 使用 prefill-only 节点；单机服务可通过 `max_new_tokens=1` 构造只进行 prefill 的请求，最终以实际执行标记验证。
- 核对 `executor.model_forward` 标记中的 `ctx_batch>0`、`gen_batch=0`，确认没有混入 decode。
- 持续提供请求直到目标执行步数完成。20 个请求可能合成少数几个 batch，不能用请求数代替 step 数。
- 当前 profiler 会跨 step 保持开启，但第一次启动发生在输入准备和 TP 同步之后，因此分析中间完整 step 及其间隔。
- 确认采集包含 GPU kernel 和 CPU/CUDA 关联信息，不能仅凭文件存在判断采集成功。

### 7.3 分析顺序

先在 timeline 中找到相邻 prefill batch，同时检查 CPU 与 GPU 时间轴：

```text
CPU: [forward i 提交] [输出/cache/调度/输入准备] [forward i+1 提交]
GPU:      [prefill i 执行..............] [间隔] [prefill i+1 执行...]
```

| 观察项 | 分析方式 |
| --- | --- |
| Host 侧 forward 间隔 | 查看相邻 `executor.model_forward` CPU range 之间的输出、调度、输入准备及等待 |
| GPU 计算间隔 | 查看前一批相关 GPU 工作结束到下一批计算开始的间隔，同时检查 copy 和通信 stream |
| 下一批提交晚 | 沿 CPU 路径检查输出处理、schedule、输入准备、资源等待和线程状态 |
| 已提交但执行晚 | 检查 GPU stream 依赖、copy、通信及其他 GPU 工作 |
| CPU range 长但缺少内部细节 | 用 perf 定位执行热点，用线程调度和 OS Runtime 信息区分运行、阻塞和被抢占 |

已有真实引擎标记包括 `executor.dispatch_output*`、`engine.normal.schedule(...)`、`executor.wait_prev_dispatch*`、`executor.gather_model_input` 和 `executor.tp_sync_input`。

从下一批第一个计算 kernel 沿 CUDA correlation 回查提交时间，确定延迟下一批执行的依赖链。GPU 上的空白不能全部归为 CPU overhead；部分 CPU 工作也可能与上一批 GPU 计算重叠，没有造成停顿。

普通火焰图用于回答“CPU 花在哪里”，timeline 用于回答“为什么这一批启动晚了”。工具使用参考 [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) 和 [仓库采集指南](nsys_collection_guide.md)。

## 8. 实现范围

| 模块 | 拟修改内容 |
| --- | --- |
| TreeWorkloadGenerator | 确定性灌注 trace、独立 key 空间、灌注与正式 workload 的 hash 和元数据 |
| TreeBenchmarkRunner | 阶段控制、压力硬门槛、稳定窗口、测量有效性和失败结果 |
| OnlineTreeScheduler | 连续执行、阶段边界、step_id、跨 step wall/CPU 计时和等待分类 |
| Cache 观测接口 | 复用 pool snapshot；确认成功降级等累计计数的可读性，缺失时增加最小统计能力 |
| JSON/report | 配置、阶段、压力、逐 step 数据、耗时分布和失败原因 |
| Profiling | 阶段与子阶段标记、连续采集和结果关联 |
| Tests/docs | 阶段行为、计时口径、GPU 降级验证及使用文档 |

不通过解析日志取得压力判据，也不为灌注绕过真实 cache 生命周期。

## 9. 结果产物与对照要求

每次运行输出：

- resolved config、代码与 workload 标识、灌注和正式 trace hash。
- 各阶段时长、进入 measured 时的压力证据及失败原因。
- 压力时间序列、达标窗口比例、最长失压时间。
- 逐 step 样本、overhead 分布、batch/tokens/load/evict 等关联上下文。
- 生命周期统计、窗口起止在途状态、最终资源清理结果。
- profiler 文件位置及采集配置。

task-pool 对照固定其他配置，并记录阶段游标和实际请求分布。相同 trace hash 不代表时间窗口内执行了完全相同的请求；需要结合长度、复用率、batch 大小、tokens 和成功完成数解释差异。

profiling run 用于归因，独立裸跑用于性能基线，并量化采样对结果的扰动。短窗口约 20 个 step 用于检查典型间隔，更长的轻量统计用于分析 p50/p99，不能只凭少量 step 给出稳定的长尾结论。

## 10. 验证与验收

### 10.1 验证顺序

1. 使用可控 cache/时钟验证压力未达标不能测量、连续阶段不清空状态、跨阶段计数正确、间隔边界与等待分类正确。
2. 使用小型 GPU 测试确认真实 GPU 到 host 降级、压力门槛、持续运行和最终资源释放。
3. 运行正式容量的裸跑基线和 profile，核对真实压力形态及采样扰动。
4. 在真实 prefill 流量上连续采集多个 step，验证 CPU 成本与 GPU 计算间隔的关系。

### 10.2 验收标准

- 正式测量前压力已达标；不达标不会输出有效满载性能结果。
- 采样期满足声明的压力持续性标准，且请求持续推进。
- warmup 到 measured 不清空活动状态，不引入人为的无请求间隔。
- 能取得多个完整 prefill step 间隔，区分墙钟时间、线程 CPU 时间和显式等待。
- 能结合 timeline 解释典型及长尾间隔，并识别下一批执行的关键依赖。
- 不混淆模拟器结论、真实 GPU 停顿和整卡物理显存压力。
- 最终活动请求、pending tickets/tasks 和 REQUEST refs 全部归零；异常路径有界清理并保留失败结果。

## 11. 实施顺序

1. 先实现请求灌注、压力门槛和低频压力观测，确认能稳定建立目标 cache 状态。
2. 再实现连续阶段切换，消除 warmup 到 measured 的 drain 和空队列重启。
3. 增加跨 step wall/CPU 指标、等待分类、逐 step 数据和 profile 范围。
4. 完成自动验证、正式容量基线和多 step timeline 分析，固化阈值与报告口径。
