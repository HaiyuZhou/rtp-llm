# Batch 录制与回放：新增代码阅读指南

本文按调用路径解释新增功能在哪里介入、记录什么，以及数据如何串联。基于 `feat/glm5_cu13_rebase` 的 `7ee79279aa`，并包含随后 JSONL 底层切换 alog 的改动；后续代码变化时，以函数名定位为准。

- 录制提交：`85ae43bfb6`，请求事件、batch 快照、执行元数据、profiler 标记和录制基础设施。
- 回放提交：`7ee79279aa`，native 形状回放、离线 kernel 分析与比较。
- 操作命令见 [使用说明](references/batch_recording.md)；需求与目标见 [设计文档](batch_execution_record_replay_design.md)。设计目标不等于当前均已实现。

## 1. 先看整体调用关系

```text
在线请求
  NormalEngine::enqueue / enqueueMultiple
    scheduler->enqueue / enqueueGroup
      GenerateStream::recordSchedulerEnqueueTime     → enqueue 事件
  NormalEngine::loop → step → scheduler 调度
    executor->setRecordingStep
    NormalExecutor::process
      过滤采集范围、分配 execution_id               → first_scheduled 事件（仅一次）
      gatherModelInput → tpSyncModelInputs
      recordBatchInputs                             → batches.jsonl（异步落盘）
      rtp.execution(id=...) scope
        model_->forward                             → profiler CPU/CUDA 事件
      success / catch                               → executions.jsonl
      后续采样、输出分发
        NormalGenerateStream::enqueueGenerateOutput → first_token 事件（仅一次）
        GenerateStream::moveToNext                  → finish / cancel / error

以上 JSONL → ExecutionRecorder::submit → 有界队列 → snapshot worker
           → alog logger/FileAppender → 异步 flush → owner 目录
profiler  → TorchProfile → trace 文件；captures.jsonl 记录窗口及路径

离线分析：batch_trace_analyze.py → 请求延迟 + batch/kernel 关联 + 报告
形状回放：batch_replay.py → replay.plan → 独立 NormalEngine::runReplay
          → NormalExecutor::replayBatch → 同一个模型 forward
```

图中为逻辑关系，不代表各文件落盘的先后顺序就是业务时序。请求事件使用打点时刻的时间戳，而不是 writer 写文件的时刻。

## 2. 初始化与公共写入层

入口文件：[ExecutionRecorder.h](../rtp_llm/cpp/observability/ExecutionRecorder.h)、[ExecutionRecorder.cc](../rtp_llm/cpp/observability/ExecutionRecorder.cc)。

### 启用与身份

`ExecutionRecorder::instance()` 创建进程内单例。必须同时设置 `RTP_LLM_RECORD_DIR`、`RTP_LLM_RECORD_SESSION` 才启用；不是新增 start_server CLI 参数。每个进程创建 `owner-<pid>-<unix_ns>` 子目录，避免多 rank 写同一文件。

[NormalEngine.cc](../rtp_llm/cpp/normal_engine/NormalEngine.cc) 构造函数调用 `configure(world_rank, dp_rank)`；[NormalExecutor.cc](../rtp_llm/cpp/normal_engine/NormalExecutor.cc) 初始化时设置模型、拓扑、cache、decode 过滤等 metadata，供 manifest 描述录制环境。

所有录制 JSONL 的公共身份由 `identity()` 生成：`schema_version`、`session_id`、`replica_id`、`dp_rank`、`world_rank`、`owner_instance_id`。`RTP_LLM_RECORD_REPLICA` 未设置时使用 `dp<rank>`，多副本应显式指定不同 replica。

### 写入、轮转与完整性

`submit(file, line/make_line)` 只入队；`run()` 后台线程取任务、必要时构造 JSON，再调用 alog `logPureMessage()`。不再用 ofstream 逐行写 JSONL，也不每条主动 flush。

每个 owner/文件分片动态创建独立 logger 和 FileAppender，关闭继承，布局为 `%%m`；alog 异步 flush 默认阈值 64 KiB、间隔 100 ms。使用纯消息接口避免 printf-style `log()` 受 `alog.max_msg_len` 截断。这是通过 alog API 配置底层，不是复用 `alog.conf` 的 engineAppender；全局日志路由和格式不变。轮转命名与提交预算仍由录制器管理，manifest 仍通过 ofstream 写临时文件再 rename。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `RTP_LLM_RECORD_QUEUE_SIZE` | 4096 | 队列最多容纳的记录条数，不是字节数 |
| `RTP_LLM_RECORD_FILE_BYTES` | 100 MiB | 单文件轮转阈值，分片追加 `.1`、`.2` 等 |
| `RTP_LLM_RECORD_TOTAL_BYTES` | 1 GiB | 每个 owner 提交给 alog 的 JSONL 字节预算，不包含 trace |

队列满时丢整条记录，不截断 requests 数组；超总预算或 snapshot worker 出错时停止接受新记录。它不是零开销：生产线程仍有元数据拼接、锁、长度快照分配/D2H；后台有 JSON 构造和 alog 消息复制。JSONL 是紧凑的一行一个对象，不做缩进美化；离线报告的格式化不在在线热路径。

`manifest()` 通过临时文件和 rename 更新，不追加。每提交 64 条给 alog 更新一次，`close()` 停止接收、drain 队列、join 后尝试 flush 自己的 alog 文件并更新状态，不调用全局 alog shutdown。运行中计数可能滞后。`errors` 是录制器可观察错误计数，不是模型推理失败次数。

新增 `storage_backend=alog`、`delivery_policy=best_effort`。`submitted_to_alog`、`bytes_submitted` 只计提交量；无法确认落盘量和 alog 内部丢弃量，因此 `written`、`bytes_written`、`sink_dropped` 为 null，`complete` 始终 false。`closed=true` 仅表示上层完成关闭流程。缺日志按缺失数据处理，不补造；完全未记录的请求也无法被分析器发现。

## 3. 请求生命周期：打在引擎边界，不是 HTTP 边界

核心状态对象是 `RecordedRequest`。它分配匿名 `request_id="r"+nextId()`，不直接暴露外部 HTTP/RPC request ID；由 stream 的 shared_ptr 持有，stream 复制时共享状态，避免重复事件。

| 事件 | 调用路径与新增动作 | 口径 |
| --- | --- | --- |
| `enqueue` | scheduler → `GenerateStream::recordSchedulerEnqueueTime` → 创建 `RecordedRequest` → `enqueue()` | 首次记录入队；跳过 fake stream |
| `first_scheduled` | `NormalEngine::step` → `NormalExecutor::process` → `RecordedRequest::scheduled` | executor 准备本轮输入前打点，不代表 forward 已成功开始或完成 |
| `first_token` | `NormalGenerateStream::enqueueGenerateOutput` → push 输出队列 → `published()` | 首次发布非空 token 输出，不是客户端接收时间 |
| `finish/cancel/error` | `GenerateStream::moveToNext` 首次进入 FINISHED → `terminal()` | 根据 error code 和 CANCELLED 判定，三者互斥 |

定位文件：[GenerateStream.cc](../rtp_llm/cpp/engine_base/stream/GenerateStream.cc)、[NormalGenerateStream.cc](../rtp_llm/cpp/normal_engine/NormalGenerateStream.cc)。

FIFO 单请求和 group enqueue 原已有入队计时入口，本次复用；[BatchDecodeScheduler.h](../rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h) 补上同一入口。HTTP 层拒绝、尚未进入引擎队列的请求不会因此自动产生事件。

`RecordedRequest` 内部用互斥锁和状态位去重：decode 每轮仍可能调用 `scheduled()` / `published()`，但不会每轮输出 first_scheduled / first_token。正常有输出且完整结束的请求通常是四行；提前取消或错误可能没有首调度/首 token，不能强求四行。

每行携带 `event_seq`、monotonic/unix 纳秒时间、`clock_id`、`output_mode`、已发布 token 计数等。`first_token` 和终态的 `execution_id` 当前为 null，避免异步输出错误归属到后续轮次；不是通过“当前全局 execution_id”补值。

离线在同一请求/时钟域内计算：

- 初始排队：`first_scheduled - enqueue`。
- 引擎首 token 延迟：`first_token - enqueue`。
- 引擎请求总耗时：`terminal - enqueue`。
- 首调度后耗时：`terminal - first_scheduled`，包含后续等待和输出，不是纯 GPU 时间。

非流式请求可能到最终输出才发布 token，必须按 `output_mode` 分开解释。worker 崩溃可能只有 enqueue/first_scheduled，不能凭 executions 中的 error 伪造请求终态或 TTFT。

## 4. Batch 快照：在实际模型输入准备完后、forward 前

重点阅读 [NormalExecutor.cc](../rtp_llm/cpp/normal_engine/NormalExecutor.cc) 的 `process()` 和文件内辅助函数 `recordBatchInputs()`。

### 4.1 分配 ID 与过滤

`NormalEngine::step()` 在录制启用时递增 `recording_step_`，经 `Executor::setRecordingStep()` 传给 executor。

`process()` 的输入 owner 条件是：录制启用、非 warmup、非 propose、`tp_rank==0`。owner 为符合范围的本轮 forward 分配 `execution_id`，同时调用请求的 `scheduled()`。

`RTP_LLM_RECORD_DECODE` 默认关闭，启动时读取并缓存：

| 本轮组成 | 开关关闭 | 开关打开 |
| --- | --- | --- |
| 纯 prefill | 记录 | 记录 |
| 纯 decode | 跳过整轮 batch/execution | 记录 |
| prefill + decode | 跳过整轮 batch/execution | 记录完整混合 batch |

过滤发生在长度快照分配、D2H、event 和 batch JSON 构造之前。不会只从混合 batch 中摘出 prefill 伪装成完整输入。请求生命周期不被该开关关闭；首次调度若在被过滤 batch 中，事件的 execution_id 为 null。scope 可能出现 ID=0，离线不将其关联为已录制执行。

### 4.2 TP 传播与快照

`gatherModelInput()` 后设置 `GptModelInputs.record_execution_id`、`record_scheduler_step_id`。[ModelTypes.cc](../rtp_llm/cpp/models/ModelTypes.cc) 将它们放入已有 shape hints 广播，所有 TP rank 得到同一 execution_id；没有为每个 rank 另造一份 batch ID。

TP 同步完成后，只有输入 owner 调用 `recordBatchInputs()`。它按模型 slot 顺序（decode 在前、prefill 在后）展开 stream 和 sequence，而不是保存一个 batch 平均长度。

| requests 字段 | prefill 来源 | decode 来源 |
| --- | --- | --- |
| `q_tokens` | `input_lengths[slot]` | 1 |
| `kv_tokens_before` | `prefix_lengths[prefill_index]` | `sequence_lengths[decode_index]` |
| `prompt_tokens` | `stream->inputLength()` | 同左，保留原 prompt 长度口径 |
| `prefix_cache_hit_tokens` | `stream->initialReuseLength()` | 同左，不是每轮累计 KV 长度 |

另外记录 request_id、sequence_id、batch_slot、phase、is_fake。一个多序列请求可占多个 slot，因此 requests 数组长度不一定等于独立请求数量。

长度 tensor 被异步复制到 pinned host buffer；在生产 stream 上记录 event。提交给 writer 的闭包持有源 tensor、目标 tensor 和 event，writer 等待 event 后读取长度，标记 `length_source=device_snapshot`。这样不需要为了 JSON 在生产线程等待 D2H 完成。

快照数量与实际输入不一致时抛异常，由调用处捕获并记 recorder error；不会输出缺项的 batch。完整快照作为一条任务写入 `batches.jsonl`。

**这里没有 TTFT 回填。** batch 写在 forward 前；之后只能用 request_id 与 request_events 联表获得请求延迟。

## 5. Forward 执行记录与 Graph 元数据

同一 `process()` 在 `model_->forward()` 外包裹 `rtp.execution(id=...)`，成功返回写 `executions.jsonl`；catch 写 status=error 后继续抛出原异常，不吞掉推理错误。

对于被采集且到达该打点的 forward，每个 rank 各写一行；成功行包含执行模式、bucket、物理 batch/token 信息和 `cpu_submit_us`。错误行只含公共身份、execution_id、status，不保证具有成功行的全部字段。

新增模式信息由以下路径提供：

- [OpData.h](../rtp_llm/models_py/bindings/core/OpData.h)：模型输入侧的录制元数据字段。
- [PyWrappedModel.cc](../rtp_llm/cpp/models/PyWrappedModel.cc) `forward()`：实际进入 eager 时标记 kind=1，decode graph=2，prefill graph=3，并记录实际 bucket；未提供信息的路径保持 unknown。
- [cuda_graph_runner.cc](../rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc) `replayGraph()`：新增 `rtp.graph_replay(bucket=...,prefill=...)` trace scope。

decode graph 的 bucket 单位为 sequences，prefill graph 为 tokens，不能混作同一种 batch size。`cpu_submit_us` 是 CPU 侧 forward 调用区间，不是 GPU kernel 总耗时，也不意味着 GPU 已全部完成。

例如 TP=8、一个被采集 batch 内两个单序列请求，成功完整执行时通常是：owner 一行 batches、各 rank 共八行 executions；两个请求各自生命周期通常各四行。生命周期不是“每个 batch 再写四行”。

## 6. Profiler 窗口与 batch → kernel 关联

[NormalEngine.cc](../rtp_llm/cpp/normal_engine/NormalEngine.cc) 的 `loop()` 支持 `RTP_LLM_RECORD_PROFILE_START` / `RTP_LLM_RECORD_PROFILE_STEPS`，配置已有 `StepWindowProfiler`；也可使用已有 profiler 控制接口。单独启用 JSONL 录制不等于开启 kernel 采集。

[TorchProfiler.cc](../rtp_llm/cpp/engine_base/TorchProfiler.cc)：

- `TorchProfile::start()` 记录 captures 的 start 事件与 capture_id。
- `stopAndCollect()` 收集结果，记录 collected 与 trace_file。
- trace 保存沿用已有 profiler 保存路径/线程；目录并不保证在录制 owner 目录内。collected 早于实际 save 完成，不能当作保存成功凭证。

关联链如下：

```text
batches.execution_id
  → trace 中 rtp.execution(id=...) CPU scope
  → 同 pid/tid、时间区间被 scope 包含的 CUDA runtime/driver 事件
  → args.correlation
  → 相同 correlation 的 GPU kernel（名称、dur、device、stream）
```

`execution_id` 和 `rtp.execution` 是本次新增；CUDA correlation 是 CUPTI/Kineto 已有机制。本次没有修改 kernel 参数来传 ID。GPU 异步执行可能落在 CPU scope 结束之后，因此不能用 GPU 时间与 CPU scope 重叠来猜归属。

`captures.jsonl` 帮助找到 trace，不直接承载 batch→kernel 映射。默认不采 decode 时，profiler 仍可能记录 decode kernel，但缺少可关联的非零 execution_id。

## 7. 离线分析：读哪些数据、如何降级

入口：[batch_trace_analyze.py](../rtp_llm/test/perf_test/batch_trace_analyze.py)。推荐阅读顺序：`main()` → `request_latencies()` → `correlate_trace()` → 汇总与报告输出。

- `main()` 读取一个 owner 的 manifest、request_events 分片和 batch 分片。非输入 rank 使用 `--batch-dir` 指向同 session/replica/DP 的输入 owner；`--world-rank` 必须匹配 manifest。trace 需由调用者选择正确 owner 的文件。
- `request_latencies()` 按请求整理事件、检查缺失/重复/时钟等问题并计算延迟。best_effort 下以每个请求实际事件为准：完整请求可计算耗时，缺事件的请求为 partial；损坏 JSONL 行警告后跳过。旧版严格录制仍在 manifest 不完整时将所有请求耗时置 null。
- `correlate_trace()` 实现上一节的两级关联；多个 CPU runtime pid 合并的 trace 会被拒绝。kernel 只有唯一 execution owner 时才 matched，否则保留 unmatched/ambiguous。跨线程提交或缺失 correlation 时不承诺全部关联成功。
- 报告检查 trace 中 execution 是否有 batch，输出 `missing_batch_execution_ids`。kernel 汇总包含 duration 总和、区间并集和跨度；有重叠执行时三者并不相等。
- 输出 `report.json`、`report.html`、`kernel_events.jsonl`。共享 batch 的 kernel 不能据此精确拆成单请求独占 GPU 时间。
- `--replay-dir` 追加回放比较，核对配置与执行模式/bucket，按 execution/rank/kernel 名称汇总。`config_match` 不证明权重一致，完整权重指纹尚未实现。

## 8. 离线回放：独立入口，不回灌线上 scheduler

### Python 计划与进程管理

[batch_replay.py](../rtp_llm/test/perf_test/batch_replay.py) 的 `make_plan()` 校验 batch、slot/phase/长度和 execution ID，将每行编码为 `RTP_BATCH_REPLAY_V1` 计划。`main()` 支持筛选 execution、warmup/repeat；要求全新输出目录，保留源 batch/manifest。

不传引擎命令时只生成计划。传命令时设置 `RTP_LLM_REPLAY_PLAN`，移除子进程的 `RTP_LLM_RECORD_DIR`，启动独立进程组；监测各 rank 的 result 文件，完成/失败/超时后清理自己创建的进程组。不是向当前在线服务发送回放请求。

### C++ native 路径

`NormalEngine::loop()` 发现 replay plan 后直接进入 [BatchReplay.cc](../rtp_llm/cpp/normal_engine/BatchReplay.cc) 的 `runReplay()`，不进入普通调度循环；enqueue 和 enqueueMultiple 拒绝在线流量。

`runReplay()` 的主要步骤：

1. 输出运行配置，拒绝不支持的引擎/cache 类型。
2. 为每个快照构造 `NormalGenerateStream`，恢复 phase、q/KV 配对与 prompt 长度元数据；确定性生成合法 token ID，分配并清零 KV，关闭前缀复用等缓存查询。
3. 每次 warmup/repeat 前重置 cache；调用 `NormalExecutor::replayBatch()`。
4. `replayBatch()` 复用 gatherModelInput → TP 同步 → 带原 execution_id 的 scope → model forward → CUDA 同步 → 释放缓冲区，不经过采样/在线请求推进。
5. 写各 rank 的同步耗时；另单独采集一次 forward 的 profiler trace，最后写 complete/error 状态。

每条 batch 快照独立回放，并不是让原请求按历史 decode 链连续生成。无 profiler 的测量包含输入准备和同步，不等同于纯 kernel 耗时；profile 那一次也不是 repeat 耗时样本本身。

当前 native 支持范围是 DP=1、普通融合文本模型、单个未量化 FP16/BF16 full-KV group，可用 TP 多卡。MTP、多模态、PD 分离、特殊/量化 KV、FFN 分离等被显式拒绝。**DSV4 的 FP8 特殊 KV 不在当前 native 回放支持范围内**，录制成功不等于可原生回放。

## 9. 测试入口与后续 TODO

| 文件 | 关注点 |
| --- | --- |
| [ExecutionRecorderTest.cc](../rtp_llm/cpp/observability/ExecutionRecorderTest.cc) | recorder、事件去重与存储行为 |
| [RecordingEngineTest.cc](../rtp_llm/cpp/normal_engine/test/RecordingEngineTest.cc) | mock 模型引擎的录制/回放接入 |
| [batch_recording_test.py](../rtp_llm/test/perf_test/batch_recording_test.py) | 离线事件、关联和计划校验 |
| [batch_trace_gpu_test.py](../rtp_llm/test/perf_test/batch_trace_gpu_test.py) | 小 GPU 工作负载下 eager/graph 的 profiler 关联 |

这些测试入口不代表当前分支真实模型、多卡、生产性能均已通过。本文是代码阅读说明，没有执行新的 GPU 验收。

后续拓展目标（当前未实现/未完成验收）：

- 动态启停、限时 shutdown drain、分类型丢弃指标，量化吞吐/TTFT/TPOT 开销。alog 的异步缓冲不等于已有性能验收。
- 更完整的软件/权重指纹；跨 rank、跨 capture 完整性验证及 trace 保存成功状态。
- 特殊/量化 KV、多 DP 回放；复现 cache 共享、真实内容/MoE 路由和原 graph bucket。
- 可选的首 token 对应执行 ID 传播，需同时覆盖异步输出，不能简单读取“当前轮 ID”。
- 请求终态在进程崩溃时的外部补充观测；保留不完整标记，不凭空补造成功或延迟。
