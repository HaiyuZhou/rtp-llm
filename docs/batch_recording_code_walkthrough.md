# Batch 流量录制：新增代码阅读指南

范围更新（2026-09-16）：本期只保留流量录制，已移除新增 profiler 打点和自动 trace 采集。下面第 7 节的 trace 解析仅兼容历史数据，第 8 节回放代码保留供后续拓展，不作为本期验收内容。

batch 日志配置更新：`ExecutionRecorder::run()` 对 batch 任务先检查 batch_schedule logger 的 INFO 级别，允许提交时调用 `logBinaryMessage(INFO, line)`，不再动态创建 batch appender。宏和 logger 获取函数统一定义在 [Logger.h](../rtp_llm/cpp/utils/Logger.h)，使用者依赖 core_utils。它取名为 batch_schedule 的 logger，检查配置级别后调用 logBinaryMessage，直接写完整 JSON，不进行 printf 格式化、不截断。默认配置在 [alog.conf](../rtp_llm/config/alog.conf)，输出到服务工作目录下固定的 `logs/batch_schedule.log`，不添加 PID 后缀。owner 目录仅保留 manifest.json。

本文按调用路径解释新增功能在哪里介入、记录什么，以及数据如何串联。代码按录制、回放两个提交拆分；后续代码变化时，以函数名定位为准。

- 录制提交：请求时间、CPU batch 快照和 alog 录制基础设施，不新增 profiler 打点。
- 后续回放提交：native 形状回放、离线分析与历史 trace 比较。
- 操作命令见 [使用说明](references/batch_recording.md)；需求与目标见 [设计文档](batch_execution_record_replay_design.md)。设计目标不等于当前均已实现。

## 1. 先看整体调用关系

```text
在线请求
  NormalEngine::enqueue / enqueueMultiple
    scheduler->enqueue / enqueueGroup
      GenerateStream::recordSchedulerEnqueueTime     → 分配不可变请求 ID，保留原有调度指标时间
  NormalEngine::loop → step → scheduler 调度
    NormalExecutor::process
      过滤采集范围、分配 exec_id
      gatherModelInput → tpSyncModelInputs
      recordBatchInputs                             → 冻结长度和请求身份快照
      model_->forward                               → 不添加录制专用 profiler scope
      success / catch                               → 不输出独立执行日志
      后续采样、输出分发
        GenerateStream::update                     → 填入已知 TTFT
        dispatch 完成                              → 提交一条完整 batch 日志

batch JSONL → ExecutionRecorder::submitBatch → 有界队列 → snapshot worker
           → batch_schedule logger → 异步 flush → logs/batch_schedule.log

离线分析：batch_trace_analyze.py → 请求延迟 + batch 长度 + 报告
形状回放：batch_replay.py → replay.plan → 独立 NormalEngine::runReplay
          → NormalExecutor::replayBatch → 同一个模型 forward
```

图中为逻辑关系，不代表各文件落盘的先后顺序就是业务时序。TTFT 在输出更新时冻结，不使用 writer 写文件的时刻计算。

## 2. 初始化与公共写入层

入口文件：[ExecutionRecorder.h](../rtp_llm/cpp/observability/ExecutionRecorder.h)、[ExecutionRecorder.cc](../rtp_llm/cpp/observability/ExecutionRecorder.cc)。

### 启用与身份

`ExecutionRecorder::instance()` 创建进程内单例。必须同时设置 `RTP_LLM_RECORD_DIR`、`RTP_LLM_RECORD_SESSION` 才启用；不是新增 start_server CLI 参数。每个进程创建 `owner-<pid>-<unix_ns>` 子目录，保存 manifest；batch 文件不在此目录，由 alog.conf 单独配置固定路径。

[NormalExecutor.cc](../rtp_llm/cpp/normal_engine/NormalExecutor.cc) 非 warmup、非 propose 初始化时统一调用 `configure(world_rank, dp_rank, metadata)`，一次性设置 rank 和模型、拓扑、cache、decode 过滤等 metadata，供 manifest 描述录制环境。

录制器显式区分 Disabled、Unconfigured、Ready、Failed 和 Closed。构造函数只准备 owner 身份；配置完整、初始 manifest 写入成功且 worker 创建成功后，才发布 Ready 并允许提交。初始化前的 submitBatch 返回 false，不增加 generated；重复 configure 返回 false，保留原配置，不再依赖提交计数。关闭或初始化失败后不能重新初始化。配置发布后保持不可变，worker 不会读取半初始化的 rank/metadata。

manifest 的完整身份由 `identity()` 生成：`schema_version`、`session_id`、`replica_id`、`dp_rank`、`world_rank`、`owner_id`。`RTP_LLM_RECORD_REPLICA` 未设置时使用 `dp<rank>`，多副本应显式指定不同 replica。

### 写入、轮转与完整性

`submitBatch(line/make_line)` 只入队；`run()` 后台线程取任务、必要时构造 JSON。batch 在序列化及预算检查前过滤 INFO 未启用的任务，计入 dropped，不计 submitted_to_alog / bytes_submitted，不消耗预算；允许提交时调用 logBinaryMessage。队列任务只携带 JSON 生成函数，不再携带文件名。不用 ofstream 逐行写 JSONL，也不每条主动 flush。

batch 的路径、级别、布局、flush 和轮转只由 alog.conf 管理，录制器不创建 appender，不依赖 alog 内部 EventBase。默认使用纯消息布局，异步 flush 阈值 64 KiB、间隔 100 ms，`max_file_size=0` 禁止轮转。总提交预算只统计 batch 提交量。manifest 通过 ofstream 写临时文件再 rename，使用 batch_log_logger/batch_log_location 描述外部 batch sink，不再输出 files 和 max_file_bytes。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `RTP_LLM_RECORD_QUEUE_SIZE` | 4096 | 队列最多容纳的记录条数，不是字节数 |
| `RTP_LLM_RECORD_TOTAL_BYTES` | 1 GiB | 每个 owner 提交给 batch logger 的 JSONL 字节预算 |

队列满时丢整条记录，不截断 requests 数组；超总预算或 snapshot worker 出错时停止接受新记录。它不是零开销：生产线程仍有元数据拼接、锁、CPU 长度快照分配；后台有 JSON 构造和 alog 消息复制。录制不再进行长度 D2H 或等待 CUDA event。JSONL 是紧凑的一行一个对象，不做缩进美化；离线报告的格式化不在在线热路径。

`manifest()` 通过临时文件和 rename 更新，不追加。每提交 64 条给 alog 更新一次，`close()` 停止接收、drain 队列、join 后尝试 flush batch_schedule logger并更新状态，不调用全局 alog shutdown。运行中计数可能滞后。`errors` 是录制器可观察错误计数，不是模型推理失败次数。

新增 `storage_backend=alog`、`delivery_policy=best_effort`。`submitted_to_alog`、`bytes_submitted` 只计提交量；无法确认落盘量和 alog 内部丢弃量，因此 `written`、`bytes_written`、`sink_dropped` 为 null，`complete` 始终 false。`closed=true` 仅表示上层完成关闭流程。缺日志按缺失数据处理，不补造；完全未记录的请求也无法被分析器发现。

## 3. 请求时间嵌入 batch，不输出独立请求日志

`GenerateStream::recordSchedulerEnqueueTime` 在调度器锁内、流入队前创建 `RecordedRequest`，只保存不可变请求 ID。原有 scheduler_enqueue_time_us_ 继续服务调度指标；录制不再维护额外的入队/首次调度时间状态。

`recordBatchInputs()` 只输出请求组成和长度，不再读取时间快照。batch 保留 schema_version=1、owner_id、exec_id；exec_id 在 JSON 中输出十进制字符串以保留 int64 精度，内部计数及 TP 广播仍使用整数；请求保留 prompt/cache/q/KV 长度与 phase。request_id 缺失时省略，sequence_id=0 和 is_fake=false 省略。`ttft_us` 由输出更新在 stream 锁内填入，已知时才序列化，后续 decode 继续携带；不输出 first_token_produced。完整身份和固定的 model_role/length_source 由 manifest 保存。

`RecordedRequest` 不再保留 published/terminal 状态、输出计数、错误码或结束时间，也不再从 `moveToNext()` 采集终态。进程退出、调度前取消、被采集开关过滤等情况下，请求可能没有任何 batch 记录；只分析观察到的数据，不恢复请求总耗时或最终状态。

## 4. Batch 快照：在实际模型输入准备完后、forward 前

重点阅读 [NormalExecutor.cc](../rtp_llm/cpp/normal_engine/NormalExecutor.cc) 的 `process()` 和文件内辅助函数 `recordBatchInputs()`。

### 4.1 分配 ID 与过滤

录制不再维护 scheduler step 计数，也不在 TP shape hints 中广播 scheduler_step_id。

`process()` 的输入 owner 条件是：录制启用、非 warmup、非 propose、`tp_rank==0`。owner 为符合范围的本轮 forward 分配 `exec_id`。

`RTP_LLM_RECORD_DECODE` 默认关闭，启动时读取并缓存：

| 本轮组成 | 开关关闭 | 开关打开 |
| --- | --- | --- |
| 纯 prefill | 记录 | 记录 |
| 纯 decode | 跳过整轮 batch/execution | 记录 |
| prefill + decode | 跳过整轮 batch/execution | 记录完整混合 batch |

过滤发生在 CPU 长度快照分配和 batch JSON 构造之前。不会只从混合 batch 中摘出 prefill 伪装成完整输入。请求 ID 在入队时分配，过滤后的 batch 不再做请求计时操作。本期不再输出携带 exec_id 的 profiler scope。

### 4.2 TP 传播与快照

`gatherModelInput()` 后设置 `GptModelInputs.record_execution_id`。[ModelTypes.cc](../rtp_llm/cpp/models/ModelTypes.cc) 将它放入已有 shape hints 广播，所有 TP rank 得到同一 exec_id；没有为每个 rank 另造一份 batch ID。

TP 同步完成后，只有输入 owner 调用 `recordBatchInputs()`。它按模型 slot 顺序（decode 在前、prefill 在后）展开 stream 和 sequence，而不是保存一个 batch 平均长度。

| requests 字段 | prefill 来源 | decode 来源 |
| --- | --- | --- |
| `q_len` | `input_lengths[slot]` | 1 |
| `kv_len` | `prefix_lengths[prefill_index]` | `sequence_lengths[decode_index]` |
| `input_len` | `stream->inputLength()` | 同左，保留原 prompt 长度口径 |
| `reuse_len` | `stream->initialReuseLength()` | 同左，不是每轮累计 KV 长度 |

另外记录 phase、已知的 request_id、非零 sequence_id 和为 true 的 is_fake。`sequence_id` 直接取遍历单个 stream 时的索引 `i`（`0 <= i < currentBatchSize()`），表示该请求在本轮执行中展开出的序列位置；每个请求从 0 独立编号，0 在日志中省略。它不保证跨轮稳定，beam search 中同一索引跨轮可能对应不同的 beam。batch_slot 用整个 requests 数组的下标表示。一个多序列请求可占多个 slot，因此 requests 数组长度不一定等于独立请求数量。

`NormalModelInputGatherer` 在实际组装输入时保存可选的 `record_lengths` CPU 快照：prefill 取本轮 CPU input/prefix 长度，普通 decode 取本轮 CPU sequence 长度，device-state decode 取 `next_real_seq_len - 1`（q 固定为 1）。仅选中录制的 batch 分配该快照；输入 owner 将其值复制给 writer，在 manifest 标记 `length_source=cpu_input_snapshot`。不读取 GPU 长度、不持有 CUDA tensor、不创建或等待录制专用 CUDA event；后续 stream 更新不会改变已保存的长度。

P2P 首 token 初始化同时填充 GPU 长度和 CPU `last_real_seq_len` / `next_real_seq_len`。device-state decode 若缺失有效 `next_real_seq_len`（小于等于 0），在组装输入前返回 FailedPrecondition；无论是否开启录制都不静默回退到可能落后的 `seqLength()`。

快照数量与实际输入不一致时抛异常，由调用处捕获并记 recorder error；快照告警按 recorder 每分钟最多一次，TTFT slot 不匹配按 batch 去重计错；不会输出缺项的 batch。完整快照通过 `submitBatch(...)` 入队，最终写入配置的 batch_schedule.log。

**长度在 forward 前冻结，日志在输出更新完成后提交。** `recordBatchInputs()` 返回本轮 `RecordedBatch`，通过 `MergedOutput` 传给 dispatch；异步路径移交给 worker，不用 execution/request ID 查表。dispatcher 按输入 slot 定位结果区间，`GenerateStream::update()` 在原有请求锁内、输出裁剪完成后填入 `ttft_us`。TTFT 复用 `CompleteTokenIds::firstTokenLatencyUs()`，单位微秒，与 aux_info 同源，但不依赖 aux_info 开关或流式输出。多 sequence 沿用请求级口径。

上下文最后一个 owner 释放时只提交一次，writer 只读取冻结后的值。未产生首 token 的 prefill chunk、未到达输出更新的失败/取消槽，内存中保持未知 TTFT，日志省略 ttft_us；已知 TTFT 在后续 decode 中继续保留。正常异常展开也提交已有快照（进程直接退出仍可能丢失）；不缓存整个请求、不回写历史 chunk、不新增 GPU 同步。

## 5. 精简文件与序列化

当前不再写 `executions.jsonl`、`request_events.jsonl` 或 `requests.jsonl`，也不再为录制维护 eager/graph bucket 字段。原有 `executor_collector.model_forward_us` 计时及监控上报保持不变。

内部队列只包含 batch 任务，没有文件名参数或 `batches.jsonl` 路由标识，实际输出由 alog 的 batch_schedule logger 决定。录制统一使用 autil::legacy JsonMap / JsonArray 和 ToJsonString(..., true)，保持紧凑单行，不手工拼 JSON。

例如 TP=8 的一个被采集 batch：输入 owner 写一行 batch；两个请求的已知时间均在这一行 requests 数组中。其他 rank 不再为每轮 forward 写 executions。

## 6. Profiler 从本期录制中解耦

已移除 `recorder.snapshot`、`rtp.execution`、`rtp.graph_replay` 新增 scope，以及 `RTP_LLM_RECORD_PROFILE_START/STEPS` 自动窗口配置。`TorchProfiler` 不再依赖 ExecutionRecorder、不再写 captures.jsonl；原仓库 profiler 控制、打点和保存功能保持不变。

`exec_id` 仍用于 JSONL 内部关联，但不会自动进入 trace。本期不能自动关联 batch 与 GPU kernel，不应按 CPU/GPU 时间重叠猜测。未来如恢复此能力，应作为独立可选扩展。

默认使用固定的 logs/batch_schedule.log，输入 owner 的 batch 逐行追加，不按进程或 batch 添加文件名后缀；文件路径可在 alog.conf 修改。owner 目录仅保留 manifest.json，不是每个 batch 一组文件。

## 7. 离线分析：读哪些数据、如何降级

入口：[batch_trace_analyze.py](../rtp_llm/test/perf_test/batch_trace_analyze.py)。推荐阅读顺序：`main()` → `batch_request_timings()`（旧格式用 `request_latencies()`） → `correlate_trace()` → 汇总与报告输出。

- `main()` 读取一个 owner 的 manifest 和 batch 日志（旧格式兼容 request_events 分片）；新 batch 日志通过 `--batch-file` 指定并按 session/replica/DP 过滤，可多次传入轮转分片。`--batch-dir` 保留兼容旧格式目录；历史 trace 的 `--world-rank` 必须匹配 manifest。
- `request_latencies()` 按请求整理事件、检查缺失/重复/时钟等问题并计算延迟。best_effort 下以每个请求实际事件为准：完整请求可计算耗时，缺事件的请求为 partial；损坏 JSONL 行警告后跳过。旧版严格录制仍在 manifest 不完整时将所有请求耗时置 null。
- `correlate_trace()` 保留解析历史 trace：历史 execution scope → 同线程 CUDA launch → correlation → GPU kernel；多个 CPU runtime pid 合并的 trace 会被拒绝。它不为本期新录制提供自动 kernel 关联。
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
4. `replayBatch()` 复用 gatherModelInput → TP 同步 → model forward → CUDA 同步 → 释放缓冲区，不经过采样/在线请求推进，不新增 profiler scope。
5. 写各 rank 的同步耗时，最后写 complete/error 状态；已移除额外一次 profiler trace 采集。

每条 batch 快照独立回放，并不是让原请求按历史 decode 链连续生成。测量包含输入准备和同步，不等同于纯 kernel 耗时。

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
