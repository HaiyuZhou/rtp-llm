# Batch 与请求录制、Kernel 分析和形状回放

本功能默认关闭。当前代码接入普通 `NormalExecutor`，逐请求生命周期由请求 owner 输出，batch 长度来自实际模型输入。完整设计见 [设计文档](../batch_execution_record_replay_design.md)。

## 开启录制

在启动各个引擎进程前设置相同的会话 ID，各进程使用独立输出子目录：

```bash
export RTP_LLM_RECORD_DIR=/tmp/rtp-recordings
export RTP_LLM_RECORD_SESSION=experiment-001
export RTP_LLM_RECORD_REPLICA=replica-001
export RTP_LLM_RECORD_DECODE=0
export RTP_LLM_RECORD_QUEUE_SIZE=4096
export RTP_LLM_RECORD_FILE_BYTES=104857600
export RTP_LLM_RECORD_TOTAL_BYTES=1073741824
```

`RTP_LLM_RECORD_DIR` 和 `RTP_LLM_RECORD_SESSION` 必须同时设置；启动后不支持动态修改。会话 ID 应在每次新实验时更换。数值参数要求为正整数，无效值回退默认值。队列单位为记录条数，字节预算按 owner 独立计算，仅覆盖录制 JSONL，不包含 profiler trace。

文件轮转保留旧数据，不删除旧分片。达到总字节预算或 writer 出错时停止接收新记录；过载整条丢弃，不截断 batch。manifest 中记录生成、写入、丢弃、错误及字节计数。正常进程退出时 drain 队列并完成 manifest；强制终止时文件可能不完整。服务运行时 manifest 定期更新，`closed=false`，分析器保守地标为 partial。

输出示例：

```text
/tmp/rtp-recordings/owner-<pid>-<timestamp>/
  manifest.json
  request_events.jsonl
  batches.jsonl
  executions.jsonl
  captures.jsonl
  request_events.jsonl.1
```

请求事件为 enqueue、first_scheduled、first_token、finish/cancel/error。仅成功进入引擎队列的真实请求进入事件流。batch 文件每行包含完整 requests 数组，q_tokens 为本轮计算长度，kv_tokens_before 为本轮之前的逻辑 KV 长度。kernel 仅在 profiler 窗口采集。

`RTP_LLM_RECORD_DECODE` 默认关闭，只有启动前设置为 `1` 才采集 decode。关闭时跳过所有包含 decode 的 batch（包括混合 prefill/decode batch）的快照和 execution 日志，在 D2H、event 分配和 JSON 构造之前过滤，不输出可能误导回放的残缺 batch。纯 prefill batch 和请求生命周期事件仍记录；first_scheduled 若落在被过滤的 batch 中，其 execution_id 为 null。manifest 的 engine.record_decode 和 decode_filter_policy 标明此采集范围；complete 仅表示范围内无丢失，不代表包含完整 decode 流量。

需要完整流量形状回放时设置 `export RTP_LLM_RECORD_DECODE=1`，各 TP rank 使用相同配置。此开关不关闭独立的 profiler 窗口，窗口内仍可能看到无 batch 关联的 decode kernel。

同一请求的多轮 prefill/decode 使用同一个 request_id，结合 session_id、replica_id、dp_rank 关联；每轮 forward 的 execution_id 不同，按 scheduler_step_id 排序即可追踪执行过程。sequence_id 区分同一请求的多序列，batch_slot 只是该轮 batch 的位置，不能用于跨轮关联。

request_id 为会话内匿名 ID，不输出 token 内容。first_token 为引擎输出队列发布边界，非客户端接收时刻；非流式输出按单独口径统计。first_token 的 execution_id 当前为 null，避免异步输出错误关联到后一轮执行；仍可通过 request_id 查询其全部 batch。

## 采集 Kernel 窗口

同一服务副本的各 TP rank 使用相同 `RTP_LLM_RECORD_REPLICA`；不同副本使用不同值。未设置时默认按 DP rank 命名，仅适合单副本录制。

使用已有 timeline 控制接口，或在启动前增加：

```bash
export RTP_LLM_RECORD_PROFILE_START=5
export RTP_LLM_RECORD_PROFILE_STEPS=3
```

各 TP rank 需要使用相同配置。此配置复用现有 StepWindowProfiler；trace 输出目录沿用引擎 profiler 配置。`captures.jsonl` 记录启动、收集状态和 trace 文件位置；collected 不等于已成功落盘，需要检查实际 trace 文件。

每个 forward 具有 `rtp.execution(id=...)` 范围，graph replay 具有 bucket 标记。离线解析通过 CPU launch correlation 连接 GPU kernel，不能仅以 GPU 时间落在 CPU 范围内作为归属依据。

## 离线分析

```bash
python -m rtp_llm.test.perf_test.batch_trace_analyze \
  --record-dir /tmp/rtp-recordings/owner-<pid>-<timestamp> \
  --trace /path/to/record_ts123456789_wr0_1.json \
  --world-rank 0 \
  --output /tmp/record-report
```

不传 trace 可仅计算请求延迟。每次指定一个 owner 和同 rank 的 Kineto trace；不接受合并多个 CPU 进程后 correlation ID 冲突的 trace。输出 report.json、report.html、kernel_events.jsonl，包含逐请求长度、延迟、kernel 汇总及按执行分组的 duration 总和、区间并集和跨度。无法关联的 kernel 保留 unmatched；这些统计不代表单请求独占 GPU 时间。

分析 TP 非输入 owner 的 trace 时，增加 `--batch-dir /path/to/input-owner` 指向同会话、同副本、同 DP 的 batch 目录。报告中的 `missing_batch_execution_ids` 表示 trace 执行缺少对应 batch，不能将此类执行视为已完整关联。

逐请求输出总耗时、首次排队、首 token 延迟和 `post_first_schedule_latency_ns`（首次调度到终态）。最后一项包含后续等待和输出开销，不是单请求独占 GPU 时间；缺少事件时相应耗时为 null。

## 执行形状回放

native 回放当前要求 DP=1，可使用 TP 多卡。多 DP 各组不同流量的联合回放尚未接入，因此明确拒绝 DP>1，避免把一个 DP 的快照复制给全部 DP 后误报为真实分布回放。

回放必须使用独立引擎进程和相同模型/拓扑配置，不能复用在线服务。支持单个 FP16/BF16 full-KV group 的普通融合文本模型，包括普通 prefill、decode 和混合 batch；MTP、多模态、特殊/量化 KV、PD 分离明确拒绝。每条快照独立执行，KV 以零值初始化，token ID 确定性生成；不复现内容、专家路由和 cache 共享。

先生成计划，不启动模型：

```bash
python -m rtp_llm.test.perf_test.batch_replay \
  --batches /path/to/batches.jsonl \
  --execution-ids 18231 \
  --warmup 5 --repeat 20 --output /tmp/replay-plan
```

指定引擎命令可自动启动独立进程并等待每个 rank 完成：

```bash
python -m rtp_llm.test.perf_test.batch_replay \
  --batches /path/to/batches.jsonl \
  --output /tmp/replay-run --world-size 1 \
  -- python -m rtp_llm.start_server <模型及引擎参数>
```

输出目录必须是新目录，防止读取历史完成标记。launcher 设置 `RTP_LLM_REPLAY_PLAN`，引擎直接运行计划并拒绝在线请求；完成后 launcher 终止自己创建的进程组。多机启动需由现有分布式启动器分发相同计划并收集结果，当前 launcher 只管理本地子进程。

结果包含每轮无 profiler 的同步执行耗时（包含输入准备），以及独立采集的一次 forward trace；不能将此基线直接与纯 kernel 耗时等同。可以给分析命令增加 `--replay-dir /tmp/replay-run`，按执行 ID、rank、kernel 名称输出调用次数、平均耗时和变化率。

## 当前验证边界与待完善项

- 已通过 CPU recorder 测试、7 项离线工具单测和 mock 引擎录制/回放测试；mock 测试覆盖同步与 `RTP_LLM_STREAM_ASYNC=1`、`RTP_LLM_DROP_BROAD_SYNC=1` 模式。测试 JSONL 已用于分析和计划生成的命令行检查。
- 小矩阵验证已确认本机 PyTorch 2.11/CUDA 13 的 eager 和 graph replay 可关联；这不替代实际模型和多卡覆盖验收。
- 当前 manifest 提供模型/拓扑/cache 配置，报告比较配置差异；完整权重/软件指纹尚待补齐，不能把 config_match 解释为相同权重。普通引擎的 scheduler_step_id 随输入传播；离线回放没有在线调度 step。
- native 回放恢复 q/KV 配对和原始 prompt 长度元数据，真实内容和 MoE 路由不受控。graph bucket 由同配置的实际执行路径选择，报告标记 bucket 差异，尚未强制复现原 bucket。
- 尚未完成线上吞吐/TTFT/TPOT 开销门槛、多卡完整性、动态启停、有时限 shutdown drain、按事件类型 metrics 与特殊 KV 回放验收。当前实现不可宣称已达到设计文档的全部生产验收项。

测试命令：

```bash
python -m unittest rtp_llm.test.perf_test.batch_recording_test
bazelisk test //rtp_llm/cpp/observability:execution_recorder_test --config=cuda13
bazelisk test //rtp_llm/cpp/normal_engine/test:recording_engine_test --config=cuda13
CUDA_VISIBLE_DEVICES=<空闲GPU> python -m rtp_llm.test.perf_test.batch_trace_gpu_test --output /tmp/new-gpu-check
```
