# Batch 与请求录制、Kernel 分析

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

## 当前边界

本提交提供录制和离线分析。native 形状回放、回放对比及其测试将在后续提交中加入。尚未完成真实模型、多卡和线上开销验收。
