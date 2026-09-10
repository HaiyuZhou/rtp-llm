# Cache grid：批内请求分布

通过现有 `--cache_grid_json` 传入下面的 JSON，其他模型和服务参数保持原有配置：

```json
{
  "cases": [
    {
      "case_id": 1,
      "batch_size": 4,
      "prefix_policy": "independent",
      "request_groups": [
        {"count": 2, "input_len": 8192, "cache_len": 4096},
        {"count": 1, "input_len": 8192, "cache_len": 0},
        {"count": 1, "input_len": 16384, "cache_len": 12288}
      ]
    }
  ]
}
```

`count` 之和必须等于正整数 `batch_size`。`input_len` 包含缓存前缀，
`cache_len` 必须小于输入长度、符合缓存块和 commit tail 对齐要求，并留出
`--cache_commit_tail_tokens` 个 token 供 seed 提交。上述示例适用 commit tail=4096。
各组独立校验；顶层 `input_len/cache_len` 在分组模式下仅保存各组最大值。

- `independent`（默认）：每个请求独立前缀，分别 seed。
- `shared_by_group`：同一组共享前缀，只 seed 一次；不同组前缀独立。

`case_id` 必须唯一。每个请求、每轮测量使用不同后缀，避免前一轮后缀缓存污染。
未填写 `request_groups` 时，原有 `input_len/cache_len` 会复制 `batch_size` 次；
原有不带分组的 batch=1 配置及结果结构保持兼容。

执行流程：先去重得到 S 条 seed，将服务端测试调度器设为 batch=S，确认设置成功后
并发发送全部 seed；S=0 时跳过。所有 seed 完成后，将调度器设为正式 batch=B，
B 个 worker 经 barrier 统一放行；等待全部请求完成，再开始下一轮。case 结束时
恢复 batch=1，以支持后续单请求 case、探针及 profile。
HTTP 每个请求槽使用独立 session，Dash-SC gRPC 也支持并发。
`--cache_measure_runs` 是整批重复次数。预生成文件支持分组配置，保存各请求的文本
和 token IDs，体积会大于旧的单请求压缩记录；旧存储仍可读取。

批量结果标记 `execution_mode="scheduler_fixed_batch"`，同时记录 `seed_batch_size`
和 `scheduler_batch_size`。执行模式、分组和共享策略进入 case key，断点续跑不会
复用旧客户端并发模式的结果。`runs` 中每项是一轮测量，包含：

- `requests`：每个请求的实际长度、缓存命中、TTFT、期望值和校验结果。
- `batch_wall_time_ms`：从 barrier 放行到最后一个请求返回的整批耗时，不含 seed。
- `completed_requests`、`total_input_tokens`、`new_prefill_tokens`：成功请求的统计。

任一请求失败或长度/缓存命中不符，整批不能记为有效；所有轮次有效时才输出
`median_batch_wall_time_ms`。逐请求 TTFT 不能当作整批耗时。

此模式要求独占的 `BatchDecodeScheduler` 测试服务及 DP=1；调度器等齐指定数量后
统一调度，不能用于混入其他流量的共享服务。CLI 会确保 `concurrency_limit`、
`max_context_batch_size` 不小于 B，并保证 `max_batch_tokens_size` 足以容纳完整 batch。
直接调用 runner 时需要调用方保证相同的服务配置。DP>1 的 CLI 配置会在启动前拒绝。
请求发送失败可能让固定 batch 等不齐，因此即使关闭 fail_fast，也会停止后续测量，
保存失败结果并恢复调度器，避免残留请求混入下一批。
可从引擎日志 `BatchDecodeScheduler::schedule: running_streams_.size()` 和
trace 的 `executor.model_forward(ctx_batch=...,gen_batch=...)` 验证实际组批。
现有只采集一次 forward 的 `--cache_profile_case_ids` 暂不接受分组 case。
