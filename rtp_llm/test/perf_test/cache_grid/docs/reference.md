# Cache Grid 格式与执行参考

常用命令和产物索引见 [README](../README.md)。本页集中说明输入契约和执行细节。

## Grid 格式

最小单请求配置：

```json
{
  "schema_version": 2,
  "generator": {"cache_alignment": 4096},
  "cases": [
    {"case_id": 0, "batch_size": 1, "input_len": 8192, "cache_len": 0},
    {"case_id": 1, "batch_size": 1, "input_len": 8192, "cache_len": 4096}
  ]
}
```

- `cases` 必须非空。`case_id` 为非负整数、全文件唯一；重复几何也会被拒绝。
- 单请求的 `case_id/batch_size/input_len/cache_len` 均必填，`batch_size` 必须为 1。
- `0 <= cache_len < input_len`，cold 请求也必须显式写 `cache_len: 0`。
- 必填整数字段不接受数字字符串、浮点数、布尔值或自动补齐。
- 缓存对齐只在 `generator.cache_alignment` 配置，必须为正整数。
- 采样在独立生成器中完成，runner 只消费显式 cases。
- `kind/description/generator` 中的来源信息及 `summary` 用于记录和检查，不能代替实际 cases。
  手动修改 case 后应同步更新 summary。

输入长度包含缓存前缀。cache-hit case 应按实际复用块对齐，并留出
`cache_grid.commit_tail_tokens` 供 seed 提交，例如 `cache_len + 4096 <= input_len`。
输入采样对齐与缓存复用粒度是不同维度；kernel block 也不能直接当成可见复用粒度。
启动时会探测实际 reuse 粒度，失败则拒绝测量。
单请求 case 若落入相同物理缓存桶，runner 会保留一个代表；被去重的 ID 不能用于 retest/profile。

## Batch 调度

batch>1 必须显式写分组和前缀策略：

```json
{
  "schema_version": 2,
  "generator": {"cache_alignment": 4096},
  "cases": [{
    "case_id": 0,
    "batch_size": 4,
    "prefix_policy": "independent",
    "request_groups": [
      {"count": 2, "input_len": 8192, "cache_len": 4096},
      {"count": 1, "input_len": 8192, "cache_len": 0},
      {"count": 1, "input_len": 16384, "cache_len": 12288}
    ]
  }]
}
```

每组的 `count/input_len/cache_len` 都必填，count 为正，各组 count 之和等于 batch_size。
分组 case 不必填写顶层 input_len/cache_len；如果提供，必须与各组最大值一致。
分组形式也支持 batch=1。

| prefix_policy | seed 行为 |
|---|---|
| `independent` | 每个请求使用独立前缀 |
| `shared_by_group` | 同组共享前缀，每个共享前缀只 seed 一次；不同组独立 |

每个请求、每轮测量使用不同后缀，避免前一轮后缀缓存污染。cold 请求无需 seed。
流程为：将专用 BatchDecodeScheduler 设为 seed 数量 S → 并发 seed 并等待完成 →
切换为正式 batch B → barrier 放行 B 个请求 → 等整轮完成 → 下一轮。case 结束恢复 batch=1。
S=0 时跳过 seed。HTTP 每个槽使用独立 session，input_ids/gRPC 也支持并发。

该机制要求 DP=1、独占服务、prefill-only（partial=2、decode=1）。
实际组批由运行时调度器控制；`engine.max_context_batch_size` 和 `engine.concurrency_limit`
始终沿用 profile，启动器不会自动提高或强制设成 1。请按 workload 配置足够的准入并发。
请求失败可能导致 batch 等不齐，因此即使关闭 fail_fast，也会停止后续测量并恢复调度器。
可通过引擎 `BatchDecodeScheduler::schedule` 日志和 forward trace 验证实际组批。

### 连续压力模式

`--cache_pressure_mode=continuous`（统一入口为 `--pressure-mode continuous`）不再按
case 执行“seed → 测量 → 下一个 case”。runner 一次并发插入
`cache_pressure_seed_count` 条长度为 `max(cache_len)+commit_tail` 的独立长前缀，
然后打乱全部 case/run。每个命中请求选择一个 seed family，截断到目标 cache_len，
并在第一个未缓存 block 内加入唯一后缀，因此共享所需缓存状态但不会复用其他测量请求的尾部。
cold 请求使用独立 token ID。

压力发送器是有界 open-loop：任务队列只保存几何和计划时间，完整 token IDs 只在工作线程内构造，
内存随 `cache_pressure_client_concurrency` 而不是总请求数增长。request rate 为 0 时立即填满
客户端并发；正数时支持 `constant` 和 `poisson` 到达。

`cache_pressure_scheduler=batch_decode` 是默认模式。BatchDecodeScheduler 只在攒满
`cache_pressure_batch_size` 后执行，因此正式请求数不是整批时，runner 自动增加不计入 metric 的
padding 请求，防止尾批永久等待。padding 失败会使本次测试失败。

`cache_pressure_scheduler=fifo` 以 `USE_BATCH_DECODE_SCHEDULER=0` 启动服务，不调用运行时
scheduler 控制接口。FIFO 不等待固定 batch，也不增加 padding；空闲时首个请求可立即运行，执行中
到达的请求在下一个 prefill 边界按并发、token budget 和 KV 资源动态组批。
`cache_pressure_batch_size` 在 FIFO 下不适用，必须保持默认值 1。未显式配置 client concurrency 时，
FIFO 使用 `engine.concurrency_limit`。结果的 `pressure_scheduler` 和
`continuous_pressure.scheduler` 区分两种口径；FIFO 的 `scheduler_batch_size` 为 null。

固定批模式若要观察服务端排队，配置 `client_concurrency > batch_size`；两者相等只保证连续组批，
不保证 scheduler 内存在等待请求。FIFO 模式通过提高 client concurrency 或 request rate 制造积压。
`engine.concurrency_limit` 必须至少覆盖 client concurrency 和 seed count。该模式目前要求
DP=1、未分组 batch=1、`dashsc_input_ids`、明确 block size 和
完整 commit tail；不支持 fixed workspace、物化输入、profiling 或额外的 `cache_shared_seed`。

结果使用 `seed_mode=continuous_shared_prefix_v1`。每个 run 记录 seed family、全局发送序号、
计划/实际发送 offset、客户端 schedule lag；顶层 `continuous_pressure` 记录实际吞吐、padding、
schedule lag 以及来自服务端 `engine_wait_time` 的排队请求数和 wait time 聚合。

`--skip-reuse-validation` 允许测试保留实测 reuse 不匹配的结果，仍记录期望值及校验状态；
输入/输出形状、请求成功和有效时间仍严格检查，启动前缓存块探针也不跳过。
这些 reuse 不匹配的结果不会进入统一 pipeline 的正式图表或公式。

## 容量与随机采样

随机生成器按剩余总 token 预算采样，允许长短请求混合，不把预算均分给 batch。
设对齐后的最小长度为 m、原始输入总预算为 T，第 i 个请求上限为
`min(max_input_tokens, T - 已采样总量 - 剩余请求数 × m)`；生成后打乱槽位。
命中请求至少留出 commit tail；cold 概率默认 0.2，实际比例受预算影响。

可选 `generator.workspace_tokens` 对 packed token 总量施加固定约束：

```text
sum(input_len) <= max_batch_tokens
sum(align_up(input_len + 1, cache_alignment)) <= workspace_tokens
```

`+1` 为输出预留。正式 batch、并发 seed、缓存探针都需通过预算校验。
不使用 `batch × 最长请求` 矩形预算，也不绑定 CP8 或固定 1M 上限。
启用 CP 时，缓存块须是执行对齐 `2 × TP` 的倍数；CP 禁用时执行对齐为 1。

| 模式 | 启动容量行为 |
|---|---|
| 有 workspace | `max_seq_len=floor(workspace_tokens/cache_alignment)×cache_alignment`；`max_batch_tokens_size=min(workspace_tokens,max_batch_tokens)` |
| 无 workspace、batch>1 | 必要时提高 `max_batch_tokens_size`，容纳最大一批输入总量 |
| 所有模式 | 不改写 profile 的 `max_context_batch_size` 和 `concurrency_limit` |

超出固定预算时拒绝，不自动拆 batch 或扩容。`--kv-budget-tokens` 仅估算活跃请求和 seed 尾部，
不包含旧轮缓存、权重、模型专用 state pool、通信或 JIT；token 预算不保证总显存不 OOM。
旧 workspace policy preset 已移除，需重新生成 grid；预算改变后不能接续原 checkpoint。

## 分析契约

原始结果要求 `schema_version=2`、`mode=prefix_cache_grid` 和 `metrics` 数组。
单请求保存 `runs[]`；分组保存 `request_groups`、`runs[].requests[]` 和调度信息。
正常 metric 明确记录 batch_size、input_len、status、measure_runs、success_runs；
单请求还包含 cache_len_requested。执行异常 `status=error` 可以没有测量轮次。

正式观测要求所有轮次成功，每个请求 input/output 形状正确、output_len=1，
reuse 与请求 cache 精确一致，服务端耗时为有限正数。
时间字段 `prefill_time_ms` 来自服务端 `aux_info.first_token_cost_time`，单位 ms，含引擎等待，
不扣减 wait_time，不含输入分词及客户端通信。HTTP 与 input_ids/gRPC 使用同一口径。
缺少服务端数据不能用 `ttft_ms/client_wall_time_ms/batch_wall_time_ms` 补齐。

每轮取批内请求服务端耗时最大值，跨轮用 median（默认）、min 或 trimmed 聚合。
统一绘图和拟合直接读取原始结果；`report/observations.json` 是审计输出，不是手工准备的前置文件。
CSV 仅用于输出或独立导出，不作为拟合输入。

restricted-symbolic 从受限候选库选择公式，使用相对平方误差，需要 CPU PyTorch。
保留请求列表；`sum(computeTokens²)` 是逐请求平方后求和，不是总 compute 平方。
候选还包括 batchSize、maxComputeTokens 等批量特征。训练按 batch 等权、同 batch 内几何等权，
跳过线性相关候选项；同一分布及其排列、跨文件重复观测进入同一数据分区。
训练/验证/测试使用确定性的 70%/15%/15% 哈希桶，测试集不参与选式或系数重拟合。

每个 batch 必须有训练、验证、测试样本并通过误差门禁；不完整来源不能通过生产验收。
`--symbolic-max-terms`、`--symbolic-complexity-tolerance-pct`、hinge/exp 参数控制搜索。
不再提供 quadratic 算法、`--model-family`、`--objective` 或旧划分参数。
异常分析检查缓存/计算单调性和拟合残差；无法形成训练与验证集时，报告 residual_check 跳过原因。
导出遵循 FlexLB PrefillTimeFormula 语法；上线仍需目标 Java 运行时与适用范围验证。

多运行合并先检查模型、拓扑、精度、缓存和引擎配置一致性，容量差异保留在来源审计中。
每个来源必须有 profile 快照或内嵌 profile，不能猜测不同模型可混合。

```text
单卡输入 TPM = sum(input_len) × 60000 / RT(ms) / cards
单卡计算 TPM = sum(input_len - cache_len) × 60000 / RT(ms) / cards
```

卡数优先 `engine.world_size`，否则 `tp_size × dp_size × pp_size`（缺省维度为 1）；
EP/CP 不额外相乘，pipeline 可用 `--cards` 覆盖。这是服务端口径的 token 速率，不是客户端端到端吞吐。

## Nsight Systems

命令见 [README 的 trace 部分](../README.md#重测与-trace)。在模型服务所在容器及用户环境执行。
启动器自动设置 Bazel `--run_under`，通过 `nsys launch` 采集 CUDA/NVTX/OS runtime、
进程树 CPU 采样和上下文切换；会覆盖其他 run-under 设置，不需要额外改 `.bazelrc`。
`--dry-run` 可查看完整命令；`nsys status --environment` 可检查容器采样权限。

默认生成唯一 session；`--nsys-session` 可显式指定，不得复用活动 session。
安装版本必须支持 `launch/start/stop`，控制命令使用相同 nsys 可执行文件。
每轮顺序为 `seed → nsys start → target request → tail → nsys stop`。
不会同时调用 Kineto `/start_profile`，不启用服务 GEN_TIMELINE_SYNC 或逐层同步。
采集覆盖整个请求和 host 处理；`--nsys-tail-seconds` 默认 0.1、上限 60，
只是返回后的等待窗口，不是跨 rank GPU 同步屏障。

结果写入 `cache_perf_replays/profile_<id>/nsys/*.nsys-rep`；manifest 记录 session、
请求结果、采集范围、报告路径、控制命令输出和退出码。stop/export 完成后才执行下一轮 seed 或退出服务。
报告缺失/为空、控制命令失败都会使补采失败；目标请求异常时仍尝试 stop，并单独记录清理异常。
`--trace-timeout`（默认 180 秒）同时约束控制命令和导出；导出超时后先检查 session 状态。

首次采一轮，确认所有预期 GPU worker 都有 CUDA 活动和 CPU 调度信息；文件存在并不等于覆盖全部 rank。
PyTorch record_function 不自动转为 NVTX，kernel 名称和 launch correlation 仍可查看。
省略 backend 或选择 kineto 则输出各 rank Chrome JSON。补采数据只用于诊断。

## 快照与恢复

统一入口保存 v2 `cache_perf_launch.json`、v4 `test_info.json` 及 profile/grid 快照。
启动清单记录有效参数、环境、Bazel 配置和 SHA256；test_info 保存状态与引用，不重复整份配置。
恢复校验快照、grid/profile/run_config 指纹及测量参数，不读取后来改动的源 profile。
底层直接 Bazel 测试的 v3 test_info 是其当前元数据，不作为统一入口的隐式恢复配置。

## 底层共享 seed 与物化输入

底层 Bazel runner 的 `--materialize_cache_cases` 可预生成文本/token IDs，
`--cache_case_files` 使用对应物化目录；grid 指纹、几何及轮次必须匹配。
物化用例的 store_info/record 必须使用当前 schema，seed 明确写成文本或 marker/filler。

`--cache_shared_seed` 是另一种实验：只支持未分组 batch=1、dashsc_input_ids、明确缓存块，
不支持 HTTP prompt、物化输入、固定 workspace 或 profiler。它与 case 内 shared_by_group 不同。
先 seed `最大 cache_len + commit tail`，命中 case 按 cache 长度降序运行，cold 最后运行。
每轮仍使用独立后缀并精确校验 reuse，测量“共享长前缀已驻留”的请求。

结果标记 seed_mode=shared_prefix_v1，保存 shared_seed ID、耗时和刷新记录，不能混用独立 seed 的目录。
重启续测会重新 seed。实际命中不足时，用当前 cache_len+tail 刷新并以新后缀重试一次；
失效请求不进入正式统计，刷新耗时计入 case elapsed_s。重试仍失败则停止，不无限重试。

## 缓存压力快照的计数口径

`BlockTreeCache::pressureSnapshot()` 是按需调用的结构化诊断接口；普通健康检查不会调用它。
采样持有 cache mutex，只枚举固定物理 pool/group 拓扑，不扫描树节点或返回请求、cache key。
物理 pool 按指针去重，启动时冻结的 `pool_id` 区分同名 pool；group 保留 DEVICE/HOST/DISK 映射。
`total_blocks/used_blocks/free_blocks` 描述物理占用；`available_blocks` 还包括可回收的纯 CACHE 引用块。
各 reference category 可能重叠，不能相加计算 used；非 DEVICE pool 的 request 引用标记为不可用，不能解释为已测量的零。
`pending_release_blocks` 是已排入驱逐流程、尚未完成源释放的块数。
水位是 group 对应 tier 的配置，`watermark_trigger_count` 累加各 tier 从未触发到触发的转换，重复轮询不累加。

`published_host_blocks` 是当前仍有效、已经成功发布的 HOST 物理块去重数。
HOST 目标预留及尚未发布的副本不计入；有效 HOST 源在 LOAD_PENDING/LOADING 期间继续计入，
直到源所有权失效或释放；detached 数据立即移除。它不由引用类别相减推算，也不依赖事件发布或外部 metrics 开关。

transfer 记录按 group、物理 pool、operation、source/target tier 分类：
`copied_blocks/copied_bytes` 记录成功复制，包括整体失败中已成功的 descriptor；
`published_blocks/published_bytes` 仅在成功接受的 settlement 后累加，重复 store 的落败副本及 detached 目标不会增加 publication。
`failed_operations` 记录失败的逻辑复制操作。回收只减少当前 residency，不倒扣累计 copy/publication。
每个源/目标 pool 分别以自己的物理 block 和 `block_size_bytes` 为单位，
DEVICE 成员块与打包 HOST/DISK record 的块数和字节数不必相等；不要跨源/目标 pool 相加后当成单一传输量。
GPU→HOST 发布量应选择 `source_tier=DEVICE,target_tier=HOST,pool_tier=HOST` 的目标记录。
cache 层不提供 scheduler waiting/running、服务实例/rank、跨 rank 原子性或采样时间；上层必须提供或显式标记 unavailable。


## 可选压力诊断 RPC（schema_version=1）

显式调用 `RpcService.GetCachePressure`，请求应发往 TP rank 0；`target_ranks` 使用真实 world rank，
留空表示当前 TP group 的全部 rank。重复或拓扑之外的 rank 会返回 INVALID_ARGUMENT。
`GetCachePressureInternal` 只采本机，绝不递归 fan-out；旧服务或不支持的 engine 返回 UNIMPLEMENTED，
不能把缺少字段视作零占用。普通 CheckHealth/GetWorkerStatus/GetCacheStatus 不调用压力采样。

公共响应逐项保留 requested_ranks，并为每个目标返回一个 rank record。
不可达、超时、身份不匹配等失败保存在该 rank 的 error，其他 rank 的证据仍然可用。
peer 调用共享最长 1000 ms deadline，并取调用者 deadline 中更早者；总远端等待不按 rank 数累加。
连接复用既有 RPCPool 设施；诊断使用独立 fan-out，避免普通 BroadcastManager 的超时 fail-fast 策略丢失部分证据。
每个服务实例有独立 service_instance 和递增 sequence；重启后实例标识变化且 sequence 从 1 开始。
这不是跨 rank 原子快照，不能直接比较不同进程的单调时钟。

`sampled_monotonic_ns`、`collect_duration_ns`、`coordinator_send_ns/coordinator_receive_ns` 及
响应级 `coordinator_received_ns/coordinator_sent_ns` 均以纳秒表示。
单调时间来自各自进程 steady clock；每个 rank 的 send/receive 是 coordinator 的查询边界，
可计算往返时长并保留采样时钟的映射不确定范围，collect_duration 包括本机快照与协议转换成本。
字段有 presence：request_ref_blocks 缺席表示不可测，activity 缺席表示 scheduler counts 不可用。
`step_history_available=false` 表示当前没有 step 记录能力，steps 为空不等于没有执行。
`max_step_records` 上限 1024；`after_step_sequence` 按 world rank 分页，step session/control 的记录由后续执行步骤 recorder 填充。

wire enums 是稳定协议值并经显式转换：tier DEVICE/HOST/DISK/REMOTE/NONE=1/2/3/4/5，
group FULL/SWA/LINEAR=1/2/3，operation LOAD/EVICT/STORE=1/2/3；0 均表示 unspecified。
transfer 保留 group_id、pool_id、pool_tier、operation、source_tier、target_tier 和独立 copy/publication/failure 计数，
不汇总不同物理 endpoint。PressureStepPB 预留 session、step/sequence、wall/CPU 纳秒边界、context/decode counts/tokens、
request IDs、phase/completed。PressureSessionControlRequestPB 预留 ENABLE/MARK_PHASE/DISABLE、session/phase/capacity；
本版本尚不提供 session mutation RPC。

## FULL pressure diagnostic contracts

`runner.pressure_diagnostics` defines immutable settings, rank/pool/group samples,
and a pure FULL target selector. This module alone does not enable diagnostics or
change normal continuous execution. Supply explicit `fill_timeout_s`,
`fill_max_requests`, `fill_no_progress_s`, `fill_input_tokens`,
`gpu_tolerance_blocks`, `activity_timeout_s`, `drain_timeout_s`, and
`capture_timeout_s`; there are no experiment defaults for these required values.
A requested Nsight capture must also resolve an explicit positive
`nsys_duration_s` before execution.

Defaults: host residency ratio 0.25; sample period 1 s; maximum sample gap and
age 2 s; steady coverage 3 s with a 60 s timeout; measurement 60 s; minimum
coverage 0.95; longest invalid interval 2 s; Kineto 20 actual steps; profile
support 60 s. Budgets must be finite and positive, integer settings reject bools,
and GPU tolerance is a nonnegative integer no greater than each selected
pool's floored low threshold. These settings must freeze before sending work.

Each expected rank must contain FULL groups with mapped positive-capacity host
pools. The predicate uses any associated device pool's `used - pending_release`
against `floor(capacity * low_ratio) - gpu_tolerance_blocks`, published host
residency against `ceil(capacity * host_ratio)`, and strictly positive trigger
and successful DEVICE-to-HOST publication deltas against the before-seed epoch
baseline. EVICT and STORE publication use only the HOST destination endpoint;
source endpoint counters are never added again. SWA/LINEAR remain observations.

`decode_pressure_response(response, client_send_s, client_receive_s,
expected_ranks)` accepts the schema-1 protobuf and returns immutable evaluator
samples plus the complete serialized wire response. `raw_pressure_record` with
the response and the same client times returns a JSON-safe full protobuf record
for `cache_samples.jsonl` (protobuf int64 values retain their JSON strings).
It retains step, activity, optional-wrapper presence, coordinator timing, transfer
operation, and unused physical counters independently of evaluator projection.
Call `select_targets(samples, ranks)` initially, then explicitly compare each
later target tuple using `validate_inventory(current, frozen)`; topology/service
changes and decreased counters are discontinuities, never negative progress.

### Conservative diagnostic pressure evidence

Pressure coverage is a sampled estimate, not proof of pressure between polls.
`runner.pressure_diagnostics` provides interval union/intersection, `coverage`
(ratio and longest unknown/invalid duration, in seconds), and `confirmed_segment`.
Unknown time, including uncovered window edges, counts as invalid. Adjacent valid
points are merged on each server timeline before uncertain outer boundaries are
mapped to the client clock; every frozen FULL rank/group timeline is intersected.

Task 7 integration contract: construct `PressureCoverage(baselines, config,
max_segments=4096)` with complete pre-fill `RankSample` baselines. For each poll,
decode via `decode_pressure_response` and call `add(decoded.samples, now)` using an
injected client monotonic clock. Call `add((), now)` on failed polls, rather than
holding the preceding value. Save `raw_pressure_record` separately (including
`rank_offset_bounds_s`) for offline recomputation. The actual decoder populates
optional `RankSample.offset_bounds` and per-endpoint, per-unit cumulative counters;
hand-created local samples can omit bounds to use direct request timing.
Fan-out bounds compose client/coordinator request timing with each rank's internal
request timing. Bounds widen across a continuous segment for observed drift;
inconsistent bounds break adjacency. Samples older than the configured maximum,
unknown ages, excessive gaps, and missing ranks cannot add coverage.

Use `intervals(start, end)` for continuous steady qualification (a full 3-second
interval normally needs four 1Hz samples; uncertainty may require more).
`assess(start, end)` returns immutable `CoverageAssessment`: ratio,
`longest_invalid_s`, `valid`, clipped intervals, reason codes, `epoch_valid`, and
`max_segments_per_target`. Measurement and profile windows use this same API.
Continue same-distribution polling after t1 until a fresh adjacent sample closes
the right window edge, then clip to [t0,t1); never extend a last point to t1.
Service-instance, sequence, or cumulative endpoint counter reset terminates the
epoch (`epoch_valid=False`); the runner must fail/restart with new baselines.
Reasons are retained for auditing even if a later window has adequate coverage.
At most 4096 maximal segments per target are retained by default. Overflow drops
the oldest segment, reports `history_overflow`, and makes discarded time unknown;
choose a frozen sufficient bound for all windows and persist raw samples to disk.

Supply is a separate gate: `ActivitySample(last_dispatch_s, last_progress_s,
active_step_started_s, running, waiting)` uses client-clock seconds.
`assess_activity(sample, now, timeout_s)` returns validity and structured reasons;
`activity_valid` returns its boolean. Unknown queue/running counts fail closed.
Update progress from completions, allocation/publication deltas, and actual
execution-step evidence. A recent active long-prefill step qualifies until the
explicit activity timeout; queued work alone cannot mask lack of progress forever.

### Full-pressure lifecycle (internal opt-in API)

`CacheGridRunner._run_full_pressure(pending, config, limits, pressure_stub,
target_ranks, sink, controller=None)` is the diagnostic entry point. Existing
finite continuous invocations still use `_run_continuous`. Public CLI validation
must resolve the existing profile/server settings and pass frozen
`ExecutionLimits(model_tokens, admission_requests, batch_tokens,
available_kv_blocks, block_tokens)`. Include the one generated token in every
model/token/KV bound. The KV bound is the conservative simultaneous-request
budget after non-reclaimable reservations, not configured total capacity; use the
most restrictive real pool/rank bound. No workload-derived default is inferred.
The pressure stub is the generated `RpcServiceStub` connected to the actual
rank-zero endpoint; target ranks are real world ranks. Startup validation remains
responsible for exclusive DP=1 service, explicit opt-in and unsupported options.

The supervisor freezes configuration, targets and raw baselines before seed
requests. Seed and fill batches obey the immutable limits. Scheduler controls
receive the remaining stage/progress budget through
`set_batch(count, timeout=remaining_s)`; the finite runner retains its existing
default timeout when no override is supplied. All-cold grids skip seeds but still
fill. Fill uses independent unique inputs, completes each feasible
batch, then re-polls actual residency before entering steady. Request/time and
pressure-progress budgets terminate fill explicitly. Steady uses contiguous
sampled intervals, not the number of samples. Formal requests repeat a lazy,
deterministically shuffled grid through steady, measurement and optional support.
No boundary drains or repair fill are inserted. Inputs and pending requests are
bounded by concurrency; completion wakes the supervisor to replenish capacity.

`PressurePhases(config, started_s)` exposes `phase`, `measurement_start`,
`measurement_end`, `failure`, and `advance(now_s, fill_ready=...,
steady_covered_s=..., profile_pending=...)`. The actual RPC's optional
`before_dispatch()` hook runs after request construction, returns its final ID,
and records phase/identity at dispatch. Preparation that crosses the final cutoff
is rejected before sending. Arrival offsets remain cumulative despite lag.

`sink(event)` receives serialized bounded events: `frozen`, `phase`, `dispatch`,
`completion`, `pressure`, `pressure_lost`, `cancellation`, `error`, `finished`.
It must persist or aggregate incrementally and return promptly. Dispatch events
carry session/job/monotonic sequence, phase, request ID, actual token SHA256 and
length, scheduled/dispatch times, lag, and nonce/family/grid epoch/case metadata.
Completion events retain dispatch membership, actual completion time and a
measurement-boundary crossing flag. Cancellation events retain the dispatch
record and whether transport cancellation was accepted; this does not assert
that the server completed cancellation. Raw pressure records retain rank clock
bounds and full protobuf evidence. The final result contains failure, fixed
measurement boundaries, measurement coverage and separate support coverage.

An optional capture controller implements nonblocking `start(t0)`,
`pending(now) -> bool`, and `stop(reason, now)`. Measurement remains [t0,t1).
Pending capture receives same-distribution support until completion, the support
cap, or the earlier `t0 + capture_timeout_s`. Stop is called explicitly even on
failure. Configure the activity timeout for expected long prefill duration;
repeatedly observing an unchanged old active step does not reset its progress
budget. No profile collection is implemented by the supervisor itself.

Final fixed-batch padding counts actual formal and support dispatches. FIFO
never receives batch controls or padding. Drain continues pressure polling past
the window edge and is bounded by `drain_timeout_s`; live RPCs are cancelled at
the deadline and cancellation evidence is emitted. Dedicated executors shut down
without waiting indefinitely. Transport deadlines bound RPC worker lifetime;
Python cannot forcibly stop arbitrary token-generation code, so late preparation
is barred from dispatch. `CancellableRequestHandle(..., on_done=callback)` provides
the supervisor wakeup without exposing its executor or future.
# Incremental full-pressure diagnostic evidence

## Full-pressure CLI and resume

`run --pressure-mode continuous --pressure-preload` selects the full-pressure
baseline. Without preload, continuous execution remains finite. Continuous profile
and replay remain rejected until their capture lifecycle is integrated.

Use the existing JSON/JSONC profile's `cache_grid.pressure` object. Required fields
are `fill_timeout_s`, `fill_max_requests`, `fill_no_progress_s`, `fill_input_tokens`,
`gpu_tolerance_blocks`, `activity_timeout_s`, `drain_timeout_s`, and
`capture_timeout_s`. These are experiment choices with no invented defaults.
The corresponding CLI overrides are `--pressure-fill-timeout`,
`--pressure-fill-max-requests`, `--pressure-fill-no-progress`,
`--pressure-fill-input-tokens`, `--pressure-gpu-tolerance-blocks`,
`--pressure-activity-timeout`, `--pressure-drain-timeout`, and
`--pressure-capture-timeout`. Explicit CLI values, including zero GPU tolerance,
override profile values; missing, negative and nonfinite settings are rejected.

Optional profile fields use the shared PressureConfig defaults: `host_ratio=0.25`,
`sample_period_s=1`, `max_sample_gap_s=2`, `max_sample_age_s=2`, `steady_s=3`,
`steady_timeout_s=60`, `measure_s=60`, `min_coverage=0.95`, `max_invalid_s=2`,
`profile_steps=20`, `support_s=60`. `nsys_duration_s` has no duration default.
CLI aliases include `--pressure-host-target-ratio`, `--pressure-steady-seconds`,
`--pressure-measure-seconds`, `--pressure-profile-steps`, and
`--pressure-profile-seconds`; the latter freezes an Nsight duration but does not
enable continuous capture yet. Direct perf flags use the same names with
`cache_pressure_` and underscores. The resolved object is frozen in the profile
snapshot and resume configuration. An inherited legacy `measure_runs` value is
ignored; explicitly supplied `--runs`/`--cache_measure_runs` is rejected.

Startup requires a known positive physical/base `seq_size_per_block` in engine
arguments, owned exclusive service, DP=1, PP=1 and the complete TP rank set. The
existing-server shortcut is rejected. Model/admission limits come from actual
startup arguments; token limits use the engine's FIFO argument resolver and
`finalize_scheduler_config`, including the real default context-batch calculation.
Before probe or seeds, the real coordinator RPC at HTTP port + 1 must return
supported FULL groups on every target rank. The KV bound is the minimum live
`available_blocks` of their physical device pools: it already excludes active,
non-reclaimable reservations. It is not total capacity, total-minus-used, or an
additional subtraction of overlapping reference counters. The physical base block
span conservatively overestimates CP-sharded request allocations; a grid alignment
of 4096 with physical blocks of 512 retains a 512-token execution bound. SWA/LINEAR
remain observed without FULL-pressure gates.

Every public invocation writes `pressure_epochs/<unique-id>/startup_pressure.json`
and, after successful preflight, `pressure/` evidence within that epoch directory.
`pressure_latest.json` identifies the latest started diagnostic epoch. Root launch
snapshots and the empty exact-results checkpoint guard immutable resume inputs;
completed case rows never skip seed/fill/steady. Prior epoch files are never replaced.
The manifest includes actual profile/grid hashes, Python pressure-toolchain content
hashes (with their explicit scope), resolved startup/settings, generated-input
identity, service-instance epoch, clock bounds and capacity RPC control timestamps.
Raw session observations and actual dispatch token hashes remain authoritative.
No diagnostic request enters exact-fit metrics.

The internal full-pressure runner writes `pressure/manifest.json`, raw
`cache_samples.jsonl`, supervisor `steps.jsonl`, terminal `requests.jsonl`, and
an atomic `summary.json`. Files are flushed on each event and close. An interrupted
run retains evidence with manifest `complete=false`; complete means the summary
was written, not that execution or pressure passed. These diagnostic files have no
cache-grid schema envelope and never populate exact-reuse `metrics[]` or formula fits.

Measurement uses the dispatch cohort in `[t0,t1)`. A late completion belongs to that
cohort and increments crossing requests, but contributes no in-window throughput.
Throughput counts actual completions (including failures) per window second;
success throughput counts only successful completions. Terminal cancellations
count cohort membership without inventing completion timestamps. Transport cancel
acceptance is recorded independently and does not prove server completion. Support,
seed, fill and padding (including warmup padding) never enter formal window metrics.

Execution validity, measurement pressure validity, profile/support pressure validity,
reuse validity and trace validity are separate fields. Baseline trace validity is
null with `not_applicable`; capture validation remains pending until profiler
integration. Raw errors and raw request timing/reuse fields remain available for
offline exact analysis; online aggregation retains counters and outstanding jobs only.

Production callers must supply `evidence_manifest` to `_run_full_pressure` with
`code_hash`, `profile_hash`, `grid_hash`, `resolved_settings`, `input_identity`,
`epoch`, `clock_bounds`, and `control_times` projected from their resolved inputs.
Missing fields are null and listed in `missing_metadata`, never invented hashes or
passing completeness flags. Frozen session/config/limits/target inventory and raw
baseline are recorded from the actual supervisor; raw pressure records retain clock
bounds. The public CLI/config layer owns that metadata projection and opt-in checks.

### Execution-step server contract

`RpcService.ControlPressureDiagnostics` fans out ENABLE, MARK_PHASE, and DISABLE
to the existing TP worker endpoints using `ControlPressureDiagnosticsInternal`.
Only TP rank zero accepts the public method. Empty target lists select the whole
TP group; duplicates and ranks outside that group are rejected. Each rank has its
own `service_instance` and reports its actual engine-thread
`applied_monotonic_ns`; these timestamps are not a simultaneous barrier. Peer
errors remain in the response alongside successful ranks. The common RPC budget
is at most one second. Expired queued controls are not applied later.
While peer acknowledgements are pending, the coordinator wakes the root scheduler
at bounded 10 ms intervals so the existing idle TP broadcast releases peers to
apply their queued controls. This also applies to peer-only targets and delayed
peer enqueue. Root session state is not changed by these wakeups. Wakeups stop on
completion, cancellation, or the common deadline; cancellation still drains every
outstanding RPC completion. No extra per-forward collective is introduced.

Session IDs must be nonempty (at most 256 bytes). ENABLE defaults to 4096 retained
completed forwards; `recorder_capacity` accepts 1 through 65536 (zero selects the
current/default capacity). Repeated ENABLE with the same session and capacity is
idempotent; a conflicting active session/capacity is rejected before mutation.
MARK_PHASE requires the matching active session and a nonempty phase of at most
256 bytes. DISABLE requires the matching session and is idempotent. Completed
history remains readable after DISABLE until the next ENABLE resets it. A new
session resets both root step IDs and each rank's completed sequence. Speculative
executors return unsupported rather than empty success. Controls wake idle FIFO
and batch-decode scheduling and are applied at engine boundaries, including pause.

`GetCachePressure` returns step data only for its matching `session_id`.
`step_history_available=false` means that no matching history exists. Completed
records use an exclusive per-rank `after_step_sequence` cursor and are copied into
pages capped at 1024 records; RPC `max_step_records=0` selects that cap.
`step_records_lost=true` means that the requested next completed record was
overwritten. Consumers must not bridge that gap. The separately stored active
forward is projected as an additional `steps` entry with `completed=false` and
`sequence=0`, regardless of cursor/page fullness. It never advances
`next_step_sequence`. A matching idle sample has `running_requests=0`; during a
forward this activity field is the synchronized context plus decode batch count.
Waiting-request activity is currently unknown (wrapper absent).

Only actual nonempty model forwards consume a step ID. Rank zero assigns a
candidate after input gathering; existing TP shape hints carry its ID, context
and decode batch counts, and executed token count. Empty TP loops and skipped
forwards do not count. All enabled ranks record the root identity and metadata,
including decode batches (which a prefill-only consumer must flag). The numeric
`request_ids` are internal stream IDs. Rank zero additionally records exact client
strings in `source_request_ids`, from authoritative dispatched StreamGroups and
`GenerateStream::traceId()` (numeric fallback only if no trace exists). The Dash
input-ID route preserves `ModelInferRequest.id` as that trace ID. Non-root ranks
do not invent membership from their empty local stream lists; clients join their
steps to root membership by session and step ID within compatible per-rank
service-instance epochs, never by completion order.

`preparation_begin_ns` is captured at executor entry. `forward_enter_ns` and
`forward_return_ns` use `CLOCK_MONOTONIC`; `cpu_start_ns` and `cpu_end_ns` use
`CLOCK_THREAD_CPUTIME_ID` on that same executor thread at the corresponding
forward boundaries. Legacy `wall_start_ns`/`wall_end_ns` alias those forward
boundaries. `end_ns` is the forward return boundary, not CUDA completion or output
delivery. No diagnostic CUDA synchronization is added. `executed_tokens` is the
root execution-token total; normal decode contributes one token per decode batch
row and the remainder is `context_tokens`. Active copies are immutable. A thrown
forward clears active state without publishing a successful completed record.

### Client step analysis and controls

The full-pressure runner enables its unique session on every target rank before
the baseline or seeds, with an explicit 4096-record capacity. Controls run on a
dedicated one-worker executor (at most 16 pending markers), separate from sender
and pressure-poll workers. Each RPC has a one-second deadline. Schema, rank set,
positive applied timestamp and stable per-rank service instance are checked;
partial enable is a failure and still attempts DISABLE for the owned session.
Phase markers are asynchronous observations, not membership barriers: actual
dispatch identities determine request phases. `control` events and manifest
`control_times.diagnostics` retain full responses, client send/receive times,
rank application times and validation errors, including partial acknowledgements.

Cleanup attempts DISABLE after request drain, then drains completed pages with
a two-second / 17-page bound (256 completed records per page). Active sequence
zero remains independently visible. Control and final-drain failures are recorded
before `finished` and make execution validity false. Raw response pages are
preserved even when rank or clock validation fails. `cache_samples.jsonl` contains
the raw pages and `steps.jsonl` also projects each untouched wire step as a
`step_record` event with `world_rank`, `service_instance`, `record` and
`offset_bounds_s`; these writes precede filtering. There is no claim that an
incomplete final drain contains all server records.

Temporary missing ranks and rank RPC errors are retained as `sampling_error` pages
and contribute unknown coverage without immediately ending measurement. Such pages
advance no completed-record cursor. Instance, session, history, schema, rank
protocol and clock corruption remain fatal `diagnostics_error` pages. Every fatal
page emits execution-error evidence, including outstanding polls consumed during
cleanup after an earlier profile failure. Final draining retries incomplete rank
pages within its existing bounds; an incomplete page cannot certify an empty,
complete history. `PressurePoll.step_history_complete=false` marks these pages.
Before treating a page as recoverable loss, all available concrete payloads pass
the shared schema decoder. An RPC error permits absent payload, but cannot hide
explicit foreign-session steps or malformed available data. Validation uses a
protobuf copy of available payloads only; raw evidence, frozen targets and cursor
eligibility continue to refer to the original full response.

`summary.json.baseline_steps` reports per-rank eligibility and exclusion counts,
forward wall/CPU durations, preparation wall durations, and adjacent forward
gap wall/CPU distributions. Membership uses root `source_request_ids` matched to
actual dispatch events, then session/root step ID and synchronized counts/tokens
to join peers. Numeric stream IDs and completion order are never substituted.
Each rank keeps its own instance identity and CPU clock. Membership, root-step
associations and pending steps are bounded to 4096 entries each (pending per rank);
evictions and cursor losses are reported. Missing or evicted membership is unknown,
and cannot become eligible. Loss breaks adjacency, including delayed pending pages.

Only complete, pure formal measurement prefill steps with known membership fit
the window. Exclusions distinguish support, mixed phases, padding, decode,
incomplete, unknown membership and uncertain boundaries. For client-minus-server
offset bounds `[low, high]`, client `[t0,t1)` maps conservatively to server bounds
`[ceil(t0-low), floor(t1-high)]`; the entire preparation-to-return interval must
fit. Gap subtraction uses the original server timestamps on the same executor
thread and requires consecutive eligible step IDs with no intervening loss.
Background-thread wall durations are never added to executor wall time.

Means and maxima use exact streaming counters. Quantiles use a deterministic
Algorithm R reservoir (seed 0, capacity 4096) and nearest-rank estimates. Each
distribution reports count, reservoir size/capacity and algorithm. p99 is null
with `p99_status=insufficient` below 1000 eligible intervals; above that threshold
it is `estimated`. This threshold is a reporting heuristic, not a confidence
guarantee. Reservoir sampling has sampling error and supplies no tail-error bound;
20-step traces should be interpreted structurally, not as reliable tail estimates.

### Thread-affine Kineto cancellation

`StopProfile` accepts `trace_name`, `enable_all_rank`, and an optional
`target_ranks` TP subset (the subset requires `enable_all_rank=true`). All-rank
requests go to TP rank zero; `StopProfileInternal` only acts locally. Both RPCs
are forwarded by `LocalRpcServiceImpl`. Use a distinct session-and-window trace
identity within the existing 64-character `[A-Za-z0-9_-]` sanitized name limit;
stop uses the exact configured identity, without sanitizing a different request
into a match. Start's default three steps, maximum 1000 steps, and nonnegative
start-step sanitization are unchanged.

Each stop rank response contains `world_rank`, its independent `service_instance`,
`accepted`, `profile`, and `error`. `accepted` means only that the matching
cancellation request was accepted. A duplicate stop of the last finished trace
is idempotent and cannot cancel a newer trace. A mismatched trace is rejected.
A stopped window cannot be resurrected by a duplicate start. A different window
can start after the prior collection/export has settled.

`GetCachePressure.ranks[].profile` adds `supported`, `trace_name`, `state`,
`stop_requested`, `collection_complete`, `export_complete`, `collection_error`,
`export_error`, and `file_name`; ordinary worker status is unchanged. States are
`armed`, `active`, `stopping`, and `finished`. `stopping` includes collection and
pending asynchronous export. `finished` is terminal, including failures; inspect
both error fields. Disarming before start finishes without creating a file.
`collection_complete` means collection has returned, including a collector failure;
it does not imply export. `export_complete` means the save returned and basic
output-file checks passed, or no collector ever started. These checks are not a
JSON parser or trace-validity verdict: the capture client must still parse the
artifact and validate activities, step coverage, identities and completion tails.
Failed/partial output files remain available for investigation.

Stop RPCs observe owner-thread collection for at most one second, bounded by
the incoming deadline. Partial failures are returned per rank. A timeout after
acceptance retains `accepted=true`, incomplete status and an explicit error when
the response can be delivered. A transport deadline can hide an acceptance;
poll status and retry the same identity within the caller's cleanup budget.
TP root continues bounded idle progress until internal owner-thread observations
return, including delayed and peer-only cancellation. Request acceptance alone
does not end that progress loop. No additional per-forward collective is added.

The engine services cancellation at idle, pause, safe execution completion and
loop shutdown boundaries without advancing profiling steps. Kineto collection
and device drain stay on the thread that started collection. RPC handlers and
wrong-thread destructors never disable Kineto. A stuck model call or collector
cannot be force-stopped from RPC: status remains incomplete, the external caller
must bound cleanup and apply existing service teardown policy. Async saver joins
and Kineto teardown themselves are not forcibly interrupted by these RPC bounds.
