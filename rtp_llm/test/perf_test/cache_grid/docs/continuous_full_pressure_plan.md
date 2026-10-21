# Cache Perf 连续 Prefill 满载压力与跨 Step Overhead 方案

状态：已实现，并于 2026-10-21 完成 DeepSeek-V4-Flash 真实 GPU 验收。FIFO mixed 的独立 baseline、Kineto 20 steps、Nsight 10 秒，以及独立容量配置的 fixed batch2/all-cold 均通过；取消实验按预期报告 drain_timeout，不作为成功性能样本。

验收证据位于 `/data0/zhouhaiyu.zhy/tmp/continuous-full-pressure-20261020-flash/`，以 `acceptance-matrix.json` 和 `acceptance-final-report.md` 为准。四个成功运行的 60 秒 measurement coverage 均为 1.0、longest invalid 为 0；阈值、水位、25% host residency 与 tolerance2 未降低。固定批变体显式扩大 FULL/SWA/host 容量，主 baseline/Kineto/Nsight 配置保持一致。所有失败尝试均保留，包括一次未归因的 Nsight rank2 native 崩溃；后续成功不代表该崩溃已修复。

Task15 运行修复提交依次为 `94f336727a`、`f1f01fc2a8`、`f62615cbf1`、`f7290fc99c`、`eeb14fa433`；各运行保留实际 source/native/profile/grid 身份。详细行为及限制见 [reference](reference.md#validated-runtime-acceptance)。

已确认的首版口径：以 FULL attention 的 cache group 为满载判定基准；本文列出的默认阈值用于首版实现，允许运行前显式配置并冻结，不在运行中为通过验收调整。

## 1. 目标与实现基线

在现有 `tools/cache_perf` 连续压力模式上，先通过真实 prefill 请求让 GPU cache 达到配置水位，并将数据降级至 memory cache，再保持请求流不断，采集连续多个 prefill step，分析相邻 step 之间的 CPU 开销及实际 GPU 执行停顿。

实现基线：

- `7a1e203e3e807a55afabf6d40128b397c2c89fe1`：支持 continuous pressure，包含多 seed family、有界客户端并发、连续发送、到达模型、固定 batch 及尾批 padding。
- `a4f2a52d60d83297781c92dccce84d0404de2637`：支持 FIFO pressure，使用真实 FIFO scheduler 动态组批，不等待固定 batch，不添加 padding。

本方案使用真实服务和模型执行，不使用 C++ BlockTreeCache benchmark 的固定 C32、32,768-block pool 或 100ms sleep，也不修改其 `OnlineTreeScheduler`。旧的 [C++ benchmark 方案](../../../../cpp/cache/block_tree_cache/benchmark/docs/continuous_prefill_pressure_plan.md) 不作为本任务的实施依据。

本方案中的 step 是一次实际非空 batch forward，不是一个客户端请求。每个请求只生成一个 token，保持 prefill-only；持续请求自然形成多个 prefill step，无需引入 decode 或 chunked prefill。

## 2. 已有能力与增量

| 方面 | 当前实现 | 本方案新增 |
| --- | --- | --- |
| 入口与环境 | cache_perf profile/grid 配置、启动、结果目录和快照 | 新阶段及 profile 配置纳入冻结配置 |
| 请求构造 | seed family、目标前缀、独立 suffix、cold 输入 | 与正式 seed 独立的容量灌注请求 |
| 压力发送 | 客户端并发上限、burst/constant/poisson | 连续阶段切换、可选时间窗口和有界持续供给 |
| 调度 | batch_decode 固定 batch；FIFO 动态组批 | 复用现有调度，不另建模拟 scheduler |
| 初始 seed | 长度为 max(cache_len)+commit_tail；验证请求结果 | 读取实际容量压力，达标后才正式测量 |
| 观测 | 请求时延、reuse、吞吐、schedule lag、engine wait | 分层 cache 压力、成功降级、step 时间和 CPU/等待归因 |
| Profiling | 单 case replay；continuous 显式禁止 profile | 连续流上的 Kineto 或 Nsight 窗口 |
| 验收 | 精确 reuse 的 cache-grid 测量契约 | 独立的满载诊断契约，不污染原公式拟合 |

主要复用文件：

- [cache_perf.py](../runner/cache_perf.py)：CLI、冻结配置、profile 后端和启动包装。
- [cache_grid_runner.py](../runner/cache_grid_runner.py)：seed、连续发送、结果和 profile 控制。
- [batch_decode_test.py](../../batch_decode_test.py)：服务启动与调度器选择。
- [perf_config.py](../../perf_config.py)：底层参数。
- [reference.md](reference.md)：已有 continuous 行为及结果契约。

## 3. 运行约束与测试口径

初版沿用 DP=1、独占服务、未分组 batch_size=1 grid、dashsc_input_ids、已知 block size、完整 commit tail，以及不与 fixed workspace、物化输入和 cache_shared_seed 混用的限制。这里 grid 的 batch_size=1 描述单请求几何，不限制服务端 batch 大小。

真实调度分析优先选择 `--pressure-scheduler fifo`；保留 batch_decode 作为固定 batch 对照。client concurrency、引擎 concurrency_limit、context batch 限制和 token budget 均来自实际配置，不自动提高容量以使测试通过。

所谓“满载”包括两个分别检查的条件：

1. 容量压力：所有目标 TP rank 的 FULL group 已达到有效 GPU cache 水位，对应 host 有已完成降级并发布的缓存数据。
2. 请求压力：持续存在可推进的请求流，能够反复形成 prefill batch；不存在长时间的客户端断供。

GPU cache 在 watermark 附近运行是正常行为，不要求物理 blocks 永远 100% 占用，也不要求整张卡物理显存耗尽。基于真实 cache 的容量、层级和低/高水位确定门槛，不沿用 C++ benchmark 的 80% 常量。

首版结果明确标为“FULL attention 压力达标”，不代表所有 cache group 都满载。SWA/LINEAR 保留观测，不参与满载硬门槛；没有 FULL group 的模型返回不支持该判据，不自动切换标准。

## 4. 完整流程

```text
启动服务与配置检查
    -> 初始化正式 seed family
    -> 容量灌注 pressure_fill
    -> 正式 workload 稳定运行 steady
    -> 满载门槛通过
    -> 连续 measurement 或 profile 窗口
    -> 停止新增请求、完成在途请求、收集结果和 trace
```

seed 和容量灌注的请求不计入正式请求性能指标。steady、measurement 和 profile 共享同一个持续运行的发送器，阶段切换只改变统计标签与采集状态，不重启服务、不清缓存，也不等待所有请求完成后再重新发压。

### 4.1 初始化与 seed

启动时确认启用了目标 cache backend、FULL group 的 GPU/host tier 和对应容量配置；没有 FULL group、缺少目标 host tier 或不可读取必要压力信号时，返回明确的配置/能力错误。冻结目标 rank/group/pool 清单、共享 pool 映射、水位、容差和 host 驻留目标，已知门槛不可达时提前返回配置错误。

复用 `_initialize_continuous_seeds()` 建立正式请求所需的共享前缀。该函数当前只验证 seed 请求成功、输入输出形状和零复用，不代表容量已满。全 cold grid 即使跳过 seed，也必须执行容量灌注。

保留 block-size 探针和输入预算检查。pressure-fill 并发也不能超过引擎准入和 token/KV 资源约束，不能一次把超过可执行上限的所有长请求同时送入固定 batch。

### 4.2 容量灌注

使用与正式 seed 分离的唯一前缀和 suffix，生成具有足够新 KV 数据的真实 prefill 请求。采用固定、可记录的请求长度与随机种子；请求长度需满足模型、token budget 和 commit tail 限制。

复用现有发送器和传输协议，分批持续提交，完成请求后由引擎自然发布 cache 并释放请求引用，通过真实淘汰策略触发 GPU 到 host 的降级。禁止直接修改 pool、pin 数据或手工驱动 eviction 来伪造压力。

成功完成请求数、输入 tokens 和估算新增 blocks 用于解释灌注进展；结束判据必须来自实际 cache 状态和成功降级证据。设置时间上限、请求预算与无进展检测，失败时输出状态并有界结束。

batch_decode 模式保持固定 batch 契约，灌注批与终止尾批必要时补 padding；FIFO 不补 padding。每种请求均标记 phase/kind，padding 不进入正式请求统计。

### 4.3 稳定窗口

灌注达标后切换到正式 grid 的长度、目标 reuse 和 cold/hit 分布，继续循环打乱发送。正式请求自然访问 seed；不在测量中额外刷新或 pin seed。

灌注可能将 seed 降级或淘汰，这是压力造成的真实状态。记录实际 reuse；若需要研究固定的 host-hit 比例，应另定义受控实验，不能把 host 有驻留数据当成正式请求一定会 host-hit。

默认连续 3 秒满足压力条件后才进入正式测量。按第 5.3 节的有效区间计算规则，需要覆盖完整 3 秒，1Hz 采样通常至少需要 4 次有效采样，不能用 3 个采样点代替 3 秒。steady 从切入正式 workload 起最多等待 60 秒，超时输出 `pressure_not_reached` 并有界收尾；所有阈值在运行前冻结并记录在结果中。

### 4.4 正式窗口与收尾

baseline 与 profile 独立运行，各自完成 seed、灌注和 steady。baseline 只启用必要的轻量观测；profile 在不停发压的情况下开启采集，该次运行中的请求性能指标只用于诊断和采集扰动对比，不替代 baseline。两次运行不声称具有逐块相同的缓存状态，需比较实际压力和请求分布。

正式时间窗口定义为 `[t0, t1)`，默认持续 60 秒，与按请求数截止互斥。Kineto 默认采集 20 个真实非空执行 step，剔除边界后报告实际可用 step 和相邻间隔数，不承诺得到 20 个有效间隔；Nsight 首版按显式配置的有界时间窗口采集。保持现有按 `--runs` 执行有限请求的模式；只有显式开启满载诊断及时间窗口时，才循环正式 grid。

时间窗口模式使用有界任务缓冲区，按需构造 token IDs，不能预先提交无限 futures。循环序号、nonce 和 seed family 分配可审计；constant/poisson 的计划时间连续推进，客户端限流造成的延迟仍计入 schedule lag，不通过重置计划时间掩盖断供。

到 `t1` 停止新增正式请求，单独计时完成在途请求。发送阶段以实际发送时刻为准，不能按任务入队时刻提前标记为 measurement。窗口内发送、窗口外完成的请求仍属于 measurement，并标记跨界。固定 batch 模式在最终停止供给时按总提交数补齐尾批，不因阶段标签切换丢弃已提交请求；padding 失败使运行无效。

profile 未完成时继续提供标为 support 的正式分布请求，默认最多延长至 `t1 + 60s`，已配置的更早采集超时仍然生效。support 期间继续采集压力证据，不把 support 请求或延长时间追加进 measurement 指标。profile 窗口及其延长部分单独报告压力有效性。超时执行显式停止采集、收集已有产物，并完成或有界取消在途请求；分别报告执行、压力和 trace 状态。

## 5. 压力信号与硬门槛

### 5.1 可观测性接口

现有 `KVCacheInfo` 主要包含 available/total KV、block size、keys 和版本；`/worker_status` 的汇总容量不能证明 host 驻留或 GPU 到 host 降级已完成。也不能把 `available` 一律当作 `free`，因为可淘汰 blocks 可能仍可用于分配。

优先复用 cache 内部 pool snapshot 和已有 transfer 完成事件，补齐可查询的累计计数，增加按需读取的结构化诊断快照，不在高频健康检查里扫描 cache keys。现有接口若缺少字段，以可选诊断查询/响应扩展提供，不改变正常健康检查成本。

快照至少包含：服务实例标识、rank、服务端单调时间和序号、pool/group/tier 及共享映射、capacity/used/free blocks、pending eviction release blocks、已配置水位及触发证据、cache/request/load/eviction 引用观测、host 已发布驻留 blocks、成功复制与成功发布的降级/回载 blocks/bytes 累计值、失败计数，以及可获取的 scheduler waiting/running 数量。引用计数可能重叠，不能直接相加作为 used，也不能用引用计数相减推导已发布驻留量。

TP 多卡需覆盖声明的全部目标 rank 和 FULL group，并检测采样缺失、陈旧数据与计数器重置；客户端轮询不能形成全 rank 原子快照，需保留时间戳及采样偏差。默认每 1 秒采样，最大采样间隔和数据年龄均为 2 秒，报告实际间隔、查询成本及 rank 间偏差。跨进程单调时钟不能直接相减；映射到统一运行时间轴时保留时钟映射及查询延迟的不确定性，无法确认的覆盖部分按未知处理。

真实 Flash TP 验证确认：当前 `BlockTreeCacheFactory` 仅让 TP0 管理可变缓存树，其他 rank 持有物理 pool 并执行共享 block ID 的传输，其本地 allocator/publication 计数不会随 TP0 的分配而增长。诊断必须显式声明独立本地所有权或受支持的 TP 共享所有权，包括 metadata owner rank、完整 participant ranks，以及发布是否要求全部 participant 的逐块传输成功。声明来自实际 cache/transfer 配置，不能根据零计数猜测；保留各 rank 原始本地快照，禁止把 owner 计数复制成 peer 本地观测值。

waiting 活动值表示本地等待队列与异步 cache loading 队列中的请求，不包含 running 请求；支持的 scheduler 在现有锁下读取 O(1) 队列大小。不支持的 scheduler/grouped 状态明确标为不可用，不能伪造零值。

### 5.2 判据

运行前冻结目标清单：所有目标 TP rank 的 FULL group 标为 `required`，SWA/LINEAR 标为 `observe_only`。记录每个 group set 关联的物理 pool、容量、水位及显式容差；共享物理 pool 去重，逻辑 group 映射保留。所有 required rank/group 分别满足门槛，不能用跨 rank 或跨 group 平均值掩盖不达标目标。

TP 共享所有权模式下，还要冻结 owner 与全部 participant 的实例、pool tag/group 映射、block geometry、容量和全 rank 发布范围。共享逻辑 slot 的 used/pending release、触发计数、发布增量和 host 驻留来自声明的 owner；各 participant 仍按自己的已验证物理布局、容量、水位和新鲜度分别检查。现有多 rank 传输汇总只有在所有 worker 的对应 block 均成功后才允许 owner 发布；部分复制成功不构成全 rank 发布证据。不支持的非对称布局、缺失 owner/peer、实例或所有权变化、发布范围不完整均不能达标。冻结清单和派生结果记录 owner 来源，原始 peer 零计数保留；准入预算也使用同一所有权模型，不能重复累加共享 slot。

GPU 判定匹配真实淘汰语义：以 group set 为淘汰单元，组内任一 device pool 达到高水位即可触发，不要求所有物理 pool 同时到达高水位。触发计算使用 `effective_used = used - pending_release`，高水位 blocks 为 `ceil(capacity * high_ratio)`，低水位 blocks 为 `floor(capacity * low_ratio)`。稳定期要求组内至少一个 device pool 的 effective_used 不低于其低水位减去预先冻结的容差；高于高水位单独报告，不直接算失压。保留各 pool 原始数据以解释触发来源和组内差异。

`host_resident_blocks` 定义为已成功发布且仍有效的 host 缓存块，包括迁移期间仍有效的 host 源数据；排除尚未发布的迁移目标块、detached 数据和单纯预分配空间。成功复制、成功发布和当前驻留是三个独立量；复制成功但未发布或已 detached 的数据不能充当成功降级证据。

| 条件 | 设计判据 |
| --- | --- |
| GPU cache 压力 | 初次达标有各 required FULL group 的本次运行水位触发证据；稳定期按上述 group set 水位语义判定 |
| Host 驻留 | 每个 required rank/FULL group 对应 host pool 的已发布驻留量达到目标，默认至少 `ceil(capacity * 0.25)` blocks |
| 成功溢出 | 各 required rank/FULL group 相对运行起点都有成功 GPU 到 host 发布增量；不能使用复制量、排队数或未完成目标块替代 |
| 稳态活动 | 请求和执行 step 持续推进，并在观察窗口内存在新的分配/降级进展；回载单独报告，不将其作为所有 workload 的必选门槛 |
| 请求供给 | 持续提交且无长时间断供；结合 waiting/running、实际 batch、schedule lag 解释是否有服务端积压 |
| 无严重异常 | 没有不可恢复请求失败、迁移失败或资源死锁 |

长 prefill 期间短暂没有请求完成，不单独判定断供或无进展；需结合正在执行的 step、waiting/running 和客户端发送记录。请求供给与进展检查独立于容量采样判定。不能让不可达到的条件无限等待，也不能运行后为得到 PASS 降低目标、扩大容差或删除不达标 rank/group/pool。

### 5.3 时间覆盖与失压

达标前不采正式性能数据。steady 未达标或灌注超时，输出 `pressure_not_reached`。measurement 全程报告“采样尺度上的 FULL attention 压力覆盖率”，不声称证明采样点之间每一瞬间都满载。

有效区间按以下规则计算：

1. 对每个 required rank/group，只有相邻两次采样都满足容量压力条件、数据未陈旧且采样间隔不超过 2 秒，两次采样间的可确认时间才计为有效。
2. 缺失、陈旧、实例变化或计数器重置的采样不能延用旧状态；窗口边缘没有证据覆盖的部分也算未知。
3. 将各 required rank/group 的有效区间映射到统一运行时间轴后求交集，并裁剪到 `[t0, t1)`。覆盖率为交集时长除以 `t1 - t0`，不能按采样点数量平均。
4. 失压和未知时间均计入无效时间。默认覆盖率必须不低于 95%，最长连续失压或未知区间必须不超过 2 秒，两项同时满足才通过时间覆盖门槛；请求供给和严重异常检查仍独立生效。

共享所有权模式的计数有效区间保留 owner 的时间戳和序号，并与全部 participant 的新鲜度/布局有效区间求交集；不能改用 peer 时钟重标记 owner 观测，也不能用 owner 响应替代缺失 peer。由此验证所有物理参与 rank，而不要求引擎并不存在的重复 allocator 树。

每次失压都保留 `pressure_lost` 事件和原因；是否使整个窗口的 `pressure_valid` 失败按覆盖率、最长无效时长及其他硬门槛判定。失压时不在窗口中补充灌注请求。support 延长期间另行计算 profile 窗口的覆盖率和最长无效时长，不改变已经结束的 measurement 统计。

## 6. 连续 Profile 的接入

当前 `cache_perf.py`、`batch_decode_test.py` 和 `CacheGridRunner` 都显式拒绝 continuous 与 profiling 组合，现有 profile 是选中 case 后 seed 并 replay 单请求。新增分支必须执行完整的 continuous 满载流程，不能仅删除校验并继续单请求 replay。

### 6.1 Kineto

复用 `_run_kineto_request()` 中的 `/start_profile` 控制协议和 `_wait_profile_traces()` 的异步文件收集思路，新增窗口级控制：

1. 发送器持续执行正式 workload，控制线程在压力达标后调用 `/start_profile`，配置独立 trace name、`num_steps=N`（默认 20）、`enable_all_rank=true`。
2. 控制调用和等待 trace 不占用发送 worker，不阻止补充请求。
3. 记录控制请求发送、确认时间与各 rank 实际 trace 起止，不能把 all-rank acknowledgement 当作同步起跑屏障。
4. 等待目标 step 完成并收集每个目标 rank 的完整 trace；measurement 期间持续正式供给，结束后仍未完成时按第 4.4 节提供有界 support 请求。异常或超时时执行显式停止/清理并保留原因。
5. 验证 GPU kernels、CPU ranges、CUDA 关联以及足够的有效 step；文件非空不等于采集完整。

当前引擎首次开启 profiler 的位置在输入准备和 TP 同步之后，因此首个 step 的前处理不完整。分析完整的内部 step/间隔，不能把边界缺失或 profiler 初始化耗时算成稳态 overhead。退出/flush 所在边界同样排除。分别报告采集的真实非空 step 数、剔除原因和剩余有效 step/间隔数，默认 20-step 采集目标不等于 20 个有效分析间隔。

### 6.2 Nsight Systems

复用已有 nsys launch session、`start/stop` 和异常收尾；将原来的“单请求加 tail”采集范围改为压力达标后的连续窗口。start/stop 由控制线程执行，发送器保持运行，服务销毁前必须完成 stop/export。

Nsight 第一版使用显式配置的有界时间窗口，事后核验包含的有效 step 数，并遵守第 4.4 节的 support 延长上限；不将 Kineto 的 20-step 默认值解释为 Nsight 秒数。若需要精确 N-step 停止，需依赖服务端 step 计数/完成信号，不能拿 N 个请求完成代替。

Kineto 的 `record_function` 不是自动可见的 NVTX 范围。Nsight 路径增加必要的阶段、step_id 和关键子阶段 NVTX，或使用已确认可用的标记桥接。两种 profiler 分开运行，避免多套 CUDA 采集器同时启用造成扰动。

## 7. Step Overhead 的定义与归因

### 7.1 三种不同的量

| 指标 | 定义与限制 |
| --- | --- |
| Host forward 间隔 | 同一 executor 线程上相邻 model_forward range 之间的墙钟时间，包含输出处理、调度、输入准备及等待 |
| Foreground CPU 时间 | 相同区间内线程 CPU 时钟差；不包含线程睡眠/被抢占时间，不代表所有后台 CPU 成本 |
| GPU 暴露停顿 | 前一批相关 GPU 工作完成到下一批计算开始的延迟，须检查通信、copy、多 stream 和异步重叠，不能直接等同于 CPU 开销 |

CPU 的 forward 返回不代表 GPU forward 完成。CPU 准备工作可能与上一批 GPU 计算重叠；也可能在下一批 CPU forward 范围内部仍有首个 kernel 提交前的准备。需要把 GPU kernel 与提交它的 CPU/CUDA 调用关联，而不是只量两条 CPU range 的空白。

### 7.2 服务端标记和轻量统计

为实际非空执行 batch 分配 engine step_id，记录 ctx_batch、gen_batch、执行 tokens、相关请求标识和单调时间。客户端请求 ID 与服务端实际 batch 映射由服务端提供，不能靠客户端完成顺序猜测；详细映射可仅在诊断模式开启。

复用现有 `engine.normal.schedule`、`executor.dispatch_output*`、`executor.wait_prev_dispatch*`、`executor.gather_model_input`、`executor.tp_sync_input` 和 `executor.model_forward`。只对无法归因的 cache match/allocate/load/release 等路径补充必要范围。

以服务端窗口标记和完整 step 对确定纳入的间隔，记录 wall 与线程 CPU 差，并单独记录明确的轮询/重试等待。正式 measurement step 分布仅纳入执行边界完整落在测量窗口内、成员均为 measurement 请求的 step；相邻间隔要求两个 step 都满足条件。跨窗口或混合 steady/measurement/support/padding 成员的 batch 保留在原始数据并单独分类。profile 内的纯 support step 可单独用于结构诊断，不能并入 measurement 分布。后台 task pool 与异步输出线程通过 trace/CPU sampling 单独分析，不把它们的时间与主线程墙钟相加。

baseline 输出轻量的 step 样本数、均值、p50/p95/p99/max、batch/tokens 分布；profile 提供详细归因。样本不足时不报告有代表性的 p99 结论。20-step trace 用于观察结构，长时间轻量统计用于分布。

### 7.3 Timeline 分析顺序

```text
CPU: forward(i) 提交 -> 输出/cache/调度/输入准备 -> forward(i+1) 提交
GPU:       prefill(i) 执行................ -> 间隔 -> prefill(i+1)
```

1. 确认 `ctx_batch>0`、`gen_batch=0`，排除空转、warmup、padding/support 的统计混用及 profile 边界效应。
2. 找到下一批首个计算 kernel，沿 correlation 回查 CPU 提交时间。
3. 提交晚则检查调度、输出处理、输入准备、cache 等待、线程抢占和客户端断供；提交早但执行晚则检查 stream 依赖、copy 和通信。
4. 将异常 step 与 cache 快照、成功降级/回载增量、batch/tokens、请求 reuse 和线程状态关联。
5. 用 perf 定位 on-CPU 热点，用上下文切换/OS Runtime 解释等待。父子范围按互斥口径归因，跨线程按关键依赖链解释。

没有 OS 调度信息时，普通 Torch CPU range 只能说明墙钟区间，不能精确区分 on/off CPU。禁止以总时长减 GPU kernel 时长或客户端排队时间代替 CPU overhead，也不为了方便计时添加每步 GPU synchronize。

## 8. 请求统计、Reuse 与数据有效性

沿用现有 request ID、case/run、seed family、发送序号、计划/实际发送时间、schedule lag、服务端 first-token cost 与 engine wait 字段；新增 phase/kind、epoch 和窗口归属。

原始客户端请求统计按实际发送阶段归属：发送时刻落在 `[t0, t1)` 的正式请求属于 measurement，晚于 `t1` 完成仍保留在该请求集合中，并标记跨界。吞吐单独按该集合中在 `[t0, t1)` 内完成的请求数除以窗口时长计算，报告完成数、跨界数和 drain 耗时；不能把 drain 完成数加进窗口吞吐。support/padding 不进入正式请求指标。服务端 step 统计以服务端执行边界及实际请求成员为准，两者不强行一一对齐。保留 inflight 起止数量以及阶段边界偏差。

满载下 seed 可能在 GPU、host 或已被淘汰，因此区分 `cache_len_requested` 和实际 reuse，能取得时进一步区分 device/host reuse。已有精确 reuse 校验和公式拟合契约不变；新增满载诊断结果类型，即使压力运行有效，也不能混入旧的精确命中率拟合数据。

reuse 偏差不自动代表请求失败，但输入输出形状错误、非预期 decode、迁移失败、缺失时序和失压仍需明确标记。现有 `--skip-reuse-validation` 不能代替这一结果契约。

分别报告 `execution_status`、`pressure_valid`、`trace_valid`、`reuse_exact` 和 `diagnostic_only`，避免一个 status 混淆“请求成功”“FULL attention 压力成立”和“数据可用于正式拟合”。measurement 与 profile 窗口分别关联压力有效性；baseline 的 `trace_valid` 标为不适用。trace 失败不抹去已完成的请求和压力证据，也不影响独立 baseline 运行的有效结果。

## 9. 拟议配置与兼容性

以下为拟议新增配置，不是当前可执行命令。最终命名沿用 CLI 到 perf_config 的现有映射规范。

| 配置 | 用途 |
| --- | --- |
| `--pressure-preload` | 显式启用容量灌注与满载诊断；未启用时保持原 continuous 行为 |
| `--pressure-fill-timeout` / `--pressure-fill-max-requests` | 灌注上限 |
| `--pressure-host-target-ratio` | FULL 对应 host pool 的已发布驻留目标，默认 0.25 |
| `--pressure-steady-seconds` | 正式分布的连续达标时长，默认 3 秒 |
| `--pressure-measure-seconds` | 正式时间窗口，默认 60 秒；与请求数截止互斥 |
| `--pressure-profile-steps` | continuous Kineto 的真实非空执行步数，默认 20；有效分析间隔数另行报告 |
| `--pressure-profile-seconds` | continuous nsys 显式配置的有界时间窗口，与按 step 停止方式区分 |

以下运行参数统一纳入 resolved config；最终是否暴露独立 CLI 参数沿用现有配置映射，先保持配置面最小，不引入另一个独立 YAML 协议：

| 参数 | 首版默认值 |
| --- | --- |
| 满载判定范围 | 所有目标 TP rank 的 FULL group；SWA/LINEAR 仅观测 |
| 压力采样周期 | 1 秒 |
| 最大采样间隔 / 数据年龄 | 各 2 秒 |
| steady 总等待上限 | 60 秒 |
| 有效压力时间覆盖率下限 | 95% |
| 最长连续失压或未知时长上限 | 2 秒 |
| measurement 结束后的 support 延长上限 | 60 秒 |

GPU 门槛由 FULL group 的真实低/高水位及运行前显式容差解析；目标清单、容差、采样及窗口参数、灌注预算和 profile 超时全部冻结并写入结果。上述数值为已确认的首版默认值，校准或覆盖只能在下一次运行前完成；尚未指定数值的灌注预算、GPU 容差、Nsight 窗口等必须在启动前解析为明确配置，不能运行中临时选择。

`cache_perf run` 的 continuous preload 分支产出 baseline；`cache_perf profile` 增加 continuous workload 分支，复用 `--profile-backend`，不再沿用该分支下单 case `--cases` 必填和单请求 replay 的语义。原 case profile/retest 保持原行为。直接底层 runner 调用也需同步验证组合。

profile/grid/环境/压力配置继续冻结。发生重启或 resume 时必须重新建立 cache 压力，因为磁盘 checkpoint 不代表内存 cache 仍在；保存新的 session/epoch，不把两个运行段拼成一段连续 timeline。

## 10. 结果产物

建议在现有结果目录下增加诊断产物，避免向原 metrics[] 混入不符合精确 reuse 契约的数据：

```text
result_dir/
    已有 profile/grid/启动配置快照与请求结果
    pressure/
        manifest.json          阶段、配置、有效性、失败原因、请求摘要
        cache_samples.jsonl    各 rank/pool 的压力时间序列
        steps.jsonl            服务端 step 边界及可用轻量指标
        summary.json           压力覆盖率、step 分布和请求/批次分布
    timelines/ 或 nsys/        沿用对应后端的产物组织
```

保存代码版本、profile/grid hash、随机种子、实际输入流标识、服务实例、phase 与 rank 时间边界，以及冻结的 FULL 目标清单、observe-only group、共享 pool 映射和全部阈值。分别保存 measurement/profile 的有效区间、未知区间、覆盖率、最长无效时长和 support/drain 边界。现有随机顺序种子并不等于完全相同 token IDs；当前 UUID suffix 等因素必须记录或提供确定性生成方式，不能宣称相同 seed 就能精确复现 cache 内容。

正式报告至少回答：采样尺度上的 FULL attention 压力是否达标、有效覆盖率及最长失压/未知时长、是否持续供给、step 间隔及 CPU 分布、长尾原因、复制/发布/驻留及回载活动、实际 reuse 形态，以及 profiling 相对裸跑的扰动。SWA/LINEAR 观测单独展示，不将 FULL 达标表述为所有 cache group 满载。

## 11. 修改边界与实施顺序

| 阶段 | 修改内容 | 完成标准 |
| --- | --- | --- |
| A. 压力观测 | 按需 cache 分层快照、成功迁移计数、rank/时间信息 | 能证明驻留和溢出，区分 free 与 available，缺失数据明确失败 |
| B. 灌注与连续窗口 | CacheGridRunner 的 fill/steady/measured 状态、有界任务供给、兼容参数 | 达标后才测量，阶段切换不断流，FIFO/固定 batch 收尾正确 |
| C. 连续 profile | cache_perf CLI/校验、Kineto 多 step、nsys 窗口、异步采集收尾 | 压力流不因控制操作中断，trace 完整且标明窗口 |
| D. Step 分析与报告 | 引擎轻量 step 标记、必要范围、诊断数据契约 | 请求与 batch 可关联，CPU/等待/GPU 停顿口径清晰 |
| E. 验证与文档 | 单测、GPU smoke、正式基线、profile 对照、使用说明 | 满足下述验收标准 |

压力 snapshot 的协议/服务端实现和必要引擎标记是跨模块改动，需要覆盖相关接口测试；不修改 C++ cache benchmark，也不重写已有 sender、传输协议或真实 scheduler。

## 12. 验证与验收

单元测试覆盖：容量达标/未达标/超时/计数重置、全 cold 仍灌注、seed 被淘汰后的 reuse 契约、阶段切换不断流、固定 batch 尾批与 FIFO 无 padding、窗口跨界计数、profile 不阻塞发送、缺少 rank/空 trace/超时收尾和原 case 模式兼容。

新增判据与窗口契约重点覆盖：FULL-only 门槛及无 FULL 配置拒绝、SWA/LINEAR 未达标不影响 FULL 容量判定、共享 pool 去重、group 内任一 pool 触发及 pending release 扣除、复制成功但未发布/detached 不计降级、全部目标 rank/group 分别达标；3 秒 steady 必须有完整时间覆盖、60 秒 steady 超时、采样间隔/年龄超限、跨 rank 有效区间交集、窗口边缘未知、95% 覆盖率与 2 秒最长无效时长同时检查；按实际发送时刻归属、跨界完成不增加窗口吞吐、混合 batch 分类、support 延长上限及其独立压力统计、20-step 采集扣除边界后的有效样本数。

GPU smoke 使用可控的小容量配置，验证真实 GPU 到 host 降级完成，再连续执行真实 prefill；采到多个非空 step 且无 decode，确认 trace 完整、压力信号可信及所有请求最终结束。相关构建和测试执行遵循仓库 test-execution 流程。

正式验收：

- 从 cache_perf 入口完成 seed、容量灌注、稳定运行、测量/profile 和收尾。
- 进入测量前有所有目标 TP rank/FULL group 的水位触发、成功 GPU 到 host 发布增量和默认至少 25% host 驻留证据；连续达标默认 3 秒，steady 默认最多等待 60 秒，未达标不输出有效满载结果。
- 默认 60 秒 measurement 中，按时间求交集的 FULL 压力覆盖率至少 95%，最长连续失压或未知区间不超过 2 秒；保留失压、采样缺失、陈旧和断供证据，明确结论仅在采样尺度成立。
- 稳态到测量/profile 不清缓存、不重启、不清空请求后重发。
- 连续多 prefill step 的内部间隔可分析，CPU 与 GPU 时序不混淆，纯 prefill 由实际 batch 标记确认。
- profile 与 baseline 分开，采样结果不混入原精确 reuse 公式拟合。
- measurement 按实际发送时刻归属，跨界请求、混合 batch、support/padding 和 drain 分开统计；Kineto 默认采集 20 个真实非空 step，报告边界排除后的有效样本数，profile 未完成时 support 默认最多延长 60 秒。
- 停止时完成或有界取消在途请求、停止采集并收集产物；不把常驻 cache 数据要求为零。能够观测的请求/迁移在途资源应完成收尾。

未通过的运行必须保留原始压力、请求和 profile 证据，给出具体失败阶段与原因。
