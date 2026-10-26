# FIFO 连续 FULL 满载压力测试操作指南

本指南对应 `refine-continuous-pressure-seeding`：从启动后的真实 FULL owner 拓扑自动计算 seed，使用有界 FIFO 请求流建立压力，并在 measurement 内分别检查执行、FULL 压力、正式配置和双向迁移。开启方式为 `--pressure-mode continuous --pressure-preload`，此路径默认且只支持 FIFO。只设置 continuous 而不 preload 仍是原有按次数运行的模式。

## 1. 正式配置与非正式 smoke

| 项目 | 正式 production | NON-FORMAL GPU smoke |
| --- | --- | --- |
| profile | [production profile](../config/continuous_full_pressure_dsv4.production.example.jsonc) | [smoke profile](../config/continuous_full_pressure_dsv4.example.jsonc) |
| grid | [production grid](../config/continuous_full_pressure_grid.production.example.json) | [smoke grid](../config/continuous_full_pressure_grid.example.json) |
| 模型 | `/mnt/hf3fs/3fs/models/DeepSeek-V4-Flash` | 同左 |
| 并行 | TP4 / EP4 / DP1 / world4 | 同左 |
| sequence / token budget | 131072 / 131072 | 65536 / 65536 |
| 引擎 context / admission 配置 | 16 / 16 | 2 / 2 |
| 客户端在途上限 | 16，正式最低值也是 16 | 显式 2 |
| seed family 数 | 不传手工值，启动后计算 | 显式 2，必为非正式 |
| host cache 配置 | 65536 MiB | 512 MiB，仅 smoke |
| GPU 容量 | 不传 test/fixed pool 容量替换 | `test_block_num=128`、`dsv4_fixed_pool_blocks=128` |
| fill 输入 | 默认 102400 tokens | 显式 16384 tokens |
| seed 独立期限 | 3600 秒 | 300 秒 |
| Bazel 外层总期限 | 10800 秒 | 3600 秒 |
| grid 最大 cache_len | 65536（包含 32768 复用请求） | 8192，不满足正式 32K 门槛 |

两份 profile 保留同一模型、FP8 KV、缓存开关、GCC12/CUDA13.2、`/opt/conda310`、Bazel cuda13 和 GPU `2,3,4,5`。启动前检查实际模型/编译器路径、空闲四卡和磁盘空间；JIT 缓存使用 `/tmp/rtp-llm-full-pressure/`，首次可能重新编译。可在首次命令添加 `--env CUDA_VISIBLE_DEVICES=实际四卡` 覆盖选择。历史硬件为 NVIDIA L20D、约 275040 MiB/卡；这是历史环境信息，不能推定本次容量。

从仓库根目录运行，让启动器用匹配的 Bazel/runfiles 独占启动服务。要求 DP1/PP1、完整 TP rank、`dashsc_input_ids`、未分组 `batch_size=1` grid、FULL attention 和已启用的 host cache；不兼容 existing server、fixed workspace、materialized inputs 或 shared-seed。grid 的 batch_size=1 描述单请求，FIFO 自行决定动态 batch。

## 2. 先执行离线校验

以下命令只打印计划，不写结果、不调用 Bazel、不占 GPU：

```bash
python3 tools/cache_perf --help
python3 -m unittest rtp_llm.test.perf_test.cache_grid.tests.cache_perf_test -v
python3 tools/cache_perf run \
  --profile rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.production.example.jsonc \
  --grid rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.production.example.json \
  --result-dir /tmp/dsv4-pressure-plan-check \
  --pressure-mode continuous --pressure-preload --dry-run
python3 tools/cache_perf run \
  --profile rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.example.jsonc \
  --grid rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.example.json \
  --result-dir /tmp/dsv4-pressure-smoke-plan-check \
  --pressure-mode continuous --pressure-preload \
  --pressure-client-concurrency 2 --pressure-seed-count 2 --dry-run
```

正式计划应包含 Bazel 外层期限 `test_timeout=10800`、`--cache_pressure_scheduler=fifo`、客户端 16、102400 fill，且没有 seed-count flag；`formal_candidate=true`、`formal_candidate_reasons=[]` 只是 `static_configuration` 范围的初筛。此时没有真实 pool 容量、自动 seed 数或 `formal_config_valid` 结论。

smoke 静态初筛必须为 false，依次保留 `manual_seed_count`、`client_inflight_below_16`、`synthetic_test_block_capacity`、`nonstandard_fill_target`、`explicit_smoke`。profile 的 `cache_grid.explicit_smoke=true` 和 grid 的 `kind=explicit_cache_grid_smoke` 均能明确标记 smoke。启动后的 planner 读到小 grid 后还会追加 `grid_reuse_below_32768`，后续成功不能消除这些原因。

## 3. 正式 baseline 命令

```bash
python3 tools/cache_perf run \
  --profile rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.production.example.jsonc \
  --grid rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.production.example.json \
  --result-dir /data0/zhouhaiyu.zhy/tmp/dsv4-formal-pressure \
  --pressure-mode continuous --pressure-preload \
  --pressure-client-concurrency 16 \
  --pressure-request-rate 0 --pressure-arrival constant \
  --pressure-measure-seconds 60
```

正式运行省略手工 seed 数；32K/64K grid 中的 cold/hit 请求在稳态与测量中重复运行。`cache_len` 是构造的复用目标，真实淘汰可能降低实际命中，不能要求它始终 GPU 命中。

请求速率 0 表示不节流，仍最多 16 个 RPC 在途，包括服务端排队和正在执行的请求。它不表示 16 个长请求同时进入 GPU，也不受当前 active batch 大小直接封顶。preflight 分别验证每种 seed/fill/grid 请求的单请求模型、token、KV 可执行性，以及有界 RPC backlog 的 admission。FIFO 根据自身 token/KV 资源决定实际 batch，不做并发静默缩减、固定批设置或 padding。

结果目录须为新目录或空目录。日志写在目录旁边；若通过 tee 保存，先设置 Bash `set -o pipefail`，避免吞掉启动器失败。60 秒仅为稳态之后的 measurement，不包括编译、加载、seed、fill、steady 和 drain。baseline 不接受显式 `--runs`。

## 4. 真实容量与自动 seed 公式

容量来自启动后 `GetCachePressure` 返回的 FULL owner，不从 profile 的块数猜测。每个 owner/group 的 device pool ID 去重；TP participant 不重复累加共享容量或发布计数，但每个 peer 的身份、布局和采样新鲜度仍必须验证。

`capacity_blocks` 是 RPC 报告的**可分配逻辑块容量**，已经反映 block zero 等保留块，不能再减一次 1。`available_blocks` 是扣除不可回收占用后当前可保留/分配预算，不能简单替换为 total-used。历史 127 是强制 `test_block_num=128` smoke 的可分配 FULL 块数，绝不是 Flash production 的固有 FULL 容量。模型权重显存也不属于 KV pool 容量。

令 A 为 cache alignment，R 为可复用 token 数（不含 commit tail）：

```text
grid_reuse_tokens = max(grid.cache_len)
R = align_up(max(32768, grid_reuse_tokens), A)
seed_input_tokens = R + commit_tail_tokens
B[g,p] = ceil(R / block_tokens[g,p])
GPU[g] = min_p floor(capacity_blocks[g,p] / B[g,p])
LOW[g] = min_p floor(floor(capacity_blocks[g,p] * low_ratio[g,p]) / B[g,p])
HOST[g] = floor(host.available_blocks / ceil(R / host.block_tokens))
minimum[g] = GPU[g] + 1
maximum[g] = LOW[g] + HOST[g]
seed_count = max_g minimum[g]
要求 seed_count <= min_g maximum[g]
```

所有 g 都是必需 FULL owner group；p 是其去重的 device pool。block token span 按 live RPC 几何解释，不能把字节或 device 块数直接当 host 块数。当前 schema1 在 group 上报告 token 几何，plan 的各 pool 保留 `geometry_source=owner_group`。其中 `block_tokens` 已由 manager 根据实际 CP policy 转为全局请求 token 跨度，capacity/occupancy 仍是 owner 物理块数。CP4 block round-robin 的物理 512-token block 对应 2048 全局 tokens，65536-token family 占 32 owner blocks；不能再除一次 CP，也不能相加四个 TP 视图。manifest 保存每项输入和上下界，可逐项复算。高/低水位及 FULL 可达性另由冻结目标校验；边界不相交就停止并报告限制 group，不自动缩短复用长度或削弱压力门槛。

本 production grid 的 R=65536，commit tail=4096，因此 seed 输入为 69632。正式 grid 必须实际包含至少 32768 的 cache_len；仅把 seed 初始化变长不能代替真实复用工作集。fill 使用独立唯一 family，目标默认 `100*1024=102400`，按 alignment 向上对齐；其一次性后缀不计入可复用 seed bank。fill 加 1 个输出 token 必须通过单请求预算，不允许悄悄缩短。

## 5. 阶段与期限

```text
validate/live plan -> rolling seed -> unique fill -> tiered rewarm -> steady -> measurement
                                                                             -> optional support -> drain
```

seed 懒生成至在途上限，任一成功完成后立即补下一个 family；不会每 16 个全体排空再开下一批。恰好提交 seed_count 个后有界排空。`seed_timeout_s` 是独立期限，不占 fill 的时间。必须保留 planned/submitted/completed/failed/cancelled/pending；缺失、失败、取消、未完成或计划数不一致都不能进入有效 measurement。

fill 通过真实新输入、DEVICE 高水位触发和成功 D2H 发布建立压力。使用 `tree_pressure + tiered_rolling` 时，fill 达标后先重热 cohort 0 的 device half（32-request cohort 对应 16 个 family）；它使用与正式首批 device lane 相同的 family/case，但不推进正式 sequence，也不形成 cohort 屏障。重热全部成功且重热后的新诊断快照仍低于 `fill_host_max_ratio`，才进入 steady；否则以 `rewarm_incomplete` 或 `fill_headroom_exhausted` 终止。随后正式 grid 连续供给 steady、measurement、support，阶段之间不清缓存、不暂停排空。测量边界固定为 `[t0,t1)`；窗口内发出、窗口外完成的请求仍属该 cohort，但 drain 完成不增加窗口内吞吐。

| profile 字段 | production / smoke | CLI 覆盖 |
| --- | --- | --- |
| `seed_timeout_s`（必需） | 3600 / 300 s | `--pressure-seed-timeout` |
| `fill_timeout_s`（必需） | 1800 / 300 s | `--pressure-fill-timeout` |
| `fill_max_requests`（必需） | 1000 / 1000 | `--pressure-fill-max-requests` |
| `fill_no_progress_s`（必需） | 300 / 60 s | `--pressure-fill-no-progress` |
| `fill_input_tokens`（默认 102400） | 省略 / 16384 | `--pressure-fill-input-tokens` |
| `gpu_tolerance_blocks`（必需） | 2 / 2 | `--pressure-gpu-tolerance-blocks` |
| `activity_timeout_s`（必需） | 300 / 120 s | `--pressure-activity-timeout` |
| `drain_timeout_s`（必需） | 300 / 60 s | `--pressure-drain-timeout` |
| `capture_timeout_s`（必需） | 600 / 120 s | `--pressure-capture-timeout` |

production 的 `bazel.test_timeout=10800` 是整个测试进程的外层期限，须覆盖 seed、fill、steady、measurement、support、drain 与启动/JIT。离线回归还保守加上 capture 600 秒：阶段预算总和为 6480 秒，余量为 4320 秒；它们有重叠，不能把外层期限误当 seed 期限。调整阶段预算时也应在首次运行前增大外层预算。smoke 保留 3600 秒外层期限。

共同默认值：采样周期 1 秒，最大样本间隔/年龄各 2 秒，连续 steady 3 秒，steady 等待 60 秒，measurement 60 秒，host residency 25%，有效覆盖率至少 95%，最长无效/未知区间不超过 2 秒，support 最多 60 秒。CLI 可在首次运行前覆盖并冻结；不要在运行中为通过验收降低阈值。

## 6. Kineto 与 Nsight

下面两个正式采集实验使用与 baseline 相同的 production profile/grid，独立新目录并依次执行；保留自动 seed：

```bash
python3 tools/cache_perf profile \
  --profile rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.production.example.jsonc \
  --grid rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.production.example.json \
  --result-dir /data0/zhouhaiyu.zhy/tmp/dsv4-formal-kineto \
  --pressure-mode continuous --pressure-preload \
  --pressure-client-concurrency 16 \
  --pressure-request-rate 0 --pressure-arrival constant \
  --pressure-measure-seconds 60 \
  --profile-backend kineto --pressure-profile-steps 20

nsys --version
python3 tools/cache_perf profile \
  --profile rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.production.example.jsonc \
  --grid rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.production.example.json \
  --result-dir /data0/zhouhaiyu.zhy/tmp/dsv4-formal-nsys \
  --pressure-mode continuous --pressure-preload \
  --pressure-client-concurrency 16 \
  --pressure-request-rate 0 --pressure-arrival constant \
  --pressure-measure-seconds 60 \
  --profile-backend nsys --pressure-profile-seconds 10
```

Kineto 与 Nsight 互斥，不传 `--cases`。20 steps 是每 rank 的真实非空步骤目标；边界排除后需分别查看 observed、usable 和相邻步骤对。历史四 rank Kineto 文件总量曾约 1.1 GB，预留空间与解析预算。Nsight 若不在 runtime PATH，首次命令添加 `--nsys-path /实际路径/nsys`；历史验证版本为 2025.6.3。

采集在 steady 之后开始，Nsight 10 秒从启动 ACK 后计时。measurement 结束后可在原期限内继续同分布 support，但不会延长正式 measurement。排空后的解析仍计入 capture deadline；启动器负责 NVTX，无须手动覆盖环境。CUDA 关联缺失/歧义使 trace 无效，GPU gap 候选不能直接证明 GPU 空闲或 CPU 瓶颈。

## 7. 明确非正式的 GPU smoke

```bash
python3 tools/cache_perf run \
  --profile rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.example.jsonc \
  --grid rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.example.json \
  --result-dir /data0/zhouhaiyu.zhy/tmp/dsv4-smoke-pressure \
  --pressure-mode continuous --pressure-preload \
  --pressure-client-concurrency 2 --pressure-seed-count 2 \
  --pressure-request-rate 0 --pressure-arrival constant \
  --pressure-measure-seconds 60
```

此配置验证真实请求、阶段、清理与产物，结果必须 non-formal。smoke 的 R 仍至少 32768，seed 输入 36864；因此 sequence 上限从旧 smoke 的 32768 调整到 65536，host 从 128 调整到 512 MiB，以容纳两 family，而保留 128 强制 GPU pool、2-way limits、16K fill。实际 host 可保留块仍由 live snapshot 决定。

仅用于解释的非 CP 离线算例：若 owner 报告 device capacity=127、全局 block span=512、low=.82，且 host available=72，则每 family=64 blocks，GPU=1、LOW=1、HOST=1，上界为 2；手工 2 family 可保留。CP4 下全局 span 若为2048，则同一32768-token family占16 blocks，必须按新的 live 几何重算。72 是算例，不能用 512 MiB 静态推断 live host 预算，也不能把这些 smoke 数字用于 production。即使执行、压力和迁移均通过，formal_config_valid 仍必须 false。

## 8. 产物与四个正式门槛

```text
result_dir/
  cache_perf_launch.json
  profile.snapshot.json
  grid.snapshot.json
  pressure_latest.json
  pressure_epochs/<epoch_id>/
    startup_pressure.json
    preflight_failure.json       # 仅 preflight 失败时
    pressure/
      manifest.json
      summary.json
      requests.jsonl
      cache_samples.jsonl
      steps.jsonl
```

每次启动/恢复创建新 epoch，latest 原子指向最新启动，即使 preflight 失败也不覆盖历史。服务未进入 runner 前失败时可能没有 latest，先检查启动日志。早期失败可能没有全部 JSONL。

以下只读命令显示最新真实结论：

```bash
python3 - /data0/zhouhaiyu.zhy/tmp/dsv4-formal-pressure <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
latest = json.loads((root / "pressure_latest.json").read_text())
epoch = Path(latest["directory"])
path = epoch / "pressure" / "summary.json"
if not path.exists():
    failure = epoch / "preflight_failure.json"
    print(failure.read_text() if failure.exists() else "摘要尚未生成，请检查运行日志。")
    raise SystemExit(1)
s = json.loads(path.read_text())
print(json.dumps({k: s.get(k) for k in (
    "validity", "validity_reasons", "classification", "seed_plan", "seed_outcome", "window"
)}, ensure_ascii=False, indent=2))
PY
```

| 正式门槛 | 证据与判定 |
| --- | --- |
| `validity.execution_valid` | 请求/形状/控制/清理成功；seed planned=submitted=completed=冻结计划数，failed=cancelled=pending=0 |
| `validity.full_pressure_valid` | measurement 内所有 FULL 目标有效时间交集满足覆盖和最长无效区间 |
| `validity.formal_config_valid` | 冻结 live plan 正式有效，无任何保留的非正式原因 |
| `validity.bidirectional_migration_valid` | measurement 内每个必需 FULL owner group 的成功 D2H 与 H2D 发布增量均 >0 |

`validity.formal_acceptance_valid` 必须是四项同时 true。FULL 门槛还要求本 epoch 实际 DEVICE 高水位触发及成功 GPU-to-host 发布、至少一个关联 device pool 的 `used-pending_release >= floor(capacity*low_ratio)-tolerance`，以及 `host_resident >= ceil(host_capacity*.25)`。HOST/DISK 触发不能替代 DEVICE；预留、复制完成未发布或 detached 数据不能算 host 驻留。

双向门槛只接受保守时间区间完整落在 measurement 内的 settlement-backed 发布：DEVICE->HOST 的 STORE/EVICT，以及 HOST->DEVICE 的 LOAD。copy-start、失败/取消、seed/fill/steady/support 阶段传输不能代替。owner 共享发布计数一次，同时校验全部 participants；计数回退、实例/拓扑变化或边界证据缺失均不能通过。

manifest 与 summary 都保留同一冻结 `seed_plan`、终态 `seed_outcome`、`classification.inputs/nonformal_reasons` 和独立 `validity_reasons`。复算需核对 raw startup、owner/participant、去重 pool、capacity/available/geometry、水位、上下界、auto/manual source、对齐及 commit/fill 长度，而不能只看程序退出码。`manifest.complete=true` 只表示写出收尾，累计 `epoch_reasons` 也不能代替窗口评估。

profiling 还要单独检查 `pressure_valid.profile` 和 `trace_valid`；baseline 的 trace null 为不适用。这些诊断结果始终 `execution_mode=full_pressure_diagnostic_v1`、`metrics=[]`，不进入 exact-reuse 拟合/严格图表。host residency 不能直接证明 H2D。

## 9. 恢复与排错

```bash
python3 tools/cache_perf resume --result-dir /data0/zhouhaiyu.zhy/tmp/dsv4-formal-pressure
```

仅有 checkpoint 后才能 resume；使用冻结的 profile/grid/环境/配置，重新获取 live plan 并执行 seed/fill/steady，不拼接 epoch。早期无 checkpoint 或修改 GPU/容量/并发/窗口后，用新目录。

| 现象 | 检查与处理 |
| --- | --- |
| `FIFO-only` / batch-size 不适用 | preload 只支持 FIFO；删除固定 batch 控制，普通 batch case 模式另行使用 |
| `pressure time window conflicts with explicit --runs` | baseline 删除 runs，以 measurement 秒数控制 |
| `infeasible ... seed_count` | 查看每个 owner 的 minimum/maximum 与 live host available；增加真实 host 可保留预算或另做非正式 smoke，不能手工小 seed 冒充正式 |
| seed 超时/部分完成 | 查看 seed_outcome、请求错误和取消；按真实模型耗时在下次运行前设置 seed 期限 |
| fill/seed/grid 单请求预算失败 | 检查 input+1、模型长度、FIFO token/KV 上限；不能通过减少客户端深度掩盖请求本身不可执行 |
| `FULL requires positive host capacity` | 检查 enable_memory_cache=1 及真实 host 容量，大小设置不等于启用开关 |
| 没有 FULL 目标 | 当前不能用纯 SWA/LINEAR 模型认证 FULL 满载 |
| fill/steady 超时、测量丢压 | 检查 DEVICE 触发、D2H 发布、host 驻留、供给与采样空洞；正请求速率可能不足以维持压力 |
| 只有 D2H、无 H2D | 压力有效仍可迁移无效；核对 reuse 工作集、host 可保留预算及测量边界，不用 fill 流量替代 |
| 模型加载 OOM | 核对四卡实际空闲显存与权重开销，smoke 小 KV pool 不会缩小模型权重 |
| trace 无效/解析超时 | 检查各 rank 文件、CUDA 关联和整个 capture deadline，不能只看文件存在 |
| cancel 已返回但仍有 GPU 活动 | transport cancel 不保证立刻中断 GPU，核对有界 drain、诊断关闭和服务清理 |

历史 2026-10-21 的 FIFO baseline/Kineto/Nsight 各 60 秒、压力覆盖 100% 记录来自强制小池 smoke；保存 trace 的离线复验未改变原 SHA。固定 batch2/all-cold 曾用另一容量通过旧版判据；该路径已被 FIFO-only capability 替代，不满足本指南的正式验收。历史记录保留在 [reference](reference.md#validated-runtime-acceptance)，不能证明真实 production 容量、自动 seed 或测量内双向迁移已通过。

历史 Nsight 一次出现尚未归因的 rank2 native SIGSEGV；后续成功不等于根因修复。保留失败日志、maps/core（如有）和原始采集文件。实现细节见 [reference](reference.md#full-pressure-cli-and-resume)。

## 10. Mock Memory 诊断配置与验收

[独立 mock profile](../config/continuous_full_pressure_dsv4.mock_memory.example.jsonc) 从 production 示例派生，保留 TP4/EP4、真实容量和工具链设置，不修改 production 或小池 smoke 的含义。它仅用于缓存迁移与 CPU 开销诊断：合成 KV 不对应输入 token，**模型输出正确性无效，不能做质量评测、exact-reuse 拟合或正式推理性能结论**。profile 的结构化 `cache_grid.diagnostic_only=true` 会冻结进 snapshot、live seed plan 和 classification，强制 `formal_candidate=false`，原因 `diagnostic_only`；标签同时写明 `DIAGNOSTIC ONLY; OUTPUT CORRECTNESS INVALID`。该可选布尔字段缺省为 false，与 `explicit_smoke` 是两个独立标记。

该配置设定 `--mock_memory_cache=1`、`--mock_memory_cache_hit_ratio=0.5`、`--mock_memory_cache_seed=1`、`--mock_memory_cache_source_blocks=64`，并显式冻结 `cache_grid.pressure.evidence_policy=mock_memory`。profile schema 中 CLI 布尔参数使用整数 0/1；这里的 1 与服务端 `true` 等效。pressure frontend 按下表校验服务开关与 evidence policy。启动仍要求 reuse 和 Memory cache 已开启、Host 容量为正且足以预留 source。只改变 ratio 不会启用 mock。

| `--mock_memory_cache` | 允许的 evidence policy | 未指定 policy 时 |
| --- | --- | --- |
| 0 或未启用 | `tree_pressure` | `tree_pressure` |
| 1 | `mock_memory`、`tree_then_mock_memory` | `mock_memory`，兼容既有配置 |

ratio 指初始 prompt 分配中累计可复用 token 的覆盖率，按完整复用块计算，**不是命中请求占比**。设累计 eligible 逻辑块为 E，计划 Host 块为 P，目标为 `floor(E * ratio)`；请求长度变化、并发 admission 由同一控制器累计，必须符合模型允许的复用边界，尾部至少保留必要的未复用 token。合法前缀粒度可暂时留下不足一个有效复用单元的缺口，后续请求继续追赶。每个请求只在首次计划时收费，分配重试沿用计划；decode/后续扩容不参与比例控制。

mock 模式绕过自然 Device、Host、Disk 前缀匹配，即使缓存中已有同 token 的条目也不采用。初始复用归因只有 Host，Device/Disk 必须为零；ratio=0 时也不能恢复自然命中。H2D 必须在 admission 中走真实 descriptor、CRC 校验（若开启）、异步提交和等待/唤醒链路。成功完成的请求尝试真实异步 D2H Host writeback，不能用响应完成或 Host 驻留代替传输完成证据；失败/取消请求不做成功完成写回。

每个 Host GroupSet/pool 在启动期预留 64 个不可变 source 块，清零整个 stride，并在开启 CRC 时预先构造合法 CRC footer。初始化在压力/profile 窗口前完成，不应计入请求 CPU 热路径。seed 仅确定 ring offset；逻辑块号循环映射，因此 `source_blocks=64` 限制 Host 预留，不限制单请求/累计传输块数。source 用独立 `MOCK_SOURCE` 引用持有，流量期间数量恒定，不进入普通树；剩余 Host 空间还要容纳真实写回。

运行前先核对实际模型路径、GPU 和工具链。下面配套使用[独立 mock grid](../config/continuous_full_pressure_grid.mock_memory.example.json)：它只保留 production 的 32768/65536-token 复用 case（原 case_id 1、3），排除所有 `cache_len=0` cold case，保留 token 长度、alignment 和 commit tail。`tiered_rolling` 在运行时严格拒绝 cold case，离线 dry-run 也使用同一兼容性检查，不会静默过滤用户提供的 grid。production/smoke profile 与 grid 均未修改。结果目录必须是新的；首次可追加 `--dry-run` 离线检查。dry-run 的 `resolved_engine_args` 列出四个实际开关，label 和静态 classification 必须明确诊断用途，且不写文件或启动服务。

```bash
python3 tools/cache_perf profile \
  --profile rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.mock_memory.example.jsonc \
  --grid rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.mock_memory.example.json \
  --result-dir /data0/zhouhaiyu.zhy/tmp/dsv4-mock-memory-nsys-acceptance \
  --pressure-mode continuous --pressure-preload \
  --pressure-client-concurrency 16 \
  --pressure-request-rate 0 --pressure-arrival constant \
  --pressure-measure-seconds 60 \
  --profile-backend nsys --pressure-profile-seconds 10
```

独立 adaptive sender 使用 `adaptive_backlog`，速率范围 50–400 requests/s、backlog 低/高水位 64/192、在途上限 256；引擎 admission 仍为 16。`tiered_rolling` 的 `cohort_size=32` 只定义 family 调度单位，不能增加 32 请求全体完成屏障，不能按上一请求完成来决定下一请求发出时间。sender 的 `memory_reuse_ratio=0.5` 是工作集生成参数，与引擎按累计 eligible token 控制的 mock ratio 是两个概念，不能互作命中证据。

mock 请求结束后会释放 request Device 引用并异步写回 Host，因此不能要求 fill 产生并持续保留普通 Device-tree 高水位。`mock_memory` evidence policy 在 seed 后重新采集 baseline，不发送 fill 请求，直接启动 steady/adaptive 正式流量；完整、连续、拓扑一致且携带 mock 计数的采样构成压力覆盖。measurement/profile 窗口内必须同时观察到 `successful_h2d_blocks` 和 `d2h_completed_blocks` 增长，任一 `failed_h2d_blocks` 或 `d2h_failed_blocks` 增长都会使 migration 验收失败。它不使用也不伪造普通 transfer tree 的 `published_blocks`。未配置该策略的 production/smoke profile 缺省为 `tree_pressure`，仍执行原 fill、Device/Host 常驻、水位触发和普通 H2D/D2H publication 判据，行为不变。

需要先建立真实缓存压力、再测 mock 迁移时，使用新增的[混合 profile](../config/continuous_full_pressure_dsv4.tree_then_mock_memory.example.jsonc)，仍配对上述 reuse-only mock grid。它保留 `diagnostic_only=true` 和输出正确性无效标签，`formal_candidate=false`、原因 `diagnostic_only`，不能用于正式推理性能结论。可将上方命令中的 `--profile` 改为该文件，并使用新的结果目录。

`tree_then_mock_memory` 严格执行 seed → real fill → steady → measurement → drain；有尚未完成的采集时，measurement 后可进入 support，再 drain。block-size probe、seed、fill、steady、support 请求显式发送 `mock_memory_cache_request=false`，只有在传输边界被归类为 measurement 的请求发送 `true`。先完成 token/protobuf 准备，再在 RPC 前判定阶段并写入该布尔值；排队请求跨越 steady/measurement 或 measurement/support 边界时，以实际发送阶段为准。这样准备流量不进入 mock 累计块控制器。

混合策略不采用纯 mock 的 seed 后直接 steady 快捷路径，也不做 tiered rewarm（`rewarm.planned=0`）。fill 使用正常树分配和淘汰，建立真实 Device 高水位及 Host 占用；Host 达到 `fill_host_max_ratio=0.8` 时仍以 `fill_headroom_exhausted` 终止，不能为了跳过 rewarm 放宽余量。准备阶段的 readiness 和 measurement 压力覆盖均继续要求真实 Device/Host 驻留、水位触发及普通 D2H 发布。mock 计数的增长不能替代树压力。

measurement mock H2D 读取服务启动时已有的不可变随机 Host reservoir（seed 决定 source ring offset，payload 实际清零并预建 CRC），**不会读取某条由 fill 产生的指定 tree entry**。成功完成的 mock 请求执行真实异步 D2H 写回；它不改变 reservoir。迁移验收要求 measurement 窗口内 mock `successful_h2d_blocks`、`d2h_completed_blocks` 均有成功增量，并且 `failed_h2d_blocks`、`d2h_failed_blocks` 无增量，同时真实树压力覆盖通过；seed/fill/steady 的普通迁移与窗口外完成都不能充当该窗口的 mock 成功证据。计数必须带受支持的 `terminal_outcome_version=1`，按物理终态记录，不能用稍后的 settlement poll 把事件移入窗口。比例仍按块对齐的 eligible token blocks 累计计算，不按请求数计算。

验收必须保留 `profile.snapshot.json`、`grid.snapshot.json`、`cache_perf_launch.json`、原始指标、请求/采样 JSONL 和原始 timeline，并逐项记录：

1. 在 measurement 末尾读取同一 owner/controller 的累计 E/P，检查 `abs(P - floor(E * ratio))` 不超过一个有效复用块单元；窗口开始也保存累计基线，不能把窗口增量误当从零开始的目标。
2. 分开报告 `planned_ratio_ppm / 1e6` 与 `success_ratio_ppm / 1e6`（均以 eligible 逻辑块为分母）。`successful_h2d_blocks` / `failed_h2d_blocks` 是终态 load attempt 计数；重试可增加传输次数而不增加计划，须结合请求日志解释重试、失败、取消和窗口边界导致的差异。
3. pressure RPC 在 mock 模式下必须携带 feature completion/failure counters；measurement 内 `successful_h2d_blocks` 与 `d2h_completed_blocks` 均有正增量，`failed_h2d_blocks` 与 `d2h_failed_blocks` 均无增量。原始 CUDA timeline 中 admission 还必须有真实 Host-to-Device load，成功请求完成后有真实 Device-to-Host store；`d2h_submitted_blocks` 与 completed/failed 分开列出，D2H 计数是物理 Host records，实际字节/延时仍以原有 transfer 指标为准。
4. 初始 mock attribution 的 Device/Disk reuse 均为 0；`source_blocks` / `source_bytes` 和每池 `mock_source_ref_blocks` / bytes 整段流量恒定，无 source 插入普通树。
5. 请求发出时间和 backlog 证明在途压力持续，不能出现每 32 请求等待全体完成后才发下一组的空洞。
6. 保留既有 transfer 与 scheduler spans，分别分析计划、descriptor 构造、调度、H2D 等待/唤醒、D2H worker 和 model-forward 区间。当前 `cache.mock_memory.plan_and_descriptors` 是合并 span，需要结合其子调用/CPU 栈区分计划与 descriptor 开销，不能把整个 span 都算作控制器成本。

特性指标前缀为 `rtp_llm_mock_memory_cache_`：`eligible_blocks`、`planned_host_blocks`、`successful_h2d_blocks`、`failed_h2d_blocks`、`d2h_submitted_blocks`、`d2h_completed_blocks`、`d2h_failed_blocks`、`planned_ratio_ppm`、`success_ratio_ppm`、`source_blocks`、`source_bytes`。预留 bytes 包含 Host stride、CRC/footer 和对齐；source 指标不代表累计搬运量。启动时的 invalid-output warning、冻结诊断标记和原始传输证据均需保留。

此文档和离线契约测试不代表真实 DeepSeek V4 Flash 连续压测已验收。该运行按本次流程安排在整分支最终 review、push、远端 pull 后执行，再填入原始产物路径与上述检查结果。
