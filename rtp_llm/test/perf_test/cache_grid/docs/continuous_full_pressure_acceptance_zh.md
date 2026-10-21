# DeepSeek-V4-Flash 连续 FULL 压力验收记录（2026-10-21）

最终 production formal attempt 4 通过独立验收：728/728 自动seed、真实生产容量、60秒测量覆盖100%，测量内D2H/H2D双向发布及四门/aggregate均通过；修复前attempt 3虽然自报通过，独立容量审计判为失败并保留。

本记录对应 `refine-continuous-pressure-seeding` 的 Task 8。原始证据根目录为 `/data0/zhouhaiyu.zhy/tmp/task8-acceptance-20261021`。GPU 测试使用真实模型、CUDA13 Bazel/runfiles 服务和真实缓存迁移；Python 等价测试使用真实生成的 protobuf bindings。

## 源码、环境与输入身份

初始源码 HEAD 为 `35ca6cc961249029756620bce333daf774aaea11`（Task 7 文档提交），smoke 1/2、formal 1/2/3 从该版本启动。随后发现并以 TDD 修复三个产品问题，最终 GPU 复验从 `15cab2c2c9` 启动，包含下文列出的三个修复提交。精确argv均见各命令receipt，最终复验receipt还保存启动HEAD和dirty state；早期未单独记HEAD的尝试按该初始身份记录。工作区预先存在的无关变更如下，均保留，未纳入验收提交：

```text
 M rtp_llm/test/perf_test/cache_grid/README.md
?? .agents/
?? docs/superpowers/
?? openspec/
?? rtp_llm/cpp/cache/block_tree_cache/benchmark/docs/continuous_prefill_pressure_plan.md

```

原始身份快照为证据根目录的 `initial-state.json`、`model-identity.json`、`model-files.txt`、`protobuf-overlay-hashes.json`。模型路径 `/mnt/hf3fs/3fs/models/DeepSeek-V4-Flash` 存在 config 和 46 个 safetensors 分片，总计约 149 GiB。模型身份记录只对配置、索引、tokenizer、generation 文件计算 SHA-256，并记录全部权重分片名称及大小；未宣称对全部权重内容进行哈希。

| 输入 | SHA-256 |
| --- | --- |
| `/mnt/hf3fs/3fs/models/DeepSeek-V4-Flash/config.json` | `b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818` |
| `rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.example.jsonc` | `29e09b4badc36beee30ec88ce89e354d1715087340510ef249cb02832e8996b6` |
| production profile 初版（formal 1/2） | `996e6d36907e96b7a3c801cd49e4a396110b559dc02ffa5e0ea8fef8c86a7439` |
| production profile 当前版（formal 4，runtime reserve 65536 MiB） | `4f8ce6779feb05ec463a3a820dd2922b25cbc3349637d928ceaf95bdeb314692` |
| `rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.example.json` | `cf3d60708cf4f14be165aa67e0803217ae9a5c2ecf41bd641e7193cfc35f787d` |
| `rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.production.example.json` | `6813eaba7c87ea90250ef09cda91baa9d0a24bf0c9f8ab7d88c800d83422b5e1` |

硬件为 NVIDIA L20D（每卡 275040 MiB），驱动 580.105.08。初始 GPU 0–3 占用约 233 GiB，4–7 为 5/23/5/55 MiB，因此 smoke 和第一次 formal 使用 `CUDA_VISIBLE_DEVICES=4,5,6,7`。第二次 formal 因 GPU 4 后续外部占用，改用 `0,5,6,7`；运行前 GPU 0 占用 4758 MiB，其余三个仍为 23/5/55 MiB，均低利用率且有足够可用内存。这里没有独占 GPU 的保证，持续占用快照保存在 `formal-attempt-2-gpu-monitor.jsonl`。未终止外部进程。

受运行容器限制，普通沙箱 exec 在 bwrap pivot_root 阶段失败（`Operation not permitted`），命令通过已授权的沙箱外执行完成；这不改变测试或服务参数。

## Python 集成测试

原生环境缺少未签入仓库的生成 protobuf 模块；以下原样命令分别运行一次，失败证据没有覆盖：

```bash
python3 -m unittest \
  rtp_llm.test.perf_test.cache_grid.tests.perf_profile_test \
  rtp_llm.test.perf_test.cache_grid.tests.cache_perf_test \
  rtp_llm.test.perf_test.cache_grid.tests.cache_perf_pipeline_test \
  rtp_llm.test.perf_test.cache_grid.tests.pressure_seed_plan_test \
  rtp_llm.test.perf_test.cache_grid.tests.continuous_pressure_test \
  rtp_llm.test.perf_test.cache_grid.tests.pressure_diagnostics_test \
  rtp_llm.test.perf_test.cache_grid.tests.pressure_results_test \
  rtp_llm.test.perf_test.cache_grid.tests.result_schema_test \
  rtp_llm.test.perf_test.batch_decode_test_test -v
python3 -m unittest discover \
  -s rtp_llm/test/perf_test/cache_grid/tests -p '*_test.py' -v
```

Focused：454 个测试，3.249s，92 errors；全 cache-grid：475 个测试，4.252s，63 errors，均为缺少生成模块导致，零 skip。日志分别为 `focused-native.log`、`full-native.log`，准确 argv/退出码保存在对应 `*-command.json`。

随后使用已存在且 Task 7 记录过的 `/tmp/task-1-proto/{dash_sc,model_rpc}` 真实生成 bindings（非 mock）。证据根目录的 `unittest_real_proto.py` 只为两个 proto namespace 指定该生成路径，然后 `runpy.run_module("unittest", run_name="__main__")`。将上述 `python3 -m unittest` 替换为下列前缀，其他 unittest 参数完全相同：

```bash
python3 /data0/zhouhaiyu.zhy/tmp/task8-acceptance-20261021/unittest_real_proto.py
```

初始 Focused 454 个测试 5.293s 全部通过；全 cache-grid 475 个测试 6.077s 全部通过，均零 skip。三个修复全部完成后，同样命令和真实 bindings 再验：focused 457 个测试 5.583s、full 478 个测试 5.916s 全部通过、零 skip；日志和完整 argv 分别为 `focused-after-fixes*`、`full-after-fixes*`。证据为 `focused-real-proto.log`、`full-real-proto.log` 及对应命令 JSON。overlay、生成文件与 wrapper 哈希见 `protobuf-overlay-hashes.json`。

本次另行执行 `tools/cache_perf` 的顶层/run/profile/resume `--help` 和两个文档 dry-run，六条命令均 exit 0。production 静态 formal_candidate=true，smoke=false 且含五个静态排除原因；全部文档公共 flag 均见实际 help。准确 argv/日志为 `offline-command-checks.json` 及 `*-help.log`、`formal-dry-run.log`、`smoke-dry-run.log`；`offline-output-audit.json` 记录复核。外部证据 helper 首次用 json.loads 解析 dry-run 输出时因 JSON 后还有渲染的 Bazel 命令而报 Extra data；已改为 raw_decode 解析同一原始输出，不是 CLI 失败或重跑。

## GPU 命令和全部尝试

smoke 只将资产默认设备改为可用卡组，未修改时长或门槛：

```bash
python3 tools/cache_perf run \
  --profile rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.example.jsonc \
  --grid rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.example.json \
  --result-dir /data0/zhouhaiyu.zhy/tmp/dsv4-fifo-pressure-smoke-20261021 \
  --env CUDA_VISIBLE_DEVICES=4,5,6,7 \
  --pressure-mode continuous --pressure-preload \
  --pressure-client-concurrency 2 --pressure-seed-count 2 \
  --pressure-request-rate 0 --pressure-arrival constant \
  --pressure-measure-seconds 10
```

smoke attempt 2 的唯一 argv 差异是 result-dir 后缀 `-attempt2`；正式命令为：

```bash
python3 tools/cache_perf run \
  --profile rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.production.example.jsonc \
  --grid rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.production.example.json \
  --result-dir /data0/zhouhaiyu.zhy/tmp/dsv4-fifo-pressure-formal-20261021 \
  --env CUDA_VISIBLE_DEVICES=4,5,6,7 \
  --pressure-mode continuous --pressure-preload \
  --pressure-client-concurrency 16 \
  --pressure-request-rate 0 --pressure-arrival constant \
  --pressure-measure-seconds 60
```

formal attempt 2 的唯二 argv 差异为 result-dir 后缀 `-attempt2`、设备 `CUDA_VISIBLE_DEVICES=0,5,6,7`。正式运行没有 `--pressure-seed-count`，未使用合成容量上限，未缩小fill、复用长度、并发、测量时长或有效性门槛。

| 尝试 | 本地时间 | 退出 | 结果与证据 |
| --- | --- | --- | --- |
| smoke 1 | 18:45:04–18:58:49 | 3 | seed/fill/steady 完成，measurement phase 控制 RPC 1s deadline 超时；无 measurement 请求；最终四 rank disable 和服务清理完成。`smoke-attempt-1.log`、`smoke-attempt-1-process.log`、`smoke-attempt-1-bazel/`。 |
| smoke 2 | 19:01:29–19:05:31 | 0 | 完全相同参数热缓存重试，完成 measurement/drain/disable；47 个请求全部成功，正确标为非正式。`smoke-attempt-2.log`、`smoke-attempt-2-bazel/`。 |
| formal 1 | 19:06:27–19:09:51 | 3 | 启动 rank 0 SIGABRT，尚无 epoch。GPU 4 外部占用增至约 213 GiB；rank 0 reserved 10272 MiB < runtime 13705 MiB，`MemoryEvaluationHelper.cc:131`。 peers reserved 约 223 GiB，退出后 GPU 4 仍高占用。`formal-attempt-1-process.log`、`formal-attempt-1-bazel/`、`nvidia-during-formal-attempt-1.txt`。 |
| formal 2 | 19:13:15–19:17:33 | 3 | 服务启动并取得 live startup，首次 block probe 即四 rank CUDA OOM。40 GiB workspace 大于剩余 12.96–17.83 GiB；表层 reuse_len=0 preflight 失败为后果。原始 startup/preflight、`formal-attempt-2-process.log`、`formal-attempt-2-bazel/` 均保留。 |

smoke 1 根因追踪：四 rank enable/seed/fill/steady ACK 成功；measurement 控制从单调时钟 1642027.992618 开始，到 1642028.994313 报 DEADLINE_EXCEEDED。rank 0 step 21 forward 区间 1642026.708191373–1642029.999031026，耗时 3.290840s，跨越了硬编码 1s 控制预算。Python phase RPC 与 runner 以及 native ACK 路径均有 1s 上限；控制在 engine step 的安全边界应用。drain 随后失败，最终 disable 成功。未更改该控制超时契约；一次不变参数热缓存重试通过仅说明未复现，不证明该潜在时延问题已修复。启动中 fastsafetensors shm errno 22、DeepGEMM MLOPart localization 子进程断言均保留在日志，服务继续初始化，不应误报为本次 phase 超时或 formal 内存失败的根因。

## 修复前 smoke 的独立复算（历史证据）

成功 smoke epoch：`/data0/zhouhaiyu.zhy/tmp/dsv4-fifo-pressure-smoke-20261021-attempt2/pressure_epochs/6d2b0e2836de4d20a5148b6d89334e9c`。失败 smoke epoch 同根目录无 `-attempt2`，epoch 为 `5b46b4f873664efeaf13c412184b0fdb`。

原始 startup/manifest 显示 FULL group 1、shared TP owner rank 0，参与 rank 0/1/2/3；每个参与者的 service_instance_id 保存在 `smoke-attempt-2-audit.json`，不是独立四组容量。owner 设备 pool_4(csa_kv)、pool_5(indexer_kv)、pool_6(hca_kv) 各 127 个已经扣除保留后的 allocatable logical blocks；不能再次扣 1，也不能把三个 pool 或四个 TP 视图相加。host pool_7 可分配/启动可用均为 75 blocks。旧快照报告 block span 512 tokens，但 formal 3 后独立证明 CP4 实际全局跨度应为 2048；下段仅复述旧 plan 的可重算历史数据，不能据此证明真实 bank 超过 GPU 容量。新版本 smoke 另行复验。

复用 32768 tokens 对应每 family B=64 blocks；seed input 36864 包含独立 4096 commit tail。GPU family capacity=min(floor(127/64))=1，GPU low families=min(floor(floor(127×.82)/64))=1，host families=floor(75/64)=1，所以可行闭区间 [2,2]，手动选择 2。planned/submitted/completed=2，failed/cancelled/pending=0，seed max_inflight=2。fill 16384 tokens、客户端深度 2、人工 seed、合成容量等均为非正式配置。

测量窗口 `[1642425.544807571,1642435.544807571)`；原始 occupancy 为 effective device 104..120（门槛 102），host resident 60..68（门槛 19）。独立从相邻原始快照及全部 TP freshness/clock 区间重建覆盖率 1.0、最长无效 0s，与 summary 相符。保守完整落在窗口内的 owner D2H 已发布增量为 50 blocks（16+17+17），H2D LOAD 为 0；不将 seed/fill/steady 增量用于 measurement。

四个门：execution=true、FULL pressure=true、formal configuration=false、bidirectional migration=false，aggregate formal acceptance=false。非正式原因完整为 `manual_seed_count`、`client_inflight_below_16`、`synthetic_test_block_capacity`、`nonstandard_fill_target`、`explicit_smoke`、`grid_reuse_below_32768`；双向门另有 `missing_h2d_load:0:1`。

独立审计脚本为证据根目录的 `audit_pressure.py`、`audit_raw_coverage.py`，只读 JSON/JSONL，不导入产品 planner/diagnostics 计算代码。前者重算 owner/TP 容量、family 上下界、请求提交/完成计数与并发时间线，以及窗口内已发布增量；后者从原始 startup 与 cache_samples 重建 owner 压力谓词、参与者时钟/新鲜度覆盖区间并求交。两次 smoke 的 receipt 为 `smoke-attempt-N-audit.json`、`smoke-attempt-N-raw-coverage.json`；包含原始文件 SHA-256。失败 smoke 虽覆盖率合格，execution=false，仍不能验收。

## 正式部署修正与运行证据

formal 2 的真正首因位于 `models_py/modules/dsv4/prefill_workspace.py:124`。最大 Q rows=524288（每 rank 的 131072/CP4×context16），Q dim=64×512=32768，BF16 Q 需要 32 GiB；CP full rows=2097152，main width=2048、index width=512，双份 gather/restore 的 FP32 region 为 `2×2097152×(2048+512)×4=42949672960` bytes，即 40 GiB。union 取两者最大并按 1 GiB 对齐，每次 forward 申请该 40 GiB，包括仅 8192 tokens 的 probe。服务未在缓存分配前进行 forward warmup（warmup consumed=0）；runtime reserve 的 8192 MiB 仅被提高到总显存 5%=13705 MiB，因此实际缓存过多，无法容纳模型 runtime workspace。这不是复用对齐真的变为 0。

formal 2 startup 按旧快照复算（`formal-attempt-2-startup-audit.json`，其 512 span 亦受后述 CP 缺陷影响）：FULL owner rank0/group1，TP0–3；pool_4/5/6 各30361 allocatable blocks，host9838 available，span512，R65536/B128，GPU237、LOW194、HOST76，可行区间[238,270]。这是仅 startup 的复算；没有生成 seed_plan、提交 seed 或 measurement，不能当作完成的正式计划。

formal 3 先仅通过外部 profile 调整部署预留运行时内存：外部 profile `continuous_full_pressure_dsv4.production.runtime64g.jsonc` 仅把 `engine_args.--reserver_runtime_mem_mb` 从8192改为65536 MiB，SHA-256 `4f8ce6779feb05ec463a3a820dd2922b25cbc3349637d928ceaf95bdeb314692`。差异和 workspace 算式保存为 `formal-runtime64g-substitution.json`；该变体 dry-run 仍为 formal_candidate=true，无排除原因（`formal-runtime64g-dry-run.log`）。这让 cache 根据实际剩余显存自动分配，未加入 test_block_num、fixed pool 或人工 seed 数量，未降低正式压力、复用长度、fill、客户端深度或测量门槛。

formal 3 完整 argv 见 `formal-attempt-3-command.json`：与上面 formal 2 命令相比只把 `--profile` 换为该外部文件、`--result-dir` 换为 `/data0/zhouhaiyu.zhy/tmp/dsv4-fifo-pressure-formal-20261021-attempt3`。该次 19:21:07–19:46:51 exit 0；epoch 为 `1e2e3dbe60b7489aab1714efa92d63a1`。609 请求均成功（seed177、fill324、steady19、measurement89），seed177/177、max_inflight16、251.663236s；测量 `[1644844.631207333,1644904.631207333)`，73 次窗口内完成、16 次跨边界，独立 coverage1.0/最长无效0s，D2H1816/H2D3168 blocks，自报四门及 aggregate 均 true。**独立容量审计拒绝本次正式验收**，因为177 seeds 未超过真实GPU705 families。原始数据、Bazel PASS 和错误自报结果全部保留，不能代替修复后的正式运行；`formal-attempt-3-independent-verdict.json` 明确 effective_formal_acceptance_valid=false。


## Task 8 中发现并修复的缺陷

- `3e91df0b2f`：Dash-SC 空流或只有非 infer_response 的流不再返回 success=true，避免原生错误被后续 reuse mismatch 掩盖。新增用真实生成 response 的回归先失败（`empty-stream-red.log`），最小修复后 continuous-pressure 85 测试通过（`empty-stream-green.log`），仅两个相关文件提交；hook/staged diff 检查通过。
- `6a771bd083`：生产 profile 的 runtime reserve 改为65536 MiB；真实 launch snapshot 的 workspace+runtime 预算回归先失败（8 GiB < 56 GiB，`runtime-reserve-red.log`），修正后53个 cache_perf 测试通过（`runtime-reserve-green.log`），仅 profile 与其测试提交。当前生产 profile SHA-256 与外部64 GiB变体相同：`4f8ce6779feb05ec463a3a820dd2922b25cbc3349637d928ceaf95bdeb314692`。
- 直接执行基础 Python 的 `-m black --check` 因环境未安装 black 而失败；随后通过仓库 pre-commit 的隔离 Black/isort/格式 hook 验证成功（`empty-stream-hooks.log`、`runtime-reserve-hooks.log`）。提交 hook 正常暂存/恢复无关 README 等未暂存内容。

formal 3 另发现 CP4 owner 几何报告缺陷，故无论其 summary 自报 gate 如何，该次都不能作为容量验收通过证据。`formal-attempt-3-cp-geometry-audit.json` 在235个稳定 seed-only快照中逐项对齐已完成请求、owner及全部参与者：owner FULL cache_ref 精确等于 `34×completed_seed+6`（6为probe基线），当前 request_ref=34，其他TP参与者本地计数为0。input69632/virtual span2048=34；可复用65536/2048=32，不是序列化plan按physical span512算出的128。native启动同时明确记录CP size4、physical512、virtual2048。因此22584个owner物理块实际能容纳705个family，正式下界应706，LOW578+HOST307得到上界885，177个seed没有超过真实family容量。

根因定位在 `KVCacheManager::getCachePressure` 原先直接转发 BlockTreeCache 的物理 group seqSizePerBlock；实际分配使用 CPSlotMapper 的 CP virtual block。提交 `15cab2c2c9` 在 manager 快照边界调用已有 policy-aware `logicalSeqSizePerBlock(config, tag)`，保持 pool capacity/occupancy 为物理块，改为报告全局请求 token 跨度。原生红灯先证明 CP4 快照512、实际8192-token请求只分配4块（期望2048跨度）；修复后定向目标通过。完整 CP 目标 `test.xml` 记录7个实际运行测试（含新增 CP4/非CP物理容量/真实分配回归）、4个已有 DISABLED、0 failures/errors；没有把已有禁用项说成新行为跳过。命令/日志/测试 XML 在 `cp-geometry-native-{red,green,full}*`，结构化 receipt 为 `cp-geometry-regression-receipt.json`。Python新增真实2048几何fixture验证 B32/GPU705/LOW578/HOST307/区间[706,885]，`geometry_source=owner_group` 不再意味着物理 token span。

## 修复后 smoke 复验

smoke 3 于19:56:02–20:00:58完成，exit0，源码 `15cab2c2c995ebf91b54d7166c9e98f21ac85790`，原资产、时长、人工seed数及门槛不变；仅使用当前可用的 `0,5,6,7`。完整命令为：

```bash
python3 tools/cache_perf run \
  --profile rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.example.jsonc \
  --grid rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.example.json \
  --result-dir /data0/zhouhaiyu.zhy/tmp/dsv4-fifo-pressure-smoke-20261021-attempt3 \
  --env CUDA_VISIBLE_DEVICES=0,5,6,7 \
  --pressure-mode continuous --pressure-preload \
  --pressure-client-concurrency 2 --pressure-seed-count 2 \
  --pressure-request-rate 0 --pressure-arrival constant \
  --pressure-measure-seconds 10
```

新 epoch 为 `8f5dd51209cc427d9fc7987c0e4193ab`。FULL owner0/group1，TP0–3为同一shared owner；设备pool_4/5/6各127 blocks，host75，真实全局span2048，R32768/B16，GPU7/LOW6/HOST4，正式闭区间[8,10]。手工smoke仍选2，明确没有超过GPU容量；这是允许的non-formal smoke，不把它当正式bank证明。seed planned/submitted/completed=2/2/2，failed/cancelled/pending=0，max_inflight2、1.184294s。

45个请求全成功：seed2/fill12/steady10/measurement21。测量 `[1645754.042664793,1645764.042664793)`，19次窗口内完成、2次跨界。独立原始覆盖1.0/最长无效0s；device effective102..120（阈值102），host60..67（阈值19）；窗口内D2H发布49 blocks，H2D0。四门依次true/true/false/false，aggregate=false；六条non-formal原因与smoke2相同，迁移另含missing_h2d_load。完成drain、四rank disable和所属服务清理。

证据为 `smoke-attempt-3-command.json`、`smoke-attempt-3-bazel/`、`smoke-attempt-3-process.log`、`smoke-attempt-3-audit.json`、`smoke-attempt-3-raw-coverage.json`。额外占用审计helper首次断言没有稳定seed样本而失败，记录于 `smoke-attempt-3-cp-audit-first-attempt.json`；两seed阶段仅1.18秒，不能保证1秒采样避开completion±0.2秒边界。修正helper仅对manual smoke报告“采样不足/占用验证不确定”，没有更改产品或正式门槛；`smoke-attempt-3-cp-geometry-audit.json` 保留null结论。自动production审计仍强制至少10个可匹配稳定样本。原生CP回归和正式长seed段分别提供几何证明。

原生验证的完整公共命令如下；红灯在修复前运行，同一命令修复后绿灯。完整CP回归只去掉 `--test_filter`，其余参数一致。

```bash
CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc \
CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ \
CUDAHOSTCXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ \
NVCC_PREPEND_FLAGS=-ccbin=/opt/rh/gcc-toolset-12/root/usr/bin/g++ \
PATH=/opt/rh/gcc-toolset-12/root/usr/bin:/opt/conda310/bin:/usr/local/cuda-13.2/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
/usr/local/bin/bazelisk test \
  //rtp_llm/cpp/cache/test:kv_cache_manager_cp_test \
  --config=cuda13 --test_output=errors --nocache_test_results \
  --test_filter=KVCacheManagerCPSlotMapperTest.PressureGeometryMatchesGlobalRequestTokensAndPhysicalCapacity \
  --test_env=CUDA_VISIBLE_DEVICES=0 --test_timeout=600
```

以上编译环境同样用于修复后GPU启动，精确环境见 `native-build-environment.json`。当前CUDA13 `libth_transformer.so` 大小1505081568 bytes，SHA-256 `2c44e65a12d119cd3f450d30caf53723db3ec6c1775786e25019e31958276aff`；解析后的实际路径、HEAD、dirty state及全部资产hash见 `after-fixes-source-input-identity.json`。

## 修复后 production formal（attempt 4）

使用已签入、可直接运行的production资产；此次不依赖外部profile变体，唯一设备覆盖为 `CUDA_VISIBLE_DEVICES=0,5,6,7`。启动前四卡显存分别约124/23/5/55 MiB、利用率0；15秒持续快照见 `formal-attempt-4-gpu-monitor.jsonl`。完整命令为：

```bash
python3 tools/cache_perf run \
  --profile rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.production.example.jsonc \
  --grid rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.production.example.json \
  --result-dir /data0/zhouhaiyu.zhy/tmp/dsv4-fifo-pressure-formal-20261021-attempt4 \
  --env CUDA_VISIBLE_DEVICES=0,5,6,7 \
  --pressure-mode continuous --pressure-preload \
  --pressure-client-concurrency 16 \
  --pressure-request-rate 0 --pressure-arrival constant \
  --pressure-measure-seconds 60
```

源码 `15cab2c2c995ebf91b54d7166c9e98f21ac85790`；epoch `/data0/zhouhaiyu.zhy/tmp/dsv4-fifo-pressure-formal-20261021-attempt4/pressure_epochs/b4f511dfee334cf58f439f105e4753ce`。launch、profile、frozen settings和native启动参数均无正数test_block_num；计划明确test_block_num=0、source=automatic、explicit_smoke=false、nonformal_reasons=[]。正式grid最大复用65536、commit tail4096、seed input69632、fill102400，客户端上界16。

startup FULL owner0/group1，TP0–3共享一套owner；`csa_kv`/`indexer_kv`/`hca_kv`（pool_4/5/6）各23278 allocatable physical blocks，host pool_7 total/available9838。保留块已在native total中扣除，不能再减1；不把三个关联池或四个TP视图相加。owner和所有participant的service_instance_id见原始startup及audit receipt。

独立容量复算：全局block span2048，B=ceil(65536/2048)=32，GPU=min(floor(23278/32))=727；LOW=min(floor(floor(23278×.82)/32))=596；HOST=floor(9838/32)=307；所以合法区间[728,903]，自动选择728。728×32=23296块的可复用bank严格大于设备23278块，并小于LOW+HOST支持的903 families。geometry_source=owner_group；native每rank启动CP4、physical512、virtual2048，不能另行再除CP。

运行中独立占用早期审计的72个稳定seed快照已逐项验证：owner三个FULL池cache_ref都为 `34×completed_seed+6`，其他三个participant本地计数为0；34=69632/2048，6是probe留下的基线。最终关闭文件的审计扩展到100个稳定快照，逐owner/device pool和全部peer均匹配；`formal-attempt-4-cp-geometry-audit.json` 的 occupancy_contract_valid=true，源文件SHA一并保存。


attempt 4 于20:01:19–20:23:36完整结束，exit0，Bazel PASS（1334.9s；外层1336.983784s）。seed planned/submitted/completed=728/728/728，failed/cancelled/pending=0，耗时1021.629112s，max_inflight16。独立按全部dispatch/completion时间重建初始16-way供给和712次补充，每一次补充时仍有15个旧请求在途；family dispatch顺序恰为0..727，没有batch drain barrier或padding。

839个请求全部成功：seed728、fill1、steady20、measurement90。seed bank本身已触发真正GPU高水位和host驻留，因此只需要1次额外fill；102400-token形状和正式门槛没有缩短。seed/steady/measurement峰值inflight均为16，fill峰值1。测量窗口为 `[1647050.111760613,1647110.111760613)`，恰60秒；74次窗口内成功完成、16次跨界，按窗口计算吞吐1.233333请求/秒。

独立从1096条完整四rank原始采样重建窗口覆盖率1.0、最长无效0s；device effective19139..20935（门槛floor(23278×.82)-2=19085），host published resident4857..6659（门槛ceil(9838×.25)=2460）。设备高水位触发及真实D2H发布证据存在，HOST/DISK触发未被替代使用。

必需FULL owner group只有0:1，窗口内保守相邻快照的完整时钟包络都落在窗口内，且全部TP参与者拓扑/实例、新鲜度、序列和gap有效。成功DEVICE→HOST STORE/EVICT发布1866 blocks（1个区间），HOST→DEVICE LOAD发布3168 blocks（26个区间），均与summary逐endpoint差分一致。H2D计数为关联device pool endpoint的已发布块增量之和，不再乘TP数；初始化和跨窗口区间均未计入。

| 判定 | 结果 |
| --- | --- |
| execution_valid | true |
| full_pressure_valid | true |
| formal_config_valid | true |
| bidirectional_migration_valid | true |
| formal_acceptance_valid | true |
| 独立 effective_formal_acceptance_valid | true |

所有gate reasons为空。enable、seed/fill/steady/measurement/drain、disable共7次控制均收到rank0/1/2/3的有效ACK（applied时间为正），measurement ACK耗时0.893769s；这次没有复现先前1s控制超时。原始ACK保存在manifest.control_times.diagnostics，独立receipt逐项保留。最后所属服务SIGTERM、GPU release done；运行后设备0/5/6/7回到124/23/5/55 MiB基线，未清理外部作业。1s控制预算的潜在冷启动时延问题仍如smoke1记录，不能由热重试通过推断其已修复。

最终receipt及原始文件：

| 文件/目录（均在证据根目录下，除注明者） | 内容 |
| --- | --- |
| `formal-attempt-4-command.json`、`.log`、`-bazel/`、`-process.log` | 精确argv、启动HEAD/dirty state、时间/退出码、完整服务/Bazel日志 |
| `/data0/zhouhaiyu.zhy/tmp/dsv4-fifo-pressure-formal-20261021-attempt4/cache_perf_launch.json`、`profile.snapshot.json`、`grid.snapshot.json` | 实际启动、环境和冻结输入 |
| 上述epoch的 `startup_pressure.json` | 原始实际owner容量、实例、TP拓扑和全局token几何 |
| 上述epoch的 `pressure/manifest.json`、`summary.json` | 完整plan各term、seed outcome、四门与aggregate |
| 上述epoch的 `pressure/requests.jsonl`、`cache_samples.jsonl`、`steps.jsonl` | 所有请求、逐rank原始RPC/时钟边界、阶段事件 |
| `formal-attempt-4-audit.json`、`-raw-coverage.json`、`-cp-geometry-audit.json` | JSON-only独立容量/请求/迁移、覆盖、实际CP占用复算及原始SHA |
| `formal-attempt-4-independent-verdict.json` | 28项检查全部true；独立最终接受结论、rolling时间线和全部控制ACK |
| `formal-attempt-4-gpu-monitor.jsonl`、`nvidia-after-formal-attempt-4.txt` | 每15秒占用与最终释放基线 |

复算命令如下；脚本只读JSON/JSONL及native日志，不导入产品planner/diagnostics实现：

```bash
EVIDENCE=/data0/zhouhaiyu.zhy/tmp/task8-acceptance-20261021
FORMAL_ROOT=/data0/zhouhaiyu.zhy/tmp/dsv4-fifo-pressure-formal-20261021-attempt4
python3 "$EVIDENCE/audit_pressure.py" "$FORMAL_ROOT" "$EVIDENCE/formal-attempt-4-audit.json"
python3 "$EVIDENCE/audit_raw_coverage.py" "$FORMAL_ROOT" "$EVIDENCE/formal-attempt-4-raw-coverage.json"
python3 "$EVIDENCE/audit_cp_geometry.py" "$FORMAL_ROOT" "$EVIDENCE/formal-attempt-4-process.log" "$EVIDENCE/formal-attempt-4-cp-geometry-audit.json"
python3 "$EVIDENCE/final_acceptance_verdict.py" "$FORMAL_ROOT" formal-attempt-4
```

额外格式检查失败也未隐藏：CP native新测试的首次pre-commit clang-format修改了排版（`cp-geometry-hooks-pre-red.log`），格式化后同组hook和提交hook均通过；这不是native回归失败。所有工作均保留初始无关README、.agents、docs/superpowers、其他OpenSpec文件及benchmark计划，不纳入本次acceptance提交。


## 最终离线检查与任务状态

当前签入production profile再次按上述formal命令加 `--dry-run` 验证，exit0、formal_candidate=true、reasons=[]；完整argv/输出为 `formal-checkedin-final-dry-run-command.json`、`.log`。这验证实际默认资产，无需未公开的64 GiB覆盖。最终运行：

```bash
openspec validate refine-continuous-pressure-seeding --strict
git diff --check
git status --short
```

严格OpenSpec验证通过（`openspec-final-validation.log`），diff无空白错误。6.1依据457项focused和478项full等价测试，6.2依据修复后明确non-formal smoke，6.3/6.4仅在attempt4独立verdict通过后勾选。验收提交只包含本报告、reference和该change的tasks.md；提交前检查完整staged diff及staged whitespace，仓库hook通过后提交。
