# Cache Grid 性能工具

Grid 只接受 `schema_version: 2` 的显式 `cases`，不在运行时展开采样模板。
`case_id/batch_size` 必填；单请求还必须填写 `input_len/cache_len`；
分组请求必须填写 `prefix_policy` 和每组的 `count/input_len/cache_len`。
运行时缓存对齐只读取 grid 的正整数 `generator.cache_alignment`，
旧 CLI/profile 覆盖项和 `generator.cache_sampling.alignment` 会报错。
生成器的 `--cache-alignment` 或 profile `cache_grid.cache_alignment` 仅用于生成 grid。

本目录集中管理 cache-grid 测试、绘图与公式工具。原来的
`//rtp_llm/test/perf_test:cache_grid_perf_test` Bazel target 不变；Python 模块和脚本统一使用本目录下的新路径。
新的实现、测试与文档请添加在本目录，不再散落到 perf_test 根目录。

```text
cache_grid/
├── runner/    # run/resume/retest/profile、网格生成、随机 batch 和 pipeline
├── plot/      # 静态/交互图、batch 图、生产流量 TTFT/TPM 图与报告
├── formula/   # 公式拟合、输入验证与异常分析
├── config/    # profile 加载器、JSON/JSONC 配置模板
├── examples/  # grid 示例与详细参数说明
├── tests/     # 单元测试及 HTML 交互回归
└── docs/      # batch 分组与随机 batch 使用说明
```

`batch_decode_test.py`、`server.py` 等供其他 perf 测试共用的入口仍在父目录；不迁移 Dash-SC、模型通信等非 cache-grid 通用代码。
父目录的旧 Python 转发入口、profiles/examples 软链接和跳转文档已删除；实现、配置和示例只保留本目录中的一份。

## 1. 首次配置：主要修改哪里

复制 [config/local.example.jsonc](config/local.example.jsonc) 为本机配置。
模板已逐项注释，重点检查：

1. `engine.model_type`、`model_label`、`engine.checkpoint_path` 和 `engine.tokenizer_path`：模型在测试容器内的实际路径。
2. `runtime_env`：CC/CXX、CUDA、PATH/LD_LIBRARY_PATH、三个可写 JIT 缓存目录。
3. `bazel.output_base`：本机 Bazel 输出目录，可删除该项使用默认值。
4. GPU 拓扑变动时检查 TP/EP/DP/world 和 WORLD_SIZE；cache block 与 seed tail 必须匹配实际模型和 CP 配置。
5. 测试规模由独立 grid 文件控制，正式轮数用 `cache_grid.measure_runs` 或 `--runs`。

标准 `.json` 仍严格遵守 JSON；`.jsonc` 支持 `//` 和 `/* … */` 注释。
不支持尾逗号，也不展开 `$VAR`、`${VAR}`、`~` 或执行命令；路径请填写明确值。
字符串中的 URL、`//`、转义引号不会当作注释。注释不参与 profile 指纹。
不要在配置中写密码、API key 或 token；敏感内容必须通过组织认可的外部方式注入。

## 2. 测试操作与 pipeline

以下从仓库根目录运行；入口仍然执行 `bazelisk test`，不会绕过 Bazel 构建。
可用 `python3 tools/cache_perf` 替代直接执行。

```bash
# 首次运行：必须是空/新结果目录。先加 --dry-run 检查最终命令。
./tools/cache_perf run \
  --profile /absolute/path/local.jsonc \
  --grid /absolute/path/grid.json \
  --result-dir /absolute/path/results \
  --dry-run

# 确认后去掉 --dry-run 执行。

# 续测：要求 checkpoint 存在；读取已保存快照，不读取后来改动的源 profile。
./tools/cache_perf resume --result-dir /absolute/path/results

# 只重新测指定 case 的正式耗时，不开 profiler。
./tools/cache_perf retest --result-dir /absolute/path/results \
  --cases 2077,4217 --runs 3

# 只对指定 case 补采 trace，不续跑正式网格。
./tools/cache_perf profile --result-dir /absolute/path/results \
  --cases 2077 --runs 1 --trace-timeout 180
```

| 模式 | 正式测量 | Profiler | 输出 |
|---|---|---|---|
| run | 全 grid | 关闭 | 指定的新目录 |
| resume | checkpoint 中未完成 case | 保留原运行配置 | 原目录 |
| retest | 只选定 case，默认原测量轮数 | 关闭 | `cache_perf_replays/retest_<id>/` |
| profile | 跳过 | 只选定 case，默认一次 | `cache_perf_replays/profile_<id>/` |

retest/profile 会保留原 case_id 和请求分布，但使用新结果目录；不会改原 checkpoint 或报告。
简化入口的 profile 输出统一在一层隔离目录下：

```text
cache_perf_replays/profile_<id>/
├── manifest.json          # 会话 ID、case、轮次、录制状态、trace 相对路径
├── test_info.json
├── profile.snapshot.json
├── grid.snapshot.json
├── cache_perf_launch.json
└── timelines/             # 各 TP rank 的 trace JSON
```

入口自动传递内部参数 `--cache_profile_flat_output`，底层不再创建重复隔离目录。
隐藏标记 `.cache_profile_started` 防止同一目录被重复或并发使用；失败后再次执行 profile 会创建新目录。
直接使用原 Bazel `--cache_profile_only`（不传内部参数）仍保留原隔离布局；已有历史结果不移动、不删除。

case_id 来自同一个原始 grid，不要拿另一份 grid 的 ID 混用。
未知或被对齐去重删除的 ID 会在启动 Bazel/模型前报错。
当前 profile 仅支持未分组 batch=1，不支持 shared seed；profiler 耗时不能替代正式性能数据。
重测/补采是重新构造同形状输入和 seed，不是原 token 序列逐字回放。

所有模式支持 `--dry-run`：打印操作模式、case 数、环境变量、输出位置和实际 Bazel 命令，不创建结果文件、不启动 Bazel、不占 GPU。
resume 显示原计划 case 总数，实际剩余 case 由底层 checkpoint/journal 决定。

## 3. 环境变量和优先级

新入口按 `--env KEY=VALUE > JSON engine_env/runtime_env > 允许继承的外部环境` 合并。
同一变量不能同时出现在两个 JSON 环境段。所有值用字符串或数字，不用 JSON 布尔值。

```bash
./tools/cache_perf run --profile /path/local.jsonc --grid /path/grid.json \
  --result-dir /path/new-results --env DSV4_CHUNK_TOKENS=8192
```

- `runtime_env` 在启动 Bazel 前生效，同时转成 `--test_env` 传入测试进程；因此编译器、动态库路径不用反复手写。
- `engine_env` 同样作为 `--test_env`，并注册到 runner 的环境记录和 run_config 指纹。
- 启动时冻结两类环境。resume 不接受 profile/grid/runs/env 覆盖，避免混合不同配置测量。
- 允许继承的变量包括 PATH/LD_LIBRARY_PATH、编译器、CUDA_VISIBLE_DEVICES，以及 DSV4_/PERF_/DG_JIT_/TRITON_/TILELANG_ 等性能相关变量。最终值在摘要和快照中可查看。
- Bazel 参数可放 `bazel` 段，或由 `--config`（可重复）、`--output-base`、`--bazel` 覆盖。

直接使用 `bazelisk test --test_arg=--profile=...` 也支持 JSONC 和环境默认值，
但只能影响测试进程；不能反向改变已经启动的 Bazel 的编译环境。
直接 Bazel 入口使用“已有进程环境/--test_env 优先于 --engine_env 默认值，后者优先于 profile 默认值”的优先级。
要统一控制构建和测试环境，应使用新入口。

## 4. 快照、旧结果和安全边界

run 保存 `profile.snapshot.json`、`grid.snapshot.json`、`cache_perf_launch.json`。
统一入口的新结果采用以下分工：

| 文件 | 内容 |
|---|---|
| `profile.snapshot.json` | 用户原始 profile |
| `grid.snapshot.json` | 测试输入 grid |
| `cache_perf_launch.json`（v2） | 启动参数、有效启动环境、Bazel 配置、快照引用与 SHA256 |
| `test_info.json`（v4） | 状态、时间、尝试次数、少量模型摘要、配置文件引用及指纹 |
| `cache_grid_results.json` | 测量结果和实际生效的 `run_config`，供独立分析与续测校验 |

启动清单不再内嵌 profile，也不再将环境值重复写入 `runner_args`；加载时从 `env` 重建参数。
`test_info.json` 不再重复保存 argv、引擎参数、环境变量、完整 profile 和 resume_config。
原始 profile 与实际生效环境仍分开保存，因为 CLI 覆盖和运行时调整可能改变值。
正常入口只读取 v2 启动清单及其快照；旧 v1 清单或只有旧 test_info 的目录必须先显式迁移。
底层直接 Bazel 测试仍写完整 v3 test_info；它是直接测试的当前元数据，不作为统一入口的隐式启动配置。
精简版 test_info 必须与启动清单、快照一起保留，不能单独作为恢复配置。

resume 校验快照 SHA256，再交由原 runner 的 grid/profile/run_config 守卫验证；不会关闭不匹配检查。
运行失败但尚未产生 checkpoint 时，不要用 resume 猜测恢复；检查错误后使用新结果目录重试。

历史数据使用独立的 [迁移工具](../../../../tools/migrate_cache_perf)。它只写新文件或新目录，不会改写原始数据，也不会启动模型：

```bash
# 旧结果 JSON 或观测 CSV → 当前结果文件（输出文件必须不存在）
python3 tools/migrate_cache_perf results --input /path/old.json --output /path/current.json

# 旧运行目录 → 冻结的 v2 启动快照、v4 元信息、当前结果
python3 tools/migrate_cache_perf run --input-dir /path/old-run --output-dir /path/migrated-run

# 旧共享 base.txt → 每个 case 的显式 seed 记录
python3 tools/migrate_cache_perf case-store --input-dir /path/old-store --output-dir /path/current-store

# 从迁移后的配置启动新的重测
python3 tools/cache_perf retest --result-dir /path/migrated-run --cases 2077 --runs 3 --dry-run
```

JSON 迁移产物或迁移报告记录来源文件的 SHA256；CSV 输出仍使用 `.csv`，来源路径与 SHA256 写入附加列。运行目录迁移检查已有快照指纹，
将 journal 中的新记录合入 checkpoint；损坏的 journal 必须先恢复，不能静默丢弃。
旧配置路径失效时可用 `run --profile ... --grid ...` 指定重测配置；
脱敏或缺失的启动参数不会被猜测补齐。这里只迁移配置与测量结果，trace 和旧报告保留在原目录。
迁移产物标记为 `resume_compatible=false`，用于分析和独立 retest/profile；
不把补写的版本号或配置指纹当作历史续测安全的证明。新测量仍可按正常流程续测。

当前数据契约：

- profile 仍为 schema v1；单请求和 grouped batch 都是当前合法用例。
- 原始结果必须为 `schema_version=2`、`mode=prefix_cache_grid`、`metrics` 数组。
  metric 明确记录 `batch_size/input_len/status/measure_runs/success_runs/runs`；
  scalar metric 使用 `cache_len_requested`，grouped metric 保留 `request_groups` 和逐轮 `requests`。
  执行异常的 `status=error` 记录可以没有测量轮次。
- 公式拟合、静态 SVG、交互图和 batch 图直接读取 `cache_grid_results.json`。
  唯一分析时间是 `runs[].prefill_time_ms`，grouped batch 为 `runs[].requests[].prefill_time_ms`，
  两者均为服务端 `aux_info.first_token_cost_time`，单位 ms，包含引擎内等待、不扣减 `wait_time`，
  不包含输入分词及客户端通信。HTTP 与 input_ids/gRPC 使用相同的服务端口径。
- 单请求默认对有效轮次取中位数；`--all-runs` 展示逐轮值。TPM 也以服务端时间为分母。
  Batch 图每轮取批内请求服务端耗时的最大值，再对各轮取中位数；不使用客户端 batch wall time。
  生成结果记录 `measurement_contract=server_first_token_cost_time_ms`。
- `ttft_ms/client_wall_time_ms/batch_wall_time_ms` 仅保留为客户端诊断，不参与公式和图表。
  缺少服务端耗时则报错，非正或非有限耗时按无效测量处理，不回退到客户端时间。
  拟合和基准图不再接受 CSV 或旧分析中间文件；`predictions.csv` 仍可作为拟合输出。
- 迁移工具仅转换已有的服务端数据；无法从客户端耗时还原服务端首 token 延迟。
- 续测要求 seed mode、transport、grid/profile/run_config 指纹和测量参数明确存在并匹配；
  `allow_resume_mismatch` 不能绕过旧格式拒绝或迁移产物的续测限制。
- 物化用例要求 store_info 和 record 均为 v2，seed 明确写成文本或 marker/filler；
  正常读取不再访问 `prefixes/base.txt`。
- 生成 grid 必须通过 `--cache-alignment` 或 profile 明确给出正的缓存对齐值，
  不再将 0 或缺省值解释为 input alignment。

## 5. 分类工具入口

- [runner/generate_cache_grid.py](runner/generate_cache_grid.py)：普通 cache grid。
- [runner/generate_random_batch_grid.py](runner/generate_random_batch_grid.py)：随机 batch 计划；[说明](docs/generate_random_batch_grid.md)。
- [runner/run_random_batch_grids.py](runner/run_random_batch_grids.py)：随机 batch 多计划执行。
  新生成的随机 grid 使用固定 CP8 容量策略：`max_seq_len=1048576`、`max_context_batch_size=1`，
  生成及启动前同时检查总 input 和 CP 对齐后的 batch 矩形；简化入口自动识别策略标记。
  这不代表 batch 被限制为 1，也不保证模型/KV 等总显存不会 OOM。详见[容量约束及命令](docs/generate_random_batch_grid.md)。
- `tools/cache_perf pipeline`：通过统一测试入口运行，完整结束后拟合和绘图。
- [plot/generate_batch_interactive_chart.py](plot/generate_batch_interactive_chart.py)：batch/cache/compute/服务端首 token 延迟交互图，直接传入 `--input cache_grid_results.json --output chart.html`。
- [plot/generate_prefill_interactive_chart.py](plot/generate_prefill_interactive_chart.py)：单 batch prefill 交互图。
- [plot/generate_prefill_3d_chart.py](plot/generate_prefill_3d_chart.py)：静态 SVG。
- [plot/generate_prefill_traffic_report.py](plot/generate_prefill_traffic_report.py)：生产流量统计报告。
- [plot/generate_production_ttft_3d.py](plot/generate_production_ttft_3d.py)、[TPM 图](plot/generate_production_tpm_3d.py)：生产指标图。
- [formula/prefill_formula_fit.py](formula/prefill_formula_fit.py)：validate-inputs / fit / analyze-anomalies。
- [详细示例](examples/README.md)、[batch 分组机制](docs/cache_grid_batches.md)。

推荐从仓库根目录用新模块路径运行，例如：

```bash
python3 -m rtp_llm.test.perf_test.cache_grid.plot.generate_prefill_interactive_chart --help
python3 -m rtp_llm.test.perf_test.cache_grid.formula.prefill_formula_fit --help
python3 -m unittest discover -s rtp_llm/test/perf_test/cache_grid/tests -p '*_test.py'
```

旧 `python3 -m rtp_llm.test.perf_test.generate_...` 及父目录脚本路径不再可用，请使用上面的分类模块路径。
已有 Bazel target 名称保持不变，直接执行本目录中的实现，不再经过旧脚本转发。
新增测试统一放 `tests/`；Bazel 定义仍集中在父目录 BUILD，避免引入破坏旧 target 的子包边界。

### 固定 cache、递增 compute 的 grid

`fixed-cache-sweep` 模式固定一个或多个 cache 长度，并按固定步长增加未命中的
compute 长度。每个 case 满足：

```text
input_len = cache_len + compute_len
compute_len = min_compute_len + n * one_compute_step
input_len <= max_input_len
```

`--compute-step` 接受从粗到细排列的逗号分隔步长。每一级只补充之前级别尚未
生成的点，因此同一个 `(cache_len, compute_len)` 不会重复执行。例如
`--compute-step 32768,8192,4096` 会先跑 32768 粒度，再补 8192 粒度，最后补
4096 粒度。

手动指定 cache 长度时，用逗号分隔传给一个 `--fixed-cache-len`：

```bash
python3 -m rtp_llm.test.perf_test.cache_grid.runner.generate_cache_grid \
  --output /path/fixed-cache-grid.json \
  --grid-mode fixed-cache-sweep \
  --fixed-cache-len 32768,65536 \
  --min-compute-len 4096 \
  --compute-step 32768,8192,4096 \
  --max-input-len 262144 \
  --cache-alignment 4096
```

也可以从对齐后的 cache 候选中无放回随机抽样。相同 `--seed`、范围和数量会生成
相同的 cache 列表：

```bash
python3 -m rtp_llm.test.perf_test.cache_grid.runner.generate_cache_grid \
  --output /path/random-fixed-cache-grid.json \
  --grid-mode fixed-cache-sweep \
  --random-cache-count 16 \
  --min-cache-len 4096 \
  --max-cache-len 131072 \
  --min-compute-len 4096 \
  --compute-step 32768,8192,4096 \
  --max-input-len 262144 \
  --cache-alignment 4096 \
  --seed 104729
```

`--fixed-cache-len` 与 `--random-cache-count` 二选一。手动值和随机候选都必须满足
`--cache-alignment`；重复的手动值会直接报错。未设置 `--min-compute-len` 时从最细的
`--compute-step` 开始。步长必须唯一并严格按从大到小排列。还要确保最小 compute 不小于运行配置中的
`cache_grid.commit_tail_tokens`，否则 runner 会拒绝 cache-hit case。

## 通用化迁移

公式入口统一为 `formula.prefill_formula_fit`，Bazel target 为 `prefill_formula_fit`，
默认输出 `<model_label>_prefill_formula.txt`；专用 profile 可以继续指定旧产物名。
新 profile 只在顶层设置一次 `model_label`，缺省时取 `engine.model_type`，再缺省为 `Model`。
图表标题自动生成，不需要 `chart` 配置；公式文件名由模型名称拼接为 `<model_label>_prefill_formula.txt`、
key `PREFILL_TIME_FORMULA`、token 单位 1024，cold 标注阈值默认 1048575。
当前 profile 的可选 `chart` 字段和工具 CLI 可覆盖图表默认值。
DSV4 配置保留在 `config/dsv4_*.json*`，作为显式选用的模型 preset。
随机 batch 启动器现在要求 `--profile`，固定 CP8 workspace 需显式选择，详见
[随机 batch 文档](docs/generate_random_batch_grid.md)。

## Pipeline：测试、拟合与绘图

```bash
# 首次测试后自动拟合、生成 SVG 和 HTML；可加 --dry-run 预览所有阶段。
./tools/cache_perf pipeline --profile /path/local.jsonc --grid /path/grid.json \
  --result-dir /path/new-results --runs 3

# 使用冻结的启动配置续测，完整结束后再拟合、绘图。
./tools/cache_perf pipeline --test-mode resume --result-dir /path/results

# 已有完整结果：只重新拟合、绘图，不启动 Bazel 或模型服务。
./tools/cache_perf pipeline --skip-test --result-dir /path/results
```

测试阶段与 `run/resume` 共用 Bazel 启动、环境和快照接口；后处理优先读取
`profile.snapshot.json`，旧结果无该文件时使用结果内嵌 profile 或工具默认值。
`--skip-test` 不接受 profile/grid/runs/env 等启动覆盖项。
引擎参数写入 profile，环境覆盖使用 `--env NAME=VALUE`，不再用末尾 `--` 转发 runner 参数。

保留 `--batch-size`（选择后处理 batch）、`--estimator median|min|trimmed`、
`--formula-output-dir`、`--svg-output`、`--cold-svg-output`、`--html-output`。
`pipeline_summary.json` 记录各阶段退出码和产物位置。
测试失败或结果不完整时不进行后处理；拟合质量门禁返回 3 时仍绘图，最终返回 3；
其他拟合或绘图错误立即停止并返回对应退出码。

例如顶层 `model_label: "DeepSeek-V4-Pro"` 输出 `DeepSeek-V4-Pro_prefill_formula.txt`。
文件名保留模型名称的大小写和连字符；空格、路径分隔符等转换为下划线。

Pipeline 的 `prefill_3d.interactive.html` 默认使用 `--all-runs`，展示每次成功测量的原始散点，
不再将重复测量折叠为中位数点；静态图和公式拟合的聚合方式不受影响。

Pipeline 还会生成 `prefill_tpm_per_card.interactive.html`，只生成总输入长度口径的 TPM 图，
不自动生成 computed-length TPM 图。每个成功测量单独计算：
`单卡 TPM = input_len × 60000 / RT(ms) / cards`，其中 `input_len` 包含缓存命中部分。
卡数优先读取冻结 profile 的 `engine.world_size`；未提供时取 `tp_size × dp_size × pp_size`
（缺省维度为 1），EP/CP 不再额外相乘。旧结果无 profile 拓扑时回退到 run_config，
仍无卡数信息时按 1 卡处理。独立绘图 CLI 可用 `--cards` 显式覆盖。
