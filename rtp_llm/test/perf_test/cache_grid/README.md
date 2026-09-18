# Cache Grid 性能工具

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

复制 [config/dsv4_local.example.jsonc](config/dsv4_local.example.jsonc) 为本机配置。
模板已逐项注释，重点检查：

1. `engine.checkpoint_path` 和 `engine.tokenizer_path`：模型在测试容器内的实际路径。
2. `runtime_env`：CC/CXX、CUDA、PATH/LD_LIBRARY_PATH、三个可写 JIT 缓存目录。
3. `bazel.output_base`：本机 Bazel 输出目录，可删除该项使用默认值。
4. GPU 拓扑变动时检查 TP/EP/DP/world 和 WORLD_SIZE；CP8、物理 block512 对应可见 cache block4096。
5. 测试规模由独立 grid 文件控制，正式轮数用 `cache_grid.measure_runs` 或 `--runs`。

标准 `.json` 仍严格遵守 JSON；`.jsonc` 支持 `//` 和 `/* … */` 注释。
不支持尾逗号，也不展开 `$VAR`、`${VAR}`、`~` 或执行命令；路径请填写明确值。
字符串中的 URL、`//`、转义引号不会当作注释。注释不参与 profile 指纹。
不要在配置中写密码、API key 或 token；敏感内容必须通过组织认可的外部方式注入。

## 2. 四种操作，不再组合多个布尔开关猜执行模式

以下从仓库根目录运行；入口仍然执行 `bazelisk test`，不会绕过 Bazel 构建。
可用 `python3 tools/cache_perf` 替代直接执行。

```bash
# 首次运行：必须是空/新结果目录。先加 --dry-run 检查最终命令。
./tools/cache_perf run \
  --profile /absolute/path/dsv4_local.jsonc \
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
| profile | 跳过 | 只选定 case，默认一次 | `cache_perf_replays/profile_<id>/cache_profile_replays/<id>/` |

retest/profile 会保留原 case_id 和请求分布，但使用新结果目录；不会改原 checkpoint 或报告。
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

直接使用旧 `bazelisk test --test_arg=--profile=...` 也支持 JSONC 和环境默认值，
但只能影响测试进程；不能反向改变已经启动的 Bazel 的编译环境。
旧入口保持“已有进程环境/--test_env 优先于 --engine_env 默认值，后者优先于 profile 默认值”的兼容行为。
要统一控制构建和测试环境，应使用新入口。

## 4. 快照、旧结果和安全边界

run 保存 `profile.snapshot.json`、`grid.snapshot.json`、`cache_perf_launch.json`。
resume 校验快照 SHA256，再交由原 runner 的 grid/profile/run_config 守卫验证；不会关闭不匹配检查。
运行失败但尚未产生 checkpoint 时，不要用 resume 猜测恢复；检查错误后使用新结果目录重试。

旧结果目录没有 launch manifest 时，可从 `test_info.json` 读取原 argv、环境与 grid 路径；
缺少 metadata、包含脱敏占位符、grid 路径不可用或内容变化会拒绝自动恢复。
旧运行仍须保留它引用的配置文件，不能声称已经获得历史完整环境。
若只需独立 retest/profile，可显式传入匹配原运行的 `--profile` 和 `--grid`；这是新隔离测试，不会尝试绕过旧 checkpoint 守卫。

```bash
./tools/cache_perf profile --result-dir /path/legacy-results \
  --profile /path/matching-local.jsonc --grid /path/original-grid.json \
  --cases 2077 --runs 1 --dry-run
```

## 5. 分类工具入口

- [runner/generate_cache_grid.py](runner/generate_cache_grid.py)：普通 cache grid。
- [runner/generate_random_batch_grid.py](runner/generate_random_batch_grid.py)：随机 batch 计划；[说明](docs/generate_random_batch_grid.md)。
- [runner/run_random_batch_grids.py](runner/run_random_batch_grids.py)：随机 batch 多计划执行。
- [runner/run_cache_grid_pipeline.py](runner/run_cache_grid_pipeline.py)：完整测试后拟合和绘图。
- [plot/generate_batch_interactive_chart.py](plot/generate_batch_interactive_chart.py)：batch/cache/compute/TTFT 交互图。
- [plot/generate_prefill_interactive_chart.py](plot/generate_prefill_interactive_chart.py)：单 batch prefill 交互图。
- [plot/generate_prefill_3d_chart.py](plot/generate_prefill_3d_chart.py)：静态 SVG。
- [plot/generate_prefill_traffic_report.py](plot/generate_prefill_traffic_report.py)：生产流量统计报告。
- [plot/generate_production_ttft_3d.py](plot/generate_production_ttft_3d.py)、[TPM 图](plot/generate_production_tpm_3d.py)：生产指标图。
- [formula/deepseek_v4_prefill_formula_fit.py](formula/deepseek_v4_prefill_formula_fit.py)：validate-inputs / fit / analyze-anomalies。
- [详细示例](examples/README.md)、[batch 分组机制](docs/cache_grid_batches.md)。

推荐从仓库根目录用新模块路径运行，例如：

```bash
python3 -m rtp_llm.test.perf_test.cache_grid.plot.generate_prefill_interactive_chart --help
python3 -m rtp_llm.test.perf_test.cache_grid.formula.deepseek_v4_prefill_formula_fit --help
python3 -m unittest discover -s rtp_llm/test/perf_test/cache_grid/tests -p '*_test.py'
```

旧 `python3 -m rtp_llm.test.perf_test.generate_...` 及父目录脚本路径不再可用，请使用上面的分类模块路径。
已有 Bazel target 名称保持不变，直接执行本目录中的实现，不再经过旧脚本转发。
新增测试统一放 `tests/`；Bazel 定义仍集中在父目录 BUILD，避免引入破坏旧 target 的子包边界。
