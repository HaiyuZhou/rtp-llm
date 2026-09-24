# 随机 batch 生成与分批测试

在仓库根目录运行。生成 JSON 和预览命令不加载模型或 GPU。

## 生成 grid

长度、cache block 和预算按实际模型配置；下面对应单卡 block64、32K 上下文的示例：

```bash
python3 -m rtp_llm.test.perf_test.cache_grid.runner.generate_random_batch_grid \
  --output-dir /tmp/cache_batch_grids --num-cases 100 --batch-sizes 2 4 8 \
  --min-input-tokens 1024 --max-input-tokens 30000 \
  --max-batch-tokens 65536 --input-alignment 64 \
  --cache-alignment 64 --commit-tail-tokens 64 --seed 20260917
```

每个 batch size 生成一个文件。默认 `workspace_policy="none"`，不强制 CP8、1M workspace、
256K 单请求上限或 block4096。默认数值只是可覆盖的采样参数，不代表模型容量。
单请求采样上限取 `max_input_tokens` 与 `max_batch_tokens // batch_size` 的较小值；
`--batch-input-limit B:TOKENS` 可对目录模式中的指定 batch 进一步设限。
`--output /path/grid.json --batch-sizes 2 4 8` 可生成混合 batch 文件，交给 `tools/cache_perf` 运行。

请求前缀相互独立，命中请求至少保留 `commit_tail_tokens` 新 token。
`--cold-probability` 默认 0.2；实际比例受长度和预算约束影响。
相同参数和 seed 生成相同计划，拒绝覆盖已有文件。
`--kv-budget-tokens` 可约束一轮活跃 batch 加 seed 尾部的 KV token 估算，
不包含旧轮次缓存、模型专用 state pool 或 runtime 显存，也不替代实际容量校验。

## 使用 profile 顺序启动

复制 `config/local.example.jsonc` 并设置模型类型、路径、拓扑、cache geometry、编译配置和环境。
模型上下文应容纳 input、输出 token、seed 和探针。固定 batch 当前要求 DP=1。

```bash
python3 -m rtp_llm.test.perf_test.cache_grid.runner.run_random_batch_grids \
  --grid-dir /tmp/cache_batch_grids --profile /path/local.jsonc \
  --result-root /tmp/cache_batch_results --dry-run
```

去掉 `--dry-run` 执行。每个文件只有一个 batch size，单独初始化服务；也可用
`--grid-json /path/one.json /path/two.json` 指定文件。
启动器复用 `cache_perf` 的参数、环境和快照机制，每个结果目录保存 profile/grid 快照及
`cache_perf_launch.json`，可直接用 `tools/cache_perf resume --result-dir ...` 续测。
普通 grid 的 batch 容量由测试入口按 case 需求配置；固定 workspace 策略见下节。
根目录 `batch_runs.json` 记录命令、状态和退出码。已有结果不覆盖，默认失败即停止，
`--continue-on-error` 可继续剩余文件。

使用 `--config`、`--output-base`、`--bazel` 覆盖构建参数，`--env NAME=VALUE` 覆盖环境。
允许实测 reuse 偏离期望时显式传入 `--skip-reuse-validation`，实际 reuse 仍被记录。
旧启动参数 `--model-dir`、`--mega-moe-se`、`--reserve-runtime-mem-mb` 和 `--jit-cache-dir`
迁移到 profile 的 `engine`、`engine_args`、`engine_env`、`runtime_env`；不再隐式选择 DSV4。

## DSV4 CP8/1M 专用 preset

原固定 workspace 实验可继续使用 `config/dsv4_local.example.jsonc`，生成时显式增加：

```text
--workspace-policy fixed_cp8_1m_v1 --cache-alignment 4096 --commit-tail-tokens 4096
```

该策略保留 TP8、DP1、CP ALL_GATHER、sharded KV、decode=1 的要求，
固定 `max_seq_len=1048576` 和 `max_context_batch_size=1`，要求
`batch_size × align_up(最长 input + 1, 16) <= 1048576`，并校验 seed 和探针容量。
模型专用开关和硬件配置仍由 profile 显式提供。
旧 grid 的策略标记继续生效；未带标记的 grid 使用通用路径。
已有实验续测必须复用原 grid 和冻结配置，不要用新默认值重新生成后接续旧结果。
