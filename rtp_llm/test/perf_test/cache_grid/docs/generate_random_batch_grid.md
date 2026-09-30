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

每个 batch size 生成一个文件。默认不启用固定 workspace；默认数值只是可覆盖的采样参数，不代表模型容量。
不再把总预算按 batch 平均分配。设对齐后的最小长度为 m、总输入预算为 T，
第 i 个请求的采样上限为 `min(max_input_tokens, T - 已采样长度之和 - 剩余请求数 × m)`。
这样允许一个长请求搭配多个短请求，同时保证后续请求至少能取最小长度。
最后打乱请求槽位，因此 JSON 中的第一个请求不一定是最先采样的长请求。
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

## 固定 token workspace

用 `--workspace-tokens` 显式指定整批固定容量，不绑定某个 TP/CP，也没有固定 1M 上限：

```bash
python3 -m rtp_llm.test.perf_test.cache_grid.runner.generate_random_batch_grid \
  --output-dir /tmp/flash_fixed_tokens_grids --num-cases 100 --batch-sizes 1 2 4 \
  --workspace-tokens 65536 --max-batch-tokens 65536 \
  --min-input-tokens 2048 --max-input-tokens 49152 \
  --input-alignment 256 --cache-alignment 1024 --commit-tail-tokens 1024
```

上述 cache 参数适用于已确认可见复用粒度为 1024 的配置，并非任意模型的默认推荐值。
`max-batch-tokens` 约束原始输入总量，`workspace-tokens` 约束逐请求补齐后总量：

```text
sum(input_len) <= max_batch_tokens
sum(align_up(input_len + 1, cache_alignment)) <= workspace_tokens
```

其中 `+1` 为当前 prefill-only 测试的输出预留。cache block 对齐是保守上界，
启动时还会检查它是当前 CP 执行对齐 (`2 × TP`，CP 禁用时为 1) 的倍数。
采样会为后续请求预留最小 padding 容量；KV budget 若启用，也独立计入剩余预算。
不再使用 `batch × 最长请求` 的矩形限制。

grid 在 `generator.workspace_tokens` 保存预算。启动器及直接调用 runner 都会检查正式测量、
并发 seed 和缓存探针，固定 `max_context_batch_size=1`，
`max_seq_len=floor(workspace_tokens / cache_alignment) × cache_alignment`，
`max_batch_tokens_size=min(workspace_tokens, max_batch_tokens)`；实际调度 batch 仍由 case 决定。
超限拒绝，不自动扩容、不拆 batch。当前固定 batch 测试仍要求 DP=1、decode=1，
不支持全局 `cache_shared_seed` 模式（case 内 `shared_by_group` 支持）。
这是 packed prefill token 容量限制，不是 workspace 字节或总显存保证；权重、KV pool、
通信、JIT 等仍需单独评估。模型必须支持此 packed batch 执行方式。

旧的 `--workspace-policy` preset 及布尔固定 workspace 参数已移除。
带旧策略标记的 grid 会明确报错，需要重新生成并使用新结果目录，不得接续旧 checkpoint。
新实验恢复时必须复用冻结 grid/profile，不要修改预算后接续原结果。
