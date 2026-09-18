# 随机 batch 生成与分批测试

在仓库根目录运行。生成 JSON 和预览命令无需加载模型或 GPU。

## 生成多个固定 batch 的文件

```bash
python3 -m rtp_llm.test.perf_test.cache_grid.runner.generate_random_batch_grid \
  --output-dir /home/admin/tmp/dsv4_batch_grids \
  --num-cases 100 \
  --batch-sizes 2 4 8 16 32 \
  --max-batch-tokens 1048576 \
  --max-input-tokens 262144 \
  --seed 20260917
```

每个 batch size 生成一个 JSON，每个文件包含 100 个随机 case。
新生成的 JSON 带有 `generator.workspace_policy="fixed_cp8_1m_v1"`。
该策略适用于 DSV4 TP8/CP8、逻辑 cache block4096、decode=1：
固定 `max_seq_len=1048576`、`max_context_batch_size=1`，不再随 batch 提升后者。
`max_batch_tokens_size` 使用 `--max-batch-tokens`，默认 1048576，不能超过该值。

生成和启动前同时检查：

- 所有请求的完整 input 长度之和（包括 cache）不超过 token 预算。
- `batch_size × align_up(最长 input + 1, 16) <= 1048576`。
  这里预留 1 个输出 token，并包含 CP8 对齐；仅限制 input 总和不够。
- seed batch 和缓存探针也必须满足容量限制；独立前缀逐请求 seed，组内共享前缀逐组 seed。

目录模式将上述单请求容量上限与 `max_input_tokens`、
`max_batch_tokens // batch_size` 取最小值，再按 input alignment 向下取整。
默认 input alignment=256 时，batch=2/4/8/16/32 的上限分别为
262144/261888/130816/65280/32512 tokens；batch=31 为 33792。
默认 min input 为 8192；无法满足最低长度时直接报错。
文件名包含 batch 和取整前的 input 上限，实际长度约束以 JSON 内容为准。

可单独指定某个 batch 的上限（仍不能超过 256K 和全局 max input）：

```bash
python3 -m rtp_llm.test.perf_test.cache_grid.runner.generate_random_batch_grid \
  --output-dir /home/admin/tmp/dsv4_batch_grids_custom \
  --batch-sizes 8 16 31 --num-cases 100 \
  --max-batch-tokens 1048576 \
  --batch-input-limit 16:32768 --batch-input-limit 31:32768
```

显式上限仍会被固定 workspace 容量上限截断，不能绕过对齐矩形约束。
它可以大于较小的自定义 token 预算除以 batch，但总 input 仍受该预算约束。

每条请求使用 count=1 的 request_group，前缀独立。input 默认按 256 对齐，
cache 默认按 4096 对齐；命中请求至少保留 4096 新 token。
`--cold-probability` 默认 0.2；长度和预算约束可能使实际 cold 比例更高。
采用剩余预算约束下的顺序采样，最后打乱请求槽位，并非所有合法 batch 的均匀采样。
相同参数和 seed 生成相同计划，脚本拒绝覆盖已有文件。

可用 `--kv-budget-tokens` 限制估算 KV 峰值，单位为逻辑 token，预算应提前扣除余量。
估算为每条请求的 block 取整输入量，加命中请求的 seed commit tail，
只覆盖一轮活跃 batch 和 seed 尾部，不包含旧轮次缓存、DSV4 state pool 或 runtime 显存。
因此不能替代服务端实际容量与缓存命中校验。

## 顺序启动多个测试

先保留原启动脚本中正确的 PATH、LD_LIBRARY_PATH 等环境配置，再运行：

```bash
python3 -m rtp_llm.test.perf_test.cache_grid.runner.run_random_batch_grids \
  --grid-dir /home/admin/tmp/dsv4_batch_grids \
  --model-dir /mnt/fuse/smoke-test/deepseek-ai/DeepSeek-V4-Flash/ \
  --mega-moe-se 0 \
  --result-root /home/admin/tmp/dsv4_batch_results_$(date +%Y%m%d_%H%M%S) \
  --output-base /home/admin/tmp/cache_grid_bazel_output \
  --jit-cache-dir /home/admin/perf_test \
  --dry-run
```

`--dry-run` 只打印参数和完整 Bazel 命令。删除它即可实际测试。
Pro 模型改用对应 model-dir 和 `--mega-moe-se 1`（默认值）。
`--reserve-runtime-mem-mb` 默认 81920，是每张 GPU 的 runtime 预留预算。

脚本读取目录下所有 `*.json`；也可用
`--grid-json /path/batch_008_input_131072.json /path/batch_016_input_65536.json`
指定文件。每个文件必须只有一个 batch size。

每个文件单独执行一次 Bazel test，重新初始化服务：

- `concurrency_limit` 设置为该文件的 batch size；测试入口保证 `max_generate_batch_size` 至少为 batch size。
- 固定 `max_context_batch_size=1`、`max_seq_len=1048576`。
- `max_batch_tokens_size` 使用生成时的 token 预算，缺省为 1048576。
- 显式启用 `--cache_fixed_workspace`，超限直接拒绝，不自动拆 batch，也不偷偷提升容量。
- 使用 `dashsc_input_ids`，逐请求验证实际 cache reuse。
- 使用 TP8/EP8/CP8、physical block512、logical cache block4096 的原配置。
- 每个 JSON 使用独立的结果子目录；根目录 `batch_runs.json` 记录实际参数、
  命令、运行状态与退出码。已有结果不会被覆盖。

默认遇到失败停止，后续文件保持 pending；`--continue-on-error` 可继续其余文件。
若 batch 场景允许实测 cache reuse 与 JSON 期望不同，可加
`--skip-reuse-validation`。启动器会传递底层 `--cache_skip_reuse_validation`，保留每个
请求的实际 `input_len`、`reuse_len` 及 `reuse_exact`，并继续运行后续 case。
复测选定文件时使用 `--grid-json` 和新的 result-root。
若需额外 Bazel 环境选项，可放在末尾 `--` 后，例如：
`-- --test_env=FLASHINFER_DISABLE_VERSION_CHECK=1`。
不接受额外 `--test_arg`，以免覆盖每个文件的 batch/长度配置。
脚本通过当前 PATH 找到 gcc/g++，不会自动修复编译工具或依赖版本不匹配。

Pro 的 CP gather workspace 在该配置下约为每卡 20 GiB，而不是乘以 batch 的约 620 GiB。
这只约束该 workspace，不保证总显存足够：KV/state pool、模型权重和其他临时缓冲仍可能 OOM。
这里的 batch 由测试专用调度器按 case 请求数凑齐，不能将其当作普通线上服务的调度保证。

## 使用简化入口

新生成的文件也可直接交给 `tools/cache_perf`：

```bash
./tools/cache_perf run --profile /path/local.jsonc \
  --grid /path/generated_batch_grid.json --result-dir /path/new_result
./tools/cache_perf resume --result-dir /path/new_result
```

允许实测 reuse 与 grid 期望不一致时，在首次运行增加
`--skip-reuse-validation`：

```bash
./tools/cache_perf run --profile /path/local.jsonc \
  --grid /path/generated_batch_grid.json --result-dir /path/new_result \
  --skip-reuse-validation
```

该选项会写入启动快照，后续 `resume` 自动沿用，无需再次传入。

策略标记会覆盖 profile 中的上述容量参数并冻结有效启动配置；模型路径、编译器和环境仍来自 profile。
`resume` 使用冻结配置，`retest` 继承固定容量策略；`profile` 仍只支持未分组 batch=1。
未带标记的旧 grid 在简化入口和底层测试入口保留原来的自动扩容行为，旧 checkpoint 不变。
多文件启动器 `run_random_batch_grids` 则一律执行固定容量校验，包括旧 JSON；超限文件需重新生成或缩短请求。

## 原单文件模式

原来的 `--output /path/grid.json --batch-sizes 2 4 8` 用法继续支持；
它在同一个文件里随机选择 batch size，可使用 `tools/cache_perf`，不适合只接受固定 batch 文件的多文件启动器。

## CPU 测试

```bash
python3 -m unittest discover -s rtp_llm/test/perf_test/cache_grid/tests \
  -p generate_random_batch_grid_test.py -v
```
