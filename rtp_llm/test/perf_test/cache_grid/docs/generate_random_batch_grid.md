# 随机 batch 生成与分批测试

在仓库根目录运行。生成 JSON 和预览命令无需加载模型或 GPU。

## 生成多个固定 batch 的文件

```bash
python3 rtp_llm/test/perf_test/cache_grid/runner/generate_random_batch_grid.py \
  --output-dir /home/admin/tmp/dsv4_batch_grids \
  --num-cases 100 \
  --batch-sizes 1 2 4 8 16 31 \
  --max-batch-tokens 1048576 \
  --max-input-tokens 262144 \
  --seed 20260917
```

每个 batch size 生成一个 JSON，每个文件包含 100 个随机 case。
目录模式下，单请求 input 上限取
`min(max_input_tokens, max_batch_tokens // batch_size)`，再按 input alignment 向下取整。
因此 batch=1/2/4 的上限是 256K，batch=8 是 128K，batch=16 是 64K，
batch=31 是 33792 tokens。默认 min input 为 8192；过大的 batch 可能无法满足最低长度。
文件名包含 batch 和取整前的 input 上限，实际长度约束以 JSON 内容为准。

可单独指定某个 batch 的上限（仍不能超过 256K 和全局 max input）：

```bash
python3 rtp_llm/test/perf_test/cache_grid/runner/generate_random_batch_grid.py \
  --output-dir /home/admin/tmp/dsv4_batch_grids_custom \
  --batch-sizes 8 16 31 --num-cases 100 \
  --max-batch-tokens 1048576 \
  --batch-input-limit 16:32768 --batch-input-limit 31:32768
```

显式上限允许大于 `max_batch_tokens // batch_size`；总 batch input 仍受总预算约束，
但启动时 `batch_size * max_seq_len` 可能变大，需要考虑 workspace 显存。

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
python3 rtp_llm/test/perf_test/cache_grid/runner/run_random_batch_grids.py \
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

- `concurrency_limit` 和 `max_context_batch_size` 设置为该文件的 batch size。
- `max_seq_len` 设置为文件中最大请求 input 长度加 1（decode=1），
  并确保覆盖缓存粒度探针需要的 8192 tokens 和 seed commit tail。
- `max_batch_tokens_size` 设置为文件中最大的总 input，且覆盖探针长度。
- 使用 TP8/EP8/CP8、physical block512、logical cache block4096 的原配置。
- 每个 JSON 使用独立的结果子目录；根目录 `batch_runs.json` 记录实际参数、
  命令、运行状态与退出码。已有结果不会被覆盖。

默认遇到失败停止，后续文件保持 pending；`--continue-on-error` 可继续其余文件。
复测选定文件时使用 `--grid-json` 和新的 result-root。
若需额外 Bazel 环境选项，可放在末尾 `--` 后，例如：
`-- --test_env=FLASHINFER_DISABLE_VERSION_CHECK=1`。
不接受额外 `--test_arg`，以免覆盖每个文件的 batch/长度配置。
脚本通过当前 PATH 找到 gcc/g++，不会自动修复编译工具或依赖版本不匹配。

降低 max_seq_len 可降低相关 workspace 的分配量，但不保证所有配置都能避免 OOM；
KV 容量、模型权重与其他 workspace 仍需要实际测量。

## 原单文件模式

原来的 `--output /path/grid.json --batch-sizes 2 4 8` 用法继续支持；
它在同一个文件里随机选择 batch size，适合原启动脚本，不适合这个固定 batch 的多文件启动器。

## CPU 测试

```bash
python3 -m unittest discover -s rtp_llm/test/perf_test \
  -p generate_random_batch_grid_test.py -v
```
