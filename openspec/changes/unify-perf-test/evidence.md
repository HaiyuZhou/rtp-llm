# Integration Evidence

## 2026-10-29: Local GPU prefill smoke

- Model: `/mnt/hf3fs/3fs/models/Qwen2.5-0.5B-Instruct/`
- GPU: one NVIDIA L20D selected through `//tools:nvidia_wheel_env`
- Config: `/tmp/unify_perf_prefill_smoke_20261029.json`
- Result directory: `/tmp/unify_perf_prefill_smoke_results_20261029/prefill-20261029T064306408943Z`
- Profile session: `profiles/profile-20261029T065126199747Z`

Each GPU action was preceded by:

```bash
../internal_source/.cursor/skills/test-execution/pre_build_check.sh local \
  /data0/zhouhaiyu.zhy/project/RTP-LLM-worktrees/cache_grid_main \
  --user=zhouhaiyu.zhy \
  --output-user-root=/home/zhouhaiyu.zhy/.cache/bazel_x86_cache_grid_main_cache
```

The common Bazel invocation was:

```bash
RTP_GPU_FREE_MEMORY_MIB=-1 bazelisk \
  --output_user_root=/home/zhouhaiyu.zhy/.cache/bazel_x86_cache_grid_main_cache \
  run --config=cuda13 --jobs=200 \
  --disk_cache=/home/zhouhaiyu.zhy/.cache/disk_cache \
  --config=daily_aone_bazel_cache \
  --remote_header=x-aone-bazel-api-key=ai-infra-cicd \
  --run_under=//tools:nvidia_wheel_env \
  //rtp_llm/test/perf_test:perf_test -- <action arguments>
```

Action arguments and outcomes:

| Action | Arguments | Outcome |
| --- | --- | --- |
| run | `--mode=prefill --action=run --config=/tmp/unify_perf_prefill_smoke_20261029.json` | Exit 0; common envelope completed with 4 planned, 4 successful, 0 failed cases. |
| resume | `--mode=prefill --action=resume --run-dir=<result directory>` | Exit 0; all completed cases were preserved. |
| profile | `--mode=prefill --action=profile --run-dir=<result directory> --cases prefill-6a6b5e35036dd4c526df6eb9` | Exit 0; isolated profile completed under the profile session directory. |
| report | `--mode=prefill --action=report --run-dir=<result directory>` | Exit 0 without the GPU wrapper or model startup; generated the formal JSON and HTML reports. |

The planned grid covered `(input, cache, compute)` values `(128, 0, 128)`,
`(256, 0, 256)`, `(384, 256, 128)`, and `(384, 0, 384)`. Every case
passed shape, timing, and exact reuse validation. The cache case observed reuse
length 256 for expected reuse length 256 in both the formal run and isolated
profile.

The following formal artifact SHA-256 values were identical before and after
resume, profile, and report:

```text
9e1f1b60ca2de220b78ce7cf047466e94361541cd8a7415cb50c65488d316c39  results/result.json
d1131208746f6e523e30b8fee40b6801ec6ede1d72b9c3c2d687d76388b92a82  results/summary.json
0af7d6391731eba52c8453a1c1bf694e368485ca33fc5e135dab3ccadc2ff69b  checkpoints/prefill_progress.json
```

The final report declared `report_kind: formal`, 4 planned observations, 4
valid observations, and 0 excluded observations. Its artifacts were:

```text
d5b1c1a50a7b7e279d97bc992f10fadc1a2de8f848238dd92a03f21a1c41e701  reports/prefill_report.json
46421baed0c12dd5f31d89cd906f1c438f2a7d3e6083a9020546dc7968808f08  reports/prefill_report.html
```

The first real-model attempt exposed two runtime defects: the launch adapter
passed the unsupported `--pp_size=1` argument to `start_server.py`, and an
engine failure was subsequently masked by empty-case finalization. The launch
adapter now omits the unsupported argument, and the default executor preserves
typed failed/interrupted outcomes without finalization. Focused direct tests and
the Bazel `runtime_adapter_test` and `default_executor_test` targets passed
before the successful smoke run.

## 2026-10-29: Local GPU decode smoke

All successful runs used the same Qwen2.5-0.5B model and L20D GPU setup as the
prefill smoke, with the required pre-build check and GPU lock. The Bazel command
was the common local command above with `WORLD_SIZE=1` and these action
arguments:

| Workload | Config | Result directory | Outcome |
| --- | --- | --- | --- |
| fixed grid | `/tmp/unify_perf_decode_fixed_smoke_20261029.json` | `/tmp/unify_perf_decode_fixed_smoke_results_20261029/decode-20261029T070036070965Z` | Exit 0; 2/2 cases succeeded for scalar output length 8 and global batches 1 and 3. |
| distribution | `/tmp/unify_perf_decode_distribution_smoke_20261029.json` | `/tmp/unify_perf_decode_distribution_smoke_results_20261029/decode-20261029T070226575066Z` | Exit 0; 2/2 cases succeeded and retained actual sequence lengths `[32]` and `[32, 48, 64]`. |
| target TPOT | `/tmp/unify_perf_decode_tpot_smoke_20261029.json` | `/tmp/unify_perf_decode_tpot_smoke_results_20261029/decode-20261029T071003771024Z` | Exit 0; bounded search measured batches 1 and 4, selected batch 4, and retained the complete trajectory. |

The fixed and distribution results retained scalar output length, global batch,
worker allocation, successful/failed request counts, wait/prefill/decode/TPOT/
total latency, throughput, topology, and the common result envelope. The TPOT
target was 10 ms; measured observations were 5.4533 ms at batch 1 and 5.49 ms
at batch 4, both with success rate 1.0. The final search result reported
`outcome: found`, `best_batch_size: 4`, and throughput 267.3888 output tokens/s.

The real smoke exposed that `target_tpot_ms` was planned but not connected to
the production fixed or distribution runners. A shared runner integration now
executes the bounded candidates, uses deterministic DP scheduling, preserves
every observation, and makes a no-satisfying-candidate result non-successful.
Nineteen focused direct tests passed. The Bazel targets `decode_fixed_test`,
`decode_distribution_test`, `decode_scheduler_test`, `decode_tpot_test`, and
`decode_metrics_test` also passed (invocation
`dc4ee26d-cdbc-4981-b846-25abfda49b0a`).

A DP=2 smoke was attempted first with the same dense Qwen model and correctly
returned engine exit code 4. The server rejects pure dense `ep_size=1` startup
with `dp_size=2`; a real DP engine smoke therefore requires an EP/DP-capable MoE
model. Global DP accounting remains covered by `decode_scheduler_test`, including
uneven DP=2 allocation and idle-worker DP=4 allocation, while the real fixed
run verifies that the persisted global batch is not replaced by a worker-local
field.
