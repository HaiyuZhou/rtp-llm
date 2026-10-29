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

## 2026-10-29: Regression, golden, and publisher gates

The following Bazel targets passed together under invocation
`c0c2fc44-d7af-4b15-a9e4-dc6f511220fc`:

```text
//rtp_llm/test/perf_test:baseline_regression_test
//rtp_llm/test/perf_test:golden_test
//rtp_llm/test/perf_test:publisher_policy_test
//rtp_llm/test/perf_test:archive_publisher_test
//rtp_llm/test/perf_test:odps_publisher_test
```

These gates cover higher/lower-is-better thresholds, missing/ineligible
baseline inputs, deterministic candidate and golden output, required publisher
exit code 6, optional publisher failure preserving measurement success, and
OSS/ODPS credential redaction from results, manifests, exceptions, and logs.

The explicit Bazel golden binaries were also run against the successful prefill
GPU result:

```bash
bazelisk run //rtp_llm/test/perf_test:generate_perf_test_golden -- \
  --result=<prefill result>/results/result.json \
  --output=/tmp/unify_perf_golden_20261029.json
bazelisk run //rtp_llm/test/perf_test:validate_perf_test_golden -- \
  --result=<prefill result>/results/result.json \
  --golden=/tmp/unify_perf_golden_20261029.json
```

Generation and validation exited 0. The golden SHA-256 remained
`9e1f1b60ca2de220b78ce7cf047466e94361541cd8a7415cb50c65488d316c39`
before and after validation, proving the validation gate is read-only.

## 2026-10-29: AOne prefill smoke

The production unified entry built all generated shard targets in one Bazel
invocation, submitted them in one remote Bazel test invocation, collected the
undeclared-output ZIPs, finalized the common result envelope, and generated the
offline prefill report. Remote workers resolved the model repository
`Qwen/Qwen2.5-0.5B-Instruct` to a worker-local Hugging Face snapshot; generated
runfiles contained the normalized config, resolved workload, and shard plan but
did not contain or read the submit-host source-config path.

Single-shard evidence:

- Result directory:
  `perf_test_results_aone_single_hf_v4_20261029/prefill-20261029T080213050307Z`
- Build invocation: `709eb3bb-04a5-4a1f-8e2b-dcfe66ac8c91`
- Remote test invocation: `aabab163-12f2-4680-8b7d-c8216953920e`
- Outcome: the remote target passed in 77.9 seconds; build, submit, collect, and
  report stages completed; all 4 cases were valid; the cache case observed 256
  reused tokens for 256 expected tokens; the offline report was formal with 4
  planned, 4 valid, and 0 excluded cases.
- Config fingerprint:
  `708a3958c94add02ca10e1d4eed721cd029ac29e253ffe1613a4053a5dc087cf`
- Workload fingerprint:
  `15c0bfb6a2672b366ce77252af644a0fe0ded609bea7eacefe69e7a120ae5c67`

Multi-shard evidence:

- Result directory:
  `perf_test_results_aone_multi_hf_v3_20261029/prefill-20261029T082246425419Z`
- Build invocation: `913df686-1a43-4ebb-b46f-0463d96a0c45`
- Remote test invocation: `ff9a58c8-e654-4051-9778-b5b8571b632f`
- Outcome: both remote targets passed, in 42.3 and 41.4 seconds. The run used
  two shards and `max_parallel=1` for the final infrastructure-stable retry.
  `shard-0000` exclusively owned one case and `shard-0001` exclusively owned
  three; their union exactly matched the four planned and collected cases.
- Both raw `shard_result.json` files declared `completed`, returned only owned
  cases, and matched config fingerprint
  `f9c99f6f4cf4cd84b8eb2fc505cfed93682574b70a2a0562afff911e696b8792`
  and workload fingerprint
  `15c0bfb6a2672b366ce77252af644a0fe0ded609bea7eacefe69e7a120ae5c67`.
- The merged common result completed with 4/4 successful cases and exact reuse
  pairs `(0, 0)`, `(0, 0)`, `(0, 0)`, and `(256, 256)`. The offline report was
  formal with 4 planned, 4 valid, and 0 excluded cases.

The first concurrent two-shard attempt proved that `max_parallel=2` launched
both remote actions together, but one worker exposed an engine automatic-memory
boundary failure: 4,865 MiB available versus 4,867 MiB required after warmup.
The smoke config was made deterministic with an explicit 512 MiB KV-cache
budget, sufficient for this 385-token model workload. A second concurrent
attempt completed one shard while the other ended with Bazel `NO STATUS` after
the AOne gRPC connection was reset by the remote peer. The final sequential
retry changed only parallelism and completed both shards. The partial attempts
also verified that setup/engine failures produce structured shard diagnostics,
strict collection does not report completion, and the offline report is marked
diagnostic.

## 2026-10-29: Final static and affected-suite gates

- `openspec validate unify-perf-test --type change --strict --no-interactive`
  passed with zero issues.
- The full repository pre-commit hook set passed for every file changed by the
  runtime fixes and integration work.
- The production entry, CI, and current user-documentation scan found no
  references to deleted entry files, commands, flags, macros, or old import
  paths. Negative compatibility tests and historical design/OpenSpec documents
  were intentionally outside that scan.
- `bazelisk test //rtp_llm/test/perf_test:all` passed all 48 test targets under
  invocation `b4ffff8a-1a21-49a8-beb6-3a4a8e1dbb57`. This included config,
  planning, storage/state, engine lifecycle, prefill actions, AOne packaging and
  strict collection, decode fixed/distribution/TPOT, regression/golden, and
  publisher targets.

The production AOne adapter and real single/multi-shard smoke evidence were
added after this initial full-suite gate. The final focused verification ran 37
direct unit/integration tests successfully and passed all 10 affected Bazel
targets under invocation `4655266b-c517-44a8-b4a9-7b4c7050424b`:

```text
aone_bazel_integration_test  aone_worker_test       aone_collector_test
aone_orchestrator_test       aone_examples_test     runtime_adapter_test
dry_run_test                 plan_test              prefill_finalize_test
entry_test
```

The first focused Bazel command timed out before test execution because the
CUDA test wrapper waited for idle GPUs that were occupied by unrelated jobs.
The successful command propagated `RTP_GPU_FREE_MEMORY_MIB=-1` through
`--test_env`, retaining the CUDA wheel environment while bypassing the
irrelevant idle-memory gate for these CPU-only tests. Final pre-commit hooks,
`git diff --check`, the boundary-aware removed-entry scan, and strict OpenSpec
validation all passed after the evidence update.

## 2026-10-29: Final review fixes

The post-implementation review identified terminal-state and production-adapter
edge cases. AOne now writes the common result envelope only after report outcome
is known, including `partial`, `failed`, and `interrupted` runs; missing cases
receive explicit diagnostic case records. An empty collection is `failed`
rather than `partial`, report failure is reflected consistently in manifest,
result, summary, and exit code, and summaries retain `run_id`.

The Bazel adapter now preserves a nonzero batched test exit even when shard JSON
was downloaded, while retaining those shard cases for diagnostics. Adapter
environment parsing is performed inside the persisted build stage, generated
targets cannot escape `bazel-testlogs`, archive/direct JSON reads enforce member,
size, total-uncompressed-size, and compression-ratio bounds, and an interrupted
wait terminates the Bazel client before propagating the interrupt.

Verification after these fixes:

- 45 direct affected unit/integration tests passed.
- The complete pre-commit hook set passed for all changed files.
- Strict OpenSpec validation, `git diff --check`, and the removed-entry scan
  passed.
- All 10 affected Bazel targets passed under invocation
  `e97a1267-0970-4e2b-b8ee-f1d95c025560` (`Executed 9 out of 10 tests: 10 tests
  pass`; one target was a cache hit).
