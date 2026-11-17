# RTP-LLM Unified Performance Benchmark

Use `tools/perf_test` for local, SSH and AOne model-level prefill/decode tests.
Each run takes one strict JSON config. Declare the model, topology, engine,
workload and output in that file. Mode is inferred from `workload.kind`; action
is `run` for a config and `report` for an existing run directory. Explicit
`--mode` remains an assertion and must agree with the workload.

## Configure and preview

```bash
./tools/perf_test --config=rtp_llm/test/perf_test/examples/prefill_local.json --dry-run
./tools/perf_test --config=rtp_llm/test/perf_test/examples/decode_ssh_gang.json --dry-run
./tools/perf_test --config=rtp_llm/test/perf_test/examples/prefill_aone_ci.json --dry-run
```

Remove `--dry-run` to execute. Preview validates the complete plan without
creating a run directory, contacting hosts, building or starting a model.

| Backend | Configuration | Execution |
| --- | --- | --- |
| local | `execution.backend="local"`, `devices` | Current machine |
| ssh | `execution.backend="ssh"`, `hosts`, `gpus_per_machine`, `source` | Named remote machines |
| aone | `execution.backend="aone"`, resource and GPU type | AOne shards or gang |

`model.topology` is the only user declaration of TP/DP/PP/EP. Workload kind
`cache_grid` selects prefill; `grid` and `distribution` select decode. Decode
uses fixed global batches divisible by DP. Current model-level measurements
require PP=1. Profiling restrictions are checked during planning.

## Prefill batches

`examples/prefill_multi_grid_local.json` contains named cold, explicit mixed,
random and traffic grids in one engine session. Freeze its sampled batches
before sharing a config:

```bash
python -m rtp_llm.test.perf_test.prefill.export \
  --config rtp_llm/test/perf_test/examples/prefill_multi_grid_local.json \
  --output /tmp/prefill-batches-frozen.json
```

The exporter detects random and traffic sources, including named grids.
`--output-dir DIR` instead exports one file per batch size for a single
top-level random workload. Existing destinations are rejected. The corresponding
Bazel binary is `//rtp_llm/test/perf_test:export_prefill_batches`.

## SSH machines

Start with `examples/decode_ssh_gang.json` under `rtp_llm/test/perf_test/` for
SSH execution settings. For prefill, combine its `execution` object with the
model, engine and workload from `examples/prefill_local.json` in one JSON.
Set topology, `gpus_per_machine` and host count together; the local prefill
example needs one host. There is no launcher YAML or referenced benchmark JSON
to maintain.

Each host has a `ssh_target` for the submitting machine's SSH connection and
an `advertise_host` reachable by the other model workers. System SSH config,
including aliases and jump hosts, is respected. Passwordless access and a
prepared Python/GPU environment are required. Credentials do not belong in
benchmark configs.

World size is TP × DP × PP. The per-machine width is the smaller of world size
and `gpus_per_machine`; world size must divide evenly across hosts. Supply
exactly the required number of hosts, in rank order. A single remote host is
supported. Model paths must be available in the remote runtime.

`source.ref="auto"` freezes the submitting public worktree's current Git commit.
Commit tracked changes first. A branch/tag must already resolve locally; it
is frozen to a full SHA before submission. Only committed Git objects are sent
to executor-owned remote mirrors/checkouts. Untracked local files are not sent.

Build policies are `build_with_deps`, `build` and `reuse`. The first two require
an installed `source.build_driver` implementing the build request/receipt
protocol. Machine admission prepares that adapter and its execution policy;
the backend does not silently run bare Bazel when it is missing. Reuse requires
matching code, build settings, toolchain and verified native artifacts.

The worker owns only its run's process group and allocated devices. Cancellation
and cleanup use the recorded owner identity, rather than process-name matching.
A disconnected submitter can inspect or cancel its run later:

```bash
./tools/perf_test --run-dir=/path/to/ssh-run --maintenance=status
./tools/perf_test --run-dir=/path/to/ssh-run --maintenance=cancel
./tools/perf_test --run-dir=/path/to/ssh-run --maintenance=collect
./tools/perf_test --run-dir=/path/to/ssh-run --maintenance=clean
```

Collection retains failure evidence and does not promote an interrupted run to
success. Cleanup refuses active or mismatched owners and preserves source/build
caches. SSH supports prefill run and collected offline report, and decode run;
SSH checkpoint resume and standalone profile are not enabled.

## Batch experiments

`examples/decode_ssh_experiment.json` contains a complete inline `base_config`
and named benchmarks. `overrides` changes explicit fields, `zip_axes` combines
equal-length arrays by index, and `product_axes` expands a Cartesian product.
Overlapping paths and backend/source/output overrides are rejected.

```bash
./tools/perf_test --config=rtp_llm/test/perf_test/examples/decode_ssh_experiment.json --dry-run
```

Every expanded task is planned before the first remote side effect. Input files
are snapshotted once; each task's final workload and seed are materialized once.
Execution uses those frozen plans in order. Each task has its own run and result;
`experiment_plan.json` and `experiment_summary.json` index the experiment.

## Results and follow-up actions

Runs are written under `output.result_root/<run_id>/`. Read
`results/result.json` for the schema-v2 result envelope, `results/summary.json`
for aggregate metrics and `reports/` for reports. Prefill records cases in
`results/cases.jsonl`; decode uses `results/cases/`. Profiles and traces retain
member provenance. Execution plans use the current schema only; old plan
versions and removed launcher entries are rejected.

```bash
./tools/perf_test --run-dir=/path/to/local-prefill-run --action=resume
./tools/perf_test --run-dir=/path/to/local-prefill-run --action=profile --cases=CASE_ID
./tools/perf_test --run-dir=/path/to/prefill-run --action=report
```

Local prefill supports run/resume/profile/report. AOne and SSH prefill support
run and collected offline report. Decode supports run, including its automatic
report; AOne decode requires a multi-machine gang. Follow-up actions read the
frozen run and do not accept config overrides.

The manual `offline_bench_test` is a separate microbenchmark. Public runs do not
automatically publish results or replace regression baselines.

## Trace analysis

Prefill and decode store profile manifests in
`profiles/<session_id>/manifest.json` and captured files in
`traces/<session_id>/<case_id>/`. Schema 2 manifests list each case's capture
status and run-relative artifact path, format, rank and phase (unknown values
are null). Case dimensions and topology come from `execution_plan.json`.

Analyze a captured run, an experiment, or a directory containing runs:

```bash
python -m rtp_llm.test.perf_test.multi_node.analyzer.batch_analyze /path/to/run
```

The command writes `reports/trace/analysis.json` and `kernels.csv` below its
input directory; `--output-dir DIR` selects another destination. It does not
launch an engine or change measurements. Missing, incomplete, invalid and
unsupported artifacts are listed in `skipped`. An input with no analyzable
trace exits nonzero. Copied collected runs are supported without rewriting
the original plan.

Only listed Kineto/Chrome Trace JSON files are analyzed. `.nsys-rep` captures
remain available for Nsight Systems and are reported as unsupported by this
JSON analyzer. Each trace retains its session/case/rank context. Kernel time is
the sum of raw kernel durations; kernel span is the first-to-last kernel
interval. Neither is end-to-end request latency. No duration outliers are
silently removed. `--min-occurrences N` filters table rows only (default 1).

For a single trace, use
`python -m rtp_llm.test.perf_test.multi_node.analyzer.trace_analyze /path/to/trace.json`.
Optional module analysis uses the restored configs with the same run input:

```bash
python -m rtp_llm.test.perf_test.multi_node.analyzer.batch_analyze /path/to/run \
  --module-conf rtp_llm/test/perf_test/multi_node/analyzer/configs/qwen_480b_prefill_fp8.py
```

`module_conf.py` provides name matching only; the two `qwen_480b_*` presets also
provide model-specific FLOPs/IO estimates. Select a preset matching the model
and phase; the supplied formulas describe TP1. `modules.csv` and each trace's
`statistics.modules` contain matched kernel names, actual invocation counts,
summed durations and percentages of total kernel time. Rules match independently,
so overlapping patterns can contribute to more than one module. Kernel row
filtering does not remove events from module totals.

FLOPs/IO describe one module invocation. Rates are computed only when the spec
explicitly declares `formula_per_kernel=True`: estimated work per invocation
is multiplied by the actual matching kernel-event count and divided by summed
duration. QKV/O GEMM presets declare this; other presets leave rates null when
logical module invocations cannot be inferred from kernel launches. Context comes from the frozen case and
topology, with a per-DP batch and the case's input length. Heterogeneous or cached
prefill shapes, unsupported TP/phase, invalid formulas and unknown capture steps
produce null estimates and an `estimation_error`, while measured module times
remain available. Decode rates require explicit decode phase metadata. These
estimates are not measured hardware counters or end-to-end latency. Config files
are Python files defining `module_config` with strings or `ModuleConf` values.
The old Task-directory scanner and shell wrappers remain removed.

## Result goldens

Generate a candidate from a completed result, review the differences, then
validate the accepted file explicitly:

```bash
python -m rtp_llm.test.perf_test.regression.golden generate \
  --result /path/to/run/results/result.json --output /tmp/result.candidate.json
python -m rtp_llm.test.perf_test.regression.golden validate \
  --result /path/to/run/results/result.json --golden /path/to/accepted.golden.json
```

`generate` atomically replaces its output; use a candidate path before promotion.
`validate` reads the golden without changing it. The corresponding Bazel binary
is `//rtp_llm/test/perf_test:perf_test_golden`.
