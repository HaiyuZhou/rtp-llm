# Local prefill performance tests

The unified prefill mode measures server first-token latency while controlling
the exact cache reuse and compute lengths. It accepts only the versioned unified
configuration format. Update the checkpoint, tokenizer, topology, and device
fields in `examples/prefill_local.json` for the target environment.

## Measurement semantics

Local, AOne, and gang execution use the same prefill runner. Cached cases first
require acknowledgement of prefill scheduler batch size 1, then seed each member
serially. After all seeds pass validation, the runner requires acknowledgement of
the target batch size B before measurement. Cold cases configure B directly.
Scheduler or seed failure stops the case before any measured requests are sent.

Each measured round submits B requests to a reusable B-worker pool and releases
them with a common signal. The runner waits for every member to finish, including
when one fails. Any failed request, invalid shape, incorrect reuse, or invalid
timing fails the whole case and leaves its metrics empty.

Throughput uses the complete round's monotonic wall-clock duration, starting at
the common release and ending after all requests finish. Prompt construction,
seeding, and scheduler updates are outside this interval. Each round contributes
the total computed tokens divided by `round_elapsed_seconds`; the case reports
the arithmetic mean of these round rates, including for B=1. Server first-token
latency is a separate metric: the case records the median across measured member
responses, while the report's batch latency is the median of each round's maximum
member latency. For example, B=4 and compute_len=8 over a 0.2-second round gives
160 tokens/s regardless of individual first-token latencies of 10–40 ms.

Flattened member evidence records the run, member index, server latency, observed
reuse length, shared `round_elapsed_ms`, and measurement contract
`concurrent_batch_v1`. Explicit request groups use
`concurrent_grouped_batch_v1` and include each member's input, reuse, and compute
lengths. All member latencies and round durations must be finite and positive.

## Grouped and ordered grids

`examples/prefill_grouped_local.json` shows `request_batches`, where every batch
has `request_groups` entries with `count`, `input_len`, and `cache_len`. Counts sum
to the measured batch size. `workload.workspace_tokens` is required; planning
checks raw and block-padded token budgets and every member's sequence length.
The runner seeds cached members independently before measuring them together.

`examples/prefill_multi_grid_local.json` shows `workload.grids`. Grid names are
unique and execution follows their declared order in one local engine session.
Each grid has generated cases or explicit `request_batches`; shared alignment,
commit tail, measurement runs, and workspace settings stay at workload level.
The merged report keeps each case's grid name. AOne accepts one grid per run.

Preview either example before running it:

```bash
./tools/perf_test --mode=prefill --action=run --config=rtp_llm/test/perf_test/examples/prefill_grouped_local.json --dry-run
```

```bash
./tools/perf_test --mode=prefill --action=run --config=rtp_llm/test/perf_test/examples/prefill_multi_grid_local.json --dry-run
```

## Run

`run` is the only action that accepts `--config`. The preview validates and
normalizes the config, expands the cache grid, computes fingerprints, and shows
the local resource plan without creating a formal run directory or starting a
model.

```bash
./tools/perf_test \
  --mode=prefill \
  --action=run \
  --config=rtp_llm/test/perf_test/examples/prefill_local.json \
  --dry-run
```

Remove `--dry-run` to execute the plan. The created run directory contains the
source and resolved configs, immutable execution plan, manifest, per-case
results, checkpoints, reports, profiles, and traces. A successful local or AOne
run generates `reports/prefill_report.json`, standalone interactive latency and
tokens-per-minute charts, SVG fallbacks, and `reports/formula/fit_report.json`
in the same command. The fit report records rejection when there is insufficient
data or the validation gates fail; a formula text file appears only for a valid
candidate.

## Resume

`resume` accepts only a run directory created by the unified entry. It checks
the frozen config, workload, model, and runner fingerprints, preserves every
successful case, and reruns the remaining planned cases.

```bash
./tools/perf_test \
  --mode=prefill \
  --action=resume \
  --run-dir=perf_test_results/prefill-run \
  --dry-run
```

Persisted prefill plans must mark every case with measurement contract
`concurrent_batch_v1` or `concurrent_grouped_batch_v1` to resume or profile,
including dry runs. Older directories remain readable with offline `report`,
but new measurements require a fresh
`run --config=...` in a new directory. Existing plans are never migrated in place.

## Profile

`profile` reconstructs selected original requests from their stable case IDs.
Its artifacts are isolated under `profiles/`; formal case results, checkpoints,
and summaries remain unchanged. Case IDs come from the run's execution plan.

```bash
./tools/perf_test \
  --mode=prefill \
  --action=profile \
  --run-dir=perf_test_results/prefill-run \
  --cases=prefill-0123456789abcdef01234567 \
  --dry-run
```

## Report

`report` is offline. `--skip-test` selects the same prefill report action when
`--action` is omitted. It reads the immutable plan and committed case files,
excludes missing or invalid points, and writes a formal report only when all
planned cases are valid. Otherwise it writes a diagnostic report and exclusion
reasons. It does not launch a model or modify measured cases.

```bash
./tools/perf_test --mode=prefill --skip-test --run-dir=perf_test_results/prefill-run
```

```bash
./tools/perf_test \
  --mode=prefill \
  --action=report \
  --run-dir=perf_test_results/prefill-run \
  --dry-run
```

`--skip-test` requires prefill mode and `--run-dir`; it accepts an optional
`--action=report` and supports `--dry-run` preview. It cannot be combined with
run, resume, profile, or a config override. `resume`, `profile`, and `report`
never accept a config override. Local prefill supports all four actions; AOne
prefill supports `run` and offline `report` only.

## Contract labels and migration

Prefill plans containing any `batch_size > 1` require `model.topology.dp == 1`
on every backend. Plans containing only batch size 1 still support DP > 1; TP
and PP remain supported. The concurrent batch measures one engine, so
distributing members across independent DP engines would change its meaning.

JSON observations and the HTML report expose `measurement_contract`:
`concurrent_batch_v1` uses measured round wall time for throughput, while
`legacy_latency_derived_v1` labels historical latency-derived observations.
Offline reports still read old directories without rewriting plans or results.
New-contract reports validate member indices, reuse, shared round durations,
and aggregate latency and throughput against the persisted raw evidence.
Inconsistent cases are excluded and make the report diagnostic.

The contract marker participates in stable case IDs, so fresh runs have different
IDs even for the same lengths and batch size. Old baseline keys fail with missing
cases; they never silently match the new measurements. Result schema v2 remains
unchanged. Explicitly generate and review a new baseline candidate and golden
from a fresh run; normal runs and report generation never rewrite accepted
baselines or goldens. See `regression_publish.md` for the commands.
