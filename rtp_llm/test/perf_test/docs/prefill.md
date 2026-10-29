# Local prefill performance tests

The unified prefill mode measures server first-token latency while controlling
the exact cache reuse and compute lengths. It accepts only the versioned unified
configuration format. Update the checkpoint, tokenizer, topology, and device
fields in `examples/prefill_local.json` for the target environment.

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
results, checkpoints, reports, profiles, and traces.

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

`report` is offline. It reads the immutable plan and committed case files,
excludes missing or invalid points, and writes a formal report only when all
planned cases are valid. Otherwise it writes a diagnostic report and exclusion
reasons. It does not launch a model or modify measured cases.

```bash
./tools/perf_test \
  --mode=prefill \
  --action=report \
  --run-dir=perf_test_results/prefill-run \
  --dry-run
```

`resume`, `profile`, and `report` never accept a config override. Local prefill
supports all four actions; AOne prefill supports `run` and offline `report`
only.
