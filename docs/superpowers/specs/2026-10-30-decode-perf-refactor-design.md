# Decode Performance Refactor Design

Date: 2026-10-30
Status: Approved and OpenSpec strict-validation passed
OpenSpec change: `refactor-decode-perf-test`

## Scope

Refactor decode performance testing behind the unified perf-test entry while adding fixed input-by-output grid coverage. The change keeps decode local-run only, keeps distribution output length scalar, preserves DP scheduling and target-TPOT optimization, retains inline timeline profiling, and migrates the externally managed multi-node path to the same execution core.

Prefill behavior, prefill follow-up actions, AOne prefill, decode resume, standalone decode profile/report, AOne decode, and decode regression publishing are outside this change.

## Configuration Contract

Exact fixed grid:

```json
{
  "kind": "grid",
  "input_lengths": [128, 512],
  "output_lengths": [32, 128],
  "batch": {
    "strategy": "fixed",
    "sizes": [1, 4, 8]
  },
  "profiling": {
    "enabled": false
  }
}
```

Target-TPOT fixed grid:

```json
{
  "kind": "grid",
  "input_lengths": [128, 512],
  "output_lengths": [32, 128],
  "batch": {
    "strategy": "target_tpot",
    "max_size": 64,
    "candidate_step": 4,
    "target_tpot_ms": 10.0
  }
}
```

Distribution retains `output_length` and uses the same `batch` strategy object. A searchable distribution must expose one sequence-length list with at least `max_size` entries; a per-batch map is valid only for exact batches.

The old fixed `output_length`, top-level `batch_sizes`, and top-level `target_tpot_ms` fields are rejected. Planned output length overrides conflicting `min_new_tokens` and `max_new_tokens` in `generate_config`.

## Execution Architecture

```text
normalized decode config
        |
        v
typed workload planner
        |
        v
DecodeCaseSpec
        |
        v
DecodeCaseExecutor
        |
        +---- FixedBatchStrategy
        |
        +---- TargetTPOTStrategy
                     |
                     v
             DecodeBatchExecutor
                     |
              DP allocation
              warmup/measure
                     |
                     v
             DecodeHttpTransport
                     |
                     v
           metrics / profile / report
```

Planning expands exact fixed workloads as `input x output x batch` and target-TPOT workloads as `input x output`. Fixed and distribution cases use the same executor after planning materializes their input lengths.

The batch strategy decides which batch sizes to measure. The batch executor decides how to measure one batch. The transport owns only service-facing scheduler and request operations.

## TPOT Semantics

Each input/output pair searches independently. Candidate construction retains the existing bounded stepped search and records every attempted candidate exactly once.

The largest candidate with full request success and TPOT at or below the target is selected. Formal case metrics and optional profiling use that selected batch. The maximum batch is only an upper bound and is not forced into the final result. If no candidate qualifies, the case becomes `target_tpot_not_met` and retains all observations.

## DP Semantics

Batch size always means global batch. Requests are allocated deterministically across DP workers, including uneven allocations and idle workers when global batch is smaller than DP degree. Results retain global batch, topology, worker allocations, and actual input lengths separately.

## Inline Profiling

Inline profiling remains part of decode `run`, not a separate action, and is disabled by default. It executes after formal measurement so profiler overhead cannot contaminate performance metrics.

Exact cases profile their measured batch. TPOT cases profile only the selected batch. Every attempt writes `profiles/<case_id>/profile_manifest.json`. Profile failures are visible in the case and summary but do not invalidate otherwise valid performance metrics.

## Results And Reports

Case identity includes workload kind, input shape, output length, and batch policy. Exact results report the configured batch; TPOT results report the selected batch and retain search limits and observations separately.

`decode_report.json` contains ordered flat observations for all workloads. Exact grids add one input-by-output matrix per batch. TPOT grids add one input-by-output matrix whose cells contain best batch, actual TPOT, throughput, and search outcome. Distribution remains an ordered table.

## Multi-Node Migration

All ranks load the same normalized config and validate it against the running TP/DP/PP/EP topology. The existing launcher continues to own distributed service lifetime. Only the master executes the unified plan and writes the standard run directory; worker ranks do not execute duplicate cases.

Collection moves from `Decode_Result.json` and log-table parsing to the standard result envelope and report. Trace analysis uses profile manifests and shape-complete trace identity, including output length.

## Removed Interfaces

The change removes `run_single`, `run_cartesian_grid`, the legacy grid runner, fixed/distribution execution duplication, `DecodeHttpBatchRunner`, `BatchPerfImpl`, and the old decode result format. No compatibility wrapper or dual-write period is provided.

## Verification

Implementation verification covers strict config/planner tests, exact and TPOT strategy tests, DP allocation, HTTP transport, selected-batch metrics, profiling failure isolation, matrix reports, multi-node master/worker behavior, legacy import removal, Bazel/Python integration tests, dry runs, and real-GPU smoke tests using `/mnt/hf3fs/3fs/models/Qwen2.5-0.5B-Instruct/`.

The authoritative proposal, deltas, detailed decisions, and implementation checklist are under `openspec/changes/refactor-decode-perf-test/`.
