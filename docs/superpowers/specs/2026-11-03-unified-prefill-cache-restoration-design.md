# Unified Prefill Cache Restoration Design

## Intent

Restore the capabilities added in `81b06495f4450ee996d27f2218e15b3e8d2e35d4` that disappeared during the unified perf-test migration, while keeping `./tools/perf_test` and its persisted execution plan as the sole public path. A local or AOne prefill run should emit reviewable cache measurements, offline charts, and a FlexLB formula candidate. A completed run can regenerate all derived analysis without GPU access.

## Scope and compatibility

- Preserve existing flat `workload.kind=cache_grid` configurations, `concurrent_batch_v1` results, and decode behavior.
- Add explicit heterogeneous request groups, each with `count`, `input_len`, and `cache_len`. The sum of counts is the batch size. Validate every member against sequence length and packed token budgets before launching the engine. Seed requests use independent prefixes and scheduler prefill batch 1; measured members submit concurrently after setting the planned batch size.
- Add an ordered `workload.grids` array, with unique grid names and per-grid generated or explicit cases. Local execution uses one model, engine session, and result directory, and runs cases grid by grid. This restores sequential grids and merged analysis, without requiring an engine restart between grids. AOne accepts one grid per run because independent shards cannot guarantee global grid order.
- Use a separate versioned measurement contract for grouped cases. Persist per-member geometry, observed reuse, server first-token latency, and one round wall time. A grouped case's aggregate token lengths are sums; fitting uses the full request distribution.
- Restore a cache-block probe before the first measured case in each engine session. The probe seeds one physical block and tests a two-block continuation; abort before formal measurements if observed reuse is wrong. Compute the physical block size from engine block size and CP sharding settings.
- Restore an opt-in DashSC gRPC `input_ids` transport alongside existing HTTP text. Preserve the same response shape and server latency units; reject unsupported environments at planning or startup rather than silently falling back.
- Restore isolated Kineto and Nsight Systems trace collection for selected local prefill cases. The `profile` action writes trace files and a manifest, never updates formal case results. Trace overhead never enters formula fitting.
- Extend offline `report` to write self-contained interactive latency/TPM HTML and static fallbacks. Use per-round maximum member server latency as the batch latency, and measured round wall time for throughput.
- Restore restricted symbolic FlexLB `PREFILL_TIME_FORMULA` candidate fitting, deterministic train/validation/test separation, per-batch and global error gates, `model.json`, `fit_report.json`, `predictions.csv`, and fit gap graphic. A rejection is explicit; no formula is marked production accepted merely because a text file exists. Homogeneous and grouped concurrent contracts share the same target semantics and may fit together after normalization; historical latency-derived observations are excluded.
- Add `--skip-test` as an offline alias for prefill report regeneration on `--run-dir`. It reads only persisted plans and case results, updates derived reports, and never starts an engine. `--action=report` retains the same behavior.

The new workload fields are `transport: "http_text" | "dashsc_input_ids"`, optional `probe_block_size`, optional `workspace_tokens`, and `profiling: {"backend": "kineto" | "nsys", "trace_timeout_seconds": positive number, "nsys_path": optional path}`. `request_batches` is an array of `{request_groups: [{count, input_len, cache_len}]}` and requires `workspace_tokens`. A `grids` member adds a unique `name` to either an ordinary generated grid or an explicit `request_batches` grid; top-level cache alignment, commit tail, measurement runs, transport, and profiling settings are shared. Flat existing workloads remain valid without new fields. The probe expects `probe_block_size` when supplied, otherwise `engine.args.seq_size_per_block` when supplied, otherwise `cache_alignment`; a wrong expectation fails the probe before measurements.

Use the historical defaults for formula admission: at least 30 unique valid geometries, held-out maximum MAPE 5%, p95 absolute percentage error 10%, and maximum absolute percentage error 40%. Keep the historical restricted FlexLB expression grammar and include the report's acceptance reason.

## Approaches

1. Restore `tools/cache_perf` and the old cache-grid result format. This directly reuses historical files but creates a second benchmark stack and a conversion problem for current concurrent results.
2. Extend the unified config, runner, transport, profiler, and offline report. Use the historical algorithms where their semantics still match and adapt outputs to versioned unified observations. **Chosen by implementation direction:** this keeps one execution plan and supports both old and new workloads.

## Data and execution design

The execution plan stores grid identity and request groups in each case identity. Existing homogeneous cases retain their identity shape. Grid order is stable and cases run in plan order. For grouped identities, the summary `input_len` and `cache_len` are the member maxima to match the existing prompt helper; metrics and report observations carry aggregate sums and the complete per-member list. For explicit grouped cases, token budget checks compare raw member input sum with `engine.max_batch_tokens` and block-padded member sum with `workspace_tokens`, accounting for one output token per member and sequential seeds. The preflight is an admission check, not a guarantee of GPU free memory.

The runner constructs exact-length prompts, validates text/token IDs before requests, and validates every response's shape, reuse, and positive latency. HTTP uses text; DashSC uses the constructed IDs. A probe result is persisted separately and excluded from formal observations. Run the probe outside the adapter's per-case exception handling; a failed probe stops the run before any formal case is committed.

Reports validate committed cases against the immutable plan. Each observation carries the full per-request length distribution and round measurements. Formula targets use the median over rounds of each round's maximum server first-token latency. TPM uses complete round wall time. Multi-grid reports include source grid and exclusions so merged rows remain auditable. A diagnostic report from partial data cannot gain production acceptance. Regeneration removes or replaces stale accepted formula artifacts before writing a fresh rejection.

## Verification

Write failing unit tests before each feature, including grouped token admission, exact reuse validation, HTTP/DashSC response parity, probe failure, trace artifact isolation, chart/fit regeneration, and no-engine `--skip-test`. Run the relevant unified perf-test suite plus a focused end-to-end fixture. GPU-dependent tests may be unavailable locally; tests must expose that limit rather than claim a live engine validation.
