# Proposal

## Why

The decode performance path currently splits fixed grids, distributions, TPOT search, HTTP execution, and externally managed multi-node runs across overlapping runners, and fixed grids cannot vary input and output lengths together. Refactoring now establishes one decode execution model before more decode capabilities are added, while preserving the existing local-only capability boundary.

## What Changes

- **BREAKING** Replace the fixed decode workload schema with explicit `input_lengths`, `output_lengths`, and a `batch` strategy object; scalar `output_length`, `batch_sizes`, and top-level `target_tpot_ms` are no longer accepted for fixed grids.
- Add fixed-grid Cartesian execution across input length, output length, and exact batch size.
- Separate exact-batch execution from target-TPOT optimization; target-TPOT searches independently for each input/output pair and reports metrics for the selected best batch.
- Keep distribution workloads on one scalar output length while allowing the same exact-batch or target-TPOT strategy contract.
- Consolidate fixed, distribution, local, and externally managed multi-node decode runs onto one typed plan, case executor, DP scheduler, HTTP transport, metric builder, and finalizer.
- **BREAKING** Remove the legacy `run_single` and `run_cartesian_grid` callable APIs, the old decode grid result format, and their runner implementation without compatibility wrappers.
- Add optional inline decode profiling during `run`, disabled by default; fixed cases profile their measured batch and TPOT cases profile only the selected best batch.
- Replace the basic decode report with machine-readable observations plus input-by-output matrix views grouped by batch strategy.
- Keep decode limited to the local `run` action; decode resume, standalone profile/report actions, and AOne decode remain unsupported.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `decode-performance`: Add fixed input-by-output grids, explicit batch strategies, selected-batch TPOT semantics, optional inline profiling, unified multi-node execution, and the breaking workload/API removals.
- `perf-test-orchestration`: Require decode execution, including externally managed multi-node execution, to consume the normalized immutable plan through the unified decode core.
- `perf-test-results`: Extend decode case identity, metrics, search, profile manifests, and reports to represent output-grid and selected-batch results without ambiguity.

## Impact

- Affected code: `rtp_llm/test/perf_test/config_schema.py`, decode planning/execution/reporting modules, the default local executor, entry planning, BUILD targets, examples, docs, and decode tests.
- Removed code and APIs: `decode/grid_runner.py`, `decode/fixed.py`, `decode/http_batch.py`, `batch_perf_impl.py`, its decode-only metric data structures, and the stable legacy grid API tests.
- Multi-node impact: launchers distribute the normalized config, the master invokes the unified decode core, result collection consumes the standard run directory, and trace analysis uses profile manifests rather than filename inference.
- Result consumers must migrate from `Decode_Result.json` and scalar fixed-grid output assumptions to the versioned common envelope and decode matrix report.
- Prefill behavior and AOne prefill execution are unchanged.
