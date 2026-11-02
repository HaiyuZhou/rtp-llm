# Proposal

## Why

Continuous cache-pressure execution still retains a legacy `batch_decode` branch with scheduler batch controls, seed barriers, tail padding, and separate result fields even though the supported full-pressure workflow is FIFO-only. Keeping two continuous implementations increases configuration ambiguity and regression surface without a current requirement for non-FIFO pressure.

## What Changes

- **BREAKING** Make every `cache_pressure_mode=continuous` invocation FIFO-only, including the finite legacy continuous mode without preload.
- Reject explicit non-FIFO scheduler selection for continuous pressure at CLI/profile resolution and at the direct `CacheGridRunner` boundary.
- Remove continuous-mode scheduler batch-size control, seed batch switching, tail padding, padding failure handling, and non-FIFO concurrency derivation.
- Simplify continuous manifests, resume configuration, metrics, and summaries so they always record FIFO semantics; any retained compatibility fields for batch size or padding remain neutral and no longer control execution.
- Preserve ordinary `case` mode, fixed-batch cache-grid measurement, batch-decode scheduler controls, and non-pressure batch tests.
- Update tests and operator documentation to describe one continuous-pressure route and a clear migration error for old non-FIFO commands.

## Capabilities

### New Capabilities

- `fifo-continuous-cache-pressure`: Defines FIFO as the sole scheduler contract for continuous cache pressure while preserving fixed-batch behavior outside continuous mode.

### Modified Capabilities

None. The durable capability inventory is currently empty; this change consolidates behavior introduced by the completed but unarchived continuous-pressure changes.

## Impact

- CLI/profile resolution in `rtp_llm/test/perf_test/cache_grid/runner/cache_perf.py` and `rtp_llm/test/perf_test/batch_decode_test.py`.
- Continuous runner construction, resume compatibility, request lifecycle, result schema, and legacy finite continuous execution in `rtp_llm/test/perf_test/cache_grid/runner/cache_grid_runner.py`.
- Continuous-pressure tests in `rtp_llm/test/perf_test/batch_decode_test_test.py` and nearby cache-grid suites.
- Continuous-pressure reference and operator documentation.
- Existing commands that combine continuous pressure with `batch_decode` or a pressure batch size will fail with a FIFO-only migration message; ordinary fixed-batch case commands remain valid.
