# Proposal

## Why

Unified prefill plans accept batch sizes greater than one, but the runner sends each member synchronously and estimates batch throughput from one request's latency. This neither exercises the requested concurrent batch nor reports measured batch throughput; the prefill scheduler is also left at its startup defaults.

## What Changes

- Configure the BatchDecodeScheduler explicitly for prefill cases. Keep cache-seed requests runnable before switching to the target measurement batch size.
- Send all members of each measured prefill round concurrently, wait for the whole round, and preserve per-member shape, reuse, and timing validation.
- Derive prefill batch throughput from the measured round duration rather than multiplying a single-request latency; retain per-request latency as a distinct metric.
- Cover batch size one and greater-than-one, cold and cached cases, scheduler failures, request failures, and local/AOne shared runner paths with tests and documentation.
- Reject multi-DP prefill cases with batch size greater than one during planning until requests can be routed to DP ranks deterministically; TP/PP multi-machine gangs with DP=1 remain supported.
- **BREAKING:** the prefill throughput denominator changes to measured round elapsed time for every batch size, most materially for batch sizes greater than one; affected baselines must be regenerated explicitly, and old prefill plans cannot be resumed under the new measurement contract.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `prefill-cache-performance`: define scheduler configuration, cache seeding, and concurrent measured-batch behavior for every prefill backend.
- `perf-test-results`: define the measured batch-throughput denominator and preserve enough round-level evidence to audit it.

## Impact

The shared prefill runner, its HTTP/runtime adapter, local and AOne execution paths, prefill result validation and reports, tests, and performance-test documentation are affected. The public command and config shape remain unchanged; decode and the standalone offline benchmark are out of scope.
