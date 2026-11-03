# Tasks

## 1. Plan and compatibility boundary

- [x] 1.1 Add the concurrent-batch measurement contract to new prefill case identities and reject `DP>1` with any batch size greater than one in both ordinary and external prefill planning; verify `prefill_planner_test` and `entry_test` cover stable IDs, early rejection, and `DP=1` gang eligibility.
- [x] 1.2 Reject resume/profile of old prefill plans without the new contract while preserving offline report access; verify `prefill_resume_test`, `prefill_profile_test`, and `prefill_report_test` cover the migration behavior.
- [x] 1.3 Document the DP restriction and changed case IDs in `docs/prefill.md` and `docs/multi_node_aone.md`; verify the documented supported/unsupported topology examples agree with dry-run planning tests.

## 2. Scheduler phase control

- [x] 2.1 Add an acknowledged, retry-bounded prefill scheduler update to the shared HTTP client for local and external sessions; verify `runtime_adapter_test` covers payload, success, rejection, and exhausted retries.
- [x] 2.2 Configure prefill batch 1 before serial cache seeds and the planned batch size before measured rounds, failing the case on either update error; verify `prefill_runner_test` covers cold, cached, batch-one, and failed-acknowledgement ordering.

## 3. Concurrent measured batches

- [x] 3.1 Replace the prefill measurement loop with a bounded thread group that releases all batch members together, waits for all futures, and preserves deterministic member indices; verify `prefill_runner_test` proves overlapping requests and no orphaned member on failure.
- [x] 3.2 Measure each complete round with a monotonic clock, retain per-member first-token timings, and calculate case throughput as the mean of per-round `batch_size * compute_length * 1000 / elapsed_ms`; verify `prefill_runner_test` covers batch 1, batch 4, unequal member latency, invalid duration, and member failure.

## 4. Result integrity and migration

- [x] 4.1 Persist round elapsed time and contract marker with each prefill member measurement, and validate round membership, timing consistency, and aggregate throughput in the result parser; verify `result_envelope_test` covers valid rounds and tampered or incomplete rounds without changing decode parsing.
- [x] 4.2 Make prefill offline reports expose the new measurement contract and exclude invalid persisted rounds, while keeping legacy run directories readable; verify `prefill_report_test` covers corrected, legacy, and inconsistent data.
- [x] 4.3 Confirm old prefill throughput baselines cannot match new case IDs and update regression/golden fixtures only where required by the new contract; verify `baseline_regression_test`, `golden_test`, and the migration guidance in `docs/regression_publish.md`.
- [x] 4.4 Document scheduler phases, concurrent batch semantics, and wall-clock throughput versus server first-token latency in `docs/prefill.md`; verify the documentation examples and assertions in `prefill_examples_test` stay aligned.

## 5. Cross-path verification

- [x] 5.1 Run focused local, single-machine AOne, and gang prefill tests, plus the full `//rtp_llm/test/perf_test/...` Bazel suite and `openspec validate --strict`; verify all pass or record any pre-existing failure separately from this change.

## Verification record (2026-11-03)

- Focused direct unittest: 56 tests passed; complete direct perf-test discovery: 480 tests passed.
- Strict validation: `openspec validate fix-concurrent-prefill-batch-measurement --strict` reports `Change 'fix-concurrent-prefill-batch-measurement' is valid`.
- Both prescribed Bazel commands were attempted. No tests executed: analysis remains blocked by pre-existing missing BUILD files in external `@pip_cpu_torch` (`flash_attn` for the focused run; `triton` on the first full run and `flash_attn` on the final full run). This is the same external dependency packaging blocker recorded during Tasks 1–6, not a test regression.
- Local, single-machine AOne, and TP2 gang integration use the shared HTTP client and concurrent runner against a fake endpoint; four-party barriers prove overlap and AOne collection preserves every case exactly once. DP2/batch4 dry-run is rejected before output creation for all three backends.
