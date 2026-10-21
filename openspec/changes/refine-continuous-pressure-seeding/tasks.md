# Tasks

## 1. Configuration and run classification

- [x] 1.1 Change continuous full-pressure profile/CLI resolution to default to FIFO, at least 16 client in-flight requests, automatic seed count, and a 102400-token fill target; verify focused `cache_perf_test.py` and `cache_perf_pipeline_test.py` cases assert the resolved values.
- [x] 1.2 Reject `batch_decode` and other fixed-batch schedulers at profile resolution and the direct runner/session boundary while preserving ordinary non-pressure batch execution; verify configuration and `continuous_pressure_test.py` regression cases pass.
- [x] 1.3 Add monotonic formal/non-formal classification with reasons for manual seed count, client depth below 16, `test_block_num > 0`, and explicit smoke profiles; verify `result_schema_test.py` covers formal, each individual reason, and multiple accumulated reasons.
- [x] 1.4 Update the reduced DeepSeek-V4-Flash example profile to identify itself as non-formal smoke and add a production profile skeleton without synthetic block capacity; verify both profiles parse and resolve to their expected classifications.

## 2. Live FULL-capacity seed planning

- [x] 2.1 Add immutable seed-plan inputs/results that select required FULL owner pools, deduplicate physical pool IDs, and retain participant provenance without summing TP views; verify unit fixtures cover independent pools, shared TP ownership, and mismatched topology rejection.
- [x] 2.2 Implement reusable-span alignment from `max(32768, max(grid.cache_len))`, keep the commit tail separate, and reject formal grids whose maximum cache length is below 32K; verify boundary cases at 32767, 32768, and unaligned larger lengths.
- [x] 2.3 Implement per-pool token-to-block conversion and the cross-group feasible seed interval using GPU capacity, low watermarks, and live host retention budget; verify exact-capacity `+1`, heterogeneous geometry, multiple groups, reserved-capacity input, and empty-interval diagnostics.
- [x] 2.4 Resolve the automatic plan only after the startup `GetCachePressure` snapshot, retain explicit counts as non-formal overrides, and fail before submission when required FULL evidence is absent; verify runner integration tests distinguish automatic, manual, and infeasible paths.
- [x] 2.5 Validate the aligned 102400-token fill request, seed request, and every formal grid shape for individual model/token/KV admission without silently shrinking them; verify each violated limit names the rejected request shape and constraint.
- [x] 2.6 Persist raw startup evidence and every planner term, source, aligned length, planned/completed count, and classification reason in epoch manifests and summaries; verify `pressure_results_test.py` can independently recalculate a fixture's seed count from serialized fields.

## 3. Rolling FIFO seed lifecycle

- [x] 3.1 Decouple bounded client in-flight depth from active execution concurrency so each request must be admissible while queued RPC count can exceed the active batch; verify a test permits 16 outstanding FIFO requests with a smaller active limit and still rejects an individually inadmissible request.
- [x] 3.2 Replace seed batch barriers with lazy rolling submission that fills the bounded sender, replenishes on completion, stops at the exact target, and never pads; verify deterministic handle tests observe no drain barrier and never exceed the bound.
- [x] 3.3 Add an independent seed-phase deadline and bounded cancellation/drain behavior with submitted, completed, failed, cancelled, and pending counts; verify failure and timeout tests never enter fill or measurement and leave no uncancelled handles.
- [x] 3.4 Integrate the rolling seed phase before fill without charging seed time or transfer evidence to later phases; verify `continuous_pressure_test.py` covers the complete `validate -> seed -> fill -> steady -> measurement -> drain` transition.

## 4. Measurement-window bidirectional migration

- [x] 4.1 Add conservative measurement-window transfer deltas per required FULL owner group for successful DEVICE-to-HOST STORE/EVICT publication and HOST-to-DEVICE LOAD publication; verify `pressure_diagnostics_test.py` covers both directions, D2H-only, H2D-only, failed copy, and out-of-window events.
- [x] 4.2 Apply shared-TP owner attribution and participant freshness/topology checks to transfer evidence without multiplying owner counters; verify owner/participant fixtures accept one valid publication scope and reject missing or changed participants.
- [x] 4.3 Expose `execution_valid`, `full_pressure_valid`, `formal_config_valid`, and `bidirectional_migration_valid` separately and require all four for aggregate formal acceptance; verify result-schema and pressure-result tests preserve each independent failure reason.

## 5. Operator artifacts and documentation

- [x] 5.1 Generate a production DeepSeek-V4-Flash grid with `max(cache_len) >= 32768` and a matching FIFO profile using automatic seed count and at least 16 in-flight requests; verify the cache-grid validation command accepts both artifacts and reports formal eligibility before service startup.
- [x] 5.2 Update `docs/continuous_full_pressure_guide_zh.md` with the production command, smoke command, live seed formula, FULL owner-pool capacity interpretation, expected phase flow, formal gates, and artifact troubleshooting; verify every documented CLI flag against `--help` and execute the documented offline validation commands.
- [x] 5.3 Update `docs/reference.md` and `docs/continuous_full_pressure_plan.md` to mark fixed-batch pressure preload and manual formal seed sizing as superseded while preserving ordinary batch guidance; verify documentation search finds no remaining claim that the 127-block smoke pool is production capacity.

## 6. Integrated and GPU verification

- [x] 6.1 Run the focused cache-grid configuration, continuous-pressure, diagnostics, result-schema, and pressure-result test files together; verify the complete command exits successfully with no skipped new behavior tests.
- [x] 6.2 Run the reduced-capacity DeepSeek-V4-Flash GPU smoke and verify it completes seed/fill/measurement cleanup, produces reproducible planner artifacts, and reports non-formal for the recorded smoke reasons.
- [x] 6.3 Start `/mnt/hf3fs/3fs/models/DeepSeek-V4-Flash` with production-sized cache settings and no `test_block_num`, then run the generated automatic-seed FIFO profile; verify the startup snapshot records actual FULL owner capacity and the client maintains a bound of at least 16 outstanding requests.
- [x] 6.4 Inspect the production run artifacts and verify planned/completed seed counts match, the reusable bank strictly exceeds every required GPU family capacity, pressure coverage passes, and successful D2H publication plus H2D LOAD publication occur inside measurement; record the commands and evidence paths in the acceptance report.

## Workflow follow-up

- Sync or archive `add-continuous-full-pressure-diagnostics` before archiving this dependent capability.
- Archive `refine-continuous-pressure-seeding` after implementation review and both GPU verification tiers are complete.
