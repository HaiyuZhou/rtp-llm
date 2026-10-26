# Tree Fill + Measurement Mock Memory Plan

## Goal

Add a diagnostic-only continuous-pressure mode that first builds real Device/Host cache pressure through the normal cache tree, then applies cumulative block-aligned mock memory reuse only to measurement requests.

## Global Constraints

- Hybrid startup probes plus `seed`, `fill`, `steady`, and `support` requests use normal cache lookup/store behavior.
- `fill` must reach the existing real tree-pressure readiness gate and preserve the existing 80% Host-pool fail-closed guard.
- Tiered rewarm is disabled for this combined policy.
- Only hybrid requests classified as `measurement` opt into mock memory planning.
- The request control is tri-state: absent preserves legacy service-global mock behavior, explicit false selects the natural tree, and explicit true selects mock only when the service source is enabled.
- Mock memory planning remains cumulative by eligible block-aligned token blocks, not by request count.
- A measurement request's mock decision is latched for its full lifetime. Phase changes and global state changes cannot alter its completion-time store behavior.
- Mock reads use the existing immutable Host reservoir and perform real Host-to-Device transfer. Successful mock requests perform real Device-to-Host writeback.
- The combined acceptance requires both real fill/tree-pressure evidence and measurement-window mock H2D/D2H evidence with zero failed transfers.
- Existing `tree_pressure` and `mock_memory` behavior remains backward compatible.
- The mode remains diagnostic-only and cannot become a formal performance candidate.

## Task 1: Request-Scoped Engine Gate

Write failing Python/C++ tests, then add a request-scoped `mock_memory_cache_request` control from the pressure RPC request through `GenerateConfig`, `StreamCacheResource`, and `MallocInfo`. Explicit true enters mock planning only when the service-level source is enabled; absence retains legacy service-global behavior. Latch the mock plan on the request resource and use that latch for retries, Host writeback, and mock metrics at completion, including zero-planned-block requests.

Verify focused codec/config, stream-resource, and cache-manager tests.

## Task 2: Combined Runner And Profile Policy

Write failing runner/profile/diagnostics tests, then add `tree_then_mock_memory`. It must run seed -> real fill -> steady -> measurement -> drain, skip tiered rewarm, mark only measurement transport requests with `mock_memory_cache_request=true`, retain tree readiness during preparation, and evaluate mock migration deltas only for measurement acceptance. Relax profile validation only for the valid engine-flag/policy combination.

Add/update an example profile and documentation describing the exact data source and phase behavior.

Verify focused cache-grid tests.

## Task 3: Integration Review And Verification

Run the complete relevant Python and C++ suites, formatting/static checks, and an independent whole-branch review. Fix blocking findings with focused regression tests. Then commit and push only after verification; remote deployment and DeepSeek V4.1 Flash runtime validation are a separate externally mutating step requiring the already established SSH/container workflow.
