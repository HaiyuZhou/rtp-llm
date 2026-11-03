# Design

## Context

See `proposal.md` for motivation. `PrefillRunner` is shared by local, single-machine AOne, and gang prefill. It currently makes synchronous HTTP calls through `HttpPrefillClient`; the client has no scheduler-control method. The server starts `BatchDecodeScheduler` with a default batch target of one, while `/update_scheduler_info` broadcasts `{batch_size, mode}` to DP backends. A target greater than one is a gate: sequential seed or measurement requests can wait forever for peers that have not yet been issued. Decode already configures this endpoint and sends each round concurrently.

The prefill result envelope is schema v2. Prefill case identity has no measurement-contract field, and the existing `measurements` list contains one item per request, not per round. Baseline comparisons key on stable case IDs.

## Goals / Non-Goals

**Goals:** Make a planned prefill batch an actual overlapping set of requests; keep cache seeding and reuse checks deterministic; calculate throughput from observed batch elapsed time; prevent old and corrected throughput from matching the same baseline identity. Preserve the public CLI and config shape.

**Non-Goals:** Redesign decode, implement DP-rank-aware HTTP routing, or turn the separate offline throughput benchmark into the unified entry. This change does not repair unrelated prefill GPU-profile triggering.

## Decisions

1. **Use a phase-aware scheduler contract on the shared prefill client.** Add `configure_scheduler(batch_size, mode="prefill")` to the HTTP client, with bounded retries and required `status: ok` acknowledgement, for local and external AOne sessions. At the start of each case, set prefill batch 1 before any cache seed. Complete and validate each seed synchronously, then set prefill batch `B` before measured rounds. Cold cases can skip seeds but still set the measured target. A scheduler error is a case failure before measurement. This reuses the existing endpoint; merely setting engine `concurrency_limit` does not configure the scheduler.

2. **Create one bounded request group per measured round.** Prepare all prompts before timing. Use a thread pool of size `B` and a common release signal so each member enters its blocking HTTP call without waiting for another member's response. Start a monotonic round clock immediately before releasing the group and stop only after every submitted future has completed or failed. Preserve member indices when gathering, validate all responses, and return a failed case rather than a partial successful batch if any member fails. The pool can be reused across rounds of one case. Threads are an implementation choice; an async transport would also satisfy the spec but would require a larger adapter rewrite.

3. **Persist an auditable measurement contract without globally bumping decode results.** Add `measurement_contract="concurrent_batch_v1"` to every new prefill case identity. Stable case IDs consequently change, so old throughput baselines do not match; reject resume/profile of old prefill plans that lack this field, while keeping offline inspection/report of old results available. Keep the current flattened per-member `metrics.measurements` shape, adding the measured round elapsed time and contract marker to each member record. The prefill result validator must require exactly `B` unique members per round, one consistent positive elapsed time per round, and an aggregate throughput equal to the arithmetic mean of `B * compute_length * 1000 / elapsed_ms` across rounds. `latency_ms` remains the median of server-reported first-token timings. Reports expose the contract and exclude inconsistent rounds. This avoids forcing a schema v3 change on decode while making new and legacy prefill results distinguishable.

4. **Limit multi-DP batch sizes until routing is controllable.** `/update_scheduler_info` broadcasts one batch target to every DP backend, but the current prefill HTTP adapter cannot direct each member to a specific rank. For `DP>1`, reject any prefill plan containing `B>1` before run-directory creation or AOne submission. `B=1` remains supported; `DP=1` includes multi-machine TP/PP gangs. This is safer than assuming the frontend balances exactly enough requests to satisfy each backend gate. The restriction is documented and covered at both ordinary and external-plan entry points.

## Risks / Trade-offs

- [Threaded requests may start a few milliseconds apart] → Use a common release signal, prepare prompts and pool before timing, and record actual elapsed time instead of claiming perfect simultaneity.
- [One request fails while peers are blocked in the scheduler] → Submit the full batch before collecting futures, use the client's bounded request timeout, and wait for/account for all futures before returning failure.
- [Client wall-clock throughput differs from server first-token latency] → Keep the metrics separate and label their sources; never derive batch throughput from server latency.
- [Historical prefill baselines and resumed plans are incompatible] → Change prefill case identities, fail old-plan resume/profile explicitly, keep old results available for offline inspection, and document explicit baseline regeneration.
- [Multi-DP batch tests are temporarily unavailable] → Reject them in planning rather than risking a hung or mislabeled result; rank-aware transport is a separate future change.

## Migration Plan

Deploy the shared runner, HTTP adapter, validator, and docs together. Regenerate prefill baselines and goldens from corrected runs; do not rewrite accepted baselines automatically. Existing run directories remain readable offline, but rerun an old plan from a new config instead of resuming it. Rollback restores the old runner and its old result contract; do not compare or merge measurements across the two contracts.
