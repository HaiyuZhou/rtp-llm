# Proposal

## Why

Topologies beyond one machine (for example TP16 across two 8-card hosts) cannot run under any supported AOne path: AOne prefill shards independent single-machine cases, multi-node decode only works through the manual SSH flow over a static IP list, and decode with the AOne backend is rejected outright. The AOne CI mechanism for one task occupying multiple machines is already proven in production by the K3 PD two-host smoke pipeline (Bazel `shard_count` + remote TestRunner + HTTP rendezvous broker), and the perf side already owns the multi-node execution core and the CI control-plane shape, so wiring them together behind the unified entry is now mostly assembly rather than invention.

## What Changes

- Add gang mode to the unified AOne backend: `machine_count > 1` derived from the new optional `execution.gpus_per_machine` field; N AOne machines form one gang, each hosting `local_world_size` ranks of one distributed model, and the master machine executes the unified perf plan.
- Allow `decode + aone + run` in gang mode (today rejected before config is read); single-machine decode + aone moves from a capability rejection to a plan-validation rejection naming `gpus_per_machine`/`machine_count`. Decode aone `resume`, `profile`, and `report` remain rejected.
- Run prefill in gang mode against one externally managed distributed server (new external prefill plan builder and session handle); the single-machine case-sharded AOne prefill path is unchanged (`machine_count == 1` keeps today's behavior exactly).
- Add an in-process HTTP rendezvous broker on the submit host generalizing the K3 PD pattern to N symmetric members: register/members/heartbeat/result/cleanup with token auth, distinct-host validation, register and run deadlines, and persisted rendezvous diagnostics.
- Generate a gang Bazel package: one `py_test` with `shard_count = machine_count`, each shard requesting the registered per-host GPU count (machine exclusivity), with the worker restricting itself to `local_world_size` devices via `CUDA_VISIBLE_DEVICES`; deliver `resolved_config.json` and `gang_plan.json` as runfiles and pass broker URL/token/run id through `env_inherit`.
- Collect the master shard's full `run/` tree from `bazel-testlogs` (envelope, reports, traces — parity with local runs); non-master shards contribute logs only.
- Add `.aoneci/multi-node-perf.yaml` (internal repo) whose job runs the identical `./tools/perf_test --mode=<mode> --action=run --config=<config>` entry (with `execution.backend: "aone"` set in the delivered config) inside the AOne CI runner (phase A submit host); local dev-machine submission is phase B and stays unvalidated.
- Reject conflicting gang semantics at config validation: explicit `machine_count`, `shard_count` in gang mode, non-divisible `world_size`, and `max_parallel < machine_count` all fail before any side effect.

## Capabilities

### New Capabilities

- `multi-node-aone-orchestration`: Gang derivation from topology, gang formation through the rendezvous broker, the gang worker contract, gang Bazel package generation with machine exclusivity, gang result collection, and the phase A CI pipeline entry.

### Modified Capabilities

- `perf-test-orchestration`: Allow `decode + aone + run` at capability validation, move the single-machine decode+aone rejection into plan validation, extend dry-run and runfile delivery to gang plans, and add the `gpus_per_machine` execution field with its derivation and cross-validation rules.
- `decode-performance`: Decode execution is no longer local-only; decode `run` is additionally supported with the AOne backend in gang mode against the externally managed distributed server.
- `prefill-cache-performance`: AOne prefill in gang mode executes the whole plan against one externally managed distributed server instead of case-sharded single-machine shards; single-machine AOne prefill keeps exclusive case ownership.

## Impact

- Affected code: `rtp_llm/test/perf_test/config_schema.py` (execution fields, gang derivation), `capabilities.py` (matrix), plan validation (single-machine decode+aone rejection), `aone/package.py` + `aone/defs.bzl` (gang rule, `gpus_per_host` registry), `execution/aone.py` (gang executor stages), `execution/runtime.py` (`HttpPrefillClient` explicit endpoint), `multi_node/decode_runtime.py` + `multi_node/local_server_runner.py` (reusable extraction; legacy SSH flow unchanged), entry wiring, BUILD targets, examples, and docs.
- New code: broker module, gang plan/package generation, gang worker, prefill external plan builder and session handle.
- Internal repo (not open source): `.aoneci/multi-node-perf.yaml` pipeline.
- Public error surface: the "unsupported capability" failure for decode+aone becomes a planning failure with a different message; decode spec scenario "Unsupported AOne decode" is superseded by gang-aware scenarios.
- Acceptance: real AOne smokes — decode TP2 across 2 machines and prefill TP2 across 2 machines with `Qwen/Qwen2.5-0.5B-Instruct` on `H20_CU13`/`daily` — plus no regression in existing single-machine AOne prefill and local paths.
