# Design

## Context

The unified perf test already runs AOne prefill by sharding independent single-machine cases (`aone/` package: submit-host plan, generated Bazel package, remote build/test, testlogs collection), runs decode only locally, and runs multi-node decode only through the manual SSH flow over a static IP list (`multi_node/`). The AOne CI mechanism for one task occupying multiple machines is proven in production by the K3 PD two-host smoke pipeline (internal repo git history `e378bfbff6..8686c07d0f`): a `py_test` with `shard_count = N` executed through `--strategy=TestRunner=remote` lands each shard on a distinct GPU machine (GPU accounting exclusivity), and an HTTP rendezvous broker on the submit host exchanges member addresses through `env_inherit`-delivered URL/token/run-id environment. The open-source implementation is clean-room: K3 code is a mechanism reference only, never copied.

Platform facts that shape the design: one remote action caps at 7200 s (test timeout stays 7000 s); one Bazel action runs on exactly one machine; broker reachability from remote workers to a CI runner and worker-to-worker connectivity are proven; hub model fetch from remote workers is proven (2026-10-29 AOne prefill smokes ran `Qwen/Qwen2.5-0.5B-Instruct` on `H20_CU13`/`daily`); model data must stay external to the generated package.

## Goals / Non-Goals

**Goals:**

- One AOne task occupies N machines as one gang running one distributed model, for both decode and prefill, driven by the existing unified entry and producing local-parity artifacts.
- Keep every existing single-machine path byte-for-byte unchanged in behavior.

**Non-Goals:**

- Phase B (local dev-machine submission) — the entry stays phase-B ready but worker-to-dev reachability is unvalidated.
- Auto-retry of failed gangs, queue/pool management beyond Bazel `--jobs`, model-weight distribution, and case partitioning across parallel gangs.
- Consuming the pending `add-cache-perf-local-aone-runners` change.

## Decisions

### Gang mechanism: Bazel shards + remote TestRunner + broker rendezvous

A gang is one generated `py_test` with `shard_count = machine_count` and remote TestRunner execution; each shard action is one machine. Members discover each other through an HTTP broker started in-process on the submit host, generalizing the K3 two-role broker to N symmetric members. Alternative — extending the SSH flow with AOne-allocated IPs — was rejected because the TestRunner resource model does not hand out SSH machine lists and it would fork a second resource-leasing system.

### Configuration: derived machine count, conflicting fields rejected

`execution.gpus_per_machine` is optional and defaults to `world_size`, so omission keeps `machine_count == 1` and today's single-machine path exactly. `machine_count` is always derived (`ceil(world_size / gpus_per_machine)`); an explicit field is rejected. In gang mode `shard_count` is rejected (case-shard and gang semantics are mutually exclusive); `world_size` must divide evenly; `max_parallel >= machine_count` is validated, not clamped. Build parallelism stays an environment override (`PERF_TEST_BAZEL_BUILD_JOBS`), not a config field. All of this fails during config validation, before the run directory or any remote side effect.

### GPU exclusivity: full-host request plus device restriction

Requesting only `local_world_size` GPUs on a larger host does not reserve the machine — the TP2 smoke (1 GPU per member on 8-card hosts) could co-land two members on one host and fail by construction. The generated gang rule therefore requests the registered per-host GPU count of the selected GPU type (the GPU registry gains `gpus_per_host`), and the worker restricts itself to `local_world_size` devices via `CUDA_VISIBLE_DEVICES`. Broker same-host validation is the backstop, not the mechanism.

### Capability layering: config-blind matrix, plan-level gang rejection

`validate_capability` stays blind to config content: the matrix unconditionally allows `decode + aone + run` (run only). The single-machine decode+aone rejection moves to plan validation, which already owns topology-derived cross-field rules; its message names `gpus_per_machine`/`machine_count`. This changes one public error surface (capability error becomes planning error) and is captured in the spec deltas.

### Broker: N symmetric members, source-address recording, two deadlines

Operations: `register` (shard index, hostname, port), `members` (blocks until all registered or the register deadline), `heartbeat`, `result`, `cleanup`. Bearer-token auth on every request; the broker serves only bounded run metadata (no commands, no source files). Workers do not self-discover their routable IP: the broker records each registration's HTTP source address and cross-checks the hostname, rejecting mismatches; the recorded address feeds `GANG_CONFIG_STRING`. Validations: one registration per shard, distinct hosts, consistent config fingerprint. Deadlines: register 3600 s (placement-starved gangs fail fast), run 43200 s safety net, heartbeat expiry 120 s. Rendezvous state and events persist under the run directory and are archived into `bazel-testlogs` next to the master outputs.

### Worker contract: runfiles in, one plan on the master

Each member reads `resolved_config.json` and `gang_plan.json` (world size, machine count, local world size, topology, master shard index, port range, fingerprints) from runfiles, resolves model references through the existing `resolve_remote_runtime_inputs`, registers, builds `GANG_CONFIG_STRING` ordered by shard index with `WORLD_RANK = shard_index * local_world_size`, and starts its local rank group through an entry extracted from `local_server_runner.py` (the legacy SSH scripts keep their behavior and call the same functions). The master waits for gang readiness and executes the frozen plan — decode reuses `build_external_decode_plan` + `execute_master_decode`; prefill uses the new external path. Non-master members wait for master completion. Any member failure, heartbeat expiry, or deadline miss fails every member nonzero while preserving the master's partial `run/` artifacts.

### Prefill multi-node: external plan builder plus session handle

A prefill counterpart of `build_external_decode_plan` validates configured topology against the gang topology and freezes the prefill plan. An external session handle exposes the same `.client` contract as the local engine handle — an `HttpPrefillClient` bound to the gang server's frontend endpoint — without owning process lifecycle. `HttpPrefillClient` gains an explicit host:port endpoint instead of the hardcoded loopback. The prefill runner, metrics, envelope, and report path are reused unchanged.

### Artifacts and staging

The master member's full `run/` tree is collected from `bazel-testlogs` (envelope, reports, traces — parity with local runs); non-master members contribute logs only. The gang executor follows the existing AOne executor stage pattern (rendezvous, build, submit, collect, report): stage transitions persist to the manifest, the original test failure code wins over collect/report errors, and partial collection always runs.

### Entry and CI: one entry, phase A on the CI runner

The CI pipeline job runs the identical `./tools/perf_test --mode=<mode> --action=run --config=<config>` command (with `execution.backend: "aone"` set in the delivered config) on the AOne CI runner; the entry process starts the broker, generates the gang package, drives remote build/test, and collects. No low-level command chaining. A concurrency group keeps overlapping gangs from double-booking pool machines. Local dev-machine submission is phase B, deferred until worker-to-dev connectivity is validated.

## Risks / Trade-offs

- [Worker-to-submit-host reachability breaks under pool isolation] → gang registration fails fast at the register deadline instead of hanging; proven for CI runners by the K3 pipeline (same runner class, 47+ production runs).
- [7200 s action budget exceeded by model load + JIT + full case list] → smoke scale is small; large grids need case selection now and multi-gang partitioning later (non-goal).
- [Two members on one machine] → full per-host GPU count request plus broker same-host rejection.
- [Broker is a submit-host SPOF] → heartbeat and deadlines bound every failure mode; rendezvous diagnostics are archived for post-mortem.
- [Legacy multi-node SSH flow regresses] → the `local_server_runner.py` refactor is additive extraction guarded by the existing multi-node tests.
- [Nondeterminism across members] → runfiles deliver one frozen plan; the broker cross-checks config fingerprints; the gang executes the same case list the local path would.

## Migration Plan

1. Broker module with CPU unit tests (no Bazel, no network beyond loopback).
2. Gang package generation, `gpus_per_host` registry, machine derivation, config validation, dry-run.
3. Gang worker (decode path) plus `local_server_runner.py` extraction.
4. Prefill external plan builder and session handle.
5. Entry wiring: capability matrix, gang-aware AOne executor stages, manifest/report integration.
6. CI YAML in the internal repo, then real decode TP2 smoke, then real prefill TP2 smoke.
7. Documentation (`docs/multi_node_aone.md`, links from `aone_prefill.md` and `decode.md`).

Rollback: gang mode is selected only by `gpus_per_machine < world_size` configs; every existing config derives `machine_count == 1` and keeps today's code path, so reverting the change removes gang mode without touching single-machine behavior.
