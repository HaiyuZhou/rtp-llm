# Tasks

## 1. Gang Config Derivation And Capability Boundary

- [x] 1.1 Add the optional `execution.gpus_per_machine` field with `world_size` default and derive `machine_count = ceil(world_size / gpus_per_machine)`; verify config tests confirm omission derives machine count one and keeps the existing single-machine AOne path, and `machine_count == 1` derivation from an explicit `gpus_per_machine` is a no-op, not an error.
- [x] 1.2 Reject conflicting gang semantics during config validation; verify tests fail on explicit `execution.machine_count`, on `shard_count` when the derived machine count exceeds one, on non-divisible `world_size`, and on `max_parallel` below the machine count, all before run-directory or remote side effects, while single-machine `shard_count` stays required exactly as today.
- [x] 1.3 Allow `decode + aone + run` in the capability matrix and keep decode aone `resume`/`profile`/`report` rejected; verify capability tests pass decode aone run before any side effect and still reject the follow-up actions.
- [x] 1.4 Move the single-machine decode+aone rejection to plan validation; verify planning tests reject decode aone with derived machine count one through a message naming `gpus_per_machine`/`machine_count` while gang-mode decode passes planning, and prefill aone passes in both modes.

## 2. Rendezvous Broker

- [x] 2.1 Implement the in-process HTTP broker core (register, members, heartbeat, result, cleanup) with bearer-token auth on every request; verify unit tests reject invalid tokens without disclosing run metadata and exercise each operation on loopback.
- [x] 2.2 Implement member identity rules; verify unit tests reject duplicate shard registration, same-host registrations by address or hostname, hostname/source-address mismatch, and config-fingerprint mismatch, and confirm the recorded source address feeds the member list.
- [x] 2.3 Implement deadline and heartbeat enforcement; verify unit tests fail the gang at the 3600 s register deadline when registration is incomplete, mark members lost after the 120 s heartbeat expiry, and fail the run at the 43200 s run-deadline safety net.
- [x] 2.4 Persist rendezvous state and events under the run directory; verify unit tests write bounded, redacted diagnostics (no token values) that archive alongside collected outputs.

## 3. Gang Package Generation

- [x] 3.1 Add `gpus_per_host` to the GPU-type registry; verify unit tests cover registered types and reject unknown GPU types with the per-host count requirement during config validation.
- [x] 3.2 Generate the gang Bazel package: one `py_test` with `shard_count = machine_count`, `exec_properties` requesting the registered per-host GPU count, `resolved_config.json` and `gang_plan.json` runfiles, and `env_inherit` for broker URL, token, and run id; verify package-generation tests assert target shape, runfile delivery, and env passing without submit-host path dependencies.
- [x] 3.3 Derive the gang plan artifact (world size, machine count, local world size, topology, master shard index, port range, fingerprints); verify unit tests round-trip the plan and freeze the same fingerprints as the execution plan.

## 4. Gang Worker Decode Path

- [x] 4.1 Extract the reusable local rank-group startup from `local_server_runner.py`; verify the legacy SSH scripts keep their behavior through the existing multi-node tests and the extraction exposes the same functions for gang workers.
- [x] 4.2 Implement the gang worker: runfile loading, `resolve_remote_runtime_inputs` model resolution, broker registration and member barrier, `GANG_CONFIG_STRING` construction ordered by shard index with `WORLD_RANK = shard_index * local_world_size`, and `CUDA_VISIBLE_DEVICES` restriction to `local_world_size` devices; verify unit tests with a fake broker and fake server runner cover registration, barrier, and rank layout.
- [x] 4.3 Wire the master member to `build_external_decode_plan` + `execute_master_decode` and non-master members to wait for master completion; verify unit tests execute the frozen decode plan on the master only and propagate a member failure to nonzero exits for every member while preserving the master's partial artifacts.
- [x] 4.4 Write gang worker outputs to Bazel undeclared test outputs and report terminal state to the broker; verify unit tests cover success, member failure, and deadline-miss exits.

## 5. Prefill External Path

- [x] 5.1 Add an explicit host:port endpoint to `HttpPrefillClient`; verify unit tests cover endpoint binding and request behavior while the existing loopback default keeps local prefill tests passing unchanged.
- [x] 5.2 Implement the prefill external plan builder validating configured topology against the gang topology and freezing the prefill plan; verify unit tests accept matching topology, reject mismatches before execution, and reuse existing prefill case planning.
- [x] 5.3 Implement the external server session handle exposing the local engine handle's `.client` contract plus diagnostics without owning process lifecycle; verify unit tests drive the prefill runner against a fake endpoint and produce the standard envelope fields.

## 6. Entry Wiring And Gang Executor

- [x] 6.1 Add the gang-aware AOne executor stages (rendezvous, build, submit, collect, report) with in-process broker lifecycle; verify executor tests with mocked Bazel and broker assert stage transitions persist to the manifest, the original test failure code wins over collect/report errors, and partial collection always runs.
- [x] 6.2 Collect the master member's full `run/` tree from `bazel-testlogs` and non-master logs only; verify collection tests produce local-parity envelope, reports, and traces with backend recorded as AOne and gang topology, broker run id, and sanitized per-shard hosts in the manifest.
- [x] 6.3 Extend dry-run to gang mode; verify dry-run tests print normalized config, derived machine count, member layout, generated target, and final build/test commands with no broker, no Bazel subprocess, no run directory, and no remote submission.
- [x] 6.4 Add gang-mode example configs (decode and prefill, small model); verify examples tests parse and dry-run them with the expected member layout, and run the full perf-test unit test suite for single-machine regression (`--config=cuda13`).

## 7. CI Pipeline And Real Smokes

- [x] 7.1 Add `.aoneci/multi-node-perf.yaml` in the internal repo with the concurrency group, parameters, checkout, submodule pinning, unified-entry invocation, and artifact upload; verify the pipeline registers and a first CI run reaches the build stage with a deliberate small config.
- [x] 7.2 Run the real decode TP2 gang smoke on `H20_CU13`/`daily` with `Qwen/Qwen2.5-0.5B-Instruct` across 2 machines; verify the collected run directory contains a formal report, local-parity envelope, gang topology in the manifest, and distinct member hosts.
- [x] 7.3 Run the real prefill TP2 gang smoke with the same resources; verify observed-reuse correctness conditions hold, the standard prefill envelope and report are produced, and single-machine AOne prefill behavior is unchanged.
- [x] 7.4 Exercise nonzero-exit behavior with an intentionally failing member in a separate diagnostic run; verify every member exits nonzero, the failure is visible in broker diagnostics and the manifest, and the master's partial artifacts are still collected.

## 8. Documentation

- [x] 8.1 Write `docs/multi_node_aone.md` covering the gang config contract, derivation rules, broker behavior, CI pipeline usage, and artifact layout, with links from `aone_prefill.md` and `decode.md`; verify documentation tests reference only delivered commands and artifacts.
