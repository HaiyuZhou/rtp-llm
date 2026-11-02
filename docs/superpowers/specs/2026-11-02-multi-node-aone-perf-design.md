# Multi-Node AOne Performance Test Design

Date: 2026-11-02
Status: Draft awaiting approval
Proposed OpenSpec change: `add-multi-node-aone-perf`

## Intent

One prefill or decode performance test task must be able to occupy N AOne CI
GPU machines simultaneously and run one distributed model (TP/DP/PP spanning
machines) against the unified perf-test plan, result, and report contracts.

Why: topologies beyond one machine (for example TP16 on two 8-card hosts)
cannot run under the existing AOne paths — AOne prefill today shards
independent single-machine cases, and multi-node decode only works through the
manual SSH flow over a static IP list. The AOne CI mechanism for "one task,
multiple machines" is already proven in production by the K3 PD two-host smoke
pipeline (`Bazel shard_count + remote TestRunner + HTTP rendezvous broker`),
and the perf side already owns the multi-node execution core
(`multi_node/decode_runtime.py`) and the CI control-plane shape
(`cache-grid-perf.yaml`). This change wires those together behind the unified
entry `tools/perf_test`.

## Goals

- `./tools/perf_test --mode=prefill|decode --action=run --config=<config>`
  (with `execution.backend: "aone"` in the config) runs a
  multi-node task: N AOne machines form one gang, each machine hosting
  `local_world_size` ranks of one distributed model; the master machine
  executes the unified perf plan; results return to the submit host and
  produce the same versioned result envelope and reports as local runs.
- The same entry is consumed by a new `.aoneci/multi-node-perf.yaml` pipeline
  whose job runs the entry inside the AOne CI runner (phase A submit host).
- Gang formation happens at runtime through a broker rendezvous generalizing
  the K3 PD pattern to N symmetric members.
- Acceptance uses small-model real AOne smokes: decode TP2 across 2 machines
  and prefill TP2 across 2 machines (Qwen2.5-0.5B-class model).

## Non-Goals

- Phase B (local dev-machine submission) is not validated by this change. The
  entry code is phase-B ready, but worker-to-dev-machine reachability stays
  unverified; the CI runner is the only supported submit host for now.
- Single-machine AOne decode stays rejected. The existing single-machine AOne
  prefill case-sharding path is untouched (`machine_count == 1` behavior is
  exactly today's).
- No auto-retry of failed gangs, no queue or pool management beyond Bazel
  `--jobs`, no model-weight distribution (checkpoints must be visible to the
  AOne runtime, as today).
- The pending `add-cache-perf-local-aone-runners` OpenSpec change is a
  separate effort and is not consumed here.
- Splitting one large grid across several parallel gangs (case partitioning
  over gangs) is future work; one gang runs its whole planned case list.

## Anti-Goals

- No SSH from workers, no credential values in configs or runfiles, no shell
  expansion in delivered configuration; all cross-machine coordination goes
  through the broker's HTTP API.
- No second execution semantics: gang mode reuses the frozen execution plan,
  the decode execution core, the prefill runner contract, and the result
  envelope. A multi-node run must remain comparable to a local run of the
  same config.
- No silent partial success: any member failure or deadline miss fails the
  gang with a nonzero exit class; partial artifacts remain for diagnostics.

## Constraints (platform facts, already verified)

- AOne caps one remote action at 7200 s; the generated gang target keeps a
  7000 s Bazel test timeout, matching the existing AOne perf paths.
- One Bazel test action runs on exactly one remote machine. A gang is one
  `py_test` with `shard_count = machine_count`, each shard action requesting
  the registered per-host GPU count of the selected GPU type through
  `exec_properties = {gpu: <type>, gpu_count: <gpus_per_host>}`. GPU
  accounting then claims the whole machine, so two gang shards cannot land
  on one machine; the broker additionally rejects same-host registration.
  The worker restricts itself to `local_world_size` devices via
  `CUDA_VISIBLE_DEVICES` (see GPU Exclusivity below).
- Broker reachability from remote workers to the submit host and
  worker-to-worker connectivity are both proven by the K3 PD pipeline, whose
  broker runs on an AOne CI runner of the same class.
- `env_inherit` on the generated target passes `*_BROKER_URL`, `*_BROKER_TOKEN`,
  and `*_RUN_ID` from the submit host into every remote shard action; the K3
  PD target uses exactly this mechanism in production (`--test_env` on the
  Bazel invocation is the in-repo alternative and is not used).
- The K3 PD reference lives in the internal repo's git history (commits
  `e378bfbff6..8686c07d0f`: `.aoneci/k3-two-host-smoke.yaml`,
  `internal_source/ci/kimi_k3_pd/`). The `github-opensource` implementation is
  clean-room: the K3 code is a mechanism reference only, never copied, keeping
  the open-source/internal source boundary intact.
- Hub model fetch from remote AOne workers is proven: the 2026-10-29 AOne
  prefill smokes ran `Qwen/Qwen2.5-0.5B-Instruct` (hub repo id) on
  `H20_CU13`/`daily`, and the smoke evidence directories retain those runs.
- Model checkpoint and tokenizer must be resolvable inside the AOne runtime
  container (shared mount path or hub repo id); the existing
  `resolve_remote_runtime_inputs` fuser/hub resolution is reused.
- Bazel 6.4 remote execution with the generated package inside the workspace
  (existing constraint from the AOne prefill path).

## Approaches Considered

1. **AOne CI pipeline wrapping the unified entry (picked).** The
   `.aoneci/multi-node-perf.yaml` job runs `./tools/perf_test
   --mode=<mode> --action=run --config=<config>` (with
   `execution.backend: "aone"` in the delivered config) inside the CI
   runner; the entry process starts the broker, generates the
   gang package, drives remote build/test, and collects. Every network link
   is already proven (K3). Phase B later runs the identical command from a
   dev machine once reachability is validated.
2. **Local-submit first (phase B first).** Rejected for now: worker-to-dev
   return connectivity is unverified, and an unstable local submit host would
   kill whole gangs.
3. **Low-level CI command chaining (cache-grid-perf style).** Rejected: it
   fragments the CLI surface, duplicates orchestration the entry already owns,
   and makes phase B a second implementation.
4. **Extend the manual SSH flow with AOne-allocated IPs.** Rejected: the AOne
   TestRunner resource model does not hand out SSH machine lists; this would
   require a separate resource-leasing system and drift away from AOne CI.

## Architecture

```text
./tools/perf_test --mode=decode --action=run --config=c.json
    (config carries execution.backend: "aone";
     submit host = AOne CI runner; broker runs inside this process)
        |
        v
capability + config validation (gang derivation, cross checks)
        |
        v
run directory + frozen execution plan (existing contracts)
        |
        v
start broker  [stage: rendezvous]
        |
        v
generate gang Bazel package (shard_count = machine_count;
    runfiles: resolved_config.json + gang_plan.json;
    env_inherit: broker URL/token/run id)   [stage: build]
        |
        v
bazel build (remote; build parallelism)   [stage: build]
        |
        v
bazel test //<generated>:gang --strategy=TestRunner=remote
    --jobs >= machine_count            [stage: submit]
        |
        |  each shard action on its machine: gang worker
        |    1. register (shard index, hostname, port) with the broker
        |    2. barrier: fetch full member list, build GANG_CONFIG_STRING
        |    3. start local_world_size ranks of the distributed server
        |    4. shard 0 (master): wait for gang readiness, execute the
        |       unified plan (decode: execute_master_decode; prefill: new
        |       external prefill execution); other shards: wait for master
        |       done, then exit
        |    5. report result/cleanup to the broker; write outputs to the
        |       Bazel undeclared test outputs
        |
        v
collect master run/ artifacts from bazel-testlogs   [stage: collect]
        |
        v
unified result envelope + report (backend=aone)   [stage: report]
        |
        v
stop broker; CI uploads artifacts
```

Stage states, failures, and exit-code classes follow the existing AOne
executor pattern (`execution/aone.py`): stage failures are persisted to the
manifest, the original test failure code wins over collect/report errors, and
partial collection always runs.

## Configuration Contract

Gang mode extends the existing `execution` object:

```json
{
  "execution": {
    "backend": "aone",
    "resource": "daily",
    "gpu_type": "H20_CU13",
    "gpus_per_machine": 8
  }
}
```

- `gpus_per_machine` is optional; its default equals `world_size`
  (`tp * dp * pp`), which keeps `machine_count == 1` and today's single-machine
  AOne prefill behavior exactly as-is. Explicit `machine_count` is a reserved
  field and is rejected: the machine count is always derived as
  `ceil(world_size / gpus_per_machine)`.
- In gang mode (`machine_count > 1`) the `shard_count` field is rejected —
  case-shard semantics and gang semantics must not be expressible at once.
  In single-machine mode `shard_count` stays required, exactly as today.
  `build_parallelism` is not a config field; build concurrency keeps today's
  `PERF_TEST_BAZEL_BUILD_JOBS` env override.
- `gpus_per_machine` that derives `machine_count == 1` is a no-op selecting
  the existing single-machine path (not an error).
- Validation: `world_size` must divide evenly across machines
  (`world_size % machine_count == 0`), `local_world_size =
  world_size / machine_count >= 1`, `max_parallel` must be at least
  `machine_count` (rejected, not clamped), and the GPU type must be a
  registered AOne pool type with a known per-host GPU count. Unknown values
  fail during config validation, before the run directory or any remote side
  effect.
- `machine_count > 1` selects gang mode:
  - decode + aone becomes allowed (today rejected) — only in gang mode;
    single-machine decode + aone stays rejected.
  - prefill + aone in gang mode executes the prefill plan against the
    externally managed distributed server; single-machine prefill + aone
    keeps the existing case-sharded shards.
- `max_parallel` still maps only to the test-phase Bazel `--jobs`; gang mode
  additionally requires `--jobs >= machine_count` so all members run
  simultaneously (validated and rejected, not clamped).

## GPU Exclusivity and Capability Layering

**GPU exclusivity.** A shard action requesting only `local_world_size` GPUs
on a larger host does not reserve the machine, so the TP2 smoke
(`local_world_size = 1` on 8-card hosts) could co-land two gang shards on one
host and fail on broker same-host rejection by construction. The generated
gang rule therefore always requests the registered per-host GPU count of the
selected GPU type (the registry gains a `gpus_per_host` entry; K3 shards use
full-host `gpu_count = 8`), and the worker restricts itself to
`local_world_size` devices via `CUDA_VISIBLE_DEVICES`. Machine exclusivity is
guaranteed by resource accounting; broker same-host validation is the
backstop.

**Capability layering.** `validate_capability` stays config-blind: the
matrix unconditionally allows `decode + aone + run` (run only — decode aone
`resume`/`profile`/`report` stay rejected, matching decode's local posture).
The single-machine decode + aone rejection moves from the capability error to
plan validation, which owns topology-derived rules and already rejects
cross-field conflicts; its message names `gpus_per_machine`/`machine_count`.
This changes one public error surface (the "unsupported capability" message
for decode+aone becomes a planning error) and requires a delta to the
`decode-performance` scenario "Unsupported AOne decode" in the OpenSpec
change. `prefill + aone + report` (offline) is unaffected.

## Broker (Rendezvous) Contract

A new in-process HTTP rendezvous generalizes the K3 PD broker from two named
roles to N symmetric members:

- Bound on the submit host with a random bearer token (private file) and run
  id `multi-perf-<hex>`; serves only bounded run metadata — no commands, no
  source files — mirroring the K3 broker safety rules.
- Operations: `register` (shard index, hostname, port), `members` (blocks until all
  `machine_count` shards registered or the register deadline), `heartbeat`,
  `result`, `cleanup`. HMAC-compared token auth on every request. Workers do
  not self-discover their routable IP: the broker records each registration's
  HTTP source address and returns it in the member list (hostname is
  cross-checked, mismatch rejected), which also feeds `GANG_CONFIG_STRING`.
- Validations: one registration per shard, distinct hosts across shards
  (reject same ip/hostname), consistent config fingerprint across members,
  heartbeat expiry (120 s), and two deadlines: a register deadline (default
  3600 s from broker start, so a placement-starved gang fails fast instead of
  hanging the CI job) and a run deadline (default 43200 s safety net, bounded
  by the CI job timeout). Bazel `--jobs` is an upper bound, not a reservation:
  concurrent placement relies on pool capacity, the pipeline's concurrency
  group, and the register deadline. `machine_count > 2` is mechanically
  supported; acceptance covers N = 2.
- Rendezvous state and events persist under the run directory and are
  archived into `bazel-testlogs` next to the master outputs, following the
  K3 controller-diagnostics pattern.

## Gang Worker Contract

One `py_test` process per machine. Inputs are runfiles: `resolved_config.json`
and `gang_plan.json` (world size, machine count, local world size, topology,
master shard index, port range, fingerprints). The worker:

1. Resolves model references through the existing
   `resolve_remote_runtime_inputs`.
2. Registers with the broker and waits for the full member list; builds
   `GANG_CONFIG_STRING` with members ordered by shard index (master first),
   and sets `WORLD_RANK = shard_index * local_world_size`.
3. Starts the local rank group through a reusable entry extracted from
   `local_server_runner.py` (the legacy SSH scripts keep their behavior and
   call the same functions).
4. Master shard: waits for gang readiness, then executes the unified plan —
   decode reuses `build_external_decode_plan` + `execute_master_decode`;
   prefill uses the new external prefill path (below). Non-master shards wait
   for master completion, then exit.
5. Writes `run/` artifacts to the Bazel undeclared test outputs and reports
   terminal state to the broker.

A member failure, heartbeat expiry, or deadline miss marks the run failed on
the broker; every member exits nonzero; the master's partial `run/` artifacts
are still collected for diagnostics.

## Prefill Multi-Node (new)

- A prefill external plan builder, the counterpart of
  `build_external_decode_plan`: validates configured topology against the
  gang topology, freezes the prefill plan, and reuses existing prefill case
  planning.
- An external server session handle for the prefill runner: exposes the same
  `.client` contract as `MagaEngineHandle` (an `HttpPrefillClient` bound to
  the gang server's frontend on the master machine) plus diagnostics, without
  owning any process lifecycle. The existing `HttpPrefillClient` needs only
  an explicit host:port endpoint instead of the hardcoded loopback.
- The prefill runner, metrics, result envelope, and report path are reused
  unchanged; only the session/transport origin differs.

## CI Pipeline (Phase A)

`.aoneci/multi-node-perf.yaml` (internal repo): one job per architecture
variant (mirroring cache-grid-perf's job anchors), parameters for model
identity, topology, `gpus_per_machine`, GPU type, resource, and the delivered
perf config. Steps: checkout, submodule pinning via the existing
`create_symlinks.sh` pattern, then `./tools/perf_test --mode=... --action=run
--config=...` (the delivered config sets `execution.backend: "aone"`)
inside the runner (the entry starts the broker
and drives remote build/test/collect), then `upload-artifact` of the run
directory and broker diagnostics. A concurrency group keeps overlapping gangs
from double-booking pool machines.

## Results, Reports, Verification

- The run produces the standard result envelope v2 (`backend=aone`, mode
  metrics unchanged) and standard reports. The manifest additionally records
  gang topology (machine count, local world size, GPU type), broker run id,
  per-shard hosts (sanitized), and stage states.
- Artifact policy: the master shard's full `run/` tree (envelope, reports,
  traces, diagnostics) is collected from `bazel-testlogs`, giving parity with
  a local run's artifacts. Non-master shards contribute logs only (server
  logs and broker member reports via their undeclared test outputs).
- Verification ladder:
  1. CPU unit tests: broker state machine (register/barrier/distinct-host/
     deadline/heartbeat/token), gang package generation (shard count, exec
     properties, runfiles, env_inherit), machine derivation and validation,
     worker plan/exit contract, prefill external plan and session.
  2. `--dry-run`: prints normalized config, gang topology, derived machine
     count, member layout, generated target, and final build/test commands —
     no broker, no Bazel, no run directory, no remote job.
  3. Real AOne smokes: decode TP2 across 2 machines and prefill TP2 across 2
     machines with `Qwen/Qwen2.5-0.5B-Instruct` (hub repo id, proven by the
     2026-10-29 AOne prefill runs) on `H20_CU13`/`daily` — one 1-card member
     per host for TP2, full-host `gpu_count` requests on both members; formal
     report generated, artifacts collected, nonzero-exit behavior exercised
     by an intentionally failing member case in a separate diagnostic run.
  4. Existing AOne prefill and local decode/prefill test targets keep
     passing (no regression in the single-machine paths).

## Risks and Mitigations

- **Worker-to-submit-host reachability**: proven for CI runners by the K3
  pipeline (same runner class). If pool isolation ever breaks it, gang
  registration fails fast at the deadline instead of hanging silently.
- **7200 s action budget**: a gang must fit model load + JIT + the whole
  planned case list. Smoke scale is small; large grids need case selection
  now and multi-gang partitioning later (documented non-goal).
- **Two shards on one machine**: prevented by GPU accounting (each gang rule
  requests the full registered per-host GPU count) and rejected by broker
  same-host validation.
- **Broker is a submit-host SPOF**: heartbeat and deadline bound every
  failure mode; the K3 pattern has run in production 47+ times.
- **Legacy multi-node SSH flow must keep working**: the
  `local_server_runner.py` refactor is additive extraction; the scripts and
  their behavior stay compatible, guarded by the existing multi-node tests.
- **Determinism across shards**: runfiles deliver one frozen plan; the broker
  cross-checks config fingerprints; the gang executes the same case list the
  local path would.

## Migration Order

1. Broker module + unit tests (pure CPU, no Bazel).
2. Gang package generation (including the `gpus_per_host` registry entry per
   registered GPU type), machine derivation, config validation, dry-run.
3. Gang worker (decode path) + `local_server_runner.py` extraction.
4. Prefill external plan builder and session handle.
5. Entry wiring: capability matrix, gang-aware AOne executor stages,
   manifest/report integration.
6. CI YAML + real decode smoke, then real prefill smoke.
7. Documentation (`docs/multi_node_aone.md`, links from `aone_prefill.md` and
   `decode.md`).
