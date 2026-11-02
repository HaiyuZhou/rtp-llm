# Multi-Node AOne Performance Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One AOne perf-test task occupies N machines as one gang running one distributed model (decode and prefill), driven by the unified `tools/perf_test` entry with local-parity artifacts.

**Architecture:** Gang mode is selected by deriving `machine_count = ceil(world_size / gpus_per_machine) > 1`. The entry process on the submit host starts an in-process HTTP rendezvous broker, generates a gang Bazel package (one `py_test`, `shard_count = machine_count`, full per-host GPU requests), drives remote build/test, and collects the master member's `run/` tree from `bazel-testlogs`. Members register with the broker, build `GANG_CONFIG_STRING`, start local ranks, and the master executes the frozen unified plan (decode: existing external core; prefill: new external plan builder + session handle). Single-machine paths are untouched.

**Tech Stack:** Python 3, stdlib `http.server`/`threading` (broker), `requests`, Bazel/rules_python, unittest with fakes; AOne CI (internal repo `.aoneci/` YAML).

**Spec:** `openspec/changes/add-multi-node-aone-perf/` (proposal, specs deltas: `multi-node-aone-orchestration`, `perf-test-orchestration`, `decode-performance`, `prefill-cache-performance`; design; tasks). Full design rationale: `docs/superpowers/specs/2026-11-02-multi-node-aone-perf-design.md`.

## Global Constraints

- Clean-room: the K3 PD broker (internal repo git history `e378bfbff6..8686c07d0f`) is a mechanism reference only — never copy its code into `github-opensource`.
- Environment variable names: `PERF_TEST_BROKER_URL`, `PERF_TEST_BROKER_TOKEN`, `PERF_TEST_BROKER_RUN_ID`. Broker token never appears in persisted diagnostics, manifests, or logs.
- Broker constants: register deadline 3600 s, run deadline 43200 s, heartbeat expiry 120 s. Gang test timeout stays 7000 s (AOne action cap 7200 s).
- `gpus_per_machine` defaults to `world_size` (`tp * dp * pp`); `machine_count` is always derived, never a config field; gang mode rejects `shard_count`; `max_parallel < machine_count` is rejected, not clamped; build parallelism stays env-only (`PERF_TEST_BAZEL_BUILD_JOBS`).
- Gang shard rules always request the registered per-host GPU count (`AONE_GPU_HOSTS`); workers restrict to `local_world_size` devices via `CUDA_VISIBLE_DEVICES`.
- Capability matrix allows only `decode + aone + run` additionally; decode aone resume/profile/report stay rejected.
- Single-machine behavior (`machine_count == 1`) must remain byte-for-byte unchanged: existing `config_schema`, `plan`, `aone package`, executor, and dry-run tests must pass without modification (only additions allowed).
- Bazel tests run with `--config=cuda13` and output base `/home/zhouhaiyu.zhy/.cache/bazel_cuda13_cache_grid_main_local_cache`; unset proxy env vars (`http_proxy` etc.) first.
- Repo conventions: docs and code comments in English; one commit per milestone; nothing pushed without an explicit user request; new test files must be registered in `rtp_llm/test/perf_test/BUILD`.
- Legacy multi-node SSH flow (`multi_runner.sh`, `multi_benchmark.py`) must keep working — extraction from `local_server_runner.py` is additive only.

## Review Focus

1. **Non-positive or oversized `gpus_per_machine`**: `gpus_per_machine <= 0` must be rejected at parse; `gpus_per_machine > world_size` derives `machine_count == 1` (ceil) and is a no-op, not an error. → Test in Task 1.1.
2. **Late or partial registration**: a `members` call that blocks must unblock and fail the gang when the register deadline passes; a registration arriving after the gang is complete must be rejected, not silently merged. → Tests in Task 2.3.
3. **Token leakage in diagnostics**: persisted rendezvous state and events must never contain the bearer token; a test greps the serialized state. → Test in Task 2.4.
4. **Missing broker env in a worker**: if `PERF_TEST_BROKER_URL`/`_TOKEN`/`_RUN_ID` are absent (env_inherit failure), the worker must fail immediately with a clear error, not block until deadlines. → Test in Task 4.2.
5. **Unknown GPU type without per-host count**: a gang config naming a GPU type absent from `AONE_GPU_HOSTS` must fail during config validation with the per-host-count requirement, before the run directory exists. → Test in Task 3.1.

---

### Task 1: Gang Config Derivation And Capability Boundary

**Files:**
- Create: `rtp_llm/test/perf_test/aone/gang.py`
- Create: `rtp_llm/test/perf_test/gang_config_test.py`
- Modify: `rtp_llm/test/perf_test/config_schema.py` (`AOneExecutionConfig`, `_parse_execution` aone branch)
- Modify: `rtp_llm/test/perf_test/plan.py` (`AOneResourcePlan`, gang derivation, plan-level rejections)
- Modify: `rtp_llm/test/perf_test/capabilities.py` (`_SUPPORTED`)
- Modify: `rtp_llm/test/perf_test/capabilities_test.py`
- Modify: `rtp_llm/test/perf_test/BUILD` (register `gang_config_test`)

**Interfaces:**
- Produces: `derive_gang_topology(world_size: int, gpus_per_machine: int) -> GangTopology` raising `PlanningError`; `GangTopology` frozen dataclass with `world_size: int`, `gpus_per_machine: int`, `machine_count: int`, `local_world_size: int`.
- Produces: `AOneExecutionConfig.gpus_per_machine: Optional[int]` (None = default).
- Produces: `AOneResourcePlan` gains `gpus_per_machine: int`, `machine_count: int`, `local_world_size: int` (defaulted to keep existing constructions valid).
- Consumes: existing `parse_config`, `build_execution_plan`, `validate_capability(mode, action, backend) -> Capability`.

- [ ] **1.1 Gang topology derivation with parse-level and plan-level validation**

Write failing tests in `gang_config_test.py` (registered in BUILD):

```python
def test_default_derives_single_machine(self):
    # aone config without gpus_per_machine -> machine_count == 1, existing path

def test_gang_derivation(self):
    # world_size 16, gpus_per_machine 8 -> machine_count 2, local_world_size 8

def test_explicit_machine_count_rejected(self):
    # execution.machine_count present -> ConfigurationError naming the reserved field

def test_invalid_gpus_per_machine_rejected(self):
    # 0, -1, non-int -> ConfigurationError at parse

def test_oversized_gpus_per_machine_is_noop(self):
    # world_size 4, gpus_per_machine 8 -> machine_count == 1 (ceil), no error

def test_non_divisible_world_size_rejected(self):
    # world_size 10, gpus_per_machine 4 -> PlanningError naming derived machine_count and world_size

def test_shard_count_rejected_in_gang_mode(self):
    # gang config also carrying shard_count -> PlanningError (mutually exclusive)

def test_shard_count_still_required_single_machine(self):
    # aone config without gpus_per_machine and without shard_count -> still fails as today
```

Implement: parse `gpus_per_machine` as optional positive int in `_parse_execution` (add to the allowed-field set); reject `machine_count` via `_reject_unknown`. In `plan.py`, after `world_size` is known, call `derive_gang_topology` and enforce divisibility, gang-mode `shard_count` rejection, and `max_parallel >= machine_count`; store gang fields on `AOneResourcePlan`.

Run: `bazel test --config=cuda13 //rtp_llm/test/perf_test:gang_config_test` — expect FAIL first (no module), then PASS.

- [ ] **1.2 Capability matrix allows decode aone run**

Extend `capabilities_test.py`: `decode + aone + run` returns a `Capability` (no `offline_only`); `decode + aone + resume/profile/report` still raise `CapabilityError`. Update `_SUPPORTED` to add `("decode", "aone", "run")`.

Run: `bazel test --config=cuda13 //rtp_llm/test/perf_test:capabilities_test` — PASS.

- [ ] **1.3 Single-machine decode+aone rejection moves to planning**

Extend `gang_config_test.py`: decode aone config deriving `machine_count == 1` fails in plan building with a `PlanningError` naming `gpus_per_machine`/`machine_count`; decode aone gang config passes planning; prefill aone passes in both modes. Implement the check where `AOneResourcePlan` is built for decode.

Run: `bazel test --config=cuda13 //rtp_llm/test/perf_test:gang_config_test //rtp_llm/test/perf_test:capabilities_test` plus the pre-existing config/plan/entry suites (`config_resolution_test`, `entry_test`, `dry_run_test`) — all PASS with no modification to existing test files.

- [ ] **1.4 Commit milestone 1**

```bash
git add rtp_llm/test/perf_test/aone/gang.py rtp_llm/test/perf_test/gang_config_test.py \
        rtp_llm/test/perf_test/config_schema.py rtp_llm/test/perf_test/plan.py \
        rtp_llm/test/perf_test/capabilities.py rtp_llm/test/perf_test/capabilities_test.py \
        rtp_llm/test/perf_test/BUILD
git commit -m "feat(perf): derive gang topology from aone execution config"
```

---

### Task 2: Rendezvous Broker

**Files:**
- Create: `rtp_llm/test/perf_test/aone/broker.py`
- Create: `rtp_llm/test/perf_test/aone_broker_test.py`
- Modify: `rtp_llm/test/perf_test/BUILD`

**Interfaces:**
- Produces: `GangBroker` (context manager): `start() -> str` (returns URL), `stop()`, properties `url: str`, `run_id: str`, `token: str`, `members: Tuple[BrokerMember, ...]`; `BrokerMember` frozen dataclass `shard_index: int, address: str, hostname: str, port: int`.
- Produces: `run_gang_broker(machine_count: int, *, state_dir: Path, host: str = "0.0.0.0", register_deadline_s: float = 3600.0, run_deadline_s: float = 43200.0, heartbeat_expiry_s: float = 120.0) -> GangBroker`.
- Produces: constants `REGISTER_DEADLINE_S`, `RUN_DEADLINE_S`, `HEARTBEAT_EXPIRY_S`.
- HTTP API (loopback tests use `requests` or `urllib` against the live socket): `POST /v1/runs/<id>/register` `{shard_index, hostname, port}`; `GET /v1/runs/<id>/members` (blocks until complete or register deadline); `POST /v1/runs/<id>/heartbeat` `{shard_index}`; `POST /v1/runs/<id>/result` `{shard_index, state, detail}`; `POST /v1/runs/<id>/cleanup`. All authenticated by `Authorization: Bearer <token>`.

- [ ] **2.1 Broker core with token auth**

Failing tests (`aone_broker_test.py`): start broker on ephemeral port; register one member for `machine_count=1`; `members` returns it with the request's source address recorded; request without/with wrong token gets 401 and discloses no run metadata; second `register` for the same shard index is rejected (409). Implement with stdlib `http.server` + `threading` (daemon thread, `allow_reuse_address`); token from `secrets.token_urlsafe`.

Run: `bazel test --config=cuda13 //rtp_llm/test/perf_test:aone_broker_test` — FAIL then PASS.

- [ ] **2.2 Member identity rules**

Failing tests: two registrations from the same address (loopback: distinguish via explicit `X-Forwarded-For`-style test seam or distinct loopback addresses `127.0.0.1`/`127.0.0.2`) rejected; hostname/source-address mismatch rejected (register claims a hostname whose reverse lookup disagrees — inject a resolver seam); config-fingerprint field in `register` payload mismatching the broker's expected fingerprint rejected. Add `expected_config_fingerprint` parameter to `run_gang_broker`.

Run: same target — PASS.

- [ ] **2.3 Deadlines and heartbeat**

Failing tests (inject a fake clock or tiny deadlines): `members` blocks then fails when the register deadline passes with fewer than `machine_count` registrations; a member whose heartbeat stops is marked lost and the gang state becomes failed; the run deadline fails an incomplete gang; a registration arriving after the gang is complete is rejected. Implement deadline tracking on a monotonic clock with a condition variable for the `members` barrier.

Run: same target — PASS.

- [ ] **2.4 Persisted, redacted rendezvous diagnostics**

Failing tests: after operations, `state_dir` contains a JSON state file and an events log (one line per event) that include shard indices, addresses, hostnames, timestamps, and gang state, and `grep -c token` on serialized content finds no token value. Implement atomic writes (reuse `storage.atomic_write_json` if importable, else `os.replace`).

Run: same target — PASS. Also run the full broker target once more — PASS.

- [ ] **2.5 Commit milestone 2**

```bash
git add rtp_llm/test/perf_test/aone/broker.py rtp_llm/test/perf_test/aone_broker_test.py rtp_llm/test/perf_test/BUILD
git commit -m "feat(perf): add gang rendezvous broker"
```

---

### Task 3: Gang Package Generation

**Files:**
- Create: `rtp_llm/test/perf_test/aone/gang_package.py`
- Create: `rtp_llm/test/perf_test/aone_gang_package_test.py`
- Modify: `rtp_llm/test/perf_test/plan.py` (`AONE_GPU_HOSTS`)
- Modify: `rtp_llm/test/perf_test/defs.bzl` (new `perf_test_gang_shard` rule)
- Modify: `rtp_llm/test/perf_test/BUILD`

**Interfaces:**
- Produces: `AONE_GPU_HOSTS: Dict[str, int]` in `plan.py` (alongside `AONE_GPU_PLATFORMS`; `H20: 8`, `H20_CU13: 8`, `L20D_DEV: 8`, `L20_CU13: 8`, `SM100_ARM_CU13: 8`, `SM103_CU13: 8`).
- Produces: `build_gang_plan_payload(plan: ExecutionPlan, run_id: str) -> Dict[str, Any]` in `gang_package.py` — keys `schema_version`, `run_id`, `config_fingerprint`, `workload_fingerprint`, `world_size`, `machine_count`, `local_world_size`, `gpus_per_machine`, `gpus_per_host`, `gpu_type`, `topology`, `master_shard_index`, `port_range`.
- Produces: `generate_gang_package(root, plan, *, run_id, package=None) -> GeneratedAOnePackage` in `gang_package.py` — writes `resolved_config.json`, `resolved_workload.json`, `gang_plan.json` and a `BUILD` invoking `perf_test_gang_shard`.
- Produces (bzl): `perf_test_gang_shard(name, resolved_config, resolved_workload, gang_plan, gpu_type, gpus_per_host, machine_count)` — `shard_count = machine_count`, `timeout = "eternal"` with test timeout 7000 s, `exec_properties = {"gpu": gpu_type, "gpu_count": str(gpus_per_host)}`, `env_inherit = ["PERF_TEST_BROKER_URL", "PERF_TEST_BROKER_TOKEN", "PERF_TEST_BROKER_RUN_ID"]`, data = config/workload/gang-plan runfiles, no `env WORLD_SIZE` (the worker computes it).
- Consumes: `GeneratedAOnePackage`, `thaw`, `AOneResourcePlan` from `aone/package.py` / `plan.py`.

- [ ] **3.1 GPU per-host registry and config validation**

Failing tests (`aone_gang_package_test.py` + extend `gang_config_test.py`): gang config with a GPU type absent from `AONE_GPU_HOSTS` fails plan validation naming the per-host-count requirement (Review Focus 5); every entry in `AONE_GPU_PLATFORMS` also has a `AONE_GPU_HOSTS` entry. Implement the registry and the plan-level check (gang mode only).

Run: `bazel test --config=cuda13 //rtp_llm/test/perf_test:gang_config_test` — PASS.

- [ ] **3.2 Gang plan payload and package generation**

Failing tests: `build_gang_plan_payload` returns the keys above with topology from the plan and `master_shard_index = 0`; `generate_gang_package` writes the three JSON artifacts plus a `BUILD` whose text contains `perf_test_gang_shard`, `shard_count`, the registered `gpus_per_host`, `env_inherit` with the three broker variables, and no absolute submit-host paths (assert on generated text). Implement in `gang_package.py` reusing `aone/package.py` helpers (import, do not duplicate, its atomic-write and path-sanitization logic).

Run: `bazel test --config=cuda13 //rtp_llm/test/perf_test:aone_gang_package_test` — FAIL then PASS.

- [ ] **3.3 Bazel rule for gang shards**

Add `perf_test_gang_shard` to `defs.bzl` per the interface above. Failing test: extend `aone_gang_package_test.py` to evaluate the generated BUILD through a parse assertion (the repo's package tests assert on generated text; follow `aone_package_test.py`'s pattern), including `timeout` wiring to 7000 s and `tags = ["manual", gpu_type]`.

Run: same target — PASS; plus `//rtp_llm/test/perf_test:aone_package_test` unchanged — PASS.

- [ ] **3.4 Commit milestone 3**

```bash
git add rtp_llm/test/perf_test/aone/gang_package.py rtp_llm/test/perf_test/aone_gang_package_test.py \
        rtp_llm/test/perf_test/plan.py rtp_llm/test/perf_test/defs.bzl rtp_llm/test/perf_test/BUILD
git commit -m "feat(perf): generate gang bazel packages with full-host gpu requests"
```

---

### Task 4: Gang Worker Decode Path

**Files:**
- Create: `rtp_llm/test/perf_test/aone/gang_worker.py`
- Create: `rtp_llm/test/perf_test/aone_gang_worker_test.py`
- Modify: `rtp_llm/test/perf_test/multi_node/local_server_runner.py` (extract reusable functions)
- Modify: `rtp_llm/test/perf_test/multi_node/decode_runtime.py` (only if the extraction moves shared helpers)
- Modify: `rtp_llm/test/perf_test/BUILD`

**Interfaces:**
- Produces: `BrokerClient(url: str, token: str, run_id: str, *, request=requests.request)` with `register(shard_index, hostname, port) -> BrokerMember list`, `members(timeout_s) -> List[BrokerMember]`, `heartbeat(shard_index)`, `report(shard_index, state, detail)`, `cleanup()`.
- Produces: `run_gang_worker(resolved_config_path: Path, gang_plan_path: Path, *, broker: BrokerClient, server_runner: Callable[[GangWorkerContext], ServerHandle], test_outputs_dir: Path, clock=time.monotonic) -> int` — returns process exit code (0 success).
- Produces: `GangWorkerContext` frozen dataclass: `gang_plan: Dict`, `members: Tuple[BrokerMember, ...]`, `shard_index: int`, `world_rank: int`, `local_world_size: int`, `gang_config_string: str`, `cuda_visible_devices: str`.
- Produces (extraction from `local_server_runner.py`): `build_gang_config_string(members, local_world_size) -> str` (master first, ordered by shard index) and `start_local_rank_group(context, config, model_paths) -> ServerHandle` — the legacy script keeps calling them with its current inputs.
- Consumes: `resolve_remote_runtime_inputs`, `build_external_decode_plan`, `execute_master_decode`, broker from Milestone 2.

- [ ] **4.1 Extract reusable rank-group startup**

Refactor `local_server_runner.py`: pull the gang-config construction and per-host rank-group startup into module-level functions with the signatures above; the script's `__main__` flow calls them unchanged. Run the existing multi-node suites: `//rtp_llm/test/perf_test:decode_multi_node_test` — PASS unmodified.

- [ ] **4.2 Gang worker registration, barrier, and rank layout**

Failing tests (`aone_gang_worker_test.py`, fake `BrokerClient` + fake `server_runner`): worker reads the gang plan, computes `world_rank = shard_index * local_world_size`, builds `GANG_CONFIG_STRING` ordered by shard index with the master first, sets `CUDA_VISIBLE_DEVICES` to exactly `local_world_size` device ids, and calls `server_runner` with the context; missing `PERF_TEST_BROKER_URL`/`_TOKEN`/`_RUN_ID` env (Review Focus 4) raises a clear error immediately (no broker call, no blocking). Implement `BrokerClient` and the worker prelude.

Run: `bazel test --config=cuda13 //rtp_llm/test/perf_test:aone_gang_worker_test` — FAIL then PASS.

- [ ] **4.3 Master execution and failure propagation**

Failing tests: with `shard_index == 0` (master), the worker invokes `build_external_decode_plan` + `execute_master_decode` with the external server handle from `server_runner` and writes case results under `test_outputs_dir/run/`; non-master workers wait on `broker.members`/master-result polling and return 0 after the master reports success; when a fake member reports failure (or the fake broker says the gang failed), every worker returns nonzero; when the master fails mid-plan, its partial `run/` artifacts remain on disk. Implement with injectable master-executor seam (`master_executor: Callable[..., int]`, defaulting to the decode core).

Run: same target — PASS.

- [ ] **4.4 Terminal-state reporting and undeclared outputs**

Failing tests: the worker calls `broker.report` exactly once with terminal state (success/failure detail); on heartbeat-loop exit paths the heartbeat thread stops; artifacts land under `test_outputs_dir` so Bazel captures them as undeclared test outputs. Run the full worker target — PASS.

- [ ] **4.5 Commit milestone 4**

```bash
git add rtp_llm/test/perf_test/aone/gang_worker.py rtp_llm/test/perf_test/aone_gang_worker_test.py \
        rtp_llm/test/perf_test/multi_node/local_server_runner.py rtp_llm/test/perf_test/multi_node/decode_runtime.py \
        rtp_llm/test/perf_test/BUILD
git commit -m "feat(perf): run gang workers with master decode execution"
```

---

### Task 5: Prefill External Path

**Files:**
- Create: `rtp_llm/test/perf_test/multi_node/prefill_runtime.py`
- Create: `rtp_llm/test/perf_test/prefill_external_test.py`
- Modify: `rtp_llm/test/perf_test/execution/runtime.py` (`HttpPrefillClient`)
- Modify: `rtp_llm/test/perf_test/BUILD`

**Interfaces:**
- Produces: `HttpPrefillClient(endpoint: str, *, post=requests.post, timeout_seconds=300)` — accepts `http://host:port` (loopback callers pass `http://127.0.0.1:{port}`; existing `port`-only constructor calls migrate at call sites inside `runtime.py` only).
- Produces: `build_external_prefill_plan(config_path, result_directory, actual_topology) -> ExecutionPlan` — mirrors `build_external_decode_plan`: validates configured topology against `actual_topology`, freezes the prefill plan, no run artifacts created.
- Produces: `ExternalPrefillSession(endpoint: str, *, topology: Dict[str, int])` with `.client -> HttpPrefillClient`, `.diagnostics -> EngineDiagnostics`-compatible object, `.close()` no-op — satisfies the prefill runner's session contract without owning process lifecycle.
- Consumes: prefill runner/planner from `modes/prefill.py`, `EngineDiagnostics` shape from `execution/engine_session.py`.

- [ ] **5.1 Explicit endpoint for HttpPrefillClient**

Failing tests (`prefill_external_test.py` + adjust in-file usages): client posts to the provided host:port; a local call site keeps `127.0.0.1`. Existing prefill local suites (`default_executor_test`, `entry_execution_test`) pass unmodified.

- [ ] **5.2 External prefill plan builder**

Failing tests: matching topology freezes a plan with the standard prefill cases and fingerprints; topology mismatch raises `PlanningError` before any filesystem artifact; the builder accepts the same config shapes as `build_external_decode_plan`.

- [ ] **5.3 External prefill session handle**

Failing tests: the prefill runner driven through `ExternalPrefillSession` against a fake HTTP endpoint produces the standard envelope fields (status, case results, reuse validation `invalid_reuse` path included).

Run: `bazel test --config=cuda13 //rtp_llm/test/perf_test:prefill_external_test` — FAIL then PASS.

- [ ] **5.4 Commit milestone 5**

```bash
git add rtp_llm/test/perf_test/multi_node/prefill_runtime.py rtp_llm/test/perf_test/prefill_external_test.py \
        rtp_llm/test/perf_test/execution/runtime.py rtp_llm/test/perf_test/BUILD
git commit -m "feat(perf): external prefill plan and session for gang execution"
```

---

### Task 6: Entry Wiring, Gang Executor, Dry-Run, Examples

**Files:**
- Create: `rtp_llm/test/perf_test/execution/aone_gang.py`
- Create: `rtp_llm/test/perf_test/aone_gang_executor_test.py`
- Modify: `rtp_llm/test/perf_test/execution/aone.py` (dispatch gang vs single-machine)
- Modify: `rtp_llm/test/perf_test/dry_run.py`
- Modify: `rtp_llm/test/perf_test/entry.py` (gang dispatch only if needed by executor seam)
- Create: `rtp_llm/test/perf_test/examples/decode_aone_gang.json`, `prefill_aone_gang.json`
- Modify: `rtp_llm/test/perf_test/BUILD`, `aone_examples_test.py` (or new `gang_examples_test.py`)

**Interfaces:**
- Produces: `GangAOneExecutor` in `execution/aone_gang.py` with `execute(plan: ExecutionPlan) -> AOneExecutionOutcome` — stages `rendezvous` (start broker via `run_gang_broker`, persist URL/run id minus token), `build` (`generate_gang_package` + remote `bazel build`), `submit` (`bazel test --strategy=TestRunner=remote --jobs >= machine_count` with env exported), `collect` (master `run/` tree from testlogs; non-master logs), `report` (standard envelope + reports).
- Consumes: everything from Milestones 1–5; the existing `AOneExecutionOutcome`, `RunDirectoryLock`, `inspect_run_directory`, stage-persistence helpers from `execution/aone.py`.
- Dispatch rule: in `execution/aone.py` (or its factory), route `machine_count > 1` plans to `GangAOneExecutor`, else the existing `AOneExecutor`.

- [ ] **6.1 Gang executor stages with broker lifecycle**

Failing tests (`aone_gang_executor_test.py`, mocked bazel-runner and real loopback broker): stage transitions persist to the manifest in order; broker env vars are exported to the bazel test invocation and the token is not persisted; `--jobs` is at least `machine_count`; a submit-stage failure records the failing stage and still attempts collection; test failure exit code wins over collect/report errors; partial collection always runs.

- [ ] **6.2 Collection parity and manifest gang metadata**

Failing tests: from a fake testlogs tree (master shard `run/` with envelope/report/trace files, non-master shard logs only), collection produces the run directory with all master artifacts and the manifest records `machine_count`, `local_world_size`, `gpu_type`, broker run id, and sanitized per-shard hosts (no token).

- [ ] **6.3 Gang dry-run**

Failing tests (extend `dry_run_test.py` or new file): gang-mode dry-run prints normalized config, derived machine count, member layout, generated target, and final build/test commands; asserts via mocks that no broker starts, no bazel subprocess runs, no run directory is created.

- [ ] **6.4 Examples and full-suite regression**

Add `examples/decode_aone_gang.json` and `examples/prefill_aone_gang.json` (small model `Qwen/Qwen2.5-0.5B-Instruct` hub id, `H20_CU13`/daily, TP2, `gpus_per_machine: 1`). Failing tests: both parse and dry-run with derived machine count 2 and member layout of two 1-GPU members. Then run the full suite:

```bash
bazel test --config=cuda13 //rtp_llm/test/perf_test/... 
```

Expected: all PASS, no existing test modified except additive extensions.

- [ ] **6.5 Commit milestone 6**

```bash
git add rtp_llm/test/perf_test/execution/aone_gang.py rtp_llm/test/perf_test/aone_gang_executor_test.py \
        rtp_llm/test/perf_test/execution/aone.py rtp_llm/test/perf_test/dry_run.py rtp_llm/test/perf_test/entry.py \
        rtp_llm/test/perf_test/examples/ rtp_llm/test/perf_test/BUILD rtp_llm/test/perf_test/aone_examples_test.py
git commit -m "feat(perf): wire gang execution into the unified aone entry"
```

---

### Task 7: CI Pipeline And Real Smokes (internal repo + live AOne)

**Files:**
- Create (internal repo, NOT github-opensource): `.aoneci/multi-node-perf.yaml`

**Interfaces:**
- Consumes: the unified entry from Milestone 6; the `cache-grid-perf.yaml` job-anchor pattern (checkout, `create_symlinks.sh` submodule pinning, `upload-artifact`); a1 CLI for pipeline run/status.

- [ ] **7.1 CI pipeline YAML**

Write `.aoneci/multi-node-perf.yaml` in the internal repo root worktree (`/data0/zhouhaiyu.zhy/project/RTP-LLM-worktrees/cache_grid_main/.aoneci/`): concurrency group `multi-node-perf` (parallelism 1, no cancel-in-progress), params for github commit, model identity, topology, `gpus_per_machine`, GPU type, resource; one job (no GPU runner) that checks out pinned sources and runs `./tools/perf_test --mode=... --action=run --config=<delivered gang config>` (the delivered config sets `execution.backend: "aone"`), then uploads the run directory and broker diagnostics. Verify: register the pipeline (`a1 ci pipeline list` shows it), trigger a first run with the decode gang example, and confirm from logs it reaches the build stage.

- [ ] **7.2 Real decode TP2 gang smoke**

Trigger the pipeline (or direct entry on a CI runner) with `examples/decode_aone_gang.json` semantics on `H20_CU13`/`daily`. Evidence to capture in the run directory: formal decode report, standard envelope with `backend=aone`, manifest gang topology with two distinct member hosts, broker diagnostics without token. Any failure: diagnose from broker events + member logs before changing code.

- [ ] **7.3 Real prefill TP2 gang smoke**

Same for prefill. Evidence: observed-reuse correctness enforced (a deliberately mismatched case would land `invalid_reuse` — at minimum assert all cases valid for the smoke config), standard prefill envelope and report, and a control single-machine AOne prefill run still passes.

- [ ] **7.4 Failing-member diagnostic run**

Run one diagnostic gang with an intentionally failing member (e.g. invalid engine arg on one member's rank). Evidence: every member exits nonzero, failure visible in broker events and manifest stage, master partial artifacts still collected.

- [ ] **7.5 Commit milestone 7**

```bash
git add <internal repo .aoneci/multi-node-perf.yaml>   # internal repo root, separate commit there
git commit -m "ci: add multi-node perf pipeline"
```

---

### Task 8: Documentation

**Files:**
- Create: `rtp_llm/test/perf_test/docs/multi_node_aone.md`
- Modify: `rtp_llm/test/perf_test/docs/aone_prefill.md`, `rtp_llm/test/perf_test/docs/decode.md` (links)

- [ ] **8.1 Multi-node AOne documentation**

Write `multi_node_aone.md`: the gang config contract (`gpus_per_machine`, derivation rules, rejected combinations), broker behavior and deadlines, worker flow, CI pipeline usage, artifact layout and manifest fields, and the phase-A/phase-B boundary. Link from `aone_prefill.md` and `decode.md`. Verify: documentation references only delivered commands and artifacts; doc examples parse and dry-run (covered by 6.4's example tests).

- [ ] **8.2 Commit milestone 8 and update OpenSpec tasks**

```bash
git add rtp_llm/test/perf_test/docs/
git commit -m "docs(perf): multi-node aone gang execution"
```

Tick the corresponding boxes in `openspec/changes/add-multi-node-aone-perf/tasks.md` as milestones complete (commit alongside each milestone's commit).
