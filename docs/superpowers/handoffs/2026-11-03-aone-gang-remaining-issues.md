# Remaining Issues Handoff — Multi-Node AOne Gang Execution

Date: 2026-11-03
For: a follow-up fix session. Everything below is verified state, with exact
references. Read top to bottom; suggested session plan at the end.

## Shipped Context (do not redo)

- Branch `cache_grid_main` (github-opensource), 62 commits ahead of
  `origin/main`, also pushed to fork `HaiyuZhou/rtp-llm` (HEAD `7a626e2a63`).
- OpenSpec change `add-multi-node-aone-perf`: 24/24 tasks ticked, **not yet
  archived** (archive blocked on the spec-wording fix in P1-2 below).
- All three smokes verified: decode TP2 (runs 85653492, 85699664), prefill
  TP2 with reuse 128/128 (run 85697379), failure propagation (organic
  evidence in runs 85658860/85664561).
- CI pipeline 333571 (`.aoneci/multi-node-perf.yaml`) runs from internal
  branch `open_merge/1497`. Trigger:
  `a1 ci pipeline run 333571 -b open_merge/1497 --params-file <json>`
  (requires ALL 15 params; see `/data0/zhouhaiyu.zhy/tmp/opencode/mnp-run-*.json`).
- Local bazel: host `/opt/conda310` was deleted; use wrapper
  `/data0/zhouhaiyu.zhy/tmp/opencode/ns_bazel.sh <targets>` (CPU tests need
  `--test_env=GPU_COUNT=0` semantics, wrapper includes it).
- SDD ledger with full history: `.superpowers/sdd/2026-11-02-multi-node-aone-perf/progress.md`
  (git-ignored scratch; this doc supersedes it).

## P1 — Fix before merge/archive

### P1-1. Broker `handle_error` override is dead code (stderr noise persists)
- `rtp_llm/test/perf_test/aone/broker.py:514-528` defines `handle_error` on
  `_BrokerHandler` (the request handler). `socketserver` calls
  `handle_error` on the **server** class (`_BrokerServer`), never on the
  handler — the override is unreachable, and its `super().handle_error(...)`
  fallback would raise AttributeError if ever called. Client-disconnect
  tracebacks (BrokenPipeError/ConnectionResetError during 30 s `members`
  polls) still print to the broker's stderr.
- Fix: move the override to `_BrokerServer` (mirror the existing
  `log_message` silencer intent). One focused test: a client that connects
  and drops mid-request produces no traceback in captured stderr.
- Final-review finding 8; the only fix-wave item judged NOT ADDRESSED.

### P1-2. Spec/design wording contradicts the shipped hostname behavior
The smoke phase changed registration semantics (commit `c7bf923d33` area —
broker tolerance, see `aone/broker.py` register): claimed hostname vs
reverse-DNS mismatch is now **tolerated and both recorded** (container name
vs machine FQDN legitimately differ on AOne workers); the same-host gate is
address/hostname **uniqueness**, not equality. The planning artifacts still
describe the old contract:
- `openspec/changes/add-multi-node-aone-perf/specs/multi-node-aone-orchestration/spec.md`
  — scenario "Register a member" says "confirms the hostname matches".
  Reword to: records the claimed hostname and the best-effort reverse-lookup
  result; rejects only duplicate address or duplicate claimed hostname.
  (Scenario "Two members on one host" is still correct.)
- `openspec/changes/add-multi-node-aone-perf/design.md` — "hostname is
  cross-checked, mismatch rejected" in the broker contract section.
- `rtp_llm/test/perf_test/docs/multi_node_aone.md` — broker section says
  "hostname cross-check"; update to tolerance + uniqueness wording.
- After fixing: `openspec validate add-multi-node-aone-perf --strict`, then
  archive the change (`openspec archive`-equivalent flow used by this repo:
  see archive dir `openspec/changes/archive/2026-11-01-refactor-decode-perf-test/`
  for the expected result layout) and sync main specs.

### P1-3. Internal-repo CI commits need merging; stale branch cleanup
- `origin/open_merge/1497` (internal repo) carries 5 local pipeline commits
  on top of its previous head: `4f02b0a949` (yaml), `1b2ab7b975`,
  `88a3a038db`, `3cbf1dd16a`, `25a71919d4` (path/symlink/host-key fixes).
  They must reach the branch the pipeline watches, or the yaml must be
  cherry-picked to the canonical CI branch, before `open_merge/1497` is
  ever rebased away. Note: the full `internal_source/` overlay (deps/,
  bazel/, ci/) exists on `open_merge/1497` but NOT on `origin/main-internal`
  (reduced tree) — pre-existing repo-state quirk worth flagging to the repo
  owners.
- Stale: `origin/feat/multi-node-perf-ci` + local worktree
  `/data0/zhouhaiyu.zhy/tmp/opencode/mnp-ci` (superseded by running from
  `open_merge/1497`). Safe to delete both once nothing references them.

### P1-4. Branch integration decision (user's call, then execute)
62 commits on `cache_grid_main` vs `origin/main`; fork already has them.
Merge locally / push + MR / keep — needs the user's choice, then the
finishing flow. Nothing pushed to origin/main yet.

## P2 — Small fixes (batchable, each with a focused test)

1. **Broker state-file lag**: partial member results appear only in
   `broker_events.jsonl` until the next state transition; call
   `_persist_state_locked()` after recording any result
   (`aone/broker.py` `record_result`). (Final-review #1.)
2. **Broker `stop()` before `start()`** raises FileNotFoundError; guard with
   `if not self._started: return` (`aone/broker.py:187`). (#4.)
3. **Flaky-risk timing assertion**: `aone_broker_test.py` register-deadline
   test asserts `elapsed >= 0.3` against a 0.4 s deadline — only ~0.1 s slack
   for broker startup + round trip on a loaded host. Compute the expected
   remaining deadline after the register call, or lower the bound. (#2.)
4. **Executor `finally`**: `broker.stop()` unguarded (persistence inside can
   mask the real outcome) and `run_gang_broker(...)` constructor call sits
   outside the rendezvous failure handler — wrap both
   (`execution/aone_gang.py` `execute()`).
5. **Coverage nits** (one commit, cheap): 405 method-not-allowed branch;
   `lost_shard_indices` persisted field; "nonzero bazel + completed envelope
   → FAILED" override branch; `max(max_parallel, machine_count)` when
   `max_parallel < machine_count` side. Files: `aone/broker.py`,
   `execution/aone_gang.py` + their tests.
6. **Stale error wording** `"backend: unsupported for result mode"` — the
   check is no longer mode-conditional (`results.py:708` area); reword
   carefully (message-matching tests may reference it).
7. **Forged out-of-universe backend** now fails at envelope parse instead of
   upfront `PlanningError` in `decode/finalize.py`; optional up-front
   `plan.backend not in ("local","aone")` check to restore the old failure
   shape.
8. **Worker retry loops** have no client-side cap when the broker becomes
   unreachable after registration (Bazel timeout is the only backstop) —
   accepted by design; add a bounded retry only if it bites.
9. **Heartbeat join** `HEARTBEAT_JOIN_TIMEOUT_S = 5` vs 30 s request timeout:
   a beat stuck in a slow call can outlive `stop()` by ~30 s (daemon thread,
   process exit unaffected). Note-only unless observed.
10. **Legacy extraction nit**: malformed inherited `GANG_CONFIG_STRING` now
    fails pre-try/except (traceback + exit 1, no log upload) instead of the
    legacy in-umbrella path — unreachable for launcher-emitted strings;
    document or normalize.
11. **Env duplication observation**: gang engine now receives topology via
    BOTH argv (`--tp_size=...`) and gang env (`TP_SIZE`/`WORLD_SIZE`...) —
    harmless (argv wins where both parsed), intentional belt-and-suspenders.
    Note in `docs/multi_node_aone.md` worker section if touched anyway.

## P3 — Parked with rulings (no action unless revisited)

- Waiter early `completed` vote persists in per-member results when the gang
  later fails — outcome propagation is via gang state; diagnostics cosmetics
  only. (Dropped in final-review triage.)
- Two-brokers-one-state_dir test concern — invalid (fresh TemporaryDirectory
  per test). (Dropped.)
- Private-helper imports (`_safe_zip_member`, `_MAX_ARCHIVE_MEMBERS`, ...)
  from `aone_bazel` — sanctioned reuse, not duplication. (Dropped.)
- `_default_prefill_master_executor` had no direct unit test at final
  review — the fix wave added `DefaultPrefillMasterExecutorTest`, and the
  live path has now run green in CI (85697379). Residual risk low.

## Pre-existing (NOT from this branch; verify before claiming)

1. **Four failing perf-test targets** (fail identically at the branch base
   `48fbfe7e97`, verified by stash + clean-tree rerun):
   - `config_resolution_test` — distribution workloads set
     `workload.batch_sizes`, which the distribution allowed-set at
     `config_schema.py:532` no longer accepts (schema drift).
   - `golden_test`, `baseline_regression_test`, `odps_publisher_test` —
     fixtures pinned to envelope `schema_version: 1` while
     `RESULT_SCHEMA_VERSION = 2` (bumped 2026-10-30 in `c1ab4d5939`,
     before this series). Fixture migration, not code.
   - Fix as a separate small change; do not mix into this branch's leftovers.
2. **Host `/opt/conda310` deleted** (2026-11-02, not us): local engine
   launches and any local-execution bazel path need it restored (platform
   ticket). Wrappers for CPU tests: `ns_bazel.sh` / `ns_entry.sh` in
   `/data0/zhouhaiyu.zhy/tmp/opencode/`.
3. **Legacy `wait_master_done`** reads `PyEnvConfigs.distribute_config` as a
   class attribute — instance-only, so non-rank-0 legacy workers would
   AttributeError at runtime (pre-existing, disclosed during the 4a
   extraction; untouched).

## Suggested session plan (one fix session)

1. P1-1 + P2-1..P2-5 as one broker/executor polish commit set (TDD each).
2. P1-2 wording fixes → strict-validate → archive OpenSpec change → sync
   main specs (mirror the 2026-11-01 archive layout).
3. P2-6..P2-11 as a second small commit if time allows.
4. P1-3 internal-repo merge/cherry-pick + stale branch cleanup (needs push
   authorization).
5. P1-4 integration decision with the user, then execute.
6. Full-suite gate before each push:
   `/data0/zhouhaiyu.zhy/tmp/opencode/ns_bazel.sh //rtp_llm/test/perf_test/...`
   — expect 57+ pass / exactly the four sanctioned pre-existing failures.
7. Optional live re-check after broker/executor edits:
   `a1 ci pipeline run 333571 -b open_merge/1497 --params-file
   /data0/zhouhaiyu.zhy/tmp/opencode/mnp-run-decode.json` → SUCCESS.

## Reference index

- Key commits (fork/branch `cache_grid_main`): `c29c1c92ba`..`7a626e2a63`
  (tasks 1–8), smoke-phase fixes `a14fb8c8d2` (proxy bypass + shard dirs),
  `c7bf923d33` (hostname tolerance), `ec72a69418` (engine argv → server),
  `7a626e2a63` (hub path resolution in argv), `1146a22a59` (engine log
  collection).
- Evidence runs (CI pipeline 333571): 85653492 / 85699664 (decode ok),
  85697379 (prefill ok, reuse 128/128), 85684042 (single-machine TP2 ok —
  discriminating experiment), 85690967 (diagnostic, engine log shows
  `reuse_cache: 0`), 85664561/85658860 (failure-propagation evidence).
- Artifact bundles for all runs: `/data0/zhouhaiyu.zhy/tmp/opencode/a<runid>/`.
- Pipeline params templates: `/data0/zhouhaiyu.zhy/tmp/opencode/mnp-run-{decode,prefill,tp2-single}.json`.
