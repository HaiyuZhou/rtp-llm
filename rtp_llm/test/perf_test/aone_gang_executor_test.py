"""Gang AOne executor tests: staged broker lifecycle with a mocked bazel runner.

The bazel runner is always a recording fake (no real bazel). Broker lifecycle
tests use the real loopback broker from the rendezvous module: the fake test
invocation registers the gang members against the live broker so the recorded
host metadata comes from a genuine rendezvous.
"""

import json
import os
import socket
import tempfile
import unittest
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.aone.broker import run_gang_broker
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.errors import CollectionError
from rtp_llm.test.perf_test.execution.aone import is_gang_plan
from rtp_llm.test.perf_test.execution.aone_bazel import create_default_aone_executor
from rtp_llm.test.perf_test.execution.aone_gang import (
    BROKER_HOST_ENV,
    BROKER_RUN_ID_ENV,
    BROKER_TOKEN_ENV,
    BROKER_URL_ENV,
    GangAOneExecutor,
    resolve_broker_host,
)
from rtp_llm.test.perf_test.plan import build_execution_plan, thaw
from rtp_llm.test.perf_test.results import RESULT_SCHEMA_VERSION
from rtp_llm.test.perf_test.run_directory import (
    initialize_run_directory,
    inspect_run_directory,
)
from rtp_llm.test.perf_test.state import RunState

TOPOLOGY = {"tp": 8, "dp": 2, "pp": 1, "ep": 1}
MACHINE_COUNT = 2
LOCAL_WORLD_SIZE = 8
RUN_ID = "run-gang"
LOOPBACK = "127.0.0.1"
STAGES = ["rendezvous", "build", "submit", "collect", "report"]


def _gang_source():
    return {
        "schema_version": 1,
        "model": {
            "label": "gang",
            "type": "registered",
            "checkpoint": "/m",
            "topology": dict(TOPOLOGY),
        },
        "engine": {
            "max_seq_len": 4097,
            "max_batch_tokens": 8192,
            "concurrency_limit": 1,
        },
        "workload": {
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [256],
            "batch_sizes": [1],
        },
        "execution": {
            "backend": "aone",
            "resource": "daily",
            "gpu_type": "H20",
            "max_parallel": MACHINE_COUNT,
            "gpus_per_machine": LOCAL_WORLD_SIZE,
        },
        "output": {},
        "regression": {},
        "publish": {},
    }


def _single_machine_source():
    source = _gang_source()
    source["model"]["topology"] = {"tp": 1, "dp": 1, "pp": 1, "ep": 1}
    source["execution"] = {
        "backend": "aone",
        "resource": "daily",
        "gpu_type": "H20",
        "max_parallel": 2,
        "shard_count": 2,
    }
    return source


def _cases():
    return [
        {
            "input_len": 512,
            "cache_len": 0,
            "compute_len": 512,
            "batch_size": 1,
            "measure_runs": 1,
            "seed": 1,
            "prefix_policy": "shared_seed",
        },
        {
            "input_len": 1024,
            "cache_len": 512,
            "compute_len": 512,
            "batch_size": 1,
            "measure_runs": 1,
            "seed": 1,
            "prefix_policy": "shared_seed",
        },
    ]


def _setup_run(root, source=None):
    source = _gang_source() if source is None else source
    plan = build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(source),
        cases=_cases(),
        result_directory=str(root),
    )
    initialize_run_directory(
        root, source_config=source, plan=plan, run_id=RUN_ID, code_version="rev"
    )
    return plan


def _case_result(case, status="ok"):
    identity = thaw(case.identity)
    ok = status == "ok"
    return {
        "case_id": case.case_id,
        "identity": identity,
        "status": status,
        "runs": 1,
        "errors": [] if ok else [{"code": status, "message": "remote case failed"}],
        "metrics": (
            {
                "input_length": identity["input_len"],
                "expected_reuse_length": identity["cache_len"],
                "observed_reuse_length": identity["cache_len"],
                "compute_length": identity["compute_len"],
                "latency_ms": 10.0,
                "throughput_tokens_per_second": 1000.0,
                "validation": {"shape": True, "reuse": True, "timing": True},
                "seed": 1,
                "prefix_policy": {"kind": "shared_seed"},
                "measurements": [{"run": 0, "latency_ms": 10.0}],
            }
            if ok
            else None
        ),
    }


def _envelope(plan, cases, status):
    successful = sum(case["status"] == "ok" for case in cases)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "mode": "prefill",
        "backend": "aone",
        "status": status,
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
        "model": {
            "label": "gang",
            "type": "registered",
            "checkpoint": "/m",
            "topology": dict(TOPOLOGY),
        },
        "summary": {
            "planned_cases": len(cases),
            "successful_cases": successful,
            "failed_cases": len(cases) - successful,
        },
        "cases": cases,
    }


class FakeBazelRun:
    """Recording stand-in for the bazel subprocess runner."""

    def __init__(self, returncodes=(0, 0), on_test_command=None, test_error=None):
        self.calls = []
        self._returncodes = list(returncodes)
        self.on_test_command = on_test_command
        self.test_error = test_error

    def __call__(self, command, **kwargs):
        entry = {"command": list(command)}
        entry.update(kwargs)
        self.calls.append(entry)
        if "test" in command:
            if self.test_error is not None:
                raise self.test_error(list(command))
            if self.on_test_command is not None:
                self.on_test_command(list(command), dict(entry.get("env") or {}))
        code = self._returncodes.pop(0) if self._returncodes else 0
        return SimpleNamespace(returncode=code)

    @property
    def build_command(self):
        return self._command_with("build")

    @property
    def test_command(self):
        return self._command_with("test")

    @property
    def test_entry(self):
        for entry in self.calls:
            if "test" in entry["command"]:
                return entry
        raise AssertionError("bazel test was never invoked")

    def _command_with(self, verb):
        for entry in self.calls:
            if verb in entry["command"]:
                return entry["command"]
        raise AssertionError(f"bazel {verb} was never invoked")


_BROKER_ENV_NAMES = (BROKER_URL_ENV, BROKER_TOKEN_ENV, BROKER_RUN_ID_ENV)


def _broker_env_from_env(env):
    return {name: env[name] for name in _BROKER_ENV_NAMES if name in env}


def _broker_env_from_entry(entry):
    return _broker_env_from_env(entry.get("env") or {})


def _token_from_entry(entry):
    token = _broker_env_from_entry(entry).get(BROKER_TOKEN_ENV)
    assert token
    return token


def _register_members(env, count):
    """Register gang members against the live broker exporting this env.

    The broker rejects two registrations from the same source address (its
    distinct-host backstop), so a loopback test can only register one real
    member; the recorded host then comes from a genuine rendezvous.
    """
    values = _broker_env_from_env(env)
    claim = socket.gethostbyaddr(LOOPBACK)[0]
    for index in range(count):
        request = urllib.request.Request(
            f"{values[BROKER_URL_ENV]}/v1/runs/"
            f"{values[BROKER_RUN_ID_ENV]}/register",
            data=json.dumps(
                {"shard_index": index, "hostname": claim, "port": 12333 + index}
            ).encode("utf-8"),
            headers={"Authorization": f"Bearer {values[BROKER_TOKEN_ENV]}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            assert 200 == response.status


def _write_master_run_tree(outputs, plan, status="completed"):
    cases = [_case_result(case) for case in plan.cases]
    return _write_master_run_tree_with_cases(outputs, plan, cases, status)


def _write_master_run_tree_with_cases(outputs, plan, cases, status):
    run = outputs / "run"
    (run / "results" / "cases").mkdir(parents=True)
    for case in cases:
        (run / "results" / "cases" / f"{case['case_id']}.json").write_text(
            json.dumps(case), encoding="utf-8"
        )
    (run / "results" / "result.json").write_text(
        json.dumps(_envelope(plan, cases, status)), encoding="utf-8"
    )
    (run / "results" / "summary.json").write_text(
        json.dumps({"run_id": RUN_ID, "status": status}), encoding="utf-8"
    )
    (run / "reports").mkdir()
    (run / "reports" / "master_report.json").write_text("{}", encoding="utf-8")
    (run / "traces").mkdir()
    (run / "traces" / "master_trace.json").write_text("{}", encoding="utf-8")
    (run / "profiles").mkdir()
    (run / "profiles" / "master_profile.json").write_text("{}", encoding="utf-8")
    return run


def _write_testlogs_tree(base, plan, *, status="completed", master=True):
    for index in range(MACHINE_COUNT):
        shard = base / f"shard_{index + 1}_of_{MACHINE_COUNT}"
        outputs = shard / "test.outputs"
        outputs.mkdir(parents=True)
        (shard / "test.log").write_text(f"shard {index} log\n", encoding="utf-8")
        (outputs / "gang_worker.json").write_text(
            json.dumps({"shard_index": index, "state": "completed"}),
            encoding="utf-8",
        )
        if index == 0 and master:
            _write_master_run_tree(outputs, plan, status=status)


class _GangExecutorTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.testlogs = self.workspace / "bazel-testlogs"

    def _setup_run(self, name="run-gang"):
        self.root = self.workspace / "runs" / name
        self.plan = _setup_run(self.root)
        return self.plan

    def _shard_base(self):
        relative = (self.root / "workload" / "aone_package").relative_to(self.workspace)
        return self.testlogs / relative / "gang"

    def _executor(self, runner, **kwargs):
        options = {
            "workspace_root": self.workspace,
            "resource": "daily",
            "gpu_type": "H20",
            "max_parallel": MACHINE_COUNT,
            "testlogs_root": self.testlogs,
            "run_command": runner,
            "broker_host": LOOPBACK,
        }
        options.update(kwargs)
        return GangAOneExecutor(**options)

    def _manifest(self):
        return inspect_run_directory(self.root).manifest

    def _assert_token_persisted_nowhere(self, token):
        files = [path for path in sorted(self.root.rglob("*")) if path.is_file()]
        self.assertTrue(files)
        for path in files:
            self.assertNotIn(
                token,
                path.read_text(encoding="utf-8", errors="replace"),
                str(path),
            )


class GangStageLifecycleTest(_GangExecutorTestCase):
    def test_broker_constructor_failure_records_rendezvous_failure(self):
        plan = self._setup_run()
        with patch(
            "rtp_llm.test.perf_test.execution.aone_gang.run_gang_broker",
            side_effect=RuntimeError("broker construction failed"),
        ):
            outcome = self._executor(FakeBazelRun()).execute(plan)

        self.assertEqual(RunState.FAILED, outcome.state)
        self.assertIn("broker construction failed", str(outcome.failure))
        self.assertEqual("failed", self._manifest()["stages"]["rendezvous"]["status"])

    def test_broker_stop_failure_does_not_mask_build_failure(self):
        plan = self._setup_run()

        def broker_with_failing_stop(*args, **kwargs):
            broker = run_gang_broker(*args, **kwargs)
            original_stop = broker.stop

            def stop():
                original_stop()
                raise RuntimeError("stop failed")

            broker.stop = stop
            return broker

        with patch(
            "rtp_llm.test.perf_test.execution.aone_gang.run_gang_broker",
            side_effect=broker_with_failing_stop,
        ):
            outcome = self._executor(FakeBazelRun(returncodes=(1,))).execute(plan)

        self.assertEqual(RunState.FAILED, outcome.state)
        self.assertIn("Bazel gang build exited 1", str(outcome.failure))
        self.assertNotIn("stop failed", str(outcome.failure))

    def test_completed_gang_persists_stages_in_order_and_exports_broker_env(self):
        plan = self._setup_run()
        _write_testlogs_tree(self._shard_base(), plan)
        runner = FakeBazelRun(
            on_test_command=lambda command, env: _register_members(env, 1)
        )

        outcome = self._executor(runner).execute(plan)

        self.assertEqual(RunState.COMPLETED, outcome.state)
        self.assertEqual(0, outcome.exit_code)
        self.assertEqual("formal", outcome.report["report_kind"])
        manifest = self._manifest()
        self.assertEqual(
            {stage: "completed" for stage in STAGES},
            {name: stage["status"] for name, stage in manifest["stages"].items()},
        )
        # The manifest is key-sorted JSON, so stage order is recorded
        # explicitly inside every stage entry.
        self.assertEqual(
            {stage: index for index, stage in enumerate(STAGES)},
            {name: stage["stage_index"] for name, stage in manifest["stages"].items()},
        )
        test_entry = runner.test_entry
        test_command = test_entry["command"]
        self.assertIn("--strategy=TestRunner=remote", test_command)
        self.assertIn("--test_timeout=7000", test_command)
        jobs = int(
            next(arg for arg in test_command if arg.startswith("--jobs=")).split(
                "=", 1
            )[1]
        )
        self.assertGreaterEqual(jobs, MACHINE_COUNT)
        # The broker coordinates travel through env_inherit declared on the
        # generated rule: they ride the submit environment, never the
        # command line (where the token would be visible to ps and logs).
        self.assertEqual(
            [],
            [
                arg
                for arg in test_command
                if arg.startswith("--test_env=PERF_TEST_BROKER_")
            ],
        )
        rendezvous = manifest["stages"]["rendezvous"]
        submit_env = test_entry["env"]
        self.assertEqual(rendezvous["broker_url"], submit_env[BROKER_URL_ENV])
        self.assertEqual(rendezvous["broker_run_id"], submit_env[BROKER_RUN_ID_ENV])
        self.assertEqual(
            list(_BROKER_ENV_NAMES),
            manifest["stages"]["submit"]["env_inherit"],
        )
        broker_env = _broker_env_from_entry(test_entry)
        self.assertEqual(3, len(broker_env))
        # Only the env var names are recorded; the secret value itself must
        # not survive anywhere under the run directory.
        self._assert_token_persisted_nowhere(broker_env[BROKER_TOKEN_ENV])

    def test_manifest_records_gang_metadata_with_sanitized_hosts(self):
        plan = self._setup_run()
        _write_testlogs_tree(self._shard_base(), plan)
        runner = FakeBazelRun(
            on_test_command=lambda command, env: _register_members(env, 1)
        )

        self._executor(runner).execute(plan)

        manifest = self._manifest()
        gang = manifest["gang"]
        self.assertEqual(MACHINE_COUNT, gang["machine_count"])
        self.assertEqual(LOCAL_WORLD_SIZE, gang["local_world_size"])
        self.assertEqual("H20", gang["gpu_type"])
        self.assertEqual(
            manifest["stages"]["rendezvous"]["broker_run_id"], gang["broker_run_id"]
        )
        claim = socket.gethostbyaddr(LOOPBACK)[0]
        self.assertEqual(
            [{"shard_index": 0, "hostname": claim}],
            gang["hosts"],
        )
        # Host records are sanitized: shard index and hostname only.
        for entry in gang["hosts"]:
            self.assertEqual({"shard_index", "hostname"}, set(entry))

    def test_build_failure_fails_build_stage_without_submit_or_collect(self):
        plan = self._setup_run()
        runner = FakeBazelRun(returncodes=(1,))

        outcome = self._executor(runner).execute(plan)

        self.assertEqual(RunState.FAILED, outcome.state)
        self.assertEqual(6, outcome.exit_code)
        self.assertEqual(1, len(runner.calls))
        manifest = self._manifest()
        self.assertEqual({"rendezvous", "build"}, set(manifest["stages"]))
        self.assertEqual("failed", manifest["stages"]["build"]["status"])
        self.assertEqual(1, manifest["stages"]["build"]["stage_index"])
        self.assertEqual("failed", manifest["state"])


class GangSubmitFailureTest(_GangExecutorTestCase):
    def test_nonzero_test_exit_overrides_completed_master_envelope(self):
        plan = self._setup_run()
        _write_testlogs_tree(self._shard_base(), plan, status="completed")

        outcome = self._executor(FakeBazelRun(returncodes=(0, 3))).execute(plan)

        self.assertEqual(RunState.FAILED, outcome.state)
        self.assertIn("Bazel gang test exited 3", str(outcome.failure))
        self.assertEqual("failed", self._manifest()["state"])

    def test_test_jobs_cover_all_machines_even_with_lower_executor_limit(self):
        plan = self._setup_run()
        _write_testlogs_tree(self._shard_base(), plan)
        runner = FakeBazelRun()

        self._executor(runner, max_parallel=1).execute(plan)

        self.assertIn(f"--jobs={MACHINE_COUNT}", runner.test_command)

    def test_submit_failure_records_stage_and_partial_collection_still_runs(self):
        plan = self._setup_run()
        first, second = plan.cases
        cases = [_case_result(first), _case_result(second, status="request_failed")]
        base = self._shard_base()
        for index in range(MACHINE_COUNT):
            shard = base / f"shard_{index + 1}_of_{MACHINE_COUNT}"
            outputs = shard / "test.outputs"
            outputs.mkdir(parents=True)
            (shard / "test.log").write_text(f"shard {index} log\n", encoding="utf-8")
            (outputs / "gang_worker.json").write_text(
                json.dumps({"shard_index": index, "state": "completed"}),
                encoding="utf-8",
            )
            if index == 0:
                _write_master_run_tree_with_cases(
                    outputs, plan, cases, status="partial"
                )
        runner = FakeBazelRun(returncodes=(0, 3))
        reporter = Mock(side_effect=RuntimeError("chart failed"))

        outcome = self._executor(runner, reporter=reporter).execute(plan)

        self.assertEqual(RunState.PARTIAL, outcome.state)
        self.assertEqual(5, outcome.exit_code)
        manifest = self._manifest()
        self.assertEqual("failed", manifest["stages"]["submit"]["status"])
        self.assertEqual(3, manifest["stages"]["submit"]["bazel_exit_code"])
        self.assertEqual("partial", manifest["stages"]["collect"]["status"])
        self.assertEqual("failed", manifest["stages"]["report"]["status"])
        self.assertEqual("partial", manifest["state"])
        # Partial collection really ran: the master envelope is on disk.
        self.assertEqual(
            "partial",
            json.loads(
                (self.root / "results" / "result.json").read_text(encoding="utf-8")
            )["status"],
        )

    def test_submit_failure_without_master_tree_keeps_the_test_failure_exit_code(self):
        self._setup_run()
        runner = FakeBazelRun(returncodes=(0, 4))
        reporter = Mock(side_effect=RuntimeError("chart failed"))

        outcome = self._executor(runner, reporter=reporter).execute(self.plan)

        self.assertEqual(RunState.FAILED, outcome.state)
        self.assertEqual(6, outcome.exit_code)
        self.assertIn("Bazel gang test exited 4", str(outcome.failure))
        self.assertNotIn("chart failed", str(outcome.failure))
        manifest = self._manifest()
        self.assertEqual("failed", manifest["stages"]["submit"]["status"])
        self.assertEqual(4, manifest["stages"]["submit"]["bazel_exit_code"])
        self.assertEqual("failed", manifest["stages"]["collect"]["status"])
        self.assertEqual("failed", manifest["stages"]["report"]["status"])
        self.assertEqual("failed", manifest["state"])

    def test_passing_test_without_master_tree_fails_collection(self):
        self._setup_run()
        runner = FakeBazelRun()

        outcome = self._executor(runner).execute(self.plan)

        self.assertEqual(RunState.FAILED, outcome.state)
        self.assertEqual(6, outcome.exit_code)
        self.assertIsInstance(outcome.failure, CollectionError)
        manifest = self._manifest()
        self.assertEqual("completed", manifest["stages"]["submit"]["status"])
        self.assertEqual("failed", manifest["stages"]["collect"]["status"])

    def test_runner_error_is_redacted_before_persistence(self):
        self._setup_run()
        runner = FakeBazelRun(
            test_error=lambda command: RuntimeError(f"cannot launch: {command}")
        )

        outcome = self._executor(runner).execute(self.plan)

        self.assertEqual(RunState.FAILED, outcome.state)
        self.assertEqual(6, outcome.exit_code)
        manifest = self._manifest()
        error = manifest["stages"]["submit"]["error"]
        self.assertIn("cannot launch", error)
        token = _token_from_entry(runner.test_entry)
        self.assertNotIn(token, error)
        self._assert_token_persisted_nowhere(token)


class GangSubmitTransportTest(_GangExecutorTestCase):
    """Broker coordinates reach the shards through env, not argv."""

    def test_broker_coordinates_travel_through_env_inherit_not_argv(self):
        plan = self._setup_run()
        _write_testlogs_tree(self._shard_base(), plan)
        seen = {}

        def on_test(command, env):
            seen["command"] = command
            seen["env"] = dict(env)

        runner = FakeBazelRun(on_test_command=on_test)
        for name in _BROKER_ENV_NAMES:
            os.environ.pop(name, None)

        outcome = self._executor(runner).execute(plan)

        self.assertEqual(RunState.COMPLETED, outcome.state)
        # Present in the submit environment during the bazel test run...
        for name in _BROKER_ENV_NAMES:
            with self.subTest(variable=name):
                self.assertIn(name, seen["env"])
        self.assertTrue(seen["env"][BROKER_URL_ENV].startswith("http://"))
        # ...absent from the recorded argv, and cleaned up afterwards so no
        # later stage of the submit process can observe them.
        self.assertEqual(
            [],
            [
                arg
                for arg in seen["command"]
                if arg.startswith("--test_env=PERF_TEST_BROKER_")
            ],
        )
        for name in _BROKER_ENV_NAMES:
            with self.subTest(cleanup=name):
                self.assertNotIn(name, os.environ)

    def test_exported_coordinates_are_restored_after_the_invocation(self):
        plan = self._setup_run()
        _write_testlogs_tree(self._shard_base(), plan)
        runner = FakeBazelRun()

        with patch.dict(os.environ, {BROKER_URL_ENV: "http://preexisting.example"}):
            outcome = self._executor(runner).execute(plan)
            # Read inside the patch: patch.dict removes the exported value
            # again when the context exits.
            restored_url = os.environ.get(BROKER_URL_ENV)

        self.assertEqual(RunState.COMPLETED, outcome.state)
        self.assertEqual("http://preexisting.example", restored_url)
        self.assertNotIn(BROKER_TOKEN_ENV, os.environ)
        self.assertNotIn(BROKER_RUN_ID_ENV, os.environ)


class GangCollectionParityTest(_GangExecutorTestCase):
    def test_collection_copies_all_master_artifacts_and_non_master_logs(self):
        plan = self._setup_run()
        _write_testlogs_tree(self._shard_base(), plan)

        outcome = self._executor(FakeBazelRun()).execute(plan)

        self.assertEqual(RunState.COMPLETED, outcome.state)
        self.assertEqual(2, len(outcome.collection.cases))
        for case in plan.cases:
            self.assertTrue(
                (self.root / "results" / "cases" / f"{case.case_id}.json").is_file()
            )
        envelope = json.loads(
            (self.root / "results" / "result.json").read_text(encoding="utf-8")
        )
        self.assertEqual("completed", envelope["status"])
        self.assertTrue((self.root / "results" / "summary.json").is_file())
        self.assertTrue((self.root / "reports" / "master_report.json").is_file())
        self.assertTrue((self.root / "traces" / "master_trace.json").is_file())
        self.assertTrue((self.root / "profiles" / "master_profile.json").is_file())
        master_logs = self.root / "workload" / "gang_shards" / "shard_1"
        self.assertTrue((master_logs / "gang_worker.json").is_file())
        waiter_logs = self.root / "workload" / "gang_shards" / "shard_2"
        self.assertTrue((waiter_logs / "test.log").is_file())
        self.assertTrue((waiter_logs / "gang_worker.json").is_file())
        self.assertFalse((waiter_logs / "run").exists())

    def test_collection_reads_the_master_tree_from_outputs_zip(self):
        plan = self._setup_run()
        base = self._shard_base()
        _write_testlogs_tree(base, plan, master=False)
        # Pack the master tree into the remote-style outputs archive.
        with tempfile.TemporaryDirectory() as staging:
            staging_root = Path(staging)
            outputs = staging_root / "test.outputs"
            tree = _write_master_run_tree(outputs, plan)
            master = base / f"shard_1_of_{MACHINE_COUNT}" / "test.outputs"
            with zipfile.ZipFile(master / "outputs.zip", "w") as archive:
                for path in sorted(tree.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(outputs))
        runner = FakeBazelRun()

        outcome = self._executor(runner).execute(plan)

        self.assertEqual(RunState.COMPLETED, outcome.state)
        self.assertTrue((self.root / "results" / "result.json").is_file())
        self.assertTrue((self.root / "traces" / "master_trace.json").is_file())


class BrokerHostResolutionTest(unittest.TestCase):
    def test_env_override_wins(self):
        self.assertEqual(
            "node-7.example",
            resolve_broker_host({"PERF_TEST_BROKER_HOST": "node-7.example"}),
        )

    def test_blank_override_falls_back_to_the_local_fqdn(self):
        self.assertEqual(socket.getfqdn(), resolve_broker_host({}))

    def test_executor_binds_the_env_override_broker_host(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "runs" / "run-gang"
            plan = _setup_run(root)
            with patch.dict(os.environ, {BROKER_HOST_ENV: LOOPBACK}):
                outcome = GangAOneExecutor(
                    workspace_root=workspace,
                    resource="daily",
                    gpu_type="H20",
                    max_parallel=MACHINE_COUNT,
                    testlogs_root=workspace / "bazel-testlogs",
                    run_command=FakeBazelRun(),
                ).execute(plan)
            self.assertEqual(RunState.FAILED, outcome.state)
            manifest = inspect_run_directory(root).manifest
            url = manifest["stages"]["rendezvous"]["broker_url"]
            self.assertTrue(url.startswith(f"http://{LOOPBACK}:"))


class GangDispatchTest(unittest.TestCase):
    def test_is_gang_plan_separates_gang_and_single_machine_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gang = _setup_run(root / "gang")
            single = _setup_run(root / "single", source=_single_machine_source())

            self.assertTrue(is_gang_plan(gang))
            self.assertFalse(is_gang_plan(single))

    def test_default_executor_routes_gang_plans_to_the_gang_executor(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = _setup_run(Path(directory) / "gang")
            sentinel = Mock()
            with patch(
                "rtp_llm.test.perf_test.execution.aone_gang.GangAOneExecutor"
            ) as gang_class:
                gang_class.return_value.execute.return_value = sentinel
                outcome = create_default_aone_executor(plan)

            self.assertIs(sentinel, outcome)
            gang_class.return_value.execute.assert_called_once_with(plan)
            constructor = gang_class.call_args.kwargs
            self.assertEqual("daily", constructor["resource"])
            self.assertEqual("H20", constructor["gpu_type"])
            self.assertEqual(MACHINE_COUNT, constructor["max_parallel"])

    def test_default_executor_keeps_single_machine_plans_on_aone_executor(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = _setup_run(
                Path(directory) / "single", source=_single_machine_source()
            )
            with patch(
                "rtp_llm.test.perf_test.execution.aone_bazel.AOneExecutor"
            ) as single_class, patch(
                "rtp_llm.test.perf_test.execution.aone_gang.GangAOneExecutor"
            ) as gang_class:
                single_class.return_value.execute.return_value = "single"
                outcome = create_default_aone_executor(plan)

            self.assertEqual("single", outcome)
            gang_class.assert_not_called()
            single_class.return_value.execute.assert_called_once_with(plan)


if __name__ == "__main__":
    unittest.main()
