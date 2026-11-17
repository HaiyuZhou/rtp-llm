"""Tests for aone gang_executor_test."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.test.perf_test.aone.config_schema import parse_execution
from rtp_llm.test.perf_test.aone.gang_worker import _merge_workload, load_gang_plan
from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.execution.aone_bazel import create_default_aone_executor
from rtp_llm.test.perf_test.execution.aone_gang import (
    BROKER_TOKEN_ENV,
    GangAOneExecutor,
)
from rtp_llm.test.perf_test.results import parse_result_envelope
from rtp_llm.test.perf_test.run_directory import inspect_run_directory
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.tests.support.gang import (
    _case_result,
    _envelope,
    _setup_run,
    _write_master_run_tree,
    _write_testlogs_tree,
)


class BrokerProxyTest(unittest.TestCase):
    def test_token_bearing_requests_ignore_all_ambient_proxy_settings(self):
        import requests

        from rtp_llm.test.perf_test.aone.gang_worker import BrokerClient

        for uppercase in (True, False):
            for scheme in ("http", "https"):
                with self.subTest(uppercase=uppercase, scheme=scheme):
                    proxy_env = {
                        name.upper() if uppercase else name: "http://proxy.invalid:8080"
                        for name in ("http_proxy", "https_proxy", "all_proxy")
                    }
                    sent = []

                    def send(session, request, **kwargs):
                        sent.append((request, kwargs))
                        response = requests.Response()
                        response.status_code = 200
                        response._content = b'{"status": "ok"}'
                        return response

                    with patch.dict("os.environ", proxy_env, clear=True), patch(
                        "requests.sessions.Session.send",
                        autospec=True,
                        side_effect=send,
                    ):
                        broker = BrokerClient(
                            f"{scheme}://broker.invalid:18000",
                            "test-secret",
                            "run-test",
                        )
                        broker.heartbeat(0)
                    self.assertEqual(1, len(sent))
                    request, settings = sent[0]
                    self.assertEqual({}, settings["proxies"])
                    self.assertEqual(
                        "Bearer test-secret", request.headers["Authorization"]
                    )
                    self.assertEqual(
                        f"{scheme}://broker.invalid:18000/v1/runs/run-test/heartbeat",
                        request.url,
                    )
                    self.assertEqual({"shard_index": 0}, json.loads(request.body))


class GangExecutorTest(unittest.TestCase):
    def test_master_collection_rejects_results_from_another_run_or_plan(self):
        mutations = {
            "run_id": lambda result: result.update(run_id="other-run"),
            "config": lambda result: result.update(config_fingerprint="a" * 64),
            "workload": lambda result: result.update(workload_fingerprint="b" * 64),
            "case": lambda result: result["cases"][0].update(case_id="other-case"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                root = workspace / "run"
                plan = _setup_run(root)
                outputs = workspace / "outputs"
                _write_master_run_tree(outputs, plan)
                path = outputs / "run/results/result.json"
                result = json.loads(path.read_text())
                mutate(result)
                path.write_text(json.dumps(result))
                executor = GangAOneExecutor(
                    workspace_root=workspace,
                    resource="daily",
                    gpu_type="H20",
                    max_parallel=2,
                )
                from rtp_llm.test.perf_test.errors import CollectionError

                with self.assertRaises(CollectionError):
                    executor._collect_master_tree(root, outputs)
                self.assertFalse((root / "results/result.json").exists())

    def test_gang_collection_retains_nonmaster_decode_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            trace_dir = outputs / "TEST_OUTPUT_run_1"
            trace_dir.mkdir(parents=True)
            (trace_dir / "decode_in8_out2_bs2_case_1.json").write_text(
                json.dumps({"traceEvents": []}), encoding="utf-8"
            )
            GangAOneExecutor._copy_worker_traces(outputs, root / "traces", 1)
            self.assertTrue(
                (root / "traces/shard_2/decode_in8_out2_bs2_case_1.json").is_file()
            )

    def _gang_flow(
        self,
        workspace,
        status="completed",
        remote_exit=0,
        member_failure=False,
        missing_master=False,
    ):
        root = workspace / "runs/gang"
        plan = _setup_run(root)
        base = workspace / "bazel-testlogs/runs/gang/workload/aone_package/gang"
        _write_testlogs_tree(base, plan, status=status, master=not missing_master)
        trace = base / "shard_2_of_2/test.outputs/TEST_OUTPUT_worker/decode_trace.json"
        trace.parent.mkdir()
        trace.write_text(json.dumps({"traceEvents": []}))
        if status == "partial":
            result = base / "shard_1_of_2/test.outputs/run/results/result.json"
            cases = [
                _case_result(plan.cases[0]),
                _case_result(plan.cases[1], "request_failed"),
            ]
            result.write_text(json.dumps(_envelope(plan, cases, status)))
            (result.parent / "cases.jsonl").write_text(
                "".join((json.dumps(case) + "\n" for case in cases))
            )
        calls = []
        brokers = []
        from rtp_llm.test.perf_test.aone.broker import run_gang_broker

        def broker_factory(*args, **kwargs):
            broker = run_gang_broker(*args, **kwargs)
            brokers.append(broker)
            return broker

        def run(command, **kwargs):
            calls.append((command, kwargs))
            if "test" in command:
                broker = brokers[0]
                for index in range(2):
                    broker.register_member(
                        shard_index=index,
                        hostname=f"host-{index}",
                        port=18000,
                        address=f"127.0.0.{index + 1}",
                    )
                broker.record_result(shard_index=0, state="completed")
                broker.record_result(
                    shard_index=1, state="failed" if member_failure else "completed"
                )
            return SimpleNamespace(returncode=remote_exit if "test" in command else 0)

        def executor(**kwargs):
            return GangAOneExecutor(**kwargs, run_command=run)

        with patch.dict(
            "os.environ",
            {
                "PERF_TEST_BAZEL_BUILD_JOBS": "obsolete",
                "PERF_TEST_BAZEL_TESTLOGS": "/obsolete/testlogs",
            },
        ), patch(
            "rtp_llm.test.perf_test.execution.aone_gang.run_gang_broker",
            side_effect=broker_factory,
        ), patch(
            "rtp_llm.test.perf_test.execution.aone_gang._workspace_root",
            return_value=workspace,
        ), patch(
            "rtp_llm.test.perf_test.execution.aone_gang.GangAOneExecutor",
            side_effect=executor,
        ):
            outcome = create_default_aone_executor(
                plan,
                broker_host="127.0.0.1",
                startup_args=("--batch",),
                common_args=("--announce_rc",),
            )
        for command, _kwargs in calls:
            self.assertEqual(["bazelisk", "--batch"], command[:2])
            self.assertIn("--announce_rc", command)
        return (root, plan, outcome, calls)

    def test_failed_member_overrides_successful_bazel_and_master_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan, outcome, _ = self._gang_flow(
                Path(directory), member_failure=True
            )
            self.assertEqual(RunState.FAILED, outcome.state)
            result = parse_result_envelope(
                json.loads((root / "results/result.json").read_text())
            )
            self.assertEqual(RunState.FAILED, result.status)

    def test_missing_master_still_collects_other_members_logs_and_traces(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan, outcome, _ = self._gang_flow(
                Path(directory), missing_master=True
            )
            self.assertEqual(RunState.FAILED, outcome.state)
            self.assertTrue((root / "workload/gang_shards/shard_2/test.log").is_file())
            self.assertTrue((root / "traces/shard_2/decode_trace.json").is_file())

    def test_gang_submission_preserves_package_inputs_artifacts_and_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            (root, plan, outcome, calls) = self._gang_flow(Path(directory))
            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertEqual(0, outcome.exit_code)
            self.assertEqual("formal", outcome.report["report_kind"])
            self.assertEqual(2, len(outcome.collection.cases))
            package = root / "workload/aone_package"
            gang = load_gang_plan(package / "gang_plan.json")
            (root / "remote-inputs").mkdir()
            merged = _merge_workload(
                package / "resolved_config.json",
                package / "resolved_workload.json",
                root / "remote-inputs",
            )
            loaded = load_resolved_config(merged, execution_parser=parse_execution)
            self.assertEqual(plan.config_fingerprint, loaded.config_fingerprint)
            self.assertEqual(16, gang["world_size"])
            self.assertEqual(2, gang["machine_count"])
            self.assertNotIn(
                "shard_count",
                json.loads((package / "resolved_config.json").read_text())["execution"],
            )
            self.assertTrue((root / "traces/master_trace.json").is_file())
            self.assertTrue((root / "profiles/master_profile.json").is_file())
            self.assertTrue((root / "workload/gang_shards/shard_2/test.log").is_file())
            self.assertTrue(
                all(
                    (
                        stage["status"] == "completed"
                        for stage in inspect_run_directory(root)
                        .manifest["stages"]
                        .values()
                    )
                )
            )
            (command, kwargs) = calls[1]
            token = kwargs["env"][BROKER_TOKEN_ENV]
            self.assertNotIn(token, " ".join(command))
            for path in root.rglob("*"):
                if path.is_file():
                    self.assertNotIn(token, path.read_text(errors="replace"))

    def test_nonzero_gang_exit_preserves_partial_results_and_overrides_completion(self):
        for status in ("partial", "completed"):
            with self.subTest(
                master_status=status
            ), tempfile.TemporaryDirectory() as directory:
                (root, plan, outcome, _) = self._gang_flow(
                    Path(directory), status=status, remote_exit=3
                )
                self.assertEqual(
                    RunState.FAILED,
                    outcome.state,
                )
                self.assertNotEqual(0, outcome.exit_code)
                self.assertEqual(
                    "failed",
                    inspect_run_directory(root).manifest["stages"]["submit"]["status"],
                )
                persisted = parse_result_envelope(
                    json.loads((root / "results/result.json").read_text())
                )
                self.assertEqual(outcome.state, persisted.status)
                self.assertEqual(2, len(persisted.cases))
                self.assertEqual("ok", persisted.cases[0].status)
                if status == "completed":
                    self.assertIn("Bazel gang test exited 3", str(outcome.failure))


if __name__ == "__main__":
    unittest.main()
