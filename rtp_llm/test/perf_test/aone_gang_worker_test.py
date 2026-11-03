"""Unit tests for the gang worker orchestration (no network, no GPUs)."""

import contextlib
import io
import json
import os
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.test.perf_test.aone.broker import BrokerMember
from rtp_llm.test.perf_test.aone.gang_worker import (
    BROKER_RUN_ID_ENV,
    BROKER_TOKEN_ENV,
    BROKER_URL_ENV,
    EXIT_CONFIG_ERROR,
    EXIT_EXECUTION_ERROR,
    EXIT_RENDEZVOUS_ERROR,
    BrokerClient,
    BrokerRejectedError,
    GangBrokerFailure,
    GangWorkerContext,
    WorkerConfigError,
    _default_master_executor,
    _default_prefill_master_executor,
    _default_server_runner,
    build_worker_context,
    load_gang_plan,
    main as worker_main,
    merged_config_mode,
    run_gang_worker,
)

ROOT = Path(__file__).resolve().parent
EXAMPLES = {
    "prefill": ROOT / "examples" / "prefill_aone_gang.json",
    "decode": ROOT / "examples" / "decode_aone_gang.json",
}


@dataclass
class _FakeResponse:
    status_code: int
    body: dict

    def json(self):
        return self.body


class _FakeRequest:
    """Callable stand-in for ``requests.request`` recording every call."""

    def __init__(self, responses=None, error=None):
        self.calls = []
        self.responses = list(responses or [])
        self.error = error

    def __call__(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if self.error is not None:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return _FakeResponse(200, {})


class FakeBroker:
    """In-memory BrokerClient double mirroring the rendezvous semantics.

    ``report`` mirrors the real broker: a failed vote fails the gang, and the
    gang completes only once every member reported success. ``on_members`` is
    a test hook invoked on every members poll before the state is read.
    """

    def __init__(self, machine_count, *, members=(), gang_state="ready"):
        self.machine_count = machine_count
        self._members = list(members)
        self._gang_state = gang_state
        self._results = {}
        self.events = []
        self.cleanups = 0
        self.members_error = None
        self.register_error = None
        self.on_members = None

    @property
    def gang_state(self):
        return self._gang_state

    def register(self, shard_index, hostname, port):
        self.events.append(("register", shard_index, hostname, port))
        if self.register_error is not None:
            raise self.register_error
        if self._gang_state != "failed":
            self._gang_state = "ready" if self._members else "registering"
        return list(self._members)

    def members(self, timeout_s):
        if self.on_members is not None:
            self.on_members(self)
        self.events.append(("members", timeout_s))
        if self.members_error is not None:
            raise self.members_error
        if self._gang_state == "failed":
            raise GangBrokerFailure("gang failed")
        return list(self._members)

    def heartbeat(self, shard_index):
        self.events.append(("heartbeat", shard_index))
        if self._gang_state == "failed":
            raise GangBrokerFailure("gang failed")

    def report(self, shard_index, state, detail=None):
        if self._gang_state == "failed":
            raise GangBrokerFailure("gang failed")
        self.events.append(("report", shard_index, state))
        self._results[shard_index] = state
        if state == "failed":
            self._gang_state = "failed"
        elif len(self._results) >= self.machine_count and all(
            value == "completed" for value in self._results.values()
        ):
            self._gang_state = "completed"

    def cleanup(self):
        self.cleanups += 1

    def reports(self):
        return [event for event in self.events if event[0] == "report"]


class FakeServerHandle:
    def __init__(self, *, start_returns=True):
        self.started = False
        self.stopped = False
        self.start_returns = start_returns
        self.printed_logs = 0

    def start_server(self):
        self.started = True
        return self.start_returns

    def print_process_log(self):
        self.printed_logs += 1

    def stop_server(self):
        self.stopped = True


def _gang_plan(machine_count=2, local_world_size=2, master_shard_index=0):
    return {
        "schema_version": 1,
        "run_id": "run-123",
        "config_fingerprint": "config-fp",
        "workload_fingerprint": "workload-fp",
        "world_size": machine_count * local_world_size,
        "machine_count": machine_count,
        "local_world_size": local_world_size,
        "gpus_per_machine": local_world_size,
        "gpus_per_host": 8,
        "gpu_type": "H20",
        "topology": {"tp": local_world_size, "dp": 1, "pp": 1, "ep": 1},
        "master_shard_index": master_shard_index,
        "port_range": {"start": 12333, "count": local_world_size},
    }


def _members(machine_count):
    return [
        BrokerMember(
            shard_index=index,
            address=f"10.0.0.{index}",
            hostname=f"host-{index}",
            port=12333,
        )
        for index in range(machine_count)
    ]


def _live_heartbeat_threads():
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("gang-worker-heartbeat") and thread.is_alive()
    ]


class _WorkerTestCase(unittest.TestCase):
    """Shared harness: plan files, shard env, fakes, and fast poll loops."""

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.resolved_config = self.root / "resolved_config.json"
        self.resolved_config.write_text(json.dumps({"model": {}}), encoding="utf-8")
        self.gang_plan_path = self.root / "gang_plan.json"
        self.outputs = self.root / "test_outputs"
        self.contexts = []
        self.env_snapshots = []
        self.handles = []

    def tearDown(self):
        self.assertEqual([], _live_heartbeat_threads())
        self.temporary.cleanup()

    def _write_plan(self, plan=None):
        plan = _gang_plan() if plan is None else plan
        self.gang_plan_path.write_text(json.dumps(plan), encoding="utf-8")
        return plan

    def _server_runner(self, handle=None, start_returns=True):
        handle = FakeServerHandle(start_returns=start_returns) if handle is None else handle
        self.handles.append(handle)

        def runner(context):
            self.contexts.append(context)
            self.env_snapshots.append(dict(os.environ))
            return handle

        return runner, handle

    def _seed_completed(self, broker, shard_indices):
        """Pre-cast the other shards' success votes, as if they reported."""
        for index in shard_indices:
            broker._results[index] = "completed"

    def _run(self, broker, server_runner, *, shard_index=1, machine_count=2,
             master_executor=None, clock=lambda: 0.0):
        with patch.dict(
            os.environ,
            {
                "TEST_SHARD_INDEX": str(shard_index),
                "TEST_TOTAL_SHARDS": str(machine_count),
            },
        ):
            with patch("rtp_llm.test.perf_test.aone.gang_worker._sleep"):
                return run_gang_worker(
                    self.resolved_config,
                    self.gang_plan_path,
                    broker=broker,
                    server_runner=server_runner,
                    test_outputs_dir=self.outputs,
                    master_executor=master_executor,
                    clock=clock,
                )


class BrokerClientTest(unittest.TestCase):
    def _client(self, transport):
        return BrokerClient(
            "http://broker.example/",
            "secret-token",
            "run-1",
            request=transport,
        )

    def test_default_transport_disables_ambient_proxies(self):
        import rtp_llm.test.perf_test.aone.gang_worker as gang_worker_module

        captured = []

        def fake_request(method, url, **kwargs):
            captured.append(
                {"method": method, "url": url, "proxies": kwargs.get("proxies", "missing")}
            )
            return _FakeResponse(
                200, {"status": "registered", "gang_state": "ready"}
            )

        client = BrokerClient("http://broker.example/", "secret-token", "run-1")
        original = gang_worker_module.requests.request
        gang_worker_module.requests.request = fake_request
        try:
            client.register(0, "host-0", 12333)
        finally:
            gang_worker_module.requests.request = original

        first = captured[0]
        self.assertEqual("POST", first["method"])
        self.assertEqual(
            {"http": None, "https": None},
            first["proxies"],
            "broker requests must bypass ambient http_proxy on remote workers",
        )
        self.assertTrue(captured, "register must issue at least one request")

    def test_members_sends_bearer_token_and_parses_members(self):
        transport = _FakeRequest(
            responses=[
                _FakeResponse(
                    200,
                    {
                        "run_id": "run-1",
                        "gang_state": "ready",
                        "machine_count": 2,
                        "members": [
                            {
                                "shard_index": 0,
                                "address": "10.0.0.0",
                                "hostname": "host-0",
                                "port": 12333,
                            }
                        ],
                    },
                )
            ]
        )
        client = self._client(transport)

        members = client.members(12.5)

        call = transport.calls[0]
        self.assertEqual("GET", call["method"])
        self.assertEqual("http://broker.example/v1/runs/run-1/members", call["url"])
        self.assertEqual({"Authorization": "Bearer secret-token"}, call["headers"])
        self.assertEqual(12.5, call["timeout"])
        self.assertEqual([BrokerMember(0, "10.0.0.0", "host-0", 12333)], members)
        self.assertEqual("ready", client.gang_state)

    def test_members_timeout_is_transient_and_returns_empty(self):
        import requests

        transport = _FakeRequest(error=requests.Timeout("read timed out"))
        client = self._client(transport)

        self.assertEqual([], client.members(30.0))

    def test_broker_reported_failure_raises_gang_failure(self):
        transport = _FakeRequest(
            responses=[_FakeResponse(504, {"error": "register_deadline_exceeded"})]
        )
        client = self._client(transport)

        with self.assertRaises(GangBrokerFailure):
            client.members(30.0)

    def test_register_rejection_raises_broker_rejected(self):
        transport = _FakeRequest(
            responses=[_FakeResponse(409, {"error": "gang_complete"})]
        )
        client = self._client(transport)

        with self.assertRaises(BrokerRejectedError):
            client.register(0, "host-0", 12333)

    def test_register_completing_gang_fetches_members(self):
        transport = _FakeRequest(
            responses=[
                _FakeResponse(200, {"status": "registered", "gang_state": "ready"}),
                _FakeResponse(
                    200,
                    {
                        "gang_state": "ready",
                        "members": [
                            {
                                "shard_index": 0,
                                "address": "10.0.0.0",
                                "hostname": "host-0",
                                "port": 12333,
                            }
                        ],
                    },
                ),
            ]
        )
        client = self._client(transport)

        members = client.register(0, "host-0", 12333)

        self.assertEqual(1, len(members))
        self.assertEqual("host-0", members[0].hostname)
        register_call = transport.calls[0]
        self.assertEqual("http://broker.example/v1/runs/run-1/register", register_call["url"])
        self.assertEqual(
            {"shard_index": 0, "hostname": "host-0", "port": 12333},
            register_call["json"],
        )

    def test_error_messages_never_contain_the_token(self):
        transport = _FakeRequest(responses=[_FakeResponse(401, {"error": "unauthorized"})])
        client = self._client(transport)

        with self.assertRaises(BrokerRejectedError) as caught:
            client.heartbeat(0)
        self.assertNotIn("secret-token", str(caught.exception))


class WorkerEnvTest(_WorkerTestCase):
    def test_missing_broker_env_fails_immediately_without_broker(self):
        self._write_plan()
        config = self.root / "merged.json"
        config.write_text("{}", encoding="utf-8")
        workload = self.root / "workload.json"
        workload.write_text("{}", encoding="utf-8")

        for hidden in (BROKER_URL_ENV, BROKER_TOKEN_ENV, BROKER_RUN_ID_ENV):
            with self.subTest(missing=hidden):
                environment = {
                    BROKER_URL_ENV: "http://broker.example",
                    BROKER_TOKEN_ENV: "secret-token",
                    BROKER_RUN_ID_ENV: "run-1",
                }
                environment.pop(hidden)
                with patch.dict(os.environ, environment, clear=True):
                    with patch(
                        "rtp_llm.test.perf_test.aone.gang_worker.BrokerClient"
                    ) as factory:
                        code = worker_main(
                            [
                                "--resolved-config", str(config),
                                "--resolved-workload", str(workload),
                                "--gang-plan", str(self.gang_plan_path),
                            ]
                        )
                self.assertEqual(EXIT_CONFIG_ERROR, code)
                factory.assert_not_called()

    def test_shard_env_mismatch_fails_before_broker_calls(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"])
        runner, handle = self._server_runner()

        with patch.dict(
            os.environ, {"TEST_SHARD_INDEX": "0", "TEST_TOTAL_SHARDS": "5"}
        ):
            code = run_gang_worker(
                self.resolved_config,
                self.gang_plan_path,
                broker=broker,
                server_runner=runner,
                test_outputs_dir=self.outputs,
            )

        self.assertEqual(EXIT_CONFIG_ERROR, code)
        self.assertEqual([], broker.events)
        self.assertFalse(handle.started)


class WorkerPreludeTest(_WorkerTestCase):
    def test_prelude_computes_rank_layout_and_calls_server_runner(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"], members=_members(2))
        self._seed_completed(broker, [0])
        runner, handle = self._server_runner()

        code = self._run(broker, runner, shard_index=1, machine_count=2,
                         master_executor=lambda *a, **k: 0)

        self.assertEqual(0, code)
        self.assertEqual(1, len(self.contexts))
        context = self.contexts[0]
        self.assertIsInstance(context, GangWorkerContext)
        self.assertEqual(1, context.shard_index)
        self.assertEqual(2, context.world_rank)
        self.assertEqual(2, context.local_world_size)
        self.assertEqual(
            "name:gang_part0,ip:host-0,port:12333;"
            "name:gang_part1,ip:host-1,port:12333",
            context.gang_config_string,
        )
        self.assertEqual("0,1", context.cuda_visible_devices)
        self.assertEqual("0,1", self.env_snapshots[0]["CUDA_VISIBLE_DEVICES"])
        self.assertEqual(
            context.gang_config_string,
            self.env_snapshots[0]["GANG_CONFIG_STRING"],
        )
        register = broker.events[0]
        self.assertEqual("register", register[0])
        self.assertEqual(1, register[1])
        self.assertEqual(12333, register[3])
        self.assertTrue(handle.started)
        self.assertTrue(handle.stopped)

    def test_single_gpu_host_gets_single_device(self):
        plan = self._write_plan(_gang_plan(machine_count=3, local_world_size=1))
        broker = FakeBroker(plan["machine_count"], members=_members(3))
        self._seed_completed(broker, [0, 1])
        runner, _ = self._server_runner()

        code = self._run(broker, runner, shard_index=2, machine_count=3,
                         master_executor=lambda *a, **k: 0)

        self.assertEqual(0, code)
        self.assertEqual("0", self.contexts[0].cuda_visible_devices)
        self.assertEqual(2, self.contexts[0].world_rank)
        self.assertIn("name:gang_part2,ip:host-2", self.contexts[0].gang_config_string)

    def test_formation_waits_until_gang_ready(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"], members=_members(2))
        self._seed_completed(broker, [0])
        runner, _ = self._server_runner()
        calls = {"count": 0}
        stash = {"members": None}

        def on_members(fake):
            calls["count"] += 1
            if calls["count"] == 1:
                stash["members"] = fake._members
                fake._members = []
                fake._gang_state = "registering"
            elif stash["members"] is not None:
                fake._members = stash["members"]
                fake._gang_state = "ready"
                stash["members"] = None

        broker.on_members = on_members

        code = self._run(broker, runner, shard_index=1, machine_count=2,
                         master_executor=lambda *a, **k: 0)

        self.assertEqual(0, code)
        members_polls = [event for event in broker.events if event[0] == "members"]
        self.assertGreaterEqual(len(members_polls), 2)

    def test_register_rejection_exits_nonzero_before_formation(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"])
        broker.register_error = BrokerRejectedError("HTTP 409 gang_complete")
        runner, handle = self._server_runner()

        code = self._run(broker, runner, shard_index=1, machine_count=2)

        self.assertEqual(EXIT_RENDEZVOUS_ERROR, code)
        self.assertEqual(1, len(broker.events))
        self.assertEqual("register", broker.events[0][0])
        self.assertFalse(handle.started)
        self.assertEqual([], broker.reports())

    def test_unexpected_startup_error_still_casts_the_failure_vote(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"], members=_members(2))

        def runner(context):
            self.contexts.append(context)
            raise ImportError("torch wheels are missing on this host")

        code = self._run(broker, runner, shard_index=0, machine_count=2)

        # The exit code was already nonzero without the vote; the vote is
        # what fails the gang immediately instead of waiting for the
        # heartbeat expiry.
        self.assertEqual(EXIT_EXECUTION_ERROR, code)
        self.assertEqual([("report", 0, "failed")], broker.reports())


class MasterExecutionTest(_WorkerTestCase):
    def test_master_runs_executor_with_handle_and_reports_completed(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"], members=_members(2))
        runner, handle = self._server_runner()
        executor_calls = []

        def executor(server_handle, **kwargs):
            executor_calls.append((server_handle, kwargs))
            return 0

        code = self._run(broker, runner, shard_index=0, machine_count=2,
                         master_executor=executor)

        self.assertEqual(0, code)
        self.assertEqual(1, len(executor_calls))
        server_handle, kwargs = executor_calls[0]
        self.assertIs(handle, server_handle)
        self.assertEqual(self.contexts[0], kwargs["context"])
        self.assertEqual(str(self.outputs / "run"), kwargs["result_directory"])
        self.assertEqual(str(self.resolved_config), kwargs["config_path"])
        self.assertEqual([("report", 0, "completed")], broker.reports())
        self.assertTrue(handle.stopped)
        self.assertEqual(0, broker.cleanups)

    def test_master_failure_keeps_partial_run_artifacts_and_reports_failed(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"], members=_members(2))
        runner, handle = self._server_runner()

        def executor(_server_handle, *, result_directory, **_kwargs):
            partial = Path(result_directory) / "case_0" / "partial.json"
            partial.parent.mkdir(parents=True)
            partial.write_text("{}", encoding="utf-8")
            return 7

        code = self._run(broker, runner, shard_index=0, machine_count=2,
                         master_executor=executor)

        self.assertEqual(7, code)
        self.assertEqual([("report", 0, "failed")], broker.reports())
        self.assertTrue(
            (self.outputs / "run" / "case_0" / "partial.json").exists(),
            "partial run artifacts must remain on disk",
        )
        self.assertTrue(handle.stopped)

    def test_master_executor_exception_reports_failed_and_exits_nonzero(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"], members=_members(2))
        runner, handle = self._server_runner()

        def executor(_server_handle, **_kwargs):
            raise RuntimeError("decode blew up")

        code = self._run(broker, runner, shard_index=0, machine_count=2,
                         master_executor=executor)

        self.assertEqual(EXIT_EXECUTION_ERROR, code)
        self.assertEqual([("report", 0, "failed")], broker.reports())
        self.assertTrue(handle.stopped)

    def test_master_success_after_gang_failure_exits_nonzero(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"], members=_members(2))
        handle = FakeServerHandle()
        self.handles.append(handle)

        def runner(context):
            self.contexts.append(context)
            # Another member reported failure while the master was starting.
            broker._gang_state = "failed"
            return handle

        code = self._run(broker, runner, shard_index=0, machine_count=2,
                         master_executor=lambda *a, **k: 0)

        self.assertEqual(EXIT_RENDEZVOUS_ERROR, code)
        self.assertTrue(handle.stopped)
        self.assertEqual([], broker.reports())

    def test_server_start_failure_reports_failed(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"], members=_members(2))
        runner, handle = self._server_runner(start_returns=False)

        code = self._run(broker, runner, shard_index=0, machine_count=2,
                         master_executor=lambda *a, **k: 0)

        self.assertEqual(EXIT_EXECUTION_ERROR, code)
        self.assertEqual([("report", 0, "failed")], broker.reports())
        self.assertEqual(1, handle.printed_logs)
        self.assertTrue(handle.stopped)


class WaiterExecutionTest(_WorkerTestCase):
    def test_waiter_returns_zero_after_master_reports_success(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"], members=_members(2))
        # The master shard (not modeled here) already reported success, so the
        # gang completes as soon as this waiter casts its own vote.
        self._seed_completed(broker, [0])
        runner, handle = self._server_runner()

        code = self._run(broker, runner, shard_index=1, machine_count=2)

        self.assertEqual(0, code)
        self.assertEqual([("report", 1, "completed")], broker.reports())
        self.assertTrue(handle.stopped)
        self.assertEqual(0, broker.cleanups)

    def test_waiter_polls_until_gang_completed(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"], members=_members(2))
        self._seed_completed(broker, [0])
        polls = {"count": 0}

        def on_members(fake):
            polls["count"] += 1
            # The waiter's own vote lands during the first completion poll.
            if polls["count"] == 2 and 1 in fake._results:
                fake._gang_state = "completed"

        broker.on_members = on_members
        runner, _ = self._server_runner()

        code = self._run(broker, runner, shard_index=1, machine_count=2)

        self.assertEqual(0, code)
        self.assertGreaterEqual(polls["count"], 2)

    def test_waiter_exits_nonzero_when_broker_reports_gang_failure(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"], members=_members(2))
        runner, handle = self._server_runner()

        def on_members(fake):
            # The gang fails only once this waiter is already waiting: the
            # first members call forms the gang, the second one reports the
            # broker-side failure.
            polls_so_far = sum(1 for event in fake.events if event[0] == "members")
            if polls_so_far >= 1:
                fake.members_error = GangBrokerFailure("member_failed")

        broker.on_members = on_members

        code = self._run(broker, runner, shard_index=1, machine_count=2)

        self.assertEqual(EXIT_RENDEZVOUS_ERROR, code)
        self.assertTrue(handle.started)
        self.assertTrue(handle.stopped)

    def test_waiter_report_rejection_is_immediate_nonzero(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"], members=_members(2))
        handle = FakeServerHandle()
        self.handles.append(handle)

        def runner(context):
            self.contexts.append(context)
            # The master failed while this waiter was starting its server.
            broker._gang_state = "failed"
            return handle

        code = self._run(broker, runner, shard_index=1, machine_count=2)

        self.assertEqual(EXIT_RENDEZVOUS_ERROR, code)
        # The rejected vote must abort before any completion polling: only
        # the formation poll ever happened.
        members_polls = [event for event in broker.events if event[0] == "members"]
        self.assertEqual(1, len(members_polls))
        self.assertEqual([], broker.reports())

    def test_waiter_exits_nonzero_when_failure_appears_while_waiting(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"], members=_members(2))
        self._seed_completed(broker, [0])
        runner, handle = self._server_runner()

        def on_members(fake):
            polls_so_far = sum(1 for event in fake.events if event[0] == "members")
            if polls_so_far >= 1:
                fake.members_error = GangBrokerFailure("heartbeat_expired")

        broker.on_members = on_members

        code = self._run(broker, runner, shard_index=1, machine_count=2)

        self.assertEqual(EXIT_RENDEZVOUS_ERROR, code)
        self.assertTrue(handle.started)
        self.assertTrue(handle.stopped)


class HeartbeatAndArtifactsTest(_WorkerTestCase):
    def test_heartbeat_starts_before_first_members_poll_and_keeps_flowing(self):
        plan = self._write_plan(_gang_plan(machine_count=3))
        broker = FakeBroker(plan["machine_count"], members=_members(3))
        # One vote short of completion, so this waiter really waits.
        self._seed_completed(broker, [0])
        runner, _ = self._server_runner()

        def on_members(fake):
            # The waiter keeps polling (and heartbeating) until several
            # beats have actually landed; then the gang completes.
            beats = sum(1 for event in fake.events if event[0] == "heartbeat")
            if beats >= 3:
                fake._gang_state = "completed"

        broker.on_members = on_members

        with patch(
            "rtp_llm.test.perf_test.aone.gang_worker.HEARTBEAT_INTERVAL_S", 0.001
        ):
            code = self._run(broker, runner, shard_index=1, machine_count=3)

        self.assertEqual(0, code)
        labels = [event[0] for event in broker.events]
        self.assertLess(labels.index("heartbeat"), labels.index("members"))
        self.assertGreaterEqual(labels.count("heartbeat"), 3)

    def test_worker_summary_artifact_lands_under_test_outputs(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"], members=_members(2))
        self._seed_completed(broker, [0])
        runner, _ = self._server_runner()

        code = self._run(broker, runner, shard_index=1, machine_count=2,
                         master_executor=lambda *a, **k: 0)

        self.assertEqual(0, code)
        summary_path = self.outputs / "gang_worker.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(1, summary["shard_index"])
        self.assertEqual(2, summary["world_rank"])
        self.assertEqual(0, summary["exit_code"])
        self.assertEqual("completed", summary["state"])
        self.assertEqual("run-123", summary["run_id"])
        self.assertNotIn("token", json.dumps(summary).lower())

    def test_failure_summary_records_failure_state(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"], members=_members(2))
        runner, _ = self._server_runner()

        code = self._run(broker, runner, shard_index=0, machine_count=2,
                         master_executor=lambda *a, **k: 3)

        self.assertEqual(3, code)
        summary = json.loads(
            (self.outputs / "gang_worker.json").read_text(encoding="utf-8")
        )
        self.assertEqual("failed", summary["state"])
        self.assertEqual(3, summary["exit_code"])

    def test_report_called_exactly_once_per_worker(self):
        plan = self._write_plan()
        broker = FakeBroker(plan["machine_count"], members=_members(2))
        self._seed_completed(broker, [0])
        runner, _ = self._server_runner()

        self._run(broker, runner, shard_index=1, machine_count=2,
                  master_executor=lambda *a, **k: 0)

        self.assertEqual([("report", 1, "completed")], broker.reports())


class WorkerMainTest(_WorkerTestCase):
    def test_main_merges_workload_and_drives_worker(self):
        plan = self._write_plan()
        workload = self.root / "resolved_workload.json"
        workload.write_text(json.dumps({"kind": "grid"}), encoding="utf-8")
        self.resolved_config.write_text(
            json.dumps({"model": {"label": "m"}}), encoding="utf-8"
        )
        runner, _ = self._server_runner()
        broker = FakeBroker(plan["machine_count"], members=_members(2))
        self._seed_completed(broker, [1])
        executor_configs = []

        def executor(_handle, *, config_path, **_kwargs):
            executor_configs.append(Path(config_path))
            return 0

        environment = {
            BROKER_URL_ENV: "http://broker.example",
            BROKER_TOKEN_ENV: "secret-token",
            BROKER_RUN_ID_ENV: "run-1",
            "TEST_UNDECLARED_OUTPUTS_DIR": str(self.outputs),
            "TEST_SHARD_INDEX": "0",
            "TEST_TOTAL_SHARDS": "2",
        }

        def factory(url, token, run_id):
            self.assertEqual("http://broker.example", url)
            self.assertEqual("secret-token", token)
            self.assertEqual("run-1", run_id)
            return broker

        with patch.dict(os.environ, environment, clear=True):
            with patch("rtp_llm.test.perf_test.aone.gang_worker._sleep"):
                code = worker_main(
                    [
                        "--resolved-config", str(self.resolved_config),
                        "--resolved-workload", str(workload),
                        "--gang-plan", str(self.gang_plan_path),
                    ],
                    broker_factory=factory,
                    server_runner=runner,
                    master_executor=executor,
                )

        self.assertEqual(0, code)
        self.assertEqual(1, len(executor_configs))
        merged = json.loads(executor_configs[0].read_text(encoding="utf-8"))
        self.assertEqual({"kind": "grid"}, merged["workload"])
        self.assertEqual("m", merged["model"]["label"])
        self.assertTrue((self.outputs / "gang_worker.json").exists())


class MergedConfigModeTest(unittest.TestCase):
    def _mode_for(self, workload):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "merged.json"
            path.write_text(json.dumps({"workload": workload}), encoding="utf-8")
            return merged_config_mode(path)

    def test_workload_kind_maps_to_the_master_execution_mode(self):
        self.assertEqual("prefill", self._mode_for({"kind": "cache_grid"}))
        self.assertEqual("decode", self._mode_for({"kind": "grid"}))
        self.assertEqual("decode", self._mode_for({"kind": "distribution"}))

    def test_unknown_or_missing_kind_is_a_config_error(self):
        for workload in ({"kind": "turbo"}, {}, None):
            with self.subTest(workload=workload):
                with self.assertRaises(WorkerConfigError):
                    self._mode_for(workload)


class LoadGangPlanTest(unittest.TestCase):
    def test_nonzero_master_shard_index_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "gang_plan.json"
            path.write_text(
                json.dumps(_gang_plan(master_shard_index=1)), encoding="utf-8"
            )
            with self.assertRaises(WorkerConfigError) as caught:
                load_gang_plan(path)
        message = str(caught.exception)
        self.assertIn("master_shard_index", message)
        self.assertIn("0", message)


class _RecordingRunnerModule:
    """Fake ``local_server_runner`` module: records the rank-group startup."""

    def __init__(self):
        self.rank_contexts = []
        self.starts = []

    def LocalRankGroupContext(self, **kwargs):
        context = SimpleNamespace(**kwargs)
        self.rank_contexts.append(context)
        return context

    def start_local_rank_group(self, rank_context, configs, model_paths):
        self.starts.append((rank_context, configs, model_paths))
        return "handle-sentinel"


def _fake_execution_plan(label="fake-plan"):
    """Minimal execution plan for exercising the engine-launch projection."""
    from rtp_llm.test.perf_test.plan import AOneResourcePlan

    return SimpleNamespace(
        label=label,
        resource_plan=AOneResourcePlan(
            backend="aone",
            resource="daily",
            gpu_type="H20_CU13",
            gpus_per_shard=2,
            shard_count=None,
            max_parallel=2,
            gpus_per_machine=1,
            machine_count=2,
            local_world_size=1,
        ),
        resolved_config={
            "model": {
                "type": "qwen_2",
                "checkpoint": "Qwen/Qwen2.5-0.5B-Instruct",
                "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct",
                "topology": {"tp": 2, "dp": 1, "pp": 1, "ep": 1},
            },
            "engine": {
                "max_seq_len": 4097,
                "max_batch_tokens": 8192,
                "concurrency_limit": 2,
                "args": {},
                "env": {},
            },
        },
    )


class DefaultServerRunnerPlanTest(_WorkerTestCase):
    """The default server runner picks the plan builder by gang mode.

    Both gang modes start their server through ``_default_server_runner``;
    only the plan builder differs. The merged configs mirror the shipped
    gang examples (TP2 across two single-GPU machines), and everything past
    plan building — rank-group startup, server launch — is faked.
    """

    TOPOLOGY = {"tp": 2, "dp": 1, "pp": 1, "ep": 1}

    def _example_gang_plan(self):
        return {
            "schema_version": 1,
            "run_id": "example-gang",
            "config_fingerprint": "config-fp",
            "workload_fingerprint": "workload-fp",
            "world_size": 2,
            "machine_count": 2,
            "local_world_size": 1,
            "gpus_per_machine": 1,
            "gpus_per_host": 8,
            "gpu_type": "H20_CU13",
            "topology": dict(self.TOPOLOGY),
            "master_shard_index": 0,
            "port_range": {"start": 12333, "count": 1},
        }

    def _run_default_server_runner(self, gang_plan, config_path):
        context = build_worker_context(
            gang_plan, tuple(_members(gang_plan["machine_count"])), 0
        )
        runner = _RecordingRunnerModule()
        decode_calls = []
        prefill_calls = []

        def fake_decode_plan(path, **kwargs):
            decode_calls.append((path, kwargs))
            return (_fake_execution_plan("decode-plan"), "decode-source")

        def fake_prefill_plan(path, **kwargs):
            prefill_calls.append((path, kwargs))
            return (_fake_execution_plan("prefill-plan"), "prefill-source")

        setup_args_calls = []

        def fake_setup_args(argv=None):
            setup_args_calls.append(argv)
            return SimpleNamespace(
                parallelism_config=object(), prefill_cp_config=object()
            )

        with patch.dict(
            os.environ,
            {
                "PERF_TEST_CONFIG_PATH": str(config_path),
                "TEST_UNDECLARED_OUTPUTS_DIR": str(self.outputs),
            },
        ):
            with patch(
                "rtp_llm.test.perf_test.aone.gang_worker._load_local_server_runner",
                return_value=runner,
            ), patch(
                "rtp_llm.test.perf_test.aone.gang_worker._resolve_model_paths",
                return_value={
                    "type": "qwen_2",
                    "checkpoint": "/resolved/checkpoint",
                    "tokenizer": "/resolved/tokenizer",
                },
            ), patch(
                "rtp_llm.config.server_config_setup.set_parallelism_config"
            ), patch(
                "rtp_llm.server.server_args.server_args.setup_args",
                side_effect=fake_setup_args,
            ), patch(
                "rtp_llm.test.perf_test.multi_node.decode_runtime"
                ".build_external_decode_plan",
                side_effect=fake_decode_plan,
            ), patch(
                "rtp_llm.test.perf_test.multi_node.prefill_runtime"
                ".build_external_prefill_plan",
                side_effect=fake_prefill_plan,
            ):
                handle = _default_server_runner(context)
                # patch.dict restores os.environ on exit, so snapshot the
                # engine model env the worker exported while still inside.
                self.model_env = {
                    name: os.environ.get(name)
                    for name in ("MODEL_TYPE", "CHECKPOINT_PATH", "TOKENIZER_PATH")
                }
        self.setup_args_calls = setup_args_calls
        return handle, runner, decode_calls, prefill_calls

    def _merged_example(self, mode):
        merged = json.loads(EXAMPLES[mode].read_text(encoding="utf-8"))
        merged_path = self.root / f"merged_{mode}.json"
        merged_path.write_text(json.dumps(merged), encoding="utf-8")
        return merged_path

    def test_setup_args_receives_plan_engine_argv_not_worker_flags(self):
        # Regression: setup_args() parsed sys.argv, so the worker's own
        # --resolved-config/--gang-plan flags reached the engine parser and
        # aborted the shard. The engine argv must come from the plan instead.
        plan = self._example_gang_plan()
        merged_path = self._merged_example("decode")

        self._run_default_server_runner(plan, merged_path)

        self.assertEqual(1, len(self.setup_args_calls))
        argv = self.setup_args_calls[0]
        self.assertIsNotNone(argv, "engine argv must be explicit, never sys.argv")
        argv_text = " ".join(argv)
        self.assertIn("--tp_size=2", argv_text)
        for worker_flag in ("--resolved-config", "--resolved-workload", "--gang-plan"):
            self.assertNotIn(worker_flag, argv_text)

    def test_engine_model_environment_is_exported_for_server_subprocess(self):
        # Regression: the engine subprocess reads MODEL_TYPE/CHECKPOINT_PATH/
        # TOKENIZER_PATH from its environment; without them it aborts with
        # "model_type is not set".
        plan = self._example_gang_plan()
        merged_path = self._merged_example("decode")

        self._run_default_server_runner(plan, merged_path)

        self.assertEqual("qwen_2", self.model_env.get("MODEL_TYPE"))
        self.assertEqual("/resolved/checkpoint", self.model_env.get("CHECKPOINT_PATH"))
        self.assertEqual("/resolved/tokenizer", self.model_env.get("TOKENIZER_PATH"))

    def test_cache_grid_workload_builds_the_prefill_rank_group_plan(self):
        plan = self._example_gang_plan()
        merged_path = self._merged_example("prefill")

        handle, runner, decode_calls, prefill_calls = (
            self._run_default_server_runner(plan, merged_path)
        )

        self.assertEqual("handle-sentinel", handle)
        # The prefill gang must build the prefill plan: the decode planner
        # rejects cache_grid workloads outright.
        self.assertEqual([], decode_calls)
        self.assertEqual(1, len(prefill_calls))
        path, kwargs = prefill_calls[0]
        self.assertEqual(str(merged_path), str(path))
        self.assertEqual(str(self.outputs / "run"), kwargs["result_directory"])
        self.assertEqual(self.TOPOLOGY, kwargs["actual_topology"])
        # The prefill plan is what the local rank group starts from.
        self.assertEqual(1, len(runner.rank_contexts))
        self.assertEqual("prefill-plan", runner.rank_contexts[0].plan.label)
        self.assertEqual(1, len(runner.starts))
        rank_context, _configs, model_paths = runner.starts[0]
        self.assertIs(runner.rank_contexts[0], rank_context)
        self.assertEqual("/resolved/tokenizer", model_paths)

    def test_grid_workload_keeps_the_decode_rank_group_plan(self):
        plan = self._example_gang_plan()
        merged_path = self._merged_example("decode")

        handle, runner, decode_calls, prefill_calls = (
            self._run_default_server_runner(plan, merged_path)
        )

        self.assertEqual("handle-sentinel", handle)
        self.assertEqual([], prefill_calls)
        self.assertEqual(1, len(decode_calls))
        path, kwargs = decode_calls[0]
        self.assertEqual(str(merged_path), str(path))
        self.assertEqual(str(self.outputs / "run"), kwargs["result_directory"])
        self.assertEqual(self.TOPOLOGY, kwargs["actual_topology"])
        self.assertEqual(1, len(runner.rank_contexts))
        self.assertEqual("decode-plan", runner.rank_contexts[0].plan.label)


class DefaultMasterExecutorTest(_WorkerTestCase):
    """The default decode master executor wires the gang's profiler seam."""

    def test_profiler_directory_comes_from_the_handle_log_dir(self):
        plan = self._write_plan()
        context = build_worker_context(plan, tuple(_members(2)), 0)
        handle = SimpleNamespace(
            tokenizer_path="/tokenizer",
            log_file_path="/logs/rank0/server.log",
            log_dir="/outputs/TEST_OUTPUT_run-123_0",
        )
        execute_calls = []

        def fake_execute(_plan, **kwargs):
            execute_calls.append(kwargs)
            return SimpleNamespace(state=SimpleNamespace(value="completed"))

        with patch(
            "rtp_llm.test.perf_test.multi_node.decode_runtime"
            ".build_external_decode_plan",
            return_value=(SimpleNamespace(cases=()), "source"),
        ), patch(
            "rtp_llm.test.perf_test.multi_node.decode_runtime.execute_master_decode",
            side_effect=fake_execute,
        ), patch(
            "rtp_llm.test.perf_test.test_util.create_query", return_value={}
        ):
            code = _default_master_executor(
                handle,
                context=context,
                config_path=str(self.resolved_config),
                result_directory=str(self.outputs / "run"),
            )

        self.assertEqual(0, code)
        self.assertEqual(1, len(execute_calls))
        # The engine writes raw chrome traces to TORCH_CUDA_PROFILER_DIR,
        # which start_local_rank_group points at the member's log dir; the
        # profiler must scan there, not the run root.
        self.assertEqual(
            Path("/outputs/TEST_OUTPUT_run-123_0"),
            execute_calls[0]["profiler_directory"],
        )
        self.assertNotEqual(
            Path(self.outputs / "run"), execute_calls[0]["profiler_directory"]
        )


class DefaultPrefillMasterExecutorTest(_WorkerTestCase):
    """The default prefill master executor fails the manifest on errors."""

    def _run_prefill_executor(self, report_error=None):
        plan = self._write_plan(_gang_plan(machine_count=2, local_world_size=1))
        merged = {
            "model": {"checkpoint": "/model", "type": "qwen_2"},
            "workload": {"kind": "cache_grid", "commit_tail_tokens": 64},
        }
        merged_path = self.root / "merged_prefill.json"
        merged_path.write_text(json.dumps(merged), encoding="utf-8")
        context = build_worker_context(plan, tuple(_members(2)), 0)
        handle = SimpleNamespace(tokenizer_path="/tokenizer", log_dir=str(self.outputs))
        run_root = self.outputs / "run"
        report_calls = []

        def fake_initialize(root, **_kwargs):
            Path(root).mkdir(parents=True, exist_ok=True)
            (Path(root) / "manifest.json").write_text(
                json.dumps({"run_id": "run-1", "state": "planned"}),
                encoding="utf-8",
            )

        def fake_report(root, **kwargs):
            report_calls.append((root, kwargs))
            if report_error is not None:
                raise report_error
            return {"report_kind": "formal"}

        def fake_adapter(_factory):
            return lambda _context: SimpleNamespace(
                case_results=(),
                state=SimpleNamespace(value="completed"),
            )

        code, raised = None, None
        with patch(
            "rtp_llm.test.perf_test.multi_node.prefill_runtime"
            ".build_external_prefill_plan",
            return_value=(SimpleNamespace(cases=()), {"model": {}}),
        ), patch(
            "rtp_llm.test.perf_test.run_directory.initialize_run_directory",
            side_effect=fake_initialize,
        ), patch(
            "rtp_llm.test.perf_test.aone.gang_worker._resolve_model_paths",
            return_value="/tokenizer",
        ), patch(
            "rtp_llm.test.perf_test.test_util._load_tokenizer",
            return_value=object(),
        ), patch(
            "rtp_llm.test.perf_test.modes.prefill.PrefillAdapter",
            side_effect=fake_adapter,
        ), patch(
            "rtp_llm.test.perf_test.prefill.finalize.finalize_prefill_run",
        ), patch(
            "rtp_llm.test.perf_test.prefill.report.generate_prefill_report",
            side_effect=fake_report,
        ):
            try:
                code = _default_prefill_master_executor(
                    handle,
                    context=context,
                    config_path=str(merged_path),
                    result_directory=str(run_root),
                )
            except Exception as error:  # the executor re-raises past failures
                raised = error
        return {
            "code": code,
            "raised": raised,
            "run_root": run_root,
            "report_calls": report_calls,
        }

    def _manifest_state(self, run_root):
        return json.loads(
            (run_root / "manifest.json").read_text(encoding="utf-8")
        )["state"]

    def test_report_success_completes_the_manifest(self):
        outcome = self._run_prefill_executor()

        self.assertIsNone(outcome["raised"])
        self.assertEqual(0, outcome["code"])
        self.assertEqual(1, len(outcome["report_calls"]))
        self.assertEqual("completed", self._manifest_state(outcome["run_root"]))

    def test_report_failure_marks_the_manifest_failed_and_propagates(self):
        outcome = self._run_prefill_executor(
            report_error=RuntimeError("chart failed")
        )

        self.assertIsInstance(outcome["raised"], RuntimeError)
        self.assertEqual(1, len(outcome["report_calls"]))
        self.assertEqual("failed", self._manifest_state(outcome["run_root"]))


class MasterPathSelectionTest(_WorkerTestCase):
    """main() picks the master executor from the resolved config's mode."""

    def _main_selecting(self, workload_kind):
        self._write_plan()
        workload = self.root / "resolved_workload.json"
        workload.write_text(json.dumps({"kind": workload_kind}), encoding="utf-8")
        self.resolved_config.write_text(
            json.dumps({"model": {"label": "m"}}), encoding="utf-8"
        )
        runner, _handle = self._server_runner()
        broker = FakeBroker(self.gang_plan_machine_count, members=_members(2))
        self._seed_completed(broker, [1])
        calls = {"decode": 0, "prefill": 0}

        def decode_executor(_handle, **_kwargs):
            calls["decode"] += 1
            return 0

        def prefill_executor(_handle, **_kwargs):
            calls["prefill"] += 1
            return 0

        environment = {
            BROKER_URL_ENV: "http://broker.example",
            BROKER_TOKEN_ENV: "secret-token",
            BROKER_RUN_ID_ENV: "run-1",
            "TEST_UNDECLARED_OUTPUTS_DIR": str(self.outputs),
            "TEST_SHARD_INDEX": "0",
            "TEST_TOTAL_SHARDS": "2",
        }
        with patch.dict(os.environ, environment, clear=True):
            with patch("rtp_llm.test.perf_test.aone.gang_worker._sleep"):
                with patch(
                    "rtp_llm.test.perf_test.aone.gang_worker._default_master_executor",
                    decode_executor,
                ), patch(
                    "rtp_llm.test.perf_test.aone.gang_worker."
                    "_default_prefill_master_executor",
                    prefill_executor,
                ):
                    code = worker_main(
                        [
                            "--resolved-config", str(self.resolved_config),
                            "--resolved-workload", str(workload),
                            "--gang-plan", str(self.gang_plan_path),
                        ],
                        broker_factory=lambda _url, _token, _run_id: broker,
                        server_runner=runner,
                    )
        return code, calls

    def setUp(self):
        super().setUp()
        self.gang_plan_machine_count = 2

    def test_unsupported_workload_kind_is_a_handled_config_error(self):
        cases = (
            ({"kind": "turbo"}, "unsupported gang workload kind: 'turbo'"),
            ({}, "cannot derive the gang mode from the merged config"),
        )
        for workload, detail in cases:
            with self.subTest(workload=workload):
                self._write_plan()
                workload_path = self.root / "resolved_workload.json"
                workload_path.write_text(json.dumps(workload), encoding="utf-8")
                self.resolved_config.write_text(
                    json.dumps({"model": {"label": "m"}}), encoding="utf-8"
                )
                runner, _handle = self._server_runner()
                factory_calls = []

                def factory(_url, _token, _run_id):
                    factory_calls.append(True)
                    raise AssertionError("broker must not be built on setup failure")

                environment = {
                    BROKER_URL_ENV: "http://broker.example",
                    BROKER_TOKEN_ENV: "secret-token",
                    BROKER_RUN_ID_ENV: "run-1",
                    "TEST_UNDECLARED_OUTPUTS_DIR": str(self.outputs),
                    "TEST_SHARD_INDEX": "0",
                    "TEST_TOTAL_SHARDS": "2",
                }
                stderr = io.StringIO()
                with patch.dict(os.environ, environment, clear=True):
                    with contextlib.redirect_stderr(stderr):
                        code = worker_main(
                            [
                                "--resolved-config", str(self.resolved_config),
                                "--resolved-workload", str(workload_path),
                                "--gang-plan", str(self.gang_plan_path),
                            ],
                            broker_factory=factory,
                            server_runner=runner,
                        )

                self.assertEqual(EXIT_CONFIG_ERROR, code)
                self.assertIn("gang worker setup failed:", stderr.getvalue())
                self.assertIn(detail, stderr.getvalue())
                self.assertEqual([], factory_calls)

    def test_cache_grid_workload_selects_the_prefill_master_path(self):
        code, calls = self._main_selecting("cache_grid")

        self.assertEqual(0, code)
        self.assertEqual({"decode": 0, "prefill": 1}, calls)

    def test_grid_workload_selects_the_decode_master_path(self):
        code, calls = self._main_selecting("grid")

        self.assertEqual(0, code)
        self.assertEqual({"decode": 1, "prefill": 0}, calls)

    def test_explicit_master_executor_wins_over_mode_selection(self):
        self._write_plan()
        workload = self.root / "resolved_workload.json"
        workload.write_text(json.dumps({"kind": "cache_grid"}), encoding="utf-8")
        self.resolved_config.write_text(
            json.dumps({"model": {"label": "m"}}), encoding="utf-8"
        )
        runner, _handle = self._server_runner()
        broker = FakeBroker(2, members=_members(2))
        self._seed_completed(broker, [1])
        provided = []

        def executor(_handle, **_kwargs):
            provided.append(True)
            return 0

        environment = {
            BROKER_URL_ENV: "http://broker.example",
            BROKER_TOKEN_ENV: "secret-token",
            BROKER_RUN_ID_ENV: "run-1",
            "TEST_UNDECLARED_OUTPUTS_DIR": str(self.outputs),
            "TEST_SHARD_INDEX": "0",
            "TEST_TOTAL_SHARDS": "2",
        }
        with patch.dict(os.environ, environment, clear=True):
            with patch("rtp_llm.test.perf_test.aone.gang_worker._sleep"):
                with patch(
                    "rtp_llm.test.perf_test.aone.gang_worker._default_master_executor"
                ) as decode_default, patch(
                    "rtp_llm.test.perf_test.aone.gang_worker."
                    "_default_prefill_master_executor"
                ) as prefill_default:
                    code = worker_main(
                        [
                            "--resolved-config", str(self.resolved_config),
                            "--resolved-workload", str(workload),
                            "--gang-plan", str(self.gang_plan_path),
                        ],
                        broker_factory=lambda _url, _token, _run_id: broker,
                        server_runner=runner,
                        master_executor=executor,
                    )

        self.assertEqual(0, code)
        self.assertEqual([True], provided)
        decode_default.assert_not_called()
        prefill_default.assert_not_called()


if __name__ == "__main__":
    unittest.main()
