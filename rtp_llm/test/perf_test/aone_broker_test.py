"""Loopback HTTP tests for the gang rendezvous broker."""

import io
import json
import socket
import struct
import threading
import time
import unittest
from contextlib import redirect_stderr
from http.client import HTTPConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import urlsplit

from rtp_llm.test.perf_test.aone.broker import (
    HEARTBEAT_EXPIRY_S,
    REGISTER_DEADLINE_S,
    RUN_DEADLINE_S,
    BrokerMember,
    run_gang_broker,
)

_LOOPBACK = "127.0.0.1"
_ALT_LOOPBACK = "127.0.0.2"
_LOOPBACK_HOSTS = {_LOOPBACK: "host-a", _ALT_LOOPBACK: "host-b"}


def _default_resolver(address):
    return _LOOPBACK_HOSTS.get(address)


def _alt_loopback_usable():
    try:
        with socket.socket() as server:
            server.bind((_LOOPBACK, 0))
            server.listen(1)
            with socket.socket() as client:
                client.bind((_ALT_LOOPBACK, 0))
                client.settimeout(2.0)
                client.connect(server.getsockname()[:2])
        return True
    except OSError:
        return False


_ALT_LOOPBACK_USABLE = _alt_loopback_usable()


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _request(
    broker,
    op,
    *,
    method="POST",
    payload=None,
    token=None,
    run_id=None,
    source_address=None,
    timeout=30.0,
):
    """Issue one HTTP request against the live broker socket.

    ``token`` semantics: ``None`` uses the broker token, ``False`` omits the
    authorization header, and any other string is sent verbatim.
    """
    target = urlsplit(broker.url)
    headers = {}
    if token is not False:
        value = broker.token if token is None else token
        headers["Authorization"] = f"Bearer {value}"
    connection = HTTPConnection(
        target.hostname,
        target.port,
        timeout=timeout,
        source_address=(source_address, 0) if source_address else None,
    )
    body = None if payload is None else json.dumps(payload)
    connection.request(
        method,
        f"/v1/runs/{broker.run_id if run_id is None else run_id}/{op}",
        body=body,
        headers=headers,
    )
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    parsed = json.loads(raw.decode("utf-8")) if raw else {}
    return response.status, parsed


def _register(
    broker,
    shard_index,
    hostname,
    port,
    *,
    token=None,
    run_id=None,
    source_address=None,
    config_fingerprint=None,
):
    payload = {"shard_index": shard_index, "hostname": hostname, "port": port}
    if config_fingerprint is not None:
        payload["config_fingerprint"] = config_fingerprint
    return _request(
        broker,
        "register",
        payload=payload,
        token=token,
        run_id=run_id,
        source_address=source_address,
    )


class GangBrokerConstantsTest(unittest.TestCase):
    def test_deadline_constants(self):
        self.assertEqual(3600.0, REGISTER_DEADLINE_S)
        self.assertEqual(43200.0, RUN_DEADLINE_S)
        self.assertEqual(120.0, HEARTBEAT_EXPIRY_S)


class GangBrokerTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state_dir = Path(temporary.name)

    def _start_broker(self, machine_count=1, resolver=None, **kwargs):
        broker = run_gang_broker(
            machine_count,
            state_dir=self.state_dir,
            host=_LOOPBACK,
            resolver=_default_resolver if resolver is None else resolver,
            **kwargs,
        )
        broker.start()
        self.addCleanup(broker.stop)
        return broker

    def test_stop_before_start_is_harmless(self):
        broker = run_gang_broker(1, state_dir=self.state_dir, host=_LOOPBACK)

        broker.stop()

        self.assertFalse(self.state_dir.joinpath("broker_events.jsonl").exists())
        broker.start()
        broker.stop()

    def test_disconnected_members_client_does_not_print_traceback(self):
        broker = self._start_broker(machine_count=2)
        entered = threading.Event()
        original_await = broker.await_members

        def await_members():
            entered.set()
            return original_await()

        broker.await_members = await_members
        target = urlsplit(broker.url)
        client = socket.create_connection((target.hostname, target.port))
        self.addCleanup(client.close)
        client.sendall(
            (
                f"GET /v1/runs/{broker.run_id}/members HTTP/1.1\r\n"
                f"Host: {target.hostname}\r\n"
                f"Authorization: Bearer {broker.token}\r\n\r\n"
            ).encode()
        )
        self.assertTrue(entered.wait(5.0))
        error_handled = threading.Event()
        original_handle_error = broker._server.handle_error

        def handle_error(request, client_address):
            try:
                original_handle_error(request, client_address)
            finally:
                error_handled.set()

        with patch.object(broker._server, "handle_error", side_effect=handle_error):
            with redirect_stderr(io.StringIO()) as stderr:
                client.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                )
                client.close()
                broker.cleanup()
                self.assertTrue(error_handled.wait(5.0))
            self.assertEqual("", stderr.getvalue())

    def test_register_records_source_address(self):
        broker = self._start_broker(machine_count=1)

        status, body = _register(broker, 0, "host-a", 9000)

        self.assertEqual(200, status)
        self.assertEqual("registered", body["status"])
        self.assertEqual(1, body["machine_count"])
        self.assertEqual((BrokerMember(0, _LOOPBACK, "host-a", 9000),), broker.members)

        status, body = _request(broker, "members", method="GET")

        self.assertEqual(200, status)
        self.assertEqual("ready", body["gang_state"])
        self.assertEqual(
            [
                {
                    "shard_index": 0,
                    "address": _LOOPBACK,
                    "hostname": "host-a",
                    "port": 9000,
                }
            ],
            body["members"],
        )

    def test_authentication_and_routing_errors(self):
        broker = self._start_broker(machine_count=2)
        operations = [
            (
                "register",
                "POST",
                {"shard_index": 0, "hostname": "host-a", "port": 9000},
            ),
            ("members", "GET", None),
            ("heartbeat", "POST", {"shard_index": 0}),
            ("result", "POST", {"shard_index": 0, "state": "completed"}),
            ("cleanup", "POST", None),
        ]
        for op, method, payload in operations:
            with self.subTest(op=op, auth="missing"):
                status, body = _request(
                    broker, op, method=method, payload=payload, token=False
                )
                self.assertEqual(401, status)
                self.assertEqual({"error": "unauthorized"}, body)
            with self.subTest(op=op, auth="wrong"):
                status, body = _request(
                    broker, op, method=method, payload=payload, token="wrong-token"
                )
                self.assertEqual(401, status)
                self.assertEqual({"error": "unauthorized"}, body)

        status, body = _register(broker, 0, "host-a", 9000, run_id="not-a-real-run")
        self.assertEqual(404, status)
        self.assertEqual({"error": "unknown_run"}, body)

        status, body = _request(broker, "explode", payload={})
        self.assertEqual(404, status)
        self.assertEqual({"error": "unknown_operation"}, body)

        status, body = _request(broker, "members", method="POST")
        self.assertEqual(405, status)
        self.assertEqual({"error": "method_not_allowed"}, body)

    def test_duplicate_shard_registration_rejected(self):
        broker = self._start_broker(machine_count=2)
        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])

        status, body = _register(broker, 0, "host-a", 9000)

        self.assertEqual(409, status)
        self.assertEqual("duplicate_shard", body["error"])
        self.assertEqual("registering", broker.gang_state)

    def test_duplicate_host_registration_rejected(self):
        broker = self._start_broker(machine_count=2)
        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])

        status, body = _register(broker, 1, "host-a", 9001)

        self.assertEqual(409, status)
        self.assertEqual("duplicate_host", body["error"])
        self.assertEqual("failed", broker.gang_state)
        self.assertEqual("duplicate_host", broker.failure_reason)

    @unittest.skipUnless(
        _ALT_LOOPBACK_USABLE, "distinct loopback source address unavailable"
    )
    def test_distinct_source_addresses_accepted(self):
        broker = self._start_broker(machine_count=2)

        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])
        status, body = _register(
            broker, 1, "host-b", 9001, source_address=_ALT_LOOPBACK
        )

        self.assertEqual(200, status)
        self.assertEqual("ready", broker.gang_state)
        self.assertEqual(
            (
                BrokerMember(0, _LOOPBACK, "host-a", 9000),
                BrokerMember(1, _ALT_LOOPBACK, "host-b", 9001),
            ),
            broker.members,
        )

    def test_hostname_mismatch_tolerated_and_recorded(self):
        # Containerized workers claim their container name while reverse DNS
        # returns the machine FQDN; the mismatch is legitimate and must not
        # gate registration. Both names are recorded for diagnostics.
        broker = self._start_broker(machine_count=2)

        status, body = _register(broker, 0, "container-name", 9000)

        self.assertEqual(200, status)
        member = broker.members[0]
        self.assertEqual("container-name", member.hostname)
        events = [
            json.loads(line)
            for line in (broker._state_dir / "broker_events.jsonl")
            .read_text()
            .splitlines()
            if "member_registered" in line
        ]
        self.assertEqual(1, len(events))
        self.assertIn("resolved_hostname", events[0])

    @unittest.skipUnless(
        _ALT_LOOPBACK_USABLE, "distinct loopback source address unavailable"
    )
    def test_duplicate_claimed_hostname_from_distinct_addresses_rejected(self):
        # The claimed hostname stays unique per gang: a second member from a
        # different address may not reuse it (same-container co-landing guard).
        broker = self._start_broker(machine_count=2)

        self.assertEqual(200, _register(broker, 0, "shared-name", 9000)[0])
        status, body = _register(
            broker, 1, "shared-name", 9001, source_address=_ALT_LOOPBACK
        )

        self.assertEqual(409, status)
        self.assertEqual("duplicate_host", body["error"])
        self.assertEqual("failed", broker.gang_state)

    def test_config_fingerprint_enforced(self):
        broker = self._start_broker(
            machine_count=2, expected_config_fingerprint="fp-expected"
        )

        status, body = _register(
            broker, 0, "host-a", 9000, config_fingerprint="fp-other"
        )
        self.assertEqual(400, status)
        self.assertEqual("config_fingerprint_mismatch", body["error"])

        status, body = _register(broker, 0, "host-a", 9000)
        self.assertEqual(400, status)
        self.assertEqual("config_fingerprint_mismatch", body["error"])

        status, body = _register(
            broker, 0, "host-a", 9000, config_fingerprint="fp-expected"
        )
        self.assertEqual(200, status)
        self.assertEqual("registering", broker.gang_state)

        plain = self._start_broker(machine_count=1)
        self.assertEqual(200, _register(plain, 0, "host-a", 9000)[0])

    @unittest.skipUnless(
        _ALT_LOOPBACK_USABLE, "distinct loopback source address unavailable"
    )
    def test_members_blocks_until_gang_complete(self):
        broker = self._start_broker(machine_count=2)
        outcome = {}

        def fetch():
            outcome["response"] = _request(broker, "members", method="GET")

        thread = threading.Thread(target=fetch, daemon=True)
        thread.start()
        time.sleep(0.3)
        self.assertIsNone(outcome.get("response"))

        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])
        self.assertEqual(
            200, _register(broker, 1, "host-b", 9001, source_address=_ALT_LOOPBACK)[0]
        )
        thread.join(timeout=10.0)
        self.assertFalse(thread.is_alive())

        status, body = outcome["response"]
        self.assertEqual(200, status)
        self.assertEqual("ready", body["gang_state"])
        self.assertEqual([0, 1], [member["shard_index"] for member in body["members"]])

    def test_register_deadline_fails_members_barrier(self):
        broker = self._start_broker(
            machine_count=2,
            register_deadline_s=0.4,
            run_deadline_s=60.0,
            heartbeat_expiry_s=60.0,
        )
        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])

        started = time.monotonic()
        expected_remaining = max(0.0, broker._start_monotonic + 0.4 - started)
        status, body = _request(broker, "members", method="GET")
        elapsed = time.monotonic() - started

        self.assertEqual(504, status)
        self.assertEqual("register_deadline_exceeded", body["error"])
        self.assertGreaterEqual(elapsed, expected_remaining - 0.05)
        self.assertLess(elapsed, 10.0)
        self.assertEqual("failed", broker.gang_state)
        self.assertEqual((BrokerMember(0, _LOOPBACK, "host-a", 9000),), broker.members)

    def test_heartbeat_expiry_marks_member_lost(self):
        broker = self._start_broker(
            machine_count=1,
            register_deadline_s=60.0,
            run_deadline_s=60.0,
            heartbeat_expiry_s=0.25,
        )
        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])
        self.assertEqual(200, _request(broker, "members", method="GET")[0])
        self.assertEqual(
            200, _request(broker, "heartbeat", payload={"shard_index": 0})[0]
        )

        self.assertTrue(_wait_for(lambda: broker.gang_state == "failed"))
        self.assertEqual("heartbeat_expired", broker.failure_reason)
        state = json.loads((self.state_dir / "broker_state.json").read_text())
        self.assertEqual([0], state["lost_shard_indices"])

    def test_run_deadline_fails_incomplete_gang(self):
        broker = self._start_broker(
            machine_count=2,
            register_deadline_s=60.0,
            run_deadline_s=0.35,
            heartbeat_expiry_s=60.0,
        )
        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])

        self.assertTrue(_wait_for(lambda: broker.gang_state == "failed"))
        self.assertEqual("run_deadline_exceeded", broker.failure_reason)

    @unittest.skipUnless(
        _ALT_LOOPBACK_USABLE, "distinct loopback source address unavailable"
    )
    def test_registration_after_gang_complete_rejected(self):
        broker = self._start_broker(machine_count=2)
        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])
        self.assertEqual(
            200, _register(broker, 1, "host-b", 9001, source_address=_ALT_LOOPBACK)[0]
        )
        self.assertEqual("ready", broker.gang_state)

        status, body = _register(broker, 0, "host-a", 9002)

        self.assertEqual(409, status)
        self.assertEqual("gang_complete", body["error"])

    def test_member_result_completes_gang(self):
        broker = self._start_broker(machine_count=1)
        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])

        status, body = _request(
            broker,
            "result",
            payload={"shard_index": 0, "state": "completed", "detail": "ok"},
        )

        self.assertEqual(200, status)
        self.assertEqual("recorded", body["status"])
        self.assertEqual("completed", broker.gang_state)

        status, body = _request(
            broker, "result", payload={"shard_index": 0, "state": "completed"}
        )
        self.assertEqual(409, status)
        self.assertEqual("duplicate_result", body["error"])

    def test_partial_member_result_is_persisted_immediately(self):
        broker = self._start_broker(machine_count=2)
        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])

        status, _ = _request(
            broker,
            "result",
            payload={"shard_index": 0, "state": "completed", "detail": "ok"},
        )

        self.assertEqual(200, status)
        state = json.loads((self.state_dir / "broker_state.json").read_text())
        self.assertEqual(
            {"0": {"state": "completed", "detail": "ok"}}, state["results"]
        )
        self.assertEqual("registering", state["gang_state"])

    def test_member_failure_fails_gang(self):
        broker = self._start_broker(machine_count=1)
        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])

        status, _ = _request(
            broker,
            "result",
            payload={"shard_index": 0, "state": "failed", "detail": "engine crashed"},
        )

        self.assertEqual(200, status)
        self.assertEqual("failed", broker.gang_state)
        self.assertEqual("member_failed", broker.failure_reason)

    def test_invalid_result_and_unknown_shard_rejected(self):
        broker = self._start_broker(machine_count=1)
        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])

        status, body = _request(
            broker, "result", payload={"shard_index": 0, "state": "weird"}
        )
        self.assertEqual(400, status)
        self.assertEqual("invalid_result", body["error"])

        status, body = _request(broker, "heartbeat", payload={"shard_index": 7})
        self.assertEqual(404, status)
        self.assertEqual("unknown_shard", body["error"])

    def test_cleanup_closes_run(self):
        broker = self._start_broker(machine_count=1)
        self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])

        status, body = _request(broker, "cleanup")
        self.assertEqual(200, status)
        self.assertEqual("cleaned", body["status"])

        status, _ = _request(broker, "cleanup")
        self.assertEqual(200, status)

        status, body = _register(broker, 0, "host-a", 9001)
        self.assertEqual(409, status)
        self.assertEqual("run_closed", body["error"])

    def test_context_manager_starts_and_stops(self):
        with run_gang_broker(
            1,
            state_dir=self.state_dir,
            host=_LOOPBACK,
            resolver=_default_resolver,
        ) as broker:
            self.assertTrue(broker.url.startswith(f"http://{_LOOPBACK}:"))
            self.assertEqual(200, _register(broker, 0, "host-a", 9000)[0])

        with self.assertRaises(OSError):
            _register(broker, 0, "host-a", 9001)

    def test_persisted_diagnostics_are_redacted(self):
        broker = self._start_broker(machine_count=1, expected_config_fingerprint="fp-1")
        self.assertEqual(
            200, _register(broker, 0, "host-a", 9000, config_fingerprint="fp-1")[0]
        )
        self.assertEqual(200, _request(broker, "members", method="GET")[0])
        self.assertEqual(
            200,
            _request(
                broker,
                "result",
                payload={"shard_index": 0, "state": "completed", "detail": "ok"},
            )[0],
        )
        self.assertEqual(200, _request(broker, "cleanup")[0])

        files = {
            path.name: path.read_text(encoding="utf-8")
            for path in sorted(self.state_dir.iterdir())
            if path.is_file()
        }
        self.assertIn("broker_state.json", files)
        self.assertIn("broker_events.jsonl", files)

        state = json.loads(files["broker_state.json"])
        self.assertEqual(broker.run_id, state["run_id"])
        self.assertEqual("completed", state["gang_state"])
        self.assertTrue(state["cleaned"])
        self.assertEqual(
            [
                {
                    "shard_index": 0,
                    "address": _LOOPBACK,
                    "hostname": "host-a",
                    "port": 9000,
                }
            ],
            state["members"],
        )
        self.assertIn("started_at", state)
        self.assertIn("updated_at", state)

        events = [
            json.loads(line)
            for line in files["broker_events.jsonl"].splitlines()
            if line.strip()
        ]
        names = [event["event"] for event in events]
        self.assertIn("broker_started", names)
        self.assertIn("member_registered", names)
        registered = [
            event for event in events if event["event"] == "member_registered"
        ][0]
        self.assertEqual(0, registered["shard_index"])
        self.assertEqual(_LOOPBACK, registered["address"])
        self.assertEqual("host-a", registered["hostname"])
        self.assertEqual(9000, registered["port"])
        for event in events:
            self.assertIn("time", event)

        for name, content in files.items():
            with self.subTest(file=name):
                self.assertNotIn(broker.token, content)
                self.assertNotIn("token", content)


if __name__ == "__main__":
    unittest.main()
