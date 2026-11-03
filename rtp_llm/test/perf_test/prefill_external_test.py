"""External prefill planning and session access against a fake endpoint."""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.errors import (
    CaseExecutionError,
    ConfigurationError,
    PlanningError,
)
from rtp_llm.test.perf_test.execution.engine_session import EngineDiagnostics
from rtp_llm.test.perf_test.execution.runtime import HttpPrefillClient, MagaEngineHandle
from rtp_llm.test.perf_test.modes.base import ModeExecutionContext
from rtp_llm.test.perf_test.modes.prefill import PrefillAdapter
from rtp_llm.test.perf_test.multi_node.prefill_runtime import (
    ExternalPrefillSession,
    build_external_prefill_plan,
)
from rtp_llm.test.perf_test.plan import case_id
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases
from rtp_llm.test.perf_test.prefill.runner import PrefillRunner
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import RunStore

TOPOLOGY = {"tp": 1, "dp": 2, "pp": 1, "ep": 1}
COMMIT_TAIL_TOKENS = 8
CASE_INPUT_LEN = 24
CASE_CACHE_LEN = 16


def _prefill_source(result_root):
    return {
        "schema_version": 1,
        "model": {
            "label": "external-prefill",
            "type": "registered",
            "checkpoint": "/m",
            "tokenizer": "/m",
            "topology": dict(TOPOLOGY),
        },
        "engine": {
            "max_seq_len": 64,
            "max_batch_tokens": 64,
            "concurrency_limit": 1,
        },
        "workload": {
            "kind": "cache_grid",
            "max_input_len": 24,
            "cache_alignment": 8,
            "commit_tail_tokens": COMMIT_TAIL_TOKENS,
            "compute_steps": [8],
            "cache_lengths": [CASE_CACHE_LEN],
            "batch_sizes": [1],
            "measure_runs": 1,
            "seed": 7,
        },
        "execution": {"backend": "local", "devices": [0, 1]},
        "output": {"result_root": str(result_root)},
        "regression": {},
        "publish": {},
    }


def _write_prefill_config(root):
    path = root / "prefill.json"
    path.write_text(json.dumps(_prefill_source(root / "results")), encoding="utf-8")
    return path


class _Tokenizer:
    def encode(self, text):
        return text.split()


def _aux(input_len, reuse_len, latency_ms):
    return {
        "input_len": input_len,
        "output_len": 1,
        "reuse_len": reuse_len,
        "first_token_cost_time": latency_ms,
    }


class _FakeModelServer:
    """Loopback model server that replays scripted aux_info payloads."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(
                    {"path": self.path, "headers": dict(self.headers), "body": body}
                )
                payload = json.dumps({"aux_info": outer._responses.pop(0)}).encode(
                    "utf-8"
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def endpoint(self):
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _run_through_session(plan, session, run_root):
    """Drive the prefill runner factory exactly like the unified executor."""
    (run_root / "results" / "cases").mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    adapter = PrefillAdapter(
        lambda context: PrefillRunner(
            context.session.client,
            _Tokenizer(),
            commit_tail_tokens=COMMIT_TAIL_TOKENS,
        )
    )
    return adapter(
        ModeExecutionContext(plan=plan, session=session, store=RunStore(run_root))
    )


class HttpPrefillClientEndpointTest(unittest.TestCase):
    def test_client_posts_to_the_explicit_endpoint(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"aux_info": _aux(24, 16, 12.5)}
        post = Mock(return_value=response)
        client = HttpPrefillClient(
            "http://10.0.0.8:9000", post=post, timeout_seconds=30
        )

        result = client.request(
            phase="measure",
            case_id="case-a",
            request_index=0,
            run_index=1,
            prompt="hello",
            input_ids=[1, 2],
            output_length=1,
        )

        self.assertEqual("http://10.0.0.8:9000", post.call_args.args[0])
        self.assertEqual(30, post.call_args.kwargs["timeout"])
        self.assertEqual(
            {
                "success": True,
                "input_len": 24,
                "output_len": 1,
                "reuse_len": 16,
                "latency_ms": 12.5,
            },
            result,
        )

    def test_local_engine_handle_keeps_the_loopback_endpoint(self):
        manager = Mock()
        manager.port = 18000

        client = MagaEngineHandle(
            manager,
            checkpoint="/models/qwen",
            model_type="qwen_2",
            tokenizer="/models/tokenizer",
        ).client

        self.assertEqual("http://127.0.0.1:18000", client.endpoint)
        self.assertEqual(18000, client.port)


class ExternalPrefillPlanTest(unittest.TestCase):
    def test_multi_dp_batch_rejected_before_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _prefill_source(root / "results")
            source["engine"]["concurrency_limit"] = 4
            source["engine"]["max_batch_tokens"] = 96
            path = root / "prefill.json"
            for batch_size in (4, 1):
                source["workload"]["batch_sizes"] = [batch_size]
                path.write_text(json.dumps(source))
                if batch_size > 1:
                    with self.assertRaisesRegex(PlanningError, "DP.*batch"):
                        build_external_prefill_plan(path, root / "run", TOPOLOGY)
                else:
                    plan, _source = build_external_prefill_plan(
                        path, root / "run", TOPOLOGY
                    )
                    self.assertTrue(plan.cases)
                self.assertEqual(
                    ["prefill.json"], sorted(item.name for item in root.iterdir())
                )

    def test_matching_topology_freezes_standard_cases_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_prefill_config(root)

            plan, source = build_external_prefill_plan(
                path, str(root / "run"), TOPOLOGY
            )

            parsed = parse_config(json.loads(path.read_text(encoding="utf-8")))
            expected_cases = plan_prefill_cases(parsed.workload, parsed.engine)
            resolved = load_resolved_config(path)
            self.assertEqual(_prefill_source(root / "results"), source)
            self.assertEqual("prefill", plan.mode)
            self.assertEqual("run", plan.action)
            self.assertEqual("local", plan.backend)
            self.assertEqual(
                [case_id("prefill", identity) for identity in expected_cases],
                [planned.case_id for planned in plan.cases],
            )
            self.assertEqual(
                [dict(identity) for identity in expected_cases],
                [dict(planned.identity) for planned in plan.cases],
            )
            identity = dict(plan.cases[0].identity)
            self.assertEqual(CASE_INPUT_LEN, identity["input_len"])
            self.assertEqual(CASE_CACHE_LEN, identity["cache_len"])
            self.assertEqual(CASE_INPUT_LEN - CASE_CACHE_LEN, identity["compute_len"])
            self.assertEqual(resolved.config_fingerprint, plan.config_fingerprint)
            self.assertEqual(resolved.workload_fingerprint, plan.workload_fingerprint)
            self.assertFalse((root / "run").exists())

    def test_topology_mismatch_raises_before_any_filesystem_artifact(self):
        cases = (
            (
                {"tp": 1, "dp": 1, "pp": 1, "ep": 1},
                "does not match configured topology",
            ),
            ({"tp": 1, "pp": 1, "ep": 1}, "must supply positive integer"),
            ({"tp": 1, "dp": 0, "pp": 1, "ep": 1}, "must supply positive integer"),
        )
        for actual, pattern in cases:
            with self.subTest(actual=actual):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    path = _write_prefill_config(root)

                    with self.assertRaisesRegex(PlanningError, pattern):
                        build_external_prefill_plan(path, str(root / "run"), actual)

                    self.assertEqual(
                        ["prefill.json"], sorted(item.name for item in root.iterdir())
                    )

    def test_decode_workload_shapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _prefill_source(root / "results")
            source["workload"] = {
                "kind": "grid",
                "input_lengths": [32],
                "output_lengths": [8],
                "batch": {"strategy": "fixed", "sizes": [1]},
                "measure_runs": 1,
                "warmup_runs": 0,
            }
            path = root / "prefill.json"
            path.write_text(json.dumps(source), encoding="utf-8")

            with self.assertRaisesRegex(ConfigurationError, "workload.kind=cache_grid"):
                build_external_prefill_plan(path, str(root / "run"), TOPOLOGY)


class ExternalPrefillSessionTest(unittest.TestCase):
    def test_session_exposes_client_diagnostics_and_topology_only(self):
        session = ExternalPrefillSession("http://10.0.0.8:9000/", topology=TOPOLOGY)

        self.assertIsInstance(session.client, HttpPrefillClient)
        self.assertEqual("http://10.0.0.8:9000", session.client.endpoint)
        self.assertEqual(TOPOLOGY, session.topology)
        self.assertIsInstance(session.diagnostics, EngineDiagnostics)
        self.assertIsNone(session.diagnostics.log_path)
        session.close()
        session.close()
        with session as active:
            self.assertIs(session, active)
        # The handle owns no lifecycle, so the client stays usable after close.
        self.assertEqual("http://10.0.0.8:9000", session.client.endpoint)

    def test_prefill_runner_completes_through_the_external_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_prefill_config(root)
            plan, _source = build_external_prefill_plan(
                path, str(root / "run"), TOPOLOGY
            )

            with _FakeModelServer(
                [
                    _aux(CASE_INPUT_LEN, 0, 5.0),
                    _aux(CASE_INPUT_LEN, CASE_CACHE_LEN, 12.5),
                ]
            ) as server:
                with ExternalPrefillSession(
                    server.endpoint, topology=TOPOLOGY
                ) as session:
                    outcome = _run_through_session(plan, session, root / "run")

            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertIsNone(outcome.failure)
            self.assertEqual(1, len(outcome.case_results))
            result = outcome.case_results[0]
            for field in ("case_id", "identity", "status", "runs", "errors", "metrics"):
                self.assertIn(field, result)
            self.assertEqual("ok", result["status"])
            self.assertEqual(CASE_INPUT_LEN, result["metrics"]["input_length"])
            self.assertEqual(CASE_CACHE_LEN, result["metrics"]["observed_reuse_length"])
            self.assertEqual(12.5, result["metrics"]["latency_ms"])
            self.assertEqual(
                (CASE_INPUT_LEN - CASE_CACHE_LEN) * 1000.0 / 12.5,
                result["metrics"]["throughput_tokens_per_second"],
            )
            persisted = json.loads(
                (
                    root / "run" / "results" / "cases" / f"{result['case_id']}.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual("ok", persisted["status"])

            self.assertEqual(2, len(server.requests))
            seed, measure = server.requests
            self.assertEqual("/", seed["path"])
            self.assertEqual("none", seed["headers"]["X-DashScope-LoggingConsent"])
            self.assertEqual(1, seed["body"]["generate_config"]["max_new_tokens"])
            self.assertNotEqual(seed["body"]["prompt"], measure["body"]["prompt"])

    def test_prefill_runner_flags_invalid_reuse_through_the_external_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_prefill_config(root)
            plan, _source = build_external_prefill_plan(
                path, str(root / "run"), TOPOLOGY
            )

            with _FakeModelServer(
                [
                    _aux(CASE_INPUT_LEN, 0, 5.0),
                    _aux(CASE_INPUT_LEN, 0, 12.5),
                ]
            ) as server:
                session = ExternalPrefillSession(server.endpoint, topology=TOPOLOGY)
                outcome = _run_through_session(plan, session, root / "run")
                session.close()

            self.assertEqual(RunState.PARTIAL, outcome.state)
            self.assertIsInstance(outcome.failure, CaseExecutionError)
            result = outcome.case_results[0]
            self.assertEqual("invalid_reuse", result["status"])
            self.assertIsNone(result["metrics"])
            self.assertEqual("invalid_reuse", result["errors"][0]["code"])
            self.assertIn("expected reuse 16", result["errors"][0]["message"])
            persisted = json.loads(
                (
                    root / "run" / "results" / "cases" / f"{result['case_id']}.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual("invalid_reuse", persisted["status"])


if __name__ == "__main__":
    unittest.main()
