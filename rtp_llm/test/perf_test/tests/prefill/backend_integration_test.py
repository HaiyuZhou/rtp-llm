"""Tests for prefill backend_integration_test."""

import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.execution.engine_session import EngineDiagnostics
from rtp_llm.test.perf_test.execution.local import UnifiedLocalExecutor
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.tests.support.execution import (
    default_executor_Session,
    default_executor_source,
    default_executor_Tokenizer,
)


class PrefillBackendIntegrationTest(unittest.TestCase):

    def test_batch_four_uses_shared_http_runner_across_backends(self):
        self._batch_four_across_backends("http_text")

    def _batch_four_across_backends(self, transport):
        from rtp_llm.test.perf_test.execution.runtime import HttpPrefillClient
        from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases

        for backend in ("local",):
            with self.subTest(
                backend=backend
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_root = root / "run"
                source = default_executor_source(root)
                source["workload"]["transport"] = transport
                source["engine"]["concurrency_limit"] = 4
                source["workload"].update(batch_sizes=[4], compute_steps=[8, 4])
                parsed = parse_config(source)
                plan = build_execution_plan(
                    mode="prefill",
                    action="run",
                    resolved=resolve_config(source),
                    cases=plan_prefill_cases(parsed.workload, parsed.engine),
                    result_directory=str(run_root),
                )
                events = []
                wire_requests = []
                barrier = threading.Barrier(4, timeout=3)
                lock = threading.Lock()
                endpoint = "http://fake-prefill:18000"

                def post(url, *, json, **_kwargs):
                    with lock:
                        events.append((url, json))
                    if url.endswith("/update_scheduler_info"):
                        self.assertIn(
                            json,
                            (
                                {"batch_size": 4, "mode": "prefill"},
                                {"batch_size": 1, "mode": "prefill"},
                            ),
                        )
                        return Mock(status_code=200, json=lambda: {"status": "ok"})
                    self.assertEqual(endpoint, url)
                    is_probe = "cache_probe_" in json["prompt"]
                    if not is_probe:
                        barrier.wait()
                    return Mock(
                        status_code=200,
                        json=lambda: {
                            "aux_info": {
                                "input_len": len(json["prompt"].split()),
                                "output_len": 1,
                                "reuse_len": (
                                    4 if is_probe and "__run_" in json["prompt"] else 0
                                ),
                                "first_token_cost_time": 10.0,
                            }
                        },
                    )

                http_client = HttpPrefillClient(
                    endpoint, post=post, scheduler_attempts=1
                )
                client = http_client
                tokenizer = default_executor_Tokenizer()
                if transport == "dashsc_input_ids":
                    from rtp_llm.dash_sc.proto import predict_v2_pb2
                    from rtp_llm.test.perf_test.prefill.dashsc_transport import (
                        DashScPrefillClient,
                    )

                    class Tokenizer:

                        def __init__(self):
                            self.tokens = {}

                        def encode(self, text):
                            with lock:
                                return [
                                    self.tokens.setdefault(word, len(self.tokens) + 1)
                                    for word in text.split()
                                ]

                    tokenizer = Tokenizer()

                    class Stub:

                        def ModelStreamInfer(self, requests, **_kwargs):
                            (request,) = list(requests)
                            request = predict_v2_pb2.ModelInferRequest.FromString(
                                request.SerializeToString()
                            )
                            wire_requests.append(request)
                            ids = next(
                                (
                                    raw
                                    for (tensor, raw) in zip(
                                        request.inputs, request.raw_input_contents
                                    )
                                    if tensor.name == "input_ids"
                                )
                            )
                            values = struct.unpack("<" + "i" * (len(ids) // 4), ids)
                            with lock:
                                words = {
                                    value: word
                                    for (word, value) in tokenizer.tokens.items()
                                }
                            prompt = " ".join((words[value] for value in values))
                            payload = post(endpoint, json={"prompt": prompt}).json()[
                                "aux_info"
                            ]
                            response = predict_v2_pb2.ModelStreamInferResponse()
                            infer = response.infer_response
                            infer.parameters[
                                "engine_first_token_cost_time_us"
                            ].int64_param = 10000
                            for name, value in (
                                ("prompt_token_num", payload["input_len"]),
                                ("prompt_cached_token_num", payload["reuse_len"]),
                                ("generated_ids", 42),
                            ):
                                output = infer.outputs.add(name=name, datatype="INT32")
                                output.shape.append(1)
                                infer.raw_output_contents.append(
                                    struct.pack("<i", value)
                                )
                            return iter([response])

                    client = DashScPrefillClient(
                        http_client, channel=Mock(), stub=Stub()
                    )
                    client.wait_ready = Mock()
                session = default_executor_Session()
                session.client = client
                session.diagnostics = EngineDiagnostics()
                with patch(
                    "rtp_llm.test.perf_test.test_util._load_tokenizer",
                    return_value=tokenizer,
                ):
                    if backend == "local":
                        initialize_run_directory(
                            run_root,
                            source_config=source,
                            plan=plan,
                            run_id="integration",
                            code_version="test",
                        )
                        outcome = UnifiedLocalExecutor(
                            session_factory=lambda _plan: session,
                            tokenizer_loader=lambda _plan: tokenizer,
                        )(plan)
                        self.assertEqual(RunState.COMPLETED, outcome.state)
                        results = outcome.case_results
                self.assertCountEqual(
                    [case.case_id for case in plan.cases],
                    [result["case_id"] for result in results],
                )
                self.assertGreater(len(results), 1)
                self.assertEqual(len(plan.cases) * 9 + 3, len(events))
                if transport == "dashsc_input_ids":
                    self.assertEqual(len(plan.cases) * 8 + 2, len(wire_requests))
                self.assertEqual({"batch_size": 1, "mode": "prefill"}, events[0][1])
                self.assertTrue(
                    (run_root / "checkpoints" / "prefill_cache_probe.json").is_file()
                )
                events = events[3:]
                for offset in range(0, len(events), 9):
                    self.assertEqual(
                        endpoint + "/update_scheduler_info", events[offset][0]
                    )
                    self.assertEqual(
                        [endpoint] * 8,
                        [url for (url, _) in events[offset + 1 : offset + 9]],
                    )
                for result in results:
                    self.assertEqual("ok", result["status"])
                    self.assertEqual(
                        "concurrent_batch_v1",
                        result["identity"]["measurement_contract"],
                    )
                    records = result["metrics"]["measurements"]
                    self.assertEqual(
                        [0, 1, 2, 3], [record["request_index"] for record in records]
                    )
                    self.assertTrue(
                        all((record["round_elapsed_ms"] > 0 for record in records))
                    )


if __name__ == "__main__":
    unittest.main()
