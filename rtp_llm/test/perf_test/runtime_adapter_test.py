import struct
import unittest
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.execution.runtime import (
    HttpPrefillClient,
    MagaEngineHandle,
    build_engine_launch_spec,
)
from rtp_llm.test.perf_test.plan import build_execution_plan


def _plan(backend="local", transport="http_text"):
    source = {
        "schema_version": 1,
        "model": {
            "label": "model",
            "type": "qwen_2",
            "checkpoint": "/models/qwen",
            "tokenizer": "/models/qwen-tokenizer",
            "topology": {"tp": 2, "dp": 1, "pp": 1, "ep": 2},
        },
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 4,
            "args": {"reuse_cache": 1, "enable_cuda_graph": 0},
            "env": {"TOKENIZERS_PARALLELISM": "false"},
        },
        "workload": {
            "kind": "cache_grid",
            "transport": transport,
            "max_input_len": 1024,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [256],
            "batch_sizes": [1],
            "cache_lengths": [0],
        },
        "execution": (
            {"backend": "local", "devices": [3, 5]}
            if backend == "local"
            else {
                "backend": "aone",
                "resource": "daily",
                "gpu_type": "H20_CU13",
                "max_parallel": 1,
                "shard_count": 1,
            }
        ),
        "output": {"result_root": "/tmp/results"},
        "regression": {},
        "publish": {},
    }
    return build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(source),
        cases=[
            {"input_len": 1024, "cache_len": 0, "compute_len": 1024, "batch_size": 1}
        ],
        result_directory="/tmp/results/run",
    )


class RuntimeAdapterTest(unittest.TestCase):
    def test_aone_launch_spec_uses_action_local_visible_devices(self):
        spec = build_engine_launch_spec(_plan("aone"))

        self.assertEqual((0, 1), spec.device_ids)
        self.assertEqual("2", spec.environment["WORLD_SIZE"])

    def test_launch_spec_is_derived_only_from_resolved_plan(self):
        spec = build_engine_launch_spec(_plan())

        self.assertEqual((3, 5), spec.device_ids)
        self.assertEqual("false", spec.environment["TOKENIZERS_PARALLELISM"])
        self.assertEqual("2", spec.environment["WORLD_SIZE"])
        self.assertEqual("1", spec.environment["USE_BATCH_DECODE_SCHEDULER"])
        self.assertIn("--model_type=qwen_2", spec.argv)
        self.assertIn("--checkpoint_path=/models/qwen", spec.argv)
        self.assertIn("--tokenizer_path=/models/qwen-tokenizer", spec.argv)
        self.assertIn("--tp_size=2", spec.argv)
        self.assertIn("--ep_size=2", spec.argv)
        self.assertNotIn("--pp_size=1", spec.argv)
        self.assertIn("--max_batch_tokens_size=8192", spec.argv)
        self.assertIn("--enable_cuda_graph=0", spec.argv)

    def test_engine_handle_starts_stops_and_reports_liveness(self):
        manager = Mock()
        manager.start_server.return_value = True
        manager.server_pid = 123
        manager.exit_code = None
        manager.port = 18000
        manager.log_file_path = "/tmp/process.log"
        handle = MagaEngineHandle(
            manager,
            checkpoint="/models/qwen",
            model_type="qwen_2",
            tokenizer="/models/tokenizer",
        )

        self.assertTrue(handle.wait_ready(120))
        self.assertTrue(handle.is_alive())
        self.assertIsInstance(handle.client, HttpPrefillClient)
        self.assertTrue(callable(handle.client.configure_scheduler))
        self.assertEqual(18000, handle.client.port)
        self.assertEqual("/tmp/process.log", handle.log_path)
        handle.stop()
        handle.stop()

        manager.start_server.assert_called_once_with(
            model_path="/models/qwen",
            model_type="qwen_2",
            tokenizer_path="/models/tokenizer",
            timeout=120,
        )
        manager.stop_server.assert_called_once_with()

    def test_scheduler_rejects_bad_ack_and_exhausted_retry(self):
        for status, payload, diagnostic in (
            (503, {"status": "ok"}, "HTTP 503"),
            (200, {"error": "scheduler unavailable"}, "scheduler unavailable"),
            (200, {"status": "ok", "error": "rejected"}, "rejected"),
            (200, {}, "acknowledgement"),
            (200, {"status": "pending"}, "acknowledgement"),
        ):
            with self.subTest(status=status, payload=payload):
                response = Mock(status_code=status, text="unavailable")
                response.json.return_value = payload
                post = Mock(return_value=response)
                sleep = Mock()
                client = HttpPrefillClient(
                    "http://10.0.0.8:9000",
                    post=post,
                    sleep=sleep,
                    scheduler_attempts=2,
                )
                with self.assertRaisesRegex(RuntimeError, diagnostic) as raised:
                    client.configure_scheduler(4)
                self.assertIn("2 attempts", str(raised.exception))
                self.assertIn("http://10.0.0.8:9000", str(raised.exception))
                self.assertEqual(2, post.call_count)
                self.assertEqual(1, sleep.call_count)

    def test_scheduler_retries_transport_and_json_failures_until_acknowledged(self):
        bad_json = Mock(status_code=200)
        bad_json.json.side_effect = ValueError("invalid JSON")
        good = Mock(status_code=200)
        good.json.return_value = {"status": "ok"}
        post = Mock(side_effect=[OSError("connection lost"), bad_json, good])
        client = HttpPrefillClient(
            "http://server:9000",
            post=post,
            sleep=lambda _: None,
            scheduler_attempts=3,
        )
        client.configure_scheduler(4)
        self.assertEqual(3, post.call_count)

    def test_scheduler_validates_batch_mode_and_retry_limit(self):
        post = Mock()
        client = HttpPrefillClient("http://server:9000", post=post)
        for batch in (0, -1, True, 1.5, "4"):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                client.configure_scheduler(batch)
        with self.assertRaises(ValueError):
            client.configure_scheduler(4, mode="decode")
        for attempts in (0, -1, True, 1.5):
            with self.subTest(attempts=attempts), self.assertRaises(ValueError):
                HttpPrefillClient("http://server:9000", scheduler_attempts=attempts)
        post.assert_not_called()

    def test_http_prefill_client_projects_server_timing_and_reuse(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "aux_info": {
                "input_len": 128,
                "output_len": 1,
                "reuse_len": 64,
                "first_token_cost_time": 12.5,
            }
        }
        post = Mock(return_value=response)
        client = HttpPrefillClient(
            "http://127.0.0.1:18000", post=post, timeout_seconds=30
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

        self.assertEqual("http://127.0.0.1:18000", post.call_args.args[0])
        self.assertEqual(
            {
                "success": True,
                "input_len": 128,
                "output_len": 1,
                "reuse_len": 64,
                "latency_ms": 12.5,
            },
            result,
        )
        request = post.call_args.kwargs
        self.assertEqual("hello", request["json"]["prompt"])
        self.assertEqual(1, request["json"]["generate_config"]["max_new_tokens"])
        self.assertNotIn("input_ids", request["json"])


class DashScTransportTest(unittest.TestCase):
    def _client(self, responses):
        from rtp_llm.dash_sc.proto import predict_v2_pb2
        from rtp_llm.test.perf_test.prefill.dashsc_transport import DashScPrefillClient

        captured = []

        class Stub:
            def ModelStreamInfer(self, requests, **kwargs):
                # Round-trip protobuf bytes: inspect exactly what goes on the wire.
                captured.extend(
                    predict_v2_pb2.ModelInferRequest.FromString(r.SerializeToString())
                    for r in requests
                )
                return iter(responses)

        channel = Mock()
        http = HttpPrefillClient("http://localhost:18000", post=Mock())
        return (
            DashScPrefillClient(http, stub=Stub(), channel=channel),
            captured,
            channel,
        )

    def _request(self, client):
        return client.request(
            phase="measure",
            case_id="case-a",
            request_index=2,
            run_index=1,
            prompt="must never be tokenized",
            input_ids=[7, 123456, 9],
            output_length=1,
        )

    def test_exact_int32_wire_ids_and_http_response_parity(self):
        from rtp_llm.dash_sc.proto import predict_v2_pb2

        response = predict_v2_pb2.ModelStreamInferResponse()
        infer = response.infer_response
        infer.parameters["engine_first_token_cost_time_us"].int64_param = 12500
        for name, shape, values in (
            ("prompt_token_num", [1], [3]),
            ("prompt_cached_token_num", [1], [2]),
            ("generated_ids", [1, 1], [42]),
        ):
            output = infer.outputs.add(name=name, datatype="INT32")
            output.shape.extend(shape)
            infer.raw_output_contents.append(
                struct.pack("<" + "i" * len(values), *values)
            )
        client, captured, channel = self._client([response])
        self.assertEqual(
            {
                "success": True,
                "input_len": 3,
                "output_len": 1,
                "reuse_len": 2,
                "latency_ms": 12.5,
            },
            self._request(client),
        )
        request = captured[0]
        tensors = {
            tensor.name: (tensor, raw)
            for tensor, raw in zip(request.inputs, request.raw_input_contents)
        }
        tensor, raw = tensors["input_ids"]
        self.assertEqual("INT32", tensor.datatype)
        self.assertEqual([3], list(tensor.shape))
        self.assertEqual((7, 123456, 9), struct.unpack("<3i", raw))
        self.assertEqual((1,), struct.unpack("<i", tensors["max_new_tokens"][1]))
        self.assertEqual((1,), struct.unpack("<i", tensors["min_new_tokens"][1]))
        self.assertTrue(request.parameters["force_sp_accept"].bool_param)
        client.close()
        client.close()
        channel.close.assert_called_once()

    def test_server_error_and_empty_stream_fail_explicitly(self):
        from rtp_llm.dash_sc.proto import predict_v2_pb2

        for responses in (
            [],
            [predict_v2_pb2.ModelStreamInferResponse(error_message="unavailable")],
        ):
            client, _, _ = self._client(responses)
            result = self._request(client)
            self.assertFalse(result["success"])
            self.assertTrue(result["error"])

    def test_scheduler_still_requires_http_acknowledgement(self):
        client, _, _ = self._client([])
        response = Mock(status_code=200)
        response.json.return_value = {"status": "ok"}
        client.http_client._post.return_value = response
        client.configure_scheduler(4)
        self.assertEqual(
            {"batch_size": 4, "mode": "prefill"},
            client.http_client._post.call_args.kwargs["json"],
        )

    def test_missing_grpc_endpoint_fails_readiness(self):
        import grpc

        from rtp_llm.test.perf_test.prefill.dashsc_transport import DashScPrefillClient

        channel = grpc.insecure_channel("127.0.0.1:1")
        client = DashScPrefillClient(
            HttpPrefillClient("http://localhost:18000"), channel=channel, stub=Mock()
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "DashSC.*endpoint"):
                client.wait_ready(0.01)
        finally:
            client.close()

    def test_shared_factory_selects_transport_for_local_and_aone(self):
        from rtp_llm.test.perf_test.execution.runtime import create_maga_handle

        for backend in ("local", "aone"):
            with patch(
                "rtp_llm.test.utils.maga_server_manager.MagaServerManager"
            ) as manager_type, patch(
                "rtp_llm.test.perf_test.prefill.dashsc_transport.DashScPrefillClient"
            ) as client_type:
                manager_type.return_value.port = 18000
                handle = create_maga_handle(_plan(backend, "dashsc_input_ids"))
                self.assertIs(handle.client, client_type.return_value)
                self.assertEqual(18000, client_type.call_args.args[0].port)
                handle.wait_ready(10)
                handle.stop()
                handle.stop()
                client_type.return_value.wait_ready.assert_called_once_with(10)
                client_type.return_value.close.assert_called_once()

    def test_external_session_selects_dashsc_and_releases_only_client(self):
        from rtp_llm.test.perf_test.multi_node.prefill_runtime import (
            ExternalPrefillSession,
        )

        with patch(
            "rtp_llm.test.perf_test.prefill.dashsc_transport.DashScPrefillClient"
        ) as client_type:
            with ExternalPrefillSession(
                "http://10.0.0.8:9000", topology={"tp": 1}, transport="dashsc_input_ids"
            ) as session:
                self.assertIs(session.client, client_type.return_value)
                self.assertEqual(9000, client_type.call_args.args[0].port)
            client_type.return_value.wait_ready.assert_called_once()
            client_type.return_value.close.assert_called_once()

    def test_external_endpoint_preserves_host_and_uses_port_offset(self):
        from rtp_llm.test.perf_test.prefill.dashsc_transport import DashScPrefillClient

        with patch(
            "rtp_llm.dash_sc.client.dash_sc_grpc_client_channel_options",
            return_value=[],
        ), patch("grpc.insecure_channel") as factory:
            client = DashScPrefillClient(
                HttpPrefillClient("http://10.0.0.8:9000"), stub=Mock()
            )
            self.assertEqual("10.0.0.8:9008", factory.call_args.args[0])
            client.close()

    def test_engine_handle_rejects_unknown_transport(self):
        with self.assertRaisesRegex(ValueError, "transport"):
            MagaEngineHandle(
                Mock(port=18000),
                checkpoint="/m",
                model_type="qwen",
                tokenizer="/m",
                transport="typo",
            )


if __name__ == "__main__":
    unittest.main()
