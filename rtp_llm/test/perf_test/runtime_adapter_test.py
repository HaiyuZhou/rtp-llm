import unittest
from unittest.mock import Mock

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.execution.runtime import (
    HttpPrefillClient,
    MagaEngineHandle,
    build_engine_launch_spec,
)
from rtp_llm.test.perf_test.plan import build_execution_plan


def _plan():
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
            "max_input_len": 1024,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [256],
            "batch_sizes": [1],
            "cache_lengths": [0],
        },
        "execution": {"backend": "local", "devices": [3, 5]},
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
        client = HttpPrefillClient(18000, post=post, timeout_seconds=30)

        result = client.request(
            phase="measure",
            case_id="case-a",
            request_index=0,
            run_index=1,
            prompt="hello",
            input_ids=[1, 2],
            output_length=1,
        )

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


if __name__ == "__main__":
    unittest.main()
