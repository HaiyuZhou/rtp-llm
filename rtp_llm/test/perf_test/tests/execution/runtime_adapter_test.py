"""Tests for execution runtime_adapter_test."""

import unittest
from dataclasses import replace
from unittest.mock import Mock

from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.execution.runtime import (
    HttpPrefillClient,
    build_engine_launch_spec,
)
from rtp_llm.test.perf_test.execution.scheduler import configure_scheduler
from rtp_llm.test.perf_test.plan import thaw
from rtp_llm.test.perf_test.tests.support.launch import runtime_adapter_plan


class RuntimeAdapterTest(unittest.TestCase):

    def test_launch_topology_is_accepted_by_engine_parallelism_setup(self):
        from rtp_llm.config.server_config_setup import set_parallelism_config
        from rtp_llm.ops import ParallelismConfig

        for topology in (
            dict(tp=1, dp=2),
            dict(tp=2, dp=1),
            dict(tp=2, dp=1, ep=1),
        ):
            with self.subTest(topology=topology):
                spec = build_engine_launch_spec(runtime_adapter_plan(topology=topology))
                options = dict(arg[2:].split("=", 1) for arg in spec.argv)
                world_size = int(options["world_size"])
                for rank in range(world_size):
                    config = ParallelismConfig()
                    for name in (
                        "tp_size",
                        "dp_size",
                        "pp_size",
                        "ep_size",
                        "world_size",
                    ):
                        setattr(config, name, int(options[name]))
                    config.local_world_size = world_size
                    set_parallelism_config(config, world_rank=rank)
                    self.assertEqual(int(options["ep_size"]), config.ep_size)
                    self.assertEqual(rank % config.tp_size, config.tp_rank)
                    self.assertEqual(
                        rank // config.tp_size % config.dp_size, config.dp_rank
                    )
                    self.assertEqual(
                        rank // (config.tp_size * config.dp_size), config.pp_rank
                    )

    def test_launch_spec_is_derived_only_from_resolved_plan(self):
        spec = build_engine_launch_spec(runtime_adapter_plan())
        self.assertEqual((3, 5), spec.device_ids)
        self.assertEqual("false", spec.environment["TOKENIZERS_PARALLELISM"])
        self.assertEqual("2", spec.environment["WORLD_SIZE"])
        self.assertEqual("1", spec.environment["USE_BATCH_DECODE_SCHEDULER"])
        self.assertIn("--model_type=qwen_2", spec.argv)
        self.assertIn("--checkpoint_path=/models/qwen", spec.argv)
        self.assertIn("--tokenizer_path=/models/qwen-tokenizer", spec.argv)
        self.assertIn("--tp_size=2", spec.argv)
        self.assertIn("--ep_size=2", spec.argv)
        self.assertIn("--pp_size=1", spec.argv)
        self.assertIn("--max_batch_tokens_size=8192", spec.argv)
        self.assertIn("--enable_cuda_graph=0", spec.argv)

    def test_launch_spec_preserves_pipeline_topology(self):
        plan = runtime_adapter_plan()
        config = thaw(plan.resolved_config)
        config["model"]["topology"] = {"tp": 1, "dp": 1, "pp": 2, "ep": 1}
        with self.assertRaisesRegex(PlanningError, "topology.pp=1"):
            build_engine_launch_spec(replace(plan, resolved_config=config))

    def test_pipeline_size_cannot_be_overridden_in_engine_args(self):
        plan = runtime_adapter_plan()
        config = thaw(plan.resolved_config)
        config["engine"]["args"]["pp_size"] = 2
        with self.assertRaisesRegex(PlanningError, "pp_size.*duplicates"):
            build_engine_launch_spec(replace(plan, resolved_config=config))

    def test_http_prefill_client_projects_server_timing_and_reuse(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "aux_info": {
                "input_len": 128,
                "output_len": 1,
                "reuse_len": 64,
                "first_token_cost_time": 12.5,
                "wait_time": 1.25,
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
                "ttft_ms": 12.5,
                "wait_time_ms": 1.25,
            },
            result,
        )
        request = post.call_args.kwargs
        self.assertEqual("hello", request["json"]["prompt"])
        self.assertEqual(1, request["json"]["generate_config"]["max_new_tokens"])
        self.assertNotIn("input_ids", request["json"])
        self.assertEqual(30, request["timeout"])
        response.json.return_value = {"status": "ok", "aux_info": {}}
        for seconds in (10, 7200):
            with self.subTest(seconds=seconds):
                client = HttpPrefillClient(
                    "http://127.0.0.1:18000", post=post, timeout_seconds=seconds
                )
                client.configure_scheduler(1)
                self.assertEqual(min(seconds, 300), post.call_args.kwargs["timeout"])
                client.request(
                    phase="warmup",
                    case_id="case-a",
                    request_index=0,
                    run_index=None,
                    prompt="hello",
                    input_ids=[1],
                    output_length=1,
                )
                self.assertEqual(seconds, post.call_args.kwargs["timeout"])
        from rtp_llm.test.perf_test.prefill.runner import PrefillRunner

        self.assertEqual(
            ("request_failed", "deadline exceeded"),
            PrefillRunner._validate(
                {"success": False, "error": "deadline exceeded"},
                input_len=1,
                expected_reuse=0,
            ),
        )


class ExecutionSchedulerTest(unittest.TestCase):

    def test_requires_explicit_success_acknowledgement(self):
        response = Mock()
        response.json.return_value = {"status": "error", "error": "bad mode"}
        post = Mock(return_value=response)
        with self.assertRaisesRegex(RuntimeError, "scheduler rejected"):
            configure_scheduler(1234, batch_size=2, mode="decode", post=post)
        response.raise_for_status.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
