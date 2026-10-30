import unittest
from unittest.mock import Mock

import requests

from rtp_llm.test.perf_test.decode.transport import DecodeHttpTransport


class _Response:
    def __init__(self, payload=None, *, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload


class DecodeHttpTransportTest(unittest.TestCase):
    def test_round_forces_planned_output_after_user_generate_config(self):
        post = Mock(
            return_value=_Response(
                {
                    "aux_info": {
                        "input_len": 128,
                        "output_len": 16,
                        "wait_time": 1.0,
                        "first_token_cost_time": 3.0,
                        "cost_time": 18.0,
                    }
                }
            )
        )
        transport = DecodeHttpTransport(
            endpoint="http://127.0.0.1:18000/",
            prompt_for_length=lambda length: f"prompt-{length}",
            post=post,
        )
        config = {
            "min_new_tokens": 2,
            "max_new_tokens": 2,
            "top_p": 0.8,
            "top_k": 3,
            "force_sp_accept": False,
            "nested": {"value": 1},
        }
        outcomes = transport.run_round(
            input_lengths=(128,), output_length=16, generate_config=config
        )
        request = post.call_args.kwargs["json"]
        self.assertEqual("http://127.0.0.1:18000", post.call_args.args[0])
        self.assertEqual("prompt-128", request["prompt"])
        self.assertEqual(16, request["generate_config"]["min_new_tokens"])
        self.assertEqual(16, request["generate_config"]["max_new_tokens"])
        self.assertEqual(0.8, request["generate_config"]["top_p"])
        self.assertEqual(3, request["generate_config"]["top_k"])
        self.assertTrue(request["generate_config"]["force_sp_accept"])
        request["generate_config"]["nested"]["value"] = 2
        self.assertEqual(1, config["nested"]["value"])
        self.assertEqual(2, config["max_new_tokens"])
        self.assertTrue(outcomes[0]["success"])
        self.assertEqual(
            {"wait": 1.0, "prefill": 2.0, "decode": 15.0, "tpot": 1.0, "total": 17.0},
            outcomes[0]["latency_ms"],
        )

    def test_round_preserves_http_failure_as_request_outcome(self):
        transport = DecodeHttpTransport(
            endpoint="http://127.0.0.1:18000",
            prompt_for_length=lambda _length: "prompt",
            post=Mock(return_value=_Response(status_code=503, text="not ready")),
        )
        outcome = transport.run_round(
            input_lengths=(128,), output_length=8, generate_config={}
        )[0]
        self.assertFalse(outcome["success"])
        self.assertIn("HTTP 503", outcome["error"])
        self.assertEqual(
            (0, 128, 8),
            tuple(
                outcome[key]
                for key in ("request_index", "input_length", "output_length")
            ),
        )

    def test_round_preserves_timeout_as_request_outcome(self):
        transport = DecodeHttpTransport(
            endpoint="http://127.0.0.1:18000",
            prompt_for_length=lambda _length: "prompt",
            post=Mock(side_effect=requests.Timeout("request timed out")),
        )
        outcome = transport.run_round(
            input_lengths=(128,), output_length=8, generate_config={}
        )[0]
        self.assertFalse(outcome["success"])
        self.assertIn("request timed out", outcome["error"])

    def test_round_preserves_input_order_and_single_token_timing(self):
        def post(_endpoint, *, json, **_kwargs):
            length = int(json["prompt"])
            return _Response({"aux_info": {"input_len": length, "output_len": 1}})

        transport = DecodeHttpTransport(
            endpoint="http://127.0.0.1:18000",
            prompt_for_length=str,
            post=post,
        )
        outcomes = transport.run_round(
            input_lengths=(256, 128), output_length=1, generate_config={}
        )
        self.assertEqual([0, 1], [item["request_index"] for item in outcomes])
        self.assertEqual([256, 128], [item["input_length"] for item in outcomes])
        self.assertEqual([0.0, 0.0], [item["latency_ms"]["tpot"] for item in outcomes])

    def test_scheduler_retries_then_succeeds(self):
        post = Mock(
            side_effect=[
                _Response(status_code=503, text="warming"),
                _Response({"status": "ok"}),
            ]
        )
        sleep = Mock()
        transport = DecodeHttpTransport(
            endpoint="http://127.0.0.1:18000",
            prompt_for_length=lambda _length: "prompt",
            post=post,
            sleep=sleep,
            scheduler_attempts=2,
        )
        transport.configure_scheduler(
            (
                {"local_batch_size": 2},
                {"local_batch_size": 1},
                {"local_batch_size": 0},
            )
        )
        self.assertEqual(2, post.call_count)
        sleep.assert_called_once_with(3)
        self.assertEqual(
            "http://127.0.0.1:18000/update_scheduler_info", post.call_args.args[0]
        )
        self.assertEqual(
            {"batch_size": 2, "mode": "decode"}, post.call_args.kwargs["json"]
        )

    def test_scheduler_reports_final_error_after_retry_exhaustion(self):
        post = Mock(side_effect=requests.Timeout("scheduler timed out"))
        sleep = Mock()
        transport = DecodeHttpTransport(
            endpoint="http://127.0.0.1:18000",
            prompt_for_length=str,
            post=post,
            sleep=sleep,
            scheduler_attempts=2,
        )
        with self.assertRaisesRegex(
            RuntimeError, "after 2 attempts: scheduler timed out"
        ):
            transport.configure_scheduler(({"local_batch_size": 1},))
        self.assertEqual(2, post.call_count)
        sleep.assert_called_once_with(3)

    def test_profile_posts_requested_trace_settings(self):
        post = Mock(return_value=_Response({"status": "ok"}))
        transport = DecodeHttpTransport(
            endpoint="http://127.0.0.1:18000/", prompt_for_length=str, post=post
        )
        transport.start_profile(
            trace_name="decode-i128-o16-b2", steps=5, enable_all_ranks=False
        )
        post.assert_called_once_with(
            "http://127.0.0.1:18000/start_profile",
            json={
                "gen_timeline": True,
                "trace_name": "decode-i128-o16-b2",
                "start_step": 0,
                "num_steps": 5,
                "enable_all_rank": False,
            },
            timeout=60,
        )

    def test_profile_rejects_http_and_non_ok_responses(self):
        for response in (
            _Response(status_code=503, text="unavailable"),
            _Response({"status": "error"}),
        ):
            with self.subTest(status=response.status_code):
                transport = DecodeHttpTransport(
                    endpoint="http://127.0.0.1:18000",
                    prompt_for_length=str,
                    post=Mock(return_value=response),
                )
                with self.assertRaisesRegex(
                    RuntimeError, "failed to start decode profile"
                ):
                    transport.start_profile(
                        trace_name="trace", steps=3, enable_all_ranks=True
                    )


if __name__ == "__main__":
    unittest.main()
