"""Tests for decode http_transport_test."""

import unittest
from unittest.mock import Mock

from rtp_llm.test.perf_test.decode.transport import DecodeHttpTransport
from rtp_llm.test.perf_test.tests.support.launch import decode_transport_Response


class DecodeHttpTransportTest(unittest.TestCase):

    def test_round_forces_planned_output_after_user_generate_config(self):
        post = Mock(
            return_value=decode_transport_Response(
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
            post=Mock(
                return_value=decode_transport_Response(
                    status_code=503, text="not ready"
                )
            ),
        )
        outcome = transport.run_round(
            input_lengths=(128,), output_length=8, generate_config={}
        )[0]
        self.assertFalse(outcome["success"])
        self.assertIn("HTTP 503", outcome["error"])
        self.assertEqual(
            (0, 128, 8),
            tuple(
                (
                    outcome[key]
                    for key in ("request_index", "input_length", "output_length")
                )
            ),
        )

    def test_scheduler_retries_http_200_error_body(self):
        post = Mock(
            side_effect=[
                decode_transport_Response(
                    {"error": "Failed on some addresses: worker unavailable"}
                ),
                decode_transport_Response({"status": "ok"}),
            ]
        )
        sleep = Mock()
        transport = DecodeHttpTransport(
            endpoint="http://127.0.0.1:18000",
            prompt_for_length=str,
            post=post,
            sleep=sleep,
            scheduler_attempts=2,
        )
        transport.configure_scheduler(({"local_batch_size": 2},))
        self.assertEqual(2, post.call_count)
        sleep.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main()
