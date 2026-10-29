import unittest
from unittest.mock import Mock

from rtp_llm.test.perf_test.decode.http_batch import DecodeHttpBatchRunner


class _Response:
    def __init__(self, payload=None, *, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload


class DecodeHttpBatchRunnerTest(unittest.TestCase):
    def test_configures_local_batch_warms_up_and_emits_old_decode_timings(self):
        responses = [
            _Response({"status": "ok"}),
            _Response(
                {
                    "aux_info": {
                        "input_len": 100,
                        "output_len": 4,
                        "wait_time": 2.0,
                        "first_token_cost_time": 12.0,
                        "cost_time": 42.0,
                    }
                }
            ),
            _Response(
                {
                    "aux_info": {
                        "input_len": 200,
                        "output_len": 4,
                        "wait_time": 3.0,
                        "first_token_cost_time": 15.0,
                        "cost_time": 51.0,
                    }
                }
            ),
            _Response(
                {
                    "aux_info": {
                        "input_len": 100,
                        "output_len": 4,
                        "wait_time": 2.0,
                        "first_token_cost_time": 12.0,
                        "cost_time": 42.0,
                    }
                }
            ),
            _Response(
                {
                    "aux_info": {
                        "input_len": 200,
                        "output_len": 4,
                        "wait_time": 3.0,
                        "first_token_cost_time": 15.0,
                        "cost_time": 51.0,
                    }
                }
            ),
        ]
        post = Mock(side_effect=responses)
        runner = DecodeHttpBatchRunner(
            18000,
            prompt_for_length=lambda length: f"prompt-{length}",
            post=post,
            monotonic=Mock(side_effect=[10.0, 10.024]),
        )

        result = runner(
            case_id="decode-a",
            input_lengths=(100, 200),
            output_length=4,
            global_batch_size=2,
            worker_allocations=[
                {
                    "worker_index": 0,
                    "local_batch_size": 1,
                    "request_indices": [0],
                    "input_lengths": [100],
                },
                {
                    "worker_index": 1,
                    "local_batch_size": 1,
                    "request_indices": [1],
                    "input_lengths": [200],
                },
            ],
            warmup_runs=1,
            measure_runs=1,
            generate_config={"top_p": 0.8},
        )

        self.assertEqual(5, post.call_count)
        scheduler_call = post.call_args_list[0]
        self.assertEqual(
            "http://127.0.0.1:18000/update_scheduler_info",
            scheduler_call.args[0],
        )
        self.assertEqual(
            {"batch_size": 1, "mode": "decode"}, scheduler_call.kwargs["json"]
        )
        self.assertEqual(1, len(result["measurements"]))
        measurement = result["measurements"][0]
        self.assertAlmostEqual(
            1000.0 / 3.0, measurement["throughput_tokens_per_second"]
        )
        self.assertEqual(
            {
                "wait": 2.0,
                "prefill": 10.0,
                "decode": 30.0,
                "tpot": 10.0,
                "total": 40.0,
            },
            measurement["requests"][0]["latency_ms"],
        )
        request_body = post.call_args_list[1].kwargs["json"]
        self.assertEqual("prompt-100", request_body["prompt"])
        self.assertEqual(4, request_body["generate_config"]["max_new_tokens"])
        self.assertEqual(0.8, request_body["generate_config"]["top_p"])

    def test_request_failures_are_preserved_without_aborting_the_batch(self):
        post = Mock(
            side_effect=[
                _Response({"status": "ok"}),
                _Response(status_code=503, text="not ready"),
            ]
        )
        runner = DecodeHttpBatchRunner(
            18000,
            prompt_for_length=lambda _length: "prompt",
            post=post,
            monotonic=Mock(side_effect=[1.0, 1.1]),
        )

        result = runner(
            case_id="decode-b",
            input_lengths=(128,),
            output_length=8,
            global_batch_size=1,
            worker_allocations=[
                {
                    "worker_index": 0,
                    "local_batch_size": 1,
                    "request_indices": [0],
                    "input_lengths": [128],
                }
            ],
            warmup_runs=0,
            measure_runs=1,
            generate_config={},
        )

        request = result["measurements"][0]["requests"][0]
        self.assertFalse(request["success"])
        self.assertIn("HTTP 503", request["error"])
        self.assertEqual(0.0, result["measurements"][0]["throughput_tokens_per_second"])


if __name__ == "__main__":
    unittest.main()
