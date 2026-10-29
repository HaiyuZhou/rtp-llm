import unittest

from rtp_llm.test.perf_test.prefill.runner import PrefillRunner


class _Tokenizer:
    def encode(self, text):
        return text.split()


class _Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def request(self, **request):
        self.requests.append(request)
        return self.responses.pop(0)


def _response(input_len, reuse_len, latency=10.0, success=True, output_len=1):
    return {
        "success": success,
        "input_len": input_len,
        "output_len": output_len,
        "reuse_len": reuse_len,
        "latency_ms": latency,
    }


def _case(cache_len=8, batch_size=1, measure_runs=2):
    return {
        "case_id": "prefill-case",
        "input_len": 16,
        "cache_len": cache_len,
        "compute_len": 16 - cache_len,
        "batch_size": batch_size,
        "measure_runs": measure_runs,
        "seed": 7,
        "prefix_policy": "shared_seed" if batch_size == 1 else "independent",
    }


class PrefillRunnerTest(unittest.TestCase):
    def test_valid_cached_case_seeds_then_measures_exact_reuse(self):
        client = _Client(
            [
                _response(12, 0),
                _response(16, 8, latency=10.0),
                _response(16, 8, latency=14.0),
            ]
        )
        runner = PrefillRunner(client, _Tokenizer(), commit_tail_tokens=4)

        result = runner.run_case(_case())

        self.assertEqual("ok", result["status"])
        self.assertEqual(3, len(client.requests))
        self.assertEqual("seed", client.requests[0]["phase"])
        self.assertEqual(
            ["measure", "measure"], [r["phase"] for r in client.requests[1:]]
        )
        self.assertEqual(12.0, result["metrics"]["latency_ms"])
        self.assertEqual(8, result["metrics"]["observed_reuse_length"])
        self.assertEqual(2, len(result["metrics"]["measurements"]))

    def test_each_invalid_condition_is_retained_without_metrics(self):
        invalid = [
            ([_response(12, 0), _response(16, 8, success=False)], "request_failed"),
            ([_response(12, 0), _response(15, 8)], "invalid_shape"),
            ([_response(12, 0), _response(16, 0)], "invalid_reuse"),
            ([_response(12, 0), _response(16, 8, latency=0)], "invalid_timing"),
        ]
        for responses, expected in invalid:
            with self.subTest(expected=expected):
                runner = PrefillRunner(
                    _Client(responses), _Tokenizer(), commit_tail_tokens=4
                )
                result = runner.run_case(_case(measure_runs=1))
                self.assertEqual(expected, result["status"])
                self.assertIsNone(result["metrics"])
                self.assertTrue(result["errors"])

    def test_seed_shape_and_reuse_are_correctness_failures(self):
        cases = [
            (_response(11, 0), "invalid_shape"),
            (_response(12, 4), "invalid_reuse"),
        ]
        for seed_response, expected in cases:
            with self.subTest(expected=expected):
                result = PrefillRunner(
                    _Client([seed_response]), _Tokenizer(), commit_tail_tokens=4
                ).run_case(_case(measure_runs=1))
                self.assertEqual(expected, result["status"])
                self.assertIsNone(result["metrics"])

    def test_cold_case_skips_seed_and_requires_zero_reuse(self):
        client = _Client([_response(16, 0), _response(16, 0)])
        result = PrefillRunner(client, _Tokenizer(), commit_tail_tokens=4).run_case(
            _case(cache_len=0)
        )

        self.assertEqual("ok", result["status"])
        self.assertEqual(["measure", "measure"], [r["phase"] for r in client.requests])

    def test_independent_batch_seeds_and_measures_every_request(self):
        responses = [_response(12, 0), _response(12, 0)]
        responses += [_response(16, 8), _response(16, 8)]
        client = _Client(responses)

        result = PrefillRunner(client, _Tokenizer(), commit_tail_tokens=4).run_case(
            _case(batch_size=2, measure_runs=1)
        )

        self.assertEqual("ok", result["status"])
        self.assertEqual(4, len(client.requests))
        self.assertEqual(2, len(result["metrics"]["measurements"]))


if __name__ == "__main__":
    unittest.main()
