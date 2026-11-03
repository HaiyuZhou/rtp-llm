import threading
import unittest

from rtp_llm.test.perf_test.prefill.runner import PrefillRunner


class _Tokenizer:
    def encode(self, text):
        return text.split()


class _Client:
    def __init__(self, responses, batch_size=1):
        self.responses = {}
        seed_index = measure_index = 0
        for response in responses:
            if response["input_len"] <= 12:
                key = ("seed", None, seed_index)
                seed_index += 1
            else:
                key = (
                    "measure",
                    measure_index // batch_size,
                    measure_index % batch_size,
                )
                measure_index += 1
            self.responses[key] = response
        self.requests = []
        self.scheduler_calls = []

    def configure_scheduler(self, batch_size, mode="prefill"):
        assert mode == "prefill"
        self.scheduler_calls.append(batch_size)

    def request(self, **request):
        self.requests.append(request)
        return self.responses[
            (request["phase"], request["run_index"], request["request_index"])
        ]


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
    def test_batch_four_measure_requests_overlap(self):
        barrier = threading.Barrier(4, timeout=2)

        class Client(_Client):
            def request(self, **request):
                if request["phase"] == "measure":
                    barrier.wait()
                return super().request(**request)

        client = Client([_response(12, 0)] * 4 + [_response(16, 8)] * 4, 4)
        result = PrefillRunner(client, _Tokenizer(), commit_tail_tokens=4).run_case(
            _case(batch_size=4, measure_runs=1)
        )
        self.assertEqual("ok", result["status"])
        self.assertEqual([1, 4], client.scheduler_calls)
        self.assertEqual(["seed"] * 4, [r["phase"] for r in client.requests[:4]])

    def test_cold_batch_configures_only_target_scheduler(self):
        client = _Client([_response(16, 0)] * 4, 4)
        result = PrefillRunner(client, _Tokenizer(), commit_tail_tokens=4).run_case(
            _case(cache_len=0, batch_size=4, measure_runs=1)
        )
        self.assertEqual("ok", result["status"])
        self.assertEqual([4], client.scheduler_calls)

    def test_scheduler_failure_prevents_requests_in_that_phase(self):
        for fail_call, expected_requests in ((1, 0), (2, 4)):
            with self.subTest(fail_call=fail_call):

                class Client(_Client):
                    def configure_scheduler(self, batch_size, mode="prefill"):
                        super().configure_scheduler(batch_size, mode)
                        if len(self.scheduler_calls) == fail_call:
                            raise RuntimeError("scheduler not acknowledged")

                client = Client([_response(12, 0)] * 4 + [_response(16, 8)] * 4, 4)
                result = PrefillRunner(
                    client, _Tokenizer(), commit_tail_tokens=4
                ).run_case(_case(batch_size=4, measure_runs=1))
                self.assertEqual("request_failed", result["status"])
                self.assertIsNone(result["metrics"])
                self.assertEqual(expected_requests, len(client.requests))

    def test_seed_failure_stops_before_measurement(self):
        client = _Client([_response(12, 0, success=False)], 4)
        result = PrefillRunner(client, _Tokenizer(), commit_tail_tokens=4).run_case(
            _case(batch_size=4, measure_runs=1)
        )
        self.assertEqual("request_failed", result["status"])
        self.assertEqual([1], client.scheduler_calls)
        self.assertEqual(["seed"], [r["phase"] for r in client.requests])

    def test_member_exception_waits_for_all_submitted_peers(self):
        completed = []
        barrier = threading.Barrier(4, timeout=2)

        class Client(_Client):
            def request(self, **request):
                barrier.wait()
                member = request["request_index"]
                if member == 0:
                    raise RuntimeError("member failed")
                completed.append(member)
                return _response(16, 0)

        result = PrefillRunner(Client([]), _Tokenizer(), commit_tail_tokens=4).run_case(
            _case(cache_len=0, batch_size=4, measure_runs=1)
        )
        self.assertEqual("request_failed", result["status"])
        self.assertIsNone(result["metrics"])
        self.assertEqual([1, 2, 3], sorted(completed))

    def test_throughput_uses_complete_round_time_for_every_batch_size(self):
        for batch_size, expected in ((4, 160.0), (1, 40.0)):
            with self.subTest(batch_size=batch_size):
                client = _Client(
                    [_response(12, 0)] * batch_size
                    + [
                        _response(16, 8, latency=10 * (i + 1))
                        for i in range(batch_size)
                    ],
                    batch_size,
                )
                times = iter([0.0, 0.2])
                result = PrefillRunner(
                    client,
                    _Tokenizer(),
                    commit_tail_tokens=4,
                    monotonic=lambda: next(times),
                ).run_case(_case(batch_size=batch_size, measure_runs=1))
                metrics = result["metrics"]
                self.assertEqual(expected, metrics["throughput_tokens_per_second"])
                self.assertEqual(
                    25.0 if batch_size == 4 else 10.0, metrics["latency_ms"]
                )
                for member, evidence in enumerate(metrics["measurements"]):
                    self.assertEqual(member, evidence["request_index"])
                    self.assertEqual(200.0, evidence["round_elapsed_ms"])
                    self.assertEqual(
                        "concurrent_batch_v1", evidence["measurement_contract"]
                    )

    def test_throughput_is_mean_of_round_rates(self):
        times = iter([0.0, 0.2, 1.0, 1.5])
        client = _Client([_response(12, 0)] + [_response(16, 8)] * 2)
        result = PrefillRunner(
            client,
            _Tokenizer(),
            commit_tail_tokens=4,
            monotonic=lambda: next(times),
        ).run_case(_case())
        self.assertEqual(28.0, result["metrics"]["throughput_tokens_per_second"])

    def test_nonpositive_and_nonfinite_round_times_fail(self):
        for elapsed in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(elapsed=elapsed):
                times = iter([0.0, elapsed])
                result = PrefillRunner(
                    _Client([_response(16, 0)]),
                    _Tokenizer(),
                    commit_tail_tokens=4,
                    monotonic=lambda: next(times),
                ).run_case(_case(cache_len=0, measure_runs=1))
                self.assertEqual("invalid_timing", result["status"])
                self.assertIsNone(result["metrics"])

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
            (
                [_response(12, 0), _response(16, 8, latency=float("nan"))],
                "invalid_timing",
            ),
            (
                [_response(12, 0), _response(16, 8, latency=float("inf"))],
                "invalid_timing",
            ),
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
        client = _Client(responses, batch_size=2)

        result = PrefillRunner(client, _Tokenizer(), commit_tail_tokens=4).run_case(
            _case(batch_size=2, measure_runs=1)
        )

        self.assertEqual("ok", result["status"])
        self.assertEqual(4, len(client.requests))
        self.assertEqual(2, len(result["metrics"]["measurements"]))


if __name__ == "__main__":
    unittest.main()
