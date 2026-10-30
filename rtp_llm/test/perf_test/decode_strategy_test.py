import unittest
from dataclasses import replace

from rtp_llm.test.perf_test.decode.executor import DecodeCapacityError
from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    ExactBatchPolicy,
    TargetTPOTPolicy,
)
from rtp_llm.test.perf_test.decode.strategy import (
    FixedBatchStrategy,
    TargetTPOTStrategy,
    strategy_for,
)


def _exact_case(batch=4):
    return DecodeCaseSpec(
        workload_kind="grid",
        input_lengths=(128,) * batch,
        output_length=16,
        batch_policy=ExactBatchPolicy(strategy="fixed", size=batch),
        measure_runs=2,
        warmup_runs=0,
    )


def _tpot_case(max_size=16, step=4, target=10.0):
    return replace(
        _exact_case(max_size),
        batch_policy=TargetTPOTPolicy("target_tpot", max_size, step, target),
    )


def _measurement(batch, tpot, *, failed=False):
    return DecodeBatchMeasurement(
        batch_size=batch,
        input_lengths=(128,) * batch,
        worker_allocations=(),
        measurements=tuple(
            {
                "run": run,
                "throughput_tokens_per_second": 100.0 + run,
                "requests": [
                    {
                        "request_index": index,
                        "success": not (failed and run == 1 and index == 0),
                        "input_length": 128,
                        "output_length": 16,
                        "latency_ms": {
                            "wait": 1.0,
                            "prefill": 2.0,
                            "decode": 15.0,
                            "tpot": tpot,
                            "total": 17.0,
                        },
                    }
                    for index in range(batch)
                ],
            }
            for run in range(2)
        ),
    )


class DecodeStrategyTest(unittest.TestCase):
    def test_fixed_strategy_measures_exactly_once(self):
        measured = []
        selected = _measurement(4, 8.0)

        def measure(case, batch):
            measured.append((case, batch))
            return selected

        case = _exact_case()
        outcome = strategy_for(case.batch_policy).execute(case, measure)
        self.assertIsInstance(strategy_for(case.batch_policy), FixedBatchStrategy)
        self.assertEqual([(case, 4)], measured)
        self.assertIs(selected, outcome.selected)
        self.assertIsNone(outcome.search)

    def test_tpot_promotes_cached_best_without_forcing_maximum(self):
        measured = {}

        def measure(_case, batch):
            self.assertNotIn(batch, measured)
            measured[batch] = _measurement(batch, 8.0 if batch <= 8 else 12.0)
            return measured[batch]

        case = _tpot_case()
        outcome = strategy_for(case.batch_policy).execute(case, measure)
        self.assertIsInstance(strategy_for(case.batch_policy), TargetTPOTStrategy)
        self.assertIs(measured[8], outcome.selected)
        self.assertEqual([8, 12], list(measured))
        self.assertEqual("found", outcome.search["outcome"])
        self.assertEqual(8, outcome.search["best_batch_size"])
        self.assertEqual(
            100.5, outcome.search["observations"][0]["throughput_tokens_per_second"]
        )

    def test_tpot_failure_retains_observations_without_formal_measurement(self):
        outcome = TargetTPOTStrategy().execute(
            _tpot_case(max_size=8, target=1.0),
            lambda _case, batch: _measurement(batch, 20.0),
        )
        self.assertIsNone(outcome.selected)
        self.assertEqual("target_tpot_not_met", outcome.error_code)
        self.assertEqual("no_satisfying_candidate", outcome.search["outcome"])
        self.assertEqual(
            [4, 1], [item["batch_size"] for item in outcome.search["observations"]]
        )

    def test_tpot_records_capacity_failure_and_keeps_best_lower_candidate(self):
        def measure(_case, batch):
            if batch >= 12:
                raise DecodeCapacityError("worker token capacity exceeded")
            return _measurement(batch, 8.0)

        outcome = TargetTPOTStrategy().execute(_tpot_case(), measure)
        self.assertEqual(8, outcome.selected.batch_size)
        self.assertEqual(
            "capacity_failed", outcome.search["observations"][1]["outcome"]
        )

    def test_tpot_records_measurement_errors_and_continues_lower(self):
        def measure(_case, batch):
            if batch > 1:
                raise RuntimeError("failed\n" + "x" * 600)
            return _measurement(batch, 8.0)

        outcome = TargetTPOTStrategy().execute(_tpot_case(max_size=8), measure)
        self.assertEqual(1, outcome.selected.batch_size)
        observation = outcome.search["observations"][0]
        self.assertEqual("measurement_failed", observation["outcome"])
        self.assertNotIn("\n", observation["error"])
        self.assertEqual(500, len(observation["error"]))

    def test_tpot_requires_success_in_every_formal_round(self):
        outcome = TargetTPOTStrategy().execute(
            _tpot_case(max_size=1),
            lambda _case, batch: _measurement(batch, 1.0, failed=True),
        )
        self.assertIsNone(outcome.selected)
        self.assertEqual(0.5, outcome.search["observations"][0]["success_rate"])

    def test_candidates_include_single_and_non_multiple_maximum(self):
        for maximum, step, expected in (
            (1, 4, [1]),
            (3, 4, [1, 3]),
            (10, 4, [4, 8, 10]),
        ):
            with self.subTest(maximum=maximum):
                measured = []

                def measure(_case, batch):
                    measured.append(batch)
                    return _measurement(batch, 8.0)

                outcome = TargetTPOTStrategy().execute(
                    _tpot_case(maximum, step), measure
                )
                self.assertEqual(expected, measured)
                self.assertEqual(maximum, outcome.selected.batch_size)

    def test_resolver_rejects_unsupported_policy(self):
        with self.assertRaisesRegex(TypeError, "unsupported decode batch policy"):
            strategy_for(object())


if __name__ == "__main__":
    unittest.main()
