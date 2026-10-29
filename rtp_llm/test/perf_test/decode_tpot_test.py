import unittest

from rtp_llm.test.perf_test.decode.tpot import DecodeCapacityError, search_target_tpot


class DecodeTpotSearchTest(unittest.TestCase):
    def test_found_boundary_preserves_legacy_attempt_order(self):
        def measure(batch_size):
            return {
                "tpot_ms": float(batch_size),
                "success_rate": 1.0,
                "throughput_tokens_per_second": batch_size * 100.0,
            }

        result = search_target_tpot(measure, target_tpot_ms=10.0, max_batch_size=16)

        self.assertEqual("found", result.outcome)
        self.assertEqual(8, result.best_batch_size)
        self.assertEqual(8.0, result.actual_tpot_ms)
        self.assertEqual([8, 12], [item.batch_size for item in result.observations])
        self.assertEqual(
            [True, False], [item.satisfies_target for item in result.observations]
        )

    def test_capacity_failure_is_recorded_and_searches_lower_batches(self):
        def measure(batch_size):
            if batch_size >= 16:
                raise DecodeCapacityError("out of KV cache")
            return {"tpot_ms": 5.0, "success_rate": 1.0}

        result = search_target_tpot(measure, target_tpot_ms=10.0, max_batch_size=32)

        self.assertEqual("found", result.outcome)
        self.assertEqual(12, result.best_batch_size)
        self.assertEqual(
            "capacity_failed",
            next(item.status for item in result.observations if item.batch_size == 16),
        )

    def test_all_candidates_failing_is_distinct_from_valid_nonmatching_measurements(
        self,
    ):
        def capacity_failure(_batch_size):
            raise DecodeCapacityError("capacity")

        failed = search_target_tpot(
            capacity_failure, target_tpot_ms=10.0, max_batch_size=16
        )

        self.assertEqual("all_candidates_failed", failed.outcome)
        self.assertIsNone(failed.best_batch_size)
        self.assertTrue(failed.observations)
        self.assertEqual(
            {"capacity_failed"}, {item.status for item in failed.observations}
        )

        not_matching = search_target_tpot(
            lambda _batch_size: {"tpot_ms": 20.0, "success_rate": 1.0},
            target_tpot_ms=10.0,
            max_batch_size=16,
        )

        self.assertEqual("no_satisfying_candidate", not_matching.outcome)
        self.assertIsNone(not_matching.best_batch_size)
        self.assertTrue(
            all(item.status == "measured" for item in not_matching.observations)
        )
        self.assertTrue(
            all(not item.satisfies_target for item in not_matching.observations)
        )


if __name__ == "__main__":
    unittest.main()
