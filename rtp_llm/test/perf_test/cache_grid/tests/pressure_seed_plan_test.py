import unittest

from rtp_llm.test.perf_test.cache_grid.runner.pressure_seed_plan import (
    FORMAL_FILL_TARGET_TOKENS,
    MIN_FORMAL_CLIENT_INFLIGHT,
    MIN_REUSE_TOKENS,
    formal_config_reasons,
)


class FormalConfigTest(unittest.TestCase):
    def test_constants_and_monotonic_reasons(self):
        self.assertEqual(MIN_REUSE_TOKENS, 32768)
        self.assertEqual(FORMAL_FILL_TARGET_TOKENS, 102400)
        self.assertEqual(MIN_FORMAL_CLIENT_INFLIGHT, 16)
        self.assertEqual(
            formal_config_reasons(
                manual_seed_count=None,
                client_inflight=16,
                test_block_num=0,
                fill_target_tokens=102400,
            ),
            (),
        )
        self.assertEqual(
            formal_config_reasons(
                manual_seed_count=2,
                client_inflight=2,
                test_block_num=128,
                fill_target_tokens=16384,
            ),
            (
                "manual_seed_count",
                "client_inflight_below_16",
                "synthetic_test_block_capacity",
                "nonstandard_fill_target",
            ),
        )

    def test_invalid_numeric_inputs_are_rejected(self):
        valid = dict(
            manual_seed_count=None,
            client_inflight=16,
            test_block_num=0,
            fill_target_tokens=102400,
        )
        for name, invalid_values in (
            ("manual_seed_count", [True, False, 0, -1, 1.5]),
            ("client_inflight", [True, False, 0, -1, 1.5]),
            ("test_block_num", [True, False, -1, 1.5]),
            ("fill_target_tokens", [True, False, 0, -1, 1.5]),
        ):
            for value in invalid_values:
                with self.subTest(name=name, value=value), self.assertRaises(
                    ValueError
                ):
                    formal_config_reasons(**(valid | {name: value}))


if __name__ == "__main__":
    unittest.main()
