import unittest

from rtp_llm.test.perf_test.prefill.prompts import (
    PrefixPromptFactory,
    build_grouped_prompts,
)


class _WhitespaceTokenizer:
    def encode(self, text):
        return text.split()


class PrefillPromptTest(unittest.TestCase):
    def test_cached_and_cold_prompts_have_exact_lengths(self):
        for case_id, total_len, cache_len, run_count in (
            ("cached", 16, 8, 3),
            ("cold", 12, 0, 2),
        ):
            with self.subTest(case_id=case_id):
                prompts = PrefixPromptFactory(
                    _WhitespaceTokenizer()
                ).build_case_prompts(case_id, total_len, cache_len, run_count)
                self.assertEqual(cache_len, len(prompts.prefix_ids))
                self.assertEqual(total_len, prompts.built_len)
                self.assertEqual(
                    [total_len] * run_count,
                    [len(request) for request in prompts.run_ids],
                )

    def test_commit_tail_seed_preserves_exact_prefix(self):
        extracted = PrefixPromptFactory(_WhitespaceTokenizer())
        prompts = extracted.build_case_prompts("seed", 16, 8, 1)

        actual = extracted.make_seed_and_ids(
            "seed", prompts.seed_text, 8, commit_tail_tokens=4
        )
        self.assertEqual(12, len(actual[1]))
        self.assertEqual(prompts.prefix_ids, actual[1][:8])

    def test_grouped_reconstruction_honors_both_prefix_policies(self):
        for policy in ("shared_by_group", "independent"):
            with self.subTest(policy=policy):
                case = {
                    "case_id": 7,
                    "batch_size": 3,
                    "input_len": 16,
                    "cache_len": 8,
                    "prefix_policy": policy,
                    "request_groups": [
                        {"count": 2, "input_len": 16, "cache_len": 8},
                        {"count": 1, "input_len": 12, "cache_len": 4},
                    ],
                }
                members = build_grouped_prompts(
                    PrefixPromptFactory(_WhitespaceTokenizer()), case, run_count=2
                )
                self.assertEqual(3, len(members))
                self.assertEqual([16, 16, 12], [m["input_len"] for m in members])
                first_prefix = members[0]["prompts"].prefix_ids
                second_prefix = members[1]["prompts"].prefix_ids
                if policy == "shared_by_group":
                    self.assertEqual(first_prefix, second_prefix)
                else:
                    self.assertNotEqual(first_prefix, second_prefix)

    def test_invalid_geometry_and_run_count_are_rejected(self):
        factory = PrefixPromptFactory(_WhitespaceTokenizer())
        for args, diagnostic in (
            (("case", 8, 8, 1), "cache_len"),
            (("case", 8, -1, 1), "cache_len"),
            (("case", 8, 0, 0), "run_count"),
        ):
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    factory.build_case_prompts(*args)


if __name__ == "__main__":
    unittest.main()
