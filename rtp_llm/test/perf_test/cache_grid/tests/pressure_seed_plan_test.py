import unittest
from dataclasses import FrozenInstanceError, asdict, replace

from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
    GroupSample,
    PoolSample,
    PressureOwnership,
    RankSample,
    select_targets,
)
from rtp_llm.test.perf_test.cache_grid.runner.pressure_seed_plan import (
    FORMAL_FILL_TARGET_TOKENS,
    MIN_FORMAL_CLIENT_INFLIGHT,
    MIN_REUSE_TOKENS,
    SeedPlan,
    align_up,
    build_seed_plan,
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


def owner(rank_id=0, *, capacity=127, available=256, block_tokens=512):
    return RankSample(
        rank_id,
        f"instance-{rank_id}",
        1,
        1.0,
        1.0,
        1.1,
        (
            PoolSample(
                "gpu",
                capacity,
                0,
                0,
                0.5,
                0.9,
                pool_name="gpu",
                tier=1,
                block_size_bytes=4096,
            ),
            PoolSample(
                "host",
                4096,
                0,
                0,
                0,
                0,
                pool_name="host",
                tier=2,
                block_size_bytes=4096,
                available=available,
            ),
        ),
        (
            GroupSample(
                0,
                "FULL",
                ("gpu",),
                "host",
                0,
                0,
                ("full",),
                geometry=(block_tokens, 4096, 4096, False),
            ),
        ),
    )


def shared_samples():
    declaration = PressureOwnership(2, 0, (0, 1), 2)
    first = replace(owner(), ownership=declaration)
    # Participant counters can differ. Only owner counters are authoritative.
    peer = replace(
        first,
        rank=1,
        instance="instance-1",
        pools=(
            first.pools[0],
            replace(first.pools[1], available=0),
        ),
    )
    return first, peer


class SeedPlannerTest(unittest.TestCase):
    def plan(self, samples=None, targets=None, **overrides):
        samples = samples if samples is not None else shared_samples()
        targets = (
            targets
            if targets is not None
            else select_targets(samples, tuple(sample.rank for sample in samples))
        )
        settings = dict(
            cases=(dict(input_len=36864, cache_len=32768, batch_size=1),),
            cache_alignment=4096,
            commit_tail_tokens=4096,
            client_inflight=16,
            manual_seed_count=None,
            fill_target_tokens=102400,
            test_block_num=0,
        )
        return build_seed_plan(samples, targets, **(settings | overrides))

    def test_shared_owner_capacity_is_counted_once_and_auditable(self):
        plan = self.plan()
        self.assertIsInstance(plan, SeedPlan)
        self.assertEqual(plan.reusable_tokens, 32768)
        self.assertEqual(plan.seed_input_tokens, 36864)
        self.assertEqual(len(plan.groups), 1)
        group = plan.groups[0]
        self.assertEqual(group.seed_blocks, 64)
        self.assertEqual(group.gpu_family_capacity, 1)
        self.assertEqual(group.gpu_low_families, 0)
        self.assertEqual(group.host_families, 4)
        self.assertEqual(group.minimum_families, 2)
        self.assertEqual(group.maximum_families, 4)
        self.assertEqual(plan.seed_count, 2)
        self.assertEqual(plan.source, "automatic")
        self.assertTrue(plan.formal_config_valid)
        evidence = asdict(plan)
        self.assertEqual(evidence["cache_alignment"], 4096)
        self.assertEqual(evidence["test_block_num"], 0)
        self.assertEqual(
            evidence["groups"][0]["participants"],
            ((0, "instance-0"), (1, "instance-1")),
        )
        pool = evidence["groups"][0]["device_pools"][0]
        self.assertEqual(pool["capacity_blocks"], 127)  # already allocatable
        self.assertEqual(pool["low_blocks"], 63)
        self.assertEqual(pool["block_tokens"], 512)
        self.assertEqual(pool["low_ratio"], 0.5)
        self.assertEqual(pool["high_ratio"], 0.9)
        self.assertEqual(pool["geometry_source"], "owner_group")
        with self.assertRaises(FrozenInstanceError):
            plan.seed_count = 9

    def test_duplicate_participant_views_and_physical_records_do_not_inflate(self):
        samples = shared_samples()
        samples = (
            replace(samples[0], pools=samples[0].pools + samples[0].pools),
            samples[1],
        )
        # Freeze against the inventory actually presented, including duplicate records.
        targets = select_targets(samples, (0, 1))
        plan = self.plan(samples, targets + targets)
        self.assertEqual(plan.seed_count, 2)
        self.assertEqual(len(plan.groups), 1)
        self.assertEqual(len(plan.groups[0].device_pools), 1)

    def test_zero_live_device_capacity_cannot_qualify_for_formal_pressure(self):
        with self.assertRaisesRegex(
            ValueError, r"owner_rank=0.*owner_group_id=0.*positive.*capacity"
        ):
            self.plan((owner(capacity=0),))

    def test_partial_blocks_round_up_for_both_device_and_host(self):
        plan = self.plan((owner(capacity=100, available=99, block_tokens=1000),))
        self.assertEqual(plan.groups[0].seed_blocks, 33)
        self.assertEqual(plan.groups[0].host_pool.seed_blocks, 33)
        self.assertEqual(plan.groups[0].gpu_family_capacity, 3)
        self.assertEqual(plan.groups[0].host_families, 3)
        self.assertEqual(plan.seed_count, 4)

    def test_distinct_full_groups_on_one_owner_remain_required(self):
        sample = owner(available=2048)
        second = owner(capacity=255, available=1024, block_tokens=1024)
        extra_pools = tuple(
            replace(pool, pool_id=pool.pool_id + "-2") for pool in second.pools
        )
        extra_group = replace(
            second.groups[0],
            group_id=7,
            device_pool_ids=("gpu-2",),
            host_pool_id="host-2",
            group_tags=("second",),
        )
        sample = replace(
            sample,
            pools=sample.pools + extra_pools,
            groups=sample.groups + (extra_group,),
        )
        plan = self.plan((sample,))
        self.assertEqual([g.owner_group_id for g in plan.groups], [0, 7])
        self.assertEqual(plan.seed_count, 8)
        self.assertEqual(plan.global_minimum_families, 8)
        self.assertEqual(plan.global_maximum_families, 32)

    def test_exact_gpu_capacity_needs_one_additional_family(self):
        plan = self.plan((owner(capacity=128),))
        self.assertEqual(plan.groups[0].gpu_family_capacity, 2)
        self.assertEqual(plan.seed_count, 3)

    def test_global_count_uses_largest_minimum_and_each_group_geometry(self):
        first = owner(available=2048)
        second = owner(1, capacity=255, available=1024, block_tokens=1024)
        plan = self.plan((second, first))
        self.assertEqual([g.owner_rank for g in plan.groups], [0, 1])
        self.assertEqual([g.seed_blocks for g in plan.groups], [64, 32])
        self.assertEqual([g.minimum_families for g in plan.groups], [2, 8])
        self.assertEqual([g.host_pool.seed_blocks for g in plan.groups], [64, 32])
        self.assertEqual(plan.seed_count, 8)

    def test_multiple_device_pools_use_bottleneck_capacity_and_low_watermark(self):
        sample = owner(capacity=255)
        extra = replace(
            sample.pools[0], pool_id="gpu-2", pool_name="gpu-2", capacity=128
        )
        sample = replace(
            sample,
            pools=sample.pools + (extra,),
            groups=(replace(sample.groups[0], device_pool_ids=("gpu", "gpu-2")),),
        )
        plan = self.plan((sample,))
        self.assertEqual(plan.groups[0].gpu_family_capacity, 2)
        self.assertEqual(plan.groups[0].gpu_low_families, 1)
        self.assertEqual(plan.seed_count, 3)

    def test_empty_interval_reports_constraining_owner_and_both_bounds(self):
        with self.assertRaisesRegex(
            ValueError, r"owner_rank=0.*owner_group_id=0.*minimum=2.*maximum=0"
        ):
            self.plan((owner(available=63),))
        with self.assertRaisesRegex(ValueError, r"lower=8.*upper=4"):
            self.plan(
                (owner(), owner(1, capacity=255, available=1024, block_tokens=1024))
            )

    def test_manual_count_is_smoke_and_still_cannot_exceed_retention(self):
        plan = self.plan(manual_seed_count=1)
        self.assertEqual(plan.source, "manual")
        self.assertEqual(plan.seed_count, 1)
        self.assertEqual(plan.groups[0].minimum_families, 2)
        self.assertFalse(plan.formal_config_valid)
        self.assertEqual(plan.nonformal_reasons, ("manual_seed_count",))
        with self.assertRaisesRegex(ValueError, r"manual.*5.*upper=4"):
            self.plan(manual_seed_count=5)

    def test_32k_formal_grid_boundary_and_independent_smoke_reasons(self):
        case = dict(input_len=36864, cache_len=32767, batch_size=1)
        with self.assertRaisesRegex(ValueError, "grid.*32768"):
            self.plan(cases=(case,))
        plan = self.plan(
            cases=(case,),
            manual_seed_count=1,
            client_inflight=2,
            test_block_num=128,
            fill_target_tokens=16385,
        )
        self.assertEqual(plan.reusable_tokens, 32768)
        self.assertEqual(plan.fill_input_tokens, 20480)
        self.assertEqual(
            plan.nonformal_reasons,
            (
                "manual_seed_count",
                "client_inflight_below_16",
                "synthetic_test_block_capacity",
                "nonstandard_fill_target",
                "grid_reuse_below_32768",
            ),
        )

    def test_reuse_and_fill_align_up_without_counting_commit_tail(self):
        plan = self.plan(
            cases=(dict(input_len=40000, cache_len=33000, batch_size=1),),
            cache_alignment=8192,
            commit_tail_tokens=512,
        )
        self.assertEqual(plan.grid_reuse_tokens, 33000)
        self.assertEqual(plan.reusable_tokens, 40960)
        self.assertEqual(plan.seed_input_tokens, 41472)
        self.assertEqual(plan.groups[0].seed_blocks, 80)
        self.assertEqual(plan.fill_target_tokens, 102400)
        self.assertEqual(plan.fill_input_tokens, 106496)

    def test_invalid_or_missing_live_evidence_is_rejected(self):
        samples = shared_samples()
        targets = select_targets(samples, (0, 1))
        for bad_samples, bad_targets in (
            ((), targets),
            (samples, ()),
            (samples[:1], targets),
            (
                (
                    samples[0],
                    replace(
                        samples[1],
                        groups=(
                            replace(
                                samples[1].groups[0], geometry=(1024, 4096, 4096, False)
                            ),
                        ),
                    ),
                ),
                targets,
            ),
            ((replace(samples[0], instance="restarted"), samples[1]), targets),
        ):
            with self.subTest(samples=bad_samples), self.assertRaises(ValueError):
                self.plan(bad_samples, bad_targets)
        for tokens in (0, -1):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(
                ValueError, "block_tokens"
            ):
                self.plan((owner(block_tokens=tokens),))

    def test_align_and_planner_input_validation(self):
        self.assertEqual(align_up(32768, 4096), 32768)
        self.assertEqual(align_up(33000, 4096), 36864)
        for value in (0, -1, True, 1.5):
            for args in ((value, 4096), (32768, value)):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    align_up(*args)
        for name, invalid in (
            ("cache_alignment", 0),
            ("commit_tail_tokens", -1),
            ("commit_tail_tokens", True),
            ("cases", ()),
            ("cases", (dict(cache_len=-1),)),
            ("manual_seed_count", True),
            ("manual_seed_count", 0),
        ):
            with self.subTest(name=name, invalid=invalid), self.assertRaises(
                ValueError
            ):
                self.plan(**{name: invalid})


if __name__ == "__main__":
    unittest.main()
