import copy
import dataclasses
import unittest
from types import SimpleNamespace as NS

from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
    GroupSample,
    PoolSample,
    PressureCoverage,
    RankSample,
    decode_pressure_response,
    group_valid,
    raw_pressure_record,
    resolve_pressure_config,
    select_targets,
    validate_inventory,
)


def settings(**changes):
    values = dict(
        seed_timeout_s=30,
        fill_timeout_s=120,
        fill_max_requests=1000,
        fill_no_progress_s=30,
        fill_input_tokens=4096,
        gpu_tolerance_blocks=0,
        activity_timeout_s=30,
        drain_timeout_s=60,
        capture_timeout_s=120,
    )
    values.update(changes)
    return values


def configured(**changes):
    return resolve_pressure_config(settings(**changes))


def rank(rank_id=0, groups=None, pools=None, instance="instance"):
    return RankSample(
        rank_id,
        instance,
        1,
        1.0,
        1.0,
        1.1,
        tuple(
            pools
            or (
                PoolSample("gpu", 100, 90, 0, 0.8, 0.9),
                PoolSample("host", 100, 30, 0, 0.8, 0.9, 25),
            )
        ),
        tuple(
            groups
            if groups is not None
            else (GroupSample(0, "FULL", ("gpu",), "host", 1, 1),)
        ),
    )


class PB(NS):
    def HasField(self, name):
        return name in self.__dict__ and self.__dict__[name] is not None

    def SerializeToString(self):
        return b"complete raw wire evidence including steps/activity"


def wire():
    gpu = PB(
        pool_id="gpu",
        tier=1,
        total_blocks=100,
        free_blocks=0,
        available_blocks=0,
        used_blocks=90,
        pending_release_blocks=0,
        published_host_blocks=0,
    )
    host = PB(
        pool_id="host",
        tier=2,
        total_blocks=100,
        free_blocks=0,
        available_blocks=0,
        used_blocks=30,
        pending_release_blocks=0,
        published_host_blocks=25,
    )
    group = PB(
        group_id=0,
        group_type=1,
        device_pool_ids=["gpu"],
        host_pool_id=PB(value="host"),
        device_watermark=PB(low_ratio=0.8, high_ratio=0.9),
        host_watermark=PB(low_ratio=0.8, high_ratio=0.9),
        watermark_trigger_count=1,
    )
    transfer = dict(
        group_id=0, operation=2, source_tier=1, target_tier=2, published_blocks=7
    )
    return PB(
        schema_version=1,
        requested_ranks=[0],
        ranks=[
            PB(
                world_rank=0,
                service_instance="instance",
                sequence=1,
                sampled_monotonic_ns=1000000000,
                collect_duration_ns=0,
                error="",
                pools=[gpu, host],
                groups=[group],
                transfers=[
                    PB(**transfer, pool_id="gpu", pool_tier=1),
                    PB(**transfer, pool_id="host", pool_tier=2),
                ],
            )
        ],
    )


def shared_wire(t=1, baseline=False):
    message = wire()
    message.requested_ranks = [0, 1]
    message.ranks.append(copy.deepcopy(message.ranks[0]))
    for i, item in enumerate(message.ranks):
        item.world_rank = i
        item.service_instance = f"instance-{i}"
        item.sequence = t + 2
        item.sampled_monotonic_ns = int((t + i * 100) * 1e9)
        item.ownership = PB(
            mode=2,
            owner_world_rank=PB(value=0),
            participant_world_ranks=[0, 1],
            publication_scope=2,
        )
        item.groups[0].group_tags = ["gpu-tag"]
        item.groups[0].block_tokens = 512
        item.groups[0].payload_bytes = item.groups[0].storage_bytes = 4096
        item.groups[0].crc_enabled = False
        for pool in item.pools:
            pool.pool_name = "gpu-tag" if pool.tier == 1 else "host-tag"
            pool.block_size_bytes = 4096
        if baseline or i:
            item.groups[0].watermark_trigger_count = 0
            for transfer in item.transfers:
                transfer.published_blocks = 0
        if i:
            for pool in item.pools:
                pool.used_blocks = pool.published_host_blocks = 0
    return message


class SharedOwnershipTest(unittest.TestCase):
    def test_first_sparse_transfer_endpoint_is_progress_not_reset(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            PressureCoverage,
        )

        initial = shared_wire(0, True)
        initial.ranks[0].transfers = []
        initial.ranks[1].transfers = []
        baseline = decode_pressure_response(initial, 0, 0, (0, 1)).samples
        evaluator = PressureCoverage(baseline, configured())
        for t in range(1, 4):
            current = shared_wire(t)
            current.ranks[1].transfers = []
            evaluator.add(decode_pressure_response(current, t, t, (0, 1)).samples, t)
        self.assertTrue(evaluator.assess(1, 3).valid)
        vanished = shared_wire(4)
        vanished.ranks[0].transfers = []
        vanished.ranks[1].transfers = []
        evaluator.add(decode_pressure_response(vanished, 4, 4, (0, 1)).samples, 4)
        self.assertFalse(evaluator.epoch_valid)
        self.assertIn("counter_reset", evaluator.reasons)

    def test_rejects_missing_scope_owner_and_incompatible_layout(self):
        mutations = [
            lambda m: setattr(m.ranks[0].ownership, "owner_world_rank", None),
            lambda m: setattr(m.ranks[0].ownership, "owner_world_rank", PB(value=9)),
            lambda m: setattr(m.ranks[0].ownership, "publication_scope", 1),
            lambda m: setattr(m.ranks[0].ownership, "participant_world_ranks", [0]),
            lambda m: setattr(m.ranks[1], "ownership", None),
            lambda m: setattr(m.ranks[1].ownership, "mode", 0),
            lambda m: setattr(m.ranks[1].groups[0], "group_tags", ["other"]),
            lambda m: setattr(m.ranks[1].groups[0], "group_type", 2),
            lambda m: setattr(m.ranks[1].groups[0], "block_tokens", 256),
            lambda m: setattr(m.ranks[1].groups[0], "payload_bytes", 2048),
            lambda m: setattr(m.ranks[1].groups[0], "storage_bytes", 8192),
            lambda m: setattr(m.ranks[1].groups[0], "crc_enabled", True),
            lambda m: setattr(m.ranks[1].pools[0], "pool_name", "other"),
            lambda m: setattr(m.ranks[1].pools[0], "block_size_bytes", 8192),
            lambda m: setattr(m.ranks[1].pools[0], "total_blocks", 101),
            lambda m: m.ranks.pop(),
            lambda m: m.ranks.pop(0),
        ]
        for mutate in mutations:
            message = shared_wire()
            mutate(message)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                decode_pressure_response(message, 1, 1, (0, 1))

    def test_partial_page_does_not_hide_observed_restart_layout_or_counter_reset(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            PressureCoverage,
        )

        baseline = decode_pressure_response(shared_wire(0, True), 0, 0, (0, 1)).samples
        current = decode_pressure_response(shared_wire(1), 1, 1, (0, 1)).samples
        mutations = [
            lambda s: dataclasses.replace(s, instance="restarted"),
            lambda s: dataclasses.replace(
                s, pools=(dataclasses.replace(s.pools[0], capacity=101), s.pools[1])
            ),
            lambda s: dataclasses.replace(
                s, groups=(dataclasses.replace(s.groups[0], triggers=0),)
            ),
            lambda s: dataclasses.replace(s, ownership=None),
        ]
        for mutate in mutations:
            evaluator = PressureCoverage(baseline, configured())
            evaluator.add(current, 1)
            observed = mutate(
                dataclasses.replace(
                    current[0],
                    sequence=4,
                    server_s=2,
                    client_send_s=2,
                    client_receive_s=2,
                )
            )
            evaluator.add((observed,), 2)
            self.assertFalse(evaluator.epoch_valid)

    def test_incomplete_page_preserves_counter_reference_without_coverage(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            PressureCoverage,
        )

        baseline = decode_pressure_response(shared_wire(0, True), 0, 0, (0, 1)).samples
        current = decode_pressure_response(shared_wire(1), 1, 1, (0, 1)).samples
        evaluator = PressureCoverage(baseline, configured())
        evaluator.add(current, 1, complete=False)
        self.assertTrue(evaluator.epoch_valid)
        self.assertEqual(evaluator.last_seen[0], current[0])
        later = dataclasses.replace(
            current[0],
            sequence=current[0].sequence + 1,
            groups=(dataclasses.replace(current[0].groups[0], triggers=0),),
        )
        evaluator.add((later,), 2, complete=False)
        self.assertFalse(evaluator.epoch_valid)
        self.assertIn("counter_reset", evaluator.reasons)

    def test_owner_clock_boundary_and_missing_stale_peer_never_gain_coverage(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            PressureCoverage,
        )

        baseline = decode_pressure_response(shared_wire(0, True), 0, 0, (0, 1)).samples
        for failure in ("missing", "stale", "none"):
            evaluator = PressureCoverage(baseline, configured())
            for t in range(1, 5):
                samples = list(
                    decode_pressure_response(shared_wire(t), t, t, (0, 1)).samples
                )
                samples[0] = dataclasses.replace(samples[0], offset_bounds=(-0.2, 0.2))
                if t >= 3 and failure == "missing":
                    samples.pop()
                if t >= 3 and failure == "stale":
                    samples[1] = dataclasses.replace(
                        samples[1], offset_bounds=(-110, -100)
                    )
                evaluator.add(samples, t)
            if failure == "none":
                self.assertEqual(evaluator.intervals(1, 4), [(1.2, 3.8)])
            else:
                self.assertEqual(
                    evaluator.intervals(1, 4),
                    [(1.2, 1.8 if failure == "missing" else 2.0)],
                )


class PressurePredicateTest(unittest.TestCase):
    def test_full_gate_needs_publication_and_effective_occupancy(self):
        cfg = configured()
        pools = {
            "gpu": PoolSample("gpu", 100, 90, 15, 0.8, 0.9),
            "host": PoolSample("host", 100, 30, 0, 0.8, 0.9, 25),
        }
        base = GroupSample(0, "FULL", ("gpu",), "host", 0, 0)
        current = dataclasses.replace(base, triggers=1, gpu_host_published=1)
        self.assertFalse(group_valid(current, pools, base, cfg))
        pools["gpu"] = dataclasses.replace(pools["gpu"], pending_release=0)
        self.assertTrue(group_valid(current, pools, base, cfg))
        self.assertFalse(group_valid(base, pools, base, cfg))
        self.assertEqual((cfg.steady_s, cfg.measure_s, cfg.profile_steps), (3, 60, 20))

    def test_rounding_and_any_device(self):
        pools = {
            "a": PoolSample("a", 101, 79, 0, 0.8, 0.9),
            "b": PoolSample("b", 101, 80, 0, 0.8, 0.9),
            "host": PoolSample("host", 101, 26, 0, 0.8, 0.9, 26),
        }
        base = GroupSample(0, "FULL", ("a", "b"), "host", 0, 0)
        group = dataclasses.replace(base, triggers=1, gpu_host_published=1)
        self.assertTrue(group_valid(group, pools, base, configured()))
        pools["host"] = dataclasses.replace(pools["host"], host_resident=25)
        self.assertFalse(group_valid(group, pools, base, configured()))

    def test_config_validation_and_frozen(self):
        cfg = configured()
        self.assertEqual(
            (
                cfg.host_ratio,
                cfg.sample_period_s,
                cfg.max_sample_age_s,
                cfg.max_sample_gap_s,
                cfg.min_coverage,
                cfg.max_invalid_s,
                cfg.steady_timeout_s,
                cfg.support_s,
            ),
            (0.25, 1, 2, 2, 0.95, 2, 60, 60),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.measure_s = 1
        for key in settings():
            values = settings()
            del values[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                resolve_pressure_config(values)
        for key, value in [
            ("fill_timeout_s", 0),
            ("fill_input_tokens", True),
            ("fill_max_requests", 1.5),
            ("fill_no_progress_s", float("nan")),
            ("activity_timeout_s", float("inf")),
            ("host_ratio", 1.1),
            ("gpu_tolerance_blocks", -1),
            ("sample_period_s", 3),
            ("nsys_duration_s", 0),
        ]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                configured(**{key: value})

    def test_targets_are_pure_all_ranks_and_full_only(self):
        first = rank(
            groups=[
                GroupSample(0, "FULL", ("gpu",), "host", 1, 1),
                GroupSample(1, "SWA", ("gpu",), "host", 0, 0),
            ]
        )
        targets = select_targets((first, rank(1)), (0, 1))
        self.assertEqual(len(targets), 2)
        self.assertEqual(
            select_targets((rank(instance="new"),), (0,))[0].instance, "new"
        )
        with self.assertRaises(ValueError):
            validate_inventory(select_targets((rank(),), (0,)), targets)
        with self.assertRaises(ValueError):
            select_targets((rank(),), (0, 1))
        with self.assertRaises(ValueError):
            select_targets((rank(groups=[]),), (0,))
        with self.assertRaises(ValueError):
            select_targets(
                (rank(pools=[PoolSample("gpu", 100, 90, 0, 0.8, 0.9)]),), (0,)
            )

    def test_invalid_pools_resets_and_mapping(self):
        sample = rank()
        group = sample.groups[0]
        pools = {p.pool_id: p for p in sample.pools}
        for bad in [
            dataclasses.replace(group, triggers=0),
            dataclasses.replace(group, gpu_host_published=0),
            dataclasses.replace(group, host_pool_id="missing"),
        ]:
            with self.assertRaises(ValueError):
                group_valid(bad, pools, group, configured())
        with self.assertRaises(ValueError):
            group_valid(group, pools, group, configured(gpu_tolerance_blocks=81))
        for bad in [
            dataclasses.replace(pools["gpu"], pending_release=91),
            dataclasses.replace(pools["gpu"], low_ratio=0.9),
        ]:
            with self.assertRaises(ValueError):
                select_targets(
                    (dataclasses.replace(sample, pools=(bad, pools["host"])),), (0,)
                )
        with self.assertRaises(ValueError):
            select_targets(
                (
                    dataclasses.replace(
                        sample,
                        pools=sample.pools
                        + (dataclasses.replace(pools["gpu"], used=99),),
                    ),
                ),
                (0,),
            )

    def test_live_free_and_available_blocks_are_projected(self):
        message = wire()
        message.ranks[0].pools[0].total_blocks = 128
        message.ranks[0].pools[0].free_blocks = 17
        message.ranks[0].pools[0].available_blocks = 23
        sample = decode_pressure_response(message, 1.0, 1.1, (0,)).samples[0]
        self.assertEqual(sample.pools[0].capacity, 128)
        self.assertEqual(sample.pools[0].free, 17)
        self.assertEqual(sample.pools[0].available, 23)
        before = select_targets((sample,), (0,))
        message.ranks[0].pools[0].free_blocks = 19
        message.ranks[0].pools[0].available_blocks = 24
        after = decode_pressure_response(message, 1.0, 1.1, (0,)).samples
        validate_inventory(select_targets(after, (0,)), before)

    def test_live_free_and_available_cannot_exceed_capacity(self):
        for field in ("free_blocks", "available_blocks"):
            with self.subTest(field=field):
                message = wire()
                setattr(message.ranks[0].pools[0], field, 101)
                with self.assertRaisesRegex(ValueError, "invalid physical occupancy"):
                    decode_pressure_response(message, 1.0, 1.1, (0,))

    def test_real_wire_endpoint_projection_and_raw_evidence(self):
        message = wire()
        decoded = decode_pressure_response(message, 1.0, 1.1, (0,))
        self.assertEqual(decoded.raw_wire, message.SerializeToString())
        self.assertEqual(decoded.samples[0].groups[0].gpu_host_published, 7)
        self.assertEqual(decoded.samples[0].server_s, 1)
        self.assertEqual(decoded.samples[0].pools[1].host_resident, 25)
        message.ranks[0].pools.append(message.ranks[0].pools[0])
        self.assertEqual(
            len(decode_pressure_response(message, 1.0, 1.1, (0,)).samples[0].pools), 2
        )

    def test_store_publication_qualifies_without_double_counting_source(self):
        message = wire()
        # Stable low-watermark occupancy cannot establish the prior DEVICE trigger.
        message.ranks[0].pools[0].used_blocks = 80
        for transfer in message.ranks[0].transfers:
            transfer.operation = 3
        sample = decode_pressure_response(message, 1.0, 1.1, (0,)).samples[0]
        group = sample.groups[0]
        self.assertEqual(group.gpu_host_published, 7)
        baseline = dataclasses.replace(group, triggers=0, gpu_host_published=0)
        self.assertTrue(
            group_valid(
                group, {p.pool_id: p for p in sample.pools}, baseline, configured()
            )
        )
        self.assertFalse(
            group_valid(
                dataclasses.replace(group, triggers=0),
                {p.pool_id: p for p in sample.pools},
                baseline,
                configured(),
            )
        )
        message.ranks[0].groups[0].watermark_trigger_count = 0
        decoded = decode_pressure_response(message, 1.0, 1.1, (0,)).samples[0]
        self.assertEqual(decoded.groups[0].triggers, 0)
        self.assertFalse(
            group_valid(
                decoded.groups[0],
                {p.pool_id: p for p in decoded.pools},
                baseline,
                configured(),
            )
        )

    def test_low_swa_occupancy_does_not_gate_full(self):
        sample = rank()
        full = sample.groups[0]
        sample = dataclasses.replace(
            sample,
            pools=sample.pools + (PoolSample("cold_swa", 100, 0, 0, 0.8, 0.9),),
            groups=sample.groups + (GroupSample(1, "SWA", ("cold_swa",), "", 0, 0),),
        )
        self.assertEqual(len(select_targets((sample,), (0,))), 1)
        self.assertTrue(
            group_valid(
                full,
                {p.pool_id: p for p in sample.pools},
                dataclasses.replace(full, triggers=0, gpu_host_published=0),
                configured(),
            )
        )

    def test_raw_record_retains_structured_evidence_and_client_times(self):
        import json

        from google.protobuf.struct_pb2 import Struct

        response = Struct()
        response.update({"steps": [{"phase": "steady"}], "activity": {"running": 2}})
        record = raw_pressure_record(response, 1.0, 1.1)
        self.assertEqual(record["response"]["steps"][0]["phase"], "steady")
        self.assertEqual(record["response"]["activity"]["running"], 2)
        self.assertEqual(record["client_receive_s"], 1.1)
        json.dumps(record, allow_nan=False)
        with self.assertRaises(ValueError):
            raw_pressure_record(response, 2.0, 1.0)

    def test_disabled_host_disk_and_observe_only_watermarks_are_preserved(self):
        message = wire()
        current = message.ranks[0]
        full = current.groups[0]
        full.host_watermark = PB(low_ratio=0.0, high_ratio=0.0)
        full.disk_pool_id = PB(value="disk")
        full.disk_watermark = PB(low_ratio=0.0, high_ratio=0.0)
        current.pools.extend(
            [
                PB(
                    pool_id="disk",
                    tier=3,
                    total_blocks=100,
                    free_blocks=0,
                    available_blocks=0,
                    used_blocks=0,
                    pending_release_blocks=0,
                    published_host_blocks=0,
                ),
                PB(
                    pool_id="swa",
                    tier=1,
                    total_blocks=100,
                    free_blocks=0,
                    available_blocks=0,
                    used_blocks=0,
                    pending_release_blocks=0,
                    published_host_blocks=0,
                ),
            ]
        )
        current.groups.append(
            PB(
                group_id=1,
                group_type=2,
                device_pool_ids=["swa"],
                device_watermark=PB(low_ratio=0.0, high_ratio=0.0),
                watermark_trigger_count=0,
            )
        )
        sample = decode_pressure_response(message, 1.0, 1.1, (0,)).samples[0]
        pools = {pool.pool_id: pool for pool in sample.pools}
        for pool_id in ("host", "disk", "swa"):
            self.assertEqual(
                (pools[pool_id].low_ratio, pools[pool_id].high_ratio), (0.0, 0.0)
            )
        self.assertEqual(len(select_targets((sample,), (0,))), 1)
        group = sample.groups[0]
        baseline = dataclasses.replace(group, triggers=0, gpu_host_published=0)
        self.assertTrue(group_valid(group, pools, baseline, configured()))

    def test_disabled_or_invalid_full_device_watermarks_are_rejected(self):
        for low, high in ((0.0, 0.0), (0.0, 0.9), (0.9, 0.8), (0.8, 1.1)):
            message = wire()
            message.ranks[0].groups[0].device_watermark = PB(
                low_ratio=low, high_ratio=high
            )
            with self.subTest(low=low, high=high), self.assertRaises(ValueError):
                decode_pressure_response(message, 1.0, 1.1, (0,))

    def test_wire_rejects_schema_presence_tier_and_unknown_mapping(self):
        mutations = [
            lambda m: setattr(m, "schema_version", 2),
            lambda m: setattr(m.ranks[0].groups[0], "host_pool_id", None),
            lambda m: setattr(m.ranks[0].groups[0], "device_watermark", None),
            lambda m: setattr(m.ranks[0].pools[1], "tier", 1),
            lambda m: setattr(m.ranks[0].transfers[1], "pool_id", "unknown"),
            lambda m: setattr(m.ranks[0].groups[0], "group_type", 0),
            lambda m: setattr(m.ranks[0], "error", "failed"),
        ]
        for mutate in mutations:
            message = wire()
            mutate(message)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                decode_pressure_response(message, 1.0, 1.1, (0,))


def migration_page(t, d2h=0, h2d=0, *, participants=1, bounds=(0, 0), operation=2):
    """Decode cumulative settlement endpoints, including both physical views."""
    message = wire() if participants == 1 else shared_wire(t, True)
    if participants > 2:
        message.ranks.extend(
            copy.deepcopy(message.ranks[1]) for _ in range(participants - 2)
        )
    message.requested_ranks = list(range(participants))
    for i, item in enumerate(message.ranks):
        item.world_rank = i
        item.service_instance = f"instance-{i}"
        item.sequence = t + 2
        item.sampled_monotonic_ns = int((t + i * 100) * 1e9)
        item.groups[0].watermark_trigger_count = 1
        if participants > 1:
            item.ownership.participant_world_ranks = list(range(participants))
        item.transfers = []
        for op, source, target, count in ((operation, 1, 2, d2h), (1, 2, 1, h2d)):
            for pool in item.pools:
                item.transfers.append(
                    PB(
                        group_id=0,
                        pool_id=pool.pool_id,
                        pool_tier=pool.tier,
                        operation=op,
                        source_tier=source,
                        target_tier=target,
                        copied_blocks=count,
                        copied_bytes=count * 4096,
                        published_blocks=count,
                        published_bytes=count * 4096,
                        failed_operations=0,
                    )
                )
    return decode_pressure_response(
        message, t + bounds[0], t + bounds[1], tuple(range(participants))
    ).samples


class MigrationTest(unittest.TestCase):
    def evaluator(self, *, participants=1, max_segments=4096):
        return PressureCoverage(
            migration_page(9, participants=participants), configured(), max_segments
        )

    def test_settled_store_or_evict_and_load_qualify_each_direction(self):
        for operation in (2, 3):
            with self.subTest(operation=operation):
                evaluator = PressureCoverage(
                    (migration_page(9, operation=operation)[0],), configured()
                )
                evaluator.add(migration_page(10, operation=operation), 10)
                evaluator.add(migration_page(11, 7, 5, operation=operation), 11)
                assessment = evaluator.assess_migration(10, 20)
                self.assertTrue(assessment.valid)
                self.assertEqual(assessment.reasons, ())
                self.assertTrue(all(group.valid for group in assessment.groups))
                self.assertEqual(len(assessment.groups), 1)
                group = assessment.groups[0]
                self.assertEqual(group.d2h_intervals, ((10, 11),))
                self.assertEqual(group.h2d_intervals, ((10, 11),))

    def test_d2h_alone_cannot_qualify_migration(self):
        evaluator = self.evaluator()
        evaluator.add(migration_page(10), 10)
        evaluator.add(migration_page(11, 7, 0), 11)
        result = evaluator.assess_migration(10, 20)
        self.assertFalse(result.valid)
        self.assertEqual(result.reasons, ("missing_h2d_load:0:0",))

    def test_h2d_alone_cannot_qualify_migration(self):
        evaluator = self.evaluator()
        evaluator.add(migration_page(10), 10)
        evaluator.add(migration_page(11, 0, 7), 11)
        result = evaluator.assess_migration(10, 20)
        self.assertFalse(result.valid)
        self.assertEqual(result.reasons, ("missing_d2h:0:0",))

    def test_copy_or_bytes_without_publication_cannot_qualify(self):
        evaluator = self.evaluator()
        evaluator.add(migration_page(10), 10)
        current = migration_page(11, 7, 5)[0]
        current = dataclasses.replace(
            current,
            counters=tuple(
                (key, 0 if key[0] == "published_blocks" else value)
                for key, value in current.counters
            ),
        )
        evaluator.add((current,), 11)
        self.assertFalse(evaluator.assess_migration(10, 20).valid)

    def test_failed_or_cancelled_copy_does_not_substitute_for_publication(self):
        evaluator = self.evaluator()
        evaluator.add(migration_page(10), 10)
        current = migration_page(11, 7, 5)[0]
        current = dataclasses.replace(
            current,
            counters=tuple(
                (
                    key,
                    (
                        0
                        if key[0] in ("published_blocks", "published_bytes")
                        else 2 if key[0] == "failed_operations" else value
                    ),
                )
                for key, value in current.counters
            ),
        )
        evaluator.add((current,), 11)
        self.assertFalse(evaluator.assess_migration(10, 20).valid)

    def test_publication_in_wrong_operation_or_endpoint_is_rejected(self):
        for direction in ("d2h", "h2d"):
            for mutation in ("wrong_operation", "source_pool_only"):
                with self.subTest(direction=direction, mutation=mutation):

                    def mutate(sample):
                        counters = []
                        for key, value in sample.counters:
                            if len(key) == 6 and key[0] == "published_blocks":
                                matches = (key[4], key[5]) == (
                                    (1, 2) if direction == "d2h" else (2, 1)
                                )
                                if matches and mutation == "wrong_operation":
                                    key = (
                                        *key[:3],
                                        1 if direction == "d2h" else 3,
                                        *key[4:],
                                    )
                                elif matches and key[2] == (
                                    "host" if direction == "d2h" else "gpu"
                                ):
                                    value = 0
                            counters.append((key, value))
                        return dataclasses.replace(sample, counters=tuple(counters))

                    evaluator = PressureCoverage(
                        (mutate(migration_page(9)[0]),), configured()
                    )
                    evaluator.add((mutate(migration_page(10)[0]),), 10)
                    evaluator.add((mutate(migration_page(11, 7, 5)[0]),), 11)
                    result = evaluator.assess_migration(10, 20)
                    self.assertTrue(result.epoch_valid)
                    self.assertFalse(result.valid)

    def test_outer_uncertainty_crossing_measurement_start_is_rejected(self):
        evaluator = self.evaluator()
        evaluator.add(migration_page(10, bounds=(-0.2, 0.2)), 10.2)
        evaluator.add(migration_page(11, 7, 5, bounds=(-0.1, 0.1)), 11.1)
        result = evaluator.assess_migration(10, 20)
        self.assertTrue(result.epoch_valid)
        self.assertFalse(result.valid)

    def test_outer_uncertainty_crossing_measurement_end_is_rejected(self):
        evaluator = self.evaluator()
        evaluator.add(migration_page(19, bounds=(-0.1, 0.1)), 19.1)
        evaluator.add(migration_page(20, 7, 5, bounds=(-0.2, 0.2)), 20.2)
        self.assertFalse(evaluator.assess_migration(10, 20).valid)

    def test_pre_and_post_window_publications_do_not_qualify(self):
        for left, right in ((8, 9), (20, 21)):
            with self.subTest(left=left):
                baseline = migration_page(left - 1)
                evaluator = PressureCoverage(baseline, configured())
                evaluator.add(migration_page(left), left)
                evaluator.add(migration_page(right, 7, 5), right)
                self.assertFalse(evaluator.assess_migration(10, 20).valid)

    def test_every_independent_owner_group_requires_both_directions(self):
        def page(t, peer_h2d):
            owner = migration_page(t, t - 10, t - 10)[0]
            peer = dataclasses.replace(migration_page(t, t - 10, peer_h2d)[0], rank=1)
            return owner, peer

        evaluator = PressureCoverage(page(10, 0), configured())
        evaluator.add(page(11, 0), 11)
        evaluator.add(page(12, 0), 12)
        result = evaluator.assess_migration(10, 20)
        self.assertFalse(result.valid)
        self.assertEqual(len(result.groups), 2)
        self.assertEqual(result.reasons, ("missing_h2d_load:1:0",))

    def test_four_shared_tp_views_yield_one_owner_group_without_summing(self):
        evaluator = self.evaluator(participants=4)
        evaluator.add(migration_page(10, participants=4), 10)
        evaluator.add(migration_page(11, 7, 5, participants=4), 11)
        result = evaluator.assess_migration(10, 20)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.groups), 1)
        self.assertEqual(
            (result.groups[0].owner_rank, result.groups[0].owner_group_id), (0, 0)
        )
        self.assertEqual(result.groups[0].d2h_evidence[0].after_blocks, 7)
        self.assertEqual(result.groups[0].h2d_evidence[0].after_blocks, 5)

    def test_participant_publication_cannot_replace_owner_publication(self):
        evaluator = self.evaluator(participants=4)
        evaluator.add(migration_page(10, participants=4), 10)
        current = migration_page(11, 7, 5, participants=4)
        owner = migration_page(11, participants=4)[0]
        evaluator.add((owner, *current[1:]), 11)
        self.assertFalse(evaluator.assess_migration(10, 20).valid)

    def test_missing_shared_participant_breaks_delta_adjacency(self):
        for complete in (True, False):
            with self.subTest(complete=complete):
                evaluator = self.evaluator(participants=4)
                evaluator.add(migration_page(10, participants=4), 10)
                evaluator.add(
                    migration_page(11, 7, 5, participants=4)[:3], 11, complete=complete
                )
                evaluator.add(migration_page(12, 7, 5, participants=4), 12)
                self.assertFalse(evaluator.assess_migration(10, 20).valid)

    def test_stale_shared_participant_cannot_certify_owner_increment(self):
        evaluator = self.evaluator(participants=4)
        evaluator.add(migration_page(10, participants=4), 10)
        current = migration_page(11, 7, 5, participants=4)
        stale = dataclasses.replace(current[3], offset_bounds=(-3, 0))
        evaluator.add((*current[:3], stale), 11)
        self.assertFalse(evaluator.assess_migration(10, 20).valid)

    def test_counter_decrease_invalidates_existing_migration_evidence(self):
        evaluator = self.evaluator()
        evaluator.add(migration_page(10), 10)
        evaluator.add(migration_page(11, 7, 5), 11)
        evaluator.add(migration_page(12, 6, 5), 12)
        result = evaluator.assess_migration(10, 20)
        self.assertFalse(result.epoch_valid)
        self.assertFalse(result.valid)
        self.assertIn("counter_reset", result.reasons)

    def test_instance_layout_and_participant_changes_invalidate_evidence(self):
        mutations = (
            lambda s: dataclasses.replace(s, instance="restarted"),
            lambda s: dataclasses.replace(
                s, pools=(dataclasses.replace(s.pools[0], capacity=101), s.pools[1])
            ),
            lambda s: dataclasses.replace(
                s, ownership=dataclasses.replace(s.ownership, participants=(0, 1))
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                evaluator = self.evaluator(participants=4)
                evaluator.add(migration_page(10, participants=4), 10)
                evaluator.add(migration_page(11, 7, 5, participants=4), 11)
                current = migration_page(12, 7, 5, participants=4)
                evaluator.add((mutate(current[0]), *current[1:]), 12)
                result = evaluator.assess_migration(10, 20)
                self.assertFalse(result.epoch_valid)
                self.assertFalse(result.valid)
                self.assertIn("inventory_discontinuity", result.reasons)

    def test_missing_boundary_and_gap_never_certify_a_delta(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                evaluator = self.evaluator()
                evaluator.add(migration_page(10), 10)
                if missing:
                    evaluator.add((), 11)
                evaluator.add(migration_page(13, 7, 5), 13)
                self.assertFalse(evaluator.assess_migration(10, 20).valid)

    def test_sparse_endpoint_zero_baseline_is_explicit_and_window_bounded(self):
        baseline = dataclasses.replace(
            migration_page(9)[0], counters=((("trigger", 0), 1),)
        )
        evaluator = PressureCoverage((baseline,), configured())
        first = migration_page(10)[0]
        evaluator.add(
            (dataclasses.replace(first, counters=((("trigger", 0), 1),)),), 10
        )
        evaluator.add(migration_page(11, 7, 5), 11)
        result = evaluator.assess_migration(10, 20)
        self.assertTrue(result.valid)
        evidence = result.groups[0].d2h_evidence[0]
        self.assertEqual(evidence.source, "published_blocks")
        self.assertEqual((evidence.before_blocks, evidence.after_blocks), (0, 7))
        self.assertEqual(
            evidence.zero_baseline_endpoints,
            (("published_blocks", 0, "host", 2, 1, 2),),
        )
        self.assertEqual(
            evidence.endpoint_deltas,
            ((("published_blocks", 0, "host", 2, 1, 2), 0, 7),),
        )

    def test_history_overflow_fails_closed_even_with_retained_directions(self):
        evaluator = self.evaluator(max_segments=1)
        evaluator.add(migration_page(10), 10)
        evaluator.add(migration_page(11, 7, 5), 11)
        evaluator.add(migration_page(12, 8, 6), 12)
        result = evaluator.assess_migration(10, 20)
        self.assertFalse(result.valid)
        self.assertIn("history_overflow", result.reasons)
        self.assertEqual(result.groups[0].d2h_intervals, ((11, 12),))
        self.assertEqual(len(result.groups[0].d2h_evidence), 1)

    def test_invalid_windows_are_rejected(self):
        evaluator = self.evaluator()
        for start, end in ((10, 10), (20, 10), (float("nan"), 20)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                evaluator.assess_migration(start, end)


class CoverageTest(unittest.TestCase):
    def test_declared_shared_owner_preserves_raw_peers_and_owner_clock(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            PressureCoverage,
        )

        def page(t, baseline=False):
            message = shared_wire(t, baseline)
            return decode_pressure_response(message, t, t, (0, 1)).samples

        baseline = page(0, True)
        evaluator = PressureCoverage(baseline, configured())
        for t in range(1, 5):
            samples = page(t)
            self.assertEqual(samples[1].pools[0].used, 0)
            self.assertEqual(samples[1].pools[1].host_resident, 0)
            evaluator.add(samples, t)
        self.assertTrue(evaluator.assess(1, 4).valid)
        self.assertEqual(evaluator.intervals(1, 4), [(1, 4)])

    def test_interval_contract(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            confirmed_segment,
            coverage,
            intersect_intervals,
            merge_intervals,
        )

        self.assertEqual(coverage([(0, 57)], 0, 60), (0.95, 3))
        self.assertEqual(merge_intervals([(0, 1), (1, 2), (2, 4)]), [(0, 4)])
        self.assertEqual(confirmed_segment(0, 4, -0.01, 0.01), (0.01, 3.99))
        self.assertEqual(intersect_intervals([(0, 60)], [(1, 59)]), [(1, 59)])

    def test_adjacent_samples_and_reset(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            PressureCoverage,
        )

        baseline = dataclasses.replace(
            rank(), groups=(GroupSample(0, "FULL", ("gpu",), "host", 0, 0),)
        )
        evaluator = PressureCoverage((baseline,), configured())
        for t in range(4):
            evaluator.add(
                (
                    dataclasses.replace(
                        rank(),
                        sequence=t + 2,
                        server_s=t,
                        client_send_s=t,
                        client_receive_s=t,
                    ),
                ),
                t,
            )
            if t == 2:
                self.assertFalse(evaluator.assess(0, 3).valid)
        self.assertEqual(evaluator.intervals(0, 3), [(0, 3)])
        evaluator.add(
            (
                dataclasses.replace(
                    rank(), sequence=1, server_s=4, client_send_s=4, client_receive_s=4
                ),
            ),
            4,
        )
        self.assertIn("sequence_reset", evaluator.reasons)
        self.assertEqual(evaluator.intervals(3, 4), [])

    def test_first_sample_sequence_must_exceed_prefill_baseline(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            PressureCoverage,
        )

        baseline = dataclasses.replace(
            rank(),
            sequence=10,
            groups=(GroupSample(0, "FULL", ("gpu",), "host", 0, 0),),
        )
        for first_sequence in (9, 10):
            with self.subTest(first_sequence=first_sequence):
                evaluator = PressureCoverage((baseline,), configured())
                for t in range(5):
                    evaluator.add(
                        (
                            dataclasses.replace(
                                rank(),
                                sequence=first_sequence + t,
                                server_s=t,
                                client_send_s=t,
                                client_receive_s=t,
                            ),
                        ),
                        t,
                    )
                self.assertFalse(evaluator.epoch_valid)
                self.assertIn("sequence_reset", evaluator.reasons)
                self.assertEqual(evaluator.intervals(0, 4), [])
                self.assertFalse(evaluator.assess(0, 4).valid)

    def test_activity_timeout(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            ActivitySample,
            activity_valid,
        )

        sample = ActivitySample(0, 0, 5, 1, 0)
        self.assertTrue(activity_valid(sample, 10, 6))
        self.assertFalse(activity_valid(sample, 12, 6))
        self.assertFalse(activity_valid(ActivitySample(0, 0, None, 0, 1), 100, 6))

    def test_uncertainty_is_applied_only_to_segment_edges(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            PressureCoverage,
        )

        base = dataclasses.replace(
            rank(), groups=(GroupSample(0, "FULL", ("gpu",), "host", 0, 0),)
        )
        e = PressureCoverage((base,), configured())
        for t in range(5):
            e.add(
                (
                    dataclasses.replace(
                        rank(),
                        sequence=t + 2,
                        server_s=t,
                        client_send_s=t,
                        client_receive_s=t + 0.02,
                        offset_bounds=(-0.01, 0.01),
                    ),
                ),
                t + 0.02,
            )
        self.assertEqual(e.intervals(0, 4), [(0.01, 3.99)])

    def test_window_reports_cumulative_reasons_with_explicit_epoch_scope(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            PressureCoverage,
        )

        base = dataclasses.replace(
            rank(), groups=(GroupSample(0, "FULL", ("gpu",), "host", 0, 0),)
        )
        evaluator = PressureCoverage((base,), configured())
        for t in range(5):
            point = dataclasses.replace(
                rank(), sequence=t + 2, server_s=t, client_send_s=t, client_receive_s=t
            )
            if t == 0:
                point = dataclasses.replace(point, groups=base.groups)
            evaluator.add((point,), t)
        result = dataclasses.asdict(evaluator.assess(2, 4))
        self.assertTrue(result["valid"])
        self.assertEqual(result["ratio"], 1)
        self.assertNotIn("reasons", result)
        self.assertIn("pressure_lost", result["epoch_reasons"])

    def test_gap_age_missing_and_retention(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            PressureCoverage,
        )

        base = dataclasses.replace(
            rank(), groups=(GroupSample(0, "FULL", ("gpu",), "host", 0, 0),)
        )

        def point(t, sequence):
            return dataclasses.replace(
                rank(),
                sequence=sequence,
                server_s=t,
                client_send_s=t,
                client_receive_s=t,
            )

        e = PressureCoverage((base,), configured(), max_segments=1)
        e.add((point(0, 2),), 0)
        e.add((point(2, 3),), 2)
        self.assertEqual(e.intervals(0, 2), [(0, 2)])
        e.add((point(4.01, 4),), 4.01)
        self.assertIn("sample_gap", e.reasons)
        e.add((point(5, 5),), 8)
        self.assertIn("stale_or_unknown_age", e.reasons)
        e.add((), 9)
        e.add((point(10, 6),), 10)
        e.add((point(11, 7),), 11)
        self.assertEqual(e.intervals(0, 11), [(10, 11)])
        self.assertIn("history_overflow", e.reasons)

    def test_rank_intersection_and_endpoint_reset(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            PressureCoverage,
        )

        bases = tuple(
            dataclasses.replace(
                rank(r), groups=(GroupSample(0, "FULL", ("gpu",), "host", 0, 0),)
            )
            for r in (0, 1)
        )
        e = PressureCoverage(bases, configured())
        for t in range(4):
            e.add(
                tuple(
                    dataclasses.replace(
                        rank(r),
                        sequence=t + 2,
                        server_s=t,
                        client_send_s=t,
                        client_receive_s=t,
                        pools=(
                            rank(r).pools
                            if t > 0 or r == 0
                            else (
                                PoolSample("gpu", 100, 0, 0, 0.8, 0.9),
                                rank(r).pools[1],
                            )
                        ),
                    )
                    for r in (0, 1)
                ),
                t,
            )
        self.assertEqual(e.intervals(0, 3), [(1, 3)])
        message = wire()
        sample = decode_pressure_response(message, 1, 1.1, (0,)).samples[0]
        base = dataclasses.replace(
            sample,
            sequence=0,
            groups=(
                dataclasses.replace(sample.groups[0], triggers=0, gpu_host_published=0),
            ),
        )
        e = PressureCoverage((base,), configured())
        e.add((sample,), 1.1)
        message.ranks[0].sequence = 2
        message.ranks[0].sampled_monotonic_ns = 2000000000
        message.ranks[0].transfers[0].published_blocks = 6
        message.ranks[0].transfers[1].published_blocks = 100
        e.add(decode_pressure_response(message, 2, 2.1, (0,)).samples, 2.1)
        self.assertFalse(e.epoch_valid)
        self.assertIn("counter_reset", e.reasons)

    def test_clock_discontinuity_and_reset_after_missing_poll(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            PressureCoverage,
        )

        base = dataclasses.replace(
            rank(), groups=(GroupSample(0, "FULL", ("gpu",), "host", 0, 0),)
        )
        e = PressureCoverage((base,), configured())
        e.add(
            (
                dataclasses.replace(
                    rank(), sequence=2, server_s=0, client_send_s=0, client_receive_s=0
                ),
            ),
            0,
        )
        e.add(
            (
                dataclasses.replace(
                    rank(),
                    sequence=3,
                    server_s=1,
                    client_send_s=1.1,
                    client_receive_s=1.1,
                ),
            ),
            1.1,
        )
        self.assertIn("clock_discontinuity", e.reasons)
        self.assertEqual(e.intervals(0, 2), [])
        e.add((), 2)
        e.add(
            (
                dataclasses.replace(
                    rank(), sequence=2, server_s=3, client_send_s=3, client_receive_s=3
                ),
            ),
            3,
        )
        self.assertFalse(e.epoch_valid)
        self.assertIn("sequence_reset", e.reasons)

    def test_fanout_offset_composition(self):
        message = wire()
        message.coordinator_received_ns = 10_000_000_000
        message.coordinator_sent_ns = 10_300_000_000
        message.ranks[0].coordinator_send_ns = 10_100_000_000
        message.ranks[0].coordinator_receive_ns = 10_200_000_000
        sample = decode_pressure_response(message, 100, 100.5, (0,)).samples[0]
        self.assertAlmostEqual(sample.offset_bounds[0], 99.1)
        self.assertAlmostEqual(sample.offset_bounds[1], 99.4)


if __name__ == "__main__":
    unittest.main()
