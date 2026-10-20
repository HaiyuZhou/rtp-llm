import dataclasses
import unittest
from types import SimpleNamespace as NS

from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
    GroupSample,
    PoolSample,
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
        used_blocks=90,
        pending_release_blocks=0,
        published_host_blocks=0,
    )
    host = PB(
        pool_id="host",
        tier=2,
        total_blocks=100,
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


if __name__ == "__main__":
    unittest.main()
