"""Frozen FULL-pressure contracts and strict schema-1 evaluator projections.

No generated model modules are imported here. Decoding accepts the actual protobuf
message; raw bytes remain separate from the deliberately smaller evaluator view.
"""

import json
import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class PressureConfig:
    fill_timeout_s: float
    fill_max_requests: int
    fill_no_progress_s: float
    fill_input_tokens: int
    gpu_tolerance_blocks: int
    activity_timeout_s: float
    drain_timeout_s: float
    capture_timeout_s: float
    host_ratio: float = 0.25
    sample_period_s: float = 1.0
    max_sample_gap_s: float = 2.0
    max_sample_age_s: float = 2.0
    steady_s: float = 3.0
    steady_timeout_s: float = 60.0
    measure_s: float = 60.0
    min_coverage: float = 0.95
    max_invalid_s: float = 2.0
    profile_steps: int = 20
    support_s: float = 60.0
    nsys_duration_s: float | None = None


@dataclass(frozen=True)
class PoolSample:
    pool_id: str
    capacity: int
    used: int
    pending_release: int
    low_ratio: float
    high_ratio: float
    host_resident: int = 0


@dataclass(frozen=True)
class GroupSample:
    group_id: int
    group_type: str
    device_pool_ids: tuple[str, ...]
    host_pool_id: str
    triggers: int
    gpu_host_published: int


@dataclass(frozen=True)
class RankSample:
    rank: int
    instance: str
    sequence: int
    server_s: float
    client_send_s: float
    client_receive_s: float
    pools: tuple[PoolSample, ...]
    groups: tuple[GroupSample, ...]


@dataclass(frozen=True)
class PressureTarget:
    rank: int
    instance: str
    group_id: int
    device_pool_ids: tuple[str, ...]
    host_pool_id: str


@dataclass(frozen=True)
class DecodedPressure:
    raw_wire: bytes
    samples: tuple[RankSample, ...]


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, minimum=0, inclusive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    if value < minimum or (not inclusive and value == minimum):
        raise ValueError(f"{name} is below its allowed minimum")
    return value


def _id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("pool/service identity must be nonempty")
    return value


def resolve_pressure_config(values: dict) -> PressureConfig:
    try:
        config = PressureConfig(**values)
    except TypeError as error:
        raise ValueError(f"invalid pressure settings: {error}") from error
    integers = {
        "fill_max_requests",
        "fill_input_tokens",
        "gpu_tolerance_blocks",
        "profile_steps",
    }
    for field in fields(config):
        value = getattr(config, field.name)
        if field.name == "nsys_duration_s" and value is None:
            continue
        if field.name in integers:
            _integer(
                value, field.name, 0 if field.name == "gpu_tolerance_blocks" else 1
            )
        else:
            _number(value, field.name)
    for name in ("host_ratio", "min_coverage"):
        if getattr(config, name) > 1:
            raise ValueError(f"{name} must be <= 1")
    if min(config.max_sample_age_s, config.max_sample_gap_s) < config.sample_period_s:
        raise ValueError("sample age and gap must be >= sample period")
    return config


def _pool_map(pools):
    result = {}
    for pool in pools:
        _id(pool.pool_id)
        for name in ("capacity", "used", "pending_release", "host_resident"):
            _integer(getattr(pool, name), name)
        _number(pool.low_ratio, "low_ratio")
        _number(pool.high_ratio, "high_ratio")
        if not 0 < pool.low_ratio < pool.high_ratio <= 1:
            raise ValueError("require 0 < low < high <= 1")
        if (
            not 0 <= pool.pending_release <= pool.used <= pool.capacity
            or pool.host_resident > pool.used
        ):
            raise ValueError("invalid physical occupancy")
        if pool.pool_id in result and result[pool.pool_id] != pool:
            raise ValueError("conflicting physical pool records")
        result[pool.pool_id] = pool
    return result


def _validate_group(group, pools):
    _integer(group.group_id, "group_id")
    _integer(group.triggers, "triggers")
    _integer(group.gpu_host_published, "gpu_host_published")
    if group.group_type not in ("FULL", "SWA", "LINEAR"):
        raise ValueError("unknown group type")
    if not isinstance(group.device_pool_ids, tuple) or len(
        set(group.device_pool_ids)
    ) != len(group.device_pool_ids):
        raise ValueError("invalid device mapping")
    if not group.device_pool_ids or any(
        key not in pools for key in group.device_pool_ids
    ):
        raise ValueError("missing device pool")
    if group.host_pool_id and group.host_pool_id not in pools:
        raise ValueError("missing host pool")
    if group.host_pool_id in group.device_pool_ids:
        raise ValueError("host and device pools must differ")
    if group.group_type == "FULL" and (
        not group.host_pool_id or pools[group.host_pool_id].capacity <= 0
    ):
        raise ValueError("FULL requires positive host capacity")


def select_targets(
    samples: tuple[RankSample, ...], ranks: tuple[int, ...]
) -> tuple[PressureTarget, ...]:
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("expected ranks must be nonempty and unique")
    for rank in ranks:
        _integer(rank, "rank")
    if len(samples) != len(ranks) or {s.rank for s in samples} != set(ranks):
        raise ValueError("missing, duplicate, or unexpected rank")
    result = []
    for sample in sorted(samples, key=lambda s: s.rank):
        _integer(sample.rank, "rank")
        _id(sample.instance)
        _integer(sample.sequence, "sequence")
        for name in ("server_s", "client_send_s", "client_receive_s"):
            _number(getattr(sample, name), name, inclusive=True)
        if sample.client_receive_s < sample.client_send_s:
            raise ValueError("reversed client timing")
        pools = _pool_map(sample.pools)
        seen = set()
        full = []
        for group in sample.groups:
            _validate_group(group, pools)
            if group.group_id in seen:
                raise ValueError("duplicate group identity")
            seen.add(group.group_id)
            if group.group_type == "FULL":
                full.append(
                    PressureTarget(
                        sample.rank,
                        sample.instance,
                        group.group_id,
                        tuple(sorted(group.device_pool_ids)),
                        group.host_pool_id,
                    )
                )
        if not full:
            raise ValueError("rank has no FULL groups")
        result.extend(sorted(full, key=lambda target: target.group_id))
    return tuple(result)


def validate_inventory(
    current: tuple[PressureTarget, ...], frozen: tuple[PressureTarget, ...]
) -> None:
    """Caller explicitly compares each later inventory with the epoch's first one."""
    if current != frozen:
        raise ValueError("pressure topology or service instance discontinuity")


def group_valid(
    group: GroupSample,
    pools: dict[str, PoolSample],
    baseline: GroupSample,
    config: PressureConfig,
) -> bool:
    validated = _pool_map(pools.values())
    if set(validated) != set(pools) or any(
        key != pool.pool_id for key, pool in pools.items()
    ):
        raise ValueError("pool keys differ from physical identities")
    _validate_group(group, validated)
    _validate_group(baseline, validated)
    if (
        group.group_id,
        group.group_type,
        group.device_pool_ids,
        group.host_pool_id,
    ) != (
        baseline.group_id,
        baseline.group_type,
        baseline.device_pool_ids,
        baseline.host_pool_id,
    ):
        raise ValueError("group topology discontinuity")
    if group.group_type != "FULL":
        raise ValueError("only FULL groups have a pressure predicate")
    if (
        group.triggers < baseline.triggers
        or group.gpu_host_published < baseline.gpu_host_published
    ):
        raise ValueError("pressure counter discontinuity")
    for key in group.device_pool_ids:
        if config.gpu_tolerance_blocks > math.floor(
            pools[key].capacity * pools[key].low_ratio
        ):
            raise ValueError("GPU tolerance exceeds selected low threshold")
        if pools[key].capacity <= 0:
            raise ValueError("unreachable device pressure target")
    device_ok = any(
        pools[key].used - pools[key].pending_release
        >= math.floor(pools[key].capacity * pools[key].low_ratio)
        - config.gpu_tolerance_blocks
        for key in group.device_pool_ids
    )
    host = pools[group.host_pool_id]
    return (
        device_ok
        and host.host_resident >= math.ceil(host.capacity * config.host_ratio)
        and group.triggers > baseline.triggers
        and group.gpu_host_published > baseline.gpu_host_published
    )


def raw_pressure_record(
    response, client_send_s: float, client_receive_s: float
) -> dict:
    """JSON-safe full protobuf evidence, preserving wire names and enum integers."""
    from google.protobuf.json_format import MessageToDict

    _number(client_send_s, "client_send_s", inclusive=True)
    _number(client_receive_s, "client_receive_s", inclusive=True)
    if client_receive_s < client_send_s:
        raise ValueError("reversed client timing")
    record = dict(
        client_send_s=client_send_s,
        client_receive_s=client_receive_s,
        response=MessageToDict(
            response, preserving_proto_field_name=True, use_integers_for_enums=True
        ),
    )

    # Reject protobuf nonfinite-double string representations as well as Python NaNs.
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, list):
            return all(finite(v) for v in value)
        return (
            value not in ("NaN", "Infinity", "-Infinity")
            if isinstance(value, str)
            else True
        )

    if not finite(record):
        raise ValueError("nonfinite raw pressure evidence")
    json.dumps(record, allow_nan=False)
    return record


def decode_pressure_response(
    response,
    client_send_s: float,
    client_receive_s: float,
    expected_ranks: tuple[int, ...],
) -> DecodedPressure:
    """Decode schema-1 protobuf fields; wrapper/message presence is mandatory."""
    if response.schema_version != 1:
        raise ValueError("unsupported pressure schema")
    if len(response.requested_ranks) != len(expected_ranks) or set(
        response.requested_ranks
    ) != set(expected_ranks):
        raise ValueError("response requested ranks mismatch")
    samples = []
    for rank in response.ranks:
        if rank.error:
            raise ValueError(f"rank {rank.world_rank}: {rank.error}")
        raw_pools = {}
        for pool in rank.pools:
            if pool.tier not in (1, 2, 3, 4):
                raise ValueError("unknown pool tier")
            if pool.pool_id in raw_pools and pool != raw_pools[pool.pool_id]:
                raise ValueError("conflicting raw physical pool records")
            raw_pools[pool.pool_id] = pool
        watermarks = {}
        groups = []
        group_ids = {g.group_id for g in rank.groups}
        for transfer in rank.transfers:
            if transfer.group_id not in group_ids or transfer.pool_id not in raw_pools:
                raise ValueError("unknown transfer mapping")
            if (
                transfer.pool_tier != raw_pools[transfer.pool_id].tier
                or transfer.operation not in (1, 2, 3)
                or transfer.source_tier not in (1, 2, 3, 4, 5)
                or transfer.target_tier not in (1, 2, 3, 4, 5)
            ):
                raise ValueError("invalid transfer tier/operation")
        for group in rank.groups:
            if group.group_type not in (1, 2, 3):
                raise ValueError("unknown group type")
            host = group.host_pool_id.value if group.HasField("host_pool_id") else ""
            mappings = [(key, 1, "device_watermark") for key in group.device_pool_ids]
            if host:
                mappings.append((host, 2, "host_watermark"))
            if group.HasField("disk_pool_id"):
                mappings.append((group.disk_pool_id.value, 3, "disk_watermark"))
            for key, tier, field in mappings:
                if (
                    key not in raw_pools
                    or raw_pools[key].tier != tier
                    or not group.HasField(field)
                ):
                    raise ValueError("missing or mismatched pool/watermark mapping")
                watermark = getattr(group, field)
                pair = (watermark.low_ratio, watermark.high_ratio)
                if key in watermarks and watermarks[key] != pair:
                    raise ValueError("conflicting shared-pool watermarks")
                watermarks[key] = pair
            published = 0
            endpoints = set()
            for transfer in rank.transfers:
                if transfer.group_id != group.group_id:
                    continue
                if transfer.pool_id not in {key for key, _, _ in mappings}:
                    raise ValueError("transfer outside group pools")
                identity = (
                    transfer.pool_id,
                    transfer.operation,
                    transfer.source_tier,
                    transfer.target_tier,
                )
                if identity in endpoints:
                    raise ValueError("duplicate physical transfer endpoint")
                endpoints.add(identity)
                if (
                    transfer.pool_tier == 2
                    and transfer.pool_id == host
                    and transfer.source_tier == 1
                    and transfer.target_tier == 2
                    and transfer.operation in (2, 3)
                ):
                    published += _integer(transfer.published_blocks, "published_blocks")
            groups.append(
                GroupSample(
                    group.group_id,
                    {1: "FULL", 2: "SWA", 3: "LINEAR"}[group.group_type],
                    tuple(group.device_pool_ids),
                    host,
                    group.watermark_trigger_count,
                    published,
                )
            )
        pools = []
        for key, pool in raw_pools.items():
            if key not in watermarks:
                raise ValueError("physical pool has no group watermark mapping")
            pools.append(
                PoolSample(
                    key,
                    pool.total_blocks,
                    pool.used_blocks,
                    pool.pending_release_blocks,
                    *watermarks[key],
                    pool.published_host_blocks,
                )
            )
        _integer(rank.sampled_monotonic_ns, "sampled_monotonic_ns")
        _integer(rank.collect_duration_ns, "collect_duration_ns")
        samples.append(
            RankSample(
                rank.world_rank,
                rank.service_instance,
                rank.sequence,
                rank.sampled_monotonic_ns / 1e9,
                client_send_s,
                client_receive_s,
                tuple(pools),
                tuple(groups),
            )
        )
    result = tuple(samples)
    select_targets(result, expected_ranks)
    return DecodedPressure(response.SerializeToString(), result)
