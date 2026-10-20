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
    offset_bounds: tuple[float, float] | None = None
    counters: tuple[tuple[tuple, int], ...] = ()


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
        # HOST/DISK and observe-only groups may disable eviction with {0, 0}.
        # Enabled, ordered thresholds are required only by the FULL GPU gate.
        _number(pool.low_ratio, "low_ratio", inclusive=True)
        _number(pool.high_ratio, "high_ratio", inclusive=True)
        if max(pool.low_ratio, pool.high_ratio) > 1:
            raise ValueError("watermark ratios must be <= 1")
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
    if group.group_type == "FULL":
        for key in group.device_pool_ids:
            pool = pools[key]
            if not 0 < pool.low_ratio < pool.high_ratio <= 1:
                raise ValueError("FULL device requires 0 < low < high <= 1")


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

    if hasattr(response, "ranks"):
        record["rank_offset_bounds_s"] = {
            str(rank.world_rank): _wire_offset_bounds(
                response, rank, client_send_s, client_receive_s
            )
            for rank in response.ranks
            if not rank.error
        }

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
                _wire_offset_bounds(response, rank, client_send_s, client_receive_s),
                _wire_counters(rank),
            )
        )
    result = tuple(samples)
    select_targets(result, expected_ranks)
    return DecodedPressure(response.SerializeToString(), result)


Interval = tuple[float, float]


def merge_intervals(values) -> list[Interval]:
    merged = []
    for start, end in sorted(values):
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("interval endpoints must be finite")
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def intersect_intervals(left, right) -> list[Interval]:
    left, right = merge_intervals(left), merge_intervals(right)
    i = j = 0
    output = []
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            output.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return output


def coverage(values, start, end) -> tuple[float, float]:
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("coverage requires a positive finite window")
    clipped = merge_intervals((max(a, start), min(b, end)) for a, b in values)
    valid = 0.0
    cursor = start
    longest = 0.0
    for a, b in clipped:
        longest = max(longest, a - cursor)
        valid += b - a
        cursor = b
    return valid / (end - start), max(longest, end - cursor)


def confirmed_segment(start_server, end_server, offset_low, offset_high):
    if (
        not all(
            math.isfinite(v)
            for v in (start_server, end_server, offset_low, offset_high)
        )
        or offset_low > offset_high
    ):
        raise ValueError("invalid clock bounds")
    start, end = start_server + offset_high, end_server + offset_low
    return (start, end) if end > start else None


def _wire_offset_bounds(response, rank, client_send, client_receive):
    """Client minus rank clock; fan-out composes both transport uncertainties."""
    received = getattr(response, "coordinator_received_ns", 0)
    sent = getattr(response, "coordinator_sent_ns", 0)
    rank_send = getattr(rank, "coordinator_send_ns", 0)
    rank_receive = getattr(rank, "coordinator_receive_ns", 0)
    if not any((received, sent, rank_send, rank_receive)):
        return (
            client_send - rank.sampled_monotonic_ns / 1e9,
            client_receive - rank.sampled_monotonic_ns / 1e9,
        )
    if (
        min(received, sent, rank_send, rank_receive) <= 0
        or not received <= rank_send <= rank_receive <= sent
    ):
        raise ValueError("invalid coordinator timing bounds")
    # Outer request establishes client-coordinator offset; internal request
    # establishes coordinator-rank offset. No cross-process clocks are compared.
    server = rank.sampled_monotonic_ns / 1e9
    low = client_send - received / 1e9 + rank_send / 1e9 - server
    high = client_receive - sent / 1e9 + rank_receive / 1e9 - server
    if low > high:
        raise ValueError("inconsistent composed clock bounds")
    return low, high


@dataclass(frozen=True)
class ActivitySample:
    last_dispatch_s: float
    last_progress_s: float
    active_step_started_s: float | None
    running: int | None
    waiting: int | None


@dataclass(frozen=True)
class ActivityAssessment:
    valid: bool
    reasons: tuple[str, ...]


def assess_activity(sample, now, timeout_s) -> ActivityAssessment:
    _number(now, "now", inclusive=True)
    _number(timeout_s, "timeout_s")

    def recent(value):
        return (
            value is not None and math.isfinite(value) and 0 <= now - value <= timeout_s
        )

    if sample.running is None or sample.waiting is None:
        return ActivityAssessment(False, ("unknown_activity",))
    _integer(sample.running, "running")
    _integer(sample.waiting, "waiting")
    active = recent(sample.active_step_started_s)
    supplied = recent(sample.last_dispatch_s) or active or sample.waiting > 0
    progressing = recent(sample.last_progress_s) or active
    reasons = tuple(
        reason
        for valid, reason in (
            (supplied, "supply_timeout"),
            (progressing, "progress_timeout"),
        )
        if not valid
    )
    return ActivityAssessment(not reasons, reasons)


def activity_valid(sample, now, timeout_s) -> bool:
    return assess_activity(sample, now, timeout_s).valid


@dataclass(frozen=True)
class CoverageAssessment:
    epoch_valid: bool
    max_segments_per_target: int
    ratio: float
    longest_invalid_s: float
    valid: bool
    intervals: tuple[Interval, ...]
    reasons: tuple[str, ...]


class PressureCoverage:
    """Bounded streaming sampled evidence, shared by all diagnostic windows.

    Baselines are captured before fill. Call add once per poll, including empty
    failed polls; persist raw evidence separately. Historical segments beyond
    max_segments per target are dropped explicitly, so their time is unknown.
    """

    def __init__(self, baselines, config, max_segments=4096):
        _integer(max_segments, "max_segments", 1)
        self.config = config
        self.ranks = tuple(sorted(s.rank for s in baselines))
        self.targets = select_targets(tuple(baselines), self.ranks)
        self.baselines = {s.rank: s for s in baselines}
        self.previous = {}
        self.last_seen = {}
        self.segments = {(t.rank, t.group_id): [] for t in self.targets}
        self.open_segments = {}
        self.max_segments = max_segments
        self.reasons = set()
        self.epoch_valid = True

    def _break(self, rank):
        self.previous.pop(rank, None)
        for key in list(self.open_segments):
            if key[0] == rank:
                del self.open_segments[key]

    def add(self, samples, now):
        _number(now, "now", inclusive=True)
        if not self.epoch_valid:
            return
        samples = tuple(samples)
        try:
            validate_inventory(select_targets(samples, self.ranks), self.targets)
        except ValueError:
            self.reasons.add("inventory_discontinuity")
            # A failed/missing poll breaks adjacency; changed service identity
            # invalidates this epoch and requires new pre-fill baselines.
            if any(
                s.rank in self.baselines
                and s.instance != self.baselines[s.rank].instance
                for s in samples
            ):
                self.epoch_valid = False
            for rank in self.ranks:
                self._break(rank)
            return
        for sample in samples:
            rank = sample.rank
            bounds = sample.offset_bounds or (
                sample.client_send_s - sample.server_s,
                sample.client_receive_s - sample.server_s,
            )
            low, high = bounds
            reference = self.last_seen.get(rank, self.baselines[rank])
            old_counters = dict(reference.counters)
            counters = dict(sample.counters)
            old_groups = {g.group_id: g for g in reference.groups}
            reset = (
                old_counters
                and (
                    set(counters) != set(old_counters)
                    or any(counters[k] < v for k, v in old_counters.items())
                )
            ) or any(
                g.triggers < old_groups[g.group_id].triggers
                or g.gpu_host_published < old_groups[g.group_id].gpu_host_published
                for g in sample.groups
                if g.group_id in old_groups
            )
            reason = (
                "counter_reset"
                if reset
                else (
                    "sequence_reset"
                    if rank in self.last_seen and sample.sequence <= reference.sequence
                    else None
                )
            )
            self.last_seen[rank] = sample
            if reason:
                self.reasons.add(reason)
                self.epoch_valid = False
                self._break(rank)
                continue
            if (
                not all(math.isfinite(v) for v in bounds)
                or low > high
                or now < sample.client_receive_s
                or now - (sample.server_s + low) > self.config.max_sample_age_s
            ):
                self.reasons.add("stale_or_unknown_age")
                self._break(rank)
                continue
            previous = self.previous.get(rank)
            if previous:
                old, old_bounds = previous
                reason = None
                if sample.sequence <= old.sequence:
                    reason = "sequence_reset"
                elif (
                    not 0
                    < sample.server_s - old.server_s
                    <= self.config.max_sample_gap_s
                    or not 0
                    <= sample.client_receive_s - old.client_receive_s
                    <= self.config.max_sample_gap_s
                ):
                    reason = "sample_gap"
                elif max(low, old_bounds[0]) > min(high, old_bounds[1]):
                    reason = "clock_discontinuity"
                old_groups = {g.group_id: g for g in old.groups}
                if any(
                    g.triggers < old_groups[g.group_id].triggers
                    or g.gpu_host_published < old_groups[g.group_id].gpu_host_published
                    for g in sample.groups
                    if g.group_id in old_groups
                ):
                    reason = "counter_reset"
                if reason:
                    self.reasons.add(reason)
                    if reason in ("counter_reset", "sequence_reset"):
                        self.epoch_valid = False
                    self._break(rank)
                    previous = None
            if not self.epoch_valid:
                continue
            base = {g.group_id: g for g in self.baselines[rank].groups}
            pools = {p.pool_id: p for p in sample.pools}
            for target in (t for t in self.targets if t.rank == rank):
                key = (rank, target.group_id)
                group = next(g for g in sample.groups if g.group_id == target.group_id)
                try:
                    valid = group_valid(group, pools, base[group.group_id], self.config)
                except ValueError:
                    valid = False
                if not valid:
                    self.reasons.add("pressure_lost")
                    self.open_segments.pop(key, None)
                    continue
                # Track a valid point even without an interval; next adjacent
                # point extends it. Merge on the server clock before mapping.
                current = self.open_segments.get(key) if previous else None
                if current is None:
                    self.open_segments[key] = [
                        sample.server_s,
                        sample.server_s,
                        low,
                        high,
                    ]
                else:
                    current[1] = sample.server_s
                    current[2] = min(current[2], low)
                    current[3] = max(current[3], high)
                    if self.segments[key] and self.segments[key][-1] is current:
                        pass
                    else:
                        self.segments[key].append(current)
                        if len(self.segments[key]) > self.max_segments:
                            self.segments[key].pop(0)
                            self.reasons.add("history_overflow")
            self.previous[rank] = sample, bounds

    def intervals(self, start, end):
        coverage([], start, end)  # validate window
        result = [(start, end)]
        for segments in self.segments.values():
            mapped = [confirmed_segment(*segment) for segment in segments]
            result = intersect_intervals(result, [v for v in mapped if v is not None])
        return result

    def assess(self, start, end) -> CoverageAssessment:
        values = self.intervals(start, end)
        ratio, longest = coverage(values, start, end)
        return CoverageAssessment(
            self.epoch_valid,
            self.max_segments,
            ratio,
            longest,
            self.epoch_valid
            and ratio >= self.config.min_coverage
            and longest <= self.config.max_invalid_s,
            tuple(values),
            tuple(sorted(self.reasons)),
        )


def _wire_counters(rank):
    """Preserve each cumulative endpoint and unit independently for reset checks."""
    output = []
    for group in rank.groups:
        output.append((("trigger", group.group_id), group.watermark_trigger_count))
    for transfer in rank.transfers:
        endpoint = (
            transfer.group_id,
            transfer.pool_id,
            transfer.operation,
            transfer.source_tier,
            transfer.target_tier,
        )
        for name in (
            "copied_blocks",
            "copied_bytes",
            "published_blocks",
            "published_bytes",
            "failed_operations",
        ):
            value = getattr(transfer, name, 0)
            output.append(((name,) + endpoint, _integer(value, name)))
    return tuple(sorted(output))
