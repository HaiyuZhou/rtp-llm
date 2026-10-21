"""Static configuration screening for formal FIFO cache pressure runs."""

from __future__ import annotations

MIN_REUSE_TOKENS = 32 * 1024
FORMAL_FILL_TARGET_TOKENS = 100 * 1024
MIN_FORMAL_CLIENT_INFLIGHT = 16


def formal_config_reasons(
    *,
    manual_seed_count: int | None,
    client_inflight: int,
    test_block_num: int,
    fill_target_tokens: int,
) -> tuple[str, ...]:
    """Return ordered static exclusions; live capacity planning is still required."""
    for name, value, minimum in (
        ("manual_seed_count", manual_seed_count, 1),
        ("client_inflight", client_inflight, 1),
        ("test_block_num", test_block_num, 0),
        ("fill_target_tokens", fill_target_tokens, 1),
    ):
        if name == "manual_seed_count" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    reasons = []
    if manual_seed_count is not None:
        reasons.append("manual_seed_count")
    if client_inflight < MIN_FORMAL_CLIENT_INFLIGHT:
        reasons.append("client_inflight_below_16")
    if test_block_num > 0:
        reasons.append("synthetic_test_block_capacity")
    if fill_target_tokens != FORMAL_FILL_TARGET_TOKENS:
        reasons.append("nonstandard_fill_target")
    return tuple(reasons)
