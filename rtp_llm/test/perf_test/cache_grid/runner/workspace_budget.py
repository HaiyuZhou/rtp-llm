"""CPU-only admission checks for the fixed DSV4 CP8 prefill workspace policy."""

FIXED_POLICY = "fixed_cp8_1m_v1"
WORKSPACE_TOKENS = 1048576
CP_ALIGNMENT = 16


def aligned(value, alignment=CP_ALIGNMENT):
    return (value + alignment - 1) // alignment * alignment


def fixed_workspace_grid(payload):
    policy = payload.get("generator", {}).get("workspace_policy")
    if policy is not None and policy != FIXED_POLICY:
        raise ValueError(f"unsupported cache-grid workspace policy: {policy}")
    return policy == FIXED_POLICY


def input_limit(batch_size, output_tokens=1):
    """Conservative per-request limit including output and CP padding."""
    if batch_size <= 0 or output_tokens <= 0:
        raise ValueError("batch size and output reserve must be positive")
    return WORKSPACE_TOKENS // batch_size // CP_ALIGNMENT * CP_ALIGNMENT - output_tokens


def grid_token_budget(payload):
    budget = int(
        payload.get("generator", {})
        .get("parameters", {})
        .get("max_batch_tokens", WORKSPACE_TOKENS)
    )
    if not 0 < budget <= WORKSPACE_TOKENS:
        raise ValueError("fixed workspace token budget must be in [1, 1048576]")
    return budget


def validate_fixed_workspace(
    cases,
    *,
    commit_tail=4096,
    block=4096,
    output_tokens=1,
    token_budget=WORKSPACE_TOKENS,
):
    """Validate raw or normalized cases; do not trust summary/estimated fields.

    Input sums include cached prefixes. Rectangle checks reserve output tokens
    before CP alignment, deliberately conservative for prefill-only measurements.
    Independent seeds are concurrent; shared-by-group seeds count once per group.
    """
    if output_tokens != 1 or block != 4096 or commit_tail <= 0 or commit_tail % block:
        raise ValueError(
            "fixed workspace requires CP8/block4096, aligned seed tail and decode=1"
        )
    if not 0 < token_budget <= WORKSPACE_TOKENS:
        raise ValueError("fixed workspace token budget must be in [1, 1048576]")
    stats = []

    def phase(case_id, name, lengths):
        count = sum(n for n, _ in lengths)
        total = sum(n * length for n, length in lengths)
        rectangle = (
            count
            * aligned(max((length for _, length in lengths), default=0) + output_tokens)
            if count
            else 0
        )
        if total > token_budget or rectangle > WORKSPACE_TOKENS:
            raise ValueError(
                f"case {case_id} {name}: input sum={total} (budget={token_budget}), "
                f"CP-padded rectangle with output reserve={rectangle} "
                f"(workspace={WORKSPACE_TOKENS}); shorten requests or reduce batch"
            )
        stats.append(
            {
                "case_id": case_id,
                "phase": name,
                "requests": count,
                "input_tokens": total,
                "workspace_tokens": rectangle,
            }
        )

    phase("probe", "probe", [(1, max(2 * block, block + commit_tail))])
    for index, case in enumerate(cases):
        case_id = case.get("case_id", index)
        batch = int(case.get("batch_size", 1))
        policy = case.get("prefix_policy", "independent")
        if batch <= 0 or policy not in ("independent", "shared_by_group"):
            raise ValueError(f"case {case_id}: invalid batch/prefix policy")
        groups = case.get("request_groups")
        if groups is None:
            groups = [
                dict(
                    count=batch,
                    input_len=case["input_len"],
                    cache_len=case.get("cache_len", 0),
                )
            ]
        measure, seeds = [], []
        for group in groups:
            count, length, cached = (
                int(group["count"]),
                int(group["input_len"]),
                int(group.get("cache_len", 0)),
            )
            if count <= 0 or not 0 <= cached < length:
                raise ValueError(f"case {case_id}: invalid request group")
            measure.append((count, length))
            if cached:
                if cached % block or cached + commit_tail > length:
                    raise ValueError(
                        f"case {case_id}: cache alignment/seed tail exceeds input"
                    )
                seeds.append(
                    (1 if policy == "shared_by_group" else count, cached + commit_tail)
                )
        if sum(n for n, _ in measure) != batch:
            raise ValueError(f"case {case_id}: group counts must sum to batch_size")
        phase(case_id, "measure", measure)
        phase(case_id, "seed", seeds)
    return stats
