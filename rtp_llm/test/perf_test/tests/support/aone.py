"""Minimal AOne configuration builders shared by worker and controller tests."""


def prefill_source(
    *, model=None, engine=None, workload=None, execution=None, output=None
):
    """Fresh minimal remote prefill config; section overrides are explicit per case."""
    source = {
        "schema_version": 1,
        "model": {"label": "test", "type": "registered", "checkpoint": "/m"},
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 2,
        },
        "workload": {
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [256],
            "batch_sizes": [1],
        },
        "execution": {
            "backend": "aone",
            "resource": "daily",
            "gpu_type": "H20",
            "max_parallel": 2,
            "shard_count": 2,
        },
        "output": {},
        "regression": {},
        "publish": {},
    }
    for section, overrides in (
        ("model", model),
        ("engine", engine),
        ("workload", workload),
        ("execution", execution),
        ("output", output),
    ):
        if overrides is not None:
            source[section].update(overrides)
    return source
