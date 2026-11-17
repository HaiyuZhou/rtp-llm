"""Fixtures shared by tests split from prefill_resume_test."""

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.plan import build_execution_plan, thaw
from rtp_llm.test.perf_test.prefill.checkpoint import resume_fingerprints
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.storage import RunStore
from rtp_llm.test.perf_test.tests.support.fixtures import (
    fixture_engine,
    fixture_execution,
    fixture_model,
    fixture_workload,
    prefill_resume_config,
    prefill_resume_plan,
)


def run_directory_source_config():
    return {
        "schema_version": 1,
        "model": fixture_model(
            label="test-model", type="registered_model", checkpoint="/models/test"
        ),
        "engine": fixture_engine(),
        "workload": fixture_workload(),
        "execution": fixture_execution(),
        "output": {},
        "regression": {},
        "publish": {},
    }


def run_directory_plan(result_directory):
    return build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(run_directory_source_config()),
        cases=[{"input_len": 512, "cache_len": 0, "batch_size": 1}],
        result_directory=str(result_directory),
    )


def run_directory_tree_bytes(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.name != ".writer.lock"
    }


def prefill_resume_case_result(planned, status):
    return {
        "case_id": planned.case_id,
        "identity": thaw(planned.identity),
        "status": status,
        "runs": 1,
        "errors": [] if status == "ok" else [{"code": status, "message": "old"}],
        "metrics": {"latency_ms": 1.0} if status == "ok" else None,
    }


def prefill_resume_tree(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.name != ".writer.lock"
    }


def prefill_resume_initialized(root):
    run_plan = prefill_resume_plan(root)
    initialize_run_directory(
        root,
        source_config=prefill_resume_config(),
        plan=run_plan,
        run_id="run",
        code_version="rev",
    )
    store = RunStore(root)
    store.write_prefill_case(
        run_plan.cases[0].case_id, prefill_resume_case_result(run_plan.cases[0], "ok")
    )
    store.write_prefill_case(
        run_plan.cases[1].case_id,
        prefill_resume_case_result(run_plan.cases[1], "invalid_reuse"),
    )
    fingerprints = resume_fingerprints(run_plan)
    store.write_checkpoint(
        "prefill_progress",
        {
            "schema_version": 1,
            **fingerprints,
            "completed_case_ids": [case.case_id for case in run_plan.cases],
            "case_statuses": {
                run_plan.cases[0].case_id: "ok",
                run_plan.cases[1].case_id: "invalid_reuse",
            },
        },
    )
    return (run_plan, store)


def prefill_adapter_plan(root):
    config = {
        "schema_version": 1,
        "model": fixture_model(),
        "engine": fixture_engine(max_seq_len=4097, concurrency_limit=1),
        "workload": fixture_workload(compute_steps=[256], cache_lengths=[0]),
        "execution": fixture_execution(),
        "output": {},
        "regression": {},
        "publish": {},
    }
    cases = [
        {
            "input_len": 512,
            "cache_len": 0,
            "compute_len": 512,
            "batch_size": 1,
            "measure_runs": 1,
            "seed": 1,
            "prefix_policy": "shared_seed",
        },
        {
            "input_len": 1024,
            "cache_len": 512,
            "compute_len": 512,
            "batch_size": 1,
            "measure_runs": 1,
            "seed": 1,
            "prefix_policy": "shared_seed",
        },
    ]
    return build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(config),
        cases=cases,
        result_directory=str(root),
    )


def prefill_adapter_prepare_store(root):
    (root / "checkpoints").mkdir(parents=True)
    (root / "results").mkdir(parents=True)
    return RunStore(root)


def prefill_adapter_result(case, status):
    return {
        "case_id": case["case_id"],
        "identity": {k: v for (k, v) in case.items() if k != "case_id"},
        "status": status,
        "runs": 1,
        "errors": [] if status == "ok" else [{"code": status, "message": "failed"}],
        "metrics": {"latency_ms": 1.0} if status == "ok" else None,
    }
