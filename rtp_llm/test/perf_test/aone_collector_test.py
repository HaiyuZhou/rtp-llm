import copy
import unittest

from rtp_llm.test.perf_test.aone.collector import collect_aone_results
from rtp_llm.test.perf_test.aone.sharding import plan_shards
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.plan import build_execution_plan, thaw
from rtp_llm.test.perf_test.state import RunState


def _plan():
    config = {
        "schema_version": 1,
        "model": {"label": "test", "type": "registered", "checkpoint": "/m"},
        "engine": {
            "max_seq_len": 4097,
            "max_batch_tokens": 8192,
            "concurrency_limit": 1,
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
    cases = [
        {"input_len": 512, "cache_len": 0, "compute_len": 512, "batch_size": 1},
        {
            "input_len": 1024,
            "cache_len": 512,
            "compute_len": 512,
            "batch_size": 1,
        },
    ]
    return build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(config),
        cases=cases,
        result_directory="runs/aone",
    )


def _case_result(case):
    identity = thaw(case.identity)
    return {
        "case_id": case.case_id,
        "identity": identity,
        "status": "ok",
        "runs": 1,
        "errors": [],
        "metrics": {"latency_ms": 10.0},
    }


def _valid_results(plan, shards):
    cases = {case.case_id: case for case in plan.cases}
    return [
        {
            "schema_version": 1,
            "run_id": "run-1",
            "shard_id": shard.shard_id,
            "mode": "prefill",
            "status": "completed",
            "config_fingerprint": plan.config_fingerprint,
            "workload_fingerprint": plan.workload_fingerprint,
            "owned_case_ids": list(shard.case_ids),
            "completed_case_ids": list(shard.case_ids),
            "cases": [_case_result(cases[case_id]) for case_id in shard.case_ids],
            "engine_diagnostics": {},
            "failure": None,
        }
        for shard in shards.shards
    ]


class AOneCollectorTest(unittest.TestCase):
    def setUp(self):
        self.plan = _plan()
        self.shards = plan_shards(self.plan)
        self.results = _valid_results(self.plan, self.shards)

    def test_valid_collection_merges_every_case_once(self):
        outcome = collect_aone_results(
            self.plan, run_id="run-1", shard_plan=self.shards, results=self.results
        )

        self.assertEqual(RunState.COMPLETED, outcome.state)
        self.assertEqual(0, outcome.exit_code)
        self.assertEqual(
            [case.case_id for case in self.plan.cases],
            [case["case_id"] for case in outcome.cases],
        )

    def test_missing_shard_or_remote_partial_is_exit_five(self):
        missing = self.results[:1]
        outcome = collect_aone_results(
            self.plan, run_id="run-1", shard_plan=self.shards, results=missing
        )
        self.assertEqual(RunState.PARTIAL, outcome.state)
        self.assertEqual(5, outcome.exit_code)

        partial = copy.deepcopy(self.results)
        partial[1]["status"] = "partial"
        partial[1]["completed_case_ids"] = []
        partial[1]["cases"] = []
        outcome = collect_aone_results(
            self.plan, run_id="run-1", shard_plan=self.shards, results=partial
        )
        self.assertEqual(RunState.PARTIAL, outcome.state)
        self.assertEqual(5, outcome.exit_code)

    def test_integrity_mismatches_are_exit_six(self):
        unknown = _case_result(self.plan.cases[0])
        unknown["case_id"] = "prefill-unknown"
        mutations = {
            "duplicate shard": lambda value: value.append(copy.deepcopy(value[0])),
            "unknown shard": lambda value: value[0].update(shard_id="shard-9999"),
            "run identity": lambda value: value[0].update(run_id="other-run"),
            "config fingerprint": lambda value: value[0].update(
                config_fingerprint="0" * 64
            ),
            "workload fingerprint": lambda value: value[0].update(
                workload_fingerprint="f" * 64
            ),
            "ownership": lambda value: value[0]["owned_case_ids"].append(
                self.plan.cases[1].case_id
            ),
            "missing declared result": lambda value: value[0].update(cases=[]),
            "unknown case": lambda value: value[0]["cases"].append(unknown),
            "duplicate case": lambda value: value[0]["cases"].append(
                copy.deepcopy(value[0]["cases"][0])
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                payloads = copy.deepcopy(self.results)
                mutate(payloads)

                outcome = collect_aone_results(
                    self.plan,
                    run_id="run-1",
                    shard_plan=self.shards,
                    results=payloads,
                )

                self.assertNotEqual(RunState.COMPLETED, outcome.state)
                self.assertEqual(6, outcome.exit_code)


if __name__ == "__main__":
    unittest.main()
