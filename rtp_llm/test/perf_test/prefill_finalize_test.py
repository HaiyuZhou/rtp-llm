import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.prefill.finalize import finalize_prefill_run
from rtp_llm.test.perf_test.results import parse_result_envelope


def _plan(root):
    config = {
        "schema_version": 1,
        "model": {"label": "model", "type": "registered", "checkpoint": "/m"},
        "engine": {
            "max_seq_len": 1024,
            "max_batch_tokens": 1024,
            "concurrency_limit": 1,
        },
        "workload": {
            "kind": "cache_grid",
            "max_input_len": 512,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [128],
            "cache_lengths": [0],
            "batch_sizes": [1],
        },
        "execution": {"backend": "local", "devices": [0]},
        "output": {},
        "regression": {},
        "publish": {},
    }
    return build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(config),
        cases=[
            {
                "input_len": 512,
                "cache_len": 0,
                "compute_len": 512,
                "batch_size": 1,
                "measure_runs": 1,
                "seed": 1,
                "prefix_policy": "shared_seed",
            }
        ],
        result_directory=str(root),
    )


class PrefillFinalizeTest(unittest.TestCase):
    def test_writes_valid_common_envelope_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results" / "cases").mkdir(parents=True)
            plan = _plan(root)
            identity = plan.cases[0].to_dict()["identity"]
            result = {
                "case_id": plan.cases[0].case_id,
                "identity": identity,
                "status": "ok",
                "runs": 1,
                "errors": [],
                "metrics": {
                    "input_length": 512,
                    "expected_reuse_length": 0,
                    "observed_reuse_length": 0,
                    "compute_length": 512,
                    "latency_ms": 10.0,
                    "throughput_tokens_per_second": 51200.0,
                    "validation": {"shape": True, "reuse": True, "timing": True},
                    "seed": 1,
                    "prefix_policy": {"kind": "shared_seed"},
                    "measurements": [{"run": 0, "latency_ms": 10.0}],
                },
            }

            envelope = finalize_prefill_run(
                root, plan=plan, run_id="prefill-run", case_results=[result]
            )

            parsed = parse_result_envelope(envelope)
            self.assertEqual("prefill", parsed.mode)
            self.assertEqual("completed", parsed.status.value)
            self.assertTrue((root / "results" / "result.json").is_file())
            self.assertTrue((root / "results" / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
