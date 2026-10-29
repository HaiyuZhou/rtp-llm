import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.cli import parse_cli
from rtp_llm.test.perf_test.entry import run_request


class EntryExecutionTest(unittest.TestCase):
    def test_run_directory_is_initialized_before_backend_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = root / "results"
            config = {
                "schema_version": 1,
                "model": {
                    "label": "model",
                    "type": "registered",
                    "checkpoint": "/m",
                },
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
                "output": {"result_root": str(result_root)},
                "regression": {},
                "publish": {},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            observed = {}

            def execute(plan):
                run_root = Path(plan.result_directory)
                observed["root"] = run_root
                observed["files"] = {
                    name: (run_root / name).is_file()
                    for name in (
                        "source_config.json",
                        "resolved_config.json",
                        "execution_plan.json",
                        "manifest.json",
                    )
                }
                return Mock(exit_code=0)

            request = parse_cli(
                [
                    "--mode=prefill",
                    "--action=run",
                    f"--config={config_path}",
                ]
            )
            run_request(request, executor_factory=lambda: {"local": execute})

            self.assertTrue(all(observed["files"].values()))
            source = json.loads(
                (observed["root"] / "source_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config, source)


if __name__ == "__main__":
    unittest.main()
