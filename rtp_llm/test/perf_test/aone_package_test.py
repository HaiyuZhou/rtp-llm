import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.aone.package import generate_aone_package
from rtp_llm.test.perf_test.aone.worker import load_remote_inputs
from rtp_llm.test.perf_test.aone.worker import main as worker_main
from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.plan import build_execution_plan


def _config(checkpoint):
    return {
        "schema_version": 1,
        "model": {
            "label": "remote-prefill",
            "type": "registered_model",
            "checkpoint": str(checkpoint),
            "tokenizer": str(checkpoint),
            "topology": {"tp": 2},
        },
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
            "cache_lengths": [0, 1024],
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


def _plan(config_path, result_directory):
    cases = [
        {
            "input_len": 512,
            "cache_len": 0,
            "compute_len": 512,
            "batch_size": 1,
        },
        {
            "input_len": 1536,
            "cache_len": 1024,
            "compute_len": 512,
            "batch_size": 1,
        },
    ]
    return build_execution_plan(
        mode="prefill",
        action="run",
        resolved=load_resolved_config(config_path),
        cases=cases,
        result_directory=str(result_directory),
    )


class AOnePackageTest(unittest.TestCase):
    def test_generated_package_contains_only_declared_runfile_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "runtime-model"
            checkpoint.mkdir()
            (checkpoint / "weights.bin").write_bytes(b"model-data-must-not-be-packaged")
            config_path = root / "submit-host-only.json"
            config_path.write_text(json.dumps(_config(checkpoint)), encoding="utf-8")
            plan = _plan(config_path, root / "results")
            ownership = [[plan.cases[0].case_id], [plan.cases[1].case_id]]

            package = generate_aone_package(
                root / "generated", plan, run_id="run-123", ownership=ownership
            )

            self.assertEqual(
                {
                    "BUILD.bazel",
                    "resolved_config.json",
                    "resolved_workload.json",
                    "shard_plan.json",
                },
                {path.name for path in package.root.iterdir()},
            )
            build = package.build_file.read_text(encoding="utf-8")
            self.assertEqual(2, build.count("perf_test_aone_shard("))
            self.assertIn('resolved_config = "resolved_config.json"', build)
            self.assertIn('resolved_workload = "resolved_workload.json"', build)
            self.assertIn('shard_plan = "shard_plan.json"', build)
            self.assertNotIn(str(config_path), build)
            macro = (Path(__file__).parent / "defs.bzl").read_text(encoding="utf-8")
            for option in (
                "--resolved-config",
                "--resolved-workload",
                "--shard-plan",
            ):
                self.assertIn(f'"{option}=$(location "', macro)

            normalized = json.loads(package.resolved_config.read_text())
            self.assertNotIn("workload", normalized)
            self.assertEqual(str(checkpoint), normalized["model"]["checkpoint"])
            self.assertFalse((package.root / "weights.bin").exists())
            for path in package.root.iterdir():
                self.assertNotIn(b"model-data-must-not-be-packaged", path.read_bytes())

    def test_remote_loader_works_from_clean_runfiles_without_source_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "submit-host-only.json"
            config_path.write_text(
                json.dumps(_config("/runtime/models/qwen")), encoding="utf-8"
            )
            plan = _plan(config_path, root / "results")
            ownership = [[plan.cases[0].case_id], [plan.cases[1].case_id]]
            package = generate_aone_package(
                root / "generated", plan, run_id="run-123", ownership=ownership
            )
            clean = root / "clean-runfiles"
            shutil.copytree(package.root, clean)
            config_path.unlink()

            previous = Path.cwd()
            try:
                os.chdir(clean)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exit_code = worker_main(
                        [
                            "--resolved-config=resolved_config.json",
                            "--resolved-workload=resolved_workload.json",
                            "--shard-plan=shard_plan.json",
                            "--shard-id=shard-0001",
                        ]
                    )
                loaded = load_remote_inputs(
                    resolved_config=Path("resolved_config.json"),
                    resolved_workload=Path("resolved_workload.json"),
                    shard_plan=Path("shard_plan.json"),
                    shard_id="shard-0001",
                )
            finally:
                os.chdir(previous)

            self.assertEqual(0, exit_code)
            self.assertEqual("shard-0001", json.loads(output.getvalue())["shard_id"])
            self.assertEqual("run-123", loaded.run_id)
            self.assertEqual(plan.config_fingerprint, loaded.config_fingerprint)
            self.assertEqual(plan.workload_fingerprint, loaded.workload_fingerprint)
            self.assertEqual((plan.cases[1].case_id,), loaded.owned_case_ids)
            self.assertEqual(str(root / "results"), plan.result_directory)
            self.assertNotIn(str(root / "results"), loaded.config)


if __name__ == "__main__":
    unittest.main()
