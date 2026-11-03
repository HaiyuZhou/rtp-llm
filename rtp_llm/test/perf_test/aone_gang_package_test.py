import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.aone.gang_package import (
    build_gang_plan_payload,
    generate_gang_package,
)
from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import (
    AONE_GPU_HOSTS,
    AONE_GPU_PLATFORMS,
    build_execution_plan,
)


def _config(checkpoint):
    return {
        "schema_version": 1,
        "model": {
            "label": "remote-gang",
            "type": "registered_model",
            "checkpoint": str(checkpoint),
            "tokenizer": str(checkpoint),
            "topology": {"tp": 8, "dp": 2, "pp": 1, "ep": 1},
        },
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 2,
        },
        "workload": {
            "kind": "grid",
            "input_lengths": [128],
            "output_lengths": [16],
            "batch": {"strategy": "fixed", "sizes": [1]},
        },
        "execution": {
            "backend": "aone",
            "resource": "daily",
            "gpu_type": "H20",
            "gpus_per_machine": 8,
            "max_parallel": 2,
        },
        "output": {},
        "regression": {},
        "publish": {},
    }


def _cases():
    return [
        {"input_len": 1024, "cache_len": 512, "batch_size": 1},
        {"batch_size": 1, "cache_len": 0, "input_len": 512},
    ]


def _gang_plan(config_path, result_directory):
    return build_execution_plan(
        mode="decode",
        action="run",
        resolved=load_resolved_config(config_path),
        cases=_cases(),
        result_directory=str(result_directory),
    )


def _rule_body(macro, name):
    start = macro.index(f"def {name}(")
    end = macro.find("\ndef ", start + 1)
    return macro[start:] if end < 0 else macro[start:end]


class AOneGangPackageTest(unittest.TestCase):
    def test_every_gpu_platform_has_a_registered_per_host_count(self):
        for gpu_type in AONE_GPU_PLATFORMS:
            with self.subTest(gpu_type=gpu_type):
                self.assertIn(gpu_type, AONE_GPU_HOSTS)

    def test_build_gang_plan_payload_describes_the_gang(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "submit-host-only.json"
            config_path.write_text(
                json.dumps(_config("/runtime/models/qwen")), encoding="utf-8"
            )
            plan = _gang_plan(config_path, root / "results")

            payload = build_gang_plan_payload(plan, "run-123")

            self.assertEqual(
                {
                    "schema_version",
                    "run_id",
                    "config_fingerprint",
                    "workload_fingerprint",
                    "world_size",
                    "machine_count",
                    "local_world_size",
                    "gpus_per_machine",
                    "gpus_per_host",
                    "gpu_type",
                    "topology",
                    "master_shard_index",
                    "port_range",
                },
                set(payload),
            )
            self.assertEqual(1, payload["schema_version"])
            self.assertEqual("run-123", payload["run_id"])
            self.assertEqual(plan.config_fingerprint, payload["config_fingerprint"])
            self.assertEqual(plan.workload_fingerprint, payload["workload_fingerprint"])
            self.assertEqual(16, payload["world_size"])
            self.assertEqual(2, payload["machine_count"])
            self.assertEqual(8, payload["local_world_size"])
            self.assertEqual(8, payload["gpus_per_machine"])
            self.assertEqual(8, payload["gpus_per_host"])
            self.assertEqual("H20", payload["gpu_type"])
            self.assertEqual({"tp": 8, "dp": 2, "pp": 1, "ep": 1}, payload["topology"])
            self.assertEqual(0, payload["master_shard_index"])
            self.assertEqual({"start": 12333, "count": 8}, payload["port_range"])

    def test_generated_gang_package_keeps_only_declared_runfile_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "submit-host-only.json"
            config_path.write_text(
                json.dumps(_config("/runtime/models/qwen")), encoding="utf-8"
            )
            plan = _gang_plan(config_path, root / "results")

            package = generate_gang_package(root / "generated", plan, run_id="run-123")

            self.assertEqual(
                {
                    "BUILD.bazel",
                    "resolved_config.json",
                    "resolved_workload.json",
                    "gang_plan.json",
                },
                {path.name for path in package.root.iterdir()},
            )
            self.assertEqual(("//generated:gang",), package.targets)
            payload = json.loads(package.shard_plan.read_text(encoding="utf-8"))
            self.assertEqual(build_gang_plan_payload(plan, "run-123"), payload)
            normalized = json.loads(package.resolved_config.read_text(encoding="utf-8"))
            self.assertNotIn("workload", normalized)
            self.assertEqual("/runtime/models/qwen", normalized["model"]["checkpoint"])
            # Gang plans have no case shards: a serialized null shard_count
            # would fail the worker-side config re-parse, so the runfile must
            # omit the key entirely.
            self.assertNotIn("shard_count", normalized["execution"])
            build = package.build_file.read_text(encoding="utf-8")
            self.assertIn("perf_test_gang_shard(", build)
            self.assertIn('resolved_config = "resolved_config.json"', build)
            self.assertIn('resolved_workload = "resolved_workload.json"', build)
            self.assertIn('gang_plan = "gang_plan.json"', build)
            self.assertIn('gpu_type = "H20"', build)
            self.assertIn("gpus_per_host = 8", build)
            self.assertIn("machine_count = 2", build)
            self.assertNotIn(str(config_path), build)
            self.assertNotIn(str(root / "results"), build)

    def test_gang_shard_rule_requests_full_hosts_without_world_size(self):
        macro = (Path(__file__).parent / "defs.bzl").read_text(encoding="utf-8")
        rule = _rule_body(macro, "perf_test_gang_shard")
        self.assertIn("shard_count = machine_count", rule)
        self.assertIn('timeout = "eternal"', rule)
        self.assertIn('main = "//rtp_llm/test/perf_test:aone_gang_worker.py"', rule)
        self.assertIn('"//rtp_llm:sdk"', rule)
        self.assertIn('"--gang-plan=$(location "', rule)
        self.assertIn('tags = ["manual", gpu_type]', rule)
        self.assertIn('"gpu_count": str(gpus_per_host)', rule)
        env_start = rule.index("env_inherit = [")
        env_block = rule[env_start : rule.index("]", env_start)]
        for variable in (
            "PERF_TEST_BROKER_URL",
            "PERF_TEST_BROKER_TOKEN",
            "PERF_TEST_BROKER_RUN_ID",
        ):
            self.assertIn(variable, env_block)
        # The worker computes WORLD_SIZE from the gang plan; the rule must
        # not pin it like the single-machine shard rule does.
        self.assertNotIn("WORLD_SIZE", rule)

    def test_gang_package_rejects_invalid_run_id_and_package_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "submit-host-only.json"
            config_path.write_text(
                json.dumps(_config("/runtime/models/qwen")), encoding="utf-8"
            )
            plan = _gang_plan(config_path, root / "results")

            with self.assertRaisesRegex(PlanningError, "run_id"):
                build_gang_plan_payload(plan, "")
            with self.assertRaisesRegex(PlanningError, "run_id"):
                generate_gang_package(root / "generated", plan, run_id="")
            with self.assertRaisesRegex(
                PlanningError, "invalid generated Bazel package"
            ):
                generate_gang_package(
                    root / "generated",
                    plan,
                    run_id="run-123",
                    package="/absolute/escape",
                )


if __name__ == "__main__":
    unittest.main()
