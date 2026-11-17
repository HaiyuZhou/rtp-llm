"""Tests for aone gang_package_test."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rtp_llm.test.perf_test.aone.gang_worker import load_gang_plan
from rtp_llm.test.perf_test.execution.aone_gang import BROKER_TOKEN_ENV
from rtp_llm.test.perf_test.tests.support.gang import RUN_ID, _gang_source, _setup_run
from rtp_llm.test.perf_test.tests.support.paths import PERF_TEST_ROOT


class GangPackageTest(unittest.TestCase):
    def test_machine_shards_request_full_hosts_even_with_partial_gpu_use(self):
        from rtp_llm.test.perf_test.aone.package import generate_gang_package

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _gang_source()
            source["execution"]["gpus_per_machine"] = 4
            source["execution"]["max_parallel"] = 4
            plan = _setup_run(root / "run", source)
            package = generate_gang_package(root / "generated", plan, run_id=RUN_ID)
            actions = []
            namespace = {
                "native": SimpleNamespace(
                    py_test=lambda **kwargs: actions.append(kwargs)
                ),
                "load": lambda *args: None,
                "test_suite": lambda **kwargs: None,
            }
            exec((PERF_TEST_ROOT / "aone_defs.bzl").read_text(), namespace)
            exec(package.build_file.read_text(), namespace)
            self.assertEqual(("//generated:gang",), package.targets)
            self.assertEqual(1, len(actions))
            action = actions[0]
            self.assertEqual(4, action["shard_count"])
            self.assertEqual("8", action["env"]["GPU_COUNT"])
            self.assertNotIn("WORLD_SIZE", action["env"])
            self.assertEqual(
                {"gpu": "H20", "gpu_count": "8"}, action["exec_properties"]
            )
            self.assertEqual(
                "//rtp_llm/test/perf_test:aone/gang_worker.py", action["main"]
            )
            self.assertTrue(
                {
                    "resolved_config.json",
                    "resolved_workload.json",
                    "gang_plan.json",
                }.issubset(action["data"])
            )
            self.assertIn(BROKER_TOKEN_ENV, action["env_inherit"])
            self.assertNotIn(BROKER_TOKEN_ENV, " ".join(action["args"]))
            payload = load_gang_plan(package.shard_plan)
            self.assertEqual(
                (16, 4, 4, 8),
                tuple(
                    payload[key]
                    for key in (
                        "world_size",
                        "machine_count",
                        "local_world_size",
                        "gpus_per_host",
                    )
                ),
            )


if __name__ == "__main__":
    unittest.main()
