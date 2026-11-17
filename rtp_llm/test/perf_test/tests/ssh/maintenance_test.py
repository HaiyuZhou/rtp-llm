import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.ssh.maintenance import maintain, read_deployment
from rtp_llm.test.perf_test.tests.ssh import executor_test as fixtures


class MaintenanceTest(unittest.TestCase):
    def test_relocated_deployment_supports_dry_run_without_rewriting_original_manifest(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            plan = fixtures.SshExecutorTest().setup_plan(directory)
            transport = fixtures.LocalTransport(plan)
            self.assertEqual(
                fixtures.SshExecutor(host_factory=lambda *args: transport)(
                    plan
                ).exit_code,
                0,
            )
            original = Path(plan.result_directory)
            original_bytes = (original / "manifest.json").read_bytes()
            relocated = Path(directory) / "relocated"
            shutil.copytree(original, relocated)
            with patch(
                "rtp_llm.test.perf_test.ssh.maintenance.RemoteHost.run",
                side_effect=AssertionError("no remote contact"),
            ):
                result = maintain(relocated, "cancel", dry_run=True)
            self.assertEqual(result["exit_code"], 0)
            self.assertIn("0", result["members"])
            self.assertEqual((original / "manifest.json").read_bytes(), original_bytes)

    def test_modified_host_is_rejected_before_any_network_call(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = fixtures.SshExecutorTest().setup_plan(directory)
            transport = fixtures.LocalTransport(plan)
            self.assertEqual(
                fixtures.SshExecutor(host_factory=lambda *args: transport)(
                    plan
                ).exit_code,
                0,
            )
            path = Path(plan.result_directory) / "execution/ssh/members.json"
            deployment = json.loads(path.read_text())
            deployment["members"][0]["ssh_target"] = "unexpected-machine"
            path.write_text(json.dumps(deployment))
            with self.assertRaisesRegex(ValueError, "host differs"):
                read_deployment(plan.result_directory)

    def test_invalid_member_indices_and_bound_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = fixtures.SshExecutorTest().setup_plan(directory)
            transport = fixtures.LocalTransport(plan)
            self.assertEqual(
                fixtures.SshExecutor(host_factory=lambda *args: transport)(
                    plan
                ).exit_code,
                0,
            )
            path = Path(plan.result_directory) / "execution/ssh/members.json"
            baseline = json.loads(path.read_text())
            for field, value in [
                ("member_index", -1),
                ("member_index", True),
                ("member_id", "1"),
                ("worker_path", "/different/worker.py"),
                ("descriptor_path", "/different/descriptor.json"),
            ]:
                with self.subTest(field=field, value=value):
                    modified = json.loads(json.dumps(baseline))
                    modified["members"][0][field] = value
                    path.write_text(json.dumps(modified))
                    with self.assertRaises(ValueError):
                        read_deployment(plan.result_directory)


if __name__ == "__main__":
    unittest.main()
