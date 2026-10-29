import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

from rtp_llm.test.perf_test.aone.package import GeneratedAOnePackage
from rtp_llm.test.perf_test.execution.aone_bazel import BazelAOneIntegration


class _Completed:
    def __init__(self, returncode=0):
        self.returncode = returncode


class _Process:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1
        return self.returncode


class _InterruptProcess:
    def __init__(self):
        self.wait_count = 0
        self.terminate_count = 0
        self.kill_count = 0

    def wait(self, timeout=None):
        self.wait_count += 1
        if self.wait_count == 1:
            raise KeyboardInterrupt()
        return 0

    def terminate(self):
        self.terminate_count += 1

    def kill(self):
        self.kill_count += 1


def _package(workspace: Path) -> GeneratedAOnePackage:
    root = workspace / "runs" / "run-1" / "workload" / "aone_package"
    root.mkdir(parents=True)
    return GeneratedAOnePackage(
        root=root,
        resolved_config=root / "resolved_config.json",
        resolved_workload=root / "resolved_workload.json",
        shard_plan=root / "shard_plan.json",
        build_file=root / "BUILD.bazel",
        targets=(
            "//runs/run-1/workload/aone_package:shard_0000",
            "//runs/run-1/workload/aone_package:shard_0001",
        ),
    )


def _payload(shard_id: str):
    return {"schema_version": 1, "shard_id": shard_id}


class BazelAOneIntegrationTest(unittest.TestCase):
    def test_builds_once_and_submits_all_shards_in_one_remote_test(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            package = _package(workspace)
            run_calls = []
            popen_calls = []
            process = _Process()

            def run(command, **kwargs):
                run_calls.append((command, kwargs))
                return _Completed()

            def popen(command, **kwargs):
                popen_calls.append((command, kwargs))
                return process

            integration = BazelAOneIntegration(
                workspace_root=workspace,
                resource="daily",
                gpu_type="L20D_DEV",
                max_parallel=2,
                build_jobs=7,
                startup_args=("--output_user_root=/cache/root",),
                common_args=("--disk_cache=/cache",),
                run_command=run,
                popen_command=popen,
            )

            integration.build(package)
            jobs = [
                integration.submit(target, f"shard-{index:04d}")
                for index, target in enumerate(package.targets)
            ]

            self.assertEqual(1, len(run_calls))
            build = run_calls[0][0]
            self.assertEqual(
                ["bazelisk", "--output_user_root=/cache/root", "build"],
                build[:3],
            )
            self.assertEqual(list(package.targets), build[3:5])
            self.assertIn("--config=cuda13", build)
            self.assertIn("--config=cicd", build)
            self.assertIn("--config=daily_aone_bazel_cache", build)
            self.assertNotIn("--config=daily_aone_bazel_remote", build)
            self.assertIn("--jobs=7", build)
            self.assertEqual(1, len(popen_calls))
            test = popen_calls[0][0]
            self.assertEqual(
                ["bazelisk", "--output_user_root=/cache/root", "test"], test[:3]
            )
            self.assertEqual(list(package.targets), test[3:5])
            self.assertIn("--config=daily_aone_bazel_remote", test)
            self.assertNotIn("--config=daily_aone_bazel_cache", test)
            self.assertIn("--strategy=TestRunner=remote", test)
            self.assertIn("--jobs=2", test)
            self.assertIn("--remote_download_outputs=all", test)
            self.assertEqual(package.targets, tuple(job.target for job in jobs))

    def test_collects_zip_and_directory_outputs_and_waits_once(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            package = _package(workspace)
            testlogs = workspace / "custom-testlogs"
            process = _Process(returncode=0)
            integration = BazelAOneIntegration(
                workspace_root=workspace,
                resource="online",
                gpu_type="SM103_CU13",
                max_parallel=1,
                testlogs_root=testlogs,
                run_command=lambda *_args, **_kwargs: _Completed(),
                popen_command=lambda *_args, **_kwargs: process,
            )
            integration.build(package)
            jobs = [
                integration.submit(target, f"shard-{index:04d}")
                for index, target in enumerate(package.targets)
            ]
            first = (
                testlogs
                / "runs/run-1/workload/aone_package"
                / "shard_0000"
                / "test.outputs"
            )
            first.mkdir(parents=True)
            with zipfile.ZipFile(first / "outputs.zip", "w") as archive:
                archive.writestr("perf_test_shard-0000/", b"")
                archive.writestr(
                    "perf_test_shard-0000/shard_result.json",
                    json.dumps(_payload("shard-0000")),
                )
            second = (
                testlogs
                / "runs/run-1/workload/aone_package"
                / "shard_0001"
                / "test.outputs"
                / "perf_test_shard-0001"
            )
            second.mkdir(parents=True)
            (second / "shard_result.json").write_text(
                json.dumps(_payload("shard-0001")), encoding="utf-8"
            )

            results = [
                integration.collect(job, f"shard-{index:04d}")
                for index, job in enumerate(jobs)
            ]

            self.assertEqual(
                ["shard-0000", "shard-0001"], [r["shard_id"] for r in results]
            )
            self.assertEqual(1, process.wait_count)

    def test_nonzero_remote_exit_marks_existing_results_failed_for_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            package = _package(workspace)
            testlogs = workspace / "bazel-testlogs"
            integration = BazelAOneIntegration(
                workspace_root=workspace,
                resource="daily",
                gpu_type="L20D_DEV",
                max_parallel=1,
                testlogs_root=testlogs,
                run_command=lambda *_args, **_kwargs: _Completed(),
                popen_command=lambda *_args, **_kwargs: _Process(returncode=9),
            )
            integration.build(package)
            job = integration.submit(package.targets[0], "shard-0000")
            result = (
                testlogs
                / "runs/run-1/workload/aone_package"
                / "shard_0000/test.outputs/perf_test_shard-0000/shard_result.json"
            )
            result.parent.mkdir(parents=True)
            payload = _payload("shard-0000")
            payload["status"] = "completed"
            payload["failure"] = None
            result.write_text(json.dumps(payload), encoding="utf-8")

            collected = integration.collect(job, "shard-0000")

            self.assertEqual("failed", collected["status"])
            self.assertEqual("BazelTestFailure", collected["failure"]["type"])
            self.assertIn("exited 9", collected["failure"]["message"])

    def test_rejects_unsafe_or_ambiguous_zip_members(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            package = _package(workspace)
            testlogs = workspace / "bazel-testlogs"
            integration = BazelAOneIntegration(
                workspace_root=workspace,
                resource="daily",
                gpu_type="L20D_DEV",
                max_parallel=1,
                testlogs_root=testlogs,
                run_command=lambda *_args, **_kwargs: _Completed(),
                popen_command=lambda *_args, **_kwargs: _Process(),
            )
            integration.build(package)
            job = integration.submit(package.targets[0], "shard-0000")
            outputs = (
                testlogs
                / "runs/run-1/workload/aone_package"
                / "shard_0000"
                / "test.outputs"
            )
            outputs.mkdir(parents=True)
            with zipfile.ZipFile(outputs / "outputs.zip", "w") as archive:
                archive.writestr("../shard_result.json", "{}")

            with self.assertRaisesRegex(ValueError, "unsafe path"):
                integration.collect(job, "shard-0000")

    def test_reports_remote_test_exit_when_shard_output_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            package = _package(workspace)
            integration = BazelAOneIntegration(
                workspace_root=workspace,
                resource="daily",
                gpu_type="L20D_DEV",
                max_parallel=1,
                run_command=lambda *_args, **_kwargs: _Completed(),
                popen_command=lambda *_args, **_kwargs: _Process(returncode=34),
            )
            integration.build(package)
            job = integration.submit(package.targets[0], "shard-0000")

            with self.assertRaisesRegex(RuntimeError, "exited 34"):
                integration.collect(job, "shard-0000")

    def test_rejects_absolute_generated_target_before_running_bazel(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            package = replace(_package(workspace), targets=("///tmp:shard_0000",))
            integration = BazelAOneIntegration(
                workspace_root=workspace,
                resource="daily",
                gpu_type="L20D_DEV",
                max_parallel=1,
                run_command=lambda *args, **kwargs: calls.append((args, kwargs)),
            )

            with self.assertRaisesRegex(ValueError, "invalid generated AOne target"):
                integration.build(package)

        self.assertEqual([], calls)

    def test_bounds_direct_and_archive_result_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            package = _package(workspace)
            testlogs = workspace / "bazel-testlogs"
            integration = BazelAOneIntegration(
                workspace_root=workspace,
                resource="daily",
                gpu_type="L20D_DEV",
                max_parallel=1,
                testlogs_root=testlogs,
                max_result_bytes=32,
                max_archive_members=2,
                max_archive_uncompressed_bytes=64,
                max_compression_ratio=2.0,
                run_command=lambda *_args, **_kwargs: _Completed(),
                popen_command=lambda *_args, **_kwargs: _Process(),
            )
            integration.build(package)
            job = integration.submit(package.targets[0], "shard-0000")
            outputs = (
                testlogs
                / "runs/run-1/workload/aone_package"
                / "shard_0000/test.outputs"
            )
            direct = outputs / "perf_test_shard-0000/shard_result.json"
            direct.parent.mkdir(parents=True)
            direct.write_bytes(b"{" + b" " * 64 + b"}")

            with self.assertRaisesRegex(ValueError, "size limit"):
                integration.collect(job, "shard-0000")

            direct.unlink()
            with zipfile.ZipFile(outputs / "outputs.zip", "w") as archive:
                archive.writestr("first", "1")
                archive.writestr("second", "2")
                archive.writestr("third", "3")
            with self.assertRaisesRegex(ValueError, "member limit"):
                integration.collect(job, "shard-0000")

            (outputs / "outputs.zip").unlink()
            with zipfile.ZipFile(outputs / "outputs.zip", "w") as archive:
                archive.writestr("first", "1" * 40)
                archive.writestr("second", "2" * 40)
            with self.assertRaisesRegex(ValueError, "uncompressed size limit"):
                integration.collect(job, "shard-0000")

    def test_rejects_excessive_result_compression_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            package = _package(workspace)
            testlogs = workspace / "bazel-testlogs"
            integration = BazelAOneIntegration(
                workspace_root=workspace,
                resource="daily",
                gpu_type="L20D_DEV",
                max_parallel=1,
                testlogs_root=testlogs,
                max_result_bytes=4096,
                max_archive_uncompressed_bytes=8192,
                max_compression_ratio=2.0,
                run_command=lambda *_args, **_kwargs: _Completed(),
                popen_command=lambda *_args, **_kwargs: _Process(),
            )
            integration.build(package)
            job = integration.submit(package.targets[0], "shard-0000")
            outputs = (
                testlogs
                / "runs/run-1/workload/aone_package"
                / "shard_0000/test.outputs"
            )
            outputs.mkdir(parents=True)
            with zipfile.ZipFile(
                outputs / "outputs.zip", "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr(
                    "perf_test_shard-0000/shard_result.json",
                    json.dumps({"padding": "x" * 1024}),
                )

            with self.assertRaisesRegex(ValueError, "compression ratio limit"):
                integration.collect(job, "shard-0000")

    def test_interrupt_terminates_remote_bazel_process(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            package = _package(workspace)
            process = _InterruptProcess()
            integration = BazelAOneIntegration(
                workspace_root=workspace,
                resource="daily",
                gpu_type="L20D_DEV",
                max_parallel=1,
                run_command=lambda *_args, **_kwargs: _Completed(),
                popen_command=lambda *_args, **_kwargs: process,
            )
            integration.build(package)
            job = integration.submit(package.targets[0], "shard-0000")

            with self.assertRaises(KeyboardInterrupt):
                integration.collect(job, "shard-0000")

            self.assertEqual(1, process.terminate_count)
            self.assertEqual(0, process.kill_count)

    def test_rejects_unknown_pool_before_running_bazel(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "supported GPU types"):
                BazelAOneIntegration(
                    workspace_root=directory,
                    resource="daily",
                    gpu_type="UNKNOWN",
                    max_parallel=1,
                    run_command=lambda *args, **kwargs: calls.append((args, kwargs)),
                )
        self.assertEqual([], calls)


if __name__ == "__main__":
    unittest.main()
