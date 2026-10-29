import io
import logging
import tarfile
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.publish.archive import (
    ArchiveSelectionError,
    OssArchivePublisher,
    OssCredentials,
)


class _FakeUploader:
    def __init__(self):
        self.calls = []
        self.members = []

    def upload(self, archive, destination, credentials):
        with tarfile.open(archive, "r:gz") as stream:
            self.members = sorted(stream.getnames())
        self.calls.append((destination, credentials))
        return destination


class _FailingUploader:
    def upload(self, archive, destination, credentials):
        raise RuntimeError(
            f"upload rejected id={credentials.key_id} secret={credentials.key_secret}"
        )


class ArchivePublisherTest(unittest.TestCase):
    def _run(self, root):
        files = {
            "manifest.json": b'{"stage":"completed"}\n',
            "results/result.json": b'{"status":"completed"}\n',
            "results/summary.json": b'{"planned_cases":1}\n',
            "reports/prefill_report.json": b'{"report_kind":"formal"}\n',
            "logs/engine.log": b"engine output\n",
        }
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return files

    def test_fake_upload_receives_only_explicitly_configured_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root)
            uploader = _FakeUploader()
            publisher = OssArchivePublisher(
                uploader=uploader,
                environment={
                    "MAGA_OSS_KEY_ID": "runtime-id",
                    "MAGA_OSS_KEY_SECRET": "runtime-secret",
                    "OSS_ENDPOINT": "oss.example.test",
                },
            )

            outcome = publisher.publish(
                root,
                destination="oss://perf-bucket/perf-test",
                artifacts=("results/result.json", "reports/prefill_report.json"),
                run_id="run-001",
            )

            self.assertEqual(
                ["reports/prefill_report.json", "results/result.json"],
                uploader.members,
            )
            self.assertEqual(
                "oss://perf-bucket/perf-test/run-001.tar.gz", uploader.calls[0][0]
            )
            self.assertEqual("succeeded", outcome["status"])
            self.assertEqual(
                ["reports/prefill_report.json", "results/result.json"],
                outcome["artifacts"],
            )
            self.assertNotIn("runtime-id", repr(outcome))
            self.assertNotIn("runtime-secret", repr(outcome))
            self.assertIsInstance(uploader.calls[0][1], OssCredentials)

    def test_selection_rejects_missing_duplicate_directory_and_escape_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root)
            publisher = OssArchivePublisher(
                uploader=_FakeUploader(),
                environment={
                    "MAGA_OSS_KEY_ID": "id",
                    "MAGA_OSS_KEY_SECRET": "secret",
                },
            )
            invalid = [
                (("missing.json",), "missing"),
                (("manifest.json", "manifest.json"), "duplicate"),
                (("results",), "regular file"),
                (("../outside.json",), "relative"),
                ((str(root / "manifest.json"),), "relative"),
            ]

            for artifacts, diagnostic in invalid:
                with self.subTest(artifacts=artifacts):
                    with self.assertRaisesRegex(ArchiveSelectionError, diagnostic):
                        publisher.publish(
                            root,
                            destination="oss://bucket/prefix",
                            artifacts=artifacts,
                            run_id="run-001",
                        )

    def test_credentials_come_only_from_environment_and_never_reach_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root)
            publisher = OssArchivePublisher(
                uploader=_FailingUploader(),
                environment={
                    "MAGA_OSS_KEY_ID": "sensitive-id",
                    "MAGA_OSS_KEY_SECRET": "sensitive-secret",
                },
            )
            captured = io.StringIO()
            handler = logging.StreamHandler(captured)
            logger = logging.getLogger("perf-test-archive-test")
            logger.handlers = [handler]
            logger.setLevel(logging.INFO)

            with self.assertRaisesRegex(RuntimeError, "\*\*\*") as raised:
                publisher.publish(
                    root,
                    destination="oss://bucket/prefix",
                    artifacts=("manifest.json",),
                    run_id="run-001",
                    logger=logger,
                )

            text = captured.getvalue() + str(raised.exception)
            self.assertNotIn("sensitive-id", text)
            self.assertNotIn("sensitive-secret", text)
            for relative in (
                "manifest.json",
                "results/result.json",
                "reports/prefill_report.json",
            ):
                self.assertNotIn("sensitive", (root / relative).read_text())

    def test_missing_runtime_credentials_fail_before_uploader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root)
            uploader = _FakeUploader()
            publisher = OssArchivePublisher(uploader=uploader, environment={})

            with self.assertRaisesRegex(RuntimeError, "MAGA_OSS_KEY_ID"):
                publisher.publish(
                    root,
                    destination="oss://bucket/prefix",
                    artifacts=("manifest.json",),
                    run_id="run-001",
                )

            self.assertEqual([], uploader.calls)


if __name__ == "__main__":
    unittest.main()
