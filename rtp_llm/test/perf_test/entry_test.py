import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "perf_test"
LOCAL = ROOT / "rtp_llm" / "test" / "perf_test" / "examples" / "prefill_local.json"
AONE = ROOT / "rtp_llm" / "test" / "perf_test" / "examples" / "prefill_aone.json"


class UnifiedEntryTest(unittest.TestCase):
    def _run(self, *arguments, cwd=None):
        return subprocess.run(
            [str(SCRIPT), *arguments],
            cwd=cwd or ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_repository_entry_help(self):
        completed = self._run("--help")

        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("--mode", completed.stdout)
        self.assertIn("--action", completed.stdout)
        self.assertNotIn("--partial", completed.stdout)

    def test_local_dry_run_prints_plan_without_creating_result_root(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = self._run(
                "--mode=prefill",
                "--action=run",
                f"--config={LOCAL}",
                "--dry-run",
                cwd=directory,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            preview = json.loads(completed.stdout)
            self.assertTrue(preview["dry_run"])
            self.assertEqual("local", preview["backend"])
            self.assertGreater(preview["cases"]["count"], 0)
            self.assertFalse((Path(directory) / "perf_test_results").exists())

    def test_aone_prefill_dry_run_includes_targets_and_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = self._run(
                "--mode=prefill",
                "--action=run",
                f"--config={AONE}",
                "--dry-run",
                cwd=directory,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            preview = json.loads(completed.stdout)
            self.assertEqual("aone", preview["backend"])
            self.assertGreater(preview["shards"]["count"], 0)
            for shard in preview["shards"]["items"]:
                self.assertTrue(shard["case_ids"])
                self.assertTrue(shard["target"].startswith("//generated/perf_test:"))
            self.assertFalse((Path(directory) / "perf_test_results").exists())

    def test_bazel_main_module_uses_the_same_help_contract(self):
        completed = subprocess.run(
            [
                "python",
                "-m",
                "rtp_llm.test.perf_test.main",
                "--help",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("--dry-run", completed.stdout)


if __name__ == "__main__":
    unittest.main()
