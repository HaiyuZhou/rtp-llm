import re
import shlex
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.cli import parse_cli

ROOT = Path(__file__).resolve().parent
DOCS = tuple(sorted((ROOT / "docs").glob("*.md")))


class UnifiedDocsTest(unittest.TestCase):
    def test_reference_covers_capabilities_layout_exit_codes_and_ci(self):
        reference = (ROOT / "docs" / "reference.md").read_text(encoding="utf-8")
        for text in (
            "Capability matrix",
            "Run directory",
            "Exit codes",
            "results/cases/<case_id>.json",
            "unified_prefill_ci_test",
            "Decode is not supported by the AOne backend",
        ):
            self.assertIn(text, reference)

    def test_every_documented_perf_command_is_accepted_by_the_parser(self):
        commands = []
        for path in DOCS:
            markdown = path.read_text(encoding="utf-8")
            for block in re.findall(r"```bash\n(.*?)```", markdown, flags=re.DOTALL):
                command = shlex.split(block.replace("\\\n", " "))
                if command and command[0] == "./tools/perf_test":
                    commands.append((path.name, command))
        self.assertGreaterEqual(len(commands), 10)
        for name, command in commands:
            with self.subTest(document=name, command=command):
                parse_cli(command[1:])

    def test_current_docs_do_not_advertise_removed_interfaces(self):
        content = "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
        for removed in (
            "batch_decode_test.py",
            "test_entry.py",
            "tools/cache_perf",
            "--partial",
            "--cache_grid_json",
        ):
            self.assertNotIn(removed, content)


if __name__ == "__main__":
    unittest.main()
