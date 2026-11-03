import json
import re
import shlex
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.cli import parse_cli
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases

ROOT = Path(__file__).resolve().parent
DOCS = tuple(sorted((ROOT / "docs").glob("*.md")))


class UnifiedDocsTest(unittest.TestCase):
    def test_prefill_docs_cover_offline_alias_and_grouped_grids(self):
        prefill = (ROOT / "docs" / "prefill.md").read_text(encoding="utf-8")
        reference = (ROOT / "docs" / "reference.md").read_text(encoding="utf-8")
        self.assertIn("--skip-test", prefill)
        self.assertIn("request_groups", prefill)
        self.assertIn("workload.grids", prefill)
        self.assertIn("prefill_grouped_local.json", prefill)
        self.assertIn("prefill_multi_grid_local.json", prefill)
        self.assertIn("--skip-test", reference)

    def test_grouped_and_multi_grid_examples_plan(self):
        examples = ROOT / "examples"
        for filename in ("prefill_grouped_local.json", "prefill_multi_grid_local.json"):
            with self.subTest(filename=filename):
                config = parse_config(json.loads((examples / filename).read_text()))
                cases = plan_prefill_cases(
                    config.workload, config.engine, dp_size=config.model.topology.dp
                )
                self.assertTrue(cases)
                self.assertTrue(any("request_groups" in case for case in cases))
                if filename == "prefill_multi_grid_local.json":
                    self.assertEqual(
                        ["cold", "mixed"],
                        list(dict.fromkeys(case["grid_name"] for case in cases)),
                    )

    def test_reference_covers_capabilities_layout_exit_codes_and_ci(self):
        reference = (ROOT / "docs" / "reference.md").read_text(encoding="utf-8")
        for text in (
            "Capability matrix",
            "Run directory",
            "Exit codes",
            "results/cases/<case_id>.json",
            "unified_prefill_ci_test",
            "the AOne backend runs",
            "decode only as a multi-machine gang",
            "gang only",
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
