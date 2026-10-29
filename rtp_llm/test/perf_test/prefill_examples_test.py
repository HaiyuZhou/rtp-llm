import json
import re
import shlex
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.cli import parse_cli
from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.dry_run import create_preview
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases

ROOT = Path(__file__).resolve().parent
DOC = ROOT / "docs" / "prefill.md"
EXAMPLE = ROOT / "examples" / "prefill_local.json"


def _documented_commands():
    markdown = DOC.read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", markdown, flags=re.DOTALL)
    return [shlex.split(block.replace("\\\n", " ")) for block in blocks]


class PrefillExamplesTest(unittest.TestCase):
    def test_local_example_is_strict_and_plannable(self):
        source = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        config = parse_config(source)

        cases = plan_prefill_cases(config.workload, config.engine)

        self.assertEqual("cache_grid", config.workload.kind)
        self.assertEqual("local", config.execution.backend)
        self.assertGreater(len(cases), 0)

    def test_every_documented_action_has_a_valid_dry_run_command(self):
        commands = _documented_commands()
        self.assertEqual(4, len(commands))
        requests = []
        for command in commands:
            self.assertEqual("./tools/perf_test", command[0])
            request = parse_cli(command[1:])
            self.assertTrue(request.dry_run)
            self.assertEqual("prefill", request.mode)
            requests.append(request)
        self.assertEqual(
            {"run", "resume", "profile", "report"},
            {request.action for request in requests},
        )

        run = next(request for request in requests if request.action == "run")
        self.assertEqual(
            "rtp_llm/test/perf_test/examples/prefill_local.json", run.config
        )
        resolved = load_resolved_config(EXAMPLE)
        source = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        config = parse_config(source)
        cases = plan_prefill_cases(config.workload, config.engine)
        plan = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=resolved,
            cases=cases,
            result_directory="perf_test_results/dry-run-preview",
        )

        preview = create_preview(plan)

        self.assertTrue(preview["dry_run"])
        self.assertEqual(len(cases), preview["cases"]["count"])


if __name__ == "__main__":
    unittest.main()
