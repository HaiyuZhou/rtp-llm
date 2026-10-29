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
DOC = ROOT / "docs" / "aone_prefill.md"
EXAMPLE = ROOT / "examples" / "prefill_aone.json"


class AOneExamplesTest(unittest.TestCase):
    def test_documented_dry_run_emits_targets_and_complete_ownership(self):
        markdown = DOC.read_text(encoding="utf-8")
        blocks = re.findall(r"```bash\n(.*?)```", markdown, flags=re.DOTALL)
        self.assertEqual(1, len(blocks))
        command = shlex.split(blocks[0].replace("\\\n", " "))
        self.assertEqual("./tools/perf_test", command[0])
        request = parse_cli(command[1:])
        self.assertTrue(request.dry_run)
        self.assertEqual("prefill", request.mode)
        self.assertEqual("run", request.action)

        source = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        config = parse_config(source)
        cases = plan_prefill_cases(config.workload, config.engine)
        plan = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=load_resolved_config(EXAMPLE),
            cases=cases,
            result_directory="perf_test_results/aone-preview",
        )

        preview = create_preview(plan)

        self.assertEqual("aone", preview["backend"])
        self.assertEqual(2, preview["shards"]["count"])
        self.assertEqual(
            [
                "//generated/perf_test:shard_0000",
                "//generated/perf_test:shard_0001",
            ],
            [
                command["argv"][-1]
                for command in preview["commands"]
                if command["stage"] == "build-shard"
            ],
        )
        owned = [
            case_id
            for shard in preview["shards"]["items"]
            for case_id in shard["case_ids"]
        ]
        self.assertEqual(sorted(case.case_id for case in plan.cases), sorted(owned))
        self.assertEqual(len(owned), len(set(owned)))


if __name__ == "__main__":
    unittest.main()
