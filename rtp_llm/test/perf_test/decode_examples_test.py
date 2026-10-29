import json
import re
import shlex
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.capabilities import validate_capability
from rtp_llm.test.perf_test.cli import parse_cli
from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.decode.distribution import plan_decode_distribution_cases
from rtp_llm.test.perf_test.decode.planner import plan_decode_grid_cases
from rtp_llm.test.perf_test.dry_run import create_preview
from rtp_llm.test.perf_test.errors import CapabilityError
from rtp_llm.test.perf_test.plan import build_execution_plan

ROOT = Path(__file__).resolve().parent
DOC = ROOT / "docs" / "decode.md"


class DecodeExamplesTest(unittest.TestCase):
    def test_documented_grid_distribution_and_tpot_dry_runs_are_plannable(self):
        markdown = DOC.read_text(encoding="utf-8")
        blocks = re.findall(r"```bash\n(.*?)```", markdown, flags=re.DOTALL)
        self.assertEqual(3, len(blocks))
        kinds = []
        for index, block in enumerate(blocks):
            command = shlex.split(block.replace("\\\n", " "))
            self.assertEqual("./tools/perf_test", command[0])
            request = parse_cli(command[1:])
            self.assertEqual(
                ("decode", "run", True), (request.mode, request.action, request.dry_run)
            )
            config_path = ROOT.parent.parent.parent / request.config
            source = json.loads(config_path.read_text(encoding="utf-8"))
            parsed = parse_config(source)
            resolved = load_resolved_config(config_path)
            dp_size = parsed.model.topology.dp
            if parsed.workload.kind == "grid":
                cases = plan_decode_grid_cases(
                    parsed.workload, parsed.engine, dp_size=dp_size
                )
            else:
                cases = plan_decode_distribution_cases(
                    resolved.resolved_config["workload"],
                    parsed.engine,
                    dp_size=dp_size,
                )
                self.assertIn("distribution", resolved.resolved_config["workload"])
            plan = build_execution_plan(
                mode="decode",
                action="run",
                resolved=resolved,
                cases=cases,
                result_directory=f"perf_test_results/decode-preview-{index}",
            )
            preview = create_preview(plan)
            self.assertTrue(preview["dry_run"])
            self.assertEqual(len(cases), preview["cases"]["count"])
            kinds.append(parsed.workload.kind)
        self.assertEqual(["grid", "distribution", "grid"], kinds)

    def test_decode_follow_up_actions_and_aone_fail_before_side_effects(self):
        side_effects = [Mock() for _ in range(4)]

        def operation(action, backend):
            validate_capability("decode", action, backend)
            for effect in side_effects:
                effect()

        for action, backend in (
            ("resume", "local"),
            ("profile", "local"),
            ("report", "local"),
            ("run", "aone"),
        ):
            with self.subTest(action=action, backend=backend):
                with self.assertRaises(CapabilityError) as context:
                    operation(action, backend)
                self.assertEqual(2, context.exception.exit_code)
        for effect in side_effects:
            effect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
