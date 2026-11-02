import json
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.dry_run import create_preview
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases

ROOT = Path(__file__).resolve().parent
EXAMPLES = {
    "decode": ROOT / "examples" / "decode_aone_gang.json",
    "prefill": ROOT / "examples" / "prefill_aone_gang.json",
}

# A TP2 gang with one GPU per machine spans two single-GPU members.
TWO_ONE_GPU_MEMBERS = [
    {"shard_index": 0, "world_ranks": [0], "is_master": True},
    {"shard_index": 1, "world_ranks": [1], "is_master": False},
]


def _planned_cases(mode, config, resolved):
    if mode == "decode":
        return plan_decode_cases(
            config.workload,
            config.engine,
            dp_size=config.model.topology.dp,
            resolved_workload=resolved.resolved_config["workload"],
        )
    return plan_prefill_cases(config.workload, config.engine)


class GangExamplesTest(unittest.TestCase):
    def test_gang_examples_parse_and_dry_run_two_machine_gangs(self):
        for mode, path in EXAMPLES.items():
            with self.subTest(mode=mode):
                source = json.loads(path.read_text(encoding="utf-8"))
                config = parse_config(source)
                resolved = load_resolved_config(path)
                cases = _planned_cases(mode, config, resolved)
                self.assertTrue(cases)

                plan = build_execution_plan(
                    mode=mode,
                    action="run",
                    resolved=resolved,
                    cases=cases,
                    result_directory=f"perf_test_results/{mode}-aone-gang-preview",
                )
                self.assertEqual(2, plan.resource_plan.machine_count)
                self.assertEqual(1, plan.resource_plan.local_world_size)

                preview = create_preview(plan)

                self.assertTrue(preview["dry_run"])
                self.assertEqual(mode, preview["mode"])
                self.assertEqual("aone", preview["backend"])
                self.assertEqual(len(cases), preview["cases"]["count"])
                self.assertEqual(
                    plan.to_dict()["resolved_config"], preview["normalized_config"]
                )
                self.assertNotIn("shards", preview)

                gang = preview["gang"]
                self.assertEqual(2, gang["machine_count"])
                self.assertEqual(1, gang["gpus_per_machine"])
                self.assertEqual(1, gang["local_world_size"])
                self.assertEqual(2, gang["world_size"])
                self.assertEqual(0, gang["master_shard_index"])
                self.assertEqual(TWO_ONE_GPU_MEMBERS, gang["members"])
                self.assertEqual("//generated/perf_test:gang", gang["target"])

                self.assertEqual(
                    ["build", "remote-test"],
                    [command["stage"] for command in preview["commands"]],
                )
                for command in preview["commands"]:
                    self.assertIn("//generated/perf_test:gang", command["argv"])
                self.assertIn("--jobs=2", preview["commands"][1]["argv"])


if __name__ == "__main__":
    unittest.main()
