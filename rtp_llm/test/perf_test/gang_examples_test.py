import io
import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.dry_run import create_preview
from rtp_llm.test.perf_test.entry import main
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
    def test_ci_prefill_plans_bounded_unique_concurrent_gang_batches(self):
        path = ROOT / "examples" / "prefill_aone_ci.json"
        self.assertTrue(path.is_file(), "CI prefill fixture is missing")
        config = parse_config(json.loads(path.read_text(encoding="utf-8")))
        workload = config.workload
        self.assertEqual("dashsc_input_ids", workload.transport)
        self.assertEqual(64, workload.probe_block_size)
        self.assertIsNotNone(workload.workspace_tokens)
        self.assertFalse(workload.grids)
        cases = plan_prefill_cases(
            workload, config.engine, dp_size=config.model.topology.dp
        )
        self.assertGreaterEqual(len(cases), 30)
        self.assertTrue(any(case["batch_size"] == 4 for case in cases))
        self.assertEqual(len(workload.request_batches), len(cases))
        geometries = set()
        for case in cases:
            self.assertEqual(
                "concurrent_grouped_batch_v1", case["measurement_contract"]
            )
            self.assertGreater(case["batch_size"], 1)
            groups = case["request_groups"]
            geometries.add(
                tuple(
                    sorted((g["count"], g["input_len"], g["cache_len"]) for g in groups)
                )
            )
            self.assertTrue(any(g["cache_len"] == 0 for g in groups))
            self.assertTrue(any(g["cache_len"] > 0 for g in groups))
            self.assertGreater(
                len({(g["input_len"], g["cache_len"]) for g in groups}), 1
            )
            self.assertLessEqual(
                sum(g["count"] * (g["input_len"] + 1) for g in groups),
                config.engine.max_batch_tokens,
            )
            self.assertLessEqual(
                sum(g["count"] * ((g["input_len"] + 64) // 64 * 64) for g in groups),
                workload.workspace_tokens,
            )
        self.assertEqual(len(cases), len(geometries))
        self.assertEqual((2, 1), (config.model.topology.tp, config.model.topology.dp))
        plan = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=load_resolved_config(path),
            cases=cases,
            result_directory="perf_test_results/prefill-ci-preview",
        )
        preview = create_preview(plan)
        self.assertEqual("aone", preview["backend"])
        self.assertEqual(len(cases), preview["cases"]["count"])
        self.assertEqual(2, preview["gang"]["machine_count"])
        self.assertEqual(1, preview["gang"]["gpus_per_machine"])
        self.assertEqual(TWO_ONE_GPU_MEMBERS, preview["gang"]["members"])
        self.assertEqual(
            ["build", "remote-test"],
            [c["stage"] for c in preview["commands"]],
        )
        for command in preview["commands"]:
            self.assertIn("//generated/perf_test:gang", command["argv"])
        self.assertIn("--jobs=2", preview["commands"][1]["argv"])

    def test_multi_dp_batch_four_dry_run_rejects_without_output(self):
        for example in (
            "prefill_local.json",
            "prefill_aone.json",
            "prefill_aone_gang.json",
        ):
            with self.subTest(
                example=example
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = json.loads((ROOT / "examples" / example).read_text())
                source["model"]["topology"] = {"tp": 1, "dp": 2, "pp": 1, "ep": 1}
                source["workload"]["batch_sizes"] = [4]
                if source["execution"]["backend"] == "local":
                    source["execution"]["devices"] = [0, 1]
                source["output"]["result_root"] = str(root / "results")
                config = root / "config.json"
                config.write_text(json.dumps(source))
                stdout, stderr = io.StringIO(), io.StringIO()
                code = main(
                    [
                        "--mode=prefill",
                        "--action=run",
                        f"--config={config}",
                        "--dry-run",
                    ],
                    stdout=stdout,
                    stderr=stderr,
                )
                self.assertNotEqual(0, code)
                self.assertIn("DP", stderr.getvalue())
                self.assertIn("batch", stderr.getvalue())
                self.assertEqual(
                    ["config.json"], sorted(item.name for item in root.iterdir())
                )

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
