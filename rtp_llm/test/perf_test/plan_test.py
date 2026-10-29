import copy
import dataclasses
import unittest

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.errors import PlanningError, ResourceError
from rtp_llm.test.perf_test.plan import (
    AOneResourcePlan,
    ExecutionPlan,
    LocalResourcePlan,
    build_execution_plan,
    case_id,
)


def _config(backend="local"):
    execution = {"backend": "local", "devices": [2, 3]}
    if backend == "aone":
        execution = {
            "backend": "aone",
            "resource": "gpu",
            "gpu_type": "H20",
            "max_parallel": 2,
            "shard_count": 2,
        }
    return {
        "schema_version": 1,
        "model": {
            "label": "test-model",
            "type": "registered_model",
            "checkpoint": "/models/test",
            "topology": {"tp": 2, "dp": 1, "pp": 1, "ep": 2},
        },
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 8,
        },
        "workload": {
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [128],
            "batch_sizes": [1],
        },
        "execution": execution,
        "output": {},
        "regression": {},
        "publish": {},
    }


def _cases():
    return [
        {"input_len": 1024, "cache_len": 512, "batch_size": 1},
        {"batch_size": 1, "cache_len": 0, "input_len": 512},
    ]


class ExecutionPlanTest(unittest.TestCase):
    def test_case_id_uses_only_mode_and_canonical_identity(self):
        left = {"input_len": 1024, "cache_len": 512, "batch_size": 1}
        right = {"batch_size": 1, "cache_len": 512, "input_len": 1024}

        self.assertEqual(case_id("prefill", left), case_id("prefill", right))
        self.assertNotEqual(case_id("prefill", left), case_id("decode", left))
        self.assertNotEqual(
            case_id("prefill", left),
            case_id("prefill", {**left, "cache_len": 0}),
        )

    def test_plan_is_input_order_independent_and_deeply_immutable(self):
        resolved = resolve_config(_config())
        forward = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=resolved,
            cases=_cases(),
            result_directory="runs/example",
        )
        reverse = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=resolved,
            cases=list(reversed(_cases())),
            result_directory="runs/example",
        )

        self.assertEqual(forward, reverse)
        self.assertEqual(
            sorted(case.case_id for case in forward.cases),
            [case.case_id for case in forward.cases],
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            forward.mode = "decode"
        with self.assertRaises(TypeError):
            forward.resolved_config["model"]["label"] = "changed"
        with self.assertRaises(AttributeError):
            forward.cases.append("changed")

    def test_plan_serialization_round_trips(self):
        plan = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=resolve_config(_config()),
            cases=_cases(),
            result_directory="runs/example",
        )

        restored = ExecutionPlan.from_dict(plan.to_dict())

        self.assertEqual(plan, restored)
        self.assertEqual(plan.to_json(), restored.to_json())
        self.assertIsInstance(restored.resource_plan, LocalResourcePlan)
        self.assertEqual((2, 3), restored.resource_plan.devices)
        self.assertEqual(2, restored.resource_plan.world_size)

    def test_plan_deserialization_revalidates_resource_plan(self):
        plan = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=resolve_config(_config()),
            cases=_cases(),
            result_directory="runs/example",
        ).to_dict()
        plan["resource_plan"]["devices"] = [2]

        with self.assertRaisesRegex(ResourceError, "requires 2"):
            ExecutionPlan.from_dict(plan)

    def test_duplicate_case_identity_is_rejected(self):
        case = _cases()[0]
        with self.assertRaisesRegex(PlanningError, "duplicate case") as context:
            build_execution_plan(
                mode="prefill",
                action="run",
                resolved=resolve_config(_config()),
                cases=[case, copy.deepcopy(case)],
                result_directory="runs/example",
            )
        self.assertEqual(3, context.exception.exit_code)

    def test_local_resource_plan_requires_exact_unique_device_count(self):
        for devices, diagnostic in (([0], "requires 2"), ([0, 0], "unique")):
            with self.subTest(devices=devices):
                payload = _config()
                payload["execution"]["devices"] = devices
                with self.assertRaisesRegex(ResourceError, diagnostic) as context:
                    build_execution_plan(
                        mode="prefill",
                        action="run",
                        resolved=resolve_config(payload),
                        cases=_cases(),
                        result_directory="runs/example",
                    )
                self.assertEqual(3, context.exception.exit_code)

    def test_local_devices_default_to_topology_world_size(self):
        payload = _config()
        payload["execution"].pop("devices")
        payload["model"]["topology"] = {"tp": 2, "dp": 2, "pp": 2, "ep": 4}

        plan = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=resolve_config(payload),
            cases=_cases(),
            result_directory="runs/example",
        )

        self.assertEqual(8, plan.resource_plan.world_size)
        self.assertEqual(tuple(range(8)), plan.resource_plan.devices)

    def test_aone_resource_plan_validates_shards_and_parallelism(self):
        plan = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=resolve_config(_config("aone")),
            cases=_cases(),
            result_directory="runs/example",
        )
        self.assertIsInstance(plan.resource_plan, AOneResourcePlan)
        self.assertEqual(2, plan.resource_plan.gpus_per_shard)

        too_parallel = _config("aone")
        too_parallel["execution"].update(max_parallel=3, shard_count=2)
        with self.assertRaisesRegex(ResourceError, "max_parallel"):
            build_execution_plan(
                mode="prefill",
                action="run",
                resolved=resolve_config(too_parallel),
                cases=_cases(),
                result_directory="runs/example",
            )

        too_many_shards = _config("aone")
        too_many_shards["execution"].update(max_parallel=1, shard_count=3)
        with self.assertRaisesRegex(ResourceError, "case count"):
            build_execution_plan(
                mode="prefill",
                action="run",
                resolved=resolve_config(too_many_shards),
                cases=_cases(),
                result_directory="runs/example",
            )


if __name__ == "__main__":
    unittest.main()
