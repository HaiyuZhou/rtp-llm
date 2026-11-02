import unittest

from rtp_llm.test.perf_test.aone.gang import GangTopology, derive_gang_topology
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import ConfigError, parse_config
from rtp_llm.test.perf_test.errors import PlanningError, ResourceError
from rtp_llm.test.perf_test.plan import (
    AOneResourcePlan,
    ExecutionPlan,
    build_execution_plan,
)


def _execution(**fields):
    execution = {
        "backend": "aone",
        "resource": "daily",
        "gpu_type": "H20",
        "max_parallel": 2,
    }
    execution.update(fields)
    return execution


def _config(execution, topology=None, mode="prefill"):
    if mode == "decode":
        workload = {
            "kind": "grid",
            "input_lengths": [128],
            "output_lengths": [16],
            "batch": {"strategy": "fixed", "sizes": [1]},
        }
    else:
        workload = {
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [128],
            "batch_sizes": [1],
        }
    return {
        "schema_version": 1,
        "model": {
            "label": "test-model",
            "type": "registered_model",
            "checkpoint": "/models/test",
            "topology": topology or {"tp": 2, "dp": 1, "pp": 1, "ep": 1},
        },
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 8,
        },
        "workload": workload,
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


def _build(execution, topology=None, mode="prefill"):
    return build_execution_plan(
        mode=mode,
        action="run",
        resolved=resolve_config(_config(execution, topology, mode)),
        cases=_cases(),
        result_directory="runs/example",
    )


class GangConfigTest(unittest.TestCase):
    def test_default_derives_single_machine(self):
        plan = _build(_execution(shard_count=2))

        resource = plan.resource_plan
        self.assertIsInstance(resource, AOneResourcePlan)
        self.assertEqual(1, resource.machine_count)
        self.assertEqual(2, resource.gpus_per_machine)
        self.assertEqual(2, resource.local_world_size)
        self.assertEqual(2, resource.shard_count)

    def test_gang_derivation(self):
        self.assertEqual(
            GangTopology(
                world_size=16,
                gpus_per_machine=8,
                machine_count=2,
                local_world_size=8,
            ),
            derive_gang_topology(16, 8),
        )

        plan = _build(
            _execution(gpus_per_machine=8),
            topology={"tp": 8, "dp": 2, "pp": 1, "ep": 1},
        )

        resource = plan.resource_plan
        self.assertEqual(2, resource.machine_count)
        self.assertEqual(8, resource.gpus_per_machine)
        self.assertEqual(8, resource.local_world_size)
        self.assertIsNone(resource.shard_count)

    def test_explicit_machine_count_rejected(self):
        with self.assertRaisesRegex(ConfigError, "execution.machine_count"):
            parse_config(_config(_execution(shard_count=2, machine_count=2)))

    def test_invalid_gpus_per_machine_rejected(self):
        for value in (0, -1, "8", True):
            with self.subTest(value=value):
                payload = _config(_execution(shard_count=2, gpus_per_machine=value))
                with self.assertRaisesRegex(ConfigError, "execution.gpus_per_machine"):
                    parse_config(payload)

    def test_oversized_gpus_per_machine_is_noop(self):
        topology = {"tp": 2, "dp": 2, "pp": 1, "ep": 1}

        plan = _build(_execution(shard_count=2, gpus_per_machine=8), topology)
        baseline = _build(_execution(shard_count=2), topology)

        self.assertEqual(1, plan.resource_plan.machine_count)
        self.assertEqual(baseline.resource_plan, plan.resource_plan)

    def test_oversized_gpus_per_machine_still_requires_shard_count(self):
        topology = {"tp": 2, "dp": 2, "pp": 1, "ep": 1}

        with self.assertRaisesRegex(PlanningError, "execution.shard_count is required"):
            _build(_execution(gpus_per_machine=8), topology)

    def test_non_divisible_world_size_rejected(self):
        topology = {"tp": 5, "dp": 2, "pp": 1, "ep": 1}

        with self.assertRaisesRegex(PlanningError, "world_size 10.*machine_count 3"):
            _build(_execution(gpus_per_machine=4), topology)

    def test_shard_count_rejected_in_gang_mode(self):
        topology = {"tp": 8, "dp": 2, "pp": 1, "ep": 1}

        with self.assertRaisesRegex(PlanningError, "mutually exclusive"):
            _build(_execution(gpus_per_machine=8, shard_count=2), topology)

    def test_shard_count_still_required_single_machine(self):
        with self.assertRaisesRegex(ConfigError, "execution.shard_count"):
            parse_config(_config(_execution()))

    def test_max_parallel_below_machine_count_rejected(self):
        topology = {"tp": 8, "dp": 2, "pp": 1, "ep": 1}

        with self.assertRaisesRegex(ResourceError, "max_parallel"):
            _build(_execution(gpus_per_machine=8, max_parallel=1), topology)

    def test_gang_resource_plan_round_trips(self):
        plan = _build(
            _execution(gpus_per_machine=8),
            topology={"tp": 8, "dp": 2, "pp": 1, "ep": 1},
        )

        restored = ExecutionPlan.from_dict(plan.to_dict())

        self.assertEqual(plan, restored)
        self.assertIsNone(restored.resource_plan.shard_count)
        self.assertEqual(2, restored.resource_plan.machine_count)
        self.assertEqual(8, restored.resource_plan.local_world_size)

    def test_single_machine_decode_aone_is_rejected_in_planning(self):
        with self.assertRaisesRegex(PlanningError, "gpus_per_machine.*machine_count"):
            _build(_execution(shard_count=2), mode="decode")

        # An oversized gpus_per_machine that still derives machine_count == 1
        # is rejected the same way.
        with self.assertRaisesRegex(PlanningError, "gpus_per_machine.*machine_count"):
            _build(
                _execution(shard_count=2, gpus_per_machine=8),
                topology={"tp": 2, "dp": 2, "pp": 1, "ep": 1},
                mode="decode",
            )

    def test_gang_decode_aone_passes_planning(self):
        plan = _build(
            _execution(gpus_per_machine=8),
            topology={"tp": 8, "dp": 2, "pp": 1, "ep": 1},
            mode="decode",
        )

        self.assertEqual(2, plan.resource_plan.machine_count)

    def test_gang_gpu_type_requires_per_host_count(self):
        topology = {"tp": 8, "dp": 2, "pp": 1, "ep": 1}

        # Gang shards request full hosts, so the GPU type must be registered
        # in AONE_GPU_HOSTS with a per-host GPU count.
        with self.assertRaisesRegex(ResourceError, "per-host GPU count"):
            _build(_execution(gpu_type="H900", gpus_per_machine=8), topology)

        # Single-machine execution keeps the platform-registry rejection.
        with self.assertRaisesRegex(ResourceError, "unsupported GPU type"):
            _build(_execution(gpu_type="H900", shard_count=2))

    def test_prefill_aone_passes_in_both_modes(self):
        single_machine = _build(_execution(shard_count=2))
        gang = _build(
            _execution(gpus_per_machine=8),
            topology={"tp": 8, "dp": 2, "pp": 1, "ep": 1},
        )

        self.assertEqual(1, single_machine.resource_plan.machine_count)
        self.assertEqual(2, single_machine.resource_plan.shard_count)
        self.assertEqual(2, gang.resource_plan.machine_count)
        self.assertIsNone(gang.resource_plan.shard_count)


if __name__ == "__main__":
    unittest.main()
