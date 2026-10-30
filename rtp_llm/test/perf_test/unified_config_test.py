import copy
import unittest

from rtp_llm.test.perf_test.config_schema import (
    AOneExecutionConfig,
    ConfigError,
    DecodeGridWorkloadConfig,
    LocalExecutionConfig,
    PrefillWorkloadConfig,
    parse_config,
)
from rtp_llm.test.perf_test.decode.model import (
    DecodeProfilingConfig,
    FixedBatchConfig,
    TargetTPOTConfig,
)


def _base_config():
    return {
        "schema_version": 1,
        "model": {
            "label": "test-model",
            "type": "registered_model",
            "checkpoint": "/models/test",
            "tokenizer": "/models/tokenizer",
            "topology": {},
        },
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 8,
            "args": {},
            "env": {},
        },
        "workload": {
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [128],
            "batch_sizes": [1],
        },
        "execution": {"backend": "local"},
        "output": {},
        "regression": {},
        "publish": {},
    }


class UnifiedConfigTest(unittest.TestCase):
    def test_prefill_config_applies_nested_defaults(self):
        config = parse_config(_base_config())

        self.assertEqual(1, config.schema_version)
        self.assertEqual((1, 1, 1, 1), config.model.topology.as_tuple())
        self.assertIsInstance(config.workload, PrefillWorkloadConfig)
        self.assertEqual(3, config.workload.measure_runs)
        self.assertEqual(0, config.workload.min_cache_len)
        self.assertEqual(128, config.workload.min_compute_len)
        self.assertEqual(1, config.workload.seed)
        self.assertIsInstance(config.execution, LocalExecutionConfig)
        self.assertEqual((), config.execution.devices)
        self.assertEqual("perf_test_results", config.output.result_root)
        self.assertFalse(config.regression.enabled)
        self.assertFalse(config.publish.oss.enabled)
        self.assertFalse(config.publish.odps.enabled)

    def test_decode_grid_preserves_declared_extension_maps(self):
        payload = _base_config()
        payload["engine"]["args"] = {"reuse_cache": 1, "act_type": "BF16"}
        payload["engine"]["env"] = {"TOKENIZERS_PARALLELISM": "false"}
        payload["workload"] = {
            "kind": "grid",
            "input_lengths": [128, 256],
            "output_lengths": [32],
            "batch": {"strategy": "fixed", "sizes": [1, 4]},
            "generate_config": {
                "top_k": 1,
                "stop_words_list": [[1, 2]],
            },
        }

        config = parse_config(payload)

        self.assertIsInstance(config.workload, DecodeGridWorkloadConfig)
        self.assertEqual((128, 256), config.workload.input_lengths)
        self.assertEqual((32,), config.workload.output_lengths)
        self.assertEqual({"reuse_cache": 1, "act_type": "BF16"}, config.engine.args)
        self.assertEqual({"TOKENIZERS_PARALLELISM": "false"}, config.engine.env)
        self.assertEqual(
            {"top_k": 1, "stop_words_list": [[1, 2]]},
            config.workload.generate_config,
        )

    def test_distribution_and_aone_models_are_discriminated(self):
        payload = _base_config()
        payload["workload"] = {
            "kind": "distribution",
            "distribution_file": "workloads/requests.json",
            "output_length": 16,
            "batch": {
                "strategy": "target_tpot",
                "max_size": 8,
                "candidate_step": 2,
                "target_tpot_ms": 20.0,
            },
        }
        payload["execution"] = {
            "backend": "aone",
            "resource": "gpu",
            "gpu_type": "H20",
            "max_parallel": 4,
            "shard_count": 2,
        }

        config = parse_config(payload)

        self.assertEqual("distribution", config.workload.kind)
        self.assertEqual(
            TargetTPOTConfig(
                strategy="target_tpot",
                max_size=8,
                candidate_step=2,
                target_tpot_ms=20.0,
            ),
            config.workload.batch,
        )
        self.assertEqual(DecodeProfilingConfig(), config.workload.profiling)
        self.assertIsInstance(config.execution, AOneExecutionConfig)
        self.assertEqual(4, config.execution.max_parallel)
        self.assertEqual(2, config.execution.shard_count)

    def test_unknown_structured_fields_report_complete_path(self):
        cases = [
            ("root_extra", "root_extra"),
            ("model_extra", "model.extra"),
            ("topology_extra", "model.topology.cp"),
            ("workload_extra", "workload.extra"),
            ("execution_extra", "execution.extra"),
        ]
        for name, expected_path in cases:
            with self.subTest(name=name):
                payload = _base_config()
                if name == "root_extra":
                    payload["root_extra"] = True
                elif name == "model_extra":
                    payload["model"]["extra"] = True
                elif name == "topology_extra":
                    payload["model"]["topology"]["cp"] = 2
                elif name == "workload_extra":
                    payload["workload"]["extra"] = True
                else:
                    payload["execution"]["extra"] = True
                with self.assertRaisesRegex(ConfigError, expected_path):
                    parse_config(payload)

    def test_range_and_type_errors_are_rejected(self):
        mutations = [
            ("schema_version", lambda value: value.update(schema_version="1")),
            (
                "model.topology.tp",
                lambda value: value["model"]["topology"].update(tp=0),
            ),
            (
                "engine.max_seq_len",
                lambda value: value["engine"].update(max_seq_len=True),
            ),
            (
                "workload.batch_sizes",
                lambda value: value["workload"].update(batch_sizes=[]),
            ),
            (
                "workload.cache_alignment",
                lambda value: value["workload"].update(cache_alignment=-1),
            ),
            (
                "execution.devices",
                lambda value: value["execution"].update(devices=[0, -1]),
            ),
        ]
        for path, mutate in mutations:
            with self.subTest(path=path):
                payload = copy.deepcopy(_base_config())
                mutate(payload)
                with self.assertRaisesRegex(ConfigError, path):
                    parse_config(payload)

    def test_decode_grid_uses_output_axis_and_fixed_batch_policy(self):
        payload = _base_config()
        payload["workload"] = {
            "kind": "grid",
            "input_lengths": [128, 256],
            "output_lengths": [16, 64],
            "batch": {"strategy": "fixed", "sizes": [1, 4]},
            "profiling": {"enabled": True, "runs": 2, "steps": 3},
            "generate_config": {"top_k": 1},
        }
        workload = parse_config(payload).workload
        self.assertEqual((16, 64), workload.output_lengths)
        self.assertEqual(
            FixedBatchConfig(strategy="fixed", sizes=(1, 4)), workload.batch
        )
        self.assertEqual(
            DecodeProfilingConfig(enabled=True, runs=2, steps=3), workload.profiling
        )

    def test_decode_batch_strategies_are_mutually_exclusive(self):
        for batch, path in (
            (
                {
                    "strategy": "target_tpot",
                    "sizes": [4],
                    "max_size": 16,
                    "candidate_step": 4,
                    "target_tpot_ms": 10.0,
                },
                "workload.batch.sizes",
            ),
            (
                {"strategy": "fixed", "sizes": [4], "target_tpot_ms": 10.0},
                "workload.batch.target_tpot_ms",
            ),
        ):
            with self.subTest(path=path):
                payload = _base_config()
                payload["workload"] = {
                    "kind": "grid",
                    "input_lengths": [128],
                    "output_lengths": [16],
                    "batch": batch,
                }
                with self.assertRaisesRegex(ConfigError, path):
                    parse_config(payload)

    def test_fixed_scalar_and_distribution_array_outputs_are_rejected(self):
        fixed = _base_config()
        fixed["workload"] = {
            "kind": "grid",
            "input_lengths": [128],
            "output_length": 16,
            "batch": {"strategy": "fixed", "sizes": [1]},
        }
        distribution = copy.deepcopy(fixed)
        distribution["workload"] = {
            "kind": "distribution",
            "distribution_file": "requests.json",
            "output_lengths": [16],
            "batch": {"strategy": "fixed", "sizes": [1]},
        }
        for payload, path in (
            (fixed, "workload.output_length"),
            (distribution, "workload.output_lengths"),
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ConfigError, path):
                    parse_config(payload)

    def test_distribution_output_length_must_be_one_positive_scalar(self):
        for value in ([8, 16], {"min": 8, "max": 16}, 0, True):
            with self.subTest(value=value):
                payload = _base_config()
                payload["workload"] = {
                    "kind": "distribution",
                    "distribution_file": "requests.json",
                    "output_length": value,
                    "batch": {"strategy": "fixed", "sizes": [1]},
                }
                with self.assertRaisesRegex(ConfigError, "workload.output_length"):
                    parse_config(payload)

    def test_decode_batch_and_profiling_fields_are_strictly_validated(self):
        cases = [
            ({"batch": {"strategy": "other"}}, "workload.batch.strategy"),
            ({"batch": {"strategy": "fixed", "sizes": []}}, "workload.batch.sizes"),
            ({"batch": {"strategy": "fixed", "sizes": [True]}}, "workload.batch.sizes"),
            (
                {"batch": {"strategy": "target_tpot", "max_size": 4}},
                "workload.batch.candidate_step",
            ),
            ({"output_lengths": []}, "workload.output_lengths"),
            ({"output_lengths": [0]}, "workload.output_lengths"),
            ({"profiling": {"extra": 1}}, "workload.profiling.extra"),
            ({"profiling": {"enabled": 1}}, "workload.profiling.enabled"),
            ({"profiling": {"runs": 0}}, "workload.profiling.runs"),
            ({"profiling": {"steps": True}}, "workload.profiling.steps"),
            (
                {"profiling": {"enable_all_ranks": 0}},
                "workload.profiling.enable_all_ranks",
            ),
            (
                {"profiling": {"flush_timeout_seconds": -1}},
                "workload.profiling.flush_timeout_seconds",
            ),
        ]
        for overrides, path in cases:
            with self.subTest(path=path, overrides=overrides):
                payload = _base_config()
                payload["workload"] = {
                    "kind": "grid",
                    "input_lengths": [128],
                    "output_lengths": [16],
                    "batch": {"strategy": "fixed", "sizes": [1]},
                    **overrides,
                }
                with self.assertRaisesRegex(ConfigError, path):
                    parse_config(payload)

    def test_target_tpot_parameters_require_positive_finite_values(self):
        for field, value in (
            ("max_size", 0),
            ("candidate_step", True),
            ("target_tpot_ms", 0),
            ("target_tpot_ms", float("inf")),
        ):
            with self.subTest(field=field, value=value):
                payload = _base_config()
                payload["workload"] = {
                    "kind": "grid",
                    "input_lengths": [128],
                    "output_lengths": [16],
                    "batch": {
                        "strategy": "target_tpot",
                        "max_size": 4,
                        "candidate_step": 1,
                        "target_tpot_ms": 10.0,
                        field: value,
                    },
                }
                with self.assertRaisesRegex(ConfigError, f"workload.batch.{field}"):
                    parse_config(payload)

    def test_interpolation_include_shell_and_secret_forms_are_rejected(self):
        mutations = [
            ("include", lambda value: value.update(include="other.json")),
            ("environment", lambda value: value["model"].update(checkpoint="${MODEL}")),
            ("shell", lambda value: value["model"].update(checkpoint="$(pwd)/model")),
            ("backtick", lambda value: value["engine"]["args"].update(path="`pwd`")),
            (
                "secret_reference",
                lambda value: value["publish"].update(oss="secret://oss"),
            ),
            (
                "credential_name",
                lambda value: value["engine"]["env"].update(API_TOKEN="value"),
            ),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                payload = copy.deepcopy(_base_config())
                mutate(payload)
                with self.assertRaises(ConfigError):
                    parse_config(payload)

    def test_legacy_profile_and_cache_file_shapes_are_rejected(self):
        legacy_values = [
            {
                "model_type": "qwen",
                "checkpoint_path": "/models/qwen",
                "engine": {"tp_size": 1},
                "cache_grid": {"measure_runs": 3},
            },
            {
                **_base_config(),
                "profile": "profiles/model.jsonc",
            },
            {
                **_base_config(),
                "cache_grid_json": "grids/cache.json",
            },
        ]

        for value in legacy_values:
            with self.subTest(fields=sorted(value)):
                with self.assertRaises(ConfigError):
                    parse_config(value)


if __name__ == "__main__":
    unittest.main()
