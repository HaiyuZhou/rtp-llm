import copy
import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import (
    ConfigResolutionError,
    canonical_json,
    load_resolved_config,
    resolve_config,
)


def _base_config():
    return {
        "schema_version": 1,
        "model": {
            "label": "test-model",
            "type": "registered_model",
            "checkpoint": "/models/test",
        },
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 8,
            "args": {"act_type": "BF16", "reuse_cache": 1},
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


class ConfigResolutionTest(unittest.TestCase):
    def test_equivalent_defaults_and_mapping_order_hash_identically(self):
        minimal = _base_config()
        explicit = copy.deepcopy(minimal)
        explicit["model"]["tokenizer"] = "/models/test"
        explicit["model"]["topology"] = {"ep": 1, "pp": 1, "dp": 1, "tp": 1}
        explicit["engine"]["env"] = {}
        explicit["engine"]["args"] = {"reuse_cache": 1, "act_type": "BF16"}
        explicit["workload"].update(
            measure_runs=3,
            min_compute_len=128,
            min_cache_len=0,
            cache_lengths=[],
            random_cache_count=8,
            seed=1,
        )
        explicit["execution"]["devices"] = []
        explicit["output"]["result_root"] = "perf_test_results"
        explicit["regression"].update(
            enabled=False, threshold_pct=10.0, generate_candidate=False
        )
        explicit["publish"] = {
            "odps": {"enabled": False, "required": False, "artifacts": []},
            "oss": {"enabled": False, "required": False, "artifacts": []},
        }

        left = resolve_config(minimal)
        right = resolve_config(explicit)

        self.assertEqual(left.resolved_config, right.resolved_config)
        self.assertEqual(left.config_fingerprint, right.config_fingerprint)
        self.assertEqual(left.workload_fingerprint, right.workload_fingerprint)
        self.assertEqual(
            canonical_json(left.resolved_config), canonical_json(right.resolved_config)
        )

    def test_engine_and_workload_changes_affect_separate_fingerprints(self):
        original = resolve_config(_base_config())
        engine_change = _base_config()
        engine_change["engine"]["args"]["reuse_cache"] = 0
        workload_change = _base_config()
        workload_change["workload"]["max_input_len"] = 4096

        changed_engine = resolve_config(engine_change)
        changed_workload = resolve_config(workload_change)

        self.assertNotEqual(
            original.config_fingerprint, changed_engine.config_fingerprint
        )
        self.assertEqual(
            original.workload_fingerprint, changed_engine.workload_fingerprint
        )
        self.assertEqual(
            original.config_fingerprint, changed_workload.config_fingerprint
        )
        self.assertNotEqual(
            original.workload_fingerprint, changed_workload.workload_fingerprint
        )

    def test_distribution_file_is_materialized_relative_to_source_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workloads = root / "workloads"
            workloads.mkdir()
            distribution = {"sequence_lengths": [128, 256, 512]}
            (workloads / "requests.json").write_text(
                json.dumps(distribution), encoding="utf-8"
            )
            payload = _base_config()
            payload["workload"] = {
                "kind": "distribution",
                "distribution_file": "workloads/requests.json",
                "output_length": 16,
                "batch_sizes": [8],
            }
            config_path = root / "perf.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            resolved = load_resolved_config(config_path)

            self.assertNotIn("distribution_file", resolved.resolved_config["workload"])
            self.assertEqual(
                distribution, resolved.resolved_config["workload"]["distribution"]
            )
            self.assertNotIn(str(root), canonical_json(resolved.resolved_config))

    def test_distribution_content_changes_only_workload_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "requests.json"
            payload = _base_config()
            payload["workload"] = {
                "kind": "distribution",
                "distribution_file": "requests.json",
                "output_length": 16,
                "batch_sizes": [8],
            }
            source.write_text("[128, 256]", encoding="utf-8")
            first = resolve_config(payload, source_dir=root)
            source.write_text("[128, 512]", encoding="utf-8")
            second = resolve_config(payload, source_dir=root)

            self.assertEqual(first.config_fingerprint, second.config_fingerprint)
            self.assertNotEqual(first.workload_fingerprint, second.workload_fingerprint)

    def test_external_workload_size_is_bounded_before_json_parsing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "requests.json").write_text("[1, 2, 3]", encoding="utf-8")
            payload = _base_config()
            payload["workload"] = {
                "kind": "distribution",
                "distribution_file": "requests.json",
                "output_length": 16,
                "batch_sizes": [8],
            }

            with self.assertRaisesRegex(
                ConfigResolutionError, "workload.distribution_file.*too large"
            ):
                resolve_config(payload, source_dir=root, max_external_bytes=4)


if __name__ == "__main__":
    unittest.main()
