import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    ProfileError,
    engine_args_section,
    engine_section,
    extract_embedded_profile,
    fingerprint,
    load_profile,
    merge_engine_args,
    pressure_settings,
    resolve_int,
    resolve_label,
    resolve_str,
    resolve_title,
    set_engine_arg,
    validate_profile,
)


def _minimal_profile(**overrides):
    base = {"schema_version": 1}
    base.update(overrides)
    return base


class ValidateProfileTest(unittest.TestCase):
    def test_diagnostic_only_requires_an_actual_boolean(self):
        for value in (True, False):
            validate_profile(_minimal_profile(cache_grid=dict(diagnostic_only=value)))
        for value in (0, 1, "true", "false", None):
            with self.subTest(value=value), self.assertRaisesRegex(
                ProfileError, "diagnostic_only"
            ):
                validate_profile(
                    _minimal_profile(cache_grid=dict(diagnostic_only=value))
                )

    def test_explicit_smoke_requires_an_actual_boolean(self):
        for value in (True, False):
            validate_profile(_minimal_profile(cache_grid=dict(explicit_smoke=value)))
        for value in (0, 1, "true", "false", None):
            with self.subTest(value=value), self.assertRaisesRegex(
                ProfileError, "explicit_smoke"
            ):
                validate_profile(
                    _minimal_profile(cache_grid=dict(explicit_smoke=value))
                )

    def test_seed_timeout_is_required_positive_finite_and_overridable(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            resolve_pressure_config,
        )

        values = dict(
            seed_timeout_s=30,
            fill_timeout_s=20,
            fill_max_requests=10,
            fill_no_progress_s=5,
            fill_input_tokens=512,
            gpu_tolerance_blocks=2,
            activity_timeout_s=10,
            drain_timeout_s=10,
            capture_timeout_s=60,
        )
        self.assertEqual(resolve_pressure_config(values).seed_timeout_s, 30)
        for invalid in (0, -1, True, float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_pressure_config(dict(values, seed_timeout_s=invalid))
        missing = dict(values)
        del missing["seed_timeout_s"]
        with self.assertRaisesRegex(ValueError, "seed_timeout_s"):
            resolve_pressure_config(missing)
        profile = _minimal_profile(cache_grid=dict(pressure=values))
        self.assertEqual(
            pressure_settings(
                profile, SimpleNamespace(pressure_seed_timeout=45)
            ).seed_timeout_s,
            45,
        )
        self.assertEqual(
            pressure_settings(
                profile, SimpleNamespace(cache_pressure_seed_timeout=60), direct=True
            ).seed_timeout_s,
            60,
        )

    def test_control_timeout_defaults_validates_and_allows_cli_overrides(self):
        values = dict(
            seed_timeout_s=30,
            fill_timeout_s=20,
            fill_max_requests=10,
            fill_no_progress_s=5,
            fill_input_tokens=512,
            gpu_tolerance_blocks=2,
            activity_timeout_s=10,
            drain_timeout_s=10,
            capture_timeout_s=60,
        )
        profile = _minimal_profile(cache_grid=dict(pressure=values))
        self.assertEqual(
            pressure_settings(profile, SimpleNamespace()).control_timeout_s, 1.0
        )
        self.assertEqual(
            pressure_settings(
                profile, SimpleNamespace(pressure_control_timeout=7.5)
            ).control_timeout_s,
            7.5,
        )
        self.assertEqual(
            pressure_settings(
                profile,
                SimpleNamespace(cache_pressure_control_timeout=8.5),
                direct=True,
            ).control_timeout_s,
            8.5,
        )
        for invalid in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                pressure_settings(
                    profile,
                    SimpleNamespace(pressure_control_timeout=invalid),
                )

    def test_pressure_fill_defaults_without_mutating_profile_and_allows_overrides(self):
        values = dict(
            seed_timeout_s=30,
            fill_timeout_s=20,
            fill_max_requests=10,
            fill_no_progress_s=5,
            gpu_tolerance_blocks=2,
            activity_timeout_s=10,
            drain_timeout_s=10,
            capture_timeout_s=60,
        )
        profile = _minimal_profile(cache_grid=dict(pressure=values))
        self.assertEqual(
            pressure_settings(profile, SimpleNamespace()).fill_input_tokens, 102400
        )
        self.assertNotIn("fill_input_tokens", values)
        values["fill_input_tokens"] = 8192
        self.assertEqual(
            pressure_settings(profile, SimpleNamespace()).fill_input_tokens, 8192
        )
        self.assertEqual(
            pressure_settings(
                profile,
                SimpleNamespace(cache_pressure_fill_input_tokens=16384),
                direct=True,
            ).fill_input_tokens,
            16384,
        )

    def test_pressure_settings_use_explicit_zero_and_reject_unknown_or_nan(self):
        values = dict(
            seed_timeout_s=30,
            fill_timeout_s=20,
            fill_max_requests=10,
            fill_no_progress_s=5,
            fill_input_tokens=512,
            gpu_tolerance_blocks=2,
            activity_timeout_s=10,
            drain_timeout_s=10,
            capture_timeout_s=60,
        )
        profile = _minimal_profile(cache_grid=dict(pressure=values))
        config = pressure_settings(
            profile, SimpleNamespace(pressure_gpu_tolerance_blocks=0)
        )
        self.assertEqual(config.gpu_tolerance_blocks, 0)
        self.assertEqual(config.measure_s, 60)
        self.assertEqual(values["gpu_tolerance_blocks"], 2)
        with self.assertRaises(ValueError):
            pressure_settings(
                profile, SimpleNamespace(pressure_fill_timeout=float("nan"))
            )
        values["invented_capacity"] = 1
        with self.assertRaisesRegex(ValueError, "pressure settings"):
            pressure_settings(profile, SimpleNamespace())

    def test_adaptive_pressure_settings_resolve_from_profile_and_cli(self):
        values = dict(
            seed_timeout_s=30,
            fill_timeout_s=20,
            fill_max_requests=10,
            fill_no_progress_s=5,
            fill_input_tokens=512,
            gpu_tolerance_blocks=2,
            activity_timeout_s=10,
            drain_timeout_s=10,
            capture_timeout_s=60,
            family_schedule="tiered_rolling",
            memory_reuse_ratio=0.5,
            fill_host_max_ratio=0.8,
            rate_control="adaptive_backlog",
            request_rate_min=50,
            request_rate_max=400,
            backlog_low=64,
            backlog_high=192,
            max_outstanding=256,
            cohort_size=32,
        )
        profile = _minimal_profile(cache_grid=dict(pressure=values))

        resolved = pressure_settings(profile, SimpleNamespace())
        self.assertEqual(resolved.request_rate_max, 400)
        self.assertEqual(resolved.max_outstanding, 256)
        overridden = pressure_settings(
            profile,
            SimpleNamespace(
                pressure_request_rate_min=25,
                pressure_request_rate_max=300,
            ),
        )
        self.assertEqual(overridden.request_rate_min, 25)
        self.assertEqual(overridden.request_rate_max, 300)

    def test_minimal_valid_profile(self):
        validate_profile({"schema_version": 1})

    def test_full_profile(self):
        validate_profile(
            {
                "schema_version": 1,
                "label": "test",
                "engine": {"tp_size": 8, "dp_size": 1},
                "engine_args": {"--use_deepep_moe": "1", "--fp8_kv_cache": 1},
                "cache_grid": {"cache_alignment": 512},
                "chart": {"title": "Test", "model_label": "T"},
            }
        )

    def test_rejects_non_dict(self):
        with self.assertRaisesRegex(ProfileError, "must be a JSON object"):
            validate_profile([1, 2, 3])

    def test_rejects_missing_schema_version(self):
        with self.assertRaisesRegex(ProfileError, "missing schema_version"):
            validate_profile({})

    def test_rejects_wrong_schema_version(self):
        with self.assertRaisesRegex(ProfileError, "unsupported schema_version 99"):
            validate_profile({"schema_version": 99})

    def test_rejects_bool_in_engine_args(self):
        with self.assertRaisesRegex(ProfileError, "bool is not a valid CLI value"):
            validate_profile({"schema_version": 1, "engine_args": {"--flag": False}})

    def test_rejects_invalid_engine_section(self):
        with self.assertRaisesRegex(ProfileError, "engine must be a JSON object"):
            validate_profile({"schema_version": 1, "engine": "bad"})

    def test_rejects_invalid_engine_args_section(self):
        with self.assertRaisesRegex(ProfileError, "engine_args must be a JSON object"):
            validate_profile({"schema_version": 1, "engine_args": [1, 2]})

    def test_rejects_explicit_null_object_sections(self):
        for section in ("engine", "engine_args", "cache_grid", "chart"):
            with self.subTest(section=section), self.assertRaisesRegex(
                ProfileError, rf"{section} must be a JSON object"
            ):
                validate_profile({"schema_version": 1, section: None})


class LoadProfileTest(unittest.TestCase):
    def test_mock_memory_profile_freezes_all_engine_options_and_diagnostic_marker(self):
        from rtp_llm.test.perf_test.cache_grid.runner import cache_perf

        config = Path(__file__).resolve().parents[1] / "config"
        path = config / "continuous_full_pressure_dsv4.mock_memory.example.jsonc"
        self.assertTrue(path.is_file(), "dedicated mock memory profile is missing")
        with tempfile.TemporaryDirectory() as directory:
            args = cache_perf.parser().parse_args(
                [
                    "profile",
                    "--profile",
                    str(path),
                    "--grid",
                    str(
                        config
                        / "continuous_full_pressure_grid.mock_memory.example.json"
                    ),
                    "--result-dir",
                    str(Path(directory) / "result"),
                    "--pressure-mode=continuous",
                    "--pressure-preload",
                ]
            )
            plan = cache_perf.build_plan(args, {})
        frozen = json.loads(plan["artifacts"]["profile.snapshot.json"])
        for key, value in {
            "--mock_memory_cache": 1,
            "--mock_memory_cache_hit_ratio": 0.5,
            "--mock_memory_cache_seed": 1,
            "--mock_memory_cache_source_blocks": 64,
        }.items():
            self.assertEqual(frozen["engine_args"][key], value)
        argv = merge_engine_args(frozen, [])
        self.assertEqual(dict(zip(argv[::2], argv[1::2]))["--mock_memory_cache"], "1")
        self.assertTrue(frozen["cache_grid"]["diagnostic_only"])
        self.assertNotIn("explicit_smoke", frozen["cache_grid"])
        self.assertIn("DIAGNOSTIC ONLY", frozen["label"])
        self.assertIn("OUTPUT CORRECTNESS INVALID", frozen["label"])
        launch = json.loads(plan["artifacts"][cache_perf.MANIFEST])
        self.assertFalse(launch["formal_candidate"])
        self.assertEqual(launch["formal_candidate_reasons"], ["diagnostic_only"])

    def test_load_from_disk(self):
        profile = _minimal_profile(label="test")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(profile, tmp)
            tmp.flush()
            loaded = load_profile(tmp.name)
        self.assertEqual(loaded, profile)

    def test_load_rejects_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write("{bad json")
            tmp.flush()
            with self.assertRaisesRegex(ProfileError, "invalid JSON"):
                load_profile(tmp.name)


class FingerprintTest(unittest.TestCase):
    def test_same_profile_same_fingerprint(self):
        p = _minimal_profile(engine={"tp_size": 8})
        self.assertEqual(fingerprint(p), fingerprint(p))

    def test_key_order_does_not_matter(self):
        p1 = {"schema_version": 1, "a": 1, "b": 2}
        p2 = {"schema_version": 1, "b": 2, "a": 1}
        self.assertEqual(fingerprint(p1), fingerprint(p2))

    def test_different_profiles_different_fingerprints(self):
        p1 = _minimal_profile(engine={"tp_size": 8})
        p2 = _minimal_profile(engine={"tp_size": 4})
        self.assertNotEqual(fingerprint(p1), fingerprint(p2))

    def test_fingerprint_is_hex_string(self):
        fp = fingerprint(_minimal_profile())
        self.assertEqual(len(fp), 64)
        int(fp, 16)


class ResolveTest(unittest.TestCase):
    def test_cli_wins_over_profile(self):
        self.assertEqual(
            resolve_int(
                {"schema_version": 1, "engine": {"tp_size": 8}},
                "engine",
                "tp_size",
                4,
                1,
            ),
            4,
        )

    def test_removed_runtime_alignment_profile_field_is_rejected(self):
        with self.assertRaisesRegex(ProfileError, "generator.cache_alignment"):
            validate_profile(
                {"schema_version": 1, "cache_grid": {"expected_block_size": 4096}}
            )

    def test_profile_wins_over_default(self):
        self.assertEqual(
            resolve_int(
                {"schema_version": 1, "engine": {"tp_size": 8}},
                "engine",
                "tp_size",
                None,
                1,
            ),
            8,
        )

    def test_default_when_all_absent(self):
        self.assertEqual(
            resolve_int({"schema_version": 1}, "engine", "tp_size", None, 1),
            1,
        )

    def test_explicit_zero_wins(self):
        self.assertEqual(
            resolve_int(
                {"schema_version": 1, "cache_grid": {"cache_alignment": 512}},
                "cache_grid",
                "cache_alignment",
                0,
                256,
            ),
            0,
        )

    def test_invalid_profile_value_falls_back_to_default(self):
        self.assertEqual(
            resolve_int(
                {"schema_version": 1, "engine": {"tp_size": "bad"}},
                "engine",
                "tp_size",
                None,
                1,
            ),
            1,
        )

    def test_resolve_str_cli_wins(self):
        self.assertEqual(
            resolve_str(
                {"schema_version": 1, "engine": {"model_type": "profile"}},
                "engine",
                "model_type",
                "cli",
                "default",
            ),
            "cli",
        )

    def test_resolve_str_empty_cli_falls_through(self):
        self.assertEqual(
            resolve_str(
                {"schema_version": 1, "engine": {"model_type": "profile"}},
                "engine",
                "model_type",
                "",
                "default",
            ),
            "profile",
        )

    def test_resolve_label_cli_wins(self):
        profile = {"schema_version": 1, "chart": {"model_label": "P"}}
        self.assertEqual(resolve_label(profile, "CLI", "D"), "CLI")

    def test_resolve_label_profile_wins(self):
        profile = {"schema_version": 1, "chart": {"model_label": "P"}}
        self.assertEqual(resolve_label(profile, None, "D"), "P")

    def test_resolve_label_uses_model_metadata(self):
        profile = {"schema_version": 1, "engine": {"model_type": "qwen_2"}}
        self.assertEqual(resolve_label(profile, None, "Model"), "qwen_2")
        profile["model_label"] = "My Model"
        self.assertEqual(resolve_label(profile, None, "Model"), "My Model")
        profile["chart"] = {"model_label": "Chart Model"}
        self.assertEqual(resolve_label(profile, None, "Model"), "Chart Model")
        self.assertEqual(resolve_label(profile, "CLI", "Model"), "CLI")

    def test_resolve_label_default_when_none(self):
        self.assertEqual(resolve_label(None, None, "D"), "D")
        self.assertEqual(resolve_label({"engine": None}, None, "D"), "D")

    def test_resolve_title(self):
        profile = {"schema_version": 1, "chart": {"title": "PT"}}
        self.assertEqual(resolve_title(profile, None, "DT"), "PT")
        self.assertEqual(resolve_title(profile, "CT", "DT"), "CT")
        self.assertEqual(resolve_title(None, None, "DT"), "DT")


class SetEngineArgTest(unittest.TestCase):
    def test_append_when_absent(self):
        result = set_engine_arg(["--other", "1"], "tp_size", "8")
        self.assertEqual(result, ["--other", "1", "--tp_size", "8"])

    def test_update_space_separated(self):
        result = set_engine_arg(["--tp_size", "4", "--other", "1"], "tp_size", "8")
        self.assertEqual(result, ["--tp_size", "8", "--other", "1"])

    def test_update_equals_form(self):
        result = set_engine_arg(["--tp_size=4", "--other", "1"], "tp_size", "8")
        self.assertEqual(result, ["--tp_size=8", "--other", "1"])

    def test_bare_flag_gets_value_inserted(self):
        result = set_engine_arg(["--reuse_cache", "--other"], "reuse_cache", "1")
        self.assertEqual(result, ["--reuse_cache", "1", "--other"])


class MergeEngineArgsTest(unittest.TestCase):
    def test_injects_missing_args(self):
        profile = {
            "schema_version": 1,
            "engine": {"tp_size": 8, "dp_size": 1},
            "engine_args": {"--use_deepep_moe": "1"},
        }
        result = merge_engine_args(profile, ["--other", "1"])
        self.assertIn("--tp_size", result)
        self.assertIn("8", result)
        self.assertIn("--use_deepep_moe", result)
        self.assertNotIn("--dp_size", result)

    def test_cli_wins_over_profile(self):
        profile = {
            "schema_version": 1,
            "engine": {"tp_size": 8},
        }
        result = merge_engine_args(profile, ["--tp_size", "4"])
        idx = result.index("--tp_size")
        self.assertEqual(result[idx + 1], "4")

    def test_cli_equals_form_wins(self):
        profile = {
            "schema_version": 1,
            "engine": {"tp_size": 8},
        }
        result = merge_engine_args(profile, ["--tp_size=4"])
        self.assertIn("--tp_size=4", result)
        self.assertNotIn("8", result)

    def test_skip_namespace_keys(self):
        profile = {
            "schema_version": 1,
            "engine": {"dp_size": 4, "max_seq_len": 65536, "concurrency_limit": 32},
        }
        result = merge_engine_args(profile, [])
        self.assertEqual(result, [])

    def test_include_namespace_keys_when_requested(self):
        profile = {
            "schema_version": 1,
            "engine": {"dp_size": 4},
        }
        result = merge_engine_args(profile, [], skip_namespace_keys=False)
        self.assertIn("--dp_size", result)


class ExtractEmbeddedProfileTest(unittest.TestCase):
    def test_extracts_profile(self):
        data = {
            "schema_version": 2,
            "mode": "prefix_cache_grid",
            "metrics": [],
            "profile": {"schema_version": 1, "label": "test"},
        }
        result = extract_embedded_profile(data)
        self.assertEqual(result, {"schema_version": 1, "label": "test"})

    def test_returns_none_when_absent(self):
        self.assertIsNone(
            extract_embedded_profile(
                {"schema_version": 2, "mode": "prefix_cache_grid", "metrics": []}
            )
        )

    def test_returns_none_when_not_dict(self):
        self.assertIsNone(extract_embedded_profile({"profile": "not a dict"}))

    def test_returns_none_for_non_dict_input(self):
        self.assertIsNone(extract_embedded_profile([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
