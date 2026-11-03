import copy
import unittest

from rtp_llm.test.perf_test.results import (
    DecodeMetrics,
    PrefillMetrics,
    ResultSchemaError,
    parse_result_envelope,
)


def _base_result(mode):
    return {
        "schema_version": 2,
        "run_id": "run-001",
        "mode": mode,
        "backend": "local",
        "status": "completed",
        "config_fingerprint": "a" * 64,
        "workload_fingerprint": "b" * 64,
        "model": {
            "label": "test-model",
            "type": "registered_model",
            "checkpoint": "/models/test",
            "topology": {"tp": 1, "dp": 1, "pp": 1, "ep": 1},
        },
        "summary": {"planned_cases": 1, "successful_cases": 1, "failed_cases": 0},
        "cases": [],
    }


def _prefill_result():
    value = _base_result("prefill")
    value["cases"] = [
        {
            "case_id": "prefill-case-a",
            "identity": {"input_len": 1024, "cache_len": 512, "batch_size": 1},
            "status": "ok",
            "runs": 3,
            "errors": [],
            "metrics": {
                "input_length": 1024,
                "expected_reuse_length": 512,
                "observed_reuse_length": 512,
                "compute_length": 512,
                "latency_ms": 10.5,
                "throughput_tokens_per_second": 48761.9,
                "validation": {"shape": True, "reuse": True, "timing": True},
                "seed": 7,
                "prefix_policy": {"kind": "shared_seed", "commit_tail_tokens": 64},
                "measurements": [{"run": 0, "latency_ms": 10.5}],
            },
        }
    ]
    return value


def _decode_result():
    value = _base_result("decode")
    value["summary"]["profiling_failed_cases"] = 0
    value["cases"] = [
        {
            "case_id": "decode-case-a",
            "identity": {
                "workload_kind": "grid",
                "input_lengths": [128] * 4,
                "output_length": 16,
                "batch_policy": {"strategy": "fixed", "size": 4},
                "measure_runs": 1,
                "warmup_runs": 0,
                "generate_config": {},
                "profiling": {
                    "enabled": False,
                    "runs": 1,
                    "steps": 3,
                    "enable_all_ranks": True,
                    "flush_timeout_seconds": 60,
                },
                "distribution": None,
            },
            "status": "ok",
            "runs": 1,
            "errors": [],
            "metrics": {
                "workload_kind": "grid",
                "input_lengths": [128, 128, 128, 128],
                "output_length": 16,
                "global_batch_size": 4,
                "topology": {"tp": 1, "dp": 2, "pp": 1, "ep": 1},
                "successful_requests": 4,
                "failed_requests": 0,
                "latency_ms": {
                    "wait": 1.0,
                    "prefill": 2.0,
                    "decode": 8.0,
                    "tpot": 0.5,
                    "total": 11.0,
                },
                "throughput_tokens_per_second": 8000.0,
                "measurements": [
                    {
                        "run": 0,
                        "throughput_tokens_per_second": 8000.0,
                        "worker_allocations": [],
                        "requests": [
                            {
                                "request_index": index,
                                "success": True,
                                "input_length": 128,
                                "output_length": 16,
                                "latency_ms": {
                                    "wait": 1.0,
                                    "prefill": 2.0,
                                    "decode": 8.0,
                                    "tpot": 0.5,
                                    "total": 11.0,
                                },
                            }
                            for index in range(4)
                        ],
                    }
                ],
                "search": None,
                "distribution": None,
                "profiling": {
                    "status": "disabled",
                    "manifest_path": None,
                    "error": None,
                },
            },
        }
    ]
    return value


class ResultEnvelopeTest(unittest.TestCase):
    def test_concurrent_prefill_evidence_accepts_complete_rounds(self):
        value = self._concurrent_prefill_result()
        parsed = parse_result_envelope(value)
        self.assertEqual(8, len(parsed.cases[0].metrics.measurements))
        self.assertEqual(2, parse_result_envelope(_decode_result()).schema_version)

    @staticmethod
    def _concurrent_prefill_result():
        value = _prefill_result()
        case = value["cases"][0]
        case["identity"].update(
            batch_size=4, measure_runs=2, measurement_contract="concurrent_batch_v1"
        )
        case["runs"] = 2
        case["metrics"].update(latency_ms=4.5, throughput_tokens_per_second=153600.0)
        case["metrics"]["measurements"] = [
            {
                "run": run,
                "request_index": member,
                "latency_ms": run * 4 + member + 1,
                "observed_reuse_length": 512,
                "round_elapsed_ms": elapsed,
                "measurement_contract": "concurrent_batch_v1",
            }
            for run, elapsed in enumerate((10.0, 20.0))
            for member in range(4)
        ]
        return value

    def test_concurrent_prefill_rejects_missing_entire_round(self):
        value = self._concurrent_prefill_result()
        case = value["cases"][0]
        case["runs"] = 1
        case["metrics"]["measurements"] = case["metrics"]["measurements"][:4]
        case["metrics"].update(latency_ms=2.5, throughput_tokens_per_second=204800.0)
        with self.assertRaisesRegex(ResultSchemaError, "measure_runs"):
            parse_result_envelope(value)

    def test_concurrent_prefill_evidence_rejects_tampering(self):
        mutations = {
            "duplicate": lambda c: c["metrics"]["measurements"][1].update(
                request_index=0
            ),
            "missing": lambda c: c["metrics"]["measurements"].pop(),
            "run": lambda c: c["metrics"]["measurements"][0].update(run=2),
            "member": lambda c: c["metrics"]["measurements"][0].update(request_index=4),
            "marker": lambda c: c["metrics"]["measurements"][0].update(
                measurement_contract="old"
            ),
            "elapsed_inconsistent": lambda c: c["metrics"]["measurements"][0].update(
                round_elapsed_ms=11
            ),
            "elapsed_zero": lambda c: c["metrics"]["measurements"][0].update(
                round_elapsed_ms=0
            ),
            "elapsed_nan": lambda c: c["metrics"]["measurements"][0].update(
                round_elapsed_ms=float("nan")
            ),
            "reuse": lambda c: c["metrics"]["measurements"][0].update(
                observed_reuse_length=511
            ),
            "latency_zero": lambda c: c["metrics"]["measurements"][0].update(
                latency_ms=0
            ),
            "latency_nan": lambda c: c["metrics"]["measurements"][0].update(
                latency_ms=float("nan")
            ),
            "latency_summary": lambda c: c["metrics"].update(latency_ms=5),
            "throughput_summary": lambda c: c["metrics"].update(
                throughput_tokens_per_second=204800
            ),
            "reuse_summary": lambda c: c["metrics"].update(observed_reuse_length=511),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                value = self._concurrent_prefill_result()
                mutate(value["cases"][0])
                with self.assertRaises(ResultSchemaError):
                    parse_result_envelope(value)

    def test_search_outcomes_require_strings_on_formal_and_failed_paths(self):
        for location in ("metrics", "details"):
            for outcome_field in ("search", "observation"):
                for malformed in ([], {}, 1, None):
                    with self.subTest(
                        location=location, field=outcome_field, malformed=malformed
                    ):
                        value = _decode_result()
                        case = value["cases"][0]
                        case["identity"]["batch_policy"] = {
                            "strategy": "target_tpot",
                            "max_size": 4,
                            "candidate_step": 2,
                            "target_tpot_ms": 10.0,
                        }
                        search = {
                            "target_tpot_ms": 10.0,
                            "max_size": 4,
                            "candidate_step": 2,
                            "outcome": "found",
                            "best_batch_size": 4,
                            "observations": [{"batch_size": 4, "outcome": "satisfied"}],
                        }
                        if location == "metrics":
                            case["metrics"]["search"] = search
                        else:
                            search.update(
                                outcome="no_satisfying_candidate",
                                best_batch_size=None,
                                observations=[
                                    {"batch_size": 4, "outcome": "target_exceeded"}
                                ],
                            )
                            case.update(
                                status="target_tpot_not_met",
                                metrics=None,
                                errors=[
                                    {
                                        "code": "target_tpot_not_met",
                                        "message": "no candidate",
                                        "details": {"search": search},
                                    }
                                ],
                            )
                            value["status"] = "partial"
                            value["summary"].update(successful_cases=0, failed_cases=1)
                        record = (
                            search
                            if outcome_field == "search"
                            else search["observations"][0]
                        )
                        record["outcome"] = malformed
                        with self.assertRaisesRegex(ResultSchemaError, "outcome"):
                            parse_result_envelope(value)

    def test_decode_v2_selected_batch_search_and_profiling(self):
        value = _decode_result()
        case = copy.deepcopy(value["cases"][0])
        case["case_id"] = "decode-tpot"
        case["identity"]["input_lengths"] = [128] * 16
        case["identity"]["batch_policy"] = {
            "strategy": "target_tpot",
            "max_size": 16,
            "candidate_step": 4,
            "target_tpot_ms": 10.0,
        }
        case["identity"]["profiling"]["enabled"] = True
        case["metrics"].update(
            input_lengths=[128] * 8,
            global_batch_size=8,
            successful_requests=8,
            search={
                "target_tpot_ms": 10.0,
                "max_size": 16,
                "candidate_step": 4,
                "outcome": "found",
                "best_batch_size": 8,
                "observations": [{"batch_size": 8, "outcome": "satisfied"}],
            },
            profiling={
                "status": "failed",
                "manifest_path": "profiles/decode-tpot/profile_manifest.json",
                "error": "trace flush timeout",
            },
        )
        requests = case["metrics"]["measurements"][0]["requests"]
        requests.extend(copy.deepcopy(requests))
        for index, request in enumerate(requests):
            request["request_index"] = index
        value["cases"].append(case)
        value["summary"].update(
            planned_cases=2, successful_cases=2, profiling_failed_cases=1
        )
        parsed = parse_result_envelope(value)
        self.assertEqual(value, parsed.to_dict())
        self.assertEqual(8, parsed.cases[1].metrics.global_batch_size)

    def test_decode_identity_requires_every_shape_field(self):
        for name in _decode_result()["cases"][0]["identity"]:
            with self.subTest(name=name):
                value = _decode_result()
                del value["cases"][0]["identity"][name]
                with self.assertRaisesRegex(ResultSchemaError, "identity." + name):
                    parse_result_envelope(value)

    def test_decode_counts_all_rounds_and_requires_case_runs(self):
        value = _decode_result()
        case = value["cases"][0]
        case["runs"] = case["identity"]["measure_runs"] = 2
        second = copy.deepcopy(case["metrics"]["measurements"][0])
        second["run"] = 1
        case["metrics"]["measurements"].append(second)
        case["metrics"]["successful_requests"] = 8
        self.assertEqual(value, parse_result_envelope(value).to_dict())
        case["runs"] = 1
        with self.assertRaisesRegex(ResultSchemaError, "runs"):
            parse_result_envelope(value)
        case["runs"] = 2
        case["metrics"]["successful_requests"] = 4
        with self.assertRaisesRegex(ResultSchemaError, "request outcomes"):
            parse_result_envelope(value)

    def test_failed_tpot_keeps_only_structured_search_diagnostics(self):
        value = _decode_result()
        case = value["cases"][0]
        case["identity"]["batch_policy"] = {
            "strategy": "target_tpot",
            "max_size": 4,
            "candidate_step": 4,
            "target_tpot_ms": 10.0,
        }
        search = {
            "target_tpot_ms": 10.0,
            "max_size": 4,
            "candidate_step": 4,
            "outcome": "no_satisfying_candidate",
            "best_batch_size": None,
            "observations": [{"batch_size": 1, "outcome": "target_exceeded"}],
        }
        case.update(
            status="target_tpot_not_met",
            metrics=None,
            errors=[
                {
                    "code": "target_tpot_not_met",
                    "message": "no candidate",
                    "details": {"search": search},
                }
            ],
        )
        value["status"] = "partial"
        value["summary"].update(successful_cases=0, failed_cases=1)
        self.assertEqual(value, parse_result_envelope(value).to_dict())
        for key, bad in (
            ("candidate_step", 0),
            ("max_size", -1),
            ("target_tpot_ms", float("nan")),
            ("outcome", "unknown"),
            ("best_batch_size", 4),
            ("observations", [{"batch_size": 5, "outcome": "satisfied"}]),
        ):
            with self.subTest(key=key):
                invalid = copy.deepcopy(value)
                invalid["cases"][0]["errors"][0]["details"]["search"][key] = bad
                with self.assertRaises(ResultSchemaError):
                    parse_result_envelope(invalid)

    def test_decode_profiling_status_and_manifest_are_validated(self):
        for profile in (
            {"status": "unknown", "manifest_path": None, "error": None},
            {"status": "completed", "manifest_path": None, "error": None},
            {"status": "failed", "manifest_path": "", "error": "timeout"},
        ):
            with self.subTest(profile=profile):
                value = _decode_result()
                value["cases"][0]["metrics"]["profiling"] = profile
                with self.assertRaisesRegex(ResultSchemaError, "profiling"):
                    parse_result_envelope(value)

    def test_decode_v2_rejects_unknown_nested_fields(self):
        for path in (
            ("identity",),
            ("identity", "batch_policy"),
            ("identity", "profiling"),
            ("metrics",),
            ("metrics", "profiling"),
        ):
            with self.subTest(path=path):
                value = _decode_result()
                record = value["cases"][0]
                for name in path:
                    record = record[name]
                record["extra"] = True
                with self.assertRaisesRegex(ResultSchemaError, "extra"):
                    parse_result_envelope(value)

    def test_decode_summary_counts_match_performance_and_profile_outcomes(self):
        for changes in (
            {"successful_cases": 0, "failed_cases": 1},
            {"profiling_failed_cases": 1},
        ):
            with self.subTest(changes=changes):
                value = _decode_result()
                value["summary"].update(changes)
                with self.assertRaisesRegex(ResultSchemaError, "summary"):
                    parse_result_envelope(value)

    def test_decode_search_and_selected_shape_must_agree_with_policy(self):
        value = _decode_result()
        case = value["cases"][0]
        case["identity"]["batch_policy"] = {
            "strategy": "target_tpot",
            "max_size": 4,
            "candidate_step": 2,
            "target_tpot_ms": 10.0,
        }
        case["metrics"]["search"] = {
            "target_tpot_ms": 10.0,
            "max_size": 4,
            "candidate_step": 2,
            "outcome": "found",
            "best_batch_size": 4,
            "observations": [{"batch_size": 4, "outcome": "satisfied"}],
        }
        self.assertEqual(value, parse_result_envelope(value).to_dict())
        for key, bad in (
            ("best_batch_size", 2),
            ("candidate_step", 3),
            ("max_size", 8),
            ("target_tpot_ms", 9.0),
            (
                "observations",
                [{"batch_size": 4, "outcome": "satisfied", "extra": True}],
            ),
            (
                "observations",
                [{"batch_size": 4, "outcome": "satisfied", "tpot_ms": None}],
            ),
            ("observations", [{"batch_size": 4, "outcome": "capacity_failed"}]),
            ("observations", [{"batch_size": 4, "outcome": "satisfied"}] * 2),
        ):
            with self.subTest(key=key, bad=bad):
                invalid = copy.deepcopy(value)
                invalid["cases"][0]["metrics"]["search"][key] = bad
                with self.assertRaises(ResultSchemaError):
                    parse_result_envelope(invalid)
        for key, bad in (
            ("output_length", 20),
            ("input_lengths", [64] * 4),
            ("distribution", {"actual_sequence_lengths": [128] * 4}),
        ):
            with self.subTest(key=key):
                invalid = copy.deepcopy(value)
                invalid["cases"][0]["metrics"][key] = bad
                with self.assertRaises(ResultSchemaError):
                    parse_result_envelope(invalid)

    def test_prefill_summary_retains_three_counts_and_v1_is_rejected(self):
        value = _prefill_result()
        value["summary"]["profiling_failed_cases"] = 0
        with self.assertRaisesRegex(
            ResultSchemaError, "summary.profiling_failed_cases"
        ):
            parse_result_envelope(value)
        value = _prefill_result()
        value["schema_version"] = 1
        with self.assertRaisesRegex(ResultSchemaError, "schema_version"):
            parse_result_envelope(value)

    def test_formal_tpot_metrics_satisfy_target_and_match_selected_observation(self):
        value = _decode_result()
        case = value["cases"][0]
        case["identity"]["batch_policy"] = {
            "strategy": "target_tpot",
            "max_size": 4,
            "candidate_step": 2,
            "target_tpot_ms": 10.0,
        }
        case["metrics"]["search"] = {
            "target_tpot_ms": 10.0,
            "max_size": 4,
            "candidate_step": 2,
            "outcome": "found",
            "best_batch_size": 4,
            "observations": [
                {
                    "batch_size": 4,
                    "outcome": "satisfied",
                    "tpot_ms": 0.5,
                    "success_rate": 1.0,
                    "throughput_tokens_per_second": 8000.0,
                }
            ],
        }
        self.assertEqual(value, parse_result_envelope(value).to_dict())
        for field, bad in (
            ("tpot", 11.0),
            ("requests", 1),
            ("observation_tpot", 0.6),
            ("observation_throughput", 7999.0),
        ):
            with self.subTest(field=field):
                invalid = copy.deepcopy(value)
                metrics = invalid["cases"][0]["metrics"]
                if field == "tpot":
                    metrics["latency_ms"]["tpot"] = bad
                elif field == "requests":
                    metrics.update(successful_requests=3, failed_requests=bad)
                else:
                    name = (
                        "tpot_ms"
                        if field == "observation_tpot"
                        else "throughput_tokens_per_second"
                    )
                    metrics["search"]["observations"][0][name] = bad
                with self.assertRaises(ResultSchemaError):
                    parse_result_envelope(invalid)

    def test_search_found_without_selected_observation_is_isolated(self):
        # A found search whose observations never include the selected batch
        # must be rejected as a schema error (isolated per case), never crash
        # the envelope parser: _parse_search requires best_batch_size to equal
        # the largest satisfied observation, so the selected-observation lookup
        # in parse_case_result is always satisfiable.
        value = _decode_result()
        case = value["cases"][0]
        case["identity"]["batch_policy"] = {
            "strategy": "target_tpot",
            "max_size": 4,
            "candidate_step": 2,
            "target_tpot_ms": 10.0,
        }
        case["metrics"]["search"] = {
            "target_tpot_ms": 10.0,
            "max_size": 4,
            "candidate_step": 2,
            "outcome": "found",
            "best_batch_size": 4,
            "observations": [{"batch_size": 2, "outcome": "satisfied"}],
        }
        with self.assertRaisesRegex(
            ResultSchemaError, "must select largest satisfied candidate"
        ):
            parse_result_envelope(value)

    def test_prefill_envelope_round_trips_as_golden_schema(self):
        source = _prefill_result()

        envelope = parse_result_envelope(source)

        self.assertIsInstance(envelope.cases[0].metrics, PrefillMetrics)
        self.assertEqual(source, envelope.to_dict())

    def test_decode_envelope_round_trips_as_golden_schema(self):
        source = _decode_result()

        envelope = parse_result_envelope(source)

        self.assertIsInstance(envelope.cases[0].metrics, DecodeMetrics)
        self.assertEqual(16, envelope.cases[0].metrics.output_length)
        self.assertEqual(source, envelope.to_dict())

    def test_every_common_envelope_field_is_required(self):
        for field in _base_result("prefill"):
            with self.subTest(field=field):
                value = _prefill_result()
                value.pop(field)
                with self.assertRaisesRegex(ResultSchemaError, field):
                    parse_result_envelope(value)

    def test_mode_specific_metrics_cannot_be_interpreted_as_shared_fields(self):
        value = _prefill_result()
        case = value["cases"][0]
        case["latency_ms"] = case["metrics"].pop("latency_ms")

        with self.assertRaisesRegex(ResultSchemaError, "cases\[0\].latency_ms"):
            parse_result_envelope(value)

    def test_metrics_are_discriminated_by_envelope_mode(self):
        prefill_with_decode = _prefill_result()
        prefill_with_decode["cases"][0]["metrics"] = _decode_result()["cases"][0][
            "metrics"
        ]
        decode_with_prefill = _decode_result()
        decode_with_prefill["cases"][0]["metrics"] = _prefill_result()["cases"][0][
            "metrics"
        ]

        with self.assertRaisesRegex(ResultSchemaError, "required field is missing"):
            parse_result_envelope(prefill_with_decode)
        with self.assertRaisesRegex(ResultSchemaError, "required field is missing"):
            parse_result_envelope(decode_with_prefill)

    def test_failed_case_may_have_diagnostics_without_metrics(self):
        value = _prefill_result()
        value["status"] = "partial"
        value["summary"] = {
            "planned_cases": 1,
            "successful_cases": 0,
            "failed_cases": 1,
        }
        value["cases"][0].update(
            status="invalid_reuse",
            errors=[{"code": "reuse_mismatch", "message": "expected 512, got 0"}],
            metrics=None,
        )

        envelope = parse_result_envelope(value)

        self.assertIsNone(envelope.cases[0].metrics)
        self.assertEqual(value, envelope.to_dict())

    def test_unknown_fields_and_invalid_counts_are_rejected(self):
        cases = []
        unknown = _decode_result()
        unknown["extra"] = True
        cases.append((unknown, "extra"))
        negative = _decode_result()
        negative["summary"]["failed_cases"] = -1
        cases.append((negative, "summary.failed_cases"))
        mismatched = copy.deepcopy(_decode_result())
        mismatched["cases"][0]["metrics"]["global_batch_size"] = 3
        cases.append((mismatched, "global_batch_size"))
        for value, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(ResultSchemaError, diagnostic):
                    parse_result_envelope(value)

    def test_decode_gang_aone_backend_is_accepted_like_prefill_aone(self):
        gang = _decode_result()
        gang["backend"] = "aone"

        parsed = parse_result_envelope(gang)

        self.assertEqual("aone", parsed.backend)
        self.assertEqual(gang, parsed.to_dict())

        prefill = _prefill_result()
        prefill["backend"] = "aone"
        self.assertEqual("aone", parse_result_envelope(prefill).backend)

        for value in (_decode_result(), _prefill_result()):
            with self.subTest(mode=value["mode"]):
                value["backend"] = "remote"
                with self.assertRaisesRegex(
                    ResultSchemaError, "backend: expected local or aone"
                ):
                    parse_result_envelope(value)

    def test_legacy_cache_grid_result_is_not_accepted_as_envelope(self):
        legacy = {
            "schema_version": 2,
            "mode": "prefix_cache_grid",
            "complete": True,
            "run_config": {"cache_grid_json": "grid.json"},
            "metrics": [],
        }

        with self.assertRaises(ResultSchemaError):
            parse_result_envelope(legacy)


if __name__ == "__main__":
    unittest.main()
