"""Archived inputs need explicit migration; normal consumers stay current-only."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.cache_grid.formula.prefill_formula_fit import (
    load_observations,
)
from rtp_llm.test.perf_test.cache_grid.plot.generate_prefill_3d_chart import (
    load_rows as static_rows,
)
from rtp_llm.test.perf_test.cache_grid.plot.generate_prefill_interactive_chart import (
    load_rows as interactive_rows,
)
from rtp_llm.test.perf_test.cache_grid.runner import cache_perf
from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
    MaterializedCaseStore,
    validate_cache_grid_resume,
)
from rtp_llm.test.perf_test.cache_grid.runner.migrate_cache_perf import (
    migrate_case_store,
    migrate_result_file,
    migrate_results,
    migrate_run,
)
from rtp_llm.test.perf_test.cache_grid.runner.result_schema import current_metrics


def old_metric():
    return dict(
        case_key="bs1_seq16_cache8",
        case_id=7,
        seq_len=16,
        cache_len=8,
        status="success",
        measure_runs=1,
        runs=[
            dict(
                success=True,
                input_len=16,
                output_len=1,
                reuse_len=8,
                client_wall_time_ms=12.0,
                prefill_time_ms=3.0,
            )
        ],
    )


def validate_resume(path, **updates):
    return validate_cache_grid_resume(
        str(path),
        **dict(
            grid_sha256=None,
            profile_sha256=None,
            measure_runs=1,
            cache_commit_tail_tokens=4,
            expected_block_size=0,
            request_transport="http_prompt",
            **updates
        )
    )


class MigrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_old_containers_are_rejected_then_migrate_to_same_server_observation(self):
        metric = old_metric()
        for payload in (
            [metric],
            {"results": [metric]},
            {"schema_version": 1, "metrics": [metric]},
        ):
            with self.subTest(payload=type(payload).__name__):
                original = copy.deepcopy(payload)
                with self.assertRaisesRegex(ValueError, "Migrate"):
                    current_metrics(payload)
                result = migrate_results(payload)
                self.assertEqual(payload, original)
                path = self.root / "result.json"
                path.write_text(json.dumps(result))
                self.assertEqual(load_observations([path])[0][0].target_ms, 3)
                self.assertEqual(static_rows(path, 1)[0]["rt"], 3)
                self.assertEqual(interactive_rows(path, 1)[0]["prefill_rt"], 3)
                self.assertEqual(result["metrics"][0]["runs"][0]["prefill_time_ms"], 3)
                self.assertFalse(result["resume_compatible"])

    def test_server_only_migrates_and_client_only_cannot_become_server_time(self):
        metric = old_metric()
        del metric["runs"][0]["client_wall_time_ms"]
        result = migrate_results([metric])
        self.assertEqual(result["metrics"][0]["runs"][0]["prefill_time_ms"], 3)
        metric = old_metric()
        del metric["runs"][0]["prefill_time_ms"]
        with self.assertRaisesRegex(ValueError, "prefill_time_ms"):
            migrate_results([metric])

    def test_no_aggregate_or_missing_validity_is_invented(self):
        for edit, message in (
            (lambda m: m.pop("runs"), "recorded runs"),
            (lambda m: m.pop("status"), "status is missing"),
            (lambda m: m.pop("measure_runs"), "missing measure_runs"),
            (lambda m: m["runs"][0].pop("success"), "success must be explicit"),
        ):
            metric = old_metric()
            edit(metric)
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                migrate_results([metric])

    def test_conflicting_aliases_and_future_versions_are_rejected(self):
        metric = old_metric()
        metric["input_len"] = 32
        with self.assertRaisesRegex(ValueError, "conflicting input_len"):
            migrate_results([metric])
        with self.assertRaisesRegex(ValueError, "unsupported result"):
            migrate_results({"schema_version": 99, "metrics": []})

    def test_json_migration_does_not_overwrite_or_modify_source(self):
        source, output = self.root / "old.json", self.root / "current.json"
        source.write_text(json.dumps([old_metric()]))
        original = source.read_bytes()
        migrate_result_file(source, output)
        result = json.loads(output.read_text())
        self.assertEqual(
            result["migration"]["source_sha256"][str(source)], cache_perf.sha(original)
        )
        with self.assertRaises(FileExistsError):
            migrate_result_file(source, output)
        self.assertEqual(source.read_bytes(), original)

    def test_csv_migration_records_server_contract(self):
        source, output = self.root / "old.csv", self.root / "current.csv"
        source.write_text("seq_len,reuse_len,first_token_cost_time\n16,8,12\n")
        for reader in (lambda p: load_observations([p]), lambda p: static_rows(p, 1)):
            with self.assertRaises(ValueError):
                reader(source)
        migrate_result_file(source, output)
        with output.open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["target_ms"], "12")
        self.assertEqual(row["measurement_contract"], "server_first_token_cost_time_ms")

    def test_csv_rejects_client_time_and_requires_matching_output_format(self):
        source, output = self.root / "old.csv", self.root / "current.csv"
        source.write_text(
            "seq_len,reuse_len,client_wall_time_ms,output_len\n16,8,12,2\n"
        )
        with self.assertRaisesRegex(ValueError, "recorded server"):
            migrate_result_file(source, output)
        source.write_text("seq_len,reuse_len,first_token_cost_time\n16,8,12\n")
        with self.assertRaisesRegex(ValueError, "preserve the file format"):
            migrate_result_file(source, self.root / "current.json")
        migrate_result_file(source, output)
        with output.open() as stream:
            self.assertEqual(next(csv.DictReader(stream))["target_ms"], "12")

    def test_grouped_requests_keep_round_boundaries_and_individual_times(self):
        source = {
            "schema_version": 1,
            "metrics": [
                dict(
                    batch_size=2,
                    input_len=16,
                    request_groups=[dict(count=2, input_len=16, cache_len=8)],
                    status="passed",
                    measure_runs=1,
                    runs=[
                        dict(
                            valid=True,
                            batch_wall_time_ms=21,
                            requests=[
                                old_metric()["runs"][0],
                                dict(
                                    old_metric()["runs"][0],
                                    prefill_time_ms=5,
                                    client_wall_time_ms=20,
                                ),
                            ],
                        )
                    ],
                )
            ],
        }
        result = migrate_results(source)
        run = result["metrics"][0]["runs"][0]
        self.assertEqual([r["prefill_time_ms"] for r in run["requests"]], [3, 5])
        self.assertEqual(run["batch_wall_time_ms"], 21)
        self.assertNotIn("ttft_ms", run)

    def make_old_run(self):
        source = self.root / "old-run"
        source.mkdir()
        grid = source / "grid.json"
        grid.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "generator": {"cache_alignment": 8},
                    "cases": [dict(case_id=7, batch_size=1, input_len=16, cache_len=8)],
                }
            )
        )
        info = dict(
            schema_version=3,
            argv=[
                "main.py",
                "--partial=2",
                "--checkpoint_path=/model",
                "--cache_measure_runs=1",
            ],
            engine_environment={"DSV4_CHUNK_TOKENS": "4096"},
            cache_grid_json=str(grid),
            profile={"schema_version": 1},
        )
        (source / "test_info.json").write_text(json.dumps(info))
        (source / "cache_grid_results.json").write_text(
            json.dumps({"results": [old_metric()], "complete": True})
        )
        return source

    def test_run_migration_freezes_config_and_supports_retest_but_not_resume(self):
        source = self.make_old_run()
        original = {str(p): p.read_bytes() for p in source.iterdir()}
        output = self.root / "new-run"
        migrate_run(source, output)
        saved = cache_perf.load_saved(output)
        self.assertEqual(saved["schema_version"], 2)
        self.assertEqual(saved["env"], {"DSV4_CHUNK_TOKENS": "4096"})
        plan = cache_perf.build_plan(
            cache_perf.parser().parse_args(
                ["retest", "--result-dir", str(output), "--cases", "7"]
            ),
            {},
        )
        self.assertIn("--test_arg=--checkpoint_path=/model", plan["command"])
        self.assertEqual(plan["summary"]["environment"]["DSV4_CHUNK_TOKENS"], "4096")
        with self.assertRaisesRegex(ValueError, "resume guards"):
            validate_resume(output, allow_resume_mismatch=True)
        self.assertEqual(original, {str(p): p.read_bytes() for p in source.iterdir()})
        with self.assertRaisesRegex(ValueError, "already exists"):
            migrate_run(source, output)

    def test_run_migration_replays_journal_and_refuses_damaged_records(self):
        source = self.make_old_run()
        journal = source / "cache_grid_results.journal.jsonl"
        latest = old_metric()
        latest["runs"][0]["prefill_time_ms"] = 19
        journal.write_text(json.dumps(latest) + "\n")
        output = self.root / "new-run"
        migrate_run(source, output)
        result = json.loads((output / "cache_grid_results.json").read_text())
        self.assertEqual(result["metrics"][0]["runs"][0]["prefill_time_ms"], 19)
        journal.write_text(journal.read_text() + "{partial")
        with self.assertRaises(ValueError):
            migrate_run(source, self.root / "failed")
        self.assertFalse((self.root / "failed").exists())

    def test_v1_manifest_conversion_checks_snapshot_integrity(self):
        source = self.root / "v1"
        source.mkdir()
        grid = b'{"cases": [{"case_id": 7, "input_len":16, "cache_len":8}]}'
        (source / "grid.snapshot.json").write_bytes(grid)
        launch = dict(
            schema_version=1,
            profile={"schema_version": 1},
            grid="/obsolete/grid.json",
            snapshots={"grid.snapshot.json": cache_perf.sha(grid)},
            env={},
            runner_args=["--partial=2"],
        )
        (source / cache_perf.MANIFEST).write_text(json.dumps(launch))
        migrate_run(source, self.root / "v2")
        self.assertEqual(
            cache_perf.load_saved(self.root / "v2")["profile"], {"schema_version": 1}
        )
        (source / "grid.snapshot.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            migrate_run(source, self.root / "tampered")
        self.assertFalse((self.root / "tampered").exists())

    def test_case_store_expands_old_base_prefix_without_tokenizer(self):
        source, output = self.root / "old-store", self.root / "new-store"
        (source / "prefixes").mkdir(parents=True)
        info = dict(
            schema_version=1,
            word=" hello",
            marker="M",
            marker_ids_len=1,
            max_cache_len=4,
            run_count=1,
        )
        record = dict(
            schema_version=1,
            case_id=7,
            batch_size=1,
            input_len=6,
            cache_len=3,
            runs=[dict(tail="R", fillers=2)],
        )
        (source / "store_info.json").write_text(json.dumps(info))
        (source / "prefixes/base.txt").write_text("M" + " hello" * 3)
        (source / "manifest.jsonl").write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "migrate_cache_perf"):
            MaterializedCaseStore(str(source)).load_cases()
        migrate_case_store(source, output)
        store = MaterializedCaseStore(str(output))
        case = store.load_cases()[0]
        prompts = store.load_case(case)
        self.assertEqual(prompts.seed_text, "M hello hello")
        self.assertEqual(prompts.run_texts, ["M hello helloR hello hello"])
        self.assertFalse((output / "prefixes").exists())
        self.assertNotIn(
            "marker_ids_len", json.loads((output / "store_info.json").read_text())
        )

    def test_missing_resume_guards_cannot_be_bypassed(self):
        result = dict(schema_version=2, mode="prefix_cache_grid", metrics=[])
        (self.root / "cache_grid_results.json").write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "resume guards"):
            validate_resume(self.root, allow_resume_mismatch=True)

    def test_current_envelope_does_not_enable_legacy_field_fallbacks(self):
        result = migrate_results([old_metric()])
        metric = result["metrics"][0]
        for field in (
            "batch_size",
            "input_len",
            "status",
            "measure_runs",
            "cache_len_requested",
        ):
            broken = copy.deepcopy(result)
            del broken["metrics"][0][field]
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "Migrate"
            ):
                current_metrics(broken)
        del metric["runs"][0]["prefill_time_ms"]
        with self.assertRaisesRegex(ValueError, "requires per-request prefill_time_ms"):
            current_metrics(result)


if __name__ == "__main__":
    unittest.main()
