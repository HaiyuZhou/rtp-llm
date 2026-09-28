"""Explicit, offline migration of archived cache-perf data. Never edits inputs."""

import argparse
import copy
import csv
import io
import json
from pathlib import Path

from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    load_profile,
    profile_environment,
    validate_profile,
)
from rtp_llm.test.perf_test.cache_grid.runner.cache_perf import (
    MANIFEST,
    encoded,
    replace_args,
    sha,
)
from rtp_llm.test.perf_test.cache_grid.runner.result_schema import current_metrics


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def version(data, allowed, kind):
    if not isinstance(data, dict):
        raise ValueError(f"{kind} must be a JSON object")
    if data.get("schema_version") not in allowed:
        raise ValueError(
            f"unsupported {kind} schema_version: {data.get('schema_version')}"
        )


def canonical_field(data, name, aliases):
    values = [data[k] for k in (name, *aliases) if k in data]
    if not values or any(value != values[0] for value in values):
        raise ValueError(f"missing or conflicting {name}")
    for alias in aliases:
        data.pop(alias, None)
    data[name] = values[0]


def migrate_metric(source):
    if not isinstance(source, dict):
        raise ValueError("metric must be a JSON object")
    item = copy.deepcopy(source)
    canonical_field(item, "input_len", ("seq_len",))
    # Historical scalar grids only supported batch one.
    if "batch_size" not in item:
        if "request_groups" in item:
            raise ValueError("grouped metrics require explicit batch_size")
        item["batch_size"] = 1
    if "request_groups" not in item:
        canonical_field(item, "cache_len_requested", ("target_cache_len", "cache_len"))
    aliases = {"success": "ok", "passed": "ok"}
    item["status"] = aliases.get(item.get("status"), item.get("status"))
    if item["status"] == "error":
        return item
    runs = item.get("runs")
    if not isinstance(runs, list) or (not runs and item["status"] == "ok"):
        raise ValueError(
            "migration requires recorded runs; aggregate latency cannot reconstruct requests"
        )
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("run must be a JSON object")
        requests = run.get("requests", [run])
        if not isinstance(requests, list) or any(
            not isinstance(r, dict) for r in requests
        ):
            raise ValueError("requests must be a list of JSON objects")
        for request in requests:
            if type(request.get("success")) is not bool:
                raise ValueError(
                    "request success must be explicit; remeasure incomplete records"
                )
            if request["success"] and "ttft_ms" not in request:
                if "client_wall_time_ms" not in request:
                    raise ValueError(
                        "missing client TTFT: server prefill_time_ms cannot be converted to ttft_ms; remeasure"
                    )
                if request.get("output_len") != 1:
                    raise ValueError(
                        "client wall time is TTFT only with a recorded output_len=1"
                    )
                request["ttft_ms"] = request["client_wall_time_ms"]
    if item["status"] is None:
        raise ValueError("metric status is missing; cannot infer measurement validity")
    if "measure_runs" not in item:
        raise ValueError(
            "missing measure_runs; recorded rounds cannot prove the intended repeat count"
        )
    # Successful rounds can be counted; planned rounds cannot be inferred.
    item.setdefault(
        "success_runs",
        sum(
            (
                run.get("valid") is True
                if "requests" in run
                else run.get("success") is True
            )
            for run in runs
        ),
    )
    return item


def migrate_results(source):
    if isinstance(source, list):
        result, metrics = {}, source
    else:
        version(source, (None, 1, 2), "result")
        if source.get("mode") not in (None, "prefix_cache_grid"):
            raise ValueError("not a prefix_cache_grid result")
        result = copy.deepcopy(source)
        if "metrics" in result and "results" in result:
            raise ValueError("ambiguous result: both metrics and results")
        metrics = result.pop("results", result.get("metrics"))
    if not isinstance(metrics, list):
        raise ValueError("expected metrics/results list or a bare metric list")
    result.update(
        schema_version=2,
        mode="prefix_cache_grid",
        metrics=[migrate_metric(item) for item in metrics],
        resume_compatible=False,
    )
    current_metrics(result)
    return result


def provenance(files):
    return {
        str(path.resolve()): sha(path.read_bytes()) for path in dict.fromkeys(files)
    }


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def migrate_result_file(source, destination):
    if (source.suffix.lower() == ".csv") != (destination.suffix.lower() == ".csv"):
        raise ValueError(
            "preserve the file format: CSV output must use .csv; JSON output must not"
        )
    if source.suffix.lower() == ".csv":
        source_hash = sha(source.read_bytes())
        rows = list(csv.DictReader(io.StringIO(source.read_text(encoding="utf-8"))))
        if not rows:
            raise ValueError("empty CSV")
        for row in rows:
            if (
                not any(key in row for key in ("target_ms", "ttft_ms", "avg_ttft_ms"))
                and "client_wall_time_ms" in row
                and row.get("output_len") not in ("1", "1.0")
            ):
                raise ValueError("CSV client wall time requires recorded output_len=1")
            canonical_field(row, "input_len", ("seq_len",))
            canonical_field(
                row,
                "cache_len",
                ("cache_len_observed", "reuse_len", "cache_len_requested"),
            )
            canonical_field(
                row, "target_ms", ("ttft_ms", "avg_ttft_ms", "client_wall_time_ms")
            )
            row.setdefault("batch_size", "1")
            row["migration_source_path"] = str(source.resolve())
            row["migration_source_sha256"] = source_hash
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        write_new(destination, output.getvalue().encode())
    else:
        result = migrate_results(read_json(source))
        result["migration"] = {
            "source_sha256": provenance([source]),
            "resume_compatible": False,
        }
        write_new(destination, encoded(result))


def migrate_run(source, destination, profile_path=None, grid_path=None):
    """Freeze archived launch metadata for analysis/retest, never assert resume safety."""
    if destination.exists():
        raise ValueError("output directory already exists")
    files = []
    manifest_path = source / MANIFEST
    info_path = source / "test_info.json"
    if manifest_path.exists():
        saved = read_json(manifest_path)
        version(saved, (1, 2), "launch")
        files.append(manifest_path)
        if info_path.exists():
            info = read_json(info_path)
            files.append(info_path)
            if info.get("schema_version") == 4 and info["config_file_sha256"][
                MANIFEST
            ] != sha(manifest_path.read_bytes()):
                raise ValueError("saved configuration changed: " + MANIFEST)
        for name, digest in saved["snapshots"].items():
            path = source / name
            if not path.resolve().is_relative_to(source.resolve()):
                raise ValueError("snapshot must be inside the input directory")
            if sha(path.read_bytes()) != digest:
                raise ValueError(f"saved configuration changed: {name}")
            files.append(path)
        if saved["schema_version"] == 2:
            name = saved["profile_file"]
            if name not in saved["snapshots"]:
                raise ValueError("profile snapshot is not fingerprinted")
            profile = load_profile(source / name)
        else:
            profile = saved["profile"]
        argv, env = saved["runner_args"], saved["env"]
        grid = grid_path or source / "grid.snapshot.json"
        if grid_path is None and not grid.exists():
            grid = Path(saved["grid"])
    else:
        info = read_json(info_path)
        version(info, (None, 1, 2, 3), "test_info")
        files.append(info_path)
        argv = info.get("argv", [])[1:]
        env = info.get("engine_environment", {})
        profile = info.get("profile") or {"schema_version": 1}
        grid = grid_path or Path(info["cache_grid_json"])
        if not grid.is_absolute():
            raise ValueError("legacy grid path is relative; supply --grid")
        saved = {}
    if profile_path:
        profile = load_profile(profile_path)
        files.append(profile_path)
    validate_profile(profile)
    if not argv or any("***" in arg for arg in argv) or "***" in env.values():
        raise ValueError(
            "missing/redacted launch arguments; remeasure with explicit configuration"
        )
    env = profile_environment({"engine_env": env})
    files.append(grid)
    artifacts = {
        "grid.snapshot.json": grid.read_bytes(),
        "profile.snapshot.json": encoded(profile),
    }
    json.loads(artifacts["grid.snapshot.json"])
    argv = replace_args(
        argv,
        {
            "profile": destination / "profile.snapshot.json",
            "cache_grid_json": destination / "grid.snapshot.json",
            "result_dir": destination,
        },
        ("require_cache_resume", "allow_resume_mismatch"),
    )
    launch = dict(
        schema_version=2,
        mode="run",
        profile_file="profile.snapshot.json",
        grid=str(destination / "grid.snapshot.json"),
        env=env,
        runner_args=[arg for arg in argv if not arg.startswith("--engine_env=")],
        bazel=saved.get("bazel", {}),
        source_result_dir=str(source),
        selected_case_ids=saved.get("selected_case_ids", []),
        snapshots={name: sha(data) for name, data in artifacts.items()},
    )
    artifacts[MANIFEST] = encoded(launch)
    artifacts["test_info.json"] = encoded(
        dict(
            schema_version=4,
            status="migrated",
            config_files=dict(
                launch=MANIFEST,
                profile="profile.snapshot.json",
                grid="grid.snapshot.json",
            ),
            config_file_sha256={name: sha(data) for name, data in artifacts.items()},
        )
    )
    result_path = source / "cache_grid_results.json"
    journal_path = source / "cache_grid_results.journal.jsonl"
    if result_path.exists():
        result = migrate_results(read_json(result_path))
        if (
            grid_path is None
            and result.get("grid_sha256")
            and result["grid_sha256"] != sha(grid.read_bytes())
        ):
            raise ValueError(
                "legacy grid differs from checkpoint; supply --grid for a deliberate retest configuration"
            )
        files.append(result_path)
        if journal_path.exists():
            # A damaged journal needs manual recovery: do not silently lose completed cases.
            records = [
                migrate_metric(json.loads(line))
                for line in journal_path.read_text().splitlines()
                if line.strip()
            ]
            merged = {item["case_key"]: item for item in result["metrics"]}
            merged.update({item["case_key"]: item for item in records})
            result["metrics"] = list(merged.values())
            files.append(journal_path)
        current_metrics(result)
        result["completed_cases"] = len(result["metrics"])
        result["migration"] = {
            "resume_compatible": False,
            "reason": "Archived metadata is for analysis/retest; start a new measurement to resume.",
        }
        artifacts["cache_grid_results.json"] = encoded(result)
    elif journal_path.exists():
        raise ValueError(
            "journal without base checkpoint; restore the checkpoint before migration"
        )
    artifacts["migration_report.json"] = encoded(
        dict(
            source_sha256=provenance(files),
            resume_compatible=False,
            note="Use cache_perf retest for new measurements. Original run guards are preserved, not fabricated.",
        )
    )
    destination.mkdir(parents=True)
    for name, data in artifacts.items():
        write_new(destination / name, data)


def migrate_case_store(source, destination):
    if destination.exists():
        raise ValueError("output directory already exists")
    info_path, manifest_path = source / "store_info.json", source / "manifest.jsonl"
    info = read_json(info_path)
    version(info, (None, 1, 2), "case store")
    if (
        not isinstance(info.get("word"), str)
        or type(info.get("run_count")) is not int
        or info["run_count"] <= 0
    ):
        raise ValueError("case store requires word and positive run_count")
    files = [info_path, manifest_path]
    records = []
    base = None
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        version(record, (None, 1, 2), "case record")
        if (
            "members" not in record
            and record["cache_len"]
            and "seed_text" not in record
            and "seed_marker" not in record
        ):
            base_path = source / "prefixes" / "base.txt"
            if base is None:
                base = base_path.read_text(encoding="utf-8")
            marker_len = info["marker_ids_len"]
            if marker_len < 0:
                raise ValueError("legacy prefix requires marker_ids_len")
            chars = len(info["marker"]) + len(info["word"]) * (
                record["cache_len"] - marker_len
            )
            if not 0 <= chars <= len(base):
                raise ValueError(
                    "legacy base prefix is shorter than the requested prefix"
                )
            record["seed_text"] = base[:chars]
            files.append(base_path)
        record["schema_version"] = 2
        records.append(record)
    if len({r["case_id"] for r in records}) != len(records):
        raise ValueError("duplicate case_id in case store")
    info.update(schema_version=2, case_count=len(records))
    for key in ("max_cache_len", "marker_ids_len"):
        info.pop(key, None)
    artifacts = {
        "store_info.json": encoded(info),
        "manifest.jsonl": b"".join(
            (json.dumps(record, ensure_ascii=False) + "\n").encode()
            for record in records
        ),
        "migration_report.json": encoded({"source_sha256": provenance(files)}),
    }
    destination.mkdir(parents=True)
    for name, data in artifacts.items():
        write_new(destination / name, data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="kind", required=True)
    result = commands.add_parser(
        "results", help="convert historical JSON or observation CSV"
    )
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    for name in ("run", "case-store"):
        command = commands.add_parser(name, help="convert into a new directory")
        command.add_argument("--input-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "run":
            command.add_argument("--profile", type=Path)
            command.add_argument("--grid", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.kind == "results":
            migrate_result_file(args.input.resolve(), args.output.resolve())
        elif args.kind == "run":
            migrate_run(
                args.input_dir.resolve(),
                args.output_dir.resolve(),
                args.profile,
                args.grid,
            )
        else:
            migrate_case_store(args.input_dir.resolve(), args.output_dir.resolve())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"migration failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
