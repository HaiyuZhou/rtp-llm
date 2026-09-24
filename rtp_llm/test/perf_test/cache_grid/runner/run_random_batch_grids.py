#!/usr/bin/env python3
"""Run fixed-batch grids serially using a model profile and frozen run snapshots."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from rtp_llm.test.perf_test.cache_grid.runner import cache_perf
from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
    normalize_cache_case,
)
from rtp_llm.test.perf_test.cache_grid.runner.workspace_budget import (
    fixed_workspace_grid,
    grid_token_budget,
    validate_fixed_workspace,
)

REPO_ROOT = Path(__file__).resolve().parents[5]


def inspect_grid(path):
    grid = json.loads(path.read_text(encoding="utf-8"))
    raw = grid.get("cases")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: requires a nonempty explicit cases list")
    cases = [normalize_cache_case(case, i) for i, case in enumerate(raw)]
    batches = {case["batch_size"] for case in cases}
    if len(batches) != 1:
        raise ValueError(f"{path}: split mixed batch sizes into separate files")
    if fixed_workspace_grid(grid):
        metadata = grid.get("generator", {})
        validate_fixed_workspace(
            cases,
            block=int(metadata.get("cache_alignment", 4096)),
            commit_tail=int(
                metadata.get("parameters", {}).get("commit_tail_tokens", 4096)
            ),
            token_budget=grid_token_budget(grid),
        )
    return {"grid_json": str(path.resolve()), "batch_size": batches.pop()}


def build_launch(args, plan):
    argv = [
        "run",
        "--profile",
        str(args.profile.resolve()),
        "--grid",
        plan["grid_json"],
        "--result-dir",
        plan["result_dir"],
    ]
    for name in ("output_base", "bazel"):
        value = getattr(args, name)
        if value is not None:
            argv += ["--" + name.replace("_", "-"), str(value)]
    for name in ("config", "env"):
        for value in getattr(args, name):
            argv += ["--" + name, value]
    if args.skip_reuse_validation:
        argv.append("--skip-reuse-validation")
    return cache_perf.build_plan(cache_perf.parser().parse_args(argv))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    grids = parser.add_mutually_exclusive_group(required=True)
    grids.add_argument("--grid-dir", type=Path)
    grids.add_argument("--grid-json", type=Path, nargs="+")
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="Model, topology, cache geometry and environment JSON/JSONC profile",
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-base", type=Path)
    parser.add_argument("--bazel", "--bazelisk", dest="bazel")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-reuse-validation", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    files = sorted(args.grid_dir.glob("*.json")) if args.grid_dir else args.grid_json
    if not files:
        raise SystemExit("No JSON grids found")
    if len({p.stem for p in files}) != len(files):
        raise SystemExit("Grid filenames must have distinct stems")
    plans, launches = [], []
    try:
        for path in files:
            plan = inspect_grid(path)
            plan["result_dir"] = str((args.result_root / path.stem).resolve())
            if Path(plan["result_dir"]).exists() and not args.dry_run:
                raise ValueError(
                    f"Result already exists; use a new result-root: {plan['result_dir']}"
                )
            launch = build_launch(args, plan)
            plan.update(command=launch["command"], status="pending")
            plans.append(plan)
            launches.append(launch)
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    for plan in plans:
        print(f"B={plan['batch_size']} result={plan['result_dir']}", flush=True)
        print(shlex.join(plan["command"]), flush=True)
    if args.dry_run:
        return 0
    args.result_root.mkdir(parents=True, exist_ok=True)
    manifest = args.result_root / "batch_runs.json"
    if manifest.exists():
        raise SystemExit(f"Refusing to overwrite {manifest}")

    def save():
        manifest.write_text(
            json.dumps({"runs": plans}, indent=2) + "\n", encoding="utf-8"
        )

    save()
    for plan, launch in zip(plans, launches):
        plan["status"] = "running"
        save()
        try:
            code = cache_perf.execute_plan(launch, allow_existing=False)
        except KeyboardInterrupt:
            plan["status"] = "interrupted"
            save()
            return 130
        except OSError as exc:
            plan["error"] = str(exc)
            code = 127
        plan["returncode"] = code
        plan["status"] = "completed" if code == 0 else "failed"
        save()
        if code and not args.continue_on_error:
            return code if code > 0 else 1
    return int(any(p["status"] == "failed" for p in plans))


if __name__ == "__main__":
    raise SystemExit(main())
