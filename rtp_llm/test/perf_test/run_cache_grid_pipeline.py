#!/usr/bin/env python3
"""Run a cache-grid test and generate all standard post-processing artifacts.

Arguments after ``--`` are forwarded unchanged to ``batch_decode_test``. The
pipeline owns the grid, mode, result directory and profile arguments so every
stage is guaranteed to use the same inputs.

Example::

    bazelisk run //rtp_llm/test/perf_test:run_cache_grid_pipeline -- \
      --cache-grid-json=/path/to/grid.json \
      --result-dir=/path/to/results \
      --profile=rtp_llm/test/perf_test/profiles/dsv4_pro_prefill.json \
      -- --cache_measure_runs=3 --expected_cache_block_size=512
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

MODULES = {
    "runner": "rtp_llm.test.perf_test.batch_decode_test",
    "fit": "rtp_llm.test.perf_test.deepseek_v4_prefill_formula_fit",
    "svg": "rtp_llm.test.perf_test.generate_prefill_3d_chart",
    "html": "rtp_llm.test.perf_test.generate_prefill_interactive_chart",
}
MANAGED_RUNNER_FLAGS = {
    "--cache_grid_json",
    "--partial",
    "--profile",
    "--result_dir",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-grid-json", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--estimator",
        choices=("median", "min", "trimmed"),
        default="median",
        help="Per-case statistic used by the formula fitter.",
    )
    parser.add_argument("--formula-output-dir", type=Path)
    parser.add_argument("--svg-output", type=Path)
    parser.add_argument("--cold-svg-output", type=Path)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Regenerate artifacts from an already completed result directory.",
    )
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are forwarded to batch_decode_test.",
    )
    return parser


def _module_command(module: str, arguments: Sequence[str]) -> list[str]:
    return [sys.executable, "-m", module, *arguments]


def _run_command(command: Sequence[str], *, check: bool) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    python_path = [entry for entry in sys.path if entry]
    inherited = env.get("PYTHONPATH")
    if inherited:
        python_path.append(inherited)
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    return subprocess.run(command, check=check, env=env)


def _validate_runner_args(arguments: Sequence[str]) -> None:
    for argument in arguments:
        name = argument.split("=", 1)[0]
        if name in MANAGED_RUNNER_FLAGS:
            raise ValueError(
                f"{name} is managed by the pipeline; configure it before "
                "the -- separator"
            )


def _profile_args(profile: Path | None) -> list[str]:
    return [f"--profile={profile}"] if profile is not None else []


def build_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    runner_args = list(args.runner_args)
    if runner_args and runner_args[0] == "--":
        runner_args.pop(0)
    _validate_runner_args(runner_args)

    result_dir = args.result_dir.resolve()
    result_json = result_dir / "cache_grid_results.json"
    formula_dir = (args.formula_output_dir or result_dir / "formula").resolve()
    svg_output = (args.svg_output or result_dir / "prefill_3d.svg").resolve()
    cold_svg_output = (
        args.cold_svg_output or result_dir / "prefill_cold_miss.svg"
    ).resolve()
    html_output = (
        args.html_output or result_dir / "prefill_3d.interactive.html"
    ).resolve()
    profile_args = _profile_args(args.profile.resolve() if args.profile else None)

    return {
        "runner": _module_command(
            MODULES["runner"],
            [
                f"--cache_grid_json={args.cache_grid_json.resolve()}",
                "--partial=2",
                f"--result_dir={result_dir}",
                *profile_args,
                *runner_args,
            ],
        ),
        "fit": _module_command(
            MODULES["fit"],
            [
                "fit",
                "--inputs",
                str(result_json),
                "--output-dir",
                str(formula_dir),
                "--batch-size",
                str(args.batch_size),
                "--estimator",
                args.estimator,
                *profile_args,
            ],
        ),
        "svg": _module_command(
            MODULES["svg"],
            [
                "--input",
                str(result_json),
                "--output",
                str(svg_output),
                "--cold-output",
                str(cold_svg_output),
                "--batch-size",
                str(args.batch_size),
                *profile_args,
            ],
        ),
        "html": _module_command(
            MODULES["html"],
            [
                "--input",
                str(result_json),
                "--output",
                str(html_output),
                "--batch-size",
                str(args.batch_size),
                *profile_args,
            ],
        ),
    }


def _load_completed_result(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"cache-grid result was not produced: {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"cache-grid result is not valid JSON: {path}") from error
    if not isinstance(payload, dict) or not payload.get("complete"):
        status = payload.get("status") if isinstance(payload, dict) else None
        raise RuntimeError(
            "refusing to post-process an incomplete cache-grid result: "
            f"status={status!r}"
        )
    return payload


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run_pipeline(
    args: argparse.Namespace,
    run_command: Callable[..., subprocess.CompletedProcess] = _run_command,
) -> int:
    commands = build_commands(args)
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    result_json = result_dir / "cache_grid_results.json"
    manifest_path = result_dir / "pipeline_summary.json"
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "result": str(result_json),
        "artifacts": {
            "formula_dir": commands["fit"][commands["fit"].index("--output-dir") + 1],
            "svg": commands["svg"][commands["svg"].index("--output") + 1],
            "cold_svg": commands["svg"][commands["svg"].index("--cold-output") + 1],
            "html": commands["html"][commands["html"].index("--output") + 1],
        },
        "stages": {},
    }
    _write_manifest(manifest_path, manifest)

    try:
        if not args.skip_test:
            completed = run_command(commands["runner"], check=False)
            manifest["stages"]["test"] = {"returncode": completed.returncode}
            _write_manifest(manifest_path, manifest)
            if completed.returncode:
                raise RuntimeError(
                    f"cache-grid test failed with exit code {completed.returncode}"
                )
        else:
            manifest["stages"]["test"] = {"skipped": True}

        result = _load_completed_result(result_json)
        manifest["completed_cases"] = result.get("completed_cases")
        manifest["total_cases"] = result.get("total_cases")

        # A failed production-acceptance gate returns 3 after still writing a
        # valid formula/report. Always generate both charts before propagating
        # that status to the caller.
        fit = run_command(commands["fit"], check=False)
        manifest["stages"]["fit"] = {"returncode": fit.returncode}
        for stage in ("svg", "html"):
            completed = run_command(commands[stage], check=False)
            manifest["stages"][stage] = {"returncode": completed.returncode}
            _write_manifest(manifest_path, manifest)
            if completed.returncode:
                raise RuntimeError(
                    f"{stage} generation failed with exit code {completed.returncode}"
                )

        manifest["status"] = "completed" if fit.returncode == 0 else "fit_rejected"
        manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_manifest(manifest_path, manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return fit.returncode
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_manifest(manifest_path, manifest)
        raise


def main() -> None:
    args = build_parser().parse_args()
    try:
        return_code = run_pipeline(args)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
