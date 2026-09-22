"""Mode-specific frontend to cache_grid_perf_test, with immutable run snapshots."""

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import uuid
from pathlib import Path

from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    load_profile,
    profile_environment,
)
from rtp_llm.test.perf_test.cache_grid.runner.workspace_budget import (
    WORKSPACE_TOKENS,
    fixed_workspace_grid,
    grid_token_budget,
    validate_fixed_workspace,
)

TARGET = "//rtp_llm/test/perf_test:cache_grid_perf_test"
REPO = Path(__file__).resolve().parents[5]
MANIFEST = "cache_perf_launch.json"
ENV_NAMES = {
    "PATH",
    "LD_LIBRARY_PATH",
    "CC",
    "CXX",
    "CUDAHOSTCXX",
    "NVCC_PREPEND_FLAGS",
    "CUDA_VISIBLE_DEVICES",
    "TOKENIZERS_PARALLELISM",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "START_PORT",
}
ENV_PREFIXES = (
    "CACHE_",
    "DG_JIT_",
    "DSV4_",
    "PERF_",
    "PREFILL_",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TILELANG_",
    "TRITON_",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def managed_env(name):
    return name in ENV_NAMES or name.startswith(ENV_PREFIXES)


def environment(profile, overrides, inherited):
    explicit = {}
    for item in overrides:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError("--env requires NAME=VALUE")
        explicit[name] = value
    profile_environment({"engine_env": explicit})
    values = {k: v for k, v in inherited.items() if managed_env(k)}
    profile_environment({"engine_env": values})
    values.update(profile_environment(profile))
    values.update(explicit)
    return values


def replace_args(argv, updates, flags=()):
    owned = set(updates) | set(flags)
    result, i = [], 0
    while i < len(argv):
        item = argv[i]
        name = item.split("=", 1)[0].lstrip("-")
        i += 1
        if name not in owned:
            result.append(item)
            continue
        if name in flags or "=" in item:
            continue
        if name == "cache_profile_case_ids":
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
        elif i < len(argv) and not argv[i].startswith("--"):
            i += 1
    result.extend(f"--{k}={v}" for k, v in updates.items() if v is not None)
    return result


def load_saved(directory):
    path = directory / MANIFEST
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("schema_version") != 1:
            raise ValueError("unsupported launch manifest")
        for filename, digest in saved["snapshots"].items():
            if sha((directory / filename).read_bytes()) != digest:
                raise ValueError(f"saved configuration changed: {filename}")
        return saved
    info_path = directory / "test_info.json"
    if not info_path.exists():
        raise ValueError(
            "no launch manifest/test_info; supply --profile and --grid for retest/profile"
        )
    info = json.loads(info_path.read_text())
    argv = info.get("argv", [])[1:]
    env = info.get("engine_environment", {})
    if not argv or any("***" in arg for arg in argv) or "***" in env.values():
        raise ValueError(
            "legacy metadata missing or redacted; use explicit --profile and --grid"
        )
    grid = Path(info["cache_grid_json"])
    if not grid.is_absolute():
        raise ValueError("legacy grid path is relative; provide explicit configuration")
    result = directory / "cache_grid_results.json"
    if result.exists():
        checkpoint = json.loads(result.read_text())
        if checkpoint.get("grid_sha256") != sha(grid.read_bytes()):
            raise ValueError("legacy grid differs from checkpoint")
    return dict(
        profile=info.get("profile") or {"schema_version": 1},
        grid=str(grid),
        env=env,
        runner_args=argv,
        bazel={},
        legacy=True,
    )


def load_cases(path, profile):
    from rtp_llm.test.perf_test.batch_decode_test import (
        _dedupe_cache_grid_cases,
        _load_cache_grid_cases,
        _resolve_cache_block_size,
    )

    payload = json.loads(path.read_text())
    cases = _load_cache_grid_cases(str(path))
    block = _resolve_cache_block_size(
        payload, int(profile.get("cache_grid", {}).get("expected_block_size", 0))
    )
    return payload, _dedupe_cache_grid_cases(cases, block) if block else cases


def build_plan(args, inherited=None):
    inherited = dict(os.environ if inherited is None else inherited)
    if args.profile_backend == "nsys" and args.mode != "profile":
        raise ValueError("--profile-backend=nsys requires profile mode")
    if not 0 <= args.nsys_tail_seconds <= 60:
        raise ValueError("--nsys-tail-seconds must be between 0 and 60")
    source = args.result_dir.resolve()
    replay = args.mode in ("retest", "profile")
    if args.mode == "run" and source.exists() and any(source.iterdir()):
        raise ValueError("run requires an empty/new result directory; use resume")
    if args.mode == "resume" and not (source / "cache_grid_results.json").is_file():
        raise ValueError("resume requires an existing cache_grid_results.json")
    if args.mode == "resume" and (
        args.profile
        or args.grid
        or args.runs is not None
        or args.env
        or args.skip_reuse_validation
    ):
        raise ValueError(
            "resume uses frozen configuration; profile/grid/runs/env overrides are forbidden"
        )
    if replay != bool(args.cases):
        raise ValueError("--cases is required only for retest/profile")
    if (args.profile is None) != (args.grid is None):
        raise ValueError("supply both --profile and --grid, or neither")
    if args.mode == "run" and not args.profile:
        raise ValueError("run requires --profile and --grid")
    if args.runs is not None and args.runs <= 0 or args.trace_timeout <= 0:
        raise ValueError("runs/trace-timeout must be positive")
    saved = load_saved(source) if not args.profile else None
    profile = (
        load_profile(args.profile) if args.profile else copy.deepcopy(saved["profile"])
    )
    grid = args.grid.resolve() if args.grid else Path(saved["grid"])
    payload, cases = load_cases(grid, profile)
    ids = sorted(set(int(s) for s in args.cases.split(","))) if args.cases else []
    missing = set(ids) - {c["case_id"] for c in cases}
    if missing:
        raise ValueError(f"unknown/deduplicated case IDs: {sorted(missing)}")
    if ids:
        cases = [c for c in cases if c["case_id"] in ids]
    if args.mode == "profile" and any(
        c.get("batch_size", 1) != 1 or "request_groups" in c for c in cases
    ):
        raise ValueError("profile currently supports ungrouped batch=1 only")
    env = environment(profile, args.env, inherited) if not saved else dict(saved["env"])
    if saved and args.env:
        overrides = environment({}, args.env, {})
        env.update(overrides)
    if saved and saved.get("legacy"):
        for key in ENV_NAMES:
            if key in inherited and key not in env:
                env[key] = inherited[key]
    bazel = copy.deepcopy((saved or {}).get("bazel") or profile.get("bazel", {}))
    configs = args.config or bazel.get("configs", ["cuda13", "sm10x"])
    if not isinstance(configs, list) or not all(isinstance(v, str) for v in configs):
        raise ValueError("bazel.configs must be a list of strings")
    output_base = args.output_base or bazel.get("output_base")
    bazel = dict(
        configs=configs,
        output_base=str(Path(output_base).resolve()) if output_base else None,
        executable=args.bazel or bazel.get("executable", "bazelisk"),
        test_timeout=int(bazel.get("test_timeout", 345600)),
    )
    dest = (
        source / "cache_perf_replays" / (args.mode + "_" + uuid.uuid4().hex)
        if replay
        else source
    )
    artifacts = {}
    baseline = (
        list(saved["runner_args"])
        if saved
        else [
            "--partial=2",
            "--decode_test_length=1",
            "--batch_size=1",
            "--cache_request_transport=dashsc_input_ids",
        ]
    )
    if args.mode == "resume":
        if saved.get("legacy"):
            # Keep the exact legacy configuration; runner performs full fingerprint validation.
            for i, arg in enumerate(baseline):
                if (
                    arg.startswith("--profile=")
                    and not Path(arg.split("=", 1)[1]).is_file()
                ):
                    raise ValueError(
                        "legacy profile no longer exists; restore it before resume"
                    )
        runner = replace_args(
            baseline, {}, ("require_cache_resume", "allow_resume_mismatch")
        ) + ["--require_cache_resume"]
    else:
        if replay:
            payload = {
                "schema_version": 2,
                "cases": cases,
                "generator": payload.get("generator", {}),
                "summary": {
                    "case_count": len(cases),
                    "input_count": len({c["input_len"] for c in cases}),
                },
            }
        artifacts["grid.snapshot.json"] = (
            encoded(payload) if replay else grid.read_bytes()
        )
        artifacts["profile.snapshot.json"] = encoded(profile)
        updates = dict(
            profile=str(dest / "profile.snapshot.json"),
            cache_grid_json=str(dest / "grid.snapshot.json"),
            result_dir=str(dest),
            cache_profile_runs=0,
            cache_profile_case_ids=None,
            cache_profile_backend="kineto",
            cache_nsys_session=None,
            cache_nsys_path=None,
            cache_nsys_tail_seconds=None,
            profile_runs=0,
        )
        if args.runs is not None and args.mode != "profile":
            updates["cache_measure_runs"] = args.runs
        elif not saved:
            updates["cache_measure_runs"] = int(
                profile.get("cache_grid", {}).get("measure_runs", 3)
            )
        if not saved:
            updates["cache_commit_tail_tokens"] = profile.get("cache_grid", {}).get(
                "commit_tail_tokens", 4096
            )
        fixed_workspace = (
            fixed_workspace_grid(payload) or "--cache_fixed_workspace" in baseline
        )
        if fixed_workspace:
            # Freeze effective capacities in argv, not just in mutable grid metadata.
            updates.update(
                max_seq_len=WORKSPACE_TOKENS,
                max_context_batch_size=1,
                max_batch_tokens_size=grid_token_budget(payload),
                concurrency_limit=max(c["batch_size"] for c in cases),
            )
        runner = replace_args(
            baseline,
            updates,
            (
                "cache_profile_only",
                "cache_profile_flat_output",
                "require_cache_resume",
                "allow_resume_mismatch",
            ),
        )
        if args.skip_reuse_validation and "--cache_skip_reuse_validation" not in runner:
            runner.append("--cache_skip_reuse_validation")
        if fixed_workspace:
            if "--cache_fixed_workspace" not in runner:
                runner.append("--cache_fixed_workspace")
            tail = next(
                (
                    int(a.split("=", 1)[1])
                    for a in runner
                    if a.startswith("--cache_commit_tail_tokens=")
                ),
                4096,
            )
            validate_fixed_workspace(
                cases, commit_tail=tail, token_budget=grid_token_budget(payload)
            )
        if args.mode == "profile":
            if any(a.split("=")[0] == "--cache_shared_seed" for a in runner):
                raise ValueError(
                    "profile cannot replay shared-seed mode; supply independent --profile and --grid"
                )
            runner = replace_args(
                runner,
                {
                    "cache_profile_runs": args.runs or 1,
                    "cache_profile_trace_timeout": args.trace_timeout,
                    "cache_profile_backend": args.profile_backend,
                    "cache_nsys_path": (
                        args.nsys_path if args.profile_backend == "nsys" else None
                    ),
                    "cache_nsys_session": (
                        args.nsys_session or ("cacheperf_" + uuid.uuid4().hex)
                        if args.profile_backend == "nsys"
                        else None
                    ),
                    "cache_nsys_tail_seconds": (
                        args.nsys_tail_seconds
                        if args.profile_backend == "nsys"
                        else None
                    ),
                },
            )
            runner += [
                "--cache_profile_only",
                "--cache_profile_flat_output",
                "--cache_profile_case_ids",
            ] + list(map(str, ids))
            if args.profile_backend == "nsys":
                env["GEN_TIMELINE_SYNC"] = "0"
                runner = replace_args(runner, {"gen_timeline_sync": "False"})
        env["PERF_PROFILE_RUNS"] = "0"
        # Make all resolved names part of test_info and the existing resume guard.
        # Bazel --test_env supplies the values before imports; these defaults only
        # register names with the runner's reproduction/environment capture.
        runner = [a for a in runner if not a.startswith("--engine_env=")]
        runner += ["--engine_env=" + k + "=" + v for k, v in sorted(env.items())]
        launch = dict(
            schema_version=1,
            mode=args.mode,
            profile=profile,
            grid=str(dest / "grid.snapshot.json"),
            env=env,
            runner_args=runner,
            bazel=bazel,
            source_result_dir=str(source) if replay else None,
            selected_case_ids=ids,
            snapshots={k: sha(v) for k, v in artifacts.items()},
        )
        artifacts[MANIFEST] = encoded(launch)
    command = [bazel["executable"]]
    if bazel["output_base"]:
        command.append("--output_base=" + bazel["output_base"])
    command += ["test", TARGET] + ["--config=" + c for c in configs]
    command += [
        f"--test_timeout={bazel['test_timeout']}",
        "--test_output=streamed",
        "--nocache_test_results",
    ]
    command += ["--test_env=" + k + "=" + v for k, v in sorted(env.items())]
    command += ["--test_arg=" + arg for arg in runner]
    if args.mode == "profile" and args.profile_backend == "nsys":
        session = next(
            a.split("=", 1)[1] for a in runner if a.startswith("--cache_nsys_session=")
        )
        command.append(
            "--run_under="
            + shlex.join(
                [
                    args.nsys_path,
                    "launch",
                    "--show-output=true",
                    "--session-new=" + session,
                    "--trace=cuda,nvtx,osrt",
                    "--sample=process-tree",
                    "--cpuctxsw=process-tree",
                    "--wait=all",
                ]
            )
        )
    process_env = {k: v for k, v in inherited.items() if not managed_env(k)}
    process_env.update(env)
    return dict(
        command=command,
        process_env=process_env,
        artifacts=artifacts,
        destination=dest,
        summary=dict(
            mode=args.mode,
            planned_cases=len(cases),
            selected_case_ids=ids,
            profiler=args.mode == "profile",
            profile_backend=args.profile_backend if args.mode == "profile" else None,
            skip_reuse_validation="--cache_skip_reuse_validation" in runner,
            reads_checkpoint=args.mode == "resume",
            output=str(dest),
            environment=env,
            timing=(
                "profiler diagnostic, not formal latency"
                if args.mode == "profile"
                else "formal client wall and engine prefill recorded separately"
            ),
        ),
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["run", "resume", "retest", "profile"])
    p.add_argument("--profile", type=Path, help="JSON or commented JSONC profile")
    p.add_argument("--grid", type=Path)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--cases", help="comma-separated original case IDs")
    p.add_argument("--runs", type=int)
    p.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--config", action="append", help="repeatable Bazel configuration")
    p.add_argument("--output-base")
    p.add_argument("--bazel")
    p.add_argument("--trace-timeout", type=int, default=180)
    p.add_argument("--profile-backend", choices=("kineto", "nsys"), default="kineto")
    p.add_argument(
        "--nsys-path", default="nsys", help="nsys executable in the test environment"
    )
    p.add_argument("--nsys-session", help="new nsys session (default: unique name)")
    p.add_argument(
        "--nsys-tail-seconds",
        type=float,
        default=0.1,
        help="extra capture after request completion; not a GPU barrier (0..60)",
    )
    p.add_argument(
        "--skip-reuse-validation",
        action="store_true",
        help=(
            "Keep cases whose observed reuse differs from the grid expectation; "
            "the actual values remain recorded"
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true", help="print only; no writes/Bazel/GPU work"
    )
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        plan = build_plan(args)
        print(json.dumps(plan["summary"], ensure_ascii=False, indent=2), flush=True)
        print(shlex.join(plan["command"]), flush=True)
        if args.dry_run:
            return 0
        dest = plan["destination"]
        dest.mkdir(parents=True, exist_ok=args.mode in ("run", "resume"))
        for name, content in plan["artifacts"].items():
            with (dest / name).open("xb") as stream:
                stream.write(content)
        return subprocess.run(
            plan["command"], cwd=REPO, env=plan["process_env"], check=False
        ).returncode
    except (ValueError, OSError, KeyError) as error:
        p.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
