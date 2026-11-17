"""Maintenance of one saved SSH deployment; never rebuild or resume a model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rtp_llm.test.perf_test.ssh.remote import RemoteHost


def read_deployment(run_dir):
    root = Path(run_dir).resolve()
    home = root / "execution" / "ssh"
    deployment = json.loads((home / "members.json").read_text())
    plan = json.loads((home / "controller_plan.json").read_text())
    payload = json.dumps(
        plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    if (
        deployment.get("schema_version") != 1
        or hashlib.sha256(payload).hexdigest() != deployment["controller_plan_hash"]
    ):
        raise ValueError("SSH deployment plan identity mismatch")
    if deployment["execution"] != plan["resolved_config"]["execution"]:
        raise ValueError("SSH deployment execution differs from the frozen plan")
    if len(deployment["members"]) > plan["resource_plan"]["machine_count"]:
        raise ValueError("SSH deployment has extra members")
    if plan["backend"] != "ssh":
        raise ValueError("maintenance requires an SSH deployment")
    seen = set()
    for member in deployment["members"]:
        index = member["member_index"]
        if (
            type(index) is not int
            or not 0 <= index < plan["resource_plan"]["machine_count"]
            or index in seen
            or member.get("member_id") != str(index)
        ):
            raise ValueError("invalid or duplicate SSH member index")
        seen.add(index)
        expected = plan["resolved_config"]["execution"]["hosts"][index]
        if (
            member["ssh_target"] != expected["ssh_target"]
            or member["advertise_host"] != expected["advertise_host"]
        ):
            raise ValueError("SSH deployment host differs from the frozen plan")
        descriptor = member["descriptor"]
        if (
            member.get("worker_path") != descriptor.get("worker_path")
            or member.get("worker_path")
            != str(
                Path(descriptor["source_dir"]) / "rtp_llm/test/perf_test/ssh/worker.py"
            )
            or member.get("descriptor_path")
            != str(Path(descriptor["run_dir"]) / "descriptor.json")
        ):
            raise ValueError("SSH deployment worker/descriptor path mismatch")
        if (
            descriptor.get("controller_plan_hash") != deployment["controller_plan_hash"]
            or descriptor.get("run_id") != deployment["run_id"]
            or descriptor.get("member_id") != member["member_id"]
        ):
            raise ValueError("SSH member descriptor identity mismatch")
    return home, deployment


def maintain(run_dir, operation, dry_run=False):
    if operation not in {"status", "cancel", "collect", "clean"}:
        raise ValueError("unsupported SSH maintenance operation")
    home, deployment = read_deployment(run_dir)
    execution = deployment["execution"]
    result = {
        "operation": operation,
        "run_id": deployment["run_id"],
        "members": {},
        "errors": {},
    }
    for member in deployment["members"]:
        member_id = member["member_id"]
        remote = RemoteHost(
            member["ssh_target"],
            user=execution.get("run_user"),
            port=execution.get("ssh_port"),
            python_bin=execution["python_bin"],
            command_log=home / "commands.jsonl",
        )
        if dry_run:
            result["members"][member_id] = {
                "argv": remote.command(
                    [
                        execution["python_bin"],
                        member["worker_path"],
                        "--descriptor",
                        member["descriptor_path"],
                        "--operation",
                        operation,
                    ]
                )
            }
            continue
        try:
            if operation == "collect":
                # A distinct destination preserves previous evidence and terminal state.
                import uuid

                destination = home / "maintenance" / uuid.uuid4().hex / member_id
                remote.fetch_tree(member["descriptor"]["run_dir"], destination)
                result["members"][member_id] = {"collected": str(destination)}
            else:
                result["members"][member_id] = remote.worker(
                    member["worker_path"],
                    member["descriptor_path"],
                    operation,
                    timeout=45,
                )
        except Exception as error:
            result["errors"][member_id] = str(error)
    result["exit_code"] = 1 if result["errors"] else 0
    return result
