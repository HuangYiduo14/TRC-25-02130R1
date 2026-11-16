"""Explicit Phase 3A command interface with phase, identity, and path guards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .authority import load_execution_config
from .preflight import PREFLIGHT_SECTIONS, run_preflight_section


MODES = (
    "preflight",
    "round1_geo",
    "round2_aug",
    "round3_finalize",
    "round4_evaluate",
    "validate",
    "reproduce",
    "handoff",
)


class Phase3ACLIError(RuntimeError):
    pass


def validate_invocation(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    expected = {
        "phase_id": "phase_03a",
        "step_id": "bacn_method_selection",
        "campaign_id": str(config["campaign_id"]),
        "run_id": str(config["run_id"]),
    }
    for key, value in expected.items():
        if str(getattr(args, key)) != value:
            raise Phase3ACLIError(f"Invocation {key} differs from the execution config")
    if args.mode not in MODES:
        raise Phase3ACLIError(f"Unsupported Phase 3A mode: {args.mode}")
    if not args.task_id.strip() or not args.tasks:
        raise Phase3ACLIError("Every invocation requires a task ID and a nonempty task list")
    configured_tasks = set(map(str, config.get("tasks", [])))
    if not set(map(str, args.tasks)).issubset(configured_tasks):
        raise Phase3ACLIError("Invocation task list is not a subset of the frozen execution plan")
    if not args.dry_run and not args.confirm_real_run:
        raise Phase3ACLIError("Real-data execution requires --confirm-real-run")
    project = Path(str(config["project_root"])).resolve(strict=True)
    authority = Path(str(config["accepted_phase2_output_root"])).resolve(strict=True)
    output_key = "preflight_output_root" if args.mode == "preflight" else "output_root"
    temp_key = "preflight_temp_root" if args.mode == "preflight" else "temp_root"
    output = Path(str(config[output_key])).resolve(strict=False)
    temporary = Path(str(config[temp_key])).resolve(strict=False)
    for root in (project, authority):
        if _is_within(output, root) or _is_within(temporary, root):
            raise Phase3ACLIError("Output/temp roots may not be inside the repository or accepted authority root")
    if output == temporary or _is_within(output, temporary) or _is_within(temporary, output):
        raise Phase3ACLIError("Final and temporary roots may not overlap")
    if not args.dry_run and (not output.is_dir() or not temporary.is_dir()):
        raise Phase3ACLIError("Explicit campaign output and temporary roots must already exist")
    if args.mode != "preflight" and not args.dry_run:
        scope = str(config.get("execution_scope", "FULL_TASK2"))
        if scope == "FULL_TASK2":
            approval = config.get("task2_approval")
            if not isinstance(approval, Mapping) or not approval.get("authorized"):
                raise Phase3ACLIError("Round/campaign execution is not authorized by this Task 1 config")
            if not approval.get("reviewed_task1_zip_sha256") or not approval.get("reviewed_task1_run_id"):
                raise Phase3ACLIError("Task 2 approval binding placeholders are incomplete")
        elif scope == "BOUNDED_TASK1_CONTINUATION":
            authority = config.get("task1_continuation_authority")
            if not isinstance(authority, Mapping) or not authority.get("authorized"):
                raise Phase3ACLIError("Bounded Task 1 continuation authority is absent")
            if any(bool(authority.get(key)) for key in ("full_task2_authorized", "phase3b_authorized", "phase4_authorized")):
                raise Phase3ACLIError("Bounded continuation authority crosses a phase boundary")
        else:
            raise Phase3ACLIError(f"Unknown execution scope: {scope}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--section", choices=PREFLIGHT_SECTIONS)
    parser.add_argument("--confirm-real-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_execution_config(args.config)
    validate_invocation(args, config)
    if args.dry_run:
        print(json.dumps({"status": "PASS", "dry_run": True, "mode": args.mode, "tasks": args.tasks}, sort_keys=True))
        return 0
    if args.mode == "preflight":
        if args.section is None:
            raise Phase3ACLIError("Preflight mode requires --section")
        result = run_preflight_section(args.config, args.section)
    else:
        from .orchestration import run_approved_mode

        result = run_approved_mode(config, args.mode, tuple(map(str, args.tasks)))
    print(json.dumps({"status": result.get("status", "PASS"), "mode": args.mode}, sort_keys=True))
    return 0


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
