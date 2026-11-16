"""Explicit command interface for the authorized Phase 3 all-suite campaign."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Sequence

from .production import (
    Phase3ProductionError,
    build_suite,
    _replace_json,
    _write_json_new,
    load_production_config,
    prepare_campaign,
    required_suite_ids,
    resource_snapshot,
    reuse_s03,
    utc_now,
)
from .resources import directory_bytes, process_tree_rss
from .production_portfolio import (
    build_all_suite_geopackage,
    build_all_suite_sqlite,
    build_summary_tables_and_figures,
)
from .production_handoff import build_final_handoff, validate_all_suite_campaign
from .production_closeout import prepare_five_suite_closeout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "prepare", "prepare-closeout", "reuse-s03", "build-suite", "assemble-sqlite",
            "assemble-gpkg", "tables-figures", "validate", "handoff",
            "resource-snapshot",
        ),
    )
    parser.add_argument("--tasks", required=True, nargs="+")
    parser.add_argument("--suite-id", choices=("s01", "s02", "s03", "s04", "s05", "s06"))
    parser.add_argument("--confirm-real-run", action="store_true")
    return parser


def validate_invocation(args: argparse.Namespace, config: dict) -> None:
    for key in ("phase_id", "step_id", "campaign_id", "run_id"):
        argument = getattr(args, key)
        if argument != config[key]:
            raise Phase3ProductionError(f"Invocation {key} differs from the execution config")
    if not args.tasks or args.task_id not in args.tasks:
        raise Phase3ProductionError("Invocation requires a nonempty task list containing task-id")
    unknown = set(args.tasks).difference(map(str, config["tasks"]))
    if unknown:
        raise Phase3ProductionError(f"Invocation names tasks outside the frozen matrix: {sorted(unknown)}")
    if args.mode == "build-suite" and (args.suite_id is None or args.suite_id == "s03"):
        raise Phase3ProductionError("build-suite requires one non-s03 suite ID")
    if args.mode == "build-suite" and args.suite_id not in required_suite_ids(config):
        raise Phase3ProductionError("build-suite cannot run an excluded production suite")
    if args.mode != "build-suite" and args.suite_id is not None:
        raise Phase3ProductionError("suite-id is allowed only for build-suite")
    if args.mode != "resource-snapshot" and not args.confirm_real_run:
        raise Phase3ProductionError("Real production mutations require --confirm-real-run")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_production_config(args.config)
    validate_invocation(args, config)
    samples = [process_tree_rss(os.getpid())]
    stop = threading.Event()
    run_root = Path(config["output_root"])
    temp_root = Path(config["temp_root"])
    before_disk = (
        (directory_bytes(run_root) if run_root.exists() else 0)
        + (directory_bytes(temp_root) if temp_root.exists() else 0)
    )
    started_utc = utc_now()
    invocation_path = _start_invocation(run_root, args, started_utc) if run_root.is_dir() else None

    def sample() -> None:
        while not stop.wait(0.1):
            samples.append(process_tree_rss(os.getpid()))

    sampler = threading.Thread(target=sample, name=f"phase3-production-{args.task_id}", daemon=True)
    sampler.start()
    began = time.perf_counter()
    result: dict = {}
    failure: BaseException | None = None
    try:
        if args.mode == "prepare":
            result = prepare_campaign(config, args.config)
        elif args.mode == "prepare-closeout":
            result = prepare_five_suite_closeout(config, args.config)
        elif args.mode == "reuse-s03":
            result = reuse_s03(config)
        elif args.mode == "build-suite":
            assert args.suite_id is not None
            result = build_suite(config, args.suite_id)
        elif args.mode == "assemble-sqlite":
            result = build_all_suite_sqlite(config)
        elif args.mode == "assemble-gpkg":
            result = build_all_suite_geopackage(config)
        elif args.mode == "tables-figures":
            result = build_summary_tables_and_figures(config)
        elif args.mode == "validate":
            result = validate_all_suite_campaign(config)
        elif args.mode == "handoff":
            result = build_final_handoff(config, str(config["handoff_id"]))
        else:
            run_root = Path(config["output_root"]).resolve(strict=True)
            temp_root = Path(config["temp_root"]).resolve(strict=True)
            result = resource_snapshot(run_root, temp_root)
    except BaseException as exc:
        failure = exc
    finally:
        stop.set()
        sampler.join(timeout=2.0)
    samples.append(process_tree_rss(os.getpid()))
    elapsed = time.perf_counter() - began
    peak_rss = max(value[0] for value in samples)
    peak_processes = max(value[1] for value in samples)
    run_root = Path(config["output_root"])
    temp_root = Path(config["temp_root"])
    after_disk = (
        (directory_bytes(run_root) if run_root.exists() else 0)
        + (directory_bytes(temp_root) if temp_root.exists() else 0)
    )
    rss_exceeded = peak_rss > int(config["resources"]["process_tree_rss_bytes"])
    resource = {
        "schema_id": "phase3.cli_task_resource.v1",
        "task_id": args.task_id,
        "mode": args.mode,
        "tasks": list(args.tasks),
        "status": "FAIL" if failure is not None or rss_exceeded else "PASS",
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "peak_process_tree_rss_bytes": int(peak_rss),
        "peak_process_count": int(peak_processes),
        "sample_count": len(samples),
        "run_root_bytes": directory_bytes(run_root) if run_root.exists() else 0,
        "temp_root_bytes": directory_bytes(temp_root) if temp_root.exists() else 0,
        "added_disk_bytes": int(max(0, after_disk - before_disk)),
        "failure_type": type(failure).__name__ if failure is not None else None,
        "failure_message": str(failure) if failure is not None else None,
        "rss_hard_limit_exceeded": rss_exceeded,
    }
    if run_root.is_dir():
        receipt_root = run_root / "evidence" / "cli_resource_receipts"
        receipt_root.mkdir(parents=True, exist_ok=True)
        receipt_path = _next_attempt_path(receipt_root, args.task_id)
        _write_json_new(receipt_path, resource)
        if invocation_path is None:
            invocation_path = _start_invocation(run_root, args, started_utc)
        invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
        invocation.update(
            {
                "status": resource["status"],
                "completed_utc": resource["completed_utc"],
                "elapsed_seconds": elapsed,
                "resource_receipt": str(receipt_path),
                "failure_type": resource["failure_type"],
                "failure_message": resource["failure_message"],
            }
        )
        _replace_json(invocation_path, invocation)
    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    if rss_exceeded:
        raise Phase3ProductionError("CLI task exceeded the 24 GiB process-tree RSS hard limit")
    result = {**result, "cli_resource": resource}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _next_attempt_path(root: Path, task_id: str) -> Path:
    attempt = len(list(root.glob(f"{task_id}__attempt_*.json"))) + 1
    return root / f"{task_id}__attempt_{attempt:03d}.json"


def _start_invocation(root: Path, args: argparse.Namespace, started_utc: str) -> Path:
    invocation_root = root / "evidence" / "cli_invocations"
    invocation_root.mkdir(parents=True, exist_ok=True)
    path = _next_attempt_path(invocation_root, args.task_id)
    _write_json_new(
        path,
        {
            "schema_id": "phase3.cli_invocation.v1",
            "phase_id": args.phase_id,
            "step_id": args.step_id,
            "campaign_id": args.campaign_id,
            "run_id": args.run_id,
            "task_id": args.task_id,
            "mode": args.mode,
            "tasks": list(args.tasks),
            "suite_id": args.suite_id,
            "config": str(Path(args.config).resolve(strict=True)),
            "confirm_real_run": bool(args.confirm_real_run),
            "status": "RUNNING",
            "started_utc": started_utc,
        },
    )
    return path


if __name__ == "__main__":
    raise SystemExit(main())
