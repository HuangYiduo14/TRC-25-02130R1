"""Explicit CLI for the repaired restartable Phase 5 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .staged_pilot import (
    authorize_engineering_retry,
    reseal_repaired_output_metadata,
    resume_run,
    seal_blocker_command,
    run_stage_command,
    run_status,
    test_hang_worker,
    worker_entry,
)
from .repaired_handoff import build_repaired_handoff


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="validate and print stage status")
    status.add_argument("--config", type=Path, required=True)
    run = commands.add_parser("run-stage", help="run one public repaired stage")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--stage", required=True)
    run.add_argument("--retry-reason")
    resume = commands.add_parser(
        "resume", help="resume from the first incomplete stage"
    )
    resume.add_argument("--config", type=Path, required=True)
    resume.add_argument("--retry-reason")
    reseal = commands.add_parser(
        "reseal", help="bind the post-run result commit and regenerate manifests"
    )
    reseal.add_argument("--config", type=Path, required=True)
    reseal.add_argument("--ending-head", required=True)
    handoff = commands.add_parser(
        "handoff", help="build one compact repaired synchronization handoff"
    )
    handoff.add_argument("--config", type=Path, required=True)
    handoff.add_argument("--handoff-id", required=True)
    handoff.add_argument("--ending-head", required=True)
    blocker = commands.add_parser(
        "seal-blocker", help="seal a reviewed exact blocker without retrying"
    )
    blocker.add_argument("--config", type=Path, required=True)
    blocker.add_argument("--blocker", required=True)
    authorize = commands.add_parser(
        "authorize-retry",
        help="bind one preserved failed attempt to a committed engineering repair",
    )
    authorize.add_argument("--config", type=Path, required=True)
    authorize.add_argument("--stage", required=True)
    authorize.add_argument("--retry-reason", required=True)
    authorize.add_argument("--validation-record", type=Path, required=True)
    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--stage", required=True)
    worker.add_argument("--attempt-dir", type=Path, required=True)
    worker.add_argument("--pre-run-code-head", required=True)
    worker.add_argument("--attempt-code-head", required=True)
    worker.add_argument("--engineering-retry-epoch-path")
    worker.add_argument("--engineering-retry-epoch-sha256")
    hang = commands.add_parser("test-hang-worker", help=argparse.SUPPRESS)
    hang.add_argument("--stack-file", type=Path, required=True)
    hang.add_argument("--stack-dump-seconds", type=float, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "status":
        print(json.dumps(run_status(arguments.config), indent=2, sort_keys=True))
        return 0
    if arguments.command == "run-stage":
        print(
            run_stage_command(
                arguments.config,
                arguments.stage,
                retry_reason=arguments.retry_reason,
            )
        )
        return 0
    if arguments.command == "resume":
        print(resume_run(arguments.config, retry_reason=arguments.retry_reason))
        return 0
    if arguments.command == "reseal":
        print(reseal_repaired_output_metadata(arguments.config, arguments.ending_head))
        return 0
    if arguments.command == "handoff":
        print(
            build_repaired_handoff(
                arguments.config,
                handoff_id=arguments.handoff_id,
                ending_head=arguments.ending_head,
            )
        )
        return 0
    if arguments.command == "seal-blocker":
        print(seal_blocker_command(arguments.config, arguments.blocker))
        return 0
    if arguments.command == "authorize-retry":
        print(
            authorize_engineering_retry(
                arguments.config,
                stage_id=arguments.stage,
                retry_reason=arguments.retry_reason,
                validation_record=arguments.validation_record,
            )
        )
        return 0
    if arguments.command == "worker":
        return worker_entry(
            config_path=arguments.config,
            stage_id=arguments.stage,
            attempt_dir=arguments.attempt_dir,
            pre_run_code_head=arguments.pre_run_code_head,
            attempt_code_head=arguments.attempt_code_head,
            engineering_retry_epoch_path=arguments.engineering_retry_epoch_path,
            engineering_retry_epoch_sha256=arguments.engineering_retry_epoch_sha256,
        )
    test_hang_worker(arguments.stack_file, arguments.stack_dump_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
