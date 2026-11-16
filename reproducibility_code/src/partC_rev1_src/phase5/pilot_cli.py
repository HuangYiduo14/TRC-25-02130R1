"""Explicit command-line boundary for the authorized Phase 5 pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from .handoff import build_handoff
from .pilot import reseal_output_metadata, run_pilot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="execute the single configured pilot")
    run.add_argument("--config", type=Path, required=True)
    reseal = subparsers.add_parser("reseal", help="bind the post-checkpoint Git HEAD")
    reseal.add_argument("--config", type=Path, required=True)
    reseal.add_argument("--ending-head", required=True)
    handoff = subparsers.add_parser("handoff", help="seal the compact review handoff")
    handoff.add_argument("--config", type=Path, required=True)
    handoff.add_argument("--ending-head", required=True)
    handoff.add_argument("--handoff-id", required=True)
    handoff.add_argument("--supersedes-handoff-id")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "run":
        output = run_pilot(arguments.config)
    elif arguments.command == "reseal":
        output = reseal_output_metadata(arguments.config, arguments.ending_head)
    else:
        output = build_handoff(
            arguments.config,
            handoff_id=arguments.handoff_id,
            ending_head=arguments.ending_head,
            supersedes_handoff_id=arguments.supersedes_handoff_id,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
