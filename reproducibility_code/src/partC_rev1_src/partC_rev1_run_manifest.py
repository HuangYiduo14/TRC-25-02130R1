"""Run manifest and master-state helpers for Prompt 4 stages."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .partC_rev1_io import partC_rev1_write_json
from .partC_rev1_paths import partC_rev1_ensure_directory, partC_rev1_resolve_path


def partC_rev1_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def partC_rev1_write_master_state(path: str | Path, state: Mapping[str, Any]) -> Path:
    resolved = partC_rev1_resolve_path(path)
    payload = {"updated_utc": partC_rev1_utc_now(), **dict(state)}
    partC_rev1_write_json(payload, resolved)
    return resolved


def partC_rev1_append_jsonl(path: str | Path, record: Mapping[str, Any]) -> Path:
    resolved = partC_rev1_resolve_path(path)
    partC_rev1_ensure_directory(resolved.parent)
    payload = {"utc": partC_rev1_utc_now(), **dict(record)}
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    return resolved


def partC_rev1_stage_manifest(
    *,
    stage: str,
    config_path: str | Path,
    graph_hash: str | None = None,
    od_hash: str | None = None,
    partition_hash: str | None = None,
    artifacts: list[Mapping[str, Any]] | None = None,
    tests: list[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "created_utc": partC_rev1_utc_now(),
        "config_path": str(config_path),
        "graph_hash": graph_hash,
        "od_hash": od_hash,
        "partition_hash": partition_hash,
        "artifacts": [dict(item) for item in artifacts or []],
        "tests": [dict(item) for item in tests or []],
        "metadata": dict(metadata or {}),
    }
