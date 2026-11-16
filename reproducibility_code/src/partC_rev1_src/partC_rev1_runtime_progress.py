"""Runtime progress and heartbeat helpers for Prompt 4 stages."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from .partC_rev1_paths import partC_rev1_ensure_directory, partC_rev1_resolve_path


def _partC_rev1_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def partC_rev1_memory_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def partC_rev1_disk_usage_mb(path: str | Path) -> float:
    resolved = partC_rev1_resolve_path(path)
    if resolved.is_file():
        return resolved.stat().st_size / (1024.0 * 1024.0)
    if not resolved.exists():
        return 0.0
    total = 0
    for child in resolved.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total / (1024.0 * 1024.0)


@dataclass(slots=True)
class PartCRev1ProgressReporter:
    """Write a stage log plus machine-readable JSONL progress records."""

    stage: str
    output_dir: str | Path = "partC_rev1_outputs"
    heartbeat_seconds: float = 15.0
    log_path: Path = field(init=False)
    progress_path: Path = field(init=False)
    started_perf: float = field(init=False)
    logger: logging.Logger = field(init=False)
    _last_heartbeat_perf: float = field(init=False)

    def __post_init__(self) -> None:
        if not self.stage.startswith("partC_rev1_"):
            raise ValueError(f"stage must start with 'partC_rev1_': {self.stage}")
        self.output_dir = partC_rev1_ensure_directory(partC_rev1_resolve_path(self.output_dir))
        self.log_path = Path(self.output_dir) / f"{self.stage}.log"
        self.progress_path = Path(self.output_dir) / f"{self.stage}_progress.jsonl"
        self.started_perf = time.perf_counter()
        self._last_heartbeat_perf = self.started_perf
        self.logger = logging.getLogger(self.stage)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.emit("stage_started", log_path=str(self.log_path), progress_path=str(self.progress_path))

    def close(self) -> None:
        self.emit("stage_finished")
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_perf

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "utc": _partC_rev1_utc_now(),
            "stage": self.stage,
            "event": event,
            "elapsed_seconds": self.elapsed_seconds(),
            "memory_rss_mb": partC_rev1_memory_rss_mb(),
        }
        record.update(fields)
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self.logger.info("%s %s", event, json.dumps(fields, sort_keys=True, default=str))

    def heartbeat(self, event: str = "heartbeat", *, force: bool = False, **fields: Any) -> None:
        now = time.perf_counter()
        if force or now - self._last_heartbeat_perf >= self.heartbeat_seconds:
            self._last_heartbeat_perf = now
            self.emit(event, **fields)


def partC_rev1_estimate_completion(
    *,
    completed: int,
    total: int | None,
    started_perf: float,
) -> dict[str, float | int | None]:
    elapsed = max(time.perf_counter() - started_perf, 0.0)
    if completed <= 0 or total is None or total <= 0:
        return {"completed": completed, "total": total, "elapsed_seconds": elapsed, "eta_seconds": None}
    rate = completed / max(elapsed, 1e-9)
    remaining = max(total - completed, 0)
    return {
        "completed": completed,
        "total": total,
        "elapsed_seconds": elapsed,
        "eta_seconds": remaining / max(rate, 1e-9),
    }
