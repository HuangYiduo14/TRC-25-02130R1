"""Runtime, memory, and disk guardrails for long Prompt 4 runs."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

import psutil

from .partC_rev1_paths import partC_rev1_resolve_path


SECONDS_PER_HOUR = 3600.0
PARTC_REV1_DEFAULT_MAX_RUN_SECONDS = 24.0 * SECONDS_PER_HOUR


@dataclass(frozen=True, slots=True)
class PartCRev1ResourceSnapshot:
    elapsed_seconds: float
    memory_rss_mb: float
    system_available_mb: float
    disk_total_mb: float
    disk_free_mb: float
    temp_disk_mb: float


@dataclass(frozen=True, slots=True)
class PartCRev1RuntimeProjection:
    completed: int
    total: int | None
    elapsed_seconds: float
    median_unit_seconds: float | None
    projected_total_seconds: float | None
    projected_remaining_seconds: float | None
    should_stop: bool
    stop_reason: str | None


def partC_rev1_directory_size_mb(path: str | Path) -> float:
    resolved = partC_rev1_resolve_path(path)
    if not resolved.exists():
        return 0.0
    if resolved.is_file():
        return resolved.stat().st_size / (1024.0 * 1024.0)
    total = 0
    for child in resolved.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total / (1024.0 * 1024.0)


def partC_rev1_resource_snapshot(started_perf: float, *, disk_path: str | Path = ".", temp_path: str | Path = "partC_rev1_cache") -> PartCRev1ResourceSnapshot:
    usage = shutil.disk_usage(partC_rev1_resolve_path(disk_path))
    memory = psutil.virtual_memory()
    return PartCRev1ResourceSnapshot(
        elapsed_seconds=max(time.perf_counter() - started_perf, 0.0),
        memory_rss_mb=psutil.Process().memory_info().rss / (1024.0 * 1024.0),
        system_available_mb=memory.available / (1024.0 * 1024.0),
        disk_total_mb=usage.total / (1024.0 * 1024.0),
        disk_free_mb=usage.free / (1024.0 * 1024.0),
        temp_disk_mb=partC_rev1_directory_size_mb(temp_path),
    )


def partC_rev1_project_runtime(
    *,
    completed: int,
    total: int | None,
    started_perf: float,
    completed_unit_seconds: Iterable[float] = (),
    max_run_seconds: float = PARTC_REV1_DEFAULT_MAX_RUN_SECONDS,
) -> PartCRev1RuntimeProjection:
    elapsed = max(time.perf_counter() - started_perf, 0.0)
    observed = [float(value) for value in completed_unit_seconds if float(value) > 0]
    med = median(observed) if observed else None
    projected_total = None
    projected_remaining = None
    if total is not None and total > 0:
        if med is not None:
            projected_total = med * total
            projected_remaining = med * max(total - completed, 0)
        elif completed > 0:
            rate = completed / max(elapsed, 1e-9)
            projected_total = total / max(rate, 1e-9)
            projected_remaining = max(total - completed, 0) / max(rate, 1e-9)
    should_stop = False
    stop_reason = None
    candidate = projected_total if projected_total is not None else elapsed
    if candidate > max_run_seconds:
        should_stop = True
        stop_reason = "projected_runtime_guard"
    return PartCRev1RuntimeProjection(
        completed=completed,
        total=total,
        elapsed_seconds=elapsed,
        median_unit_seconds=med,
        projected_total_seconds=projected_total,
        projected_remaining_seconds=projected_remaining,
        should_stop=should_stop,
        stop_reason=stop_reason,
    )
