"""Phase 3A process-tree resource guards, heavy ledger, and admission policy."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psutil


GIB = 1024**3
TASK1_HEAVY_SECONDS_HARD = 5_400.0
PHASE3A_HEAVY_SECONDS_HARD = 43_200.0
TASK2_SEALING_RESERVE_SECONDS = 5_400.0
PROCESS_TREE_RSS_HARD_BYTES = 24 * GIB
PROCESS_TREE_RSS_WARNING_BYTES = 20 * GIB
ADDED_DISK_HARD_BYTES = 150 * GIB
ADDED_DISK_WARNING_BYTES = 50 * GIB
FREE_DISK_HEADROOM_BYTES = 20 * GIB


class Phase3AResourceError(RuntimeError):
    """Raised when a hard Phase 3A resource guard requires safe stop."""


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    wall_seconds_hard: float
    rss_bytes_warning: int = PROCESS_TREE_RSS_WARNING_BYTES
    rss_bytes_hard: int = PROCESS_TREE_RSS_HARD_BYTES
    added_disk_bytes_warning: int = ADDED_DISK_WARNING_BYTES
    added_disk_bytes_hard: int = ADDED_DISK_HARD_BYTES
    free_disk_headroom_bytes: int = FREE_DISK_HEADROOM_BYTES


@dataclass(frozen=True, slots=True)
class ResourceSample:
    timestamp_utc: str
    stage_id: str
    elapsed_seconds: float
    process_tree_rss_bytes: int
    process_tree_peak_rss_bytes: int
    process_count: int
    output_added_bytes: int
    temp_added_bytes: int
    free_disk_bytes: int


@dataclass(frozen=True, slots=True)
class ResourceDecision:
    action: str
    reason_codes: tuple[str, ...]
    checkpoint_required: bool
    admit_new_work: bool
    scientific_rules_unchanged: bool = True


@dataclass(slots=True)
class HeavyResourceLedger:
    command_records: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        command_id: str,
        tag: str,
        elapsed_seconds: float,
        peak_rss_bytes: int,
        added_disk_bytes: int,
        status: str,
    ) -> None:
        heavy = str(tag).upper() == "HEAVY_REAL_DATA"
        self.command_records.append(
            {
                "command_id": str(command_id),
                "tag": str(tag),
                "heavy_real_data": heavy,
                "elapsed_seconds": float(elapsed_seconds),
                "charged_heavy_seconds": float(elapsed_seconds) if heavy else 0.0,
                "peak_process_tree_rss_bytes": int(peak_rss_bytes),
                "added_disk_bytes": int(added_disk_bytes),
                "status": str(status),
            }
        )

    @property
    def heavy_seconds(self) -> float:
        return float(sum(row["charged_heavy_seconds"] for row in self.command_records))

    @property
    def elapsed_seconds(self) -> float:
        return float(sum(row["elapsed_seconds"] for row in self.command_records))

    @property
    def peak_rss_bytes(self) -> int:
        return max((int(row["peak_process_tree_rss_bytes"]) for row in self.command_records), default=0)

    @property
    def added_disk_bytes(self) -> int:
        return max((int(row["added_disk_bytes"]) for row in self.command_records), default=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "tagging_rule": "HEAVY_REAL_DATA iff command uses authoritative/substantial real data; coding and small synthetic tests are uncharged",
            "commands": list(self.command_records),
            "total_elapsed_seconds": self.elapsed_seconds,
            "heavy_real_data_seconds": self.heavy_seconds,
            "peak_process_tree_rss_bytes": self.peak_rss_bytes,
            "added_disk_bytes": self.added_disk_bytes,
        }


class ProcessTreeResourceGuard:
    """Sample the orchestrator plus descendants and checkpoint before abort."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        temp_root: str | Path,
        stage_id: str,
        budget: ResourceBudget,
        checkpoint_callback: Callable[[ResourceSample, ResourceDecision], None] | None = None,
        pid: int | None = None,
    ) -> None:
        self.output_root = Path(output_root).resolve(strict=True)
        self.temp_root = Path(temp_root).resolve(strict=True)
        self.stage_id = str(stage_id)
        self.budget = budget
        self.checkpoint_callback = checkpoint_callback
        self.pid = os.getpid() if pid is None else int(pid)
        self.started = time.monotonic()
        self.baseline_output = directory_bytes(self.output_root)
        self.baseline_temp = directory_bytes(self.temp_root)
        self.peak_rss = 0

    def sample(self) -> tuple[ResourceSample, ResourceDecision]:
        rss, process_count = process_tree_rss(self.pid)
        self.peak_rss = max(self.peak_rss, rss)
        output_added = max(0, directory_bytes(self.output_root) - self.baseline_output)
        temp_added = max(0, directory_bytes(self.temp_root) - self.baseline_temp)
        sample = ResourceSample(
            timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            stage_id=self.stage_id,
            elapsed_seconds=float(time.monotonic() - self.started),
            process_tree_rss_bytes=rss,
            process_tree_peak_rss_bytes=self.peak_rss,
            process_count=process_count,
            output_added_bytes=output_added,
            temp_added_bytes=temp_added,
            free_disk_bytes=int(psutil.disk_usage(str(self.output_root)).free),
        )
        decision = evaluate_resource_sample(sample, self.budget)
        if decision.checkpoint_required and self.checkpoint_callback is not None:
            self.checkpoint_callback(sample, decision)
        if decision.action == "HARD_STOP":
            raise Phase3AResourceError(",".join(decision.reason_codes))
        return sample, decision


def process_tree_rss(pid: int) -> tuple[int, int]:
    try:
        root = psutil.Process(int(pid))
        processes = [root, *root.children(recursive=True)]
    except (psutil.Error, ValueError) as exc:
        raise Phase3AResourceError(f"Cannot inspect process tree {pid}") from exc
    total = 0
    count = 0
    for process in processes:
        try:
            total += int(process.memory_info().rss)
            count += 1
        except psutil.Error:
            continue
    return total, count


def directory_bytes(root: str | Path) -> int:
    base = Path(root).resolve(strict=True)
    total = 0
    for directory, _, files in os.walk(base):
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                raise Phase3AResourceError(f"Resource tree contains a symlink: {path}")
            total += int(path.stat().st_size)
    return total


def evaluate_resource_sample(sample: ResourceSample, budget: ResourceBudget) -> ResourceDecision:
    added = int(sample.output_added_bytes + sample.temp_added_bytes)
    hard: list[str] = []
    warning: list[str] = []
    if sample.elapsed_seconds >= budget.wall_seconds_hard:
        hard.append("RESOURCE_ABORTED_WALL")
    if sample.process_tree_rss_bytes >= budget.rss_bytes_hard:
        hard.append("RESOURCE_ABORTED_RSS")
    if added >= budget.added_disk_bytes_hard or sample.free_disk_bytes <= budget.free_disk_headroom_bytes:
        hard.append("RESOURCE_ABORTED_DISK")
    if sample.process_tree_rss_bytes >= budget.rss_bytes_warning:
        warning.append("RSS_WARNING_REFORECAST")
    if added >= budget.added_disk_bytes_warning:
        warning.append("DISK_WARNING_REFORECAST")
    if hard:
        return ResourceDecision("HARD_STOP", tuple(hard), True, False)
    if warning:
        return ResourceDecision("WARN_REFORECAST", tuple(warning), False, True)
    return ResourceDecision("PASS", (), False, True)


def task2_admission_decision(
    *,
    approved_task1_heavy_seconds: float,
    task2_spent_seconds: float,
    forecast_batch_seconds: float,
    forecast_unfinished_mandatory_seconds: float,
    sealing_reserve_seconds: float = TASK2_SEALING_RESERVE_SECONDS,
) -> dict[str, Any]:
    remaining = PHASE3A_HEAVY_SECONDS_HARD - float(approved_task1_heavy_seconds) - float(task2_spent_seconds)
    required = float(forecast_batch_seconds) + float(forecast_unfinished_mandatory_seconds) + float(sealing_reserve_seconds)
    admit = required <= remaining
    return {
        "remaining_budget_seconds": remaining,
        "forecast_batch_seconds": float(forecast_batch_seconds),
        "forecast_unfinished_mandatory_seconds": float(forecast_unfinished_mandatory_seconds),
        "sealing_reserve_seconds": float(sealing_reserve_seconds),
        "required_seconds": required,
        "admit": bool(admit),
        "reason_code": "ADMITTED_WITH_RESERVE" if admit else "SHED_OR_STOP_FOR_RESERVE",
        "scientific_rules_unchanged": True,
    }


def policy_evidence() -> dict[str, Any]:
    return {
        "network_disabled": True,
        "package_installation_prohibited": True,
        "write_capable_orchestrators": 1,
        "resource_shedding_order": [
            "omit_optional_full_SL_ablation",
            "omit_optional_global_scale",
            "reduce_structural_mode_candidates",
            "omit_nonreserved_cheap_census_candidates",
            "reduce_OD_target_to_100_from_20_origins",
            "stop_admitting_new_finalists",
            "seal_valid_evidence",
        ],
        "never_shed": [
            "input_verification", "pairwise_validity", "bundle_closure", "centroid_semantics",
            "completed_finalist_projection_liftability", "status_records", "topology_reproduction",
            "final_validation_manifests",
        ],
    }
