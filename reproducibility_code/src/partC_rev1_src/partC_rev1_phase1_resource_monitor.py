"""Single-process resource sampling and graceful-stop decisions."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import psutil


class PartCRev1ResourceMonitorError(RuntimeError):
    """Raised when resource state cannot be sampled or monitored safely."""


@dataclass(frozen=True, slots=True)
class PartCRev1ResourceSample:
    timestamp_utc: str
    stage_id: str
    task_id: str | None
    pid: int
    rss_bytes: int
    system_available_memory_bytes: int
    output_added_bytes: int
    temp_added_bytes: int
    output_volume_free_bytes: int
    elapsed_seconds: float
    state: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PartCRev1ResourceDecision:
    action: str
    reasons: tuple[str, ...]
    warning: bool
    stop_starting_new_tasks: bool
    hard_stop: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


def _tree_bytes(root: Path) -> int:
    if not root.is_dir():
        raise PartCRev1ResourceMonitorError(f"Resource root is not a directory: {root}")
    total = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise PartCRev1ResourceMonitorError(f"Resource tree contains a symlink/reparse entry: {entry.path}")
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += int(entry.stat(follow_symlinks=False).st_size)
    return total


def partC_rev1_sample_resources(
    output_root: str | Path,
    temp_root: str | Path,
    *,
    stage_id: str,
    task_id: str | None,
    state: str,
    started_monotonic: float,
    baseline_output_bytes: int = 0,
    baseline_temp_bytes: int = 0,
    pid: int | None = None,
) -> PartCRev1ResourceSample:
    """Sample one process and the two exact campaign-owned write trees."""

    if not stage_id.strip() or not state.strip():
        raise PartCRev1ResourceMonitorError("stage_id and state must be explicit")
    output = Path(output_root).resolve(strict=True)
    temporary = Path(temp_root).resolve(strict=True)
    process_id = os.getpid() if pid is None else int(pid)
    if process_id <= 0:
        raise PartCRev1ResourceMonitorError("Resource sample pid must be positive")
    process = psutil.Process(process_id)
    output_bytes = _tree_bytes(output)
    temp_bytes = _tree_bytes(temporary)
    elapsed = float(time.monotonic() - float(started_monotonic))
    if elapsed < 0:
        raise PartCRev1ResourceMonitorError("Resource elapsed time cannot be negative")
    return PartCRev1ResourceSample(
        timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        stage_id=str(stage_id),
        task_id=None if task_id is None else str(task_id),
        pid=process_id,
        rss_bytes=int(process.memory_info().rss),
        system_available_memory_bytes=int(psutil.virtual_memory().available),
        output_added_bytes=max(0, int(output_bytes) - int(baseline_output_bytes)),
        temp_added_bytes=max(0, int(temp_bytes) - int(baseline_temp_bytes)),
        output_volume_free_bytes=int(shutil.disk_usage(output).free),
        elapsed_seconds=elapsed,
        state=str(state),
    )


def partC_rev1_evaluate_resource_policy(
    sample: PartCRev1ResourceSample, policy: Mapping[str, Any]
) -> PartCRev1ResourceDecision:
    """Return PASS, WARN, STOP_NEW_TASKS, or HARD_STOP from frozen thresholds."""

    required = (
        "wall_clock_seconds",
        "process_rss_bytes",
        "added_disk_bytes",
        "system_available_memory_bytes",
        "output_volume_free_bytes",
    )
    if any(key not in policy for key in required):
        raise PartCRev1ResourceMonitorError("Resource policy is missing a required threshold section")
    added_disk = sample.output_added_bytes + sample.temp_added_bytes
    values = {
        "wall_clock_seconds": sample.elapsed_seconds,
        "process_rss_bytes": sample.rss_bytes,
        "added_disk_bytes": added_disk,
    }
    hard: list[str] = []
    stop: list[str] = []
    warning: list[str] = []
    for name, observed in values.items():
        thresholds = policy[name]
        if observed >= float(thresholds["hard_stop"]):
            hard.append(f"{name}={observed} reached hard_stop={thresholds['hard_stop']}")
        elif observed >= float(thresholds["stop_starting_new_tasks"]):
            stop.append(
                f"{name}={observed} reached stop_starting_new_tasks={thresholds['stop_starting_new_tasks']}"
            )
        elif observed >= float(thresholds["warning"]):
            warning.append(f"{name}={observed} reached warning={thresholds['warning']}")
    available = policy["system_available_memory_bytes"]
    if sample.system_available_memory_bytes <= int(available["hard_stop_below"]):
        hard.append("system available memory reached the hard-stop floor")
    elif sample.system_available_memory_bytes <= int(available["warning_below"]):
        warning.append("system available memory reached the warning floor")
    free_disk = policy["output_volume_free_bytes"]
    if sample.output_volume_free_bytes <= int(free_disk["hard_stop_below"]):
        hard.append("output-volume free space reached the hard-stop floor")
    elif sample.output_volume_free_bytes <= int(free_disk["warning_below"]):
        warning.append("output-volume free space reached the warning floor")
    if hard:
        return PartCRev1ResourceDecision("HARD_STOP", tuple(hard), True, True, True)
    if stop:
        return PartCRev1ResourceDecision("STOP_NEW_TASKS", tuple(stop), True, True, False)
    if warning:
        return PartCRev1ResourceDecision("WARN", tuple(warning), True, False, False)
    return PartCRev1ResourceDecision("PASS", (), False, False, False)


def partC_rev1_write_heartbeat(
    path: str | Path,
    sample: PartCRev1ResourceSample,
    decision: PartCRev1ResourceDecision,
) -> Path:
    destination = Path(path).resolve(strict=False)
    if not destination.parent.is_dir():
        raise PartCRev1ResourceMonitorError("Heartbeat parent must exist after path validation")
    payload = {
        "schema_id": "partC_rev1_phase1_resource_sample",
        "schema_version": "1.0",
        "sample": sample.to_dict(),
        "decision": decision.to_dict(),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


class PartCRev1ResourceMonitor:
    """A single-process context manager with periodic heartbeat sampling."""

    _active_lock = threading.Lock()
    _active_pids: set[int] = set()

    def __init__(
        self,
        output_root: str | Path,
        temp_root: str | Path,
        policy: Mapping[str, Any],
        heartbeat_path: str | Path,
        *,
        stage_id: str,
        task_id: str | None,
        interval_seconds: float | None = None,
    ) -> None:
        self.output_root = Path(output_root).resolve(strict=True)
        self.temp_root = Path(temp_root).resolve(strict=True)
        self.policy = dict(policy)
        self.heartbeat_path = Path(heartbeat_path).resolve(strict=False)
        self.stage_id = str(stage_id)
        self.task_id = None if task_id is None else str(task_id)
        configured = float(self.policy.get("sampling", {}).get("heartbeat_interval_seconds", 300))
        self.interval_seconds = configured if interval_seconds is None else float(interval_seconds)
        if self.interval_seconds <= 0:
            raise PartCRev1ResourceMonitorError("Heartbeat interval must be positive")
        self.pid = os.getpid()
        self.started_monotonic = 0.0
        self.baseline_output_bytes = 0
        self.baseline_temp_bytes = 0
        self.latest_sample: PartCRev1ResourceSample | None = None
        self.latest_decision: PartCRev1ResourceDecision | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self, state: str = "RUNNING") -> tuple[PartCRev1ResourceSample, PartCRev1ResourceDecision]:
        if self.started_monotonic <= 0:
            raise PartCRev1ResourceMonitorError("Resource monitor has not entered its context")
        sample = partC_rev1_sample_resources(
            self.output_root,
            self.temp_root,
            stage_id=self.stage_id,
            task_id=self.task_id,
            state=state,
            started_monotonic=self.started_monotonic,
            baseline_output_bytes=self.baseline_output_bytes,
            baseline_temp_bytes=self.baseline_temp_bytes,
            pid=self.pid,
        )
        decision = partC_rev1_evaluate_resource_policy(sample, self.policy)
        partC_rev1_write_heartbeat(self.heartbeat_path, sample, decision)
        self.latest_sample = sample
        self.latest_decision = decision
        return sample, decision

    def _periodic(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                _, decision = self.sample("RUNNING")
            except Exception:
                self._stop.set()
                return
            if decision.hard_stop:
                self._stop.set()
                return

    def __enter__(self) -> "PartCRev1ResourceMonitor":
        with self._active_lock:
            if self.pid in self._active_pids:
                raise PartCRev1ResourceMonitorError("Only one resource monitor may write in a process")
            self._active_pids.add(self.pid)
        try:
            self.started_monotonic = time.monotonic()
            self.baseline_output_bytes = _tree_bytes(self.output_root)
            self.baseline_temp_bytes = _tree_bytes(self.temp_root)
            self.sample("STARTED")
            self._thread = threading.Thread(target=self._periodic, name="partC-rev1-resource-monitor", daemon=True)
            self._thread.start()
            return self
        except Exception:
            with self._active_lock:
                self._active_pids.discard(self.pid)
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, min(self.interval_seconds * 2.0, 5.0)))
        try:
            self.sample("FAILED" if exc_type is not None else "COMPLETED")
        finally:
            with self._active_lock:
                self._active_pids.discard(self.pid)
        return False
