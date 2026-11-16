"""Authorized Phase 4 physical-reference production campaign.

The campaign runs safeguarded CFW with exact simultaneous s02-s06 tracing,
write-new factor attempts, append-safe scalar records, restartable full
checkpoints, the frozen adaptive factor policy, and fresh-process refreshes.
It stops at a numerical/finalization gate; it never authorizes Phase 5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import psutil

from partC_rev1_src.partC_rev1_shortest_paths import partC_rev1_arc_tie_rank

from .aon import AONExecutionConfig, run_exact_aon
from .checkpoint import (
    Phase4RestartState,
    load_phase4_checkpoint,
    load_phase4_checkpoint_aon,
    save_phase4_checkpoint,
)
from .costs import phase4_bpr_cost, phase4_scenario_capacity, refreshed_state_metrics
from .inputs import file_sha256, load_or_build_aggregated_od, load_phase4_graph
from .refinement import _array_payload_sha256, _save_result, _trace_equal, sanitize_prepared_capacity
from .solver import Phase4SolverConfig, solve_assignment
from .tracing import load_suite_trace_maps, pack_trace_maps


ALGORITHM = "SAFEGUARDED_CFW_WITH_IMMEDIATE_ORDINARY_FW_FALLBACK"
SOLVER_METHOD = "SAFEGUARDED_CFW"
SUITE_IDS = ("s02", "s03", "s04", "s05", "s06")
FACTORS = (0.30, 0.45, 0.50)
EARLY_CHECKPOINTS = (1, 2, 3, 5, 10, 15, 25, 50, 75, 100, 150, 200, 300)


class ProductionError(RuntimeError):
    """A frozen production, identity, numerical, or lifecycle gate failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


def _write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False, default=_json_default))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _rss_bytes() -> int:
    process = psutil.Process()
    total = int(process.memory_info().rss)
    for child in process.children(recursive=True):
        try:
            total += int(child.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _directory_size(root: Path) -> int:
    if not root.exists():
        return 0
    return int(sum(path.stat().st_size for path in root.rglob("*") if path.is_file()))


def _git(project_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(project_root), "-c", "core.longpaths=true", *args],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _factor_key(factor: float) -> str:
    return f"{factor:.2f}".replace(".", "p")


def _checkpoint_iterations(first: int, last: int) -> tuple[int, ...]:
    values = {value for value in EARLY_CHECKPOINTS if first <= value <= last}
    values.update(value for value in range(400, last + 1, 100) if value >= first)
    return tuple(sorted(values))


def _state_gap_map(state: Mapping[str, Any]) -> dict[int, float]:
    return {int(key): float(value) for key, value in state.get("state_gaps", {}).items()}


def _progress_assessment(
    state_gaps: Mapping[int, float],
    iteration_seconds: Sequence[float],
    *,
    target: float,
) -> dict[str, Any]:
    ordered = sorted((int(key), float(value)) for key, value in state_gaps.items() if value > 0.0)
    if not ordered:
        return {"promising": False, "reason": "no_positive_refreshed_gap_evidence"}
    best_iteration, best_gap = min(ordered, key=lambda item: item[1])
    final_iteration, final_gap = ordered[-1]
    recent = ordered[-101:]
    start_gap = recent[0][1]
    improvement = max(0.0, (start_gap - best_gap) / max(start_gap, 1e-300))
    recent_best = best_iteration >= recent[0][0]
    forecast_seconds = float("inf")
    slope = None
    fit = ordered[-min(100, len(ordered)) :]
    if len(fit) >= 4:
        x = np.asarray([item[0] for item in fit], dtype=np.float64)
        y = np.log(np.asarray([item[1] for item in fit], dtype=np.float64))
        slope = float(np.polyfit(x, y, 1)[0])
        if slope < 0.0 and best_gap > target:
            updates = max(0.0, (np.log(target) - np.log(best_gap)) / slope)
            seconds_per_update = float(np.median(iteration_seconds[-20:])) if iteration_seconds else 90.0
            forecast_seconds = float(updates * seconds_per_update * 1.25)
        elif best_gap <= target:
            forecast_seconds = 0.0
    condition_one = bool(best_gap <= 3e-4 and recent_best)
    condition_two = bool(improvement >= 0.15 and forecast_seconds <= 21600.0)
    return {
        "promising": bool(condition_one or condition_two),
        "condition_best_lte_3e4_and_recent": condition_one,
        "condition_improvement_15pct_and_forecast_lte_6h": condition_two,
        "best_iteration": best_iteration,
        "best_rg_sptt": best_gap,
        "final_iteration": final_iteration,
        "final_rg_sptt": final_gap,
        "recent_window_first_iteration": recent[0][0],
        "recent_improvement_fraction": improvement,
        "fit_slope_log_gap_per_update": slope,
        "conservative_additional_seconds_to_target": forecast_seconds,
    }


class ProductionCampaign:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve(strict=True)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        frozen = {
            "phase_id": "phase_04",
            "step_id": "physical_reference_production_24h",
            "algorithm": ALGORITHM,
            "solver_method": SOLVER_METHOD,
            "worker_count": 4,
            "gap_denominator": "SPTT",
            "primary_capacity_factor": 0.30,
            "authorized_fallback_factors": [0.45, 0.50],
            "suite_ids": list(SUITE_IDS),
            "excluded_suite_ids": ["s01"],
        }
        for key, expected in frozen.items():
            if self.config.get(key) != expected:
                raise ProductionError(f"Execution config {key} differs from the frozen authority")
        if not self.config.get("tasks") or not self.config.get("expected_outputs") or not self.config.get("acceptance_tests"):
            raise ProductionError("Explicit task, output, and acceptance-test lists must be nonempty")
        resources = self.config["resources"]
        if resources["wall_clock_seconds"] != 86400 or resources["protected_finalization_seconds"] != 7200:
            raise ProductionError("Wall-time or protected reserve differs from the authority")
        if resources["process_tree_rss_bytes"] != 24 * 1024**3:
            raise ProductionError("RSS limit differs from the 24-GiB authority")
        if resources["added_disk_bytes"] != 150 * 1024**3:
            raise ProductionError("Disk limit differs from the 150-GiB authority")
        self.project_root = Path(self.config["project_root"]).resolve(strict=True)
        self.output_root = Path(self.config["output_root"])
        self.temp_root = Path(self.config["temp_root"])
        self.status_path = self.output_root / "status" / "campaign_status.json"
        self.state_path = self.output_root / "status" / "orchestration_state.json"
        self.event_path = self.output_root / "logs" / "campaign_events.jsonl"

    def event(self, event: str, **payload: Any) -> None:
        _append_jsonl(
            self.event_path,
            {
                "timestamp_utc": _utc_now(),
                "event": event,
                "phase_id": self.config["phase_id"],
                "step_id": self.config["step_id"],
                "run_id": self.config["run_id"],
                **payload,
            },
        )

    def status(self, state: str, **payload: Any) -> None:
        prior = json.loads(self.status_path.read_text(encoding="utf-8")) if self.status_path.exists() else {}
        _write_json_atomic(
            self.status_path,
            {
                **prior,
                "schema_id": "phase4.physical_reference_production.status.v1",
                "phase_id": self.config["phase_id"],
                "step_id": self.config["step_id"],
                "run_id": self.config["run_id"],
                "state": state,
                "updated_utc": _utc_now(),
                "phase4_solver_tracer_preflight": "APPROVED_COMPLETE_WITH_NOTES",
                "phase4_capacity_factor_refinement": "APPROVED_COMPLETE_WITH_NOTES",
                "phase4_production_execution_authorized": True,
                "phase4_overall": "EXECUTING",
                "phase5_authorized": False,
                "next_step_authorized": False,
                **payload,
            },
        )

    def _validate_authority(self) -> dict[str, Any]:
        authority = self.config["authority"]
        package = Path(authority["package_path"]).resolve(strict=True)
        launch = Path(authority["launch_path"]).resolve(strict=True)
        if package.stat().st_size != authority["package_size_bytes"] or file_sha256(package) != authority["package_sha256"]:
            raise ProductionError("Production authority package identity changed")
        if launch.stat().st_size != authority["launch_size_bytes"] or file_sha256(launch) != authority["launch_sha256"]:
            raise ProductionError("Production launch-document identity changed")
        with zipfile.ZipFile(package) as archive:
            names = set(archive.namelist())
            verified = 0
            for line in archive.read("CHECKSUMS.sha256").decode("utf-8").splitlines():
                if not line.strip():
                    continue
                digest, relative = line.split("  ", 1)
                if relative not in names or hashlib.sha256(archive.read(relative)).hexdigest() != digest:
                    raise ProductionError(f"Authority checksum mismatch: {relative}")
                verified += 1
            if verified != authority["package_checksum_entries_verified"]:
                raise ProductionError("Authority checksum-entry count changed")
            nested = archive.read(authority["reviewed_handoff_member"])
            source = archive.read(authority["source_snapshot_member"])
            if len(nested) != authority["reviewed_handoff_size_bytes"] or hashlib.sha256(nested).hexdigest() != authority["reviewed_handoff_sha256"]:
                raise ProductionError("Reviewed refinement handoff identity changed")
            if len(source) != authority["source_snapshot_size_bytes"] or hashlib.sha256(source).hexdigest() != authority["source_snapshot_sha256"]:
                raise ProductionError("Reviewed source snapshot identity changed")
        return {
            "status": "PASS",
            "package_path": str(package),
            "package_size_bytes": package.stat().st_size,
            "package_sha256": file_sha256(package),
            "launch_path": str(launch),
            "launch_sha256": file_sha256(launch),
            "verified_checksum_entries": verified,
            "reviewed_handoff_size_bytes": len(nested),
            "reviewed_handoff_sha256": hashlib.sha256(nested).hexdigest(),
            "source_snapshot_size_bytes": len(source),
            "source_snapshot_sha256": hashlib.sha256(source).hexdigest(),
        }

    def _prepare_or_resume(self) -> tuple[dict[str, Any], bool]:
        resumed = self.output_root.exists()
        if resumed:
            if not self.state_path.exists() or not self.status_path.exists():
                raise ProductionError("Existing production root lacks restart state")
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("run_id") != self.config["run_id"]:
                raise ProductionError("Existing production root belongs to another run")
            self.event("campaign_process_resumed", prior_state=state.get("state"))
            return state, True
        if self.temp_root.exists():
            raise FileExistsError("Production temp root must be fresh when output root is fresh")
        self.output_root.mkdir(parents=True)
        self.temp_root.mkdir(parents=True)
        for name in (
            "authority", "config", "status", "logs", "validation", "inputs", "common_initialization",
            "attempts", "final_selected_reference", "tables", "figures", "reports", "manifests",
        ):
            (self.output_root / name).mkdir()
        shutil.copy2(self.config_path, self.output_root / "config" / self.config_path.name)
        authority = self._validate_authority()
        _write_json_new(self.output_root / "validation" / "authority_validation.json", authority)
        package = Path(self.config["authority"]["package_path"])
        with zipfile.ZipFile(package) as archive:
            for info in archive.infolist():
                member = PurePosixPath(info.filename)
                if info.is_dir() or info.file_size > 1024 * 1024:
                    continue
                if member.is_absolute() or ".." in member.parts:
                    raise ProductionError(f"Unsafe authority member: {info.filename}")
                if member.suffix.lower() not in {".md", ".json", ".csv", ".sha256"}:
                    continue
                destination = self.output_root / "authority" / Path(*member.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(archive.read(info.filename))
        branch = _git(self.project_root, "branch", "--show-current").stdout.strip()
        head = _git(self.project_root, "rev-parse", "HEAD").stdout.strip()
        if branch != self.config["expected_branch"]:
            raise ProductionError("Live branch differs from the frozen branch")
        ancestor = _git(
            self.project_root,
            "merge-base", "--is-ancestor", self.config["accepted_refinement_checkpoint_commit"], head,
            check=False,
        )
        if ancestor.returncode != 0:
            raise ProductionError("Accepted refinement checkpoint is not an ancestor of live HEAD")
        git_snapshot = {
            "branch": branch,
            "head": head,
            "accepted_refinement_checkpoint_commit": self.config["accepted_refinement_checkpoint_commit"],
            "status_short": _git(self.project_root, "status", "--short").stdout,
            "diff_stat": _git(self.project_root, "diff", "--stat").stdout,
            "diff": _git(self.project_root, "diff", "--").stdout,
            "dirty_state_action": "preserved; no reset, clean, stash, delete, or overwrite",
        }
        _write_json_new(self.output_root / "validation" / "git_start_snapshot.json", git_snapshot)
        started_epoch = time.time()
        state = {
            "schema_id": "phase4.physical_reference_production.orchestration.v1",
            "run_id": self.config["run_id"],
            "state": "STARTING",
            "campaign_started_utc": _utc_now(),
            "campaign_started_epoch": started_epoch,
            "attempts": [],
            "active_attempt_index": None,
            "next_factor_index": 0,
            "selected_attempt_index": None,
            "peak_rss_bytes": _rss_bytes(),
            "initial_output_bytes": _directory_size(self.output_root),
        }
        _write_json_atomic(self.state_path, state)
        self.status("EXECUTING_STARTUP", campaign_started_utc=state["campaign_started_utc"])
        self.event("campaign_started", authority_status=authority["status"], git_head=head)
        return state, False

    def _resource_guard(self, state: dict[str, Any], next_task: str) -> dict[str, Any]:
        resources = self.config["resources"]
        elapsed = time.time() - float(state["campaign_started_epoch"])
        rss = _rss_bytes()
        state["peak_rss_bytes"] = max(int(state.get("peak_rss_bytes", 0)), rss)
        output_bytes = _directory_size(self.output_root)
        temp_bytes = _directory_size(self.temp_root)
        added = output_bytes + temp_bytes - int(state.get("initial_output_bytes", 0))
        receipt = {
            "timestamp_utc": _utc_now(),
            "next_task": next_task,
            "campaign_elapsed_seconds": elapsed,
            "seconds_before_protected_reserve": resources["wall_clock_seconds"] - resources["protected_finalization_seconds"] - elapsed,
            "rss_bytes": rss,
            "peak_rss_bytes": state["peak_rss_bytes"],
            "rss_hard_bytes": resources["process_tree_rss_bytes"],
            "added_disk_bytes": added,
            "added_disk_hard_bytes": resources["added_disk_bytes"],
        }
        _append_jsonl(self.output_root / "validation" / "resource_guards.jsonl", receipt)
        if rss >= resources["process_tree_rss_bytes"]:
            raise MemoryError("Process-tree RSS reached the 24-GiB hard limit")
        if added >= resources["added_disk_bytes"]:
            raise OSError("Added disk reached the 150-GiB hard limit")
        return receipt

    def _load_inputs(self, *, first_process: bool) -> tuple[Any, Any, list[Any], Any, dict[str, Any]]:
        graph = load_phase4_graph(self.config)
        graph, audit, _register = sanitize_prepared_capacity(graph)
        if audit["clean_c_order_float64_payload_sha256"] != self.config["input_versions"]["clean_capacity_payload_sha256"]:
            raise ProductionError("Clean prepared-capacity payload differs from authority")
        od = load_or_build_aggregated_od(
            self.config,
            output_root=self.output_root,
            existing_root=Path(self.config["input_versions"]["accepted_aggregated_od_root"]),
        )
        trace_maps = load_suite_trace_maps(self.config, graph, verify_directory_digests=True)
        if tuple(item.suite_id for item in trace_maps) != SUITE_IDS:
            raise ProductionError("Loaded trace suite order differs from s02-s06")
        tie_rank = np.ascontiguousarray(partC_rev1_arc_tie_rank(graph), dtype=np.int64)
        receipt = {
            "status": "PASS",
            "graph_identity": graph.graph_hash,
            "graph_node_count": len(graph.node_idx_to_uid),
            "graph_arc_count": len(graph.arc_idx_to_directed_arc_id),
            "od_identity": od.od_hash,
            "aggregated_od_rows": len(od.demand),
            "original_od_rows": od.original_row_count,
            "origin_count": len(od.origin_group_node_idx),
            "total_demand": float(od.demand.sum(dtype=np.float64)),
            "clean_capacity_payload_sha256": _array_payload_sha256(graph.capacity),
            "suite_ids": [item.suite_id for item in trace_maps],
            "suite_directory_digests": {item.suite_id: item.directory_digest for item in trace_maps},
            "s01_absent": True,
            "first_process": first_process,
        }
        if first_process:
            _write_json_new(self.output_root / "validation" / "input_integrity.json", receipt)
            clean_path = self.output_root / "inputs" / "clean_prepared_capacity.npy"
            with clean_path.open("xb") as stream:
                np.save(stream, np.ascontiguousarray(graph.capacity))
            _write_json_new(
                self.output_root / "inputs" / "clean_prepared_capacity_receipt.json",
                {"path": str(clean_path.resolve()), "size_bytes": clean_path.stat().st_size, "sha256": file_sha256(clean_path), **receipt},
            )
        return graph, od, trace_maps, tie_rank, receipt

    def _common_initialization(self, graph: Any, od: Any, trace_maps: Sequence[Any], tie_rank: np.ndarray, *, first_process: bool) -> Any:
        root = self.output_root / "common_initialization" / "checkpoint_iteration_0000"
        if not first_process:
            return load_phase4_checkpoint_aon(root, trace_maps=trace_maps)
        initial = run_exact_aon(
            graph,
            od,
            graph.free_time,
            trace_maps=trace_maps,
            config=AONExecutionConfig(worker_count=4),
            arc_tie_rank=tie_rank,
            packed_trace_maps=pack_trace_maps(trace_maps),
        )
        state = Phase4RestartState(
            next_iteration=1,
            physical_flow=initial.physical_flow,
            suite_traces=initial.suite_traces,
            history=[],
            metadata={"state": "common_exact_free_flow_aon", "suite_ids": list(SUITE_IDS)},
        )
        receipt = save_phase4_checkpoint(
            root,
            state,
            trace_maps=trace_maps,
            latest_aon=initial,
            run_metadata={
                "run_id": self.config["run_id"],
                "graph_hash": graph.graph_hash,
                "od_hash": od.od_hash,
                "clean_capacity_payload_sha256": _array_payload_sha256(graph.capacity),
                "suite_ids": list(SUITE_IDS),
            },
        )
        _write_json_new(
            self.output_root / "common_initialization" / "initialization_receipt.json",
            {
                **receipt,
                "physical_flow_payload_sha256": _array_payload_sha256(initial.physical_flow),
                "sptt": initial.sptt,
                "reachable_demand": initial.reachable_demand,
                "unreachable_demand": initial.unreachable_demand,
                "mapping_failure_counts": initial.mapping_failure_counts,
            },
        )
        return initial

    def _start_attempt(self, state: dict[str, Any], factor_index: int) -> dict[str, Any]:
        factor = FACTORS[factor_index]
        attempt_index = factor_index + 1
        root = self.output_root / "attempts" / f"attempt_{attempt_index:02d}_mu_{_factor_key(factor)}"
        root.mkdir()
        for name in ("checkpoints", "chunks", "logs", "validation", "diagnostics", "archive"):
            (root / name).mkdir()
        attempt = {
            "attempt_index": attempt_index,
            "capacity_factor": factor,
            "root": str(root.resolve()),
            "status": "EXECUTING",
            "started_utc": _utc_now(),
            "started_epoch": time.time(),
            "current_iteration": 0,
            "chunk_index": 0,
            "last_checkpoint": None,
            "state_gaps": {},
            "iteration_seconds": [],
            "primary_gate_evaluated": False,
            "fallback_gate_evaluated": False,
            "best_rg_sptt": None,
        }
        state["attempts"].append(attempt)
        state["active_attempt_index"] = len(state["attempts"]) - 1
        state["next_factor_index"] = factor_index + 1
        state["state"] = "EXECUTING_ATTEMPT"
        _write_json_atomic(self.state_path, state)
        _write_json_new(root / "attempt_metadata.json", attempt)
        self.event("attempt_started", attempt_index=attempt_index, capacity_factor=factor)
        return attempt

    def _refresh_subprocess(self, checkpoint: Path, factor: float, output: Path, label: str) -> dict[str, Any]:
        command = [
            self.config["python_executable"], "-m", "partC_rev1_src.phase4.production",
            "--config", str(self.config_path), "--phase-id", self.config["phase_id"],
            "--step-id", self.config["step_id"], "--run-id", self.config["run_id"],
            "--stage", "refresh", "--checkpoint", str(checkpoint), "--capacity-factor", str(factor),
            "--refresh-output", str(output), "--refresh-label", label,
        ]
        result = subprocess.run(command, cwd=self.project_root, capture_output=True, text=True)
        _append_jsonl(
            self.output_root / "logs" / "fresh_process_commands.jsonl",
            {"timestamp_utc": _utc_now(), "command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
        )
        if result.returncode != 0:
            raise ProductionError(f"Fresh-process refresh failed: {result.stderr or result.stdout}")
        return json.loads(output.read_text(encoding="utf-8"))

    def _run_attempt(self, state: dict[str, Any], attempt: dict[str, Any], graph: Any, od: Any, trace_maps: Sequence[Any], common_initial: Any) -> tuple[str, dict[str, Any]]:
        resources = self.config["resources"]
        factor = float(attempt["capacity_factor"])
        root = Path(attempt["root"])
        target = float(self.config["gap_target"])
        chunk_updates = int(self.config["chunk_updates"])
        max_updates = int(self.config["max_updates_per_attempt"])
        restart_state = None
        if attempt.get("last_checkpoint"):
            restart_state = load_phase4_checkpoint(attempt["last_checkpoint"], trace_maps=trace_maps)
        while int(attempt["current_iteration"]) < max_updates:
            receipt = self._resource_guard(state, "NEXT-PRODUCTION-CHUNK")
            if receipt["seconds_before_protected_reserve"] <= float(self.config["minimum_seconds_to_start_update"]):
                attempt["status"] = "STOPPED_PROTECTED_RESERVE"
                return "reserve", attempt
            first = int(attempt["current_iteration"]) + 1
            last = min(max_updates, int(attempt["current_iteration"]) + chunk_updates)
            attempt["chunk_index"] = int(attempt["chunk_index"]) + 1
            chunk_key = f"chunk_{attempt['chunk_index']:04d}_to_{last:05d}"
            checkpoint_root = root / "checkpoints" / chunk_key
            iteration_log = root / "logs" / f"{chunk_key}_iterations.jsonl"
            # Persist the write-new chunk identity before entering heavy work.
            # An interrupted retry therefore advances to a new chunk directory
            # while preserving any partial files from the interrupted process.
            _write_json_atomic(self.state_path, state)
            solve_budget = max(1.0, receipt["seconds_before_protected_reserve"] - 60.0)
            result = solve_assignment(
                graph,
                od,
                trace_maps=trace_maps,
                config=Phase4SolverConfig(
                    method=SOLVER_METHOD,
                    capacity_multiplier=factor,
                    max_iterations=last,
                    minimum_iterations=first,
                    gap_target=0.0,
                    worker_count=4,
                    max_runtime_seconds=solve_budget,
                    rss_hard_bytes=resources["process_tree_rss_bytes"],
                    checkpoint_directory=checkpoint_root,
                    checkpoint_iterations=_checkpoint_iterations(first, last),
                    final_checkpoint=True,
                    iteration_record_path=iteration_log,
                ),
                initial_aon=common_initial if restart_state is None else None,
                restart_state=restart_state,
                run_metadata={
                    "authorization_id": self.config["authorization_id"],
                    "run_id": self.config["run_id"],
                    "attempt_index": attempt["attempt_index"],
                    "capacity_factor": factor,
                    "graph_hash": graph.graph_hash,
                    "od_hash": od.od_hash,
                    "clean_capacity_payload_sha256": _array_payload_sha256(graph.capacity),
                    "suite_ids": list(SUITE_IDS),
                },
            )
            if not result.histories:
                raise ProductionError("Production chunk completed zero updates")
            saved = _save_result(root / "chunks" / chunk_key, result, run_scope=chunk_key)
            final_checkpoint = Path(result.checkpoint_receipts[-1]["directory"])
            completed = int(result.histories[-1].iteration)
            gaps = _state_gap_map(attempt)
            for record in result.histories:
                index = int(record.evaluated_state_iteration)
                value = float(record.relative_gap_sptt)
                if index in gaps and not np.isclose(gaps[index], value, rtol=0.0, atol=1e-15):
                    raise ProductionError("Restarted refreshed-gap history is inconsistent")
                gaps[index] = value
                attempt["iteration_seconds"].append(float(record.iteration_wall_seconds))
            gaps[completed] = float(result.final_metrics.relative_gap_sptt)
            attempt["state_gaps"] = {str(key): value for key, value in sorted(gaps.items())}
            attempt["current_iteration"] = completed
            attempt["last_checkpoint"] = str(final_checkpoint.resolve())
            attempt["best_rg_sptt"] = min(gaps.values())
            attempt["final_rg_sptt"] = float(result.final_metrics.relative_gap_sptt)
            attempt["last_chunk_runtime_seconds"] = float(result.runtime_seconds)
            attempt["peak_rss_bytes"] = max(int(attempt.get("peak_rss_bytes", 0)), int(result.peak_rss_bytes))
            attempt["mapping_failure_counts"] = dict(result.latest_aon.mapping_failure_counts)
            chunk_receipt = {
                "timestamp_utc": _utc_now(),
                "chunk_key": chunk_key,
                "first_iteration": first,
                "completed_iteration": completed,
                "final_checkpoint": str(final_checkpoint.resolve()),
                "final_metrics": asdict(result.final_metrics),
                "stop_reason": result.stop_reason,
                "runtime_seconds": result.runtime_seconds,
                "peak_rss_bytes": result.peak_rss_bytes,
                "solver_summary": saved["summary"],
            }
            _append_jsonl(root / "logs" / "chunk_registry.jsonl", chunk_receipt)
            _append_jsonl(
                root / "logs" / "official_refreshes.jsonl",
                {"timestamp_utc": _utc_now(), "state_iteration": completed, "relative_gap_sptt": result.final_metrics.relative_gap_sptt, "checkpoint": str(final_checkpoint.resolve())},
            )
            _write_json_atomic(self.state_path, state)
            self.status(
                "EXECUTING_PRODUCTION_ATTEMPT",
                active_capacity_factor=factor,
                active_attempt_index=attempt["attempt_index"],
                completed_iteration=completed,
                best_refreshed_rg_sptt=attempt["best_rg_sptt"],
                final_refreshed_rg_sptt=attempt["final_rg_sptt"],
                last_checkpoint=str(final_checkpoint.resolve()),
            )
            self.event(
                "chunk_completed",
                attempt_index=attempt["attempt_index"],
                capacity_factor=factor,
                completed_iteration=completed,
                final_rg_sptt=attempt["final_rg_sptt"],
                best_rg_sptt=attempt["best_rg_sptt"],
            )
            previous_gap = gaps.get(completed - 1)
            if (
                previous_gap is not None
                and previous_gap <= target
                and result.final_metrics.relative_gap_sptt <= target
                and all(value == 0 for value in result.latest_aon.mapping_failure_counts.values())
            ):
                refresh_path = root / "validation" / f"independent_final_refresh_iteration_{completed:05d}.json"
                refresh = self._refresh_subprocess(final_checkpoint, factor, refresh_path, "FINAL_CANDIDATE")
                if refresh["relative_gap_sptt"] <= target and refresh["status"] == "PASS":
                    attempt["status"] = "SELECTED_NUMERICAL_SUCCESS"
                    attempt["independent_refresh"] = str(refresh_path.resolve())
                    attempt["independent_refreshed_rg_sptt"] = refresh["relative_gap_sptt"]
                    attempt["completed_utc"] = _utc_now()
                    _write_json_atomic(self.state_path, state)
                    return "success", attempt
                raise ProductionError("Fresh-process final refresh did not preserve the target")
            campaign_elapsed = time.time() - float(state["campaign_started_epoch"])
            attempt_elapsed = time.time() - float(attempt["started_epoch"])
            assessment = _progress_assessment(gaps, attempt["iteration_seconds"], target=target)
            if factor == 0.30 and campaign_elapsed >= 43200.0 and not attempt["primary_gate_evaluated"]:
                attempt["primary_gate_evaluated"] = True
                attempt["primary_gate_assessment"] = assessment
                _write_json_new(root / "diagnostics" / "primary_12h_assessment.json", assessment)
                if not assessment["promising"]:
                    return "switch", attempt
            if factor == 0.45 and completed >= 25 and (attempt_elapsed >= 7200.0 or completed >= 100) and not attempt["fallback_gate_evaluated"]:
                attempt["fallback_gate_evaluated"] = True
                remaining = resources["wall_clock_seconds"] - resources["protected_finalization_seconds"] - campaign_elapsed
                promising = bool(assessment.get("best_rg_sptt", float("inf")) <= 5e-4 and assessment.get("conservative_additional_seconds_to_target", float("inf")) <= remaining)
                assessment["fallback_promising"] = promising
                assessment["remaining_solve_seconds"] = remaining
                _write_json_new(root / "diagnostics" / "fallback_assessment.json", assessment)
                attempt["fallback_gate_assessment"] = assessment
                if not promising and remaining >= 14400.0:
                    return "switch", attempt
            restart_state = load_phase4_checkpoint(final_checkpoint, trace_maps=trace_maps)
        attempt["status"] = "STOPPED_MAX_AUTHORIZED_UPDATES"
        return "max_updates", attempt

    def _seal_nonselected(self, state: dict[str, Any], attempt: dict[str, Any], reason: str) -> None:
        root = Path(attempt["root"])
        checkpoint = Path(attempt["last_checkpoint"])
        refresh_path = root / "validation" / "independent_switch_refresh.json"
        refresh = self._refresh_subprocess(checkpoint, float(attempt["capacity_factor"]), refresh_path, "FACTOR_SWITCH")
        attempt["status"] = "SEALED_NONSELECTED_PRODUCTION_ATTEMPT"
        attempt["switch_reason"] = reason
        attempt["switch_timestamp_utc"] = _utc_now()
        attempt["independent_switch_refresh"] = str(refresh_path.resolve())
        attempt["independent_switch_rg_sptt"] = refresh["relative_gap_sptt"]
        _write_json_new(root / "archive" / "ATTEMPT_SEAL.json", attempt)
        _write_json_atomic(self.state_path, state)
        self.event("attempt_sealed_for_switch", attempt_index=attempt["attempt_index"], capacity_factor=attempt["capacity_factor"], reason=reason)

    def execute(self) -> None:
        state, resumed = self._prepare_or_resume()
        try:
            authority = self._validate_authority()
            if resumed:
                self.status("RESUMING_FROM_LATEST_VALID_CHECKPOINT", authority_status=authority["status"])
            graph, od, trace_maps, tie_rank, input_receipt = self._load_inputs(first_process=not resumed)
            common_initial = self._common_initialization(graph, od, trace_maps, tie_rank, first_process=not resumed)
            self._resource_guard(state, "BEGIN-OR-RESUME-ATTEMPT")
            while state.get("selected_attempt_index") is None and int(state.get("next_factor_index", 0)) <= len(FACTORS):
                active_index = state.get("active_attempt_index")
                if active_index is None or state["attempts"][active_index]["status"] != "EXECUTING":
                    factor_index = int(state["next_factor_index"])
                    if factor_index >= len(FACTORS):
                        break
                    attempt = self._start_attempt(state, factor_index)
                else:
                    attempt = state["attempts"][active_index]
                outcome, attempt = self._run_attempt(state, attempt, graph, od, trace_maps, common_initial)
                if outcome == "success":
                    selected_index = int(state["active_attempt_index"])
                    state["selected_attempt_index"] = selected_index
                    state["state"] = "NUMERICAL_TARGET_AND_INDEPENDENT_REFRESH_PASS"
                    binding = {
                        "selected_attempt_index": attempt["attempt_index"],
                        "capacity_factor": attempt["capacity_factor"],
                        "checkpoint": attempt["last_checkpoint"],
                        "two_consecutive_target_states": [attempt["current_iteration"] - 1, attempt["current_iteration"]],
                        "final_refreshed_rg_sptt": attempt["final_rg_sptt"],
                        "independent_refresh": attempt["independent_refresh"],
                        "independent_refreshed_rg_sptt": attempt["independent_refreshed_rg_sptt"],
                        "graph_hash": graph.graph_hash,
                        "od_hash": od.od_hash,
                        "clean_capacity_payload_sha256": input_receipt["clean_capacity_payload_sha256"],
                        "suite_ids": list(SUITE_IDS),
                    }
                    _write_json_new(self.output_root / "final_selected_reference" / "selected_checkpoint_binding.json", binding)
                    _write_json_atomic(self.state_path, state)
                    self.status(
                        "NUMERICAL_TARGET_AND_INDEPENDENT_REFRESH_PASS_READY_FOR_FINALIZATION",
                        phase4_overall="VALIDATING",
                        final_physical_reference_produced=False,
                        final_capacity_factor=attempt["capacity_factor"],
                        final_refreshed_rg_sptt=attempt["final_rg_sptt"],
                        independent_refreshed_rg_sptt=attempt["independent_refreshed_rg_sptt"],
                        selected_checkpoint=attempt["last_checkpoint"],
                    )
                    self.event("numerical_target_candidate_completed", capacity_factor=attempt["capacity_factor"], iteration=attempt["current_iteration"])
                    return
                if outcome == "switch":
                    if state["next_factor_index"] >= len(FACTORS):
                        break
                    reason = "frozen_promising_rule_not_met"
                    self._seal_nonselected(state, attempt, reason)
                    state["active_attempt_index"] = None
                    _write_json_atomic(self.state_path, state)
                    continue
                break
            state["state"] = "PHASE4_TARGET_NOT_REACHED_WITHIN_AUTHORIZED_24H"
            state["active_attempt_index"] = None
            _write_json_atomic(self.state_path, state)
            best = min(
                (item for item in state["attempts"] if item.get("best_rg_sptt") is not None),
                key=lambda item: item["best_rg_sptt"],
                default=None,
            )
            self.status(
                "PHASE4_TARGET_NOT_REACHED_WITHIN_AUTHORIZED_24H",
                phase4_overall="INCOMPLETE_TARGET_NOT_REACHED",
                final_physical_reference_produced=False,
                best_capacity_factor=best.get("capacity_factor") if best else None,
                best_refreshed_rg_sptt=best.get("best_rg_sptt") if best else None,
                phase5_authorized=False,
            )
            self.event("campaign_target_not_reached", best_attempt=best)
        except BaseException as exc:
            if self.output_root.exists():
                self.status(
                    "EXECUTION_FAILED_REQUIRES_REPAIR_HANDOFF",
                    phase4_overall="BLOCKED",
                    failure_type=type(exc).__name__,
                    failure_message=str(exc),
                    phase5_authorized=False,
                )
                self.event("campaign_failed", failure_type=type(exc).__name__, failure_message=str(exc))
            raise


def independent_refresh(config_path: Path, checkpoint: Path, factor: float, output: Path, label: str) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    graph = load_phase4_graph(config)
    graph, audit, _register = sanitize_prepared_capacity(graph)
    if audit["clean_c_order_float64_payload_sha256"] != config["input_versions"]["clean_capacity_payload_sha256"]:
        raise ProductionError("Fresh refresh clean-capacity identity mismatch")
    od = load_or_build_aggregated_od(
        config,
        output_root=Path(config["output_root"]),
        existing_root=Path(config["input_versions"]["accepted_aggregated_od_root"]),
    )
    trace_maps = load_suite_trace_maps(config, graph, verify_directory_digests=True)
    state = load_phase4_checkpoint(checkpoint, trace_maps=trace_maps)
    stored_aon = load_phase4_checkpoint_aon(checkpoint, trace_maps=trace_maps)
    used_capacity = phase4_scenario_capacity(graph, factor)
    cost = phase4_bpr_cost(graph, state.physical_flow, used_capacity)
    refreshed = run_exact_aon(
        graph,
        od,
        cost,
        trace_maps=trace_maps,
        config=AONExecutionConfig(worker_count=4),
        arc_tie_rank=np.ascontiguousarray(partC_rev1_arc_tie_rank(graph), dtype=np.int64),
        packed_trace_maps=pack_trace_maps(trace_maps),
    )
    metrics = refreshed_state_metrics(graph, state.physical_flow, used_capacity, sptt=refreshed.sptt, epsilon=1e-12)
    trace_comparisons = {item.suite_id: _trace_equal(stored_aon.suite_traces[item.suite_id], refreshed.suite_traces[item.suite_id]) for item in trace_maps}
    physical_aon_equal = bool(np.array_equal(stored_aon.physical_flow, refreshed.physical_flow))
    traces_equal = bool(all(value["pass"] for value in trace_comparisons.values()))
    mapping_zero = bool(all(value == 0 for value in refreshed.mapping_failure_counts.values()))
    demand_total = float(refreshed.reachable_demand + refreshed.unreachable_demand)
    demand_equal = bool(np.isclose(demand_total, float(od.demand.sum(dtype=np.float64)), rtol=1e-12, atol=1e-8))
    report = {
        "schema_id": "phase4.physical_reference_production.independent_refresh.v1",
        "status": "PASS" if physical_aon_equal and traces_equal and mapping_zero and demand_equal else "FAIL",
        "label": label,
        "refreshed_utc": _utc_now(),
        "checkpoint": str(checkpoint.resolve()),
        "capacity_factor": factor,
        "graph_hash": graph.graph_hash,
        "od_hash": od.od_hash,
        "clean_capacity_payload_sha256": _array_payload_sha256(graph.capacity),
        "suite_ids": [item.suite_id for item in trace_maps],
        "relative_gap_sptt": metrics.relative_gap_sptt,
        "relative_gap_tstt": metrics.relative_gap_tstt,
        "objective": metrics.objective,
        "tstt": metrics.tstt,
        "sptt": metrics.sptt,
        "absolute_gap": metrics.absolute_gap,
        "reachable_demand": refreshed.reachable_demand,
        "unreachable_demand": refreshed.unreachable_demand,
        "demand_equal": demand_equal,
        "mapping_failure_counts": refreshed.mapping_failure_counts,
        "mapping_failures_zero": mapping_zero,
        "physical_aon_array_equal_to_checkpoint": physical_aon_equal,
        "trace_comparisons": trace_comparisons,
        "all_trace_arrays_equal_to_checkpoint": traces_equal,
        "current_physical_flow_payload_sha256": _array_payload_sha256(state.physical_flow),
        "refreshed_aon_flow_payload_sha256": _array_payload_sha256(refreshed.physical_flow),
        "used_capacity_payload_sha256": _array_payload_sha256(used_capacity),
        "cost_payload_sha256": _array_payload_sha256(cost),
    }
    _write_json_new(output, report)
    if report["status"] != "PASS":
        raise ProductionError("Independent refresh did not reproduce the checkpoint AON/trace state")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("execute", "refresh"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--capacity-factor", type=float)
    parser.add_argument("--refresh-output")
    parser.add_argument("--refresh-label")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = Path(args.config).resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, supplied in (("phase_id", args.phase_id), ("step_id", args.step_id), ("run_id", args.run_id)):
        if config.get(key) != supplied:
            raise ProductionError(f"CLI {key} differs from explicit execution config")
    if args.stage == "execute":
        ProductionCampaign(config_path).execute()
    else:
        if args.checkpoint is None or args.capacity_factor is None or args.refresh_output is None or args.refresh_label is None:
            raise ProductionError("Refresh stage requires checkpoint, factor, output, and label")
        independent_refresh(
            config_path,
            Path(args.checkpoint).resolve(strict=True),
            float(args.capacity_factor),
            Path(args.refresh_output),
            str(args.refresh_label),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
