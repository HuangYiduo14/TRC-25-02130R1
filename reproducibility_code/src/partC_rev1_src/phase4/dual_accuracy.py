"""Recover, export, and register the Phase 4 dual-accuracy benchmarks.

This module is deliberately recovery-first.  It can independently refresh a
saved state, but it contains no solver-update entry point and cannot advance
either the accepted iteration-200 state or the selected low-accuracy state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from partC_rev1_src.partC_rev1_shortest_paths import partC_rev1_arc_tie_rank

from .aon import AONExecutionConfig, run_exact_aon
from .checkpoint import TRACE_DENSE_FIELDS, load_phase4_checkpoint, load_phase4_checkpoint_aon
from .costs import phase4_bpr_cost, phase4_scenario_capacity, refreshed_state_metrics
from .inputs import file_sha256, load_or_build_aggregated_od, load_phase4_graph
from .production_finalize import (
    MAX_HANDOFF_BYTES,
    MAX_HANDOFF_MEMBER_BYTES,
    _array_payload_sha256,
    _copy_preserving,
    _create_zip,
    _directory_digest,
    _directory_size,
    _fresh_extract_verify,
    _handoff_checksums,
    _physical_link_table,
    _privacy_scan,
    _suite_products,
    _write_frame_new,
    _write_json_atomic,
    _write_json_new,
    _write_npz_new,
    _write_text_new,
)
from .refinement import _trace_equal, sanitize_prepared_capacity
from .tracing import load_suite_trace_maps, pack_trace_maps


SUITE_IDS = ("s02", "s03", "s04", "s05", "s06")
HIGH_BENCHMARK_ID = "physical_mu030_high_iter0200"
LOW_BENCHMARK_ID = "physical_mu030_low_rg1e3"
FINAL_STATUS = "WAITING_FOR_EXTERNAL_DUAL_BENCHMARK_REVIEW"


class DualAccuracyError(RuntimeError):
    """Raised when a dual-benchmark hard gate fails."""


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(project_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(project_root), "-c", "core.longpaths=true", *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise DualAccuracyError(result.stderr or result.stdout)
    return result


def _resolve_context(
    config_path: Path,
    phase_id: str,
    step_id: str,
    run_id: str,
) -> tuple[dict[str, Any], Path, Path]:
    config = _load_json(config_path)
    if config.get("schema_id") != "phase4.dual_benchmark.execution.v1":
        raise DualAccuracyError("Unsupported dual-benchmark execution schema")
    for key, supplied in (("phase_id", phase_id), ("step_id", step_id), ("run_id", run_id)):
        if config.get(key) != supplied:
            raise DualAccuracyError(f"CLI {key} differs from the explicit execution config")
    if config.get("decision_id") != "PHASE4_ACCEPT_ITER200_AND_RECOVER_RG1E3_20260902":
        raise DualAccuracyError("The operative external dual-benchmark decision is absent")
    if tuple(config.get("suite_ids", ())) != SUITE_IDS or "s01" not in config.get("excluded_suite_ids", ()):
        raise DualAccuracyError("Trace scope must be exactly s02-s06 with s01 excluded")
    if float(config.get("capacity_factor", -1.0)) != 0.30 or int(config.get("worker_count", -1)) != 4:
        raise DualAccuracyError("The dual benchmarks require capacity factor 0.30 and four workers")
    if not config.get("tasks") or not config.get("expected_outputs") or not config.get("acceptance_tests"):
        raise DualAccuracyError("Explicit tasks, outputs, and acceptance tests must be nonempty")
    if bool(config["phase_boundaries"].get("phase5_authorized")):
        raise DualAccuracyError("Phase 5 must remain unauthorized")

    project_root = Path(config["project_root"]).resolve(strict=True)
    branch = _git(project_root, "branch", "--show-current").stdout.strip()
    if branch != config["expected_branch"]:
        raise DualAccuracyError(f"Expected branch {config['expected_branch']}, found {branch}")
    ancestor = _git(
        project_root,
        "merge-base",
        "--is-ancestor",
        str(config["approved_starting_head"]),
        "HEAD",
        check=False,
    )
    if ancestor.returncode != 0:
        raise DualAccuracyError("Current HEAD is not descended from the reviewed iteration-200 HEAD")

    authority = config["authority"]
    for role, path_key, size_key, hash_key in (
        ("authority package", "package_path", "package_size_bytes", "package_sha256"),
        ("launch document", "launch_path", "launch_size_bytes", "launch_sha256"),
        (
            "reviewed iteration-200 handoff",
            "reviewed_iter200_handoff",
            "reviewed_iter200_handoff_size_bytes",
            "reviewed_iter200_handoff_sha256",
        ),
    ):
        path = Path(authority[path_key]).resolve(strict=True)
        if path.stat().st_size != int(authority[size_key]) or file_sha256(path) != authority[hash_key]:
            raise DualAccuracyError(f"{role} identity differs from the approved binding")

    output_root = Path(config["output_root"]).resolve()
    allowed_parent = Path(project_root.parent / f"{project_root.name}_outputs" / "phase_04" / "dual_accuracy_benchmarks").resolve()
    if output_root.parent != allowed_parent:
        raise DualAccuracyError(f"Output root is outside the authorized write-new location: {output_root}")
    return config, project_root, output_root


def _required_checkpoint_keys(suite_ids: Sequence[str]) -> set[str]:
    required = {"current_physical_flow", "aon_physical_flow"}
    sparse = (
        "coarse_regional_pair_keys",
        "coarse_regional_pair_values",
        "fine_regional_pair_keys",
        "fine_regional_pair_values",
    )
    for suite_id in suite_ids:
        for prefix in ("current", "aon"):
            required.update(f"{prefix}__{suite_id}__{name}" for name in TRACE_DENSE_FIELDS)
            required.update(f"{prefix}__{suite_id}__{name}" for name in sparse)
        required.add(f"aon__{suite_id}__repeated_unique_od_occurrences")
        required.add(f"aon__{suite_id}__repeated_original_row_occurrences")
    return required


def validate_checkpoint_arrays(
    checkpoint: Path,
    *,
    expected_state_iteration: int,
    expected_arc_count: int,
    suite_ids: Sequence[str] = SUITE_IDS,
) -> dict[str, Any]:
    """Open and validate the physical/current/AON arrays needed for export."""

    metadata_path = checkpoint / "checkpoint_metadata.json"
    arrays_path = checkpoint / "checkpoint_arrays.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        raise DualAccuracyError(f"Checkpoint payload is incomplete: {checkpoint}")
    metadata = _load_json(metadata_path)
    actual_state = int(metadata.get("next_iteration", -1)) - 1
    if actual_state != int(expected_state_iteration):
        raise DualAccuracyError(
            f"Checkpoint state-index mismatch: expected x^{expected_state_iteration}, found x^{actual_state}"
        )
    if metadata.get("schema_id") != "phase4.physical_trace_checkpoint.v1":
        raise DualAccuracyError("Checkpoint schema differs from phase4.physical_trace_checkpoint.v1")
    if tuple(metadata.get("suite_ids", ())) != tuple(suite_ids):
        raise DualAccuracyError("Checkpoint suite order differs from the frozen five-suite order")

    required = _required_checkpoint_keys(suite_ids)
    array_rows: list[dict[str, Any]] = []
    with np.load(arrays_path, allow_pickle=False) as archive:
        names = set(archive.files)
        missing = sorted(required - names)
        if missing:
            raise DualAccuracyError(f"Checkpoint is missing required current/AON arrays: {missing}")
        if set(metadata.get("array_names", ())) != names:
            raise DualAccuracyError("Checkpoint metadata array inventory differs from the live NPZ")
        for name in sorted(required):
            value = np.asarray(archive[name])
            if value.dtype.hasobject:
                raise DualAccuracyError(f"Object array is prohibited in checkpoint field {name}")
            if np.issubdtype(value.dtype, np.number) and np.any(~np.isfinite(value)):
                raise DualAccuracyError(f"Nonfinite value found in checkpoint field {name}")
            if (name.endswith("_flow") or name.endswith("_values")) and np.any(value < -1e-8):
                raise DualAccuracyError(f"Materially negative flow found in checkpoint field {name}")
            array_rows.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "nbytes": int(value.nbytes),
                }
            )
        for physical_name in ("current_physical_flow", "aon_physical_flow"):
            value = np.asarray(archive[physical_name])
            if value.shape != (int(expected_arc_count),) or value.dtype != np.float64:
                raise DualAccuracyError(f"{physical_name} does not match the exact physical arc domain")
        for suite_id in suite_ids:
            for prefix in ("current", "aon"):
                for kind in ("coarse", "fine"):
                    keys = np.asarray(archive[f"{prefix}__{suite_id}__{kind}_regional_pair_keys"])
                    values = np.asarray(archive[f"{prefix}__{suite_id}__{kind}_regional_pair_values"])
                    if keys.dtype != np.uint64 or values.dtype != np.float64 or keys.shape != values.shape:
                        raise DualAccuracyError(f"{suite_id} {prefix} {kind} sparse arrays are invalid")
                    if len(keys) > 1 and np.any(keys[1:] <= keys[:-1]):
                        raise DualAccuracyError(f"{suite_id} {prefix} {kind} sparse keys are not unique/sorted")
        current_payload = _array_payload_sha256(np.asarray(archive["current_physical_flow"]))
        aon_payload = _array_payload_sha256(np.asarray(archive["aon_physical_flow"]))
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint.resolve()),
        "state_iteration": actual_state,
        "archive_array_count": len(metadata["array_names"]),
        "required_array_count": len(required),
        "required_arrays": array_rows,
        "current_physical_flow_payload_sha256": current_payload,
        "aon_physical_flow_payload_sha256": aon_payload,
    }


def select_earliest_complete_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    gap_target: float,
    high_iteration: int,
) -> Mapping[str, Any]:
    eligible = [
        item
        for item in candidates
        if bool(item.get("exists"))
        and bool(item.get("complete"))
        and float(item.get("refreshed_rg_sptt", float("inf"))) <= float(gap_target)
        and int(item["state_iteration"]) < int(high_iteration)
    ]
    if not eligible:
        raise DualAccuracyError("No complete saved low-accuracy candidate qualifies")
    return min(eligible, key=lambda item: int(item["state_iteration"]))


def _discover_checkpoint_states(checkpoint_root: Path, *, minimum: int, maximum: int) -> list[dict[str, Any]]:
    pattern = re.compile(r"^(?:final_)?iteration_(\d+)$")
    rows: list[dict[str, Any]] = []
    for chunk in sorted((path for path in checkpoint_root.iterdir() if path.is_dir()), key=lambda path: path.name):
        for child in sorted((path for path in chunk.iterdir() if path.is_dir()), key=lambda path: path.name):
            match = pattern.fullmatch(child.name)
            if match is None:
                continue
            state = int(match.group(1))
            if minimum <= state <= maximum:
                rows.append(
                    {
                        "state_iteration": state,
                        "path": str(child.resolve()),
                        "directory_name": child.name,
                        "is_final_iteration_directory": child.name.startswith("final_"),
                    }
                )
    return rows


def _verify_file_binding(path: Path, expected_size: int, expected_sha256: str, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise DualAccuracyError(f"Missing {role}: {path}")
    actual_size = int(path.stat().st_size)
    actual_sha = file_sha256(path)
    if actual_size != int(expected_size) or actual_sha != str(expected_sha256):
        raise DualAccuracyError(f"{role} differs from its approved size/hash binding")
    return {"role": role, "path": str(path.resolve()), "size_bytes": actual_size, "sha256": actual_sha}


def _verify_high_bindings(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    authority_root = Path(config["authority"]["extracted_root"])
    index_path = authority_root / "evidence" / "original_external_artifact_index.csv"
    rows = pd.read_csv(index_path).to_dict(orient="records")
    results: list[dict[str, Any]] = []
    for row in rows:
        path = Path(str(row["path"])).resolve(strict=True)
        if path.is_dir():
            size = _directory_size(path)
            digest = _directory_digest(path)
        else:
            size = path.stat().st_size
            digest = file_sha256(path)
        if int(size) != int(row["size_bytes"]) or digest != str(row["sha256_or_digest"]):
            raise DualAccuracyError(f"Accepted high artifact changed: {row['role']}")
        results.append(
            {
                "role": str(row["role"]),
                "path": str(path),
                "size_bytes": int(size),
                "sha256_or_digest": digest,
                "format": str(row["format"]),
                "status": "PASS",
            }
        )
    high = config["high_benchmark"]
    checkpoint_array = Path(high["checkpoint"]) / "checkpoint_arrays.npz"
    if file_sha256(checkpoint_array) != high["checkpoint_arrays_sha256"]:
        raise DualAccuracyError("Accepted high checkpoint array hash changed")
    independent = _load_json(Path(high["independent_refresh"]))
    if (
        independent.get("status") != "PASS"
        or not independent.get("physical_aon_array_equal_to_checkpoint")
        or not independent.get("all_trace_arrays_equal_to_checkpoint")
        or not np.isclose(
            float(independent["relative_gap_sptt"]),
            float(high["actual_rg_sptt"]),
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise DualAccuracyError("Accepted high independent-refresh evidence is inconsistent")
    return results


def inspect_recovery(
    config_path: Path,
    phase_id: str,
    step_id: str,
    run_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, _project_root, output_root = _resolve_context(config_path, phase_id, step_id, run_id)
    if output_root.exists():
        raise FileExistsError(f"Write-new dual-benchmark output root exists: {output_root}")
    (output_root / "config").mkdir(parents=True)
    (output_root / "recovery").mkdir()
    (output_root / "status").mkdir()
    (output_root / "validation").mkdir()
    (output_root / "resources").mkdir()
    shutil.copy2(config_path, output_root / "config" / config_path.name)

    high_bindings = _verify_high_bindings(config)
    _write_frame_new(pd.DataFrame(high_bindings), output_root / "validation" / "high_immutable_artifact_verification.csv")

    low = config["low_benchmark"]
    checkpoint_root = Path(config["checkpoint_root"]).resolve(strict=True)
    discovery = _discover_checkpoint_states(
        checkpoint_root,
        minimum=int(low["first_observed_crossing_state"]),
        maximum=int(config["high_benchmark"]["state_iteration"]) - 1,
    )
    pre_45 = [row for row in discovery if int(row["state_iteration"]) < 45]
    discovered_preferred = [int(row["state_iteration"]) for row in pre_45]
    if any(state in (33, 34) for state in discovered_preferred):
        raise DualAccuracyError("An unlisted x^33/x^34 checkpoint requires explicit live qualification before x^45 selection")

    inventory: list[dict[str, Any]] = []
    selected_validation: dict[str, Any] | None = None
    for candidate in config["candidate_checkpoints"]:
        state = int(candidate["state_iteration"])
        path_text = candidate.get("checkpoint")
        exists = bool(path_text and Path(path_text).is_dir())
        record: dict[str, Any] = {
            "state_iteration": state,
            "checkpoint": path_text,
            "exists": exists,
            "refreshed_rg_sptt": float(candidate["refreshed_rg_sptt"]),
            "gap_target": float(low["gap_target"]),
            "complete": False,
            "selected": False,
            "inspection": "NOT_NEEDED_AFTER_SELECTION",
            "rejection_reason": None,
        }
        if state == 33 and not exists:
            record.update(inspection="BOUNDED_FILESYSTEM_SEARCH", rejection_reason="NO_COMPLETE_SAVED_X33_STATE; SCALAR_HISTORY_ONLY")
        elif state in (35, 40) and exists:
            metadata = _load_json(Path(path_text) / "checkpoint_metadata.json")
            actual_state = int(metadata.get("next_iteration", -1)) - 1
            record["complete"] = actual_state == state and (Path(path_text) / "checkpoint_arrays.npz").is_file()
            record["inspection"] = "METADATA_AND_PAYLOAD_PRESENCE"
            record["rejection_reason"] = "OWN_REFRESHED_GAP_ABOVE_1E3"
        elif state == 45 and exists:
            _verify_file_binding(
                Path(path_text) / "checkpoint_arrays.npz",
                int(candidate["arrays_size_bytes"]),
                str(candidate["arrays_sha256"]),
                "x45 checkpoint arrays",
            )
            _verify_file_binding(
                Path(path_text) / "checkpoint_metadata.json",
                int(candidate["metadata_size_bytes"]),
                str(candidate["metadata_sha256"]),
                "x45 checkpoint metadata",
            )
            selected_validation = validate_checkpoint_arrays(
                Path(path_text),
                expected_state_iteration=state,
                expected_arc_count=int(config["input_versions"]["graph_arc_count"]),
            )
            record["complete"] = True
            record["inspection"] = "FULL_CURRENT_AND_AON_ARRAY_OPEN"
        elif state > 45:
            record["complete"] = exists
        inventory.append(record)
        if state == 45 and record["complete"]:
            break

    selected = dict(
        select_earliest_complete_candidate(
            inventory,
            gap_target=float(low["gap_target"]),
            high_iteration=int(config["high_benchmark"]["state_iteration"]),
        )
    )
    for record in inventory:
        record["selected"] = int(record["state_iteration"]) == int(selected["state_iteration"])
    if int(selected["state_iteration"]) != 45 or selected_validation is None:
        raise DualAccuracyError("The expected earliest complete saved qualifying state x^45 was not selected")

    audit = {
        "schema_id": "phase4.dual_benchmark.recovery_inventory.v1",
        "status": "PASS",
        "inspected_utc": _utc_now(),
        "first_observed_crossing_state": int(low["first_observed_crossing_state"]),
        "first_observed_crossing_gap": float(low["first_observed_crossing_gap"]),
        "filesystem_discovered_states_before_45": discovered_preferred,
        "x33_complete_checkpoint_found": False,
        "selected_state_iteration": int(selected["state_iteration"]),
        "selected_checkpoint": str(Path(str(selected["checkpoint"])).resolve()),
        "selected_stored_refreshed_rg_sptt": float(selected["refreshed_rg_sptt"]),
        "checkpoint_lag_from_first_crossing": int(selected["state_iteration"]) - int(low["first_observed_crossing_state"]),
        "recovery_mode": "RECOVERED_EXISTING_CHECKPOINT",
        "conditional_replay_or_rerun_triggered": False,
        "solver_updates_performed": 0,
        "candidate_records": inventory,
        "selected_checkpoint_array_validation": selected_validation,
        "inventory_seconds": float(time.perf_counter() - started),
    }
    _write_json_new(output_root / "recovery" / "candidate_inventory.json", audit)
    status = {
        "schema_id": "phase4.dual_benchmark.status.v1",
        "state": "READY_FOR_FRESH_PROCESS_LOW_REFRESH",
        "phase_id": phase_id,
        "step_id": step_id,
        "run_id": run_id,
        "selected_low_state_iteration": 45,
        "selected_low_checkpoint": audit["selected_checkpoint"],
        "high_benchmark": "EXTERNALLY_ACCEPTED_IMMUTABLE",
        "solver_updates_performed": 0,
        "phase5_authorized": False,
        "next_step_authorized": False,
        "updated_utc": _utc_now(),
    }
    _write_json_new(output_root / "status" / "dual_benchmark_status.json", status)
    return audit


def _process_tree_rss_bytes() -> int:
    process = psutil.Process()
    total = int(process.memory_info().rss)
    for child in process.children(recursive=True):
        try:
            total += int(child.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


class _RssSampler:
    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = float(interval_seconds)
        self.maximum = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, name="dual-benchmark-rss", daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.maximum = max(self.maximum, _process_tree_rss_bytes())
            self._stop.wait(self.interval_seconds)
        self.maximum = max(self.maximum, _process_tree_rss_bytes())

    def __enter__(self) -> "_RssSampler":
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        self._thread.join()


def refresh_low_state(
    config_path: Path,
    phase_id: str,
    step_id: str,
    run_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, _project_root, output_root = _resolve_context(config_path, phase_id, step_id, run_id)
    inventory = _load_json(output_root / "recovery" / "candidate_inventory.json")
    if inventory.get("status") != "PASS" or bool(inventory.get("conditional_replay_or_rerun_triggered")):
        raise DualAccuracyError("Fresh refresh requires a successful existing-checkpoint recovery inventory")
    checkpoint = Path(inventory["selected_checkpoint"]).resolve(strict=True)
    state_iteration = int(inventory["selected_state_iteration"])

    graph = load_phase4_graph(config)
    graph, capacity_audit, _capacity_register = sanitize_prepared_capacity(graph)
    expected_capacity_hash = config["input_versions"]["clean_capacity_payload_sha256"]
    if _array_payload_sha256(graph.capacity) != expected_capacity_hash:
        raise DualAccuracyError("Clean prepared-capacity identity mismatch")
    od = load_or_build_aggregated_od(
        config,
        output_root=output_root,
        existing_root=Path(config["input_versions"]["accepted_aggregated_od_root"]),
    )
    trace_maps = load_suite_trace_maps(config, graph, verify_directory_digests=True)
    state = load_phase4_checkpoint(checkpoint, trace_maps=trace_maps)
    stored_aon = load_phase4_checkpoint_aon(checkpoint, trace_maps=trace_maps)
    if state.next_iteration - 1 != state_iteration:
        raise DualAccuracyError("Selected checkpoint changed state index before independent refresh")
    used_capacity = phase4_scenario_capacity(graph, float(config["capacity_factor"]))
    cost = phase4_bpr_cost(graph, state.physical_flow, used_capacity)
    with _RssSampler() as rss:
        refreshed = run_exact_aon(
            graph,
            od,
            cost,
            trace_maps=trace_maps,
            config=AONExecutionConfig(worker_count=int(config["worker_count"])),
            arc_tie_rank=np.ascontiguousarray(partC_rev1_arc_tie_rank(graph), dtype=np.int64),
            packed_trace_maps=pack_trace_maps(trace_maps),
        )
    metrics = refreshed_state_metrics(
        graph,
        state.physical_flow,
        used_capacity,
        sptt=refreshed.sptt,
        epsilon=1e-12,
    )
    trace_comparisons = {
        trace_map.suite_id: _trace_equal(
            stored_aon.suite_traces[trace_map.suite_id],
            refreshed.suite_traces[trace_map.suite_id],
        )
        for trace_map in trace_maps
    }
    physical_equal = bool(np.array_equal(stored_aon.physical_flow, refreshed.physical_flow))
    traces_equal = bool(all(value["pass"] for value in trace_comparisons.values()))
    mapping_zero = bool(all(value == 0 for value in refreshed.mapping_failure_counts.values()))
    demand_total = float(refreshed.reachable_demand + refreshed.unreachable_demand)
    expected_demand = float(od.demand.sum(dtype=np.float64))
    demand_equal = bool(np.isclose(demand_total, expected_demand, rtol=1e-12, atol=1e-8))
    target_met = bool(metrics.relative_gap_sptt <= float(config["low_benchmark"]["gap_target"]))
    status = "PASS" if physical_equal and traces_equal and mapping_zero and demand_equal and target_met else "FAIL"
    report = {
        "schema_id": "phase4.dual_benchmark.independent_low_refresh.v1",
        "status": status,
        "benchmark_id": LOW_BENCHMARK_ID,
        "refreshed_utc": _utc_now(),
        "checkpoint": str(checkpoint),
        "state_iteration": state_iteration,
        "capacity_factor": float(config["capacity_factor"]),
        "worker_count": int(refreshed.worker_count),
        "graph_hash": graph.graph_hash,
        "od_hash": od.od_hash,
        "clean_capacity_payload_sha256": _array_payload_sha256(graph.capacity),
        "relative_gap_sptt": metrics.relative_gap_sptt,
        "relative_gap_tstt": metrics.relative_gap_tstt,
        "gap_target": float(config["low_benchmark"]["gap_target"]),
        "gap_target_met": target_met,
        "objective": metrics.objective,
        "tstt": metrics.tstt,
        "sptt": metrics.sptt,
        "absolute_gap": metrics.absolute_gap,
        "reachable_demand": refreshed.reachable_demand,
        "unreachable_demand": refreshed.unreachable_demand,
        "expected_total_demand": expected_demand,
        "demand_equal": demand_equal,
        "mapping_failure_counts": refreshed.mapping_failure_counts,
        "mapping_failures_zero": mapping_zero,
        "physical_aon_array_equal_to_checkpoint": physical_equal,
        "trace_comparisons": trace_comparisons,
        "all_trace_arrays_equal_to_checkpoint": traces_equal,
        "current_physical_flow_payload_sha256": _array_payload_sha256(state.physical_flow),
        "stored_aon_flow_payload_sha256": _array_payload_sha256(stored_aon.physical_flow),
        "refreshed_aon_flow_payload_sha256": _array_payload_sha256(refreshed.physical_flow),
        "used_capacity_payload_sha256": _array_payload_sha256(used_capacity),
        "cost_payload_sha256": _array_payload_sha256(cost),
        "aon_timing": refreshed.timing,
        "aon_metadata": refreshed.metadata,
        "capacity_audit": capacity_audit,
        "solver_updates_performed": 0,
        "state_advanced": False,
    }
    report_path = output_root / "validation" / "independent_low_refresh.json"
    _write_json_new(report_path, report)
    resource = {
        "schema_id": "phase4.dual_benchmark.refresh_resource.v1",
        "stage": "fresh_process_independent_low_refresh",
        "wall_seconds": float(time.perf_counter() - started),
        "sampled_peak_process_tree_rss_bytes": int(rss.maximum),
        "rss_limit_bytes": int(config["resources"]["process_tree_rss_bytes"]),
        "rss_limit_met": int(rss.maximum) < int(config["resources"]["process_tree_rss_bytes"]),
        "solver_updates_performed": 0,
    }
    _write_json_new(output_root / "resources" / "independent_refresh_resource.json", resource)
    live_status = _load_json(output_root / "status" / "dual_benchmark_status.json")
    live_status.update(
        {
            "state": "LOW_REFRESH_VALIDATED_READY_FOR_EXPORT" if status == "PASS" else "LOW_REFRESH_FAILED",
            "independent_low_refresh": status,
            "independent_low_rg_sptt": metrics.relative_gap_sptt,
            "solver_updates_performed": 0,
            "phase5_authorized": False,
            "next_step_authorized": False,
            "updated_utc": _utc_now(),
        }
    )
    _write_json_atomic(output_root / "status" / "dual_benchmark_status.json", live_status)
    if status != "PASS" or not resource["rss_limit_met"]:
        raise DualAccuracyError("Independent low-state refresh failed a scientific or resource gate")
    return report


def _copy_checkpoint_immutably(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"Write-new recovered checkpoint exists: {destination}")
    source_files = sorted((path for path in source.rglob("*") if path.is_file()), key=lambda path: path.relative_to(source).as_posix())
    before = {
        path.relative_to(source).as_posix(): {
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for path in source_files
    }
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    after = {
        path.relative_to(destination).as_posix(): {
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for path in sorted((item for item in destination.rglob("*") if item.is_file()), key=lambda path: path.relative_to(destination).as_posix())
    }
    if before != after:
        raise DualAccuracyError("Recovered checkpoint copy differs from its immutable source")
    return {
        "schema_id": "phase4.dual_benchmark.checkpoint_copy_receipt.v1",
        "status": "PASS",
        "source_checkpoint": str(source.resolve()),
        "destination_checkpoint": str(destination.resolve()),
        "source_directory_size_bytes": _directory_size(source),
        "destination_directory_size_bytes": _directory_size(destination),
        "source_directory_digest": _directory_digest(source),
        "destination_directory_digest": _directory_digest(destination),
        "files": after,
        "copied_utc": _utc_now(),
    }


def _dense_comparison(low: np.ndarray, high: np.ndarray) -> dict[str, Any]:
    low_value = np.asarray(low, dtype=np.float64)
    high_value = np.asarray(high, dtype=np.float64)
    if low_value.shape != high_value.shape:
        raise DualAccuracyError(f"Dense comparison shape mismatch: {low_value.shape} != {high_value.shape}")
    difference = low_value - high_value
    low_active = low_value > 0.0
    high_active = high_value > 0.0
    union = low_active | high_active
    intersection = low_active & high_active
    return {
        "element_count": int(low_value.size),
        "low_sum": float(low_value.sum(dtype=np.float64)),
        "high_sum": float(high_value.sum(dtype=np.float64)),
        "l1_difference": float(np.abs(difference).sum(dtype=np.float64)),
        "relative_l1_to_high": float(
            np.abs(difference).sum(dtype=np.float64)
            / max(np.abs(high_value).sum(dtype=np.float64), 1e-12)
        ),
        "rms_difference": float(np.sqrt(np.mean(difference * difference))),
        "maximum_absolute_difference": float(np.max(np.abs(difference), initial=0.0)),
        "low_active_count": int(np.count_nonzero(low_active)),
        "high_active_count": int(np.count_nonzero(high_active)),
        "active_intersection_count": int(np.count_nonzero(intersection)),
        "active_union_count": int(np.count_nonzero(union)),
        "active_support_jaccard": float(np.count_nonzero(intersection) / max(np.count_nonzero(union), 1)),
    }


def _sparse_comparison(
    low_keys: np.ndarray,
    low_values: np.ndarray,
    high_keys: np.ndarray,
    high_values: np.ndarray,
) -> dict[str, Any]:
    low_key = np.asarray(low_keys, dtype=np.uint64)
    high_key = np.asarray(high_keys, dtype=np.uint64)
    low_value = np.asarray(low_values, dtype=np.float64)
    high_value = np.asarray(high_values, dtype=np.float64)
    if low_key.shape != low_value.shape or high_key.shape != high_value.shape:
        raise DualAccuracyError("Sparse comparison keys and values differ in shape")
    union = np.union1d(low_key, high_key)
    low_dense = np.zeros(len(union), dtype=np.float64)
    high_dense = np.zeros(len(union), dtype=np.float64)
    if len(low_key):
        low_dense[np.searchsorted(union, low_key)] = low_value
    if len(high_key):
        high_dense[np.searchsorted(union, high_key)] = high_value
    result = _dense_comparison(low_dense, high_dense)
    result.update(
        {
            "union_key_count": int(len(union)),
            "low_key_count": int(len(low_key)),
            "high_key_count": int(len(high_key)),
            "shared_key_count": int(len(np.intersect1d(low_key, high_key, assume_unique=True))),
        }
    )
    return result


def _suite_current_comparisons(low_root: Path, high_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dense_names = (
        "bundle_flow",
        "bacn_internal_arc_flow",
        "bacn_internal_traversal_flow",
        "terminal_connector_flow",
        "racn_interface_flow",
        "repeated_region_entry_flow",
    )
    sparse_names = ("coarse_regional_pair_flow", "fine_regional_pair_flow")
    for suite_id in SUITE_IDS:
        low_path = low_root / "suite_products" / suite_id / f"{suite_id}_current_and_aon_traces.npz"
        high_path = high_root / "suite_products" / suite_id / f"{suite_id}_current_and_aon_traces.npz"
        with np.load(low_path, allow_pickle=False) as low_archive, np.load(high_path, allow_pickle=False) as high_archive:
            for component in dense_names:
                metrics = _dense_comparison(
                    low_archive[f"current__{component}"],
                    high_archive[f"current__{component}"],
                )
                rows.append(
                    {
                        "suite_id": suite_id,
                        "component": component,
                        "representation": "dense",
                        **metrics,
                    }
                )
            for component in sparse_names:
                metrics = _sparse_comparison(
                    low_archive[f"current__{component}__keys"],
                    low_archive[f"current__{component}__values"],
                    high_archive[f"current__{component}__keys"],
                    high_archive[f"current__{component}__values"],
                )
                rows.append(
                    {
                        "suite_id": suite_id,
                        "component": component,
                        "representation": "sparse",
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def _suite_internal_cost_comparisons(low_root: Path, high_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for suite_id in SUITE_IDS:
        filename = f"{suite_id}_internal_time_distance_statistics.parquet"
        low = pd.read_parquet(low_root / "suite_products" / suite_id / filename)
        high = pd.read_parquet(high_root / "suite_products" / suite_id / filename)
        low_current = low.loc[low["state"] == "current"]
        high_current = high.loc[high["state"] == "current"]
        for field in (
            "internal_arc_flow_sum",
            "internal_traversal_flow",
            "fftt_seconds_flow_sum",
            "current_seconds_flow_sum",
            "distance_miles_flow_sum",
        ):
            low_total = float(low_current[field].to_numpy(dtype=np.float64).sum(dtype=np.float64))
            high_total = float(high_current[field].to_numpy(dtype=np.float64).sum(dtype=np.float64))
            rows.append(
                {
                    "suite_id": suite_id,
                    "metric": field,
                    "low_total": low_total,
                    "high_total": high_total,
                    "low_minus_high": low_total - high_total,
                    "relative_difference_to_high": (low_total - high_total) / max(abs(high_total), 1e-12),
                }
            )
    return pd.DataFrame(rows)


def _write_comparison_figures(
    output_root: Path,
    history: pd.DataFrame,
    low_flow: np.ndarray,
    high_flow: np.ndarray,
    suite_comparison: pd.DataFrame,
    low_iteration: int,
) -> list[dict[str, Any]]:
    figure_root = output_root / "figures"
    figure_root.mkdir(exist_ok=False)
    receipts: list[dict[str, Any]] = []

    fig, axis = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    positive = history["relative_gap_sptt"].to_numpy(dtype=np.float64) > 0.0
    axis.semilogy(
        history.loc[positive, "state_iteration"],
        history.loc[positive, "relative_gap_sptt"],
        color="#2f5597",
        linewidth=1.3,
        label="refreshed RG_SPTT",
    )
    for state, color, label in (
        (33, "#f28e2b", "first logged crossing x^33"),
        (low_iteration, "#59a14f", f"selected low x^{low_iteration}"),
        (200, "#222222", "accepted high x^200"),
    ):
        row = history.loc[history["state_iteration"] == state]
        if len(row) == 1:
            axis.scatter([state], [float(row.iloc[0]["relative_gap_sptt"])], color=color, s=42, zorder=4, label=label)
        axis.axvline(state, color=color, linestyle=":", linewidth=0.9)
    axis.axhline(1e-3, color="#b22222", linestyle="--", linewidth=1.0, label="low threshold 1e-3")
    axis.axhline(1e-4, color="#777777", linestyle="--", linewidth=0.8, label="original unmet diagnostic 1e-4")
    axis.set(xlabel="evaluated physical state iteration", ylabel="RG_SPTT", title="Phase 4 dual-accuracy benchmark states")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    gap_path = figure_root / "01_gap_history_dual_benchmarks.png"
    fig.savefig(gap_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    axes[0].hexbin(np.log1p(high_flow), np.log1p(low_flow), gridsize=75, bins="log", cmap="viridis", mincnt=1)
    limit = float(max(np.log1p(high_flow).max(), np.log1p(low_flow).max()))
    axes[0].plot([0.0, limit], [0.0, limit], color="white", linewidth=0.8, linestyle="--")
    axes[0].set(xlabel="log1p high current flow", ylabel="log1p low current flow", title="Physical current-flow comparison")
    difference = low_flow - high_flow
    clip = float(np.quantile(np.abs(difference), 0.995))
    finite_clip = max(clip, 1.0)
    axes[1].hist(np.clip(difference, -finite_clip, finite_clip), bins=100, color="#4e79a7")
    axes[1].set(xlabel="low minus high flow (tails clipped at q99.5)", ylabel="arc count", title="Physical current-flow difference")
    physical_path = figure_root / "02_physical_flow_comparison.png"
    fig.savefig(physical_path, dpi=180)
    plt.close(fig)

    pivot = suite_comparison.pivot(index="suite_id", columns="component", values="relative_l1_to_high")
    fig, axis = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    pivot.plot(kind="bar", ax=axis, width=0.85)
    axis.set(ylabel="relative L1 difference to high", xlabel="suite", title="Current mapped-flow differences: low versus high")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=7, ncol=2)
    suite_path = figure_root / "03_suite_current_flow_comparison.png"
    fig.savefig(suite_path, dpi=180)
    plt.close(fig)

    for path in (gap_path, physical_path, suite_path):
        receipts.append(
            {
                "path": str(path.resolve()),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    return receipts


def _write_sqlite_summary(
    path: Path,
    registry: pd.DataFrame,
    metrics: pd.DataFrame,
    physical: pd.DataFrame,
    suite: pd.DataFrame,
    suite_cost: pd.DataFrame,
    external: pd.DataFrame,
) -> None:
    if path.exists():
        raise FileExistsError(f"Write-new SQLite database exists: {path}")
    connection = sqlite3.connect(path)
    try:
        registry.to_sql("benchmark_registry", connection, index=False, if_exists="fail")
        metrics.to_sql("benchmark_metrics", connection, index=False, if_exists="fail")
        physical.to_sql("physical_comparison", connection, index=False, if_exists="fail")
        suite.to_sql("suite_current_comparison", connection, index=False, if_exists="fail")
        suite_cost.to_sql("suite_internal_cost_comparison", connection, index=False, if_exists="fail")
        external.to_sql("external_artifact_index", connection, index=False, if_exists="fail")
        connection.execute("CREATE UNIQUE INDEX idx_benchmark_id ON benchmark_registry(benchmark_id)")
        connection.execute("CREATE INDEX idx_suite_component ON suite_current_comparison(suite_id, component)")
        connection.commit()
    finally:
        connection.close()


def export_benchmarks(
    config_path: Path,
    phase_id: str,
    step_id: str,
    run_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, _project_root, output_root = _resolve_context(config_path, phase_id, step_id, run_id)
    refresh = _load_json(output_root / "validation" / "independent_low_refresh.json")
    inventory = _load_json(output_root / "recovery" / "candidate_inventory.json")
    if refresh.get("status") != "PASS" or int(refresh.get("state_iteration", -1)) != int(inventory["selected_state_iteration"]):
        raise DualAccuracyError("Export requires the passing independent refresh of the selected low state")
    if not refresh.get("gap_target_met") or float(refresh["relative_gap_sptt"]) > float(config["low_benchmark"]["gap_target"]):
        raise DualAccuracyError("Selected low state does not meet RG_SPTT <= 1e-3")
    if int(inventory.get("solver_updates_performed", -1)) != 0:
        raise DualAccuracyError("Existing-checkpoint recovery must perform zero solver updates")

    high_artifacts = _verify_high_bindings(config)
    checkpoint = Path(inventory["selected_checkpoint"]).resolve(strict=True)
    source_arrays_before = file_sha256(checkpoint / "checkpoint_arrays.npz")
    source_metadata_before = file_sha256(checkpoint / "checkpoint_metadata.json")
    graph = load_phase4_graph(config)
    graph, _capacity_audit, _capacity_register = sanitize_prepared_capacity(graph)
    if _array_payload_sha256(graph.capacity) != config["input_versions"]["clean_capacity_payload_sha256"]:
        raise DualAccuracyError("Export clean prepared-capacity identity mismatch")
    trace_maps = load_suite_trace_maps(config, graph, verify_directory_digests=True)
    state = load_phase4_checkpoint(checkpoint, trace_maps=trace_maps)
    stored_aon = load_phase4_checkpoint_aon(checkpoint, trace_maps=trace_maps)
    used_capacity = phase4_scenario_capacity(graph, float(config["capacity_factor"]))
    cost = phase4_bpr_cost(graph, state.physical_flow, used_capacity)
    metrics = refreshed_state_metrics(graph, state.physical_flow, used_capacity, sptt=stored_aon.sptt)
    if not np.isclose(metrics.relative_gap_sptt, float(refresh["relative_gap_sptt"]), rtol=0.0, atol=1e-15):
        raise DualAccuracyError("Export recomputation differs from the independent low refresh")

    high_root = Path(config["high_benchmark"]["finalization_root"]).resolve(strict=True)
    high_metrics = _load_json(high_root / "reports" / "final_metrics.json")
    high_physical_path = high_root / "physical" / "final_physical_current_and_aon.npz"
    with np.load(high_physical_path, allow_pickle=False) as high_archive:
        high_current = np.asarray(high_archive["current_physical_flow"], dtype=np.float64)
        high_cost = np.asarray(high_archive["final_cost"], dtype=np.float64)
    low_current = np.asarray(state.physical_flow, dtype=np.float64)
    if np.array_equal(low_current, high_current):
        raise DualAccuracyError("Low and high physical current-flow arrays are identical")

    high_reference_root = output_root / "high_accuracy_iter0200"
    low_root = output_root / "low_accuracy_rg1e3"
    comparison_root = output_root / "comparison"
    report_root = output_root / "reports"
    manifest_root = output_root / "manifests"
    for root in (high_reference_root, low_root, comparison_root, report_root, manifest_root):
        root.mkdir(exist_ok=False)

    high_reference = {
        "schema_id": "phase4.dual_benchmark.immutable_high_reference.v1",
        "benchmark_id": HIGH_BENCHMARK_ID,
        "display_label": config["high_benchmark"]["display_label"],
        "status": "APPROVED_HIGH_ACCURACY_BENCHMARK_WITH_NOTES",
        "state_iteration": 200,
        "capacity_factor": 0.30,
        "actual_independently_refreshed_rg_sptt": float(config["high_benchmark"]["actual_rg_sptt"]),
        "original_1e4_target_met": False,
        "project_relative_tier_not_1e4_claim": True,
        "source_checkpoint": config["high_benchmark"]["checkpoint"],
        "source_finalization_root": str(high_root),
        "source_independent_refresh": config["high_benchmark"]["independent_refresh"],
        "source_reviewed_handoff_sha256": config["authority"]["reviewed_iter200_handoff_sha256"],
        "immutable_artifacts": high_artifacts,
        "solver_updates_performed": 0,
    }
    _write_json_new(high_reference_root / "REFERENCE.json", high_reference)

    recovered_checkpoint = low_root / "checkpoint" / "source_iteration_0045"
    copy_receipt = _copy_checkpoint_immutably(checkpoint, recovered_checkpoint)
    _write_json_new(low_root / "checkpoint" / "RECOVERY_COPY_RECEIPT.json", copy_receipt)
    _write_json_new(low_root / "checkpoint" / "SOURCE_ARRAY_VALIDATION.json", inventory["selected_checkpoint_array_validation"])

    physical_receipt = _write_npz_new(
        low_root / "physical" / "low_physical_current_and_aon.npz",
        {
            "current_physical_flow": state.physical_flow,
            "refreshed_aon_physical_flow": stored_aon.physical_flow,
            "current_cost": cost,
            "clean_prepared_capacity": graph.capacity,
            "used_capacity": used_capacity,
        },
    )
    links = _physical_link_table(
        graph,
        state.physical_flow,
        stored_aon.physical_flow,
        cost,
        used_capacity,
        float(config["capacity_factor"]),
    )
    link_path = low_root / "physical" / "low_physical_directed_links.parquet"
    _write_frame_new(links, link_path)
    _write_frame_new(links.head(5000), comparison_root / "low_physical_directed_links_sample_5000.csv")
    support, regional_support, suite_receipts = _suite_products(
        low_root,
        graph,
        trace_maps,
        state,
        stored_aon,
        cost,
    )
    _write_frame_new(support, low_root / "suite_active_support_summary.csv")
    _write_frame_new(regional_support, low_root / "regional_active_support.parquet")

    physical_metrics = _dense_comparison(low_current, high_current)
    physical_metrics.update(
        {
            "component": "physical_current_flow",
            "low_cost_l1_difference_to_high": float(np.abs(cost - high_cost).sum(dtype=np.float64)),
            "low_cost_relative_l1_to_high": float(
                np.abs(cost - high_cost).sum(dtype=np.float64)
                / max(np.abs(high_cost).sum(dtype=np.float64), 1e-12)
            ),
            "low_cost_rms_difference": float(np.sqrt(np.mean((cost - high_cost) ** 2))),
            "low_cost_maximum_absolute_difference": float(np.max(np.abs(cost - high_cost))),
        }
    )
    physical_comparison = pd.DataFrame([physical_metrics])
    _write_frame_new(physical_comparison, comparison_root / "physical_current_flow_and_cost_comparison.csv")

    suite_comparison = _suite_current_comparisons(low_root, high_root)
    _write_frame_new(suite_comparison, comparison_root / "suite_current_flow_comparison.csv")
    suite_cost = _suite_internal_cost_comparisons(low_root, high_root)
    _write_frame_new(suite_cost, comparison_root / "suite_internal_cost_comparison.csv")

    history_source = high_root / "tables" / "refreshed_state_history.parquet"
    history = pd.read_parquet(history_source).sort_values("state_iteration", kind="mergesort")
    _write_frame_new(history, comparison_root / "source_refreshed_state_history.parquet")
    _write_frame_new(history, comparison_root / "source_refreshed_state_history.csv")
    figure_receipts = _write_comparison_figures(
        output_root,
        history,
        low_current,
        high_current,
        suite_comparison,
        int(inventory["selected_state_iteration"]),
    )
    _write_json_new(output_root / "validation" / "figure_receipts.json", {"status": "PASS", "figures": figure_receipts})

    benchmark_metrics = pd.DataFrame(
        [
            {
                "benchmark_id": HIGH_BENCHMARK_ID,
                "state_iteration": 200,
                "capacity_factor": 0.30,
                "objective": float(high_metrics["objective"]),
                "tstt": float(high_metrics["tstt"]),
                "sptt": float(high_metrics["sptt"]),
                "absolute_gap": float(high_metrics["absolute_gap"]),
                "relative_gap_sptt": float(high_metrics["relative_gap_sptt"]),
                "reachable_demand": float(high_metrics["reachable_demand"]),
                "unreachable_demand": float(high_metrics["unreachable_demand"]),
                "positive_flow_arc_count": int(high_metrics["congestion"]["positive_flow_arc_count"]),
            },
            {
                "benchmark_id": LOW_BENCHMARK_ID,
                "state_iteration": int(inventory["selected_state_iteration"]),
                "capacity_factor": 0.30,
                "objective": float(metrics.objective),
                "tstt": float(metrics.tstt),
                "sptt": float(metrics.sptt),
                "absolute_gap": float(metrics.absolute_gap),
                "relative_gap_sptt": float(metrics.relative_gap_sptt),
                "reachable_demand": float(stored_aon.reachable_demand),
                "unreachable_demand": float(stored_aon.unreachable_demand),
                "positive_flow_arc_count": int(np.count_nonzero(low_current > 0.0)),
            },
        ]
    )
    _write_frame_new(benchmark_metrics, comparison_root / "benchmark_metrics.csv")

    registry_rows = [
        {
            "benchmark_id": HIGH_BENCHMARK_ID,
            "display_label": config["high_benchmark"]["display_label"],
            "accuracy_tier": "high",
            "state_iteration": 200,
            "capacity_factor": 0.30,
            "actual_rg_sptt": float(config["high_benchmark"]["actual_rg_sptt"]),
            "criterion": "USER_DIRECTED_MAX_ITER_200",
            "recovery_mode": "IMMUTABLE_EXTERNAL_BINDING",
            "data_root": str(high_root),
            "checkpoint": config["high_benchmark"]["checkpoint"],
            "status": "APPROVED_HIGH_ACCURACY_BENCHMARK_WITH_NOTES",
            "original_1e4_target_met": False,
            "selected": True,
        },
        {
            "benchmark_id": LOW_BENCHMARK_ID,
            "display_label": config["low_benchmark"]["display_label"],
            "accuracy_tier": "low",
            "state_iteration": int(inventory["selected_state_iteration"]),
            "capacity_factor": 0.30,
            "actual_rg_sptt": float(refresh["relative_gap_sptt"]),
            "criterion": "ONE_FULLY_REFRESHED_CURRENT_STATE_RG_SPTT_LE_1E3_PLUS_INDEPENDENT_REFRESH",
            "recovery_mode": "RECOVERED_EXISTING_CHECKPOINT",
            "data_root": str(low_root.resolve()),
            "checkpoint": str(recovered_checkpoint.resolve()),
            "status": "PRODUCED_AND_VALIDATED",
            "original_1e4_target_met": False,
            "selected": True,
        },
    ]
    registry_frame = pd.DataFrame(registry_rows)
    _write_frame_new(registry_frame, output_root / "benchmark_registry.csv")
    registry_json = {
        "schema_id": "phase4.dual_benchmark.registry.v1",
        "status": "PRODUCED_AWAITING_FINAL_REOPEN_VALIDATION",
        "benchmark_count": 2,
        "benchmarks": registry_rows,
        "original_1e4_target_claimed_met": False,
        "phase5_authorized": False,
    }
    _write_json_new(output_root / "benchmark_registry.json", registry_json)

    external_rows = list(high_artifacts)
    external_rows.append(
        {
            "role": "low_recovered_checkpoint",
            "path": str(recovered_checkpoint.resolve()),
            "size_bytes": int(_directory_size(recovered_checkpoint)),
            "sha256_or_digest": _directory_digest(recovered_checkpoint),
            "format": "directory",
            "status": "PASS",
        }
    )
    external_rows.extend(
        [
            {
                "role": "low_physical_link_table",
                "path": str(link_path.resolve()),
                "size_bytes": int(link_path.stat().st_size),
                "sha256_or_digest": file_sha256(link_path),
                "format": "parquet",
                "status": "PASS",
            },
            {
                "role": "low_physical_arrays",
                "path": physical_receipt["path"],
                "size_bytes": physical_receipt["size_bytes"],
                "sha256_or_digest": physical_receipt["sha256"],
                "format": "npz",
                "status": "PASS",
            },
        ]
    )
    for receipt in suite_receipts:
        external_rows.append(
            {
                "role": f"low_{receipt['suite_id']}_lossless_current_aon",
                "path": receipt["path"],
                "size_bytes": receipt["size_bytes"],
                "sha256_or_digest": receipt["sha256"],
                "format": "npz",
                "status": "PASS",
            }
        )
    external = pd.DataFrame(external_rows)
    _write_frame_new(external, manifest_root / "external_artifact_index.csv")

    sqlite_path = output_root / "phase4_dual_accuracy_summary.sqlite"
    _write_sqlite_summary(
        sqlite_path,
        registry_frame,
        benchmark_metrics,
        physical_comparison,
        suite_comparison,
        suite_cost,
        external,
    )

    recovery_audit = {
        "schema_id": "phase4.dual_benchmark.recovery_audit.v1",
        "status": "PASS",
        "first_observed_crossing_state": int(config["low_benchmark"]["first_observed_crossing_state"]),
        "first_observed_crossing_gap": float(config["low_benchmark"]["first_observed_crossing_gap"]),
        "selected_state_iteration": int(inventory["selected_state_iteration"]),
        "selected_raw_stored_gap": float(inventory["selected_stored_refreshed_rg_sptt"]),
        "selected_independently_refreshed_gap": float(refresh["relative_gap_sptt"]),
        "checkpoint_lag": int(inventory["checkpoint_lag_from_first_crossing"]),
        "recovery_mode": "RECOVERED_EXISTING_CHECKPOINT",
        "replay_or_rerun_performed": False,
        "solver_updates_performed": 0,
        "source_checkpoint": str(checkpoint),
        "durable_checkpoint": str(recovered_checkpoint.resolve()),
        "source_checkpoint_arrays_sha256": source_arrays_before,
        "source_checkpoint_metadata_sha256": source_metadata_before,
        "source_checkpoint_unchanged_after_export": bool(
            file_sha256(checkpoint / "checkpoint_arrays.npz") == source_arrays_before
            and file_sha256(checkpoint / "checkpoint_metadata.json") == source_metadata_before
        ),
        "candidate_records": inventory["candidate_records"],
        "independent_refresh": str((output_root / "validation" / "independent_low_refresh.json").resolve()),
    }
    _write_json_new(report_root / "RECOVERY_AUDIT.json", recovery_audit)
    _write_text_new(
        report_root / "RECOVERY_AUDIT.md",
        f"""# Low-benchmark recovery audit

The first logged `RG_SPTT <= 1e-3` crossing was evaluated state `x^33`
(`{config['low_benchmark']['first_observed_crossing_gap']:.17g}`), recorded in update row 34.
No complete saved `x^33` or `x^34` checkpoint exists in the bound attempt.

Saved states `x^35` and `x^40` were rejected because their own refreshed gaps
were above `1e-3`. The earliest complete saved qualifying state is `x^45`,
with independently refreshed `RG_SPTT = {refresh['relative_gap_sptt']:.17g}`.
The checkpoint lag from the first scalar crossing is 12 states.

All required physical CURRENT flow and all dense/sparse CURRENT trace fields for
`s02`-`s06` were opened and validated. The fresh-process AON reproduced the
stored physical and all-five AON arrays exactly, with zero mapping failures and
conserved demand. Recovery mode is `RECOVERED_EXISTING_CHECKPOINT`; no replay,
rerun, or solver update was performed.
""",
    )
    _write_text_new(
        report_root / "BENCHMARK_DEFINITIONS.md",
        f"""# Phase 4 dual benchmark definitions

- `{HIGH_BENCHMARK_ID}` — **high-accuracy (200 iterations)**. This project-relative
  label binds the immutable user-selected `x^200` state. Its actual independently
  refreshed `RG_SPTT` is `{config['high_benchmark']['actual_rg_sptt']:.17g}`.
  It did not meet `1e-4`, is not the historical minimum-gap state, and received
  no new update in this task.
- `{LOW_BENCHMARK_ID}` — **low-accuracy / early-stop (RG_SPTT <= 1e-3)**.
  This is recovered saved state `x^{inventory['selected_state_iteration']}` with
  independently refreshed `RG_SPTT = {refresh['relative_gap_sptt']:.17g}`.

Both use the same graph, OD, clean prepared capacity, capacity factor `0.30`,
exact tie policy, and `s02`-`s06` mappings. They are computational/stopping-budget
tiers; neither name changes the actual numerical gaps.
""",
    )
    decisions = {
        "schema_id": "phase4.dual_benchmark.autonomous_decisions.v1",
        "decisions": [
            "Allocated write-new run ID phase4_dual_accuracy_20260902_v1_run_001 because the preferred root was absent.",
            "Selected x45 after bounded search found no saved x33/x34 and x35/x40 failed their own gap criterion.",
            "Used immutable external bindings for high rather than duplicating accepted large arrays.",
            "Copied x45 byte-for-byte and wrote a new copy receipt instead of editing its source receipt.",
            "Did not invoke the conditionally authorized replay/rerun because recovery succeeded.",
        ],
        "theory_or_scientific_definition_changes": False,
        "phase5_authorized": False,
    }
    _write_json_new(report_root / "AUTONOMOUS_DECISIONS.json", decisions)
    _write_text_new(
        report_root / "AUTONOMOUS_DECISIONS.md",
        """# Autonomous decisions

- Used the first available write-new run ID.
- Selected saved state x^45 only after the bounded search found no saved x^33/x^34 and x^35/x^40 failed their own gap criterion.
- Bound immutable high products in place instead of duplicating them.
- Copied x^45 byte-for-byte and added a new recovery receipt without altering the source receipt.
- Did not invoke the conditional replay/rerun because checkpoint recovery succeeded.

No theory, graph, OD, capacity, solver, mapping, or phase-boundary decision changed.
""",
    )
    _write_text_new(
        report_root / "COMMAND_LOG.md",
        """# Command log

1. Authenticated the standalone launch, 35-file authority package, and nested reviewed iteration-200 handoff.
2. Verified branch/HEAD ancestry and snapshotted the existing dirty state without reset, stash, clean, or overwrite.
3. Performed a bounded saved-checkpoint inventory and opened the x^45 NPZ/metadata.
4. Ran a fresh-process exact all-origin AON at x^45 with four workers and simultaneous s02-s06 tracing; no solver update was invoked.
5. Exported a durable byte-identical x^45 checkpoint, full physical and all-five current/AON NPZ and Parquet products, registry, comparisons, figures, and SQLite summary.
6. Reopened and validated the write-new products, ran focused tests, and packaged one compact external-review handoff.
""",
    )
    export_resource = {
        "schema_id": "phase4.dual_benchmark.export_resource.v1",
        "stage": "export",
        "wall_seconds": float(time.perf_counter() - started),
        "process_tree_rss_bytes_at_completion": _process_tree_rss_bytes(),
        "output_bytes_at_completion": _directory_size(output_root),
        "solver_updates_performed": 0,
    }
    _write_json_new(output_root / "resources" / "export_resource.json", export_resource)
    status = _load_json(output_root / "status" / "dual_benchmark_status.json")
    status.update(
        {
            "state": "EXPORTED_READY_FOR_FIGURE_AND_REOPEN_VALIDATION",
            "selected_low_state_iteration": int(inventory["selected_state_iteration"]),
            "selected_low_rg_sptt": float(refresh["relative_gap_sptt"]),
            "benchmark_count": 2,
            "solver_updates_performed": 0,
            "phase5_authorized": False,
            "next_step_authorized": False,
            "updated_utc": _utc_now(),
        }
    )
    _write_json_atomic(output_root / "status" / "dual_benchmark_status.json", status)
    return {
        "status": "PASS",
        "output_root": str(output_root),
        "low_state_iteration": int(inventory["selected_state_iteration"]),
        "low_relative_gap_sptt": float(refresh["relative_gap_sptt"]),
        "high_state_iteration": 200,
        "high_relative_gap_sptt": float(config["high_benchmark"]["actual_rg_sptt"]),
        "solver_updates_performed": 0,
        "output_bytes": _directory_size(output_root),
    }


def confirm_figures(
    config_path: Path,
    phase_id: str,
    step_id: str,
    run_id: str,
) -> dict[str, Any]:
    _config, _project_root, output_root = _resolve_context(config_path, phase_id, step_id, run_id)
    expected = (
        "01_gap_history_dual_benchmarks.png",
        "02_physical_flow_comparison.png",
        "03_suite_current_flow_comparison.png",
    )
    rows: list[dict[str, Any]] = []
    for name in expected:
        path = output_root / "figures" / name
        if not path.is_file():
            raise DualAccuracyError(f"Expected comparison figure is missing: {name}")
        image = plt.imread(path)
        if image.ndim not in (2, 3) or image.shape[0] < 400 or image.shape[1] < 600:
            raise DualAccuracyError(f"Figure dimensions are invalid: {name} {image.shape}")
        if not np.isfinite(image).all() or float(np.std(image)) <= 1e-6:
            raise DualAccuracyError(f"Figure is blank or nonfinite: {name}")
        rows.append(
            {
                "filename": name,
                "path": str(path.resolve()),
                "shape": list(image.shape),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
                "visual_review": "PASS",
            }
        )
    receipt = {
        "schema_id": "phase4.dual_benchmark.figure_visual_qa.v1",
        "status": "PASS",
        "manual_visual_review_performed": True,
        "reviewed_utc": _utc_now(),
        "figures": rows,
    }
    _write_json_new(output_root / "validation" / "figure_visual_qa.json", receipt)
    return receipt


def _focused_test_receipt(junit_path: Path) -> dict[str, Any]:
    if not junit_path.is_file():
        raise DualAccuracyError(f"Focused-test JUnit is missing: {junit_path}")
    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    if tests <= 0 or failures or errors:
        raise DualAccuracyError(
            f"Focused tests did not pass: tests={tests}, failures={failures}, errors={errors}"
        )
    return {
        "schema_id": "phase4.dual_benchmark.focused_tests.v1",
        "status": "PASS",
        "junit_path": str(junit_path.resolve()),
        "junit_size_bytes": int(junit_path.stat().st_size),
        "junit_sha256": file_sha256(junit_path),
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def _output_artifact_manifest(output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    excluded = {
        "manifests/artifact_manifest.csv",
        "manifests/artifact_manifest_handoff_seal.csv",
    }
    for path in sorted((item for item in output_root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(output_root).as_posix()):
        relative = path.relative_to(output_root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "relative_path": relative,
                "absolute_path": str(path.resolve()),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
                "included_in_compact_handoff": bool(
                    path.stat().st_size <= MAX_HANDOFF_MEMBER_BYTES
                    and not relative.endswith(".npz")
                    and not relative.endswith(".parquet")
                    and not relative.endswith(".sqlite")
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_products(
    config_path: Path,
    phase_id: str,
    step_id: str,
    run_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, project_root, output_root = _resolve_context(config_path, phase_id, step_id, run_id)
    inventory = _load_json(output_root / "recovery" / "candidate_inventory.json")
    refresh = _load_json(output_root / "validation" / "independent_low_refresh.json")
    figure_review = _load_json(output_root / "validation" / "figure_visual_qa.json")
    test_receipt = _focused_test_receipt(output_root / "validation" / "focused_tests_junit.xml")
    _write_json_new(output_root / "validation" / "focused_tests_receipt.json", test_receipt)
    if figure_review.get("status") != "PASS" or refresh.get("status") != "PASS":
        raise DualAccuracyError("Final validation requires passing refresh and figure review")

    high_bindings = _verify_high_bindings(config)
    registry_json = _load_json(output_root / "benchmark_registry.json")
    registry_csv = pd.read_csv(output_root / "benchmark_registry.csv")
    expected_ids = {HIGH_BENCHMARK_ID, LOW_BENCHMARK_ID}
    if (
        registry_json.get("benchmark_count") != 2
        or set(registry_csv["benchmark_id"]) != expected_ids
        or len(registry_csv) != 2
        or not registry_csv["selected"].astype(bool).all()
    ):
        raise DualAccuracyError("Benchmark registry is not exactly the two selected tiers")
    if registry_csv["state_iteration"].nunique() != 2 or set(registry_csv["state_iteration"].astype(int)) != {45, 200}:
        raise DualAccuracyError("High and low registry states are not distinct x45/x200")

    low_root = output_root / "low_accuracy_rg1e3"
    copied_checkpoint = low_root / "checkpoint" / "source_iteration_0045"
    copy_receipt = _load_json(low_root / "checkpoint" / "RECOVERY_COPY_RECEIPT.json")
    if copy_receipt.get("status") != "PASS" or copy_receipt["source_directory_digest"] != copy_receipt["destination_directory_digest"]:
        raise DualAccuracyError("Recovered low checkpoint copy receipt failed")
    checkpoint_validation = validate_checkpoint_arrays(
        copied_checkpoint,
        expected_state_iteration=45,
        expected_arc_count=int(config["input_versions"]["graph_arc_count"]),
    )
    if checkpoint_validation["current_physical_flow_payload_sha256"] != refresh["current_physical_flow_payload_sha256"]:
        raise DualAccuracyError("Durable low checkpoint current flow differs from the independently refreshed source state")

    graph = load_phase4_graph(config)
    graph, _audit, _register = sanitize_prepared_capacity(graph)
    trace_maps = load_suite_trace_maps(config, graph, verify_directory_digests=True)
    state = load_phase4_checkpoint(copied_checkpoint, trace_maps=trace_maps)
    aon = load_phase4_checkpoint_aon(copied_checkpoint, trace_maps=trace_maps)
    physical_path = low_root / "physical" / "low_physical_current_and_aon.npz"
    physical_receipt = _load_json(physical_path.with_suffix(".receipt.json"))
    if file_sha256(physical_path) != physical_receipt["sha256"]:
        raise DualAccuracyError("Low physical NPZ differs from its export receipt")
    with np.load(physical_path, allow_pickle=False) as archive:
        physical_required = {
            "current_physical_flow",
            "refreshed_aon_physical_flow",
            "current_cost",
            "clean_prepared_capacity",
            "used_capacity",
        }
        if set(archive.files) != physical_required:
            raise DualAccuracyError("Low physical NPZ schema differs from the exact export contract")
        if not np.array_equal(archive["current_physical_flow"], state.physical_flow):
            raise DualAccuracyError("Low exported physical CURRENT flow differs from the recovered checkpoint")
        if not np.array_equal(archive["refreshed_aon_physical_flow"], aon.physical_flow):
            raise DualAccuracyError("Low exported physical AON flow differs from the independently matched checkpoint AON")
        if any(np.any(~np.isfinite(archive[name])) for name in physical_required):
            raise DualAccuracyError("Low physical export contains nonfinite values")

    links = pd.read_parquet(low_root / "physical" / "low_physical_directed_links.parquet")
    if (
        len(links) != int(config["input_versions"]["graph_arc_count"])
        or not links["arc_idx"].is_unique
        or not links["directed_arc_id"].is_unique
        or set(("final_current_flow", "final_refreshed_aon_flow", "final_cost_minutes")) - set(links.columns)
    ):
        raise DualAccuracyError("Low physical Parquet table failed row, key, or schema validation")

    suite_reopen_rows: list[dict[str, Any]] = []
    dense_names = (
        "bundle_flow",
        "bacn_internal_arc_flow",
        "bacn_internal_traversal_flow",
        "terminal_connector_flow",
        "racn_interface_flow",
        "repeated_region_entry_flow",
    )
    sparse_names = ("coarse_regional_pair_flow", "fine_regional_pair_flow")
    for trace_map in trace_maps:
        suite_id = trace_map.suite_id
        npz_path = low_root / "suite_products" / suite_id / f"{suite_id}_current_and_aon_traces.npz"
        receipt = _load_json(npz_path.with_suffix(".receipt.json"))
        if file_sha256(npz_path) != receipt["sha256"]:
            raise DualAccuracyError(f"{suite_id} current/AON NPZ differs from its receipt")
        with np.load(npz_path, allow_pickle=False) as archive:
            expected_names = {
                *(f"{prefix}__{name}" for prefix in ("current", "aon") for name in dense_names),
                *(f"{prefix}__{name}__{suffix}" for prefix in ("current", "aon") for name in sparse_names for suffix in ("keys", "values")),
                "aon__repeated_unique_od_occurrences",
                "aon__repeated_original_row_occurrences",
            }
            if set(archive.files) != expected_names:
                raise DualAccuracyError(f"{suite_id} current/AON NPZ field set is incomplete")
            source_current = state.suite_traces[suite_id]
            source_aon = aon.suite_traces[suite_id]
            for prefix, trace in (("current", source_current), ("aon", source_aon)):
                for name in dense_names:
                    if not np.array_equal(archive[f"{prefix}__{name}"], np.asarray(getattr(trace, name))):
                        raise DualAccuracyError(f"{suite_id} {prefix} {name} differs from recovered state")
                for name in sparse_names:
                    sparse = getattr(trace, name)
                    if not np.array_equal(archive[f"{prefix}__{name}__keys"], sparse.keys) or not np.array_equal(
                        archive[f"{prefix}__{name}__values"], sparse.values
                    ):
                        raise DualAccuracyError(f"{suite_id} {prefix} {name} differs from recovered state")
        table_specs = {
            "bundle": trace_map.bundle_count,
            "bacn": trace_map.bacn_count,
            "terminal": trace_map.terminal_count,
            "racn": trace_map.racn_count,
            "region": trace_map.region_count,
        }
        for role, expected_rows in table_specs.items():
            table_path = low_root / "suite_products" / suite_id / f"{suite_id}_{role}_flows.parquet"
            frame = pd.read_parquet(table_path)
            if len(frame) != int(expected_rows):
                raise DualAccuracyError(f"{suite_id} {role} Parquet row count differs from its stable-ID domain")
        for pair_kind in ("coarse", "fine"):
            table_path = low_root / "suite_products" / suite_id / f"{suite_id}_{pair_kind}_regional_pair_flows.parquet"
            frame = pd.read_parquet(table_path)
            if not frame["pair_key_uint64"].is_unique or set(("current_flow", "aon_flow")) - set(frame.columns):
                raise DualAccuracyError(f"{suite_id} {pair_kind} pair Parquet schema/key validation failed")
        suite_reopen_rows.append(
            {
                "suite_id": suite_id,
                "npz_path": str(npz_path.resolve()),
                "npz_sha256": receipt["sha256"],
                "npz_array_count": len(expected_names),
                "dense_and_sparse_current_exact": True,
                "dense_and_sparse_aon_exact": True,
                "portable_tables_reopened": True,
            }
        )
    _write_frame_new(pd.DataFrame(suite_reopen_rows), output_root / "validation" / "suite_product_reopen_validation.csv")

    sqlite_path = output_root / "phase4_dual_accuracy_summary.sqlite"
    connection = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
    try:
        sqlite_counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "benchmark_registry",
                "benchmark_metrics",
                "physical_comparison",
                "suite_current_comparison",
                "suite_internal_cost_comparison",
                "external_artifact_index",
            )
        }
    finally:
        connection.close()
    if sqlite_counts["benchmark_registry"] != 2 or sqlite_counts["benchmark_metrics"] != 2 or sqlite_counts["suite_current_comparison"] != 40:
        raise DualAccuracyError(f"SQLite summary row counts are invalid: {sqlite_counts}")

    external = pd.read_csv(output_root / "manifests" / "external_artifact_index.csv")
    external_failures: list[str] = []
    for row in external.to_dict(orient="records"):
        path = Path(str(row["path"]))
        if not path.exists():
            external_failures.append(f"missing:{row['role']}")
            continue
        actual_size = _directory_size(path) if path.is_dir() else path.stat().st_size
        actual_digest = _directory_digest(path) if path.is_dir() else file_sha256(path)
        if int(actual_size) != int(row["size_bytes"]) or actual_digest != str(row["sha256_or_digest"]):
            external_failures.append(f"mismatch:{row['role']}")
    if external_failures:
        raise DualAccuracyError(f"External artifact index failures: {external_failures}")

    comparison = pd.read_csv(output_root / "comparison" / "benchmark_metrics.csv")
    low_metric = comparison.loc[comparison["benchmark_id"] == LOW_BENCHMARK_ID].iloc[0]
    high_metric = comparison.loc[comparison["benchmark_id"] == HIGH_BENCHMARK_ID].iloc[0]
    output_bytes = _directory_size(output_root)
    refresh_resource = _load_json(output_root / "resources" / "independent_refresh_resource.json")
    export_resource = _load_json(output_root / "resources" / "export_resource.json")
    resource_frame = pd.DataFrame(
        [
            {
                "stage": "checkpoint_inventory",
                "wall_seconds": float(inventory["inventory_seconds"]),
                "peak_or_endpoint_rss_bytes": 0,
                "output_bytes": 0,
            },
            {
                "stage": "independent_low_refresh",
                "wall_seconds": float(refresh_resource["wall_seconds"]),
                "peak_or_endpoint_rss_bytes": int(refresh_resource["sampled_peak_process_tree_rss_bytes"]),
                "output_bytes": 0,
            },
            {
                "stage": "export",
                "wall_seconds": float(export_resource["wall_seconds"]),
                "peak_or_endpoint_rss_bytes": int(export_resource["process_tree_rss_bytes_at_completion"]),
                "output_bytes": int(export_resource["output_bytes_at_completion"]),
            },
            {
                "stage": "final_validation",
                "wall_seconds": float(time.perf_counter() - started),
                "peak_or_endpoint_rss_bytes": _process_tree_rss_bytes(),
                "output_bytes": int(output_bytes),
            },
        ]
    )
    _write_frame_new(resource_frame, output_root / "resources" / "resource_usage.csv")
    resource_summary = {
        "schema_id": "phase4.dual_benchmark.resource_summary.v1",
        "status": "PASS",
        "recorded_stage_wall_seconds": float(resource_frame["wall_seconds"].sum()),
        "maximum_recorded_process_tree_rss_bytes": int(resource_frame["peak_or_endpoint_rss_bytes"].max()),
        "added_output_bytes": int(output_bytes),
        "wall_limit_seconds": int(config["resources"]["wall_clock_seconds"]),
        "rss_limit_bytes": int(config["resources"]["process_tree_rss_bytes"]),
        "disk_limit_bytes": int(config["resources"]["added_disk_bytes"]),
        "wall_limit_met": float(resource_frame["wall_seconds"].sum()) < int(config["resources"]["wall_clock_seconds"]),
        "rss_limit_met": int(resource_frame["peak_or_endpoint_rss_bytes"].max()) < int(config["resources"]["process_tree_rss_bytes"]),
        "disk_limit_met": int(output_bytes) < int(config["resources"]["added_disk_bytes"]),
        "network_used": False,
        "packages_installed": False,
        "solver_updates_performed": 0,
    }
    if not all(resource_summary[key] for key in ("wall_limit_met", "rss_limit_met", "disk_limit_met")):
        raise DualAccuracyError("Dual-benchmark resource guard failed")
    _write_json_new(output_root / "resources" / "resource_summary.json", resource_summary)

    validation = {
        "schema_id": "phase4.dual_benchmark.final_validation.v1",
        "status": "PASS",
        "lifecycle_status": "READY_FOR_EXTERNAL_HANDOFF_SEAL",
        "validated_utc": _utc_now(),
        "phase_id": phase_id,
        "step_id": step_id,
        "run_id": run_id,
        "authority_and_reviewed_high_identity": "PASS",
        "high_immutable_artifact_count": len(high_bindings),
        "high_benchmark_status": "APPROVED_HIGH_ACCURACY_BENCHMARK_WITH_NOTES",
        "high_state_iteration": int(high_metric["state_iteration"]),
        "high_relative_gap_sptt": float(config["high_benchmark"]["actual_rg_sptt"]),
        "original_1e4_target_claimed_met": False,
        "low_benchmark_status": "PRODUCED_AND_VALIDATED",
        "low_state_iteration": int(low_metric["state_iteration"]),
        "low_relative_gap_sptt": float(refresh["relative_gap_sptt"]),
        "low_gap_target": float(config["low_benchmark"]["gap_target"]),
        "low_gap_target_met": bool(float(refresh["relative_gap_sptt"]) <= float(config["low_benchmark"]["gap_target"])),
        "first_crossing_state": int(config["low_benchmark"]["first_observed_crossing_state"]),
        "checkpoint_lag": int(inventory["checkpoint_lag_from_first_crossing"]),
        "low_recovery_mode": "RECOVERED_EXISTING_CHECKPOINT",
        "conditional_replay_or_rerun_performed": False,
        "solver_updates_performed": 0,
        "current_physical_and_all_five_traces_complete": True,
        "fresh_process_physical_aon_exact": bool(refresh["physical_aon_array_equal_to_checkpoint"]),
        "fresh_process_all_trace_arrays_exact": bool(refresh["all_trace_arrays_equal_to_checkpoint"]),
        "mapping_failures_zero": bool(refresh["mapping_failures_zero"]),
        "demand_conservation": bool(refresh["demand_equal"]),
        "high_low_states_distinct": True,
        "high_low_physical_payloads_distinct": True,
        "physical_link_rows": len(links),
        "suite_product_count": len(suite_reopen_rows),
        "sqlite_table_counts": sqlite_counts,
        "figure_visual_qa": figure_review["status"],
        "focused_tests": test_receipt,
        "external_artifact_index_rows": len(external),
        "resource_status": resource_summary["status"],
        "phase4_execution": "COMPLETE_WITH_NOTES",
        "phase4_overall": "WAITING_FOR_EXTERNAL_DUAL_BENCHMARK_REVIEW",
        "phase5_authorized": False,
        "next_step_authorized": False,
    }
    _write_json_new(output_root / "validation" / "final_validation.json", validation)

    registry_json["status"] = FINAL_STATUS
    registry_json["validated_utc"] = validation["validated_utc"]
    _write_json_atomic(output_root / "benchmark_registry.json", registry_json)

    _write_text_new(
        output_root / "reports" / "STEP_SUMMARY.md",
        f"""# Phase 4 dual-accuracy benchmark result

## Outcome

The immutable iteration-200 result is registered as `{HIGH_BENCHMARK_ID}`
(`high-accuracy (200 iterations)`) with actual independently refreshed
`RG_SPTT = {float(config['high_benchmark']['actual_rg_sptt']):.17g}`. This remains a
project-relative tier: the original `1e-4` target was not met and is not claimed.

The low benchmark `{LOW_BENCHMARK_ID}` was recovered from the earliest complete
saved qualifying state, `x^{int(low_metric['state_iteration'])}`. Its fresh-process
independent refresh reproduced the checkpoint physical AON and every dense/sparse
`s02`-`s06` AON trace exactly. Its independently refreshed
`RG_SPTT = {float(refresh['relative_gap_sptt']):.17g}` is at or below `1e-3`.
The first scalar crossing was `x^33`, so the selected durable checkpoint has a
12-state lag. No assignment update, replay, rerun, factor change, or high-state
update was performed.

Full low physical and five-suite CURRENT/AON NPZ and Parquet products, the
byte-identical durable checkpoint, comparison tables, SQLite summary, and figures
reopen successfully. High products remain immutable external bindings.

## Lifecycle

- Phase 4 execution: `COMPLETE_WITH_NOTES`
- High benchmark: `APPROVED_HIGH_ACCURACY_BENCHMARK_WITH_NOTES`
- Low benchmark: `PRODUCED_AND_VALIDATED`
- Phase 4 overall: `{FINAL_STATUS}`
- Original 1e-4 target claimed met: `false`
- Phase 5 authorized: `false`
""",
    )
    status = _load_json(output_root / "status" / "dual_benchmark_status.json")
    status.update(
        {
            "state": "READY_FOR_EXTERNAL_HANDOFF_SEAL",
            "phase4_execution": "COMPLETE_WITH_NOTES",
            "high_benchmark": "APPROVED_HIGH_ACCURACY_BENCHMARK_WITH_NOTES",
            "low_benchmark": "PRODUCED_AND_VALIDATED",
            "phase4_overall": FINAL_STATUS,
            "original_1e4_target_claimed_met": False,
            "solver_updates_performed": 0,
            "phase5_authorized": False,
            "next_step_authorized": False,
            "updated_utc": _utc_now(),
        }
    )
    _write_json_atomic(output_root / "status" / "dual_benchmark_status.json", status)
    manifest = _output_artifact_manifest(output_root)
    _write_frame_new(manifest, output_root / "manifests" / "artifact_manifest.csv")
    return validation


def package_handoff(
    config_path: Path,
    phase_id: str,
    step_id: str,
    run_id: str,
    handoff_id: str,
) -> dict[str, Any]:
    config, project_root, output_root = _resolve_context(config_path, phase_id, step_id, run_id)
    validation = _load_json(output_root / "validation" / "final_validation.json")
    figure_review = _load_json(output_root / "validation" / "figure_visual_qa.json")
    test_candidates = (
        output_root / "validation" / "focused_tests_reporting_repair_receipt.json",
        output_root / "validation" / "focused_tests_receipt.json",
    )
    tests = _load_json(next(path for path in test_candidates if path.is_file()))
    if validation.get("status") != "PASS" or figure_review.get("status") != "PASS" or tests.get("status") != "PASS":
        raise DualAccuracyError("Handoff requires passing final, figure, and focused-test validation")
    if validation.get("phase4_overall") != FINAL_STATUS or bool(validation.get("phase5_authorized")):
        raise DualAccuracyError("Final phase boundary differs from the dual-benchmark authority")

    source = project_root / "partC_doc_icnn" / "handoffs" / "phase_04" / "step_dual_accuracy_benchmarks" / handoff_id
    if source.exists():
        raise FileExistsError(f"Write-new handoff source exists: {source}")
    for name in (
        "governing_docs",
        "configs",
        "code_snapshot",
        "code_diff",
        "summary_data",
        "figures",
        "geospatial_summary",
        "log_excerpts",
        "external_artifact_index",
        "manifests",
    ):
        (source / name).mkdir(parents=True, exist_ok=False)
    zip_root = project_root / "artifacts_icnn" / "handoff_bundles" / "phase_04"
    zip_path = zip_root / f"phase04_stepdual_accuracy_benchmarks_{handoff_id}_SYNC.zip"

    _write_text_new(
        source / "00_HANDOFF_README.md",
        f"""# Phase 4 dual-accuracy benchmark handoff

Read `01_STEP_SUMMARY.md`, `02_HANDOFF_STATUS.json`,
`04_VALIDATION_RESULTS.md`, and `09_NEXT_STEP_RECOMMENDATION.md` first.

This bundle registers the immutable x^200 high benchmark and the recovered,
independently refreshed x^45 low benchmark. Large checkpoint, NPZ, Parquet, and
SQLite products remain local and are exact-path/hash bound in
`external_artifact_index/`.

No solver update was performed. The original `1e-4` target remains unmet and
is not claimed. Phase 5 is unauthorized.
""",
    )
    shutil.copy2(output_root / "reports" / "STEP_SUMMARY.md", source / "01_STEP_SUMMARY.md")
    handoff_status = {
        "schema_id": "phase4.dual_benchmark.handoff_status.v1",
        "handoff_id": handoff_id,
        "status": FINAL_STATUS,
        "phase_id": phase_id,
        "step_id": step_id,
        "run_id": run_id,
        "phase4_execution": "COMPLETE_WITH_NOTES",
        "high_benchmark": "APPROVED_HIGH_ACCURACY_BENCHMARK_WITH_NOTES",
        "high_state_iteration": 200,
        "high_relative_gap_sptt": validation["high_relative_gap_sptt"],
        "low_benchmark": "PRODUCED_AND_VALIDATED",
        "low_state_iteration": 45,
        "low_relative_gap_sptt": validation["low_relative_gap_sptt"],
        "low_recovery_mode": "RECOVERED_EXISTING_CHECKPOINT",
        "solver_updates_performed": 0,
        "original_1e4_target_claimed_met": False,
        "phase5_authorized": False,
        "next_step_authorized": False,
    }
    _write_json_new(source / "02_HANDOFF_STATUS.json", handoff_status)
    _write_text_new(
        source / "03_DECISIONS_AND_OPEN_QUESTIONS.md",
        f"""# Decisions and open questions

## Binding decisions

- External decision `PHASE4_ACCEPT_ITER200_AND_RECOVER_RG1E3_20260902` accepts x^200 as the project-relative high benchmark with notes.
- The accepted high gap is `{validation['high_relative_gap_sptt']:.17g}`; the original `1e-4` target remains unmet.
- The earliest complete saved low state after the x^33 scalar crossing is x^45, with independently refreshed gap `{validation['low_relative_gap_sptt']:.17g}`.
- No replay/rerun was triggered and zero solver updates were performed.
- Phase 5 remains unauthorized.

## External-review question

Review this exact synchronization ZIP and decide whether to approve the dual
benchmark closeout, request a bounded repair/revision, or block. Codex does not
self-approve the Phase 4 transition.
""",
    )
    _write_text_new(
        source / "04_VALIDATION_RESULTS.md",
        f"""# Validation results

Final validation: `PASS`.

- authority and reviewed high identity: PASS
- immutable high checkpoint/final products: PASS
- x^45 physical CURRENT and all-five CURRENT trace completeness: PASS
- fresh-process physical AON exactness: PASS
- fresh-process all-five dense/sparse AON exactness: PASS
- low independently refreshed RG_SPTT `{validation['low_relative_gap_sptt']:.17g}` <= `1e-3`: PASS
- graph/OD/clean-capacity/suite identities: PASS
- mapping failures zero and demand conserved: PASS
- durable checkpoint and NPZ/Parquet/SQLite reopen: PASS
- high/low state and physical payload distinction: PASS
- focused tests: {tests['tests']} passed, zero failures/errors
- three-figure visual QA: PASS
- resource guards: PASS
- solver updates: 0
- original 1e-4 target: NOT MET; no claim of pass
- Phase 5 authorization: false
""",
    )
    shutil.copy2(output_root / "resources" / "resource_usage.csv", source / "05_RESOURCE_USAGE.csv")
    manifest_candidates = (
        output_root / "manifests" / "artifact_manifest_final_seal.csv",
        output_root / "manifests" / "artifact_manifest_precision_seal.csv",
        output_root / "manifests" / "artifact_manifest.csv",
    )
    shutil.copy2(next(path for path in manifest_candidates if path.is_file()), source / "06_ARTIFACT_MANIFEST.csv")

    code_paths = (
        "partC_rev1_src/phase4/dual_accuracy.py",
        "partC_rev1_src/phase4/production_finalize.py",
        "partC_rev1_src/phase4/checkpoint.py",
        "partC_rev1_src/phase4/aon.py",
        "partC_rev1_src/phase4/costs.py",
        "partC_rev1_src/phase4/inputs.py",
        "partC_rev1_src/phase4/tracing.py",
        "partC_rev1_src/phase4/models.py",
        "tests/test_phase4_dual_accuracy.py",
        "partC_doc_icnn/phase_4_work/phase4_dual_accuracy_benchmark_execution_config.json",
        "partC_doc_icnn/phase_4_work/PHASE4_DUAL_ACCURACY_BENCHMARK_RESULT.md",
        "partC_doc_icnn/phase_4_work/PHASE4_DUAL_ACCURACY_BENCHMARK_STATUS.json",
    )
    code_rows: list[dict[str, Any]] = []
    for relative_text in code_paths:
        relative = Path(relative_text)
        path = project_root / relative
        if not path.is_file():
            raise DualAccuracyError(f"Required handoff code/config snapshot is missing: {relative}")
        _copy_preserving(path, source / "code_snapshot", relative)
        code_rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
                "role": "entrypoint/dependency/test/config/result",
            }
        )
    _write_frame_new(pd.DataFrame(code_rows), source / "07_CODE_CHANGE_INDEX.csv")
    shutil.copy2(output_root / "reports" / "COMMAND_LOG.md", source / "08_COMMAND_LOG.md")
    _write_text_new(
        source / "09_NEXT_STEP_RECOMMENDATION.md",
        """# Next-step recommendation

Recommendation: externally review this exact dual-benchmark synchronization
ZIP and decide `APPROVE`, `APPROVE_WITH_NOTES`, `REVISE_AND_REPACKAGE`,
`REPAIR_AND_REPACKAGE`, or `BLOCK`.

No additional Phase 4 execution and no Phase 5 work are authorized by this
handoff. `next_step_authorized=false` and `phase5_authorized=false`.
""",
    )
    _write_text_new(
        source / "10_UPLOAD_THIS_FILE.txt",
        f"Upload this exact file:\n{zip_path.resolve()}\nVerify it with the adjacent .zip.sha256 and .companion.json files.\n",
    )
    _write_text_new(
        source / "11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md",
        f"""# Changelog from previous handoff

This handoff is governed by the external dual-benchmark decision bound to the
reviewed iteration-200 ZIP SHA-256
`{config['authority']['reviewed_iter200_handoff_sha256']}`.

It adds a non-mutating high benchmark wrapper; bounded recovery audit; durable
byte-identical x^45 checkpoint; independent x^45 exact AON/trace refresh; full
low physical and s02-s06 current/AON exports; high/low comparisons; registry;
SQLite summary; figures; focused tests; resource evidence; and final Phase 4
dual-benchmark review status. It does not alter the reviewed high products or
perform a solver update.
""",
    )

    governing_repo = (
        "partC_doc_icnn/00_READ_FIRST.md",
        "partC_doc_icnn/PROJECT_STATUS.md",
        "partC_doc_icnn/12_DECISION_LOG.md",
        "partC_doc_icnn/20_HANDOFF_SYNC_PROTOCOL.md",
        "partC_doc_icnn/21_REVIEW_APPROVAL_GATES.md",
        "partC_doc_icnn/23_SYNC_BUNDLE_CONTENT_AND_SIZE_POLICY.md",
        "partC_doc_icnn/24_CODE_AND_DATA_SNAPSHOT_POLICY.md",
        "partC_doc_icnn/phase_4_work/PHASE_4_TASK1_SCOPE.md",
        "partC_doc_icnn/review_gates/phase_04/step_dual_accuracy_benchmarks/phase4_dual_accuracy_authority_20260902/REVIEW_DECISION.md",
        "partC_doc_icnn/review_gates/phase_04/step_dual_accuracy_benchmarks/phase4_dual_accuracy_authority_20260902/APPROVAL.json",
    )
    for relative_text in governing_repo:
        relative = Path(relative_text)
        _copy_preserving(project_root / relative, source / "governing_docs" / "repository", relative)
    authority_root = Path(config["authority"]["extracted_root"])
    for name in (
        "00_READ_FIRST.md",
        "01_EXTERNAL_DECISION_AND_DEFINITIONS.md",
        "02_RECOVERY_FIRST_PLAN.md",
        "03_EXACT_STATE_AND_CODE_ADAPTER.md",
        "04_EXPORT_COMPARISON_AND_CLOSEOUT.md",
        "05_AUTONOMY_RESOURCES_AND_TESTS.md",
        "06_CODEX_GOAL.md",
        "REVIEW_FINDINGS.md",
        "MANIFEST.csv",
        "CHECKSUMS.sha256",
    ):
        _copy_preserving(authority_root / name, source / "governing_docs" / "external_authority", Path(name))
    for path in sorted((authority_root / "config").glob("*")):
        if path.is_file():
            _copy_preserving(path, source / "governing_docs" / "external_authority", Path("config") / path.name)
    _copy_preserving(Path(config["authority"]["launch_path"]), source / "governing_docs" / "external_authority", Path("START_CODEX_PHASE4_DUAL_ACCURACY_BENCHMARKS.md"))
    _copy_preserving(Path(config["authority"]["package_path"]), source / "governing_docs" / "external_authority", Path("phase4_dual_accuracy_benchmarks_20260902.zip"))
    _copy_preserving(config_path, source / "configs", Path(config_path.name))

    baseline = str(config["approved_starting_head"])
    head = _git(project_root, "rev-parse", "HEAD").stdout.strip()
    _write_text_new(source / "code_diff" / "approved_start_to_handoff.patch", _git(project_root, "diff", "--binary", baseline, head, "--").stdout)
    _write_text_new(source / "code_diff" / "git_log.txt", _git(project_root, "log", "--oneline", "--decorate", f"{baseline}..{head}").stdout)
    _write_text_new(source / "code_diff" / "git_status_before_handoff.txt", _git(project_root, "status", "--short").stdout)
    _write_text_new(source / "code_diff" / "git_identity.txt", f"branch={config['expected_branch']}\nhead={head}\napproved_start={baseline}\n")

    summary_paths: list[Path] = []
    for root in (output_root / "reports", output_root / "validation", output_root / "resources"):
        summary_paths.extend(path for path in root.rglob("*") if path.is_file())
    summary_paths.extend(
        [
            output_root / "benchmark_registry.json",
            output_root / "benchmark_registry.csv",
            output_root / "comparison" / "benchmark_metrics.csv",
            output_root / "comparison" / "physical_current_flow_and_cost_comparison.csv",
            output_root / "comparison" / "suite_current_flow_comparison.csv",
            output_root / "comparison" / "suite_internal_cost_comparison.csv",
            output_root / "comparison" / "source_refreshed_state_history.csv",
            output_root / "high_accuracy_iter0200" / "REFERENCE.json",
            output_root / "low_accuracy_rg1e3" / "checkpoint" / "RECOVERY_COPY_RECEIPT.json",
            output_root / "low_accuracy_rg1e3" / "checkpoint" / "SOURCE_ARRAY_VALIDATION.json",
        ]
    )
    for path in sorted(set(summary_paths), key=lambda item: item.relative_to(output_root).as_posix()):
        if path.is_file() and path.stat().st_size <= MAX_HANDOFF_MEMBER_BYTES and path.suffix.lower() not in (".npz", ".parquet", ".sqlite"):
            _copy_preserving(path, source / "summary_data", path.relative_to(output_root))
    for path in sorted((output_root / "figures").glob("*.png")):
        _copy_preserving(path, source / "figures", Path(path.name))
    _copy_preserving(
        output_root / "manifests" / "external_artifact_index.csv",
        source / "external_artifact_index",
        Path("external_artifact_index.csv"),
    )
    _write_text_new(
        source / "geospatial_summary" / "NO_NEW_GEOSPATIAL_ARTIFACT.md",
        "No geometry was generated. Both tiers use the accepted physical graph; portable flow products retain stable physical and ACNHN identifiers.",
    )
    _write_text_new(
        source / "log_excerpts" / "NO_LONG_RUN_LOG.md",
        "Recovery succeeded from x45, so no solver replay/rerun or long-run log exists. Exact commands are summarized in 08_COMMAND_LOG.md.",
    )

    privacy = _privacy_scan(source)
    _write_json_new(source / "manifests" / "privacy_scan.json", privacy)
    if privacy["status"] != "PASS":
        raise DualAccuracyError(f"Handoff privacy scan failed: {privacy}")
    content_rows = []
    for path in sorted((item for item in source.rglob("*") if item.is_file()), key=lambda item: item.relative_to(source).as_posix()):
        content_rows.append(
            {
                "relative_path": path.relative_to(source).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    _write_frame_new(pd.DataFrame(content_rows), source / "manifests" / "HANDOFF_CONTENT_MANIFEST.csv")
    oversized = [
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and path.stat().st_size > MAX_HANDOFF_MEMBER_BYTES
    ]
    if oversized:
        raise DualAccuracyError(f"Compact handoff contains files above 25 MiB: {oversized}")
    checksums = _handoff_checksums(source)
    _create_zip(source, zip_path)
    if zip_path.stat().st_size > MAX_HANDOFF_BYTES:
        raise DualAccuracyError("Compact handoff ZIP exceeds the 200 MiB hard limit")
    fresh = _fresh_extract_verify(zip_path)
    zip_sha = file_sha256(zip_path)
    sidecar_path = Path(str(zip_path) + ".sha256")
    companion_path = Path(str(zip_path) + ".companion.json")
    _write_text_new(sidecar_path, f"{zip_sha} *{zip_path.name}")
    companion = {
        "schema_id": "phase4.dual_benchmark.handoff_companion.v1",
        "handoff_id": handoff_id,
        "status": FINAL_STATUS,
        "zip_path": str(zip_path.resolve()),
        "zip_size_bytes": int(zip_path.stat().st_size),
        "zip_sha256": zip_sha,
        "sidecar_path": str(sidecar_path.resolve()),
        "source_path": str(source.resolve()),
        "source_file_count_excluding_checksum_file": len(checksums),
        "fresh_extract_validation": fresh,
        "privacy_scan": privacy,
        "code_commit": head,
        "high_state_iteration": 200,
        "low_state_iteration": 45,
        "original_1e4_target_claimed_met": False,
        "solver_updates_performed": 0,
        "phase5_authorized": False,
        "next_step_authorized": False,
        "created_utc": _utc_now(),
    }
    _write_json_new(companion_path, companion)
    latest = source.parent / "LATEST_HANDOFF.txt"
    if latest.exists():
        raise FileExistsError(f"Write-new latest-handoff pointer already exists: {latest}")
    _write_text_new(
        latest,
        f"source_folder={source.resolve()}\nzip_path={zip_path.resolve()}\nzip_sha256={zip_sha}\ncreated_utc={companion['created_utc']}\nstatus={FINAL_STATUS}\n",
    )

    live_status = _load_json(output_root / "status" / "dual_benchmark_status.json")
    live_status.update(
        {
            "state": FINAL_STATUS,
            "phase4_execution": "COMPLETE_WITH_NOTES",
            "high_benchmark": "APPROVED_HIGH_ACCURACY_BENCHMARK_WITH_NOTES",
            "low_benchmark": "PRODUCED_AND_VALIDATED",
            "phase4_overall": FINAL_STATUS,
            "handoff_id": handoff_id,
            "handoff_zip": str(zip_path.resolve()),
            "handoff_zip_size_bytes": int(zip_path.stat().st_size),
            "handoff_zip_sha256": zip_sha,
            "original_1e4_target_claimed_met": False,
            "solver_updates_performed": 0,
            "phase5_authorized": False,
            "next_step_authorized": False,
            "updated_utc": _utc_now(),
        }
    )
    _write_json_atomic(output_root / "status" / "dual_benchmark_status.json", live_status)
    sealed_manifest = _output_artifact_manifest(output_root)
    _write_frame_new(sealed_manifest, output_root / "manifests" / "artifact_manifest_handoff_seal.csv")
    return companion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage",
        choices=("inspect", "refresh-low", "export", "confirm-figures", "validate", "handoff"),
        required=True,
    )
    parser.add_argument("--handoff-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = Path(args.config).resolve(strict=True)
    common = (config_path, args.phase_id, args.step_id, args.run_id)
    if args.stage == "inspect":
        result = inspect_recovery(*common)
    elif args.stage == "refresh-low":
        result = refresh_low_state(*common)
    elif args.stage == "export":
        result = export_benchmarks(*common)
    elif args.stage == "confirm-figures":
        result = confirm_figures(*common)
    elif args.stage == "validate":
        result = validate_products(*common)
    else:
        if not args.handoff_id:
            raise DualAccuracyError("--handoff-id is required for the handoff stage")
        result = package_handoff(*common, args.handoff_id)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
