"""Write-new Phase 4 physical/trace checkpoints and deterministic restart state."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .models import AONTraceResult, SparseFlowVector, SuiteLinearTrace, SuiteTraceMap


TRACE_DENSE_FIELDS = (
    "bundle_flow",
    "bacn_internal_arc_flow",
    "bacn_internal_traversal_flow",
    "terminal_connector_flow",
    "racn_interface_flow",
    "repeated_region_entry_flow",
)


@dataclass(slots=True)
class Phase4RestartState:
    next_iteration: int
    physical_flow: np.ndarray
    suite_traces: dict[str, SuiteLinearTrace]
    previous_sight_flow: np.ndarray | None = None
    previous_sight_traces: dict[str, SuiteLinearTrace] | None = None
    previous_direction: np.ndarray | None = None
    previous_step_size: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trace_arrays(prefix: str, trace: SuiteLinearTrace, arrays: dict[str, np.ndarray]) -> None:
    safe_suite = trace.suite_id.replace("-", "_")
    stem = f"{prefix}__{safe_suite}__"
    for field_name in TRACE_DENSE_FIELDS:
        arrays[stem + field_name] = np.ascontiguousarray(
            np.asarray(getattr(trace, field_name), dtype=np.float64)
        )
    arrays[stem + "coarse_regional_pair_keys"] = trace.coarse_regional_pair_flow.keys
    arrays[stem + "coarse_regional_pair_values"] = trace.coarse_regional_pair_flow.values
    arrays[stem + "fine_regional_pair_keys"] = trace.fine_regional_pair_flow.keys
    arrays[stem + "fine_regional_pair_values"] = trace.fine_regional_pair_flow.values


def _read_trace(
    archive: Mapping[str, np.ndarray],
    prefix: str,
    trace_map: SuiteTraceMap,
) -> SuiteLinearTrace:
    safe_suite = trace_map.suite_id.replace("-", "_")
    stem = f"{prefix}__{safe_suite}__"
    trace = SuiteLinearTrace(
        suite_id=trace_map.suite_id,
        bundle_flow=np.asarray(archive[stem + "bundle_flow"], dtype=np.float64),
        coarse_regional_pair_flow=SparseFlowVector(
            np.asarray(archive[stem + "coarse_regional_pair_keys"], dtype=np.uint64),
            np.asarray(archive[stem + "coarse_regional_pair_values"], dtype=np.float64),
        ),
        fine_regional_pair_flow=SparseFlowVector(
            np.asarray(archive[stem + "fine_regional_pair_keys"], dtype=np.uint64),
            np.asarray(archive[stem + "fine_regional_pair_values"], dtype=np.float64),
        ),
        bacn_internal_arc_flow=np.asarray(
            archive[stem + "bacn_internal_arc_flow"], dtype=np.float64
        ),
        bacn_internal_traversal_flow=np.asarray(
            archive[stem + "bacn_internal_traversal_flow"], dtype=np.float64
        ),
        terminal_connector_flow=np.asarray(
            archive[stem + "terminal_connector_flow"], dtype=np.float64
        ),
        racn_interface_flow=np.asarray(archive[stem + "racn_interface_flow"], dtype=np.float64),
        repeated_region_entry_flow=np.asarray(
            archive[stem + "repeated_region_entry_flow"], dtype=np.float64
        ),
    )
    trace.validate(trace_map)
    return trace


def save_phase4_checkpoint(
    directory: str | Path,
    state: Phase4RestartState,
    *,
    trace_maps: Sequence[SuiteTraceMap],
    latest_aon: AONTraceResult,
    run_metadata: Mapping[str, Any],
    compressed: bool = False,
) -> dict[str, Any]:
    """Create one immutable checkpoint directory containing current and AON states."""

    started = time.perf_counter()
    root = Path(directory)
    if root.exists():
        raise FileExistsError(f"Checkpoint path already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    expected_ids = [item.suite_id for item in trace_maps]
    if list(state.suite_traces) != expected_ids:
        raise ValueError("Current checkpoint trace order differs from the frozen suite order")
    if list(latest_aon.suite_traces) != expected_ids:
        raise ValueError("AON checkpoint trace order differs from the frozen suite order")

    arrays: dict[str, np.ndarray] = {
        "current_physical_flow": np.ascontiguousarray(state.physical_flow, dtype=np.float64),
        "aon_physical_flow": np.ascontiguousarray(latest_aon.physical_flow, dtype=np.float64),
    }
    if state.previous_sight_flow is not None:
        arrays["previous_sight_physical_flow"] = np.ascontiguousarray(
            state.previous_sight_flow, dtype=np.float64
        )
    if state.previous_direction is not None:
        arrays["previous_direction"] = np.ascontiguousarray(
            state.previous_direction, dtype=np.float64
        )
    for suite_id in expected_ids:
        _trace_arrays("current", state.suite_traces[suite_id], arrays)
        _trace_arrays("aon", latest_aon.suite_traces[suite_id], arrays)
        if state.previous_sight_traces is not None:
            _trace_arrays("previous_sight", state.previous_sight_traces[suite_id], arrays)
        arrays[f"aon__{suite_id}__repeated_unique_od_occurrences"] = np.asarray(
            latest_aon.repeated_unique_od_occurrences[suite_id], dtype=np.int64
        )
        arrays[f"aon__{suite_id}__repeated_original_row_occurrences"] = np.asarray(
            latest_aon.repeated_original_row_occurrences[suite_id], dtype=np.int64
        )

    arrays_path = root / "checkpoint_arrays.npz"
    with arrays_path.open("xb") as stream:
        if compressed:
            np.savez_compressed(stream, **arrays)
        else:
            np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    metadata = {
        "schema_id": "phase4.physical_trace_checkpoint.v1",
        "next_iteration": int(state.next_iteration),
        "suite_ids": expected_ids,
        "has_previous_sight": state.previous_sight_flow is not None,
        "has_previous_direction": state.previous_direction is not None,
        "previous_step_size": state.previous_step_size,
        "history": state.history,
        "state_metadata": state.metadata,
        "latest_aon": {
            "sptt": float(latest_aon.sptt),
            "reachable_demand": float(latest_aon.reachable_demand),
            "unreachable_demand": float(latest_aon.unreachable_demand),
            "processed_origins": int(latest_aon.processed_origins),
            "total_origins": int(latest_aon.total_origins),
            "mapping_failure_counts": latest_aon.mapping_failure_counts,
            "timing": latest_aon.timing,
            "worker_count": int(latest_aon.worker_count),
            "metadata": latest_aon.metadata,
        },
        "run_metadata": dict(run_metadata),
        "array_names": sorted(arrays),
        "compressed": bool(compressed),
    }
    metadata_path = root / "checkpoint_metadata.json"
    with metadata_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {
        "schema_id": "phase4.checkpoint_receipt.v1",
        "directory": str(root.resolve()),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_size_bytes": int(arrays_path.stat().st_size),
        "arrays_sha256": _sha256(arrays_path),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_size_bytes": int(metadata_path.stat().st_size),
        "metadata_sha256": _sha256(metadata_path),
        "write_seconds": float(time.perf_counter() - started),
    }
    receipt_path = root / "checkpoint_receipt.json"
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return receipt


def load_phase4_checkpoint(
    directory: str | Path,
    *,
    trace_maps: Sequence[SuiteTraceMap],
) -> Phase4RestartState:
    root = Path(directory)
    metadata_path = root / "checkpoint_metadata.json"
    arrays_path = root / "checkpoint_arrays.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_id") != "phase4.physical_trace_checkpoint.v1":
        raise ValueError("Unsupported Phase 4 checkpoint schema")
    expected_ids = [item.suite_id for item in trace_maps]
    if metadata.get("suite_ids") != expected_ids:
        raise ValueError("Checkpoint suite scope differs from the requested frozen portfolio")
    receipt_path = root / "checkpoint_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if int(receipt["arrays_size_bytes"]) != int(arrays_path.stat().st_size):
            raise ValueError("Checkpoint array size differs from its receipt")
        if str(receipt["arrays_sha256"]) != _sha256(arrays_path):
            raise ValueError("Checkpoint array digest differs from its receipt")
        if str(receipt["metadata_sha256"]) != _sha256(metadata_path):
            raise ValueError("Checkpoint metadata digest differs from its receipt")

    with np.load(arrays_path, allow_pickle=False) as archive:
        files = set(archive.files)
        current = {item.suite_id: _read_trace(archive, "current", item) for item in trace_maps}
        previous = (
            {item.suite_id: _read_trace(archive, "previous_sight", item) for item in trace_maps}
            if bool(metadata.get("has_previous_sight"))
            else None
        )
        state = Phase4RestartState(
            next_iteration=int(metadata["next_iteration"]),
            physical_flow=np.asarray(archive["current_physical_flow"], dtype=np.float64),
            suite_traces=current,
            previous_sight_flow=(
                np.asarray(archive["previous_sight_physical_flow"], dtype=np.float64)
                if "previous_sight_physical_flow" in files
                else None
            ),
            previous_sight_traces=previous,
            previous_direction=(
                np.asarray(archive["previous_direction"], dtype=np.float64)
                if "previous_direction" in files
                else None
            ),
            previous_step_size=(
                float(metadata["previous_step_size"])
                if metadata.get("previous_step_size") is not None
                else None
            ),
            history=list(metadata.get("history", [])),
            metadata=dict(metadata.get("state_metadata", {})),
        )
    return state


def load_phase4_checkpoint_aon(
    directory: str | Path,
    *,
    trace_maps: Sequence[SuiteTraceMap],
) -> AONTraceResult:
    """Reopen the fully refreshed AON state stored beside a restart state."""

    root = Path(directory)
    metadata = json.loads((root / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    expected_ids = [item.suite_id for item in trace_maps]
    if metadata.get("schema_id") != "phase4.physical_trace_checkpoint.v1":
        raise ValueError("Unsupported Phase 4 checkpoint schema")
    if metadata.get("suite_ids") != expected_ids:
        raise ValueError("Checkpoint AON suite scope differs from the frozen portfolio")
    scalar = metadata["latest_aon"]
    with np.load(root / "checkpoint_arrays.npz", allow_pickle=False) as archive:
        traces = {item.suite_id: _read_trace(archive, "aon", item) for item in trace_maps}
        repeated_unique = {
            suite_id: np.asarray(
                archive[f"aon__{suite_id}__repeated_unique_od_occurrences"], dtype=np.int64
            )
            for suite_id in expected_ids
        }
        repeated_rows = {
            suite_id: np.asarray(
                archive[f"aon__{suite_id}__repeated_original_row_occurrences"], dtype=np.int64
            )
            for suite_id in expected_ids
        }
        result = AONTraceResult(
            physical_flow=np.asarray(archive["aon_physical_flow"], dtype=np.float64),
            sptt=float(scalar["sptt"]),
            reachable_demand=float(scalar["reachable_demand"]),
            unreachable_demand=float(scalar["unreachable_demand"]),
            processed_origins=int(scalar["processed_origins"]),
            total_origins=int(scalar["total_origins"]),
            suite_traces=traces,
            repeated_unique_od_occurrences=repeated_unique,
            repeated_original_row_occurrences=repeated_rows,
            mapping_failure_counts={
                str(key): int(value) for key, value in scalar["mapping_failure_counts"].items()
            },
            timing={str(key): float(value) for key, value in scalar["timing"].items()},
            worker_count=int(scalar["worker_count"]),
            metadata=dict(scalar["metadata"]),
        )
    result.validate(trace_maps)
    return result
