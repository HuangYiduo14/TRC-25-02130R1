"""Authorized Phase 4 prepared-capacity-factor refinement preflight.

The module is deliberately narrow: it sanitizes one frozen prepared-capacity
array, screens the authorized factor grid with the already selected
safeguarded-CFW algorithm, continues at most two plausible primary factors,
and performs an exact all-five trace/restart verification for a recommendation.
It does not run the 24-hour production assignment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import zipfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil

from partC_rev1_src.partC_rev1_shortest_paths import (
    _partC_rev1_numba_dijkstra,
    partC_rev1_arc_tie_rank,
)

from .aon import AONExecutionConfig, run_exact_aon
from .checkpoint import load_phase4_checkpoint
from .costs import phase4_bpr_cost
from .diagnostics import capacity_congestion_diagnostics
from .inputs import file_sha256, load_or_build_aggregated_od, load_phase4_graph
from .models import AONTraceResult, AggregatedOD, SuiteLinearTrace, SuiteTraceMap
from .solver import Phase4SolverConfig, Phase4SolverResult, solve_assignment
from .tracing import load_suite_trace_maps, pack_trace_maps


FINAL_STATUS = "WAITING_FOR_EXTERNAL_CAPACITY_FACTOR_REVIEW"
SELECTED_ALGORITHM = "SAFEGUARDED_CFW_WITH_IMMEDIATE_ORDINARY_FW_FALLBACK"
CAPACITY_EPSILON = 1e-12
QUANTILE_PROBABILITIES = np.asarray(
    [0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0], dtype=np.float64
)


class RefinementError(RuntimeError):
    """Raised when a frozen scientific or resource gate fails."""


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
        json.dump(
            payload,
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        stream.write("\n")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_frame_new(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False, lineterminator="\n")


def _array_payload_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _directory_size(root: Path) -> int:
    if not root.exists():
        return 0
    return int(sum(item.stat().st_size for item in root.rglob("*") if item.is_file()))


def _rss_bytes() -> int:
    process = psutil.Process()
    total = int(process.memory_info().rss)
    for child in process.children(recursive=True):
        try:
            total += int(child.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _git(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), "-c", "core.longpaths=true", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def _factor_key(factor: float) -> str:
    return f"{float(factor):.2f}".replace(".", "p")


def _weighted_quantile(value: np.ndarray, weight: np.ndarray, probability: float) -> float:
    values = np.asarray(value, dtype=np.float64)
    weights = np.asarray(weight, dtype=np.float64)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(keep):
        return float("nan")
    values = values[keep]
    weights = weights[keep]
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    return float(
        np.interp(
            float(probability) * cumulative[-1],
            cumulative,
            values,
            left=values[0],
            right=values[-1],
        )
    )


def _trace_equal(left: SuiteLinearTrace, right: SuiteLinearTrace) -> dict[str, Any]:
    dense_fields = (
        "bundle_flow",
        "bacn_internal_arc_flow",
        "bacn_internal_traversal_flow",
        "terminal_connector_flow",
        "racn_interface_flow",
        "repeated_region_entry_flow",
    )
    dense = {}
    for name in dense_fields:
        a = np.asarray(getattr(left, name))
        b = np.asarray(getattr(right, name))
        dense[name] = {
            "array_equal": bool(np.array_equal(a, b)),
            "max_abs_difference": float(np.max(np.abs(a - b))) if len(a) else 0.0,
        }
    sparse = {}
    for name in ("coarse_regional_pair_flow", "fine_regional_pair_flow"):
        a = getattr(left, name)
        b = getattr(right, name)
        sparse[name] = {
            "keys_equal": bool(np.array_equal(a.keys, b.keys)),
            "values_equal": bool(np.array_equal(a.values, b.values)),
            "max_abs_difference": (
                float(np.max(np.abs(a.values - b.values)))
                if np.array_equal(a.keys, b.keys) and len(a.values)
                else (0.0 if len(a.values) == len(b.values) == 0 else None)
            ),
        }
    passed = all(item["array_equal"] for item in dense.values()) and all(
        item["keys_equal"] and item["values_equal"] for item in sparse.values()
    )
    return {"pass": bool(passed), "dense": dense, "sparse": sparse}


def _aon_equal(
    left: AONTraceResult,
    right: AONTraceResult,
    trace_maps: Sequence[SuiteTraceMap],
) -> dict[str, Any]:
    trace = {
        item.suite_id: _trace_equal(
            left.suite_traces[item.suite_id], right.suite_traces[item.suite_id]
        )
        for item in trace_maps
    }
    result = {
        "physical_flow_array_equal": bool(
            np.array_equal(left.physical_flow, right.physical_flow)
        ),
        "physical_flow_max_abs_difference": float(
            np.max(np.abs(left.physical_flow - right.physical_flow))
        ),
        "sptt_equal": bool(left.sptt == right.sptt),
        "sptt_difference": float(right.sptt - left.sptt),
        "reachable_demand_equal": bool(left.reachable_demand == right.reachable_demand),
        "unreachable_demand_equal": bool(left.unreachable_demand == right.unreachable_demand),
        "trace_comparisons": trace,
    }
    result["pass"] = bool(
        result["physical_flow_array_equal"]
        and result["sptt_equal"]
        and result["reachable_demand_equal"]
        and result["unreachable_demand_equal"]
        and all(value["pass"] for value in trace.values())
    )
    return result


def sanitize_prepared_capacity(graph: Any) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    """Apply the frozen replace-only-invalid prepared-capacity rule once."""

    raw = np.ascontiguousarray(np.asarray(graph.capacity, dtype=np.float64))
    if raw.shape != graph.free_time.shape or raw.size == 0:
        raise RefinementError("Prepared-capacity array is empty or graph-misaligned")
    valid = np.isfinite(raw) & (raw > CAPACITY_EPSILON)
    if not np.any(valid):
        raise RefinementError("No finite prepared capacities exceed the frozen epsilon")
    q0001 = float(np.quantile(raw[valid], 0.001))
    floor = float(max(10.0 * CAPACITY_EPSILON, 1e-6 * q0001))
    clean = raw.copy()
    repair = ~valid
    clean[repair] = floor
    if np.any(~np.isfinite(clean)) or np.any(clean <= 0.0):
        raise RefinementError("Prepared-capacity sanitation did not produce a positive finite array")
    if not np.array_equal(clean[valid], raw[valid]):
        raise RefinementError("Sanitation changed a prepared capacity that was valid under the contract")

    reasons = np.full(raw.shape, "", dtype=object)
    reasons[np.isnan(raw)] = "nan"
    reasons[np.isposinf(raw)] = "positive_infinity"
    reasons[np.isneginf(raw)] = "negative_infinity"
    reasons[np.isfinite(raw) & (raw < 0.0)] = "negative"
    reasons[raw == 0.0] = "zero"
    reasons[np.isfinite(raw) & (raw > 0.0) & (raw <= CAPACITY_EPSILON)] = (
        "positive_at_or_below_epsilon"
    )
    repaired_idx = np.flatnonzero(repair)
    register_columns = [
        "arc_idx",
        "directed_arc_id",
        "link_uid",
        "tail_idx",
        "head_idx",
        "tail_node_uid",
        "head_node_uid",
        "capacity_multiplier_type",
        "region_code",
        "region_label",
        "raw_prepared_capacity",
        "clean_prepared_capacity",
        "repair_reason",
        "is_backbone_metadata",
        "is_connector_metadata",
        "used_in_selected_factor",
        "selected_factor_bottleneck",
        "selected_factor_sampled_path",
    ]
    rows: list[dict[str, Any]] = []
    for index in repaired_idx.tolist():
        label = graph.region_idx_to_label[int(graph.region_code[index])]
        type_value = str(graph.capacity_multiplier_type[index])
        rows.append(
            {
                "arc_idx": index,
                "directed_arc_id": str(graph.arc_idx_to_directed_arc_id[index]),
                "link_uid": str(graph.arc_idx_to_link_uid[index]),
                "tail_idx": int(graph.tail[index]),
                "head_idx": int(graph.head[index]),
                "tail_node_uid": str(graph.node_idx_to_uid[graph.tail[index]]),
                "head_node_uid": str(graph.node_idx_to_uid[graph.head[index]]),
                "capacity_multiplier_type": type_value,
                "region_code": int(graph.region_code[index]),
                "region_label": str(label),
                "raw_prepared_capacity": float(raw[index]),
                "clean_prepared_capacity": float(clean[index]),
                "repair_reason": str(reasons[index]),
                "is_backbone_metadata": "backbone" in type_value.lower(),
                "is_connector_metadata": "connector" in type_value.lower(),
                "used_in_selected_factor": None,
                "selected_factor_bottleneck": None,
                "selected_factor_sampled_path": None,
            }
        )
    register = pd.DataFrame(rows, columns=register_columns)

    def quantiles(value: np.ndarray) -> dict[str, float]:
        selected = value[np.isfinite(value)]
        result = np.quantile(selected, QUANTILE_PROBABILITIES)
        return {
            f"q{probability:.3f}": float(item)
            for probability, item in zip(QUANTILE_PROBABILITIES, result, strict=True)
        }

    audit = {
        "schema_id": "phase4.prepared_capacity_sanitation.v1",
        "epsilon": CAPACITY_EPSILON,
        "valid_definition": "finite and strictly greater than epsilon",
        "array_shape": list(raw.shape),
        "array_dtype": str(raw.dtype),
        "q0p001_valid": q0001,
        "deterministic_floor": floor,
        "floor_formula": "max(10*epsilon, 1e-6*q0.001(valid))",
        "counts": {
            "total": int(raw.size),
            "valid": int(np.count_nonzero(valid)),
            "nan": int(np.count_nonzero(np.isnan(raw))),
            "positive_infinity": int(np.count_nonzero(np.isposinf(raw))),
            "negative_infinity": int(np.count_nonzero(np.isneginf(raw))),
            "negative": int(np.count_nonzero(np.isfinite(raw) & (raw < 0.0))),
            "zero": int(np.count_nonzero(raw == 0.0)),
            "positive_at_or_below_epsilon": int(
                np.count_nonzero(np.isfinite(raw) & (raw > 0.0) & (raw <= CAPACITY_EPSILON))
            ),
            "repaired": int(np.count_nonzero(repair)),
        },
        "raw_finite_quantiles": quantiles(raw),
        "clean_quantiles": quantiles(clean),
        "raw_c_order_float64_payload_sha256": _array_payload_sha256(raw),
        "clean_c_order_float64_payload_sha256": _array_payload_sha256(clean),
        "raw_and_clean_payload_identical": bool(np.array_equal(raw, clean)),
        "sanitation_is_factor_selection_gate": False,
        "one_clean_immutable_array_for_all_factors": True,
    }
    clean.setflags(write=False)
    return replace(graph, capacity=clean), audit, register


def _trajectory_rows(result: Phase4SolverResult, *, run_scope: str) -> list[dict[str, Any]]:
    rows = []
    for record in result.histories:
        rows.append(
            {
                "run_scope": run_scope,
                "capacity_factor": float(result.capacity_multiplier),
                "update": int(record.evaluated_state_iteration),
                "record_kind": "refreshed_pre_update_state",
                "relative_gap_sptt": float(record.relative_gap_sptt),
                "relative_gap_tstt": float(record.relative_gap_tstt),
                "objective": float(record.objective_before),
                "tstt": float(record.tstt),
                "sptt": float(record.sptt),
                "accepted_step_to_update": int(record.iteration),
                "objective_after": float(record.objective_after),
                "step_size": float(record.step_size),
                "cfw_alpha": record.cfw_alpha,
                "cfw_reset": bool(record.cfw_reset),
                "fallback_reason": record.fallback_reason,
                "line_search_json": json.dumps(record.line_search, sort_keys=True),
                "trace_mixture_coefficients_json": json.dumps(
                    record.trace_mixture_coefficients, sort_keys=True
                ),
                "reachable_demand": float(record.reachable_demand),
                "unreachable_demand": float(record.unreachable_demand),
                "aon_seconds": float(record.aon_tree_load_trace_seconds),
                "iteration_wall_seconds": float(record.iteration_wall_seconds),
                "rss_bytes": int(record.rss_bytes),
            }
        )
    final_update = result.histories[-1].iteration if result.histories else 0
    rows.append(
        {
            "run_scope": run_scope,
            "capacity_factor": float(result.capacity_multiplier),
            "update": int(final_update),
            "record_kind": "final_refreshed_state",
            "relative_gap_sptt": float(result.final_metrics.relative_gap_sptt),
            "relative_gap_tstt": float(result.final_metrics.relative_gap_tstt),
            "objective": float(result.final_metrics.objective),
            "tstt": float(result.final_metrics.tstt),
            "sptt": float(result.final_metrics.sptt),
            "accepted_step_to_update": None,
            "objective_after": None,
            "step_size": None,
            "cfw_alpha": None,
            "cfw_reset": None,
            "fallback_reason": None,
            "line_search_json": None,
            "trace_mixture_coefficients_json": None,
            "reachable_demand": float(result.latest_aon.reachable_demand),
            "unreachable_demand": float(result.latest_aon.unreachable_demand),
            "aon_seconds": float(result.final_refresh_seconds),
            "iteration_wall_seconds": float(result.final_refresh_seconds),
            "rss_bytes": int(result.peak_rss_bytes),
        }
    )
    return rows


def _solver_summary(result: Phase4SolverResult, *, run_scope: str) -> dict[str, Any]:
    return {
        "schema_id": "phase4.capacity_factor_solver_result.v1",
        "run_scope": run_scope,
        "algorithm": SELECTED_ALGORITHM,
        "method": result.method,
        "capacity_factor": float(result.capacity_multiplier),
        "iteration_count_this_call": len(result.histories),
        "first_iteration_this_call": (
            int(result.histories[0].iteration) if result.histories else None
        ),
        "last_iteration_this_call": (
            int(result.histories[-1].iteration) if result.histories else None
        ),
        "final_metrics": asdict(result.final_metrics),
        "runtime_seconds": float(result.runtime_seconds),
        "initialization_seconds": float(result.initialization_seconds),
        "final_refresh_seconds": float(result.final_refresh_seconds),
        "peak_rss_bytes": int(result.peak_rss_bytes),
        "stop_reason": result.stop_reason,
        "flow_sum": float(result.flow.sum(dtype=np.float64)),
        "positive_flow_arc_count": int(np.count_nonzero(result.flow > 0.0)),
        "reachable_demand": float(result.latest_aon.reachable_demand),
        "unreachable_demand": float(result.latest_aon.unreachable_demand),
        "mapping_failure_counts": dict(result.latest_aon.mapping_failure_counts),
        "metadata": result.metadata,
        "checkpoint_receipts": result.checkpoint_receipts,
    }


def _save_result(
    root: Path,
    result: Phase4SolverResult,
    *,
    run_scope: str,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    summary = _solver_summary(result, run_scope=run_scope)
    _write_json_new(root / "solver_summary.json", summary)
    trajectory = pd.DataFrame(_trajectory_rows(result, run_scope=run_scope))
    _write_frame_new(trajectory, root / "trajectory.parquet")
    arrays = root / "final_physical_state.npz"
    with arrays.open("xb") as stream:
        np.savez(
            stream,
            physical_flow=np.ascontiguousarray(result.flow),
            physical_cost=np.ascontiguousarray(result.cost),
            used_capacity=np.ascontiguousarray(result.used_capacity),
        )
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {
        "path": str(arrays.resolve()),
        "size_bytes": int(arrays.stat().st_size),
        "sha256": file_sha256(arrays),
        "arrays": {
            "physical_flow": {"shape": list(result.flow.shape), "dtype": str(result.flow.dtype)},
            "physical_cost": {"shape": list(result.cost.shape), "dtype": str(result.cost.dtype)},
            "used_capacity": {
                "shape": list(result.used_capacity.shape),
                "dtype": str(result.used_capacity.dtype),
            },
        },
    }
    _write_json_new(root / "final_physical_state_receipt.json", receipt)
    return {"summary": summary, "trajectory": trajectory, "array_receipt": receipt}


def _reconstruct_path(
    source: int,
    target: int,
    predecessor_node: np.ndarray,
    predecessor_arc: np.ndarray,
) -> np.ndarray:
    if source == target:
        return np.empty(0, dtype=np.int64)
    current = int(target)
    reversed_arcs: list[int] = []
    for _ in range(len(predecessor_node) + 1):
        arc = int(predecessor_arc[current])
        parent = int(predecessor_node[current])
        if arc < 0 or parent < 0:
            return np.empty(0, dtype=np.int64)
        reversed_arcs.append(arc)
        current = parent
        if current == source:
            return np.asarray(reversed_arcs[::-1], dtype=np.int64)
    raise RefinementError("Sampled path predecessor chain contains a cycle")


def _od_demand(od: AggregatedOD, origin: int, destination: int) -> float:
    location = int(np.searchsorted(od.origin_group_node_idx, origin))
    if location >= len(od.origin_group_node_idx) or int(od.origin_group_node_idx[location]) != origin:
        raise RefinementError(f"Sampled origin {origin} is absent from accepted OD")
    begin, end = int(od.origin_offsets[location]), int(od.origin_offsets[location + 1])
    local = int(np.searchsorted(od.destination_node_idx[begin:end], destination))
    index = begin + local
    if index >= end or int(od.destination_node_idx[index]) != destination:
        raise RefinementError(f"Sampled OD ({origin}, {destination}) is absent from accepted OD")
    return float(od.demand[index])


def sampled_route_diagnostics(
    graph: Any,
    od: AggregatedOD,
    final_cost: np.ndarray,
    *,
    sample_path: Path,
    arc_tie_rank: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare the frozen 200 free-flow paths with exact final-cost paths."""

    started = time.perf_counter()
    sample = pd.read_parquet(sample_path)
    required = {
        "sample_od_id",
        "origin_idx",
        "destination_idx",
        "physical_fftt_s",
        "physical_arc_count",
        "arc_indices_json",
        "arc_ids_sha256",
    }
    if set(sample.columns) != required or len(sample) != 200:
        raise RefinementError("Frozen sampled-path table differs from its accepted 200-row schema")
    sample = sample.sort_values(
        ["origin_idx", "destination_idx", "sample_od_id"], kind="mergesort"
    ).reset_index(drop=True)
    if sample.duplicated(["origin_idx", "destination_idx"]).any():
        raise RefinementError("Frozen sampled-path table has duplicate physical endpoints")
    baseline_paths = [
        np.asarray(json.loads(value), dtype=np.int64) for value in sample["arc_indices_json"]
    ]
    rows: list[dict[str, Any]] = []
    cost = np.ascontiguousarray(np.asarray(final_cost, dtype=np.float64))
    for origin, group in sample.groupby("origin_idx", sort=True):
        dist, predecessor_node, predecessor_arc, _settled, _count = _partC_rev1_numba_dijkstra(
            len(graph.node_idx_to_uid),
            np.ascontiguousarray(graph.csr_indptr, dtype=np.int64),
            np.ascontiguousarray(graph.csr_arc_idx, dtype=np.int64),
            np.ascontiguousarray(graph.head, dtype=np.int64),
            cost,
            arc_tie_rank,
            int(origin),
            1e-12,
        )
        for row_index in group.index.tolist():
            source = int(sample.at[row_index, "origin_idx"])
            target = int(sample.at[row_index, "destination_idx"])
            selected = _reconstruct_path(
                source, target, predecessor_node, predecessor_arc
            )
            if not np.isfinite(dist[target]) or (source != target and len(selected) == 0):
                raise RefinementError(f"Accepted sampled OD ({source}, {target}) became unreachable")
            baseline = baseline_paths[row_index]
            selected_final_cost = float(cost[selected].sum(dtype=np.float64))
            baseline_final_cost = float(cost[baseline].sum(dtype=np.float64))
            shortest_distance = float(dist[target])
            reconciliation = abs(selected_final_cost - shortest_distance)
            tolerance = max(1e-9, 1e-10 * max(abs(shortest_distance), 1.0))
            if reconciliation > tolerance:
                raise RefinementError("Reconstructed sampled path does not reconcile to exact distance")
            switched = not np.array_equal(selected, baseline)
            strict_advantage = baseline_final_cost - selected_final_cost
            genuine_non_tie = bool(switched and strict_advantage > tolerance)
            baseline_set = np.unique(baseline)
            selected_set = np.unique(selected)
            intersection = int(np.intersect1d(baseline_set, selected_set).size)
            union = int(np.union1d(baseline_set, selected_set).size)
            overlap = float(intersection / union) if union else 1.0
            selected_arc_ids = graph.arc_idx_to_directed_arc_id[selected].astype(str)
            selected_digest = hashlib.sha256(
                "\n".join(selected_arc_ids.tolist()).encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    "sample_od_id": str(sample.at[row_index, "sample_od_id"]),
                    "origin_idx": source,
                    "destination_idx": target,
                    "accepted_od_demand": _od_demand(od, source, target),
                    "free_flow_arc_count": int(len(baseline)),
                    "selected_arc_count": int(len(selected)),
                    "route_switched": bool(switched),
                    "genuine_non_tie_route_switch": genuine_non_tie,
                    "arc_set_jaccard_overlap": overlap,
                    "common_unique_arc_count": intersection,
                    "free_flow_path_cost_under_final_weights": baseline_final_cost,
                    "selected_path_cost_under_final_weights": selected_final_cost,
                    "strict_cost_advantage": float(strict_advantage),
                    "non_tie_tolerance": tolerance,
                    "distance_reconciliation_abs": reconciliation,
                    "free_flow_arc_ids_sha256": str(sample.at[row_index, "arc_ids_sha256"]),
                    "selected_arc_ids_sha256": selected_digest,
                    "selected_arc_indices_json": json.dumps(selected.tolist()),
                }
            )
    frame = pd.DataFrame(rows).sort_values("sample_od_id", kind="mergesort").reset_index(drop=True)
    demand = frame["accepted_od_demand"].to_numpy(dtype=np.float64)
    switched = frame["route_switched"].to_numpy(dtype=bool)
    genuine = frame["genuine_non_tie_route_switch"].to_numpy(dtype=bool)
    overlap = frame["arc_set_jaccard_overlap"].to_numpy(dtype=np.float64)
    summary = {
        "schema_id": "phase4.sampled_route_switch.v1",
        "sample_path": str(sample_path.resolve()),
        "sample_sha256": file_sha256(sample_path),
        "sample_count": int(len(frame)),
        "unique_origin_count": int(frame["origin_idx"].nunique()),
        "route_switch_count": int(np.count_nonzero(switched)),
        "genuine_non_tie_route_switch_count": int(np.count_nonzero(genuine)),
        "has_genuine_non_tie_route_switch": bool(np.any(genuine)),
        "route_switch_sample_share": float(np.mean(switched)),
        "genuine_non_tie_sample_share": float(np.mean(genuine)),
        "route_switch_demand_weighted_share": float(
            demand[switched].sum(dtype=np.float64) / demand.sum(dtype=np.float64)
        ),
        "genuine_non_tie_demand_weighted_share": float(
            demand[genuine].sum(dtype=np.float64) / demand.sum(dtype=np.float64)
        ),
        "arc_set_jaccard_overlap_quantiles": {
            f"q{int(probability * 100):03d}": float(np.quantile(overlap, probability))
            for probability in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return frame, summary


def _capacity_gate_metrics(
    graph: Any,
    result: Phase4SolverResult,
) -> dict[str, Any]:
    congestion = capacity_congestion_diagnostics(
        graph, result.flow, result.cost, result.used_capacity
    )
    tti = np.divide(
        result.cost,
        graph.free_time,
        out=np.ones_like(result.cost),
        where=graph.free_time > 0.0,
    )
    positive = result.flow > 0.0
    return {
        "congestion": congestion,
        "flow_weighted_tti_p95": _weighted_quantile(tti, result.flow, 0.95),
        "positive_flow_link_tti_max": float(np.max(tti[positive])) if np.any(positive) else 1.0,
        "positive_flow_link_count_tti_above_2": int(np.count_nonzero(positive & (tti > 2.0))),
        "finite_flow": bool(np.all(np.isfinite(result.flow))),
        "finite_cost": bool(np.all(np.isfinite(result.cost))),
        "nonnegative_flow": bool(np.all(result.flow >= -1e-8)),
        "classified_demand": float(
            result.latest_aon.reachable_demand + result.latest_aon.unreachable_demand
        ),
        "objective_nonincreasing": bool(
            all(
                item.objective_after
                <= item.objective_before + 1e-9 * max(abs(item.objective_before), 1.0)
                for item in result.histories
            )
        ),
    }


class RefinementCampaign:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve(strict=True)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        expected = {
            "phase_id": "phase_04",
            "step_id": "capacity_factor_refinement_preflight",
            "algorithm": SELECTED_ALGORITHM,
            "worker_count": 4,
            "gap_denominator": "SPTT",
        }
        for key, value in expected.items():
            if self.config.get(key) != value:
                raise RefinementError(f"Execution config {key} differs from the frozen contract")
        if not self.config.get("tasks") or not self.config.get("expected_outputs") or not self.config.get(
            "acceptance_tests"
        ):
            raise RefinementError("Explicit task/output/test lists must be nonempty")
        if self.config["primary_capacity_factors"] != [0.3, 0.35, 0.4, 0.45, 0.5]:
            raise RefinementError("Primary factor grid differs from authority")
        if self.config["conditional_capacity_factors"] != [0.55, 0.6]:
            raise RefinementError("Conditional factor grid differs from authority")
        self.project_root = Path(self.config["project_root"]).resolve(strict=True)
        self.output_root = Path(self.config["output_root"])
        self.temp_root = Path(self.config["temp_root"])
        self.status_path = self.output_root / "status" / "campaign_status.json"
        self.event_path = self.output_root / "logs" / "campaign_events.jsonl"
        self.started_perf = 0.0
        self.started_utc = ""
        self.start_disk_free = 0
        self.peak_rss = 0

    def event(self, name: str, **payload: Any) -> None:
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp_utc": _utc_now(),
            "event": name,
            "phase_id": self.config["phase_id"],
            "step_id": self.config["step_id"],
            "run_id": self.config["run_id"],
            **payload,
        }
        with self.event_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False, default=_json_default) + "\n")
            stream.flush()

    def update_status(self, state: str, **payload: Any) -> None:
        prior = (
            json.loads(self.status_path.read_text(encoding="utf-8"))
            if self.status_path.exists()
            else {}
        )
        status = {
            **prior,
            "schema_id": "phase4.capacity_factor_refinement.status.v1",
            "phase_id": self.config["phase_id"],
            "step_id": self.config["step_id"],
            "run_id": self.config["run_id"],
            "state": state,
            "updated_utc": _utc_now(),
            "phase4_solver_tracer_preflight_status": "APPROVED_COMPLETE_WITH_NOTES",
            "selected_algorithm": SELECTED_ALGORITHM,
            "recommended_capacity_factor": "unresolved",
            "phase4_24h_production_assignment_authorized": False,
            "phase5_authorized": False,
            "next_step_authorized": False,
            **payload,
        }
        _write_json_atomic(self.status_path, status)

    def resource_guard(self, next_task: str) -> dict[str, Any]:
        elapsed = time.perf_counter() - self.started_perf
        resources = self.config["resources"]
        rss = _rss_bytes()
        self.peak_rss = max(self.peak_rss, rss)
        output_size = _directory_size(self.output_root)
        temp_size = _directory_size(self.temp_root)
        receipt = {
            "timestamp_utc": _utc_now(),
            "next_task": next_task,
            "elapsed_seconds": float(elapsed),
            "seconds_before_protected_reserve": float(
                resources["wall_clock_seconds"]
                - resources["protected_sealing_seconds"]
                - elapsed
            ),
            "rss_bytes": int(rss),
            "rss_hard_bytes": int(resources["process_tree_rss_bytes"]),
            "added_disk_bytes": int(output_size + temp_size),
            "added_disk_hard_bytes": int(resources["added_disk_bytes"]),
            "output_disk_free_bytes": int(shutil.disk_usage(self.output_root.anchor).free),
        }
        self.event("resource_guard", **receipt)
        if rss >= int(resources["process_tree_rss_bytes"]):
            raise MemoryError("Process-tree RSS reached the frozen 24-GiB hard limit")
        if output_size + temp_size >= int(resources["added_disk_bytes"]):
            raise OSError("Added disk reached the frozen 40-GiB hard limit")
        if receipt["seconds_before_protected_reserve"] <= 0.0:
            raise TimeoutError("Protected 900-second handoff reserve has begun")
        return receipt

    def _authority_validation(self) -> dict[str, Any]:
        authority = self.config["authority"]
        package = Path(authority["package_path"]).resolve(strict=True)
        launch = Path(authority["launch_path"]).resolve(strict=True)
        if package.stat().st_size != int(authority["package_size_bytes"]):
            raise RefinementError("Authority package size changed")
        if file_sha256(package) != authority["package_sha256"]:
            raise RefinementError("Authority package SHA-256 changed")
        if launch.stat().st_size != int(authority["launch_size_bytes"]):
            raise RefinementError("Launch document size changed")
        if file_sha256(launch) != authority["launch_sha256"]:
            raise RefinementError("Launch document SHA-256 changed")
        with zipfile.ZipFile(package) as archive:
            names = set(archive.namelist())
            checksums = archive.read("CHECKSUMS.sha256").decode("utf-8")
            verified = 0
            for line in checksums.splitlines():
                if not line.strip():
                    continue
                digest, relative = line.split("  ", 1)
                if relative not in names:
                    raise RefinementError(f"Authority checksum member missing: {relative}")
                actual = hashlib.sha256(archive.read(relative)).hexdigest()
                if actual != digest:
                    raise RefinementError(f"Authority checksum mismatch: {relative}")
                verified += 1
            if verified != int(authority["package_checksum_entries_verified"]):
                raise RefinementError("Authority checksum-entry count changed")
            nested_candidates = [
                name
                for name in names
                if PurePosixPath(name).name.startswith("phase04_step_task1_preflight_")
                and name.endswith("_SYNC.zip")
            ]
            source_candidates = [
                name for name in names if PurePosixPath(name).name == "partC_rev1_src(1).zip"
            ]
            if len(nested_candidates) != 1 or len(source_candidates) != 1:
                raise RefinementError("Reviewed preflight/source members are not unique in authority")
            nested = archive.read(nested_candidates[0])
            source = archive.read(source_candidates[0])
            if len(nested) != int(authority["reviewed_preflight_zip_size_bytes"]):
                raise RefinementError("Reviewed preflight size changed")
            if hashlib.sha256(nested).hexdigest() != authority["reviewed_preflight_zip_sha256"]:
                raise RefinementError("Reviewed preflight SHA-256 changed")
            if len(source) != int(authority["source_snapshot_size_bytes"]):
                raise RefinementError("Reviewed source snapshot size changed")
            if hashlib.sha256(source).hexdigest() != authority["source_snapshot_sha256"]:
                raise RefinementError("Reviewed source snapshot SHA-256 changed")
            manifest_hash = hashlib.sha256(archive.read("MANIFEST.csv")).hexdigest()
            if manifest_hash != authority["package_manifest_sha256"]:
                raise RefinementError("Authority MANIFEST.csv hash changed")
        return {
            "status": "PASS_WITH_DOCUMENTED_MINOR_CHECKSUM_CLOSURE_NOTE",
            "package_path": str(package),
            "package_size_bytes": int(package.stat().st_size),
            "package_sha256": file_sha256(package),
            "launch_path": str(launch),
            "launch_sha256": file_sha256(launch),
            "verified_checksum_entries": verified,
            "manifest_sha256": manifest_hash,
            "minor_resolution": authority["package_checksum_closure_note"],
            "numeric_authority_recoverable": True,
            "reviewed_preflight_status": "APPROVED_COMPLETE_WITH_NOTES",
        }

    def _prepare_output(self) -> dict[str, Any]:
        if self.output_root.exists() or self.temp_root.exists():
            raise FileExistsError("Refinement output/temp root must be write-new and unused")
        self.output_root.mkdir(parents=True)
        self.temp_root.mkdir(parents=True)
        for name in (
            "authority",
            "capacity",
            "config",
            "status",
            "logs",
            "validation",
            "timing",
            "runs",
            "checkpoints",
            "routes",
            "tables",
            "figures",
            "manifests",
            "reports",
        ):
            (self.output_root / name).mkdir()
        self.started_perf = time.perf_counter()
        self.started_utc = _utc_now()
        self.start_disk_free = int(shutil.disk_usage(self.output_root.anchor).free)
        self.update_status(
            "EXECUTING_STARTUP",
            campaign_started_utc=self.started_utc,
            completed_tasks=[],
            next_task="STARTUP-AUTHORITY",
        )
        persisted = self.output_root / "config" / self.config_path.name
        shutil.copy2(self.config_path, persisted)
        launch = Path(self.config["authority"]["launch_path"])
        shutil.copy2(launch, self.output_root / "config" / launch.name)
        package = Path(self.config["authority"]["package_path"])
        with zipfile.ZipFile(package) as archive:
            for info in archive.infolist():
                name = PurePosixPath(info.filename)
                if info.is_dir() or info.file_size > 1024 * 1024:
                    continue
                if name.suffix.lower() not in {".md", ".json", ".csv", ".sha256"}:
                    continue
                destination = self.output_root / "authority" / Path(*name.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(archive.read(info.filename))

        branch = _git(self.project_root, "branch", "--show-current").strip()
        head = _git(self.project_root, "rev-parse", "HEAD").strip()
        if branch != self.config["expected_branch"]:
            raise RefinementError(f"Expected branch {self.config['expected_branch']}, found {branch}")
        if head != self.config["reviewed_source_head"]:
            raise RefinementError("Live HEAD differs from the source head reviewed by this authority")
        git_evidence = {
            "branch": branch,
            "head": head,
            "status_short": _git(self.project_root, "status", "--short"),
            "diff_stat": _git(self.project_root, "diff", "--stat"),
            "diff": _git(self.project_root, "diff", "--"),
            "untracked_top_level_paths_preserved": [
                "artifacts_icnn/",
                "partC_doc_icnn/handoffs/phase_03/",
                "partC_doc_icnn/handoffs/phase_04/",
            ],
            "dirty_state_action": "preserved; no reset, clean, stash, delete, or overwrite",
        }
        _write_json_new(self.output_root / "validation" / "git_start_snapshot.json", git_evidence)
        (self.output_root / "validation" / "git_diff.patch").write_text(
            git_evidence["diff"], encoding="utf-8", newline="\n"
        )
        authority = self._authority_validation()
        _write_json_new(self.output_root / "validation" / "authority_validation.json", authority)
        self.event("startup_authority_passed", authority_status=authority["status"])
        return authority

    def _load_and_sanitize(self) -> tuple[Any, AggregatedOD, list[SuiteTraceMap], dict[str, Any]]:
        graph = load_phase4_graph(self.config)
        clean_graph, audit, register = sanitize_prepared_capacity(graph)
        raw_path = self.output_root / "capacity" / "raw_prepared_capacity.npy"
        clean_path = self.output_root / "capacity" / "clean_prepared_capacity.npy"
        with raw_path.open("xb") as stream:
            np.save(stream, np.ascontiguousarray(graph.capacity))
        with clean_path.open("xb") as stream:
            np.save(stream, np.ascontiguousarray(clean_graph.capacity))
        audit["raw_npy"] = {
            "path": str(raw_path.resolve()),
            "size_bytes": int(raw_path.stat().st_size),
            "file_sha256": file_sha256(raw_path),
        }
        audit["clean_npy"] = {
            "path": str(clean_path.resolve()),
            "size_bytes": int(clean_path.stat().st_size),
            "file_sha256": file_sha256(clean_path),
        }
        _write_json_new(self.output_root / "capacity" / "capacity_sanitation_audit.json", audit)
        audit_rows = []
        for domain in ("raw_finite_quantiles", "clean_quantiles"):
            for quantile, value in audit[domain].items():
                audit_rows.append(
                    {"array_domain": domain, "quantile": quantile, "value": value}
                )
        _write_frame_new(
            pd.DataFrame(audit_rows),
            self.output_root / "tables" / "prepared_capacity_quantiles.csv",
        )
        _write_frame_new(
            register,
            self.output_root / "tables" / "prepared_capacity_repair_register.csv",
        )
        _write_frame_new(
            pd.DataFrame(
                [
                    {
                        "raw_payload_sha256": audit[
                            "raw_c_order_float64_payload_sha256"
                        ],
                        "clean_payload_sha256": audit[
                            "clean_c_order_float64_payload_sha256"
                        ],
                        "raw_npy_sha256": audit["raw_npy"]["file_sha256"],
                        "clean_npy_sha256": audit["clean_npy"]["file_sha256"],
                        "repair_count": audit["counts"]["repaired"],
                        "floor": audit["deterministic_floor"],
                        "same_clean_array_all_factors": True,
                    }
                ]
            ),
            self.output_root / "tables" / "prepared_capacity_hashes.csv",
        )
        accepted_root = Path(self.config["input_versions"]["accepted_aggregated_od_root"])
        od = load_or_build_aggregated_od(
            self.config,
            output_root=self.output_root,
            existing_root=accepted_root,
        )
        trace_maps = load_suite_trace_maps(
            self.config,
            clean_graph,
            verify_directory_digests=True,
        )
        input_receipt = {
            "status": "PASS",
            "graph_identity": clean_graph.graph_hash,
            "graph_node_count": int(len(clean_graph.node_idx_to_uid)),
            "graph_arc_count": int(len(clean_graph.arc_idx_to_directed_arc_id)),
            "od_identity": od.od_hash,
            "aggregated_od_row_count": int(len(od.demand)),
            "original_od_row_count": int(od.original_row_count),
            "origin_count": int(len(od.origin_group_node_idx)),
            "total_demand": float(od.demand.sum(dtype=np.float64)),
            "accepted_aggregated_od_root": str(accepted_root.resolve()),
            "accepted_aggregated_od_metadata": od.metadata,
            "suite_directory_digests": {
                item.suite_id: item.directory_digest for item in trace_maps
            },
            "suite_ids": [item.suite_id for item in trace_maps],
            "s01_absent": all(item.suite_id != "s01" for item in trace_maps),
            "raw_capacity_payload_sha256": audit[
                "raw_c_order_float64_payload_sha256"
            ],
            "clean_capacity_payload_sha256": audit[
                "clean_c_order_float64_payload_sha256"
            ],
        }
        _write_json_new(self.output_root / "validation" / "input_integrity.json", input_receipt)
        self.event(
            "input_integrity_and_sanitation_passed",
            repair_count=audit["counts"]["repaired"],
            clean_capacity_sha256=audit["clean_c_order_float64_payload_sha256"],
        )
        return clean_graph, od, trace_maps, audit

    def _optimization_equivalence(
        self,
        graph: Any,
        od: AggregatedOD,
        trace_maps: Sequence[SuiteTraceMap],
    ) -> tuple[np.ndarray, Any]:
        self.resource_guard("EXACT-OPTIMIZATION")
        tie_rank_started = time.perf_counter()
        tie_rank = np.ascontiguousarray(partC_rev1_arc_tie_rank(graph), dtype=np.int64)
        tie_rank_build_seconds = time.perf_counter() - tie_rank_started
        pack_started = time.perf_counter()
        packed = pack_trace_maps(trace_maps)
        packed_build_seconds = time.perf_counter() - pack_started
        # Warm only the already-compiled national kernel specialization; this
        # one-origin result is never used as scientific evidence.
        run_exact_aon(
            graph,
            od,
            graph.free_time,
            trace_maps=trace_maps,
            config=AONExecutionConfig(worker_count=4, max_origins=1),
            arc_tie_rank=tie_rank,
            packed_trace_maps=packed,
        )
        before = run_exact_aon(
            graph,
            od,
            graph.free_time,
            trace_maps=trace_maps,
            config=AONExecutionConfig(worker_count=4, max_origins=128),
        )
        after = run_exact_aon(
            graph,
            od,
            graph.free_time,
            trace_maps=trace_maps,
            config=AONExecutionConfig(worker_count=4, max_origins=128),
            arc_tie_rank=tie_rank,
            packed_trace_maps=packed,
        )
        comparison = _aon_equal(before, after, trace_maps)
        if not comparison["pass"]:
            raise RefinementError("Cached immutable AON context changed exact numerical results")
        rows = []
        for label, result in (("before_uncached", before), ("after_cached", after)):
            rows.append(
                {
                    "timing_scope": label,
                    "origin_count": 128,
                    "worker_count": 4,
                    **result.timing,
                }
            )
        timing = {
            "schema_id": "phase4.aon_immutable_cache_optimization.v1",
            "status": "PASS",
            "implemented": [
                "cache deterministic physical-arc tie rank once per solver run",
                "pack immutable five-suite trace maps once per solver run",
                "separate setup, worker orchestration, reduction, and public-call timing",
            ],
            "rejected_or_deferred": [
                {
                    "optimization": "persistent Dijkstra workspaces/core structure rewrite",
                    "disposition": "DEFERRED",
                    "reason": "no measured need; frozen authority prohibits speculative structural risk",
                },
                {
                    "optimization": "regional-pair reduction-order change",
                    "disposition": "REJECTED",
                    "reason": "would risk frozen precision/reduction order",
                },
            ],
            "tie_rank_build_seconds": tie_rank_build_seconds,
            "packed_trace_map_build_seconds": packed_build_seconds,
            "before_public_call_seconds": before.timing["public_call_seconds"],
            "after_public_call_seconds": after.timing["public_call_seconds"],
            "measured_speedup": float(
                before.timing["public_call_seconds"] / after.timing["public_call_seconds"]
            ),
            "equivalence": comparison,
        }
        _write_json_new(self.output_root / "timing" / "aon_cache_optimization.json", timing)
        _write_frame_new(
            pd.DataFrame(rows),
            self.output_root / "tables" / "aon_before_after_timing.csv",
        )
        self.event(
            "exact_optimization_validated",
            speedup=timing["measured_speedup"],
            exact_equivalence=True,
        )
        return tie_rank, packed

    def _common_initialization(
        self,
        graph: Any,
        od: AggregatedOD,
        tie_rank: np.ndarray,
    ) -> AONTraceResult:
        self.resource_guard("COMMON-EXACT-FREE-FLOW-AON")
        initial = run_exact_aon(
            graph,
            od,
            graph.free_time,
            trace_maps=(),
            config=AONExecutionConfig(worker_count=4),
            arc_tie_rank=tie_rank,
            packed_trace_maps=pack_trace_maps(()),
        )
        path = self.output_root / "capacity" / "common_free_flow_aon.npz"
        with path.open("xb") as stream:
            np.savez(
                stream,
                physical_flow=np.ascontiguousarray(initial.physical_flow),
                sptt=np.asarray([initial.sptt], dtype=np.float64),
            )
        receipt = {
            "schema_id": "phase4.common_exact_free_flow_aon.v1",
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
            "physical_flow_payload_sha256": _array_payload_sha256(initial.physical_flow),
            "sptt": float(initial.sptt),
            "reachable_demand": float(initial.reachable_demand),
            "unreachable_demand": float(initial.unreachable_demand),
            "processed_origins": int(initial.processed_origins),
            "worker_count": int(initial.worker_count),
            "timing": initial.timing,
            "used_for_every_factor": True,
        }
        _write_json_new(self.output_root / "capacity" / "common_free_flow_aon_receipt.json", receipt)
        return initial

    def _screen_factor(
        self,
        graph: Any,
        od: AggregatedOD,
        initial: AONTraceResult,
        tie_rank: np.ndarray,
        factor: float,
        *,
        grid_scope: str,
    ) -> dict[str, Any]:
        key = _factor_key(factor)
        self.resource_guard(f"SHORT-SCREEN-{key}")
        self.event("factor_short_screen_started", capacity_factor=factor, grid_scope=grid_scope)
        checkpoint_root = self.output_root / "checkpoints" / f"factor_{key}" / "screen"
        result = solve_assignment(
            graph,
            od,
            trace_maps=(),
            config=Phase4SolverConfig(
                method="SAFEGUARDED_CFW",
                capacity_multiplier=float(factor),
                max_iterations=5,
                minimum_iterations=5,
                worker_count=4,
                checkpoint_directory=checkpoint_root,
                checkpoint_iterations=(3,),
                final_checkpoint=True,
                rss_hard_bytes=int(self.config["resources"]["process_tree_rss_bytes"]),
            ),
            initial_aon=initial,
            run_metadata={
                "run_id": self.config["run_id"],
                "grid_scope": grid_scope,
                "capacity_factor": factor,
                "clean_capacity_payload_sha256": _array_payload_sha256(graph.capacity),
            },
        )
        saved = _save_result(
            self.output_root / "runs" / f"factor_{key}" / "screen",
            result,
            run_scope=f"{grid_scope}_short_screen",
        )
        route_frame, route_summary = sampled_route_diagnostics(
            graph,
            od,
            result.cost,
            sample_path=Path(self.config["input_versions"]["frozen_path_sample"]),
            arc_tie_rank=tie_rank,
        )
        route_root = self.output_root / "routes" / f"factor_{key}" / "screen"
        _write_frame_new(route_frame, route_root / "sampled_route_switches.csv")
        _write_json_new(route_root / "sampled_route_summary.json", route_summary)
        metrics = _capacity_gate_metrics(graph, result)
        trajectory = saved["trajectory"]
        initial_gap = float(
            trajectory.loc[trajectory["update"].eq(0), "relative_gap_sptt"].iloc[0]
        )
        gap_after_three = float(
            trajectory.loc[trajectory["update"].eq(3), "relative_gap_sptt"].iloc[0]
        )
        demand_conserved = bool(
            np.isclose(
                metrics["classified_demand"],
                float(od.demand.sum(dtype=np.float64)),
                rtol=1e-12,
                atol=1e-8,
            )
        )
        gates = {
            "initial_relative_gap_sptt_above_0p10": initial_gap > 0.10,
            "relative_gap_sptt_after_3_updates_above_0p01": gap_after_three > 0.01,
            "flow_weighted_tti_p95_at_least_1p05": metrics[
                "flow_weighted_tti_p95"
            ]
            >= 1.05,
            "positive_flow_link_tti_above_2_exists": metrics[
                "positive_flow_link_count_tti_above_2"
            ]
            > 0,
            "genuine_non_tie_sampled_route_switch_exists": route_summary[
                "has_genuine_non_tie_route_switch"
            ],
            "finite_nonnegative_numerics": bool(
                metrics["finite_flow"]
                and metrics["finite_cost"]
                and metrics["nonnegative_flow"]
            ),
            "demand_conserved": demand_conserved,
            "accepted_step_objective_nonincreasing": metrics[
                "objective_nonincreasing"
            ],
        }
        plausible = bool(all(gates.values()))
        screen = {
            "schema_id": "phase4.capacity_factor_short_screen.v1",
            "capacity_factor": float(factor),
            "grid_scope": grid_scope,
            "fully_refreshed_updates": 5,
            "common_initial_aon_payload_sha256": _array_payload_sha256(
                initial.physical_flow
            ),
            "clean_capacity_payload_sha256": _array_payload_sha256(graph.capacity),
            "used_capacity_payload_sha256": _array_payload_sha256(result.used_capacity),
            "factor_applied_exactly_once": bool(
                np.array_equal(result.used_capacity, factor * graph.capacity)
            ),
            "initial_relative_gap_sptt": initial_gap,
            "relative_gap_sptt_after_3_updates": gap_after_three,
            "final_relative_gap_sptt_after_5_updates": float(
                result.final_metrics.relative_gap_sptt
            ),
            "flow_weighted_tti_p95": metrics["flow_weighted_tti_p95"],
            "positive_flow_link_tti_max": metrics["positive_flow_link_tti_max"],
            "positive_flow_link_count_tti_above_2": metrics[
                "positive_flow_link_count_tti_above_2"
            ],
            "route_summary": route_summary,
            "gates": gates,
            "plausible_for_continuation": plausible,
            "solver_summary": saved["summary"],
            "checkpoint_update_3": str(
                (checkpoint_root / "iteration_0003").resolve()
            ),
            "checkpoint_update_5": str(
                (checkpoint_root / "final_iteration_0005").resolve()
            ),
        }
        _write_json_new(
            self.output_root / "runs" / f"factor_{key}" / "short_screen_decision.json",
            screen,
        )
        self.event(
            "factor_short_screen_completed",
            capacity_factor=factor,
            plausible=plausible,
            final_relative_gap_sptt=screen["final_relative_gap_sptt_after_5_updates"],
            genuine_non_tie_route_switch_count=route_summary[
                "genuine_non_tie_route_switch_count"
            ],
        )
        return screen

    @staticmethod
    def _deduplicate_trajectory(frame: pd.DataFrame) -> pd.DataFrame:
        ordered = frame.copy()
        ordered["_kind_order"] = ordered["record_kind"].map(
            {"refreshed_pre_update_state": 0, "final_refreshed_state": 1}
        )
        ordered = ordered.sort_values(
            ["update", "_kind_order"], kind="mergesort"
        ).drop_duplicates("update", keep="last")
        return ordered.drop(columns="_kind_order").sort_values("update").reset_index(drop=True)

    def _continue_factor(
        self,
        graph: Any,
        od: AggregatedOD,
        tie_rank: np.ndarray,
        screen: Mapping[str, Any],
    ) -> dict[str, Any]:
        factor = float(screen["capacity_factor"])
        key = _factor_key(factor)
        trajectory_parts = [
            pd.read_parquet(
                self.output_root / "runs" / f"factor_{key}" / "screen" / "trajectory.parquet"
            )
        ]
        restart_directory = Path(screen["checkpoint_update_5"])
        final_state_root = self.output_root / "runs" / f"factor_{key}" / "screen"
        final_summary = dict(screen["solver_summary"])
        final_gap = float(screen["final_relative_gap_sptt_after_5_updates"])
        final_update = 5
        chunk_objectives_pass = True
        chunk_summaries: list[dict[str, Any]] = []
        final_result: Phase4SolverResult | None = None
        for target_update in range(10, 51, 5):
            if final_gap <= 0.01:
                break
            self.resource_guard(f"CONTINUE-{key}-TO-{target_update:02d}")
            restart = load_phase4_checkpoint(restart_directory, trace_maps=())
            checkpoint_root = (
                self.output_root
                / "checkpoints"
                / f"factor_{key}"
                / f"continuation_to_{target_update:04d}"
            )
            self.event(
                "factor_continuation_chunk_started",
                capacity_factor=factor,
                from_update=int(restart.next_iteration - 1),
                target_update=target_update,
            )
            result = solve_assignment(
                graph,
                od,
                trace_maps=(),
                config=Phase4SolverConfig(
                    method="SAFEGUARDED_CFW",
                    capacity_multiplier=factor,
                    max_iterations=target_update,
                    minimum_iterations=1,
                    worker_count=4,
                    checkpoint_directory=checkpoint_root,
                    final_checkpoint=True,
                    rss_hard_bytes=int(
                        self.config["resources"]["process_tree_rss_bytes"]
                    ),
                ),
                restart_state=restart,
                run_metadata={
                    "run_id": self.config["run_id"],
                    "capacity_factor": factor,
                    "continuation_target_update": target_update,
                    "clean_capacity_payload_sha256": _array_payload_sha256(graph.capacity),
                },
            )
            saved = _save_result(
                self.output_root
                / "runs"
                / f"factor_{key}"
                / f"continuation_to_{target_update:04d}",
                result,
                run_scope=f"qualification_continuation_to_{target_update:04d}",
            )
            trajectory_parts.append(saved["trajectory"])
            final_state_root = (
                self.output_root
                / "runs"
                / f"factor_{key}"
                / f"continuation_to_{target_update:04d}"
            )
            final_summary = saved["summary"]
            final_gap = float(result.final_metrics.relative_gap_sptt)
            final_update = target_update
            final_result = result
            chunk_pass = bool(
                all(
                    item.objective_after
                    <= item.objective_before
                    + 1e-9 * max(abs(item.objective_before), 1.0)
                    for item in result.histories
                )
            )
            chunk_objectives_pass &= chunk_pass
            chunk_summaries.append(
                {
                    "target_update": target_update,
                    "relative_gap_sptt": final_gap,
                    "runtime_seconds": result.runtime_seconds,
                    "objective_nonincreasing": chunk_pass,
                    "checkpoint": str(
                        (checkpoint_root / f"final_iteration_{target_update:04d}").resolve()
                    ),
                }
            )
            restart_directory = checkpoint_root / f"final_iteration_{target_update:04d}"
            self.event(
                "factor_continuation_chunk_completed",
                capacity_factor=factor,
                target_update=target_update,
                relative_gap_sptt=final_gap,
                observed_gap_target_met=final_gap <= 0.01,
            )

        combined = self._deduplicate_trajectory(pd.concat(trajectory_parts, ignore_index=True))
        _write_frame_new(
            combined,
            self.output_root
            / "tables"
            / f"factor_{key}_qualification_trajectory.csv",
        )
        below = combined.loc[combined["relative_gap_sptt"].le(0.01), "update"]
        first_below = int(below.iloc[0]) if len(below) else None
        tail_below = 0
        for value in combined["relative_gap_sptt"].to_numpy(dtype=np.float64)[::-1]:
            if value <= 0.01:
                tail_below += 1
            else:
                break

        if final_result is None:
            route_summary = dict(screen["route_summary"])
            final_metrics = {
                "flow_weighted_tti_p95": float(screen["flow_weighted_tti_p95"]),
                "positive_flow_link_tti_max": float(screen["positive_flow_link_tti_max"]),
                "positive_flow_link_count_tti_above_2": int(
                    screen["positive_flow_link_count_tti_above_2"]
                ),
                "finite_flow": bool(screen["gates"]["finite_nonnegative_numerics"]),
                "finite_cost": bool(screen["gates"]["finite_nonnegative_numerics"]),
                "nonnegative_flow": bool(screen["gates"]["finite_nonnegative_numerics"]),
                "objective_nonincreasing": bool(
                    screen["gates"]["accepted_step_objective_nonincreasing"]
                ),
                "classified_demand": float(od.demand.sum(dtype=np.float64)),
            }
        else:
            route_frame, route_summary = sampled_route_diagnostics(
                graph,
                od,
                final_result.cost,
                sample_path=Path(self.config["input_versions"]["frozen_path_sample"]),
                arc_tie_rank=tie_rank,
            )
            route_root = self.output_root / "routes" / f"factor_{key}" / "qualification"
            _write_frame_new(route_frame, route_root / "sampled_route_switches.csv")
            _write_json_new(route_root / "sampled_route_summary.json", route_summary)
            final_metrics = _capacity_gate_metrics(graph, final_result)

        demand_conserved = bool(
            np.isclose(
                final_metrics["classified_demand"],
                float(od.demand.sum(dtype=np.float64)),
                rtol=1e-12,
                atol=1e-8,
            )
        )
        final_gates = {
            "initial_relative_gap_sptt_above_0p10": bool(
                screen["gates"]["initial_relative_gap_sptt_above_0p10"]
            ),
            "relative_gap_sptt_after_3_updates_above_0p01": bool(
                screen["gates"]["relative_gap_sptt_after_3_updates_above_0p01"]
            ),
            "flow_weighted_tti_p95_at_least_1p05": final_metrics[
                "flow_weighted_tti_p95"
            ]
            >= 1.05,
            "positive_flow_link_tti_above_2_exists": final_metrics[
                "positive_flow_link_count_tti_above_2"
            ]
            > 0,
            "genuine_non_tie_sampled_route_switch_exists": route_summary[
                "has_genuine_non_tie_route_switch"
            ],
            "finite_nonnegative_numerics": bool(
                final_metrics["finite_flow"]
                and final_metrics["finite_cost"]
                and final_metrics["nonnegative_flow"]
            ),
            "demand_conserved": demand_conserved,
            "accepted_step_objective_nonincreasing": bool(
                screen["gates"]["accepted_step_objective_nonincreasing"]
                and chunk_objectives_pass
                and final_metrics["objective_nonincreasing"]
            ),
            "observed_refreshed_relative_gap_sptt_at_or_below_0p01_within_50_updates": bool(
                first_below is not None and first_below <= 50
            ),
        }
        qualified = bool(all(final_gates.values()))
        qualification = {
            "schema_id": "phase4.capacity_factor_qualification.v1",
            "capacity_factor": factor,
            "algorithm": SELECTED_ALGORITHM,
            "last_observed_update": final_update,
            "final_relative_gap_sptt": final_gap,
            "first_observed_update_at_or_below_0p01": first_below,
            "consecutive_final_observed_checkpoints_at_or_below_0p01": tail_below,
            "qualification_prefers_consecutive_but_requires_one_observed_checkpoint": True,
            "flow_weighted_tti_p95": final_metrics["flow_weighted_tti_p95"],
            "positive_flow_link_tti_max": final_metrics["positive_flow_link_tti_max"],
            "positive_flow_link_count_tti_above_2": final_metrics[
                "positive_flow_link_count_tti_above_2"
            ],
            "route_summary": route_summary,
            "gates": final_gates,
            "qualified": qualified,
            "chunk_summaries": chunk_summaries,
            "final_solver_summary": final_summary,
            "final_state_root": str(final_state_root.resolve()),
            "trajectory_path": str(
                (
                    self.output_root
                    / "tables"
                    / f"factor_{key}_qualification_trajectory.csv"
                ).resolve()
            ),
        }
        _write_json_new(
            self.output_root / "runs" / f"factor_{key}" / "qualification_decision.json",
            qualification,
        )
        self.event(
            "factor_qualification_completed",
            capacity_factor=factor,
            qualified=qualified,
            last_observed_update=final_update,
            final_relative_gap_sptt=final_gap,
        )
        return qualification

    def _all_five_verification(
        self,
        graph: Any,
        od: AggregatedOD,
        trace_maps: Sequence[SuiteTraceMap],
        common_initial: AONTraceResult,
        tie_rank: np.ndarray,
        packed: Any,
        factor: float,
    ) -> dict[str, Any]:
        key = _factor_key(factor)
        self.resource_guard(f"ALL-FIVE-TRACE-VERIFY-{key}")
        traced_initial = run_exact_aon(
            graph,
            od,
            graph.free_time,
            trace_maps=trace_maps,
            config=AONExecutionConfig(worker_count=4),
            arc_tie_rank=tie_rank,
            packed_trace_maps=packed,
        )
        initial_physical_equal = bool(
            np.array_equal(traced_initial.physical_flow, common_initial.physical_flow)
        )
        if not initial_physical_equal or traced_initial.sptt != common_initial.sptt:
            raise RefinementError("All-five traced free-flow AON differs from common physical AON")
        continuous_checkpoint_root = (
            self.output_root
            / "checkpoints"
            / f"factor_{key}"
            / "all_five_continuous"
        )
        continuous = solve_assignment(
            graph,
            od,
            trace_maps=trace_maps,
            config=Phase4SolverConfig(
                method="SAFEGUARDED_CFW",
                capacity_multiplier=factor,
                max_iterations=3,
                minimum_iterations=3,
                worker_count=4,
                checkpoint_directory=continuous_checkpoint_root,
                checkpoint_iterations=(2,),
                final_checkpoint=True,
                rss_hard_bytes=int(self.config["resources"]["process_tree_rss_bytes"]),
            ),
            initial_aon=traced_initial,
            run_metadata={
                "run_id": self.config["run_id"],
                "verification": "all_five_continuous_three_updates",
                "capacity_factor": factor,
            },
        )
        continuous_saved = _save_result(
            self.output_root
            / "runs"
            / f"factor_{key}"
            / "all_five_continuous_3_updates",
            continuous,
            run_scope="all_five_continuous_3_updates",
        )
        split_checkpoint = continuous_checkpoint_root / "iteration_0002"
        restart = load_phase4_checkpoint(split_checkpoint, trace_maps=trace_maps)
        resumed_checkpoint_root = (
            self.output_root / "checkpoints" / f"factor_{key}" / "all_five_resumed"
        )
        resumed = solve_assignment(
            graph,
            od,
            trace_maps=trace_maps,
            config=Phase4SolverConfig(
                method="SAFEGUARDED_CFW",
                capacity_multiplier=factor,
                max_iterations=3,
                worker_count=4,
                checkpoint_directory=resumed_checkpoint_root,
                final_checkpoint=True,
                rss_hard_bytes=int(self.config["resources"]["process_tree_rss_bytes"]),
            ),
            restart_state=restart,
            run_metadata={
                "run_id": self.config["run_id"],
                "verification": "all_five_restart_from_update_2_to_3",
                "capacity_factor": factor,
            },
        )
        _save_result(
            self.output_root
            / "runs"
            / f"factor_{key}"
            / "all_five_resumed_update_3",
            resumed,
            run_scope="all_five_resumed_update_3",
        )
        physical_update3 = load_phase4_checkpoint(
            self.output_root
            / "checkpoints"
            / f"factor_{key}"
            / "screen"
            / "iteration_0003",
            trace_maps=(),
        )
        physical_flow_equal = bool(
            np.array_equal(continuous.flow, physical_update3.physical_flow)
        )
        restart_flow_equal = bool(np.array_equal(continuous.flow, resumed.flow))
        suite_comparisons = {
            item.suite_id: _trace_equal(
                continuous.suite_traces[item.suite_id],
                resumed.suite_traces[item.suite_id],
            )
            for item in trace_maps
        }
        screen_trajectory = pd.read_parquet(
            self.output_root / "runs" / f"factor_{key}" / "screen" / "trajectory.parquet"
        )
        coefficient_rows = []
        coefficients_equal = True
        for record in continuous.histories:
            source = screen_trajectory.loc[
                screen_trajectory["accepted_step_to_update"].eq(record.iteration)
            ].iloc[0]
            source_line = json.loads(source["line_search_json"])
            row_equal = bool(
                float(source["step_size"]) == float(record.step_size)
                and (
                    (pd.isna(source["cfw_alpha"]) and record.cfw_alpha is None)
                    or float(source["cfw_alpha"]) == float(record.cfw_alpha)
                )
                and source_line == record.line_search
            )
            coefficients_equal &= row_equal
            coefficient_rows.append(
                {
                    "iteration": record.iteration,
                    "physical_only_step_size": float(source["step_size"]),
                    "all_five_step_size": record.step_size,
                    "physical_only_cfw_alpha": (
                        None if pd.isna(source["cfw_alpha"]) else float(source["cfw_alpha"])
                    ),
                    "all_five_cfw_alpha": record.cfw_alpha,
                    "line_search_equal": source_line == record.line_search,
                    "coefficients_equal": row_equal,
                }
            )
        mapping_failures_zero = all(
            int(value) == 0 for value in continuous.latest_aon.mapping_failure_counts.values()
        )
        topology_od_unchanged = bool(
            continuous.metadata["graph_hash"] == self.config["input_versions"]["graph_identity"]
            and continuous.metadata["od_hash"] == self.config["input_versions"]["od_identity"]
        )
        passed = bool(
            initial_physical_equal
            and physical_flow_equal
            and restart_flow_equal
            and coefficients_equal
            and all(value["pass"] for value in suite_comparisons.values())
            and mapping_failures_zero
            and topology_od_unchanged
            and continuous.metadata["physical_and_trace_coefficients_identical"]
        )
        verification = {
            "schema_id": "phase4.all_five_recommended_factor_verification.v1",
            "status": "PASS" if passed else "FAIL",
            "capacity_factor": factor,
            "algorithm": SELECTED_ALGORITHM,
            "completed_fully_refreshed_updates": 3,
            "suite_ids": [item.suite_id for item in trace_maps],
            "s01_absent": all(item.suite_id != "s01" for item in trace_maps),
            "common_initial_physical_flow_equal": initial_physical_equal,
            "physical_flow_matches_physical_only_update_3": physical_flow_equal,
            "physical_flow_max_abs_difference": float(
                np.max(np.abs(continuous.flow - physical_update3.physical_flow))
            ),
            "checkpoint_restart_physical_flow_equal": restart_flow_equal,
            "checkpoint_restart_physical_flow_max_abs_difference": float(
                np.max(np.abs(continuous.flow - resumed.flow))
            ),
            "checkpoint_restart_suite_trace_comparisons": suite_comparisons,
            "physical_and_trace_coefficients_identical": continuous.metadata[
                "physical_and_trace_coefficients_identical"
            ],
            "physical_only_and_all_five_coefficients_equal": coefficients_equal,
            "mapping_failure_counts": continuous.latest_aon.mapping_failure_counts,
            "mapping_failures_zero": mapping_failures_zero,
            "topology_and_od_unchanged": topology_od_unchanged,
            "continuous_solver_summary": continuous_saved["summary"],
            "split_checkpoint": str(split_checkpoint.resolve()),
        }
        _write_json_new(
            self.output_root / "validation" / "all_five_trace_verification.json",
            verification,
        )
        _write_frame_new(
            pd.DataFrame(coefficient_rows),
            self.output_root / "tables" / "all_five_coefficient_equivalence.csv",
        )
        _write_frame_new(
            pd.DataFrame(
                [
                    {
                        "suite_id": suite_id,
                        "restart_trace_equal": comparison["pass"],
                        "mapping_failure_count": continuous.latest_aon.mapping_failure_counts[
                            suite_id
                        ],
                        "physical_flow_matches_physical_only": physical_flow_equal,
                        "coefficients_match_physical_only": coefficients_equal,
                    }
                    for suite_id, comparison in suite_comparisons.items()
                ]
            ),
            self.output_root / "tables" / "all_five_traced_verification.csv",
        )
        if not passed:
            raise RefinementError("Recommended-factor exact all-five verification failed")
        self.event(
            "all_five_trace_verification_completed",
            capacity_factor=factor,
            status="PASS",
        )
        return verification

    def _selected_link_tables(
        self,
        graph: Any,
        qualification: Mapping[str, Any],
    ) -> dict[str, Any]:
        state_root = Path(qualification["final_state_root"])
        state_path = state_root / "final_physical_state.npz"
        with np.load(state_path, allow_pickle=False) as archive:
            flow = np.asarray(archive["physical_flow"], dtype=np.float64)
            cost = np.asarray(archive["physical_cost"], dtype=np.float64)
            capacity = np.asarray(archive["used_capacity"], dtype=np.float64)
        vc = np.divide(flow, capacity, out=np.zeros_like(flow), where=capacity > 0.0)
        tti = np.divide(cost, graph.free_time, out=np.ones_like(cost), where=graph.free_time > 0.0)
        delay = flow * np.maximum(cost - graph.free_time, 0.0)
        rows = []
        for metric_name, values in (
            ("vc", vc),
            ("tti", tti),
            ("flow_weighted_delay", delay),
        ):
            count = min(100, len(values))
            selected = np.argpartition(values, -count)[-count:]
            selected = selected[np.argsort(values[selected], kind="mergesort")[::-1]]
            for rank, arc_idx in enumerate(selected.tolist(), start=1):
                region_code = int(graph.region_code[arc_idx])
                rows.append(
                    {
                        "rank_metric": metric_name,
                        "rank": rank,
                        "arc_idx": arc_idx,
                        "directed_arc_id": str(graph.arc_idx_to_directed_arc_id[arc_idx]),
                        "link_uid": str(graph.arc_idx_to_link_uid[arc_idx]),
                        "tail_idx": int(graph.tail[arc_idx]),
                        "head_idx": int(graph.head[arc_idx]),
                        "tail_node_uid": str(graph.node_idx_to_uid[graph.tail[arc_idx]]),
                        "head_node_uid": str(graph.node_idx_to_uid[graph.head[arc_idx]]),
                        "region_code": region_code,
                        "region_label": str(graph.region_idx_to_label[region_code]),
                        "capacity_multiplier_type": str(
                            graph.capacity_multiplier_type[arc_idx]
                        ),
                        "free_time": float(graph.free_time[arc_idx]),
                        "flow": float(flow[arc_idx]),
                        "used_capacity": float(capacity[arc_idx]),
                        "vc": float(vc[arc_idx]),
                        "cost": float(cost[arc_idx]),
                        "tti": float(tti[arc_idx]),
                        "flow_weighted_delay": float(delay[arc_idx]),
                    }
                )
        top = pd.DataFrame(rows)
        _write_frame_new(top, self.output_root / "tables" / "selected_top100_links.csv")
        bins = np.asarray(
            [0.0, 0.5, 0.8, 1.0, 1.05, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf]
        )
        distribution_rows = []
        for metric_name, values, weights in (
            ("vc", vc, flow),
            ("tti", tti, flow),
        ):
            counts, _ = np.histogram(values, bins=bins)
            weighted, _ = np.histogram(values, bins=bins, weights=weights)
            for index in range(len(bins) - 1):
                distribution_rows.append(
                    {
                        "metric": metric_name,
                        "bin_lower": float(bins[index]),
                        "bin_upper": (
                            None if np.isinf(bins[index + 1]) else float(bins[index + 1])
                        ),
                        "arc_count": int(counts[index]),
                        "flow_weight": float(weighted[index]),
                    }
                )
        distribution = pd.DataFrame(distribution_rows)
        _write_frame_new(
            distribution,
            self.output_root / "tables" / "selected_vc_tti_distributions.csv",
        )
        receipt = {
            "selected_state_path": str(state_path.resolve()),
            "selected_state_sha256": file_sha256(state_path),
            "flow_shape": list(flow.shape),
            "finite_flow": bool(np.all(np.isfinite(flow))),
            "finite_cost": bool(np.all(np.isfinite(cost))),
            "positive_flow_arc_count": int(np.count_nonzero(flow > 0.0)),
            "max_vc": float(np.max(vc)),
            "max_tti_positive_flow": float(np.max(tti[flow > 0.0])),
            "flow_weighted_tti_p95": _weighted_quantile(tti, flow, 0.95),
        }
        _write_json_new(
            self.output_root / "validation" / "selected_state_reopen.json", receipt
        )
        return receipt

    @staticmethod
    def _forecast_rows(
        factor: float,
        trajectory: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        frame = RefinementCampaign._deduplicate_trajectory(trajectory)
        usable = frame[
            np.isfinite(frame["relative_gap_sptt"].to_numpy(dtype=np.float64))
            & frame["relative_gap_sptt"].gt(0.0)
        ].copy()
        late = usable.tail(min(10, len(usable)))
        slope = float("nan")
        intercept = float("nan")
        residual_std = float("nan")
        if len(late) >= 3:
            x = late["update"].to_numpy(dtype=np.float64)
            y = np.log(late["relative_gap_sptt"].to_numpy(dtype=np.float64))
            slope, intercept = np.polyfit(x, y, 1)
            residual_std = float(np.std(y - (slope * x + intercept), ddof=1))
        accepted = frame[frame["accepted_step_to_update"].notna()]
        seconds_per_update = float(
            accepted["iteration_wall_seconds"].mean()
        ) if len(accepted) else float("nan")
        current_update = int(frame["update"].max())
        rows = []
        for label, target in (
            ("1%", 1e-2),
            ("0.5%", 5e-3),
            ("0.1%", 1e-3),
            ("0.05%", 5e-4),
            ("0.01%", 1e-4),
            ("0.005%", 5e-5),
        ):
            observed = frame.loc[frame["relative_gap_sptt"].le(target), "update"]
            if len(observed):
                projected_update = float(observed.iloc[0])
                basis = "OBSERVED"
                lower = projected_update
                upper = projected_update
            elif np.isfinite(slope) and slope < 0.0:
                projected_update = max(
                    float(current_update), float((math.log(target) - intercept) / slope)
                )
                spread = (
                    1.96 * residual_std / abs(slope)
                    if np.isfinite(residual_std)
                    else float("nan")
                )
                lower = max(float(current_update), projected_update - spread)
                upper = projected_update + spread
                basis = "LOG_LINEAR_LATE_TRAJECTORY_PROJECTION_NOT_QUALIFICATION"
            else:
                projected_update = float("nan")
                lower = float("nan")
                upper = float("nan")
                basis = "UNRESOLVED_NONDECREASING_OR_INSUFFICIENT_TRAJECTORY"
            rows.append(
                {
                    "capacity_factor": factor,
                    "target_label": label,
                    "relative_gap_sptt_target": target,
                    "basis": basis,
                    "observed_update": int(observed.iloc[0]) if len(observed) else None,
                    "projected_update": projected_update if np.isfinite(projected_update) else None,
                    "projected_update_lower_95": lower if np.isfinite(lower) else None,
                    "projected_update_upper_95": upper if np.isfinite(upper) else None,
                    "seconds_per_update_measured": (
                        seconds_per_update if np.isfinite(seconds_per_update) else None
                    ),
                    "projected_runtime_seconds": (
                        projected_update * seconds_per_update
                        if np.isfinite(projected_update) and np.isfinite(seconds_per_update)
                        else None
                    ),
                    "fit_slope_log_gap_per_update": slope if np.isfinite(slope) else None,
                    "fit_residual_std_log_gap": (
                        residual_std if np.isfinite(residual_std) else None
                    ),
                    "latest_observed_update": current_update,
                    "prediction_used_for_factor_qualification": False,
                }
            )
        return rows

    def _write_aggregate_tables(
        self,
        screens: Sequence[Mapping[str, Any]],
        qualifications: Sequence[Mapping[str, Any]],
        recommendation: float | None,
    ) -> None:
        screen_frames = []
        for screen in screens:
            key = _factor_key(float(screen["capacity_factor"]))
            frame = pd.read_parquet(
                self.output_root / "runs" / f"factor_{key}" / "screen" / "trajectory.parquet"
            )
            frame["grid_scope"] = screen["grid_scope"]
            screen_frames.append(frame)
        all_screens = pd.concat(screen_frames, ignore_index=True)
        _write_frame_new(
            all_screens,
            self.output_root / "tables" / "five_factor_short_screen_trajectory.csv",
        )
        conditional = all_screens[all_screens["grid_scope"].eq("conditional")]
        if len(conditional):
            _write_frame_new(
                conditional,
                self.output_root / "tables" / "fallback_factor_trajectory.csv",
            )
        else:
            _write_frame_new(
                pd.DataFrame(
                    [
                        {
                            "status": "NOT_USED_PRIMARY_FACTOR_QUALIFIED_OR_REMAINED_PLAUSIBLE",
                            "authorized_conditional_factors": "0.55,0.60",
                        }
                    ]
                ),
                self.output_root / "tables" / "fallback_factor_trajectory.csv",
            )

        qualification_frames = []
        forecast_rows = []
        for qualification in qualifications:
            path = Path(qualification["trajectory_path"])
            frame = pd.read_csv(path)
            qualification_frames.append(frame)
            forecast_rows.extend(
                self._forecast_rows(float(qualification["capacity_factor"]), frame)
            )
        if qualification_frames:
            _write_frame_new(
                pd.concat(qualification_frames, ignore_index=True),
                self.output_root / "tables" / "qualification_checkpoints.csv",
            )
        else:
            _write_frame_new(
                pd.DataFrame([{"status": "NO_PLAUSIBLE_FACTOR_CONTINUED"}]),
                self.output_root / "tables" / "qualification_checkpoints.csv",
            )
        if recommendation is not None and not any(
            float(item["capacity_factor"]) == float(recommendation)
            for item in qualifications
        ):
            key = _factor_key(recommendation)
            frame = pd.read_parquet(
                self.output_root / "runs" / f"factor_{key}" / "screen" / "trajectory.parquet"
            )
            forecast_rows.extend(self._forecast_rows(recommendation, frame))
        _write_frame_new(
            pd.DataFrame(forecast_rows),
            self.output_root / "tables" / "runtime_convergence_forecasts.csv",
        )

        coefficient_frames = []
        for path in sorted((self.output_root / "runs").rglob("trajectory.parquet")):
            frame = pd.read_parquet(path)
            selected = frame[frame["accepted_step_to_update"].notna()].copy()
            selected["source_path"] = str(path.resolve())
            coefficient_frames.append(selected)
        _write_frame_new(
            pd.concat(coefficient_frames, ignore_index=True),
            self.output_root / "tables" / "cfw_coefficients_resets_line_search.csv",
        )

    def execute(self) -> None:
        self._prepare_output()
        try:
            self.update_status(
                "EXECUTING_INPUT_AND_SANITATION",
                completed_tasks=["STARTUP-AUTHORITY"],
                next_task="INPUT-INTEGRITY",
            )
            graph, od, trace_maps, audit = self._load_and_sanitize()
            startup_elapsed = time.perf_counter() - self.started_perf
            if startup_elapsed > float(
                self.config["resources"]["startup_sanitation_code_review_limit_seconds"]
            ):
                raise TimeoutError("Startup/sanitation/code-review work exceeded 600 seconds")
            self.update_status(
                "EXECUTING_EXACT_OPTIMIZATION",
                completed_tasks=[
                    "STARTUP-AUTHORITY",
                    "INPUT-INTEGRITY",
                    "CAPACITY-SANITATION",
                    "FOCUSED-CODE-REVIEW",
                ],
                next_task="EXACT-OPTIMIZATION",
            )
            tie_rank, packed = self._optimization_equivalence(graph, od, trace_maps)
            pytest_started = time.perf_counter()
            pytest_result = subprocess.run(
                [
                    self.config["python_executable"],
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_phase4_preflight.py",
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            pytest_receipt = {
                "returncode": int(pytest_result.returncode),
                "runtime_seconds": float(time.perf_counter() - pytest_started),
                "stdout": pytest_result.stdout,
                "stderr": pytest_result.stderr,
            }
            _write_json_new(
                self.output_root / "validation" / "focused_pytest_receipt.json",
                pytest_receipt,
            )
            if pytest_result.returncode != 0:
                raise RefinementError("Focused Phase 4 tests failed after exact optimization")
            common_initial = self._common_initialization(graph, od, tie_rank)
            self.update_status(
                "EXECUTING_PRIMARY_SHORT_SCREENS",
                completed_tasks=[
                    "STARTUP-AUTHORITY",
                    "INPUT-INTEGRITY",
                    "CAPACITY-SANITATION",
                    "FOCUSED-CODE-REVIEW",
                    "EXACT-OPTIMIZATION",
                    "FOCUSED-TESTS",
                ],
                next_task="PRIMARY-SHORT-SCREENS",
            )
            screens: list[dict[str, Any]] = []
            for factor in self.config["primary_capacity_factors"]:
                screens.append(
                    self._screen_factor(
                        graph,
                        od,
                        common_initial,
                        tie_rank,
                        float(factor),
                        grid_scope="primary",
                    )
                )

            qualifications: list[dict[str, Any]] = []
            recommendation: float | None = None
            plausible_primary = [
                item for item in screens if item["plausible_for_continuation"]
            ]
            for screen in plausible_primary[:2]:
                qualification = self._continue_factor(graph, od, tie_rank, screen)
                qualifications.append(qualification)
                if qualification["qualified"]:
                    recommendation = float(qualification["capacity_factor"])
                    break

            if recommendation is None:
                self.update_status(
                    "EXECUTING_CONDITIONAL_FACTOR_SCREEN",
                    completed_tasks=["PRIMARY-SHORT-SCREENS"],
                    next_task="CONDITIONAL-0.55-0.60",
                )
                conditional_screens = []
                for factor in self.config["conditional_capacity_factors"]:
                    conditional_screens.append(
                        self._screen_factor(
                            graph,
                            od,
                            common_initial,
                            tie_rank,
                            float(factor),
                            grid_scope="conditional",
                        )
                    )
                screens.extend(conditional_screens)
                for screen in conditional_screens:
                    if not screen["plausible_for_continuation"]:
                        continue
                    qualification = self._continue_factor(graph, od, tie_rank, screen)
                    qualifications.append(qualification)
                    if qualification["qualified"]:
                        recommendation = float(qualification["capacity_factor"])
                        break

            qualification_by_factor = {
                float(item["capacity_factor"]): item for item in qualifications
            }
            decision_rows = []
            for screen in screens:
                factor = float(screen["capacity_factor"])
                qualification = qualification_by_factor.get(factor)
                decision_rows.append(
                    {
                        "capacity_factor": factor,
                        "grid_scope": screen["grid_scope"],
                        "short_screen_plausible": screen["plausible_for_continuation"],
                        "initial_relative_gap_sptt": screen["initial_relative_gap_sptt"],
                        "relative_gap_sptt_after_3_updates": screen[
                            "relative_gap_sptt_after_3_updates"
                        ],
                        "relative_gap_sptt_after_5_updates": screen[
                            "final_relative_gap_sptt_after_5_updates"
                        ],
                        "screen_flow_weighted_tti_p95": screen["flow_weighted_tti_p95"],
                        "screen_positive_flow_link_tti_max": screen[
                            "positive_flow_link_tti_max"
                        ],
                        "screen_genuine_non_tie_route_switch_count": screen[
                            "route_summary"
                        ]["genuine_non_tie_route_switch_count"],
                        "continued": qualification is not None,
                        "last_observed_update": (
                            qualification["last_observed_update"] if qualification else 5
                        ),
                        "final_observed_relative_gap_sptt": (
                            qualification["final_relative_gap_sptt"]
                            if qualification
                            else screen["final_relative_gap_sptt_after_5_updates"]
                        ),
                        "qualified": bool(qualification and qualification["qualified"]),
                        "recommended": bool(recommendation is not None and factor == recommendation),
                    }
                )
            decision_frame = pd.DataFrame(decision_rows).sort_values("capacity_factor")
            _write_frame_new(
                decision_frame,
                self.output_root / "tables" / "factor_decision_matrix.csv",
            )
            decision = {
                "schema_id": "phase4.capacity_factor_decision.v1",
                "status": (
                    "RECOMMENDED_FACTOR_OBSERVED_AND_QUALIFIED"
                    if recommendation is not None
                    else "NO_CAPACITY_FACTOR_QUALIFIED_WITHIN_2H_PREFLIGHT"
                ),
                "recommended_capacity_factor": (
                    recommendation if recommendation is not None else "unresolved"
                ),
                "smallest_qualified_factor_rule_applied": True,
                "primary_factor_continuation_count": sum(
                    1
                    for item in qualifications
                    if float(item["capacity_factor"])
                    in self.config["primary_capacity_factors"]
                ),
                "primary_factor_continuation_limit": 2,
                "conditional_grid_used": any(
                    item["grid_scope"] == "conditional" for item in screens
                ),
                "qualification_decisions": qualifications,
                "prediction_used_to_qualify": False,
                "algorithm": SELECTED_ALGORITHM,
                "clean_capacity_payload_sha256": audit[
                    "clean_c_order_float64_payload_sha256"
                ],
            }
            _write_json_new(
                self.output_root / "validation" / "capacity_factor_decision.json",
                decision,
            )

            all_five = None
            selected_receipt = None
            if recommendation is not None:
                all_five = self._all_five_verification(
                    graph,
                    od,
                    trace_maps,
                    common_initial,
                    tie_rank,
                    packed,
                    recommendation,
                )
                selected_receipt = self._selected_link_tables(
                    graph, qualification_by_factor[recommendation]
                )
            else:
                _write_frame_new(
                    pd.DataFrame([{"status": "NOT_APPLICABLE_NO_RECOMMENDED_FACTOR"}]),
                    self.output_root / "tables" / "selected_top100_links.csv",
                )
                _write_frame_new(
                    pd.DataFrame([{"status": "NOT_APPLICABLE_NO_RECOMMENDED_FACTOR"}]),
                    self.output_root / "tables" / "selected_vc_tti_distributions.csv",
                )
                _write_frame_new(
                    pd.DataFrame([{"status": "NOT_APPLICABLE_NO_RECOMMENDED_FACTOR"}]),
                    self.output_root / "tables" / "all_five_traced_verification.csv",
                )
            self._write_aggregate_tables(
                screens, qualifications, recommendation
            )
            completion = {
                "schema_id": "phase4.capacity_factor_refinement.execution_completion.v1",
                "status": "PASS_EXECUTION_COMPLETE",
                "campaign_started_utc": self.started_utc,
                "execution_completed_utc": _utc_now(),
                "elapsed_seconds": float(time.perf_counter() - self.started_perf),
                "peak_rss_bytes": int(self.peak_rss),
                "output_size_bytes": _directory_size(self.output_root),
                "temp_size_bytes": _directory_size(self.temp_root),
                "recommended_capacity_factor": (
                    recommendation if recommendation is not None else "unresolved"
                ),
                "decision_status": decision["status"],
                "all_five_verification": (
                    all_five["status"] if all_five is not None else "NOT_APPLICABLE"
                ),
                "selected_state_reopen": selected_receipt,
                "phase4_24h_production_assignment_authorized": False,
                "phase5_authorized": False,
            }
            _write_json_new(
                self.output_root / "status" / "execution_completion.json", completion
            )
            self.update_status(
                "EXECUTION_COMPLETE_AWAITING_FINAL_VALIDATION_AND_HANDOFF",
                completed_tasks=self.config["tasks"][:-3],
                next_task="FINAL-VALIDATE",
                recommended_capacity_factor=(
                    recommendation if recommendation is not None else "unresolved"
                ),
                decision_status=decision["status"],
                execution_elapsed_seconds=completion["elapsed_seconds"],
            )
            self.event(
                "scientific_execution_completed",
                recommended_capacity_factor=(
                    recommendation if recommendation is not None else "unresolved"
                ),
                decision_status=decision["status"],
            )
        except BaseException as exc:
            if self.status_path.exists():
                self.update_status(
                    "EXECUTION_FAILED_REQUIRES_REPAIR_HANDOFF",
                    failure_type=type(exc).__name__,
                    failure_message=str(exc),
                    phase4_24h_production_assignment_authorized=False,
                    phase5_authorized=False,
                )
                self.event(
                    "scientific_execution_failed",
                    failure_type=type(exc).__name__,
                    failure_message=str(exc),
                )
            raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("execute",), required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    campaign = RefinementCampaign(args.config)
    expected = {
        "phase_id": args.phase_id,
        "step_id": args.step_id,
        "run_id": args.run_id,
    }
    for key, value in expected.items():
        if campaign.config[key] != value:
            raise RefinementError(f"CLI {key} differs from explicit execution config")
    campaign.execute()


if __name__ == "__main__":
    main()
