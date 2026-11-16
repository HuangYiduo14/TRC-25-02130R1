"""Physical static assignment solvers for Prompt 4B.

The solvers consume ``graph.capacity`` from the Prompt-4 graph snapshot. That
array may be an assignment-only scaled copy of the map-derived capacity; raw
SQLite/GPKG network capacity fields are not changed by these solvers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .partC_rev1_aon import PartCRev1AONResult, partC_rev1_aon_load
from .partC_rev1_bpr import (
    PartCRev1LineSearchResult,
    partC_rev1_bpr_cost,
    partC_rev1_bpr_derivative,
    partC_rev1_bpr_objective,
    partC_rev1_fw_line_search,
    partC_rev1_relative_gap,
)
from .partC_rev1_graph_snapshot import PartCRev1GraphSnapshot
from .partC_rev1_od_snapshot import PartCRev1ODSnapshot
from .partC_rev1_progress_logging import PartCRev1ProgressReporter
from .partC_rev1_runtime_guard import partC_rev1_resource_snapshot


@dataclass(frozen=True, slots=True)
class PartCRev1AssignmentConfig:
    max_iterations: int = 20
    gap_target: float = 1e-4
    backend: str = "numba_csr"
    origin_batch_size: int = 32
    max_origins: int | None = None
    fw_warm_start_iterations: int = 1
    line_search_derivative_tol: float = 1e-8
    line_search_interval_tol: float = 1e-10
    descent_tol: float = 1e-9
    denominator_tol: float = 1e-12
    delta_cfw: float = 1e-4
    epsilon: float = 1e-12
    store_paths: bool = False
    max_runtime_seconds: float | None = None


@dataclass(slots=True)
class PartCRev1IterationRecord:
    iteration: int
    method: str
    backend: str
    objective: float
    tstt: float
    sptt: float
    relative_gap: float | None
    step_size: float
    aon_runtime_seconds: float
    iteration_runtime_seconds: float
    reachable_demand: float
    unreachable_demand: float
    flow_l1_change: float
    line_search: dict[str, Any]
    cfw_alpha: float | None = None
    fallback_reason: str | None = None


@dataclass(slots=True)
class PartCRev1SolverResult:
    method: str
    backend: str
    flow: np.ndarray
    cost: np.ndarray
    histories: list[PartCRev1IterationRecord]
    stop_reason: str
    runtime_seconds: float
    peak_memory_mb: float
    graph_hash: str
    od_hash: str
    unreachable_demand: float
    metadata: dict[str, Any]


def _partC_rev1_record_to_dict(record: PartCRev1IterationRecord) -> dict[str, Any]:
    return {
        "iteration": record.iteration,
        "method": record.method,
        "backend": record.backend,
        "objective": record.objective,
        "tstt": record.tstt,
        "sptt": record.sptt,
        "relative_gap": record.relative_gap,
        "step_size": record.step_size,
        "aon_runtime_seconds": record.aon_runtime_seconds,
        "iteration_runtime_seconds": record.iteration_runtime_seconds,
        "reachable_demand": record.reachable_demand,
        "unreachable_demand": record.unreachable_demand,
        "flow_l1_change": record.flow_l1_change,
        "line_search": record.line_search,
        "cfw_alpha": record.cfw_alpha,
        "fallback_reason": record.fallback_reason,
    }


def _partC_rev1_line_result_to_dict(result: PartCRev1LineSearchResult | None) -> dict[str, Any]:
    if result is None:
        return {}
    return {
        "eta": result.eta,
        "derivative_at_zero": result.derivative_at_zero,
        "derivative_at_one": result.derivative_at_one,
        "iterations": result.iterations,
        "stop_reason": result.stop_reason,
    }


def partC_rev1_cfw_alpha(
    previous_s: np.ndarray,
    current_flow: np.ndarray,
    fw_direction: np.ndarray,
    previous_tau: float,
    previous_cfw_direction: np.ndarray,
    hessian_diag: np.ndarray,
    *,
    denominator_tol: float,
    delta_cfw: float,
) -> tuple[float, dict[str, float | str]]:
    del previous_s
    dbar = (1.0 - float(previous_tau)) * np.asarray(previous_cfw_direction, dtype=np.float64)
    numerator = float(np.dot(dbar * hessian_diag, fw_direction))
    denominator = float(np.dot(dbar * hessian_diag, np.asarray(fw_direction, dtype=np.float64) - dbar))
    if abs(denominator) <= denominator_tol:
        return 0.0, {"numerator": numerator, "denominator": denominator, "reason": "denominator_tol"}
    ratio = numerator / denominator
    if 0.0 <= ratio <= 1.0 - delta_cfw:
        alpha = ratio
    elif ratio > 1.0 - delta_cfw:
        alpha = 1.0 - delta_cfw
    else:
        alpha = 0.0
    if not np.isfinite(alpha):
        alpha = 0.0
    return float(alpha), {"numerator": numerator, "denominator": denominator, "ratio": float(ratio), "reason": "ok"}


def partC_rev1_solve_physical_assignment(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    *,
    method: str,
    config: PartCRev1AssignmentConfig | None = None,
    reporter: PartCRev1ProgressReporter | None = None,
) -> PartCRev1SolverResult:
    cfg = config or PartCRev1AssignmentConfig()
    method_key = method.lower()
    if method_key not in {"msa", "fw", "cfw"}:
        raise ValueError(f"Unsupported physical assignment method: {method}")
    if cfg.store_paths:
        raise ValueError("store_paths=true is reserved for the optional GP solver, not MSA/FW/CFW")

    started = time.perf_counter()
    flow = np.zeros(len(graph.arc_idx_to_directed_arc_id), dtype=np.float64)
    histories: list[PartCRev1IterationRecord] = []
    previous_s: np.ndarray | None = None
    previous_tau: float | None = None
    previous_cfw_direction: np.ndarray | None = None
    stop_reason = "max_iterations"
    peak_memory_mb = 0.0
    last_aon: PartCRev1AONResult | None = None

    for iteration in range(int(cfg.max_iterations)):
        iter_started = time.perf_counter()
        cost = partC_rev1_bpr_cost(flow, graph.free_time, graph.capacity, graph.alpha, graph.beta)
        aon = partC_rev1_aon_load(
            graph,
            od,
            cost,
            backend=cfg.backend,
            max_origins=cfg.max_origins,
            origin_batch_size=cfg.origin_batch_size,
            reporter=reporter,
        )
        last_aon = aon
        y = aon.flow
        fw_direction = y - flow
        objective = partC_rev1_bpr_objective(flow, graph.free_time, graph.capacity, graph.alpha, graph.beta)
        tstt = float(np.dot(flow, cost))
        if iteration == 0 and tstt <= cfg.epsilon:
            relative_gap: float | None = None
        else:
            relative_gap = partC_rev1_relative_gap(tstt, aon.sptt, epsilon=cfg.epsilon)

        fallback_reason: str | None = None
        cfw_alpha_value: float | None = None
        line_result: PartCRev1LineSearchResult | None = None

        if method_key == "msa":
            direction = fw_direction
            step_size = 1.0 / float(iteration + 1)
            sight = y
        elif method_key == "fw":
            direction = fw_direction
            sight = y
            if iteration < cfg.fw_warm_start_iterations:
                step_size = 1.0
                fallback_reason = "fw_warm_start_msa_step"
            else:
                line_result = partC_rev1_fw_line_search(
                    flow,
                    direction,
                    graph.free_time,
                    graph.capacity,
                    graph.alpha,
                    graph.beta,
                    derivative_tol=cfg.line_search_derivative_tol,
                    interval_tol=cfg.line_search_interval_tol,
                )
                step_size = line_result.eta
        else:
            if (
                iteration == 0
                or previous_s is None
                or previous_tau is None
                or previous_cfw_direction is None
                or previous_tau >= 1.0 - 1e-8
            ):
                sight = y
                direction = fw_direction
                cfw_alpha_value = 0.0
            else:
                hessian_diag = partC_rev1_bpr_derivative(flow, graph.free_time, graph.capacity, graph.alpha, graph.beta)
                cfw_alpha_value, _alpha_meta = partC_rev1_cfw_alpha(
                    previous_s,
                    flow,
                    fw_direction,
                    previous_tau,
                    previous_cfw_direction,
                    hessian_diag,
                    denominator_tol=cfg.denominator_tol,
                    delta_cfw=cfg.delta_cfw,
                )
                sight = cfw_alpha_value * previous_s + (1.0 - cfw_alpha_value) * y
                direction = sight - flow
                descent = float(np.dot(cost, direction))
                if not np.isfinite(descent) or descent >= -cfg.descent_tol:
                    sight = y
                    direction = fw_direction
                    fallback_reason = "cfw_non_descent_fallback_to_fw"
            if iteration < cfg.fw_warm_start_iterations:
                step_size = 1.0
                fallback_reason = fallback_reason or "cfw_warm_start_msa_step"
            else:
                line_result = partC_rev1_fw_line_search(
                    flow,
                    direction,
                    graph.free_time,
                    graph.capacity,
                    graph.alpha,
                    graph.beta,
                    derivative_tol=cfg.line_search_derivative_tol,
                    interval_tol=cfg.line_search_interval_tol,
                )
                step_size = line_result.eta

        new_flow = flow + float(step_size) * direction
        new_flow[np.abs(new_flow) < 1e-10] = 0.0
        if np.any(new_flow < -1e-7):
            raise RuntimeError("Assignment update produced negative flow")
        new_flow = np.maximum(new_flow, 0.0)
        flow_l1_change = float(np.sum(np.abs(new_flow - flow), dtype=np.float64))
        iteration_elapsed = time.perf_counter() - iter_started
        resource = partC_rev1_resource_snapshot(started)
        peak_memory_mb = max(peak_memory_mb, resource.memory_rss_mb)
        record = PartCRev1IterationRecord(
            iteration=iteration,
            method=method_key,
            backend=cfg.backend,
            objective=objective,
            tstt=tstt,
            sptt=float(aon.sptt),
            relative_gap=relative_gap,
            step_size=float(step_size),
            aon_runtime_seconds=float(aon.metadata.get("runtime_seconds", aon.shortest_path_runtime_seconds)),
            iteration_runtime_seconds=iteration_elapsed,
            reachable_demand=float(aon.reachable_demand),
            unreachable_demand=float(aon.unreachable_demand),
            flow_l1_change=flow_l1_change,
            line_search=_partC_rev1_line_result_to_dict(line_result),
            cfw_alpha=cfw_alpha_value,
            fallback_reason=fallback_reason,
        )
        histories.append(record)
        if reporter is not None:
            reporter.emit(
                "assignment_iteration",
                **_partC_rev1_record_to_dict(record),
                memory_rss_mb=resource.memory_rss_mb,
                temp_disk_mb=resource.temp_disk_mb,
            )
        flow = new_flow
        previous_s = np.asarray(sight, dtype=np.float64).copy()
        previous_tau = float(step_size)
        previous_cfw_direction = np.asarray(direction, dtype=np.float64).copy()
        if relative_gap is not None and relative_gap <= cfg.gap_target and iteration > 0:
            stop_reason = "gap_target_met"
            break
        if cfg.max_runtime_seconds is not None and time.perf_counter() - started >= float(cfg.max_runtime_seconds):
            stop_reason = "runtime_limit"
            break

    final_cost = partC_rev1_bpr_cost(flow, graph.free_time, graph.capacity, graph.alpha, graph.beta)
    return PartCRev1SolverResult(
        method=method_key,
        backend=cfg.backend,
        flow=flow,
        cost=final_cost,
        histories=histories,
        stop_reason=stop_reason,
        runtime_seconds=time.perf_counter() - started,
        peak_memory_mb=peak_memory_mb,
        graph_hash=graph.graph_hash,
        od_hash=od.od_hash,
        unreachable_demand=float(last_aon.unreachable_demand if last_aon is not None else 0.0),
        metadata={
            "max_iterations": cfg.max_iterations,
            "gap_target": cfg.gap_target,
            "max_runtime_seconds": cfg.max_runtime_seconds,
            "partial": bool(cfg.max_origins is not None and cfg.max_origins < len(od.origin_group_node_idx)),
            "processed_origins": int(last_aon.processed_origins if last_aon is not None else 0),
            "total_origins": int(len(od.origin_group_node_idx)),
        },
    )


def partC_rev1_solver_result_summary(result: PartCRev1SolverResult) -> dict[str, Any]:
    last = result.histories[-1] if result.histories else None
    return {
        "method": result.method,
        "backend": result.backend,
        "stop_reason": result.stop_reason,
        "runtime_seconds": result.runtime_seconds,
        "peak_memory_mb": result.peak_memory_mb,
        "graph_hash": result.graph_hash,
        "od_hash": result.od_hash,
        "unreachable_demand": result.unreachable_demand,
        "iterations": [_partC_rev1_record_to_dict(record) for record in result.histories],
        "last_relative_gap": last.relative_gap if last is not None else None,
        "last_objective": last.objective if last is not None else None,
        "flow_sum": float(np.sum(result.flow, dtype=np.float64)),
        "metadata": dict(result.metadata),
    }
