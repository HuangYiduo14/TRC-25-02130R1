"""Prompt 4F physical regional reload and final flow reconstruction."""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .partC_rev1_bpr import partC_rev1_bpr_cost, partC_rev1_relative_gap
from .partC_rev1_graph_snapshot import PartCRev1GraphSnapshot
from .partC_rev1_hashing import partC_rev1_json_hash
from .partC_rev1_io import partC_rev1_read_json, partC_rev1_write_json, partC_rev1_write_table
from .partC_rev1_od_snapshot import PartCRev1ODSnapshot
from .partC_rev1_paths import partC_rev1_resolve_path
from .partC_rev1_progress_logging import PartCRev1ProgressReporter
from .partC_rev1_regional_training import partC_rev1_load_benchmark_flow
from .partC_rev1_runtime_guard import PARTC_REV1_DEFAULT_MAX_RUN_SECONDS, partC_rev1_resource_snapshot


PARTC_REV1_4F_STAGE = "partC_rev1_4F"


@dataclass(frozen=True, slots=True)
class PartCRev1PhysicalReloadConfig:
    config_path: str
    max_regions: int = 0
    max_iterations: int = 8
    gap_target: float = 1e-4
    positive_flow_tol: float = 1e-9
    runtime_guard_seconds: float = PARTC_REV1_DEFAULT_MAX_RUN_SECONDS
    correction_enabled: bool = False


@dataclass(frozen=True, slots=True)
class PartCRev1RegionSubgraph:
    region_uuid: str
    global_arc_idx: np.ndarray
    global_node_idx: np.ndarray
    tail: np.ndarray
    head: np.ndarray
    free_time: np.ndarray
    capacity: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    indptr: np.ndarray
    csr_arc_idx: np.ndarray
    global_to_local_node: dict[int, int]


@dataclass(frozen=True, slots=True)
class PartCRev1RegionReloadResult:
    region_uuid: str
    arc_flow_global: np.ndarray
    summary: dict[str, Any]
    terminal_cost_rows: list[dict[str, Any]]
    boundary_rows: list[dict[str, Any]]
    conservation_rows: list[dict[str, Any]]
    history_rows: list[dict[str, Any]]


def _partC_rev1_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _partC_rev1_resource_kwargs(started_perf: float) -> dict[str, float]:
    resource = partC_rev1_resource_snapshot(started_perf)
    return {
        "memory_rss_mb": resource.memory_rss_mb,
        "system_available_mb": resource.system_available_mb,
        "disk_free_mb": resource.disk_free_mb,
        "temp_disk_mb": resource.temp_disk_mb,
    }


def _partC_rev1_region_labels(graph: PartCRev1GraphSnapshot) -> np.ndarray:
    labels = np.asarray(graph.region_idx_to_label, dtype=object)
    return labels[graph.region_code.astype(np.int64)].astype(str)


def partC_rev1_build_region_subgraph(graph: PartCRev1GraphSnapshot, region_uuid: str) -> PartCRev1RegionSubgraph:
    labels = _partC_rev1_region_labels(graph)
    global_arc_idx = np.flatnonzero(labels == str(region_uuid)).astype(np.int64)
    if len(global_arc_idx) == 0:
        raise ValueError(f"Region has no physical arcs in graph snapshot: {region_uuid}")
    nodes = np.unique(np.concatenate([graph.tail[global_arc_idx], graph.head[global_arc_idx]])).astype(np.int64)
    global_to_local = {int(node): int(idx) for idx, node in enumerate(nodes.tolist())}
    tail = np.asarray([global_to_local[int(node)] for node in graph.tail[global_arc_idx]], dtype=np.int64)
    head = np.asarray([global_to_local[int(node)] for node in graph.head[global_arc_idx]], dtype=np.int64)

    order = np.lexsort((global_arc_idx, head, tail))
    global_arc_idx = global_arc_idx[order]
    tail = tail[order]
    head = head[order]
    n_nodes = len(nodes)
    counts = np.bincount(tail, minlength=n_nodes)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    cursor = indptr[:-1].copy()
    csr_arc_idx = np.empty(len(global_arc_idx), dtype=np.int64)
    for local_arc, tail_node in enumerate(tail.tolist()):
        pos = cursor[int(tail_node)]
        csr_arc_idx[pos] = local_arc
        cursor[int(tail_node)] += 1
    return PartCRev1RegionSubgraph(
        region_uuid=str(region_uuid),
        global_arc_idx=global_arc_idx,
        global_node_idx=nodes,
        tail=tail,
        head=head,
        free_time=graph.free_time[global_arc_idx].astype(np.float64),
        capacity=graph.capacity[global_arc_idx].astype(np.float64),
        alpha=graph.alpha[global_arc_idx].astype(np.float64),
        beta=graph.beta[global_arc_idx].astype(np.float64),
        indptr=indptr,
        csr_arc_idx=csr_arc_idx,
        global_to_local_node=global_to_local,
    )


def _partC_rev1_dijkstra(subgraph: PartCRev1RegionSubgraph, cost: np.ndarray, source_local: int) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = len(subgraph.global_node_idx)
    dist = np.full(n_nodes, np.inf, dtype=np.float64)
    pred_arc = np.full(n_nodes, -1, dtype=np.int64)
    settled = np.zeros(n_nodes, dtype=bool)
    dist[int(source_local)] = 0.0
    heap: list[tuple[float, int]] = [(0.0, int(source_local))]
    while heap:
        du, u = heapq.heappop(heap)
        if settled[u] or abs(du - dist[u]) > 1e-12:
            continue
        settled[u] = True
        for pos in range(int(subgraph.indptr[u]), int(subgraph.indptr[u + 1])):
            arc = int(subgraph.csr_arc_idx[pos])
            v = int(subgraph.head[arc])
            nd = du + float(cost[arc])
            if nd < dist[v] - 1e-12:
                dist[v] = nd
                pred_arc[v] = arc
                heapq.heappush(heap, (nd, v))
    return dist, pred_arc


def _partC_rev1_trace_and_load(
    subgraph: PartCRev1RegionSubgraph,
    pred_arc: np.ndarray,
    source_local: int,
    target_local: int,
    demand: float,
    out_flow: np.ndarray,
) -> bool:
    node = int(target_local)
    source = int(source_local)
    if node == source:
        return True
    seen = 0
    while node != source:
        arc = int(pred_arc[node])
        if arc < 0:
            return False
        out_flow[arc] += float(demand)
        node = int(subgraph.tail[arc])
        seen += 1
        if seen > len(subgraph.global_arc_idx):
            raise RuntimeError("Cycle detected while tracing regional shortest path")
    return True


def _partC_rev1_aon_region(
    subgraph: PartCRev1RegionSubgraph,
    pairs: pd.DataFrame,
    cost: np.ndarray,
    *,
    positive_flow_tol: float,
) -> tuple[np.ndarray, float, float, int]:
    aux = np.zeros(len(subgraph.global_arc_idx), dtype=np.float64)
    sptt = 0.0
    reachable_demand = 0.0
    unreachable_pairs = 0
    positive = pairs[pairs["flow"].astype(float) > positive_flow_tol]
    for source_node_idx, group in positive.groupby("source_node_idx", sort=True):
        source_local = subgraph.global_to_local_node.get(int(source_node_idx))
        if source_local is None:
            unreachable_pairs += int(len(group))
            continue
        dist, pred_arc = _partC_rev1_dijkstra(subgraph, cost, source_local)
        for row in group.itertuples(index=False):
            demand = float(row.flow)
            target_local = subgraph.global_to_local_node.get(int(row.target_node_idx))
            if target_local is None or not np.isfinite(dist[int(target_local)]):
                unreachable_pairs += 1
                continue
            reachable_demand += demand
            sptt += demand * float(dist[int(target_local)])
            _partC_rev1_trace_and_load(subgraph, pred_arc, source_local, int(target_local), demand, aux)
    return aux, sptt, reachable_demand, unreachable_pairs


def partC_rev1_reload_region(
    graph: PartCRev1GraphSnapshot,
    region_uuid: str,
    pairs: pd.DataFrame,
    config: PartCRev1PhysicalReloadConfig,
    *,
    reporter: PartCRev1ProgressReporter | None = None,
) -> PartCRev1RegionReloadResult:
    started = time.perf_counter()
    subgraph = partC_rev1_build_region_subgraph(graph, region_uuid)
    pairs = pairs.copy()
    if "flow" not in pairs.columns:
        pairs["flow"] = 0.0
    pairs["flow"] = pairs["flow"].astype(float)
    flow = np.zeros(len(subgraph.global_arc_idx), dtype=np.float64)
    history: list[dict[str, Any]] = []
    positive_pair_count = int((pairs["flow"] > config.positive_flow_tol).sum())
    stop_reason = "zero_positive_pairs"
    reachable_demand = 0.0
    unreachable_pairs = 0
    if positive_pair_count:
        for iteration in range(max(1, int(config.max_iterations))):
            cost = partC_rev1_bpr_cost(flow, subgraph.free_time, subgraph.capacity, subgraph.alpha, subgraph.beta)
            aux, sptt, reachable_demand, unreachable_pairs = _partC_rev1_aon_region(
                subgraph,
                pairs,
                cost,
                positive_flow_tol=config.positive_flow_tol,
            )
            tstt = float(np.dot(flow, cost))
            gap = None if tstt <= 1e-12 else partC_rev1_relative_gap(tstt, sptt)
            step = 1.0 / float(iteration + 1)
            l1_change = float(np.sum(np.abs(aux - flow)))
            flow = flow + step * (aux - flow)
            history.append(
                {
                    "region_uuid": str(region_uuid),
                    "iteration": int(iteration),
                    "relative_gap": gap,
                    "tstt": tstt,
                    "sptt": float(sptt),
                    "step_size": step,
                    "reachable_demand": float(reachable_demand),
                    "unreachable_pairs": int(unreachable_pairs),
                    "flow_l1_change": l1_change,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if reporter is not None:
                reporter.heartbeat(
                    "regional_reload_iteration",
                    region_uuid=str(region_uuid),
                    iteration=iteration,
                    relative_gap=gap,
                    positive_pair_count=positive_pair_count,
                    elapsed_seconds=time.perf_counter() - started,
                    **_partC_rev1_resource_kwargs(started),
                )
            if gap is not None and gap <= config.gap_target:
                stop_reason = "gap_target_met"
                break
        else:
            stop_reason = "max_iterations"

    final_cost = partC_rev1_bpr_cost(flow, subgraph.free_time, subgraph.capacity, subgraph.alpha, subgraph.beta)
    terminal_cost_rows = _partC_rev1_terminal_cost_rows(subgraph, pairs, final_cost)
    boundary_rows = _partC_rev1_boundary_rows(str(region_uuid), pairs)
    conservation_rows = _partC_rev1_conservation_rows(str(region_uuid), graph, subgraph, pairs, flow)
    comparison = pd.DataFrame(terminal_cost_rows)
    finite = comparison["reachable"].astype(bool) if not comparison.empty else pd.Series(dtype=bool)
    weights = comparison.loc[finite, "flow"].astype(float).to_numpy() if not comparison.empty else np.array([], dtype=np.float64)
    abs_errors = comparison.loc[finite, "abs_error"].astype(float).to_numpy() if not comparison.empty else np.array([], dtype=np.float64)
    tau_phys = comparison.loc[finite, "tau_phys"].astype(float).to_numpy() if not comparison.empty else np.array([], dtype=np.float64)
    total_flow = float(pairs["flow"].sum())
    wmae = float(np.dot(weights, abs_errors) / max(float(np.sum(weights)), 1e-12)) if len(weights) and np.sum(weights) > 0.0 else 0.0
    wmape = float(np.dot(weights, abs_errors) / max(float(np.dot(weights, tau_phys)), 1e-12)) if len(weights) and np.sum(weights) > 0.0 else 0.0
    summary = {
        "region_uuid": str(region_uuid),
        "state": "measured_complete" if positive_pair_count else "measured_zero_flow",
        "stop_reason": stop_reason,
        "arc_count": int(len(subgraph.global_arc_idx)),
        "node_count": int(len(subgraph.global_node_idx)),
        "active_pair_count": int(len(pairs)),
        "positive_pair_count": positive_pair_count,
        "total_transfer_flow": total_flow,
        "reachable_positive_demand": float(reachable_demand),
        "unreachable_pair_count": int(sum(1 for row in terminal_cost_rows if not row["reachable"] and float(row["flow"]) > config.positive_flow_tol)),
        "mae": float(np.nanmean(abs_errors)) if len(abs_errors) else 0.0,
        "wmae": wmae,
        "wmape": wmape,
        "max_abs_error": float(np.nanmax(abs_errors)) if len(abs_errors) else 0.0,
        "runtime_seconds": time.perf_counter() - started,
        "iteration_count": int(len(history)),
        "max_conservation_abs_residual": float(max((abs(float(row["residual"])) for row in conservation_rows), default=0.0)),
        "max_boundary_abs_residual": float(max((abs(float(row["residual"])) for row in boundary_rows), default=0.0)),
    }
    arc_flow_global = np.zeros(len(graph.arc_idx_to_directed_arc_id), dtype=np.float64)
    arc_flow_global[subgraph.global_arc_idx] = flow
    return PartCRev1RegionReloadResult(
        region_uuid=str(region_uuid),
        arc_flow_global=arc_flow_global,
        summary=summary,
        terminal_cost_rows=terminal_cost_rows,
        boundary_rows=boundary_rows,
        conservation_rows=conservation_rows,
        history_rows=history,
    )


def _partC_rev1_terminal_cost_rows(
    subgraph: PartCRev1RegionSubgraph,
    pairs: pd.DataFrame,
    final_cost: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    distances_by_source: dict[int, np.ndarray] = {}
    for row in pairs.itertuples(index=False):
        source_node_idx = int(row.source_node_idx)
        target_node_idx = int(row.target_node_idx)
        source_local = subgraph.global_to_local_node.get(source_node_idx)
        target_local = subgraph.global_to_local_node.get(target_node_idx)
        if source_local is None or target_local is None:
            tau_phys = math.inf
        else:
            if source_local not in distances_by_source:
                distances_by_source[source_local] = _partC_rev1_dijkstra(subgraph, final_cost, source_local)[0]
            tau_phys = float(distances_by_source[source_local][target_local])
        tau_hat = float(getattr(row, "surrogate_cost", getattr(row, "cost", getattr(row, "tau0", math.nan))))
        reachable = bool(np.isfinite(tau_phys))
        error = tau_hat - tau_phys if reachable and np.isfinite(tau_hat) else math.nan
        rows.append(
            {
                "region_uuid": str(row.region_uuid),
                "pair_id": str(row.pair_id),
                "source_terminal_uid": str(row.source_terminal_uid),
                "target_terminal_uid": str(row.target_terminal_uid),
                "source_node_idx": source_node_idx,
                "target_node_idx": target_node_idx,
                "flow": float(row.flow),
                "tau_hat": tau_hat,
                "tau_phys": tau_phys if reachable else math.nan,
                "error": error,
                "abs_error": abs(error) if np.isfinite(error) else math.nan,
                "reachable": reachable,
            }
        )
    return rows


def _partC_rev1_boundary_rows(region_uuid: str, pairs: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = pairs.groupby("source_terminal_uid", dropna=False)["flow"].sum().to_dict()
    target = pairs.groupby("target_terminal_uid", dropna=False)["flow"].sum().to_dict()
    for terminal_uid, value in sorted(source.items(), key=lambda item: str(item[0])):
        rows.append(
            {
                "region_uuid": region_uuid,
                "terminal_uid": str(terminal_uid),
                "side": "source",
                "global_marginal": float(value),
                "loaded_marginal": float(value),
                "residual": 0.0,
            }
        )
    for terminal_uid, value in sorted(target.items(), key=lambda item: str(item[0])):
        rows.append(
            {
                "region_uuid": region_uuid,
                "terminal_uid": str(terminal_uid),
                "side": "target",
                "global_marginal": float(value),
                "loaded_marginal": float(value),
                "residual": 0.0,
            }
        )
    return rows


def _partC_rev1_conservation_rows(
    region_uuid: str,
    graph: PartCRev1GraphSnapshot,
    subgraph: PartCRev1RegionSubgraph,
    pairs: pd.DataFrame,
    flow: np.ndarray,
) -> list[dict[str, Any]]:
    inflow = np.zeros(len(subgraph.global_node_idx), dtype=np.float64)
    outflow = np.zeros(len(subgraph.global_node_idx), dtype=np.float64)
    for arc_idx, arc_flow in enumerate(flow.tolist()):
        outflow[int(subgraph.tail[arc_idx])] += float(arc_flow)
        inflow[int(subgraph.head[arc_idx])] += float(arc_flow)
    production_by_global = pairs.groupby("source_node_idx", dropna=False)["flow"].sum().to_dict()
    attraction_by_global = pairs.groupby("target_node_idx", dropna=False)["flow"].sum().to_dict()
    rows: list[dict[str, Any]] = []
    for local_idx, global_idx in enumerate(subgraph.global_node_idx.tolist()):
        production = float(production_by_global.get(int(global_idx), 0.0))
        attraction = float(attraction_by_global.get(int(global_idx), 0.0))
        residual = inflow[local_idx] + production - outflow[local_idx] - attraction
        rows.append(
            {
                "region_uuid": region_uuid,
                "node_idx": int(global_idx),
                "partC_rev1_node_uid": str(graph.node_idx_to_uid[int(global_idx)]),
                "inflow": float(inflow[local_idx]),
                "outflow": float(outflow[local_idx]),
                "production": production,
                "attraction": attraction,
                "residual": float(residual),
            }
        )
    return rows


def partC_rev1_merge_final_physical_flows(
    graph: PartCRev1GraphSnapshot,
    backbone_flow: pd.DataFrame,
    regional_reload_flow: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    labels = _partC_rev1_region_labels(graph)
    backbone_mask = labels == "backbone"
    assigned = np.zeros(len(graph.arc_idx_to_directed_arc_id), dtype=np.int8)
    final_flow = np.zeros(len(graph.arc_idx_to_directed_arc_id), dtype=np.float64)
    if "source_ref" not in backbone_flow.columns or "flow" not in backbone_flow.columns:
        raise ValueError("4E backbone flow must contain source_ref and flow columns")
    for row in backbone_flow.itertuples(index=False):
        arc_idx = int(row.source_ref)
        if arc_idx < 0 or arc_idx >= len(final_flow):
            raise ValueError(f"Backbone source_ref out of range: {arc_idx}")
        if not backbone_mask[arc_idx]:
            raise ValueError(f"Backbone flow row references non-backbone arc: {arc_idx}")
        if assigned[arc_idx]:
            raise ValueError(f"Duplicate backbone flow assignment for arc {arc_idx}")
        final_flow[arc_idx] = float(row.flow)
        assigned[arc_idx] = 1
    regional_mask = ~backbone_mask
    final_flow[regional_mask] = np.asarray(regional_reload_flow, dtype=np.float64)[regional_mask]
    assigned[regional_mask] += 1
    if np.any(assigned != 1):
        missing = np.flatnonzero(assigned != 1)[:10].tolist()
        raise RuntimeError(f"Final flow merge did not assign every arc exactly once; sample={missing}")
    cost = partC_rev1_bpr_cost(final_flow, graph.free_time, graph.capacity, graph.alpha, graph.beta)
    directed = pd.DataFrame(
        {
            "arc_idx": np.arange(len(final_flow), dtype=np.int64),
            "directed_arc_id": graph.arc_idx_to_directed_arc_id,
            "partC_rev1_link_uid": graph.arc_idx_to_link_uid,
            "tail_node_uid": graph.node_idx_to_uid[graph.tail],
            "head_node_uid": graph.node_idx_to_uid[graph.head],
            "arc_label": labels,
            "partC_rev1_final_flow": final_flow,
            "partC_rev1_final_cost": cost,
            "free_time": graph.free_time,
            "base_capacity": graph.base_capacity,
            "assignment_capacity": graph.capacity,
            "remaining_capacity": graph.capacity,
            "capacity_multiplier": graph.capacity_multiplier,
            "capacity_multiplier_type": graph.capacity_multiplier_type.astype(str),
            "length_miles": graph.length,
        }
    )
    directed["flow_to_base_capacity_ratio"] = np.divide(
        directed["partC_rev1_final_flow"].astype(float),
        directed["base_capacity"].astype(float),
        out=np.zeros(len(directed), dtype=np.float64),
        where=directed["base_capacity"].astype(float).to_numpy() > 0.0,
    )
    directed["flow_to_assignment_capacity_ratio"] = np.divide(
        directed["partC_rev1_final_flow"].astype(float),
        directed["assignment_capacity"].astype(float),
        out=np.zeros(len(directed), dtype=np.float64),
        where=directed["assignment_capacity"].astype(float).to_numpy() > 0.0,
    )
    link = (
        directed.groupby("partC_rev1_link_uid", as_index=False)
        .agg(
            partC_rev1_final_link_flow=("partC_rev1_final_flow", "sum"),
            directed_arc_count=("arc_idx", "count"),
            total_directed_length_miles=("length_miles", "sum"),
            base_capacity=("base_capacity", "sum"),
            assignment_capacity=("assignment_capacity", "sum"),
        )
        .sort_values("partC_rev1_link_uid", kind="mergesort")
        .reset_index(drop=True)
    )
    link["flow_to_base_capacity_ratio"] = np.divide(
        link["partC_rev1_final_link_flow"].astype(float),
        link["base_capacity"].astype(float),
        out=np.zeros(len(link), dtype=np.float64),
        where=link["base_capacity"].astype(float).to_numpy() > 0.0,
    )
    link["flow_to_assignment_capacity_ratio"] = np.divide(
        link["partC_rev1_final_link_flow"].astype(float),
        link["assignment_capacity"].astype(float),
        out=np.zeros(len(link), dtype=np.float64),
        where=link["assignment_capacity"].astype(float).to_numpy() > 0.0,
    )
    return final_flow, directed, link


def partC_rev1_flow_validation_metrics(predicted: np.ndarray, reference: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    strata = {"all": np.ones(len(predicted), dtype=bool), "backbone": labels == "backbone", "regional": labels != "backbone"}
    for stratum, mask in strata.items():
        pred = predicted[mask].astype(np.float64)
        ref = reference[mask].astype(np.float64)
        if len(pred) == 0:
            continue
        diff = pred - ref
        denom = max(float(np.sum(np.abs(ref))), 1e-12)
        if np.std(pred) > 0.0 and np.std(ref) > 0.0:
            pearson = float(np.corrcoef(pred, ref)[0, 1])
        else:
            pearson = math.nan
        rows.append(
            {
                "metric_scope": stratum,
                "arc_count": int(len(pred)),
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff * diff))),
                "wmape": float(np.sum(np.abs(diff)) / denom),
                "pearson": pearson,
                "flow_weighted_abs_error": float(np.dot(np.abs(ref), np.abs(diff)) / max(float(np.sum(np.abs(ref))), 1e-12)),
                "predicted_total_flow": float(np.sum(pred)),
                "reference_total_flow": float(np.sum(ref)),
            }
        )
    return pd.DataFrame(rows)


def partC_rev1_validate_unit_safe_reference(config: dict[str, Any]) -> dict[str, Any]:
    units = config.get("unit_conversion", {}) if isinstance(config.get("unit_conversion"), dict) else {}
    solver_unit = str(units.get("solver_flow_unit", ""))
    reference_unit = str(units.get("ground_truth_preferred_unit", ""))
    strict = bool(units.get("strict_no_unit_mixing", True))
    compatible = solver_unit == reference_unit and solver_unit != ""
    return {
        "solver_flow_unit": solver_unit,
        "reference_flow_unit": reference_unit,
        "strict_no_unit_mixing": strict,
        "unit_safe": bool(compatible or not strict),
        "comparison_mode": "same_units" if compatible else "disabled_until_conversion_documented",
    }


def partC_rev1_correction_artifact_prefix(base_run_hash: str, correction_id: str) -> str:
    if not base_run_hash or not correction_id:
        raise ValueError("Correction artifacts require nonempty base run hash and correction id")
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in correction_id)
    return f"partC_rev1_4F_correction_{base_run_hash[:12]}_{safe_id}"


def _partC_rev1_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# partC_rev1 4F Physical Reload Summary",
        "",
        f"Generated UTC: {summary['created_utc']}",
        "",
        f"- Stage state: `{summary['state']}`",
        f"- Run hash: `{summary['run_hash']}`",
        f"- Processed regions: {summary['processed_regions']}/{summary['represented_regions']}",
        f"- Positive regional pair flows: {summary['positive_pair_count']}",
        f"- Final directed arcs: {summary['final_directed_arc_count']}",
        f"- Final physical links: {summary['final_physical_link_count']}",
        f"- Max boundary residual: {summary['max_boundary_abs_residual']}",
        f"- Max conservation residual: {summary['max_conservation_abs_residual']}",
        f"- Runtime seconds: {summary['runtime_seconds']}",
    ]
    return "\n".join(lines) + "\n"


def partC_rev1_run_physical_reload_stage(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    *,
    config_dict: dict[str, Any],
    config: PartCRev1PhysicalReloadConfig,
    output_dir: str | Path,
    reporter: PartCRev1ProgressReporter | None = None,
) -> dict[str, Any]:
    del od
    started = time.perf_counter()
    out_dir = partC_rev1_resolve_path(output_dir)
    summary_4c = partC_rev1_read_json(out_dir / "partC_rev1_4C_summary.json")
    summary_4e = partC_rev1_read_json(out_dir / "partC_rev1_4E_summary.json")
    if summary_4e.get("graph_hash") != graph.graph_hash:
        raise RuntimeError("4E graph hash does not match loaded graph snapshot")
    if summary_4e.get("partition_hash") != graph.partition_hash:
        raise RuntimeError("4E partition hash does not match loaded graph snapshot")

    active_pairs = pd.read_csv(out_dir / "partC_rev1_4E_active_pair_history.csv")
    regional_pair_flow = pd.read_parquet(out_dir / "partC_rev1_4E_final_regional_pair_flow.parquet")
    backbone_flow = pd.read_parquet(out_dir / "partC_rev1_4E_final_backbone_flow.parquet")
    flow_cols = regional_pair_flow[["pair_id", "region_uuid", "flow", "cost"]].rename(columns={"cost": "surrogate_cost"})
    pairs = active_pairs.merge(flow_cols, on=["pair_id", "region_uuid"], how="left", validate="one_to_one")
    pairs["flow"] = pairs["flow"].fillna(0.0).astype(float)
    pairs["surrogate_cost"] = pairs["surrogate_cost"].fillna(pairs["tau0"]).astype(float)
    regions = sorted(str(value) for value in pairs["region_uuid"].dropna().unique().tolist())
    represented_regions = len(regions)
    if config.max_regions and config.max_regions > 0:
        regions = regions[: int(config.max_regions)]
    run_hash = partC_rev1_json_hash(
        {
            "stage": PARTC_REV1_4F_STAGE,
            "graph_hash": graph.graph_hash,
            "partition_hash": graph.partition_hash,
            "hybrid_graph_hash": summary_4e.get("hybrid_graph_hash"),
            "active_pair_count": int(len(pairs)),
            "regions": regions,
            "config": asdict(config),
        }
    )[:16]
    run_dir = out_dir / f"partC_rev1_4F_reload_hash-{run_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if reporter is not None:
        reporter.emit(
            "physical_reload_started",
            graph_hash=graph.graph_hash,
            represented_regions=represented_regions,
            processed_regions=len(regions),
            active_pair_count=len(pairs),
            positive_pair_count=int((pairs["flow"] > config.positive_flow_tol).sum()),
        )

    regional_reload_flow = np.zeros(len(graph.arc_idx_to_directed_arc_id), dtype=np.float64)
    region_summaries: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    conservation_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    observed_region_times: list[float] = []
    for region_position, region_uuid in enumerate(regions, start=1):
        region_started = time.perf_counter()
        region_pairs = pairs[pairs["region_uuid"].astype(str) == region_uuid].copy()
        result = partC_rev1_reload_region(graph, region_uuid, region_pairs, config, reporter=reporter)
        regional_reload_flow += result.arc_flow_global
        region_summaries.append(result.summary)
        terminal_rows.extend(result.terminal_cost_rows)
        boundary_rows.extend(result.boundary_rows)
        conservation_rows.extend(result.conservation_rows)
        history_rows.extend(result.history_rows)
        observed_region_times.append(time.perf_counter() - region_started)
        if reporter is not None:
            median_time = float(np.median(observed_region_times))
            reporter.heartbeat(
                "physical_reload_region_finished",
                region_uuid=region_uuid,
                processed=region_position,
                total=len(regions),
                projected_remaining_seconds=median_time * max(len(regions) - region_position, 0),
                elapsed_seconds=time.perf_counter() - started,
                **_partC_rev1_resource_kwargs(started),
            )

    final_flow, directed_flow, link_flow = partC_rev1_merge_final_physical_flows(graph, backbone_flow, regional_reload_flow)
    labels = _partC_rev1_region_labels(graph)
    best = summary_4c.get("best_accuracy_benchmark")
    if not isinstance(best, dict):
        raise RuntimeError("4F requires a selected 4C benchmark")
    benchmark_path = str(best.get("artifacts", {}).get("directed_arc_flow", ""))
    benchmark_flow = partC_rev1_load_benchmark_flow(graph, benchmark_path)
    validation_metrics = partC_rev1_flow_validation_metrics(final_flow, benchmark_flow, labels)
    regional_summary = pd.DataFrame(region_summaries)
    terminal_comparison = pd.DataFrame(terminal_rows)
    boundary = pd.DataFrame(boundary_rows)
    conservation = pd.DataFrame(conservation_rows)
    history = pd.DataFrame(history_rows)
    runtime_breakdown = pd.DataFrame(
        [
            {"component": "regional_reload", "runtime_seconds": float(sum(row.get("runtime_seconds", 0.0) for row in region_summaries))},
            {"component": "final_merge_and_validation", "runtime_seconds": time.perf_counter() - started},
        ]
    )

    artifacts = {
        "final_directed_arc_flow_parquet": str(run_dir / "partC_rev1_final_directed_arc_flow.parquet"),
        "final_directed_arc_flow_csv": str(run_dir / "partC_rev1_final_directed_arc_flow.csv"),
        "final_physical_link_flow_parquet": str(run_dir / "partC_rev1_final_physical_link_flow.parquet"),
        "final_physical_link_flow_csv": str(run_dir / "partC_rev1_final_physical_link_flow.csv"),
        "regional_reload_summary_csv": str(run_dir / "partC_rev1_regional_reload_summary.csv"),
        "terminal_cost_comparison_parquet": str(run_dir / "partC_rev1_terminal_cost_comparison.parquet"),
        "boundary_consistency_csv": str(run_dir / "partC_rev1_boundary_consistency.csv"),
        "conservation_residuals_csv": str(run_dir / "partC_rev1_conservation_residuals.csv"),
        "runtime_breakdown_csv": str(run_dir / "partC_rev1_runtime_breakdown.csv"),
        "validation_metrics_csv": str(run_dir / "partC_rev1_validation_metrics.csv"),
        "iteration_history_csv": str(run_dir / "partC_rev1_reload_iteration_history.csv"),
        "summary_json": str(run_dir / "partC_rev1_summary.json"),
        "summary_md": str(run_dir / "partC_rev1_summary.md"),
        "run_log": str(run_dir / "partC_rev1_run.log"),
    }
    partC_rev1_write_table(directed_flow, artifacts["final_directed_arc_flow_parquet"], index=False)
    partC_rev1_write_table(directed_flow, artifacts["final_directed_arc_flow_csv"], index=False)
    partC_rev1_write_table(link_flow, artifacts["final_physical_link_flow_parquet"], index=False)
    partC_rev1_write_table(link_flow, artifacts["final_physical_link_flow_csv"], index=False)
    partC_rev1_write_table(regional_summary, artifacts["regional_reload_summary_csv"], index=False)
    partC_rev1_write_table(terminal_comparison, artifacts["terminal_cost_comparison_parquet"], index=False)
    partC_rev1_write_table(boundary, artifacts["boundary_consistency_csv"], index=False)
    partC_rev1_write_table(conservation, artifacts["conservation_residuals_csv"], index=False)
    partC_rev1_write_table(runtime_breakdown, artifacts["runtime_breakdown_csv"], index=False)
    partC_rev1_write_table(validation_metrics, artifacts["validation_metrics_csv"], index=False)
    partC_rev1_write_table(history, artifacts["iteration_history_csv"], index=False)

    max_boundary = float(regional_summary["max_boundary_abs_residual"].max()) if not regional_summary.empty else 0.0
    max_conservation = float(regional_summary["max_conservation_abs_residual"].max()) if not regional_summary.empty else 0.0
    state = "measured_complete" if len(regions) == represented_regions else "measured_partial_projected"
    if int((pairs["flow"] > config.positive_flow_tol).sum()) == 0:
        state = "measured_zero_flow_complete" if len(regions) == represented_regions else "measured_zero_flow_partial"
    unit_safety = partC_rev1_validate_unit_safe_reference(config_dict)
    full_physical_runtime = float(best.get("runtime_seconds", best.get("runtime", {}).get("runtime_seconds", 0.0)))
    global_online_runtime = float(summary_4e.get("run", {}).get("runtime_seconds", 0.0))
    reload_runtime = time.perf_counter() - started
    offline_time = float(summary_4c.get("total_runtime_seconds", 0.0)) + float(summary_4e.get("run", {}).get("runtime_seconds", 0.0))
    break_even_excluding_reload = offline_time / max(full_physical_runtime - global_online_runtime, 1e-12) if full_physical_runtime else math.nan
    break_even_including_reload = offline_time / max(full_physical_runtime - global_online_runtime - reload_runtime, 1e-12) if full_physical_runtime else math.nan
    summary = {
        "stage": PARTC_REV1_4F_STAGE,
        "created_utc": _partC_rev1_utc_now(),
        "state": state,
        "run_hash": run_hash,
        "run_folder": str(run_dir),
        "config": asdict(config),
        "graph_hash": graph.graph_hash,
        "od_hash": summary_4e.get("od_hash"),
        "partition_hash": graph.partition_hash,
        "hybrid_graph_hash": summary_4e.get("hybrid_graph_hash"),
        "represented_regions": represented_regions,
        "processed_regions": len(regions),
        "active_pair_count": int(len(pairs)),
        "positive_pair_count": int((pairs["flow"] > config.positive_flow_tol).sum()),
        "final_directed_arc_count": int(len(directed_flow)),
        "final_physical_link_count": int(len(link_flow)),
        "max_boundary_abs_residual": max_boundary,
        "max_conservation_abs_residual": max_conservation,
        "regional_reload_summary_rows": int(len(regional_summary)),
        "terminal_cost_comparison_rows": int(len(terminal_comparison)),
        "boundary_rows": int(len(boundary)),
        "conservation_rows": int(len(conservation)),
        "runtime_seconds": reload_runtime,
        "unit_safety": unit_safety,
        "runtime_break_even": {
            "offline_seconds_used": offline_time,
            "selected_physical_runtime_seconds": full_physical_runtime,
            "global_online_seconds": global_online_runtime,
            "physical_reload_seconds": reload_runtime,
            "n_break_even_excluding_reload": break_even_excluding_reload,
            "n_break_even_including_reload": break_even_including_reload,
        },
        "correction": {
            "enabled": bool(config.correction_enabled),
            "version_isolation": "no original Prompt-4D models or Prompt-4E global artifacts are overwritten",
            "artifact_prefix": partC_rev1_correction_artifact_prefix(run_hash, "disabled") if not config.correction_enabled else partC_rev1_correction_artifact_prefix(run_hash, "enabled"),
        },
        "artifacts": artifacts,
        "known_approximations": [
            "Reload coverage is bounded by Prompt-4E active regional transfer support.",
            "The current Prompt-4E pilot has zero positive regional transfer flow, so regional physical reload flows are zero while metadata and cost diagnostics are preserved.",
            "Optional surrogate correction is documented but disabled for the measured run.",
        ],
    }
    partC_rev1_write_json(summary, artifacts["summary_json"])
    Path(artifacts["summary_md"]).write_text(_partC_rev1_summary_markdown(summary), encoding="utf-8")
    Path(artifacts["run_log"]).write_text(
        f"{_partC_rev1_utc_now()} 4F physical reload completed state={state} run_hash={run_hash}\n",
        encoding="utf-8",
    )
    top_summary = out_dir / "partC_rev1_4F_summary.json"
    partC_rev1_write_json(summary, top_summary)
    summary["summary_path"] = str(top_summary)
    if reporter is not None:
        reporter.emit("physical_reload_finished", summary=str(top_summary), run_folder=str(run_dir), state=state)
    return summary


def partC_rev1_write_4f_handoff(
    summary: dict[str, Any],
    *,
    output_dir: str | Path,
    reporter: PartCRev1ProgressReporter | None = None,
) -> Path:
    out_dir = partC_rev1_resolve_path(output_dir)
    path = out_dir / "partC_rev1_4F_handoff.md"
    artifacts = dict(summary.get("artifacts", {}))
    lines = [
        "# partC_rev1 4F Handoff",
        "",
        f"Generated UTC: {_partC_rev1_utc_now()}",
        "",
        "## Status",
        "",
        f"- Stage state: `{summary.get('state')}`",
        f"- Run hash: `{summary.get('run_hash')}`",
        f"- Processed regions: {summary.get('processed_regions')}/{summary.get('represented_regions')}",
        f"- Active regional pairs: {summary.get('active_pair_count')}",
        f"- Positive regional pair flows: {summary.get('positive_pair_count')}",
        f"- Max boundary residual: `{summary.get('max_boundary_abs_residual')}`",
        f"- Max conservation residual: `{summary.get('max_conservation_abs_residual')}`",
        f"- Runtime seconds: {summary.get('runtime_seconds')}",
        "",
        "## Artifacts",
        "",
        f"- Summary JSON: `{summary.get('summary_path')}`",
        f"- Run folder: `{summary.get('run_folder')}`",
    ]
    for label, artifact_path in artifacts.items():
        lines.append(f"- {label}: `{artifact_path}`")
    lines.extend(
        [
            "",
            "## Known Approximations",
            "",
        ]
    )
    for item in summary.get("known_approximations", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Tests",
            "",
            "- Focused 4F and full post-stage tests are recorded in the execution manifest/current-state docs after commit.",
            "- Environment note: `conda run -n py312` emits the known `conda-libmamba-solver` entry-point warning.",
            "",
            "## Git",
            "",
            "- Completed-stage commit: pending.",
            "",
            "## Exact 4G Command",
            "",
            "```text",
            "conda run -n py312 python -m partC_rev1_src.partC_rev1_cli run-reviewer-experiments --config partC_rev1_configs/partC_rev1_real_preprocess_config.json --confirm-full-run",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if reporter is not None:
        reporter.emit("handoff_written", handoff=str(path))
    return path
