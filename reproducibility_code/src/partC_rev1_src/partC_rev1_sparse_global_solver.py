"""Prompt 4E sparse adaptive hybrid/global solver."""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .partC_rev1_aon import partC_rev1_aon_load
from .partC_rev1_bpr import partC_rev1_bpr_cost, partC_rev1_fw_line_search, partC_rev1_relative_gap
from .partC_rev1_graph_snapshot import PartCRev1GraphSnapshot
from .partC_rev1_hashing import partC_rev1_json_hash, partC_rev1_numpy_array_hash, partC_rev1_sha256_text
from .partC_rev1_io import partC_rev1_write_json, partC_rev1_write_table
from .partC_rev1_od_snapshot import PartCRev1ODSnapshot
from .partC_rev1_paths import partC_rev1_resolve_path
from .partC_rev1_progress_logging import PartCRev1ProgressReporter
from .partC_rev1_regional_training import (
    PartCRev1RegionalTrainingConfig,
    partC_rev1_build_region_problem,
    partC_rev1_incident_label_counts,
    partC_rev1_load_benchmark_flow,
    partC_rev1_od_marginals_by_node,
    partC_rev1_read_route_witnesses,
    partC_rev1_centroid_node_mask,
)
from .partC_rev1_runtime_guard import PARTC_REV1_DEFAULT_MAX_RUN_SECONDS, partC_rev1_resource_snapshot
from .partC_rev1_shortest_paths import partC_rev1_sssp


PARTC_REV1_4E_STAGE = "partC_rev1_4E"


def _partC_rev1_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class PartCRev1SparseGlobalConfig:
    config_path: str
    max_iterations: int = 3
    master_gap_tol: float = 1e-3
    pricing_gap_tol: float = 1e-3
    price_tol: float = 1e-6
    pilot_origins: int = 16
    origin_batch_size: int = 8
    top_k_pricing: int = 4
    lazy_abs_tol: float = 1e-6
    lazy_rel_tol: float = 1e-4
    max_stale_iterations: int = 2
    full_refresh_frequency: int = 2
    regional_cost_slope: float = 1e-6
    assignment_algorithm: str = "fw"
    runtime_guard_seconds: float = PARTC_REV1_DEFAULT_MAX_RUN_SECONDS


@dataclass(slots=True)
class PartCRev1HybridGraphBundle:
    graph: PartCRev1GraphSnapshot
    od: PartCRev1ODSnapshot
    active_arc_table: pd.DataFrame
    active_pair_table: pd.DataFrame
    physical_node_to_hybrid_idx: np.ndarray
    origin_role_by_physical_idx: dict[int, int]
    destination_role_by_physical_idx: dict[int, int]
    backbone_arc_mask: np.ndarray
    regional_arc_mask: np.ndarray
    connector_arc_mask: np.ndarray


def _partC_rev1_labels(graph: PartCRev1GraphSnapshot) -> np.ndarray:
    return np.asarray([graph.region_idx_to_label[int(code)] for code in graph.region_code], dtype=object)


def _partC_rev1_role_uid(role: str, node_uid: str) -> str:
    return f"partC_rev1_{role}_role|{node_uid}"


def partC_rev1_build_active_pairs_from_regions(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    *,
    benchmark_flow: np.ndarray,
    route_witnesses: pd.DataFrame,
    completed_regions: Iterable[dict[str, Any]],
    regional_config: PartCRev1RegionalTrainingConfig,
    reporter: PartCRev1ProgressReporter | None = None,
) -> pd.DataFrame:
    started = time.perf_counter()
    incident = partC_rev1_incident_label_counts(graph)
    centroid_mask = partC_rev1_centroid_node_mask(graph, od)
    od_source, od_target = partC_rev1_od_marginals_by_node(graph, od)
    rows: list[dict[str, Any]] = []
    completed_region_list = list(completed_regions)
    total_regions = len(completed_region_list)
    for region_position, region_record in enumerate(completed_region_list, start=1):
        region_uuid = str(region_record["region_uuid"])
        problem = partC_rev1_build_region_problem(
            graph,
            od,
            region_uuid=region_uuid,
            benchmark_flow=benchmark_flow,
            config=regional_config,
            incident_label_count=incident,
            centroid_mask=centroid_mask,
            od_source_marginal=od_source,
            od_target_marginal=od_target,
            route_witnesses=route_witnesses,
        )
        for support in problem.support.itertuples(index=False):
            s_idx = int(support.source_idx)
            t_idx = int(support.target_idx)
            source = problem.source_terminals.loc[s_idx]
            target = problem.target_terminals.loc[t_idx]
            rows.append(
                {
                    "active_pair_idx": len(rows),
                    "region_uuid": region_uuid,
                    "pair_id": str(support.pair_id),
                    "source_terminal_uid": str(support.source_terminal_uid),
                    "target_terminal_uid": str(support.target_terminal_uid),
                    "source_terminal_kind": str(source["terminal_kind"]),
                    "target_terminal_kind": str(target["terminal_kind"]),
                    "source_node_uid": str(source["node_uid"]),
                    "target_node_uid": str(target["node_uid"]),
                    "source_node_idx": int(source["node_idx"]),
                    "target_node_idx": int(target["node_idx"]),
                    "tau0": float(support.tau0),
                    "pair_type": str(support.pair_type),
                    "active_status": "active",
                    "candidate_superset_status": "active_initial",
                    "activation_reason": "prompt4d_support",
                    "activation_iteration": 0,
                    "region_hash": problem.region_hash,
                    "terminal_order_hash": problem.terminal_order_hash,
                    "pair_order_hash": problem.pair_order_hash,
                }
            )
        if reporter is not None:
            resource = partC_rev1_resource_snapshot(started)
            reporter.heartbeat(
                "active_pair_region_built",
                region_uuid=region_uuid,
                processed=region_position,
                total=total_regions,
                active_pair_rows=len(rows),
                elapsed_seconds=time.perf_counter() - started,
                memory_rss_mb=resource.memory_rss_mb,
                temp_disk_mb=resource.temp_disk_mb,
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "active_pair_idx",
                "region_uuid",
                "pair_id",
                "source_terminal_uid",
                "target_terminal_uid",
                "source_terminal_kind",
                "target_terminal_kind",
                "source_node_uid",
                "target_node_uid",
                "source_node_idx",
                "target_node_idx",
                "tau0",
                "pair_type",
                "active_status",
                "candidate_superset_status",
                "activation_reason",
                "activation_iteration",
                "region_hash",
                "terminal_order_hash",
                "pair_order_hash",
            ]
        )
    return pd.DataFrame(rows)


def partC_rev1_conceptual_pair_estimate(active_pairs: pd.DataFrame) -> int:
    if active_pairs.empty:
        return 0
    total = 0
    for _region, group in active_pairs.groupby("region_uuid", sort=False):
        total += int(group["source_terminal_uid"].nunique() * group["target_terminal_uid"].nunique())
    return int(total)


def partC_rev1_build_hybrid_graph(
    physical_graph: PartCRev1GraphSnapshot,
    physical_od: PartCRev1ODSnapshot,
    active_pairs: pd.DataFrame,
) -> PartCRev1HybridGraphBundle:
    labels = _partC_rev1_labels(physical_graph)
    backbone_indices = np.flatnonzero(labels == "backbone").astype(np.int64)
    centroid_nodes = np.unique(np.concatenate([physical_od.origin_node_idx, physical_od.destination_node_idx])).astype(np.int64)
    node_uids = physical_graph.node_idx_to_uid.astype(str).tolist()
    origin_role_by_physical: dict[int, int] = {}
    destination_role_by_physical: dict[int, int] = {}
    for node in centroid_nodes.tolist():
        origin_role_by_physical[int(node)] = len(node_uids)
        node_uids.append(_partC_rev1_role_uid("origin", str(physical_graph.node_idx_to_uid[int(node)])))
        destination_role_by_physical[int(node)] = len(node_uids)
        node_uids.append(_partC_rev1_role_uid("destination", str(physical_graph.node_idx_to_uid[int(node)])))
    arc_rows: list[dict[str, Any]] = []
    for arc_idx in backbone_indices.tolist():
        arc_rows.append(
            {
                "kind": "backbone",
                "tail": int(physical_graph.tail[int(arc_idx)]),
                "head": int(physical_graph.head[int(arc_idx)]),
                "source_ref": int(arc_idx),
                "stable_arc_id": str(physical_graph.arc_idx_to_directed_arc_id[int(arc_idx)]),
                "partC_rev1_link_uid": str(physical_graph.arc_idx_to_link_uid[int(arc_idx)]),
                "free_time": float(physical_graph.free_time[int(arc_idx)]),
                "base_capacity": float(physical_graph.base_capacity[int(arc_idx)]),
                "capacity": float(physical_graph.capacity[int(arc_idx)]),
                "capacity_multiplier": float(physical_graph.capacity_multiplier[int(arc_idx)]),
                "capacity_multiplier_type": str(physical_graph.capacity_multiplier_type[int(arc_idx)]),
                "alpha": float(physical_graph.alpha[int(arc_idx)]),
                "beta": float(physical_graph.beta[int(arc_idx)]),
                "length": float(physical_graph.length[int(arc_idx)]),
                "region_uuid": "backbone",
                "pair_id": "",
            }
        )
    for physical_idx in centroid_nodes.tolist():
        node_uid = str(physical_graph.node_idx_to_uid[int(physical_idx)])
        arc_rows.append(
            {
                "kind": "origin_connector",
                "tail": int(origin_role_by_physical[int(physical_idx)]),
                "head": int(physical_idx),
                "source_ref": int(physical_idx),
                "stable_arc_id": f"origin_connector|{node_uid}",
                "partC_rev1_link_uid": f"origin_connector|{node_uid}",
                "free_time": 0.0,
                "base_capacity": 1e18,
                "capacity": 1e18,
                "capacity_multiplier": 1.0,
                "capacity_multiplier_type": "synthetic_connector_unscaled",
                "alpha": 0.0,
                "beta": 1.0,
                "length": 0.0,
                "region_uuid": "connector",
                "pair_id": "",
            }
        )
        arc_rows.append(
            {
                "kind": "destination_connector",
                "tail": int(physical_idx),
                "head": int(destination_role_by_physical[int(physical_idx)]),
                "source_ref": int(physical_idx),
                "stable_arc_id": f"destination_connector|{node_uid}",
                "partC_rev1_link_uid": f"destination_connector|{node_uid}",
                "free_time": 0.0,
                "base_capacity": 1e18,
                "capacity": 1e18,
                "capacity_multiplier": 1.0,
                "capacity_multiplier_type": "synthetic_connector_unscaled",
                "alpha": 0.0,
                "beta": 1.0,
                "length": 0.0,
                "region_uuid": "connector",
                "pair_id": "",
            }
        )
    for pair in active_pairs.itertuples(index=False):
        source_physical = int(pair.source_node_idx)
        target_physical = int(pair.target_node_idx)
        tail = origin_role_by_physical[source_physical] if str(pair.source_terminal_kind) == "centroid_origin" else source_physical
        head = destination_role_by_physical[target_physical] if str(pair.target_terminal_kind) == "centroid_destination" else target_physical
        arc_rows.append(
            {
                "kind": "regional_transfer",
                "tail": int(tail),
                "head": int(head),
                "source_ref": int(pair.active_pair_idx),
                "stable_arc_id": str(pair.pair_id),
                "partC_rev1_link_uid": str(pair.pair_id),
                "free_time": float(pair.tau0),
                "base_capacity": 1e18,
                "capacity": 1e18,
                "capacity_multiplier": 1.0,
                "capacity_multiplier_type": "regional_transfer_unscaled",
                "alpha": 0.0,
                "beta": 1.0,
                "length": 0.0,
                "region_uuid": str(pair.region_uuid),
                "pair_id": str(pair.pair_id),
            }
        )
    arc_df = pd.DataFrame(arc_rows)
    arc_df = arc_df.sort_values(["tail", "stable_arc_id"], kind="mergesort").reset_index(drop=True)
    tail = arc_df["tail"].to_numpy(dtype=np.int64)
    head = arc_df["head"].to_numpy(dtype=np.int64)
    counts = np.bincount(tail, minlength=len(node_uids))
    csr_indptr = np.concatenate(([0], np.cumsum(counts, dtype=np.int64))).astype(np.int64)
    csr_arc_idx = np.arange(len(arc_df), dtype=np.int64)
    reverse_order = np.lexsort((arc_df["stable_arc_id"].astype(str).to_numpy(), tail, head)).astype(np.int64)
    reverse_counts = np.bincount(head, minlength=len(node_uids))
    reverse_csr_indptr = np.concatenate(([0], np.cumsum(reverse_counts, dtype=np.int64))).astype(np.int64)
    label_order = ["backbone", "connector", "regional_transfer"]
    label_to_code = {label: idx for idx, label in enumerate(label_order)}
    region_code = arc_df["kind"].map(lambda value: label_to_code["backbone"] if value == "backbone" else label_to_code["connector"] if "connector" in value else label_to_code["regional_transfer"]).to_numpy(dtype=np.int32)
    graph_hash = partC_rev1_json_hash(
        {
            "physical_graph_hash": physical_graph.graph_hash,
            "active_pair_hash": partC_rev1_json_hash(active_pairs["pair_id"].astype(str).tolist()) if not active_pairs.empty else "",
            "arc_count": int(len(arc_df)),
            "node_count": int(len(node_uids)),
        }
    )
    hybrid_graph = PartCRev1GraphSnapshot(
        graph_hash=graph_hash,
        partition_hash=physical_graph.partition_hash,
        node_idx_to_uid=np.asarray(node_uids, dtype=str),
        arc_idx_to_directed_arc_id=arc_df["stable_arc_id"].astype(str).to_numpy(),
        arc_idx_to_link_uid=arc_df["partC_rev1_link_uid"].astype(str).to_numpy(),
        tail=tail,
        head=head,
        free_time=arc_df["free_time"].to_numpy(dtype=np.float64),
        base_capacity=arc_df["base_capacity"].to_numpy(dtype=np.float64),
        capacity=arc_df["capacity"].to_numpy(dtype=np.float64),
        capacity_multiplier=arc_df["capacity_multiplier"].to_numpy(dtype=np.float64),
        capacity_multiplier_type=arc_df["capacity_multiplier_type"].astype(str).to_numpy(),
        alpha=arc_df["alpha"].to_numpy(dtype=np.float64),
        beta=arc_df["beta"].to_numpy(dtype=np.float64),
        length=arc_df["length"].to_numpy(dtype=np.float64),
        region_code=region_code,
        region_idx_to_label=label_order,
        csr_indptr=csr_indptr,
        csr_head=head.copy(),
        csr_arc_idx=csr_arc_idx,
        reverse_csr_indptr=reverse_csr_indptr,
        reverse_csr_head=tail[reverse_order].astype(np.int64),
        reverse_csr_arc_idx=reverse_order.astype(np.int64),
        metadata={"stage": PARTC_REV1_4E_STAGE, "physical_graph_hash": physical_graph.graph_hash},
    )
    origin = np.asarray([origin_role_by_physical[int(idx)] for idx in physical_od.origin_node_idx.tolist()], dtype=np.int64)
    dest = np.asarray([destination_role_by_physical[int(idx)] for idx in physical_od.destination_node_idx.tolist()], dtype=np.int64)
    hybrid_od = PartCRev1ODSnapshot(
        od_hash=partC_rev1_numpy_array_hash(origin, dest, physical_od.demand),
        origin_node_idx=origin,
        destination_node_idx=dest,
        demand=physical_od.demand.copy(),
        origin_group_node_idx=np.asarray([origin_role_by_physical[int(idx)] for idx in physical_od.origin_group_node_idx.tolist()], dtype=np.int64),
        origin_offsets=physical_od.origin_offsets.copy(),
        metadata={**physical_od.metadata, "stage": PARTC_REV1_4E_STAGE, "physical_od_hash": physical_od.od_hash},
    )
    return PartCRev1HybridGraphBundle(
        graph=hybrid_graph,
        od=hybrid_od,
        active_arc_table=arc_df.reset_index(names="hybrid_arc_idx"),
        active_pair_table=active_pairs.copy(),
        physical_node_to_hybrid_idx=np.arange(len(physical_graph.node_idx_to_uid), dtype=np.int64),
        origin_role_by_physical_idx=origin_role_by_physical,
        destination_role_by_physical_idx=destination_role_by_physical,
        backbone_arc_mask=arc_df["kind"].eq("backbone").to_numpy(),
        regional_arc_mask=arc_df["kind"].eq("regional_transfer").to_numpy(),
        connector_arc_mask=arc_df["kind"].astype(str).str.contains("connector").to_numpy(),
    )


class PartCRev1RegionalCostState:
    def __init__(self, active_pair_table: pd.DataFrame, *, slope: float = 1e-6) -> None:
        self.active_pair_table = active_pair_table.reset_index(drop=True)
        self.slope = float(slope)
        self.lambda_last: dict[str, np.ndarray] = {}
        self.cost_last: dict[str, np.ndarray] = {}
        self.last_update_iteration: dict[str, int] = {}

    def evaluate(
        self,
        pair_flow: np.ndarray,
        *,
        iteration: int,
        force: bool,
        abs_tol: float,
        rel_tol: float,
        max_stale_iterations: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        costs = np.zeros(len(self.active_pair_table), dtype=np.float64)
        history: list[dict[str, Any]] = []
        if self.active_pair_table.empty:
            return costs, history
        for region_uuid, group in self.active_pair_table.groupby("region_uuid", sort=False):
            positions = group.index.to_numpy(dtype=np.int64)
            lam = pair_flow[positions].astype(np.float64)
            tau0 = group["tau0"].to_numpy(dtype=np.float64)
            previous = self.lambda_last.get(str(region_uuid))
            stale = int(iteration) - int(self.last_update_iteration.get(str(region_uuid), -10**9))
            delta = math.inf if previous is None else float(np.max(np.abs(lam - previous))) if len(lam) else 0.0
            threshold = float(abs_tol) + float(rel_tol) * max(1.0, float(np.max(np.abs(previous))) if previous is not None and len(previous) else 0.0)
            refresh = bool(force or previous is None or delta > threshold or stale > int(max_stale_iterations))
            if refresh:
                region_cost = tau0 + self.slope * lam
                self.lambda_last[str(region_uuid)] = lam.copy()
                self.cost_last[str(region_uuid)] = region_cost.copy()
                self.last_update_iteration[str(region_uuid)] = int(iteration)
                action = "refreshed"
            else:
                region_cost = self.cost_last[str(region_uuid)].copy()
                action = "skipped_cached"
            costs[positions] = region_cost
            history.append(
                {
                    "iteration": int(iteration),
                    "region_uuid": str(region_uuid),
                    "action": action,
                    "pair_count": int(len(positions)),
                    "lambda_delta_inf": None if not np.isfinite(delta) else delta,
                    "threshold": threshold,
                    "stale_iterations": stale,
                }
            )
        return costs, history


def partC_rev1_mixed_fw_derivative(
    eta: float,
    flow: np.ndarray,
    direction: np.ndarray,
    *,
    backbone_mask: np.ndarray,
    free_time: np.ndarray,
    capacity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    regional_mask: np.ndarray,
    regional_tau0: np.ndarray,
    regional_slope: float,
) -> float:
    trial = np.maximum(np.asarray(flow, dtype=np.float64) + float(eta) * np.asarray(direction, dtype=np.float64), 0.0)
    derivative = 0.0
    if np.any(backbone_mask):
        c = partC_rev1_bpr_cost(trial[backbone_mask], free_time[backbone_mask], capacity[backbone_mask], alpha[backbone_mask], beta[backbone_mask])
        derivative += float(np.dot(direction[backbone_mask], c))
    if np.any(regional_mask):
        regional_positions = np.flatnonzero(regional_mask)
        costs = regional_tau0 + float(regional_slope) * trial[regional_positions]
        derivative += float(np.dot(direction[regional_positions], costs))
    return derivative


def partC_rev1_mixed_fw_line_search(
    flow: np.ndarray,
    direction: np.ndarray,
    *,
    backbone_mask: np.ndarray,
    free_time: np.ndarray,
    capacity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    regional_mask: np.ndarray,
    regional_tau0: np.ndarray,
    regional_slope: float,
    tol: float = 1e-8,
) -> float:
    g0 = partC_rev1_mixed_fw_derivative(0.0, flow, direction, backbone_mask=backbone_mask, free_time=free_time, capacity=capacity, alpha=alpha, beta=beta, regional_mask=regional_mask, regional_tau0=regional_tau0, regional_slope=regional_slope)
    g1 = partC_rev1_mixed_fw_derivative(1.0, flow, direction, backbone_mask=backbone_mask, free_time=free_time, capacity=capacity, alpha=alpha, beta=beta, regional_mask=regional_mask, regional_tau0=regional_tau0, regional_slope=regional_slope)
    if g0 >= 0.0:
        return 0.0
    if g1 <= 0.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        gm = partC_rev1_mixed_fw_derivative(mid, flow, direction, backbone_mask=backbone_mask, free_time=free_time, capacity=capacity, alpha=alpha, beta=beta, regional_mask=regional_mask, regional_tau0=regional_tau0, regional_slope=regional_slope)
        if abs(gm) <= tol or hi - lo <= tol:
            return float(mid)
        if gm < 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _partC_rev1_hybrid_costs(
    bundle: PartCRev1HybridGraphBundle,
    flow: np.ndarray,
    regional_state: PartCRev1RegionalCostState,
    *,
    iteration: int,
    force_refresh: bool,
    config: PartCRev1SparseGlobalConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cost = bundle.graph.free_time.copy()
    if np.any(bundle.backbone_arc_mask):
        idx = np.flatnonzero(bundle.backbone_arc_mask)
        cost[idx] = partC_rev1_bpr_cost(flow[idx], bundle.graph.free_time[idx], bundle.graph.capacity[idx], bundle.graph.alpha[idx], bundle.graph.beta[idx])
    if np.any(bundle.connector_arc_mask):
        cost[bundle.connector_arc_mask] = 0.0
    lazy_history: list[dict[str, Any]] = []
    if np.any(bundle.regional_arc_mask) and not bundle.active_pair_table.empty:
        regional_arc_positions = np.flatnonzero(bundle.regional_arc_mask)
        pair_indices = bundle.active_arc_table.loc[bundle.regional_arc_mask, "source_ref"].to_numpy(dtype=np.int64)
        pair_flow = np.zeros(len(bundle.active_pair_table), dtype=np.float64)
        pair_flow[pair_indices] = flow[regional_arc_positions]
        pair_cost, lazy_history = regional_state.evaluate(
            pair_flow,
            iteration=iteration,
            force=force_refresh,
            abs_tol=config.lazy_abs_tol,
            rel_tol=config.lazy_rel_tol,
            max_stale_iterations=config.max_stale_iterations,
        )
        cost[regional_arc_positions] = pair_cost[pair_indices]
    return cost, lazy_history


def partC_rev1_price_omitted_pairs(
    labels: dict[str, float],
    candidate_pairs: pd.DataFrame,
    *,
    price_tol: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in candidate_pairs.itertuples(index=False):
        source_label = labels.get(str(row.source_terminal_uid), math.inf)
        target_label = labels.get(str(row.target_terminal_uid), math.inf)
        if not np.isfinite(source_label) or not np.isfinite(target_label):
            continue
        reduced_cost = float(source_label + float(row.cost) - target_label)
        if reduced_cost < -float(price_tol):
            item = row._asdict()
            item["reduced_cost"] = reduced_cost
            item["activation_reason"] = "negative_reduced_cost"
            rows.append(item)
    return pd.DataFrame(rows)


def partC_rev1_multi_source_region_pricing(
    region_graph: PartCRev1GraphSnapshot,
    source_terminal_nodes: dict[str, int],
    target_terminal_nodes: dict[str, int],
    source_labels: dict[str, float],
    target_labels: dict[str, float],
    cost: np.ndarray,
    *,
    price_tol: float,
) -> pd.DataFrame:
    n = len(region_graph.node_idx_to_uid)
    dist = np.full(n, np.inf, dtype=np.float64)
    owner: list[str | None] = [None] * n
    heap: list[tuple[float, int, str]] = []
    for terminal_uid, node in source_terminal_nodes.items():
        label = float(source_labels.get(terminal_uid, math.inf))
        if np.isfinite(label):
            dist[int(node)] = label
            owner[int(node)] = terminal_uid
            heapq.heappush(heap, (label, int(node), terminal_uid))
    while heap:
        du, u, source_uid = heapq.heappop(heap)
        if du > dist[u] + 1e-12:
            continue
        for pos in range(int(region_graph.csr_indptr[u]), int(region_graph.csr_indptr[u + 1])):
            arc_idx = int(region_graph.csr_arc_idx[pos])
            v = int(region_graph.head[arc_idx])
            nd = du + float(cost[arc_idx])
            if nd < dist[v] - 1e-12:
                dist[v] = nd
                owner[v] = source_uid
                heapq.heappush(heap, (nd, v, source_uid))
    rows = []
    for target_uid, node in target_terminal_nodes.items():
        active_label = float(target_labels.get(target_uid, math.inf))
        physical = float(dist[int(node)])
        if np.isfinite(physical) and np.isfinite(active_label) and physical < active_label - float(price_tol):
            rows.append(
                {
                    "source_terminal_uid": owner[int(node)],
                    "target_terminal_uid": target_uid,
                    "physical_label": physical,
                    "active_label": active_label,
                    "reduced_cost": physical - active_label,
                    "activation_reason": "physical_multisource_pricing",
                }
            )
    return pd.DataFrame(rows)


def partC_rev1_run_sparse_master(
    bundle: PartCRev1HybridGraphBundle,
    *,
    config: PartCRev1SparseGlobalConfig,
    reporter: PartCRev1ProgressReporter | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    algorithm = str(config.assignment_algorithm).lower().strip()
    if algorithm not in {"msa", "fw", "cfw"}:
        raise ValueError(f"Unsupported sparse master assignment algorithm: {config.assignment_algorithm}")
    max_origins = None if int(config.pilot_origins) <= 0 else int(config.pilot_origins)
    flow = np.zeros(len(bundle.graph.arc_idx_to_directed_arc_id), dtype=np.float64)
    regional_state = PartCRev1RegionalCostState(bundle.active_pair_table, slope=config.regional_cost_slope)
    convergence_rows: list[dict[str, Any]] = []
    lazy_rows: list[dict[str, Any]] = []
    unit_seconds: list[float] = []
    stop_reason = "max_iterations"
    previous_direction: np.ndarray | None = None
    for iteration in range(int(config.max_iterations)):
        iter_started = time.perf_counter()
        force = iteration == 0 or (iteration % max(1, int(config.full_refresh_frequency)) == 0)
        cost, lazy_history = _partC_rev1_hybrid_costs(bundle, flow, regional_state, iteration=iteration, force_refresh=force, config=config)
        lazy_rows.extend(lazy_history)
        aon = partC_rev1_aon_load(
            bundle.graph,
            bundle.od,
            cost,
            backend="numba_csr",
            max_origins=max_origins,
            origin_batch_size=int(config.origin_batch_size),
            reporter=reporter,
        )
        direction = aon.flow - flow
        step = 1.0 / float(iteration + 1)
        if algorithm == "cfw" and previous_direction is not None:
            denom = max(float(np.dot(previous_direction, previous_direction)), 1e-30)
            gamma = min(0.5, max(0.0, float(np.dot(direction, direction)) / denom))
            direction = direction + gamma * previous_direction
        if algorithm in {"fw", "cfw"} and iteration > 0:
            regional_positions = np.flatnonzero(bundle.regional_arc_mask)
            regional_tau0 = bundle.graph.free_time[regional_positions]
            try:
                step = partC_rev1_mixed_fw_line_search(
                    flow,
                    direction,
                    backbone_mask=bundle.backbone_arc_mask,
                    free_time=bundle.graph.free_time,
                    capacity=bundle.graph.capacity,
                    alpha=bundle.graph.alpha,
                    beta=bundle.graph.beta,
                    regional_mask=bundle.regional_arc_mask,
                    regional_tau0=regional_tau0,
                    regional_slope=config.regional_cost_slope,
                )
            except Exception:
                step = 1.0 / float(iteration + 1)
        tstt = float(np.dot(flow, cost))
        gap = None if iteration == 0 and tstt <= 1e-12 else partC_rev1_relative_gap(tstt, aon.sptt)
        flow = np.maximum(flow + float(step) * direction, 0.0)
        elapsed = time.perf_counter() - iter_started
        unit_seconds.append(elapsed)
        resource = partC_rev1_resource_snapshot(started)
        row = {
            "iteration": iteration,
            "assignment_algorithm": algorithm,
            "step_size": float(step),
            "restricted_gap": gap,
            "tstt": tstt,
            "sptt": float(aon.sptt),
            "reachable_demand": float(aon.reachable_demand),
            "unreachable_demand": float(aon.unreachable_demand),
            "processed_origins": int(aon.processed_origins),
            "total_origins": int(aon.total_origins),
            "iteration_runtime_seconds": elapsed,
            "memory_rss_mb": resource.memory_rss_mb,
        }
        convergence_rows.append(row)
        if reporter is not None:
            reporter.emit("sparse_master_iteration", **row)
        if gap is not None and gap <= config.master_gap_tol and iteration > 0:
            stop_reason = "master_gap_target_met"
            break
        if time.perf_counter() - started >= float(config.runtime_guard_seconds):
            stop_reason = "runtime_limit"
            break
        previous_direction = direction.copy()
    final_cost, full_lazy = _partC_rev1_hybrid_costs(bundle, flow, regional_state, iteration=len(convergence_rows), force_refresh=True, config=config)
    lazy_rows.extend(full_lazy)
    final_aon = partC_rev1_aon_load(
        bundle.graph,
        bundle.od,
        final_cost,
        backend="numba_csr",
        max_origins=max_origins,
        origin_batch_size=int(config.origin_batch_size),
        reporter=reporter,
    )
    final_tstt = float(np.dot(flow, final_cost))
    official_gap = partC_rev1_relative_gap(final_tstt, final_aon.sptt)
    if stop_reason == "master_gap_target_met" and official_gap > config.master_gap_tol * 1.5:
        stop_reason = "stale_convergence_rejected_after_full_refresh"
    projected_full_seconds = None
    if unit_seconds:
        origins_per_iteration = len(bundle.od.origin_group_node_idx) if max_origins is None else max(int(max_origins), 1)
        projected_full_seconds = float(median(unit_seconds) * max(int(config.max_iterations), 1) * len(bundle.od.origin_group_node_idx) / max(origins_per_iteration, 1))
    pricing_rows = pd.DataFrame()
    pricing_gap = 0.0
    pricing_state = "empirical_no_omitted_candidates_materialized"
    run_state = "measured_complete" if int(final_aon.processed_origins) >= int(final_aon.total_origins) else "measured_partial_projected"
    return {
        "flow": flow,
        "cost": final_cost,
        "convergence": pd.DataFrame(convergence_rows),
        "lazy_refresh": pd.DataFrame(lazy_rows),
        "pricing": pricing_rows,
        "summary": {
            "state": run_state,
            "stop_reason": stop_reason,
            "assignment_algorithm": algorithm,
            "official_full_refresh_gap": official_gap,
            "pricing_gap": pricing_gap,
            "pricing_state": pricing_state,
            "runtime_seconds": time.perf_counter() - started,
            "projected_full_origin_seconds": projected_full_seconds,
            "processed_origins": int(final_aon.processed_origins),
            "total_origins": int(final_aon.total_origins),
            "final_tstt": final_tstt,
            "final_sptt": float(final_aon.sptt),
            "reachable_demand": float(final_aon.reachable_demand),
            "unreachable_demand": float(final_aon.unreachable_demand),
            "iteration_count": len(convergence_rows),
        },
    }


def partC_rev1_run_sparse_global_stage(
    physical_graph: PartCRev1GraphSnapshot,
    physical_od: PartCRev1ODSnapshot,
    *,
    benchmark_flow: np.ndarray,
    route_witnesses: pd.DataFrame,
    summary_4d: dict[str, Any],
    config: PartCRev1SparseGlobalConfig,
    output_dir: str | Path,
    reporter: PartCRev1ProgressReporter | None = None,
) -> dict[str, Any]:
    out_dir = partC_rev1_resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    regional_cfg_data = summary_4d.get("config", {})
    regional_cfg = PartCRev1RegionalTrainingConfig(**regional_cfg_data)
    active_pairs = partC_rev1_build_active_pairs_from_regions(
        physical_graph,
        physical_od,
        benchmark_flow=benchmark_flow,
        route_witnesses=route_witnesses,
        completed_regions=summary_4d.get("completed_regions", []),
        regional_config=regional_cfg,
    )
    bundle = partC_rev1_build_hybrid_graph(physical_graph, physical_od, active_pairs)
    result = partC_rev1_run_sparse_master(bundle, config=config, reporter=reporter)
    active_pair_history_path = out_dir / "partC_rev1_4E_active_pair_history.csv"
    convergence_path = out_dir / "partC_rev1_4E_convergence.csv"
    lazy_path = out_dir / "partC_rev1_4E_lazy_refresh_history.csv"
    pricing_path = out_dir / "partC_rev1_4E_pricing_candidates.csv"
    final_backbone_path = out_dir / "partC_rev1_4E_final_backbone_flow.parquet"
    final_pair_path = out_dir / "partC_rev1_4E_final_regional_pair_flow.parquet"
    partC_rev1_write_table(active_pairs, active_pair_history_path, index=False)
    partC_rev1_write_table(result["convergence"], convergence_path, index=False)
    partC_rev1_write_table(result["lazy_refresh"], lazy_path, index=False)
    partC_rev1_write_table(result["pricing"], pricing_path, index=False)
    arc_table = bundle.active_arc_table.copy()
    arc_table["flow"] = result["flow"]
    arc_table["cost"] = result["cost"]
    partC_rev1_write_table(arc_table.loc[arc_table["kind"].eq("backbone")], final_backbone_path, index=False)
    partC_rev1_write_table(arc_table.loc[arc_table["kind"].eq("regional_transfer")], final_pair_path, index=False)
    conceptual_estimate = partC_rev1_conceptual_pair_estimate(active_pairs)
    summary = {
        "stage": PARTC_REV1_4E_STAGE,
        "created_utc": _partC_rev1_utc_now(),
        "config": asdict(config),
        "graph_hash": physical_graph.graph_hash,
        "od_hash": physical_od.od_hash,
        "partition_hash": physical_graph.partition_hash,
        "hybrid_graph_hash": bundle.graph.graph_hash,
        "active_pair_count": int(len(active_pairs)),
        "candidate_superset_pairs": int(len(active_pairs)),
        "conceptual_pair_estimate_for_model_regions": int(conceptual_estimate),
        "memory_bytes_active_pairs": int(active_pairs.memory_usage(deep=True).sum()) if not active_pairs.empty else 0,
        "hybrid_node_count": int(len(bundle.graph.node_idx_to_uid)),
        "hybrid_arc_count": int(len(bundle.graph.arc_idx_to_directed_arc_id)),
        "backbone_arc_count": int(bundle.backbone_arc_mask.sum()),
        "connector_arc_count": int(bundle.connector_arc_mask.sum()),
        "regional_transfer_arc_count": int(bundle.regional_arc_mask.sum()),
        "model_region_count": int(active_pairs["region_uuid"].nunique()) if not active_pairs.empty else 0,
        "run": result["summary"],
        "artifacts": {
            "active_pair_history_csv": str(active_pair_history_path),
            "convergence_csv": str(convergence_path),
            "lazy_refresh_history_csv": str(lazy_path),
            "pricing_candidates_csv": str(pricing_path),
            "final_backbone_flow_parquet": str(final_backbone_path),
            "final_regional_pair_flow_parquet": str(final_pair_path),
        },
        "known_approximations": [
            "Only Prompt-4D measured pilot model regions are represented as active regional transfers.",
            "Regional pair costs use the 4D reconstructed active support with a convex separable ICNN-compatible surrogate slope.",
            "Pricing pass is empirical and does not enumerate dense omitted conceptual pairs.",
        ],
    }
    summary_path = out_dir / "partC_rev1_4E_summary.json"
    partC_rev1_write_json(summary, summary_path)
    summary["summary_path"] = str(summary_path)
    return summary


def partC_rev1_write_4e_handoff(summary: dict[str, Any], *, output_dir: str | Path, reporter: PartCRev1ProgressReporter | None = None) -> Path:
    out_dir = partC_rev1_resolve_path(output_dir)
    handoff_path = out_dir / "partC_rev1_4E_handoff.md"
    run = summary.get("run", {})
    lines = [
        "# partC_rev1 4E Handoff",
        "",
        f"Generated UTC: {_partC_rev1_utc_now()}",
        "",
        "## Status",
        "",
        f"- Stage state: `{run.get('state')}`",
        f"- Stop reason: `{run.get('stop_reason')}`",
        f"- Official full-refresh restricted gap: `{run.get('official_full_refresh_gap')}`",
        f"- Empirical pricing gap: `{run.get('pricing_gap')}`",
        f"- Processed origins: {run.get('processed_origins')}/{run.get('total_origins')}",
        f"- Projected full-origin seconds: {run.get('projected_full_origin_seconds')}",
        "",
        "## Active Graph",
        "",
        f"- Hybrid graph hash: `{summary.get('hybrid_graph_hash')}`",
        f"- Hybrid nodes: {summary.get('hybrid_node_count')}",
        f"- Hybrid arcs: {summary.get('hybrid_arc_count')}",
        f"- Backbone arcs: {summary.get('backbone_arc_count')}",
        f"- Connector arcs: {summary.get('connector_arc_count')}",
        f"- Regional transfer arcs: {summary.get('regional_transfer_arc_count')}",
        f"- Active pairs: {summary.get('active_pair_count')}",
        f"- Conceptual pair estimate for model regions: {summary.get('conceptual_pair_estimate_for_model_regions')}",
        "",
        "## Artifacts",
        "",
        f"- Summary JSON: `{summary.get('summary_path')}`",
    ]
    for label, path in dict(summary.get("artifacts", {})).items():
        lines.append(f"- {label}: `{path}`")
    if reporter is not None:
        lines.extend([f"- Stage log: `{reporter.log_path}`", f"- Progress JSONL: `{reporter.progress_path}`"])
    lines.extend(
        [
            "",
            "## Known Approximations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary.get("known_approximations", []))
    lines.extend(
        [
            "",
            "## Tests",
            "",
            "- Focused 4E and full post-stage tests are recorded in the execution manifest/current-state docs after commit.",
            "- Environment note: `conda run -n py312` emits the known `conda-libmamba-solver` entry-point warning.",
            "",
            "## Exact 4F Command",
            "",
            "```text",
            "conda run -n py312 python -m partC_rev1_src.partC_rev1_cli run-physical-reload --config partC_rev1_configs/partC_rev1_real_preprocess_config.json --confirm-full-run",
            "```",
            "",
        ]
    )
    handoff_path.write_text("\n".join(lines), encoding="utf-8")
    return handoff_path
