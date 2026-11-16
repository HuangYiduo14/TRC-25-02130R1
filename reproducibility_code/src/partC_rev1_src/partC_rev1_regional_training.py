"""Prompt 4D regional terminals, samples, and surrogate training."""

from __future__ import annotations

import gc
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from .partC_rev1_assignment import PartCRev1AssignmentConfig, partC_rev1_solve_physical_assignment
from .partC_rev1_graph_snapshot import PartCRev1GraphSnapshot
from .partC_rev1_hashing import (
    partC_rev1_json_hash,
    partC_rev1_numpy_array_hash,
    partC_rev1_sha256_text,
    partC_rev1_sort_key,
)
from .partC_rev1_io import partC_rev1_write_json, partC_rev1_write_table
from .partC_rev1_od_snapshot import PartCRev1ODSnapshot, partC_rev1_build_od_snapshot_from_frames
from .partC_rev1_paths import partC_rev1_resolve_path
from .partC_rev1_progress_logging import PartCRev1ProgressReporter
from .partC_rev1_runtime_guard import PARTC_REV1_DEFAULT_MAX_RUN_SECONDS, partC_rev1_resource_snapshot
from .partC_rev1_shortest_paths import partC_rev1_sssp


PARTC_REV1_4D_STAGE = "partC_rev1_4D"


def _partC_rev1_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _partC_rev1_json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


@dataclass(frozen=True, slots=True)
class PartCRev1RegionalTrainingConfig:
    config_path: str
    requested_samples_per_region: int = 100
    pilot_samples_per_region: int = 12
    pilot_region_count: int = 3
    production_region_limit: int = 0
    max_pilot_region_arcs: int = 600
    max_active_pairs: int = 64
    top_k_per_source: int = 6
    ipf_tol: float = 1e-5
    ipf_max_iter: int = 500
    marginal_tolerance: float = 0.20
    gravity_theta: float = 1.0
    marginal_sigma: float = 0.25
    pair_noise_sigma: float = 0.05
    regional_solver_method: str = "fw"
    regional_solver_iterations: int = 3
    regional_gap_target: float = 1e-3
    regional_origin_batch_size: int = 8
    xgb_estimators: int = 12
    nn_epochs: int = 24
    icnn_epochs: int = 24
    dense_threshold: int = 64
    random_seed: int = 42
    runtime_guard_seconds: float = PARTC_REV1_DEFAULT_MAX_RUN_SECONDS
    od_sampler: str = "legacy_sparse_ipf_lognormal"
    gravity_epsilon: float = 1e-12
    support_mask_renormalization: str = "none"
    terminal_perturbation: str = "uniform_0_200"
    skip_on_marginal_imbalance: bool = False
    split_train_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)


@dataclass(slots=True)
class PartCRev1RegionProblem:
    region_uuid: str
    region_hash: str
    region_graph: PartCRev1GraphSnapshot
    global_arc_indices: np.ndarray
    source_terminals: pd.DataFrame
    target_terminals: pd.DataFrame
    source_marginal: np.ndarray
    target_marginal: np.ndarray
    support: pd.DataFrame
    terminal_order_hash: str
    pair_order_hash: str
    imbalance: dict[str, Any]
    status: str


@dataclass(slots=True)
class PartCRev1RegionDataset:
    problem: PartCRev1RegionProblem
    sample_table: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    tau0: np.ndarray
    pair_features: np.ndarray
    temp_path: Path
    metadata: dict[str, Any]


def _partC_rev1_node_uid(graph: PartCRev1GraphSnapshot, node_idx: int) -> str:
    return str(graph.node_idx_to_uid[int(node_idx)])


def _partC_rev1_terminal_uid(region_uuid: str, kind: str, node_uid: str) -> str:
    return f"{region_uuid}|{kind}|{node_uid}"


def _partC_rev1_pair_uid(region_uuid: str, source_terminal_uid: str, target_terminal_uid: str) -> str:
    return partC_rev1_sha256_text(f"{region_uuid}\x1f{source_terminal_uid}\x1f{target_terminal_uid}")[:24]


def _partC_rev1_labels_for_arcs(graph: PartCRev1GraphSnapshot) -> np.ndarray:
    return np.asarray([graph.region_idx_to_label[int(code)] for code in graph.region_code], dtype=object)


def _partC_rev1_uses_benchmark_flow_gravity(config: PartCRev1RegionalTrainingConfig) -> bool:
    return str(config.od_sampler).lower() in {"benchmark_flow_gravity", "benchmark_flow_gravity_additive"}


def partC_rev1_centroid_node_mask(graph: PartCRev1GraphSnapshot, od: PartCRev1ODSnapshot) -> np.ndarray:
    mask = np.zeros(len(graph.node_idx_to_uid), dtype=bool)
    if len(od.origin_node_idx):
        mask[np.unique(od.origin_node_idx)] = True
    if len(od.destination_node_idx):
        mask[np.unique(od.destination_node_idx)] = True
    return mask


def partC_rev1_od_marginals_by_node(graph: PartCRev1GraphSnapshot, od: PartCRev1ODSnapshot) -> tuple[np.ndarray, np.ndarray]:
    n = len(graph.node_idx_to_uid)
    source = np.bincount(od.origin_node_idx.astype(np.int64), weights=od.demand.astype(np.float64), minlength=n).astype(np.float64)
    target = np.bincount(od.destination_node_idx.astype(np.int64), weights=od.demand.astype(np.float64), minlength=n).astype(np.float64)
    return source, target


def partC_rev1_incident_label_counts(graph: PartCRev1GraphSnapshot) -> np.ndarray:
    endpoint_rows = pd.DataFrame(
        {
            "node_idx": np.concatenate([graph.tail.astype(np.int64), graph.head.astype(np.int64)]),
            "label": np.concatenate([graph.region_code.astype(np.int64), graph.region_code.astype(np.int64)]),
        }
    )
    counts = endpoint_rows.groupby("node_idx", sort=False)["label"].nunique()
    out = np.zeros(len(graph.node_idx_to_uid), dtype=np.int32)
    out[counts.index.to_numpy(dtype=np.int64)] = counts.to_numpy(dtype=np.int32)
    return out


def partC_rev1_region_size_table(graph: PartCRev1GraphSnapshot, od: PartCRev1ODSnapshot) -> pd.DataFrame:
    labels = _partC_rev1_labels_for_arcs(graph)
    endpoint = pd.DataFrame(
        {
            "region_uuid": np.concatenate([labels, labels]),
            "node_idx": np.concatenate([graph.tail.astype(np.int64), graph.head.astype(np.int64)]),
        }
    )
    endpoint = endpoint.loc[endpoint["region_uuid"].ne("backbone")].copy()
    incident_counts = partC_rev1_incident_label_counts(graph)
    centroid_mask = partC_rev1_centroid_node_mask(graph, od)
    endpoint["is_portal"] = incident_counts[endpoint["node_idx"].to_numpy(dtype=np.int64)] >= 2
    endpoint["is_centroid"] = centroid_mask[endpoint["node_idx"].to_numpy(dtype=np.int64)]
    arc_counts = pd.Series(labels[labels != "backbone"]).value_counts(sort=False).rename_axis("region_uuid").reset_index(name="arc_count")
    grouped = endpoint.groupby("region_uuid", sort=False).agg(
        node_count=("node_idx", "nunique"),
        portal_count=("is_portal", "sum"),
        centroid_endpoint_count=("is_centroid", "sum"),
    )
    grouped["portal_count"] = endpoint.loc[endpoint["is_portal"]].groupby("region_uuid", sort=False)["node_idx"].nunique()
    grouped["centroid_count"] = endpoint.loc[endpoint["is_centroid"]].groupby("region_uuid", sort=False)["node_idx"].nunique()
    grouped = grouped.fillna(0).reset_index()
    table = arc_counts.merge(grouped, on="region_uuid", how="left").fillna(0)
    table["arc_count"] = table["arc_count"].astype(int)
    table["node_count"] = table["node_count"].astype(int)
    table["portal_count"] = table["portal_count"].astype(int)
    table["centroid_count"] = table["centroid_count"].astype(int)
    table["terminal_proxy_count"] = table["portal_count"] * 2 + table["centroid_count"] * 2
    table = table.sort_values(["arc_count", "region_uuid"], kind="mergesort").reset_index(drop=True)
    return table


def partC_rev1_select_stratified_regions(
    region_table: pd.DataFrame,
    *,
    count: int,
    max_pilot_region_arcs: int,
) -> list[str]:
    eligible = region_table.loc[region_table["terminal_proxy_count"].gt(1)].copy()
    capped = eligible.loc[eligible["arc_count"].le(max_pilot_region_arcs)].copy()
    if capped.empty:
        capped = eligible
    if capped.empty or count <= 0:
        return []
    capped = capped.sort_values(["arc_count", "region_uuid"], kind="mergesort").reset_index(drop=True)
    if len(capped) <= count:
        return capped["region_uuid"].astype(str).tolist()
    quantiles = np.linspace(0.05, 0.95, num=max(count * 4, count), dtype=np.float64)
    picked: list[str] = []
    for q in quantiles:
        idx = int(round(q * (len(capped) - 1)))
        region = str(capped.loc[idx, "region_uuid"])
        if region not in picked:
            picked.append(region)
        if len(picked) >= count * 4:
            break
    return picked


def _partC_rev1_build_region_graph(graph: PartCRev1GraphSnapshot, region_uuid: str) -> tuple[PartCRev1GraphSnapshot, np.ndarray]:
    labels = _partC_rev1_labels_for_arcs(graph)
    arc_indices = np.flatnonzero(labels == region_uuid).astype(np.int64)
    if len(arc_indices) == 0:
        raise ValueError(f"Region has no arcs: {region_uuid}")
    global_nodes = np.unique(np.concatenate([graph.tail[arc_indices], graph.head[arc_indices]])).astype(np.int64)
    node_order = sorted(global_nodes.tolist(), key=lambda idx: partC_rev1_sort_key(graph.node_idx_to_uid[int(idx)]))
    local_node_idx_to_global = np.asarray(node_order, dtype=np.int64)
    global_to_local = np.full(len(graph.node_idx_to_uid), -1, dtype=np.int64)
    global_to_local[local_node_idx_to_global] = np.arange(len(local_node_idx_to_global), dtype=np.int64)
    tail = global_to_local[graph.tail[arc_indices]]
    head = global_to_local[graph.head[arc_indices]]
    order = np.lexsort((graph.arc_idx_to_directed_arc_id[arc_indices].astype(str), tail))
    arc_indices = arc_indices[order]
    tail = tail[order].astype(np.int64)
    head = head[order].astype(np.int64)
    counts = np.bincount(tail, minlength=len(local_node_idx_to_global))
    csr_indptr = np.concatenate(([0], np.cumsum(counts, dtype=np.int64))).astype(np.int64)
    csr_arc_idx = np.arange(len(arc_indices), dtype=np.int64)
    reverse_order = np.lexsort((graph.arc_idx_to_directed_arc_id[arc_indices].astype(str), tail, head)).astype(np.int64)
    reverse_counts = np.bincount(head, minlength=len(local_node_idx_to_global))
    reverse_csr_indptr = np.concatenate(([0], np.cumsum(reverse_counts, dtype=np.int64))).astype(np.int64)
    region_hash = partC_rev1_numpy_array_hash(
        graph.arc_idx_to_directed_arc_id[arc_indices].astype(str),
        graph.node_idx_to_uid[graph.tail[arc_indices]].astype(str),
        graph.node_idx_to_uid[graph.head[arc_indices]].astype(str),
        graph.free_time[arc_indices].astype(np.float64),
        graph.capacity[arc_indices].astype(np.float64),
    )
    region_graph = PartCRev1GraphSnapshot(
        graph_hash=region_hash,
        partition_hash=graph.partition_hash,
        node_idx_to_uid=graph.node_idx_to_uid[local_node_idx_to_global].astype(str),
        arc_idx_to_directed_arc_id=graph.arc_idx_to_directed_arc_id[arc_indices].astype(str),
        arc_idx_to_link_uid=graph.arc_idx_to_link_uid[arc_indices].astype(str),
        tail=tail,
        head=head,
        free_time=graph.free_time[arc_indices].astype(np.float64),
        base_capacity=graph.base_capacity[arc_indices].astype(np.float64),
        capacity=graph.capacity[arc_indices].astype(np.float64),
        capacity_multiplier=graph.capacity_multiplier[arc_indices].astype(np.float64),
        capacity_multiplier_type=graph.capacity_multiplier_type[arc_indices].astype(str),
        alpha=graph.alpha[arc_indices].astype(np.float64),
        beta=graph.beta[arc_indices].astype(np.float64),
        length=graph.length[arc_indices].astype(np.float64),
        region_code=np.zeros(len(arc_indices), dtype=np.int32),
        region_idx_to_label=[region_uuid],
        csr_indptr=csr_indptr,
        csr_head=head.copy(),
        csr_arc_idx=csr_arc_idx,
        reverse_csr_indptr=reverse_csr_indptr,
        reverse_csr_head=tail[reverse_order].astype(np.int64),
        reverse_csr_arc_idx=reverse_order.astype(np.int64),
        metadata={
            "parent_graph_hash": graph.graph_hash,
            "region_uuid": region_uuid,
            "global_arc_count": int(len(arc_indices)),
            "global_node_count": int(len(local_node_idx_to_global)),
        },
    )
    return region_graph, arc_indices


def _partC_rev1_terminal_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["terminal_idx", "terminal_uid", "terminal_kind", "node_idx", "node_uid", "marginal"])
    df = pd.DataFrame(rows)
    df = df.sort_values(["terminal_kind", "node_uid"], kind="mergesort").reset_index(drop=True)
    df["terminal_idx"] = np.arange(len(df), dtype=np.int64)
    return df.loc[:, ["terminal_idx", "terminal_uid", "terminal_kind", "node_idx", "node_uid", "marginal"]]


def _partC_rev1_positive_centroid_od_pairs(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    region_nodes: set[int],
) -> pd.DataFrame:
    if not region_nodes or not len(od.demand):
        return pd.DataFrame(columns=["origin_node_uid", "destination_node_uid", "demand"])
    node_array = np.asarray(sorted(region_nodes), dtype=np.int64)
    origin_mask = np.isin(od.origin_node_idx, node_array, assume_unique=False)
    dest_mask = np.isin(od.destination_node_idx, node_array, assume_unique=False)
    mask = origin_mask & dest_mask
    if not bool(np.any(mask)):
        return pd.DataFrame(columns=["origin_node_uid", "destination_node_uid", "demand"])
    origins = od.origin_node_idx[mask]
    dests = od.destination_node_idx[mask]
    demand = od.demand[mask]
    df = pd.DataFrame(
        {
            "origin_node_uid": graph.node_idx_to_uid[origins],
            "destination_node_uid": graph.node_idx_to_uid[dests],
            "demand": demand,
        }
    )
    return df.groupby(["origin_node_uid", "destination_node_uid"], as_index=False)["demand"].sum()


def _partC_rev1_add_pair_candidate(
    candidates: dict[tuple[int, int], dict[str, Any]],
    *,
    region_uuid: str,
    source_terms: pd.DataFrame,
    target_terms: pd.DataFrame,
    source_idx: int,
    target_idx: int,
    tau0: float,
    pair_type: str,
    priority: int,
) -> None:
    if source_idx == target_idx and str(source_terms.loc[source_idx, "node_uid"]) == str(target_terms.loc[target_idx, "node_uid"]):
        return
    if not np.isfinite(tau0):
        return
    key = (int(source_idx), int(target_idx))
    item = candidates.get(key)
    source_uid = str(source_terms.loc[source_idx, "terminal_uid"])
    target_uid = str(target_terms.loc[target_idx, "terminal_uid"])
    if item is None:
        candidates[key] = {
            "source_idx": int(source_idx),
            "target_idx": int(target_idx),
            "source_terminal_uid": source_uid,
            "target_terminal_uid": target_uid,
            "pair_id": _partC_rev1_pair_uid(region_uuid, source_uid, target_uid),
            "tau0": float(tau0),
            "pair_type": pair_type,
            "priority": int(priority),
        }
    else:
        item["priority"] = min(int(item["priority"]), int(priority))
        types = set(str(item["pair_type"]).split("+"))
        types.add(pair_type)
        item["pair_type"] = "+".join(sorted(types))
        item["tau0"] = min(float(item["tau0"]), float(tau0))


def partC_rev1_build_sparse_support(
    region_graph: PartCRev1GraphSnapshot,
    *,
    parent_region_uuid: str,
    source_terms: pd.DataFrame,
    target_terms: pd.DataFrame,
    route_witnesses: pd.DataFrame | None = None,
    positive_od_pairs: pd.DataFrame | None = None,
    top_k_per_source: int = 6,
    max_active_pairs: int = 64,
) -> pd.DataFrame:
    local_idx = region_graph.node_uid_to_idx
    source_locs = np.asarray([local_idx[str(uid)] for uid in source_terms["node_uid"].astype(str).tolist()], dtype=np.int64)
    target_locs = np.asarray([local_idx[str(uid)] for uid in target_terms["node_uid"].astype(str).tolist()], dtype=np.int64)
    candidates: dict[tuple[int, int], dict[str, Any]] = {}
    distance_cache: dict[int, np.ndarray] = {}
    for s_idx, source_local in enumerate(source_locs.tolist()):
        sssp = partC_rev1_sssp(region_graph, int(source_local), region_graph.free_time, backend="numba_csr", targets=target_locs.tolist())
        distance_cache[int(s_idx)] = sssp.distances[target_locs].astype(np.float64)
        reachable = np.flatnonzero(np.isfinite(distance_cache[int(s_idx)]))
        reachable = [int(t) for t in reachable if source_locs[int(s_idx)] != target_locs[int(t)]]
        reachable = sorted(reachable, key=lambda t: (distance_cache[int(s_idx)][t], str(target_terms.loc[t, "terminal_uid"])))
        for t_idx in reachable[: max(0, int(top_k_per_source))]:
            _partC_rev1_add_pair_candidate(
                candidates,
                region_uuid=parent_region_uuid,
                source_terms=source_terms,
                target_terms=target_terms,
                source_idx=int(s_idx),
                target_idx=int(t_idx),
                tau0=float(distance_cache[int(s_idx)][t_idx]),
                pair_type="topk_freeflow",
                priority=2,
            )
    if route_witnesses is not None and not route_witnesses.empty:
        witness = route_witnesses.loc[route_witnesses["region_uuid"].astype(str).eq(parent_region_uuid)]
        source_by_node = {str(row.node_uid): int(row.terminal_idx) for row in source_terms.itertuples(index=False) if str(row.terminal_kind) == "portal"}
        target_by_node = {str(row.node_uid): int(row.terminal_idx) for row in target_terms.itertuples(index=False) if str(row.terminal_kind) == "portal"}
        for row in witness.itertuples(index=False):
            s_idx = source_by_node.get(str(row.block_start_node_uid))
            t_idx = target_by_node.get(str(row.block_end_node_uid))
            if s_idx is None or t_idx is None:
                continue
            tau0 = float(distance_cache.get(s_idx, np.full(len(target_terms), np.inf))[t_idx])
            _partC_rev1_add_pair_candidate(
                candidates,
                region_uuid=parent_region_uuid,
                source_terms=source_terms,
                target_terms=target_terms,
                source_idx=s_idx,
                target_idx=t_idx,
                tau0=tau0,
                pair_type="route_witness",
                priority=0,
            )
    if positive_od_pairs is not None and not positive_od_pairs.empty:
        source_by_node = {str(row.node_uid): int(row.terminal_idx) for row in source_terms.itertuples(index=False) if str(row.terminal_kind) == "centroid_origin"}
        target_by_node = {str(row.node_uid): int(row.terminal_idx) for row in target_terms.itertuples(index=False) if str(row.terminal_kind) == "centroid_destination"}
        for row in positive_od_pairs.itertuples(index=False):
            s_idx = source_by_node.get(str(row.origin_node_uid))
            t_idx = target_by_node.get(str(row.destination_node_uid))
            if s_idx is None or t_idx is None:
                continue
            tau0 = float(distance_cache.get(s_idx, np.full(len(target_terms), np.inf))[t_idx])
            _partC_rev1_add_pair_candidate(
                candidates,
                region_uuid=parent_region_uuid,
                source_terms=source_terms,
                target_terms=target_terms,
                source_idx=s_idx,
                target_idx=t_idx,
                tau0=tau0,
                pair_type="positive_centroid_od",
                priority=1,
            )
    rows = list(candidates.values())
    if not rows:
        return pd.DataFrame(
            columns=["pair_position", "pair_id", "source_idx", "target_idx", "source_terminal_uid", "target_terminal_uid", "tau0", "reachable", "pair_type"]
        )
    support = pd.DataFrame(rows).sort_values(["priority", "tau0", "pair_id"], kind="mergesort").reset_index(drop=True)
    if len(support) > int(max_active_pairs):
        keep = support.head(int(max_active_pairs)).copy()
        positive_sources = set(np.flatnonzero(source_terms["marginal"].to_numpy(dtype=np.float64) > 0.0).tolist())
        positive_targets = set(np.flatnonzero(target_terms["marginal"].to_numpy(dtype=np.float64) > 0.0).tolist())
        for source_idx in sorted(positive_sources):
            if not keep["source_idx"].eq(source_idx).any():
                candidate = support.loc[support["source_idx"].eq(source_idx)].head(1)
                keep = pd.concat([keep, candidate], ignore_index=True)
        for target_idx in sorted(positive_targets):
            if not keep["target_idx"].eq(target_idx).any():
                candidate = support.loc[support["target_idx"].eq(target_idx)].head(1)
                keep = pd.concat([keep, candidate], ignore_index=True)
        support = keep.drop_duplicates("pair_id").sort_values(["priority", "tau0", "pair_id"], kind="mergesort").head(int(max_active_pairs)).reset_index(drop=True)
    support["pair_position"] = np.arange(len(support), dtype=np.int64)
    support["reachable"] = True
    return support.loc[:, ["pair_position", "pair_id", "source_idx", "target_idx", "source_terminal_uid", "target_terminal_uid", "tau0", "reachable", "pair_type"]]


def partC_rev1_build_region_problem(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    *,
    region_uuid: str,
    benchmark_flow: np.ndarray,
    config: PartCRev1RegionalTrainingConfig,
    incident_label_count: np.ndarray | None = None,
    centroid_mask: np.ndarray | None = None,
    od_source_marginal: np.ndarray | None = None,
    od_target_marginal: np.ndarray | None = None,
    route_witnesses: pd.DataFrame | None = None,
) -> PartCRev1RegionProblem:
    incident_label_count = incident_label_count if incident_label_count is not None else partC_rev1_incident_label_counts(graph)
    centroid_mask = centroid_mask if centroid_mask is not None else partC_rev1_centroid_node_mask(graph, od)
    if od_source_marginal is None or od_target_marginal is None:
        od_source_marginal, od_target_marginal = partC_rev1_od_marginals_by_node(graph, od)
    region_graph, global_arc_indices = _partC_rev1_build_region_graph(graph, region_uuid)
    region_nodes = set(np.unique(np.concatenate([graph.tail[global_arc_indices], graph.head[global_arc_indices]])).astype(np.int64).tolist())
    portal_nodes = sorted([node for node in region_nodes if int(incident_label_count[int(node)]) >= 2], key=lambda idx: partC_rev1_sort_key(graph.node_idx_to_uid[int(idx)]))
    centroid_nodes = sorted([node for node in region_nodes if bool(centroid_mask[int(node)])], key=lambda idx: partC_rev1_sort_key(graph.node_idx_to_uid[int(idx)]))
    source_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for node in portal_nodes:
        node_uid = _partC_rev1_node_uid(graph, node)
        out_flow = float(np.sum(benchmark_flow[global_arc_indices[graph.tail[global_arc_indices] == node]], dtype=np.float64))
        in_flow = float(np.sum(benchmark_flow[global_arc_indices[graph.head[global_arc_indices] == node]], dtype=np.float64))
        source_rows.append(
            {
                "terminal_uid": _partC_rev1_terminal_uid(region_uuid, "portal", node_uid),
                "terminal_kind": "portal",
                "node_idx": int(node),
                "node_uid": node_uid,
                "marginal": out_flow,
            }
        )
        target_rows.append(
            {
                "terminal_uid": _partC_rev1_terminal_uid(region_uuid, "portal", node_uid),
                "terminal_kind": "portal",
                "node_idx": int(node),
                "node_uid": node_uid,
                "marginal": in_flow,
            }
        )
    for node in centroid_nodes:
        node_uid = _partC_rev1_node_uid(graph, node)
        source_rows.append(
            {
                "terminal_uid": _partC_rev1_terminal_uid(region_uuid, "centroid_origin", node_uid),
                "terminal_kind": "centroid_origin",
                "node_idx": int(node),
                "node_uid": node_uid,
                "marginal": float(od_source_marginal[int(node)]),
            }
        )
        target_rows.append(
            {
                "terminal_uid": _partC_rev1_terminal_uid(region_uuid, "centroid_destination", node_uid),
                "terminal_kind": "centroid_destination",
                "node_idx": int(node),
                "node_uid": node_uid,
                "marginal": float(od_target_marginal[int(node)]),
            }
        )
    source_terms = _partC_rev1_terminal_dataframe(source_rows)
    target_terms = _partC_rev1_terminal_dataframe(target_rows)
    source_marginal = source_terms["marginal"].to_numpy(dtype=np.float64) if len(source_terms) else np.array([], dtype=np.float64)
    target_marginal = target_terms["marginal"].to_numpy(dtype=np.float64) if len(target_terms) else np.array([], dtype=np.float64)
    source_total = float(np.sum(source_marginal, dtype=np.float64))
    target_total = float(np.sum(target_marginal, dtype=np.float64))
    mean_total = 0.5 * (source_total + target_total)
    relative_imbalance = abs(source_total - target_total) / max(mean_total, 1e-12)
    use_gravity = _partC_rev1_uses_benchmark_flow_gravity(config)
    epsilon = float(config.gravity_epsilon)
    status = "ok"
    if len(source_terms) == 0 or len(target_terms) == 0:
        status = "skipped_no_terminals"
    elif not use_gravity and mean_total <= 0.0:
        status = "skipped_no_positive_terminals"
    elif not use_gravity and relative_imbalance > float(config.marginal_tolerance):
        status = "skipped_marginal_imbalance"
    elif not use_gravity:
        source_marginal = source_marginal * (mean_total / max(source_total, 1e-12))
        target_marginal = target_marginal * (mean_total / max(target_total, 1e-12))
        source_terms.loc[:, "marginal"] = source_marginal
        target_terms.loc[:, "marginal"] = target_marginal
    terminal_order_hash = partC_rev1_json_hash(
        {
            "region_uuid": region_uuid,
            "source": source_terms[["terminal_uid", "terminal_kind", "node_uid"]].to_dict("records"),
            "target": target_terms[["terminal_uid", "terminal_kind", "node_uid"]].to_dict("records"),
        }
    )
    if status == "ok":
        positive_od_pairs = _partC_rev1_positive_centroid_od_pairs(graph, od, region_nodes)
        support = partC_rev1_build_sparse_support(
            region_graph,
            parent_region_uuid=region_uuid,
            source_terms=source_terms,
            target_terms=target_terms,
            route_witnesses=route_witnesses,
            positive_od_pairs=positive_od_pairs,
            top_k_per_source=config.top_k_per_source,
            max_active_pairs=config.max_active_pairs,
        )
        if support.empty:
            status = "skipped_no_reachable_support"
        elif use_gravity and (source_total <= epsilon or target_total <= epsilon):
            status = "ok"
    else:
        support = pd.DataFrame(
            columns=["pair_position", "pair_id", "source_idx", "target_idx", "source_terminal_uid", "target_terminal_uid", "tau0", "reachable", "pair_type"]
        )
    pair_order_hash = partC_rev1_json_hash({"region_uuid": region_uuid, "pairs": support["pair_id"].astype(str).tolist()}) if not support.empty else ""
    supported_sources = set(support["source_idx"].astype(int).tolist()) if not support.empty else set()
    supported_targets = set(support["target_idx"].astype(int).tolist()) if not support.empty else set()
    positive_sources = set(np.flatnonzero(source_marginal > epsilon).astype(int).tolist())
    positive_targets = set(np.flatnonzero(target_marginal > epsilon).astype(int).tolist())
    unservable_sources = sorted(positive_sources.difference(supported_sources))
    unservable_targets = sorted(positive_targets.difference(supported_targets))
    supported_source_mass = float(np.sum(source_marginal[list(supported_sources)], dtype=np.float64)) if supported_sources else 0.0
    supported_target_mass = float(np.sum(target_marginal[list(supported_targets)], dtype=np.float64)) if supported_targets else 0.0
    positive_supported_pair_od = 0.0
    if not support.empty and source_total > epsilon and target_total > epsilon:
        s_idx = support["source_idx"].to_numpy(dtype=np.int64)
        t_idx = support["target_idx"].to_numpy(dtype=np.int64)
        positive_supported_pair_od = float(np.sum((source_marginal[s_idx] > epsilon) & (target_marginal[t_idx] > epsilon)))
    if use_gravity and status == "ok" and source_total > epsilon and target_total > epsilon and positive_supported_pair_od <= 0.0:
        status = "skipped_no_positive_supported_pair"
    imbalance = {
        "source_total_raw": source_total,
        "target_total_raw": target_total,
        "balanced_total": mean_total,
        "relative_imbalance": relative_imbalance,
        "tolerance": config.marginal_tolerance,
        "od_sampler": config.od_sampler,
        "zero_total_reason": (
            "zero_source_total"
            if source_total <= epsilon and target_total > epsilon
            else "zero_target_total"
            if target_total <= epsilon and source_total > epsilon
            else "zero_source_and_target_total"
            if source_total <= epsilon and target_total <= epsilon
            else ""
        ),
        "origin_component_coverage": supported_source_mass / max(source_total, epsilon),
        "destination_component_coverage": supported_target_mass / max(target_total, epsilon),
        "unservable_positive_source_count": int(len(unservable_sources)),
        "unservable_positive_target_count": int(len(unservable_targets)),
        "unservable_positive_source_mass": float(np.sum(source_marginal[unservable_sources], dtype=np.float64)) if unservable_sources else 0.0,
        "unservable_positive_target_mass": float(np.sum(target_marginal[unservable_targets], dtype=np.float64)) if unservable_targets else 0.0,
        "positive_supported_pair_count": int(positive_supported_pair_od),
    }
    return PartCRev1RegionProblem(
        region_uuid=region_uuid,
        region_hash=region_graph.graph_hash,
        region_graph=region_graph,
        global_arc_indices=global_arc_indices,
        source_terminals=source_terms,
        target_terminals=target_terms,
        source_marginal=source_marginal,
        target_marginal=target_marginal,
        support=support,
        terminal_order_hash=terminal_order_hash,
        pair_order_hash=pair_order_hash,
        imbalance=imbalance,
        status=status,
    )


def partC_rev1_sparse_ipf(
    source_marginal: np.ndarray,
    target_marginal: np.ndarray,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    prior_weight: np.ndarray,
    *,
    tol: float = 1e-5,
    max_iter: int = 500,
) -> tuple[np.ndarray, dict[str, Any]]:
    q = np.asarray(prior_weight, dtype=np.float64).copy()
    q[~np.isfinite(q) | (q <= 0.0)] = 1e-12
    source = np.asarray(source_marginal, dtype=np.float64)
    target = np.asarray(target_marginal, dtype=np.float64)
    s_idx = np.asarray(source_idx, dtype=np.int64)
    t_idx = np.asarray(target_idx, dtype=np.int64)
    positive_source = np.flatnonzero(source > 0.0)
    positive_target = np.flatnonzero(target > 0.0)
    if any(not np.any(s_idx == idx) for idx in positive_source):
        return q * 0.0, {"status": "infeasible_missing_positive_source_support", "iterations": 0}
    if any(not np.any(t_idx == idx) for idx in positive_target):
        return q * 0.0, {"status": "infeasible_missing_positive_target_support", "iterations": 0}
    row_error = math.inf
    col_error = math.inf
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        row_sum = np.bincount(s_idx, weights=q, minlength=len(source))
        row_scale = np.ones(len(source), dtype=np.float64)
        mask = source > 0.0
        row_scale[mask] = source[mask] / np.maximum(row_sum[mask], 1e-12)
        row_scale[~mask] = 0.0
        q *= row_scale[s_idx]
        col_sum = np.bincount(t_idx, weights=q, minlength=len(target))
        col_scale = np.ones(len(target), dtype=np.float64)
        mask = target > 0.0
        col_scale[mask] = target[mask] / np.maximum(col_sum[mask], 1e-12)
        col_scale[~mask] = 0.0
        q *= col_scale[t_idx]
        if iterations % 10 == 0 or iterations == 1:
            row_sum = np.bincount(s_idx, weights=q, minlength=len(source))
            col_sum = np.bincount(t_idx, weights=q, minlength=len(target))
            row_error = float(np.max(np.abs(row_sum[positive_source] - source[positive_source]) / np.maximum(source[positive_source], 1e-12))) if len(positive_source) else 0.0
            col_error = float(np.max(np.abs(col_sum[positive_target] - target[positive_target]) / np.maximum(target[positive_target], 1e-12))) if len(positive_target) else 0.0
            if row_error <= tol and col_error <= tol:
                return q, {"status": "converged", "iterations": iterations, "row_error": row_error, "col_error": col_error}
    row_sum = np.bincount(s_idx, weights=q, minlength=len(source))
    col_sum = np.bincount(t_idx, weights=q, minlength=len(target))
    row_error = float(np.max(np.abs(row_sum[positive_source] - source[positive_source]) / np.maximum(source[positive_source], 1e-12))) if len(positive_source) else 0.0
    col_error = float(np.max(np.abs(col_sum[positive_target] - target[positive_target]) / np.maximum(target[positive_target], 1e-12))) if len(positive_target) else 0.0
    return q, {"status": "max_iterations", "iterations": iterations, "row_error": row_error, "col_error": col_error}


def partC_rev1_benchmark_flow_gravity_od_vector(
    source_marginal: np.ndarray,
    target_marginal: np.ndarray,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    tau0: np.ndarray,
    *,
    theta: float,
    tau_scale: float,
    epsilon: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate additive benchmark-flow gravity OD on the existing support.

    For Prompt cap2over3 resume runs this deliberately avoids IPF,
    balancing, distance denominators, degree denominators, and support-mask
    renormalization. If support excludes self or unreachable pairs, the
    generated total is lower and that truncation is reported in metadata.
    """

    source = np.asarray(source_marginal, dtype=np.float64)
    target = np.asarray(target_marginal, dtype=np.float64)
    s_idx = np.asarray(source_idx, dtype=np.int64)
    t_idx = np.asarray(target_idx, dtype=np.int64)
    source_total = float(np.sum(source, dtype=np.float64))
    target_total = float(np.sum(target, dtype=np.float64))
    if source_total <= epsilon or target_total <= epsilon or len(s_idx) == 0:
        reason = "zero_source_total" if source_total <= epsilon else "zero_target_total"
        if source_total <= epsilon and target_total <= epsilon:
            reason = "zero_source_and_target_total"
        return np.zeros(len(s_idx), dtype=np.float64), {"status": "zero_total", "reason": reason, "generated_total": 0.0}
    q = (
        np.maximum(source[s_idx], 0.0) * np.maximum(target[t_idx], 0.0) / max(target_total, epsilon)
        + np.maximum(target[t_idx], 0.0) * np.maximum(source[s_idx], 0.0) / max(source_total, epsilon)
    )
    q[~np.isfinite(q)] = 0.0
    q[q < epsilon] = 0.0
    positive = q > epsilon
    if not bool(np.any(positive)):
        return np.zeros(len(s_idx), dtype=np.float64), {"status": "no_positive_supported_pair", "generated_total": 0.0}
    full_cross_total = source_total + target_total
    generated_total = float(np.sum(q, dtype=np.float64))
    return q.astype(np.float64), {
        "status": "generated",
        "formula": "q_ij=O_i*I_j/I_total+I_j*O_i/O_total",
        "support_mask_renormalization": "none",
        "source_total": source_total,
        "target_total": target_total,
        "full_cross_generated_total": full_cross_total,
        "generated_total": generated_total,
        "support_truncation_loss": float(max(full_cross_total - generated_total, 0.0)),
        "support_retained_share": generated_total / max(full_cross_total, epsilon),
        "positive_pair_count": int(np.count_nonzero(q > epsilon)),
    }


def _partC_rev1_pair_distances(region_graph: PartCRev1GraphSnapshot, support: pd.DataFrame, source_terms: pd.DataFrame, target_terms: pd.DataFrame, cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local_idx = region_graph.node_uid_to_idx
    source_locs = np.asarray([local_idx[str(uid)] for uid in source_terms["node_uid"].astype(str).tolist()], dtype=np.int64)
    target_locs = np.asarray([local_idx[str(uid)] for uid in target_terms["node_uid"].astype(str).tolist()], dtype=np.int64)
    distance = np.full(len(support), np.inf, dtype=np.float64)
    reachable = np.zeros(len(support), dtype=bool)
    for source_idx in sorted(set(support["source_idx"].astype(int).tolist())):
        rows = np.flatnonzero(support["source_idx"].to_numpy(dtype=np.int64) == int(source_idx))
        targets = [int(target_locs[int(t)]) for t in support.iloc[rows]["target_idx"].astype(int).tolist()]
        result = partC_rev1_sssp(region_graph, int(source_locs[int(source_idx)]), cost, backend="numba_csr", targets=targets)
        for pos, target_local in zip(rows, targets, strict=True):
            distance[int(pos)] = float(result.distances[int(target_local)])
            reachable[int(pos)] = bool(result.reached_mask[int(target_local)])
    return distance, reachable


def _partC_rev1_od_snapshot_for_sample(problem: PartCRev1RegionProblem, q: np.ndarray) -> PartCRev1ODSnapshot:
    rows: list[dict[str, Any]] = []
    support = problem.support.reset_index(drop=True)
    for pos, demand in enumerate(q.tolist()):
        if float(demand) <= 1e-12:
            continue
        s_idx = int(support.loc[pos, "source_idx"])
        t_idx = int(support.loc[pos, "target_idx"])
        rows.append(
            {
                "origin_node_uid": str(problem.source_terminals.loc[s_idx, "node_uid"]),
                "destination_node_uid": str(problem.target_terminals.loc[t_idx, "node_uid"]),
                "demand_solver_unit": float(demand),
            }
        )
    return partC_rev1_build_od_snapshot_from_frames(pd.DataFrame(rows), node_uid_to_idx=problem.region_graph.node_uid_to_idx)


def partC_rev1_generate_region_samples(
    problem: PartCRev1RegionProblem,
    *,
    config: PartCRev1RegionalTrainingConfig,
    sample_count: int,
    temp_dir: str | Path,
    reporter: PartCRev1ProgressReporter | None = None,
) -> PartCRev1RegionDataset:
    if problem.status != "ok":
        raise ValueError(f"Cannot generate samples for region {problem.region_uuid} with status {problem.status}")
    support = problem.support.reset_index(drop=True)
    source_idx = support["source_idx"].to_numpy(dtype=np.int64)
    target_idx = support["target_idx"].to_numpy(dtype=np.int64)
    tau0 = support["tau0"].to_numpy(dtype=np.float64)
    tau_scale = float(np.median(tau0[np.isfinite(tau0) & (tau0 > 0.0)])) if np.any(np.isfinite(tau0) & (tau0 > 0.0)) else 1.0
    seed = int(partC_rev1_sha256_text(f"{config.random_seed}:{problem.region_uuid}")[:8], 16)
    rng = np.random.default_rng(seed)
    use_gravity = _partC_rev1_uses_benchmark_flow_gravity(config)
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    table_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    sample_runtimes: list[float] = []
    for experiment_number in range(int(sample_count)):
        sample_started = time.perf_counter()
        s = problem.source_marginal.copy()
        d = problem.target_marginal.copy()
        if experiment_number > 0:
            s_mask = s > 0.0
            d_mask = d > 0.0
            if use_gravity and str(config.terminal_perturbation).lower() == "uniform_0_200":
                s[s_mask] *= rng.uniform(0.0, 2.0, size=int(s_mask.sum()))
                d[d_mask] *= rng.uniform(0.0, 2.0, size=int(d_mask.sum()))
            else:
                s[s_mask] *= np.exp(config.marginal_sigma * rng.normal(size=int(s_mask.sum())) - 0.5 * config.marginal_sigma**2)
                d[d_mask] *= np.exp(config.marginal_sigma * rng.normal(size=int(d_mask.sum())) - 0.5 * config.marginal_sigma**2)
        if use_gravity:
            q, od_meta = partC_rev1_benchmark_flow_gravity_od_vector(
                s,
                d,
                source_idx,
                target_idx,
                tau0,
                theta=float(config.gravity_theta),
                tau_scale=tau_scale,
                epsilon=float(config.gravity_epsilon),
            )
        else:
            total = 0.5 * (float(np.sum(s)) + float(np.sum(d)))
            if total > 0.0:
                s *= total / max(float(np.sum(s)), 1e-12)
                d *= total / max(float(np.sum(d)), 1e-12)
            prior = np.exp(-float(config.gravity_theta) * tau0 / max(tau_scale, 1e-12))
            prior *= np.exp(float(config.pair_noise_sigma) * rng.normal(size=len(tau0)))
            q, od_meta = partC_rev1_sparse_ipf(
                s,
                d,
                source_idx,
                target_idx,
                prior,
                tol=config.ipf_tol,
                max_iter=config.ipf_max_iter,
            )
            if od_meta["status"] not in {"converged", "max_iterations"}:
                raise RuntimeError(f"IPF infeasible for {problem.region_uuid}: {od_meta}")
        regional_gap: float | None = None
        regional_runtime = 0.0
        if np.sum(q) > 0.0:
            regional_od = _partC_rev1_od_snapshot_for_sample(problem, q)
            result = partC_rev1_solve_physical_assignment(
                problem.region_graph,
                regional_od,
                method=config.regional_solver_method,
                config=PartCRev1AssignmentConfig(
                    max_iterations=config.regional_solver_iterations,
                    gap_target=config.regional_gap_target,
                    backend="numba_csr",
                    origin_batch_size=config.regional_origin_batch_size,
                ),
                reporter=None,
            )
            cost = result.cost
            regional_runtime = result.runtime_seconds
            regional_gap = result.histories[-1].relative_gap if result.histories else None
        else:
            cost = problem.region_graph.free_time
        distance, reachable = _partC_rev1_pair_distances(problem.region_graph, support, problem.source_terminals, problem.target_terminals, cost)
        distance = np.where(np.isfinite(distance), distance, tau0 * 10.0 + 1e6)
        x_rows.append(q.astype(np.float64))
        y_rows.append(distance.astype(np.float64))
        for pos, row in support.iterrows():
            table_rows.append(
                {
                    "region_uuid": problem.region_uuid,
                    "experiment_number": int(experiment_number),
                    "pair_position": int(pos),
                    "pair_id": str(row["pair_id"]),
                    "source_terminal_uid": str(row["source_terminal_uid"]),
                    "target_terminal_uid": str(row["target_terminal_uid"]),
                    "demand": float(q[int(pos)]),
                    "distance": float(distance[int(pos)]),
                    "reachable": bool(reachable[int(pos)]),
                    "regional_gap": regional_gap,
                    "regional_runtime_sec": float(regional_runtime),
                    "seed": int(seed + experiment_number),
                    "od_sampler": config.od_sampler,
                    "od_sampler_status": str(od_meta.get("status")),
                    "od_sampler_reason": str(od_meta.get("reason", "")),
                }
            )
        elapsed = time.perf_counter() - sample_started
        sample_runtimes.append(elapsed)
        if reporter is not None:
            resource = partC_rev1_resource_snapshot(started)
            reporter.heartbeat(
                "regional_sample",
                force=True,
                region_uuid=problem.region_uuid,
                experiment_number=experiment_number,
                processed=experiment_number + 1,
                total=sample_count,
                regional_gap=regional_gap,
                sample_runtime_seconds=elapsed,
                projected_region_seconds=median(sample_runtimes) * sample_count,
                memory_rss_mb=resource.memory_rss_mb,
                temp_disk_mb=resource.temp_disk_mb,
            )
    sample_table = pd.DataFrame(table_rows)
    out_dir = partC_rev1_resolve_path(temp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_path = out_dir / f"partC_rev1_training_samples_{problem.region_uuid}.parquet"
    partC_rev1_write_table(sample_table, temp_path, index=False)
    x = np.vstack(x_rows).astype(np.float32)
    y = np.vstack(y_rows).astype(np.float32)
    pair_features = np.column_stack(
        [
            tau0 / max(float(np.max(tau0)), 1e-12),
            support["pair_type"].astype(str).str.contains("route_witness").to_numpy(dtype=np.float32),
            support["pair_type"].astype(str).str.contains("positive_centroid_od").to_numpy(dtype=np.float32),
        ]
    ).astype(np.float32)
    metadata = {
        "region_uuid": problem.region_uuid,
        "region_hash": problem.region_hash,
        "sample_count": int(sample_count),
        "pair_count": int(len(support)),
        "pair_order_hash": problem.pair_order_hash,
        "terminal_order_hash": problem.terminal_order_hash,
        "runtime_seconds": time.perf_counter() - started,
        "median_sample_seconds": float(median(sample_runtimes)) if sample_runtimes else None,
        "training_data_hash": partC_rev1_numpy_array_hash(x, y),
        "od_sampler": config.od_sampler,
        "all_generated_od_zero": bool(np.all(x <= float(config.gravity_epsilon))),
        "support_diagnostics": problem.imbalance,
        "split_manifest": partC_rev1_nested_training_split_manifest(sample_count, fractions=config.split_train_fractions),
    }
    return PartCRev1RegionDataset(problem=problem, sample_table=sample_table, x=x, y=y, tau0=tau0.astype(np.float32), pair_features=pair_features, temp_path=temp_path, metadata=metadata)


def _partC_rev1_split_indices(n: int) -> dict[str, np.ndarray]:
    indices = np.arange(n, dtype=np.int64)
    if n <= 0:
        empty = np.array([], dtype=np.int64)
        return {"train": empty, "validation": empty, "test": empty}
    if n == 1:
        empty = np.array([], dtype=np.int64)
        return {"train": indices[:1], "validation": empty, "test": empty}
    if n == 2:
        empty = np.array([], dtype=np.int64)
        return {"train": indices[:1], "validation": indices[1:2], "test": empty}
    if n == 3:
        return {"train": indices[:1], "validation": indices[1:2], "test": indices[2:3]}
    train_end = max(1, int(math.floor(0.8 * n)))
    validation_end = max(train_end + 1, int(math.floor(0.9 * n)))
    validation_end = min(validation_end, n)
    return {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end] if train_end < validation_end else indices[:1],
        "test": indices[validation_end:] if validation_end < n else indices[-1:],
    }


def partC_rev1_nested_training_split_manifest(
    n: int,
    *,
    fractions: Iterable[float] = (0.10, 0.25, 0.50, 0.75, 1.00),
) -> dict[str, Any]:
    base = _partC_rev1_split_indices(int(n))
    train_pool = np.asarray(base["train"], dtype=np.int64)
    if len(train_pool) and int(train_pool[0]) != 0:
        train_pool = np.concatenate([np.array([0], dtype=np.int64), train_pool[train_pool != 0]])
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        fraction_float = float(fraction)
        take = max(1, int(math.ceil(fraction_float * len(train_pool)))) if len(train_pool) else 0
        take = min(take, len(train_pool))
        rows.append(
            {
                "train_fraction": fraction_float,
                "train_ids": train_pool[:take].astype(int).tolist(),
                "validation_ids": np.asarray(base["validation"], dtype=np.int64).astype(int).tolist(),
                "test_ids": np.asarray(base["test"], dtype=np.int64).astype(int).tolist(),
            }
        )
    return {
        "sample_count": int(n),
        "sample_zero_in_training_pool": bool(0 in set(train_pool.astype(int).tolist())),
        "train_pool_ids": train_pool.astype(int).tolist(),
        "validation_ids": np.asarray(base["validation"], dtype=np.int64).astype(int).tolist(),
        "test_ids": np.asarray(base["test"], dtype=np.int64).astype(int).tolist(),
        "fractions": rows,
        "split_hash": partC_rev1_json_hash(rows),
    }


def _partC_rev1_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    abs_err = np.abs(err)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(err)))) if err.size else 0.0,
        "mae": float(np.mean(abs_err)) if err.size else 0.0,
        "wmape": float(np.sum(abs_err) / max(np.sum(np.abs(y_true)), 1e-12)) if err.size else 0.0,
        "max_abs_error": float(np.max(abs_err)) if err.size else 0.0,
    }


class PartCRev1SimpleDecoderNN(nn.Module):
    def __init__(self, pair_count: int, feature_count: int, tau0: np.ndarray, latent_dim: int = 16) -> None:
        super().__init__()
        self.register_buffer("tau0", torch.as_tensor(tau0, dtype=torch.float32))
        self.encoder = nn.Sequential(nn.Linear(pair_count, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim + feature_count + 1, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1), nn.Softplus())

    def forward(self, x: torch.Tensor, pair_features: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        batch, pair_count = x.shape
        h_expanded = h[:, None, :].expand(batch, pair_count, h.shape[-1])
        direct = x[:, :, None]
        features = pair_features[None, :, :].expand(batch, pair_count, pair_features.shape[-1])
        decoded = self.decoder(torch.cat([h_expanded, features, direct], dim=-1)).squeeze(-1)
        return self.tau0[None, :] + decoded


class PartCRev1DenseICNNPotential(nn.Module):
    def __init__(self, pair_count: int, tau0: np.ndarray, hidden_layers: Iterable[int] = (16,), mu: float = 1e-3) -> None:
        super().__init__()
        self.pair_count = int(pair_count)
        self.mu = float(mu)
        self.register_buffer("tau0", torch.as_tensor(tau0, dtype=torch.float32))
        layers = list(hidden_layers)
        self.input_layers = nn.ParameterList()
        self.hidden_layers = nn.ParameterList()
        self.biases = nn.ParameterList()
        prev = 0
        for width in layers:
            self.input_layers.append(nn.Parameter(torch.randn(width, pair_count) * 0.03))
            if prev:
                self.hidden_layers.append(nn.Parameter(torch.randn(width, prev) * 0.03))
            else:
                self.hidden_layers.append(nn.Parameter(torch.empty(width, 0)))
            self.biases.append(nn.Parameter(torch.zeros(width)))
            prev = width
        self.output_raw = nn.Parameter(torch.randn(prev) * 0.03 if prev else torch.zeros(1))
        self.kappa_raw = nn.Parameter(torch.zeros(pair_count))

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        z: torch.Tensor | None = None
        for input_raw, hidden_raw, bias in zip(self.input_layers, self.hidden_layers, self.biases, strict=True):
            value = torch.nn.functional.linear(x, torch.nn.functional.softplus(input_raw), bias)
            if z is not None and hidden_raw.numel():
                value = value + torch.nn.functional.linear(z, torch.nn.functional.softplus(hidden_raw), None)
            z = torch.nn.functional.softplus(value)
        extra = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        if z is not None:
            extra = z @ torch.nn.functional.softplus(self.output_raw)
        quad = 0.5 * ((torch.nn.functional.softplus(self.kappa_raw) + self.mu) * x * x).sum(dim=1)
        return (x * self.tau0[None, :]).sum(dim=1) + extra + quad

    def costs(self, x: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
        xx = x.clone().detach().requires_grad_(True) if not x.requires_grad else x
        potential = self.potential(xx).sum()
        grad = torch.autograd.grad(potential, xx, create_graph=create_graph, retain_graph=create_graph)[0]
        return grad


class PartCRev1LowDimICNNPotential(nn.Module):
    def __init__(self, tau0: np.ndarray, embeddings: np.ndarray, hidden_units: int = 8, mu: float = 1e-3) -> None:
        super().__init__()
        emb = np.asarray(embeddings, dtype=np.float32)
        self.register_buffer("tau0", torch.as_tensor(tau0, dtype=torch.float32))
        self.register_buffer("embeddings", torch.as_tensor(np.maximum(emb, 0.0), dtype=torch.float32))
        self.weight_raw = nn.Parameter(torch.randn(emb.shape[1], hidden_units) * 0.03)
        self.bias = nn.Parameter(torch.zeros(hidden_units))
        self.kappa_raw = nn.Parameter(torch.zeros(len(tau0)))
        self.mu = float(mu)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.embeddings
        w = torch.nn.functional.softplus(self.weight_raw)
        psi = torch.nn.functional.softplus(z @ w + self.bias).sum(dim=1)
        quad = 0.5 * ((torch.nn.functional.softplus(self.kappa_raw) + self.mu) * x * x).sum(dim=1)
        return (x * self.tau0[None, :]).sum(dim=1) + psi + quad

    def costs(self, x: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
        xx = x.clone().detach().requires_grad_(True) if not x.requires_grad else x
        potential = self.potential(xx).sum()
        return torch.autograd.grad(potential, xx, create_graph=create_graph, retain_graph=create_graph)[0]

    def query_active_pair_costs(self, x: torch.Tensor, pair_indices: torch.Tensor) -> torch.Tensor:
        z = x @ self.embeddings
        w = torch.nn.functional.softplus(self.weight_raw)
        grad_z = torch.sigmoid(z @ w + self.bias) @ w.T
        kappa = torch.nn.functional.softplus(self.kappa_raw) + self.mu
        full = self.tau0[None, :] + grad_z @ self.embeddings.T + kappa[None, :] * x
        return full[:, pair_indices]


def _partC_rev1_train_torch_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    pair_features: np.ndarray,
    *,
    epochs: int,
    lr: float,
    is_icnn: bool,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.as_tensor(x_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.float32)
    pf = torch.as_tensor(pair_features, dtype=torch.float32)
    for _epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        if is_icnn:
            xb = xt.clone().detach().requires_grad_(True)
            pred = model.costs(xb, create_graph=True)  # type: ignore[attr-defined]
        else:
            pred = model(xt, pf)  # type: ignore[operator]
        loss = torch.mean((pred - yt) ** 2)
        loss.backward()
        optimizer.step()


def _partC_rev1_predict_torch(model: nn.Module, x: np.ndarray, pair_features: np.ndarray, *, is_icnn: bool) -> np.ndarray:
    xt = torch.as_tensor(x, dtype=torch.float32)
    pf = torch.as_tensor(pair_features, dtype=torch.float32)
    if is_icnn:
        pred = model.costs(xt, create_graph=False)  # type: ignore[attr-defined]
    else:
        with torch.no_grad():
            pred = model(xt, pf)  # type: ignore[operator]
    return pred.detach().cpu().numpy()


def _partC_rev1_monotonicity_diagnostic(model: nn.Module, x: np.ndarray, pair_features: np.ndarray, *, is_icnn: bool, seed: int) -> dict[str, float]:
    if len(x) == 0:
        return {"monotonicity_violation_rate": 0.0, "min_monotonicity_dot": 0.0}
    rng = np.random.default_rng(seed)
    dots: list[float] = []
    for _ in range(8):
        a = x[rng.integers(0, len(x)) : rng.integers(0, len(x)) + 1]
        if len(a) == 0:
            a = x[:1]
        b = x[rng.integers(0, len(x)) : rng.integers(0, len(x)) + 1]
        if len(b) == 0:
            b = x[-1:]
        ca = _partC_rev1_predict_torch(model, a, pair_features, is_icnn=is_icnn)[0]
        cb = _partC_rev1_predict_torch(model, b, pair_features, is_icnn=is_icnn)[0]
        dots.append(float(np.dot(ca - cb, a[0] - b[0])))
    return {
        "monotonicity_violation_rate": float(np.mean(np.asarray(dots) < -1e-6)),
        "min_monotonicity_dot": float(np.min(dots)) if dots else 0.0,
    }


def partC_rev1_load_torch_model_bundle(path: str | Path) -> nn.Module:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    kind = str(bundle["model_kind"])
    metadata = dict(bundle["metadata"])
    tau0 = np.asarray(bundle["tau0"], dtype=np.float32)
    if kind == "simple_nn":
        model = PartCRev1SimpleDecoderNN(int(metadata["pair_count"]), int(metadata["feature_count"]), tau0, int(metadata["latent_dim"]))
    elif kind in {"icnn_shallow", "icnn_deep"}:
        model = PartCRev1DenseICNNPotential(int(metadata["pair_count"]), tau0, metadata["hidden_layers"], float(metadata["mu"]))
    else:
        raise ValueError(f"Unsupported torch model kind: {kind}")
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model


def partC_rev1_train_region_models(
    dataset: PartCRev1RegionDataset,
    *,
    output_dir: str | Path,
    config: PartCRev1RegionalTrainingConfig,
    reporter: PartCRev1ProgressReporter | None = None,
) -> list[dict[str, Any]]:
    out_dir = partC_rev1_resolve_path(output_dir) / "partC_rev1_4D_models" / dataset.problem.region_uuid
    out_dir.mkdir(parents=True, exist_ok=True)
    x = dataset.x.astype(np.float32)
    y = dataset.y.astype(np.float32)
    tau0 = dataset.tau0.astype(np.float32)
    pair_features = dataset.pair_features.astype(np.float32)
    x_scale = max(float(np.max(x)), 1.0)
    y_scale = max(float(np.max(y)), float(np.max(tau0)), 1.0)
    x_norm = x / x_scale
    y_norm = y / y_scale
    tau0_norm = tau0 / y_scale
    splits = _partC_rev1_split_indices(len(x_norm))
    rows: list[dict[str, Any]] = []
    seed = int(partC_rev1_sha256_text(dataset.problem.region_uuid)[:8], 16)
    model_specs = [
        ("xgboost", "joblib"),
        ("simple_nn", "torch"),
        ("icnn_shallow", "torch"),
        ("icnn_deep", "torch"),
    ]
    for model_name, storage in model_specs:
        started = time.perf_counter()
        if reporter is not None:
            reporter.emit("regional_model_training_started", region_uuid=dataset.problem.region_uuid, model=model_name)
        if model_name == "xgboost":
            from sklearn.multioutput import MultiOutputRegressor
            from xgboost import XGBRegressor

            model = MultiOutputRegressor(
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=int(config.xgb_estimators),
                    max_depth=2,
                    learning_rate=0.15,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    n_jobs=1,
                    random_state=config.random_seed,
                    verbosity=0,
                )
            )
            model.fit(x_norm[splits["train"]], y_norm[splits["train"]])
            pred_norm = model.predict(x_norm[splits["test"]])
            artifact_path = out_dir / "partC_rev1_model_xgboost.joblib"
            joblib.dump({"model": model, "metadata": dataset.metadata, "x_scale": x_scale, "y_scale": y_scale}, artifact_path)
            loaded = joblib.load(artifact_path)
            _ = loaded["model"].predict(x_norm[:1])
            pred = pred_norm * y_scale
            diagnostics = {"monotonicity_violation_rate": None, "min_monotonicity_dot": None}
            param_count = 0
        elif model_name == "simple_nn":
            model = PartCRev1SimpleDecoderNN(x_norm.shape[1], pair_features.shape[1], tau0_norm, latent_dim=16)
            _partC_rev1_train_torch_model(model, x_norm[splits["train"]], y_norm[splits["train"]], pair_features, epochs=config.nn_epochs, lr=0.01, is_icnn=False)
            pred = _partC_rev1_predict_torch(model, x_norm[splits["test"]], pair_features, is_icnn=False) * y_scale
            artifact_path = out_dir / "partC_rev1_model_simple_nn.pt"
            metadata = {**dataset.metadata, "pair_count": int(x_norm.shape[1]), "feature_count": int(pair_features.shape[1]), "latent_dim": 16}
            torch.save({"model_kind": "simple_nn", "state_dict": model.state_dict(), "metadata": metadata, "tau0": tau0_norm.tolist()}, artifact_path)
            loaded_model = partC_rev1_load_torch_model_bundle(artifact_path)
            _ = _partC_rev1_predict_torch(loaded_model, x_norm[:1], pair_features, is_icnn=False)
            diagnostics = _partC_rev1_monotonicity_diagnostic(model, x_norm, pair_features, is_icnn=False, seed=seed)
            param_count = int(sum(p.numel() for p in model.parameters()))
        else:
            hidden = (16,) if model_name == "icnn_shallow" else (24, 24, 16)
            model = PartCRev1DenseICNNPotential(x_norm.shape[1], tau0_norm, hidden_layers=hidden, mu=1e-3)
            _partC_rev1_train_torch_model(model, x_norm[splits["train"]], y_norm[splits["train"]], pair_features, epochs=config.icnn_epochs, lr=0.01, is_icnn=True)
            pred = _partC_rev1_predict_torch(model, x_norm[splits["test"]], pair_features, is_icnn=True) * y_scale
            artifact_path = out_dir / f"partC_rev1_model_{model_name}.pt"
            metadata = {**dataset.metadata, "pair_count": int(x_norm.shape[1]), "hidden_layers": list(hidden), "mu": 1e-3}
            torch.save({"model_kind": model_name, "state_dict": model.state_dict(), "metadata": metadata, "tau0": tau0_norm.tolist()}, artifact_path)
            loaded_model = partC_rev1_load_torch_model_bundle(artifact_path)
            _ = _partC_rev1_predict_torch(loaded_model, x_norm[:1], pair_features, is_icnn=True)
            diagnostics = _partC_rev1_monotonicity_diagnostic(model, x_norm, pair_features, is_icnn=True, seed=seed)
            param_count = int(sum(p.numel() for p in model.parameters()))
        metrics = _partC_rev1_metrics(y[splits["test"]], pred)
        lower_bound_violations = float(np.mean(pred < (tau0[None, :] - 1e-5))) if pred.size else 0.0
        runtime = time.perf_counter() - started
        row = {
            "region_uuid": dataset.problem.region_uuid,
            "model": model_name,
            "storage": storage,
            "status": "verified",
            "artifact_path": str(artifact_path),
            "region_hash": dataset.problem.region_hash,
            "terminal_order_hash": dataset.problem.terminal_order_hash,
            "pair_order_hash": dataset.problem.pair_order_hash,
            "training_data_hash": dataset.metadata["training_data_hash"],
            "sample_count": int(len(x)),
            "pair_count": int(x.shape[1]),
            "train_count": int(len(splits["train"])),
            "validation_count": int(len(splits["validation"])),
            "test_count": int(len(splits["test"])),
            "runtime_seconds": runtime,
            "parameter_count": param_count,
            "file_size_bytes": int(Path(artifact_path).stat().st_size),
            "freeflow_lower_bound_violation_rate": lower_bound_violations,
            **metrics,
            **diagnostics,
        }
        rows.append(row)
        metadata_path = out_dir / f"partC_rev1_model_{model_name}_metadata.json"
        partC_rev1_write_json(row, metadata_path)
        if reporter is not None:
            reporter.emit("regional_model_training_finished", **row)
    return rows


def _partC_rev1_write_model_registry(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    registry = pd.DataFrame(rows)
    csv_path = output_dir / "partC_rev1_4D_model_registry.csv"
    md_path = output_dir / "partC_rev1_4D_model_registry.md"
    partC_rev1_write_table(registry, csv_path, index=False)
    lines = [
        "# partC_rev1 4D Model Registry",
        "",
        f"Generated UTC: {_partC_rev1_utc_now()}",
        "",
        "| Region | Model | Status | Samples | Pairs | RMSE | WMAPE | Runtime seconds |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("region_uuid")),
                    str(row.get("model")),
                    str(row.get("status")),
                    str(row.get("sample_count")),
                    str(row.get("pair_count")),
                    f"{float(row.get('rmse', 0.0)):.6g}",
                    f"{float(row.get('wmape', 0.0)):.6g}",
                    f"{float(row.get('runtime_seconds', 0.0)):.3f}",
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def partC_rev1_run_regional_training_stage(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    *,
    benchmark_flow: np.ndarray,
    route_witnesses: pd.DataFrame | None,
    selected_benchmark_hash: str | None = None,
    directed_flow_path: str | None = None,
    config: PartCRev1RegionalTrainingConfig,
    output_dir: str | Path,
    cache_dir: str | Path,
    reporter: PartCRev1ProgressReporter | None = None,
) -> dict[str, Any]:
    out_dir = partC_rev1_resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = partC_rev1_resolve_path(cache_dir) / "partC_rev1_training_tmp"
    cache_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    incident_counts = partC_rev1_incident_label_counts(graph)
    centroid_mask = partC_rev1_centroid_node_mask(graph, od)
    od_source, od_target = partC_rev1_od_marginals_by_node(graph, od)
    region_table = partC_rev1_region_size_table(graph, od)
    terminal_registry_path = out_dir / "partC_rev1_4D_region_terminal_registry.csv"
    partC_rev1_write_table(region_table, terminal_registry_path, index=False)
    candidates = partC_rev1_select_stratified_regions(
        region_table,
        count=config.pilot_region_count,
        max_pilot_region_arcs=config.max_pilot_region_arcs,
    )
    completed_regions: list[dict[str, Any]] = []
    skipped_regions: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    cleanup_rows: list[dict[str, Any]] = []
    runtimes: list[float] = []
    for region_uuid in candidates:
        if len(completed_regions) >= int(config.pilot_region_count):
            break
        region_started = time.perf_counter()
        if reporter is not None:
            reporter.emit("regional_training_region_started", region_uuid=region_uuid)
        try:
            problem = partC_rev1_build_region_problem(
                graph,
                od,
                region_uuid=region_uuid,
                benchmark_flow=benchmark_flow,
                config=config,
                incident_label_count=incident_counts,
                centroid_mask=centroid_mask,
                od_source_marginal=od_source,
                od_target_marginal=od_target,
                route_witnesses=route_witnesses,
            )
            if problem.status != "ok":
                skipped_regions.append({"region_uuid": region_uuid, "status": problem.status, "imbalance": problem.imbalance})
                continue
            dataset = partC_rev1_generate_region_samples(
                problem,
                config=config,
                sample_count=config.pilot_samples_per_region,
                temp_dir=cache_root,
                reporter=reporter,
            )
            region_model_rows = partC_rev1_train_region_models(dataset, output_dir=out_dir, config=config, reporter=reporter)
            model_rows.extend(region_model_rows)
            exemplar_path = out_dir / f"partC_rev1_4D_exemplar_samples_{region_uuid}.parquet"
            partC_rev1_write_table(dataset.sample_table.head(min(len(dataset.sample_table), 500)), exemplar_path, index=False)
            deleted = False
            if dataset.temp_path.exists() and all(row.get("status") == "verified" for row in region_model_rows):
                dataset.temp_path.unlink()
                deleted = True
            cleanup_rows.append(
                {
                    "region_uuid": region_uuid,
                    "temp_path": str(dataset.temp_path),
                    "deleted_after_verified_model_save": deleted,
                    "exemplar_path": str(exemplar_path),
                }
            )
            runtime = time.perf_counter() - region_started
            runtimes.append(runtime)
            completed_regions.append(
                {
                    "region_uuid": region_uuid,
                    "status": "measured_pilot_complete",
                    "runtime_seconds": runtime,
                    "sample_count": int(config.pilot_samples_per_region),
                    "pair_count": int(len(problem.support)),
                    "source_terminal_count": int(len(problem.source_terminals)),
                    "target_terminal_count": int(len(problem.target_terminals)),
                    "region_hash": problem.region_hash,
                    "terminal_order_hash": problem.terminal_order_hash,
                    "pair_order_hash": problem.pair_order_hash,
                    "imbalance": problem.imbalance,
                }
            )
        except Exception as exc:
            skipped_regions.append({"region_uuid": region_uuid, "status": "failed", "error": repr(exc)})
            if reporter is not None:
                reporter.emit("regional_training_region_failed", region_uuid=region_uuid, error=repr(exc))
        finally:
            gc.collect()
    registry_csv, registry_md = _partC_rev1_write_model_registry(model_rows, out_dir)
    cleanup_path = out_dir / "partC_rev1_4D_cleanup_manifest.csv"
    partC_rev1_write_table(pd.DataFrame(cleanup_rows), cleanup_path, index=False)
    eligible_count = int(region_table["terminal_proxy_count"].gt(1).sum())
    pilot_median = float(median(runtimes)) if runtimes else None
    sample_scale = float(config.requested_samples_per_region) / max(float(config.pilot_samples_per_region), 1.0)
    projected_all_region_seconds = pilot_median * eligible_count * sample_scale if pilot_median is not None else None
    state = "measured_partial_projected"
    production_icnn_family = "icnn_shallow"
    summary = {
        "stage": PARTC_REV1_4D_STAGE,
        "created_utc": _partC_rev1_utc_now(),
        "config": asdict(config),
        "graph_hash": graph.graph_hash,
        "topology_hash": getattr(graph, "topology_hash", ""),
        "backbone_hash": getattr(graph, "backbone_hash", ""),
        "assignment_graph_hash": getattr(graph, "assignment_graph_hash", graph.graph_hash),
        "od_hash": od.od_hash,
        "partition_hash": graph.partition_hash,
        "selected_benchmark_hash": selected_benchmark_hash,
        "selected_benchmark_directed_flow_path": directed_flow_path,
        "route_witness_rows": int(len(route_witnesses)) if route_witnesses is not None else 0,
        "runtime_seconds": time.perf_counter() - started,
        "state": state,
        "eligible_region_count": eligible_count,
        "total_nonbackbone_regions": int(len(region_table)),
        "pilot_region_count_requested": int(config.pilot_region_count),
        "pilot_region_count_completed": int(len(completed_regions)),
        "requested_samples_per_region": int(config.requested_samples_per_region),
        "pilot_samples_per_region": int(config.pilot_samples_per_region),
        "projected_all_region_seconds": projected_all_region_seconds,
        "runtime_guard_seconds": float(config.runtime_guard_seconds),
        "completed_regions": completed_regions,
        "skipped_regions": skipped_regions,
        "model_registry_csv": str(registry_csv),
        "model_registry_md": str(registry_md),
        "terminal_registry_csv": str(terminal_registry_path),
        "cleanup_manifest_csv": str(cleanup_path),
        "production_icnn_family": production_icnn_family,
        "coverage_note": "Four model families were trained and reload-verified on measured pilot regions; all-region 100-sample coverage is projected only because the pilot was capped to feasible regional sizes.",
    }
    summary_path = out_dir / "partC_rev1_4D_summary.json"
    partC_rev1_write_json(summary, summary_path)
    summary["summary_path"] = str(summary_path)
    return summary


def partC_rev1_write_4d_handoff(summary: dict[str, Any], *, output_dir: str | Path, reporter: PartCRev1ProgressReporter | None = None) -> Path:
    out_dir = partC_rev1_resolve_path(output_dir)
    handoff_path = out_dir / "partC_rev1_4D_handoff.md"
    lines = [
        "# partC_rev1 4D Handoff",
        "",
        f"Generated UTC: {_partC_rev1_utc_now()}",
        "",
        "## Status",
        "",
        f"- Stage state: `{summary.get('state')}`",
        f"- Completed pilot regions: {summary.get('pilot_region_count_completed')}/{summary.get('pilot_region_count_requested')}",
        f"- Eligible non-backbone regions: {summary.get('eligible_region_count')}",
        f"- Requested samples per region for projection: {summary.get('requested_samples_per_region')}",
        f"- Pilot samples per completed region: {summary.get('pilot_samples_per_region')}",
        f"- Projected all-region seconds: {summary.get('projected_all_region_seconds')}",
        f"- Runtime guard seconds: {summary.get('runtime_guard_seconds')}",
        f"- Selected production ICNN family for 4E: `{summary.get('production_icnn_family')}`",
        f"- Selected 4C benchmark hash: `{summary.get('selected_benchmark_hash')}`",
        f"- Selected 4C directed flow: `{summary.get('selected_benchmark_directed_flow_path')}`",
        f"- Route witness rows loaded: {summary.get('route_witness_rows')}",
        "",
        "## Artifacts",
        "",
        f"- Summary JSON: `{summary.get('summary_path')}`",
        f"- Model registry CSV: `{summary.get('model_registry_csv')}`",
        f"- Model registry Markdown: `{summary.get('model_registry_md')}`",
        f"- Terminal registry CSV: `{summary.get('terminal_registry_csv')}`",
        f"- Cleanup manifest CSV: `{summary.get('cleanup_manifest_csv')}`",
    ]
    if reporter is not None:
        lines.extend([f"- Stage log: `{reporter.log_path}`", f"- Progress JSONL: `{reporter.progress_path}`"])
    lines.extend(["", "## Completed Regions", "", "| Region | Samples | Pairs | Runtime seconds | Pair hash |", "| --- | ---: | ---: | ---: | --- |"])
    for row in summary.get("completed_regions", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("region_uuid")),
                    str(row.get("sample_count")),
                    str(row.get("pair_count")),
                    f"{float(row.get('runtime_seconds', 0.0)):.3f}",
                    str(row.get("pair_order_hash")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- All-region training was not launched unless the pilot projection fit under the 24-hour stage guard.",
            "- Temporary bulk sample files were deleted only after model save/load verification; compact exemplars and model registries remain.",
            "",
            "## Tests",
            "",
            "- Focused 4D and full post-stage tests are recorded in the execution manifest/current-state docs after commit.",
            "- Environment note: `conda run -n py312` emits the known `conda-libmamba-solver` entry-point warning.",
            "",
            "## Exact 4E Command",
            "",
            "```text",
            f"conda run -n py312 python -m partC_rev1_src.partC_rev1_cli run-sparse-global-solver --config {summary.get('config', {}).get('config_path', 'partC_rev1_configs/partC_rev1_real_preprocess_config.json')} --confirm-full-run",
            "```",
            "",
        ]
    )
    handoff_path.write_text("\n".join(lines), encoding="utf-8")
    return handoff_path


def partC_rev1_read_route_witnesses(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        p = partC_rev1_resolve_path(path)
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def partC_rev1_load_benchmark_flow(graph: PartCRev1GraphSnapshot, directed_arc_flow_path: str | Path) -> np.ndarray:
    path = partC_rev1_resolve_path(directed_arc_flow_path)
    df = pd.read_parquet(path, columns=["arc_idx", "flow"])
    flow = np.zeros(len(graph.arc_idx_to_directed_arc_id), dtype=np.float64)
    flow[df["arc_idx"].to_numpy(dtype=np.int64)] = df["flow"].to_numpy(dtype=np.float64)
    return flow
