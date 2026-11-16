"""Prompt 3A igraph-Leiden partitioning for non-backbone physical nodes."""

from __future__ import annotations

import random
import time
from typing import Any

import igraph as ig
import numpy as np
import pandas as pd

from .partC_rev1_ids import partC_rev1_stable_id


def _partC_rev1_arc_fftt(arcs: pd.DataFrame) -> pd.Series:
    for column in ("partC_rev1_fftt_min", "free_time_min"):
        if column in arcs.columns:
            values = pd.to_numeric(arcs[column], errors="coerce")
            if values.notna().any():
                return values
    return pd.Series(np.nan, index=arcs.index)


def _partC_rev1_arc_capacity(arcs: pd.DataFrame) -> pd.Series:
    for column in ("partC_rev1_capacity_annual_ktons_dir", "capacity"):
        if column in arcs.columns:
            values = pd.to_numeric(arcs[column], errors="coerce")
            if values.notna().any():
                return values
    return pd.Series(1.0, index=arcs.index)


def _partC_rev1_backbone_nodes(nodes: pd.DataFrame) -> set[str]:
    if nodes.empty or "partC_rev1_node_uid" not in nodes.columns:
        return set()
    flag = nodes.get("is_backbone_node", nodes.get("is_seed_backbone_node", pd.Series(False, index=nodes.index)))
    return set(nodes.loc[flag.fillna(False).astype(bool), "partC_rev1_node_uid"].map(partC_rev1_stable_id).astype(str))


def partC_rev1_build_partition_graph(
    nodes: pd.DataFrame,
    nonbackbone_arcs: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build an igraph graph using only non-backbone physical nodes and arcs."""

    del config
    node_table = nodes.copy()
    node_table["partC_rev1_node_uid"] = node_table["partC_rev1_node_uid"].map(partC_rev1_stable_id)
    backbone_nodes = _partC_rev1_backbone_nodes(node_table)
    vertex_ids = sorted(set(node_table["partC_rev1_node_uid"].dropna().astype(str)) - backbone_nodes)
    vertex_index = {node_id: pos for pos, node_id in enumerate(vertex_ids)}
    graph_arcs = nonbackbone_arcs.copy()
    if graph_arcs.empty:
        graph_arcs = graph_arcs.assign(weight=pd.Series(dtype=float), weight_source=pd.Series(dtype=str))
    else:
        if "tail_node_uid" not in graph_arcs.columns:
            graph_arcs["tail_node_uid"] = graph_arcs["from_node_uid"]
        if "head_node_uid" not in graph_arcs.columns:
            graph_arcs["head_node_uid"] = graph_arcs["to_node_uid"]
        graph_arcs["tail_node_uid"] = graph_arcs["tail_node_uid"].map(partC_rev1_stable_id)
        graph_arcs["head_node_uid"] = graph_arcs["head_node_uid"].map(partC_rev1_stable_id)
        graph_arcs = graph_arcs.loc[
            graph_arcs["tail_node_uid"].astype(str).isin(vertex_index)
            & graph_arcs["head_node_uid"].astype(str).isin(vertex_index)
            & graph_arcs["tail_node_uid"].ne(graph_arcs["head_node_uid"])
        ].copy()
        graph_arcs["partC_rev1_partition_fftt_min"] = _partC_rev1_arc_fftt(graph_arcs).fillna(1.0).clip(lower=1e-9)
        graph_arcs["partC_rev1_partition_capacity"] = _partC_rev1_arc_capacity(graph_arcs).fillna(1.0).clip(lower=0.0)
        weights = graph_arcs["partC_rev1_partition_capacity"] / graph_arcs["partC_rev1_partition_fftt_min"].clip(lower=1e-9)
        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=1e-9)
        graph_arcs["weight"] = weights
        graph_arcs["weight_source"] = "capacity_over_fftt"

    edges = [(vertex_index[str(row.tail_node_uid)], vertex_index[str(row.head_node_uid)]) for row in graph_arcs.itertuples(index=False)]
    graph = ig.Graph(n=len(vertex_ids), edges=edges, directed=False)
    graph.vs["partC_rev1_node_uid"] = vertex_ids
    if len(edges):
        graph.es["weight"] = graph_arcs["weight"].astype(float).tolist()
    vertices = pd.DataFrame({"partC_rev1_node_uid": vertex_ids, "is_internal_region_vertex": True})
    edge_columns = ["directed_arc_id", "partC_rev1_link_uid", "tail_node_uid", "head_node_uid", "weight", "weight_source"]
    for column in edge_columns:
        if column not in graph_arcs.columns:
            graph_arcs[column] = pd.NA
    edges_table = graph_arcs[edge_columns].copy().reset_index(drop=True)
    return {
        "graph": graph,
        "vertices": vertices,
        "edges": edges_table,
        "backbone_nodes": backbone_nodes,
    }


def partC_rev1_run_leiden_partition(
    nodes: pd.DataFrame,
    nonbackbone_arcs: pd.DataFrame,
    config: dict[str, Any],
    *,
    resolution: float | None = None,
    random_seed: int | None = None,
) -> dict[str, pd.DataFrame | float]:
    """Run Leiden and return persistent physical-node partition tables."""

    partition_cfg = config.get("backbone_partition", {}).get("partition", {})
    resolution_value = float(partition_cfg.get("resolution", 1.0) if resolution is None else resolution)
    seed_value = int(partition_cfg.get("random_seed", 42) if random_seed is None else random_seed)
    built = partC_rev1_build_partition_graph(nodes, nonbackbone_arcs, config)
    graph: ig.Graph = built["graph"]

    start = time.perf_counter()
    if graph.vcount() == 0:
        membership: list[int] = []
    elif graph.ecount() == 0:
        membership = list(range(graph.vcount()))
    else:
        rng = random.Random(seed_value)
        ig.set_random_number_generator(rng)
        try:
            clustering = graph.community_leiden(
                objective_function=str(partition_cfg.get("objective_function", "modularity")),
                weights="weight",
                resolution=resolution_value,
                beta=float(partition_cfg.get("beta", 0.01)),
                n_iterations=int(partition_cfg.get("n_iterations", -1)),
            )
        finally:
            ig.set_random_number_generator(None)
        membership = list(clustering.membership)
    leiden_seconds = time.perf_counter() - start

    vertex_ids = list(graph.vs["partC_rev1_node_uid"]) if graph.vcount() else []
    raw_to_members: dict[int, list[str]] = {}
    for node_id, raw_region in zip(vertex_ids, membership):
        raw_to_members.setdefault(int(raw_region), []).append(str(node_id))
    ordered = sorted(raw_to_members.values(), key=lambda members: min(members))
    region_by_node: dict[str, str] = {}
    region_rows: list[dict[str, Any]] = []
    for pos, members in enumerate(ordered, start=1):
        region_id = f"partC_rev1_region_{pos:08d}"
        for node_id in sorted(members):
            region_by_node[node_id] = region_id
        region_rows.append(
            {
                "region_id": region_id,
                "min_node_uid": min(members),
                "node_count": len(members),
                "member_node_uids": "|".join(sorted(members)),
                "resolution": resolution_value,
                "random_seed": seed_value,
            }
        )
    partition = pd.DataFrame(
        [
            {
                "partC_rev1_node_uid": node_id,
                "region_id": region_by_node[node_id],
                "resolution": resolution_value,
                "random_seed": seed_value,
            }
            for node_id in sorted(region_by_node)
        ]
    )
    regions = pd.DataFrame(region_rows)
    metadata = pd.DataFrame(
        [
            {
                "partition_method": "igraph_leiden",
                "resolution": resolution_value,
                "random_seed": seed_value,
                "nonbackbone_physical_nodes": len(vertex_ids),
                "partition_graph_edges": graph.ecount(),
                "regions": len(regions),
                "leiden_seconds": leiden_seconds,
            }
        ]
    )
    return {
        "physical_partition_vertices": built["vertices"],
        "physical_partition_edges": built["edges"],
        "node_partition": partition,
        "regions": regions,
        "partition_metadata": metadata,
        "leiden_seconds": leiden_seconds,
    }


def partC_rev1_assign_physical_arc_regions(
    physical_arcs: pd.DataFrame,
    nodes: pd.DataFrame,
    node_partition: pd.DataFrame,
) -> pd.DataFrame:
    """Classify every physical directed arc by backbone or physical-node region label."""

    backbone_nodes = _partC_rev1_backbone_nodes(nodes)
    region_by_node = node_partition.set_index("partC_rev1_node_uid")["region_id"].astype(str).to_dict() if not node_partition.empty else {}
    rows: list[dict[str, Any]] = []
    for row in physical_arcs.itertuples(index=False):
        tail_raw = getattr(row, "tail_node_uid", getattr(row, "from_node_uid", None))
        head_raw = getattr(row, "head_node_uid", getattr(row, "to_node_uid", None))
        tail = partC_rev1_stable_id(tail_raw)
        head = partC_rev1_stable_id(head_raw)
        tail_backbone = tail in backbone_nodes
        head_backbone = head in backbone_nodes
        tail_region = region_by_node.get(tail)
        head_region = region_by_node.get(head)
        final_value = getattr(row, "is_final_backbone", False)
        is_final_backbone = False if pd.isna(final_value) else bool(final_value)
        if is_final_backbone:
            arc_type = "backbone_network"
            region_id = "backbone"
            neighbor_region_id = pd.NA
            backbone_node_uid = f"{tail}|{head}"
        elif not tail_backbone and not head_backbone and tail_region == head_region and tail_region is not None:
            arc_type = "regional_internal"
            region_id = tail_region
            neighbor_region_id = pd.NA
            backbone_node_uid = pd.NA
        elif not tail_backbone and not head_backbone:
            arc_type = "region_region_boundary_connector"
            region_id = tail_region
            neighbor_region_id = head_region
            backbone_node_uid = pd.NA
        elif tail_backbone != head_backbone:
            arc_type = "backbone_region_boundary_connector"
            region_id = head_region if tail_backbone else tail_region
            neighbor_region_id = pd.NA
            backbone_node_uid = tail if tail_backbone else head
        else:
            arc_type = "backbone_network"
            region_id = "backbone"
            neighbor_region_id = pd.NA
            backbone_node_uid = f"{tail}|{head}"
        rows.append(
            {
                "directed_arc_id": getattr(row, "directed_arc_id", pd.NA),
                "partC_rev1_link_uid": getattr(row, "partC_rev1_link_uid", pd.NA),
                "tail_node_uid": tail,
                "head_node_uid": head,
                "arc_partition_type": arc_type,
                "region_id": region_id,
                "neighbor_region_id": neighbor_region_id,
                "backbone_node_uid": backbone_node_uid,
            }
        )
    return pd.DataFrame(rows)
