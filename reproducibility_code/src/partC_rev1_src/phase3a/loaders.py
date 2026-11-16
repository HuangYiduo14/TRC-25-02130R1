"""Validated loaders for the accepted graph, OD snapshot, and s03 suite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from partC_rev1_src.partC_rev1_graph_snapshot import (
    partC_rev1_load_graph_snapshot,
    partC_rev1_open_sqlite_readonly,
)
from partC_rev1_src.partC_rev1_od_snapshot import partC_rev1_load_od_snapshot

from .authority import EXPECTED_GRAPH_IDENTITY, EXPECTED_OD_IDENTITY
from .geodesy import validate_lon_lat
from .models import AuthorityPaths, GraphData, ODData, SuiteData
from .units import MILES_TO_METERS, MINUTES_TO_SECONDS, validate_fftt_length


class Phase3ALoaderError(RuntimeError):
    pass


def load_graph(paths: AuthorityPaths) -> GraphData:
    snapshot = partC_rev1_load_graph_snapshot(paths.graph_npz, paths.graph_metadata_json)
    if snapshot.graph_hash != EXPECTED_GRAPH_IDENTITY:
        raise Phase3ALoaderError("Loaded graph snapshot identity does not match authority")
    arc_columns = [
        "arc_idx",
        "directed_arc_id",
        "tail_idx",
        "head_idx",
        "tail_node_uid",
        "head_node_uid",
        "free_time",
        "base_capacity",
        "remaining_capacity",
        "length_miles",
    ]
    arcs = pd.read_parquet(paths.graph_arcs_parquet, columns=arc_columns).sort_values("arc_idx", kind="mergesort")
    if len(arcs) != len(snapshot.arc_idx_to_directed_arc_id):
        raise Phase3ALoaderError("Graph arc Parquet row count differs from graph snapshot")
    if not np.array_equal(arcs["arc_idx"].to_numpy(dtype=np.int64), np.arange(len(arcs), dtype=np.int64)):
        raise Phase3ALoaderError("Graph arc_idx is not an exact dense vector order")
    arc_ids = arcs["directed_arc_id"].astype(str).to_numpy(dtype=str)
    if not np.array_equal(arc_ids, snapshot.arc_idx_to_directed_arc_id.astype(str)):
        raise Phase3ALoaderError("Graph arc stable-ID order differs between Parquet and NPZ")
    node_uid = snapshot.node_idx_to_uid.astype(str)
    node_attributes = _load_node_attributes(paths.rich_sqlite, node_uid)
    arc_fftt_s = snapshot.free_time.astype(np.float64) * MINUTES_TO_SECONDS
    arc_length_m = snapshot.length.astype(np.float64) * MILES_TO_METERS
    sparse = csr_matrix(
        (np.ones(len(snapshot.tail), dtype=np.uint8), (snapshot.tail, snapshot.head)),
        shape=(len(node_uid), len(node_uid)),
    )
    weak_count, weak = connected_components(sparse, directed=True, connection="weak", return_labels=True)
    strong_count, strong = connected_components(sparse, directed=True, connection="strong", return_labels=True)
    unit_validation = validate_fftt_length(arc_fftt_s, arc_length_m)
    coordinate_validation = validate_lon_lat(node_attributes["lon"], node_attributes["lat"])
    if coordinate_validation["nonfinite_count"] or coordinate_validation["out_of_range_count"]:
        raise Phase3ALoaderError(f"Authoritative coordinates are invalid: {coordinate_validation}")
    if unit_validation["nonfinite_fftt_count"] or unit_validation["nonpositive_fftt_count"]:
        raise Phase3ALoaderError(f"Authoritative FFTT contains invalid values: {unit_validation}")
    return GraphData(
        snapshot=snapshot,
        node_uid=node_uid,
        arc_id=arc_ids,
        tail=snapshot.tail.astype(np.int64),
        head=snapshot.head.astype(np.int64),
        arc_fftt_s=arc_fftt_s,
        arc_length_m=arc_length_m,
        base_capacity=arcs["base_capacity"].to_numpy(dtype=np.float64),
        remaining_capacity=arcs["remaining_capacity"].to_numpy(dtype=np.float64),
        lon=node_attributes["lon"],
        lat=node_attributes["lat"],
        county_fips=node_attributes["county_fips"],
        state_fips=node_attributes["state_fips"],
        physical_weak_component=weak.astype(np.int32),
        physical_strong_component=strong.astype(np.int32),
        validation={
            "graph_identity": snapshot.graph_hash,
            "node_count": len(node_uid),
            "arc_count": len(arc_ids),
            "weak_component_count": int(weak_count),
            "strong_component_count": int(strong_count),
            "coordinate": coordinate_validation,
            "fftt_length": unit_validation,
            "source_fftt_unit": "minutes",
            "prepared_fftt_unit": "seconds",
            "source_length_unit": "miles",
            "prepared_length_unit": "meters",
            "crs": "EPSG:4269 / NAD83 longitude-latitude",
            "axis_order": "longitude,latitude",
        },
    )


def _load_node_attributes(sqlite_path: Path, node_uid: np.ndarray) -> dict[str, np.ndarray]:
    query = (
        "SELECT partC_rev1_node_uid, partC_rev1_lon, partC_rev1_lat, "
        "partC_rev1_county_fips FROM partC_rev1_clean_physical_nodes"
    )
    with partC_rev1_open_sqlite_readonly(sqlite_path) as connection:
        frame = pd.read_sql_query(query, connection)
    numeric_uid = pd.to_numeric(frame["partC_rev1_node_uid"], errors="coerce")
    if numeric_uid.isna().any():
        raise Phase3ALoaderError("Node coordinate table contains a missing/non-numeric stable node ID")
    frame["node_uid"] = numeric_uid.astype(np.int64).astype(str)
    if frame["node_uid"].duplicated().any():
        raise Phase3ALoaderError("Node coordinate table contains duplicate stable node IDs")
    frame["county_fips"] = frame["partC_rev1_county_fips"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    frame.loc[frame["partC_rev1_county_fips"].isna(), "county_fips"] = ""
    frame["state_fips"] = frame["county_fips"].str.slice(0, 2)
    order = pd.DataFrame({"node_uid": node_uid}).merge(
        frame[["node_uid", "partC_rev1_lon", "partC_rev1_lat", "county_fips", "state_fips"]],
        on="node_uid",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if order[["partC_rev1_lon", "partC_rev1_lat"]].isna().any().any():
        missing = int(order["partC_rev1_lon"].isna().sum())
        raise Phase3ALoaderError(f"{missing} graph nodes lack authoritative coordinates")
    return {
        "lon": order["partC_rev1_lon"].to_numpy(dtype=np.float64),
        "lat": order["partC_rev1_lat"].to_numpy(dtype=np.float64),
        "county_fips": order["county_fips"].fillna("").to_numpy(dtype=str),
        "state_fips": order["state_fips"].fillna("").to_numpy(dtype=str),
    }


def load_suite(paths: AuthorityPaths, graph: GraphData) -> SuiteData:
    root = paths.suite_root
    identity = json.loads((root / "suite_identity.json").read_text(encoding="utf-8"))
    expected_suite_id = str(root.name)
    if identity.get("suite_id") != expected_suite_id:
        raise Phase3ALoaderError(
            f"Resolved suite identity differs from its accepted directory: {identity.get('suite_id')!r} != {expected_suite_id!r}"
        )
    arcs = pd.read_parquet(root / "directed_arc_membership.parquet").sort_values("arc_idx", kind="mergesort").reset_index(drop=True)
    if len(arcs) != graph.arc_count:
        raise Phase3ALoaderError("Accepted suite directed membership does not cover the graph")
    if not np.array_equal(arcs["arc_idx"].to_numpy(dtype=np.int64), np.arange(graph.arc_count, dtype=np.int64)):
        raise Phase3ALoaderError("Accepted suite directed membership arc_idx order is invalid")
    if not np.array_equal(arcs["directed_arc_id"].astype(str).to_numpy(dtype=str), graph.arc_id):
        raise Phase3ALoaderError("Accepted suite directed membership stable arc IDs do not match graph order")
    final_arc = arcs["is_final_backbone"].to_numpy(dtype=bool)
    final_node = np.zeros(graph.node_count, dtype=bool)
    final_node[graph.tail[final_arc]] = True
    final_node[graph.head[final_arc]] = True
    portals = pd.read_parquet(root / "portals.parquet").sort_values(
        ["region_uuid", "partC_rev1_node_uid"], kind="mergesort"
    ).reset_index(drop=True)
    centroid_roles = pd.read_parquet(root / "centroid_roles.parquet").sort_values(
        ["region_uuid", "centroid_role", "partC_rev1_node_uid"], kind="mergesort"
    ).reset_index(drop=True)
    terminals = pd.read_parquet(root / "terminals.parquet").sort_values(
        ["region_uuid", "terminal_order"], kind="mergesort"
    ).reset_index(drop=True)
    incident = pd.read_parquet(root / "node_incident_registry.parquet").sort_values("node_idx", kind="mergesort").reset_index(drop=True)
    regions = pd.read_parquet(root / "retained_regions.parquet").sort_values("region_uuid", kind="mergesort").reset_index(drop=True)
    arc_region = arcs["suite_label"].astype(str).to_numpy(dtype=str)
    region_ids = tuple(sorted(set(arc_region.tolist()).difference({"backbone"})))
    if len(region_ids) == 0 or final_arc.sum() == 0:
        raise Phase3ALoaderError("Accepted suite load produced a no-op graph decomposition")
    return SuiteData(
        suite_id=expected_suite_id,
        suite_root=root,
        suite_identity=identity,
        directed_arc_membership=arcs,
        portals=portals,
        centroid_roles=centroid_roles,
        terminals=terminals,
        node_incident_registry=incident,
        retained_regions=regions,
        is_final_backbone_arc=final_arc,
        is_final_backbone_node=final_node,
        arc_region_id=arc_region,
        region_ids=region_ids,
    )


def load_od(paths: AuthorityPaths) -> ODData:
    snapshot = partC_rev1_load_od_snapshot(paths.od_npz, paths.od_metadata_json)
    if snapshot.od_hash != EXPECTED_OD_IDENTITY:
        raise Phase3ALoaderError("Loaded OD snapshot identity differs from authority")
    origin = snapshot.origin_node_idx
    destination = snapshot.destination_node_idx
    demand = snapshot.demand
    if not (origin.shape == destination.shape == demand.shape) or origin.ndim != 1:
        raise Phase3ALoaderError("OD arrays have inconsistent shapes")
    if demand.size == 0 or np.any(~np.isfinite(demand)) or np.any(demand <= 0.0):
        raise Phase3ALoaderError("OD snapshot contains empty, nonfinite, or nonpositive demand")
    if np.any(origin[1:] < origin[:-1]):
        raise Phase3ALoaderError("OD snapshot is not ordered by origin index")
    same_origin = origin[1:] == origin[:-1]
    if np.any(destination[1:][same_origin] < destination[:-1][same_origin]):
        raise Phase3ALoaderError("OD snapshot is not ordered by destination within origin")
    total = float(demand.sum(dtype=np.float64))
    declared = float(snapshot.metadata.get("total_demand", total))
    if not np.isclose(total, declared, atol=1e-9, rtol=1e-12):
        raise Phase3ALoaderError(f"OD demand conservation failed: observed={total}, declared={declared}")
    return ODData(
        snapshot=snapshot,
        validation={
            "od_identity": snapshot.od_hash,
            "row_count": int(demand.size),
            "origin_count": int(snapshot.origin_group_node_idx.size),
            "total_demand": total,
            "duplicate_and_self_rows_preserved": True,
            "demand_unit": "2017_truck_annual_thousand_tons",
        },
    )
