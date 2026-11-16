"""Topology resolution by explicit columns or endpoint snapping."""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point

from .partC_rev1_io import partC_rev1_write_table
from .partC_rev1_paths import partC_rev1_output_file
from .partC_rev1_schema import partC_rev1_coerce_string_id


_PARTC_REV1_LOG = logging.getLogger("partC_rev1")


def _partC_rev1_first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        match = lower.get(str(candidate).lower())
        if match:
            return match
    return None


def _partC_rev1_line_component(geom: Any) -> LineString | None:
    if geom is None:
        return None
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        parts = list(geom.geoms)
        if not parts:
            return None
        return max(parts, key=lambda item: item.length)
    return None


def _partC_rev1_endpoint(geom: Any, which: str) -> Point | None:
    line = _partC_rev1_line_component(geom)
    if line is None or len(line.coords) == 0:
        return None
    coord = line.coords[0] if which == "A" else line.coords[-1]
    return Point(coord)


def _partC_rev1_ensure_crs(gdf: gpd.GeoDataFrame, config: dict[str, Any], label: str) -> gpd.GeoDataFrame:
    if gdf.crs is not None:
        return gdf
    override = config.get("preprocess", {}).get("crs_override") or config.get("inputs", {}).get("network_crs_override")
    if override:
        return gdf.set_crs(override)
    raise ValueError(f"{label} CRS is missing; set preprocess.crs_override before endpoint snapping")


def _partC_rev1_project_for_distance(
    link_gdf: gpd.GeoDataFrame,
    node_gdf: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    link_gdf = _partC_rev1_ensure_crs(link_gdf, config, "Link layer")
    node_gdf = _partC_rev1_ensure_crs(node_gdf, config, "Node layer")
    projected_crs = config.get("preprocess", {}).get("snap_projected_crs", "EPSG:5070")
    if link_gdf.crs.is_geographic:
        link_projected = link_gdf.to_crs(projected_crs)
    else:
        link_projected = link_gdf
    if node_gdf.crs.is_geographic:
        node_projected = node_gdf.to_crs(projected_crs)
    else:
        node_projected = node_gdf.to_crs(link_projected.crs) if node_gdf.crs != link_projected.crs else node_gdf
    return link_projected, node_projected


def _partC_rev1_try_explicit_topology(
    clean_link_gdf: gpd.GeoDataFrame,
    clean_node_gdf: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> gpd.GeoDataFrame | None:
    schema = config.get("input_schema", {})
    from_col = _partC_rev1_first_existing(
        list(clean_link_gdf.columns),
        list(schema.get("link_from_col_candidates", [])),
    )
    to_col = _partC_rev1_first_existing(
        list(clean_link_gdf.columns),
        list(schema.get("link_to_col_candidates", [])),
    )
    if from_col is None or to_col is None:
        return None

    node_lookup = clean_node_gdf[["partC_rev1_original_node_id", "partC_rev1_node_uid"]].copy()
    node_lookup["partC_rev1_original_node_id"] = node_lookup["partC_rev1_original_node_id"].astype("string")
    out = clean_link_gdf.copy()
    out["_partC_rev1_A_lookup"] = partC_rev1_coerce_string_id(out[from_col])
    out["_partC_rev1_B_lookup"] = partC_rev1_coerce_string_id(out[to_col])
    out = out.merge(
        node_lookup.rename(columns={"partC_rev1_original_node_id": "_partC_rev1_A_lookup", "partC_rev1_node_uid": "partC_rev1_A_node_uid"}),
        on="_partC_rev1_A_lookup",
        how="left",
    )
    out = out.merge(
        node_lookup.rename(columns={"partC_rev1_original_node_id": "_partC_rev1_B_lookup", "partC_rev1_node_uid": "partC_rev1_B_node_uid"}),
        on="_partC_rev1_B_lookup",
        how="left",
    )
    if out["partC_rev1_A_node_uid"].isna().any() or out["partC_rev1_B_node_uid"].isna().any():
        return None
    out["partC_rev1_A_original_node_id"] = out["_partC_rev1_A_lookup"]
    out["partC_rev1_B_original_node_id"] = out["_partC_rev1_B_lookup"]
    out["partC_rev1_snap_distance_A_m"] = 0.0
    out["partC_rev1_snap_distance_B_m"] = 0.0
    out["partC_rev1_topology_source"] = "explicit_columns"
    out["partC_rev1_snap_failed"] = False
    out["partC_rev1_snap_failure_reason"] = ""
    out["partC_rev1_topology_status"] = "snapped"
    return gpd.GeoDataFrame(out.drop(columns=["_partC_rev1_A_lookup", "_partC_rev1_B_lookup"]), geometry="geometry", crs=clean_link_gdf.crs)


def _partC_rev1_query_endpoints(
    endpoint_points: list[Point | None],
    tree: cKDTree,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.array(
        [[point.x, point.y] if point is not None else [np.nan, np.nan] for point in endpoint_points],
        dtype=float,
    )
    valid = np.isfinite(coords).all(axis=1)
    distances = np.full(len(coords), np.inf)
    indices = np.full(len(coords), -1, dtype=int)
    if valid.any():
        queried_distances, queried_indices = tree.query(coords[valid], k=1)
        distances[valid] = queried_distances
        indices[valid] = queried_indices
    return distances, indices


def partC_rev1_resolve_link_endpoints(
    clean_link_gdf: gpd.GeoDataFrame,
    clean_node_gdf: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> gpd.GeoDataFrame:
    _PARTC_REV1_LOG.info("Resolving link endpoints for %s links and %s nodes", len(clean_link_gdf), len(clean_node_gdf))
    explicit = _partC_rev1_try_explicit_topology(clean_link_gdf, clean_node_gdf, config)
    if explicit is not None:
        _PARTC_REV1_LOG.info("Resolved topology using explicit columns")
        failures = explicit.loc[explicit["partC_rev1_snap_failed"]]
        summary = pd.DataFrame([{"topology_source": "explicit_columns", "link_rows": len(explicit), "snap_failures": len(failures)}])
        partC_rev1_write_table(summary, partC_rev1_output_file(config, "partC_rev1_topology_snap_summary.csv"), index=False)
        partC_rev1_write_table(failures, partC_rev1_output_file(config, "partC_rev1_topology_snap_failures.csv"), index=False)
        return explicit

    projected_links, projected_nodes = _partC_rev1_project_for_distance(clean_link_gdf, clean_node_gdf, config)
    _PARTC_REV1_LOG.info("Endpoint snapping projected CRS: links=%s nodes=%s", projected_links.crs, projected_nodes.crs)
    node_points = projected_nodes.geometry
    if not all(isinstance(geom, Point) for geom in node_points.dropna()):
        raise ValueError("Clean node geometry must be Point for endpoint snapping")
    node_coords = np.array([[geom.x, geom.y] for geom in node_points], dtype=float)
    tree = cKDTree(node_coords)
    _PARTC_REV1_LOG.info("KDTree built for %s cleaned node points; querying endpoints in batch", len(node_coords))

    endpoint_a = [_partC_rev1_endpoint(geom, "A") for geom in projected_links.geometry]
    endpoint_b = [_partC_rev1_endpoint(geom, "B") for geom in projected_links.geometry]
    dist_a, idx_a = _partC_rev1_query_endpoints(endpoint_a, tree)
    dist_b, idx_b = _partC_rev1_query_endpoints(endpoint_b, tree)

    out = clean_link_gdf.copy()
    node_uids = clean_node_gdf["partC_rev1_node_uid"].to_numpy()
    node_ids = clean_node_gdf["partC_rev1_original_node_id"].astype("string").to_numpy()
    out["partC_rev1_A_node_uid"] = np.where(idx_a >= 0, node_uids[idx_a.clip(min=0)], np.nan)
    out["partC_rev1_B_node_uid"] = np.where(idx_b >= 0, node_uids[idx_b.clip(min=0)], np.nan)
    out["partC_rev1_A_original_node_id"] = np.where(idx_a >= 0, node_ids[idx_a.clip(min=0)], None)
    out["partC_rev1_B_original_node_id"] = np.where(idx_b >= 0, node_ids[idx_b.clip(min=0)], None)
    out["partC_rev1_snap_distance_A_m"] = dist_a
    out["partC_rev1_snap_distance_B_m"] = dist_b
    tolerance = float(config.get("preprocess", {}).get("snap_tolerance_meters", 100.0))
    out["partC_rev1_topology_source"] = "geometry_snap"
    out["partC_rev1_snap_failed"] = (
        ~np.isfinite(dist_a)
        | ~np.isfinite(dist_b)
        | (dist_a > tolerance)
        | (dist_b > tolerance)
        | pd.isna(out["partC_rev1_A_node_uid"])
        | pd.isna(out["partC_rev1_B_node_uid"])
    )
    out["partC_rev1_snap_failure_reason"] = np.where(
        out["partC_rev1_snap_failed"],
        "endpoint_distance_exceeds_tolerance_or_invalid_geometry",
        "",
    )
    out["partC_rev1_topology_status"] = np.where(out["partC_rev1_snap_failed"], "snap_failed", "snapped")

    failures = out.loc[out["partC_rev1_snap_failed"]].copy()
    summary = pd.DataFrame(
        [
            {
                "topology_source": "geometry_snap",
                "link_rows": len(out),
                "snap_failures": int(out["partC_rev1_snap_failed"].sum()),
                "snap_tolerance_meters": tolerance,
                "max_snap_distance_A_m": float(np.nanmax(np.where(np.isfinite(dist_a), dist_a, np.nan))) if len(dist_a) else np.nan,
                "max_snap_distance_B_m": float(np.nanmax(np.where(np.isfinite(dist_b), dist_b, np.nan))) if len(dist_b) else np.nan,
            }
        ]
    )
    partC_rev1_write_table(summary, partC_rev1_output_file(config, "partC_rev1_topology_snap_summary.csv"), index=False)
    partC_rev1_write_table(failures, partC_rev1_output_file(config, "partC_rev1_topology_snap_failures.csv"), index=False)
    _PARTC_REV1_LOG.info(
        "Endpoint snapping complete: failures=%s tolerance_m=%s",
        int(out["partC_rev1_snap_failed"].sum()),
        tolerance,
    )
    return gpd.GeoDataFrame(out, geometry="geometry", crs=clean_link_gdf.crs)
