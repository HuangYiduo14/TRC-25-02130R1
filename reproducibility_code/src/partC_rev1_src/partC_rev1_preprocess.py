"""Prompt 2 preprocessing orchestration for partC_rev1."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import pandas as pd

from .partC_rev1_cache import (
    partC_rev1_connect_sqlite,
    partC_rev1_write_metadata,
    partC_rev1_write_table_sqlite,
)
from .partC_rev1_centroids import partC_rev1_extract_county_centroids
from .partC_rev1_clean_geodatabase import partC_rev1_write_clean_geodatabase
from .partC_rev1_cleaning import (
    partC_rev1_clean_exact_duplicate_links,
    partC_rev1_clean_exact_duplicate_nodes,
)
from .partC_rev1_config import load_partC_rev1_config
from .partC_rev1_ground_truth import partC_rev1_match_ground_truth_to_clean_links, partC_rev1_read_ground_truth_flow
from .partC_rev1_io import partC_rev1_read_table, partC_rev1_write_table
from .partC_rev1_logging import partC_rev1_setup_logging
from .partC_rev1_od import (
    partC_rev1_expand_county_od_to_centroids,
    partC_rev1_expand_faf_to_county_od,
    partC_rev1_read_faf_od,
)
from .partC_rev1_paths import partC_rev1_output_file, partC_rev1_resolve_path
from .partC_rev1_physical_graph import partC_rev1_build_directed_arcs, partC_rev1_read_network_layers
from .partC_rev1_topology_snap import partC_rev1_resolve_link_endpoints


_PARTC_REV1_LOG = logging.getLogger("partC_rev1")


def _partC_rev1_cache_path(config: dict[str, Any]) -> Path:
    return partC_rev1_resolve_path(str(config.get("preprocess", {}).get("cache_sqlite", "partC_rev1_cache/partC_rev1_network.sqlite")))


def _partC_rev1_timed(
    runtimes: dict[str, float],
    name: str,
    fn: Callable[[], Any],
) -> Any:
    _PARTC_REV1_LOG.info("Starting stage: %s", name)
    start = time.perf_counter()
    try:
        result = fn()
        return result
    finally:
        runtimes[name] = time.perf_counter() - start
        _PARTC_REV1_LOG.info("Finished stage: %s in %.3f seconds", name, runtimes[name])


def _partC_rev1_raw_layer_summary(link_gdf: gpd.GeoDataFrame, node_gdf: gpd.GeoDataFrame, config: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layer_role": "links",
                "configured_layer": config.get("inputs", {}).get("links_layer", ""),
                "row_count": len(link_gdf),
                "crs": str(link_gdf.crs) if link_gdf.crs is not None else "",
                "geometry_types": ",".join(map(str, sorted(link_gdf.geometry.geom_type.dropna().unique()))),
            },
            {
                "layer_role": "nodes",
                "configured_layer": config.get("inputs", {}).get("nodes_layer", ""),
                "row_count": len(node_gdf),
                "crs": str(node_gdf.crs) if node_gdf.crs is not None else "",
                "geometry_types": ",".join(map(str, sorted(node_gdf.geometry.geom_type.dropna().unique()))),
            },
        ]
    )


def _partC_rev1_empty_gt() -> pd.DataFrame:
    return pd.DataFrame(columns=["ID", "flow", "flow_unit", "source_file", "source_column"])


def _partC_rev1_summary_value(report: pd.DataFrame, report_type: str) -> int:
    if report.empty or "report_type" not in report.columns:
        return 0
    return int((report["report_type"] == report_type).sum())


def _partC_rev1_write_sqlite_outputs(
    config: dict[str, Any],
    *,
    raw_summary: pd.DataFrame,
    clean_nodes: gpd.GeoDataFrame,
    clean_links: gpd.GeoDataFrame,
    directed_arcs: gpd.GeoDataFrame,
    centroids: pd.DataFrame,
    od_raw: pd.DataFrame,
    od_county: pd.DataFrame,
    od_centroid: pd.DataFrame,
    ground_truth: pd.DataFrame | None,
) -> None:
    cache_path = _partC_rev1_cache_path(config)
    _PARTC_REV1_LOG.info("Writing SQLite cache to %s", cache_path)
    with partC_rev1_connect_sqlite(cache_path) as conn:
        _PARTC_REV1_LOG.info("Writing SQLite table partC_rev1_raw_layer_summary rows=%s", len(raw_summary))
        partC_rev1_write_table_sqlite(conn, "partC_rev1_raw_layer_summary", raw_summary)
        _PARTC_REV1_LOG.info("Writing SQLite table partC_rev1_clean_physical_nodes rows=%s", len(clean_nodes))
        partC_rev1_write_table_sqlite(conn, "partC_rev1_clean_physical_nodes", clean_nodes)
        _PARTC_REV1_LOG.info("Writing SQLite table partC_rev1_clean_physical_links rows=%s", len(clean_links))
        partC_rev1_write_table_sqlite(conn, "partC_rev1_clean_physical_links", clean_links)
        _PARTC_REV1_LOG.info("Writing SQLite table partC_rev1_directed_arcs rows=%s", len(directed_arcs))
        partC_rev1_write_table_sqlite(conn, "partC_rev1_directed_arcs", directed_arcs)
        partC_rev1_write_table_sqlite(conn, "partC_rev1_county_centroids", centroids)
        partC_rev1_write_table_sqlite(conn, "partC_rev1_od_raw_2017_truck", od_raw)
        partC_rev1_write_table_sqlite(conn, "partC_rev1_od_county", od_county)
        partC_rev1_write_table_sqlite(conn, "partC_rev1_od_centroid", od_centroid)
        if ground_truth is not None:
            partC_rev1_write_table_sqlite(conn, "partC_rev1_ground_truth_link_flow", ground_truth)
        partC_rev1_write_metadata(conn, "partC_rev1_run_id", config.get("project", {}).get("run_id", ""))
        partC_rev1_write_metadata(conn, "partC_rev1_cache_path", str(cache_path))


def partC_rev1_run_preprocess(config_path: str) -> dict[str, Any]:
    config = load_partC_rev1_config(config_path)
    output_dir = config.get("outputs", {}).get("output_dir", config.get("project", {}).get("output_dir", "partC_rev1_outputs"))
    partC_rev1_setup_logging(output_dir)
    _PARTC_REV1_LOG.info("Starting partC_rev1 preprocessing with config %s", config_path)
    _PARTC_REV1_LOG.info("Run id: %s", config.get("project", {}).get("run_id", ""))
    runtimes: dict[str, float] = {}

    link_gdf, node_gdf = _partC_rev1_timed(runtimes, "read_network_layers", lambda: partC_rev1_read_network_layers(config))
    raw_summary = _partC_rev1_raw_layer_summary(link_gdf, node_gdf, config)

    clean_nodes, node_report = _partC_rev1_timed(
        runtimes,
        "clean_nodes",
        lambda: partC_rev1_clean_exact_duplicate_nodes(node_gdf, config),
    )
    clean_links, link_report = _partC_rev1_timed(
        runtimes,
        "clean_links",
        lambda: partC_rev1_clean_exact_duplicate_links(link_gdf, config),
    )
    clean_links = _partC_rev1_timed(
        runtimes,
        "resolve_topology",
        lambda: partC_rev1_resolve_link_endpoints(clean_links, clean_nodes, config),
    )
    directed_arcs = _partC_rev1_timed(
        runtimes,
        "build_directed_arcs",
        lambda: partC_rev1_build_directed_arcs(clean_links, config),
    )
    centroids = _partC_rev1_timed(runtimes, "extract_centroids", lambda: partC_rev1_extract_county_centroids(clean_nodes, config))

    od_raw = _partC_rev1_timed(runtimes, "read_faf_od", lambda: partC_rev1_read_faf_od(config))
    origin_factors = _partC_rev1_timed(
        runtimes,
        "read_origin_factors",
        lambda: partC_rev1_read_table(partC_rev1_resolve_path(str(config.get("inputs", {}).get("origin_factors_path", "")))),
    )
    destination_factors = _partC_rev1_timed(
        runtimes,
        "read_destination_factors",
        lambda: partC_rev1_read_table(partC_rev1_resolve_path(str(config.get("inputs", {}).get("destination_factors_path", "")))),
    )
    od_county = _partC_rev1_timed(
        runtimes,
        "expand_faf_to_county_od",
        lambda: partC_rev1_expand_faf_to_county_od(od_raw, origin_factors, destination_factors, config),
    )
    od_centroid = _partC_rev1_timed(
        runtimes,
        "expand_county_od_to_centroids",
        lambda: partC_rev1_expand_county_od_to_centroids(od_county, centroids, config),
    )

    ground_truth = _partC_rev1_timed(runtimes, "read_ground_truth", lambda: partC_rev1_read_ground_truth_flow(config))
    gt_for_cache = ground_truth if ground_truth is not None else _partC_rev1_empty_gt()
    clean_links = _partC_rev1_timed(
        runtimes,
        "match_ground_truth",
        lambda: partC_rev1_match_ground_truth_to_clean_links(clean_links, ground_truth, config),
    )
    clean_links = gpd.GeoDataFrame(clean_links, geometry="geometry", crs=link_gdf.crs)

    _partC_rev1_timed(
        runtimes,
        "write_sqlite",
        lambda: _partC_rev1_write_sqlite_outputs(
            config,
            raw_summary=raw_summary,
            clean_nodes=clean_nodes,
            clean_links=clean_links,
            directed_arcs=directed_arcs,
            centroids=centroids,
            od_raw=od_raw,
            od_county=od_county,
            od_centroid=od_centroid,
            ground_truth=gt_for_cache,
        ),
    )
    geodatabase_result = _partC_rev1_timed(
        runtimes,
        "write_clean_geodatabase",
        lambda: partC_rev1_write_clean_geodatabase(clean_nodes, clean_links, directed_arcs, config),
    )

    gt_match_report_path = partC_rev1_output_file(config, "partC_rev1_ground_truth_match_report.csv")
    gt_match_rate = 0.0
    if gt_match_report_path.exists():
        gt_match_report = pd.read_csv(gt_match_report_path)
        if not gt_match_report.empty and "match_rate" in gt_match_report.columns:
            gt_match_rate = float(gt_match_report["match_rate"].iloc[0])
    summary = pd.DataFrame(
        [
            {
                "raw link rows": len(link_gdf),
                "clean link rows": len(clean_links),
                "exact duplicate link rows dropped": _partC_rev1_summary_value(link_report, "link_exact_duplicate"),
                "suspicious same-geometry non-identical link groups": _partC_rev1_summary_value(link_report, "link_same_geometry_nonidentical"),
                "raw node rows": len(node_gdf),
                "clean node rows": len(clean_nodes),
                "exact duplicate node rows dropped": _partC_rev1_summary_value(node_report, "node_exact_duplicate"),
                "suspicious same-geometry non-identical node groups": _partC_rev1_summary_value(node_report, "node_same_geometry_nonidentical"),
                "directed arc rows": len(directed_arcs),
                "ignored links due to no lane direction info": int(
                    clean_links.get("partC_rev1_direction_case", pd.Series(dtype=str)).astype(str).eq("IGNORED_NO_LANE_INFO").sum()
                ),
                "defaulted no-lane bidirectional links": int(
                    clean_links.get("partC_rev1_missing_lane_defaulted", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
                ),
                "defaulted no-lane directed arcs added": int(
                    clean_links.get("partC_rev1_missing_lane_defaulted", pd.Series(dtype=bool)).fillna(False).astype(bool).sum() * 2
                ),
                "snap failure count": int(clean_links.get("partC_rev1_snap_failed", pd.Series(dtype=bool)).fillna(False).sum()),
                "centroid count": len(centroids),
                "county OD rows": len(od_county),
                "centroid OD rows": len(od_centroid),
                "total county demand annual ktons": float(od_county["demand_annual_ktons"].sum()) if "demand_annual_ktons" in od_county else 0.0,
                "total centroid demand annual ktons": float(od_centroid["demand_annual_ktons"].sum())
                if "demand_annual_ktons" in od_centroid
                else 0.0,
                "ground-truth rows": len(gt_for_cache),
                "ground-truth match rate": gt_match_rate,
                "cache path": str(_partC_rev1_cache_path(config)),
                "clean geodatabase path": geodatabase_result.get("clean_geodatabase_path"),
                "runtime by stage": "; ".join(f"{key}={value:.6f}" for key, value in sorted(runtimes.items())),
            }
        ]
    )
    partC_rev1_write_table(summary, partC_rev1_output_file(config, "partC_rev1_preprocess_summary.csv"), index=False)
    _PARTC_REV1_LOG.info("Preprocessing complete. Cache=%s GeoPackage=%s", _partC_rev1_cache_path(config), geodatabase_result.get("clean_geodatabase_path"))
    return {
        "summary": summary.iloc[0].to_dict(),
        "cache_path": str(_partC_rev1_cache_path(config)),
        "clean_geodatabase_path": geodatabase_result.get("clean_geodatabase_path"),
        "runtimes": runtimes,
    }
