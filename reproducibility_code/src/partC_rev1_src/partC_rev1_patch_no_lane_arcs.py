"""Patch existing real preprocessing outputs with default arcs for no-lane links."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from .partC_rev1_cache import partC_rev1_connect_sqlite, partC_rev1_write_metadata, partC_rev1_write_table_sqlite
from .partC_rev1_clean_geodatabase import partC_rev1_write_clean_geodatabase
from .partC_rev1_config import load_partC_rev1_config
from .partC_rev1_io import partC_rev1_write_table
from .partC_rev1_logging import partC_rev1_setup_logging
from .partC_rev1_paths import partC_rev1_output_file, partC_rev1_resolve_path
from .partC_rev1_physical_graph import (
    _partC_rev1_arc_frame,
    _partC_rev1_concat_and_sort_arcs,
    _partC_rev1_find_optional_column,
    _partC_rev1_order_arc_columns,
    _partC_rev1_schema_col,
)


_PARTC_REV1_LOG = logging.getLogger("partC_rev1")


def _partC_rev1_clean_gpkg_path(config: dict[str, Any]) -> Path:
    return partC_rev1_resolve_path(
        str(config.get("clean_geodatabase", {}).get("output_gpkg_path", "partC_rev1_cache/partC_rev1_clean_network.gpkg"))
    )


def _partC_rev1_cache_path(config: dict[str, Any]) -> Path:
    return partC_rev1_resolve_path(str(config.get("preprocess", {}).get("cache_sqlite", "partC_rev1_cache/partC_rev1_network.sqlite")))


def _partC_rev1_layer_names(config: dict[str, Any]) -> dict[str, str]:
    layers = config.get("clean_geodatabase", {}).get("layers", {})
    return {
        "clean_nodes": str(layers.get("clean_nodes", "partC_rev1_clean_physical_nodes")),
        "clean_links": str(layers.get("clean_links", "partC_rev1_clean_physical_links")),
        "directed_arcs": str(layers.get("directed_arcs", "partC_rev1_directed_arcs")),
    }


def _partC_rev1_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "1.0", "yes"})


def _partC_rev1_no_lane_mask(clean_links: gpd.GeoDataFrame, config: dict[str, Any]) -> pd.Series:
    direction_cfg = config.get("direction_inference", {})
    threshold = float(direction_cfg.get("positive_lane_threshold", 0.0))
    ab_lanes_col = _partC_rev1_find_optional_column(
        clean_links,
        str(direction_cfg.get("ab_lanes_col") or _partC_rev1_schema_col(config, "ab_lanes_col", "AB_Lanes")),
    )
    ba_lanes_col = _partC_rev1_find_optional_column(
        clean_links,
        str(direction_cfg.get("ba_lanes_col") or _partC_rev1_schema_col(config, "ba_lanes_col", "BA_Lanes")),
    )
    ab = pd.to_numeric(clean_links[ab_lanes_col], errors="coerce") if ab_lanes_col in clean_links.columns else pd.Series(np.nan, index=clean_links.index)
    ba = pd.to_numeric(clean_links[ba_lanes_col], errors="coerce") if ba_lanes_col in clean_links.columns else pd.Series(np.nan, index=clean_links.index)
    return ~(ab.fillna(0.0).gt(threshold) | ba.fillna(0.0).gt(threshold))


def _partC_rev1_build_default_arcs(
    default_links: gpd.GeoDataFrame,
    existing_arcs: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> gpd.GeoDataFrame:
    ab_time_col = _partC_rev1_find_optional_column(default_links, _partC_rev1_schema_col(config, "ab_time_col", "AB_FreeFlowTime"))
    ba_time_col = _partC_rev1_find_optional_column(default_links, _partC_rev1_schema_col(config, "ba_time_col", "BA_FreeFlowTime"))
    ab_speed_col = _partC_rev1_find_optional_column(default_links, _partC_rev1_schema_col(config, "ab_speed_col", "AB_FinalSpeed"))
    ba_speed_col = _partC_rev1_find_optional_column(default_links, _partC_rev1_schema_col(config, "ba_speed_col", "BA_FinalSpeed"))

    frames = [
        _partC_rev1_arc_frame(default_links, config, direction="AB", lane_col=None, time_col=ab_time_col, speed_col=ab_speed_col),
        _partC_rev1_arc_frame(default_links, config, direction="BA", lane_col=None, time_col=ba_time_col, speed_col=ba_speed_col),
    ]
    default_arcs = _partC_rev1_concat_and_sort_arcs(frames, default_links.crs)
    start_id = int(pd.to_numeric(existing_arcs["directed_arc_id"], errors="coerce").max()) + 1 if len(existing_arcs) else 0
    default_arcs["directed_arc_id"] = range(start_id, start_id + len(default_arcs))
    default_arcs["partC_rev1_missing_lane_defaulted"] = True
    default_arcs["partC_rev1_default_lane_count"] = float(config.get("preprocess", {}).get("default_lanes", 1.0))
    return _partC_rev1_order_arc_columns(default_arcs, default_links.crs)


def _partC_rev1_update_direction_reports(
    clean_links: gpd.GeoDataFrame,
    all_arcs: gpd.GeoDataFrame,
    candidate_mask: pd.Series,
    config: dict[str, Any],
) -> None:
    ignored = clean_links.loc[clean_links["partC_rev1_direction_case"].astype(str).eq("IGNORED_NO_LANE_INFO")].copy()
    defaulted = clean_links.loc[_partC_rev1_bool_series(clean_links["partC_rev1_missing_lane_defaulted"])].copy()
    summary = (
        clean_links.groupby("partC_rev1_direction_case", dropna=False)
        .size()
        .reset_index(name="link_count")
        .sort_values("partC_rev1_direction_case")
    )
    summary["directed_arc_rows"] = len(all_arcs)
    partC_rev1_write_table(summary, partC_rev1_output_file(config, "partC_rev1_direction_inference_summary.csv"), index=False)
    partC_rev1_write_table(ignored, partC_rev1_output_file(config, "partC_rev1_ignored_links_no_lane_info.csv"), index=False)
    partC_rev1_write_table(ignored, partC_rev1_output_file(config, "partC_rev1_links_excluded_no_lane_direction.csv"), index=False)
    partC_rev1_write_table(defaulted, partC_rev1_output_file(config, "partC_rev1_links_defaulted_no_lane_direction.csv"), index=False)
    patch_report = pd.DataFrame(
        [
            {
                "candidate_links_patched": int(candidate_mask.sum()),
                "default_arcs_added": int(candidate_mask.sum() * 2),
                "remaining_ignored_links": len(ignored),
                "total_directed_arcs": len(all_arcs),
            }
        ]
    )
    partC_rev1_write_table(patch_report, partC_rev1_output_file(config, "partC_rev1_no_lane_arc_patch_report.csv"), index=False)


def _partC_rev1_update_preprocess_summary(
    all_arcs: gpd.GeoDataFrame,
    remaining_ignored: int,
    defaulted_links: int,
    config: dict[str, Any],
) -> None:
    path = partC_rev1_output_file(config, "partC_rev1_preprocess_summary.csv")
    if not path.exists():
        return
    summary = pd.read_csv(path)
    if summary.empty:
        return
    summary.loc[0, "directed arc rows"] = len(all_arcs)
    summary.loc[0, "ignored links due to no lane direction info"] = remaining_ignored
    summary.loc[0, "defaulted no-lane bidirectional links"] = defaulted_links
    summary.loc[0, "defaulted no-lane directed arcs added"] = defaulted_links * 2
    partC_rev1_write_table(summary, path, index=False)


def partC_rev1_patch_no_lane_arcs(config_path: str) -> dict[str, Any]:
    config = load_partC_rev1_config(config_path)
    output_dir = config.get("outputs", {}).get("output_dir", config.get("project", {}).get("output_dir", "partC_rev1_outputs"))
    partC_rev1_setup_logging(output_dir)

    gpkg_path = _partC_rev1_clean_gpkg_path(config)
    cache_path = _partC_rev1_cache_path(config)
    layers = _partC_rev1_layer_names(config)
    default_case = str(config.get("direction_inference", {}).get("missing_lane_direction_case", "DEFAULT_AB_BA_1_LANE"))
    default_lane_count = float(config.get("preprocess", {}).get("default_lanes", 1.0))

    _PARTC_REV1_LOG.info("Starting no-lane arc patch with config %s", config_path)
    _PARTC_REV1_LOG.info("Reading GeoPackage network layers from %s", gpkg_path)
    clean_nodes = gpd.read_file(gpkg_path, layer=layers["clean_nodes"])
    clean_links = gpd.read_file(gpkg_path, layer=layers["clean_links"])
    existing_arcs = gpd.read_file(gpkg_path, layer=layers["directed_arcs"])

    if "partC_rev1_missing_lane_defaulted" not in clean_links.columns:
        clean_links["partC_rev1_missing_lane_defaulted"] = False
    if "partC_rev1_default_lane_count" not in clean_links.columns:
        clean_links["partC_rev1_default_lane_count"] = np.nan

    valid_topology = ~_partC_rev1_bool_series(clean_links.get("partC_rev1_snap_failed", pd.Series(False, index=clean_links.index)))
    no_lane = _partC_rev1_no_lane_mask(clean_links, config)
    existing_link_uids = set(pd.to_numeric(existing_arcs["partC_rev1_link_uid"], errors="coerce").dropna().astype("int64").tolist())
    no_existing_arc = ~pd.to_numeric(clean_links["partC_rev1_link_uid"], errors="coerce").astype("int64").isin(existing_link_uids)
    candidate_mask = valid_topology & no_lane & no_existing_arc
    default_links = clean_links.loc[candidate_mask].copy().sort_values("partC_rev1_link_uid", kind="mergesort")

    _PARTC_REV1_LOG.info(
        "No-lane patch candidates=%s existing_arcs=%s current_max_arc_id=%s",
        len(default_links),
        len(existing_arcs),
        existing_arcs["directed_arc_id"].max() if len(existing_arcs) else None,
    )
    if default_links.empty:
        result = {
            "candidate_links_patched": 0,
            "default_arcs_added": 0,
            "total_directed_arcs": len(existing_arcs),
            "message": "No unpatched no-lane links found; outputs left unchanged.",
        }
        _PARTC_REV1_LOG.info(json.dumps(result, sort_keys=True))
        return result

    clean_links.loc[candidate_mask, "partC_rev1_direction_case"] = default_case
    clean_links.loc[candidate_mask, "partC_rev1_missing_lane_defaulted"] = True
    clean_links.loc[candidate_mask, "partC_rev1_default_lane_count"] = default_lane_count

    new_arcs = _partC_rev1_build_default_arcs(default_links, existing_arcs, config)
    existing_arcs = _partC_rev1_order_arc_columns(existing_arcs, existing_arcs.crs)
    all_arcs = gpd.GeoDataFrame(pd.concat([existing_arcs, new_arcs], ignore_index=True), geometry="geometry", crs=existing_arcs.crs)
    all_arcs = _partC_rev1_order_arc_columns(all_arcs, existing_arcs.crs)

    expected_ids = list(range(len(all_arcs)))
    actual_ids = pd.to_numeric(all_arcs["directed_arc_id"], errors="raise").astype(int).tolist()
    if actual_ids != expected_ids:
        raise ValueError("Patched directed_arc_id sequence is not contiguous from zero")
    duplicate_pairs = all_arcs.duplicated(subset=["partC_rev1_link_uid", "direction"]).sum()
    if duplicate_pairs:
        raise ValueError(f"Patch would create duplicate link/direction arcs: {duplicate_pairs}")

    _PARTC_REV1_LOG.info("Updating SQLite cache tables clean_links and directed_arcs at %s", cache_path)
    with partC_rev1_connect_sqlite(cache_path) as conn:
        partC_rev1_write_table_sqlite(conn, "partC_rev1_clean_physical_links", clean_links)
        partC_rev1_write_table_sqlite(conn, "partC_rev1_directed_arcs", all_arcs)
        partC_rev1_write_metadata(conn, "partC_rev1_no_lane_arc_patch", {"default_links": len(default_links), "default_arcs": len(new_arcs)})

    _PARTC_REV1_LOG.info("Writing patched GeoPackage through temp file before replacing original")
    temp_config = copy.deepcopy(config)
    temp_gpkg = gpkg_path.with_name("partC_rev1_clean_network_no_lane_patch_tmp.gpkg")
    temp_config.setdefault("clean_geodatabase", {})["output_gpkg_path"] = str(temp_gpkg)
    partC_rev1_write_clean_geodatabase(clean_nodes, clean_links, all_arcs, temp_config)
    os.replace(temp_gpkg, gpkg_path)

    remaining_ignored = int(clean_links["partC_rev1_direction_case"].astype(str).eq("IGNORED_NO_LANE_INFO").sum())
    defaulted_links = int(_partC_rev1_bool_series(clean_links["partC_rev1_missing_lane_defaulted"]).sum())
    _partC_rev1_update_direction_reports(clean_links, all_arcs, candidate_mask, config)
    _partC_rev1_update_preprocess_summary(all_arcs, remaining_ignored, defaulted_links, config)

    result = {
        "candidate_links_patched": int(len(default_links)),
        "default_arcs_added": int(len(new_arcs)),
        "first_new_directed_arc_id": int(new_arcs["directed_arc_id"].min()),
        "last_new_directed_arc_id": int(new_arcs["directed_arc_id"].max()),
        "total_directed_arcs": int(len(all_arcs)),
        "remaining_ignored_links": remaining_ignored,
        "defaulted_links_total": defaulted_links,
        "cache_path": str(cache_path),
        "gpkg_path": str(gpkg_path),
    }
    _PARTC_REV1_LOG.info("No-lane arc patch complete: %s", json.dumps(result, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="partC_rev1_patch_no_lane_arcs")
    parser.add_argument("--config", default="partC_rev1_configs/partC_rev1_real_preprocess_config.json")
    args = parser.parse_args(argv)
    result = partC_rev1_patch_no_lane_arcs(args.config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
