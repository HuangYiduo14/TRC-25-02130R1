"""Physical network loading and lane-based directed arc construction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import reverse as shapely_reverse

from .partC_rev1_gdb import partC_rev1_guess_link_node_layers
from .partC_rev1_io import partC_rev1_write_table
from .partC_rev1_paths import partC_rev1_output_file, partC_rev1_resolve_path
from .partC_rev1_schema import partC_rev1_find_column


_PARTC_REV1_LOG = logging.getLogger("partC_rev1")


def _partC_rev1_schema_col(config: dict[str, Any], key: str, default: str) -> str:
    return str(config.get("input_schema", {}).get(key, default) or default)


def _partC_rev1_find_optional_column(df: pd.DataFrame, preferred: str, candidates: list[str] | None = None) -> str | None:
    try:
        return partC_rev1_find_column(df, preferred, candidates=candidates, required=False)
    except ValueError:
        return None


def partC_rev1_read_network_layers(config: dict[str, Any]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    inputs = config.get("inputs", {})
    gdb_path = partC_rev1_resolve_path(str(inputs.get("network_gdb_path", "")))
    if not gdb_path.exists():
        raise FileNotFoundError(f"Network geodatabase not found: {gdb_path}")

    links_layer = str(inputs.get("links_layer", "") or "")
    nodes_layer = str(inputs.get("nodes_layer", "") or "")
    if not links_layer or not nodes_layer:
        guesses = partC_rev1_guess_link_node_layers(str(gdb_path))
        if not links_layer:
            links_layer = guesses.get("links_layer") or ""
        if not nodes_layer:
            nodes_layer = guesses.get("nodes_layer") or ""
        if guesses.get("ambiguous_links") or guesses.get("ambiguous_nodes") or not links_layer or not nodes_layer:
            raise ValueError(
                "Could not uniquely determine network link/node layers; set inputs.links_layer and inputs.nodes_layer in config"
            )

    _PARTC_REV1_LOG.info("Reading network link layer %s from %s", links_layer, gdb_path)
    link_gdf = gpd.read_file(str(gdb_path), layer=links_layer)
    _PARTC_REV1_LOG.info("Read link layer rows=%s columns=%s crs=%s", len(link_gdf), len(link_gdf.columns), link_gdf.crs)
    _PARTC_REV1_LOG.info("Reading network node layer %s from %s", nodes_layer, gdb_path)
    node_gdf = gpd.read_file(str(gdb_path), layer=nodes_layer)
    _PARTC_REV1_LOG.info("Read node layer rows=%s columns=%s crs=%s", len(node_gdf), len(node_gdf.columns), node_gdf.crs)
    link_id_col = _partC_rev1_schema_col(config, "link_id_col", "ID")
    node_id_col = _partC_rev1_schema_col(config, "node_id_col", "ID")
    if link_id_col in link_gdf.columns:
        link_gdf[link_id_col] = link_gdf[link_id_col].astype("string")
    if node_id_col in node_gdf.columns:
        node_gdf[node_id_col] = node_gdf[node_id_col].astype("string")
    return link_gdf, node_gdf


def _partC_rev1_numeric(df: pd.DataFrame, column: str | None, default: float | None = None) -> pd.Series:
    if column is None or column not in df.columns:
        series = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    else:
        series = pd.to_numeric(df[column], errors="coerce")
    if default is not None:
        series = series.fillna(default)
    return series


def _partC_rev1_positive(series: pd.Series, threshold: float) -> pd.Series:
    return series.fillna(0.0) > threshold


def _partC_rev1_free_time(
    df: pd.DataFrame,
    *,
    time_col: str | None,
    speed_col: str | None,
    length_miles: pd.Series,
    default_speed_mph: float,
) -> pd.Series:
    explicit = _partC_rev1_numeric(df, time_col)
    speed = _partC_rev1_numeric(df, speed_col)
    fallback_speed = speed.where(speed > 0, default_speed_mph)
    computed = length_miles.fillna(0.0) / fallback_speed * 60.0
    return explicit.where(explicit > 0, computed)


def _partC_rev1_status_open(df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    status_col = _partC_rev1_find_optional_column(df, _partC_rev1_schema_col(config, "status_col", "Status"))
    if status_col is None:
        return pd.Series([True] * len(df), index=df.index)
    status = df[status_col]
    return status.isna() | status.astype(str).str.strip().isin({"1", "1.0", "True", "true", "open", "Open"})


def _partC_rev1_truck_valid(df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    truck_col = _partC_rev1_find_optional_column(df, _partC_rev1_schema_col(config, "truck_col", "Truck"))
    if truck_col is None:
        return pd.Series([True] * len(df), index=df.index)
    return ~df[truck_col].astype(str).str.strip().str.lower().eq("prohibited")


def _partC_rev1_reverse_geometry(geom: Any) -> Any:
    if geom is None:
        return None
    try:
        return shapely_reverse(geom)
    except Exception:
        return geom


def _partC_rev1_arc_frame(
    links: gpd.GeoDataFrame,
    config: dict[str, Any],
    *,
    direction: str,
    lane_col: str | None,
    time_col: str | None,
    speed_col: str | None,
) -> gpd.GeoDataFrame:
    length = pd.to_numeric(links.get("partC_rev1_length_miles", pd.Series([np.nan] * len(links))), errors="coerce")
    default_speed = float(config.get("preprocess", {}).get("default_speed_mph", 45.0))
    lanes = _partC_rev1_numeric(links, lane_col)
    default_lanes = float(config.get("preprocess", {}).get("default_lanes", 1.0))
    capacity_per_lane = float(config.get("preprocess", {}).get("capacity_per_lane", 2000.0))
    free_time = _partC_rev1_free_time(
        links,
        time_col=time_col,
        speed_col=speed_col,
        length_miles=length,
        default_speed_mph=default_speed,
    )
    effective_lanes = lanes.where(lanes > 0, default_lanes)
    out = gpd.GeoDataFrame(
        {
            "partC_rev1_link_uid": links["partC_rev1_link_uid"].values,
            "physical_link_id": links["physical_link_id"].astype("string").values,
            "direction": direction,
            "length_miles": length.values,
            "free_time_min": free_time.values,
            "capacity": (effective_lanes * capacity_per_lane).values,
            "alpha": float(config.get("preprocess", {}).get("default_alpha", 0.15)),
            "beta": float(config.get("preprocess", {}).get("default_beta", 4.0)),
            "class_code": links.get("partC_rev1_class", pd.Series([None] * len(links))).values,
            "nhs": links.get("partC_rev1_nhs", pd.Series([None] * len(links))).values,
            "nhfn": links.get("partC_rev1_nhfn", pd.Series([None] * len(links))).values,
            "truck": links.get("partC_rev1_truck", pd.Series([None] * len(links))).values,
            "status": links.get("partC_rev1_status", pd.Series([None] * len(links))).values,
            "road_name": links.get("partC_rev1_road_name", pd.Series([None] * len(links))).values,
            "sign_route": links.get("partC_rev1_sign_route", pd.Series([None] * len(links))).values,
            "route_type": links.get("partC_rev1_route_type", pd.Series([None] * len(links))).values,
            "route_number": links.get("partC_rev1_route_number", pd.Series([None] * len(links))).values,
            "ctfips": links.get("partC_rev1_ctfips", pd.Series([None] * len(links))).values,
            "fafzone": links.get("partC_rev1_fafzone", pd.Series([None] * len(links))).values,
            "is_valid_for_truck": _partC_rev1_truck_valid(links, config).values,
            "is_open": _partC_rev1_status_open(links, config).values,
            "is_backbone": False,
            "region_id": pd.NA,
            "partC_rev1_snap_distance_A_m": links.get("partC_rev1_snap_distance_A_m", pd.Series([None] * len(links))).values,
            "partC_rev1_snap_distance_B_m": links.get("partC_rev1_snap_distance_B_m", pd.Series([None] * len(links))).values,
        },
        geometry=(
            links.geometry.map(_partC_rev1_reverse_geometry).tolist()
            if direction == "BA"
            else links.geometry.tolist()
        ),
        crs=links.crs,
    )
    if direction == "AB":
        out["from_node_uid"] = links["partC_rev1_A_node_uid"].values
        out["to_node_uid"] = links["partC_rev1_B_node_uid"].values
        out["from_node_id"] = links["partC_rev1_A_original_node_id"].astype("string").values
        out["to_node_id"] = links["partC_rev1_B_original_node_id"].astype("string").values
    else:
        out["from_node_uid"] = links["partC_rev1_B_node_uid"].values
        out["to_node_uid"] = links["partC_rev1_A_node_uid"].values
        out["from_node_id"] = links["partC_rev1_B_original_node_id"].astype("string").values
        out["to_node_id"] = links["partC_rev1_A_original_node_id"].astype("string").values
    out["partC_rev1_geom_wkb"] = out.geometry.map(lambda geom: getattr(geom, "wkb_hex", None) if geom is not None else None)
    return out


def _partC_rev1_order_arc_columns(arcs: gpd.GeoDataFrame, crs: Any) -> gpd.GeoDataFrame:
    ordered_cols = [
        "directed_arc_id",
        "partC_rev1_link_uid",
        "physical_link_id",
        "from_node_uid",
        "to_node_uid",
        "from_node_id",
        "to_node_id",
        "direction",
        "length_miles",
        "free_time_min",
        "capacity",
        "partC_rev1_missing_lane_defaulted",
        "partC_rev1_default_lane_count",
        "alpha",
        "beta",
        "class_code",
        "nhs",
        "nhfn",
        "truck",
        "status",
        "road_name",
        "sign_route",
        "route_type",
        "route_number",
        "ctfips",
        "fafzone",
        "is_valid_for_truck",
        "is_open",
        "is_backbone",
        "region_id",
        "partC_rev1_snap_distance_A_m",
        "partC_rev1_snap_distance_B_m",
        "partC_rev1_geom_wkb",
        "geometry",
    ]
    for column in ordered_cols:
        if column not in arcs.columns:
            arcs[column] = pd.NA
    return gpd.GeoDataFrame(arcs[ordered_cols], geometry="geometry", crs=crs)


def _partC_rev1_concat_and_sort_arcs(frames: list[gpd.GeoDataFrame], crs: Any) -> gpd.GeoDataFrame:
    if not frames:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
    arcs = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs)
    arcs["_partC_rev1_direction_order"] = arcs["direction"].map({"AB": 0, "BA": 1}).fillna(9)
    arcs = arcs.sort_values(["partC_rev1_link_uid", "_partC_rev1_direction_order"], kind="mergesort").drop(
        columns=["_partC_rev1_direction_order"]
    )
    return arcs.reset_index(drop=True)


def partC_rev1_build_directed_arcs(
    clean_links_with_topology: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> gpd.GeoDataFrame:
    _PARTC_REV1_LOG.info("Building directed arcs from %s cleaned links", len(clean_links_with_topology))
    direction_cfg = config.get("direction_inference", {})
    threshold = float(direction_cfg.get("positive_lane_threshold", 0.0))
    ab_lanes_col = _partC_rev1_find_optional_column(
        clean_links_with_topology,
        str(direction_cfg.get("ab_lanes_col") or _partC_rev1_schema_col(config, "ab_lanes_col", "AB_Lanes")),
    )
    ba_lanes_col = _partC_rev1_find_optional_column(
        clean_links_with_topology,
        str(direction_cfg.get("ba_lanes_col") or _partC_rev1_schema_col(config, "ba_lanes_col", "BA_Lanes")),
    )
    dir_col = _partC_rev1_find_optional_column(
        clean_links_with_topology,
        str(direction_cfg.get("dir_col") or _partC_rev1_schema_col(config, "dir_col", "Dir")),
    )
    ab_time_col = _partC_rev1_find_optional_column(clean_links_with_topology, _partC_rev1_schema_col(config, "ab_time_col", "AB_FreeFlowTime"))
    ba_time_col = _partC_rev1_find_optional_column(clean_links_with_topology, _partC_rev1_schema_col(config, "ba_time_col", "BA_FreeFlowTime"))
    ab_speed_col = _partC_rev1_find_optional_column(clean_links_with_topology, _partC_rev1_schema_col(config, "ab_speed_col", "AB_FinalSpeed"))
    ba_speed_col = _partC_rev1_find_optional_column(clean_links_with_topology, _partC_rev1_schema_col(config, "ba_speed_col", "BA_FinalSpeed"))

    links = clean_links_with_topology.copy()
    valid_topology = ~links.get("partC_rev1_snap_failed", pd.Series([False] * len(links), index=links.index)).fillna(False)
    ab_lanes = _partC_rev1_numeric(links, ab_lanes_col)
    ba_lanes = _partC_rev1_numeric(links, ba_lanes_col)
    ab_positive = _partC_rev1_positive(ab_lanes, threshold)
    ba_positive = _partC_rev1_positive(ba_lanes, threshold)

    fallback_enabled = bool(direction_cfg.get("fallback_to_dir_when_lanes_missing", False))
    default_no_lane_bidirectional = bool(direction_cfg.get("assume_missing_lanes_bidirectional", True))
    default_no_lane_case = str(direction_cfg.get("missing_lane_direction_case", "DEFAULT_AB_BA_1_LANE"))
    fallback_ab = pd.Series([False] * len(links), index=links.index)
    fallback_ba = pd.Series([False] * len(links), index=links.index)
    no_lane_info = ~(ab_positive | ba_positive)
    if fallback_enabled and dir_col is not None:
        dir_values = links[dir_col].astype(str).str.strip()
        two_way = {str(value) for value in direction_cfg.get("dir_two_way_values", [0, "0"])}
        ab_only = {str(value) for value in direction_cfg.get("dir_ab_only_values", [1, "1"])}
        ba_only = {str(value) for value in direction_cfg.get("dir_ba_only_values", [])}
        fallback_ab = no_lane_info & (dir_values.isin(two_way) | dir_values.isin(ab_only))
        fallback_ba = no_lane_info & (dir_values.isin(two_way) | dir_values.isin(ba_only))
        _PARTC_REV1_LOG.info("Direction fallback to Dir enabled for links with missing/non-positive lanes")

    make_ab = valid_topology & (ab_positive | fallback_ab)
    make_ba = valid_topology & (ba_positive | fallback_ba)
    make_default_ab_ba = valid_topology & no_lane_info & default_no_lane_bidirectional & ~(fallback_ab | fallback_ba)
    case = np.select(
        [
            ab_positive & ba_positive,
            ab_positive & ~ba_positive,
            ~ab_positive & ba_positive,
            make_default_ab_ba,
            fallback_ab & fallback_ba,
            fallback_ab & ~fallback_ba,
            no_lane_info,
        ],
        ["AB_BA", "AB_ONLY", "BA_ONLY", default_no_lane_case, "FALLBACK_DIR_AB_BA", "FALLBACK_DIR_AB", "IGNORED_NO_LANE_INFO"],
        default="IGNORED_NO_LANE_INFO",
    )
    links["partC_rev1_direction_case"] = case
    links["partC_rev1_missing_lane_defaulted"] = make_default_ab_ba
    links["partC_rev1_default_lane_count"] = np.where(
        make_default_ab_ba,
        float(config.get("preprocess", {}).get("default_lanes", 1.0)),
        np.nan,
    )

    lane_frames: list[gpd.GeoDataFrame] = []
    if make_ab.any():
        _PARTC_REV1_LOG.info("Creating %s AB directed arcs", int(make_ab.sum()))
        lane_frames.append(
            _partC_rev1_arc_frame(
                links.loc[make_ab].copy(),
                config,
                direction="AB",
                lane_col=ab_lanes_col,
                time_col=ab_time_col,
                speed_col=ab_speed_col,
            )
        )
    if make_ba.any():
        _PARTC_REV1_LOG.info("Creating %s BA directed arcs", int(make_ba.sum()))
        lane_frames.append(
            _partC_rev1_arc_frame(
                links.loc[make_ba].copy(),
                config,
                direction="BA",
                lane_col=ba_lanes_col,
                time_col=ba_time_col,
                speed_col=ba_speed_col,
            )
        )

    lane_arcs = _partC_rev1_concat_and_sort_arcs(lane_frames, links.crs)
    lane_arcs["directed_arc_id"] = range(len(lane_arcs))

    default_frames: list[gpd.GeoDataFrame] = []
    if make_default_ab_ba.any():
        default_links = links.loc[make_default_ab_ba].copy()
        default_links["partC_rev1_missing_lane_defaulted"] = True
        default_links["partC_rev1_default_lane_count"] = float(config.get("preprocess", {}).get("default_lanes", 1.0))
        _PARTC_REV1_LOG.info(
            "Creating %s default 1-lane AB and %s default 1-lane BA arcs for no-lane links",
            int(make_default_ab_ba.sum()),
            int(make_default_ab_ba.sum()),
        )
        default_frames.append(
            _partC_rev1_arc_frame(
                default_links,
                config,
                direction="AB",
                lane_col=None,
                time_col=ab_time_col,
                speed_col=ab_speed_col,
            )
        )
        default_frames.append(
            _partC_rev1_arc_frame(
                default_links,
                config,
                direction="BA",
                lane_col=None,
                time_col=ba_time_col,
                speed_col=ba_speed_col,
            )
        )
    default_arcs = _partC_rev1_concat_and_sort_arcs(default_frames, links.crs)
    default_arcs["directed_arc_id"] = range(len(lane_arcs), len(lane_arcs) + len(default_arcs))
    if len(default_arcs):
        default_arcs["partC_rev1_missing_lane_defaulted"] = True
        default_arcs["partC_rev1_default_lane_count"] = float(config.get("preprocess", {}).get("default_lanes", 1.0))

    arcs = gpd.GeoDataFrame(pd.concat([lane_arcs, default_arcs], ignore_index=True), geometry="geometry", crs=links.crs)
    arcs = _partC_rev1_order_arc_columns(arcs, links.crs)

    ignored = links.loc[
        (links["partC_rev1_direction_case"].eq("IGNORED_NO_LANE_INFO") & ~make_default_ab_ba) | ~valid_topology
    ].copy()
    summary = (
        links.groupby("partC_rev1_direction_case", dropna=False)
        .size()
        .reset_index(name="link_count")
        .sort_values("partC_rev1_direction_case")
    )
    summary["directed_arc_rows"] = len(arcs)
    partC_rev1_write_table(summary, partC_rev1_output_file(config, "partC_rev1_direction_inference_summary.csv"), index=False)
    partC_rev1_write_table(ignored, partC_rev1_output_file(config, "partC_rev1_ignored_links_no_lane_info.csv"), index=False)
    partC_rev1_write_table(ignored, partC_rev1_output_file(config, "partC_rev1_links_excluded_no_lane_direction.csv"), index=False)
    partC_rev1_write_table(
        links.loc[make_default_ab_ba].copy(),
        partC_rev1_output_file(config, "partC_rev1_links_defaulted_no_lane_direction.csv"),
        index=False,
    )
    clean_links_with_topology.loc[:, "partC_rev1_direction_case"] = links["partC_rev1_direction_case"].values
    clean_links_with_topology.loc[:, "partC_rev1_missing_lane_defaulted"] = links["partC_rev1_missing_lane_defaulted"].values
    clean_links_with_topology.loc[:, "partC_rev1_default_lane_count"] = links["partC_rev1_default_lane_count"].values
    _PARTC_REV1_LOG.info(
        "Directed arc creation complete: arc rows=%s ignored/no-lane-or-invalid=%s",
        len(arcs),
        len(ignored),
    )
    return arcs
