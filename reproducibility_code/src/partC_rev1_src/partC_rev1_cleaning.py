"""Exact duplicate cleaning and normalized physical table construction."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import set_precision

from .partC_rev1_io import partC_rev1_write_table
from .partC_rev1_paths import partC_rev1_output_file
from .partC_rev1_schema import partC_rev1_coerce_string_id, partC_rev1_find_column, partC_rev1_parse_county_fips_from_centroid_id


_PARTC_REV1_LOG = logging.getLogger("partC_rev1")


def _partC_rev1_config_output_path(config: dict[str, Any], filename: str) -> Path:
    return partC_rev1_output_file(config, filename)


def _partC_rev1_column(config: dict[str, Any], key: str, default: str) -> str:
    return str(config.get("input_schema", {}).get(key, default) or default)


def _partC_rev1_normalize_scalar(value: Any) -> str:
    if value is None:
        return "<NA>"
    try:
        if pd.isna(value):
            return "<NA>"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(value, sort_keys=True, default=str)
    return repr(value)


def _partC_rev1_geometry_key(geom: Any, config: dict[str, Any]) -> str:
    if geom is None:
        return "<NULL_GEOMETRY>"
    try:
        if geom.is_empty:
            return "<EMPTY_GEOMETRY>"
    except AttributeError:
        return _partC_rev1_normalize_scalar(geom)
    precision = int(config.get("data_cleaning", {}).get("geometry_comparison_precision", 8))
    try:
        geom = set_precision(geom, 10.0 ** (-precision))
    except Exception:
        pass
    if bool(config.get("data_cleaning", {}).get("deduplicate_geometry_using_wkb", True)):
        return geom.wkb_hex
    return geom.wkt


def _partC_rev1_geom_wkb(geom: Any) -> str | None:
    if geom is None:
        return None
    try:
        return geom.wkb_hex
    except Exception:
        return None


def _partC_rev1_comparable_frame(gdf: gpd.GeoDataFrame, config: dict[str, Any]) -> pd.DataFrame:
    non_geometry = [column for column in gdf.columns if column != "geometry"]
    comparable = pd.DataFrame(index=gdf.index)
    for column in non_geometry:
        comparable[column] = gdf[column].map(_partC_rev1_normalize_scalar)
    comparable["partC_rev1_geometry_key"] = gdf.geometry.map(lambda geom: _partC_rev1_geometry_key(geom, config))
    return comparable


def _partC_rev1_group_id(series: pd.Series, prefix: str) -> pd.Series:
    counts = series.fillna("<NA>").astype(str).map(series.fillna("<NA>").astype(str).value_counts())
    return series.fillna("<NA>").astype(str).where(counts > 1, None).map(
        lambda value: None if value is None else f"{prefix}{hashlib.sha1(str(value).encode('utf-8')).hexdigest()[:10]}"
    )


def _partC_rev1_same_geometry_nonidentical_report(
    gdf: gpd.GeoDataFrame,
    comparable: pd.DataFrame,
    *,
    report_type: str,
) -> pd.DataFrame:
    non_geometry = [column for column in comparable.columns if column != "partC_rev1_geometry_key"]
    geom_key = comparable["partC_rev1_geometry_key"]
    duplicated_geometry = geom_key.duplicated(keep=False)
    if not duplicated_geometry.any():
        return pd.DataFrame(columns=["report_type", "partC_rev1_geometry_key", "row_count", "unique_non_geometry_rows", "source_indices"])
    comparable = comparable.loc[duplicated_geometry]
    _PARTC_REV1_LOG.info(
        "Building %s report over %s rows in %s duplicated-geometry groups",
        report_type,
        len(comparable),
        comparable["partC_rev1_geometry_key"].nunique(dropna=False),
    )
    rows: list[dict[str, Any]] = []
    grouped = comparable.groupby("partC_rev1_geometry_key", dropna=False, sort=True)
    for idx, (geom_key, group) in enumerate(grouped, start=1):
        if idx % 100000 == 0:
            _PARTC_REV1_LOG.info("Processed %s duplicated-geometry groups for %s", idx, report_type)
        if len(group) <= 1:
            continue
        unique_non_geom = group[non_geometry].drop_duplicates()
        if len(unique_non_geom) <= 1:
            continue
        rows.append(
            {
                "report_type": report_type,
                "partC_rev1_geometry_key": geom_key,
                "row_count": len(group),
                "unique_non_geometry_rows": len(unique_non_geom),
                "source_indices": ",".join(map(str, group.index.tolist())),
            }
        )
    return pd.DataFrame(rows)


def _partC_rev1_exact_duplicate_report(
    original: gpd.GeoDataFrame,
    comparable: pd.DataFrame,
    duplicate_mask: pd.Series,
    *,
    report_type: str,
) -> pd.DataFrame:
    if not duplicate_mask.any():
        return pd.DataFrame(columns=["report_type", "dropped_source_index", "duplicate_key"])
    keys = comparable.astype(str).agg("||".join, axis=1)
    return pd.DataFrame(
        {
            "report_type": report_type,
            "dropped_source_index": original.index[duplicate_mask].astype(str),
            "duplicate_key": keys.loc[duplicate_mask].values,
        }
    )


def _partC_rev1_duplicate_id_report(
    gdf: gpd.GeoDataFrame,
    comparable: pd.DataFrame,
    id_col: str,
    *,
    report_type: str,
) -> pd.DataFrame:
    if id_col not in gdf.columns:
        return pd.DataFrame(columns=["report_type", "original_id", "row_count", "unique_row_count", "source_indices"])
    rows: list[dict[str, Any]] = []
    ids = partC_rev1_coerce_string_id(gdf[id_col])
    duplicated_ids = ids.duplicated(keep=False)
    if not duplicated_ids.any():
        return pd.DataFrame(columns=["report_type", "original_id", "row_count", "unique_row_count", "source_indices"])
    comparable = comparable.loc[duplicated_ids]
    ids = ids.loc[duplicated_ids]
    _PARTC_REV1_LOG.info(
        "Building %s report over %s rows in %s duplicated-id groups",
        report_type,
        len(comparable),
        ids.nunique(dropna=False),
    )
    for idx, (original_id, group) in enumerate(comparable.groupby(ids, dropna=False, sort=True), start=1):
        if idx % 100000 == 0:
            _PARTC_REV1_LOG.info("Processed %s duplicated-id groups for %s", idx, report_type)
        if len(group) <= 1:
            continue
        rows.append(
            {
                "report_type": report_type,
                "original_id": original_id,
                "row_count": len(group),
                "unique_row_count": len(group.drop_duplicates()),
                "source_indices": ",".join(map(str, group.index.tolist())),
            }
        )
    return pd.DataFrame(rows)


def _partC_rev1_write_reports(config: dict[str, Any], reports: dict[str, pd.DataFrame]) -> None:
    for filename, report in reports.items():
        _PARTC_REV1_LOG.info("Writing report %s with %s rows", filename, len(report))
        partC_rev1_write_table(report, _partC_rev1_config_output_path(config, filename), index=False)


def _partC_rev1_concat_reports(reports: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [report for report in reports if not report.empty]
    if nonempty:
        return pd.concat(nonempty, ignore_index=True)
    columns: list[str] = []
    for report in reports:
        columns.extend([column for column in report.columns if column not in columns])
    return pd.DataFrame(columns=columns)


def _partC_rev1_sorted_clean_gdf(gdf: gpd.GeoDataFrame, comparable: pd.DataFrame, id_values: pd.Series) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["_partC_rev1_sort_original_id"] = id_values.fillna("").astype(str).values
    out["_partC_rev1_sort_geometry_key"] = comparable.loc[out.index, "partC_rev1_geometry_key"].astype(str).values
    out["_partC_rev1_sort_source_index"] = out.index.astype(str)
    out = out.sort_values(
        ["_partC_rev1_sort_original_id", "_partC_rev1_sort_geometry_key", "_partC_rev1_sort_source_index"],
        kind="mergesort",
    ).drop(columns=["_partC_rev1_sort_original_id", "_partC_rev1_sort_geometry_key", "_partC_rev1_sort_source_index"])
    return out.reset_index(drop=True)


def partC_rev1_clean_exact_duplicate_nodes(
    node_gdf: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    node_id_col = _partC_rev1_column(config, "node_id_col", "ID")
    original = node_gdf.copy()
    _PARTC_REV1_LOG.info("Cleaning exact duplicate nodes: raw rows=%s columns=%s", len(original), len(original.columns))
    comparable = _partC_rev1_comparable_frame(original, config)
    duplicate_mask = comparable.duplicated(keep="first")
    _PARTC_REV1_LOG.info("Node exact duplicate rows detected=%s", int(duplicate_mask.sum()))
    clean = original.loc[~duplicate_mask].copy()
    clean_comparable = comparable.loc[~duplicate_mask].copy()
    original_ids = partC_rev1_coerce_string_id(clean[node_id_col]) if node_id_col in clean.columns else pd.Series([None] * len(clean))
    clean = _partC_rev1_sorted_clean_gdf(clean, clean_comparable, original_ids)
    original_ids = partC_rev1_coerce_string_id(clean[node_id_col]) if node_id_col in clean.columns else pd.Series([None] * len(clean))
    clean["partC_rev1_node_uid"] = range(len(clean))
    clean["partC_rev1_original_node_id"] = original_ids.values
    clean["partC_rev1_node_duplicate_group_id"] = _partC_rev1_group_id(original_ids, "partC_rev1_node_dup_").values

    lon_col = str(config.get("input_schema", {}).get("node_lon_col", "") or "")
    lat_col = str(config.get("input_schema", {}).get("node_lat_col", "") or "")
    if lon_col and lat_col and lon_col in clean.columns and lat_col in clean.columns:
        clean["partC_rev1_lon"] = pd.to_numeric(clean[lon_col], errors="coerce")
        clean["partC_rev1_lat"] = pd.to_numeric(clean[lat_col], errors="coerce")
    else:
        clean["partC_rev1_lon"] = clean.geometry.map(lambda geom: geom.x if geom is not None and not geom.is_empty else None)
        clean["partC_rev1_lat"] = clean.geometry.map(lambda geom: geom.y if geom is not None and not geom.is_empty else None)

    centroid_col = _partC_rev1_column(config, "centroid_col", "Centroid")
    centroid_id_col = _partC_rev1_column(config, "centroid_id_col", "CentroidID")
    fafid_col = _partC_rev1_column(config, "fafid_col", "FAFID")
    clean["partC_rev1_centroid"] = (
        pd.to_numeric(clean[centroid_col], errors="coerce").fillna(0).astype(int)
        if centroid_col in clean.columns
        else 0
    )
    clean["partC_rev1_centroid_id_raw"] = (
        partC_rev1_coerce_string_id(clean[centroid_id_col]) if centroid_id_col in clean.columns else pd.Series([None] * len(clean), dtype="string")
    )
    clean["partC_rev1_county_fips"] = clean["partC_rev1_centroid_id_raw"].map(partC_rev1_parse_county_fips_from_centroid_id)
    clean["partC_rev1_fafid"] = (
        partC_rev1_coerce_string_id(clean[fafid_col]) if fafid_col in clean.columns else pd.Series([None] * len(clean), dtype="string")
    )
    clean["partC_rev1_geom_wkb"] = clean.geometry.map(_partC_rev1_geom_wkb)
    clean["partC_rev1_duplicate_status"] = clean["partC_rev1_node_duplicate_group_id"].map(
        lambda value: "duplicate_original_id" if value is not None and not pd.isna(value) else "unique"
    )

    exact_report = _partC_rev1_exact_duplicate_report(original, comparable, duplicate_mask, report_type="node_exact_duplicate")
    same_geom_report = _partC_rev1_same_geometry_nonidentical_report(original, comparable, report_type="node_same_geometry_nonidentical")
    duplicate_id_report = _partC_rev1_duplicate_id_report(original, comparable, node_id_col, report_type="node_duplicate_original_id")
    _partC_rev1_write_reports(
        config,
        {
            "partC_rev1_duplicate_node_report.csv": _partC_rev1_concat_reports([exact_report, duplicate_id_report]),
            "partC_rev1_same_geometry_nonidentical_node_report.csv": same_geom_report,
            "partC_rev1_suspicious_duplicate_node_report.csv": same_geom_report,
        },
    )
    _PARTC_REV1_LOG.info("Cleaned nodes complete: clean rows=%s", len(clean))
    return gpd.GeoDataFrame(clean, geometry="geometry", crs=node_gdf.crs), _partC_rev1_concat_reports([exact_report, duplicate_id_report, same_geom_report])


def partC_rev1_clean_exact_duplicate_links(
    link_gdf: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    link_id_col = _partC_rev1_column(config, "link_id_col", "ID")
    original = link_gdf.copy()
    _PARTC_REV1_LOG.info("Cleaning exact duplicate links: raw rows=%s columns=%s", len(original), len(original.columns))
    comparable = _partC_rev1_comparable_frame(original, config)
    original["partC_rev1_original_row_number"] = range(len(original))
    duplicate_mask = comparable.duplicated(keep="first")
    _PARTC_REV1_LOG.info("Link exact duplicate rows detected=%s", int(duplicate_mask.sum()))
    clean = original.loc[~duplicate_mask].copy()
    clean_comparable = comparable.loc[~duplicate_mask].copy()
    original_ids = partC_rev1_coerce_string_id(clean[link_id_col]) if link_id_col in clean.columns else pd.Series([None] * len(clean))
    clean = _partC_rev1_sorted_clean_gdf(clean, clean_comparable, original_ids)
    original_ids = partC_rev1_coerce_string_id(clean[link_id_col]) if link_id_col in clean.columns else pd.Series([None] * len(clean))
    clean["partC_rev1_link_uid"] = range(len(clean))
    clean["physical_link_id"] = original_ids.values
    clean["partC_rev1_link_duplicate_group_id"] = _partC_rev1_group_id(original_ids, "partC_rev1_link_dup_").values

    schema = config.get("input_schema", {})
    normalized_cols = {
        "length_col": "partC_rev1_length_miles",
        "class_col": "partC_rev1_class",
        "nhs_col": "partC_rev1_nhs",
        "nhfn_col": "partC_rev1_nhfn",
        "truck_col": "partC_rev1_truck",
        "status_col": "partC_rev1_status",
        "road_name_col": "partC_rev1_road_name",
        "sign_route_col": "partC_rev1_sign_route",
        "route_type_col": "partC_rev1_route_type",
        "route_number_col": "partC_rev1_route_number",
        "ctfips_col": "partC_rev1_ctfips",
        "fafzone_col": "partC_rev1_fafzone",
    }
    defaults = {
        "length_col": "LENGTH",
        "class_col": "Class",
        "nhs_col": "NHS",
        "nhfn_col": "NHFN",
        "truck_col": "Truck",
        "status_col": "Status",
        "road_name_col": "Road_Name",
        "sign_route_col": "Sign_Rte",
        "route_type_col": "Rte_Type",
        "route_number_col": "Rte_Number",
        "ctfips_col": "CTFIPS",
        "fafzone_col": "FAFZONE",
    }
    source_candidates = {
        "road_name_col": ["Road_Name", "road_name", "partC_rev1_road_name"],
        "sign_route_col": ["Sign_Rte", "sign_route", "partC_rev1_sign_route"],
        "route_type_col": ["Rte_Type", "route_type", "partC_rev1_route_type"],
        "route_number_col": ["Rte_Number", "route_number", "partC_rev1_route_number"],
    }
    for key, out_col in normalized_cols.items():
        preferred = str(schema.get(key, defaults[key]) or defaults[key])
        source_col = partC_rev1_find_column(clean, preferred, candidates=source_candidates.get(key, [defaults[key], out_col]), required=False)
        clean[out_col] = clean[source_col] if source_col is not None else None
    clean["partC_rev1_geom_wkb"] = clean.geometry.map(_partC_rev1_geom_wkb)
    clean["partC_rev1_duplicate_status"] = clean["partC_rev1_link_duplicate_group_id"].map(
        lambda value: "duplicate_original_id" if value is not None and not pd.isna(value) else "unique"
    )

    exact_report = _partC_rev1_exact_duplicate_report(original, comparable, duplicate_mask, report_type="link_exact_duplicate")
    same_geom_report = _partC_rev1_same_geometry_nonidentical_report(original, comparable, report_type="link_same_geometry_nonidentical")
    duplicate_id_report = _partC_rev1_duplicate_id_report(original, comparable, link_id_col, report_type="link_duplicate_original_id")
    _partC_rev1_write_reports(
        config,
        {
            "partC_rev1_duplicate_link_report.csv": _partC_rev1_concat_reports([exact_report, duplicate_id_report]),
            "partC_rev1_same_geometry_nonidentical_link_report.csv": same_geom_report,
            "partC_rev1_suspicious_duplicate_link_report.csv": same_geom_report,
        },
    )
    _PARTC_REV1_LOG.info("Cleaned links complete: clean rows=%s", len(clean))
    return gpd.GeoDataFrame(clean, geometry="geometry", crs=link_gdf.crs), _partC_rev1_concat_reports([exact_report, duplicate_id_report, same_geom_report])
