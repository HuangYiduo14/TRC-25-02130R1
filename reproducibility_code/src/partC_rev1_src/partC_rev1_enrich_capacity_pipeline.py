"""Prompt 2.5 enrichment pipeline orchestration."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from .partC_rev1_cache import (
    partC_rev1_connect_sqlite,
    partC_rev1_make_unique_columns,
    partC_rev1_read_table_sqlite,
    partC_rev1_write_metadata,
    partC_rev1_write_table_sqlite,
)
from .partC_rev1_capacity_enrichment import (
    partC_rev1_attach_link_aadt_defaults,
    partC_rev1_enrich_directed_arcs_capacity,
    partC_rev1_preserve_existing_columns,
)
from .partC_rev1_config import load_partC_rev1_config
from .partC_rev1_gdb import partC_rev1_read_gdb_layer
from .partC_rev1_io import partC_rev1_write_table
from .partC_rev1_logging import partC_rev1_setup_logging
from .partC_rev1_ntad_io import (
    partC_rev1_inspect_ntad_sources,
    partC_rev1_ntad_sqlite_table,
    partC_rev1_read_ntad_source,
    partC_rev1_resolve_ntad_source_paths,
)
from .partC_rev1_paths import partC_rev1_output_file, partC_rev1_resolve_path
from .partC_rev1_road_match import (
    _partC_rev1_midpoint,
    partC_rev1_apply_source_matches,
    partC_rev1_empty_best_matches,
    partC_rev1_inherit_opposite_direction_matches,
    partC_rev1_match_single_source,
    partC_rev1_prepare_faf_links,
    partC_rev1_prepare_ntad_segments,
)


_PARTC_REV1_LOG = logging.getLogger("partC_rev1")
PARTC_REV1_MATCHING_ALGORITHM_VERSION = "buffer_geo_v3"


def _partC_rev1_cache_path(config: dict[str, Any]) -> Path:
    return partC_rev1_resolve_path(str(config.get("preprocess", {}).get("cache_sqlite", "partC_rev1_cache/partC_rev1_network.sqlite")))


def _partC_rev1_clean_gpkg_path(config: dict[str, Any]) -> Path:
    return partC_rev1_resolve_path(
        str(config.get("clean_geodatabase", {}).get("output_gpkg_path", "partC_rev1_cache/partC_rev1_clean_network.gpkg"))
    )


def _partC_rev1_enriched_gpkg_path(config: dict[str, Any]) -> Path:
    return _partC_rev1_clean_gpkg_path(config).with_name("partC_rev1_clean_network_enriched.gpkg")


def _partC_rev1_layer_names(config: dict[str, Any]) -> dict[str, str]:
    layers = config.get("clean_geodatabase", {}).get("layers", {})
    return {
        "links": str(layers.get("clean_links", "partC_rev1_clean_physical_links")),
        "arcs": str(layers.get("directed_arcs", "partC_rev1_directed_arcs")),
    }


def _partC_rev1_matching_algorithm_version(config: dict[str, Any]) -> str:
    return str(config.get("matching", {}).get("matching_algorithm_version", PARTC_REV1_MATCHING_ALGORITHM_VERSION))


def _partC_rev1_escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _partC_rev1_write_markdown_table(df: pd.DataFrame, path: Path, title: str) -> None:
    lines = [f"# {title}", "", f"Generated UTC: {datetime.now(timezone.utc).isoformat()}", ""]
    if df.empty:
        lines.append("(No rows.)")
    else:
        columns = list(df.columns)
        lines.append("| " + " | ".join(_partC_rev1_escape_md(column) for column in columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in df.head(200).itertuples(index=False):
            lines.append("| " + " | ".join(_partC_rev1_escape_md(value) for value in row) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _partC_rev1_write_csv(config: dict[str, Any], df: pd.DataFrame, filename: str) -> None:
    partC_rev1_write_table(df, partC_rev1_output_file(config, filename), index=False)


def _partC_rev1_layer_ready(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = partC_rev1_make_unique_columns(gdf)
    for column in list(out.columns):
        if column == out.geometry.name:
            continue
        if str(out[column].dtype) == "geometry":
            out[column] = out[column].map(lambda value: getattr(value, "wkb_hex", None) if value is not None else None)
        elif out[column].dtype == "object":
            out[column] = out[column].map(lambda value: None if value is None or (not isinstance(value, (list, dict, tuple, set)) and pd.isna(value)) else str(value))
        elif str(out[column].dtype) == "boolean":
            out[column] = out[column].astype("Int64")
    return out


def _partC_rev1_write_gpkg_layer(gdf: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    _PARTC_REV1_LOG.info("Writing enriched GeoPackage layer %s rows=%s", layer, len(gdf))
    _partC_rev1_layer_ready(gdf).to_file(path, layer=layer, driver="GPKG")


def _partC_rev1_validate_full_run_inputs(config: dict[str, Any], confirm_full_run: bool) -> None:
    matching = config.get("matching", {})
    testing = config.get("testing", {})
    real_run = not bool(testing.get("use_toy_data", True))
    if real_run and not confirm_full_run:
        raise ValueError("Full real-data enrichment requires --confirm-full-run.")
    if confirm_full_run:
        if bool(testing.get("use_toy_data", False)):
            raise ValueError("Full real-data enrichment requires testing.use_toy_data=false.")
        if matching.get("max_faf_links") is not None:
            raise ValueError("Full real-data enrichment requires matching.max_faf_links=null.")
        if bool(matching.get("dry_run", False)):
            raise ValueError("Full real-data enrichment requires matching.dry_run=false.")
    cache_path = _partC_rev1_cache_path(config)
    gpkg_path = _partC_rev1_clean_gpkg_path(config)
    if not cache_path.exists():
        raise FileNotFoundError(f"Prompt 2 SQLite cache missing: {cache_path}")
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Prompt 2 clean GeoPackage missing: {gpkg_path}")
    partC_rev1_resolve_ntad_source_paths(config)


def _partC_rev1_recover_strahnet(links: gpd.GeoDataFrame, config: dict[str, Any]) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    out = links.copy()
    if "STRAHNET" in out.columns:
        out["partC_rev1_raw_STRAHNET"] = out["STRAHNET"]
        source = "existing_clean_physical_links"
        warning = ""
    else:
        source = "raw_faf_gdb_join_by_ID"
        warning = ""
        out["STRAHNET"] = np.nan
        try:
            inputs = config.get("inputs", {})
            raw = partC_rev1_read_gdb_layer(str(partC_rev1_resolve_path(str(inputs.get("network_gdb_path", "")))), str(inputs.get("links_layer", "FAF5_Links")))
            if "ID" in raw.columns and "STRAHNET" in raw.columns and "ID" in out.columns:
                raw_lookup = raw[["ID", "STRAHNET"]].drop_duplicates("ID")
                out = out.merge(raw_lookup, on="ID", how="left", suffixes=("", "_raw_recovered"))
                out["STRAHNET"] = out["STRAHNET_raw_recovered"]
                out = out.drop(columns=["STRAHNET_raw_recovered"])
            else:
                warning = "raw FAF layer missing ID or STRAHNET"
        except Exception as exc:
            warning = f"raw STRAHNET recovery failed: {exc}"
    out["partC_rev1_raw_STRAHNET"] = out.get("partC_rev1_raw_STRAHNET", out["STRAHNET"])
    out["partC_rev1_faf_strahnet_nonempty"] = out["STRAHNET"].notna() & out["STRAHNET"].astype(str).str.strip().ne("")
    report = pd.DataFrame(
        [
            {
                "source": source,
                "warning": warning,
                "link_rows": len(out),
                "recovered_or_existing_nonempty": int(out["partC_rev1_faf_strahnet_nonempty"].sum()),
            }
        ]
    )
    return out, report


def _partC_rev1_required_match_columns() -> list[str]:
    return [
        "partC_rev1_ntad_match_status",
        "partC_rev1_match_dataset",
        "partC_rev1_match_source_priority",
        "partC_rev1_match_method",
        "partC_rev1_match_score",
        "partC_rev1_match_second_score",
        "partC_rev1_match_ambiguity_ratio",
        "partC_rev1_match_close_competitor",
        "partC_rev1_match_angle_diff_deg",
        "partC_rev1_match_min_distance_m",
        "partC_rev1_match_name_similarity",
        "partC_rev1_match_route_similarity",
        "partC_rev1_match_semantic_score",
        "partC_rev1_match_spatial_only",
        "partC_rev1_match_fips_score",
        "partC_rev1_faf_route_key",
        "partC_rev1_ntad_route_key",
        "partC_rev1_faf_normalized_sign_key",
        "partC_rev1_ntad_normalized_sign_key",
        "partC_rev1_normalized_sign_match",
        "partC_rev1_normalized_sign_global_fallback",
        "partC_rev1_normalized_sign_selected_distance_m",
        "partC_rev1_geo_buffer_radius_m",
        "partC_rev1_geo_buffer_intersects",
        "partC_rev1_geo_buffer_overlap_score",
        "partC_rev1_geo_direct_distance_m",
        "partC_rev1_faf_stfips",
        "partC_rev1_faf_ctfips",
        "partC_rev1_ntad_stfips",
        "partC_rev1_ntad_ctfips",
        "partC_rev1_ntad_uid",
        "partC_rev1_ntad_objectid",
        "partC_rev1_ntad_lname",
        "partC_rev1_ntad_sign1",
        "partC_rev1_ntad_signt1",
        "partC_rev1_ntad_signn1",
        "partC_rev1_ntad_routeid",
        "partC_rev1_ntad_source_label",
        "partC_rev1_ntad_nhs",
        "partC_rev1_ntad_aadt_total",
        "partC_rev1_ntad_aadt_combination",
        "partC_rev1_ntad_aadt_single",
        "partC_rev1_ntad_speed_limit_mph",
    ]


def _partC_rev1_bool_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"} if not pd.isna(value) else False)


def _partC_rev1_ensure_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in _partC_rev1_required_match_columns():
        if column not in out.columns:
            out[column] = np.nan
    out["partC_rev1_ntad_match_status"] = out["partC_rev1_ntad_match_status"].fillna("unassigned")
    out["partC_rev1_match_dataset"] = out["partC_rev1_match_dataset"].fillna("NONE")
    out["partC_rev1_match_method"] = out["partC_rev1_match_method"].fillna("unassigned")
    out["partC_rev1_match_spatial_only"] = out["partC_rev1_match_method"].eq("spatial_only")
    out["partC_rev1_match_close_competitor"] = _partC_rev1_bool_series(out["partC_rev1_match_close_competitor"])
    out["partC_rev1_geo_buffer_intersects"] = _partC_rev1_bool_series(out["partC_rev1_geo_buffer_intersects"])
    out["partC_rev1_normalized_sign_match"] = _partC_rev1_bool_series(out["partC_rev1_normalized_sign_match"])
    out["partC_rev1_normalized_sign_global_fallback"] = _partC_rev1_bool_series(out["partC_rev1_normalized_sign_global_fallback"])
    return out


def _partC_rev1_make_match_lines(
    accepted: pd.DataFrame,
    faf_projected: gpd.GeoDataFrame,
    source_projected: gpd.GeoDataFrame,
    source_label: str,
) -> gpd.GeoDataFrame:
    if accepted.empty:
        return gpd.GeoDataFrame(columns=["partC_rev1_link_uid", "partC_rev1_ntad_uid", "partC_rev1_match_dataset", "geometry"], geometry="geometry", crs=faf_projected.crs)
    faf_by_uid = faf_projected.set_index("partC_rev1_match_link_uid")
    source_by_uid = source_projected.set_index("partC_rev1_ntad_uid")
    rows: list[dict[str, Any]] = []
    for row in accepted.itertuples(index=False):
        link_uid = str(getattr(row, "partC_rev1_link_uid"))
        ntad_uid = str(getattr(row, "partC_rev1_ntad_uid"))
        if link_uid not in faf_by_uid.index or ntad_uid not in source_by_uid.index:
            continue
        faf_mid = _partC_rev1_midpoint(faf_by_uid.loc[link_uid].geometry)
        ntad_mid = _partC_rev1_midpoint(source_by_uid.loc[ntad_uid].geometry)
        if faf_mid is None or ntad_mid is None:
            continue
        rows.append(
            {
                "partC_rev1_link_uid": link_uid,
                "partC_rev1_ntad_uid": ntad_uid,
                "partC_rev1_match_dataset": source_label,
                "partC_rev1_match_method": getattr(row, "partC_rev1_match_method", ""),
                "partC_rev1_match_score": getattr(row, "partC_rev1_match_score", np.nan),
                "geometry": LineString([faf_mid, ntad_mid]),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=faf_projected.crs)


def _partC_rev1_sqlite_table_exists(conn: Any, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _partC_rev1_cached_matching_algorithm_is_current(conn: Any, expected_version: str) -> bool:
    if not _partC_rev1_sqlite_table_exists(conn, "partC_rev1_capacity_enrichment_metadata"):
        return False
    rows = conn.execute(
        'SELECT value FROM "partC_rev1_capacity_enrichment_metadata" WHERE key=?',
        ("partC_rev1_capacity_enrichment",),
    ).fetchall()
    for (raw_value,) in rows:
        try:
            payload = json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            continue
        if str(payload.get("matching_algorithm_version", "")) == expected_version:
            return True
    return False


def _partC_rev1_try_load_cached_matches(
    links: gpd.GeoDataFrame,
    config: dict[str, Any],
    conn: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, gpd.GeoDataFrame], gpd.GeoDataFrame] | None:
    matching = config.get("matching", {})
    if not bool(matching.get("reuse_cached_best_matches", False)):
        return None
    if not _partC_rev1_sqlite_table_exists(conn, "partC_rev1_faf_ntad_best_matches"):
        _PARTC_REV1_LOG.info("Cached best-match reuse requested, but partC_rev1_faf_ntad_best_matches is missing.")
        return None

    cached_count = int(conn.execute('SELECT COUNT(*) FROM "partC_rev1_faf_ntad_best_matches"').fetchone()[0])
    if cached_count != len(links):
        _PARTC_REV1_LOG.info(
            "Cached best-match reuse skipped: cached rows=%s current links=%s.",
            cached_count,
            len(links),
        )
        return None
    expected_version = _partC_rev1_matching_algorithm_version(config)
    if not _partC_rev1_cached_matching_algorithm_is_current(conn, expected_version):
        _PARTC_REV1_LOG.info(
            "Cached best-match reuse skipped: stored matching algorithm is not %s.",
            expected_version,
        )
        return None

    _PARTC_REV1_LOG.info("Reusing cached best NTAD matches from SQLite rows=%s.", cached_count)
    best_matches = _partC_rev1_ensure_match_columns(partC_rev1_read_table_sqlite(conn, "partC_rev1_faf_ntad_best_matches"))
    attempted_matches = (
        partC_rev1_read_table_sqlite(conn, "partC_rev1_faf_ntad_match_candidates")
        if _partC_rev1_sqlite_table_exists(conn, "partC_rev1_faf_ntad_match_candidates")
        else pd.DataFrame()
    )

    projected_crs = str(config.get("ntad", {}).get("projected_crs", "EPSG:5070"))
    faf_projected = partC_rev1_prepare_faf_links(links).to_crs(projected_crs)
    matched_segments: dict[str, gpd.GeoDataFrame] = {}
    match_line_layers: list[gpd.GeoDataFrame] = []
    registry = partC_rev1_resolve_ntad_source_paths(config)

    for source_label in matching.get("source_priority", ["STRAHNET", "NHS", "NN"]):
        source_row = registry.loc[registry["source_label"].eq(source_label)].iloc[0]
        priority = int(source_row["source_priority"])
        _PARTC_REV1_LOG.info("Reading NTAD source %s for cached-match geometry outputs", source_label)
        raw_source = partC_rev1_read_ntad_source(config, source_label)
        raw_source = partC_rev1_prepare_ntad_segments(raw_source, source_label, priority)
        max_source = matching.get("max_ntad_candidates_per_source")
        if max_source is not None:
            raw_source = raw_source.head(int(max_source)).copy()
        partC_rev1_write_table_sqlite(conn, partC_rev1_ntad_sqlite_table(source_label), raw_source)
        accepted = best_matches.loc[
            best_matches["partC_rev1_ntad_match_status"].eq("matched")
            & best_matches["partC_rev1_match_dataset"].eq(source_label)
        ].copy()
        accepted_uids = set(accepted.get("partC_rev1_ntad_uid", pd.Series(dtype=str)).dropna().astype(str))
        matched_segments[source_label] = raw_source.loc[raw_source["partC_rev1_ntad_uid"].astype(str).isin(accepted_uids)].copy()
        if not accepted.empty:
            source_projected = raw_source.to_crs(projected_crs)
            match_line_layers.append(_partC_rev1_make_match_lines(accepted, faf_projected, source_projected, source_label))
        _PARTC_REV1_LOG.info("Cached source %s accepted=%s", source_label, int(len(accepted)))

    match_lines = (
        gpd.GeoDataFrame(pd.concat(match_line_layers, ignore_index=True), geometry="geometry", crs=projected_crs)
        if match_line_layers
        else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=projected_crs)
    )
    return best_matches, attempted_matches, matched_segments, match_lines


def _partC_rev1_match_links(
    links: gpd.GeoDataFrame,
    config: dict[str, Any],
    conn: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, gpd.GeoDataFrame], gpd.GeoDataFrame]:
    cached = _partC_rev1_try_load_cached_matches(links, config, conn)
    if cached is not None:
        return cached

    matching = config.get("matching", {})
    projected_crs = str(config.get("ntad", {}).get("projected_crs", "EPSG:5070"))
    faf_projected = partC_rev1_prepare_faf_links(links).to_crs(projected_crs)
    current = partC_rev1_empty_best_matches(faf_projected["partC_rev1_match_link_uid"])
    all_attempted: list[pd.DataFrame] = []
    matched_segments: dict[str, gpd.GeoDataFrame] = {}
    match_line_layers: list[gpd.GeoDataFrame] = []
    candidate_written = False

    registry = partC_rev1_resolve_ntad_source_paths(config)
    for source_label in matching.get("source_priority", ["STRAHNET", "NHS", "NN"]):
        source_row = registry.loc[registry["source_label"].eq(source_label)].iloc[0]
        priority = int(source_row["source_priority"])
        unmatched_uids = set(current.loc[~current["partC_rev1_ntad_match_status"].eq("matched"), "partC_rev1_link_uid"].astype(str))
        subset = faf_projected.loc[faf_projected["partC_rev1_match_link_uid"].astype(str).isin(unmatched_uids)].copy()
        if subset.empty:
            _PARTC_REV1_LOG.info("No unmatched links remain before source %s", source_label)
            break
        _PARTC_REV1_LOG.info("Reading NTAD source %s", source_label)
        raw_source = partC_rev1_read_ntad_source(config, source_label)
        raw_source = partC_rev1_prepare_ntad_segments(raw_source, source_label, priority)
        max_source = matching.get("max_ntad_candidates_per_source")
        if max_source is not None:
            raw_source = raw_source.head(int(max_source)).copy()
        partC_rev1_write_table_sqlite(conn, partC_rev1_ntad_sqlite_table(source_label), raw_source)
        source_projected = raw_source.to_crs(projected_crs)
        source_matches, candidates = partC_rev1_match_single_source(subset, source_projected, source_label, priority, config)
        if not candidates.empty:
            partC_rev1_write_table_sqlite(
                conn,
                "partC_rev1_faf_ntad_match_candidates",
                candidates,
                if_exists="replace" if not candidate_written else "append",
            )
            candidate_written = True
        if not source_matches.empty:
            all_attempted.append(source_matches.assign(partC_rev1_attempted_source_label=source_label))
            accepted = source_matches.loc[source_matches["partC_rev1_ntad_match_status"].eq("matched")].copy()
            current = partC_rev1_apply_source_matches(current, source_matches)
            accepted_uids = set(accepted.get("partC_rev1_ntad_uid", pd.Series(dtype=str)).dropna().astype(str))
            matched_segments[source_label] = raw_source.loc[raw_source["partC_rev1_ntad_uid"].astype(str).isin(accepted_uids)].copy()
            match_line_layers.append(_partC_rev1_make_match_lines(accepted, faf_projected, source_projected, source_label))
        else:
            matched_segments[source_label] = raw_source.iloc[0:0].copy()
        _PARTC_REV1_LOG.info(
            "Source %s complete: accepted=%s remaining_unmatched=%s",
            source_label,
            int(current["partC_rev1_match_dataset"].eq(source_label).sum()),
            int((~current["partC_rev1_ntad_match_status"].eq("matched")).sum()),
        )
    attempted = pd.concat(all_attempted, ignore_index=True) if all_attempted else pd.DataFrame()
    if not candidate_written:
        partC_rev1_write_table_sqlite(conn, "partC_rev1_faf_ntad_match_candidates", pd.DataFrame())
    match_lines = (
        gpd.GeoDataFrame(pd.concat(match_line_layers, ignore_index=True), geometry="geometry", crs=projected_crs)
        if match_line_layers
        else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=projected_crs)
    )
    return _partC_rev1_ensure_match_columns(current), attempted, matched_segments, match_lines


def _partC_rev1_summary_quantiles(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in df.dropna(subset=[value_col]).groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = pd.to_numeric(group[value_col], errors="coerce").dropna()
        if values.empty:
            continue
        row = {column: key for column, key in zip(group_cols, keys)}
        row.update(
            {
                "metric": value_col,
                "count": int(values.count()),
                "p10": float(values.quantile(0.10)),
                "p50": float(values.quantile(0.50)),
                "p90": float(values.quantile(0.90)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


_PARTC_REV1_COLUMN_DOCS: dict[str, tuple[str, str, str]] = {
    "partC_rev1_link_uid": ("id", "Stable Prompt 2 physical-link identifier.", "Assigned during Prompt 2 link cleaning."),
    "physical_link_id": ("id", "Original FAF physical link ID preserved from the raw network.", "Copied from the configured link ID column."),
    "directed_arc_id": ("id", "Directed arc identifier used by assignment-ready graph tables.", "Assigned during Prompt 2 AB/BA directed-arc construction."),
    "partC_rev1_ntad_match_status": ("category", "NTAD match state for the FAF link or directed arc.", "Prompt 2.5 matcher; values include matched, unassigned, and ambiguous."),
    "partC_rev1_match_dataset": ("category", "Selected NTAD source for the match.", "Prompt 2.5 source-priority matcher using STRAHNET, NHS, then NN."),
    "partC_rev1_match_method": ("category", "Rule that selected the NTAD segment.", "Prompt 2.5 buffer-geometry matcher; route/name/sign labels only rank geometry-valid candidates."),
    "partC_rev1_match_close_competitor": ("boolean", "True when the second-best geometry-valid candidate had a very similar score.", "Computed from the best and second-best match score ratio after selection."),
    "partC_rev1_faf_route_key": ("category", "FAF route key used by legacy route/name geometry scoring.", "Normalized from FAF sign or route columns."),
    "partC_rev1_ntad_route_key": ("category", "NTAD route key used by legacy route/name geometry scoring.", "Normalized from NTAD SIGN1 or SIGNT1/SIGNN1 columns."),
    "partC_rev1_faf_normalized_sign_key": ("category", "Canonical FAF signed-route key from explicit route type and number columns.", "Normalized from Rte_Type/route_type/partC_rev1_route_type plus Rte_Number/route_number/partC_rev1_route_number."),
    "partC_rev1_ntad_normalized_sign_key": ("category", "Canonical NTAD signed-route key from explicit SIGNT1 and SIGNN1 columns.", "Normalized from SIGNT1/SIGNN1 with case-insensitive lookup."),
    "partC_rev1_normalized_sign_match": ("boolean", "True when FAF and selected NTAD segment have the same canonical signed-route key.", "Diagnostic only in buffer_geo_v3; sign equality never bypasses the geometry gate."),
    "partC_rev1_normalized_sign_global_fallback": ("boolean", "Compatibility field retained from normalized_sign_v2; always false for buffer_geo_v3.", "Prompt 2.5 buffer_geo_v3 retired global same-sign fallback."),
    "partC_rev1_normalized_sign_selected_distance_m": ("m", "Compatibility field retained from normalized_sign_v2.", "Not used by buffer_geo_v3; use partC_rev1_geo_direct_distance_m instead."),
    "partC_rev1_geo_buffer_radius_m": ("m", "Line-buffer radius used for the NTAD/FAF geometry gate.", "Configured by matching.geo_buffer_radius_m or matching.geo_buffer_radius_miles."),
    "partC_rev1_geo_buffer_intersects": ("boolean", "True when FAF and NTAD line buffers intersect.", "Computed in projected CRS before any label/name scoring can accept a match."),
    "partC_rev1_geo_buffer_overlap_score": ("unitless", "Area-overlap score between FAF and NTAD line buffers.", "Intersection area divided by the smaller buffer area."),
    "partC_rev1_geo_direct_distance_m": ("m", "Minimum direct distance between FAF and NTAD geometries.", "Computed in projected CRS for geometry diagnostics and candidate ordering."),
    "partC_rev1_faf_stfips": ("category", "FAF state FIPS used as match diagnostic.", "Normalized from FAF STFIPS/state FIPS variants."),
    "partC_rev1_faf_ctfips": ("category", "FAF county FIPS used as match diagnostic.", "Normalized from FAF CTFIPS/county FIPS variants."),
    "partC_rev1_ntad_stfips": ("category", "NTAD state FIPS used as match diagnostic.", "Normalized from NTAD STFIPS/state FIPS variants."),
    "partC_rev1_ntad_ctfips": ("category", "NTAD county FIPS used as match diagnostic.", "Normalized from NTAD CTFIPS/county FIPS variants."),
    "partC_rev1_match_fips_score": ("unitless", "Administrative tie-break score for state/county agreement.", "1.0 for same county, 0.5 for same state, 0.0 otherwise or missing."),
    "partC_rev1_ntad_aadt_total": ("vehicles/day", "Total AADT attached to the matched link.", "NTAD total AADT if positive; otherwise FAF-class default."),
    "partC_rev1_ntad_aadt_combination": ("trucks/day", "Combination-truck AADT from NTAD when available.", "Copied from AADT_COM or equivalent NTAD field."),
    "partC_rev1_ntad_aadt_single": ("trucks/day", "Single-unit-truck AADT from NTAD when available.", "Copied from AADT_SINGL or equivalent NTAD field."),
    "partC_rev1_ntad_aadt_truck": ("trucks/day", "Total truck AADT used for background traffic.", "Combination plus single-unit AADT when present; otherwise total AADT times default truck ratio."),
    "partC_rev1_aadt_source": ("category", "Source used for total AADT.", "ntad_aadt when NTAD has positive total AADT; otherwise default_by_class."),
    "partC_rev1_aadt_truck_source": ("category", "Source used for truck AADT.", "ntad_truck_aadt_fields when truck fields exist; otherwise default_ratio_from_total_aadt."),
    "partC_rev1_fftt_min": ("minutes", "Final free-flow travel time used by assignment.", "Existing FAF free-flow time when positive; otherwise computed from speed fields, NTAD speed, class default speed, or one-minute fallback."),
    "partC_rev1_lanes_dir": ("lanes", "Directional lane count used for capacity.", "AB_Lanes or BA_Lanes by arc direction; then Prompt 2 default lanes; then one-lane fallback."),
    "partC_rev1_capacity_vphpl": ("vehicles/hour/lane", "Capacity per hour per lane before directional lane multiplication.", "Freeway speed formula for FAF classes 11-13; otherwise FAF-class default capacity."),
    "partC_rev1_vehicle_capacity_vph_dir": ("vehicles/hour", "Directional vehicle capacity after lanes and PHF.", "partC_rev1_lanes_dir times partC_rev1_capacity_vphpl times PHF."),
    "partC_rev1_aadt_total_dir": ("vehicles/day", "Directional total AADT.", "partC_rev1_ntad_aadt_total times direction_split."),
    "partC_rev1_aadt_truck_dir": ("trucks/day", "Directional truck AADT.", "partC_rev1_ntad_aadt_truck times direction_split."),
    "partC_rev1_background_total_vph_dir": ("vehicles/hour", "Directional peak-hour background traffic volume.", "Directional total AADT times k_factor times d_factor."),
    "partC_rev1_background_truck_vph_dir": ("trucks/hour", "Directional peak-hour background truck volume.", "Directional truck AADT times k_factor times d_factor."),
    "partC_rev1_background_nontruck_vph_dir": ("vehicles/hour", "Directional peak-hour non-truck background volume.", "Max of total background volume minus truck background volume and zero."),
    "partC_rev1_remaining_truck_vph_dir": ("trucks/hour", "Remaining directional truck capacity after background traffic.", "Vehicle capacity minus non-truck PCE and truck PCE demand, converted to trucks/hour with minimum capacity floor."),
    "partC_rev1_remaining_truck_aadt_dir": ("trucks/day", "Remaining directional daily truck capacity.", "Remaining truck vph divided by k_factor times d_factor."),
    "partC_rev1_capacity_annual_ktons_dir": ("thousand tons/year", "Directional annual freight capacity available to FAF assignment.", "Remaining truck AADT times 365 times payload_tons_per_truck divided by 1000."),
    "partC_rev1_background_truck_annual_ktons_dir": ("thousand tons/year", "Directional annual background truck tonnage.", "Directional truck AADT times 365 times payload_tons_per_truck divided by 1000."),
}


def _partC_rev1_unit_for_column(column: str) -> str:
    if column in _PARTC_REV1_COLUMN_DOCS:
        return _PARTC_REV1_COLUMN_DOCS[column][0]
    lower = column.lower()
    if lower.endswith("_m") or "_distance_m" in lower:
        return "m"
    if lower.endswith("_mph"):
        return "mph"
    if "aadt" in lower:
        return "vehicles/day or trucks/day"
    if "vphpl" in lower:
        return "vehicles/hour/lane"
    if "vph" in lower:
        return "vehicles/hour or trucks/hour"
    if "annual_ktons" in lower:
        return "thousand tons/year"
    if lower.endswith("_min") or "time" in lower:
        return "minutes"
    if "score" in lower or "ratio" in lower:
        return "unitless"
    if lower.endswith("_id") or lower.endswith("_uid") or "objectid" in lower:
        return "id"
    return ""


def _partC_rev1_meaning_for_column(column: str) -> tuple[str, str]:
    if column in _PARTC_REV1_COLUMN_DOCS:
        return _PARTC_REV1_COLUMN_DOCS[column][1], _PARTC_REV1_COLUMN_DOCS[column][2]
    lower = column.lower()
    if lower.startswith("partc_rev1_before_match_"):
        original = column.replace("partC_rev1_before_match_", "", 1)
        return f"Backup copy of {original} before Prompt 2.5 enrichment.", "Copied before any Prompt 2.5 value changes."
    if lower.startswith("partc_rev1_match_"):
        return "NTAD match diagnostic score or selection attribute.", "Computed by Prompt 2.5 road matcher."
    if lower.startswith("partc_rev1_ntad_"):
        return "Attribute copied or normalized from the selected NTAD segment.", "Copied from STRAHNET, NHS, or NN after source-priority matching."
    if lower.startswith("partc_rev1_"):
        return "Prompt-generated normalized, diagnostic, or enriched field.", "Generated by the partC_rev1 preprocessing or enrichment pipeline."
    return "Source or derived network attribute preserved for inspection.", "Copied from the source layer or generated by prior Prompt 0-2 preprocessing."


def _partC_rev1_column_dictionary_rows(table_or_layer: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        meaning, generated_by = _partC_rev1_meaning_for_column(str(column))
        rows.append(
            {
                "table_or_layer": table_or_layer,
                "column": column,
                "type": str(df[column].dtype) if column in df.columns else "",
                "unit": _partC_rev1_unit_for_column(str(column)),
                "meaning": meaning,
                "generated_by": generated_by,
                "default_or_formula": _PARTC_REV1_COLUMN_DOCS.get(str(column), ("", "", "As described by generated_by."))[2],
            }
        )
    return rows


def _partC_rev1_write_preprocess_steps_summary(
    config: dict[str, Any],
    *,
    links_enriched: gpd.GeoDataFrame,
    arcs_enriched: gpd.GeoDataFrame,
    best_matches: pd.DataFrame,
    attempted_matches: pd.DataFrame,
    matched_segments: dict[str, gpd.GeoDataFrame],
    match_lines: gpd.GeoDataFrame,
    enriched_gpkg_path: Path,
    cache_path: Path,
) -> None:
    candidate_schema = pd.DataFrame(columns=["partC_rev1_link_uid", "candidate_rank", *[c for c in best_matches.columns if c != "partC_rev1_link_uid"]])
    tables: dict[str, pd.DataFrame] = {
        "sqlite:partC_rev1_faf_ntad_best_matches": best_matches,
        "sqlite:partC_rev1_faf_ntad_match_candidates": candidate_schema,
        "sqlite:partC_rev1_directed_arc_ntad_matches": arcs_enriched[
            [
                column
                for column in arcs_enriched.columns
                if column.startswith("partC_rev1_match_")
                or column.startswith("partC_rev1_ntad_")
                or column.startswith("partC_rev1_faf_")
                or column.startswith("partC_rev1_geo_")
                or column.startswith("partC_rev1_normalized_sign_")
                or column in {"directed_arc_id", "partC_rev1_link_uid"}
            ]
        ],
        "sqlite:partC_rev1_capacity_enriched_links": links_enriched,
        "sqlite:partC_rev1_capacity_enriched_directed_arcs": arcs_enriched,
        "gpkg:partC_rev1_clean_physical_links_enriched": links_enriched,
        "gpkg:partC_rev1_directed_arcs_enriched": arcs_enriched,
        "gpkg:partC_rev1_faf_ntad_match_lines": match_lines,
    }
    for label, gdf in matched_segments.items():
        tables[f"gpkg:partC_rev1_ntad_{label.lower()}_segments_matched"] = gdf
    if not attempted_matches.empty:
        tables["diagnostic:partC_rev1_source_attempted_matches"] = attempted_matches

    dictionary_rows: list[dict[str, Any]] = []
    for table_or_layer, df in tables.items():
        dictionary_rows.extend(_partC_rev1_column_dictionary_rows(table_or_layer, df))
    dictionary = pd.DataFrame(dictionary_rows)

    lines = [
        "# partC_rev1 Preprocess Steps Summary",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Prompt Stage Summary",
        "",
        "| Stage | Purpose | Main outputs |",
        "| --- | --- | --- |",
        "| Prompt 0 | Establish project paths, config defaults, and raw data inventory. | Config files, inspection reports, source-path decisions. |",
        "| Prompt 1 | Inspect FAF/NTAD/ground-truth inputs and identify usable layers and fields. | Data inspection reports and source-layer metadata. |",
        "| Prompt 2 | Clean FAF network topology, preserve source attributes, build physical links and directed arcs. | `partC_rev1_clean_network.gpkg`, `partC_rev1_network.sqlite`, directed arcs, duplicate/topology reports. |",
        "| Prompt 2.5 | Enrich Prompt 2 graph with NTAD source-priority matches, AADT, FFTT, background traffic, and annual freight capacity. | Enriched SQLite tables, `partC_rev1_clean_network_enriched.gpkg`, QGIS guide, capacity summary. |",
        "",
        "## Prompt 2.5 Matching Rule",
        "",
        f"- Matching algorithm version: `{_partC_rev1_matching_algorithm_version(config)}`.",
        "- Source priority remains `STRAHNET -> NHS -> NN`; once a link is matched by a higher-priority source, lower sources do not overwrite it.",
        "- Every accepted NTAD candidate must pass the buffer-geometry gate: FAF and NTAD centerlines are each buffered and the buffers must intersect.",
        "- The default buffer radius is 0.08 miles on each line, evaluated in the configured projected CRS.",
        "- Normalized signed routes and road names are diagnostics/ranking bonuses only; they never bypass the geometry gate.",
        "- `STFIPS` and `CTFIPS` agreement is a tie-break score only, not a hard filter.",
        "",
        "## Key Artifacts",
        "",
        f"- SQLite cache: `{cache_path}`",
        f"- Enriched GeoPackage: `{enriched_gpkg_path}`",
        f"- Enriched physical links: {len(links_enriched):,}",
        f"- Enriched directed arcs: {len(arcs_enriched):,}",
        f"- Best-match rows: {len(best_matches):,}",
        "",
        "## Capacity And AADT Defaults",
        "",
        "- `partC_rev1_ntad_aadt_total` uses positive NTAD total AADT when present; otherwise it falls back to `capacity_enrichment.default_aadt_by_faf_class`.",
        "- `partC_rev1_ntad_aadt_combination` and `partC_rev1_ntad_aadt_single` copy NTAD combination and single-unit truck AADT fields when available.",
        "- `partC_rev1_ntad_aadt_truck` is combination plus single-unit truck AADT when either field exists; otherwise it is total AADT times `default_truck_ratio`.",
        "- Directional AADT uses `direction_split`; peak-hour background traffic uses `k_factor * d_factor`.",
        "- Annual kton capacity is remaining directional truck AADT times 365 times `payload_tons_per_truck`, divided by 1000.",
        "",
        "## Column Dictionary",
        "",
    ]
    if dictionary.empty:
        lines.append("(No columns documented.)")
    else:
        columns = ["table_or_layer", "column", "type", "unit", "meaning", "generated_by", "default_or_formula"]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in dictionary[columns].itertuples(index=False):
            lines.append("| " + " | ".join(_partC_rev1_escape_md(value) for value in row) + " |")

    partC_rev1_output_file(config, "partC_rev1_preprocess_steps_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _partC_rev1_write_reports(
    config: dict[str, Any],
    *,
    inspection: pd.DataFrame,
    recovery_report: pd.DataFrame,
    links_enriched: gpd.GeoDataFrame,
    arcs_enriched: gpd.GeoDataFrame,
    best_matches: pd.DataFrame,
    attempted_matches: pd.DataFrame,
    matched_segments: dict[str, gpd.GeoDataFrame],
    match_lines: gpd.GeoDataFrame,
    inherited: pd.DataFrame,
    conflicts: pd.DataFrame,
    preservation: pd.DataFrame,
    cache_path: Path,
    enriched_gpkg_path: Path,
) -> pd.DataFrame:
    _partC_rev1_write_csv(config, inspection, "partC_rev1_ntad_three_source_layer_column_inspection.csv")
    _partC_rev1_write_markdown_table(
        inspection,
        partC_rev1_output_file(config, "partC_rev1_ntad_three_source_layer_column_inspection.md"),
        "partC_rev1 NTAD Three Source Layer Column Inspection",
    )
    _partC_rev1_write_csv(config, recovery_report, "partC_rev1_faf_strahnet_recovery_report.csv")
    _partC_rev1_write_csv(config, preservation, "partC_rev1_before_match_column_preservation_report.csv")

    quality = pd.concat(
        [
            _partC_rev1_summary_quantiles(links_enriched, "partC_rev1_match_score", ["partC_rev1_match_dataset", "partC_rev1_match_method"]),
            _partC_rev1_summary_quantiles(links_enriched, "partC_rev1_match_angle_diff_deg", ["partC_rev1_match_dataset", "partC_rev1_match_method"]),
            _partC_rev1_summary_quantiles(links_enriched, "partC_rev1_match_min_distance_m", ["partC_rev1_match_dataset", "partC_rev1_match_method"]),
            _partC_rev1_summary_quantiles(links_enriched, "partC_rev1_geo_direct_distance_m", ["partC_rev1_match_dataset", "partC_rev1_match_method"]),
            _partC_rev1_summary_quantiles(links_enriched, "partC_rev1_geo_buffer_overlap_score", ["partC_rev1_match_dataset", "partC_rev1_match_method"]),
            _partC_rev1_summary_quantiles(links_enriched, "partC_rev1_match_fips_score", ["partC_rev1_match_dataset", "partC_rev1_match_method"]),
        ],
        ignore_index=True,
    )
    _partC_rev1_write_csv(config, quality, "partC_rev1_ntad_match_quality_summary.csv")

    no_route_name = links_enriched.loc[
        links_enriched["partC_rev1_ntad_match_status"].ne("matched")
        & links_enriched.get("partC_rev1_faf_route_key", pd.Series("", index=links_enriched.index)).fillna("").astype(str).eq("")
        & links_enriched.get("partC_rev1_faf_road_name_norm", pd.Series("", index=links_enriched.index)).fillna("").astype(str).eq("")
    ].copy()
    no_good = links_enriched.loc[links_enriched["partC_rev1_ntad_match_status"].ne("matched")].copy()
    ambiguous = attempted_matches.loc[attempted_matches.get("partC_rev1_ntad_match_status", pd.Series(dtype=str)).eq("ambiguous")].copy()
    spatial_failed = attempted_matches.loc[
        attempted_matches.get("partC_rev1_match_method", pd.Series(dtype=str)).fillna("").eq("spatial_only")
        & attempted_matches.get("partC_rev1_ntad_match_status", pd.Series(dtype=str)).ne("matched")
    ].copy()
    _partC_rev1_write_csv(config, no_route_name.drop(columns="geometry", errors="ignore"), "partC_rev1_unassigned_no_road_name_links.csv")
    _partC_rev1_write_csv(config, no_good.drop(columns="geometry", errors="ignore"), "partC_rev1_unassigned_no_good_match_links.csv")
    _partC_rev1_write_csv(config, spatial_failed.drop(columns="geometry", errors="ignore"), "partC_rev1_unassigned_spatial_only_failed_links.csv")
    _partC_rev1_write_csv(config, ambiguous.drop(columns="geometry", errors="ignore"), "partC_rev1_ambiguous_match_links.csv")
    _partC_rev1_write_csv(config, inherited, "partC_rev1_opposite_direction_inherited_matches.csv")
    _partC_rev1_write_csv(config, conflicts, "partC_rev1_match_conflicts.csv")
    _partC_rev1_write_csv(
        config,
        links_enriched.loc[links_enriched.get("partC_rev1_aadt_source", pd.Series(index=links_enriched.index)).eq("default_by_class")].drop(
            columns="geometry", errors="ignore"
        ),
        "partC_rev1_capacity_defaulted_links.csv",
    )
    _partC_rev1_write_csv(
        config,
        arcs_enriched.loc[~arcs_enriched["partC_rev1_fftt_source"].astype(str).str.startswith("existing_", na=False)].drop(
            columns="geometry", errors="ignore"
        ),
        "partC_rev1_fftt_filled_links.csv",
    )

    summary = pd.DataFrame(
        [
            {
                "faf_cleaned_links": len(links_enriched),
                "directed_arcs": len(arcs_enriched),
                "links_with_recovered_faf_strahnet": int(links_enriched["partC_rev1_faf_strahnet_nonempty"].sum()),
                "matched_strahnet": int(links_enriched["partC_rev1_match_dataset"].eq("STRAHNET").sum()),
                "matched_nhs": int(links_enriched["partC_rev1_match_dataset"].eq("NHS").sum()),
                "matched_nn": int(links_enriched["partC_rev1_match_dataset"].eq("NN").sum()),
                "matched_route_geometry": int(links_enriched["partC_rev1_match_method"].eq("route_geometry").sum()),
                "matched_name_geometry": int(links_enriched["partC_rev1_match_method"].eq("name_geometry").sum()),
                "matched_spatial_only": int(links_enriched["partC_rev1_match_method"].eq("spatial_only").sum()),
                "matched_buffer_geometry_total": int(links_enriched["partC_rev1_geo_buffer_intersects"].sum()),
                "matched_same_county_fips": int(pd.to_numeric(links_enriched["partC_rev1_match_fips_score"], errors="coerce").eq(1.0).sum()),
                "matched_same_state_fips_only": int(pd.to_numeric(links_enriched["partC_rev1_match_fips_score"], errors="coerce").eq(0.5).sum()),
                "matched_close_competitor": int(links_enriched["partC_rev1_match_close_competitor"].sum()),
                "matched_normalized_sign_nearby": int(links_enriched["partC_rev1_match_method"].eq("normalized_sign_nearby").sum()),
                "matched_normalized_sign_global_nearest": int(links_enriched["partC_rev1_match_method"].eq("normalized_sign_global_nearest").sum()),
                "matched_normalized_sign_total": int(links_enriched["partC_rev1_normalized_sign_match"].sum()),
                "opposite_direction_inherited": len(inherited),
                "unassigned_no_road_name": len(no_route_name),
                "unassigned_no_good_match": len(no_good),
                "ambiguous": len(ambiguous),
                "aadt_coverage": int(pd.to_numeric(links_enriched["partC_rev1_ntad_aadt_total"], errors="coerce").notna().sum()),
                "truck_aadt_coverage": int(pd.to_numeric(links_enriched["partC_rev1_ntad_aadt_truck"], errors="coerce").notna().sum()),
                "links_using_default_aadt": int(links_enriched["partC_rev1_aadt_source"].eq("default_by_class").sum()),
                "enriched_gpkg_path": str(enriched_gpkg_path),
                "before_match_columns_preserved": int(len(preservation)),
            }
        ]
    )
    _partC_rev1_write_csv(config, summary, "partC_rev1_capacity_enrichment_summary.csv")

    lines = [
        "# partC_rev1 Capacity Enrichment Summary",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    for key, value in summary.iloc[0].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## FFTT Source Distribution",
            arcs_enriched["partC_rev1_fftt_source"].value_counts(dropna=False).to_string(),
            "",
            "## Capacity Source Distribution",
            arcs_enriched["partC_rev1_capacity_source"].value_counts(dropna=False).head(50).to_string(),
            "",
            "## QGIS",
            f"Open `{enriched_gpkg_path}` and inspect `partC_rev1_clean_physical_links_enriched`, `partC_rev1_directed_arcs_enriched`, and `partC_rev1_faf_ntad_match_lines`.",
        ]
    )
    partC_rev1_output_file(config, "partC_rev1_capacity_enrichment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    partC_rev1_output_file(config, "partC_rev1_qgis_enriched_layer_guide.md").write_text(
        "\n".join(
            [
                "# partC_rev1 Enriched QGIS Layer Guide",
                "",
                f"GeoPackage: `{enriched_gpkg_path}`",
                "",
                "- `partC_rev1_clean_physical_links_enriched`: color by `partC_rev1_match_dataset` or `partC_rev1_match_method`.",
                "- `partC_rev1_clean_physical_links_enriched`: inspect `partC_rev1_geo_buffer_intersects`, `partC_rev1_geo_buffer_overlap_score`, and `partC_rev1_geo_direct_distance_m` for geometry-match diagnostics.",
                "- `partC_rev1_clean_physical_links_enriched`: inspect `partC_rev1_match_fips_score`, `partC_rev1_faf_ctfips`, and `partC_rev1_ntad_ctfips` for state/county agreement checks.",
                "- `partC_rev1_directed_arcs_enriched`: inspect `partC_rev1_fftt_min`, `partC_rev1_capacity_annual_ktons_dir`, and background traffic fields.",
                "- `partC_rev1_faf_ntad_match_lines`: inspect selected match offsets between FAF and NTAD midpoints.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _partC_rev1_write_preprocess_steps_summary(
        config,
        links_enriched=links_enriched,
        arcs_enriched=arcs_enriched,
        best_matches=best_matches,
        attempted_matches=attempted_matches,
        matched_segments=matched_segments,
        match_lines=match_lines,
        enriched_gpkg_path=enriched_gpkg_path,
        cache_path=cache_path,
    )
    return summary


def _partC_rev1_write_enriched_gpkg(
    config: dict[str, Any],
    links: gpd.GeoDataFrame,
    arcs: gpd.GeoDataFrame,
    matched_segments: dict[str, gpd.GeoDataFrame],
    match_lines: gpd.GeoDataFrame,
) -> Path:
    path = _partC_rev1_enriched_gpkg_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _PARTC_REV1_LOG.info("Removing previous enriched GeoPackage before rewrite: %s", path)
        path.unlink()
    _partC_rev1_write_gpkg_layer(links, path, "partC_rev1_clean_physical_links_enriched")
    _partC_rev1_write_gpkg_layer(arcs, path, "partC_rev1_directed_arcs_enriched")
    for label, layer_name in [
        ("STRAHNET", "partC_rev1_ntad_strahnet_segments_matched"),
        ("NHS", "partC_rev1_ntad_nhs_segments_matched"),
        ("NN", "partC_rev1_ntad_nn_segments_matched"),
    ]:
        gdf = matched_segments.get(label)
        if gdf is None:
            gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=links.crs)
        _partC_rev1_write_gpkg_layer(gdf, path, layer_name)
    _partC_rev1_write_gpkg_layer(match_lines, path, "partC_rev1_faf_ntad_match_lines")
    return path


def partC_rev1_run_enrich_capacity(config_path: str, *, confirm_full_run: bool = False) -> dict[str, Any]:
    config = load_partC_rev1_config(config_path)
    output_dir = config.get("outputs", {}).get("output_dir", config.get("project", {}).get("output_dir", "partC_rev1_outputs"))
    partC_rev1_setup_logging(output_dir)
    _partC_REV1_start = time.perf_counter()
    _PARTC_REV1_LOG.info("Starting Prompt 2.5 capacity enrichment with config=%s confirm_full_run=%s", config_path, confirm_full_run)
    _partC_rev1_validate_full_run_inputs(config, confirm_full_run)
    if not bool(config.get("matching", {}).get("enabled", True)):
        raise ValueError("matching.enabled=false is not supported for Prompt 2.5 enrichment.")

    inspection = partC_rev1_inspect_ntad_sources(config)
    layers = _partC_rev1_layer_names(config)
    clean_gpkg = _partC_rev1_clean_gpkg_path(config)
    _PARTC_REV1_LOG.info("Reading Prompt 2 cleaned links and arcs from %s", clean_gpkg)
    links = gpd.read_file(clean_gpkg, layer=layers["links"])
    arcs = gpd.read_file(clean_gpkg, layer=layers["arcs"])
    max_links = config.get("matching", {}).get("max_faf_links")
    if max_links is not None:
        links = links.head(int(max_links)).copy()
        keep = set(links["partC_rev1_link_uid"].astype(str))
        arcs = arcs.loc[arcs["partC_rev1_link_uid"].astype(str).isin(keep)].copy()

    links, link_preservation = partC_rev1_preserve_existing_columns(
        links,
        ["STRAHNET", "NHS", "NHFN", "AB_FreeFlowTime", "BA_FreeFlowTime", "AB_Lanes", "BA_Lanes", "partC_rev1_ground_truth_flow"],
        table_name="partC_rev1_clean_physical_links_enriched",
        reason="Prompt 2.5 preserves Prompt 2 columns before matching and enrichment.",
    )
    links, recovery_report = _partC_rev1_recover_strahnet(links, config)

    cache_path = _partC_rev1_cache_path(config)
    with partC_rev1_connect_sqlite(cache_path) as conn:
        partC_rev1_write_table_sqlite(conn, "partC_rev1_ntad_three_source_column_inspection", inspection)
        partC_rev1_write_table_sqlite(conn, "partC_rev1_faf_strahnet_recovery", recovery_report)
        best_matches, attempted_matches, matched_segments, match_lines = _partC_rev1_match_links(links, config, conn)

        links_keyed = links.copy()
        links_keyed["_partC_rev1_link_uid_key"] = links_keyed["partC_rev1_link_uid"].astype(str)
        links_keyed = partC_rev1_prepare_faf_links(links_keyed)
        links_enriched = links_keyed.merge(
            best_matches,
            left_on="_partC_rev1_link_uid_key",
            right_on="partC_rev1_link_uid",
            how="left",
            suffixes=("", "_match"),
        )
        if "partC_rev1_link_uid_match" in links_enriched.columns:
            links_enriched = links_enriched.drop(columns=["partC_rev1_link_uid_match"])
        links_enriched = links_enriched.drop(columns=["_partC_rev1_link_uid_key"], errors="ignore")
        links_enriched = _partC_rev1_ensure_match_columns(links_enriched)
        links_enriched = partC_rev1_attach_link_aadt_defaults(links_enriched, config)
        links_enriched = gpd.GeoDataFrame(links_enriched, geometry="geometry", crs=links.crs)

        arcs_enriched, arc_preservation, capacity_report = partC_rev1_enrich_directed_arcs_capacity(arcs, links_enriched, config)
        arcs_enriched, inherited, conflicts = partC_rev1_inherit_opposite_direction_matches(arcs_enriched)
        arcs_enriched = gpd.GeoDataFrame(arcs_enriched, geometry="geometry", crs=arcs.crs)

        preservation = pd.concat([link_preservation, arc_preservation], ignore_index=True)
        _PARTC_REV1_LOG.info("Writing enriched SQLite table partC_rev1_faf_ntad_best_matches rows=%s", len(best_matches))
        partC_rev1_write_table_sqlite(conn, "partC_rev1_faf_ntad_best_matches", best_matches)
        _PARTC_REV1_LOG.info("Writing enriched SQLite table partC_rev1_directed_arc_ntad_matches rows=%s", len(arcs_enriched))
        partC_rev1_write_table_sqlite(
            conn,
            "partC_rev1_directed_arc_ntad_matches",
            arcs_enriched[
                [
                    column
                    for column in arcs_enriched.columns
                    if column.startswith("partC_rev1_match_")
                    or column.startswith("partC_rev1_ntad_")
                    or column.startswith("partC_rev1_faf_")
                    or column.startswith("partC_rev1_geo_")
                    or column.startswith("partC_rev1_normalized_sign_")
                    or column in {"directed_arc_id", "partC_rev1_link_uid"}
                ]
            ],
        )
        _PARTC_REV1_LOG.info("Writing enriched SQLite table partC_rev1_capacity_enriched_links rows=%s", len(links_enriched))
        partC_rev1_write_table_sqlite(conn, "partC_rev1_capacity_enriched_links", links_enriched)
        _PARTC_REV1_LOG.info("Writing enriched SQLite table partC_rev1_capacity_enriched_directed_arcs rows=%s", len(arcs_enriched))
        partC_rev1_write_table_sqlite(conn, "partC_rev1_capacity_enriched_directed_arcs", arcs_enriched)
        _PARTC_REV1_LOG.info("Writing enriched SQLite table partC_rev1_directed_arcs_enriched rows=%s", len(arcs_enriched))
        partC_rev1_write_table_sqlite(conn, "partC_rev1_directed_arcs_enriched", arcs_enriched)
        _PARTC_REV1_LOG.info("Writing enriched SQLite table partC_rev1_before_match_column_preservation rows=%s", len(preservation))
        partC_rev1_write_table_sqlite(conn, "partC_rev1_before_match_column_preservation", preservation)
        metadata = pd.DataFrame(
            [
                {
                    "key": "partC_rev1_capacity_enrichment",
                    "value": json.dumps(
                        {
                            "config_path": config_path,
                            "confirm_full_run": confirm_full_run,
                            "runtime_seconds": time.perf_counter() - _partC_REV1_start,
                            "matching_algorithm_version": _partC_rev1_matching_algorithm_version(config),
                            "capacity_report": capacity_report.to_dict(orient="records"),
                        },
                        sort_keys=True,
                    ),
                }
            ]
        )
        _PARTC_REV1_LOG.info("Writing enriched SQLite metadata tables")
        partC_rev1_write_table_sqlite(conn, "partC_rev1_capacity_enrichment_metadata", metadata)
        partC_rev1_write_metadata(conn, "partC_rev1_capacity_enrichment_completed_utc", datetime.now(timezone.utc).isoformat())
        partC_rev1_write_metadata(conn, "partC_rev1_matching_algorithm_version", _partC_rev1_matching_algorithm_version(config))

    _PARTC_REV1_LOG.info("Writing enriched GeoPackage outputs")
    enriched_gpkg = _partC_rev1_write_enriched_gpkg(config, links_enriched, arcs_enriched, matched_segments, match_lines)
    _PARTC_REV1_LOG.info("Writing capacity enrichment reports")
    summary = _partC_rev1_write_reports(
        config,
        inspection=inspection,
        recovery_report=recovery_report,
        links_enriched=links_enriched,
        arcs_enriched=arcs_enriched,
        best_matches=best_matches,
        attempted_matches=attempted_matches,
        matched_segments=matched_segments,
        match_lines=match_lines,
        inherited=inherited,
        conflicts=conflicts,
        preservation=preservation,
        cache_path=cache_path,
        enriched_gpkg_path=enriched_gpkg,
    )
    runtime = time.perf_counter() - _partC_REV1_start
    _PARTC_REV1_LOG.info("Prompt 2.5 capacity enrichment complete in %.3f seconds. Enriched GPKG=%s", runtime, enriched_gpkg)
    return {
        "summary": summary.iloc[0].to_dict(),
        "enriched_gpkg_path": str(enriched_gpkg),
        "cache_path": str(cache_path),
        "runtime_seconds": runtime,
    }
