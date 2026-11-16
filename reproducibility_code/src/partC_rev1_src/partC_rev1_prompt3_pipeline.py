"""Prompt 3A/3B backbone extraction and physical-node partition pipeline."""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkb

from .partC_rev1_backbone_expansion import partC_rev1_expand_backbone
from .partC_rev1_backbone_seed import partC_rev1_extract_seed_backbone
from .partC_rev1_cache import partC_rev1_connect_sqlite, partC_rev1_read_table_sqlite, partC_rev1_write_metadata, partC_rev1_write_table_sqlite
from .partC_rev1_config import load_partC_rev1_config
from .partC_rev1_ids import partC_rev1_stable_id
from .partC_rev1_io import partC_rev1_write_table
from .partC_rev1_logging import partC_rev1_setup_logging
from .partC_rev1_partition_igraph import partC_rev1_assign_physical_arc_regions, partC_rev1_run_leiden_partition
from .partC_rev1_partition_stability import partC_rev1_compute_partition_stability, partC_rev1_nonbackbone_graph_hash
from .partC_rev1_paths import partC_rev1_output_file, partC_rev1_resolve_path


_PARTC_REV1_LOG = logging.getLogger("partC_rev1")


def _partC_rev1_cache_path(config: dict[str, Any]) -> Path:
    return partC_rev1_resolve_path(str(config.get("preprocess", {}).get("cache_sqlite", "partC_rev1_cache/partC_rev1_network.sqlite")))


def _partC_rev1_clean_gpkg_path(config: dict[str, Any]) -> Path:
    return partC_rev1_resolve_path(
        str(config.get("clean_geodatabase", {}).get("output_gpkg_path", "partC_rev1_cache/partC_rev1_clean_network.gpkg"))
    )


def _partC_rev1_enriched_gpkg_path(config: dict[str, Any]) -> Path:
    return _partC_rev1_clean_gpkg_path(config).with_name("partC_rev1_clean_network_enriched.gpkg")


def _partC_rev1_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,)).fetchone()
    return row is not None


def _partC_rev1_sqlite_geometry_crs(config: dict[str, Any]) -> str:
    if bool(config.get("testing", {}).get("use_toy_data", True)):
        return str(config.get("preprocess", {}).get("snap_projected_crs", "EPSG:3857"))
    return str(config.get("inspection_defaults", {}).get("gdb_crs", "EPSG:4269"))


def _partC_rev1_from_wkb(df: pd.DataFrame, *, crs: str | None = None) -> gpd.GeoDataFrame:
    out = df.copy()
    geom_col = None
    for column in ("partC_rev1_geom_wkb", "geometry"):
        if column in out.columns:
            geom_col = column
            break
    if geom_col is not None:
        out["geometry"] = out[geom_col].map(
            lambda value: value
            if hasattr(value, "geom_type")
            else wkb.loads(bytes.fromhex(str(value)))
            if pd.notna(value) and str(value).strip()
            else None
        )
        return gpd.GeoDataFrame(out, geometry="geometry", crs=crs)
    return gpd.GeoDataFrame(out)


def partC_rev1_load_prompt3_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Prompt 3 inputs, requiring enriched directed arcs for real runs by default."""

    cache = _partC_rev1_cache_path(config)
    real_run = not bool(config.get("testing", {}).get("use_toy_data", True))
    allow_fallback = bool(config.get("backbone_partition", {}).get("input", {}).get("allow_directed_arcs_fallback", False))
    crs = _partC_rev1_sqlite_geometry_crs(config)
    if cache.exists():
        with sqlite3.connect(cache) as conn:
            if _partC_rev1_table_exists(conn, "partC_rev1_directed_arcs_enriched"):
                arc_table = "partC_rev1_directed_arcs_enriched"
            elif allow_fallback or not real_run:
                arc_table = "partC_rev1_directed_arcs" if _partC_rev1_table_exists(conn, "partC_rev1_directed_arcs") else None
            else:
                arc_table = None
            if arc_table is None:
                raise FileNotFoundError(
                    f"Missing partC_rev1_directed_arcs_enriched in {cache}; set backbone_partition.input.allow_directed_arcs_fallback=true to fallback."
                )
            if not _partC_rev1_table_exists(conn, "partC_rev1_clean_physical_nodes"):
                raise FileNotFoundError(f"Missing partC_rev1_clean_physical_nodes in {cache}")
            arcs = _partC_rev1_from_wkb(partC_rev1_read_table_sqlite(conn, arc_table), crs=crs)
            nodes = _partC_rev1_from_wkb(partC_rev1_read_table_sqlite(conn, "partC_rev1_clean_physical_nodes"), crs=crs)
            return arcs, nodes

    enriched = _partC_rev1_enriched_gpkg_path(config)
    clean = _partC_rev1_clean_gpkg_path(config)
    gpkg = enriched if enriched.exists() else clean
    if real_run and gpkg == clean and not allow_fallback:
        raise FileNotFoundError(f"Missing enriched GeoPackage for real Prompt 3 run: {enriched}")
    if not gpkg.exists():
        raise FileNotFoundError(f"Prompt 2/2.5 cache missing: {cache}; GeoPackage fallback missing: {gpkg}")
    layers = config.get("clean_geodatabase", {}).get("layers", {})
    arc_layer = "partC_rev1_directed_arcs_enriched" if gpkg == enriched else str(layers.get("directed_arcs", "partC_rev1_directed_arcs"))
    node_layer = str(layers.get("clean_nodes", "partC_rev1_clean_physical_nodes"))
    return gpd.read_file(gpkg, layer=arc_layer), gpd.read_file(gpkg, layer=node_layer)


def _partC_rev1_prepare_partition_nodes(nodes: pd.DataFrame, arcs: pd.DataFrame) -> pd.DataFrame:
    if isinstance(nodes, gpd.GeoDataFrame) and "geometry" in nodes.columns:
        out: pd.DataFrame = nodes.copy()
    elif "geometry" in nodes.columns:
        out = gpd.GeoDataFrame(nodes.copy(), geometry="geometry", crs=getattr(nodes, "crs", None))
    else:
        out = nodes.copy()
    out["partC_rev1_node_uid"] = out["partC_rev1_node_uid"].map(partC_rev1_stable_id)
    backbone = arcs.loc[arcs.get("is_final_backbone", pd.Series(False, index=arcs.index)).fillna(False).astype(bool)]
    backbone_nodes = set(backbone.get("from_node_uid", pd.Series(dtype=str)).map(partC_rev1_stable_id).dropna().astype(str)) | set(
        backbone.get("to_node_uid", pd.Series(dtype=str)).map(partC_rev1_stable_id).dropna().astype(str)
    )
    all_arc_nodes = set(arcs.get("from_node_uid", pd.Series(dtype=str)).map(partC_rev1_stable_id).dropna().astype(str)) | set(
        arcs.get("to_node_uid", pd.Series(dtype=str)).map(partC_rev1_stable_id).dropna().astype(str)
    )
    existing = set(out["partC_rev1_node_uid"].dropna().astype(str))
    missing = sorted(all_arc_nodes - existing)
    if missing:
        missing_df = pd.DataFrame({"partC_rev1_node_uid": missing})
        if isinstance(out, gpd.GeoDataFrame) and "geometry" in out.columns:
            missing_df = gpd.GeoDataFrame(missing_df, geometry=[None] * len(missing_df), crs=out.crs)
        out = pd.concat([out, missing_df], ignore_index=True, sort=False)
    out["is_backbone_node"] = out["partC_rev1_node_uid"].astype(str).isin(backbone_nodes)
    out["is_internal_region_vertex"] = ~out["is_backbone_node"]
    return out


def partC_rev1_build_physical_partition_state(arcs: pd.DataFrame, nodes: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    stage_rows: list[dict[str, Any]] = []

    def timed(stage: str, func: Any) -> Any:
        _PARTC_REV1_LOG.info("Prompt 3 stage started: %s", stage)
        start = time.perf_counter()
        result = func()
        runtime_seconds = time.perf_counter() - start
        stage_rows.append({"stage": stage, "runtime_seconds": runtime_seconds})
        _PARTC_REV1_LOG.info(
            "Prompt 3 stage completed: %s in %.3f seconds%s",
            stage,
            runtime_seconds,
            _partC_rev1_stage_result_summary(result),
        )
        return result

    seeded = timed("seed_candidate_lcc_extraction", lambda: partC_rev1_extract_seed_backbone(arcs, config))
    expanded = timed("direct_seed_lcc_connector_expansion", lambda: partC_rev1_expand_backbone(seeded, nodes, config))
    partition_nodes = timed("physical_partition_node_preparation", lambda: _partC_rev1_prepare_partition_nodes(nodes, expanded["arcs"]))
    nonbackbone_arcs = expanded["arcs"].loc[
        ~expanded["arcs"].get("is_final_backbone", pd.Series(False, index=expanded["arcs"].index)).fillna(False).astype(bool)
    ].copy()
    partition = timed(
        "default_leiden_partition",
        lambda: partC_rev1_run_leiden_partition(
            partition_nodes,
            nonbackbone_arcs,
            config,
        ),
    )
    physical_arc_regions = timed(
        "physical_arc_region_labeling",
        lambda: partC_rev1_assign_physical_arc_regions(
            expanded["arcs"],
            partition_nodes,
            partition["node_partition"],
        ),
    )
    metadata = partition["partition_metadata"].copy()
    metadata["generated_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["graph_hash"] = partC_rev1_nonbackbone_graph_hash(nonbackbone_arcs)
    backbone_nodes = partition_nodes.loc[partition_nodes["is_backbone_node"].fillna(False).astype(bool)].copy()
    outputs = {
        "arcs": expanded["arcs"],
        "backbone_seed_arcs": expanded["arcs"].loc[expanded["arcs"]["is_seed_backbone"].fillna(False).astype(bool)].copy(),
        "backbone_seed_nodes": backbone_nodes,
        "backbone_candidate_direct_connector_arcs": expanded["candidate_direct_connector_arcs"],
        "backbone_direct_connector_links": expanded["direct_connector_links"],
        "backbone_accepted_direct_connector_links": expanded["accepted_direct_connector_links"],
        "final_backbone_arcs": expanded["final_backbone_arcs"],
        "partition_nodes": partition_nodes,
        "nonbackbone_physical_arcs": nonbackbone_arcs,
        "physical_partition_vertices": partition["physical_partition_vertices"],
        "physical_partition_edges": partition["physical_partition_edges"],
        "node_partition": partition["node_partition"],
        "regions": partition["regions"],
        "physical_arc_regions": physical_arc_regions,
        "partition_metadata": metadata,
        "prompt3_stage_runtimes": pd.DataFrame(stage_rows),
    }
    return _partC_rev1_attach_partition_layers(outputs)


def _partC_rev1_stage_result_summary(result: Any) -> str:
    if isinstance(result, pd.DataFrame):
        return f" rows={len(result)}"
    if isinstance(result, dict):
        pieces = []
        for key, value in result.items():
            if isinstance(value, pd.DataFrame):
                pieces.append(f"{key}={len(value)}")
            if len(pieces) >= 4:
                break
        if pieces:
            return " " + ", ".join(pieces)
    return ""


def _partC_rev1_validate_prompt3_run(config: dict[str, Any], *, confirm_full_run: bool, command: str) -> None:
    real_run = not bool(config.get("testing", {}).get("use_toy_data", True))
    if real_run and not confirm_full_run:
        raise ValueError(f"Full real-data {command} requires --confirm-full-run.")


def _partC_rev1_write_ignored_label_variants_warning(config: dict[str, Any]) -> None:
    stability = config.get("backbone_partition", {}).get("stability", {})
    if "label_variants" not in stability:
        return
    partC_rev1_output_file(config, "partC_rev1_prompt3B_ignored_old_label_variants_warning.md").write_text(
        "\n".join(
            [
                "# partC_rev1 Prompt 3B Ignored Old Label Variants Warning",
                "",
                "The config contains `backbone_partition.stability.label_variants`, but Prompt 3B intentionally ignores backbone-label variants.",
                "The stability experiment varies only Leiden resolution and random seed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _partC_rev1_sql_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _partC_rev1_add_sqlite_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({_partC_rev1_sql_identifier(table)})").fetchall()}
    for column, sql_type in columns.items():
        if column not in existing:
            conn.execute(
                f"ALTER TABLE {_partC_rev1_sql_identifier(table)} "
                f"ADD COLUMN {_partC_rev1_sql_identifier(column)} {sql_type}"
            )


def _partC_rev1_bool_int_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: None if pd.isna(value) else int(bool(value)))


def _partC_rev1_update_table_from_label_frame(
    conn: sqlite3.Connection,
    *,
    target_table: str,
    key_column: str,
    labels: pd.DataFrame,
    source_to_target: dict[str, str],
    target_types: dict[str, str],
    temp_table: str,
) -> None:
    if labels.empty or not _partC_rev1_table_exists(conn, target_table):
        return
    available_sources = [column for column in source_to_target if column in labels.columns]
    if not available_sources or key_column not in labels.columns:
        return
    label_frame = labels[[key_column, *available_sources]].copy()
    label_frame = label_frame.drop_duplicates(subset=[key_column], keep="last")
    for column in available_sources:
        if str(label_frame[column].dtype) == "bool":
            label_frame[column] = _partC_rev1_bool_int_series(label_frame[column])
    partC_rev1_write_table_sqlite(conn, temp_table, label_frame)
    _partC_rev1_add_sqlite_columns(conn, target_table, target_types)
    temp_index = f"{temp_table}_{key_column}_idx"
    target_index = f"{target_table}_{key_column}_prompt3_idx"
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {_partC_rev1_sql_identifier(temp_index)} "
        f"ON {_partC_rev1_sql_identifier(temp_table)} ({_partC_rev1_sql_identifier(key_column)})"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {_partC_rev1_sql_identifier(target_index)} "
        f"ON {_partC_rev1_sql_identifier(target_table)} ({_partC_rev1_sql_identifier(key_column)})"
    )
    assignments = []
    for source, target in source_to_target.items():
        if source not in available_sources:
            continue
        assignments.append(f"{_partC_rev1_sql_identifier(target)} = labels.{_partC_rev1_sql_identifier(source)}")
    if assignments:
        conn.execute(
            f"UPDATE {_partC_rev1_sql_identifier(target_table)} AS target "
            f"SET {', '.join(assignments)} "
            f"FROM {_partC_rev1_sql_identifier(temp_table)} AS labels "
            f"WHERE target.{_partC_rev1_sql_identifier(key_column)} = labels.{_partC_rev1_sql_identifier(key_column)}"
        )
    conn.execute(f"DROP TABLE IF EXISTS {_partC_rev1_sql_identifier(temp_table)}")
    conn.commit()


def _partC_rev1_update_canonical_partition_columns(conn: sqlite3.Connection, outputs: dict[str, pd.DataFrame]) -> None:
    link_layer = pd.DataFrame(outputs.get("partition_link_layer", pd.DataFrame()))
    node_layer = pd.DataFrame(outputs.get("partition_node_layer", pd.DataFrame()))
    arc_table = "partC_rev1_directed_arcs_enriched"
    if not _partC_rev1_table_exists(conn, arc_table):
        arc_table = "partC_rev1_directed_arcs"
    link_map = {
        "region_id": "partC_rev1_prompt3_region_id",
        "region_label": "partC_rev1_prompt3_region_label",
        "arc_partition_type": "partC_rev1_prompt3_arc_partition_type",
        "nonbackbone_region_id": "partC_rev1_prompt3_nonbackbone_region_id",
        "neighbor_region_id": "partC_rev1_prompt3_neighbor_region_id",
        "backbone_node_uid": "partC_rev1_prompt3_backbone_node_uid",
        "is_final_backbone": "partC_rev1_prompt3_is_final_backbone",
        "is_post_partition_backbone": "partC_rev1_prompt3_is_post_partition_backbone",
        "is_backbone_region_label": "partC_rev1_prompt3_is_backbone_region",
        "backbone_stage": "partC_rev1_prompt3_backbone_stage",
        "partition_resolution": "partC_rev1_prompt3_partition_resolution",
        "partition_seed": "partC_rev1_prompt3_partition_seed",
    }
    link_types = {
        "partC_rev1_prompt3_region_id": "TEXT",
        "partC_rev1_prompt3_region_label": "TEXT",
        "partC_rev1_prompt3_arc_partition_type": "TEXT",
        "partC_rev1_prompt3_nonbackbone_region_id": "TEXT",
        "partC_rev1_prompt3_neighbor_region_id": "TEXT",
        "partC_rev1_prompt3_backbone_node_uid": "TEXT",
        "partC_rev1_prompt3_is_final_backbone": "INTEGER",
        "partC_rev1_prompt3_is_post_partition_backbone": "INTEGER",
        "partC_rev1_prompt3_is_backbone_region": "INTEGER",
        "partC_rev1_prompt3_backbone_stage": "TEXT",
        "partC_rev1_prompt3_partition_resolution": "REAL",
        "partC_rev1_prompt3_partition_seed": "INTEGER",
    }
    _partC_rev1_update_table_from_label_frame(
        conn,
        target_table=arc_table,
        key_column="directed_arc_id",
        labels=link_layer,
        source_to_target=link_map,
        target_types=link_types,
        temp_table="partC_rev1_prompt3_link_labels_tmp",
    )

    node_map = {
        "region_id": "partC_rev1_prompt3_region_id",
        "region_label": "partC_rev1_prompt3_region_label",
        "nonbackbone_region_id": "partC_rev1_prompt3_nonbackbone_region_id",
        "is_backbone_node": "partC_rev1_prompt3_is_backbone_node",
        "is_internal_region_vertex": "partC_rev1_prompt3_is_internal_region_vertex",
        "partition_resolution": "partC_rev1_prompt3_partition_resolution",
        "partition_seed": "partC_rev1_prompt3_partition_seed",
    }
    node_types = {
        "partC_rev1_prompt3_region_id": "TEXT",
        "partC_rev1_prompt3_region_label": "TEXT",
        "partC_rev1_prompt3_nonbackbone_region_id": "TEXT",
        "partC_rev1_prompt3_is_backbone_node": "INTEGER",
        "partC_rev1_prompt3_is_internal_region_vertex": "INTEGER",
        "partC_rev1_prompt3_partition_resolution": "REAL",
        "partC_rev1_prompt3_partition_seed": "INTEGER",
    }
    _partC_rev1_update_table_from_label_frame(
        conn,
        target_table="partC_rev1_clean_physical_nodes",
        key_column="partC_rev1_node_uid",
        labels=node_layer,
        source_to_target=node_map,
        target_types=node_types,
        temp_table="partC_rev1_prompt3_node_labels_tmp",
    )


def _partC_rev1_write_sqlite_outputs(config: dict[str, Any], outputs: dict[str, pd.DataFrame]) -> None:
    outputs = _partC_rev1_attach_partition_layers(outputs)
    _PARTC_REV1_LOG.info("Prompt 3 SQLite write started")
    table_map = {
        "backbone_seed_arcs": "partC_rev1_backbone_seed_arcs",
        "backbone_seed_nodes": "partC_rev1_backbone_seed_nodes",
        "backbone_candidate_direct_connector_arcs": "partC_rev1_backbone_candidate_direct_connector_arcs",
        "backbone_direct_connector_links": "partC_rev1_backbone_direct_connector_links",
        "backbone_accepted_direct_connector_links": "partC_rev1_backbone_accepted_direct_connector_links",
        "final_backbone_arcs": "partC_rev1_final_backbone_arcs",
        "partition_nodes": "partC_rev1_partition_nodes",
        "node_partition": "partC_rev1_node_partition",
        "regions": "partC_rev1_regions",
        "partition_metadata": "partC_rev1_partition_metadata",
        "prompt3_stage_runtimes": "partC_rev1_prompt3_stage_runtimes",
    }
    obsolete_tables = [
        "partC_rev1_backbone_candidate_expansion_arcs",
        "partC_rev1_backbone_interface_nodes",
        "partC_rev1_backbone_expansion_paths",
        "partC_rev1_supernodes",
        "partC_rev1_node_supernode_map",
        "partC_rev1_supernode_internal_arcs",
        "partC_rev1_supernode_partition",
        "partC_rev1_backbone_partition_links",
        "partC_rev1_backbone_partition_nodes",
        "partC_rev1_physical_self_loop_arcs",
        "partC_rev1_abstract_arcs_raw",
        "partC_rev1_backbone_abstract_arcs",
        "partC_rev1_backbone_abstract_arc_members",
        "partC_rev1_nonbackbone_abstract_arcs",
        "partC_rev1_abstract_arc_region",
        "partC_rev1_igraph_vertices",
        "partC_rev1_igraph_edges",
    ]
    with partC_rev1_connect_sqlite(_partC_rev1_cache_path(config)) as conn:
        for table in obsolete_tables:
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        conn.commit()
        for key, table in table_map.items():
            _PARTC_REV1_LOG.info("Writing SQLite table %s rows=%d", table, len(outputs.get(key, pd.DataFrame())))
            partC_rev1_write_table_sqlite(conn, table, outputs.get(key, pd.DataFrame()))
        _PARTC_REV1_LOG.info("Updating canonical enriched network Prompt 3 columns")
        _partC_rev1_update_canonical_partition_columns(conn, outputs)
        partC_rev1_write_metadata(conn, "partC_rev1_prompt3_completed_utc", datetime.now(timezone.utc).isoformat())
        partC_rev1_write_metadata(conn, "partC_rev1_prompt3_model", "physical_graph_labels_no_supernodes_no_abstract_graph")
    _PARTC_REV1_LOG.info("Prompt 3 SQLite write completed")


def partC_rev1_load_physical_partition_state_from_sqlite(config: dict[str, Any]) -> dict[str, pd.DataFrame] | None:
    table_map = {
        "backbone_seed_arcs": "partC_rev1_backbone_seed_arcs",
        "backbone_seed_nodes": "partC_rev1_backbone_seed_nodes",
        "backbone_direct_connector_links": "partC_rev1_backbone_direct_connector_links",
        "backbone_accepted_direct_connector_links": "partC_rev1_backbone_accepted_direct_connector_links",
        "final_backbone_arcs": "partC_rev1_final_backbone_arcs",
        "partition_nodes": "partC_rev1_partition_nodes",
        "node_partition": "partC_rev1_node_partition",
        "partition_metadata": "partC_rev1_partition_metadata",
    }
    cache = _partC_rev1_cache_path(config)
    if not cache.exists():
        return None
    with sqlite3.connect(cache) as conn:
        if any(not _partC_rev1_table_exists(conn, table) for table in table_map.values()):
            return None
        crs = _partC_rev1_sqlite_geometry_crs(config)
        outputs: dict[str, pd.DataFrame] = {}
        for key, table in table_map.items():
            df = partC_rev1_read_table_sqlite(conn, table)
            if key in {
                "backbone_seed_arcs",
                "backbone_seed_nodes",
                "backbone_direct_connector_links",
                "backbone_accepted_direct_connector_links",
                "final_backbone_arcs",
                "partition_nodes",
            }:
                df = _partC_rev1_from_wkb(df, crs=crs)
            outputs[key] = df
        if _partC_rev1_table_exists(conn, "partC_rev1_directed_arcs_enriched"):
            outputs["arcs"] = _partC_rev1_from_wkb(partC_rev1_read_table_sqlite(conn, "partC_rev1_directed_arcs_enriched"), crs=crs)
        else:
            outputs["arcs"] = pd.DataFrame()
        if not outputs["arcs"].empty:
            if "is_final_backbone" not in outputs["arcs"].columns and "partC_rev1_prompt3_is_final_backbone" in outputs["arcs"].columns:
                outputs["arcs"]["is_final_backbone"] = (
                    pd.to_numeric(outputs["arcs"]["partC_rev1_prompt3_is_final_backbone"], errors="coerce").fillna(0).astype(int).astype(bool)
                )
            outputs["nonbackbone_physical_arcs"] = outputs["arcs"].loc[
                ~outputs["arcs"].get("is_final_backbone", pd.Series(False, index=outputs["arcs"].index)).fillna(False).astype(bool)
            ].copy()
            if "node_partition" in outputs and "partition_nodes" in outputs:
                outputs["physical_arc_regions"] = partC_rev1_assign_physical_arc_regions(
                    outputs["arcs"],
                    outputs["partition_nodes"],
                    outputs["node_partition"],
                )
    return outputs


def _partC_rev1_default_complexity_proxy(regions: pd.DataFrame, arc_regions: pd.DataFrame) -> pd.DataFrame:
    if regions.empty:
        return pd.DataFrame(columns=["region_id", "boundary_interface_count", "m_r_est"])
    region_ids = regions["region_id"].astype(str)
    arc_region_id = arc_regions.get("region_id", pd.Series(dtype=str)).astype(str)
    arc_type = arc_regions.get("arc_partition_type", pd.Series(dtype=str)).astype(str)
    backbone_counts = (
        arc_regions.loc[arc_type.eq("backbone_region_boundary_connector")]
        .assign(region_id=arc_region_id.loc[arc_type.eq("backbone_region_boundary_connector")])
        .groupby("region_id")["backbone_node_uid"]
        .nunique()
        if {"backbone_node_uid", "region_id"}.issubset(arc_regions.columns)
        else pd.Series(dtype=int)
    )
    neighbor_counts = (
        arc_regions.loc[arc_type.eq("region_region_boundary_connector")]
        .assign(region_id=arc_region_id.loc[arc_type.eq("region_region_boundary_connector")])
        .groupby("region_id")["neighbor_region_id"]
        .nunique()
        if {"neighbor_region_id", "region_id"}.issubset(arc_regions.columns)
        else pd.Series(dtype=int)
    )
    proxy = pd.DataFrame({"region_id": region_ids})
    proxy["boundary_interface_count"] = (
        proxy["region_id"].map(backbone_counts).fillna(0).astype(int)
        + proxy["region_id"].map(neighbor_counts).fillna(0).astype(int)
        + 1
    )
    proxy["m_r_est"] = proxy["boundary_interface_count"] * (proxy["boundary_interface_count"] - 1) + 2 * proxy["boundary_interface_count"]
    return proxy


def _partC_rev1_default_partition_summary(outputs: dict[str, pd.DataFrame], runtime_seconds: float | None = None) -> pd.DataFrame:
    arc_regions = outputs["physical_arc_regions"]
    regions = outputs["regions"]
    stage_runtimes = outputs.get("prompt3_stage_runtimes", pd.DataFrame())
    proxy = _partC_rev1_default_complexity_proxy(regions, arc_regions)
    m = pd.to_numeric(proxy.get("m_r_est", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    internal = arc_regions.get("arc_partition_type", pd.Series(dtype=str)).astype(str).eq("regional_internal")
    final_stage = outputs["final_backbone_arcs"].get("backbone_stage", pd.Series(dtype=str)).astype(str)
    row: dict[str, Any] = {
        "seed_candidate_directed_arcs": int(outputs["arcs"].get("is_seed_candidate_backbone", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        if not outputs.get("arcs", pd.DataFrame()).empty
        else np.nan,
        "seed_lcc_directed_arcs": len(outputs["backbone_seed_arcs"]),
        "seed_candidates_excluded_by_lcc": int(
            outputs["arcs"].get("partC_rev1_seed_lcc_status", pd.Series(dtype=str)).astype(str).eq("candidate_outside_largest_component").sum()
        )
        if not outputs.get("arcs", pd.DataFrame()).empty
        else np.nan,
        "direct_connector_added_directed_arcs": int(final_stage.eq("added_direct_seed_lcc_link").sum()),
        "direct_connector_accepted_links": len(outputs["backbone_accepted_direct_connector_links"]),
        "final_backbone_directed_arcs": len(outputs["final_backbone_arcs"]),
        "final_backbone_nodes": len(outputs["backbone_seed_nodes"]),
        "physical_partition_nodes": len(outputs["partition_nodes"]),
        "nonbackbone_physical_directed_arcs": len(outputs["nonbackbone_physical_arcs"]),
        "num_regions": len(regions),
        "regional_internal_arcs": int(internal.sum()),
        "backbone_region_boundary_connectors": int(arc_regions.get("arc_partition_type", pd.Series(dtype=str)).astype(str).eq("backbone_region_boundary_connector").sum()),
        "region_region_boundary_connectors": int(arc_regions.get("arc_partition_type", pd.Series(dtype=str)).astype(str).eq("region_region_boundary_connector").sum()),
        "backbone_labeled_directed_arcs": int(arc_regions.get("region_id", pd.Series(dtype=str)).astype(str).eq("backbone").sum()),
        "sum_m_r_est": float(m.sum()),
        "proxy_physical_label_graph_size": float(len(outputs["final_backbone_arcs"]) + m.sum()),
        "graph_hash": outputs["partition_metadata"].get("graph_hash", pd.Series([""])).iloc[0],
        "total_runtime_seconds": runtime_seconds if runtime_seconds is not None else np.nan,
    }
    for stage in stage_runtimes.itertuples(index=False):
        row[f"runtime_{getattr(stage, 'stage')}_seconds"] = float(getattr(stage, "runtime_seconds"))
    return pd.DataFrame([row])


def _partC_rev1_write_reports(config: dict[str, Any], outputs: dict[str, pd.DataFrame], *, runtime_seconds: float | None = None) -> None:
    _PARTC_REV1_LOG.info("Prompt 3 report write started")
    arcs = outputs.get("arcs", pd.DataFrame())
    seed_summary = pd.DataFrame(
        [
            {
                "seed_candidate_directed_arcs": int(arcs.get("is_seed_candidate_backbone", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
                if not arcs.empty
                else np.nan,
                "seed_lcc_directed_arcs": len(outputs["backbone_seed_arcs"]),
                "seed_lcc_nodes": len(outputs["backbone_seed_nodes"]),
                "seed_candidate_reason_counts": arcs.get("seed_candidate_reason", pd.Series(dtype=str)).value_counts().to_dict() if not arcs.empty else {},
                "seed_lcc_status_counts": arcs.get("partC_rev1_seed_lcc_status", pd.Series(dtype=str)).value_counts().to_dict() if not arcs.empty else {},
            }
        ]
    )
    direct_links = outputs["backbone_direct_connector_links"]
    direct_connector_summary = pd.DataFrame(
        [
            {
                "candidate_direct_connector_arcs": len(outputs["backbone_candidate_direct_connector_arcs"]),
                "candidate_direct_connector_links": len(direct_links),
                "accepted_direct_connector_links": len(outputs["backbone_accepted_direct_connector_links"]),
                "added_direct_connector_directed_arcs": int(
                    outputs["final_backbone_arcs"].get("backbone_stage", pd.Series(dtype=str)).astype(str).eq("added_direct_seed_lcc_link").sum()
                ),
                "rejection_reason_counts": direct_links.get("direct_connector_rejection_reason", pd.Series(dtype=str)).value_counts().to_dict(),
            }
        ]
    )
    partition_summary = _partC_rev1_default_partition_summary(outputs, runtime_seconds=runtime_seconds)
    partC_rev1_write_table(seed_summary, partC_rev1_output_file(config, "partC_rev1_backbone_seed_summary.csv"), index=False)
    partC_rev1_write_table(direct_connector_summary, partC_rev1_output_file(config, "partC_rev1_backbone_direct_connector_summary.csv"), index=False)
    partC_rev1_write_table(partition_summary, partC_rev1_output_file(config, "partC_rev1_partition_summary.csv"), index=False)
    if "prompt3_stage_runtimes" in outputs:
        partC_rev1_write_table(outputs["prompt3_stage_runtimes"], partC_rev1_output_file(config, "partC_rev1_prompt3_stage_runtimes.csv"), index=False)
    partC_rev1_output_file(config, "partC_rev1_prompt3_qgis_layer_guide.md").write_text(
        "\n".join(
            [
                "# partC_rev1 Prompt 3 QGIS Layer Guide",
                "",
                "Open `partC_rev1_cache/partC_rev1_backbone_partition.gpkg`.",
                "If that file was locked during generation, open `partC_rev1_cache/partC_rev1_backbone_partition_physical_nodes.gpkg` instead.",
                "",
                "This Prompt 3 run partitions physical nodes directly. It does not create or use supernodes.",
                "The GeoPackage is written from the canonical enriched network geometry and uses the same CRS as the enriched network source.",
                "",
                "SQLite keeps the enriched network tables as the canonical network tables. Prompt 3 does not create full duplicate SQLite link/node geometry tables for the viewer export.",
                "Instead, it appends or refreshes `partC_rev1_prompt3_*` label columns on:",
                "",
                "- `partC_rev1_directed_arcs_enriched`",
                "- `partC_rev1_clean_physical_nodes`",
                "",
                "Detailed diagnostics and summary tables remain separate, including `partC_rev1_final_backbone_arcs`, `partC_rev1_node_partition`, `partC_rev1_regions`, and `partC_rev1_partition_metadata`.",
                "",
                "Inspect these layers:",
                "",
                "- `partC_rev1_backbone_partition_links`",
                "- `partC_rev1_backbone_partition_nodes`",
                "",
                "Suggested symbology:",
                "",
                "- Links: categorize `partC_rev1_backbone_partition_links` by `region_label`; backbone and post-partition backbone connectors are labeled `backbone`.",
                "- Nodes: categorize `partC_rev1_backbone_partition_nodes` by `region_label`.",
                "- Link boundary roles remain available in `arc_partition_type`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _PARTC_REV1_LOG.info("Prompt 3 report write completed")


def _partC_rev1_partition_resolution_seed(outputs: dict[str, pd.DataFrame]) -> tuple[float, int]:
    metadata = outputs.get("partition_metadata", pd.DataFrame())
    if metadata.empty:
        resolution = 1.0
        seed = 42
    else:
        resolution = pd.to_numeric(metadata.get("resolution", pd.Series([1.0])), errors="coerce").fillna(1.0).iloc[0]
        seed = pd.to_numeric(metadata.get("random_seed", pd.Series([42])), errors="coerce").fillna(42).iloc[0]
    return float(resolution), int(seed)


def _partC_rev1_as_geodataframe(df: pd.DataFrame, *, crs: Any = None) -> gpd.GeoDataFrame:
    if isinstance(df, gpd.GeoDataFrame) and "geometry" in df.columns:
        return df
    if "geometry" in df.columns:
        return gpd.GeoDataFrame(df, geometry="geometry", crs=crs)
    return gpd.GeoDataFrame(df, geometry=[None] * len(df), crs=crs)


def _partC_rev1_build_partition_node_layer(outputs: dict[str, pd.DataFrame]) -> gpd.GeoDataFrame:
    nodes = _partC_rev1_as_geodataframe(outputs.get("partition_nodes", pd.DataFrame()).copy())
    partition = outputs.get("node_partition", pd.DataFrame()).copy()
    if nodes.empty:
        return nodes
    nodes["partC_rev1_node_uid"] = nodes["partC_rev1_node_uid"].map(partC_rev1_stable_id)
    if "region_id" in nodes.columns:
        nodes = nodes.rename(columns={"region_id": "source_region_id"})
    if not partition.empty:
        partition = partition[["partC_rev1_node_uid", "region_id"]].copy()
        partition["partC_rev1_node_uid"] = partition["partC_rev1_node_uid"].map(partC_rev1_stable_id)
        partition = partition.rename(columns={"region_id": "nonbackbone_region_id"})
        nodes = nodes.merge(partition, on="partC_rev1_node_uid", how="left")
    else:
        nodes["nonbackbone_region_id"] = pd.NA
    is_backbone = nodes.get("is_backbone_node", pd.Series(False, index=nodes.index)).fillna(False).astype(bool)
    nodes["region_label"] = np.where(is_backbone, "backbone", nodes["nonbackbone_region_id"])
    nodes["region_label"] = nodes["region_label"].fillna("unassigned")
    nodes["region_id"] = nodes["region_label"]
    nodes["is_backbone_region_label"] = nodes["region_label"].eq("backbone")
    resolution, seed = _partC_rev1_partition_resolution_seed(outputs)
    nodes["partition_resolution"] = resolution
    nodes["partition_seed"] = seed
    return _partC_rev1_as_geodataframe(nodes, crs=getattr(nodes, "crs", None))


def _partC_rev1_build_partition_link_layer(outputs: dict[str, pd.DataFrame], node_layer: pd.DataFrame) -> gpd.GeoDataFrame:
    links = _partC_rev1_as_geodataframe(outputs.get("arcs", pd.DataFrame()).copy())
    if links.empty:
        return links
    links["directed_arc_id_key"] = links["directed_arc_id"].astype(str)
    for column, replacement in {
        "region_id": "source_region_id",
        "is_final_backbone": "source_is_final_backbone",
        "backbone_stage": "source_backbone_stage",
    }.items():
        if column in links.columns:
            links = links.rename(columns={column: replacement})

    final = outputs.get("final_backbone_arcs", pd.DataFrame())
    if not final.empty:
        final_map = final[["directed_arc_id", "backbone_stage"]].copy()
        final_map["directed_arc_id_key"] = final_map["directed_arc_id"].astype(str)
        final_map = final_map.drop(columns=["directed_arc_id"]).drop_duplicates("directed_arc_id_key")
        final_map = final_map.rename(columns={"backbone_stage": "final_backbone_stage"})
        final_map["is_final_backbone"] = True
        links = links.merge(final_map, on="directed_arc_id_key", how="left")
    else:
        links["final_backbone_stage"] = pd.NA
        links["is_final_backbone"] = False
    links["is_final_backbone"] = links["is_final_backbone"].map(lambda value: False if pd.isna(value) else bool(value))

    arc_regions = outputs.get("physical_arc_regions", pd.DataFrame())
    if not arc_regions.empty:
        region_cols = [
            "directed_arc_id",
            "tail_node_uid",
            "head_node_uid",
            "arc_partition_type",
            "region_id",
            "neighbor_region_id",
            "backbone_node_uid",
        ]
        region_map = arc_regions[[c for c in region_cols if c in arc_regions.columns]].copy()
        region_map["directed_arc_id_key"] = region_map["directed_arc_id"].astype(str)
        region_map = region_map.drop(columns=["directed_arc_id"])
        region_map = region_map.rename(columns={"region_id": "physical_region_id"})
        links = links.merge(region_map, on="directed_arc_id_key", how="left", suffixes=("", "_partition"))
    else:
        links["tail_node_uid"] = pd.NA
        links["head_node_uid"] = pd.NA
        links["arc_partition_type"] = pd.NA
        links["physical_region_id"] = pd.NA
        links["neighbor_region_id"] = pd.NA
        links["backbone_node_uid"] = pd.NA

    node_region = {}
    if not node_layer.empty and {"partC_rev1_node_uid", "region_label"}.issubset(node_layer.columns):
        node_region = node_layer.set_index("partC_rev1_node_uid")["region_label"].astype(str).to_dict()
    if "from_node_uid" in links.columns:
        links["from_node_uid"] = links["from_node_uid"].map(partC_rev1_stable_id)
    if "to_node_uid" in links.columns:
        links["to_node_uid"] = links["to_node_uid"].map(partC_rev1_stable_id)

    post_partition_backbone = links.get("arc_partition_type", pd.Series(dtype=str)).fillna("").astype(str).eq("backbone_network") & ~links["is_final_backbone"]
    links["is_post_partition_backbone"] = post_partition_backbone
    links["is_backbone_region_label"] = links["is_final_backbone"] | links["is_post_partition_backbone"]
    links["region_label"] = np.where(links["is_backbone_region_label"], "backbone", links.get("physical_region_id", pd.Series(pd.NA, index=links.index)))
    links["nonbackbone_region_id"] = np.where(links["is_backbone_region_label"], pd.NA, links.get("physical_region_id", pd.Series(pd.NA, index=links.index)))

    missing_label = pd.Series(links["region_label"]).isna()
    if node_region and {"from_node_uid", "to_node_uid"}.issubset(links.columns):
        same_node = links["from_node_uid"].astype(str).eq(links["to_node_uid"].astype(str))
        links.loc[missing_label & same_node, "region_label"] = links.loc[missing_label & same_node, "from_node_uid"].map(node_region)
    links["region_label"] = links["region_label"].fillna("unassigned")
    links["region_id"] = links["region_label"]
    links["backbone_stage"] = links["final_backbone_stage"]
    links.loc[links["is_post_partition_backbone"], "backbone_stage"] = "post_partition_backbone_network"
    links["backbone_stage"] = links["backbone_stage"].fillna("non_backbone")
    resolution, seed = _partC_rev1_partition_resolution_seed(outputs)
    links["partition_resolution"] = resolution
    links["partition_seed"] = seed
    return _partC_rev1_as_geodataframe(links.drop(columns=["directed_arc_id_key"], errors="ignore"), crs=getattr(links, "crs", None))


def _partC_rev1_attach_partition_layers(outputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    node_layer = outputs.get("partition_node_layer", pd.DataFrame())
    link_layer = outputs.get("partition_link_layer", pd.DataFrame())
    if node_layer.empty:
        node_layer = _partC_rev1_build_partition_node_layer(outputs)
        outputs["partition_node_layer"] = node_layer
    if link_layer.empty:
        outputs["partition_link_layer"] = _partC_rev1_build_partition_link_layer(outputs, node_layer)
    return outputs


def _partC_rev1_write_gpkg_outputs(config: dict[str, Any], outputs: dict[str, pd.DataFrame]) -> Path | None:
    outputs = _partC_rev1_attach_partition_layers(outputs)
    cache_dir = _partC_rev1_cache_path(config).parent
    gpkg = cache_dir / "partC_rev1_backbone_partition.gpkg"
    layers = {
        "partC_rev1_backbone_partition_links": outputs["partition_link_layer"],
        "partC_rev1_backbone_partition_nodes": outputs["partition_node_layer"],
    }
    writable_layers = [
        (layer, df)
        for layer, df in layers.items()
        if isinstance(df, gpd.GeoDataFrame) and "geometry" in df.columns and df.geometry.notna().any()
    ]
    if not writable_layers:
        return None
    _PARTC_REV1_LOG.info("Prompt 3 GeoPackage write started: %s", gpkg)
    if gpkg.exists():
        try:
            gpkg.unlink()
        except PermissionError:
            gpkg = cache_dir / "partC_rev1_backbone_partition_physical_nodes.gpkg"
            if gpkg.exists():
                gpkg.unlink()
    gpkg.parent.mkdir(parents=True, exist_ok=True)
    for layer, df in writable_layers:
        _PARTC_REV1_LOG.info("Writing GeoPackage layer %s rows=%d", layer, len(df))
        df.to_file(gpkg, layer=layer, driver="GPKG")
    _PARTC_REV1_LOG.info("Prompt 3 GeoPackage write completed: %s", gpkg)
    return gpkg


def partC_rev1_run_partition(config_path: str, *, confirm_full_run: bool = False) -> dict[str, Any]:
    config = load_partC_rev1_config(config_path)
    output_dir = config.get("outputs", {}).get("output_dir", config.get("project", {}).get("output_dir", "partC_rev1_outputs"))
    partC_rev1_setup_logging(output_dir)
    _partC_rev1_validate_prompt3_run(config, confirm_full_run=confirm_full_run, command="partition")
    _partC_rev1_write_ignored_label_variants_warning(config)
    start = time.perf_counter()
    arcs, nodes = partC_rev1_load_prompt3_inputs(config)
    _PARTC_REV1_LOG.info("Prompt 3 inputs loaded: arcs=%d nodes=%d", len(arcs), len(nodes))
    outputs = partC_rev1_build_physical_partition_state(arcs, nodes, config)
    _partC_rev1_write_sqlite_outputs(config, outputs)
    runtime = time.perf_counter() - start
    _partC_rev1_write_reports(config, outputs, runtime_seconds=runtime)
    gpkg = _partC_rev1_write_gpkg_outputs(config, outputs)
    _PARTC_REV1_LOG.info("Prompt 3A partition complete in %.3f seconds", runtime)
    return {
        "outputs": outputs,
        "cache_path": str(_partC_rev1_cache_path(config)),
        "gpkg_path": str(gpkg) if gpkg is not None else "",
        "runtime_seconds": runtime,
    }


def partC_rev1_run_partition_stability(config_path: str, *, confirm_full_run: bool = False) -> dict[str, Any]:
    config = load_partC_rev1_config(config_path)
    output_dir = config.get("outputs", {}).get("output_dir", config.get("project", {}).get("output_dir", "partC_rev1_outputs"))
    partC_rev1_setup_logging(output_dir)
    _partC_rev1_validate_prompt3_run(config, confirm_full_run=confirm_full_run, command="partition-stability")
    _partC_rev1_write_ignored_label_variants_warning(config)
    start = time.perf_counter()
    partition_state = partC_rev1_load_physical_partition_state_from_sqlite(config)
    if partition_state is None:
        arcs, nodes = partC_rev1_load_prompt3_inputs(config)
        partition_state = partC_rev1_build_physical_partition_state(arcs, nodes, config)
        _partC_rev1_write_sqlite_outputs(config, partition_state)
        _partC_rev1_write_reports(config, partition_state)
    stability = partC_rev1_compute_partition_stability(partition_state, config, write_outputs=True)
    runtime = time.perf_counter() - start
    return {
        "stability": stability,
        "cache_path": str(_partC_rev1_cache_path(config)),
        "runtime_seconds": runtime,
        "varies_only_resolution_and_seed": True,
    }
