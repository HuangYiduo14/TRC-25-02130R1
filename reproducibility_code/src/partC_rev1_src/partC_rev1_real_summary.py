"""Summarize a completed real-data partC_rev1 preprocessing run."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import pyogrio


def _partC_rev1_csv_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except UnicodeDecodeError:
        with path.open("r") as handle:
            return max(sum(1 for _ in handle) - 1, 0)


def _partC_rev1_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _partC_rev1_table_counts(cache_path: Path) -> dict[str, int]:
    with sqlite3.connect(cache_path) as conn:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
        return {table: int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]) for table in tables}


def _partC_rev1_layer_infos(gpkg_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_name, fallback_geom_type in pyogrio.list_layers(gpkg_path):
        info = pyogrio.read_info(gpkg_path, layer=layer_name)
        fields = info.get("fields")
        if fields is None:
            field_list: list[str] = []
        elif hasattr(fields, "tolist"):
            field_list = [str(value) for value in fields.tolist()]
        else:
            field_list = [str(value) for value in fields]
        rows.append(
            {
                "layer_name": str(layer_name),
                "row_count": int(info.get("features") or 0),
                "crs": str(info.get("crs") or ""),
                "geometry_type": str(info.get("geometry_type") or fallback_geom_type),
                "fields": ", ".join(field_list),
            }
        )
    return rows


def partC_rev1_write_real_preprocess_summaries(
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    out_dir = root / "partC_rev1_outputs"
    docs_dir = root / "partC_rev1_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    cache = root / "partC_rev1_cache" / "partC_rev1_network.sqlite"
    gpkg = root / "partC_rev1_cache" / "partC_rev1_clean_network.gpkg"
    summary_src = out_dir / "partC_rev1_preprocess_summary.csv"
    summary_real = out_dir / "partC_rev1_real_preprocess_summary.csv"
    run_md = docs_dir / "partC_rev1_real_preprocess_run_summary.md"
    qgis_md = docs_dir / "partC_rev1_real_preprocess_qgis_layers.md"

    pre = _partC_rev1_read_csv(summary_src)
    pre_row = pre.iloc[0].to_dict() if not pre.empty else {}
    table_counts = _partC_rev1_table_counts(cache)
    layer_infos = _partC_rev1_layer_infos(gpkg)

    with sqlite3.connect(cache) as conn:
        centroids = pd.read_sql_query(
            "SELECT county_fips, COUNT(*) AS n FROM partC_rev1_county_centroids GROUP BY county_fips",
            conn,
        )
    centroid_count = int(centroids["n"].sum()) if not centroids.empty else 0
    multi_centroid_counties = int((centroids["n"] > 1).sum()) if not centroids.empty else 0

    gt = _partC_rev1_read_csv(out_dir / "partC_rev1_ground_truth_match_report.csv")
    gt_matched = int(gt.get("matched_link_rows", pd.Series([0])).iloc[0]) if not gt.empty else 0
    gt_missing = int(gt.get("missing_link_rows", pd.Series([0])).iloc[0]) if not gt.empty else 0
    gt_rate = float(gt.get("match_rate", pd.Series([0.0])).iloc[0]) if not gt.empty else 0.0

    excluded_rows = _partC_rev1_csv_count(out_dir / "partC_rev1_links_excluded_no_lane_direction.csv")
    defaulted_rows = _partC_rev1_csv_count(out_dir / "partC_rev1_links_defaulted_no_lane_direction.csv")
    node_dup_rows = _partC_rev1_csv_count(out_dir / "partC_rev1_duplicate_node_report.csv")
    link_dup_rows = _partC_rev1_csv_count(out_dir / "partC_rev1_duplicate_link_report.csv")
    susp_node_rows = _partC_rev1_csv_count(out_dir / "partC_rev1_suspicious_duplicate_node_report.csv")
    susp_link_rows = _partC_rev1_csv_count(out_dir / "partC_rev1_suspicious_duplicate_link_report.csv")
    snap_fail_rows = _partC_rev1_csv_count(out_dir / "partC_rev1_topology_snap_failures.csv")
    unmatched_od_rows = _partC_rev1_csv_count(out_dir / "partC_rev1_unmatched_od_counties.csv")

    real_summary = {
        "run_id": "partC_rev1_real_preprocess",
        "cache_path": str(cache),
        "clean_geopackage_path": str(gpkg),
        "sqlite_size_bytes": cache.stat().st_size if cache.exists() else 0,
        "geopackage_size_bytes": gpkg.stat().st_size if gpkg.exists() else 0,
        "raw_link_rows": int(pre_row.get("raw link rows", table_counts.get("partC_rev1_clean_physical_links", 0)) or 0),
        "clean_link_rows": int(pre_row.get("clean link rows", table_counts.get("partC_rev1_clean_physical_links", 0)) or 0),
        "raw_node_rows": int(pre_row.get("raw node rows", table_counts.get("partC_rev1_clean_physical_nodes", 0)) or 0),
        "clean_node_rows": int(pre_row.get("clean node rows", table_counts.get("partC_rev1_clean_physical_nodes", 0)) or 0),
        "directed_arc_rows": int(pre_row.get("directed arc rows", table_counts.get("partC_rev1_directed_arcs", 0)) or 0),
        "excluded_no_lane_direction_links": excluded_rows,
        "defaulted_no_lane_bidirectional_links": defaulted_rows,
        "defaulted_no_lane_directed_arcs": defaulted_rows * 2,
        "duplicate_node_report_rows": node_dup_rows,
        "duplicate_link_report_rows": link_dup_rows,
        "suspicious_duplicate_node_rows": susp_node_rows,
        "suspicious_duplicate_link_rows": susp_link_rows,
        "snap_failure_rows": snap_fail_rows,
        "centroid_count": centroid_count,
        "counties_with_multiple_centroids": multi_centroid_counties,
        "county_od_rows": table_counts.get("partC_rev1_od_county", int(pre_row.get("county OD rows", 0) or 0)),
        "centroid_od_rows": table_counts.get("partC_rev1_od_centroid", int(pre_row.get("centroid OD rows", 0) or 0)),
        "unmatched_od_county_rows": unmatched_od_rows,
        "ground_truth_rows": table_counts.get("partC_rev1_ground_truth_link_flow", int(pre_row.get("ground-truth rows", 0) or 0)),
        "ground_truth_matched_links": gt_matched,
        "ground_truth_missing_links_assigned_zero": gt_missing,
        "ground_truth_match_rate": gt_rate,
        "total_county_demand_annual_ktons": float(pre_row.get("total county demand annual ktons", 0.0) or 0.0),
        "total_centroid_demand_annual_ktons": float(pre_row.get("total centroid demand annual ktons", 0.0) or 0.0),
        "runtime_by_stage": str(pre_row.get("runtime by stage", "")),
    }
    pd.DataFrame([real_summary]).to_csv(summary_real, index=False)

    layer_lines = [
        f"| {row['layer_name']} | {row['row_count']:,} | {row['crs']} | {row['geometry_type']} |"
        for row in layer_infos
    ]
    qgis_md.write_text(
        "\n".join(
            [
                "# partC_rev1 Real Preprocess QGIS Layers",
                "",
                f"GeoPackage: `{gpkg}`",
                "",
                "Open first: `partC_rev1_clean_physical_links` for the cleaned physical graph, then add `partC_rev1_directed_arcs` for direction-aware arrows.",
                "",
                "| Layer | Row count | CRS | Geometry type |",
                "| --- | ---: | --- | --- |",
                *layer_lines,
                "",
                "## Suggested Symbolization",
                "",
                "- `partC_rev1_clean_physical_links`: color by `partC_rev1_topology_status` or `partC_rev1_direction_case`; inspect `partC_rev1_ground_truth_flow` with a graduated color ramp.",
                "- `partC_rev1_directed_arcs`: symbolize arrows by `direction`; color by `is_open`, `is_valid_for_truck`, `class_code`, or `capacity`.",
                "- `partC_rev1_clean_physical_nodes`: filter `partC_rev1_centroid = 1` and color by `partC_rev1_county_fips` or `partC_rev1_fafid`.",
                "",
                "## Diagnostics",
                "",
                f"- Links excluded because AB/BA lane information was missing or unusable: {excluded_rows:,}",
                f"- Links defaulted to bidirectional 1-lane arcs: {defaulted_rows:,}",
                f"- Default no-lane directed arcs appended: {defaulted_rows * 2:,}",
                f"- Duplicate node report rows: {node_dup_rows:,}",
                f"- Duplicate link report rows: {link_dup_rows:,}",
                f"- Suspicious duplicate node geometry rows/groups reported: {susp_node_rows:,}",
                f"- Suspicious duplicate link geometry rows/groups reported: {susp_link_rows:,}",
                f"- Endpoint snap failures over tolerance: {snap_fail_rows:,}",
                f"- Centroids found: {centroid_count:,}",
                f"- Counties with multiple centroids: {multi_centroid_counties:,}",
                f"- OD rows unmatched to centroids: {unmatched_od_rows:,}",
                f"- Physical links matched to ground-truth flow: {gt_matched:,}",
                f"- Physical links missing ground-truth flow and assigned 0.0: {gt_missing:,}",
                "",
                "## Fields To Inspect",
                "",
                "`partC_rev1_clean_physical_nodes`: `partC_rev1_node_uid`, `partC_rev1_original_node_id`, `partC_rev1_centroid`, `partC_rev1_centroid_id_raw`, `partC_rev1_county_fips`, `partC_rev1_fafid`.",
                "",
                "`partC_rev1_clean_physical_links`: `partC_rev1_link_uid`, `ID`, `Dir`, `AB_Lanes`, `BA_Lanes`, `Class`, `NHS`, `NHFN`, `Truck`, `Status`, `Road_Name`, `Sign_Rte`, `Rte_Type`, `Rte_Number`, `partC_rev1_ground_truth_flow`, `partC_rev1_duplicate_status`, `partC_rev1_topology_status`, `partC_rev1_missing_lane_defaulted`, `partC_rev1_default_lane_count`.",
                "",
                "`partC_rev1_directed_arcs`: `directed_arc_id`, `partC_rev1_link_uid`, `physical_link_id`, `from_node_uid`, `to_node_uid`, `from_node_id`, `to_node_id`, `direction`, `length_miles`, `free_time_min`, `capacity`, `class_code`, `nhs`, `nhfn`, `truck`, `status`, `is_open`, `is_valid_for_truck`, `partC_rev1_snap_distance_A_m`, `partC_rev1_snap_distance_B_m`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    run_md.write_text(
        "\n".join(
            [
                "# partC_rev1 Real Preprocess Run Summary",
                "",
                f"Config: `{root / 'partC_rev1_configs' / 'partC_rev1_real_preprocess_config.json'}`",
                "",
                f"SQLite cache: `{cache}` ({real_summary['sqlite_size_bytes']:,} bytes)",
                "",
                f"Clean GeoPackage: `{gpkg}` ({real_summary['geopackage_size_bytes']:,} bytes)",
                "",
                "## Run Counts",
                "",
                f"- Raw links: {real_summary['raw_link_rows']:,}",
                f"- Clean links: {real_summary['clean_link_rows']:,}",
                f"- Raw nodes: {real_summary['raw_node_rows']:,}",
                f"- Clean nodes: {real_summary['clean_node_rows']:,}",
                f"- Directed arcs: {real_summary['directed_arc_rows']:,}",
                f"- County OD rows: {real_summary['county_od_rows']:,}",
                f"- Centroid OD rows: {real_summary['centroid_od_rows']:,}",
                f"- Total county demand annual ktons: {real_summary['total_county_demand_annual_ktons']:,.6f}",
                f"- Total centroid demand annual ktons: {real_summary['total_centroid_demand_annual_ktons']:,.6f}",
                "",
                "## Diagnostics",
                "",
                f"- Links excluded for no lane direction: {excluded_rows:,}",
                f"- Links defaulted to bidirectional 1-lane arcs: {defaulted_rows:,}",
                f"- Default no-lane directed arcs appended: {defaulted_rows * 2:,}",
                f"- Exact/duplicate node report rows: {node_dup_rows:,}",
                f"- Exact/duplicate link report rows: {link_dup_rows:,}",
                f"- Suspicious duplicate node rows/groups: {susp_node_rows:,}",
                f"- Suspicious duplicate link rows/groups: {susp_link_rows:,}",
                f"- Snap failures: {snap_fail_rows:,}",
                f"- Centroids: {centroid_count:,}",
                f"- Counties with multiple centroids: {multi_centroid_counties:,}",
                f"- Unmatched OD county rows: {unmatched_od_rows:,}",
                f"- Ground-truth matched links: {gt_matched:,}",
                f"- Ground-truth missing links assigned 0.0: {gt_missing:,}",
                f"- Ground-truth match rate: {gt_rate:.6f}",
                "",
                "## QGIS",
                "",
                f"Open `{gpkg}` in QGIS. Start with `partC_rev1_clean_physical_links`, then add `partC_rev1_directed_arcs` for direction arrows. Details are in `{qgis_md}`.",
                "",
                "## Runtime By Stage",
                "",
                real_summary["runtime_by_stage"],
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "summary_csv": str(summary_real),
        "run_md": str(run_md),
        "qgis_md": str(qgis_md),
        "layers": layer_infos,
        "table_counts": table_counts,
        "summary": real_summary,
    }


def main() -> int:
    result = partC_rev1_write_real_preprocess_summaries()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
