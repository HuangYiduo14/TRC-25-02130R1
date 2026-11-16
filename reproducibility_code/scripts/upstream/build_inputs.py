"""Explicit portable FAF-to-GIS-to-array stages; no repository-local data required."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config_for_stage(path):
    control = json.loads(Path(path).read_text(encoding="utf-8"))
    if "REPRODUCTION_RUN_ID" in os.environ:
        control["run_id"] = os.environ["REPRODUCTION_RUN_ID"]
    for key in ("phase_id", "run_id", "input_versions", "tasks", "expected_outputs", "acceptance_tests", "resource_budget"):
        if not control.get(key):
            raise ValueError(f"Explicit nonempty configuration field required: {key}")
    root = Path(control.get("package_root", PACKAGE_ROOT)).expanduser().resolve()
    out = Path(control["output_root"])
    out = (root / out).resolve() if not out.is_absolute() else out.resolve()
    raw = Path(control["raw_root"])
    raw = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    if out == raw or raw in out.parents:
        raise ValueError("Output root may not be inside protected raw-input root")
    from partC_rev1_src.partC_rev1_config import partC_rev1_default_config
    cfg = partC_rev1_default_config()
    for section, value in control["legacy_options"].items():
        cfg[section] = value
    cfg["project"].update(run_id=control["run_id"], strict_inputs=False)
    # Legacy filenames are retained because the original path validator requires
    # its prefix; all folder values resolve inside this new output root.
    for key in ("source_code_dir", "tests_dir", "config_dir", "cache_dir", "output_dir", "docs_dir"):
        cfg["project"][key] = str(out / ("partC_rev1_" + key))
    cfg["outputs"]["output_dir"] = str(out / "partC_rev1_reports")
    cfg["outputs"]["docs_dir"] = str(out / "partC_rev1_docs")
    cfg["project"]["output_dir"] = cfg["outputs"]["output_dir"]
    cfg["project"]["docs_dir"] = cfg["outputs"]["docs_dir"]
    cfg["preprocess"]["cache_sqlite"] = str(out / "partC_rev1_cache" / "partC_rev1_network.sqlite")
    cfg["clean_geodatabase"]["output_gpkg_path"] = str(out / "partC_rev1_cache" / "partC_rev1_clean_network.gpkg")
    for key, relative in control["raw_paths"].items():
        if key in {"strahnet_gpkg_path", "nhs_gpkg_path", "nn_gpkg_path"}:
            cfg["ntad"][key] = str(raw / relative)
        else:
            cfg["inputs"][key] = str(raw / relative)
    # External FAF assignments are optional validation, not an input to physics.
    cfg["inputs"]["ground_truth_flow_dir"] = ""
    cfg["inputs"]["ground_truth_flow_path"] = ""
    cfg["ground_truth"]["add_ground_truth_to_clean_links"] = False
    cfg["testing"]["use_toy_data"] = False
    return control, cfg, out, raw


def require_raw(cfg, role):
    keys = {
        "network": ["network_gdb_path"],
        "demand": ["od_path", "origin_factors_path", "destination_factors_path"],
        "ntad": ["strahnet_gpkg_path", "nhs_gpkg_path", "nn_gpkg_path"],
    }[role]
    section = cfg["ntad"] if role == "ntad" else cfg["inputs"]
    for key in keys:
        if not Path(section[key]).exists():
            raise FileNotFoundError(f"Download/extract required input first: {key} -> {section[key]}")


def validate_inputs(cfg, out):
    import pandas as pd
    import pyogrio
    for role in ("network", "demand", "ntad"):
        require_raw(cfg, role)
    gdb = cfg["inputs"]["network_gdb_path"]
    links = pyogrio.read_info(gdb, layer=cfg["inputs"]["links_layer"])
    nodes = pyogrio.read_info(gdb, layer=cfg["inputs"]["nodes_layer"])
    required_links = {"ID", "LENGTH", "Class", "AB_Lanes", "BA_Lanes", "STRAHNET"}
    required_nodes = {"ID", "Centroid", "CentroidID"}
    for required, info in ((required_links, links), (required_nodes, nodes)):
        missing = required - set(info["fields"])
        if missing or not info["features"] or not info["crs"]:
            raise ValueError(f"Wrong FAF network schema/empty layer/missing CRS: {missing}")
    columns = set(pd.read_csv(cfg["inputs"]["od_path"], nrows=0).columns)
    expected = {"dms_orig", "dms_dest", "dms_mode", "sctg2", "tons_2017"}
    if expected - columns:
        raise ValueError(f"Wrong demand schema; missing {expected - columns}")
    for key, required in [("origin_factors_path", {"dms_orig", "dms_orig_cnty", "sctgG5", "f_orig"}),
                          ("destination_factors_path", {"dms_dest", "dms_dest_cnty", "sctgG5", "f_dest"})]:
        missing = required - set(pd.read_csv(cfg["inputs"][key], nrows=0).columns)
        if missing:
            raise ValueError(f"Wrong published truck factor schema: {key}: {missing}")
    identities = {}
    from partC_rev1_src.partC_rev1_hashing import partC_rev1_file_signature
    for key, value in {**cfg["inputs"], **cfg["ntad"]}.items():
        p = Path(str(value))
        if key.endswith("_path") and p.is_file():
            identities[key] = {"path": str(p), "size_bytes": p.stat().st_size, "signature": partC_rev1_file_signature(p)}
    identities["network_gdb_path"] = {"path": gdb, "format": "GDAL-readable FAF geodatabase", "members":
        [{"path": p.relative_to(gdb).as_posix(), "size_bytes": p.stat().st_size, "signature": partC_rev1_file_signature(p)}
         for p in sorted(Path(gdb).rglob("*")) if p.is_file()]} if Path(gdb).is_dir() else identities["network_gdb_path"]
    return {"input_versions": identities, "link_rows": int(links["features"]), "node_rows": int(nodes["features"]), "status": "PASS"}


def clean_network(cfg, out):
    require_raw(cfg, "network")
    from partC_rev1_src.partC_rev1_physical_graph import partC_rev1_read_network_layers, partC_rev1_build_directed_arcs
    from partC_rev1_src.partC_rev1_cleaning import partC_rev1_clean_exact_duplicate_nodes, partC_rev1_clean_exact_duplicate_links
    from partC_rev1_src.partC_rev1_topology_snap import partC_rev1_resolve_link_endpoints
    from partC_rev1_src.partC_rev1_centroids import partC_rev1_extract_county_centroids
    from partC_rev1_src.partC_rev1_clean_geodatabase import partC_rev1_write_clean_geodatabase
    from partC_rev1_src.partC_rev1_cache import partC_rev1_connect_sqlite, partC_rev1_write_table_sqlite
    database = Path(cfg["preprocess"]["cache_sqlite"])
    if database.exists() or Path(cfg["clean_geodatabase"]["output_gpkg_path"]).exists():
        raise FileExistsError("Network already exists; use a new output_root to rerun")
    links, nodes = partC_rev1_read_network_layers(cfg)
    raw_counts = {"raw_links": len(links), "raw_nodes": len(nodes)}
    nodes, _ = partC_rev1_clean_exact_duplicate_nodes(nodes, cfg)
    links, _ = partC_rev1_clean_exact_duplicate_links(links, cfg)
    links = partC_rev1_resolve_link_endpoints(links, nodes, cfg)
    arcs = partC_rev1_build_directed_arcs(links, cfg)
    centroids = partC_rev1_extract_county_centroids(nodes, cfg)
    if min(len(nodes), len(links), len(arcs), len(centroids)) <= 0:
        raise ValueError("No-op/empty network or centroid set")
    if arcs["directed_arc_id"].duplicated().any() or nodes["partC_rev1_node_uid"].duplicated().any():
        raise ValueError("Stable physical IDs must be unique")
    with partC_rev1_connect_sqlite(database) as conn:
        for name, data in [("clean_physical_nodes", nodes), ("clean_physical_links", links),
                           ("directed_arcs", arcs), ("county_centroids", centroids)]:
            partC_rev1_write_table_sqlite(conn, "partC_rev1_" + name, data)
    gis = partC_rev1_write_clean_geodatabase(nodes, links, arcs, cfg)
    return {**raw_counts, "clean_links": len(links), "clean_nodes": len(nodes), "directed_arcs": len(arcs),
            "centroids": len(centroids), "rich_sqlite": str(database), "clean_gpkg": gis["clean_geodatabase_path"]}


def disaggregate_demand(cfg, out):
    """Same commodity-factor/equal-centroid disaggregation in bounded SQL batches."""
    import numpy as np
    import pandas as pd
    require_raw(cfg, "demand")
    from partC_rev1_src.partC_rev1_od import partC_rev1_read_faf_od, partC_rev1_expand_faf_to_county_od, partC_rev1_expand_county_od_to_centroids
    from partC_rev1_src.partC_rev1_io import partC_rev1_read_table
    from partC_rev1_src.partC_rev1_cache import partC_rev1_write_table_sqlite
    database = Path(cfg["preprocess"]["cache_sqlite"])
    if not database.exists():
        raise FileNotFoundError("Run clean-network first")
    # The demand parser groups to five commodity classes, following the original
    # 2017 truck semantics. CSV usecols avoids reading unused forecast/value data.
    original_read = partC_rev1_read_table
    import partC_rev1_src.partC_rev1_od as od_module
    def read_required(path):
        if Path(path).resolve() == Path(cfg["inputs"]["od_path"]).resolve():
            return pd.read_csv(path, usecols=["dms_orig", "dms_dest", "dms_mode", "sctg2", "tons_2017"])
        return original_read(path)
    od_module.partC_rev1_read_table = read_required
    try:
        raw = partC_rev1_read_faf_od(cfg)
    finally:
        od_module.partC_rev1_read_table = original_read
    origin = pd.read_csv(cfg["inputs"]["origin_factors_path"])
    dest = pd.read_csv(cfg["inputs"]["destination_factors_path"])
    for data, region, county in [(origin, "dms_orig", "dms_orig_cnty"), (dest, "dms_dest", "dms_dest_cnty")]:
        if data[[region, county, "sctgG5"]].duplicated().any():
            raise ValueError("Duplicate county/FAF/commodity factor keys")
        # Batches may be appended without another national groupby only if each
        # county has a unique FAF-zone membership in these published factors.
        if data.groupby(county)[region].nunique().max() > 1:
            raise ValueError("County maps to more than one FAF zone; cannot append disjoint batches")
    with sqlite3.connect(database) as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='partC_rev1_od_centroid'").fetchone():
            raise FileExistsError("Demand already exists; use a new output_root to rerun")
        centroids = pd.read_sql_query('SELECT * FROM "partC_rev1_county_centroids"', conn)
        partC_rev1_write_table_sqlite(conn, "partC_rev1_od_raw_2017_truck", raw)
        county_rows = centroid_rows = 0
        county_total = centroid_total = 0.0
        batch_count = 0
        for (region, commodity), group in raw.groupby(["dms_orig", "sctgG5"], sort=True):
            # At most one origin-zone/commodity is expanded in memory.
            selected_origin = origin[(origin["dms_orig"].astype(str) == str(region)) & (origin["sctgG5"] == commodity)]
            selected_dest = dest[dest["sctgG5"] == commodity]
            county = partC_rev1_expand_faf_to_county_od(group, selected_origin, selected_dest, cfg)
            centroid = partC_rev1_expand_county_od_to_centroids(county, centroids, cfg)
            for name, data in [("od_county", county), ("od_centroid", centroid)]:
                partC_rev1_write_table_sqlite(conn, "partC_rev1_" + name, data, if_exists="append")
            county_rows += len(county)
            centroid_rows += len(centroid)
            county_total += float(county["demand_annual_ktons"].sum())
            centroid_total += float(centroid["demand_annual_ktons"].sum())
            batch_count += 1
        if not centroid_rows or not np.isfinite(centroid_total) or centroid_total <= 0:
            raise ValueError("No positive centroid demand was produced")
        conn.execute('CREATE INDEX partC_rev1_od_origin ON partC_rev1_od_centroid(origin_node_uid)')
        conn.commit()
    result = {"raw_grouped_rows": len(raw), "raw_annual_ktons": float(raw["demand_annual_ktons"].sum()),
              "county_rows": county_rows, "county_annual_ktons": county_total,
              "centroid_rows": centroid_rows, "centroid_annual_ktons": centroid_total, "batches": batch_count,
              "warning": "County/centroid coverage loss is measured; factors are never normalized or missing counties invented."}
    write_json(out / "demand_disaggregation_summary.json", result)
    return result


def enrich_network(cfg, out):
    require_raw(cfg, "ntad")
    import partC_rev1_src.partC_rev1_enrich_capacity_pipeline as pipeline
    from vectorized_capacity import attach_link_aadt_defaults, enrich_directed_arcs_capacity
    database = Path(cfg["preprocess"]["cache_sqlite"])
    if not database.exists():
        raise FileNotFoundError("Run clean-network first")
    with sqlite3.connect(database) as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='partC_rev1_directed_arcs_enriched'").fetchone():
            raise FileExistsError("Enrichment already exists; use a new output_root to rerun")
    pipeline.partC_rev1_attach_link_aadt_defaults = attach_link_aadt_defaults
    pipeline.partC_rev1_enrich_directed_arcs_capacity = enrich_directed_arcs_capacity
    resolved = out / "resolved_legacy_config.json"
    write_json(resolved, cfg)
    return pipeline.partC_rev1_run_enrich_capacity(str(resolved), confirm_full_run=True)


def snapshots(cfg, out):
    import numpy as np
    import pandas as pd
    from partC_rev1_src.partC_rev1_graph_snapshot import partC_rev1_build_graph_snapshot_from_frames
    from partC_rev1_src.partC_rev1_od_snapshot import partC_rev1_build_od_snapshot_from_sqlite
    database = Path(cfg["preprocess"]["cache_sqlite"])
    target = out / "snapshots"
    if target.exists():
        raise FileExistsError("Snapshots already exist; use a new output_root to rerun")
    with sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True) as conn:
        # Geometry, match-candidate strings and report columns are kept in GIS/SQL
        # outputs and need not be materialized in the numerical snapshot kernel.
        needed = ["directed_arc_id", "partC_rev1_link_uid", "from_node_uid", "to_node_uid", "partC_rev1_fftt_min",
                  "partC_rev1_capacity_annual_ktons_dir", "class_code", "alpha", "beta", "length_miles", "is_open", "is_valid_for_truck"]
        quoted = ", ".join('"' + name + '"' for name in needed)
        arcs = pd.read_sql_query(f'SELECT {quoted} FROM "partC_rev1_directed_arcs_enriched"', conn)
        endpoint = pd.read_sql_query('SELECT DISTINCT origin_node_uid AS uid FROM partC_rev1_od_centroid WHERE demand_solver_unit > 0 UNION SELECT DISTINCT destination_node_uid AS uid FROM partC_rev1_od_centroid WHERE demand_solver_unit > 0', conn)["uid"]
    # Upstream labels carry no partition assumption. Middle stages reconstruct
    # the frozen structural backbone and all region suites from stable IDs.
    arcs["arc_label"] = "physical"
    for name in ("partC_rev1_fftt_min", "partC_rev1_capacity_annual_ktons_dir"):
        values = pd.to_numeric(arcs[name], errors="coerce").to_numpy(float)
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"Invalid {name}: snapshot construction may not silently drop cleaned physical arcs")
    if arcs[["directed_arc_id", "from_node_uid", "to_node_uid"]].isna().any().any():
        raise ValueError("Missing stable arc/endpoint ID: cannot silently drop a cleaned physical arc")
    graph = partC_rev1_build_graph_snapshot_from_frames(arcs, od_endpoint_uids=endpoint,
        assignment_capacity_config=cfg["assignment_capacity_multipliers"],
        metadata={"stage": "unpartitioned_physical_input", "source_sqlite": str(database), "flow_unit": "annual_ktons",
                  "temporary_label_semantics": "physical is an upstream staging label; no backbone membership is inferred"})
    if not len(graph.tail) or len(np.unique(graph.arc_idx_to_directed_arc_id)) != len(graph.tail):
        raise ValueError("Invalid/empty graph snapshot")
    expected_usable = np.ones(len(arcs), dtype=bool)
    for name in ("is_open", "is_valid_for_truck"):
        if name in arcs:
            expected_usable &= pd.to_numeric(arcs[name], errors="coerce").fillna(1).to_numpy(int) == 1
    if len(graph.tail) != int(expected_usable.sum()):
        raise ValueError("Graph silently changed physical arc coverage beyond explicit closed/truck-prohibited exclusions")
    graph_paths = graph.save(target)
    od, od_paths = partC_rev1_build_od_snapshot_from_sqlite(database, node_uid_to_idx=graph.node_uid_to_idx,
                                                         chunksize=250_000, output_dir=target)
    if not len(od.demand) or not np.isfinite(od.demand).all() or np.any(od.demand <= 0):
        raise ValueError("Invalid/empty demand snapshot")
    versions = {
        "graph_npz": graph_paths["npz"], "graph_nodes_parquet": graph_paths["nodes"], "graph_arcs_parquet": graph_paths["arcs"],
        "graph_metadata_json": graph_paths["metadata"], "graph_identity": graph.graph_hash,
        "graph_node_count": len(graph.node_idx_to_uid), "graph_arc_count": len(graph.tail),
        "od_npz": od_paths["npz"], "od_metadata_json": od_paths["metadata"], "od_identity": od.od_hash,
        "od_row_count": len(od.demand), "od_total_demand": float(od.demand.sum()),
        "origin_count": len(od.origin_group_node_idx), "total_demand": float(od.demand.sum()),
        "raw_od_row_count": len(od.demand), "raw_od_total_demand": float(od.demand.sum()), "rich_sqlite": str(database),
    }
    result = {"graph_paths": graph_paths, "od_paths": od_paths, "input_versions": versions, "rich_sqlite": str(database),
              "graph_identity": graph.graph_hash, "graph_node_count": len(graph.node_idx_to_uid), "graph_arc_count": len(graph.tail),
              "raw_od_rows": len(od.demand), "raw_od_annual_ktons": float(od.demand.sum()), "staging_labels_only": True}
    write_json(out / "receipt.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--step", required=True, choices=["validate-inputs", "clean-network", "disaggregate-demand", "enrich-network", "snapshots"])
    args = parser.parse_args(argv)
    control, cfg, out, raw = config_for_stage(args.config)
    if args.step not in control["tasks"]:
        raise ValueError("Requested step is absent from configured explicit task list")
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / (args.step + "_status.json")
    if status_path.exists():
        raise FileExistsError(f"Stage status already exists; inspect it or choose a new output_root: {status_path}")
    start = time.perf_counter()
    status = {"phase_id": control["phase_id"], "step_id": args.step, "run_id": control["run_id"],
              "config_sha256": file_hash(args.config), "input_versions": control["input_versions"], "status": "RUNNING"}
    write_json(status_path, status)
    try:
        funcs = {"validate-inputs": validate_inputs, "clean-network": clean_network, "disaggregate-demand": disaggregate_demand,
                 "enrich-network": enrich_network, "snapshots": snapshots}
        result = funcs[args.step](cfg, out)
        status.update(status="COMPLETE", results=result)
    except BaseException as error:
        status.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        status["wall_seconds"] = time.perf_counter() - start
        import psutil
        status["resource_usage"] = {"rss_bytes_at_exit": psutil.Process().memory_info().rss,
                                     "output_bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file())}
        write_json(status_path, status)
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
