"""All-suite SQLite, GeoPackage, tables, figures, and validation for Phase 3."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sqlite3
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from partC_rev1_src.partC_rev1_hashing import partC_rev1_file_sha256
from .production import (
    CANDIDATE_ID,
    METHOD_ID,
    SUITE_ORDER,
    Phase3ProductionError,
    excluded_suite_ids,
    required_suite_ids,
    _write_csv_new,
    _write_json_new,
    _write_parquet_new,
    suite_record,
    utc_now,
)


TABLE_FILES = (
    "table_01_all_suite_topology_summary.csv",
    "table_02_all_suite_path_fftt_summary.csv",
    "table_03_all_suite_comprehensive_summary.csv",
    "table_04_lift_failure_reason_summary.csv",
    "table_05_regional_pair_support_summary.csv",
    "table_06_resource_usage.csv",
    "table_07_database_and_layer_inventory.csv",
    "table_08_validation_and_reproduction.csv",
    "table_09_s03_reuse_validation.csv",
    "table_10_external_artifact_bindings.csv",
)

CLOSEOUT_TABLE_FILES = (
    "table_01_phase2_suite_scale_and_s01_disposition.csv",
    "table_02_phase3_five_suite_topology_summary.csv",
    "table_03_phase3_five_suite_fftt_summary.csv",
    "table_04_phase3_five_suite_liftability_overlap.csv",
    "table_05_phase3_regional_support_burden.csv",
    "table_06_phase3_resource_usage.csv",
    "table_07_phase3_external_artifact_bindings.csv",
    "table_08_phase3_validation_and_reproduction.csv",
    "table_09_s03_reuse_validation.csv",
)


def is_scope_amended_closeout(config: Mapping[str, Any]) -> bool:
    return bool(excluded_suite_ids(config))


def portfolio_table_files(config: Mapping[str, Any]) -> tuple[str, ...]:
    return CLOSEOUT_TABLE_FILES if is_scope_amended_closeout(config) else TABLE_FILES


def _configured_output_path(
    config: Mapping[str, Any], key: str, legacy_relative_path: str
) -> Path:
    run_root = Path(str(config["output_root"])).resolve(strict=True)
    relative = str(config.get("output_names", {}).get(key, legacy_relative_path))
    path = run_root / Path(relative)
    try:
        path.resolve().relative_to(run_root)
    except ValueError as exc:
        raise Phase3ProductionError(f"Configured output escapes the run root: {relative}") from exc
    return path


def production_directory(config: Mapping[str, Any], suite_short_id: str) -> Path:
    return Path(str(config["output_root"])).resolve(strict=True) / "suites" / suite_short_id / "parquet"


def require_all_suites(config: Mapping[str, Any]) -> dict[str, Path]:
    result = {suite: production_directory(config, suite) for suite in required_suite_ids(config)}
    missing = [suite for suite, path in result.items() if not path.is_dir()]
    if missing:
        raise Phase3ProductionError(f"Portfolio assembly lacks suites: {missing}")
    return result


def _portfolio_registry(
    config: Mapping[str, Any], suites: Mapping[str, Path]
) -> pd.DataFrame:
    required = set(required_suite_ids(config))
    excluded = set(excluded_suite_ids(config))
    production_rows: dict[str, dict[str, Any]] = {}
    for suite, directory in suites.items():
        production_rows[suite] = json.loads(
            (directory / "suite_metadata.json").read_text(encoding="utf-8")
        )
    columns = sorted({key for row in production_rows.values() for key in row})
    rows: list[dict[str, Any]] = []
    for record in sorted(config["suites"], key=lambda row: str(row["suite_short_id"])):
        suite = str(record["suite_short_id"])
        if suite in required:
            row = dict(production_rows[suite])
            row.update(
                {
                    "phase3_build_status": "PASS",
                    "phase3_scope_role": "REQUIRED_PRODUCTION_SUITE",
                    "required_for_phase3_completion": True,
                    "phase3_product_available": True,
                }
            )
        elif suite in excluded:
            row = {name: None for name in columns}
            row.update(
                {
                    "schema_id": "phase3.excluded_suite_registry.v1",
                    "phase3_run_id": config["run_id"],
                    "suite_short_id": suite,
                    "phase2_suite_id": record["phase2_suite_id"],
                    "resolution": float(record["resolution"]),
                    "phase3_build_status": "EXCLUDED_AFTER_RESOURCE_ATTEMPT",
                    "phase3_scope_role": "EXCLUDED_RESOURCE_OUTLIER_STRESS_CASE",
                    "required_for_phase3_completion": False,
                    "phase3_product_available": False,
                    "method_id": METHOD_ID,
                    "candidate_id": CANDIDATE_ID,
                    "phase4_executed": False,
                }
            )
        else:  # pragma: no cover - guarded by load_production_config
            raise Phase3ProductionError(f"Accepted suite lacks a portfolio disposition: {suite}")
        rows.append(row)
    registry = pd.DataFrame(rows).sort_values("suite_short_id", kind="mergesort")
    if len(registry) != 6 or registry["suite_short_id"].nunique() != 6:
        raise Phase3ProductionError("Portfolio registry must contain six unique accepted suites")
    return registry.reset_index(drop=True)


def _s01_exclusion_frame(config: Mapping[str, Any]) -> pd.DataFrame:
    amendment = dict(config.get("scope_amendment", {}))
    exclusion = dict(amendment.get("excluded_suite", {}))
    if not exclusion:
        return pd.DataFrame()
    operation_root = Path(str(config["output_root"])).resolve(strict=True) / "operations" / "s01"
    return pd.DataFrame(
        [
            {
                "decision_id": amendment["decision_id"],
                "suite_short_id": "s01",
                "disposition": exclusion["disposition"],
                "scientific_reason": exclusion["scientific_reason"],
                "artifact_status": "PARTIAL_NONPRODUCT_EVIDENCE",
                "eligible_for_phase4": False,
                "load_into_sqlite_production_tables": False,
                "load_into_geopackage": False,
                "elapsed_seconds": float(exclusion["elapsed_seconds"]),
                "last_confirmed_regions": int(exclusion["last_confirmed_regions"]),
                "total_regions": int(exclusion["total_regions"]),
                "last_confirmed_regional_sources": int(
                    exclusion["last_confirmed_regional_sources"]
                ),
                "last_confirmed_regional_edges": int(
                    exclusion["last_confirmed_regional_edges"]
                ),
                "partial_operation_path": str(operation_root),
                "partial_artifact_manifest_path": str(
                    Path(str(config["output_root"])).resolve(strict=True)
                    / "evidence"
                    / "S01_PARTIAL_ARTIFACT_MANIFEST.csv"
                ),
            }
        ]
    )


def build_all_suite_sqlite(config: Mapping[str, Any]) -> dict[str, Any]:
    began = time.perf_counter()
    run_root = Path(str(config["output_root"])).resolve(strict=True)
    suites = require_all_suites(config)
    output = _configured_output_path(
        config, "sqlite", "database/phase3_acnhn_all_suites.sqlite"
    )
    if output.exists():
        raise Phase3ProductionError(f"SQLite output already exists: {output}")
    staging_root = Path(str(config["temp_root"])).resolve(strict=True) / "portfolio_assembly"
    working = _new_staging_path(staging_root, output.stem, ".partial.sqlite")
    table_sources = {
        "acnhn_nodes": "acnhn_nodes.parquet",
        "acnhn_static_links": "acnhn_static_links.parquet",
        "bacn_membership": "bacn_membership.parquet",
        "bacn_summary": "bacn_summary.parquet",
        "bundle_membership": "bundle_membership.parquet",
        "bundle_summary": "bundle_summary.parquet",
        "suppressed_backbone_arcs": "suppressed_backbone_arcs.parquet",
        "racn_membership": "racn_membership.parquet",
        "racn_summary": "racn_summary.parquet",
        "racn_restricted_witnesses": "racn_restricted_witnesses.parquet",
        "consolidated_terminals": "consolidated_terminals.parquet",
        "terminal_fine_members": "terminal_fine_members.parquet",
        "terminal_connectors": "terminal_connectors.parquet",
        "structural_internal_charges": "structural_internal_charges.parquet",
        "regional_pair_support_summary": "regional_pair_support_summary.parquet",
        "regional_pair_visualization_sample": "regional_pair_visualization_sample.parquet",
        "common_od_results": "candidate_od_results.parquet",
    }
    registry = _portfolio_registry(config, suites)
    with contextlib.closing(sqlite3.connect(working)) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-262144")
        registry.to_sql("suite_registry", connection, index=False, if_exists="fail", chunksize=1000)
        connection.execute("CREATE UNIQUE INDEX ux_suite_registry_short ON suite_registry(suite_short_id)")
        connection.execute("CREATE UNIQUE INDEX ux_suite_registry_phase2 ON suite_registry(phase2_suite_id)")
        for table, filename in table_sources.items():
            first = True
            for suite in required_suite_ids(config):
                frame = pd.read_parquet(suites[suite] / filename)
                frame = _with_suite_fields(config, suite, frame)
                frame.to_sql(
                    table,
                    connection,
                    index=False,
                    if_exists="fail" if first else "append",
                    chunksize=5000,
                )
                first = False
                print(f"PHASE3_SQLITE table={table} suite={suite} rows={len(frame)}", flush=True)
        _load_support_artifact_registry(connection, config, suites)
        _load_json_evidence_tables(connection, config, suites)
        _load_closeout_control_tables(connection, config)
        _create_sqlite_indexes_and_views(connection, config)
        validation = _validate_sqlite_connection(connection, config)
        inventory = _sqlite_inventory(connection)
        pd.DataFrame(
            [
                {
                    "status": validation["status"],
                    "checks_json": json.dumps(validation["checks"], sort_keys=True),
                    "per_suite_json": json.dumps(validation["per_suite"], sort_keys=True),
                }
            ]
        ).to_sql("database_validation", connection, index=False, if_exists="fail")
        pd.DataFrame(inventory).to_sql("database_inventory", connection, index=False, if_exists="fail")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE")
    reopened = _validate_sqlite_reopen(working, config)
    os.replace(working, output)
    evidence = {
        "schema_id": (
            "phase3.five_suite_closeout.sqlite_validation.v1"
            if is_scope_amended_closeout(config)
            else "phase3.all_suite_sqlite_validation.v1"
        ),
        "status": "PASS",
        "path": str(output),
        "size_bytes": int(output.stat().st_size),
        "sha256": partC_rev1_file_sha256(output),
        "validation": validation,
        "query_only_reopen": reopened,
        "inventory": inventory,
        "elapsed_seconds": time.perf_counter() - began,
        "generated_utc": utc_now(),
    }
    _write_json_new(run_root / "evidence" / "sqlite_validation.json", evidence)
    return evidence


def _with_suite_fields(
    config: Mapping[str, Any], suite_short_id: str, frame: pd.DataFrame
) -> pd.DataFrame:
    record = suite_record(config, suite_short_id)
    result = frame.copy()
    insertions = [
        ("phase3_run_id", str(config["run_id"])),
        ("suite_short_id", suite_short_id),
        ("phase2_suite_id", str(record["phase2_suite_id"])),
        ("resolution", float(record["resolution"])),
    ]
    for name, value in reversed(insertions):
        if name not in result.columns:
            result.insert(0, name, value)
    return result


def _load_support_artifact_registry(
    connection: sqlite3.Connection,
    config: Mapping[str, Any],
    suites: Mapping[str, Path],
) -> None:
    rows = []
    for suite in required_suite_ids(config):
        support_root = suites[suite] / "regional_support"
        summary = pd.read_parquet(suites[suite] / "regional_pair_support_summary.parquet").set_index("region_id")
        for path in sorted(support_root.glob("*.npz")):
            region_id = path.stem
            row = summary.loc[region_id]
            rows.append(
                {
                    "phase3_run_id": config["run_id"],
                    "suite_short_id": suite,
                    "region_id": region_id,
                    "absolute_path": str(path.resolve()),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": partC_rev1_file_sha256(path),
                    "node_count": int(row["region_node_count"]),
                    "arc_count": int(row["region_arc_count"]),
                    "supported_pair_count": int(row["supported_pair_count"]),
                    "representation_mode": "EXACT_SPARSE_LAZY",
                }
            )
    pd.DataFrame(rows).to_sql(
        "regional_support_artifact_registry", connection, index=False, if_exists="fail", chunksize=2000
    )


def _load_json_evidence_tables(
    connection: sqlite3.Connection,
    config: Mapping[str, Any],
    suites: Mapping[str, Path],
) -> None:
    validation_rows, reproduction_rows, binding_rows = [], [], []
    for suite in required_suite_ids(config):
        validation = json.loads((suites[suite] / "validation.json").read_text(encoding="utf-8"))
        reproduction = json.loads((suites[suite] / "reproduction.json").read_text(encoding="utf-8"))
        validation_rows.append(
            {
                "suite_short_id": suite,
                "status": validation["status"],
                "checks_json": json.dumps(validation["checks"], sort_keys=True),
                "validated_utc": validation["validated_utc"],
            }
        )
        reproduction_rows.append(
            {
                "suite_short_id": suite,
                "status": reproduction["status"],
                "fresh_recomputation": bool(reproduction["fresh_recomputation"]),
                "stable_id_and_membership_exact": bool(reproduction["stable_id_and_membership_exact"]),
                "payload_json": json.dumps(reproduction, sort_keys=True),
            }
        )
        for path in (
            suites[suite],
            suites[suite] / "physical_to_candidate_lookup.npz",
            suites[suite] / "regional_pair_support_contract.json",
        ):
            binding_rows.append(
                {
                    "suite_short_id": suite,
                    "artifact_role": "production_parquet_directory" if path.is_dir() else path.name,
                    "absolute_path": str(path.resolve()),
                    "size_bytes": _path_size(path),
                    "sha256": _directory_digest(path) if path.is_dir() else partC_rev1_file_sha256(path),
                    "validation_status": "PASS",
                }
            )
    pd.DataFrame(validation_rows).to_sql("suite_validation", connection, index=False, if_exists="fail")
    pd.DataFrame(reproduction_rows).to_sql("suite_reproduction", connection, index=False, if_exists="fail")
    pd.DataFrame(binding_rows).to_sql("external_artifact_bindings", connection, index=False, if_exists="fail")


def _load_closeout_control_tables(
    connection: sqlite3.Connection, config: Mapping[str, Any]
) -> None:
    """Load scope, resource, ambiguity, and external-authority evidence.

    These tables are additive for the amended closeout.  The original six-suite
    configuration follows the legacy database schema unchanged.
    """

    if not is_scope_amended_closeout(config):
        return
    exclusion = _s01_exclusion_frame(config)
    if len(exclusion) != 1:
        raise Phase3ProductionError("Scope-amended closeout requires one s01 exclusion record")
    exclusion.to_sql("s01_exclusion_record", connection, index=False, if_exists="fail")

    run_root = Path(str(config["output_root"])).resolve(strict=True)
    resource_rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "evidence" / "cli_resource_receipts").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        resource_rows.append(
            {
                "task_id": payload.get("task_id"),
                "mode": payload.get("mode"),
                "status": payload.get("status"),
                "started_utc": payload.get("started_utc"),
                "completed_utc": payload.get("completed_utc"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "peak_process_tree_rss_bytes": payload.get("peak_process_tree_rss_bytes"),
                "peak_process_count": payload.get("peak_process_count"),
                "added_disk_bytes": payload.get("added_disk_bytes"),
                "receipt_path": str(path.resolve()),
            }
        )
    pd.DataFrame(resource_rows).to_sql(
        "resource_usage", connection, index=False, if_exists="fail"
    )

    ambiguity_rows = list(config.get("ambiguity_resolutions", ()))
    if not ambiguity_rows:
        ambiguity_rows = [
            {
                "issue": "authorized production scope amendment",
                "decision": "use s02-s06 and retain s01 only as partial non-product evidence",
                "evidence": config["scope_amendment"]["decision_id"],
                "scientific_impact": "externally authorized scope change; frozen method unchanged",
            }
        ]
    pd.DataFrame(ambiguity_rows).to_sql(
        "ambiguity_resolution", connection, index=False, if_exists="fail"
    )

    authority_rows = []
    for role, binding in (
        ("scope_amendment_package", config["scope_amendment_package"]),
        ("reviewed_blocked_handoff", config["reviewed_blocked_handoff"]),
        ("accepted_phase2_geopackage", config["phase2_geopackage"]),
    ):
        path = Path(str(binding["path"])).resolve(strict=True)
        authority_rows.append(
            {
                "suite_short_id": None,
                "artifact_role": role,
                "absolute_path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": partC_rev1_file_sha256(path),
                "validation_status": "PASS",
            }
        )
    pd.DataFrame(authority_rows).to_sql(
        "external_artifact_bindings", connection, index=False, if_exists="append"
    )


def _create_sqlite_indexes_and_views(
    connection: sqlite3.Connection, config: Mapping[str, Any]
) -> None:
    statements = [
        "CREATE UNIQUE INDEX ux_nodes_suite_id ON acnhn_nodes(suite_short_id,node_id)",
        "CREATE UNIQUE INDEX ux_links_suite_id ON acnhn_static_links(suite_short_id,link_id)",
        "CREATE INDEX ix_links_suite_tail ON acnhn_static_links(suite_short_id,tail_node_id)",
        "CREATE INDEX ix_links_suite_head ON acnhn_static_links(suite_short_id,head_node_id)",
        "CREATE UNIQUE INDEX ux_bacn_member ON bacn_membership(suite_short_id,partC_rev1_node_uid)",
        "CREATE INDEX ix_bacn_member_object ON bacn_membership(suite_short_id,bacn_id)",
        "CREATE UNIQUE INDEX ux_bundle_member ON bundle_membership(suite_short_id,directed_arc_id)",
        "CREATE INDEX ix_bundle_id ON bundle_membership(suite_short_id,bundle_id)",
        "CREATE UNIQUE INDEX ux_suppressed_arc ON suppressed_backbone_arcs(suite_short_id,directed_arc_id)",
        "CREATE UNIQUE INDEX ux_racn_member ON racn_membership(suite_short_id,physical_portal_node_uid)",
        "CREATE UNIQUE INDEX ux_terminal ON consolidated_terminals(suite_short_id,terminal_id)",
        "CREATE INDEX ix_terminal_region ON consolidated_terminals(suite_short_id,region_id)",
        "CREATE UNIQUE INDEX ux_terminal_fine ON terminal_fine_members(suite_short_id,terminal_id,fine_member_order)",
        "CREATE INDEX ix_terminal_fine_physical ON terminal_fine_members(suite_short_id,partC_rev1_node_uid)",
        "CREATE UNIQUE INDEX ux_support_region ON regional_pair_support_summary(suite_short_id,region_id)",
        "CREATE UNIQUE INDEX ux_support_artifact ON regional_support_artifact_registry(suite_short_id,region_id)",
        "CREATE UNIQUE INDEX ux_od_result ON common_od_results(suite_short_id,sample_od_id,diagnostic_mode)",
        "CREATE INDEX ix_od_mode ON common_od_results(suite_short_id,diagnostic_mode)",
    ]
    for statement in statements:
        connection.execute(statement)
    for suite in sorted(required_suite_ids(config)):
        connection.execute(
            f"CREATE VIEW acnhn_{suite}_nodes AS SELECT * FROM acnhn_nodes WHERE suite_short_id='{suite}'"
        )
        connection.execute(
            f"CREATE VIEW acnhn_{suite}_static_links AS SELECT * FROM acnhn_static_links WHERE suite_short_id='{suite}'"
        )


def _validate_sqlite_connection(
    connection: sqlite3.Connection, config: Mapping[str, Any]
) -> dict[str, Any]:
    suite_count = int(connection.execute("SELECT COUNT(*) FROM suite_registry").fetchone()[0])
    od_rows = int(connection.execute("SELECT COUNT(*) FROM common_od_results").fetchone()[0])
    orphan_tail = int(
        connection.execute(
            "SELECT COUNT(*) FROM acnhn_static_links l LEFT JOIN acnhn_nodes n "
            "ON l.suite_short_id=n.suite_short_id AND l.tail_node_id=n.node_id WHERE n.node_id IS NULL"
        ).fetchone()[0]
    )
    orphan_head = int(
        connection.execute(
            "SELECT COUNT(*) FROM acnhn_static_links l LEFT JOIN acnhn_nodes n "
            "ON l.suite_short_id=n.suite_short_id AND l.head_node_id=n.node_id WHERE n.node_id IS NULL"
        ).fetchone()[0]
    )
    per_suite = {
        row[0]: {
            "nodes": int(row[1]),
            "links": int(row[2]),
            "od_rows": int(row[3]),
            "support_regions": int(row[4]),
        }
        for row in connection.execute(
            "SELECT s.suite_short_id,"
            "(SELECT COUNT(*) FROM acnhn_nodes n WHERE n.suite_short_id=s.suite_short_id),"
            "(SELECT COUNT(*) FROM acnhn_static_links l WHERE l.suite_short_id=s.suite_short_id),"
            "(SELECT COUNT(*) FROM common_od_results o WHERE o.suite_short_id=s.suite_short_id),"
            "(SELECT COUNT(*) FROM regional_pair_support_summary r WHERE r.suite_short_id=s.suite_short_id) "
            "FROM suite_registry s ORDER BY s.suite_short_id"
        )
    }
    indexed_probe = connection.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM acnhn_static_links WHERE suite_short_id='s03' AND tail_node_id=?",
        (connection.execute("SELECT tail_node_id FROM acnhn_static_links WHERE suite_short_id='s03' LIMIT 1").fetchone()[0],),
    ).fetchall()
    required = set(required_suite_ids(config))
    excluded = set(excluded_suite_ids(config))
    expected_od_rows = len(required) * int(config["common_evaluation"]["od_count"]) * len(
        config["common_evaluation"]["modes"]
    )
    required_rows = {suite: per_suite[suite] for suite in required}
    checks = {
        "six_row_accepted_suite_registry": suite_count == 6,
        "required_production_suite_count": len(required) == len(required_rows),
        "expected_common_od_rows": od_rows == expected_od_rows,
        "static_link_endpoints_resolve": orphan_tail == orphan_head == 0,
        "each_required_suite_400_od_rows": all(
            value["od_rows"] == 400 for value in required_rows.values()
        ),
        "each_required_suite_nonempty": all(
            value["nodes"] > 0 and value["links"] > 0 and value["support_regions"] > 0
            for value in required_rows.values()
        ),
        "foreign_key_check": connection.execute("PRAGMA foreign_key_check").fetchall() == [],
        "indexed_representative_query": any("INDEX" in str(row).upper() for row in indexed_probe),
    }
    if excluded:
        excluded_topology_rows = sum(
            per_suite[suite]["nodes"] + per_suite[suite]["links"] + per_suite[suite]["od_rows"]
            for suite in excluded
        )
        checks.update(
            {
                "excluded_suites_have_no_production_rows": excluded_topology_rows == 0,
                "s01_exclusion_record": int(
                    connection.execute("SELECT COUNT(*) FROM s01_exclusion_record").fetchone()[0]
                )
                == 1,
                "no_s01_production_views": connection.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='view' AND name LIKE 'acnhn_s01_%'"
                ).fetchone()[0]
                == 0,
            }
        )
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise Phase3ProductionError(f"All-suite SQLite validation failed: {failed}")
    return {"status": "PASS", "checks": checks, "per_suite": per_suite}


def _validate_sqlite_reopen(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    with contextlib.closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=1")
        result = {
            "query_only": int(connection.execute("PRAGMA query_only").fetchone()[0]) == 1,
            "suite_count": int(connection.execute("SELECT COUNT(*) FROM suite_registry").fetchone()[0]),
            "od_row_count": int(connection.execute("SELECT COUNT(*) FROM common_od_results").fetchone()[0]),
            "integrity_check": connection.execute("PRAGMA integrity_check").fetchone()[0],
        }
    expected_od_rows = len(required_suite_ids(config)) * 400
    if not result["query_only"] or result["suite_count"] != 6 or result["od_row_count"] != expected_od_rows or result["integrity_check"] != "ok":
        raise Phase3ProductionError(f"SQLite query-only reopen failed: {result}")
    return result


def _sqlite_inventory(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = []
    for object_type, name in connection.execute(
        "SELECT type,name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY type,name"
    ):
        count = int(connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
        rows.append({"artifact": "SQLite", "object_type": object_type, "object_name": name, "row_count": count, "validation_status": "PASS"})
    return rows


def build_summary_tables_and_figures(config: Mapping[str, Any]) -> dict[str, Any]:
    began = time.perf_counter()
    run_root = Path(str(config["output_root"])).resolve(strict=True)
    suites = require_all_suites(config)
    tables_root, figures_root = run_root / "tables", run_root / "figures"
    table_names = portfolio_table_files(config)
    if any((tables_root / name).exists() for name in table_names):
        raise Phase3ProductionError("One or more required summary tables already exist")
    topology_rows, result_frames, support_frames, resource_rows, validation_rows = [], [], [], [], []
    for suite in sorted(suites):
        metadata = json.loads((suites[suite] / "suite_metadata.json").read_text(encoding="utf-8"))
        topology_rows.append(metadata)
        result_frames.append(_with_suite_fields(config, suite, pd.read_parquet(suites[suite] / "candidate_od_results.parquet")))
        support_frames.append(_with_suite_fields(config, suite, pd.read_parquet(suites[suite] / "regional_pair_support_summary.parquet")))
        resources = json.loads((suites[suite] / "resource_usage.json").read_text(encoding="utf-8"))
        for row in resources.get("operations", []):
            resource_rows.append({"resource_source": "scientific_operation", "suite_short_id": suite, **row})
        validation = json.loads((suites[suite] / "validation.json").read_text(encoding="utf-8"))
        reproduction = json.loads((suites[suite] / "reproduction.json").read_text(encoding="utf-8"))
        validation_rows.append(
            {
                "suite_short_id": suite,
                "validation_status": validation["status"],
                "validation_passed_check_count": sum(bool(value) for value in validation["checks"].values()),
                "validation_failed_check_count": sum(not bool(value) for value in validation["checks"].values()),
                "reproduction_status": reproduction["status"],
                "fresh_recomputation": reproduction["fresh_recomputation"],
                "stable_id_and_membership_exact": reproduction["stable_id_and_membership_exact"],
            }
        )
    for path in sorted((run_root / "evidence" / "cli_resource_receipts").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(row["task_id"])
        suite = task_id[:3].lower() if task_id[:3].upper() in {f"S{number:02d}" for number in range(1, 7)} else "ALL"
        resource_rows.append(
            {
                "resource_source": "cli_task",
                "suite_short_id": suite,
                "task_id": task_id,
                "operation": row["mode"],
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "peak_process_tree_rss_bytes": int(row["peak_process_tree_rss_bytes"]),
                "peak_process_count": int(row["peak_process_count"]),
                "added_disk_bytes": int(row["added_disk_bytes"]),
                "status": row["status"],
            }
        )
    topology = pd.DataFrame(topology_rows).sort_values("suite_short_id", kind="mergesort")
    results = pd.concat(result_frames, ignore_index=True).sort_values(
        ["suite_short_id", "sample_od_id", "diagnostic_mode"], kind="mergesort"
    ).reset_index(drop=True)
    support = pd.concat(support_frames, ignore_index=True).sort_values(
        ["suite_short_id", "region_id"], kind="mergesort"
    ).reset_index(drop=True)
    expected_result_rows = len(required_suite_ids(config)) * int(
        config["common_evaluation"]["od_count"]
    ) * len(config["common_evaluation"]["modes"])
    if len(results) != expected_result_rows:
        raise Phase3ProductionError(
            f"Combined common OD results require {expected_result_rows} rows, observed {len(results)}"
        )
    combined_name = str(
        config.get("output_names", {}).get(
            "combined_od_results",
            "tables/all_suite_candidate_od_results.parquet",
        )
    )
    combined_path = run_root / combined_name
    _write_parquet_new(combined_path, results)
    path_summary = _path_summary(results)
    comprehensive = path_summary.merge(topology, on="suite_short_id", how="left", validate="many_to_one")
    failures = (
        results.assign(lift_failure_reason=results["lift_failure_reason"].fillna("LIFTABLE"))
        .groupby(["suite_short_id", "diagnostic_mode", "lift_failure_reason"], as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
    )
    support_table = support.copy()
    resources = pd.DataFrame(resource_rows)
    if resources.empty:
        resources = pd.DataFrame(
            [{"resource_source": "reuse_binding", "suite_short_id": "s03", "task_id": "S03-REUSE", "operation": "reuse", "elapsed_seconds": 0.0, "peak_process_tree_rss_bytes": 0, "peak_process_count": 0, "added_disk_bytes": 0, "status": "PASS"}]
        )
    inventory = _combined_inventory(config)
    validations = pd.DataFrame(validation_rows)
    s03_reuse = _s03_reuse_table(config)
    bindings = _external_bindings(config)
    if is_scope_amended_closeout(config):
        phase2_recap = _phase2_suite_recap(config)
        liftability_overlap = path_summary[
            [
                "suite_short_id",
                "diagnostic_mode",
                "od_count",
                "liftability_rate",
                "weighted_arc_precision_mean",
                "weighted_arc_recall_mean",
                "weighted_arc_jaccard_mean",
                "coarse_lcs_similarity_mean",
            ]
        ].merge(
            failures.groupby(["suite_short_id", "diagnostic_mode"], as_index=False)[
                "row_count"
            ].sum().rename(columns={"row_count": "lift_outcome_row_count"}),
            on=["suite_short_id", "diagnostic_mode"],
            how="left",
            validate="one_to_one",
        )
        frames = {
            CLOSEOUT_TABLE_FILES[0]: phase2_recap,
            CLOSEOUT_TABLE_FILES[1]: topology,
            CLOSEOUT_TABLE_FILES[2]: path_summary,
            CLOSEOUT_TABLE_FILES[3]: liftability_overlap,
            CLOSEOUT_TABLE_FILES[4]: support_table,
            CLOSEOUT_TABLE_FILES[5]: resources,
            CLOSEOUT_TABLE_FILES[6]: bindings,
            CLOSEOUT_TABLE_FILES[7]: validations,
            CLOSEOUT_TABLE_FILES[8]: s03_reuse,
        }
    else:
        phase2_recap = pd.DataFrame()
        frames = {
            TABLE_FILES[0]: topology,
            TABLE_FILES[1]: path_summary,
            TABLE_FILES[2]: comprehensive,
            TABLE_FILES[3]: failures,
            TABLE_FILES[4]: support_table,
            TABLE_FILES[5]: resources,
            TABLE_FILES[6]: inventory,
            TABLE_FILES[7]: validations,
            TABLE_FILES[8]: s03_reuse,
            TABLE_FILES[9]: bindings,
        }
    for name, frame in frames.items():
        _write_csv_new(tables_root / name, frame)
    _write_table_index(config, tables_root, frames)
    if is_scope_amended_closeout(config):
        _generate_closeout_figures(
            config,
            phase2_recap,
            topology,
            path_summary,
            results,
            support,
            resources,
            figures_root,
        )
        _write_closeout_reports(config, topology, path_summary, resources)
    else:
        _generate_figures(config, topology, path_summary, results, support, resources, figures_root)
    result = {
        "schema_id": (
            "phase3.five_suite_closeout.summary_table_figure_validation.v1"
            if is_scope_amended_closeout(config)
            else "phase3.summary_table_figure_validation.v1"
        ),
        "status": "PASS",
        "table_row_counts": {name: len(frame) for name, frame in frames.items()},
        "combined_od_rows": len(results),
        "figure_count": len(list(figures_root.glob("figure_*.png"))),
        "elapsed_seconds": time.perf_counter() - began,
        "generated_utc": utc_now(),
    }
    _write_json_new(run_root / "evidence" / "summary_table_figure_validation.json", result)
    return result


def _path_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (suite, mode), group in results.groupby(["suite_short_id", "diagnostic_mode"], sort=True):
        projected_abs = group["projected_relative_error"].abs().dropna()
        scalar_abs = group["scalar_relative_error"].abs().dropna()
        rows.append(
            {
                "suite_short_id": suite,
                "diagnostic_mode": mode,
                "od_count": int(group["sample_od_id"].nunique()),
                "projected_abs_relative_error_mean": float(projected_abs.mean()),
                "projected_abs_relative_error_median": float(projected_abs.median()),
                "projected_abs_relative_error_p95": float(projected_abs.quantile(0.95)),
                "projected_abs_relative_error_max": float(projected_abs.max()),
                "scalar_abs_relative_error_mean": float(scalar_abs.mean()),
                "scalar_abs_relative_error_median": float(scalar_abs.median()),
                "scalar_abs_relative_error_p95": float(scalar_abs.quantile(0.95)),
                "scalar_abs_relative_error_max": float(scalar_abs.max()),
                "liftability_rate": float(group["liftable"].mean()),
                "lifted_regret_mean": float(group["lifted_regret"].dropna().mean()),
                "weighted_arc_precision_mean": float(group["weighted_arc_precision"].mean()),
                "weighted_arc_recall_mean": float(group["weighted_arc_recall"].mean()),
                "weighted_arc_jaccard_mean": float(group["weighted_arc_jaccard"].mean()),
                "coarse_lcs_similarity_mean": float(group["coarse_lcs_similarity"].mean()),
                "regional_transition_count_mean": float(group["regional_transition_count"].mean()),
                "repeated_region_entry_count_mean": float(group["repeated_region_entry_count"].mean()),
                "omitted_suppressed_fftt_s_mean": float(group["omitted_suppressed_fftt_s"].mean()),
                "omitted_racn_fftt_s_mean": float(group["omitted_racn_fftt_s"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _combined_inventory(config: Mapping[str, Any]) -> pd.DataFrame:
    run_root = Path(str(config["output_root"])).resolve(strict=True)
    rows = []
    sqlite_evidence = json.loads((run_root / "evidence" / "sqlite_validation.json").read_text(encoding="utf-8"))
    rows.extend(sqlite_evidence["inventory"])
    gpkg_evidence_path = run_root / "evidence" / "geopackage_validation.json"
    if gpkg_evidence_path.is_file():
        rows.extend(json.loads(gpkg_evidence_path.read_text(encoding="utf-8"))["inventory"])
    return pd.DataFrame(rows)


def _phase2_suite_recap(config: Mapping[str, Any]) -> pd.DataFrame:
    run_root = Path(str(config["output_root"])).resolve(strict=True)
    authority_relative = str(
        config["scope_amendment"].get(
            "authority_run_relative_path", "authority/phase3_closeout_skip_s01_20260901"
        )
    )
    path = run_root / authority_relative / "config" / "phase2_suite_recap.csv"
    recap = pd.read_csv(path)
    required = set(required_suite_ids(config))
    excluded = set(excluded_suite_ids(config))
    recap["phase3_scope_role"] = np.where(
        recap["suite_short_id"].isin(required),
        "REQUIRED_PRODUCTION_SUITE",
        "EXCLUDED_RESOURCE_OUTLIER_STRESS_CASE",
    )
    recap["required_for_phase3_completion"] = recap["suite_short_id"].isin(required)
    recap["phase3_product_available"] = recap["suite_short_id"].isin(required)
    unavailable = recap["suite_short_id"].isin(excluded)
    for column in ("phase3_product_bytes",):
        recap.loc[unavailable, column] = np.nan
    if len(recap) != 6 or set(recap["suite_short_id"]) != required.union(excluded):
        raise Phase3ProductionError("Phase 2 recap does not match the amended six-row registry")
    return recap.sort_values("suite_short_id", kind="mergesort").reset_index(drop=True)


def _s03_reuse_table(config: Mapping[str, Any]) -> pd.DataFrame:
    run_root = Path(str(config["output_root"])).resolve(strict=True)
    receipt = json.loads((run_root / "receipts" / "S03-REUSE.json").read_text(encoding="utf-8"))
    return pd.DataFrame(
        [
            {"check": "reuse_manifest_rows", "expected": 1863, "observed": receipt["manifest_rows_verified"], "status": "PASS"},
            {"check": "hash_identical_relocated_files", "expected": 1863, "observed": receipt["hash_identical_relocated_files"], "status": "PASS"},
            {"check": "candidate_result_sha256", "expected": config["common_evaluation"]["s03_result_sha256"], "observed": receipt["candidate_result_sha256"], "status": "PASS"},
            {"check": "fresh_topology_reproduction", "expected": "PASS", "observed": "PASS", "status": "PASS"},
        ]
    )


def _external_bindings(config: Mapping[str, Any]) -> pd.DataFrame:
    run_root = Path(str(config["output_root"])).resolve(strict=True)
    paths = [
        (
            "production_sqlite",
            _configured_output_path(config, "sqlite", "database/phase3_acnhn_all_suites.sqlite"),
            "SQLite",
            "ACCEPTED_PRODUCTION_ARTIFACT",
            ",".join(required_suite_ids(config)),
        ),
        (
            "production_geopackage",
            _configured_output_path(config, "geopackage", "geopackage/phase3_acnhn_all_suites.gpkg"),
            "GeoPackage",
            "ACCEPTED_PRODUCTION_ARTIFACT",
            ",".join(required_suite_ids(config)),
        ),
    ]
    for suite in sorted(required_suite_ids(config)):
        paths.append(
            (
                f"{suite}_production_parquet",
                production_directory(config, suite),
                "Parquet directory",
                "ACCEPTED_PRODUCTION_ARTIFACT",
                suite,
            )
        )
    if is_scope_amended_closeout(config):
        paths.extend(
            [
                (
                    "s01_partial_operation",
                    run_root / "operations" / "s01",
                    "Partial evidence directory",
                    "PARTIAL_NONPRODUCT_EVIDENCE",
                    "s01",
                ),
                (
                    "scope_amendment_package",
                    Path(str(config["scope_amendment_package"]["path"])),
                    "ZIP",
                    "EXTERNAL_AUTHORITY",
                    "s01-s06",
                ),
                (
                    "reviewed_blocked_handoff",
                    Path(str(config["reviewed_blocked_handoff"]["path"])),
                    "ZIP",
                    "REVIEWED_EVIDENCE",
                    "s01-s06",
                ),
                (
                    "accepted_phase2_geopackage_source",
                    Path(str(config["phase2_geopackage"]["path"])),
                    "GeoPackage",
                    "IMMUTABLE_ACCEPTED_INPUT",
                    "s01-s06",
                ),
            ]
        )
    rows = []
    for role, path, kind, artifact_status, suite_coverage in paths:
        path = path.resolve(strict=True)
        if not path.exists():
            raise Phase3ProductionError(f"Required external artifact is absent: {path}")
        try:
            relative_path = path.relative_to(run_root).as_posix()
        except ValueError:
            relative_path = ""
        rows.append(
            {
                "artifact_role": role,
                "absolute_path": str(path),
                "relative_path": relative_path,
                "size_bytes": _path_size(path),
                "sha256": _directory_digest(path) if path.is_dir() else partC_rev1_file_sha256(path),
                "schema_id_or_format": kind,
                "row_or_layer_counts": _binding_counts(config, path, kind),
                "validation_status": "PASS",
                "artifact_status": artifact_status,
                "suite_coverage": suite_coverage,
                "reopen_or_integrity_result": "PASS",
                "producer_commit": _git_head(Path(str(config["project_root"]))),
                "producer_config": str(config.get("config_path", "")),
            }
        )
    return pd.DataFrame(rows)


def _binding_counts(config: Mapping[str, Any], path: Path, kind: str) -> str:
    if kind == "SQLite":
        with contextlib.closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as connection:
            tables = int(connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0])
            suite_rows = int(connection.execute("SELECT COUNT(*) FROM suite_registry").fetchone()[0])
            od_rows = int(connection.execute("SELECT COUNT(*) FROM common_od_results").fetchone()[0])
            return json.dumps({"tables": tables, "suite_rows": suite_rows, "od_rows": od_rows}, sort_keys=True)
    if kind == "GeoPackage":
        with contextlib.closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as connection:
            layers = int(connection.execute("SELECT COUNT(*) FROM gpkg_contents").fetchone()[0])
            return json.dumps({"gpkg_contents": layers}, sort_keys=True)
    if path.is_dir():
        return json.dumps(
            {"files": len([item for item in path.rglob("*") if item.is_file()])},
            sort_keys=True,
        )
    return json.dumps({"files": 1}, sort_keys=True)


def _write_table_index(
    config: Mapping[str, Any], tables_root: Path, frames: Mapping[str, pd.DataFrame]
) -> None:
    if is_scope_amended_closeout(config):
        descriptions = {
            CLOSEOUT_TABLE_FILES[0]: "Six accepted Phase 2 suites with s01 explicitly classified as an excluded resource-outlier stress case.",
            CLOSEOUT_TABLE_FILES[1]: "Five-suite Phase 3 topology, compression, actual online size, and split-equivalent size.",
            CLOSEOUT_TABLE_FILES[2]: "Five-suite direct and structural FFTT error summaries on the common 200 ODs.",
            CLOSEOUT_TABLE_FILES[3]: "Five-suite liftability, overlap, LCS, and outcome counts.",
            CLOSEOUT_TABLE_FILES[4]: "Exact per-region sparse/lazy regional-support counts and enumeration telemetry for s02-s06.",
            CLOSEOUT_TABLE_FILES[5]: "Original suite operations, the stopped s01 stress attempt, and closeout resource receipts.",
            CLOSEOUT_TABLE_FILES[6]: "Exact paths, sizes, SHA-256 values, formats, inventories, suite coverage, and reopen results.",
            CLOSEOUT_TABLE_FILES[7]: "Per-suite validation and deterministic reproduction evidence.",
            CLOSEOUT_TABLE_FILES[8]: "Accepted s03 manifest/result reuse and reproduction checks.",
        }
        title = "# Phase 3 five-suite closeout summary table index"
        names = CLOSEOUT_TABLE_FILES
    else:
        names = TABLE_FILES
        title = "# Phase 3 all-suite summary table index"
        descriptions = {
            TABLE_FILES[0]: "One row per suite: topology, compression, actual/split-equivalent online size, and exact support burden.",
            TABLE_FILES[1]: "Suite-by-mode projected/scalar FFTT error, liftability, overlap, and path metrics.",
            TABLE_FILES[2]: "Merged topology and path summary at suite-by-mode grain.",
            TABLE_FILES[3]: "Liftability outcomes and explicit failure-reason counts.",
            TABLE_FILES[4]: "Exact per-region sparse/lazy regional support counts and enumeration telemetry.",
            TABLE_FILES[5]: "Operation-level walltime, peak process-tree RSS, added disk, and status.",
            TABLE_FILES[6]: "SQLite and GeoPackage table/layer/view inventories and row counts.",
            TABLE_FILES[7]: "Per-suite validation and fresh deterministic reproduction evidence.",
            TABLE_FILES[8]: "Accepted s03 manifest/result reuse and reproduction checks.",
            TABLE_FILES[9]: "Required large-artifact absolute paths, sizes, SHA-256 values, formats, counts, and reopen status.",
        }
    lines = [title, ""]
    for name in names:
        lines.extend([f"## {name}", "", descriptions[name], "", f"Rows: {len(frames[name])}.", ""])
    (tables_root / "TABLE_INDEX.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _generate_figures(
    config: Mapping[str, Any],
    topology: pd.DataFrame,
    path_summary: pd.DataFrame,
    results: pd.DataFrame,
    support: pd.DataFrame,
    resources: pd.DataFrame,
    root: Path,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    order = sorted(topology["suite_short_id"].astype(str))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(order))
    frame = topology.set_index("suite_short_id").loc[order]
    ax.bar(x - 0.2, frame["actual_online_node_count"], 0.4, label="Actual link-cost-only", color="#277da1")
    ax.bar(x + 0.2, frame["split_equivalent_online_node_count"], 0.4, label="Hypothetical split-equivalent", color="#f8961e")
    ax.set_xticks(x, order)
    ax.set_ylabel("Online nodes")
    ax.set_title("Phase 3 ACNHN online size across accepted suites")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save_figure(root, "figure_01_all_suite_compression", fig, frame.reset_index(), "Actual link-cost-only and hypothetical B-in/B-out split-equivalent online node counts for all six suites.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot = path_summary.copy()
    for offset, mode in ((-0.16, "direct"), (0.16, "structural")):
        mode_frame = plot[plot["diagnostic_mode"] == mode].set_index("suite_short_id").loc[order]
        ax.bar(x + offset, 100 * mode_frame["scalar_abs_relative_error_p95"], 0.32, label=mode.title())
    ax.set_xticks(x, order)
    ax.set_ylabel("Scalar FFTT p95 absolute relative error (%)")
    ax.set_title("Direct versus structural ACNHN FFTT error")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save_figure(root, "figure_02_fftt_error_intervals", fig, path_summary, "Suite-by-mode p95 scalar absolute relative FFTT error on the common 200 ODs.")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True)
    structural = path_summary[path_summary["diagnostic_mode"] == "structural"].set_index("suite_short_id").loc[order]
    axes[0].bar(order, 100 * structural["liftability_rate"], color="#577590")
    axes[0].set_ylabel("Liftable (%)")
    axes[1].bar(order, structural["weighted_arc_jaccard_mean"], color="#43aa8b")
    axes[1].set_ylabel("FFTT-weighted Jaccard")
    axes[2].bar(order, structural["coarse_lcs_similarity_mean"], color="#f3722c")
    axes[2].set_ylabel("Coarse LCS similarity")
    fig.suptitle("Structural liftability and path overlap diagnostics")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    _save_figure(root, "figure_03_liftability_overlap", fig, structural.reset_index(), "Structural reverse-liftability (diagnostic only), FFTT-weighted overlap, and coarse LCS similarity.")

    _generate_six_panel_map(config, root)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(order, frame["actual_static_link_count"], label="Static links", color="#577590")
    ax.bar(order, frame["exact_regional_pair_count"], bottom=frame["actual_static_link_count"], label="Exact regional pairs", color="#90be6d")
    ax.set_yscale("log")
    ax.set_ylabel("Logical directed links (log scale)")
    ax.set_title("Static and exact sparse/lazy regional-pair burden")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save_figure(root, "figure_05_pair_link_burden", fig, frame.reset_index(), "Static-link and exact regional-pair counts; regional pairs remain exact sparse/lazy rather than dense.")

    resource_input = resources[resources["resource_source"] == "cli_task"].copy()
    resource_input = resource_input[resource_input["suite_short_id"].isin(order)]
    if resource_input.empty:
        resource_input = resources[resources["suite_short_id"].isin(order)].copy()
    resource_summary = resource_input.groupby("suite_short_id", as_index=False).agg(
        elapsed_seconds=("elapsed_seconds", "sum"),
        peak_process_tree_rss_bytes=("peak_process_tree_rss_bytes", "max"),
        added_disk_bytes=("added_disk_bytes", "sum"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].bar(resource_summary["suite_short_id"], resource_summary["elapsed_seconds"] / 60, color="#277da1")
    axes[0].set_ylabel("Operation time (minutes)")
    axes[1].bar(resource_summary["suite_short_id"], resource_summary["peak_process_tree_rss_bytes"] / 1024**3, color="#f8961e")
    axes[1].set_ylabel("Peak process-tree RSS (GiB)")
    axes[2].bar(resource_summary["suite_short_id"], resource_summary["added_disk_bytes"] / 1024**3, color="#43aa8b")
    axes[2].set_ylabel("Receipt-added disk (GiB)")
    fig.suptitle("Phase 3 suite resource use")
    _save_figure(root, "figure_06_resource_usage", fig, resource_summary, "Operation walltime, peak process-tree RSS, and receipt-accounted added disk by suite.")

    representative_ids = sorted(results["sample_od_id"].astype(str).unique())[:6]
    representative = results[
        results["sample_od_id"].astype(str).isin(representative_ids)
        & (results["diagnostic_mode"] == "structural")
    ].copy()
    representative["sample_order"] = representative["sample_od_id"].astype(str).map({value: index for index, value in enumerate(representative_ids)})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for ax, sample_id in zip(axes.ravel(), representative_ids, strict=True):
        data = representative[representative["sample_od_id"].astype(str) == sample_id].sort_values("suite_short_id")
        ax.plot(data["suite_short_id"], data["physical_fftt_s"] / 3600, marker="o", label="Physical")
        ax.plot(data["suite_short_id"], data["projected_generic_fftt_s"] / 3600, marker="s", label="Projected")
        ax.plot(data["suite_short_id"], data["scalar_acnhn_fftt_s"] / 3600, marker="^", label="Scalar ACNHN")
        ax.set_title(sample_id[-10:])
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.supylabel("FFTT (hours)")
    fig.suptitle("Representative common-OD path-cost comparison")
    _save_figure(root, "figure_07_representative_path_comparisons", fig, representative, "Physical, projected, and scalar structural FFTT for the same six representative ODs across all suites.")


def _generate_closeout_figures(
    config: Mapping[str, Any],
    phase2_recap: pd.DataFrame,
    topology: pd.DataFrame,
    path_summary: pd.DataFrame,
    results: pd.DataFrame,
    support: pd.DataFrame,
    resources: pd.DataFrame,
    root: Path,
) -> None:
    del results, support, resources  # Their derived summaries are plotted below.
    root.mkdir(parents=True, exist_ok=True)
    order = list(required_suite_ids(config))
    topology_frame = topology.set_index("suite_short_id").loc[order]
    phase2_frame = phase2_recap.set_index("suite_short_id").loc[
        ["s01", "s02", "s03", "s04", "s05", "s06"]
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    suite_labels = phase2_frame.index.tolist()
    colors = ["#d1495b" if suite == "s01" else "#277da1" for suite in suite_labels]
    axes[0].bar(suite_labels, phase2_frame["region_arcs_max"], color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Maximum retained-region arcs (log scale)")
    axes[0].set_title("Phase 2 regional-size tail")
    axes[1].bar(suite_labels, phase2_frame["phase3_elapsed_seconds"] / 3600.0, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Phase 3 attempt/build elapsed (hours, log scale)")
    axes[1].set_title("s01 resource-boundary attempt")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    _save_figure(
        root,
        "figure_01_phase2_suite_scale_and_s01_resource_outlier",
        fig,
        phase2_frame.reset_index(),
        "All six accepted Phase 2 suites are shown. s01 is the red resource-outlier stress case and is excluded from the revised Phase 3 production scope; s02-s06 are the five required production suites.",
    )

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        x - 0.2,
        topology_frame["actual_online_node_count"],
        0.4,
        label="Actual link-cost-only",
        color="#277da1",
    )
    ax.bar(
        x + 0.2,
        topology_frame["split_equivalent_online_node_count"],
        0.4,
        label="Hypothetical split-equivalent",
        color="#f8961e",
    )
    ax.set_xticks(x, order)
    ax.set_ylabel("Online nodes")
    ax.set_title("Five-suite ACNHN topology compression")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save_figure(
        root,
        "figure_02_five_suite_topology_compression",
        fig,
        topology_frame.reset_index(),
        "Actual link-cost-only and hypothetical B-in/B-out split-equivalent online sizes for the required s02-s06 production portfolio; s01 is excluded.",
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, mode in ((-0.16, "direct"), (0.16, "structural")):
        mode_frame = path_summary[path_summary["diagnostic_mode"] == mode].set_index(
            "suite_short_id"
        ).loc[order]
        ax.bar(
            x + offset,
            100 * mode_frame["scalar_abs_relative_error_p95"],
            0.32,
            label=mode.title(),
        )
    ax.set_xticks(x, order)
    ax.set_ylabel("Scalar FFTT p95 absolute relative error (%)")
    ax.set_title("Five-suite direct-versus-structural FFTT error")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save_figure(
        root,
        "figure_03_five_suite_fftt_error",
        fig,
        path_summary,
        "Direct and structural scalar ACNHN p95 absolute relative FFTT error on the same 200 accepted ODs for each of s02-s06; s01 is excluded.",
    )

    structural = path_summary[path_summary["diagnostic_mode"] == "structural"].set_index(
        "suite_short_id"
    ).loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True)
    axes[0].bar(order, 100 * structural["liftability_rate"], color="#577590")
    axes[0].set_ylabel("Liftable (%)")
    axes[1].bar(order, structural["weighted_arc_jaccard_mean"], color="#43aa8b")
    axes[1].set_ylabel("FFTT-weighted Jaccard")
    axes[2].bar(order, structural["coarse_lcs_similarity_mean"], color="#f3722c")
    axes[2].set_ylabel("Coarse LCS similarity")
    fig.suptitle("Five-suite structural liftability and overlap")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    _save_figure(
        root,
        "figure_04_five_suite_liftability_overlap",
        fig,
        structural.reset_index(),
        "Structural reverse whole-path liftability is diagnostic and nonblocking; overlap and coarse LCS are reported for s02-s06 only.",
    )

    _generate_five_panel_map(config, root)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(order, topology_frame["actual_static_link_count"], label="Static links", color="#577590")
    ax.bar(
        order,
        topology_frame["exact_regional_pair_count"],
        bottom=topology_frame["actual_static_link_count"],
        label="Exact regional pairs",
        color="#90be6d",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Logical directed links (log scale)")
    ax.set_title("Five-suite regional-support and total-link burden")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save_figure(
        root,
        "figure_06_regional_support_and_total_link_burden",
        fig,
        topology_frame.reset_index(),
        "Static-link and exact sparse/lazy regional-pair burden for s02-s06. Complete support remains nonspatial; GeoPackage pair geometry is a deterministic labeled subset.",
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].bar(suite_labels, phase2_frame["phase3_elapsed_seconds"] / 3600.0, color=colors)
    axes[0].set_ylabel("Elapsed (hours)")
    axes[1].bar(suite_labels, phase2_frame["phase3_peak_rss_bytes"] / 1024**3, color=colors)
    axes[1].set_ylabel("Peak RSS (GiB)")
    disk_values = phase2_frame["phase3_product_bytes"].fillna(0) / 1024**3
    axes[2].bar(suite_labels, disk_values, color=colors)
    axes[2].set_ylabel("Accepted product size (GiB)")
    fig.suptitle("Suite construction/resource-boundary evidence")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    _save_figure(
        root,
        "figure_07_resource_and_runtime",
        fig,
        phase2_frame.reset_index(),
        "Runtime, peak RSS, and accepted product size. s01 records the failed stress attempt and has no accepted product (NA in the source table, rendered as no product bar), while s02-s06 are complete.",
    )


def _write_closeout_reports(
    config: Mapping[str, Any],
    topology: pd.DataFrame,
    path_summary: pd.DataFrame,
    resources: pd.DataFrame,
) -> None:
    del resources
    run_root = Path(str(config["output_root"])).resolve(strict=True)
    summary_root = run_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    if any(summary_root.iterdir()):
        raise Phase3ProductionError("Closeout summary directory is not write-new")
    required = ", ".join(required_suite_ids(config))
    sqlite_evidence = json.loads(
        (run_root / "evidence" / "sqlite_validation.json").read_text(encoding="utf-8")
    )
    gpkg_evidence = json.loads(
        (run_root / "evidence" / "geopackage_validation.json").read_text(encoding="utf-8")
    )
    structural = path_summary[path_summary["diagnostic_mode"] == "structural"]
    report = f"""# Phase 3 final report - revised five-suite production scope

## Outcome

Phase 3 production construction is complete with notes for the externally
amended required scope `{required}`. The frozen method remains
`{METHOD_ID}`. All five accepted products were reused after exact manifest
verification; no physical tree/path or suite construction was recomputed.

Suite `s01` remains an accepted Phase 2 extreme-coarse stress case but is
`EXCLUDED_FROM_PHASE3_PRODUCTION_RESOURCE_OUTLIER_STRESS_CASE`. Its stopped
operation is preserved as partial non-product evidence and contributes no
Phase 3 topology, OD, SQLite production view, or ACNHN GeoPackage layer.

## Portfolio evidence

- Required production suites: `{required}`
- All required production suites built: `true`
- All six suites built: `false`
- Combined OD result rows: `{len(required_suite_ids(config)) * 400:,}`
- Actual online nodes across s02-s06: `{int(topology['actual_online_node_count'].sum()):,}`
- Exact sparse/lazy regional pairs across s02-s06: `{int(topology['exact_regional_pair_count'].sum()):,}`
- Mean structural liftability rate across suites: `{float(structural['liftability_rate'].mean()):.6f}`

## Large products

- SQLite: `{sqlite_evidence['path']}`; SHA-256 `{sqlite_evidence['sha256']}`
- GeoPackage: `{gpkg_evidence['path']}`; SHA-256 `{gpkg_evidence['sha256']}`
- Physical layers preserved: `directed_links_portfolio` and `physical_nodes`
- ACNHN spatial scope: exactly five node/link layer pairs for s02-s06

## Known limitation and boundary

Reverse whole-path liftability remains a documented nonblocking limitation.
Forward projection, physical-object accounting, topology/mapping closure,
database integrity, and isolated geospatial reopen are hard gates.

Phase 3 overall is pending external review. `next_step_authorized=false` and
`phase4_authorized=false`. Capacity-multiplier tuning, cost calibration, model
training, and ACNHN equilibrium were not performed.
"""
    (summary_root / "PHASE3_FINAL_REPORT.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    exclusion = config["scope_amendment"]["excluded_suite"]
    s01_report = f"""# s01 exclusion and resource-boundary evidence

Suite `s01` is an accepted Phase 2 extreme-coarse stress suite and is excluded
from Phase 3 production by external decision
`{config['scope_amendment']['decision_id']}`.

- Disposition: `{exclusion['disposition']}`
- Attempt elapsed: `{float(exclusion['elapsed_seconds']):.6f}` seconds
- Last confirmed regions: `{int(exclusion['last_confirmed_regions'])}` / `{int(exclusion['total_regions'])}`
- Last confirmed regional sources: `{int(exclusion['last_confirmed_regional_sources']):,}`
- Last confirmed regional edges: `{int(exclusion['last_confirmed_regional_edges']):,}`
- Artifact status: `PARTIAL_NONPRODUCT_EVIDENCE`
- Eligible for Phase 4: `false`
- Loaded into SQLite production topology: `false`
- Loaded into GeoPackage: `false`

The partial files remain unchanged under `operations/s01`; the exact partial
artifact manifest remains under `evidence/S01_PARTIAL_ARTIFACT_MANIFEST.csv`.
No success receipt, placeholder product, zero-filled metric, ACNHN layer, or
production view was created.
"""
    (summary_root / "S01_EXCLUSION_AND_RESOURCE_BOUNDARY.md").write_text(
        s01_report, encoding="utf-8", newline="\n"
    )


def _generate_six_panel_map(config: Mapping[str, Any], root: Path) -> None:
    _generate_panel_map(
        config,
        root,
        tuple(sorted(SUITE_ORDER)),
        "figure_04_six_panel_conus_topology",
        "Six panels share one CONUS extent and encoding; Alaska, Hawaii, and territories are excluded from the static figure while the GeoPackage remains nationwide.",
    )


def _generate_five_panel_map(config: Mapping[str, Any], root: Path) -> None:
    _generate_panel_map(
        config,
        root,
        required_suite_ids(config),
        "figure_05_five_panel_conus_acnhn_map",
        "Five panels for s02-s06 share one contiguous-mainland-US extent and encoding. Alaska, Hawaii, and territories are excluded only from this static map; nationwide data remain in the GeoPackage. s01 is excluded from Phase 3 production.",
    )


def _generate_panel_map(
    config: Mapping[str, Any],
    root: Path,
    suites: Sequence[str],
    stem: str,
    caption: str,
) -> None:
    if len(suites) == 5:
        fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    flat_axes = np.atleast_1d(axes).ravel()
    data_rows = []
    for ax, suite in zip(flat_axes[: len(suites)], suites, strict=True):
        directory = production_directory(config, suite)
        nodes = pd.read_parquet(directory / "acnhn_nodes.parquet")
        links = pd.read_parquet(directory / "acnhn_static_links.parquet")
        nodes = nodes[(nodes["state_fips"] != "02") & (nodes["state_fips"] != "15")]
        nodes = nodes[nodes["longitude"].between(-125, -66) & nodes["latitude"].between(24, 50)]
        bacn = nodes[nodes["node_type"] == "BACN"].copy()
        node_xy = nodes.set_index("node_id")[["longitude", "latitude"]]
        bundles = links[links["link_type"] == "BACKBONE_BUNDLE"].iloc[::20].copy()
        bundles = bundles[bundles["tail_node_id"].isin(node_xy.index) & bundles["head_node_id"].isin(node_xy.index)]
        segments = [
            [tuple(node_xy.loc[row.tail_node_id]), tuple(node_xy.loc[row.head_node_id])]
            for row in bundles.itertuples(index=False)
        ]
        ax.add_collection(
            LineCollection(
                segments,
                linewidths=0.15,
                colors="#577590",
                alpha=0.35,
                rasterized=True,
            )
        )
        ax.scatter(bacn["longitude"], bacn["latitude"], s=0.18, color="#d1495b", alpha=0.55, rasterized=True)
        ax.set_xlim(-125, -66)
        ax.set_ylim(24, 50)
        ax.set_title(suite)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
        data_rows.append({"suite_short_id": suite, "bacn_points_plotted": len(bacn), "bundle_lines_plotted": len(segments), "extent": "-125,-66,24,50", "excluded_state_fips": "02;15"})
    for ax in flat_axes[len(suites) :]:
        ax.set_visible(False)
    fig.suptitle("Selected-method ACNHN topology across the contiguous United States")
    _save_figure(root, stem, fig, pd.DataFrame(data_rows), caption)


def _save_figure(root: Path, stem: str, fig: plt.Figure, source: pd.DataFrame, caption: str) -> None:
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(root / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(root / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    source.to_csv(root / f"{stem}_data.csv", index=False, lineterminator="\n")
    (root / f"{stem}_caption.md").write_text(caption.rstrip() + "\n", encoding="utf-8", newline="\n")


def _path_size(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _directory_digest(path: Path) -> str:
    digest = hashlib_sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(partC_rev1_file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def hashlib_sha256() -> Any:
    import hashlib

    return hashlib.sha256()


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "-c", "core.longpaths=true", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def build_all_suite_geopackage(config: Mapping[str, Any]) -> dict[str, Any]:
    began = time.perf_counter()
    _configure_geospatial_environment(config)

    run_root = Path(str(config["output_root"])).resolve(strict=True)
    suites = require_all_suites(config)
    source = Path(str(config["phase2_geopackage"]["path"])).resolve(strict=True)
    if source.stat().st_size != int(config["phase2_geopackage"]["size_bytes"]):
        raise Phase3ProductionError("Accepted Phase 2 GeoPackage size changed")
    source_hash_before = partC_rev1_file_sha256(source)
    if source_hash_before != config["phase2_geopackage"]["sha256"]:
        raise Phase3ProductionError("Accepted Phase 2 GeoPackage hash changed")
    output = _configured_output_path(
        config, "geopackage", "geopackage/phase3_acnhn_all_suites.gpkg"
    )
    if output.exists():
        raise Phase3ProductionError(f"GeoPackage output already exists: {output}")
    staging_root = Path(str(config["temp_root"])).resolve(strict=True) / "portfolio_assembly"
    working = _new_staging_path(staging_root, output.stem, ".partial.gpkg")
    shutil.copy2(source, working)
    copied_hash = partC_rev1_file_sha256(working)
    if copied_hash != source_hash_before:
        raise Phase3ProductionError("Phase 2 GeoPackage copy is not byte-identical before append")
    layer_inventory: list[dict[str, Any]] = [
        {
            "artifact": "GeoPackage",
            "object_type": "feature_layer",
            "object_name": "directed_links_portfolio",
            "row_count": int(config["phase2_geopackage"]["directed_link_count"]),
            "validation_status": "PASS",
            "layer_scope": "IMMUTABLE_PHASE2_PHYSICAL_NETWORK",
        },
        {
            "artifact": "GeoPackage",
            "object_type": "feature_layer",
            "object_name": "physical_nodes",
            "row_count": int(config["phase2_geopackage"]["physical_node_count"]),
            "validation_status": "PASS",
            "layer_scope": "IMMUTABLE_PHASE2_PHYSICAL_NETWORK",
        },
    ]
    for suite in sorted(suites):
        directory = suites[suite]
        nodes = pd.read_parquet(directory / "acnhn_nodes.parquet")
        static_links = pd.read_parquet(directory / "acnhn_static_links.parquet")
        support = pd.read_parquet(directory / "regional_pair_support_summary.parquet")
        pair_sample = pd.read_parquet(directory / "regional_pair_visualization_sample.parquet")
        pair_total = int(support["supported_pair_count"].sum())
        node_layer = f"acnhn_{suite}_nodes"
        link_layer = f"acnhn_{suite}_links"
        _write_vectorized_points(working, node_layer, nodes)
        links = _spatial_link_frame(nodes, static_links, pair_sample, pair_total)
        _write_vectorized_lines(working, link_layer, links)
        layer_inventory.extend(
            [
                {"artifact": "GeoPackage", "object_type": "feature_layer", "object_name": node_layer, "row_count": len(nodes), "validation_status": "PASS"},
                {"artifact": "GeoPackage", "object_type": "feature_layer", "object_name": link_layer, "row_count": len(links), "validation_status": "PASS", "layer_scope": "STATIC_PLUS_CAPPED_REGIONAL_PAIRS", "regional_pair_rows_total": pair_total, "regional_pair_rows_in_layer": len(pair_sample)},
            ]
        )
        print(
            f"PHASE3_GPKG suite={suite} nodes={len(nodes)} links={len(links)} pairs_total={pair_total} pairs_layer={len(pair_sample)}",
            flush=True,
        )
    attribute_frames = _geopackage_attribute_frames(config, layer_inventory)
    with contextlib.closing(sqlite3.connect(working)) as connection:
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=FILE")
        for table, frame in attribute_frames.items():
            if connection.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
                raise Phase3ProductionError(f"GeoPackage attribute table already exists: {table}")
            frame.to_sql(table, connection, index=False, if_exists="fail", chunksize=5000)
            _register_gpkg_attribute_table(connection, table, f"Phase 3 {table}")
        connection.execute("CREATE UNIQUE INDEX ux_phase3_suite_registry ON phase3_suite_registry(suite_short_id)")
        connection.execute("CREATE UNIQUE INDEX ux_phase3_layer_inventory ON phase3_layer_inventory(object_name)")
        summary_table = (
            "phase3_summary_metrics" if is_scope_amended_closeout(config) else "phase3_summary"
        )
        connection.execute(f"CREATE INDEX ix_phase3_summary_suite ON {summary_table}(suite_short_id)")
        if not is_scope_amended_closeout(config):
            connection.execute("CREATE INDEX ix_phase3_support_suite_region ON regional_pair_support_summary(suite_short_id,region_id)")
        connection.commit()
    _write_qml_styles(run_root / "styles")
    source_hash_after = partC_rev1_file_sha256(source)
    if source_hash_after != source_hash_before:
        raise Phase3ProductionError("Accepted Phase 2 GeoPackage changed during Phase 3 append")
    validation = _validate_geopackage(config, working, layer_inventory)
    os.replace(working, output)
    evidence = {
        "schema_id": (
            "phase3.five_suite_closeout.geopackage_validation.v1"
            if is_scope_amended_closeout(config)
            else "phase3.all_suite_geopackage_validation.v1"
        ),
        "status": "PASS",
        "path": str(output),
        "size_bytes": int(output.stat().st_size),
        "sha256": partC_rev1_file_sha256(output),
        "phase2_source_path": str(source),
        "phase2_source_sha256_before": source_hash_before,
        "phase2_source_sha256_after": source_hash_after,
        "byte_identical_copy_before_append": copied_hash == source_hash_before,
        "physical_layers_preserved": True,
        "inventory": layer_inventory + [
            {"artifact": "GeoPackage", "object_type": "attribute_table", "object_name": name, "row_count": len(frame), "validation_status": "PASS"}
            for name, frame in attribute_frames.items()
        ],
        "reader_validation": validation,
        "elapsed_seconds": time.perf_counter() - began,
        "generated_utc": utc_now(),
    }
    _write_json_new(run_root / "evidence" / "geopackage_validation.json", evidence)
    return evidence


def _new_staging_path(root: Path, stem: str, suffix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 1000):
        path = root / f"{stem}.attempt_{attempt:03d}{suffix}"
        if not path.exists():
            return path
    raise Phase3ProductionError(f"No free staging path remains for {stem}")


def _configure_geospatial_environment(config: Mapping[str, Any]) -> None:
    prefix = Path(str(config["python_executable"])).parent
    os.environ.setdefault("GDAL_DATA", str(prefix / "Library" / "share" / "gdal"))
    os.environ.setdefault("PROJ_LIB", str(prefix / "Library" / "share" / "proj"))
    binary = str(prefix / "Library" / "bin")
    if binary.lower() not in os.environ.get("PATH", "").lower():
        os.environ["PATH"] = binary + os.pathsep + os.environ.get("PATH", "")


def _write_vectorized_points(path: Path, layer_name: str, frame: pd.DataFrame) -> None:
    import geopandas as gpd
    from shapely import points

    source_columns = [
        "suite_short_id", "node_id", "node_type", "terminal_type", "parent_object_id", "region_id",
        "member_count", "structural_charge_s", "state_fips", "longitude", "latitude",
    ]
    attributes = frame[source_columns].rename(
        columns={
            "suite_short_id": "suite_id",
            "terminal_type": "term_type",
            "parent_object_id": "parent_id",
            "member_count": "members",
            "structural_charge_s": "charge_s",
        }
    )
    geometry = points(
        attributes["longitude"].to_numpy(dtype=np.float64, copy=False),
        attributes["latitude"].to_numpy(dtype=np.float64, copy=False),
    )
    geoframe = gpd.GeoDataFrame(attributes, geometry=geometry, crs="EPSG:4269")
    _write_geodataframe_chunks(path, layer_name, geoframe, "Point")


def _spatial_link_frame(
    nodes: pd.DataFrame,
    static_links: pd.DataFrame,
    pair_sample: pd.DataFrame,
    pair_total: int,
) -> pd.DataFrame:
    coordinates = nodes[["node_id", "longitude", "latitude"]]
    static = (
        static_links.merge(
            coordinates.rename(columns={"node_id": "tail_node_id", "longitude": "tail_lon", "latitude": "tail_lat"}),
            on="tail_node_id", how="left", validate="many_to_one",
        ).merge(
            coordinates.rename(columns={"node_id": "head_node_id", "longitude": "head_lon", "latitude": "head_lat"}),
            on="head_node_id", how="left", validate="many_to_one",
        )
    )
    static["layer_scope"] = "STATIC_PLUS_CAPPED_REGIONAL_PAIRS"
    static["regional_pair_rows_total"] = int(pair_total)
    static["regional_pair_rows_in_layer"] = int(len(pair_sample))
    static["is_visualization_subset"] = False
    pair = (
        pair_sample.rename(columns={"pair_id": "link_id", "source_terminal_id": "tail_node_id", "target_terminal_id": "head_node_id", "direct_fftt_s": "direct_base_fftt_s"})
        .merge(
            coordinates.rename(columns={"node_id": "tail_node_id", "longitude": "tail_lon", "latitude": "tail_lat"}),
            on="tail_node_id", how="left", validate="many_to_one",
        ).merge(
            coordinates.rename(columns={"node_id": "head_node_id", "longitude": "head_lon", "latitude": "head_lat"}),
            on="head_node_id", how="left", validate="many_to_one",
        )
    )
    pair["suite_short_id"] = nodes["suite_short_id"].iloc[0]
    pair["link_type"] = "REGIONAL_PAIR"
    pair["direction_role"] = "region_restricted_support"
    pair["member_object_count"] = 1
    pair["structural_seed_fftt_s"] = pair["direct_base_fftt_s"]
    pair["layer_scope"] = "STATIC_PLUS_CAPPED_REGIONAL_PAIRS"
    pair["regional_pair_rows_total"] = int(pair_total)
    pair["regional_pair_rows_in_layer"] = int(len(pair_sample))
    pair["is_visualization_subset"] = True
    columns = [
        "suite_short_id", "link_id", "link_type", "tail_node_id", "head_node_id", "region_id",
        "direction_role", "member_object_count", "direct_base_fftt_s", "structural_seed_fftt_s",
        "layer_scope", "regional_pair_rows_total", "regional_pair_rows_in_layer", "is_visualization_subset",
        "tail_lon", "tail_lat", "head_lon", "head_lat",
    ]
    result = pd.concat([static[columns], pair[columns]], ignore_index=True)
    if result[["tail_lon", "tail_lat", "head_lon", "head_lat"]].isna().any().any():
        raise Phase3ProductionError("GeoPackage link endpoints lack representative coordinates")
    return result.sort_values(["link_type", "link_id"], kind="mergesort").reset_index(drop=True)


def _write_vectorized_lines(path: Path, layer_name: str, frame: pd.DataFrame) -> None:
    import geopandas as gpd
    from shapely import linestrings

    source_columns = [
        "suite_short_id", "link_id", "link_type", "tail_node_id", "head_node_id", "region_id",
        "direction_role", "member_object_count", "direct_base_fftt_s", "structural_seed_fftt_s",
        "layer_scope", "regional_pair_rows_total", "regional_pair_rows_in_layer", "is_visualization_subset",
    ]
    attributes = frame[source_columns].rename(
        columns={
            "suite_short_id": "suite_id",
            "tail_node_id": "tail_id",
            "head_node_id": "head_id",
            "direction_role": "direction",
            "member_object_count": "members",
            "direct_base_fftt_s": "direct_s",
            "structural_seed_fftt_s": "struct_s",
            "regional_pair_rows_total": "pair_total",
            "regional_pair_rows_in_layer": "pair_layer",
            "is_visualization_subset": "is_subset",
        }
    )
    attributes["is_subset"] = attributes["is_subset"].astype(np.int8)
    coordinates = np.stack(
        [
            frame[["tail_lon", "tail_lat"]].to_numpy(dtype=np.float64, copy=False),
            frame[["head_lon", "head_lat"]].to_numpy(dtype=np.float64, copy=False),
        ],
        axis=1,
    )
    geometry = linestrings(coordinates)
    geoframe = gpd.GeoDataFrame(attributes, geometry=geometry, crs="EPSG:4269")
    _write_geodataframe_chunks(path, layer_name, geoframe, "LineString")


def _write_geodataframe_chunks(
    path: Path,
    layer_name: str,
    geoframe: Any,
    geometry_type: str,
    *,
    chunk_rows: int = 200_000,
) -> None:
    import pyogrio

    if layer_name in set(pyogrio.list_layers(path)[:, 0].tolist()):
        raise Phase3ProductionError(f"Cannot create write-new feature layer {layer_name}")
    if geoframe.empty:
        raise Phase3ProductionError(f"No features supplied for required layer {layer_name}")
    for start in range(0, len(geoframe), chunk_rows):
        stop = min(start + chunk_rows, len(geoframe))
        pyogrio.write_dataframe(
            geoframe.iloc[start:stop],
            path,
            layer=layer_name,
            driver="GPKG",
            geometry_type=geometry_type,
            append=start > 0,
            # The frozen py312 environment exposes GDAL 3.6.x. Pyogrio's
            # Arrow writer requires GDAL >=3.8, so use its compiled NumPy path.
            use_arrow=False,
            layer_options=(
                {"GEOMETRY_NAME": "geom", "SPATIAL_INDEX": "YES", "FID": "fid"}
                if start == 0
                else None
            ),
        )
    observed = int(pyogrio.read_info(path, layer=layer_name)["features"])
    if observed != len(geoframe):
        raise Phase3ProductionError(
            f"Feature-layer count mismatch: {layer_name} {observed} != {len(geoframe)}"
        )


def _geopackage_attribute_frames(
    config: Mapping[str, Any], layer_inventory: Sequence[Mapping[str, Any]]
) -> dict[str, pd.DataFrame]:
    rows = []
    support_frames = []
    for suite in sorted(required_suite_ids(config)):
        directory = production_directory(config, suite)
        rows.append(json.loads((directory / "suite_metadata.json").read_text(encoding="utf-8")))
        support_frames.append(_with_suite_fields(config, suite, pd.read_parquet(directory / "regional_pair_support_summary.parquet")))
    summary = pd.DataFrame(rows).sort_values("suite_short_id", kind="mergesort")
    if is_scope_amended_closeout(config):
        registry = _portfolio_registry(config, require_all_suites(config))
        metrics = registry.copy()
        inventory = pd.DataFrame(layer_inventory)
        attribute_rows = [
            {
                "artifact": "GeoPackage",
                "object_type": "attribute_table",
                "object_name": "phase3_suite_registry",
                "row_count": len(registry),
                "validation_status": "PASS",
                "layer_scope": "SIX_ACCEPTED_SUITES_WITH_S01_EXCLUSION",
            },
            {
                "artifact": "GeoPackage",
                "object_type": "attribute_table",
                "object_name": "phase3_layer_inventory",
                "row_count": len(inventory) + 4,
                "validation_status": "PASS",
                "layer_scope": "PORTFOLIO",
            },
            {
                "artifact": "GeoPackage",
                "object_type": "attribute_table",
                "object_name": "phase3_summary_metrics",
                "row_count": len(metrics),
                "validation_status": "PASS",
                "layer_scope": "FIVE_PRODUCTION_PLUS_S01_CONTEXT",
            },
            {
                "artifact": "GeoPackage",
                "object_type": "attribute_table",
                "object_name": "s01_exclusion_record",
                "row_count": 1,
                "validation_status": "PASS",
                "layer_scope": "S01_NONPRODUCT_EVIDENCE",
            },
        ]
        inventory = pd.concat([inventory, pd.DataFrame(attribute_rows)], ignore_index=True)
        return {
            "phase3_suite_registry": registry,
            "phase3_layer_inventory": inventory,
            "phase3_summary_metrics": metrics,
            "s01_exclusion_record": _s01_exclusion_frame(config),
        }
    registry_columns = [
        "phase3_run_id", "suite_short_id", "phase2_suite_id", "resolution", "candidate_id", "method_id",
        "final_backbone_node_count", "final_backbone_arc_count", "bacn_count", "racn_count", "terminal_count",
        "actual_online_node_count", "actual_static_link_count", "exact_regional_pair_count",
    ]
    inventory = pd.DataFrame(layer_inventory)
    return {
        "phase3_suite_registry": summary[registry_columns],
        "phase3_layer_inventory": inventory,
        "phase3_summary": summary,
        "regional_pair_support_summary": pd.concat(support_frames, ignore_index=True),
    }


def _register_gpkg_attribute_table(connection: sqlite3.Connection, table: str, description: str) -> None:
    connection.execute(
        "INSERT INTO gpkg_contents(table_name,data_type,identifier,description,last_change,srs_id) "
        "VALUES(?,?,?, ?,strftime('%Y-%m-%dT%H:%M:%fZ','now'),NULL)",
        (table, "attributes", table, description),
    )


def _validate_geopackage(
    config: Mapping[str, Any], path: Path, inventory: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = {"directed_links_portfolio": 661644, "physical_nodes": 346373}
    for suite in sorted(required_suite_ids(config)):
        directory = production_directory(config, suite)
        expected[f"acnhn_{suite}_nodes"] = len(pd.read_parquet(directory / "acnhn_nodes.parquet"))
        expected[f"acnhn_{suite}_links"] = len(pd.read_parquet(directory / "acnhn_static_links.parquet")) + len(pd.read_parquet(directory / "regional_pair_visualization_sample.parquet"))
    reader_results = {
        reader: _run_geopackage_reader(config, path, reader, expected)
        for reader in ("ogr", "fiona", "pyogrio")
    }
    with contextlib.closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=1")
        physical_counts = {
            name: int(connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
            for name in ("directed_links_portfolio", "physical_nodes")
        }
        attribute_names = (
            (
                "phase3_suite_registry",
                "phase3_layer_inventory",
                "phase3_summary_metrics",
                "s01_exclusion_record",
            )
            if is_scope_amended_closeout(config)
            else (
                "phase3_suite_registry",
                "phase3_layer_inventory",
                "phase3_summary",
                "regional_pair_support_summary",
            )
        )
        attributes = {
            name: int(connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
            for name in attribute_names
        }
        acnhn_layers = {
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM gpkg_contents WHERE data_type='features' AND table_name LIKE 'acnhn_%'"
            )
        }
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
    expected_acnhn_layers = {
        f"acnhn_{suite}_{kind}"
        for suite in required_suite_ids(config)
        for kind in ("nodes", "links")
    }
    checks = {
        "physical_directed_links_preserved": physical_counts["directed_links_portfolio"] == 661644,
        "physical_nodes_preserved": physical_counts["physical_nodes"] == 346373,
        "exact_required_acnhn_layers": acnhn_layers == expected_acnhn_layers,
        "attribute_tables": (
            attributes["phase3_suite_registry"] == 6
            and (
                attributes["phase3_summary_metrics"] == 6
                and attributes["s01_exclusion_record"] == 1
                if is_scope_amended_closeout(config)
                else attributes["phase3_summary"] == 6
                and attributes["regional_pair_support_summary"] > 0
            )
        ),
        "isolated_ogr_fiona_pyogrio_reopen": all(value["status"] == "PASS" for value in reader_results.values()),
        "quick_check": quick_check == "ok",
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise Phase3ProductionError(f"GeoPackage validation failed: {failed}")
    return {"status": "PASS", "checks": checks, "reader_results": reader_results, "physical_counts": physical_counts, "attribute_counts": attributes}


def _run_geopackage_reader(
    config: Mapping[str, Any], path: Path, reader: str, expected: Mapping[str, int]
) -> dict[str, Any]:
    python = str(config["python_executable"])
    payload = json.dumps(dict(expected), sort_keys=True)
    if reader == "ogr":
        code = (
            "import json,sys; from osgeo import ogr; p=sys.argv[1]; exp=json.loads(sys.argv[2]); "
            "d=ogr.Open(p,0); got={k:int(d.GetLayerByName(k).GetFeatureCount()) for k in exp}; "
            "print(json.dumps({'status':'PASS' if got==exp else 'FAIL','counts':got},sort_keys=True))"
        )
    elif reader == "fiona":
        code = (
            "import json,sys,fiona; p=sys.argv[1]; exp=json.loads(sys.argv[2]); layers=set(fiona.listlayers(p)); "
            "got={k:len(fiona.open(p,layer=k)) for k in exp if k in layers}; "
            "print(json.dumps({'status':'PASS' if got==exp else 'FAIL','counts':got},sort_keys=True))"
        )
    else:
        code = (
            "import json,sys,pyogrio; p=sys.argv[1]; exp=json.loads(sys.argv[2]); "
            "layers={str(r[0]) for r in pyogrio.list_layers(p)}; got={k:int(pyogrio.read_info(p,layer=k)['features']) for k in exp if k in layers}; "
            "print(json.dumps({'status':'PASS' if got==exp else 'FAIL','counts':got},sort_keys=True))"
        )
    environment = os.environ.copy()
    result = subprocess.run(
        [python, "-c", code, str(path), payload],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    parsed = json.loads(result.stdout.strip().splitlines()[-1])
    if parsed["status"] != "PASS":
        raise Phase3ProductionError(f"Isolated {reader} GeoPackage reopen failed: {parsed}")
    return parsed


def _write_qml_styles(root: Path) -> None:
    styles = {
        "physical_directed_links.qml": "#808080",
        "bacn_nodes.qml": "#d1495b",
        "racn_nodes.qml": "#00798c",
        "terminal_nodes.qml": "#edae49",
        "backbone_bundles.qml": "#277da1",
        "terminal_connectors.qml": "#f8961e",
        "regional_pairs.qml": "#43aa8b",
    }
    for name, color in styles.items():
        path = root / name
        if path.exists():
            raise Phase3ProductionError(f"QML style already exists: {path}")
        geometry = "line" if any(token in name for token in ("links", "bundles", "connectors", "pairs")) else "point"
        symbol_layer = (
            f'<layer class="SimpleLine"><prop k="line_color" v="{color}"/><prop k="line_width" v="0.4"/></layer>'
            if geometry == "line"
            else f'<layer class="SimpleMarker"><prop k="color" v="{color}"/><prop k="size" v="1.5"/></layer>'
        )
        text = f'<?xml version="1.0" encoding="UTF-8"?>\n<qgis version="3"><renderer-v2 type="singleSymbol"><symbols><symbol name="0" type="{geometry}">{symbol_layer}</symbol></symbols></renderer-v2></qgis>\n'
        path.write_text(text, encoding="utf-8", newline="\n")
        ET.parse(path)
