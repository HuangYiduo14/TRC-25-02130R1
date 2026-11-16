"""Final Phase 3 validation and compact external-review handoff sealing."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import pandas as pd
from PIL import Image

from partC_rev1_src.partC_rev1_hashing import partC_rev1_file_sha256

from .production import (
    METHOD_ID,
    SUITE_ORDER,
    Phase3ProductionError,
    excluded_suite_ids,
    required_suite_ids,
)
from .production import _replace_json, _write_csv_new, _write_json_new, utc_now
from .production_portfolio import (
    CLOSEOUT_TABLE_FILES,
    TABLE_FILES,
    is_scope_amended_closeout,
    portfolio_table_files,
    require_all_suites,
)
from .resources import directory_bytes


def validate_all_suite_campaign(config: Mapping[str, Any]) -> dict[str, Any]:
    began = time.perf_counter()
    run_root = Path(str(config["output_root"])).resolve(strict=True)
    temp_root = Path(str(config["temp_root"])).resolve(strict=True)
    suites = require_all_suites(config)
    _write_ambiguity_logs(run_root, config)
    _write_command_log(config, run_root)
    suite_validation = {}
    suite_scientific_validation = {}
    for suite in sorted(suites):
        suite_validation[suite] = _verify_suite_manifest(suites[suite])
        scientific = json.loads(
            (suites[suite] / "validation.json").read_text(encoding="utf-8")
        )
        suite_scientific_validation[suite] = scientific
    if partC_rev1_file_sha256(suites["s03"] / "candidate_od_results.parquet") != config["common_evaluation"]["s03_result_sha256"]:
        raise Phase3ProductionError("Production s03 result is not the exact accepted result")
    phase2_invariance = _verify_phase2_invariance(config)
    active_package_binding = (
        config["scope_amendment_package"]
        if is_scope_amended_closeout(config)
        else config["authorization_package"]
    )
    package = Path(str(active_package_binding["path"])).resolve(strict=True)
    package_invariant = (
        package.stat().st_size == int(active_package_binding["size_bytes"])
        and partC_rev1_file_sha256(package) == active_package_binding["sha256"]
    )
    source_gpkg = Path(str(config["phase2_geopackage"]["path"])).resolve(strict=True)
    gpkg_source_invariant = partC_rev1_file_sha256(source_gpkg) == config["phase2_geopackage"]["sha256"]
    sqlite_evidence = json.loads((run_root / "evidence" / "sqlite_validation.json").read_text(encoding="utf-8"))
    gpkg_evidence = json.loads((run_root / "evidence" / "geopackage_validation.json").read_text(encoding="utf-8"))
    summary_evidence = json.loads((run_root / "evidence" / "summary_table_figure_validation.json").read_text(encoding="utf-8"))
    table_checks = _verify_tables(config, run_root / "tables")
    figure_checks = _verify_figures(run_root / "figures")
    style_checks = _verify_styles(run_root / "styles")
    resources = _resource_validation(config, run_root, temp_root)
    status = json.loads((run_root / "RUN_STATUS.json").read_text(encoding="utf-8"))
    required = set(required_suite_ids(config))
    excluded = set(excluded_suite_ids(config))
    combined_od = _verify_combined_od_portfolio(config, run_root)
    s01_evidence = _verify_s01_partial_evidence(config, run_root)
    phase_boundaries = {
        "phase4_authorized": status.get("phase4_authorized") is False,
        "capacity_multiplier_tuned": status.get("capacity_multiplier_tuned") is False,
        "cost_calibration_performed": status.get("cost_calibration_performed") is False,
        "model_training_performed": status.get("model_training_performed") is False,
        "acnhn_equilibrium_run": status.get("acnhn_equilibrium_run") is False,
    }
    checks = {
        "all_required_suite_manifests": all(value["status"] == "PASS" for value in suite_validation.values()),
        "all_required_suite_scientific_checks": all(
            value.get("status") == "PASS"
            and all(bool(check) for check in value.get("checks", {}).values())
            and value.get("checks", {}).get("forward_projection_complete") is True
            for value in suite_scientific_validation.values()
        ),
        "all_required_suites_marked_complete": required.issubset(
            set(status.get("completed_suites", ()))
        ),
        "all_required_production_suites_built": status.get(
            "all_required_production_suites_built"
        )
        is True,
        "all_six_suites_transparently_false": (
            status.get("all_six_suites_built") is False
            if excluded
            else set(status.get("completed_suites", ())) == set(SUITE_ORDER)
        ),
        "s03_exact_reuse": True,
        "phase2_suite_inputs_invariant": phase2_invariance["status"] == "PASS",
        "authorization_package_invariant": package_invariant,
        "phase2_geopackage_source_invariant": gpkg_source_invariant,
        "sqlite_valid": sqlite_evidence["status"] == "PASS",
        "geopackage_valid": gpkg_evidence["status"] == "PASS",
        "tables_and_figures_valid": summary_evidence["status"] == "PASS" and table_checks["status"] == "PASS" and figure_checks["status"] == "PASS",
        "qml_styles_valid": style_checks["status"] == "PASS",
        "resources_within_hard_limits": resources["status"] == "PASS",
        "phase_boundaries_preserved": all(phase_boundaries.values()),
        "combined_common_od_portfolio": combined_od["status"] == "PASS",
        "s01_partial_nonproduct_evidence": s01_evidence["status"] == "PASS",
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise Phase3ProductionError(f"Final Phase 3 validation failed: {failed}")
    artifact_manifest = _final_artifact_manifest(run_root)
    _write_csv_new(run_root / "evidence" / "FINAL_ARTIFACT_MANIFEST.csv", artifact_manifest)
    result = {
        "schema_id": (
            "phase3.five_suite_closeout.final_validation.v1"
            if is_scope_amended_closeout(config)
            else "phase3.final_campaign_validation.v1"
        ),
        "status": "PASS",
        "checks": checks,
        "suite_validation": suite_validation,
        "suite_scientific_validation": suite_scientific_validation,
        "phase2_invariance": phase2_invariance,
        "sqlite": {"path": sqlite_evidence["path"], "sha256": sqlite_evidence["sha256"], "status": sqlite_evidence["status"]},
        "geopackage": {"path": gpkg_evidence["path"], "sha256": gpkg_evidence["sha256"], "status": gpkg_evidence["status"]},
        "table_checks": table_checks,
        "figure_checks": figure_checks,
        "style_checks": style_checks,
        "resources": resources,
        "phase_boundaries": phase_boundaries,
        "combined_od_portfolio": combined_od,
        "s01_partial_evidence": s01_evidence,
        "required_suite_ids": list(required_suite_ids(config)),
        "excluded_suite_ids": list(excluded_suite_ids(config)),
        "all_required_production_suites_built": True,
        "all_six_suites_built": not bool(excluded),
        "artifact_manifest_rows": len(artifact_manifest),
        "elapsed_seconds": time.perf_counter() - began,
        "validated_utc": utc_now(),
        "next_status": "PACKAGING_HANDOFF",
    }
    _write_json_new(run_root / "evidence" / "FINAL_VALIDATION.json", result)
    if is_scope_amended_closeout(config):
        _write_json_new(
            run_root / "validation" / "phase3_five_suite_final_validation.json", result
        )
        reproduction = {
            "schema_id": "phase3.five_suite_closeout.reproduction.v1",
            "status": "PASS",
            "required_suite_ids": list(required_suite_ids(config)),
            "suite_reproduction": {
                suite: json.loads(
                    (suites[suite] / "reproduction.json").read_text(encoding="utf-8")
                )
                for suite in required_suite_ids(config)
            },
            "manifest_verification": suite_validation,
            "common_od_portfolio": combined_od,
            "reviewed_blocked_handoff_sha256": config["reviewed_blocked_handoff"][
                "sha256"
            ],
            "s01_artifact_status": "PARTIAL_NONPRODUCT_EVIDENCE",
            "validated_utc": utc_now(),
        }
        _write_json_new(
            run_root / "validation" / "phase3_five_suite_reproduction.json",
            reproduction,
        )
    status["status"] = "PACKAGING_HANDOFF"
    status["updated_utc"] = utc_now()
    status["all_required_production_suites_built"] = True
    status["all_six_suites_built"] = not bool(excluded)
    _replace_json(run_root / "RUN_STATUS.json", status)
    return result


def build_final_handoff(config: Mapping[str, Any], handoff_id: str) -> dict[str, Any]:
    run_root = Path(str(config["output_root"])).resolve(strict=True)
    validation = json.loads((run_root / "evidence" / "FINAL_VALIDATION.json").read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise Phase3ProductionError("Final handoff requires a passing final validation")
    project_root = Path(str(config["project_root"])).resolve(strict=True)
    source = project_root / "partC_doc_icnn" / "handoffs" / "phase_03" / "all_suite_acnhn" / handoff_id
    if source.exists():
        raise Phase3ProductionError(f"Handoff source already exists: {source}")
    directories = (
        "governing_docs", "configs", "code_snapshot", "code_diff", "summary_data",
        "figures", "geospatial_summary", "log_excerpts", "external_artifact_index",
    )
    source.mkdir(parents=True)
    for name in directories:
        (source / name).mkdir()
    git_head = _git(project_root, "rev-parse", "HEAD").strip()
    git_status_before = _git(project_root, "status", "--short", "-uall").splitlines()
    summary = _handoff_summary(config, validation, git_head)
    zip_root = project_root / "artifacts_icnn" / "handoff_bundles" / "phase_03"
    zip_root.mkdir(parents=True, exist_ok=True)
    zip_prefix = str(config.get("handoff_zip_prefix", "phase03_all_suite_acnhn"))
    zip_path = zip_root / f"{zip_prefix}_{handoff_id}_SYNC.zip"
    _write_handoff_top_files(config, source, summary, validation, zip_path, git_head, git_status_before)
    _copy_handoff_payload(config, source)
    _write_code_evidence(config, source, git_head, git_status_before)
    checksums = _source_checksums(source)
    (source / "CHECKSUMS.sha256").write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in checksums.items()),
        encoding="utf-8", newline="\n",
    )
    _create_zip(source, zip_path)
    if zip_path.stat().st_size > 200 * 1024**2:
        raise Phase3ProductionError("Compact handoff exceeds the 200 MB hard synchronization limit")
    digest = partC_rev1_file_sha256(zip_path)
    sidecar = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8", newline="\n")
    fresh = _fresh_extract_verify(config, zip_path)
    companion = {
        "schema_id": "phase3.final_handoff_companion.v1",
        "handoff_id": handoff_id,
        "sync_zip": str(zip_path),
        "sync_zip_size_bytes": int(zip_path.stat().st_size),
        "sync_zip_sha256": digest,
        "source_folder": str(source),
        "source_commit": git_head,
        "fresh_extraction": fresh,
        "status": "WAITING_FOR_EXTERNAL_PHASE3_REVIEW",
        "next_step_authorized": False,
        "phase4_authorized": False,
        "required_suite_ids": list(required_suite_ids(config)),
        "excluded_suite_ids": list(excluded_suite_ids(config)),
        "all_required_production_suites_built": True,
        "all_six_suites_built": not bool(excluded_suite_ids(config)),
        "created_utc": utc_now(),
    }
    companion_path = zip_path.with_suffix(".companion.json")
    _write_json_new(companion_path, companion)
    latest = source.parent / "LATEST_HANDOFF.txt"
    latest.write_text(
        f"source_folder={source}\nzip_path={zip_path}\nzip_sha256={digest}\ncreated_utc={companion['created_utc']}\nstatus=WAITING_FOR_EXTERNAL_PHASE3_REVIEW\n",
        encoding="utf-8", newline="\n",
    )
    status_path = run_root / "RUN_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "status": "WAITING_FOR_EXTERNAL_PHASE3_REVIEW",
            "phase3a_stage1": "COMPLETE_WITH_NOTES",
            "phase3_production_construction": "COMPLETE_WITH_NOTES",
            "phase3_overall": "WAITING_FOR_EXTERNAL_PHASE3_REVIEW",
            "required_suite_ids": list(required_suite_ids(config)),
            "excluded_suite_ids": list(excluded_suite_ids(config)),
            "all_required_production_suites_built": True,
            "all_six_suites_built": not bool(excluded_suite_ids(config)),
            "s01_disposition": (
                config["scope_amendment"]["excluded_suite"]["disposition"]
                if is_scope_amended_closeout(config)
                else None
            ),
            "next_step_authorized": False,
            "winning_method": METHOD_ID,
            "phase4_authorized": False,
            "capacity_multiplier_tuned": False,
            "cost_calibration_performed": False,
            "model_training_performed": False,
            "acnhn_equilibrium_run": False,
            "handoff_zip": str(zip_path),
            "handoff_zip_sha256": digest,
            "updated_utc": utc_now(),
        }
    )
    _replace_json(status_path, status)
    _write_json_new(run_root / "evidence" / "FINAL_HANDOFF.json", companion)
    return companion


def _verify_suite_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.csv"
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8-sig", newline="")))
    failures = []
    for row in rows:
        path = root.joinpath(*PurePosixPath(row["relative_path"]).parts)
        if not path.is_file():
            failures.append(f"missing:{row['relative_path']}")
        elif path.stat().st_size != int(row["size_bytes"]):
            failures.append(f"size:{row['relative_path']}")
        elif partC_rev1_file_sha256(path) != row["sha256"]:
            failures.append(f"sha256:{row['relative_path']}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    declared = {row["relative_path"] for row in rows} | {"artifact_manifest.csv"}
    if actual != declared:
        failures.append(f"closure:missing={sorted(declared-actual)[:5]}:extra={sorted(actual-declared)[:5]}")
    if failures:
        raise Phase3ProductionError(f"Production suite manifest failed at {root}: {failures[:10]}")
    return {"status": "PASS", "manifest_rows": len(rows), "file_count": len(actual)}


def _verify_phase2_invariance(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(config["accepted_phase2_output_root"])).resolve(strict=True) / "suites"
    failures = []
    member_count = 0
    for suite in config["suites"]:
        suite_root = root / str(suite["phase2_suite_id"])
        rows = list(csv.DictReader((suite_root / "artifact_manifest.csv").open("r", encoding="utf-8-sig", newline="")))
        member_count += len(rows)
        for row in rows:
            path = suite_root / row["relative_path"]
            if not path.is_file() or path.stat().st_size != int(row["size_bytes"]) or partC_rev1_file_sha256(path) != row["sha256"]:
                failures.append(f"{suite['suite_short_id']}:{row['relative_path']}")
    if failures or member_count != 120:
        raise Phase3ProductionError(f"Accepted Phase 2 invariance failed: {failures[:10]}")
    return {"status": "PASS", "suite_count": 6, "manifest_member_count": member_count, "failures": []}


def _verify_combined_od_portfolio(
    config: Mapping[str, Any], run_root: Path
) -> dict[str, Any]:
    relative = str(
        config.get("output_names", {}).get(
            "combined_od_results", "tables/all_suite_candidate_od_results.parquet"
        )
    )
    combined = pd.read_parquet(run_root / relative)
    required = tuple(required_suite_ids(config))
    expected_rows = len(required) * int(config["common_evaluation"]["od_count"]) * len(
        config["common_evaluation"]["modes"]
    )
    reference_ids: set[str] | None = None
    reference_physical: set[tuple[Any, ...]] | None = None
    per_suite = {}
    for suite in required:
        frame = combined[combined["suite_short_id"] == suite]
        ids = set(frame["sample_od_id"].astype(str))
        physical = set(
            map(
                tuple,
                frame[["sample_od_id", "physical_path_id", "physical_fftt_s"]]
                .drop_duplicates()
                .to_numpy(),
            )
        )
        if reference_ids is None:
            reference_ids = ids
            reference_physical = physical
        modes = frame["diagnostic_mode"].value_counts().to_dict()
        checks = {
            "rows_400": len(frame) == 400,
            "unique_ods_200": len(ids) == 200,
            "direct_rows_200": modes.get("direct", 0) == 200,
            "structural_rows_200": modes.get("structural", 0) == 200,
            "common_od_ids": ids == reference_ids,
            "common_physical_paths_and_fftt": physical == reference_physical,
        }
        if not all(checks.values()):
            raise Phase3ProductionError(f"Combined OD portfolio failed for {suite}: {checks}")
        per_suite[suite] = checks
    checks = {
        "row_count": len(combined) == expected_rows,
        "suite_scope": set(combined["suite_short_id"].astype(str)) == set(required),
        "unique_od_count": combined["sample_od_id"].nunique() == 200,
        "s01_absent": "s01" not in set(combined["suite_short_id"].astype(str)),
    }
    if not all(checks.values()):
        raise Phase3ProductionError(f"Combined OD portfolio scope/count failed: {checks}")
    return {
        "status": "PASS",
        "path": str((run_root / relative).resolve()),
        "sha256": partC_rev1_file_sha256(run_root / relative),
        "row_count": len(combined),
        "unique_od_count": combined["sample_od_id"].nunique(),
        "checks": checks,
        "per_suite": per_suite,
    }


def _verify_s01_partial_evidence(
    config: Mapping[str, Any], run_root: Path
) -> dict[str, Any]:
    if not is_scope_amended_closeout(config):
        return {"status": "NOT_APPLICABLE"}
    readiness = json.loads(
        (run_root / "evidence" / "CLOSEOUT_READINESS.json").read_text(encoding="utf-8")
    )
    expected = readiness["s01_partial_evidence"]
    operation = run_root / "operations" / "s01"
    manifest = run_root / "evidence" / "S01_PARTIAL_ARTIFACT_MANIFEST.csv"
    actual_files = [path for path in operation.rglob("*") if path.is_file()]
    checks = {
        "operation_path_unchanged": str(operation) == expected["path"],
        "file_count_unchanged": len(actual_files) == int(expected["file_count"]),
        "size_bytes_unchanged": directory_bytes(operation) == int(expected["size_bytes"]),
        "directory_digest_unchanged": _directory_digest(operation)
        == expected["directory_sha256"],
        "partial_manifest_unchanged": partC_rev1_file_sha256(manifest)
        == expected["manifest_sha256"],
        "no_s01_production_directory": not (run_root / "suites" / "s01").exists(),
        "artifact_status": expected["artifact_status"] == "PARTIAL_NONPRODUCT_EVIDENCE",
    }
    if not all(checks.values()):
        raise Phase3ProductionError(f"s01 partial non-product evidence changed: {checks}")
    return {
        "status": "PASS",
        "checks": checks,
        "path": str(operation),
        "size_bytes": directory_bytes(operation),
        "file_count": len(actual_files),
        "directory_sha256": expected["directory_sha256"],
        "eligible_for_phase4": False,
        "load_into_sqlite_production_tables": False,
        "load_into_geopackage": False,
    }


def _directory_digest(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(partC_rev1_file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _verify_tables(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    counts = {}
    names = portfolio_table_files(config)
    for name in names:
        path = root / name
        if not path.is_file():
            raise Phase3ProductionError(f"Required table missing: {path}")
        counts[name] = len(pd.read_csv(path))
    if is_scope_amended_closeout(config):
        rules = {
            CLOSEOUT_TABLE_FILES[0]: counts[CLOSEOUT_TABLE_FILES[0]] == 6,
            CLOSEOUT_TABLE_FILES[1]: counts[CLOSEOUT_TABLE_FILES[1]] == 5,
            CLOSEOUT_TABLE_FILES[2]: counts[CLOSEOUT_TABLE_FILES[2]] == 10,
            CLOSEOUT_TABLE_FILES[3]: counts[CLOSEOUT_TABLE_FILES[3]] == 10,
            CLOSEOUT_TABLE_FILES[4]: counts[CLOSEOUT_TABLE_FILES[4]] >= 5,
            CLOSEOUT_TABLE_FILES[6]: counts[CLOSEOUT_TABLE_FILES[6]] >= 9,
            CLOSEOUT_TABLE_FILES[7]: counts[CLOSEOUT_TABLE_FILES[7]] == 5,
            CLOSEOUT_TABLE_FILES[8]: counts[CLOSEOUT_TABLE_FILES[8]] >= 1,
        }
        combined_relative = str(config["output_names"]["combined_od_results"])
        combined = pd.read_parquet(Path(str(config["output_root"])) / combined_relative)
        rules["five_suite_candidate_od_results"] = len(combined) == 2000
        rules["final_reports"] = all(
            (Path(str(config["output_root"])) / relative).is_file()
            for relative in (
                "summary/PHASE3_FINAL_REPORT.md",
                "summary/S01_EXCLUSION_AND_RESOURCE_BOUNDARY.md",
            )
        )
    else:
        rules = {
            TABLE_FILES[0]: counts[TABLE_FILES[0]] == 6,
            TABLE_FILES[1]: counts[TABLE_FILES[1]] == 12,
            TABLE_FILES[2]: counts[TABLE_FILES[2]] == 12,
            TABLE_FILES[4]: counts[TABLE_FILES[4]] >= 6,
            TABLE_FILES[7]: counts[TABLE_FILES[7]] >= 6,
            TABLE_FILES[8]: counts[TABLE_FILES[8]] >= 1,
        }
        combined = pd.read_parquet(root / "all_suite_candidate_od_results.parquet")
        rules["all_suite_candidate_od_results"] = len(combined) == 2400
    if not all(rules.values()):
        raise Phase3ProductionError(f"Summary table row rules failed: {rules}")
    return {"status": "PASS", "row_counts": counts, "rules": rules}


def _verify_figures(root: Path) -> dict[str, Any]:
    details = {}
    for number in range(1, 8):
        matches = list(root.glob(f"figure_{number:02d}_*.png"))
        if len(matches) != 1:
            raise Phase3ProductionError(f"Figure {number} PNG is absent or ambiguous")
        png = matches[0]
        stem = png.stem
        required = [root / f"{stem}.pdf", root / f"{stem}_data.csv", root / f"{stem}_caption.md"]
        if not all(path.is_file() and path.stat().st_size > 0 for path in required):
            raise Phase3ProductionError(f"Figure traceability bundle incomplete: {stem}")
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0, 0))
            width, height = image.size
        details[stem] = {"width": width, "height": height, "dpi": dpi, "status": "PASS"}
    return {"status": "PASS", "figures": details}


def _verify_styles(root: Path) -> dict[str, Any]:
    import xml.etree.ElementTree as ET

    paths = sorted(root.glob("*.qml"))
    if len(paths) < 7:
        raise Phase3ProductionError("Fewer than seven required QML styles were produced")
    for path in paths:
        ET.parse(path)
    return {"status": "PASS", "qml_count": len(paths), "files": [path.name for path in paths]}


def _resource_validation(
    config: Mapping[str, Any], run_root: Path, temp_root: Path
) -> dict[str, Any]:
    receipts = []
    for path in sorted((run_root / "evidence" / "cli_resource_receipts").glob("*.json")):
        receipts.append(json.loads(path.read_text(encoding="utf-8")))
    if not receipts:
        raise Phase3ProductionError("No CLI resource receipts were produced")
    status = json.loads((run_root / "RUN_STATUS.json").read_text(encoding="utf-8"))
    started_key = "closeout_started_utc" if is_scope_amended_closeout(config) else "started_utc"
    started = datetime.fromisoformat(status[started_key].replace("Z", "+00:00"))
    if is_scope_amended_closeout(config):
        receipts = [
            row
            for row in receipts
            if row.get("task_id") == "PREPARE-CLOSEOUT"
            or datetime.fromisoformat(str(row["started_utc"]).replace("Z", "+00:00"))
            >= started
        ]
        if not receipts:
            raise Phase3ProductionError("No closeout CLI resource receipts were produced")
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    if is_scope_amended_closeout(config):
        baseline = status["closeout_resource_baseline"]
        added_disk = max(
            0,
            directory_bytes(run_root)
            + directory_bytes(temp_root)
            - int(baseline["run_root_bytes"])
            - int(baseline["temp_root_bytes"]),
        )
    else:
        added_disk = directory_bytes(run_root) + directory_bytes(temp_root)
    peak_rss = max(int(row["peak_process_tree_rss_bytes"]) for row in receipts)
    peak_processes = max(int(row["peak_process_count"]) for row in receipts)
    checks = {
        "wall_clock_within_authorized_envelope": elapsed
        <= int(config["resources"]["wall_clock_seconds"]),
        "process_tree_rss_under_24gib": peak_rss
        <= int(config["resources"]["process_tree_rss_bytes"]),
        "added_disk_under_150gib": added_disk
        <= int(config["resources"]["added_disk_bytes"]),
        "process_count_within_orchestrator_plus_two_workers": peak_processes <= 3,
    }
    if not all(checks.values()):
        raise Phase3ProductionError(f"Resource hard gate failed: {checks}")
    return {
        "status": "PASS",
        "checks": checks,
        "campaign_elapsed_seconds": elapsed,
        "peak_process_tree_rss_bytes": peak_rss,
        "peak_process_count": peak_processes,
        "logical_added_disk_bytes": added_disk,
        "receipt_count": len(receipts),
    }


def _write_ambiguity_logs(run_root: Path, config: Mapping[str, Any] | None = None) -> None:
    records = [
        {"issue": "s03 relocation", "decision": "Use verified hash-identical hardlinks with copy fallback.", "evidence": "All 1863 manifest members and the accepted result hash passed.", "scientific_impact": "none"},
        {"issue": "regional-pair materialization", "decision": "Keep the complete exact sparse/lazy representation; do not create a global dense or full row registry.", "evidence": "Exact per-region supported-pair counts and deterministic query artifacts are registered.", "scientific_impact": "none"},
        {"issue": "GeoPackage regional-pair burden", "decision": "Use a deterministic region-stratified 5000-pair visualization cap per suite while retaining all bundles/connectors and complete nonspatial support.", "evidence": "Each link layer declares STATIC_PLUS_CAPPED_REGIONAL_PAIRS and total/in-layer counts.", "scientific_impact": "visualization only"},
        {"issue": "production node cost representation", "decision": "Keep link-cost-only nodes and store structural half-endpoint charges; report split-equivalent counts only.", "evidence": "No B-in/B-out production split was instantiated.", "scientific_impact": "matches frozen method"},
        {"issue": "R-ACN singleton control closure", "decision": "Include every terminal-declared ineligible singleton R control as an online R-ACN node derived from its one frozen physical member, without changing accepted R-ACN membership.", "evidence": "Per-suite metadata separates summarized fixed-rule R-ACNs from singleton ineligible controls, and the endpoint closure gate requires every R-ACN terminal parent to resolve.", "scientific_impact": "none; production schema adapter for an already executed frozen control object"},
    ]
    closeout = config is not None and is_scope_amended_closeout(config)
    if closeout:
        records.extend(list(config.get("ambiguity_resolutions", ())))
        json_name = "CLOSEOUT_AMBIGUITY_RESOLUTION_LOG.json"
        markdown_name = "CLOSEOUT_AMBIGUITY_RESOLUTION_LOG.md"
        schema_id = "phase3.five_suite_closeout.ambiguity_log.v1"
        title = "# Phase 3 five-suite closeout ambiguity-resolution log"
    else:
        json_name = "AMBIGUITY_RESOLUTION_LOG.json"
        markdown_name = "AMBIGUITY_RESOLUTION_LOG.md"
        schema_id = "phase3.ambiguity_log.v1"
        title = "# Phase 3 ambiguity-resolution log"
    _write_json_new(
        run_root / "evidence" / json_name,
        {"schema_id": schema_id, "records": records},
    )
    lines = [title, ""]
    for number, row in enumerate(records, 1):
        lines.extend([f"## {number}. {row['issue']}", "", f"Decision: {row['decision']}", "", f"Evidence: {row['evidence']}", "", f"Scientific impact: {row['scientific_impact']}.", ""])
    (run_root / "evidence" / markdown_name).write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def _write_command_log(config: Mapping[str, Any], run_root: Path) -> None:
    receipts = sorted((run_root / "evidence" / "cli_invocations").glob("*.json"))
    lines = ["# Phase 3 production command log", "", "Every real invocation used the explicit phase, step, campaign, run, task list, config, and confirmation guard.", ""]
    for path in receipts:
        row = json.loads(path.read_text(encoding="utf-8"))
        elapsed = row.get("elapsed_seconds")
        elapsed_text = f"{float(elapsed):.3f} s" if elapsed is not None else "in progress"
        lines.extend([f"- `{row['task_id']}` — mode `{row['mode']}`; tasks `{', '.join(row['tasks'])}`; elapsed {elapsed_text}; status `{row['status']}`."])
    (run_root / "evidence" / "COMMAND_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _final_artifact_manifest(run_root: Path) -> pd.DataFrame:
    rows = []
    excluded = {"evidence/FINAL_ARTIFACT_MANIFEST.csv", "RUN_STATUS.json"}
    for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
        relative = path.relative_to(run_root).as_posix()
        if relative in excluded or relative.startswith("handoff/"):
            continue
        rows.append({"relative_path": relative, "absolute_path": str(path.resolve()), "size_bytes": int(path.stat().st_size), "sha256": partC_rev1_file_sha256(path)})
    return pd.DataFrame(rows)


def _handoff_summary(
    config: Mapping[str, Any], validation: Mapping[str, Any], git_head: str
) -> str:
    run_root = Path(str(config["output_root"])).resolve(strict=True)
    if is_scope_amended_closeout(config):
        topology = pd.read_csv(run_root / "tables" / CLOSEOUT_TABLE_FILES[1])
        required = ", ".join(required_suite_ids(config))
        return f"""# Phase 3 five-suite production closeout

## Outcome

The frozen `{METHOD_ID}` construction is complete with notes for the externally
amended required production scope `{required}`. All five accepted suite
products passed exact manifest verification and were reused. The combined
2,000-row common-OD portfolio, normalized SQLite, nationwide GeoPackage,
tables, figures, focused validation, and reproduction evidence passed.

Suite `s01` is transparently retained as
`EXCLUDED_FROM_PHASE3_PRODUCTION_RESOURCE_OUTLIER_STRESS_CASE`. Its 67,490-second
partial attempt remains immutable non-product evidence; no s01 production
topology, OD results, SQLite production view, placeholder, or ACNHN spatial
layer was created.

## Identity

- Run ID: `{config['run_id']}`
- Code commit at sealing: `{git_head}`
- Scope-amendment package SHA-256: `{config['scope_amendment_package']['sha256']}`
- Reviewed blocked handoff SHA-256: `{config['reviewed_blocked_handoff']['sha256']}`
- Required production suites: `{required}`
- Actual online nodes across s02-s06: `{int(topology['actual_online_node_count'].sum()):,}`
- Exact sparse/lazy regional pairs across s02-s06: `{int(topology['exact_regional_pair_count'].sum()):,}`

## Terminal state

- Phase 3A Stage 1: `COMPLETE_WITH_NOTES`
- Winning method: `{METHOD_ID}`
- Required production suites: `s02, s03, s04, s05, s06`
- All required production suites built: `true`
- All six suites built: `false`
- s01: `EXCLUDED_FROM_PHASE3_PRODUCTION_RESOURCE_OUTLIER_STRESS_CASE`
- Phase 3 production construction: `COMPLETE_WITH_NOTES`
- Phase 3 overall: `WAITING_FOR_EXTERNAL_PHASE3_REVIEW`
- Next step authorized: `false`
- Phase 4 authorized: `false`
- Capacity multiplier tuned: `false`
- Cost calibration performed: `false`
- Model training performed: `false`
- ACNHN equilibrium run: `false`

This is an execution handoff, not self-approval. The next action is external
review of the exact synchronization ZIP.
"""
    topology = pd.read_csv(run_root / "tables" / TABLE_FILES[0])
    return f"""# Phase 3 all-suite production completion

## Outcome

The frozen `{METHOD_ID}` construction completed for all six accepted Phase 2
suites in the authorized order. Exact sparse/lazy regional support, the common
200-OD direct and structural diagnostics, deterministic reproduction, per-suite
Parquet, all-suite SQLite, and the nationwide GeoPackage all passed validation.

## Identity

- Run ID: `{config['run_id']}`
- Code commit at sealing: `{git_head}`
- Authority package SHA-256: `{config['authorization_package']['sha256']}`
- Suites built: `{', '.join(SUITE_ORDER)}`
- All-suite actual online nodes: `{int(topology['actual_online_node_count'].sum()):,}`
- All-suite exact regional pairs: `{int(topology['exact_regional_pair_count'].sum()):,}`

## Terminal state

- Phase 3A Stage 1: `COMPLETE_WITH_NOTES`
- Winning method: `{METHOD_ID}`
- All six suites built: `true`
- Phase 3 production construction: `COMPLETE_WITH_NOTES`
- Phase 3 overall: `WAITING_FOR_EXTERNAL_PHASE3_REVIEW`
- Phase 4 authorized: `false`
- Capacity multiplier tuned: `false`
- Cost calibration performed: `false`
- Model training performed: `false`
- ACNHN equilibrium run: `false`

This is an execution handoff, not self-approval. The next action is external
review of the exact synchronization ZIP.
"""


def _write_handoff_top_files(
    config: Mapping[str, Any],
    source: Path,
    summary: str,
    validation: Mapping[str, Any],
    zip_path: Path,
    git_head: str,
    git_status: Sequence[str],
) -> None:
    closeout = is_scope_amended_closeout(config)
    (source / "00_HANDOFF_README.md").write_text(summary, encoding="utf-8", newline="\n")
    (source / "01_STEP_SUMMARY.md").write_text(summary, encoding="utf-8", newline="\n")
    _write_json_new(
        source / "02_HANDOFF_STATUS.json",
        {
            "phase_id": "03",
            "step_id": "all_suite_acnhn",
            "handoff_id": source.name,
            "status": "WAITING_FOR_EXTERNAL_PHASE3_REVIEW",
            "next_step_authorized": False,
            "required_suite_ids": list(required_suite_ids(config)),
            "excluded_suite_ids": list(excluded_suite_ids(config)),
            "all_required_production_suites_built": True,
            "all_six_suites_built": not bool(excluded_suite_ids(config)),
            "s01_disposition": (
                config["scope_amendment"]["excluded_suite"]["disposition"]
                if closeout
                else None
            ),
            "winning_method": METHOD_ID,
            "phase4_authorized": False,
            "code_commit": git_head,
            "companion_binds_final_zip_hash": True,
        },
    )
    (source / "03_DECISIONS_AND_OPEN_QUESTIONS.md").write_text(
        (
            "# Decisions and open questions\n\nThe external scope amendment accepts the reviewed resource-boundary handoff, requires production suites s02-s06, and excludes s01 as a resource-outlier stress case. The winning GEO-GC 1,000 m method and structural seed are unchanged. Reverse whole-path liftability remains a documented nonblocking limitation. No scientific-method question is open; external review of this exact closeout is required.\n"
            if closeout
            else "# Decisions and open questions\n\nThe external decision selected GEO-GC 1,000 m with the structural seed. Reverse liftability remains a documented nonblocking approximation limitation. No scientific-method question is open; external review of the completed portfolio is required.\n"
        ),
        encoding="utf-8", newline="\n",
    )
    (source / "04_VALIDATION_RESULTS.md").write_text(
        (
            "# Validation results\n\nAll amended five-suite scientific and artifact hard gates passed. See `summary_data/FINAL_VALIDATION.json`, `summary_data/phase3_five_suite_final_validation.json`, the SQLite/GeoPackage evidence, and `table_08_phase3_validation_and_reproduction.csv`. The transparent `all_six_suites_built=false` fact is not a failure under the reviewed amendment.\n"
            if closeout
            else "# Validation results\n\nAll final campaign hard gates passed. See `summary_data/FINAL_VALIDATION.json`, the database/GeoPackage evidence, and `table_08_validation_and_reproduction.csv`.\n"
        ),
        encoding="utf-8", newline="\n",
    )
    resource_table = CLOSEOUT_TABLE_FILES[5] if closeout else TABLE_FILES[5]
    shutil.copy2(
        Path(config["output_root"]) / "tables" / resource_table,
        source / "05_RESOURCE_USAGE.csv",
    )
    shutil.copy2(Path(config["output_root"]) / "evidence" / "FINAL_ARTIFACT_MANIFEST.csv", source / "06_ARTIFACT_MANIFEST.csv")
    (source / "08_COMMAND_LOG.md").write_text((Path(config["output_root"]) / "evidence" / "COMMAND_LOG.md").read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    (source / "09_NEXT_STEP_RECOMMENDATION.md").write_text(
        "# Next-step recommendation\n\nRecommendation: externally review the exact synchronization ZIP. Do not start Phase 4. `next_step_authorized=false`.\n",
        encoding="utf-8", newline="\n",
    )
    (source / "10_UPLOAD_THIS_FILE.txt").write_text(f"Upload this exact file:\n{zip_path}\nThe adjacent companion JSON and .zip.sha256 sidecar bind its final size and SHA-256.\n", encoding="utf-8", newline="\n")
    (source / "11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md").write_text(
        (
            "# Changelog from previous handoff\n\nThis handoff supersedes only the completion status of the immutable blocked revision-002 handoff under the explicit external scope amendment. It reuses the five accepted suite products, adds the five-suite SQLite and GeoPackage, all required tables/figures/reports/validation, and preserves s01 as unchanged partial non-product evidence. The historical blocked handoff is not rewritten.\n"
            if closeout
            else "# Changelog from previous handoff\n\nThis is the first all-suite Phase 3 production-completion handoff. It follows the accepted s03 Stage 1 method-selection handoff and adds the selected production construction for all six suites.\n"
        ),
        encoding="utf-8", newline="\n",
    )


def _copy_handoff_payload(config: Mapping[str, Any], source: Path) -> None:
    run_root = Path(str(config["output_root"])).resolve(strict=True)
    project_root = Path(str(config["project_root"])).resolve(strict=True)
    for path in sorted((run_root / "authority").glob("*.md")):
        shutil.copy2(path, source / "governing_docs" / path.name)
    for path in (
        project_root / "partC_doc_icnn" / "20_HANDOFF_SYNC_PROTOCOL.md",
        project_root / "partC_doc_icnn" / "21_REVIEW_APPROVAL_GATES.md",
        project_root / "partC_doc_icnn" / "23_SYNC_BUNDLE_CONTENT_AND_SIZE_POLICY.md",
        project_root / "partC_doc_icnn" / "24_CODE_AND_DATA_SNAPSHOT_POLICY.md",
        project_root / "partC_doc_icnn" / "review_gates" / "phase_03a" / "step_stage1_method_selection" / "phase3_all_suites_authority_20260831" / "REVIEW_DECISION.md",
    ):
        shutil.copy2(path, source / "governing_docs" / path.name)
    for path in sorted((run_root / "authority" / "config").glob("*")):
        shutil.copy2(path, source / "configs" / path.name)
    shutil.copy2(run_root / "authority" / "executed_phase3_all_suite_config.json", source / "configs" / "executed_phase3_all_suite_config.json")
    if is_scope_amended_closeout(config):
        amendment_root = run_root / str(
            config["scope_amendment"]["authority_run_relative_path"]
        )
        amendment_target = source / "governing_docs" / "scope_amendment"
        for path in sorted(item for item in amendment_root.rglob("*") if item.is_file()):
            relative = path.relative_to(amendment_root)
            if path.suffix.lower() == ".zip" or path.stat().st_size > 25 * 1024**2:
                continue
            target = amendment_target / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        review_gate = project_root / str(
            config["scope_amendment"]["review_gate_relative_path"]
        )
        for name in ("REVIEW_DECISION.md", "APPROVAL.json"):
            shutil.copy2(review_gate / name, source / "governing_docs" / f"scope_amendment_{name}")
        shutil.copy2(
            amendment_root / "executed_closeout_config.json",
            source / "configs" / "executed_closeout_config.json",
        )
    for path in sorted((run_root / "tables").glob("*")):
        if path.is_file() and path.stat().st_size <= 25 * 1024**2:
            shutil.copy2(path, source / "summary_data" / path.name)
    for path in sorted((run_root / "evidence").glob("*.json")):
        shutil.copy2(path, source / "summary_data" / path.name)
    for path in sorted((run_root / "validation").glob("*.json")):
        shutil.copy2(path, source / "summary_data" / path.name)
    for path in sorted((run_root / "summary").glob("*.md")):
        shutil.copy2(path, source / "summary_data" / path.name)
    for path in sorted((run_root / "figures").glob("*")):
        if path.is_file() and path.stat().st_size <= 25 * 1024**2:
            shutil.copy2(path, source / "figures" / path.name)
    shutil.copy2(run_root / "evidence" / "geopackage_validation.json", source / "geospatial_summary" / "geopackage_validation.json")
    for path in sorted((run_root / "styles").glob("*.qml")):
        shutil.copy2(path, source / "geospatial_summary" / path.name)
    shutil.copy2(run_root / "evidence" / "COMMAND_LOG.md", source / "log_excerpts" / "COMMAND_LOG.md")
    if is_scope_amended_closeout(config):
        for name in (
            "CLOSEOUT_READINESS.md",
            "CAMPAIGN_FAILURE.md",
            "CAMPAIGN_FAILURE.json",
        ):
            path = run_root / "evidence" / name
            if path.is_file():
                shutil.copy2(path, source / "log_excerpts" / name)
        shutil.copy2(
            run_root / "evidence" / "S01_PARTIAL_ARTIFACT_MANIFEST.csv",
            source / "external_artifact_index" / "S01_PARTIAL_ARTIFACT_MANIFEST.csv",
        )
        binding_table = CLOSEOUT_TABLE_FILES[6]
    else:
        binding_table = TABLE_FILES[9]
    shutil.copy2(
        run_root / "tables" / binding_table,
        source / "external_artifact_index" / binding_table,
    )


def _write_code_evidence(
    config: Mapping[str, Any], source: Path, git_head: str, git_status: Sequence[str]
) -> None:
    project_root = Path(str(config["project_root"])).resolve(strict=True)
    files = [
        "partC_rev1_src/phase3a/authority.py", "partC_rev1_src/phase3a/loaders.py",
        "partC_rev1_src/phase3a/operations.py", "partC_rev1_src/phase3a/production.py",
        "partC_rev1_src/phase3a/production_cli.py", "partC_rev1_src/phase3a/production_portfolio.py",
        "partC_rev1_src/phase3a/production_handoff.py",
        "partC_doc_icnn/phase_3_work/phase3_all_suite_execution_config.json",
        "partC_doc_icnn/phase_3_work/phase3_selected_candidate_matrix.csv",
    ]
    if is_scope_amended_closeout(config):
        files.extend(
            [
                "partC_rev1_src/phase3a/production_closeout.py",
                "partC_doc_icnn/phase_3_work/phase3_five_suite_closeout_execution_config.json",
                str(config["scope_amendment"]["review_gate_relative_path"])
                + "/REVIEW_DECISION.md",
                str(config["scope_amendment"]["review_gate_relative_path"])
                + "/APPROVAL.json",
            ]
        )
    rows = []
    for relative in files:
        path = project_root / relative
        target = source / "code_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        rows.append({"relative_path": relative, "sha256": partC_rev1_file_sha256(path), "size_bytes": path.stat().st_size, "role": "executed_or_direct_dependency"})
    _write_csv_new(source / "07_CODE_CHANGE_INDEX.csv", pd.DataFrame(rows))
    starting = str(config.get("closeout_starting_head", config["reviewed_starting_head"]))
    patch = _git(project_root, "diff", "--binary", f"{starting}..{git_head}")
    (source / "code_diff" / "phase3_from_reviewed_start.patch").write_text(patch, encoding="utf-8", newline="\n")
    (source / "code_diff" / "git_status_before_handoff.txt").write_text("\n".join(git_status) + "\n", encoding="utf-8", newline="\n")


def _source_checksums(source: Path) -> dict[str, str]:
    return {
        path.relative_to(source).as_posix(): partC_rev1_file_sha256(path)
        for path in sorted(item for item in source.rglob("*") if item.is_file() and item.name != "CHECKSUMS.sha256")
    }


def _create_zip(source: Path, zip_path: Path) -> None:
    if zip_path.exists():
        raise Phase3ProductionError(f"Handoff ZIP already exists: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source).as_posix()
            archive.write(path, relative)
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise Phase3ProductionError(f"Handoff ZIP CRC failure: {bad}")


def _fresh_extract_verify(config: Mapping[str, Any], zip_path: Path) -> dict[str, Any]:
    root = (
        Path(str(config["temp_root"])).resolve(strict=True)
        / "handoff_fresh_extract"
        / zip_path.stem
    )
    if root.exists():
        raise Phase3ProductionError(f"Fresh-extraction target exists: {root}")
    root.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(root)
    checksums = {}
    for line in (root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksums[relative] = digest
    failures = [relative for relative, digest in checksums.items() if partC_rev1_file_sha256(root / relative) != digest]
    if failures:
        raise Phase3ProductionError(f"Fresh handoff extraction checksum failures: {failures[:10]}")
    return {"status": "PASS", "member_count": len(checksums) + 1, "checksum_failures": []}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "-c", "core.longpaths=true", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
