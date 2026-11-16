"""Authority/readiness preparation for the externally amended Phase 3 closeout."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import pandas as pd

from partC_rev1_src.partC_rev1_hashing import partC_rev1_file_sha256

from .production import (
    METHOD_ID,
    Phase3ProductionError,
    _replace_json,
    _write_json_new,
    excluded_suite_ids,
    required_suite_ids,
    utc_now,
)
from .resources import directory_bytes


def prepare_five_suite_closeout(
    config: Mapping[str, Any], config_path: str | Path
) -> dict[str, Any]:
    """Bind the amendment and persist the bounded closeout-readiness audit."""

    required = required_suite_ids(config)
    excluded = excluded_suite_ids(config)
    if required != ("s02", "s03", "s04", "s05", "s06") or excluded != ("s01",):
        raise Phase3ProductionError("Five-suite closeout scope is not the authorized s02-s06/s01 partition")
    run_root = Path(str(config["output_root"])).resolve(strict=True)
    temp_root = Path(str(config["temp_root"])).resolve(strict=True)
    project_root = Path(str(config["project_root"])).resolve(strict=True)
    status_path = run_root / "RUN_STATUS.json"
    old_status = json.loads(status_path.read_text(encoding="utf-8"))
    if old_status.get("all_six_suites_built") is not False:
        raise Phase3ProductionError("Closeout preparation requires the transparent all-six=false fact")
    if (run_root / "suites" / "s01").exists():
        raise Phase3ProductionError("s01 production output exists; the exclusion contract forbids using it")
    initial_usage = shutil.disk_usage(run_root)
    baseline = {
        "run_root_bytes": directory_bytes(run_root),
        "temp_root_bytes": directory_bytes(temp_root),
        "free_disk_bytes": initial_usage.free,
    }

    scope_package = Path(str(config["scope_amendment_package"]["path"])).resolve(strict=True)
    if scope_package.stat().st_size != int(config["scope_amendment_package"]["size_bytes"]):
        raise Phase3ProductionError("Scope-amendment package size changed")
    if partC_rev1_file_sha256(scope_package) != str(config["scope_amendment_package"]["sha256"]):
        raise Phase3ProductionError("Scope-amendment package SHA-256 changed")
    extract_root = temp_root / "phase3_closeout_skip_s01_20260901_authority"
    authority_relative = str(config["scope_amendment"]["authority_run_relative_path"])
    authority_root = run_root / authority_relative
    if extract_root.exists() or authority_root.exists():
        raise Phase3ProductionError("Closeout authority extraction/destination must be write-new")
    extract_root.mkdir(parents=True)
    with zipfile.ZipFile(scope_package, "r") as archive:
        infos = archive.infolist()
        if len(infos) != int(config["scope_amendment_package"]["entry_count"]):
            raise Phase3ProductionError("Scope-amendment package entry count changed")
        for info in infos:
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise Phase3ProductionError(f"Unsafe scope-amendment member: {info.filename}")
        if archive.testzip() is not None:
            raise Phase3ProductionError("Scope-amendment package CRC validation failed")
        archive.extractall(extract_root)
    package_checksums = _verify_checksum_file(extract_root)
    shutil.copytree(extract_root, authority_root)
    shutil.copy2(Path(config_path).resolve(strict=True), authority_root / "executed_closeout_config.json")

    embedded = authority_root / str(config["reviewed_blocked_handoff"]["package_member"])
    live_blocked = Path(str(config["reviewed_blocked_handoff"]["path"])).resolve(strict=True)
    blocked_expected_size = int(config["reviewed_blocked_handoff"]["size_bytes"])
    blocked_expected_hash = str(config["reviewed_blocked_handoff"]["sha256"])
    for path in (embedded, live_blocked):
        if path.stat().st_size != blocked_expected_size or partC_rev1_file_sha256(path) != blocked_expected_hash:
            raise Phase3ProductionError(f"Reviewed blocked handoff binding failed: {path}")
    nested_checksums = _verify_zip_checksums(embedded)

    binding_rows = pd.read_csv(authority_root / "config" / "completed_suite_bindings.csv")
    suite_checks: dict[str, Any] = {}
    for suite in required:
        selected = binding_rows[binding_rows["suite_short_id"] == suite]
        if len(selected) != 1 or selected.iloc[0]["status"] != "PASS":
            raise Phase3ProductionError(f"Completed-suite binding is absent or invalid: {suite}")
        binding = selected.iloc[0]
        product_root = Path(str(binding["production_directory"])).resolve(strict=True)
        manifest = Path(str(binding["artifact_manifest_path"])).resolve(strict=True)
        receipt = Path(str(binding["receipt_path"])).resolve(strict=True)
        manifest_check = _verify_suite_manifest(product_root)
        checks = {
            "manifest_sha256": partC_rev1_file_sha256(manifest)
            == str(binding["artifact_manifest_sha256"]),
            "manifest_rows": manifest_check["manifest_rows"]
            == int(binding["artifact_manifest_rows"]),
            "file_count": manifest_check["file_count"] == int(binding["file_count"]),
            "size_bytes": manifest_check["size_bytes"] == int(binding["size_bytes"]),
            "receipt_sha256": partC_rev1_file_sha256(receipt) == str(binding["receipt_sha256"]),
        }
        if not all(checks.values()):
            raise Phase3ProductionError(f"Completed-suite readiness binding failed: {suite}: {checks}")
        suite_checks[suite] = {"status": "PASS", **checks, **manifest_check}

    od_check = _verify_common_od_results(run_root, required)
    gpkg = Path(str(config["phase2_geopackage"]["path"])).resolve(strict=True)
    gpkg_hash = partC_rev1_file_sha256(gpkg)
    if gpkg.stat().st_size != int(config["phase2_geopackage"]["size_bytes"]) or gpkg_hash != str(
        config["phase2_geopackage"]["sha256"]
    ):
        raise Phase3ProductionError("Accepted Phase 2 GeoPackage binding changed")
    with sqlite3.connect(f"file:{gpkg.as_posix()}?mode=ro", uri=True) as connection:
        gpkg_counts = {
            "directed_links_portfolio": int(
                connection.execute("SELECT COUNT(*) FROM directed_links_portfolio").fetchone()[0]
            ),
            "physical_nodes": int(
                connection.execute("SELECT COUNT(*) FROM physical_nodes").fetchone()[0]
            ),
        }
    if gpkg_counts != {
        "directed_links_portfolio": int(config["phase2_geopackage"]["directed_link_count"]),
        "physical_nodes": int(config["phase2_geopackage"]["physical_node_count"]),
    }:
        raise Phase3ProductionError("Accepted Phase 2 GeoPackage row counts changed")

    if initial_usage.free < int(config["resources"]["added_disk_bytes"]):
        raise Phase3ProductionError("Free disk is below the independent closeout disk envelope")
    repository = _repository_snapshot(project_root)
    if repository["branch"] != config["expected_branch"]:
        raise Phase3ProductionError("Repository branch differs from the amended closeout authority")

    s01_root = run_root / "operations" / "s01"
    s01_partial = {
        "path": str(s01_root),
        "file_count": len([path for path in s01_root.rglob("*") if path.is_file()]),
        "size_bytes": directory_bytes(s01_root),
        "directory_sha256": _directory_digest(s01_root),
        "manifest_path": str(run_root / "evidence" / "S01_PARTIAL_ARTIFACT_MANIFEST.csv"),
        "manifest_sha256": partC_rev1_file_sha256(
            run_root / "evidence" / "S01_PARTIAL_ARTIFACT_MANIFEST.csv"
        ),
        "artifact_status": "PARTIAL_NONPRODUCT_EVIDENCE",
    }
    started_utc = utc_now()
    readiness = {
        "schema_id": "phase3.five_suite_closeout.readiness.v1",
        "status": "PASS",
        "started_utc": started_utc,
        "decision_id": config["scope_amendment"]["decision_id"],
        "required_suite_ids": list(required),
        "excluded_suite_ids": list(excluded),
        "repository": repository,
        "scope_package": {
            "path": str(scope_package),
            "size_bytes": scope_package.stat().st_size,
            "sha256": partC_rev1_file_sha256(scope_package),
            "checksum_entries": package_checksums,
        },
        "reviewed_blocked_handoff": {
            "embedded_path": str(embedded),
            "live_path": str(live_blocked),
            "size_bytes": blocked_expected_size,
            "sha256": blocked_expected_hash,
            "internal_checksum_entries": nested_checksums,
        },
        "completed_suite_checks": suite_checks,
        "common_od_check": od_check,
        "phase2_geopackage": {
            "path": str(gpkg),
            "size_bytes": gpkg.stat().st_size,
            "sha256": gpkg_hash,
            "row_counts": gpkg_counts,
            "sqlite_readonly_reopen": "PASS",
        },
        "s01_partial_evidence": s01_partial,
        "resource_baseline": baseline,
        "writable_output_verified": True,
        "phase4_authorized": False,
    }
    before_status = run_root / "evidence" / "RUN_STATUS_BEFORE_FIVE_SUITE_CLOSEOUT.json"
    _write_json_new(before_status, old_status)
    _write_json_new(run_root / "evidence" / "CLOSEOUT_READINESS.json", readiness)
    (run_root / "evidence" / "CLOSEOUT_READINESS.md").write_text(
        _readiness_markdown(readiness), encoding="utf-8", newline="\n"
    )
    for directory in (run_root / "summary", run_root / "validation"):
        directory.mkdir()

    status = dict(old_status)
    status.update(
        {
            "schema_id": "phase3.five_suite_closeout.run_status.v1",
            "status": "EXECUTING",
            "closeout_started_utc": started_utc,
            "scope_amendment_decision_id": config["scope_amendment"]["decision_id"],
            "scope_amendment_package_sha256": config["scope_amendment_package"]["sha256"],
            "reviewed_blocked_handoff_sha256": blocked_expected_hash,
            "required_suite_ids": list(required),
            "excluded_suite_ids": list(excluded),
            "all_required_production_suites_built": True,
            "all_six_suites_built": False,
            "s01_disposition": config["scope_amendment"]["excluded_suite"]["disposition"],
            "s01_artifact_status": "PARTIAL_NONPRODUCT_EVIDENCE",
            "phase3_production_construction": "CLOSEOUT_EXECUTING",
            "phase3_overall": "EXECUTING_AUTHORIZED_FIVE_SUITE_CLOSEOUT",
            "next_step_authorized": False,
            "phase4_authorized": False,
            "capacity_multiplier_tuned": False,
            "cost_calibration_performed": False,
            "model_training_performed": False,
            "acnhn_equilibrium_run": False,
            "winning_method": METHOD_ID,
            "closeout_resource_budget": dict(config["resources"]),
            "closeout_resource_baseline": baseline,
            "updated_utc": utc_now(),
        }
    )
    _replace_json(status_path, status)
    receipt = {
        "schema_id": "phase3.five_suite_closeout.receipt.v1",
        "task_id": "PREPARE-CLOSEOUT",
        "status": "PASS",
        "completed_utc": utc_now(),
        "meaningful_output_count": 5,
        "required_suite_ids": list(required),
        "excluded_suite_ids": list(excluded),
        "readiness_path": str(run_root / "evidence" / "CLOSEOUT_READINESS.json"),
        "scope_package_sha256": config["scope_amendment_package"]["sha256"],
        "reviewed_blocked_handoff_sha256": blocked_expected_hash,
    }
    _write_json_new(run_root / "receipts" / "PREPARE-CLOSEOUT.json", receipt)
    return receipt


def _verify_checksum_file(root: Path) -> int:
    failures = []
    count = 0
    for line in (root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        count += 1
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or partC_rev1_file_sha256(path) != digest:
            failures.append(relative)
    if failures:
        raise Phase3ProductionError(f"Scope-package checksum failures: {failures[:10]}")
    return count


def _verify_zip_checksums(path: Path) -> int:
    with zipfile.ZipFile(path, "r") as archive:
        checksum_text = archive.read("CHECKSUMS.sha256").decode("utf-8")
        members = {info.filename: info for info in archive.infolist()}
        failures = []
        count = 0
        for line in checksum_text.splitlines():
            if not line.strip():
                continue
            digest, relative = line.split("  ", 1)
            count += 1
            if relative not in members or hashlib.sha256(archive.read(relative)).hexdigest() != digest:
                failures.append(relative)
    if failures:
        raise Phase3ProductionError(f"Blocked-handoff internal checksum failures: {failures[:10]}")
    if count != 95:
        raise Phase3ProductionError(f"Blocked-handoff checksum entry count differs: {count}")
    return count


def _verify_suite_manifest(root: Path) -> dict[str, int]:
    manifest_path = root / "artifact_manifest.csv"
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8-sig", newline="")))
    failures = []
    for row in rows:
        path = root.joinpath(*PurePosixPath(row["relative_path"]).parts)
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or partC_rev1_file_sha256(path) != row["sha256"]
        ):
            failures.append(row["relative_path"])
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    declared = {row["relative_path"] for row in rows} | {"artifact_manifest.csv"}
    if actual != declared:
        failures.append("manifest_closure")
    if failures:
        raise Phase3ProductionError(f"Completed-suite manifest failures at {root}: {failures[:10]}")
    return {
        "manifest_rows": len(rows),
        "file_count": len(actual),
        "size_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
    }


def _verify_common_od_results(run_root: Path, suites: tuple[str, ...]) -> dict[str, Any]:
    result_frames = {
        suite: pd.read_parquet(
            run_root / "suites" / suite / "parquet" / "candidate_od_results.parquet"
        )
        for suite in suites
    }
    sample_frames = {
        suite: pd.read_parquet(
            run_root / "operations" / suite / "round4_evaluation" / "od_sample.parquet"
        )
        for suite in suites
    }
    reference_ids = set(result_frames[suites[0]]["sample_od_id"].astype(str))
    reference_physical = set(
        map(
            tuple,
            result_frames[suites[0]][
                ["sample_od_id", "physical_path_id", "physical_fftt_s"]
            ].drop_duplicates().to_numpy(),
        )
    )
    per_suite = {}
    for suite in suites:
        results = result_frames[suite]
        sample = sample_frames[suite]
        modes = results["diagnostic_mode"].value_counts().to_dict()
        physical = set(
            map(
                tuple,
                results[["sample_od_id", "physical_path_id", "physical_fftt_s"]]
                .drop_duplicates()
                .to_numpy(),
            )
        )
        checks = {
            "result_rows_400": len(results) == 400,
            "unique_od_count_200": results["sample_od_id"].nunique() == 200,
            "direct_rows_200": modes.get("direct", 0) == 200,
            "structural_rows_200": modes.get("structural", 0) == 200,
            "common_result_od_ids": set(results["sample_od_id"].astype(str)) == reference_ids,
            "common_physical_paths_and_fftt": physical == reference_physical,
            "sample_rows_200": len(sample) == 200,
            "sample_origins_40": sample["origin_uid"].nunique() == 40,
            "sample_ids_match_results": set(sample["sample_od_id"].astype(str)) == reference_ids,
        }
        if not all(checks.values()):
            raise Phase3ProductionError(f"Common OD readiness failed for {suite}: {checks}")
        per_suite[suite] = checks
    return {
        "status": "PASS",
        "unique_od_count": len(reference_ids),
        "origin_count": sample_frames[suites[0]]["origin_uid"].nunique(),
        "combined_result_rows": sum(len(frame) for frame in result_frames.values()),
        "per_suite": per_suite,
    }


def _repository_snapshot(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), "-c", "core.longpaths=true", *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout

    status = git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    return {
        "branch": git("branch", "--show-current").strip(),
        "head": git("rev-parse", "HEAD").strip(),
        "status_porcelain": status,
        "tracked_diff": git("diff", "--binary", "--no-ext-diff"),
        "tracked_diff_stat": git("diff", "--stat", "--no-ext-diff"),
        "untracked_path_count": sum(line.startswith("?? ") for line in status),
        "tracked_change_count": sum(not line.startswith("?? ") for line in status),
    }


def _directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(partC_rev1_file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _readiness_markdown(readiness: Mapping[str, Any]) -> str:
    repository = readiness["repository"]
    return f"""# Phase 3 five-suite closeout readiness

Status: `PASS`.

- Decision: `{readiness['decision_id']}`
- Branch / HEAD: `{repository['branch']}` / `{repository['head']}`
- Tracked changes at execution snapshot: `{repository['tracked_change_count']}`
- Preserved untracked paths: `{repository['untracked_path_count']}`
- Scope-package checksum entries: `{readiness['scope_package']['checksum_entries']}`
- Reviewed blocked-handoff internal checksum entries: `{readiness['reviewed_blocked_handoff']['internal_checksum_entries']}`
- Required suites: `{', '.join(readiness['required_suite_ids'])}`
- Excluded suite: `s01` as `PARTIAL_NONPRODUCT_EVIDENCE`
- Common results: `{readiness['common_od_check']['combined_result_rows']}` rows, `{readiness['common_od_check']['unique_od_count']}` ODs, `{readiness['common_od_check']['origin_count']}` origins
- Phase 2 GeoPackage read-only reopen: `PASS`
- Free disk at baseline: `{readiness['resource_baseline']['free_disk_bytes']}` bytes
- Phase 4 authorized: `false`

The dirty state is compatible: tracked scientific changes are absent at the
execution snapshot, and preserved untracked paths are prior handoff evidence.
No reset, clean, stash, deletion, or s01 continuation occurred.
"""
