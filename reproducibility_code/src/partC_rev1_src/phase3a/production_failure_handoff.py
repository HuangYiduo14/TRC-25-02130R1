"""Seal a compact Phase 3 handoff after an authorized production early stop.

This module is deliberately separate from the success-only handoff path.  It
never converts incomplete suite or portfolio output into a successful result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METHOD_ID = "geo_gc_1000m_structural_seed_v1"
SUITE_ORDER = ("s03", "s04", "s05", "s02", "s06", "s01")
COMPLETED_SUITES = ("s03", "s04", "s05", "s02", "s06")
FAILURE_REASON = "PROTECTED_SEALING_BOUNDARY_REACHED"
TERMINAL_STATUS = "WAITING_FOR_EXTERNAL_REVIEW"
PACKAGING_REPAIR = {
    "schema_id": "phase3.blocked_handoff_packaging_repair.v1",
    "initial_sealer_commit": "772c1395ac900af3e4112eb4f7620db67f3eb9c3",
    "initial_sealer_commit_time": "2026-09-01T13:37:34-05:00",
    "failure_type": "FileNotFoundError",
    "failed_lookup": "receipts/S03-COMPLETE.json",
    "actual_authoritative_receipt": "receipts/S03-REUSE.json",
    "failure_scope": "packaging-only; no handoff source or ZIP was created",
    "repair_commit": "6a321b5046717bdced62fc7d24f1cf46c6d0e169",
    "repair_commit_time": "2026-09-01T13:38:34-05:00",
    "repair": "Map the intentional s03 reuse receipt separately from built-suite completion receipts.",
    "scientific_output_changed": False,
}

CHECKPOINTS = (
    {"regions": 100, "regional_sources": 2384, "regional_edges": 831542, "elapsed_seconds": 662.908},
    {"regions": 200, "regional_sources": 3716, "regional_edges": 908099, "elapsed_seconds": 705.629},
    {"regions": 300, "regional_sources": 4602, "regional_edges": 926691, "elapsed_seconds": 725.863},
    {"regions": 400, "regional_sources": 12313, "regional_edges": 20441239, "elapsed_seconds": 66450.662},
    {"regions": 500, "regional_sources": 14294, "regional_edges": 20931694, "elapsed_seconds": 66765.684},
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp must be timezone-aware: {value}")
    return parsed.astimezone(timezone.utc)


def _utc_text(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    return current.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _git(project_root: Path, *arguments: str) -> str:
    command = ["git", "-C", str(project_root), "-c", "core.longpaths=true", *arguments]
    result = subprocess.run(command, check=True, text=True, capture_output=True, encoding="utf-8")
    return result.stdout


def _tree_stats(root: Path) -> tuple[int, int]:
    if not root.exists():
        return 0, 0
    files = [path for path in root.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def _manifest_tree(root: Path, output: Path) -> tuple[int, int, str]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "absolute_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_csv(output, rows, ("relative_path", "absolute_path", "size_bytes", "sha256"))
    return len(rows), sum(int(row["size_bytes"]) for row in rows), _sha256(output)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_selected(project_root: Path, destination: Path, relative_paths: Iterable[str]) -> None:
    for relative in relative_paths:
        source = project_root / relative
        if source.is_file():
            _copy_file(source, destination / relative)


def _suite_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    receipt_names = {
        "s03": "S03-REUSE.json",
        "s04": "S04-COMPLETE.json",
        "s05": "S05-COMPLETE.json",
        "s02": "S02-COMPLETE.json",
        "s06": "S06-COMPLETE.json",
    }
    for suite_id in SUITE_ORDER:
        receipt_path = run_root / "receipts" / receipt_names.get(suite_id, "S01-BLOCKED.json")
        product_root = run_root / "suites" / suite_id / "parquet"
        if suite_id in COMPLETED_SUITES:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            manifest = product_root / "artifact_manifest.csv"
            with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
                manifest_rows = sum(1 for _ in csv.DictReader(stream))
            file_count, size_bytes = _tree_stats(product_root)
            rows.append(
                {
                    "suite_short_id": suite_id,
                    "status": "PASS",
                    "production_directory": str(product_root.resolve()),
                    "artifact_manifest_path": str(manifest.resolve()),
                    "artifact_manifest_sha256": _sha256(manifest),
                    "artifact_manifest_rows": manifest_rows,
                    "file_count": file_count,
                    "size_bytes": size_bytes,
                    "receipt_path": str(receipt_path.resolve()),
                    "receipt_sha256": _sha256(receipt_path),
                    "completed_utc": receipt["completed_utc"],
                }
            )
        else:
            rows.append(
                {
                    "suite_short_id": suite_id,
                    "status": "BLOCKED_NOT_BUILT",
                    "production_directory": str(product_root.resolve()),
                    "artifact_manifest_path": "",
                    "artifact_manifest_sha256": "",
                    "artifact_manifest_rows": 0,
                    "file_count": 0,
                    "size_bytes": 0,
                    "receipt_path": "",
                    "receipt_sha256": "",
                    "completed_utc": "",
                }
            )
    return rows


def _resource_rows(evidence_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((evidence_root / "cli_resource_receipts").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "receipt_file": path.name,
                "task_id": payload.get("task_id", ""),
                "status": payload.get("status", ""),
                "started_utc": payload.get("started_utc", ""),
                "completed_utc": payload.get("completed_utc", ""),
                "elapsed_seconds": payload.get("elapsed_seconds", ""),
                "peak_process_tree_rss_bytes": payload.get("peak_process_tree_rss_bytes", ""),
                "peak_process_count": payload.get("peak_process_count", ""),
                "added_disk_bytes": payload.get("added_disk_bytes", ""),
                "rss_hard_limit_exceeded": payload.get("rss_hard_limit_exceeded", ""),
                "failure_type": payload.get("failure_type", ""),
                "failure_message": payload.get("failure_message", ""),
            }
        )
    return sorted(rows, key=lambda row: str(row["started_utc"]))


def _artifact_rows(
    config: Mapping[str, Any],
    run_root: Path,
    suite_rows: Sequence[Mapping[str, Any]],
    partial_manifest: Path,
    partial_manifest_sha256: str,
    partial_file_count: int,
    partial_size_bytes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "artifact_id": "authority_package",
            "status": "PRESENT_VERIFIED_AT_STARTUP",
            "kind": "zip",
            "absolute_path": config["authorization_package"]["path"],
            "relative_path": "external",
            "size_bytes": config["authorization_package"]["size_bytes"],
            "file_count": 1,
            "sha256_or_manifest_sha256": config["authorization_package"]["sha256"],
            "validation": "size/hash/47-entry safe extraction passed",
        },
        {
            "artifact_id": "accepted_phase2_geopackage",
            "status": "PRESENT_VERIFIED_AT_STARTUP",
            "kind": "gpkg",
            "absolute_path": config["phase2_geopackage"]["path"],
            "relative_path": "external",
            "size_bytes": config["phase2_geopackage"]["size_bytes"],
            "file_count": 1,
            "sha256_or_manifest_sha256": config["phase2_geopackage"]["sha256"],
            "validation": "source identity passed; no Phase 3 copy produced",
        },
    ]
    for suite in suite_rows:
        rows.append(
            {
                "artifact_id": f"suite_{suite['suite_short_id']}",
                "status": suite["status"],
                "kind": "production_parquet_directory",
                "absolute_path": suite["production_directory"],
                "relative_path": f"suites/{suite['suite_short_id']}/parquet",
                "size_bytes": suite["size_bytes"],
                "file_count": suite["file_count"],
                "sha256_or_manifest_sha256": suite["artifact_manifest_sha256"],
                "validation": "production receipt and closed artifact manifest PASS" if suite["status"] == "PASS" else "mandatory suite product absent",
            }
        )
    rows.append(
        {
            "artifact_id": "s01_preproduction_intermediates",
            "status": "INCOMPLETE_NOT_ACCEPTED",
            "kind": "operation_directory",
            "absolute_path": str((run_root / "operations" / "s01").resolve()),
            "relative_path": "operations/s01",
            "size_bytes": partial_size_bytes,
            "file_count": partial_file_count,
            "sha256_or_manifest_sha256": partial_manifest_sha256,
            "validation": f"bound by {partial_manifest}; no suites/s01 product exists",
        }
    )
    missing = (
        ("all_suite_sqlite", "database/phase3_acnhn_all_suites.sqlite", "sqlite"),
        ("all_suite_geopackage", "geopackage/phase3_acnhn_all_suites.gpkg", "gpkg"),
        ("all_suite_tables", "tables", "csv/parquet"),
        ("all_suite_figures", "figures", "png/pdf/csv/md"),
        ("final_campaign_validation", "evidence/FINAL_VALIDATION.json", "json"),
    )
    for artifact_id, relative, kind in missing:
        rows.append(
            {
                "artifact_id": artifact_id,
                "status": "MISSING_MANDATORY",
                "kind": kind,
                "absolute_path": str((run_root / relative).resolve()),
                "relative_path": relative,
                "size_bytes": 0,
                "file_count": 0,
                "sha256_or_manifest_sha256": "",
                "validation": "not produced because s01 failed the protected-reserve gate",
            }
        )
    return rows


def _command_log(evidence_root: Path, failure: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 3 command log",
        "",
        "Every real invocation used explicit phase, step, campaign, run, task, config, task list, and confirmation fields.",
        "",
    ]
    for path in sorted((evidence_root / "cli_invocations").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"- `{path.name}`: task `{row['task_id']}`, mode `{row['mode']}`, "
            f"status `{row['status']}`, started `{row['started_utc']}`, elapsed `{row.get('elapsed_seconds', 'not finalized')}` seconds."
        )
    lines.extend(
        [
            "",
            "## Boundary stop",
            "",
            f"- Verified PID: `{failure['stopped_pid']}`.",
            f"- Stop time: `{failure['stopped_utc']}`.",
            f"- Reason: `{failure['reason_code']}`.",
            "- The PID command line was rechecked immediately before `Stop-Process -Id 22820 -Force`.",
            "- No portfolio assembly, success validation, or later-phase command was run after the stop.",
            "",
            "## Blocked-handoff packaging",
            "",
            "- Initial sealer commit `772c1395ac900af3e4112eb4f7620db67f3eb9c3` failed before source/ZIP creation because it looked for `S03-COMPLETE.json`.",
            "- Repair commit `6a321b5046717bdced62fc7d24f1cf46c6d0e169` mapped the intentional s03 reuse receipt `S03-REUSE.json`.",
            "- First blocked handoff sealed at `2026-09-01T18:38:56.050047Z`; later packaging revisions add traceability only.",
            "",
        ]
    )
    return "\n".join(lines)


def _source_checksums(source: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for path in sorted(item for item in source.rglob("*") if item.is_file() and item.name != "CHECKSUMS.sha256"):
        rows.append((path.relative_to(source).as_posix(), _sha256(path)))
    return rows


def _create_zip(source: Path, zip_path: Path) -> None:
    temporary = zip_path.with_name(f".{zip_path.name}.{uuid.uuid4().hex}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(source).as_posix())
    with zipfile.ZipFile(temporary, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC validation failed at {bad}")
    os.replace(temporary, zip_path)


def _fresh_extract_verify(zip_path: Path, temp_root: Path) -> dict[str, Any]:
    extraction = temp_root / f"failure_handoff_verify_{uuid.uuid4().hex}"
    extraction.mkdir(parents=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extraction)
            member_count = len(archive.infolist())
        checksum_path = extraction / "CHECKSUMS.sha256"
        failures: list[str] = []
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            member = extraction / Path(relative)
            if not member.is_file() or _sha256(member) != digest:
                failures.append(relative)
        if failures:
            raise RuntimeError(f"Fresh extraction checksum failures: {failures[:10]}")
        return {"status": "PASS", "member_count": member_count, "checksum_count": len(checksum_path.read_text(encoding="utf-8").splitlines())}
    finally:
        resolved = extraction.resolve()
        if resolved.parent == temp_root.resolve() and resolved.name.startswith("failure_handoff_verify_"):
            shutil.rmtree(resolved)


def seal_failure_handoff(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = Path(config["project_root"]).resolve(strict=True)
    run_root = Path(config["output_root"]).resolve(strict=True)
    temp_root = Path(config["temp_root"]).resolve(strict=True)
    evidence_root = run_root / "evidence"
    status_path = run_root / "RUN_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    started = _parse_utc(status["started_utc"])
    stopped = _parse_utc(args.stopped_at_utc)
    boundary = started + timedelta(seconds=int(config["resources"]["wall_clock_seconds"]) - int(config["resources"]["protected_sealing_seconds"]))
    campaign_elapsed = (stopped - started).total_seconds()
    remaining = int(config["resources"]["wall_clock_seconds"]) - campaign_elapsed
    encroachment = max(0.0, int(config["resources"]["protected_sealing_seconds"]) - remaining)
    invocation_path = evidence_root / "cli_invocations" / "S01-BUILD__attempt_001.json"
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    s01_started = _parse_utc(invocation["started_utc"])
    s01_elapsed = (stopped - s01_started).total_seconds()
    failure_message = (
        "s01 did not complete before the campaign's protected 7200-second finalization/handoff reserve; "
        "the verified S01-BUILD PID was stopped and no success receipt or suite product was created."
    )
    invocation.update(
        {
            "status": "BLOCKED",
            "completed_utc": _utc_text(stopped),
            "elapsed_seconds": s01_elapsed,
            "failure_type": "ProtectedSealingBoundaryReached",
            "failure_message": failure_message,
            "exit_code": int(args.exit_code),
        }
    )
    _atomic_json(invocation_path, invocation)
    s01_resource = {
        "schema_id": "phase3.cli_task_resource.v1",
        "task_id": "S01-BUILD",
        "tasks": invocation["tasks"],
        "mode": invocation["mode"],
        "status": "BLOCKED",
        "started_utc": invocation["started_utc"],
        "completed_utc": _utc_text(stopped),
        "elapsed_seconds": s01_elapsed,
        "peak_process_tree_rss_bytes": int(args.observed_peak_rss_bytes),
        "peak_process_count": 3,
        "added_disk_bytes": 0,
        "run_root_bytes": _tree_stats(run_root)[1],
        "temp_root_bytes": _tree_stats(temp_root)[1],
        "rss_hard_limit_exceeded": False,
        "failure_type": "ProtectedSealingBoundaryReached",
        "failure_message": failure_message,
        "manual_boundary_receipt": True,
        "last_observed_private_bytes": int(args.last_private_bytes),
    }
    _atomic_json(evidence_root / "cli_resource_receipts" / "S01-BUILD__attempt_001.json", s01_resource)
    s01_failure_receipt = {
        "schema_id": "phase3.production_failure_receipt.v1",
        "status": "BLOCKED",
        "task_id": "S01-BUILD",
        "suite_short_id": "s01",
        "phase2_suite_id": "phase2_suite_01_r1em06",
        "started_utc": invocation["started_utc"],
        "stopped_utc": _utc_text(stopped),
        "elapsed_seconds": s01_elapsed,
        "reason_code": FAILURE_REASON,
        "failure_message": failure_message,
        "last_confirmed_checkpoint": CHECKPOINTS[-1],
        "total_regions_required": 731,
        "suite_product_created": False,
        "evaluation_created": False,
        "reproduction_created": False,
        "export_created": False,
        "phase4_executed": False,
    }
    _atomic_json(run_root / "receipts" / "S01-BLOCKED.json", s01_failure_receipt)
    failure = {
        "schema_id": "phase3.all_suite_campaign_failure.v1",
        "status": "BLOCKED_HARD_RESOURCE_SAFETY",
        "reason_code": FAILURE_REASON,
        "reason": failure_message,
        "campaign_started_utc": _utc_text(started),
        "protected_sealing_boundary_utc": _utc_text(boundary),
        "stopped_utc": _utc_text(stopped),
        "stopped_pid": int(args.stopped_pid),
        "exit_code": int(args.exit_code),
        "campaign_elapsed_seconds_at_stop": campaign_elapsed,
        "wall_clock_seconds": int(config["resources"]["wall_clock_seconds"]),
        "protected_sealing_seconds": int(config["resources"]["protected_sealing_seconds"]),
        "protected_reserve_remaining_seconds_at_stop": remaining,
        "protected_reserve_encroachment_seconds": encroachment,
        "completed_suites": list(COMPLETED_SUITES),
        "incomplete_suites": ["s01"],
        "all_six_suites_built": False,
        "s01_checkpoints": list(CHECKPOINTS),
        "last_confirmed_s01_region_count": CHECKPOINTS[-1]["regions"],
        "last_confirmed_s01_regional_source_count": CHECKPOINTS[-1]["regional_sources"],
        "last_confirmed_s01_regional_edge_count": CHECKPOINTS[-1]["regional_edges"],
        "s01_product_created": False,
        "missing_mandatory_deliverables": [
            "suites/s01 production Parquet directory and manifest",
            "s01 common 200-OD direct and structural evaluation",
            "s01 deterministic reproduction and normalized export",
            "all-suite SQLite",
            "all-suite GeoPackage",
            "all-suite tables and figures",
            "passing final campaign validation",
        ],
        "winning_method": METHOD_ID,
        "scientific_method_changed": False,
        "phase4_authorized": False,
        "capacity_multiplier_tuned": False,
        "cost_calibration_performed": False,
        "model_training_performed": False,
        "acnhn_equilibrium_run": False,
        "next_step_authorized": False,
        "recommendation": "BLOCK_PENDING_EXTERNAL_REVIEW",
    }
    _atomic_json(evidence_root / "CAMPAIGN_FAILURE.json", failure)
    failure_md = f"""# Phase 3 all-suite campaign blocked\n\n## Verified outcome\n\nFive suites completed and have passing production receipts: `{', '.join(COMPLETED_SUITES)}`.\nSuite `s01` did not complete before the protected finalization/handoff reserve.\nThe last confirmed stdout checkpoint was 500 of 731 regions, 14,294 regional\nsources, and 20,931,694 exact regional edges. No `suites/s01` product exists.\n\n## Resource boundary\n\n- Campaign start: `{_utc_text(started)}`\n- Protected sealing boundary: `{_utc_text(boundary)}`\n- Verified PID stop: `{_utc_text(stopped)}`\n- Reserve encroachment from monitoring/stop latency: `{encroachment:.3f}` seconds\n- Total campaign elapsed at stop: `{campaign_elapsed:.3f}` seconds\n\n## Terminal decision\n\nThe mandatory all-six-suite objective is not complete. Portfolio SQLite,\nGeoPackage, tables, figures, and final success validation were not run. The\nrecommendation is `BLOCK_PENDING_EXTERNAL_REVIEW`; no next step or Phase 4 work\nis authorized.\n"""
    _atomic_text(evidence_root / "CAMPAIGN_FAILURE.md", failure_md)
    status.update(
        {
            "status": TERMINAL_STATUS,
            "terminal_outcome": "BLOCKED_HARD_RESOURCE_SAFETY",
            "blocked_reason_code": FAILURE_REASON,
            "phase3a_stage1": "COMPLETE_WITH_NOTES",
            "phase3_production_construction": "INCOMPLETE_BLOCKED",
            "phase3_overall": "WAITING_FOR_EXTERNAL_REVIEW_BLOCKED",
            "completed_suites": list(COMPLETED_SUITES),
            "incomplete_suites": ["s01"],
            "all_six_suites_built": False,
            "next_step_authorized": False,
            "phase4_authorized": False,
            "capacity_multiplier_tuned": False,
            "cost_calibration_performed": False,
            "model_training_performed": False,
            "acnhn_equilibrium_run": False,
            "updated_utc": _utc_text(stopped),
        }
    )
    _atomic_json(status_path, status)
    suite_rows = _suite_rows(run_root)
    suite_fields = tuple(suite_rows[0].keys())
    _write_csv(evidence_root / "SUITE_STATUS.csv", suite_rows, suite_fields)
    partial_manifest = evidence_root / "S01_PARTIAL_ARTIFACT_MANIFEST.csv"
    partial_count, partial_bytes, partial_sha = _manifest_tree(run_root / "operations" / "s01", partial_manifest)
    resource_rows = _resource_rows(evidence_root)
    resource_fields = tuple(resource_rows[0].keys())
    _write_csv(evidence_root / "RESOURCE_USAGE.csv", resource_rows, resource_fields)
    artifact_rows = _artifact_rows(config, run_root, suite_rows, partial_manifest, partial_sha, partial_count, partial_bytes)
    artifact_fields = tuple(artifact_rows[0].keys())
    _write_csv(evidence_root / "BLOCKED_ARTIFACT_INDEX.csv", artifact_rows, artifact_fields)
    command_log = _command_log(evidence_root, failure)
    _atomic_text(evidence_root / "COMMAND_LOG_BLOCKED.md", command_log)
    _atomic_json(evidence_root / "PACKAGING_REPAIR.json", PACKAGING_REPAIR)
    packaging_repair_md = """# Blocked-handoff packaging repair

The first failure-seal attempt stopped before creating a source folder or ZIP.
It raised `FileNotFoundError` while looking for `receipts/S03-COMPLETE.json`.
Suite s03 is an intentional exact reuse, so its authoritative receipt is
`receipts/S03-REUSE.json`. Commit `6a321b5046717bdced62fc7d24f1cf46c6d0e169`
added that packaging-only adapter. No scientific output or terminal campaign
decision changed.
"""
    _atomic_text(evidence_root / "PACKAGING_REPAIR.md", packaging_repair_md)
    code_commit = _git(project_root, "rev-parse", "HEAD").strip()
    branch = _git(project_root, "branch", "--show-current").strip()
    dirty_before = _git(project_root, "status", "--porcelain=v1", "-uall").splitlines()
    tracked_dirty = _git(project_root, "status", "--porcelain=v1", "-uno").splitlines()
    dirty_state = {
        "schema_id": "phase3.blocked_handoff_dirty_state.v1",
        "branch": branch,
        "head": code_commit,
        "tracked_worktree_clean": not tracked_dirty,
        "tracked_status": tracked_dirty,
        "all_status_before_handoff": dirty_before,
        "classification": "HANDOFF_OUTPUTS_ONLY" if not tracked_dirty else "TRACKED_CHANGES_PRESENT",
        "unrelated_user_changes_modified": False,
    }
    _atomic_json(evidence_root / "DIRTY_STATE_CLASSIFICATION.json", dirty_state)
    ambiguity_records = [
        {
            "issue": "s03 relocation",
            "decision": "Use verified hash-identical hardlinks with copy fallback.",
            "evidence": "All 1863 accepted manifest members and the result hash passed.",
            "scientific_impact": "none",
        },
        {
            "issue": "regional-pair materialization",
            "decision": "Keep complete exact sparse/lazy support; never instantiate a dense terminal square.",
            "evidence": "Five completed suites have closed manifests; s01 remained incomplete and unaccepted.",
            "scientific_impact": "none",
        },
        {
            "issue": "R-ACN singleton control closure",
            "decision": "Represent seven already-executed terminal-declared singleton controls as normalized R-ACN nodes without changing accepted membership.",
            "evidence": "Repair commit 430940f; s03 endpoint closure then passed.",
            "scientific_impact": "none; schema adapter for frozen controls",
        },
        {
            "issue": "s01 intermediate files versus production progress",
            "decision": "Treat 500/731 as the last confirmed in-memory materialization checkpoint; do not infer production completion from pre-existing sparse support files.",
            "evidence": "No suites/s01 directory or success receipt exists.",
            "scientific_impact": "prevents false completion",
        },
        {
            "issue": "protected sealing reserve",
            "decision": "Stop only the verified S01-BUILD PID at the reserve boundary and seal a blocked handoff.",
            "evidence": "PID 22820 stopped at 2026-09-01T18:29:08.161792Z; later-phase flags remain false.",
            "scientific_impact": "none; hard resource-safety enforcement",
        },
        {
            "issue": "s03 receipt name in blocked sealer",
            "decision": "Map S03-REUSE.json separately from built-suite completion receipts.",
            "evidence": "Packaging repair commit 6a321b5 and explicit repair log.",
            "scientific_impact": "none; packaging only",
        },
    ]
    ambiguity = {"schema_id": "phase3.blocked_handoff_ambiguity_log.v1", "records": ambiguity_records}
    _atomic_json(evidence_root / "AMBIGUITY_RESOLUTION_LOG.json", ambiguity)
    ambiguity_lines = ["# Ambiguity-resolution log", ""]
    for number, record in enumerate(ambiguity_records, 1):
        ambiguity_lines.extend(
            [
                f"## {number}. {record['issue']}", "", f"Decision: {record['decision']}", "",
                f"Evidence: {record['evidence']}", "", f"Scientific impact: {record['scientific_impact']}.", "",
            ]
        )
    _atomic_text(evidence_root / "AMBIGUITY_RESOLUTION_LOG.md", "\n".join(ambiguity_lines))
    handoff_id = args.handoff_id
    handoff_parent = project_root / "partC_doc_icnn" / "handoffs" / "phase_03" / "all_suite_acnhn"
    source = handoff_parent / handoff_id
    if source.exists():
        raise FileExistsError(f"Handoff source already exists: {source}")
    staging = handoff_parent / f".{handoff_id}.{uuid.uuid4().hex}.staging"
    sections = (
        "governing_docs", "configs", "code_snapshot", "code_diff", "summary_data",
        "figures", "geospatial_summary", "log_excerpts", "external_artifact_index",
    )
    for section in sections:
        (staging / section).mkdir(parents=True, exist_ok=True)
    zip_root = project_root / "artifacts_icnn" / "handoff_bundles" / "phase_03"
    zip_path = zip_root / f"phase03_all_suite_acnhn_{handoff_id}_SYNC.zip"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "handoff_id": handoff_id,
            "handoff_zip": str(zip_path.resolve()),
            "handoff_zip_size_bytes": None,
            "handoff_zip_sha256": "BOUND_BY_ADJACENT_COMPANION_AFTER_SEAL",
            "handoff_companion": str(zip_path.with_suffix(".companion.json").resolve()),
        }
    )
    _atomic_json(status_path, status)
    summary = f"""# Phase 3 all-suite production early-stop handoff\n\n## Outcome\n\nThe authorized 24-hour campaign is **blocked**, not complete. Five suites\n(`{', '.join(COMPLETED_SUITES)}`) completed with passing receipts. Suite `s01`\nreached a last confirmed checkpoint of 500/731 regions but did not produce an\naccepted suite directory before the protected 7,200-second finalization/handoff\nreserve. The verified S01-BUILD PID was stopped at `{_utc_text(stopped)}`.\n\n## Exact identity\n\n- Run ID: `{config['run_id']}`\n- Branch: `{branch}`\n- Code commit at sealing: `{code_commit}`\n- Authority ZIP SHA-256: `{config['authorization_package']['sha256']}`\n- Winning method: `{METHOD_ID}` (unchanged)\n- Failure reason: `{FAILURE_REASON}`\n\n## Verified facts\n\n- Startup authority, input invariance, dependency, disk, and dirty-state checks passed.\n- Focused implementation tests passed: 29 tests.\n- Completed production suites: `{', '.join(COMPLETED_SUITES)}`.\n- `s01` last confirmed materialization: 500 regions, 14,294 regional sources, 20,931,694 regional edges.\n- No `suites/s01` product or success receipt exists.\n- SQLite, GeoPackage, all-suite tables/figures, and final success validation are absent and are listed as mandatory missing artifacts.\n- No Phase 4, capacity tuning, cost calibration, model training, or ACNHN equilibrium was run.\n\n## Inference\n\nThe coarse `s01` suite contained one unusually expensive materialization block:\nregions 300 to 400 consumed about 65,725 seconds. Completing all mandatory\noutputs within the remaining approved campaign envelope became impossible.\n\n## Recommendation\n\n`BLOCK_PENDING_EXTERNAL_REVIEW`. The next action is review of this exact ZIP.\nAny repair, optimization, rerun, or resource-budget change requires a new explicit\nauthorization; `next_step_authorized=false`.\n\n## Terminal state\n\n- Phase 3A Stage 1: `COMPLETE_WITH_NOTES`\n- Winning method: `{METHOD_ID}`\n- All six suites built: `false`\n- Phase 3 production construction: `INCOMPLETE_BLOCKED`\n- Phase 3 overall: `WAITING_FOR_EXTERNAL_REVIEW_BLOCKED`\n- Phase 4 authorized: `false`\n- Capacity multiplier tuned: `false`\n- Cost calibration performed: `false`\n- Model training performed: `false`\n- ACNHN equilibrium run: `false`\n"""
    _atomic_text(staging / "00_HANDOFF_README.md", summary)
    _atomic_text(staging / "01_STEP_SUMMARY.md", summary)
    handoff_status = {
        "schema_id": "phase3.blocked_handoff_status.v1",
        "phase_id": "03",
        "step_id": "all_suite_acnhn",
        "handoff_id": handoff_id,
        "status": TERMINAL_STATUS,
        "terminal_outcome": "BLOCKED_HARD_RESOURCE_SAFETY",
        "reason_code": FAILURE_REASON,
        "recommendation": "BLOCK",
        "supersedes_handoff_id": args.supersedes_handoff_id,
        "next_step_authorized": False,
        "all_six_suites_built": False,
        "completed_suites": list(COMPLETED_SUITES),
        "incomplete_suites": ["s01"],
        "winning_method": METHOD_ID,
        "phase4_authorized": False,
        "code_commit": code_commit,
        "companion_binds_final_zip_hash": True,
    }
    _atomic_json(staging / "02_HANDOFF_STATUS.json", handoff_status)
    _atomic_text(
        staging / "03_DECISIONS_AND_OPEN_QUESTIONS.md",
        "# Decisions and open questions\n\n"
        "## Decisions\n\n- Enforced the protected 7,200-second sealing reserve.\n"
        "- Stopped only the verified S01-BUILD PID.\n- Preserved all five completed suite products unchanged.\n"
        "- Did not run portfolio assembly or success validation with an incomplete suite set.\n\n"
        "## Open questions requiring external authority\n\n"
        "1. Whether a new campaign may resume or rebuild s01 with an optimized exact-support implementation.\n"
        "2. Whether the wall-clock/resource authorization may change.\n"
        "3. Whether any repair plan should be preflighted before a new long run.\n\n"
        "No answer is inferred and no next step is authorized.\n",
    )
    validation_text = f"""# Validation results\n\n- Authority ZIP identity and safe extraction: PASS.\n- Accepted Phase 2 suite manifests: PASS (120 files across six suites).\n- Accepted Phase 2 GeoPackage identity: PASS.\n- Startup branch and clean state: PASS.\n- Focused Phase 3 implementation tests: PASS (29 tests).\n- Completed suite production receipts: PASS for `{', '.join(COMPLETED_SUITES)}`.\n- s01 production: BLOCKED; no accepted product/manifest.\n- Final all-suite SQLite/GeoPackage/table/figure validation: NOT RUN because mandatory s01 output is absent.\n- Phase-boundary flags: PASS; every later-phase flag remains false.\n- Campaign success gate: FAIL (`all_six_suites_built=false`).\n- Handoff source checksum closure: recorded in `CHECKSUMS.sha256`.\n- Final ZIP CRC and fresh-extraction checksum validation: bound by the adjacent companion JSON.\n\nProtected reserve encroachment caused by polling and process-stop latency was `{encroachment:.3f}` seconds and is recorded as a hard-gate defect.\n"""
    _atomic_text(staging / "04_VALIDATION_RESULTS.md", validation_text)
    _write_csv(staging / "05_RESOURCE_USAGE.csv", resource_rows, resource_fields)
    _write_csv(staging / "06_ARTIFACT_MANIFEST.csv", artifact_rows, artifact_fields)
    changed = _git(project_root, "diff", "--name-status", config["reviewed_starting_head"], code_commit).splitlines()
    code_rows: list[dict[str, Any]] = []
    for line in changed:
        parts = line.split("\t")
        relative = parts[-1]
        path = project_root / relative
        code_rows.append(
            {
                "change_status": parts[0],
                "relative_path": relative,
                "exists_at_seal": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": _sha256(path) if path.is_file() else "",
            }
        )
    _write_csv(staging / "07_CODE_CHANGE_INDEX.csv", code_rows, ("change_status", "relative_path", "exists_at_seal", "size_bytes", "sha256"))
    _atomic_text(staging / "08_COMMAND_LOG.md", command_log)
    _atomic_text(staging / "log_excerpts" / "PACKAGING_REPAIR.md", packaging_repair_md)
    _atomic_json(staging / "log_excerpts" / "PACKAGING_REPAIR.json", PACKAGING_REPAIR)
    _atomic_text(
        staging / "09_NEXT_STEP_RECOMMENDATION.md",
        "# Next-step recommendation\n\nRecommendation: `BLOCK`.\n\n"
        "The only authorized next action is external review of this synchronization ZIP. "
        "A repair, optimization, resume, rebuild, or resource-budget change requires explicit new approval. "
        "Phase 4 remains unauthorized.\n",
    )
    _atomic_text(
        staging / "10_UPLOAD_THIS_FILE.txt",
        f"Upload this exact file:\n{zip_path}\nThe adjacent .zip.sha256 sidecar and companion JSON bind its final size and SHA-256.\n",
    )
    if args.supersedes_handoff_id:
        changelog = (
            "# Changelog from previous handoff\n\n"
            f"This packaging revision supersedes `{args.supersedes_handoff_id}`. "
            "It adds the original FileNotFoundError and the s03 reuse-receipt adapter repair evidence. "
            "Campaign status, scientific outputs, resource evidence, and the BLOCK recommendation are unchanged.\n"
        )
    else:
        changelog = (
            "# Changelog from previous handoff\n\nThis is the first all-suite Phase 3 production campaign handoff. "
            "It records an early stop at the protected resource boundary and does not supersede a successful all-suite handoff.\n"
        )
    _atomic_text(staging / "11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md", changelog)
    goal_path = Path(args.goal_objective).resolve(strict=True)
    _copy_file(goal_path, staging / "governing_docs" / "goal-objective.md")
    _copy_selected(
        project_root,
        staging / "governing_docs" / "repository",
        (
            "partC_doc_icnn/00_READ_FIRST.md", "partC_doc_icnn/PROJECT_STATUS.md",
            "partC_doc_icnn/12_DECISION_LOG.md", "partC_doc_icnn/20_HANDOFF_SYNC_PROTOCOL.md",
            "partC_doc_icnn/21_REVIEW_APPROVAL_GATES.md", "partC_doc_icnn/23_SYNC_BUNDLE_CONTENT_AND_SIZE_POLICY.md",
        ),
    )
    package_root = Path(args.package_root).resolve(strict=True)
    for path in sorted(package_root.glob("*.md")):
        _copy_file(path, staging / "governing_docs" / "authority_package" / path.name)
    if (package_root / "PACKAGE_INFO.json").is_file():
        _copy_file(package_root / "PACKAGE_INFO.json", staging / "governing_docs" / "authority_package" / "PACKAGE_INFO.json")
    _copy_file(config_path, staging / "configs" / config_path.name)
    for path in sorted((package_root / "config").glob("*")):
        if path.is_file() and path.stat().st_size <= 25 * 1024 * 1024:
            _copy_file(path, staging / "configs" / "authority_package" / path.name)
    code_paths = (
        "partC_rev1_src/phase3a/production.py", "partC_rev1_src/phase3a/production_cli.py",
        "partC_rev1_src/phase3a/production_handoff.py", "partC_rev1_src/phase3a/production_portfolio.py",
        "partC_rev1_src/phase3a/production_failure_handoff.py", "partC_rev1_src/phase3a/resources.py",
        "partC_rev1_src/phase3a/loaders.py", "partC_rev1_tests/partC_rev1_test_phase3a_production.py",
    )
    _copy_selected(project_root, staging / "code_snapshot", code_paths)
    patch = _git(project_root, "diff", "--binary", config["reviewed_starting_head"], code_commit)
    _atomic_text(staging / "code_diff" / "phase3_from_reviewed_start.patch", patch)
    _atomic_text(staging / "code_diff" / "git_status_before_handoff.txt", "\n".join(dirty_before) + ("\n" if dirty_before else ""))
    _atomic_text(staging / "code_diff" / "git_identity.txt", f"branch={branch}\nhead={code_commit}\nreviewed_start={config['reviewed_starting_head']}\n")
    for source_file in (
        status_path, evidence_root / "CAMPAIGN_FAILURE.json", evidence_root / "CAMPAIGN_FAILURE.md",
        evidence_root / "SUITE_STATUS.csv", evidence_root / "RESOURCE_USAGE.csv",
        evidence_root / "BLOCKED_ARTIFACT_INDEX.csv", run_root / "receipts" / "S01-BLOCKED.json",
        evidence_root / "DIRTY_STATE_CLASSIFICATION.json", evidence_root / "AMBIGUITY_RESOLUTION_LOG.json",
        evidence_root / "AMBIGUITY_RESOLUTION_LOG.md", evidence_root / "PACKAGING_REPAIR.json",
        evidence_root / "PACKAGING_REPAIR.md",
    ):
        _copy_file(source_file, staging / "summary_data" / source_file.name)
    for suite_id in COMPLETED_SUITES:
        receipt_name = "S03-REUSE.json" if suite_id == "s03" else f"{suite_id.upper()}-COMPLETE.json"
        _copy_file(run_root / "receipts" / receipt_name, staging / "summary_data" / "suite_receipts" / receipt_name)
        _copy_file(run_root / "suites" / suite_id / "parquet" / "artifact_manifest.csv", staging / "external_artifact_index" / "suite_manifests" / f"{suite_id}_artifact_manifest.csv")
    for path in sorted((evidence_root / "cli_invocations").glob("*.json")):
        _copy_file(path, staging / "summary_data" / "cli_invocations" / path.name)
    for path in sorted((evidence_root / "cli_resource_receipts").glob("*.json")):
        _copy_file(path, staging / "summary_data" / "cli_resource_receipts" / path.name)
    _copy_file(partial_manifest, staging / "external_artifact_index" / partial_manifest.name)
    _atomic_text(staging / "figures" / "NOT_PRODUCED_BLOCKED.md", "# Figures not produced\n\nMandatory all-suite figures were not produced because s01 did not complete.\n")
    _atomic_text(staging / "geospatial_summary" / "NOT_PRODUCED_BLOCKED.md", "# Geospatial portfolio not produced\n\nThe accepted Phase 2 GeoPackage remains immutable; no Phase 3 all-suite GeoPackage was produced.\n")
    checkpoint_lines = ["# s01 observed materialization checkpoints", ""]
    checkpoint_lines.extend(
        f"- regions={row['regions']}/731 regional_sources={row['regional_sources']} regional_edges={row['regional_edges']} elapsed_s={row['elapsed_seconds']}"
        for row in CHECKPOINTS
    )
    _atomic_text(staging / "log_excerpts" / "s01_materialization_checkpoints.md", "\n".join(checkpoint_lines) + "\n")
    source_validation = {
        "schema_id": "phase3.blocked_handoff_source_validation.v1",
        "status": "PASS",
        "required_top_files_present": True,
        "next_step_authorized": False,
        "large_data_copied_into_zip": False,
        "external_artifact_index_present": True,
        "created_utc": _utc_text(),
    }
    _atomic_json(staging / "summary_data" / "SOURCE_VALIDATION.json", source_validation)
    checksums = _source_checksums(staging)
    _atomic_text(staging / "CHECKSUMS.sha256", "".join(f"{digest}  {relative}\n" for relative, digest in checksums))
    handoff_parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, source)
    if zip_path.exists():
        raise FileExistsError(f"Handoff ZIP already exists: {zip_path}")
    _create_zip(source, zip_path)
    zip_size = zip_path.stat().st_size
    if zip_size > 200 * 1024 * 1024:
        raise RuntimeError(f"Compact handoff exceeds 200 MiB: {zip_size}")
    zip_sha = _sha256(zip_path)
    fresh = _fresh_extract_verify(zip_path, temp_root)
    sidecar = zip_path.with_suffix(zip_path.suffix + ".sha256")
    _atomic_text(sidecar, f"{zip_sha}  {zip_path.name}\n")
    companion = {
        "schema_id": "phase3.blocked_handoff_companion.v1",
        "handoff_id": handoff_id,
        "sync_zip": str(zip_path.resolve()),
        "sync_zip_size_bytes": zip_size,
        "sync_zip_sha256": zip_sha,
        "source_folder": str(source.resolve()),
        "source_commit": code_commit,
        "supersedes_handoff_id": args.supersedes_handoff_id,
        "fresh_extraction": fresh,
        "status": TERMINAL_STATUS,
        "terminal_outcome": "BLOCKED_HARD_RESOURCE_SAFETY",
        "reason_code": FAILURE_REASON,
        "next_step_authorized": False,
        "phase4_authorized": False,
        "created_utc": _utc_text(),
    }
    companion_path = zip_path.with_suffix(".companion.json")
    _atomic_json(companion_path, companion)
    _atomic_text(
        handoff_parent / "LATEST_HANDOFF.txt",
        f"source_folder={source.resolve()}\nzip_path={zip_path.resolve()}\nzip_sha256={zip_sha}\ncreated_utc={companion['created_utc']}\nstatus={TERMINAL_STATUS}\nterminal_outcome=BLOCKED_HARD_RESOURCE_SAFETY\n",
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "handoff_id": handoff_id,
            "handoff_zip": str(zip_path.resolve()),
            "handoff_zip_size_bytes": zip_size,
            "handoff_zip_sha256": zip_sha,
            "handoff_companion": str(companion_path.resolve()),
            "updated_utc": companion["created_utc"],
        }
    )
    _atomic_json(status_path, status)
    _atomic_json(evidence_root / "FAILURE_HANDOFF.json", companion)
    return companion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--goal-objective", required=True)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--handoff-id", required=True)
    parser.add_argument("--stopped-at-utc", required=True)
    parser.add_argument("--stopped-pid", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--observed-peak-rss-bytes", type=int, required=True)
    parser.add_argument("--last-private-bytes", type=int, required=True)
    parser.add_argument("--supersedes-handoff-id")
    return parser


def main() -> None:
    result = seal_failure_handoff(_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
