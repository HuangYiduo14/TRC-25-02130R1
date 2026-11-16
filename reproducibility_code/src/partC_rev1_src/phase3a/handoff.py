"""Compact Task 1 handoff sealing with manifest and fresh-extraction checks."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from partC_rev1_src.partC_rev1_phase1_handoff import (
    PartCRev1HandoffError,
    partC_rev1_build_manifest,
    partC_rev1_verify_zip_and_fresh_extraction,
    partC_rev1_write_checksums,
)


class Phase3AHandoffError(PartCRev1HandoffError):
    pass


def prepare_task1_handoff_source(
    source_root: str | Path,
    *,
    provenance: str = "phase3a_task1_preflight",
) -> list[dict[str, Any]]:
    """Create stable source controls before the source-tree commit.

    The ZIP companion may then bind the commit containing these exact
    controls without introducing a self-referential dirty working tree.
    """

    source = Path(source_root).resolve(strict=True)
    manifest_path = source / "MANIFEST.csv"
    checksums_path = source / "CHECKSUMS.sha256"
    if manifest_path.exists() or checksums_path.exists():
        raise Phase3AHandoffError("Handoff source controls already exist")
    manifest = partC_rev1_build_manifest(source, provenance=str(provenance))
    with manifest_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)
    partC_rev1_write_checksums(checksums_path, manifest)
    return manifest


def seal_task1_handoff(
    *,
    source_root: str | Path,
    zip_path: str | Path,
    companion_json_path: str | Path,
    extraction_temp_root: str | Path,
    run_id: str,
    source_commit: str,
    status: str,
    external_artifacts: Sequence[Mapping[str, Any]],
    companion_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in {"READY_FOR_EXTERNAL_PREFLIGHT_REVIEW", "BLOCKED_REQUIRES_EXTERNAL_DECISION", "FAILED_PREFLIGHT"}:
        raise Phase3AHandoffError(f"Invalid Task 1 handoff status: {status}")
    source = Path(source_root).resolve(strict=True)
    archive = Path(zip_path).resolve(strict=False)
    companion = Path(companion_json_path).resolve(strict=False)
    sidecar = Path(str(archive) + ".sha256")
    if any(path.exists() for path in (archive, companion, sidecar)):
        raise Phase3AHandoffError("Task 1 handoff destinations are write-new-only")
    if not archive.parent.is_dir() or not companion.parent.is_dir():
        raise Phase3AHandoffError("Task 1 handoff destination parents must exist")
    manifest_path = source / "MANIFEST.csv"
    checksums_path = source / "CHECKSUMS.sha256"
    if manifest_path.exists() != checksums_path.exists():
        raise Phase3AHandoffError("Handoff source controls are incomplete")
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            manifest = list(csv.DictReader(handle))
        # Reuse the established closure validator through the normal ZIP
        # verification below; reject an empty or malformed prepared control.
        if not manifest:
            raise Phase3AHandoffError("Prepared handoff manifest is empty")
    else:
        manifest = prepare_task1_handoff_source(source)
    try:
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
            for path in sorted((item for item in source.rglob("*") if item.is_file()), key=lambda value: value.relative_to(source).as_posix()):
                relative = path.relative_to(source).as_posix()
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                output.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        verification = partC_rev1_verify_zip_and_fresh_extraction(
            archive, manifest, extraction_temp_root
        )
        digest = _sha256(archive)
        sidecar.write_text(f"{digest}  {archive.name}\n", encoding="ascii", newline="\n")
        payload = {
            "schema_id": "phase3a.task1_handoff_companion",
            "schema_version": "phase3a-1.0",
            "status": status,
            "run_id": str(run_id),
            "source_commit": str(source_commit),
            "zip_path": str(archive),
            "zip_size_bytes": int(archive.stat().st_size),
            "zip_sha256": digest,
            "zip_verification": verification,
            "included_artifact_count": len(manifest),
            "external_artifacts": [dict(value) for value in external_artifacts],
            "phase3a_goal_authorized": False,
            "phase3a_task2_authorized": False,
            "phase3b_authorized": False,
            "phase4_authorized": False,
            "next_action": "External review of this exact Task 1 preflight ZIP; do not run Task 2",
        }
        if companion_fields:
            overlap = set(companion_fields).intersection(payload)
            if overlap:
                raise Phase3AHandoffError(
                    f"Companion extension may not replace sealed fields: {sorted(overlap)}"
                )
            payload.update(dict(companion_fields))
        companion.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return payload
    except Exception:
        # A failed seal must not leave a deceptively usable partial archive or
        # companion. Source evidence and accepted inputs remain untouched.
        for path in (archive, sidecar, companion):
            if path.exists():
                path.unlink()
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
