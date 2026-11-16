"""Versioned identifiers, strict serialization, and artifact manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from partC_rev1_src.partC_rev1_hashing import partC_rev1_file_sha256


PHASE3A_SCHEMA_VERSION = "phase3a-1.0"
PHASE3A_ID_NAMESPACE = "partC_rev1/phase3a/1.0"


class Phase3ASchemaError(ValueError):
    """Raised when a Phase 3A object violates a frozen schema contract."""


def _finite_json(value: Any, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise Phase3ASchemaError(f"Nonfinite JSON value at {location}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_json(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_json(item, f"{location}[{index}]")


def canonical_json_bytes(value: Any) -> bytes:
    if is_dataclass(value):
        value = asdict(value)
    _finite_json(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Phase3ASchemaError("Value is not canonical finite JSON") from exc


def stable_object_id(
    object_type: str,
    *,
    suite_id: str,
    candidate_id: str,
    members: Sequence[str | int] = (),
    ordered_members: bool = False,
    signature: Mapping[str, Any] | Sequence[Any] | str | None = None,
) -> str:
    """Return a row-order-independent, versioned stable object identifier."""

    if not object_type or not suite_id or not candidate_id:
        raise Phase3ASchemaError("object_type, suite_id, and candidate_id are required")
    member_text = [str(value) for value in members]
    if not ordered_members:
        member_text.sort(
            key=lambda value: (
                not value.lstrip("-").isdigit(),
                int(value) if value.lstrip("-").isdigit() else value,
            )
        )
    payload = {
        "namespace": PHASE3A_ID_NAMESPACE,
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "object_type": object_type,
        "suite_id": suite_id,
        "candidate_id": candidate_id,
        "members": member_text,
        "signature": signature,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"p3a_{object_type.lower()}_{digest[:24]}"


def write_json_new(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload) + b"\n"
    try:
        with destination.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise Phase3ASchemaError(f"Write-new-only path already exists: {destination}") from exc
    return destination


def write_text_new(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if "\x00" in text:
        raise Phase3ASchemaError("Text output contains a NUL byte")
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            if text and not text.endswith("\n"):
                handle.write("\n")
    except FileExistsError as exc:
        raise Phase3ASchemaError(f"Write-new-only path already exists: {destination}") from exc
    return destination


def validate_candidate_matrix(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_ids = {
        "identity_bacn",
        *(f"geo_{algorithm}_{scale:04d}m" for algorithm in ("cl", "gc") for scale in (500, 1000, 2000, 4000, 8000)),
        "geo_sl_1000m",
        "geo_sl_4000m",
        "aug_cl_030s",
        "aug_gc_030s",
        "aug_sl_030s",
        "aug_cl_120s",
        "aug_gc_120s",
        "aug_sl_120s",
    }
    actual = [row.get("candidate_id", "") for row in rows]
    if len(actual) != len(set(actual)):
        raise Phase3ASchemaError("Candidate matrix contains duplicate candidate IDs")
    if set(actual) != expected_ids:
        raise Phase3ASchemaError(
            f"Candidate matrix differs from frozen grid; missing={sorted(expected_ids-set(actual))}, extra={sorted(set(actual)-expected_ids)}"
        )
    required_columns = {
        "candidate_id", "round", "family", "algorithm", "parameter_value", "parameter_unit",
        "production_eligible", "full_evaluation_default", "notes",
    }
    if not rows or set(rows[0]) != required_columns:
        raise Phase3ASchemaError("Candidate matrix columns differ from the frozen schema")
    if len(rows) != 19:
        raise Phase3ASchemaError("Candidate matrix must contain exactly 19 frozen rows")
    for row in rows:
        family, algorithm = row["family"], row["algorithm"]
        if family == "GEO" and algorithm not in {"CL", "GC", "SL"}:
            raise Phase3ASchemaError("GEO candidate has an invalid algorithm")
        if family == "AUG" and algorithm not in {"CL", "GC", "SL"}:
            raise Phase3ASchemaError("AUG candidate has an invalid algorithm")
        if algorithm == "SL" and row["production_eligible"].lower() != "false":
            raise Phase3ASchemaError("SL ablations may not be production eligible")
    return rows


def build_artifact_manifest(root: str | Path) -> list[dict[str, Any]]:
    base = Path(root).resolve(strict=True)
    rows: list[dict[str, Any]] = []
    for path in sorted((item for item in base.rglob("*") if item.is_file()), key=lambda p: p.relative_to(base).as_posix()):
        relative = PurePosixPath(path.relative_to(base).as_posix())
        if relative.is_absolute() or ".." in relative.parts:
            raise Phase3ASchemaError(f"Unsafe manifest path: {relative}")
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": partC_rev1_file_sha256(path),
            }
        )
    return rows


def verify_artifact_manifest(
    root: str | Path, rows: Iterable[Mapping[str, Any]], *, strict_closure: bool = True
) -> None:
    base = Path(root).resolve(strict=True)
    records = list(rows)
    declared: set[str] = set()
    for row in records:
        relative = PurePosixPath(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise Phase3ASchemaError(f"Unsafe manifest entry: {relative}")
        path = base.joinpath(*relative.parts)
        if not path.is_file():
            raise Phase3ASchemaError(f"Manifest member missing: {relative}")
        if int(path.stat().st_size) != int(row["size_bytes"]):
            raise Phase3ASchemaError(f"Manifest size mismatch: {relative}")
        if partC_rev1_file_sha256(path) != str(row["sha256"]):
            raise Phase3ASchemaError(f"Manifest hash mismatch: {relative}")
        if relative.as_posix() in declared:
            raise Phase3ASchemaError(f"Duplicate manifest path: {relative}")
        declared.add(relative.as_posix())
    if strict_closure:
        actual = {
            path.relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.is_file()
        }
        if actual != declared:
            raise Phase3ASchemaError(
                f"Manifest closure mismatch: missing={sorted(declared-actual)}, extra={sorted(actual-declared)}"
            )


ARTIFACT_SCHEMAS: dict[str, dict[str, Any]] = {
    "bacn_membership": {
        "schema_id": "phase3a.bacn_membership",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["candidate_id", "partC_rev1_node_uid"],
        "units": {"parameter_value": "declared_by_parameter_unit"},
    },
    "bacn_summary": {
        "schema_id": "phase3a.bacn_summary",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["candidate_id", "bacn_id"],
        "units": {
            "exact_geodesic_diameter_m": "meters",
            "requested_geodesic_threshold_m": "meters",
            "maximum_mutual_aug_fftt_s": "seconds",
            "requested_aug_threshold_s": "seconds",
        },
    },
    "aug_witness_registry": {
        "schema_id": "phase3a.aug_witness_registry",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["witness_id"],
        "units": {"fftt_s": "seconds", "length_m": "meters"},
    },
    "bundle_membership": {
        "schema_id": "phase3a.bundle_membership",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["candidate_id", "directed_arc_id"],
        "units": {"member_fftt_s": "seconds", "member_length_m": "meters"},
    },
    "bundle_summary": {
        "schema_id": "phase3a.bundle_summary",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["candidate_id", "bundle_id"],
        "units": {
            "minimum_member_fftt_s": "seconds", "median_member_fftt_s": "seconds",
            "maximum_member_fftt_s": "seconds", "direct_diagnostic_fftt_s": "seconds",
        },
    },
    "suppressed_backbone_arcs": {
        "schema_id": "phase3a.suppressed_backbone_arcs",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["candidate_id", "directed_arc_id"],
        "units": {"fftt_s": "seconds", "length_m": "meters"},
    },
    "racn_membership": {
        "schema_id": "phase3a.racn_membership",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["candidate_id", "physical_portal_node_uid"],
        "units": {},
    },
    "racn_summary": {
        "schema_id": "phase3a.racn_summary",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["candidate_id", "racn_id"],
        "units": {
            "exact_geodesic_diameter_m": "meters",
            "maximum_mutual_restricted_fftt_s": "seconds",
        },
    },
    "consolidated_terminals": {
        "schema_id": "phase3a.consolidated_terminals",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["candidate_id", "region_id", "terminal_id"],
        "units": {},
    },
    "regional_pair_cache": {
        "schema_id": "phase3a.regional_pair_cache",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["candidate_id", "region_id", "source_terminal_id", "target_terminal_id"],
        "units": {"direct_fftt_s": "seconds"},
    },
    "regional_pair_cache_index": {
        "schema_id": "phase3a.regional_pair_cache_index",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["candidate_id", "region_id", "source_terminal_id"],
        "units": {"restricted_dijkstra_seconds": "seconds"},
    },
    "physical_to_candidate_lookup": {
        "schema_id": "phase3a.physical_to_candidate_lookup",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["candidate_id"],
        "units": {},
    },
    "tracer_interface": {
        "schema_id": "phase3a.tracer_interface",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["candidate_id"],
        "units": {"flow_arrays": "OD demand units"},
    },
    "od_sample": {
        "schema_id": "phase3a.od_sample",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["sample_od_id"],
        "units": {"demand": "2017_truck_annual_thousand_tons", "geodesic_distance_km": "kilometers"},
    },
    "candidate_od_result": {
        "schema_id": "phase3a.candidate_od_result",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["candidate_id", "sample_od_id", "diagnostic_mode"],
        "units": {
            "physical_fftt_s": "seconds", "projected_generic_fftt_s": "seconds",
            "scalar_acnhn_fftt_s": "seconds", "lifted_physical_fftt_s": "seconds",
        },
    },
    "capacity_unit_audit": {
        "schema_id": "phase3a.capacity_unit_audit",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["base_field", "remaining_field"],
        "units": {"source_capacity_fields": "declared_or_ktons_per_year_fallback"},
    },
    "candidate_registry": {
        "schema_id": "phase3a.candidate_registry",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["candidate_id"],
        "units": {"actual_seconds": "seconds", "peak_rss_bytes": "bytes"},
    },
    "resource_ledger": {
        "schema_id": "phase3a.resource_ledger",
        "schema_version": PHASE3A_SCHEMA_VERSION,
        "primary_key": ["command_id"],
        "units": {"elapsed_seconds": "seconds", "peak_process_tree_rss_bytes": "bytes", "added_disk_bytes": "bytes"},
    },
}
