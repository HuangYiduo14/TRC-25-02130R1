"""Phase 1 authoritative-input manifest and fail-closed access contracts.

This module contains no scientific solver logic.  It resolves an ignored
machine-local root registry, validates a tracked logical manifest, rejects
unsafe destinations, and opens SQLite/GeoPackage inputs strictly read-only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import jsonschema


SUPPORTED_SCHEMA_VERSION = "1.0"
EXPECTED_INPUT_ROLES = (
    "graph_npz",
    "graph_nodes_parquet",
    "graph_arcs_parquet",
    "graph_metadata",
    "od_npz",
    "od_metadata",
    "rich_sqlite",
    "parent_gpkg",
    "corrected_terminal_registry",
)
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[/\\]")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class PartCRev1ValidationDiagnostic:
    code: str
    severity: str
    message: str
    role: str | None = None
    path: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PartCRev1ValidationReport:
    validation_level: str
    status: str = "RUNNING"
    started_at: str = field(default_factory=_utc_now)
    finished_at: str | None = None
    checks: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[PartCRev1ValidationDiagnostic] = field(default_factory=list)
    resolved_inputs: dict[str, str] = field(default_factory=dict)
    resource_usage: dict[str, Any] = field(default_factory=dict)

    def add_check(self, code: str, status: str, **details: Any) -> None:
        self.checks.append({"code": code, "status": status, **details})

    def add_diagnostic(self, diagnostic: PartCRev1ValidationDiagnostic) -> None:
        self.diagnostics.append(diagnostic)

    def finish(self, status: str) -> None:
        self.status = status
        self.finished_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_level": self.validation_level,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "checks": self.checks,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "resolved_inputs": self.resolved_inputs,
            "resource_usage": self.resource_usage,
        }


class PartCRev1InputValidationError(RuntimeError):
    code = "PARTC_REV1_INPUT_VALIDATION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        role: str | None = None,
        path: str | Path | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic = PartCRev1ValidationDiagnostic(
            code=self.code,
            severity="ERROR",
            message=message,
            role=role,
            path=None if path is None else str(path),
            details=dict(details or {}),
        )


class PartCRev1InputSchemaError(PartCRev1InputValidationError):
    code = "PARTC_REV1_INPUT_SCHEMA_ERROR"


class PartCRev1InputPathError(PartCRev1InputValidationError):
    code = "PARTC_REV1_INPUT_PATH_ERROR"


class PartCRev1InputIdentityError(PartCRev1InputValidationError):
    code = "PARTC_REV1_INPUT_IDENTITY_ERROR"


class PartCRev1InputSignatureError(PartCRev1InputValidationError):
    code = "PARTC_REV1_INPUT_SIGNATURE_ERROR"


class PartCRev1ReadOnlyError(PartCRev1InputValidationError):
    code = "PARTC_REV1_READ_ONLY_ERROR"


class PartCRev1ResourceLimitError(PartCRev1InputValidationError):
    code = "PARTC_REV1_RESOURCE_LIMIT_ERROR"


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path)
    try:
        raw = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PartCRev1InputSchemaError(
            f"Cannot read {label} as UTF-8: {exc}", path=resolved
        ) from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PartCRev1InputSchemaError(
            f"Malformed {label} JSON: {exc}", path=resolved
        ) from exc
    if not isinstance(value, dict):
        raise PartCRev1InputSchemaError(f"{label} must be a JSON object", path=resolved)
    return value


def _validate_json_schema(payload: Mapping[str, Any], schema_path: str | Path, *, label: str) -> None:
    schema = _load_json(schema_path, label=f"{label} schema")
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(payload)
    except jsonschema.exceptions.SchemaError as exc:
        raise PartCRev1InputSchemaError(
            f"Invalid {label} schema: {exc.message}", path=schema_path
        ) from exc
    except jsonschema.exceptions.ValidationError as exc:
        location = "/".join(str(item) for item in exc.absolute_path) or "<root>"
        raise PartCRev1InputSchemaError(
            f"{label} schema validation failed at {location}: {exc.message}",
            path=schema_path,
            details={"instance_path": location},
        ) from exc


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_reparse_point(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError as exc:
        raise PartCRev1InputPathError(f"Cannot inspect path component: {exc}", path=path) from exc
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _assert_no_reparse_points(path: Path, *, allow_missing_tail: bool) -> None:
    absolute = Path(os.path.abspath(path))
    anchor = Path(absolute.anchor)
    current = anchor
    for part in absolute.parts[1:]:
        current = current / part
        if not current.exists():
            if allow_missing_tail:
                return
            raise PartCRev1InputPathError("Required path component is missing", path=current)
        if _is_reparse_point(current):
            raise PartCRev1InputPathError(
                "Symlink, junction, or reparse-point ambiguity is prohibited", path=current
            )


def _absolute_path(value: Any, *, label: str, must_exist: bool) -> Path:
    text = str(value)
    raw = Path(text)
    if not raw.is_absolute() and not _WINDOWS_DRIVE.match(text):
        raise PartCRev1InputPathError(f"{label} must be absolute", path=text)
    normalized = Path(os.path.abspath(raw))
    _assert_no_reparse_points(normalized, allow_missing_tail=not must_exist)
    if must_exist and not normalized.exists():
        raise PartCRev1InputPathError(f"{label} does not exist", path=normalized)
    return normalized.resolve(strict=must_exist)


@dataclass(frozen=True, slots=True)
class PartCRev1Phase1DataRoots:
    registry_path: Path
    payload: Mapping[str, Any]
    original_repository_root: Path
    phase1_worktree_root: Path
    phase1_output_root: Path
    phase1_temp_root: Path
    root_aliases: Mapping[str, Path]
    declared_input_paths: Mapping[str, Path]


def partC_rev1_load_phase1_data_roots(
    registry_path: str | Path, schema_path: str | Path
) -> PartCRev1Phase1DataRoots:
    payload = _load_json(registry_path, label="Phase 1 data-root registry")
    if payload.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise PartCRev1InputSchemaError(
            f"Unsupported data-root registry schema version: {payload.get('schema_version')!r}",
            path=registry_path,
        )
    _validate_json_schema(payload, schema_path, label="Phase 1 data-root registry")

    original = _absolute_path(payload["original_repository_root"], label="original repository root", must_exist=True)
    worktree = _absolute_path(payload["phase1_worktree_root"], label="Phase 1 worktree root", must_exist=True)
    output = _absolute_path(payload["phase1_output_root"], label="Phase 1 output root", must_exist=False)
    temp = _absolute_path(payload["phase1_temp_root"], label="Phase 1 temporary root", must_exist=False)
    if original == worktree or _is_relative_to(original, worktree) or _is_relative_to(worktree, original):
        raise PartCRev1InputPathError("Original repository and Phase 1 worktree overlap")
    protected = (original, worktree)
    for label, candidate in (("output", output), ("temporary", temp)):
        if any(_is_relative_to(candidate, root) or _is_relative_to(root, candidate) for root in protected):
            raise PartCRev1InputPathError(f"Phase 1 {label} root overlaps a Git checkout", path=candidate)
    if _is_relative_to(output, temp) or _is_relative_to(temp, output):
        raise PartCRev1InputPathError("Phase 1 output and temporary roots overlap")

    authoritative = payload["authoritative_inputs"]
    graph = authoritative["corrected_graph"]
    od = authoritative["od"]
    graph_root = _absolute_path(graph["root"], label="corrected graph root", must_exist=True)
    od_root = _absolute_path(od["root"], label="OD root", must_exist=True)
    root_aliases = {
        "original_repository_root": original,
        "phase1_worktree_root": worktree,
        "phase1_output_root": output,
        "phase1_temp_root": temp,
        "corrected_graph_root": graph_root,
        "od_root": od_root,
    }
    declared = {
        "graph_npz": _absolute_path(graph["graph_npz"], label="graph NPZ", must_exist=True),
        "graph_nodes_parquet": _absolute_path(graph["nodes_parquet"], label="graph nodes Parquet", must_exist=True),
        "graph_arcs_parquet": _absolute_path(graph["arcs_parquet"], label="graph arcs Parquet", must_exist=True),
        "graph_metadata": _absolute_path(graph["metadata_json"], label="graph metadata", must_exist=True),
        "od_npz": _absolute_path(od["npz"], label="OD NPZ", must_exist=True),
        "od_metadata": _absolute_path(od["metadata_json"], label="OD metadata", must_exist=True),
        "rich_sqlite": _absolute_path(authoritative["rich_sqlite"]["path"], label="rich SQLite", must_exist=True),
        "parent_gpkg": _absolute_path(authoritative["parent_geometry"]["path"], label="parent GeoPackage", must_exist=True),
        "corrected_terminal_registry": _absolute_path(
            authoritative["corrected_terminal_registry"]["path"],
            label="corrected terminal registry",
            must_exist=True,
        ),
    }
    for role, path in declared.items():
        if not _is_relative_to(path, original):
            raise PartCRev1InputPathError(
                "Authoritative input is outside the approved original repository root",
                role=role,
                path=path,
            )
    return PartCRev1Phase1DataRoots(
        registry_path=Path(registry_path).resolve(),
        payload=payload,
        original_repository_root=original,
        phase1_worktree_root=worktree,
        phase1_output_root=output,
        phase1_temp_root=temp,
        root_aliases=root_aliases,
        declared_input_paths=declared,
    )


def _reject_nonfinite(value: Any, *, location: str = "<root>") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PartCRev1InputSchemaError(f"Non-finite JSON number at {location}")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nonfinite(nested, location=f"{location}/{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_nonfinite(nested, location=f"{location}/{index}")


def partC_rev1_manifest_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = copy.deepcopy(dict(payload))
    canonical.pop("manifest_fingerprint", None)
    _reject_nonfinite(canonical)
    try:
        encoded = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PartCRev1InputSchemaError(f"Manifest is not canonical-JSON serializable: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PartCRev1AuthoritativeInputManifest:
    manifest_path: Path
    payload: Mapping[str, Any]
    inputs: tuple[Mapping[str, Any], ...]
    manifest_fingerprint: str

    def resolve_inputs(self, roots: PartCRev1Phase1DataRoots) -> dict[str, Path]:
        identities = self.payload["identities"]
        registry_identity_checks = {
            "corrected_graph": roots.payload["authoritative_inputs"]["corrected_graph"]["identity_sha256"],
            "od": roots.payload["authoritative_inputs"]["od"]["identity_sha256"],
            "parent_partition": roots.payload["authoritative_inputs"]["parent_geometry"]["identity_sha256"],
        }
        for key, actual in registry_identity_checks.items():
            if actual != identities[key]:
                raise PartCRev1InputIdentityError(
                    "Machine-local registry identity does not match the tracked manifest",
                    role=key,
                    details={"manifest": identities[key], "registry": actual},
                )
        terminal_item = next(item for item in self.inputs if item["role"] == "corrected_terminal_registry")
        registry_terminal_sha = roots.payload["authoritative_inputs"]["corrected_terminal_registry"]["sha256"]
        if registry_terminal_sha != terminal_item["signature"]["value"]:
            raise PartCRev1InputIdentityError(
                "Machine-local terminal-registry SHA-256 does not match the tracked manifest",
                role="corrected_terminal_registry",
                details={"manifest": terminal_item["signature"]["value"], "registry": registry_terminal_sha},
            )
        resolved: dict[str, Path] = {}
        for item in self.inputs:
            role = str(item["role"])
            root_key = str(item["root_key"])
            if root_key not in roots.root_aliases:
                raise PartCRev1InputSchemaError(f"Unknown manifest root_key {root_key!r}", role=role)
            relative = PurePosixPath(str(item["relative_path"]))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise PartCRev1InputPathError("Manifest relative_path is unsafe", role=role, path=str(relative))
            candidate = roots.root_aliases[root_key].joinpath(*relative.parts)
            candidate = _absolute_path(candidate, label=f"{role} manifest path", must_exist=True)
            declared = roots.declared_input_paths.get(role)
            if declared is None or candidate != declared:
                raise PartCRev1InputIdentityError(
                    "Manifest path does not match the machine-local registry",
                    role=role,
                    path=candidate,
                    details={"registry_path": None if declared is None else str(declared)},
                )
            resolved[role] = candidate
        return resolved


def partC_rev1_load_authoritative_input_manifest(
    manifest_path: str | Path, schema_path: str | Path
) -> PartCRev1AuthoritativeInputManifest:
    payload = _load_json(manifest_path, label="authoritative-input manifest")
    if payload.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise PartCRev1InputSchemaError(
            f"Unsupported authoritative-input manifest schema version: {payload.get('schema_version')!r}",
            path=manifest_path,
        )
    _validate_json_schema(payload, schema_path, label="authoritative-input manifest")
    inputs = payload.get("inputs")
    if not isinstance(inputs, list):
        raise PartCRev1InputSchemaError("Manifest inputs must be a list", path=manifest_path)
    roles = [str(item.get("role")) for item in inputs if isinstance(item, Mapping)]
    if len(inputs) != 9 or len(set(roles)) != 9 or set(roles) != set(EXPECTED_INPUT_ROLES):
        raise PartCRev1InputSchemaError(
            "Manifest must contain exactly the nine unique authoritative roles",
            path=manifest_path,
            details={"roles": roles},
        )
    identities = payload["identities"]
    for item in inputs:
        for key, value in item["identity_references"].items():
            if key in identities and value != identities[key]:
                raise PartCRev1InputIdentityError(
                    "Input identity reference does not match the manifest identity chain",
                    role=item["role"],
                    details={"identity": key, "manifest": identities[key], "input_reference": value},
                )
    terminal_item = next(item for item in inputs if item["role"] == "corrected_terminal_registry")
    terminal_identity = terminal_item["signature"]["value"]
    if terminal_item["signature"]["algorithm"] != "SHA256" or len(terminal_identity) != 64:
        raise PartCRev1InputIdentityError("Corrected terminal registry must use its exact complete SHA-256")
    fingerprint = partC_rev1_manifest_fingerprint(payload)
    recorded = payload.get("manifest_fingerprint", {})
    if recorded.get("algorithm") != "sha256-canonical-json-v1" or recorded.get("value") != fingerprint:
        raise PartCRev1InputIdentityError(
            "Manifest fingerprint mismatch",
            path=manifest_path,
            details={"expected": fingerprint, "recorded": recorded},
        )
    return PartCRev1AuthoritativeInputManifest(
        manifest_path=Path(manifest_path).resolve(),
        payload=payload,
        inputs=tuple(inputs),
        manifest_fingerprint=fingerprint,
    )


def partC_rev1_assert_phase1_destination(
    path: str | Path,
    roots: PartCRev1Phase1DataRoots,
    destination_kind: str,
) -> Path:
    if destination_kind not in {"output", "temp"}:
        raise PartCRev1InputPathError(f"Unsupported destination kind: {destination_kind!r}")
    raw = Path(path)
    if not raw.is_absolute() and not _WINDOWS_DRIVE.match(str(path)):
        raise PartCRev1InputPathError("Runtime destination must be absolute", path=path)
    destination = _absolute_path(raw, label=f"Phase 1 {destination_kind} destination", must_exist=False)
    approved_root = roots.phase1_output_root if destination_kind == "output" else roots.phase1_temp_root
    if not _is_relative_to(destination, approved_root):
        raise PartCRev1InputPathError(
            f"Destination is outside the approved Phase 1 {destination_kind} root",
            path=destination,
            details={"approved_root": str(approved_root)},
        )
    protected = [roots.original_repository_root, roots.phase1_worktree_root, *roots.declared_input_paths.values()]
    if any(_is_relative_to(destination, item) or _is_relative_to(item, destination) for item in protected):
        raise PartCRev1InputPathError("Runtime destination overlaps a protected input or checkout", path=destination)
    return destination


def partC_rev1_open_sqlite_strict_readonly(path: str | Path) -> sqlite3.Connection:
    resolved = _absolute_path(path, label="SQLite/GeoPackage input", must_exist=True)
    if not resolved.is_file():
        raise PartCRev1InputPathError("SQLite/GeoPackage input is not a regular file", path=resolved)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        value = connection.execute("PRAGMA query_only").fetchone()
        if value is None or int(value[0]) != 1:
            raise PartCRev1ReadOnlyError("PRAGMA query_only did not return 1", path=resolved)
        try:
            connection.execute("CREATE TABLE main.partC_rev1_step1_1_write_probe(x INTEGER)")
        except sqlite3.DatabaseError as exc:
            message = str(exc).lower()
            if not any(marker in message for marker in ("readonly", "read-only", "query only", "write")):
                raise PartCRev1ReadOnlyError(
                    "Bounded write probe failed for a reason other than read-only/query-only enforcement",
                    path=resolved,
                    details={"sqlite_error": str(exc)},
                ) from exc
        else:
            raise PartCRev1ReadOnlyError("Bounded write probe unexpectedly succeeded", path=resolved)
        return connection
    except BaseException:
        if connection is not None:
            connection.close()
        raise


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1 authoritative-input contract utilities")
    parser.add_argument("--fingerprint", metavar="MANIFEST_JSON")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if not args.fingerprint:
        parser.error("--fingerprint is required")
    payload = _load_json(args.fingerprint, label="authoritative-input manifest")
    print(partC_rev1_manifest_fingerprint(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
