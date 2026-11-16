"""Disabled Phase 2 interfaces and fail-before-write authorization guard."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import jsonschema


class PartCRev1Phase2AuthorizationError(PermissionError):
    """Raised before any Phase 2 output path can be created."""


class PartCRev1Phase2InterfaceError(ValueError):
    """Raised when a synthetic future-interface fixture is malformed."""


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PartCRev1Phase2InterfaceError(f"Duplicate JSON key is prohibited: {key}")
        result[key] = value
    return result


def _load(value: str | Path | Mapping[str, Any], label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        path = Path(value).resolve(strict=True)
        try:
            payload = json.loads(
                path.read_text(encoding="utf-8", errors="strict"),
                object_pairs_hook=_reject_duplicates,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PartCRev1Phase2InterfaceError(f"Cannot load strict UTF-8 {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise PartCRev1Phase2InterfaceError(f"{label} must be a JSON object")
    return payload


def partC_rev1_validate_phase2_interface(
    payload: str | Path | Mapping[str, Any], schema_path: str | Path
) -> dict[str, Any]:
    """Validate a synthetic future-interface payload without enabling execution."""

    value = _load(payload, "Phase 2 interface")
    schema = _load(schema_path, "Phase 2 schema")
    try:
        validator = jsonschema.Draft202012Validator(schema)
        validator.check_schema(schema)
        errors = sorted(validator.iter_errors(value), key=lambda item: list(item.absolute_path))
    except jsonschema.SchemaError as exc:
        raise PartCRev1Phase2InterfaceError(f"Invalid Phase 2 schema: {exc.message}") from exc
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.absolute_path) or "$"
        raise PartCRev1Phase2InterfaceError(
            f"Phase 2 interface schema failed at {location}: {first.message}"
        )
    schema_id = str(value.get("schema_id"))
    if schema_id == "partC_rev1_phase2_pair_table":
        ordered = value["ordered_pair_ids"]
        rows = [row["pair_id"] for row in value["pairs"]]
        if ordered != rows:
            raise PartCRev1Phase2InterfaceError("Pair-table row order differs from ordered_pair_ids")
    elif schema_id == "partC_rev1_phase2_split_suite":
        resolution_ids = [row["resolution_id"] for row in value["resolutions"]]
        if len(resolution_ids) != len(set(resolution_ids)):
            raise PartCRev1Phase2InterfaceError("Split-suite resolution IDs must be unique")
    elif schema_id == "partC_rev1_phase2_acn":
        node_ids = {row["acn_node_id"] for row in value["nodes"]}
        for arc in value["arcs"]:
            if arc["tail_acn_node_id"] not in node_ids or arc["head_acn_node_id"] not in node_ids:
                raise PartCRev1Phase2InterfaceError("ACN arc endpoint is absent from the stable node set")
    elif schema_id == "partC_rev1_phase2_gpkg_inventory":
        layer_names = [row["layer_name"] for row in value["layers"]]
        if len(layer_names) != len(set(layer_names)):
            raise PartCRev1Phase2InterfaceError("GeoPackage inventory layer names must be unique")
    return value


def partC_rev1_require_phase2_authorization(
    config: str | Path | Mapping[str, Any], destination: str | Path
) -> Path:
    """Return a destination only under a future exact external authorization.

    The current frozen template always raises before inspecting or creating the
    destination's parent.
    """

    value = _load(config, "Phase 2 configuration")
    raw_destination = str(destination)
    if not value.get("phase_2_authorized", False):
        raise PartCRev1Phase2AuthorizationError("Phase 2 is disabled; authorization is false")
    if value.get("review_decision") != "APPROVE" or value.get("next_step_authorized") is not True:
        raise PartCRev1Phase2AuthorizationError("Phase 2 lacks an explicit external review approval")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise PartCRev1Phase2AuthorizationError("Phase 2 task list is empty")
    approval_sha = str(value.get("approval_handoff_sha256", ""))
    if len(approval_sha) != 64 or any(character not in "0123456789abcdef" for character in approval_sha):
        raise PartCRev1Phase2AuthorizationError("Phase 2 approval handoff SHA-256 is absent")
    pure = PurePosixPath(raw_destination)
    if not raw_destination.strip() or pure.is_absolute() or ".." in pure.parts or "\\" in raw_destination:
        raise PartCRev1Phase2AuthorizationError("Phase 2 destination must be a safe approved relative path")
    output_root = value.get("approved_output_root")
    if not isinstance(output_root, str) or not output_root.strip():
        raise PartCRev1Phase2AuthorizationError("Phase 2 approved output root is absent")
    root = Path(output_root).resolve(strict=True)
    candidate = (root / Path(*pure.parts)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PartCRev1Phase2AuthorizationError("Phase 2 destination escapes its approved root") from exc
    return candidate
