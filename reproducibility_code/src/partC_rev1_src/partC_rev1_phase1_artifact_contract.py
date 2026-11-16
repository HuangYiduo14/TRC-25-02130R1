"""Versioned identity envelopes for future Part C scientific artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema
import numpy as np


class PartCRev1ArtifactContractError(ValueError):
    """Base error for malformed or mismatched artifact identities."""


class PartCRev1ArtifactSchemaError(PartCRev1ArtifactContractError):
    """Raised when an artifact identity fails its versioned JSON schema."""


class PartCRev1ArtifactIdentityError(PartCRev1ArtifactContractError):
    """Raised when an identity, lineage value, or vector order differs."""


@dataclass(frozen=True, slots=True)
class PartCRev1ArtifactIdentity:
    schema_id: str
    schema_version: str
    artifact_type: str
    artifact_id: str
    identity_sha256: str
    producer: Mapping[str, Any]
    upstream_identities: Mapping[str, Any]
    stable_keys: tuple[str, ...]
    vector_order: Mapping[str, Any]
    units: Mapping[str, Any]
    crs: str | None
    artifact: Mapping[str, Any]
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.payload, allow_nan=False))


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PartCRev1ArtifactSchemaError(f"Duplicate JSON key is prohibited: {key}")
        result[key] = value
    return result


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PartCRev1ArtifactSchemaError(f"Cannot read strict UTF-8 JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PartCRev1ArtifactSchemaError(f"Expected a JSON object in {path}")
    return value


def _reject_nonfinite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PartCRev1ArtifactSchemaError(f"Nonfinite number is prohibited at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{path}[{index}]")


def partC_rev1_artifact_identity_hash(payload: Mapping[str, Any]) -> str:
    """Hash canonical JSON after removing only the self-referential hash."""

    canonical = dict(payload)
    canonical.pop("identity_sha256", None)
    _reject_nonfinite(canonical)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def partC_rev1_vector_order_hash(stable_ids: Sequence[Any] | np.ndarray) -> str:
    array = np.asarray(stable_ids)
    if array.ndim != 1 or array.size == 0:
        raise PartCRev1ArtifactIdentityError("Stable-ID vector order must be one-dimensional and nonempty")
    values = [str(value) for value in array.tolist()]
    if any(not value.strip() for value in values):
        raise PartCRev1ArtifactIdentityError("Stable-ID vector order contains a blank value")
    if len(set(values)) != len(values):
        raise PartCRev1ArtifactIdentityError("Stable-ID vector order contains a duplicate value")
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def partC_rev1_require_vector_order(stable_ids: Sequence[Any] | np.ndarray, declared_hash: str) -> str:
    """Require the declared canonical ordered-stable-ID hash."""

    observed = partC_rev1_vector_order_hash(stable_ids)
    declared = str(declared_hash)
    if declared.startswith("sha256-canonical-json-v1:"):
        declared = declared.split(":", 1)[1]
    if len(declared) != 64 or any(character not in "0123456789abcdef" for character in declared.lower()):
        raise PartCRev1ArtifactIdentityError("Declared vector-order hash must be a 64-character SHA-256")
    if observed != declared.lower():
        raise PartCRev1ArtifactIdentityError(
            f"Stable-ID vector order mismatch: expected {declared.lower()}, observed {observed}"
        )
    return observed


def partC_rev1_load_artifact_identity(
    path: str | Path, schema_path: str | Path
) -> PartCRev1ArtifactIdentity:
    identity_path = Path(path).resolve(strict=True)
    resolved_schema = Path(schema_path).resolve(strict=True)
    payload = _load_object(identity_path)
    schema = _load_object(resolved_schema)
    try:
        validator = jsonschema.Draft202012Validator(schema)
        validator.check_schema(schema)
        errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.absolute_path))
    except jsonschema.SchemaError as exc:
        raise PartCRev1ArtifactSchemaError(f"Invalid artifact schema {resolved_schema}: {exc.message}") from exc
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.absolute_path) or "$"
        raise PartCRev1ArtifactSchemaError(f"Artifact schema validation failed at {location}: {first.message}")
    if payload.get("schema_version") != "1.0":
        raise PartCRev1ArtifactSchemaError("Only artifact schema_version 1.0 is supported")
    expected_hash = str(payload.get("identity_sha256", "")).lower()
    observed_hash = partC_rev1_artifact_identity_hash(payload)
    if expected_hash != observed_hash:
        raise PartCRev1ArtifactIdentityError(
            f"Artifact identity hash mismatch: expected {expected_hash}, observed {observed_hash}"
        )
    vector_order = payload["vector_order"]
    if vector_order.get("algorithm") != "sha256-canonical-json-v1":
        raise PartCRev1ArtifactSchemaError("Unsupported vector-order algorithm")
    return PartCRev1ArtifactIdentity(
        schema_id=str(payload["schema_id"]),
        schema_version=str(payload["schema_version"]),
        artifact_type=str(payload["artifact_type"]),
        artifact_id=str(payload["artifact_id"]),
        identity_sha256=expected_hash,
        producer=dict(payload["producer"]),
        upstream_identities=dict(payload["upstream_identities"]),
        stable_keys=tuple(str(value) for value in payload["stable_keys"]),
        vector_order=dict(vector_order),
        units=dict(payload["units"]),
        crs=None if payload.get("crs") is None else str(payload["crs"]),
        artifact=dict(payload["artifact"]),
        payload=payload,
    )


def _compare_subset(actual: Any, expected: Any, location: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise PartCRev1ArtifactIdentityError(f"Expected mapping at {location}")
        for key, value in expected.items():
            if key not in actual:
                raise PartCRev1ArtifactIdentityError(f"Missing expected identity field {location}.{key}")
            _compare_subset(actual[key], value, f"{location}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if list(actual) != list(expected):
            raise PartCRev1ArtifactIdentityError(f"Identity sequence mismatch at {location}")
        return
    if actual != expected:
        raise PartCRev1ArtifactIdentityError(
            f"Identity mismatch at {location}: expected {expected!r}, observed {actual!r}"
        )


def partC_rev1_validate_artifact_identity(
    identity: PartCRev1ArtifactIdentity, expected: Mapping[str, Any] | PartCRev1ArtifactIdentity
) -> PartCRev1ArtifactIdentity:
    """Validate exact or subset identity expectations before output creation."""

    if isinstance(expected, PartCRev1ArtifactIdentity):
        expected_value: Mapping[str, Any] = {
            "schema_id": expected.schema_id,
            "schema_version": expected.schema_version,
            "artifact_type": expected.artifact_type,
            "artifact_id": expected.artifact_id,
            "identity_sha256": expected.identity_sha256,
        }
    elif isinstance(expected, Mapping) and expected:
        expected_value = expected
    else:
        raise PartCRev1ArtifactIdentityError("Expected artifact identity constraints must be nonempty")
    _compare_subset(identity.payload, expected_value, "$")
    return identity
