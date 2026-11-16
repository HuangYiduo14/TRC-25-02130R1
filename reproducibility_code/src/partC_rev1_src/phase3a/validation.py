"""Consolidated Phase 3A schema, closure, and invariance validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jsonschema
import pandas as pd

from partC_rev1_src.partC_rev1_hashing import partC_rev1_file_sha256


class Phase3AValidationError(ValueError):
    pass


def validate_json_schema(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    validator = jsonschema.Draft202012Validator(dict(schema))
    validator.check_schema(dict(schema))
    errors = sorted(validator.iter_errors(dict(payload)), key=lambda value: list(value.absolute_path))
    if errors:
        first = errors[0]
        location = ".".join(map(str, first.absolute_path)) or "$"
        raise Phase3AValidationError(f"JSON schema failed at {location}: {first.message}")


def validate_table_contract(
    frame: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    primary_key: Sequence[str],
    nonnullable: Sequence[str] = (),
    sort_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    missing = set(required_columns).difference(frame.columns)
    if missing:
        raise Phase3AValidationError(f"Table columns missing: {sorted(missing)}")
    if frame.duplicated(list(primary_key)).any():
        raise Phase3AValidationError(f"Table primary key is not unique: {list(primary_key)}")
    for column in nonnullable:
        if frame[column].isna().any():
            raise Phase3AValidationError(f"Table nonnullable column contains nulls: {column}")
    sorted_ok = True
    if sort_columns:
        expected = frame.sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
        sorted_ok = frame.reset_index(drop=True).equals(expected)
        if not sorted_ok:
            raise Phase3AValidationError(f"Table is not in deterministic order: {list(sort_columns)}")
    return {"row_count": len(frame), "primary_key_unique": True, "nonnullable_valid": True, "sorted": sorted_ok}


def snapshot_file_bindings(paths: Mapping[str, str | Path]) -> dict[str, dict[str, Any]]:
    return {
        str(role): {
            "path": str(Path(path).resolve(strict=True)),
            "size_bytes": int(Path(path).resolve(strict=True).stat().st_size),
            "sha256": partC_rev1_file_sha256(Path(path).resolve(strict=True)),
        }
        for role, path in sorted(paths.items())
    }


def assert_file_bindings_unchanged(before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]]) -> None:
    if dict(before) != dict(after):
        raise Phase3AValidationError("Accepted Phase 0-2 file bindings changed")
