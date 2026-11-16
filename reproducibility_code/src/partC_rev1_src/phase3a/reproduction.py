"""Deterministic shard merge, table fingerprints, and reproduction evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .schemas import canonical_json_bytes


class Phase3AReproductionError(ValueError):
    pass


def deterministic_frame_hash(frame: pd.DataFrame, *, sort_columns: Sequence[str]) -> str:
    missing = set(sort_columns).difference(frame.columns)
    if missing:
        raise Phase3AReproductionError(f"Hash sort columns missing: {sorted(missing)}")
    ordered = frame.sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"columns": list(ordered.columns), "dtypes": [str(value) for value in ordered.dtypes]}))
    values = pd.util.hash_pandas_object(ordered, index=False, categorize=True).to_numpy(dtype=np.uint64)
    digest.update(values.astype("<u8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def deterministic_shard_merge(
    shards: Mapping[str, pd.DataFrame], *, sort_columns: Sequence[str], primary_key: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not shards:
        raise Phase3AReproductionError("Shard merge requires a nonempty shard map")
    ordered_ids = sorted(map(str, shards))
    merged = pd.concat([shards[value].copy() for value in ordered_ids], ignore_index=True)
    missing = (set(sort_columns) | set(primary_key)) - set(merged.columns)
    if missing:
        raise Phase3AReproductionError(f"Shard merge columns missing: {sorted(missing)}")
    if merged.duplicated(list(primary_key)).any():
        raise Phase3AReproductionError("Shard merge primary key contains duplicates")
    merged = merged.sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
    return merged, {
        "shard_ids": ordered_ids,
        "row_count": len(merged),
        "frame_sha256": deterministic_frame_hash(merged, sort_columns=sort_columns),
        "deterministic_sorted_merge": True,
    }


def compare_reproduction(
    first: Mapping[str, str], second: Mapping[str, str]
) -> dict[str, Any]:
    keys = sorted(set(first) | set(second))
    rows = [
        {"artifact": key, "first_sha256": first.get(key), "second_sha256": second.get(key), "match": first.get(key) == second.get(key)}
        for key in keys
    ]
    return {"status": "PASS" if all(row["match"] for row in rows) else "FAIL", "artifacts": rows}
