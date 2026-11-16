"""Stable hashing helpers for partC_rev1 Prompt 4 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def partC_rev1_canonical_scalar(value: Any) -> str:
    """Return a deterministic text form for stable IDs and hash payloads."""

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return str(numeric)
        if numeric.is_integer():
            return str(int(numeric))
        return format(numeric, ".17g")
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            numeric = float(text)
        except ValueError:
            return text
        if numeric.is_integer():
            return str(int(numeric))
    return text


def partC_rev1_sort_key(value: Any) -> tuple[int, int | str]:
    text = partC_rev1_canonical_scalar(value)
    if text and text.lstrip("-").isdigit():
        return (0, int(text))
    return (1, text)


def partC_rev1_sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def partC_rev1_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def partC_rev1_file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def partC_rev1_file_signature(
    path: str | Path,
    small_limit: int = 64 * 1024 * 1024,
    block_size: int = 8 * 1024 * 1024,
) -> str:
    """Reproduce the reviewed Phase 0 ``FAST-SHA256-v1`` file signature.

    Files at or below 64 MiB receive a complete SHA-256.  Larger files receive
    the audited size-domain-prefixed digest over the start, middle, and end
    8 MiB blocks in that exact order.  The latter is a reviewed bounded
    signature, not a full cryptographic hash of every byte in the file.
    """

    resolved = Path(path)
    size = resolved.stat().st_size
    if size <= small_limit:
        return "SHA256:" + partC_rev1_file_sha256(resolved)
    digest = hashlib.sha256()
    digest.update(f"FAST-SHA256-v1\0{size}\0".encode())
    with resolved.open("rb") as handle:
        for offset in (
            0,
            max(0, size // 2 - block_size // 2),
            max(0, size - block_size),
        ):
            handle.seek(offset)
            digest.update(handle.read(block_size))
    return "FAST-SHA256-v1:" + digest.hexdigest()


def partC_rev1_json_hash(obj: Mapping[str, Any] | Sequence[Any]) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=partC_rev1_canonical_scalar)
    return partC_rev1_sha256_text(text)


def partC_rev1_ordered_records_hash(records: Iterable[Sequence[Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update("\x1f".join(partC_rev1_canonical_scalar(value) for value in record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def partC_rev1_numpy_array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def partC_rev1_dataframe_records_hash(df: pd.DataFrame, columns: Sequence[str]) -> str:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot hash DataFrame; missing columns: {missing}")
    return partC_rev1_ordered_records_hash(df.loc[:, list(columns)].itertuples(index=False, name=None))
