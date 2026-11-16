"""Robust table and JSON IO helpers for partC_rev1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def _partC_rev1_path(path: str | Path) -> Path:
    if path is None or str(path).strip() == "":
        raise ValueError("Path is empty")
    return Path(path).expanduser()


def partC_rev1_read_table(
    path: str | Path,
    *,
    dtype_overrides: Mapping[str, Any] | None = None,
    sheet_name: str | int | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read CSV/TXT, Excel, or Parquet into a DataFrame."""

    table_path = _partC_rev1_path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Table not found: {table_path}")

    lower_name = table_path.name.lower()
    lower_suffix = table_path.suffix.lower()
    dtype_overrides = dict(dtype_overrides or {})

    if lower_name.endswith(".csv.gz"):
        return pd.read_csv(table_path, dtype=dtype_overrides or None, nrows=nrows)
    if lower_suffix in {".csv", ".txt"}:
        return pd.read_csv(table_path, dtype=dtype_overrides or None, nrows=nrows)
    if lower_suffix in {".xlsx", ".xls"}:
        kwargs: dict[str, Any] = {"dtype": dtype_overrides or None}
        if sheet_name is not None:
            kwargs["sheet_name"] = sheet_name
        if nrows is not None:
            kwargs["nrows"] = nrows
        return pd.read_excel(table_path, **kwargs)
    if lower_suffix == ".parquet":
        df = pd.read_parquet(table_path)
        if nrows is not None:
            return df.head(nrows)
        return df
    raise ValueError(f"Unsupported table extension for {table_path}")


def partC_rev1_write_table(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> None:
    """Write a DataFrame to CSV, Excel, or Parquet based on extension."""

    table_path = _partC_rev1_path(path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    lower_suffix = table_path.suffix.lower()
    lower_name = table_path.name.lower()

    if lower_name.endswith(".csv.gz") or lower_suffix in {".csv", ".txt"}:
        df.to_csv(table_path, index=index)
        return
    if lower_suffix in {".xlsx", ".xls"}:
        df.to_excel(table_path, index=index)
        return
    if lower_suffix == ".parquet":
        df.to_parquet(table_path, index=index)
        return
    raise ValueError(f"Unsupported table extension for {table_path}")


def partC_rev1_read_json(path: str | Path) -> dict[str, Any]:
    json_path = _partC_rev1_path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON config not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object in {json_path}")
    return obj


def partC_rev1_write_json(obj: Mapping[str, Any], path: str | Path) -> None:
    json_path = _partC_rev1_path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
        handle.write("\n")


def partC_rev1_preserve_string_ids(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a copy with selected ID/FIPS-like columns converted to pandas string dtype."""

    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].astype("string")
    return out
