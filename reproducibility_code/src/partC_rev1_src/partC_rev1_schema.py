"""Schema and column coercion helpers for partC_rev1."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import pandas as pd


def _partC_rev1_normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def partC_rev1_find_column(
    df: pd.DataFrame,
    preferred_name: str,
    candidates: Iterable[str] | None = None,
    required: bool = True,
) -> str | None:
    """Find a column by exact, case-insensitive, then normalized candidate matching."""

    names = list(df.columns)
    search_names = [preferred_name, *(list(candidates or []))]

    for candidate in search_names:
        if candidate in df.columns:
            return candidate

    lower_lookup = {str(name).lower(): name for name in names}
    for candidate in search_names:
        match = lower_lookup.get(str(candidate).lower())
        if match is not None:
            return str(match)

    normalized_lookup = {_partC_rev1_normalize_column_name(str(name)): name for name in names}
    for candidate in search_names:
        match = normalized_lookup.get(_partC_rev1_normalize_column_name(str(candidate)))
        if match is not None:
            return str(match)

    if required:
        raise ValueError(f"Missing required column. Tried: {search_names}")
    return None


def partC_rev1_find_first_existing_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = True,
) -> str | None:
    return partC_rev1_find_column(df, "", candidates=candidates, required=required)


def partC_rev1_require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if partC_rev1_find_column(df, column, required=False) is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def partC_rev1_coerce_numeric(series: pd.Series, default: float | int | None = None) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if default is not None:
        numeric = numeric.fillna(default)
    return numeric


def partC_rev1_coerce_int_nullable(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def partC_rev1_coerce_string_id(series: pd.Series) -> pd.Series:
    def convert(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return str(int(value))
        text = str(value).strip()
        if re.fullmatch(r"[+-]?\d+\.0+", text):
            return str(int(float(text)))
        return text

    return series.map(convert).astype("string")


def partC_rev1_parse_county_fips_from_centroid_id(value: Any, allow_left_pad: bool = True) -> str | None:
    """Parse a 5-digit county FIPS from common FAF centroid encodings."""

    if value is None or pd.isna(value):
        return None

    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
        if re.fullmatch(r"[+-]?\d+\.0+", text):
            text = str(int(float(text)))

    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    if len(digits) >= 5:
        return digits[:5]
    if allow_left_pad:
        return digits.zfill(5)
    return None
