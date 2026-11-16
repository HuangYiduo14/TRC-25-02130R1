"""SCTG commodity normalization helpers for partC_rev1."""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


def partC_rev1_normalize_sctg2(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return None

    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return int(float(text))

    match = re.search(r"(\d+)", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def partC_rev1_sctg2_to_sctgG5(value: Any) -> str | None:
    sctg2 = partC_rev1_normalize_sctg2(value)
    if sctg2 is None:
        return None
    if 1 <= sctg2 <= 9:
        return "sctg0109"
    if 10 <= sctg2 <= 14:
        return "sctg1014"
    if 15 <= sctg2 <= 19:
        return "sctg1519"
    if 20 <= sctg2 <= 33:
        return "sctg2033"
    if 34 <= sctg2 <= 99:
        return "sctg3499"
    return None


def partC_rev1_add_sctgG5_column(
    df: pd.DataFrame,
    sctg2_col: str,
    output_col: str = "sctgG5",
) -> pd.DataFrame:
    out = df.copy()
    if sctg2_col not in out.columns:
        raise ValueError(f"Missing SCTG2 column: {sctg2_col}")
    out[output_col] = out[sctg2_col].map(partC_rev1_sctg2_to_sctgG5)
    return out
