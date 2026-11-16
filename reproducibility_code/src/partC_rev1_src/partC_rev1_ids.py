"""Stable identifier normalization helpers for partC_rev1."""

from __future__ import annotations

from typing import Any

import pandas as pd


def partC_rev1_stable_id(value: Any) -> str:
    """Normalize SQLite/pandas numeric-looking IDs without changing their stable value."""

    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return text
    if numeric.is_integer() and text.replace(".", "", 1).replace("-", "", 1).isdigit():
        return str(int(numeric))
    return text
