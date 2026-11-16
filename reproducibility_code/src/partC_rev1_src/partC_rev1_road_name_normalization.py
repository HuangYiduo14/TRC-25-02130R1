"""Road-name and route-key normalization for Prompt 2.5 matching."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

try:  # pragma: no cover - exercised when dependency is unavailable.
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None


_PARTC_REV1_TERM_MAP = {
    "FWY": "FREEWAY",
    "FRWY": "FREEWAY",
    "FREEWY": "FREEWAY",
    "HWY": "HIGHWAY",
    "HIGHWY": "HIGHWAY",
    "EXPY": "EXPRESSWAY",
    "EXWY": "EXPRESSWAY",
    "PKWY": "PARKWAY",
    "BLVD": "BOULEVARD",
    "ST": "STREET",
    "RD": "ROAD",
    "AVE": "AVENUE",
    "DR": "DRIVE",
    "LN": "LANE",
    "CT": "COURT",
    "CIR": "CIRCLE",
    "PL": "PLACE",
    "TER": "TERRACE",
}

_PARTC_REV1_ROUTE_TYPE_MAP = {
    "I": "I",
    "INTERSTATE": "I",
    "U": "U",
    "US": "U",
    "U S": "U",
    "U.S": "U",
    "U.S.": "U",
    "US ROUTE": "U",
    "UNITED STATES": "U",
    "S": "S",
    "SR": "S",
    "STATE": "S",
    "STATE ROUTE": "S",
    "C": "C",
    "CR": "C",
    "COUNTY": "C",
    "COUNTY ROUTE": "C",
    "O": "O",
    "BUSINESS": "O",
    "OFF INTERSTATE BUSINESS": "O",
    "OFF-INTERSTATE BUSINESS": "O",
}


def _partC_rev1_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except TypeError:
        pass
    text = str(value).strip()
    return text == "" or text.upper() in {"NAN", "NONE", "NULL", "<NA>"}


def _partC_rev1_clean_text(value: Any) -> str:
    if _partC_rev1_missing(value):
        return ""
    text = str(value).upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def partC_rev1_normalize_road_name(value: str | None) -> str:
    """Normalize road names for fuzzy matching inside a spatial candidate set."""

    text = _partC_rev1_clean_text(value)
    if not text:
        return ""
    words = [_PARTC_REV1_TERM_MAP.get(word, word) for word in text.split()]
    return " ".join(words)


def partC_rev1_normalize_route_type(value: str | None) -> str | None:
    text = _partC_rev1_clean_text(value)
    if not text:
        return None
    return _PARTC_REV1_ROUTE_TYPE_MAP.get(text, text[:1] if text[:1] in {"I", "U", "S", "C", "O"} else None)


def partC_rev1_normalize_route_number(value: str | int | float | None) -> str | None:
    if _partC_rev1_missing(value):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = _partC_rev1_clean_text(value)
    if not text:
        return None
    match = re.search(r"\d+[A-Z]?", text)
    if not match:
        return None
    number = match.group(0).lstrip("0")
    return number or "0"


def _partC_rev1_parse_sign_route(value: Any) -> tuple[str | None, str | None]:
    text = _partC_rev1_clean_text(value)
    if not text:
        return None, None
    compact = text.replace(" ", "")
    match = re.fullmatch(r"(INTERSTATE|US|U|SR|S|CR|C|I)(\d+[A-Z]?)", compact)
    if match:
        return partC_rev1_normalize_route_type(match.group(1)), partC_rev1_normalize_route_number(match.group(2))
    match = re.match(r"(INTERSTATE|US ROUTE|US|U S|U|SR|STATE ROUTE|STATE|S|CR|COUNTY|C|I)\s+(\d+[A-Z]?)", text)
    if match:
        return partC_rev1_normalize_route_type(match.group(1)), partC_rev1_normalize_route_number(match.group(2))
    return None, None


def partC_rev1_normalize_route_key(
    route_type: str | None,
    route_number: str | int | float | None,
    sign_route: str | None = None,
    sign1: str | None = None,
) -> str | None:
    """Return a compact route key such as ``I 95``, ``U 101``, or ``S 83``."""

    for sign_value in (sign_route, sign1):
        sign_type, sign_number = _partC_rev1_parse_sign_route(sign_value)
        if sign_type and sign_number:
            return f"{sign_type} {sign_number}"
    normalized_type = partC_rev1_normalize_route_type(route_type)
    normalized_number = partC_rev1_normalize_route_number(route_number)
    if normalized_type and normalized_number:
        return f"{normalized_type} {normalized_number}"
    return None


def _partC_rev1_get(row: Mapping[str, Any] | Any, names: list[str]) -> Any:
    if isinstance(row, Mapping):
        lower = {str(key).lower(): key for key in row.keys()}
        for name in names:
            key = lower.get(name.lower())
            if key is not None:
                return row[key]
        return None
    if hasattr(row, "keys") and hasattr(row, "__getitem__"):
        lower = {str(key).lower(): key for key in row.keys()}
        for name in names:
            key = lower.get(name.lower())
            if key is not None:
                return row[key]
        return None
    for name in names:
        if hasattr(row, name):
            return getattr(row, name)
    return None


def partC_rev1_normalize_sign_key(
    route_type: str | None,
    route_number: str | int | float | None,
) -> str | None:
    """Return the canonical signed-route key from explicit type/number columns."""

    normalized_type = partC_rev1_normalize_route_type(route_type)
    normalized_number = partC_rev1_normalize_route_number(route_number)
    if normalized_type and normalized_number:
        return f"{normalized_type} {normalized_number}"
    return None


def partC_rev1_extract_faf_normalized_sign_key(row: Mapping[str, Any] | Any) -> str | None:
    return partC_rev1_normalize_sign_key(
        _partC_rev1_get(row, ["Rte_Type", "route_type", "partC_rev1_route_type"]),
        _partC_rev1_get(row, ["Rte_Number", "route_number", "partC_rev1_route_number"]),
    )


def partC_rev1_extract_ntad_normalized_sign_key(row: Mapping[str, Any] | Any) -> str | None:
    return partC_rev1_normalize_sign_key(
        _partC_rev1_get(row, ["SIGNT1", "signt1", "partC_rev1_ntad_signt1"]),
        _partC_rev1_get(row, ["SIGNN1", "signn1", "partC_rev1_ntad_signn1"]),
    )


def partC_rev1_extract_faf_route_key(row: Mapping[str, Any] | Any) -> str | None:
    return partC_rev1_normalize_route_key(
        _partC_rev1_get(row, ["Rte_Type", "route_type", "partC_rev1_route_type"]),
        _partC_rev1_get(row, ["Rte_Number", "route_number", "partC_rev1_route_number"]),
        sign_route=_partC_rev1_get(row, ["Sign_Rte", "sign_route", "partC_rev1_sign_route"]),
    )


def partC_rev1_extract_ntad_route_key(row: Mapping[str, Any] | Any) -> str | None:
    return partC_rev1_normalize_route_key(
        _partC_rev1_get(row, ["SIGNT1", "signt1"]),
        _partC_rev1_get(row, ["SIGNN1", "signn1"]),
        sign1=_partC_rev1_get(row, ["SIGN1", "sign1"]),
    )


def partC_rev1_name_similarity(left: str | None, right: str | None) -> float:
    left_norm = partC_rev1_normalize_road_name(left)
    right_norm = partC_rev1_normalize_road_name(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if fuzz is not None:
        return float(fuzz.token_set_ratio(left_norm, right_norm)) / 100.0
    left_words = set(left_norm.split())
    right_words = set(right_norm.split())
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)
