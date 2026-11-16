"""Coordinate, FFTT, length, and frozen capacity-unit audits."""

from __future__ import annotations

from typing import Any

import numpy as np


MILES_TO_METERS = 1609.344
MINUTES_TO_SECONDS = 60.0
CAPACITY_SCALE_FACTORS = (0.001, 1.0, 1000.0)


class Phase3AUnitError(ValueError):
    pass


def validate_fftt_length(fftt_s: np.ndarray, length_m: np.ndarray) -> dict[str, Any]:
    fftt = np.asarray(fftt_s, dtype=np.float64)
    length = np.asarray(length_m, dtype=np.float64)
    if fftt.shape != length.shape or fftt.ndim != 1:
        raise Phase3AUnitError("FFTT and length must be equal-length vectors")
    result = {
        "row_count": int(fftt.size),
        "fftt_unit": "seconds",
        "length_unit": "meters",
        "nonfinite_fftt_count": int((~np.isfinite(fftt)).sum()),
        "nonpositive_fftt_count": int((np.isfinite(fftt) & (fftt <= 0.0)).sum()),
        "nonfinite_length_count": int((~np.isfinite(length)).sum()),
        "negative_length_count": int((np.isfinite(length) & (length < 0.0)).sum()),
        "zero_length_count": int((np.isfinite(length) & (length == 0.0)).sum()),
    }
    finite_fftt = fftt[np.isfinite(fftt)]
    finite_length = length[np.isfinite(length)]
    result["fftt_quantiles_s"] = _quantiles(finite_fftt)
    result["length_quantiles_m"] = _quantiles(finite_length)
    return result


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {key: None for key in ("min", "p01", "p50", "p99", "max")}
    q = np.quantile(values, [0.0, 0.01, 0.5, 0.99, 1.0])
    return {key: float(value) for key, value in zip(("min", "p01", "p50", "p99", "max"), q, strict=True)}


def capacity_unit_audit(
    base_capacity: np.ndarray,
    remaining_capacity: np.ndarray,
    *,
    metadata_unit_evidence: str | None,
    metadata_contradicts_correction: bool = False,
) -> dict[str, Any]:
    """Apply the frozen robust factor-1000 diagnostic without changing source values."""

    base = np.asarray(base_capacity, dtype=np.float64)
    remaining = np.asarray(remaining_capacity, dtype=np.float64)
    if base.shape != remaining.shape or base.ndim != 1:
        raise Phase3AUnitError("Base and remaining capacity must be equal-length vectors")
    valid = np.isfinite(base) & np.isfinite(remaining) & (base > 0.0) & (remaining > 0.0)
    base_valid = base[valid]
    remaining_valid = remaining[valid]
    if base_valid.size == 0:
        raise Phase3AUnitError("Capacity audit has no nonzero finite base/remaining pairs")
    log_ratio = np.log10(remaining_valid / base_valid)
    factor_evidence: dict[str, Any] = {}
    scores: dict[float, float] = {}
    for factor in CAPACITY_SCALE_FACTORS:
        scaled_log = log_ratio + np.log10(factor)
        score = float(np.median(np.abs(scaled_log)))
        scores[factor] = score
        factor_evidence[str(factor)] = {
            "median_absolute_log10_ratio": score,
            "median_log10_ratio": float(np.median(scaled_log)),
            "ratio_quantiles": _ratio_quantiles(scaled_log),
        }
    best = min(CAPACITY_SCALE_FACTORS, key=lambda value: (scores[value], CAPACITY_SCALE_FACTORS.index(value)))
    improvement = scores[1.0] - scores[best]
    corrected_median_ratio = float(10.0 ** np.median(log_ratio + np.log10(best)))
    correction_allowed = (
        best != 1.0
        and improvement >= 2.5
        and 0.1 <= corrected_median_ratio <= 10.0
        and not metadata_contradicts_correction
    )
    selected = float(best if correction_allowed else 1.0)
    before = log_ratio
    after = log_ratio + np.log10(selected)
    return {
        "schema_version": "1.0",
        "source_capacity_fields": ["base_capacity", "remaining_capacity"],
        "source_metadata_unit_evidence": metadata_unit_evidence,
        "fallback_unit_assumption": None if metadata_unit_evidence else "ktons/year",
        "base_field": "base_capacity",
        "remaining_field": "remaining_capacity",
        "nonzero_pair_count": int(valid.sum()),
        "candidate_scale_factors": list(CAPACITY_SCALE_FACTORS),
        "factor_evidence": factor_evidence,
        "median_log10_ratio_before": float(np.median(before)),
        "quantiles_before": _ratio_quantiles(before),
        "selected_scale_factor": selected,
        "median_log10_ratio_after": float(np.median(after)),
        "quantiles_after": _ratio_quantiles(after),
        "improvement_orders_of_magnitude": float(improvement if correction_allowed else 0.0),
        "correction_applied": bool(correction_allowed),
        "source_values_modified": False,
        "prepared_scale_factor": selected,
        "global_capacity_multiplier_tuned": False,
        "metadata_contradicts_correction": bool(metadata_contradicts_correction),
    }


def _ratio_quantiles(log_ratio: np.ndarray) -> dict[str, float]:
    values = np.power(10.0, np.quantile(log_ratio, [0.01, 0.1, 0.5, 0.9, 0.99]))
    return {key: float(value) for key, value in zip(("p01", "p10", "p50", "p90", "p99"), values, strict=True)}
