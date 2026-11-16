"""Frozen Phase 3A FFTT, route-similarity, and summary metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class Phase3AMetricError(ValueError):
    pass


def cost_errors(reference_s: float, candidate_s: float | None) -> dict[str, float | None]:
    reference = float(reference_s)
    if not np.isfinite(reference) or reference <= 0.0:
        raise Phase3AMetricError("Physical reference FFTT must be finite and positive")
    if candidate_s is None:
        return {
            "signed_error_s": None,
            "absolute_error_s": None,
            "relative_error": None,
            "absolute_percentage_error": None,
        }
    candidate = float(candidate_s)
    if not np.isfinite(candidate) or candidate < 0.0:
        raise Phase3AMetricError("Candidate FFTT must be finite and nonnegative")
    signed = candidate - reference
    return {
        "signed_error_s": signed,
        "absolute_error_s": abs(signed),
        "relative_error": signed / reference,
        "absolute_percentage_error": abs(signed) / reference,
    }


def weighted_arc_similarity(
    reference_arc_ids: Sequence[str],
    candidate_arc_ids: Sequence[str] | None,
    arc_weight_s: Mapping[str, float],
) -> dict[str, float | None]:
    if candidate_arc_ids is None:
        return {"weighted_arc_precision": None, "weighted_arc_recall": None, "weighted_arc_jaccard": None}
    reference = set(map(str, reference_arc_ids))
    candidate = set(map(str, candidate_arc_ids))
    universe = reference | candidate
    missing = universe.difference(map(str, arc_weight_s.keys()))
    if missing:
        raise Phase3AMetricError(f"Arc similarity weights are missing IDs: {sorted(missing)[:5]}")
    weight = {arc: float(arc_weight_s[arc]) for arc in universe}
    if any(not np.isfinite(value) or value < 0.0 for value in weight.values()):
        raise Phase3AMetricError("Arc similarity weights must be finite and nonnegative")
    intersection = reference & candidate
    common = sum(weight[value] for value in intersection)
    reference_total = sum(weight[value] for value in reference)
    candidate_total = sum(weight[value] for value in candidate)
    union_total = sum(weight[value] for value in universe)
    return {
        "weighted_arc_precision": common / candidate_total if candidate_total else (1.0 if not reference else 0.0),
        "weighted_arc_recall": common / reference_total if reference_total else (1.0 if not candidate else 0.0),
        "weighted_arc_jaccard": common / union_total if union_total else 1.0,
    }


def lcs_similarity(left: Sequence[str], right: Sequence[str] | None) -> float | None:
    if right is None:
        return None
    a, b = tuple(map(str, left)), tuple(map(str, right))
    if not a and not b:
        return 1.0
    previous = np.zeros(len(b) + 1, dtype=np.int32)
    for value in a:
        current = np.zeros(len(b) + 1, dtype=np.int32)
        for index, other in enumerate(b, start=1):
            current[index] = previous[index - 1] + 1 if value == other else max(previous[index], current[index - 1])
        previous = current
    return float(2 * int(previous[-1]) / (len(a) + len(b)))


def lifted_regret(reference_s: float, lifted_s: float | None) -> float | None:
    return cost_errors(reference_s, lifted_s)["relative_error"] if lifted_s is not None else None


def distribution_summary(values: Sequence[float], weights: Sequence[float] | None = None) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or np.any(~np.isfinite(array)):
        raise Phase3AMetricError("Distribution summary requires nonempty finite values")
    result: dict[str, Any] = {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
        "negative_count": int((array < 0.0).sum()),
    }
    if weights is not None:
        weight = np.asarray(weights, dtype=np.float64)
        if weight.shape != array.shape or np.any(~np.isfinite(weight)) or np.any(weight < 0.0) or weight.sum() <= 0.0:
            raise Phase3AMetricError("Distribution weights are invalid")
        result["demand_weighted_mean"] = float(np.average(array, weights=weight))
    return result
