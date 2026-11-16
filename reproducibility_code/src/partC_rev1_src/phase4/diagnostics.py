"""Vectorized Phase 4 congestion, active-support, and B-ACN diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from partC_rev1_src.partC_rev1_graph_snapshot import PartCRev1GraphSnapshot

from .models import SuiteLinearTrace, SuiteTraceMap


QUANTILES = np.asarray([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0], dtype=np.float64)


def _quantile_dict(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    result = np.quantile(array, QUANTILES)
    return {f"q{int(round(q * 100)):03d}": float(value) for q, value in zip(QUANTILES, result, strict=True)}


def _weighted_quantile(values: np.ndarray, weights: np.ndarray) -> dict[str, float | None]:
    value = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(value) & np.isfinite(weight) & (weight > 0.0)
    if not np.any(keep):
        return {f"q{int(round(q * 100)):03d}": None for q in QUANTILES}
    value = value[keep]
    weight = weight[keep]
    order = np.argsort(value, kind="mergesort")
    value = value[order]
    weight = weight[order]
    cumulative = np.cumsum(weight, dtype=np.float64)
    points = QUANTILES * cumulative[-1]
    result = np.interp(points, cumulative, value, left=value[0], right=value[-1])
    return {f"q{int(round(q * 100)):03d}": float(item) for q, item in zip(QUANTILES, result, strict=True)}


def capacity_congestion_diagnostics(
    graph: PartCRev1GraphSnapshot,
    flow: np.ndarray,
    cost: np.ndarray,
    used_capacity: np.ndarray,
) -> dict[str, Any]:
    current = np.asarray(flow, dtype=np.float64)
    current_cost = np.asarray(cost, dtype=np.float64)
    capacity = np.asarray(used_capacity, dtype=np.float64)
    if not (current.shape == current_cost.shape == capacity.shape == graph.free_time.shape):
        raise ValueError("Congestion diagnostic arrays are not aligned")
    vc = np.divide(current, capacity, out=np.zeros_like(current), where=capacity > 0.0)
    tti = np.divide(current_cost, graph.free_time, out=np.ones_like(current_cost), where=graph.free_time > 0.0)
    total_arc_flow = float(current.sum(dtype=np.float64))

    def flow_share(mask: np.ndarray) -> float:
        if total_arc_flow <= 0.0:
            return 0.0
        return float(current[mask].sum(dtype=np.float64) / total_arc_flow)

    return {
        "arc_count": int(len(current)),
        "positive_flow_arc_count": int(np.count_nonzero(current > 0.0)),
        "total_arc_flow": total_arc_flow,
        "vc_unweighted_quantiles": _quantile_dict(vc),
        "vc_flow_weighted_quantiles": _weighted_quantile(vc, current),
        "tti_unweighted_quantiles": _quantile_dict(tti),
        "tti_flow_weighted_quantiles": _weighted_quantile(tti, current),
        "arc_flow_share_vc_above": {
            str(threshold): flow_share(vc > threshold) for threshold in (0.8, 1.0, 1.2)
        },
        "arc_flow_share_tti_above": {
            str(threshold): flow_share(tti > threshold) for threshold in (1.05, 1.25, 1.5, 2.0)
        },
        "nonfinite_flow_count": int(np.count_nonzero(~np.isfinite(current))),
        "nonfinite_cost_count": int(np.count_nonzero(~np.isfinite(current_cost))),
        "nonfinite_vc_count": int(np.count_nonzero(~np.isfinite(vc))),
        "nonfinite_tti_count": int(np.count_nonzero(~np.isfinite(tti))),
    }


def _top_concentration(values: np.ndarray) -> dict[str, float]:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[positive > 0.0]
    if len(positive) == 0:
        return {str(value): 0.0 for value in (0.001, 0.01, 0.05, 0.1)}
    ordered = np.sort(positive)[::-1]
    total = float(ordered.sum(dtype=np.float64))
    return {
        str(fraction): float(
            ordered[: max(1, int(np.ceil(fraction * len(ordered))))].sum(dtype=np.float64) / total
        )
        for fraction in (0.001, 0.01, 0.05, 0.1)
    }


def _regions_for_coarse_keys(keys: np.ndarray, trace_map: SuiteTraceMap) -> np.ndarray:
    if len(keys) == 0:
        return np.empty(0, dtype=np.int32)
    entry_terminal = (np.asarray(keys, dtype=np.uint64) >> np.uint64(32)).astype(np.int64)
    if np.any(entry_terminal >= trace_map.terminal_count):
        raise ValueError(f"{trace_map.suite_id} active pair contains an invalid entry terminal")
    return trace_map.terminal_region_local[entry_terminal]


def _self_pair_mask(keys: np.ndarray, trace_map: SuiteTraceMap) -> np.ndarray:
    """Identify same-terminal pairs excluded from Phase 3 conceptual support."""

    pair_keys = np.asarray(keys, dtype=np.uint64)
    if len(pair_keys) == 0:
        return np.empty(0, dtype=bool)
    entry_terminal = (pair_keys >> np.uint64(32)).astype(np.int64)
    exit_terminal = (pair_keys & np.uint64(0xFFFFFFFF)).astype(np.int64)
    if np.any(entry_terminal >= trace_map.terminal_count) or np.any(
        exit_terminal >= trace_map.terminal_count
    ):
        raise ValueError(f"{trace_map.suite_id} active pair contains an invalid terminal")
    return entry_terminal == exit_terminal


def active_support_diagnostics(
    trace_map: SuiteTraceMap,
    current: SuiteLinearTrace,
    latest_aon: SuiteLinearTrace,
    *,
    ever_active_pair_keys: np.ndarray,
    ever_active_bundle_mask: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    current.validate(trace_map)
    latest_aon.validate(trace_map)
    ever_keys = np.asarray(ever_active_pair_keys, dtype=np.uint64)
    if len(ever_keys) > 1 and np.any(ever_keys[1:] <= ever_keys[:-1]):
        raise ValueError("Ever-active regional-pair keys must be strictly sorted")
    current_region = _regions_for_coarse_keys(current.coarse_regional_pair_flow.keys, trace_map)
    latest_region = _regions_for_coarse_keys(latest_aon.coarse_regional_pair_flow.keys, trace_map)
    ever_region = _regions_for_coarse_keys(ever_keys, trace_map)
    current_self = _self_pair_mask(current.coarse_regional_pair_flow.keys, trace_map)
    latest_self = _self_pair_mask(latest_aon.coarse_regional_pair_flow.keys, trace_map)
    ever_self = _self_pair_mask(ever_keys, trace_map)
    current_count = np.bincount(current_region, minlength=trace_map.region_count).astype(np.int64)
    latest_count = np.bincount(latest_region, minlength=trace_map.region_count).astype(np.int64)
    ever_count = np.bincount(ever_region, minlength=trace_map.region_count).astype(np.int64)
    current_self_count = np.bincount(
        current_region[current_self], minlength=trace_map.region_count
    ).astype(np.int64)
    latest_self_count = np.bincount(
        latest_region[latest_self], minlength=trace_map.region_count
    ).astype(np.int64)
    ever_self_count = np.bincount(
        ever_region[ever_self], minlength=trace_map.region_count
    ).astype(np.int64)
    current_conceptual_count = current_count - current_self_count
    latest_conceptual_count = latest_count - latest_self_count
    ever_conceptual_count = ever_count - ever_self_count
    conceptual = trace_map.conceptual_pair_count_by_region.astype(np.int64)
    for state_name, state_count in (
        ("ever-active", ever_conceptual_count),
        ("positive-current", current_conceptual_count),
        ("latest-AON", latest_conceptual_count),
    ):
        if np.any(state_count > conceptual):
            raise ValueError(
                f"{trace_map.suite_id} {state_name} non-self regional pairs exceed "
                "Phase 3 conceptual support"
            )
    region_table = pd.DataFrame(
        {
            "suite_id": trace_map.suite_id,
            "region_id": trace_map.region_ids,
            "conceptual_pair_count": conceptual,
            "ever_active_pair_count": ever_count,
            "ever_active_conceptual_support_pair_count": ever_conceptual_count,
            "ever_active_out_of_conceptual_support_self_pair_count": ever_self_count,
            "positive_current_pair_count": current_count,
            "positive_current_conceptual_support_pair_count": current_conceptual_count,
            "positive_current_out_of_conceptual_support_self_pair_count": current_self_count,
            "latest_aon_active_pair_count": latest_count,
            "latest_aon_conceptual_support_pair_count": latest_conceptual_count,
            "latest_aon_out_of_conceptual_support_self_pair_count": latest_self_count,
        }
    )
    for fraction_name, count_name in (
        ("ever_active_pair_fraction", "ever_active_conceptual_support_pair_count"),
        ("positive_current_pair_fraction", "positive_current_conceptual_support_pair_count"),
        ("latest_aon_active_pair_fraction", "latest_aon_conceptual_support_pair_count"),
    ):
        region_table[fraction_name] = np.divide(
            region_table[count_name].to_numpy(dtype=np.float64),
            conceptual,
            out=np.zeros(trace_map.region_count, dtype=np.float64),
            where=conceptual > 0,
        )
    conceptual_total = int(trace_map.conceptual_regional_pair_count)
    current_self_total = int(np.count_nonzero(current_self))
    latest_self_total = int(np.count_nonzero(latest_self))
    ever_self_total = int(np.count_nonzero(ever_self))
    summary = {
        "suite_id": trace_map.suite_id,
        "conceptual_regional_pair_count": conceptual_total,
        "conceptual_support_definition": (
            "Phase 3 reachable ordered source/target terminal pairs with source_terminal_id "
            "different from target_terminal_id"
        ),
        "ever_active_pair_count": int(len(ever_keys)),
        "ever_active_conceptual_support_pair_count": int(len(ever_keys) - ever_self_total),
        "ever_active_out_of_conceptual_support_self_pair_count": ever_self_total,
        "ever_active_pair_fraction": float(
            (len(ever_keys) - ever_self_total) / max(conceptual_total, 1)
        ),
        "ever_active_raw_pair_to_conceptual_ratio": float(
            len(ever_keys) / max(conceptual_total, 1)
        ),
        "regions_with_ever_raw_pair_count_above_conceptual": int(
            np.count_nonzero(ever_count > conceptual)
        ),
        "ever_raw_excess_over_conceptual_pair_count": int(
            np.maximum(ever_count - conceptual, 0).sum(dtype=np.int64)
        ),
        "positive_current_pair_count": int(len(current.coarse_regional_pair_flow.keys)),
        "positive_current_conceptual_support_pair_count": int(
            len(current.coarse_regional_pair_flow.keys) - current_self_total
        ),
        "positive_current_out_of_conceptual_support_self_pair_count": current_self_total,
        "positive_current_pair_fraction": float(
            (len(current.coarse_regional_pair_flow.keys) - current_self_total)
            / max(conceptual_total, 1)
        ),
        "positive_current_raw_pair_to_conceptual_ratio": float(
            len(current.coarse_regional_pair_flow.keys) / max(conceptual_total, 1)
        ),
        "positive_current_self_pair_flow": float(
            current.coarse_regional_pair_flow.values[current_self].sum(dtype=np.float64)
        ),
        "regions_with_positive_current_raw_pair_count_above_conceptual": int(
            np.count_nonzero(current_count > conceptual)
        ),
        "positive_current_raw_excess_over_conceptual_pair_count": int(
            np.maximum(current_count - conceptual, 0).sum(dtype=np.int64)
        ),
        "latest_aon_active_pair_count": int(len(latest_aon.coarse_regional_pair_flow.keys)),
        "latest_aon_conceptual_support_pair_count": int(
            len(latest_aon.coarse_regional_pair_flow.keys) - latest_self_total
        ),
        "latest_aon_out_of_conceptual_support_self_pair_count": latest_self_total,
        "latest_aon_active_pair_fraction": float(
            (len(latest_aon.coarse_regional_pair_flow.keys) - latest_self_total)
            / max(conceptual_total, 1)
        ),
        "latest_aon_raw_pair_to_conceptual_ratio": float(
            len(latest_aon.coarse_regional_pair_flow.keys) / max(conceptual_total, 1)
        ),
        "latest_aon_self_pair_flow": float(
            latest_aon.coarse_regional_pair_flow.values[latest_self].sum(dtype=np.float64)
        ),
        "regions_with_latest_aon_raw_pair_count_above_conceptual": int(
            np.count_nonzero(latest_count > conceptual)
        ),
        "latest_aon_raw_excess_over_conceptual_pair_count": int(
            np.maximum(latest_count - conceptual, 0).sum(dtype=np.int64)
        ),
        "current_pair_flow_concentration": _top_concentration(
            current.coarse_regional_pair_flow.values
        ),
        "per_region_ever_active_count_quantiles": _quantile_dict(ever_count),
        "per_region_current_active_count_quantiles": _quantile_dict(current_count),
        "per_region_latest_aon_count_quantiles": _quantile_dict(latest_count),
        "bundle_count": trace_map.bundle_count,
        "ever_used_bundle_count": int(np.count_nonzero(ever_active_bundle_mask)),
        "positive_current_bundle_count": int(np.count_nonzero(current.bundle_flow > 0.0)),
        "latest_aon_bundle_count": int(np.count_nonzero(latest_aon.bundle_flow > 0.0)),
        "current_bundle_flow_concentration": _top_concentration(current.bundle_flow),
        "bacn_count": trace_map.bacn_count,
        "positive_current_bacn_arc_flow_count": int(
            np.count_nonzero(current.bacn_internal_arc_flow > 0.0)
        ),
        "positive_current_bacn_traversal_count": int(
            np.count_nonzero(current.bacn_internal_traversal_flow > 0.0)
        ),
    }
    return summary, region_table


def bacn_internal_statistics(
    graph: PartCRev1GraphSnapshot,
    trace_map: SuiteTraceMap,
    physical_flow: np.ndarray,
    physical_cost: np.ndarray,
    trace: SuiteLinearTrace,
) -> pd.DataFrame:
    flow = np.asarray(physical_flow, dtype=np.float64)
    cost = np.asarray(physical_cost, dtype=np.float64)
    internal = trace_map.arc_internal_bacn_local
    mask = internal >= 0
    local = internal[mask]
    arc_flow = flow[mask]

    def aggregate(weights: np.ndarray) -> np.ndarray:
        return np.bincount(local, weights=weights, minlength=trace_map.bacn_count).astype(
            np.float64, copy=False
        )

    fftt_seconds_sum = aggregate(arc_flow * graph.free_time[mask] * 60.0)
    current_seconds_sum = aggregate(arc_flow * cost[mask] * 60.0)
    distance_miles_sum = aggregate(arc_flow * graph.length[mask])
    fftt_seconds_sq_sum = aggregate(arc_flow * np.square(graph.free_time[mask] * 60.0))
    current_seconds_sq_sum = aggregate(arc_flow * np.square(cost[mask] * 60.0))
    distance_miles_sq_sum = aggregate(arc_flow * np.square(graph.length[mask]))
    return pd.DataFrame(
        {
            "suite_id": trace_map.suite_id,
            "bacn_id": trace_map.bacn_ids,
            "internal_arc_flow_sum": trace.bacn_internal_arc_flow,
            "internal_traversal_flow": trace.bacn_internal_traversal_flow,
            "fftt_seconds_flow_sum": fftt_seconds_sum,
            "current_seconds_flow_sum": current_seconds_sum,
            "distance_miles_flow_sum": distance_miles_sum,
            "fftt_seconds_squared_flow_sum": fftt_seconds_sq_sum,
            "current_seconds_squared_flow_sum": current_seconds_sq_sum,
            "distance_miles_squared_flow_sum": distance_miles_sq_sum,
        }
    )
