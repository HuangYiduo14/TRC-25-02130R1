"""Exact charged-incidence reconstruction and B-ACN charge diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from partC_rev1_src.phase4.models import SuiteTraceMap
from partC_rev1_src.phase5.scc_reachability import (
    ReachableODSupply,
    load_reachable_od_supply_cache,
    reconstruct_reachable_od_supply,
    write_reachable_od_supply_cache,
)

__all__ = (
    "ReachableODSupply",
    "load_reachable_od_supply_cache",
    "reconstruct_reachable_od_supply",
    "write_reachable_od_supply_cache",
)


def physical_flow_conservation(
    graph: Any,
    physical_flow: np.ndarray,
    supply: ReachableODSupply,
) -> dict[str, float | int | bool]:
    node_count = len(graph.node_idx_to_uid)
    outgoing = np.bincount(graph.tail, weights=physical_flow, minlength=node_count)
    incoming = np.bincount(graph.head, weights=physical_flow, minlength=node_count)
    residual = outgoing - incoming - (
        supply.origin_node_flow - supply.destination_node_flow
    )
    absolute = np.abs(residual)
    return {
        "max_abs_residual": float(absolute.max(initial=0.0)),
        "l1_residual": float(absolute.sum(dtype=np.float64)),
        "node_count_above_1e_6": int(np.count_nonzero(absolute > 1.0e-6)),
        "pass": bool(np.all(absolute <= 1.0e-6)),
    }


def _bacn_endpoint_flows(
    trace_map: SuiteTraceMap,
    supply: ReachableODSupply,
) -> tuple[np.ndarray, np.ndarray]:
    mapping = trace_map.node_to_bacn_local
    count = trace_map.bacn_count
    origin_mask = mapping >= 0
    origin = np.bincount(
        mapping[origin_mask],
        weights=supply.origin_node_flow[origin_mask],
        minlength=count,
    ).astype(np.float64, copy=False)
    destination = np.bincount(
        mapping[origin_mask],
        weights=supply.destination_node_flow[origin_mask],
        minlength=count,
    ).astype(np.float64, copy=False)
    return origin, destination


def reconstruct_bacn_visit_semantics(
    *,
    trace_map: SuiteTraceMap,
    bundle_summary: pd.DataFrame,
    terminals: pd.DataFrame,
    trace_arrays: dict[str, np.ndarray],
    benchmark_id: str,
    supply: ReachableODSupply,
    prefix: str = "current__",
    static_incidence_only: bool = False,
) -> pd.DataFrame:
    """Reconstruct the exact half-incidence visit denominator by direction."""

    bundle_rows = bundle_summary.assign(
        bundle_id=bundle_summary["bundle_id"].astype(str),
        tail_bacn_id=bundle_summary["tail_bacn_id"].astype(str),
        head_bacn_id=bundle_summary["head_bacn_id"].astype(str),
    ).sort_values("bundle_id", kind="mergesort")
    if tuple(bundle_rows["bundle_id"]) != trace_map.bundle_ids:
        raise ValueError(f"{trace_map.suite_id}: bundle flow registry mismatch")
    local = {value: index for index, value in enumerate(trace_map.bacn_ids)}
    tail = bundle_rows["tail_bacn_id"].map(local).to_numpy(dtype=np.int32)
    head = bundle_rows["head_bacn_id"].map(local).to_numpy(dtype=np.int32)
    bundle_flow = np.asarray(trace_arrays[prefix + "bundle_flow"], dtype=np.float64)
    bundle_out = np.bincount(tail, weights=bundle_flow, minlength=trace_map.bacn_count)
    bundle_in = np.bincount(head, weights=bundle_flow, minlength=trace_map.bacn_count)

    terminal_rows = terminals.assign(
        terminal_id=terminals["terminal_id"].astype(str),
        parent_object_id=terminals["parent_object_id"].astype(str),
        terminal_type=terminals["terminal_type"].astype(str),
    ).drop_duplicates("terminal_id", keep="first").set_index("terminal_id").loc[list(trace_map.terminal_ids)]
    terminal_parent = np.full(trace_map.terminal_count, -1, dtype=np.int32)
    is_bacn = terminal_rows["terminal_type"].eq("BACN") & terminal_rows["parent_object_id"].isin(local)
    terminal_parent[np.flatnonzero(is_bacn.to_numpy())] = terminal_rows.loc[
        is_bacn, "parent_object_id"
    ].map(local).to_numpy(dtype=np.int32)
    keys = np.asarray(trace_arrays[prefix + "coarse_regional_pair_flow__keys"], dtype=np.uint64)
    values = np.asarray(trace_arrays[prefix + "coarse_regional_pair_flow__values"], dtype=np.float64)
    entry = (keys >> np.uint64(32)).astype(np.int64)
    exit_ = (keys & np.uint64(0xFFFFFFFF)).astype(np.int64)
    if np.any(entry >= trace_map.terminal_count) or np.any(exit_ >= trace_map.terminal_count):
        raise ValueError(f"{trace_map.suite_id}: coarse pair key exceeds terminal registry")
    entry_parent = terminal_parent[entry]
    exit_parent = terminal_parent[exit_]
    valid_entry = entry_parent >= 0
    valid_exit = exit_parent >= 0
    connector_out = np.bincount(
        entry_parent[valid_entry], weights=values[valid_entry], minlength=trace_map.bacn_count
    )
    connector_in = np.bincount(
        exit_parent[valid_exit], weights=values[valid_exit], minlength=trace_map.bacn_count
    )
    endpoint_in, endpoint_out = _bacn_endpoint_flows(trace_map, supply)
    incoming = bundle_in + connector_in + endpoint_in
    outgoing = bundle_out + connector_out + endpoint_out
    incidence = 0.5 * (incoming + outgoing)
    augmented_incidence = incidence.copy()
    static_incidence = 0.5 * (bundle_in + connector_in + bundle_out + connector_out)
    if static_incidence_only:
        incidence = static_incidence
    stored = np.asarray(trace_arrays[prefix + "bacn_internal_traversal_flow"], dtype=np.float64)
    suppressed = np.asarray(trace_arrays[prefix + "bacn_internal_arc_flow"], dtype=np.float64)
    if (
        bundle_flow.shape != (trace_map.bundle_count,)
        or stored.shape != (trace_map.bacn_count,)
        or suppressed.shape != (trace_map.bacn_count,)
        or keys.shape != values.shape
        or np.any(~np.isfinite(bundle_flow))
        or np.any(~np.isfinite(stored))
        or np.any(~np.isfinite(suppressed))
        or np.any(bundle_flow < 0.0)
        or np.any(stored < 0.0)
        or np.any(suppressed < 0.0)
    ):
        raise ValueError(f"{trace_map.suite_id}: invalid trace arrays for visit reconstruction")
    tolerance = 1.0e-8 + 1.0e-10 * np.maximum(1.0, incidence)
    classification = np.full(trace_map.bacn_count, "MIXED_INTERNAL_AND_ZERO_EPISODE_VISITS", dtype=object)
    equal = np.abs(stored - incidence) <= tolerance
    classification[equal & (incidence <= tolerance)] = "NO_CHARGED_VISIT"
    classification[equal & (incidence > tolerance)] = "ALL_VISITS_HAVE_NONEMPTY_INTERNAL_EPISODE"
    zero_run = (~equal) & (stored <= tolerance) & (suppressed <= tolerance) & (incidence > tolerance)
    classification[zero_run] = "INCIDENCE_ONLY_ZERO_INTERNAL_EPISODE"
    # Endpoint supply closes flow balance, but is not a charged static link.
    # A nonempty internal episode at an OD boundary can exceed static half-incidence.
    endpoint_explained = (stored > incidence + tolerance) & (stored <= augmented_incidence + tolerance)
    classification[endpoint_explained] = "OD_ENDPOINT_EPISODE_EXCEEDS_STATIC_HALF_INCIDENCE"
    defect = stored > augmented_incidence + tolerance
    classification[defect] = "NUMERICAL_OR_MAPPING_DEFECT"
    ratio = np.divide(stored, incidence, out=np.full_like(stored, np.nan), where=incidence > 0.0)
    evidence = [
        json.dumps(
            {
                "stored_rule": "one increment per maximal nonempty contiguous same-B suppressed-arc episode",
                "stored_rule_source": "partC_rev1_src/phase4/kernels.py:486-490,520-522",
                "incidence_rule": "half incoming plus outgoing charged-link endpoint flow",
                "directional_balance_residual": float(incoming[index] - outgoing[index]),
                "od_endpoint_half_flow": float(0.5 * (endpoint_in[index] + endpoint_out[index])),
                "terminal_connector_half_flow": float(0.5 * (connector_in[index] + connector_out[index])),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for index in range(trace_map.bacn_count)
    ]
    return pd.DataFrame(
        {
            "suite_short_id": trace_map.suite_id,
            "benchmark_id": benchmark_id,
            "bacn_id": trace_map.bacn_ids,
            "stored_traversal_flow": stored,
            "incidence_visit_flow": incidence,
            "static_link_incidence_visit_flow": static_incidence,
            "legacy_od_augmented_incidence_visit_flow": augmented_incidence,
            "od_endpoint_half_flow": 0.5*(endpoint_in+endpoint_out),
            "charge_denominator_policy": "STATIC_LINK_INCIDENCE_ONLY_V4" if static_incidence_only else "LEGACY_OD_AUGMENTED_V2",
            "path_episode_visit_flow": np.full(trace_map.bacn_count, np.nan),
            "path_episode_observation_status": "NOT_OBSERVED_IN_ACCEPTED_PATH_SAMPLE",
            "incoming_charged_flow": incoming,
            "outgoing_charged_flow": outgoing,
            "suppressed_arc_flow_sum": suppressed,
            "stored_minus_incidence": stored - incidence,
            "stored_to_incidence_ratio": ratio,
            "semantic_classification": classification,
            "classification_evidence_json": evidence,
            "status": np.where(defect, "FAIL", "PASS"),
        }
    )


class StaticPathProjector:
    """Project existing arc witnesses through the accepted directional terminals.

    No routing is performed. Physical B-member visits inside a regional run
    cannot create extra coarse parent incidence.
    """

    def __init__(self, graph, trace_map, frames):
        self.graph, self.trace_map = graph, trace_map
        self.n_nodes = len(trace_map.node_to_bacn_local)
        static = frames["acnhn_static_links"]
        self.endpoints = {str(r.link_id):(str(r.tail_node_id),str(r.head_node_id)) for r in static.itertuples(index=False)}
        self.bundle_links = dict(zip(trace_map.bundle_ids, trace_map.bundle_ids, strict=True))
        terminal = frames["consolidated_terminals"].drop_duplicates("terminal_id").set_index("terminal_id")
        self.terminal_parent = terminal.parent_object_id.astype(str).to_dict()
        self.connector = {}
        for row in frames["terminal_connectors"].itertuples(index=False):
            key = (str(row.tail_node_id),str(row.head_node_id))
            if key in self.connector:
                raise ValueError("Ambiguous directional static terminal connector")
            self.connector[key] = str(row.link_id)
        self.bacn_local = {n:i for i,n in enumerate(trace_map.bacn_ids)}

    def _terminal(self, region, node, mode):
        key = np.uint64(region)*np.uint64(self.n_nodes)+np.uint64(node)
        keys = self.trace_map.terminal_region_node_keys
        i = int(np.searchsorted(keys, key))
        if i >= len(keys) or keys[i] != key:
            raise ValueError("Accepted path regional boundary has no terminal lookup")
        local = int(self.trace_map.terminal_region_node_values[mode, i])
        if local < 0:
            raise ValueError("Accepted path regional boundary has no directional terminal")
        return self.trace_map.terminal_ids[local]

    def project(self, arcs):
        trace = self.trace_map
        region = trace.arc_region_local[arcs]
        bundle = trace.arc_to_bundle_local[arcs]
        counts = Counter(trace.bundle_ids[int(i)] for i in bundle[bundle >= 0])
        starts = np.flatnonzero((region >= 0) & np.r_[True, region[1:] != region[:-1]])
        ends = np.flatnonzero((region >= 0) & np.r_[region[:-1] != region[1:], True])+1
        for start, end in zip(starts, ends, strict=True):
            entry = self._terminal(int(region[start]), int(self.graph.tail[arcs[start]]), 2 if start == 0 else 0)
            exit_ = self._terminal(int(region[start]), int(self.graph.head[arcs[end-1]]), 3 if end == len(arcs) else 1)
            for terminal, entering in ((entry,True),(exit_,False)):
                parent = self.terminal_parent[terminal]
                pair = (parent,terminal) if entering else (terminal,parent)
                link = self.connector.get(pair)
                if link is None:
                    # Centroid terminals can be terminal nodes themselves; they
                    # own no B charge. A BACN parent always needs its static link.
                    if parent in self.bacn_local:
                        raise ValueError("BACN terminal lacks its accepted static connector")
                else:
                    counts[link] += 1
        weights = Counter()
        for link, count in counts.items():
            if link not in self.endpoints:
                raise ValueError("Projected static link is missing from the accepted registry")
            for node in self.endpoints[link]:
                if node in self.bacn_local:
                    weights[self.bacn_local[node]] += 0.5*count
        return counts, weights


def internal_exposure(
    trace_map: SuiteTraceMap,
    physical_flow: np.ndarray,
    physical_cost_s: np.ndarray,
) -> np.ndarray:
    labels = trace_map.arc_internal_bacn_local
    keep = labels >= 0
    return np.bincount(
        labels[keep],
        weights=physical_flow[keep] * physical_cost_s[keep],
        minlength=trace_map.bacn_count,
    ).astype(np.float64, copy=False)


def path_episode_statistics(
    path_table: str | Path,
    trace_map: SuiteTraceMap,
    physical_cost_s: np.ndarray,
    *,
    arc_column: str = "arc_indices_json",
    sample_weight_by_id: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute unweighted path-sample episode duration diagnostics."""

    columns = [arc_column]
    if sample_weight_by_id is not None:
        columns.append("sample_od_id")
    paths = pd.read_parquet(path_table, columns=columns)
    observations: list[tuple[int, float, float]] = []
    for row in paths.itertuples(index=False):
        encoded = str(getattr(row, arc_column))
        weight = (
            float(sample_weight_by_id[str(row.sample_od_id)])
            if sample_weight_by_id is not None
            else 1.0
        )
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("Accepted path sample weights must be finite and positive")
        arcs = np.asarray(json.loads(encoded), dtype=np.int64)
        labels = trace_map.arc_internal_bacn_local[arcs]
        costs = physical_cost_s[arcs]
        position = 0
        while position < len(arcs):
            label = int(labels[position])
            if label < 0:
                position += 1
                continue
            end = position + 1
            while end < len(arcs) and int(labels[end]) == label:
                end += 1
            observations.append(
                (label, float(costs[position:end].sum(dtype=np.float64)), weight)
            )
            position = end
    grouped: dict[int, list[tuple[float, float]]] = {}
    for label, duration, weight in observations:
        grouped.setdefault(label, []).append((duration, weight))
    rows = []
    for label, values in sorted(grouped.items()):
        array = np.sort(np.asarray([value[0] for value in values], dtype=np.float64))
        weighted_flow = float(sum(value[1] for value in values))
        trim = int(np.floor(0.10 * len(array)))
        selected = array[trim : len(array) - trim] if trim and 2 * trim < len(array) else array
        rows.append(
            {
                "bacn_id": trace_map.bacn_ids[label],
                "sample_episode_count": len(array),
                "sample_demand_weighted_episode_flow": weighted_flow,
                "median_episode_seconds": float(np.median(array)),
                "trimmed_mean_episode_seconds": float(np.mean(selected)),
            }
        )
    return pd.DataFrame(rows)


def fixed_charge_candidates(
    *,
    trace_map: SuiteTraceMap,
    low_visits: np.ndarray,
    high_visits: np.ndarray,
    low_exposure: np.ndarray,
    high_exposure: np.ndarray,
    structural_charges: pd.DataFrame,
    episode_statistics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    structural = structural_charges.assign(
        bacn_id=structural_charges["bacn_id"].astype(str)
    ).set_index("bacn_id").loc[list(trace_map.bacn_ids)]
    seed = structural["structural_charge_s"].to_numpy(dtype=np.float64)
    mean = np.divide(low_exposure, low_visits, out=np.full_like(low_exposure, np.nan), where=low_visits > 0.0)
    candidate_thresholds = np.unique(
        np.r_[0.0, np.quantile(low_visits[low_visits > 0.0], [0.01, 0.05, 0.10, 0.25])]
    )
    diagnostics: list[dict[str, float]] = []
    estimates: list[np.ndarray] = []
    high_residuals: list[np.ndarray] = []
    for threshold in candidate_thresholds:
        estimate = np.divide(
            low_exposure + threshold * seed,
            low_visits + threshold,
            out=seed.copy(),
            where=(low_visits + threshold) > 0.0,
        )
        residual = high_exposure - high_visits * estimate
        estimates.append(estimate)
        high_residuals.append(residual)
        diagnostics.append(
            {
                "pseudo_visit_strength": float(threshold),
                "aggregate_signed_fraction": float(residual.sum(dtype=np.float64) / max(1.0, high_exposure.sum(dtype=np.float64))),
                "aggregate_absolute_fraction": float(np.abs(residual).sum(dtype=np.float64) / max(1.0, high_exposure.sum(dtype=np.float64))),
            }
        )
    # Evidence is diagnostic only: two operating states do not freeze a
    # universal threshold.  Raw mean remains the well-observed primary class.
    effective = np.where(np.isfinite(mean), mean, seed)
    residual = np.where(
        np.isfinite(mean), high_exposure - high_visits * mean, np.nan
    )
    recommended_residual = high_exposure - high_visits * effective
    per_node_diagnostics = [
        json.dumps(
            {
                "pseudo_visit_strengths": [float(value) for value in candidate_thresholds],
                "estimate_seconds": [float(value[index]) for value in estimates],
                "high_residual_flow_seconds": [
                    float(value[index]) for value in high_residuals
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for index in range(trace_map.bacn_count)
    ]
    result = pd.DataFrame(
        {
            "suite_short_id": trace_map.suite_id,
            "bacn_id": trace_map.bacn_ids,
            "low_visit_flow": low_visits,
            "low_internal_exposure_flow_seconds": low_exposure,
            "low_operating_mean_seconds": mean,
            "structural_seed_seconds": seed,
            "median_episode_seconds": np.nan,
            "trimmed_mean_episode_seconds": np.nan,
            "high_visit_flow": high_visits,
            "high_internal_exposure_flow_seconds": high_exposure,
            "high_residual_using_low_mean": residual,
            "recommended_charge_seconds": effective,
            "high_residual_using_recommended": recommended_residual,
            "candidate_shrinkage_json": per_node_diagnostics,
            "candidate_shrinkage_aggregate_diagnostics_json": json.dumps(
                diagnostics, sort_keys=True, separators=(",", ":")
            ),
            "unvisited_fallback_class": np.where(
                low_visits > 0.0,
                "NOT_APPLICABLE_VISITED",
                np.where(seed > 0.0, "STRUCTURAL_SEED", "STRUCTURAL_ZERO_NO_INTERNAL_EVIDENCE"),
            ),
            "recommended_estimator_class": np.where(
                low_visits > 0.0, "LOW_OPERATING_EXPOSURE_MEAN", "STRUCTURAL_FALLBACK"
            ),
            "status": "PILOT_DIAGNOSTIC_THRESHOLD_NOT_FROZEN",
        }
    )
    if episode_statistics is not None and not episode_statistics.empty:
        mapped = episode_statistics.set_index("bacn_id")
        for column in ("median_episode_seconds", "trimmed_mean_episode_seconds"):
            result[column] = result["bacn_id"].map(mapped[column])
        result["sample_episode_count"] = result["bacn_id"].map(
            mapped["sample_episode_count"]
        )
        result["sample_demand_weighted_episode_flow"] = result["bacn_id"].map(
            mapped["sample_demand_weighted_episode_flow"]
        )
    return result
