"""Deterministic long-distance OD sampling and origin-level holdout splits."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import pandas as pd

from .geodesy import grs80_distance_m
from .schemas import stable_object_id


CONUS_STATE_FIPS = frozenset(
    f"{value:02d}" for value in range(1, 57) if value not in {2, 15}
)


class Phase3AODSamplingError(ValueError):
    """Raised when the frozen OD sample cannot be formed or is malformed."""


def physical_reachability_mask(
    *,
    origin_node_idx: np.ndarray,
    destination_node_idx: np.ndarray,
    strong_component: np.ndarray,
    tail: np.ndarray,
    head: np.ndarray,
) -> np.ndarray:
    """Return exact directed reachability using the physical SCC condensation."""

    origin = np.asarray(origin_node_idx, dtype=np.int64)
    destination = np.asarray(destination_node_idx, dtype=np.int64)
    labels = np.asarray(strong_component, dtype=np.int64)
    if origin.shape != destination.shape or origin.ndim != 1:
        raise Phase3AODSamplingError("Reachability OD arrays must align")
    if np.any(origin < 0) or np.any(destination < 0) or np.any(origin >= labels.size) or np.any(destination >= labels.size):
        raise Phase3AODSamplingError("Reachability OD node index is out of range")
    component_count = int(labels.max(initial=-1)) + 1
    adjacency: list[set[int]] = [set() for _ in range(component_count)]
    for left, right in zip(labels[np.asarray(tail, dtype=np.int64)], labels[np.asarray(head, dtype=np.int64)], strict=True):
        if int(left) != int(right):
            adjacency[int(left)].add(int(right))
    result = np.zeros(origin.size, dtype=bool)
    origin_components = labels[origin]
    target_components = labels[destination]
    cache: dict[int, frozenset[int]] = {}
    for component in np.unique(origin_components).tolist():
        source = int(component)
        if source not in cache:
            seen = {source}
            stack = [source]
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            cache[source] = frozenset(seen)
        rows = np.flatnonzero(origin_components == source)
        result[rows] = np.fromiter(
            (int(value) in cache[source] for value in target_components[rows]), dtype=bool, count=rows.size
        )
    return result


def sample_od_pairs(
    *,
    suite_id: str,
    origin_node_idx: np.ndarray,
    destination_node_idx: np.ndarray,
    demand: np.ndarray,
    node_uid: Sequence[str],
    lon: np.ndarray,
    lat: np.ndarray,
    state_fips: Sequence[str],
    physical_weak_component: np.ndarray,
    directed_reachable: np.ndarray,
    minimum_distance_km: float = 500.0,
    target_origins: int = 40,
    destinations_per_origin: int = 5,
    hard_minimum_origins: int = 20,
    hard_minimum_od_count: int = 100,
    seed: int = 42,
    enforce_hard_minimum: bool = True,
    distance_chunk_size: int = 1_000_000,
) -> pd.DataFrame:
    """Apply the frozen sample score and split complete origins 50/50."""

    origin = np.asarray(origin_node_idx, dtype=np.int64)
    destination = np.asarray(destination_node_idx, dtype=np.int64)
    values = np.asarray(demand, dtype=np.float64)
    reachable = np.asarray(directed_reachable, dtype=bool)
    n_nodes = len(node_uid)
    if not (origin.shape == destination.shape == values.shape == reachable.shape) or origin.ndim != 1:
        raise Phase3AODSamplingError("OD sample arrays must be aligned vectors")
    if np.any(origin < 0) or np.any(destination < 0) or np.any(origin >= n_nodes) or np.any(destination >= n_nodes):
        raise Phase3AODSamplingError("OD sample node index is out of range")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise Phase3AODSamplingError("OD sample demand must be finite and positive")
    x = np.asarray(lon, dtype=np.float64)
    y = np.asarray(lat, dtype=np.float64)
    states = np.asarray(state_fips, dtype=str)
    weak = np.asarray(physical_weak_component, dtype=np.int64)
    if not (x.size == y.size == states.size == weak.size == n_nodes):
        raise Phase3AODSamplingError("OD node attributes have inconsistent lengths")
    distance_km = np.empty(origin.size, dtype=np.float64)
    chunk = max(1, int(distance_chunk_size))
    for begin in range(0, origin.size, chunk):
        end = min(origin.size, begin + chunk)
        distance_km[begin:end] = grs80_distance_m(
            x[origin[begin:end]], y[origin[begin:end]], x[destination[begin:end]], y[destination[begin:end]]
        ) / 1000.0
    conus = np.fromiter(
        (states[o] in CONUS_STATE_FIPS and states[d] in CONUS_STATE_FIPS for o, d in zip(origin, destination, strict=True)),
        dtype=bool,
        count=origin.size,
    )
    eligible = (
        (origin != destination)
        & reachable
        & conus
        & np.isfinite(distance_km)
        & (distance_km >= float(minimum_distance_km))
    )
    rows = pd.DataFrame(
        {
            "origin_idx": origin[eligible],
            "destination_idx": destination[eligible],
            "demand": values[eligible],
            "geodesic_distance_km": distance_km[eligible],
        }
    )
    if rows.empty:
        raise Phase3AODSamplingError("No OD rows satisfy the frozen eligible pool")
    # Sampling is by destination, while the accepted OD snapshot may preserve
    # duplicate rows. Aggregate only inside this derived sample view.
    rows = rows.groupby(["origin_idx", "destination_idx"], sort=True, as_index=False).agg(
        demand=("demand", "sum"), geodesic_distance_km=("geodesic_distance_km", "first")
    )
    rows["demand_rank"] = _percentile_rank(np.log1p(rows["demand"].to_numpy(dtype=np.float64)))
    rows["distance_rank"] = _percentile_rank(np.log1p(rows["geodesic_distance_km"].to_numpy(dtype=np.float64)))
    rows["score"] = 0.5 * rows["demand_rank"] + 0.5 * rows["distance_rank"]
    origin_summary = rows.groupby("origin_idx", sort=False).agg(
        eligible_destination_count=("destination_idx", "size"), total_eligible_demand=("demand", "sum")
    ).reset_index()
    origin_summary = origin_summary[origin_summary["eligible_destination_count"] >= int(destinations_per_origin)].copy()
    origin_summary["origin_uid"] = [str(node_uid[int(value)]) for value in origin_summary["origin_idx"]]
    origin_summary = origin_summary.sort_values(
        ["total_eligible_demand", "origin_uid"], ascending=[False, True], kind="mergesort"
    ).head(int(target_origins))
    selected_rows: list[pd.DataFrame] = []
    for origin_index in origin_summary["origin_idx"].tolist():
        group = rows[rows["origin_idx"] == int(origin_index)].copy()
        group["destination_uid"] = [str(node_uid[int(value)]) for value in group["destination_idx"]]
        group = group.sort_values(["score", "destination_uid"], ascending=[False, True], kind="mergesort")
        selected_rows.append(group.head(int(destinations_per_origin)))
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    origin_count = int(selected["origin_idx"].nunique()) if not selected.empty else 0
    if enforce_hard_minimum and (origin_count < int(hard_minimum_origins) or len(selected) < int(hard_minimum_od_count)):
        raise Phase3AODSamplingError(
            f"Frozen hard minimum unavailable at {minimum_distance_km:g} km: origins={origin_count}, ods={len(selected)}"
        )
    selected["origin_uid"] = [str(node_uid[int(value)]) for value in selected["origin_idx"]]
    selected["destination_uid"] = [str(node_uid[int(value)]) for value in selected["destination_idx"]]
    unique_origins = sorted(selected["origin_uid"].unique().tolist(), key=_id_key)
    shuffled = np.asarray(unique_origins, dtype=object)[np.random.default_rng(int(seed)).permutation(len(unique_origins))].tolist()
    development_count = (len(shuffled) + 1) // 2
    development = set(map(str, shuffled[:development_count]))
    selected["split"] = ["development" if value in development else "holdout" for value in selected["origin_uid"]]
    selected["origin_rank"] = selected["origin_uid"].map(
        {value: index + 1 for index, value in enumerate(origin_summary["origin_uid"].astype(str).tolist())}
    )
    selected["sample_od_id"] = [
        stable_object_id(
            "sample_od",
            suite_id=suite_id,
            candidate_id="shared_od_sample",
            members=[row.origin_uid, row.destination_uid],
            ordered_members=True,
            signature={"minimum_distance_km": float(minimum_distance_km), "seed": int(seed)},
        )
        for row in selected.itertuples(index=False)
    ]
    selected["eligibility_reason"] = f"positive_conus_directed_reachable_geodesic_gte_{minimum_distance_km:g}km"
    selected["physical_component_id"] = weak[selected["origin_idx"].to_numpy(dtype=np.int64)]
    result = selected[
        [
            "sample_od_id", "origin_uid", "destination_uid", "origin_idx", "destination_idx",
            "demand", "geodesic_distance_km", "score", "origin_rank", "split", "eligibility_reason",
            "physical_component_id",
        ]
    ].sort_values(["origin_rank", "score", "destination_uid"], ascending=[True, False, True], kind="mergesort")
    _validate_origin_split(result)
    return result.reset_index(drop=True)


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranked = np.empty(array.size, dtype=np.float64)
    position = 0
    while position < array.size:
        end = position + 1
        while end < array.size and array[order[end]] == array[order[position]]:
            end += 1
        average_one_based_rank = 0.5 * ((position + 1) + end)
        ranked[order[position:end]] = average_one_based_rank / array.size
        position = end
    return ranked


def _validate_origin_split(sample: pd.DataFrame) -> None:
    counts = sample.groupby("origin_uid", sort=False)["split"].nunique()
    if (counts != 1).any():
        raise Phase3AODSamplingError("Development/holdout split cuts through an origin")


def _id_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)
