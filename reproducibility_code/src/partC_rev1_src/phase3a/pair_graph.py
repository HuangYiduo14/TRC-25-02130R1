"""Sparse GEO/AUG pair-admissibility graph construction."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from .geodesy import ecef_candidate_pairs, exact_filter_pairs
from .models import PairGraph


class Phase3APairGraphError(ValueError):
    pass


def build_geo_pair_graph(
    node_ids: Iterable[str],
    lon: np.ndarray,
    lat: np.ndarray,
    physical_weak_component: np.ndarray,
    threshold_m: float,
) -> PairGraph:
    """Build the exact GRS80 relation after conservative ECEF filtering."""

    ids = tuple(str(value) for value in node_ids)
    x = np.asarray(lon, dtype=np.float64)
    y = np.asarray(lat, dtype=np.float64)
    component = np.asarray(physical_weak_component, dtype=np.int64)
    if not (len(ids) == x.size == y.size == component.size):
        raise Phase3APairGraphError("Pair-graph input vectors have inconsistent lengths")
    if x.size and (np.any(~np.isfinite(x)) or np.any(~np.isfinite(y))):
        raise Phase3APairGraphError("Pair-graph coordinates must be finite")
    candidate = ecef_candidate_pairs(x, y, threshold_m)
    if candidate.size:
        same_component = component[candidate[:, 0]] == component[candidate[:, 1]]
        candidate = candidate[same_component]
    accepted, distance = exact_filter_pairs(x, y, candidate, threshold_m)
    if accepted.size:
        order = np.lexsort((accepted[:, 1], accepted[:, 0]))
        accepted = accepted[order]
        distance = distance[order]
    return PairGraph(
        node_ids=ids,
        edge_u=accepted[:, 0].astype(np.int64) if accepted.size else np.empty(0, dtype=np.int64),
        edge_v=accepted[:, 1].astype(np.int64) if accepted.size else np.empty(0, dtype=np.int64),
        dissimilarity=distance.astype(np.float64),
        component_ids=component.copy(),
        metric="GRS80_GEODESIC",
        threshold=float(threshold_m),
        threshold_unit="meters",
    )


def build_relation_pair_graph(
    node_ids: Iterable[str],
    component_ids: np.ndarray,
    pair_dissimilarity: Mapping[tuple[int, int], float],
    *,
    threshold: float,
    metric: str,
    threshold_unit: str,
) -> PairGraph:
    ids = tuple(str(value) for value in node_ids)
    components = np.asarray(component_ids, dtype=np.int64)
    if len(ids) != components.size:
        raise Phase3APairGraphError("Relation graph component vector has wrong length")
    records: list[tuple[int, int, float]] = []
    for pair, value in pair_dissimilarity.items():
        if len(pair) != 2:
            raise Phase3APairGraphError("Pair key must contain two node indexes")
        u, v = sorted((int(pair[0]), int(pair[1])))
        dissimilarity = float(value)
        if u == v or u < 0 or v >= len(ids):
            raise Phase3APairGraphError(f"Invalid relation pair: {pair}")
        if components[u] != components[v]:
            continue
        if not np.isfinite(dissimilarity) or dissimilarity < 0.0:
            raise Phase3APairGraphError("Pair dissimilarity must be finite and nonnegative")
        if dissimilarity <= float(threshold):
            records.append((u, v, dissimilarity))
    records.sort(key=lambda row: (row[0], row[1], row[2]))
    deduplicated: list[tuple[int, int, float]] = []
    for record in records:
        if deduplicated and record[:2] == deduplicated[-1][:2]:
            if record[2] < deduplicated[-1][2]:
                deduplicated[-1] = record
        else:
            deduplicated.append(record)
    return PairGraph(
        node_ids=ids,
        edge_u=np.asarray([row[0] for row in deduplicated], dtype=np.int64),
        edge_v=np.asarray([row[1] for row in deduplicated], dtype=np.int64),
        dissimilarity=np.asarray([row[2] for row in deduplicated], dtype=np.float64),
        component_ids=components.copy(),
        metric=str(metric),
        threshold=float(threshold),
        threshold_unit=str(threshold_unit),
    )


def induced_pair_graph(graph: PairGraph, selected_indexes: Iterable[int]) -> PairGraph:
    selected = sorted(set(int(value) for value in selected_indexes))
    if any(value < 0 or value >= len(graph.node_ids) for value in selected):
        raise Phase3APairGraphError("Induced graph contains an out-of-range node index")
    remap = {old: new for new, old in enumerate(selected)}
    keep = np.fromiter(
        (int(u) in remap and int(v) in remap for u, v in zip(graph.edge_u, graph.edge_v, strict=True)),
        dtype=bool,
        count=graph.edge_count(),
    )
    old_u = graph.edge_u[keep]
    old_v = graph.edge_v[keep]
    return PairGraph(
        node_ids=tuple(graph.node_ids[index] for index in selected),
        edge_u=np.asarray([remap[int(value)] for value in old_u], dtype=np.int64),
        edge_v=np.asarray([remap[int(value)] for value in old_v], dtype=np.int64),
        dissimilarity=graph.dissimilarity[keep].copy(),
        component_ids=graph.component_ids[np.asarray(selected, dtype=np.int64)].copy(),
        metric=graph.metric,
        threshold=graph.threshold,
        threshold_unit=graph.threshold_unit,
    )


def connected_components(graph: PairGraph) -> tuple[tuple[int, ...], ...]:
    adjacency = graph.adjacency()
    unseen = set(range(len(graph.node_ids)))
    result: list[tuple[int, ...]] = []
    while unseen:
        seed = min(unseen, key=lambda index: _id_key(graph.node_ids[index]))
        stack = [seed]
        unseen.remove(seed)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], key=lambda index: _id_key(graph.node_ids[index]), reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        result.append(tuple(sorted(component, key=lambda index: _id_key(graph.node_ids[index]))))
    result.sort(key=lambda values: _id_key(graph.node_ids[values[0]]))
    return tuple(result)


def _id_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)
