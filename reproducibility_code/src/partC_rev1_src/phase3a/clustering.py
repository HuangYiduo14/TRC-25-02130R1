"""Deterministic complete-link, greedy clique-cover, and SL clustering."""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass
from collections.abc import Callable
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sparse_connected_components

from .models import ClusterResult, PairGraph


class Phase3AClusteringError(ValueError):
    pass


def cluster_pair_graph(
    graph: PairGraph,
    algorithm: str,
    *,
    complete_distance: Callable[[str, str], float] | None = None,
) -> ClusterResult:
    method = algorithm.upper()
    compact = _CompactPairIndex.from_graph(graph)
    if method == "CL":
        cluster_indexes = _complete_link(graph, compact)
    elif method == "GC":
        cluster_indexes = _greedy_clique_cover(graph, compact)
    elif method == "SL":
        cluster_indexes = _single_link_components(graph, compact)
    else:
        raise Phase3AClusteringError(f"Unsupported clustering algorithm: {algorithm}")
    clusters = tuple(
        tuple(sorted((graph.node_ids[index] for index in values), key=_id_key))
        for values in cluster_indexes
    )
    clusters = tuple(sorted(clusters, key=lambda values: _id_key(values[0])))
    diameters = tuple(
        compact.diameter(values, complete_distance)
        for values in cluster_indexes
    )
    valid = all(value <= graph.threshold + max(1e-12, abs(graph.threshold) * 1e-12) for value in diameters)
    if method in {"CL", "GC"} and not valid:
        raise Phase3AClusteringError(f"{method} produced a non-clique/over-diameter cluster")
    digest = hashlib.sha256()
    for cluster in clusters:
        digest.update("\x1f".join(cluster).encode("utf-8"))
        digest.update(b"\n")
    return ClusterResult(
        algorithm=method,
        clusters=clusters,
        actual_diameters=diameters,
        production_pairwise_valid=bool(valid and method != "SL"),
        membership_hash=digest.hexdigest(),
    )


def _complete_link(graph: PairGraph, compact: "_CompactPairIndex") -> tuple[tuple[int, ...], ...]:
    """Run the reviewed complete-link rule without per-edge Python objects.

    The former implementation materialized every singleton relation twice in
    dictionaries and once more as a five-item Python heap tuple.  At the
    national 8 km threshold that exhausted the Windows commit limit before the
    clustering rule could finish.  Here the immutable singleton heap is an
    exact NumPy lexicographic stream.  Only relations involving newly merged
    clusters enter the dynamic heap.
    """

    node_count = len(graph.node_ids)
    if node_count == 0:
        return ()
    uid_rank = _uid_ranks(graph.node_ids)
    edge_u, edge_v, distance = compact.edge_u, compact.edge_v, compact.distance
    left_rank = np.minimum(uid_rank[edge_u], uid_rank[edge_v])
    right_rank = np.maximum(uid_rank[edge_u], uid_rank[edge_v])
    initial_order = np.lexsort((right_rank, left_rank, distance))
    del left_rank, right_rank

    capacity = max(1, 2 * node_count)
    active = np.zeros(capacity, dtype=bool)
    active[:node_count] = True
    minimum_rank = np.full(capacity, np.iinfo(np.int64).max, dtype=np.int64)
    minimum_rank[:node_count] = uid_rank
    members: list[np.ndarray | None] = [None] * capacity
    node_cluster = np.arange(node_count, dtype=np.int64)
    dynamic: list[tuple[float, int, int, int, int]] = []
    initial_position = 0
    next_cluster_id = node_count

    def cluster_members(cluster_id: int) -> np.ndarray:
        value = members[cluster_id]
        if value is None:
            return np.asarray([cluster_id], dtype=np.int64)
        return value

    def next_initial() -> tuple[float, int, int, int, int] | None:
        nonlocal initial_position
        while initial_position < initial_order.size:
            edge_index = int(initial_order[initial_position])
            first, second = int(edge_u[edge_index]), int(edge_v[edge_index])
            if minimum_rank[first] > minimum_rank[second]:
                first, second = second, first
            entry = (
                float(distance[edge_index]),
                int(minimum_rank[first]),
                int(minimum_rank[second]),
                first,
                second,
            )
            if active[first] and active[second]:
                return entry
            initial_position += 1
        return None

    while True:
        initial = next_initial()
        while dynamic and (not active[dynamic[0][3]] or not active[dynamic[0][4]]):
            heapq.heappop(dynamic)
        queued_dynamic = dynamic[0] if dynamic else None
        if initial is None and queued_dynamic is None:
            break
        if queued_dynamic is None or (initial is not None and initial <= queued_dynamic):
            queued = initial
            initial_position += 1
        else:
            queued = heapq.heappop(dynamic)
        assert queued is not None
        queued_distance, _, _, left, right = queued
        if not active[left] or not active[right] or left == right:
            continue
        legal, exact_distance = compact.cross_distance(cluster_members(left), cluster_members(right))
        if not legal:
            continue
        ordered_left, ordered_right = _ordered_rank_pair(left, right, minimum_rank)
        expected = (
            float(exact_distance),
            int(minimum_rank[ordered_left]),
            int(minimum_rank[ordered_right]),
            ordered_left,
            ordered_right,
        )
        observed = (
            float(queued_distance),
            int(minimum_rank[left]),
            int(minimum_rank[right]),
            left,
            right,
        )
        if observed != expected:
            heapq.heappush(dynamic, expected)
            continue

        merged = np.concatenate((cluster_members(left), cluster_members(right)))
        merged = merged[np.argsort(uid_rank[merged], kind="stable")]
        active[left] = False
        active[right] = False
        active[next_cluster_id] = True
        members[next_cluster_id] = merged
        minimum_rank[next_cluster_id] = min(minimum_rank[left], minimum_rank[right])
        node_cluster[merged] = next_cluster_id

        # Any legal complete-link neighbor must contain a node adjacent to the
        # merged cluster's first member.  This exact necessary condition avoids
        # rescanning every active national cluster after each merge.
        incident_clusters = np.unique(node_cluster[compact.neighbors(int(merged[0]))])
        for other_value in incident_clusters.tolist():
            other = int(other_value)
            if other == next_cluster_id or not active[other]:
                continue
            legal, candidate_distance = compact.cross_distance(merged, cluster_members(other))
            if legal:
                first, second = _ordered_rank_pair(next_cluster_id, other, minimum_rank)
                heapq.heappush(
                    dynamic,
                    (
                        float(candidate_distance),
                        int(minimum_rank[first]),
                        int(minimum_rank[second]),
                        first,
                        second,
                    ),
                )
        next_cluster_id += 1

    result: list[tuple[int, ...]] = []
    for cluster_id in np.flatnonzero(active).tolist():
        values = cluster_members(int(cluster_id))
        result.append(tuple(int(value) for value in values.tolist()))
    result.sort(key=lambda values: int(uid_rank[values[0]]))
    return tuple(result)


def _greedy_clique_cover(
    graph: PairGraph, compact: "_CompactPairIndex"
) -> tuple[tuple[int, ...], ...]:
    node_count = len(graph.node_ids)
    unassigned = np.ones(node_count, dtype=bool)
    degree = np.bincount(
        np.concatenate((compact.edge_u, compact.edge_v)), minlength=node_count
    )
    seed_order = sorted(
        range(node_count), key=lambda index: (-int(degree[index]), _id_key(graph.node_ids[index]))
    )
    result: list[tuple[int, ...]] = []
    for seed in seed_order:
        if not unassigned[seed]:
            continue
        cluster = [seed]
        unassigned[seed] = False
        # A node is eligible exactly when it is still unassigned and belongs
        # to every current member's admissibility neighborhood.  Maintaining
        # this intersection is mathematically identical to rescanning the
        # entire national unassigned set after each addition, but avoids the
        # former O(number_of_nodes^2) behavior on sparse pair graphs.
        eligible = {int(value) for value in compact.neighbors(seed).tolist() if unassigned[int(value)]}
        while True:
            if not eligible:
                break
            selected = min(
                eligible,
                key=lambda node: (
                    compact.maximum_distance(node, cluster),
                    -int(np.count_nonzero(unassigned[compact.neighbors(node)])),
                    _id_key(graph.node_ids[node]),
                ),
            )
            cluster.append(selected)
            unassigned[selected] = False
            eligible.remove(selected)
            eligible.intersection_update(int(value) for value in compact.neighbors(selected).tolist())
            eligible = {value for value in eligible if unassigned[value]}
        result.append(tuple(sorted(cluster, key=lambda index: _id_key(graph.node_ids[index]))))
    result.sort(key=lambda values: _id_key(graph.node_ids[values[0]]))
    return tuple(result)


def _single_link_components(
    graph: PairGraph, compact: "_CompactPairIndex"
) -> tuple[tuple[int, ...], ...]:
    node_count = len(graph.node_ids)
    if node_count == 0:
        return ()
    if compact.edge_u.size == 0:
        return tuple((index,) for index in sorted(range(node_count), key=lambda value: _id_key(graph.node_ids[value])))
    row = np.concatenate((compact.edge_u, compact.edge_v)).astype(np.int32, copy=False)
    column = np.concatenate((compact.edge_v, compact.edge_u)).astype(np.int32, copy=False)
    data = np.ones(row.size, dtype=np.uint8)
    adjacency = csr_matrix((data, (row, column)), shape=(node_count, node_count))
    _, labels = sparse_connected_components(adjacency, directed=False, return_labels=True)
    groups: dict[int, list[int]] = {}
    for node, label in enumerate(labels.tolist()):
        groups.setdefault(int(label), []).append(node)
    result = [tuple(sorted(values, key=lambda index: _id_key(graph.node_ids[index]))) for values in groups.values()]
    result.sort(key=lambda values: _id_key(graph.node_ids[values[0]]))
    return tuple(result)


def _ordered_rank_pair(left: int, right: int, minimum_rank: np.ndarray) -> tuple[int, int]:
    if int(minimum_rank[left]) <= int(minimum_rank[right]):
        return left, right
    return right, left


def _uid_ranks(node_ids: tuple[str, ...]) -> np.ndarray:
    order = sorted(range(len(node_ids)), key=lambda index: _id_key(node_ids[index]))
    rank = np.empty(len(node_ids), dtype=np.int64)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(len(order), dtype=np.int64)
    return rank


@dataclass(slots=True)
class _CompactPairIndex:
    node_ids: tuple[str, ...]
    edge_u: np.ndarray
    edge_v: np.ndarray
    distance: np.ndarray
    edge_key: np.ndarray
    reverse_order: np.ndarray
    reverse_v: np.ndarray

    @classmethod
    def from_graph(cls, graph: PairGraph) -> "_CompactPairIndex":
        node_count = len(graph.node_ids)
        raw_u = np.asarray(graph.edge_u, dtype=np.int64)
        raw_v = np.asarray(graph.edge_v, dtype=np.int64)
        raw_distance = np.asarray(graph.dissimilarity, dtype=np.float64)
        if not (raw_u.ndim == raw_v.ndim == raw_distance.ndim == 1):
            raise Phase3AClusteringError("Pair graph arrays must be one-dimensional")
        if not (raw_u.size == raw_v.size == raw_distance.size):
            raise Phase3AClusteringError("Pair graph arrays have inconsistent lengths")
        if raw_u.size and (
            np.any(raw_u < 0) or np.any(raw_v < 0) or np.any(raw_u >= node_count) or np.any(raw_v >= node_count)
        ):
            raise Phase3AClusteringError("Pair graph contains an out-of-range node index")
        if raw_u.size and (np.any(raw_u == raw_v) or np.any(~np.isfinite(raw_distance)) or np.any(raw_distance < 0.0)):
            raise Phase3AClusteringError("Pair graph contains an invalid edge or distance")
        low, high = np.minimum(raw_u, raw_v), np.maximum(raw_u, raw_v)
        key = low * np.int64(max(1, node_count)) + high
        if key.size > 1 and np.any(key[1:] <= key[:-1]):
            order = np.argsort(key, kind="stable")
            low, high, raw_distance, key = low[order], high[order], raw_distance[order], key[order]
        if key.size > 1 and np.any(key[1:] == key[:-1]):
            raise Phase3AClusteringError("Pair graph contains duplicate undirected edges")
        reverse_order = np.argsort(high, kind="stable")
        return cls(
            node_ids=tuple(map(str, graph.node_ids)),
            edge_u=low,
            edge_v=high,
            distance=raw_distance,
            edge_key=key,
            reverse_order=reverse_order,
            reverse_v=high[reverse_order],
        )

    @property
    def node_count(self) -> int:
        return len(self.node_ids)

    def neighbors(self, node: int) -> np.ndarray:
        forward_begin = int(np.searchsorted(self.edge_u, node, side="left"))
        forward_end = int(np.searchsorted(self.edge_u, node, side="right"))
        reverse_begin = int(np.searchsorted(self.reverse_v, node, side="left"))
        reverse_end = int(np.searchsorted(self.reverse_v, node, side="right"))
        forward = self.edge_v[forward_begin:forward_end]
        reverse = self.edge_u[self.reverse_order[reverse_begin:reverse_end]]
        if forward.size == 0:
            return reverse
        if reverse.size == 0:
            return forward
        return np.concatenate((forward, reverse))

    def cross_distance(self, left: Iterable[int], right: Iterable[int]) -> tuple[bool, float]:
        first = np.asarray(tuple(left), dtype=np.int64)
        second = np.asarray(tuple(right), dtype=np.int64)
        if first.size > second.size:
            first, second = second, first
        maximum = 0.0
        for node in first.tolist():
            low = np.minimum(int(node), second)
            high = np.maximum(int(node), second)
            keys = low * np.int64(max(1, self.node_count)) + high
            positions = np.searchsorted(self.edge_key, keys)
            valid = positions < self.edge_key.size
            if not np.all(valid):
                return False, float("inf")
            if not np.array_equal(self.edge_key[positions], keys):
                return False, float("inf")
            if positions.size:
                maximum = max(maximum, float(np.max(self.distance[positions])))
        return True, maximum

    def maximum_distance(self, node: int, members: Iterable[int]) -> float:
        legal, maximum = self.cross_distance((int(node),), members)
        if not legal:
            raise Phase3AClusteringError("Clique candidate is missing a required pair edge")
        return maximum

    def diameter(
        self,
        members: Iterable[int],
        complete_distance: Callable[[str, str], float] | None,
    ) -> float:
        values = np.asarray(tuple(members), dtype=np.int64)
        if values.size < 2:
            return 0.0
        maximum = 0.0
        for position, left in enumerate(values[:-1].tolist()):
            right = values[position + 1 :]
            low = np.minimum(int(left), right)
            high = np.maximum(int(left), right)
            keys = low * np.int64(max(1, self.node_count)) + high
            indexes = np.searchsorted(self.edge_key, keys)
            found = indexes < self.edge_key.size
            if np.any(found):
                found_positions = np.flatnonzero(found)
                found[found_positions] = self.edge_key[indexes[found_positions]] == keys[found_positions]
            if np.any(found):
                maximum = max(maximum, float(np.max(self.distance[indexes[found]])))
            if np.any(~found) and complete_distance is not None:
                cluster_diameter = getattr(complete_distance, "cluster_diameter", None)
                if callable(cluster_diameter):
                    candidate = float(
                        cluster_diameter(tuple(self.node_ids[int(value)] for value in values.tolist()))
                    )
                    if np.isnan(candidate) or candidate < 0.0:
                        raise Phase3AClusteringError(
                            "Complete-distance provider returned an invalid cluster diameter"
                        )
                    return candidate
            for missing in np.flatnonzero(~found).tolist():
                if complete_distance is None:
                    candidate = float("inf")
                else:
                    candidate = float(
                        complete_distance(self.node_ids[int(left)], self.node_ids[int(right[missing])])
                    )
                if np.isnan(candidate) or candidate < 0.0:
                    raise Phase3AClusteringError("Complete-distance provider returned an invalid value")
                maximum = max(maximum, candidate)
        return maximum


# Retained only for focused equivalence tests of the memory repair.  Production
# execution always uses the compact implementation above.
def _complete_link_legacy(graph: PairGraph) -> tuple[tuple[int, ...], ...]:
    adjacency = graph.adjacency()
    active: dict[int, frozenset[int]] = {index: frozenset((index,)) for index in range(len(graph.node_ids))}
    minimum_uid: dict[int, str] = {index: graph.node_ids[index] for index in active}
    heap: list[tuple[float, tuple[int, int | str], tuple[int, int | str], int, int]] = []
    for u, v, distance in zip(graph.edge_u.tolist(), graph.edge_v.tolist(), graph.dissimilarity.tolist(), strict=True):
        left, right = _ordered_cluster_pair_legacy(int(u), int(v), minimum_uid)
        heapq.heappush(heap, (float(distance), _id_key(minimum_uid[left]), _id_key(minimum_uid[right]), left, right))
    next_cluster_id = len(active)
    while heap:
        queued_distance, _, _, left, right = heapq.heappop(heap)
        if left not in active or right not in active or left == right:
            continue
        legal, exact_distance = _cross_pair_distance_legacy(active[left], active[right], adjacency)
        if not legal:
            continue
        ordered_left, ordered_right = _ordered_cluster_pair_legacy(left, right, minimum_uid)
        expected = (exact_distance, _id_key(minimum_uid[ordered_left]), _id_key(minimum_uid[ordered_right]))
        queued = (queued_distance, _id_key(minimum_uid[left]), _id_key(minimum_uid[right]))
        if queued != expected:
            heapq.heappush(heap, (*expected, ordered_left, ordered_right))
            continue
        merged = active.pop(left) | active.pop(right)
        minimum_uid.pop(left)
        minimum_uid.pop(right)
        new_id = next_cluster_id
        next_cluster_id += 1
        active[new_id] = merged
        minimum_uid[new_id] = min((graph.node_ids[index] for index in merged), key=_id_key)
        for other in list(active):
            if other == new_id:
                continue
            legal, candidate_distance = _cross_pair_distance_legacy(merged, active[other], adjacency)
            if legal:
                first, second = _ordered_cluster_pair_legacy(new_id, other, minimum_uid)
                heapq.heappush(
                    heap,
                    (candidate_distance, _id_key(minimum_uid[first]), _id_key(minimum_uid[second]), first, second),
                )
    result = [tuple(sorted(values, key=lambda index: _id_key(graph.node_ids[index]))) for values in active.values()]
    result.sort(key=lambda values: _id_key(graph.node_ids[values[0]]))
    return tuple(result)


def _cross_pair_distance_legacy(
    left: Iterable[int], right: Iterable[int], adjacency: tuple[dict[int, float], ...]
) -> tuple[bool, float]:
    maximum = 0.0
    for u in left:
        for v in right:
            if v not in adjacency[u]:
                return False, float("inf")
            maximum = max(maximum, float(adjacency[u][v]))
    return True, maximum


def _ordered_cluster_pair_legacy(left: int, right: int, minimum_uid: dict[int, str]) -> tuple[int, int]:
    if _id_key(minimum_uid[left]) <= _id_key(minimum_uid[right]):
        return left, right
    return right, left


def _id_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)
