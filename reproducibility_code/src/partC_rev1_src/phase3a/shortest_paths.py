"""Deterministic sparse shortest-path interfaces used by Phase 3A.

The national physical-tree wrapper reuses the validated Phase 1/2 Numba CSR
backend.  Restricted searches use the same nonnegative-weight and stable-arc
tie contract while supporting multi-source, target, hop, and time bounds.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from partC_rev1_src.partC_rev1_shortest_paths import PartCRev1SSSPResult, partC_rev1_sssp

from .models import GraphData


DEFAULT_TIE_TOLERANCE_S = 1.0e-10


class Phase3AShortestPathError(ValueError):
    """Raised when a restricted or physical path request is invalid."""


@dataclass(frozen=True, slots=True)
class SparseDirectedGraph:
    """CSR directed graph retaining global node and physical arc identities."""

    node_ids: tuple[int, ...]
    indptr: np.ndarray
    head_local: np.ndarray
    arc_index: np.ndarray
    cost_s: np.ndarray
    arc_stable_id: tuple[str, ...]

    @property
    def node_count(self) -> int:
        return len(self.node_ids)

    @property
    def arc_count(self) -> int:
        return int(self.arc_index.size)


@dataclass(frozen=True, slots=True)
class RestrictedTree:
    source_local: tuple[int, ...]
    distances_s: np.ndarray
    predecessor_local: np.ndarray
    predecessor_arc_position: np.ndarray
    settled_order: np.ndarray


def build_sparse_directed_graph(
    *,
    tail: np.ndarray,
    head: np.ndarray,
    cost_s: np.ndarray,
    arc_stable_id: Sequence[str],
    arc_mask: np.ndarray | None = None,
    node_subset: Iterable[int] | None = None,
) -> SparseDirectedGraph:
    """Build deterministic local CSR arrays from global primitive arrays."""

    tail_array = np.asarray(tail, dtype=np.int64)
    head_array = np.asarray(head, dtype=np.int64)
    costs = np.asarray(cost_s, dtype=np.float64)
    ids = tuple(str(value) for value in arc_stable_id)
    if not (tail_array.ndim == head_array.ndim == costs.ndim == 1):
        raise Phase3AShortestPathError("Graph arrays must be one-dimensional")
    if not (tail_array.size == head_array.size == costs.size == len(ids)):
        raise Phase3AShortestPathError("Graph arrays must have identical lengths")
    if np.any(~np.isfinite(costs)) or np.any(costs < 0.0):
        raise Phase3AShortestPathError("Sparse graph costs must be finite and nonnegative")
    selected_arcs = np.ones(tail_array.size, dtype=bool) if arc_mask is None else np.asarray(arc_mask, dtype=bool)
    if selected_arcs.shape != tail_array.shape:
        raise Phase3AShortestPathError("Arc mask shape differs from graph arrays")
    if node_subset is None:
        selected_nodes = np.unique(np.concatenate((tail_array[selected_arcs], head_array[selected_arcs])))
    else:
        selected_nodes = np.asarray(sorted(set(int(value) for value in node_subset)), dtype=np.int64)
        allowed = set(selected_nodes.tolist())
        selected_arcs &= np.fromiter(
            (int(u) in allowed and int(v) in allowed for u, v in zip(tail_array, head_array, strict=True)),
            dtype=bool,
            count=tail_array.size,
        )
    if selected_nodes.size == 0:
        raise Phase3AShortestPathError("Sparse graph may not have zero nodes")
    local_by_global = {int(value): index for index, value in enumerate(selected_nodes.tolist())}
    positions = np.flatnonzero(selected_arcs).tolist()
    positions.sort(
        key=lambda pos: (
            local_by_global[int(tail_array[pos])],
            _id_key(ids[pos]),
            local_by_global[int(head_array[pos])],
            pos,
        )
    )
    local_tail = np.asarray([local_by_global[int(tail_array[pos])] for pos in positions], dtype=np.int64)
    local_head = np.asarray([local_by_global[int(head_array[pos])] for pos in positions], dtype=np.int64)
    counts = np.bincount(local_tail, minlength=selected_nodes.size)
    indptr = np.empty(selected_nodes.size + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return SparseDirectedGraph(
        node_ids=tuple(int(value) for value in selected_nodes.tolist()),
        indptr=indptr,
        head_local=local_head,
        arc_index=np.asarray(positions, dtype=np.int64),
        cost_s=costs[np.asarray(positions, dtype=np.int64)],
        arc_stable_id=tuple(ids[pos] for pos in positions),
    )


def restricted_dijkstra(
    graph: SparseDirectedGraph,
    source_global_nodes: Iterable[int],
    *,
    maximum_distance_s: float | None = None,
    maximum_arcs: int | None = None,
    target_global_nodes: Iterable[int] | None = None,
    prohibited_through_global_nodes: Iterable[int] = (),
    tie_tolerance_s: float = DEFAULT_TIE_TOLERANCE_S,
) -> RestrictedTree:
    """Run deterministic bounded multi-source Dijkstra on a restricted CSR."""

    global_to_local = {value: index for index, value in enumerate(graph.node_ids)}
    sources = tuple(sorted({global_to_local[int(value)] for value in source_global_nodes if int(value) in global_to_local}))
    if not sources:
        raise Phase3AShortestPathError("Restricted Dijkstra has no source in the graph")
    limit = float("inf") if maximum_distance_s is None else float(maximum_distance_s)
    if not np.isfinite(limit) and limit != float("inf") or limit < 0.0:
        raise Phase3AShortestPathError("Maximum distance must be nonnegative")
    hop_limit = np.iinfo(np.int32).max if maximum_arcs is None else int(maximum_arcs)
    if hop_limit < 0:
        raise Phase3AShortestPathError("Maximum arc count must be nonnegative")
    targets = None
    if target_global_nodes is not None:
        targets = {global_to_local[int(value)] for value in target_global_nodes if int(value) in global_to_local}
    prohibited = {global_to_local[int(value)] for value in prohibited_through_global_nodes if int(value) in global_to_local}
    n = graph.node_count
    distances = np.full(n, np.inf, dtype=np.float64)
    hops = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
    predecessor = np.full(n, -1, dtype=np.int64)
    predecessor_arc = np.full(n, -1, dtype=np.int64)
    path_key: list[tuple[tuple[int, int | str], ...] | None] = [None] * n
    heap: list[tuple[float, int, tuple[tuple[int, int | str], ...], int]] = []
    for source in sources:
        distances[source] = 0.0
        hops[source] = 0
        path_key[source] = ()
        heapq.heappush(heap, (0.0, 0, (), source))
    settled: list[int] = []
    settled_mask = np.zeros(n, dtype=bool)
    remaining_targets = None if targets is None else set(targets)
    while heap:
        distance, hop_count, key, node = heapq.heappop(heap)
        if settled_mask[node]:
            continue
        if distance > distances[node] + tie_tolerance_s or hop_count != int(hops[node]) or key != path_key[node]:
            continue
        if distance > limit + tie_tolerance_s:
            break
        settled_mask[node] = True
        settled.append(node)
        if remaining_targets is not None:
            remaining_targets.discard(node)
            if not remaining_targets:
                break
        if node in prohibited and node not in sources:
            continue
        if hop_count >= hop_limit:
            continue
        for position in range(int(graph.indptr[node]), int(graph.indptr[node + 1])):
            target = int(graph.head_local[position])
            if settled_mask[target]:
                continue
            candidate_distance = distance + float(graph.cost_s[position])
            if candidate_distance > limit + tie_tolerance_s:
                continue
            candidate_hops = hop_count + 1
            candidate_key = key + (_id_key(graph.arc_stable_id[position]),)
            better = candidate_distance < distances[target] - tie_tolerance_s
            tied = abs(candidate_distance - distances[target]) <= tie_tolerance_s
            if better or (
                tied
                and (candidate_hops, candidate_key, graph.node_ids[node])
                < (
                    int(hops[target]),
                    path_key[target] if path_key[target] is not None else ((2, "~"),),
                    graph.node_ids[int(predecessor[target])] if predecessor[target] >= 0 else np.iinfo(np.int64).max,
                )
            ):
                distances[target] = candidate_distance
                hops[target] = candidate_hops
                predecessor[target] = node
                predecessor_arc[target] = position
                path_key[target] = candidate_key
                heapq.heappush(heap, (candidate_distance, candidate_hops, candidate_key, target))
    return RestrictedTree(
        source_local=sources,
        distances_s=distances,
        predecessor_local=predecessor,
        predecessor_arc_position=predecessor_arc,
        settled_order=np.asarray(settled, dtype=np.int64),
    )


def reconstruct_restricted_path(
    graph: SparseDirectedGraph, tree: RestrictedTree, target_global_node: int
) -> tuple[int, ...]:
    global_to_local = {value: index for index, value in enumerate(graph.node_ids)}
    target = global_to_local.get(int(target_global_node))
    if target is None or not np.isfinite(tree.distances_s[target]):
        raise Phase3AShortestPathError("Target is unreachable in restricted tree")
    source_set = set(tree.source_local)
    current = target
    positions: list[int] = []
    seen: set[int] = set()
    while current not in source_set:
        if current in seen:
            raise Phase3AShortestPathError("Cycle in restricted predecessor chain")
        seen.add(current)
        position = int(tree.predecessor_arc_position[current])
        previous = int(tree.predecessor_local[current])
        if position < 0 or previous < 0:
            raise Phase3AShortestPathError("Incomplete restricted predecessor chain")
        positions.append(position)
        current = previous
    positions.reverse()
    return tuple(int(graph.arc_index[position]) for position in positions)


def physical_fftt_tree(
    graph: GraphData, origin_node_idx: int, *, tie_tolerance_s: float = DEFAULT_TIE_TOLERANCE_S
) -> PartCRev1SSSPResult:
    """Run one authoritative physical FFTT tree through the reused solver API."""

    origin = int(origin_node_idx)
    if origin < 0 or origin >= graph.node_count:
        raise Phase3AShortestPathError("Physical origin node index is out of range")
    result = partC_rev1_sssp(
        graph.snapshot,
        origin,
        graph.arc_fftt_s,
        backend="numba_csr",
        tie_tol=float(tie_tolerance_s),
    )
    result.metadata.update(
        {
            "phase3a_wrapper": "physical_fftt_tree",
            "cost_unit": "seconds",
            "tie_tolerance_s": float(tie_tolerance_s),
            "reused_api": "partC_rev1_src.partC_rev1_shortest_paths.partC_rev1_sssp",
        }
    )
    return result


def reconstruct_physical_path(
    graph: GraphData, tree: PartCRev1SSSPResult, destination_node_idx: int
) -> tuple[int, ...]:
    destination = int(destination_node_idx)
    if destination < 0 or destination >= graph.node_count or not tree.reached_mask[destination]:
        raise Phase3AShortestPathError("Physical destination is unreachable")
    source = int(tree.metadata["source_idx"])
    current = destination
    arcs: list[int] = []
    seen: set[int] = set()
    while current != source:
        if current in seen:
            raise Phase3AShortestPathError("Cycle in physical predecessor chain")
        seen.add(current)
        arc = int(tree.predecessor_arc[current])
        previous = int(tree.predecessor_node[current])
        if arc < 0 or previous < 0 or int(graph.head[arc]) != current or int(graph.tail[arc]) != previous:
            raise Phase3AShortestPathError("Invalid physical predecessor chain")
        arcs.append(arc)
        current = previous
    arcs.reverse()
    return tuple(arcs)


def _id_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)
