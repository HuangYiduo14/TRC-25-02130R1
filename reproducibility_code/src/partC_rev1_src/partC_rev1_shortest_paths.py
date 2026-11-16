"""Shortest-path backends for Prompt 4 physical assignment kernels."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numba import njit

from .partC_rev1_graph_snapshot import PartCRev1GraphSnapshot
from .partC_rev1_hashing import partC_rev1_numpy_array_hash, partC_rev1_sort_key
from .partC_rev1_runtime_guard import partC_rev1_project_runtime


PARTC_REV1_INF = np.inf


@dataclass(slots=True)
class PartCRev1SSSPResult:
    distances: np.ndarray
    predecessor_node: np.ndarray
    predecessor_arc: np.ndarray
    settled_order: np.ndarray
    reached_mask: np.ndarray
    metadata: dict[str, Any]


@dataclass(slots=True)
class PartCRev1ManyToManyResult:
    distance: np.ndarray
    reachable: np.ndarray
    graph_hash: str
    cost_hash: str
    batch_timings: list[dict[str, Any]]
    metadata: dict[str, Any]


def partC_rev1_arc_tie_rank(graph: PartCRev1GraphSnapshot) -> np.ndarray:
    keys = [partC_rev1_sort_key(value) for value in graph.arc_idx_to_directed_arc_id.tolist()]
    order = sorted(range(len(keys)), key=lambda idx: keys[idx])
    rank = np.empty(len(keys), dtype=np.int64)
    for position, arc_idx in enumerate(order):
        rank[arc_idx] = position
    return rank


@njit(cache=True)
def _partC_rev1_heap_swap(heap_nodes: np.ndarray, heap_dist: np.ndarray, left: int, right: int) -> None:
    node = heap_nodes[left]
    dist = heap_dist[left]
    heap_nodes[left] = heap_nodes[right]
    heap_dist[left] = heap_dist[right]
    heap_nodes[right] = node
    heap_dist[right] = dist


@njit(cache=True)
def _partC_rev1_heap_push(heap_nodes: np.ndarray, heap_dist: np.ndarray, size: int, node: int, dist: float) -> int:
    idx = size
    heap_nodes[idx] = node
    heap_dist[idx] = dist
    size += 1
    while idx > 0:
        parent = (idx - 1) // 2
        if heap_dist[parent] <= heap_dist[idx]:
            break
        _partC_rev1_heap_swap(heap_nodes, heap_dist, parent, idx)
        idx = parent
    return size


@njit(cache=True)
def _partC_rev1_heap_pop(heap_nodes: np.ndarray, heap_dist: np.ndarray, size: int) -> tuple[int, float, int]:
    node = heap_nodes[0]
    dist = heap_dist[0]
    size -= 1
    heap_nodes[0] = heap_nodes[size]
    heap_dist[0] = heap_dist[size]
    idx = 0
    while True:
        left = 2 * idx + 1
        right = left + 1
        if left >= size:
            break
        smallest = left
        if right < size and heap_dist[right] < heap_dist[left]:
            smallest = right
        if heap_dist[idx] <= heap_dist[smallest]:
            break
        _partC_rev1_heap_swap(heap_nodes, heap_dist, idx, smallest)
        idx = smallest
    return node, dist, size


@njit(cache=True)
def _partC_rev1_numba_dijkstra(
    n_nodes: int,
    indptr: np.ndarray,
    csr_arc_idx: np.ndarray,
    head: np.ndarray,
    cost: np.ndarray,
    arc_tie_rank: np.ndarray,
    source: int,
    tie_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    dist = np.empty(n_nodes, dtype=np.float64)
    pred_node = np.empty(n_nodes, dtype=np.int64)
    pred_arc = np.empty(n_nodes, dtype=np.int64)
    settled = np.zeros(n_nodes, dtype=np.uint8)
    settled_order = np.empty(n_nodes, dtype=np.int64)
    for idx in range(n_nodes):
        dist[idx] = np.inf
        pred_node[idx] = -1
        pred_arc[idx] = -1
    heap_capacity = max(2 * (len(csr_arc_idx) + 1), n_nodes + 1)
    heap_nodes = np.empty(heap_capacity, dtype=np.int64)
    heap_dist = np.empty(heap_capacity, dtype=np.float64)
    heap_size = 0
    settled_count = 0
    dist[source] = 0.0
    heap_size = _partC_rev1_heap_push(heap_nodes, heap_dist, heap_size, source, 0.0)
    while heap_size > 0:
        u, du, heap_size = _partC_rev1_heap_pop(heap_nodes, heap_dist, heap_size)
        if settled[u] != 0:
            continue
        if abs(du - dist[u]) > tie_tol:
            continue
        settled[u] = 1
        settled_order[settled_count] = u
        settled_count += 1
        for pos in range(indptr[u], indptr[u + 1]):
            arc_idx = csr_arc_idx[pos]
            v = head[arc_idx]
            nd = du + cost[arc_idx]
            if nd < dist[v] - tie_tol:
                dist[v] = nd
                pred_node[v] = u
                pred_arc[v] = arc_idx
                heap_size = _partC_rev1_heap_push(heap_nodes, heap_dist, heap_size, v, nd)
            elif abs(nd - dist[v]) <= tie_tol:
                old_arc = pred_arc[v]
                if old_arc < 0 or arc_tie_rank[arc_idx] < arc_tie_rank[old_arc]:
                    pred_node[v] = u
                    pred_arc[v] = arc_idx
                    heap_size = _partC_rev1_heap_push(heap_nodes, heap_dist, heap_size, v, nd)
    return dist, pred_node, pred_arc, settled_order, settled_count


def _partC_rev1_numba_sssp(
    graph: PartCRev1GraphSnapshot,
    source_idx: int,
    cost_array: np.ndarray,
    *,
    tie_tol: float,
) -> PartCRev1SSSPResult:
    started = time.perf_counter()
    ranks = partC_rev1_arc_tie_rank(graph)
    dist, pred_node, pred_arc, settled_order, settled_count = _partC_rev1_numba_dijkstra(
        len(graph.node_idx_to_uid),
        graph.csr_indptr.astype(np.int64),
        graph.csr_arc_idx.astype(np.int64),
        graph.head.astype(np.int64),
        np.asarray(cost_array, dtype=np.float64),
        ranks,
        int(source_idx),
        float(tie_tol),
    )
    settled = settled_order[:settled_count].copy()
    reached = np.isfinite(dist)
    return PartCRev1SSSPResult(
        distances=dist,
        predecessor_node=pred_node,
        predecessor_arc=pred_arc,
        settled_order=settled,
        reached_mask=reached,
        metadata={
            "backend": "numba_csr",
            "source_idx": int(source_idx),
            "settled_count": int(settled_count),
            "runtime_seconds": time.perf_counter() - started,
        },
    )


def _partC_rev1_networkx_sssp(
    graph: PartCRev1GraphSnapshot,
    source_idx: int,
    cost_array: np.ndarray,
    *,
    tie_tol: float,
) -> PartCRev1SSSPResult:
    import networkx as nx

    started = time.perf_counter()
    g = nx.MultiDiGraph()
    g.add_nodes_from(range(len(graph.node_idx_to_uid)))
    for arc_idx, (tail, head) in enumerate(zip(graph.tail.tolist(), graph.head.tolist(), strict=True)):
        g.add_edge(int(tail), int(head), key=int(arc_idx), weight=float(cost_array[arc_idx]), arc_idx=int(arc_idx))
    ranks = partC_rev1_arc_tie_rank(graph)
    n = len(graph.node_idx_to_uid)
    dist = np.full(n, np.inf, dtype=np.float64)
    pred_node = np.full(n, -1, dtype=np.int64)
    pred_arc = np.full(n, -1, dtype=np.int64)
    settled = np.zeros(n, dtype=bool)
    settled_order: list[int] = []
    heap: list[tuple[float, int]] = [(0.0, int(source_idx))]
    dist[int(source_idx)] = 0.0
    while heap:
        du, u = heapq.heappop(heap)
        if settled[u] or abs(du - dist[u]) > tie_tol:
            continue
        settled[u] = True
        settled_order.append(u)
        for _, v, key, data in g.out_edges(u, keys=True, data=True):
            arc_idx = int(data["arc_idx"])
            nd = du + float(data["weight"])
            if nd < dist[v] - tie_tol:
                dist[v] = nd
                pred_node[v] = u
                pred_arc[v] = arc_idx
                heapq.heappush(heap, (nd, int(v)))
            elif abs(nd - dist[v]) <= tie_tol:
                old_arc = pred_arc[v]
                if old_arc < 0 or ranks[arc_idx] < ranks[old_arc]:
                    pred_node[v] = u
                    pred_arc[v] = arc_idx
                    heapq.heappush(heap, (nd, int(v)))
    return PartCRev1SSSPResult(
        distances=dist,
        predecessor_node=pred_node,
        predecessor_arc=pred_arc,
        settled_order=np.asarray(settled_order, dtype=np.int64),
        reached_mask=np.isfinite(dist),
        metadata={
            "backend": "networkx_python",
            "source_idx": int(source_idx),
            "settled_count": len(settled_order),
            "runtime_seconds": time.perf_counter() - started,
        },
    )


def _partC_rev1_igraph_sssp(
    graph: PartCRev1GraphSnapshot,
    source_idx: int,
    cost_array: np.ndarray,
    *,
    targets: Iterable[int] | None,
) -> PartCRev1SSSPResult:
    import igraph as ig

    started = time.perf_counter()
    edges = list(zip(graph.tail.astype(int).tolist(), graph.head.astype(int).tolist(), strict=True))
    g = ig.Graph(n=len(graph.node_idx_to_uid), edges=edges, directed=True)
    weights = np.asarray(cost_array, dtype=np.float64).tolist()
    dist_row = g.distances(source=[int(source_idx)], weights=weights, mode="OUT")[0]
    distances = np.asarray(dist_row, dtype=np.float64)
    predecessor_node = np.full(len(graph.node_idx_to_uid), -1, dtype=np.int64)
    predecessor_arc = np.full(len(graph.node_idx_to_uid), -1, dtype=np.int64)
    if targets is None:
        requested = np.flatnonzero(np.isfinite(distances)).astype(int).tolist()
    else:
        requested = [int(value) for value in targets if np.isfinite(distances[int(value)])]
    if requested:
        paths = g.get_shortest_paths(int(source_idx), to=requested, weights=weights, output="epath")
        for target, edge_path in zip(requested, paths, strict=True):
            current = int(source_idx)
            for edge_id in edge_path:
                edge = g.es[int(edge_id)]
                head = int(edge.target)
                predecessor_node[head] = current
                predecessor_arc[head] = int(edge_id)
                current = head
            if target == source_idx:
                predecessor_node[target] = -1
                predecessor_arc[target] = -1
    settled_order = np.argsort(np.where(np.isfinite(distances), distances, np.inf), kind="mergesort")
    settled_order = settled_order[np.isfinite(distances[settled_order])].astype(np.int64)
    return PartCRev1SSSPResult(
        distances=distances,
        predecessor_node=predecessor_node,
        predecessor_arc=predecessor_arc,
        settled_order=settled_order,
        reached_mask=np.isfinite(distances),
        metadata={
            "backend": "igraph",
            "source_idx": int(source_idx),
            "settled_count": int(len(settled_order)),
            "runtime_seconds": time.perf_counter() - started,
        },
    )


def partC_rev1_sssp(
    graph: PartCRev1GraphSnapshot,
    source_idx: int,
    cost_array: np.ndarray,
    *,
    backend: str = "numba_csr",
    targets: Iterable[int] | None = None,
    return_predecessors: bool = True,
    tie_tol: float = 1e-12,
) -> PartCRev1SSSPResult:
    del return_predecessors
    cost = np.asarray(cost_array, dtype=np.float64)
    if cost.shape != graph.free_time.shape:
        raise ValueError("cost_array must have one value per graph arc")
    if np.any(~np.isfinite(cost)) or np.any(cost < 0.0):
        raise ValueError("cost_array must contain finite nonnegative weights")
    if backend == "numba_csr":
        return _partC_rev1_numba_sssp(graph, source_idx, cost, tie_tol=tie_tol)
    if backend == "igraph":
        return _partC_rev1_igraph_sssp(graph, source_idx, cost, targets=targets)
    if backend == "networkx_python":
        return _partC_rev1_networkx_sssp(graph, source_idx, cost, tie_tol=tie_tol)
    raise ValueError(f"Unsupported shortest-path backend: {backend}")


def partC_rev1_many_to_many(
    graph: PartCRev1GraphSnapshot,
    terminals: Iterable[int],
    current_cost: np.ndarray,
    *,
    backend: str = "numba_csr",
    source_batch_size: int | None = None,
    return_predecessors: bool = False,
) -> PartCRev1ManyToManyResult:
    terminal_array = np.asarray(list(terminals), dtype=np.int64)
    if terminal_array.ndim != 1:
        raise ValueError("terminals must be one-dimensional")
    q = len(terminal_array)
    distance = np.full((q, q), np.inf, dtype=np.float64)
    reachable = np.zeros((q, q), dtype=bool)
    timings: list[dict[str, Any]] = []
    started = time.perf_counter()
    batch_size = int(source_batch_size or max(1, min(16, q)))
    unit_seconds: list[float] = []
    for batch_start in range(0, q, batch_size):
        batch_end = min(q, batch_start + batch_size)
        batch_started = time.perf_counter()
        for row in range(batch_start, batch_end):
            source_started = time.perf_counter()
            result = partC_rev1_sssp(
                graph,
                int(terminal_array[row]),
                current_cost,
                backend=backend,
                targets=terminal_array.tolist(),
                return_predecessors=return_predecessors,
            )
            source_elapsed = time.perf_counter() - source_started
            unit_seconds.append(source_elapsed)
            distance[row, :] = result.distances[terminal_array]
            reachable[row, :] = result.reached_mask[terminal_array]
        projection = partC_rev1_project_runtime(
            completed=batch_end,
            total=q,
            started_perf=started,
            completed_unit_seconds=unit_seconds,
        )
        timings.append(
            {
                "batch_start": batch_start,
                "batch_end": batch_end,
                "runtime_seconds": time.perf_counter() - batch_started,
                "projected_total_seconds": projection.projected_total_seconds,
            }
        )
    return PartCRev1ManyToManyResult(
        distance=distance,
        reachable=reachable,
        graph_hash=graph.graph_hash,
        cost_hash=partC_rev1_numpy_array_hash(np.asarray(current_cost, dtype=np.float64)),
        batch_timings=timings,
        metadata={
            "backend": backend,
            "terminal_count": q,
            "runtime_seconds": time.perf_counter() - started,
        },
    )
