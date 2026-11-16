"""Direct, structural, and optional global-scale diagnostic FFTT helpers."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra

from .models import ProjectionResult


class Phase3ADiagnosticError(ValueError):
    """Raised for invalid diagnostic costs or coarse walks."""


@dataclass(frozen=True, slots=True)
class ScalarPath:
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    fftt_s: float


@dataclass(slots=True)
class MaterializedScalarGraph:
    """Array-backed exact candidate graph used for batched OD evaluation."""

    node_ids: tuple[str, ...]
    node_index: dict[str, int]
    edge_ids: tuple[str, ...]
    tail: np.ndarray
    head: np.ndarray
    base_cost_s: np.ndarray
    edge_lex_rank: np.ndarray
    forward_indptr: np.ndarray
    forward_edge_index: np.ndarray
    reverse_indptr: np.ndarray
    reverse_edge_index: np.ndarray

    @property
    def node_count(self) -> int:
        return len(self.node_ids)

    @property
    def edge_count(self) -> int:
        return len(self.edge_ids)


def build_materialized_scalar_graph(
    edges: Sequence[tuple[str, str, str, float]],
) -> MaterializedScalarGraph:
    """Build a parallel-edge-preserving CSR index from frozen coarse edges."""

    if not edges:
        raise Phase3ADiagnosticError("Materialized scalar graph may not be empty")
    edge_ids = tuple(str(value[0]) for value in edges)
    if len(set(edge_ids)) != len(edge_ids):
        raise Phase3ADiagnosticError("Materialized scalar edge IDs must be globally unique")
    node_ids = tuple(sorted({str(value) for edge in edges for value in edge[1:3]}))
    node_index = {value: index for index, value in enumerate(node_ids)}
    tail = np.fromiter((node_index[str(edge[1])] for edge in edges), dtype=np.int64, count=len(edges))
    head = np.fromiter((node_index[str(edge[2])] for edge in edges), dtype=np.int64, count=len(edges))
    costs = np.fromiter((float(edge[3]) for edge in edges), dtype=np.float64, count=len(edges))
    if np.any(~np.isfinite(costs)) or np.any(costs < 0.0):
        raise Phase3ADiagnosticError("Materialized scalar costs must be finite and nonnegative")
    lex_order = sorted(range(len(edge_ids)), key=edge_ids.__getitem__)
    lex_rank = np.empty(len(edge_ids), dtype=np.int64)
    lex_rank[np.asarray(lex_order, dtype=np.int64)] = np.arange(len(edge_ids), dtype=np.int64)
    forward_order = np.argsort(tail, kind="stable")
    reverse_order = np.argsort(head, kind="stable")
    forward_indptr = _indptr_from_sorted_tail(tail[forward_order], len(node_ids))
    reverse_indptr = _indptr_from_sorted_tail(head[reverse_order], len(node_ids))
    return MaterializedScalarGraph(
        node_ids=node_ids,
        node_index=node_index,
        edge_ids=edge_ids,
        tail=tail,
        head=head,
        base_cost_s=costs,
        edge_lex_rank=lex_rank,
        forward_indptr=forward_indptr,
        forward_edge_index=forward_order.astype(np.int64, copy=False),
        reverse_indptr=reverse_indptr,
        reverse_edge_index=reverse_order.astype(np.int64, copy=False),
    )


def materialized_scalar_shortest_paths(
    graph: MaterializedScalarGraph,
    queries: Sequence[tuple[Hashable, str, str]],
    *,
    endpoint_charge_s: Mapping[str, float] | None = None,
    centroid_non_through_nodes: frozenset[str] = frozenset(),
    tie_tolerance_s: float = 1.0e-10,
    target_batch_size: int = 8,
) -> dict[Hashable, ScalarPath | None]:
    """Solve many queries by reverse-distance batches and exact lexicographic reconstruction.

    Distances are computed over the complete materialized sparse graph.  For
    each OD, reconstruction greedily chooses the smallest stable edge ID that
    can still attain the frozen shortest distance.  This is equivalent to the
    tuple-key Dijkstra tie rule while avoiding one Python heap search per OD.
    """

    if not queries:
        raise Phase3ADiagnosticError("Materialized scalar evaluation requires nonempty queries")
    if int(target_batch_size) < 1:
        raise Phase3ADiagnosticError("Target batch size must be positive")
    charge = endpoint_charge_s or {}
    node_charge = np.fromiter(
        (float(charge.get(node_id, 0.0)) for node_id in graph.node_ids),
        dtype=np.float64,
        count=graph.node_count,
    )
    if np.any(~np.isfinite(node_charge)) or np.any(node_charge < 0.0):
        raise Phase3ADiagnosticError("Materialized scalar endpoint charges must be finite and nonnegative")
    edge_cost = graph.base_cost_s + 0.5 * node_charge[graph.tail] + 0.5 * node_charge[graph.head]
    query_rows: list[tuple[Hashable, int, int]] = []
    for query_id, source_id, target_id in queries:
        source = graph.node_index.get(str(source_id))
        target = graph.node_index.get(str(target_id))
        if source is None or target is None:
            raise Phase3ADiagnosticError("Materialized scalar query endpoint is absent from the graph")
        query_rows.append((query_id, int(source), int(target)))
    centroid_indexes = {
        graph.node_index[value]
        for value in centroid_non_through_nodes
        if value in graph.node_index
    }
    query_source_indexes = {source for _, source, _ in query_rows}
    permitted_centroid_sources = centroid_indexes.intersection(query_source_indexes)
    for source in permitted_centroid_sources:
        incoming = np.flatnonzero(graph.head == int(source))
        if incoming.size and np.any(graph.tail[incoming] != int(source)):
            raise Phase3ADiagnosticError(
                "A permitted centroid query source has an incoming edge and could become a through node"
            )
    active_edge = np.fromiter(
        (
            int(tail) not in centroid_indexes or int(tail) in permitted_centroid_sources
            for tail in graph.tail
        ),
        dtype=bool,
        count=graph.edge_count,
    )
    by_target: dict[int, list[tuple[Hashable, int]]] = {}
    for query_id, source, target in query_rows:
        by_target.setdefault(target, []).append((query_id, source))
    target_indexes = sorted(by_target)
    result: dict[Hashable, ScalarPath | None] = {}
    active_reverse_order = graph.reverse_edge_index[active_edge[graph.reverse_edge_index]]
    active_reverse_indptr = _indptr_from_sorted_tail(
        graph.head[active_reverse_order], graph.node_count
    )
    reverse_graph = csr_matrix(
        (
            edge_cost[active_reverse_order],
            graph.tail[active_reverse_order],
            active_reverse_indptr,
        ),
        shape=(graph.node_count, graph.node_count),
    )
    for begin in range(0, len(target_indexes), int(target_batch_size)):
        batch = target_indexes[begin : begin + int(target_batch_size)]
        distance_rows = sparse_dijkstra(
            reverse_graph,
            directed=True,
            indices=np.asarray(batch, dtype=np.int64),
            return_predecessors=False,
        )
        if np.asarray(distance_rows).ndim == 1:
            distance_rows = np.asarray(distance_rows, dtype=np.float64)[None, :]
        for row_position, target in enumerate(batch):
            distances = np.asarray(distance_rows[row_position], dtype=np.float64)
            for query_id, source in by_target[target]:
                result[query_id] = _lex_path_from_reverse_distances(
                    graph,
                    source,
                    target,
                    edge_cost,
                    distances,
                    centroid_non_through_nodes=centroid_non_through_nodes,
                    tie_tolerance_s=float(tie_tolerance_s),
                )
    if len(result) != len(queries):
        raise Phase3ADiagnosticError("Materialized scalar query result count is incomplete")
    return result


def _lex_path_from_reverse_distances(
    graph: MaterializedScalarGraph,
    source: int,
    target: int,
    edge_cost_s: np.ndarray,
    reverse_distance_s: np.ndarray,
    *,
    centroid_non_through_nodes: frozenset[str],
    tie_tolerance_s: float,
) -> ScalarPath | None:
    if not np.isfinite(reverse_distance_s[source]):
        return None
    greedy = _greedy_lex_path_from_reverse_distances(
        graph,
        source,
        target,
        edge_cost_s,
        reverse_distance_s,
        centroid_non_through_nodes=centroid_non_through_nodes,
        tie_tolerance_s=tie_tolerance_s,
    )
    if greedy is not None:
        return greedy
    return _dfs_lex_path_from_reverse_distances(
        graph,
        source,
        target,
        edge_cost_s,
        reverse_distance_s,
        centroid_non_through_nodes=centroid_non_through_nodes,
        tie_tolerance_s=tie_tolerance_s,
    )


def _greedy_lex_path_from_reverse_distances(
    graph: MaterializedScalarGraph,
    source: int,
    target: int,
    edge_cost_s: np.ndarray,
    reverse_distance_s: np.ndarray,
    *,
    centroid_non_through_nodes: frozenset[str],
    tie_tolerance_s: float,
) -> ScalarPath | None:
    node_indexes = [int(source)]
    edge_indexes: list[int] = []
    visited = {int(source)}
    current = int(source)
    path_cost = 0.0
    while current != int(target):
        if graph.node_ids[current] in centroid_non_through_nodes and current != int(source):
            return None
        candidates: list[int] = []
        for position in range(int(graph.forward_indptr[current]), int(graph.forward_indptr[current + 1])):
            edge = int(graph.forward_edge_index[position])
            neighbor = int(graph.head[edge])
            if neighbor in visited or not np.isfinite(reverse_distance_s[neighbor]):
                continue
            left = float(edge_cost_s[edge]) + float(reverse_distance_s[neighbor])
            right = float(reverse_distance_s[current])
            roundoff_guard = 16.0 * np.finfo(np.float64).eps * max(1.0, abs(left), abs(right))
            if abs(left - right) <= float(tie_tolerance_s) + roundoff_guard:
                candidates.append(edge)
        if not candidates:
            return None
        selected = min(candidates, key=lambda edge: int(graph.edge_lex_rank[edge]))
        current = int(graph.head[selected])
        edge_indexes.append(selected)
        node_indexes.append(current)
        visited.add(current)
        path_cost += float(edge_cost_s[selected])
        if len(node_indexes) > graph.node_count:
            raise Phase3ADiagnosticError("Materialized scalar path contains a cycle")
    return ScalarPath(
        tuple(graph.node_ids[index] for index in node_indexes),
        tuple(graph.edge_ids[index] for index in edge_indexes),
        float(path_cost),
    )


def _dfs_lex_path_from_reverse_distances(
    graph: MaterializedScalarGraph,
    source: int,
    target: int,
    edge_cost_s: np.ndarray,
    reverse_distance_s: np.ndarray,
    *,
    centroid_non_through_nodes: frozenset[str],
    tie_tolerance_s: float,
) -> ScalarPath:
    """Find the first simple tight path in global edge-lexicographic DFS order."""

    def tight_outgoing(node: int) -> list[int]:
        if graph.node_ids[node] in centroid_non_through_nodes and node != int(source):
            return []
        values: list[int] = []
        for position in range(int(graph.forward_indptr[node]), int(graph.forward_indptr[node + 1])):
            edge = int(graph.forward_edge_index[position])
            neighbor = int(graph.head[edge])
            if not np.isfinite(reverse_distance_s[neighbor]):
                continue
            left = float(edge_cost_s[edge]) + float(reverse_distance_s[neighbor])
            right = float(reverse_distance_s[node])
            roundoff_guard = 16.0 * np.finfo(np.float64).eps * max(1.0, abs(left), abs(right))
            if abs(left - right) <= float(tie_tolerance_s) + roundoff_guard:
                values.append(edge)
        values.sort(key=lambda edge: int(graph.edge_lex_rank[edge]))
        return values

    seen = {int(source)}
    node_stack = [int(source)]
    edge_stack: list[int] = []
    option_stack = [iter(tight_outgoing(int(source)))]
    while node_stack:
        current = node_stack[-1]
        if current == int(target):
            path_cost = float(sum(float(edge_cost_s[edge]) for edge in edge_stack))
            return ScalarPath(
                tuple(graph.node_ids[index] for index in node_stack),
                tuple(graph.edge_ids[index] for index in edge_stack),
                path_cost,
            )
        try:
            edge = next(option_stack[-1])
        except StopIteration:
            node_stack.pop()
            option_stack.pop()
            if edge_stack:
                edge_stack.pop()
            continue
        neighbor = int(graph.head[edge])
        if neighbor in seen:
            continue
        seen.add(neighbor)
        edge_stack.append(edge)
        node_stack.append(neighbor)
        option_stack.append(iter(tight_outgoing(neighbor)))
    raise Phase3ADiagnosticError(
        "Shortest-distance labels cannot be reconstructed as a simple lexicographic path"
    )


def _indptr_from_sorted_tail(sorted_tail: np.ndarray, node_count: int) -> np.ndarray:
    counts = np.bincount(np.asarray(sorted_tail, dtype=np.int64), minlength=int(node_count))
    indptr = np.empty(int(node_count) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return indptr


@dataclass(frozen=True, slots=True)
class _ScalarPathLabel:
    """Persistent node path with one shared lexicographic edge key."""

    node_id: str
    parent: _ScalarPathLabel | None
    edge_id: str | None
    edge_key: tuple[str, ...]


def _label_edge_ids(label: _ScalarPathLabel) -> tuple[str, ...]:
    return label.edge_key


def _label_node_ids(label: _ScalarPathLabel) -> tuple[str, ...]:
    values: list[str] = []
    current: _ScalarPathLabel | None = label
    while current is not None:
        values.append(current.node_id)
        current = current.parent
    values.reverse()
    return tuple(values)


def projected_generic_cost(
    projection: ProjectionResult,
    object_cost_s: Mapping[str, float],
    *,
    internal_online_cost_s: float = 0.0,
) -> float:
    """Price projection occurrences with generic coarse costs, never witness cost."""

    total = 0.0
    for occurrence in projection.occurrences:
        if occurrence.omitted_online:
            total += float(internal_online_cost_s)
        else:
            if occurrence.object_id not in object_cost_s:
                raise Phase3ADiagnosticError(f"Missing generic cost for {occurrence.object_id}")
            value = float(object_cost_s[occurrence.object_id])
            if not np.isfinite(value) or value < 0.0:
                raise Phase3ADiagnosticError("Generic coarse costs must be finite and nonnegative")
            total += value
    return float(total)


def structural_half_charge_adjustments(
    coarse_node_sequence: Sequence[str], structural_charge_s: Mapping[str, float]
) -> np.ndarray:
    """Allocate one charge per through visit, half to each incident edge."""

    nodes = tuple(map(str, coarse_node_sequence))
    if len(nodes) < 2:
        return np.empty(0, dtype=np.float64)
    result = np.zeros(len(nodes) - 1, dtype=np.float64)
    for position in range(1, len(nodes) - 1):
        charge = float(structural_charge_s.get(nodes[position], 0.0))
        if not np.isfinite(charge) or charge < 0.0:
            raise Phase3ADiagnosticError("Structural charges must be finite and nonnegative")
        result[position - 1] += 0.5 * charge
        result[position] += 0.5 * charge
    return result


def median_structural_charge(feasible_directed_internal_fftt_s: Sequence[float]) -> float | None:
    values = np.asarray(feasible_directed_internal_fftt_s, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    if np.any(values < 0.0):
        raise Phase3ADiagnosticError("Internal FFTTs must be nonnegative")
    return float(np.median(values))


def fit_global_nonnegative_scale(
    *, residual_s: np.ndarray, x_supp: np.ndarray, x_aug: np.ndarray
) -> dict[str, float]:
    y = np.asarray(residual_s, dtype=np.float64)
    design = np.column_stack((np.asarray(x_supp, dtype=np.float64), np.asarray(x_aug, dtype=np.float64)))
    if y.ndim != 1 or design.shape != (y.size, 2) or y.size == 0:
        raise Phase3ADiagnosticError("Global scale fit requires aligned nonempty vectors")
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(design)) or np.any(design < 0.0):
        raise Phase3ADiagnosticError("Global scale fit contains invalid values")
    coefficients, residual_norm = nnls(design, y)
    return {
        "theta_supp": float(coefficients[0]),
        "theta_aug": float(coefficients[1]),
        "residual_norm": float(residual_norm),
        "nonnegative": True,
    }


def scalar_shortest_path(
    *,
    source_node_id: str,
    target_node_id: str,
    neighbor_provider: Callable[[str], Sequence[tuple[str, str, float]]],
    centroid_non_through_nodes: frozenset[str] = frozenset(),
    endpoint_charge_s: Mapping[str, float] | None = None,
    tie_tolerance_s: float = 1.0e-10,
) -> ScalarPath | None:
    source, target = str(source_node_id), str(target_node_id)
    charge = endpoint_charge_s or {}
    source_label = _ScalarPathLabel(source, None, None, ())
    best: dict[str, tuple[float, _ScalarPathLabel]] = {source: (0.0, source_label)}
    heap: list[tuple[float, tuple[str, ...], str, int, _ScalarPathLabel]] = [
        (0.0, (), source, 0, source_label)
    ]
    serial = 0
    while heap:
        distance, edge_key, node, _, label = heapq.heappop(heap)
        current = best.get(node)
        if (
            current is None
            or label is not current[1]
            or edge_key != label.edge_key
            or abs(distance - current[0]) > tie_tolerance_s
        ):
            continue
        if node == target:
            return ScalarPath(_label_node_ids(label), label.edge_key, float(current[0]))
        if node in centroid_non_through_nodes and node != source:
            continue
        neighbors = sorted(neighbor_provider(node), key=lambda item: (str(item[1]), str(item[0])))
        for edge_id, neighbor, base_cost in neighbors:
            value = float(base_cost)
            if not np.isfinite(value) or value < 0.0:
                raise Phase3ADiagnosticError("Scalar ACNHN edge cost is invalid")
            endpoint = 0.5 * float(charge.get(node, 0.0)) + 0.5 * float(charge.get(str(neighbor), 0.0))
            candidate_distance = distance + value + endpoint
            neighbor_id = str(neighbor)
            edge_id = str(edge_id)
            old = best.get(neighbor_id)
            replace = old is None or candidate_distance < old[0] - tie_tolerance_s
            candidate_edges: tuple[str, ...] | None = None
            if not replace and old is not None and abs(candidate_distance - old[0]) <= tie_tolerance_s:
                candidate_edges = label.edge_key + (edge_id,)
                replace = candidate_edges < old[1].edge_key
            if replace:
                if candidate_edges is None:
                    candidate_edges = label.edge_key + (edge_id,)
                candidate_label = _ScalarPathLabel(neighbor_id, label, edge_id, candidate_edges)
                best[neighbor_id] = (candidate_distance, candidate_label)
                serial += 1
                heapq.heappush(
                    heap,
                    (candidate_distance, candidate_edges, neighbor_id, serial, candidate_label),
                )
    return None
