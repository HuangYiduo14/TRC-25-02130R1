"""Sampled endpoint-state liftability with deterministic witness recovery."""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence

from .models import CoarseEdgeRelation, LiftabilityResult, PhysicalTransition


class Phase3ALiftabilityError(ValueError):
    """Raised for malformed state relations or coarse walks."""


def audit_liftability(
    *,
    coarse_edges: Sequence[CoarseEdgeRelation],
    source_states: Sequence[int],
    destination_states: Sequence[int],
    internal_relations: Mapping[str, Sequence[PhysicalTransition]],
    centroid_coarse_nodes: frozenset[str] = frozenset(),
) -> LiftabilityResult:
    """Propagate sparse feasible physical endpoint states along one coarse walk."""

    if not coarse_edges:
        overlap = sorted(set(map(int, source_states)).intersection(map(int, destination_states)))
        return LiftabilityResult(bool(overlap), None if overlap else 0, "LIFTABLE" if overlap else "FINAL_STATE_INCOMPATIBLE", 0.0 if overlap else None, () if overlap else None, overlap[0] if overlap else None)
    for left, right in zip(coarse_edges[:-1], coarse_edges[1:], strict=True):
        if left.head_coarse_node_id != right.tail_coarse_node_id:
            raise Phase3ALiftabilityError("Coarse edge sequence is discontinuous")
    current: dict[int, tuple[float, tuple[str, ...]]] = {
        int(state): (0.0, ()) for state in sorted(set(map(int, source_states)))
    }
    if not current:
        return LiftabilityResult(False, 0, "NO_SOURCE_STATE", None, None, None)
    for step, edge in enumerate(coarse_edges):
        if not edge.transitions:
            return LiftabilityResult(False, step, "NO_PHYSICAL_BUNDLE_MEMBER", None, None, None)
        next_states: dict[int, tuple[float, tuple[str, ...]]] = {}
        for from_state, prefix in current.items():
            reachable = _internal_reachable(
                edge.tail_coarse_node_id,
                from_state,
                internal_relations,
                centroid_coarse_nodes,
            )
            for transition in edge.transitions:
                internal = reachable.get(int(transition.tail_state))
                if internal is None:
                    continue
                candidate = (
                    prefix[0] + internal[0] + float(transition.fftt_s),
                    prefix[1] + internal[1] + tuple(map(str, transition.arc_ids)),
                )
                head = int(transition.head_state)
                if head not in next_states or candidate < next_states[head]:
                    next_states[head] = candidate
        if not next_states:
            reason = "CENTROID_THROUGH_PROHIBITED" if edge.tail_coarse_node_id in centroid_coarse_nodes else "NO_INTERNAL_STATE_TRANSITION"
            return LiftabilityResult(False, step, reason, None, None, None)
        current = next_states
    final_coarse = coarse_edges[-1].head_coarse_node_id
    destinations = set(map(int, destination_states))
    finals: list[tuple[float, tuple[str, ...], int]] = []
    for state, prefix in current.items():
        reachable = _internal_reachable(final_coarse, state, internal_relations, centroid_coarse_nodes)
        for destination in sorted(destinations):
            suffix = reachable.get(destination)
            if suffix is not None:
                finals.append((prefix[0] + suffix[0], prefix[1] + suffix[1], destination))
    if not finals:
        return LiftabilityResult(False, len(coarse_edges), "FINAL_STATE_INCOMPATIBLE", None, None, None)
    cost, arc_ids, final_state = min(finals)
    return LiftabilityResult(True, None, "LIFTABLE", float(cost), arc_ids, int(final_state))


def _internal_reachable(
    coarse_node_id: str,
    source_state: int,
    internal_relations: Mapping[str, Sequence[PhysicalTransition]],
    centroid_nodes: frozenset[str],
) -> dict[int, tuple[float, tuple[str, ...]]]:
    source = int(source_state)
    if coarse_node_id in centroid_nodes:
        return {source: (0.0, ())}
    adjacency: dict[int, list[PhysicalTransition]] = {}
    for transition in internal_relations.get(coarse_node_id, ()):
        if transition.fftt_s < 0:
            raise Phase3ALiftabilityError("Internal transition has negative FFTT")
        adjacency.setdefault(int(transition.tail_state), []).append(transition)
    for values in adjacency.values():
        values.sort(key=lambda item: (float(item.fftt_s), tuple(map(str, item.arc_ids)), int(item.head_state)))
    best: dict[int, tuple[float, tuple[str, ...]]] = {source: (0.0, ())}
    heap: list[tuple[float, tuple[str, ...], int]] = [(0.0, (), source)]
    while heap:
        cost, arc_ids, state = heapq.heappop(heap)
        if best.get(state) != (cost, arc_ids):
            continue
        for transition in adjacency.get(state, ()):
            target = int(transition.head_state)
            candidate = (cost + float(transition.fftt_s), arc_ids + tuple(map(str, transition.arc_ids)))
            if target not in best or candidate < best[target]:
                best[target] = candidate
                heapq.heappush(heap, (candidate[0], candidate[1], target))
    return best
