"""Bounded directed nonbackbone AUG direct-witness census."""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Iterable, Mapping

import numpy as np

from .geodesy import grs80_distance_m
from .models import AugPolicy, AugWitness, AugWitnessSearchStats, GraphData, SuiteData
from .schemas import stable_object_id


class Phase3AAugWitnessError(ValueError):
    pass


def direct_witnesses_from_source(
    graph: GraphData,
    suite: SuiteData,
    source_node_idx: int,
    policy: AugPolicy,
) -> tuple[AugWitness, ...]:
    """Enumerate canonical direct witnesses from one final-backbone source.

    Positive FFTT implies that a canonical useful witness is simple, so cycle
    states are safely rejected.  The search never expands through a reached
    final-backbone node, including one that later fails radius/detour policy.
    """

    witnesses, _ = direct_witnesses_from_source_with_stats(graph, suite, source_node_idx, policy)
    return witnesses


def direct_witnesses_from_source_with_stats(
    graph: GraphData,
    suite: SuiteData,
    source_node_idx: int,
    policy: AugPolicy,
) -> tuple[tuple[AugWitness, ...], AugWitnessSearchStats]:
    """Return witnesses and counters without scanning the backbone target set.

    The directed bounded search is performed first.  A reached final-backbone
    node is an unconditional terminal state; only then is its exact GRS80
    source-to-target distance evaluated.  This preserves the frozen science
    while removing the former O(number-of-backbone-nodes) scan per source.
    """

    source = int(source_node_idx)
    if source < 0 or source >= graph.node_count or not suite.is_final_backbone_node[source]:
        raise Phase3AAugWitnessError("AUG source must be a valid final-backbone node")
    if policy.max_physical_arcs <= 0 or policy.threshold_s <= 0.0 or policy.search_radius_m <= 0.0:
        raise Phase3AAugWitnessError("AUG policy bounds must be positive")
    indptr = graph.snapshot.csr_indptr
    csr_arc_idx = graph.snapshot.csr_arc_idx
    # heap key follows the frozen canonical witness order before node/path data.
    heap: list[tuple[float, float, int, tuple[tuple[int, int | str], ...], int, tuple[int, ...], tuple[int, ...]]] = [
        (0.0, 0.0, 0, (), source, (), (source,))
    ]
    best_state: dict[tuple[int, int], tuple[float, float, tuple[tuple[int, int | str], ...]]] = {
        (source, 0): (0.0, 0.0, ())
    }
    best_target: dict[int, AugWitness] = {}
    states_settled = 0
    arcs_scanned = 0
    reached_backbone = 0
    rejected_radius = 0
    rejected_detour = 0
    while heap:
        fftt, length, hops, arc_key, node, path_arcs, path_nodes = heapq.heappop(heap)
        recorded = best_state.get((node, hops))
        if recorded != (fftt, length, arc_key):
            continue
        states_settled += 1
        if node != source and suite.is_final_backbone_node[node]:
            reached_backbone += 1
            geodesic = float(
                grs80_distance_m(
                    float(graph.lon[source]),
                    float(graph.lat[source]),
                    float(graph.lon[node]),
                    float(graph.lat[node]),
                )
            )
            if geodesic <= policy.search_radius_m + 1e-9:
                detour = length / max(geodesic, policy.epsilon_distance_m)
                if detour <= policy.max_length_geodesic_detour + 1e-12:
                    witness = _make_witness(graph, suite, source, node, path_arcs, fftt, length, detour, policy)
                    existing = best_target.get(node)
                    if existing is None or witness_rank(witness) < witness_rank(existing):
                        best_target[node] = witness
                else:
                    rejected_detour += 1
            else:
                rejected_radius += 1
            # First-other-backbone stopping is unconditional.
            continue
        if hops >= policy.max_physical_arcs or fftt >= policy.threshold_s:
            continue
        for position in range(int(indptr[node]), int(indptr[node + 1])):
            arcs_scanned += 1
            arc_index = int(csr_arc_idx[position])
            if suite.is_final_backbone_arc[arc_index]:
                continue
            next_node = int(graph.head[arc_index])
            if next_node in path_nodes:
                continue
            next_fftt = fftt + float(graph.arc_fftt_s[arc_index])
            next_length = length + float(graph.arc_length_m[arc_index])
            next_hops = hops + 1
            if next_fftt > policy.threshold_s + 1e-12 or next_hops > policy.max_physical_arcs:
                continue
            next_arc_key = arc_key + (_id_key(str(graph.arc_id[arc_index])),)
            next_rank = (next_fftt, next_length, next_arc_key)
            state = (next_node, next_hops)
            if state in best_state and best_state[state] <= next_rank:
                continue
            best_state[state] = next_rank
            heapq.heappush(
                heap,
                (
                    next_fftt,
                    next_length,
                    next_hops,
                    next_arc_key,
                    next_node,
                    path_arcs + (arc_index,),
                    path_nodes + (next_node,),
                ),
            )
    result = tuple(sorted(best_target.values(), key=lambda item: (_id_key(item.target_node_uid), witness_rank(item))))
    stats = AugWitnessSearchStats(
        source_node_idx=source,
        states_settled=states_settled,
        arcs_scanned=arcs_scanned,
        reached_other_backbone_count=reached_backbone,
        accepted_witness_count=len(result),
        rejected_radius_count=rejected_radius,
        rejected_detour_count=rejected_detour,
    )
    return result, stats


def witness_census(
    graph: GraphData,
    suite: SuiteData,
    source_node_indexes: Iterable[int],
    policy: AugPolicy,
) -> tuple[AugWitness, ...]:
    witnesses: list[AugWitness] = []
    for source in sorted(set(int(value) for value in source_node_indexes), key=lambda index: _id_key(str(graph.node_uid[index]))):
        witnesses.extend(direct_witnesses_from_source(graph, suite, source, policy))
    witnesses.sort(key=lambda item: (_id_key(item.source_node_uid), _id_key(item.target_node_uid), witness_rank(item)))
    return tuple(witnesses)


def witness_census_with_stats(
    graph: GraphData,
    suite: SuiteData,
    source_node_indexes: Iterable[int],
    policy: AugPolicy,
) -> tuple[tuple[AugWitness, ...], tuple[AugWitnessSearchStats, ...]]:
    """Census an explicit deterministic source shard and retain counters."""

    witnesses: list[AugWitness] = []
    stats: list[AugWitnessSearchStats] = []
    ordered = sorted(
        set(int(value) for value in source_node_indexes),
        key=lambda index: _id_key(str(graph.node_uid[index])),
    )
    for source in ordered:
        found, counters = direct_witnesses_from_source_with_stats(graph, suite, source, policy)
        witnesses.extend(found)
        stats.append(counters)
    witnesses.sort(key=lambda item: (_id_key(item.source_node_uid), _id_key(item.target_node_uid), witness_rank(item)))
    return tuple(witnesses), tuple(stats)


def mutual_direct_witness_pairs(
    witnesses: Iterable[AugWitness], threshold_s: float
) -> dict[tuple[str, str], float]:
    directed: dict[tuple[str, str], float] = {}
    for witness in witnesses:
        if witness.fftt_s <= float(threshold_s) + 1e-12:
            key = (witness.source_node_uid, witness.target_node_uid)
            directed[key] = min(directed.get(key, float("inf")), witness.fftt_s)
    result: dict[tuple[str, str], float] = {}
    for (source, target), forward in directed.items():
        reverse = directed.get((target, source))
        if reverse is not None:
            result[tuple(sorted((source, target), key=_id_key))] = max(forward, reverse)
    return result


def absorbed_witnesses(
    witnesses: Iterable[AugWitness], node_uid_to_bacn: Mapping[str, str]
) -> tuple[AugWitness, ...]:
    result = [
        witness
        for witness in witnesses
        if node_uid_to_bacn.get(witness.source_node_uid) is not None
        and node_uid_to_bacn.get(witness.source_node_uid) == node_uid_to_bacn.get(witness.target_node_uid)
    ]
    result.sort(key=lambda item: (_id_key(item.source_node_uid), _id_key(item.target_node_uid), witness_rank(item)))
    return tuple(result)


def validate_witness(
    witness: AugWitness,
    graph: GraphData,
    suite: SuiteData,
    policy: AugPolicy,
) -> None:
    if witness.policy_id != policy.policy_id:
        raise Phase3AAugWitnessError("Witness policy ID mismatch")
    if not witness.arc_indices or len(witness.arc_indices) > policy.max_physical_arcs:
        raise Phase3AAugWitnessError("Witness hop bound failed")
    current = witness.source_node_idx
    total_fftt = 0.0
    total_length = 0.0
    for offset, arc_index in enumerate(witness.arc_indices):
        if suite.is_final_backbone_arc[arc_index]:
            raise Phase3AAugWitnessError("Witness contains a final-backbone arc")
        if int(graph.tail[arc_index]) != current:
            raise Phase3AAugWitnessError("Witness arc continuity failed")
        current = int(graph.head[arc_index])
        if offset < len(witness.arc_indices) - 1 and suite.is_final_backbone_node[current]:
            raise Phase3AAugWitnessError("Witness contains an intermediate final-backbone node")
        total_fftt += float(graph.arc_fftt_s[arc_index])
        total_length += float(graph.arc_length_m[arc_index])
    if current != witness.target_node_idx or not suite.is_final_backbone_node[current]:
        raise Phase3AAugWitnessError("Witness target is not the declared final-backbone node")
    if not np.isclose(total_fftt, witness.fftt_s, rtol=0.0, atol=1e-9):
        raise Phase3AAugWitnessError("Witness FFTT sum mismatch")
    if not np.isclose(total_length, witness.length_m, rtol=0.0, atol=1e-6):
        raise Phase3AAugWitnessError("Witness length sum mismatch")
    if witness.fftt_s > policy.threshold_s + 1e-12 or witness.detour_ratio > policy.max_length_geodesic_detour + 1e-12:
        raise Phase3AAugWitnessError("Witness policy bound failed")


def witness_rank(witness: AugWitness) -> tuple[float, float, int, tuple[tuple[int, int | str], ...]]:
    return (
        float(witness.fftt_s),
        float(witness.length_m),
        len(witness.arc_ids),
        tuple(_id_key(value) for value in witness.arc_ids),
    )


def _make_witness(
    graph: GraphData,
    suite: SuiteData,
    source: int,
    target: int,
    arc_indices: tuple[int, ...],
    fftt: float,
    length: float,
    detour: float,
    policy: AugPolicy,
) -> AugWitness:
    arc_ids = tuple(str(graph.arc_id[index]) for index in arc_indices)
    digest = hashlib.sha256("\x1f".join(arc_ids).encode("utf-8")).hexdigest()
    regions = tuple(
        sorted(
            {str(suite.arc_region_id[index]) for index in arc_indices if str(suite.arc_region_id[index]) != "backbone"}
        )
    )
    witness_id = stable_object_id(
        "augwitness",
        suite_id=suite.suite_id,
        candidate_id=policy.policy_id,
        members=arc_ids,
        ordered_members=True,
        signature={"source": str(graph.node_uid[source]), "target": str(graph.node_uid[target])},
    )
    return AugWitness(
        witness_id=witness_id,
        policy_id=policy.policy_id,
        source_node_idx=source,
        target_node_idx=target,
        source_node_uid=str(graph.node_uid[source]),
        target_node_uid=str(graph.node_uid[target]),
        fftt_s=float(fftt),
        length_m=float(length),
        arc_indices=arc_indices,
        arc_ids=arc_ids,
        crossed_region_ids=regions,
        detour_ratio=float(detour),
        witness_sha256=digest,
    )


def _id_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)
