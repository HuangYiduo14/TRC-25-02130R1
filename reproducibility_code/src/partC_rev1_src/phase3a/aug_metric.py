"""Source-sharded bounded distances on the temporary augmented backbone."""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from .geodesy import grs80_distance_m
from .models import (
    AugmentedDistanceShard,
    AugPolicy,
    AugWitness,
    GraphData,
    PairGraph,
    SuiteData,
    TemporaryAugmentedAdjacency,
)


class Phase3AAugMetricError(ValueError):
    pass


def build_temporary_augmented_adjacency(
    graph: GraphData,
    suite: SuiteData,
    witnesses: Iterable[AugWitness],
    *,
    backbone_node_indexes: Iterable[int] | None = None,
) -> TemporaryAugmentedAdjacency:
    """Build one deterministic primitive CSR over the selected backbone universe.

    Production callers omit ``backbone_node_indexes`` and therefore receive
    the complete accepted s03 final-backbone universe.  The optional argument
    is retained only for bounded tests and scoped continuation probes.
    """

    if backbone_node_indexes is None:
        selected_array = np.flatnonzero(suite.is_final_backbone_node).astype(np.int64)
    else:
        selected_array = np.asarray(sorted(set(map(int, backbone_node_indexes))), dtype=np.int64)
    if selected_array.size == 0:
        raise Phase3AAugMetricError("Temporary AUG graph requires backbone nodes")
    if np.any(selected_array < 0) or np.any(selected_array >= graph.node_count):
        raise Phase3AAugMetricError("Temporary AUG backbone index is out of range")
    if not np.all(suite.is_final_backbone_node[selected_array]):
        raise Phase3AAugMetricError("Temporary AUG universe contains a nonbackbone node")
    selected = np.asarray(
        sorted(selected_array.tolist(), key=lambda index: _id_key(str(graph.node_uid[index]))),
        dtype=np.int64,
    )
    local_by_global = np.full(graph.node_count, -1, dtype=np.int64)
    local_by_global[selected] = np.arange(selected.size, dtype=np.int64)

    final_indexes = np.flatnonzero(suite.is_final_backbone_arc).astype(np.int64)
    final_tail = local_by_global[graph.tail[final_indexes]]
    final_head = local_by_global[graph.head[final_indexes]]
    keep = (final_tail >= 0) & (final_head >= 0)
    tails = final_tail[keep].tolist()
    heads = final_head[keep].tolist()
    costs = graph.arc_fftt_s[final_indexes[keep]].astype(np.float64).tolist()
    stable_keys = [f"a:{graph.arc_id[index]}" for index in final_indexes[keep].tolist()]

    for witness in witnesses:
        source = int(witness.source_node_idx)
        target = int(witness.target_node_idx)
        if source < 0 or source >= graph.node_count or target < 0 or target >= graph.node_count:
            raise Phase3AAugMetricError("AUG witness endpoint is out of range")
        left, right = int(local_by_global[source]), int(local_by_global[target])
        if left < 0 or right < 0:
            continue
        tails.append(left)
        heads.append(right)
        costs.append(float(witness.fftt_s))
        stable_keys.append(f"w:{witness.witness_id}")
    if not tails:
        raise Phase3AAugMetricError("Temporary AUG graph has zero directed arcs")

    tail_array = np.asarray(tails, dtype=np.int64)
    head_array = np.asarray(heads, dtype=np.int64)
    cost_array = np.asarray(costs, dtype=np.float64)
    stable_array = np.asarray(stable_keys, dtype=str)
    if np.any(~np.isfinite(cost_array)) or np.any(cost_array < 0.0):
        raise Phase3AAugMetricError("Temporary AUG costs must be finite and nonnegative")
    order = np.lexsort((stable_array, cost_array, head_array, tail_array))
    tail_array = tail_array[order]
    head_array = head_array[order]
    cost_array = cost_array[order]
    first = np.ones(tail_array.size, dtype=bool)
    if tail_array.size > 1:
        first[1:] = (tail_array[1:] != tail_array[:-1]) | (head_array[1:] != head_array[:-1])
    tail_array = tail_array[first]
    head_array = head_array[first]
    cost_array = cost_array[first]
    counts = np.bincount(tail_array, minlength=selected.size)
    indptr = np.empty(selected.size + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    digest = _adjacency_digest(selected, indptr, head_array, cost_array)
    return TemporaryAugmentedAdjacency(
        backbone_node_indices=selected,
        indptr=indptr,
        head_local=head_array,
        cost_s=cost_array,
        adjacency_sha256=digest,
    )


def compute_augmented_distance_shard(
    adjacency: TemporaryAugmentedAdjacency,
    source_node_indexes: Iterable[int],
    policy: AugPolicy,
) -> AugmentedDistanceShard:
    """Compute compact threshold-bounded rows for an explicit source shard."""

    global_to_local = {
        int(value): index for index, value in enumerate(adjacency.backbone_node_indices.tolist())
    }
    requested = sorted(
        {int(value) for value in source_node_indexes},
        key=lambda value: global_to_local.get(value, np.iinfo(np.int64).max),
    )
    if not requested or any(value not in global_to_local for value in requested):
        raise Phase3AAugMetricError("Every AUG distance source must belong to the selected backbone universe")
    sources = np.asarray([global_to_local[value] for value in requested], dtype=np.int64)
    n = adjacency.node_count
    distance = np.full(n, np.inf, dtype=np.float64)
    settled_stamp = np.zeros(n, dtype=np.int64)
    offsets = np.zeros(sources.size + 1, dtype=np.int64)
    all_targets: list[int] = []
    all_distances: list[float] = []
    states_settled = np.zeros(sources.size, dtype=np.int64)
    arcs_scanned = np.zeros(sources.size, dtype=np.int64)
    threshold = float(policy.threshold_s)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise Phase3AAugMetricError("AUG distance threshold must be positive")
    for row, source in enumerate(sources.tolist(), start=1):
        stamp = row
        touched = [source]
        distance[source] = 0.0
        heap: list[tuple[float, int]] = [(0.0, source)]
        while heap:
            value, node = heapq.heappop(heap)
            if settled_stamp[node] == stamp or abs(value - distance[node]) > 1e-12:
                continue
            if value > threshold + 1e-12:
                break
            settled_stamp[node] = stamp
            states_settled[row - 1] += 1
            begin, end = int(adjacency.indptr[node]), int(adjacency.indptr[node + 1])
            arcs_scanned[row - 1] += end - begin
            for position in range(begin, end):
                target = int(adjacency.head_local[position])
                candidate = value + float(adjacency.cost_s[position])
                if candidate > threshold + 1e-12 or candidate >= distance[target] - 1e-12:
                    continue
                if not np.isfinite(distance[target]):
                    touched.append(target)
                distance[target] = candidate
                heapq.heappush(heap, (candidate, target))
        reached = sorted(
            (value for value in touched if value != source and np.isfinite(distance[value]) and distance[value] <= threshold + 1e-12)
        )
        all_targets.extend(reached)
        all_distances.extend(float(distance[value]) for value in reached)
        offsets[row] = len(all_targets)
        distance[np.asarray(touched, dtype=np.int64)] = np.inf
    return AugmentedDistanceShard(
        policy_id=policy.policy_id,
        threshold_s=threshold,
        adjacency_sha256=adjacency.adjacency_sha256,
        source_local=sources,
        source_offsets=offsets,
        target_local=np.asarray(all_targets, dtype=np.int64),
        distance_s=np.asarray(all_distances, dtype=np.float64),
        states_settled=states_settled,
        arcs_scanned=arcs_scanned,
    )


def merge_augmented_distance_shards(
    shards: Sequence[AugmentedDistanceShard],
    *,
    expected_source_local: Iterable[int] | None = None,
) -> AugmentedDistanceShard:
    """Merge shards in source order and prove every expected source occurs once."""

    values = tuple(shards)
    if not values:
        raise Phase3AAugMetricError("At least one AUG distance shard is required")
    policy = values[0].policy_id
    threshold = float(values[0].threshold_s)
    adjacency_hash = values[0].adjacency_sha256
    rows: list[tuple[int, np.ndarray, np.ndarray, int, int]] = []
    for shard in values:
        _validate_distance_shard(shard)
        if (
            shard.policy_id != policy
            or float(shard.threshold_s) != threshold
            or shard.adjacency_sha256 != adjacency_hash
        ):
            raise Phase3AAugMetricError("AUG distance shards have incompatible identities")
        for position, source in enumerate(shard.source_local.tolist()):
            begin, end = int(shard.source_offsets[position]), int(shard.source_offsets[position + 1])
            rows.append(
                (
                    int(source),
                    shard.target_local[begin:end],
                    shard.distance_s[begin:end],
                    int(shard.states_settled[position]),
                    int(shard.arcs_scanned[position]),
                )
            )
    rows.sort(key=lambda item: item[0])
    sources = np.asarray([row[0] for row in rows], dtype=np.int64)
    if sources.size != np.unique(sources).size:
        raise Phase3AAugMetricError("AUG distance source appears in more than one shard")
    if expected_source_local is not None:
        expected = np.asarray(sorted(set(map(int, expected_source_local))), dtype=np.int64)
        if not np.array_equal(sources, expected):
            raise Phase3AAugMetricError("AUG distance shard merge does not cover expected sources exactly once")
    offsets = np.zeros(sources.size + 1, dtype=np.int64)
    targets: list[int] = []
    distances: list[float] = []
    for position, row in enumerate(rows, start=1):
        targets.extend(map(int, row[1].tolist()))
        distances.extend(map(float, row[2].tolist()))
        offsets[position] = len(targets)
    result = AugmentedDistanceShard(
        policy_id=policy,
        threshold_s=threshold,
        adjacency_sha256=adjacency_hash,
        source_local=sources,
        source_offsets=offsets,
        target_local=np.asarray(targets, dtype=np.int64),
        distance_s=np.asarray(distances, dtype=np.float64),
        states_settled=np.asarray([row[3] for row in rows], dtype=np.int64),
        arcs_scanned=np.asarray([row[4] for row in rows], dtype=np.int64),
    )
    _validate_distance_shard(result)
    return result


def build_aug_pair_graph_from_distances(
    graph: GraphData,
    adjacency: TemporaryAugmentedAdjacency,
    distances: AugmentedDistanceShard,
    policy: AugPolicy,
) -> PairGraph:
    """Create mutual AUG admissibility using only compact source rows."""

    _validate_distance_shard(distances)
    if distances.adjacency_sha256 != adjacency.adjacency_sha256 or distances.policy_id != policy.policy_id:
        raise Phase3AAugMetricError("AUG distance identity differs from adjacency/policy")
    row_by_source = np.full(adjacency.node_count, -1, dtype=np.int64)
    row_by_source[distances.source_local] = np.arange(distances.source_count, dtype=np.int64)
    edge_u: list[int] = []
    edge_v: list[int] = []
    dissimilarity: list[float] = []
    physical = adjacency.backbone_node_indices
    for row, left in enumerate(distances.source_local.tolist()):
        begin, end = int(distances.source_offsets[row]), int(distances.source_offsets[row + 1])
        for position in range(begin, end):
            right = int(distances.target_local[position])
            if right <= left:
                continue
            reverse_row = int(row_by_source[right])
            if reverse_row < 0:
                continue
            reverse_begin = int(distances.source_offsets[reverse_row])
            reverse_end = int(distances.source_offsets[reverse_row + 1])
            reverse_targets = distances.target_local[reverse_begin:reverse_end]
            found = int(np.searchsorted(reverse_targets, left))
            if found >= reverse_targets.size or int(reverse_targets[found]) != left:
                continue
            reverse = float(distances.distance_s[reverse_begin + found])
            forward = float(distances.distance_s[position])
            left_global, right_global = int(physical[left]), int(physical[right])
            if graph.physical_weak_component[left_global] != graph.physical_weak_component[right_global]:
                continue
            geodesic = float(
                grs80_distance_m(
                    float(graph.lon[left_global]),
                    float(graph.lat[left_global]),
                    float(graph.lon[right_global]),
                    float(graph.lat[right_global]),
                )
            )
            if geodesic <= policy.search_radius_m + 1e-9:
                edge_u.append(left)
                edge_v.append(right)
                dissimilarity.append(max(forward, reverse))
    return PairGraph(
        node_ids=tuple(str(graph.node_uid[index]) for index in physical.tolist()),
        edge_u=np.asarray(edge_u, dtype=np.int64),
        edge_v=np.asarray(edge_v, dtype=np.int64),
        dissimilarity=np.asarray(dissimilarity, dtype=np.float64),
        component_ids=graph.physical_weak_component[physical].astype(np.int64),
        metric="MUTUAL_AUGMENTED_FFTT",
        threshold=float(policy.threshold_s),
        threshold_unit="seconds",
    )


def build_aug_pair_graph(
    graph: GraphData,
    suite: SuiteData,
    backbone_node_indexes: Iterable[int],
    witnesses: Iterable[AugWitness],
    policy: AugPolicy,
) -> PairGraph:
    """Compatibility wrapper using the new full-adjacency/source-shard API."""

    selected = tuple(sorted(set(map(int, backbone_node_indexes))))
    adjacency = build_temporary_augmented_adjacency(
        graph,
        suite,
        witnesses,
        backbone_node_indexes=selected,
    )
    shard = compute_augmented_distance_shard(adjacency, selected, policy)
    return build_aug_pair_graph_from_distances(graph, adjacency, shard, policy)


def write_augmented_distance_shard_new(path: str | Path, shard: AugmentedDistanceShard) -> dict[str, object]:
    """Persist one immutable compact shard and return its binding."""

    _validate_distance_shard(shard)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise Phase3AAugMetricError(f"Write-new-only shard path exists: {destination}")
    np.savez_compressed(
        destination,
        policy_id=np.asarray(shard.policy_id),
        threshold_s=np.asarray(shard.threshold_s, dtype=np.float64),
        adjacency_sha256=np.asarray(shard.adjacency_sha256),
        source_local=shard.source_local,
        source_offsets=shard.source_offsets,
        target_local=shard.target_local,
        distance_s=shard.distance_s,
        states_settled=shard.states_settled,
        arcs_scanned=shard.arcs_scanned,
    )
    return {
        "path": str(destination.resolve()),
        "size_bytes": int(destination.stat().st_size),
        "sha256": _file_sha256(destination),
        "source_count": shard.source_count,
        "distance_count": shard.distance_count,
    }


def load_augmented_distance_shard(path: str | Path) -> AugmentedDistanceShard:
    source = Path(path).resolve(strict=True)
    with np.load(source, allow_pickle=False) as payload:
        result = AugmentedDistanceShard(
            policy_id=str(payload["policy_id"].item()),
            threshold_s=float(payload["threshold_s"].item()),
            adjacency_sha256=str(payload["adjacency_sha256"].item()),
            source_local=payload["source_local"].astype(np.int64),
            source_offsets=payload["source_offsets"].astype(np.int64),
            target_local=payload["target_local"].astype(np.int64),
            distance_s=payload["distance_s"].astype(np.float64),
            states_settled=payload["states_settled"].astype(np.int64),
            arcs_scanned=payload["arcs_scanned"].astype(np.int64),
        )
    _validate_distance_shard(result)
    return result


def _validate_distance_shard(shard: AugmentedDistanceShard) -> None:
    if shard.source_offsets.size != shard.source_count + 1 or int(shard.source_offsets[0]) != 0:
        raise Phase3AAugMetricError("AUG distance shard offsets are invalid")
    if int(shard.source_offsets[-1]) != shard.distance_count or shard.distance_s.size != shard.distance_count:
        raise Phase3AAugMetricError("AUG distance shard vectors do not close")
    if shard.states_settled.size != shard.source_count or shard.arcs_scanned.size != shard.source_count:
        raise Phase3AAugMetricError("AUG distance counters do not align with sources")
    if shard.source_count != np.unique(shard.source_local).size or np.any(shard.source_local[1:] <= shard.source_local[:-1]):
        raise Phase3AAugMetricError("AUG distance sources must be unique and sorted")
    if np.any(~np.isfinite(shard.distance_s)) or np.any(shard.distance_s < 0.0):
        raise Phase3AAugMetricError("AUG distance values are invalid")
    for position in range(shard.source_count):
        begin, end = int(shard.source_offsets[position]), int(shard.source_offsets[position + 1])
        row = shard.target_local[begin:end]
        if row.size and (row.size != np.unique(row).size or np.any(row[1:] <= row[:-1])):
            raise Phase3AAugMetricError("AUG distance targets must be unique and sorted within each row")


def _adjacency_digest(
    nodes: np.ndarray, indptr: np.ndarray, heads: np.ndarray, costs: np.ndarray
) -> str:
    digest = hashlib.sha256()
    for value in (nodes, indptr, heads, costs):
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)
