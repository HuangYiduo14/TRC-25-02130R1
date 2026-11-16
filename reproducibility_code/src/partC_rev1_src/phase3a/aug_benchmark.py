"""Bounded real-data AUG equivalence and 1/2/4-worker benchmark harness."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .aug_metric import (
    build_temporary_augmented_adjacency,
    compute_augmented_distance_shard,
    merge_augmented_distance_shards,
    write_augmented_distance_shard_new,
)
from .aug_witness import (
    _make_witness,
    direct_witnesses_from_source,
    witness_census_with_stats,
    witness_rank,
)
from .authority import load_execution_config, resolve_authority_paths
from .geodesy import geodetic_to_ecef, grs80_distance_m
from .loaders import load_graph, load_suite
from .models import AugPolicy, AugWitness, GraphData, SuiteData
from .operations import _aug_policy
from .resources import directory_bytes, process_tree_rss
from .schemas import canonical_json_bytes, write_json_new


class Phase3AAugBenchmarkError(RuntimeError):
    pass


def run_aug_benchmark(
    config_path: str | Path,
    output_root: str | Path,
    *,
    source_target: int = 2_048,
    per_stratum_target: int = 512,
    reference_count: int = 128,
    shard_size: int = 512,
) -> dict[str, Any]:
    config = load_execution_config(config_path)
    destination = Path(output_root).resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=False)
    began_total = time.perf_counter()
    graph = load_graph(resolve_authority_paths(config))
    suite = load_suite(resolve_authority_paths(config), graph)
    source_frame, combined = select_aug_benchmark_sources(
        graph,
        suite,
        source_target=source_target,
        per_stratum_target=per_stratum_target,
    )
    source_path = destination / "aug_benchmark_sources.parquet"
    source_frame.to_parquet(source_path, index=False)
    combined_path = destination / "aug_benchmark_combined_sources.parquet"
    pd.DataFrame(
        {
            "combined_rank": np.arange(1, len(combined) + 1, dtype=np.int64),
            "node_idx": combined,
            "source_node_uid": [str(graph.node_uid[index]) for index in combined],
        }
    ).to_parquet(combined_path, index=False)
    reference_sources = [
        combined[position]
        for position in np.linspace(0, len(combined) - 1, min(reference_count, len(combined)), dtype=np.int64)
    ]
    policy_reports: list[dict[str, Any]] = []
    scaling_rows: list[dict[str, Any]] = []
    total_reference_seconds = 0.0
    peak_rss = process_tree_rss(os.getpid())[0]
    for policy_name in ("conservative", "moderate"):
        policy = _aug_policy(config, policy_name)
        reference_started = time.perf_counter()
        equivalence_rows = []
        optimized_source_seconds: list[float] = []
        for source in reference_sources:
            source_reference_started = time.perf_counter()
            reference = _reference_direct_witnesses_from_source(graph, suite, source, policy)
            source_reference_seconds = time.perf_counter() - source_reference_started
            source_optimized_started = time.perf_counter()
            optimized = direct_witnesses_from_source(graph, suite, source, policy)
            source_optimized_seconds = time.perf_counter() - source_optimized_started
            optimized_source_seconds.append(source_optimized_seconds)
            reference_signature = _witness_signature(reference)
            optimized_signature = _witness_signature(optimized)
            if reference_signature != optimized_signature:
                raise Phase3AAugBenchmarkError(
                    f"Optimized witness differs from reviewed reference: {policy.policy_id}/{graph.node_uid[source]}"
                )
            equivalence_rows.append(
                {
                    "source_node_uid": str(graph.node_uid[source]),
                    "witness_count": len(reference),
                    "reference_seconds": source_reference_seconds,
                    "optimized_seconds": source_optimized_seconds,
                    "exact": True,
                }
            )
        reference_seconds = time.perf_counter() - reference_started
        total_reference_seconds += reference_seconds
        baseline_signature = None
        baseline_distance_signature = None
        policy_scaling: list[dict[str, Any]] = []
        for worker_count in (1, 2, 4):
            worker_root = destination / policy.policy_id / f"workers_{worker_count}"
            worker_root.mkdir(parents=True, exist_ok=False)
            run_started = time.perf_counter()
            witness_started = time.perf_counter()
            chunks = _balanced_chunks(combined, worker_count)
            if worker_count == 1:
                chunk_results = [witness_census_with_stats(graph, suite, chunks[0], policy)]
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    chunk_results = list(
                        executor.map(lambda values: witness_census_with_stats(graph, suite, values, policy), chunks)
                    )
            witnesses = tuple(
                sorted(
                    (value for result, _ in chunk_results for value in result),
                    key=lambda value: (_id_key(value.source_node_uid), _id_key(value.target_node_uid), witness_rank(value)),
                )
            )
            stats = tuple(value for _, counters in chunk_results for value in counters)
            witness_seconds = time.perf_counter() - witness_started
            signature = hashlib.sha256(canonical_json_bytes(_witness_signature(witnesses))).hexdigest()
            if baseline_signature is None:
                baseline_signature = signature
            elif signature != baseline_signature:
                raise Phase3AAugBenchmarkError("Witness output differs across worker counts")
            adjacency_started = time.perf_counter()
            adjacency = build_temporary_augmented_adjacency(graph, suite, witnesses)
            adjacency_seconds = time.perf_counter() - adjacency_started
            source_shards = [combined[begin : begin + shard_size] for begin in range(0, len(combined), shard_size)]
            distance_started = time.perf_counter()
            if worker_count == 1:
                distance_shards = [
                    compute_augmented_distance_shard(adjacency, values, policy) for values in source_shards
                ]
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    distance_shards = list(
                        executor.map(
                            lambda values: compute_augmented_distance_shard(adjacency, values, policy),
                            source_shards,
                        )
                    )
            distance_seconds = time.perf_counter() - distance_started
            write_started = time.perf_counter()
            shard_paths = []
            for position, shard in enumerate(distance_shards):
                path = worker_root / f"distance_{position:05d}.npz"
                write_augmented_distance_shard_new(path, shard)
                shard_paths.append(path)
            local_by_global = {
                int(value): position
                for position, value in enumerate(adjacency.backbone_node_indices.tolist())
            }
            expected = sorted(local_by_global[value] for value in combined)
            shuffled = list(reversed(distance_shards))
            merge_started = time.perf_counter()
            merged = merge_augmented_distance_shards(shuffled, expected_source_local=expected)
            merge_seconds = time.perf_counter() - merge_started
            merged_path = worker_root / "merged_distances.npz"
            write_augmented_distance_shard_new(merged_path, merged)
            write_seconds = time.perf_counter() - write_started
            distance_signature = _distance_signature(merged)
            if baseline_distance_signature is None:
                baseline_distance_signature = distance_signature
            elif distance_signature != baseline_distance_signature:
                raise Phase3AAugBenchmarkError("Augmented distances differ across workers/shard completion order")
            # Independent dict-based reference on the equivalence source set.
            if worker_count == 1:
                _verify_distance_reference(adjacency, merged, reference_sources)
            elapsed = time.perf_counter() - run_started
            peak_rss = max(peak_rss, process_tree_rss(os.getpid())[0])
            witness_counts = np.asarray([value.accepted_witness_count for value in stats], dtype=np.float64)
            settled = np.asarray([value.states_settled for value in stats], dtype=np.float64)
            scanned = np.asarray([value.arcs_scanned for value in stats], dtype=np.float64)
            reached = np.asarray(
                [value.reached_other_backbone_count for value in stats], dtype=np.float64
            )
            rejected_radius = np.asarray(
                [value.rejected_radius_count for value in stats], dtype=np.float64
            )
            rejected_detour = np.asarray(
                [value.rejected_detour_count for value in stats], dtype=np.float64
            )
            row = {
                "policy_id": policy.policy_id,
                "worker_count": worker_count,
                "source_count": len(combined),
                "source_shard_size": shard_size,
                "source_shard_count": len(source_shards),
                "witness_count": len(witnesses),
                "zero_witness_share": float(np.mean(witness_counts == 0.0)),
                "witness_count_p50": float(np.quantile(witness_counts, 0.50)),
                "witness_count_p95": float(np.quantile(witness_counts, 0.95)),
                "search_states_p95": float(np.quantile(settled, 0.95)),
                "arcs_scanned_p95": float(np.quantile(scanned, 0.95)),
                "reached_other_backbone_total": int(reached.sum()),
                "reached_other_backbone_p95": float(np.quantile(reached, 0.95)),
                "rejected_radius_total": int(rejected_radius.sum()),
                "rejected_detour_total": int(rejected_detour.sum()),
                "witness_seconds": witness_seconds,
                "adjacency_build_seconds": adjacency_seconds,
                "bounded_distance_seconds": distance_seconds,
                "shard_write_and_merge_seconds": write_seconds,
                "merge_seconds": merge_seconds,
                "elapsed_seconds": elapsed,
                "temporary_adjacency_node_count": adjacency.node_count,
                "temporary_adjacency_arc_count": adjacency.arc_count,
                "distance_count": merged.distance_count,
                "output_bytes": directory_bytes(worker_root),
                "peak_process_tree_rss_bytes": peak_rss,
                "witness_sha256": signature,
                "distance_sha256": distance_signature,
                "deterministic": True,
            }
            policy_scaling.append(row)
            scaling_rows.append(row)
        one_seconds = policy_scaling[0]["elapsed_seconds"]
        for row in policy_scaling:
            row["parallel_efficiency"] = float(
                one_seconds / (row["worker_count"] * row["elapsed_seconds"])
            )
        policy_reports.append(
            {
                "policy": {
                    "policy_id": policy.policy_id,
                    "threshold_s": policy.threshold_s,
                    "search_radius_m": policy.search_radius_m,
                    "max_physical_arcs": policy.max_physical_arcs,
                    "max_length_geodesic_detour": policy.max_length_geodesic_detour,
                },
                "reference_source_count": len(reference_sources),
                "reference_seconds": reference_seconds,
                "optimized_runtime_per_source_seconds": {
                    "p50": float(np.quantile(optimized_source_seconds, 0.50)),
                    "p95": float(np.quantile(optimized_source_seconds, 0.95)),
                    "p99": float(np.quantile(optimized_source_seconds, 0.99)),
                    "max": float(np.max(optimized_source_seconds)),
                },
                "reference_equivalence": "PASS",
                "reference_rows": equivalence_rows,
                "worker_scaling": policy_scaling,
                "full_s03_backbone_universe": True,
                "full_graph_512_source_shard": len(combined) >= 512,
            }
        )
    scaling = pd.DataFrame(scaling_rows)
    scaling_path = destination / "aug_worker_scaling.csv"
    scaling.to_csv(scaling_path, index=False, lineterminator="\n")
    recommended = int(
        scaling.groupby("worker_count", sort=True)["elapsed_seconds"].sum().idxmin()
    )
    report = {
        "schema_version": "phase3a-task1-continuation-1.0",
        "status": "PASS",
        "source_selection": {
            "per_stratum_target": per_stratum_target,
            "combined_source_count": len(combined),
            "strata": {
                str(key): int(value)
                for key, value in source_frame.groupby("stratum", sort=True).size().to_dict().items()
            },
            "deduplicated": True,
        },
        "reference_equivalence_source_count_each_policy": len(reference_sources),
        "policies": policy_reports,
        "worker_and_shard_order_determinism": "PASS",
        "recommended_worker_count_from_measured_elapsed": recommended,
        "total_elapsed_seconds": time.perf_counter() - began_total,
        "reference_seconds": total_reference_seconds,
        "peak_process_tree_rss_bytes": peak_rss,
        "added_disk_bytes": directory_bytes(destination),
        "scientific_preference_declared": False,
    }
    report_path = destination / "aug_benchmark_report.json"
    write_json_new(report_path, report)
    return report


def select_aug_benchmark_sources(
    graph: GraphData,
    suite: SuiteData,
    *,
    source_target: int,
    per_stratum_target: int,
) -> tuple[pd.DataFrame, list[int]]:
    backbone = np.flatnonzero(suite.is_final_backbone_node).astype(np.int64)
    if backbone.size < source_target:
        source_target = int(backbone.size)
    stable = np.asarray(
        sorted(backbone.tolist(), key=lambda index: _id_key(str(graph.node_uid[index]))),
        dtype=np.int64,
    )
    rows: list[dict[str, Any]] = []

    geography_order = np.lexsort(
        (graph.node_uid[backbone].astype(str), graph.lon[backbone], graph.lat[backbone], graph.physical_weak_component[backbone])
    )
    geography = backbone[geography_order]
    geography = geography[
        np.linspace(0, geography.size - 1, min(per_stratum_target, geography.size), dtype=np.int64)
    ]

    ecef = geodetic_to_ecef(graph.lon[backbone], graph.lat[backbone])
    tiles = np.floor(ecef / 10_000.0).astype(np.int64)
    _, inverse, counts = np.unique(tiles, axis=0, return_inverse=True, return_counts=True)
    dense_order = sorted(
        range(backbone.size),
        key=lambda position: (-int(counts[inverse[position]]), _id_key(str(graph.node_uid[int(backbone[position])]))),
    )
    dense = backbone[np.asarray(dense_order[:per_stratum_target], dtype=np.int64)]

    nonbackbone = ~suite.is_final_backbone_arc
    outdegree = np.bincount(graph.tail[nonbackbone], minlength=graph.node_count)
    high_degree = sorted(
        backbone.tolist(),
        key=lambda index: (-int(outdegree[index]), _id_key(str(graph.node_uid[index]))),
    )[:per_stratum_target]

    tree = cKDTree(ecef)
    nearby_count = tree.query_ball_point(ecef, r=10_050.0, return_length=True, workers=1)
    nearby = sorted(
        range(backbone.size),
        key=lambda position: (-int(nearby_count[position]), _id_key(str(graph.node_uid[int(backbone[position])]))),
    )[:per_stratum_target]
    strata = {
        "component_geography_coverage": list(map(int, geography.tolist())),
        "dense_spatial_neighborhoods": list(map(int, dense.tolist())),
        "high_nonbackbone_outdegree": list(map(int, high_degree)),
        "nearby_target_or_witness_rich": [int(backbone[position]) for position in nearby],
    }
    combined: list[int] = []
    seen: set[int] = set()
    for stratum, values in strata.items():
        for rank, index in enumerate(values, start=1):
            rows.append(
                {
                    "stratum": stratum,
                    "stratum_rank": rank,
                    "node_idx": index,
                    "source_node_uid": str(graph.node_uid[index]),
                    "nonbackbone_outdegree": int(outdegree[index]),
                    "nearby_backbone_count_10km": int(nearby_count[np.searchsorted(backbone, index)]),
                    "physical_weak_component_id": int(graph.physical_weak_component[index]),
                    "lon": float(graph.lon[index]),
                    "lat": float(graph.lat[index]),
                }
            )
            if index not in seen:
                seen.add(index)
                combined.append(index)
    for index in stable.tolist():
        if len(combined) >= source_target:
            break
        if int(index) not in seen:
            seen.add(int(index))
            combined.append(int(index))
    combined = sorted(combined[:source_target], key=lambda index: _id_key(str(graph.node_uid[index])))
    combined_set = set(combined)
    frame = pd.DataFrame(rows)
    frame["in_combined_source_set"] = frame["node_idx"].isin(combined_set)
    return frame.sort_values(["stratum", "stratum_rank"], kind="mergesort"), combined


def _reference_direct_witnesses_from_source(
    graph: GraphData,
    suite: SuiteData,
    source_node_idx: int,
    policy: AugPolicy,
) -> tuple[AugWitness, ...]:
    """Reviewed pre-optimization algorithm retained only as an audit oracle."""

    source = int(source_node_idx)
    final_indexes = np.flatnonzero(suite.is_final_backbone_node)
    endpoint_distance = grs80_distance_m(
        np.full(final_indexes.size, graph.lon[source], dtype=np.float64),
        np.full(final_indexes.size, graph.lat[source], dtype=np.float64),
        graph.lon[final_indexes],
        graph.lat[final_indexes],
    )
    target_distance = {
        int(node): float(distance)
        for node, distance in zip(final_indexes.tolist(), endpoint_distance.tolist(), strict=True)
        if int(node) != source and float(distance) <= policy.search_radius_m
    }
    indptr, csr_arc_idx = graph.snapshot.csr_indptr, graph.snapshot.csr_arc_idx
    heap = [(0.0, 0.0, 0, (), source, (), (source,))]
    best_state = {(source, 0): (0.0, 0.0, ())}
    best_target: dict[int, AugWitness] = {}
    while heap:
        fftt, length, hops, arc_key, node, path_arcs, path_nodes = heapq.heappop(heap)
        if best_state.get((node, hops)) != (fftt, length, arc_key):
            continue
        if node != source and suite.is_final_backbone_node[node]:
            if node in target_distance:
                geodesic = target_distance[node]
                detour = length / max(geodesic, policy.epsilon_distance_m)
                if detour <= policy.max_length_geodesic_detour + 1e-12:
                    witness = _make_witness(
                        graph, suite, source, node, path_arcs, fftt, length, detour, policy
                    )
                    existing = best_target.get(node)
                    if existing is None or witness_rank(witness) < witness_rank(existing):
                        best_target[node] = witness
            continue
        if hops >= policy.max_physical_arcs or fftt >= policy.threshold_s:
            continue
        for position in range(int(indptr[node]), int(indptr[node + 1])):
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
                (next_fftt, next_length, next_hops, next_arc_key, next_node, path_arcs + (arc_index,), path_nodes + (next_node,)),
            )
    return tuple(sorted(best_target.values(), key=lambda item: (_id_key(item.target_node_uid), witness_rank(item))))


def _verify_distance_reference(adjacency: Any, merged: Any, source_globals: Sequence[int]) -> None:
    local_by_global = {
        int(value): position for position, value in enumerate(adjacency.backbone_node_indices.tolist())
    }
    row_by_source = {int(value): position for position, value in enumerate(merged.source_local.tolist())}
    selected = source_globals[: min(128, len(source_globals))]
    for global_source in selected:
        source = local_by_global[int(global_source)]
        reference = {source: 0.0}
        heap = [(0.0, source)]
        while heap:
            value, node = heapq.heappop(heap)
            if value != reference.get(node) or value > merged.threshold_s:
                continue
            for position in range(int(adjacency.indptr[node]), int(adjacency.indptr[node + 1])):
                target = int(adjacency.head_local[position])
                candidate = value + float(adjacency.cost_s[position])
                if candidate <= merged.threshold_s and candidate < reference.get(target, float("inf")) - 1e-12:
                    reference[target] = candidate
                    heapq.heappush(heap, (candidate, target))
        reference.pop(source, None)
        row = row_by_source[source]
        begin, end = int(merged.source_offsets[row]), int(merged.source_offsets[row + 1])
        observed = dict(
            zip(merged.target_local[begin:end].tolist(), merged.distance_s[begin:end].tolist(), strict=True)
        )
        if set(observed) != set(reference) or any(abs(observed[key] - reference[key]) > 1e-9 for key in reference):
            raise Phase3AAugBenchmarkError("Compact augmented distance differs from dict-based reference")


def _balanced_chunks(values: Sequence[int], count: int) -> list[list[int]]:
    result = [[] for _ in range(max(1, int(count)))]
    for position, value in enumerate(values):
        result[position % len(result)].append(int(value))
    return [value for value in result if value]


def _witness_signature(values: Sequence[AugWitness]) -> list[list[Any]]:
    return [
        [
            value.policy_id,
            value.source_node_uid,
            value.target_node_uid,
            value.fftt_s,
            value.length_m,
            len(value.arc_ids),
            list(value.arc_ids),
            value.detour_ratio,
        ]
        for value in values
    ]


def _distance_signature(value: Any) -> str:
    digest = hashlib.sha256()
    for array in (
        value.source_local,
        value.source_offsets,
        value.target_local,
        value.distance_s,
        value.states_settled,
        value.arcs_scanned,
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _id_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-target", type=int, default=2_048)
    parser.add_argument("--per-stratum-target", type=int, default=512)
    parser.add_argument("--reference-count", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=512)
    args = parser.parse_args(argv)
    report = run_aug_benchmark(
        args.config,
        args.output_root,
        source_target=args.source_target,
        per_stratum_target=args.per_stratum_target,
        reference_count=args.reference_count,
        shard_size=args.shard_size,
    )
    print(json.dumps({"status": report["status"], "total_elapsed_seconds": report["total_elapsed_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
