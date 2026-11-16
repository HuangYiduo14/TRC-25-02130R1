"""Concrete reusable Phase 3A campaign operations.

The same handlers execute either the externally approved ``FULL_TASK2`` scope
or the explicitly bounded ``BOUNDED_TASK1_CONTINUATION`` scope.  Scope changes
resource size and output disposition only; scientific thresholds and primitive
implementations are shared.
"""

from __future__ import annotations

import gc
import hashlib
import heapq
import json
import os
import subprocess
import threading
import time
import zipfile
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from partC_rev1_src.partC_rev1_hashing import partC_rev1_file_sha256

from .aug_metric import (
    build_aug_pair_graph_from_distances,
    build_temporary_augmented_adjacency,
    compute_augmented_distance_shard,
    merge_augmented_distance_shards,
    write_augmented_distance_shard_new,
)
from .aug_witness import absorbed_witnesses, witness_census_with_stats
from .admission import admit_round3_finalists
from .authority import config_sha256, resolve_authority_paths, verify_authority
from .bacn import build_bacn_membership, node_to_bacn
from .bundles import build_backbone_bundles, validate_bundle_closure
from .clustering import cluster_pair_graph
from .geodesy import grs80_distance_m
from .loaders import load_graph, load_od, load_suite
from .mapping import build_physical_candidate_lookup, tracer_interface_contract, write_lookup_npz_new
from .models import (
    AugPolicy,
    AugWitness,
    CoarseEdgeRelation,
    GraphData,
    PhysicalTransition,
    SuiteData,
    TemporaryAugmentedAdjacency,
)
from .pair_graph import build_geo_pair_graph
from .qgis import write_overlay_smoke
from .racn import build_fixed_racns, canonical_portal_signature
from .regional_pairs import (
    LazyRegionalPairOracle,
    RegionSupport,
    build_region_support,
    regional_support_contract,
)
from .reproduction import deterministic_frame_hash
from .resources import (
    ADDED_DISK_HARD_BYTES,
    PROCESS_TREE_RSS_HARD_BYTES,
    PROCESS_TREE_RSS_WARNING_BYTES,
    directory_bytes,
    process_tree_rss,
)
from .schemas import (
    canonical_json_bytes,
    stable_object_id,
    validate_candidate_matrix,
    write_json_new,
    write_text_new,
)
from .shortest_paths import (
    build_sparse_directed_graph,
    physical_fftt_tree,
    reconstruct_physical_path,
    reconstruct_restricted_path,
    restricted_dijkstra,
)
from .terminals import build_consolidated_terminals
from .diagnostics import (
    MaterializedScalarGraph,
    build_materialized_scalar_graph,
    materialized_scalar_shortest_paths,
    projected_generic_cost,
    scalar_shortest_path,
    structural_half_charge_adjustments,
)
from .liftability import audit_liftability
from .metrics import cost_errors, lcs_similarity, lifted_regret, weighted_arc_similarity
from .od_sampling import physical_reachability_mask, sample_od_pairs
from .projection import (
    compile_internal_aug_patterns,
    match_internal_aug_pattern,
    project_physical_path,
)


FULL_TASK2 = "FULL_TASK2"
BOUNDED_TASK1_CONTINUATION = "BOUNDED_TASK1_CONTINUATION"
EXECUTION_SCOPES = frozenset({FULL_TASK2, BOUNDED_TASK1_CONTINUATION})


class Phase3AOperationError(RuntimeError):
    """Raised when a concrete operation cannot produce meaningful output."""


def execute_operation(
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    implementation: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run one operation, enforce its receipt contract, and write it once."""

    task_id = str(contract.get("task_id", "")).strip()
    operation = str(contract.get("operation", "")).strip()
    scope = str(config.get("execution_scope", FULL_TASK2))
    if not task_id or not operation or scope not in EXECUTION_SCOPES:
        raise Phase3AOperationError("Operation requires task_id, operation, and a recognized execution scope")
    output_root = Path(str(config["output_root"])).resolve(strict=True)
    temp_root = Path(str(config["temp_root"])).resolve(strict=True)
    before_disk = directory_bytes(output_root) + directory_bytes(temp_root)
    began = time.perf_counter()
    start_rss, start_process_count = process_tree_rss(os.getpid())
    samples: list[tuple[int, int]] = [(start_rss, start_process_count)]
    sample_errors: list[str] = []
    stop_sampling = threading.Event()

    def sample_process_tree() -> None:
        while not stop_sampling.wait(0.05):
            try:
                samples.append(process_tree_rss(os.getpid()))
            except Exception as exc:  # pragma: no cover - exceptional psutil/OS failure
                sample_errors.append(f"{type(exc).__name__}: {exc}")
                return

    sampler = threading.Thread(
        target=sample_process_tree,
        name=f"phase3a-resource-{task_id}",
        daemon=True,
    )
    sampler.start()
    try:
        payload = dict(implementation(config, contract))
    finally:
        stop_sampling.set()
        sampler.join(timeout=2.0)
    elapsed = time.perf_counter() - began
    end_rss, process_count = process_tree_rss(os.getpid())
    samples.append((end_rss, process_count))
    if sample_errors:
        raise Phase3AOperationError(f"Continuous process-tree sampling failed: {sample_errors[0]}")
    peak_rss = max(value[0] for value in samples)
    peak_process_count = max(value[1] for value in samples)
    after_disk = directory_bytes(output_root) + directory_bytes(temp_root)
    added_disk = int(max(0, after_disk - before_disk))
    if peak_rss > PROCESS_TREE_RSS_HARD_BYTES:
        raise Phase3AOperationError("Operation exceeded the 24 GiB process-tree RSS hard maximum")
    if added_disk > ADDED_DISK_HARD_BYTES:
        raise Phase3AOperationError("Operation exceeded the 150 GiB added-disk hard maximum")
    meaningful = int(payload.get("meaningful_output_count", 0))
    if meaningful <= 0:
        raise Phase3AOperationError(f"Concrete operation was a no-op: {operation}")
    artifact_paths = [Path(str(value)).resolve(strict=True) for value in payload.pop("artifact_paths", ())]
    manifest = [_file_binding(path, output_root) for path in sorted(set(artifact_paths))]
    receipt = {
        "schema_id": "phase3a.operation_receipt",
        "schema_version": "phase3a-task1-continuation-1.0",
        "task_id": task_id,
        "operation": operation,
        "status": "PASS",
        "scope": scope,
        "input_identities": dict(config.get("accepted_identities", {})),
        "config_sha256": config_sha256(config),
        "code_commit": _git_head(Path(str(config["project_root"]))),
        "output_manifest": manifest,
        "meaningful_output_count": meaningful,
        "elapsed_seconds": float(elapsed),
        "peak_process_tree_rss_bytes": int(peak_rss),
        "process_count_at_completion": int(process_count),
        "peak_process_count": int(peak_process_count),
        "added_disk_bytes": added_disk,
        "resource_guard": {
            "continuous_sampling": True,
            "sample_interval_seconds": 0.05,
            "sample_count": len(samples),
            "rss_warning_bytes": PROCESS_TREE_RSS_WARNING_BYTES,
            "rss_hard_bytes": PROCESS_TREE_RSS_HARD_BYTES,
            "rss_warning_exceeded": peak_rss > PROCESS_TREE_RSS_WARNING_BYTES,
            "added_disk_hard_bytes": ADDED_DISK_HARD_BYTES,
            "hard_limit_exceeded": False,
        },
        "checkpoint": {
            "state": "COMPLETE_UNSEALED",
            "restart_contract": "verify receipt output manifest then skip or regenerate into a new run root",
            "write_new_only": True,
        },
        **payload,
    }
    receipt_path = output_root / "receipts" / f"{task_id}.json"
    write_json_new(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    receipt["receipt_sha256"] = partC_rev1_file_sha256(receipt_path)
    return receipt


def construct_geo_candidate(config: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    return execute_operation(config, contract, _construct_geo_candidate)


def census_aug_witness_shard(config: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    return execute_operation(config, contract, _census_aug_witness_shard)


def construct_aug_candidate(config: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    return execute_operation(config, contract, _construct_aug_candidate)


def finalize_candidate_topology(config: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    return execute_operation(config, contract, _finalize_candidate_topology)


def evaluate_candidate_od_batch(config: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    return execute_operation(config, contract, _evaluate_candidate_od_batch)


def validate_campaign(config: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    return execute_operation(config, contract, _validate_campaign)


def reproduce_topologies(config: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    return execute_operation(config, contract, _reproduce_topologies)


def seal_handoff(config: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    return execute_operation(config, contract, _seal_handoff)


def _construct_geo_candidate(config: Mapping[str, Any], contract: Mapping[str, Any]) -> Mapping[str, Any]:
    graph, suite = _load_graph_suite(config)
    matrix = _candidate_rows(config)
    requested = contract.get("candidate_ids")
    if requested is None:
        rows = [row for row in matrix if int(row["round"]) == 1]
    else:
        wanted = set(map(str, requested))
        rows = [row for row in matrix if row["candidate_id"] in wanted]
        if {row["candidate_id"] for row in rows} != wanted:
            raise Phase3AOperationError("GEO contract references a candidate outside the frozen matrix")
    resume_empty = set(map(str, contract.get("resume_empty_candidate_ids", ())))
    selected_ids = {str(row["candidate_id"]) for row in rows}
    if not resume_empty.issubset(selected_ids):
        raise Phase3AOperationError("GEO recovery names an empty candidate outside this task")
    backbone = np.flatnonzero(suite.is_final_backbone_node).astype(np.int64)
    weak = {
        str(graph.node_uid[index]): int(graph.physical_weak_component[index])
        for index in backbone.tolist()
    }
    registry_rows: list[dict[str, Any]] = []
    artifacts: list[Path] = []
    output_root = Path(str(config["output_root"]))
    for row in rows:
        candidate_started = time.perf_counter()
        candidate_id = str(row["candidate_id"])
        candidate_dir = output_root / "round1_geo" / "candidates" / candidate_id
        if candidate_id in resume_empty:
            if not candidate_dir.is_dir() or any(candidate_dir.iterdir()):
                raise Phase3AOperationError(
                    f"GEO recovery may resume only an existing empty candidate directory: {candidate_id}"
                )
        else:
            candidate_dir.mkdir(parents=True, exist_ok=False)
        family, algorithm = str(row["family"]), str(row["algorithm"])
        if family == "IDENTITY":
            cluster_result = None
            pair_edge_count = 0
        else:
            threshold_m = float(row["parameter_value"]) * 1000.0
            pair_graph = build_geo_pair_graph(
                graph.node_uid[backbone],
                graph.lon[backbone],
                graph.lat[backbone],
                graph.physical_weak_component[backbone],
                threshold_m,
            )
            complete_distance: Callable[[str, str], float] | None = None
            if algorithm == "SL":
                uid_to_index = {str(graph.node_uid[index]): int(index) for index in backbone.tolist()}

                def _sl_complete_distance(left: str, right: str) -> float:
                    left_index, right_index = uid_to_index[str(left)], uid_to_index[str(right)]
                    return float(
                        grs80_distance_m(
                            graph.lon[left_index], graph.lat[left_index],
                            graph.lon[right_index], graph.lat[right_index],
                        )
                    )

                complete_distance = _sl_complete_distance

            cluster_result = cluster_pair_graph(pair_graph, algorithm, complete_distance=complete_distance)
            pair_edge_count = pair_graph.edge_count()
        membership, summary = build_bacn_membership(
            suite_id=suite.suite_id,
            candidate_id=candidate_id,
            family=family,
            algorithm=algorithm,
            parameter_value=float(row["parameter_value"]),
            parameter_unit=str(row["parameter_unit"]),
            final_backbone_node_uids=graph.node_uid[backbone],
            weak_component_by_uid=weak,
            cluster_result=cluster_result,
        )
        bundle_membership, bundle_summary, suppressed = _bundles_for_membership(
            graph, suite, candidate_id, membership
        )
        summary = _complete_bacn_summary(summary, membership, suppressed, ())
        artifacts.extend(
            _write_candidate_artifacts(
                candidate_dir,
                membership=membership,
                summary=summary,
                bundle_membership=bundle_membership,
                bundle_summary=bundle_summary,
                suppressed=suppressed,
                spec={
                    "schema_version": "phase3a-task1-continuation-1.0",
                    "candidate_id": candidate_id,
                    "family": family,
                    "algorithm": algorithm,
                    "parameter_value": float(row["parameter_value"]),
                    "parameter_unit": str(row["parameter_unit"]),
                    "pair_edge_count": pair_edge_count,
                    "full_s03_backbone_universe": True,
                    "bounded_scientific_result": False,
                },
            )
        )
        registry_row = _candidate_registry_row(row, membership, bundle_membership, suppressed, pair_edge_count)
        registry_row["projected_full_build_seconds"] = time.perf_counter() - candidate_started
        registry_rows.append(registry_row)
    registry = pd.DataFrame(registry_rows).sort_values("candidate_id", kind="mergesort")
    registry_path = output_root / "round1_geo" / f"candidate_registry_{contract['task_id']}.parquet"
    _write_frame_new(registry_path, registry)
    artifacts.append(registry_path)
    return {
        "meaningful_output_count": len(registry),
        "artifact_paths": artifacts,
        "candidate_ids": registry["candidate_id"].astype(str).tolist(),
        "final_backbone_node_count": int(backbone.size),
        "full_s03_geo_candidate_completed": bool(
            any(row["family"] == "GEO" and row["algorithm"] in {"CL", "GC"} for row in rows)
        ),
        "resumed_empty_candidate_ids": sorted(resume_empty),
        "scientific_preference_declared": False,
    }


def _census_aug_witness_shard(config: Mapping[str, Any], contract: Mapping[str, Any]) -> Mapping[str, Any]:
    graph, suite = _load_graph_suite(config)
    policy = _aug_policy(config, str(contract["policy_id"]))
    sources = _resolve_source_indexes(graph, suite, contract)
    shard_size = int(contract.get("source_shard_size", len(sources)))
    if shard_size <= 0:
        raise Phase3AOperationError("AUG witness shard size must be positive")
    output_root = Path(str(config["output_root"]))
    policy_root = output_root / "round2_aug" / "witness_cache" / policy.policy_id
    artifacts: list[Path] = []
    shard_records: list[dict[str, Any]] = []
    total_witnesses = 0
    for begin in range(0, len(sources), shard_size):
        shard_sources = sources[begin : begin + shard_size]
        shard_id = f"{contract.get('shard_id', contract['task_id'])}_{begin // shard_size:05d}"
        shard_dir = policy_root / shard_id
        shard_dir.mkdir(parents=True, exist_ok=False)
        began = time.perf_counter()
        witnesses, stats = witness_census_with_stats(graph, suite, shard_sources, policy)
        seconds = time.perf_counter() - began
        paths = _write_witness_artifacts(shard_dir, witnesses, stats, graph, policy)
        artifacts.extend(paths)
        total_witnesses += len(witnesses)
        shard_records.append(
            {
                "shard_id": shard_id,
                "source_count": len(shard_sources),
                "witness_count": len(witnesses),
                "elapsed_seconds": seconds,
                "source_node_uids": [str(graph.node_uid[index]) for index in shard_sources],
                "zero_witness_source_count": int(sum(value.accepted_witness_count == 0 for value in stats)),
                "states_settled": int(sum(value.states_settled for value in stats)),
                "arcs_scanned": int(sum(value.arcs_scanned for value in stats)),
            }
        )
    index_path = policy_root / f"index_{contract['task_id']}.json"
    write_json_new(
        index_path,
        {
            "schema_version": "phase3a-task1-continuation-1.0",
            "policy": asdict(policy),
            "full_graph": True,
            "explicit_source_shards": True,
            "shards": shard_records,
        },
    )
    artifacts.append(index_path)
    return {
        "meaningful_output_count": len(sources),
        "artifact_paths": artifacts,
        "policy_id": policy.policy_id,
        "source_count": len(sources),
        "witness_count": total_witnesses,
        "shard_count": len(shard_records),
        "first_backbone_stop_unconditional": True,
        "geodesic_evaluated_only_for_reached_backbone": True,
    }


def _construct_aug_candidate(config: Mapping[str, Any], contract: Mapping[str, Any]) -> Mapping[str, Any]:
    graph, suite = _load_graph_suite(config)
    matrix = _candidate_rows(config)
    requested = contract.get("candidate_ids")
    if requested is None:
        rows = [row for row in matrix if int(row["round"]) == 2]
    else:
        wanted = set(map(str, requested))
        rows = [row for row in matrix if row["candidate_id"] in wanted]
        if {row["candidate_id"] for row in rows} != wanted:
            raise Phase3AOperationError("AUG contract references a candidate outside the frozen matrix")
    resume_empty = set(map(str, contract.get("resume_empty_candidate_ids", ())))
    selected_ids = {str(row["candidate_id"]) for row in rows}
    if not resume_empty.issubset(selected_ids):
        raise Phase3AOperationError("AUG recovery names an empty candidate outside this task")
    output_root = Path(str(config["output_root"]))
    artifacts: list[Path] = []
    registry_rows: list[dict[str, Any]] = []
    for threshold, policy_rows in _group_aug_rows(rows).items():
        policy = _aug_policy_for_threshold(config, threshold)
        witnesses, witness_paths = _load_policy_witnesses(output_root, policy.policy_id, graph, suite)
        if not witness_paths:
            raise Phase3AOperationError(f"No immutable witness shards found for AUG policy {policy.policy_id}")
        adjacency = build_temporary_augmented_adjacency(graph, suite, witnesses)
        source_contract = contract.get("distance_source_node_uids")
        source_registry = contract.get("distance_source_registry_path")
        if source_contract is None and source_registry is None:
            source_indexes = np.flatnonzero(suite.is_final_backbone_node).astype(np.int64).tolist()
        else:
            source_indexes = _resolve_source_indexes(
                graph,
                suite,
                {
                    "source_node_uids": source_contract,
                    "source_registry_path": source_registry,
                    "source_limit": contract.get("distance_source_limit"),
                },
            )
        source_indexes = sorted(set(source_indexes), key=lambda index: _id_key(str(graph.node_uid[index])))
        shard_size = int(contract.get("distance_source_shard_size", config.get("batching", {}).get("aug_source_shard_size", 512)))
        computed = []
        shard_paths: list[Path] = []
        distance_root = output_root / "round2_aug" / "distance_cache" / policy.policy_id / str(contract["task_id"])
        distance_root.mkdir(parents=True, exist_ok=False)
        for begin in range(0, len(source_indexes), shard_size):
            shard = compute_augmented_distance_shard(adjacency, source_indexes[begin : begin + shard_size], policy)
            path = distance_root / f"distance_{begin // shard_size:05d}.npz"
            write_augmented_distance_shard_new(path, shard)
            computed.append(shard)
            shard_paths.append(path)
        local_by_global = {
            int(value): index for index, value in enumerate(adjacency.backbone_node_indices.tolist())
        }
        expected_local = [local_by_global[int(value)] for value in source_indexes]
        merged = merge_augmented_distance_shards(computed, expected_source_local=expected_local)
        merged_path = distance_root / "merged_distances.npz"
        write_augmented_distance_shard_new(merged_path, merged)
        artifacts.extend([*shard_paths, merged_path])
        pair_graph = build_aug_pair_graph_from_distances(graph, adjacency, merged, policy)
        for row in policy_rows:
            candidate_started = time.perf_counter()
            candidate_id = str(row["candidate_id"])
            candidate_dir = output_root / "round2_aug" / "candidates" / candidate_id
            if candidate_id in resume_empty:
                if not candidate_dir.is_dir() or any(candidate_dir.iterdir()):
                    raise Phase3AOperationError(
                        f"AUG recovery may resume only an existing empty candidate directory: {candidate_id}"
                    )
            else:
                candidate_dir.mkdir(parents=True, exist_ok=False)
            complete_distance = None
            if str(row["algorithm"]) == "SL":
                complete_distance = _aug_complete_distance_provider(graph, adjacency)
            cluster_result = cluster_pair_graph(
                pair_graph,
                str(row["algorithm"]),
                complete_distance=complete_distance,
            )
            backbone = adjacency.backbone_node_indices
            weak = {
                str(graph.node_uid[index]): int(graph.physical_weak_component[index])
                for index in backbone.tolist()
            }
            membership, summary = build_bacn_membership(
                suite_id=suite.suite_id,
                candidate_id=candidate_id,
                family="AUG",
                algorithm=str(row["algorithm"]),
                parameter_value=float(row["parameter_value"]),
                parameter_unit=str(row["parameter_unit"]),
                final_backbone_node_uids=graph.node_uid[backbone],
                weak_component_by_uid=weak,
                cluster_result=cluster_result,
            )
            mapping = node_to_bacn(membership)
            absorbed = absorbed_witnesses(witnesses, mapping)
            bundle_membership, bundle_summary, suppressed = _bundles_for_membership(
                graph, suite, candidate_id, membership
            )
            summary = _complete_bacn_summary(summary, membership, suppressed, absorbed)
            candidate_artifacts = _write_candidate_artifacts(
                candidate_dir,
                membership=membership,
                summary=summary,
                bundle_membership=bundle_membership,
                bundle_summary=bundle_summary,
                suppressed=suppressed,
                spec={
                    "schema_version": "phase3a-task1-continuation-1.0",
                    "candidate_id": candidate_id,
                    "family": "AUG",
                    "algorithm": str(row["algorithm"]),
                    "parameter_value": float(row["parameter_value"]),
                    "parameter_unit": str(row["parameter_unit"]),
                    "policy": asdict(policy),
                    "witness_paths": [str(path) for path in witness_paths],
                    "distance_source_node_uids": [str(graph.node_uid[index]) for index in source_indexes],
                    "distance_source_count": len(source_indexes),
                    "full_s03_backbone_universe": True,
                    "distance_rows_complete_for_universe": len(source_indexes) == adjacency.node_count,
                    "temporary_adjacency_sha256": adjacency.adjacency_sha256,
                    "temporary_adjacency_arc_count": adjacency.arc_count,
                    "pair_edge_count": pair_graph.edge_count(),
                    "temporary_witnesses_not_final_bundles": True,
                    "bounded_scientific_result": False,
                },
                absorbed_witnesses=absorbed,
            )
            artifacts.extend(candidate_artifacts)
            registry_row = _candidate_registry_row(row, membership, bundle_membership, suppressed, pair_graph.edge_count())
            registry_row["projected_full_build_seconds"] = time.perf_counter() - candidate_started
            registry_rows.append(registry_row)
    registry = pd.DataFrame(registry_rows).sort_values("candidate_id", kind="mergesort")
    registry_path = output_root / "round2_aug" / f"candidate_registry_{contract['task_id']}.parquet"
    _write_frame_new(registry_path, registry)
    artifacts.append(registry_path)
    return {
        "meaningful_output_count": len(registry),
        "artifact_paths": artifacts,
        "candidate_ids": registry["candidate_id"].astype(str).tolist(),
        "source_sharded_distances": True,
        "full_temporary_backbone_universe": True,
        "mutual_rule": True,
        "scientific_preference_declared": False,
    }


def _finalize_candidate_topology(config: Mapping[str, Any], contract: Mapping[str, Any]) -> Mapping[str, Any]:
    substage = str(contract.get("substage", "full_topology"))
    if substage == "admission":
        return _run_candidate_admission(config, contract)
    if substage != "full_topology":
        raise Phase3AOperationError(f"Unknown finalization substage: {substage}")
    graph, suite = _load_graph_suite(config)
    output_root = Path(str(config["output_root"]))
    candidate_ids = _resolve_finalist_ids(output_root, contract)
    artifacts: list[Path] = []
    summaries: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        source_dir = _find_candidate_dir(output_root, candidate_id)
        membership = pd.read_parquet(source_dir / "bacn_membership.parquet")
        bacn_summary = pd.read_parquet(source_dir / "bacn_summary.parquet")
        bundle_membership = pd.read_parquet(source_dir / "bundle_membership.parquet")
        bundle_summary = pd.read_parquet(source_dir / "bundle_summary.parquet")
        suppressed = pd.read_parquet(source_dir / "suppressed_backbone_arcs.parquet")
        validate_bundle_closure(suite.is_final_backbone_arc, graph.arc_id, bundle_membership, suppressed)
        candidate_dir = output_root / "round3_finalists" / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=False)
        for name, frame in (
            ("bacn_membership.parquet", membership),
            ("bacn_summary.parquet", bacn_summary),
            ("bundle_membership.parquet", bundle_membership),
            ("bundle_summary.parquet", bundle_summary),
            ("suppressed_backbone_arcs.parquet", suppressed),
        ):
            path = candidate_dir / name
            _write_frame_new(path, frame)
            artifacts.append(path)
        bacn_map = node_to_bacn(membership)
        racn_membership, racn_summary, racn_witnesses, racn_evidence = _construct_racns(
            config, contract, graph, suite, candidate_id, bacn_map
        )
        for name, frame in (
            ("racn_membership.parquet", racn_membership),
            ("racn_summary.parquet", racn_summary),
            ("racn_restricted_witnesses.parquet", racn_witnesses),
        ):
            path = candidate_dir / name
            _write_frame_new(path, frame)
            artifacts.append(path)
        racn_map = (
            dict(
                zip(
                    racn_membership["physical_portal_node_uid"].astype(str),
                    racn_membership["racn_id"].astype(str),
                    strict=True,
                )
            )
            if not racn_membership.empty
            else {}
        )
        terminals, terminal_members = build_consolidated_terminals(
            suite_id=suite.suite_id,
            candidate_id=candidate_id,
            portals=suite.portals,
            centroid_roles=suite.centroid_roles,
            node_to_bacn=bacn_map,
            node_to_racn=racn_map,
        )
        terminal_path = candidate_dir / "consolidated_terminals.parquet"
        member_path = candidate_dir / "terminal_fine_members.parquet"
        _write_frame_new(terminal_path, terminals)
        _write_frame_new(member_path, terminal_members)
        artifacts.extend([terminal_path, member_path])

        support_regions = _resolve_support_regions(suite, contract, config)
        supports: list[RegionSupport] = []
        support_rows: list[dict[str, Any]] = []
        support_root = candidate_dir / "regional_support"
        support_root.mkdir(parents=True, exist_ok=False)
        for region in support_regions:
            support = build_region_support(
                region_id=region,
                tail=graph.tail,
                head=graph.head,
                fftt_s=graph.arc_fftt_s,
                arc_ids=graph.arc_id,
                region_arc_mask=suite.arc_region_id == region,
            )
            supports.append(support)
            support_path = support_root / f"{region}.npz"
            _write_region_support_new(support_path, support)
            artifacts.append(support_path)
            support_rows.append(
                {
                    "region_id": region,
                    "node_count": support.graph.node_count,
                    "arc_count": support.graph.arc_count,
                    "scc_count": support.scc_count,
                    "artifact_path": str(support_path),
                    "artifact_sha256": partC_rev1_file_sha256(support_path),
                }
            )
        terminal_hash = deterministic_frame_hash(
            terminals, sort_columns=["candidate_id", "region_id", "terminal_id"]
        )
        support_contract = regional_support_contract(
            candidate_id=candidate_id,
            region_supports=supports,
            terminal_registry_hash=terminal_hash,
        )
        support_contract.update(
            {
                "scope": str(config.get("execution_scope", FULL_TASK2)),
                "constructed_region_count": len(supports),
                "total_region_count": len(suite.region_ids),
                "full_region_support_constructed": len(supports) == len(suite.region_ids),
                "region_artifacts": support_rows,
            }
        )
        support_contract_path = candidate_dir / "regional_pair_support_contract.json"
        write_json_new(support_contract_path, support_contract)
        artifacts.append(support_contract_path)

        lookup = build_physical_candidate_lookup(
            node_uid=graph.node_uid,
            arc_id=graph.arc_id,
            bacn_membership=membership,
            racn_membership=racn_membership,
            bundle_membership=bundle_membership,
            suppressed_arcs=suppressed,
            portal_roles=suite.portals,
            centroid_roles=suite.centroid_roles,
            terminals=terminals,
            terminal_members=terminal_members,
        )
        lookup_path = candidate_dir / "physical_to_candidate_lookup.npz"
        lookup_binding = write_lookup_npz_new(lookup_path, lookup)
        tracer = tracer_interface_contract(lookup)
        tracer.update(
            {
                "candidate_id": candidate_id,
                "lookup_binding": lookup_binding,
                "phase4_execution_authorized": False,
            }
        )
        tracer_path = candidate_dir / "tracer_interface.json"
        write_json_new(tracer_path, tracer)
        artifacts.extend([lookup_path, tracer_path])

        absorbed = _load_absorbed_witnesses(source_dir, graph, suite)
        source_absorbed_path = source_dir / "aug_internal_witnesses.parquet"
        if source_absorbed_path.is_file():
            finalist_absorbed_path = candidate_dir / "aug_internal_witnesses.parquet"
            _write_frame_new(finalist_absorbed_path, pd.read_parquet(source_absorbed_path))
            artifacts.append(finalist_absorbed_path)
        structural = _structural_charge_table(candidate_id, bacn_summary, suppressed, absorbed, bacn_map)
        structural_path = candidate_dir / "structural_internal_charges.parquet"
        _write_frame_new(structural_path, structural)
        artifacts.append(structural_path)
        final_spec = {
            "schema_version": "phase3a-task1-continuation-1.0",
            "candidate_id": candidate_id,
            "source_candidate_dir": str(source_dir),
            "source_candidate_spec_sha256": partC_rev1_file_sha256(source_dir / "candidate_spec.json"),
            "fixed_racn": racn_evidence,
            "direct_mode_required": True,
            "structural_mode_required": True,
            "global_scale_mode_optional": True,
            "centroid_through_allowed": False,
            "phase4_executed": False,
        }
        spec_path = candidate_dir / "finalist_spec.json"
        write_json_new(spec_path, final_spec)
        artifacts.append(spec_path)
        summaries.append(
            {
                "candidate_id": candidate_id,
                "bacn_count": int(bacn_summary["bacn_id"].nunique()),
                "racn_count": int(racn_summary["racn_id"].nunique()),
                "merged_racn_count": int((racn_summary["member_count"] > 1).sum()),
                "terminal_count": len(terminals),
                "bundle_count": int(bundle_membership["bundle_id"].nunique()),
                "support_region_count": len(supports),
                "direct_mode": True,
                "structural_mode": True,
            }
        )
    summary_path = output_root / "round3_finalists" / f"finalization_{contract['task_id']}.parquet"
    _write_frame_new(summary_path, pd.DataFrame(summaries))
    artifacts.append(summary_path)
    return {
        "meaningful_output_count": len(summaries),
        "artifact_paths": artifacts,
        "candidate_ids": candidate_ids,
        "fixed_racn_applied": True,
        "direct_and_structural_modes_mandatory": True,
        "phase4_executed": False,
    }


def _run_candidate_admission(config: Mapping[str, Any], contract: Mapping[str, Any]) -> Mapping[str, Any]:
    output_root = Path(str(config["output_root"]))
    registry_paths = sorted(output_root.glob("round[12]_*/*candidate_registry_*.parquet"))
    if not registry_paths:
        registry_paths = sorted(output_root.rglob("candidate_registry_*.parquet"))
    if not registry_paths:
        raise Phase3AOperationError("Finalist admission found no candidate census registry")
    frame = pd.concat([pd.read_parquet(path) for path in registry_paths], ignore_index=True)
    frame = frame.sort_values("candidate_id", kind="mergesort").drop_duplicates("candidate_id", keep="last")
    if str(config.get("execution_scope", FULL_TASK2)) == BOUNDED_TASK1_CONTINUATION:
        requested = set(map(str, contract.get("candidate_ids", frame["candidate_id"].astype(str).tolist())))
        result = frame[frame["candidate_id"].astype(str).isin(requested)].copy()
        result["selected_slot"] = np.arange(1, len(result) + 1)
        result["selected_slot_role"] = "BOUNDED_CONTINUATION_EVIDENCE"
        result["admission_reason"] = "NOT_SCIENTIFIC_SELECTION_BOUNDED_ENGINE_PROOF"
        result["disposition"] = "ADMITTED_BOUNDED_CONTINUATION"
        result["winning_method_selected"] = False
    else:
        result = admit_round3_finalists(
            frame,
            allow_optional_sl=bool(contract.get("allow_optional_sl", True)),
            maximum_production_finalists=int(contract.get("maximum_production_finalists", 6)),
        )
    path = output_root / "round3_finalists" / "candidate_admission.parquet"
    _write_frame_new(path, result)
    return {
        "meaningful_output_count": len(result),
        "artifact_paths": [path],
        "admitted_candidate_ids": result.loc[
            result["disposition"].astype(str).str.startswith("ADMITTED"), "candidate_id"
        ].astype(str).tolist(),
        "operational_admission_only": True,
        "winning_method_selected": False,
    }


def _evaluate_candidate_od_batch(config: Mapping[str, Any], contract: Mapping[str, Any]) -> Mapping[str, Any]:
    substage = str(contract.get("substage", "projected_scalar_lifted"))
    if substage in {"physical_sample_and_trees", "sample_and_trees"}:
        return _build_physical_od_sample(config, contract)
    if substage not in {"projected_scalar_lifted", "evaluate"}:
        raise Phase3AOperationError(f"Unknown OD evaluation substage: {substage}")
    graph, suite = _load_graph_suite(config)
    output_root = Path(str(config["output_root"]))
    sample_path = Path(str(contract.get("sample_path", output_root / "round4_evaluation" / "od_sample.parquet")))
    paths_path = Path(str(contract.get("physical_paths_path", output_root / "round4_evaluation" / "physical_paths.parquet")))
    sample = pd.read_parquet(sample_path)
    physical_paths = pd.read_parquet(paths_path)
    candidate_ids = _resolve_evaluation_candidate_ids(output_root, contract)
    artifacts: list[Path] = []
    summary_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        candidate_dir = output_root / "round3_finalists" / "candidates" / candidate_id
        results, engine_stats = _evaluate_one_candidate(
            config,
            contract,
            graph,
            suite,
            candidate_id,
            candidate_dir,
            sample,
            physical_paths,
        )
        if results.empty:
            raise Phase3AOperationError(f"Candidate evaluation produced zero rows: {candidate_id}")
        result_path = output_root / "round4_evaluation" / "candidates" / candidate_id / "candidate_od_results.parquet"
        _write_frame_new(result_path, results)
        artifacts.append(result_path)
        support_summary_path = candidate_dir / "regional_pair_support_summary.parquet"
        if support_summary_path.is_file():
            artifacts.append(support_summary_path)
        mode_counts = results.groupby("diagnostic_mode", sort=True).size().to_dict()
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "unique_od_count": int(results["sample_od_id"].nunique()),
                "result_row_count": len(results),
                "direct_row_count": int(mode_counts.get("direct", 0)),
                "structural_row_count": int(mode_counts.get("structural", 0)),
                "liftable_share": float(results["liftable"].mean()),
                "scalar_relative_error_median": float(results["scalar_relative_error"].dropna().median()),
                "scientific_result": False if str(config.get("execution_scope")) == BOUNDED_TASK1_CONTINUATION else True,
                **engine_stats,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = output_root / "round4_evaluation" / f"evaluation_summary_{contract['task_id']}.parquet"
    _write_frame_new(summary_path, summary)
    artifacts.append(summary_path)
    minimum = int(contract.get("minimum_unique_od_count", 20 if str(config.get("execution_scope")) == BOUNDED_TASK1_CONTINUATION else 100))
    if (summary["unique_od_count"] < minimum).any():
        raise Phase3AOperationError("OD evaluation did not meet the scoped unique-OD minimum")
    if (summary[["direct_row_count", "structural_row_count"]] < minimum).any().any():
        raise Phase3AOperationError("Every evaluated finalist requires direct and structural rows")
    return {
        "meaningful_output_count": int(summary["result_row_count"].sum()),
        "artifact_paths": artifacts,
        "candidate_ids": candidate_ids,
        "minimum_unique_od_count": minimum,
        "direct_and_structural_modes_complete": True,
        "global_scale_mode_executed": False,
        "phase4_executed": False,
    }


def _build_physical_od_sample(config: Mapping[str, Any], contract: Mapping[str, Any]) -> Mapping[str, Any]:
    graph, suite = _load_graph_suite(config)
    od = load_od(resolve_authority_paths(config))
    snapshot = od.snapshot
    reachable = physical_reachability_mask(
        origin_node_idx=snapshot.origin_node_idx,
        destination_node_idx=snapshot.destination_node_idx,
        strong_component=graph.physical_strong_component,
        tail=graph.tail,
        head=graph.head,
    )
    scoped = str(config.get("execution_scope", FULL_TASK2)) == BOUNDED_TASK1_CONTINUATION
    target_origins = int(contract.get("target_origins", 4 if scoped else config["od_sampling"]["target_origins"]))
    destinations_per_origin = int(contract.get("destinations_per_origin", 5))
    minimum_od = int(contract.get("minimum_unique_od_count", 20 if scoped else config["od_sampling"]["hard_minimum_od_count"]))
    minimum_origins = int(contract.get("hard_minimum_origins", max(2, minimum_od // max(1, destinations_per_origin))))
    minimum_distance = float(contract.get("minimum_distance_km", config["od_sampling"]["minimum_geodesic_distance_km"]))
    try:
        sample = sample_od_pairs(
            suite_id=suite.suite_id,
            origin_node_idx=snapshot.origin_node_idx,
            destination_node_idx=snapshot.destination_node_idx,
            demand=snapshot.demand,
            node_uid=graph.node_uid,
            lon=graph.lon,
            lat=graph.lat,
            state_fips=graph.state_fips,
            physical_weak_component=graph.physical_weak_component,
            directed_reachable=reachable,
            minimum_distance_km=minimum_distance,
            target_origins=target_origins,
            destinations_per_origin=destinations_per_origin,
            hard_minimum_origins=minimum_origins,
            hard_minimum_od_count=minimum_od,
            seed=int(config.get("random_seed", 42)),
        )
    except Exception:
        fallback = float(config["od_sampling"].get("fallback_minimum_distance_km", 300.0))
        if minimum_distance <= fallback:
            raise
        sample = sample_od_pairs(
            suite_id=suite.suite_id,
            origin_node_idx=snapshot.origin_node_idx,
            destination_node_idx=snapshot.destination_node_idx,
            demand=snapshot.demand,
            node_uid=graph.node_uid,
            lon=graph.lon,
            lat=graph.lat,
            state_fips=graph.state_fips,
            physical_weak_component=graph.physical_weak_component,
            directed_reachable=reachable,
            minimum_distance_km=fallback,
            target_origins=target_origins,
            destinations_per_origin=destinations_per_origin,
            hard_minimum_origins=minimum_origins,
            hard_minimum_od_count=minimum_od,
            seed=int(config.get("random_seed", 42)),
        )
    if sample.duplicated(["origin_idx", "destination_idx"]).any():
        raise Phase3AOperationError("Derived OD sample contains duplicate physical endpoints")
    path_rows: list[dict[str, Any]] = []
    for origin, rows in sample.groupby("origin_idx", sort=True):
        tree = physical_fftt_tree(graph, int(origin))
        for row in rows.itertuples(index=False):
            destination = int(row.destination_idx)
            arc_indexes = reconstruct_physical_path(graph, tree, destination)
            if not arc_indexes:
                raise Phase3AOperationError("Nonself sampled OD produced an empty physical path")
            arc_ids = tuple(str(graph.arc_id[index]) for index in arc_indexes)
            path_rows.append(
                {
                    "sample_od_id": str(row.sample_od_id),
                    "origin_idx": int(origin),
                    "destination_idx": destination,
                    "physical_fftt_s": float(tree.distances[destination]),
                    "physical_arc_count": len(arc_indexes),
                    "arc_indices_json": json.dumps(list(map(int, arc_indexes)), separators=(",", ":")),
                    "arc_ids_sha256": hashlib.sha256("\x1f".join(arc_ids).encode("utf-8")).hexdigest(),
                }
            )
    paths = pd.DataFrame(path_rows).sort_values("sample_od_id", kind="mergesort")
    if len(paths) != len(sample) or len(sample) < minimum_od:
        raise Phase3AOperationError("Physical path sample does not close over the OD sample")
    output_root = Path(str(config["output_root"]))
    sample_path = output_root / "round4_evaluation" / "od_sample.parquet"
    paths_path = output_root / "round4_evaluation" / "physical_paths.parquet"
    _write_frame_new(sample_path, sample)
    _write_frame_new(paths_path, paths)
    return {
        "meaningful_output_count": len(sample),
        "artifact_paths": [sample_path, paths_path],
        "unique_od_count": len(sample),
        "unique_origin_count": int(sample["origin_idx"].nunique()),
        "positive_demand": bool((sample["demand"] > 0.0).all()),
        "conus": True,
        "origin_disjoint_split": bool(sample.groupby("origin_uid")["split"].nunique().eq(1).all()),
        "one_physical_tree_per_origin": True,
    }


def _evaluate_one_candidate(
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    graph: GraphData,
    suite: SuiteData,
    candidate_id: str,
    candidate_dir: Path,
    sample: pd.DataFrame,
    physical_paths: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    membership = pd.read_parquet(candidate_dir / "bacn_membership.parquet")
    bundle_membership = pd.read_parquet(candidate_dir / "bundle_membership.parquet")
    bundle_summary = pd.read_parquet(candidate_dir / "bundle_summary.parquet")
    suppressed = pd.read_parquet(candidate_dir / "suppressed_backbone_arcs.parquet")
    terminals = pd.read_parquet(candidate_dir / "consolidated_terminals.parquet")
    terminal_members = pd.read_parquet(candidate_dir / "terminal_fine_members.parquet")
    structural = pd.read_parquet(candidate_dir / "structural_internal_charges.parquet")
    racn_witnesses = pd.read_parquet(candidate_dir / "racn_restricted_witnesses.parquet")
    bacn_map = node_to_bacn(membership)
    bundle_by_arc = dict(
        zip(bundle_membership["directed_arc_id"].astype(str), bundle_membership["bundle_id"].astype(str), strict=True)
    )
    bundle_cost = dict(
        zip(bundle_summary["bundle_id"].astype(str), bundle_summary["direct_diagnostic_fftt_s"].astype(float), strict=True)
    )
    charge = dict(
        zip(structural["bacn_id"].astype(str), structural["structural_charge_s"].astype(float), strict=True)
    )
    uid_to_index = {str(value): index for index, value in enumerate(graph.node_uid.tolist())}
    fine_global = {
        str(terminal_id): tuple(uid_to_index[str(uid)] for uid in group["partC_rev1_node_uid"].astype(str).tolist())
        for terminal_id, group in terminal_members.groupby("terminal_id", sort=False)
    }
    source_candidate_dir = _find_candidate_dir(Path(str(config["output_root"])), candidate_id)
    absorbed_witnesses = _load_absorbed_witnesses(source_candidate_dir, graph, suite)
    internal_aug_patterns = _absorbed_pattern_map(
        candidate_id, candidate_dir, graph, suite, membership
    )
    compiled_aug_patterns = compile_internal_aug_patterns(internal_aug_patterns)
    compiled_terminal_interface = _compile_terminal_interface(terminals, terminal_members)
    path_by_id = physical_paths.set_index("sample_od_id")
    induced_only = bool(contract.get("bounded_path_induced_scalar", str(config.get("execution_scope")) == BOUNDED_TASK1_CONTINUATION))
    full_network = None
    endpoint_interface: _ODEndpointInterface | None = None
    materialized_scalar_paths: dict[str, dict[Hashable, Any | None]] = {}
    evaluation_engine_stats: dict[str, Any] = {
        "scalar_evaluation_backend": "PATH_INDUCED_BOUNDED_SMOKE" if induced_only else "MATERIALIZED_SPARSE_REVERSE_BATCH",
    }
    if not induced_only:
        endpoint_interface = _build_od_endpoint_interface(
            graph=graph,
            suite=suite,
            candidate_id=candidate_id,
            membership=membership,
            terminals=terminals,
            terminal_members=terminal_members,
            sample=sample,
        )
        full_network = _FullCandidateNetwork(
            graph=graph,
            suite=suite,
            candidate_id=candidate_id,
            endpoint_interface=endpoint_interface,
            membership=membership,
            bundle_membership=bundle_membership,
            bundle_summary=bundle_summary,
            suppressed=suppressed,
            terminals=terminals,
            terminal_members=terminal_members,
            racn_witnesses=racn_witnesses,
            structural_charge=charge,
            absorbed_witnesses=absorbed_witnesses,
            regional_oracle_cache_max_regions=int(contract.get("regional_oracle_cache_max_regions", 2)),
            retain_regional_query_cache=bool(contract.get("retain_regional_query_cache", False)),
        )
        materialized_graph = full_network.materialize_scalar_graph()
        if bool(config.get("production_write_support_summary", False)):
            support_summary_path = candidate_dir / "regional_pair_support_summary.parquet"
            _write_frame_new(support_summary_path, full_network.regional_pair_summary)
        scalar_queries = [
            (
                str(row.sample_od_id),
                *endpoint_interface.terminals(str(row.origin_uid), str(row.destination_uid)),
            )
            for row in sample.itertuples(index=False)
        ]
        if len({str(value[0]) for value in scalar_queries}) != len(scalar_queries):
            raise Phase3AOperationError("Shared OD sample IDs are not unique for materialized evaluation")
        for mode, mode_charge in (("direct", {}), ("structural", charge)):
            started = time.perf_counter()
            materialized_scalar_paths[mode] = materialized_scalar_shortest_paths(
                materialized_graph,
                scalar_queries,
                endpoint_charge_s=mode_charge,
                centroid_non_through_nodes=full_network.centroid_nodes,
                target_batch_size=int(contract.get("scalar_target_batch_size", 8)),
            )
            evaluation_engine_stats[f"{mode}_scalar_batch_elapsed_s"] = time.perf_counter() - started
        evaluation_engine_stats.update(endpoint_interface.stats())
        evaluation_engine_stats.update(
            {f"materialized_{key}": value for key, value in full_network.materialization_stats.items()}
        )
        del materialized_graph
        full_network.reset_lazy_state()
        gc.collect()
    region_oracles: dict[str, LazyRegionalPairOracle] = (
        {} if full_network is None else full_network.region_oracles
    )
    records: list[dict[str, Any]] = []
    arc_index_by_id = {str(value): index for index, value in enumerate(graph.arc_id.tolist())}
    ordered_sample = sample.sort_values(["origin_idx", "sample_od_id"], kind="mergesort")
    lazy_state_release_scope = str(contract.get("lazy_state_release_scope", "PER_ORIGIN"))
    if lazy_state_release_scope not in {"PER_ORIGIN", "PER_OD", "PER_DIAGNOSTIC_MODE"}:
        raise Phase3AOperationError("Unknown lazy evaluation state release scope")
    active_origin_idx: int | None = None
    for od_row in ordered_sample.itertuples(index=False):
        origin_idx = int(od_row.origin_idx)
        if active_origin_idx is None:
            active_origin_idx = origin_idx
        elif origin_idx != active_origin_idx:
            if lazy_state_release_scope == "PER_ORIGIN":
                if full_network is not None:
                    full_network.reset_lazy_state()
                region_oracles.clear()
                gc.collect()
            active_origin_idx = origin_idx
        physical = path_by_id.loc[str(od_row.sample_od_id)]
        arc_path = tuple(map(int, json.loads(str(physical.arc_indices_json))))
        terminal_map, pair_ids = _path_terminal_maps(
            graph,
            suite,
            arc_path,
            int(od_row.origin_idx),
            int(od_row.destination_idx),
            terminals,
            terminal_members,
            compiled_aug_patterns=compiled_aug_patterns,
            compiled_terminal_interface=compiled_terminal_interface,
        )
        projection = project_physical_path(
            arc_indices=arc_path,
            arc_id=graph.arc_id,
            tail=graph.tail,
            head=graph.head,
            arc_fftt_s=graph.arc_fftt_s,
            is_final_backbone_arc=suite.is_final_backbone_arc,
            arc_region_id=suite.arc_region_id,
            node_uid=graph.node_uid,
            node_to_bacn=bacn_map,
            bundle_by_arc_id=bundle_by_arc,
            terminal_by_region_node_role=terminal_map,
            regional_pair_id=pair_ids,
            terminal_type_by_id=compiled_terminal_interface.terminal_type,
            terminal_parent_object_by_id=compiled_terminal_interface.terminal_parent,
            compiled_internal_aug_patterns=compiled_aug_patterns,
        )
        object_cost = dict(bundle_cost)
        relations, internal = _relations_from_projection(projection, graph, arc_index_by_id)
        for occurrence in projection.occurrences:
            if occurrence.kind != "REGIONAL_PAIR":
                continue
            region = str(occurrence.region_id)
            if full_network is None:
                if region not in region_oracles:
                    region_oracles[region] = _build_region_oracle(
                        graph, suite, candidate_id, region, terminals, fine_global
                    )
                oracle = region_oracles[region]
            else:
                oracle = full_network.region_oracle(region)
            source_terminal = str(occurrence.coarse_tail_id)
            target_terminal = str(occurrence.coarse_head_id)
            witnesses = oracle.query(source_terminal)
            match = next((value for value in witnesses if value.target_terminal_id == target_terminal), None)
            if match is None:
                raise Phase3AOperationError("Projected regional occurrence lacks frozen direct support")
            if match.pair_id != occurrence.object_id:
                raise Phase3AOperationError("Regional pair stable ID differs between projection and lazy oracle")
            object_cost[occurrence.object_id] = float(match.fftt_s)
        direct_projected = projected_generic_cost(projection, object_cost)
        structural_projected = direct_projected + float(
            structural_half_charge_adjustments(projection.coarse_sequence, charge).sum()
        )
        if lazy_state_release_scope == "PER_DIAGNOSTIC_MODE":
            if full_network is not None:
                full_network.reset_lazy_state()
            region_oracles.clear()
            gc.collect()
        for mode, projected_cost in (("direct", direct_projected), ("structural", structural_projected)):
            mode_charge = {} if mode == "direct" else charge
            if induced_only:
                scalar, lift = _induced_scalar_path_and_lift(
                    relations,
                    internal,
                    int(od_row.origin_idx),
                    int(od_row.destination_idx),
                    mode_charge,
                )
                coarse_edge_ids = scalar.edge_ids if scalar is not None else ()
            else:
                assert full_network is not None
                scalar = materialized_scalar_paths[mode][str(od_row.sample_od_id)]
                if scalar is None:
                    lift = None
                    coarse_edge_ids = ()
                else:
                    coarse_relations = full_network.path_relations(scalar.edge_ids, scalar.node_ids)
                    lift = audit_liftability(
                        coarse_edges=coarse_relations,
                        source_states=[int(od_row.origin_idx)],
                        destination_states=[int(od_row.destination_idx)],
                        internal_relations=full_network.internal_relations,
                        centroid_coarse_nodes=full_network.centroid_nodes,
                    )
                    coarse_edge_ids = scalar.edge_ids
            scalar_cost = None if scalar is None else float(scalar.fftt_s)
            lift_cost = None if lift is None else lift.lifted_fftt_s
            physical_cost = float(physical.physical_fftt_s)
            projected_error = cost_errors(physical_cost, projected_cost)
            scalar_error = cost_errors(physical_cost, scalar_cost)
            reference_arc_ids = [str(graph.arc_id[index]) for index in arc_path]
            lifted_arc_ids = None if lift is None else lift.physical_arc_ids
            similarity_ids = set(reference_arc_ids) | set(lifted_arc_ids or ())
            similarity = weighted_arc_similarity(
                reference_arc_ids,
                lifted_arc_ids,
                {value: float(graph.arc_fftt_s[arc_index_by_id[value]]) for value in similarity_ids},
            )
            records.append(
                {
                    "candidate_id": candidate_id,
                    "sample_od_id": str(od_row.sample_od_id),
                    "diagnostic_mode": mode,
                    "physical_fftt_s": physical_cost,
                    "physical_path_id": str(physical.arc_ids_sha256),
                    "projected_coarse_path_id": hashlib.sha256(canonical_json_bytes(projection.coarse_sequence)).hexdigest(),
                    "projected_generic_fftt_s": float(projected_cost),
                    "scalar_acnhn_path_id": None if scalar is None else hashlib.sha256(canonical_json_bytes(coarse_edge_ids)).hexdigest(),
                    "scalar_acnhn_fftt_s": scalar_cost,
                    "liftable": bool(lift is not None and lift.liftable),
                    "lift_failure_step": None if lift is None else lift.failure_step,
                    "lift_failure_reason": "NO_SCALAR_PATH" if lift is None else lift.failure_reason,
                    "lifted_physical_path_id": None if lift is None or lift.physical_arc_ids is None else hashlib.sha256(canonical_json_bytes(lift.physical_arc_ids)).hexdigest(),
                    "lifted_physical_fftt_s": lift_cost,
                    "projected_signed_error_s": projected_error["signed_error_s"],
                    "projected_relative_error": projected_error["relative_error"],
                    "scalar_signed_error_s": scalar_error["signed_error_s"],
                    "scalar_relative_error": scalar_error["relative_error"],
                    "lifted_regret": lifted_regret(physical_cost, lift_cost),
                    "weighted_arc_precision": similarity["weighted_arc_precision"],
                    "weighted_arc_recall": similarity["weighted_arc_recall"],
                    "weighted_arc_jaccard": similarity["weighted_arc_jaccard"],
                    "coarse_lcs_similarity": lcs_similarity(projection.coarse_sequence, None if scalar is None else scalar.node_ids),
                    "regional_transition_count": projection.regional_transition_count,
                    "repeated_region_entry_count": projection.repeated_region_entry_count,
                    "omitted_suppressed_fftt_s": projection.omitted_suppressed_fftt_s,
                    "omitted_aug_fftt_s": projection.omitted_aug_fftt_s,
                    "omitted_racn_fftt_s": projection.omitted_racn_fftt_s,
                    "runtime_s": 0.0,
                    "evaluation_scope": "PATH_INDUCED_BOUNDED_SMOKE" if induced_only else "FULL_CANDIDATE_LAZY_GRAPH",
                }
            )
            if lazy_state_release_scope == "PER_DIAGNOSTIC_MODE":
                if full_network is not None:
                    full_network.reset_lazy_state()
                region_oracles.clear()
                gc.collect()
        if lazy_state_release_scope == "PER_OD":
            if full_network is not None:
                full_network.reset_lazy_state()
            region_oracles.clear()
            gc.collect()
    if full_network is not None:
        full_network.reset_lazy_state()
    region_oracles.clear()
    gc.collect()
    result = pd.DataFrame(records).sort_values(
        ["candidate_id", "sample_od_id", "diagnostic_mode"], kind="mergesort"
    ).reset_index(drop=True)
    if result.duplicated(["candidate_id", "sample_od_id", "diagnostic_mode"]).any():
        raise Phase3AOperationError("Candidate OD result primary key is duplicated")
    return result, evaluation_engine_stats


def _validate_campaign(config: Mapping[str, Any], contract: Mapping[str, Any]) -> Mapping[str, Any]:
    graph, suite = _load_graph_suite(config)
    output_root = Path(str(config["output_root"]))
    required_operations = set(
        contract.get(
            "required_completed_operations",
            [
                "construct_geo_candidate",
                "census_aug_witness_shard",
                "construct_aug_candidate",
                "finalize_candidate_topology",
                "evaluate_candidate_od_batch",
            ],
        )
    )
    receipts = []
    for path in sorted((output_root / "receipts").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "PASS":
            receipts.append(payload)
    completed = {str(value.get("operation")) for value in receipts}
    missing = required_operations.difference(completed)
    if missing:
        raise Phase3AOperationError(f"Campaign validation lacks completed operations: {sorted(missing)}")
    candidate_ids = _resolve_evaluation_candidate_ids(output_root, contract)
    candidate_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        candidate_dir = output_root / "round3_finalists" / "candidates" / candidate_id
        membership = pd.read_parquet(candidate_dir / "bacn_membership.parquet")
        bundles = pd.read_parquet(candidate_dir / "bundle_membership.parquet")
        suppressed = pd.read_parquet(candidate_dir / "suppressed_backbone_arcs.parquet")
        racn = pd.read_parquet(candidate_dir / "racn_membership.parquet")
        terminals = pd.read_parquet(candidate_dir / "consolidated_terminals.parquet")
        validate_bundle_closure(suite.is_final_backbone_arc, graph.arc_id, bundles, suppressed)
        expected_backbone = set(graph.node_uid[suite.is_final_backbone_node].astype(str).tolist())
        actual_backbone = set(membership["partC_rev1_node_uid"].astype(str).tolist())
        if actual_backbone != expected_backbone or membership["partC_rev1_node_uid"].duplicated().any():
            raise Phase3AOperationError("B-ACN membership does not partition final-backbone nodes exactly")
        if not racn.empty and racn["physical_portal_node_uid"].duplicated().any():
            raise Phase3AOperationError("R-ACN membership duplicates a physical portal")
        centroid = terminals[terminals["terminal_type"].isin(["ORIGIN_CENTROID", "DESTINATION_CENTROID"])]
        if centroid["centroid_through_allowed"].any():
            raise Phase3AOperationError("Centroid through movement is enabled")
        results_path = output_root / "round4_evaluation" / "candidates" / candidate_id / "candidate_od_results.parquet"
        results = pd.read_parquet(results_path)
        modes = set(results["diagnostic_mode"].astype(str))
        if not {"direct", "structural"}.issubset(modes):
            raise Phase3AOperationError("A production finalist lacks direct or structural diagnostic rows")
        if results.duplicated(["candidate_id", "sample_od_id", "diagnostic_mode"]).any():
            raise Phase3AOperationError("Candidate OD result key is not unique")
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "bacn_partition": "PASS",
                "bundle_suppressed_closure": "PASS",
                "racn_unique_membership": "PASS",
                "centroid_non_through": "PASS",
                "direct_structural_modes": "PASS",
                "unique_od_count": int(results["sample_od_id"].nunique()),
            }
        )
    authority = verify_authority(
        resolve_authority_paths(config),
        verify_large_signatures=False,
        expected_identity=config.get("accepted_identities"),
    )
    first_candidate = candidate_ids[0]
    membership = pd.read_parquet(
        output_root / "round3_finalists" / "candidates" / first_candidate / "bacn_membership.parquet"
    )
    representative = membership.groupby("bacn_id", sort=True).first().reset_index().head(20)
    uid_to_index = {str(value): index for index, value in enumerate(graph.node_uid.tolist())}
    rep_indexes = np.asarray([uid_to_index[str(value)] for value in representative["partC_rev1_node_uid"]], dtype=np.int64)
    point_rows = pd.DataFrame(
        {
            "candidate_id": first_candidate,
            "object_id": representative["bacn_id"].astype(str),
            "object_type": "BACN_REPRESENTATIVE",
            "lon": graph.lon[rep_indexes],
            "lat": graph.lat[rep_indexes],
        }
    )
    arc_indexes = np.flatnonzero(suite.is_final_backbone_arc)[:20]
    line_rows = pd.DataFrame(
        {
            "candidate_id": first_candidate,
            "object_id": graph.arc_id[arc_indexes].astype(str),
            "object_type": "FINAL_BACKBONE_ARC_SAMPLE",
            "tail_lon": graph.lon[graph.tail[arc_indexes]],
            "tail_lat": graph.lat[graph.tail[arc_indexes]],
            "head_lon": graph.lon[graph.head[arc_indexes]],
            "head_lat": graph.lat[graph.head[arc_indexes]],
        }
    )
    qgis_root = output_root / "validation" / "qgis"
    qgis = write_overlay_smoke(
        geopackage_path=qgis_root / "phase3a_corrective_overlay_smoke.gpkg",
        qml_path=qgis_root / "phase3a_corrective_overlay_smoke.qml",
        point_rows=point_rows,
        line_rows=line_rows,
    )
    validation = {
        "schema_version": "phase3a-task1-continuation-1.0",
        "status": "PASS",
        "scope": str(config.get("execution_scope")),
        "completed_operations": sorted(completed),
        "candidate_validation": candidate_rows,
        "authority_status": authority["status"],
        "accepted_artifacts_modified": authority["accepted_artifacts_modified"],
        "qgis_reopen": qgis,
        "resource_limits": {
            "process_tree_rss_hard_bytes": 24 * 1024**3,
            "added_disk_hard_bytes": 150 * 1024**3,
        },
        "phase3a_task2_authorized": False,
        "phase3b_authorized": False,
        "phase4_authorized": False,
    }
    path = output_root / "validation" / "campaign_validation.json"
    write_json_new(path, validation)
    artifacts = [
        path,
        qgis_root / "phase3a_corrective_overlay_smoke.gpkg",
        qgis_root / "phase3a_corrective_overlay_smoke.qml",
    ]
    return {
        "meaningful_output_count": len(candidate_rows),
        "artifact_paths": artifacts,
        "candidate_count": len(candidate_rows),
        "qgis_ogr_fiona_pyogrio_reopen": True,
        "accepted_phase0_2_artifacts_modified": False,
        "validation_status": "PASS",
    }


def _reproduce_topologies(config: Mapping[str, Any], contract: Mapping[str, Any]) -> Mapping[str, Any]:
    graph, suite = _load_graph_suite(config)
    output_root = Path(str(config["output_root"]))
    candidate_ids = _resolve_evaluation_candidate_ids(output_root, contract)
    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        finalist_dir = output_root / "round3_finalists" / "candidates" / candidate_id
        finalist_spec = json.loads((finalist_dir / "finalist_spec.json").read_text(encoding="utf-8"))
        source_dir = Path(str(finalist_spec["source_candidate_dir"])).resolve(strict=True)
        observed_membership = pd.read_parquet(finalist_dir / "bacn_membership.parquet")
        reproduced_membership = _recompute_candidate_membership(graph, suite, source_dir)
        observed_hash = deterministic_frame_hash(
            observed_membership, sort_columns=["candidate_id", "partC_rev1_node_uid"]
        )
        reproduced_hash = deterministic_frame_hash(
            reproduced_membership, sort_columns=["candidate_id", "partC_rev1_node_uid"]
        )
        if observed_hash != reproduced_hash:
            raise Phase3AOperationError(f"Fresh B-ACN reproduction differs: {candidate_id}")
        bundles, bundle_summary, suppressed = _bundles_for_membership(
            graph, suite, candidate_id, reproduced_membership
        )
        observed_bundle_hash = deterministic_frame_hash(
            pd.read_parquet(finalist_dir / "bundle_membership.parquet"),
            sort_columns=["candidate_id", "directed_arc_id"],
        )
        reproduced_bundle_hash = deterministic_frame_hash(
            bundles, sort_columns=["candidate_id", "directed_arc_id"]
        )
        if observed_bundle_hash != reproduced_bundle_hash:
            raise Phase3AOperationError(f"Fresh bundle reproduction differs: {candidate_id}")
        validate_bundle_closure(suite.is_final_backbone_arc, graph.arc_id, bundles, suppressed)
        rows.append(
            {
                "candidate_id": candidate_id,
                "bacn_membership_sha256": observed_hash,
                "reproduced_bacn_membership_sha256": reproduced_hash,
                "bundle_membership_sha256": observed_bundle_hash,
                "reproduced_bundle_membership_sha256": reproduced_bundle_hash,
                "bacn_exact": True,
                "bundle_exact": True,
                "suppressed_arc_count": len(suppressed),
                "bundle_count": len(bundle_summary),
            }
        )
    report = {
        "schema_version": "phase3a-task1-continuation-1.0",
        "status": "PASS",
        "fresh_recomputation": True,
        "candidates": rows,
        "stable_id_and_membership_exact": True,
        "shuffled_shard_merge_covered_by_test_suite": True,
    }
    path = output_root / "validation" / "topology_reproduction.json"
    write_json_new(path, report)
    return {
        "meaningful_output_count": len(rows),
        "artifact_paths": [path],
        "candidate_count": len(rows),
        "clean_reproduction": True,
    }


def _seal_handoff(config: Mapping[str, Any], contract: Mapping[str, Any]) -> Mapping[str, Any]:
    output_root = Path(str(config["output_root"])).resolve(strict=True)
    handoff_root = output_root / "handoff"
    handoff_root.mkdir(parents=True, exist_ok=False)
    scope = str(config.get("execution_scope", FULL_TASK2))
    name = str(contract.get("handoff_name", "phase3a_bounded_mini_campaign" if scope == BOUNDED_TASK1_CONTINUATION else "phase3a_campaign"))
    zip_path = handoff_root / f"{name}.zip"
    sidecar = Path(str(zip_path) + ".sha256")
    companion = handoff_root / f"{name}.companion.json"
    if any(path.exists() for path in (zip_path, sidecar, companion)):
        raise Phase3AOperationError("Write-new-only handoff target already exists")
    candidates = _resolve_evaluation_candidate_ids(output_root, contract)
    review_payload_root = output_root / "review_payload"
    if scope == FULL_TASK2 and not review_payload_root.is_dir():
        raise Phase3AOperationError("Full campaign handoff requires the completed review_payload directory")
    payload_paths = (
        sorted(
            path
            for path in review_payload_root.rglob("*")
            if path.is_file() and path.relative_to(review_payload_root).as_posix() != "CHECKSUMS.sha256"
        )
        if review_payload_root.is_dir()
        else []
    )
    evidence_paths = sorted(
        [
            *list((output_root / "receipts").glob("*.json")),
            *list((output_root / "validation").rglob("*")),
            *list((output_root / "startup").glob("*")),
            *list((output_root / "resource" / "incidents").glob("*")),
            output_root / "resource" / "round4_candidate_contract_checks.json",
            output_root / "resource" / "post_heavy_broad_pytest.xml",
            output_root / "round4_evaluation" / "od_sample.parquet",
            output_root / "round4_evaluation" / "physical_paths.parquet",
            *[
                output_root / "round4_evaluation" / "candidates" / candidate / "candidate_od_results.parquet"
                for candidate in candidates
            ],
        ]
    )
    selected_entries: list[tuple[Path, str]] = []
    archive_names: set[str] = set()
    for path in payload_paths:
        archive_name = path.relative_to(review_payload_root).as_posix()
        if archive_name in archive_names:
            raise Phase3AOperationError(f"Duplicate handoff archive member: {archive_name}")
        archive_names.add(archive_name)
        selected_entries.append((path, archive_name))
    for path in evidence_paths:
        if not path.is_file():
            continue
        archive_name = f"evidence/{path.relative_to(output_root).as_posix()}"
        if archive_name in archive_names:
            raise Phase3AOperationError(f"Duplicate handoff archive member: {archive_name}")
        archive_names.add(archive_name)
        selected_entries.append((path, archive_name))
    member_rows = [
        {
            "archive_path": archive_name,
            "external_relative_path": path.relative_to(output_root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": partC_rev1_file_sha256(path),
        }
        for path, archive_name in selected_entries
    ]
    source_commit = _git_head(Path(str(config["project_root"])))
    status_path = review_payload_root / "02_HANDOFF_STATUS.json"
    review_status = "WAITING_FOR_EXTERNAL_REVIEW"
    handoff_id = name
    if status_path.is_file():
        status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        review_status = str(status_payload.get("status", review_status))
        handoff_id = str(status_payload.get("handoff_id", handoff_id))
    external_inventory = review_payload_root / "06_ARTIFACT_MANIFEST.csv"
    manifest = {
        "schema_version": "phase3a-task1-continuation-1.0",
        "scope": scope,
        "status": "BOUNDED_MINI_CAMPAIGN_SEALED" if scope == BOUNDED_TASK1_CONTINUATION else review_status,
        "handoff_id": handoff_id,
        "source_commit": source_commit,
        "members": member_rows,
        "external_output_root": str(output_root),
        "external_output_bytes_before_zip": directory_bytes(output_root),
        "external_artifact_inventory_sha256": (
            partC_rev1_file_sha256(external_inventory) if external_inventory.is_file() else None
        ),
        "candidate_count": len(candidates),
        "winning_method_selected": False,
        "candidate_selection_performed": False,
        "capacity_multiplier_tuned": False,
        "cost_calibration_performed": False,
        "model_training_performed": False,
        "phase3a_task2_authorized": False,
        "phase3b_authorized": False,
        "phase4_authorized": False,
    }
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, archive_name in selected_entries:
            archive.write(path, archive_name)
        archive.writestr("MANIFEST.json", canonical_json_bytes(manifest) + b"\n")
        archive.writestr(
            "CHECKSUMS.sha256",
            "".join(f"{row['sha256']}  {row['archive_path']}\n" for row in member_rows),
        )
    with zipfile.ZipFile(zip_path, "r") as archive:
        if archive.testzip() is not None:
            raise Phase3AOperationError("Fresh handoff ZIP CRC validation failed")
    digest = partC_rev1_file_sha256(zip_path)
    write_text_new(sidecar, f"{digest}  {zip_path.name}")
    companion_payload = {
        "schema_version": "phase3a-task1-continuation-1.0",
        "status": manifest["status"],
        "scope": scope,
        "source_commit": source_commit,
        "handoff_id": handoff_id,
        "zip_path": str(zip_path),
        "zip_size_bytes": int(zip_path.stat().st_size),
        "zip_sha256": digest,
        "fresh_extraction_and_crc": True,
        "external_output_root": str(output_root),
        "external_artifact_inventory_sha256": manifest["external_artifact_inventory_sha256"],
        "candidate_count": len(candidates),
        "winning_method_selected": False,
        "candidate_selection_performed": False,
        "capacity_multiplier_tuned": False,
        "cost_calibration_performed": False,
        "model_training_performed": False,
        "phase3a_task2_authorized": False,
        "phase3b_authorized": False,
        "phase4_authorized": False,
    }
    write_json_new(companion, companion_payload)
    return {
        "meaningful_output_count": len(member_rows),
        "artifact_paths": [zip_path, sidecar, companion],
        "zip_path": str(zip_path),
        "zip_size_bytes": int(zip_path.stat().st_size),
        "zip_sha256": digest,
        "source_commit": source_commit,
        "fresh_crc_pass": True,
    }


def _load_graph_suite(config: Mapping[str, Any]) -> tuple[GraphData, SuiteData]:
    paths = resolve_authority_paths(config)
    graph = load_graph(paths)
    suite = load_suite(paths, graph)
    return graph, suite


def _candidate_rows(config: Mapping[str, Any]) -> list[dict[str, str]]:
    path = Path(str(config["candidate_matrix_path"])).resolve(strict=True)
    return validate_candidate_matrix(path)


def _bundles_for_membership(
    graph: GraphData, suite: SuiteData, candidate_id: str, membership: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return build_backbone_bundles(
        suite_id=suite.suite_id,
        candidate_id=candidate_id,
        final_backbone_arc_mask=suite.is_final_backbone_arc,
        arc_id=graph.arc_id,
        tail_node_uid=graph.node_uid[graph.tail],
        head_node_uid=graph.node_uid[graph.head],
        arc_fftt_s=graph.arc_fftt_s,
        arc_length_m=graph.arc_length_m,
        node_to_bacn_id=node_to_bacn(membership),
    )


def _complete_bacn_summary(
    summary: pd.DataFrame,
    membership: pd.DataFrame,
    suppressed: pd.DataFrame,
    absorbed: Sequence[AugWitness],
) -> pd.DataFrame:
    result = summary.copy()
    suppressed_group = (
        suppressed.groupby("bacn_id", sort=False).agg(
            suppressed_arc_count=("directed_arc_id", "size"),
            suppressed_fftt_s_sum=("fftt_s", "sum"),
            suppressed_length_m_sum=("length_m", "sum"),
        )
        if not suppressed.empty
        else pd.DataFrame(columns=["suppressed_arc_count", "suppressed_fftt_s_sum", "suppressed_length_m_sum"])
    )
    absorbed_by_bacn: dict[str, list[AugWitness]] = defaultdict(list)
    member_to_bacn = node_to_bacn(membership)
    for witness in absorbed:
        bacn_id = member_to_bacn.get(witness.source_node_uid)
        if bacn_id is None or bacn_id != member_to_bacn.get(witness.target_node_uid):
            raise Phase3AOperationError("Absorbed AUG witness endpoints do not share one B-ACN")
        absorbed_by_bacn[bacn_id].append(witness)
    for column, default in (
        ("boundary_state_count", 0),
        ("suppressed_arc_count", 0),
        ("suppressed_fftt_s_sum", 0.0),
        ("suppressed_length_m_sum", 0.0),
        ("internal_aug_witness_count", 0),
        ("internal_aug_fftt_s_sum", 0.0),
        ("internal_aug_length_m_sum", 0.0),
    ):
        result[column] = default
    if not suppressed_group.empty:
        lookup = suppressed_group.to_dict("index")
        for position, bacn_id in enumerate(result["bacn_id"].astype(str)):
            values = lookup.get(bacn_id)
            if values is None:
                continue
            for column in ("suppressed_arc_count", "suppressed_fftt_s_sum", "suppressed_length_m_sum"):
                result.at[position, column] = values[column]
    for position, bacn_id in enumerate(result["bacn_id"].astype(str)):
        values = absorbed_by_bacn.get(bacn_id, ())
        result.at[position, "internal_aug_witness_count"] = len(values)
        result.at[position, "internal_aug_fftt_s_sum"] = sum(value.fftt_s for value in values)
        result.at[position, "internal_aug_length_m_sum"] = sum(value.length_m for value in values)
    return result


def _write_candidate_artifacts(
    candidate_dir: Path,
    *,
    membership: pd.DataFrame,
    summary: pd.DataFrame,
    bundle_membership: pd.DataFrame,
    bundle_summary: pd.DataFrame,
    suppressed: pd.DataFrame,
    spec: Mapping[str, Any],
    absorbed_witnesses: Sequence[AugWitness] = (),
) -> list[Path]:
    artifacts: list[Path] = []
    for name, frame in (
        ("bacn_membership.parquet", membership),
        ("bacn_summary.parquet", summary),
        ("bundle_membership.parquet", bundle_membership),
        ("bundle_summary.parquet", bundle_summary),
        ("suppressed_backbone_arcs.parquet", suppressed),
    ):
        path = candidate_dir / name
        _write_frame_new(path, frame)
        artifacts.append(path)
    if absorbed_witnesses:
        mapping = node_to_bacn(membership)
        rows = []
        for witness in absorbed_witnesses:
            bacn_id = mapping[witness.source_node_uid]
            rows.append(
                {
                    "candidate_id": str(spec["candidate_id"]),
                    "bacn_id": bacn_id,
                    "witness_id": witness.witness_id,
                    "source_node_uid": witness.source_node_uid,
                    "target_node_uid": witness.target_node_uid,
                    "fftt_s": witness.fftt_s,
                    "length_m": witness.length_m,
                    "arc_ids_json": json.dumps(witness.arc_ids, separators=(",", ":")),
                    "arc_indices_json": json.dumps(witness.arc_indices, separators=(",", ":")),
                    "omitted_from_regional_pairs": True,
                    "audit_only": True,
                }
            )
        absorbed_path = candidate_dir / "aug_internal_witnesses.parquet"
        _write_frame_new(absorbed_path, pd.DataFrame(rows))
        artifacts.append(absorbed_path)
        grouped = pd.DataFrame(rows).groupby("bacn_id", sort=False).agg(
            internal_aug_witness_count=("witness_id", "size"),
            internal_aug_fftt_s_sum=("fftt_s", "sum"),
            internal_aug_length_m_sum=("length_m", "sum"),
        )
        updated = pd.read_parquet(candidate_dir / "bacn_summary.parquet")
        for column in grouped.columns:
            updated[column] = updated["bacn_id"].map(grouped[column]).fillna(updated[column])
        # The original file is write-new-only and cannot be replaced.  Persist
        # the fully augmented summary under an explicit audit name.
        audit_summary = candidate_dir / "bacn_summary_with_aug_audit.parquet"
        _write_frame_new(audit_summary, updated)
        artifacts.append(audit_summary)
    spec_path = candidate_dir / "candidate_spec.json"
    write_json_new(spec_path, dict(spec))
    artifacts.append(spec_path)
    hashes = {
        path.name: {"size_bytes": int(path.stat().st_size), "sha256": partC_rev1_file_sha256(path)}
        for path in artifacts
    }
    manifest_path = candidate_dir / "candidate_manifest.json"
    write_json_new(
        manifest_path,
        {
            "schema_version": "phase3a-task1-continuation-1.0",
            "candidate_id": str(spec["candidate_id"]),
            "artifacts": hashes,
            "write_new_only": True,
        },
    )
    artifacts.append(manifest_path)
    return artifacts


def _candidate_registry_row(
    row: Mapping[str, str],
    membership: pd.DataFrame,
    bundles: pd.DataFrame,
    suppressed: pd.DataFrame,
    pair_edge_count: int,
) -> dict[str, Any]:
    member_counts = membership.groupby("bacn_id", sort=False).size().to_numpy(dtype=np.float64)
    final_fftt = float(bundles["member_fftt_s"].sum() + suppressed["fftt_s"].sum())
    suppressed_share = float(suppressed["fftt_s"].sum() / final_fftt) if final_fftt > 0.0 else 0.0
    return {
        "candidate_id": str(row["candidate_id"]),
        "family": str(row["family"]),
        "algorithm": str(row["algorithm"]),
        "parameter_value": float(row["parameter_value"]),
        "parameter_unit": str(row["parameter_unit"]),
        "production_eligible": str(row["production_eligible"]).lower() == "true",
        "valid": True,
        "status": "COMPLETE_TOPOLOGY_ONLY",
        "reason_code": "PASS",
        "bacn_count": int(membership["bacn_id"].nunique()),
        "bundle_count": int(bundles["bundle_id"].nunique()) if not bundles.empty else 0,
        "suppressed_arc_count": len(suppressed),
        "suppressed_fftt_share": suppressed_share,
        "p99_member_count": float(np.quantile(member_counts, 0.99)) if member_counts.size else 0.0,
        "projected_full_build_seconds": 0.0,
        "pair_edge_count": int(pair_edge_count),
    }


def _write_frame_new(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Phase3AOperationError(f"Write-new-only table path exists: {path}")
    frame.to_parquet(path, index=False)


def _aug_policy(config: Mapping[str, Any], policy_id: str) -> AugPolicy:
    text = str(policy_id).lower()
    key = "conservative" if "conservative" in text or "30" in text else "moderate" if "moderate" in text or "120" in text else text
    policies = config.get("aug", {}).get("policies", {})
    if key not in policies:
        raise Phase3AOperationError(f"Unknown frozen AUG policy: {policy_id}")
    row = policies[key]
    return AugPolicy(
        policy_id=key,
        threshold_s=float(row.get("threshold_s_each_direction", row.get("threshold_s"))),
        search_radius_m=1000.0 * float(row.get("search_radius_km")),
        max_physical_arcs=int(row.get("max_physical_arcs")),
        max_length_geodesic_detour=float(row.get("max_length_geodesic_detour")),
        epsilon_distance_m=0.01,
    )


def _aug_policy_for_threshold(config: Mapping[str, Any], threshold: float) -> AugPolicy:
    return _aug_policy(config, "conservative" if abs(float(threshold) - 30.0) <= 1e-9 else "moderate")


def _resolve_source_indexes(
    graph: GraphData, suite: SuiteData, contract: Mapping[str, Any]
) -> list[int]:
    uid_values = contract.get("source_node_uids")
    if uid_values is not None:
        uid_to_index = {str(value): index for index, value in enumerate(graph.node_uid.tolist())}
        try:
            sources = [uid_to_index[str(value)] for value in uid_values]
        except KeyError as exc:
            raise Phase3AOperationError(f"AUG source stable ID is absent: {exc}") from exc
    elif contract.get("source_registry_path") is not None:
        registry = pd.read_parquet(Path(str(contract["source_registry_path"])).resolve(strict=True))
        if "source_node_uid" not in registry.columns:
            raise Phase3AOperationError("AUG source registry lacks source_node_uid")
        if "combined_rank" in registry.columns:
            registry = registry.sort_values("combined_rank", kind="mergesort")
        limit = contract.get("source_limit")
        if limit is not None:
            registry = registry.head(int(limit))
        uid_to_index = {str(value): index for index, value in enumerate(graph.node_uid.tolist())}
        sources = [uid_to_index[str(value)] for value in registry["source_node_uid"].astype(str)]
    elif contract.get("source_node_indexes") is not None:
        sources = list(map(int, contract["source_node_indexes"]))
    else:
        sources = np.flatnonzero(suite.is_final_backbone_node).astype(np.int64).tolist()
    sources = sorted(set(sources), key=lambda index: _id_key(str(graph.node_uid[index])))
    if not sources or not np.all(suite.is_final_backbone_node[np.asarray(sources, dtype=np.int64)]):
        raise Phase3AOperationError("AUG source shard must be nonempty and final-backbone-only")
    return sources


def _write_witness_artifacts(
    shard_dir: Path,
    witnesses: Sequence[AugWitness],
    stats: Sequence[Any],
    graph: GraphData,
    policy: AugPolicy,
) -> list[Path]:
    registry_rows: list[dict[str, Any]] = []
    arc_rows: list[dict[str, Any]] = []
    offset = 0
    for witness in witnesses:
        begin = offset
        for arc_index, arc_id in zip(witness.arc_indices, witness.arc_ids, strict=True):
            arc_rows.append(
                {
                    "witness_id": witness.witness_id,
                    "arc_offset": offset,
                    "directed_arc_id": str(arc_id),
                    "arc_index": int(arc_index),
                }
            )
            offset += 1
        registry_rows.append(
            {
                "witness_id": witness.witness_id,
                "policy_id": witness.policy_id,
                "source_node_idx": witness.source_node_idx,
                "target_node_idx": witness.target_node_idx,
                "source_node_uid": witness.source_node_uid,
                "target_node_uid": witness.target_node_uid,
                "fftt_s": witness.fftt_s,
                "length_m": witness.length_m,
                "arc_count": len(witness.arc_indices),
                "detour_ratio": witness.detour_ratio,
                "arc_offset_begin": begin,
                "arc_offset_end": offset,
                "crossed_region_ids_json": json.dumps(witness.crossed_region_ids, separators=(",", ":")),
                "canonical_rank_fields": json.dumps(
                    [witness.fftt_s, witness.length_m, len(witness.arc_ids), list(witness.arc_ids)], separators=(",", ":")
                ),
                "witness_sha256": witness.witness_sha256,
            }
        )
    registry = pd.DataFrame(registry_rows, columns=[
        "witness_id", "policy_id", "source_node_idx", "target_node_idx", "source_node_uid",
        "target_node_uid", "fftt_s", "length_m", "arc_count", "detour_ratio",
        "arc_offset_begin", "arc_offset_end", "crossed_region_ids_json",
        "canonical_rank_fields", "witness_sha256",
    ])
    arc_vector = pd.DataFrame(arc_rows, columns=["witness_id", "arc_offset", "directed_arc_id", "arc_index"])
    stats_frame = pd.DataFrame([asdict(value) for value in stats])
    registry_path = shard_dir / "aug_witness_registry.parquet"
    arc_path = shard_dir / "aug_witness_arc_vector.parquet"
    stats_path = shard_dir / "aug_witness_search_stats.parquet"
    _write_frame_new(registry_path, registry)
    _write_frame_new(arc_path, arc_vector)
    _write_frame_new(stats_path, stats_frame)
    metadata_path = shard_dir / "witness_shard.json"
    write_json_new(
        metadata_path,
        {
            "schema_version": "phase3a-task1-continuation-1.0",
            "policy": asdict(policy),
            "source_count": len(stats),
            "witness_count": len(witnesses),
            "arc_vector_count": len(arc_vector),
            "graph_identity": graph.validation["graph_identity"],
        },
    )
    return [registry_path, arc_path, stats_path, metadata_path]


def _group_aug_rows(rows: Sequence[Mapping[str, str]]) -> dict[float, list[Mapping[str, str]]]:
    result: dict[float, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        result[float(row["parameter_value"])].append(row)
    return dict(sorted(result.items()))


def _load_policy_witnesses(
    output_root: Path, policy_id: str, graph: GraphData, suite: SuiteData
) -> tuple[tuple[AugWitness, ...], list[Path]]:
    policy_root = output_root / "round2_aug" / "witness_cache" / policy_id
    registry_paths = sorted(policy_root.rglob("aug_witness_registry.parquet"))
    witnesses: list[AugWitness] = []
    artifacts: list[Path] = []
    for registry_path in registry_paths:
        loaded, paths = _read_witness_directory(registry_path.parent, graph, suite)
        witnesses.extend(loaded)
        artifacts.extend(paths)
    witnesses.sort(key=lambda value: (_id_key(value.source_node_uid), _id_key(value.target_node_uid), value.witness_id))
    keys = [(value.policy_id, value.source_node_uid, value.target_node_uid) for value in witnesses]
    if len(keys) != len(set(keys)):
        raise Phase3AOperationError("Merged witness cache contains duplicate directed source-target keys")
    return tuple(witnesses), artifacts


def _read_witness_directory(
    directory: Path, graph: GraphData, suite: SuiteData
) -> tuple[tuple[AugWitness, ...], list[Path]]:
    registry_path = directory / "aug_witness_registry.parquet"
    arc_path = directory / "aug_witness_arc_vector.parquet"
    registry = pd.read_parquet(registry_path)
    arcs = pd.read_parquet(arc_path)
    result: list[AugWitness] = []
    for row in registry.itertuples(index=False):
        begin, end = int(row.arc_offset_begin), int(row.arc_offset_end)
        segment = arcs.iloc[begin:end]
        arc_indices = tuple(map(int, segment["arc_index"].tolist()))
        arc_ids = tuple(map(str, segment["directed_arc_id"].tolist()))
        if tuple(graph.arc_id[np.asarray(arc_indices, dtype=np.int64)].astype(str).tolist()) != arc_ids:
            raise Phase3AOperationError("Witness flat arc vector differs from authoritative arc identities")
        result.append(
            AugWitness(
                witness_id=str(row.witness_id),
                policy_id=str(row.policy_id),
                source_node_idx=int(row.source_node_idx),
                target_node_idx=int(row.target_node_idx),
                source_node_uid=str(row.source_node_uid),
                target_node_uid=str(row.target_node_uid),
                fftt_s=float(row.fftt_s),
                length_m=float(row.length_m),
                arc_indices=arc_indices,
                arc_ids=arc_ids,
                crossed_region_ids=tuple(map(str, json.loads(str(row.crossed_region_ids_json)))),
                detour_ratio=float(row.detour_ratio),
                witness_sha256=str(row.witness_sha256),
            )
        )
    return tuple(result), [registry_path, arc_path, directory / "aug_witness_search_stats.parquet", directory / "witness_shard.json"]


class _AugCompleteDistanceProvider:
    """Exact targeted AUG distances with bounded working memory.

    Single-link components require exact diameter evidence for pairs absent from
    the threshold-bounded pair graph.  Retaining one national distance vector
    per encountered source grows linearly to tens of gigabytes.  This provider
    reuses one vector and stops each Dijkstra as soon as the small component's
    requested targets are settled.  Every ordered source/target distance is
    still evaluated, so the resulting diameter is mathematically unchanged.
    """

    def __init__(self, graph: GraphData, adjacency: TemporaryAugmentedAdjacency) -> None:
        self.adjacency = adjacency
        self.uid_to_local = {
            str(graph.node_uid[global_index]): local
            for local, global_index in enumerate(adjacency.backbone_node_indices.tolist())
        }
        self.distance = np.full(adjacency.node_count, np.inf, dtype=np.float64)

    def _maximum_directed_distance(self, source: int, targets: Sequence[int]) -> float:
        remaining = {int(value) for value in targets if int(value) != source}
        if not remaining:
            return 0.0
        touched = [source]
        self.distance[source] = 0.0
        heap: list[tuple[float, int]] = [(0.0, source)]
        maximum = 0.0
        try:
            while heap and remaining:
                value, node = heapq.heappop(heap)
                if abs(value - self.distance[node]) > 1e-12:
                    continue
                if node in remaining:
                    remaining.remove(node)
                    maximum = max(maximum, float(value))
                for position in range(
                    int(self.adjacency.indptr[node]), int(self.adjacency.indptr[node + 1])
                ):
                    target = int(self.adjacency.head_local[position])
                    candidate = value + float(self.adjacency.cost_s[position])
                    if candidate < self.distance[target] - 1e-12:
                        if not np.isfinite(self.distance[target]):
                            touched.append(target)
                        self.distance[target] = candidate
                        heapq.heappush(heap, (candidate, target))
            return float("inf") if remaining else maximum
        finally:
            self.distance[np.asarray(touched, dtype=np.int64)] = np.inf

    def __call__(self, left: str, right: str) -> float:
        source, target = self.uid_to_local[str(left)], self.uid_to_local[str(right)]
        return max(
            self._maximum_directed_distance(source, (target,)),
            self._maximum_directed_distance(target, (source,)),
        )

    def cluster_diameter(self, node_uids: Sequence[str]) -> float:
        members = tuple(self.uid_to_local[str(value)] for value in node_uids)
        maximum = 0.0
        for source in members:
            maximum = max(maximum, self._maximum_directed_distance(source, members))
            if np.isinf(maximum):
                return maximum
        return maximum


def _aug_complete_distance_provider(
    graph: GraphData, adjacency: TemporaryAugmentedAdjacency
) -> _AugCompleteDistanceProvider:
    return _AugCompleteDistanceProvider(graph, adjacency)


def _resolve_finalist_ids(output_root: Path, contract: Mapping[str, Any]) -> list[str]:
    if contract.get("candidate_ids") is not None:
        result = sorted(set(map(str, contract["candidate_ids"])))
    else:
        admission_path = output_root / "round3_finalists" / "candidate_admission.parquet"
        if not admission_path.is_file():
            raise Phase3AOperationError("Full finalization requires explicit candidates or an admission artifact")
        admission = pd.read_parquet(admission_path)
        result = admission.loc[
            admission["disposition"].astype(str).str.startswith("ADMITTED"), "candidate_id"
        ].astype(str).tolist()
    if not result:
        raise Phase3AOperationError("Finalization candidate list is empty")
    return result


def _find_candidate_dir(output_root: Path, candidate_id: str) -> Path:
    matches = [
        output_root / "round1_geo" / "candidates" / candidate_id,
        output_root / "round2_aug" / "candidates" / candidate_id,
    ]
    existing = [path for path in matches if (path / "candidate_spec.json").is_file()]
    if len(existing) != 1:
        raise Phase3AOperationError(f"Candidate source directory is absent or ambiguous: {candidate_id}")
    return existing[0]


class _RestrictedPortalProvider:
    def __init__(self, graph: GraphData, suite: SuiteData) -> None:
        self.graph = graph
        self.suite = suite
        self.centroids = frozenset(map(int, suite.centroid_roles["node_idx"].to_numpy(dtype=np.int64)))
        self.backbone = frozenset(np.flatnonzero(suite.is_final_backbone_node).astype(np.int64).tolist())
        self.cache: dict[frozenset[str], Any] = {}

    def __call__(
        self, source: int, target: int, regions: frozenset[str]
    ) -> tuple[float, tuple[int, ...]] | None:
        if regions not in self.cache:
            mask = (~self.suite.is_final_backbone_arc) & np.isin(self.suite.arc_region_id, list(regions))
            if not mask.any():
                return None
            self.cache[regions] = build_sparse_directed_graph(
                tail=self.graph.tail,
                head=self.graph.head,
                cost_s=self.graph.arc_fftt_s,
                arc_stable_id=self.graph.arc_id,
                arc_mask=mask,
            )
        restricted = self.cache[regions]
        local = {value: index for index, value in enumerate(restricted.node_ids)}
        if int(source) not in local or int(target) not in local:
            return None
        tree = restricted_dijkstra(
            restricted,
            [int(source)],
            maximum_distance_s=120.0,
            target_global_nodes=[int(target)],
            prohibited_through_global_nodes=self.backbone | self.centroids,
        )
        target_local = local[int(target)]
        if not np.isfinite(tree.distances_s[target_local]):
            return None
        path = reconstruct_restricted_path(restricted, tree, int(target))
        return float(tree.distances_s[target_local]), path


def _construct_racns(
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    graph: GraphData,
    suite: SuiteData,
    candidate_id: str,
    bacn_map: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    uid_to_index = {str(value): index for index, value in enumerate(graph.node_uid.tolist())}
    centroid_uids = set(suite.centroid_roles["partC_rev1_node_uid"].astype(str))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in suite.portals.itertuples(index=False):
        uid = str(row.partC_rev1_node_uid)
        if uid in bacn_map or uid in centroid_uids:
            continue
        grouped[uid].append(
            {"region_id": str(row.region_uuid), "can_enter": bool(row.can_enter), "can_exit": bool(row.can_exit)}
        )
    uids = sorted(grouped, key=_id_key)
    signatures = [canonical_portal_signature(grouped[uid]) for uid in uids]
    indexes = np.asarray([uid_to_index[uid] for uid in uids], dtype=np.int64)
    if not uids:
        empty, summary, witnesses = build_fixed_racns(
            suite_id=suite.suite_id,
            candidate_id=candidate_id,
            portal_node_uid=[],
            portal_node_idx=np.empty(0, dtype=np.int64),
            lon=np.empty(0),
            lat=np.empty(0),
            weak_component=np.empty(0, dtype=np.int64),
            signatures=[],
            restricted_distance=lambda *_: None,
        )
        return empty, summary, witnesses, {"eligible_portal_count": 0, "actual_fixed_rule_portal_count": 0}
    singleton_membership, singleton_summary, _ = build_fixed_racns(
        suite_id=suite.suite_id,
        candidate_id=candidate_id,
        portal_node_uid=uids,
        portal_node_idx=indexes,
        lon=graph.lon[indexes],
        lat=graph.lat[indexes],
        weak_component=graph.physical_weak_component[indexes],
        signatures=signatures,
        restricted_distance=lambda *_: None,
        singleton_control=True,
    )
    scoped = str(config.get("execution_scope", FULL_TASK2)) == BOUNDED_TASK1_CONTINUATION
    if scoped:
        cap = int(contract.get("racn_fixed_rule_portal_cap", 64))
        by_signature: dict[Any, list[str]] = defaultdict(list)
        for uid, signature in zip(uids, signatures, strict=True):
            by_signature[signature].append(uid)
        ranked = sorted(
            by_signature.items(),
            key=lambda item: (-len(item[1]), canonical_json_bytes(item[0])),
        )
        selected_uids = sorted(ranked[0][1], key=_id_key)[:cap]
    else:
        selected_uids = uids
    selected_set = set(selected_uids)
    selected_positions = [position for position, uid in enumerate(uids) if uid in selected_set]
    provider = _RestrictedPortalProvider(graph, suite)
    fixed_membership, fixed_summary, fixed_witnesses = build_fixed_racns(
        suite_id=suite.suite_id,
        candidate_id=candidate_id,
        portal_node_uid=selected_uids,
        portal_node_idx=indexes[np.asarray(selected_positions, dtype=np.int64)],
        lon=graph.lon[indexes[np.asarray(selected_positions, dtype=np.int64)]],
        lat=graph.lat[indexes[np.asarray(selected_positions, dtype=np.int64)]],
        weak_component=graph.physical_weak_component[indexes[np.asarray(selected_positions, dtype=np.int64)]],
        signatures=[signatures[position] for position in selected_positions],
        restricted_distance=provider,
    )
    remaining_membership = singleton_membership[
        ~singleton_membership["physical_portal_node_uid"].astype(str).isin(selected_set)
    ]
    removed_ids = set(
        singleton_membership.loc[
            singleton_membership["physical_portal_node_uid"].astype(str).isin(selected_set), "racn_id"
        ].astype(str)
    )
    remaining_summary = singleton_summary[~singleton_summary["racn_id"].astype(str).isin(removed_ids)]
    membership = pd.concat([fixed_membership, remaining_membership], ignore_index=True).sort_values(
        ["candidate_id", "physical_portal_node_uid"], kind="mergesort"
    ).reset_index(drop=True)
    summary = pd.concat([fixed_summary, remaining_summary], ignore_index=True).sort_values(
        ["candidate_id", "racn_id"], kind="mergesort"
    ).reset_index(drop=True)
    if membership["physical_portal_node_uid"].duplicated().any() or set(membership["physical_portal_node_uid"].astype(str)) != set(uids):
        raise Phase3AOperationError("Fixed R-ACN plus singleton closure failed")
    evidence = {
        "exact_incident_directional_signature": True,
        "same_component": True,
        "complete_link_geodesic_diameter_m": 500.0,
        "restricted_mutual_fftt_s_each_direction": 120.0,
        "no_intermediate_backbone_or_centroid": True,
        "eligible_portal_count": len(uids),
        "actual_fixed_rule_portal_count": len(selected_uids),
        "actual_fixed_rule_portal_uids": selected_uids,
        "singleton_remainder_count": len(uids) - len(selected_uids),
        "bounded_scope": scoped,
    }
    return membership, summary, fixed_witnesses, evidence


def _resolve_support_regions(
    suite: SuiteData, contract: Mapping[str, Any], config: Mapping[str, Any]
) -> list[str]:
    if contract.get("region_ids") is not None:
        result = sorted(set(map(str, contract["region_ids"])), key=_id_key)
    elif str(config.get("execution_scope", FULL_TASK2)) == FULL_TASK2:
        result = list(suite.region_ids)
    else:
        counts = suite.directed_arc_membership[
            suite.directed_arc_membership["suite_label"].astype(str) != "backbone"
        ].groupby("suite_label", sort=True).size().sort_values()
        positions = sorted(set([0, len(counts) // 2, max(0, int(0.95 * (len(counts) - 1))), len(counts) - 1]))
        result = [str(counts.index[position]) for position in positions]
    unknown = set(result).difference(suite.region_ids)
    if not result or unknown:
        raise Phase3AOperationError(f"Regional support scope is empty or unknown: {sorted(unknown)}")
    return result


def _write_region_support_new(path: Path, support: RegionSupport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Phase3AOperationError(f"Write-new-only regional support path exists: {path}")
    condensation_offsets = np.zeros(support.scc_count + 1, dtype=np.int64)
    condensation_targets: list[int] = []
    for position, values in enumerate(support.condensation_out, start=1):
        condensation_targets.extend(map(int, values))
        condensation_offsets[position] = len(condensation_targets)
    np.savez_compressed(
        path,
        region_id=np.asarray(support.region_id),
        node_ids=np.asarray(support.graph.node_ids, dtype=np.int64),
        indptr=support.graph.indptr,
        head_local=support.graph.head_local,
        arc_index=support.graph.arc_index,
        cost_s=support.graph.cost_s,
        scc_label=support.scc_label,
        condensation_offsets=condensation_offsets,
        condensation_targets=np.asarray(condensation_targets, dtype=np.int64),
    )


def _load_absorbed_witnesses(
    candidate_dir: Path, graph: GraphData, suite: SuiteData
) -> tuple[AugWitness, ...]:
    path = candidate_dir / "aug_internal_witnesses.parquet"
    if not path.is_file():
        return ()
    frame = pd.read_parquet(path)
    result = []
    for row in frame.itertuples(index=False):
        arc_indices = tuple(map(int, json.loads(str(row.arc_indices_json))))
        arc_ids = tuple(map(str, json.loads(str(row.arc_ids_json))))
        result.append(
            AugWitness(
                witness_id=str(row.witness_id),
                policy_id="absorbed",
                source_node_idx=int(graph.tail[arc_indices[0]]),
                target_node_idx=int(graph.head[arc_indices[-1]]),
                source_node_uid=str(row.source_node_uid),
                target_node_uid=str(row.target_node_uid),
                fftt_s=float(row.fftt_s),
                length_m=float(row.length_m),
                arc_indices=arc_indices,
                arc_ids=arc_ids,
                crossed_region_ids=tuple(
                    sorted({str(suite.arc_region_id[index]) for index in arc_indices})
                ),
                detour_ratio=0.0,
                witness_sha256=hashlib.sha256("\x1f".join(arc_ids).encode("utf-8")).hexdigest(),
            )
        )
    return tuple(result)


def _structural_charge_table(
    candidate_id: str,
    bacn_summary: pd.DataFrame,
    suppressed: pd.DataFrame,
    absorbed: Sequence[AugWitness],
    bacn_map: Mapping[str, str],
) -> pd.DataFrame:
    values: dict[str, list[float]] = defaultdict(list)
    for row in suppressed.itertuples(index=False):
        values[str(row.bacn_id)].append(float(row.fftt_s))
    for witness in absorbed:
        bacn_id = bacn_map.get(witness.source_node_uid)
        if bacn_id is None or bacn_id != bacn_map.get(witness.target_node_uid):
            raise Phase3AOperationError("Structural AUG observation is not internal to one B-ACN")
        values[bacn_id].append(float(witness.fftt_s))
    rows = []
    for bacn_id in bacn_summary["bacn_id"].astype(str).tolist():
        observations = values.get(bacn_id, [])
        rows.append(
            {
                "candidate_id": candidate_id,
                "bacn_id": bacn_id,
                "feasible_directed_internal_observation_count": len(observations),
                "structural_charge_s": float(np.median(observations)) if observations else 0.0,
                "base_statistic": "median_feasible_directed_internal_fftt",
                "half_arc_allocation": True,
                "direct_mode_internal_cost_s": 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("bacn_id", kind="mergesort").reset_index(drop=True)


def _resolve_evaluation_candidate_ids(output_root: Path, contract: Mapping[str, Any]) -> list[str]:
    if contract.get("candidate_ids") is not None:
        values = sorted(set(map(str, contract["candidate_ids"])))
    else:
        values = sorted(
            path.name
            for path in (output_root / "round3_finalists" / "candidates").glob("*")
            if (path / "finalist_spec.json").is_file()
        )
    if not values:
        raise Phase3AOperationError("Evaluation/review candidate list is empty")
    for candidate in values:
        if not (output_root / "round3_finalists" / "candidates" / candidate / "finalist_spec.json").is_file():
            raise Phase3AOperationError(f"Finalist artifact is missing: {candidate}")
    return values


@dataclass(frozen=True, slots=True)
class _CompiledTerminalInterface:
    candidate_id: str
    terminal_type: dict[str, str]
    terminal_parent: dict[str, str]
    terminal_direction: dict[str, dict[str, Any]]
    by_region_uid: dict[tuple[str, str], tuple[str, ...]]


def _compile_terminal_interface(
    terminals: pd.DataFrame, terminal_members: pd.DataFrame
) -> _CompiledTerminalInterface:
    if terminals.empty:
        raise Phase3AOperationError("Terminal interface may not be empty")
    terminal_type = terminals.set_index("terminal_id")["terminal_type"].astype(str).to_dict()
    terminal_parent = (
        terminals.set_index("terminal_id")["parent_object_id"].astype(str).to_dict()
        if "parent_object_id" in terminals.columns
        else {}
    )
    terminal_region = terminals.set_index("terminal_id")["region_id"].astype(str).to_dict()
    terminal_direction = terminals.set_index("terminal_id")[[
        "can_enter_region", "can_exit_region"
    ]].to_dict("index")
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in terminal_members.itertuples(index=False):
        terminal_id = str(row.terminal_id)
        region = terminal_region.get(terminal_id)
        if region is None:
            raise Phase3AOperationError("Terminal member references an absent terminal")
        grouped[(region, str(row.partC_rev1_node_uid))].append(terminal_id)
    return _CompiledTerminalInterface(
        candidate_id=str(terminals["candidate_id"].iloc[0]),
        terminal_type=terminal_type,
        terminal_parent=terminal_parent,
        terminal_direction=terminal_direction,
        by_region_uid={key: tuple(sorted(values)) for key, values in grouped.items()},
    )


def _path_terminal_maps(
    graph: GraphData,
    suite: SuiteData,
    arc_path: Sequence[int],
    origin_idx: int,
    destination_idx: int,
    terminals: pd.DataFrame,
    terminal_members: pd.DataFrame,
    *,
    compiled_aug_patterns: Mapping[
        str, Sequence[tuple[tuple[str, ...], str]]
    ] | None = None,
    compiled_terminal_interface: _CompiledTerminalInterface | None = None,
) -> tuple[dict[tuple[str, str, str], str], dict[tuple[str, str, str], str]]:
    interface = compiled_terminal_interface or _compile_terminal_interface(terminals, terminal_members)
    result: dict[tuple[str, str, str], str] = {}
    pair_ids: dict[tuple[str, str, str], str] = {}
    path_values = tuple(map(int, arc_path))
    aug_patterns = compiled_aug_patterns or {}
    position = 0
    while position < len(path_values):
        aug_match = match_internal_aug_pattern(
            path_values,
            position,
            graph.arc_id,
            aug_patterns,
        )
        if aug_match is not None:
            position += len(aug_match[0])
            continue
        arc = int(path_values[position])
        if suite.is_final_backbone_arc[arc]:
            position += 1
            continue
        region = str(suite.arc_region_id[arc])
        end = position + 1
        while (
            end < len(path_values)
            and not suite.is_final_backbone_arc[int(path_values[end])]
            and str(suite.arc_region_id[int(path_values[end])]) == region
            and match_internal_aug_pattern(
                path_values,
                end,
                graph.arc_id,
                aug_patterns,
            ) is None
        ):
            end += 1
        source_index = int(graph.tail[int(path_values[position])])
        target_index = int(graph.head[int(path_values[end - 1])])
        source_uid = str(graph.node_uid[source_index])
        target_uid = str(graph.node_uid[target_index])
        source_terminal = _select_directional_terminal(
            interface.by_region_uid.get((region, source_uid), ()),
            interface.terminal_type,
            interface.terminal_direction,
            role="source", prefer_centroid=source_index == int(origin_idx),
        )
        target_terminal = _select_directional_terminal(
            interface.by_region_uid.get((region, target_uid), ()),
            interface.terminal_type,
            interface.terminal_direction,
            role="target", prefer_centroid=target_index == int(destination_idx),
        )
        result[(region, source_uid, "source")] = source_terminal
        result[(region, target_uid, "target")] = target_terminal
        pair_id = stable_object_id(
            "regional_pair",
            suite_id=suite.suite_id,
            candidate_id=interface.candidate_id,
            members=[source_terminal, target_terminal],
            ordered_members=True,
            signature={"region_id": region, "cost_version": "direct_fftt_v1"},
        )
        pair_ids[(region, source_terminal, target_terminal)] = pair_id
        position = end
    return result, pair_ids


def _select_directional_terminal(
    candidates: Sequence[str],
    terminal_type: Mapping[str, str],
    terminal_direction: Mapping[str, Mapping[str, Any]],
    *,
    role: str,
    prefer_centroid: bool,
) -> str:
    required = "can_exit_region" if role == "source" else "can_enter_region"
    eligible = [value for value in candidates if bool(terminal_direction[value][required])]
    preferred_type = "ORIGIN_CENTROID" if role == "source" else "DESTINATION_CENTROID"
    if prefer_centroid:
        preferred = [value for value in eligible if terminal_type[value] == preferred_type]
        if len(preferred) == 1:
            return preferred[0]
    physical = [value for value in eligible if terminal_type[value] in {"BACN", "RACN"}]
    if len(physical) == 1:
        return physical[0]
    if len(eligible) == 1:
        return eligible[0]
    raise Phase3AOperationError(f"Path endpoint maps to {len(eligible)} {role} terminals")


def _absorbed_pattern_map(
    candidate_id: str,
    candidate_dir: Path,
    graph: GraphData,
    suite: SuiteData,
    membership: pd.DataFrame,
) -> dict[tuple[str, ...], str]:
    del candidate_id, graph, suite
    source_spec = json.loads((candidate_dir / "finalist_spec.json").read_text(encoding="utf-8"))
    source_dir = Path(str(source_spec["source_candidate_dir"]))
    path = source_dir / "aug_internal_witnesses.parquet"
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path)
    mapping = node_to_bacn(membership)
    return {
        tuple(map(str, json.loads(str(row.arc_ids_json)))): mapping[str(row.source_node_uid)]
        for row in frame.itertuples(index=False)
    }


def _relations_from_projection(
    projection: Any,
    graph: GraphData,
    arc_index_by_id: Mapping[str, int],
) -> tuple[list[CoarseEdgeRelation], dict[str, list[PhysicalTransition]]]:
    edges: list[CoarseEdgeRelation] = []
    internal: dict[str, list[PhysicalTransition]] = defaultdict(list)
    previous_head: str | None = None
    previous_state: int | None = None
    for occurrence in projection.occurrences:
        indexes = [arc_index_by_id[str(value)] for value in occurrence.physical_arc_ids]
        tail_state = int(graph.tail[indexes[0]])
        head_state = int(graph.head[indexes[-1]])
        transition = PhysicalTransition(
            tail_state,
            head_state,
            float(occurrence.physical_fftt_s),
            tuple(map(str, occurrence.physical_arc_ids)),
        )
        if occurrence.omitted_online:
            coarse = str(occurrence.object_id)
            internal[coarse].append(transition)
            if previous_head is not None and previous_head != coarse:
                bridge_state = tail_state if previous_state is None else previous_state
                edges.append(
                    CoarseEdgeRelation(
                        f"bridge:{len(edges)}",
                        previous_head,
                        coarse,
                        (PhysicalTransition(bridge_state, tail_state, 0.0, ()),),
                    )
                )
            previous_head = coarse
            previous_state = head_state
            continue
        tail_coarse = str(occurrence.coarse_tail_id)
        head_coarse = str(occurrence.coarse_head_id)
        if previous_head is not None and previous_head != tail_coarse:
            bridge_state = tail_state if previous_state is None else previous_state
            edges.append(
                CoarseEdgeRelation(
                    f"bridge:{len(edges)}",
                    previous_head,
                    tail_coarse,
                    (PhysicalTransition(bridge_state, tail_state, 0.0, ()),),
                )
            )
        edges.append(
            CoarseEdgeRelation(
                f"{occurrence.kind}:{occurrence.occurrence_index}",
                tail_coarse,
                head_coarse,
                (transition,),
            )
        )
        previous_head = head_coarse
        previous_state = head_state
    return edges, dict(internal)


def _build_region_oracle(
    graph: GraphData,
    suite: SuiteData,
    candidate_id: str,
    region: str,
    terminals: pd.DataFrame,
    fine_global: Mapping[str, tuple[int, ...]],
    *,
    retain_query_cache: bool = True,
) -> LazyRegionalPairOracle:
    support = build_region_support(
        region_id=region,
        tail=graph.tail,
        head=graph.head,
        fftt_s=graph.arc_fftt_s,
        arc_ids=graph.arc_id,
        region_arc_mask=suite.arc_region_id == region,
    )
    region_terminals = terminals[terminals["region_id"].astype(str) == region].reset_index(drop=True)
    centroid_nodes = frozenset(
        int(value)
        for value in suite.centroid_roles.loc[
            suite.centroid_roles["region_uuid"].astype(str) == region, "node_idx"
        ].to_numpy(dtype=np.int64)
    )
    return LazyRegionalPairOracle(
        suite_id=suite.suite_id,
        candidate_id=candidate_id,
        support=support,
        terminals=region_terminals,
        fine_members_global={
            terminal: fine_global[terminal]
            for terminal in region_terminals["terminal_id"].astype(str).tolist()
        },
        physical_arc_ids=graph.arc_id,
        centroid_global_nodes=centroid_nodes,
        retain_query_cache=retain_query_cache,
    )


def _induced_scalar_path_and_lift(
    relations: Sequence[CoarseEdgeRelation],
    internal: Mapping[str, Sequence[PhysicalTransition]],
    origin_idx: int,
    destination_idx: int,
    endpoint_charge: Mapping[str, float],
) -> tuple[Any | None, Any | None]:
    if not relations:
        raise Phase3AOperationError("Induced scalar smoke requires a nonempty coarse relation sequence")
    adjacency: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    relation_by_id: dict[str, CoarseEdgeRelation] = {}
    for relation in relations:
        cost = min(float(value.fftt_s) for value in relation.transitions)
        adjacency[relation.tail_coarse_node_id].append(
            (relation.coarse_edge_id, relation.head_coarse_node_id, cost)
        )
        relation_by_id[relation.coarse_edge_id] = relation
    scalar = scalar_shortest_path(
        source_node_id=relations[0].tail_coarse_node_id,
        target_node_id=relations[-1].head_coarse_node_id,
        neighbor_provider=lambda node: adjacency.get(str(node), ()),
        endpoint_charge_s=endpoint_charge,
    )
    if scalar is None:
        return None, None
    coarse = [relation_by_id[value] for value in scalar.edge_ids]
    lift = audit_liftability(
        coarse_edges=coarse,
        source_states=[int(origin_idx)],
        destination_states=[int(destination_idx)],
        internal_relations=internal,
    )
    return scalar, lift


def _od_centroid_terminals(
    terminals: pd.DataFrame,
    terminal_members: pd.DataFrame,
    origin_uid: str,
    destination_uid: str,
) -> tuple[str, str]:
    terminal_types = terminals.set_index("terminal_id")["terminal_type"].astype(str).to_dict()
    origin = [
        str(row.terminal_id)
        for row in terminal_members.itertuples(index=False)
        if str(row.partC_rev1_node_uid) == str(origin_uid)
        and terminal_types[str(row.terminal_id)] == "ORIGIN_CENTROID"
    ]
    destination = [
        str(row.terminal_id)
        for row in terminal_members.itertuples(index=False)
        if str(row.partC_rev1_node_uid) == str(destination_uid)
        and terminal_types[str(row.terminal_id)] == "DESTINATION_CENTROID"
    ]
    if len(origin) != 1 or len(destination) != 1:
        raise Phase3AOperationError("OD endpoint does not map to one separate centroid-role terminal")
    return origin[0], destination[0]


@dataclass(slots=True)
class _ODEndpointInterface:
    """Candidate-specific role nodes for the immutable shared OD sample."""

    origin_terminal_by_uid: dict[str, str]
    destination_terminal_by_uid: dict[str, str]
    virtual_roles: tuple[dict[str, Any], ...]

    def terminals(self, origin_uid: str, destination_uid: str) -> tuple[str, str]:
        origin = self.origin_terminal_by_uid.get(str(origin_uid))
        destination = self.destination_terminal_by_uid.get(str(destination_uid))
        if origin is None or destination is None:
            raise Phase3AOperationError("Shared OD endpoint interface is incomplete")
        return origin, destination

    def stats(self) -> dict[str, int]:
        return {
            "origin_endpoint_count": len(self.origin_terminal_by_uid),
            "destination_endpoint_count": len(self.destination_terminal_by_uid),
            "backbone_origin_role_clone_count": sum(
                str(row["terminal_type"]) == "ORIGIN_CENTROID" for row in self.virtual_roles
            ),
            "backbone_destination_role_clone_count": sum(
                str(row["terminal_type"]) == "DESTINATION_CENTROID" for row in self.virtual_roles
            ),
        }


def _build_od_endpoint_interface(
    *,
    graph: GraphData,
    suite: SuiteData,
    candidate_id: str,
    membership: pd.DataFrame,
    terminals: pd.DataFrame,
    terminal_members: pd.DataFrame,
    sample: pd.DataFrame,
) -> _ODEndpointInterface:
    """Close sampled backbone-only endpoints without changing the frozen topology.

    Accepted Phase 2 centroid roles are regional: a physical OD centroid with
    only final-backbone incident arcs has no region role row.  Such a sampled
    endpoint receives a separate non-through query role connected at zero cost
    and in the permitted direction to its containing candidate B-ACN.
    """

    required_sample = {"origin_uid", "destination_uid", "origin_idx", "destination_idx"}
    if not required_sample.issubset(sample.columns):
        raise Phase3AOperationError("Shared OD sample lacks endpoint identity columns")
    typed_members = terminal_members.merge(
        terminals[["terminal_id", "terminal_type"]],
        on="terminal_id",
        how="left",
        validate="many_to_one",
    )
    if typed_members["terminal_type"].isna().any():
        raise Phase3AOperationError("Terminal member references an absent consolidated terminal")
    role_maps: dict[str, dict[str, list[str]]] = {}
    for terminal_type in ("ORIGIN_CENTROID", "DESTINATION_CENTROID"):
        selected = typed_members[typed_members["terminal_type"].astype(str) == terminal_type]
        role_maps[terminal_type] = {
            str(uid): sorted(group["terminal_id"].astype(str).tolist())
            for uid, group in selected.groupby("partC_rev1_node_uid", sort=False)
        }
    sample_endpoint_index: dict[str, int] = {}
    for uid_column, index_column in (
        ("origin_uid", "origin_idx"),
        ("destination_uid", "destination_idx"),
    ):
        for uid, index in sample[[uid_column, index_column]].drop_duplicates().itertuples(index=False):
            key, value = str(uid), int(index)
            previous = sample_endpoint_index.setdefault(key, value)
            if previous != value or str(graph.node_uid[value]) != key:
                raise Phase3AOperationError("Shared OD endpoint UID/index binding is inconsistent")
    bacn_by_uid = node_to_bacn(membership)
    incident_by_index = suite.node_incident_registry.set_index("node_idx").to_dict("index")
    virtual: list[dict[str, Any]] = []

    def resolve_role(uid: str, terminal_type: str) -> str:
        matches = role_maps[terminal_type].get(str(uid), [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise Phase3AOperationError("Sampled OD endpoint maps to multiple accepted centroid roles")
        node_index = sample_endpoint_index[str(uid)]
        incident = incident_by_index.get(int(node_index))
        if incident is None:
            raise Phase3AOperationError("Sampled OD endpoint is absent from the s03 incident registry")
        backbone_only = (
            bool(suite.is_final_backbone_node[int(node_index)])
            and str(incident["primary_partition_region_uuid"]) == "backbone"
            and int(incident["incident_label_count"]) == 1
            and str(incident["incident_labels"]) == "backbone"
            and not bool(incident["is_physical_portal"])
        )
        if not backbone_only or str(uid) not in bacn_by_uid:
            raise Phase3AOperationError(
                "Missing sampled centroid role is not a validated backbone-only B-ACN endpoint"
            )
        bacn_id = str(bacn_by_uid[str(uid)])
        role_object_id = stable_object_id(
            "centroid_role",
            suite_id=suite.suite_id,
            candidate_id=candidate_id,
            members=[str(uid)],
            signature={"region_id": "backbone", "role": terminal_type},
        )
        terminal_id = stable_object_id(
            "terminal",
            suite_id=suite.suite_id,
            candidate_id=candidate_id,
            members=[str(uid)],
            signature={
                "region_id": "backbone",
                "terminal_type": terminal_type,
                "parent_object_id": role_object_id,
            },
        )
        if terminal_type == "ORIGIN_CENTROID":
            direction = "origin_endpoint_to_backbone_parent"
            tail_id, head_id = terminal_id, bacn_id
        else:
            direction = "backbone_parent_to_destination_endpoint"
            tail_id, head_id = bacn_id, terminal_id
        connector_edge_id = stable_object_id(
            "terminal_connector",
            suite_id=suite.suite_id,
            candidate_id=candidate_id,
            members=[tail_id, head_id],
            ordered_members=True,
            signature={"direction": direction, "endpoint_uid": str(uid)},
        )
        virtual.append(
            {
                "partC_rev1_node_uid": str(uid),
                "node_idx": int(node_index),
                "terminal_type": terminal_type,
                "terminal_id": terminal_id,
                "parent_bacn_id": bacn_id,
                "connector_edge_id": connector_edge_id,
                "connector_tail_id": tail_id,
                "connector_head_id": head_id,
                "connector_direction": direction,
                "centroid_through_allowed": False,
            }
        )
        return terminal_id

    origin_uids = sorted(sample["origin_uid"].astype(str).unique().tolist(), key=_id_key)
    destination_uids = sorted(sample["destination_uid"].astype(str).unique().tolist(), key=_id_key)
    origin_map = {uid: resolve_role(uid, "ORIGIN_CENTROID") for uid in origin_uids}
    destination_map = {uid: resolve_role(uid, "DESTINATION_CENTROID") for uid in destination_uids}
    if len(origin_map) != len(origin_uids) or len(destination_map) != len(destination_uids):
        raise Phase3AOperationError("Shared OD endpoint interface did not close exactly")
    virtual_sorted = tuple(
        sorted(
            virtual,
            key=lambda row: (str(row["terminal_type"]), _id_key(str(row["partC_rev1_node_uid"]))),
        )
    )
    if len({str(row["terminal_id"]) for row in virtual_sorted}) != len(virtual_sorted):
        raise Phase3AOperationError("Backbone-only endpoint role terminal IDs are not unique")
    return _ODEndpointInterface(origin_map, destination_map, virtual_sorted)


class _FullCandidateNetwork:
    """Lazy national scalar neighbor and endpoint-transition interface."""

    def __init__(
        self,
        *,
        graph: GraphData,
        suite: SuiteData,
        candidate_id: str,
        endpoint_interface: _ODEndpointInterface,
        membership: pd.DataFrame,
        bundle_membership: pd.DataFrame,
        bundle_summary: pd.DataFrame,
        suppressed: pd.DataFrame,
        terminals: pd.DataFrame,
        terminal_members: pd.DataFrame,
        racn_witnesses: pd.DataFrame,
        structural_charge: Mapping[str, float],
        absorbed_witnesses: Sequence[AugWitness],
        regional_oracle_cache_max_regions: int,
        retain_regional_query_cache: bool,
    ) -> None:
        del structural_charge
        self.graph = graph
        self.suite = suite
        self.candidate_id = candidate_id
        self.endpoint_interface = endpoint_interface
        self.terminals = terminals
        self.terminal_members = terminal_members
        self.edge_relations: dict[str, CoarseEdgeRelation] = {}
        self.internal_relations: dict[str, list[PhysicalTransition]] = defaultdict(list)
        self.static: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        if int(regional_oracle_cache_max_regions) < 1:
            raise Phase3AOperationError("Regional oracle cache requires at least one region")
        self.regional_oracle_cache_max_regions = int(regional_oracle_cache_max_regions)
        self.retain_regional_query_cache = bool(retain_regional_query_cache)
        self.region_oracles: OrderedDict[str, LazyRegionalPairOracle] = OrderedDict()
        self.regional_pair_summary = pd.DataFrame()
        self.terminal_by_id = terminals.set_index("terminal_id").to_dict("index")
        uid_to_index = {str(value): index for index, value in enumerate(graph.node_uid.tolist())}
        self.fine_global = {
            str(terminal_id): tuple(uid_to_index[str(uid)] for uid in group["partC_rev1_node_uid"].astype(str))
            for terminal_id, group in terminal_members.groupby("terminal_id", sort=False)
        }
        self.centroid_nodes = frozenset(
            terminals.loc[
                terminals["terminal_type"].isin(["ORIGIN_CENTROID", "DESTINATION_CENTROID"]),
                "terminal_id",
            ].astype(str)
        ) | frozenset(str(row["terminal_id"]) for row in endpoint_interface.virtual_roles)
        bundle_cost_by_id = dict(
            zip(
                bundle_summary["bundle_id"].astype(str),
                bundle_summary["direct_diagnostic_fftt_s"].astype(float),
                strict=True,
            )
        )
        if len(bundle_cost_by_id) != len(bundle_summary):
            raise Phase3AOperationError("Bundle summary IDs are not unique")
        for bundle_id, group in bundle_membership.groupby("bundle_id", sort=True):
            first = group.iloc[0]
            cost = bundle_cost_by_id.get(str(bundle_id))
            if cost is None:
                raise Phase3AOperationError("Bundle membership references an absent bundle summary")
            transitions = tuple(
                PhysicalTransition(
                    uid_to_index[str(row.physical_tail_node_uid)],
                    uid_to_index[str(row.physical_head_node_uid)],
                    float(row.member_fftt_s),
                    (str(row.directed_arc_id),),
                )
                for row in group.itertuples(index=False)
            )
            relation = CoarseEdgeRelation(
                str(bundle_id), str(first.tail_bacn_id), str(first.head_bacn_id), transitions
            )
            self.edge_relations[str(bundle_id)] = relation
            self.static[str(first.tail_bacn_id)].append((str(bundle_id), str(first.head_bacn_id), cost))
        for row in terminals.itertuples(index=False):
            terminal_id = str(row.terminal_id)
            terminal_type = str(row.terminal_type)
            if terminal_type not in {"BACN", "RACN"}:
                continue
            parent = str(row.parent_object_id)
            members = self.fine_global[terminal_id]
            transitions = tuple(PhysicalTransition(value, value, 0.0, ()) for value in members)
            if bool(row.can_exit_region):
                edge_id = stable_object_id(
                    "terminal_connector",
                    suite_id=suite.suite_id,
                    candidate_id=candidate_id,
                    members=[parent, terminal_id],
                    ordered_members=True,
                    signature={"direction": "parent_to_region"},
                )
                self.static[parent].append((edge_id, terminal_id, 0.0))
                self.edge_relations[edge_id] = CoarseEdgeRelation(edge_id, parent, terminal_id, transitions)
            if bool(row.can_enter_region):
                edge_id = stable_object_id(
                    "terminal_connector",
                    suite_id=suite.suite_id,
                    candidate_id=candidate_id,
                    members=[terminal_id, parent],
                    ordered_members=True,
                    signature={"direction": "region_to_parent"},
                )
                self.static[terminal_id].append((edge_id, parent, 0.0))
                self.edge_relations[edge_id] = CoarseEdgeRelation(edge_id, terminal_id, parent, transitions)
        for row in endpoint_interface.virtual_roles:
            edge_id = str(row["connector_edge_id"])
            tail_id = str(row["connector_tail_id"])
            head_id = str(row["connector_head_id"])
            node_index = int(row["node_idx"])
            self.static[tail_id].append((edge_id, head_id, 0.0))
            self.edge_relations[edge_id] = CoarseEdgeRelation(
                edge_id,
                tail_id,
                head_id,
                (PhysicalTransition(node_index, node_index, 0.0, ()),),
            )
        for row in suppressed.itertuples(index=False):
            self.internal_relations[str(row.bacn_id)].append(
                PhysicalTransition(
                    uid_to_index[str(row.physical_tail_node_uid)],
                    uid_to_index[str(row.physical_head_node_uid)],
                    float(row.fftt_s),
                    (str(row.directed_arc_id),),
                )
            )
        bacn_by_uid = node_to_bacn(membership)
        for witness in absorbed_witnesses:
            bacn_id = bacn_by_uid[witness.source_node_uid]
            self.internal_relations[bacn_id].append(
                PhysicalTransition(
                    witness.source_node_idx,
                    witness.target_node_idx,
                    witness.fftt_s,
                    witness.arc_ids,
                )
            )
        if not racn_witnesses.empty:
            # Recover parent R-ACN through consolidated terminal parent IDs.
            racn_by_uid = {
                str(uid): str(parent)
                for terminal_id, group in terminal_members.groupby("terminal_id", sort=False)
                for uid in group["partC_rev1_node_uid"].astype(str)
                for parent in [self.terminal_by_id[str(terminal_id)]["parent_object_id"]]
                if str(self.terminal_by_id[str(terminal_id)]["terminal_type"]) == "RACN"
            }
            for row in racn_witnesses.itertuples(index=False):
                source_uid = str(row.source_node_uid)
                if source_uid not in racn_by_uid:
                    continue
                arc_indexes = tuple(map(int, json.loads(str(row.physical_arc_indices_json))))
                self.internal_relations[racn_by_uid[source_uid]].append(
                    PhysicalTransition(
                        uid_to_index[source_uid],
                        uid_to_index[str(row.target_node_uid)],
                        float(row.restricted_fftt_s),
                        tuple(str(graph.arc_id[index]) for index in arc_indexes),
                    )
                )
        self._static_edge_ids = tuple(sorted(self.edge_relations))

    def od_terminals(self, origin_uid: str, destination_uid: str) -> tuple[str, str]:
        return self.endpoint_interface.terminals(origin_uid, destination_uid)

    def materialize_scalar_graph(self) -> MaterializedScalarGraph:
        """Materialize the complete sparse scalar graph without retaining witnesses."""

        edges: list[tuple[str, str, str, float]] = []
        for tail_id in sorted(self.static):
            for edge_id, head_id, cost_s in sorted(
                self.static[tail_id], key=lambda value: (str(value[0]), str(value[1]))
            ):
                edges.append((str(edge_id), str(tail_id), str(head_id), float(cost_s)))
        grouped = list(self.terminals.groupby("region_id", sort=True))
        regional_source_count = 0
        regional_edge_count = 0
        regional_rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        for position, (region, region_terminals) in enumerate(grouped, 1):
            region_started = time.perf_counter()
            oracle = _build_region_oracle(
                self.graph,
                self.suite,
                self.candidate_id,
                str(region),
                self.terminals,
                self.fine_global,
                retain_query_cache=False,
            )
            sources = sorted(
                region_terminals.loc[
                    region_terminals["can_exit_region"].astype(bool), "terminal_id"
                ].astype(str).tolist()
            )
            region_edge_count = 0
            for source_id in sources:
                witnesses = oracle.query(source_id)
                regional_source_count += 1
                regional_edge_count += len(witnesses)
                region_edge_count += len(witnesses)
                edges.extend(
                    (
                        str(witness.pair_id),
                        str(witness.source_terminal_id),
                        str(witness.target_terminal_id),
                        float(witness.fftt_s),
                    )
                    for witness in witnesses
                )
            regional_rows.append(
                {
                    "suite_id": self.suite.suite_id,
                    "candidate_id": self.candidate_id,
                    "region_id": str(region),
                    "region_node_count": int(oracle.support.graph.node_count),
                    "region_arc_count": int(oracle.support.graph.arc_count),
                    "terminal_count": int(len(region_terminals)),
                    "source_terminal_count": int(len(sources)),
                    "target_terminal_count": int(region_terminals["can_enter_region"].astype(bool).sum()),
                    "supported_pair_count": int(region_edge_count),
                    "enumeration_seconds": float(time.perf_counter() - region_started),
                    "representation_mode": "EXACT_SPARSE_LAZY",
                    "dense_terminal_square_allocated": False,
                }
            )
            if position % 100 == 0 or position == len(grouped):
                print(
                    "PHASE3A_MATERIALIZE "
                    f"candidate={self.candidate_id} regions={position}/{len(grouped)} "
                    f"regional_sources={regional_source_count} regional_edges={regional_edge_count} "
                    f"elapsed_s={time.perf_counter() - started:.3f}",
                    flush=True,
                )
        self.region_oracles.clear()
        self.regional_pair_summary = pd.DataFrame(regional_rows).sort_values(
            "region_id", kind="mergesort"
        ).reset_index(drop=True)
        graph = build_materialized_scalar_graph(edges)
        self.materialization_stats = {
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "static_edge_count": graph.edge_count - regional_edge_count,
            "regional_source_count": regional_source_count,
            "regional_edge_count": regional_edge_count,
            "elapsed_s": time.perf_counter() - started,
        }
        return graph

    def reset_lazy_state(self) -> None:
        """Release exact regional query state while retaining frozen static relations."""
        self.edge_relations = {
            edge_id: self.edge_relations[edge_id]
            for edge_id in self._static_edge_ids
        }
        self.region_oracles.clear()

    def region_oracle(self, region: str) -> LazyRegionalPairOracle:
        """Return an exact regional oracle from a bounded LRU cache."""

        region_id = str(region)
        oracle = self.region_oracles.pop(region_id, None)
        if oracle is None:
            oracle = _build_region_oracle(
                self.graph,
                self.suite,
                self.candidate_id,
                region_id,
                self.terminals,
                self.fine_global,
                retain_query_cache=self.retain_regional_query_cache,
            )
        self.region_oracles[region_id] = oracle
        while len(self.region_oracles) > self.regional_oracle_cache_max_regions:
            self.region_oracles.popitem(last=False)
        return oracle

    def path_relations(
        self,
        edge_ids: Sequence[str],
        node_ids: Sequence[str],
    ) -> tuple[CoarseEdgeRelation, ...]:
        """Rehydrate only the dynamic relations used by a completed scalar path."""

        if len(node_ids) != len(edge_ids) + 1:
            raise Phase3AOperationError("Scalar path nodes and edges are misaligned")
        relations: list[CoarseEdgeRelation] = []
        for index, raw_edge_id in enumerate(edge_ids):
            edge_id = str(raw_edge_id)
            static_relation = self.edge_relations.get(edge_id)
            if static_relation is not None:
                relations.append(static_relation)
                continue
            source_id = str(node_ids[index])
            target_id = str(node_ids[index + 1])
            source_row = self.terminal_by_id.get(source_id)
            if source_row is None or not bool(source_row["can_exit_region"]):
                raise Phase3AOperationError("Dynamic scalar edge lacks an exiting regional source")
            oracle = self.region_oracle(str(source_row["region_id"]))
            witness = next(
                (
                    value
                    for value in oracle.query(source_id)
                    if value.pair_id == edge_id and value.target_terminal_id == target_id
                ),
                None,
            )
            if witness is None:
                raise Phase3AOperationError("Dynamic scalar edge cannot be rehydrated for liftability")
            relations.append(
                CoarseEdgeRelation(
                    witness.pair_id,
                    witness.source_terminal_id,
                    witness.target_terminal_id,
                    (
                        PhysicalTransition(
                            witness.source_node_idx,
                            witness.target_node_idx,
                            witness.fftt_s,
                            witness.arc_ids,
                        ),
                    ),
                )
            )
        return tuple(relations)

    def neighbors(self, node: str) -> tuple[tuple[str, str, float], ...]:
        node_id = str(node)
        values = list(self.static.get(node_id, ()))
        row = self.terminal_by_id.get(node_id)
        if row is not None and bool(row["can_exit_region"]):
            region = str(row["region_id"])
            for witness in self.region_oracle(region).query(node_id):
                values.append((witness.pair_id, witness.target_terminal_id, float(witness.fftt_s)))
        return tuple(sorted(values, key=lambda value: (value[0], value[1], value[2])))


def _recompute_candidate_membership(
    graph: GraphData, suite: SuiteData, source_dir: Path
) -> pd.DataFrame:
    spec = json.loads((source_dir / "candidate_spec.json").read_text(encoding="utf-8"))
    candidate_id = str(spec["candidate_id"])
    family = str(spec["family"])
    algorithm = str(spec["algorithm"])
    backbone = np.flatnonzero(suite.is_final_backbone_node).astype(np.int64)
    weak = {
        str(graph.node_uid[index]): int(graph.physical_weak_component[index])
        for index in backbone.tolist()
    }
    if family == "IDENTITY":
        cluster_result = None
    elif family == "GEO":
        pair_graph = build_geo_pair_graph(
            graph.node_uid[backbone],
            graph.lon[backbone],
            graph.lat[backbone],
            graph.physical_weak_component[backbone],
            float(spec["parameter_value"]) * 1000.0,
        )
        complete_distance: Callable[[str, str], float] | None = None
        if algorithm == "SL":
            uid_to_index = {str(graph.node_uid[index]): int(index) for index in backbone.tolist()}

            def _sl_complete_distance(left: str, right: str) -> float:
                left_index, right_index = uid_to_index[str(left)], uid_to_index[str(right)]
                return float(
                    grs80_distance_m(
                        graph.lon[left_index], graph.lat[left_index],
                        graph.lon[right_index], graph.lat[right_index],
                    )
                )

            complete_distance = _sl_complete_distance

        cluster_result = cluster_pair_graph(pair_graph, algorithm, complete_distance=complete_distance)
    elif family == "AUG":
        witness_dirs = sorted(
            {
                Path(str(path)).resolve(strict=True).parent
                for path in spec["witness_paths"]
                if Path(str(path)).name == "aug_witness_registry.parquet"
            }
        )
        persisted_witnesses: list[AugWitness] = []
        witness_sources: list[int] = []
        for directory in witness_dirs:
            values, _ = _read_witness_directory(directory, graph, suite)
            persisted_witnesses.extend(values)
            stats = pd.read_parquet(directory / "aug_witness_search_stats.parquet")
            witness_sources.extend(map(int, stats["source_node_idx"].tolist()))
        policy_row = spec["policy"]
        policy = AugPolicy(
            policy_id=str(policy_row["policy_id"]),
            threshold_s=float(policy_row["threshold_s"]),
            search_radius_m=float(policy_row["search_radius_m"]),
            max_physical_arcs=int(policy_row["max_physical_arcs"]),
            max_length_geodesic_detour=float(policy_row["max_length_geodesic_detour"]),
            epsilon_distance_m=float(policy_row["epsilon_distance_m"]),
        )
        reproduced_witnesses, _ = witness_census_with_stats(
            graph, suite, witness_sources, policy
        )
        persisted_signature = [
            (value.source_node_uid, value.target_node_uid, value.fftt_s, value.length_m, value.arc_ids)
            for value in persisted_witnesses
        ]
        reproduced_signature = [
            (value.source_node_uid, value.target_node_uid, value.fftt_s, value.length_m, value.arc_ids)
            for value in reproduced_witnesses
        ]
        if persisted_signature != reproduced_signature:
            raise Phase3AOperationError(f"Fresh AUG witness reproduction differs: {candidate_id}")
        witnesses = list(reproduced_witnesses)
        adjacency = build_temporary_augmented_adjacency(graph, suite, witnesses)
        uid_to_index = {str(value): index for index, value in enumerate(graph.node_uid.tolist())}
        sources = [uid_to_index[str(value)] for value in spec["distance_source_node_uids"]]
        distance = compute_augmented_distance_shard(adjacency, sources, policy)
        pair_graph = build_aug_pair_graph_from_distances(graph, adjacency, distance, policy)
        complete_distance = _aug_complete_distance_provider(graph, adjacency) if algorithm == "SL" else None
        cluster_result = cluster_pair_graph(pair_graph, algorithm, complete_distance=complete_distance)
    else:
        raise Phase3AOperationError(f"Unknown candidate family in reproduction: {family}")
    membership, _ = build_bacn_membership(
        suite_id=suite.suite_id,
        candidate_id=candidate_id,
        family=family,
        algorithm=algorithm,
        parameter_value=float(spec["parameter_value"]),
        parameter_unit=str(spec["parameter_unit"]),
        final_backbone_node_uids=graph.node_uid[backbone],
        weak_component_by_uid=weak,
        cluster_result=cluster_result,
    )
    return membership


def _file_binding(path: Path, output_root: Path) -> dict[str, Any]:
    try:
        relative = path.relative_to(output_root).as_posix()
    except ValueError:
        relative = str(path)
    return {
        "path": str(path),
        "relative_path": relative,
        "size_bytes": int(path.stat().st_size),
        "sha256": partC_rev1_file_sha256(path),
    }


def _git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(project_root), "-c", "core.longpaths=true", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _id_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)
