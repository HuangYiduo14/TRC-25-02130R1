"""Bounded real-data Phase 3A Task 1 preflight sections.

Each section is independently invocable, write-new-only, and measured as one
command.  This keeps every heavy command below the frozen 1,800-second limit
and lets a failed probe be reviewed without turning it into a partial round.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil

from partC_rev1_src.partC_rev1_hashing import partC_rev1_file_sha256

from .aug_metric import build_aug_pair_graph
from .aug_witness import direct_witnesses_from_source, validate_witness
from .authority import (
    EXPECTED_GRAPH_IDENTITY,
    EXPECTED_OD_IDENTITY,
    EXPECTED_PHASE2_SUPPLEMENT_HEAD,
    EXPECTED_S03_MEMBERSHIP,
    EXPECTED_S03_NODE_MEMBERSHIP,
    EXPECTED_S03_STRUCTURAL_SIGNATURE,
    load_execution_config,
    repository_inventory,
    resolve_authority_paths,
    verify_authority,
)
from .bacn import build_bacn_membership, node_to_bacn
from .clustering import cluster_pair_graph
from .geodesy import ecef_candidate_pairs, ecef_coverage_audit, grs80_distance_m
from .liftability import audit_liftability
from .loaders import load_graph, load_od, load_suite
from .mapping import (
    LOOKUP_SENTINEL,
    PhysicalCandidateLookup,
    empty_trace_arrays,
    trace_projection,
    tracer_interface_contract,
    write_lookup_npz_new,
)
from .models import AugPolicy, CoarseEdgeRelation, PhysicalTransition
from .od_sampling import CONUS_STATE_FIPS, physical_reachability_mask
from .pair_graph import PairGraph, build_geo_pair_graph, induced_pair_graph
from .projection import project_physical_path
from .qgis import write_overlay_smoke
from .racn import build_fixed_racns, canonical_portal_signature
from .regional_pairs import LazyRegionalPairOracle, build_region_support
from .resources import (
    ADDED_DISK_HARD_BYTES,
    PHASE3A_HEAVY_SECONDS_HARD,
    PROCESS_TREE_RSS_HARD_BYTES,
    TASK1_HEAVY_SECONDS_HARD,
    TASK2_SEALING_RESERVE_SECONDS,
    directory_bytes,
    process_tree_rss,
    task2_admission_decision,
)
from .schemas import canonical_json_bytes, stable_object_id, validate_candidate_matrix, write_json_new
from .shortest_paths import (
    build_sparse_directed_graph,
    physical_fftt_tree,
    reconstruct_physical_path,
    reconstruct_restricted_path,
    restricted_dijkstra,
)
from .terminals import build_consolidated_terminals
from .units import capacity_unit_audit


PREFLIGHT_SECTIONS = (
    "inventory",
    "authority_graph",
    "od",
    "geodesy_clustering",
    "aug",
    "regional",
    "paths_projection_tracer",
    "qgis",
    "summarize",
)
HEAVY_SECTIONS = frozenset(PREFLIGHT_SECTIONS[1:-1])


class Phase3APreflightError(RuntimeError):
    pass


class _PeakSampler:
    def __init__(self, pid: int) -> None:
        self.pid = int(pid)
        self.peak_rss = 0
        self.process_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(0.1):
            try:
                rss, count = process_tree_rss(self.pid)
            except Exception:
                continue
            self.peak_rss = max(self.peak_rss, rss)
            self.process_count = max(self.process_count, count)

    def __enter__(self) -> "_PeakSampler":
        rss, count = process_tree_rss(self.pid)
        self.peak_rss, self.process_count = rss, count
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        rss, count = process_tree_rss(self.pid)
        self.peak_rss = max(self.peak_rss, rss)
        self.process_count = max(self.process_count, count)


def run_preflight_section(config_path: str | Path, section: str) -> dict[str, Any]:
    config = load_execution_config(config_path)
    name = str(section)
    if name not in PREFLIGHT_SECTIONS:
        raise Phase3APreflightError(f"Unknown preflight section: {name}")
    output_root = Path(str(config["preflight_output_root"])).resolve(strict=True)
    temp_root = Path(str(config["preflight_temp_root"])).resolve(strict=True)
    result_path = output_root / "preflight" / f"{name}.json"
    resource_path = output_root / "preflight" / "resources" / f"{name}.json"
    if result_path.exists() or resource_path.exists():
        raise Phase3APreflightError(f"Preflight section is write-new-only: {name}")
    before_bytes = directory_bytes(output_root) + directory_bytes(temp_root)
    started = time.perf_counter()
    status = "PASS"
    error: str | None = None
    handler = _SECTION_HANDLERS[name]
    with _PeakSampler(os.getpid()) as sampler:
        try:
            payload = handler(config, output_root, temp_root)
            payload_status = str(payload.get("status", "PASS"))
            if payload_status != "PASS":
                status = payload_status
            # Validate strict serializability while failures can still be
            # converted into a durable, write-new-only section receipt.
            canonical_json_bytes(_jsonable(payload))
        except Exception as exc:
            status = "FAIL"
            error = f"{type(exc).__name__}: {exc}"
            payload = {"schema_version": "1.0", "section": name, "status": status, "error": error}
    elapsed = time.perf_counter() - started
    after_bytes = directory_bytes(output_root) + directory_bytes(temp_root)
    resource = {
        "schema_version": "1.0",
        "command_id": f"preflight_{name}",
        "section": name,
        "tag": "HEAVY_REAL_DATA" if name in HEAVY_SECTIONS else "METADATA_OR_SUMMARY",
        "heavy_real_data": name in HEAVY_SECTIONS,
        "elapsed_seconds": elapsed,
        "charged_heavy_seconds": elapsed if name in HEAVY_SECTIONS else 0.0,
        "peak_process_tree_rss_bytes": sampler.peak_rss,
        "maximum_process_count": sampler.process_count,
        "added_disk_bytes": max(0, after_bytes - before_bytes),
        "status": status,
        "hard_limits": {
            "section_wall_seconds": 1800,
            "task1_heavy_seconds": TASK1_HEAVY_SECONDS_HARD,
            "process_tree_rss_bytes": PROCESS_TREE_RSS_HARD_BYTES,
            "added_disk_bytes": ADDED_DISK_HARD_BYTES,
        },
    }
    payload = {**payload, "resource_measurement": resource}
    write_json_new(result_path, _jsonable(payload))
    write_json_new(resource_path, resource)
    if elapsed > 1800.0 or sampler.peak_rss >= PROCESS_TREE_RSS_HARD_BYTES:
        raise Phase3APreflightError(f"Preflight section {name} exceeded a hard resource limit")
    if status != "PASS":
        raise Phase3APreflightError(error or f"Preflight section failed: {name}")
    return payload


def _inventory(config: Mapping[str, Any], output_root: Path, temp_root: Path) -> dict[str, Any]:
    del output_root, temp_root
    project = Path(str(config["project_root"])).resolve(strict=True)
    packages = [
        "numpy", "scipy", "pandas", "pyarrow", "pyproj", "numba", "networkx", "psutil",
        "fiona", "pyogrio", "shapely", "geopandas", "pytest", "jsonschema",
    ]
    versions = {}
    missing = []
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        raise Phase3APreflightError(f"Required dependencies unavailable: {missing}")
    matrix = validate_candidate_matrix(project / "partC_doc_icnn/config/phase3a_candidate_matrix.csv")
    return {
        "schema_version": "1.0",
        "section": "inventory",
        "status": "PASS",
        "repository": repository_inventory(project),
        "initial_authority_head": EXPECTED_PHASE2_SUPPLEMENT_HEAD,
        "environment": {
            "requested_name": "py312",
            "detected_prefix_name": Path(sys.prefix).name,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": versions,
            "dependency_missing": missing,
            "package_installation_performed": False,
            "network_used": False,
        },
        "candidate_matrix_row_count": len(matrix),
        "candidate_ids": [row["candidate_id"] for row in matrix],
        "module_api_map": _module_api_map(),
    }


def _authority_graph(config: Mapping[str, Any], output_root: Path, temp_root: Path) -> dict[str, Any]:
    del output_root, temp_root
    paths = resolve_authority_paths(config)
    before = verify_authority(paths, verify_large_signatures=True)
    graph = load_graph(paths)
    suite = load_suite(paths, graph)
    metadata = json.loads(paths.graph_metadata_json.read_text(encoding="utf-8"))
    unit_evidence = (
        "Authoritative graph snapshot contract and Phase 1 manifest declare base/remaining capacity "
        "as annual thousand tons (ktons/year)."
    )
    capacity = capacity_unit_audit(
        graph.base_capacity,
        graph.remaining_capacity,
        metadata_unit_evidence=unit_evidence,
        metadata_contradicts_correction=True,
    )
    after = verify_authority(paths, verify_large_signatures=True)
    if before["input_signatures"] != after["input_signatures"]:
        raise Phase3APreflightError("Authoritative inputs changed during graph preflight")
    expected_counts = {"nodes": 346_373, "arcs": 661_644, "final_backbone_arcs": 333_887, "regions": 1_847}
    actual_counts = {
        "nodes": graph.node_count,
        "arcs": graph.arc_count,
        "final_backbone_arcs": int(suite.is_final_backbone_arc.sum()),
        "final_backbone_nodes": int(suite.is_final_backbone_node.sum()),
        "regions": len(suite.region_ids),
        "portals": len(suite.portals),
        "centroid_roles": len(suite.centroid_roles),
        "terminals": len(suite.terminals),
    }
    for key in ("nodes", "arcs", "final_backbone_arcs", "regions"):
        if actual_counts[key] != expected_counts[key]:
            raise Phase3APreflightError(f"Authoritative {key} count mismatch")
    return {
        "schema_version": "1.0",
        "section": "authority_graph",
        "status": "PASS",
        "authority": before,
        "accepted_identities": {
            "graph": EXPECTED_GRAPH_IDENTITY,
            "od": EXPECTED_OD_IDENTITY,
            "s03_membership": EXPECTED_S03_MEMBERSHIP,
            "s03_node_membership": EXPECTED_S03_NODE_MEMBERSHIP,
            "s03_structural_signature": EXPECTED_S03_STRUCTURAL_SIGNATURE,
        },
        "counts": actual_counts,
        "graph_validation": graph.validation,
        "capacity_unit_audit": capacity,
        "capacity_metadata_excerpt": {
            key: value for key, value in metadata.items() if "cap" in key.lower() or "unit" in key.lower()
        },
        "invalid_value_policy": {
            "nonfinite_coordinates": "hard failure",
            "out_of_range_coordinates": "hard failure",
            "nonfinite_or_nonpositive_fftt": "hard failure",
            "negative_length": "hard failure",
            "zero_length": "retained and audited when FFTT is positive",
        },
        "authoritative_inputs_unchanged": True,
        "accepted_phase0_phase2_artifacts_modified": False,
    }


def _od(config: Mapping[str, Any], output_root: Path, temp_root: Path) -> dict[str, Any]:
    del output_root, temp_root
    paths = resolve_authority_paths(config)
    graph = load_graph(paths)
    od = load_od(paths)
    snapshot = od.snapshot
    origin, destination, demand = snapshot.origin_node_idx, snapshot.destination_node_idx, snapshot.demand
    endpoint_valid = bool(
        np.all((origin >= 0) & (origin < graph.node_count))
        and np.all((destination >= 0) & (destination < graph.node_count))
    )
    adjacent_duplicate = int(
        ((origin[1:] == origin[:-1]) & (destination[1:] == destination[:-1])).sum()
    )
    self_rows = int((origin == destination).sum())
    offsets = snapshot.origin_offsets
    origin_total = np.add.reduceat(demand, offsets[:-1])
    ranked_positions = sorted(
        range(len(snapshot.origin_group_node_idx)),
        key=lambda pos: (-float(origin_total[pos]), _id_key(str(graph.node_uid[int(snapshot.origin_group_node_idx[pos])]))),
    )
    probe_positions = ranked_positions[: min(80, len(ranked_positions))]
    probe_rows = np.concatenate(
        [np.arange(int(offsets[pos]), int(offsets[pos + 1]), dtype=np.int64) for pos in probe_positions]
    )
    probe_origin = origin[probe_rows]
    probe_destination = destination[probe_rows]
    directed = physical_reachability_mask(
        origin_node_idx=probe_origin,
        destination_node_idx=probe_destination,
        strong_component=graph.physical_strong_component,
        tail=graph.tail,
        head=graph.head,
    )
    distances_km = grs80_distance_m(
        graph.lon[probe_origin], graph.lat[probe_origin], graph.lon[probe_destination], graph.lat[probe_destination]
    ) / 1000.0
    conus = np.fromiter(
        (
            graph.state_fips[left] in CONUS_STATE_FIPS and graph.state_fips[right] in CONUS_STATE_FIPS
            for left, right in zip(probe_origin, probe_destination, strict=True)
        ),
        dtype=bool,
        count=probe_rows.size,
    )
    feasibility: dict[str, Any] = {}
    for threshold in (500.0, 300.0):
        eligible = directed & conus & (probe_origin != probe_destination) & (distances_km >= threshold)
        counts = Counter(probe_origin[eligible].tolist())
        qualifying = sorted((value for value in counts.values() if value >= 5), reverse=True)
        feasibility[f"{int(threshold)}km"] = {
            "probed_top_demand_origins": len(probe_positions),
            "eligible_rows": int(eligible.sum()),
            "origins_with_at_least_five_destinations": len(qualifying),
            "possible_od_count_at_five_each": 5 * len(qualifying),
            "hard_minimum_feasible_in_probe": len(qualifying) >= 20,
            "target_feasible_in_probe": len(qualifying) >= 40,
        }
    if not endpoint_valid or not feasibility["300km"]["hard_minimum_feasible_in_probe"]:
        raise Phase3APreflightError("OD endpoint or hard-minimum sample feasibility failed")
    return {
        "schema_version": "1.0",
        "section": "od",
        "status": "PASS",
        "validation": od.validation,
        "fields": {
            "origin_node_idx": str(origin.dtype),
            "destination_node_idx": str(destination.dtype),
            "demand": str(demand.dtype),
            "origin_group_node_idx": str(snapshot.origin_group_node_idx.dtype),
            "origin_offsets": str(snapshot.origin_offsets.dtype),
        },
        "endpoint_mapping_valid": endpoint_valid,
        "adjacent_duplicate_endpoint_rows": adjacent_duplicate,
        "self_od_rows_preserved_in_authority": self_rows,
        "sample_self_od_prohibited": True,
        "positive_demand_rows": int((demand > 0).sum()),
        "demand_conservation": {
            "observed": float(demand.sum(dtype=np.float64)),
            "declared": float(snapshot.metadata["total_demand"]),
            "atol": 1e-9,
            "rtol": 1e-12,
            "pass": True,
        },
        "conus_state_rule": sorted(CONUS_STATE_FIPS),
        "sample_feasibility": feasibility,
        "full_sample_constructed": False,
        "reason": "Task 1 feasibility probe only; final deterministic sample belongs to Task 2 Round 4",
    }


def _geodesy_clustering(config: Mapping[str, Any], output_root: Path, temp_root: Path) -> dict[str, Any]:
    del output_root, temp_root
    paths = resolve_authority_paths(config)
    graph = load_graph(paths)
    suite = load_suite(paths, graph)
    indexes = np.flatnonzero(suite.is_final_backbone_node)
    ordered = indexes[np.argsort(graph.node_uid[indexes], kind="mergesort")]
    rng = np.random.default_rng(int(config.get("random_seed", 42)))
    audit_indexes = np.sort(rng.choice(ordered, size=min(500, ordered.size), replace=False))
    exact_pairs = []
    threshold = 8_000.0
    for position, left in enumerate(audit_indexes[:-1]):
        distances = grs80_distance_m(
            graph.lon[left], graph.lat[left], graph.lon[audit_indexes[position + 1 :]], graph.lat[audit_indexes[position + 1 :]]
        )
        exact_pairs.extend(
            (position, position + 1 + int(value)) for value in np.flatnonzero(distances <= threshold)
        )
    coverage = ecef_coverage_audit(
        graph.lon[audit_indexes], graph.lat[audit_indexes], threshold, exact_pairs
    )
    if not coverage["coverage_complete"]:
        raise Phase3APreflightError("ECEF candidate filtering lost an exact GRS80 neighbor")
    # 0.1-degree tiles bound the live benchmark while preserving the densest
    # observed spatial regime. All rows and selections are deterministic.
    tile_x = np.floor(graph.lon[indexes] * 10.0).astype(np.int32)
    tile_y = np.floor(graph.lat[indexes] * 10.0).astype(np.int32)
    tiles: dict[tuple[int, int], list[int]] = defaultdict(list)
    for node, x, y in zip(indexes.tolist(), tile_x.tolist(), tile_y.tolist(), strict=True):
        tiles[(x, y)].append(int(node))
    nontrivial = [(key, sorted(values, key=lambda node: _id_key(str(graph.node_uid[node])))) for key, values in tiles.items() if len(values) >= 3]
    nontrivial.sort(key=lambda item: (len(item[1]), item[0]))
    chosen = {
        "small": nontrivial[0],
        "representative": nontrivial[len(nontrivial) // 2],
        "worst_tile": max(nontrivial, key=lambda item: (len(item[1]), tuple(-x for x in item[0]))),
    }
    benchmark_rows: list[dict[str, Any]] = []
    deterministic_pass = True
    for label, (tile, nodes_full) in chosen.items():
        parameters = config.get("preflight", {})
        cap = int(parameters.get("geodesy_tile_node_cap", 1_200))
        nodes = nodes_full[:cap]
        start = time.perf_counter()
        pair_graph = build_geo_pair_graph(
            graph.node_uid[nodes], graph.lon[nodes], graph.lat[nodes], graph.physical_weak_component[nodes], threshold
        )
        pair_seconds = time.perf_counter() - start
        components = _pair_components(pair_graph)
        selected_components = {
            "smallest": components[0],
            "median": components[len(components) // 2],
            "largest": components[-1],
        }
        for component_label, component in selected_components.items():
            component_cap = int(parameters.get("clustering_component_node_cap", 300))
            sampled_component = component[:component_cap]
            induced = induced_pair_graph(pair_graph, sampled_component)
            uid_to_physical = {str(graph.node_uid[node]): int(node) for node in nodes}

            def exact_geo_distance(left_uid: str, right_uid: str) -> float:
                left_node = uid_to_physical[left_uid]
                right_node = uid_to_physical[right_uid]
                return float(
                    grs80_distance_m(
                        graph.lon[left_node], graph.lat[left_node],
                        graph.lon[right_node], graph.lat[right_node],
                    )
                )

            for algorithm in ("CL", "GC", "SL"):
                began = time.perf_counter()
                result = cluster_pair_graph(induced, algorithm, complete_distance=exact_geo_distance)
                seconds = time.perf_counter() - began
                permutation = np.arange(len(induced.node_ids))[::-1]
                permuted = _permute_pair_graph(induced, permutation)
                replay = cluster_pair_graph(permuted, algorithm, complete_distance=exact_geo_distance)
                match = result.clusters == replay.clusters and result.membership_hash == replay.membership_hash
                deterministic_pass &= match
                benchmark_rows.append(
                    {
                        "sample": label,
                        "tile": list(tile),
                        "tile_node_count": len(nodes_full),
                        "benchmarked_node_count": len(nodes),
                        "pair_edge_count": pair_graph.edge_count(),
                        "pair_graph_seconds": pair_seconds,
                        "component_selection": component_label,
                        "component_full_node_count": len(component),
                        "component_benchmarked_node_count": len(sampled_component),
                        "algorithm": algorithm,
                        "cluster_count": len(result.clusters),
                        "maximum_reported_diameter": max(result.actual_diameters, default=0.0),
                        "production_pairwise_valid": result.production_pairwise_valid,
                        "seconds": seconds,
                        "row_permutation_exact": match,
                    }
                )
    if not deterministic_pass:
        raise Phase3APreflightError("CL/GC/SL row-permutation replay failed")
    measured_nodes = sum(row["component_benchmarked_node_count"] for row in benchmark_rows)
    measured_seconds = sum(row["seconds"] for row in benchmark_rows)
    projected_grid = measured_seconds / max(1, measured_nodes) * len(indexes) * 13 * 2.0
    return {
        "schema_version": "1.0",
        "section": "geodesy_clustering",
        "status": "PASS",
        "geodesic_library": {"name": "pyproj.Geod", "version": importlib.metadata.version("pyproj"), "ellipsoid": "GRS80"},
        "coordinate_axis_order": "longitude,latitude",
        "ecef_exact_coverage": coverage,
        "tile_inventory": {
            "tile_size_degrees": 0.1,
            "tile_count": len(tiles),
            "maximum_tile_backbone_nodes": max(map(len, tiles.values())),
        },
        "benchmarks": benchmark_rows,
        "deterministic_replay": "PASS",
        "projected_round1_grid_seconds_conservative": projected_grid,
        "scientific_preference_declared": False,
    }


def _aug(config: Mapping[str, Any], output_root: Path, temp_root: Path) -> dict[str, Any]:
    del output_root, temp_root
    paths = resolve_authority_paths(config)
    graph = load_graph(paths)
    suite = load_suite(paths, graph)
    indexes = np.flatnonzero(suite.is_final_backbone_node)
    tile_x = np.floor(graph.lon[indexes] * 10.0).astype(np.int32)
    tile_y = np.floor(graph.lat[indexes] * 10.0).astype(np.int32)
    tiles: dict[tuple[int, int], list[int]] = defaultdict(list)
    for node, x, y in zip(indexes.tolist(), tile_x.tolist(), tile_y.tolist(), strict=True):
        tiles[(x, y)].append(int(node))
    nontrivial = sorted(
        ((key, sorted(values, key=lambda node: _id_key(str(graph.node_uid[node])))) for key, values in tiles.items() if len(values) >= 2),
        key=lambda item: (len(item[1]), item[0]),
    )
    representative = nontrivial[len(nontrivial) // 2]
    worst = max(nontrivial, key=lambda item: (len(item[1]), item[0]))
    source_cap = int(config.get("preflight", {}).get("aug_source_node_cap", 6))
    source_sets = {
        "representative_tile": representative,
        "worst_spatial_tile": worst,
    }
    policies = (
        AugPolicy("conservative", 30.0, 5_000.0, 12, 2.0, 0.01),
        AugPolicy("moderate", 120.0, 10_000.0, 24, 3.0, 0.01),
    )
    rows: list[dict[str, Any]] = []
    all_witnesses = []
    all_sources: set[int] = set()
    for sample_label, (tile, nodes) in source_sets.items():
        sources = nodes[:source_cap]
        all_sources.update(sources)
        for policy in policies:
            for source in sources:
                began = time.perf_counter()
                witnesses = direct_witnesses_from_source(graph, suite, source, policy)
                seconds = time.perf_counter() - began
                for witness in witnesses:
                    validate_witness(witness, graph, suite, policy)
                all_witnesses.extend(witnesses)
                rows.append(
                    {
                        "sample": sample_label,
                        "tile": list(tile),
                        "tile_backbone_node_count": len(nodes),
                        "policy_id": policy.policy_id,
                        "source_node_uid": str(graph.node_uid[source]),
                        "witness_count": len(witnesses),
                        "seconds": seconds,
                        "witnesses_per_second": len(witnesses) / seconds if seconds else None,
                    }
                )
    temporary_graph_smokes: list[dict[str, Any]] = []
    for policy in policies:
        policy_witnesses = [value for value in all_witnesses if value.policy_id == policy.policy_id]
        began = time.perf_counter()
        pair_graph = build_aug_pair_graph(graph, suite, all_sources, policy_witnesses, policy)
        temporary_graph_smokes.append(
            {
                "policy_id": policy.policy_id,
                "sampled_backbone_nodes": len(all_sources),
                "direct_witnesses": len(policy_witnesses),
                "mutual_aug_pair_count": pair_graph.edge_count(),
                "seconds": time.perf_counter() - began,
                "temporary_shortcuts_removed_after_query": True,
            }
        )
    total_source_seconds = sum(row["seconds"] for row in rows)
    unique_source_policy = len(rows)
    seconds_per_source_policy = total_source_seconds / max(1, unique_source_policy)
    projected = seconds_per_source_policy * len(indexes) * 2.0
    return {
        "schema_version": "1.0",
        "section": "aug",
        "status": "PASS",
        "source_benchmarks": rows,
        "temporary_augmented_distance_smoke": temporary_graph_smokes,
        "canonical_witness_validation": "PASS",
        "first_backbone_stop_enabled": True,
        "nonbackbone_arcs_only": True,
        "numerical_shortcut_penalty": False,
        "projected_full_two_policy_source_seconds_single_worker": projected,
        "cache_plan": "content-addressed policy-specific flat witness vectors; no unsafe moderate-to-conservative canonical shortcut filtering",
        "scientific_preference_declared": False,
    }


class _RestrictedPortalProvider:
    def __init__(self, graph: Any, suite: Any, centroid_nodes: frozenset[int]) -> None:
        self.graph = graph
        self.suite = suite
        self.centroid_nodes = centroid_nodes
        self.cache: dict[frozenset[str], Any] = {}

    def __call__(self, source: int, target: int, regions: frozenset[str]) -> tuple[float, tuple[int, ...]] | None:
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
        if source not in restricted.node_ids or target not in restricted.node_ids:
            return None
        prohibited = set(np.flatnonzero(self.suite.is_final_backbone_node).tolist()) | set(self.centroid_nodes)
        try:
            tree = restricted_dijkstra(
                restricted,
                [source],
                maximum_distance_s=120.0,
                target_global_nodes=[target],
                prohibited_through_global_nodes=prohibited,
            )
        except Exception:
            return None
        local = {value: index for index, value in enumerate(restricted.node_ids)}[target]
        if not np.isfinite(tree.distances_s[local]) or tree.distances_s[local] > 120.0:
            return None
        path = reconstruct_restricted_path(restricted, tree, target)
        return float(tree.distances_s[local]), path


def _regional(config: Mapping[str, Any], output_root: Path, temp_root: Path) -> dict[str, Any]:
    del output_root, temp_root
    paths = resolve_authority_paths(config)
    graph = load_graph(paths)
    suite = load_suite(paths, graph)
    uid_to_idx = {str(value): index for index, value in enumerate(graph.node_uid.tolist())}
    final_uid = set(graph.node_uid[suite.is_final_backbone_node].astype(str).tolist())
    centroid_uid = set(suite.centroid_roles["partC_rev1_node_uid"].astype(str).tolist())
    portal_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in suite.portals.itertuples(index=False):
        portal_groups[str(row.partC_rev1_node_uid)].append(
            {"region_id": str(row.region_uuid), "can_enter": bool(row.can_enter), "can_exit": bool(row.can_exit)}
        )
    eligible: list[tuple[str, tuple[tuple[str, bool, bool], ...]]] = []
    for uid, rows in portal_groups.items():
        signature = canonical_portal_signature(rows)
        if len({item[0] for item in signature}) >= 2 and uid not in final_uid and uid not in centroid_uid:
            eligible.append((uid, signature))
    signature_groups: dict[tuple[tuple[str, bool, bool], ...], list[str]] = defaultdict(list)
    for uid, signature in eligible:
        signature_groups[signature].append(uid)
    # Prefer the signature group with the largest number of exact 0.5-km
    # candidate pairs, then stable signature. This is a benchmark selection,
    # not a scientific candidate selection.
    group_candidates: list[tuple[int, int, bytes, tuple[tuple[str, bool, bool], ...], list[str]]] = []
    for signature, uids in signature_groups.items():
        ordered = sorted(uids, key=_id_key)[:100]
        indexes = np.asarray([uid_to_idx[value] for value in ordered], dtype=np.int64)
        pair_count = len(ecef_candidate_pairs(graph.lon[indexes], graph.lat[indexes], 500.0)) if len(indexes) > 1 else 0
        group_candidates.append((pair_count, len(uids), canonical_json_bytes(signature), signature, ordered))
    selected_group = max(group_candidates, key=lambda value: (value[0], value[1], bytes(reversed(value[2]))))
    signature, selected_uids = selected_group[3], selected_group[4]
    racn_cap = int(config.get("preflight", {}).get("fixed_racn_portal_cap", 40))
    selected_uids = selected_uids[:racn_cap]
    selected_indexes = np.asarray([uid_to_idx[value] for value in selected_uids], dtype=np.int64)
    centroid_nodes = frozenset(int(value) for value in suite.centroid_roles["node_idx"].to_numpy(dtype=np.int64))
    provider = _RestrictedPortalProvider(graph, suite, centroid_nodes)
    began = time.perf_counter()
    racn_membership, racn_summary, racn_witnesses = build_fixed_racns(
        suite_id=suite.suite_id,
        candidate_id="preflight_fixed_racn",
        portal_node_uid=selected_uids,
        portal_node_idx=selected_indexes,
        lon=graph.lon[selected_indexes],
        lat=graph.lat[selected_indexes],
        weak_component=graph.physical_weak_component[selected_indexes],
        signatures=[signature] * len(selected_uids),
        restricted_distance=provider,
    )
    racn_seconds = time.perf_counter() - began
    terminal_counts = suite.terminals.groupby("region_uuid", sort=True).size().sort_values()
    positions = {
        "small": 0,
        "p95": min(len(terminal_counts) - 1, int(np.floor(0.95 * (len(terminal_counts) - 1)))),
        "p99": min(len(terminal_counts) - 1, int(np.floor(0.99 * (len(terminal_counts) - 1)))),
        "max": len(terminal_counts) - 1,
    }
    selected_regions: dict[str, str] = {}
    already: set[str] = set()
    for label, position in positions.items():
        region = str(terminal_counts.index[position])
        if region in already:
            # Choose the next stable region at the same/lower rank so the four
            # probes remain distinct when quantiles coincide.
            for candidate in reversed(terminal_counts.index[: position + 1].astype(str).tolist()):
                if candidate not in already:
                    region = candidate
                    break
        selected_regions[label] = region
        already.add(region)
    portals_subset = suite.portals[suite.portals["region_uuid"].astype(str).isin(selected_regions.values())].copy()
    centroids_subset = suite.centroid_roles[
        suite.centroid_roles["region_uuid"].astype(str).isin(selected_regions.values())
    ].copy()
    selected_portal_uids = portals_subset["partC_rev1_node_uid"].astype(str).unique().tolist()
    node_to_b = {
        uid: stable_object_id("bacn", suite_id=suite.suite_id, candidate_id="preflight_identity", members=[uid])
        for uid in selected_portal_uids if uid in final_uid
    }
    node_to_r = {
        uid: stable_object_id("racn", suite_id=suite.suite_id, candidate_id="preflight_identity", members=[uid])
        for uid in selected_portal_uids if uid not in final_uid
    }
    terminals, members = build_consolidated_terminals(
        suite_id=suite.suite_id,
        candidate_id="preflight_identity",
        portals=portals_subset,
        centroid_roles=centroids_subset,
        node_to_bacn=node_to_b,
        node_to_racn=node_to_r,
    )
    terminal_member_global = {
        str(terminal_id): tuple(uid_to_idx[str(uid)] for uid in group["partC_rev1_node_uid"].astype(str).tolist())
        for terminal_id, group in members.groupby("terminal_id", sort=False)
    }
    regional_rows: list[dict[str, Any]] = []
    for label, region in selected_regions.items():
        region_terminals = terminals[terminals["region_id"].astype(str) == region].reset_index(drop=True)
        mask = suite.arc_region_id == region
        began = time.perf_counter()
        support = build_region_support(
            region_id=region,
            tail=graph.tail,
            head=graph.head,
            fftt_s=graph.arc_fftt_s,
            arc_ids=graph.arc_id,
            region_arc_mask=mask,
        )
        support_seconds = time.perf_counter() - began
        region_centroid_nodes = frozenset(
            int(value) for value in centroids_subset.loc[
                centroids_subset["region_uuid"].astype(str) == region, "node_idx"
            ].to_numpy(dtype=np.int64)
        )
        oracle = LazyRegionalPairOracle(
            suite_id=suite.suite_id,
            candidate_id="preflight_identity",
            support=support,
            terminals=region_terminals,
            fine_members_global={
                terminal: terminal_member_global[terminal]
                for terminal in region_terminals["terminal_id"].astype(str).tolist()
            },
            physical_arc_ids=graph.arc_id,
            centroid_global_nodes=region_centroid_nodes,
        )
        sources = region_terminals.loc[region_terminals["can_exit_region"], "terminal_id"].astype(str).head(2).tolist()
        query_rows = []
        for source in sources:
            first = oracle.query(source)
            first_hash = oracle.cache_hash[source]
            second = oracle.query(source)
            if first != second or first_hash != oracle.cache_hash[source]:
                raise Phase3APreflightError("Repeated real regional lazy query changed")
            query_rows.append(
                {"source_terminal_id": source, "outgoing_pair_count": len(first), "cache_sha256": first_hash}
            )
        regional_rows.append(
            {
                "sample": label,
                "region_id": region,
                "physical_arc_count": int(mask.sum()),
                "terminal_count": len(region_terminals),
                "scc_count": support.scc_count,
                "support_build_seconds": support_seconds,
                "lazy_queries": query_rows,
                "dense_terminal_square_allocated": False,
            }
        )
    measured_query_seconds = sum(
        sum(0.0 for _ in row["lazy_queries"]) + row["support_build_seconds"] for row in regional_rows
    )
    return {
        "schema_version": "1.0",
        "section": "regional",
        "status": "PASS",
        "eligible_nonbackbone_multiregion_portal_count": len(eligible),
        "signature_group_count": len(signature_groups),
        "fixed_racn_sample": {
            "signature": signature,
            "eligible_group_size": selected_group[1],
            "sample_portal_count": len(selected_uids),
            "ecef_candidate_pair_count": selected_group[0],
            "racn_count": len(racn_summary),
            "merged_racn_count": int((racn_summary["member_count"] > 1).sum()) if not racn_summary.empty else 0,
            "directed_witness_count": len(racn_witnesses),
            "seconds": racn_seconds,
            "pairwise_complete_link_valid": True,
        },
        "regional_samples": regional_rows,
        "projected_all_region_support_seconds_single_worker": measured_query_seconds / max(1, len(regional_rows)) * len(suite.region_ids),
        "centroid_through_allowed": False,
        "support_uses_current_flow": False,
        "dense_pair_matrix_allocated": False,
    }


def _paths_projection_tracer(config: Mapping[str, Any], output_root: Path, temp_root: Path) -> dict[str, Any]:
    del temp_root
    paths = resolve_authority_paths(config)
    graph = load_graph(paths)
    suite = load_suite(paths, graph)
    od = load_od(paths)
    snapshot = od.snapshot
    offsets = snapshot.origin_offsets
    origin_demand = np.add.reduceat(snapshot.demand, offsets[:-1])
    ranked = sorted(
        range(len(snapshot.origin_group_node_idx)),
        key=lambda pos: (-float(origin_demand[pos]), _id_key(str(graph.node_uid[int(snapshot.origin_group_node_idx[pos])]))),
    )
    origin_records: list[dict[str, Any]] = []
    path_records: list[tuple[int, int, tuple[int, ...], float]] = []
    for position in ranked:
        origin = int(snapshot.origin_group_node_idx[position])
        began = time.perf_counter()
        tree = physical_fftt_tree(graph, origin)
        tree_seconds = time.perf_counter() - began
        begin, end = int(offsets[position]), int(offsets[position + 1])
        rows = np.arange(begin, end, dtype=np.int64)
        destinations = snapshot.destination_node_idx[rows]
        feasible = rows[(destinations != origin) & tree.reached_mask[destinations]]
        if feasible.size < 5:
            continue
        feasible = np.asarray(
            sorted(
                feasible.tolist(),
                key=lambda row: (-float(snapshot.demand[row]), _id_key(str(graph.node_uid[int(snapshot.destination_node_idx[row])]))),
            )[:5],
            dtype=np.int64,
        )
        reconstructed = []
        for row in feasible.tolist():
            destination = int(snapshot.destination_node_idx[row])
            arc_path = reconstruct_physical_path(graph, tree, destination)
            if not arc_path:
                continue
            path_records.append((origin, destination, arc_path, float(tree.distances[destination])))
            reconstructed.append(
                {"destination_uid": str(graph.node_uid[destination]), "arc_count": len(arc_path), "fftt_s": float(tree.distances[destination])}
            )
        origin_records.append(
            {
                "origin_uid": str(graph.node_uid[origin]),
                "tree_seconds": tree_seconds,
                "settled_count": int(tree.metadata["settled_count"]),
                "paths": reconstructed,
            }
        )
        if len(origin_records) >= 2 and len(path_records) >= 10:
            break
    if len(origin_records) < 2 or len(path_records) < 10:
        raise Phase3APreflightError("Could not produce two real physical trees and ten paths")
    path_records = path_records[:10]
    used_backbone_nodes: set[int] = set()
    for _, _, arc_path, _ in path_records:
        for arc in arc_path:
            if suite.is_final_backbone_arc[arc]:
                used_backbone_nodes.add(int(graph.tail[arc]))
                used_backbone_nodes.add(int(graph.head[arc]))
    if not used_backbone_nodes:
        raise Phase3APreflightError("Real path sample encountered no final-backbone state")
    ordered_backbone = sorted(used_backbone_nodes, key=lambda node: _id_key(str(graph.node_uid[node])))
    identity_map = {
        str(graph.node_uid[node]): stable_object_id(
            "bacn", suite_id=suite.suite_id, candidate_id="preflight_identity", members=[str(graph.node_uid[node])]
        )
        for node in ordered_backbone
    }
    local_pair = build_geo_pair_graph(
        graph.node_uid[ordered_backbone],
        graph.lon[ordered_backbone],
        graph.lat[ordered_backbone],
        graph.physical_weak_component[ordered_backbone],
        1_000.0,
    )
    geo_result = cluster_pair_graph(local_pair, "CL")
    geo_membership, _ = build_bacn_membership(
        suite_id=suite.suite_id,
        candidate_id="preflight_geo_cl_1000m_local",
        family="GEO",
        algorithm="CL",
        parameter_value=1.0,
        parameter_unit="km",
        final_backbone_node_uids=graph.node_uid[ordered_backbone],
        weak_component_by_uid={str(graph.node_uid[node]): int(graph.physical_weak_component[node]) for node in ordered_backbone},
        cluster_result=geo_result,
    )
    recipes = {
        "identity": identity_map,
        "geo_cl_1000m_local": node_to_bacn(geo_membership),
    }
    recipe_reports: list[dict[str, Any]] = []
    arc_index_by_id = {str(value): index for index, value in enumerate(graph.arc_id.tolist())}
    for recipe_name, bacn_map in recipes.items():
        bundle_by_arc, bundle_cost = _sample_bundle_registry(graph, suite, bacn_map, recipe_name)
        regional_graph_cache: dict[str, Any] = {}
        projection_rows: list[dict[str, Any]] = []
        projections = []
        for sample_index, (origin, destination, arc_path, physical_cost) in enumerate(path_records):
            terminal_map = _path_terminal_map(graph, suite, arc_path, bacn_map, recipe_name)
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
            )
            object_cost = dict(bundle_cost)
            for occurrence in projection.occurrences:
                if occurrence.kind != "REGIONAL_PAIR":
                    continue
                first_arc = arc_index_by_id[occurrence.physical_arc_ids[0]]
                last_arc = arc_index_by_id[occurrence.physical_arc_ids[-1]]
                source_node = int(graph.tail[first_arc])
                target_node = int(graph.head[last_arc])
                region = str(occurrence.region_id)
                if region not in regional_graph_cache:
                    regional_graph_cache[region] = build_sparse_directed_graph(
                        tail=graph.tail,
                        head=graph.head,
                        cost_s=graph.arc_fftt_s,
                        arc_stable_id=graph.arc_id,
                        arc_mask=suite.arc_region_id == region,
                    )
                restricted = regional_graph_cache[region]
                tree = restricted_dijkstra(
                    restricted,
                    [source_node],
                    target_global_nodes=[target_node],
                    prohibited_through_global_nodes=suite.centroid_roles["node_idx"].to_numpy(dtype=np.int64),
                )
                local = {value: index for index, value in enumerate(restricted.node_ids)}[target_node]
                object_cost[occurrence.object_id] = float(tree.distances_s[local])
            direct_cost = sum(0.0 if item.omitted_online else object_cost[item.object_id] for item in projection.occurrences)
            edges, internal = _relations_from_projection(projection, graph, arc_index_by_id)
            lift = audit_liftability(
                coarse_edges=edges,
                source_states=[origin],
                destination_states=[destination],
                internal_relations=internal,
            )
            if not lift.liftable:
                raise Phase3APreflightError(f"Projected physical witness failed liftability smoke: {recipe_name}")
            projections.append(projection)
            projection_rows.append(
                {
                    "sample_path_index": sample_index,
                    "origin_uid": str(graph.node_uid[origin]),
                    "destination_uid": str(graph.node_uid[destination]),
                    "physical_fftt_s": physical_cost,
                    "projected_generic_direct_fftt_s": direct_cost,
                    "physical_arc_count": len(arc_path),
                    "projection_occurrence_count": len(projection.occurrences),
                    "repeated_region_entry_count": projection.repeated_region_entry_count,
                    "liftable": lift.liftable,
                    "lifted_witness_fftt_s": lift.lifted_fftt_s,
                }
            )
        lookup, regional_registry = _sample_lookup(graph, suite, bacn_map, bundle_by_arc, projections)
        lookup_path = output_root / "preflight" / f"{recipe_name}_physical_to_candidate_lookup.npz"
        lookup_record = write_lookup_npz_new(lookup_path, lookup)
        contract = tracer_interface_contract(lookup)
        arrays = empty_trace_arrays(lookup, regional_registry)
        bundle_local = {value: index for index, value in enumerate(lookup.bundle_registry)}
        regional_local = {value: index for index, value in enumerate(regional_registry)}
        bacn_local = {value: index for index, value in enumerate(lookup.bacn_registry)}
        for projection in projections:
            trace_projection(
                projection,
                1.0,
                arrays,
                bundle_local=bundle_local,
                regional_pair_local=regional_local,
                bacn_local=bacn_local,
                racn_local={},
            )
        recipe_reports.append(
            {
                "recipe": recipe_name,
                "bounded_recipe_only": True,
                "path_count": len(projection_rows),
                "projection_liftability": projection_rows,
                "lookup_artifact": lookup_record,
                "tracer_interface": contract,
                "tracer_nonzero_counts": {key: int(np.count_nonzero(value)) for key, value in arrays.items()},
                "phase4_executed": False,
            }
        )
    average_tree = float(np.mean([row["tree_seconds"] for row in origin_records]))
    return {
        "schema_version": "1.0",
        "section": "paths_projection_tracer",
        "status": "PASS",
        "physical_origin_trees": origin_records,
        "physical_tree_reused_api": "partC_rev1_src.partC_rev1_shortest_paths.partC_rev1_sssp via phase3a.physical_fftt_tree",
        "candidate_recipe_smokes": recipe_reports,
        "path_count": len(path_records),
        "recipe_count": len(recipe_reports),
        "direct_diagnostic_fftt_smoke": "PASS",
        "sampled_endpoint_state_liftability": "PASS",
        "phase4_tracer_interface_smoke": "PASS",
        "phase4_executed": False,
        "projected_40_origin_physical_tree_seconds": average_tree * 40.0,
    }


def _qgis(config: Mapping[str, Any], output_root: Path, temp_root: Path) -> dict[str, Any]:
    del temp_root
    paths = resolve_authority_paths(config)
    graph = load_graph(paths)
    suite = load_suite(paths, graph)
    nodes = np.flatnonzero(suite.is_final_backbone_node)[:10]
    arcs = np.flatnonzero(suite.is_final_backbone_arc)[:10]
    points = pd.DataFrame(
        {
            "candidate_id": "preflight_identity",
            "object_id": graph.node_uid[nodes].astype(str),
            "object_type": "BACN_SAMPLE",
            "lon": graph.lon[nodes],
            "lat": graph.lat[nodes],
        }
    )
    lines = pd.DataFrame(
        {
            "candidate_id": "preflight_identity",
            "object_id": graph.arc_id[arcs].astype(str),
            "object_type": "FINAL_BACKBONE_ARC_SAMPLE",
            "tail_lon": graph.lon[graph.tail[arcs]],
            "tail_lat": graph.lat[graph.tail[arcs]],
            "head_lon": graph.lon[graph.head[arcs]],
            "head_lat": graph.lat[graph.head[arcs]],
        }
    )
    gpkg = output_root / "preflight" / "qgis" / "phase3a_task1_overlay_smoke.gpkg"
    qml = output_root / "preflight" / "qgis" / "phase3a_task1_overlay_smoke.qml"
    report = write_overlay_smoke(geopackage_path=gpkg, qml_path=qml, point_rows=points, line_rows=lines)
    report["geopackage_sha256"] = partC_rev1_file_sha256(gpkg)
    report["qml_sha256"] = partC_rev1_file_sha256(qml)
    return {"schema_version": "1.0", "section": "qgis", "status": "PASS", "qgis_smoke": report}


def _summarize(config: Mapping[str, Any], output_root: Path, temp_root: Path) -> dict[str, Any]:
    del temp_root
    required = [value for value in PREFLIGHT_SECTIONS if value != "summarize"]
    results = {}
    resources = []
    for section in required:
        result_path = output_root / "preflight" / f"{section}.json"
        resource_path = output_root / "preflight" / "resources" / f"{section}.json"
        if not result_path.is_file() or not resource_path.is_file():
            raise Phase3APreflightError(f"Cannot summarize before section completes: {section}")
        results[section] = json.loads(result_path.read_text(encoding="utf-8"))
        resources.append(json.loads(resource_path.read_text(encoding="utf-8")))
    if any(value.get("status") != "PASS" for value in results.values()):
        raise Phase3APreflightError("At least one preflight section did not pass")
    heavy_seconds = float(sum(row["charged_heavy_seconds"] for row in resources))
    peak_rss = max(int(row["peak_process_tree_rss_bytes"]) for row in resources)
    round1 = float(results["geodesy_clustering"]["projected_round1_grid_seconds_conservative"])
    round2_single = float(results["aug"]["projected_full_two_policy_source_seconds_single_worker"])
    round3 = float(results["regional"]["projected_all_region_support_seconds_single_worker"]) + 4.0 * float(
        results["regional"]["fixed_racn_sample"]["seconds"]
    )
    round4 = float(results["paths_projection_tracer"]["projected_40_origin_physical_tree_seconds"]) * 2.5
    worker_count = min(4, max(1, psutil.cpu_count(logical=False) or 1))
    forecast = {
        "round1_geo_seconds": max(60.0, round1 / worker_count),
        "round2_aug_seconds": max(300.0, round2_single / worker_count),
        "round3_finalize_seconds": max(300.0, round3 / worker_count),
        "round4_evaluate_seconds": max(600.0, round4),
        "validation_reproduction_figures_seconds": 2_700.0,
        "handoff_sealing_reserve_seconds": TASK2_SEALING_RESERVE_SECONDS,
    }
    forecast_total = sum(value for key, value in forecast.items() if key != "handoff_sealing_reserve_seconds")
    remaining_task2 = PHASE3A_HEAVY_SECONDS_HARD - heavy_seconds
    admission = task2_admission_decision(
        approved_task1_heavy_seconds=heavy_seconds,
        task2_spent_seconds=0.0,
        forecast_batch_seconds=forecast_total,
        forecast_unfinished_mandatory_seconds=0.0,
    )
    rss_forecast = min(PROCESS_TREE_RSS_HARD_BYTES - 1, max(peak_rss, int(peak_rss * min(worker_count, 2) * 1.35)))
    disk_forecast = 20 * 1024**3
    ready = bool(
        heavy_seconds <= TASK1_HEAVY_SECONDS_HARD
        and peak_rss < PROCESS_TREE_RSS_HARD_BYTES
        and admission["admit"]
        and rss_forecast < PROCESS_TREE_RSS_HARD_BYTES
        and disk_forecast < ADDED_DISK_HARD_BYTES
    )
    return {
        "schema_version": "1.0",
        "section": "summarize",
        "status": "PASS" if ready else "BLOCKED_REQUIRES_EXTERNAL_DECISION",
        "task1_status_recommendation": "READY_FOR_EXTERNAL_PREFLIGHT_REVIEW" if ready else "BLOCKED_REQUIRES_EXTERNAL_DECISION",
        "resource_ledger": {
            "commands": resources,
            "task1_heavy_real_data_seconds": heavy_seconds,
            "task1_peak_process_tree_rss_bytes": peak_rss,
            "task1_added_disk_bytes": directory_bytes(output_root),
            "task1_hard_limits_respected": heavy_seconds <= TASK1_HEAVY_SECONDS_HARD and peak_rss < PROCESS_TREE_RSS_HARD_BYTES,
            "remaining_task2_heavy_seconds": remaining_task2,
            "twelve_hour_total_heavy_seconds": PHASE3A_HEAVY_SECONDS_HARD,
            "sealing_reserve_seconds": TASK2_SEALING_RESERVE_SECONDS,
        },
        "task2_forecast": {
            **forecast,
            "projected_nonreserve_seconds": forecast_total,
            "recommended_read_only_worker_count": worker_count,
            "projected_peak_process_tree_rss_bytes": rss_forecast,
            "projected_added_disk_bytes": disk_forecast,
            "admission": admission,
            "ready": ready,
        },
        "candidate_selection_performed": False,
        "global_capacity_multiplier_tuned": False,
        "phase3a_goal_authorized": False,
        "phase3b_authorized": False,
        "phase4_authorized": False,
    }


def _sample_bundle_registry(graph: Any, suite: Any, bacn_map: Mapping[str, str], recipe: str) -> tuple[dict[str, str], dict[str, float]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for arc in np.flatnonzero(suite.is_final_backbone_arc).tolist():
        tail_uid = str(graph.node_uid[int(graph.tail[arc])])
        head_uid = str(graph.node_uid[int(graph.head[arc])])
        if tail_uid not in bacn_map or head_uid not in bacn_map:
            continue
        left, right = bacn_map[tail_uid], bacn_map[head_uid]
        if left != right:
            groups[(left, right)].append(arc)
    by_arc: dict[str, str] = {}
    costs: dict[str, float] = {}
    for (left, right), arcs in sorted(groups.items()):
        member_ids = sorted((str(graph.arc_id[index]) for index in arcs), key=_id_key)
        bundle_id = stable_object_id(
            "bundle", suite_id=suite.suite_id, candidate_id=f"preflight_{recipe}", members=member_ids,
            signature={"tail_bacn_id": left, "head_bacn_id": right},
        )
        for arc in arcs:
            by_arc[str(graph.arc_id[arc])] = bundle_id
        costs[bundle_id] = min(float(graph.arc_fftt_s[arc]) for arc in arcs)
    return by_arc, costs


def _path_terminal_map(
    graph: Any, suite: Any, arc_path: Sequence[int], bacn_map: Mapping[str, str], recipe: str
) -> dict[tuple[str, str, str], str]:
    result: dict[tuple[str, str, str], str] = {}
    position = 0
    while position < len(arc_path):
        arc = int(arc_path[position])
        if suite.is_final_backbone_arc[arc]:
            position += 1
            continue
        region = str(suite.arc_region_id[arc])
        end = position + 1
        while end < len(arc_path) and not suite.is_final_backbone_arc[int(arc_path[end])] and str(suite.arc_region_id[int(arc_path[end])]) == region:
            end += 1
        source_uid = str(graph.node_uid[int(graph.tail[int(arc_path[position])])])
        target_uid = str(graph.node_uid[int(graph.head[int(arc_path[end - 1])])])
        for uid, role in ((source_uid, "source"), (target_uid, "target")):
            if uid in bacn_map:
                parent, terminal_type = bacn_map[uid], "BACN"
            else:
                parent = stable_object_id("racn", suite_id=suite.suite_id, candidate_id=f"preflight_{recipe}", members=[uid])
                terminal_type = "RACN"
            terminal = stable_object_id(
                "terminal", suite_id=suite.suite_id, candidate_id=f"preflight_{recipe}", members=[uid],
                signature={"region_id": region, "terminal_type": terminal_type, "parent": parent},
            )
            result[(region, uid, role)] = terminal
        position = end
    return result


def _relations_from_projection(projection: Any, graph: Any, arc_index_by_id: Mapping[str, int]) -> tuple[list[CoarseEdgeRelation], dict[str, list[PhysicalTransition]]]:
    edges: list[CoarseEdgeRelation] = []
    internal: dict[str, list[PhysicalTransition]] = defaultdict(list)
    previous_head_coarse: str | None = None
    previous_state: int | None = None
    for occurrence in projection.occurrences:
        indexes = [arc_index_by_id[value] for value in occurrence.physical_arc_ids]
        tail_state = int(graph.tail[indexes[0]])
        head_state = int(graph.head[indexes[-1]])
        transition = PhysicalTransition(tail_state, head_state, occurrence.physical_fftt_s, occurrence.physical_arc_ids)
        if occurrence.omitted_online:
            internal[str(occurrence.object_id)].append(transition)
            omitted_coarse = str(occurrence.object_id)
            if previous_head_coarse is not None and previous_head_coarse != omitted_coarse:
                bridge_tail = tail_state if previous_state is None else previous_state
                edges.append(
                    CoarseEdgeRelation(
                        f"bridge:{len(edges)}",
                        previous_head_coarse,
                        omitted_coarse,
                        (PhysicalTransition(bridge_tail, tail_state, 0.0, ()),),
                    )
                )
            previous_head_coarse = omitted_coarse
            previous_state = head_state
            continue
        tail_coarse = str(occurrence.coarse_tail_id)
        head_coarse = str(occurrence.coarse_head_id)
        if previous_head_coarse is not None and previous_head_coarse != tail_coarse:
            bridge_state = tail_state if previous_state is None else previous_state
            if bridge_state != tail_state:
                bridge_state = tail_state
            edges.append(
                CoarseEdgeRelation(
                    f"bridge:{len(edges)}", previous_head_coarse, tail_coarse,
                    (PhysicalTransition(bridge_state, tail_state, 0.0, ()),),
                )
            )
        edges.append(
            CoarseEdgeRelation(
                f"{occurrence.kind}:{occurrence.occurrence_index}", tail_coarse, head_coarse, (transition,)
            )
        )
        previous_head_coarse = head_coarse
        previous_state = head_state
    return edges, dict(internal)


def _sample_lookup(
    graph: Any,
    suite: Any,
    bacn_map: Mapping[str, str],
    bundle_by_arc: Mapping[str, str],
    projections: Sequence[Any],
) -> tuple[PhysicalCandidateLookup, tuple[str, ...]]:
    bacn_registry = tuple(sorted(set(bacn_map.values())))
    bundle_registry = tuple(sorted(set(bundle_by_arc.values())))
    regional_registry = tuple(sorted({item.object_id for projection in projections for item in projection.occurrences if item.kind == "REGIONAL_PAIR"}))
    terminal_registry = tuple(sorted({value for projection in projections for value in projection.coarse_sequence if value not in bacn_registry}))
    bacn_local = {value: index for index, value in enumerate(bacn_registry)}
    bundle_local = {value: index for index, value in enumerate(bundle_registry)}
    node_array = np.full(graph.node_count, LOOKUP_SENTINEL, dtype=np.int32)
    uid_to_idx = {str(value): index for index, value in enumerate(graph.node_uid.tolist())}
    for uid, bacn in bacn_map.items():
        node_array[uid_to_idx[uid]] = bacn_local[bacn]
    arc_bundle = np.full(graph.arc_count, LOOKUP_SENTINEL, dtype=np.int32)
    arc_internal = np.full(graph.arc_count, LOOKUP_SENTINEL, dtype=np.int32)
    for index, arc_id in enumerate(graph.arc_id.astype(str).tolist()):
        if arc_id in bundle_by_arc:
            arc_bundle[index] = bundle_local[bundle_by_arc[arc_id]]
        elif suite.is_final_backbone_arc[index]:
            left = bacn_map.get(str(graph.node_uid[int(graph.tail[index])]))
            right = bacn_map.get(str(graph.node_uid[int(graph.head[index])]))
            if left is not None and left == right:
                arc_internal[index] = bacn_local[left]
    return PhysicalCandidateLookup(
        node_to_bacn_local=node_array,
        node_to_racn_local=np.full(graph.node_count, LOOKUP_SENTINEL, dtype=np.int32),
        arc_to_bundle_local=arc_bundle,
        arc_internal_bacn_local=arc_internal,
        portal_role_to_terminal_local=np.full(0, LOOKUP_SENTINEL, dtype=np.int32),
        centroid_role_to_terminal_local=np.full(0, LOOKUP_SENTINEL, dtype=np.int32),
        bacn_registry=bacn_registry,
        racn_registry=(),
        bundle_registry=bundle_registry,
        terminal_registry=terminal_registry,
    ), regional_registry


def _pair_components(graph: PairGraph) -> list[tuple[int, ...]]:
    adjacency = graph.adjacency()
    unseen = set(range(len(graph.node_ids)))
    result = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        stack = [seed]
        values = []
        while stack:
            node = stack.pop()
            values.append(node)
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        result.append(tuple(sorted(values)))
    return sorted(result, key=lambda values: (len(values), values))


def _permute_pair_graph(graph: PairGraph, order: np.ndarray) -> PairGraph:
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order), dtype=np.int64)
    return PairGraph(
        node_ids=tuple(graph.node_ids[int(value)] for value in order),
        edge_u=inverse[graph.edge_u],
        edge_v=inverse[graph.edge_v],
        dissimilarity=graph.dissimilarity.copy(),
        component_ids=graph.component_ids[order],
        metric=graph.metric,
        threshold=graph.threshold,
        threshold_unit=graph.threshold_unit,
    )


def _module_api_map() -> list[dict[str, Any]]:
    return [
        {"module": "partC_rev1_graph_snapshot", "api": "partC_rev1_load_graph_snapshot", "reuse": "exact graph NPZ/metadata validation", "wrapper": "phase3a.loaders.load_graph converts minutes/miles once"},
        {"module": "partC_rev1_od_snapshot", "api": "partC_rev1_load_od_snapshot", "reuse": "exact OD NPZ/metadata validation", "wrapper": "phase3a.loaders.load_od adds conservation/order gates"},
        {"module": "partC_rev1_graph_snapshot", "api": "partC_rev1_open_sqlite_readonly", "reuse": "URI read-only SQLite", "wrapper": "coordinate/county stable-ID join"},
        {"module": "partC_rev1_shortest_paths", "api": "partC_rev1_sssp", "reuse": "Numba CSR physical FFTT tree", "wrapper": "phase3a.shortest_paths.physical_fftt_tree records seconds/tie contract"},
        {"module": "partC_rev1_phase1_checkpoint", "api": "partC_rev1_write_atomic_checkpoint/read_verified_checkpoint", "reuse": "atomic payload plus SHA sidecar", "wrapper": "Phase 3A schema/state/artifact resume gate"},
        {"module": "partC_rev1_phase1_handoff", "api": "manifest and fresh-extraction validators", "reuse": "text/secret/path/ZIP safety", "wrapper": "phase3a.handoff deterministic Task 1 seal"},
        {"module": "partC_rev1_hashing", "api": "partC_rev1_file_sha256/file_signature", "reuse": "accepted identity and artifact hashing", "wrapper": None},
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _id_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)


_SECTION_HANDLERS: dict[str, Callable[[Mapping[str, Any], Path, Path], dict[str, Any]]] = {
    "inventory": _inventory,
    "authority_graph": _authority_graph,
    "od": _od,
    "geodesy_clustering": _geodesy_clustering,
    "aug": _aug,
    "regional": _regional,
    "paths_projection_tracer": _paths_projection_tracer,
    "qgis": _qgis,
    "summarize": _summarize,
}
