"""Deterministic worker orchestration for exact physical AON plus tracing."""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import psutil

from partC_rev1_src.partC_rev1_graph_snapshot import PartCRev1GraphSnapshot
from partC_rev1_src.partC_rev1_shortest_paths import partC_rev1_arc_tie_rank

from .kernels import aon_trace_range_kernel
from .models import AONTraceResult, AggregatedOD, SparseFlowVector, SuiteTraceMap
from .tracing import PackedTraceMaps, assemble_suite_aon_trace, pack_trace_maps


@dataclass(frozen=True, slots=True)
class AONExecutionConfig:
    worker_count: int = 1
    max_origins: int | None = None
    tie_tolerance: float = 1e-12
    coarse_hash_capacity: int | None = None
    fine_hash_capacity: int | None = None


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value - 1).bit_length())


def _default_hash_capacities(
    trace_maps: Sequence[SuiteTraceMap],
    *,
    origin_count_per_worker: int,
) -> tuple[int, int]:
    if not trace_maps:
        return 1, 1
    max_conceptual = max(item.conceptual_regional_pair_count for item in trace_maps)
    event_bound = max(4096, int(origin_count_per_worker) * 8192)
    coarse_target = min(max_conceptual * 2, event_bound)
    coarse = _next_power_of_two(max(2048, coarse_target))
    fine = _next_power_of_two(max(4096, min(max_conceptual * 4, event_bound * 2)))
    return coarse, fine


def _validate_power_of_two(value: int, name: str) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two")


def _rss_bytes() -> int:
    process = psutil.Process()
    total = int(process.memory_info().rss)
    for child in process.children(recursive=True):
        try:
            total += int(child.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _split_origin_ranges(origin_count: int, worker_count: int) -> list[tuple[int, int]]:
    workers = max(1, min(int(worker_count), int(origin_count)))
    boundaries = np.linspace(0, origin_count, workers + 1, dtype=np.int64)
    return [(int(boundaries[index]), int(boundaries[index + 1])) for index in range(workers)]


def run_exact_aon(
    graph: PartCRev1GraphSnapshot,
    od: AggregatedOD,
    cost: np.ndarray,
    *,
    trace_maps: Sequence[SuiteTraceMap] = (),
    config: AONExecutionConfig | None = None,
    arc_tie_rank: np.ndarray | None = None,
    packed_trace_maps: PackedTraceMaps | None = None,
) -> AONTraceResult:
    """Run exact original-origin shortest paths once and trace selected suites."""

    public_started = time.perf_counter()
    cfg = config or AONExecutionConfig()
    weight = np.ascontiguousarray(np.asarray(cost, dtype=np.float64))
    if weight.shape != graph.free_time.shape or np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("AON cost must be one finite nonnegative value per physical arc")
    od.validate()
    total_origins = len(od.origin_group_node_idx)
    selected_origins = total_origins if cfg.max_origins is None else min(total_origins, int(cfg.max_origins))
    if selected_origins <= 0:
        raise ValueError("AON task list must contain at least one physical origin")
    ranges = _split_origin_ranges(selected_origins, cfg.worker_count)
    pack_started = time.perf_counter()
    packed = pack_trace_maps(trace_maps) if packed_trace_maps is None else packed_trace_maps
    pack_seconds = time.perf_counter() - pack_started
    expected_suite_ids = tuple(item.suite_id for item in trace_maps)
    if packed.suite_ids != expected_suite_ids:
        raise ValueError("Packed trace-map suite order differs from the requested trace maps")
    if trace_maps and packed.arc_region_local.shape[1] != len(graph.arc_idx_to_directed_arc_id):
        raise ValueError("Packed trace maps do not align to the physical graph")
    if len(graph.node_idx_to_uid) >= (1 << 19):
        raise ValueError("Physical node index exceeds the frozen fine-pair key encoding")
    if trace_maps and max(item.region_count for item in trace_maps) >= (1 << 13):
        raise ValueError("Region index exceeds the frozen fine-pair key encoding")

    origins_per_worker = max(end - begin for begin, end in ranges)
    default_coarse, default_fine = _default_hash_capacities(
        trace_maps, origin_count_per_worker=origins_per_worker
    )
    coarse_capacity = int(cfg.coarse_hash_capacity or default_coarse)
    fine_capacity = int(cfg.fine_hash_capacity or default_fine)
    _validate_power_of_two(coarse_capacity, "coarse_hash_capacity")
    _validate_power_of_two(fine_capacity, "fine_hash_capacity")
    tie_rank_started = time.perf_counter()
    ranks = np.ascontiguousarray(
        partC_rev1_arc_tie_rank(graph) if arc_tie_rank is None else np.asarray(arc_tie_rank, dtype=np.int64)
    )
    tie_rank_seconds = time.perf_counter() - tie_rank_started
    if ranks.shape != graph.free_time.shape:
        raise ValueError("Arc tie rank does not align to the physical graph")

    kernel_arguments = (
        len(graph.node_idx_to_uid),
        len(graph.arc_idx_to_directed_arc_id),
        np.ascontiguousarray(graph.csr_indptr, dtype=np.int64),
        np.ascontiguousarray(graph.csr_arc_idx, dtype=np.int64),
        np.ascontiguousarray(graph.head, dtype=np.int64),
        weight,
        ranks,
        np.ascontiguousarray(od.origin_group_node_idx, dtype=np.int64),
        np.ascontiguousarray(od.origin_offsets, dtype=np.int64),
        np.ascontiguousarray(od.destination_node_idx, dtype=np.int64),
        np.ascontiguousarray(od.demand, dtype=np.float64),
        np.ascontiguousarray(od.original_row_multiplicity, dtype=np.int64),
    )
    trace_arguments = (
        float(cfg.tie_tolerance),
        packed.arc_region_local,
        packed.arc_internal_bacn_local,
        packed.terminal_keys,
        packed.terminal_values,
        packed.terminal_offsets,
        packed.region_counts,
        packed.bacn_counts,
        coarse_capacity,
        fine_capacity,
    )

    def execute(origin_range: tuple[int, int]) -> tuple[tuple, float]:
        begin, end = origin_range
        started = time.perf_counter()
        result = aon_trace_range_kernel(
            *kernel_arguments,
            begin,
            end,
            *trace_arguments,
        )
        return result, time.perf_counter() - started

    setup_seconds = time.perf_counter() - public_started
    started = time.perf_counter()
    rss_before = _rss_bytes()
    if len(ranges) == 1:
        worker_results = [execute(ranges[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(ranges), thread_name_prefix="phase4-aon") as pool:
            worker_results = list(pool.map(execute, ranges))
    wall_seconds = time.perf_counter() - started
    rss_after = _rss_bytes()

    reduction_started = time.perf_counter()
    overflow = np.zeros((len(trace_maps), 2), dtype=np.int64)
    failures = np.zeros(len(trace_maps), dtype=np.int64)
    for raw, _ in worker_results:
        failures += raw[12]
        overflow += raw[13]
    if np.any(overflow):
        raise RuntimeError(
            "Trace hash capacity exhausted; rerun this write-new benchmark with explicitly larger capacities: "
            f"{overflow.tolist()}"
        )

    physical_flow = np.zeros(len(graph.arc_idx_to_directed_arc_id), dtype=np.float64)
    for raw, _ in worker_results:
        physical_flow += raw[0]
    sptt = math.fsum(float(raw[1]) for raw, _ in worker_results)
    reachable = math.fsum(float(raw[2]) for raw, _ in worker_results)
    unreachable = math.fsum(float(raw[3]) for raw, _ in worker_results)
    selected_end = int(od.origin_offsets[selected_origins])
    expected_demand = float(od.demand[:selected_end].sum(dtype=np.float64))
    if not np.isclose(reachable + unreachable, expected_demand, rtol=1e-12, atol=1e-8):
        raise RuntimeError("AON reachable/unreachable demand does not conserve the selected OD demand")

    suite_traces: dict[str, object] = {}
    repeated_unique_result: dict[str, np.ndarray] = {}
    repeated_rows_result: dict[str, np.ndarray] = {}
    failure_result: dict[str, int] = {}
    for suite_index, trace_map in enumerate(trace_maps):
        coarse_parts: list[np.ndarray] = []
        coarse_value_parts: list[np.ndarray] = []
        fine_parts: list[np.ndarray] = []
        fine_value_parts: list[np.ndarray] = []
        repeated_flow = np.zeros(trace_map.region_count, dtype=np.float64)
        repeated_unique = np.zeros(trace_map.region_count, dtype=np.int64)
        repeated_rows = np.zeros(trace_map.region_count, dtype=np.int64)
        bacn_traversal = np.zeros(trace_map.bacn_count, dtype=np.float64)
        for raw, _ in worker_results:
            coarse_mask = raw[4][suite_index] != np.uint64(0xFFFFFFFFFFFFFFFF)
            fine_mask = raw[6][suite_index] != np.uint64(0xFFFFFFFFFFFFFFFF)
            coarse_parts.append(raw[4][suite_index][coarse_mask])
            coarse_value_parts.append(raw[5][suite_index][coarse_mask])
            fine_parts.append(raw[6][suite_index][fine_mask])
            fine_value_parts.append(raw[7][suite_index][fine_mask])
            repeated_flow += raw[8][suite_index, : trace_map.region_count]
            repeated_unique += raw[9][suite_index, : trace_map.region_count]
            repeated_rows += raw[10][suite_index, : trace_map.region_count]
            bacn_traversal += raw[11][suite_index, : trace_map.bacn_count]
        coarse = SparseFlowVector.from_unsorted(
            np.concatenate(coarse_parts) if coarse_parts else np.empty(0, dtype=np.uint64),
            np.concatenate(coarse_value_parts) if coarse_value_parts else np.empty(0, dtype=np.float64),
        )
        fine = SparseFlowVector.from_unsorted(
            np.concatenate(fine_parts) if fine_parts else np.empty(0, dtype=np.uint64),
            np.concatenate(fine_value_parts) if fine_value_parts else np.empty(0, dtype=np.float64),
        )
        if len(coarse.keys):
            entry = (coarse.keys >> np.uint64(32)).astype(np.int64)
            exit_ = (coarse.keys & np.uint64(0xFFFFFFFF)).astype(np.int64)
            if np.any(~trace_map.terminal_can_exit[entry]) or np.any(~trace_map.terminal_can_enter[exit_]):
                failures[suite_index] += 1
        trace = assemble_suite_aon_trace(
            trace_map=trace_map,
            physical_flow=physical_flow,
            coarse_pairs=coarse,
            fine_pairs=fine,
            bacn_traversal_flow=bacn_traversal,
            repeated_region_entry_flow=repeated_flow,
        )
        suite_traces[trace_map.suite_id] = trace
        repeated_unique_result[trace_map.suite_id] = repeated_unique
        repeated_rows_result[trace_map.suite_id] = repeated_rows
        failure_result[trace_map.suite_id] = int(failures[suite_index])

    reduction_seconds = time.perf_counter() - reduction_started
    public_call_seconds = time.perf_counter() - public_started
    result = AONTraceResult(
        physical_flow=physical_flow,
        sptt=float(sptt),
        reachable_demand=float(reachable),
        unreachable_demand=float(unreachable),
        processed_origins=int(selected_origins),
        total_origins=int(total_origins),
        suite_traces=suite_traces,  # type: ignore[arg-type]
        repeated_unique_od_occurrences=repeated_unique_result,
        repeated_original_row_occurrences=repeated_rows_result,
        mapping_failure_counts=failure_result,
        timing={
            "aon_wall_seconds": float(wall_seconds),
            "worker_kernel_seconds_sum": float(math.fsum(value for _, value in worker_results)),
            "worker_kernel_seconds_max": float(max(value for _, value in worker_results)),
            "setup_seconds": float(setup_seconds),
            "trace_map_pack_seconds": float(pack_seconds),
            "tie_rank_seconds": float(tie_rank_seconds),
            "worker_orchestration_seconds": float(wall_seconds),
            "reduction_assembly_seconds": float(reduction_seconds),
            "public_call_seconds": float(public_call_seconds),
        },
        worker_count=len(ranges),
        metadata={
            "origin_ranges": [list(value) for value in ranges],
            "coarse_hash_capacity_per_worker_per_suite": coarse_capacity,
            "fine_hash_capacity_per_worker_per_suite": fine_capacity,
            "settled_nodes_total": int(sum(int(raw[14]) for raw, _ in worker_results)),
            "relaxed_arcs_total": int(sum(int(raw[15]) for raw, _ in worker_results)),
            "positive_tree_nodes_total": int(sum(int(raw[16]) for raw, _ in worker_results)),
            "rss_before_bytes": int(rss_before),
            "rss_after_bytes": int(rss_after),
            "trace_suite_ids": [item.suite_id for item in trace_maps],
            "same_physical_tree_for_all_suites": True,
            "packed_trace_maps_reused": packed_trace_maps is not None,
            "arc_tie_rank_reused": arc_tie_rank is not None,
        },
    )
    result.validate(trace_maps)
    return result
