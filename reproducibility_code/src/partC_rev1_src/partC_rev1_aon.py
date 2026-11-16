"""All-or-nothing loading kernels for Prompt 4B assignment solvers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit

from .partC_rev1_graph_snapshot import PartCRev1GraphSnapshot
from .partC_rev1_od_snapshot import PartCRev1ODSnapshot
from .partC_rev1_progress_logging import PartCRev1ProgressReporter
from .partC_rev1_runtime_guard import partC_rev1_project_runtime, partC_rev1_resource_snapshot
from .partC_rev1_shortest_paths import (
    _partC_rev1_numba_dijkstra,
    partC_rev1_arc_tie_rank,
    partC_rev1_sssp,
)


@dataclass(slots=True)
class PartCRev1AONResult:
    flow: np.ndarray
    sptt: float
    reachable_demand: float
    unreachable_demand: float
    processed_origins: int
    total_origins: int
    shortest_path_runtime_seconds: float
    loading_runtime_seconds: float
    metadata: dict[str, Any]


@njit(cache=True)
def _partC_rev1_numba_aon_range(
    n_nodes: int,
    n_arcs: int,
    indptr: np.ndarray,
    csr_arc_idx: np.ndarray,
    head: np.ndarray,
    cost: np.ndarray,
    arc_tie_rank: np.ndarray,
    origin_group_node_idx: np.ndarray,
    origin_offsets: np.ndarray,
    destination_node_idx: np.ndarray,
    demand: np.ndarray,
    group_start: int,
    group_end: int,
    tie_tol: float,
) -> tuple[np.ndarray, float, float, float, int]:
    flow = np.zeros(n_arcs, dtype=np.float64)
    node_load = np.zeros(n_nodes, dtype=np.float64)
    touched = np.empty(n_nodes, dtype=np.int64)
    sptt = 0.0
    reachable = 0.0
    unreachable = 0.0
    settled_total = 0
    for group_pos in range(group_start, group_end):
        source = int(origin_group_node_idx[group_pos])
        dist, pred_node, pred_arc, settled_order, settled_count = _partC_rev1_numba_dijkstra(
            n_nodes,
            indptr,
            csr_arc_idx,
            head,
            cost,
            arc_tie_rank,
            source,
            tie_tol,
        )
        settled_total += settled_count
        touched_count = 0
        od_start = int(origin_offsets[group_pos])
        od_end = int(origin_offsets[group_pos + 1])
        for od_pos in range(od_start, od_end):
            dest = int(destination_node_idx[od_pos])
            q = float(demand[od_pos])
            if np.isfinite(dist[dest]):
                reachable += q
                sptt += q * dist[dest]
                if node_load[dest] == 0.0:
                    touched[touched_count] = dest
                    touched_count += 1
                node_load[dest] += q
            else:
                unreachable += q
        for settled_pos in range(settled_count - 1, -1, -1):
            node = int(settled_order[settled_pos])
            z = node_load[node]
            if z == 0.0:
                continue
            if node != source:
                arc_idx = int(pred_arc[node])
                parent = int(pred_node[node])
                if arc_idx >= 0 and parent >= 0:
                    flow[arc_idx] += z
                    if node_load[parent] == 0.0:
                        touched[touched_count] = parent
                        touched_count += 1
                    node_load[parent] += z
        for idx in range(touched_count):
            node_load[int(touched[idx])] = 0.0
    return flow, sptt, reachable, unreachable, settled_total


def _partC_rev1_aon_numba(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    cost_array: np.ndarray,
    *,
    max_origins: int | None,
    origin_batch_size: int,
    reporter: PartCRev1ProgressReporter | None,
    tie_tol: float,
) -> PartCRev1AONResult:
    total_origins = len(od.origin_group_node_idx)
    selected_origins = min(total_origins, int(max_origins)) if max_origins is not None else total_origins
    started = time.perf_counter()
    flow = np.zeros(len(graph.arc_idx_to_directed_arc_id), dtype=np.float64)
    sptt = 0.0
    reachable = 0.0
    unreachable = 0.0
    unit_seconds: list[float] = []
    ranks = partC_rev1_arc_tie_rank(graph)
    for batch_start in range(0, selected_origins, origin_batch_size):
        batch_end = min(selected_origins, batch_start + origin_batch_size)
        batch_started = time.perf_counter()
        batch_flow, batch_sptt, batch_reachable, batch_unreachable, settled_total = _partC_rev1_numba_aon_range(
            len(graph.node_idx_to_uid),
            len(graph.arc_idx_to_directed_arc_id),
            graph.csr_indptr.astype(np.int64),
            graph.csr_arc_idx.astype(np.int64),
            graph.head.astype(np.int64),
            np.asarray(cost_array, dtype=np.float64),
            ranks,
            od.origin_group_node_idx.astype(np.int64),
            od.origin_offsets.astype(np.int64),
            od.destination_node_idx.astype(np.int64),
            od.demand.astype(np.float64),
            batch_start,
            batch_end,
            tie_tol,
        )
        batch_elapsed = time.perf_counter() - batch_started
        flow += batch_flow
        sptt += batch_sptt
        reachable += batch_reachable
        unreachable += batch_unreachable
        unit_seconds.append(batch_elapsed / max(batch_end - batch_start, 1))
        if reporter is not None:
            projection = partC_rev1_project_runtime(
                completed=batch_end,
                total=selected_origins,
                started_perf=started,
                completed_unit_seconds=unit_seconds,
            )
            resource = partC_rev1_resource_snapshot(started)
            reporter.heartbeat(
                "aon_origin_batch",
                force=True,
                backend="numba_csr",
                processed=batch_end,
                total=selected_origins,
                elapsed_seconds=projection.elapsed_seconds,
                projected_total_seconds=projection.projected_total_seconds,
                reachable_demand=reachable,
                unreachable_demand=unreachable,
                settled_nodes=settled_total,
                memory_rss_mb=resource.memory_rss_mb,
                temp_disk_mb=resource.temp_disk_mb,
            )
    elapsed = time.perf_counter() - started
    return PartCRev1AONResult(
        flow=flow,
        sptt=float(sptt),
        reachable_demand=float(reachable),
        unreachable_demand=float(unreachable),
        processed_origins=int(selected_origins),
        total_origins=int(total_origins),
        shortest_path_runtime_seconds=float(elapsed),
        loading_runtime_seconds=float(elapsed),
        metadata={
            "backend": "numba_csr",
            "origin_batch_size": int(origin_batch_size),
            "partial": selected_origins < total_origins,
            "runtime_seconds": elapsed,
            "full_run_projected_seconds": elapsed * total_origins / max(selected_origins, 1),
        },
    )


def _partC_rev1_aon_path_backend(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    cost_array: np.ndarray,
    *,
    backend: str,
    max_origins: int | None,
    reporter: PartCRev1ProgressReporter | None,
) -> PartCRev1AONResult:
    total_origins = len(od.origin_group_node_idx)
    selected_origins = min(total_origins, int(max_origins)) if max_origins is not None else total_origins
    flow = np.zeros(len(graph.arc_idx_to_directed_arc_id), dtype=np.float64)
    sptt = 0.0
    reachable = 0.0
    unreachable = 0.0
    started = time.perf_counter()
    unit_seconds: list[float] = []
    for group_pos in range(selected_origins):
        source_started = time.perf_counter()
        od_start = int(od.origin_offsets[group_pos])
        od_end = int(od.origin_offsets[group_pos + 1])
        destinations = od.destination_node_idx[od_start:od_end]
        result = partC_rev1_sssp(
            graph,
            int(od.origin_group_node_idx[group_pos]),
            cost_array,
            backend=backend,
            targets=destinations.tolist(),
        )
        for od_pos in range(od_start, od_end):
            dest = int(od.destination_node_idx[od_pos])
            q = float(od.demand[od_pos])
            if not result.reached_mask[dest]:
                unreachable += q
                continue
            reachable += q
            sptt += q * float(result.distances[dest])
            node = dest
            while node != int(od.origin_group_node_idx[group_pos]):
                arc_idx = int(result.predecessor_arc[node])
                parent = int(result.predecessor_node[node])
                if arc_idx < 0 or parent < 0:
                    break
                flow[arc_idx] += q
                node = parent
        unit_seconds.append(time.perf_counter() - source_started)
        if reporter is not None:
            projection = partC_rev1_project_runtime(
                completed=group_pos + 1,
                total=selected_origins,
                started_perf=started,
                completed_unit_seconds=unit_seconds,
            )
            resource = partC_rev1_resource_snapshot(started)
            reporter.heartbeat(
                "aon_origin_backend_batch",
                backend=backend,
                processed=group_pos + 1,
                total=selected_origins,
                elapsed_seconds=projection.elapsed_seconds,
                projected_total_seconds=projection.projected_total_seconds,
                reachable_demand=reachable,
                unreachable_demand=unreachable,
                memory_rss_mb=resource.memory_rss_mb,
                temp_disk_mb=resource.temp_disk_mb,
            )
    elapsed = time.perf_counter() - started
    full_projection = elapsed * total_origins / max(selected_origins, 1)
    return PartCRev1AONResult(
        flow=flow,
        sptt=float(sptt),
        reachable_demand=float(reachable),
        unreachable_demand=float(unreachable),
        processed_origins=int(selected_origins),
        total_origins=int(total_origins),
        shortest_path_runtime_seconds=float(elapsed),
        loading_runtime_seconds=float(elapsed),
        metadata={
            "backend": backend,
            "partial": selected_origins < total_origins,
            "runtime_seconds": elapsed,
            "full_run_projected_seconds": full_projection,
        },
    )


def partC_rev1_aon_load(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    cost_array: np.ndarray,
    *,
    backend: str = "numba_csr",
    max_origins: int | None = None,
    origin_batch_size: int = 32,
    reporter: PartCRev1ProgressReporter | None = None,
    tie_tol: float = 1e-12,
) -> PartCRev1AONResult:
    cost = np.asarray(cost_array, dtype=np.float64)
    if cost.shape != graph.free_time.shape:
        raise ValueError("cost_array must have one value per graph arc")
    if np.any(~np.isfinite(cost)) or np.any(cost < 0.0):
        raise ValueError("cost_array must contain finite nonnegative weights")
    if backend == "numba_csr":
        return _partC_rev1_aon_numba(
            graph,
            od,
            cost,
            max_origins=max_origins,
            origin_batch_size=max(1, int(origin_batch_size)),
            reporter=reporter,
            tie_tol=tie_tol,
        )
    if backend in {"igraph", "networkx_python"}:
        return _partC_rev1_aon_path_backend(
            graph,
            od,
            cost,
            backend=backend,
            max_origins=max_origins,
            reporter=reporter,
        )
    raise ValueError(f"Unsupported AON backend: {backend}")
