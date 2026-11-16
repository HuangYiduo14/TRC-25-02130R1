"""Bounded explicit path oracle for validating the compiled Phase 4 tree tracer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from partC_rev1_src.partC_rev1_graph_snapshot import PartCRev1GraphSnapshot

from .models import AONTraceResult, SparseFlowVector, SuiteTraceMap
from .tracing import assemble_suite_aon_trace


FINE_NODE_BITS = 19
FINE_REGION_SHIFT = 2 * FINE_NODE_BITS


@dataclass(frozen=True, slots=True)
class ExplicitPathBatch:
    origin_node_idx: np.ndarray
    destination_node_idx: np.ndarray
    path_offsets: np.ndarray
    path_arc_idx: np.ndarray
    demand: np.ndarray
    original_row_multiplicity: np.ndarray
    path_ids: tuple[str, ...]

    def validate(self, graph: PartCRev1GraphSnapshot) -> None:
        count = len(self.demand)
        for name, value in (
            ("origin_node_idx", self.origin_node_idx),
            ("destination_node_idx", self.destination_node_idx),
            ("original_row_multiplicity", self.original_row_multiplicity),
        ):
            if np.asarray(value).shape != (count,):
                raise ValueError(f"Explicit path field {name} is not aligned")
        if self.path_offsets.shape != (count + 1,):
            raise ValueError("Explicit path offsets are not aligned")
        if int(self.path_offsets[0]) != 0 or int(self.path_offsets[-1]) != len(self.path_arc_idx):
            raise ValueError("Explicit path offsets do not cover the arc array")
        if len(self.path_ids) != count:
            raise ValueError("Explicit path IDs are not aligned")
        if np.any(~np.isfinite(self.demand)) or np.any(self.demand <= 0.0):
            raise ValueError("Explicit path demand must be finite and positive")
        if np.any(self.original_row_multiplicity <= 0):
            raise ValueError("Explicit path multiplicity must be positive")
        if np.any(self.origin_node_idx < 0) or np.any(self.origin_node_idx >= len(graph.node_idx_to_uid)):
            raise ValueError("Explicit path contains an invalid origin")
        if np.any(self.destination_node_idx < 0) or np.any(
            self.destination_node_idx >= len(graph.node_idx_to_uid)
        ):
            raise ValueError("Explicit path contains an invalid destination")
        if np.any(self.path_arc_idx < 0) or np.any(
            self.path_arc_idx >= len(graph.arc_idx_to_directed_arc_id)
        ):
            raise ValueError("Explicit path contains an invalid physical arc")


def _terminal_lookup(
    trace_map: SuiteTraceMap,
    region: int,
    node: int,
    node_count: int,
    *,
    mode: int,
) -> int:
    key = np.uint64(region) * np.uint64(node_count) + np.uint64(node)
    position = int(np.searchsorted(trace_map.terminal_region_node_keys, key))
    if (
        position < len(trace_map.terminal_region_node_keys)
        and trace_map.terminal_region_node_keys[position] == key
    ):
        return int(trace_map.terminal_region_node_values[mode, position])
    return -1


def explicit_trace_paths(
    graph: PartCRev1GraphSnapshot,
    paths: ExplicitPathBatch,
    cost: np.ndarray,
    *,
    trace_maps: Sequence[SuiteTraceMap],
) -> AONTraceResult:
    """Trace a bounded list of materialized paths; never use this in the hot path."""

    paths.validate(graph)
    weight = np.asarray(cost, dtype=np.float64)
    if weight.shape != graph.free_time.shape or np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("Explicit oracle cost is invalid")
    node_count = len(graph.node_idx_to_uid)
    physical = np.zeros(len(graph.arc_idx_to_directed_arc_id), dtype=np.float64)
    sptt_terms: list[float] = []
    failures: dict[str, list[dict[str, Any]]] = {item.suite_id: [] for item in trace_maps}
    coarse_keys: dict[str, list[np.uint64]] = {item.suite_id: [] for item in trace_maps}
    coarse_values: dict[str, list[float]] = {item.suite_id: [] for item in trace_maps}
    fine_keys: dict[str, list[np.uint64]] = {item.suite_id: [] for item in trace_maps}
    fine_values: dict[str, list[float]] = {item.suite_id: [] for item in trace_maps}
    repeated_flow = {
        item.suite_id: np.zeros(item.region_count, dtype=np.float64) for item in trace_maps
    }
    repeated_unique = {
        item.suite_id: np.zeros(item.region_count, dtype=np.int64) for item in trace_maps
    }
    repeated_rows = {
        item.suite_id: np.zeros(item.region_count, dtype=np.int64) for item in trace_maps
    }
    bacn_traversal = {
        item.suite_id: np.zeros(item.bacn_count, dtype=np.float64) for item in trace_maps
    }
    witness_rows: list[dict[str, Any]] = []

    for path_index in range(len(paths.demand)):
        begin = int(paths.path_offsets[path_index])
        end = int(paths.path_offsets[path_index + 1])
        arcs = paths.path_arc_idx[begin:end]
        origin = int(paths.origin_node_idx[path_index])
        destination = int(paths.destination_node_idx[path_index])
        amount = float(paths.demand[path_index])
        multiplicity = int(paths.original_row_multiplicity[path_index])
        if len(arcs) == 0:
            if origin != destination:
                raise ValueError(f"Nontrivial explicit path {paths.path_ids[path_index]} is empty")
        else:
            if int(graph.tail[int(arcs[0])]) != origin or int(graph.head[int(arcs[-1])]) != destination:
                raise ValueError(f"Explicit path {paths.path_ids[path_index]} has wrong endpoints")
            if len(arcs) > 1 and np.any(graph.head[arcs[:-1]] != graph.tail[arcs[1:]]):
                raise ValueError(f"Explicit path {paths.path_ids[path_index]} is discontinuous")
            np.add.at(physical, arcs, amount)
            sptt_terms.append(amount * float(weight[arcs].sum(dtype=np.float64)))

        witness: dict[str, Any] = {
            "path_id": paths.path_ids[path_index],
            "origin_node_idx": origin,
            "destination_node_idx": destination,
            "physical_arc_count": int(len(arcs)),
            "physical_cost": float(weight[arcs].sum(dtype=np.float64)) if len(arcs) else 0.0,
            "suite_walks": {},
        }
        for trace_map in trace_maps:
            suite_id = trace_map.suite_id
            seen_regions: set[int] = set()
            regional_segments: list[dict[str, int]] = []
            bacn_segments: list[int] = []
            position = 0
            while position < len(arcs):
                arc = int(arcs[position])
                region = int(trace_map.arc_region_local[arc])
                bacn = int(trace_map.arc_internal_bacn_local[arc])
                if region >= 0:
                    segment_begin = position
                    position += 1
                    while (
                        position < len(arcs)
                        and int(trace_map.arc_region_local[int(arcs[position])]) == region
                    ):
                        position += 1
                    entry_node = int(graph.tail[int(arcs[segment_begin])])
                    exit_node = int(graph.head[int(arcs[position - 1])])
                    entry_terminal = _terminal_lookup(
                        trace_map,
                        region,
                        entry_node,
                        node_count,
                        mode=2 if entry_node == origin else 0,
                    )
                    exit_terminal = _terminal_lookup(
                        trace_map,
                        region,
                        exit_node,
                        node_count,
                        mode=3 if exit_node == destination else 1,
                    )
                    if entry_terminal < 0 or exit_terminal < 0:
                        failures[suite_id].append(
                            {
                                "path_id": paths.path_ids[path_index],
                                "origin_node_idx": origin,
                                "destination_node_idx": destination,
                                "region_id": trace_map.region_ids[region],
                                "entry_node_idx": entry_node,
                                "exit_node_idx": exit_node,
                                "reason": "missing_consolidated_terminal",
                            }
                        )
                    else:
                        coarse_keys[suite_id].append(
                            (np.uint64(entry_terminal) << np.uint64(32)) | np.uint64(exit_terminal)
                        )
                        coarse_values[suite_id].append(amount)
                        fine_keys[suite_id].append(
                            (np.uint64(region) << np.uint64(FINE_REGION_SHIFT))
                            | (np.uint64(entry_node) << np.uint64(FINE_NODE_BITS))
                            | np.uint64(exit_node)
                        )
                        fine_values[suite_id].append(amount)
                        if (
                            not bool(trace_map.terminal_can_exit[entry_terminal])
                            or not bool(trace_map.terminal_can_enter[exit_terminal])
                        ):
                            failures[suite_id].append(
                                {
                                    "path_id": paths.path_ids[path_index],
                                    "origin_node_idx": origin,
                                    "destination_node_idx": destination,
                                    "region_id": trace_map.region_ids[region],
                                    "entry_terminal": trace_map.terminal_ids[entry_terminal],
                                    "exit_terminal": trace_map.terminal_ids[exit_terminal],
                                    "reason": "terminal_direction_mismatch",
                                }
                            )
                    if region in seen_regions:
                        repeated_flow[suite_id][region] += amount
                        repeated_unique[suite_id][region] += 1
                        repeated_rows[suite_id][region] += multiplicity
                    seen_regions.add(region)
                    regional_segments.append(
                        {
                            "region_local": region,
                            "entry_node_idx": entry_node,
                            "exit_node_idx": exit_node,
                            "entry_terminal_local": entry_terminal,
                            "exit_terminal_local": exit_terminal,
                        }
                    )
                elif bacn >= 0:
                    position += 1
                    while (
                        position < len(arcs)
                        and int(trace_map.arc_internal_bacn_local[int(arcs[position])]) == bacn
                    ):
                        position += 1
                    bacn_traversal[suite_id][bacn] += amount
                    bacn_segments.append(bacn)
                else:
                    position += 1
            witness["suite_walks"][suite_id] = {
                "regional_segments": regional_segments,
                "bacn_internal_segments": bacn_segments,
            }
        witness_rows.append(witness)

    suite_traces: dict[str, Any] = {}
    for trace_map in trace_maps:
        suite_id = trace_map.suite_id
        coarse = SparseFlowVector.from_unsorted(
            np.asarray(coarse_keys[suite_id], dtype=np.uint64),
            np.asarray(coarse_values[suite_id], dtype=np.float64),
        )
        fine = SparseFlowVector.from_unsorted(
            np.asarray(fine_keys[suite_id], dtype=np.uint64),
            np.asarray(fine_values[suite_id], dtype=np.float64),
        )
        suite_traces[suite_id] = assemble_suite_aon_trace(
            trace_map=trace_map,
            physical_flow=physical,
            coarse_pairs=coarse,
            fine_pairs=fine,
            bacn_traversal_flow=bacn_traversal[suite_id],
            repeated_region_entry_flow=repeated_flow[suite_id],
        )

    total_demand = float(np.sum(paths.demand, dtype=np.float64))
    result = AONTraceResult(
        physical_flow=physical,
        sptt=float(math_fsum(sptt_terms)),
        reachable_demand=total_demand,
        unreachable_demand=0.0,
        processed_origins=int(len(np.unique(paths.origin_node_idx))),
        total_origins=int(len(np.unique(paths.origin_node_idx))),
        suite_traces=suite_traces,
        repeated_unique_od_occurrences=repeated_unique,
        repeated_original_row_occurrences=repeated_rows,
        mapping_failure_counts={key: len(value) for key, value in failures.items()},
        timing={},
        worker_count=1,
        metadata={
            "implementation": "bounded_explicit_per_path_oracle",
            "path_count": len(paths.demand),
            "mapping_failures": failures,
            "witnesses": witness_rows,
        },
    )
    result.validate(trace_maps)
    return result


def math_fsum(values: Sequence[float]) -> float:
    # Local wrapper keeps the explicit implementation dependency surface tiny.
    import math

    return float(math.fsum(float(value) for value in values))
