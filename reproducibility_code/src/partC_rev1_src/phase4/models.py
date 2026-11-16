"""Typed Phase 4 numerical and trace artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


UINT64_SENTINEL = np.uint64(0xFFFFFFFFFFFFFFFF)


@dataclass(frozen=True, slots=True)
class AggregatedOD:
    """Exact identical-origin/destination aggregation of the accepted OD table."""

    od_hash: str
    origin_group_node_idx: np.ndarray
    origin_offsets: np.ndarray
    destination_node_idx: np.ndarray
    demand: np.ndarray
    original_row_multiplicity: np.ndarray
    original_row_count: int
    original_total_demand: float
    mapping_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        n = len(self.demand)
        if self.destination_node_idx.shape != (n,):
            raise ValueError("Aggregated destination array has the wrong shape")
        if self.original_row_multiplicity.shape != (n,):
            raise ValueError("Aggregated multiplicity array has the wrong shape")
        if self.origin_offsets.shape != (len(self.origin_group_node_idx) + 1,):
            raise ValueError("Aggregated origin offsets have the wrong shape")
        if int(self.origin_offsets[0]) != 0 or int(self.origin_offsets[-1]) != n:
            raise ValueError("Aggregated origin offsets do not cover all rows")
        if np.any(np.diff(self.origin_offsets) < 0):
            raise ValueError("Aggregated origin offsets are not monotone")
        if np.any(~np.isfinite(self.demand)) or np.any(self.demand <= 0.0):
            raise ValueError("Aggregated demand must be finite and positive")
        if np.any(self.original_row_multiplicity <= 0):
            raise ValueError("Aggregated OD multiplicities must be positive")
        if int(self.original_row_multiplicity.sum(dtype=np.int64)) != int(self.original_row_count):
            raise ValueError("Aggregated OD multiplicities do not recover the original row count")
        if not np.isclose(
            float(self.demand.sum(dtype=np.float64)),
            float(self.original_total_demand),
            rtol=1e-12,
            atol=1e-9,
        ):
            raise ValueError("Aggregated OD demand does not conserve the original total")
        for group in range(len(self.origin_group_node_idx)):
            lo, hi = int(self.origin_offsets[group]), int(self.origin_offsets[group + 1])
            destinations = self.destination_node_idx[lo:hi]
            if len(destinations) > 1 and np.any(destinations[1:] <= destinations[:-1]):
                raise ValueError("Aggregated destinations are not strictly ordered within an origin")


@dataclass(frozen=True, slots=True)
class SuiteTraceMap:
    """Immutable numeric lookup arrays for one accepted Phase 3 suite."""

    suite_id: str
    phase2_suite_id: str
    candidate_id: str
    root: Path
    arc_region_local: np.ndarray
    arc_to_bundle_local: np.ndarray
    arc_internal_bacn_local: np.ndarray
    node_to_bacn_local: np.ndarray
    node_to_racn_local: np.ndarray
    terminal_region_node_keys: np.ndarray
    terminal_region_node_values: np.ndarray
    terminal_can_enter: np.ndarray
    terminal_can_exit: np.ndarray
    terminal_region_local: np.ndarray
    terminal_parent_racn_local: np.ndarray
    terminal_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    bundle_ids: tuple[str, ...]
    bacn_ids: tuple[str, ...]
    racn_ids: tuple[str, ...]
    conceptual_regional_pair_count: int
    conceptual_pair_count_by_region: np.ndarray
    directory_digest: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def terminal_count(self) -> int:
        return len(self.terminal_ids)

    @property
    def region_count(self) -> int:
        return len(self.region_ids)

    @property
    def bundle_count(self) -> int:
        return len(self.bundle_ids)

    @property
    def bacn_count(self) -> int:
        return len(self.bacn_ids)

    @property
    def racn_count(self) -> int:
        return len(self.racn_ids)

    def validate(self, *, node_count: int, arc_count: int) -> None:
        for name, value in (
            ("arc_region_local", self.arc_region_local),
            ("arc_to_bundle_local", self.arc_to_bundle_local),
            ("arc_internal_bacn_local", self.arc_internal_bacn_local),
        ):
            if value.shape != (arc_count,):
                raise ValueError(f"{self.suite_id} {name} does not align to the physical arcs")
        for name, value in (
            ("node_to_bacn_local", self.node_to_bacn_local),
            ("node_to_racn_local", self.node_to_racn_local),
        ):
            if value.shape != (node_count,):
                raise ValueError(f"{self.suite_id} {name} does not align to the physical nodes")
        categories = (
            (self.arc_region_local >= 0).astype(np.int8)
            + (self.arc_to_bundle_local >= 0).astype(np.int8)
            + (self.arc_internal_bacn_local >= 0).astype(np.int8)
        )
        if not np.all(categories == 1):
            counts = dict(zip(*np.unique(categories, return_counts=True), strict=True))
            raise ValueError(f"{self.suite_id} physical arc classification is not exact: {counts}")
        if self.terminal_region_node_keys.ndim != 1:
            raise ValueError(f"{self.suite_id} terminal lookup keys must be one-dimensional")
        if self.terminal_region_node_values.shape != (4, len(self.terminal_region_node_keys)):
            raise ValueError(
                f"{self.suite_id} directional terminal lookup must have four aligned rows"
            )
        if len(self.terminal_region_node_keys) > 1 and np.any(
            self.terminal_region_node_keys[1:] <= self.terminal_region_node_keys[:-1]
        ):
            raise ValueError(f"{self.suite_id} terminal lookup keys are not strictly sorted")
        if self.terminal_can_enter.shape != (self.terminal_count,) or self.terminal_can_exit.shape != (
            self.terminal_count,
        ):
            raise ValueError(f"{self.suite_id} terminal direction arrays have the wrong shape")
        if self.terminal_region_local.shape != (self.terminal_count,):
            raise ValueError(f"{self.suite_id} terminal/region lookup has the wrong shape")
        if np.any(self.terminal_region_local < 0) or np.any(
            self.terminal_region_local >= self.region_count
        ):
            raise ValueError(f"{self.suite_id} terminal/region lookup is invalid")
        if self.terminal_parent_racn_local.shape != (self.terminal_count,):
            raise ValueError(f"{self.suite_id} terminal/R-ACN lookup has the wrong shape")
        if self.conceptual_pair_count_by_region.shape != (self.region_count,):
            raise ValueError(f"{self.suite_id} regional conceptual counts have the wrong shape")
        if int(self.conceptual_pair_count_by_region.sum(dtype=np.int64)) != int(
            self.conceptual_regional_pair_count
        ):
            raise ValueError(f"{self.suite_id} regional conceptual pair totals disagree")


@dataclass(frozen=True, slots=True)
class SparseFlowVector:
    """Sorted sparse uint64-keyed nonnegative flow vector."""

    keys: np.ndarray
    values: np.ndarray

    @staticmethod
    def empty() -> "SparseFlowVector":
        return SparseFlowVector(np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.float64))

    @staticmethod
    def from_unsorted(keys: np.ndarray, values: np.ndarray, *, zero_tol: float = 0.0) -> "SparseFlowVector":
        key_array = np.asarray(keys, dtype=np.uint64)
        value_array = np.asarray(values, dtype=np.float64)
        if key_array.shape != value_array.shape or key_array.ndim != 1:
            raise ValueError("Sparse keys and values must be aligned one-dimensional arrays")
        if len(key_array) == 0:
            return SparseFlowVector.empty()
        if np.any(key_array == UINT64_SENTINEL):
            raise ValueError("Sparse key collides with the reserved sentinel")
        if np.any(~np.isfinite(value_array)) or np.any(value_array < -1e-9):
            raise ValueError("Sparse flows must be finite and nonnegative")
        order = np.argsort(key_array, kind="mergesort")
        ordered_keys = key_array[order]
        ordered_values = value_array[order]
        starts = np.r_[True, ordered_keys[1:] != ordered_keys[:-1]]
        locations = np.flatnonzero(starts)
        reduced_keys = ordered_keys[locations]
        reduced_values = np.add.reduceat(ordered_values, locations)
        keep = reduced_values > float(zero_tol)
        return SparseFlowVector(reduced_keys[keep], reduced_values[keep])

    def validate(self) -> None:
        if self.keys.dtype != np.uint64 or self.values.dtype != np.float64:
            raise ValueError("Sparse flow dtypes must be uint64 and float64")
        if self.keys.shape != self.values.shape or self.keys.ndim != 1:
            raise ValueError("Sparse flow arrays must be aligned and one-dimensional")
        if len(self.keys) > 1 and np.any(self.keys[1:] <= self.keys[:-1]):
            raise ValueError("Sparse flow keys must be strictly increasing")
        if np.any(~np.isfinite(self.values)) or np.any(self.values < -1e-9):
            raise ValueError("Sparse flows must be finite and nonnegative")


def linear_combine_sparse(
    terms: Iterable[tuple[float, SparseFlowVector]], *, zero_tol: float = 1e-14
) -> SparseFlowVector:
    """Return a deterministic nonnegative linear combination of sparse vectors."""

    materialized = [(float(weight), vector) for weight, vector in terms if float(weight) != 0.0]
    if not materialized:
        return SparseFlowVector.empty()
    for weight, vector in materialized:
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("Trace mixture weights must be finite and nonnegative")
        vector.validate()
    keys = np.unique(np.concatenate([vector.keys for _, vector in materialized]))
    values = np.zeros(len(keys), dtype=np.float64)
    for weight, vector in materialized:
        if len(vector.keys):
            values[np.searchsorted(keys, vector.keys)] += weight * vector.values
    keep = values > float(zero_tol)
    return SparseFlowVector(keys[keep], values[keep])


@dataclass(slots=True)
class SuiteLinearTrace:
    """Linear component flows that must mirror the physical convex mixture."""

    suite_id: str
    bundle_flow: np.ndarray
    coarse_regional_pair_flow: SparseFlowVector
    fine_regional_pair_flow: SparseFlowVector
    bacn_internal_arc_flow: np.ndarray
    bacn_internal_traversal_flow: np.ndarray
    terminal_connector_flow: np.ndarray
    racn_interface_flow: np.ndarray
    repeated_region_entry_flow: np.ndarray

    @staticmethod
    def zeros(trace_map: SuiteTraceMap) -> "SuiteLinearTrace":
        return SuiteLinearTrace(
            suite_id=trace_map.suite_id,
            bundle_flow=np.zeros(trace_map.bundle_count, dtype=np.float64),
            coarse_regional_pair_flow=SparseFlowVector.empty(),
            fine_regional_pair_flow=SparseFlowVector.empty(),
            bacn_internal_arc_flow=np.zeros(trace_map.bacn_count, dtype=np.float64),
            bacn_internal_traversal_flow=np.zeros(trace_map.bacn_count, dtype=np.float64),
            terminal_connector_flow=np.zeros(trace_map.terminal_count, dtype=np.float64),
            racn_interface_flow=np.zeros(trace_map.racn_count, dtype=np.float64),
            repeated_region_entry_flow=np.zeros(trace_map.region_count, dtype=np.float64),
        )

    def validate(self, trace_map: SuiteTraceMap) -> None:
        expected = {
            "bundle_flow": trace_map.bundle_count,
            "bacn_internal_arc_flow": trace_map.bacn_count,
            "bacn_internal_traversal_flow": trace_map.bacn_count,
            "terminal_connector_flow": trace_map.terminal_count,
            "racn_interface_flow": trace_map.racn_count,
            "repeated_region_entry_flow": trace_map.region_count,
        }
        for name, length in expected.items():
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (length,) or np.any(~np.isfinite(value)) or np.any(value < -1e-8):
                raise ValueError(f"{self.suite_id} {name} is invalid")
        self.coarse_regional_pair_flow.validate()
        self.fine_regional_pair_flow.validate()


def linear_combine_suite_traces(
    terms: Sequence[tuple[float, SuiteLinearTrace]], *, zero_tol: float = 1e-14
) -> SuiteLinearTrace:
    if not terms:
        raise ValueError("At least one suite trace is required")
    suite_id = terms[0][1].suite_id
    if any(trace.suite_id != suite_id for _, trace in terms):
        raise ValueError("Cannot combine traces from different suites")
    weights = [float(weight) for weight, _ in terms]
    if any(not np.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("Trace mixture weights must be finite and nonnegative")

    def dense(name: str) -> np.ndarray:
        result = np.zeros_like(np.asarray(getattr(terms[0][1], name), dtype=np.float64))
        for weight, trace in terms:
            value = np.asarray(getattr(trace, name), dtype=np.float64)
            if value.shape != result.shape:
                raise ValueError(f"Trace field {name} differs in shape")
            result += float(weight) * value
        result[np.abs(result) <= zero_tol] = 0.0
        return result

    return SuiteLinearTrace(
        suite_id=suite_id,
        bundle_flow=dense("bundle_flow"),
        coarse_regional_pair_flow=linear_combine_sparse(
            [(weight, trace.coarse_regional_pair_flow) for weight, trace in terms], zero_tol=zero_tol
        ),
        fine_regional_pair_flow=linear_combine_sparse(
            [(weight, trace.fine_regional_pair_flow) for weight, trace in terms], zero_tol=zero_tol
        ),
        bacn_internal_arc_flow=dense("bacn_internal_arc_flow"),
        bacn_internal_traversal_flow=dense("bacn_internal_traversal_flow"),
        terminal_connector_flow=dense("terminal_connector_flow"),
        racn_interface_flow=dense("racn_interface_flow"),
        repeated_region_entry_flow=dense("repeated_region_entry_flow"),
    )


@dataclass(slots=True)
class AONTraceResult:
    physical_flow: np.ndarray
    sptt: float
    reachable_demand: float
    unreachable_demand: float
    processed_origins: int
    total_origins: int
    suite_traces: dict[str, SuiteLinearTrace]
    repeated_unique_od_occurrences: dict[str, np.ndarray]
    repeated_original_row_occurrences: dict[str, np.ndarray]
    mapping_failure_counts: dict[str, int]
    timing: dict[str, float]
    worker_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, trace_maps: Sequence[SuiteTraceMap]) -> None:
        if np.any(~np.isfinite(self.physical_flow)) or np.any(self.physical_flow < -1e-8):
            raise ValueError("Physical AON flow is invalid")
        if not np.isfinite(self.sptt) or self.sptt < 0.0:
            raise ValueError("SPTT is invalid")
        if not np.isfinite(self.reachable_demand) or not np.isfinite(self.unreachable_demand):
            raise ValueError("Reachability demand totals are invalid")
        expected_ids = {item.suite_id for item in trace_maps}
        if set(self.suite_traces) != expected_ids:
            raise ValueError("AON suite trace scope differs from the requested scope")
        for trace_map in trace_maps:
            self.suite_traces[trace_map.suite_id].validate(trace_map)
            if int(self.mapping_failure_counts.get(trace_map.suite_id, 0)) != 0:
                raise ValueError(f"{trace_map.suite_id} has physical-to-ACNHN mapping failures")
