"""Bounded exact SCC and OD-reachability reconstruction for Phase 5 R0 v2.

The implementation deliberately keeps the accepted physical graph in its
existing CSR representation.  It builds the transpose directly from the
accepted ``tail``/``head`` arrays, uses iterative (non-recursive) compiled
depth-first traversals, and then independently audits every OD row whose
endpoints have different strong-component labels with exhaustive directed
breadth-first search.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil
from numba import njit

from partC_rev1_src.phase4.models import AggregatedOD


SCC_ALGORITHM_ID = "NUMBA_ITERATIVE_KOSARAJU_SCC_V1"
REACHABLE_OD_CACHE_SCHEMA_ID = "partC.phase5.reachable_od_supply.v2"
LABEL_PAYLOAD_DTYPE = "<i4"


@dataclass(frozen=True, slots=True)
class SCCBenchmark:
    algorithm_id: str
    node_count: int
    arc_count: int
    scc_count: int
    label_payload_dtype: str
    label_payload_sha256: str
    warmup_completed: bool
    warmup_compiled_this_call: bool
    warmup_seconds: float
    reverse_csr_seconds: float
    reverse_csr_peak_process_rss_bytes: int
    reverse_csr_validation_seconds: float
    first_pass_seconds: float
    first_pass_validation_seconds: float
    second_pass_seconds: float
    second_pass_validation_seconds: float
    canonicalization_and_hash_seconds: float
    total_timed_seconds: float
    reverse_csr_stable_original_arc_order: bool

    def to_dict(self) -> dict[str, bool | float | int | str]:
        return {
            "algorithm_id": self.algorithm_id,
            "node_count": self.node_count,
            "arc_count": self.arc_count,
            "scc_count": self.scc_count,
            "label_payload_dtype": self.label_payload_dtype,
            "label_payload_sha256": self.label_payload_sha256,
            "warmup_completed": self.warmup_completed,
            "warmup_compiled_this_call": self.warmup_compiled_this_call,
            "warmup_seconds": self.warmup_seconds,
            "reverse_csr_seconds": self.reverse_csr_seconds,
            "reverse_csr_peak_process_rss_bytes": (
                self.reverse_csr_peak_process_rss_bytes
            ),
            "reverse_csr_validation_seconds": self.reverse_csr_validation_seconds,
            "first_pass_seconds": self.first_pass_seconds,
            "first_pass_validation_seconds": self.first_pass_validation_seconds,
            "second_pass_seconds": self.second_pass_seconds,
            "second_pass_validation_seconds": self.second_pass_validation_seconds,
            "first_and_second_pass_seconds": (
                self.first_pass_seconds + self.second_pass_seconds
            ),
            "canonicalization_and_hash_seconds": (
                self.canonicalization_and_hash_seconds
            ),
            "total_timed_seconds": self.total_timed_seconds,
            "reverse_csr_stable_original_arc_order": (
                self.reverse_csr_stable_original_arc_order
            ),
        }


@dataclass(frozen=True, slots=True)
class SCCResult:
    scc_count: int
    labels: np.ndarray
    label_payload_sha256: str
    benchmark: SCCBenchmark


@dataclass(frozen=True, slots=True)
class CrossSCCOriginAudit:
    origin_group_index: int
    origin_node_idx: int
    candidate_row_count: int
    candidate_demand: float
    visited_node_count: int
    traversed_arc_count: int
    reachable_candidate_row_count: int
    reachable_candidate_demand: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "origin_group_index": self.origin_group_index,
            "origin_node_idx": self.origin_node_idx,
            "candidate_row_count": self.candidate_row_count,
            "candidate_demand": self.candidate_demand,
            "visited_node_count": self.visited_node_count,
            "traversed_arc_count": self.traversed_arc_count,
            "reachable_candidate_row_count": self.reachable_candidate_row_count,
            "reachable_candidate_demand": self.reachable_candidate_demand,
        }


@dataclass(frozen=True, slots=True)
class CrossSCCAudit:
    audit_id: str
    candidate_origin_count: int
    candidate_row_count: int
    candidate_demand: float
    audited_origin_count: int
    audited_row_count: int
    reachable_candidate_row_count: int
    reachable_candidate_demand: float
    visited_node_count_total: int
    traversed_arc_count_total: int
    origin_audits: tuple[CrossSCCOriginAudit, ...]
    first_reachable_witness: dict[str, Any] | None

    @property
    def passed(self) -> bool:
        return (
            self.audited_row_count == self.candidate_row_count
            and self.audited_origin_count == self.candidate_origin_count
            and self.reachable_candidate_row_count == 0
            and self.reachable_candidate_demand == 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "status": "PASS" if self.passed else "FAIL",
            "candidate_origin_count": self.candidate_origin_count,
            "candidate_row_count": self.candidate_row_count,
            "candidate_demand": self.candidate_demand,
            "audited_origin_count": self.audited_origin_count,
            "audited_row_count": self.audited_row_count,
            "reachable_candidate_row_count": self.reachable_candidate_row_count,
            "reachable_candidate_demand": self.reachable_candidate_demand,
            "visited_node_count_total": self.visited_node_count_total,
            "traversed_arc_count_total": self.traversed_arc_count_total,
            "origin_audits": [value.to_dict() for value in self.origin_audits],
            "first_reachable_witness": self.first_reachable_witness,
        }


class CrossSCCReachabilityError(RuntimeError):
    """Raised when the independent directed audit contradicts SCC filtering."""

    def __init__(self, audit: CrossSCCAudit) -> None:
        super().__init__(
            "Cross-SCC exact directed audit found reachable accepted OD rows: "
            f"rows={audit.reachable_candidate_row_count}, "
            f"demand={audit.reachable_candidate_demand:.17g}"
        )
        self.audit = audit


@dataclass(frozen=True, slots=True)
class ReachableODSupply:
    """Complete SCC-filtered OD supply, retaining the legacy field prefix."""

    scc_count: int
    scc_labels: np.ndarray
    origin_node_flow: np.ndarray
    destination_node_flow: np.ndarray
    unreachable_demand: float
    reachable_demand: float
    row_count: int
    unreachable_row_count: int
    algorithm_id: str = SCC_ALGORITHM_ID
    label_payload_sha256: str = ""
    label_payload_dtype: str = LABEL_PAYLOAD_DTYPE
    cross_scc_candidate_row_count: int = 0
    cross_scc_candidate_demand: float = 0.0
    cross_scc_reachable_row_count: int = 0
    cross_scc_reachable_demand: float = 0.0
    graph_identity: str = ""
    od_identity: str = ""
    code_sha256: str = ""
    config_sha256: str = ""
    timing_seconds: dict[str, float] | None = None
    exact_audit: dict[str, Any] | None = None
    scc_benchmark: SCCBenchmark | None = None
    cross_scc_audit: CrossSCCAudit | None = None


@njit(cache=True, nogil=True)
def _build_stable_reverse_csr_numba(
    tail: np.ndarray,
    head: np.ndarray,
    node_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(node_count, dtype=np.int64)
    for arc_idx in range(head.size):
        counts[head[arc_idx]] += 1
    indptr = np.empty(node_count + 1, dtype=np.int64)
    indptr[0] = 0
    for node_idx in range(node_count):
        indptr[node_idx + 1] = indptr[node_idx] + counts[node_idx]
    cursor = indptr[:-1].copy()
    reverse_head = np.empty(head.size, dtype=np.int64)
    reverse_arc_idx = np.empty(head.size, dtype=np.int64)
    # Iteration in the original arc order makes every reverse row stable.
    for arc_idx in range(head.size):
        row = head[arc_idx]
        position = cursor[row]
        reverse_head[position] = tail[arc_idx]
        reverse_arc_idx[position] = arc_idx
        cursor[row] += 1
    return indptr, reverse_head, reverse_arc_idx


@njit(cache=True, nogil=True)
def _iterative_finish_order_numba(
    forward_indptr: np.ndarray,
    forward_head: np.ndarray,
) -> np.ndarray:
    node_count = forward_indptr.size - 1
    # 0=UNSEEN, 1=ACTIVE, 2=FINISHED, as required by the R0 v2 contract.
    state = np.zeros(node_count, dtype=np.uint8)
    finish_order = np.empty(node_count, dtype=np.int64)
    stack_node = np.empty(node_count, dtype=np.int64)
    stack_next = np.empty(node_count, dtype=np.int64)
    finish_count = 0

    for root in range(node_count):
        if state[root] != 0:
            continue
        top = 0
        stack_node[0] = root
        stack_next[0] = forward_indptr[root]
        state[root] = 1
        while top >= 0:
            node = stack_node[top]
            position = stack_next[top]
            if position < forward_indptr[node + 1]:
                neighbor = forward_head[position]
                stack_next[top] = position + 1
                if state[neighbor] == 0:
                    state[neighbor] = 1
                    top += 1
                    stack_node[top] = neighbor
                    stack_next[top] = forward_indptr[neighbor]
            else:
                finish_order[finish_count] = node
                finish_count += 1
                state[node] = 2
                top -= 1

    return finish_order


@njit(cache=True, nogil=True)
def _iterative_reverse_components_numba(
    finish_order: np.ndarray,
    reverse_indptr: np.ndarray,
    reverse_head: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    node_count = reverse_indptr.size - 1
    stack_node = np.empty(node_count, dtype=np.int64)
    raw_labels = np.full(node_count, -1, dtype=np.int64)
    component_minimum = np.empty(node_count, dtype=np.int64)
    component_count = 0
    for order_position in range(node_count - 1, -1, -1):
        root = finish_order[order_position]
        if raw_labels[root] >= 0:
            continue
        top = 0
        stack_node[0] = root
        raw_labels[root] = component_count
        minimum = root
        while top >= 0:
            node = stack_node[top]
            top -= 1
            if node < minimum:
                minimum = node
            for position in range(reverse_indptr[node], reverse_indptr[node + 1]):
                neighbor = reverse_head[position]
                if raw_labels[neighbor] < 0:
                    raw_labels[neighbor] = component_count
                    top += 1
                    stack_node[top] = neighbor
        component_minimum[component_count] = minimum
        component_count += 1
    return component_count, raw_labels, component_minimum[:component_count]


@njit(cache=True, nogil=True)
def _audit_cross_scc_groups_numba(
    forward_indptr: np.ndarray,
    forward_head: np.ndarray,
    origin_group_node_idx: np.ndarray,
    origin_offsets: np.ndarray,
    destination_node_idx: np.ndarray,
    cross_scc_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_count = forward_indptr.size - 1
    group_count = origin_group_node_idx.size
    stamps = np.zeros(node_count, dtype=np.int64)
    queue = np.empty(node_count, dtype=np.int64)
    audited_groups = np.empty(group_count, dtype=np.int64)
    visited_counts = np.empty(group_count, dtype=np.int64)
    traversed_arc_counts = np.empty(group_count, dtype=np.int64)
    reachable_cross = np.zeros(destination_node_idx.size, dtype=np.uint8)
    audited_count = 0
    stamp = 0

    for group_idx in range(group_count):
        row_start = origin_offsets[group_idx]
        row_stop = origin_offsets[group_idx + 1]
        has_candidate = False
        for row_idx in range(row_start, row_stop):
            if cross_scc_mask[row_idx]:
                has_candidate = True
                break
        if not has_candidate:
            continue

        stamp += 1
        origin = origin_group_node_idx[group_idx]
        queue_start = 0
        queue_stop = 1
        queue[0] = origin
        stamps[origin] = stamp
        traversed = 0
        while queue_start < queue_stop:
            node = queue[queue_start]
            queue_start += 1
            for position in range(forward_indptr[node], forward_indptr[node + 1]):
                traversed += 1
                neighbor = forward_head[position]
                if stamps[neighbor] != stamp:
                    stamps[neighbor] = stamp
                    queue[queue_stop] = neighbor
                    queue_stop += 1

        for row_idx in range(row_start, row_stop):
            if (
                cross_scc_mask[row_idx]
                and stamps[destination_node_idx[row_idx]] == stamp
            ):
                reachable_cross[row_idx] = 1
        audited_groups[audited_count] = group_idx
        visited_counts[audited_count] = queue_stop
        traversed_arc_counts[audited_count] = traversed
        audited_count += 1

    return (
        audited_groups[:audited_count],
        visited_counts[:audited_count],
        traversed_arc_counts[:audited_count],
        reachable_cross,
    )


@njit(cache=True, nogil=True)
def _directed_bfs_predecessor_numba(
    forward_indptr: np.ndarray,
    forward_head: np.ndarray,
    origin: int,
    destination: int,
) -> np.ndarray:
    node_count = forward_indptr.size - 1
    predecessor = np.full(node_count, -2, dtype=np.int64)
    queue = np.empty(node_count, dtype=np.int64)
    queue_start = 0
    queue_stop = 1
    queue[0] = origin
    predecessor[origin] = -1
    while queue_start < queue_stop:
        node = queue[queue_start]
        queue_start += 1
        for position in range(forward_indptr[node], forward_indptr[node + 1]):
            neighbor = forward_head[position]
            if predecessor[neighbor] == -2:
                predecessor[neighbor] = node
                queue[queue_stop] = neighbor
                queue_stop += 1
    return predecessor


_WARMUP_LOCK = threading.Lock()
_WARMUP_COMPLETE = False


def _call_with_peak_process_rss(function: Callable[[], Any]) -> tuple[Any, int]:
    """Sample this process while a nogil compiled graph kernel executes."""

    process = psutil.Process(os.getpid())
    stop = threading.Event()
    peak = [int(process.memory_info().rss)]

    def sample() -> None:
        while not stop.wait(0.01):
            try:
                peak[0] = max(peak[0], int(process.memory_info().rss))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        result = function()
        peak[0] = max(peak[0], int(process.memory_info().rss))
    finally:
        stop.set()
        sampler.join(timeout=1.0)
    return result, peak[0]


def warm_numba_scc_kernels() -> tuple[bool, float]:
    """Compile every R0 v2 graph kernel on a tiny fixture of production dtypes."""

    global _WARMUP_COMPLETE
    with _WARMUP_LOCK:
        if _WARMUP_COMPLETE:
            return False, 0.0
        started = time.perf_counter()
        tail = np.asarray([0, 1, 1, 2], dtype=np.int64)
        head = np.asarray([1, 0, 2, 2], dtype=np.int64)
        forward_indptr = np.asarray([0, 1, 3, 4], dtype=np.int64)
        forward_head = np.asarray([1, 0, 2, 2], dtype=np.int64)
        reverse_indptr, reverse_head, _ = _build_stable_reverse_csr_numba(
            tail, head, 3
        )
        finish_order = _iterative_finish_order_numba(forward_indptr, forward_head)
        _iterative_reverse_components_numba(
            finish_order, reverse_indptr, reverse_head
        )
        origin_nodes = np.asarray([0], dtype=np.int64)
        origin_offsets = np.asarray([0, 1], dtype=np.int64)
        destinations = np.asarray([2], dtype=np.int64)
        cross_mask = np.asarray([True], dtype=np.bool_)
        _audit_cross_scc_groups_numba(
            forward_indptr,
            forward_head,
            origin_nodes,
            origin_offsets,
            destinations,
            cross_mask,
        )
        _directed_bfs_predecessor_numba(forward_indptr, forward_head, 0, 2)
        elapsed = time.perf_counter() - started
        _WARMUP_COMPLETE = True
        return True, elapsed


def _as_int64_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return np.ascontiguousarray(array, dtype=np.int64)


def _validated_graph_arrays(
    graph: Any,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    tail = _as_int64_vector(graph.tail, "tail")
    head = _as_int64_vector(graph.head, "head")
    forward_indptr = _as_int64_vector(graph.csr_indptr, "csr_indptr")
    forward_head = _as_int64_vector(graph.csr_head, "csr_head")
    node_count = (
        len(graph.node_idx_to_uid)
        if hasattr(graph, "node_idx_to_uid")
        else forward_indptr.size - 1
    )
    if node_count <= 0:
        raise ValueError("The accepted graph must contain at least one node")
    if tail.shape != head.shape:
        raise ValueError("tail and head must have identical shapes")
    if forward_indptr.shape != (node_count + 1,):
        raise ValueError("csr_indptr has the wrong shape")
    if (
        int(forward_indptr[0]) != 0
        or int(forward_indptr[-1]) != forward_head.size
        or np.any(np.diff(forward_indptr) < 0)
    ):
        raise ValueError("csr_indptr is not a complete monotone CSR pointer")
    if forward_head.size != head.size:
        raise ValueError("The forward CSR does not contain every accepted arc")
    if (
        np.any(tail < 0)
        or np.any(head < 0)
        or np.any(tail >= node_count)
        or np.any(head >= node_count)
        or np.any(forward_head < 0)
        or np.any(forward_head >= node_count)
    ):
        raise ValueError("Graph endpoint indices are outside the accepted node range")

    csr_arc_idx: np.ndarray | None = None
    if hasattr(graph, "csr_arc_idx"):
        csr_arc_idx = _as_int64_vector(graph.csr_arc_idx, "csr_arc_idx")
        if csr_arc_idx.shape != head.shape:
            raise ValueError("csr_arc_idx has the wrong shape")
        if (
            np.any(csr_arc_idx < 0)
            or np.any(csr_arc_idx >= head.size)
            or np.unique(csr_arc_idx).size != head.size
        ):
            raise ValueError("csr_arc_idx is not a permutation of accepted arcs")
        if not np.array_equal(forward_head, head[csr_arc_idx]):
            raise ValueError("csr_head is inconsistent with head[csr_arc_idx]")
        row_idx = np.repeat(
            np.arange(node_count, dtype=np.int64), np.diff(forward_indptr)
        )
        if not np.array_equal(row_idx, tail[csr_arc_idx]):
            raise ValueError("csr_arc_idx places an arc under the wrong tail node")
    return node_count, tail, head, forward_indptr, forward_head, csr_arc_idx


def build_stable_reverse_csr(
    tail: np.ndarray,
    head: np.ndarray,
    node_count: int,
    *,
    warmup: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the transpose CSR, retaining original arc order within each row."""

    if warmup:
        warm_numba_scc_kernels()
    tail_i64 = _as_int64_vector(tail, "tail")
    head_i64 = _as_int64_vector(head, "head")
    if tail_i64.shape != head_i64.shape or node_count <= 0:
        raise ValueError("Invalid accepted graph arrays for reverse CSR")
    if (
        np.any(tail_i64 < 0)
        or np.any(head_i64 < 0)
        or np.any(tail_i64 >= node_count)
        or np.any(head_i64 >= node_count)
    ):
        raise ValueError("Graph endpoint indices are outside the node range")
    indptr, reverse_head, reverse_arc_idx = _build_stable_reverse_csr_numba(
        tail_i64, head_i64, int(node_count)
    )
    return indptr, reverse_head, reverse_arc_idx


def label_payload_sha256(labels: np.ndarray) -> str:
    """Hash canonical labels as a contiguous explicit little-endian int32 payload."""

    payload = np.ascontiguousarray(labels, dtype=np.dtype(LABEL_PAYLOAD_DTYPE))
    if payload.ndim != 1:
        raise ValueError("SCC labels must be one-dimensional")
    return hashlib.sha256(payload.tobytes(order="C")).hexdigest()


def compute_numba_iterative_kosaraju_scc(
    graph: Any,
    *,
    progress: Callable[[str], None] | None = None,
) -> SCCResult:
    """Run the exact R0 v2 SCC implementation on an accepted graph."""

    (
        node_count,
        tail,
        head,
        forward_indptr,
        forward_head,
        _,
    ) = _validated_graph_arrays(graph)
    compiled_this_call, warmup_seconds = warm_numba_scc_kernels()
    if progress is not None:
        progress("r0_v2_scc_warmup_complete")

    total_started = time.perf_counter()
    reverse_started = time.perf_counter()
    (
        (reverse_indptr, reverse_head, reverse_arc_idx),
        reverse_peak_rss,
    ) = _call_with_peak_process_rss(
        lambda: _build_stable_reverse_csr_numba(tail, head, node_count)
    )
    reverse_seconds = time.perf_counter() - reverse_started
    # A strict monotone arc-index sequence in each row proves stable insertion.
    reverse_validation_started = time.perf_counter()
    for node_idx in range(node_count):
        lo = int(reverse_indptr[node_idx])
        hi = int(reverse_indptr[node_idx + 1])
        if hi - lo > 1 and np.any(reverse_arc_idx[lo + 1 : hi] <= reverse_arc_idx[lo : hi - 1]):
            raise RuntimeError("Reverse CSR did not preserve original arc order")
    reverse_validation_seconds = time.perf_counter() - reverse_validation_started
    if progress is not None:
        progress("r0_v2_reverse_csr_complete")

    first_started = time.perf_counter()
    finish_order = _iterative_finish_order_numba(forward_indptr, forward_head)
    first_seconds = time.perf_counter() - first_started
    first_validation_started = time.perf_counter()
    if finish_order.shape != (node_count,) or np.unique(finish_order).size != node_count:
        raise RuntimeError("First SCC pass did not finish every node exactly once")
    first_validation_seconds = time.perf_counter() - first_validation_started
    if progress is not None:
        progress("r0_v2_first_iterative_dfs_complete")

    second_started = time.perf_counter()
    component_count, raw_labels, component_minimum = _iterative_reverse_components_numba(
        finish_order, reverse_indptr, reverse_head
    )
    second_seconds = time.perf_counter() - second_started
    second_validation_started = time.perf_counter()
    if (
        component_count <= 0
        or component_count > node_count
        or raw_labels.shape != (node_count,)
        or np.any(raw_labels < 0)
        or np.any(raw_labels >= component_count)
        or np.unique(raw_labels).size != component_count
        or component_minimum.shape != (component_count,)
        or np.unique(component_minimum).size != component_count
        or np.any(component_minimum < 0)
        or np.any(component_minimum >= node_count)
    ):
        raise RuntimeError("Second SCC pass did not assign one valid label per node")
    second_validation_seconds = time.perf_counter() - second_validation_started
    if progress is not None:
        progress("r0_v2_second_iterative_dfs_complete")

    canonical_started = time.perf_counter()
    minimum_order = np.argsort(component_minimum, kind="stable")
    canonical_component_id = np.empty(component_count, dtype=np.int32)
    canonical_component_id[minimum_order] = np.arange(
        component_count, dtype=np.int32
    )
    labels = np.ascontiguousarray(
        canonical_component_id[raw_labels], dtype=np.dtype(LABEL_PAYLOAD_DTYPE)
    )
    digest = label_payload_sha256(labels)
    canonical_seconds = time.perf_counter() - canonical_started
    if np.any(labels < 0) or np.any(labels >= component_count):
        raise RuntimeError("Canonical SCC labels are outside the accepted node range")
    unique_labels = np.unique(labels)
    if unique_labels.size != int(component_count):
        raise RuntimeError("Canonical SCC label count differs from traversal result")
    if not np.array_equal(unique_labels, np.arange(component_count, dtype=np.int32)):
        raise RuntimeError("Canonical SCC labels are not contiguous component ids")
    observed_minimum = np.full(component_count, node_count, dtype=np.int64)
    np.minimum.at(observed_minimum, labels, np.arange(node_count, dtype=np.int64))
    if np.any(observed_minimum[1:] <= observed_minimum[:-1]):
        raise RuntimeError("Canonical SCC ids are not ordered by component minimum")
    total_seconds = time.perf_counter() - total_started
    benchmark = SCCBenchmark(
        algorithm_id=SCC_ALGORITHM_ID,
        node_count=node_count,
        arc_count=head.size,
        scc_count=int(component_count),
        label_payload_dtype=LABEL_PAYLOAD_DTYPE,
        label_payload_sha256=digest,
        warmup_completed=True,
        warmup_compiled_this_call=compiled_this_call,
        warmup_seconds=float(warmup_seconds),
        reverse_csr_seconds=float(reverse_seconds),
        reverse_csr_peak_process_rss_bytes=int(reverse_peak_rss),
        reverse_csr_validation_seconds=float(reverse_validation_seconds),
        first_pass_seconds=float(first_seconds),
        first_pass_validation_seconds=float(first_validation_seconds),
        second_pass_seconds=float(second_seconds),
        second_pass_validation_seconds=float(second_validation_seconds),
        canonicalization_and_hash_seconds=float(canonical_seconds),
        total_timed_seconds=float(total_seconds),
        reverse_csr_stable_original_arc_order=True,
    )
    return SCCResult(int(component_count), labels, digest, benchmark)


def classify_od_same_scc(
    od: AggregatedOD,
    labels: np.ndarray,
    *,
    origin_chunk_size: int = 256,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Classify OD rows in origin chunks without changing accepted row order."""

    if origin_chunk_size <= 0:
        raise ValueError("origin_chunk_size must be positive")
    labels_i64 = _as_int64_vector(labels, "labels")
    origins = _as_int64_vector(od.origin_group_node_idx, "origin_group_node_idx")
    offsets = _as_int64_vector(od.origin_offsets, "origin_offsets")
    destinations = _as_int64_vector(od.destination_node_idx, "destination_node_idx")
    if offsets.shape != (origins.size + 1,):
        raise ValueError("origin_offsets has the wrong shape")
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != destinations.size
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("Every accepted origin group must cover at least one OD row")
    if (
        np.any(origins < 0)
        or np.any(destinations < 0)
        or np.any(origins >= labels_i64.size)
        or np.any(destinations >= labels_i64.size)
    ):
        raise ValueError("OD endpoint indices are outside the SCC label range")

    reachable = np.empty(destinations.size, dtype=np.bool_)
    for group_start in range(0, origins.size, origin_chunk_size):
        group_stop = min(origins.size, group_start + origin_chunk_size)
        row_start = int(offsets[group_start])
        row_stop = int(offsets[group_stop])
        group_counts = np.diff(offsets[group_start : group_stop + 1])
        origin_labels = np.repeat(
            labels_i64[origins[group_start:group_stop]], group_counts
        )
        reachable[row_start:row_stop] = (
            origin_labels == labels_i64[destinations[row_start:row_stop]]
        )
        if progress is not None:
            progress(f"reachability_origins_{group_stop}_of_{origins.size}")
    return reachable


def _witness_path(
    forward_indptr: np.ndarray,
    forward_head: np.ndarray,
    origin: int,
    destination: int,
) -> list[int]:
    predecessor = _directed_bfs_predecessor_numba(
        forward_indptr, forward_head, origin, destination
    )
    if int(predecessor[destination]) == -2:
        return []
    reversed_path = [destination]
    cursor = destination
    while cursor != origin:
        cursor = int(predecessor[cursor])
        if cursor < 0:
            raise RuntimeError("Directed witness predecessor chain is incomplete")
        reversed_path.append(cursor)
    reversed_path.reverse()
    return reversed_path


def audit_cross_scc_od_reachability(
    graph: Any,
    od: AggregatedOD,
    labels: np.ndarray,
    *,
    cross_scc_mask: np.ndarray | None = None,
    progress: Callable[[str], None] | None = None,
) -> CrossSCCAudit:
    """Exhaustively audit every cross-component OD row by directed BFS.

    Candidate rows are grouped by the accepted origin registry, so a complete
    traversal is performed once per distinct candidate origin and reused only
    for that origin's candidate destinations.
    """

    (
        node_count,
        _,
        _,
        forward_indptr,
        forward_head,
        _,
    ) = _validated_graph_arrays(graph)
    labels_i64 = _as_int64_vector(labels, "labels")
    if labels_i64.shape != (node_count,):
        raise ValueError("SCC labels have the wrong node count")
    origins = _as_int64_vector(od.origin_group_node_idx, "origin_group_node_idx")
    offsets = _as_int64_vector(od.origin_offsets, "origin_offsets")
    destinations = _as_int64_vector(od.destination_node_idx, "destination_node_idx")
    demand = np.ascontiguousarray(od.demand, dtype=np.float64)
    expected_cross = ~classify_od_same_scc(
        od, labels_i64, origin_chunk_size=max(1, min(256, origins.size))
    )
    if cross_scc_mask is None:
        cross_mask = expected_cross
    else:
        cross_mask = np.ascontiguousarray(cross_scc_mask, dtype=np.bool_)
        if cross_mask.shape != destinations.shape or not np.array_equal(
            cross_mask, expected_cross
        ):
            raise ValueError("cross_scc_mask does not cover exactly every cross-SCC OD row")
    if demand.shape != destinations.shape or np.any(~np.isfinite(demand)) or np.any(demand <= 0.0):
        raise ValueError("Accepted OD demand must be finite and positive")

    warm_numba_scc_kernels()
    if progress is not None:
        progress("r0_v2_cross_scc_exact_audit_started")
    groups, visited_counts, arc_counts, reachable_u8 = _audit_cross_scc_groups_numba(
        forward_indptr,
        forward_head,
        origins,
        offsets,
        destinations,
        cross_mask,
    )
    reachable_cross = reachable_u8.astype(np.bool_, copy=False)
    if np.any(reachable_cross & ~cross_mask):
        raise RuntimeError("Directed audit marked a non-candidate OD row")

    origin_audits: list[CrossSCCOriginAudit] = []
    audited_rows = 0
    for position, raw_group in enumerate(groups):
        group_idx = int(raw_group)
        row_start = int(offsets[group_idx])
        row_stop = int(offsets[group_idx + 1])
        group_candidates = cross_mask[row_start:row_stop]
        group_reachable = reachable_cross[row_start:row_stop]
        candidate_count = int(np.count_nonzero(group_candidates))
        reachable_count = int(np.count_nonzero(group_reachable))
        audited_rows += candidate_count
        origin_audits.append(
            CrossSCCOriginAudit(
                origin_group_index=group_idx,
                origin_node_idx=int(origins[group_idx]),
                candidate_row_count=candidate_count,
                candidate_demand=float(
                    demand[row_start:row_stop][group_candidates].sum(dtype=np.float64)
                ),
                visited_node_count=int(visited_counts[position]),
                traversed_arc_count=int(arc_counts[position]),
                reachable_candidate_row_count=reachable_count,
                reachable_candidate_demand=float(
                    demand[row_start:row_stop][group_reachable].sum(dtype=np.float64)
                ),
            )
        )

    candidate_count = int(np.count_nonzero(cross_mask))
    candidate_demand = float(demand[cross_mask].sum(dtype=np.float64))
    reachable_count = int(np.count_nonzero(reachable_cross))
    reachable_demand = float(demand[reachable_cross].sum(dtype=np.float64))
    witness: dict[str, Any] | None = None
    if reachable_count:
        row_idx = int(np.flatnonzero(reachable_cross)[0])
        group_idx = int(np.searchsorted(offsets, row_idx, side="right") - 1)
        origin = int(origins[group_idx])
        destination = int(destinations[row_idx])
        path = _witness_path(
            forward_indptr, forward_head, origin, destination
        )
        witness = {
            "aggregated_od_row_index": row_idx,
            "origin_group_index": group_idx,
            "origin_node_idx": origin,
            "destination_node_idx": destination,
            "origin_scc_label": int(labels_i64[origin]),
            "destination_scc_label": int(labels_i64[destination]),
            "demand": float(demand[row_idx]),
            "directed_node_path": path,
            "directed_path_arc_count": max(0, len(path) - 1),
        }
    candidate_origins = int(np.count_nonzero([
        np.any(cross_mask[int(offsets[group_idx]) : int(offsets[group_idx + 1])])
        for group_idx in range(origins.size)
    ]))
    audit = CrossSCCAudit(
        audit_id="EXHAUSTIVE_DIRECTED_BFS_CROSS_SCC_OD_V1",
        candidate_origin_count=candidate_origins,
        candidate_row_count=candidate_count,
        candidate_demand=candidate_demand,
        audited_origin_count=len(origin_audits),
        audited_row_count=audited_rows,
        reachable_candidate_row_count=reachable_count,
        reachable_candidate_demand=reachable_demand,
        visited_node_count_total=int(visited_counts.sum(dtype=np.int64)),
        traversed_arc_count_total=int(arc_counts.sum(dtype=np.int64)),
        origin_audits=tuple(origin_audits),
        first_reachable_witness=witness,
    )
    if audit.audited_origin_count != audit.candidate_origin_count or audit.audited_row_count != audit.candidate_row_count:
        raise RuntimeError("Cross-SCC exact audit did not cover every candidate")
    if progress is not None:
        progress("r0_v2_cross_scc_exact_audit_complete")
    return audit


def reconstruct_reachable_od_supply(
    graph: Any,
    od: AggregatedOD,
    *,
    origin_chunk_size: int = 256,
    progress: Callable[[str], None] | None = None,
    exact_cross_scc_audit: bool = True,
    graph_identity: str = "",
    od_identity: str = "",
) -> ReachableODSupply:
    """Run SCC, chunked OD classification, and the independent exact audit."""

    if not exact_cross_scc_audit:
        raise ValueError("The R0 v2 exact cross-SCC audit cannot be disabled")
    science_started = time.perf_counter()
    od.validate()
    node_count = len(graph.node_idx_to_uid)
    scc = compute_numba_iterative_kosaraju_scc(graph, progress=progress)
    classification_started = time.perf_counter()
    reachable = classify_od_same_scc(
        od,
        scc.labels,
        origin_chunk_size=origin_chunk_size,
        progress=progress,
    )
    classification_seconds = time.perf_counter() - classification_started
    cross_mask = ~reachable
    audit_started = time.perf_counter()
    audit = audit_cross_scc_od_reachability(
        graph,
        od,
        scc.labels,
        cross_scc_mask=cross_mask,
        progress=progress,
    )
    audit_seconds = time.perf_counter() - audit_started

    reduction_started = time.perf_counter()
    demand = np.ascontiguousarray(od.demand, dtype=np.float64)
    reachable_demand_rows = np.where(reachable, demand, 0.0)
    origin_group_flow = np.add.reduceat(
        reachable_demand_rows, np.asarray(od.origin_offsets[:-1], dtype=np.int64)
    )
    origin_node_flow = np.bincount(
        np.asarray(od.origin_group_node_idx, dtype=np.int64),
        weights=origin_group_flow,
        minlength=node_count,
    ).astype(np.float64, copy=False)
    destination_node_flow = np.bincount(
        np.asarray(od.destination_node_idx, dtype=np.int64),
        weights=reachable_demand_rows,
        minlength=node_count,
    ).astype(np.float64, copy=False)
    unreachable_demand = float(demand[cross_mask].sum(dtype=np.float64))
    reachable_demand = float(reachable_demand_rows.sum(dtype=np.float64))
    if not np.isclose(
        reachable_demand + unreachable_demand,
        float(od.original_total_demand),
        rtol=1.0e-12,
        atol=1.0e-7,
    ):
        raise RuntimeError("SCC OD classification does not conserve accepted demand")
    if not np.isclose(
        origin_node_flow.sum(dtype=np.float64),
        reachable_demand,
        rtol=1.0e-12,
        atol=1.0e-7,
    ) or not np.isclose(
        destination_node_flow.sum(dtype=np.float64),
        reachable_demand,
        rtol=1.0e-12,
        atol=1.0e-7,
    ):
        raise RuntimeError("Reachable OD node supplies do not conserve demand")
    reduction_seconds = time.perf_counter() - reduction_started
    timing_seconds = {
        "numba_warmup": float(scc.benchmark.warmup_seconds),
        "reverse_csr": float(scc.benchmark.reverse_csr_seconds),
        "reverse_csr_validation": float(
            scc.benchmark.reverse_csr_validation_seconds
        ),
        "first_iterative_dfs": float(scc.benchmark.first_pass_seconds),
        "first_iterative_dfs_validation": float(
            scc.benchmark.first_pass_validation_seconds
        ),
        "second_iterative_dfs": float(scc.benchmark.second_pass_seconds),
        "second_iterative_dfs_validation": float(
            scc.benchmark.second_pass_validation_seconds
        ),
        "canonicalization_and_label_hash": float(
            scc.benchmark.canonicalization_and_hash_seconds
        ),
        "scc_total_excluding_warmup": float(scc.benchmark.total_timed_seconds),
        "od_classification": float(classification_seconds),
        "exact_cross_scc_audit": float(audit_seconds),
        "reachable_supply_reduction": float(reduction_seconds),
        "r0_science_before_physical_conservation": float(
            time.perf_counter() - science_started
        ),
    }
    return ReachableODSupply(
        scc_count=scc.scc_count,
        scc_labels=scc.labels,
        origin_node_flow=origin_node_flow,
        destination_node_flow=destination_node_flow,
        unreachable_demand=unreachable_demand,
        reachable_demand=reachable_demand,
        row_count=reachable.size,
        unreachable_row_count=int(np.count_nonzero(cross_mask)),
        algorithm_id=SCC_ALGORITHM_ID,
        label_payload_sha256=scc.label_payload_sha256,
        label_payload_dtype=LABEL_PAYLOAD_DTYPE,
        cross_scc_candidate_row_count=audit.candidate_row_count,
        cross_scc_candidate_demand=audit.candidate_demand,
        cross_scc_reachable_row_count=audit.reachable_candidate_row_count,
        cross_scc_reachable_demand=audit.reachable_candidate_demand,
        graph_identity=str(graph_identity),
        od_identity=str(od_identity),
        timing_seconds=timing_seconds,
        exact_audit=audit.to_dict(),
        scc_benchmark=scc.benchmark,
        cross_scc_audit=audit,
    )


_CACHE_MEMBERS = {
    "schema_id",
    "algorithm_id",
    "label_payload_dtype",
    "label_payload_sha256",
    "scc_labels",
    "origin_node_flow",
    "destination_node_flow",
    "scc_count",
    "unreachable_demand",
    "reachable_demand",
    "row_count",
    "unreachable_row_count",
    "cross_scc_candidate_row_count",
    "cross_scc_candidate_demand",
    "cross_scc_reachable_row_count",
    "cross_scc_reachable_demand",
    "graph_identity",
    "od_identity",
    "code_sha256",
    "config_sha256",
    "timing_seconds_json",
}


def _cache_scalar(data: Any, name: str) -> Any:
    value = data[name]
    if value.shape != (1,):
        raise ValueError(f"Reachable-supply cache scalar {name} has the wrong shape")
    return value[0]


def write_reachable_od_supply_cache(
    path: str | Path,
    supply: ReachableODSupply,
) -> None:
    """Atomically persist the complete R0 v2 supply cache."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    if temporary.exists():
        raise FileExistsError(f"Stale reachable-supply temporary exists: {temporary}")
    labels = np.ascontiguousarray(supply.scc_labels, dtype=np.dtype(LABEL_PAYLOAD_DTYPE))
    digest = label_payload_sha256(labels)
    if supply.label_payload_sha256 and supply.label_payload_sha256 != digest:
        raise ValueError("Reachable-supply label hash differs before cache write")
    candidate_rows = int(supply.cross_scc_candidate_row_count)
    candidate_demand = float(supply.cross_scc_candidate_demand)
    # Legacy constructors did not carry these two v2 mirrors.
    if candidate_rows == 0 and supply.unreachable_row_count > 0:
        candidate_rows = int(supply.unreachable_row_count)
    if candidate_demand == 0.0 and supply.unreachable_demand > 0.0:
        candidate_demand = float(supply.unreachable_demand)
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_id=np.asarray([REACHABLE_OD_CACHE_SCHEMA_ID]),
            algorithm_id=np.asarray([SCC_ALGORITHM_ID]),
            label_payload_dtype=np.asarray([LABEL_PAYLOAD_DTYPE]),
            label_payload_sha256=np.asarray([digest]),
            scc_labels=labels,
            origin_node_flow=np.ascontiguousarray(
                supply.origin_node_flow, dtype=np.dtype("<f8")
            ),
            destination_node_flow=np.ascontiguousarray(
                supply.destination_node_flow, dtype=np.dtype("<f8")
            ),
            scc_count=np.asarray([supply.scc_count], dtype=np.dtype("<i8")),
            unreachable_demand=np.asarray(
                [supply.unreachable_demand], dtype=np.dtype("<f8")
            ),
            reachable_demand=np.asarray(
                [supply.reachable_demand], dtype=np.dtype("<f8")
            ),
            row_count=np.asarray([supply.row_count], dtype=np.dtype("<i8")),
            unreachable_row_count=np.asarray(
                [supply.unreachable_row_count], dtype=np.dtype("<i8")
            ),
            cross_scc_candidate_row_count=np.asarray(
                [candidate_rows], dtype=np.dtype("<i8")
            ),
            cross_scc_candidate_demand=np.asarray(
                [candidate_demand], dtype=np.dtype("<f8")
            ),
            cross_scc_reachable_row_count=np.asarray(
                [supply.cross_scc_reachable_row_count], dtype=np.dtype("<i8")
            ),
            cross_scc_reachable_demand=np.asarray(
                [supply.cross_scc_reachable_demand], dtype=np.dtype("<f8")
            ),
            graph_identity=np.asarray([str(supply.graph_identity)]),
            od_identity=np.asarray([str(supply.od_identity)]),
            code_sha256=np.asarray([str(supply.code_sha256)]),
            config_sha256=np.asarray([str(supply.config_sha256)]),
            timing_seconds_json=np.asarray(
                [
                    json.dumps(
                        supply.timing_seconds or {},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ]
            ),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def load_reachable_od_supply_cache(
    path: str | Path,
    *,
    expected_node_count: int | None = None,
    expected_row_count: int | None = None,
    expected_graph_identity: str | None = None,
    expected_od_identity: str | None = None,
    expected_code_sha256: str | None = None,
    expected_config_sha256: str | None = None,
) -> ReachableODSupply:
    """Reopen a v2 cache and reject schema, hash, or conservation defects."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        if set(data.files) != _CACHE_MEMBERS:
            raise ValueError(
                f"Reachable-supply cache members differ: {sorted(data.files)}"
            )
        schema_id = str(_cache_scalar(data, "schema_id"))
        algorithm_id = str(_cache_scalar(data, "algorithm_id"))
        label_dtype = str(_cache_scalar(data, "label_payload_dtype"))
        stored_digest = str(_cache_scalar(data, "label_payload_sha256"))
        graph_identity = str(_cache_scalar(data, "graph_identity"))
        od_identity = str(_cache_scalar(data, "od_identity"))
        code_sha256 = str(_cache_scalar(data, "code_sha256"))
        config_sha256 = str(_cache_scalar(data, "config_sha256"))
        timing_seconds = json.loads(str(_cache_scalar(data, "timing_seconds_json")))
        if not isinstance(timing_seconds, dict) or any(
            not isinstance(key, str)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) < 0.0
            for key, value in timing_seconds.items()
        ):
            raise ValueError("Reachable-supply cache timing payload is invalid")
        labels_raw = data["scc_labels"]
        origin_flow_raw = data["origin_node_flow"]
        destination_flow_raw = data["destination_node_flow"]
        if labels_raw.ndim != 1 or labels_raw.dtype != np.dtype(LABEL_PAYLOAD_DTYPE):
            raise ValueError("Reachable-supply cache has the wrong SCC label dtype")
        if (
            origin_flow_raw.ndim != 1
            or destination_flow_raw.ndim != 1
            or origin_flow_raw.dtype != np.dtype("<f8")
            or destination_flow_raw.dtype != np.dtype("<f8")
        ):
            raise ValueError("Reachable-supply cache has the wrong node-flow dtype")
        labels = labels_raw.copy()
        origin_flow = origin_flow_raw.copy()
        destination_flow = destination_flow_raw.copy()
        result = ReachableODSupply(
            scc_count=int(_cache_scalar(data, "scc_count")),
            scc_labels=labels,
            origin_node_flow=origin_flow,
            destination_node_flow=destination_flow,
            unreachable_demand=float(_cache_scalar(data, "unreachable_demand")),
            reachable_demand=float(_cache_scalar(data, "reachable_demand")),
            row_count=int(_cache_scalar(data, "row_count")),
            unreachable_row_count=int(
                _cache_scalar(data, "unreachable_row_count")
            ),
            algorithm_id=algorithm_id,
            label_payload_sha256=stored_digest,
            label_payload_dtype=label_dtype,
            cross_scc_candidate_row_count=int(
                _cache_scalar(data, "cross_scc_candidate_row_count")
            ),
            cross_scc_candidate_demand=float(
                _cache_scalar(data, "cross_scc_candidate_demand")
            ),
            cross_scc_reachable_row_count=int(
                _cache_scalar(data, "cross_scc_reachable_row_count")
            ),
            cross_scc_reachable_demand=float(
                _cache_scalar(data, "cross_scc_reachable_demand")
            ),
            graph_identity=graph_identity,
            od_identity=od_identity,
            code_sha256=code_sha256,
            config_sha256=config_sha256,
            timing_seconds={key: float(value) for key, value in timing_seconds.items()},
        )

    node_count = labels.size
    if schema_id != REACHABLE_OD_CACHE_SCHEMA_ID:
        raise ValueError("Reachable-supply cache schema id differs")
    if algorithm_id != SCC_ALGORITHM_ID:
        raise ValueError("Reachable-supply cache algorithm id differs")
    if label_dtype != LABEL_PAYLOAD_DTYPE:
        raise ValueError("Reachable-supply cache label payload dtype differs")
    if stored_digest != label_payload_sha256(labels):
        raise ValueError("Reachable-supply cache SCC label hash differs")
    if expected_node_count is not None and node_count != int(expected_node_count):
        raise ValueError("Reachable-supply cache node count differs")
    if expected_row_count is not None and result.row_count != int(expected_row_count):
        raise ValueError("Reachable-supply cache OD row count differs")
    for name, observed, expected in (
        ("graph identity", result.graph_identity, expected_graph_identity),
        ("OD identity", result.od_identity, expected_od_identity),
        ("code hash", result.code_sha256, expected_code_sha256),
        ("config hash", result.config_sha256, expected_config_sha256),
    ):
        if expected is not None and observed != str(expected):
            raise ValueError(f"Reachable-supply cache {name} differs")
    if (
        result.origin_node_flow.shape != (node_count,)
        or result.destination_node_flow.shape != (node_count,)
        or np.any(~np.isfinite(result.origin_node_flow))
        or np.any(~np.isfinite(result.destination_node_flow))
        or np.any(result.origin_node_flow < 0.0)
        or np.any(result.destination_node_flow < 0.0)
        or not np.isfinite(result.reachable_demand)
        or not np.isfinite(result.unreachable_demand)
        or result.reachable_demand < 0.0
        or result.unreachable_demand < 0.0
        or result.scc_count <= 0
        or result.scc_count > node_count
        or result.row_count <= 0
        or result.unreachable_row_count < 0
        or result.unreachable_row_count > result.row_count
        or result.cross_scc_candidate_row_count != result.unreachable_row_count
        or result.cross_scc_reachable_row_count != 0
        or result.cross_scc_reachable_demand != 0.0
        or not np.isclose(
            result.cross_scc_candidate_demand,
            result.unreachable_demand,
            rtol=1.0e-12,
            atol=1.0e-7,
        )
    ):
        raise ValueError("Reachable-supply cache has invalid shapes or values")
    unique_labels = np.unique(labels)
    if (
        np.any(labels < 0)
        or np.any(labels >= node_count)
        or unique_labels.size != result.scc_count
        or not np.array_equal(
            unique_labels, np.arange(result.scc_count, dtype=labels.dtype)
        )
    ):
        raise ValueError("Reachable-supply cache labels are not canonical")
    component_minima = np.full(result.scc_count, node_count, dtype=np.int64)
    np.minimum.at(component_minima, labels, np.arange(node_count, dtype=np.int64))
    if np.any(component_minima[1:] <= component_minima[:-1]):
        raise ValueError("Reachable-supply cache component order is not canonical")
    if not np.isclose(
        result.origin_node_flow.sum(dtype=np.float64),
        result.reachable_demand,
        rtol=1.0e-12,
        atol=1.0e-7,
    ) or not np.isclose(
        result.destination_node_flow.sum(dtype=np.float64),
        result.reachable_demand,
        rtol=1.0e-12,
        atol=1.0e-7,
    ):
        raise ValueError("Reachable-supply cache node flows do not conserve demand")
    return result


def reachable_od_supply_metadata(supply: ReachableODSupply) -> dict[str, Any]:
    """Return the machine-readable companion metadata for a v2 cache."""

    digest = supply.label_payload_sha256 or label_payload_sha256(supply.scc_labels)
    value: dict[str, Any] = {
        "schema_id": REACHABLE_OD_CACHE_SCHEMA_ID,
        "algorithm_id": supply.algorithm_id,
        "label_payload_dtype": LABEL_PAYLOAD_DTYPE,
        "label_payload_sha256": digest,
        "graph_identity": str(supply.graph_identity),
        "od_identity": str(supply.od_identity),
        "code_sha256": str(supply.code_sha256),
        "config_sha256": str(supply.config_sha256),
        "node_count": int(len(supply.scc_labels)),
        "scc_count": int(supply.scc_count),
        "row_count": int(supply.row_count),
        "reachable_row_count": int(supply.row_count - supply.unreachable_row_count),
        "unreachable_row_count": int(supply.unreachable_row_count),
        "reachable_demand": float(supply.reachable_demand),
        "unreachable_demand": float(supply.unreachable_demand),
        "cross_scc_candidate_row_count": int(
            supply.cross_scc_candidate_row_count
        ),
        "cross_scc_candidate_demand": float(supply.cross_scc_candidate_demand),
        "cross_scc_reachable_row_count": int(
            supply.cross_scc_reachable_row_count
        ),
        "cross_scc_reachable_demand": float(supply.cross_scc_reachable_demand),
        "timing_seconds": {
            key: float(value) for key, value in (supply.timing_seconds or {}).items()
        },
    }
    if supply.scc_benchmark is not None:
        value["scc_benchmark"] = supply.scc_benchmark.to_dict()
    if supply.cross_scc_audit is not None:
        value["cross_scc_exact_audit"] = supply.cross_scc_audit.to_dict()
    return value
