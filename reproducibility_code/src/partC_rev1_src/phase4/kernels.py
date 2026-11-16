"""Compiled exact shortest-path-tree loading and simultaneous tracing kernels."""

from __future__ import annotations

import numpy as np
from numba import njit


UINT64_SENTINEL = np.uint64(0xFFFFFFFFFFFFFFFF)
FINE_NODE_BITS = 19
FINE_REGION_SHIFT = 2 * FINE_NODE_BITS


@njit(cache=True, nogil=True, inline="always")
def _heap_less(key_a: float, node_a: int, key_b: float, node_b: int) -> bool:
    return key_a < key_b or (key_a == key_b and node_a < node_b)


@njit(cache=True, nogil=True, inline="always")
def _heap_swap(
    heap_nodes: np.ndarray,
    heap_keys: np.ndarray,
    heap_position: np.ndarray,
    left: int,
    right: int,
) -> None:
    left_node = int(heap_nodes[left])
    right_node = int(heap_nodes[right])
    left_key = float(heap_keys[left])
    heap_nodes[left] = right_node
    heap_keys[left] = heap_keys[right]
    heap_position[right_node] = left
    heap_nodes[right] = left_node
    heap_keys[right] = left_key
    heap_position[left_node] = right


@njit(cache=True, nogil=True, inline="always")
def _heap_sift_up(
    heap_nodes: np.ndarray,
    heap_keys: np.ndarray,
    heap_position: np.ndarray,
    position: int,
) -> None:
    current = int(position)
    while current > 0:
        parent = (current - 1) // 2
        if not _heap_less(
            float(heap_keys[current]),
            int(heap_nodes[current]),
            float(heap_keys[parent]),
            int(heap_nodes[parent]),
        ):
            break
        _heap_swap(heap_nodes, heap_keys, heap_position, current, parent)
        current = parent


@njit(cache=True, nogil=True, inline="always")
def _heap_sift_down(
    heap_nodes: np.ndarray,
    heap_keys: np.ndarray,
    heap_position: np.ndarray,
    size: int,
    position: int,
) -> None:
    current = int(position)
    while True:
        left = 2 * current + 1
        if left >= size:
            break
        right = left + 1
        smallest = left
        if right < size and _heap_less(
            float(heap_keys[right]),
            int(heap_nodes[right]),
            float(heap_keys[left]),
            int(heap_nodes[left]),
        ):
            smallest = right
        if not _heap_less(
            float(heap_keys[smallest]),
            int(heap_nodes[smallest]),
            float(heap_keys[current]),
            int(heap_nodes[current]),
        ):
            break
        _heap_swap(heap_nodes, heap_keys, heap_position, current, smallest)
        current = smallest


@njit(cache=True, nogil=True, inline="always")
def _heap_push_or_decrease(
    heap_nodes: np.ndarray,
    heap_keys: np.ndarray,
    heap_position: np.ndarray,
    size: int,
    node: int,
    key: float,
) -> int:
    position = int(heap_position[node])
    if position >= 0:
        if key < float(heap_keys[position]):
            heap_keys[position] = key
            _heap_sift_up(heap_nodes, heap_keys, heap_position, position)
        return size
    heap_nodes[size] = node
    heap_keys[size] = key
    heap_position[node] = size
    _heap_sift_up(heap_nodes, heap_keys, heap_position, size)
    return size + 1


@njit(cache=True, nogil=True, inline="always")
def _heap_pop(
    heap_nodes: np.ndarray,
    heap_keys: np.ndarray,
    heap_position: np.ndarray,
    size: int,
) -> tuple[int, float, int]:
    node = int(heap_nodes[0])
    key = float(heap_keys[0])
    new_size = size - 1
    heap_position[node] = -1
    if new_size > 0:
        replacement = int(heap_nodes[new_size])
        heap_nodes[0] = replacement
        heap_keys[0] = heap_keys[new_size]
        heap_position[replacement] = 0
        _heap_sift_down(heap_nodes, heap_keys, heap_position, new_size, 0)
    return node, key, new_size


@njit(cache=True, nogil=True, inline="always")
def _terminal_lookup(
    keys: np.ndarray,
    values: np.ndarray,
    begin: int,
    end: int,
    key: np.uint64,
    mode: int,
) -> int:
    lo = int(begin)
    hi = int(end)
    while lo < hi:
        mid = (lo + hi) // 2
        value = keys[mid]
        if value < key:
            lo = mid + 1
        else:
            hi = mid
    if lo < end and keys[lo] == key:
        return int(values[mode, lo])
    return -1


@njit(cache=True, nogil=True, inline="always")
def _mix_uint64(key: np.uint64) -> np.uint64:
    value = key
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return value


@njit(cache=True, nogil=True, inline="always")
def _hash_add(keys: np.ndarray, values: np.ndarray, key: np.uint64, amount: float) -> bool:
    capacity = len(keys)
    mask = np.uint64(capacity - 1)
    # Numba may infer ``int(uint64)`` as a floating scalar on Windows when the
    # source can span the full uint64 domain.  The mask proves this value is a
    # valid signed array index, so make that narrowing explicit.
    position = np.int64(_mix_uint64(key) & mask)
    for _ in range(capacity):
        found = keys[position]
        if found == UINT64_SENTINEL:
            keys[position] = key
            values[position] = amount
            return True
        if found == key:
            values[position] += amount
            return True
        position = np.int64((position + 1) & (capacity - 1))
    return False


@njit(cache=True, nogil=True, inline="always")
def _record_regional_pair(
    suite: int,
    region: int,
    entry_node: int,
    exit_node: int,
    entry_terminal: int,
    exit_terminal: int,
    amount: float,
    coarse_keys: np.ndarray,
    coarse_values: np.ndarray,
    fine_keys: np.ndarray,
    fine_values: np.ndarray,
    failure_counts: np.ndarray,
    overflow_counts: np.ndarray,
) -> None:
    if amount <= 0.0:
        return
    if entry_terminal < 0 or exit_terminal < 0 or entry_node < 0 or exit_node < 0:
        failure_counts[suite] += 1
        return
    coarse_key = (np.uint64(entry_terminal) << np.uint64(32)) | np.uint64(exit_terminal)
    fine_key = (
        (np.uint64(region) << np.uint64(FINE_REGION_SHIFT))
        | (np.uint64(entry_node) << np.uint64(FINE_NODE_BITS))
        | np.uint64(exit_node)
    )
    if not _hash_add(coarse_keys[suite], coarse_values[suite], coarse_key, amount):
        overflow_counts[suite, 0] += 1
    if not _hash_add(fine_keys[suite], fine_values[suite], fine_key, amount):
        overflow_counts[suite, 1] += 1


@njit(cache=True, nogil=True)
def aon_trace_range_kernel(
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
    original_row_multiplicity: np.ndarray,
    group_start: int,
    group_end: int,
    tie_tolerance: float,
    arc_region_local: np.ndarray,
    arc_internal_bacn_local: np.ndarray,
    terminal_keys: np.ndarray,
    terminal_values: np.ndarray,
    terminal_offsets: np.ndarray,
    region_counts: np.ndarray,
    bacn_counts: np.ndarray,
    coarse_hash_capacity: int,
    fine_hash_capacity: int,
) -> tuple:
    """Load one exact origin range and trace every suite from each shared tree."""

    suite_count = arc_region_local.shape[0]
    physical_flow = np.zeros(n_arcs, dtype=np.float64)
    coarse_keys = np.full((suite_count, coarse_hash_capacity), UINT64_SENTINEL, dtype=np.uint64)
    coarse_values = np.zeros((suite_count, coarse_hash_capacity), dtype=np.float64)
    fine_keys = np.full((suite_count, fine_hash_capacity), UINT64_SENTINEL, dtype=np.uint64)
    fine_values = np.zeros((suite_count, fine_hash_capacity), dtype=np.float64)
    max_regions = 1
    max_bacns = 1
    for suite in range(suite_count):
        if int(region_counts[suite]) > max_regions:
            max_regions = int(region_counts[suite])
        if int(bacn_counts[suite]) > max_bacns:
            max_bacns = int(bacn_counts[suite])
    repeated_flow = np.zeros((suite_count, max_regions), dtype=np.float64)
    repeated_unique = np.zeros((suite_count, max_regions), dtype=np.int64)
    repeated_rows = np.zeros((suite_count, max_regions), dtype=np.int64)
    bacn_traversal_flow = np.zeros((suite_count, max_bacns), dtype=np.float64)
    failure_counts = np.zeros(suite_count, dtype=np.int64)
    overflow_counts = np.zeros((suite_count, 2), dtype=np.int64)

    distance = np.empty(n_nodes, dtype=np.float64)
    seen_epoch = np.zeros(n_nodes, dtype=np.int32)
    settled_epoch = np.zeros(n_nodes, dtype=np.int32)
    predecessor_node = np.empty(n_nodes, dtype=np.int64)
    predecessor_arc = np.empty(n_nodes, dtype=np.int64)
    settled_order = np.empty(n_nodes, dtype=np.int64)
    heap_nodes = np.empty(n_nodes, dtype=np.int64)
    heap_keys = np.empty(n_nodes, dtype=np.float64)
    heap_position = np.full(n_nodes, -1, dtype=np.int64)
    direct_demand = np.zeros(n_nodes, dtype=np.float64)
    direct_unique = np.zeros(n_nodes, dtype=np.int64)
    direct_rows = np.zeros(n_nodes, dtype=np.int64)
    subtree_demand = np.zeros(n_nodes, dtype=np.float64)
    subtree_unique = np.zeros(n_nodes, dtype=np.int64)
    subtree_rows = np.zeros(n_nodes, dtype=np.int64)

    active_region = np.empty(n_nodes, dtype=np.int32)
    entry_node = np.empty(n_nodes, dtype=np.int64)
    entry_terminal = np.empty(n_nodes, dtype=np.int32)
    occurrence_history = np.empty(n_nodes, dtype=np.int64)
    occurrence_region = np.empty(n_nodes, dtype=np.int32)
    occurrence_previous = np.empty(n_nodes, dtype=np.int64)
    active_bacn = np.empty(n_nodes, dtype=np.int32)

    sptt = 0.0
    reachable_demand = 0.0
    unreachable_demand = 0.0
    settled_total = 0
    relaxed_total = 0
    positive_tree_nodes_total = 0
    epoch = 0

    for group in range(group_start, group_end):
        epoch += 1
        source = int(origin_group_node_idx[group])
        od_begin = int(origin_offsets[group])
        od_end = int(origin_offsets[group + 1])
        destination_count = od_end - od_begin
        for od_position in range(od_begin, od_end):
            destination = int(destination_node_idx[od_position])
            direct_demand[destination] = float(demand[od_position])
            direct_unique[destination] = 1
            direct_rows[destination] = int(original_row_multiplicity[od_position])
            subtree_demand[destination] = float(demand[od_position])
            subtree_unique[destination] = 1
            subtree_rows[destination] = int(original_row_multiplicity[od_position])

        heap_size = 0
        settled_count = 0
        remaining_destinations = destination_count
        distance[source] = 0.0
        seen_epoch[source] = epoch
        predecessor_node[source] = -1
        predecessor_arc[source] = -1
        heap_size = _heap_push_or_decrease(
            heap_nodes, heap_keys, heap_position, heap_size, source, 0.0
        )
        while heap_size > 0 and remaining_destinations > 0:
            node, node_distance, heap_size = _heap_pop(
                heap_nodes, heap_keys, heap_position, heap_size
            )
            if settled_epoch[node] == epoch:
                continue
            if seen_epoch[node] != epoch or node_distance != distance[node]:
                continue
            settled_epoch[node] = epoch
            settled_order[settled_count] = node
            settled_count += 1
            if direct_unique[node] > 0:
                remaining_destinations -= 1
            for position in range(int(indptr[node]), int(indptr[node + 1])):
                arc = int(csr_arc_idx[position])
                child = int(head[arc])
                if settled_epoch[child] == epoch:
                    continue
                relaxed_total += 1
                candidate = node_distance + float(cost[arc])
                if seen_epoch[child] != epoch or candidate < distance[child] - tie_tolerance:
                    distance[child] = candidate
                    seen_epoch[child] = epoch
                    predecessor_node[child] = node
                    predecessor_arc[child] = arc
                    heap_size = _heap_push_or_decrease(
                        heap_nodes,
                        heap_keys,
                        heap_position,
                        heap_size,
                        child,
                        candidate,
                    )
                elif abs(candidate - distance[child]) <= tie_tolerance:
                    old_arc = int(predecessor_arc[child])
                    if old_arc < 0 or int(arc_tie_rank[arc]) < int(arc_tie_rank[old_arc]):
                        predecessor_node[child] = node
                        predecessor_arc[child] = arc

        for position in range(heap_size):
            heap_position[int(heap_nodes[position])] = -1
        settled_total += settled_count

        for od_position in range(od_begin, od_end):
            destination = int(destination_node_idx[od_position])
            amount = float(demand[od_position])
            if settled_epoch[destination] == epoch:
                reachable_demand += amount
                sptt += amount * float(distance[destination])
            else:
                unreachable_demand += amount

        for position in range(settled_count - 1, -1, -1):
            node = int(settled_order[position])
            amount = float(subtree_demand[node])
            if amount <= 0.0:
                continue
            positive_tree_nodes_total += 1
            if node != source:
                arc = int(predecessor_arc[node])
                parent = int(predecessor_node[node])
                if arc < 0 or parent < 0:
                    continue
                physical_flow[arc] += amount
                subtree_demand[parent] += amount
                subtree_unique[parent] += subtree_unique[node]
                subtree_rows[parent] += subtree_rows[node]

        for suite in range(suite_count):
            terminal_begin = int(terminal_offsets[suite])
            terminal_end = int(terminal_offsets[suite + 1])
            active_region[source] = -1
            entry_node[source] = -1
            entry_terminal[source] = -1
            occurrence_history[source] = -1
            occurrence_region[source] = -1
            occurrence_previous[source] = -1
            active_bacn[source] = -1
            for position in range(1, settled_count):
                node = int(settled_order[position])
                amount = float(subtree_demand[node])
                if amount <= 0.0:
                    continue
                parent = int(predecessor_node[node])
                arc = int(predecessor_arc[node])
                previous_region = int(active_region[parent])
                current_region = int(arc_region_local[suite, arc])
                history = int(occurrence_history[parent])

                if previous_region >= 0 and current_region != previous_region:
                    exit_key = np.uint64(previous_region) * np.uint64(n_nodes) + np.uint64(parent)
                    exit_terminal = _terminal_lookup(
                        terminal_keys,
                        terminal_values,
                        terminal_begin,
                        terminal_end,
                        exit_key,
                        1,
                    )
                    _record_regional_pair(
                        suite,
                        previous_region,
                        int(entry_node[parent]),
                        parent,
                        int(entry_terminal[parent]),
                        exit_terminal,
                        amount,
                        coarse_keys,
                        coarse_values,
                        fine_keys,
                        fine_values,
                        failure_counts,
                        overflow_counts,
                    )

                if current_region >= 0:
                    if current_region == previous_region:
                        active_region[node] = previous_region
                        entry_node[node] = entry_node[parent]
                        entry_terminal[node] = entry_terminal[parent]
                        occurrence_history[node] = history
                        occurrence_region[node] = -1
                        occurrence_previous[node] = -1
                    else:
                        active_region[node] = current_region
                        entry_node[node] = parent
                        entry_key = np.uint64(current_region) * np.uint64(n_nodes) + np.uint64(parent)
                        terminal = _terminal_lookup(
                            terminal_keys,
                            terminal_values,
                            terminal_begin,
                            terminal_end,
                            entry_key,
                            2 if parent == source else 0,
                        )
                        entry_terminal[node] = terminal
                        repeated = False
                        pointer = history
                        while pointer >= 0:
                            if int(occurrence_region[pointer]) == current_region:
                                repeated = True
                                break
                            pointer = int(occurrence_previous[pointer])
                        if repeated:
                            repeated_flow[suite, current_region] += amount
                            repeated_unique[suite, current_region] += subtree_unique[node]
                            repeated_rows[suite, current_region] += subtree_rows[node]
                        occurrence_region[node] = current_region
                        occurrence_previous[node] = history
                        occurrence_history[node] = node
                else:
                    active_region[node] = -1
                    entry_node[node] = -1
                    entry_terminal[node] = -1
                    occurrence_history[node] = history
                    occurrence_region[node] = -1
                    occurrence_previous[node] = -1

                previous_bacn = int(active_bacn[parent])
                current_bacn = int(arc_internal_bacn_local[suite, arc])
                if previous_bacn >= 0 and current_bacn != previous_bacn:
                    bacn_traversal_flow[suite, previous_bacn] += amount
                active_bacn[node] = current_bacn if current_bacn >= 0 else -1

                direct_amount = float(direct_demand[node])
                if direct_amount > 0.0:
                    final_region = int(active_region[node])
                    if final_region >= 0:
                        exit_key = np.uint64(final_region) * np.uint64(n_nodes) + np.uint64(node)
                        exit_terminal = _terminal_lookup(
                            terminal_keys,
                            terminal_values,
                            terminal_begin,
                            terminal_end,
                            exit_key,
                            3,
                        )
                        _record_regional_pair(
                            suite,
                            final_region,
                            int(entry_node[node]),
                            node,
                            int(entry_terminal[node]),
                            exit_terminal,
                            direct_amount,
                            coarse_keys,
                            coarse_values,
                            fine_keys,
                            fine_values,
                            failure_counts,
                            overflow_counts,
                        )
                    final_bacn = int(active_bacn[node])
                    if final_bacn >= 0:
                        bacn_traversal_flow[suite, final_bacn] += direct_amount

        for position in range(settled_count):
            node = int(settled_order[position])
            subtree_demand[node] = 0.0
            subtree_unique[node] = 0
            subtree_rows[node] = 0
        for od_position in range(od_begin, od_end):
            destination = int(destination_node_idx[od_position])
            direct_demand[destination] = 0.0
            direct_unique[destination] = 0
            direct_rows[destination] = 0
            subtree_demand[destination] = 0.0
            subtree_unique[destination] = 0
            subtree_rows[destination] = 0

    return (
        physical_flow,
        sptt,
        reachable_demand,
        unreachable_demand,
        coarse_keys,
        coarse_values,
        fine_keys,
        fine_values,
        repeated_flow,
        repeated_unique,
        repeated_rows,
        bacn_traversal_flow,
        failure_counts,
        overflow_counts,
        settled_total,
        relaxed_total,
        positive_tree_nodes_total,
    )
