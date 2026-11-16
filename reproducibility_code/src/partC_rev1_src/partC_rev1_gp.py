"""Optional path-based gradient-projection primitives for Prompt 4B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class PartCRev1GPShiftResult:
    shifted_flow: float
    updated_path_flows: np.ndarray
    updated_link_flow: np.ndarray
    pruned_active_indices: list[int]


def partC_rev1_gradient_projection_shift(
    *,
    path_flows: np.ndarray,
    path_costs: np.ndarray,
    path_arc_incidence: list[Iterable[int]],
    shortest_path_arcs: Iterable[int],
    link_flow: np.ndarray,
    link_derivative: np.ndarray,
    active_path_index: int,
    zeta: float = 1e-9,
    prune_tol: float = 1e-10,
) -> PartCRev1GPShiftResult:
    """Shift flow from one active path to the current shortest path.

    This is deliberately a primitive rather than a default national solver. It
    exists so GP can be enabled explicitly by a future `store_paths=true` mode.
    """
    flows = np.asarray(path_flows, dtype=np.float64).copy()
    costs = np.asarray(path_costs, dtype=np.float64)
    link = np.asarray(link_flow, dtype=np.float64).copy()
    deriv = np.asarray(link_derivative, dtype=np.float64)
    if active_path_index < 0 or active_path_index >= len(flows):
        raise IndexError("active_path_index is out of range")
    if flows[active_path_index] <= 0.0:
        return PartCRev1GPShiftResult(0.0, flows, link, [idx for idx, value in enumerate(flows) if value > prune_tol])
    active_arcs = set(int(value) for value in path_arc_incidence[active_path_index])
    shortest_arcs = set(int(value) for value in shortest_path_arcs)
    d_plus = shortest_arcs.difference(active_arcs)
    d_minus = active_arcs.difference(shortest_arcs)
    gamma = 0.0
    for arc in d_plus:
        gamma += float(deriv[arc])
    for arc in d_minus:
        gamma += float(deriv[arc])
    shortest_cost = float(np.sum([0.0])) if len(costs) == 0 else float(np.min(costs))
    active_cost = float(costs[active_path_index])
    omega = min(float(flows[active_path_index]), max(0.0, (active_cost - shortest_cost) / max(gamma, zeta)))
    if omega > 0.0:
        for arc in d_plus:
            link[arc] += omega
        for arc in d_minus:
            link[arc] -= omega
        flows[active_path_index] = max(0.0, flows[active_path_index] - omega)
    active_indices = [idx for idx, value in enumerate(flows) if value > prune_tol]
    return PartCRev1GPShiftResult(
        shifted_flow=float(omega),
        updated_path_flows=flows,
        updated_link_flow=link,
        pruned_active_indices=active_indices,
    )
