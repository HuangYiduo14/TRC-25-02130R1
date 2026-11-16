"""Phase 4 capacity-scenario and vectorized BPR helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from partC_rev1_src.partC_rev1_bpr import (
    partC_rev1_bpr_cost,
    partC_rev1_bpr_derivative,
    partC_rev1_bpr_objective,
)
from partC_rev1_src.partC_rev1_graph_snapshot import PartCRev1GraphSnapshot


@dataclass(frozen=True, slots=True)
class RefreshedStateMetrics:
    objective: float
    tstt: float
    sptt: float
    absolute_gap: float
    relative_gap_sptt: float
    relative_gap_tstt: float
    negative_gap_tolerance_applied: bool


def phase4_scenario_capacity(
    graph: PartCRev1GraphSnapshot,
    capacity_multiplier: float,
) -> np.ndarray:
    """Apply the Phase 4 scenario multiplier exactly once to prepared capacity."""

    multiplier = float(capacity_multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("Phase 4 capacity multiplier must be finite and positive")
    prepared = np.asarray(graph.capacity, dtype=np.float64)
    if prepared.shape != graph.free_time.shape or np.any(~np.isfinite(prepared)) or np.any(prepared <= 0.0):
        raise ValueError("Prepared graph capacity is invalid")
    used = np.ascontiguousarray(prepared * multiplier, dtype=np.float64)
    if np.any(~np.isfinite(used)) or np.any(used <= 0.0):
        raise ValueError("Scenario capacity is invalid")
    return used


def phase4_bpr_cost(
    graph: PartCRev1GraphSnapshot,
    flow: np.ndarray,
    used_capacity: np.ndarray,
) -> np.ndarray:
    return partC_rev1_bpr_cost(
        np.asarray(flow, dtype=np.float64),
        graph.free_time,
        used_capacity,
        graph.alpha,
        graph.beta,
    )


def phase4_bpr_derivative(
    graph: PartCRev1GraphSnapshot,
    flow: np.ndarray,
    used_capacity: np.ndarray,
) -> np.ndarray:
    return partC_rev1_bpr_derivative(
        np.asarray(flow, dtype=np.float64),
        graph.free_time,
        used_capacity,
        graph.alpha,
        graph.beta,
    )


def phase4_bpr_objective(
    graph: PartCRev1GraphSnapshot,
    flow: np.ndarray,
    used_capacity: np.ndarray,
) -> float:
    return partC_rev1_bpr_objective(
        np.asarray(flow, dtype=np.float64),
        graph.free_time,
        used_capacity,
        graph.alpha,
        graph.beta,
    )


def refreshed_state_metrics(
    graph: PartCRev1GraphSnapshot,
    flow: np.ndarray,
    used_capacity: np.ndarray,
    *,
    sptt: float,
    epsilon: float = 1e-12,
    roundoff_tolerance: float = 1e-10,
) -> RefreshedStateMetrics:
    """Evaluate both required refreshed-gap denominators at one current state."""

    current = np.asarray(flow, dtype=np.float64)
    cost = phase4_bpr_cost(graph, current, used_capacity)
    tstt = float(np.dot(current, cost))
    shortest = float(sptt)
    raw_gap = tstt - shortest
    scale = max(abs(tstt), abs(shortest), 1.0)
    adjusted = False
    if raw_gap < 0.0 and abs(raw_gap) <= float(roundoff_tolerance) * scale:
        raw_gap = 0.0
        adjusted = True
    if raw_gap < 0.0:
        raise ValueError(
            f"Refreshed shortest-path gap is materially negative: TSTT={tstt}, SPTT={shortest}"
        )
    return RefreshedStateMetrics(
        objective=phase4_bpr_objective(graph, current, used_capacity),
        tstt=tstt,
        sptt=shortest,
        absolute_gap=float(raw_gap),
        relative_gap_sptt=float(raw_gap / max(abs(shortest), float(epsilon))),
        relative_gap_tstt=float(raw_gap / max(abs(tstt), float(epsilon))),
        negative_gap_tolerance_applied=adjusted,
    )
