"""BPR link-cost model and Beckmann line-search helpers for Prompt 4B."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _partC_rev1_as_float_array(value: np.ndarray | float, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def partC_rev1_validate_bpr_inputs(
    flow: np.ndarray,
    free_time: np.ndarray,
    capacity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flow_a = _partC_rev1_as_float_array(flow, "flow")
    free_a = _partC_rev1_as_float_array(free_time, "free_time")
    cap_a = _partC_rev1_as_float_array(capacity, "capacity")
    alpha_a = _partC_rev1_as_float_array(alpha, "alpha")
    beta_a = _partC_rev1_as_float_array(beta, "beta")
    if not (flow_a.shape == free_a.shape == cap_a.shape == alpha_a.shape == beta_a.shape):
        raise ValueError("flow, free_time, capacity, alpha, and beta must have identical shapes")
    if np.any(flow_a < -1e-9):
        raise ValueError("flow must be nonnegative")
    if np.any(free_a <= 0.0):
        raise ValueError("free_time must be positive")
    if np.any(cap_a <= 0.0):
        raise ValueError("capacity must be positive")
    if np.any(alpha_a < 0.0):
        raise ValueError("alpha must be nonnegative")
    if np.any(beta_a <= 0.0):
        raise ValueError("beta must be positive")
    return flow_a, free_a, cap_a, alpha_a, beta_a


def partC_rev1_bpr_cost(
    flow: np.ndarray,
    free_time: np.ndarray,
    capacity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    flow_a, free_a, cap_a, alpha_a, beta_a = partC_rev1_validate_bpr_inputs(flow, free_time, capacity, alpha, beta)
    nonnegative = np.maximum(flow_a, 0.0)
    return free_a * (1.0 + alpha_a * np.power(nonnegative / cap_a, beta_a))


def partC_rev1_bpr_derivative(
    flow: np.ndarray,
    free_time: np.ndarray,
    capacity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    flow_a, free_a, cap_a, alpha_a, beta_a = partC_rev1_validate_bpr_inputs(flow, free_time, capacity, alpha, beta)
    out = np.zeros_like(flow_a, dtype=np.float64)
    positive = flow_a > 0.0
    out[positive] = (
        free_a[positive]
        * alpha_a[positive]
        * beta_a[positive]
        * np.power(flow_a[positive], beta_a[positive] - 1.0)
        / np.power(cap_a[positive], beta_a[positive])
    )
    beta_one_zero = (~positive) & np.isclose(beta_a, 1.0) & (alpha_a > 0.0)
    out[beta_one_zero] = free_a[beta_one_zero] * alpha_a[beta_one_zero] / cap_a[beta_one_zero]
    beta_lt_one_zero = (~positive) & (beta_a < 1.0) & (alpha_a > 0.0)
    out[beta_lt_one_zero] = np.inf
    return out


def partC_rev1_bpr_integral(
    flow: np.ndarray,
    free_time: np.ndarray,
    capacity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    flow_a, free_a, cap_a, alpha_a, beta_a = partC_rev1_validate_bpr_inputs(flow, free_time, capacity, alpha, beta)
    nonnegative = np.maximum(flow_a, 0.0)
    congested = alpha_a * np.power(nonnegative, beta_a + 1.0) / ((beta_a + 1.0) * np.power(cap_a, beta_a))
    return free_a * (nonnegative + congested)


def partC_rev1_bpr_objective(
    flow: np.ndarray,
    free_time: np.ndarray,
    capacity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> float:
    return float(np.sum(partC_rev1_bpr_integral(flow, free_time, capacity, alpha, beta), dtype=np.float64))


def partC_rev1_line_derivative(
    eta: float,
    flow: np.ndarray,
    direction: np.ndarray,
    free_time: np.ndarray,
    capacity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> float:
    trial = np.asarray(flow, dtype=np.float64) + float(eta) * np.asarray(direction, dtype=np.float64)
    if np.any(trial < -1e-8):
        raise ValueError("line-search trial flow became negative")
    cost = partC_rev1_bpr_cost(np.maximum(trial, 0.0), free_time, capacity, alpha, beta)
    return float(np.dot(np.asarray(direction, dtype=np.float64), cost))


@dataclass(frozen=True, slots=True)
class PartCRev1LineSearchResult:
    eta: float
    derivative_at_zero: float
    derivative_at_one: float
    iterations: int
    stop_reason: str


def partC_rev1_fw_line_search(
    flow: np.ndarray,
    direction: np.ndarray,
    free_time: np.ndarray,
    capacity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    *,
    derivative_tol: float = 1e-8,
    interval_tol: float = 1e-10,
    max_iterations: int = 80,
) -> PartCRev1LineSearchResult:
    direction_a = np.asarray(direction, dtype=np.float64)
    if np.allclose(direction_a, 0.0):
        return PartCRev1LineSearchResult(0.0, 0.0, 0.0, 0, "zero_direction")
    g0 = partC_rev1_line_derivative(0.0, flow, direction_a, free_time, capacity, alpha, beta)
    g1 = partC_rev1_line_derivative(1.0, flow, direction_a, free_time, capacity, alpha, beta)
    if g0 >= 0.0:
        return PartCRev1LineSearchResult(0.0, g0, g1, 0, "boundary_zero")
    if g1 <= 0.0:
        return PartCRev1LineSearchResult(1.0, g0, g1, 0, "boundary_one")
    lo = 0.0
    hi = 1.0
    mid = 0.5
    for iteration in range(1, max_iterations + 1):
        mid = 0.5 * (lo + hi)
        gm = partC_rev1_line_derivative(mid, flow, direction_a, free_time, capacity, alpha, beta)
        if abs(gm) <= derivative_tol or (hi - lo) <= interval_tol:
            return PartCRev1LineSearchResult(float(mid), g0, g1, iteration, "bisect_converged")
        if gm < 0.0:
            lo = mid
        else:
            hi = mid
    return PartCRev1LineSearchResult(float(mid), g0, g1, max_iterations, "bisect_max_iterations")


def partC_rev1_relative_gap(tstt: float, sptt: float, *, epsilon: float = 1e-12) -> float:
    return float(max(0.0, (float(tstt) - float(sptt)) / max(float(sptt), epsilon)))
