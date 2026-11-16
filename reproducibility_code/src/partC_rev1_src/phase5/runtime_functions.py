"""Root-free convex runtime functions and deterministic Phase 5 pilot fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from scipy.optimize import nnls


class RuntimeFunction(Protocol):
    def cost(self, flow: np.ndarray | float) -> np.ndarray: ...
    def potential(self, flow: np.ndarray | float) -> np.ndarray: ...
    def derivative(self, flow: np.ndarray | float) -> np.ndarray: ...


def _flow_array(flow: np.ndarray | float) -> np.ndarray:
    value = np.asarray(flow, dtype=np.float64)
    if np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("Runtime flow must be finite and nonnegative")
    return value


@dataclass(frozen=True, slots=True)
class PowerSumRuntime:
    """Explicit ``intercept + sum coefficient*q**exponent`` marginal cost."""

    intercept_s: float
    exponents: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        intercept = float(self.intercept_s)
        exponents = np.ascontiguousarray(self.exponents, dtype=np.float64)
        coefficients = np.ascontiguousarray(self.coefficients, dtype=np.float64)
        if not np.isfinite(intercept) or intercept < 0.0:
            raise ValueError("Runtime intercept must be finite and nonnegative")
        if exponents.ndim != 1 or coefficients.shape != exponents.shape:
            raise ValueError("Power arrays must be aligned one-dimensional arrays")
        if np.any(~np.isfinite(exponents)) or np.any(exponents <= 0.0):
            raise ValueError("Power exponents must be finite and positive")
        if np.any(~np.isfinite(coefficients)) or np.any(coefficients < 0.0):
            raise ValueError("Power coefficients must be finite and nonnegative")
        exponents.setflags(write=False)
        coefficients.setflags(write=False)
        object.__setattr__(self, "intercept_s", intercept)
        object.__setattr__(self, "exponents", exponents)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def arithmetic_term_count(self) -> int:
        return int(len(self.exponents))

    def cost(self, flow: np.ndarray | float) -> np.ndarray:
        q = _flow_array(flow)
        result = np.full(q.shape, self.intercept_s, dtype=np.float64)
        for exponent, coefficient in zip(self.exponents, self.coefficients, strict=True):
            if coefficient:
                result += coefficient * np.power(q, exponent)
        return result

    def potential(self, flow: np.ndarray | float) -> np.ndarray:
        q = _flow_array(flow)
        result = self.intercept_s * q
        for exponent, coefficient in zip(self.exponents, self.coefficients, strict=True):
            if coefficient:
                result += coefficient * np.power(q, exponent + 1.0) / (exponent + 1.0)
        return result

    def derivative(self, flow: np.ndarray | float) -> np.ndarray:
        q = _flow_array(flow)
        result = np.zeros(q.shape, dtype=np.float64)
        for exponent, coefficient in zip(self.exponents, self.coefficients, strict=True):
            if not coefficient:
                continue
            positive = q > 0.0
            result[positive] += coefficient * exponent * np.power(q[positive], exponent - 1.0)
            if exponent == 1.0:
                result[~positive] += coefficient
            elif exponent < 1.0:
                result[~positive] = np.inf
        return result


@dataclass(frozen=True, slots=True)
class OneTermRuntime:
    intercept_s: float
    scale_flow: float
    exponent: float
    eta_s: float

    def __post_init__(self) -> None:
        values = tuple(map(float, (self.intercept_s, self.scale_flow, self.exponent, self.eta_s)))
        if any(not np.isfinite(value) for value in values):
            raise ValueError("One-term parameters must be finite")
        if values[0] < 0.0 or values[1] <= 0.0 or values[2] <= 0.0 or values[3] < 0.0:
            raise ValueError("One-term parameters are outside their convex domain")
        for name, value in zip(
            ("intercept_s", "scale_flow", "exponent", "eta_s"), values, strict=True
        ):
            object.__setattr__(self, name, value)

    @property
    def arithmetic_term_count(self) -> int:
        return 1

    def cost(self, flow: np.ndarray | float) -> np.ndarray:
        q = _flow_array(flow)
        return self.intercept_s + self.eta_s * np.power(q / self.scale_flow, self.exponent)

    def potential(self, flow: np.ndarray | float) -> np.ndarray:
        q = _flow_array(flow)
        return self.intercept_s * q + (
            self.eta_s
            * self.scale_flow
            * np.power(q / self.scale_flow, self.exponent + 1.0)
            / (self.exponent + 1.0)
        )

    def derivative(self, flow: np.ndarray | float) -> np.ndarray:
        q = _flow_array(flow)
        result = np.zeros(q.shape, dtype=np.float64)
        positive = q > 0.0
        result[positive] = (
            self.eta_s
            * self.exponent
            * np.power(q[positive] / self.scale_flow, self.exponent - 1.0)
            / self.scale_flow
        )
        if self.exponent == 1.0:
            result[~positive] = self.eta_s / self.scale_flow
        elif self.exponent < 1.0 and self.eta_s > 0.0:
            result[~positive] = np.inf
        return result


def _segment_length_basis(flow: np.ndarray, knots: np.ndarray) -> np.ndarray:
    starts = np.r_[0.0, knots]
    result = np.empty((len(flow), len(starts)), dtype=np.float64)
    for index, start in enumerate(starts):
        value = np.maximum(flow - start, 0.0)
        if index < len(knots):
            value = np.minimum(value, knots[index] - start)
        result[:, index] = value
    return result


def _segment_integral_basis(flow: np.ndarray, knots: np.ndarray) -> np.ndarray:
    starts = np.r_[0.0, knots]
    result = np.empty((len(flow), len(starts)), dtype=np.float64)
    for index, start in enumerate(starts):
        positive = np.maximum(flow - start, 0.0)
        if index < len(knots):
            width = knots[index] - start
            length = np.minimum(positive, width)
            result[:, index] = length * (positive - 0.5 * length)
        else:
            result[:, index] = 0.5 * np.square(positive)
    return result


@dataclass(frozen=True, slots=True)
class HingeRuntime:
    """Piecewise-linear cost parameterized by nonnegative actual slopes."""

    intercept_s: float
    knots_flow: np.ndarray
    segment_slopes_s_per_flow: np.ndarray

    def __post_init__(self) -> None:
        intercept = float(self.intercept_s)
        knots = np.ascontiguousarray(self.knots_flow, dtype=np.float64)
        slopes = np.ascontiguousarray(self.segment_slopes_s_per_flow, dtype=np.float64)
        if not np.isfinite(intercept) or intercept < 0.0:
            raise ValueError("Hinge intercept must be finite and nonnegative")
        if knots.ndim != 1 or slopes.shape != (len(knots) + 1,):
            raise ValueError("Hinge knots/slopes have incompatible dimensions")
        if np.any(~np.isfinite(knots)) or np.any(knots <= 0.0) or np.any(np.diff(knots) <= 0.0):
            raise ValueError("Hinge knots must be finite, positive, and strictly ordered")
        if np.any(~np.isfinite(slopes)) or np.any(slopes < 0.0):
            raise ValueError("Actual hinge segment slopes must be finite and nonnegative")
        knots.setflags(write=False)
        slopes.setflags(write=False)
        object.__setattr__(self, "intercept_s", intercept)
        object.__setattr__(self, "knots_flow", knots)
        object.__setattr__(self, "segment_slopes_s_per_flow", slopes)

    @property
    def arithmetic_term_count(self) -> int:
        return int(len(self.segment_slopes_s_per_flow))

    def cost(self, flow: np.ndarray | float) -> np.ndarray:
        q = _flow_array(flow)
        flat = q.reshape(-1)
        value = self.intercept_s + _segment_length_basis(flat, self.knots_flow) @ self.segment_slopes_s_per_flow
        return value.reshape(q.shape)

    def potential(self, flow: np.ndarray | float) -> np.ndarray:
        q = _flow_array(flow)
        flat = q.reshape(-1)
        value = self.intercept_s * flat + _segment_integral_basis(flat, self.knots_flow) @ self.segment_slopes_s_per_flow
        return value.reshape(q.shape)

    def derivative(self, flow: np.ndarray | float) -> np.ndarray:
        q = _flow_array(flow)
        indexes = np.searchsorted(self.knots_flow, q, side="right")
        return self.segment_slopes_s_per_flow[indexes]


def pilot_weights(flow: Sequence[float], low_flow: float, activation_mask: Sequence[bool]) -> np.ndarray:
    q = np.asarray(flow, dtype=np.float64)
    activation = np.asarray(activation_mask, dtype=bool)
    if q.ndim != 1 or activation.shape != q.shape:
        raise ValueError("Pilot weights require aligned one-dimensional arrays")
    weights = np.ones(len(q), dtype=np.float64)
    low = float(low_flow)
    if low > 0.0:
        weights[(q >= 0.5 * low) & (q <= 1.5 * low)] = 4.0
        weights[q == low] = 8.0
    weights[activation] = np.maximum(weights[activation], 2.0)
    return weights


def fit_one_term(
    flow: Sequence[float],
    exact_cost_s: Sequence[float],
    *,
    intercept_s: float,
    exponent: float,
    scale_flow: float,
    weights: Sequence[float],
) -> OneTermRuntime:
    q = np.asarray(flow, dtype=np.float64)
    target = np.asarray(exact_cost_s, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if q.shape != target.shape or q.shape != weight.shape or q.ndim != 1:
        raise ValueError("One-term fit arrays must be aligned and one-dimensional")
    keep = q > 0.0
    denominator = np.maximum.reduce(
        [target[keep], np.full(np.count_nonzero(keep), float(intercept_s)), np.ones(np.count_nonzero(keep))]
    )
    regressor = np.power(q[keep] / float(scale_flow), float(exponent))
    scaled_x = np.sqrt(weight[keep]) * regressor / denominator
    scaled_y = np.sqrt(weight[keep]) * (target[keep] - float(intercept_s)) / denominator
    norm = float(np.dot(scaled_x, scaled_x))
    eta = max(0.0, float(np.dot(scaled_x, scaled_y) / norm)) if norm > 0.0 else 0.0
    return OneTermRuntime(float(intercept_s), float(scale_flow), float(exponent), eta)


def transformed_knots(flow_domain_max: float, segment_count: int) -> np.ndarray:
    maximum = float(flow_domain_max)
    count = int(segment_count)
    if not np.isfinite(maximum) or maximum <= 0.0 or count < 2:
        raise ValueError("Hinge domain must be positive and segment count at least two")
    u = np.arange(1, count, dtype=np.float64) / float(count)
    return maximum * np.square(u)


def fit_hinge(
    flow: Sequence[float],
    exact_cost_s: Sequence[float],
    *,
    intercept_s: float,
    flow_domain_max: float,
    segment_count: int,
    weights: Sequence[float],
) -> HingeRuntime:
    q = np.asarray(flow, dtype=np.float64)
    target = np.asarray(exact_cost_s, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if q.shape != target.shape or q.shape != weight.shape or q.ndim != 1:
        raise ValueError("Hinge fit arrays must be aligned and one-dimensional")
    keep = q > 0.0
    knots = transformed_knots(flow_domain_max, segment_count)
    basis = _segment_length_basis(q[keep], knots)
    denominator = np.maximum.reduce(
        [target[keep], np.full(np.count_nonzero(keep), float(intercept_s)), np.ones(np.count_nonzero(keep))]
    )
    scale = np.sqrt(weight[keep]) / denominator
    slopes, _ = nnls(basis * scale[:, None], (target[keep] - float(intercept_s)) * scale)
    return HingeRuntime(float(intercept_s), knots, slopes)


def finite_difference_potential_error(
    function: RuntimeFunction, flow: Sequence[float]
) -> float:
    return float(potential_gradient_audit(function, flow)["max_corrected_absolute_error"])


def _hinge_interval_integral(function: HingeRuntime, lo, hi):
    """Independent trapezoid integration, explicitly split at every knot."""
    result = np.zeros_like(lo, dtype=float)
    starts = np.r_[0.0, function.knots_flow]
    ends = np.r_[function.knots_flow, np.inf]
    for start, end in zip(starts, ends, strict=True):
        left = np.maximum(lo, start)
        right = np.maximum(left, np.minimum(hi, end))
        result += (right-left) * 0.5 * (function.cost(left)+function.cost(right))
    return result


def potential_gradient_audit(function: RuntimeFunction, flow: Sequence[float]) -> dict:
    """Local-scale identity checks, preserving the old raw centered diagnostic.

    Smooth interiors use a step ladder; hinge interiors remain within their
    segment when representable. Crossing stencils subtract their exact
    piecewise-linear integral bias, including h*(s_right-s_left)/4 at knots.
    No cost or potential tolerance is widened.
    """
    q = np.asarray(flow, dtype=np.float64)
    if not len(q):
        raise ValueError("Potential validation needs a nonempty probe set")
    if isinstance(function, HingeRuntime):
        q = np.unique(np.r_[q, function.knots_flow[function.knots_flow <= q.max()], 0.0])
    costs = function.cost(q)
    scale = np.maximum(1.0, np.abs(costs))
    h0 = np.cbrt(np.finfo(float).eps)*np.maximum(q, 1.0)
    raw_lo, raw_hi = np.maximum(0.0, q-h0), q+h0
    raw = (function.potential(raw_hi)-function.potential(raw_lo))/(raw_hi-raw_lo)-costs
    best = np.full(len(q), np.inf); best_step = h0.copy(); best_bias = np.zeros(len(q)); best_roundoff = np.full(len(q), np.inf)
    for factor in (1.0, 8.0, 0.125):
        h = h0*factor
        if isinstance(function, HingeRuntime):
            # Stay inside a segment unless this is a knot or its ulp neighbor.
            if len(function.knots_flow):
                distance = np.min(np.abs(q[:, None]-function.knots_flow), axis=1)
                interior = distance > 128*np.finfo(float).eps*np.maximum(q, 1)
                h = np.where(interior, np.minimum(h, 0.25*distance), h)
            lo, hi = np.maximum(0.0, q-h), q+h
            pl, ph = function.potential(lo), function.potential(hi)
            width = hi-lo
            integral = _hinge_interval_integral(function, lo, hi)
            bias = integral/width-costs
            error = (ph-pl)/width-costs-bias
            roundoff = 8*np.finfo(float).eps*(np.abs(ph)+np.abs(pl))/width
        else:
            boundary = q < 2*h
            lo, hi = np.maximum(0.0, q-h), q+h
            pl, ph = function.potential(lo), function.potential(hi)
            centered = (ph-pl)/(hi-lo)
            f0, f1, f2 = function.potential(q), function.potential(q+h), function.potential(q+2*h)
            forward = (-3*f0+4*f1-f2)/(2*h)
            error = np.where(boundary, forward, centered)-costs
            roundoff = 16*np.finfo(float).eps*(np.abs(f0)+np.abs(f1)+np.abs(f2))/h
            bias = np.zeros_like(q)
        better = np.abs(error) < np.abs(best)
        best[better] = error[better]; best_step[better] = h[better]
        best_bias[better] = bias[better]; best_roundoff[better] = roundoff[better]
    interval_relative = 0.0; continuity_relative = 0.0; potential_continuity_relative = 0.0
    if isinstance(function, HingeRuntime):
        bounds = np.unique(np.r_[0.0, function.knots_flow, q.max()])
        left, right = bounds[:-1], bounds[1:]
        integral = _hinge_interval_integral(function, left, right)
        actual = function.potential(right)-function.potential(left)
        local_integral_scale = np.maximum((right-left)*np.maximum(1.0, 0.5*(function.cost(left)+function.cost(right))), np.finfo(float).tiny)
        interval_relative = float(np.max(np.abs(actual-integral)/local_integral_scale, initial=0.0))
        knots = function.knots_flow
        if len(knots):
            lower, upper = np.nextafter(knots, 0.0), np.nextafter(knots, np.inf)
            continuity_relative = float(np.max(np.abs(function.cost(upper)-function.cost(lower))/np.maximum(1.0, function.cost(knots))))
            potential_continuity_relative = float(np.max(np.abs(function.potential(upper)-function.potential(lower))/np.maximum(1.0, np.abs(function.potential(knots)))))
    relative = np.abs(best)/scale
    return {"policy":"LOCAL_KNOT_AWARE_AND_INTERVAL_INTEGRAL_V4", "target_relative":1e-6,
            "max_raw_absolute_error":float(np.max(np.abs(raw))),
            "max_raw_local_relative_error":float(np.max(np.abs(raw)/scale)),
            "max_corrected_absolute_error":float(np.max(np.abs(best))),
            "max_corrected_local_relative_error":float(np.max(relative)),
            "interval_integral_relative_error":interval_relative, "knot_continuity_relative_error":continuity_relative,
            "potential_knot_continuity_relative_error":potential_continuity_relative,
            "status":"PASS" if np.isfinite(relative).all() and max(float(relative.max()), interval_relative, continuity_relative, potential_continuity_relative) <= 1e-6 else "FAIL",
            "worst_probe":{"q":float(q[np.argmax(relative)]),"step":float(best_step[np.argmax(relative)]),
                           "crossing_bias":float(best_bias[np.argmax(relative)]),"roundoff_scale":float(best_roundoff[np.argmax(relative)])}}
