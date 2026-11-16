"""Exact offline primitive-bundle and series-macro reference oracles.

All time-valued inputs and outputs are in canonical seconds.  The safeguarded
scalar inversion in this module is deliberately instrumented and must never be
called by the explicit runtime kernels in :mod:`runtime_functions`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Sequence

import numpy as np


_EXACT_ROOT_CALLS = 0


def reset_exact_root_call_count() -> None:
    global _EXACT_ROOT_CALLS
    _EXACT_ROOT_CALLS = 0
    # A measured offline root benchmark must execute roots, not time cache hits.
    if "_solve_parameter_query" in globals():
        _solve_parameter_query.cache_clear()


def get_exact_root_call_count() -> int:
    return int(_EXACT_ROOT_CALLS)


@dataclass(frozen=True, slots=True)
class BundleSpec:
    """Physical members of one directed primitive parallel bundle."""

    member_ids: tuple[str, ...]
    free_time_s: np.ndarray
    capacity: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    parameter_key: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        arrays = tuple(
            np.ascontiguousarray(np.asarray(value, dtype=np.float64))
            for value in (self.free_time_s, self.capacity, self.alpha, self.beta)
        )
        n = len(self.member_ids)
        if n == 0 or any(value.shape != (n,) for value in arrays):
            raise ValueError("A bundle must contain aligned nonempty member arrays")
        if len(set(self.member_ids)) != n:
            raise ValueError("Primitive bundle member IDs must be unique")
        if any(np.any(~np.isfinite(value)) for value in arrays):
            raise ValueError("Primitive bundle parameters must be finite")
        if np.any(arrays[0] <= 0.0) or np.any(arrays[1] <= 0.0):
            raise ValueError("Free-flow time and capacity must be positive")
        if np.any(arrays[2] < 0.0) or np.any(arrays[3] <= 0.0):
            raise ValueError("BPR alpha must be nonnegative and beta positive")
        for name, value in zip(
            ("free_time_s", "capacity", "alpha", "beta"), arrays, strict=True
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "member_ids", tuple(map(str, self.member_ids)))
        object.__setattr__(self, "parameter_key", tuple(tuple(map(float, a)) for a in arrays))

    @property
    def member_count(self) -> int:
        return len(self.member_ids)


@dataclass(frozen=True, slots=True)
class OracleResult:
    query_flow: float
    member_allocation: np.ndarray
    marginal_cost_s: float
    potential_flow_seconds: float
    cost_derivative_s_per_flow: float
    active_member_indices: tuple[int, ...]
    allocation_residual: float
    kkt_residual_s: float
    root_iteration_count: int
    root_solve_used: bool
    status: str
    raw_allocation_residual: float = 0.0
    allocation_correction_l1: float = 0.0
    fw_gap_flow_seconds: float = 0.0
    fw_gap_per_flow_s: float = 0.0
    pivot_free_time_s: float = 0.0
    shifted_increment_s: float = 0.0
    certified: bool = False
    derivative_convention: str = "RIGHT_AT_ACTIVATION"
    primitive_certificates: tuple = ()


def member_cost(spec: BundleSpec, flow: np.ndarray) -> np.ndarray:
    value = np.asarray(flow, dtype=np.float64)
    if value.shape != (spec.member_count,) or np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("Member flow is invalid")
    result = spec.free_time_s.copy()
    dynamic = spec.alpha > 0.0
    if np.any(dynamic):
        result[dynamic] += (
            spec.free_time_s[dynamic]
            * spec.alpha[dynamic]
            * np.power(value[dynamic] / spec.capacity[dynamic], spec.beta[dynamic])
        )
    return result


def member_potential(spec: BundleSpec, flow: np.ndarray) -> np.ndarray:
    value = np.asarray(flow, dtype=np.float64)
    if value.shape != (spec.member_count,) or np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("Member flow is invalid")
    congestion = spec.free_time_s * spec.alpha * np.power(value / spec.capacity, spec.beta)
    # Evaluate congestion directly: subtracting free time from a rounded cost
    # erases the very increment needed at tiny flow.
    return value * (
        spec.free_time_s + congestion / (spec.beta + 1.0)
    )


def member_cost_derivative(spec: BundleSpec, flow: np.ndarray) -> np.ndarray:
    value = np.asarray(flow, dtype=np.float64)
    if value.shape != (spec.member_count,) or np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("Member flow is invalid")
    result = np.zeros(spec.member_count, dtype=np.float64)
    positive = value > 0.0
    result[positive] = (
        spec.free_time_s[positive]
        * spec.alpha[positive]
        * spec.beta[positive]
        * np.power(value[positive] / spec.capacity[positive], spec.beta[positive] - 1.0)
        / spec.capacity[positive]
    )
    beta_one = (~positive) & (spec.alpha > 0.0) & (spec.beta == 1.0)
    result[beta_one] = (
        spec.free_time_s[beta_one] * spec.alpha[beta_one] / spec.capacity[beta_one]
    )
    beta_below_one = (~positive) & (spec.alpha > 0.0) & (spec.beta < 1.0)
    result[beta_below_one] = np.inf
    return result


def _inverse_dynamic_supply(spec: BundleSpec, cost_s: float) -> np.ndarray:
    result = np.zeros(spec.member_count, dtype=np.float64)
    dynamic = spec.alpha > 0.0
    active = dynamic & (float(cost_s) > spec.free_time_s)
    if np.any(active):
        ratio = (float(cost_s) - spec.free_time_s[active]) / (
            spec.free_time_s[active] * spec.alpha[active]
        )
        result[active] = spec.capacity[active] * np.power(
            np.maximum(ratio, 0.0), 1.0 / spec.beta[active]
        )
    return result


class OracleCertificationError(ArithmeticError):
    """A numerical failure carries the exact primitive input and failed result."""

    def __init__(self, witness: dict):
        self.witness = witness
        super().__init__("UNCERTIFIED_PARALLEL_ORACLE: " + json.dumps(witness, sort_keys=True))


@lru_cache(maxsize=65536)
def _activation_profile(parameter_key: tuple) -> tuple:
    t, capacity, alpha, beta = (np.asarray(a) for a in parameter_key)
    dynamic = alpha > 0
    constant = float(t[~dynamic].min()) if (~dynamic).any() else math.inf
    levels = np.unique(t[dynamic & (t < constant)])

    def supply(cost):
        z = np.zeros(len(t))
        active = dynamic & (t < cost)
        z[active] = capacity[active] * np.power((cost-t[active]) / (t[active]*alpha[active]), 1/beta[active])
        return z

    bases = tuple(supply(float(level)) for level in levels)
    thresholds = np.asarray([math.fsum(z) for z in bases])
    cap_supply = supply(constant) if math.isfinite(constant) else np.zeros(len(t))
    for value in (*bases, thresholds, levels, cap_supply):
        value.setflags(write=False)
    return levels, thresholds, bases, constant, cap_supply


def _allocation_derivative(spec, z, pivot, delta):
    positive = z > 0
    dynamic = spec.alpha > 0
    if np.any(positive & ~dynamic):
        return 0.0
    # Use direct increments/slopes, never cost minus the large free-time base.
    slopes = member_cost_derivative(spec, z)
    if positive.any():
        active_slopes = slopes[positive]
        inverse = math.fsum(1.0 / v for v in active_slopes if v > 0)
        if np.any(active_slopes == 0):
            return 0.0
        new = (~positive) & (spec.free_time_s == pivot) & (delta == 0)
    else:
        inverse = 0.0
        new = spec.free_time_s == spec.free_time_s.min()
    if np.any(new & ((~dynamic) | (spec.beta > 1))):
        return 0.0
    linear = new & dynamic & (spec.beta == 1)
    inverse += math.fsum(spec.capacity[linear] / (spec.free_time_s[linear] * spec.alpha[linear]))
    return 1.0 / inverse if inverse > 0 else math.inf


def _certify(spec, q, z, pivot, delta, iterations, root_used, status, raw, correction):
    congestion = spec.free_time_s * spec.alpha * np.power(z / spec.capacity, spec.beta)
    shifted = (spec.free_time_s-pivot) + congestion
    minimum = float(shifted.min())
    positive = z > 0
    kkt = float(np.max(shifted[positive]-minimum)) if positive.any() else 0.0
    gap = math.fsum(z * (shifted-minimum))
    normalized_gap = gap / q if q > 0 else 0.0
    cost = pivot + minimum
    potential = math.fsum(spec.free_time_s*z + z*congestion/(spec.beta+1))
    residual = abs(math.fsum(z)-q)
    derivative = _allocation_derivative(spec, z, pivot, delta)
    flow_tolerance = 1e-8 + 1e-10*max(1.0, q)
    cost_tolerance = 1e-7 + 1e-10*max(1.0, abs(cost))
    finite = np.isfinite([cost, potential, residual, kkt, gap, normalized_gap]).all()
    valid_derivative = (np.isfinite(derivative) and derivative >= 0) or (q == 0 and derivative == math.inf)
    passed = bool(finite and valid_derivative and np.isfinite(z).all() and (z >= 0).all()
                  and residual <= flow_tolerance and kkt <= cost_tolerance
                  and 0 <= normalized_gap <= cost_tolerance)
    if not passed:
        raise OracleCertificationError({
            'member_ids':spec.member_ids, 'parameters':spec.parameter_key, 'q':q,
            'allocation':z.tolist(), 'primal':residual, 'raw_primal':raw, 'correction_l1':correction,
            'kkt_s':kkt, 'gap_per_flow_s':normalized_gap, 'potential':potential,
            'pivot':pivot, 'shifted_increment':delta, 'iterations':iterations, 'status':status,
        })
    z.setflags(write=False)
    return OracleResult(q, z, cost, potential, derivative, tuple(np.flatnonzero(positive).tolist()),
                        residual, kkt, iterations, root_used, status, raw, correction,
                        gap, normalized_gap, pivot, delta, True)


@lru_cache(maxsize=131072)
def _solve_parameter_query(parameter_key, q, flow_atol, flow_rtol, max_iterations):
    global _EXACT_ROOT_CALLS
    spec = BundleSpec(tuple(str(i) for i in range(len(parameter_key[0]))),
                      *(np.asarray(v) for v in parameter_key))
    t, capacity, alpha, beta = spec.free_time_s, spec.capacity, spec.alpha, spec.beta
    z = np.zeros(len(t)); pivot = float(t.min()); delta = 0.0
    iterations = 0; root_used = False; raw = 0.0; correction = 0.0
    homogeneous = np.all(alpha > 0) and np.all(t == t[0]) and np.all(beta == beta[0])
    if q == 0:
        status = 'ZERO_FLOW'
    elif len(t) == 1:
        z[0] = q; delta = float(t[0]*alpha[0]*(q/capacity[0])**beta[0]); status = 'SINGLETON_EXACT'
    elif homogeneous:
        # Normalized log shares avoid overflowing C**beta or inverse coefficients.
        logw = np.log(capacity)-np.log(t*alpha)/beta[0]
        weights = np.exp(logw-logw.max()); weights /= math.fsum(weights)
        z = q*weights; z[int(np.argmax(z))] += q-math.fsum(z)
        delta = float(t[0]*alpha[0]*(z[0]/capacity[0])**beta[0])
        status = 'HOMOGENEOUS_PARALLEL_EXACT'
    else:
        levels, thresholds, bases, constant, cap_supply = _activation_profile(parameter_key)
        cap_q = math.fsum(cap_supply) if math.isfinite(constant) else math.inf
        if math.isfinite(constant) and q >= cap_q:
            z = cap_supply.copy(); chosen = int(np.flatnonzero((alpha == 0) & (t == constant))[0])
            z[chosen] = q-cap_q; pivot = constant; status = 'CONSTANT_COST_CAP'
        else:
            j = max(0, int(np.searchsorted(thresholds, q, side='right')-1))
            pivot = float(levels[j]); z = bases[j].copy(); base_q = math.fsum(z)
            active = (alpha > 0) & (t <= pivot)
            newly_active = (alpha > 0) & (t == pivot)
            offset = pivot-t[active]
            remaining = max(0.0, q-base_q)
            next_level = min(float(levels[j+1]) if j+1 < len(levels) else math.inf, constant)
            upper = float(np.min(t[newly_active]*alpha[newly_active] * np.power(remaining/capacity[newly_active], beta[newly_active])))
            upper = min(upper, next_level-pivot)
            # Target is tighter than the caller's external demand tolerance;
            # the floating-point floor is a lower bound (MAX, never MIN).
            requested = min(flow_atol+flow_rtol*q, 2e-14*q)
            target = max(requested, 16*np.finfo(float).eps*q, np.nextafter(0.0, 1.0))
            def supply(d):
                out = np.zeros(len(t))
                out[active] = capacity[active]*np.power((offset+d)/(t[active]*alpha[active]), 1/beta[active])
                return out
            if remaining == 0:
                status = 'ACTIVATION_EXACT'
            elif upper == 0:
                z[int(np.flatnonzero(newly_active)[0])] += remaining
                status = 'UNDERFLOW_CERTIFIED'
            elif not np.isfinite(upper):
                raise OracleCertificationError({'parameters':parameter_key,'q':q,'reason':'NONFINITE_ACTIVATION_BRACKET'})
            else:
                root_used = True; _EXACT_ROOT_CALLS += 1
                lo, hi = 0.0, 1.0
                best = supply(upper); best_delta = upper; best_err = abs(math.fsum(best)-q)
                if math.fsum(best) < q-8*target:
                    raise OracleCertificationError({'parameters':parameter_key,'q':q,'reason':'ACTIVATION_BRACKET_FAILURE','upper':upper})
                for iterations in range(1, max_iterations+1):
                    mid = lo+(hi-lo)*0.5; d = upper*mid; trial = supply(d)
                    total = math.fsum(trial); error = abs(total-q)
                    if error < best_err:
                        best, best_delta, best_err = trial, d, error
                    if error <= target:
                        break
                    if total < q:
                        lo = mid
                    else:
                        hi = mid
                    if np.nextafter(lo, hi) >= hi:
                        break
                z, delta = best, best_delta
                raw = abs(math.fsum(z)-q)
                # Adjacent floats are only a stop condition. No successful label
                # until both tiny correction and an independent primal certificate.
                if raw > 8*target:
                    raise OracleCertificationError({'parameters':parameter_key,'q':q,'reason':'ROOT_DID_NOT_CONVERGE','raw_primal':raw,'target':target,'iterations':iterations})
                before = z.copy(); total = math.fsum(z)
                if total <= 0:
                    raise OracleCertificationError({'parameters':parameter_key,'q':q,'reason':'ZERO_ALLOCATION'})
                z *= q/total; z[int(np.argmax(z))] += q-math.fsum(z)
                correction = math.fsum(np.abs(z-before)); status = 'ROOT_SOLVED_CERTIFIED'
    return _certify(spec, q, z, pivot, delta, iterations, root_used, status, raw, correction)


def exact_bundle_oracle(spec: BundleSpec, query_flow: float, *, flow_atol: float = 1e-10,
                        flow_rtol: float = 2e-13, max_iterations: int = 160) -> OracleResult:
    """Activation-relative OFFLINE solve; every return has a primal/FW certificate."""
    q = float(query_flow)
    if not np.isfinite(q) or q < 0 or flow_atol < 0 or flow_rtol < 0 or max_iterations < 1:
        raise ValueError('Invalid oracle query/tolerance/iteration limit')
    try:
        return _solve_parameter_query(spec.parameter_key, q, flow_atol, flow_rtol, int(max_iterations))
    except OracleCertificationError as error:
        error.witness['member_ids'] = spec.member_ids
        raise


def oracle_cache_profile() -> dict:
    return {'activation_profiles':_activation_profile.cache_info()._asdict(),
            'queries':_solve_parameter_query.cache_info()._asdict(), 'offline_root_calls':get_exact_root_call_count()}


def clear_oracle_query_cache() -> None:
    """Separate uncached offline throughput measurements from memoized reuse."""
    _solve_parameter_query.cache_clear()


def exact_series_oracle(primitive_bundles: Sequence[BundleSpec], query_flow: float, *,
                        internal_charge_s: float = 0.0) -> OracleResult:
    """Certify each primitive at q, then sum dynamic responses and exact H once."""
    if not primitive_bundles:
        raise ValueError('A series macro-link must contain at least one primitive bundle')
    charge = float(internal_charge_s)
    if not np.isfinite(charge) or charge < 0:
        raise ValueError('Internal charge must be finite and nonnegative')
    pieces = [exact_bundle_oracle(spec, query_flow) for spec in primitive_bundles]
    allocation = np.concatenate([p.member_allocation for p in pieces]); allocation.setflags(write=False)
    active = []; offset = 0; certificates = []
    for spec, piece in zip(primitive_bundles, pieces, strict=True):
        active.extend(offset+i for i in piece.active_member_indices); offset += len(piece.member_allocation)
        certificates.append({'member_ids':spec.member_ids, 'q':piece.query_flow,
                             'marginal_cost_s':piece.marginal_cost_s,
                             'allocation_residual':piece.allocation_residual,'raw_allocation_residual':piece.raw_allocation_residual,
                             'correction_l1':piece.allocation_correction_l1,'kkt_s':piece.kkt_residual_s,
                             'fw_gap_per_flow_s':piece.fw_gap_per_flow_s,'pivot_s':piece.pivot_free_time_s,
                             'shifted_increment_s':piece.shifted_increment_s,'status':piece.status,'certified':piece.certified})
    return OracleResult(float(query_flow), allocation, math.fsum(p.marginal_cost_s for p in pieces)+charge,
                        math.fsum(p.potential_flow_seconds for p in pieces)+float(query_flow)*charge,
                        math.fsum(p.cost_derivative_s_per_flow for p in pieces), tuple(active),
                        max(p.allocation_residual for p in pieces), max(p.kkt_residual_s for p in pieces),
                        sum(p.root_iteration_count for p in pieces), any(p.root_solve_used for p in pieces), 'SERIES_EXACT_CERTIFIED',
                        max(p.raw_allocation_residual for p in pieces), math.fsum(p.allocation_correction_l1 for p in pieces),
                        math.fsum(p.fw_gap_flow_seconds for p in pieces), max(p.fw_gap_per_flow_s for p in pieces),
                        0.0, 0.0, all(p.certified for p in pieces), 'SUM_OF_PRIMITIVE_RIGHT_DERIVATIVES', tuple(certificates))


def exact_power_terms(spec: BundleSpec) -> tuple[float, tuple[tuple[float, float], ...]] | None:
    """Return ``intercept, ((beta, coefficient), ...)`` when analytically exact."""

    if spec.member_count == 1:
        coefficient = float(
            spec.free_time_s[0] * spec.alpha[0] / np.power(spec.capacity[0], spec.beta[0])
        )
        terms = () if coefficient == 0.0 else ((float(spec.beta[0]), coefficient),)
        return float(spec.free_time_s[0]), terms
    if np.all(spec.alpha == 0.0):
        return float(np.min(spec.free_time_s)), ()
    if np.any(spec.alpha == 0.0):
        cheapest_constant = float(np.min(spec.free_time_s[spec.alpha == 0.0]))
        if cheapest_constant == float(np.min(spec.free_time_s)):
            return cheapest_constant, ()
        return None
    if not np.all(spec.free_time_s == spec.free_time_s[0]):
        return None
    if not np.all(spec.beta == spec.beta[0]):
        return None
    exponent = float(spec.beta[0])
    member_coefficient = (
        spec.free_time_s * spec.alpha / np.power(spec.capacity, spec.beta)
    )
    equivalent_coefficient = float(
        np.power(np.sum(np.power(member_coefficient, -1.0 / exponent)), -exponent)
    )
    return float(spec.free_time_s[0]), ((exponent, equivalent_coefficient),)


def series_power_terms(
    primitive_bundles: Sequence[BundleSpec], *, internal_charge_s: float = 0.0
) -> tuple[float, tuple[tuple[float, float], ...]] | None:
    grouped: dict[float, float] = {}
    intercept = float(internal_charge_s)
    for spec in primitive_bundles:
        result = exact_power_terms(spec)
        if result is None:
            return None
        base, terms = result
        intercept += base
        for exponent, coefficient in terms:
            grouped[exponent] = grouped.get(exponent, 0.0) + coefficient
    return intercept, tuple(sorted(grouped.items()))


def primitive_function_key(
    spec: BundleSpec,
    *,
    graph_identity: str,
    clean_capacity_identity: str,
    capacity_factor: float,
    capacity_role: str = "USED_CAPACITY",
) -> str:
    order = np.argsort(np.asarray(spec.member_ids), kind="mergesort")
    payload = {
        "graph_identity": str(graph_identity),
        "clean_capacity_identity": str(clean_capacity_identity),
        "capacity_factor": float(capacity_factor).hex(),
        "capacity_role": str(capacity_role),
        "members": [
            {
                "member_id": str(spec.member_ids[int(index)]),
                "free_time_s": float(spec.free_time_s[index]).hex(),
                "capacity": float(spec.capacity[index]).hex(),
                "alpha": float(spec.alpha[index]).hex(),
                "beta": float(spec.beta[index]).hex(),
            }
            for index in order
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "p5_primitive_" + hashlib.sha256(encoded).hexdigest()


def macro_function_key(
    primitive_keys: Sequence[str],
    *,
    internal_component_keys: Sequence[str] = (),
    internal_charge_s: float = 0.0,
    internal_charge_estimator_id: str = "NO_INTERNAL_CHARGE",
    graph_identity: str,
) -> str:
    charge = float(internal_charge_s)
    if not np.isfinite(charge) or charge < 0.0:
        raise ValueError("Canonical macro internal charge must be finite and nonnegative")
    payload = {
        "graph_identity": str(graph_identity),
        "ordered_primitive_keys": list(map(str, primitive_keys)),
        "ordered_internal_component_keys": list(map(str, internal_component_keys)),
        "internal_charge_s": charge.hex(),
        "internal_charge_estimator_id": str(internal_charge_estimator_id),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "p5_macro_" + hashlib.sha256(encoded).hexdigest()
