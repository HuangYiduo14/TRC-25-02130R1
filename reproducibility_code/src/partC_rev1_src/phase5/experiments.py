"""Deterministic exact-oracle sampling and runtime-fit experiments."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .bundle_oracle import (
    BundleSpec,
    OracleResult,
    exact_bundle_oracle,
    exact_series_oracle,
    series_power_terms,
)
from .runtime_functions import (
    HingeRuntime,
    OneTermRuntime,
    PowerSumRuntime,
    RuntimeFunction,
    finite_difference_potential_error,
    fit_hinge,
    fit_one_term,
    pilot_weights,
    potential_gradient_audit,
)


@dataclass(frozen=True, slots=True)
class MacroExperiment:
    canonical_signature: str
    occurrence_id: str
    suite_id: str
    primitive_bundles: tuple[BundleSpec, ...]
    low_flow: float
    low_aon_flow: float
    high_flow: float
    chain_length: int
    endpoint_collision: bool
    internal_charge_s: float = 0.0
    internal_charge_estimator_id: str = "NO_INTERNAL_CHARGE"
    internal_component_ids: tuple[str, ...] = ()
    internal_component_charge_s: tuple[float, ...] = ()
    inclusion_reasons: tuple[str, ...] = ()
    dynamic_expression_key: str = ""
    calibration_request_key: str = ""


def dynamic_expression_key(bundles: Sequence[BundleSpec]) -> str:
    """Algebraic series sum; physical IDs/order remain in occurrence provenance."""
    primitive = []
    for spec in bundles:
        primitive.append(sorted(tuple(float(v).hex() for v in values) for values in zip(spec.free_time_s, spec.capacity, spec.alpha, spec.beta, strict=True)))
    encoded = json.dumps(sorted(primitive), separators=(",", ":")).encode()
    return "p5_dynamic_v4_" + hashlib.sha256(encoded).hexdigest()


def calibration_request_key(bundles, low_flow, low_aon_flow):
    payload = [dynamic_expression_key(bundles), float(low_flow).hex(), float(low_aon_flow).hex(), "DYNAMIC_Q4_GRID65_ACTIVATION_ADAPTIVE_WEIGHT_MAX_V4"]
    return "p5_calibration_request_v4_" + hashlib.sha256(json.dumps(payload).encode()).hexdigest()


_FIT_CODE_HASH = hashlib.sha256(b"".join((Path(__file__).parent / n).read_bytes() for n in ("experiments.py", "bundle_oracle.py", "runtime_functions.py"))).hexdigest()


def calibrated_model_key(experiment, flow, roles, weights, domain):
    payload = {"dynamic_expression_key":dynamic_expression_key(experiment.primitive_bundles),
               "low_current_hex":float(experiment.low_flow).hex(), "low_aon_hex":float(experiment.low_aon_flow).hex(),
               "flow_hex":[float(q).hex() for q in flow], "roles":roles, "weights_hex":[float(w).hex() for w in weights],
               "domain_hex":float(domain).hex(), "fit_policy":"DYNAMIC_ONLY_MAX_WEIGHTS_V4", "code_sha256":_FIT_CODE_HASH}
    return "p5_calibrated_v4_"+hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def _hash_rank(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).hexdigest()


def _chain_bucket(value: int) -> str:
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 5:
        return "3_5"
    if value <= 10:
        return "6_10"
    return "GT10"


def _member_bucket(value: int) -> str:
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 5:
        return "3_5"
    return "GT5"


def select_macro_sample(
    candidates: Sequence[MacroExperiment],
    *,
    maximum: int,
    force_extreme: int,
    namespace: str,
    force_signatures: Iterable[str] = (),
    progress_guard: Callable[[str], None] | None = None,
) -> tuple[list[MacroExperiment], dict[str, tuple[str, ...]]]:
    """Stable-hash stratification with deterministic forced extremes."""

    unique: dict[str, MacroExperiment] = {}
    for item in sorted(candidates, key=lambda value: (value.canonical_signature, value.occurrence_id)):
        current = unique.get(item.canonical_signature)
        if current is None or (item.suite_id == "s03" and current.suite_id != "s03"):
            unique[item.canonical_signature] = item
    values = list(unique.values())
    if len(values) <= maximum:
        reasons = {item.canonical_signature: ("ALL_UNIQUE_WITHIN_LIMIT",) for item in values}
        return sorted(values, key=lambda item: item.canonical_signature), reasons

    metrics: dict[str, dict[str, float]] = {}
    for ordinal, item in enumerate(values):
        if progress_guard is not None and ordinal % 64 == 0:
            progress_guard(f"sample_metric_{ordinal}_of_{len(values)}")
        members = [spec.member_count for spec in item.primitive_bundles]
        fftt = np.concatenate([spec.free_time_s for spec in item.primitive_bundles])
        zero_cost = float(
            sum(np.min(spec.free_time_s) for spec in item.primitive_bundles)
        )
        low_cost = exact_series_oracle(
            item.primitive_bundles,
            item.low_flow,
            internal_charge_s=0.0,
        ).marginal_cost_s
        metrics[item.canonical_signature] = {
            "chain_length": float(item.chain_length),
            "physical_member_count": float(sum(members)),
            "max_bundle_members": float(max(members)),
            "low_flow": float(item.low_flow),
            "zero_flow_cost": zero_cost,
            "low_exact_congestion_ratio": float(low_cost / zero_cost),
            "fftt_heterogeneity": float(fftt.max() / max(fftt.min(), np.finfo(float).tiny)),
            "endpoint_collision": float(item.endpoint_collision),
        }
    reasons_set: dict[str, set[str]] = {item.canonical_signature: set() for item in values}
    requested_forced = set(map(str, force_signatures)) & set(unique)
    if len(requested_forced) > maximum:
        requested_forced = set(
            sorted(
                requested_forced,
                key=lambda signature: _hash_rank(
                    namespace + ":explicit-force-trim", signature
                ),
            )[:maximum]
        )
    forced: set[str] = set(requested_forced)
    for signature in requested_forced:
        reasons_set[signature].add("FORCED_ACCEPTED_PATH_INTERSECTION")
    for metric in (
        "chain_length",
        "physical_member_count",
        "low_flow",
        "low_exact_congestion_ratio",
        "fftt_heterogeneity",
        "endpoint_collision",
    ):
        ordered = sorted(
            values,
            key=lambda item: (
                -metrics[item.canonical_signature][metric],
                _hash_rank(namespace + ":extreme:" + metric, item.canonical_signature),
            ),
        )[:force_extreme]
        for item in ordered:
            forced.add(item.canonical_signature)
            reasons_set[item.canonical_signature].add("FORCED_TOP_" + metric.upper())
    # Preserve deterministic representatives of the requested quantitative
    # spectrum in every suite.  These are evidence strata, not fit choices.
    quantile_metrics = (
        "zero_flow_cost",
        "low_flow",
        "low_exact_congestion_ratio",
        "fftt_heterogeneity",
        "physical_member_count",
    )
    for suite_id in sorted({item.suite_id for item in values}):
        suite_values = [item for item in values if item.suite_id == suite_id]
        for metric in quantile_metrics:
            ordered = sorted(
                suite_values,
                key=lambda item: (
                    metrics[item.canonical_signature][metric],
                    _hash_rank(namespace + ":quantile:" + metric, item.canonical_signature),
                ),
            )
            for quantile in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0):
                position = int(round(quantile * (len(ordered) - 1)))
                item = ordered[position]
                forced.add(item.canonical_signature)
                reasons_set[item.canonical_signature].add(
                    f"FORCED_{suite_id.upper()}_{metric.upper()}_Q{int(100 * quantile):03d}"
                )
    if len(forced) > maximum:
        # Every forced object retains at least one reason; rank first by number
        # of distinct extreme lists, then by stable hash.
        other_forced = forced - requested_forced
        retained_other = sorted(
                other_forced,
                key=lambda signature: (
                    -len(reasons_set[signature]),
                    _hash_rank(namespace + ":forced-trim", signature),
                ),
            )[: max(0, maximum - len(requested_forced))]
        forced = requested_forced | set(retained_other)

    remaining_slots = maximum - len(forced)
    not_forced = [item for item in values if item.canonical_signature not in forced]
    strata: dict[tuple[str, str, str, str], list[MacroExperiment]] = {}
    for item in not_forced:
        maximum_members = max(spec.member_count for spec in item.primitive_bundles)
        key = (
            item.suite_id,
            "ZERO" if item.low_flow == 0.0 else "POSITIVE",
            _chain_bucket(item.chain_length),
            _member_bucket(maximum_members),
        )
        strata.setdefault(key, []).append(item)
    weighted_total = sum(len(group) * (2 if key[0] == "s03" else 1) for key, group in strata.items())
    selected = set(forced)
    allocations: list[tuple[tuple[str, str, str, str], int]] = []
    for key, group in sorted(strata.items()):
        share = remaining_slots * len(group) * (2 if key[0] == "s03" else 1) / max(1, weighted_total)
        allocations.append((key, min(len(group), int(np.floor(share)))))
    used = sum(count for _, count in allocations)
    allocation_map = dict(allocations)
    for key, group in sorted(strata.items()):
        take = allocation_map[key]
        ordered = sorted(group, key=lambda item: _hash_rank(namespace + ":stratum", item.canonical_signature))
        for item in ordered[:take]:
            selected.add(item.canonical_signature)
            reasons_set[item.canonical_signature].add("STABLE_HASH_STRATUM_" + "_".join(key))
    if len(selected) < maximum:
        leftovers = sorted(
            (item for item in values if item.canonical_signature not in selected),
            key=lambda item: _hash_rank(namespace + ":remainder", item.canonical_signature),
        )
        for item in leftovers[: maximum - len(selected)]:
            selected.add(item.canonical_signature)
            reasons_set[item.canonical_signature].add("STABLE_HASH_REMAINDER")
    chosen = sorted((unique[signature] for signature in selected), key=lambda item: item.canonical_signature)
    reasons = {signature: tuple(sorted(reasons_set[signature])) for signature in selected}
    return chosen, reasons


def activation_flows(spec: BundleSpec) -> np.ndarray:
    values: list[float] = []
    for threshold in np.unique(spec.free_time_s):
        active = (spec.alpha > 0.0) & (threshold > spec.free_time_s)
        if not np.any(active):
            continue
        ratio = (threshold - spec.free_time_s[active]) / (
            spec.free_time_s[active] * spec.alpha[active]
        )
        flow = float(
            np.sum(
                spec.capacity[active] * np.power(ratio, 1.0 / spec.beta[active]),
                dtype=np.float64,
            )
        )
        if flow > 0.0 and np.isfinite(flow):
            values.append(flow)
    return np.unique(np.asarray(values, dtype=np.float64))


def _structural_flow_scale(bundles: Sequence[BundleSpec]) -> float:
    values = []
    for spec in bundles:
        dynamic = spec.alpha > 0.0
        if np.any(dynamic):
            values.append(
                float(
                    np.sum(
                        spec.capacity[dynamic]
                        * np.power(spec.alpha[dynamic], -1.0 / spec.beta[dynamic]),
                        dtype=np.float64,
                    )
                )
            )
    return max([1.0, *values])


def _q_at_cost_ratio_four(
    bundles: Sequence[BundleSpec],
    c0: float,
    *,
    internal_charge_s: float,
) -> tuple[float, str]:
    target = 4.0 * c0
    high = _structural_flow_scale(bundles)
    guard = high * 1.0e4
    value = exact_series_oracle(
        bundles, high, internal_charge_s=internal_charge_s
    ).marginal_cost_s
    while value < target and high < guard:
        high *= 2.0
        value = exact_series_oracle(
            bundles, high, internal_charge_s=internal_charge_s
        ).marginal_cost_s
    if value < target:
        return np.nan, "RATIO4_NOT_REACHED_BEFORE_STRUCTURAL_GUARD"
    low = 0.0
    for _ in range(64):
        mid = 0.5 * (low + high)
        value = exact_series_oracle(
            bundles, mid, internal_charge_s=internal_charge_s
        ).marginal_cost_s
        if value < target:
            low = mid
        else:
            high = mid
        if np.nextafter(low, high) >= high:
            break
    return float(high), "RATIO4_SOLVED"


def build_oracle_grid(experiment: MacroExperiment) -> tuple[np.ndarray, list[str], dict[float, OracleResult], dict[str, object]]:
    bundles = experiment.primitive_bundles
    charge = 0.0
    zero = exact_series_oracle(bundles, 0.0, internal_charge_s=charge)
    q4, q4_status = _q_at_cost_ratio_four(
        bundles, zero.marginal_cost_s, internal_charge_s=charge
    )
    candidates = [2.0 * experiment.low_flow, 1.5 * experiment.low_aon_flow]
    if np.isfinite(q4):
        candidates.append(q4)
    structural = _structural_flow_scale(bundles)
    domain = max(candidates) if max(candidates, default=0.0) > 0.0 else structural
    guard = structural * 1.0e4
    domain_status = "FROZEN_PILOT_QMAX_RULE"
    if domain > guard:
        if experiment.low_flow > guard or experiment.low_aon_flow > guard:
            raise ArithmeticError("Low reference exceeds the explicit physical query guard; cannot relabel a clamped point")
        domain = guard
        domain_status = "PHYSICAL_RESOURCE_GUARD_APPLIED_AND_RECORDED"
    role_by_hex: dict[str, set[str]] = {}
    value_by_hex: dict[str, float] = {}

    def add(value: float, role: str) -> None:
        if float(value) > domain:
            return  # Neighborhood/domain evidence outside the recorded guard is not relabeled.
        q = max(float(value), 0.0)
        key = q.hex()
        value_by_hex[key] = q
        role_by_hex.setdefault(key, set()).add(role)

    for q in domain * np.square(np.linspace(0.0, 1.0, 65)):
        add(float(q), "BASE_NORMALIZED_65")
    add(0.0, "ZERO")
    add(experiment.low_flow, "LOW_CURRENT")
    for factor in (0.50, 0.75, 0.90, 0.99, 1.01, 1.10, 1.25, 1.50):
        add(experiment.low_flow * factor, "LOW_CURRENT_NEIGHBORHOOD")
    add(experiment.low_aon_flow, "LOW_AON_DOMAIN_EVIDENCE")
    add(domain, "PILOT_DOMAIN_MAX")
    if np.isfinite(q4):
        add(q4, "EXACT_COST_RATIO_4")
    for spec in bundles:
        for activation in activation_flows(spec):
            if activation <= domain:
                add(activation * (1.0 - 1.0e-7), "ACTIVATION_NEIGHBORHOOD")
                add(activation, "ACTIVATION_POINT")
                add(activation * (1.0 + 1.0e-7), "ACTIVATION_NEIGHBORHOOD")
    ordered = np.asarray(sorted(value_by_hex.values()), dtype=np.float64)
    cache = {
        float(q): exact_series_oracle(bundles, float(q), internal_charge_s=charge)
        for q in ordered
    }
    if len(ordered) > 1:
        adaptive: list[tuple[float, float]] = []
        for left, right in zip(ordered[:-1], ordered[1:], strict=True):
            mid = 0.5 * (float(left) + float(right))
            left_result = cache[float(left)]
            right_result = cache[float(right)]
            score = (
                abs(
                    right_result.cost_derivative_s_per_flow
                    - left_result.cost_derivative_s_per_flow
                )
                * (float(right) - float(left))
                / max(
                    1.0,
                    left_result.marginal_cost_s,
                    right_result.marginal_cost_s,
                )
            )
            adaptive.append((score, mid))
        for _, mid in sorted(adaptive, reverse=True)[:8]:
            add(mid, "ADAPTIVE_INTERPOLATION_REFINEMENT")
    ordered = np.asarray(sorted(value_by_hex.values()), dtype=np.float64)
    roles = [";".join(sorted(role_by_hex[q.hex()])) for q in ordered]
    # Retain only selected adaptive cache entries plus the calibration grid.
    for q in ordered:
        if float(q) not in cache:
            cache[float(q)] = exact_series_oracle(
                bundles, float(q), internal_charge_s=charge
            )
    cache = {float(q): cache[float(q)] for q in ordered}
    metadata = {
        "flow_domain_max": float(domain),
        "q_ratio4": float(q4) if np.isfinite(q4) else None,
        "q_ratio4_status": q4_status,
        "domain_status": domain_status,
        "structural_flow_scale": structural,
        "calibration_point_count": len(ordered),
        "internal_charge_s": experiment.internal_charge_s,
        "grid_cost_role": "DYNAMIC_ONLY_B_EXCLUDES_H",
        "internal_charge_estimator_id": experiment.internal_charge_estimator_id,
    }
    activation = np.asarray(["ACTIVATION" in role for role in roles])
    weights = pilot_weights(ordered, experiment.low_flow, activation)
    metadata["dynamic_expression_key"] = dynamic_expression_key(bundles)
    metadata["calibration_request_key"] = calibration_request_key(bundles, experiment.low_flow, experiment.low_aon_flow)
    metadata["calibrated_model_key"] = calibrated_model_key(experiment, ordered, roles, weights, domain)
    metadata["calibration_grid_sha256"] = hashlib.sha256(ordered.tobytes()).hexdigest()
    metadata["calibration_weights_sha256"] = hashlib.sha256(weights.tobytes()).hexdigest()
    return ordered, roles, cache, metadata


def _model_parameters(function: RuntimeFunction) -> tuple[dict[str, object], list[float], int, int]:
    if isinstance(function, PowerSumRuntime):
        return (
            {
                "intercept_s": function.intercept_s,
                "exponents": function.exponents.tolist(),
                "coefficients": function.coefficients.tolist(),
            },
            [],
            1 + 2 * len(function.exponents),
            function.arithmetic_term_count,
        )
    if isinstance(function, OneTermRuntime):
        return (
            {
                "intercept_s": function.intercept_s,
                "scale_flow": function.scale_flow,
                "exponent": function.exponent,
                "eta_s": function.eta_s,
            },
            [],
            4,
            1,
        )
    if isinstance(function, HingeRuntime):
        return (
            {
                "intercept_s": function.intercept_s,
                "segment_slopes_s_per_flow": function.segment_slopes_s_per_flow.tolist(),
            },
            function.knots_flow.tolist(),
            1 + len(function.knots_flow) + len(function.segment_slopes_s_per_flow),
            function.arithmetic_term_count,
        )
    raise TypeError(type(function))


def _fit_row(
    experiment: MacroExperiment,
    runtime_class: str,
    variant_id: str,
    function: RuntimeFunction,
    flow: np.ndarray,
    exact_cost: np.ndarray,
    exact_potential: np.ndarray,
    exact_derivative: np.ndarray,
    weights: np.ndarray,
    domain: float,
    low_exact: OracleResult,
    high_exact: OracleResult,
) -> dict[str, object]:
    predicted = function.cost(flow)
    predicted_potential = function.potential(flow)
    absolute = np.abs(predicted - exact_cost)
    charge = float(experiment.internal_charge_s)
    exact_base_cost = exact_cost
    exact_base_potential = exact_potential
    predicted_base_potential = predicted_potential
    relative = absolute / np.maximum(exact_cost + charge, 1.0)
    base_relative = absolute / np.maximum(exact_base_cost, 1.0)
    potential_relative = np.abs(predicted_potential - exact_potential) / np.maximum(
        np.abs(exact_potential + charge*flow), 1.0
    )
    base_potential_relative = np.abs(
        predicted_base_potential - exact_base_potential
    ) / np.maximum(
        np.abs(exact_base_potential), 1.0
    )
    predicted_derivative = function.derivative(flow)
    derivative_absolute = np.abs(predicted_derivative - exact_derivative)
    finite_derivative = np.isfinite(derivative_absolute)
    weighted_rms = float(
        np.sqrt(np.sum(weights * np.square(relative)) / np.sum(weights))
    )
    low_absolute = abs(
        float(function.cost(experiment.low_flow)) - low_exact.marginal_cost_s
    )
    high_absolute = abs(
        float(function.cost(experiment.high_flow)) - high_exact.marginal_cost_s
    )
    low_relative = low_absolute / max(low_exact.marginal_cost_s + charge, 1.0)
    high_relative = high_absolute / max(high_exact.marginal_cost_s + charge, 1.0)
    parameter, knots, parameter_count, term_count = _model_parameters(function)
    monotonicity = bool(np.all(np.diff(predicted) >= -1.0e-10))
    convexity = bool(np.all(predicted_derivative >= -1.0e-12))
    zero_pass = bool(abs(float(function.cost(0.0)) - exact_cost[0]) <= 1.0e-10)
    fd = potential_gradient_audit(function, flow)
    finite_difference_error = fd["max_corrected_absolute_error"]
    return {
        "canonical_macro_signature": experiment.canonical_signature,
        "runtime_class": runtime_class,
        "variant_id": variant_id,
        "flow_domain_max": domain,
        "internal_charge_s": charge,
        "internal_charge_estimator_id": experiment.internal_charge_estimator_id,
        "parameter_json": json.dumps(parameter, sort_keys=True, separators=(",", ":")),
        "knot_json": json.dumps(knots, separators=(",", ":")),
        "calibration_point_count": len(flow),
        "max_abs_cost_error": float(absolute.max(initial=0.0)),
        "max_rel_cost_error": float(relative.max(initial=0.0)),
        "mean_rel_cost_error": float(relative.mean()),
        "median_rel_cost_error": float(np.median(relative)),
        "p95_rel_cost_error": float(np.quantile(relative, 0.95)),
        "weighted_rms_rel_cost_error": weighted_rms,
        "max_base_rel_cost_error": float(base_relative.max(initial=0.0)),
        "p95_base_rel_cost_error": float(np.quantile(base_relative, 0.95)),
        "low_rel_cost_error": float(low_relative),
        "low_base_rel_cost_error": float(
            low_absolute / max(low_exact.marginal_cost_s, 1.0)
        ),
        "high_heldout_rel_cost_error": float(high_relative),
        "high_heldout_base_rel_cost_error": float(
            high_absolute / max(high_exact.marginal_cost_s, 1.0)
        ),
        "domain_boundary_rel_cost_error": float(relative[-1]),
        "max_rel_potential_error": float(potential_relative.max(initial=0.0)),
        "max_base_rel_potential_error": float(
            base_potential_relative.max(initial=0.0)
        ),
        "max_abs_cost_derivative_error": float(
            derivative_absolute[finite_derivative].max(initial=0.0)
        ),
        "nonfinite_cost_derivative_error_count": int(np.count_nonzero(~finite_derivative)),
        "finite_difference_error": finite_difference_error,
        "finite_difference_relative_error": fd["max_corrected_local_relative_error"],
        "raw_centered_fd_relative_error": fd["max_raw_local_relative_error"],
        "raw_centered_fd_absolute_error": fd["max_raw_absolute_error"],
        "interval_integral_relative_error": fd["interval_integral_relative_error"],
        "knot_continuity_relative_error": fd["knot_continuity_relative_error"],
        "potential_knot_continuity_relative_error": fd["potential_knot_continuity_relative_error"],
        "potential_identity_status": fd["status"],
        "fd_worst_probe_json": json.dumps(fd["worst_probe"],sort_keys=True),
        "model_schema_version": "DYNAMIC_RESPONSE_PLUS_OCCURRENCE_H_V4",
        "parameter_cost_role": "DYNAMIC_ONLY",
        "weighted_rms_dynamic_rel_cost_error": float(np.sqrt(np.sum(weights*base_relative**2)/np.sum(weights))),
        "monotonicity_pass": monotonicity,
        "convexity_pass": convexity,
        "zero_intercept_pass": zero_pass,
        "parameter_count": int(parameter_count),
        "arithmetic_term_count": int(term_count),
        "recommended_status": "EVALUATED_NOT_SELECTED",
        "failure_reasons_json": "[]",
    }


def _diagnostic_best_runtime(
    bundles: Sequence[BundleSpec],
    *,
    internal_charge_s: float,
    flow: np.ndarray,
    exact_cost: np.ndarray,
    weights: np.ndarray,
    domain: float,
) -> tuple[str, RuntimeFunction, float]:
    """Best bounded candidate under the frozen weighted calibration objective."""

    charge = float(internal_charge_s)
    exact_cost = exact_cost-charge
    intercept = float(exact_cost[0])
    candidates: list[tuple[str, RuntimeFunction]] = []
    exact_terms = series_power_terms(
        bundles, internal_charge_s=0.0
    )
    if exact_terms is not None:
        base, terms = exact_terms
        candidates.append(
            (
                "EXACT_CLOSED_FORM",
                PowerSumRuntime(
                    base,
                    np.asarray([value[0] for value in terms], dtype=np.float64),
                    np.asarray([value[1] for value in terms], dtype=np.float64),
                ),
            )
        )
    candidates.append(
        (
            "FIT_ONE_TERM_GLOBAL_EXPONENT",
            fit_one_term(
                flow,
                exact_cost,
                intercept_s=intercept,
                exponent=4.0,
                scale_flow=max(domain, 1.0),
                weights=weights,
            ),
        )
    )
    discrete: list[tuple[float, str, RuntimeFunction]] = []
    denominator = np.maximum(exact_cost, np.maximum(intercept, 1.0))
    for exponent in (2.0, 3.0, 4.0, 5.0, 6.0):
        function = fit_one_term(
            flow,
            exact_cost,
            intercept_s=intercept,
            exponent=exponent,
            scale_flow=max(domain, 1.0),
            weights=weights,
        )
        relative = (function.cost(flow) - exact_cost) / denominator
        objective = float(
            np.sqrt(np.sum(weights * np.square(relative)) / np.sum(weights))
        )
        discrete.append((objective, f"BETA_{exponent:g}", function))
    _, discrete_variant, discrete_function = min(
        discrete, key=lambda value: (value[0], value[1])
    )
    candidates.append(
        (
            f"FIT_ONE_TERM_DISCRETE_EXPONENT:{discrete_variant}",
            discrete_function,
        )
    )
    for segments in (4, 8):
        candidates.append(
            (
                f"MONOTONE_HINGE_{segments}",
                fit_hinge(
                    flow,
                    exact_cost,
                    intercept_s=intercept,
                    flow_domain_max=domain,
                    segment_count=segments,
                    weights=weights,
                ),
            )
        )
    ranked: list[tuple[float, str, RuntimeFunction]] = []
    for name, function in candidates:
        predicted = function.cost(flow)
        derivative = function.derivative(flow)
        if (
            not np.isfinite(predicted).all()
            or not np.isfinite(derivative).all()
            or np.any(derivative < -1.0e-12)
            or np.any(np.diff(predicted) < -1.0e-10)
        ):
            continue
        relative = (predicted - exact_cost) / denominator
        objective = float(
            np.sqrt(np.sum(weights * np.square(relative)) / np.sum(weights))
        )
        ranked.append((objective, name, function))
    if not ranked:
        raise RuntimeError("No valid bounded runtime exists for split diagnostic")
    objective, name, function = min(ranked, key=lambda value: (value[0], value[1]))
    return name, replace(function, intercept_s=function.intercept_s+charge), objective


def _median_cost_runtime_ns(
    functions: Sequence[RuntimeFunction],
    *,
    additive_charge_s: float,
    domain: float,
) -> float:
    probe = max(domain, 1.0) * np.square(np.linspace(0.0, 1.0, 4096))

    def evaluate() -> np.ndarray:
        result = np.full_like(probe, additive_charge_s)
        for function in functions:
            result += function.cost(probe)
        return result

    for _ in range(5):
        evaluate()
    samples = []
    for _ in range(30):
        started = time.perf_counter_ns()
        evaluate()
        samples.append(time.perf_counter_ns() - started)
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def run_macro_experiments(
    experiments: Sequence[MacroExperiment],
    *,
    progress_guard: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oracle_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    domain_rows: list[dict[str, object]] = []
    for ordinal, experiment in enumerate(experiments):
        object_started = time.perf_counter()
        if progress_guard is not None and ordinal % 8 == 0:
            progress_guard(f"oracle_fit_{ordinal}_of_{len(experiments)}")
        flow, roles, cache, metadata = build_oracle_grid(experiment)
        grid_certified_seconds = time.perf_counter() - object_started
        exact_cost = np.asarray([cache[float(q)].marginal_cost_s for q in flow])
        exact_potential = np.asarray([cache[float(q)].potential_flow_seconds for q in flow])
        exact_derivative = np.asarray([cache[float(q)].cost_derivative_s_per_flow for q in flow])
        for q, role in zip(flow, roles, strict=True):
            result = cache[float(q)]
            oracle_rows.append(
                {
                    "canonical_macro_signature": experiment.canonical_signature,
                    "suite_occurrence_id": experiment.occurrence_id,
                    "query_flow": float(q),
                    "query_role": role,
                    "exact_cost": result.marginal_cost_s + experiment.internal_charge_s,
                    "exact_base_cost": result.marginal_cost_s,
                    "exact_potential": result.potential_flow_seconds + experiment.internal_charge_s*float(q),
                    "exact_base_potential": result.potential_flow_seconds,
                    "internal_charge_s": experiment.internal_charge_s,
                    "internal_charge_estimator_id": experiment.internal_charge_estimator_id,
                    "exact_cost_derivative": result.cost_derivative_s_per_flow,
                    "active_member_count": len(result.active_member_indices),
                    "allocation_residual": result.allocation_residual,
                    "kkt_residual": result.kkt_residual_s,
                    "root_iteration_count": result.root_iteration_count,
                    "status": result.status,
                    "certified": result.certified,
                    "raw_allocation_residual": result.raw_allocation_residual,
                    "allocation_correction_l1": result.allocation_correction_l1,
                    "fw_gap_flow_seconds": result.fw_gap_flow_seconds,
                    "fw_gap_per_flow_s": result.fw_gap_per_flow_s,
                    "primitive_certificates_json": json.dumps(result.primitive_certificates,separators=(",", ":")),
                    "primitive_count": len(experiment.primitive_bundles),
                }
            )
        high = exact_series_oracle(
            experiment.primitive_bundles,
            experiment.high_flow,
            internal_charge_s=0.0,
        )
        oracle_rows.append(
            {
                "canonical_macro_signature": experiment.canonical_signature,
                "suite_occurrence_id": experiment.occurrence_id,
                "query_flow": experiment.high_flow,
                "query_role": "HIGH_CURRENT_HELDOUT_ONLY",
                "exact_cost": high.marginal_cost_s + experiment.internal_charge_s,
                "exact_base_cost": high.marginal_cost_s,
                "exact_potential": high.potential_flow_seconds + experiment.internal_charge_s*experiment.high_flow,
                "exact_base_potential": high.potential_flow_seconds,
                "internal_charge_s": experiment.internal_charge_s,
                "internal_charge_estimator_id": experiment.internal_charge_estimator_id,
                "exact_cost_derivative": high.cost_derivative_s_per_flow,
                "active_member_count": len(high.active_member_indices),
                "allocation_residual": high.allocation_residual,
                "kkt_residual": high.kkt_residual_s,
                "root_iteration_count": high.root_iteration_count,
                "status": high.status,
                    "certified": high.certified,
                    "raw_allocation_residual": high.raw_allocation_residual,
                    "allocation_correction_l1": high.allocation_correction_l1,
                    "fw_gap_flow_seconds": high.fw_gap_flow_seconds,
                    "fw_gap_per_flow_s": high.fw_gap_per_flow_s,
                    "primitive_certificates_json": json.dumps(high.primitive_certificates,separators=(",", ":")),
                    "primitive_count": len(experiment.primitive_bundles),
            }
        )
        activation = np.asarray(["ACTIVATION" in role for role in roles], dtype=bool)
        weights = pilot_weights(flow, experiment.low_flow, activation)
        intercept = float(exact_cost[0])
        domain = float(metadata["flow_domain_max"])
        low = cache.get(float(experiment.low_flow))
        if low is None:
            low = exact_series_oracle(
                experiment.primitive_bundles,
                experiment.low_flow,
                internal_charge_s=0.0,
            )
        exact_terms = series_power_terms(
            experiment.primitive_bundles,
            internal_charge_s=0.0,
        )
        local_indexes: list[int] = []
        if exact_terms is not None:
            base, terms = exact_terms
            function = PowerSumRuntime(
                base,
                np.asarray([value[0] for value in terms], dtype=np.float64),
                np.asarray([value[1] for value in terms], dtype=np.float64),
            )
            local_indexes.append(len(fit_rows))
            fit_rows.append(
                _fit_row(
                    experiment,
                    "EXACT_CLOSED_FORM",
                    "EXACT_POWER_SUM",
                    function,
                    flow,
                    exact_cost,
                    exact_potential,
                    exact_derivative,
                    weights,
                    domain,
                    low,
                    high,
                )
            )
        global_function = fit_one_term(
            flow,
            exact_cost,
            intercept_s=intercept,
            exponent=4.0,
            scale_flow=max(domain, 1.0),
            weights=weights,
        )
        global_index = len(fit_rows)
        local_indexes.append(global_index)
        fit_rows.append(
            _fit_row(
                experiment,
                "FIT_ONE_TERM_GLOBAL_EXPONENT",
                "GLOBAL_BETA_4",
                global_function,
                flow,
                exact_cost,
                exact_potential,
                exact_derivative,
                weights,
                domain,
                low,
                high,
            )
        )
        discrete_indexes: list[int] = []
        for exponent in (2.0, 3.0, 4.0, 5.0, 6.0):
            function = fit_one_term(
                flow,
                exact_cost,
                intercept_s=intercept,
                exponent=exponent,
                scale_flow=max(domain, 1.0),
                weights=weights,
            )
            discrete_indexes.append(len(fit_rows))
            local_indexes.append(len(fit_rows))
            fit_rows.append(
                _fit_row(
                    experiment,
                    "FIT_ONE_TERM_DISCRETE_EXPONENT",
                    f"BETA_{exponent:g}",
                    function,
                    flow,
                    exact_cost,
                    exact_potential,
                    exact_derivative,
                    weights,
                    domain,
                    low,
                    high,
                )
            )
        hinge_indexes: dict[int, int] = {}
        for segments in (4, 8):
            function = fit_hinge(
                flow,
                exact_cost,
                intercept_s=intercept,
                flow_domain_max=domain,
                segment_count=segments,
                weights=weights,
            )
            hinge_indexes[segments] = len(fit_rows)
            local_indexes.append(len(fit_rows))
            fit_rows.append(
                _fit_row(
                    experiment,
                    f"MONOTONE_HINGE_{segments}",
                    f"SQRT_KNOT_{segments}_SEGMENTS",
                    function,
                    flow,
                    exact_cost,
                    exact_potential,
                    exact_derivative,
                    weights,
                    domain,
                    low,
                    high,
                )
            )
        for index in local_indexes:
            fit_rows[index]["dynamic_expression_key"] = metadata["dynamic_expression_key"]
            fit_rows[index]["calibration_request_key"] = metadata["calibration_request_key"]
            fit_rows[index]["calibrated_model_key"] = metadata["calibrated_model_key"]
            if fit_rows[index]["potential_identity_status"] != "PASS":
                raise ArithmeticError("Runtime potential identity failed: " + json.dumps(fit_rows[index], default=str))
        threshold = 0.01
        exact_indexes = [index for index in local_indexes if fit_rows[index]["runtime_class"] == "EXACT_CLOSED_FORM"]
        best_discrete = min(
            discrete_indexes,
            key=lambda index: (
                fit_rows[index]["weighted_rms_dynamic_rel_cost_error"],
                fit_rows[index]["variant_id"],
            ),
        )
        fit_rows[best_discrete]["discrete_registry_status"] = (
            "BEST_WEIGHTED_PILOT_OBJECTIVE"
        )
        for index in discrete_indexes:
            fit_rows[index].setdefault(
                "discrete_registry_status", "NOT_BEST_WEIGHTED_PILOT_OBJECTIVE"
            )
        cascade = [*exact_indexes, global_index, best_discrete, hinge_indexes[4], hinge_indexes[8]]
        chosen = next(
            (
                index
                for index in cascade
                if fit_rows[index]["max_base_rel_cost_error"] <= threshold
                and fit_rows[index]["monotonicity_pass"]
                and fit_rows[index]["convexity_pass"]
                and fit_rows[index]["zero_intercept_pass"]
                and fit_rows[index]["nonfinite_cost_derivative_error_count"] == 0
                and fit_rows[index]["finite_difference_relative_error"] <= 1.0e-6
            ),
            None,
        )
        if chosen is None:
            for index in local_indexes:
                reasons = []
                if fit_rows[index]["max_base_rel_cost_error"] > threshold:
                    reasons.append("MAX_REL_COST_ERROR_EXCEEDS_PILOT_1_PERCENT_DIAGNOSTIC")
                if not fit_rows[index]["monotonicity_pass"]:
                    reasons.append("MONOTONICITY_FAILURE")
                if not fit_rows[index]["convexity_pass"]:
                    reasons.append("CONVEXITY_FAILURE")
                if not fit_rows[index]["zero_intercept_pass"]:
                    reasons.append("ZERO_INTERCEPT_FAILURE")
                if fit_rows[index]["nonfinite_cost_derivative_error_count"]:
                    reasons.append("NONFINITE_DERIVATIVE_ERROR")
                if fit_rows[index]["finite_difference_relative_error"] > 1.0e-6:
                    reasons.append("POTENTIAL_COST_FINITE_DIFFERENCE_FAILURE")
                fit_rows[index]["failure_reasons_json"] = json.dumps(
                    reasons, separators=(",", ":")
                )
            status = "SPLIT_OR_RETAIN_DIAGNOSTIC"
        else:
            fit_rows[chosen]["recommended_status"] = "CASCADE_SELECTED_PILOT_1PCT_DIAGNOSTIC"
            status = str(fit_rows[chosen]["runtime_class"])
        split_scores: list[tuple[float, int]] = []
        for boundary in range(1, len(experiment.primitive_bundles)):
            left = experiment.primitive_bundles[boundary - 1]
            right = experiment.primitive_bundles[boundary]
            left_levels = max(0, len(np.unique(left.free_time_s)) - 1)
            right_levels = max(0, len(np.unique(right.free_time_s)) - 1)
            time_jump = abs(
                np.log(
                    max(float(np.min(right.free_time_s)), 1.0e-300)
                    / max(float(np.min(left.free_time_s)), 1.0e-300)
                )
            )
            split_scores.append(
                (
                    float(left_levels + right_levels + time_jump),
                    boundary,
                )
            )
        split_score, split_boundary = (
            max(split_scores, key=lambda value: (value[0], -value[1]))
            if split_scores
            else (0.0, -1)
        )
        split_evidence: dict[str, object] = {
            "split_evaluation_status": (
                "NOT_APPLICABLE_RUNTIME_SELECTED"
                if chosen is not None
                else "NOT_APPLICABLE_SINGLE_PRIMITIVE_RETAIN"
            ),
            "full_chain_best_runtime_class": "",
            "one_split_child_runtime_classes_json": "[]",
            "full_chain_best_weighted_rms_rel_cost_error": np.nan,
            "full_chain_best_max_rel_cost_error": np.nan,
            "one_split_weighted_rms_rel_cost_error": np.nan,
            "one_split_max_rel_cost_error": np.nan,
            "no_contraction_primitive_runtime_classes_json": "[]",
            "no_contraction_weighted_rms_rel_cost_error": np.nan,
            "no_contraction_max_rel_cost_error": np.nan,
            "full_chain_max_rel_potential_error": np.nan,
            "one_split_max_rel_potential_error": np.nan,
            "no_contraction_max_rel_potential_error": np.nan,
            "full_chain_cost_runtime_median_ns": np.nan,
            "one_split_cost_runtime_median_ns": np.nan,
            "one_split_runtime_to_full_ratio": np.nan,
            "no_contraction_cost_runtime_median_ns": np.nan,
            "no_contraction_runtime_to_full_ratio": np.nan,
            "full_contraction_retained_link_count": 1,
            "one_split_retained_link_count": (
                2 if len(experiment.primitive_bundles) > 1 else 1
            ),
            "no_contraction_retained_link_count": len(
                experiment.primitive_bundles
            ),
            "full_contraction_removed_internal_node_count": max(
                0, len(experiment.primitive_bundles) - 1
            ),
            "one_split_removed_internal_node_count": max(
                0, len(experiment.primitive_bundles) - 2
            ),
            "no_contraction_runtime_status": (
                "NOT_EVALUATED"
            ),
            "split_error_reduction_per_additional_retained_link": np.nan,
            "split_error_reduction_per_additional_retained_node": np.nan,
            "split_charge_accounting_rule": (
                "LEFT_AND_RIGHT_INTERNAL_CHARGES_STAY_IN_CHILDREN;SPLIT_BOUNDARY_CHARGE_RETAINED_EXTERNALLY_ONCE"
            ),
            "split_boundary_internal_component_id": "",
            "split_boundary_external_charge_s": np.nan,
        }
        exact_cost = exact_cost + experiment.internal_charge_s
        exact_potential = exact_potential + experiment.internal_charge_s*flow
        split_tradeoff_evaluated = False
        if chosen is None and len(experiment.primitive_bundles) > 1:
            if len(experiment.internal_component_charge_s) != len(
                experiment.primitive_bundles
            ) - 1 or len(experiment.internal_component_ids) != len(
                experiment.internal_component_charge_s
            ):
                raise RuntimeError(
                    "A splittable hard case lacks ordered per-internal-component charge provenance"
                )
            left_bundles = experiment.primitive_bundles[:split_boundary]
            right_bundles = experiment.primitive_bundles[split_boundary:]
            left_charge = float(
                sum(experiment.internal_component_charge_s[: split_boundary - 1])
            )
            boundary_charge = float(
                experiment.internal_component_charge_s[split_boundary - 1]
            )
            right_charge = float(
                sum(experiment.internal_component_charge_s[split_boundary:])
            )
            if not np.isclose(
                left_charge + boundary_charge + right_charge,
                experiment.internal_charge_s,
                rtol=1.0e-13,
                atol=1.0e-10,
            ):
                raise RuntimeError("Split charge partition does not close to macro charge")
            full_name, full_function, full_objective = _diagnostic_best_runtime(
                experiment.primitive_bundles,
                internal_charge_s=experiment.internal_charge_s,
                flow=flow,
                exact_cost=exact_cost,
                weights=weights,
                domain=domain,
            )
            left_results = [
                exact_series_oracle(
                    left_bundles, float(q), internal_charge_s=left_charge
                )
                for q in flow
            ]
            right_results = [
                exact_series_oracle(
                    right_bundles, float(q), internal_charge_s=right_charge
                )
                for q in flow
            ]
            left_cost = np.asarray(
                [value.marginal_cost_s for value in left_results],
                dtype=np.float64,
            )
            right_cost = np.asarray(
                [value.marginal_cost_s for value in right_results],
                dtype=np.float64,
            )
            left_name, left_function, _ = _diagnostic_best_runtime(
                left_bundles,
                internal_charge_s=left_charge,
                flow=flow,
                exact_cost=left_cost,
                weights=weights,
                domain=domain,
            )
            right_name, right_function, _ = _diagnostic_best_runtime(
                right_bundles,
                internal_charge_s=right_charge,
                flow=flow,
                exact_cost=right_cost,
                weights=weights,
                domain=domain,
            )
            primitive_names: list[str] = []
            primitive_functions: list[RuntimeFunction] = []
            for primitive in experiment.primitive_bundles:
                primitive_results = [
                    exact_bundle_oracle(primitive, float(q)) for q in flow
                ]
                primitive_cost = np.asarray(
                    [value.marginal_cost_s for value in primitive_results],
                    dtype=np.float64,
                )
                primitive_name, primitive_function, _ = _diagnostic_best_runtime(
                    (primitive,),
                    internal_charge_s=0.0,
                    flow=flow,
                    exact_cost=primitive_cost,
                    weights=weights,
                    domain=domain,
                )
                primitive_names.append(primitive_name)
                primitive_functions.append(primitive_function)
            denominator = np.maximum(exact_cost-experiment.internal_charge_s, 1.0)
            full_prediction = full_function.cost(flow)
            split_prediction = (
                left_function.cost(flow)
                + right_function.cost(flow)
                + boundary_charge
            )
            full_relative = np.abs(full_prediction - exact_cost) / denominator
            split_relative = np.abs(split_prediction - exact_cost) / denominator
            no_contraction_prediction = np.full_like(
                flow, experiment.internal_charge_s
            )
            for primitive_function in primitive_functions:
                no_contraction_prediction += primitive_function.cost(flow)
            no_contraction_relative = np.abs(
                no_contraction_prediction - exact_cost
            ) / denominator
            split_objective = float(
                np.sqrt(
                    np.sum(weights * np.square(split_relative))
                    / np.sum(weights)
                )
            )
            no_contraction_objective = float(
                np.sqrt(
                    np.sum(weights * np.square(no_contraction_relative))
                    / np.sum(weights)
                )
            )
            exact_potential_denominator = np.maximum(
                np.abs(exact_potential), 1.0
            )
            full_potential_relative = np.abs(
                full_function.potential(flow) - exact_potential
            ) / exact_potential_denominator
            split_potential = (
                left_function.potential(flow)
                + right_function.potential(flow)
                + boundary_charge * flow
            )
            split_potential_relative = np.abs(
                split_potential - exact_potential
            ) / exact_potential_denominator
            no_contraction_potential = experiment.internal_charge_s * flow
            for primitive_function in primitive_functions:
                no_contraction_potential += primitive_function.potential(flow)
            no_contraction_potential_relative = np.abs(
                no_contraction_potential - exact_potential
            ) / exact_potential_denominator
            full_runtime_ns = _median_cost_runtime_ns(
                (full_function,), additive_charge_s=0.0, domain=domain
            )
            split_runtime_ns = _median_cost_runtime_ns(
                (left_function, right_function),
                additive_charge_s=boundary_charge,
                domain=domain,
            )
            no_contraction_runtime_ns = _median_cost_runtime_ns(
                tuple(primitive_functions),
                additive_charge_s=experiment.internal_charge_s,
                domain=domain,
            )
            full_max = float(full_relative.max(initial=0.0))
            split_max = float(split_relative.max(initial=0.0))
            reduction = full_max - split_max
            split_evidence.update(
                {
                    "split_evaluation_status": "MEASURED_FULL_VS_ONE_SPLIT_VS_RETAIN",
                    "full_chain_best_runtime_class": full_name,
                    "one_split_child_runtime_classes_json": json.dumps(
                        [left_name, right_name], separators=(",", ":")
                    ),
                    "full_chain_best_weighted_rms_rel_cost_error": full_objective,
                    "full_chain_best_max_rel_cost_error": full_max,
                    "one_split_weighted_rms_rel_cost_error": split_objective,
                    "one_split_max_rel_cost_error": split_max,
                    "no_contraction_primitive_runtime_classes_json": json.dumps(
                        primitive_names, separators=(",", ":")
                    ),
                    "no_contraction_weighted_rms_rel_cost_error": no_contraction_objective,
                    "no_contraction_max_rel_cost_error": float(
                        no_contraction_relative.max(initial=0.0)
                    ),
                    "full_chain_max_rel_potential_error": float(
                        full_potential_relative.max(initial=0.0)
                    ),
                    "one_split_max_rel_potential_error": float(
                        split_potential_relative.max(initial=0.0)
                    ),
                    "no_contraction_max_rel_potential_error": float(
                        no_contraction_potential_relative.max(initial=0.0)
                    ),
                    "full_chain_cost_runtime_median_ns": full_runtime_ns,
                    "one_split_cost_runtime_median_ns": split_runtime_ns,
                    "one_split_runtime_to_full_ratio": split_runtime_ns
                    / max(full_runtime_ns, np.finfo(np.float64).tiny),
                    "no_contraction_cost_runtime_median_ns": no_contraction_runtime_ns,
                    "no_contraction_runtime_to_full_ratio": no_contraction_runtime_ns
                    / max(full_runtime_ns, np.finfo(np.float64).tiny),
                    "no_contraction_runtime_status": "MEASURED_ROOT_FREE_BOUNDED_PRIMITIVE_RUNTIME",
                    "split_error_reduction_per_additional_retained_link": reduction,
                    "split_error_reduction_per_additional_retained_node": reduction,
                    "split_boundary_internal_component_id": experiment.internal_component_ids[
                        split_boundary - 1
                    ],
                    "split_boundary_external_charge_s": boundary_charge,
                }
            )
            split_tradeoff_evaluated = True
        domain_rows.append(
            {
                "canonical_macro_signature": experiment.canonical_signature,
                "suite_occurrence_id": experiment.occurrence_id,
                "sample_ordinal": ordinal,
                "certified_grid_seconds": grid_certified_seconds,
                "total_oracle_fit_diagnostic_seconds": time.perf_counter() - object_started,
                "primitive_parameter_class": "EXACT_SERIES" if exact_terms is not None else "HETEROGENEOUS_SERIES",
                "primitive_link_count": experiment.chain_length,
                "selection_class": status,
                "provisional_action": (
                    "SELECT_ROOT_FREE_RUNTIME"
                    if chosen is not None
                    else "RETAIN_ORIGINAL_PRIMITIVE_LINKS"
                ),
                "provisional_split_after_primitive_index": split_boundary,
                "provisional_split_structural_score": split_score,
                "provisional_split_rule": (
                    "largest adjacent count of staggered physical activation levels "
                    "plus absolute log minimum-FFTT jump; diagnostic only"
                ),
                "split_runtime_tradeoff_evaluated": split_tradeoff_evaluated,
                "high_flow_inside_pilot_domain": bool(experiment.high_flow <= domain),
                **split_evidence,
                **metadata,
            }
        )
    return pd.DataFrame(oracle_rows), pd.DataFrame(fit_rows), pd.DataFrame(domain_rows)
