"""Bounded benchmarks of the actual Phase 5 root-free runtime classes."""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import psutil

from .bundle_oracle import (
    exact_power_terms,
    exact_series_oracle,
    get_exact_root_call_count,
    reset_exact_root_call_count,
)
from .runtime_functions import HingeRuntime, OneTermRuntime, PowerSumRuntime, RuntimeFunction


def process_tree_rss_bytes() -> int:
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return int(total)


def _times(
    function: Callable[[], np.ndarray], warmups: int, repetitions: int
) -> tuple[np.ndarray, int]:
    peak = process_tree_rss_bytes()
    for _ in range(warmups):
        value = function()
        if value.shape == () or not np.all(np.isfinite(value)):
            raise RuntimeError("Kernel benchmark produced nonfinite output")
        peak = max(peak, process_tree_rss_bytes())
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        started = time.perf_counter_ns()
        value = function()
        samples[index] = (time.perf_counter_ns() - started) * 1.0e-9
        if value.shape == () or not np.isfinite(value[-1]):
            raise RuntimeError("Kernel benchmark output was not materialized")
        peak = max(peak, process_tree_rss_bytes())
    return samples, peak


def _percentiles(samples: np.ndarray) -> tuple[float, float]:
    return float(np.median(samples)), float(np.quantile(samples, 0.95))


def _noise_band(samples: np.ndarray) -> tuple[float, float]:
    median = float(np.median(samples))
    return float(np.quantile(samples, 0.05)), float(np.median(np.abs(samples - median)))


def _offline_query_flow(value: object) -> float:
    """Use only calibration-side flows for the offline timing query."""

    return max(
        float(getattr(value, "low_flow")),
        float(getattr(value, "low_aon_flow")),
        1.0,
    )


def _offline_oracle_sample_for_suite(
    suite_id: str,
    offline_experiments: Sequence[object],
    chain_candidates: pd.DataFrame,
    *,
    maximum_objects: int = 64,
) -> tuple[list[object], np.ndarray, dict[str, object]]:
    """Resolve suite coverage through selected canonical-signature occurrences.

    The canonical pilot sample owns one representative experiment per signature,
    and deterministic cross-suite deduplication intentionally prefers s03.  A
    representative's ``suite_id`` therefore is provenance, not the set of suites
    in which that selected signature occurs.  Keep that provenance intact while
    using the immutable chain-occurrence table to establish suite coverage.
    """

    if maximum_objects <= 0:
        raise ValueError("Offline oracle sample size must be positive")
    required = {
        "suite_short_id",
        "chain_id",
        "canonical_signature",
        "sample_included",
        "protected_or_blocked",
    }
    missing = sorted(required - set(chain_candidates.columns))
    if missing:
        raise RuntimeError(f"Offline oracle suite coverage lacks columns: {missing}")

    by_signature: dict[str, object] = {}
    for value in offline_experiments:
        signature = str(getattr(value, "canonical_signature"))
        if signature in by_signature:
            raise RuntimeError(
                f"Offline canonical sample contains duplicate signature: {signature}"
            )
        by_signature[signature] = value
    if not by_signature:
        raise RuntimeError("Offline canonical sample is empty")

    target = chain_candidates.loc[
        chain_candidates["suite_short_id"].astype(str).eq(str(suite_id))
        & chain_candidates["sample_included"].astype(bool)
        & ~chain_candidates["protected_or_blocked"].astype(bool)
        & chain_candidates["canonical_signature"].notna(),
        ["canonical_signature", "chain_id"],
    ].copy()
    target["canonical_signature"] = target["canonical_signature"].astype(str)
    target["chain_id"] = target["chain_id"].astype(str)
    target = target.sort_values(
        ["canonical_signature", "chain_id"], kind="mergesort"
    )
    if target.empty:
        raise RuntimeError(f"{suite_id}: no selected unblocked canonical occurrence")
    target_occurrences = {
        str(signature): list(group["chain_id"].astype(str))
        for signature, group in target.groupby("canonical_signature", sort=True)
    }
    missing_signatures = sorted(set(target_occurrences) - set(by_signature))
    if missing_signatures:
        raise RuntimeError(
            f"{suite_id}: selected occurrence is absent from canonical sample: "
            f"{missing_signatures[:3]}"
        )

    suite_objects = [by_signature[value] for value in sorted(target_occurrences)]
    root_objects = [
        value
        for value in suite_objects
        if any(
            exact_power_terms(specification) is None
            for specification in getattr(value, "primitive_bundles")
        )
    ]
    observed_root_object = None
    observed_root_calls = 0
    observed_query = 0.0
    for value in root_objects:
        query = _offline_query_flow(value)
        reset_exact_root_call_count()
        exact_series_oracle(
            getattr(value, "primitive_bundles"),
            query,
            internal_charge_s=float(getattr(value, "internal_charge_s")),
        )
        root_calls = get_exact_root_call_count()
        if root_calls > 0:
            observed_root_object = value
            observed_root_calls = int(root_calls)
            observed_query = float(query)
            break
    if observed_root_object is None:
        raise RuntimeError(
            f"{suite_id}: bounded canonical sample contains no observed exact-root case"
        )

    observed_id = id(observed_root_object)
    root_object_ids = {id(value) for value in root_objects}
    remaining_objects = [
        value for value in suite_objects if id(value) not in root_object_ids
    ]
    ordered_objects = [
        observed_root_object,
        *(value for value in root_objects if id(value) != observed_id),
        *remaining_objects,
    ]
    offline_objects = ordered_objects[:maximum_objects]
    sample_q = np.asarray(
        [_offline_query_flow(value) for value in offline_objects], dtype=np.float64
    )
    observed_signature = str(
        getattr(observed_root_object, "canonical_signature")
    )
    provenance: dict[str, object] = {
        "selection_policy": (
            "SELECTED_CANONICAL_SIGNATURE_WITH_UNBLOCKED_TARGET_SUITE_OCCURRENCE"
        ),
        "available_target_suite_signature_count": len(suite_objects),
        "available_root_candidate_count": len(root_objects),
        "benchmarked_object_count": len(offline_objects),
        "root_source_suite_id": str(getattr(observed_root_object, "suite_id")),
        "root_source_occurrence_id": str(
            getattr(observed_root_object, "occurrence_id")
        ),
        "root_target_suite_id": str(suite_id),
        "root_target_occurrence_id": target_occurrences[observed_signature][0],
        "root_target_occurrence_ids": target_occurrences[observed_signature],
        "root_target_occurrence_count": len(
            target_occurrences[observed_signature]
        ),
        "root_canonical_signature": observed_signature,
        "root_query_flow": observed_query,
        "root_query_basis": (
            "max(source_representative_low_current_flow,"
            "source_representative_low_aon_flow,1.0)"
        ),
        "root_probe_call_count": observed_root_calls,
        "benchmarked_object_bindings": [
            {
                "canonical_signature": str(
                    getattr(value, "canonical_signature")
                ),
                "source_suite_id": str(getattr(value, "suite_id")),
                "source_occurrence_id": str(getattr(value, "occurrence_id")),
                "target_suite_id": str(suite_id),
                "target_occurrence_ids": target_occurrences[
                    str(getattr(value, "canonical_signature"))
                ],
                "query_flow": float(query),
            }
            for value, query in zip(offline_objects, sample_q, strict=True)
        ],
        "source_representative_relabelled": False,
        "selected_signature_membership_changed": False,
        "heldout_high_used_for_selection": False,
    }
    reset_exact_root_call_count()
    return offline_objects, sample_q, provenance


@dataclass(frozen=True, slots=True)
class _BprRuntime:
    """Vectorized accepted physical BPR baseline in seconds."""

    free_time_s: np.ndarray
    used_capacity: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray

    def cost(self, flow: np.ndarray | float) -> np.ndarray:
        q = np.asarray(flow, dtype=np.float64)
        return self.free_time_s * (
            1.0 + self.alpha * np.power(q / self.used_capacity, self.beta)
        )

    def potential(self, flow: np.ndarray | float) -> np.ndarray:
        q = np.asarray(flow, dtype=np.float64)
        return self.free_time_s * (
            q
            + self.alpha
            * self.used_capacity
            * np.power(q / self.used_capacity, self.beta + 1.0)
            / (self.beta + 1.0)
        )

    def derivative(self, flow: np.ndarray | float) -> np.ndarray:
        q = np.asarray(flow, dtype=np.float64)
        return (
            self.free_time_s
            * self.alpha
            * self.beta
            * np.power(q / self.used_capacity, self.beta - 1.0)
            / self.used_capacity
        )


def _runtime_from_row(row: pd.Series) -> RuntimeFunction:
    parameter = json.loads(str(row["parameter_json"]))
    if row.get("parameter_cost_role") == "DYNAMIC_ONLY":
        parameter["intercept_s"] += float(row["internal_charge_s"])
    runtime_class = str(row["runtime_class"])
    if runtime_class == "EXACT_CLOSED_FORM":
        return PowerSumRuntime(
            parameter["intercept_s"],
            np.asarray(parameter["exponents"], dtype=np.float64),
            np.asarray(parameter["coefficients"], dtype=np.float64),
        )
    if runtime_class.startswith("FIT_ONE_TERM"):
        return OneTermRuntime(
            parameter["intercept_s"],
            parameter["scale_flow"],
            parameter["exponent"],
            parameter["eta_s"],
        )
    if runtime_class.startswith("MONOTONE_HINGE_"):
        return HingeRuntime(
            parameter["intercept_s"],
            np.asarray(json.loads(str(row["knot_json"])), dtype=np.float64),
            np.asarray(parameter["segment_slopes_s_per_flow"], dtype=np.float64),
        )
    raise ValueError(f"Unsupported serialized runtime class: {runtime_class}")


def _stable_rows(frame: pd.DataFrame, maximum: int = 2048) -> pd.DataFrame:
    if frame.empty:
        return frame
    ordered = frame.sort_values(
        ["canonical_macro_signature", "variant_id"], kind="mergesort"
    )
    if len(ordered) <= maximum:
        return ordered
    indexes = np.linspace(0, len(ordered) - 1, maximum).round().astype(np.int64)
    return ordered.iloc[indexes].drop_duplicates(
        ["canonical_macro_signature", "variant_id"]
    )


@dataclass(frozen=True, slots=True)
class _PackedPowerSum:
    intercept_s: np.ndarray
    exponents: np.ndarray
    coefficients: np.ndarray

    def cost(self, flow: np.ndarray | float) -> np.ndarray:
        q = np.asarray(flow, dtype=np.float64)
        result = self.intercept_s.copy()
        for column in range(self.exponents.shape[1]):
            result += self.coefficients[:, column] * np.power(
                q, self.exponents[:, column]
            )
        return result

    def potential(self, flow: np.ndarray | float) -> np.ndarray:
        q = np.asarray(flow, dtype=np.float64)
        result = self.intercept_s * q
        for column in range(self.exponents.shape[1]):
            exponent = self.exponents[:, column]
            result += (
                self.coefficients[:, column]
                * np.power(q, exponent + 1.0)
                / (exponent + 1.0)
            )
        return result

    def derivative(self, flow: np.ndarray | float) -> np.ndarray:
        q = np.asarray(flow, dtype=np.float64)
        result = np.zeros_like(q)
        for column in range(self.exponents.shape[1]):
            exponent = self.exponents[:, column]
            result += (
                self.coefficients[:, column]
                * exponent
                * np.power(q, exponent - 1.0)
            )
        return result


@dataclass(frozen=True, slots=True)
class _PackedOneTerm:
    intercept_s: np.ndarray
    scale_flow: np.ndarray
    exponent: np.ndarray
    eta_s: np.ndarray

    def cost(self, flow: np.ndarray | float) -> np.ndarray:
        q = np.asarray(flow, dtype=np.float64)
        return self.intercept_s + self.eta_s * np.power(
            q / self.scale_flow, self.exponent
        )

    def potential(self, flow: np.ndarray | float) -> np.ndarray:
        q = np.asarray(flow, dtype=np.float64)
        return self.intercept_s * q + (
            self.eta_s
            * self.scale_flow
            * np.power(q / self.scale_flow, self.exponent + 1.0)
            / (self.exponent + 1.0)
        )

    def derivative(self, flow: np.ndarray | float) -> np.ndarray:
        q = np.asarray(flow, dtype=np.float64)
        return (
            self.eta_s
            * self.exponent
            * np.power(q / self.scale_flow, self.exponent - 1.0)
            / self.scale_flow
        )


@dataclass(frozen=True, slots=True)
class _PackedGlobalBeta4OneTerm:
    """One record per link with the globally fixed beta=4 kept out of storage."""

    intercept_s: np.ndarray
    scale_flow: np.ndarray
    eta_s: np.ndarray

    def cost(self, flow: np.ndarray | float) -> np.ndarray:
        q = np.asarray(flow, dtype=np.float64)
        normalized = q / self.scale_flow
        square = normalized * normalized
        return self.intercept_s + self.eta_s * square * square

    def potential(self, flow: np.ndarray | float) -> np.ndarray:
        q = np.asarray(flow, dtype=np.float64)
        normalized = q / self.scale_flow
        square = normalized * normalized
        fourth = square * square
        return self.intercept_s * q + self.eta_s * self.scale_flow * fourth * normalized / 5.0

    def derivative(self, flow: np.ndarray | float) -> np.ndarray:
        q = np.asarray(flow, dtype=np.float64)
        normalized = q / self.scale_flow
        return 4.0 * self.eta_s * normalized * normalized * normalized / self.scale_flow


@dataclass(frozen=True, slots=True)
class _PackedHinge:
    intercept_s: np.ndarray
    knots_flow: np.ndarray
    slopes_s_per_flow: np.ndarray

    def cost(self, flow: np.ndarray | float) -> np.ndarray:
        q = np.asarray(flow, dtype=np.float64)
        result = self.intercept_s.copy()
        for segment in range(self.slopes_s_per_flow.shape[1]):
            start = 0.0 if segment == 0 else self.knots_flow[:, segment - 1]
            positive = np.maximum(q - start, 0.0)
            if segment < self.knots_flow.shape[1]:
                positive = np.minimum(
                    positive, self.knots_flow[:, segment] - start
                )
            result += self.slopes_s_per_flow[:, segment] * positive
        return result

    def potential(self, flow: np.ndarray | float) -> np.ndarray:
        q = np.asarray(flow, dtype=np.float64)
        result = self.intercept_s * q
        for segment in range(self.slopes_s_per_flow.shape[1]):
            start = 0.0 if segment == 0 else self.knots_flow[:, segment - 1]
            positive = np.maximum(q - start, 0.0)
            if segment < self.knots_flow.shape[1]:
                end = self.knots_flow[:, segment]
                length = np.minimum(positive, end - start)
                basis = length * (positive - 0.5 * length)
            else:
                basis = 0.5 * np.square(positive)
            result += self.slopes_s_per_flow[:, segment] * basis
        return result

    def derivative(self, flow: np.ndarray | float) -> np.ndarray:
        q = np.asarray(flow, dtype=np.float64)
        result = self.slopes_s_per_flow[:, 0].copy()
        for knot in range(self.knots_flow.shape[1]):
            result = np.where(
                q >= self.knots_flow[:, knot],
                self.slopes_s_per_flow[:, knot + 1],
                result,
            )
        return result


def _flow_for_domain(count: int, domain: float, ordinal: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    maximum = max(float(domain), 1.0)
    index = np.arange(count, dtype=np.float64)
    # Exercise the fitted domain deterministically without fitting to the heldout state.
    return maximum * (0.025 + 0.95 * ((index + 37.0 * ordinal) % 1024.0) / 1023.0)


@dataclass(slots=True)
class _ExecutionPlan:
    chunks: list[tuple[slice, RuntimeFunction, np.ndarray]]
    size: int
    scalar_roundtrip_max_abs_error: float = 0.0

    def operation(self, name: str) -> Callable[[], np.ndarray]:
        def evaluate() -> np.ndarray:
            result = np.empty(self.size, dtype=np.float64)
            for target, function, flow in self.chunks:
                result[target] = getattr(function, name)(flow)
            return result

        return evaluate

    def dispatch_only(self) -> np.ndarray:
        result = np.empty(self.size, dtype=np.float64)
        for target, _, flow in self.chunks:
            result[target] = flow
        return result


def _packed_runtime(
    functions: Sequence[RuntimeFunction],
    source: np.ndarray,
    *,
    globally_fixed_beta4: bool = False,
) -> RuntimeFunction:
    if not functions or source.ndim != 1:
        raise ValueError("Packed runtime requires functions and a one-dimensional index")
    first = functions[0]
    if isinstance(first, PowerSumRuntime):
        values = [value for value in functions if isinstance(value, PowerSumRuntime)]
        if len(values) != len(functions):
            raise TypeError("Mixed runtime classes cannot share one packed class plan")
        term_count = max(len(value.exponents) for value in values)
        intercept = np.asarray([value.intercept_s for value in values])[source]
        exponents = np.ones((len(values), term_count), dtype=np.float64)
        coefficients = np.zeros((len(values), term_count), dtype=np.float64)
        for row, value in enumerate(values):
            exponents[row, : len(value.exponents)] = value.exponents
            coefficients[row, : len(value.coefficients)] = value.coefficients
        return _PackedPowerSum(intercept, exponents[source], coefficients[source])
    if isinstance(first, OneTermRuntime):
        values = [value for value in functions if isinstance(value, OneTermRuntime)]
        if len(values) != len(functions):
            raise TypeError("Mixed runtime classes cannot share one packed class plan")
        intercept = np.asarray([value.intercept_s for value in values])[source]
        scale = np.asarray([value.scale_flow for value in values])[source]
        eta = np.asarray([value.eta_s for value in values])[source]
        if globally_fixed_beta4:
            if any(value.exponent != 4.0 for value in values):
                raise ValueError("Global-beta-4 plan contains a non-beta-4 fit")
            return _PackedGlobalBeta4OneTerm(intercept, scale, eta)
        return _PackedOneTerm(
            intercept,
            scale,
            np.asarray([value.exponent for value in values])[source],
            eta,
        )
    if isinstance(first, HingeRuntime):
        values = [value for value in functions if isinstance(value, HingeRuntime)]
        if len(values) != len(functions):
            raise TypeError("Mixed runtime classes cannot share one packed class plan")
        segment_counts = {len(value.segment_slopes_s_per_flow) for value in values}
        if len(segment_counts) != 1:
            raise ValueError("One packed hinge plan requires a common segment count")
        return _PackedHinge(
            np.asarray([value.intercept_s for value in values])[source],
            np.stack([value.knots_flow for value in values], axis=0)[source],
            np.stack([value.segment_slopes_s_per_flow for value in values], axis=0)[
                source
            ],
        )
    raise TypeError(type(first))


def _scalar_roundtrip_error(
    packed: RuntimeFunction,
    functions: Sequence[RuntimeFunction],
    source: np.ndarray,
    flow: np.ndarray,
) -> float:
    if len(flow) == 0:
        return 0.0
    indexes = np.linspace(0, len(flow) - 1, min(32, len(flow))).round().astype(np.int64)
    maximum = 0.0
    for operation in ("cost", "potential", "derivative"):
        packed_value = getattr(packed, operation)(flow)[indexes]
        scalar_value = np.asarray(
            [
                float(getattr(functions[int(source[index])], operation)(flow[index]))
                for index in indexes
            ],
            dtype=np.float64,
        )
        error = float(np.max(np.abs(packed_value - scalar_value)))
        scale = max(
            1.0,
            float(np.max(np.abs(packed_value))),
            float(np.max(np.abs(scalar_value))),
        )
        if error > 1.0e-12 + 1.0e-10 * scale:
            raise RuntimeError(
                f"Packed {operation} differs from serialized scalar runtime: "
                f"error={error}, scale={scale}"
            )
        maximum = max(maximum, error)
    return maximum


def _allocate_counts(total: int, weights: Sequence[int]) -> list[int]:
    weight = np.asarray(weights, dtype=np.float64)
    if total < 0 or len(weight) == 0 or np.any(weight < 0.0) or weight.sum() <= 0.0:
        raise ValueError("Invalid benchmark allocation")
    raw = total * weight / weight.sum()
    result = np.floor(raw).astype(np.int64)
    remainder = total - int(result.sum())
    if remainder:
        order = np.argsort(-(raw - result), kind="stable")
        result[order[:remainder]] += 1
    return result.astype(int).tolist()


def _bpr_plan(
    size: int,
    bpr_parameters: dict[str, np.ndarray],
) -> _ExecutionPlan:
    source_size = len(bpr_parameters["free_time_s"])
    if source_size == 0:
        raise ValueError("BPR benchmark source is empty")
    source = np.arange(size, dtype=np.int64) % source_size
    function = _BprRuntime(
        np.asarray(bpr_parameters["free_time_s"], dtype=np.float64)[source],
        np.asarray(bpr_parameters["used_capacity"], dtype=np.float64)[source],
        np.asarray(bpr_parameters["alpha"], dtype=np.float64)[source],
        np.asarray(bpr_parameters["beta"], dtype=np.float64)[source],
    )
    if "flow" in bpr_parameters:
        flow = np.asarray(bpr_parameters["flow"], dtype=np.float64)[source]
    else:
        flow = 0.75 * function.used_capacity
    return _ExecutionPlan([(slice(0, size), function, flow)], size)


def _fitted_plan(
    size: int, rows: pd.DataFrame, *, kernel_class: str
) -> _ExecutionPlan:
    selected = _stable_rows(rows)
    if selected.empty:
        raise ValueError("No serialized fitted rows available for benchmark class")
    functions = [
        _runtime_from_row(pd.Series(row._asdict()))
        for row in selected.itertuples(index=False)
    ]
    domains = selected["flow_domain_max"].to_numpy(dtype=np.float64)
    source = np.arange(size, dtype=np.int64) % len(functions)
    index = np.arange(size, dtype=np.float64)
    flow = np.maximum(domains[source], 1.0) * (
        0.025 + 0.95 * ((index + 37.0 * source) % 1024.0) / 1023.0
    )
    packed = _packed_runtime(
        functions,
        source,
        globally_fixed_beta4=(
            kernel_class == "FIT_ONE_TERM_GLOBAL_EXPONENT"
        ),
    )
    error = _scalar_roundtrip_error(packed, functions, source, flow)
    return _ExecutionPlan([(slice(0, size), packed, flow)], size, error)


def _class_rows(fit_results: pd.DataFrame, kernel_class: str) -> pd.DataFrame:
    runtime = fit_results["runtime_class"].astype(str)
    rows = fit_results.loc[runtime.eq(kernel_class)]
    if kernel_class == "FIT_ONE_TERM_DISCRETE_EXPONENT" and len(rows):
        rows = (
            rows.sort_values(
                [
                    "canonical_macro_signature",
                    "weighted_rms_rel_cost_error",
                    "variant_id",
                ],
                kind="mergesort",
            )
            .drop_duplicates("canonical_macro_signature")
        )
    return rows


def _mixed_projection_counts(
    suite_id: str,
    projected: int,
    fit_results: pd.DataFrame,
    chain_candidates: pd.DataFrame,
) -> dict[str, int]:
    suite = chain_candidates.loc[
        chain_candidates["suite_short_id"].astype(str).eq(suite_id)
        & ~chain_candidates["protected_or_blocked"].astype(bool)
    ]
    selected = fit_results.loc[
        fit_results["recommended_status"].astype(str).str.startswith("CASCADE_SELECTED")
    ][["canonical_macro_signature", "runtime_class"]]
    if "runtime_resolution" in suite.columns:
        resolved = suite.loc[
            suite["runtime_resolution"].astype(str).eq(
                "SELECTED_ROOT_FREE_RUNTIME"
            )
        ]
    else:
        resolved = suite.loc[suite["sample_included"].astype(bool)]
    joined = resolved[["canonical_signature"]].merge(
        selected,
        left_on="canonical_signature",
        right_on="canonical_macro_signature",
        how="left",
        validate="many_to_one",
    )
    if len(joined):
        if joined["runtime_class"].isna().any():
            raise RuntimeError(
                f"{suite_id}: selected runtime resolution lacks a serialized fit"
            )
        labels = joined["runtime_class"]
        counts = {
            str(key): int(value)
            for key, value in labels.value_counts().sort_index().items()
        }
    else:
        counts = {}
    selected_count = sum(counts.values())
    if selected_count > projected:
        raise RuntimeError(f"{suite_id}: selected macro count exceeds projection")
    counts["ORDINARY_BPR_BASELINE"] = projected - selected_count
    return {key: int(value) for key, value in counts.items() if value > 0}


def _mixed_plan(
    suite_id: str,
    projected: int,
    fit_results: pd.DataFrame,
    chain_candidates: pd.DataFrame,
    bpr_parameters: dict[str, np.ndarray],
) -> tuple[_ExecutionPlan, dict[str, int]]:
    counts = _mixed_projection_counts(
        suite_id, projected, fit_results, chain_candidates
    )
    chunks: list[tuple[slice, RuntimeFunction, np.ndarray]] = []
    roundtrip_error = 0.0
    start = 0
    for kernel_class in (
        "ORDINARY_BPR_BASELINE",
        "EXACT_CLOSED_FORM",
        "FIT_ONE_TERM_GLOBAL_EXPONENT",
        "FIT_ONE_TERM_DISCRETE_EXPONENT",
        "MONOTONE_HINGE_4",
        "MONOTONE_HINGE_8",
    ):
        count = counts.get(kernel_class, 0)
        if not count:
            continue
        if kernel_class == "ORDINARY_BPR_BASELINE":
            local = _bpr_plan(count, bpr_parameters)
        else:
            selected_fits = fit_results.loc[
                fit_results["recommended_status"]
                .astype(str)
                .str.startswith("CASCADE_SELECTED")
            ]
            rows = _class_rows(selected_fits, kernel_class)
            # Prefer suite-observed signatures while retaining a global fallback for
            # a class represented only in another suite's deterministic pilot sample.
            signatures = set(
                chain_candidates.loc[
                    chain_candidates["suite_short_id"].astype(str).eq(suite_id),
                    "canonical_signature",
                ].astype(str)
            )
            local_rows = rows.loc[
                rows["canonical_macro_signature"].astype(str).isin(signatures)
            ]
            local = _fitted_plan(
                count,
                local_rows if len(local_rows) else rows,
                kernel_class=kernel_class,
            )
        roundtrip_error = max(
            roundtrip_error, local.scalar_roundtrip_max_abs_error
        )
        for target, function, flow in local.chunks:
            chunks.append(
                (
                    slice(start + int(target.start), start + int(target.stop)),
                    function,
                    flow,
                )
            )
        start += count
    if start != projected:
        raise RuntimeError("Mixed benchmark projection does not close to link count")
    return _ExecutionPlan(chunks, projected, roundtrip_error), counts


def _benchmark_plan(
    *,
    suite_id: str,
    kernel_class: str,
    projected: int,
    fit_results: pd.DataFrame,
    chain_candidates: pd.DataFrame,
    bpr_parameters: dict[str, np.ndarray],
) -> tuple[_ExecutionPlan, dict[str, int]]:
    if kernel_class == "ORDINARY_BPR_BASELINE":
        return _bpr_plan(projected, bpr_parameters), {kernel_class: projected}
    if kernel_class == "PROJECTED_MIXED_CLASS":
        return _mixed_plan(
            suite_id,
            projected,
            fit_results,
            chain_candidates,
            bpr_parameters,
        )
    rows = _class_rows(fit_results, kernel_class)
    if rows.empty and kernel_class == "EXACT_CLOSED_FORM":
        # A physical singleton is an exact closed-form Phase 5 member even if the
        # deterministic chain sample happens not to select an exact-series macro.
        t0 = float(np.asarray(bpr_parameters["free_time_s"])[0])
        capacity = float(np.asarray(bpr_parameters["used_capacity"])[0])
        alpha = float(np.asarray(bpr_parameters["alpha"])[0])
        beta = float(np.asarray(bpr_parameters["beta"])[0])
        fallback = pd.DataFrame(
            [
                {
                    "canonical_macro_signature": "ACCEPTED_PHYSICAL_SINGLETON",
                    "runtime_class": "EXACT_CLOSED_FORM",
                    "variant_id": "ACCEPTED_SINGLETON_POWER",
                    "flow_domain_max": max(capacity * 2.0, 1.0),
                    "parameter_json": json.dumps(
                        {
                            "intercept_s": t0,
                            "exponents": [beta],
                            "coefficients": [t0 * alpha / capacity**beta],
                        },
                        sort_keys=True,
                    ),
                    "knot_json": "[]",
                }
            ]
        )
        rows = fallback
    return _fitted_plan(
        projected, rows, kernel_class=kernel_class
    ), {kernel_class: projected}


def run_kernel_benchmarks(
    projected_counts: dict[str, int],
    *,
    fit_results: pd.DataFrame,
    chain_candidates: pd.DataFrame,
    bpr_parameters: dict[str, np.ndarray],
    offline_experiments: Sequence[object],
    warmups: int = 5,
    repetitions: int = 30,
    progress_guard: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Benchmark accepted inputs, reopened fits, and sampled real offline objects."""

    if warmups < 5 or repetitions < 30:
        raise ValueError("Official Phase 5 benchmark requires >=5 warmups and >=30 repetitions")
    rows: list[dict[str, object]] = []
    machine_metadata = json.dumps(
        {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
            "numpy": np.__version__,
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    classes = (
        "ORDINARY_BPR_BASELINE",
        "EXACT_CLOSED_FORM",
        "FIT_ONE_TERM_GLOBAL_EXPONENT",
        "FIT_ONE_TERM_DISCRETE_EXPONENT",
        "MONOTONE_HINGE_4",
        "MONOTONE_HINGE_8",
        "PROJECTED_MIXED_CLASS",
    )
    for suite_id, projected_value in sorted(projected_counts.items()):
        projected = int(projected_value)
        if projected <= 0:
            raise ValueError(f"{suite_id}: projected benchmark link count is empty")
        for kernel_class in classes:
            if progress_guard is not None:
                progress_guard(f"benchmark_{suite_id}_{kernel_class}")
            reset_exact_root_call_count()
            setup_started = time.perf_counter_ns()
            plan, class_counts = _benchmark_plan(
                suite_id=suite_id,
                kernel_class=kernel_class,
                projected=projected,
                fit_results=fit_results,
                chain_candidates=chain_candidates,
                bpr_parameters=bpr_parameters,
            )
            setup_seconds = (time.perf_counter_ns() - setup_started) * 1.0e-9
            dispatch_samples, dispatch_peak = _times(
                plan.dispatch_only, warmups, repetitions
            )
            dispatch_median, dispatch_p95 = _percentiles(dispatch_samples)
            timings: dict[str, tuple[float, float]] = {}
            noise: dict[str, tuple[float, float]] = {}
            peak = dispatch_peak
            for operation in ("cost", "potential", "derivative"):
                samples, operation_peak = _times(
                    plan.operation(operation), warmups, repetitions
                )
                timings[operation] = _percentiles(samples)
                noise[operation] = _noise_band(samples)
                peak = max(peak, operation_peak)
            root_calls = get_exact_root_call_count()
            if root_calls != 0:
                raise RuntimeError(
                    f"Production runtime benchmark invoked {root_calls} exact roots"
                )
            rows.append(
                {
                    "suite_short_id": suite_id,
                    "kernel_class": kernel_class,
                    "projected_link_count": projected,
                    "projection_reference_link_count": projected,
                    "projection_scope": "BACKBONE_LINK_FAMILY_ONLY",
                    "projected_class_counts_json": json.dumps(
                        class_counts, sort_keys=True, separators=(",", ":")
                    ),
                    "warmup_count": warmups,
                    "timed_repetition_count": repetitions,
                    "setup_seconds": setup_seconds,
                    "packed_parameter_record_count": projected,
                    "scalar_roundtrip_max_abs_error": plan.scalar_roundtrip_max_abs_error,
                    "median_dispatch_buffer_seconds": dispatch_median,
                    "p95_dispatch_buffer_seconds": dispatch_p95,
                    "median_cost_eval_seconds": timings["cost"][0],
                    "p05_cost_eval_seconds": noise["cost"][0],
                    "p95_cost_eval_seconds": timings["cost"][1],
                    "mad_cost_eval_seconds": noise["cost"][1],
                    "median_potential_eval_seconds": timings["potential"][0],
                    "p05_potential_eval_seconds": noise["potential"][0],
                    "p95_potential_eval_seconds": timings["potential"][1],
                    "mad_potential_eval_seconds": noise["potential"][1],
                    "median_derivative_eval_seconds": timings["derivative"][0],
                    "p05_derivative_eval_seconds": noise["derivative"][0],
                    "p95_derivative_eval_seconds": timings["derivative"][1],
                    "mad_derivative_eval_seconds": noise["derivative"][1],
                    "median_cost_arithmetic_net_seconds": max(
                        0.0, timings["cost"][0] - dispatch_median
                    ),
                    "queries_per_second": projected / timings["cost"][0],
                    "relative_to_bpr_baseline": np.nan,
                    "exact_root_calls": root_calls,
                    "peak_rss_bytes": peak,
                    "machine_metadata_json": machine_metadata,
                    "offline_provenance_json": "{}",
                    "notes": (
                        "actual reopened Phase 5 parameters packed one record per projected "
                        "link in structure-of-arrays vector kernels; scalar-runtime roundtrip "
                        "checked; setup excluded; dispatch/buffer copy measured separately"
                        + (
                            "; mixed projection uses exact selected-sample classes and times "
                            "retained/uncompiled backbone slots with the ordinary-BPR "
                            "reference kernel; terminal and regional/ICNN families are excluded"
                            if kernel_class == "PROJECTED_MIXED_CLASS"
                            else ""
                        )
                    ),
                }
            )

        offline_objects, sample_q, offline_provenance = (
            _offline_oracle_sample_for_suite(
                suite_id,
                offline_experiments,
                chain_candidates,
                maximum_objects=64,
            )
        )
        sample_count = len(offline_objects)

        def offline(attribute: str) -> Callable[[], np.ndarray]:
            def evaluate() -> np.ndarray:
                from .bundle_oracle import clear_oracle_query_cache
                clear_oracle_query_cache()
                results = [
                    exact_series_oracle(
                        getattr(experiment, "primitive_bundles"),
                        float(query),
                        internal_charge_s=float(
                            getattr(experiment, "internal_charge_s")
                        ),
                    )
                    for experiment, query in zip(
                        offline_objects, sample_q, strict=True
                    )
                ]
                return np.asarray([getattr(value, attribute) for value in results])

            return evaluate

        reset_exact_root_call_count()
        timing: dict[str, tuple[float, float]] = {}
        noise: dict[str, tuple[float, float]] = {}
        peak = process_tree_rss_bytes()
        for operation, attribute in (
            ("cost", "marginal_cost_s"),
            ("potential", "potential_flow_seconds"),
            ("derivative", "cost_derivative_s_per_flow"),
        ):
            if progress_guard is not None:
                progress_guard(f"benchmark_{suite_id}_offline_{operation}")
            samples, operation_peak = _times(
                offline(attribute), warmups, repetitions
            )
            timing[operation] = _percentiles(samples)
            noise[operation] = _noise_band(samples)
            peak = max(peak, operation_peak)
        rows.append(
            {
                "suite_short_id": suite_id,
                "kernel_class": "EXACT_ROOT_ORACLE_OFFLINE_ONLY",
                "projected_link_count": sample_count,
                "projection_reference_link_count": projected,
                "projection_scope": "BOUNDED_OFFLINE_ORACLE_SAMPLE_ONLY",
                "projected_class_counts_json": json.dumps(
                    {"EXACT_ROOT_ORACLE_OFFLINE_ONLY": sample_count}, separators=(",", ":")
                ),
                "warmup_count": warmups,
                "timed_repetition_count": repetitions,
                "setup_seconds": 0.0,
                "packed_parameter_record_count": sample_count,
                "scalar_roundtrip_max_abs_error": 0.0,
                "median_dispatch_buffer_seconds": 0.0,
                "p95_dispatch_buffer_seconds": 0.0,
                "median_cost_eval_seconds": timing["cost"][0],
                "p05_cost_eval_seconds": noise["cost"][0],
                "p95_cost_eval_seconds": timing["cost"][1],
                "mad_cost_eval_seconds": noise["cost"][1],
                "median_potential_eval_seconds": timing["potential"][0],
                "p05_potential_eval_seconds": noise["potential"][0],
                "p95_potential_eval_seconds": timing["potential"][1],
                "mad_potential_eval_seconds": noise["potential"][1],
                "median_derivative_eval_seconds": timing["derivative"][0],
                "p05_derivative_eval_seconds": noise["derivative"][0],
                "p95_derivative_eval_seconds": timing["derivative"][1],
                "mad_derivative_eval_seconds": noise["derivative"][1],
                "median_cost_arithmetic_net_seconds": timing["cost"][0],
                "queries_per_second": sample_count / timing["cost"][0],
                "relative_to_bpr_baseline": np.nan,
                "exact_root_calls": get_exact_root_call_count(),
                "peak_rss_bytes": peak,
                "machine_metadata_json": machine_metadata,
                "offline_provenance_json": json.dumps(
                    offline_provenance, sort_keys=True, separators=(",", ":")
                ),
                "notes": (
                    f"offline exact root comparator on {sample_count} deterministic "
                    "real macro objects from the bounded canonical pilot sample; "
                    "suite coverage is established by immutable canonical-signature "
                    "occurrences while representative source provenance is retained; "
                    "never a production fallback"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    for suite_id, group in frame.groupby("suite_short_id", sort=False):
        baseline = float(
            group.loc[
                group["kernel_class"].eq("ORDINARY_BPR_BASELINE"),
                "median_cost_eval_seconds",
            ].iloc[0]
        )
        online = group.index[
            ~group["kernel_class"].eq("EXACT_ROOT_ORACLE_OFFLINE_ONLY")
        ]
        frame.loc[online, "relative_to_bpr_baseline"] = (
            frame.loc[online, "median_cost_eval_seconds"] / baseline
        )
    return frame
