"""Frozen deterministic Round 3 operational finalist admission."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


METRICS = (
    "bacn_count",
    "suppressed_fftt_share",
    "p99_member_count",
    "projected_full_build_seconds",
)


class Phase3AAdmissionError(ValueError):
    """Raised when the frozen shortlist lacks required census evidence."""


def admit_round3_finalists(
    candidate_metrics: pd.DataFrame,
    *,
    allow_optional_sl: bool,
    maximum_production_finalists: int = 6,
) -> pd.DataFrame:
    """Apply the exact diversity/representative rule to complete census rows.

    All input candidates are returned with their metrics, deterministic slot,
    disposition, tie fields, and reason.  This is resource admission, never a
    scientific winning-method decision.
    """

    required = {
        "candidate_id", "family", "algorithm", "parameter_value",
        "production_eligible", "valid", *METRICS,
    }
    if not required.issubset(candidate_metrics.columns):
        raise Phase3AAdmissionError(
            f"Candidate admission lacks columns: {sorted(required-set(candidate_metrics.columns))}"
        )
    frame = candidate_metrics.copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if frame.empty or frame.candidate_id.duplicated().any():
        raise Phase3AAdmissionError("Candidate census must be nonempty with unique IDs")
    for column in METRICS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or (~np.isfinite(frame[column].to_numpy(dtype=np.float64))).any():
            raise Phase3AAdmissionError(f"Candidate admission metric is nonfinite: {column}")
    if (frame[list(METRICS)] < 0.0).any().any():
        raise Phase3AAdmissionError("Candidate admission metrics must be nonnegative")

    selected: list[tuple[str, str, str]] = []
    production = frame[frame.production_eligible.astype(bool) & frame.valid.astype(bool)].copy()
    geo_cl = production[(production.family == "GEO") & (production.algorithm == "CL")]
    if not geo_cl.empty:
        counts = geo_cl.bacn_count.to_numpy(dtype=np.float64)
        selected.append((*_nearest(geo_cl, float(np.quantile(counts, 0.75))), "GEO_CL_BALANCED"))
        remaining_geo_cl = geo_cl[~geo_cl.candidate_id.isin([selected[-1][0]])]
        if not remaining_geo_cl.empty:
            selected.append((*_nearest(remaining_geo_cl, float(np.quantile(counts, 0.25))), "GEO_CL_AGGRESSIVE"))

    geo_gc = production[(production.family == "GEO") & (production.algorithm == "GC")]
    chosen_cl_counts = [
        float(frame.loc[frame.candidate_id == candidate_id, "bacn_count"].iloc[0])
        for candidate_id, _, slot in selected if slot.startswith("GEO_CL")
    ]
    if not geo_gc.empty and chosen_cl_counts:
        target = math.sqrt(float(np.prod(chosen_cl_counts)))
        selected.append((*_nearest(geo_gc, target, logarithmic=True), "GEO_GC_COMPRESSION_MATCH"))

    aug_choices: dict[float, tuple[str, str, str]] = {}
    aug = production[(production.family == "AUG") & (production.algorithm.isin(["CL", "GC"]))]
    for threshold in (30.0, 120.0):
        group = aug[np.isclose(aug.parameter_value.to_numpy(dtype=np.float64), threshold)]
        if not group.empty:
            candidate_id, reason = _aug_threshold_choice(group)
            slot = f"AUG_{int(threshold):03d}S_PRIMARY"
            aug_choices[threshold] = (candidate_id, reason, slot)
            selected.append(aug_choices[threshold])

    if aug_choices:
        difference_rows: list[tuple[float, float]] = []
        for threshold in sorted(aug_choices):
            group = aug[np.isclose(aug.parameter_value.to_numpy(dtype=np.float64), threshold)]
            by_algorithm = {str(row.algorithm): float(row.bacn_count) for row in group.itertuples(index=False)}
            if {"CL", "GC"}.issubset(by_algorithm) and min(by_algorithm.values()) > 0.0:
                difference_rows.append((abs(math.log(by_algorithm["CL"] / by_algorithm["GC"])), threshold))
        for _, threshold in sorted(difference_rows, key=lambda item: (-item[0], item[1])):
            group = aug[np.isclose(aug.parameter_value.to_numpy(dtype=np.float64), threshold)]
            primary = aug_choices[threshold][0]
            complement = group[group.candidate_id != primary]
            if not complement.empty:
                row = complement.sort_values("candidate_id", kind="mergesort").iloc[0]
                selected.append((str(row.candidate_id), "OTHER_ALGORITHM_AT_MAX_LOG_COUNT_GAP", "AUG_COMPLEMENTARY"))
                break

    # Fill missing strata/slots deterministically from the strongest remaining
    # valid production rows, without exceeding the frozen maximum.
    selected_ids = {value[0] for value in selected}
    if len(selected) < int(maximum_production_finalists):
        remaining = production[~production.candidate_id.isin(selected_ids)].copy()
        remaining["fallback_rank_sum"] = _rank_sum(remaining)
        for row in remaining.sort_values(
            ["fallback_rank_sum", "candidate_id"], kind="mergesort"
        ).itertuples(index=False):
            selected.append((str(row.candidate_id), "BEST_REMAINING_VALID_PRODUCTION", "FALLBACK_PRODUCTION"))
            if len(selected) >= int(maximum_production_finalists):
                break
    selected = selected[: int(maximum_production_finalists)]

    if allow_optional_sl and selected:
        sl = frame[(frame.algorithm == "SL") & frame.valid.astype(bool)]
        if not sl.empty:
            selected_counts = frame.set_index("candidate_id").loc[[value[0] for value in selected], "bacn_count"].to_numpy(dtype=np.float64)
            target_rows: list[tuple[float, str, str]] = []
            for row in sl.itertuples(index=False):
                nearest = float(np.min(np.abs(np.log1p(selected_counts) - math.log1p(float(row.bacn_count)))))
                target_rows.append((nearest, str(row.candidate_id), "SL_NEAREST_SELECTED_BACN_COUNT"))
            _, candidate_id, reason = min(target_rows)
            selected.append((candidate_id, reason, "OPTIONAL_SL_ABLATION"))

    slot_by_id = {candidate_id: (index + 1, slot, reason) for index, (candidate_id, reason, slot) in enumerate(selected)}
    output = frame.sort_values("candidate_id", kind="mergesort").copy()
    output["selected_slot"] = [slot_by_id.get(value, (None, None, None))[0] for value in output.candidate_id]
    output["selected_slot_role"] = [slot_by_id.get(value, (None, None, None))[1] for value in output.candidate_id]
    output["admission_reason"] = [
        slot_by_id.get(value, (None, None, "NOT_ADMITTED_BY_FROZEN_DIVERSITY_RULE"))[2]
        for value in output.candidate_id
    ]
    output["disposition"] = [
        "ADMITTED_OPTIONAL_SL" if slot_by_id.get(value, (None, "", ""))[1] == "OPTIONAL_SL_ABLATION"
        else "ADMITTED_PRODUCTION" if value in slot_by_id
        else "PRUNED_BY_ADMISSION_POLICY" if bool(valid)
        else "FAILED_CENSUS_VALIDATION"
        for value, valid in zip(output.candidate_id, output.valid, strict=True)
    ]
    output["winning_method_selected"] = False
    return output.reset_index(drop=True)


def _tie_key(row: Any) -> tuple[float, float, float, str]:
    return (
        float(row.suppressed_fftt_share),
        float(row.p99_member_count),
        float(row.projected_full_build_seconds),
        str(row.candidate_id),
    )


def _nearest(
    rows: pd.DataFrame, target: float, *, logarithmic: bool = False
) -> tuple[str, str]:
    values = []
    for row in rows.itertuples(index=False):
        count = float(row.bacn_count)
        distance = abs(math.log(max(count, 1e-300)) - math.log(max(target, 1e-300))) if logarithmic else abs(count - target)
        values.append((distance, _tie_key(row), str(row.candidate_id)))
    _, _, candidate_id = min(values)
    return candidate_id, "NEAREST_FROZEN_REPRESENTATIVE_TARGET"


def _dominates(left: Any, right: Any) -> bool:
    left_values = np.asarray([float(getattr(left, value)) for value in METRICS])
    right_values = np.asarray([float(getattr(right, value)) for value in METRICS])
    return bool(np.all(left_values <= right_values) and np.any(left_values < right_values))


def _aug_threshold_choice(group: pd.DataFrame) -> tuple[str, str]:
    rows = list(group.sort_values("candidate_id", kind="mergesort").itertuples(index=False))
    nondominated = [row for row in rows if not any(_dominates(other, row) for other in rows if other.candidate_id != row.candidate_id)]
    if len(nondominated) == 1:
        return str(nondominated[0].candidate_id), "WITHIN_THRESHOLD_NONDOMINATED"
    subset = pd.DataFrame([row._asdict() for row in nondominated])
    subset["rank_sum"] = _rank_sum(subset)
    row = subset.sort_values(["rank_sum", "candidate_id"], kind="mergesort").iloc[0]
    return str(row.candidate_id), "WITHIN_THRESHOLD_ORDINAL_RANK_SUM"


def _rank_sum(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=np.float64, index=rows.index)
    total = pd.Series(np.zeros(len(rows), dtype=np.float64), index=rows.index)
    for metric in METRICS:
        total = total + rows[metric].rank(method="min", ascending=True)
    return total
