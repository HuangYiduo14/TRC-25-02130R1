"""FFTT, AADT, background-traffic, and map-derived capacity enrichment formulas.

Assignment-only capacity multipliers are applied later when building Prompt-4
graph snapshots, so this module does not mutate stored network maps for solver
experiments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _partC_rev1_float(value: Any, default: float | None = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default) if default is not None else np.nan
        return float(value)
    except (TypeError, ValueError):
        return float(default) if default is not None else np.nan


def _partC_rev1_class_key(value: Any) -> str:
    numeric = _partC_rev1_float(value)
    if np.isfinite(numeric):
        return str(int(numeric))
    text = str(value or "").strip()
    return text.split(".")[0] if text else ""


def partC_rev1_preserve_existing_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    table_name: str,
    reason: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in out.columns:
            continue
        backup = f"partC_rev1_before_match_{column}"
        if backup not in out.columns:
            out[backup] = out[column]
        rows.append(
            {
                "table_name": table_name,
                "column_name": column,
                "backup_column_name": backup,
                "reason_for_backup": reason,
                "num_non_null_before": int(out[backup].notna().sum()),
                "num_non_null_after": int(out[column].notna().sum()),
                "num_changed_values": int((out[column].astype(str) != out[backup].astype(str)).sum()),
            }
        )
    return out, pd.DataFrame(rows)


def _partC_rev1_default_by_class(config: dict[str, Any], section_key: str, class_code: Any, fallback: float) -> float:
    defaults = config.get("capacity_enrichment", {}).get(section_key, {})
    key = _partC_rev1_class_key(class_code)
    return float(defaults.get(key, fallback))


def partC_rev1_fill_fftt_for_row(row: pd.Series, config: dict[str, Any]) -> tuple[float, str]:
    cap_cfg = config.get("capacity_enrichment", {})
    direction = str(row.get("direction", "AB")).upper()
    length_miles = _partC_rev1_float(row.get("length_miles", row.get("partC_rev1_length_miles")), 0.0)
    class_code = row.get("class_code", row.get("Class", row.get("partC_rev1_class")))
    default_speed = _partC_rev1_default_by_class(config, "default_speed_by_faf_class", class_code, float(cap_cfg.get("default_speed_mph", 45.0)))

    time_col = "AB_FreeFlowTime" if direction == "AB" else "BA_FreeFlowTime"
    speed_col = "AB_FinalSpeed" if direction == "AB" else "BA_FinalSpeed"
    existing_time = _partC_rev1_float(row.get(time_col), np.nan)
    if np.isfinite(existing_time) and existing_time > 0:
        return existing_time, f"existing_{time_col}"
    arc_time = _partC_rev1_float(row.get("free_time_min"), np.nan)
    if np.isfinite(arc_time) and arc_time > 0:
        return arc_time, "existing_free_time_min"
    speed = _partC_rev1_float(row.get(speed_col), np.nan)
    if np.isfinite(speed) and speed > 0 and length_miles > 0:
        return length_miles / speed * 60.0, f"computed_from_{speed_col}"
    speed_limit = _partC_rev1_float(row.get("Speed_Limit"), np.nan)
    if np.isfinite(speed_limit) and speed_limit > 0 and length_miles > 0:
        return length_miles / speed_limit * 60.0, "computed_from_Speed_Limit"
    ntad_speed = _partC_rev1_float(row.get("partC_rev1_ntad_speed_limit_mph"), np.nan)
    if np.isfinite(ntad_speed) and ntad_speed > 0 and length_miles > 0:
        return length_miles / ntad_speed * 60.0, "computed_from_ntad_speed"
    if length_miles > 0:
        return length_miles / max(default_speed, 1e-9) * 60.0, "computed_from_default_speed_by_class"
    return 1.0, "fallback_one_minute"


def partC_rev1_capacity_vphpl(class_code: Any, speed_mph: float, config: dict[str, Any]) -> tuple[float, str]:
    key = _partC_rev1_class_key(class_code)
    if key in {"11", "12", "13"}:
        return min(2400.0, 1700.0 + 10.0 * max(speed_mph, 0.0)), "freeway_speed_formula"
    default = _partC_rev1_default_by_class(config, "default_capacity_vphpl_by_faf_class", class_code, 1800.0)
    return default, "default_by_faf_class"


def partC_rev1_enrich_directed_arcs_capacity(
    directed_arcs: pd.DataFrame,
    enriched_links: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cap_cfg = config.get("capacity_enrichment", {})
    link_cols = [
        "partC_rev1_link_uid",
        "partC_rev1_faf_strahnet_nonempty",
        "STRAHNET",
        "NHS",
        "NHFN",
        "AB_Lanes",
        "BA_Lanes",
        "AB_FreeFlowTime",
        "BA_FreeFlowTime",
        "AB_FinalSpeed",
        "BA_FinalSpeed",
        "Speed_Limit",
        "Class",
        "F_Class",
        "partC_rev1_ntad_match_status",
        "partC_rev1_match_dataset",
        "partC_rev1_match_source_priority",
        "partC_rev1_match_method",
        "partC_rev1_match_score",
        "partC_rev1_match_second_score",
        "partC_rev1_match_ambiguity_ratio",
        "partC_rev1_match_close_competitor",
        "partC_rev1_match_angle_diff_deg",
        "partC_rev1_match_min_distance_m",
        "partC_rev1_match_name_similarity",
        "partC_rev1_match_route_similarity",
        "partC_rev1_match_semantic_score",
        "partC_rev1_match_spatial_only",
        "partC_rev1_match_fips_score",
        "partC_rev1_faf_route_key",
        "partC_rev1_ntad_route_key",
        "partC_rev1_faf_normalized_sign_key",
        "partC_rev1_ntad_normalized_sign_key",
        "partC_rev1_normalized_sign_match",
        "partC_rev1_normalized_sign_global_fallback",
        "partC_rev1_normalized_sign_selected_distance_m",
        "partC_rev1_geo_buffer_radius_m",
        "partC_rev1_geo_buffer_intersects",
        "partC_rev1_geo_buffer_overlap_score",
        "partC_rev1_geo_direct_distance_m",
        "partC_rev1_faf_stfips",
        "partC_rev1_faf_ctfips",
        "partC_rev1_ntad_stfips",
        "partC_rev1_ntad_ctfips",
        "partC_rev1_ntad_uid",
        "partC_rev1_ntad_objectid",
        "partC_rev1_ntad_lname",
        "partC_rev1_ntad_sign1",
        "partC_rev1_ntad_signt1",
        "partC_rev1_ntad_signn1",
        "partC_rev1_ntad_routeid",
        "partC_rev1_ntad_source_label",
        "partC_rev1_ntad_nhs",
        "partC_rev1_ntad_aadt_total",
        "partC_rev1_ntad_aadt_combination",
        "partC_rev1_ntad_aadt_single",
        "partC_rev1_ntad_aadt_truck",
        "partC_rev1_ntad_speed_limit_mph",
        "partC_rev1_aadt_source",
        "partC_rev1_aadt_truck_source",
    ]
    available_link_cols = [column for column in link_cols if column in enriched_links.columns]
    out = directed_arcs.copy()
    out, preservation = partC_rev1_preserve_existing_columns(
        out,
        ["free_time_min", "capacity"],
        table_name="partC_rev1_directed_arcs_enriched",
        reason="Prompt 2.5 computes enriched FFTT and freight capacity while preserving Prompt 2 values.",
    )
    out = out.merge(enriched_links[available_link_cols], on="partC_rev1_link_uid", how="left", suffixes=("", "_link"))

    fftt_values: list[float] = []
    fftt_sources: list[str] = []
    lanes_values: list[float] = []
    lanes_sources: list[str] = []
    capacity_vphpl_values: list[float] = []
    vehicle_capacity_values: list[float] = []
    aadt_total_dir: list[float] = []
    aadt_truck_dir: list[float] = []
    background_total: list[float] = []
    background_truck: list[float] = []
    background_nontruck: list[float] = []
    remaining_truck_vph: list[float] = []
    remaining_truck_aadt: list[float] = []
    capacity_ktons: list[float] = []
    background_truck_ktons: list[float] = []
    capacity_sources: list[str] = []
    warnings: list[str] = []

    payload = float(cap_cfg.get("payload_tons_per_truck", 18.0))
    truck_pce = float(cap_cfg.get("truck_pce", 2.0))
    direction_split = float(cap_cfg.get("direction_split", 0.5))
    k_factor = float(cap_cfg.get("k_factor", 0.10))
    d_factor = float(cap_cfg.get("d_factor", 0.55))
    kd = max(k_factor * d_factor, 1e-9)
    phf = float(cap_cfg.get("fallback_phf", 0.90))
    min_fraction = float(cap_cfg.get("min_remaining_capacity_fraction", 0.05))

    for _, row in out.iterrows():
        warning_parts: list[str] = []
        fftt, fftt_source = partC_rev1_fill_fftt_for_row(row, config)
        length_miles = _partC_rev1_float(row.get("length_miles", row.get("partC_rev1_length_miles")), 0.0)
        speed_mph = length_miles / max(fftt, 1e-9) * 60.0 if length_miles > 0 else float(cap_cfg.get("default_speed_mph", 45.0))
        direction = str(row.get("direction", "AB")).upper()
        lane_col = "AB_Lanes" if direction == "AB" else "BA_Lanes"
        lanes = _partC_rev1_float(row.get(lane_col), np.nan)
        lanes_source = lane_col
        if not np.isfinite(lanes) or lanes <= 0:
            lanes = _partC_rev1_float(row.get("partC_rev1_default_lane_count"), np.nan)
            lanes_source = "partC_rev1_default_lane_count"
        if not np.isfinite(lanes) or lanes <= 0:
            lanes = 1.0
            lanes_source = "fallback_one_lane"
            warning_parts.append("missing_or_nonpositive_lanes")

        class_code = row.get("class_code", row.get("Class"))
        cap_vphpl, cap_source = partC_rev1_capacity_vphpl(class_code, speed_mph, config)
        vehicle_capacity = lanes * cap_vphpl * phf

        aadt_total = _partC_rev1_float(row.get("partC_rev1_ntad_aadt_total"), np.nan)
        aadt_source = str(row.get("partC_rev1_aadt_source") or "ntad_aadt")
        if not np.isfinite(aadt_total) or aadt_total <= 0:
            aadt_total = _partC_rev1_default_by_class(config, "default_aadt_by_faf_class", class_code, 15000.0)
            aadt_source = "default_by_class"
            warning_parts.append("default_aadt")
        truck_aadt = _partC_rev1_float(row.get("partC_rev1_ntad_aadt_truck"), np.nan)
        truck_source = str(row.get("partC_rev1_aadt_truck_source") or "ntad_truck_aadt")
        if not np.isfinite(truck_aadt) or truck_aadt < 0:
            truck_aadt = aadt_total * float(cap_cfg.get("default_truck_ratio", 0.12))
            truck_source = "default_ratio_from_total_aadt"

        total_dir = aadt_total * direction_split
        truck_dir = truck_aadt * direction_split
        bg_total = total_dir * kd
        bg_truck = truck_dir * kd
        bg_nontruck = max(0.0, bg_total - bg_truck)
        remaining_pce = vehicle_capacity - bg_nontruck - truck_pce * bg_truck
        remaining_vph = max(0.0, remaining_pce / truck_pce)
        remaining_vph = max(remaining_vph, min_fraction * vehicle_capacity / truck_pce)
        remaining_aadt = remaining_vph / kd
        # This is the map-derived base freight capacity in annual thousand tons.
        # Prompt-4 assignment-only multipliers are applied later when graph
        # snapshots are built, never in these authoritative map/cache fields.
        annual_ktons = remaining_aadt * 365.0 * payload / 1000.0
        bg_truck_annual_ktons = truck_dir * 365.0 * payload / 1000.0

        fftt_values.append(fftt)
        fftt_sources.append(fftt_source)
        lanes_values.append(lanes)
        lanes_sources.append(lanes_source)
        capacity_vphpl_values.append(cap_vphpl)
        vehicle_capacity_values.append(vehicle_capacity)
        aadt_total_dir.append(total_dir)
        aadt_truck_dir.append(truck_dir)
        background_total.append(bg_total)
        background_truck.append(bg_truck)
        background_nontruck.append(bg_nontruck)
        remaining_truck_vph.append(remaining_vph)
        remaining_truck_aadt.append(remaining_aadt)
        capacity_ktons.append(annual_ktons)
        background_truck_ktons.append(bg_truck_annual_ktons)
        capacity_sources.append(f"{cap_source};aadt={aadt_source};truck={truck_source}")
        warnings.append(";".join(warning_parts))

    out["partC_rev1_fftt_min"] = fftt_values
    out["partC_rev1_fftt_source"] = fftt_sources
    out["partC_rev1_lanes_dir"] = lanes_values
    out["partC_rev1_lanes_source"] = lanes_sources
    out["partC_rev1_capacity_vphpl"] = capacity_vphpl_values
    out["partC_rev1_vehicle_capacity_vph_dir"] = vehicle_capacity_values
    out["partC_rev1_aadt_total_dir"] = aadt_total_dir
    out["partC_rev1_aadt_truck_dir"] = aadt_truck_dir
    out["partC_rev1_background_total_vph_dir"] = background_total
    out["partC_rev1_background_truck_vph_dir"] = background_truck
    out["partC_rev1_background_nontruck_vph_dir"] = background_nontruck
    out["partC_rev1_remaining_truck_vph_dir"] = remaining_truck_vph
    out["partC_rev1_remaining_truck_aadt_dir"] = remaining_truck_aadt
    out["partC_rev1_capacity_annual_ktons_dir"] = capacity_ktons
    out["partC_rev1_background_truck_annual_ktons_dir"] = background_truck_ktons
    out["partC_rev1_capacity_source"] = capacity_sources
    out["partC_rev1_enrichment_warning"] = warnings

    defaulted = out.loc[out["partC_rev1_enrichment_warning"].astype(str).str.contains("default", na=False)].copy()
    fftt_filled = out.loc[~out["partC_rev1_fftt_source"].astype(str).str.startswith("existing_", na=False)].copy()
    return out, preservation, pd.DataFrame({"capacity_defaulted_rows": [len(defaulted)], "fftt_filled_rows": [len(fftt_filled)]})


def partC_rev1_attach_link_aadt_defaults(enriched_links: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cap_cfg = config.get("capacity_enrichment", {})
    out = enriched_links.copy()
    combination = pd.to_numeric(out.get("partC_rev1_ntad_aadt_combination"), errors="coerce")
    single = pd.to_numeric(out.get("partC_rev1_ntad_aadt_single"), errors="coerce")
    default_total: list[float] = []
    aadt_source: list[str] = []
    for _, row in out.iterrows():
        value = _partC_rev1_float(row.get("partC_rev1_ntad_aadt_total"), np.nan)
        if np.isfinite(value) and value > 0:
            default_total.append(value)
            aadt_source.append("ntad_aadt")
        else:
            default_total.append(
                _partC_rev1_default_by_class(config, "default_aadt_by_faf_class", row.get("Class", row.get("partC_rev1_class")), 15000.0)
            )
            aadt_source.append("default_by_class")
    out["partC_rev1_ntad_aadt_total"] = default_total
    out["partC_rev1_aadt_source"] = aadt_source

    total = pd.to_numeric(out["partC_rev1_ntad_aadt_total"], errors="coerce")
    truck = combination.fillna(0.0) + single.fillna(0.0)
    has_truck = combination.notna() | single.notna()
    truck = truck.where(has_truck, total * float(cap_cfg.get("default_truck_ratio", 0.12)))
    out["partC_rev1_ntad_aadt_truck"] = truck
    out["partC_rev1_aadt_truck_source"] = np.where(has_truck, "ntad_truck_aadt_fields", "default_ratio_from_total_aadt")
    return out
