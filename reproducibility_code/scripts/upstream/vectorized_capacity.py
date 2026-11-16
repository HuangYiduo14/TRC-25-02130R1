"""Array implementation of the preserved Prompt 2.5 freight capacity formulas.

The formulas are taken from partC_rev1_capacity_enrichment.py. Only evaluation
is changed: NumPy vectors replace per-arc Python rows. Source columns, map base
capacity and assignment-only scaling remain separate.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def numeric(frame, name, default=np.nan):
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").astype(float)


def class_defaults(frame, config, key, fallback):
    codes = numeric(frame, "class_code" if "class_code" in frame else "Class")
    lookup = {int(k): float(v) for k, v in config.get("capacity_enrichment", {}).get(key, {}).items()}
    return codes.map(lookup).fillna(fallback).to_numpy(float)


def attach_link_aadt_defaults(links, config):
    out = links.copy()
    cap = config.get("capacity_enrichment", {})
    total = numeric(out, "partC_rev1_ntad_aadt_total").to_numpy()
    has_total = np.isfinite(total) & (total > 0)
    defaults = class_defaults(out, config, "default_aadt_by_faf_class", 15000.0)
    total = np.where(has_total, total, defaults)
    combination = numeric(out, "partC_rev1_ntad_aadt_combination")
    single = numeric(out, "partC_rev1_ntad_aadt_single")
    has_truck = combination.notna().to_numpy() | single.notna().to_numpy()
    truck = combination.fillna(0).to_numpy() + single.fillna(0).to_numpy()
    out["partC_rev1_ntad_aadt_total"] = total
    out["partC_rev1_aadt_source"] = np.where(has_total, "ntad_aadt", "default_by_class")
    out["partC_rev1_ntad_aadt_truck"] = np.where(has_truck, truck, total * float(cap.get("default_truck_ratio", .12)))
    out["partC_rev1_aadt_truck_source"] = np.where(has_truck, "ntad_truck_aadt_fields", "default_ratio_from_total_aadt")
    return out


def enrich_directed_arcs_capacity(arcs, links, config):
    from partC_rev1_src.partC_rev1_capacity_enrichment import partC_rev1_preserve_existing_columns
    # Match the original join exactly; duplicate names use the original suffix.
    cap = config.get("capacity_enrichment", {})
    out, preservation = partC_rev1_preserve_existing_columns(
        arcs, ["free_time_min", "capacity"], table_name="partC_rev1_directed_arcs_enriched",
        reason="Prompt 2.5 computes enriched FFTT and freight capacity while preserving Prompt 2 values.")
    source_names = [n for n in links if n not in {"geometry", "partC_rev1_geom_wkb"}]
    out = out.merge(links[source_names], on="partC_rev1_link_uid", how="left", suffixes=("", "_link"), validate="many_to_one")
    n = len(out)
    is_ab = out.get("direction", pd.Series("AB", index=out.index)).astype(str).str.upper().eq("AB").to_numpy()
    length = numeric(out, "length_miles").fillna(0).to_numpy()
    fftt = np.ones(n)
    sources = np.full(n, "fallback_one_minute", dtype=object)
    default_speed = class_defaults(out, config, "default_speed_by_faf_class", float(cap.get("default_speed_mph", 45)))
    ntad_speed = numeric(out, "partC_rev1_ntad_speed_limit_mph").to_numpy()
    speed_limit = numeric(out, "Speed_Limit").to_numpy()
    final_speed = np.where(is_ab, numeric(out, "AB_FinalSpeed"), numeric(out, "BA_FinalSpeed"))
    explicit_time = np.where(is_ab, numeric(out, "AB_FreeFlowTime"), numeric(out, "BA_FreeFlowTime"))
    arc_time = numeric(out, "free_time_min").to_numpy()
    # Assign in reverse precedence so a later successful source wins.
    candidates = [
        (default_speed, np.full(n, "computed_from_default_speed_by_class", object)),
        (ntad_speed, np.full(n, "computed_from_ntad_speed", object)),
        (speed_limit, np.full(n, "computed_from_Speed_Limit", object)),
        (final_speed, np.where(is_ab, "computed_from_AB_FinalSpeed", "computed_from_BA_FinalSpeed")),
    ]
    for speed, label in candidates:  # Four whole-array operations, never per row.
        valid = np.isfinite(speed) & (speed > 0) & (length > 0)
        fftt[valid] = length[valid] / np.maximum(speed[valid], 1e-9) * 60
        sources[valid] = label[valid]
    for time, label in [(arc_time, np.full(n, "existing_free_time_min", object)),
                        (explicit_time, np.where(is_ab, "existing_AB_FreeFlowTime", "existing_BA_FreeFlowTime"))]:
        valid = np.isfinite(time) & (time > 0)
        fftt[valid], sources[valid] = time[valid], label[valid]
    speed = np.where(length > 0, length / np.maximum(fftt, 1e-9) * 60, float(cap.get("default_speed_mph", 45)))
    lanes = np.where(is_ab, numeric(out, "AB_Lanes"), numeric(out, "BA_Lanes"))
    lanes_source = np.where(is_ab, "AB_Lanes", "BA_Lanes").astype(object)
    missing = ~np.isfinite(lanes) | (lanes <= 0)
    lanes[missing] = numeric(out, "partC_rev1_default_lane_count").to_numpy()[missing]
    lanes_source[missing] = "partC_rev1_default_lane_count"
    missing = ~np.isfinite(lanes) | (lanes <= 0)
    lanes[missing], lanes_source[missing] = 1, "fallback_one_lane"
    codes = numeric(out, "class_code" if "class_code" in out else "Class").to_numpy()
    freeway = np.isin(codes, [11, 12, 13])
    cap_vphpl = class_defaults(out, config, "default_capacity_vphpl_by_faf_class", 1800)
    cap_vphpl = np.where(freeway, np.minimum(2400, 1700 + 10 * np.maximum(speed, 0)), cap_vphpl)
    vehicle = lanes * cap_vphpl * float(cap.get("fallback_phf", .9))
    total = numeric(out, "partC_rev1_ntad_aadt_total").to_numpy()
    total_missing = ~np.isfinite(total) | (total <= 0)
    total[total_missing] = class_defaults(out, config, "default_aadt_by_faf_class", 15000)[total_missing]
    truck = numeric(out, "partC_rev1_ntad_aadt_truck").to_numpy()
    truck_missing = ~np.isfinite(truck) | (truck < 0)
    truck[truck_missing] = (total * float(cap.get("default_truck_ratio", .12)))[truck_missing]
    split, pce = float(cap.get("direction_split", .5)), float(cap.get("truck_pce", 2))
    kd = max(float(cap.get("k_factor", .1)) * float(cap.get("d_factor", .55)), 1e-9)
    total_dir, truck_dir = total * split, truck * split
    bg_total, bg_truck = total_dir * kd, truck_dir * kd
    bg_nontruck = np.maximum(0, bg_total - bg_truck)
    remaining = np.maximum(0, (vehicle - bg_nontruck - pce * bg_truck) / pce)
    remaining = np.maximum(remaining, float(cap.get("min_remaining_capacity_fraction", .05)) * vehicle / pce)
    payload = float(cap.get("payload_tons_per_truck", 18))
    out["partC_rev1_fftt_min"], out["partC_rev1_fftt_source"] = fftt, sources
    columns = {
        "lanes_dir": lanes, "lanes_source": lanes_source, "capacity_vphpl": cap_vphpl,
        "vehicle_capacity_vph_dir": vehicle, "aadt_total_dir": total_dir, "aadt_truck_dir": truck_dir,
        "background_total_vph_dir": bg_total, "background_truck_vph_dir": bg_truck,
        "background_nontruck_vph_dir": bg_nontruck, "remaining_truck_vph_dir": remaining,
        "remaining_truck_aadt_dir": remaining / kd, "capacity_annual_ktons_dir": remaining / kd * 365 * payload / 1000,
        "background_truck_annual_ktons_dir": truck_dir * 365 * payload / 1000,
    }
    for key, values in columns.items():
        out["partC_rev1_" + key] = values
    aadt_source = out.get("partC_rev1_aadt_source", pd.Series("ntad_aadt", index=out.index)).fillna("ntad_aadt").astype(str).to_numpy(object)
    truck_source = out.get("partC_rev1_aadt_truck_source", pd.Series("ntad_truck_aadt", index=out.index)).fillna("ntad_truck_aadt").astype(str).to_numpy(object)
    aadt_source[total_missing] = "default_by_class"
    truck_source[truck_missing] = "default_ratio_from_total_aadt"
    out["partC_rev1_capacity_source"] = pd.Series(np.where(freeway, "freeway_speed_formula", "default_by_faf_class")) + ";aadt=" + pd.Series(aadt_source) + ";truck=" + pd.Series(truck_source)
    warnings = np.full(n, "", object)
    warnings[missing] = "missing_or_nonpositive_lanes"
    warnings[total_missing] = np.where(warnings[total_missing] == "", "default_aadt", warnings[total_missing] + ";default_aadt")
    out["partC_rev1_enrichment_warning"] = warnings
    report = pd.DataFrame({"capacity_defaulted_rows": [int(total_missing.sum())],
                           "fftt_filled_rows": [int((~pd.Series(sources).str.startswith("existing_")).sum())]})
    return out, preservation, report
