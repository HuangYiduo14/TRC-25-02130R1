"""FAF OD reading and vectorized county/centroid expansion."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from .partC_rev1_io import partC_rev1_read_table, partC_rev1_write_table
from .partC_rev1_paths import partC_rev1_output_file, partC_rev1_resolve_path
from .partC_rev1_schema import partC_rev1_find_column
from .partC_rev1_units import partC_rev1_convert_flow_units


_PARTC_REV1_LOG = logging.getLogger("partC_rev1")


def partC_rev1_normalize_fips(value: Any, width: int = 5) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
        if text.endswith(".0") and text.replace(".", "", 1).isdigit():
            text = str(int(float(text)))
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    if len(digits) > width:
        digits = digits[:width]
    return digits.zfill(width)


def partC_rev1_map_sctg2_to_sctgG5(value: Any) -> str | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    code = int(numeric)
    if 1 <= code <= 9:
        return "sctg0109"
    if 10 <= code <= 14:
        return "sctg1014"
    if 15 <= code <= 19:
        return "sctg1519"
    if 20 <= code <= 33:
        return "sctg2033"
    if 34 <= code <= 99:
        return "sctg3499"
    return None


def _partC_rev1_find_first(df: pd.DataFrame, preferred: str | None, candidates: list[str], required: bool = True) -> str | None:
    search = []
    if preferred:
        search.append(preferred)
    search.extend(candidates)
    return partC_rev1_find_column(df, search[0] if search else "", candidates=search[1:], required=required)


def partC_rev1_read_faf_od(config: dict[str, Any]) -> pd.DataFrame:
    od_cfg = config.get("od", {})
    od_path = partC_rev1_resolve_path(str(config.get("inputs", {}).get("od_path", "")))
    _PARTC_REV1_LOG.info("Reading FAF OD table from %s", od_path)
    df = partC_rev1_read_table(od_path)
    _PARTC_REV1_LOG.info("Read FAF OD raw rows=%s columns=%s", len(df), len(df.columns))
    demand_col = _partC_rev1_find_first(
        df,
        od_cfg.get("demand_col"),
        list(od_cfg.get("demand_col_candidates", ["tons_2017", "Tons_2017", "tons2017", "tons_17", "tons"])),
    )
    origin_col = _partC_rev1_find_first(df, od_cfg.get("origin_col", "dms_orig"), ["dms_orig"])
    dest_col = _partC_rev1_find_first(df, od_cfg.get("destination_col", "dms_dest"), ["dms_dest"])
    sctg_col = _partC_rev1_find_first(df, od_cfg.get("commodity_col", "sctg2"), ["sctg2"])
    mode_col = _partC_rev1_find_first(
        df,
        od_cfg.get("mode_col"),
        list(od_cfg.get("mode_col_candidates", ["dms_mode", "mode", "Mode"])),
        required=False,
    )
    if mode_col is not None:
        allowed = {str(value).strip().lower() for value in od_cfg.get("truck_mode_codes", [1, "1", "Truck", "truck"])}
        df = df.loc[df[mode_col].astype(str).str.strip().str.lower().isin(allowed)].copy()
        _PARTC_REV1_LOG.info("Filtered FAF OD to truck mode rows=%s using column=%s", len(df), mode_col)
    else:
        warnings = config.setdefault("_partC_rev1_warnings", [])
        warnings.append("No FAF OD mode column found; using all OD rows")

    raw_unit = str(config.get("unit_conversion", {}).get("od_raw_unit", od_cfg.get("demand_units", "annual_ktons")))
    solver_unit = str(config.get("unit_conversion", {}).get("solver_flow_unit", "annual_ktons"))
    demand_raw = pd.to_numeric(df[demand_col], errors="coerce").fillna(0.0)
    demand_annual_ktons = partC_rev1_convert_flow_units(demand_raw, raw_unit, "annual_ktons", config)
    demand_solver = partC_rev1_convert_flow_units(demand_raw, raw_unit, solver_unit, config)
    out = pd.DataFrame(
        {
            "dms_orig": df[origin_col].astype("string").str.strip(),
            "dms_dest": df[dest_col].astype("string").str.strip(),
            "sctg2": pd.to_numeric(df[sctg_col], errors="coerce"),
            "sctgG5": df[sctg_col].map(partC_rev1_map_sctg2_to_sctgG5),
            "demand_annual_ktons": demand_annual_ktons,
            "demand_solver_unit": demand_solver,
        }
    )
    invalid = out.loc[out["sctgG5"].isna() | (out["demand_annual_ktons"] <= 0)].copy()
    partC_rev1_write_table(invalid, partC_rev1_output_file(config, "partC_rev1_od_invalid_or_zero_report.csv"), index=False)
    out = out.loc[out["sctgG5"].notna()]
    if bool(od_cfg.get("drop_zero_demand", True)):
        out = out.loc[out["demand_annual_ktons"] > 0]
    grouped = (
        out.groupby(["dms_orig", "dms_dest", "sctgG5"], dropna=False, as_index=False)[
            ["demand_annual_ktons", "demand_solver_unit"]
        ]
        .sum()
        .reset_index(drop=True)
    )
    _PARTC_REV1_LOG.info("Normalized FAF OD rows=%s total_annual_ktons=%s", len(grouped), float(grouped["demand_annual_ktons"].sum()))
    return grouped


def _partC_rev1_factor_table(
    df: pd.DataFrame,
    *,
    region_col: str,
    county_col: str,
    factor_col: str,
    region_out: str,
    county_out: str,
    factor_out: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    sctg_col = config.get("od", {}).get("commodity_group_col", "sctgG5")
    if sctg_col not in df.columns and "sctg2" in df.columns:
        df = df.copy()
        df[sctg_col] = df["sctg2"].map(partC_rev1_map_sctg2_to_sctgG5)
    required = [region_col, county_col, factor_col, sctg_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Factor table missing columns: {missing}")
    width = int(config.get("od", {}).get("county_fips_width", 5))
    out = pd.DataFrame(
        {
            region_out: df[region_col].astype("string").str.strip(),
            county_out: df[county_col].map(lambda value: partC_rev1_normalize_fips(value, width)),
            "sctgG5": df[sctg_col].astype("string").str.strip(),
            factor_out: pd.to_numeric(df[factor_col], errors="coerce").fillna(0.0),
        }
    )
    return out


def partC_rev1_expand_faf_to_county_od(
    od_df: pd.DataFrame,
    origin_factors_df: pd.DataFrame,
    destination_factors_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    _PARTC_REV1_LOG.info(
        "Expanding FAF OD to county OD: od_rows=%s origin_factor_rows=%s destination_factor_rows=%s",
        len(od_df),
        len(origin_factors_df),
        len(destination_factors_df),
    )
    od_cfg = config.get("od", {})
    origin = _partC_rev1_factor_table(
        origin_factors_df,
        region_col=str(od_cfg.get("origin_factor_region_col", "dms_orig")),
        county_col=str(od_cfg.get("origin_factor_county_col", "dms_orig_cnty")),
        factor_col=str(od_cfg.get("origin_factor_col", "f_orig")),
        region_out="dms_orig",
        county_out="origin_county_fips",
        factor_out="f_orig",
        config=config,
    )
    dest = _partC_rev1_factor_table(
        destination_factors_df,
        region_col=str(od_cfg.get("destination_factor_region_col", "dms_dest")),
        county_col=str(od_cfg.get("destination_factor_county_col", "dms_dest_cnty")),
        factor_col=str(od_cfg.get("destination_factor_col", "f_dest")),
        region_out="dms_dest",
        county_out="destination_county_fips",
        factor_out="f_dest",
        config=config,
    )
    origin_sums = origin.groupby(["dms_orig", "sctgG5"], dropna=False)["f_orig"].sum().reset_index(name="factor_sum")
    dest_sums = dest.groupby(["dms_dest", "sctgG5"], dropna=False)["f_dest"].sum().reset_index(name="factor_sum")
    factor_report = pd.concat(
        [
            origin_sums.assign(factor_side="origin"),
            dest_sums.rename(columns={"dms_dest": "dms_orig"}).assign(factor_side="destination"),
        ],
        ignore_index=True,
    )
    partC_rev1_write_table(factor_report, partC_rev1_output_file(config, "partC_rev1_factor_sum_report.csv"), index=False)
    if bool(od_cfg.get("normalize_factors", False)):
        origin = origin.merge(origin_sums, on=["dms_orig", "sctgG5"], how="left")
        origin["f_orig"] = np.where(origin["factor_sum"] > 0, origin["f_orig"] / origin["factor_sum"], origin["f_orig"])
        origin = origin.drop(columns=["factor_sum"])
        dest = dest.merge(dest_sums, on=["dms_dest", "sctgG5"], how="left")
        dest["f_dest"] = np.where(dest["factor_sum"] > 0, dest["f_dest"] / dest["factor_sum"], dest["f_dest"])
        dest = dest.drop(columns=["factor_sum"])

    merged = od_df.merge(origin, on=["dms_orig", "sctgG5"], how="left").merge(dest, on=["dms_dest", "sctgG5"], how="left")
    unmatched = merged.loc[merged["origin_county_fips"].isna() | merged["destination_county_fips"].isna()].copy()
    partC_rev1_write_table(unmatched, partC_rev1_output_file(config, "partC_rev1_od_factor_unmatched_report.csv"), index=False)
    matched = merged.dropna(subset=["origin_county_fips", "destination_county_fips"]).copy()
    matched["demand_annual_ktons"] = matched["demand_annual_ktons"] * matched["f_orig"] * matched["f_dest"]
    matched["demand_solver_unit"] = matched["demand_solver_unit"] * matched["f_orig"] * matched["f_dest"]
    county = (
        matched.groupby(["origin_county_fips", "destination_county_fips", "sctgG5"], dropna=False, as_index=False)[
            ["demand_annual_ktons", "demand_solver_unit"]
        ]
        .sum()
        .reset_index(drop=True)
    )
    report = pd.DataFrame(
        [
            {
                "input_demand_annual_ktons": float(od_df["demand_annual_ktons"].sum()),
                "county_demand_annual_ktons": float(county["demand_annual_ktons"].sum()),
                "relative_difference": (
                    float(county["demand_annual_ktons"].sum() - od_df["demand_annual_ktons"].sum())
                    / float(od_df["demand_annual_ktons"].sum())
                    if float(od_df["demand_annual_ktons"].sum()) != 0
                    else 0.0
                ),
            }
        ]
    )
    partC_rev1_write_table(report, partC_rev1_output_file(config, "partC_rev1_od_county_demand_preservation_report.csv"), index=False)
    _PARTC_REV1_LOG.info("County OD expansion complete: rows=%s total_annual_ktons=%s", len(county), float(county["demand_annual_ktons"].sum()))
    return county


def partC_rev1_expand_county_od_to_centroids(
    county_od_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    _PARTC_REV1_LOG.info("Expanding county OD to centroids: county_od_rows=%s centroid_rows=%s", len(county_od_df), len(centroids_df))
    origin_centroids = centroids_df.rename(
        columns={
            "county_fips": "origin_county_fips",
            "partC_rev1_node_uid": "origin_node_uid",
            "node_id": "origin_node_id",
            "weight_within_county": "origin_weight",
        }
    )[["origin_county_fips", "origin_node_uid", "origin_node_id", "origin_weight"]]
    dest_centroids = centroids_df.rename(
        columns={
            "county_fips": "destination_county_fips",
            "partC_rev1_node_uid": "destination_node_uid",
            "node_id": "destination_node_id",
            "weight_within_county": "destination_weight",
        }
    )[["destination_county_fips", "destination_node_uid", "destination_node_id", "destination_weight"]]
    merged = county_od_df.merge(origin_centroids, on="origin_county_fips", how="left").merge(
        dest_centroids,
        on="destination_county_fips",
        how="left",
    )
    unmatched = merged.loc[merged["origin_node_uid"].isna() | merged["destination_node_uid"].isna()].copy()
    partC_rev1_write_table(unmatched, partC_rev1_output_file(config, "partC_rev1_od_centroid_unmatched_counties.csv"), index=False)
    partC_rev1_write_table(unmatched, partC_rev1_output_file(config, "partC_rev1_unmatched_od_counties.csv"), index=False)
    matched = merged.dropna(subset=["origin_node_uid", "destination_node_uid"]).copy()
    matched["demand_annual_ktons"] = matched["demand_annual_ktons"] * matched["origin_weight"] * matched["destination_weight"]
    matched["demand_solver_unit"] = matched["demand_solver_unit"] * matched["origin_weight"] * matched["destination_weight"]
    centroid = (
        matched.groupby(
            [
                "origin_node_uid",
                "destination_node_uid",
                "origin_node_id",
                "destination_node_id",
                "origin_county_fips",
                "destination_county_fips",
                "sctgG5",
            ],
            dropna=False,
            as_index=False,
        )[["demand_annual_ktons", "demand_solver_unit"]]
        .sum()
        .reset_index(drop=True)
    )
    _PARTC_REV1_LOG.info(
        "Centroid OD expansion complete: rows=%s unmatched_county_rows=%s total_annual_ktons=%s",
        len(centroid),
        len(unmatched),
        float(centroid["demand_annual_ktons"].sum()) if "demand_annual_ktons" in centroid else 0.0,
    )
    return centroid
