"""County centroid extraction from cleaned FAF node layers."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd

from .partC_rev1_io import partC_rev1_write_table
from .partC_rev1_paths import partC_rev1_output_file
from .partC_rev1_schema import partC_rev1_coerce_string_id, partC_rev1_parse_county_fips_from_centroid_id


def _partC_rev1_col(config: dict[str, Any], key: str, default: str) -> str:
    return str(config.get("input_schema", {}).get(key, default) or default)


def _partC_rev1_is_centroid(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    return numeric.eq(1) | text.isin({"true", "yes", "centroid"})


def partC_rev1_extract_county_centroids(clean_node_gdf: gpd.GeoDataFrame, config: dict[str, Any]) -> pd.DataFrame:
    centroid_col = _partC_rev1_col(config, "centroid_col", "Centroid")
    centroid_id_col = _partC_rev1_col(config, "centroid_id_col", "CentroidID")
    fafid_col = _partC_rev1_col(config, "fafid_col", "FAFID")

    if centroid_col in clean_node_gdf.columns:
        mask = _partC_rev1_is_centroid(clean_node_gdf[centroid_col])
    else:
        mask = clean_node_gdf.get("partC_rev1_centroid", pd.Series([0] * len(clean_node_gdf))).astype(bool)
    centroids = clean_node_gdf.loc[mask].copy()
    if centroids.empty:
        out = pd.DataFrame(
            columns=[
                "node_id",
                "partC_rev1_node_uid",
                "centroid_id_raw",
                "county_fips",
                "fafid",
                "weight_within_county",
                "lon",
                "lat",
            ]
        )
        partC_rev1_write_table(out, partC_rev1_output_file(config, "partC_rev1_county_centroids_report.csv"), index=False)
        return out

    centroid_id = (
        partC_rev1_coerce_string_id(centroids[centroid_id_col])
        if centroid_id_col in centroids.columns
        else centroids.get("partC_rev1_centroid_id_raw", pd.Series([None] * len(centroids), dtype="string"))
    )
    county_fips = centroid_id.map(partC_rev1_parse_county_fips_from_centroid_id)
    node_ids = centroids.get("partC_rev1_original_node_id", centroids.get(_partC_rev1_col(config, "node_id_col", "ID")))
    lon = centroids.get("partC_rev1_lon")
    lat = centroids.get("partC_rev1_lat")
    if lon is None or lat is None:
        lon = centroids.geometry.map(lambda geom: geom.x if geom is not None and not geom.is_empty else None)
        lat = centroids.geometry.map(lambda geom: geom.y if geom is not None and not geom.is_empty else None)
    out = pd.DataFrame(
        {
            "node_id": partC_rev1_coerce_string_id(pd.Series(node_ids, index=centroids.index)).values,
            "partC_rev1_node_uid": centroids["partC_rev1_node_uid"].values,
            "centroid_id_raw": centroid_id.values,
            "county_fips": county_fips.values,
            "fafid": (
                partC_rev1_coerce_string_id(centroids[fafid_col]).values
                if fafid_col in centroids.columns
                else centroids.get("partC_rev1_fafid", pd.Series([None] * len(centroids), dtype="string")).values
            ),
            "lon": pd.to_numeric(lon, errors="coerce").values,
            "lat": pd.to_numeric(lat, errors="coerce").values,
        }
    )
    counts = out.groupby("county_fips", dropna=False)["partC_rev1_node_uid"].transform("count")
    out["weight_within_county"] = 1.0 / counts
    out = out[
        [
            "node_id",
            "partC_rev1_node_uid",
            "centroid_id_raw",
            "county_fips",
            "fafid",
            "weight_within_county",
            "lon",
            "lat",
        ]
    ]
    partC_rev1_write_table(out, partC_rev1_output_file(config, "partC_rev1_county_centroids_report.csv"), index=False)
    return out.reset_index(drop=True)
