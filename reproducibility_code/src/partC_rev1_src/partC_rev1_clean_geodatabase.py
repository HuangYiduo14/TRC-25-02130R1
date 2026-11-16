"""Write cleaned normalized geospatial outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from .partC_rev1_io import partC_rev1_write_table
from .partC_rev1_paths import partC_rev1_output_file, partC_rev1_resolve_path


_PARTC_REV1_LOG = logging.getLogger("partC_rev1")


def _partC_rev1_file_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _partC_rev1_layer_ready(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    for column in list(out.columns):
        if column == out.geometry.name:
            continue
        if out[column].dtype == "object":
            out[column] = out[column].map(_partC_rev1_file_scalar)
        elif str(out[column].dtype) == "boolean":
            out[column] = out[column].astype("Int64")
    return out


def _partC_rev1_write_gpkg_layer(gdf: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    _PARTC_REV1_LOG.info("Writing GeoPackage layer %s rows=%s path=%s", layer, len(gdf), path)
    ready = _partC_rev1_layer_ready(gdf)
    ready.to_file(path, layer=layer, driver="GPKG")


def partC_rev1_write_clean_geodatabase(
    clean_nodes_gdf: gpd.GeoDataFrame,
    clean_links_gdf: gpd.GeoDataFrame,
    directed_arcs_gdf: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    clean_cfg = config.get("clean_geodatabase", {})
    if not bool(clean_cfg.get("write_clean_geodatabase", True)):
        report = pd.DataFrame([{"format": "none", "path": "", "status": "disabled", "details": ""}])
        partC_rev1_write_table(report, partC_rev1_output_file(config, "partC_rev1_clean_geodatabase_report.csv"), index=False)
        return {"clean_geodatabase_path": None, "status": "disabled"}

    layers = clean_cfg.get("layers", {})
    gpkg_path = partC_rev1_resolve_path(str(clean_cfg.get("output_gpkg_path", "partC_rev1_cache/partC_rev1_clean_network.gpkg")))
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    if gpkg_path.exists():
        _PARTC_REV1_LOG.info("Removing existing GeoPackage before rewrite: %s", gpkg_path)
        gpkg_path.unlink()

    rows: list[dict[str, Any]] = []
    for label, gdf in [
        (str(layers.get("clean_nodes", "partC_rev1_clean_physical_nodes")), clean_nodes_gdf),
        (str(layers.get("clean_links", "partC_rev1_clean_physical_links")), clean_links_gdf),
        (str(layers.get("directed_arcs", "partC_rev1_directed_arcs")), directed_arcs_gdf),
    ]:
        _partC_rev1_write_gpkg_layer(gdf, gpkg_path, label)
        rows.append({"format": "gpkg", "path": str(gpkg_path), "layer": label, "status": "written", "details": f"rows={len(gdf)}"})

    if bool(clean_cfg.get("attempt_filegdb_if_supported", False)):
        filegdb_path = partC_rev1_resolve_path(str(clean_cfg.get("output_filegdb_path", "partC_rev1_cache/partC_rev1_clean_network.gdb")))
        rows.append(
            {
                "format": "filegdb",
                "path": str(filegdb_path),
                "layer": "",
                "status": "not_attempted",
                "details": "Optional FileGDB writing is GDAL-build dependent; GeoPackage written",
            }
        )

    report = pd.DataFrame(rows)
    partC_rev1_write_table(report, partC_rev1_output_file(config, "partC_rev1_clean_geodatabase_report.csv"), index=False)
    return {"clean_geodatabase_path": str(gpkg_path), "status": "written", "report": report}
