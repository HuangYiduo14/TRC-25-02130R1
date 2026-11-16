"""Normalized Phase 3A overlay GeoPackage/QML writing and reopen smoke."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

import fiona
import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import LineString, Point


class Phase3AQGISError(RuntimeError):
    pass


def write_overlay_smoke(
    *,
    geopackage_path: str | Path,
    qml_path: str | Path,
    point_rows: pd.DataFrame,
    line_rows: pd.DataFrame,
) -> dict[str, Any]:
    """Write a tiny normalized EPSG:4269 overlay and reopen it three ways."""

    destination = Path(geopackage_path)
    style = Path(qml_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    style.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or style.exists():
        raise Phase3AQGISError("QGIS smoke outputs are write-new-only")
    required_point = {"candidate_id", "object_id", "object_type", "lon", "lat"}
    required_line = {"candidate_id", "object_id", "object_type", "tail_lon", "tail_lat", "head_lon", "head_lat"}
    if not required_point.issubset(point_rows.columns) or not required_line.issubset(line_rows.columns):
        raise Phase3AQGISError("QGIS smoke frames lack required stable-ID geometry fields")
    points = gpd.GeoDataFrame(
        point_rows.drop(columns=["lon", "lat"]).copy(),
        geometry=[Point(float(row.lon), float(row.lat)) for row in point_rows.itertuples(index=False)],
        crs="EPSG:4269",
    )
    lines = gpd.GeoDataFrame(
        line_rows.drop(columns=["tail_lon", "tail_lat", "head_lon", "head_lat"]).copy(),
        geometry=[
            LineString([(float(row.tail_lon), float(row.tail_lat)), (float(row.head_lon), float(row.head_lat))])
            for row in line_rows.itertuples(index=False)
        ],
        crs="EPSG:4269",
    )
    points.to_file(destination, layer="phase3a_object_points", driver="GPKG", index=False, engine="pyogrio")
    lines.to_file(destination, layer="phase3a_object_lines", driver="GPKG", index=False, engine="pyogrio", mode="a")
    style.write_text(_QML, encoding="utf-8", newline="\n")
    fiona_layers = sorted(fiona.listlayers(destination))
    pyogrio_layers = sorted(pyogrio.list_layers(destination)[:, 0].tolist())
    expected = ["phase3a_object_lines", "phase3a_object_points"]
    if fiona_layers != expected or pyogrio_layers != expected:
        raise Phase3AQGISError(f"GeoPackage layer reopen mismatch: fiona={fiona_layers}, pyogrio={pyogrio_layers}")
    reopened_points = gpd.read_file(destination, layer="phase3a_object_points", engine="fiona")
    reopened_lines = pyogrio.read_dataframe(destination, layer="phase3a_object_lines")
    if len(reopened_points) != len(points) or len(reopened_lines) != len(lines):
        raise Phase3AQGISError("GeoPackage reopen row counts differ")
    ogr_binding = "pyogrio_GDAL_OGR"
    osgeo_status: str
    if importlib.util.find_spec("osgeo") is not None:
        from osgeo import ogr

        data_source = ogr.Open(str(destination), 0)
        osgeo_status = "PASS" if data_source is not None and data_source.GetLayerCount() == 2 else "FAIL"
        data_source = None
    elif shutil.which("ogrinfo"):
        osgeo_status = "AVAILABLE_NOT_INVOKED"
    else:
        osgeo_status = "PYOGRIO_OGR_BINDING_PASS_OSGEO_PYTHON_UNAVAILABLE"
    return {
        "status": "PASS" if osgeo_status != "FAIL" else "FAIL",
        "geopackage_path": str(destination.resolve()),
        "geopackage_size_bytes": int(destination.stat().st_size),
        "crs": "EPSG:4269",
        "layers": expected,
        "row_counts": {"phase3a_object_points": len(points), "phase3a_object_lines": len(lines)},
        "fiona_reopen": "PASS",
        "pyogrio_reopen": "PASS",
        "ogr_reopen": osgeo_status,
        "ogr_binding": ogr_binding,
        "qml_path": str(style.resolve()),
        "qgis_desktop_required": False,
    }


_QML = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<qgis version=\"3\" styleCategories=\"Symbology\">
  <renderer-v2 type=\"categorizedSymbol\" attr=\"object_type\" symbollevels=\"0\"/>
</qgis>
"""
