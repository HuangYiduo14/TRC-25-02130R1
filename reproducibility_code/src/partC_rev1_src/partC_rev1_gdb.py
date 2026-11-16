"""Geodatabase inspection helpers for partC_rev1."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pandas as pd


_PARTC_REV1_GEO_INSTALL_MSG = (
    "Install geospatial dependencies with geopandas plus fiona or pyogrio "
    "to read FileGDB/GPKG layers."
)


def _partC_rev1_import_geopandas():
    try:
        return importlib.import_module("geopandas")
    except Exception as exc:  # pragma: no cover - exact dependency failure varies by platform.
        raise ImportError(_PARTC_REV1_GEO_INSTALL_MSG) from exc


def _partC_rev1_import_optional(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def partC_rev1_list_gdb_layers(gdb_path: str) -> list[str]:
    path = Path(gdb_path)
    if not path.exists():
        raise FileNotFoundError(f"Geodatabase not found: {gdb_path}")

    fiona = _partC_rev1_import_optional("fiona")
    if fiona is not None:
        try:
            return list(fiona.listlayers(str(path)))
        except Exception:
            pass

    pyogrio = _partC_rev1_import_optional("pyogrio")
    if pyogrio is not None:
        try:
            layers = pyogrio.list_layers(str(path))
            if hasattr(layers, "tolist"):
                layers = layers.tolist()
            names: list[str] = []
            for layer in layers:
                names.append(str(layer[0] if isinstance(layer, (list, tuple)) else layer))
            return names
        except Exception:
            pass

    geopandas = _partC_rev1_import_geopandas()
    if hasattr(geopandas, "list_layers"):
        layers_df = geopandas.list_layers(str(path))
        return [str(value) for value in layers_df["name"].tolist()]
    raise ImportError(_PARTC_REV1_GEO_INSTALL_MSG)


def partC_rev1_read_gdb_layer(gdb_path: str, layer: str, rows: int | None = None):
    path = Path(gdb_path)
    if not path.exists():
        raise FileNotFoundError(f"Geodatabase not found: {gdb_path}")
    geopandas = _partC_rev1_import_geopandas()
    kwargs: dict[str, Any] = {"layer": layer}
    if rows is not None:
        kwargs["rows"] = rows
    try:
        gdf = geopandas.read_file(str(path), **kwargs)
    except TypeError:
        kwargs.pop("rows", None)
        gdf = geopandas.read_file(str(path), **kwargs)
    if rows is not None:
        return gdf.head(rows)
    return gdf


def _partC_rev1_layer_metadata(gdb_path: str, layer: str) -> dict[str, Any]:
    fiona = _partC_rev1_import_optional("fiona")
    if fiona is not None:
        try:
            with fiona.open(str(gdb_path), layer=layer) as src:
                return {
                    "layer": layer,
                    "feature_count": len(src),
                    "geometry_type": src.schema.get("geometry"),
                    "crs": src.crs_wkt or (str(src.crs) if src.crs else None),
                    "columns": list(src.schema.get("properties", {}).keys()),
                }
        except Exception:
            pass

    gdf = partC_rev1_read_gdb_layer(gdb_path, layer, rows=5)
    geometry_type = None
    if "geometry" in gdf.columns and not gdf.empty:
        geometry_type = str(gdf.geometry.geom_type.dropna().mode().iloc[0])
    return {
        "layer": layer,
        "feature_count": None,
        "geometry_type": geometry_type,
        "crs": str(gdf.crs) if getattr(gdf, "crs", None) is not None else None,
        "columns": [column for column in gdf.columns if column != "geometry"],
    }


def _partC_rev1_score_layer(metadata: dict[str, Any], role: str) -> float:
    geometry = str(metadata.get("geometry_type") or "").lower()
    columns = {str(column).lower() for column in metadata.get("columns", [])}
    layer_name = str(metadata.get("layer") or "").lower()
    score = 0.0

    if role == "link":
        if "line" in geometry:
            score += 0.45
        score += 0.08 if "id" in columns else 0.0
        score += 0.12 if "dir" in columns else 0.0
        score += 0.10 if "length" in columns else 0.0
        score += 0.10 if "class" in columns else 0.0
        score += 0.08 if "nhs" in columns else 0.0
        score += 0.04 if "link" in layer_name else 0.0
        score += 0.03 if {"ab_lanes", "ba_lanes"} & columns else 0.0
    elif role == "node":
        if "point" in geometry:
            score += 0.45
        score += 0.08 if "id" in columns else 0.0
        score += 0.14 if "centroid" in columns else 0.0
        score += 0.12 if "centroidid" in columns else 0.0
        score += 0.10 if "fafid" in columns else 0.0
        score += 0.04 if "node" in layer_name else 0.0
    return min(score, 1.0)


def partC_rev1_inspect_gdb(gdb_path: str) -> pd.DataFrame:
    layers = partC_rev1_list_gdb_layers(gdb_path)
    rows: list[dict[str, Any]] = []
    for layer in layers:
        metadata = _partC_rev1_layer_metadata(gdb_path, layer)
        rows.append(
            {
                "layer": layer,
                "geometry_type": metadata.get("geometry_type"),
                "crs": metadata.get("crs"),
                "feature_count": metadata.get("feature_count"),
                "columns": ",".join(metadata.get("columns", [])),
                "link_score": _partC_rev1_score_layer(metadata, "link"),
                "node_score": _partC_rev1_score_layer(metadata, "node"),
            }
        )
    return pd.DataFrame(rows)


def _partC_rev1_unique_best(candidates: list[dict[str, Any]]) -> tuple[str | None, bool]:
    if not candidates:
        return None, False
    ordered = sorted(candidates, key=lambda item: item["confidence"], reverse=True)
    best = ordered[0]
    if best["confidence"] <= 0:
        return None, False
    if len(ordered) > 1 and abs(best["confidence"] - ordered[1]["confidence"]) < 1e-9:
        return None, True
    return str(best["layer"]), False


def partC_rev1_guess_link_node_layers(gdb_path: str) -> dict[str, Any]:
    inspection = partC_rev1_inspect_gdb(gdb_path)
    link_candidates = [
        {
            "layer": row.layer,
            "geometry_type": row.geometry_type,
            "confidence": float(row.link_score),
        }
        for row in inspection.itertuples(index=False)
        if float(row.link_score) > 0
    ]
    node_candidates = [
        {
            "layer": row.layer,
            "geometry_type": row.geometry_type,
            "confidence": float(row.node_score),
        }
        for row in inspection.itertuples(index=False)
        if float(row.node_score) > 0
    ]
    link_candidates.sort(key=lambda item: item["confidence"], reverse=True)
    node_candidates.sort(key=lambda item: item["confidence"], reverse=True)

    links_layer, ambiguous_links = _partC_rev1_unique_best(link_candidates)
    nodes_layer, ambiguous_nodes = _partC_rev1_unique_best(node_candidates)
    return {
        "links_layer": links_layer,
        "nodes_layer": nodes_layer,
        "ambiguous_links": ambiguous_links,
        "ambiguous_nodes": ambiguous_nodes,
        "link_candidates": link_candidates,
        "node_candidates": node_candidates,
    }
