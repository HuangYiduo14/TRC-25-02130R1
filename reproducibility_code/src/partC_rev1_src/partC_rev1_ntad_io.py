"""NTAD source resolution, inspection, and reading for Prompt 2.5."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from .partC_rev1_paths import partC_rev1_resolve_path


_PARTC_REV1_SOURCE_META = {
    "STRAHNET": {
        "priority": 1,
        "path_key": "strahnet_gpkg_path",
        "glob_key": "strahnet_gpkg_glob",
        "layer_key": "strahnet_layer",
        "sqlite_table": "partC_rev1_ntad_strahnet_segments",
    },
    "NHS": {
        "priority": 2,
        "path_key": "nhs_gpkg_path",
        "glob_key": "nhs_gpkg_glob",
        "layer_key": "nhs_layer",
        "sqlite_table": "partC_rev1_ntad_nhs_segments",
    },
    "NN": {
        "priority": 3,
        "path_key": "nn_gpkg_path",
        "glob_key": "nn_gpkg_glob",
        "layer_key": "nn_layer",
        "sqlite_table": "partC_rev1_ntad_nn_segments",
    },
}

_PARTC_REV1_FIELD_CANDIDATES = {
    "aadt": ["AADT", "AADT_TOTAL", "TOT_AADT", "AADT_TOT"],
    "truck_aadt": ["AADT_COM", "AADTCOM", "COM_AADT", "AADT_SINGL", "AADT_SINGLE", "AADTSINGL", "SINGLE_AADT"],
    "speed": ["SPEED_LIMI", "SPEED_LIMIT", "SPEED"],
    "route": ["SIGN1", "SIGNT1", "SIGNN1", "ROUTEID"],
    "name": ["LNAME", "ROAD_NAME", "NAME"],
    "class": ["FCLASS", "ACLASS", "FACILITYT"],
}


def _partC_rev1_list_layers_with_types(path: str | Path) -> list[tuple[str, str]]:
    import pyogrio

    layers = pyogrio.list_layers(str(path))
    if hasattr(layers, "tolist"):
        layers = layers.tolist()
    out: list[tuple[str, str]] = []
    for layer in layers:
        if isinstance(layer, (list, tuple)) and len(layer) >= 2:
            out.append((str(layer[0]), str(layer[1])))
        else:
            out.append((str(layer), ""))
    return out


def partC_rev1_list_gpkg_layers(path: str | Path) -> list[str]:
    gpkg_path = Path(path)
    if not gpkg_path.exists():
        raise FileNotFoundError(f"NTAD GeoPackage not found: {gpkg_path}")
    return [name for name, _ in _partC_rev1_list_layers_with_types(gpkg_path)]


def _partC_rev1_resolve_one_source(config: dict[str, Any], label: str, excluded_paths: set[Path]) -> dict[str, Any]:
    ntad_cfg = config.get("ntad", {})
    meta = _PARTC_REV1_SOURCE_META[label]
    configured_path = str(ntad_cfg.get(meta["path_key"], "") or "").strip()
    configured_glob = str(ntad_cfg.get(meta["glob_key"], "") or "").strip()
    layer = str(ntad_cfg.get(meta["layer_key"], "") or "").strip()

    candidates: list[Path] = []
    if configured_path:
        path = partC_rev1_resolve_path(configured_path)
        if not path.exists():
            raise FileNotFoundError(f"{label} NTAD GeoPackage not found: {path}")
        candidates = [path]
    elif configured_glob:
        pattern = str(partC_rev1_resolve_path(configured_glob))
        candidates = [Path(value).resolve() for value in glob.glob(pattern)]
        candidates = [path for path in candidates if path not in excluded_paths]
        if not candidates:
            raise FileNotFoundError(f"{label} NTAD glob found no candidates after exclusions: {configured_glob}")
        if len(candidates) > 1:
            joined = "\n".join(str(path) for path in sorted(candidates))
            raise ValueError(f"Multiple {label} NTAD candidates found; set ntad.{meta['path_key']} explicitly:\n{joined}")
    else:
        raise FileNotFoundError(f"No NTAD path or glob configured for {label}")

    resolved = candidates[0].resolve()
    layers = partC_rev1_list_gpkg_layers(resolved)
    if layer and layer not in layers:
        raise ValueError(f"Configured {label} layer {layer!r} not found in {resolved}; available={layers}")
    chosen_layer = layer or (layers[0] if len(layers) == 1 else "")
    if not chosen_layer:
        raise ValueError(f"Multiple layers found for {label}; configure ntad.{meta['layer_key']}: {layers}")

    warning = ""
    lower_name = resolved.name.lower()
    if label == "STRAHNET" and "strategic" not in lower_name and "strahnet" not in lower_name:
        warning = "Configured STRAHNET filename does not look strategic; respecting config label."
    if label == "NHS" and "national_highway_system" not in lower_name:
        warning = "Configured NHS filename does not look like National Highway System; respecting config label."
    if label == "NN" and "national_network" not in lower_name:
        warning = "Configured NN filename does not look like National Network; respecting config label."

    return {
        "source_label": label,
        "source_priority": meta["priority"],
        "configured_path": configured_path,
        "configured_glob": configured_glob,
        "resolved_path": str(resolved),
        "layer": chosen_layer,
        "warning": warning,
    }


def partC_rev1_resolve_ntad_source_paths(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    excluded: set[Path] = set()
    for label in ("STRAHNET", "NHS", "NN"):
        row = _partC_rev1_resolve_one_source(config, label, excluded)
        rows.append(row)
        if label == "STRAHNET":
            excluded.add(Path(row["resolved_path"]).resolve())
    return pd.DataFrame(rows)


def _partC_rev1_present_fields(columns: list[str], candidates: list[str]) -> list[str]:
    lower = {column.lower(): column for column in columns}
    return [lower[value.lower()] for value in candidates if value.lower() in lower]


def partC_rev1_detect_ntad_columns(columns: list[str]) -> dict[str, list[str]]:
    return {name: _partC_rev1_present_fields(columns, candidates) for name, candidates in _PARTC_REV1_FIELD_CANDIDATES.items()}


def partC_rev1_inspect_ntad_sources(config: dict[str, Any]) -> pd.DataFrame:
    import pyogrio

    registry = partC_rev1_resolve_ntad_source_paths(config)
    rows: list[dict[str, Any]] = []
    for row in registry.itertuples(index=False):
        info = pyogrio.read_info(row.resolved_path, layer=row.layer)
        fields = list(info.get("fields", []))
        dtypes = [str(value) for value in info.get("dtypes", [])]
        detected = partC_rev1_detect_ntad_columns(fields)
        rows.append(
            {
                "source_label": row.source_label,
                "source_priority": row.source_priority,
                "configured_path": row.configured_path,
                "configured_glob": row.configured_glob,
                "resolved_path": row.resolved_path,
                "available_layers": ";".join(partC_rev1_list_gpkg_layers(row.resolved_path)),
                "layer": row.layer,
                "row_count": info.get("features"),
                "crs": str(info.get("crs") or ""),
                "geometry_type": str(info.get("geometry_type") or ""),
                "column_names": ",".join(fields),
                "column_dtypes": ",".join(dtypes),
                "likely_route_name_fields": ",".join(detected["name"]),
                "likely_route_number_fields": ",".join(detected["route"]),
                "likely_aadt_fields": ",".join(detected["aadt"]),
                "likely_truck_aadt_fields": ",".join(detected["truck_aadt"]),
                "likely_speed_limit_fields": ",".join(detected["speed"]),
                "likely_classification_fields": ",".join(detected["class"]),
                "has_aadt": bool(detected["aadt"]),
                "has_truck_aadt": bool(detected["truck_aadt"]),
                "has_speed": bool(detected["speed"]),
                "has_route_fields": bool(detected["route"]),
                "has_name_fields": bool(detected["name"]),
                "warning": row.warning,
            }
        )
    return pd.DataFrame(rows)


def partC_rev1_read_ntad_source(config: dict[str, Any], source_label: str) -> gpd.GeoDataFrame:
    label = source_label.upper()
    registry = partC_rev1_resolve_ntad_source_paths(config)
    source_rows = registry.loc[registry["source_label"].eq(label)]
    if source_rows.empty:
        raise ValueError(f"Unknown NTAD source label: {source_label}")
    row = source_rows.iloc[0]
    gdf = gpd.read_file(row["resolved_path"], layer=row["layer"])
    gdf["partC_rev1_ntad_source_label"] = label
    gdf["partC_rev1_ntad_source_priority"] = int(row["source_priority"])
    gdf["partC_rev1_ntad_resolved_path"] = str(row["resolved_path"])
    gdf["partC_rev1_ntad_layer"] = str(row["layer"])
    return gdf


def partC_rev1_ntad_sqlite_table(source_label: str) -> str:
    return str(_PARTC_REV1_SOURCE_META[source_label.upper()]["sqlite_table"])
