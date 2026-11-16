"""Geometry, route, and name based NTAD/FAF matching for Prompt 2.5."""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point

from .partC_rev1_road_name_normalization import (
    partC_rev1_extract_faf_normalized_sign_key,
    partC_rev1_extract_faf_route_key,
    partC_rev1_extract_ntad_normalized_sign_key,
    partC_rev1_extract_ntad_route_key,
    partC_rev1_name_similarity,
    partC_rev1_normalize_road_name,
)


_PARTC_REV1_LOG = logging.getLogger("partC_rev1")
PARTC_REV1_MILES_TO_METERS = 1609.344


def _partC_rev1_value(row: Any, *names: str) -> Any:
    if isinstance(row, dict):
        lower = {str(key).lower(): key for key in row.keys()}
        for name in names:
            key = lower.get(name.lower())
            if key is not None:
                return row[key]
        return None
    if isinstance(row, pd.Series):
        lower = {str(key).lower(): key for key in row.index}
        for name in names:
            key = lower.get(name.lower())
            if key is not None:
                return row[key]
        return None
    for name in names:
        if hasattr(row, name):
            return getattr(row, name)
    return None


def _partC_rev1_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _partC_rev1_normalize_fips(value: Any, width: int) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    elif "." in text:
        try:
            number = float(text)
            if number.is_integer():
                text = str(int(number))
        except ValueError:
            pass
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    if len(digits) <= width:
        return digits.zfill(width)
    return digits


def _partC_rev1_fips_score(faf_stfips: Any, faf_ctfips: Any, ntad_stfips: Any, ntad_ctfips: Any) -> float:
    faf_state = _partC_rev1_normalize_fips(faf_stfips, 2)
    ntad_state = _partC_rev1_normalize_fips(ntad_stfips, 2)
    faf_county = _partC_rev1_normalize_fips(faf_ctfips, 5)
    ntad_county = _partC_rev1_normalize_fips(ntad_ctfips, 5)
    if faf_county and ntad_county and faf_county == ntad_county:
        return 1.0
    if faf_state and ntad_state and faf_state == ntad_state:
        return 0.5
    return 0.0


def _partC_rev1_main_line(geom: Any) -> LineString | None:
    if geom is None or getattr(geom, "is_empty", True):
        return None
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        parts = list(geom.geoms)
        if not parts:
            return None
        return max(parts, key=lambda part: part.length)
    if hasattr(geom, "geoms"):
        lines = [part for part in geom.geoms if isinstance(part, LineString)]
        if lines:
            return max(lines, key=lambda part: part.length)
    return None


def _partC_rev1_endpoints(geom: Any) -> tuple[Point, Point] | None:
    line = _partC_rev1_main_line(geom)
    if line is None:
        return None
    coords = list(line.coords)
    if len(coords) < 2:
        return None
    return Point(coords[0]), Point(coords[-1])


def _partC_rev1_midpoint(geom: Any) -> Point | None:
    line = _partC_rev1_main_line(geom)
    if line is None:
        return None
    return line.interpolate(0.5, normalized=True)


def _partC_rev1_heading_deg(geom: Any) -> float | None:
    endpoints = _partC_rev1_endpoints(geom)
    if endpoints is None:
        return None
    start, end = endpoints
    dx = end.x - start.x
    dy = end.y - start.y
    if dx == 0 and dy == 0:
        return None
    return (math.degrees(math.atan2(dy, dx)) + 360.0) % 180.0


def partC_rev1_angle_diff_deg(left_geom: Any, right_geom: Any) -> float:
    left = _partC_rev1_heading_deg(left_geom)
    right = _partC_rev1_heading_deg(right_geom)
    if left is None or right is None:
        return 180.0
    diff = abs(left - right)
    return float(min(diff, 180.0 - diff))


def partC_rev1_endpoint_distance_m(left_geom: Any, right_geom: Any) -> float:
    left = _partC_rev1_endpoints(left_geom)
    right = _partC_rev1_endpoints(right_geom)
    if left is None or right is None:
        return float("inf")
    return float(min(a.distance(b) for a in left for b in right))


def partC_rev1_line_distance_m(left_geom: Any, right_geom: Any) -> float:
    left_line = _partC_rev1_main_line(left_geom)
    right_line = _partC_rev1_main_line(right_geom)
    if left_line is None or right_line is None:
        return float("inf")
    left_points = [Point(left_line.coords[0]), left_line.interpolate(0.5, normalized=True), Point(left_line.coords[-1])]
    right_mid = right_line.interpolate(0.5, normalized=True)
    distances = [point.distance(right_line) for point in left_points]
    distances.append(right_mid.distance(left_line))
    return float(np.mean(distances))


def partC_rev1_overlap_score(left_geom: Any, right_geom: Any, buffer_m: float) -> float:
    left_line = _partC_rev1_main_line(left_geom)
    right_line = _partC_rev1_main_line(right_geom)
    if left_line is None or right_line is None:
        return 0.0
    try:
        left_buffer = left_line.buffer(buffer_m)
        right_buffer = right_line.buffer(buffer_m)
        denom = min(left_buffer.area, right_buffer.area)
        if denom <= 0:
            return 0.0
        return float(max(0.0, min(1.0, left_buffer.intersection(right_buffer).area / denom)))
    except Exception:
        return 0.0


def partC_rev1_buffer_intersects(left_geom: Any, right_geom: Any, buffer_m: float) -> bool:
    left_line = _partC_rev1_main_line(left_geom)
    right_line = _partC_rev1_main_line(right_geom)
    if left_line is None or right_line is None:
        return False
    try:
        return bool(left_line.distance(right_line) <= (2.0 * buffer_m + 1e-9))
    except Exception:
        return False


def _partC_rev1_geo_buffer_radius_m(config: dict[str, Any]) -> float:
    matching = config.get("matching", {})
    if matching.get("geo_buffer_radius_m") is not None:
        return float(matching.get("geo_buffer_radius_m"))
    return float(matching.get("geo_buffer_radius_miles", 0.08)) * PARTC_REV1_MILES_TO_METERS


def _partC_rev1_decay(distance_m: float, scale_m: float) -> float:
    if not np.isfinite(distance_m):
        return 0.0
    return float(math.exp(-max(0.0, distance_m) / max(scale_m, 1e-9)))


def _partC_rev1_ntad_uid(row: pd.Series, source_label: str) -> str:
    values = [
        source_label,
        str(_partC_rev1_value(row, "OBJECTID", "ID") or ""),
        str(_partC_rev1_value(row, "ROUTEID") or ""),
        str(_partC_rev1_value(row, "BEGINPOINT") or ""),
        str(_partC_rev1_value(row, "ENDPOINT") or ""),
        str(_partC_rev1_value(row, "SIGN1") or ""),
        str(_partC_rev1_value(row, "LNAME") or ""),
    ]
    geom = _partC_rev1_value(row, "geometry")
    if geom is not None:
        values.append(getattr(geom, "wkb_hex", ""))
    digest = hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()[:20]
    return f"{source_label}_{digest}"


def _partC_rev1_first_present(row: pd.Series, candidates: list[str]) -> Any:
    return _partC_rev1_value(row, *candidates)


def partC_rev1_prepare_faf_links(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["partC_rev1_match_link_uid"] = out.get("partC_rev1_link_uid", out.get("physical_link_id", out.get("ID", out.index))).astype(str)
    out["partC_rev1_faf_route_key"] = out.apply(partC_rev1_extract_faf_route_key, axis=1)
    out["partC_rev1_faf_normalized_sign_key"] = out.apply(partC_rev1_extract_faf_normalized_sign_key, axis=1)
    name_source = out.get("Road_Name", out.get("road_name", out.get("partC_rev1_road_name", pd.Series([""] * len(out), index=out.index))))
    out["partC_rev1_faf_road_name_norm"] = name_source.map(partC_rev1_normalize_road_name)
    out["partC_rev1_faf_stfips"] = out.apply(
        lambda row: _partC_rev1_normalize_fips(_partC_rev1_first_present(row, ["STFIPS", "state_fips", "STATEFP", "partC_rev1_stfips"]), 2),
        axis=1,
    )
    out["partC_rev1_faf_ctfips"] = out.apply(
        lambda row: _partC_rev1_normalize_fips(_partC_rev1_first_present(row, ["CTFIPS", "county_fips", "partC_rev1_ctfips"]), 5),
        axis=1,
    )
    return out


def partC_rev1_prepare_ntad_segments(gdf: gpd.GeoDataFrame, source_label: str, source_priority: int) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["partC_rev1_ntad_source_label"] = source_label
    out["partC_rev1_ntad_source_priority"] = source_priority
    out["partC_rev1_ntad_uid"] = out.apply(lambda row: _partC_rev1_ntad_uid(row, source_label), axis=1)
    out["partC_rev1_ntad_route_key"] = out.apply(partC_rev1_extract_ntad_route_key, axis=1)
    out["partC_rev1_ntad_normalized_sign_key"] = out.apply(partC_rev1_extract_ntad_normalized_sign_key, axis=1)
    name_source = out.get("LNAME", out.get("ROAD_NAME", out.get("NAME", pd.Series([""] * len(out), index=out.index))))
    out["partC_rev1_ntad_road_name_norm"] = name_source.map(partC_rev1_normalize_road_name)
    out["partC_rev1_ntad_aadt_total"] = pd.to_numeric(
        out.apply(lambda row: _partC_rev1_first_present(row, ["AADT", "AADT_TOTAL", "TOT_AADT", "AADT_TOT"]), axis=1),
        errors="coerce",
    )
    out["partC_rev1_ntad_aadt_combination"] = pd.to_numeric(
        out.apply(lambda row: _partC_rev1_first_present(row, ["AADT_COM", "AADTCOM", "COM_AADT"]), axis=1),
        errors="coerce",
    )
    out["partC_rev1_ntad_aadt_single"] = pd.to_numeric(
        out.apply(lambda row: _partC_rev1_first_present(row, ["AADT_SINGL", "AADT_SINGLE", "AADTSINGL", "SINGLE_AADT"]), axis=1),
        errors="coerce",
    )
    out["partC_rev1_ntad_speed_limit_mph"] = pd.to_numeric(
        out.apply(lambda row: _partC_rev1_first_present(row, ["SPEED_LIMI", "SPEED_LIMIT", "SPEED"]), axis=1),
        errors="coerce",
    )
    for target, candidates in {
        "partC_rev1_ntad_objectid": ["OBJECTID", "ID"],
        "partC_rev1_ntad_lname": ["LNAME", "ROAD_NAME", "NAME"],
        "partC_rev1_ntad_sign1": ["SIGN1"],
        "partC_rev1_ntad_signt1": ["SIGNT1"],
        "partC_rev1_ntad_signn1": ["SIGNN1"],
        "partC_rev1_ntad_routeid": ["ROUTEID"],
        "partC_rev1_ntad_nhs": ["NHS", "NN"],
        "partC_rev1_ntad_fclass": ["FCLASS", "ACLASS"],
        "partC_rev1_ntad_stfips": ["STFIPS", "STATEFP", "STATE_FIPS"],
        "partC_rev1_ntad_ctfips": ["CTFIPS", "COUNTYFP", "COUNTY_FIPS"],
    }.items():
        out[target] = out.apply(lambda row, cols=candidates: _partC_rev1_first_present(row, cols), axis=1)
    out["partC_rev1_ntad_stfips"] = out["partC_rev1_ntad_stfips"].map(lambda value: _partC_rev1_normalize_fips(value, 2))
    out["partC_rev1_ntad_ctfips"] = out["partC_rev1_ntad_ctfips"].map(lambda value: _partC_rev1_normalize_fips(value, 5))
    return out


def partC_rev1_score_candidate(faf_row: Any, ntad_row: Any, config: dict[str, Any], source_label: str) -> dict[str, Any]:
    matching = config.get("matching", {})
    faf_geom = _partC_rev1_value(faf_row, "geometry")
    ntad_geom = _partC_rev1_value(ntad_row, "geometry")
    faf_route = _partC_rev1_value(faf_row, "partC_rev1_faf_route_key") or partC_rev1_extract_faf_route_key(faf_row)
    ntad_route = _partC_rev1_value(ntad_row, "partC_rev1_ntad_route_key") or partC_rev1_extract_ntad_route_key(ntad_row)
    faf_sign_key = _partC_rev1_value(faf_row, "partC_rev1_faf_normalized_sign_key") or partC_rev1_extract_faf_normalized_sign_key(faf_row)
    ntad_sign_key = _partC_rev1_value(ntad_row, "partC_rev1_ntad_normalized_sign_key") or partC_rev1_extract_ntad_normalized_sign_key(ntad_row)
    normalized_sign_match = bool(faf_sign_key and ntad_sign_key and str(faf_sign_key) == str(ntad_sign_key))
    route_similarity = 1.0 if (faf_route and ntad_route and str(faf_route) == str(ntad_route)) or normalized_sign_match else 0.0
    faf_name = _partC_rev1_value(faf_row, "partC_rev1_faf_road_name_norm", "Road_Name", "road_name", "partC_rev1_road_name")
    ntad_name = _partC_rev1_value(ntad_row, "partC_rev1_ntad_road_name_norm", "LNAME", "partC_rev1_ntad_lname")
    name_similarity = partC_rev1_name_similarity(str(faf_name or ""), str(ntad_name or ""))
    endpoint_distance = partC_rev1_endpoint_distance_m(faf_geom, ntad_geom)
    line_distance = partC_rev1_line_distance_m(faf_geom, ntad_geom)
    angle_diff = partC_rev1_angle_diff_deg(faf_geom, ntad_geom)
    endpoint_score = _partC_rev1_decay(endpoint_distance, float(matching.get("endpoint_distance_scale_m", 100.0)))
    line_score = _partC_rev1_decay(line_distance, float(matching.get("line_distance_scale_m", 100.0)))
    angle_tolerance = float(matching.get("angle_tolerance_deg", 30.0))
    angle_score = max(0.0, 1.0 - angle_diff / max(angle_tolerance, 1e-9))
    geo_buffer_radius_m = _partC_rev1_geo_buffer_radius_m(config)
    direct_distance = _partC_rev1_direct_geometry_distance_m(faf_geom, ntad_geom)
    geo_buffer_intersects = partC_rev1_buffer_intersects(faf_geom, ntad_geom, geo_buffer_radius_m)
    overlap = partC_rev1_overlap_score(faf_geom, ntad_geom, geo_buffer_radius_m)
    faf_stfips = _partC_rev1_normalize_fips(_partC_rev1_value(faf_row, "partC_rev1_faf_stfips", "STFIPS", "state_fips", "STATEFP"), 2)
    faf_ctfips = _partC_rev1_normalize_fips(_partC_rev1_value(faf_row, "partC_rev1_faf_ctfips", "CTFIPS", "county_fips", "partC_rev1_ctfips"), 5)
    ntad_stfips = _partC_rev1_normalize_fips(_partC_rev1_value(ntad_row, "partC_rev1_ntad_stfips", "STFIPS", "STATEFP"), 2)
    ntad_ctfips = _partC_rev1_normalize_fips(_partC_rev1_value(ntad_row, "partC_rev1_ntad_ctfips", "CTFIPS", "COUNTYFP"), 5)
    fips_score = _partC_rev1_fips_score(faf_stfips, faf_ctfips, ntad_stfips, ntad_ctfips)
    semantic_score = max(route_similarity, name_similarity)
    strahnet_flag = bool(_partC_rev1_value(faf_row, "partC_rev1_faf_strahnet_nonempty"))
    class_score = 1.0 if source_label == "STRAHNET" and strahnet_flag else 0.8 if source_label == "NHS" else 0.6 if source_label == "NN" else 0.0
    weights = {
        "buffer_overlap": float(matching.get("score_weight_buffer_overlap", 0.35)),
        "line_distance": float(matching.get("score_weight_line_distance", 0.25)),
        "angle_parallel": float(matching.get("score_weight_angle_parallel", 0.20)),
        "endpoint_distance": float(matching.get("score_weight_endpoint_distance", 0.10)),
        "semantic": float(matching.get("score_weight_semantic", 0.07)),
        "fips": float(matching.get("score_weight_fips", 0.03)),
    }
    total = (
        weights["buffer_overlap"] * overlap
        + weights["line_distance"] * line_score
        + weights["angle_parallel"] * angle_score
        + weights["endpoint_distance"] * endpoint_score
        + weights["semantic"] * semantic_score
        + weights["fips"] * fips_score
    )
    return {
        "partC_rev1_match_dataset": source_label,
        "partC_rev1_match_source_priority": _partC_rev1_value(ntad_row, "partC_rev1_ntad_source_priority"),
        "partC_rev1_match_score": float(total),
        "partC_rev1_match_angle_diff_deg": float(angle_diff),
        "partC_rev1_match_endpoint_distance_m": float(endpoint_distance),
        "partC_rev1_match_line_distance_m": float(line_distance),
        "partC_rev1_match_min_distance_m": float(min(endpoint_distance, line_distance)),
        "partC_rev1_match_name_similarity": float(name_similarity),
        "partC_rev1_match_route_similarity": float(route_similarity),
        "partC_rev1_match_semantic_score": float(semantic_score),
        "partC_rev1_match_overlap_score": float(overlap),
        "partC_rev1_match_endpoint_distance_score": float(endpoint_score),
        "partC_rev1_match_line_distance_score": float(line_score),
        "partC_rev1_match_angle_parallel_score": float(angle_score),
        "partC_rev1_match_classification_score": float(class_score),
        "partC_rev1_geo_buffer_radius_m": float(geo_buffer_radius_m),
        "partC_rev1_geo_buffer_intersects": bool(geo_buffer_intersects),
        "partC_rev1_geo_buffer_overlap_score": float(overlap),
        "partC_rev1_geo_direct_distance_m": float(direct_distance),
        "partC_rev1_faf_stfips": faf_stfips,
        "partC_rev1_faf_ctfips": faf_ctfips,
        "partC_rev1_ntad_stfips": ntad_stfips,
        "partC_rev1_ntad_ctfips": ntad_ctfips,
        "partC_rev1_match_fips_score": float(fips_score),
        "partC_rev1_match_close_competitor": False,
        "partC_rev1_ntad_uid": _partC_rev1_value(ntad_row, "partC_rev1_ntad_uid"),
        "partC_rev1_ntad_objectid": _partC_rev1_value(ntad_row, "partC_rev1_ntad_objectid"),
        "partC_rev1_ntad_lname": _partC_rev1_value(ntad_row, "partC_rev1_ntad_lname"),
        "partC_rev1_ntad_sign1": _partC_rev1_value(ntad_row, "partC_rev1_ntad_sign1"),
        "partC_rev1_ntad_signt1": _partC_rev1_value(ntad_row, "partC_rev1_ntad_signt1"),
        "partC_rev1_ntad_signn1": _partC_rev1_value(ntad_row, "partC_rev1_ntad_signn1"),
        "partC_rev1_ntad_routeid": _partC_rev1_value(ntad_row, "partC_rev1_ntad_routeid"),
        "partC_rev1_ntad_nhs": _partC_rev1_value(ntad_row, "partC_rev1_ntad_nhs"),
        "partC_rev1_ntad_source_label": source_label,
        "partC_rev1_ntad_aadt_total": _partC_rev1_float(_partC_rev1_value(ntad_row, "partC_rev1_ntad_aadt_total")),
        "partC_rev1_ntad_aadt_combination": _partC_rev1_float(_partC_rev1_value(ntad_row, "partC_rev1_ntad_aadt_combination")),
        "partC_rev1_ntad_aadt_single": _partC_rev1_float(_partC_rev1_value(ntad_row, "partC_rev1_ntad_aadt_single")),
        "partC_rev1_ntad_speed_limit_mph": _partC_rev1_float(_partC_rev1_value(ntad_row, "partC_rev1_ntad_speed_limit_mph")),
        "partC_rev1_faf_route_key": faf_route,
        "partC_rev1_ntad_route_key": ntad_route,
        "partC_rev1_faf_normalized_sign_key": faf_sign_key,
        "partC_rev1_ntad_normalized_sign_key": ntad_sign_key,
        "partC_rev1_normalized_sign_match": normalized_sign_match,
        "partC_rev1_normalized_sign_global_fallback": False,
        "partC_rev1_normalized_sign_selected_distance_m": np.nan,
    }


def partC_rev1_match_method_for_score(score: dict[str, Any], faf_has_route_or_name: bool, config: dict[str, Any]) -> str | None:
    matching = config.get("matching", {})
    def numeric(key: str, default: float) -> float:
        value = score.get(key)
        try:
            if value is None or pd.isna(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    route_sim = numeric("partC_rev1_match_route_similarity", 0.0)
    name_sim = numeric("partC_rev1_match_name_similarity", 0.0)
    raw_buffer_intersects = score.get("partC_rev1_geo_buffer_intersects")
    buffer_intersects = False if raw_buffer_intersects is None or pd.isna(raw_buffer_intersects) else bool(raw_buffer_intersects)
    if not buffer_intersects:
        return None
    if route_sim >= float(matching.get("min_name_or_route_similarity", 0.65)):
        return "route_geometry"
    if name_sim >= float(matching.get("min_name_or_route_similarity", 0.65)):
        return "name_geometry"
    return "spatial_only"


def partC_rev1_select_best_candidate(
    scores: list[dict[str, Any]],
    faf_has_route_or_name: bool,
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    ordered = sorted(scores, key=lambda item: float(item.get("partC_rev1_match_score", 0.0) or 0.0), reverse=True)
    for item in ordered:
        item["partC_rev1_match_method"] = partC_rev1_match_method_for_score(item, faf_has_route_or_name, config)
    acceptable = [item for item in ordered if item.get("partC_rev1_match_method")]
    if not acceptable:
        return None, ordered
    best = acceptable[0].copy()
    second_score = float(acceptable[1]["partC_rev1_match_score"]) if len(acceptable) > 1 else 0.0
    best_score = max(float(best["partC_rev1_match_score"]), 1e-12)
    ambiguity_ratio = second_score / best_score
    threshold = float(config.get("matching", {}).get("close_competitor_ratio_threshold", config.get("matching", {}).get("ambiguity_ratio_threshold", 0.95)))
    best["partC_rev1_ntad_match_status"] = "matched"
    best["partC_rev1_match_second_score"] = second_score
    best["partC_rev1_match_ambiguity_ratio"] = ambiguity_ratio
    best["partC_rev1_match_close_competitor"] = bool(len(acceptable) > 1 and ambiguity_ratio > threshold)
    best["partC_rev1_match_spatial_only"] = best.get("partC_rev1_match_method") == "spatial_only"
    return best, ordered


def _partC_rev1_expanded_bounds(geom: Any, radius: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = geom.bounds
    return minx - radius, miny - radius, maxx + radius, maxy + radius


def _partC_rev1_candidate_indices(
    faf_row: pd.Series,
    source: gpd.GeoDataFrame,
    source_sindex: Any,
    config: dict[str, Any],
) -> list[int]:
    geom = faf_row.geometry
    if geom is None or getattr(geom, "is_empty", True):
        return []
    matching = config.get("matching", {})
    buffer_radius = _partC_rev1_geo_buffer_radius_m(config)
    rough_indices = list(source_sindex.intersection(_partC_rev1_expanded_bounds(geom, buffer_radius * 2.0)))
    candidate_indices = [
        int(idx)
        for idx in rough_indices
        if partC_rev1_buffer_intersects(geom, source.geometry.iloc[int(idx)], buffer_radius)
    ]
    max_candidates = int(matching.get("max_geometry_candidates_per_link", matching.get("max_candidates_per_link", 120)) or 120)
    if len(candidate_indices) > max_candidates:
        distances = [(int(idx), _partC_rev1_direct_geometry_distance_m(geom, source.geometry.iloc[int(idx)])) for idx in candidate_indices]
        candidate_indices = [idx for idx, _ in sorted(distances, key=lambda item: item[1])[:max_candidates]]
    return candidate_indices


def _partC_rev1_direct_geometry_distance_m(left_geom: Any, right_geom: Any) -> float:
    try:
        if left_geom is None or right_geom is None:
            return float("inf")
        return float(left_geom.distance(right_geom))
    except Exception:
        return float("inf")


def partC_rev1_match_single_source(
    faf_links_projected: gpd.GeoDataFrame,
    ntad_projected: gpd.GeoDataFrame,
    source_label: str,
    source_priority: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match a set of currently-unassigned FAF links against one NTAD source."""

    if faf_links_projected.empty:
        return pd.DataFrame(), pd.DataFrame()
    source = ntad_projected.reset_index(drop=True).copy()
    source["partC_rev1_ntad_source_priority"] = source_priority
    source_sindex = source.sindex
    best_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    keep_candidates = int(config.get("matching", {}).get("keep_candidate_rows_per_source", 2) or 2)
    total = len(faf_links_projected)
    _PARTC_REV1_LOG.info("Matching %s FAF links against %s source rows for %s", total, len(source), source_label)

    for ordinal, (_, faf_row) in enumerate(faf_links_projected.iterrows(), start=1):
        if ordinal % 25000 == 0:
            _PARTC_REV1_LOG.info("Matched source %s progress: %s/%s links", source_label, ordinal, total)
        link_uid = str(faf_row.get("partC_rev1_match_link_uid", faf_row.get("partC_rev1_link_uid", "")))
        faf_has_route_or_name = bool(faf_row.get("partC_rev1_faf_route_key")) or bool(faf_row.get("partC_rev1_faf_road_name_norm"))
        indices = _partC_rev1_candidate_indices(faf_row, source, source_sindex, config)
        scores = [partC_rev1_score_candidate(faf_row, source.iloc[int(idx)], config, source_label) for idx in indices]
        best, ordered = partC_rev1_select_best_candidate(scores, faf_has_route_or_name, config)
        for rank, candidate in enumerate(ordered[:keep_candidates], start=1):
            candidate_rows.append({"partC_rev1_link_uid": link_uid, "candidate_rank": rank, **candidate})
        if best is not None:
            best_rows.append({"partC_rev1_link_uid": link_uid, **best})
    return pd.DataFrame(best_rows), pd.DataFrame(candidate_rows)


def partC_rev1_empty_best_matches(link_uids: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "partC_rev1_link_uid": link_uids.astype(str),
            "partC_rev1_ntad_match_status": "unassigned",
            "partC_rev1_match_dataset": "NONE",
            "partC_rev1_match_method": "unassigned",
            "partC_rev1_match_score": np.nan,
            "partC_rev1_match_second_score": np.nan,
            "partC_rev1_match_ambiguity_ratio": np.nan,
            "partC_rev1_match_close_competitor": False,
            "partC_rev1_faf_normalized_sign_key": pd.NA,
            "partC_rev1_ntad_normalized_sign_key": pd.NA,
            "partC_rev1_normalized_sign_match": False,
            "partC_rev1_normalized_sign_global_fallback": False,
            "partC_rev1_normalized_sign_selected_distance_m": np.nan,
            "partC_rev1_geo_buffer_radius_m": np.nan,
            "partC_rev1_geo_buffer_intersects": False,
            "partC_rev1_geo_buffer_overlap_score": np.nan,
            "partC_rev1_geo_direct_distance_m": np.nan,
            "partC_rev1_faf_stfips": pd.NA,
            "partC_rev1_faf_ctfips": pd.NA,
            "partC_rev1_ntad_stfips": pd.NA,
            "partC_rev1_ntad_ctfips": pd.NA,
            "partC_rev1_match_fips_score": np.nan,
        }
    )


def partC_rev1_apply_source_matches(current: pd.DataFrame, source_matches: pd.DataFrame) -> pd.DataFrame:
    if source_matches.empty:
        return current
    out = current.copy()
    out = out.set_index("partC_rev1_link_uid", drop=False)
    source_matches = source_matches.set_index("partC_rev1_link_uid", drop=False)
    accepted = source_matches["partC_rev1_ntad_match_status"].eq("matched")
    for link_uid, row in source_matches.loc[accepted].iterrows():
        if link_uid not in out.index:
            continue
        if str(out.at[link_uid, "partC_rev1_ntad_match_status"]) != "matched":
            for column, value in row.items():
                out.at[link_uid, column] = value
    return out.reset_index(drop=True)


def partC_rev1_inherit_opposite_direction_matches(arcs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = arcs.copy()
    inherited_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    if "partC_rev1_link_uid" not in out.columns or "directed_arc_id" not in out.columns:
        return out, pd.DataFrame(), pd.DataFrame()
    for link_uid, group in out.groupby("partC_rev1_link_uid", dropna=False):
        matched = group[group.get("partC_rev1_ntad_match_status", pd.Series(index=group.index, dtype=str)).eq("matched")]
        unassigned = group[~group.index.isin(matched.index)]
        if len(matched) >= 2 and matched["partC_rev1_ntad_uid"].nunique(dropna=True) > 1:
            conflict_rows.append({"partC_rev1_link_uid": link_uid, "conflict_arc_count": len(matched)})
            continue
        if len(matched) == 1 and len(unassigned) > 0:
            source = matched.iloc[0]
            for idx in unassigned.index:
                for column in matched.columns:
                    if (
                        column.startswith("partC_rev1_match_")
                        or column.startswith("partC_rev1_ntad_")
                        or column.startswith("partC_rev1_geo_")
                        or column.startswith("partC_rev1_faf_")
                    ):
                        out.at[idx, column] = source[column]
                out.at[idx, "partC_rev1_ntad_match_status"] = "matched"
                out.at[idx, "partC_rev1_match_method"] = "opposite_direction_inherited"
                out.at[idx, "partC_rev1_inherited_from_directed_arc_id"] = source.get("directed_arc_id")
                inherited_rows.append(
                    {
                        "partC_rev1_link_uid": link_uid,
                        "directed_arc_id": out.at[idx, "directed_arc_id"],
                        "partC_rev1_inherited_from_directed_arc_id": source.get("directed_arc_id"),
                    }
                )
    return out, pd.DataFrame(inherited_rows), pd.DataFrame(conflict_rows)
