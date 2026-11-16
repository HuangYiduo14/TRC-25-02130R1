"""Exact GRS80 geodesy with conservative ECEF candidate acceleration."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from pyproj import Geod
from scipy.spatial import cKDTree


GRS80_GEOD = Geod(ellps="GRS80")
GRS80_A_M = 6_378_137.0
GRS80_INV_F = 298.257222101
GRS80_F = 1.0 / GRS80_INV_F
GRS80_E2 = GRS80_F * (2.0 - GRS80_F)


class Phase3AGeodesyError(ValueError):
    pass


def validate_lon_lat(lon: np.ndarray, lat: np.ndarray) -> dict[str, int | float]:
    x = np.asarray(lon, dtype=np.float64)
    y = np.asarray(lat, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise Phase3AGeodesyError("Longitude and latitude must be equal-length vectors")
    finite = np.isfinite(x) & np.isfinite(y)
    valid_range = (x >= -180.0) & (x <= 180.0) & (y >= -90.0) & (y <= 90.0)
    return {
        "row_count": int(x.size),
        "nonfinite_count": int((~finite).sum()),
        "out_of_range_count": int((finite & ~valid_range).sum()),
        "longitude_min": float(np.nanmin(x)) if x.size else float("nan"),
        "longitude_max": float(np.nanmax(x)) if x.size else float("nan"),
        "latitude_min": float(np.nanmin(y)) if y.size else float("nan"),
        "latitude_max": float(np.nanmax(y)) if y.size else float("nan"),
    }


def grs80_distance_m(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: np.ndarray | float,
    lat2: np.ndarray | float,
) -> np.ndarray:
    first_lon, first_lat, second_lon, second_lat = np.broadcast_arrays(
        np.asarray(lon1, dtype=np.float64),
        np.asarray(lat1, dtype=np.float64),
        np.asarray(lon2, dtype=np.float64),
        np.asarray(lat2, dtype=np.float64),
    )
    if first_lon.ndim == 0:
        _, _, distance = GRS80_GEOD.inv(
            float(first_lon), float(first_lat), float(second_lon), float(second_lat)
        )
    elif first_lon.size == 1:
        _, _, scalar_distance = GRS80_GEOD.inv(
            float(first_lon.ravel()[0]),
            float(first_lat.ravel()[0]),
            float(second_lon.ravel()[0]),
            float(second_lat.ravel()[0]),
        )
        distance = np.asarray([scalar_distance], dtype=np.float64).reshape(first_lon.shape)
    else:
        _, _, distance = GRS80_GEOD.inv(first_lon, first_lat, second_lon, second_lat)
    return np.asarray(distance, dtype=np.float64)


def geodetic_to_ecef(lon: np.ndarray, lat: np.ndarray, height_m: float | np.ndarray = 0.0) -> np.ndarray:
    longitude = np.deg2rad(np.asarray(lon, dtype=np.float64))
    latitude = np.deg2rad(np.asarray(lat, dtype=np.float64))
    height = np.broadcast_to(np.asarray(height_m, dtype=np.float64), longitude.shape)
    sin_lat = np.sin(latitude)
    cos_lat = np.cos(latitude)
    prime_vertical = GRS80_A_M / np.sqrt(1.0 - GRS80_E2 * sin_lat * sin_lat)
    x = (prime_vertical + height) * cos_lat * np.cos(longitude)
    y = (prime_vertical + height) * cos_lat * np.sin(longitude)
    z = (prime_vertical * (1.0 - GRS80_E2) + height) * sin_lat
    return np.column_stack((x, y, z))


def conservative_ecef_radius_m(geodesic_threshold_m: float) -> float:
    threshold = float(geodesic_threshold_m)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise Phase3AGeodesyError("Geodesic threshold must be finite and nonnegative")
    # A surface geodesic is never shorter than its straight chord.  The small
    # additive/relative margin protects boundary pairs from floating rounding.
    return threshold + max(0.01, threshold * 1e-10)


def ecef_candidate_pairs(lon: np.ndarray, lat: np.ndarray, threshold_m: float) -> np.ndarray:
    xyz = geodetic_to_ecef(lon, lat)
    if xyz.shape[0] < 2:
        return np.empty((0, 2), dtype=np.int64)
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(conservative_ecef_radius_m(threshold_m), output_type="ndarray")
    return np.asarray(pairs, dtype=np.int64).reshape((-1, 2))


def exact_filter_pairs(
    lon: np.ndarray,
    lat: np.ndarray,
    pairs: np.ndarray,
    threshold_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.asarray(pairs, dtype=np.int64).reshape((-1, 2))
    if candidate.size == 0:
        return candidate, np.empty(0, dtype=np.float64)
    distance = grs80_distance_m(
        np.asarray(lon)[candidate[:, 0]],
        np.asarray(lat)[candidate[:, 0]],
        np.asarray(lon)[candidate[:, 1]],
        np.asarray(lat)[candidate[:, 1]],
    )
    accepted = distance <= float(threshold_m)
    return candidate[accepted], distance[accepted]


def ecef_coverage_audit(
    lon: np.ndarray,
    lat: np.ndarray,
    threshold_m: float,
    exact_pairs: Iterable[tuple[int, int]],
) -> dict[str, int | float | bool]:
    candidate = ecef_candidate_pairs(lon, lat, threshold_m)
    candidate_set = {tuple(map(int, row)) for row in candidate.tolist()}
    expected = {tuple(sorted((int(u), int(v)))) for u, v in exact_pairs if int(u) != int(v)}
    missing = expected.difference(candidate_set)
    return {
        "threshold_m": float(threshold_m),
        "exact_pair_count": len(expected),
        "ecef_candidate_pair_count": len(candidate_set),
        "missing_exact_pair_count": len(missing),
        "coverage_complete": not missing,
    }
