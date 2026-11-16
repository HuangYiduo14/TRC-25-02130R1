"""Frozen fixed R-ACN construction and audit witnesses."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .clustering import cluster_pair_graph
from .geodesy import ecef_candidate_pairs, exact_filter_pairs
from .models import PairGraph
from .schemas import canonical_json_bytes, stable_object_id


RACN_GEODESIC_DIAMETER_M = 500.0
RACN_MUTUAL_RESTRICTED_FFTT_S = 120.0


class Phase3ARACNError(ValueError):
    """Raised when a fixed R-ACN violates the frozen pairwise rule."""


RestrictedDistanceProvider = Callable[[int, int, frozenset[str]], tuple[float, tuple[int, ...]] | None]


def canonical_portal_signature(rows: Iterable[Mapping[str, Any]]) -> tuple[tuple[str, bool, bool], ...]:
    values = {
        (str(row["region_id"]), bool(row["can_enter"]), bool(row["can_exit"]))
        for row in rows
    }
    if not values:
        raise Phase3ARACNError("A portal signature may not be empty")
    return tuple(sorted(values, key=lambda value: (_id_key(value[0]), value[1], value[2])))


def build_fixed_racns(
    *,
    suite_id: str,
    candidate_id: str,
    portal_node_uid: Sequence[str],
    portal_node_idx: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    weak_component: np.ndarray,
    signatures: Sequence[tuple[tuple[str, bool, bool], ...]],
    restricted_distance: RestrictedDistanceProvider,
    singleton_control: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct singleton or common fixed-rule R-ACNs.

    Eligibility (multi-region, nonbackbone, non-centroid) is resolved by the
    caller.  This routine enforces exact signature equality, component,
    complete-link GRS80 diameter, and mutual restricted 120-second paths.
    """

    uids = tuple(str(value) for value in portal_node_uid)
    indexes = np.asarray(portal_node_idx, dtype=np.int64)
    x = np.asarray(lon, dtype=np.float64)
    y = np.asarray(lat, dtype=np.float64)
    components = np.asarray(weak_component, dtype=np.int64)
    n = len(uids)
    if not (indexes.size == x.size == y.size == components.size == len(signatures) == n):
        raise Phase3ARACNError("R-ACN input vectors have inconsistent lengths")
    if len(set(uids)) != n:
        raise Phase3ARACNError("Eligible R-portal node UIDs must be unique")
    if n == 0:
        return _empty_membership(), _empty_summary(), _empty_witnesses()
    witness_rows: list[dict[str, Any]] = []
    if singleton_control:
        clusters = tuple((uid,) for uid in sorted(uids, key=_id_key))
        diameter_by_cluster = {cluster: 0.0 for cluster in clusters}
        time_by_pair: dict[tuple[str, str], float] = {}
    else:
        accepted_pairs: list[tuple[int, int, float, float]] = []
        for signature in sorted(set(signatures), key=lambda value: canonical_json_bytes(value)):
            group = np.asarray([i for i, value in enumerate(signatures) if value == signature], dtype=np.int64)
            if group.size < 2:
                continue
            candidates = ecef_candidate_pairs(x[group], y[group], RACN_GEODESIC_DIAMETER_M)
            exact, distances = exact_filter_pairs(x[group], y[group], candidates, RACN_GEODESIC_DIAMETER_M)
            for (left_local, right_local), geodesic_m in zip(exact.tolist(), distances.tolist(), strict=True):
                left = int(group[int(left_local)])
                right = int(group[int(right_local)])
                if components[left] != components[right]:
                    continue
                common_regions = frozenset(item[0] for item in signature)
                forward = restricted_distance(int(indexes[left]), int(indexes[right]), common_regions)
                reverse = restricted_distance(int(indexes[right]), int(indexes[left]), common_regions)
                if forward is None or reverse is None:
                    continue
                if forward[0] > RACN_MUTUAL_RESTRICTED_FFTT_S or reverse[0] > RACN_MUTUAL_RESTRICTED_FFTT_S:
                    continue
                dissimilarity = max(float(forward[0]), float(reverse[0]))
                accepted_pairs.append((left, right, float(geodesic_m), dissimilarity))
                for source, target, result in ((left, right, forward), (right, left, reverse)):
                    witness_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "source_node_uid": uids[source],
                            "target_node_uid": uids[target],
                            "restricted_fftt_s": float(result[0]),
                            "physical_arc_indices_json": json.dumps(list(map(int, result[1])), separators=(",", ":")),
                            "common_region_signature_json": _signature_json(signature),
                        }
                    )
        pair_graph = PairGraph(
            node_ids=uids,
            edge_u=np.asarray([row[0] for row in accepted_pairs], dtype=np.int64),
            edge_v=np.asarray([row[1] for row in accepted_pairs], dtype=np.int64),
            # CL legality is carried by existence of the mutual edge.  Exact
            # geodesic distance is the frozen merge priority/dissimilarity.
            dissimilarity=np.asarray([row[2] for row in accepted_pairs], dtype=np.float64),
            component_ids=components,
            metric="RACN_GRS80_WITH_MUTUAL_RESTRICTED_FFTT",
            threshold=RACN_GEODESIC_DIAMETER_M,
            threshold_unit="meters",
        )
        result = cluster_pair_graph(pair_graph, "CL")
        clusters = result.clusters
        diameter_by_cluster = dict(zip(clusters, result.actual_diameters, strict=True))
        time_by_pair = {
            tuple(sorted((uids[left], uids[right]), key=_id_key)): time_value
            for left, right, _, time_value in accepted_pairs
        }
    membership_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    signature_by_uid = dict(zip(uids, signatures, strict=True))
    component_by_uid = dict(zip(uids, components.tolist(), strict=True))
    for members in sorted(clusters, key=lambda value: _id_key(value[0])):
        ordered = tuple(sorted(members, key=_id_key))
        member_signatures = {signature_by_uid[value] for value in ordered}
        member_components = {int(component_by_uid[value]) for value in ordered}
        if len(member_signatures) != 1 or len(member_components) != 1:
            raise Phase3ARACNError("R-ACN members differ in signature or weak component")
        signature = next(iter(member_signatures))
        racn_id = stable_object_id(
            "racn",
            suite_id=suite_id,
            candidate_id=candidate_id,
            members=ordered,
            signature={"incident_directional_signature": signature},
        )
        maximum_mutual = 0.0
        for left_pos, left in enumerate(ordered[:-1]):
            for right in ordered[left_pos + 1 :]:
                key = tuple(sorted((left, right), key=_id_key))
                if key not in time_by_pair and not singleton_control:
                    raise Phase3ARACNError("R-ACN complete-link closure lacks a mutual witness")
                maximum_mutual = max(maximum_mutual, time_by_pair.get(key, 0.0))
        signature_json = _signature_json(signature)
        for order, uid in enumerate(ordered):
            membership_rows.append(
                {
                    "candidate_id": candidate_id,
                    "racn_id": racn_id,
                    "physical_portal_node_uid": uid,
                    "incident_region_signature_json": json.dumps([item[0] for item in signature], separators=(",", ":")),
                    "entry_exit_signature_json": signature_json,
                    "physical_weak_component_id": next(iter(member_components)),
                    "member_order": order,
                }
            )
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "racn_id": racn_id,
                "member_count": len(ordered),
                "minimum_member_uid": ordered[0],
                "exact_geodesic_diameter_m": float(diameter_by_cluster[tuple(ordered)]),
                "maximum_mutual_restricted_fftt_s": float(maximum_mutual),
                "entry_exit_signature_json": signature_json,
                "physical_weak_component_id": next(iter(member_components)),
                "singleton_control": bool(singleton_control),
                # A singleton is a valid (vacuously pairwise-complete) fixed
                # R-ACN control used by Rounds 1-2.
                "production_pairwise_valid": True,
            }
        )
    membership = pd.DataFrame(membership_rows).sort_values(
        ["candidate_id", "physical_portal_node_uid"], kind="mergesort"
    ).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows).sort_values(["candidate_id", "racn_id"], kind="mergesort").reset_index(drop=True)
    witnesses = pd.DataFrame(witness_rows)
    if not witnesses.empty:
        witnesses = witnesses.sort_values(
            ["candidate_id", "source_node_uid", "target_node_uid"], kind="mergesort"
        ).reset_index(drop=True)
    return membership, summary, witnesses


def _signature_json(signature: tuple[tuple[str, bool, bool], ...]) -> str:
    return json.dumps([[region, can_enter, can_exit] for region, can_enter, can_exit in signature], separators=(",", ":"))


def _empty_membership() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "candidate_id", "racn_id", "physical_portal_node_uid", "incident_region_signature_json",
        "entry_exit_signature_json", "physical_weak_component_id", "member_order",
    ])


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "candidate_id", "racn_id", "member_count", "minimum_member_uid",
        "exact_geodesic_diameter_m", "maximum_mutual_restricted_fftt_s",
        "entry_exit_signature_json", "physical_weak_component_id", "singleton_control",
        "production_pairwise_valid",
    ])


def _empty_witnesses() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "candidate_id", "source_node_uid", "target_node_uid", "restricted_fftt_s",
        "physical_arc_indices_json", "common_region_signature_json",
    ])


def _id_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)
