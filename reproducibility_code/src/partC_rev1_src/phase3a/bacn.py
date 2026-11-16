"""Stable B-ACN memberships and summaries."""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

from .models import ClusterResult
from .schemas import stable_object_id


class Phase3ABACNError(ValueError):
    pass


def build_bacn_membership(
    *,
    suite_id: str,
    candidate_id: str,
    family: str,
    algorithm: str,
    parameter_value: float,
    parameter_unit: str,
    final_backbone_node_uids: Iterable[str],
    weak_component_by_uid: Mapping[str, int],
    cluster_result: ClusterResult | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    backbone = tuple(sorted({str(value) for value in final_backbone_node_uids}, key=_id_key))
    if not backbone:
        raise Phase3ABACNError("B-ACN construction received zero final-backbone nodes")
    if cluster_result is None:
        clusters = tuple((value,) for value in backbone)
        diameters = (0.0,) * len(clusters)
        pairwise_valid = True
    else:
        clusters = cluster_result.clusters
        diameters = cluster_result.actual_diameters
        pairwise_valid = cluster_result.production_pairwise_valid or algorithm.upper() == "SL"
    flattened = [member for cluster in clusters for member in cluster]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(backbone):
        raise Phase3ABACNError("B-ACN clusters are not an exhaustive one-to-one backbone partition")
    membership_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for cluster, diameter in zip(clusters, diameters, strict=True):
        components = {int(weak_component_by_uid[member]) for member in cluster}
        if len(components) != 1:
            raise Phase3ABACNError("B-ACN cluster crosses physical weak components")
        bacn_id = stable_object_id(
            "bacn",
            suite_id=suite_id,
            candidate_id=candidate_id,
            members=cluster,
            signature={"family": family, "algorithm": algorithm, "parameter_value": parameter_value, "parameter_unit": parameter_unit},
        )
        for order, member in enumerate(sorted(cluster, key=_id_key)):
            membership_rows.append(
                {
                    "candidate_id": candidate_id,
                    "bacn_id": bacn_id,
                    "partC_rev1_node_uid": member,
                    "physical_weak_component_id": next(iter(components)),
                    "member_order_by_stable_uid": order,
                    "family": family,
                    "algorithm": algorithm,
                    "parameter_value": float(parameter_value),
                    "parameter_unit": parameter_unit,
                }
            )
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "bacn_id": bacn_id,
                "member_count": len(cluster),
                "minimum_member_uid": min(cluster, key=_id_key),
                "exact_geodesic_diameter_m": float(diameter) if family.upper() == "GEO" else None,
                "requested_geodesic_threshold_m": float(parameter_value) * 1000.0 if family.upper() == "GEO" else None,
                "maximum_mutual_aug_fftt_s": float(diameter) if family.upper() == "AUG" else None,
                "requested_aug_threshold_s": float(parameter_value) if family.upper() == "AUG" else None,
                "physical_weak_component_id": next(iter(components)),
                "production_pairwise_valid": bool(pairwise_valid and algorithm.upper() != "SL"),
                "not_applicable_reason": "family-specific fields are null when not applicable",
            }
        )
    membership = pd.DataFrame(membership_rows).sort_values(
        ["candidate_id", "partC_rev1_node_uid"], kind="mergesort"
    ).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows).sort_values(["candidate_id", "bacn_id"], kind="mergesort").reset_index(drop=True)
    return membership, summary


def node_to_bacn(membership: pd.DataFrame) -> dict[str, str]:
    required = {"partC_rev1_node_uid", "bacn_id"}
    if not required.issubset(membership.columns):
        raise Phase3ABACNError("B-ACN membership table is missing stable-ID columns")
    if membership["partC_rev1_node_uid"].duplicated().any():
        raise Phase3ABACNError("A physical node appears in more than one B-ACN")
    return dict(zip(membership["partC_rev1_node_uid"].astype(str), membership["bacn_id"].astype(str), strict=True))


def _id_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)
