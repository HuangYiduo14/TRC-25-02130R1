"""Actual-final-backbone directed bundles and exact suppressed-arc closure."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .schemas import stable_object_id


class Phase3ABundleError(ValueError):
    pass


def build_backbone_bundles(
    *,
    suite_id: str,
    candidate_id: str,
    final_backbone_arc_mask: np.ndarray,
    arc_id: np.ndarray,
    tail_node_uid: np.ndarray,
    head_node_uid: np.ndarray,
    arc_fftt_s: np.ndarray,
    arc_length_m: np.ndarray,
    node_to_bacn_id: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mask = np.asarray(final_backbone_arc_mask, dtype=bool)
    arrays = [np.asarray(value) for value in (arc_id, tail_node_uid, head_node_uid, arc_fftt_s, arc_length_m)]
    if any(value.ndim != 1 or value.size != mask.size for value in arrays):
        raise Phase3ABundleError("Bundle input arrays must align with the final-backbone mask")
    final_indexes = np.flatnonzero(mask)
    bundle_groups: dict[tuple[str, str], list[int]] = {}
    suppressed_rows: list[dict[str, object]] = []
    for arc_index in final_indexes.tolist():
        tail_uid = str(tail_node_uid[arc_index])
        head_uid = str(head_node_uid[arc_index])
        if tail_uid not in node_to_bacn_id or head_uid not in node_to_bacn_id:
            raise Phase3ABundleError(f"Final-backbone endpoint is missing B-ACN membership: {arc_id[arc_index]}")
        tail_bacn = str(node_to_bacn_id[tail_uid])
        head_bacn = str(node_to_bacn_id[head_uid])
        if tail_bacn == head_bacn:
            suppressed_rows.append(
                {
                    "candidate_id": candidate_id,
                    "directed_arc_id": str(arc_id[arc_index]),
                    "bacn_id": tail_bacn,
                    "physical_tail_node_uid": tail_uid,
                    "physical_head_node_uid": head_uid,
                    "fftt_s": float(arc_fftt_s[arc_index]),
                    "length_m": float(arc_length_m[arc_index]),
                    "internal_component_information": "actual_final_backbone_intra_bacn",
                }
            )
        else:
            bundle_groups.setdefault((tail_bacn, head_bacn), []).append(arc_index)
    membership_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for (tail_bacn, head_bacn), indexes in sorted(bundle_groups.items()):
        indexes.sort(key=lambda index: _id_key(str(arc_id[index])))
        member_ids = [str(arc_id[index]) for index in indexes]
        bundle_id = stable_object_id(
            "bundle",
            suite_id=suite_id,
            candidate_id=candidate_id,
            members=member_ids,
            signature={"tail_bacn_id": tail_bacn, "head_bacn_id": head_bacn},
        )
        fftt = np.asarray([float(arc_fftt_s[index]) for index in indexes], dtype=np.float64)
        minimum_position = min(range(len(indexes)), key=lambda pos: (fftt[pos], _id_key(member_ids[pos])))
        for index in indexes:
            membership_rows.append(
                {
                    "candidate_id": candidate_id,
                    "bundle_id": bundle_id,
                    "tail_bacn_id": tail_bacn,
                    "head_bacn_id": head_bacn,
                    "directed_arc_id": str(arc_id[index]),
                    "physical_tail_node_uid": str(tail_node_uid[index]),
                    "physical_head_node_uid": str(head_node_uid[index]),
                    "member_fftt_s": float(arc_fftt_s[index]),
                    "member_length_m": float(arc_length_m[index]),
                }
            )
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "bundle_id": bundle_id,
                "tail_bacn_id": tail_bacn,
                "head_bacn_id": head_bacn,
                "member_arc_count": len(indexes),
                "minimum_member_fftt_s": float(np.min(fftt)),
                "median_member_fftt_s": float(np.median(fftt)),
                "maximum_member_fftt_s": float(np.max(fftt)),
                "direct_diagnostic_fftt_s": float(np.min(fftt)),
                "minimum_witness_arc_id": member_ids[minimum_position],
                "unique_physical_tail_state_count": len({str(tail_node_uid[index]) for index in indexes}),
                "unique_physical_head_state_count": len({str(head_node_uid[index]) for index in indexes}),
            }
        )
    membership = pd.DataFrame(membership_rows, columns=[
        "candidate_id", "bundle_id", "tail_bacn_id", "head_bacn_id", "directed_arc_id",
        "physical_tail_node_uid", "physical_head_node_uid", "member_fftt_s", "member_length_m",
    ])
    suppressed = pd.DataFrame(suppressed_rows, columns=[
        "candidate_id", "directed_arc_id", "bacn_id", "physical_tail_node_uid", "physical_head_node_uid",
        "fftt_s", "length_m", "internal_component_information",
    ])
    summary = pd.DataFrame(summary_rows)
    if not membership.empty:
        membership = membership.sort_values(["candidate_id", "directed_arc_id"], kind="mergesort").reset_index(drop=True)
    if not suppressed.empty:
        suppressed = suppressed.sort_values(["candidate_id", "directed_arc_id"], kind="mergesort").reset_index(drop=True)
    if not summary.empty:
        summary = summary.sort_values(["candidate_id", "bundle_id"], kind="mergesort").reset_index(drop=True)
    validate_bundle_closure(mask, arc_id, membership, suppressed)
    return membership, summary, suppressed


def validate_bundle_closure(
    final_backbone_arc_mask: np.ndarray,
    arc_id: np.ndarray,
    membership: pd.DataFrame,
    suppressed: pd.DataFrame,
) -> None:
    expected = {str(value) for value in np.asarray(arc_id)[np.asarray(final_backbone_arc_mask, dtype=bool)].tolist()}
    bundled = set(membership["directed_arc_id"].astype(str)) if not membership.empty else set()
    hidden = set(suppressed["directed_arc_id"].astype(str)) if not suppressed.empty else set()
    if bundled.intersection(hidden):
        raise Phase3ABundleError("Bundle and suppressed registries overlap")
    if bundled.union(hidden) != expected:
        raise Phase3ABundleError(
            f"Final-backbone closure failed: missing={len(expected-bundled-hidden)}, extra={len((bundled|hidden)-expected)}"
        )
    if len(bundled) != len(membership) or len(hidden) != len(suppressed):
        raise Phase3ABundleError("Bundle/suppressed registry contains duplicate directed arc IDs")


def _id_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)
