"""Prompt 3A seed backbone extraction with weak-LCC filtering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .partC_rev1_ids import partC_rev1_stable_id


def _partC_rev1_bool_series(values: pd.Series, *, default: bool = False) -> pd.Series:
    if values is None:
        return pd.Series(dtype=bool)
    if values.dtype == bool:
        return values.fillna(default)
    return values.map(
        lambda value: default
        if pd.isna(value)
        else str(value).strip().lower() in {"1", "1.0", "true", "t", "yes", "y", "open", "allowed"}
    ).astype(bool)


def _partC_rev1_nonempty(values: pd.Series, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(False, index=index)
    return values.notna() & values.astype(str).str.strip().ne("") & values.astype(str).str.strip().str.lower().ne("nan")


def partC_rev1_strahnet_nonempty(arcs: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """Return STRAHNET seed flags from Prompt 2.5 preferred and fallback fields."""

    if not bool(config.get("backbone_partition", {}).get("use_strahnet_nonempty", True)):
        return pd.Series(False, index=arcs.index)
    flags = pd.Series(False, index=arcs.index)
    if "partC_rev1_faf_strahnet_nonempty" in arcs.columns:
        flags |= _partC_rev1_bool_series(arcs["partC_rev1_faf_strahnet_nonempty"], default=False).reindex(arcs.index, fill_value=False)
    for column in ("STRAHNET", "partC_rev1_raw_STRAHNET"):
        if column in arcs.columns:
            flags |= _partC_rev1_nonempty(arcs[column], arcs.index)
    return flags.astype(bool)


def partC_rev1_extract_seed_backbone(arcs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add seed-candidate and final seed-backbone flags.

    Prompt 3 now treats class/STRAHNET matches as candidate seed arcs, then keeps
    only the weakly connected largest component as the actual seed backbone.
    """

    out = arcs.copy()
    bp = config.get("backbone_partition", {})
    main_classes = {int(value) for value in bp.get("main_class_codes", [11, 12, 13, 21, 22, 23, 33])}
    class_codes = pd.to_numeric(out.get("class_code", pd.Series(np.nan, index=out.index)), errors="coerce")
    class_main = class_codes.astype("Int64").isin(main_classes).fillna(False)
    strahnet = partC_rev1_strahnet_nonempty(out, config)
    is_open = _partC_rev1_bool_series(out.get("is_open", pd.Series(True, index=out.index)), default=True)
    truck_valid = _partC_rev1_bool_series(out.get("is_valid_for_truck", pd.Series(True, index=out.index)), default=True)

    eligible = is_open & truck_valid
    out["is_seed_candidate_backbone"] = (eligible & (class_main | strahnet)).astype(bool)
    reason = np.select(
        [
            out["is_seed_candidate_backbone"] & class_main & strahnet,
            out["is_seed_candidate_backbone"] & class_main,
            out["is_seed_candidate_backbone"] & strahnet,
        ],
        ["class_main+strahnet_nonempty", "class_main", "strahnet_nonempty"],
        default="not_seed",
    )
    out["seed_candidate_reason"] = reason

    lcc_enabled = bool(bp.get("seed_lcc", {}).get("enabled", True))
    out["partC_rev1_seed_component_id"] = pd.NA
    out["partC_rev1_seed_component_size"] = 0
    out["is_seed_backbone_lcc"] = False
    if not lcc_enabled:
        out["is_seed_backbone"] = out["is_seed_candidate_backbone"].astype(bool)
        out["seed_backbone_reason"] = out["seed_candidate_reason"]
        return out

    if "from_node_uid" not in out.columns or "to_node_uid" not in out.columns:
        out["is_seed_backbone"] = out["is_seed_candidate_backbone"].astype(bool)
        out["is_seed_backbone_lcc"] = out["is_seed_backbone"]
        out["seed_backbone_reason"] = out["seed_candidate_reason"]
        out["partC_rev1_seed_lcc_status"] = "not_computed_missing_node_uids"
        return out

    out["from_node_uid"] = out["from_node_uid"].map(partC_rev1_stable_id)
    out["to_node_uid"] = out["to_node_uid"].map(partC_rev1_stable_id)
    candidate = out["is_seed_candidate_backbone"].fillna(False).astype(bool)
    component_by_node = _partC_rev1_seed_lcc_components(out.loc[candidate])
    component_sizes: dict[str, int] = {}
    for component_id in component_by_node.values():
        component_sizes[component_id] = component_sizes.get(component_id, 0) + 1
    if component_sizes:
        selected_component = sorted(
            component_sizes,
            key=lambda component_id: (-component_sizes[component_id], component_id),
        )[0]
    else:
        selected_component = None

    from_component = out["from_node_uid"].map(component_by_node)
    to_component = out["to_node_uid"].map(component_by_node)
    same_component = from_component.notna() & from_component.eq(to_component)
    out.loc[candidate & same_component, "partC_rev1_seed_component_id"] = from_component.loc[candidate & same_component]
    out.loc[candidate & same_component, "partC_rev1_seed_component_size"] = (
        from_component.loc[candidate & same_component].map(component_sizes).fillna(0).astype(int)
    )
    if selected_component is not None:
        out["is_seed_backbone_lcc"] = (candidate & same_component & from_component.eq(selected_component)).astype(bool)
    out["is_seed_backbone"] = out["is_seed_backbone_lcc"].astype(bool)
    out["seed_backbone_reason"] = np.where(out["is_seed_backbone"], out["seed_candidate_reason"], "not_seed")
    out["partC_rev1_seed_lcc_status"] = np.select(
        [
            out["is_seed_backbone"],
            candidate & same_component,
            candidate,
        ],
        [
            "selected_largest_component",
            "candidate_outside_largest_component",
            "candidate_without_component",
        ],
        default="not_seed_candidate",
    )
    return out


def _partC_rev1_seed_lcc_components(seed_candidates: pd.DataFrame) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if root_right < root_left:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left

    for row in seed_candidates.itertuples(index=False):
        left = partC_rev1_stable_id(getattr(row, "from_node_uid"))
        right = partC_rev1_stable_id(getattr(row, "to_node_uid"))
        if left == "" or right == "" or left == right:
            continue
        union(left, right)

    members_by_root: dict[str, list[str]] = {}
    for node in sorted(parent):
        members_by_root.setdefault(find(node), []).append(node)
    component_by_node: dict[str, str] = {}
    for number, (_root, members) in enumerate(
        sorted(members_by_root.items(), key=lambda item: (min(item[1]), len(item[1]))),
        start=1,
    ):
        component_id = f"partC_rev1_seed_cc_{number:08d}"
        for node in members:
            component_by_node[node] = component_id
    return component_by_node


def partC_rev1_seed_stage(reason: Any) -> str:
    value = str(reason)
    if value == "class_main+strahnet_nonempty":
        return "seed_lcc_class_and_strahnet"
    if value == "class_main":
        return "seed_lcc_class"
    if value == "strahnet_nonempty":
        return "seed_lcc_strahnet"
    return "non_backbone"
