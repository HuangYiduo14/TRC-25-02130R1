"""Consolidated B/R/centroid terminal registries with frozen role semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .schemas import stable_object_id


class Phase3ATerminalError(ValueError):
    """Raised when consolidated terminal roles are inconsistent."""


def build_consolidated_terminals(
    *,
    suite_id: str,
    candidate_id: str,
    portals: pd.DataFrame,
    centroid_roles: pd.DataFrame,
    node_to_bacn: Mapping[str, str],
    node_to_racn: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build region-specific coarse terminals and a sorted flat member vector."""

    required_portal = {"region_uuid", "partC_rev1_node_uid", "can_enter", "can_exit"}
    required_centroid = {"region_uuid", "partC_rev1_node_uid", "centroid_role", "through_allowed"}
    if not required_portal.issubset(portals.columns) or not required_centroid.issubset(centroid_roles.columns):
        raise Phase3ATerminalError("Portal or centroid registry lacks required columns")
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in portals.sort_values(["region_uuid", "partC_rev1_node_uid"], kind="mergesort").to_dict("records"):
        region = str(row["region_uuid"])
        uid = str(row["partC_rev1_node_uid"])
        if uid in node_to_bacn:
            terminal_type = "BACN"
            parent = str(node_to_bacn[uid])
        elif uid in node_to_racn:
            terminal_type = "RACN"
            parent = str(node_to_racn[uid])
        else:
            # A nonbackbone portal omitted from the eligible R set remains a
            # singleton R control object so no physical interface disappears.
            terminal_type = "RACN"
            parent = stable_object_id(
                "racn",
                suite_id=suite_id,
                candidate_id=candidate_id,
                members=[uid],
                signature={"singleton_ineligible_portal": True},
            )
        key = (region, terminal_type, parent)
        record = groups.setdefault(
            key,
            {"members": set(), "can_enter": False, "can_exit": False, "centroid_through_allowed": True},
        )
        record["members"].add(uid)
        record["can_enter"] = bool(record["can_enter"] or bool(row["can_enter"]))
        record["can_exit"] = bool(record["can_exit"] or bool(row["can_exit"]))
    for row in centroid_roles.sort_values(
        ["region_uuid", "centroid_role", "partC_rev1_node_uid"], kind="mergesort"
    ).to_dict("records"):
        if bool(row["through_allowed"]):
            raise Phase3ATerminalError("Phase 3A centroid roles must prohibit through movement")
        region = str(row["region_uuid"])
        uid = str(row["partC_rev1_node_uid"])
        role = str(row["centroid_role"])
        if role == "centroid_origin":
            terminal_type = "ORIGIN_CENTROID"
            can_enter, can_exit = False, True
        elif role == "centroid_destination":
            terminal_type = "DESTINATION_CENTROID"
            can_enter, can_exit = True, False
        else:
            raise Phase3ATerminalError(f"Unknown centroid role: {role}")
        # Origin and destination copies intentionally never consolidate, even
        # when they share a physical centroid node.
        parent = stable_object_id(
            "centroid_role",
            suite_id=suite_id,
            candidate_id=candidate_id,
            members=[uid],
            signature={"region_id": region, "role": terminal_type},
        )
        groups[(region, terminal_type, parent)] = {
            "members": {uid},
            "can_enter": can_enter,
            "can_exit": can_exit,
            "centroid_through_allowed": False,
        }
    rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    offset = 0
    for region, terminal_type, parent in sorted(groups, key=lambda item: (_id_key(item[0]), item[1], item[2])):
        record = groups[(region, terminal_type, parent)]
        members = tuple(sorted(record["members"], key=_id_key))
        terminal_id = stable_object_id(
            "terminal",
            suite_id=suite_id,
            candidate_id=candidate_id,
            members=members,
            signature={"region_id": region, "terminal_type": terminal_type, "parent_object_id": parent},
        )
        begin = offset
        for order, uid in enumerate(members):
            flat_rows.append(
                {
                    "candidate_id": candidate_id,
                    "region_id": region,
                    "terminal_id": terminal_id,
                    "fine_member_order": order,
                    "partC_rev1_node_uid": uid,
                }
            )
            offset += 1
        rows.append(
            {
                "candidate_id": candidate_id,
                "region_id": region,
                "terminal_id": terminal_id,
                "terminal_type": terminal_type,
                "parent_object_id": parent,
                "can_enter_region": bool(record["can_enter"]),
                "can_exit_region": bool(record["can_exit"]),
                "fine_member_offset_begin": begin,
                "fine_member_offset_end": offset,
                "centroid_through_allowed": bool(record["centroid_through_allowed"]),
            }
        )
    terminals = pd.DataFrame(rows)
    members = pd.DataFrame(flat_rows)
    if terminals.empty:
        raise Phase3ATerminalError("Consolidated terminal construction produced zero terminals")
    if terminals.duplicated(["candidate_id", "region_id", "terminal_id"]).any():
        raise Phase3ATerminalError("Consolidated terminal primary key is not unique")
    _validate_centroid_roles(terminals)
    return terminals.reset_index(drop=True), members.reset_index(drop=True)


def terminal_fine_members(terminals: pd.DataFrame, members: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    grouped = members.groupby("terminal_id", sort=False)["partC_rev1_node_uid"].agg(list)
    result = {str(key): tuple(str(value) for value in values) for key, values in grouped.items()}
    missing = set(terminals["terminal_id"].astype(str)).difference(result)
    if missing:
        raise Phase3ATerminalError(f"Terminals lack fine members: {sorted(missing)[:5]}")
    return result


def _validate_centroid_roles(terminals: pd.DataFrame) -> None:
    origin = terminals["terminal_type"] == "ORIGIN_CENTROID"
    destination = terminals["terminal_type"] == "DESTINATION_CENTROID"
    if (terminals.loc[origin, "can_enter_region"].any() or not terminals.loc[origin, "can_exit_region"].all()):
        raise Phase3ATerminalError("Origin centroid direction semantics are invalid")
    if (not terminals.loc[destination, "can_enter_region"].all() or terminals.loc[destination, "can_exit_region"].any()):
        raise Phase3ATerminalError("Destination centroid direction semantics are invalid")
    if terminals.loc[origin | destination, "centroid_through_allowed"].any():
        raise Phase3ATerminalError("Centroid through movement must be false")
    both = terminals[origin | destination].groupby(
        ["region_id", "fine_member_offset_begin", "fine_member_offset_end"], sort=False
    )["terminal_type"].nunique()
    # Shared physical nodes are legal, but their role objects and offsets must
    # remain distinct; identical offsets would mean accidental consolidation.
    if (both > 1).any():
        raise Phase3ATerminalError("Origin and destination centroid roles were consolidated")


def _id_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)
