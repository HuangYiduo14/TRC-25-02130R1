"""Verify no-lane default arc patch integrity."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def partC_rev1_verify_no_lane_patch(cache_path: str | Path = "partC_rev1_cache/partC_rev1_network.sqlite") -> dict[str, object]:
    path = Path(cache_path)
    with sqlite3.connect(path) as conn:
        arc_count, min_id, max_id = conn.execute(
            "SELECT COUNT(1), MIN(directed_arc_id), MAX(directed_arc_id) FROM partC_rev1_directed_arcs"
        ).fetchone()
        defaulted_links = conn.execute(
            "SELECT COUNT(1) FROM partC_rev1_clean_physical_links WHERE partC_rev1_direction_case=?",
            ("DEFAULT_AB_BA_1_LANE",),
        ).fetchone()[0]
        defaulted_arcs = conn.execute(
            """
            SELECT COUNT(1), MIN(directed_arc_id), MAX(directed_arc_id), MIN(capacity), MAX(capacity)
            FROM partC_rev1_directed_arcs
            WHERE partC_rev1_missing_lane_defaulted=1
            """
        ).fetchone()
        remaining_ignored = conn.execute(
            "SELECT COUNT(1) FROM partC_rev1_clean_physical_links WHERE partC_rev1_direction_case=?",
            ("IGNORED_NO_LANE_INFO",),
        ).fetchone()[0]
        duplicate_link_direction_arcs = conn.execute(
            """
            SELECT COUNT(1)
            FROM (
                SELECT partC_rev1_link_uid, direction, COUNT(1) AS n
                FROM partC_rev1_directed_arcs
                GROUP BY partC_rev1_link_uid, direction
                HAVING n > 1
            )
            """
        ).fetchone()[0]
    result = {
        "arc_count": int(arc_count),
        "min_directed_arc_id": int(min_id),
        "max_directed_arc_id": int(max_id),
        "ids_contiguous_from_zero": int(min_id) == 0 and int(max_id) == int(arc_count) - 1,
        "defaulted_links": int(defaulted_links),
        "defaulted_arcs": int(defaulted_arcs[0]),
        "first_defaulted_arc_id": int(defaulted_arcs[1]),
        "last_defaulted_arc_id": int(defaulted_arcs[2]),
        "defaulted_min_capacity": float(defaulted_arcs[3]),
        "defaulted_max_capacity": float(defaulted_arcs[4]),
        "remaining_ignored_links": int(remaining_ignored),
        "duplicate_link_direction_arcs": int(duplicate_link_direction_arcs),
    }
    if not result["ids_contiguous_from_zero"]:
        raise ValueError(f"directed_arc_id values are not contiguous: {result}")
    if result["remaining_ignored_links"] != 0:
        raise ValueError(f"remaining ignored no-lane links: {result}")
    if result["duplicate_link_direction_arcs"] != 0:
        raise ValueError(f"duplicate link/direction arcs: {result}")
    return result


def main() -> int:
    print(json.dumps(partC_rev1_verify_no_lane_patch(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
