"""Prompt 3A strict direct-link backbone expansion.

The only post-seed backbone additions are short non-seed direct links whose
endpoints are both in the seed-backbone weak-LCC.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from .partC_rev1_backbone_seed import partC_rev1_seed_stage
from .partC_rev1_ids import partC_rev1_stable_id


def _partC_rev1_arc_fftt(arcs: pd.DataFrame) -> pd.Series:
    for column in ("partC_rev1_fftt_min", "free_time_min"):
        if column in arcs.columns:
            values = pd.to_numeric(arcs[column], errors="coerce")
            if values.notna().any():
                return values
    return pd.Series(np.nan, index=arcs.index)


def _partC_rev1_arc_length_m(arcs: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    if isinstance(arcs, gpd.GeoDataFrame) and "geometry" in arcs.columns and arcs.geometry.notna().any():
        try:
            geom = arcs.geometry
            if arcs.crs is not None and getattr(arcs.crs, "is_geographic", False):
                target = str(config.get("ntad", {}).get("projected_crs", config.get("preprocess", {}).get("snap_projected_crs", "EPSG:5070")))
                geom = arcs.to_crs(target).geometry
            lengths = pd.Series(geom.length, index=arcs.index)
            if lengths.replace([np.inf, -np.inf], np.nan).notna().any() and lengths.max(skipna=True) > 0:
                return lengths
        except Exception:
            pass
    if "length_m" in arcs.columns:
        return pd.to_numeric(arcs["length_m"], errors="coerce")
    if "partC_rev1_length_m" in arcs.columns:
        return pd.to_numeric(arcs["partC_rev1_length_m"], errors="coerce")
    if "length_miles" in arcs.columns:
        return pd.to_numeric(arcs["length_miles"], errors="coerce") * 1609.344
    if "LENGTH" in arcs.columns:
        return pd.to_numeric(arcs["LENGTH"], errors="coerce") * 1609.344
    return pd.Series(np.nan, index=arcs.index)


def _partC_rev1_bool(values: pd.Series, default: bool = True) -> pd.Series:
    return values.map(
        lambda value: default
        if pd.isna(value)
        else str(value).strip().lower() in {"1", "1.0", "true", "t", "yes", "y", "open", "allowed"}
    ).astype(bool)


def _partC_rev1_empty_direct_connector_links(crs: Any = None) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        columns=[
            "backbone_direct_connector_id",
            "tail_node_uid",
            "head_node_uid",
            "undirected_node_pair",
            "directed_arc_ids",
            "partC_rev1_link_uids",
            "length_m",
            "direct_connector_tail_degree",
            "direct_connector_head_degree",
            "accepted",
            "direct_connector_rejection_reason",
            "geometry",
        ],
        geometry="geometry",
        crs=crs,
    )


def partC_rev1_candidate_expansion_arcs(arcs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Return non-seed candidate arcs for strict direct-link expansion.

    The function name is kept for compatibility with earlier Prompt 3 tests and
    callers, but the behavior is now direct-link-only.
    """

    bp = config.get("backbone_partition", {})
    connector_cfg = bp.get("direct_connectors", {})
    out = arcs.copy()
    out["from_node_uid"] = out["from_node_uid"].map(partC_rev1_stable_id)
    out["to_node_uid"] = out["to_node_uid"].map(partC_rev1_stable_id)
    out["partC_rev1_direct_connector_fftt_min"] = _partC_rev1_arc_fftt(out)
    out["partC_rev1_direct_connector_length_m"] = _partC_rev1_arc_length_m(out, config)
    is_open = _partC_rev1_bool(out.get("is_open", pd.Series(True, index=out.index)), True)
    truck_valid = _partC_rev1_bool(out.get("is_valid_for_truck", pd.Series(True, index=out.index)), True)
    finite_fftt = np.isfinite(out["partC_rev1_direct_connector_fftt_min"]) & (out["partC_rev1_direct_connector_fftt_min"] > 0)
    non_seed = ~out.get("is_seed_backbone", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    not_loop = out["from_node_uid"].ne(out["to_node_uid"])
    allowed_class = pd.Series(True, index=out.index)
    if bool(connector_cfg.get("exclude_centroid_connectors", bp.get("exclude_centroid_connectors_from_expansion", True))):
        excluded = {int(value) for value in bp.get("centroid_connector_class_codes", [50])}
        allowed_class = ~pd.to_numeric(out.get("class_code", pd.Series(np.nan, index=out.index)), errors="coerce").astype("Int64").isin(excluded)
    out["is_candidate_direct_connector_arc"] = (non_seed & is_open & truck_valid & finite_fftt & not_loop & allowed_class).astype(bool)
    out["candidate_direct_connector_reason"] = np.where(out["is_candidate_direct_connector_arc"], "eligible_direct_connector", "not_candidate")
    if not bool(connector_cfg.get("enabled", True)):
        out["is_candidate_direct_connector_arc"] = False
        out["candidate_direct_connector_reason"] = "direct_connector_expansion_disabled"
    return out.loc[out["is_candidate_direct_connector_arc"]].copy()


def partC_rev1_expand_backbone(arcs: pd.DataFrame, nodes: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Add only accepted direct seed-LCC-to-seed-LCC connectors to the backbone."""

    del nodes
    bp = config.get("backbone_partition", {})
    connector_cfg = bp.get("direct_connectors", {})
    max_length_m = float(connector_cfg.get("max_length_m", 300.0))
    max_degree = int(connector_cfg.get("max_degree", 1))

    all_arcs = arcs.copy()
    all_arcs["from_node_uid"] = all_arcs["from_node_uid"].map(partC_rev1_stable_id)
    all_arcs["to_node_uid"] = all_arcs["to_node_uid"].map(partC_rev1_stable_id)
    seed_flags = all_arcs.get("is_seed_backbone", pd.Series(False, index=all_arcs.index)).fillna(False).astype(bool)
    seed_nodes = set(all_arcs.loc[seed_flags, "from_node_uid"].dropna().astype(str)) | set(
        all_arcs.loc[seed_flags, "to_node_uid"].dropna().astype(str)
    )

    candidates = partC_rev1_candidate_expansion_arcs(all_arcs, config)
    if candidates.empty:
        links = _partC_rev1_empty_direct_connector_links(getattr(candidates, "crs", None))
        accepted_arc_ids: set[Any] = set()
    else:
        candidates["_edge_a"] = candidates[["from_node_uid", "to_node_uid"]].min(axis=1)
        candidates["_edge_b"] = candidates[["from_node_uid", "to_node_uid"]].max(axis=1)
        pair_degree_candidates = candidates.loc[
            candidates["_edge_a"].astype(str).isin(seed_nodes) & candidates["_edge_b"].astype(str).isin(seed_nodes)
        ].copy()
        pair_degree_candidates = pair_degree_candidates.loc[
            pd.to_numeric(pair_degree_candidates["partC_rev1_direct_connector_length_m"], errors="coerce").lt(max_length_m)
        ]
        pair_degree_edges = pair_degree_candidates[["_edge_a", "_edge_b"]].drop_duplicates()
        degree = pd.concat([pair_degree_edges["_edge_a"], pair_degree_edges["_edge_b"]], ignore_index=True).value_counts().to_dict()
        rows: list[dict[str, Any]] = []
        accepted_arc_ids = set()
        for number, ((edge_a, edge_b), group) in enumerate(candidates.groupby(["_edge_a", "_edge_b"], sort=True), start=1):
            edge_a = str(edge_a)
            edge_b = str(edge_b)
            length_m = float(pd.to_numeric(group["partC_rev1_direct_connector_length_m"], errors="coerce").min())
            tail_degree = int(degree.get(edge_a, 0))
            head_degree = int(degree.get(edge_b, 0))
            endpoints_in_seed_lcc = edge_a in seed_nodes and edge_b in seed_nodes
            under_length = np.isfinite(length_m) and length_m < max_length_m
            degree_ok = tail_degree <= max_degree and head_degree <= max_degree
            accepted = bool(endpoints_in_seed_lcc and under_length and degree_ok)
            reasons: list[str] = []
            if not endpoints_in_seed_lcc:
                reasons.append("endpoint_not_in_seed_lcc")
            if not under_length:
                reasons.append("length_not_under_threshold")
            if not degree_ok:
                reasons.append("degree_over_threshold")
            direct_id = f"partC_rev1_direct_connector_{number:08d}"
            directed_arc_ids = sorted(group["directed_arc_id"].tolist(), key=str)
            if accepted:
                accepted_arc_ids.update(directed_arc_ids)
            geometry = None
            if "geometry" in group.columns:
                geoms = [geom for geom in group["geometry"] if geom is not None and not pd.isna(geom)]
                geometry = geoms[0] if geoms else None
            rows.append(
                {
                    "backbone_direct_connector_id": direct_id,
                    "tail_node_uid": edge_a,
                    "head_node_uid": edge_b,
                    "undirected_node_pair": f"{edge_a}|{edge_b}",
                    "directed_arc_ids": "|".join(str(value) for value in directed_arc_ids),
                    "partC_rev1_link_uids": "|".join(
                        str(value) for value in sorted(group.get("partC_rev1_link_uid", pd.Series(dtype=str)).dropna().astype(str).unique())
                    ),
                    "length_m": length_m,
                    "direct_connector_tail_degree": tail_degree,
                    "direct_connector_head_degree": head_degree,
                    "accepted": accepted,
                    "direct_connector_rejection_reason": "accepted" if accepted else ";".join(reasons),
                    "geometry": geometry,
                }
            )
        links = gpd.GeoDataFrame(rows, geometry="geometry", crs=getattr(candidates, "crs", None))
        candidates = candidates.drop(columns=["_edge_a", "_edge_b"], errors="ignore")

    arc_id = all_arcs.get("directed_arc_id", pd.Series(index=all_arcs.index))
    added_flags = arc_id.isin(accepted_arc_ids)
    accepted_link_by_arc: dict[str, str] = {}
    if not links.empty:
        for link in links.loc[links["accepted"].fillna(False).astype(bool)].itertuples(index=False):
            for value in str(getattr(link, "directed_arc_ids", "")).split("|"):
                if value:
                    accepted_link_by_arc[value] = getattr(link, "backbone_direct_connector_id")
    all_arcs["backbone_direct_connector_id"] = arc_id.astype(str).map(accepted_link_by_arc)
    all_arcs["is_final_backbone"] = (seed_flags | added_flags).astype(bool)
    all_arcs["backbone_stage"] = "non_backbone"
    if "seed_backbone_reason" in all_arcs.columns:
        all_arcs.loc[seed_flags, "backbone_stage"] = all_arcs.loc[seed_flags, "seed_backbone_reason"].map(partC_rev1_seed_stage)
    all_arcs.loc[added_flags, "backbone_stage"] = "added_direct_seed_lcc_link"

    seed_node_rows = [{"partC_rev1_node_uid": node_id, "is_seed_backbone_node": True} for node_id in sorted(seed_nodes)]
    seed_nodes_table = pd.DataFrame(seed_node_rows)
    accepted_links = links.loc[links.get("accepted", pd.Series(False, index=links.index)).fillna(False).astype(bool)].copy()
    return {
        "arcs": all_arcs,
        "candidate_expansion_arcs": candidates,
        "candidate_direct_connector_arcs": candidates,
        "backbone_seed_nodes": seed_nodes_table,
        "direct_connector_links": links,
        "backbone_direct_connector_links": links,
        "accepted_direct_connector_links": accepted_links,
        "backbone_accepted_direct_connector_links": accepted_links,
        "final_backbone_arcs": all_arcs.loc[all_arcs["is_final_backbone"]].copy(),
    }
