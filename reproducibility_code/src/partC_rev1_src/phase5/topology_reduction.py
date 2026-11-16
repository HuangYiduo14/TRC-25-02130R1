"""Deterministic, non-mutating Phase 5 topology-census utilities."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


BACKBONE_LINK = "BACKBONE_BUNDLE"


def _stable_id(prefix: str, payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


def _json(values: Iterable[str]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class TopologyResult:
    census: dict[str, object]
    chains: pd.DataFrame
    node_protection: pd.DataFrame
    ownership_ledger: pd.DataFrame = field(default_factory=pd.DataFrame)
    incidence_validation: dict = field(default_factory=dict)


def _static_primitive_ids(static_links, bundle_summary):
    """Resolve primitive incidence by accepted directed endpoints, never ID guessing."""
    pairs = pd.MultiIndex.from_arrays([bundle_summary.tail_bacn_id.astype(str), bundle_summary.head_bacn_id.astype(str)])
    result = np.full(len(static_links), "", dtype=object)
    backbone = static_links.link_type.astype(str).eq(BACKBONE_LINK).to_numpy()
    targets = pd.MultiIndex.from_arrays([static_links.loc[backbone, "tail_node_id"].astype(str), static_links.loc[backbone, "head_node_id"].astype(str)])
    indexes = pairs.get_indexer(targets)
    if np.any(indexes < 0):
        raise ValueError("Static backbone link has no accepted primitive")
    result[backbone] = bundle_summary.bundle_id.astype(str).to_numpy()[indexes]
    if Counter(result[backbone]) != Counter(bundle_summary.bundle_id.astype(str)):
        raise ValueError("Full static incidence is not one-to-one with primitive links")
    return result


def remaining_incidence_audit(suite_id, chains, static_links, bundle_summary, od_endpoint_bacn_ids=()):
    """Exhaustive original-link ownership and all surviving endpoint checks."""
    primitive = _static_primitive_ids(static_links, bundle_summary)
    accepted = chains.loc[~chains.protected_or_blocked.astype(bool)]
    owners = {}; removed = set(); replacement_endpoints = set()
    for row in accepted.itertuples(index=False):
        if row.retained_tail_bacn_id == row.retained_head_bacn_id:
            raise ValueError("Self-endpoint replacement cannot be accepted")
        removed.update(json.loads(row.internal_bacn_ids_json))
        replacement_endpoints.update((row.retained_tail_bacn_id, row.retained_head_bacn_id))
        for edge in json.loads(row.primitive_bundle_ids_json):
            if edge in owners:
                raise ValueError("Primitive link consumed by more than one replacement")
            owners[edge] = row.chain_id
    consumed = np.array([edge in owners for edge in primitive])
    dangling_mask = (~consumed) & (static_links.tail_node_id.astype(str).isin(removed).to_numpy() | static_links.head_node_id.astype(str).isin(removed).to_numpy())
    dangling = int(dangling_mask.sum())
    endpoint_conflicts = len(removed & replacement_endpoints)
    od_conflicts = len(removed & set(map(str, od_endpoint_bacn_ids)))
    ledger = static_links[["link_id", "link_type", "tail_node_id", "head_node_id"]].copy()
    if ledger.link_id.duplicated().any():
        raise ValueError("Duplicate original static link ID")
    ledger.insert(0, "suite_short_id", suite_id)
    ledger["suite_qualified_link_id"] = suite_id + ":" + ledger.link_id.astype(str)
    ledger["primitive_bundle_id"] = primitive
    ledger["ownership"] = np.where(consumed, "CONSUMED_ONCE_BY_REPLACEMENT", "RETAINED_UNCHANGED")
    ledger["replacement_chain_id"] = [owners.get(edge, "") for edge in primitive]
    ledger["owner_count"] = 1
    validation = {"schema_id":"partC.phase5.remaining_incidence.v4", "suite_short_id":suite_id,
                  "original_static_link_count":len(ledger), "original_primitive_count":len(bundle_summary),
                  "consumed_primitive_count":len(owners), "retained_static_count":int((~consumed).sum()),
                  "removed_bacn_count":len(removed), "dangling_surviving_incidence":dangling,
                  "replacement_endpoint_conflicts":endpoint_conflicts, "removed_od_nodes":od_conflicts,
                  "exactly_one_owner":True, "terminal_connectors_included":True,
                  "status":"PASS" if dangling == endpoint_conflicts == od_conflicts == 0 else "FAIL"}
    return ledger, validation


def _flow_diagnostics(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return np.nan, np.nan, np.nan, np.nan
    array = np.asarray(values, dtype=np.float64)
    low = float(array.min())
    high = float(array.max())
    spread = high - low
    relative = spread / max(1.0, float(np.max(np.abs(array))))
    return low, high, spread, relative


def _consistent(minimum: float, maximum: float, *, atol: float, rtol: float) -> bool:
    return bool(maximum - minimum <= atol + rtol * max(1.0, abs(maximum)))


def enumerate_topology(
    *,
    suite_id: str,
    bacn_summary: pd.DataFrame,
    static_links: pd.DataFrame,
    bundle_summary: pd.DataFrame,
    low_bundle_flow: Mapping[str, float],
    high_bundle_flow: Mapping[str, float],
    od_endpoint_bacn_ids: Iterable[str],
    flow_atol: float = 1.0e-8,
    flow_rtol: float = 1.0e-10,
) -> TopologyResult:
    """Classify all B-ACNs and enumerate conservative candidate macro-links.

    The accepted tables are never modified.  Candidate rows form an overlay;
    blocked rows remain present as auditable negative evidence.
    """

    bacn_ids = tuple(sorted(bacn_summary["bacn_id"].astype(str).unique()))
    bacn_set = set(bacn_ids)
    bundles = bundle_summary[
        ["bundle_id", "tail_bacn_id", "head_bacn_id"]
    ].copy()
    bundles = bundles.astype(str).sort_values("bundle_id", kind="mergesort")
    if bundles["bundle_id"].duplicated().any():
        raise ValueError(f"{suite_id}: duplicate bundle ID")
    if bundles[["tail_bacn_id", "head_bacn_id"]].duplicated().any():
        raise ValueError(f"{suite_id}: accepted primitive bundle endpoints are not unique")

    edge_by_pair: dict[tuple[str, str], str] = {}
    tail_by_edge: dict[str, str] = {}
    head_by_edge: dict[str, str] = {}
    incoming: dict[str, list[str]] = {node: [] for node in bacn_ids}
    outgoing: dict[str, list[str]] = {node: [] for node in bacn_ids}
    for edge, tail, head in bundles.itertuples(index=False, name=None):
        edge_by_pair[(tail, head)] = edge
        tail_by_edge[edge] = tail
        head_by_edge[edge] = head
        incoming[head].append(edge)
        outgoing[tail].append(edge)
    for values in incoming.values():
        values.sort()
    for values in outgoing.values():
        values.sort()

    reasons: dict[str, set[str]] = {node: set() for node in bacn_ids}
    nonbackbone = static_links[static_links["link_type"].astype(str) != BACKBONE_LINK]
    for tail, head, link_type in nonbackbone[
        ["tail_node_id", "head_node_id", "link_type"]
    ].astype(str).itertuples(index=False, name=None):
        for node in (tail, head):
            if node in bacn_set:
                reasons[node].add("REGIONAL_TERMINAL_CONNECTOR")
                reasons[node].add("INCOMPATIBLE_LINK_CLASS")
    for node in set(map(str, od_endpoint_bacn_ids)) & bacn_set:
        reasons[node].add("OD_OR_CENTROID_ROLE")

    boundary_state = dict(
        zip(
            bacn_summary["bacn_id"].astype(str),
            pd.to_numeric(bacn_summary["boundary_state_count"], errors="coerce").fillna(0).astype(int),
            strict=True,
        )
    )
    for node, count in boundary_state.items():
        if count > 1:
            reasons[node].add("TURN_OR_STATE_SPECIFIC_ROLE")

    neighbors: dict[str, set[str]] = {node: set() for node in bacn_ids}
    for _, tail, head in bundles.itertuples(index=False, name=None):
        if tail == head:
            reasons[tail].add("SELF_LOOP_CONSEQUENCE")
        else:
            neighbors[tail].add(head)
            neighbors[head].add(tail)

    strict_raw: set[str] = set()
    reciprocal_raw: set[str] = set()
    irregular_degree2: set[str] = set()
    for node in bacn_ids:
        indegree, outdegree = len(incoming[node]), len(outgoing[node])
        predecessor = tail_by_edge[incoming[node][0]] if indegree == 1 else None
        successor = head_by_edge[outgoing[node][0]] if outdegree == 1 else None
        strict_shape = indegree == 1 and outdegree == 1 and predecessor != successor
        adjacent = sorted(neighbors[node])
        reciprocal_shape = (
            len(adjacent) == 2
            and indegree == 2
            and outdegree == 2
            and all((other, node) in edge_by_pair and (node, other) in edge_by_pair for other in adjacent)
        )
        if not reasons[node] and reciprocal_shape:
            reciprocal_raw.add(node)
        elif not reasons[node] and strict_shape:
            strict_raw.add(node)
        elif len(adjacent) == 2 and not (strict_shape or reciprocal_shape):
            irregular_degree2.add(node)
            reasons[node].add("IRREGULAR_DIRECTED_DEGREE2")
            reasons[node].add("INCOMPATIBLE_LINK_CLASS")
        elif not reasons[node]:
            reasons[node].add("BRANCHING_OR_ENDPOINT_ROLE")

    rows: list[dict[str, object]] = []
    cycle_count = 0

    def add_row(
        direction_class: str,
        tail: str,
        head: str,
        edges: list[str],
        internal: list[str],
        *,
        cycle_anchor: str = "",
        block_reasons: Iterable[str] = (),
    ) -> None:
        low_values = [float(low_bundle_flow[edge]) for edge in edges]
        high_values = [float(high_bundle_flow[edge]) for edge in edges]
        lmin, lmax, labs, lrel = _flow_diagnostics(low_values)
        hmin, hmax, habs, hrel = _flow_diagnostics(high_values)
        low_ok = _consistent(lmin, lmax, atol=flow_atol, rtol=flow_rtol)
        high_ok = _consistent(hmin, hmax, atol=flow_atol, rtol=flow_rtol)
        blocks = sorted(set(map(str, block_reasons)))
        if tail == head:
            blocks.append("SELF_LOOP_CONSEQUENCE")
        if not low_ok:
            blocks.append("LOW_CURRENT_FLOW_MISMATCH")
        if not high_ok:
            blocks.append("HIGH_CURRENT_FLOW_MISMATCH")
        payload = [suite_id, direction_class, tail, head, edges, internal]
        rows.append(
            {
                "suite_short_id": suite_id,
                "chain_id": _stable_id("p5_chain_", payload),
                "reciprocal_group_id": (_stable_id("p5_reciprocal_", [suite_id, sorted((tail, head)), sorted(internal)]) if "RECIPROCAL" in direction_class else ""),
                "direction_class": direction_class,
                "retained_tail_bacn_id": tail,
                "retained_head_bacn_id": head,
                "primitive_bundle_ids_json": _json(edges),
                "internal_bacn_ids_json": _json(internal),
                "primitive_link_count": len(edges),
                "internal_node_count": len(internal),
                "low_flow_min": lmin,
                "low_flow_max": lmax,
                "low_flow_abs_spread": labs,
                "low_flow_rel_spread": lrel,
                "high_flow_min": hmin,
                "high_flow_max": hmax,
                "high_flow_abs_spread": habs,
                "high_flow_rel_spread": hrel,
                "low_flow_consistent": low_ok,
                "high_flow_consistent": high_ok,
                "protected_or_blocked": bool(blocks),
                "block_reasons_json": _json(sorted(set(blocks))),
                "parallel_collision_group_id": "",
                "cycle_anchor_id": cycle_anchor,
                "canonical_signature": "",
                "sample_included": False,
                "sample_inclusion_reasons_json": "[]",
            }
        )

    # Directed 1-in/1-out chains.
    visited_strict: set[str] = set()
    starts = sorted(
        node
        for node in strict_raw
        if tail_by_edge[incoming[node][0]] not in strict_raw
    )
    for start in starts:
        if start in visited_strict:
            continue
        tail = tail_by_edge[incoming[start][0]]
        edges = [incoming[start][0]]
        internal: list[str] = []
        current = start
        while current in strict_raw and current not in visited_strict:
            visited_strict.add(current)
            internal.append(current)
            edge = outgoing[current][0]
            edges.append(edge)
            current = head_by_edge[edge]
        blocks: list[str] = []
        if tail in reciprocal_raw or current in reciprocal_raw:
            blocks.append("MIXED_CANDIDATE_CLASS_BOUNDARY")
        if tail == current:
            blocks.append("SELF_LOOP_CONSEQUENCE")
        add_row("STRICT_DIRECTED_1IN_1OUT", tail, current, edges, internal, block_reasons=blocks)

    remaining = sorted(strict_raw - visited_strict)
    while remaining:
        seed = remaining[0]
        cycle: list[str] = []
        current = seed
        seen: set[str] = set()
        while current not in seen and current in strict_raw:
            seen.add(current)
            cycle.append(current)
            current = head_by_edge[outgoing[current][0]]
        for node in seen:
            visited_strict.add(node)
        anchor = min(cycle)
        index = cycle.index(anchor)
        ordered = cycle[index:] + cycle[:index]
        edges = [outgoing[node][0] for node in ordered]
        cycle_count += 1
        reasons[anchor].add("REQUIRED_CYCLE_ANCHOR_ROLE")
        add_row(
            "STRICT_DIRECTED_CYCLE",
            anchor,
            anchor,
            edges,
            [node for node in ordered if node != anchor],
            cycle_anchor=anchor,
            block_reasons=("CLOSED_CYCLE", "SELF_LOOP_CONSEQUENCE"),
        )
        remaining = sorted(strict_raw - visited_strict)

    # Undirected regular reciprocal corridors, emitted once per direction.
    eligible_adjacency = {
        node: sorted(neighbors[node] & reciprocal_raw) for node in reciprocal_raw
    }
    reciprocal_seen: set[str] = set()
    for seed in sorted(reciprocal_raw):
        if seed in reciprocal_seen:
            continue
        stack = [seed]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(other for other in eligible_adjacency[node] if other not in component)
        reciprocal_seen.update(component)
        endpoints = sorted(node for node in component if len(eligible_adjacency[node]) < 2)
        if not endpoints:
            anchor = min(component)
            cycle_count += 1
            reasons[anchor].add("REQUIRED_CYCLE_ANCHOR_ROLE")
            next_node = min(eligible_adjacency[anchor])
            ordered = [anchor]
            previous, current = anchor, next_node
            while current != anchor:
                ordered.append(current)
                choices = [value for value in eligible_adjacency[current] if value != previous]
                previous, current = current, choices[0]
            forward = [edge_by_pair[(ordered[i], ordered[(i + 1) % len(ordered)])] for i in range(len(ordered))]
            reverse = [edge_by_pair[(ordered[(i + 1) % len(ordered)], ordered[i])] for i in reversed(range(len(ordered)))]
            for label, edges in (("FORWARD", forward), ("REVERSE", reverse)):
                add_row(
                    f"REGULAR_RECIPROCAL_CYCLE_{label}",
                    anchor,
                    anchor,
                    edges,
                    [node for node in ordered if node != anchor],
                    cycle_anchor=anchor,
                    block_reasons=("CLOSED_CYCLE", "SELF_LOOP_CONSEQUENCE"),
                )
            continue
        first = endpoints[0]
        ordered: list[str] = []
        previous = ""
        current = first
        while True:
            ordered.append(current)
            choices = [value for value in eligible_adjacency[current] if value != previous]
            if not choices:
                break
            previous, current = current, choices[0]
        external_first = sorted(neighbors[ordered[0]] - component)
        external_last = sorted(neighbors[ordered[-1]] - component)
        if len(ordered) == 1:
            externals = sorted(neighbors[ordered[0]] - component)
            if len(externals) != 2:
                raise RuntimeError(f"{suite_id}: reciprocal singleton corridor is malformed")
            tail, head = externals
        else:
            if len(external_first) != 1 or len(external_last) != 1:
                raise RuntimeError(f"{suite_id}: reciprocal corridor endpoints are malformed")
            tail, head = external_first[0], external_last[0]
        path = [tail, *ordered, head]
        forward = [edge_by_pair[(path[i], path[i + 1])] for i in range(len(path) - 1)]
        reverse = [edge_by_pair[(path[i + 1], path[i])] for i in reversed(range(len(path) - 1))]
        blocks = []
        if tail in strict_raw or head in strict_raw:
            blocks.append("MIXED_CANDIDATE_CLASS_BOUNDARY")
        add_row("REGULAR_RECIPROCAL_FORWARD", tail, head, forward, ordered, block_reasons=blocks)
        add_row("REGULAR_RECIPROCAL_REVERSE", head, tail, reverse, list(reversed(ordered)), block_reasons=blocks)

    chains = pd.DataFrame(rows)
    if chains.empty:
        raise RuntimeError(f"{suite_id}: topology census produced zero candidate evidence")

    # An edge cannot belong to two proposed replacements.  Preserve both rows
    # as negative evidence and block both instead of selecting silently.
    edge_rows: dict[str, list[int]] = defaultdict(list)
    for index, text in chains["primitive_bundle_ids_json"].items():
        for edge in json.loads(text):
            edge_rows[edge].append(int(index))
    for edge, indexes in edge_rows.items():
        active = [index for index in indexes if not bool(chains.at[index, "protected_or_blocked"])]
        if len(active) > 1:
            for index in active:
                blocks = set(json.loads(chains.at[index, "block_reasons_json"]))
                blocks.add("OVERLAPPING_PRIMITIVE_LINK_" + edge)
                chains.at[index, "block_reasons_json"] = _json(sorted(blocks))
                chains.at[index, "protected_or_blocked"] = True

    # Forward/reverse deletion is atomic. Recompute full surviving incidence
    # after every rejection wave; rejection can expose a new dangling edge.
    static_primitive = _static_primitive_ids(static_links, bundles)
    edge_sets = [set(json.loads(v)) for v in chains.primitive_bundle_ids_json]
    node_sets = [set(json.loads(v)) for v in chains.internal_bacn_ids_json]
    groups = defaultdict(list)
    for i, group in enumerate(chains.reciprocal_group_id):
        groups[str(group) if group else str(chains.at[i, "chain_id"])].append(i)
    changed = True
    while changed:
        changed = False
        reject = defaultdict(set)
        active_mask = ~chains.protected_or_blocked.to_numpy(dtype=bool)
        for indexes in groups.values():
            reciprocal = bool(chains.at[indexes[0], "reciprocal_group_id"])
            if reciprocal and (len(indexes) != 2 or not all(active_mask[i] for i in indexes)):
                for i in indexes:
                    if active_mask[i]:
                        reject[i].add("RECIPROCAL_GROUP_ATOMIC_RETENTION")
        consumed = set().union(*(edge_sets[i] for i in np.flatnonzero(active_mask)))
        removed = set().union(*(node_sets[i] for i in np.flatnonzero(active_mask)))
        survives = np.array([edge not in consumed for edge in static_primitive])
        surviving_nodes = set(static_links.loc[survives, "tail_node_id"].astype(str)) | set(static_links.loc[survives, "head_node_id"].astype(str))
        replacement_nodes = set(chains.loc[active_mask, "retained_tail_bacn_id"]) | set(chains.loc[active_mask, "retained_head_bacn_id"])
        conflicts = removed & (surviving_nodes | replacement_nodes | set(map(str, od_endpoint_bacn_ids)))
        node_groups = defaultdict(set)
        for group, indexes in groups.items():
            for i in indexes:
                if active_mask[i]:
                    for node in node_sets[i]:
                        node_groups[node].add(group)
        conflicts.update(node for node, values in node_groups.items() if len(values) > 1)
        for i in np.flatnonzero(active_mask):
            if node_sets[i] & conflicts:
                reject[int(i)].add("FULL_STATIC_INCIDENCE_OR_GROUP_OVERLAP")
        for i, values in reject.items():
            chains.at[i, "protected_or_blocked"] = True
            chains.at[i, "block_reasons_json"] = _json(sorted(set(json.loads(chains.at[i, "block_reasons_json"])) | values))
            changed = True

    ledger, incidence = remaining_incidence_audit(suite_id, chains, static_links, bundles, od_endpoint_bacn_ids)
    if incidence["status"] != "PASS":
        raise RuntimeError(f"Topology surviving-incidence closure failed: {incidence}")

    endpoint_to_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    primitive_pairs = set(edge_by_pair)
    for index, row in chains.iterrows():
        if bool(row["protected_or_blocked"]):
            continue
        pair = (str(row["retained_tail_bacn_id"]), str(row["retained_head_bacn_id"]))
        endpoint_to_rows[pair].append(int(index))
        chain_edges = set(json.loads(str(row["primitive_bundle_ids_json"])))
        existing = edge_by_pair.get(pair)
        if existing is not None and existing not in chain_edges:
            endpoint_to_rows[pair].append(-1)
    collision_count = 0
    for pair, indexes in sorted(endpoint_to_rows.items()):
        if len(indexes) <= 1:
            continue
        collision_count += 1
        group = _stable_id("p5_parallel_", [suite_id, *pair])
        for index in indexes:
            if index >= 0:
                chains.at[index, "parallel_collision_group_id"] = group

    unblocked = chains.loc[~chains["protected_or_blocked"].astype(bool)]
    removed_nodes: set[str] = set()
    consumed_edges: set[str] = set()
    for row in unblocked.itertuples(index=False):
        removed_nodes.update(json.loads(row.internal_bacn_ids_json))
        consumed_edges.update(json.loads(row.primitive_bundle_ids_json))
    projected_links = len(bundles) - len(consumed_edges) + len(unblocked)
    lengths = unblocked["primitive_link_count"].to_numpy(dtype=np.float64)
    protected = {node for node, values in reasons.items() if values}
    reason_counts = Counter(reason for node in bacn_ids for reason in reasons[node])
    census: dict[str, object] = {
        "suite_short_id": suite_id,
        "original_bacn_count": len(bacn_ids),
        "original_backbone_bundle_count": len(bundles),
        "protected_bacn_count": len(protected),
        "protected_reason_counts_json": json.dumps(dict(sorted(reason_counts.items())), separators=(",", ":")),
        "strict_directed_candidate_count": len(strict_raw),
        "reciprocal_candidate_count": len(reciprocal_raw),
        "irregular_degree2_count": len(irregular_degree2),
        "cycle_count": cycle_count,
        "maximal_chain_count": len(unblocked),
        "chain_length_mean": float(np.mean(lengths)) if len(lengths) else 0.0,
        "chain_length_p50": float(np.quantile(lengths, 0.50)) if len(lengths) else 0.0,
        "chain_length_p95": float(np.quantile(lengths, 0.95)) if len(lengths) else 0.0,
        "chain_length_max": int(np.max(lengths)) if len(lengths) else 0,
        "projected_retained_bacn_count": len(bacn_ids) - len(removed_nodes),
        "projected_macro_link_count": int(projected_links),
        "node_reduction_fraction": len(removed_nodes) / len(bacn_ids),
        "backbone_link_reduction_fraction": (len(bundles) - projected_links) / len(bundles),
        "parallel_endpoint_collision_count": collision_count,
        "self_loop_block_count": int(chains["block_reasons_json"].str.contains("SELF_LOOP_CONSEQUENCE").sum()),
        "status": "CENSUS_COMPLETE_CANDIDATE_OVERLAY_ONLY",
    }
    protection = pd.DataFrame(
        {
            "suite_short_id": suite_id,
            "bacn_id": bacn_ids,
            "protection_reasons_json": [_json(sorted(reasons[node])) for node in bacn_ids],
            "protected": [bool(reasons[node]) for node in bacn_ids],
            "backbone_indegree": [len(incoming[node]) for node in bacn_ids],
            "backbone_outdegree": [len(outgoing[node]) for node in bacn_ids],
            "undirected_neighbor_count": [len(neighbors[node]) for node in bacn_ids],
        }
    )
    return TopologyResult(census, chains, protection, ledger, incidence)
