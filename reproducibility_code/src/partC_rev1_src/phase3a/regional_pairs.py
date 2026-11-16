"""SCC-assisted lazy sparse regional support and direct FFTT queries."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .models import RegionalPairWitness
from .schemas import canonical_json_bytes, stable_object_id
from .shortest_paths import (
    DEFAULT_TIE_TOLERANCE_S,
    SparseDirectedGraph,
    build_sparse_directed_graph,
    reconstruct_restricted_path,
    restricted_dijkstra,
)


class Phase3ARegionalPairError(ValueError):
    """Raised when regional pair support violates direction or role rules."""


@dataclass(slots=True)
class RegionSupport:
    region_id: str
    graph: SparseDirectedGraph
    scc_label: np.ndarray
    scc_count: int
    condensation_out: tuple[tuple[int, ...], ...]
    node_local_by_global: dict[int, int]
    _reachable_components: dict[int, frozenset[int]] = field(default_factory=dict)

    def component_reaches(self, source: int, target: int) -> bool:
        if source == target:
            return True
        if source not in self._reachable_components:
            seen = {source}
            stack = [source]
            while stack:
                current = stack.pop()
                for neighbor in self.condensation_out[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            self._reachable_components[source] = frozenset(seen)
        return target in self._reachable_components[source]


@dataclass(slots=True)
class LazyRegionalPairOracle:
    suite_id: str
    candidate_id: str
    support: RegionSupport
    terminals: pd.DataFrame
    fine_members_global: Mapping[str, tuple[int, ...]]
    physical_arc_ids: Sequence[str]
    centroid_global_nodes: frozenset[int]
    cost_version: str = "direct_fftt_v1"
    retain_query_cache: bool = True
    cache: dict[str, tuple[RegionalPairWitness, ...]] = field(default_factory=dict)
    cache_hash: dict[str, str] = field(default_factory=dict)
    query_seconds: dict[str, float] = field(default_factory=dict)

    def query(self, source_terminal_id: str) -> tuple[RegionalPairWitness, ...]:
        source_id = str(source_terminal_id)
        if self.retain_query_cache and source_id in self.cache:
            return self.cache[source_id]
        if not self.retain_query_cache:
            self.support._reachable_components.clear()
        source_rows = self.terminals[self.terminals["terminal_id"].astype(str) == source_id]
        if len(source_rows) != 1:
            raise Phase3ARegionalPairError("Regional source terminal is absent or duplicated")
        source_row = source_rows.iloc[0]
        if not bool(source_row["can_exit_region"]):
            if self.retain_query_cache:
                self.cache[source_id] = ()
                self.cache_hash[source_id] = hashlib.sha256(b"[]").hexdigest()
                self.query_seconds[source_id] = 0.0
            else:
                self.support._reachable_components.clear()
            return ()
        source_nodes = self.fine_members_global.get(source_id, ())
        if not source_nodes:
            raise Phase3ARegionalPairError("Regional source terminal has no fine members")
        possible_targets: list[pd.Series] = []
        for _, row in self.terminals.iterrows():
            target_id = str(row["terminal_id"])
            if target_id == source_id or not bool(row["can_enter_region"]):
                continue
            target_nodes = self.fine_members_global.get(target_id, ())
            if target_nodes and self._scc_possible(source_nodes, target_nodes):
                possible_targets.append(row)
        target_nodes_union = {
            node for row in possible_targets for node in self.fine_members_global[str(row["terminal_id"])]
        }
        started = time.perf_counter()
        if not target_nodes_union:
            result: tuple[RegionalPairWitness, ...] = ()
        else:
            tree = restricted_dijkstra(
                self.support.graph,
                source_nodes,
                target_global_nodes=target_nodes_union,
                prohibited_through_global_nodes=self.centroid_global_nodes,
                tie_tolerance_s=DEFAULT_TIE_TOLERANCE_S,
            )
            global_to_local = self.support.node_local_by_global
            witnesses: list[RegionalPairWitness] = []
            for row in possible_targets:
                target_id = str(row["terminal_id"])
                candidates = []
                for node in self.fine_members_global[target_id]:
                    local = global_to_local.get(int(node))
                    if local is not None and np.isfinite(tree.distances_s[local]):
                        arc_indexes = reconstruct_restricted_path(self.support.graph, tree, int(node))
                        arc_ids = tuple(str(self.physical_arc_ids[index]) for index in arc_indexes)
                        candidates.append((float(tree.distances_s[local]), arc_ids, int(node), arc_indexes))
                if not candidates:
                    continue
                distance, arc_ids, target_node, arc_indexes = min(candidates)
                # Recover the actual source by tracing the predecessor chain.
                current = global_to_local[target_node]
                while int(tree.predecessor_local[current]) >= 0:
                    current = int(tree.predecessor_local[current])
                source_node = int(self.support.graph.node_ids[current])
                pair_id = stable_object_id(
                    "regional_pair",
                    suite_id=self.suite_id,
                    candidate_id=self.candidate_id,
                    members=[source_id, target_id],
                    ordered_members=True,
                    signature={"region_id": self.support.region_id, "cost_version": self.cost_version},
                )
                witnesses.append(
                    RegionalPairWitness(
                        pair_id=pair_id,
                        region_id=self.support.region_id,
                        source_terminal_id=source_id,
                        target_terminal_id=target_id,
                        source_node_idx=source_node,
                        target_node_idx=target_node,
                        fftt_s=distance,
                        arc_indices=tuple(int(value) for value in arc_indexes),
                        arc_ids=arc_ids,
                    )
                )
            result = tuple(sorted(witnesses, key=lambda item: item.target_terminal_id))
        elapsed = time.perf_counter() - started
        if self.retain_query_cache:
            payload = [
                {
                    "pair_id": item.pair_id,
                    "source": item.source_terminal_id,
                    "target": item.target_terminal_id,
                    "fftt_s": item.fftt_s,
                    "arc_ids": item.arc_ids,
                }
                for item in result
            ]
            digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
            self.cache[source_id] = result
            self.cache_hash[source_id] = digest
            self.query_seconds[source_id] = elapsed
        else:
            self.support._reachable_components.clear()
        return result

    def _scc_possible(self, source_nodes: tuple[int, ...], target_nodes: tuple[int, ...]) -> bool:
        for source_node in source_nodes:
            source_local = self.support.node_local_by_global.get(int(source_node))
            if source_local is None:
                continue
            source_component = int(self.support.scc_label[source_local])
            for target_node in target_nodes:
                target_local = self.support.node_local_by_global.get(int(target_node))
                if target_local is None:
                    continue
                if self.support.component_reaches(source_component, int(self.support.scc_label[target_local])):
                    return True
        return False

    def cache_index(self) -> pd.DataFrame:
        offsets = 0
        rows: list[dict[str, Any]] = []
        for source in sorted(self.cache):
            count = len(self.cache[source])
            rows.append(
                {
                    "candidate_id": self.candidate_id,
                    "region_id": self.support.region_id,
                    "source_terminal_id": source,
                    "outgoing_pair_count": count,
                    "pair_offset_begin": offsets,
                    "pair_offset_end": offsets + count,
                    "restricted_dijkstra_seconds": self.query_seconds[source],
                    "cache_sha256": self.cache_hash[source],
                }
            )
            offsets += count
        return pd.DataFrame(rows)


def build_region_support(
    *,
    region_id: str,
    tail: np.ndarray,
    head: np.ndarray,
    fftt_s: np.ndarray,
    arc_ids: Sequence[str],
    region_arc_mask: np.ndarray,
) -> RegionSupport:
    graph = build_sparse_directed_graph(
        tail=tail,
        head=head,
        cost_s=fftt_s,
        arc_stable_id=arc_ids,
        arc_mask=region_arc_mask,
    )
    local_tail = np.repeat(np.arange(graph.node_count, dtype=np.int64), np.diff(graph.indptr))
    sparse = csr_matrix(
        (np.ones(graph.arc_count, dtype=np.uint8), (local_tail, graph.head_local)),
        shape=(graph.node_count, graph.node_count),
    )
    count, labels = connected_components(sparse, directed=True, connection="strong", return_labels=True)
    condensation: list[set[int]] = [set() for _ in range(int(count))]
    for source, target in zip(local_tail.tolist(), graph.head_local.tolist(), strict=True):
        left, right = int(labels[source]), int(labels[target])
        if left != right:
            condensation[left].add(right)
    return RegionSupport(
        region_id=str(region_id),
        graph=graph,
        scc_label=labels.astype(np.int32),
        scc_count=int(count),
        condensation_out=tuple(tuple(sorted(values)) for values in condensation),
        node_local_by_global={value: index for index, value in enumerate(graph.node_ids)},
    )


def regional_support_contract(
    *, candidate_id: str, region_supports: Sequence[RegionSupport], terminal_registry_hash: str
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "region_graph_identity": hashlib.sha256(
            canonical_json_bytes(
                [{"region_id": item.region_id, "nodes": item.graph.node_count, "arcs": item.graph.arc_count} for item in region_supports]
            )
        ).hexdigest(),
        "terminal_registry_identity": terminal_registry_hash,
        "scc_condensation_identity": hashlib.sha256(
            canonical_json_bytes(
                [{"region_id": item.region_id, "scc_count": item.scc_count, "edges": item.condensation_out} for item in region_supports]
            )
        ).hexdigest(),
        "source_terminal_cache_format": "sparse_outgoing_v1",
        "physical_support_definition": "directed reachability through retained-region physical arcs regardless of flow",
        "direct_fftt_cost_version": "direct_fftt_v1",
        "centroid_non_through": True,
        "pair_id_recipe": "phase3a stable_object_id(regional_pair, ordered terminals, region, cost version)",
        "dense_terminal_square_allocated": False,
        "cache_completeness_status": "LAZY_PARTIAL_UNTIL_QUERIED",
    }
