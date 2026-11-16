"""Typed shared data models for Phase 3A."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from partC_rev1_src.partC_rev1_graph_snapshot import PartCRev1GraphSnapshot
from partC_rev1_src.partC_rev1_od_snapshot import PartCRev1ODSnapshot


@dataclass(frozen=True, slots=True)
class AuthorityPaths:
    project_root: Path
    phase2_output_root: Path
    suite_root: Path
    graph_npz: Path
    graph_nodes_parquet: Path
    graph_arcs_parquet: Path
    graph_metadata_json: Path
    od_npz: Path
    od_metadata_json: Path
    rich_sqlite: Path


@dataclass(slots=True)
class GraphData:
    snapshot: PartCRev1GraphSnapshot
    node_uid: np.ndarray
    arc_id: np.ndarray
    tail: np.ndarray
    head: np.ndarray
    arc_fftt_s: np.ndarray
    arc_length_m: np.ndarray
    base_capacity: np.ndarray
    remaining_capacity: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    county_fips: np.ndarray
    state_fips: np.ndarray
    physical_weak_component: np.ndarray
    physical_strong_component: np.ndarray
    validation: dict[str, Any] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return int(self.node_uid.size)

    @property
    def arc_count(self) -> int:
        return int(self.arc_id.size)


@dataclass(slots=True)
class SuiteData:
    suite_id: str
    suite_root: Path
    suite_identity: dict[str, Any]
    directed_arc_membership: pd.DataFrame
    portals: pd.DataFrame
    centroid_roles: pd.DataFrame
    terminals: pd.DataFrame
    node_incident_registry: pd.DataFrame
    retained_regions: pd.DataFrame
    is_final_backbone_arc: np.ndarray
    is_final_backbone_node: np.ndarray
    arc_region_id: np.ndarray
    region_ids: tuple[str, ...]


@dataclass(slots=True)
class ODData:
    snapshot: PartCRev1ODSnapshot
    validation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PairGraph:
    node_ids: tuple[str, ...]
    edge_u: np.ndarray
    edge_v: np.ndarray
    dissimilarity: np.ndarray
    component_ids: np.ndarray
    metric: str
    threshold: float
    threshold_unit: str

    def edge_count(self) -> int:
        return int(self.edge_u.size)

    def adjacency(self) -> tuple[dict[int, float], ...]:
        result: list[dict[int, float]] = [dict() for _ in self.node_ids]
        for u, v, value in zip(
            self.edge_u.tolist(), self.edge_v.tolist(), self.dissimilarity.tolist(), strict=True
        ):
            result[int(u)][int(v)] = float(value)
            result[int(v)][int(u)] = float(value)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ClusterResult:
    algorithm: str
    clusters: tuple[tuple[str, ...], ...]
    actual_diameters: tuple[float, ...]
    production_pairwise_valid: bool
    membership_hash: str


@dataclass(frozen=True, slots=True)
class AugPolicy:
    policy_id: str
    threshold_s: float
    search_radius_m: float
    max_physical_arcs: int
    max_length_geodesic_detour: float
    epsilon_distance_m: float


@dataclass(frozen=True, slots=True)
class AugWitness:
    witness_id: str
    policy_id: str
    source_node_idx: int
    target_node_idx: int
    source_node_uid: str
    target_node_uid: str
    fftt_s: float
    length_m: float
    arc_indices: tuple[int, ...]
    arc_ids: tuple[str, ...]
    crossed_region_ids: tuple[str, ...]
    detour_ratio: float
    witness_sha256: str


@dataclass(frozen=True, slots=True)
class AugWitnessSearchStats:
    """Deterministic counters for one bounded direct-witness search."""

    source_node_idx: int
    states_settled: int
    arcs_scanned: int
    reached_other_backbone_count: int
    accepted_witness_count: int
    rejected_radius_count: int
    rejected_detour_count: int


@dataclass(frozen=True, slots=True)
class TemporaryAugmentedAdjacency:
    """Primitive CSR representation of the full temporary AUG universe."""

    backbone_node_indices: np.ndarray
    indptr: np.ndarray
    head_local: np.ndarray
    cost_s: np.ndarray
    adjacency_sha256: str

    @property
    def node_count(self) -> int:
        return int(self.backbone_node_indices.size)

    @property
    def arc_count(self) -> int:
        return int(self.head_local.size)


@dataclass(frozen=True, slots=True)
class AugmentedDistanceShard:
    """Compact CSR-like bounded distances for an explicit source shard."""

    policy_id: str
    threshold_s: float
    adjacency_sha256: str
    source_local: np.ndarray
    source_offsets: np.ndarray
    target_local: np.ndarray
    distance_s: np.ndarray
    states_settled: np.ndarray
    arcs_scanned: np.ndarray

    @property
    def source_count(self) -> int:
        return int(self.source_local.size)

    @property
    def distance_count(self) -> int:
        return int(self.target_local.size)


@dataclass(frozen=True, slots=True)
class RegionalPairWitness:
    pair_id: str
    region_id: str
    source_terminal_id: str
    target_terminal_id: str
    source_node_idx: int
    target_node_idx: int
    fftt_s: float
    arc_indices: tuple[int, ...]
    arc_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProjectionOccurrence:
    occurrence_index: int
    kind: str
    object_id: str
    region_id: str | None
    arc_offset_begin: int
    arc_offset_end: int
    physical_arc_ids: tuple[str, ...]
    physical_fftt_s: float
    coarse_tail_id: str | None
    coarse_head_id: str | None
    omitted_online: bool


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    occurrences: tuple[ProjectionOccurrence, ...]
    coarse_sequence: tuple[str, ...]
    classifications: tuple[str, ...]
    accounted_arc_count: int
    physical_arc_count: int
    repeated_region_entry_count: int
    regional_transition_count: int
    omitted_suppressed_fftt_s: float
    omitted_aug_fftt_s: float
    omitted_racn_fftt_s: float


@dataclass(frozen=True, slots=True)
class PhysicalTransition:
    tail_state: int
    head_state: int
    fftt_s: float
    arc_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CoarseEdgeRelation:
    coarse_edge_id: str
    tail_coarse_node_id: str
    head_coarse_node_id: str
    transitions: tuple[PhysicalTransition, ...]


@dataclass(frozen=True, slots=True)
class LiftabilityResult:
    liftable: bool
    failure_step: int | None
    failure_reason: str
    lifted_fftt_s: float | None
    physical_arc_ids: tuple[str, ...] | None
    final_state: int | None


NeighborProvider = Callable[[str], Iterable[tuple[str, str, float]]]
ReachabilityProvider = Callable[[int, int, frozenset[str]], tuple[float, tuple[int, ...]] | None]
