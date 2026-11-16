"""Lightweight dataclasses used by later partC_rev1 modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PartCRev1DirectedArc:
    directed_arc_id: int
    physical_link_id: str
    from_node_id: str
    to_node_id: str
    direction: str
    length_miles: float
    free_time_min: float
    capacity: float
    alpha: float
    beta: float
    class_code: int | None
    nhs: int | None
    nhfn: int | None
    truck: str | None
    status: int | None
    road_name: str | None
    sign_route: str | None
    route_type: str | None
    route_number: str | None
    ctfips: str | None
    fafzone: str | None
    is_valid_for_truck: bool
    is_open: bool
    is_backbone: bool = False
    region_id: int | None = None


@dataclass(slots=True)
class PartCRev1Centroid:
    node_id: str
    centroid_id_raw: str
    county_fips: str
    fafid: str | None
    weight_within_county: float
    lon: float | None
    lat: float | None


@dataclass(slots=True)
class PartCRev1BoundaryEvent:
    event_id: int
    region_id: int
    neighbor_type: str
    neighbor_region_id: int | None
    event_direction: str
    physical_link_id: str
    directed_arc_id: int
    inside_node_id: str
    outside_node_id: str
    backbone_anchor_node_id: str | None
    lon: float | None
    lat: float | None


@dataclass(slots=True)
class PartCRev1PortalGroup:
    portal_group_id: int
    region_id: int
    neighbor_type: str
    neighbor_region_id: int | None
    representative_node_id: str | None
    num_events: int
    dispersion_miles: float | None
    entry_overlay_node_id: int
    exit_overlay_node_id: int


@dataclass(slots=True)
class PartCRev1OverlayNode:
    overlay_node_id: int
    node_type: str
    physical_node_id: str | None
    region_id: int | None
    portal_group_id: int | None
    centroid_county_fips: str | None


@dataclass(slots=True)
class PartCRev1OverlayOrdinaryArc:
    overlay_arc_id: int
    tail_overlay_node_id: int
    head_overlay_node_id: int
    arc_type: str
    physical_link_id: str | None
    directed_arc_id: int | None
    free_time_min: float
    capacity: float
    alpha: float
    beta: float
    flow: float = 0.0
    cost: float | None = None


@dataclass(slots=True)
class PartCRev1RegionTransferPair:
    transfer_pair_id: int
    region_id: int
    source_overlay_node_id: int
    target_overlay_node_id: int
    source_type: str
    target_type: str
    local_source_index: int
    local_target_index: int
    flat_index: int
    allowed: bool
    tau0: float


@dataclass(slots=True)
class PartCRev1SolverResult:
    algorithm: str
    converged: bool
    iterations: int
    final_relative_gap: float
    runtime_sec: float
    ordinary_arc_flow_path: str | None
    regional_transfer_flow_path: str | None
    physical_flow_output_path: str | None
    virtual_flow_output_path: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
