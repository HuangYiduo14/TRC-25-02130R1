"""Physical-to-candidate lookup arrays and simultaneous tracer interface."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .models import ProjectionResult
from .schemas import canonical_json_bytes


LOOKUP_SENTINEL = -1


class Phase3AMappingError(ValueError):
    """Raised for lookup or tracer closure failures."""


@dataclass(frozen=True, slots=True)
class PhysicalCandidateLookup:
    node_to_bacn_local: np.ndarray
    node_to_racn_local: np.ndarray
    arc_to_bundle_local: np.ndarray
    arc_internal_bacn_local: np.ndarray
    portal_role_to_terminal_local: np.ndarray
    centroid_role_to_terminal_local: np.ndarray
    bacn_registry: tuple[str, ...]
    racn_registry: tuple[str, ...]
    bundle_registry: tuple[str, ...]
    terminal_registry: tuple[str, ...]


def build_physical_candidate_lookup(
    *,
    node_uid: Sequence[str],
    arc_id: Sequence[str],
    bacn_membership: pd.DataFrame,
    racn_membership: pd.DataFrame,
    bundle_membership: pd.DataFrame,
    suppressed_arcs: pd.DataFrame,
    portal_roles: pd.DataFrame,
    centroid_roles: pd.DataFrame,
    terminals: pd.DataFrame,
    terminal_members: pd.DataFrame,
) -> PhysicalCandidateLookup:
    node_index = {str(value): index for index, value in enumerate(node_uid)}
    arc_index = {str(value): index for index, value in enumerate(arc_id)}
    bacn_registry = tuple(sorted(bacn_membership["bacn_id"].astype(str).unique().tolist()))
    racn_registry = tuple(sorted(racn_membership["racn_id"].astype(str).unique().tolist())) if not racn_membership.empty else ()
    bundle_registry = tuple(sorted(bundle_membership["bundle_id"].astype(str).unique().tolist())) if not bundle_membership.empty else ()
    terminal_registry = tuple(sorted(terminals["terminal_id"].astype(str).unique().tolist()))
    bacn_local = {value: index for index, value in enumerate(bacn_registry)}
    racn_local = {value: index for index, value in enumerate(racn_registry)}
    bundle_local = {value: index for index, value in enumerate(bundle_registry)}
    terminal_local = {value: index for index, value in enumerate(terminal_registry)}
    node_to_bacn = np.full(len(node_uid), LOOKUP_SENTINEL, dtype=np.int32)
    node_to_racn = np.full(len(node_uid), LOOKUP_SENTINEL, dtype=np.int32)
    arc_to_bundle = np.full(len(arc_id), LOOKUP_SENTINEL, dtype=np.int32)
    arc_internal = np.full(len(arc_id), LOOKUP_SENTINEL, dtype=np.int32)
    for row in bacn_membership.itertuples(index=False):
        node_to_bacn[node_index[str(row.partC_rev1_node_uid)]] = bacn_local[str(row.bacn_id)]
    for row in racn_membership.itertuples(index=False):
        node_to_racn[node_index[str(row.physical_portal_node_uid)]] = racn_local[str(row.racn_id)]
    for row in bundle_membership.itertuples(index=False):
        arc_to_bundle[arc_index[str(row.directed_arc_id)]] = bundle_local[str(row.bundle_id)]
    for row in suppressed_arcs.itertuples(index=False):
        arc_internal[arc_index[str(row.directed_arc_id)]] = bacn_local[str(row.bacn_id)]
    terminal_type_by_id = terminals.set_index("terminal_id")["terminal_type"].astype(str).to_dict()
    member_terminal = {
        (
            str(row.region_id),
            str(row.partC_rev1_node_uid),
            terminal_type_by_id[str(row.terminal_id)],
        ): terminal_local[str(row.terminal_id)]
        for row in terminal_members.itertuples(index=False)
    }
    portal_array = np.full(len(portal_roles), LOOKUP_SENTINEL, dtype=np.int32)
    for position, row in enumerate(portal_roles.itertuples(index=False)):
        uid, region = str(row.partC_rev1_node_uid), str(row.region_uuid)
        values = {
            member_terminal[key]
            for key in ((region, uid, "BACN"), (region, uid, "RACN"))
            if key in member_terminal
        }
        if len(values) != 1:
            raise Phase3AMappingError("Portal role does not map to exactly one consolidated B/R terminal")
        portal_array[position] = next(iter(values))
    centroid_array = np.full(len(centroid_roles), LOOKUP_SENTINEL, dtype=np.int32)
    for position, row in enumerate(centroid_roles.itertuples(index=False)):
        terminal_type = "ORIGIN_CENTROID" if str(row.centroid_role) == "centroid_origin" else "DESTINATION_CENTROID"
        key = (str(row.region_uuid), str(row.partC_rev1_node_uid), terminal_type)
        if key not in member_terminal:
            raise Phase3AMappingError("Centroid role lacks a separate consolidated terminal")
        centroid_array[position] = member_terminal[key]
    return PhysicalCandidateLookup(
        node_to_bacn, node_to_racn, arc_to_bundle, arc_internal,
        portal_array, centroid_array, bacn_registry, racn_registry, bundle_registry, terminal_registry,
    )


def write_lookup_npz_new(path: str | Path, lookup: PhysicalCandidateLookup) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise Phase3AMappingError(f"Lookup path already exists: {destination}")
    np.savez_compressed(
        destination,
        node_to_bacn_local=lookup.node_to_bacn_local,
        node_to_racn_local=lookup.node_to_racn_local,
        arc_to_bundle_local=lookup.arc_to_bundle_local,
        arc_internal_bacn_local=lookup.arc_internal_bacn_local,
        portal_role_to_terminal_local=lookup.portal_role_to_terminal_local,
        centroid_role_to_terminal_local=lookup.centroid_role_to_terminal_local,
    )
    digest = _file_sha256(destination)
    return {
        "path": str(destination.resolve()),
        "size_bytes": int(destination.stat().st_size),
        "sha256": digest,
        "arrays": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape), "sentinel": LOOKUP_SENTINEL}
            for name, value in (
                ("node_to_bacn_local", lookup.node_to_bacn_local),
                ("node_to_racn_local", lookup.node_to_racn_local),
                ("arc_to_bundle_local", lookup.arc_to_bundle_local),
                ("arc_internal_bacn_local", lookup.arc_internal_bacn_local),
                ("portal_role_to_terminal_local", lookup.portal_role_to_terminal_local),
                ("centroid_role_to_terminal_local", lookup.centroid_role_to_terminal_local),
            )
        },
    }


def tracer_interface_contract(lookup: PhysicalCandidateLookup) -> dict[str, Any]:
    registries = {
        "bacn": lookup.bacn_registry,
        "racn": lookup.racn_registry,
        "bundle": lookup.bundle_registry,
        "terminal": lookup.terminal_registry,
    }
    return {
        "schema_id": "phase3a.tracer_interface",
        "schema_version": "phase3a-1.0",
        "lookup_arrays": [
            "node_to_bacn_local", "node_to_racn_local", "arc_to_bundle_local",
            "arc_internal_bacn_local", "portal_role_to_terminal_local", "centroid_role_to_terminal_local",
        ],
        "sentinel": LOOKUP_SENTINEL,
        "stable_registry_hashes": {
            key: hashlib.sha256(canonical_json_bytes(value)).hexdigest() for key, value in registries.items()
        },
        "path_segmentation_state_machine": ["REGIONAL_PAIR", "BUNDLE", "INTERNAL_BACN", "INTERNAL_AUG", "INTERNAL_RACN"],
        "regional_pair_cache_api": "LazyRegionalPairOracle.query(source_terminal_id)",
        "flow_arrays": {
            "bundle": {"dtype": "float64", "shape": [len(lookup.bundle_registry)]},
            "regional_pair": {"dtype": "float64", "shape": ["lazy_pair_registry_size"]},
            "internal_bacn": {"dtype": "float64", "shape": [len(lookup.bacn_registry)]},
            "internal_aug": {"dtype": "float64", "shape": [len(lookup.bacn_registry)]},
            "internal_racn": {"dtype": "float64", "shape": [len(lookup.racn_registry)]},
        },
        "repeated_region_entry_semantics": "every maximal occurrence retained in order",
        "linear_update_contract": "x_next=(1-alpha)*x_current+alpha*y_aux for the same physical FW/MSA/CFW alpha",
        "sampled_witness_retention": "full witnesses only for frozen validation sample and failures",
        "conservation_checks": ["classified demand", "bundle flow", "regional pair flow", "component demand"],
        "centroid_through_allowed": False,
        "phase4_execution_authorized": False,
    }


def empty_trace_arrays(
    lookup: PhysicalCandidateLookup, regional_pair_registry: Sequence[str]
) -> dict[str, np.ndarray]:
    return {
        "bundle": np.zeros(len(lookup.bundle_registry), dtype=np.float64),
        "regional_pair": np.zeros(len(regional_pair_registry), dtype=np.float64),
        "internal_bacn": np.zeros(len(lookup.bacn_registry), dtype=np.float64),
        "internal_aug": np.zeros(len(lookup.bacn_registry), dtype=np.float64),
        "internal_racn": np.zeros(len(lookup.racn_registry), dtype=np.float64),
    }


def trace_projection(
    projection: ProjectionResult,
    demand: float,
    arrays: Mapping[str, np.ndarray],
    *,
    bundle_local: Mapping[str, int],
    regional_pair_local: Mapping[str, int],
    bacn_local: Mapping[str, int],
    racn_local: Mapping[str, int],
) -> None:
    value = float(demand)
    if not np.isfinite(value) or value < 0.0:
        raise Phase3AMappingError("Trace demand must be finite and nonnegative")
    for occurrence in projection.occurrences:
        if occurrence.kind == "BUNDLE":
            arrays["bundle"][bundle_local[occurrence.object_id]] += value
        elif occurrence.kind == "REGIONAL_PAIR":
            arrays["regional_pair"][regional_pair_local[occurrence.object_id]] += value
        elif occurrence.kind == "INTERNAL_BACN":
            arrays["internal_bacn"][bacn_local[occurrence.object_id]] += value
        elif occurrence.kind == "INTERNAL_AUG":
            arrays["internal_aug"][bacn_local[occurrence.object_id]] += value
        elif occurrence.kind == "INTERNAL_RACN":
            arrays["internal_racn"][racn_local[occurrence.object_id]] += value
        else:
            raise Phase3AMappingError(f"Unknown projection occurrence: {occurrence.kind}")


def apply_linear_assignment_update(
    current: Mapping[str, np.ndarray], auxiliary: Mapping[str, np.ndarray], alpha: float
) -> dict[str, np.ndarray]:
    step = float(alpha)
    if not np.isfinite(step) or not 0.0 <= step <= 1.0 or set(current) != set(auxiliary):
        raise Phase3AMappingError("Tracer update inputs or coefficient are invalid")
    result: dict[str, np.ndarray] = {}
    for key in current:
        left, right = np.asarray(current[key], dtype=np.float64), np.asarray(auxiliary[key], dtype=np.float64)
        if left.shape != right.shape:
            raise Phase3AMappingError("Tracer arrays differ in shape")
        result[key] = (1.0 - step) * left + step * right
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
