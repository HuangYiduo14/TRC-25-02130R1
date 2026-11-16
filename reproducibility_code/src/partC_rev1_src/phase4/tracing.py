"""Phase 3 lookup loading and Phase 4 trace-vector assembly."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from partC_rev1_src.partC_rev1_graph_snapshot import PartCRev1GraphSnapshot

from .models import SparseFlowVector, SuiteLinearTrace, SuiteTraceMap, UINT64_SENTINEL


def _directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        file_digest = hashlib.sha256()
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(file_digest.hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sorted_registry(frame: pd.DataFrame, column: str) -> tuple[str, ...]:
    return tuple(sorted(frame[column].astype(str).unique().tolist()))


def _load_one_suite(
    *,
    suite_id: str,
    root: Path,
    graph: PartCRev1GraphSnapshot,
    expected_digest: str,
    verify_directory_digest: bool,
) -> SuiteTraceMap:
    if suite_id == "s01":
        raise ValueError("s01 is excluded from Phase 4 and cannot be loaded")
    required = [
        "suite_metadata.json",
        "tracer_interface.json",
        "physical_to_candidate_lookup.npz",
        "bundle_summary.parquet",
        "bacn_summary.parquet",
        "racn_summary.parquet",
        "consolidated_terminals.parquet",
        "terminal_fine_members.parquet",
        "regional_pair_support_summary.parquet",
        "regional_support",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"{suite_id} trace input is incomplete: {missing}")
    actual_digest = _directory_digest(root) if verify_directory_digest else expected_digest
    if actual_digest != expected_digest:
        raise ValueError(f"{suite_id} production directory digest differs from external approval")

    metadata = json.loads((root / "suite_metadata.json").read_text(encoding="utf-8"))
    interface = json.loads((root / "tracer_interface.json").read_text(encoding="utf-8"))
    if str(metadata["suite_short_id"]) != suite_id:
        raise ValueError(f"{suite_id} suite metadata identity mismatch")
    if bool(metadata.get("phase4_executed", False)):
        raise ValueError(f"{suite_id} accepted Phase 3 product was unexpectedly marked as Phase 4 output")
    if interface.get("sentinel") != -1:
        raise ValueError(f"{suite_id} tracer sentinel differs from the frozen contract")

    bundles = pd.read_parquet(root / "bundle_summary.parquet", columns=["bundle_id"])
    bacns = pd.read_parquet(root / "bacn_summary.parquet", columns=["bacn_id"])
    racns = pd.read_parquet(root / "racn_summary.parquet", columns=["racn_id"])
    terminals = pd.read_parquet(
        root / "consolidated_terminals.parquet",
        columns=[
            "region_id",
            "terminal_id",
            "terminal_type",
            "parent_object_id",
            "can_enter_region",
            "can_exit_region",
        ],
    )
    support_summary = pd.read_parquet(
        root / "regional_pair_support_summary.parquet",
        columns=["region_id", "supported_pair_count"],
    )
    if support_summary["region_id"].astype(str).duplicated().any():
        raise ValueError(f"{suite_id} regional support summary has duplicate regions")
    support_summary = support_summary.assign(region_id=support_summary["region_id"].astype(str)).sort_values(
        "region_id", kind="mergesort"
    )
    region_ids = tuple(support_summary["region_id"].tolist())
    conceptual_by_region = support_summary["supported_pair_count"].to_numpy(dtype=np.int64)
    if int(conceptual_by_region.sum(dtype=np.int64)) != int(metadata["exact_regional_pair_count"]):
        raise ValueError(f"{suite_id} exact regional-pair total differs from suite metadata")

    bundle_ids = _sorted_registry(bundles, "bundle_id")
    bacn_ids = _sorted_registry(bacns, "bacn_id")
    racn_ids = _sorted_registry(racns, "racn_id")
    terminal_ids = _sorted_registry(terminals, "terminal_id")
    region_local = {value: index for index, value in enumerate(region_ids)}
    terminal_local = {value: index for index, value in enumerate(terminal_ids)}
    racn_local = {value: index for index, value in enumerate(racn_ids)}

    node_count = len(graph.node_idx_to_uid)
    arc_count = len(graph.arc_idx_to_directed_arc_id)
    node_uid_to_idx = graph.node_uid_to_idx
    fine = pd.read_parquet(
        root / "terminal_fine_members.parquet",
        columns=["region_id", "terminal_id", "partC_rev1_node_uid"],
    )
    fine_region = fine["region_id"].astype(str).map(region_local)
    fine_terminal = fine["terminal_id"].astype(str).map(terminal_local)
    fine_node = fine["partC_rev1_node_uid"].astype(str).map(node_uid_to_idx)
    if fine_region.isna().any() or fine_terminal.isna().any() or fine_node.isna().any():
        raise ValueError(f"{suite_id} terminal fine members do not close to graph/registries")
    keys = (
        fine_region.to_numpy(dtype=np.uint64) * np.uint64(node_count)
        + fine_node.to_numpy(dtype=np.uint64)
    )
    member_terminal_values = fine_terminal.to_numpy(dtype=np.int32)
    order = np.argsort(keys, kind="mergesort")
    keys, member_terminal_values = keys[order], member_terminal_values[order]

    terminal_rows = terminals.assign(terminal_id=terminals["terminal_id"].astype(str)).drop_duplicates(
        "terminal_id", keep="first"
    )
    terminal_rows = terminal_rows.set_index("terminal_id").loc[list(terminal_ids)]
    can_enter = terminal_rows["can_enter_region"].to_numpy(dtype=bool)
    can_exit = terminal_rows["can_exit_region"].to_numpy(dtype=bool)
    terminal_types = terminal_rows["terminal_type"].astype(str).to_numpy()
    # A region/node can legitimately expose separate source/target centroid
    # roles.  Preserve the exact Phase 3 directional selector:
    # source requires can_exit_region, target requires can_enter_region;
    # OD endpoints prefer their matching centroid role, while non-endpoint
    # boundaries prefer the unique physical B-ACN/R-ACN terminal.
    if len(keys):
        starts = np.r_[True, keys[1:] != keys[:-1]]
        locations = np.flatnonzero(starts)
        ends = np.r_[locations[1:], len(keys)]
        directional_values = np.full((4, len(locations)), -1, dtype=np.int32)
        ambiguous_counts = np.zeros(4, dtype=np.int64)

        def select(candidates: np.ndarray, *, source: bool, prefer_centroid: bool) -> int:
            direction = can_exit if source else can_enter
            eligible = np.unique(candidates[direction[candidates]])
            preferred_type = "ORIGIN_CENTROID" if source else "DESTINATION_CENTROID"
            if prefer_centroid:
                preferred = eligible[terminal_types[eligible] == preferred_type]
                if len(preferred) == 1:
                    return int(preferred[0])
            physical = eligible[np.isin(terminal_types[eligible], ("BACN", "RACN"))]
            if len(physical) == 1:
                return int(physical[0])
            if len(eligible) == 1:
                return int(eligible[0])
            return -1

        for position, (begin, end) in enumerate(zip(locations, ends, strict=True)):
            candidates = member_terminal_values[begin:end]
            selections = (
                select(candidates, source=True, prefer_centroid=False),
                select(candidates, source=False, prefer_centroid=False),
                select(candidates, source=True, prefer_centroid=True),
                select(candidates, source=False, prefer_centroid=True),
            )
            directional_values[:, position] = selections
            for mode, value in enumerate(selections):
                if value < 0:
                    ambiguous_counts[mode] += 1
        keys = keys[locations]
    else:
        directional_values = np.empty((4, 0), dtype=np.int32)
        ambiguous_counts = np.zeros(4, dtype=np.int64)
    terminal_region = terminal_rows["region_id"].astype(str).map(region_local)
    if terminal_region.isna().any():
        raise ValueError(f"{suite_id} consolidated terminal region is missing from the registry")
    parent_racn = np.full(len(terminal_ids), -1, dtype=np.int32)
    parent_text = terminal_rows["parent_object_id"].astype(str)
    is_racn_terminal = terminal_rows["terminal_type"].astype(str).eq("RACN")
    is_racn = parent_text.isin(racn_local)
    if is_racn.any():
        mapped = parent_text[is_racn].map(racn_local)
        parent_racn[np.flatnonzero(is_racn.to_numpy())] = mapped.to_numpy(dtype=np.int32)
    singleton_racn_control_ids = sorted(
        parent_text[is_racn_terminal & ~is_racn].astype(str).unique().tolist()
    )

    with np.load(root / "physical_to_candidate_lookup.npz", allow_pickle=False) as lookup:
        node_to_bacn = lookup["node_to_bacn_local"].astype(np.int32)
        node_to_racn = lookup["node_to_racn_local"].astype(np.int32)
        arc_to_bundle = lookup["arc_to_bundle_local"].astype(np.int32)
        arc_internal = lookup["arc_internal_bacn_local"].astype(np.int32)
    for name, values, registry_size in (
        ("bundle", arc_to_bundle, len(bundle_ids)),
        ("B-ACN", arc_internal, len(bacn_ids)),
        ("R-ACN", node_to_racn, len(racn_ids)),
    ):
        if np.any(values < -1) or np.any(values >= registry_size):
            raise ValueError(f"{suite_id} {name} lookup contains an out-of-registry index")

    arc_region = np.full(arc_count, -1, dtype=np.int32)
    support_root = root / "regional_support"
    support_files = {path.stem: path for path in support_root.glob("*.npz")}
    if set(support_files) != set(region_ids):
        raise ValueError(f"{suite_id} regional-support files differ from the accepted region registry")
    for local, region_id in enumerate(region_ids):
        with np.load(support_files[region_id], allow_pickle=False) as support:
            if str(support["region_id"].item()) != region_id:
                raise ValueError(f"{suite_id} regional-support identity mismatch for {region_id}")
            arc_indexes = support["arc_index"].astype(np.int64, copy=False)
        if np.any(arc_indexes < 0) or np.any(arc_indexes >= arc_count):
            raise ValueError(f"{suite_id} regional support contains an invalid physical arc index")
        if np.any(arc_region[arc_indexes] >= 0):
            raise ValueError(f"{suite_id} physical arc belongs to more than one retained region")
        arc_region[arc_indexes] = local

    trace_map = SuiteTraceMap(
        suite_id=suite_id,
        phase2_suite_id=str(metadata["phase2_suite_id"]),
        candidate_id=str(metadata["candidate_id"]),
        root=root,
        arc_region_local=arc_region,
        arc_to_bundle_local=arc_to_bundle,
        arc_internal_bacn_local=arc_internal,
        node_to_bacn_local=node_to_bacn,
        node_to_racn_local=node_to_racn,
        terminal_region_node_keys=keys.astype(np.uint64, copy=False),
        terminal_region_node_values=directional_values,
        terminal_can_enter=can_enter,
        terminal_can_exit=can_exit,
        terminal_region_local=terminal_region.to_numpy(dtype=np.int32),
        terminal_parent_racn_local=parent_racn,
        terminal_ids=terminal_ids,
        region_ids=region_ids,
        bundle_ids=bundle_ids,
        bacn_ids=bacn_ids,
        racn_ids=racn_ids,
        conceptual_regional_pair_count=int(metadata["exact_regional_pair_count"]),
        conceptual_pair_count_by_region=conceptual_by_region,
        directory_digest=actual_digest,
        metadata={
            "suite_metadata": metadata,
            "tracer_interface": interface,
            "directional_terminal_lookup_modes": [
                "source_non_endpoint_can_exit",
                "target_non_endpoint_can_enter",
                "source_endpoint_prefer_origin_centroid",
                "target_endpoint_prefer_destination_centroid",
            ],
            "unresolved_directional_region_node_key_counts": ambiguous_counts.tolist(),
            "terminal_declared_singleton_racn_control_ids": singleton_racn_control_ids,
            "singleton_racn_controls_traced_via_terminal_connector_flow": True,
        },
    )
    trace_map.validate(node_count=node_count, arc_count=arc_count)
    return trace_map


def load_suite_trace_maps(
    config: str | Path | Mapping[str, Any],
    graph: PartCRev1GraphSnapshot,
    *,
    suite_ids: Sequence[str] | None = None,
    verify_directory_digests: bool = True,
) -> list[SuiteTraceMap]:
    """Load exactly the externally accepted Phase 4 suite portfolio."""

    if isinstance(config, Mapping):
        cfg = dict(config)
    else:
        cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    requested = list(suite_ids or cfg["suite_ids"])
    authorized = list(cfg["suite_ids"])
    if any(value not in authorized for value in requested) or "s01" in requested:
        raise ValueError("Requested trace scope is outside the accepted s02-s06 portfolio")
    phase3_root = Path(cfg["input_versions"]["phase3_root"])
    digests = cfg["input_versions"]["suite_directory_digests"]
    return [
        _load_one_suite(
            suite_id=suite_id,
            root=phase3_root / "suites" / suite_id / "parquet",
            graph=graph,
            expected_digest=str(digests[suite_id]),
            verify_directory_digest=verify_directory_digests,
        )
        for suite_id in requested
    ]


@dataclass(frozen=True, slots=True)
class PackedTraceMaps:
    suite_ids: tuple[str, ...]
    arc_region_local: np.ndarray
    arc_to_bundle_local: np.ndarray
    arc_internal_bacn_local: np.ndarray
    terminal_keys: np.ndarray
    terminal_values: np.ndarray
    terminal_offsets: np.ndarray
    region_counts: np.ndarray
    bacn_counts: np.ndarray


def pack_trace_maps(trace_maps: Sequence[SuiteTraceMap]) -> PackedTraceMaps:
    if not trace_maps:
        return PackedTraceMaps(
            suite_ids=(),
            arc_region_local=np.empty((0, 0), dtype=np.int32),
            arc_to_bundle_local=np.empty((0, 0), dtype=np.int32),
            arc_internal_bacn_local=np.empty((0, 0), dtype=np.int32),
            terminal_keys=np.empty(0, dtype=np.uint64),
            terminal_values=np.empty((4, 0), dtype=np.int32),
            terminal_offsets=np.zeros(1, dtype=np.int64),
            region_counts=np.empty(0, dtype=np.int32),
            bacn_counts=np.empty(0, dtype=np.int32),
        )
    arc_region = np.ascontiguousarray(np.stack([item.arc_region_local for item in trace_maps]))
    arc_bundle = np.ascontiguousarray(np.stack([item.arc_to_bundle_local for item in trace_maps]))
    arc_internal = np.ascontiguousarray(np.stack([item.arc_internal_bacn_local for item in trace_maps]))
    offsets = np.zeros(len(trace_maps) + 1, dtype=np.int64)
    for index, item in enumerate(trace_maps):
        offsets[index + 1] = offsets[index] + len(item.terminal_region_node_keys)
    keys = np.concatenate([item.terminal_region_node_keys for item in trace_maps]).astype(np.uint64, copy=False)
    values = np.concatenate(
        [item.terminal_region_node_values for item in trace_maps], axis=1
    ).astype(np.int32, copy=False)
    return PackedTraceMaps(
        suite_ids=tuple(item.suite_id for item in trace_maps),
        arc_region_local=arc_region,
        arc_to_bundle_local=arc_bundle,
        arc_internal_bacn_local=arc_internal,
        terminal_keys=np.ascontiguousarray(keys),
        terminal_values=np.ascontiguousarray(values),
        terminal_offsets=offsets,
        region_counts=np.asarray([item.region_count for item in trace_maps], dtype=np.int32),
        bacn_counts=np.asarray([item.bacn_count for item in trace_maps], dtype=np.int32),
    )


def hash_table_to_sparse(keys: np.ndarray, values: np.ndarray) -> SparseFlowVector:
    occupied = keys != UINT64_SENTINEL
    return SparseFlowVector.from_unsorted(keys[occupied], values[occupied], zero_tol=0.0)


def terminal_and_racn_flow(
    coarse_pairs: SparseFlowVector,
    trace_map: SuiteTraceMap,
) -> tuple[np.ndarray, np.ndarray]:
    terminal = np.zeros(trace_map.terminal_count, dtype=np.float64)
    if len(coarse_pairs.keys):
        entry = (coarse_pairs.keys >> np.uint64(32)).astype(np.int64)
        exit_ = (coarse_pairs.keys & np.uint64(0xFFFFFFFF)).astype(np.int64)
        if np.any(entry >= trace_map.terminal_count) or np.any(exit_ >= trace_map.terminal_count):
            raise ValueError(f"{trace_map.suite_id} coarse pair key contains an invalid terminal")
        terminal += np.bincount(entry, weights=coarse_pairs.values, minlength=trace_map.terminal_count)
        terminal += np.bincount(exit_, weights=coarse_pairs.values, minlength=trace_map.terminal_count)
    racn = np.zeros(trace_map.racn_count, dtype=np.float64)
    valid = trace_map.terminal_parent_racn_local >= 0
    if np.any(valid):
        racn += np.bincount(
            trace_map.terminal_parent_racn_local[valid],
            weights=terminal[valid],
            minlength=trace_map.racn_count,
        )
    return terminal, racn


def assemble_suite_aon_trace(
    *,
    trace_map: SuiteTraceMap,
    physical_flow: np.ndarray,
    coarse_pairs: SparseFlowVector,
    fine_pairs: SparseFlowVector,
    bacn_traversal_flow: np.ndarray,
    repeated_region_entry_flow: np.ndarray,
) -> SuiteLinearTrace:
    bundle_mask = trace_map.arc_to_bundle_local >= 0
    internal_mask = trace_map.arc_internal_bacn_local >= 0
    bundle_flow = np.bincount(
        trace_map.arc_to_bundle_local[bundle_mask],
        weights=physical_flow[bundle_mask],
        minlength=trace_map.bundle_count,
    ).astype(np.float64, copy=False)
    internal_flow = np.bincount(
        trace_map.arc_internal_bacn_local[internal_mask],
        weights=physical_flow[internal_mask],
        minlength=trace_map.bacn_count,
    ).astype(np.float64, copy=False)
    terminal, racn = terminal_and_racn_flow(coarse_pairs, trace_map)
    result = SuiteLinearTrace(
        suite_id=trace_map.suite_id,
        bundle_flow=bundle_flow,
        coarse_regional_pair_flow=coarse_pairs,
        fine_regional_pair_flow=fine_pairs,
        bacn_internal_arc_flow=internal_flow,
        bacn_internal_traversal_flow=np.asarray(bacn_traversal_flow, dtype=np.float64),
        terminal_connector_flow=terminal,
        racn_interface_flow=racn,
        repeated_region_entry_flow=np.asarray(repeated_region_entry_flow, dtype=np.float64),
    )
    result.validate(trace_map)
    return result
