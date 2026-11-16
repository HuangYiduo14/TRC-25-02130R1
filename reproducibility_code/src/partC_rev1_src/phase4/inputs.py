"""Immutable Phase 4 graph/OD loading and exact duplicate-OD aggregation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from partC_rev1_src.partC_rev1_graph_snapshot import (
    PartCRev1GraphSnapshot,
    partC_rev1_load_graph_snapshot,
)
from partC_rev1_src.partC_rev1_od_snapshot import partC_rev1_load_od_snapshot

from .models import AggregatedOD


def _read_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    return json.loads(Path(config).read_text(encoding="utf-8"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def write_json_new(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload) + b"\n"
    with destination.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def load_phase4_graph(config: str | Path | Mapping[str, Any]) -> PartCRev1GraphSnapshot:
    """Load and hard-validate the accepted physical graph without modifying it."""

    cfg = _read_config(config)
    inputs = cfg["input_versions"]
    graph = partC_rev1_load_graph_snapshot(inputs["graph_npz"], inputs["graph_metadata_json"])
    expected_hash = str(inputs["graph_identity"])
    expected_nodes = int(inputs["graph_node_count"])
    expected_arcs = int(inputs["graph_arc_count"])
    if graph.graph_hash != expected_hash or graph.assignment_graph_hash != expected_hash:
        raise ValueError("Physical graph identity differs from the authorized graph")
    if len(graph.node_idx_to_uid) != expected_nodes or len(graph.arc_idx_to_directed_arc_id) != expected_arcs:
        raise ValueError("Physical graph dimensions differ from the authorized graph")
    numeric_arrays = {
        "free_time": graph.free_time,
        "prepared_capacity": graph.capacity,
        "alpha": graph.alpha,
        "beta": graph.beta,
        "length": graph.length,
    }
    for name, value in numeric_arrays.items():
        if value.shape != (expected_arcs,) or np.any(~np.isfinite(value)):
            raise ValueError(f"Physical graph {name} is invalid")
    if np.any(graph.free_time <= 0.0) or np.any(graph.capacity <= 0.0):
        raise ValueError("Physical free-flow times and prepared capacities must be positive")
    if np.any(graph.alpha < 0.0) or np.any(graph.beta <= 0.0):
        raise ValueError("Physical BPR parameters are invalid")
    if graph.csr_indptr.shape != (expected_nodes + 1,) or len(graph.csr_arc_idx) != expected_arcs:
        raise ValueError("Physical graph CSR arrays are invalid")
    if not np.array_equal(graph.csr_head, graph.head[graph.csr_arc_idx]):
        raise ValueError("Physical graph CSR head mapping is inconsistent")
    return graph


def _load_aggregated_od(
    aggregate_path: Path,
    mapping_path: Path,
    metadata_path: Path,
) -> AggregatedOD:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if file_sha256(aggregate_path) != metadata["aggregate_npz_sha256"]:
        raise ValueError("Existing aggregated OD artifact hash differs from its metadata")
    if file_sha256(mapping_path) != metadata["mapping_npz_sha256"]:
        raise ValueError("Existing OD row-map artifact hash differs from its metadata")
    with np.load(aggregate_path, allow_pickle=False) as data:
        result = AggregatedOD(
            od_hash=str(metadata["od_hash"]),
            origin_group_node_idx=data["origin_group_node_idx"].astype(np.int64, copy=False),
            origin_offsets=data["origin_offsets"].astype(np.int64, copy=False),
            destination_node_idx=data["destination_node_idx"].astype(np.int64, copy=False),
            demand=data["demand"].astype(np.float64, copy=False),
            original_row_multiplicity=data["original_row_multiplicity"].astype(np.int64, copy=False),
            original_row_count=int(metadata["original_row_count"]),
            original_total_demand=float(metadata["original_total_demand"]),
            mapping_path=mapping_path,
            metadata=metadata,
        )
    result.validate()
    return result


def load_or_build_aggregated_od(
    config: str | Path | Mapping[str, Any],
    *,
    output_root: str | Path,
    existing_root: str | Path | None = None,
) -> AggregatedOD:
    """Build or reopen the exact, reversible identical-OD aggregation."""

    cfg = _read_config(config)
    inputs = cfg["input_versions"]
    root = Path(existing_root or output_root) / "derived_inputs"
    aggregate_path = root / "aggregated_od.npz"
    mapping_path = root / "original_to_aggregated_od_index.npz"
    metadata_path = root / "aggregated_od_metadata.json"
    present = [path.exists() for path in (aggregate_path, mapping_path, metadata_path)]
    if existing_root is not None and not all(present):
        raise FileNotFoundError(
            "Accepted existing aggregated-OD root does not contain the complete immutable artifact set"
        )
    if any(present):
        if not all(present):
            raise ValueError("Aggregated OD artifact set is incomplete; write-new-only recovery is required")
        return _load_aggregated_od(aggregate_path, mapping_path, metadata_path)

    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = partC_rev1_load_od_snapshot(inputs["od_npz"], inputs["od_metadata_json"])
    if source.od_hash != str(inputs["od_identity"]):
        raise ValueError("OD snapshot identity differs from the authorized OD")
    if len(source.demand) != int(inputs["od_row_count"]):
        raise ValueError("OD snapshot row count differs from the authorized OD")
    if len(source.origin_group_node_idx) != int(inputs["origin_count"]):
        raise ValueError("OD origin count differs from the authorized OD")
    source_total = float(source.demand.sum(dtype=np.float64))
    if not np.isclose(source_total, float(inputs["total_demand"]), rtol=1e-12, atol=1e-9):
        raise ValueError("OD snapshot demand differs from the authorized total")
    if np.any(~np.isfinite(source.demand)) or np.any(source.demand <= 0.0):
        raise ValueError("OD snapshot contains nonpositive or nonfinite demand")
    if np.any(source.origin_node_idx[1:] < source.origin_node_idx[:-1]):
        raise ValueError("OD snapshot origins are not grouped in stable order")
    same_origin = source.origin_node_idx[1:] == source.origin_node_idx[:-1]
    if np.any(source.destination_node_idx[1:][same_origin] < source.destination_node_idx[:-1][same_origin]):
        raise ValueError("OD destinations are not sorted within origin")

    original_count = len(source.demand)
    starts_mask = np.empty(original_count, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = (
        (source.origin_node_idx[1:] != source.origin_node_idx[:-1])
        | (source.destination_node_idx[1:] != source.destination_node_idx[:-1])
    )
    starts = np.flatnonzero(starts_mask)
    aggregate_count = len(starts)
    if aggregate_count >= np.iinfo(np.int32).max:
        raise ValueError("Aggregated OD index exceeds the reversible int32 mapping domain")
    destination = source.destination_node_idx[starts].astype(np.int64, copy=True)
    demand = np.add.reduceat(source.demand, starts).astype(np.float64, copy=False)
    multiplicity = np.diff(np.r_[starts, original_count]).astype(np.int64, copy=False)
    aggregate_offsets = np.searchsorted(starts, source.origin_offsets, side="left").astype(np.int64)
    original_to_aggregate = np.cumsum(starts_mask, dtype=np.int64) - 1
    original_to_aggregate = original_to_aggregate.astype(np.int32, copy=False)

    result = AggregatedOD(
        od_hash=source.od_hash,
        origin_group_node_idx=source.origin_group_node_idx.astype(np.int64, copy=True),
        origin_offsets=aggregate_offsets,
        destination_node_idx=destination,
        demand=demand,
        original_row_multiplicity=multiplicity,
        original_row_count=original_count,
        original_total_demand=source_total,
        mapping_path=mapping_path,
        metadata={},
    )
    result.validate()

    with aggregate_path.open("xb") as stream:
        np.savez(
            stream,
            origin_group_node_idx=result.origin_group_node_idx,
            origin_offsets=result.origin_offsets,
            destination_node_idx=result.destination_node_idx,
            demand=result.demand,
            original_row_multiplicity=result.original_row_multiplicity,
        )
        stream.flush()
        os.fsync(stream.fileno())
    with mapping_path.open("xb") as stream:
        np.savez_compressed(stream, original_to_aggregated_index=original_to_aggregate)
        stream.flush()
        os.fsync(stream.fileno())

    metadata = {
        "schema_id": "phase4.exact_aggregated_od.v1",
        "od_hash": source.od_hash,
        "source_npz": str(Path(inputs["od_npz"]).resolve()),
        "source_npz_size_bytes": int(Path(inputs["od_npz"]).stat().st_size),
        "source_metadata": str(Path(inputs["od_metadata_json"]).resolve()),
        "original_row_count": int(original_count),
        "aggregated_row_count": int(aggregate_count),
        "row_reduction_fraction": float(1.0 - aggregate_count / original_count),
        "origin_count": int(len(result.origin_group_node_idx)),
        "original_total_demand": source_total,
        "aggregated_total_demand": float(result.demand.sum(dtype=np.float64)),
        "aggregation_rule": "sum exact duplicate rows with identical physical origin and destination",
        "commodity_independent_cost": True,
        "reversible_mapping": str(mapping_path.resolve()),
        "aggregate_npz": str(aggregate_path.resolve()),
        "aggregate_npz_size_bytes": int(aggregate_path.stat().st_size),
        "aggregate_npz_sha256": file_sha256(aggregate_path),
        "mapping_npz_size_bytes": int(mapping_path.stat().st_size),
        "mapping_npz_sha256": file_sha256(mapping_path),
        "build_seconds": float(time.perf_counter() - started),
    }
    write_json_new(metadata_path, metadata)
    return _load_aggregated_od(aggregate_path, mapping_path, metadata_path)
