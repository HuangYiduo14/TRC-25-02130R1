"""Approved Phase 1 OD projection and zero-length-arc policy helpers.

The module deliberately exposes streaming/batch operations.  It never writes a
full solver OD projection and its CLI reads only the first configured origin
groups from the immutable OD NPZ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


NETWORK_LOADABLE = "NETWORK_LOADABLE"
INTRAZONAL_NON_NETWORK_LOADED = "INTRAZONAL_NON_NETWORK_LOADED"
UNREACHABLE_CROSS_COMPONENT = "UNREACHABLE_CROSS_COMPONENT"
_CATEGORIES = (
    NETWORK_LOADABLE,
    INTRAZONAL_NON_NETWORK_LOADED,
    UNREACHABLE_CROSS_COMPONENT,
)
_CAMPAIGN_ID = "phase1_completion_campaign_20260812_v1"
_RUN_ID = "phase1_completion_campaign_20260812_v1_run_001"
_APPROVAL_RELATIVE = Path(
    "partC_doc_icnn/review_gates/phase_01/completion_campaign/"
    "phase1_completion_preflight_20260812T041155Z/APPROVAL.json"
)
_REGISTRY_RELATIVE = Path("partC_rev1_cache/phase1_local/phase1_data_roots.local.json")
_MANIFEST_RELATIVE = Path("partC_doc_icnn/phase_1_work/manifests/phase1_authoritative_inputs.v1.json")
_PREDICATES_RELATIVE = Path(
    "partC_doc_icnn/phase_1_work/completion_campaign/preflight/"
    "phase1_completion_campaign_20260812_v1/PHASE1_COMPLETION_POLICY_PREDICATES.json"
)
_RUN_MATRIX_RELATIVE = Path(
    "partC_doc_icnn/phase_1_work/completion_campaign/preflight/"
    "phase1_completion_campaign_20260812_v1/PHASE1_COMPLETION_RUN_MATRIX.json"
)
_RESOURCE_POLICY_RELATIVE = Path(
    "partC_doc_icnn/phase_1_work/completion_campaign/preflight/"
    "phase1_completion_campaign_20260812_v1/PHASE1_COMPLETION_RESOURCE_POLICY.json"
)


class PartCRev1DataPolicyError(ValueError):
    """Raised when approved data-policy predicates or boundaries do not hold."""


@dataclass(frozen=True, slots=True)
class PartCRev1SingleClassPredicate:
    passed: bool
    single_arc_cost_vector: bool
    single_arc_capacity_vector: bool
    scalar_od_demand: bool
    commodity_dependent_semantics: bool
    graph_arc_count: int
    od_row_count: int
    details: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "single_arc_cost_vector": self.single_arc_cost_vector,
            "single_arc_capacity_vector": self.single_arc_capacity_vector,
            "scalar_od_demand": self.scalar_od_demand,
            "commodity_dependent_semantics": self.commodity_dependent_semantics,
            "graph_arc_count": self.graph_arc_count,
            "od_row_count": self.od_row_count,
            "details": list(self.details),
        }


@dataclass(frozen=True, slots=True)
class PartCRev1ODProjectionBatch:
    origin_node_idx: np.ndarray
    destination_node_idx: np.ndarray
    origin_node_uid: np.ndarray
    destination_node_uid: np.ndarray
    demand: np.ndarray
    classification: np.ndarray
    raw_row_count: np.ndarray
    raw_row_offset_min: np.ndarray
    raw_row_offset_max: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        lengths = {
            len(self.origin_node_idx),
            len(self.destination_node_idx),
            len(self.origin_node_uid),
            len(self.destination_node_uid),
            len(self.demand),
            len(self.classification),
            len(self.raw_row_count),
            len(self.raw_row_offset_min),
            len(self.raw_row_offset_max),
        }
        if len(lengths) != 1:
            raise PartCRev1DataPolicyError("OD projection batch vectors must have identical lengths")

    @property
    def row_count(self) -> int:
        return int(len(self.demand))

    def report(self) -> dict[str, Any]:
        categories: dict[str, dict[str, float | int]] = {}
        for category in _CATEGORIES:
            mask = self.classification == category
            categories[category] = {
                "rows": int(np.count_nonzero(mask)),
                "raw_rows": int(np.sum(self.raw_row_count[mask], dtype=np.int64)),
                "demand": _finite_sum(self.demand[mask]),
            }
        raw_demand = _finite_sum(self.demand)
        return {
            "raw_rows": int(np.sum(self.raw_row_count, dtype=np.int64)),
            "raw_positive_demand": raw_demand,
            "network_loadable_demand": categories[NETWORK_LOADABLE]["demand"],
            "intrazonal_non_network_loaded_demand": categories[INTRAZONAL_NON_NETWORK_LOADED]["demand"],
            "unreachable_cross_component_demand": categories[UNREACHABLE_CROSS_COMPONENT]["demand"],
            "other_excluded_demand": 0.0,
            "categories": categories,
        }


def _finite_sum(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(array)):
        raise PartCRev1DataPolicyError("Demand totals require finite values")
    return float(math.fsum(float(value) for value in array))


def _value(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def partC_rev1_inspect_single_class_assignment(graph: Any, od_snapshot: Any) -> PartCRev1SingleClassPredicate:
    """Confirm one graph-level cost/capacity vector and scalar OD demand."""

    free_time = np.asarray(_value(graph, "free_time"), dtype=np.float64)
    capacity = np.asarray(_value(graph, "capacity"), dtype=np.float64)
    demand = np.asarray(_value(od_snapshot, "demand"), dtype=np.float64)
    origin = np.asarray(_value(od_snapshot, "origin_node_idx"), dtype=np.int64)
    destination = np.asarray(_value(od_snapshot, "destination_node_idx"), dtype=np.int64)
    single_cost = free_time.ndim == 1 and free_time.size > 0 and np.all(np.isfinite(free_time)) and np.all(free_time > 0)
    single_capacity = (
        capacity.ndim == 1
        and capacity.shape == free_time.shape
        and np.all(np.isfinite(capacity))
        and np.all(capacity > 0)
    )
    scalar_demand = (
        demand.ndim == 1
        and demand.size > 0
        and origin.shape == demand.shape
        and destination.shape == demand.shape
        and np.all(np.isfinite(demand))
        and np.all(demand > 0)
    )
    metadata: dict[str, Any] = {}
    for source in (graph, od_snapshot):
        value = _value(source, "metadata")
        if isinstance(value, Mapping):
            metadata.update(value)
    forbidden_names = (
        "commodity_dependent",
        "commodity_specific_cost",
        "commodity_specific_capacity",
        "class_specific_cost",
        "class_specific_constraint",
    )
    commodity_dependent = any(bool(metadata.get(name, False)) for name in forbidden_names)
    commodity_dependent = commodity_dependent or any(
        _value(graph, name) is not None or _value(od_snapshot, name) is not None
        for name in ("commodity_cost", "commodity_capacity", "class_cost", "class_constraint")
    )
    details = (
        "one graph-level free-time/cost vector",
        "one graph-level assignment-capacity vector",
        "one scalar positive demand vector",
        "no commodity/class-specific cost, capacity, feasibility, or constraint field",
    )
    passed = bool(single_cost and single_capacity and scalar_demand and not commodity_dependent)
    return PartCRev1SingleClassPredicate(
        passed=passed,
        single_arc_cost_vector=bool(single_cost),
        single_arc_capacity_vector=bool(single_capacity),
        scalar_od_demand=bool(scalar_demand),
        commodity_dependent_semantics=bool(commodity_dependent),
        graph_arc_count=int(free_time.size if free_time.ndim == 1 else 0),
        od_row_count=int(demand.size if demand.ndim == 1 else 0),
        details=details,
    )


def _component_and_uid_arrays(component_id_by_node: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(component_id_by_node, Mapping):
        component_source = component_id_by_node.get("component_id_by_node")
        uid_source = component_id_by_node.get("node_uid_by_idx")
    else:
        component_source = component_id_by_node
        uid_source = None
    components = np.asarray(component_source, dtype=np.int64)
    if components.ndim != 1 or components.size == 0:
        raise PartCRev1DataPolicyError("component_id_by_node must be a nonempty vector")
    if uid_source is None:
        width = max(1, len(str(max(int(components.size - 1), 0))))
        uids = np.arange(components.size, dtype=np.int64).astype(f"U{width}")
    else:
        uids = np.asarray(uid_source).astype(str)
        if uids.ndim != 1 or uids.shape != components.shape:
            raise PartCRev1DataPolicyError("node_uid_by_idx must align with component_id_by_node")
        if np.any(np.char.strip(uids) == "") or len(np.unique(uids)) != len(uids):
            raise PartCRev1DataPolicyError("node_uid_by_idx must contain unique nonblank stable IDs")
    return components, uids


def partC_rev1_classify_od_batch(
    origin_node_idx: Sequence[int] | np.ndarray,
    destination_node_idx: Sequence[int] | np.ndarray,
    demand: Sequence[float] | np.ndarray,
    component_id_by_node: Any,
    *,
    raw_row_offset: int,
) -> PartCRev1ODProjectionBatch:
    """Classify a bounded raw OD batch without changing or aggregating it."""

    origin = np.asarray(origin_node_idx, dtype=np.int64)
    destination = np.asarray(destination_node_idx, dtype=np.int64)
    quantity = np.asarray(demand, dtype=np.float64)
    if origin.ndim != 1 or origin.size == 0 or destination.shape != origin.shape or quantity.shape != origin.shape:
        raise PartCRev1DataPolicyError("OD origin, destination, and demand must be equal nonempty vectors")
    if np.any(~np.isfinite(quantity)) or np.any(quantity <= 0):
        raise PartCRev1DataPolicyError("Raw solver OD demand must be finite and positive")
    components, uid_by_idx = _component_and_uid_arrays(component_id_by_node)
    if np.any(origin < 0) or np.any(destination < 0) or np.any(origin >= len(components)) or np.any(destination >= len(components)):
        raise PartCRev1DataPolicyError("OD node index is outside the frozen graph vector order")
    offset = int(raw_row_offset)
    if offset < 0:
        raise PartCRev1DataPolicyError("raw_row_offset must be nonnegative")
    self_mask = origin == destination
    same_component = (components[origin] >= 0) & (components[origin] == components[destination])
    classification = np.full(origin.shape, UNREACHABLE_CROSS_COMPONENT, dtype="U32")
    classification[same_component & ~self_mask] = NETWORK_LOADABLE
    classification[self_mask] = INTRAZONAL_NON_NETWORK_LOADED
    offsets = np.arange(offset, offset + len(origin), dtype=np.int64)
    return PartCRev1ODProjectionBatch(
        origin_node_idx=origin.copy(),
        destination_node_idx=destination.copy(),
        origin_node_uid=uid_by_idx[origin].copy(),
        destination_node_uid=uid_by_idx[destination].copy(),
        demand=quantity.copy(),
        classification=classification,
        raw_row_count=np.ones(len(origin), dtype=np.int64),
        raw_row_offset_min=offsets,
        raw_row_offset_max=offsets.copy(),
        metadata={"raw_row_offset": offset, "classification_policy": "phase1-approved-v1"},
    )


def partC_rev1_aggregate_loadable_pairs(batch: PartCRev1ODProjectionBatch) -> PartCRev1ODProjectionBatch:
    """Aggregate only network-loadable rows by persistent endpoint IDs."""

    mask = batch.classification == NETWORK_LOADABLE
    if not np.any(mask):
        raise PartCRev1DataPolicyError("Cannot produce a no-op solver projection with zero loadable rows")
    origin_uid = batch.origin_node_uid[mask].astype(str)
    destination_uid = batch.destination_node_uid[mask].astype(str)
    order = np.argsort(destination_uid, kind="mergesort")
    order = order[np.argsort(origin_uid[order], kind="mergesort")]
    origin_uid = origin_uid[order]
    destination_uid = destination_uid[order]
    origin_idx = batch.origin_node_idx[mask][order]
    destination_idx = batch.destination_node_idx[mask][order]
    demand = batch.demand[mask][order]
    raw_count = batch.raw_row_count[mask][order]
    raw_min = batch.raw_row_offset_min[mask][order]
    raw_max = batch.raw_row_offset_max[mask][order]
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero((origin_uid[1:] != origin_uid[:-1]) | (destination_uid[1:] != destination_uid[:-1])).astype(np.int64) + 1,
        )
    )
    aggregated_demand = np.add.reduceat(demand, starts).astype(np.float64)
    aggregated_count = np.add.reduceat(raw_count, starts).astype(np.int64)
    aggregated_min = np.minimum.reduceat(raw_min, starts).astype(np.int64)
    aggregated_max = np.maximum.reduceat(raw_max, starts).astype(np.int64)
    first_origin_idx = origin_idx[starts]
    first_destination_idx = destination_idx[starts]
    for start, stop in zip(starts, np.concatenate((starts[1:], np.array([len(order)], dtype=np.int64)))):
        if np.any(origin_idx[start:stop] != origin_idx[start]) or np.any(destination_idx[start:stop] != destination_idx[start]):
            raise PartCRev1DataPolicyError("Persistent endpoint ID maps to inconsistent graph vector positions")
    return PartCRev1ODProjectionBatch(
        origin_node_idx=first_origin_idx.copy(),
        destination_node_idx=first_destination_idx.copy(),
        origin_node_uid=origin_uid[starts].copy(),
        destination_node_uid=destination_uid[starts].copy(),
        demand=aggregated_demand,
        classification=np.full(len(starts), NETWORK_LOADABLE, dtype="U32"),
        raw_row_count=aggregated_count,
        raw_row_offset_min=aggregated_min,
        raw_row_offset_max=aggregated_max,
        metadata={
            "projection_policy": "stable_uid_mergesort_float64_sum_v1",
            "source_batch_rows": batch.row_count,
            "source_report": batch.report(),
        },
    )


def partC_rev1_projection_conservation(
    report: Mapping[str, Any], *, atol: float = 1e-9, rtol: float = 1e-12
) -> bool:
    """Check raw = loadable + self + unreachable + named exclusions."""

    if atol < 0 or rtol < 0:
        raise PartCRev1DataPolicyError("Conservation tolerances must be nonnegative")
    keys = (
        "raw_positive_demand",
        "network_loadable_demand",
        "intrazonal_non_network_loaded_demand",
        "unreachable_cross_component_demand",
    )
    try:
        values = [float(report[key]) for key in keys]
    except (KeyError, TypeError, ValueError) as exc:
        raise PartCRev1DataPolicyError("Conservation report is missing a required finite total") from exc
    other = float(report.get("other_excluded_demand", 0.0))
    if any(not math.isfinite(value) or value < 0 for value in (*values, other)):
        raise PartCRev1DataPolicyError("Conservation totals must be finite and nonnegative")
    raw, loadable, intrazonal, unreachable = values
    classified = math.fsum((loadable, intrazonal, unreachable, other))
    tolerance = max(float(atol), float(rtol) * max(abs(raw), abs(classified)))
    return bool(abs(raw - classified) <= tolerance)


def partC_rev1_reachable_nonself_gap(tstt: float, reachable_nonself_sptt: float) -> float:
    denominator = float(reachable_nonself_sptt)
    numerator_total = float(tstt)
    if not math.isfinite(denominator) or denominator <= 0:
        raise PartCRev1DataPolicyError("Reachable nonself SPTT denominator must be finite and positive")
    if not math.isfinite(numerator_total) or numerator_total < 0:
        raise PartCRev1DataPolicyError("TSTT must be finite and nonnegative")
    return float(max(0.0, (numerator_total - denominator) / denominator))


def partC_rev1_flag_zero_length_arcs(
    length: Sequence[float] | np.ndarray,
    free_time: Sequence[float] | np.ndarray,
    capacity: Sequence[float] | np.ndarray,
    endpoint_exists: Sequence[bool] | np.ndarray,
) -> np.ndarray:
    length_a = np.asarray(length, dtype=np.float64)
    free_a = np.asarray(free_time, dtype=np.float64)
    capacity_a = np.asarray(capacity, dtype=np.float64)
    endpoint_a = np.asarray(endpoint_exists, dtype=bool)
    if length_a.ndim != 1 or not (length_a.shape == free_a.shape == capacity_a.shape == endpoint_a.shape):
        raise PartCRev1DataPolicyError("Arc policy vectors must be aligned one-dimensional arrays")
    if np.any(~np.isfinite(length_a)) or np.any(length_a < 0):
        raise PartCRev1DataPolicyError("Arc length must be finite and nonnegative")
    zero = length_a == 0.0
    invalid = zero & ((~np.isfinite(free_a)) | (free_a <= 0) | (~np.isfinite(capacity_a)) | (capacity_a <= 0) | (~endpoint_a))
    if np.any(invalid):
        raise PartCRev1DataPolicyError("A zero-length arc failed the positive-time/capacity/endpoint predicate")
    return zero


def partC_rev1_safe_speed(
    length: Sequence[float] | np.ndarray,
    time: Sequence[float] | np.ndarray,
    is_zero_length: Sequence[bool] | np.ndarray,
) -> np.ndarray:
    """Return miles/hour for minute inputs, with retained zero arcs as NaN."""

    length_a = np.asarray(length, dtype=np.float64)
    time_a = np.asarray(time, dtype=np.float64)
    zero_a = np.asarray(is_zero_length, dtype=bool)
    if length_a.ndim != 1 or length_a.shape != time_a.shape or length_a.shape != zero_a.shape:
        raise PartCRev1DataPolicyError("Speed vectors must be aligned one-dimensional arrays")
    if np.any(~np.isfinite(length_a)) or np.any(length_a < 0) or np.any(~np.isfinite(time_a)) or np.any(time_a <= 0):
        raise PartCRev1DataPolicyError("Speed inputs require finite nonnegative length and positive time")
    if not np.array_equal(zero_a, length_a == 0.0):
        raise PartCRev1DataPolicyError("Zero-length arcs require an exact explicit guard mask")
    speed = np.full(length_a.shape, np.nan, dtype=np.float64)
    positive = ~zero_a
    speed[positive] = 60.0 * length_a[positive] / time_a[positive]
    return speed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _database_state(path: Path) -> dict[str, Any]:
    return {
        "main": _file_state(path),
        "sidecars": {
            candidate.name: _file_state(candidate)
            for candidate in sorted(path.parent.glob(path.name + "*"))
            if candidate != path and candidate.is_file()
        },
    }


def _resolve_input_paths(repository_root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    registry = json.loads((repository_root / _REGISTRY_RELATIVE).read_text(encoding="utf-8"))
    manifest = json.loads((repository_root / _MANIFEST_RELATIVE).read_text(encoding="utf-8"))
    roots = {
        "corrected_graph_root": Path(registry["authoritative_inputs"]["corrected_graph"]["root"]),
        "od_root": Path(registry["authoritative_inputs"]["od"]["root"]),
        "original_repository_root": Path(registry["original_repository_root"]),
    }
    paths: dict[str, Path] = {}
    for item in manifest["inputs"]:
        path = (roots[item["root_key"]] / Path(item["relative_path"])).resolve(strict=True)
        if path.stat().st_size != int(item["size_bytes"]):
            raise PartCRev1DataPolicyError(f"Authoritative input size changed for {item['role']}")
        paths[str(item["role"])] = path
    if len(paths) != 9:
        raise PartCRev1DataPolicyError("Exactly nine authoritative roles are required")
    return paths, manifest


def _verify_bindings(config_path: Path, config: Mapping[str, Any], repository_root: Path) -> dict[str, Any]:
    if config.get("campaign_id") != _CAMPAIGN_ID or config.get("run_id") != _RUN_ID:
        raise PartCRev1DataPolicyError("Campaign/run identity mismatch")
    approval = json.loads((repository_root / _APPROVAL_RELATIVE).read_text(encoding="utf-8"))
    if not approval.get("completion_campaign_execution_authorized") or approval.get("phase_2_authorized"):
        raise PartCRev1DataPolicyError("Tracked execution approval is absent or Phase 2 is enabled")
    expected_config = approval["approved_config_hashes"]["execution_config_sha256"]
    if _sha256(config_path) != expected_config:
        raise PartCRev1DataPolicyError("Execution configuration hash mismatch")
    if _sha256(repository_root / _RUN_MATRIX_RELATIVE) != approval["approved_run_matrix_hash"]:
        raise PartCRev1DataPolicyError("Run-matrix hash mismatch")
    if _sha256(repository_root / _RESOURCE_POLICY_RELATIVE) != approval["approved_config_hashes"]["resource_policy_sha256"]:
        raise PartCRev1DataPolicyError("Resource-policy hash mismatch")
    prompt = repository_root / approval["approved_goal_prompt"]["path"]
    if _sha256(prompt) != approval["approved_goal_prompt"]["sha256"]:
        raise PartCRev1DataPolicyError("Approved Goal-prompt hash mismatch")
    command = ["git", "-C", str(repository_root), "-c", "core.longpaths=true"]
    branch = subprocess.run(command + ["branch", "--show-current"], check=True, capture_output=True, text=True).stdout.strip()
    if branch != approval["approved_branch"]:
        raise PartCRev1DataPolicyError("Campaign branch mismatch")
    subprocess.run(
        command + ["merge-base", "--is-ancestor", approval["approved_starting_head"], "HEAD"],
        check=True,
        capture_output=True,
    )
    return approval


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise PartCRev1DataPolicyError(f"Refusing to overwrite campaign evidence: {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_sample(config_path: Path, task_id: str, sample_origin_groups: int) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repository_root = Path(str(config["repository_root"])).resolve(strict=True)
    if repository_root != Path(__file__).resolve().parents[1]:
        raise PartCRev1DataPolicyError("Execution config repository root does not own this module")
    approval = _verify_bindings(config_path, config, repository_root)
    if task_id != "S1-T02" or sample_origin_groups <= 0:
        raise PartCRev1DataPolicyError("The bounded CLI requires task S1-T02 and a positive origin-group count")
    if sample_origin_groups != 2:
        raise PartCRev1DataPolicyError("The frozen run matrix requires exactly two sampled origin groups")
    output_root = Path(str(config["output_root"])).resolve()
    if output_root.as_posix() != approval["approved_output_root"]:
        raise PartCRev1DataPolicyError("Approved output-root mismatch")
    original_root = Path(
        json.loads((repository_root / _REGISTRY_RELATIVE).read_text(encoding="utf-8"))["original_repository_root"]
    ).resolve(strict=True)
    if output_root.is_relative_to(repository_root) or output_root.is_relative_to(original_root):
        raise PartCRev1DataPolicyError("Campaign output may not be inside either checkout")
    paths, manifest = _resolve_input_paths(repository_root)
    if manifest["manifest_fingerprint"]["value"] != approval["approved_input_hashes"]["authoritative_manifest_fingerprint"]:
        raise PartCRev1DataPolicyError("Authoritative manifest fingerprint mismatch")
    predicates = json.loads((repository_root / _PREDICATES_RELATIVE).read_text(encoding="utf-8"))
    if predicates.get("overall_status") != "PASS":
        raise PartCRev1DataPolicyError("Frozen policy predicates are not PASS")
    before = {
        role: (_database_state(path) if role in {"rich_sqlite", "parent_gpkg"} else _file_state(path))
        for role, path in paths.items()
    }

    import pyarrow.parquet as parquet
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from .partC_rev1_phase1_input_smoke import PartCRev1BoundedNPZReader

    nodes = parquet.read_table(paths["graph_nodes_parquet"], columns=["node_idx", "partC_rev1_node_uid"])
    node_idx = nodes["node_idx"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    node_uid_source = nodes["partC_rev1_node_uid"].to_numpy(zero_copy_only=False).astype(str)
    if len(node_idx) == 0 or len(np.unique(node_idx)) != len(node_idx) or node_idx.min() != 0 or node_idx.max() != len(node_idx) - 1:
        raise PartCRev1DataPolicyError("Graph node_idx must be a complete unique dense vector order")
    uid_width = max(len(value) for value in node_uid_source)
    node_uid_by_idx = np.empty(len(node_idx), dtype=f"U{uid_width}")
    node_uid_by_idx[node_idx] = node_uid_source
    if len(np.unique(node_uid_by_idx)) != len(node_uid_by_idx):
        raise PartCRev1DataPolicyError("Graph stable node IDs must be unique")

    arc_columns = [
        "arc_idx",
        "directed_arc_id",
        "partC_rev1_link_uid",
        "tail_idx",
        "head_idx",
        "free_time",
        "assignment_capacity",
        "length_miles",
    ]
    arcs = parquet.read_table(paths["graph_arcs_parquet"], columns=arc_columns)
    arc_idx = arcs["arc_idx"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    if len(arc_idx) == 0 or len(np.unique(arc_idx)) != len(arc_idx) or arc_idx.min() != 0 or arc_idx.max() != len(arc_idx) - 1:
        raise PartCRev1DataPolicyError("Graph arc_idx must be a complete unique dense vector order")
    arc_order = np.argsort(arc_idx, kind="mergesort")
    tail = arcs["tail_idx"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)[arc_order]
    head = arcs["head_idx"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)[arc_order]
    free_time = arcs["free_time"].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)[arc_order]
    capacity = arcs["assignment_capacity"].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)[arc_order]
    length = arcs["length_miles"].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)[arc_order]
    directed_arc_id = arcs["directed_arc_id"].to_numpy(zero_copy_only=False).astype(str)[arc_order]
    link_uid = arcs["partC_rev1_link_uid"].to_numpy(zero_copy_only=False).astype(str)[arc_order]
    endpoint_exists = (tail >= 0) & (head >= 0) & (tail < len(node_idx)) & (head < len(node_idx))
    if not np.all(endpoint_exists):
        raise PartCRev1DataPolicyError("Graph contains an arc endpoint outside the node vector")
    zero = partC_rev1_flag_zero_length_arcs(length, free_time, capacity, endpoint_exists)
    if int(np.count_nonzero(zero)) != 4:
        raise PartCRev1DataPolicyError("The frozen graph must retain exactly four zero-length arcs")
    expected_zero = {
        str(item["directed_arc_id"])
        for predicate in predicates["predicates"]
        if predicate["predicate_id"] == "four_zero_length_arcs"
        for item in predicate["arcs"]
    }
    if set(directed_arc_id[zero].tolist()) != expected_zero:
        raise PartCRev1DataPolicyError("Zero-length stable arc identities changed")
    partC_rev1_safe_speed(length, free_time, zero)

    adjacency = coo_matrix(
        (
            np.ones(len(tail) * 2, dtype=np.uint8),
            (np.concatenate((tail, head)), np.concatenate((head, tail))),
        ),
        shape=(len(node_idx), len(node_idx)),
    ).tocsr()
    component_count, component_id = connected_components(adjacency, directed=False, return_labels=True)

    with PartCRev1BoundedNPZReader(paths["od_npz"], max_prefix_count=128) as reader:
        origin_groups = reader.read_full_member("origin_group_node_idx.npy")
        origin_offsets = reader.read_full_member("origin_offsets.npy")
    if sample_origin_groups > len(origin_groups):
        raise PartCRev1DataPolicyError("Requested origin-group sample exceeds the authoritative OD group count")
    sample_rows = int(origin_offsets[sample_origin_groups])
    if sample_rows <= 0 or sample_rows > 1_000_000:
        raise PartCRev1DataPolicyError("Bounded OD sample must contain 1..1,000,000 rows")
    with PartCRev1BoundedNPZReader(paths["od_npz"], max_prefix_count=sample_rows) as reader:
        origins = reader.read_prefix("origin_node_idx.npy", sample_rows)
        destinations = reader.read_prefix("destination_node_idx.npy", sample_rows)
        demand = reader.read_prefix("demand.npy", sample_rows)
        od_bytes_requested = int(reader.bytes_requested)
    if not np.array_equal(origins[: int(origin_offsets[1])], np.full(int(origin_offsets[1]), origin_groups[0])):
        raise PartCRev1DataPolicyError("First OD origin group is not contiguous in frozen vector order")
    predicate = partC_rev1_inspect_single_class_assignment(
        SimpleNamespace(free_time=free_time, capacity=capacity, metadata={"commodity_dependent": False}),
        SimpleNamespace(origin_node_idx=origins, destination_node_idx=destinations, demand=demand, metadata={}),
    )
    if not predicate.passed:
        raise PartCRev1DataPolicyError("BLOCKED_COMMODITY_DEPENDENT_SOLVER")
    batch = partC_rev1_classify_od_batch(
        origins,
        destinations,
        demand,
        {"component_id_by_node": component_id, "node_uid_by_idx": node_uid_by_idx},
        raw_row_offset=0,
    )
    report = batch.report()
    if not partC_rev1_projection_conservation(report, atol=1e-9, rtol=1e-12):
        raise PartCRev1DataPolicyError("Bounded OD policy sample failed demand conservation")
    projected = partC_rev1_aggregate_loadable_pairs(batch)
    if projected.row_count <= 0 or int(np.sum(projected.raw_row_count)) != report["categories"][NETWORK_LOADABLE]["raw_rows"]:
        raise PartCRev1DataPolicyError("Bounded loadable-pair projection is empty or lost provenance")

    after = {
        role: (_database_state(path) if role in {"rich_sqlite", "parent_gpkg"} else _file_state(path))
        for role, path in paths.items()
    }
    if after != before:
        raise PartCRev1DataPolicyError("An authoritative input or database sidecar changed during the bounded sample")
    zero_rows = [
        {
            "arc_idx": int(index),
            "directed_arc_id": str(directed_arc_id[index]),
            "partC_rev1_link_uid": str(link_uid[index]),
            "tail_node_uid": str(node_uid_by_idx[tail[index]]),
            "head_node_uid": str(node_uid_by_idx[head[index]]),
            "free_time_minutes": float(free_time[index]),
            "assignment_capacity_annual_ktons": float(capacity[index]),
            "length_miles": float(length[index]),
            "length_status": "ZERO_LENGTH_RETAINED",
        }
        for index in np.flatnonzero(zero)
    ]
    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "phase_id": "01",
        "step_ids": ["1.2", "1.3"],
        "campaign_id": _CAMPAIGN_ID,
        "run_id": _RUN_ID,
        "task_id": task_id,
        "approval_commit": subprocess.run(
            ["git", "-C", str(repository_root), "-c", "core.longpaths=true", "log", "--format=%H", "--grep=^phase1: record completion campaign execution approval$", "-1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "single_class_predicate": predicate.to_dict(),
        "sample_origin_group_count": sample_origin_groups,
        "sample_origin_node_idx": [int(value) for value in origin_groups[:sample_origin_groups]],
        "sample_origin_offsets": [int(value) for value in origin_offsets[: sample_origin_groups + 1]],
        "sample_raw_rows": batch.row_count,
        "sample_aggregated_loadable_pairs": projected.row_count,
        "od_npz_data_bytes_requested": od_bytes_requested,
        "component_count": int(component_count),
        "conservation": {
            **report,
            "residual": float(
                report["raw_positive_demand"]
                - report["network_loadable_demand"]
                - report["intrazonal_non_network_loaded_demand"]
                - report["unreachable_cross_component_demand"]
                - report["other_excluded_demand"]
            ),
            "atol": 1e-9,
            "rtol": 1e-12,
            "passed": True,
        },
        "projection_provenance": {
            "raw_rows": int(np.sum(projected.raw_row_count, dtype=np.int64)),
            "aggregated_rows": projected.row_count,
            "first_raw_offset": int(np.min(projected.raw_row_offset_min)),
            "last_raw_offset": int(np.max(projected.raw_row_offset_max)),
            "full_projection_materialized": False,
        },
        "zero_length_arcs": zero_rows,
        "zero_length_arc_count": len(zero_rows),
        "stable_id_join_policy": "named node_idx->partC_rev1_node_uid and arc_idx->directed_arc_id/link_uid arrays only",
        "authoritative_role_count": len(paths),
        "authoritative_invariance": True,
        "database_sidecars_unchanged": True,
        "phase_2_artifacts_created": 0,
        "scientific_execution_performed": False,
    }
    output = output_root / "stage_01/policy_sample.json"
    _atomic_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the approved bounded Phase 1 OD policy sample")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--sample-origin-groups", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _run_sample(Path(args.config).resolve(strict=True), str(args.task_id), int(args.sample_origin_groups))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
