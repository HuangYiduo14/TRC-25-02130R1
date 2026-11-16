"""Compact graph snapshots for Prompt 4 physical assignment solvers.

Snapshots preserve map-derived ``base_capacity`` and store assignment/BPR
``capacity`` separately so capacity multiplier experiments never rewrite the
authoritative network maps.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .partC_rev1_hashing import (
    partC_rev1_canonical_scalar,
    partC_rev1_dataframe_records_hash,
    partC_rev1_json_hash,
    partC_rev1_sort_key,
)
from .partC_rev1_io import partC_rev1_write_json, partC_rev1_write_table
from .partC_rev1_paths import partC_rev1_ensure_directory, partC_rev1_resolve_path
from .partC_rev1_progress_logging import PartCRev1ProgressReporter


@dataclass(slots=True)
class PartCRev1PartitionCheck:
    config_resolution: float | None
    config_seed: int | None
    artifact_resolution: float | None
    artifact_seed: int | None
    artifact_graph_hash: str | None
    matches: bool
    action: str


@dataclass(slots=True)
class PartCRev1GraphSnapshot:
    graph_hash: str
    partition_hash: str
    node_idx_to_uid: np.ndarray
    arc_idx_to_directed_arc_id: np.ndarray
    arc_idx_to_link_uid: np.ndarray
    tail: np.ndarray
    head: np.ndarray
    free_time: np.ndarray
    base_capacity: np.ndarray
    capacity: np.ndarray
    capacity_multiplier: np.ndarray
    capacity_multiplier_type: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    length: np.ndarray
    region_code: np.ndarray
    region_idx_to_label: list[str]
    csr_indptr: np.ndarray
    csr_head: np.ndarray
    csr_arc_idx: np.ndarray
    reverse_csr_indptr: np.ndarray
    reverse_csr_head: np.ndarray
    reverse_csr_arc_idx: np.ndarray
    metadata: dict[str, Any]

    @property
    def node_uid_to_idx(self) -> dict[str, int]:
        return {str(uid): idx for idx, uid in enumerate(self.node_idx_to_uid.tolist())}

    @property
    def topology_hash(self) -> str:
        return str(self.metadata.get("topology_hash") or self.graph_hash)

    @property
    def backbone_hash(self) -> str:
        return str(self.metadata.get("backbone_hash") or self.partition_hash or self.graph_hash)

    @property
    def assignment_graph_hash(self) -> str:
        return str(self.metadata.get("assignment_graph_hash") or self.graph_hash)

    def save(self, output_dir: str | Path) -> dict[str, str]:
        out_dir = partC_rev1_ensure_directory(partC_rev1_resolve_path(output_dir))
        base = f"partC_rev1_graph_snapshot_{self.graph_hash}"
        npz_path = out_dir / f"{base}.npz"
        node_path = out_dir / f"partC_rev1_graph_snapshot_nodes_{self.graph_hash}.parquet"
        arc_path = out_dir / f"partC_rev1_graph_snapshot_arcs_{self.graph_hash}.parquet"
        metadata_path = out_dir / f"partC_rev1_graph_snapshot_metadata_{self.graph_hash}.json"

        np.savez_compressed(
            npz_path,
            node_idx_to_uid=self.node_idx_to_uid,
            arc_idx_to_directed_arc_id=self.arc_idx_to_directed_arc_id,
            arc_idx_to_link_uid=self.arc_idx_to_link_uid,
            tail=self.tail,
            head=self.head,
            free_time=self.free_time,
            base_capacity=self.base_capacity,
            capacity=self.capacity,
            capacity_multiplier=self.capacity_multiplier,
            capacity_multiplier_type=self.capacity_multiplier_type.astype(str),
            alpha=self.alpha,
            beta=self.beta,
            length=self.length,
            region_code=self.region_code,
            csr_indptr=self.csr_indptr,
            csr_head=self.csr_head,
            csr_arc_idx=self.csr_arc_idx,
            reverse_csr_indptr=self.reverse_csr_indptr,
            reverse_csr_head=self.reverse_csr_head,
            reverse_csr_arc_idx=self.reverse_csr_arc_idx,
        )
        nodes = pd.DataFrame({"node_idx": np.arange(len(self.node_idx_to_uid), dtype=np.int64), "partC_rev1_node_uid": self.node_idx_to_uid})
        arcs = pd.DataFrame(
            {
                "arc_idx": np.arange(len(self.arc_idx_to_directed_arc_id), dtype=np.int64),
                "directed_arc_id": self.arc_idx_to_directed_arc_id,
                "partC_rev1_link_uid": self.arc_idx_to_link_uid,
                "tail_idx": self.tail,
                "head_idx": self.head,
                "tail_node_uid": self.node_idx_to_uid[self.tail],
                "head_node_uid": self.node_idx_to_uid[self.head],
                "free_time": self.free_time,
                "base_capacity": self.base_capacity,
                "assignment_capacity": self.capacity,
                "remaining_capacity": self.capacity,
                "capacity_multiplier": self.capacity_multiplier,
                "capacity_multiplier_type": self.capacity_multiplier_type.astype(str),
                "alpha": self.alpha,
                "beta": self.beta,
                "length_miles": self.length,
                "region_code": self.region_code,
                "arc_label": [self.region_idx_to_label[int(code)] for code in self.region_code],
            }
        )
        partC_rev1_write_table(nodes, node_path, index=False)
        partC_rev1_write_table(arcs, arc_path, index=False)
        partC_rev1_write_json(
            {
                **self.metadata,
                "graph_hash": self.graph_hash,
                "topology_hash": self.topology_hash,
                "backbone_hash": self.backbone_hash,
                "partition_hash": self.partition_hash,
                "assignment_graph_hash": self.assignment_graph_hash,
                "node_count": int(len(self.node_idx_to_uid)),
                "arc_count": int(len(self.arc_idx_to_directed_arc_id)),
                "region_labels": self.region_idx_to_label,
                "npz_path": str(npz_path),
                "nodes_path": str(node_path),
                "arcs_path": str(arc_path),
            },
            metadata_path,
        )
        return {
            "npz": str(npz_path),
            "nodes": str(node_path),
            "arcs": str(arc_path),
            "metadata": str(metadata_path),
        }


def partC_rev1_load_graph_snapshot(npz_path: str | Path, metadata_path: str | Path | None = None) -> PartCRev1GraphSnapshot:
    """Load an immutable graph snapshot written by ``PartCRev1GraphSnapshot.save``."""
    resolved_npz = partC_rev1_resolve_path(npz_path)
    if metadata_path is None:
        name = resolved_npz.name
        if not name.startswith("partC_rev1_graph_snapshot_") or not name.endswith(".npz"):
            raise ValueError(f"Cannot infer graph metadata path from snapshot name: {resolved_npz}")
        graph_hash = name.removeprefix("partC_rev1_graph_snapshot_").removesuffix(".npz")
        resolved_metadata = resolved_npz.with_name(f"partC_rev1_graph_snapshot_metadata_{graph_hash}.json")
    else:
        resolved_metadata = partC_rev1_resolve_path(metadata_path)
    import json

    metadata = json.loads(resolved_metadata.read_text(encoding="utf-8")) if resolved_metadata.exists() else {}
    with np.load(resolved_npz, allow_pickle=False) as data:
        graph_hash = str(metadata.get("graph_hash") or resolved_npz.stem.removeprefix("partC_rev1_graph_snapshot_"))
        region_labels = [str(value) for value in metadata.get("region_labels", [])]
        if not region_labels:
            max_code = int(np.max(data["region_code"])) if len(data["region_code"]) else -1
            region_labels = [str(idx) for idx in range(max_code + 1)]
        capacity = data["capacity"].astype(np.float64)
        base_capacity = data["base_capacity"].astype(np.float64) if "base_capacity" in data.files else capacity.copy()
        capacity_multiplier = data["capacity_multiplier"].astype(np.float64) if "capacity_multiplier" in data.files else np.ones_like(capacity)
        capacity_multiplier_type = data["capacity_multiplier_type"].astype(str) if "capacity_multiplier_type" in data.files else np.full(len(capacity), "legacy_unscaled", dtype=str)
        return PartCRev1GraphSnapshot(
            graph_hash=graph_hash,
            partition_hash=str(metadata.get("partition_hash", "")),
            node_idx_to_uid=data["node_idx_to_uid"].astype(str),
            arc_idx_to_directed_arc_id=data["arc_idx_to_directed_arc_id"].astype(str),
            arc_idx_to_link_uid=data["arc_idx_to_link_uid"].astype(str),
            tail=data["tail"].astype(np.int64),
            head=data["head"].astype(np.int64),
            free_time=data["free_time"].astype(np.float64),
            base_capacity=base_capacity,
            capacity=capacity,
            capacity_multiplier=capacity_multiplier,
            capacity_multiplier_type=capacity_multiplier_type,
            alpha=data["alpha"].astype(np.float64),
            beta=data["beta"].astype(np.float64),
            length=data["length"].astype(np.float64),
            region_code=data["region_code"].astype(np.int32),
            region_idx_to_label=region_labels,
            csr_indptr=data["csr_indptr"].astype(np.int64),
            csr_head=data["csr_head"].astype(np.int64),
            csr_arc_idx=data["csr_arc_idx"].astype(np.int64),
            reverse_csr_indptr=data["reverse_csr_indptr"].astype(np.int64),
            reverse_csr_head=data["reverse_csr_head"].astype(np.int64),
            reverse_csr_arc_idx=data["reverse_csr_arc_idx"].astype(np.int64),
            metadata=dict(metadata),
        )


def partC_rev1_open_sqlite_readonly(sqlite_path: str | Path) -> sqlite3.Connection:
    """Open an existing SQLite input read-only and enforce query-only mode."""

    resolved = partC_rev1_resolve_path(sqlite_path)
    connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        query_only = connection.execute("PRAGMA query_only").fetchone()
        if query_only is None or int(query_only[0]) != 1:
            raise sqlite3.DatabaseError("SQLite read-only connection did not enter query_only mode")
        return connection
    except BaseException:
        connection.close()
        raise


def partC_rev1_check_partition_consistency(sqlite_path: str | Path, config: Mapping[str, Any]) -> PartCRev1PartitionCheck:
    bp = config.get("backbone_partition", {}) if isinstance(config.get("backbone_partition", {}), Mapping) else {}
    partition_cfg = bp.get("leiden", bp.get("partition", {}))
    partition = partition_cfg if isinstance(partition_cfg, Mapping) else {}
    config_resolution = float(partition.get("resolution")) if partition.get("resolution") is not None else None
    config_seed = int(partition.get("random_seed")) if partition.get("random_seed") is not None else None
    with partC_rev1_open_sqlite_readonly(sqlite_path) as conn:
        row = conn.execute(
            'SELECT resolution, random_seed, graph_hash FROM "partC_rev1_partition_metadata" ORDER BY rowid DESC LIMIT 1'
        ).fetchone()
    artifact_resolution = float(row[0]) if row and row[0] is not None else None
    artifact_seed = int(row[1]) if row and row[1] is not None else None
    artifact_graph_hash = str(row[2]) if row and row[2] is not None else None
    matches = config_resolution == artifact_resolution and config_seed == artifact_seed
    return PartCRev1PartitionCheck(
        config_resolution=config_resolution,
        config_seed=config_seed,
        artifact_resolution=artifact_resolution,
        artifact_seed=artifact_seed,
        artifact_graph_hash=artifact_graph_hash,
        matches=matches,
        action="use_current_region_labels" if matches else "regenerate_prompt3_partition_required",
    )


def partC_rev1_collect_positive_od_endpoints(
    sqlite_path: str | Path,
    *,
    od_table: str = "partC_rev1_od_centroid",
    demand_col: str = "demand_solver_unit",
    chunksize: int = 1_000_000,
    reporter: PartCRev1ProgressReporter | None = None,
) -> set[str]:
    endpoints: set[str] = set()
    started = time.perf_counter()
    query = f'SELECT origin_node_uid, destination_node_uid, "{demand_col}" AS demand FROM "{od_table}" WHERE "{demand_col}" > 0'
    with partC_rev1_open_sqlite_readonly(sqlite_path) as conn:
        for chunk_idx, chunk in enumerate(pd.read_sql_query(query, conn, chunksize=chunksize), start=1):
            endpoints.update(partC_rev1_canonical_scalar(value) for value in chunk["origin_node_uid"].dropna().unique())
            endpoints.update(partC_rev1_canonical_scalar(value) for value in chunk["destination_node_uid"].dropna().unique())
            if reporter is not None:
                reporter.heartbeat(
                    "od_endpoint_scan",
                    chunk=chunk_idx,
                    rows=int(len(chunk)),
                    endpoints=len(endpoints),
                    elapsed_seconds=time.perf_counter() - started,
                )
    return endpoints


def _partC_rev1_pick_column(df: pd.DataFrame, candidates: Iterable[str], *, required: bool = True) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    if required:
        raise ValueError(f"Missing required column; tried {list(candidates)}")
    return None


def _partC_rev1_assignment_capacity_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    source = config.get("assignment_capacity_multipliers", config) if isinstance(config, Mapping) else {}
    section = source if isinstance(source, Mapping) else {}
    class_codes = section.get("centroid_connector_class_codes", [50])
    return {
        "enabled": bool(section.get("enabled", True)),
        "centroid_connector_multiplier": float(section.get("centroid_connector_multiplier", 100.0)),
        "overall_capacity_multiplier": float(section.get("overall_capacity_multiplier", 0.3)),
        "centroid_connector_class_codes": [int(float(value)) for value in class_codes],
    }


def _partC_rev1_capacity_multiplier_label(multiplier_config: Mapping[str, Any]) -> str:
    if not bool(multiplier_config.get("enabled", True)):
        return "assignment_capacity_unscaled"
    centroid = str(float(multiplier_config["centroid_connector_multiplier"])).replace(".", "p")
    overall = str(float(multiplier_config["overall_capacity_multiplier"])).replace(".", "p")
    return f"assignment_capacity_centroid-{centroid}_overall-{overall}"


def _partC_rev1_apply_assignment_capacity_multipliers(
    work: pd.DataFrame,
    *,
    class_col: str | None,
    config: Mapping[str, Any] | None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    multiplier_config = _partC_rev1_assignment_capacity_config(config)
    enabled = bool(multiplier_config["enabled"])
    base_capacity = pd.to_numeric(work["_base_capacity"], errors="coerce")
    if not enabled:
        multiplier = pd.Series(1.0, index=work.index, dtype="float64")
        multiplier_type = pd.Series("unscaled_disabled", index=work.index, dtype="string")
    else:
        if class_col is not None:
            class_code = pd.to_numeric(work[class_col], errors="coerce")
        else:
            class_code = pd.Series(np.nan, index=work.index, dtype="float64")
        connector_codes = set(int(value) for value in multiplier_config["centroid_connector_class_codes"])
        connector_mask = class_code.dropna().astype(int).isin(connector_codes).reindex(work.index, fill_value=False)
        multiplier = pd.Series(float(multiplier_config["overall_capacity_multiplier"]), index=work.index, dtype="float64")
        multiplier.loc[connector_mask] = float(multiplier_config["centroid_connector_multiplier"])
        multiplier_type = pd.Series("overall", index=work.index, dtype="string")
        multiplier_type.loc[connector_mask] = "centroid_connector"

    assignment_capacity = base_capacity * multiplier
    metadata = {
        "assignment_capacity_multipliers": {
            **multiplier_config,
            "label": _partC_rev1_capacity_multiplier_label(multiplier_config),
            "class_column": class_col,
            "connector_arc_rows": int(multiplier_type.eq("centroid_connector").sum()),
            "overall_arc_rows": int(multiplier_type.eq("overall").sum()),
        }
    }
    return base_capacity, assignment_capacity, multiplier, multiplier_type, metadata


def partC_rev1_build_graph_snapshot_from_frames(
    arcs: pd.DataFrame,
    *,
    od_endpoint_uids: Iterable[Any] = (),
    assignment_capacity_config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PartCRev1GraphSnapshot:
    arc_id_col = _partC_rev1_pick_column(arcs, ["directed_arc_id"])
    link_col = _partC_rev1_pick_column(arcs, ["partC_rev1_link_uid"])
    tail_col = _partC_rev1_pick_column(arcs, ["from_node_uid"])
    head_col = _partC_rev1_pick_column(arcs, ["to_node_uid"])
    fftt_col = _partC_rev1_pick_column(arcs, ["partC_rev1_fftt_min", "free_time_min"])
    capacity_col = _partC_rev1_pick_column(arcs, ["partC_rev1_capacity_annual_ktons_dir", "remaining_capacity", "capacity"])
    class_col = _partC_rev1_pick_column(arcs, ["class_code", "F_Class", "Class"], required=False)
    alpha_col = _partC_rev1_pick_column(arcs, ["alpha"])
    beta_col = _partC_rev1_pick_column(arcs, ["beta"])
    length_col = _partC_rev1_pick_column(arcs, ["length_miles", "length"], required=False)
    label_col = _partC_rev1_pick_column(arcs, ["partC_rev1_prompt3_region_label", "arc_label", "region_id"], required=False)

    work = arcs.copy()
    work["_tail_uid"] = work[tail_col].map(partC_rev1_canonical_scalar)
    work["_head_uid"] = work[head_col].map(partC_rev1_canonical_scalar)
    work["_arc_id"] = work[arc_id_col].map(partC_rev1_canonical_scalar)
    work["_link_uid"] = work[link_col].map(partC_rev1_canonical_scalar)
    work["_free_time"] = pd.to_numeric(work[fftt_col], errors="coerce")
    work["_base_capacity"] = pd.to_numeric(work[capacity_col], errors="coerce")
    base_capacity_series, assignment_capacity_series, multiplier_series, multiplier_type_series, multiplier_metadata = _partC_rev1_apply_assignment_capacity_multipliers(
        work,
        class_col=class_col,
        config=assignment_capacity_config,
    )
    # Source maps keep the enriched annual-kton capacity; assignment solvers use
    # this scaled copy only, so preprocessing artifacts remain read-only.
    work["_base_capacity"] = base_capacity_series
    work["_capacity"] = assignment_capacity_series
    work["_capacity_multiplier"] = multiplier_series
    work["_capacity_multiplier_type"] = multiplier_type_series
    work["_alpha"] = pd.to_numeric(work[alpha_col], errors="coerce").fillna(0.15)
    work["_beta"] = pd.to_numeric(work[beta_col], errors="coerce").fillna(4.0)
    work["_length"] = pd.to_numeric(work[length_col], errors="coerce").fillna(0.0) if length_col is not None else 0.0
    if label_col is not None:
        work["_arc_label"] = work[label_col].map(partC_rev1_canonical_scalar)
    else:
        work["_arc_label"] = ""
    if "partC_rev1_prompt3_is_final_backbone" in work.columns:
        backbone_mask = pd.to_numeric(work["partC_rev1_prompt3_is_final_backbone"], errors="coerce").fillna(0).astype(int).eq(1)
        work.loc[backbone_mask, "_arc_label"] = "backbone"
    if "is_final_backbone" in work.columns:
        backbone_mask = pd.to_numeric(work["is_final_backbone"], errors="coerce").fillna(0).astype(int).eq(1)
        work.loc[backbone_mask, "_arc_label"] = "backbone"
    valid = (
        work["_tail_uid"].ne("")
        & work["_head_uid"].ne("")
        & work["_arc_id"].ne("")
        & np.isfinite(work["_free_time"])
        & np.isfinite(work["_capacity"])
        & work["_free_time"].gt(0)
        & work["_capacity"].gt(0)
    )
    if "is_open" in work.columns:
        valid &= pd.to_numeric(work["is_open"], errors="coerce").fillna(1).astype(int).eq(1)
    if "is_valid_for_truck" in work.columns:
        valid &= pd.to_numeric(work["is_valid_for_truck"], errors="coerce").fillna(1).astype(int).eq(1)
    usable = work.loc[valid].copy()
    if usable.empty:
        raise ValueError("No usable directed arcs were available for graph snapshot")

    labels = usable["_arc_label"].replace("", pd.NA)
    if labels.isna().any():
        raise ValueError("Every usable directed arc must have a backbone/region label")
    node_uids = set(usable["_tail_uid"].tolist()) | set(usable["_head_uid"].tolist())
    node_uids.update(partC_rev1_canonical_scalar(value) for value in od_endpoint_uids)
    node_uids.discard("")
    node_idx_to_uid = np.array(sorted(node_uids, key=partC_rev1_sort_key), dtype=str)
    node_uid_to_idx = {uid: idx for idx, uid in enumerate(node_idx_to_uid.tolist())}
    usable["_tail_idx"] = usable["_tail_uid"].map(node_uid_to_idx).astype("int64")
    usable["_head_idx"] = usable["_head_uid"].map(node_uid_to_idx).astype("int64")
    usable["_arc_sort_key"] = usable["_arc_id"].map(partC_rev1_sort_key)
    usable = usable.sort_values(["_tail_idx", "_arc_sort_key"], kind="mergesort").reset_index(drop=True)
    labels_sorted = sorted(set(usable["_arc_label"].tolist()), key=lambda value: (value != "backbone", partC_rev1_sort_key(value)))
    label_to_code = {label: idx for idx, label in enumerate(labels_sorted)}

    tail = usable["_tail_idx"].to_numpy(dtype=np.int64)
    head = usable["_head_idx"].to_numpy(dtype=np.int64)
    arc_idx_to_id = usable["_arc_id"].to_numpy(dtype=str)
    arc_idx_to_link = usable["_link_uid"].to_numpy(dtype=str)
    free_time = usable["_free_time"].to_numpy(dtype=np.float64)
    base_capacity = usable["_base_capacity"].to_numpy(dtype=np.float64)
    capacity = usable["_capacity"].to_numpy(dtype=np.float64)
    capacity_multiplier = usable["_capacity_multiplier"].to_numpy(dtype=np.float64)
    capacity_multiplier_type = usable["_capacity_multiplier_type"].astype(str).to_numpy(dtype=str)
    alpha = usable["_alpha"].to_numpy(dtype=np.float64)
    beta = usable["_beta"].to_numpy(dtype=np.float64)
    length = usable["_length"].to_numpy(dtype=np.float64)
    region_code = usable["_arc_label"].map(label_to_code).to_numpy(dtype=np.int32)

    counts = np.bincount(tail, minlength=len(node_idx_to_uid))
    csr_indptr = np.concatenate(([0], np.cumsum(counts, dtype=np.int64))).astype(np.int64)
    csr_arc_idx = np.arange(len(usable), dtype=np.int64)
    csr_head = head.copy()
    reverse_order = np.lexsort((arc_idx_to_id.astype(str), tail, head))
    reverse_counts = np.bincount(head, minlength=len(node_idx_to_uid))
    reverse_csr_indptr = np.concatenate(([0], np.cumsum(reverse_counts, dtype=np.int64))).astype(np.int64)
    reverse_csr_arc_idx = reverse_order.astype(np.int64)
    reverse_csr_head = tail[reverse_csr_arc_idx].astype(np.int64)

    topology_hash = partC_rev1_dataframe_records_hash(
        usable,
        ["_arc_id", "_link_uid", "_tail_uid", "_head_uid"],
    )
    backbone_hash = partC_rev1_dataframe_records_hash(
        usable.assign(_is_backbone=usable["_arc_label"].astype(str).eq("backbone").astype(int)),
        ["_arc_id", "_is_backbone"],
    )
    partition_hash = partC_rev1_dataframe_records_hash(usable, ["_arc_id", "_arc_label"])
    assignment_graph_hash = partC_rev1_dataframe_records_hash(
        usable,
        ["_arc_id", "_link_uid", "_tail_uid", "_head_uid", "_free_time", "_base_capacity", "_capacity", "_capacity_multiplier_type", "_alpha", "_beta"],
    )
    snapshot_metadata = {
        "source": "dataframe",
        "topology_hash": topology_hash,
        "backbone_hash": backbone_hash,
        "assignment_graph_hash": assignment_graph_hash,
        "graph_hash_alias": "assignment_graph_hash",
        "input_arc_rows": int(len(arcs)),
        "usable_arc_rows": int(len(usable)),
        "excluded_arc_rows": int(len(arcs) - len(usable)),
        "node_count": int(len(node_idx_to_uid)),
        "arc_count": int(len(usable)),
        "od_endpoint_count": int(len(set(partC_rev1_canonical_scalar(value) for value in od_endpoint_uids if partC_rev1_canonical_scalar(value)))),
        **multiplier_metadata,
        **dict(metadata or {}),
    }
    return PartCRev1GraphSnapshot(
        graph_hash=assignment_graph_hash,
        partition_hash=partition_hash,
        node_idx_to_uid=node_idx_to_uid,
        arc_idx_to_directed_arc_id=arc_idx_to_id,
        arc_idx_to_link_uid=arc_idx_to_link,
        tail=tail,
        head=head,
        free_time=free_time,
        base_capacity=base_capacity,
        capacity=capacity,
        capacity_multiplier=capacity_multiplier,
        capacity_multiplier_type=capacity_multiplier_type,
        alpha=alpha,
        beta=beta,
        length=length,
        region_code=region_code,
        region_idx_to_label=labels_sorted,
        csr_indptr=csr_indptr,
        csr_head=csr_head,
        csr_arc_idx=csr_arc_idx,
        reverse_csr_indptr=reverse_csr_indptr,
        reverse_csr_head=reverse_csr_head,
        reverse_csr_arc_idx=reverse_csr_arc_idx,
        metadata=snapshot_metadata,
    )


def partC_rev1_build_graph_snapshot_from_sqlite(
    sqlite_path: str | Path,
    *,
    od_endpoint_uids: Iterable[Any] = (),
    arc_table: str = "partC_rev1_directed_arcs_enriched",
    output_dir: str | Path = "partC_rev1_cache",
    assignment_capacity_config: Mapping[str, Any] | None = None,
    reporter: PartCRev1ProgressReporter | None = None,
) -> tuple[PartCRev1GraphSnapshot, dict[str, str]]:
    started = time.perf_counter()
    desired_columns = [
        "directed_arc_id",
        "partC_rev1_link_uid",
        "from_node_uid",
        "to_node_uid",
        "partC_rev1_fftt_min",
        "free_time_min",
        "partC_rev1_capacity_annual_ktons_dir",
        "capacity",
        "class_code",
        "F_Class",
        "Class",
        "alpha",
        "beta",
        "length_miles",
        "partC_rev1_prompt3_region_label",
        "partC_rev1_prompt3_arc_partition_type",
        "partC_rev1_prompt3_is_final_backbone",
        "is_final_backbone",
        "is_open",
        "is_valid_for_truck",
    ]
    with partC_rev1_open_sqlite_readonly(sqlite_path) as conn:
        available = {row[1] for row in conn.execute(f'PRAGMA table_info("{arc_table}")')}
        selected = [column for column in desired_columns if column in available]
        required = {"directed_arc_id", "partC_rev1_link_uid", "from_node_uid", "to_node_uid", "alpha", "beta"}
        missing = sorted(required.difference(selected))
        if missing:
            raise ValueError(f"Arc table {arc_table} is missing required graph snapshot columns: {missing}")
        quoted = ", ".join(f'"{column}"' for column in selected)
        arcs = pd.read_sql_query(f'SELECT {quoted} FROM "{arc_table}"', conn)
    if reporter is not None:
        reporter.emit("graph_arcs_loaded", rows=len(arcs), elapsed_seconds=time.perf_counter() - started)
    snapshot = partC_rev1_build_graph_snapshot_from_frames(
        arcs,
        od_endpoint_uids=od_endpoint_uids,
        assignment_capacity_config=assignment_capacity_config,
        metadata={"source_sqlite": str(partC_rev1_resolve_path(sqlite_path)), "arc_table": arc_table},
    )
    paths = snapshot.save(output_dir)
    if reporter is not None:
        reporter.emit(
            "graph_snapshot_saved",
            graph_hash=snapshot.graph_hash,
            partition_hash=snapshot.partition_hash,
            node_count=len(snapshot.node_idx_to_uid),
            arc_count=len(snapshot.arc_idx_to_directed_arc_id),
            paths=paths,
        )
    return snapshot, paths
