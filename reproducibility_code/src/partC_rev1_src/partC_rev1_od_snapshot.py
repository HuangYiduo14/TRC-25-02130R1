"""Compact OD snapshots for Prompt 4 assignment solvers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .partC_rev1_graph_snapshot import partC_rev1_open_sqlite_readonly
from .partC_rev1_hashing import partC_rev1_canonical_scalar, partC_rev1_numpy_array_hash
from .partC_rev1_io import partC_rev1_write_json
from .partC_rev1_paths import partC_rev1_ensure_directory, partC_rev1_resolve_path
from .partC_rev1_progress_logging import PartCRev1ProgressReporter


@dataclass(slots=True)
class PartCRev1ODSnapshot:
    od_hash: str
    origin_node_idx: np.ndarray
    destination_node_idx: np.ndarray
    demand: np.ndarray
    origin_group_node_idx: np.ndarray
    origin_offsets: np.ndarray
    metadata: dict[str, Any]

    def save(self, output_dir: str | Path) -> dict[str, str]:
        out_dir = partC_rev1_ensure_directory(partC_rev1_resolve_path(output_dir))
        npz_path = out_dir / f"partC_rev1_od_snapshot_{self.od_hash}.npz"
        metadata_path = out_dir / f"partC_rev1_od_snapshot_metadata_{self.od_hash}.json"
        np.savez_compressed(
            npz_path,
            origin_node_idx=self.origin_node_idx,
            destination_node_idx=self.destination_node_idx,
            demand=self.demand,
            origin_group_node_idx=self.origin_group_node_idx,
            origin_offsets=self.origin_offsets,
        )
        partC_rev1_write_json(
            {
                **self.metadata,
                "od_hash": self.od_hash,
                "od_rows": int(len(self.demand)),
                "origin_count": int(len(self.origin_group_node_idx)),
                "total_demand": float(self.demand.sum()) if len(self.demand) else 0.0,
                "npz_path": str(npz_path),
            },
            metadata_path,
        )
        return {"npz": str(npz_path), "metadata": str(metadata_path)}


def partC_rev1_load_od_snapshot(npz_path: str | Path, metadata_path: str | Path | None = None) -> PartCRev1ODSnapshot:
    """Load an immutable OD snapshot written by ``PartCRev1ODSnapshot.save``."""
    resolved_npz = partC_rev1_resolve_path(npz_path)
    if metadata_path is None:
        name = resolved_npz.name
        if not name.startswith("partC_rev1_od_snapshot_") or not name.endswith(".npz"):
            raise ValueError(f"Cannot infer OD metadata path from snapshot name: {resolved_npz}")
        od_hash = name.removeprefix("partC_rev1_od_snapshot_").removesuffix(".npz")
        resolved_metadata = resolved_npz.with_name(f"partC_rev1_od_snapshot_metadata_{od_hash}.json")
    else:
        resolved_metadata = partC_rev1_resolve_path(metadata_path)
    import json

    metadata = json.loads(resolved_metadata.read_text(encoding="utf-8")) if resolved_metadata.exists() else {}
    with np.load(resolved_npz, allow_pickle=False) as data:
        od_hash = str(metadata.get("od_hash") or resolved_npz.stem.removeprefix("partC_rev1_od_snapshot_"))
        return PartCRev1ODSnapshot(
            od_hash=od_hash,
            origin_node_idx=data["origin_node_idx"].astype(np.int64),
            destination_node_idx=data["destination_node_idx"].astype(np.int64),
            demand=data["demand"].astype(np.float64),
            origin_group_node_idx=data["origin_group_node_idx"].astype(np.int64),
            origin_offsets=data["origin_offsets"].astype(np.int64),
            metadata=dict(metadata),
        )


def partC_rev1_build_od_snapshot_from_frames(
    od: pd.DataFrame,
    *,
    node_uid_to_idx: Mapping[str, int],
    demand_col: str = "demand_solver_unit",
    metadata: Mapping[str, Any] | None = None,
) -> PartCRev1ODSnapshot:
    required = {"origin_node_uid", "destination_node_uid", demand_col}
    missing = sorted(required.difference(od.columns))
    if missing:
        raise ValueError(f"OD table missing columns: {missing}")
    work = od.copy()
    work["_origin_uid"] = work["origin_node_uid"].map(partC_rev1_canonical_scalar)
    work["_destination_uid"] = work["destination_node_uid"].map(partC_rev1_canonical_scalar)
    work["_demand"] = pd.to_numeric(work[demand_col], errors="coerce").fillna(0.0)
    positive = work["_demand"].gt(0)
    work = work.loc[positive].copy()
    work["_origin_idx"] = work["_origin_uid"].map(node_uid_to_idx)
    work["_destination_idx"] = work["_destination_uid"].map(node_uid_to_idx)
    unmatched = work["_origin_idx"].isna() | work["_destination_idx"].isna()
    unmatched_rows = int(unmatched.sum())
    unmatched_demand = float(work.loc[unmatched, "_demand"].sum()) if unmatched_rows else 0.0
    work = work.loc[~unmatched].copy()
    if work.empty:
        origin = np.array([], dtype=np.int64)
        dest = np.array([], dtype=np.int64)
        demand = np.array([], dtype=np.float64)
    else:
        work["_origin_idx"] = work["_origin_idx"].astype("int64")
        work["_destination_idx"] = work["_destination_idx"].astype("int64")
        order = np.lexsort((work["_destination_idx"].to_numpy(), work["_origin_idx"].to_numpy()))
        origin = work["_origin_idx"].to_numpy(dtype=np.int64)[order]
        dest = work["_destination_idx"].to_numpy(dtype=np.int64)[order]
        demand = work["_demand"].to_numpy(dtype=np.float64)[order]
    group_nodes, starts = np.unique(origin, return_index=True)
    origin_offsets = np.concatenate((starts.astype(np.int64), np.array([len(origin)], dtype=np.int64)))
    od_hash = partC_rev1_numpy_array_hash(origin, dest, demand)
    return PartCRev1ODSnapshot(
        od_hash=od_hash,
        origin_node_idx=origin,
        destination_node_idx=dest,
        demand=demand,
        origin_group_node_idx=group_nodes.astype(np.int64),
        origin_offsets=origin_offsets.astype(np.int64),
        metadata={
            "source": "dataframe",
            "input_rows": int(len(od)),
            "positive_rows": int(positive.sum()),
            "matched_rows": int(len(origin)),
            "unmatched_rows": unmatched_rows,
            "unmatched_demand": unmatched_demand,
            "demand_col": demand_col,
            **dict(metadata or {}),
        },
    )


def partC_rev1_build_od_snapshot_from_sqlite(
    sqlite_path: str | Path,
    *,
    node_uid_to_idx: Mapping[str, int],
    od_table: str = "partC_rev1_od_centroid",
    demand_col: str = "demand_solver_unit",
    chunksize: int = 1_000_000,
    output_dir: str | Path = "partC_rev1_cache",
    reporter: PartCRev1ProgressReporter | None = None,
) -> tuple[PartCRev1ODSnapshot, dict[str, str]]:
    origins: list[np.ndarray] = []
    destinations: list[np.ndarray] = []
    demands: list[np.ndarray] = []
    input_rows = 0
    positive_rows = 0
    unmatched_rows = 0
    unmatched_demand = 0.0
    started = time.perf_counter()
    query = f'SELECT origin_node_uid, destination_node_uid, "{demand_col}" AS demand FROM "{od_table}" WHERE "{demand_col}" > 0'
    with partC_rev1_open_sqlite_readonly(sqlite_path) as conn:
        for chunk_idx, chunk in enumerate(pd.read_sql_query(query, conn, chunksize=chunksize), start=1):
            input_rows += len(chunk)
            chunk["_origin_uid"] = chunk["origin_node_uid"].map(partC_rev1_canonical_scalar)
            chunk["_destination_uid"] = chunk["destination_node_uid"].map(partC_rev1_canonical_scalar)
            chunk["_demand"] = pd.to_numeric(chunk["demand"], errors="coerce").fillna(0.0)
            positive = chunk["_demand"].gt(0)
            positive_rows += int(positive.sum())
            chunk = chunk.loc[positive].copy()
            chunk["_origin_idx"] = chunk["_origin_uid"].map(node_uid_to_idx)
            chunk["_destination_idx"] = chunk["_destination_uid"].map(node_uid_to_idx)
            unmatched = chunk["_origin_idx"].isna() | chunk["_destination_idx"].isna()
            unmatched_rows += int(unmatched.sum())
            if unmatched.any():
                unmatched_demand += float(chunk.loc[unmatched, "_demand"].sum())
            matched = chunk.loc[~unmatched]
            if not matched.empty:
                origins.append(matched["_origin_idx"].to_numpy(dtype=np.int64))
                destinations.append(matched["_destination_idx"].to_numpy(dtype=np.int64))
                demands.append(matched["_demand"].to_numpy(dtype=np.float64))
            if reporter is not None:
                reporter.heartbeat(
                    "od_snapshot_scan",
                    chunk=chunk_idx,
                    input_rows=input_rows,
                    matched_rows=sum(len(item) for item in origins),
                    unmatched_rows=unmatched_rows,
                    elapsed_seconds=time.perf_counter() - started,
                )
    origin = np.concatenate(origins) if origins else np.array([], dtype=np.int64)
    dest = np.concatenate(destinations) if destinations else np.array([], dtype=np.int64)
    demand = np.concatenate(demands) if demands else np.array([], dtype=np.float64)
    if len(origin):
        order = np.lexsort((dest, origin))
        origin = origin[order]
        dest = dest[order]
        demand = demand[order]
    group_nodes, starts = np.unique(origin, return_index=True)
    origin_offsets = np.concatenate((starts.astype(np.int64), np.array([len(origin)], dtype=np.int64)))
    od_hash = partC_rev1_numpy_array_hash(origin, dest, demand)
    snapshot = PartCRev1ODSnapshot(
        od_hash=od_hash,
        origin_node_idx=origin,
        destination_node_idx=dest,
        demand=demand,
        origin_group_node_idx=group_nodes.astype(np.int64),
        origin_offsets=origin_offsets.astype(np.int64),
        metadata={
            "source_sqlite": str(partC_rev1_resolve_path(sqlite_path)),
            "od_table": od_table,
            "demand_col": demand_col,
            "input_rows": int(input_rows),
            "positive_rows": int(positive_rows),
            "matched_rows": int(len(origin)),
            "unmatched_rows": int(unmatched_rows),
            "unmatched_demand": float(unmatched_demand),
        },
    )
    paths = snapshot.save(output_dir)
    if reporter is not None:
        reporter.emit(
            "od_snapshot_saved",
            od_hash=snapshot.od_hash,
            rows=len(snapshot.demand),
            origins=len(snapshot.origin_group_node_idx),
            total_demand=float(snapshot.demand.sum()) if len(snapshot.demand) else 0.0,
            paths=paths,
        )
    return snapshot, paths
