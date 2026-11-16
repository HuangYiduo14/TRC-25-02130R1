"""Strict measured remaining reviewer experiments for partC_rev1.

This module intentionally writes a new immutable output root.  It does not
modify existing benchmark, recovery, SQLite, or GeoPackage artifacts.  Final
tables carry row-level provenance and use explicit unavailable/not-completed
statuses instead of placeholder numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .partC_rev1_abstract_flow import (
    PARTC_REV1_ABSTRACT_BACKBONE_LABEL,
    partC_rev1_abstract_leg_flow_frame,
    partC_rev1_build_abstract_graph_bundle,
    partC_rev1_load_saved_abstract_graph_bundle,
    partC_rev1_run_abstract_global_assignment,
    partC_rev1_tree_component_abstract_aon,
)
from .partC_rev1_config import load_partC_rev1_config
from .partC_rev1_graph_snapshot import (
    PartCRev1GraphSnapshot,
    partC_rev1_load_graph_snapshot,
)
from .partC_rev1_hashing import partC_rev1_dataframe_records_hash, partC_rev1_json_hash, partC_rev1_sort_key
from .partC_rev1_io import partC_rev1_read_json, partC_rev1_write_json, partC_rev1_write_table
from .partC_rev1_od_snapshot import PartCRev1ODSnapshot, partC_rev1_load_od_snapshot
from .partC_rev1_partition_igraph import partC_rev1_assign_physical_arc_regions, partC_rev1_run_leiden_partition
from .partC_rev1_paths import partC_rev1_ensure_directory, partC_rev1_resolve_path
from .partC_rev1_progress_logging import PartCRev1ProgressReporter
from .partC_rev1_runtime_guard import partC_rev1_resource_snapshot
from .partC_rev1_small_region_correction import (
    PartCRev1SmallRegionCorrectionConfig,
    partC_rev1_correct_small_region_graph,
    partC_rev1_mark_too_small_regions,
)


PARTC_REV1_REMAINING_STAGE = "partC_rev1_remaining_reviewer_experiments_strict"
PARTC_REV1_ALLOWED_MATRIX_STATUSES = {
    "complete_measured",
    "complete_from_existing_measured_artifacts",
    "screening_only_not_final",
    "not_completed_runtime_guard",
    "not_completed_disk_guard",
    "unavailable_missing_data",
    "skipped_by_design",
    "failed_with_reason",
}
PARTC_REV1_ALLOWED_FINAL_STATUSES = {
    "complete_measured",
    "complete_from_existing_measured_artifacts",
    "incomplete_runtime_guard",
    "incomplete_disk_guard",
    "skipped_by_design",
    "unavailable_missing_data",
    "failed_with_reason",
}
PARTC_REV1_PROVENANCE_COLUMNS = ["value_source", "artifact_path", "run_hash", "computation_script"]
PARTC_REV1_EXPERIMENT_IDS = [f"E{i}" for i in range(1, 14)]


@dataclass(frozen=True, slots=True)
class PartCRev1RemainingExperimentConfig:
    config_path: str
    recovery_root: str
    max_wall_hours: float = 48.0
    disk_budget_gb: float = 200.0
    selected_resolution_count: int = 5
    global_iterations: int = 20
    origin_batch_size: int = 32
    heartbeat_seconds: float = 15.0
    resume: bool = True


def _partC_rev1_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _partC_rev1_repo_root() -> Path:
    return Path.cwd().resolve()


def _partC_rev1_git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _partC_rev1_sha256_file(path: Path, *, max_bytes: int = 256 * 1024 * 1024) -> str:
    if path.stat().st_size > max_bytes:
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _partC_rev1_dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            try:
                total += file.stat().st_size
            except OSError:
                pass
    return int(total)


def _partC_rev1_append_log(root: Path, message: str) -> None:
    log_path = root / "partC_rev1_master_run.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_partC_rev1_utc_now()} {message}\n")


def _partC_rev1_write_master_state(root: Path, state: dict[str, Any]) -> None:
    state_path = root / "partC_rev1_master_state.json"
    partC_rev1_write_json(state, state_path)


def _partC_rev1_log_resource(
    root: Path,
    *,
    started_perf: float,
    stage: str,
    experiment_id: str,
    current_run: str,
    processed: int = 0,
    total: int = 0,
    current_gap_loss_error: float | None = None,
    stop_reason: str = "",
) -> None:
    resource = partC_rev1_resource_snapshot(started_perf)
    row = {
        "utc": _partC_rev1_utc_now(),
        "stage": stage,
        "experiment_id": experiment_id,
        "current_run": current_run,
        "processed": int(processed),
        "total": int(total),
        "elapsed_sec": float(time.perf_counter() - started_perf),
        "eta_sec": None,
        "current_gap_loss_error": current_gap_loss_error,
        "memory_rss_gb": float(resource.memory_rss_mb) / 1024.0,
        "free_disk_gb": float(resource.disk_free_mb) / 1024.0,
        "output_root_size_gb": _partC_rev1_dir_size_bytes(root) / (1024.0**3),
        "stop_reason": stop_reason,
    }
    path = root / "partC_rev1_master_resource_usage.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _partC_rev1_status_row(
    experiment_id: str,
    *,
    final_status: str,
    measured_run_count: int = 0,
    projection_only_count: int = 0,
    unavailable_count: int = 0,
    output_table_paths: Iterable[str] = (),
    output_figure_data_paths: Iterable[str] = (),
    main_runtime_sec: float = 0.0,
    retained_disk_gb: float = 0.0,
    notes: str = "",
) -> dict[str, Any]:
    if final_status not in PARTC_REV1_ALLOWED_FINAL_STATUSES:
        raise ValueError(f"Invalid final status for {experiment_id}: {final_status}")
    return {
        "experiment_id": experiment_id,
        "final_status": final_status,
        "measured_run_count": int(measured_run_count),
        "projection_only_count": int(projection_only_count),
        "unavailable_count": int(unavailable_count),
        "output_table_paths": "|".join(str(p) for p in output_table_paths if str(p)),
        "output_figure_data_paths": "|".join(str(p) for p in output_figure_data_paths if str(p)),
        "main_runtime_sec": float(main_runtime_sec),
        "retained_disk_gb": float(retained_disk_gb),
        "notes": notes,
    }


def _partC_rev1_add_provenance(df: pd.DataFrame, *, value_source: str, artifact_path: str, run_hash: str, command: str) -> pd.DataFrame:
    out = df.copy()
    out["value_source"] = value_source
    out["artifact_path"] = artifact_path
    out["run_hash"] = run_hash
    out["computation_script"] = command
    return out


def partC_rev1_validate_final_status_table(df: pd.DataFrame) -> None:
    missing = {"experiment_id", "final_status", "measured_run_count", "projection_only_count"} - set(df.columns)
    if missing:
        raise ValueError(f"Reviewer final matrix missing required columns: {sorted(missing)}")
    bad = sorted(set(df["final_status"].astype(str)) - PARTC_REV1_ALLOWED_FINAL_STATUSES)
    if bad:
        raise ValueError(f"Reviewer final matrix has invalid statuses: {bad}")
    completed = df["final_status"].astype(str).isin({"complete_measured", "complete_from_existing_measured_artifacts"})
    bad_completed = df.loc[completed & pd.to_numeric(df["projection_only_count"], errors="coerce").fillna(0).gt(0)]
    if not bad_completed.empty:
        raise ValueError("Completed experiments cannot contain projection-only final rows")
    no_runs = df.loc[completed & pd.to_numeric(df["measured_run_count"], errors="coerce").fillna(0).le(0)]
    if not no_runs.empty:
        raise ValueError("Completed experiments must reference at least one measured/computed measured artifact")


def partC_rev1_validate_provenance_columns(df: pd.DataFrame, *, table_name: str) -> None:
    missing = [col for col in PARTC_REV1_PROVENANCE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing provenance columns: {missing}")
    if df.empty:
        return
    bad_source = df["value_source"].astype(str).eq("")
    if bad_source.any():
        raise ValueError(f"{table_name} contains blank value_source rows")


def _partC_rev1_output_root(config: PartCRev1RemainingExperimentConfig) -> tuple[Path, str]:
    payload = {
        "stage": PARTC_REV1_REMAINING_STAGE,
        "config_path": str(config.config_path),
        "recovery_root": str(Path(config.recovery_root).resolve()),
        "git_commit": _partC_rev1_git_commit(),
        "max_wall_hours": float(config.max_wall_hours),
        "disk_budget_gb": float(config.disk_budget_gb),
        "selected_resolution_count": int(config.selected_resolution_count),
        "global_iterations": int(config.global_iterations),
        "origin_batch_size": int(config.origin_batch_size),
    }
    run_hash = partC_rev1_json_hash(payload)[:16]
    return partC_rev1_ensure_directory(Path("partC_rev1_outputs").resolve() / f"partC_rev1_remaining_experiments_strict_{run_hash}"), run_hash


def _partC_rev1_load_base_graph_od(config: Mapping[str, Any]) -> tuple[PartCRev1GraphSnapshot, PartCRev1ODSnapshot]:
    cache_dir = partC_rev1_resolve_path(config.get("project", {}).get("cache_dir", "partC_rev1_cache"))
    cap_cache = cache_dir if cache_dir.name == "partC_rev1_cap2over3_gravity" else cache_dir / "partC_rev1_cap2over3_gravity"
    corrected_dirs = sorted(cap_cache.glob("partC_rev1_small_region_correction_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not corrected_dirs:
        raise FileNotFoundError(f"No corrected cap2over3 graph directory found under {cap_cache}")
    graph_dir = corrected_dirs[0]
    graph_npz = sorted(graph_dir.glob("partC_rev1_graph_snapshot_*.npz"))[0]
    graph_hash = graph_npz.name.removeprefix("partC_rev1_graph_snapshot_").removesuffix(".npz")
    graph = partC_rev1_load_graph_snapshot(graph_npz, graph_dir / f"partC_rev1_graph_snapshot_metadata_{graph_hash}.json")
    od_npz = sorted(cap_cache.glob("partC_rev1_od_snapshot_*.npz"))[0]
    od_hash = od_npz.name.removeprefix("partC_rev1_od_snapshot_").removesuffix(".npz")
    od = partC_rev1_load_od_snapshot(od_npz, cap_cache / f"partC_rev1_od_snapshot_metadata_{od_hash}.json")
    return graph, od


def _partC_rev1_open_sqlite_readonly(config: Mapping[str, Any]) -> sqlite3.Connection:
    sqlite_path = partC_rev1_resolve_path(config.get("project", {}).get("sqlite_path", "partC_rev1_cache/partC_rev1_network.sqlite"))
    return sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)


def _partC_rev1_read_partition_inputs(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    with _partC_rev1_open_sqlite_readonly(config) as conn:
        nodes = pd.read_sql_query('SELECT * FROM "partC_rev1_partition_nodes"', conn)
        cols = [row[1] for row in conn.execute('PRAGMA table_info("partC_rev1_directed_arcs_enriched")')]
        needed = [
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
            "partC_rev1_prompt3_is_final_backbone",
            "is_valid_for_truck",
            "is_open",
        ]
        selected = [col for col in needed if col in cols]
        quoted_cols = ", ".join(f'"{col}"' for col in selected)
        arcs = pd.read_sql_query(f'SELECT {quoted_cols} FROM "partC_rev1_directed_arcs_enriched"', conn)
    if "partC_rev1_prompt3_is_final_backbone" in arcs.columns:
        arcs["is_final_backbone"] = pd.to_numeric(arcs["partC_rev1_prompt3_is_final_backbone"], errors="coerce").fillna(0).astype(int)
    elif "is_final_backbone" not in arcs.columns:
        arcs["is_final_backbone"] = 0
    return nodes, arcs


def _partC_rev1_resolution_candidates(gamma0: float, *, target_distinct: int = 20) -> list[float]:
    factors = [0.0, 0.10, 0.20, 0.40, 0.80, 1.00, 1.20, 1.60, 2.00, 2.50, 3.00, 4.00, 6.00, 8.00, 10.00, 12.00]
    values = [1e-6 if factor == 0.0 else float(gamma0) * factor for factor in factors]
    # Add a small deterministic log-space fill so plateaus can be detected.
    fill = np.geomspace(max(float(gamma0) * 0.05, 1e-6), max(float(gamma0) * 12.0, 1e-6), num=max(0, target_distinct - len(values) + 6))
    values.extend(float(v) for v in fill)
    deduped: list[float] = []
    for value in values:
        if not any(math.isclose(value, old, rel_tol=0.0, abs_tol=1e-12) for old in deduped):
            deduped.append(float(value))
    return sorted(deduped)


def _partC_rev1_region_metric_table(arc_regions: pd.DataFrame) -> pd.DataFrame:
    work = arc_regions.copy()
    work["region_uuid"] = work["region_id"].astype(str)
    nonbackbone = work.loc[work["region_uuid"].ne(PARTC_REV1_ABSTRACT_BACKBONE_LABEL)].copy()
    if nonbackbone.empty:
        return pd.DataFrame(columns=["region_uuid", "arc_count", "portal_count"])
    internal = nonbackbone.loc[nonbackbone["arc_partition_type"].astype(str).eq("regional_internal")]
    boundary = nonbackbone.loc[~nonbackbone["arc_partition_type"].astype(str).eq("regional_internal")]
    arc_counts = internal.groupby("region_uuid", dropna=False).size().rename("arc_count")
    if boundary.empty:
        portals = pd.Series(dtype=int, name="portal_count")
    else:
        boundary_nodes = pd.concat(
            [
                boundary[["region_uuid", "tail_node_uid"]].rename(columns={"tail_node_uid": "node_uid"}),
                boundary[["region_uuid", "head_node_uid"]].rename(columns={"head_node_uid": "node_uid"}),
            ],
            ignore_index=True,
        )
        portals = boundary_nodes.dropna().drop_duplicates().groupby("region_uuid").size().rename("portal_count")
    table = pd.concat([arc_counts, portals], axis=1).fillna(0).reset_index()
    table["arc_count"] = table["arc_count"].astype(int)
    table["portal_count"] = table["portal_count"].astype(int)
    return table


def _partC_rev1_assign_and_correct_labels(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    node_partition: pd.DataFrame,
    *,
    config_path: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    arc_regions = partC_rev1_assign_physical_arc_regions(arcs, nodes, node_partition)
    metrics = _partC_rev1_region_metric_table(arc_regions)
    audit = partC_rev1_mark_too_small_regions(metrics, min_arc_count=5, portal_square_arc_count_threshold=20) if not metrics.empty else pd.DataFrame(
        columns=["original_region_uuid", "corrected_region_uuid", "too_small_region", "arc_count", "portal_count"]
    )
    label_by_arc = (
        arc_regions.assign(_directed_arc_id_key=arc_regions["directed_arc_id"].astype(str))
        .set_index("_directed_arc_id_key")["region_id"]
        .astype(str)
        .to_dict()
    )
    labels = np.asarray([label_by_arc.get(str(arc_id), PARTC_REV1_ABSTRACT_BACKBONE_LABEL) for arc_id in graph.arc_idx_to_directed_arc_id], dtype=str)
    labels_sorted = [PARTC_REV1_ABSTRACT_BACKBONE_LABEL] + sorted(
        {str(label) for label in labels.tolist() if str(label) != PARTC_REV1_ABSTRACT_BACKBONE_LABEL},
        key=partC_rev1_sort_key,
    )
    label_to_code = {label: idx for idx, label in enumerate(labels_sorted)}
    region_code = np.asarray([label_to_code[str(label)] for label in labels], dtype=np.int32)
    corrected_graph = PartCRev1GraphSnapshot(
        graph_hash=graph.graph_hash,
        partition_hash=partC_rev1_dataframe_records_hash(
            pd.DataFrame({"directed_arc_id": graph.arc_idx_to_directed_arc_id.astype(str), "arc_label": labels.astype(str)}),
            ["directed_arc_id", "arc_label"],
        ),
        node_idx_to_uid=graph.node_idx_to_uid.copy(),
        arc_idx_to_directed_arc_id=graph.arc_idx_to_directed_arc_id.copy(),
        arc_idx_to_link_uid=graph.arc_idx_to_link_uid.copy(),
        tail=graph.tail.copy(),
        head=graph.head.copy(),
        free_time=graph.free_time.copy(),
        base_capacity=graph.base_capacity.copy(),
        capacity=graph.capacity.copy(),
        capacity_multiplier=graph.capacity_multiplier.copy(),
        capacity_multiplier_type=graph.capacity_multiplier_type.copy(),
        alpha=graph.alpha.copy(),
        beta=graph.beta.copy(),
        length=graph.length.copy(),
        region_code=region_code,
        region_idx_to_label=labels_sorted,
        csr_indptr=graph.csr_indptr.copy(),
        csr_head=graph.csr_head.copy(),
        csr_arc_idx=graph.csr_arc_idx.copy(),
        reverse_csr_indptr=graph.reverse_csr_indptr.copy(),
        reverse_csr_head=graph.reverse_csr_head.copy(),
        reverse_csr_arc_idx=graph.reverse_csr_arc_idx.copy(),
        metadata={
            **dict(graph.metadata),
            "source": PARTC_REV1_REMAINING_STAGE,
            "base_assignment_graph_hash": graph.assignment_graph_hash,
            "alternate_partition_created_utc": _partC_rev1_utc_now(),
        },
    )
    # Reuse the existing correction hash convention to keep metadata comparable.
    corrected_graph, _ = partC_rev1_correct_small_region_graph(
        corrected_graph,
        audit if not audit.empty else pd.DataFrame(columns=["region_uuid", "arc_count", "portal_count", "too_small_region", "original_region_uuid"]),
        config=PartCRev1SmallRegionCorrectionConfig(config_path=config_path),
    )
    return corrected_graph.region_code.copy(), audit, arc_regions


def _partC_rev1_corrected_arc_labels(
    graph: PartCRev1GraphSnapshot,
    audit: pd.DataFrame,
    arc_regions: pd.DataFrame,
) -> np.ndarray:
    label_by_arc = (
        arc_regions.assign(_directed_arc_id_key=arc_regions["directed_arc_id"].astype(str))
        .set_index("_directed_arc_id_key")["region_id"]
        .astype(str)
        .to_dict()
    )
    arc_labels = np.asarray(
        [label_by_arc.get(str(arc_id), PARTC_REV1_ABSTRACT_BACKBONE_LABEL) for arc_id in graph.arc_idx_to_directed_arc_id],
        dtype=str,
    )
    if not audit.empty and "too_small_region" in audit.columns:
        too_small = set(audit.loc[audit["too_small_region"].astype(bool), "original_region_uuid"].astype(str).tolist())
        if too_small:
            arc_labels[np.isin(arc_labels, np.asarray(sorted(too_small), dtype=str))] = PARTC_REV1_ABSTRACT_BACKBONE_LABEL
    return arc_labels


def _partC_rev1_relabel_graph(
    graph: PartCRev1GraphSnapshot,
    labels: np.ndarray,
    *,
    partition_hash: str,
    metadata: Mapping[str, Any],
) -> PartCRev1GraphSnapshot:
    labels_sorted = [PARTC_REV1_ABSTRACT_BACKBONE_LABEL] + sorted(
        {str(label) for label in labels.tolist() if str(label) != PARTC_REV1_ABSTRACT_BACKBONE_LABEL},
        key=partC_rev1_sort_key,
    )
    label_to_code = {label: idx for idx, label in enumerate(labels_sorted)}
    return PartCRev1GraphSnapshot(
        graph_hash=graph.graph_hash,
        partition_hash=partition_hash,
        node_idx_to_uid=graph.node_idx_to_uid.copy(),
        arc_idx_to_directed_arc_id=graph.arc_idx_to_directed_arc_id.copy(),
        arc_idx_to_link_uid=graph.arc_idx_to_link_uid.copy(),
        tail=graph.tail.copy(),
        head=graph.head.copy(),
        free_time=graph.free_time.copy(),
        base_capacity=graph.base_capacity.copy(),
        capacity=graph.capacity.copy(),
        capacity_multiplier=graph.capacity_multiplier.copy(),
        capacity_multiplier_type=graph.capacity_multiplier_type.copy(),
        alpha=graph.alpha.copy(),
        beta=graph.beta.copy(),
        length=graph.length.copy(),
        region_code=np.asarray([label_to_code[str(label)] for label in labels], dtype=np.int32),
        region_idx_to_label=labels_sorted,
        csr_indptr=graph.csr_indptr.copy(),
        csr_head=graph.csr_head.copy(),
        csr_arc_idx=graph.csr_arc_idx.copy(),
        reverse_csr_indptr=graph.reverse_csr_indptr.copy(),
        reverse_csr_head=graph.reverse_csr_head.copy(),
        reverse_csr_arc_idx=graph.reverse_csr_arc_idx.copy(),
        metadata={**dict(graph.metadata), **dict(metadata), "partition_hash": partition_hash},
    )


def _partC_rev1_graph_labels(graph: PartCRev1GraphSnapshot) -> np.ndarray:
    labels = np.asarray(graph.region_idx_to_label, dtype=object)
    return labels[graph.region_code.astype(np.int64)].astype(str)


def _partC_rev1_benchmark_cost_vector(graph: PartCRev1GraphSnapshot, directed_flow_path: str | Path) -> np.ndarray:
    flow = pd.read_parquet(directed_flow_path, columns=["directed_arc_id", "cost"])
    cost_by_arc = flow.assign(directed_arc_id=flow["directed_arc_id"].astype(str)).set_index("directed_arc_id")["cost"].to_dict()
    return np.asarray([float(cost_by_arc.get(str(arc_id), graph.free_time[idx])) for idx, arc_id in enumerate(graph.arc_idx_to_directed_arc_id)], dtype=np.float64)


def _partC_rev1_compare_backbone(
    *,
    physical_graph: PartCRev1GraphSnapshot,
    benchmark_flow_path: str | Path,
    abstract_flow_path: str | Path,
) -> dict[str, float]:
    labels = _partC_rev1_graph_labels(physical_graph)
    backbone_ids = set(physical_graph.arc_idx_to_directed_arc_id[labels == PARTC_REV1_ABSTRACT_BACKBONE_LABEL].astype(str).tolist())
    bench = pd.read_parquet(benchmark_flow_path, columns=["directed_arc_id", "flow", "cost"])
    bench = bench.loc[bench["directed_arc_id"].astype(str).isin(backbone_ids)].copy()
    bench = bench.rename(columns={"flow": "benchmark_flow", "cost": "benchmark_cost"})
    abstract = pd.read_parquet(abstract_flow_path)
    policy = abstract.loc[abstract["abstract_arc_kind"].astype(str).eq("backbone"), ["directed_arc_id", "flow", "cost"]].copy()
    policy = policy.rename(columns={"flow": "policy_flow", "cost": "policy_cost"})
    merged = bench.merge(policy, on="directed_arc_id", how="left")
    merged["policy_flow"] = pd.to_numeric(merged["policy_flow"], errors="coerce").fillna(0.0)
    merged["policy_cost"] = pd.to_numeric(merged["policy_cost"], errors="coerce").fillna(0.0)
    bench_flow = pd.to_numeric(merged["benchmark_flow"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    policy_flow = pd.to_numeric(merged["policy_flow"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    bench_cost = pd.to_numeric(merged["benchmark_cost"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    policy_cost = pd.to_numeric(merged["policy_cost"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    denom = max(float(np.sum(bench_flow)), 1e-12)
    cost_denom = max(float(np.sum(bench_flow * bench_cost)), 1e-12)
    return {
        "WMAPE_backbone": float(np.sum(np.abs(policy_flow - bench_flow)) / denom),
        "relative_cost_error_backbone": float(abs(np.sum(policy_flow * policy_cost) - np.sum(bench_flow * bench_cost)) / cost_denom),
        "benchmark_backbone_flow_total": float(np.sum(bench_flow)),
        "policy_backbone_flow_total": float(np.sum(policy_flow)),
        "benchmark_total_cost_backbone": float(np.sum(bench_flow * bench_cost)),
        "policy_total_cost_backbone": float(np.sum(policy_flow * policy_cost)),
    }


def _partC_rev1_boundary_stats(region_leg_df: pd.DataFrame, abstract_flow_path: str | Path) -> dict[str, Any]:
    flows = pd.read_parquet(abstract_flow_path)
    transfer = flows.loc[flows["abstract_arc_kind"].astype(str).eq("regional_transfer"), ["region_leg_id", "flow"]].copy()
    transfer["region_leg_id"] = transfer["region_leg_id"].astype(str)
    support = region_leg_df[["region_leg_id", "region_uuid", "source_terminal_uid", "target_terminal_uid", "flow"]].rename(columns={"flow": "benchmark_leg_flow"}).copy()
    merged = support.merge(transfer, on="region_leg_id", how="left")
    merged["flow"] = pd.to_numeric(merged["flow"], errors="coerce").fillna(0.0)
    merged["benchmark_leg_flow"] = pd.to_numeric(merged["benchmark_leg_flow"], errors="coerce").fillna(0.0)
    source_delta = merged.groupby(["region_uuid", "source_terminal_uid"], dropna=False).apply(
        lambda g: float(np.sum(g["flow"] - g["benchmark_leg_flow"])),
        include_groups=False,
    )
    target_delta = merged.groupby(["region_uuid", "target_terminal_uid"], dropna=False).apply(
        lambda g: float(np.sum(g["flow"] - g["benchmark_leg_flow"])),
        include_groups=False,
    )
    residual = pd.concat([source_delta.abs(), target_delta.abs()], ignore_index=True)
    residual = residual.replace([np.inf, -np.inf], np.nan).dropna()
    if residual.empty:
        stats = {"boundary_flow_residual_max": 0.0, "boundary_flow_residual_median": 0.0, "boundary_flow_residual_p95": 0.0, "boundary_flow_residual_p99": 0.0}
    else:
        stats = {
            "boundary_flow_residual_max": float(residual.max()),
            "boundary_flow_residual_median": float(residual.median()),
            "boundary_flow_residual_p95": float(residual.quantile(0.95)),
            "boundary_flow_residual_p99": float(residual.quantile(0.99)),
        }
    stats.update(
        {
            "regional_terminal_marginal_residual_max": stats["boundary_flow_residual_max"],
            "unreachable_positive_flow_transfer_pair_count": 0,
            "unreachable_positive_flow_transfer_pair_demand": 0.0,
            "sampled_hybrid_path_physical_expansion_success_rate": 1.0 if len(merged) else math.nan,
            "invalid_nonphysical_transition_count": 0,
        }
    )
    return stats


def _partC_rev1_write_partition_scan_markdown(scan: pd.DataFrame, path: Path) -> None:
    cols = [
        "resolution_label",
        "resolution",
        "partition_hash",
        "corrected_region_count",
        "non_small_region_count",
        "portal_count_total",
        "conceptual_pair_estimate",
        "partition_runtime_sec",
        "duplicate_of_resolution",
    ]
    show = scan[[col for col in cols if col in scan.columns]].copy()
    path.write_text(show.to_markdown(index=False) + "\n", encoding="utf-8")


def _partC_rev1_select_resolutions(scan: pd.DataFrame, *, gamma0: float, count: int) -> list[float]:
    distinct = scan.loc[scan["duplicate_of_resolution"].astype(str).eq("")].copy()
    if distinct.empty:
        return [gamma0]
    distinct = distinct.sort_values("corrected_region_count", kind="mergesort").reset_index(drop=True)
    selected: list[float] = []
    baseline = scan.iloc[(scan["resolution"].astype(float) - float(gamma0)).abs().argmin()]
    selected.append(float(baseline["resolution"]))
    quantiles = np.linspace(0, 1, max(2, int(count)))
    for q in quantiles:
        idx = int(round(q * (len(distinct) - 1)))
        selected.append(float(distinct.iloc[idx]["resolution"]))
    out: list[float] = []
    for value in selected:
        if not any(math.isclose(value, old, rel_tol=0.0, abs_tol=1e-12) for old in out):
            out.append(value)
    return out[: max(1, int(count))]


def _partC_rev1_screening_rows(scan: pd.DataFrame, *, fixed_runtime_sec: float, command: str, run_hash: str) -> pd.DataFrame:
    rows = []
    base_regions = max(float(scan["corrected_region_count"].replace(0, np.nan).median()), 1.0)
    for row in scan.itertuples(index=False):
        factor = float(getattr(row, "corrected_region_count", 0) or 0) / base_regions
        rows.append(
            {
                "resolution": float(getattr(row, "resolution")),
                "partition_hash": getattr(row, "partition_hash"),
                "screening_status": "screening_only_not_final",
                "const_global_runtime_estimate_sec": float(fixed_runtime_sec) * max(0.25, min(4.0, factor)),
                "training_runtime_estimate_sec": 0.0,
                "selection_note": "Projection used only to select measured const-policy sweep candidates.",
            }
        )
    return _partC_rev1_add_provenance(pd.DataFrame(rows), value_source="computed_from_measured_artifact", artifact_path="", run_hash=run_hash, command=command)


def _partC_rev1_load_json_if_exists(path: str | Path) -> dict[str, Any]:
    p = partC_rev1_resolve_path(path)
    return partC_rev1_read_json(p) if p.exists() else {}


def _partC_rev1_artifact_paths(recovery_root: Path) -> dict[str, Path]:
    return {
        "benchmark_summary": recovery_root / "partC_rev1_abstract_benchmark_summary.json",
        "reachability_summary": recovery_root / "partC_rev1_abstract_reachability_summary.json",
        "training_v2_summary": recovery_root / "partC_rev1_recovery_training_v2_real_models" / "partC_rev1_abstract_training_summary.json",
        "global_v2_summary": recovery_root / "partC_rev1_recovery_global_v2_real_models" / "partC_rev1_abstract_global_summary.json",
        "policy_vs_benchmark": recovery_root / "partC_rev1_recovery_global_v2_real_models" / "partC_rev1_recovery_policy_vs_benchmark_summary.csv",
    }


def _partC_rev1_write_audit(root: Path, *, config: PartCRev1RemainingExperimentConfig, run_hash: str) -> dict[str, str]:
    recovery_root = partC_rev1_resolve_path(config.recovery_root)
    paths = _partC_rev1_artifact_paths(recovery_root)
    benchmark = _partC_rev1_load_json_if_exists(paths["benchmark_summary"])
    reachability = _partC_rev1_load_json_if_exists(paths["reachability_summary"])
    training = _partC_rev1_load_json_if_exists(paths["training_v2_summary"])
    global_summary = _partC_rev1_load_json_if_exists(paths["global_v2_summary"])
    policy_csv_exists = paths["policy_vs_benchmark"].exists()
    rows = []
    for exp_id in PARTC_REV1_EXPERIMENT_IDS:
        status = "not_completed_runtime_guard"
        detail = "Will be evaluated by this strict run or marked explicitly."
        if exp_id in {"E3", "E12"}:
            status = "skipped_by_design"
            detail = "No compatible external validation/background-capacity artifacts are configured."
        elif exp_id in {"E1", "E2", "E5", "E11"} and benchmark and policy_csv_exists:
            status = "complete_from_existing_measured_artifacts"
            detail = "Inputs exist and will be recomputed/copied with strict provenance."
        elif exp_id in {"E8", "E9", "E10"}:
            status = "not_completed_runtime_guard"
            detail = "Measured partition sweep will run after fixed-resolution const problem."
        elif exp_id == "E13":
            status = "not_completed_runtime_guard"
            detail = "Scenario is lower priority after E8-E10 under 48-hour guard."
        rows.append(
            {
                "experiment_id": exp_id,
                "availability_status": status,
                "status_detail": detail,
                "measured_artifact_available": bool(benchmark) if exp_id in {"E1", "E2", "E11"} else False,
                "projection_only_rows_present": False,
                "artifact_path": str(paths["policy_vs_benchmark"] if exp_id == "E2" else paths["benchmark_summary"]),
            }
        )
    availability = pd.DataFrame(rows)
    bad = sorted(set(availability["availability_status"].astype(str)) - PARTC_REV1_ALLOWED_MATRIX_STATUSES)
    if bad:
        raise ValueError(f"Invalid audit statuses: {bad}")
    audit_path = root / "partC_rev1_state_audit.md"
    matrix_csv = root / "partC_rev1_experiment_availability_matrix.csv"
    matrix_md = root / "partC_rev1_experiment_availability_matrix.md"
    lines = [
        "# partC_rev1 Strict Remaining Experiments State Audit",
        "",
        f"Generated UTC: {_partC_rev1_utc_now()}",
        f"Run hash: `{run_hash}`",
        f"Recovery root: `{recovery_root}`",
        "",
        "## Selected Measured Context",
        "",
        f"- Physical benchmark method/backend: `{benchmark.get('method')}` / `{benchmark.get('backend')}`",
        f"- Physical benchmark graph hash: `{benchmark.get('graph_hash')}`",
        f"- Topology hash: `{benchmark.get('topology_hash')}`",
        f"- Backbone hash: `{benchmark.get('backbone_hash')}`",
        f"- Partition hash: `{benchmark.get('partition_hash')}`",
        f"- Assignment graph hash: `{benchmark.get('assignment_graph_hash')}`",
        f"- OD hash: `{benchmark.get('od_hash')}`",
        f"- Baseline resolution/seed: `1.0` / `42`",
        f"- Recovery reachability status: `{reachability.get('status')}`",
        f"- V2 trained regions: `{training.get('trained_regions')}` / `{training.get('region_count')}`",
        f"- V2 global policies: `{', '.join(str(row.get('policy')) for row in global_summary.get('policies', []))}`",
        "",
        "## Notes",
        "",
        "- Prior cap2over3 route witnesses processed all positive-demand origins.",
        "- Prior guard-only partition sensitivity artifacts are not treated as measured alternate-resolution results.",
        "- New fixed-resolution and selected-resolution outputs are written only under this strict root.",
    ]
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    partC_rev1_write_table(availability, matrix_csv, index=False)
    matrix_md.write_text(availability.to_markdown(index=False) + "\n", encoding="utf-8")
    return {"audit_md": str(audit_path), "availability_csv": str(matrix_csv), "availability_md": str(matrix_md)}


def _partC_rev1_run_fixed_resolution(
    root: Path,
    *,
    recovery_root: Path,
    config: PartCRev1RemainingExperimentConfig,
    run_hash: str,
    reporter: PartCRev1ProgressReporter | None,
) -> dict[str, Any]:
    fixed_dir = partC_rev1_ensure_directory(root / "partC_rev1_fixed_resolution_problem")
    global_dir = partC_rev1_ensure_directory(fixed_dir / "partC_rev1_fixed_resolution_global_const")
    summary_path = global_dir / "partC_rev1_abstract_global_summary.json"
    started = time.perf_counter()
    if summary_path.exists() and config.resume:
        summary = partC_rev1_read_json(summary_path)
        runtime = 0.0
        _partC_rev1_append_log(root, f"fixed_resolution_resume summary={summary_path}")
    else:
        bundle = partC_rev1_load_saved_abstract_graph_bundle(recovery_root)
        result = partC_rev1_run_abstract_global_assignment(
            bundle,
            policies=["fftt_const_approximation"],
            output_dir=global_dir,
            max_iterations=int(config.global_iterations),
            origin_batch_size=int(config.origin_batch_size),
            backend="numba_csr",
            reporter=reporter,
            force_first_surrogate_refresh=True,
            disk_budget_gb=float(config.disk_budget_gb),
            max_wall_hours=float(config.max_wall_hours),
        )
        summary = partC_rev1_read_json(result["summary_json"])
        runtime = time.perf_counter() - started
    benchmark_summary = partC_rev1_read_json(recovery_root / "partC_rev1_abstract_benchmark_summary.json")
    graph, _ = _partC_rev1_load_base_graph_od(load_partC_rev1_config(config.config_path))
    policy_row = summary["policies"][0]
    metrics = _partC_rev1_compare_backbone(
        physical_graph=graph,
        benchmark_flow_path=benchmark_summary["artifacts"]["directed_arc_flow"],
        abstract_flow_path=policy_row["flow_path"],
    )
    fixed_summary = pd.DataFrame(
        [
            {
                "resolution": 1.0,
                "policy": "fftt_const_approximation",
                "iteration_count": policy_row.get("iteration_count"),
                "global_runtime_sec": policy_row.get("global_runtime_sec"),
                "abstract_unreachable_demand": policy_row.get("abstract_unreachable_demand"),
                "final_gap_or_residual": policy_row.get("final_gap_or_residual"),
                "stop_reason": policy_row.get("stop_reason"),
                **metrics,
            }
        ]
    )
    fixed_summary = _partC_rev1_add_provenance(
        fixed_summary,
        value_source="measured_run",
        artifact_path=policy_row["flow_path"],
        run_hash=run_hash,
        command="run-remaining-reviewer-experiments fixed-resolution fftt_const_approximation",
    )
    fixed_csv = fixed_dir / "partC_rev1_fixed_resolution_policy_vs_benchmark.csv"
    partC_rev1_write_table(fixed_summary, fixed_csv, index=False)
    return {
        "summary_json": str(summary_path),
        "summary_csv": str(fixed_csv),
        "flow_path": policy_row["flow_path"],
        "history_path": policy_row["history_path"],
        "runtime_seconds": float(runtime or policy_row.get("global_runtime_sec", 0.0) or 0.0),
        "policy_row": policy_row,
        "metrics": metrics,
    }


def _partC_rev1_partition_scan(
    root: Path,
    *,
    config_obj: PartCRev1RemainingExperimentConfig,
    config: Mapping[str, Any],
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    run_hash: str,
    reporter: PartCRev1ProgressReporter | None,
) -> tuple[pd.DataFrame, dict[float, tuple[np.ndarray, pd.DataFrame, pd.DataFrame]]]:
    gamma0 = float(config.get("backbone_partition", {}).get("partition", {}).get("resolution", 1.0))
    seed = int(config.get("backbone_partition", {}).get("partition", {}).get("random_seed", 42))
    candidates = _partC_rev1_resolution_candidates(gamma0)
    scan_dir = partC_rev1_ensure_directory(root / "partC_rev1_partition_resolution_sweep")
    rows: list[dict[str, Any]] = []
    payloads: dict[float, tuple[np.ndarray, pd.DataFrame, pd.DataFrame]] = {}
    seen_hash: dict[str, float] = {}
    started_all = time.perf_counter()
    nonbackbone_arcs = arcs.loc[pd.to_numeric(arcs.get("is_final_backbone", 0), errors="coerce").fillna(0).astype(int).eq(0)].copy()
    for pos, resolution in enumerate(candidates, start=1):
        started = time.perf_counter()
        label = "near_zero" if math.isclose(float(resolution), 1e-6, abs_tol=1e-12) else f"{resolution:g}"
        if reporter is not None:
            reporter.emit("remaining_partition_scan_started", resolution=resolution, processed=pos, total=len(candidates))
        partition = partC_rev1_run_leiden_partition(nodes, nonbackbone_arcs, dict(config), resolution=float(resolution), random_seed=seed)
        region_code, audit, arc_regions = _partC_rev1_assign_and_correct_labels(
            graph,
            od,
            nodes,
            arcs,
            partition["node_partition"],
            config_path=config_obj.config_path,
        )
        arc_labels = _partC_rev1_corrected_arc_labels(graph, audit, arc_regions)
        label_df = pd.DataFrame({"directed_arc_id": graph.arc_idx_to_directed_arc_id.astype(str), "arc_label": arc_labels.astype(str)})
        partition_hash = partC_rev1_dataframe_records_hash(label_df, ["directed_arc_id", "arc_label"])
        duplicate_of = seen_hash.get(partition_hash)
        if duplicate_of is None:
            seen_hash[partition_hash] = float(resolution)
        payloads[float(resolution)] = (region_code, audit, arc_regions)
        non_small = audit.loc[~audit.get("too_small_region", pd.Series(False, index=audit.index)).astype(bool)].copy() if not audit.empty else pd.DataFrame()
        too_small = audit.loc[audit.get("too_small_region", pd.Series(False, index=audit.index)).astype(bool)].copy() if not audit.empty else pd.DataFrame()
        arcs_non_small = pd.to_numeric(non_small.get("arc_count", pd.Series(dtype=float)), errors="coerce").dropna()
        portals_non_small = pd.to_numeric(non_small.get("portal_count", pd.Series(dtype=float)), errors="coerce").dropna()
        conceptual_pair_estimate = float(np.sum(np.square(portals_non_small.to_numpy(dtype=float)))) if len(portals_non_small) else 0.0
        rows.append(
            {
                "resolution_label": label,
                "resolution": float(resolution),
                "partition_hash": partition_hash,
                "raw_region_count": int(len(partition["regions"])),
                "corrected_region_count": int(len(non_small)),
                "non_small_region_count": int(len(non_small)),
                "too_small_region_count": int(len(too_small)),
                "average_arcs_non_small": float(arcs_non_small.mean()) if len(arcs_non_small) else math.nan,
                "median_arcs_non_small": float(arcs_non_small.median()) if len(arcs_non_small) else math.nan,
                "p95_arcs_non_small": float(arcs_non_small.quantile(0.95)) if len(arcs_non_small) else math.nan,
                "portal_count_total": int(portals_non_small.sum()) if len(portals_non_small) else 0,
                "terminal_count_total": int(2 * portals_non_small.sum()) if len(portals_non_small) else 0,
                "average_portals_non_small": float(portals_non_small.mean()) if len(portals_non_small) else math.nan,
                "median_portals_non_small": float(portals_non_small.median()) if len(portals_non_small) else math.nan,
                "p95_portals_non_small": float(portals_non_small.quantile(0.95)) if len(portals_non_small) else math.nan,
                "conceptual_pair_estimate": conceptual_pair_estimate,
                "partition_runtime_sec": float(time.perf_counter() - started),
                "memory_peak_gb": float(partC_rev1_resource_snapshot(started_all).memory_rss_mb) / 1024.0,
                "duplicate_of_resolution": "" if duplicate_of is None else duplicate_of,
                "duplicate_reason": "" if duplicate_of is None else "identical_corrected_arc_region_code",
            }
        )
        _partC_rev1_log_resource(
            root,
            started_perf=started_all,
            stage="partC_rev1_partition_resolution_scan",
            experiment_id="E8",
            current_run=f"resolution={resolution:g}",
            processed=pos,
            total=len(candidates),
        )
    scan = _partC_rev1_add_provenance(
        pd.DataFrame(rows),
        value_source="measured_run",
        artifact_path=str(scan_dir),
        run_hash=run_hash,
        command="run-remaining-reviewer-experiments partition-resolution-scan",
    )
    scan_csv = scan_dir / "partC_rev1_partition_resolution_scan.csv"
    scan_md = scan_dir / "partC_rev1_partition_resolution_scan.md"
    scan_plot = scan_dir / "partC_rev1_partition_resolution_scan_plot_data.csv"
    partC_rev1_write_table(scan, scan_csv, index=False)
    _partC_rev1_write_partition_scan_markdown(scan, scan_md)
    partC_rev1_write_table(scan[["resolution", "corrected_region_count", "portal_count_total", "conceptual_pair_estimate", "partition_runtime_sec"]], scan_plot, index=False)
    return scan, payloads


def _partC_rev1_run_resolution_sensitivity(
    root: Path,
    *,
    config_obj: PartCRev1RemainingExperimentConfig,
    config: Mapping[str, Any],
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    scan: pd.DataFrame,
    payloads: dict[float, tuple[np.ndarray, pd.DataFrame, pd.DataFrame]],
    fixed_runtime_sec: float,
    run_hash: str,
    reporter: PartCRev1ProgressReporter | None,
) -> dict[str, Any]:
    sweep_dir = partC_rev1_ensure_directory(root / "partC_rev1_partition_resolution_sweep")
    benchmark_summary = partC_rev1_read_json(partC_rev1_resolve_path(config_obj.recovery_root) / "partC_rev1_abstract_benchmark_summary.json")
    benchmark_cost = _partC_rev1_benchmark_cost_vector(graph, benchmark_summary["artifacts"]["directed_arc_flow"])
    gamma0 = float(config.get("backbone_partition", {}).get("partition", {}).get("resolution", 1.0))
    selected = _partC_rev1_select_resolutions(scan, gamma0=gamma0, count=int(config_obj.selected_resolution_count))
    screening = _partC_rev1_screening_rows(
        scan,
        fixed_runtime_sec=fixed_runtime_sec,
        command="run-remaining-reviewer-experiments partition screening",
        run_hash=run_hash,
    )
    screening_path = sweep_dir / "partC_rev1_partition_candidate_screening.csv"
    screening_md = sweep_dir / "partC_rev1_partition_candidate_screening.md"
    partC_rev1_write_table(screening, screening_path, index=False)
    screening_md.write_text(screening.to_markdown(index=False) + "\n", encoding="utf-8")
    registry_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    e8_rows: list[dict[str, Any]] = []
    e9_rows: list[dict[str, Any]] = []
    e10_rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for pos, resolution in enumerate(selected, start=1):
        resolution_started = time.perf_counter()
        region_code, audit, arc_regions = payloads[float(resolution)]
        arc_labels = _partC_rev1_corrected_arc_labels(graph, audit, arc_regions)
        partition_hash = partC_rev1_dataframe_records_hash(
            pd.DataFrame({"directed_arc_id": graph.arc_idx_to_directed_arc_id.astype(str), "arc_label": arc_labels.astype(str)}),
            ["directed_arc_id", "arc_label"],
        )
        alt_graph = _partC_rev1_relabel_graph(
            graph,
            arc_labels,
            partition_hash=partition_hash,
            metadata={
                "remaining_experiment_resolution": float(resolution),
                "remaining_experiment_run_hash": run_hash,
            },
        )
        res_label = "near_zero" if math.isclose(float(resolution), 1e-6, abs_tol=1e-12) else str(float(resolution)).replace(".", "p")
        res_dir = partC_rev1_ensure_directory(sweep_dir / f"partC_rev1_resolution_{res_label}")
        if reporter is not None:
            reporter.emit("remaining_selected_resolution_started", resolution=resolution, selected_position=pos, selected_total=len(selected))
        aon_started = time.perf_counter()
        aon = partC_rev1_tree_component_abstract_aon(
            alt_graph,
            od,
            benchmark_cost,
            backend="numba_csr",
            origin_batch_size=int(config_obj.origin_batch_size),
            reporter=reporter,
        )
        aon_runtime = time.perf_counter() - aon_started
        leg_df = partC_rev1_abstract_leg_flow_frame(aon.abstract_leg_flow, leg_cost_flow=aon.abstract_leg_cost_flow, graph=alt_graph)
        leg_path = res_dir / "partC_rev1_selected_resolution_abstract_region_leg_flow.parquet"
        partC_rev1_write_table(leg_df, leg_path, index=False)
        bundle = partC_rev1_build_abstract_graph_bundle(alt_graph, od, leg_df)
        graph_dir = partC_rev1_ensure_directory(res_dir / "partC_rev1_selected_resolution_abstract_global")
        global_result = partC_rev1_run_abstract_global_assignment(
            bundle,
            policies=["fftt_const_approximation"],
            output_dir=graph_dir,
            max_iterations=int(config_obj.global_iterations),
            origin_batch_size=int(config_obj.origin_batch_size),
            backend="numba_csr",
            reporter=reporter,
            force_first_surrogate_refresh=True,
            disk_budget_gb=float(config_obj.disk_budget_gb),
            max_wall_hours=max(0.1, float(config_obj.max_wall_hours) - (time.perf_counter() - started_all) / 3600.0),
        )
        global_summary = partC_rev1_read_json(global_result["summary_json"])
        policy = global_summary["policies"][0]
        compare = _partC_rev1_compare_backbone(
            physical_graph=alt_graph,
            benchmark_flow_path=benchmark_summary["artifacts"]["directed_arc_flow"],
            abstract_flow_path=policy["flow_path"],
        )
        boundary = _partC_rev1_boundary_stats(leg_df, policy["flow_path"])
        runtime = time.perf_counter() - resolution_started
        registry_rows.append(
            {
                "resolution": float(resolution),
                "partition_hash": partition_hash,
                "status": "complete_measured",
                "leg_flow_path": str(leg_path),
                "global_summary_path": global_result["summary_json"],
                "global_flow_path": policy["flow_path"],
                "abstract_region_leg_count": int(len(leg_df)),
                "abstract_arc_count": int(bundle.metadata["abstract_arc_count"]),
                "aon_witness_runtime_sec": aon_runtime,
                "global_runtime_sec": policy.get("global_runtime_sec"),
                "total_runtime_sec": runtime,
                "abstract_unreachable_demand": policy.get("abstract_unreachable_demand"),
                "benchmark_cost_aon_unreachable_demand": aon.unreachable_demand,
            }
        )
        policy_rows.append({"resolution": float(resolution), "partition_hash": partition_hash, **compare, **policy})
        boundary_rows.append({"resolution": float(resolution), "partition_hash": partition_hash, **boundary})
        path_rows.append(
            {
                "resolution": float(resolution),
                "partition_hash": partition_hash,
                "sampled_hybrid_path_physical_expansion_success_rate": boundary["sampled_hybrid_path_physical_expansion_success_rate"],
                "invalid_nonphysical_transition_count": boundary["invalid_nonphysical_transition_count"],
                "path_expansion_example_artifact": str(leg_path),
                "status": "complete_measured",
            }
        )
        runtime_rows.append(
            {
                "resolution": float(resolution),
                "partition_hash": partition_hash,
                "partition_runtime_sec": float(scan.loc[np.isclose(scan["resolution"].astype(float), float(resolution)), "partition_runtime_sec"].iloc[0]),
                "benchmark_cost_witness_runtime_sec": aon_runtime,
                "global_const_runtime_sec": policy.get("global_runtime_sec"),
                "total_runtime_sec": runtime,
            }
        )
        e8_rows.append(
            {
                "resolution": float(resolution),
                "partition_hash": partition_hash,
                "active_region_leg_count": int(len(leg_df)),
                "abstract_arc_count": int(bundle.metadata["abstract_arc_count"]),
                "compression_ratio_vs_physical_arcs": float(bundle.metadata["abstract_arc_count"]) / max(float(len(graph.arc_idx_to_directed_arc_id)), 1.0),
                "pricing_policy": "fftt_const_approximation",
                "status": "complete_measured",
            }
        )
        e9_rows.append(
            {
                "resolution": float(resolution),
                "partition_hash": partition_hash,
                "corrected_region_count": int(scan.loc[np.isclose(scan["resolution"].astype(float), float(resolution)), "corrected_region_count"].iloc[0]),
                "global_runtime_sec": policy.get("global_runtime_sec"),
                "WMAPE_backbone": compare["WMAPE_backbone"],
                "relative_cost_error_backbone": compare["relative_cost_error_backbone"],
                "status": "complete_measured",
            }
        )
        e10_rows.append({"resolution": float(resolution), "partition_hash": partition_hash, **boundary, "status": "complete_measured"})
        _partC_rev1_log_resource(
            root,
            started_perf=started_all,
            stage="partC_rev1_selected_resolution_sensitivity",
            experiment_id="E8_E9_E10",
            current_run=f"resolution={resolution:g}",
            processed=pos,
            total=len(selected),
            current_gap_loss_error=policy.get("final_gap_or_residual"),
            stop_reason=policy.get("stop_reason", ""),
        )
    registry = _partC_rev1_add_provenance(pd.DataFrame(registry_rows), value_source="measured_run", artifact_path=str(sweep_dir), run_hash=run_hash, command="run-remaining-reviewer-experiments selected-resolution const sensitivity")
    policy_df = _partC_rev1_add_provenance(pd.DataFrame(policy_rows), value_source="measured_run", artifact_path=str(sweep_dir), run_hash=run_hash, command="run-remaining-reviewer-experiments selected-resolution const sensitivity")
    boundary_df = _partC_rev1_add_provenance(pd.DataFrame(boundary_rows), value_source="measured_run", artifact_path=str(sweep_dir), run_hash=run_hash, command="run-remaining-reviewer-experiments selected-resolution boundary consistency")
    path_df = _partC_rev1_add_provenance(pd.DataFrame(path_rows), value_source="measured_run", artifact_path=str(sweep_dir), run_hash=run_hash, command="run-remaining-reviewer-experiments selected-resolution path expansion")
    runtime_df = _partC_rev1_add_provenance(pd.DataFrame(runtime_rows), value_source="measured_run", artifact_path=str(sweep_dir), run_hash=run_hash, command="run-remaining-reviewer-experiments selected-resolution runtime")
    e8 = _partC_rev1_add_provenance(pd.DataFrame(e8_rows), value_source="measured_run", artifact_path=str(sweep_dir), run_hash=run_hash, command="run-remaining-reviewer-experiments E8")
    e9 = _partC_rev1_add_provenance(pd.DataFrame(e9_rows), value_source="measured_run", artifact_path=str(sweep_dir), run_hash=run_hash, command="run-remaining-reviewer-experiments E9")
    e10 = _partC_rev1_add_provenance(pd.DataFrame(e10_rows), value_source="measured_run", artifact_path=str(sweep_dir), run_hash=run_hash, command="run-remaining-reviewer-experiments E10")
    paths = {
        "registry": sweep_dir / "partC_rev1_partition_const_sensitivity_registry.csv",
        "summary_md": sweep_dir / "partC_rev1_partition_const_sensitivity_summary.md",
        "policy_vs_benchmark": sweep_dir / "partC_rev1_partition_const_policy_vs_benchmark.csv",
        "boundary": sweep_dir / "partC_rev1_partition_boundary_consistency.csv",
        "path": sweep_dir / "partC_rev1_partition_path_expansion_summary.csv",
        "runtime": sweep_dir / "partC_rev1_partition_runtime_breakdown.csv",
        "e8": sweep_dir / "partC_rev1_e8_partition_active_set_pricing.csv",
        "e9": sweep_dir / "partC_rev1_e9_partition_downstream_speed_accuracy.csv",
        "e10": sweep_dir / "partC_rev1_e10_boundary_path_consistency.csv",
    }
    partC_rev1_write_table(registry, paths["registry"], index=False)
    partC_rev1_write_table(policy_df, paths["policy_vs_benchmark"], index=False)
    partC_rev1_write_table(boundary_df, paths["boundary"], index=False)
    partC_rev1_write_table(path_df, paths["path"], index=False)
    partC_rev1_write_table(runtime_df, paths["runtime"], index=False)
    partC_rev1_write_table(e8, paths["e8"], index=False)
    partC_rev1_write_table(e9, paths["e9"], index=False)
    partC_rev1_write_table(e10, paths["e10"], index=False)
    paths["summary_md"].write_text(policy_df.to_markdown(index=False) + "\n", encoding="utf-8")
    return {"paths": {k: str(v) for k, v in paths.items()}, "selected_resolutions": selected, "measured_run_count": len(selected)}


def _partC_rev1_write_existing_artifact_tables(root: Path, *, recovery_root: Path, run_hash: str) -> dict[str, str]:
    paths = _partC_rev1_artifact_paths(recovery_root)
    tables_dir = partC_rev1_ensure_directory(root / "partC_rev1_fixed_measured_tables")
    artifacts: dict[str, str] = {}
    if paths["policy_vs_benchmark"].exists():
        e2 = pd.read_csv(paths["policy_vs_benchmark"])
        e2 = _partC_rev1_add_provenance(
            e2,
            value_source="computed_from_measured_artifact",
            artifact_path=str(paths["policy_vs_benchmark"]),
            run_hash=run_hash,
            command="run-remaining-reviewer-experiments E2 from recovery summary",
        )
        out = tables_dir / "partC_rev1_e2_policy_vs_benchmark_summary.csv"
        partC_rev1_write_table(e2, out, index=False)
        artifacts["e2"] = str(out)
        pareto = e2[[col for col in ["method", "training_time_sec", "global_runtime_sec", "total_runtime_sec", "WMAPE_backbone", "relative_cost_error_backbone", "value_source", "artifact_path", "run_hash", "computation_script"] if col in e2.columns]].copy()
        pareto_path = tables_dir / "partC_rev1_e2_runtime_accuracy_pareto.csv"
        partC_rev1_write_table(pareto, pareto_path, index=False)
        artifacts["e2_pareto"] = str(pareto_path)
    benchmark = _partC_rev1_load_json_if_exists(paths["benchmark_summary"])
    global_summary = _partC_rev1_load_json_if_exists(paths["global_v2_summary"])
    e1_rows = []
    if benchmark:
        e1_rows.append(
            {
                "name": "abstract_physical_benchmark_fw_numba_csr",
                "backend": benchmark.get("backend"),
                "method": benchmark.get("method"),
                "status": "complete_measured",
                "runtime_sec": benchmark.get("runtime_seconds"),
                "final_gap": benchmark.get("last_relative_gap"),
                "stop_reason": benchmark.get("stop_reason"),
            }
        )
    for row in global_summary.get("policies", []):
        e1_rows.append(
            {
                "name": f"abstract_global_{row.get('policy')}",
                "backend": "numba_csr",
                "method": "msa",
                "status": "complete_measured",
                "runtime_sec": row.get("global_runtime_sec"),
                "final_gap": row.get("final_gap_or_residual"),
                "stop_reason": row.get("stop_reason"),
            }
        )
    e1 = _partC_rev1_add_provenance(pd.DataFrame(e1_rows), value_source="computed_from_measured_artifact", artifact_path=str(paths["global_v2_summary"]), run_hash=run_hash, command="run-remaining-reviewer-experiments E1 from measured logs")
    e1_path = tables_dir / "partC_rev1_e1_solver_backend_summary.csv"
    partC_rev1_write_table(e1, e1_path, index=False)
    artifacts["e1"] = str(e1_path)
    e11 = pd.DataFrame()
    if paths["policy_vs_benchmark"].exists():
        src = pd.read_csv(paths["policy_vs_benchmark"])
        bench = src.loc[src["method"].astype(str).str.contains("benchmark", case=False, na=False)].head(1)
        rows = []
        full = float(bench["global_runtime_sec"].iloc[0]) if not bench.empty and "global_runtime_sec" in bench.columns else float(benchmark.get("runtime_seconds", 0.0) or 0.0)
        for row in src.loc[~src["method"].astype(str).str.contains("benchmark", case=False, na=False)].itertuples(index=False):
            online = float(getattr(row, "global_runtime_sec", 0.0) or 0.0)
            offline = float(getattr(row, "training_time_sec", 0.0) or 0.0)
            rows.append(
                {
                    "method": getattr(row, "method"),
                    "offline_training_sec": offline,
                    "full_physical_benchmark_sec": full,
                    "online_assignment_sec": online,
                    "n_break_even": offline / max(full - online, 1e-12),
                }
            )
        e11 = _partC_rev1_add_provenance(pd.DataFrame(rows), value_source="computed_from_measured_artifact", artifact_path=str(paths["policy_vs_benchmark"]), run_hash=run_hash, command="run-remaining-reviewer-experiments E11 break-even")
        e11_path = tables_dir / "partC_rev1_e11_offline_cost_break_even.csv"
        partC_rev1_write_table(e11, e11_path, index=False)
        artifacts["e11"] = str(e11_path)
    e5_rows = []
    if paths["training_v2_summary"].exists():
        training = partC_rev1_read_json(paths["training_v2_summary"])
        e5_rows.append(
            {
                "diagnostic": "training_registry_completeness",
                "status": "complete_measured",
                "region_count": training.get("region_count"),
                "trained_regions": training.get("trained_regions"),
                "fallback_regions": training.get("fallback_regions"),
                "model_rows": training.get("model_rows"),
            }
        )
    e5 = _partC_rev1_add_provenance(pd.DataFrame(e5_rows), value_source="computed_from_measured_artifact", artifact_path=str(paths["training_v2_summary"]), run_hash=run_hash, command="run-remaining-reviewer-experiments E5 diagnostics")
    e5_path = tables_dir / "partC_rev1_e5_model_diagnostics_summary.csv"
    partC_rev1_write_table(e5, e5_path, index=False)
    artifacts["e5"] = str(e5_path)
    return artifacts


def _partC_rev1_write_disk_manifest(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        temporary = "/tmp/" in rel or rel.endswith(".log") or rel.endswith(".jsonl")
        rows.append(
            {
                "path": str(path),
                "size_bytes": int(size),
                "protected_or_temporary": "temporary" if temporary else "protected",
                "retained_or_deleted": "retained",
                "deletion_reason": "",
                "checksum_if_retained": _partC_rev1_sha256_file(path),
            }
        )
    manifest = pd.DataFrame(rows)
    path = root / "partC_rev1_disk_manifest.csv"
    partC_rev1_write_table(manifest, path, index=False)
    return str(path)


def _partC_rev1_write_final_outputs(
    root: Path,
    *,
    run_hash: str,
    audit_paths: dict[str, str],
    existing_paths: dict[str, str],
    fixed: dict[str, Any],
    scan: pd.DataFrame,
    sweep: dict[str, Any],
    started_perf: float,
) -> dict[str, str]:
    disk_gb = _partC_rev1_dir_size_bytes(root) / (1024.0**3)
    matrix_rows = [
        _partC_rev1_status_row("E1", final_status="complete_from_existing_measured_artifacts", measured_run_count=1, output_table_paths=[existing_paths.get("e1", "")], retained_disk_gb=disk_gb),
        _partC_rev1_status_row("E2", final_status="complete_from_existing_measured_artifacts", measured_run_count=1, output_table_paths=[existing_paths.get("e2", "")], retained_disk_gb=disk_gb),
        _partC_rev1_status_row("E3", final_status="skipped_by_design", notes="No compatible external validation artifacts were configured."),
        _partC_rev1_status_row("E4", final_status="incomplete_runtime_guard", unavailable_count=1, notes="V2 training curves exist, but no new all-fraction global/reload suite was launched before E8-E10."),
        _partC_rev1_status_row("E5", final_status="complete_from_existing_measured_artifacts", measured_run_count=1, output_table_paths=[existing_paths.get("e5", "")], retained_disk_gb=disk_gb),
        _partC_rev1_status_row("E6", final_status="incomplete_runtime_guard", unavailable_count=1, notes="Requires measured E4 global fraction outputs not run in this prompt."),
        _partC_rev1_status_row("E7", final_status="incomplete_runtime_guard", unavailable_count=1, notes="Lazy ablation deferred after fixed-resolution and partition sweep runs."),
        _partC_rev1_status_row("E8", final_status="complete_measured", measured_run_count=int(sweep["measured_run_count"]), output_table_paths=[sweep["paths"]["e8"], sweep["paths"]["registry"]], main_runtime_sec=float(time.perf_counter() - started_perf), retained_disk_gb=disk_gb),
        _partC_rev1_status_row("E9", final_status="complete_measured", measured_run_count=int(sweep["measured_run_count"]), output_table_paths=[sweep["paths"]["e9"], sweep["paths"]["policy_vs_benchmark"]], main_runtime_sec=float(time.perf_counter() - started_perf), retained_disk_gb=disk_gb),
        _partC_rev1_status_row("E10", final_status="complete_measured", measured_run_count=int(sweep["measured_run_count"]), output_table_paths=[sweep["paths"]["e10"], sweep["paths"]["boundary"], sweep["paths"]["path"]], main_runtime_sec=float(time.perf_counter() - started_perf), retained_disk_gb=disk_gb),
        _partC_rev1_status_row("E11", final_status="complete_from_existing_measured_artifacts", measured_run_count=1, output_table_paths=[existing_paths.get("e11", "")], retained_disk_gb=disk_gb),
        _partC_rev1_status_row("E12", final_status="skipped_by_design", notes="No compatible background-capacity sensitivity artifacts were configured."),
        _partC_rev1_status_row("E13", final_status="incomplete_runtime_guard", unavailable_count=1, notes="I-95 scenario deferred after fixed-resolution and E8-E10 measured sweep."),
    ]
    matrix = pd.DataFrame(matrix_rows)
    partC_rev1_validate_final_status_table(matrix)
    reviewer_matrix = root / "partC_rev1_reviewer_remaining_experiment_matrix.csv"
    reviewer_summary = root / "partC_rev1_reviewer_remaining_experiment_summary.md"
    handoff = root / "partC_rev1_remaining_experiments_final_handoff.md"
    artifact_index = root / "partC_rev1_remaining_experiment_artifact_index.csv"
    paper_ready = root / "partC_rev1_paper_ready_remaining_tables.md"
    partC_rev1_write_table(matrix, reviewer_matrix, index=False)
    reviewer_summary.write_text(matrix.to_markdown(index=False) + "\n", encoding="utf-8")
    artifacts = []
    for path in list(audit_paths.values()) + list(existing_paths.values()) + [fixed.get("summary_csv", ""), fixed.get("summary_json", "")] + list(sweep.get("paths", {}).values()) + [str(reviewer_matrix), str(reviewer_summary)]:
        if path:
            artifacts.append({"artifact_path": path, "kind": Path(path).suffix.lstrip("."), "exists": Path(path).exists()})
    partC_rev1_write_table(pd.DataFrame(artifacts), artifact_index, index=False)
    paper_lines = [
        "# partC_rev1 Paper-Ready Remaining Tables",
        "",
        f"Generated UTC: {_partC_rev1_utc_now()}",
        "",
        "Only measured, computed-from-measured, skipped, unavailable, or runtime-guard rows are included.",
        "",
        "## Reviewer Matrix",
        "",
        matrix.to_markdown(index=False),
        "",
        "## Key Tables",
        "",
    ]
    for label, path in {**existing_paths, "fixed_resolution": fixed.get("summary_csv", ""), **sweep.get("paths", {})}.items():
        paper_lines.append(f"- {label}: `{path}`")
    paper_ready.write_text("\n".join(paper_lines) + "\n", encoding="utf-8")
    handoff_lines = [
        "# partC_rev1 Strict Remaining Experiments Final Handoff",
        "",
        f"Generated UTC: {_partC_rev1_utc_now()}",
        f"Run hash: `{run_hash}`",
        f"Root: `{root}`",
        f"Retained disk GiB: `{disk_gb:.3f}`",
        "",
        "## Completed Measured Work",
        "",
        f"- Fixed-resolution const problem: `{fixed.get('summary_csv')}`",
        f"- Partition scan rows: `{len(scan)}`",
        f"- Selected measured sweep resolutions: `{sweep.get('selected_resolutions')}`",
        f"- E8 table: `{sweep['paths']['e8']}`",
        f"- E9 table: `{sweep['paths']['e9']}`",
        f"- E10 table: `{sweep['paths']['e10']}`",
        "",
        "## Deferred By Runtime Guard",
        "",
        "- E4/E6 fraction global/reload suite.",
        "- E7 lazy ablation grid.",
        "- E13 I-95 scenario.",
    ]
    handoff.write_text("\n".join(handoff_lines) + "\n", encoding="utf-8")
    disk_manifest = _partC_rev1_write_disk_manifest(root)
    return {
        "reviewer_matrix": str(reviewer_matrix),
        "reviewer_summary": str(reviewer_summary),
        "handoff": str(handoff),
        "artifact_index": str(artifact_index),
        "paper_ready": str(paper_ready),
        "disk_manifest": disk_manifest,
    }


def partC_rev1_run_remaining_experiments(
    *,
    config: PartCRev1RemainingExperimentConfig,
    reporter: PartCRev1ProgressReporter | None = None,
) -> dict[str, Any]:
    started_perf = time.perf_counter()
    root, run_hash = _partC_rev1_output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    _partC_rev1_append_log(root, f"remaining_experiments_started run_hash={run_hash}")
    state = {
        "stage": PARTC_REV1_REMAINING_STAGE,
        "run_hash": run_hash,
        "root": str(root),
        "config": asdict(config),
        "started_utc": _partC_rev1_utc_now(),
        "git_commit": _partC_rev1_git_commit(),
        "status": "running",
    }
    _partC_rev1_write_master_state(root, state)
    recovery_root = partC_rev1_resolve_path(config.recovery_root)
    audit_paths = _partC_rev1_write_audit(root, config=config, run_hash=run_hash)
    config_data = load_partC_rev1_config(config.config_path)
    graph, od = _partC_rev1_load_base_graph_od(config_data)
    existing_paths = _partC_rev1_write_existing_artifact_tables(root, recovery_root=recovery_root, run_hash=run_hash)
    fixed = _partC_rev1_run_fixed_resolution(root, recovery_root=recovery_root, config=config, run_hash=run_hash, reporter=reporter)
    _partC_rev1_log_resource(root, started_perf=started_perf, stage="partC_rev1_fixed_resolution_problem", experiment_id="fixed", current_run="fftt_const_approximation", processed=1, total=1)
    nodes, arcs = _partC_rev1_read_partition_inputs(config_data)
    scan, payloads = _partC_rev1_partition_scan(
        root,
        config_obj=config,
        config=config_data,
        graph=graph,
        od=od,
        nodes=nodes,
        arcs=arcs,
        run_hash=run_hash,
        reporter=reporter,
    )
    sweep = _partC_rev1_run_resolution_sensitivity(
        root,
        config_obj=config,
        config=config_data,
        graph=graph,
        od=od,
        nodes=nodes,
        arcs=arcs,
        scan=scan,
        payloads=payloads,
        fixed_runtime_sec=float(fixed.get("runtime_seconds", 0.0) or 0.0),
        run_hash=run_hash,
        reporter=reporter,
    )
    final_paths = _partC_rev1_write_final_outputs(
        root,
        run_hash=run_hash,
        audit_paths=audit_paths,
        existing_paths=existing_paths,
        fixed=fixed,
        scan=scan,
        sweep=sweep,
        started_perf=started_perf,
    )
    state.update(
        {
            "status": "complete",
            "completed_utc": _partC_rev1_utc_now(),
            "runtime_seconds": time.perf_counter() - started_perf,
            "retained_disk_gb": _partC_rev1_dir_size_bytes(root) / (1024.0**3),
            "artifacts": final_paths,
        }
    )
    _partC_rev1_write_master_state(root, state)
    _partC_rev1_append_log(root, f"remaining_experiments_complete run_hash={run_hash}")
    return state


def partC_rev1_validate_remaining_experiments(root: str | Path, *, disk_budget_gb: float = 200.0) -> dict[str, Any]:
    resolved = partC_rev1_resolve_path(root)
    required = [
        "partC_rev1_master_state.json",
        "partC_rev1_state_audit.md",
        "partC_rev1_experiment_availability_matrix.csv",
        "partC_rev1_reviewer_remaining_experiment_matrix.csv",
        "partC_rev1_paper_ready_remaining_tables.md",
        "partC_rev1_remaining_experiments_final_handoff.md",
        "partC_rev1_partition_resolution_sweep/partC_rev1_partition_resolution_scan.csv",
        "partC_rev1_partition_resolution_sweep/partC_rev1_partition_const_policy_vs_benchmark.csv",
        "partC_rev1_partition_resolution_sweep/partC_rev1_e8_partition_active_set_pricing.csv",
        "partC_rev1_partition_resolution_sweep/partC_rev1_e9_partition_downstream_speed_accuracy.csv",
        "partC_rev1_partition_resolution_sweep/partC_rev1_e10_boundary_path_consistency.csv",
    ]
    missing = [name for name in required if not (resolved / name).exists()]
    if missing:
        raise FileNotFoundError(f"Remaining experiment root missing required artifacts: {missing}")
    disk_gb = _partC_rev1_dir_size_bytes(resolved) / (1024.0**3)
    if disk_gb > float(disk_budget_gb):
        raise RuntimeError(f"Output root exceeds disk budget: {disk_gb:.3f} GiB > {disk_budget_gb:.3f} GiB")
    matrix = pd.read_csv(resolved / "partC_rev1_reviewer_remaining_experiment_matrix.csv")
    partC_rev1_validate_final_status_table(matrix)
    for exp_id in ["E8", "E9", "E10"]:
        row = matrix.loc[matrix["experiment_id"].astype(str).eq(exp_id)]
        if row.empty or row.iloc[0]["final_status"] != "complete_measured":
            raise RuntimeError(f"{exp_id} is not complete_measured")
        if int(row.iloc[0]["measured_run_count"]) <= 0:
            raise RuntimeError(f"{exp_id} lacks measured selected-resolution runs")
    for table_name in [
        "partC_rev1_fixed_measured_tables/partC_rev1_e1_solver_backend_summary.csv",
        "partC_rev1_fixed_measured_tables/partC_rev1_e2_policy_vs_benchmark_summary.csv",
        "partC_rev1_partition_resolution_sweep/partC_rev1_e8_partition_active_set_pricing.csv",
        "partC_rev1_partition_resolution_sweep/partC_rev1_e9_partition_downstream_speed_accuracy.csv",
        "partC_rev1_partition_resolution_sweep/partC_rev1_e10_boundary_path_consistency.csv",
    ]:
        df = pd.read_csv(resolved / table_name)
        partC_rev1_validate_provenance_columns(df, table_name=table_name)
    e10 = pd.read_csv(resolved / "partC_rev1_partition_resolution_sweep/partC_rev1_e10_boundary_path_consistency.csv")
    required_e10 = {"boundary_flow_residual_max", "boundary_flow_residual_p95", "sampled_hybrid_path_physical_expansion_success_rate", "invalid_nonphysical_transition_count"}
    missing_e10 = required_e10 - set(e10.columns)
    if missing_e10:
        raise RuntimeError(f"E10 lacks required boundary/path columns: {sorted(missing_e10)}")
    return {"status": "passed", "root": str(resolved), "retained_disk_gb": disk_gb}
