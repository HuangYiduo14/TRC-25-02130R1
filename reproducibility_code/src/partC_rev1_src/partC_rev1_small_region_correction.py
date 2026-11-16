"""Post-capacity one-pass correction for too-small training regions."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .partC_rev1_graph_snapshot import PartCRev1GraphSnapshot
from .partC_rev1_hashing import partC_rev1_json_hash, partC_rev1_numpy_array_hash, partC_rev1_sort_key
from .partC_rev1_io import partC_rev1_write_json, partC_rev1_write_table
from .partC_rev1_od_snapshot import PartCRev1ODSnapshot
from .partC_rev1_paths import partC_rev1_resolve_path
from .partC_rev1_progress_logging import PartCRev1ProgressReporter
from .partC_rev1_regional_training import partC_rev1_region_size_table
from .partC_rev1_runtime_guard import partC_rev1_resource_snapshot


PARTC_REV1_SMALL_REGION_STAGE = "partC_rev1_small_region_correction"


@dataclass(frozen=True, slots=True)
class PartCRev1SmallRegionCorrectionConfig:
    config_path: str
    min_arc_count: int = 5
    portal_square_arc_count_threshold: int = 20
    heartbeat_seconds: float = 15.0


def _partC_rev1_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _partC_rev1_graph_labels(graph: PartCRev1GraphSnapshot) -> np.ndarray:
    labels = np.asarray(graph.region_idx_to_label, dtype=object)
    return labels[graph.region_code.astype(np.int64)].astype(str)


def partC_rev1_mark_too_small_regions(
    region_table: pd.DataFrame,
    *,
    min_arc_count: int = 5,
    portal_square_arc_count_threshold: int = 20,
) -> pd.DataFrame:
    """Return a one-pass too-small-region audit table.

    The rule comes from the post-capacity continuation prompt. It is evaluated
    against the original labels once; corrected labels are not recursively
    reclassified.
    """

    required = {"region_uuid", "arc_count", "portal_count"}
    missing = sorted(required - set(region_table.columns))
    if missing:
        raise ValueError(f"Region table missing required columns: {missing}")
    audit = region_table.copy()
    audit["original_region_uuid"] = audit["region_uuid"].astype(str)
    audit["too_small_region"] = audit["arc_count"].astype(int).lt(int(min_arc_count)) | (
        audit["arc_count"].astype(int).lt(int(portal_square_arc_count_threshold))
        & (audit["portal_count"].astype(int) ** 2 > audit["arc_count"].astype(int))
    )
    audit["corrected_region_uuid"] = np.where(audit["too_small_region"], "backbone", audit["original_region_uuid"])
    audit["correction_reason"] = ""
    audit.loc[audit["arc_count"].astype(int).lt(int(min_arc_count)), "correction_reason"] = "arc_count_below_min"
    portal_rule = (
        audit["arc_count"].astype(int).ge(int(min_arc_count))
        & audit["arc_count"].astype(int).lt(int(portal_square_arc_count_threshold))
        & (audit["portal_count"].astype(int) ** 2 > audit["arc_count"].astype(int))
    )
    audit.loc[portal_rule, "correction_reason"] = "portal_square_exceeds_arc_count"
    return audit


def partC_rev1_correct_small_region_graph(
    graph: PartCRev1GraphSnapshot,
    region_audit: pd.DataFrame,
    *,
    config: PartCRev1SmallRegionCorrectionConfig,
) -> tuple[PartCRev1GraphSnapshot, dict[str, Any]]:
    """Create a corrected snapshot by recoding too-small regions to backbone."""

    original_labels = _partC_rev1_graph_labels(graph)
    too_small = set(
        region_audit.loc[region_audit["too_small_region"].astype(bool), "original_region_uuid"].astype(str).tolist()
    )
    corrected_labels = original_labels.copy()
    if too_small:
        mask = np.isin(corrected_labels, np.asarray(sorted(too_small), dtype=str))
        corrected_labels[mask] = "backbone"
    labels_sorted = ["backbone"] + sorted(
        {str(label) for label in corrected_labels.tolist() if str(label) != "backbone"},
        key=partC_rev1_sort_key,
    )
    label_to_code = {label: idx for idx, label in enumerate(labels_sorted)}
    corrected_region_code = np.asarray([label_to_code[str(label)] for label in corrected_labels.tolist()], dtype=np.int32)
    correction_full_hash = partC_rev1_numpy_array_hash(
        graph.arc_idx_to_directed_arc_id.astype(str),
        original_labels.astype(str),
        corrected_labels.astype(str),
        np.asarray([partC_rev1_json_hash(asdict(config))], dtype=str),
        np.asarray([graph.topology_hash, graph.backbone_hash, graph.partition_hash], dtype=str),
    )
    correction_hash = correction_full_hash[:16]
    corrected_arc_count = int(np.sum(original_labels != corrected_labels))
    metadata = {
        **dict(graph.metadata),
        "stage": PARTC_REV1_SMALL_REGION_STAGE,
        "parent_graph_hash": graph.graph_hash,
        "parent_topology_hash": graph.topology_hash,
        "parent_backbone_hash": graph.backbone_hash,
        "parent_assignment_graph_hash": graph.assignment_graph_hash,
        "parent_partition_hash": graph.partition_hash,
        "correction_hash": correction_hash,
        "correction_full_hash": correction_full_hash,
        "small_region_rule": asdict(config),
        "small_region_count": int(len(too_small)),
        "corrected_arc_count": corrected_arc_count,
        "unchanged_arc_count": int(len(original_labels) - corrected_arc_count),
        "created_utc": _partC_rev1_utc_now(),
    }
    graph_hash = partC_rev1_json_hash(
        {
            "stage": PARTC_REV1_SMALL_REGION_STAGE,
            "parent_assignment_graph_hash": graph.assignment_graph_hash,
            "parent_partition_hash": graph.partition_hash,
            "correction_full_hash": correction_full_hash,
            "assignment_capacity_multipliers": graph.metadata.get("assignment_capacity_multipliers", {}),
            "arc_count": int(len(graph.arc_idx_to_directed_arc_id)),
            "node_count": int(len(graph.node_idx_to_uid)),
        }
    )
    corrected_graph = PartCRev1GraphSnapshot(
        graph_hash=graph_hash,
        partition_hash=f"{graph.partition_hash}:small_region_correction:{correction_hash}",
        node_idx_to_uid=graph.node_idx_to_uid.copy(),
        arc_idx_to_directed_arc_id=graph.arc_idx_to_directed_arc_id.copy(),
        arc_idx_to_link_uid=graph.arc_idx_to_link_uid.copy(),
        tail=graph.tail.copy(),
        head=graph.head.copy(),
        free_time=graph.free_time.copy(),
        base_capacity=graph.base_capacity.copy(),
        capacity=graph.capacity.copy(),
        capacity_multiplier=graph.capacity_multiplier.copy(),
        capacity_multiplier_type=graph.capacity_multiplier_type.astype(str).copy(),
        alpha=graph.alpha.copy(),
        beta=graph.beta.copy(),
        length=graph.length.copy(),
        region_code=corrected_region_code,
        region_idx_to_label=labels_sorted,
        csr_indptr=graph.csr_indptr.copy(),
        csr_head=graph.csr_head.copy(),
        csr_arc_idx=graph.csr_arc_idx.copy(),
        reverse_csr_indptr=graph.reverse_csr_indptr.copy(),
        reverse_csr_head=graph.reverse_csr_head.copy(),
        reverse_csr_arc_idx=graph.reverse_csr_arc_idx.copy(),
        metadata=metadata,
    )
    corrected_graph.metadata["topology_hash"] = graph.topology_hash
    corrected_graph.metadata["backbone_hash"] = graph.backbone_hash
    corrected_graph.metadata["assignment_graph_hash"] = graph_hash
    return corrected_graph, metadata


def partC_rev1_run_small_region_correction_stage(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    *,
    config: PartCRev1SmallRegionCorrectionConfig,
    output_dir: str | Path,
    cache_dir: str | Path,
    reporter: PartCRev1ProgressReporter | None = None,
) -> dict[str, Any]:
    out_root = partC_rev1_resolve_path(output_dir)
    cache_root = partC_rev1_resolve_path(cache_dir)
    started_perf = time.perf_counter()
    started = datetime.now(timezone.utc)
    region_table = partC_rev1_region_size_table(graph, od)
    audit = partC_rev1_mark_too_small_regions(
        region_table,
        min_arc_count=config.min_arc_count,
        portal_square_arc_count_threshold=config.portal_square_arc_count_threshold,
    )
    corrected_graph, correction_metadata = partC_rev1_correct_small_region_graph(graph, audit, config=config)
    correction_hash = str(correction_metadata["correction_hash"])
    run_name = f"partC_rev1_small_region_correction_{correction_hash}_assignment-{graph.assignment_graph_hash[:12]}"
    run_output_dir = out_root / run_name
    run_cache_dir = cache_root / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_cache_dir.mkdir(parents=True, exist_ok=True)
    if reporter is not None:
        reporter.emit(
            "small_region_correction_started",
            parent_graph_hash=graph.graph_hash,
            corrected_graph_hash=corrected_graph.graph_hash,
            small_region_count=correction_metadata["small_region_count"],
            corrected_arc_count=correction_metadata["corrected_arc_count"],
        )

    graph_paths = corrected_graph.save(run_cache_dir)
    audit_path = run_output_dir / "partC_rev1_small_region_correction_region_audit.csv"
    original_terminal_path = run_output_dir / "partC_rev1_small_region_correction_original_terminal_registry.csv"
    corrected_terminal_path = run_output_dir / "partC_rev1_small_region_correction_terminal_registry.csv"
    partC_rev1_write_table(audit, audit_path, index=False)
    partC_rev1_write_table(region_table, original_terminal_path, index=False)
    corrected_region_table = partC_rev1_region_size_table(corrected_graph, od)
    partC_rev1_write_table(corrected_region_table, corrected_terminal_path, index=False)
    summary = {
        "stage": PARTC_REV1_SMALL_REGION_STAGE,
        "created_utc": _partC_rev1_utc_now(),
        "config": asdict(config),
        "parent_graph_hash": graph.graph_hash,
        "topology_hash": graph.topology_hash,
        "backbone_hash": graph.backbone_hash,
        "assignment_graph_hash": graph.assignment_graph_hash,
        "corrected_graph_hash": corrected_graph.graph_hash,
        "parent_partition_hash": graph.partition_hash,
        "corrected_partition_hash": corrected_graph.partition_hash,
        "od_hash": od.od_hash,
        "correction_hash": correction_hash,
        "small_region_count": int(correction_metadata["small_region_count"]),
        "corrected_arc_count": int(correction_metadata["corrected_arc_count"]),
        "original_nonbackbone_region_count": int(len(region_table)),
        "corrected_nonbackbone_region_count": int(len(corrected_region_table)),
        "run_output_dir": str(run_output_dir),
        "run_cache_dir": str(run_cache_dir),
        "graph_paths": graph_paths,
        "artifacts": {
            "region_audit_csv": str(audit_path),
            "original_terminal_registry_csv": str(original_terminal_path),
            "corrected_terminal_registry_csv": str(corrected_terminal_path),
        },
        "source_maps_modified": False,
        "started_utc": started.isoformat(),
    }
    summary_path = run_output_dir / "partC_rev1_small_region_correction_summary.json"
    summary_md_path = run_output_dir / "partC_rev1_small_region_correction_summary.md"
    partC_rev1_write_json(summary, summary_path)
    summary["summary_path"] = str(summary_path)
    summary["summary_md_path"] = str(summary_md_path)
    lines = [
        "# partC_rev1 Small Region Correction Summary",
        "",
        f"Generated UTC: {summary['created_utc']}",
        "",
        f"- Parent graph hash: `{graph.graph_hash}`",
        f"- Corrected graph hash: `{corrected_graph.graph_hash}`",
        f"- Correction hash: `{correction_hash}`",
        f"- Too-small regions recoded to backbone: {summary['small_region_count']}",
        f"- Directed arcs recoded to backbone: {summary['corrected_arc_count']}",
        f"- Corrected non-backbone regions: {summary['corrected_nonbackbone_region_count']}",
        f"- Source maps modified: `{summary['source_maps_modified']}`",
        "",
        "## Artifacts",
        "",
    ]
    for label, path in {**graph_paths, **summary["artifacts"], "summary_json": str(summary_path)}.items():
        lines.append(f"- {label}: `{path}`")
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    partC_rev1_write_json(summary, summary_path)
    if reporter is not None:
        resource = partC_rev1_resource_snapshot(started_perf)
        reporter.emit(
            "small_region_correction_finished",
            summary_path=str(summary_path),
            corrected_graph_hash=corrected_graph.graph_hash,
            memory_rss_mb=resource.memory_rss_mb,
            disk_free_mb=resource.disk_free_mb,
        )
    return summary


def partC_rev1_load_cost_from_directed_flow(graph: PartCRev1GraphSnapshot, directed_flow_path: str | Path) -> np.ndarray:
    path = partC_rev1_resolve_path(directed_flow_path)
    df = pd.read_parquet(path, columns=["arc_idx", "cost"])
    cost = graph.free_time.copy()
    idx = df["arc_idx"].to_numpy(dtype=np.int64)
    if np.any(idx < 0) or np.any(idx >= len(cost)):
        raise ValueError(f"Directed flow arc_idx is incompatible with graph: {path}")
    cost[idx] = df["cost"].to_numpy(dtype=np.float64)
    return cost


def partC_rev1_append_route_witnesses_to_correction_summary(summary_path: str | Path, route_witnesses: list[dict[str, Any]]) -> dict[str, Any]:
    path = partC_rev1_resolve_path(summary_path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["route_witnesses"] = route_witnesses
    partC_rev1_write_json(summary, path)
    md_path = Path(str(summary.get("summary_md_path", "")))
    if md_path.exists():
        lines = md_path.read_text(encoding="utf-8").rstrip().splitlines()
        lines.extend(["", "## Route Witnesses", ""])
        for witness in route_witnesses:
            lines.append(
                f"- `{witness.get('witness_set')}`: `{witness.get('path')}`; rows {witness.get('rows')}, "
                f"processed origins {witness.get('processed_origins')}/{witness.get('total_origins')}, "
                f"coverage share {witness.get('coverage_share')}"
            )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
