"""Postcapacity physical reload suite for corrected global policy runs."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .partC_rev1_bpr import partC_rev1_bpr_cost
from .partC_rev1_graph_snapshot import PartCRev1GraphSnapshot
from .partC_rev1_hashing import partC_rev1_json_hash
from .partC_rev1_io import partC_rev1_read_json, partC_rev1_write_json, partC_rev1_write_table
from .partC_rev1_paths import partC_rev1_resolve_path
from .partC_rev1_physical_reload import (
    PartCRev1PhysicalReloadConfig,
    partC_rev1_reload_region,
)
from .partC_rev1_progress_logging import PartCRev1ProgressReporter
from .partC_rev1_regional_training import partC_rev1_load_benchmark_flow
from .partC_rev1_runtime_guard import PARTC_REV1_DEFAULT_MAX_RUN_SECONDS, partC_rev1_resource_snapshot


PARTC_REV1_PHYSICAL_RELOAD_SUITE_STAGE = "partC_rev1_physical_reload_suite"


@dataclass(frozen=True, slots=True)
class PartCRev1PhysicalReloadSuiteConfig:
    config_path: str
    global_policy_summary_path: str = ""
    max_runs: int = 0
    max_regions: int = 0
    max_iterations: int = 8
    gap_target: float = 1e-4
    positive_flow_tol: float = 1e-9
    runtime_guard_seconds: float = PARTC_REV1_DEFAULT_MAX_RUN_SECONDS
    resume: bool = True


def _partC_rev1_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def partC_rev1_find_latest_global_policy_summary(output_dir: str | Path) -> Path:
    out_dir = partC_rev1_resolve_path(output_dir)
    candidates = sorted(
        out_dir.glob("partC_rev1_global_policy_runs_*/partC_rev1_global_policy_suite_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No partC_rev1_global_policy_runs_* summary was found")
    return candidates[0]


def _partC_rev1_region_labels(graph: PartCRev1GraphSnapshot) -> np.ndarray:
    labels = np.asarray(graph.region_idx_to_label, dtype=object)
    return labels[graph.region_code.astype(np.int64)].astype(str)


def _partC_rev1_safe_ratio(numerator: np.ndarray | pd.Series, denominator: np.ndarray | pd.Series) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 0.0)


def _partC_rev1_metric_row(
    *,
    scope_kind: str,
    scope_value: str,
    predicted: np.ndarray,
    reference: np.ndarray,
    free_time: np.ndarray,
    capacity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    length: np.ndarray,
) -> dict[str, Any]:
    pred = predicted.astype(np.float64)
    ref = reference.astype(np.float64)
    diff = pred - ref
    abs_diff = np.abs(diff)
    pred_cost = partC_rev1_bpr_cost(pred, free_time, capacity, alpha, beta)
    ref_cost = partC_rev1_bpr_cost(ref, free_time, capacity, alpha, beta)
    pred_tstt = float(np.dot(pred, pred_cost))
    ref_tstt = float(np.dot(ref, ref_cost))
    pred_ton_mile = float(np.dot(pred, length))
    ref_ton_mile = float(np.dot(ref, length))
    denom = max(float(np.sum(np.abs(ref))), 1e-12)
    if len(pred) > 1 and np.std(pred) > 0.0 and np.std(ref) > 0.0:
        pearson = float(np.corrcoef(pred, ref)[0, 1])
    else:
        pearson = math.nan
    return {
        "metric_scope_kind": scope_kind,
        "metric_scope_value": scope_value,
        "arc_count": int(len(pred)),
        "reference_positive_arc_count": int(np.count_nonzero(ref > 0.0)),
        "predicted_positive_arc_count": int(np.count_nonzero(pred > 0.0)),
        "predicted_total_flow": float(np.sum(pred)),
        "reference_total_flow": float(np.sum(ref)),
        "mae": float(np.mean(abs_diff)) if len(abs_diff) else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if len(diff) else 0.0,
        "wmape": float(np.sum(abs_diff) / denom),
        "flow_weighted_mae": float(np.dot(np.abs(ref), abs_diff) / max(float(np.sum(np.abs(ref))), 1e-12)),
        "pearson": pearson,
        "max_abs_flow_error": float(np.max(abs_diff)) if len(abs_diff) else 0.0,
        "p95_abs_flow_error": float(np.quantile(abs_diff, 0.95)) if len(abs_diff) else 0.0,
        "p99_abs_flow_error": float(np.quantile(abs_diff, 0.99)) if len(abs_diff) else 0.0,
        "predicted_tstt": pred_tstt,
        "reference_tstt": ref_tstt,
        "relative_tstt_error": (pred_tstt - ref_tstt) / max(abs(ref_tstt), 1e-12),
        "predicted_ton_mile": pred_ton_mile,
        "reference_ton_mile": ref_ton_mile,
        "relative_ton_mile_error": (pred_ton_mile - ref_ton_mile) / max(abs(ref_ton_mile), 1e-12),
    }


def partC_rev1_physical_reload_validation_metrics(
    graph: PartCRev1GraphSnapshot,
    predicted: np.ndarray,
    reference: np.ndarray,
) -> pd.DataFrame:
    labels = _partC_rev1_region_labels(graph)
    rows: list[dict[str, Any]] = []

    def append(scope_kind: str, scope_value: str, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        rows.append(
            _partC_rev1_metric_row(
                scope_kind=scope_kind,
                scope_value=scope_value,
                predicted=predicted[mask],
                reference=reference[mask],
                free_time=graph.free_time[mask],
                capacity=graph.capacity[mask],
                alpha=graph.alpha[mask],
                beta=graph.beta[mask],
                length=graph.length[mask],
            )
        )

    all_mask = np.ones(len(predicted), dtype=bool)
    append("overall", "all", all_mask)
    append("component", "backbone", labels == "backbone")
    append("component", "regional", labels != "backbone")
    for value in sorted(set(graph.capacity_multiplier_type.astype(str).tolist())):
        append("capacity_multiplier_type", value, graph.capacity_multiplier_type.astype(str) == value)
    for value in sorted(set(labels.tolist())):
        append("region_label", value, labels == value)

    nonzero_reference = reference[reference > 0.0]
    if len(nonzero_reference) >= 4:
        quantiles = np.quantile(nonzero_reference, [0.25, 0.5, 0.75])
        bins = np.digitize(reference, quantiles, right=True)
        for idx, name in enumerate(["q1_lowest", "q2", "q3", "q4_highest"]):
            append("reference_flow_quantile", name, (reference > 0.0) & (bins == idx))
        append("reference_flow_quantile", "zero_reference_flow", reference <= 0.0)
    else:
        append("reference_flow_quantile", "all_positive_reference_flow", reference > 0.0)
        append("reference_flow_quantile", "zero_reference_flow", reference <= 0.0)
    return pd.DataFrame(rows)


def _partC_rev1_join_policy_pairs(active_pairs_path: str | Path, regional_pair_flow_path: str | Path) -> pd.DataFrame:
    active_pairs = pd.read_csv(partC_rev1_resolve_path(active_pairs_path))
    regional_pair_flow = pd.read_parquet(partC_rev1_resolve_path(regional_pair_flow_path))
    flow_cols = regional_pair_flow[["pair_id", "region_uuid", "flow", "cost"]].rename(columns={"cost": "surrogate_cost"})
    pairs = active_pairs.merge(flow_cols, on=["pair_id", "region_uuid"], how="left", validate="one_to_one")
    pairs["flow"] = pairs["flow"].fillna(0.0).astype(float)
    pairs["surrogate_cost"] = pairs["surrogate_cost"].fillna(pairs["tau0"]).astype(float)
    return pairs


def _partC_rev1_augment_final_tables(
    graph: PartCRev1GraphSnapshot,
    directed_flow: pd.DataFrame,
    link_flow: pd.DataFrame,
    benchmark_flow: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    directed = directed_flow.copy()
    directed["partC_rev1_benchmark_flow"] = benchmark_flow.astype(np.float64)
    directed["partC_rev1_flow_error"] = directed["partC_rev1_final_flow"].astype(float) - directed["partC_rev1_benchmark_flow"].astype(float)
    directed["partC_rev1_abs_flow_error"] = directed["partC_rev1_flow_error"].abs()
    directed["partC_rev1_benchmark_cost"] = partC_rev1_bpr_cost(
        benchmark_flow.astype(np.float64),
        graph.free_time,
        graph.capacity,
        graph.alpha,
        graph.beta,
    )
    directed["benchmark_flow_to_base_capacity_ratio"] = _partC_rev1_safe_ratio(directed["partC_rev1_benchmark_flow"], directed["base_capacity"])
    directed["benchmark_flow_to_assignment_capacity_ratio"] = _partC_rev1_safe_ratio(
        directed["partC_rev1_benchmark_flow"],
        directed["assignment_capacity"],
    )

    benchmark_by_link = (
        pd.DataFrame({"partC_rev1_link_uid": graph.arc_idx_to_link_uid, "partC_rev1_benchmark_link_flow": benchmark_flow})
        .groupby("partC_rev1_link_uid", as_index=False)["partC_rev1_benchmark_link_flow"]
        .sum()
    )
    link = link_flow.merge(benchmark_by_link, on="partC_rev1_link_uid", how="left", validate="one_to_one")
    link["partC_rev1_benchmark_link_flow"] = link["partC_rev1_benchmark_link_flow"].fillna(0.0).astype(float)
    link["partC_rev1_link_flow_error"] = link["partC_rev1_final_link_flow"].astype(float) - link["partC_rev1_benchmark_link_flow"]
    link["partC_rev1_abs_link_flow_error"] = link["partC_rev1_link_flow_error"].abs()
    link["benchmark_flow_to_base_capacity_ratio"] = _partC_rev1_safe_ratio(link["partC_rev1_benchmark_link_flow"], link["base_capacity"])
    link["benchmark_flow_to_assignment_capacity_ratio"] = _partC_rev1_safe_ratio(
        link["partC_rev1_benchmark_link_flow"],
        link["assignment_capacity"],
    )
    return directed, link


def _partC_rev1_merge_policy_physical_flows(
    graph: PartCRev1GraphSnapshot,
    explicit_physical_flow: pd.DataFrame,
    regional_reload_flow: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    labels = _partC_rev1_region_labels(graph)
    assigned = np.zeros(len(graph.arc_idx_to_directed_arc_id), dtype=np.int8)
    final_flow = np.zeros(len(graph.arc_idx_to_directed_arc_id), dtype=np.float64)
    if "source_ref" not in explicit_physical_flow.columns or "flow" not in explicit_physical_flow.columns:
        raise ValueError("Phase 3 explicit physical flow must contain source_ref and flow columns")
    for row in explicit_physical_flow.itertuples(index=False):
        arc_idx = int(row.source_ref)
        if arc_idx < 0 or arc_idx >= len(final_flow):
            raise ValueError(f"Explicit physical source_ref out of range: {arc_idx}")
        if assigned[arc_idx]:
            raise ValueError(f"Duplicate explicit physical flow assignment for arc {arc_idx}")
        final_flow[arc_idx] = float(row.flow)
        assigned[arc_idx] = 1
    unassigned_mask = assigned == 0
    final_flow[unassigned_mask] = np.asarray(regional_reload_flow, dtype=np.float64)[unassigned_mask]
    assigned[unassigned_mask] = 1
    if np.any(assigned != 1):
        missing = np.flatnonzero(assigned != 1)[:10].tolist()
        raise RuntimeError(f"Final policy flow merge did not assign every arc exactly once; sample={missing}")
    cost = partC_rev1_bpr_cost(final_flow, graph.free_time, graph.capacity, graph.alpha, graph.beta)
    directed = pd.DataFrame(
        {
            "arc_idx": np.arange(len(final_flow), dtype=np.int64),
            "directed_arc_id": graph.arc_idx_to_directed_arc_id,
            "partC_rev1_link_uid": graph.arc_idx_to_link_uid,
            "tail_node_uid": graph.node_idx_to_uid[graph.tail],
            "head_node_uid": graph.node_idx_to_uid[graph.head],
            "arc_label": labels,
            "physical_reload_source": np.where(unassigned_mask, "regional_reload", "explicit_physical_fallback_or_backbone"),
            "partC_rev1_final_flow": final_flow,
            "partC_rev1_final_cost": cost,
            "free_time": graph.free_time,
            "base_capacity": graph.base_capacity,
            "assignment_capacity": graph.capacity,
            "remaining_capacity": graph.capacity,
            "capacity_multiplier": graph.capacity_multiplier,
            "capacity_multiplier_type": graph.capacity_multiplier_type.astype(str),
            "length_miles": graph.length,
        }
    )
    directed["flow_to_base_capacity_ratio"] = _partC_rev1_safe_ratio(directed["partC_rev1_final_flow"], directed["base_capacity"])
    directed["flow_to_assignment_capacity_ratio"] = _partC_rev1_safe_ratio(
        directed["partC_rev1_final_flow"],
        directed["assignment_capacity"],
    )
    link = (
        directed.groupby("partC_rev1_link_uid", as_index=False)
        .agg(
            partC_rev1_final_link_flow=("partC_rev1_final_flow", "sum"),
            directed_arc_count=("arc_idx", "count"),
            total_directed_length_miles=("length_miles", "sum"),
            base_capacity=("base_capacity", "sum"),
            assignment_capacity=("assignment_capacity", "sum"),
        )
        .sort_values("partC_rev1_link_uid", kind="mergesort")
        .reset_index(drop=True)
    )
    link["flow_to_base_capacity_ratio"] = _partC_rev1_safe_ratio(link["partC_rev1_final_link_flow"], link["base_capacity"])
    link["flow_to_assignment_capacity_ratio"] = _partC_rev1_safe_ratio(
        link["partC_rev1_final_link_flow"],
        link["assignment_capacity"],
    )
    return final_flow, directed, link


def _partC_rev1_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# partC_rev1 Postcapacity Physical Reload Suite",
        "",
        f"Generated UTC: {summary.get('created_utc')}",
        "",
        f"- State: `{summary.get('state')}`",
        f"- Suite hash: `{summary.get('suite_hash')}`",
        f"- Corrected graph hash: `{summary.get('corrected_graph_hash')}`",
        f"- Selected 4C benchmark hash: `{summary.get('selected_4c_benchmark_hash')}`",
        f"- Processed runs: {summary.get('processed_run_count')}/{summary.get('candidate_run_count')}",
        f"- Runtime seconds: {summary.get('runtime_seconds')}",
        "",
        "## Runs",
        "",
        "| Policy | Algorithm | State | Processed regions | Overall WMAPE | Runtime seconds |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for run in summary.get("runs", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run.get("policy", "")),
                    str(run.get("algorithm", "")),
                    str(run.get("state", "")),
                    f"{run.get('processed_regions')}/{run.get('represented_regions')}",
                    "" if run.get("overall_wmape") is None else f"{float(run.get('overall_wmape')):.6g}",
                    f"{float(run.get('runtime_seconds', 0.0)):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Artifacts", "", f"- Summary JSON: `{summary.get('summary_path')}`"])
    for label, path in dict(summary.get("artifacts", {})).items():
        lines.append(f"- {label}: `{path}`")
    return "\n".join(lines) + "\n"


def _partC_rev1_run_one_reload(
    graph: PartCRev1GraphSnapshot,
    *,
    run_entry: dict[str, Any],
    global_policy_summary: dict[str, Any],
    benchmark_flow: np.ndarray,
    benchmark_hash: str,
    config: PartCRev1PhysicalReloadSuiteConfig,
    run_dir: Path,
    reporter: PartCRev1ProgressReporter | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_summary_path = partC_rev1_resolve_path(str(run_entry.get("summary_path", "")))
    run_summary = partC_rev1_read_json(run_summary_path)
    artifacts = dict(run_summary.get("artifacts", {}))
    pairs = _partC_rev1_join_policy_pairs(artifacts["active_pair_history_csv"], artifacts["final_regional_pair_flow_parquet"])
    regions = sorted(str(value) for value in pairs["region_uuid"].dropna().unique().tolist())
    represented_regions = len(regions)
    if config.max_regions and config.max_regions > 0:
        regions = regions[: int(config.max_regions)]
    reload_config = PartCRev1PhysicalReloadConfig(
        config_path=config.config_path,
        max_regions=int(config.max_regions),
        max_iterations=int(config.max_iterations),
        gap_target=float(config.gap_target),
        positive_flow_tol=float(config.positive_flow_tol),
        runtime_guard_seconds=float(config.runtime_guard_seconds),
        correction_enabled=False,
    )
    if reporter is not None:
        reporter.emit(
            "physical_reload_policy_run_started",
            policy=run_entry.get("policy"),
            algorithm=run_entry.get("algorithm"),
            represented_regions=represented_regions,
            processed_regions=len(regions),
            active_pair_count=int(len(pairs)),
            positive_pair_count=int((pairs["flow"] > config.positive_flow_tol).sum()),
        )

    regional_reload_flow = np.zeros(len(graph.arc_idx_to_directed_arc_id), dtype=np.float64)
    region_summaries: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    conservation_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    stop_reason = "completed_regions"
    observed_region_times: list[float] = []
    for region_position, region_uuid in enumerate(regions, start=1):
        region_started = time.perf_counter()
        region_pairs = pairs[pairs["region_uuid"].astype(str) == region_uuid].copy()
        result = partC_rev1_reload_region(graph, region_uuid, region_pairs, reload_config, reporter=reporter)
        regional_reload_flow += result.arc_flow_global
        region_summaries.append(result.summary)
        terminal_rows.extend(result.terminal_cost_rows)
        boundary_rows.extend(result.boundary_rows)
        conservation_rows.extend(result.conservation_rows)
        history_rows.extend(result.history_rows)
        observed_region_times.append(time.perf_counter() - region_started)
        elapsed = time.perf_counter() - started
        if reporter is not None:
            resource = partC_rev1_resource_snapshot(started)
            median_time = float(np.median(observed_region_times))
            reporter.heartbeat(
                "physical_reload_policy_region_finished",
                policy=run_entry.get("policy"),
                algorithm=run_entry.get("algorithm"),
                region_uuid=region_uuid,
                processed=region_position,
                total=len(regions),
                elapsed_seconds=elapsed,
                projected_remaining_seconds=median_time * max(len(regions) - region_position, 0),
                memory_rss_mb=resource.memory_rss_mb,
                temp_disk_mb=resource.temp_disk_mb,
                disk_free_mb=resource.disk_free_mb,
            )
        if elapsed > float(config.runtime_guard_seconds) and region_position < len(regions):
            stop_reason = "runtime_guard"
            break

    backbone_flow = pd.read_parquet(partC_rev1_resolve_path(artifacts["final_backbone_flow_parquet"]))
    final_flow, directed_flow, link_flow = _partC_rev1_merge_policy_physical_flows(graph, backbone_flow, regional_reload_flow)
    directed_flow, link_flow = _partC_rev1_augment_final_tables(graph, directed_flow, link_flow, benchmark_flow)
    validation_metrics = partC_rev1_physical_reload_validation_metrics(graph, final_flow, benchmark_flow)
    regional_summary = pd.DataFrame(region_summaries)
    terminal_comparison = pd.DataFrame(terminal_rows)
    boundary = pd.DataFrame(boundary_rows)
    conservation = pd.DataFrame(conservation_rows)
    history = pd.DataFrame(history_rows)
    runtime_breakdown = pd.DataFrame(
        [
            {"component": "regional_reload", "runtime_seconds": float(sum(row.get("runtime_seconds", 0.0) for row in region_summaries))},
            {"component": "final_merge_and_validation", "runtime_seconds": time.perf_counter() - started},
        ]
    )

    artifacts_out = {
        "final_directed_arc_flow_parquet": str(run_dir / "partC_rev1_final_directed_arc_flow.parquet"),
        "final_physical_link_flow_parquet": str(run_dir / "partC_rev1_final_physical_link_flow.parquet"),
        "regional_reload_summary_csv": str(run_dir / "partC_rev1_regional_reload_summary.csv"),
        "terminal_cost_comparison_parquet": str(run_dir / "partC_rev1_terminal_cost_comparison.parquet"),
        "boundary_consistency_csv": str(run_dir / "partC_rev1_boundary_consistency.csv"),
        "conservation_residuals_csv": str(run_dir / "partC_rev1_conservation_residuals.csv"),
        "runtime_breakdown_csv": str(run_dir / "partC_rev1_runtime_breakdown.csv"),
        "validation_metrics_csv": str(run_dir / "partC_rev1_validation_metrics.csv"),
        "iteration_history_csv": str(run_dir / "partC_rev1_reload_iteration_history.csv"),
        "summary_json": str(run_dir / "partC_rev1_reload_summary.json"),
        "summary_md": str(run_dir / "partC_rev1_reload_summary.md"),
        "run_log": str(run_dir / "partC_rev1_run.log"),
    }
    partC_rev1_write_table(directed_flow, artifacts_out["final_directed_arc_flow_parquet"], index=False)
    partC_rev1_write_table(link_flow, artifacts_out["final_physical_link_flow_parquet"], index=False)
    partC_rev1_write_table(regional_summary, artifacts_out["regional_reload_summary_csv"], index=False)
    partC_rev1_write_table(terminal_comparison, artifacts_out["terminal_cost_comparison_parquet"], index=False)
    partC_rev1_write_table(boundary, artifacts_out["boundary_consistency_csv"], index=False)
    partC_rev1_write_table(conservation, artifacts_out["conservation_residuals_csv"], index=False)
    partC_rev1_write_table(runtime_breakdown, artifacts_out["runtime_breakdown_csv"], index=False)
    partC_rev1_write_table(validation_metrics, artifacts_out["validation_metrics_csv"], index=False)
    partC_rev1_write_table(history, artifacts_out["iteration_history_csv"], index=False)

    max_boundary = float(regional_summary["max_boundary_abs_residual"].max()) if not regional_summary.empty else 0.0
    max_conservation = float(regional_summary["max_conservation_abs_residual"].max()) if not regional_summary.empty else 0.0
    unreachable_pairs = int(regional_summary["unreachable_pair_count"].sum()) if not regional_summary.empty else 0
    positive_pair_count = int((pairs["flow"] > config.positive_flow_tol).sum())
    processed_regions = int(len(region_summaries))
    if positive_pair_count == 0:
        state = "measured_zero_flow_complete" if processed_regions == represented_regions else "measured_zero_flow_partial"
    else:
        state = "measured_complete" if processed_regions == represented_regions else "measured_partial_projected"
    overall_rows = validation_metrics.loc[
        validation_metrics["metric_scope_kind"].astype(str).eq("overall")
        & validation_metrics["metric_scope_value"].astype(str).eq("all")
    ]
    overall_wmape = None if overall_rows.empty else float(overall_rows.iloc[0]["wmape"])
    reload_hash = partC_rev1_json_hash(
        {
            "stage": PARTC_REV1_PHYSICAL_RELOAD_SUITE_STAGE,
            "global_policy_run_hash": global_policy_summary.get("run_hash"),
            "policy": run_entry.get("policy"),
            "algorithm": run_entry.get("algorithm"),
            "policy_hash": run_entry.get("policy_hash"),
            "corrected_graph_hash": graph.graph_hash,
            "selected_4c_benchmark_hash": benchmark_hash,
            "config": asdict(config),
        }
    )[:16]
    summary = {
        "stage": PARTC_REV1_PHYSICAL_RELOAD_SUITE_STAGE,
        "created_utc": _partC_rev1_utc_now(),
        "state": state,
        "stop_reason": stop_reason,
        "reload_hash": reload_hash,
        "run_folder": str(run_dir),
        "policy": str(run_entry.get("policy", "")),
        "algorithm": str(run_entry.get("algorithm", "")),
        "policy_hash": str(run_entry.get("policy_hash", "")),
        "global_policy_summary_path": str(global_policy_summary.get("summary_path", "")),
        "global_policy_run_summary_path": str(run_summary_path),
        "selected_4c_benchmark_hash": benchmark_hash,
        "corrected_graph_hash": graph.graph_hash,
        "policy_graph_hash": run_summary.get("policy_graph_hash"),
        "hybrid_graph_hash": run_summary.get("hybrid_graph_hash"),
        "config": asdict(config),
        "represented_regions": represented_regions,
        "processed_regions": processed_regions,
        "active_pair_count": int(len(pairs)),
        "positive_pair_count": positive_pair_count,
        "positive_flow_unreachable_pair_count": unreachable_pairs,
        "final_directed_arc_count": int(len(directed_flow)),
        "final_physical_link_count": int(len(link_flow)),
        "max_boundary_abs_residual": max_boundary,
        "max_conservation_abs_residual": max_conservation,
        "overall_wmape": overall_wmape,
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": artifacts_out,
        "known_approximations": [
            "Validation compares only against the selected post-capacity Prompt 4C physical benchmark.",
            "Corrected-region labels are used for stratification; the selected 4C benchmark shares directed-arc ordering.",
            "Road-class stratification is represented by capacity-multiplier type because class_code is not stored in graph snapshots.",
        ],
    }
    partC_rev1_write_json(summary, artifacts_out["summary_json"])
    Path(artifacts_out["summary_md"]).write_text(_partC_rev1_summary_md({**summary, "suite_hash": reload_hash, "runs": [summary]}), encoding="utf-8")
    Path(artifacts_out["run_log"]).write_text(
        f"{_partC_rev1_utc_now()} physical reload suite run completed state={state} reload_hash={reload_hash}\n",
        encoding="utf-8",
    )
    return summary


def partC_rev1_run_physical_reload_suite_stage(
    graph: PartCRev1GraphSnapshot,
    *,
    global_policy_summary: dict[str, Any],
    benchmark_summary: dict[str, Any],
    config: PartCRev1PhysicalReloadSuiteConfig,
    output_dir: str | Path,
    reporter: PartCRev1ProgressReporter | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_root = partC_rev1_resolve_path(output_dir)
    best = benchmark_summary.get("best_accuracy_benchmark")
    if not isinstance(best, dict):
        raise RuntimeError("Physical reload suite requires a selected Prompt 4C best_accuracy_benchmark")
    benchmark_hash = str(best.get("run_hash", ""))
    benchmark_flow = partC_rev1_load_benchmark_flow(graph, str(best.get("artifacts", {}).get("directed_arc_flow", "")))
    if str(global_policy_summary.get("corrected_graph_hash")) != graph.graph_hash:
        raise RuntimeError("Global policy suite summary does not match the loaded corrected graph")
    candidate_runs = [dict(run) for run in global_policy_summary.get("runs", []) if isinstance(run, dict)]
    candidate_runs = [
        run
        for run in candidate_runs
        if str(dict(run.get("run", {})).get("state", "")).startswith("measured")
        and str(run.get("summary_path", "")).strip()
    ]
    if config.max_runs and config.max_runs > 0:
        candidate_runs = candidate_runs[: int(config.max_runs)]
    suite_hash = partC_rev1_json_hash(
        {
            "stage": PARTC_REV1_PHYSICAL_RELOAD_SUITE_STAGE,
            "corrected_graph_hash": graph.graph_hash,
            "global_policy_run_hash": global_policy_summary.get("run_hash"),
            "selected_4c_benchmark_hash": benchmark_hash,
            "candidate_runs": [(run.get("policy"), run.get("algorithm"), run.get("policy_hash")) for run in candidate_runs],
            "config": asdict(config),
        }
    )[:16]
    suite_root = out_root / f"partC_rev1_physical_reload_suite_{suite_hash}"
    suite_root.mkdir(parents=True, exist_ok=True)
    if reporter is not None:
        reporter.emit(
            "physical_reload_suite_started",
            suite_hash=suite_hash,
            corrected_graph_hash=graph.graph_hash,
            selected_4c_benchmark_hash=benchmark_hash,
            candidate_run_count=len(candidate_runs),
        )

    run_summaries: list[dict[str, Any]] = []
    for idx, run_entry in enumerate(candidate_runs, start=1):
        safe_policy = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(run_entry.get("policy", "")))
        safe_algorithm = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(run_entry.get("algorithm", "")))
        run_id_hash = partC_rev1_json_hash(
            {
                "suite_hash": suite_hash,
                "policy": run_entry.get("policy"),
                "algorithm": run_entry.get("algorithm"),
                "summary_path": run_entry.get("summary_path"),
            }
        )[:16]
        run_dir = suite_root / f"partC_rev1_reload_{safe_policy}_{safe_algorithm}_hash-{run_id_hash}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "partC_rev1_reload_summary.json"
        if config.resume and summary_path.exists():
            run_summary = partC_rev1_read_json(summary_path)
        else:
            run_summary = _partC_rev1_run_one_reload(
                graph,
                run_entry=run_entry,
                global_policy_summary=global_policy_summary,
                benchmark_flow=benchmark_flow,
                benchmark_hash=benchmark_hash,
                config=config,
                run_dir=run_dir,
                reporter=reporter,
            )
        run_summaries.append(run_summary)
        if reporter is not None:
            resource = partC_rev1_resource_snapshot(started)
            reporter.emit(
                "physical_reload_suite_run_finished",
                processed=idx,
                total=len(candidate_runs),
                policy=run_entry.get("policy"),
                algorithm=run_entry.get("algorithm"),
                state=run_summary.get("state"),
                runtime_seconds=run_summary.get("runtime_seconds"),
                elapsed_seconds=time.perf_counter() - started,
                memory_rss_mb=resource.memory_rss_mb,
                temp_disk_mb=resource.temp_disk_mb,
            )

    run_index = pd.DataFrame(
        [
            {
                "policy": row.get("policy"),
                "algorithm": row.get("algorithm"),
                "state": row.get("state"),
                "stop_reason": row.get("stop_reason"),
                "processed_regions": row.get("processed_regions"),
                "represented_regions": row.get("represented_regions"),
                "positive_pair_count": row.get("positive_pair_count"),
                "positive_flow_unreachable_pair_count": row.get("positive_flow_unreachable_pair_count"),
                "overall_wmape": row.get("overall_wmape"),
                "runtime_seconds": row.get("runtime_seconds"),
                "summary_json": row.get("artifacts", {}).get("summary_json"),
                "run_folder": row.get("run_folder"),
            }
            for row in run_summaries
        ]
    )
    run_index_path = suite_root / "partC_rev1_physical_reload_suite_run_index.csv"
    partC_rev1_write_table(run_index, run_index_path, index=False)
    completed = [row for row in run_summaries if str(row.get("state")) in {"measured_complete", "measured_zero_flow_complete"}]
    state = "measured_complete" if len(completed) == len(candidate_runs) else "measured_partial_projected"
    if not candidate_runs:
        state = "failed_no_completed_global_policy_runs"
    best_reload = None
    if run_summaries:
        best_reload = min(
            run_summaries,
            key=lambda row: (
                math.inf if row.get("overall_wmape") is None else float(row.get("overall_wmape")),
                str(row.get("policy", "")),
                str(row.get("algorithm", "")),
            ),
        )
    summary_path = suite_root / "partC_rev1_physical_reload_suite_summary.json"
    handoff_path = suite_root / "partC_rev1_physical_reload_suite_handoff.md"
    summary = {
        "stage": PARTC_REV1_PHYSICAL_RELOAD_SUITE_STAGE,
        "created_utc": _partC_rev1_utc_now(),
        "state": state,
        "suite_hash": suite_hash,
        "suite_root": str(suite_root),
        "config": asdict(config),
        "corrected_graph_hash": graph.graph_hash,
        "corrected_partition_hash": graph.partition_hash,
        "global_policy_run_hash": global_policy_summary.get("run_hash"),
        "global_policy_summary_path": global_policy_summary.get("summary_path"),
        "selected_4c_benchmark_hash": benchmark_hash,
        "selected_4c_benchmark_summary": {
            "method": best.get("method"),
            "backend": best.get("backend"),
            "final_gap": best.get("final_gap"),
            "runtime_seconds": best.get("runtime_seconds"),
            "directed_arc_flow": best.get("artifacts", {}).get("directed_arc_flow"),
        },
        "candidate_run_count": len(candidate_runs),
        "processed_run_count": len(run_summaries),
        "runtime_seconds": time.perf_counter() - started,
        "best_reload": None
        if best_reload is None
        else {
            "policy": best_reload.get("policy"),
            "algorithm": best_reload.get("algorithm"),
            "overall_wmape": best_reload.get("overall_wmape"),
            "summary_json": best_reload.get("artifacts", {}).get("summary_json"),
            "run_folder": best_reload.get("run_folder"),
        },
        "runs": run_summaries,
        "artifacts": {
            "run_index_csv": str(run_index_path),
            "summary_json": str(summary_path),
            "handoff_md": str(handoff_path),
        },
    }
    summary["summary_path"] = str(summary_path)
    summary["handoff_path"] = str(handoff_path)
    partC_rev1_write_json(summary, summary_path)
    handoff_path.write_text(_partC_rev1_summary_md(summary), encoding="utf-8")
    if reporter is not None:
        reporter.emit("physical_reload_suite_finished", summary=str(summary_path), handoff=str(handoff_path), state=state)
    return summary
