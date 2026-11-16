"""Post-capacity full regional training over corrected regions."""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd

from .partC_rev1_graph_snapshot import PartCRev1GraphSnapshot, partC_rev1_load_graph_snapshot
from .partC_rev1_hashing import partC_rev1_json_hash, partC_rev1_numpy_array_hash
from .partC_rev1_io import partC_rev1_write_json, partC_rev1_write_table
from .partC_rev1_od_snapshot import PartCRev1ODSnapshot
from .partC_rev1_paths import partC_rev1_resolve_path
from .partC_rev1_progress_logging import PartCRev1ProgressReporter
from .partC_rev1_regional_training import (
    PartCRev1RegionalTrainingConfig,
    _partC_rev1_metrics,
    _partC_rev1_split_indices,
    partC_rev1_build_region_problem,
    partC_rev1_centroid_node_mask,
    partC_rev1_generate_region_samples,
    partC_rev1_incident_label_counts,
    partC_rev1_od_marginals_by_node,
    partC_rev1_read_route_witnesses,
    partC_rev1_train_region_models,
)
from .partC_rev1_runtime_guard import partC_rev1_project_runtime, partC_rev1_resource_snapshot


PARTC_REV1_FULL_REGIONAL_STAGE = "partC_rev1_full_regional_training"


@dataclass(frozen=True, slots=True)
class PartCRev1FullRegionalTrainingConfig:
    config_path: str
    min_samples_per_region: int = 100
    max_samples_per_region: int = 1000
    region_limit: int = 0
    max_wall_hours: float = 72.0
    projection_min_completed_regions: int = 3
    max_active_pairs: int = 64
    top_k_per_source: int = 6
    regional_solver_iterations: int = 3
    regional_gap_target: float = 1e-3
    regional_origin_batch_size: int = 8
    xgb_estimators: int = 12
    nn_epochs: int = 24
    icnn_epochs: int = 24
    retune_multiplier: float = 2.0
    random_seed: int = 42
    resume: bool = True
    od_sampler: str = "legacy_sparse_ipf_lognormal"
    support_mask_renormalization: str = "none"
    terminal_perturbation: str = "uniform_0_200"
    skip_on_marginal_imbalance: bool = False


def _partC_rev1_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def partC_rev1_find_latest_small_region_summary(output_dir: str | Path) -> Path:
    out_dir = partC_rev1_resolve_path(output_dir)
    candidates = sorted(
        out_dir.glob("partC_rev1_small_region_correction_*/partC_rev1_small_region_correction_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No partC_rev1_small_region_correction_* summary was found")
    return candidates[0]


def partC_rev1_load_corrected_graph_from_summary(summary_path: str | Path) -> tuple[PartCRev1GraphSnapshot, dict[str, Any]]:
    path = partC_rev1_resolve_path(summary_path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    graph_paths = dict(summary.get("graph_paths", {}))
    graph = partC_rev1_load_graph_snapshot(graph_paths["npz"], graph_paths["metadata"])
    return graph, summary


def partC_rev1_dynamic_region_sample_count(region_row: pd.Series, *, config: PartCRev1FullRegionalTrainingConfig) -> int:
    portal_count = int(region_row.get("portal_count", 0) or 0)
    requested = max(int(config.min_samples_per_region), min(portal_count * portal_count, int(config.max_samples_per_region)))
    return int(max(1, requested))


def _partC_rev1_training_cfg(config: PartCRev1FullRegionalTrainingConfig) -> PartCRev1RegionalTrainingConfig:
    return PartCRev1RegionalTrainingConfig(
        config_path=config.config_path,
        requested_samples_per_region=config.min_samples_per_region,
        pilot_samples_per_region=config.min_samples_per_region,
        pilot_region_count=0,
        production_region_limit=config.region_limit,
        max_active_pairs=config.max_active_pairs,
        top_k_per_source=config.top_k_per_source,
        regional_solver_iterations=config.regional_solver_iterations,
        regional_gap_target=config.regional_gap_target,
        regional_origin_batch_size=config.regional_origin_batch_size,
        xgb_estimators=config.xgb_estimators,
        nn_epochs=config.nn_epochs,
        icnn_epochs=config.icnn_epochs,
        random_seed=config.random_seed,
        od_sampler=config.od_sampler,
        support_mask_renormalization=config.support_mask_renormalization,
        terminal_perturbation=config.terminal_perturbation,
        skip_on_marginal_imbalance=config.skip_on_marginal_imbalance,
    )


def _partC_rev1_fftt_const_row(dataset: Any, output_root: Path) -> dict[str, Any]:
    splits = _partC_rev1_split_indices(len(dataset.x))
    test_idx = splits["test"]
    pred = np.tile(dataset.tau0.astype(np.float32), (len(test_idx), 1))
    truth = dataset.y[test_idx]
    metrics = _partC_rev1_metrics(truth, pred)
    out_dir = output_root / "partC_rev1_4D_models" / dataset.problem.region_uuid
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "partC_rev1_model_fftt_const_approximation.json"
    payload = {
        "model": "fftt_const_approximation",
        "region_uuid": dataset.problem.region_uuid,
        "pair_count": int(len(dataset.tau0)),
        "tau0": dataset.tau0.astype(float).tolist(),
        "metadata": dataset.metadata,
    }
    partC_rev1_write_json(payload, artifact_path)
    return {
        "region_uuid": dataset.problem.region_uuid,
        "model": "fftt_const_approximation",
        "storage": "json",
        "status": "verified",
        "artifact_path": str(artifact_path),
        "region_hash": dataset.problem.region_hash,
        "terminal_order_hash": dataset.problem.terminal_order_hash,
        "pair_order_hash": dataset.problem.pair_order_hash,
        "training_data_hash": dataset.metadata["training_data_hash"],
        "sample_count": int(len(dataset.x)),
        "pair_count": int(dataset.x.shape[1]),
        "train_count": int(len(splits["train"])),
        "validation_count": int(len(splits["validation"])),
        "test_count": int(len(test_idx)),
        "runtime_seconds": 0.0,
        "parameter_count": 0,
        "file_size_bytes": int(artifact_path.stat().st_size),
        "freeflow_lower_bound_violation_rate": 0.0,
        **metrics,
        "monotonicity_violation_rate": 0.0,
        "min_monotonicity_dot": 0.0,
    }


def _partC_rev1_select_best_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [row for row in rows if str(row.get("status")) == "verified"]
    if not verified:
        return {"model": "", "selection_status": "no_verified_model"}
    return sorted(
        verified,
        key=lambda row: (
            float(row.get("wmape", math.inf) if row.get("wmape") is not None else math.inf),
            float(row.get("rmse", math.inf) if row.get("rmse") is not None else math.inf),
            str(row.get("model", "")),
        ),
    )[0]


def _partC_rev1_write_registry(rows: list[dict[str, Any]], path: Path) -> tuple[Path, Path]:
    registry = pd.DataFrame(rows)
    partC_rev1_write_table(registry, path, index=False)
    md_path = path.with_suffix(".md")
    lines = [
        "# partC_rev1 Full Regional Training Model Registry",
        "",
        f"Generated UTC: {_partC_rev1_utc_now()}",
        "",
        "| Region | Model | Status | Samples | Pairs | WMAPE | Selected |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("region_uuid", "")),
                    str(row.get("model", "")),
                    str(row.get("status", "")),
                    str(row.get("sample_count", "")),
                    str(row.get("pair_count", "")),
                    "" if row.get("wmape") is None else f"{float(row.get('wmape')):.6g}",
                    "yes" if bool(row.get("selected_for_region")) else "no",
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, md_path


def _partC_rev1_write_compact_md_table(df: pd.DataFrame, path: Path, *, title: str, max_rows: int = 50) -> Path:
    columns = [str(col) for col in df.columns.tolist()]
    lines = [f"# {title}", "", f"Generated UTC: {_partC_rev1_utc_now()}", ""]
    if not columns:
        lines.extend(["No rows were available.", ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
    shown = df.head(max_rows).copy()
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    if len(df) > len(shown):
        lines.extend(["", f"Showing first {len(shown)} of {len(df)} rows."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _partC_rev1_read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _partC_rev1_write_contract_artifacts(
    *,
    run_root: Path,
    summary: dict[str, Any],
    status_df: pd.DataFrame,
    registry_path: Path,
    selected_path: Path,
) -> dict[str, str]:
    registry_df = _partC_rev1_read_csv_if_exists(registry_path)
    selected_df = _partC_rev1_read_csv_if_exists(selected_path)

    canonical_status_path = run_root / "partC_rev1_region_status.csv"
    canonical_model_registry_path = run_root / "partC_rev1_model_registry.csv"
    canonical_model_registry_md = run_root / "partC_rev1_model_registry.md"
    training_registry_path = run_root / "partC_rev1_training_registry.csv"
    training_registry_md = run_root / "partC_rev1_training_registry.md"
    selected_by_region_path = run_root / "partC_rev1_selected_models_by_region.csv"
    sample_metadata_path = run_root / "partC_rev1_region_sample_metadata.parquet"
    runtime_by_region_path = run_root / "partC_rev1_training_runtime_by_region.csv"
    learning_curve_path = run_root / "partC_rev1_learning_curve_summary.csv"
    exemplar_path = run_root / "partC_rev1_exemplar_region_list.csv"

    partC_rev1_write_table(status_df, canonical_status_path, index=False)
    partC_rev1_write_table(registry_df, canonical_model_registry_path, index=False)
    _partC_rev1_write_compact_md_table(registry_df, canonical_model_registry_md, title="partC_rev1 Model Registry")
    partC_rev1_write_table(selected_df, selected_by_region_path, index=False)

    training_columns = [
        "region_uuid",
        "status",
        "reason",
        "sample_count",
        "pair_count",
        "selected_model",
        "selected_wmape",
        "fftt_const_wmape",
        "retune_status",
        "runtime_seconds",
        "region_hash",
        "terminal_order_hash",
        "pair_order_hash",
    ]
    training_registry = status_df.reindex(columns=[col for col in training_columns if col in status_df.columns]).copy()
    training_registry["od_sampler"] = summary["config"].get("od_sampler")
    training_registry["support_mask_renormalization"] = summary["config"].get("support_mask_renormalization")
    training_registry["terminal_perturbation"] = summary["config"].get("terminal_perturbation")
    training_registry["assignment_graph_hash"] = summary.get("corrected_graph_hash")
    training_registry["partition_hash"] = summary.get("corrected_partition_hash")
    training_registry["correction_hash"] = summary.get("correction_hash")
    partC_rev1_write_table(training_registry, training_registry_path, index=False)
    _partC_rev1_write_compact_md_table(training_registry, training_registry_md, title="partC_rev1 Training Registry")

    sample_columns = [
        "region_uuid",
        "status",
        "reason",
        "sample_count",
        "pair_count",
        "region_hash",
        "terminal_order_hash",
        "pair_order_hash",
    ]
    sample_metadata = status_df.reindex(columns=[col for col in sample_columns if col in status_df.columns]).copy()
    sample_metadata["od_sampler"] = summary["config"].get("od_sampler")
    sample_metadata["support_mask_renormalization"] = summary["config"].get("support_mask_renormalization")
    sample_metadata["terminal_perturbation"] = summary["config"].get("terminal_perturbation")
    partC_rev1_write_table(sample_metadata, sample_metadata_path, index=False)

    runtime_columns = ["region_uuid", "status", "reason", "runtime_seconds", "sample_count", "pair_count"]
    runtime_by_region = status_df.reindex(columns=[col for col in runtime_columns if col in status_df.columns]).copy()
    partC_rev1_write_table(runtime_by_region, runtime_by_region_path, index=False)

    if registry_df.empty:
        learning_curve = pd.DataFrame(
            columns=[
                "region_uuid",
                "model",
                "curve_type",
                "sample_count",
                "train_count",
                "validation_count",
                "test_count",
                "wmape",
                "rmse",
                "mae",
                "status",
            ]
        )
    else:
        learning_columns = [
            "region_uuid",
            "model",
            "sample_count",
            "train_count",
            "validation_count",
            "test_count",
            "wmape",
            "rmse",
            "mae",
            "status",
        ]
        learning_curve = registry_df.reindex(columns=[col for col in learning_columns if col in registry_df.columns]).copy()
        learning_curve.insert(2, "curve_type", "final_holdout_metrics")
    partC_rev1_write_table(learning_curve, learning_curve_path, index=False)

    exemplar_frames: list[pd.DataFrame] = []
    if not status_df.empty and "runtime_seconds" in status_df.columns:
        exemplar_frames.append(status_df.assign(exemplar_reason="largest_runtime").sort_values("runtime_seconds", ascending=False, kind="mergesort").head(5))
    if not status_df.empty and "pair_count" in status_df.columns:
        exemplar_frames.append(status_df.assign(exemplar_reason="largest_pair_count").sort_values("pair_count", ascending=False, kind="mergesort").head(5))
    if not status_df.empty and "selected_wmape" in status_df.columns:
        ranked_wmape = status_df.copy()
        ranked_wmape["_selected_wmape_rank"] = pd.to_numeric(ranked_wmape["selected_wmape"], errors="coerce").fillna(-math.inf)
        exemplar_frames.append(ranked_wmape.assign(exemplar_reason="largest_selected_wmape").sort_values("_selected_wmape_rank", ascending=False, kind="mergesort").head(5))
    exemplar = pd.concat(exemplar_frames, ignore_index=True) if exemplar_frames else pd.DataFrame()
    if not exemplar.empty:
        exemplar = exemplar.drop(columns=[col for col in ["_selected_wmape_rank"] if col in exemplar.columns])
        exemplar = exemplar.drop_duplicates(subset=["region_uuid", "exemplar_reason"], keep="first")
    partC_rev1_write_table(exemplar, exemplar_path, index=False)

    handoff_path = run_root / "partC_rev1_4D_full_handoff.md"
    handoff_lines = [
        "# partC_rev1 4D Full Regional Training Handoff",
        "",
        f"Generated UTC: {_partC_rev1_utc_now()}",
        "",
        f"- State: `{summary['state']}`",
        f"- Stop reason: `{summary['stop_reason']}`",
        f"- Run hash: `{summary['run_hash']}`",
        f"- Corrected assignment graph hash: `{summary['corrected_graph_hash']}`",
        f"- Corrected partition hash: `{summary['corrected_partition_hash']}`",
        f"- OD hash: `{summary['od_hash']}`",
        f"- Candidate regions: {summary['candidate_region_count']}",
        f"- Completed regions: {summary['completed_region_count']}",
        f"- Trained regions: {summary['trained_region_count']}",
        f"- Skipped regions: {summary['skipped_region_count']}",
        f"- Failed regions: {summary['failed_region_count']}",
        f"- Runtime seconds: {summary['runtime_seconds']}",
        "",
        "## Canonical Artifacts",
        "",
    ]
    artifact_preview = {
        "training_registry_csv": str(training_registry_path),
        "training_registry_md": str(training_registry_md),
        "model_registry_csv": str(canonical_model_registry_path),
        "model_registry_md": str(canonical_model_registry_md),
        "region_status_csv": str(canonical_status_path),
        "region_sample_metadata_parquet": str(sample_metadata_path),
        "selected_models_by_region_csv": str(selected_by_region_path),
        "training_runtime_by_region_csv": str(runtime_by_region_path),
        "learning_curve_summary_csv": str(learning_curve_path),
        "exemplar_region_list_csv": str(exemplar_path),
    }
    handoff_lines.extend(f"- {label}: `{path}`" for label, path in artifact_preview.items())
    handoff_path.write_text("\n".join(handoff_lines) + "\n", encoding="utf-8")
    artifact_preview["handoff_md"] = str(handoff_path)
    return artifact_preview


def _partC_rev1_write_runtime_guard_report(*, run_root: Path, summary: dict[str, Any], status_df: pd.DataFrame) -> Path | None:
    if summary.get("state") == "measured_complete":
        return None
    if str(summary.get("stop_reason", "")) not in {"projected_runtime_guard", "wall_time_limit"}:
        return None
    path = run_root / "partC_rev1_4D_runtime_guard_blocked.md"
    runtimes = pd.to_numeric(status_df.get("runtime_seconds", pd.Series(dtype=float)), errors="coerce").dropna()
    trained = status_df.loc[status_df.get("status", pd.Series(dtype=str)).astype(str).eq("trained")].copy() if not status_df.empty else pd.DataFrame()
    lines = [
        "# partC_rev1 4D Runtime Guard Blocked Report",
        "",
        f"Generated UTC: {_partC_rev1_utc_now()}",
        "",
        "## Status",
        "",
        f"- State: `{summary.get('state')}`",
        f"- Stop reason: `{summary.get('stop_reason')}`",
        f"- Run hash: `{summary.get('run_hash')}`",
        f"- Corrected graph hash: `{summary.get('corrected_graph_hash')}`",
        f"- Corrected partition hash: `{summary.get('corrected_partition_hash')}`",
        f"- Candidate regions: {summary.get('candidate_region_count')}",
        f"- Completed regions: {summary.get('completed_region_count')}",
        f"- Trained regions: {summary.get('trained_region_count')}",
        f"- Skipped regions: {summary.get('skipped_region_count')}",
        f"- Failed regions: {summary.get('failed_region_count')}",
        f"- Runtime seconds: {summary.get('runtime_seconds')}",
        f"- Projected total seconds: {summary.get('projected_total_seconds')}",
        f"- Projected remaining seconds: {summary.get('projected_remaining_seconds')}",
        "",
        "## Runtime Evidence",
        "",
        f"- Observed region runtime rows: {int(len(runtimes))}",
        f"- Median observed region seconds: {float(runtimes.median()) if not runtimes.empty else ''}",
        f"- P95 observed region seconds: {float(runtimes.quantile(0.95)) if not runtimes.empty else ''}",
        f"- Maximum observed region seconds: {float(runtimes.max()) if not runtimes.empty else ''}",
        "",
        "## Required Next Decision",
        "",
        "The prompt requires user approval before reducing the 4D scope. Resume with the same command if more wall time is allowed; use a smaller or staged regional scope only after explicit approval.",
        "",
        "## Resume Command",
        "",
        "```text",
        f"conda run -n py312 python -m partC_rev1_src.partC_rev1_cli run-full-regional-training --config {summary.get('config', {}).get('config_path', '')} --confirm-full-run --max-wall-hours {summary.get('config', {}).get('max_wall_hours', 72.0)} --heartbeat-seconds 15 --validate-dependency-hashes",
        "```",
    ]
    if not trained.empty:
        preview_cols = [col for col in ["region_uuid", "runtime_seconds", "sample_count", "pair_count", "selected_model", "selected_wmape"] if col in trained.columns]
        lines.extend(["", "## Slowest Completed Regions", "", "| " + " | ".join(preview_cols) + " |", "| " + " | ".join("---" for _ in preview_cols) + " |"])
        slowest = trained.sort_values("runtime_seconds", ascending=False, kind="mergesort").head(10)
        for row in slowest[preview_cols].itertuples(index=False):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def partC_rev1_run_full_regional_training_stage(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    *,
    correction_summary: dict[str, Any],
    benchmark_flow: np.ndarray,
    route_witnesses: pd.DataFrame,
    config: PartCRev1FullRegionalTrainingConfig,
    output_dir: str | Path,
    cache_dir: str | Path,
    reporter: PartCRev1ProgressReporter | None = None,
) -> dict[str, Any]:
    started_perf = time.perf_counter()
    out_root = partC_rev1_resolve_path(output_dir)
    cache_root = partC_rev1_resolve_path(cache_dir)
    run_hash = partC_rev1_json_hash(
        {
            "stage": PARTC_REV1_FULL_REGIONAL_STAGE,
            "corrected_graph_hash": graph.graph_hash,
            "od_hash": od.od_hash,
            "correction_hash": correction_summary.get("correction_hash"),
            "config": asdict(config),
        }
    )[:16]
    run_root = out_root / f"partC_rev1_full_regional_training_{run_hash}"
    temp_root = cache_root / f"partC_rev1_full_regional_training_{run_hash}" / "partC_rev1_training_tmp"
    run_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    terminal_registry_path = partC_rev1_resolve_path(correction_summary["artifacts"]["corrected_terminal_registry_csv"])
    region_table = pd.read_csv(terminal_registry_path)
    candidates = region_table.loc[region_table["terminal_proxy_count"].astype(float).gt(1)].copy()
    candidates = candidates.sort_values(["arc_count", "region_uuid"], kind="mergesort").reset_index(drop=True)
    if config.region_limit and config.region_limit > 0:
        candidates = candidates.head(int(config.region_limit)).copy()

    status_path = run_root / "partC_rev1_full_regional_training_region_status.csv"
    registry_path = run_root / "partC_rev1_full_regional_training_model_registry.csv"
    selected_path = run_root / "partC_rev1_full_regional_training_selected_models.csv"
    if config.resume and status_path.exists():
        existing_status = pd.read_csv(status_path)
        completed = set(existing_status.loc[existing_status["status"].isin(["trained", "skipped"]), "region_uuid"].astype(str))
        status_rows: list[dict[str, Any]] = existing_status.loc[existing_status["status"].isin(["trained", "skipped"])].to_dict("records")
    else:
        existing_status = pd.DataFrame()
        completed = set()
        status_rows = []

    regional_cfg = _partC_rev1_training_cfg(config)
    incident_counts = partC_rev1_incident_label_counts(graph)
    centroid_mask = partC_rev1_centroid_node_mask(graph, od)
    od_source, od_target = partC_rev1_od_marginals_by_node(graph, od)
    model_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    region_runtimes: list[float] = []
    stop_reason = "completed_all_candidates"
    max_run_seconds = float(config.max_wall_hours) * 3600.0

    for position, region_row in enumerate(candidates.itertuples(index=False), start=1):
        region_uuid = str(region_row.region_uuid)
        if region_uuid in completed:
            continue
        region_started = time.perf_counter()
        sample_count = partC_rev1_dynamic_region_sample_count(pd.Series(region_row._asdict()), config=config)
        if reporter is not None:
            reporter.emit("full_regional_training_region_started", region_uuid=region_uuid, position=position, total=len(candidates), sample_count=sample_count)
        try:
            problem = partC_rev1_build_region_problem(
                graph,
                od,
                region_uuid=region_uuid,
                benchmark_flow=benchmark_flow,
                config=regional_cfg,
                incident_label_count=incident_counts,
                centroid_mask=centroid_mask,
                od_source_marginal=od_source,
                od_target_marginal=od_target,
                route_witnesses=route_witnesses,
            )
            if problem.status != "ok":
                status_rows.append(
                    {
                        "region_uuid": region_uuid,
                        "status": "skipped",
                        "reason": problem.status,
                        "sample_count": sample_count,
                        "pair_count": int(len(problem.support)),
                        "runtime_seconds": time.perf_counter() - region_started,
                    }
                )
            else:
                dataset = partC_rev1_generate_region_samples(problem, config=regional_cfg, sample_count=sample_count, temp_dir=temp_root, reporter=reporter)
                region_model_rows = [_partC_rev1_fftt_const_row(dataset, run_root)]
                region_model_rows.extend(partC_rev1_train_region_models(dataset, output_dir=run_root, config=regional_cfg, reporter=reporter))
                baseline = next(row for row in region_model_rows if row["model"] == "fftt_const_approximation")
                best = _partC_rev1_select_best_model(region_model_rows)
                threshold = min(0.001, float(baseline.get("wmape", math.inf) if baseline.get("wmape") is not None else math.inf))
                retune_status = "not_needed"
                if float(best.get("wmape", math.inf) if best.get("wmape") is not None else math.inf) > threshold:
                    retune_status = "attempted"
                    multiplier = max(float(config.retune_multiplier), 1.0)
                    retune_cfg = PartCRev1RegionalTrainingConfig(
                        **{
                            **asdict(regional_cfg),
                            "xgb_estimators": int(math.ceil(regional_cfg.xgb_estimators * multiplier)),
                            "nn_epochs": int(math.ceil(regional_cfg.nn_epochs * multiplier)),
                            "icnn_epochs": int(math.ceil(regional_cfg.icnn_epochs * multiplier)),
                        }
                    )
                    retuned_rows = partC_rev1_train_region_models(dataset, output_dir=run_root / "partC_rev1_retuned", config=retune_cfg, reporter=reporter)
                    for row in retuned_rows:
                        row["model"] = f"retuned_{row['model']}"
                    region_model_rows.extend(retuned_rows)
                    best = _partC_rev1_select_best_model(region_model_rows)
                for row in region_model_rows:
                    row["selected_for_region"] = row.get("model") == best.get("model")
                    row["selection_threshold_wmape"] = threshold
                    row["retune_status"] = retune_status
                model_rows.extend(region_model_rows)
                selected_rows.append({**best, "selection_threshold_wmape": threshold, "retune_status": retune_status})
                if dataset.temp_path.exists() and all(str(row.get("status")) == "verified" for row in region_model_rows):
                    dataset.temp_path.unlink()
                runtime = time.perf_counter() - region_started
                region_runtimes.append(runtime)
                status_rows.append(
                    {
                        "region_uuid": region_uuid,
                        "status": "trained",
                        "reason": "ok",
                        "sample_count": sample_count,
                        "pair_count": int(len(problem.support)),
                        "selected_model": best.get("model"),
                        "selected_wmape": best.get("wmape"),
                        "fftt_const_wmape": baseline.get("wmape"),
                        "retune_status": retune_status,
                        "runtime_seconds": runtime,
                        "region_hash": problem.region_hash,
                        "terminal_order_hash": problem.terminal_order_hash,
                        "pair_order_hash": problem.pair_order_hash,
                    }
                )
        except Exception as exc:
            reason = repr(exc)
            if isinstance(exc, RuntimeError) and "IPF infeasible" in reason:
                status_rows.append(
                    {
                        "region_uuid": region_uuid,
                        "status": "skipped",
                        "reason": f"skipped_infeasible_sparse_support: {reason}",
                        "sample_count": sample_count,
                        "runtime_seconds": time.perf_counter() - region_started,
                    }
                )
                if reporter is not None:
                    reporter.emit("full_regional_training_region_skipped", region_uuid=region_uuid, reason="skipped_infeasible_sparse_support")
            else:
                status_rows.append(
                    {
                        "region_uuid": region_uuid,
                        "status": "failed",
                        "reason": reason,
                        "sample_count": sample_count,
                        "runtime_seconds": time.perf_counter() - region_started,
                    }
                )
                if reporter is not None:
                    reporter.emit("full_regional_training_region_failed", region_uuid=region_uuid, error=repr(exc))
        finally:
            gc.collect()

        partC_rev1_write_table(pd.DataFrame(status_rows), status_path, index=False)
        if model_rows:
            _partC_rev1_write_registry(model_rows, registry_path)
        if selected_rows:
            partC_rev1_write_table(pd.DataFrame(selected_rows), selected_path, index=False)
        trained_count = sum(1 for row in status_rows if row.get("status") == "trained")
        projection = partC_rev1_project_runtime(
            completed=max(trained_count, len(region_runtimes)),
            total=int(len(candidates)),
            started_perf=started_perf,
            completed_unit_seconds=region_runtimes,
            max_run_seconds=max_run_seconds,
        )
        if reporter is not None:
            resource = partC_rev1_resource_snapshot(started_perf)
            reporter.heartbeat(
                "full_regional_training_checkpoint",
                force=True,
                processed=len(status_rows),
                total=len(candidates),
                trained_count=trained_count,
                projected_total_seconds=projection.projected_total_seconds,
                projected_remaining_seconds=projection.projected_remaining_seconds,
                memory_rss_mb=resource.memory_rss_mb,
                disk_free_mb=resource.disk_free_mb,
                temp_disk_mb=resource.temp_disk_mb,
            )
        if len(region_runtimes) >= int(config.projection_min_completed_regions) and projection.should_stop:
            stop_reason = projection.stop_reason or "projected_runtime_guard"
            break
        if time.perf_counter() - started_perf > max_run_seconds:
            stop_reason = "wall_time_limit"
            break

    trained_count = sum(1 for row in status_rows if row.get("status") == "trained")
    failed_count = sum(1 for row in status_rows if row.get("status") == "failed")
    skipped_count = sum(1 for row in status_rows if row.get("status") == "skipped")
    completed_count = trained_count + failed_count + skipped_count
    status_df = pd.DataFrame(status_rows)
    summed_region_runtime_seconds = (
        float(pd.to_numeric(status_df.get("runtime_seconds", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        if not status_df.empty
        else 0.0
    )
    state = "measured_complete" if completed_count >= len(candidates) and failed_count == 0 else "measured_partial_projected"
    if failed_count and completed_count >= len(candidates):
        state = "measured_complete_with_failures"
    projection = partC_rev1_project_runtime(
        completed=max(trained_count, len(region_runtimes)),
        total=int(len(candidates)),
        started_perf=started_perf,
        completed_unit_seconds=region_runtimes,
        max_run_seconds=max_run_seconds,
    )
    summary = {
        "stage": PARTC_REV1_FULL_REGIONAL_STAGE,
        "created_utc": _partC_rev1_utc_now(),
        "state": state,
        "stop_reason": stop_reason,
        "run_hash": run_hash,
        "run_root": str(run_root),
        "temp_root": str(temp_root),
        "config": asdict(config),
        "corrected_graph_hash": graph.graph_hash,
        "corrected_partition_hash": graph.partition_hash,
        "od_hash": od.od_hash,
        "correction_hash": correction_summary.get("correction_hash"),
        "candidate_region_count": int(len(candidates)),
        "completed_region_count": int(completed_count),
        "trained_region_count": int(trained_count),
        "skipped_region_count": int(skipped_count),
        "failed_region_count": int(failed_count),
        "median_trained_region_seconds": float(median(region_runtimes)) if region_runtimes else None,
        "projected_total_seconds": projection.projected_total_seconds,
        "projected_remaining_seconds": projection.projected_remaining_seconds,
        "runtime_seconds": time.perf_counter() - started_perf,
        "summed_region_runtime_seconds": summed_region_runtime_seconds,
        "model_registry_rows": int(len(pd.read_csv(registry_path))) if registry_path.exists() else int(len(model_rows)),
        "selected_model_rows": int(len(pd.read_csv(selected_path))) if selected_path.exists() else int(len(selected_rows)),
        "artifacts": {
            "status_csv": str(status_path),
            "model_registry_csv": str(registry_path),
            "model_registry_md": str(registry_path.with_suffix(".md")),
            "selected_models_csv": str(selected_path),
        },
    }
    summary["artifacts"].update(
        _partC_rev1_write_contract_artifacts(
            run_root=run_root,
            summary=summary,
            status_df=status_df,
            registry_path=registry_path,
            selected_path=selected_path,
        )
    )
    blocked_report = _partC_rev1_write_runtime_guard_report(run_root=run_root, summary=summary, status_df=status_df)
    if blocked_report is not None:
        summary["artifacts"]["runtime_guard_blocked_md"] = str(blocked_report)
    summary_path = run_root / "partC_rev1_full_regional_training_summary.json"
    summary_md = run_root / "partC_rev1_full_regional_training_summary.md"
    partC_rev1_write_json(summary, summary_path)
    lines = [
        "# partC_rev1 Full Regional Training Summary",
        "",
        f"Generated UTC: {summary['created_utc']}",
        "",
        f"- State: `{state}`",
        f"- Stop reason: `{stop_reason}`",
        f"- Run hash: `{run_hash}`",
        f"- Corrected graph hash: `{graph.graph_hash}`",
        f"- Candidate regions: {len(candidates)}",
        f"- Trained regions: {trained_count}",
        f"- Skipped regions: {skipped_count}",
        f"- Failed regions: {failed_count}",
        f"- Projected total seconds: {summary['projected_total_seconds']}",
        f"- Current command runtime seconds: {summary['runtime_seconds']}",
        f"- Summed per-region runtime seconds: {summary['summed_region_runtime_seconds']}",
        "",
        "## Artifacts",
        "",
    ]
    for label, path in summary["artifacts"].items():
        lines.append(f"- {label}: `{path}`")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    summary["summary_md_path"] = str(summary_md)
    partC_rev1_write_json(summary, summary_path)
    return summary
