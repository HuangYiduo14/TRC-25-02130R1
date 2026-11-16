"""Postcapacity revised reviewer experiment suite.

This module reads completed postcapacity artifacts and builds reviewer-facing
tables/figures without mutating Prompt-4 benchmark, training, solver, or reload
outputs. Expensive requested reruns are represented as skipped/projected rows
unless their immutable artifacts already exist.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .partC_rev1_hashing import partC_rev1_json_hash
from .partC_rev1_io import partC_rev1_read_json, partC_rev1_write_json, partC_rev1_write_table
from .partC_rev1_partition_sweep_guard import partC_rev1_find_latest_partition_sensitivity_summary
from .partC_rev1_paths import partC_rev1_resolve_path
from .partC_rev1_progress_logging import PartCRev1ProgressReporter


PARTC_REV1_REVISED_EXPERIMENT_STAGE = "partC_rev1_revised_reviewer_experiments"
PARTC_REV1_REVISED_EXPERIMENTS = [
    ("E1", "Solver/backend runtime summary"),
    ("E2", "Global flow accuracy versus best physical benchmark"),
    ("E3", "Skipped by continuation prompt"),
    ("E4", "Dynamic-sample learning and global-flow propagation"),
    ("E5", "Monotonicity, derivative, and convergence diagnostics"),
    ("E6", "National error propagation"),
    ("E7", "Lazy-update parameter ablation"),
    ("E8", "Adaptive Leiden partition scan"),
    ("E9", "Adaptive Leiden downstream evaluation"),
    ("E10", "Boundary and path consistency"),
    ("E11", "Offline cost and break-even"),
    ("E12", "Skipped by continuation prompt"),
    ("E13", "I-95 scenario demonstration"),
]


@dataclass(frozen=True, slots=True)
class PartCRev1RevisedReviewerExperimentConfig:
    config_path: str
    max_scatter_points: int = 20000
    bootstrap_seed: int = 42
    resume: bool = True


def _partC_rev1_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def partC_rev1_find_latest_physical_reload_suite_summary(output_dir: str | Path) -> Path:
    out_dir = partC_rev1_resolve_path(output_dir)
    candidates = sorted(
        out_dir.glob("partC_rev1_physical_reload_suite_*/partC_rev1_physical_reload_suite_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No partC_rev1_physical_reload_suite_* summary was found")
    return candidates[0]


def partC_rev1_find_latest_full_training_summary(output_dir: str | Path) -> Path:
    out_dir = partC_rev1_resolve_path(output_dir)
    candidates = sorted(
        out_dir.glob("partC_rev1_full_regional_training_*/partC_rev1_full_regional_training_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No partC_rev1_full_regional_training_* summary was found")
    return candidates[0]


def partC_rev1_nested_fraction_counts(total: int, fractions: list[float] | tuple[float, ...]) -> list[int]:
    counts: list[int] = []
    for fraction in fractions:
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError(f"Sample fraction must be in (0, 1]: {fraction}")
        counts.append(int(min(total, max(10, math.ceil(float(fraction) * int(total))))))
    if counts != sorted(counts):
        raise ValueError("Nested sample fractions must produce nondecreasing counts")
    return counts


def partC_rev1_break_even_rows(*, offline_seconds: float, full_seconds: float, hybrid_online_seconds: float, reload_seconds: float) -> pd.DataFrame:
    eps = 1e-12
    return pd.DataFrame(
        [
            {
                "formula": "online_only",
                "offline_seconds": float(offline_seconds),
                "full_physical_seconds": float(full_seconds),
                "hybrid_online_seconds": float(hybrid_online_seconds),
                "reload_seconds": 0.0,
                "n_break_even": float(offline_seconds) / max(float(full_seconds) - float(hybrid_online_seconds), eps),
            },
            {
                "formula": "online_plus_reload",
                "offline_seconds": float(offline_seconds),
                "full_physical_seconds": float(full_seconds),
                "hybrid_online_seconds": float(hybrid_online_seconds),
                "reload_seconds": float(reload_seconds),
                "n_break_even": float(offline_seconds) / max(float(full_seconds) - float(hybrid_online_seconds) - float(reload_seconds), eps),
            },
        ]
    )


def _partC_rev1_read_csv(path: str | Path) -> pd.DataFrame:
    if path is None or str(path).strip() == "":
        return pd.DataFrame()
    resolved = partC_rev1_resolve_path(path)
    return pd.read_csv(resolved) if resolved.exists() else pd.DataFrame()


def _partC_rev1_setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _partC_rev1_save_figures(root: Path, stem: str, table: pd.DataFrame, caption: str, plotter: Any) -> dict[str, str]:
    plt = _partC_rev1_setup_matplotlib()
    fig_dir = root / "partC_rev1_figures"
    data_dir = root / "partC_rev1_underlying_data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    plotter(ax)
    svg = fig_dir / f"{stem}.svg"
    png = fig_dir / f"{stem}.png"
    pdf = fig_dir / f"{stem}.pdf"
    table_path = data_dir / f"{stem}_underlying.csv"
    caption_path = data_dir / f"{stem}_caption.json"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    partC_rev1_write_table(table, table_path, index=False)
    partC_rev1_write_json({"caption": caption, "created_utc": _partC_rev1_utc_now()}, caption_path)
    return {"svg": str(svg), "png": str(png), "pdf": str(pdf), "underlying_csv": str(table_path), "caption_json": str(caption_path)}


def _partC_rev1_global_policy_runtime_rows(global_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in global_summary.get("runs", []):
        if not isinstance(run, dict):
            continue
        run_state = dict(run.get("run", {}))
        rows.append(
            {
                "source": "global_policy",
                "policy": run.get("policy"),
                "method": run.get("algorithm"),
                "backend": "sparse_hybrid",
                "state": run_state.get("state"),
                "stop_reason": run_state.get("stop_reason"),
                "iterations": run_state.get("iteration_count"),
                "final_gap": run_state.get("official_full_refresh_gap"),
                "pricing_gap": run_state.get("pricing_gap"),
                "runtime_seconds": run_state.get("runtime_seconds"),
                "projected_seconds": run_state.get("projected_full_origin_seconds"),
                "processed_origins": run_state.get("processed_origins"),
                "total_origins": run_state.get("total_origins"),
            }
        )
    return rows


def _partC_rev1_solver_runtime_table(summary_4c: dict[str, Any], global_summary: dict[str, Any], reload_summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in summary_4c.get("run_summaries", []):
        if not isinstance(run, dict):
            continue
        rows.append(
            {
                "source": "physical_4c",
                "policy": "physical",
                "method": run.get("method"),
                "backend": run.get("backend"),
                "state": run.get("state"),
                "stop_reason": run.get("stop_reason"),
                "iterations": run.get("iteration_count"),
                "final_gap": run.get("final_gap"),
                "pricing_gap": np.nan,
                "runtime_seconds": run.get("runtime_seconds"),
                "projected_seconds": run.get("projected_target_seconds"),
                "processed_origins": run.get("validation", {}).get("total_processed_demand"),
                "total_origins": np.nan,
            }
        )
    rows.extend(_partC_rev1_global_policy_runtime_rows(global_summary))
    for run in reload_summary.get("runs", []):
        rows.append(
            {
                "source": "physical_reload",
                "policy": run.get("policy"),
                "method": run.get("algorithm"),
                "backend": "regional_reload",
                "state": run.get("state"),
                "stop_reason": run.get("stop_reason"),
                "iterations": np.nan,
                "final_gap": np.nan,
                "pricing_gap": np.nan,
                "runtime_seconds": run.get("runtime_seconds"),
                "projected_seconds": np.nan,
                "processed_origins": run.get("processed_regions"),
                "total_origins": run.get("represented_regions"),
            }
        )
    return pd.DataFrame(rows)


def _partC_rev1_reload_accuracy_tables(reload_summary: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    for run in reload_summary.get("runs", []):
        artifacts = dict(run.get("artifacts", {}))
        metrics_path = artifacts.get("validation_metrics_csv")
        metrics = _partC_rev1_read_csv(metrics_path) if metrics_path else pd.DataFrame()
        if not metrics.empty:
            metrics.insert(0, "algorithm", run.get("algorithm"))
            metrics.insert(0, "policy", run.get("policy"))
            metric_frames.append(metrics)
        run_rows.append(
            {
                "policy": run.get("policy"),
                "algorithm": run.get("algorithm"),
                "state": run.get("state"),
                "overall_wmape": run.get("overall_wmape"),
                "runtime_seconds": run.get("runtime_seconds"),
                "positive_pair_count": run.get("positive_pair_count"),
                "unreachable_pair_count": run.get("positive_flow_unreachable_pair_count"),
                "max_boundary_abs_residual": run.get("max_boundary_abs_residual"),
                "max_conservation_abs_residual": run.get("max_conservation_abs_residual"),
            }
        )
    metrics_all = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    return pd.DataFrame(run_rows), metrics_all


def _partC_rev1_training_tables(training_summary: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    artifacts = dict(training_summary.get("artifacts", {}))
    registry = _partC_rev1_read_csv(artifacts.get("model_registry_csv", ""))
    selected = _partC_rev1_read_csv(artifacts.get("selected_models_csv", ""))
    status = _partC_rev1_read_csv(artifacts.get("status_csv", ""))
    return registry, selected, status


def _partC_rev1_dynamic_sample_projection(status: pd.DataFrame) -> pd.DataFrame:
    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    families = ["xgboost", "simple_nn", "icnn_shallow", "icnn_deep"]
    rows: list[dict[str, Any]] = []
    trained = status.loc[status["status"].astype(str).eq("trained")] if "status" in status.columns else status
    for row in trained.itertuples(index=False):
        total = int(getattr(row, "requested_samples", getattr(row, "sample_count", 100)) or 100)
        counts = partC_rev1_nested_fraction_counts(total, fractions)
        for fraction, count in zip(fractions, counts):
            for family in families:
                rows.append(
                    {
                        "region_uuid": getattr(row, "region_uuid", ""),
                        "model_family": family,
                        "sample_fraction": fraction,
                        "nested_sample_count": count,
                        "status": "projected_only",
                        "reason": "Nested fraction retraining was not launched in this continuation run.",
                    }
                )
    return pd.DataFrame(rows)


def _partC_rev1_convergence_table(global_summary: dict[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in global_summary.get("runs", []):
        summary_path = Path(str(run.get("summary_path", "")))
        run_summary = partC_rev1_read_json(summary_path) if summary_path.exists() else {}
        conv_path = dict(run_summary.get("artifacts", {})).get("convergence_csv")
        conv = _partC_rev1_read_csv(conv_path) if conv_path else pd.DataFrame()
        if conv.empty:
            continue
        conv.insert(0, "algorithm", run.get("algorithm"))
        conv.insert(0, "policy", run.get("policy"))
        frames.append(conv)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _partC_rev1_lazy_table(global_summary: dict[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in global_summary.get("runs", []):
        summary_path = Path(str(run.get("summary_path", "")))
        run_summary = partC_rev1_read_json(summary_path) if summary_path.exists() else {}
        lazy_path = dict(run_summary.get("artifacts", {})).get("lazy_refresh_history_csv")
        lazy = _partC_rev1_read_csv(lazy_path) if lazy_path else pd.DataFrame()
        if lazy.empty:
            continue
        lazy.insert(0, "algorithm", run.get("algorithm"))
        lazy.insert(0, "policy", run.get("policy"))
        frames.append(lazy)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _partC_rev1_pricing_table(global_summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for run in global_summary.get("runs", []):
        run_state = dict(run.get("run", {}))
        rows.append(
            {
                "policy": run.get("policy"),
                "algorithm": run.get("algorithm"),
                "pricing_gap": run_state.get("pricing_gap"),
                "pricing_state": run_state.get("pricing_state"),
                "active_pairs_csv": run.get("active_pairs_csv"),
                "active_pair_count": run_state.get("active_pair_count", run.get("active_pair_count")),
                "runtime_seconds": run_state.get("runtime_seconds"),
                "status": "measured_complete",
            }
        )
    return pd.DataFrame(rows)


def _partC_rev1_partition_sensitivity_tables(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    try:
        summary_path = partC_rev1_find_latest_partition_sensitivity_summary(out_dir)
    except FileNotFoundError:
        missing = pd.DataFrame(
            [
                {
                    "status": "guarded_missing_partition_sensitivity_package",
                    "limitation": "Run run-cap2over3-partition-sweep after 4F to populate E8/E9/E10 partition artifacts.",
                    "resume_command": "conda run -n py312 python -m partC_rev1_src.partC_rev1_cli run-cap2over3-partition-sweep --config partC_rev1_configs/partC_rev1_real_cap2over3_gravity_training.json --confirm-full-run --heartbeat-seconds 15 --validate-dependency-hashes",
                }
            ]
        )
        return missing.copy(), missing.copy(), missing.copy(), None
    summary = partC_rev1_read_json(summary_path)
    artifacts = dict(summary.get("artifacts", {}))
    e8 = _partC_rev1_read_csv(artifacts.get("e8_partition_active_set_pricing_csv", ""))
    e9 = _partC_rev1_read_csv(artifacts.get("e9_partition_downstream_speed_accuracy_csv", ""))
    e10 = _partC_rev1_read_csv(artifacts.get("e10_boundary_path_consistency_csv", ""))
    return e8, e9, e10, summary


def _partC_rev1_break_even_table(summary_4c: dict[str, Any], training_summary: dict[str, Any], global_summary: dict[str, Any], reload_summary: dict[str, Any]) -> pd.DataFrame:
    best = dict(summary_4c.get("best_accuracy_benchmark", {}))
    full_seconds = float(best.get("runtime_seconds", 0.0) or 0.0)
    training_seconds = float(training_summary.get("runtime_seconds", 0.0) or 0.0)
    global_seconds = float(global_summary.get("runtime_seconds", 0.0) or 0.0)
    reload_best = dict(reload_summary.get("best_reload", {}))
    reload_seconds = 0.0
    if reload_best.get("summary_json"):
        reload_seconds = float(partC_rev1_read_json(reload_best["summary_json"]).get("runtime_seconds", 0.0) or 0.0)
    global_online = min((float(dict(run.get("run", {})).get("runtime_seconds", math.inf) or math.inf) for run in global_summary.get("runs", [])), default=0.0)
    offline_seconds = training_seconds + global_seconds
    table = partC_rev1_break_even_rows(
        offline_seconds=offline_seconds,
        full_seconds=full_seconds,
        hybrid_online_seconds=global_online,
        reload_seconds=reload_seconds,
    )
    table["training_seconds"] = training_seconds
    table["global_policy_suite_seconds"] = global_seconds
    table["selected_reload_seconds"] = reload_seconds
    return table


def _partC_rev1_availability(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    table_by_exp = {
        "E1": "partC_rev1_solver_backend_runtime",
        "E2": "partC_rev1_flow_accuracy_metrics",
        "E3": "partC_rev1_experiment_skips",
        "E4": "partC_rev1_dynamic_sample_projection",
        "E5": "partC_rev1_monotonicity_convergence_diagnostics",
        "E6": "partC_rev1_regional_to_national_error",
        "E7": "partC_rev1_lazy_update_trace",
        "E8": "partC_rev1_partition_active_set_pricing",
        "E9": "partC_rev1_partition_downstream_speed_accuracy",
        "E10": "partC_rev1_boundary_consistency",
        "E11": "partC_rev1_break_even",
        "E12": "partC_rev1_experiment_skips",
        "E13": "partC_rev1_i95_projection",
    }
    statuses = {
        "E1": ("complete", ""),
        "E2": ("complete", ""),
        "E3": ("skipped", "Explicitly skipped by the master continuation prompt."),
        "E4": ("projected_only", "Nested retraining/global reruns require a separate guarded run; projection table is written."),
        "E5": ("partial", "Uses saved model registry metrics and global convergence traces; no new derivative/Hessian probes were run."),
        "E6": ("projected_only", "Depends on Experiment 4 retraining/global reruns, which were not launched here."),
        "E7": ("partial", "Reads completed lazy traces; ablation grid is not launched here."),
        "E8": ("unavailable", "Partition sensitivity package was not found."),
        "E9": ("unavailable", "Partition sensitivity package was not found."),
        "E10": ("unavailable", "E10-specific partition/path consistency package was not found."),
        "E11": ("complete", ""),
        "E12": ("skipped", "Explicitly skipped by the master continuation prompt."),
        "E13": ("unavailable", "No isolated I-95 scenario snapshot/run exists yet."),
    }
    partition_exp_tables = {
        "E8": "partC_rev1_partition_active_set_pricing",
        "E9": "partC_rev1_partition_downstream_speed_accuracy",
        "E10": "partC_rev1_boundary_path_consistency",
    }
    for exp_id, table_name in partition_exp_tables.items():
        table = tables.get(table_name, pd.DataFrame())
        if table.empty:
            continue
        status_values = set(table.get("status", pd.Series(dtype=str)).astype(str))
        limitation_values = [
            str(value)
            for value in table.get("limitation", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
            if str(value).strip()
        ]
        limitation = limitation_values[0] if limitation_values else ""
        if any(value == "measured_current_partition" for value in status_values):
            statuses[exp_id] = (
                "partial_guarded",
                limitation or "Measured current partition is available; alternate-resolution reruns remain guarded.",
            )
        elif any(value.startswith("guarded_") for value in status_values):
            statuses[exp_id] = (
                "guarded_unavailable",
                limitation or "Guard report required before claiming alternate-resolution completion.",
            )
    rows = []
    for exp_id, title in PARTC_REV1_REVISED_EXPERIMENTS:
        table_name = table_by_exp[exp_id]
        status, limitations = statuses[exp_id]
        table = tables.get(table_name, pd.DataFrame())
        rows.append(
            {
                "experiment_id": exp_id,
                "experiment_title": title,
                "status": status,
                "table_name": table_name,
                "row_count": int(len(table)),
                "limitations": limitations,
                "resume_command": _partC_rev1_resume_command(exp_id),
            }
        )
    return pd.DataFrame(rows)


def _partC_rev1_resume_command(exp_id: str) -> str:
    if exp_id in {"E4", "E6"}:
        return "Implement and run a guarded nested sample-fraction training/global/reload suite."
    if exp_id in {"E8", "E9"}:
        return "Implement and run a guarded adaptive Leiden resolution sweep with downstream training/global/reload."
    if exp_id == "E7":
        return "Run a guarded lazy-update ablation grid in isolated global-policy folders."
    if exp_id == "E13":
        return "Create isolated I-95 capacity-reduction scenario snapshots and run global/reload comparisons."
    return ""


def _partC_rev1_write_paper_ready(summary: dict[str, Any], availability: pd.DataFrame, path: Path) -> None:
    lines = [
        "# partC_rev1 Revised Reviewer Experiments",
        "",
        f"Generated UTC: {summary['created_utc']}",
        f"Run hash: `{summary['run_hash']}`",
        f"Experiment root: `{summary['experiment_root']}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in availability["status"].value_counts().sort_index().items():
        lines.append(f"- `{status}`: {int(count)}")
    lines.extend(["", "## Key Completed Results", ""])
    best = dict(summary.get("best_reload", {}))
    lines.append(f"- Best Phase 4 reload: `{best.get('policy')}` + `{best.get('algorithm')}`, WMAPE `{best.get('overall_wmape')}`.")
    lines.append(f"- Physical benchmark reference: `{summary.get('selected_4c_benchmark_hash')}`.")
    lines.append("- Experiments 3 and 12 are skipped by instruction, not treated as failed.")
    lines.extend(["", "## Artifacts", ""])
    for label, artifact_path in dict(summary.get("artifacts", {})).items():
        lines.append(f"- {label}: `{artifact_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def partC_rev1_run_revised_reviewer_experiment_stage(
    *,
    config: PartCRev1RevisedReviewerExperimentConfig,
    output_dir: str | Path,
    reporter: PartCRev1ProgressReporter | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = partC_rev1_resolve_path(output_dir)
    summary_4c = partC_rev1_read_json(out_dir / "partC_rev1_4C_summary.json")
    training_summary = partC_rev1_read_json(partC_rev1_find_latest_full_training_summary(out_dir))
    global_summary_path = sorted(
        out_dir.glob("partC_rev1_global_policy_runs_*/partC_rev1_global_policy_suite_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[0]
    global_summary = partC_rev1_read_json(global_summary_path)
    reload_summary = partC_rev1_read_json(partC_rev1_find_latest_physical_reload_suite_summary(out_dir))
    run_hash = partC_rev1_json_hash(
        {
            "stage": PARTC_REV1_REVISED_EXPERIMENT_STAGE,
            "config": asdict(config),
            "training": training_summary.get("run_hash"),
            "global": global_summary.get("run_hash"),
            "reload": reload_summary.get("suite_hash"),
            "benchmark": dict(summary_4c.get("best_accuracy_benchmark", {})).get("run_hash"),
        }
    )[:16]
    root = out_dir / f"partC_rev1_revised_experiments_{run_hash}"
    tables_dir = root / "partC_rev1_tables"
    figs_dir = root / "partC_rev1_figures"
    data_dir = root / "partC_rev1_underlying_data"
    logs_dir = root / "partC_rev1_logs"
    for folder in [tables_dir, figs_dir, data_dir, logs_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    if reporter is not None:
        reporter.emit("revised_reviewer_experiments_started", run_hash=run_hash, root=str(root))

    registry, selected, status = _partC_rev1_training_tables(training_summary)
    reload_run_index, flow_accuracy = _partC_rev1_reload_accuracy_tables(reload_summary)
    convergence = _partC_rev1_convergence_table(global_summary)
    lazy = _partC_rev1_lazy_table(global_summary)
    pricing = _partC_rev1_pricing_table(global_summary)
    partition_e8, partition_e9, partition_e10, partition_summary = _partC_rev1_partition_sensitivity_tables(out_dir)
    monotonic_cols = [col for col in ["region_uuid", "model", "monotonicity_violation_rate", "min_monotonicity_dot", "wmape", "rmse"] if col in registry.columns]
    monotonic = registry[monotonic_cols].copy() if monotonic_cols else pd.DataFrame()
    if not convergence.empty:
        convergence_summary = convergence.copy()
        convergence_summary["diagnostic_type"] = "global_convergence"
        monotonic["diagnostic_type"] = "regional_model_registry" if not monotonic.empty else ""
        monotonic_convergence = pd.concat([monotonic, convergence_summary], ignore_index=True, sort=False)
    else:
        monotonic_convergence = monotonic

    tables: dict[str, pd.DataFrame] = {
        "partC_rev1_solver_backend_runtime": _partC_rev1_solver_runtime_table(summary_4c, global_summary, reload_summary),
        "partC_rev1_reload_run_index": reload_run_index,
        "partC_rev1_flow_accuracy_metrics": flow_accuracy,
        "partC_rev1_dynamic_sample_projection": _partC_rev1_dynamic_sample_projection(status),
        "partC_rev1_monotonicity_convergence_diagnostics": monotonic_convergence,
        "partC_rev1_regional_to_national_error": pd.DataFrame(
            [{"status": "projected_only", "reason": "Depends on Experiment 4 nested retraining/global/reload outputs."}]
        ),
        "partC_rev1_lazy_update_trace": lazy,
        "partC_rev1_sparse_pricing_trace": pricing,
        "partC_rev1_partition_active_set_pricing": partition_e8,
        "partC_rev1_partition_downstream_speed_accuracy": partition_e9,
        "partC_rev1_boundary_consistency": partition_e10,
        "partC_rev1_break_even": _partC_rev1_break_even_table(summary_4c, training_summary, global_summary, reload_summary),
        "partC_rev1_i95_projection": pd.DataFrame(
            [{"status": "unavailable", "reason": "No I-95 scenario snapshot/run exists.", "resume_command": _partC_rev1_resume_command("E13")}]
        ),
        "partC_rev1_experiment_skips": pd.DataFrame(
            [
                {"experiment_id": "E3", "status": "skipped", "reason": "Skipped by continuation prompt."},
                {"experiment_id": "E12", "status": "skipped", "reason": "Skipped by continuation prompt."},
            ]
        ),
    }
    table_paths: dict[str, str] = {}
    for name, table in tables.items():
        path = tables_dir / f"{name}.csv"
        partC_rev1_write_table(table, path, index=False)
        table_paths[name] = str(path)
        if reporter is not None:
            reporter.heartbeat("revised_experiment_table_written", table=name, rows=int(len(table)), elapsed_seconds=time.perf_counter() - started)

    figures: list[dict[str, str]] = []
    runtime_table = tables["partC_rev1_solver_backend_runtime"].dropna(subset=["runtime_seconds"]).copy()
    if not runtime_table.empty:
        figures.append(
            _partC_rev1_save_figures(
                root,
                "partC_rev1_fig_e1_runtime_by_source",
                runtime_table,
                "Runtime by physical/global/reload source.",
                lambda ax: (
                    runtime_table.groupby("source")["runtime_seconds"].median().plot(kind="bar", ax=ax),
                    ax.set_ylabel("Median runtime seconds"),
                    ax.set_xlabel("Source"),
                    ax.set_title("Runtime By Solver Source"),
                ),
            )
        )
    overall = flow_accuracy.loc[
        flow_accuracy.get("metric_scope_kind", pd.Series(dtype=str)).astype(str).eq("overall")
        & flow_accuracy.get("metric_scope_value", pd.Series(dtype=str)).astype(str).eq("all")
    ].copy()
    if not overall.empty:
        overall["run_label"] = overall["policy"].astype(str) + "/" + overall["algorithm"].astype(str)
        figures.append(
            _partC_rev1_save_figures(
                root,
                "partC_rev1_fig_e2_flow_wmape",
                overall,
                "All-arc WMAPE for completed reload outputs.",
                lambda ax: (
                    ax.bar(overall["run_label"], overall["wmape"].astype(float)),
                    ax.set_ylabel("WMAPE"),
                    ax.set_xlabel("Policy/algorithm"),
                    ax.tick_params(axis="x", rotation=25),
                    ax.set_title("Reload Flow Accuracy Versus 4C Benchmark"),
                ),
            )
        )
    break_even = tables["partC_rev1_break_even"]
    if not break_even.empty:
        figures.append(
            _partC_rev1_save_figures(
                root,
                "partC_rev1_fig_e11_break_even",
                break_even,
                "Break-even query count with and without reload.",
                lambda ax: (
                    ax.bar(break_even["formula"], break_even["n_break_even"].astype(float)),
                    ax.set_ylabel("Break-even scenario count"),
                    ax.set_xlabel("Formula"),
                    ax.set_title("Offline Cost Break-Even"),
                ),
            )
        )

    availability = _partC_rev1_availability(tables)
    availability_path = root / "partC_rev1_experiment_availability_matrix.csv"
    partC_rev1_write_table(availability, availability_path, index=False)
    comment_rows = []
    for row in availability.itertuples(index=False):
        comment_rows.append(
            {
                "reviewer_comment": f"Reviewer concern addressed by {row.experiment_title}",
                "experiment_id": row.experiment_id,
                "status": row.status,
                "table_path": table_paths.get(row.table_name, ""),
                "limitations": row.limitations,
                "resume_command": row.resume_command,
            }
        )
    comment_matrix = pd.DataFrame(comment_rows)
    comment_path = root / "partC_rev1_reviewer_comment_matrix.csv"
    partC_rev1_write_table(comment_matrix, comment_path, index=False)
    artifact_rows = []
    for label, path in table_paths.items():
        artifact_rows.append({"label": label, "path": path, "kind": Path(path).suffix.lstrip(".")})
    for figure in figures:
        for kind, path in figure.items():
            artifact_rows.append({"label": f"figure_{kind}", "path": path, "kind": Path(path).suffix.lstrip(".")})
    artifact_index_path = root / "partC_rev1_paper_ready_artifact_index.csv"
    partC_rev1_write_table(pd.DataFrame(artifact_rows), artifact_index_path, index=False)

    summary_path = root / "partC_rev1_revised_experiment_manifest.json"
    paper_ready_path = root / "partC_rev1_paper_ready_results.md"
    handoff_path = root / "partC_rev1_revised_experiments_handoff.md"
    summary = {
        "stage": PARTC_REV1_REVISED_EXPERIMENT_STAGE,
        "created_utc": _partC_rev1_utc_now(),
        "run_hash": run_hash,
        "experiment_root": str(root),
        "config": asdict(config),
        "source_hashes": {
            "training_run_hash": training_summary.get("run_hash"),
            "global_policy_run_hash": global_summary.get("run_hash"),
            "physical_reload_suite_hash": reload_summary.get("suite_hash"),
            "partition_sensitivity_run_hash": None if partition_summary is None else partition_summary.get("run_hash"),
        },
        "selected_4c_benchmark_hash": dict(summary_4c.get("best_accuracy_benchmark", {})).get("run_hash"),
        "best_reload": reload_summary.get("best_reload"),
        "availability_counts": {str(key): int(value) for key, value in availability["status"].value_counts().to_dict().items()},
        "table_count": len(tables) + 3,
        "figure_count": len(figures),
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {
            "manifest_json": str(summary_path),
            "handoff_md": str(handoff_path),
            "paper_ready_results_md": str(paper_ready_path),
            "availability_matrix_csv": str(availability_path),
            "reviewer_comment_matrix_csv": str(comment_path),
            "artifact_index_csv": str(artifact_index_path),
            "tables_dir": str(tables_dir),
            "figures_dir": str(figs_dir),
            "underlying_data_dir": str(data_dir),
        },
        "limitations": availability.loc[availability["status"].ne("complete"), ["experiment_id", "status", "limitations", "resume_command"]].to_dict("records"),
    }
    partC_rev1_write_json(summary, summary_path)
    _partC_rev1_write_paper_ready(summary, availability, paper_ready_path)
    handoff_path.write_text(_partC_rev1_revised_handoff(summary, availability), encoding="utf-8")
    (logs_dir / "partC_rev1_run.log").write_text(
        f"{_partC_rev1_utc_now()} revised reviewer experiments completed run_hash={run_hash}\n",
        encoding="utf-8",
    )
    if reporter is not None:
        reporter.emit("revised_reviewer_experiments_finished", summary=str(summary_path), handoff=str(handoff_path))
    return summary


def _partC_rev1_revised_handoff(summary: dict[str, Any], availability: pd.DataFrame) -> str:
    lines = [
        "# partC_rev1 Revised Reviewer Experiments Handoff",
        "",
        f"Generated UTC: {_partC_rev1_utc_now()}",
        "",
        f"- State: `measured_complete_with_projected_and_skipped_experiments`",
        f"- Run hash: `{summary.get('run_hash')}`",
        f"- Root: `{summary.get('experiment_root')}`",
        f"- Availability counts: `{summary.get('availability_counts')}`",
        f"- Runtime seconds: {summary.get('runtime_seconds')}",
        "",
        "## Experiment Status",
        "",
        "| Experiment | Status | Rows | Limitation |",
        "| --- | --- | ---: | --- |",
    ]
    for row in availability.itertuples(index=False):
        lines.append(f"| {row.experiment_id} | {row.status} | {row.row_count} | {row.limitations} |")
    lines.extend(["", "## Artifacts", ""])
    for label, path in dict(summary.get("artifacts", {})).items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "## Resume",
            "",
            "```text",
            "conda run -n py312 python -m partC_rev1_src.partC_rev1_cli run-revised-reviewer-experiments --config partC_rev1_configs/partC_rev1_real_preprocess_config.json --confirm-full-run",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"
