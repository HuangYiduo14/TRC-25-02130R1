"""Prompt 4G reviewer experiment tables, figures, and artifact index."""

from __future__ import annotations

import json
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
from .partC_rev1_paths import partC_rev1_resolve_path
from .partC_rev1_progress_logging import PartCRev1ProgressReporter


PARTC_REV1_4G_STAGE = "partC_rev1_4G"
PARTC_REV1_REVIEWER_EXPERIMENTS = [
    ("E1", "Backend and algorithm fairness"),
    ("E2", "Hybrid versus full physical benchmark"),
    ("E3", "FAF/NTAD reference validation"),
    ("E4", "Regional learning curves and reproducibility"),
    ("E5", "Monotonicity, derivatives, and solver convergence"),
    ("E6", "National propagation of regional error"),
    ("E7", "Lazy-update ablation"),
    ("E8", "Sparse active-set and pricing ablation"),
    ("E9", "Leiden partition stability"),
    ("E10", "Boundary and path consistency"),
    ("E11", "Offline cost and break-even"),
    ("E12", "Background-capacity sensitivity"),
    ("E13", "I-95 closure and optional event/cross-year checks"),
]


@dataclass(frozen=True, slots=True)
class PartCRev1ReviewerExperimentConfig:
    config_path: str
    bootstrap_seed: int = 42
    max_scatter_points: int = 20000


def _partC_rev1_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def partC_rev1_metric_summary(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    pred = np.asarray(predicted, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError("predicted and reference arrays must have identical shape")
    diff = pred - ref
    abs_diff = np.abs(diff)
    weight = np.ones_like(pred) if weights is None else np.asarray(weights, dtype=np.float64)
    if weight.shape != pred.shape:
        raise ValueError("weights must match predicted/reference shape")
    pearson = math.nan if len(pred) < 2 or np.std(pred) <= 0.0 or np.std(ref) <= 0.0 else float(np.corrcoef(pred, ref)[0, 1])
    return {
        "mae": float(np.mean(abs_diff)) if len(pred) else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if len(pred) else 0.0,
        "wmape": float(np.sum(abs_diff) / max(float(np.sum(np.abs(ref))), epsilon)),
        "flow_weighted_mae": float(np.dot(weight, abs_diff) / max(float(np.sum(weight)), epsilon)),
        "pearson": pearson,
    }


def partC_rev1_geh_summary(predicted: np.ndarray, reference: np.ndarray, *, compatible_vehicle_counts: bool, epsilon: float = 1e-12) -> dict[str, float | str]:
    if not compatible_vehicle_counts:
        return {"status": "unavailable_incompatible_units"}
    pred = np.asarray(predicted, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError("predicted and reference arrays must have identical shape")
    geh = np.sqrt(2.0 * np.square(pred - ref) / np.maximum(pred + ref, epsilon))
    return {
        "status": "computed",
        "geh_lt_5_share": float(np.mean(geh < 5.0)) if len(geh) else 0.0,
        "geh_5_to_10_share": float(np.mean((geh >= 5.0) & (geh <= 10.0))) if len(geh) else 0.0,
        "geh_gt_10_share": float(np.mean(geh > 10.0)) if len(geh) else 0.0,
        "geh_mean": float(np.mean(geh)) if len(geh) else 0.0,
    }


def partC_rev1_deterministic_sample(values: list[Any], n: int, *, seed: int) -> list[Any]:
    rng = np.random.default_rng(int(seed))
    if n >= len(values):
        return list(values)
    idx = np.sort(rng.choice(len(values), size=int(n), replace=False))
    return [values[int(i)] for i in idx.tolist()]


def partC_rev1_join_by_stable_id(left: pd.DataFrame, right: pd.DataFrame, id_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if id_col not in left.columns or id_col not in right.columns:
        raise ValueError(f"Stable ID column missing for join: {id_col}")
    joined = left.merge(right, on=id_col, how="left", indicator=True, suffixes=("_pred", "_ref"))
    matched = int(joined["_merge"].eq("both").sum())
    coverage = matched / max(int(len(left)), 1)
    return joined, {"left_rows": int(len(left)), "matched_rows": matched, "coverage": float(coverage)}


def _partC_rev1_read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _partC_rev1_flatten(prefix: str, obj: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            row.update(_partC_rev1_flatten(f"{prefix}{key}.", value))
        elif not isinstance(value, (list, tuple)):
            row[f"{prefix}{key}"] = value
    return row


def _partC_rev1_figure_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _partC_rev1_save_figure(
    fig: Any,
    *,
    figure_dir: Path,
    underlying_dir: Path,
    stem: str,
    table: pd.DataFrame,
    caption: str,
) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    underlying_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{stem}.png"
    svg = figure_dir / f"{stem}.svg"
    pdf = figure_dir / f"{stem}.pdf"
    table_path = underlying_dir / f"{stem}_underlying.csv"
    meta_path = underlying_dir / f"{stem}_caption.json"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    partC_rev1_write_table(table, table_path, index=False)
    partC_rev1_write_json({"caption": caption, "created_utc": _partC_rev1_utc_now()}, meta_path)
    return {"png": str(png), "svg": str(svg), "pdf": str(pdf), "underlying_csv": str(table_path), "caption_json": str(meta_path)}


def _partC_rev1_benchmark_matrix(summary_4c: dict[str, Any], summary_4b: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in summary_4c.get("run_summaries", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "source_stage": "4C",
                "method": item.get("method"),
                "backend": item.get("backend"),
                "state": item.get("state"),
                "iterations": item.get("iteration_count"),
                "final_gap": item.get("final_gap"),
                "runtime_seconds": item.get("runtime_seconds"),
                "peak_memory_mb": item.get("peak_memory_mb"),
                "projected_full_backend_seconds": item.get("projected_full_backend_seconds"),
                "origins_per_second": item.get("reachable_demand", 0.0) / max(float(item.get("runtime_seconds", 0.0) or 0.0), 1e-12),
            }
        )
    for key, value in summary_4b.items():
        if isinstance(value, dict) and {"backend", "runtime_seconds"} & set(value):
            rows.append(
                {
                    "source_stage": "4B",
                    "method": value.get("method", key),
                    "backend": value.get("backend"),
                    "state": value.get("state", value.get("completion_state", "measured_partial_projected")),
                    "iterations": value.get("iteration_count"),
                    "final_gap": value.get("final_gap"),
                    "runtime_seconds": value.get("runtime_seconds"),
                    "peak_memory_mb": value.get("peak_memory_mb"),
                    "projected_full_backend_seconds": value.get("projected_full_backend_seconds"),
                    "origins_per_second": value.get("processed_origins", 0.0) / max(float(value.get("runtime_seconds", 0.0) or 0.0), 1e-12),
                }
            )
    return pd.DataFrame(rows)


def _partC_rev1_partition_summary(outputs: Path) -> pd.DataFrame:
    runs = _partC_rev1_read_csv_if_exists(outputs / "partC_rev1_partition_stability_runs.csv")
    if runs.empty:
        return pd.DataFrame(columns=["resolution", "seed_count", "mean_region_count", "min_region_count", "max_region_count"])
    region_col = "region_count" if "region_count" in runs.columns else next((col for col in runs.columns if "region" in col and "count" in col), None)
    if region_col is None:
        runs["region_count_proxy"] = 1
        region_col = "region_count_proxy"
    return (
        runs.groupby("resolution", as_index=False)
        .agg(
            seed_count=("random_seed", "nunique") if "random_seed" in runs.columns else (region_col, "count"),
            mean_region_count=(region_col, "mean"),
            min_region_count=(region_col, "min"),
            max_region_count=(region_col, "max"),
        )
        .sort_values("resolution")
    )


def _partC_rev1_boundary_summary(run_folder: Path) -> pd.DataFrame:
    boundary = _partC_rev1_read_csv_if_exists(run_folder / "partC_rev1_boundary_consistency.csv")
    conservation = _partC_rev1_read_csv_if_exists(run_folder / "partC_rev1_conservation_residuals.csv")
    rows = []
    if not boundary.empty:
        rows.append(
            {
                "check": "terminal_boundary",
                "row_count": int(len(boundary)),
                "max_abs_residual": float(boundary["residual"].abs().max()),
                "p95_abs_residual": float(boundary["residual"].abs().quantile(0.95)),
            }
        )
    if not conservation.empty:
        rows.append(
            {
                "check": "internal_conservation",
                "row_count": int(len(conservation)),
                "max_abs_residual": float(conservation["residual"].abs().max()),
                "p95_abs_residual": float(conservation["residual"].abs().quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def _partC_rev1_write_plot_pack(tables: dict[str, pd.DataFrame], summary_4e: dict[str, Any], summary_4f: dict[str, Any], root: Path) -> list[dict[str, str]]:
    plt = _partC_rev1_figure_setup()
    fig_dir = root / "partC_rev1_figures"
    data_dir = root / "partC_rev1_underlying_data"
    outputs: list[dict[str, str]] = []

    benchmark = tables.get("solver_benchmark_matrix", pd.DataFrame())
    if not benchmark.empty and {"method", "runtime_seconds", "final_gap"}.issubset(benchmark.columns):
        plot_df = benchmark.dropna(subset=["runtime_seconds"]).copy()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(plot_df["runtime_seconds"], plot_df["final_gap"].fillna(np.nan), c=np.arange(len(plot_df)), cmap="viridis")
        for row in plot_df.itertuples(index=False):
            ax.annotate(f"{row.method}/{row.backend}", (float(row.runtime_seconds), float(row.final_gap) if pd.notna(row.final_gap) else 0.0), fontsize=7)
        ax.set_xlabel("Runtime seconds")
        ax.set_ylabel("Final relative gap")
        ax.set_yscale("log")
        ax.set_title("Prompt-4C Solver Runtime and Gap")
        outputs.append(_partC_rev1_save_figure(fig, figure_dir=fig_dir, underlying_dir=data_dir, stem="partC_rev1_fig01_solver_runtime_gap", table=plot_df, caption="Runtime versus final relative gap for available Prompt-4C solver benchmarks."))
        plt.close(fig)

    validation = tables.get("hybrid_vs_physical_metrics", pd.DataFrame())
    if not validation.empty and {"metric_scope", "wmape"}.issubset(validation.columns):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(validation["metric_scope"].astype(str), validation["wmape"].astype(float))
        ax.set_xlabel("Flow scope")
        ax.set_ylabel("WMAPE")
        ax.set_title("Hybrid-Reloaded Flow Error Versus Physical Benchmark")
        outputs.append(_partC_rev1_save_figure(fig, figure_dir=fig_dir, underlying_dir=data_dir, stem="partC_rev1_fig02_hybrid_physical_wmape", table=validation, caption="WMAPE by flow scope for Prompt-4F final flow compared with the selected Prompt-4C physical benchmark."))
        plt.close(fig)

    models = tables.get("surrogate_model_metrics", pd.DataFrame())
    if not models.empty and {"model", "wmape"}.issubset(models.columns):
        model_plot = models.groupby("model", as_index=False).agg(median_wmape=("wmape", "median"), median_runtime_seconds=("runtime_seconds", "median"))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(model_plot["model"], model_plot["median_wmape"])
        ax.set_xlabel("Model")
        ax.set_ylabel("Median validation WMAPE")
        ax.set_title("Regional Surrogate Pilot Accuracy")
        ax.tick_params(axis="x", rotation=20)
        outputs.append(_partC_rev1_save_figure(fig, figure_dir=fig_dir, underlying_dir=data_dir, stem="partC_rev1_fig04_surrogate_wmape", table=model_plot, caption="Median validation WMAPE by surrogate family over measured Prompt-4D pilot regions."))
        plt.close(fig)

    sparse = tables.get("sparse_pricing_ablation", pd.DataFrame())
    if not sparse.empty and "compression_ratio" in sparse.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["4E active support"], sparse["compression_ratio"].astype(float))
        ax.set_ylabel("active / conceptual")
        ax.set_title("Sparse Active-Set Compression")
        outputs.append(_partC_rev1_save_figure(fig, figure_dir=fig_dir, underlying_dir=data_dir, stem="partC_rev1_fig08_active_compression", table=sparse, caption="Sparse active-pair compression ratio from Prompt-4E measured support."))
        plt.close(fig)

    partition = tables.get("partition_stability_summary", pd.DataFrame())
    if not partition.empty and {"resolution", "mean_region_count"}.issubset(partition.columns):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(partition["resolution"], partition["mean_region_count"], marker="o")
        ax.set_xlabel("Leiden resolution")
        ax.set_ylabel("Mean region count")
        ax.set_title("Prompt-3 Partition Stability")
        outputs.append(_partC_rev1_save_figure(fig, figure_dir=fig_dir, underlying_dir=data_dir, stem="partC_rev1_fig09_partition_region_count", table=partition, caption="Mean region count versus fixed-backbone Leiden resolution from Prompt-3 stability artifacts."))
        plt.close(fig)

    boundary = tables.get("boundary_consistency_summary", pd.DataFrame())
    if not boundary.empty and {"check", "max_abs_residual"}.issubset(boundary.columns):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(boundary["check"], boundary["max_abs_residual"].astype(float))
        ax.set_xlabel("Check")
        ax.set_ylabel("Max absolute residual")
        ax.set_title("Physical Reload Consistency")
        ax.tick_params(axis="x", rotation=15)
        outputs.append(_partC_rev1_save_figure(fig, figure_dir=fig_dir, underlying_dir=data_dir, stem="partC_rev1_fig10_boundary_residuals", table=boundary, caption="Maximum terminal-boundary and internal-conservation residuals from Prompt-4F physical reload."))
        plt.close(fig)

    break_even = tables.get("break_even_analysis", pd.DataFrame())
    if not break_even.empty:
        queries = np.arange(1, 11, dtype=float)
        full_time = float(summary_4f.get("runtime_break_even", {}).get("selected_physical_runtime_seconds", 0.0))
        hybrid_online = float(summary_4f.get("runtime_break_even", {}).get("global_online_seconds", 0.0))
        reload_time = float(summary_4f.get("runtime_break_even", {}).get("physical_reload_seconds", 0.0))
        offline = float(summary_4f.get("runtime_break_even", {}).get("offline_seconds_used", 0.0))
        plot_df = pd.DataFrame(
            {
                "queries": queries.astype(int),
                "full_physical_seconds": queries * full_time,
                "hybrid_online_seconds": offline + queries * hybrid_online,
                "hybrid_with_reload_seconds": offline + queries * (hybrid_online + reload_time),
            }
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(plot_df["queries"], plot_df["full_physical_seconds"], marker="o", label="Full physical")
        ax.plot(plot_df["queries"], plot_df["hybrid_online_seconds"], marker="o", label="Hybrid online")
        ax.plot(plot_df["queries"], plot_df["hybrid_with_reload_seconds"], marker="o", label="Hybrid + reload")
        ax.set_xlabel("Repeated scenario count")
        ax.set_ylabel("Cumulative seconds")
        ax.set_title("Offline Cost and Break-Even")
        ax.legend()
        outputs.append(_partC_rev1_save_figure(fig, figure_dir=fig_dir, underlying_dir=data_dir, stem="partC_rev1_fig11_break_even", table=plot_df, caption="Cumulative runtime comparison using Prompt-4F break-even inputs."))
        plt.close(fig)

    return outputs


def _partC_rev1_availability_rows(paths: dict[str, Path], tables: dict[str, pd.DataFrame], summary_4e: dict[str, Any], summary_4f: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    table_names = {
        "E1": "solver_benchmark_matrix",
        "E2": "hybrid_vs_physical_metrics",
        "E3": "reference_validation_summary",
        "E4": "surrogate_model_metrics",
        "E5": "monotonicity_diagnostics",
        "E6": "regional_error_propagation",
        "E7": "lazy_update_ablation",
        "E8": "sparse_pricing_ablation",
        "E9": "partition_stability_summary",
        "E10": "boundary_consistency_summary",
        "E11": "break_even_analysis",
        "E12": "background_capacity_sensitivity",
        "E13": "i95_counterfactual",
    }
    unavailable = {
        "E6": "Controlled perturbation/replacement runs were not executed under Prompt-4 runtime guard.",
        "E12": "Background-capacity sensitivity scenarios were not generated in prior stages.",
        "E13": "I-95 disruption/event/cross-year scenario inputs and runs are absent; cannot validate or demonstrate without a new scenario run.",
    }
    partial = {
        "E3": "FAF reference summary is available; NTAD vehicle-count GEH is unavailable without compatible high-confidence truck-AADT conversion table.",
        "E4": "Surrogate metrics are from measured 4D pilot regions, not all regions or full learning-curve sample sizes.",
        "E5": "Monotonicity diagnostics use 4D registry summaries; finite-difference/autodiff Hessian checks were not run.",
        "E7": "Only the realized 4E lazy-refresh trace is available; no grid ablation was launched.",
        "E8": "Only the realized 4E active-set/pricing trace is available; no top-K/pricing grid ablation was launched.",
        "E10": "Boundary and conservation checks are complete for 4F represented active regions; path expansion maps are not generated.",
    }
    for exp_id, title in PARTC_REV1_REVIEWER_EXPERIMENTS:
        table_name = table_names[exp_id]
        table = tables.get(table_name, pd.DataFrame())
        if exp_id in unavailable:
            status = "unavailable"
            limitation = unavailable[exp_id]
        elif exp_id in partial:
            status = "partial"
            limitation = partial[exp_id]
        else:
            status = "complete" if not table.empty else "unavailable"
            limitation = "" if not table.empty else "Required artifact table is absent."
        rows.append(
            {
                "experiment_id": exp_id,
                "experiment_title": title,
                "status": status,
                "table_name": table_name,
                "row_count": int(len(table)),
                "limitations": limitation,
                "source_state_4e": summary_4e.get("run", {}).get("state"),
                "source_state_4f": summary_4f.get("state"),
            }
        )
    return pd.DataFrame(rows)


def _partC_rev1_comment_matrix(availability: pd.DataFrame, figure_outputs: list[dict[str, str]], table_paths: dict[str, str]) -> pd.DataFrame:
    figure_by_exp = {
        "E1": "partC_rev1_fig01_solver_runtime_gap",
        "E2": "partC_rev1_fig02_hybrid_physical_wmape",
        "E4": "partC_rev1_fig04_surrogate_wmape",
        "E8": "partC_rev1_fig08_active_compression",
        "E9": "partC_rev1_fig09_partition_region_count",
        "E10": "partC_rev1_fig10_boundary_residuals",
        "E11": "partC_rev1_fig11_break_even",
    }
    fig_lookup = {Path(item.get("svg", "")).stem: item for item in figure_outputs}
    rows: list[dict[str, Any]] = []
    for item in availability.itertuples(index=False):
        fig_key = figure_by_exp.get(item.experiment_id)
        fig = fig_lookup.get(fig_key, {})
        rows.append(
            {
                "reviewer_comment": f"Reviewer concern addressed by {item.experiment_title}",
                "experiment_id": item.experiment_id,
                "table_path": table_paths.get(item.table_name, ""),
                "figure_path": fig.get("svg", ""),
                "status": item.status,
                "limitations": item.limitations,
            }
        )
    return pd.DataFrame(rows)


def _partC_rev1_paper_ready_results(summary: dict[str, Any], availability: pd.DataFrame) -> str:
    lines = [
        "# partC_rev1 Paper-Ready Results Index",
        "",
        f"Generated UTC: {summary['created_utc']}",
        "",
        f"Reviewer experiment root: `{summary['experiment_root']}`",
        f"Run hash: `{summary['run_hash']}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in availability["status"].value_counts().sort_index().items():
        lines.append(f"- `{status}`: {int(count)} experiments")
    lines.extend(
        [
            "",
            "## Important Interpretation Notes",
            "",
            "- Prompt 4G does not fabricate missing experiments. Missing perturbation, sensitivity, and I-95 scenario runs are marked unavailable.",
            "- Prompt 4F final flow is a measured zero-flow complete reload for represented 4E active regions, so hybrid-versus-physical error plots describe that zero-flow pilot outcome.",
            "- Regional surrogate and sparse active-set results are measured pilot/partial unless explicitly labeled complete.",
            "",
            "## Artifact Manifest",
            "",
            f"- Experiment manifest: `{summary['artifacts']['experiment_manifest_json']}`",
            f"- Availability matrix: `{summary['artifacts']['availability_matrix_csv']}`",
            f"- Reviewer comment matrix: `{summary['artifacts']['reviewer_comment_matrix_csv']}`",
            f"- Tables directory: `{summary['tables_dir']}`",
            f"- Figures directory: `{summary['figures_dir']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def partC_rev1_run_reviewer_experiment_stage(
    *,
    config: PartCRev1ReviewerExperimentConfig,
    output_dir: str | Path,
    reporter: PartCRev1ProgressReporter | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = partC_rev1_resolve_path(output_dir)
    summary_4b = partC_rev1_read_json(out_dir / "partC_rev1_4B_solver_pilot_summary.json")
    summary_4c = partC_rev1_read_json(out_dir / "partC_rev1_4C_summary.json")
    summary_4d = partC_rev1_read_json(out_dir / "partC_rev1_4D_summary.json")
    summary_4e = partC_rev1_read_json(out_dir / "partC_rev1_4E_summary.json")
    summary_4f = partC_rev1_read_json(out_dir / "partC_rev1_4F_summary.json")
    run_hash = partC_rev1_json_hash(
        {
            "stage": PARTC_REV1_4G_STAGE,
            "config": asdict(config),
            "4c": summary_4c.get("best_accuracy_benchmark", {}).get("run_hash"),
            "4d": summary_4d.get("stage"),
            "4e": summary_4e.get("hybrid_graph_hash"),
            "4f": summary_4f.get("run_hash"),
        }
    )[:16]
    root = out_dir / f"partC_rev1_reviewer_experiments_{run_hash}"
    tables_dir = root / "partC_rev1_tables"
    figures_dir = root / "partC_rev1_figures"
    maps_dir = root / "partC_rev1_maps"
    logs_dir = root / "partC_rev1_logs"
    data_dir = root / "partC_rev1_underlying_data"
    for path in [tables_dir, figures_dir, maps_dir, logs_dir, data_dir]:
        path.mkdir(parents=True, exist_ok=True)
    if reporter is not None:
        reporter.emit("reviewer_experiments_started", run_hash=run_hash, root=str(root))

    run_folder_4f = Path(str(summary_4f.get("run_folder", "")))
    tables: dict[str, pd.DataFrame] = {}
    tables["solver_benchmark_matrix"] = _partC_rev1_benchmark_matrix(summary_4c, summary_4b)
    tables["hybrid_vs_physical_metrics"] = _partC_rev1_read_csv_if_exists(run_folder_4f / "partC_rev1_validation_metrics.csv")
    reference_summary_path = out_dir / "partC_rev1_reference_flow_validation_summary.json"
    tables["reference_validation_summary"] = pd.DataFrame([_partC_rev1_flatten("", partC_rev1_read_json(reference_summary_path))]) if reference_summary_path.exists() else pd.DataFrame()
    tables["surrogate_model_metrics"] = _partC_rev1_read_csv_if_exists(out_dir / "partC_rev1_4D_model_registry.csv")
    model_metrics = tables["surrogate_model_metrics"]
    monotonic_cols = [col for col in ["region_uuid", "model", "monotonicity_violation_rate", "min_monotonicity_dot", "rmse", "wmape"] if col in model_metrics.columns]
    tables["monotonicity_diagnostics"] = model_metrics[monotonic_cols].copy() if monotonic_cols else pd.DataFrame()
    tables["regional_error_propagation"] = pd.DataFrame(
        [{"status": "unavailable", "reason": "Controlled perturbation or exact-replacement national reruns were not executed."}]
    )
    tables["lazy_update_ablation"] = _partC_rev1_read_csv_if_exists(out_dir / "partC_rev1_4E_lazy_refresh_history.csv")
    active_pairs = int(summary_4e.get("active_pair_count", 0))
    conceptual = int(summary_4e.get("conceptual_pair_estimate_for_model_regions", 0))
    tables["sparse_pricing_ablation"] = pd.DataFrame(
        [
            {
                "state": summary_4e.get("run", {}).get("state"),
                "active_pairs": active_pairs,
                "conceptual_pair_estimate": conceptual,
                "compression_ratio": active_pairs / max(conceptual, 1),
                "pricing_gap": summary_4e.get("run", {}).get("pricing_gap"),
                "pricing_state": summary_4e.get("run", {}).get("pricing_state"),
                "runtime_seconds": summary_4e.get("run", {}).get("runtime_seconds"),
            }
        ]
    )
    tables["partition_stability_summary"] = _partC_rev1_partition_summary(out_dir)
    tables["boundary_consistency_summary"] = _partC_rev1_boundary_summary(run_folder_4f)
    tables["break_even_analysis"] = pd.DataFrame([_partC_rev1_flatten("", summary_4f.get("runtime_break_even", {}))])
    tables["background_capacity_sensitivity"] = pd.DataFrame(
        [{"status": "unavailable", "reason": "Capacity sensitivity scenarios were not generated."}]
    )
    tables["i95_counterfactual"] = pd.DataFrame(
        [{"status": "unavailable", "reason": "No I-95 closure/counterfactual scenario run exists in saved artifacts."}]
    )

    table_paths: dict[str, str] = {}
    for name, table in tables.items():
        path = tables_dir / f"partC_rev1_{name}.csv"
        partC_rev1_write_table(table, path, index=False)
        table_paths[name] = str(path)
        if reporter is not None:
            reporter.heartbeat("reviewer_table_written", table=name, rows=int(len(table)), elapsed_seconds=time.perf_counter() - started)

    availability = _partC_rev1_availability_rows({}, tables, summary_4e, summary_4f)
    availability_path = root / "partC_rev1_experiment_availability_matrix.csv"
    partC_rev1_write_table(availability, availability_path, index=False)
    table_paths["experiment_availability_matrix"] = str(availability_path)
    figure_outputs = _partC_rev1_write_plot_pack(tables, summary_4e, summary_4f, root)
    comment_matrix = _partC_rev1_comment_matrix(availability, figure_outputs, table_paths)
    comment_matrix_path = root / "partC_rev1_reviewer_comment_matrix.csv"
    partC_rev1_write_table(comment_matrix, comment_matrix_path, index=False)

    artifact_rows: list[dict[str, Any]] = []
    for label, path in table_paths.items():
        artifact_rows.append({"label": label, "path": path, "kind": Path(path).suffix.lstrip("."), "tracked_candidate": True})
    for idx, fig in enumerate(figure_outputs, start=1):
        for kind, path in fig.items():
            artifact_rows.append({"label": f"figure_{idx}_{kind}", "path": path, "kind": Path(path).suffix.lstrip("."), "tracked_candidate": kind in {"svg", "pdf", "underlying_csv", "caption_json"}})
    artifact_index = pd.DataFrame(artifact_rows)
    artifact_index_path = root / "partC_rev1_paper_ready_artifact_index.csv"
    partC_rev1_write_table(artifact_index, artifact_index_path, index=False)

    summary = {
        "stage": PARTC_REV1_4G_STAGE,
        "created_utc": _partC_rev1_utc_now(),
        "run_hash": run_hash,
        "experiment_root": str(root),
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "maps_dir": str(maps_dir),
        "logs_dir": str(logs_dir),
        "underlying_data_dir": str(data_dir),
        "config": asdict(config),
        "availability_counts": {str(k): int(v) for k, v in availability["status"].value_counts().to_dict().items()},
        "figure_count": len(figure_outputs),
        "table_count": len(tables) + 3,
        "source_stage_states": {
            "4C_best": summary_4c.get("best_accuracy_benchmark", {}).get("state"),
            "4D": summary_4d.get("stage"),
            "4E": summary_4e.get("run", {}).get("state"),
            "4F": summary_4f.get("state"),
        },
        "artifacts": {
            "experiment_manifest_json": str(root / "partC_rev1_experiment_manifest.json"),
            "availability_matrix_csv": str(availability_path),
            "reviewer_comment_matrix_csv": str(comment_matrix_path),
            "paper_ready_results_md": str(root / "partC_rev1_paper_ready_results.md"),
            "paper_ready_artifact_index_csv": str(artifact_index_path),
        },
        "known_limitations": availability.loc[availability["status"].ne("complete"), ["experiment_id", "limitations"]].to_dict("records"),
        "runtime_seconds": time.perf_counter() - started,
    }
    paper_ready = root / "partC_rev1_paper_ready_results.md"
    paper_ready.write_text(_partC_rev1_paper_ready_results(summary, availability), encoding="utf-8")
    partC_rev1_write_json(summary, root / "partC_rev1_experiment_manifest.json")
    partC_rev1_write_json(summary, out_dir / "partC_rev1_4G_summary.json")
    (logs_dir / "partC_rev1_run.log").write_text(
        f"{_partC_rev1_utc_now()} 4G reviewer experiments completed run_hash={run_hash}\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(out_dir / "partC_rev1_4G_summary.json")
    if reporter is not None:
        reporter.emit("reviewer_experiments_finished", summary=summary["summary_path"], root=str(root))
    return summary


def partC_rev1_write_4g_handoff(
    summary: dict[str, Any],
    *,
    output_dir: str | Path,
    reporter: PartCRev1ProgressReporter | None = None,
) -> Path:
    out_dir = partC_rev1_resolve_path(output_dir)
    path = out_dir / "partC_rev1_4G_handoff.md"
    lines = [
        "# partC_rev1 4G Handoff",
        "",
        f"Generated UTC: {_partC_rev1_utc_now()}",
        "",
        "## Status",
        "",
        f"- Run hash: `{summary.get('run_hash')}`",
        f"- Experiment root: `{summary.get('experiment_root')}`",
        f"- Availability counts: `{summary.get('availability_counts')}`",
        f"- Figure count: {summary.get('figure_count')}",
        f"- Table count: {summary.get('table_count')}",
        f"- Runtime seconds: {summary.get('runtime_seconds')}",
        "",
        "## Artifacts",
        "",
    ]
    for label, artifact_path in dict(summary.get("artifacts", {})).items():
        lines.append(f"- {label}: `{artifact_path}`")
    lines.extend(["", "## Limitations", ""])
    for item in summary.get("known_limitations", []):
        lines.append(f"- {item.get('experiment_id')}: {item.get('limitations')}")
    lines.extend(
        [
            "",
            "## Tests",
            "",
            "- Focused 4G and full post-stage tests are recorded in the execution manifest/current-state docs after commit.",
            "- Environment note: `conda run -n py312` emits the known `conda-libmamba-solver` entry-point warning.",
            "",
            "## Git",
            "",
            "- Completed-stage commit: pending.",
            "",
            "## Resume Command",
            "",
            "```text",
            "conda run -n py312 python -m partC_rev1_src.partC_rev1_cli run-reviewer-experiments --config partC_rev1_configs/partC_rev1_real_preprocess_config.json --confirm-full-run",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if reporter is not None:
        reporter.emit("handoff_written", handoff=str(path))
    return path
