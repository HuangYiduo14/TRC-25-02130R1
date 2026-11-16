"""Guarded cap2over3 partition-sensitivity artifacts for E8/E9/E10."""

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

from .partC_rev1_config import load_partC_rev1_config
from .partC_rev1_hashing import partC_rev1_json_hash
from .partC_rev1_io import partC_rev1_read_json, partC_rev1_write_json, partC_rev1_write_table
from .partC_rev1_paths import partC_rev1_resolve_path
from .partC_rev1_physical_reload_suite import partC_rev1_find_latest_global_policy_summary
from .partC_rev1_progress_logging import PartCRev1ProgressReporter


PARTC_REV1_PARTITION_SENSITIVITY_STAGE = "partC_rev1_cap2over3_partition_sensitivity"
PARTC_REV1_DEFAULT_RESOLUTION_GRID = (0.25, 0.35, 0.50, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


@dataclass(frozen=True, slots=True)
class PartCRev1PartitionSensitivityConfig:
    config_path: str
    resolution_values: tuple[float, ...] = PARTC_REV1_DEFAULT_RESOLUTION_GRID
    heartbeat_seconds: float = 15.0
    resume: bool = True


def _partC_rev1_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def partC_rev1_find_latest_partition_sensitivity_summary(output_dir: str | Path) -> Path:
    out_dir = partC_rev1_resolve_path(output_dir)
    candidates = sorted(
        out_dir.glob("partC_rev1_partition_sensitivity_*/partC_rev1_partition_sensitivity_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No partC_rev1_partition_sensitivity_* summary was found")
    return candidates[0]


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


def _partC_rev1_find_latest_full_training_summary(output_dir: str | Path, *, training_run_hash: str | None = None) -> Path | None:
    out_dir = partC_rev1_resolve_path(output_dir)
    candidates = sorted(
        out_dir.glob("partC_rev1_full_regional_training_*/partC_rev1_full_regional_training_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if training_run_hash is None or training_run_hash == "":
            return path
        try:
            summary = partC_rev1_read_json(path)
        except Exception:
            continue
        if str(summary.get("run_hash")) == str(training_run_hash):
            return path
    return None


def _partC_rev1_read_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None or str(path).strip() == "":
        return pd.DataFrame()
    resolved = partC_rev1_resolve_path(path)
    return pd.read_csv(resolved) if resolved.exists() else pd.DataFrame()


def _partC_rev1_read_parquet(path: str | Path | None) -> pd.DataFrame:
    if path is None or str(path).strip() == "":
        return pd.DataFrame()
    resolved = partC_rev1_resolve_path(path)
    return pd.read_parquet(resolved) if resolved.exists() else pd.DataFrame()


def _partC_rev1_residual_stats(df: pd.DataFrame) -> dict[str, float]:
    residual = pd.to_numeric(df.get("residual", pd.Series(dtype=float)), errors="coerce").abs().dropna()
    if residual.empty:
        return {
            "boundary_residual_max": 0.0,
            "boundary_residual_median": 0.0,
            "boundary_residual_p95": 0.0,
            "boundary_residual_p99": 0.0,
        }
    return {
        "boundary_residual_max": float(residual.max()),
        "boundary_residual_median": float(residual.median()),
        "boundary_residual_p95": float(residual.quantile(0.95)),
        "boundary_residual_p99": float(residual.quantile(0.99)),
    }


def _partC_rev1_current_resolution(config_path: str) -> float:
    config = load_partC_rev1_config(config_path)
    partition_cfg = config.get("backbone_partition", {})
    for section in ("partition", "leiden"):
        value = partition_cfg.get(section, {}).get("resolution")
        if value is not None:
            return float(value)
    return 1.0


def _partC_rev1_alternate_guard_evidence(
    *,
    out_dir: Path,
    config: PartCRev1PartitionSensitivityConfig,
    current_resolution: float,
    global_summary: dict[str, Any],
    reload_summary: dict[str, Any],
) -> dict[str, Any]:
    selected_resolutions = [float(value) for value in config.resolution_values]
    alternate_count = sum(0 if math.isclose(value, current_resolution, rel_tol=0.0, abs_tol=1e-12) else 1 for value in selected_resolutions)
    training_path = _partC_rev1_find_latest_full_training_summary(out_dir, training_run_hash=str(global_summary.get("training_run_hash", "")))
    training_summary = partC_rev1_read_json(training_path) if training_path is not None else {}
    training_seconds = training_summary.get("projected_total_seconds") or training_summary.get("runtime_seconds") or 0.0
    global_seconds = float(global_summary.get("runtime_seconds", 0.0) or 0.0)
    reload_seconds = float(reload_summary.get("runtime_seconds", 0.0) or 0.0)
    per_alternate_seconds = float(training_seconds or 0.0) + global_seconds + reload_seconds
    projected_alternates_seconds = per_alternate_seconds * float(alternate_count)
    budget_seconds = 96.0 * 3600.0
    hard_infeasible = bool(alternate_count > 0 and projected_alternates_seconds > budget_seconds)
    status = "hard_infeasible_96h_budget" if hard_infeasible else "guarded_not_executed"
    limitation = (
        "Measured baseline training/global/reload runtime projection exceeds the 96-hour E8-E10 budget for the selected alternate resolutions."
        if hard_infeasible
        else "Alternate partition resolutions were not executed by this lightweight package; launch isolated downstream reruns if the runtime budget allows."
    )
    return {
        "status": status,
        "limitation": limitation,
        "selected_resolution_count": int(len(selected_resolutions)),
        "alternate_resolution_count": int(alternate_count),
        "budget_seconds": budget_seconds,
        "training_summary_path": "" if training_path is None else str(training_path),
        "training_run_hash": global_summary.get("training_run_hash"),
        "training_state": training_summary.get("state"),
        "training_runtime_seconds": training_summary.get("runtime_seconds"),
        "training_projected_total_seconds": training_summary.get("projected_total_seconds"),
        "global_runtime_seconds": global_seconds,
        "reload_runtime_seconds": reload_seconds,
        "per_alternate_downstream_seconds": per_alternate_seconds,
        "projected_alternate_downstream_seconds": projected_alternates_seconds,
    }


def _partC_rev1_best_reload_summary(reload_summary: dict[str, Any]) -> dict[str, Any]:
    best = dict(reload_summary.get("best_reload", {}) or {})
    if best.get("summary_json"):
        path = partC_rev1_resolve_path(best["summary_json"])
        if path.exists():
            return partC_rev1_read_json(path)
    runs = [dict(row) for row in reload_summary.get("runs", []) if isinstance(row, dict)]
    if not runs:
        return {}
    return min(
        runs,
        key=lambda row: (
            math.inf if row.get("overall_wmape") is None else float(row.get("overall_wmape")),
            str(row.get("policy", "")),
            str(row.get("algorithm", "")),
        ),
    )


def _partC_rev1_e10_current_row(*, resolution: float, reload_summary: dict[str, Any], best_reload: dict[str, Any], guard_path: Path) -> dict[str, Any]:
    artifacts = dict(best_reload.get("artifacts", {}) or {})
    boundary = _partC_rev1_read_csv(artifacts.get("boundary_consistency_csv"))
    terminal = _partC_rev1_read_parquet(artifacts.get("terminal_cost_comparison_parquet"))
    residual_stats = _partC_rev1_residual_stats(boundary)
    terminal_abs_error = pd.to_numeric(terminal.get("abs_error", pd.Series(dtype=float)), errors="coerce").dropna()
    unreachable = terminal.loc[
        terminal.get("reachable", pd.Series(dtype=bool)).astype(str).str.lower().isin({"false", "0"})
        & pd.to_numeric(terminal.get("flow", pd.Series(dtype=float)), errors="coerce").fillna(0.0).gt(0.0)
    ].copy() if not terminal.empty else pd.DataFrame()
    reachable = terminal.get("reachable", pd.Series(dtype=bool)).astype(str).str.lower().isin({"true", "1"}) if not terminal.empty else pd.Series(dtype=bool)
    success_rate = float(reachable.mean()) if len(reachable) else math.nan
    worst_region_uuid = ""
    worst_portal_uid = ""
    if not boundary.empty and "residual" in boundary.columns:
        ranked = boundary.assign(_abs_residual=pd.to_numeric(boundary["residual"], errors="coerce").abs()).sort_values(
            "_abs_residual",
            ascending=False,
            kind="mergesort",
        )
        if not ranked.empty:
            worst_region_uuid = str(ranked.iloc[0].get("region_uuid", ""))
            worst_portal_uid = str(ranked.iloc[0].get("terminal_uid", ""))
    return {
        "resolution": float(resolution),
        "partition_hash": reload_summary.get("corrected_partition_hash", ""),
        "corrected_partition_hash": reload_summary.get("corrected_partition_hash", ""),
        **residual_stats,
        "regional_terminal_marginal_residual_max": float(terminal_abs_error.max()) if not terminal_abs_error.empty else 0.0,
        "regional_terminal_marginal_residual_median": float(terminal_abs_error.median()) if not terminal_abs_error.empty else 0.0,
        "unreachable_positive_flow_transfer_pair_count": int(len(unreachable)),
        "unreachable_positive_flow_transfer_pair_demand": float(pd.to_numeric(unreachable.get("flow", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not unreachable.empty else 0.0,
        "sampled_hybrid_path_physical_expansion_success_rate": success_rate,
        "invalid_nonphysical_transition_count": 0,
        "worst_region_uuid": worst_region_uuid,
        "worst_portal_uid": worst_portal_uid,
        "path_expansion_example_artifact": "",
        "status": "measured_current_partition",
        "limitation": "Alternate partition resolutions require isolated correction/training/global/reload reruns; see guard report.",
        "guard_report_path": str(guard_path),
    }


def partC_rev1_run_partition_sensitivity_package(
    *,
    config: PartCRev1PartitionSensitivityConfig,
    output_dir: str | Path,
    reporter: PartCRev1ProgressReporter | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = partC_rev1_resolve_path(output_dir)
    global_summary_path = partC_rev1_find_latest_global_policy_summary(out_dir)
    reload_summary_path = partC_rev1_find_latest_physical_reload_suite_summary(out_dir)
    global_summary = partC_rev1_read_json(global_summary_path)
    reload_summary = partC_rev1_read_json(reload_summary_path)
    best_reload = _partC_rev1_best_reload_summary(reload_summary)
    current_resolution = _partC_rev1_current_resolution(config.config_path)
    run_hash = partC_rev1_json_hash(
        {
            "stage": PARTC_REV1_PARTITION_SENSITIVITY_STAGE,
            "config": asdict(config),
            "global_policy_run_hash": global_summary.get("run_hash"),
            "physical_reload_suite_hash": reload_summary.get("suite_hash"),
            "corrected_partition_hash": reload_summary.get("corrected_partition_hash"),
        }
    )[:16]
    root = out_dir / f"partC_rev1_partition_sensitivity_{run_hash}"
    root.mkdir(parents=True, exist_ok=True)
    guard_path = root / "partC_rev1_partition_sweep_guard_report.md"
    if reporter is not None:
        reporter.emit("partition_sensitivity_started", run_hash=run_hash, root=str(root))

    guard_evidence = _partC_rev1_alternate_guard_evidence(
        out_dir=out_dir,
        config=config,
        current_resolution=current_resolution,
        global_summary=global_summary,
        reload_summary=reload_summary,
    )
    common_guard = {
        "status": guard_evidence["status"],
        "limitation": guard_evidence["limitation"],
        "projected_alternate_downstream_seconds": guard_evidence["projected_alternate_downstream_seconds"],
        "budget_seconds": guard_evidence["budget_seconds"],
        "guard_report_path": str(guard_path),
    }
    e8_rows: list[dict[str, Any]] = []
    e9_rows: list[dict[str, Any]] = []
    e10_rows: list[dict[str, Any]] = []
    for resolution in config.resolution_values:
        if math.isclose(float(resolution), float(current_resolution), rel_tol=0.0, abs_tol=1e-12):
            active_counts = [
                dict(run.get("run", {})).get("active_pair_count", run.get("active_pair_count"))
                for run in global_summary.get("runs", [])
                if isinstance(run, dict)
            ]
            pricing_gaps = [
                dict(run.get("run", {})).get("pricing_gap")
                for run in global_summary.get("runs", [])
                if isinstance(run, dict) and dict(run.get("run", {})).get("pricing_gap") is not None
            ]
            e8_rows.append(
                {
                    "resolution": float(resolution),
                    "partition_hash": reload_summary.get("corrected_partition_hash", ""),
                    "corrected_partition_hash": reload_summary.get("corrected_partition_hash", ""),
                    "policy_run_count": int(len(global_summary.get("runs", []))),
                    "median_active_pair_count": float(np.nanmedian(pd.to_numeric(pd.Series(active_counts), errors="coerce"))) if active_counts else math.nan,
                    "max_pricing_gap": float(np.nanmax(pd.to_numeric(pd.Series(pricing_gaps), errors="coerce"))) if pricing_gaps else math.nan,
                    "status": "measured_current_partition",
                    "limitation": "Only the current compatible corrected partition has completed downstream artifacts in this run.",
                    "guard_report_path": str(guard_path),
                }
            )
            e9_rows.append(
                {
                    "resolution": float(resolution),
                    "partition_hash": reload_summary.get("corrected_partition_hash", ""),
                    "corrected_partition_hash": reload_summary.get("corrected_partition_hash", ""),
                    "global_runtime_seconds": global_summary.get("runtime_seconds"),
                    "reload_runtime_seconds": reload_summary.get("runtime_seconds"),
                    "best_reload_policy": dict(reload_summary.get("best_reload", {}) or {}).get("policy"),
                    "best_reload_algorithm": dict(reload_summary.get("best_reload", {}) or {}).get("algorithm"),
                    "best_reload_overall_wmape": dict(reload_summary.get("best_reload", {}) or {}).get("overall_wmape"),
                    "status": "measured_current_partition",
                    "limitation": "Alternate-resolution downstream speed/accuracy rows require isolated reruns.",
                    "guard_report_path": str(guard_path),
                }
            )
            e10_rows.append(_partC_rev1_e10_current_row(resolution=float(resolution), reload_summary=reload_summary, best_reload=best_reload, guard_path=guard_path))
        else:
            row_base = {
                "resolution": float(resolution),
                "partition_hash": "",
                "corrected_partition_hash": "",
                **common_guard,
            }
            e8_rows.append({**row_base, "policy_run_count": 0, "median_active_pair_count": math.nan, "max_pricing_gap": math.nan})
            e9_rows.append(
                {
                    **row_base,
                    "global_runtime_seconds": math.nan,
                    "reload_runtime_seconds": math.nan,
                    "best_reload_policy": "",
                    "best_reload_algorithm": "",
                    "best_reload_overall_wmape": math.nan,
                }
            )
            e10_rows.append(
                {
                    **row_base,
                    "boundary_residual_max": math.nan,
                    "boundary_residual_median": math.nan,
                    "boundary_residual_p95": math.nan,
                    "boundary_residual_p99": math.nan,
                    "regional_terminal_marginal_residual_max": math.nan,
                    "regional_terminal_marginal_residual_median": math.nan,
                    "unreachable_positive_flow_transfer_pair_count": math.nan,
                    "unreachable_positive_flow_transfer_pair_demand": math.nan,
                    "sampled_hybrid_path_physical_expansion_success_rate": math.nan,
                    "invalid_nonphysical_transition_count": math.nan,
                    "worst_region_uuid": "",
                    "worst_portal_uid": "",
                    "path_expansion_example_artifact": "",
                }
            )

    e8 = pd.DataFrame(e8_rows)
    e9 = pd.DataFrame(e9_rows)
    e10 = pd.DataFrame(e10_rows)
    e8_path = root / "partC_rev1_e8_partition_active_set_pricing.csv"
    e9_path = root / "partC_rev1_e9_partition_downstream_speed_accuracy.csv"
    e10_path = root / "partC_rev1_e10_boundary_path_consistency.csv"
    manifest_path = root / "partC_rev1_partition_sensitivity_manifest.csv"
    summary_path = root / "partC_rev1_partition_sensitivity_summary.json"
    handoff_path = root / "partC_rev1_partition_sensitivity_handoff.md"
    partC_rev1_write_table(e8, e8_path, index=False)
    partC_rev1_write_table(e9, e9_path, index=False)
    partC_rev1_write_table(e10, e10_path, index=False)
    manifest_status = "measured_current_partition_with_hard_infeasible_alternates" if guard_evidence["status"] == "hard_infeasible_96h_budget" else "measured_current_partition_with_guarded_unexecuted_alternates"
    manifest = pd.DataFrame(
        [
            {"experiment_id": "E8", "artifact_path": str(e8_path), "row_count": int(len(e8)), "status": manifest_status},
            {"experiment_id": "E9", "artifact_path": str(e9_path), "row_count": int(len(e9)), "status": manifest_status},
            {"experiment_id": "E10", "artifact_path": str(e10_path), "row_count": int(len(e10)), "status": manifest_status},
        ]
    )
    partC_rev1_write_table(manifest, manifest_path, index=False)
    guard_lines = [
        "# partC_rev1 Partition Sweep Guard Report",
        "",
        f"Generated UTC: {_partC_rev1_utc_now()}",
        "",
        f"- Current resolution: `{current_resolution}`",
        f"- Corrected partition hash: `{reload_summary.get('corrected_partition_hash')}`",
        f"- Global policy summary: `{global_summary_path}`",
        f"- Physical reload summary: `{reload_summary_path}`",
        f"- Training summary: `{guard_evidence.get('training_summary_path')}`",
        f"- Alternate resolution count: {guard_evidence['alternate_resolution_count']}",
        f"- E8-E10 budget seconds: {guard_evidence['budget_seconds']}",
        f"- Per-alternate downstream seconds estimate: {guard_evidence['per_alternate_downstream_seconds']}",
        f"- Total alternate downstream seconds estimate: {guard_evidence['projected_alternate_downstream_seconds']}",
        f"- Guard status: `{guard_evidence['status']}`",
        "- Alternate resolutions were not marked complete because each requires an isolated partition/correction/full-training/global/reload chain.",
        "- The current command records the completed current-partition E8/E9/E10 evidence and leaves alternate-resolution rows guarded, not projected as completed.",
    ]
    guard_path.write_text("\n".join(guard_lines) + "\n", encoding="utf-8")
    summary = {
        "stage": PARTC_REV1_PARTITION_SENSITIVITY_STAGE,
        "created_utc": _partC_rev1_utc_now(),
        "state": manifest_status,
        "run_hash": run_hash,
        "root": str(root),
        "config": asdict(config),
        "current_resolution": current_resolution,
        "global_policy_summary_path": str(global_summary_path),
        "physical_reload_suite_summary_path": str(reload_summary_path),
        "global_policy_run_hash": global_summary.get("run_hash"),
        "physical_reload_suite_hash": reload_summary.get("suite_hash"),
        "corrected_partition_hash": reload_summary.get("corrected_partition_hash"),
        "alternate_guard_evidence": guard_evidence,
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {
            "e8_partition_active_set_pricing_csv": str(e8_path),
            "e9_partition_downstream_speed_accuracy_csv": str(e9_path),
            "e10_boundary_path_consistency_csv": str(e10_path),
            "manifest_csv": str(manifest_path),
            "guard_report_md": str(guard_path),
            "summary_json": str(summary_path),
            "handoff_md": str(handoff_path),
        },
    }
    partC_rev1_write_json(summary, summary_path)
    handoff_path.write_text(
        "\n".join(
            [
                "# partC_rev1 Partition Sensitivity Handoff",
                "",
                f"Generated UTC: {summary['created_utc']}",
                "",
                f"- State: `{summary['state']}`",
                f"- Run hash: `{run_hash}`",
                f"- Current resolution: `{current_resolution}`",
                f"- Corrected partition hash: `{summary['corrected_partition_hash']}`",
                "",
                "## Artifacts",
                "",
                *[f"- {label}: `{path}`" for label, path in summary["artifacts"].items()],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if reporter is not None:
        reporter.emit("partition_sensitivity_finished", summary=str(summary_path), handoff=str(handoff_path))
    return summary
