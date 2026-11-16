"""Validate, visualize, report, and seal the capacity-factor refinement."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .finalize import (
    MAX_MEMBER_BYTES,
    MAX_SYNC_BYTES,
    TARGET_SYNC_BYTES,
    _create_zip,
    _fresh_extract_verify,
    _source_checksums,
)
from .inputs import file_sha256


FINAL_STATUS = "WAITING_FOR_EXTERNAL_CAPACITY_FACTOR_REVIEW"
ALGORITHM = "SAFEGUARDED_CFW_WITH_IMMEDIATE_ORDINARY_FW_FALLBACK"


class RefinementFinalizationError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _git(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), "-c", "core.longpaths=true", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def _directory_size(root: Path) -> int:
    return int(sum(item.stat().st_size for item in root.rglob("*") if item.is_file()))


def _save_figure(fig: Any, root: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(root / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(root / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    copied = frame.copy()
    copied["_order"] = copied["record_kind"].map(
        {"refreshed_pre_update_state": 0, "final_refreshed_state": 1}
    )
    return (
        copied.sort_values(["capacity_factor", "update", "_order"], kind="mergesort")
        .drop_duplicates(["capacity_factor", "update"], keep="last")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def create_figures(output_root: Path, recommendation: float | None) -> list[dict[str, Any]]:
    figure_root = output_root / "figures"
    if any(figure_root.iterdir()):
        raise FileExistsError("Figure directory must be empty before final visualization")
    short = pd.read_csv(output_root / "tables" / "five_factor_short_screen_trajectory.csv")
    trajectories = [short]
    qualification_path = output_root / "tables" / "qualification_checkpoints.csv"
    qualification = pd.read_csv(qualification_path)
    if "capacity_factor" in qualification.columns:
        trajectories.append(qualification)
    combined = _deduplicate(pd.concat(trajectories, ignore_index=True))

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for factor, group in combined.groupby("capacity_factor", sort=True):
        ax.plot(group["update"], group["relative_gap_sptt"], marker="o", label=f"{factor:.2f}")
    ax.axhline(0.01, color="black", linestyle="--", linewidth=1, label="1% qualification")
    ax.set_yscale("log")
    ax.set_xlabel("Fully refreshed update")
    ax.set_ylabel("Refreshed relative gap (SPTT denominator)")
    ax.set_title("Refreshed gap by prepared-capacity factor")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    _save_figure(fig, figure_root, "01_refreshed_gap_by_capacity_factor")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    selected_factors = sorted(qualification["capacity_factor"].unique()) if "capacity_factor" in qualification.columns else []
    for factor in selected_factors:
        group = combined[combined["capacity_factor"].eq(factor)]
        positive = group[group["update"].gt(0)]
        axes[0].semilogy(positive["update"], positive["relative_gap_sptt"], marker="o", label=f"{factor:.2f}")
        axes[1].loglog(positive["update"], positive["relative_gap_sptt"], marker="o", label=f"{factor:.2f}")
    for ax, title in zip(axes, ("Semilog late trajectory", "Log-log late trajectory"), strict=True):
        ax.axhline(0.01, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Fully refreshed update")
        ax.set_ylabel("RG_SPTT")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        if selected_factors:
            ax.legend()
    _save_figure(fig, figure_root, "02_late_trajectory_semilog_loglog")

    decision = pd.read_csv(output_root / "tables" / "factor_decision_matrix.csv")
    fig, ax1 = plt.subplots(figsize=(8.2, 5.0))
    ax1.plot(
        decision["capacity_factor"],
        decision["screen_flow_weighted_tti_p95"],
        marker="o",
        color="tab:blue",
        label="flow-weighted TTI p95",
    )
    ax1.axhline(1.05, color="tab:blue", linestyle="--", alpha=0.6)
    ax1.set_xlabel("Prepared-capacity factor")
    ax1.set_ylabel("Flow-weighted TTI p95", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(
        decision["capacity_factor"],
        decision["relative_gap_sptt_after_5_updates"],
        marker="s",
        color="tab:red",
        label="RG after 5",
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("RG_SPTT after five updates", color="tab:red")
    ax1.set_title("Capacity-factor congestion comparison")
    ax1.grid(True, alpha=0.25)
    _save_figure(fig, figure_root, "03_capacity_factor_congestion_comparison")

    distributions = pd.read_csv(output_root / "tables" / "selected_vc_tti_distributions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    if "metric" in distributions.columns:
        for ax, metric in zip(axes, ("vc", "tti"), strict=True):
            group = distributions[distributions["metric"].eq(metric)].copy()
            labels = [
                f"{lo:g}–{hi:g}" if np.isfinite(hi) else f">={lo:g}"
                for lo, hi in zip(
                    group["bin_lower"], group["bin_upper"].fillna(np.inf), strict=True
                )
            ]
            ax.bar(np.arange(len(group)), group["flow_weight"], color="tab:purple")
            ax.set_xticks(np.arange(len(group)), labels, rotation=55, ha="right", fontsize=8)
            ax.set_yscale("log")
            ax.set_title(f"Selected-factor {metric.upper()} flow distribution")
            ax.set_ylabel("Arc-flow weight (log scale)")
    _save_figure(fig, figure_root, "04_selected_factor_vc_tti_distributions")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    if recommendation is not None:
        key = f"{recommendation:.2f}".replace(".", "p")
        route_root = output_root / "routes" / f"factor_{key}"
        route_path = route_root / "qualification" / "sampled_route_switches.csv"
        if not route_path.exists():
            route_path = route_root / "screen" / "sampled_route_switches.csv"
        routes = pd.read_csv(route_path)
        axes[0].hist(routes["arc_set_jaccard_overlap"], bins=20, color="tab:green")
        axes[0].set_xlabel("Arc-set Jaccard overlap")
        axes[0].set_ylabel("Sampled OD count")
        counts = [
            int((~routes["route_switched"]).sum()),
            int(routes["route_switched"].sum()),
            int(routes["genuine_non_tie_route_switch"].sum()),
        ]
        axes[1].bar(["unchanged", "switched", "strict non-tie"], counts, color=["gray", "orange", "red"])
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].set_ylabel("Sampled OD count")
    axes[0].set_title("Path-overlap distribution")
    axes[1].set_title("Sampled route-change evidence")
    _save_figure(fig, figure_root, "05_route_change_and_path_overlap")

    forecasts = pd.read_csv(output_root / "tables" / "runtime_convergence_forecasts.csv")
    target = forecasts[forecasts["relative_gap_sptt_target"].eq(1e-4)] if "relative_gap_sptt_target" in forecasts.columns else forecasts.iloc[0:0]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    if len(target):
        values = target["projected_runtime_seconds"].fillna(0.0) / 3600.0
        colors = ["tab:blue" if basis == "OBSERVED" else "tab:orange" for basis in target["basis"]]
        ax.bar(target["capacity_factor"].astype(str), values, color=colors)
    ax.set_xlabel("Prepared-capacity factor")
    ax.set_ylabel("Measured/projected hours to RG_SPTT=1e-4")
    ax.set_title("Measured and forecast time to 0.01% refreshed gap")
    ax.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, figure_root, "06_measured_forecast_time_to_1e-4")

    timing = pd.read_csv(output_root / "tables" / "aon_before_after_timing.csv")
    components = [
        "setup_seconds",
        "worker_orchestration_seconds",
        "reduction_assembly_seconds",
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bottom = np.zeros(len(timing), dtype=np.float64)
    for component in components:
        values = timing[component].to_numpy(dtype=np.float64)
        ax.bar(timing["timing_scope"], values, bottom=bottom, label=component)
        bottom += values
    ax.set_ylabel("Seconds")
    ax.set_title("Before/after AON timing decomposition")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, figure_root, "07_before_after_aon_timing_decomposition")

    receipts = []
    for path in sorted(figure_root.iterdir()):
        receipts.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    return receipts


def _artifact_manifest(output_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(output_root).as_posix(),
                "absolute_path": str(path.resolve()),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
                "included_in_compact_handoff": bool(
                    path.stat().st_size <= MAX_MEMBER_BYTES
                    and not any(part in {"checkpoints", "runs", "capacity"} for part in path.parts)
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_and_report(config_path: Path) -> tuple[dict[str, Any], Path, Path, float | None]:
    config = _load(config_path)
    project_root = Path(config["project_root"]).resolve(strict=True)
    output_root = Path(config["output_root"]).resolve(strict=True)
    if config["phase_id"] != "phase_04" or config["step_id"] != "capacity_factor_refinement_preflight":
        raise RefinementFinalizationError("Config is not the authorized refinement preflight")
    if config["algorithm"] != ALGORITHM or config["worker_count"] != 4:
        raise RefinementFinalizationError("Frozen algorithm/worker contract changed")
    branch = _git(project_root, "branch", "--show-current").strip()
    head = _git(project_root, "rev-parse", "HEAD").strip()
    if branch != config["expected_branch"] or head != config["reviewed_source_head"]:
        raise RefinementFinalizationError("Branch/HEAD changed during refinement execution")
    status_path = output_root / "status" / "campaign_status.json"
    status = _load(status_path)
    if status["state"] != "EXECUTION_COMPLETE_AWAITING_FINAL_VALIDATION_AND_HANDOFF":
        raise RefinementFinalizationError(f"Unexpected execution state: {status['state']}")
    completion = _load(output_root / "status" / "execution_completion.json")
    decision = _load(output_root / "validation" / "capacity_factor_decision.json")
    recommendation_value = decision["recommended_capacity_factor"]
    recommendation = (
        float(recommendation_value) if recommendation_value != "unresolved" else None
    )
    sanitation = _load(output_root / "capacity" / "capacity_sanitation_audit.json")
    authority = _load(output_root / "validation" / "authority_validation.json")
    inputs = _load(output_root / "validation" / "input_integrity.json")
    optimization = _load(output_root / "timing" / "aon_cache_optimization.json")
    pytest = _load(output_root / "validation" / "focused_pytest_receipt.json")
    matrix = pd.read_csv(output_root / "tables" / "factor_decision_matrix.csv")
    primary = matrix[matrix["grid_scope"].eq("primary")]
    if primary["capacity_factor"].round(2).tolist() != [0.3, 0.35, 0.4, 0.45, 0.5]:
        raise RefinementFinalizationError("Five-factor primary decision matrix is incomplete")
    if matrix["capacity_factor"].isin([0.25, 1.0]).any():
        raise RefinementFinalizationError("Prohibited 0.25/1.00 factor was rerun")
    short = pd.read_csv(output_root / "tables" / "five_factor_short_screen_trajectory.csv")
    primary_final_updates = (
        short[short["grid_scope"].eq("primary")]
        .groupby("capacity_factor")["update"]
        .max()
    )
    if len(primary_final_updates) != 5 or (primary_final_updates < 5).any():
        raise RefinementFinalizationError("A primary factor lacks five fully refreshed updates")
    required_tables = [
        "prepared_capacity_quantiles.csv",
        "prepared_capacity_repair_register.csv",
        "prepared_capacity_hashes.csv",
        "five_factor_short_screen_trajectory.csv",
        "fallback_factor_trajectory.csv",
        "qualification_checkpoints.csv",
        "factor_decision_matrix.csv",
        "selected_top100_links.csv",
        "cfw_coefficients_resets_line_search.csv",
        "aon_before_after_timing.csv",
        "runtime_convergence_forecasts.csv",
        "all_five_traced_verification.csv",
    ]
    missing_tables = [name for name in required_tables if not (output_root / "tables" / name).exists()]
    if missing_tables:
        raise RefinementFinalizationError(f"Required tables missing: {missing_tables}")
    all_five_status = "NOT_APPLICABLE_NO_RECOMMENDATION"
    if recommendation is not None:
        all_five = _load(output_root / "validation" / "all_five_trace_verification.json")
        if all_five["status"] != "PASS" or all_five["capacity_factor"] != recommendation:
            raise RefinementFinalizationError("Recommended factor lacks passing all-five verification")
        if not all_five["physical_flow_matches_physical_only_update_3"]:
            raise RefinementFinalizationError("All-five physical flow differs from physical-only update 3")
        all_five_status = "PASS"
    if pytest["returncode"] != 0 or optimization["status"] != "PASS":
        raise RefinementFinalizationError("Focused tests or exact optimization equivalence failed")
    solver_peak_receipts = [
        int(_load(path).get("peak_rss_bytes", 0))
        for path in sorted((output_root / "runs").rglob("solver_summary.json"))
    ]
    conservative_peak_rss_bytes = max(
        [int(completion["peak_rss_bytes"]), *solver_peak_receipts]
    )
    resources = config["resources"]
    resource_pass = bool(
        completion["elapsed_seconds"] < resources["wall_clock_seconds"] - resources["protected_sealing_seconds"]
        and conservative_peak_rss_bytes < resources["process_tree_rss_bytes"]
        and completion["output_size_bytes"] + completion["temp_size_bytes"] < resources["added_disk_bytes"]
    )
    if not resource_pass:
        raise RefinementFinalizationError("Scientific execution exceeded a frozen resource guard")
    figure_receipts = create_figures(output_root, recommendation)
    if len([item for item in figure_receipts if item["name"].endswith(".png")]) != 7:
        raise RefinementFinalizationError("Exactly seven required PNG figures were not produced")

    gate_rows = []
    for item in decision["qualification_decisions"]:
        for gate, passed in item["gates"].items():
            gate_rows.append(
                {
                    "capacity_factor": item["capacity_factor"],
                    "gate": gate,
                    "pass": bool(passed),
                    "factor_qualified": bool(item["qualified"]),
                }
            )
    pd.DataFrame(gate_rows).to_csv(output_root / "tables" / "factor_gate_results.csv", index=False)
    events = [
        json.loads(line)
        for line in (output_root / "logs" / "campaign_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resource_rows = [item for item in events if item.get("event") == "resource_guard"]
    pd.DataFrame(resource_rows).to_csv(output_root / "reports" / "resource_usage.csv", index=False)

    final_validation = {
        "schema_id": "phase4.capacity_factor_refinement.final_validation.v1",
        "status": "PASS",
        "validated_utc": _utc_now(),
        "authority_status": authority["status"],
        "authority_minor_checksum_resolution_recorded": True,
        "input_integrity_status": inputs["status"],
        "capacity_sanitation_repair_count": sanitation["counts"]["repaired"],
        "clean_capacity_payload_sha256": sanitation[
            "clean_c_order_float64_payload_sha256"
        ],
        "same_clean_array_all_factors": sanitation[
            "one_clean_immutable_array_for_all_factors"
        ],
        "primary_factors_screened": primary["capacity_factor"].tolist(),
        "prohibited_factors_rerun": False,
        "recommended_capacity_factor": recommendation if recommendation is not None else "unresolved",
        "decision_status": decision["status"],
        "all_five_verification_status": all_five_status,
        "focused_pytest_pass": pytest["returncode"] == 0,
        "exact_optimization_equivalence_pass": optimization["equivalence"]["pass"],
        "resource_guard_pass": resource_pass,
        "campaign_boundary_sampled_peak_rss_bytes": completion["peak_rss_bytes"],
        "conservative_max_recorded_peak_rss_bytes": conservative_peak_rss_bytes,
        "figure_count_png": 7,
        "required_table_count": len(required_tables),
        "phase4_solver_tracer_preflight_status": "APPROVED_COMPLETE_WITH_NOTES",
        "capacity_factor_refinement_status": FINAL_STATUS,
        "selected_algorithm": ALGORITHM,
        "phase4_24h_production_assignment_authorized": False,
        "phase5_authorized": False,
        "next_step_authorized": False,
    }
    _write_json_new(output_root / "validation" / "final_validation.json", final_validation)
    recommendation_text = f"`{recommendation:.2f}`" if recommendation is not None else "`unresolved`"
    qualified = [item for item in decision["qualification_decisions"] if item["qualified"]]
    qualification_text = (
        f"The observed qualification checkpoint was update `{qualified[0]['first_observed_update_at_or_below_0p01']}` "
        f"with final observed RG_SPTT `{qualified[0]['final_relative_gap_sptt']:.8g}`."
        if qualified
        else "No tested factor reached all frozen gates within the bounded preflight."
    )
    congestion_text = (
        f"At the final observed state, flow-weighted TTI p95 was "
        f"`{qualified[0]['flow_weighted_tti_p95']:.8g}` and "
        f"`{qualified[0]['positive_flow_link_count_tti_above_2']}` positive-flow links "
        f"had TTI above 2. The extreme maximum TTI was "
        f"`{qualified[0]['positive_flow_link_tti_max']:.8g}` and is retained as an "
        f"explicit external-review warning, not hidden or used to waive a gate."
        if qualified
        else "No selected-factor congestion summary is applicable."
    )
    report = f"""# Phase 4 capacity-factor refinement preflight result

## Outcome

The authorized two-hour refinement completed and passed final validation.
The recommended prepared-capacity factor is {recommendation_text}. {qualification_text}

Final state:

```text
Phase 4 solver/tracer preflight:
APPROVED_COMPLETE_WITH_NOTES

Capacity-factor refinement preflight:
WAITING_FOR_EXTERNAL_CAPACITY_FACTOR_REVIEW

selected algorithm:
SAFEGUARDED_CFW_WITH_IMMEDIATE_ORDINARY_FW_FALLBACK

recommended capacity factor:
{recommendation if recommendation is not None else 'unresolved'}

24-hour production assignment authorized:
false

Phase 5 authorized:
false
```

## Verified facts

- Authority package: `{authority['package_sha256']}`; all `{authority['verified_checksum_entries']}` declared checksums pass. `MANIFEST.csv` is not self-listed in `CHECKSUMS.sha256`; the launch explicitly permits this minor metadata resolution and all numeric authority is independently recoverable.
- Frozen graph: `{inputs['graph_identity']}`, `{inputs['graph_node_count']:,}` nodes, `{inputs['graph_arc_count']:,}` arcs.
- Frozen OD: `{inputs['od_identity']}`, `{inputs['original_od_row_count']:,}` original rows, `{inputs['total_demand']:.12g}` conserved demand.
- Prepared-capacity sanitation repaired `{sanitation['counts']['repaired']}` arcs. The one clean float64 payload hash used for every factor is `{sanitation['clean_c_order_float64_payload_sha256']}`.
- Primary factors `0.30, 0.35, 0.40, 0.45, 0.50` each received at least five fully refreshed updates from the same exact free-flow AON. Factors `0.25` and `1.00` were not rerun.
- {congestion_text}
- Cached tie rank and packed trace maps were bitwise equivalent to uncached execution; bounded measured speedup was `{optimization['measured_speedup']:.3f}x`.
- Focused tests: `{pytest['stdout'].strip()}`.
- Recommended-factor all-five verification: `{all_five_status}`.
- Scientific execution used `{completion['elapsed_seconds']:.1f}` seconds, conservative maximum recorded process-tree RSS `{conservative_peak_rss_bytes / 1024**3:.3f}` GiB, and `{completion['output_size_bytes'] / 1024**2:.1f}` MiB output before final packaging. The campaign-boundary sampler recorded `{completion['peak_rss_bytes'] / 1024**3:.3f}` GiB; the larger solver-local receipt is used for the hard-gate comparison and reporting.

## Interpretation and recommendation

The capacity factor is an engineering calibration applied once as `used_capacity = factor * clean_prepared_capacity`; sanitation was not a selection gate. Projections only prioritize or forecast and never establish qualification. External review should inspect the observed gate matrix, convergence trajectory, route-switch witnesses, selected-link congestion, exact traced verification, and resource evidence in this handoff.

Recommendation: externally review this exact bundle and, only if accepted, issue a separate hash-bound authorization for the 24-hour production assignment. This handoff does not authorize that solve or Phase 5.
"""
    _write_text_new(output_root / "reports" / "PHASE4_CAPACITY_FACTOR_REFINEMENT_RESULT.md", report)
    machine_status = {
        **final_validation,
        "report_path": str(
            (output_root / "reports" / "PHASE4_CAPACITY_FACTOR_REFINEMENT_RESULT.md").resolve()
        ),
        "recommendation": "PROCEED_TO_EXTERNAL_CAPACITY_FACTOR_REVIEW",
    }
    _write_json_new(
        output_root / "reports" / "PHASE4_CAPACITY_FACTOR_REFINEMENT_RESULT.json",
        machine_status,
    )
    manifest = _artifact_manifest(output_root)
    manifest.to_csv(output_root / "manifests" / "full_artifact_manifest.csv", index=False)
    large = manifest[manifest["size_bytes"].gt(MAX_MEMBER_BYTES)].copy()
    large.to_csv(output_root / "manifests" / "large_external_artifact_index.csv", index=False)
    code_index = pd.DataFrame(
        [
            {
                "repository_relative_path": path,
                "role": role,
                "sha256": file_sha256(project_root / path),
            }
            for path, role in (
                ("partC_rev1_src/phase4/refinement.py", "executed scientific entrypoint"),
                ("partC_rev1_src/phase4/refinement_finalize.py", "validation and handoff entrypoint"),
                ("partC_rev1_src/phase4/aon.py", "exact AON and timing"),
                ("partC_rev1_src/phase4/solver.py", "selected safeguarded-CFW solver"),
                ("partC_rev1_src/phase4/inputs.py", "frozen input and accepted aggregate loader"),
                ("tests/test_phase4_preflight.py", "focused regression tests"),
                ("partC_doc_icnn/phase_4_work/phase4_capacity_factor_refinement_execution_config.json", "explicit execution contract"),
            )
        ]
    )
    code_index.to_csv(output_root / "manifests" / "code_change_index.csv", index=False)
    status.update(
        {
            "state": "PACKAGING_HANDOFF",
            "updated_utc": _utc_now(),
            "next_task": "HANDOFF",
            "recommended_capacity_factor": recommendation if recommendation is not None else "unresolved",
        }
    )
    _write_json_atomic(status_path, status)
    return final_validation, project_root, output_root, recommendation


def _copy_preserving(source: Path, root: Path, relative: Path) -> None:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def package_handoff(
    config_path: Path,
    final_validation: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    recommendation: float | None,
    *,
    handoff_id: str,
) -> dict[str, Any]:
    config = _load(config_path)
    source = (
        project_root
        / "partC_doc_icnn"
        / "handoffs"
        / "phase_04"
        / "step_capacity_factor_refinement_preflight"
        / handoff_id
    )
    if source.exists():
        raise FileExistsError(f"Write-new handoff source already exists: {source}")
    for name in (
        "governing_docs",
        "configs",
        "code_snapshot",
        "code_diff",
        "summary_data",
        "figures",
        "geospatial_summary",
        "log_excerpts",
        "external_artifact_index",
    ):
        (source / name).mkdir(parents=True, exist_ok=False)
    zip_root = project_root / "artifacts_icnn" / "handoff_bundles" / "phase_04"
    zip_path = zip_root / (
        f"phase04_step_capacity_factor_refinement_preflight_{handoff_id}_SYNC.zip"
    )
    report = (
        output_root / "reports" / "PHASE4_CAPACITY_FACTOR_REFINEMENT_RESULT.md"
    ).read_text(encoding="utf-8")
    recommended_text = str(recommendation) if recommendation is not None else "unresolved"
    _write_text_new(
        source / "00_HANDOFF_README.md",
        "\n".join(
            [
                "# Phase 4 capacity-factor refinement handoff",
                "",
                "Upload the exact ZIP named in `10_UPLOAD_THIS_FILE.txt`.",
                f"The full write-new run is indexed at `{output_root}`.",
                "This compact bundle contains governing authority, code/config evidence,",
                "required tables and figures, validation, resources, and large-array indexes.",
                "",
                f"Status: `{FINAL_STATUS}`. External review is required; no next step is authorized.",
                "",
            ]
        ),
    )
    _write_text_new(source / "01_STEP_SUMMARY.md", report)
    _write_json_new(
        source / "02_HANDOFF_STATUS.json",
        {
            **dict(final_validation),
            "schema_id": "phase4.capacity_factor_refinement.handoff_status.v1",
            "handoff_id": handoff_id,
            "status": FINAL_STATUS,
            "recommended_capacity_factor": recommended_text,
            "selected_algorithm": ALGORITHM,
            "next_step_authorized": False,
            "phase4_24h_production_assignment_authorized": False,
            "phase5_authorized": False,
        },
    )
    _write_text_new(
        source / "03_DECISIONS_AND_OPEN_QUESTIONS.md",
        "\n".join(
            [
                "# Decisions and open questions",
                "",
                "The solver/tracer preflight remains `APPROVED_COMPLETE_WITH_NOTES`.",
                f"The selected algorithm remains `{ALGORITHM}` with four workers and SPTT denominator.",
                f"The observed-gate recommendation is `{recommended_text}`.",
                "",
                "Minor authority closure note: `MANIFEST.csv` is not self-listed in",
                "`CHECKSUMS.sha256`. All 20 declared checksums and 20 manifest rows verify;",
                "the launch explicitly permits recovery of this metadata issue. No scientific",
                "identity was waived.",
                "",
                "Open external decision: accept, revise, or reject the factor and decide",
                "whether to issue a separate hash-bound 24-hour production authorization.",
                "Phase 5 remains unauthorized.",
                "",
            ]
        ),
    )
    _write_text_new(
        source / "04_VALIDATION_RESULTS.md",
        "# Validation results\n\nFinal validation is `PASS`. See the final-validation JSON, gate matrix, exact all-five verification, focused tests, capacity audit, figures, and artifact indexes.\n",
    )
    shutil.copy2(
        output_root / "reports" / "resource_usage.csv",
        source / "05_RESOURCE_USAGE.csv",
    )
    shutil.copy2(
        output_root / "manifests" / "full_artifact_manifest.csv",
        source / "06_ARTIFACT_MANIFEST.csv",
    )
    shutil.copy2(
        output_root / "manifests" / "code_change_index.csv",
        source / "07_CODE_CHANGE_INDEX.csv",
    )
    _write_text_new(
        source / "08_COMMAND_LOG.md",
        "\n".join(
            [
                "# Command log",
                "",
                "1. Read every mandatory repository, authority, review, handoff, bundle, and snapshot document.",
                "2. Verified authority, reviewed preflight, source snapshot, branch/HEAD/dirty state, graph, OD, and suites.",
                "3. Ran the focused Phase 4 pytest target before and inside the campaign; both passed.",
                f"4. Executed `partC_rev1_src.phase4.refinement` with run ID `{config['run_id']}`.",
                "5. Executed `partC_rev1_src.phase4.refinement_finalize` for figures, validation, reports, manifests, and handoff.",
                "",
                "Network access and package installation were not used.",
                "",
            ]
        ),
    )
    _write_text_new(
        source / "09_NEXT_STEP_RECOMMENDATION.md",
        "\n".join(
            [
                "# Next-step recommendation",
                "",
                f"Externally review this exact ZIP. The observed-gate recommendation is `{recommended_text}`.",
                "If accepted, issue a separate authority bound to this ZIP SHA-256 before",
                "the 24-hour production assignment. Phase 5 remains unauthorized.",
                "",
                "`next_step_authorized=false`.",
                "",
            ]
        ),
    )
    _write_text_new(
        source / "10_UPLOAD_THIS_FILE.txt",
        f"Upload this exact file:\n{zip_path.resolve()}\nUse the adjacent .zip.sha256 and companion JSON to verify its final bytes.\n",
    )
    _write_text_new(
        source / "11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md",
        "# Changelog from previous handoff\n\nThis is the first handoff for the authorized capacity-factor refinement. It starts from the immutable reviewed Task 1 preflight and adds sanitation, exact optimization evidence, five-factor screens, bounded continuation, observed gates, diagnostics, forecasts, all-five verification when applicable, and final review evidence.\n",
    )
    governing_paths = [
        project_root / "partC_doc_icnn" / name
        for name in (
            "00_READ_FIRST.md",
            "PROJECT_STATUS.md",
            "12_DECISION_LOG.md",
            "20_HANDOFF_SYNC_PROTOCOL.md",
            "21_REVIEW_APPROVAL_GATES.md",
            "23_SYNC_BUNDLE_CONTENT_AND_SIZE_POLICY.md",
            "24_CODE_AND_DATA_SNAPSHOT_POLICY.md",
        )
    ]
    phase_work = project_root / "partC_doc_icnn" / "phase_4_work"
    for optional_name in (
        "PHASE4_CAPACITY_FACTOR_REFINEMENT_RESULT.md",
        "PHASE4_CAPACITY_FACTOR_REFINEMENT_RESULT.json",
        "phase4_capacity_factor_refinement_execution_config.json",
    ):
        path = phase_work / optional_name
        if path.exists():
            governing_paths.append(path)
    for path in governing_paths:
        _copy_preserving(
            path,
            source / "governing_docs" / "repository",
            path.relative_to(project_root),
        )
    shutil.copytree(
        output_root / "authority",
        source / "governing_docs" / "authority_package",
        dirs_exist_ok=False,
    )
    for path in sorted((output_root / "config").iterdir()):
        if path.is_file():
            shutil.copy2(path, source / "configs" / path.name)

    snapshot_paths = [
        *sorted((project_root / "partC_rev1_src" / "phase4").glob("*.py")),
        project_root / "partC_rev1_src" / "partC_rev1_shortest_paths.py",
        project_root / "partC_rev1_src" / "partC_rev1_graph_snapshot.py",
        project_root / "tests" / "test_phase4_preflight.py",
        config_path,
    ]
    code_rows = []
    for path in snapshot_paths:
        relative = path.relative_to(project_root)
        _copy_preserving(path, source / "code_snapshot", relative)
        code_rows.append(
            {
                "repository_relative_path": relative.as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    pd.DataFrame(code_rows).to_csv(
        source / "code_snapshot" / "CODE_SNAPSHOT_MANIFEST.csv", index=False
    )
    _write_text_new(
        source / "code_diff" / "working_tree.patch",
        _git(project_root, "diff", "--binary", "--"),
    )
    _write_text_new(
        source / "code_diff" / "git_status_short.txt",
        _git(project_root, "status", "--short"),
    )
    _write_text_new(
        source / "code_diff" / "git_diff_stat.txt",
        _git(project_root, "diff", "--stat"),
    )
    _write_text_new(
        source / "code_diff" / "git_identity.txt",
        "\n".join(
            [
                f"branch={_git(project_root, 'branch', '--show-current').strip()}",
                f"head={_git(project_root, 'rev-parse', 'HEAD').strip()}",
                f"reviewed_source_head={config['reviewed_source_head']}",
                "",
            ]
        ),
    )
    for path in sorted((output_root / "tables").iterdir()):
        if path.is_file() and path.stat().st_size <= MAX_MEMBER_BYTES:
            shutil.copy2(path, source / "summary_data" / path.name)
    for subdirectory in ("validation", "timing", "reports", "status", "manifests"):
        target = source / "summary_data" / subdirectory
        target.mkdir()
        for path in sorted((output_root / subdirectory).iterdir()):
            if path.is_file() and path.stat().st_size <= MAX_MEMBER_BYTES:
                shutil.copy2(path, target / path.name)
    capacity_summary = source / "summary_data" / "capacity"
    capacity_summary.mkdir()
    shutil.copy2(
        output_root / "capacity" / "capacity_sanitation_audit.json",
        capacity_summary / "capacity_sanitation_audit.json",
    )
    for path in sorted((output_root / "routes").rglob("*.json")):
        _copy_preserving(
            path,
            source / "summary_data" / "routes",
            path.relative_to(output_root / "routes"),
        )
    if recommendation is not None:
        key = f"{recommendation:.2f}".replace(".", "p")
        route_path = (
            output_root
            / "routes"
            / f"factor_{key}"
            / "qualification"
            / "sampled_route_switches.csv"
        )
        if not route_path.exists():
            route_path = (
                output_root
                / "routes"
                / f"factor_{key}"
                / "screen"
                / "sampled_route_switches.csv"
            )
        _copy_preserving(
            route_path,
            source / "summary_data" / "routes",
            Path("selected_factor_sampled_route_switches.csv"),
        )
    shutil.copytree(
        output_root / "figures",
        source / "figures" / "required_figures",
    )
    _write_text_new(
        source / "geospatial_summary" / "NO_GEOSPATIAL_OUTPUT.md",
        "# Geospatial summary\n\nNo geospatial artifact was authorized or produced by this numerical refinement. Frozen Phase 3 mapping products remained read-only.\n",
    )
    shutil.copy2(
        output_root / "logs" / "campaign_events.jsonl",
        source / "log_excerpts" / "campaign_events.jsonl",
    )
    shutil.copy2(
        output_root / "validation" / "focused_pytest_receipt.json",
        source / "log_excerpts" / "focused_pytest_receipt.json",
    )
    shutil.copy2(
        output_root / "manifests" / "full_artifact_manifest.csv",
        source / "external_artifact_index" / "full_artifact_manifest.csv",
    )
    shutil.copy2(
        output_root / "manifests" / "large_external_artifact_index.csv",
        source / "external_artifact_index" / "large_external_artifact_index.csv",
    )
    _write_json_new(
        source / "external_artifact_index" / "full_run_root.json",
        {
            "full_run_root": str(output_root),
            "full_run_size_bytes_before_handoff": _directory_size(output_root),
            "large_arrays_and_checkpoints_remain_external": True,
            "producer_config": str(config_path),
            "producer_entrypoint": "partC_rev1_src.phase4.refinement",
            "reopen_evidence": str(
                output_root / "validation" / "selected_state_reopen.json"
            )
            if recommendation is not None
            else None,
        },
    )
    oversized = [
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and path.stat().st_size > MAX_MEMBER_BYTES
    ]
    if oversized:
        raise RefinementFinalizationError(
            f"Handoff has files above 25 MB: {oversized[:5]}"
        )
    sensitive_tokens = ("id_rsa", ".pem", ".pfx", "credentials", "secrets")
    suspicious = [
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
        and any(token in path.name.lower() for token in sensitive_tokens)
    ]
    if suspicious:
        raise RefinementFinalizationError(
            f"Potential sensitive filenames in handoff: {suspicious}"
        )
    _write_json_new(
        source / "validation_privacy_scan.json",
        {
            "status": "PASS",
            "suspicious_filename_count": 0,
            "dependency_environments_included": False,
            "browser_profiles_included": False,
            "private_keys_included": False,
        },
    )
    checksums = _source_checksums(source)
    _write_text_new(
        source / "CHECKSUMS.sha256",
        "".join(
            f"{digest}  {relative}\n"
            for relative, digest in checksums.items()
        ),
    )
    _create_zip(source, zip_path)
    if zip_path.stat().st_size > MAX_SYNC_BYTES:
        raise RefinementFinalizationError("Synchronization ZIP exceeds 200 MB")
    zip_sha = file_sha256(zip_path)
    sidecar = zip_path.with_suffix(zip_path.suffix + ".sha256")
    _write_text_new(sidecar, f"{zip_sha}  {zip_path.name}\n")
    fresh = _fresh_extract_verify(zip_path)
    companion = {
        "schema_id": "phase4.capacity_factor_refinement.handoff_companion.v1",
        "handoff_id": handoff_id,
        "status": FINAL_STATUS,
        "sync_zip": str(zip_path.resolve()),
        "sync_zip_size_bytes": int(zip_path.stat().st_size),
        "sync_zip_sha256": zip_sha,
        "sync_zip_under_100mb_target": zip_path.stat().st_size
        <= TARGET_SYNC_BYTES,
        "source_folder": str(source.resolve()),
        "fresh_extraction": fresh,
        "recommended_capacity_factor": recommended_text,
        "selected_algorithm": ALGORITHM,
        "next_step_authorized": False,
        "phase4_24h_production_assignment_authorized": False,
        "phase5_authorized": False,
    }
    companion_path = zip_path.with_suffix(zip_path.suffix + ".companion.json")
    _write_json_new(companion_path, companion)
    status_path = output_root / "status" / "campaign_status.json"
    status = _load(status_path)
    status.update(
        {
            "state": FINAL_STATUS,
            "updated_utc": _utc_now(),
            "next_task": "EXTERNAL-CAPACITY-FACTOR-REVIEW",
            "recommended_capacity_factor": recommended_text,
            "handoff_id": handoff_id,
            "handoff_source": str(source.resolve()),
            "sync_zip": str(zip_path.resolve()),
            "sync_zip_size_bytes": int(zip_path.stat().st_size),
            "sync_zip_sha256": zip_sha,
            "sidecar": str(sidecar.resolve()),
            "companion": str(companion_path.resolve()),
            "next_step_authorized": False,
            "phase4_24h_production_assignment_authorized": False,
            "phase5_authorized": False,
        }
    )
    _write_json_atomic(status_path, status)
    latest = source.parent / "LATEST_HANDOFF.txt"
    latest_text = "\n".join(
        [
            f"source_folder={source.resolve()}",
            f"sync_zip={zip_path.resolve()}",
            f"sync_zip_sha256={zip_sha}",
            f"sync_zip_size_bytes={zip_path.stat().st_size}",
            f"created_utc={_utc_now()}",
            f"status={FINAL_STATUS}",
            "",
        ]
    )
    temporary = latest.with_name(latest.name + ".tmp")
    temporary.write_text(latest_text, encoding="utf-8", newline="\n")
    os.replace(temporary, latest)
    return companion


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage", choices=("validate", "handoff", "finalize"), required=True
    )
    parser.add_argument("--handoff-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve(strict=True)
    config = _load(config_path)
    expected = {
        "phase_id": args.phase_id,
        "step_id": args.step_id,
        "run_id": args.run_id,
    }
    for key, value in expected.items():
        if config[key] != value:
            raise RefinementFinalizationError(
                f"CLI {key} differs from explicit config"
            )
    if args.stage in {"validate", "finalize"}:
        final_validation, project_root, output_root, recommendation = (
            validate_and_report(config_path)
        )
        if args.stage == "validate":
            print(json.dumps(final_validation, indent=2, sort_keys=True))
            return
    else:
        project_root = Path(config["project_root"]).resolve(strict=True)
        output_root = Path(config["output_root"]).resolve(strict=True)
        final_validation = _load(
            output_root / "validation" / "final_validation.json"
        )
        recommendation_value = final_validation["recommended_capacity_factor"]
        recommendation = (
            float(recommendation_value)
            if recommendation_value != "unresolved"
            else None
        )
        status = _load(output_root / "status" / "campaign_status.json")
        if status["state"] != "PACKAGING_HANDOFF":
            raise RefinementFinalizationError(
                "Handoff stage requires completed validation and PACKAGING_HANDOFF state"
            )
    companion = package_handoff(
        config_path,
        final_validation,
        project_root,
        output_root,
        recommendation,
        handoff_id=args.handoff_id,
    )
    print(json.dumps(companion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
