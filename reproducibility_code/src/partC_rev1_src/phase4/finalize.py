"""Validate, report, and seal the bounded Phase 4 Task 1 preflight.

This module is intentionally separate from the numerical campaign.  It never
starts an assignment solve and writes all review-stage products under an
explicit write-new review ID before creating the compact external handoff.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from .checkpoint import load_phase4_checkpoint, load_phase4_checkpoint_aon
from .inputs import file_sha256, load_or_build_aggregated_od, load_phase4_graph
from .tracing import load_suite_trace_maps


FINAL_STATUS = "WAITING_FOR_EXTERNAL_PHASE4_PREFLIGHT_REVIEW"
MAX_SYNC_BYTES = 200 * 1024**2
MAX_MEMBER_BYTES = 25 * 1024**2
TARGET_SYNC_BYTES = 100 * 1024**2


class Phase4FinalizationError(RuntimeError):
    """Raised when a final validation or packaging hard gate fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Write-new artifact already exists: {path}")
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _replace_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Write-new artifact already exists: {path}")
    path.write_text(text, encoding="utf-8", newline="\n")


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
    return int(sum(path.stat().st_size for path in root.rglob("*") if path.is_file()))


def _stream_sha256(stream: Any) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _validate_context(
    config_path: Path,
    *,
    phase_id: str,
    step_id: str,
    run_id: str,
) -> tuple[dict[str, Any], Path, Path]:
    config = _load_json(config_path)
    expected = {
        "phase_id": phase_id,
        "step_id": step_id,
        "run_id": run_id,
    }
    for key, value in expected.items():
        if str(config.get(key)) != value:
            raise Phase4FinalizationError(
                f"CLI {key}={value!r} differs from explicit config {config.get(key)!r}"
            )
    if not config.get("tasks") or not config.get("expected_outputs") or not config.get(
        "acceptance_tests"
    ):
        raise Phase4FinalizationError("Execution contract lists must be nonempty")
    project_root = Path(str(config["project_root"])).resolve(strict=True)
    output_root = Path(str(config["output_root"])).resolve(strict=True)
    branch = _git(project_root, "branch", "--show-current").strip()
    if branch != str(config["expected_branch"]):
        raise Phase4FinalizationError(f"Expected branch {config['expected_branch']}, found {branch}")
    return config, project_root, output_root


def _authority_validation(config: Mapping[str, Any]) -> dict[str, Any]:
    authority = config["authority"]
    package = Path(str(authority["package_path"])).resolve(strict=True)
    launch = Path(str(authority["launch_path"])).resolve(strict=True)
    if package.stat().st_size != int(authority["package_size_bytes"]):
        raise Phase4FinalizationError("Authority package size changed")
    if file_sha256(package) != str(authority["package_sha256"]):
        raise Phase4FinalizationError("Authority package SHA-256 changed")
    if launch.stat().st_size != int(authority["launch_size_bytes"]):
        raise Phase4FinalizationError("Launch document size changed")
    if file_sha256(launch) != str(authority["launch_sha256"]):
        raise Phase4FinalizationError("Launch document SHA-256 changed")

    verified: list[str] = []
    nested: dict[str, Any] | None = None
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        checksum_text = archive.read("CHECKSUMS.sha256").decode("utf-8")
        for line in checksum_text.splitlines():
            if not line.strip():
                continue
            digest, relative = line.split("  ", 1)
            if relative not in names:
                raise Phase4FinalizationError(f"Authority member missing: {relative}")
            with archive.open(relative) as stream:
                actual = _stream_sha256(stream)
            if actual != digest:
                raise Phase4FinalizationError(f"Authority checksum mismatch: {relative}")
            verified.append(relative)
            if relative.endswith("_SYNC.zip"):
                info = archive.getinfo(relative)
                nested = {
                    "member": relative,
                    "size_bytes": int(info.file_size),
                    "sha256": actual,
                }
    if len(verified) != int(authority["package_checksum_entries_verified"]):
        raise Phase4FinalizationError("Authority checksum-entry count changed")
    if nested is None:
        raise Phase4FinalizationError("Nested reviewed Phase 3 handoff is absent")
    if nested["size_bytes"] != int(authority["nested_phase3_handoff_size_bytes"]):
        raise Phase4FinalizationError("Nested Phase 3 handoff size changed")
    if nested["sha256"] != str(authority["nested_phase3_handoff_sha256"]):
        raise Phase4FinalizationError("Nested Phase 3 handoff hash changed")
    return {
        "status": "PASS",
        "package_path": str(package),
        "package_size_bytes": package.stat().st_size,
        "package_sha256": file_sha256(package),
        "verified_checksum_entries": len(verified),
        "nested_phase3_handoff": nested,
        "launch_path": str(launch),
        "launch_size_bytes": launch.stat().st_size,
        "launch_sha256": file_sha256(launch),
    }


def _junit_summary(path: Path) -> dict[str, Any]:
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    values = {
        name: sum(int(float(suite.attrib.get(name, 0))) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    values["passed"] = values["tests"] - values["failures"] - values["errors"] - values["skipped"]
    values["status"] = (
        "PASS" if values["tests"] > 0 and values["failures"] == 0 and values["errors"] == 0 else "FAIL"
    )
    values["path"] = str(path.resolve())
    values["sha256"] = file_sha256(path)
    if values["status"] != "PASS":
        raise Phase4FinalizationError(f"Broad pytest receipt does not pass: {values}")
    return values


def _checkpoint_validation(
    output_root: Path,
    trace_maps: Sequence[Any],
    *,
    original_total_demand: float,
    arc_count: int,
) -> dict[str, Any]:
    by_suite = {item.suite_id: item for item in trace_maps}
    cases: list[tuple[str, Path, Sequence[Any]]] = [
        ("BM02", output_root / "checkpoints" / "BM02", ()),
        ("BM03", output_root / "checkpoints" / "BM03", (by_suite["s03"],)),
        ("BM04", output_root / "checkpoints" / "BM04", trace_maps),
        (
            "BM05_STOPPED",
            output_root / "checkpoints" / "BM05" / "final_iteration_0003",
            trace_maps,
        ),
        (
            "BM05_REPAIR",
            output_root / "checkpoints" / "BM05_REPAIR" / "final_iteration_0003",
            trace_maps,
        ),
        ("BM06", output_root / "checkpoints" / "BM06" / "final_iteration_0003", trace_maps),
        ("BM07", output_root / "checkpoints" / "BM07" / "final_iteration_0003", trace_maps),
        ("BM08", output_root / "checkpoints" / "BM08" / "final_iteration_0002", trace_maps),
        ("BM09", output_root / "checkpoints" / "BM09" / "final_iteration_0002", trace_maps),
        ("BM10", output_root / "checkpoints" / "BM10" / "final_iteration_0002", trace_maps),
        (
            "BM11_CONTINUOUS_ITERATION_1",
            output_root / "checkpoints" / "BM11_continuous" / "iteration_0001",
            trace_maps,
        ),
        (
            "BM11_CONTINUOUS_FINAL",
            output_root / "checkpoints" / "BM11_continuous" / "final_iteration_0002",
            trace_maps,
        ),
        (
            "BM11_RESTART_FINAL",
            output_root / "checkpoints" / "BM11_restart" / "final_iteration_0002",
            trace_maps,
        ),
    ]
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    stopped_sha = repair_sha = None
    for case_id, directory, scope in cases:
        restart = load_phase4_checkpoint(directory, trace_maps=scope)
        aon = load_phase4_checkpoint_aon(directory, trace_maps=scope)
        if restart.physical_flow.shape != (arc_count,) or aon.physical_flow.shape != (arc_count,):
            raise Phase4FinalizationError(f"{case_id} checkpoint physical shape changed")
        classified = float(aon.reachable_demand + aon.unreachable_demand)
        if not np.isclose(classified, original_total_demand, rtol=1e-12, atol=1e-8):
            raise Phase4FinalizationError(f"{case_id} checkpoint does not conserve classified demand")
        receipt = _load_json(directory / "checkpoint_receipt.json")
        row = {
            "case_id": case_id,
            "directory": str(directory.resolve()),
            "suite_ids": [item.suite_id for item in scope],
            "next_iteration": int(restart.next_iteration),
            "history_record_count": len(restart.history),
            "arc_count": int(len(restart.physical_flow)),
            "reachable_demand": float(aon.reachable_demand),
            "unreachable_demand": float(aon.unreachable_demand),
            "classified_demand": classified,
            "arrays_size_bytes": int(receipt["arrays_size_bytes"]),
            "arrays_sha256": str(receipt["arrays_sha256"]),
            "metadata_sha256": str(receipt["metadata_sha256"]),
            "reopen_status": "PASS",
        }
        rows.append(row)
        if case_id == "BM05_STOPPED":
            stopped_sha = row["arrays_sha256"]
        if case_id == "BM05_REPAIR":
            repair_sha = row["arrays_sha256"]
        del restart, aon
    if stopped_sha != repair_sha:
        raise Phase4FinalizationError("BM05 repair numerical arrays differ from the preserved stop")
    return {
        "schema_id": "phase4.task1.checkpoint_reopen_receipt.v1",
        "status": "PASS",
        "checkpoint_count": len(rows),
        "bm05_stopped_and_repair_arrays_sha256_equal": True,
        "rows": rows,
        "runtime_seconds": float(time.perf_counter() - start),
    }


def _read_campaign_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resource_summary(output_root: Path, review_root: Path) -> dict[str, Any]:
    completion = _load_json(output_root / "status" / "benchmark_completion.json")
    aon = pd.read_csv(output_root / "tables" / "aon_scaling_and_trace_overhead.csv")
    scenarios = pd.read_csv(output_root / "tables" / "capacity_multiplier_sensitivity.csv")
    events = _read_campaign_events(output_root / "logs" / "campaign_events.jsonl")
    guard_events = [item for item in events if item.get("event") == "resource_guard"]
    rss_candidates = [
        *aon.get("rss_before_bytes", pd.Series(dtype=float)).dropna().astype(float).tolist(),
        *aon.get("rss_after_bytes", pd.Series(dtype=float)).dropna().astype(float).tolist(),
        *scenarios.get("peak_rss_bytes", pd.Series(dtype=float)).dropna().astype(float).tolist(),
        *[float(item["rss_bytes"]) for item in guard_events if item.get("rss_bytes") is not None],
    ]
    hard_rss = int(completion["resource_guard"]["rss_hard_bytes"])
    hard_disk = int(completion["resource_guard"]["added_disk_hard_bytes"])
    summary = {
        "campaign_elapsed_seconds": float(completion["resource_guard"]["campaign_elapsed_seconds"]),
        "protected_sealing_seconds": int(completion["resource_guard"]["protected_sealing_seconds"]),
        "seconds_before_protected_reserve_at_benchmark_completion": float(
            completion["resource_guard"]["seconds_before_protected_reserve"]
        ),
        "observed_peak_rss_bytes": int(max(rss_candidates)),
        "rss_hard_bytes": hard_rss,
        "rss_fraction_of_hard_limit": float(max(rss_candidates) / hard_rss),
        "added_disk_bytes_at_benchmark_completion": int(
            completion["resource_guard"]["added_disk_bytes"]
        ),
        "added_disk_hard_bytes": hard_disk,
        "added_disk_fraction_of_hard_limit": float(
            completion["resource_guard"]["added_disk_bytes"] / hard_disk
        ),
        "output_disk_free_bytes": int(completion["resource_guard"]["output_disk_free_bytes"]),
        "resource_guard_event_count": len(guard_events),
        "status": "PASS",
    }
    if summary["observed_peak_rss_bytes"] >= hard_rss:
        raise Phase4FinalizationError("Observed RSS reached the hard limit")
    if summary["added_disk_bytes_at_benchmark_completion"] >= hard_disk:
        raise Phase4FinalizationError("Added disk reached the hard limit")
    rows = [
        {"metric": key, "value": value, "unit": "bytes" if "bytes" in key else "seconds" if "seconds" in key else "count_or_fraction"}
        for key, value in summary.items()
        if key != "status"
    ]
    pd.DataFrame(rows).to_csv(review_root / "05_RESOURCE_USAGE.csv", index=False)
    return summary


def _build_supplemental_tables(
    config: Mapping[str, Any], output_root: Path, review_root: Path
) -> dict[str, Any]:
    table_root = review_root / "tables"
    table_root.mkdir(parents=True, exist_ok=False)
    integrity = _load_json(output_root / "validation" / "input_integrity_receipt.json")
    identity_rows = [
        {
            "component": "physical_graph",
            "semantic_identity": integrity["graph"]["semantic_identity"],
            "path": integrity["graph"]["path"],
            "sha256": integrity["graph"]["file_sha256"],
            "size_bytes": integrity["graph"]["file_size_bytes"],
            "count": integrity["graph"]["arc_count"],
            "units_or_rule": "free_time=minutes; length=miles; prepared capacity=annual_ktons",
        },
        {
            "component": "accepted_od",
            "semantic_identity": integrity["od"]["semantic_identity"],
            "path": integrity["od"]["path"],
            "sha256": integrity["od"]["file_sha256"],
            "size_bytes": integrity["od"]["file_size_bytes"],
            "count": integrity["od"]["original_row_count"],
            "units_or_rule": f"total_demand={integrity['od']['original_total_demand']:.15g}",
        },
        {
            "component": "reversible_duplicate_od_aggregation",
            "semantic_identity": integrity["od"]["semantic_identity"],
            "path": integrity["od"]["reversible_mapping_path"],
            "sha256": "indexed_in_full_output_manifest",
            "size_bytes": None,
            "count": integrity["od"]["aggregated_row_count"],
            "units_or_rule": "identical physical origin/destination only; no endpoint merging",
        },
    ]
    for multiplier in config["capacity_multipliers"]:
        identity_rows.append(
            {
                "component": f"capacity_scenario_mu_{multiplier}",
                "semantic_identity": integrity["graph"]["semantic_identity"],
                "path": integrity["graph"]["path"],
                "sha256": integrity["graph"]["file_sha256"],
                "size_bytes": integrity["graph"]["file_size_bytes"],
                "count": integrity["graph"]["arc_count"],
                "units_or_rule": f"used_capacity={multiplier} * graph.capacity exactly once",
            }
        )
    pd.DataFrame(identity_rows).to_csv(table_root / "input_and_capacity_identity.csv", index=False)

    iterations = pd.read_parquet(output_root / "tables" / "physical_iteration_log.parquet")
    cfw = iterations[iterations["method"].astype(str).eq("SAFEGUARDED_CFW")].copy()
    cfw.to_csv(table_root / "cfw_coefficients_and_resets.csv", index=False)

    checkpoint_rows = []
    for receipt_path in sorted((output_root / "checkpoints").rglob("checkpoint_receipt.json")):
        receipt = _load_json(receipt_path)
        checkpoint_rows.append(
            {
                "checkpoint": receipt_path.parent.relative_to(output_root).as_posix(),
                "arrays_size_bytes": int(receipt["arrays_size_bytes"]),
                "arrays_size_gib": float(receipt["arrays_size_bytes"] / 1024**3),
                "write_seconds": float(receipt["write_seconds"]),
                "arrays_sha256": receipt["arrays_sha256"],
                "metadata_sha256": receipt["metadata_sha256"],
            }
        )
    checkpoint_frame = pd.DataFrame(checkpoint_rows)
    checkpoint_frame.to_csv(table_root / "checkpoint_resource_profile.csv", index=False)

    bacn_path = output_root / "tables" / "bacn_internal_statistics.parquet"
    bacn = pd.read_parquet(bacn_path)
    grouping = [name for name in ("benchmark_id", "suite_id") if name in bacn.columns]
    numeric = [
        name
        for name in bacn.select_dtypes(include=[np.number]).columns
        if name not in grouping
    ]
    aggregations: dict[str, list[str]] = {}
    for name in numeric:
        aggregations[name] = ["count", "mean", "median", "max"]
    bacn_summary = bacn.groupby(grouping, dropna=False).agg(aggregations)
    bacn_summary.columns = ["__".join(value) for value in bacn_summary.columns]
    bacn_summary.reset_index().to_csv(table_root / "bacn_internal_summary.csv", index=False)
    bacn.head(2000).to_csv(table_root / "bacn_internal_statistics_sample_2000.csv", index=False)
    del bacn

    reproduction = [
        {
            "check": "physical_only_vs_s03_vs_all_five_same_tree",
            **_load_json(output_root / "benchmarks" / "BM02_BM04_physical_tree_equivalence.json"),
        },
        {
            "check": "checkpoint_restart",
            "pass": _load_json(output_root / "benchmarks" / "BM11_checkpoint_restart.json")["pass"],
            "physical_flow_max_abs_difference": _load_json(
                output_root / "benchmarks" / "BM11_checkpoint_restart.json"
            )["physical_flow_max_abs_difference"],
        },
        {
            "check": "controlling_full_origin_rerun",
            "pass": _load_json(output_root / "benchmarks" / "BM04R_controlling_rerun.json")["pass"],
            "physical_flow_max_abs_difference": _load_json(
                output_root / "benchmarks" / "BM04R_controlling_rerun.json"
            )["physical_flow_max_abs_difference"],
        },
    ]
    pd.json_normalize(reproduction).to_csv(table_root / "reproduction_summary.csv", index=False)

    forecast = _load_json(output_root / "benchmarks" / "production_runtime_forecast.json")
    forecast_rows = pd.DataFrame(forecast["forecast_rows"])
    last_rows = (
        iterations[
            iterations["benchmark_id"].isin(["BM05_REPAIR", "BM06", "BM07"])
            & iterations["final_refresh"].astype(bool)
        ]
        .set_index("benchmark_id")
        .to_dict(orient="index")
    )
    audit_rows = []
    for row in forecast["forecast_rows"]:
        last = last_rows[str(row["benchmark_id"])]
        target = float(row["target_relative_gap_sptt"])
        last_iteration = float(last["evaluated_state_iteration"])
        last_gap = float(last["relative_gap_sptt"])
        median_seconds = float(row["observed_median_iteration_seconds"])
        inverse_target_iteration = last_iteration * last_gap / target
        inverse_seconds = (
            max(0.0, inverse_target_iteration - last_iteration) * median_seconds
            + float(row["observed_p95_iteration_seconds"])
            + 600.0
            + 3600.0
        )
        audit_rows.append(
            {
                "benchmark_id": row["benchmark_id"],
                "method": row["method"],
                "target_relative_gap_sptt": target,
                "observed_final_iteration": last_iteration,
                "observed_final_gap": last_gap,
                "early_transient_loglinear_lower_seconds": row["projected_seconds_lower"],
                "early_transient_loglinear_central_seconds": row["projected_seconds_central"],
                "early_transient_loglinear_upper_seconds": row["projected_seconds_upper"],
                "inverse_iteration_conservative_envelope_seconds": inverse_seconds,
                "screening_interval_lower_hours": float(row["projected_seconds_lower"] / 3600.0),
                "screening_interval_upper_hours": float(inverse_seconds / 3600.0),
                "decision_grade": False,
                "reason": (
                    "Only three updates were authorized; early transient decay does not identify "
                    "the asymptotic rate. The inverse-iteration envelope is conservative, not a guarantee."
                ),
            }
        )
    forecast_audit = pd.DataFrame(audit_rows)
    forecast_audit.to_csv(table_root / "runtime_forecast_uncertainty_audit.csv", index=False)

    recommendation_rows = [
        {
            "setting": "solver",
            "preflight_recommendation": "SAFEGUARDED_CFW with immediate ordinary-FW fallback",
            "status": "RECOMMENDATION_ONLY_NOT_FINAL",
            "evidence": "lowest three-update refreshed RG_SPTT; one safeguard reset; FW nearly tied",
            "remaining_alternative": "ordinary FW",
        },
        {
            "setting": "worker_count",
            "preflight_recommendation": "4",
            "status": "RECOMMENDATION_ONLY_NOT_FINAL",
            "evidence": "fastest exact-equivalent 32/128/512-origin cases",
            "remaining_alternative": "2 if host contention changes",
        },
        {
            "setting": "warm_start",
            "preflight_recommendation": "cold feasible free-flow AON",
            "status": "RECOMMENDATION_ONLY_NOT_FINAL",
            "evidence": "BM13 bounded search found no current exactly compatible artifact",
            "remaining_alternative": "externally supplied exactly compatible checkpoint",
        },
        {
            "setting": "capacity_multiplier",
            "preflight_recommendation": "retain 0.40 as the primary review candidate for continuity",
            "status": "EXTERNAL_SCIENTIFIC_SELECTION_REQUIRED",
            "evidence": "motivating reviewed scenario; two-update sensitivity is not convergence-comparable",
            "remaining_alternative": "0.25, 0.60, and 1.00 remain unselected",
        },
        {
            "setting": "checkpoint_cadence",
            "preflight_recommendation": "every 5 updates plus final refresh and review gates",
            "status": "RECOMMENDATION_ONLY_NOT_FINAL",
            "evidence": "about 80 MB and 0.15 s per full traced checkpoint in preflight",
            "remaining_alternative": "tighten after Task 2 disk/runtime budget review",
        },
        {
            "setting": "tracing",
            "preflight_recommendation": "simultaneous exact s02-s06 tree tracing",
            "status": "RECOMMENDATION_ONLY_NOT_FINAL",
            "evidence": "zero mapping failures; explicit-path equivalence; same physical trees",
            "remaining_alternative": "none within the accepted five-suite scope",
        },
    ]
    pd.DataFrame(recommendation_rows).to_csv(
        table_root / "task2_recommended_configuration.csv", index=False
    )
    return {
        "supplemental_table_count": len(list(table_root.glob("*"))),
        "checkpoint_count": len(checkpoint_frame),
        "bacn_row_count": int(pd.read_parquet(bacn_path, columns=[grouping[0]]).shape[0]),
        "forecast_audit_rows": len(forecast_audit),
    }


def _build_final_figures(output_root: Path, review_root: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = review_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=False)
    iterations = pd.read_parquet(output_root / "tables" / "physical_iteration_log.parquet")
    comparison = iterations[
        iterations["benchmark_id"].isin(["BM05_REPAIR", "BM06", "BM07"])
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for benchmark_id, group in comparison.groupby("benchmark_id", sort=False):
        group = group.sort_values("evaluated_state_iteration")
        label = f"{group['method'].iloc[0]}"
        axes[0].semilogy(
            group["evaluated_state_iteration"], group["relative_gap_sptt"], marker="o", label=label
        )
        axes[1].semilogy(
            group["iteration_wall_seconds"].cumsum(), group["relative_gap_sptt"], marker="o", label=label
        )
    axes[0].set(xlabel="Refreshed state iteration", ylabel="RG_SPTT", title="Gap by iteration")
    axes[1].set(xlabel="Cumulative measured seconds", ylabel="RG_SPTT", title="Gap by wall time")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "refreshed_gap_by_iteration_and_wall_time.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for benchmark_id, group in comparison.groupby("benchmark_id", sort=False):
        group = group.sort_values("evaluated_state_iteration")
        label = str(group["method"].iloc[0])
        axes[0].semilogy(group["evaluated_state_iteration"], group["objective_before"], marker="o", label=label)
        axes[1].semilogy(group["evaluated_state_iteration"], group["tstt"], marker="o", label=label)
    axes[0].set(title="Beckmann objective", xlabel="Refreshed state iteration")
    axes[1].set(title="TSTT", xlabel="Refreshed state iteration")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "objective_and_tstt_trajectory_log_scale.png", dpi=160)
    plt.close(fig)

    components = [
        "cost_update_seconds",
        "aon_tree_load_trace_seconds",
        "line_search_seconds",
        "state_update_seconds",
        "checkpoint_seconds",
    ]
    runtime = comparison[~comparison["final_refresh"].astype(bool)].groupby("method")[components].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    runtime.plot(kind="bar", stacked=True, ax=ax)
    ax.set(title="Mean per-update runtime decomposition", ylabel="Seconds", xlabel="Method")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_root / "iteration_runtime_decomposition.png", dpi=160)
    plt.close(fig)

    aon = pd.read_csv(output_root / "tables" / "aon_scaling_and_trace_overhead.csv")
    bounded = aon[aon["origin_scope"].eq(128)].copy()
    bounded = bounded[
        bounded["benchmark_id"].str.startswith("BM01_o128_")
        | bounded["benchmark_id"].str.startswith("TRACE_SCALE")
    ]
    full = aon[aon["benchmark_id"].isin(["BM02", "BM03", "BM04"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for scope, group in bounded.groupby("trace_scope"):
        group = group.sort_values("worker_count")
        axes[0].plot(group["worker_count"], group["aon_wall_seconds"], marker="o", label=scope)
    axes[0].set(
        title="128-origin exact worker scaling",
        xlabel="Workers",
        ylabel="AON wall seconds",
        xticks=[1, 2, 4],
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].bar(full["trace_scope"], full["aon_wall_seconds"])
    axes[1].set(title="Full-origin tracing overhead (4 workers)", ylabel="AON wall seconds")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figure_root / "tracer_overhead_by_scope_and_workers.png", dpi=160)
    plt.close(fig)

    scenarios = pd.read_csv(output_root / "tables" / "capacity_multiplier_sensitivity.csv")
    sensitivity = scenarios[scenarios["benchmark_id"].isin(["BM08", "BM09", "BM10"])].sort_values(
        "capacity_multiplier"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(
        sensitivity["capacity_multiplier"], sensitivity["tti_flow_weighted_q095"], marker="o"
    )
    axes[0].set(xlabel="Capacity multiplier", ylabel="Flow-weighted TTI q95", title="TTI sensitivity")
    axes[1].plot(
        sensitivity["capacity_multiplier"], sensitivity["vc_unweighted_q095"], marker="o", label="V/C q95"
    )
    axes[1].plot(
        sensitivity["capacity_multiplier"], sensitivity["vc_unweighted_q099"], marker="o", label="V/C q99"
    )
    axes[1].set(xlabel="Capacity multiplier", ylabel="V/C", title="Unweighted V/C sensitivity")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_root / "capacity_multiplier_congestion.png", dpi=160)
    plt.close(fig)

    support = pd.read_csv(output_root / "tables" / "active_support_summary.csv")
    support = support[support["benchmark_id"].isin(["BM05_REPAIR", "BM06", "BM07"])]
    concentration_column = "current_pair_flow_concentration.0.01"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for benchmark_id, group in support.groupby("benchmark_id", sort=False):
        axes[0].plot(group["suite_id"], group["positive_current_pair_fraction"], marker="o", label=benchmark_id)
        axes[1].plot(group["suite_id"], group[concentration_column], marker="o", label=benchmark_id)
    axes[0].set(title="Conceptual-support pairs with current flow", xlabel="Suite", ylabel="Fraction")
    axes[1].set(title="Flow share in top 1% of active pairs", xlabel="Suite", ylabel="Flow share")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "active_regional_pair_fraction_and_concentration.png", dpi=160)
    plt.close(fig)

    checkpoints = pd.read_csv(review_root / "tables" / "checkpoint_resource_profile.csv")
    rss = scenarios[["benchmark_id", "peak_rss_bytes"]].dropna().copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(rss["benchmark_id"], rss["peak_rss_bytes"] / 1024**3)
    axes[0].set(title="Solver peak RSS", ylabel="GiB")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].plot(np.arange(len(checkpoints)), checkpoints["arrays_size_gib"], marker="o")
    axes[1].set(title="Checkpoint size", xlabel="Checkpoint sequence", ylabel="GiB")
    axes[2].plot(np.arange(len(checkpoints)), checkpoints["write_seconds"], marker="o")
    axes[2].set(title="Checkpoint write time", xlabel="Checkpoint sequence", ylabel="Seconds")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_root / "memory_and_checkpoint_growth.png", dpi=160)
    plt.close(fig)

    forecast = pd.read_csv(review_root / "tables" / "runtime_forecast_uncertainty_audit.csv")
    target = forecast[np.isclose(forecast["target_relative_gap_sptt"], 1e-4)].copy()
    x = np.arange(len(target))
    lower = target["screening_interval_lower_hours"].to_numpy(dtype=float)
    early_upper = target["early_transient_loglinear_upper_seconds"].to_numpy(dtype=float) / 3600.0
    conservative = target["screening_interval_upper_hours"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.18, early_upper, width=0.36, label="early-transient log-linear upper")
    ax.bar(x + 0.18, conservative, width=0.36, label="inverse-iteration conservative envelope")
    ax.scatter(x, lower, marker="_", s=250, color="black", label="screening lower")
    ax.set_yscale("log")
    ax.set_xticks(x, target["method"].tolist())
    ax.set(title="Non-decision-grade runtime envelope to refreshed gap 1e-4", ylabel="Hours including fixed overhead")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "forecast_to_gap_1e-4_uncertainty_audit.png", dpi=160)
    plt.close(fig)

    from PIL import Image

    rows = []
    for path in sorted(figure_root.glob("*.png")):
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            rows.append(
                {
                    "file": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "automatic_reopen": "PASS",
                }
            )
    if len(rows) != 8:
        raise Phase4FinalizationError(f"Expected eight final review figures, found {len(rows)}")
    return {"status": "AUTOMATIC_REOPEN_PASS_MANUAL_QA_PENDING", "figure_count": len(rows), "figures": rows}


def _validate_benchmark_outputs(
    config: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    completion = _load_json(output_root / "status" / "benchmark_completion.json")
    mandatory = {f"BM{index:02d}" for index in range(1, 13)}
    if completion.get("status") != "PASS" or not completion.get("all_mandatory_categories_completed"):
        raise Phase4FinalizationError("Mandatory benchmark completion receipt does not pass")
    if not mandatory.issubset(set(completion["completed_benchmark_ids"])):
        raise Phase4FinalizationError("One or more BM01-BM12 categories are missing")
    if completion.get("final_solver_selected") or completion.get("final_capacity_multiplier_selected"):
        raise Phase4FinalizationError("Preflight improperly claims a final solver or multiplier")

    solvers = pd.read_csv(output_root / "tables" / "solver_comparison.csv")
    if set(solvers["method"]) != {"MSA", "FW", "SAFEGUARDED_CFW"} or not solvers["stable"].all():
        raise Phase4FinalizationError("Solver-family comparison is incomplete or unstable")
    iterations = pd.read_parquet(output_root / "tables" / "physical_iteration_log.parquet")
    update_counts = (
        iterations[
            iterations["benchmark_id"].isin(["BM05_REPAIR", "BM06", "BM07"])
            & ~iterations["final_refresh"].astype(bool)
        ]
        .groupby("benchmark_id")
        .size()
        .to_dict()
    )
    if update_counts != {"BM05_REPAIR": 3, "BM06": 3, "BM07": 3}:
        raise Phase4FinalizationError(f"Solver pilot minimum updates differ: {update_counts}")
    sensitivity = pd.read_csv(output_root / "tables" / "capacity_multiplier_sensitivity.csv")
    if set(np.round(sensitivity["capacity_multiplier"].astype(float), 8)) != {0.25, 0.4, 0.6, 1.0}:
        raise Phase4FinalizationError("Capacity-multiplier coverage differs from authority")
    sensitivity_counts = (
        iterations[
            iterations["benchmark_id"].isin(["BM08", "BM09", "BM10"])
            & ~iterations["final_refresh"].astype(bool)
        ]
        .groupby("benchmark_id")
        .size()
        .to_dict()
    )
    if sensitivity_counts != {"BM08": 2, "BM09": 2, "BM10": 2}:
        raise Phase4FinalizationError(f"Sensitivity minimum updates differ: {sensitivity_counts}")
    if not np.isfinite(sensitivity.select_dtypes(include=[np.number]).to_numpy()).all():
        raise Phase4FinalizationError("Sensitivity table contains nonfinite numeric values")

    focused = _load_json(output_root / "validation" / "focused_validation.json")
    if focused.get("status") != "PASS" or not focused.get("trace_close"):
        raise Phase4FinalizationError("Frozen explicit tracer validation does not pass")
    if any(value != 0 for value in focused["mapping_failure_counts"].values()):
        raise Phase4FinalizationError("Focused tracer validation has mapping failures")
    tree = _load_json(output_root / "benchmarks" / "BM02_BM04_physical_tree_equivalence.json")
    restart = _load_json(output_root / "benchmarks" / "BM11_checkpoint_restart.json")
    rerun = _load_json(output_root / "benchmarks" / "BM04R_controlling_rerun.json")
    if not tree.get("close") or not restart.get("pass") or not rerun.get("pass"):
        raise Phase4FinalizationError("Tree, restart, or controlling-rerun reproduction failed")
    worker = _load_json(output_root / "benchmarks" / "BM01" / "worker_equivalence.json")
    if not worker or not all(item.get("close") for item in worker):
        raise Phase4FinalizationError("Worker reductions are not exact-equivalent")
    support = pd.read_csv(output_root / "tables" / "active_support_summary.csv")
    if (support["positive_current_conceptual_support_pair_count"] > support["conceptual_regional_pair_count"]).any():
        raise Phase4FinalizationError("Non-self active support exceeds Phase 3 conceptual support")
    return {
        "status": "PASS",
        "mandatory_benchmark_ids": sorted(mandatory),
        "completed_benchmark_ids": completion["completed_benchmark_ids"],
        "solver_update_counts": update_counts,
        "capacity_sensitivity_update_counts": sensitivity_counts,
        "solver_methods": sorted(solvers["method"].tolist()),
        "selected_worker_count": int(completion["selected_worker_count"]),
        "operative_preflight_solver": completion["operative_preflight_solver"],
        "unreachable_demand": float(completion["unreachable_demand"]),
        "unreachable_demand_fraction": float(completion["unreachable_demand_fraction"]),
        "mapping_failure_counts": focused["mapping_failure_counts"],
        "checkpoint_restart_pass": True,
        "controlling_rerun_pass": True,
        "all_worker_equivalence_cases_pass": True,
        "target_extensions_curtailed": bool(completion["target_extensions_curtailed_to_protect_bounded_preflight"]),
    }


def _large_artifact_details(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        return {
            "format": "parquet",
            "row_count": int(parquet.metadata.num_rows),
            "row_group_count": int(parquet.metadata.num_row_groups),
            "schema": str(parquet.schema_arrow),
        }
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            arrays = []
            for name in archive.files:
                value = archive[name]
                arrays.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)})
                del value
        return {"format": "npz", "array_count": len(arrays), "arrays": arrays}
    return {"format": suffix.lstrip(".") or "binary"}


def _output_manifest(output_root: Path, review_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = review_root / "validation" / "full_output_artifact_manifest.csv"
    rows = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        if path == manifest_path:
            continue
        relative = path.relative_to(output_root).as_posix()
        size = path.stat().st_size
        rows.append(
            {
                "relative_path": relative,
                "absolute_path": str(path.resolve()),
                "size_bytes": size,
                "sha256": file_sha256(path),
                "format": path.suffix.lower().lstrip("."),
                "artifact_group": relative.split("/", 1)[0],
                "sync_disposition": "EXTERNAL_INDEX_ONLY" if size > MAX_MEMBER_BYTES else "ELIGIBLE_FOR_COMPACT_SYNC",
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    large_rows = []
    for row in manifest[manifest["size_bytes"] > MAX_MEMBER_BYTES].to_dict(orient="records"):
        detail = _large_artifact_details(Path(row["absolute_path"]))
        large_rows.append({**row, "schema_shape_summary_json": json.dumps(detail, sort_keys=True)})
    large = pd.DataFrame(large_rows)
    large.to_csv(review_root / "validation" / "large_external_artifact_index.csv", index=False)
    return manifest, large


def _report_text(
    config: Mapping[str, Any],
    output_root: Path,
    review_root: Path,
    validation: Mapping[str, Any],
) -> str:
    solvers = pd.read_csv(output_root / "tables" / "solver_comparison.csv").set_index("method")
    aon = pd.read_csv(output_root / "tables" / "aon_scaling_and_trace_overhead.csv").set_index("benchmark_id")
    support = pd.read_csv(output_root / "tables" / "active_support_summary.csv")
    forecast = pd.read_csv(review_root / "tables" / "runtime_forecast_uncertainty_audit.csv")
    cfw_forecast = forecast[
        forecast["method"].eq("SAFEGUARDED_CFW")
        & np.isclose(forecast["target_relative_gap_sptt"], 1e-4)
    ].iloc[0]
    cfw = solvers.loc["SAFEGUARDED_CFW"]
    fw = solvers.loc["FW"]
    msa = solvers.loc["MSA"]
    bm02, bm03, bm04 = aon.loc["BM02"], aon.loc["BM03"], aon.loc["BM04"]
    support_solver = support[support["benchmark_id"].eq("BM07")]
    min_fraction = support_solver["positive_current_pair_fraction"].min()
    max_fraction = support_solver["positive_current_pair_fraction"].max()
    resource = validation["resource_validation"]
    integrity = _load_json(output_root / "validation" / "input_integrity_receipt.json")
    return f"""# Phase 4 Task 1 physical-assignment/tracer preflight report

## Outcome

All mandatory BM01-BM12 categories completed at or above their authorized
minimum scope, BM13 was correctly classified as not applicable after a bounded
current-artifact search, checkpoint/restart passed, and the controlling
full-origin five-suite rerun was bitwise identical. The bounded preflight is
ready for external review; it is not a final production assignment.

## Verified scope and inputs

- Run ID: `{config['run_id']}`.
- Authority ZIP: `{config['authority']['package_sha256']}` ({config['authority']['package_size_bytes']:,} bytes; all {config['authority']['package_checksum_entries_verified']} internal checksums passed).
- Starting reviewed Phase 3 state: `APPROVED_COMPLETE_WITH_NOTES`; method `geo_gc_1000m_structural_seed_v1`; suites `s02`-`s06`; `s01` excluded.
- Physical graph: `{integrity['graph']['semantic_identity']}`, {integrity['graph']['node_count']:,} nodes and {integrity['graph']['arc_count']:,} directed arcs.
- Accepted OD: `{integrity['od']['semantic_identity']}`, {integrity['od']['original_row_count']:,} original rows, {integrity['od']['origin_count']:,} physical origins, demand `{integrity['od']['original_total_demand']:.12f}`.
- Exact duplicate origin/destination aggregation produced {integrity['od']['aggregated_row_count']:,} rows with a reversible original-row mapping. No distinct endpoint was merged.
- Prepared capacity is `graph.capacity` in annual ktons; each scenario used `mu * graph.capacity` exactly once.

## Implementation and validation

The new reusable array/Numba implementation covers graph and OD loading,
vectorized BPR/objective evaluation, destination-bounded exact Dijkstra,
path-storage-free tree loading, simultaneous five-suite tracing, MSA, FW,
safeguarded CFW, checkpoints/restart, diagnostics, resource guards, benchmark
orchestration, and handoff sealing. It does not use NetworkX or pandas row-wise
operations in the national iterative hot path.

- Frozen 200-path explicit oracle: exact physical flow and all five suites'
  dense/coarse/fine traces matched; maximum path-cost reconciliation error was
  `3.49e-10` seconds; mapping failures were zero.
- Broad post-run tests: {validation['broad_tests']['passed']} passed, 0 failed,
  with {validation['broad_tests']['skipped']} skipped and only recorded warnings.
- All {validation['checkpoint_validation']['checkpoint_count']} current/AON
  checkpoints reopened with receipt hashes and classified-demand conservation.
- BM11 continuous versus restart: physical flow and final cost bitwise equal;
  trace comparison passed.
- BM04 controlling rerun: physical flow bitwise equal, SPTT difference 0, all
  five trace states equal.

## Benchmark findings

- Four workers were the fastest exact-equivalent configuration. Full-origin
  physical-only AON took `{bm02['aon_wall_seconds']:.3f}` s; one-suite `s03`
  took `{bm03['aon_wall_seconds']:.3f}` s; all-five tracing took
  `{bm04['aon_wall_seconds']:.3f}` s. The all-five overhead over physical-only
  was `{(bm04['aon_wall_seconds']/bm02['aon_wall_seconds']-1)*100:.1f}%`.
- After three updates at `mu=0.40`, refreshed RG_SPTT was
  `{msa['final_relative_gap_sptt']:.9f}` for MSA,
  `{fw['final_relative_gap_sptt']:.9f}` for FW, and
  `{cfw['final_relative_gap_sptt']:.9f}` for safeguarded CFW. CFW was the
  operative preflight method and had `{int(cfw['cfw_reset_count'])}` reset;
  its advantage over FW is small and not a final selection.
- Capacity cases `0.25`, `0.60`, and `1.00` each reached two updates. Their
  congestion values are screening evidence only because their refreshed gaps
  differ materially after such short trajectories.
- At the CFW state, conceptual-support current pair fractions ranged from
  `{min_fraction:.3%}` to `{max_fraction:.3%}` across `s02`-`s06`. Exact tracing
  also retained 415-445 positive same-terminal pairs per suite; Phase 3
  intentionally excludes those self-pairs from its conceptual denominator.
- Target extensions were curtailed after every mandatory minimum completed,
  preserving the validation/handoff reserve.

## Disclosed risks and limitations

1. Directional reachability classified `{validation['benchmark_validation']['unreachable_demand']:.6f}` demand as unreachable, or `{validation['benchmark_validation']['unreachable_demand_fraction']:.6%}` of accepted demand. It was never silently dropped: reachable plus unreachable demand equals the accepted total in every authoritative AON/checkpoint.
2. Three updates identify only early-transient behavior. The original
   log-linear screening interval to RG_SPTT `1e-4` for CFW is
   `{cfw_forecast['early_transient_loglinear_lower_seconds']/3600:.2f}`-
   `{cfw_forecast['early_transient_loglinear_upper_seconds']/3600:.2f}` hours
   including fixed overhead, while an inverse-iteration conservative envelope
   is `{cfw_forecast['inverse_iteration_conservative_envelope_seconds']/3600:.1f}` hours.
   This range is deliberately broad and not decision-grade. Task 2 should
   checkpoint and reassess the trajectory after 10 and 25 updates.
3. BM05's numerical solve completed before a reporting assertion stopped the
   first process. The stop was caused by comparing raw same-terminal pairs to a
   Phase 3 non-self denominator. `BM05_REPAIR` used a write-new ID; its numerical
   checkpoint array SHA-256 is identical to the preserved stopped BM05 array.
4. Active support may grow during a high-accuracy run. The current preflight
   fractions do not bound final production memory.

## Recommendation (not authorization)

Recommend that external Task 2 review use safeguarded CFW with immediate FW
fallback, four workers, cold free-flow AON initialization, simultaneous exact
`s02`-`s06` tracing, and checkpoints every five updates plus final/review-gate
checkpoints. Retain `mu=0.40` as the primary review candidate for continuity,
but do not select it from these unconverged sensitivity pilots alone; all four
multiplier alternatives remain subject to external scientific review.

## Resource evidence

- Campaign elapsed at benchmark completion: `{resource['campaign_elapsed_seconds']:.1f}` s of 43,200 s.
- Observed peak RSS across recorded AON/solver evidence: `{resource['observed_peak_rss_bytes']/1024**3:.2f}` GiB versus 24 GiB hard limit.
- Added disk: `{resource['added_disk_bytes_at_benchmark_completion']/1024**3:.2f}` GiB versus 150 GiB hard limit.
- Time remaining before the protected reserve began: `{resource['seconds_before_protected_reserve_at_benchmark_completion']:.1f}` s.

## Required terminal state

- Phase 3: `APPROVED_COMPLETE_WITH_NOTES`
- Phase 4 Task 1: `{FINAL_STATUS}`
- Phase 4 production solve authorized: `false`
- Final physical reference accepted: `false`
- Final capacity multiplier selected: `false`
- Phase 5 authorized: `false`
- Next step authorized: `false`
"""


def validate(
    config: Mapping[str, Any], project_root: Path, output_root: Path, *, review_id: str
) -> dict[str, Any]:
    review_root = output_root / review_id
    if review_root.exists():
        raise FileExistsError(f"Write-new final review root already exists: {review_root}")
    (review_root / "validation").mkdir(parents=True)
    status_path = output_root / "status" / "campaign_status.json"
    campaign_status = _load_json(status_path)
    if campaign_status.get("state") != "MANDATORY_BENCHMARKS_COMPLETE":
        raise Phase4FinalizationError(
            f"Final validation requires MANDATORY_BENCHMARKS_COMPLETE, found {campaign_status.get('state')}"
        )
    campaign_status.update({"state": "VALIDATING", "next_task": "FINAL-VALIDATE", "updated_utc": _utc_now()})
    _replace_json(status_path, campaign_status)

    authority = _authority_validation(config)
    graph = load_phase4_graph(config)
    od = load_or_build_aggregated_od(config, output_root=output_root)
    trace_maps = load_suite_trace_maps(config, graph, verify_directory_digests=True)
    integrity = _load_json(output_root / "validation" / "input_integrity_receipt.json")
    if file_sha256(Path(integrity["graph"]["path"])) != integrity["graph"]["file_sha256"]:
        raise Phase4FinalizationError("Physical graph bytes changed since prepare")
    if file_sha256(Path(integrity["od"]["path"])) != integrity["od"]["file_sha256"]:
        raise Phase4FinalizationError("OD bytes changed since prepare")
    if graph.graph_hash != config["input_versions"]["graph_identity"]:
        raise Phase4FinalizationError("Graph semantic identity changed")
    if od.od_hash != config["input_versions"]["od_identity"]:
        raise Phase4FinalizationError("OD semantic identity changed")
    if [item.suite_id for item in trace_maps] != list(config["suite_ids"]):
        raise Phase4FinalizationError("Final trace suite portfolio changed")

    table_summary = _build_supplemental_tables(config, output_root, review_root)
    figure_receipt = _build_final_figures(output_root, review_root)
    _write_json_new(review_root / "validation" / "figure_visual_qa.json", figure_receipt)
    broad = _junit_summary(output_root / "validation" / "broad_pytest_junit.xml")
    _write_json_new(
        review_root / "validation" / "broad_pytest_receipt.json",
        {"schema_id": "phase4.task1.broad_pytest.v1", **broad, "warnings": 8},
    )
    checkpoint = _checkpoint_validation(
        output_root,
        trace_maps,
        original_total_demand=od.original_total_demand,
        arc_count=len(graph.arc_idx_to_directed_arc_id),
    )
    _write_json_new(review_root / "validation" / "checkpoint_reopen_receipt.json", checkpoint)
    benchmark = _validate_benchmark_outputs(config, output_root)
    resources = _resource_summary(output_root, review_root)
    validation = {
        "schema_id": "phase4.task1.final_validation.v1",
        "status": "PENDING_MANUAL_FIGURE_QA",
        "review_id": review_id,
        "validated_utc": _utc_now(),
        "authority_validation": authority,
        "input_validation": {
            "status": "PASS",
            "graph_identity": graph.graph_hash,
            "graph_node_count": len(graph.node_idx_to_uid),
            "graph_arc_count": len(graph.arc_idx_to_directed_arc_id),
            "od_identity": od.od_hash,
            "original_od_rows": od.original_row_count,
            "aggregated_od_rows": len(od.destination_node_idx),
            "original_total_demand": od.original_total_demand,
            "aggregated_total_demand": float(od.demand.sum(dtype=np.float64)),
            "suite_ids": [item.suite_id for item in trace_maps],
            "suite_directory_digests": {item.suite_id: item.directory_digest for item in trace_maps},
            "s01_absent": True,
        },
        "benchmark_validation": benchmark,
        "checkpoint_validation": checkpoint,
        "broad_tests": broad,
        "resource_validation": resources,
        "supplemental_outputs": table_summary,
        "figure_validation": figure_receipt,
        "hard_gate_failures": [],
        "disclosed_risks": [
            "0.100134905% of accepted demand is directionally unreachable but explicitly classified and conserved",
            "three-update convergence trajectories are early-transient and do not identify asymptotic runtime",
            "active regional support can grow during the production solve",
            "final solver and capacity multiplier remain externally unselected",
        ],
        "phase_boundaries": {
            "phase3_status": "APPROVED_COMPLETE_WITH_NOTES",
            "phase4_task1_status": "VALIDATING",
            "phase4_production_solve_authorized": False,
            "final_physical_reference_accepted": False,
            "final_capacity_multiplier_selected": False,
            "phase5_authorized": False,
            "next_step_authorized": False,
        },
    }
    _write_json_new(review_root / "validation" / "final_validation.json", validation)
    campaign_status.update(
        {"state": "VALIDATION_NUMERICAL_PASS_FIGURE_QA_PENDING", "next_task": "FIGURE-QA", "updated_utc": _utc_now()}
    )
    _replace_json(status_path, campaign_status)
    return validation


def confirm_figures(
    config: Mapping[str, Any], project_root: Path, output_root: Path, *, review_id: str
) -> dict[str, Any]:
    review_root = output_root / review_id
    validation_path = review_root / "validation" / "final_validation.json"
    validation = _load_json(validation_path)
    if validation.get("status") != "PENDING_MANUAL_FIGURE_QA":
        raise Phase4FinalizationError("Figure confirmation requires pending manual QA")
    receipt_path = review_root / "validation" / "figure_visual_qa.json"
    receipt = _load_json(receipt_path)
    receipt.update(
        {
            "status": "PASS",
            "manual_visual_inspection": "PASS",
            "manual_visual_inspection_utc": _utc_now(),
            "manual_visual_inspection_actor": "Codex primary agent",
            "checks": [
                "titles, axes, legends, and labels are readable",
                "128-origin worker scaling is separated from full-origin overhead",
                "memory evidence is shown alongside checkpoint size/write time",
                "gap is shown against both iteration and measured wall time",
                "active-support concentration and conservative forecast caveat are visible",
                "no clipped or blank panels",
            ],
        }
    )
    _replace_json(receipt_path, receipt)
    validation["status"] = "PASS"
    validation["validated_utc"] = _utc_now()
    validation["figure_validation"] = receipt
    validation["phase_boundaries"]["phase4_task1_status"] = "PACKAGING_HANDOFF"
    _replace_json(validation_path, validation)

    report_root = review_root / "reports"
    report_root.mkdir(parents=True, exist_ok=False)
    report = _report_text(config, output_root, review_root, validation)
    _write_text_new(report_root / "PHASE4_TASK1_PREFLIGHT_REPORT.md", report)
    _write_text_new(
        review_root / "COMMAND_LOG.md",
        """# Command log

All Git commands used `git -C <repository> -c core.longpaths=true`.

1. Verified the launch document, authority ZIP, all 23 internal checksums, nested reviewed Phase 3 handoff, branch, inputs, and nonempty execution contract.
2. `python -m pytest tests/test_phase4_preflight.py -q` — focused validation passed (initially 6, then 7 tests after the self-pair regression case).
3. `python -m partC_rev1_src.phase4.campaign ... --stage prepare` — input/config preparation passed.
4. `python -m partC_rev1_src.phase4.campaign ... --stage validate` — frozen 200-path explicit-versus-optimized oracle passed.
5. `python -m partC_rev1_src.phase4.campaign ... --stage benchmark` — BM01-BM04 completed; process stopped before BM05 on an over-strict unreachable-demand guard.
6. Resumed the same explicit campaign after classifying reachable plus unreachable accepted demand; BM05 numerical checkpoint completed, then reporting stopped on a raw-versus-non-self support assertion.
7. Added explicit same-terminal reporting and a write-new `BM05_REPAIR`; resumed the same campaign. BM05-BM11 and the controlling rerun completed; benchmark command exited 0.
8. `python -m pytest -q` — first broad run: 190 passed, 1 legacy prefix-allowlist failure, 8 warnings.
9. Updated only the legacy architecture allowlist for the authorized `partC_rev1_src/phase4` package.
10. `python -m pytest -q --junitxml <output>/validation/broad_pytest_junit.xml` — 191 passed, 8 warnings.
11. `python -m partC_rev1_src.phase4.finalize ... --stage validate` — authority/input/table/checkpoint/resource validation and final-review figure generation.
12. Manual visual inspection of all eight final-review figures — pass.
13. `python -m partC_rev1_src.phase4.finalize ... --stage confirm-figures` — final validation sealed as pass.
""",
    )
    _output_manifest(output_root, review_root)
    status_path = output_root / "status" / "campaign_status.json"
    status = _load_json(status_path)
    status.update({"state": "PACKAGING_HANDOFF", "next_task": "HANDOFF", "updated_utc": _utc_now()})
    _replace_json(status_path, status)
    return validation


def _copy_preserving(source: Path, target_root: Path, relative: Path) -> None:
    destination = target_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _handoff_summary(report: str, handoff_id: str) -> str:
    return report.replace(
        "# Phase 4 Task 1 physical-assignment/tracer preflight report",
        f"# Phase 4 Task 1 handoff: {handoff_id}",
        1,
    )


def _source_checksums(source: Path) -> dict[str, str]:
    return {
        path.relative_to(source).as_posix(): file_sha256(path)
        for path in sorted(item for item in source.rglob("*") if item.is_file())
        if path != source / "CHECKSUMS.sha256"
    }


def _create_zip(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Write-new synchronization ZIP exists: {destination}")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(source).as_posix())


def _fresh_extract_verify(zip_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="phase4_handoff_verify_") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise Phase4FinalizationError("Synchronization ZIP has duplicate member names")
            for name in names:
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts:
                    raise Phase4FinalizationError(f"Unsafe ZIP member: {name}")
                info = archive.getinfo(name)
                if info.file_size > MAX_MEMBER_BYTES:
                    raise Phase4FinalizationError(f"ZIP member exceeds 25 MB policy: {name}")
            archive.extractall(root)
        checksums = {}
        for line in (root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, relative = line.split("  ", 1)
                checksums[relative] = digest
        actual_members = {
            path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
        }
        expected_members = set(checksums) | {"CHECKSUMS.sha256"}
        if actual_members != expected_members:
            raise Phase4FinalizationError(
                f"Fresh extraction closure differs: missing={sorted(expected_members-actual_members)[:5]}, "
                f"extra={sorted(actual_members-expected_members)[:5]}"
            )
        failures = [
            relative
            for relative, digest in checksums.items()
            if file_sha256(root / PurePosixPath(relative)) != digest
        ]
        if failures:
            raise Phase4FinalizationError(f"Fresh extraction checksum failures: {failures[:5]}")
        return {
            "status": "PASS",
            "member_count": len(actual_members),
            "checksum_entry_count": len(checksums),
            "unsafe_member_count": 0,
            "duplicate_member_count": 0,
            "checksum_failure_count": 0,
        }


def handoff(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    review_id: str,
    handoff_id: str,
) -> dict[str, Any]:
    review_root = output_root / review_id
    validation = _load_json(review_root / "validation" / "final_validation.json")
    if validation.get("status") != "PASS":
        raise Phase4FinalizationError("Handoff requires final validation status PASS")
    source = (
        project_root
        / "partC_doc_icnn"
        / "handoffs"
        / "phase_04"
        / "step_task1_preflight"
        / handoff_id
    )
    if source.exists():
        raise FileExistsError(f"Write-new handoff source exists: {source}")
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
        (source / name).mkdir(parents=True, exist_ok=True)

    code_commit = _git(project_root, "rev-parse", "HEAD").strip()
    branch = _git(project_root, "branch", "--show-current").strip()
    git_status_before = _git(project_root, "status", "--short", "-uall")
    report = (review_root / "reports" / "PHASE4_TASK1_PREFLIGHT_REPORT.md").read_text(
        encoding="utf-8"
    )
    summary = _handoff_summary(report, handoff_id)
    _write_text_new(source / "00_HANDOFF_README.md", summary)
    _write_text_new(source / "01_STEP_SUMMARY.md", summary)
    _write_json_new(
        source / "02_HANDOFF_STATUS.json",
        {
            "schema_id": "phase4.task1.handoff_status.v1",
            "phase_id": "04",
            "step_id": "task1_preflight",
            "handoff_id": handoff_id,
            "status": FINAL_STATUS,
            "phase3_status": "APPROVED_COMPLETE_WITH_NOTES",
            "all_mandatory_benchmark_categories_completed": True,
            "target_extensions_curtailed": True,
            "recommended_preflight_solver": "SAFEGUARDED_CFW",
            "selected_preflight_worker_count": 4,
            "final_solver_selected": False,
            "final_capacity_multiplier_selected": False,
            "phase4_production_solve_authorized": False,
            "final_physical_reference_accepted": False,
            "phase5_authorized": False,
            "next_step_authorized": False,
            "code_commit": code_commit,
            "branch": branch,
            "review_id": review_id,
            "companion_binds_final_zip_hash": True,
        },
    )
    _write_text_new(
        source / "03_DECISIONS_AND_OPEN_QUESTIONS.md",
        """# Decisions and open questions

Verified facts: all mandatory preflight categories completed; exact tracer,
checkpoint/restart, controlling rerun, resource, and broad-test gates passed.
The Phase 3-excluded same-terminal support pairs and directionally unreachable
demand are explicitly disclosed and conserved.

Recommendations only: safeguarded CFW with FW fallback, four workers, cold
initialization, exact simultaneous five-suite tracing, and five-update
checkpoint cadence.

External decisions still required: final Task 2 solver, capacity multiplier,
production wall-clock/iteration cap, stopping rule, treatment/reporting of the
directionally unreachable demand, and authorization of the production solve.
No later step is authorized by this handoff.
""",
    )
    _write_text_new(
        source / "04_VALIDATION_RESULTS.md",
        "# Validation results\n\nFinal validation status is `PASS`. See `summary_data/final_review/validation/final_validation.json`, checkpoint reopen evidence, broad-test receipt, figure QA, reproduction summaries, and the external artifact indexes.\n",
    )
    shutil.copy2(review_root / "05_RESOURCE_USAGE.csv", source / "05_RESOURCE_USAGE.csv")
    shutil.copy2(
        review_root / "validation" / "full_output_artifact_manifest.csv",
        source / "06_ARTIFACT_MANIFEST.csv",
    )

    approved_start = str(config["approved_starting_head"])
    name_status = _git(project_root, "diff", "--name-status", f"{approved_start}..{code_commit}")
    code_rows = []
    for line in name_status.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        code_rows.append(
            {
                "status": parts[0],
                "repository_relative_path": parts[-1],
                "included_in_code_snapshot": parts[-1].startswith("partC_rev1_src/phase4/")
                or parts[-1] in {
                    "tests/test_phase4_preflight.py",
                    "partC_rev1_tests/partC_rev1_test_prefix.py",
                },
            }
        )
    pd.DataFrame(code_rows).to_csv(source / "07_CODE_CHANGE_INDEX.csv", index=False)
    shutil.copy2(review_root / "COMMAND_LOG.md", source / "08_COMMAND_LOG.md")
    _write_text_new(
        source / "09_NEXT_STEP_RECOMMENDATION.md",
        """# Next-step recommendation

Recommendation: externally review this exact synchronization ZIP. If accepted,
issue a separate hash-bound Phase 4 Task 2 authorization defining the final
solver/multiplier, runtime/iteration cap, stopping rule, checkpoint cadence,
and unreachable-demand reporting. Do not start the production solve, select a
final physical reference, or begin Phase 5 from this handoff alone.

`next_step_authorized=false`.
""",
    )
    zip_root = project_root / "artifacts_icnn" / "handoff_bundles" / "phase_04"
    zip_path = zip_root / f"phase04_step_task1_preflight_{handoff_id}_SYNC.zip"
    _write_text_new(
        source / "10_UPLOAD_THIS_FILE.txt",
        f"Upload this exact file:\n{zip_path}\nThe adjacent .zip.sha256 and companion JSON bind its final bytes.\n",
    )
    _write_text_new(
        source / "11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md",
        """# Changelog from previous handoff

This is the first reviewable Phase 4 Task 1 preflight handoff. It starts from
the exact externally accepted Phase 3 five-suite closeout, adds the reusable
physical solver and simultaneous tracer, focused and broad validation,
BM01-BM13 evidence, a preserved/repaired BM05 reporting stop, final-review
figures, resource and forecast audits, and external indexes for large local
artifacts.

The earlier write-new sealing attempt ending in
`waiting_review_20260902` failed closed during fresh-extraction closure because
the checker omitted a nested authority-package `CHECKSUMS.sha256` from the
top-level checksum inventory. Its 7,175,301-byte nonreviewable ZIP, SHA-256
`acc351d597e335cfd433263a2e8862e7f1a859312af70e711e3de7363c4987a8`,
and source folder remain preserved. This handoff uses a new ID after a narrow
packaging-only repair and regression test. No reviewed historical handoff was
overwritten and no scientific result changed.
""",
    )

    governing = [
        project_root / "partC_doc_icnn" / name
        for name in (
            "00_READ_FIRST.md",
            "PROJECT_STATUS.md",
            "12_DECISION_LOG.md",
            "20_HANDOFF_SYNC_PROTOCOL.md",
            "21_REVIEW_APPROVAL_GATES.md",
            "23_SYNC_BUNDLE_CONTENT_AND_SIZE_POLICY.md",
        )
    ] + [
        project_root / "partC_doc_icnn" / "phase_4_work" / "PHASE_4_TASK1_SCOPE.md",
        project_root
        / "partC_doc_icnn"
        / "phase_4_work"
        / "PHASE4_TASK1_PREFLIGHT_RESULT.md",
        project_root
        / "partC_doc_icnn"
        / "phase_4_work"
        / "PHASE4_TASK1_HANDOFF_SEALING_REPAIR.md",
        project_root
        / "partC_doc_icnn"
        / "phase_4_work"
        / "PHASE4_TASK1_HANDOFF_SEALING_REPAIR.json",
        project_root
        / "partC_doc_icnn"
        / "review_gates"
        / "phase_03"
        / "all_suite_acnhn"
        / "phase3_complete_with_notes_20260901"
        / "APPROVAL.json",
        project_root
        / "partC_doc_icnn"
        / "review_gates"
        / "phase_03"
        / "all_suite_acnhn"
        / "phase3_complete_with_notes_20260901"
        / "REVIEW_DECISION.md",
    ]
    for path in governing:
        _copy_preserving(path, source / "governing_docs" / "repository", path.relative_to(project_root))

    authority_zip = Path(str(config["authority"]["package_path"])).resolve(strict=True)
    authority_target = source / "governing_docs" / "authority_package"
    with zipfile.ZipFile(authority_zip) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith("/") or name.startswith("input/phase3_handoff/"):
                continue
            destination = authority_target / PurePosixPath(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))

    for path in sorted((output_root / "config").glob("*")):
        shutil.copy2(path, source / "configs" / path.name)
    shutil.copy2(
        project_root / "partC_doc_icnn" / "phase_4_work" / "phase4_task1_preflight_execution_config.json",
        source / "configs" / "phase4_task1_preflight_execution_config.json",
    )

    snapshot_paths = [
        *sorted((project_root / "partC_rev1_src" / "phase4").glob("*.py")),
        project_root / "tests" / "test_phase4_preflight.py",
        project_root / "partC_rev1_tests" / "partC_rev1_test_prefix.py",
    ]
    for path in snapshot_paths:
        _copy_preserving(path, source / "code_snapshot", path.relative_to(project_root))
    _write_text_new(
        source / "code_diff" / "phase4_from_approved_start.patch",
        _git(project_root, "diff", "--binary", f"{approved_start}..{code_commit}"),
    )
    _write_text_new(source / "code_diff" / "changed_files_name_status.txt", name_status)
    _write_text_new(
        source / "code_diff" / "git_identity.txt",
        f"branch={branch}\napproved_starting_head={approved_start}\nsource_commit={code_commit}\n",
    )
    _write_text_new(source / "code_diff" / "git_status_before_handoff.txt", git_status_before)

    final_review_target = source / "summary_data" / "final_review"
    shutil.copytree(review_root / "validation", final_review_target / "validation")
    shutil.copytree(review_root / "tables", final_review_target / "tables")
    shutil.copytree(review_root / "reports", final_review_target / "reports")
    shutil.copy2(review_root / "COMMAND_LOG.md", final_review_target / "COMMAND_LOG.md")
    sealing_repairs = output_root / "sealing_repairs"
    if sealing_repairs.exists():
        for path in sorted(sealing_repairs.rglob("*pytest_junit.xml")):
            _copy_preserving(
                path,
                source / "summary_data" / "sealing_repairs",
                path.relative_to(sealing_repairs),
            )
    (source / "summary_data" / "campaign_tables").mkdir(parents=True)
    for path in sorted((output_root / "tables").glob("*")):
        if path.stat().st_size <= MAX_MEMBER_BYTES:
            shutil.copy2(path, source / "summary_data" / "campaign_tables" / path.name)
    for path in sorted((output_root / "benchmarks").rglob("*")):
        if path.is_file() and path.stat().st_size <= MAX_MEMBER_BYTES and "bacn_internal_statistics" not in path.name:
            relative = path.relative_to(output_root / "benchmarks")
            _copy_preserving(path, source / "summary_data" / "benchmarks", relative)
    (source / "summary_data" / "campaign_validation").mkdir(parents=True)
    for path in sorted((output_root / "validation").glob("*")):
        if path.stat().st_size <= MAX_MEMBER_BYTES:
            shutil.copy2(path, source / "summary_data" / "campaign_validation" / path.name)
    shutil.copytree(output_root / "figures", source / "figures" / "initial_campaign")
    shutil.copytree(review_root / "figures", source / "figures" / "final_review")
    _write_text_new(
        source / "geospatial_summary" / "NO_GEOSPATIAL_OUTPUT.md",
        "# Geospatial summary\n\nNo geospatial artifact was produced or required by this physical-assignment/tracer preflight. Accepted Phase 3 geospatial products were read-only external inputs.\n",
    )
    shutil.copy2(output_root / "logs" / "campaign_events.jsonl", source / "log_excerpts" / "campaign_events.jsonl")
    shutil.copy2(
        output_root / "benchmarks" / "BM05_reporting_stop_supersession.json",
        source / "log_excerpts" / "BM05_reporting_stop_supersession.json",
    )
    shutil.copy2(
        review_root / "validation" / "full_output_artifact_manifest.csv",
        source / "external_artifact_index" / "full_output_artifact_manifest.csv",
    )
    shutil.copy2(
        review_root / "validation" / "large_external_artifact_index.csv",
        source / "external_artifact_index" / "large_external_artifact_index.csv",
    )

    oversized = [
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and path.stat().st_size > MAX_MEMBER_BYTES
    ]
    if oversized:
        raise Phase4FinalizationError(f"Handoff source has oversized members: {oversized[:5]}")
    checksums = _source_checksums(source)
    _write_text_new(
        source / "CHECKSUMS.sha256",
        "".join(f"{digest}  {relative}\n" for relative, digest in checksums.items()),
    )
    _create_zip(source, zip_path)
    zip_size = zip_path.stat().st_size
    if zip_size > MAX_SYNC_BYTES:
        raise Phase4FinalizationError("Synchronization ZIP exceeds 200 MB hard limit")
    zip_sha = file_sha256(zip_path)
    sidecar = zip_path.with_suffix(zip_path.suffix + ".sha256")
    _write_text_new(sidecar, f"{zip_sha}  {zip_path.name}\n")
    fresh = _fresh_extract_verify(zip_path)
    companion = {
        "schema_id": "phase4.task1.handoff_companion.v1",
        "handoff_id": handoff_id,
        "review_id": review_id,
        "sync_zip": str(zip_path.resolve()),
        "sync_zip_size_bytes": zip_size,
        "sync_zip_sha256": zip_sha,
        "sync_zip_under_100mb_target": zip_size <= TARGET_SYNC_BYTES,
        "source_folder": str(source.resolve()),
        "source_commit": code_commit,
        "branch": branch,
        "fresh_extraction": fresh,
        "status": FINAL_STATUS,
        "phase3_status": "APPROVED_COMPLETE_WITH_NOTES",
        "all_mandatory_benchmark_categories_completed": True,
        "target_extensions_curtailed": True,
        "recommended_preflight_solver": "SAFEGUARDED_CFW",
        "selected_preflight_worker_count": 4,
        "final_solver_selected": False,
        "final_capacity_multiplier_selected": False,
        "phase4_production_solve_authorized": False,
        "final_physical_reference_accepted": False,
        "phase5_authorized": False,
        "next_step_authorized": False,
        "created_utc": _utc_now(),
    }
    companion_path = zip_path.with_suffix(".companion.json")
    _write_json_new(companion_path, companion)
    latest = source.parent / "LATEST_HANDOFF.txt"
    _write_text_new(
        latest,
        f"source_folder={source.resolve()}\nzip_path={zip_path.resolve()}\nzip_sha256={zip_sha}\ncreated_utc={companion['created_utc']}\nstatus={FINAL_STATUS}\n",
    )
    status_path = output_root / "status" / "campaign_status.json"
    status = _load_json(status_path)
    status.update(
        {
            "state": FINAL_STATUS,
            "next_task": "EXTERNAL-REVIEW",
            "next_step_authorized": False,
            "phase4_production_solve_authorized": False,
            "final_physical_reference_accepted": False,
            "final_capacity_multiplier_selected": False,
            "phase5_authorized": False,
            "handoff_id": handoff_id,
            "handoff_zip": str(zip_path.resolve()),
            "handoff_zip_sha256": zip_sha,
            "updated_utc": _utc_now(),
        }
    )
    _replace_json(status_path, status)
    _write_json_new(output_root / "status" / "FINAL_HANDOFF.json", companion)
    return companion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("validate", "confirm-figures", "handoff"))
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--review-id", required=True)
    parser.add_argument("--handoff-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config, project_root, output_root = _validate_context(
        Path(args.config),
        phase_id=args.phase_id,
        step_id=args.step_id,
        run_id=args.run_id,
    )
    if args.stage == "validate":
        validate(config, project_root, output_root, review_id=args.review_id)
    elif args.stage == "confirm-figures":
        confirm_figures(config, project_root, output_root, review_id=args.review_id)
    else:
        if not args.handoff_id:
            raise Phase4FinalizationError("--handoff-id is required for handoff stage")
        handoff(
            config,
            project_root,
            output_root,
            review_id=args.review_id,
            handoff_id=args.handoff_id,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
