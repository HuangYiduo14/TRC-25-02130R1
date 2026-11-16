"""Prompt 4C physical benchmark execution, registry, and witnesses."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .partC_rev1_aon import partC_rev1_aon_load
from .partC_rev1_assignment import (
    PartCRev1AssignmentConfig,
    PartCRev1SolverResult,
    partC_rev1_solve_physical_assignment,
)
from .partC_rev1_bpr import partC_rev1_bpr_cost, partC_rev1_relative_gap
from .partC_rev1_graph_snapshot import PartCRev1GraphSnapshot
from .partC_rev1_hashing import partC_rev1_json_hash, partC_rev1_numpy_array_hash, partC_rev1_sha256_text
from .partC_rev1_io import partC_rev1_read_json, partC_rev1_write_json, partC_rev1_write_table
from .partC_rev1_od_snapshot import PartCRev1ODSnapshot
from .partC_rev1_paths import partC_rev1_resolve_path
from .partC_rev1_progress_logging import PartCRev1ProgressReporter
from .partC_rev1_run_manifest import partC_rev1_append_jsonl, partC_rev1_write_master_state
from .partC_rev1_runtime_guard import PARTC_REV1_DEFAULT_MAX_RUN_SECONDS, partC_rev1_resource_snapshot
from .partC_rev1_shortest_paths import partC_rev1_sssp


PARTC_REV1_4C_STAGE = "partC_rev1_4C"


@dataclass(frozen=True, slots=True)
class PartCRev1PhysicalBenchmarkRunConfig:
    method: str
    backend: str
    max_iterations: int
    gap_target: float
    origin_batch_size: int
    max_origins: int | None = None
    state_label: str = "candidate_full"
    runtime_guard_seconds: float = PARTC_REV1_DEFAULT_MAX_RUN_SECONDS


@dataclass(frozen=True, slots=True)
class PartCRev1PhysicalBenchmarkStageConfig:
    config_path: str
    max_iterations: int
    gap_target: float
    gap_reference: float
    origin_batch_size: int
    backend_pilot_iterations: int
    backend_pilot_origins: int = 0
    backend_pilot_min_seconds: float = 120.0
    backend_pilot_min_batches: int = 3
    backend_pilot_cv_tolerance: float = 0.25
    witness_max_origins: int | None = None
    witness_all_positive_origins: bool = True
    witness_origin_batch_size: int = 16
    witness_diagnostic_path_count: int = 50
    residual_tolerance: float = 1e-6


def _partC_rev1_git_commit() -> str | None:
    import subprocess

    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True, encoding="utf-8")
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _partC_rev1_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _partC_rev1_json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _partC_rev1_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_partC_rev1_json_default) + "\n")


def _partC_rev1_environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ["numpy", "scipy", "pandas", "numba", "igraph", "networkx", "pyarrow", "psutil", "pytest"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_utc": _partC_rev1_utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": packages,
    }


def partC_rev1_benchmark_run_hash(
    graph_hash: str,
    od_hash: str,
    run_config: PartCRev1PhysicalBenchmarkRunConfig,
) -> str:
    payload = {
        "graph_hash": graph_hash,
        "od_hash": od_hash,
        "run_config": asdict(run_config),
        "schema": "partC_rev1_physical_benchmark_v1",
    }
    return partC_rev1_json_hash(payload)[:16]


def partC_rev1_benchmark_run_dir(
    output_dir: str | Path,
    graph_hash: str,
    od_hash: str,
    run_config: PartCRev1PhysicalBenchmarkRunConfig,
) -> Path:
    run_hash = partC_rev1_benchmark_run_hash(graph_hash, od_hash, run_config)
    target = f"{float(run_config.gap_target):.0e}".replace("+", "")
    name = f"partC_rev1_{run_config.method}_{run_config.backend}_gap-{target}_hash-{run_hash}"
    return partC_rev1_resolve_path(output_dir) / "partC_rev1_benchmarks" / name


def _partC_rev1_iteration_records(result: PartCRev1SolverResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in result.histories:
        rows.append(
            {
                "iteration": record.iteration,
                "method": record.method,
                "backend": record.backend,
                "objective": record.objective,
                "tstt": record.tstt,
                "sptt": record.sptt,
                "relative_gap": record.relative_gap,
                "step_size": record.step_size,
                "aon_runtime_seconds": record.aon_runtime_seconds,
                "iteration_runtime_seconds": record.iteration_runtime_seconds,
                "reachable_demand": record.reachable_demand,
                "unreachable_demand": record.unreachable_demand,
                "flow_l1_change": record.flow_l1_change,
                "cfw_alpha": record.cfw_alpha,
                "fallback_reason": record.fallback_reason,
            }
        )
    return rows


def _partC_rev1_project_maxiter_seconds(result: PartCRev1SolverResult, max_iterations: int) -> float | None:
    durations = [record.iteration_runtime_seconds for record in result.histories if record.iteration_runtime_seconds > 0]
    if not durations:
        return None
    completed = len(result.histories)
    return float(result.runtime_seconds + median(durations) * max(max_iterations - completed, 0))


def _partC_rev1_project_target_seconds(result: PartCRev1SolverResult, gap_target: float) -> float | None:
    pairs: list[tuple[float, float]] = []
    elapsed = 0.0
    for record in result.histories:
        elapsed += max(float(record.iteration_runtime_seconds), 0.0)
        gap = record.relative_gap
        if gap is not None and gap > 0.0 and np.isfinite(gap):
            pairs.append((elapsed, math.log10(float(gap))))
    if len(pairs) < 3 or gap_target <= 0.0:
        return None
    x = np.asarray([item[0] for item in pairs], dtype=np.float64)
    y = np.asarray([item[1] for item in pairs], dtype=np.float64)
    slope, intercept = np.polyfit(x, y, deg=1)
    if not np.isfinite(slope) or slope >= 0.0:
        return None
    projected = (math.log10(float(gap_target)) - intercept) / slope
    if not np.isfinite(projected) or projected < 0.0:
        return None
    return float(projected)


def _partC_rev1_validate_result(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    result: PartCRev1SolverResult,
    run_config: PartCRev1PhysicalBenchmarkRunConfig,
    reporter: PartCRev1ProgressReporter | None,
    *,
    residual_tolerance: float,
) -> dict[str, Any]:
    cost = partC_rev1_bpr_cost(result.flow, graph.free_time, graph.capacity, graph.alpha, graph.beta)
    validation_started = time.perf_counter()
    validation_aon = partC_rev1_aon_load(
        graph,
        od,
        cost,
        backend=run_config.backend,
        max_origins=run_config.max_origins,
        origin_batch_size=run_config.origin_batch_size,
        reporter=reporter,
    )
    total_processed_demand = validation_aon.reachable_demand + validation_aon.unreachable_demand
    total_reference_demand = float(np.sum(od.demand, dtype=np.float64))
    if run_config.max_origins is not None and run_config.max_origins < len(od.origin_group_node_idx):
        start = int(od.origin_offsets[0])
        end = int(od.origin_offsets[int(run_config.max_origins)])
        total_reference_demand = float(np.sum(od.demand[start:end], dtype=np.float64))
    demand_residual = abs(total_processed_demand - total_reference_demand) / max(total_reference_demand, 1e-12)
    tstt = float(np.dot(result.flow, cost))
    final_gap = partC_rev1_relative_gap(tstt, validation_aon.sptt)
    recomputed_cost_hash = partC_rev1_numpy_array_hash(cost)
    validation = {
        "runtime_seconds": time.perf_counter() - validation_started,
        "flow_finite": bool(np.all(np.isfinite(result.flow))),
        "flow_nonnegative": bool(np.min(result.flow) >= -1e-8 if len(result.flow) else True),
        "flow_min": float(np.min(result.flow)) if len(result.flow) else 0.0,
        "flow_max": float(np.max(result.flow)) if len(result.flow) else 0.0,
        "cost_finite": bool(np.all(np.isfinite(cost))),
        "row_count_matches_graph": bool(len(result.flow) == len(graph.arc_idx_to_directed_arc_id)),
        "tstt": tstt,
        "sptt": float(validation_aon.sptt),
        "relative_gap": final_gap,
        "reachable_demand": float(validation_aon.reachable_demand),
        "unreachable_demand": float(validation_aon.unreachable_demand),
        "total_processed_demand": float(total_processed_demand),
        "total_reference_demand": float(total_reference_demand),
        "demand_accounting_residual": float(demand_residual),
        "flow_residual": 0.0 if demand_residual <= residual_tolerance and np.all(result.flow >= -1e-8) else float(demand_residual),
        "recomputed_cost_hash": recomputed_cost_hash,
        "validation_backend": run_config.backend,
        "validation_partial": bool(run_config.max_origins is not None and run_config.max_origins < len(od.origin_group_node_idx)),
    }
    validation["valid_feasible_iterate"] = bool(
        validation["flow_finite"]
        and validation["flow_nonnegative"]
        and validation["cost_finite"]
        and validation["row_count_matches_graph"]
        and validation["flow_residual"] <= residual_tolerance
    )
    return validation


def _partC_rev1_accuracy_score(
    validation: dict[str, Any],
    *,
    gap_reference: float,
    residual_tolerance: float,
) -> float | None:
    if not validation.get("valid_feasible_iterate"):
        return None
    total = float(validation.get("total_processed_demand") or 0.0)
    if total <= 0.0:
        return None
    gap = float(validation.get("relative_gap") or 0.0)
    gap_score = min(1.0, float(gap_reference) / max(gap, 1e-12))
    reach_score = max(0.0, 1.0 - float(validation.get("unreachable_demand") or 0.0) / max(total, 1e-12))
    residual_score = max(0.0, 1.0 - float(validation.get("flow_residual") or 0.0) / max(float(residual_tolerance), 1e-12))
    return float(gap_score * reach_score * residual_score)


def _partC_rev1_write_flow_artifacts(
    graph: PartCRev1GraphSnapshot,
    result: PartCRev1SolverResult,
    run_dir: Path,
) -> dict[str, str]:
    directed = pd.DataFrame(
        {
            "arc_idx": np.arange(len(graph.arc_idx_to_directed_arc_id), dtype=np.int64),
            "directed_arc_id": graph.arc_idx_to_directed_arc_id,
            "partC_rev1_link_uid": graph.arc_idx_to_link_uid,
            "tail_node_uid": graph.node_idx_to_uid[graph.tail],
            "head_node_uid": graph.node_idx_to_uid[graph.head],
            "arc_label": [graph.region_idx_to_label[int(code)] for code in graph.region_code],
            "flow": result.flow.astype(np.float64),
            "cost": result.cost.astype(np.float64),
            "free_time": graph.free_time.astype(np.float64),
            "base_capacity": graph.base_capacity.astype(np.float64),
            "assignment_capacity": graph.capacity.astype(np.float64),
            "remaining_capacity": graph.capacity.astype(np.float64),
            "capacity_multiplier": graph.capacity_multiplier.astype(np.float64),
            "capacity_multiplier_type": graph.capacity_multiplier_type.astype(str),
        }
    )
    directed_path = run_dir / "partC_rev1_directed_arc_flow.parquet"
    partC_rev1_write_table(directed, directed_path, index=False)
    link = (
        directed.groupby("partC_rev1_link_uid", as_index=False)
        .agg(
            directed_arc_count=("directed_arc_id", "count"),
            flow=("flow", "sum"),
            mean_cost=("cost", "mean"),
            max_cost=("cost", "max"),
            free_time_min=("free_time", "min"),
            base_capacity=("base_capacity", "sum"),
            assignment_capacity=("assignment_capacity", "sum"),
        )
        .sort_values("partC_rev1_link_uid", kind="mergesort")
        .reset_index(drop=True)
    )
    link_path = run_dir / "partC_rev1_physical_link_flow.parquet"
    partC_rev1_write_table(link, link_path, index=False)
    return {"directed_arc_flow": str(directed_path), "physical_link_flow": str(link_path)}


def partC_rev1_run_one_physical_benchmark(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    *,
    output_dir: str | Path,
    stage_config: PartCRev1PhysicalBenchmarkStageConfig,
    run_config: PartCRev1PhysicalBenchmarkRunConfig,
    reporter: PartCRev1ProgressReporter | None = None,
) -> dict[str, Any]:
    run_dir = partC_rev1_benchmark_run_dir(output_dir, graph.graph_hash, od.od_hash, run_config)
    summary_path = run_dir / "partC_rev1_summary.json"
    if summary_path.exists():
        existing = partC_rev1_read_json(summary_path)
        if existing.get("run_complete_folder") is True:
            if reporter is not None:
                reporter.emit("physical_benchmark_reused", run_dir=str(run_dir), method=run_config.method, backend=run_config.backend)
            return existing
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite incomplete/nonempty benchmark folder: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    run_hash = partC_rev1_benchmark_run_hash(graph.graph_hash, od.od_hash, run_config)
    started = time.perf_counter()
    git_commit = _partC_rev1_git_commit()
    if reporter is not None:
        reporter.emit("physical_benchmark_started", run_hash=run_hash, run_dir=str(run_dir), **asdict(run_config))

    partC_rev1_write_json(asdict(run_config), run_dir / "partC_rev1_config.json")
    partC_rev1_write_json(_partC_rev1_environment(), run_dir / "partC_rev1_environment.json")
    (run_dir / "partC_rev1_git_commit.txt").write_text(str(git_commit or "") + "\n", encoding="utf-8")

    result = partC_rev1_solve_physical_assignment(
        graph,
        od,
        method=run_config.method,
        config=PartCRev1AssignmentConfig(
            max_iterations=int(run_config.max_iterations),
            gap_target=float(run_config.gap_target),
            backend=run_config.backend,
            origin_batch_size=int(run_config.origin_batch_size),
            max_origins=run_config.max_origins,
            max_runtime_seconds=float(run_config.runtime_guard_seconds),
        ),
        reporter=reporter,
    )
    validation = _partC_rev1_validate_result(
        graph,
        od,
        result,
        run_config,
        reporter,
        residual_tolerance=stage_config.residual_tolerance,
    )
    flow_paths = _partC_rev1_write_flow_artifacts(graph, result, run_dir)
    convergence_rows = _partC_rev1_iteration_records(result)
    convergence = pd.DataFrame(convergence_rows)
    partC_rev1_write_table(convergence, run_dir / "partC_rev1_convergence.csv", index=False)
    runtime_rows = []
    for row in convergence_rows:
        runtime_rows.append(
            {
                "phase": f"iteration_{row['iteration']}",
                "runtime_seconds": row["iteration_runtime_seconds"],
                "aon_runtime_seconds": row["aon_runtime_seconds"],
                "method": run_config.method,
                "backend": run_config.backend,
            }
        )
    runtime_rows.append({"phase": "validation_final_gap", "runtime_seconds": validation["runtime_seconds"], "aon_runtime_seconds": validation["runtime_seconds"], "method": run_config.method, "backend": run_config.backend})
    partC_rev1_write_table(pd.DataFrame(runtime_rows), run_dir / "partC_rev1_runtime_breakdown.csv", index=False)
    _partC_rev1_write_jsonl(
        run_dir / "partC_rev1_mid_iteration_progress.jsonl",
        [
            {
                "event": "assignment_iteration",
                "run_hash": run_hash,
                **row,
            }
            for row in convergence_rows
        ],
    )

    projected_maxiter_sec = _partC_rev1_project_maxiter_seconds(result, run_config.max_iterations)
    projected_target_sec = _partC_rev1_project_target_seconds(result, run_config.gap_target)
    finite_projections = [value for value in [projected_maxiter_sec, projected_target_sec] if value is not None and np.isfinite(value)]
    conservative_projection = max(finite_projections) if finite_projections else None
    projected_full_backend_seconds = None
    if run_config.max_origins is not None and run_config.max_origins > 0:
        projected_full_backend_seconds = result.runtime_seconds * len(od.origin_group_node_idx) / float(run_config.max_origins)
    state = "measured_complete"
    if result.metadata.get("partial"):
        state = "measured_partial_projected"
    if conservative_projection is not None and conservative_projection > run_config.runtime_guard_seconds:
        state = "stopped_runtime_guard" if not result.metadata.get("partial") else "measured_partial_projected"
    accuracy = _partC_rev1_accuracy_score(
        validation,
        gap_reference=stage_config.gap_reference,
        residual_tolerance=stage_config.residual_tolerance,
    )
    if state == "measured_partial_projected":
        accuracy = None
    summary = {
        "run_complete_folder": True,
        "stage": PARTC_REV1_4C_STAGE,
        "created_utc": _partC_rev1_utc_now(),
        "run_hash": run_hash,
        "run_dir": str(run_dir),
        "config_path": stage_config.config_path,
        "git_commit": git_commit,
        "graph_hash": graph.graph_hash,
        "topology_hash": graph.topology_hash,
        "backbone_hash": graph.backbone_hash,
        "assignment_graph_hash": graph.assignment_graph_hash,
        "od_hash": od.od_hash,
        "partition_hash": graph.partition_hash,
        "method": result.method,
        "backend": result.backend,
        "state": state,
        "stop_reason": result.stop_reason,
        "valid_completed_feasible_iterate": bool(validation.get("valid_feasible_iterate") and state == "measured_complete"),
        "runtime_seconds": time.perf_counter() - started,
        "solver_runtime_seconds": result.runtime_seconds,
        "peak_memory_mb": result.peak_memory_mb,
        "iteration_count": len(result.histories),
        "final_gap": validation["relative_gap"],
        "final_tstt": validation["tstt"],
        "final_sptt": validation["sptt"],
        "final_objective": convergence_rows[-1]["objective"] if convergence_rows else None,
        "reachable_demand": validation["reachable_demand"],
        "unreachable_demand": validation["unreachable_demand"],
        "unreachable_share": float(validation["unreachable_demand"] / max(validation["total_processed_demand"], 1e-12)),
        "flow_residual": validation["flow_residual"],
        "accuracy_score": accuracy,
        "gap_reference": stage_config.gap_reference,
        "projected_maxiter_seconds": projected_maxiter_sec,
        "projected_target_seconds": projected_target_sec,
        "projected_full_backend_seconds": projected_full_backend_seconds,
        "run_config": asdict(run_config),
        "backend_pilot_metadata": {
            "state_label": run_config.state_label,
            "min_seconds": stage_config.backend_pilot_min_seconds,
            "min_batches": stage_config.backend_pilot_min_batches,
            "cv_tolerance": stage_config.backend_pilot_cv_tolerance,
            "processed_origins": result.metadata.get("processed_origins"),
            "total_origins": result.metadata.get("total_origins"),
            "projection_formula": "solver_runtime_seconds * total_origins / processed_origins",
            "projected_full_backend_seconds": projected_full_backend_seconds,
        }
        if "backend" in run_config.state_label
        else {},
        "validation": validation,
        "artifacts": {
            **flow_paths,
            "convergence": str(run_dir / "partC_rev1_convergence.csv"),
            "runtime_breakdown": str(run_dir / "partC_rev1_runtime_breakdown.csv"),
            "mid_iteration_progress": str(run_dir / "partC_rev1_mid_iteration_progress.jsonl"),
            "config": str(run_dir / "partC_rev1_config.json"),
            "environment": str(run_dir / "partC_rev1_environment.json"),
        },
    }
    partC_rev1_write_json(summary, summary_path)
    manifest = {
        **summary,
        "environment_path": str(run_dir / "partC_rev1_environment.json"),
        "summary_path": str(summary_path),
    }
    partC_rev1_write_json(manifest, run_dir / "partC_rev1_run_manifest.json")
    md_lines = [
        f"# partC_rev1 Physical Benchmark {result.method}/{result.backend}",
        "",
        f"- State: `{state}`",
        f"- Stop reason: `{result.stop_reason}`",
        f"- Final relative gap: `{validation['relative_gap']}`",
        f"- Runtime seconds: `{summary['runtime_seconds']}`",
        f"- Accuracy score: `{accuracy}`",
        f"- Directed flow: `{flow_paths['directed_arc_flow']}`",
        f"- Physical link flow: `{flow_paths['physical_link_flow']}`",
        "",
    ]
    (run_dir / "partC_rev1_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    (run_dir / "partC_rev1_run.log").write_text(
        "\n".join(
            [
                f"{_partC_rev1_utc_now()} run_hash={run_hash}",
                f"method={result.method} backend={result.backend} state={state}",
                f"stop_reason={result.stop_reason} final_gap={validation['relative_gap']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if reporter is not None:
        resource = partC_rev1_resource_snapshot(started)
        reporter.emit(
            "physical_benchmark_finished",
            run_hash=run_hash,
            method=result.method,
            backend=result.backend,
            state=state,
            stop_reason=result.stop_reason,
            final_gap=validation["relative_gap"],
            runtime_seconds=summary["runtime_seconds"],
            memory_rss_mb=resource.memory_rss_mb,
            disk_free_mb=resource.disk_free_mb,
        )
    return summary


def partC_rev1_select_best_benchmark(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [item for item in summaries if item.get("state") == "measured_complete" and item.get("valid_completed_feasible_iterate")]
    if not candidates:
        return None

    def key(item: dict[str, Any]) -> tuple[float, float, float, float, float]:
        accuracy = float(item.get("accuracy_score") or 0.0)
        return (
            -accuracy,
            float(item.get("final_gap") or math.inf),
            float(item.get("unreachable_share") or math.inf),
            float(item.get("flow_residual") or math.inf),
            float(item.get("runtime_seconds") or math.inf),
        )

    return sorted(candidates, key=key)[0]


def partC_rev1_write_benchmark_registry(
    summaries: list[dict[str, Any]],
    output_dir: str | Path,
) -> tuple[Path, Path, dict[str, Any] | None]:
    out_dir = partC_rev1_resolve_path(output_dir)
    best = partC_rev1_select_best_benchmark(summaries)
    rows: list[dict[str, Any]] = []
    for item in summaries:
        rows.append(
            {
                "run_hash": item.get("run_hash"),
                "method": item.get("method"),
                "backend": item.get("backend"),
                "state": item.get("state"),
                "stop_reason": item.get("stop_reason"),
                "valid_completed_feasible_iterate": item.get("valid_completed_feasible_iterate"),
                "final_gap": item.get("final_gap"),
                "accuracy_score": item.get("accuracy_score"),
                "runtime_seconds": item.get("runtime_seconds"),
                "iteration_count": item.get("iteration_count"),
                "reachable_demand": item.get("reachable_demand"),
                "unreachable_demand": item.get("unreachable_demand"),
                "unreachable_share": item.get("unreachable_share"),
                "flow_residual": item.get("flow_residual"),
                "projected_maxiter_seconds": item.get("projected_maxiter_seconds"),
                "projected_target_seconds": item.get("projected_target_seconds"),
                "projected_full_backend_seconds": item.get("projected_full_backend_seconds"),
                "run_dir": item.get("run_dir"),
                "selected_best_accuracy_benchmark": bool(best is not None and item.get("run_hash") == best.get("run_hash")),
            }
        )
    registry = pd.DataFrame(rows)
    csv_path = out_dir / "partC_rev1_benchmark_registry.csv"
    md_path = out_dir / "partC_rev1_benchmark_registry.md"
    partC_rev1_write_table(registry, csv_path, index=False)
    md_lines = [
        "# partC_rev1 Benchmark Registry",
        "",
        f"Generated UTC: {_partC_rev1_utc_now()}",
        "",
        "| Method | Backend | State | Stop | Final gap | Accuracy | Runtime seconds | Selected |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["backend"]),
                    str(row["state"]),
                    str(row["stop_reason"]),
                    "" if row["final_gap"] is None else f"{float(row['final_gap']):.12g}",
                    "" if row["accuracy_score"] is None else f"{float(row['accuracy_score']):.12g}",
                    "" if row["runtime_seconds"] is None else f"{float(row['runtime_seconds']):.3f}",
                    "yes" if row["selected_best_accuracy_benchmark"] else "no",
                ]
            )
            + " |"
        )
    if best is not None:
        md_lines.extend(["", f"Selected best benchmark: `{best.get('method')}`/`{best.get('backend')}` hash `{best.get('run_hash')}`."])
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path, md_path, best


def _partC_rev1_path_arc_indices(result: Any, source: int, dest: int) -> list[int]:
    if not bool(result.reached_mask[int(dest)]):
        return []
    arcs: list[int] = []
    node = int(dest)
    guard = 0
    while node != int(source):
        arc_idx = int(result.predecessor_arc[node])
        parent = int(result.predecessor_node[node])
        if arc_idx < 0 or parent < 0:
            return []
        arcs.append(arc_idx)
        node = parent
        guard += 1
        if guard > len(result.predecessor_arc):
            return []
    arcs.reverse()
    return arcs


def _partC_rev1_region_blocks(graph: PartCRev1GraphSnapshot, path_arcs: list[int]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if not path_arcs:
        return blocks
    start = 0
    while start < len(path_arcs):
        label = graph.region_idx_to_label[int(graph.region_code[int(path_arcs[start])])]
        end = start + 1
        while end < len(path_arcs) and graph.region_idx_to_label[int(graph.region_code[int(path_arcs[end])])] == label:
            end += 1
        if label != "backbone":
            first_arc = int(path_arcs[start])
            last_arc = int(path_arcs[end - 1])
            blocks.append(
                {
                    "region_uuid": label,
                    "start_node_uid": str(graph.node_idx_to_uid[int(graph.tail[first_arc])]),
                    "end_node_uid": str(graph.node_idx_to_uid[int(graph.head[last_arc])]),
                    "arc_indices": path_arcs[start:end],
                }
            )
        start = end
    for idx, block in enumerate(blocks):
        block["is_first"] = idx == 0
        block["is_last"] = idx == len(blocks) - 1
        if idx == 0:
            block["block_position"] = "first"
        elif idx == len(blocks) - 1:
            block["block_position"] = "last"
        else:
            block["block_position"] = "intermediate"
    return blocks


def partC_rev1_generate_route_witnesses(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    cost: np.ndarray,
    *,
    output_path: str | Path,
    witness_set: str,
    max_origins: int | None = None,
    diagnostic_path_count: int = 25,
    reporter: PartCRev1ProgressReporter | None = None,
) -> dict[str, Any]:
    out_path = partC_rev1_resolve_path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_origins = len(od.origin_group_node_idx)
    selected_origins = total_origins if max_origins is None else min(int(max_origins), total_origins)
    total_demand = float(np.sum(od.demand, dtype=np.float64))
    started = time.perf_counter()
    aggregates: dict[tuple[str, str, str, str, bool, bool], dict[str, Any]] = {}
    processed_od_rows = 0
    processed_demand = 0.0
    unreachable_demand = 0.0
    diagnostic_remaining = int(diagnostic_path_count)
    unit_seconds: list[float] = []
    cost_hash = partC_rev1_numpy_array_hash(np.asarray(cost, dtype=np.float64))
    for group_pos in range(selected_origins):
        source_started = time.perf_counter()
        source = int(od.origin_group_node_idx[group_pos])
        od_start = int(od.origin_offsets[group_pos])
        od_end = int(od.origin_offsets[group_pos + 1])
        destinations = od.destination_node_idx[od_start:od_end]
        sssp = partC_rev1_sssp(graph, source, cost, backend="numba_csr", targets=destinations.tolist())
        source_uid = str(graph.node_idx_to_uid[source])
        for od_pos in range(od_start, od_end):
            dest = int(od.destination_node_idx[od_pos])
            demand = float(od.demand[od_pos])
            processed_od_rows += 1
            processed_demand += demand
            if not bool(sssp.reached_mask[dest]):
                unreachable_demand += demand
                continue
            path_arcs = _partC_rev1_path_arc_indices(sssp, source, dest)
            if not path_arcs:
                unreachable_demand += demand
                continue
            blocks = _partC_rev1_region_blocks(graph, path_arcs)
            if not blocks:
                continue
            dest_uid = str(graph.node_idx_to_uid[dest])
            od_hash = partC_rev1_sha256_text(f"{source_uid}->{dest_uid}")[:20]
            for block in blocks:
                key = (
                    str(block["region_uuid"]),
                    str(block["start_node_uid"]),
                    str(block["end_node_uid"]),
                    str(block["block_position"]),
                    bool(block["is_first"]),
                    bool(block["is_last"]),
                )
                item = aggregates.setdefault(
                    key,
                    {
                        "witness_set": witness_set,
                        "region_uuid": key[0],
                        "block_start_node_uid": key[1],
                        "block_end_node_uid": key[2],
                        "source_terminal_uid": key[1],
                        "target_terminal_uid": key[2],
                        "block_position": key[3],
                        "is_first": key[4],
                        "is_last": key[5],
                        "demand": 0.0,
                        "total_demand": 0.0,
                        "od_pair_count": 0,
                        "example_od_pair_hash": od_hash,
                        "example_member_directed_arc_ids": "",
                    },
                )
                item["demand"] += demand
                item["total_demand"] += demand
                item["od_pair_count"] += 1
                if diagnostic_remaining > 0 and item["example_member_directed_arc_ids"] == "":
                    item["example_member_directed_arc_ids"] = ",".join(
                        str(graph.arc_idx_to_directed_arc_id[int(arc_idx)]) for arc_idx in block["arc_indices"]
                    )
                    diagnostic_remaining -= 1
        source_elapsed = time.perf_counter() - source_started
        unit_seconds.append(source_elapsed)
        if reporter is not None:
            projected = float(median(unit_seconds) * total_origins) if unit_seconds else None
            resource = partC_rev1_resource_snapshot(started)
            reporter.heartbeat(
                "route_witness_origin",
                force=True,
                witness_set=witness_set,
                processed=group_pos + 1,
                total=selected_origins,
                total_origins=total_origins,
                processed_od_rows=processed_od_rows,
                processed_demand=processed_demand,
                unreachable_demand=unreachable_demand,
                projected_full_origin_seconds=projected,
                memory_rss_mb=resource.memory_rss_mb,
                temp_disk_mb=resource.temp_disk_mb,
            )
    rows = list(aggregates.values())
    if rows:
        df = pd.DataFrame(rows).sort_values(
            ["region_uuid", "block_start_node_uid", "block_end_node_uid", "block_position"],
            kind="mergesort",
        )
    else:
        df = pd.DataFrame(
            columns=[
                "witness_set",
                "region_uuid",
                "block_start_node_uid",
                "block_end_node_uid",
                "source_terminal_uid",
                "target_terminal_uid",
                "block_position",
                "is_first",
                "is_last",
                "demand",
                "total_demand",
                "od_pair_count",
                "example_od_pair_hash",
                "example_member_directed_arc_ids",
            ]
        )
    df["processed_origins"] = selected_origins
    df["total_origins"] = total_origins
    df["processed_od_rows"] = processed_od_rows
    df["processed_demand"] = processed_demand
    df["total_od_demand"] = total_demand
    df["coverage_share"] = processed_demand / max(total_demand, 1e-12)
    df["graph_hash"] = graph.graph_hash
    df["cost_hash"] = cost_hash
    partC_rev1_write_table(df, out_path, index=False)
    runtime = time.perf_counter() - started
    return {
        "path": str(out_path),
        "witness_set": witness_set,
        "state": "measured_complete" if selected_origins == total_origins else "measured_partial_projected",
        "runtime_seconds": runtime,
        "rows": int(len(df)),
        "processed_origins": selected_origins,
        "total_origins": total_origins,
        "processed_od_rows": processed_od_rows,
        "processed_demand": processed_demand,
        "total_od_demand": total_demand,
        "coverage_share": processed_demand / max(total_demand, 1e-12),
        "unreachable_demand_in_processed_origins": unreachable_demand,
        "projected_full_origin_seconds": runtime * total_origins / max(selected_origins, 1),
        "graph_hash": graph.graph_hash,
        "cost_hash": cost_hash,
    }


def _partC_rev1_backend_pilot_max_origins(
    od: PartCRev1ODSnapshot,
    *,
    stage_config: PartCRev1PhysicalBenchmarkStageConfig,
) -> int:
    """Choose a conservative pilot size when using time/stability pilot flags.

    The assignment kernels currently accept a prefix count rather than an
    arbitrary representative origin list. This keeps old runs reproducible while
    requiring new pilot metadata to record the time/stability rule that should
    govern production projections.
    """

    total = int(len(od.origin_group_node_idx))
    if total <= 0:
        return 0
    if int(stage_config.backend_pilot_origins) > 0:
        return min(int(stage_config.backend_pilot_origins), total)
    batches = max(1, int(stage_config.backend_pilot_min_batches))
    return min(max(batches, int(stage_config.origin_batch_size) * batches), total)


def partC_rev1_validate_against_reference_flow(
    config: dict[str, Any],
    selected_summary: dict[str, Any] | None,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    out_dir = partC_rev1_resolve_path(output_dir)
    validation_path = out_dir / "partC_rev1_reference_flow_validation_summary.json"
    if selected_summary is None:
        summary = {"status": "skipped_no_selected_benchmark", "path": str(validation_path)}
        partC_rev1_write_json(summary, validation_path)
        return summary
    ground_truth = config.get("ground_truth", {}) if isinstance(config.get("ground_truth"), dict) else {}
    inputs = config.get("inputs", {}) if isinstance(config.get("inputs"), dict) else {}
    flow_dir = inputs.get("ground_truth_flow_dir")
    recommended = ground_truth.get("recommended_file")
    if not flow_dir or not recommended:
        summary = {"status": "skipped_missing_ground_truth_config", "path": str(validation_path)}
        partC_rev1_write_json(summary, validation_path)
        return summary
    gt_path = partC_rev1_resolve_path(str(flow_dir)) / str(recommended)
    link_flow_path = Path(str(selected_summary.get("artifacts", {}).get("physical_link_flow", "")))
    if not gt_path.exists() or not link_flow_path.exists():
        summary = {"status": "skipped_missing_files", "ground_truth_path": str(gt_path), "link_flow_path": str(link_flow_path), "path": str(validation_path)}
        partC_rev1_write_json(summary, validation_path)
        return summary
    id_candidates = [str(value) for value in ground_truth.get("id_col_candidates", ["ID"]) if value]
    flow_candidates = [str(value) for value in ground_truth.get("preferred_flow_col_candidates", []) if value]
    if ground_truth.get("default_total_flow_col"):
        flow_candidates.append(str(ground_truth["default_total_flow_col"]))
    sample = pd.read_csv(gt_path, nrows=0)
    id_col = next((column for column in id_candidates if column in sample.columns), None)
    flow_col = next((column for column in flow_candidates if column in sample.columns), None)
    if id_col is None or flow_col is None:
        summary = {
            "status": "skipped_missing_join_or_flow_column",
            "ground_truth_path": str(gt_path),
            "available_columns": sample.columns.tolist(),
            "id_candidates": id_candidates,
            "flow_candidates": flow_candidates,
            "path": str(validation_path),
        }
        partC_rev1_write_json(summary, validation_path)
        return summary
    gt = pd.read_csv(gt_path, usecols=[id_col, flow_col], dtype={id_col: "string"})
    gt[flow_col] = pd.to_numeric(gt[flow_col], errors="coerce").fillna(0.0)
    gt_agg = gt.groupby(id_col, as_index=False)[flow_col].sum()
    pred = pd.read_parquet(link_flow_path, columns=["partC_rev1_link_uid", "flow"])
    pred["partC_rev1_link_uid"] = pred["partC_rev1_link_uid"].astype("string")
    joined = pred.merge(gt_agg, left_on="partC_rev1_link_uid", right_on=id_col, how="left", indicator=True)
    observed = joined[flow_col].fillna(0.0).astype(float)
    predicted = joined["flow"].fillna(0.0).astype(float)
    diff = predicted - observed
    mae = float(np.mean(np.abs(diff))) if len(joined) else None
    rmse = float(np.sqrt(np.mean(np.square(diff)))) if len(joined) else None
    wmape = float(np.sum(np.abs(diff)) / max(np.sum(np.abs(observed)), 1e-12)) if len(joined) else None
    pearson = float(np.corrcoef(predicted, observed)[0, 1]) if len(joined) > 1 and np.std(predicted) > 0 and np.std(observed) > 0 else None
    geh = np.sqrt(2.0 * np.square(diff) / np.maximum(predicted + observed, 1e-12))
    metrics_path = out_dir / "partC_rev1_reference_flow_validation_rows.csv"
    partC_rev1_write_table(
        joined.loc[:, ["partC_rev1_link_uid", "flow", flow_col, "_merge"]].rename(columns={flow_col: "reference_flow"}),
        metrics_path,
        index=False,
    )
    summary = {
        "status": "completed_units_configured_compatible",
        "ground_truth_path": str(gt_path),
        "link_flow_path": str(link_flow_path),
        "rows_path": str(metrics_path),
        "join_key_predicted": "partC_rev1_link_uid",
        "join_key_reference": id_col,
        "reference_flow_col": flow_col,
        "reference_units": config.get("unit_conversion", {}).get("ground_truth_preferred_unit"),
        "predicted_units": config.get("unit_conversion", {}).get("solver_flow_unit"),
        "predicted_rows": int(len(pred)),
        "reference_rows": int(len(gt_agg)),
        "matched_predicted_rows": int(joined["_merge"].eq("both").sum()),
        "unmatched_predicted_rows": int(joined["_merge"].ne("both").sum()),
        "coverage_share": float(joined["_merge"].eq("both").mean()) if len(joined) else 0.0,
        "mae": mae,
        "rmse": rmse,
        "wmape": wmape,
        "pearson": pearson,
        "geh_mean": float(np.mean(geh)) if len(geh) else None,
        "geh_95th_percentile": float(np.quantile(geh, 0.95)) if len(geh) else None,
        "path": str(validation_path),
    }
    partC_rev1_write_json(summary, validation_path)
    return summary


def partC_rev1_run_physical_benchmark_stage(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    *,
    config: dict[str, Any],
    output_dir: str | Path,
    stage_config: PartCRev1PhysicalBenchmarkStageConfig,
    reporter: PartCRev1ProgressReporter | None = None,
) -> dict[str, Any]:
    out_dir = partC_rev1_resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    numba_runtime_seconds = 4.0 * 3600.0
    backend_pilot_origins = _partC_rev1_backend_pilot_max_origins(od, stage_config=stage_config)
    run_configs = [
        PartCRev1PhysicalBenchmarkRunConfig("fw", "numba_csr", stage_config.max_iterations, stage_config.gap_target, stage_config.origin_batch_size, runtime_guard_seconds=numba_runtime_seconds),
        PartCRev1PhysicalBenchmarkRunConfig("cfw", "numba_csr", stage_config.max_iterations, stage_config.gap_target, stage_config.origin_batch_size, runtime_guard_seconds=numba_runtime_seconds),
        PartCRev1PhysicalBenchmarkRunConfig("msa", "numba_csr", stage_config.max_iterations, stage_config.gap_target, stage_config.origin_batch_size, runtime_guard_seconds=numba_runtime_seconds),
        PartCRev1PhysicalBenchmarkRunConfig(
            "fw",
            "igraph",
            stage_config.backend_pilot_iterations,
            stage_config.gap_target,
            stage_config.origin_batch_size,
            max_origins=backend_pilot_origins,
            state_label="backend_time_stability_pilot",
            runtime_guard_seconds=max(float(stage_config.backend_pilot_min_seconds), 1.0),
        ),
        PartCRev1PhysicalBenchmarkRunConfig(
            "fw",
            "networkx_python",
            stage_config.backend_pilot_iterations,
            stage_config.gap_target,
            stage_config.origin_batch_size,
            max_origins=backend_pilot_origins,
            state_label="backend_time_stability_pilot",
            runtime_guard_seconds=max(float(stage_config.backend_pilot_min_seconds), 1.0),
        ),
    ]
    summaries: list[dict[str, Any]] = []
    for run_config in run_configs:
        summaries.append(
            partC_rev1_run_one_physical_benchmark(
                graph,
                od,
                output_dir=out_dir,
                stage_config=stage_config,
                run_config=run_config,
                reporter=reporter,
            )
        )
    registry_csv, registry_md, best = partC_rev1_write_benchmark_registry(summaries, out_dir)
    witness_summaries: list[dict[str, Any]] = []
    freeflow_path = out_dir / "partC_rev1_route_witnesses_freeflow.parquet"
    witness_summaries.append(
        partC_rev1_generate_route_witnesses(
            graph,
            od,
            graph.free_time,
            output_path=freeflow_path,
            witness_set="freeflow",
            max_origins=None if stage_config.witness_all_positive_origins else stage_config.witness_max_origins,
            diagnostic_path_count=stage_config.witness_diagnostic_path_count,
            reporter=reporter,
        )
    )
    if best is not None:
        directed_path = Path(str(best["artifacts"]["directed_arc_flow"]))
        best_flow = pd.read_parquet(directed_path, columns=["cost"])
        benchmark_cost = best_flow["cost"].to_numpy(dtype=np.float64)
    else:
        benchmark_cost = graph.free_time
    benchmark_witness_path = out_dir / "partC_rev1_route_witnesses_benchmark.parquet"
    witness_summaries.append(
        partC_rev1_generate_route_witnesses(
            graph,
            od,
            benchmark_cost,
            output_path=benchmark_witness_path,
            witness_set="benchmark",
            max_origins=None if stage_config.witness_all_positive_origins else stage_config.witness_max_origins,
            diagnostic_path_count=stage_config.witness_diagnostic_path_count,
            reporter=reporter,
        )
    )
    reference_validation = partC_rev1_validate_against_reference_flow(config, best, output_dir=out_dir)
    stage_summary = {
        "stage": PARTC_REV1_4C_STAGE,
        "created_utc": _partC_rev1_utc_now(),
        "config_path": stage_config.config_path,
        "git_commit": _partC_rev1_git_commit(),
        "graph_hash": graph.graph_hash,
        "od_hash": od.od_hash,
        "partition_hash": graph.partition_hash,
        "topology_hash": graph.topology_hash,
        "backbone_hash": graph.backbone_hash,
        "assignment_graph_hash": graph.assignment_graph_hash,
        "backend_pilot_rule": {
            "pilot_origins_used": int(backend_pilot_origins),
            "min_seconds": float(stage_config.backend_pilot_min_seconds),
            "min_batches": int(stage_config.backend_pilot_min_batches),
            "cv_tolerance": float(stage_config.backend_pilot_cv_tolerance),
        },
        "runtime_seconds": time.perf_counter() - started,
        "run_summaries": summaries,
        "registry_csv": str(registry_csv),
        "registry_md": str(registry_md),
        "best_accuracy_benchmark": best,
        "route_witnesses": witness_summaries,
        "reference_flow_validation": reference_validation,
    }
    summary_path = out_dir / "partC_rev1_4C_summary.json"
    partC_rev1_write_json(stage_summary, summary_path)
    stage_summary["summary_path"] = str(summary_path)
    return stage_summary
