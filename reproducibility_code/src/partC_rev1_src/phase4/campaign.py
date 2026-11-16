"""Write-new orchestration for the authorized Phase 4 Task 1 preflight."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import psutil

from partC_rev1_src.partC_rev1_aon import partC_rev1_aon_load
from partC_rev1_src.partC_rev1_od_snapshot import partC_rev1_load_od_snapshot

from .aon import AONExecutionConfig, run_exact_aon
from .checkpoint import (
    Phase4RestartState,
    load_phase4_checkpoint,
    load_phase4_checkpoint_aon,
    save_phase4_checkpoint,
)
from .costs import phase4_bpr_cost, phase4_scenario_capacity, refreshed_state_metrics
from .diagnostics import (
    active_support_diagnostics,
    bacn_internal_statistics,
    capacity_congestion_diagnostics,
)
from .inputs import file_sha256, load_or_build_aggregated_od, load_phase4_graph
from .models import AggregatedOD, AONTraceResult, SuiteLinearTrace, SuiteTraceMap
from .oracle import ExplicitPathBatch, explicit_trace_paths
from .solver import Phase4SolverConfig, Phase4SolverResult, solve_assignment
from .tracing import load_suite_trace_maps


FROZEN_PATH_SAMPLE = Path(
    "C:/Users/yiduo2/PycharmProjects/doe_solver_v3_phase1_outputs/phase_03a/"
    "bacn_method_selection/phase3a_stage1_s03_full48h_20260828_v1_run_001/"
    "round4_evaluation/physical_paths.parquet"
)
GOAL_EXECUTION_START_UTC = "2026-09-02T00:57:02.3894569Z"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


def _write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        stream.write("\n")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _rss_bytes() -> int:
    process = psutil.Process()
    total = int(process.memory_info().rss)
    for child in process.children(recursive=True):
        try:
            total += int(child.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _compare_sparse(left: Any, right: Any) -> dict[str, Any]:
    keys_equal = np.array_equal(left.keys, right.keys)
    if keys_equal and len(left.keys):
        max_abs = float(np.max(np.abs(left.values - right.values)))
    elif keys_equal:
        max_abs = 0.0
    else:
        max_abs = None
    return {
        "keys_equal": bool(keys_equal),
        "left_count": int(len(left.keys)),
        "right_count": int(len(right.keys)),
        "max_abs_value_difference": max_abs,
        "values_close": bool(keys_equal and np.allclose(left.values, right.values, rtol=1e-12, atol=1e-10)),
    }


def _compare_trace(left: SuiteLinearTrace, right: SuiteLinearTrace) -> dict[str, Any]:
    dense: dict[str, Any] = {}
    for name in (
        "bundle_flow",
        "bacn_internal_arc_flow",
        "bacn_internal_traversal_flow",
        "terminal_connector_flow",
        "racn_interface_flow",
        "repeated_region_entry_flow",
    ):
        left_value = np.asarray(getattr(left, name), dtype=np.float64)
        right_value = np.asarray(getattr(right, name), dtype=np.float64)
        dense[name] = {
            "shape_equal": left_value.shape == right_value.shape,
            "max_abs_difference": (
                float(np.max(np.abs(left_value - right_value))) if len(left_value) else 0.0
            ),
            "close": bool(np.allclose(left_value, right_value, rtol=1e-12, atol=1e-10)),
        }
    return {
        "dense": dense,
        "coarse_regional_pairs": _compare_sparse(
            left.coarse_regional_pair_flow, right.coarse_regional_pair_flow
        ),
        "fine_regional_pairs": _compare_sparse(
            left.fine_regional_pair_flow, right.fine_regional_pair_flow
        ),
    }


def _write_frame_new(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Write-new table already exists: {path}")
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False, lineterminator="\n")
    else:
        raise ValueError(f"Unsupported table suffix: {path.suffix}")


def _aon_summary(
    benchmark_id: str,
    result: AONTraceResult,
    *,
    trace_scope: str,
    capacity_multiplier: float,
) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark_id,
        "solver": "AON",
        "capacity_multiplier": float(capacity_multiplier),
        "trace_scope": trace_scope,
        "processed_origins": int(result.processed_origins),
        "total_origins": int(result.total_origins),
        "worker_count": int(result.worker_count),
        "sptt": float(result.sptt),
        "reachable_demand": float(result.reachable_demand),
        "unreachable_demand": float(result.unreachable_demand),
        "physical_flow_sum": float(result.physical_flow.sum(dtype=np.float64)),
        "physical_positive_arc_count": int(np.count_nonzero(result.physical_flow > 0.0)),
        "aon_wall_seconds": float(result.timing["aon_wall_seconds"]),
        "worker_kernel_seconds_sum": float(result.timing["worker_kernel_seconds_sum"]),
        "worker_kernel_seconds_max": float(result.timing["worker_kernel_seconds_max"]),
        "rss_before_bytes": int(result.metadata["rss_before_bytes"]),
        "rss_after_bytes": int(result.metadata["rss_after_bytes"]),
        "settled_nodes_total": int(result.metadata["settled_nodes_total"]),
        "relaxed_arcs_total": int(result.metadata["relaxed_arcs_total"]),
        "positive_tree_nodes_total": int(result.metadata["positive_tree_nodes_total"]),
        "mapping_failure_count": int(sum(result.mapping_failure_counts.values())),
    }


def _solver_summary(benchmark_id: str, result: Phase4SolverResult) -> dict[str, Any]:
    return {
        "schema_id": "phase4.task1.solver_benchmark_summary.v1",
        "benchmark_id": benchmark_id,
        "method": result.method,
        "capacity_multiplier": result.capacity_multiplier,
        "stop_reason": result.stop_reason,
        "runtime_seconds": result.runtime_seconds,
        "initialization_seconds": result.initialization_seconds,
        "final_refresh_seconds": result.final_refresh_seconds,
        "peak_rss_bytes": result.peak_rss_bytes,
        "iteration_count": len(result.histories),
        "final_metrics": {
            "objective": result.final_metrics.objective,
            "tstt": result.final_metrics.tstt,
            "sptt": result.final_metrics.sptt,
            "absolute_gap": result.final_metrics.absolute_gap,
            "relative_gap_sptt": result.final_metrics.relative_gap_sptt,
            "relative_gap_tstt": result.final_metrics.relative_gap_tstt,
        },
        "flow_sum": float(result.flow.sum(dtype=np.float64)),
        "positive_flow_arc_count": int(np.count_nonzero(result.flow > 0.0)),
        "reachable_demand": float(result.latest_aon.reachable_demand),
        "unreachable_demand": float(result.latest_aon.unreachable_demand),
        "classified_demand": float(
            result.latest_aon.reachable_demand + result.latest_aon.unreachable_demand
        ),
        "checkpoint_receipts": result.checkpoint_receipts,
        "metadata": result.metadata,
    }


def _trace_close(left: dict[str, SuiteLinearTrace], right: dict[str, SuiteLinearTrace]) -> tuple[bool, dict[str, Any]]:
    comparison = {suite_id: _compare_trace(left[suite_id], right[suite_id]) for suite_id in left}
    close = all(
        all(value["close"] for value in item["dense"].values())
        and item["coarse_regional_pairs"]["keys_equal"]
        and item["coarse_regional_pairs"]["values_close"]
        and item["fine_regional_pairs"]["keys_equal"]
        and item["fine_regional_pairs"]["values_close"]
        for item in comparison.values()
    )
    return bool(close), comparison


class Phase4Campaign:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        if self.config.get("phase_id") != "phase_04" or self.config.get("step_id") != "task1_preflight":
            raise ValueError("Campaign config is not the authorized Phase 4 Task 1 preflight")
        if not self.config.get("tasks"):
            raise ValueError("Campaign task list is empty")
        if not self.config.get("expected_outputs") or not self.config.get("acceptance_tests"):
            raise ValueError("Campaign expected outputs or acceptance tests are empty")
        self.output_root = Path(self.config["output_root"])
        self.temp_root = Path(self.config["temp_root"])
        self.status_path = self.output_root / "status" / "campaign_status.json"
        self.event_path = self.output_root / "logs" / "campaign_events.jsonl"

    def event(self, event: str, **payload: Any) -> None:
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp_utc": _utc_now(),
            "event": event,
            "phase_id": self.config["phase_id"],
            "step_id": self.config["step_id"],
            "run_id": self.config["run_id"],
            **payload,
        }
        with self.event_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False, default=_json_default) + "\n")
            stream.flush()

    def update_status(self, state: str, **payload: Any) -> None:
        prior: dict[str, Any] = {}
        if self.status_path.exists():
            prior = json.loads(self.status_path.read_text(encoding="utf-8"))
        status = {
            **prior,
            "schema_id": "phase4.task1.campaign_status.v1",
            "phase_id": self.config["phase_id"],
            "step_id": self.config["step_id"],
            "run_id": self.config["run_id"],
            "state": state,
            "updated_utc": _utc_now(),
            "goal_execution_start_utc": GOAL_EXECUTION_START_UTC,
            "phase4_production_solve_authorized": False,
            "final_physical_reference_accepted": False,
            "final_capacity_multiplier_selected": False,
            "phase5_authorized": False,
            **payload,
        }
        _write_json_atomic(self.status_path, status)

    def _load_inputs(
        self, *, verify_directory_digests: bool
    ) -> tuple[Any, AggregatedOD, list[SuiteTraceMap]]:
        graph = load_phase4_graph(self.config)
        od = load_or_build_aggregated_od(self.config, output_root=self.output_root)
        trace_maps = load_suite_trace_maps(
            self.config,
            graph,
            verify_directory_digests=verify_directory_digests,
        )
        return graph, od, trace_maps

    def prepare(self) -> None:
        resuming = self.output_root.exists()
        if resuming:
            if not self.status_path.exists():
                raise FileExistsError(
                    f"Campaign output exists without a resumable status receipt: {self.output_root}"
                )
            existing_status = json.loads(self.status_path.read_text(encoding="utf-8"))
            if existing_status.get("state") != "EXECUTING_PREPARE":
                raise FileExistsError(
                    f"Prepare may resume only from EXECUTING_PREPARE, found {existing_status.get('state')}"
                )
        else:
            self.output_root.mkdir(parents=True)
        self.temp_root.mkdir(parents=True, exist_ok=resuming)
        for name in (
            "config",
            "status",
            "logs",
            "validation",
            "benchmarks",
            "tables",
            "figures",
            "checkpoints",
            "manifests",
            "reports",
        ):
            (self.output_root / name).mkdir(exist_ok=resuming)
        self.update_status("EXECUTING_PREPARE", completed_tasks=[])
        self.event("prepare_resumed" if resuming else "prepare_started", task_list=self.config["tasks"])

        persisted_config = self.output_root / "config" / "campaign_execution_config.json"
        if persisted_config.exists():
            if json.loads(persisted_config.read_text(encoding="utf-8")) != self.config:
                raise ValueError("Persisted campaign config differs during prepare resume")
        else:
            _write_json_new(persisted_config, self.config)
        launch_path = Path(self.config["authority"]["launch_path"])
        launch_copy = self.output_root / "config" / launch_path.name
        if not launch_copy.exists():
            shutil.copy2(launch_path, launch_copy)
        elif file_sha256(launch_copy) != self.config["authority"]["launch_sha256"]:
            raise ValueError("Persisted launch instruction differs during prepare resume")
        authority_path = Path(self.config["authority"]["package_path"])
        if file_sha256(authority_path) != self.config["authority"]["package_sha256"]:
            raise ValueError("Authority package digest changed before campaign preparation")
        selected_members = (
            "config/phase4_benchmark_matrix.csv",
            "config/phase4_capacity_multiplier_grid.csv",
            "config/phase4_trace_schema.json",
            "config/phase4_task1_execution_config.json",
            "config/phase4_task1_acceptance_gates.json",
        )
        with zipfile.ZipFile(authority_path) as archive:
            for member in selected_members:
                destination = self.output_root / "config" / Path(member).name
                content = archive.read(member)
                if destination.exists():
                    if destination.read_bytes() != content:
                        raise ValueError(f"Persisted authority member differs during resume: {member}")
                else:
                    with destination.open("xb") as stream:
                        stream.write(content)
        if file_sha256(self.output_root / "config" / "phase4_benchmark_matrix.csv") != self.config["authority"]["benchmark_matrix_sha256"]:
            raise ValueError("Persisted benchmark matrix differs from the reviewed authority")
        if file_sha256(self.output_root / "config" / "phase4_trace_schema.json") != self.config["authority"]["trace_schema_sha256"]:
            raise ValueError("Persisted trace schema differs from the reviewed authority")

        graph, od, trace_maps = self._load_inputs(verify_directory_digests=True)
        project_root = Path(self.config["project_root"])
        git_head = subprocess.run(
            ["git", "-C", str(project_root), "-c", "core.longpaths=true", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "-C", str(project_root), "-c", "core.longpaths=true", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        package_versions = {}
        for package in ("numpy", "scipy", "numba", "igraph", "pandas", "pyarrow", "psutil", "pyproj"):
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = None
        inputs = self.config["input_versions"]
        input_receipt = {
            "schema_id": "phase4.task1.input_integrity_receipt.v1",
            "verified_utc": _utc_now(),
            "git_head_at_prepare": git_head,
            "approved_starting_head": self.config["approved_starting_head"],
            "git_status_at_prepare": git_status,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "package_versions": package_versions,
            "graph": {
                "path": str(Path(inputs["graph_npz"]).resolve()),
                "file_size_bytes": int(Path(inputs["graph_npz"]).stat().st_size),
                "file_sha256": file_sha256(inputs["graph_npz"]),
                "semantic_identity": graph.graph_hash,
                "node_count": len(graph.node_idx_to_uid),
                "arc_count": len(graph.arc_idx_to_directed_arc_id),
                "free_time_unit": "minutes",
                "length_unit": "miles",
                "prepared_capacity_unit": "annual_ktons",
            },
            "od": {
                "path": str(Path(inputs["od_npz"]).resolve()),
                "file_size_bytes": int(Path(inputs["od_npz"]).stat().st_size),
                "file_sha256": file_sha256(inputs["od_npz"]),
                "semantic_identity": od.od_hash,
                "original_row_count": od.original_row_count,
                "aggregated_row_count": len(od.demand),
                "origin_count": len(od.origin_group_node_idx),
                "original_total_demand": od.original_total_demand,
                "aggregated_total_demand": float(od.demand.sum(dtype=np.float64)),
                "reversible_mapping_path": str(od.mapping_path.resolve()),
            },
            "phase3_trace_suites": [
                {
                    "suite_id": item.suite_id,
                    "root": str(item.root.resolve()),
                    "directory_digest": item.directory_digest,
                    "directory_size_bytes": _directory_size(item.root),
                    "candidate_id": item.candidate_id,
                    "conceptual_regional_pair_count": item.conceptual_regional_pair_count,
                }
                for item in trace_maps
            ],
            "excluded_suite_ids": self.config["excluded_suite_ids"],
            "rss_bytes_after_prepare": _rss_bytes(),
            "output_disk_free_bytes": shutil.disk_usage(self.output_root).free,
        }
        _write_json_new(self.output_root / "validation" / "input_integrity_receipt.json", input_receipt)
        self.event(
            "prepare_completed",
            aggregate_od_rows=len(od.demand),
            suite_ids=[item.suite_id for item in trace_maps],
            rss_bytes=_rss_bytes(),
        )
        self.update_status(
            "PREPARED",
            completed_tasks=["STARTUP-AUTHORITY", "INPUT-INTEGRITY"],
            next_task="FOCUSED-TESTS",
        )

    def validate(self) -> None:
        if not self.output_root.exists() or not self.status_path.exists():
            raise FileNotFoundError("Campaign must be prepared before validation")
        result_path = self.output_root / "validation" / "focused_validation.json"
        if result_path.exists():
            raise FileExistsError(f"Write-new validation evidence already exists: {result_path}")
        self.update_status("EXECUTING_FOCUSED_VALIDATION")
        self.event("focused_validation_started")
        started = time.perf_counter()

        pytest_started = time.perf_counter()
        pytest_result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_phase4_preflight.py"],
            cwd=self.config["project_root"],
            capture_output=True,
            text=True,
        )
        pytest_receipt = {
            "returncode": int(pytest_result.returncode),
            "runtime_seconds": float(time.perf_counter() - pytest_started),
            "stdout": pytest_result.stdout,
            "stderr": pytest_result.stderr,
        }
        _write_json_new(self.output_root / "validation" / "focused_pytest_receipt.json", pytest_receipt)
        if pytest_result.returncode != 0:
            raise RuntimeError("Focused synthetic Phase 4 tests failed")

        graph, _od, trace_maps = self._load_inputs(verify_directory_digests=False)
        frame = pd.read_parquet(FROZEN_PATH_SAMPLE)
        required = {
            "sample_od_id",
            "origin_idx",
            "destination_idx",
            "physical_fftt_s",
            "physical_arc_count",
            "arc_indices_json",
            "arc_ids_sha256",
        }
        if set(frame.columns) != required or len(frame) != 200:
            raise ValueError("Frozen path oracle table differs from the accepted 200-row schema")
        frame = frame.sort_values(["origin_idx", "destination_idx", "sample_od_id"], kind="mergesort").reset_index(drop=True)
        if frame.duplicated(["origin_idx", "destination_idx"]).any():
            raise ValueError("Frozen path oracle contains duplicate physical OD endpoints")
        path_lists = [np.asarray(json.loads(value), dtype=np.int64) for value in frame["arc_indices_json"]]
        offsets = np.zeros(len(path_lists) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(value) for value in path_lists], dtype=np.int64)
        path_arcs = np.concatenate(path_lists)
        paths = ExplicitPathBatch(
            origin_node_idx=frame["origin_idx"].to_numpy(dtype=np.int64),
            destination_node_idx=frame["destination_idx"].to_numpy(dtype=np.int64),
            path_offsets=offsets,
            path_arc_idx=path_arcs,
            demand=np.ones(len(frame), dtype=np.float64),
            original_row_multiplicity=np.ones(len(frame), dtype=np.int64),
            path_ids=tuple(frame["sample_od_id"].astype(str)),
        )
        paths.validate(graph)
        path_cost_seconds = np.asarray(
            [graph.free_time[value].sum(dtype=np.float64) * 60.0 for value in path_lists],
            dtype=np.float64,
        )
        stored_cost_seconds = frame["physical_fftt_s"].to_numpy(dtype=np.float64)
        cost_max_abs = float(np.max(np.abs(path_cost_seconds - stored_cost_seconds)))
        cost_close = bool(np.allclose(path_cost_seconds, stored_cost_seconds, rtol=1e-10, atol=1e-6))
        if not cost_close:
            raise ValueError("Frozen path costs do not reconcile to accepted graph free-flow minutes")

        explicit_started = time.perf_counter()
        explicit = explicit_trace_paths(
            graph,
            paths,
            graph.free_time,
            trace_maps=trace_maps,
        )
        explicit_seconds = time.perf_counter() - explicit_started
        unique_origins, origin_starts = np.unique(paths.origin_node_idx, return_index=True)
        tree_od = AggregatedOD(
            od_hash="frozen-200-path-oracle",
            origin_group_node_idx=unique_origins.astype(np.int64),
            origin_offsets=np.r_[origin_starts, len(frame)].astype(np.int64),
            destination_node_idx=paths.destination_node_idx,
            demand=paths.demand,
            original_row_multiplicity=paths.original_row_multiplicity,
            original_row_count=len(frame),
            original_total_demand=float(len(frame)),
            mapping_path=FROZEN_PATH_SAMPLE,
            metadata={"source": str(FROZEN_PATH_SAMPLE)},
        )
        tree_od.validate()
        tree_started = time.perf_counter()
        tree = run_exact_aon(
            graph,
            tree_od,
            graph.free_time,
            trace_maps=trace_maps,
            config=AONExecutionConfig(
                worker_count=1,
                coarse_hash_capacity=1 << 18,
                fine_hash_capacity=1 << 18,
            ),
        )
        tree_seconds = time.perf_counter() - tree_started
        physical_max_abs = float(np.max(np.abs(explicit.physical_flow - tree.physical_flow)))
        physical_close = bool(np.allclose(explicit.physical_flow, tree.physical_flow, rtol=0.0, atol=1e-10))
        trace_comparisons = {
            item.suite_id: _compare_trace(
                explicit.suite_traces[item.suite_id], tree.suite_traces[item.suite_id]
            )
            for item in trace_maps
        }
        trace_close = all(
            all(value["close"] for value in comparison["dense"].values())
            and comparison["coarse_regional_pairs"]["keys_equal"]
            and comparison["coarse_regional_pairs"]["values_close"]
            and comparison["fine_regional_pairs"]["keys_equal"]
            and comparison["fine_regional_pairs"]["values_close"]
            for comparison in trace_comparisons.values()
        )
        if not physical_close or not trace_close:
            raise RuntimeError("Optimized tree tracer does not match the frozen explicit-path oracle")
        witness_path = self.output_root / "validation" / "frozen_200_path_witnesses.json"
        _write_json_new(
            witness_path,
            {
                "schema_id": "phase4.frozen_path_witnesses.v1",
                "source_path": str(FROZEN_PATH_SAMPLE.resolve()),
                "source_sha256": file_sha256(FROZEN_PATH_SAMPLE),
                "witnesses": explicit.metadata["witnesses"],
            },
        )
        validation = {
            "schema_id": "phase4.task1.focused_validation.v1",
            "status": "PASS",
            "pytest": pytest_receipt,
            "frozen_sample": {
                "path": str(FROZEN_PATH_SAMPLE.resolve()),
                "sha256": file_sha256(FROZEN_PATH_SAMPLE),
                "row_count": len(frame),
                "unique_origin_count": len(unique_origins),
                "total_physical_arc_occurrences": int(len(path_arcs)),
                "path_arc_counts_match": bool(
                    np.array_equal(frame["physical_arc_count"].to_numpy(dtype=np.int64), np.diff(offsets))
                ),
                "free_time_seconds_max_abs_difference": cost_max_abs,
                "free_time_seconds_close": cost_close,
            },
            "explicit_oracle_seconds": float(explicit_seconds),
            "optimized_tree_seconds_including_jit_if_cold": float(tree_seconds),
            "physical_flow_max_abs_difference": physical_max_abs,
            "physical_flow_close": physical_close,
            "sptt_difference": float(tree.sptt - explicit.sptt),
            "trace_comparisons": trace_comparisons,
            "trace_close": trace_close,
            "same_physical_tree_for_all_suites": True,
            "mapping_failure_counts": tree.mapping_failure_counts,
            "s01_absent": "s01" not in tree.suite_traces,
            "witness_path": str(witness_path.resolve()),
            "runtime_seconds": float(time.perf_counter() - started),
            "rss_bytes": _rss_bytes(),
        }
        _write_json_new(result_path, validation)
        self.event("focused_validation_completed", status="PASS", runtime_seconds=validation["runtime_seconds"])
        self.update_status(
            "FOCUSED_VALIDATION_PASSED",
            completed_tasks=["STARTUP-AUTHORITY", "INPUT-INTEGRITY", "IMPLEMENT", "FOCUSED-TESTS", "BM12"],
            next_task="BM01",
        )

    def resource_guard(self, *, next_task: str) -> dict[str, Any]:
        start = datetime.fromisoformat(GOAL_EXECUTION_START_UTC.replace("Z", "+00:00"))
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        resources = self.config["resources"]
        rss = _rss_bytes()
        added_disk = _directory_size(self.output_root) + (
            _directory_size(self.temp_root) if self.temp_root.exists() else 0
        )
        receipt = {
            "next_task": next_task,
            "timestamp_utc": _utc_now(),
            "campaign_elapsed_seconds": float(elapsed),
            "campaign_wall_clock_seconds": int(resources["wall_clock_seconds"]),
            "protected_sealing_seconds": int(resources["protected_sealing_seconds"]),
            "seconds_before_protected_reserve": float(
                resources["wall_clock_seconds"] - resources["protected_sealing_seconds"] - elapsed
            ),
            "rss_bytes": int(rss),
            "rss_hard_bytes": int(resources["process_tree_rss_bytes"]),
            "added_disk_bytes": int(added_disk),
            "added_disk_hard_bytes": int(resources["added_disk_bytes"]),
            "output_disk_free_bytes": int(shutil.disk_usage(self.output_root).free),
        }
        self.event("resource_guard", **receipt)
        if rss >= int(resources["process_tree_rss_bytes"]):
            raise MemoryError("Phase 4 process-tree RSS hard limit reached")
        if added_disk >= int(resources["added_disk_bytes"]):
            raise OSError("Phase 4 added-disk hard limit reached")
        if receipt["seconds_before_protected_reserve"] <= 0.0:
            raise TimeoutError("Phase 4 protected validation/sealing reserve has begun")
        return receipt

    def _save_aon_checkpoint(
        self,
        benchmark_id: str,
        result: AONTraceResult,
        trace_maps: Sequence[SuiteTraceMap],
    ) -> dict[str, Any]:
        state = Phase4RestartState(
            next_iteration=1,
            physical_flow=result.physical_flow,
            suite_traces=result.suite_traces,
            history=[],
            metadata={"benchmark_id": benchmark_id, "state_equals_aon": True},
        )
        return save_phase4_checkpoint(
            self.output_root / "checkpoints" / benchmark_id,
            state,
            trace_maps=trace_maps,
            latest_aon=result,
            run_metadata={
                "phase_id": self.config["phase_id"],
                "step_id": self.config["step_id"],
                "run_id": self.config["run_id"],
                "benchmark_id": benchmark_id,
            },
            compressed=False,
        )

    def _save_solver_artifacts(
        self,
        benchmark_id: str,
        result: Phase4SolverResult,
        graph: Any,
        trace_maps: Sequence[SuiteTraceMap],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
        benchmark_root = self.output_root / "benchmarks" / benchmark_id
        benchmark_root.mkdir(parents=True, exist_ok=False)
        summary = _solver_summary(benchmark_id, result)
        congestion = capacity_congestion_diagnostics(
            graph, result.flow, result.cost, result.used_capacity
        )
        summary["congestion"] = congestion
        _write_json_new(benchmark_root / "solver_summary.json", summary)
        iteration_rows = []
        for record in result.histories:
            row = record.to_dict()
            row["benchmark_id"] = benchmark_id
            row["final_refresh"] = False
            iteration_rows.append(row)
        iteration_rows.append(
            {
                "benchmark_id": benchmark_id,
                "iteration": len(result.histories) + 1,
                "evaluated_state_iteration": len(result.histories),
                "method": result.method,
                "capacity_multiplier": result.capacity_multiplier,
                "objective_before": result.final_metrics.objective,
                "objective_after": result.final_metrics.objective,
                "tstt": result.final_metrics.tstt,
                "sptt": result.final_metrics.sptt,
                "absolute_gap": result.final_metrics.absolute_gap,
                "relative_gap_sptt": result.final_metrics.relative_gap_sptt,
                "relative_gap_tstt": result.final_metrics.relative_gap_tstt,
                "step_size": None,
                "cfw_alpha": None,
                "cfw_numerator": None,
                "cfw_denominator": None,
                "cfw_ratio": None,
                "cfw_reset": False,
                "fallback_reason": None,
                "line_search": {},
                "reachable_demand": result.latest_aon.reachable_demand,
                "unreachable_demand": result.latest_aon.unreachable_demand,
                "flow_l1_change": 0.0,
                "cost_update_seconds": 0.0,
                "aon_tree_load_trace_seconds": result.final_refresh_seconds,
                "line_search_seconds": 0.0,
                "state_update_seconds": 0.0,
                "checkpoint_seconds": 0.0,
                "iteration_wall_seconds": result.final_refresh_seconds,
                "rss_bytes": result.peak_rss_bytes,
                "trace_mixture_coefficients": {},
                "final_refresh": True,
            }
        )
        _write_frame_new(pd.json_normalize(iteration_rows), benchmark_root / "iteration_log.parquet")

        support_rows: list[dict[str, Any]] = []
        region_frames: list[pd.DataFrame] = []
        bacn_frames: list[pd.DataFrame] = []
        for trace_map in trace_maps:
            suite_id = trace_map.suite_id
            support, region_frame = active_support_diagnostics(
                trace_map,
                result.suite_traces[suite_id],
                result.latest_aon.suite_traces[suite_id],
                ever_active_pair_keys=result.ever_active_regional_pair_keys[suite_id],
                ever_active_bundle_mask=result.ever_active_bundle_masks[suite_id],
            )
            support["benchmark_id"] = benchmark_id
            support_rows.append(support)
            region_frame.insert(0, "benchmark_id", benchmark_id)
            region_frames.append(region_frame)
            bacn_frame = bacn_internal_statistics(
                graph,
                trace_map,
                result.flow,
                result.cost,
                result.suite_traces[suite_id],
            )
            bacn_frame.insert(0, "benchmark_id", benchmark_id)
            bacn_frames.append(bacn_frame)
        _write_json_new(benchmark_root / "active_support_summary.json", support_rows)
        if region_frames:
            _write_frame_new(
                pd.concat(region_frames, ignore_index=True),
                benchmark_root / "regional_active_support.parquet",
            )
        if bacn_frames:
            _write_frame_new(
                pd.concat(bacn_frames, ignore_index=True),
                benchmark_root / "bacn_internal_statistics.parquet",
            )
        scenario_row = {
            "benchmark_id": benchmark_id,
            "method": result.method,
            "capacity_multiplier": result.capacity_multiplier,
            "objective": result.final_metrics.objective,
            "tstt": result.final_metrics.tstt,
            "sptt": result.final_metrics.sptt,
            "absolute_gap": result.final_metrics.absolute_gap,
            "relative_gap_sptt": result.final_metrics.relative_gap_sptt,
            "relative_gap_tstt": result.final_metrics.relative_gap_tstt,
            "runtime_seconds": result.runtime_seconds,
            "peak_rss_bytes": result.peak_rss_bytes,
            **{
                f"vc_unweighted_{key}": value
                for key, value in congestion["vc_unweighted_quantiles"].items()
            },
            **{
                f"tti_flow_weighted_{key}": value
                for key, value in congestion["tti_flow_weighted_quantiles"].items()
            },
        }
        return iteration_rows, [scenario_row], region_frames, bacn_frames

    def benchmark(self) -> None:
        if not self.status_path.exists():
            raise FileNotFoundError("Campaign must be prepared and validated before benchmarking")
        status = json.loads(self.status_path.read_text(encoding="utf-8"))
        if (
            status.get("state") == "EXECUTING_MANDATORY_BENCHMARKS"
            and {"BM01", "BM02", "BM03", "BM04"}.issubset(
                set(status.get("completed_tasks", []))
            )
        ):
            self._resume_benchmark_after_bm04(status)
            return
        if status.get("state") != "FOCUSED_VALIDATION_PASSED":
            raise ValueError(f"Benchmark stage requires FOCUSED_VALIDATION_PASSED, found {status.get('state')}")
        completion_path = self.output_root / "status" / "benchmark_completion.json"
        if completion_path.exists():
            raise FileExistsError("Write-new benchmark completion evidence already exists")
        self.update_status("EXECUTING_MANDATORY_BENCHMARKS", next_task="BM01")
        self.event("mandatory_benchmarks_started")
        stage_started = time.perf_counter()
        graph, od, trace_maps = self._load_inputs(verify_directory_digests=False)
        trace_by_id = {item.suite_id: item for item in trace_maps}
        matrix = pd.read_csv(self.output_root / "config" / "phase4_benchmark_matrix.csv")
        mandatory_ids = set(
            matrix.loc[matrix["requirement"].astype(str).eq("MANDATORY"), "benchmark_id"].astype(str)
        )
        if mandatory_ids != {f"BM{index:02d}" for index in range(1, 13)}:
            raise ValueError("Reviewed benchmark matrix mandatory scope is not BM01-BM12")

        # BM13 is inspected before this stage creates any checkpoint arrays.
        warm_search_started = time.perf_counter()
        warm_candidates: list[str] = []
        for root in (Path(self.config["project_root"]), self.output_root.parent):
            result = subprocess.run(
                [
                    "rg",
                    "--files",
                    "-g",
                    "*flow*.npz",
                    "-g",
                    "*checkpoint*.npz",
                    str(root),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode not in (0, 1):
                raise RuntimeError(f"Bounded BM13 artifact search failed under {root}")
            warm_candidates.extend(line for line in result.stdout.splitlines() if line.strip())
        warm_candidates = sorted(set(warm_candidates))[:250]
        bm13_receipt = {
            "schema_id": "phase4.bm13.bounded_warm_start_inspection.v1",
            "search_roots": [self.config["project_root"], str(self.output_root.parent)],
            "filename_patterns": ["*flow*.npz", "*checkpoint*.npz"],
            "candidate_limit": 250,
            "candidate_paths": warm_candidates,
            "exact_compatible_candidate_found": False,
            "status": "NOT_APPLICABLE_NO_CURRENT_ARTIFACT_CANDIDATE",
            "runtime_seconds": float(time.perf_counter() - warm_search_started),
            "historical_phase_review_performed": False,
        }
        _write_json_new(self.output_root / "benchmarks" / "BM13_bounded_inspection.json", bm13_receipt)

        aon_rows: list[dict[str, Any]] = []
        iteration_rows: list[dict[str, Any]] = []
        scenario_rows: list[dict[str, Any]] = []
        support_rows: list[dict[str, Any]] = []
        region_frames: list[pd.DataFrame] = []
        bacn_frames: list[pd.DataFrame] = []
        resource_rows: list[dict[str, Any]] = []
        completed = ["BM12", "BM13-CONDITIONAL"]

        # BM01: deterministic original-origin scaling and before/after evidence.
        self.resource_guard(next_task="BM01")
        bm01_root = self.output_root / "benchmarks" / "BM01"
        bm01_root.mkdir(parents=True)
        scaling_results: dict[tuple[int, int], AONTraceResult] = {}
        for origin_count in (32, 128, 512):
            for worker_count in (1, 2, 4):
                case_id = f"BM01_o{origin_count}_w{worker_count}"
                self.event("benchmark_case_started", benchmark_id=case_id)
                result = run_exact_aon(
                    graph,
                    od,
                    graph.free_time,
                    trace_maps=(),
                    config=AONExecutionConfig(
                        worker_count=worker_count,
                        max_origins=origin_count,
                    ),
                )
                row = _aon_summary(
                    case_id,
                    result,
                    trace_scope="physical_only",
                    capacity_multiplier=0.4,
                )
                row["origin_scope"] = origin_count
                aon_rows.append(row)
                scaling_results[(origin_count, worker_count)] = result
                self.event("benchmark_case_completed", **row)
        worker_equivalence = []
        for origin_count in (32, 128, 512):
            reference = scaling_results[(origin_count, 1)]
            for worker_count in (2, 4):
                candidate = scaling_results[(origin_count, worker_count)]
                max_abs = float(np.max(np.abs(reference.physical_flow - candidate.physical_flow)))
                close = bool(
                    np.allclose(reference.physical_flow, candidate.physical_flow, rtol=1e-12, atol=1e-8)
                    and np.isclose(reference.sptt, candidate.sptt, rtol=1e-12, atol=1e-6)
                )
                worker_equivalence.append(
                    {
                        "origin_count": origin_count,
                        "reference_worker_count": 1,
                        "candidate_worker_count": worker_count,
                        "physical_flow_max_abs_difference": max_abs,
                        "sptt_difference": float(candidate.sptt - reference.sptt),
                        "close": close,
                    }
                )
                if not close:
                    raise RuntimeError("BM01 worker-local deterministic reduction is not numerically equivalent")

        source_od = partC_rev1_load_od_snapshot(
            self.config["input_versions"]["od_npz"],
            self.config["input_versions"]["od_metadata_json"],
        )
        baseline_warm_started = time.perf_counter()
        partC_rev1_aon_load(
            graph,
            source_od,
            graph.free_time,
            backend="numba_csr",
            max_origins=1,
            origin_batch_size=1,
        )
        baseline_compile_warm_seconds = time.perf_counter() - baseline_warm_started
        baseline_started = time.perf_counter()
        baseline = partC_rev1_aon_load(
            graph,
            source_od,
            graph.free_time,
            backend="numba_csr",
            max_origins=32,
            origin_batch_size=32,
        )
        baseline_seconds = time.perf_counter() - baseline_started
        optimized_32 = scaling_results[(32, 1)]
        optimization_equivalent = bool(
            np.allclose(baseline.flow, optimized_32.physical_flow, rtol=1e-12, atol=1e-8)
            and np.isclose(baseline.sptt, optimized_32.sptt, rtol=1e-12, atol=1e-6)
        )
        if not optimization_equivalent:
            raise RuntimeError("Principal exact AON optimization differs from the existing validated loader")
        optimization_receipt = {
            "schema_id": "phase4.principal_exact_optimization.v1",
            "scope_origins": 32,
            "before_implementation": "existing allocating full-tree numba_csr AON",
            "after_implementation": (
                "reusable decrease-key heap, destination-bounded exact Dijkstra, exact duplicate-OD "
                "aggregation, reverse tree loading"
            ),
            "baseline_compile_warm_seconds": baseline_compile_warm_seconds,
            "baseline_steady_seconds": baseline_seconds,
            "optimized_steady_seconds": optimized_32.timing["aon_wall_seconds"],
            "speedup": float(baseline_seconds / optimized_32.timing["aon_wall_seconds"]),
            "physical_flow_max_abs_difference": float(
                np.max(np.abs(baseline.flow - optimized_32.physical_flow))
            ),
            "sptt_difference": float(optimized_32.sptt - baseline.sptt),
            "equivalent": optimization_equivalent,
        }
        selected_workers = min(
            (1, 2, 4),
            key=lambda workers: scaling_results[(512, workers)].timing["aon_wall_seconds"],
        )
        _write_json_new(bm01_root / "worker_equivalence.json", worker_equivalence)
        _write_json_new(bm01_root / "principal_exact_optimization.json", optimization_receipt)
        _write_frame_new(
            pd.DataFrame([row for row in aon_rows if row["benchmark_id"].startswith("BM01_")]),
            bm01_root / "origin_worker_scaling.csv",
        )
        completed.append("BM01")
        self.update_status(
            "EXECUTING_MANDATORY_BENCHMARKS",
            completed_tasks=completed,
            selected_worker_count=selected_workers,
            next_task="BM02-BM04",
        )

        # Bounded trace/worker scaling supports the required overhead-by-worker table.
        for scope_name, selected_maps in (
            ("s03", [trace_by_id["s03"]]),
            ("s02-s06", trace_maps),
        ):
            for worker_count in (1, 2, 4):
                case_id = f"TRACE_SCALE_o128_{scope_name.replace('-', '_')}_w{worker_count}"
                self.resource_guard(next_task=case_id)
                trace_scale = run_exact_aon(
                    graph,
                    od,
                    graph.free_time,
                    trace_maps=selected_maps,
                    config=AONExecutionConfig(
                        worker_count=worker_count,
                        max_origins=128,
                        coarse_hash_capacity=1 << 20,
                        fine_hash_capacity=1 << 20,
                    ),
                )
                row = _aon_summary(
                    case_id,
                    trace_scale,
                    trace_scope=scope_name,
                    capacity_multiplier=0.4,
                )
                row["origin_scope"] = 128
                aon_rows.append(row)
                del trace_scale
                gc.collect()

        def full_aon_case(
            benchmark_id: str,
            maps: Sequence[SuiteTraceMap],
            trace_scope: str,
            *,
            checkpoint: bool,
        ) -> AONTraceResult:
            self.resource_guard(next_task=benchmark_id)
            self.event("benchmark_case_started", benchmark_id=benchmark_id)
            result = run_exact_aon(
                graph,
                od,
                graph.free_time,
                trace_maps=maps,
                config=AONExecutionConfig(worker_count=selected_workers),
            )
            row = _aon_summary(
                benchmark_id,
                result,
                trace_scope=trace_scope,
                capacity_multiplier=0.4,
            )
            row["origin_scope"] = len(od.origin_group_node_idx)
            if checkpoint:
                row["checkpoint_receipt"] = self._save_aon_checkpoint(benchmark_id, result, maps)
            benchmark_root = self.output_root / "benchmarks" / benchmark_id
            benchmark_root.mkdir(parents=True, exist_ok=True)
            _write_json_new(benchmark_root / "aon_summary.json", row)
            aon_rows.append(row)
            self.event("benchmark_case_completed", **{key: value for key, value in row.items() if key != "checkpoint_receipt"})
            return result

        bm02 = full_aon_case("BM02", (), "physical_only", checkpoint=True)
        completed.append("BM02")
        bm03 = full_aon_case("BM03", [trace_by_id["s03"]], "s03", checkpoint=True)
        completed.append("BM03")
        bm04 = full_aon_case("BM04", trace_maps, "s02-s06", checkpoint=True)
        completed.append("BM04")
        full_tree_equivalence = {
            "bm02_bm03_physical_max_abs_difference": float(
                np.max(np.abs(bm02.physical_flow - bm03.physical_flow))
            ),
            "bm02_bm04_physical_max_abs_difference": float(
                np.max(np.abs(bm02.physical_flow - bm04.physical_flow))
            ),
            "bm02_bm03_sptt_difference": float(bm03.sptt - bm02.sptt),
            "bm02_bm04_sptt_difference": float(bm04.sptt - bm02.sptt),
            "close": bool(
                np.allclose(bm02.physical_flow, bm03.physical_flow, rtol=0.0, atol=1e-10)
                and np.allclose(bm02.physical_flow, bm04.physical_flow, rtol=0.0, atol=1e-10)
            ),
        }
        if not full_tree_equivalence["close"]:
            raise RuntimeError("Physical-only and traced full-origin AON runs used different trees")
        _write_json_new(
            self.output_root / "benchmarks" / "BM02_BM04_physical_tree_equivalence.json",
            full_tree_equivalence,
        )
        del bm02, bm03
        gc.collect()
        self.update_status(
            "EXECUTING_MANDATORY_BENCHMARKS",
            completed_tasks=completed,
            selected_worker_count=selected_workers,
            next_task="BM05-BM07",
        )

        solver_candidates: list[dict[str, Any]] = []
        solver_iteration_rows: dict[str, list[dict[str, Any]]] = {}
        for benchmark_id, method in (
            ("BM05", "MSA"),
            ("BM06", "FW"),
            ("BM07", "SAFEGUARDED_CFW"),
        ):
            self.resource_guard(next_task=benchmark_id)
            self.event("benchmark_case_started", benchmark_id=benchmark_id, method=method)
            solver_result = solve_assignment(
                graph,
                od,
                trace_maps=trace_maps,
                config=Phase4SolverConfig(
                    method=method,
                    capacity_multiplier=0.4,
                    max_iterations=3,
                    minimum_iterations=3,
                    worker_count=selected_workers,
                    rss_hard_bytes=int(self.config["resources"]["process_tree_rss_bytes"]),
                    checkpoint_directory=self.output_root / "checkpoints" / benchmark_id,
                    final_checkpoint=True,
                ),
                initial_aon=bm04,
                run_metadata={"benchmark_id": benchmark_id, "run_id": self.config["run_id"]},
            )
            rows, scenarios, regions, bacns = self._save_solver_artifacts(
                benchmark_id, solver_result, graph, trace_maps
            )
            iteration_rows.extend(rows)
            solver_iteration_rows[benchmark_id] = rows
            scenario_rows.extend(scenarios)
            region_frames.extend(regions)
            bacn_frames.extend(bacns)
            for trace_map in trace_maps:
                support, _ = active_support_diagnostics(
                    trace_map,
                    solver_result.suite_traces[trace_map.suite_id],
                    solver_result.latest_aon.suite_traces[trace_map.suite_id],
                    ever_active_pair_keys=solver_result.ever_active_regional_pair_keys[trace_map.suite_id],
                    ever_active_bundle_mask=solver_result.ever_active_bundle_masks[trace_map.suite_id],
                )
                support["benchmark_id"] = benchmark_id
                support_rows.append(support)
            solver_candidates.append(
                {
                    "benchmark_id": benchmark_id,
                    "method": method,
                    "final_relative_gap_sptt": solver_result.final_metrics.relative_gap_sptt,
                    "final_relative_gap_tstt": solver_result.final_metrics.relative_gap_tstt,
                    "runtime_seconds": solver_result.runtime_seconds,
                    "cfw_reset_count": solver_result.metadata["cfw_reset_count"],
                    "stable": bool(
                        np.isfinite(solver_result.final_metrics.relative_gap_sptt)
                        and np.isclose(
                            solver_result.latest_aon.reachable_demand
                            + solver_result.latest_aon.unreachable_demand,
                            od.original_total_demand,
                            rtol=1e-12,
                            atol=1e-8,
                        )
                    ),
                }
            )
            completed.append(benchmark_id)
            self.event(
                "benchmark_case_completed",
                benchmark_id=benchmark_id,
                method=method,
                final_relative_gap_sptt=solver_result.final_metrics.relative_gap_sptt,
                runtime_seconds=solver_result.runtime_seconds,
            )
            del solver_result
            gc.collect()
        stable_candidates = [item for item in solver_candidates if item["stable"]]
        if not stable_candidates:
            raise RuntimeError("BM05-BM07 produced no stable solver for sensitivity pilots")
        operative = min(
            stable_candidates,
            key=lambda item: (item["final_relative_gap_sptt"], item["runtime_seconds"]),
        )
        operative_method = str(operative["method"])
        solver_selection_receipt = {
            "schema_id": "phase4.preflight_operative_solver.v1",
            "candidates": solver_candidates,
            "operative_method_for_bm08_bm11": operative_method,
            "rule": "lowest stable three-update refreshed RG_SPTT, runtime as tie-breaker; FW fallback",
            "external_final_solver_selection_made": False,
            "task2_review_required": True,
        }
        _write_json_new(
            self.output_root / "benchmarks" / "BM05_BM07_operative_solver.json",
            solver_selection_receipt,
        )
        self.update_status(
            "EXECUTING_MANDATORY_BENCHMARKS",
            completed_tasks=completed,
            operative_preflight_solver=operative_method,
            final_capacity_multiplier_selected=False,
            next_task="BM08-BM10",
        )

        for benchmark_id, multiplier in (("BM08", 0.25), ("BM09", 0.60), ("BM10", 1.00)):
            self.resource_guard(next_task=benchmark_id)
            sensitivity = solve_assignment(
                graph,
                od,
                trace_maps=trace_maps,
                config=Phase4SolverConfig(
                    method=operative_method,
                    capacity_multiplier=multiplier,
                    max_iterations=2,
                    minimum_iterations=2,
                    worker_count=selected_workers,
                    rss_hard_bytes=int(self.config["resources"]["process_tree_rss_bytes"]),
                    checkpoint_directory=self.output_root / "checkpoints" / benchmark_id,
                    final_checkpoint=True,
                ),
                initial_aon=bm04,
                run_metadata={"benchmark_id": benchmark_id, "run_id": self.config["run_id"]},
            )
            rows, scenarios, regions, bacns = self._save_solver_artifacts(
                benchmark_id, sensitivity, graph, trace_maps
            )
            iteration_rows.extend(rows)
            scenario_rows.extend(scenarios)
            region_frames.extend(regions)
            bacn_frames.extend(bacns)
            for trace_map in trace_maps:
                support, _ = active_support_diagnostics(
                    trace_map,
                    sensitivity.suite_traces[trace_map.suite_id],
                    sensitivity.latest_aon.suite_traces[trace_map.suite_id],
                    ever_active_pair_keys=sensitivity.ever_active_regional_pair_keys[trace_map.suite_id],
                    ever_active_bundle_mask=sensitivity.ever_active_bundle_masks[trace_map.suite_id],
                )
                support["benchmark_id"] = benchmark_id
                support_rows.append(support)
            completed.append(benchmark_id)
            del sensitivity
            gc.collect()
        self.update_status(
            "EXECUTING_MANDATORY_BENCHMARKS",
            completed_tasks=completed,
            operative_preflight_solver=operative_method,
            next_task="BM11",
        )

        # BM11: continuous two-update state versus one-update checkpoint + restart.
        self.resource_guard(next_task="BM11")
        continuous = solve_assignment(
            graph,
            od,
            trace_maps=trace_maps,
            config=Phase4SolverConfig(
                method=operative_method,
                capacity_multiplier=0.4,
                max_iterations=2,
                minimum_iterations=2,
                worker_count=selected_workers,
                rss_hard_bytes=int(self.config["resources"]["process_tree_rss_bytes"]),
                checkpoint_directory=self.output_root / "checkpoints" / "BM11_continuous",
                checkpoint_iterations=(1,),
                final_checkpoint=True,
            ),
            initial_aon=bm04,
            run_metadata={"benchmark_id": "BM11_CONTINUOUS", "run_id": self.config["run_id"]},
        )
        restart_state = load_phase4_checkpoint(
            self.output_root / "checkpoints" / "BM11_continuous" / "iteration_0001",
            trace_maps=trace_maps,
        )
        resumed = solve_assignment(
            graph,
            od,
            trace_maps=trace_maps,
            config=Phase4SolverConfig(
                method=operative_method,
                capacity_multiplier=0.4,
                max_iterations=2,
                minimum_iterations=1,
                worker_count=selected_workers,
                rss_hard_bytes=int(self.config["resources"]["process_tree_rss_bytes"]),
                checkpoint_directory=self.output_root / "checkpoints" / "BM11_restart",
                final_checkpoint=True,
            ),
            restart_state=restart_state,
            run_metadata={"benchmark_id": "BM11_RESTART", "run_id": self.config["run_id"]},
        )
        trace_reproduces, restart_trace_comparison = _trace_close(
            continuous.suite_traces, resumed.suite_traces
        )
        restart_receipt = {
            "schema_id": "phase4.bm11.checkpoint_restart.v1",
            "method": operative_method,
            "capacity_multiplier": 0.4,
            "continuous_completed_updates": 2,
            "checkpoint_after_updates": 1,
            "restart_completed_updates": 2,
            "physical_flow_bitwise_equal": bool(np.array_equal(continuous.flow, resumed.flow)),
            "physical_flow_max_abs_difference": float(np.max(np.abs(continuous.flow - resumed.flow))),
            "final_cost_bitwise_equal": bool(np.array_equal(continuous.cost, resumed.cost)),
            "final_cost_max_abs_difference": float(np.max(np.abs(continuous.cost - resumed.cost))),
            "trace_close": trace_reproduces,
            "trace_comparison": restart_trace_comparison,
            "continuous_final_metrics": _solver_summary("BM11_CONTINUOUS", continuous)["final_metrics"],
            "restart_final_metrics": _solver_summary("BM11_RESTART", resumed)["final_metrics"],
            "pass": bool(
                np.array_equal(continuous.flow, resumed.flow)
                and np.array_equal(continuous.cost, resumed.cost)
                and trace_reproduces
            ),
        }
        if not restart_receipt["pass"]:
            raise RuntimeError("BM11 checkpoint/restart did not reproduce the continuous state")
        _write_json_new(self.output_root / "benchmarks" / "BM11_checkpoint_restart.json", restart_receipt)
        rows, scenarios, regions, bacns = self._save_solver_artifacts(
            "BM11_CONTINUOUS", continuous, graph, trace_maps
        )
        iteration_rows.extend(rows)
        scenario_rows.extend(scenarios)
        region_frames.extend(regions)
        bacn_frames.extend(bacns)
        rows, _, _, _ = self._save_solver_artifacts("BM11_RESTART", resumed, graph, trace_maps)
        iteration_rows.extend(rows)
        completed.append("BM11")
        del continuous, resumed, restart_state
        gc.collect()

        # Required controlling full-origin rerun after the principal exact optimization.
        bm04_rerun = full_aon_case(
            "BM04R_CONTROLLING_RERUN", trace_maps, "s02-s06", checkpoint=False
        )
        rerun_trace_close, rerun_trace_comparison = _trace_close(
            bm04.suite_traces, bm04_rerun.suite_traces
        )
        rerun_receipt = {
            "schema_id": "phase4.controlling_rerun.v1",
            "original_benchmark_id": "BM04",
            "rerun_benchmark_id": "BM04R_CONTROLLING_RERUN",
            "physical_flow_bitwise_equal": bool(
                np.array_equal(bm04.physical_flow, bm04_rerun.physical_flow)
            ),
            "physical_flow_max_abs_difference": float(
                np.max(np.abs(bm04.physical_flow - bm04_rerun.physical_flow))
            ),
            "sptt_difference": float(bm04_rerun.sptt - bm04.sptt),
            "trace_close": rerun_trace_close,
            "trace_comparison": rerun_trace_comparison,
            "pass": bool(
                np.array_equal(bm04.physical_flow, bm04_rerun.physical_flow)
                and rerun_trace_close
            ),
        }
        if not rerun_receipt["pass"]:
            raise RuntimeError("Controlling full-origin rerun is not deterministic")
        _write_json_new(
            self.output_root / "benchmarks" / "BM04R_controlling_rerun.json",
            rerun_receipt,
        )
        completed.append("CONTROLLING-RERUN")

        # Consolidated machine-readable benchmark tables.
        tables = self.output_root / "tables"
        _write_frame_new(pd.json_normalize(aon_rows), tables / "aon_scaling_and_trace_overhead.csv")
        _write_frame_new(pd.json_normalize(iteration_rows), tables / "physical_iteration_log.parquet")
        _write_frame_new(pd.json_normalize(scenario_rows), tables / "capacity_multiplier_sensitivity.csv")
        _write_frame_new(pd.json_normalize(support_rows), tables / "active_support_summary.csv")
        if region_frames:
            _write_frame_new(
                pd.concat(region_frames, ignore_index=True),
                tables / "regional_active_support.parquet",
            )
        if bacn_frames:
            _write_frame_new(
                pd.concat(bacn_frames, ignore_index=True),
                tables / "bacn_internal_statistics.parquet",
            )
        _write_frame_new(pd.DataFrame(solver_candidates), tables / "solver_comparison.csv")
        _write_frame_new(pd.DataFrame(worker_equivalence), tables / "worker_equivalence.csv")

        forecast = self._build_runtime_forecast(
            solver_candidates=solver_candidates,
            solver_iteration_rows=solver_iteration_rows,
            operative_method=operative_method,
        )
        _write_json_new(self.output_root / "benchmarks" / "production_runtime_forecast.json", forecast)
        self._build_figures(
            iteration_rows=iteration_rows,
            aon_rows=aon_rows,
            scenario_rows=scenario_rows,
            support_rows=support_rows,
            forecast=forecast,
        )
        resource_rows.append(self.resource_guard(next_task="FINAL-VALIDATE"))
        _write_json_new(self.output_root / "validation" / "benchmark_resource_receipt.json", resource_rows)
        completion = {
            "schema_id": "phase4.task1.benchmark_completion.v1",
            "status": "PASS",
            "mandatory_benchmark_ids": sorted(mandatory_ids),
            "completed_benchmark_ids": sorted(set(value for value in completed if value.startswith("BM"))),
            "all_mandatory_categories_completed": mandatory_ids.issubset(set(completed)),
            "target_extensions_completed": [],
            "target_extensions_curtailed_to_protect_bounded_preflight": True,
            "selected_worker_count": selected_workers,
            "operative_preflight_solver": operative_method,
            "final_solver_selected": False,
            "final_capacity_multiplier_selected": False,
            "bm13": bm13_receipt,
            "controlling_rerun_pass": rerun_receipt["pass"],
            "checkpoint_restart_pass": restart_receipt["pass"],
            "frozen_explicit_trace_oracle_pass": True,
            "runtime_seconds": float(time.perf_counter() - stage_started),
            "resource_guard": resource_rows[-1],
        }
        if not completion["all_mandatory_categories_completed"]:
            raise RuntimeError("Not every mandatory BM01-BM12 category completed")
        _write_json_new(completion_path, completion)
        self.event("mandatory_benchmarks_completed", **completion)
        self.update_status(
            "MANDATORY_BENCHMARKS_COMPLETE",
            completed_tasks=completed,
            selected_worker_count=selected_workers,
            operative_preflight_solver=operative_method,
            next_task="FINAL-VALIDATE",
        )

    def _resume_benchmark_after_bm04(self, prior_status: dict[str, Any]) -> None:
        """Resume the write-new matrix after the recorded pre-BM05 guard stop."""

        completion_path = self.output_root / "status" / "benchmark_completion.json"
        if completion_path.exists():
            raise FileExistsError("Benchmark completion evidence already exists")
        self.event(
            "mandatory_benchmarks_resumed",
            resume_after="BM04",
            reason="accepted unreachable demand now conserved as explicit classified demand",
        )
        stage_started = time.perf_counter()
        graph, od, trace_maps = self._load_inputs(verify_directory_digests=False)
        trace_by_id = {item.suite_id: item for item in trace_maps}
        selected_workers = int(prior_status["selected_worker_count"])
        bm04 = load_phase4_checkpoint_aon(
            self.output_root / "checkpoints" / "BM04",
            trace_maps=trace_maps,
        )
        if not np.isclose(
            bm04.reachable_demand + bm04.unreachable_demand,
            od.original_total_demand,
            rtol=1e-12,
            atol=1e-8,
        ):
            raise RuntimeError("Reopened BM04 checkpoint does not classify all accepted demand")

        aon_rows = pd.read_csv(
            self.output_root / "benchmarks" / "BM01" / "origin_worker_scaling.csv"
        ).to_dict(orient="records")
        for benchmark_id in ("BM02", "BM03", "BM04"):
            aon_rows.append(
                json.loads(
                    (self.output_root / "benchmarks" / benchmark_id / "aon_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
        # The interrupted process had not persisted its bounded trace-scaling
        # rows, so repeat only those small exact cases into the consolidated table.
        for scope_name, selected_maps in (
            ("s03", [trace_by_id["s03"]]),
            ("s02-s06", trace_maps),
        ):
            for worker_count in (1, 2, 4):
                case_id = f"TRACE_SCALE_RESUME_o128_{scope_name.replace('-', '_')}_w{worker_count}"
                self.resource_guard(next_task=case_id)
                trace_scale = run_exact_aon(
                    graph,
                    od,
                    graph.free_time,
                    trace_maps=selected_maps,
                    config=AONExecutionConfig(
                        worker_count=worker_count,
                        max_origins=128,
                        coarse_hash_capacity=1 << 20,
                        fine_hash_capacity=1 << 20,
                    ),
                )
                row = _aon_summary(
                    case_id,
                    trace_scale,
                    trace_scope=scope_name,
                    capacity_multiplier=0.4,
                )
                row["origin_scope"] = 128
                aon_rows.append(row)
                del trace_scale
                gc.collect()

        iteration_rows: list[dict[str, Any]] = []
        scenario_rows: list[dict[str, Any]] = []
        support_rows: list[dict[str, Any]] = []
        region_frames: list[pd.DataFrame] = []
        bacn_frames: list[pd.DataFrame] = []
        solver_candidates: list[dict[str, Any]] = []
        solver_iteration_rows: dict[str, list[dict[str, Any]]] = {}
        completed = list(prior_status.get("completed_tasks", []))

        bm05_execution_id = "BM05"
        bm05_partial_root = self.output_root / "benchmarks" / "BM05"
        bm05_partial_checkpoint = (
            self.output_root / "checkpoints" / "BM05" / "final_iteration_0003"
        )
        bm05_reporting_stop_repaired = False
        if (
            "BM05" not in completed
            and (bm05_partial_root / "solver_summary.json").exists()
            and bm05_partial_checkpoint.exists()
        ):
            # BM05's numerical state and immutable checkpoint completed before
            # its reporting-only active-support assertion stopped the process.
            # Preserve that evidence and rerun under a write-new ID after
            # classifying the Phase 3-excluded same-terminal pairs explicitly.
            bm05_execution_id = "BM05_REPAIR"
            bm05_reporting_stop_repaired = True
            if (self.output_root / "benchmarks" / bm05_execution_id).exists() or (
                self.output_root / "checkpoints" / bm05_execution_id
            ).exists():
                raise FileExistsError("BM05 write-new repair target already exists")
            stopped_state = load_phase4_checkpoint(
                bm05_partial_checkpoint,
                trace_maps=trace_maps,
            )
            inspection_rows: list[dict[str, Any]] = []
            for trace_map in trace_maps:
                keys = stopped_state.suite_traces[
                    trace_map.suite_id
                ].coarse_regional_pair_flow.keys.astype(np.uint64, copy=False)
                entry = (keys >> np.uint64(32)).astype(np.int64)
                exit_terminal = (keys & np.uint64(0xFFFFFFFF)).astype(np.int64)
                region = trace_map.terminal_region_local[entry]
                raw_count = np.bincount(
                    region, minlength=trace_map.region_count
                ).astype(np.int64)
                self_mask = entry == exit_terminal
                self_count = np.bincount(
                    region[self_mask], minlength=trace_map.region_count
                ).astype(np.int64)
                conceptual_compatible_count = raw_count - self_count
                if np.any(
                    conceptual_compatible_count
                    > trace_map.conceptual_pair_count_by_region
                ):
                    raise RuntimeError(
                        f"{trace_map.suite_id} BM05 stop is not explained by "
                        "Phase 3-excluded same-terminal pairs"
                    )
                inspection_rows.append(
                    {
                        "suite_id": trace_map.suite_id,
                        "positive_current_raw_pair_count": int(len(keys)),
                        "positive_current_self_pair_count": int(np.count_nonzero(self_mask)),
                        "regions_with_raw_count_above_conceptual": int(
                            np.count_nonzero(
                                raw_count > trace_map.conceptual_pair_count_by_region
                            )
                        ),
                        "raw_excess_over_conceptual_pair_count": int(
                            np.maximum(
                                raw_count - trace_map.conceptual_pair_count_by_region, 0
                            ).sum(dtype=np.int64)
                        ),
                        "non_self_count_within_phase3_conceptual_support": True,
                    }
                )
            supersession = {
                "schema_id": "phase4.bm05.reporting_stop_supersession.v1",
                "matrix_benchmark_id": "BM05",
                "stopped_execution_id": "BM05",
                "replacement_execution_id": bm05_execution_id,
                "failure_stage": "reporting_after_numerical_checkpoint",
                "numerical_checkpoint_preserved": True,
                "stopped_checkpoint_directory": str(bm05_partial_checkpoint.resolve()),
                "stopped_checkpoint_receipt_sha256": file_sha256(
                    bm05_partial_checkpoint / "checkpoint_receipt.json"
                ),
                "stopped_solver_summary_sha256": file_sha256(
                    bm05_partial_root / "solver_summary.json"
                ),
                "root_cause": (
                    "Exact tracing retains positive same-terminal regional traversals, while "
                    "Phase 3 conceptual support intentionally excludes source_terminal_id "
                    "equal to target_terminal_id. The stopped assertion compared raw traced "
                    "counts directly with that non-self denominator."
                ),
                "phase3_definition_source": (
                    "partC_rev1_src/phase3a/regional_pairs.py: target_id == source_id is excluded"
                ),
                "stopped_checkpoint_inspection": inspection_rows,
                "scientific_state_changed": False,
                "reporting_classification_changed": True,
            }
            _write_json_new(
                self.output_root
                / "benchmarks"
                / "BM05_reporting_stop_supersession.json",
                supersession,
            )
            del stopped_state
            gc.collect()

        for benchmark_id, method, matrix_benchmark_id in (
            (bm05_execution_id, "MSA", "BM05"),
            ("BM06", "FW", "BM06"),
            ("BM07", "SAFEGUARDED_CFW", "BM07"),
        ):
            self.resource_guard(next_task=benchmark_id)
            self.event("benchmark_case_started", benchmark_id=benchmark_id, method=method, resumed=True)
            result = solve_assignment(
                graph,
                od,
                trace_maps=trace_maps,
                config=Phase4SolverConfig(
                    method=method,
                    capacity_multiplier=0.4,
                    max_iterations=3,
                    minimum_iterations=3,
                    worker_count=selected_workers,
                    rss_hard_bytes=int(self.config["resources"]["process_tree_rss_bytes"]),
                    checkpoint_directory=self.output_root / "checkpoints" / benchmark_id,
                    final_checkpoint=True,
                ),
                initial_aon=bm04,
                run_metadata={
                    "benchmark_id": benchmark_id,
                    "matrix_benchmark_id": matrix_benchmark_id,
                    "run_id": self.config["run_id"],
                    "resumed_after_bm04_guard_stop": True,
                },
            )
            rows, scenarios, regions, bacns = self._save_solver_artifacts(
                benchmark_id, result, graph, trace_maps
            )
            for row in rows:
                row["matrix_benchmark_id"] = matrix_benchmark_id
            for row in scenarios:
                row["matrix_benchmark_id"] = matrix_benchmark_id
            for frame in (*regions, *bacns):
                frame.insert(1, "matrix_benchmark_id", matrix_benchmark_id)
            iteration_rows.extend(rows)
            solver_iteration_rows[benchmark_id] = rows
            scenario_rows.extend(scenarios)
            region_frames.extend(regions)
            bacn_frames.extend(bacns)
            for trace_map in trace_maps:
                support, _ = active_support_diagnostics(
                    trace_map,
                    result.suite_traces[trace_map.suite_id],
                    result.latest_aon.suite_traces[trace_map.suite_id],
                    ever_active_pair_keys=result.ever_active_regional_pair_keys[trace_map.suite_id],
                    ever_active_bundle_mask=result.ever_active_bundle_masks[trace_map.suite_id],
                )
                support["benchmark_id"] = benchmark_id
                support["matrix_benchmark_id"] = matrix_benchmark_id
                support_rows.append(support)
            solver_candidates.append(
                {
                    "benchmark_id": benchmark_id,
                    "matrix_benchmark_id": matrix_benchmark_id,
                    "method": method,
                    "final_relative_gap_sptt": result.final_metrics.relative_gap_sptt,
                    "final_relative_gap_tstt": result.final_metrics.relative_gap_tstt,
                    "runtime_seconds": result.runtime_seconds,
                    "cfw_reset_count": result.metadata["cfw_reset_count"],
                    "reachable_demand": result.latest_aon.reachable_demand,
                    "unreachable_demand": result.latest_aon.unreachable_demand,
                    "classified_demand": (
                        result.latest_aon.reachable_demand + result.latest_aon.unreachable_demand
                    ),
                    "stable": bool(
                        np.isfinite(result.final_metrics.relative_gap_sptt)
                        and np.isclose(
                            result.latest_aon.reachable_demand + result.latest_aon.unreachable_demand,
                            od.original_total_demand,
                            rtol=1e-12,
                            atol=1e-8,
                        )
                    ),
                }
            )
            completed.append(matrix_benchmark_id)
            self.event(
                "benchmark_case_completed",
                benchmark_id=benchmark_id,
                matrix_benchmark_id=matrix_benchmark_id,
                method=method,
                final_relative_gap_sptt=result.final_metrics.relative_gap_sptt,
                runtime_seconds=result.runtime_seconds,
                unreachable_demand=result.latest_aon.unreachable_demand,
            )
            del result
            gc.collect()

        stable = [item for item in solver_candidates if item["stable"]]
        if not stable:
            raise RuntimeError("BM05-BM07 produced no stable conserved-demand solver")
        operative = min(
            stable,
            key=lambda item: (item["final_relative_gap_sptt"], item["runtime_seconds"]),
        )
        operative_method = str(operative["method"])
        _write_json_new(
            self.output_root / "benchmarks" / "BM05_BM07_operative_solver.json",
            {
                "schema_id": "phase4.preflight_operative_solver.v1",
                "candidates": solver_candidates,
                "operative_method_for_bm08_bm11": operative_method,
                "rule": "lowest stable three-update refreshed RG_SPTT, runtime as tie-breaker; FW fallback",
                "external_final_solver_selection_made": False,
                "task2_review_required": True,
            },
        )
        self.update_status(
            "EXECUTING_MANDATORY_BENCHMARKS",
            completed_tasks=completed,
            operative_preflight_solver=operative_method,
            next_task="BM08-BM10",
        )

        for benchmark_id, multiplier in (("BM08", 0.25), ("BM09", 0.60), ("BM10", 1.00)):
            self.resource_guard(next_task=benchmark_id)
            result = solve_assignment(
                graph,
                od,
                trace_maps=trace_maps,
                config=Phase4SolverConfig(
                    method=operative_method,
                    capacity_multiplier=multiplier,
                    max_iterations=2,
                    minimum_iterations=2,
                    worker_count=selected_workers,
                    rss_hard_bytes=int(self.config["resources"]["process_tree_rss_bytes"]),
                    checkpoint_directory=self.output_root / "checkpoints" / benchmark_id,
                    final_checkpoint=True,
                ),
                initial_aon=bm04,
                run_metadata={"benchmark_id": benchmark_id, "run_id": self.config["run_id"]},
            )
            rows, scenarios, regions, bacns = self._save_solver_artifacts(
                benchmark_id, result, graph, trace_maps
            )
            iteration_rows.extend(rows)
            scenario_rows.extend(scenarios)
            region_frames.extend(regions)
            bacn_frames.extend(bacns)
            for trace_map in trace_maps:
                support, _ = active_support_diagnostics(
                    trace_map,
                    result.suite_traces[trace_map.suite_id],
                    result.latest_aon.suite_traces[trace_map.suite_id],
                    ever_active_pair_keys=result.ever_active_regional_pair_keys[trace_map.suite_id],
                    ever_active_bundle_mask=result.ever_active_bundle_masks[trace_map.suite_id],
                )
                support["benchmark_id"] = benchmark_id
                support_rows.append(support)
            completed.append(benchmark_id)
            del result
            gc.collect()

        self.update_status(
            "EXECUTING_MANDATORY_BENCHMARKS",
            completed_tasks=completed,
            operative_preflight_solver=operative_method,
            next_task="BM11",
        )
        self.resource_guard(next_task="BM11")
        continuous = solve_assignment(
            graph,
            od,
            trace_maps=trace_maps,
            config=Phase4SolverConfig(
                method=operative_method,
                capacity_multiplier=0.4,
                max_iterations=2,
                minimum_iterations=2,
                worker_count=selected_workers,
                rss_hard_bytes=int(self.config["resources"]["process_tree_rss_bytes"]),
                checkpoint_directory=self.output_root / "checkpoints" / "BM11_continuous",
                checkpoint_iterations=(1,),
                final_checkpoint=True,
            ),
            initial_aon=bm04,
            run_metadata={"benchmark_id": "BM11_CONTINUOUS", "run_id": self.config["run_id"]},
        )
        restart_state = load_phase4_checkpoint(
            self.output_root / "checkpoints" / "BM11_continuous" / "iteration_0001",
            trace_maps=trace_maps,
        )
        resumed = solve_assignment(
            graph,
            od,
            trace_maps=trace_maps,
            config=Phase4SolverConfig(
                method=operative_method,
                capacity_multiplier=0.4,
                max_iterations=2,
                minimum_iterations=1,
                worker_count=selected_workers,
                rss_hard_bytes=int(self.config["resources"]["process_tree_rss_bytes"]),
                checkpoint_directory=self.output_root / "checkpoints" / "BM11_restart",
                final_checkpoint=True,
            ),
            restart_state=restart_state,
            run_metadata={"benchmark_id": "BM11_RESTART", "run_id": self.config["run_id"]},
        )
        traces_equal, trace_comparison = _trace_close(
            continuous.suite_traces, resumed.suite_traces
        )
        restart_receipt = {
            "schema_id": "phase4.bm11.checkpoint_restart.v1",
            "method": operative_method,
            "capacity_multiplier": 0.4,
            "physical_flow_bitwise_equal": bool(np.array_equal(continuous.flow, resumed.flow)),
            "physical_flow_max_abs_difference": float(np.max(np.abs(continuous.flow - resumed.flow))),
            "final_cost_bitwise_equal": bool(np.array_equal(continuous.cost, resumed.cost)),
            "final_cost_max_abs_difference": float(np.max(np.abs(continuous.cost - resumed.cost))),
            "trace_close": traces_equal,
            "trace_comparison": trace_comparison,
            "pass": bool(
                np.array_equal(continuous.flow, resumed.flow)
                and np.array_equal(continuous.cost, resumed.cost)
                and traces_equal
            ),
        }
        if not restart_receipt["pass"]:
            raise RuntimeError("BM11 checkpoint/restart did not reproduce the continuous state")
        _write_json_new(
            self.output_root / "benchmarks" / "BM11_checkpoint_restart.json",
            restart_receipt,
        )
        rows, scenarios, regions, bacns = self._save_solver_artifacts(
            "BM11_CONTINUOUS", continuous, graph, trace_maps
        )
        iteration_rows.extend(rows)
        scenario_rows.extend(scenarios)
        region_frames.extend(regions)
        bacn_frames.extend(bacns)
        rows, _, _, _ = self._save_solver_artifacts(
            "BM11_RESTART", resumed, graph, trace_maps
        )
        iteration_rows.extend(rows)
        completed.append("BM11")
        del continuous, resumed, restart_state
        gc.collect()

        self.resource_guard(next_task="BM04R_CONTROLLING_RERUN")
        rerun = run_exact_aon(
            graph,
            od,
            graph.free_time,
            trace_maps=trace_maps,
            config=AONExecutionConfig(worker_count=selected_workers),
        )
        rerun_row = _aon_summary(
            "BM04R_CONTROLLING_RERUN",
            rerun,
            trace_scope="s02-s06",
            capacity_multiplier=0.4,
        )
        rerun_row["origin_scope"] = len(od.origin_group_node_idx)
        aon_rows.append(rerun_row)
        rerun_trace_close, rerun_comparison = _trace_close(
            bm04.suite_traces, rerun.suite_traces
        )
        rerun_receipt = {
            "schema_id": "phase4.controlling_rerun.v1",
            "original_benchmark_id": "BM04",
            "rerun_benchmark_id": "BM04R_CONTROLLING_RERUN",
            "physical_flow_bitwise_equal": bool(np.array_equal(bm04.physical_flow, rerun.physical_flow)),
            "physical_flow_max_abs_difference": float(
                np.max(np.abs(bm04.physical_flow - rerun.physical_flow))
            ),
            "sptt_difference": float(rerun.sptt - bm04.sptt),
            "trace_close": rerun_trace_close,
            "trace_comparison": rerun_comparison,
            "pass": bool(np.array_equal(bm04.physical_flow, rerun.physical_flow) and rerun_trace_close),
        }
        if not rerun_receipt["pass"]:
            raise RuntimeError("Controlling full-origin rerun is not deterministic")
        _write_json_new(
            self.output_root / "benchmarks" / "BM04R_controlling_rerun.json",
            rerun_receipt,
        )
        completed.append("CONTROLLING-RERUN")

        tables = self.output_root / "tables"
        _write_frame_new(pd.json_normalize(aon_rows), tables / "aon_scaling_and_trace_overhead.csv")
        _write_frame_new(pd.json_normalize(iteration_rows), tables / "physical_iteration_log.parquet")
        _write_frame_new(pd.json_normalize(scenario_rows), tables / "capacity_multiplier_sensitivity.csv")
        _write_frame_new(pd.json_normalize(support_rows), tables / "active_support_summary.csv")
        _write_frame_new(pd.DataFrame(solver_candidates), tables / "solver_comparison.csv")
        worker_equivalence = json.loads(
            (self.output_root / "benchmarks" / "BM01" / "worker_equivalence.json").read_text(
                encoding="utf-8"
            )
        )
        _write_frame_new(pd.DataFrame(worker_equivalence), tables / "worker_equivalence.csv")
        if region_frames:
            _write_frame_new(
                pd.concat(region_frames, ignore_index=True),
                tables / "regional_active_support.parquet",
            )
        if bacn_frames:
            _write_frame_new(
                pd.concat(bacn_frames, ignore_index=True),
                tables / "bacn_internal_statistics.parquet",
            )
        forecast = self._build_runtime_forecast(
            solver_candidates=solver_candidates,
            solver_iteration_rows=solver_iteration_rows,
            operative_method=operative_method,
        )
        _write_json_new(
            self.output_root / "benchmarks" / "production_runtime_forecast.json",
            forecast,
        )
        self._build_figures(
            iteration_rows=iteration_rows,
            aon_rows=aon_rows,
            scenario_rows=scenario_rows,
            support_rows=support_rows,
            forecast=forecast,
        )
        final_resource = self.resource_guard(next_task="FINAL-VALIDATE")
        _write_json_new(
            self.output_root / "validation" / "benchmark_resource_receipt.json",
            [final_resource],
        )
        mandatory = {f"BM{index:02d}" for index in range(1, 13)}
        completion = {
            "schema_id": "phase4.task1.benchmark_completion.v1",
            "status": "PASS",
            "mandatory_benchmark_ids": sorted(mandatory),
            "completed_benchmark_ids": sorted(set(value for value in completed if value.startswith("BM"))),
            "all_mandatory_categories_completed": mandatory.issubset(set(completed)),
            "target_extensions_completed": [],
            "target_extensions_curtailed_to_protect_bounded_preflight": True,
            "selected_worker_count": selected_workers,
            "operative_preflight_solver": operative_method,
            "final_solver_selected": False,
            "final_capacity_multiplier_selected": False,
            "bm13": json.loads(
                (self.output_root / "benchmarks" / "BM13_bounded_inspection.json").read_text(
                    encoding="utf-8"
                )
            ),
            "unreachable_demand": bm04.unreachable_demand,
            "unreachable_demand_fraction": float(
                bm04.unreachable_demand / od.original_total_demand
            ),
            "all_demand_classified_and_conserved": True,
            "controlling_rerun_pass": rerun_receipt["pass"],
            "checkpoint_restart_pass": restart_receipt["pass"],
            "frozen_explicit_trace_oracle_pass": True,
            "resumed_after_pre_bm05_guard_stop": True,
            "bm05_reporting_stop_repaired_with_write_new_execution": (
                bm05_reporting_stop_repaired
            ),
            "runtime_seconds_after_resume": float(time.perf_counter() - stage_started),
            "resource_guard": final_resource,
        }
        if not completion["all_mandatory_categories_completed"]:
            raise RuntimeError("Not every mandatory BM01-BM12 category completed after resume")
        _write_json_new(completion_path, completion)
        self.event("mandatory_benchmarks_completed", **completion)
        self.update_status(
            "MANDATORY_BENCHMARKS_COMPLETE",
            completed_tasks=completed,
            selected_worker_count=selected_workers,
            operative_preflight_solver=operative_method,
            next_task="FINAL-VALIDATE",
        )

    def _build_runtime_forecast(
        self,
        *,
        solver_candidates: Sequence[dict[str, Any]],
        solver_iteration_rows: dict[str, list[dict[str, Any]]],
        operative_method: str,
    ) -> dict[str, Any]:
        targets = (5e-4, 1e-4, 5e-5)
        forecasts: list[dict[str, Any]] = []
        for candidate in solver_candidates:
            benchmark_id = str(candidate["benchmark_id"])
            rows = solver_iteration_rows[benchmark_id]
            points = [
                row
                for row in rows
                if row.get("relative_gap_sptt") is not None
                and float(row["relative_gap_sptt"]) > 0.0
            ]
            iterations = np.asarray(
                [float(row["evaluated_state_iteration"]) for row in points], dtype=np.float64
            )
            log_gap = np.log(
                np.asarray([float(row["relative_gap_sptt"]) for row in points], dtype=np.float64)
            )
            times = np.asarray(
                [
                    float(row["iteration_wall_seconds"])
                    for row in rows
                    if not bool(row.get("final_refresh", False))
                ],
                dtype=np.float64,
            )
            slope = None
            intercept = None
            if len(iterations) >= 3 and np.ptp(iterations) > 0.0:
                slope_value, intercept_value = np.polyfit(iterations, log_gap, 1)
                slope, intercept = float(slope_value), float(intercept_value)
            median_time = float(np.median(times)) if len(times) else None
            p95_time = float(np.quantile(times, 0.95)) if len(times) else None
            last_iteration = float(iterations[-1]) if len(iterations) else 0.0
            for target in targets:
                if slope is not None and slope < 0.0 and median_time is not None:
                    target_iteration = max(last_iteration, (np.log(target) - float(intercept)) / slope)
                    additional = max(0.0, target_iteration - last_iteration)
                    compute_seconds = additional * median_time
                    fixed_overhead = (p95_time or median_time) + 600.0 + 3600.0
                    central = compute_seconds + fixed_overhead
                    lower = 0.8 * central
                    upper = 1.5 * central
                else:
                    target_iteration = None
                    additional = None
                    central = lower = upper = None
                forecasts.append(
                    {
                        "benchmark_id": benchmark_id,
                        "method": candidate["method"],
                        "cold_start": True,
                        "compatible_warm_start_available": False,
                        "target_relative_gap_sptt": target,
                        "observed_point_count": len(points),
                        "log_gap_per_iteration_slope": slope,
                        "log_gap_intercept": intercept,
                        "observed_median_iteration_seconds": median_time,
                        "observed_p95_iteration_seconds": p95_time,
                        "projected_target_iteration": target_iteration,
                        "projected_additional_iterations": additional,
                        "projected_seconds_central": central,
                        "projected_seconds_lower": lower,
                        "projected_seconds_upper": upper,
                        "uncertainty_margin_lower_fraction": 0.20,
                        "uncertainty_margin_upper_fraction": 0.50,
                        "fixed_overhead_includes": [
                            "one p95 final refresh/checkpoint allowance",
                            "600 seconds validation allowance",
                            "3600 seconds sealing allowance",
                        ],
                        "trajectory_strength": "WEAK_SHORT_PREFLIGHT_TRAJECTORY",
                    }
                )
        return {
            "schema_id": "phase4.task1.production_runtime_forecast.v1",
            "official_gap_denominator_recommendation": "SPTT",
            "epsilon": 1e-12,
            "operative_preflight_method": operative_method,
            "forecast_rows": forecasts,
            "warning": (
                "Three-update pilots do not identify an asymptotic rate reliably; projections are "
                "screening estimates with deliberately wide uncertainty and are not production commitments."
            ),
            "final_solver_selected": False,
            "final_capacity_multiplier_selected": False,
        }

    def _build_figures(
        self,
        *,
        iteration_rows: Sequence[dict[str, Any]],
        aon_rows: Sequence[dict[str, Any]],
        scenario_rows: Sequence[dict[str, Any]],
        support_rows: Sequence[dict[str, Any]],
        forecast: dict[str, Any],
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure_root = self.output_root / "figures"
        iterations = pd.DataFrame(iteration_rows)
        comparison_scope = (
            iterations["matrix_benchmark_id"]
            if "matrix_benchmark_id" in iterations
            else iterations["benchmark_id"]
        )
        comparison = iterations[comparison_scope.isin(["BM05", "BM06", "BM07"])]

        fig, ax = plt.subplots(figsize=(8, 5))
        for benchmark_id, group in comparison.groupby("benchmark_id"):
            ax.semilogy(
                group["evaluated_state_iteration"],
                group["relative_gap_sptt"],
                marker="o",
                label=f"{benchmark_id} {group['method'].iloc[0]}",
            )
        ax.set(xlabel="Refreshed state iteration", ylabel="RG_SPTT", title="Refreshed gap trajectory")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_root / "refreshed_gap_by_algorithm.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for benchmark_id, group in comparison.groupby("benchmark_id"):
            label = f"{benchmark_id} {group['method'].iloc[0]}"
            axes[0].plot(group["evaluated_state_iteration"], group["objective_before"], marker="o", label=label)
            axes[1].plot(group["evaluated_state_iteration"], group["tstt"], marker="o", label=label)
        axes[0].set(title="Beckmann objective", xlabel="Iteration")
        axes[1].set(title="TSTT", xlabel="Iteration")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_root / "objective_and_tstt_trajectory.png", dpi=160)
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
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figure_root / "iteration_runtime_decomposition.png", dpi=160)
        plt.close(fig)

        overhead = pd.DataFrame(aon_rows)
        overhead = overhead[
            overhead["benchmark_id"].str.startswith("TRACE_SCALE")
            | overhead["benchmark_id"].str.startswith("BM01_o128_")
            | overhead["benchmark_id"].isin(["BM02", "BM03", "BM04"])
        ]
        fig, ax = plt.subplots(figsize=(9, 5))
        for scope, group in overhead.groupby("trace_scope"):
            ax.plot(group["worker_count"], group["aon_wall_seconds"], marker="o", label=scope)
        ax.set(title="Tracer overhead by suite and worker scope", xlabel="Workers", ylabel="AON wall seconds")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_root / "tracer_overhead_by_scope_and_workers.png", dpi=160)
        plt.close(fig)

        scenarios = pd.DataFrame(scenario_rows)
        sensitivity = scenarios[scenarios["benchmark_id"].isin(["BM08", "BM09", "BM10"])]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            sensitivity["capacity_multiplier"],
            sensitivity["tti_flow_weighted_q095"],
            marker="o",
        )
        ax.set(
            title="Capacity multiplier congestion sensitivity",
            xlabel="Capacity multiplier",
            ylabel="Flow-weighted TTI q95",
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figure_root / "capacity_multiplier_congestion.png", dpi=160)
        plt.close(fig)

        support = pd.DataFrame(support_rows)
        support_scope = (
            support["matrix_benchmark_id"]
            if "matrix_benchmark_id" in support
            else support["benchmark_id"]
        )
        support = support[support_scope.isin(["BM05", "BM06", "BM07"])]
        fig, ax = plt.subplots(figsize=(10, 5))
        for benchmark_id, group in support.groupby("benchmark_id"):
            ax.plot(group["suite_id"], group["positive_current_pair_fraction"], marker="o", label=benchmark_id)
        ax.set(title="Positive current regional-pair fraction", xlabel="Suite", ylabel="Fraction")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_root / "active_regional_pair_fraction.png", dpi=160)
        plt.close(fig)

        checkpoint_rows = []
        for receipt_path in (self.output_root / "checkpoints").rglob("checkpoint_receipt.json"):
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            checkpoint_rows.append(
                {
                    "checkpoint": receipt_path.parent.name,
                    "size_gib": receipt["arrays_size_bytes"] / 1024**3,
                    "write_seconds": receipt["write_seconds"],
                }
            )
        checkpoint_frame = pd.DataFrame(checkpoint_rows)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        if not checkpoint_frame.empty:
            axes[0].plot(np.arange(len(checkpoint_frame)), checkpoint_frame["size_gib"], marker="o")
            axes[1].plot(np.arange(len(checkpoint_frame)), checkpoint_frame["write_seconds"], marker="o")
        axes[0].set(title="Checkpoint size", ylabel="GiB", xlabel="Checkpoint sequence")
        axes[1].set(title="Checkpoint write time", ylabel="Seconds", xlabel="Checkpoint sequence")
        fig.tight_layout()
        fig.savefig(figure_root / "memory_and_checkpoint_growth.png", dpi=160)
        plt.close(fig)

        forecast_frame = pd.DataFrame(forecast["forecast_rows"])
        forecast_1e4 = forecast_frame[forecast_frame["target_relative_gap_sptt"] == 1e-4].dropna(
            subset=["projected_seconds_central"]
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        if not forecast_1e4.empty:
            central = forecast_1e4["projected_seconds_central"].to_numpy(dtype=float) / 3600.0
            lower = forecast_1e4["projected_seconds_lower"].to_numpy(dtype=float) / 3600.0
            upper = forecast_1e4["projected_seconds_upper"].to_numpy(dtype=float) / 3600.0
            x = np.arange(len(forecast_1e4))
            ax.errorbar(x, central, yerr=np.vstack([central - lower, upper - central]), fmt="o", capsize=5)
            ax.set_xticks(x, forecast_1e4["method"].tolist())
        ax.set(title="Screening forecast to refreshed gap 1e-4", ylabel="Hours including fixed overhead")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figure_root / "forecast_to_gap_1e-4.png", dpi=160)
        plt.close(fig)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("prepare", "validate", "benchmark"))
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign = Phase4Campaign(args.config)
    if args.phase_id != campaign.config["phase_id"]:
        raise ValueError("CLI phase ID differs from the explicit execution config")
    if args.step_id != campaign.config["step_id"]:
        raise ValueError("CLI step ID differs from the explicit execution config")
    if args.run_id != campaign.config["run_id"]:
        raise ValueError("CLI run ID differs from the explicit execution config")
    if args.stage == "prepare":
        campaign.prepare()
    elif args.stage == "validate":
        campaign.validate()
    elif args.stage == "benchmark":
        campaign.benchmark()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
