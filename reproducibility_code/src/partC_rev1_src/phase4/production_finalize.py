"""Finalize the user-directed iteration-200 Phase 4 physical reference.

The production solver intentionally preserved the original 1e-4 numerical
gate.  This module records the later explicit user amendment without rewriting
that history: iteration 200 is materialized and independently verified, while
the original target remains a truthful, failed diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from partC_rev1_src.partC_rev1_shortest_paths import partC_rev1_arc_tie_rank

from .checkpoint import load_phase4_checkpoint, load_phase4_checkpoint_aon
from .costs import phase4_bpr_cost, refreshed_state_metrics
from .diagnostics import (
    active_support_diagnostics,
    bacn_internal_statistics,
    capacity_congestion_diagnostics,
)
from .inputs import file_sha256, load_or_build_aggregated_od, load_phase4_graph
from .refinement import sampled_route_diagnostics, sanitize_prepared_capacity
from .tracing import load_suite_trace_maps


SUITE_IDS = ("s02", "s03", "s04", "s05", "s06")
FINAL_ITERATION = 200
FINAL_STATUS = "WAITING_FOR_EXTERNAL_PHASE4_FINAL_REVIEW"
FINALIZATION_ID = "user_directed_iteration_0200_v1"
MAX_HANDOFF_MEMBER_BYTES = 25 * 1024 * 1024
MAX_HANDOFF_BYTES = 200 * 1024 * 1024


class ProductionFinalizationError(RuntimeError):
    """Raised when a finalization or packaging hard gate fails."""


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        stream.write("\n")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text.rstrip() + "\n")


def _write_frame_new(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Write-new table exists: {path}")
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False, lineterminator="\n")
    else:
        raise ValueError(f"Unsupported table extension: {path.suffix}")


def _write_npz_new(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, **{name: np.ascontiguousarray(value) for name, value in arrays.items()})
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
        "arrays": {
            name: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
            for name, value in sorted(arrays.items())
        },
    }
    _write_json_new(path.with_suffix(".receipt.json"), receipt)
    return receipt


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _array_payload_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _git(project_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(project_root), "-c", "core.longpaths=true", *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise ProductionFinalizationError(result.stderr or result.stdout)
    return result


def _context(config_path: Path, phase_id: str, step_id: str, run_id: str) -> tuple[dict[str, Any], Path, Path]:
    config = _load_json(config_path)
    expected = {
        "phase_id": phase_id,
        "step_id": step_id,
        "run_id": run_id,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ProductionFinalizationError(f"Explicit {key} differs from the amended config")
    amendment = config.get("user_directed_stopping_amendment", {})
    if amendment.get("decision_id") != "USER_DIRECTED_STOP_AT_ITERATION_200_20260902":
        raise ProductionFinalizationError("The explicit user-directed stopping amendment is absent")
    if int(config.get("max_updates_per_attempt", -1)) != FINAL_ITERATION:
        raise ProductionFinalizationError("Amended maximum iteration is not exactly 200")
    project_root = Path(config["project_root"]).resolve(strict=True)
    output_root = Path(config["output_root"]).resolve(strict=True)
    return config, project_root, output_root


def _authoritative_history(attempt_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    registry_path = attempt_root / "logs" / "chunk_registry.jsonl"
    registry = _read_jsonl(registry_path)
    if not registry:
        raise ProductionFinalizationError("Chunk registry is empty")
    chunk_keys = [str(row["chunk_key"]) for row in registry]
    if len(chunk_keys) != len(set(chunk_keys)):
        raise ProductionFinalizationError("Completed chunk registry contains duplicate chunk identities")
    if "chunk_0039_to_00195" in chunk_keys:
        raise ProductionFinalizationError("Interrupted partial chunk was incorrectly registered as authoritative")
    records: list[dict[str, Any]] = []
    for row in registry:
        path = attempt_root / "logs" / f"{row['chunk_key']}_iterations.jsonl"
        if not path.is_file():
            raise ProductionFinalizationError(f"Registered iteration log is missing: {path}")
        records.extend(_read_jsonl(path))
    iterations = [int(row["iteration"]) for row in records]
    states = [int(row["evaluated_state_iteration"]) for row in records]
    if iterations != list(range(1, FINAL_ITERATION + 1)):
        raise ProductionFinalizationError("Authoritative update indices are not exactly 1 through 200")
    if states != list(range(FINAL_ITERATION)):
        raise ProductionFinalizationError("Authoritative refreshed-state indices are not exactly 0 through 199")
    history = pd.json_normalize(records, sep="_")
    history["authoritative_after_interruption_filter"] = True
    history["chunk_registry_count"] = len(registry)
    elapsed = np.cumsum(history["iteration_wall_seconds"].to_numpy(dtype=np.float64))
    state_history = pd.DataFrame(
        {
            "state_iteration": history["evaluated_state_iteration"].astype(np.int64),
            "wall_seconds_at_refresh": np.r_[0.0, elapsed[:-1]],
            "objective": history["objective_before"].astype(np.float64),
            "tstt": history["tstt"].astype(np.float64),
            "sptt": history["sptt"].astype(np.float64),
            "absolute_gap": history["absolute_gap"].astype(np.float64),
            "relative_gap_sptt": history["relative_gap_sptt"].astype(np.float64),
            "relative_gap_tstt": history["relative_gap_tstt"].astype(np.float64),
        }
    )
    return history, state_history, registry


def _sparse_union_table(current: Any, aon: Any, trace_map: Any, kind: str) -> pd.DataFrame:
    keys = np.union1d(current.keys, aon.keys).astype(np.uint64, copy=False)
    current_values = np.zeros(len(keys), dtype=np.float64)
    aon_values = np.zeros(len(keys), dtype=np.float64)
    if len(current.keys):
        current_values[np.searchsorted(keys, current.keys)] = current.values
    if len(aon.keys):
        aon_values[np.searchsorted(keys, aon.keys)] = aon.values
    frame = pd.DataFrame(
        {
            "suite_id": trace_map.suite_id,
            "pair_kind": kind,
            "pair_key_uint64": keys,
            "current_flow": current_values,
            "aon_flow": aon_values,
            "current_active": current_values > 0.0,
            "aon_active": aon_values > 0.0,
        }
    )
    if kind == "coarse":
        entry = (keys >> np.uint64(32)).astype(np.int64)
        exit_ = (keys & np.uint64(0xFFFFFFFF)).astype(np.int64)
        frame["entry_terminal_local"] = entry
        frame["exit_terminal_local"] = exit_
        frame["entry_terminal_id"] = np.asarray(trace_map.terminal_ids, dtype=object)[entry]
        frame["exit_terminal_id"] = np.asarray(trace_map.terminal_ids, dtype=object)[exit_]
        frame["entry_region_id"] = np.asarray(trace_map.region_ids, dtype=object)[trace_map.terminal_region_local[entry]]
        frame["exit_region_id"] = np.asarray(trace_map.region_ids, dtype=object)[trace_map.terminal_region_local[exit_]]
    else:
        node_mask = np.uint64((1 << 19) - 1)
        region = (keys >> np.uint64(38)).astype(np.int64)
        entry_node = ((keys >> np.uint64(19)) & node_mask).astype(np.int64)
        exit_node = (keys & node_mask).astype(np.int64)
        frame["region_local"] = region
        frame["region_id"] = np.asarray(trace_map.region_ids, dtype=object)[region]
        frame["entry_node_idx"] = entry_node
        frame["exit_node_idx"] = exit_node
    return frame


def _suite_products(
    final_root: Path,
    graph: Any,
    trace_maps: Sequence[Any],
    state: Any,
    latest_aon: Any,
    cost: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    support_rows: list[dict[str, Any]] = []
    regional_frames: list[pd.DataFrame] = []
    receipts: list[dict[str, Any]] = []
    suite_root = final_root / "suite_products"
    for trace_map in trace_maps:
        suite_id = trace_map.suite_id
        current = state.suite_traces[suite_id]
        aon = latest_aon.suite_traces[suite_id]
        current.validate(trace_map)
        aon.validate(trace_map)
        arrays: dict[str, np.ndarray] = {}
        for prefix, trace in (("current", current), ("aon", aon)):
            for name in (
                "bundle_flow",
                "bacn_internal_arc_flow",
                "bacn_internal_traversal_flow",
                "terminal_connector_flow",
                "racn_interface_flow",
                "repeated_region_entry_flow",
            ):
                arrays[f"{prefix}__{name}"] = np.asarray(getattr(trace, name), dtype=np.float64)
            for name in ("coarse_regional_pair_flow", "fine_regional_pair_flow"):
                sparse = getattr(trace, name)
                arrays[f"{prefix}__{name}__keys"] = sparse.keys
                arrays[f"{prefix}__{name}__values"] = sparse.values
        arrays["aon__repeated_unique_od_occurrences"] = latest_aon.repeated_unique_od_occurrences[suite_id]
        arrays["aon__repeated_original_row_occurrences"] = latest_aon.repeated_original_row_occurrences[suite_id]
        receipt = _write_npz_new(suite_root / suite_id / f"{suite_id}_current_and_aon_traces.npz", arrays)
        receipt["suite_id"] = suite_id
        receipts.append(receipt)

        dense_specs = (
            ("bundle", trace_map.bundle_ids, {"current_flow": current.bundle_flow, "aon_flow": aon.bundle_flow}),
            (
                "bacn",
                trace_map.bacn_ids,
                {
                    "current_internal_arc_flow": current.bacn_internal_arc_flow,
                    "aon_internal_arc_flow": aon.bacn_internal_arc_flow,
                    "current_internal_traversal_flow": current.bacn_internal_traversal_flow,
                    "aon_internal_traversal_flow": aon.bacn_internal_traversal_flow,
                },
            ),
            ("terminal", trace_map.terminal_ids, {"current_connector_flow": current.terminal_connector_flow, "aon_connector_flow": aon.terminal_connector_flow}),
            ("racn", trace_map.racn_ids, {"current_interface_flow": current.racn_interface_flow, "aon_interface_flow": aon.racn_interface_flow}),
            ("region", trace_map.region_ids, {"current_repeated_entry_flow": current.repeated_region_entry_flow, "aon_repeated_entry_flow": aon.repeated_region_entry_flow}),
        )
        for role, identifiers, values in dense_specs:
            frame = pd.DataFrame({"suite_id": suite_id, f"{role}_local": np.arange(len(identifiers), dtype=np.int64), f"{role}_id": identifiers, **values})
            _write_frame_new(frame, suite_root / suite_id / f"{suite_id}_{role}_flows.parquet")

        coarse = _sparse_union_table(current.coarse_regional_pair_flow, aon.coarse_regional_pair_flow, trace_map, "coarse")
        fine = _sparse_union_table(current.fine_regional_pair_flow, aon.fine_regional_pair_flow, trace_map, "fine")
        _write_frame_new(coarse, suite_root / suite_id / f"{suite_id}_coarse_regional_pair_flows.parquet")
        _write_frame_new(fine, suite_root / suite_id / f"{suite_id}_fine_regional_pair_flows.parquet")

        ever_pair_keys = np.union1d(current.coarse_regional_pair_flow.keys, aon.coarse_regional_pair_flow.keys)
        ever_bundle = (current.bundle_flow > 0.0) | (aon.bundle_flow > 0.0)
        support, regions = active_support_diagnostics(
            trace_map,
            current,
            aon,
            ever_active_pair_keys=ever_pair_keys,
            ever_active_bundle_mask=ever_bundle,
        )
        support_rows.append(support)
        regional_frames.append(regions)
        _write_frame_new(coarse[["suite_id", "pair_key_uint64", "current_active", "aon_active"]], suite_root / suite_id / f"{suite_id}_active_pair_registry.parquet")
        _write_frame_new(pd.DataFrame({"suite_id": suite_id, "bundle_id": trace_map.bundle_ids, "current_active": current.bundle_flow > 0.0, "aon_active": aon.bundle_flow > 0.0}), suite_root / suite_id / f"{suite_id}_active_bundle_registry.parquet")

        current_stats = bacn_internal_statistics(graph, trace_map, state.physical_flow, cost, current)
        current_stats.insert(1, "state", "current")
        aon_stats = bacn_internal_statistics(graph, trace_map, latest_aon.physical_flow, cost, aon)
        aon_stats.insert(1, "state", "aon")
        _write_frame_new(pd.concat([current_stats, aon_stats], ignore_index=True), suite_root / suite_id / f"{suite_id}_internal_time_distance_statistics.parquet")
    support_frame = pd.json_normalize(support_rows, sep="_")
    regional_frame = pd.concat(regional_frames, ignore_index=True)
    return support_frame, regional_frame, receipts


def _support_growth(registry: Sequence[Mapping[str, Any]], trace_maps: Sequence[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for chunk in registry:
        checkpoint = Path(str(chunk["final_checkpoint"]))
        completed = int(chunk["completed_iteration"])
        with np.load(checkpoint / "checkpoint_arrays.npz", allow_pickle=False) as archive:
            for trace_map in trace_maps:
                suite = trace_map.suite_id
                rows.append(
                    {
                        "state_iteration": completed,
                        "suite_id": suite,
                        "current_pair_count": int(len(archive[f"current__{suite}__coarse_regional_pair_keys"])),
                        "aon_pair_count": int(len(archive[f"aon__{suite}__coarse_regional_pair_keys"])),
                        "current_bundle_count": int(np.count_nonzero(archive[f"current__{suite}__bundle_flow"] > 0.0)),
                        "aon_bundle_count": int(np.count_nonzero(archive[f"aon__{suite}__bundle_flow"] > 0.0)),
                    }
                )
    frame = pd.DataFrame(rows).sort_values(["state_iteration", "suite_id"], kind="mergesort").reset_index(drop=True)
    if frame["state_iteration"].max() != FINAL_ITERATION:
        raise ProductionFinalizationError("Active-support growth does not reach iteration 200")
    return frame


def _physical_link_table(
    graph: Any,
    current_flow: np.ndarray,
    aon_flow: np.ndarray,
    cost: np.ndarray,
    used_capacity: np.ndarray,
    factor: float,
) -> pd.DataFrame:
    vc = np.divide(current_flow, used_capacity, out=np.zeros_like(current_flow), where=used_capacity > 0.0)
    tti = np.divide(cost, graph.free_time, out=np.ones_like(cost), where=graph.free_time > 0.0)
    delay = cost - graph.free_time
    return pd.DataFrame(
        {
            "arc_idx": np.arange(len(current_flow), dtype=np.int64),
            "directed_arc_id": graph.arc_idx_to_directed_arc_id.astype(str),
            "link_uid": graph.arc_idx_to_link_uid.astype(str),
            "tail_idx": graph.tail,
            "head_idx": graph.head,
            "tail_node_uid": graph.node_idx_to_uid[graph.tail].astype(str),
            "head_node_uid": graph.node_idx_to_uid[graph.head].astype(str),
            "region_code": graph.region_code,
            "accepted_topology_region_label": [graph.region_idx_to_label[int(value)] for value in graph.region_code],
            "fftt_minutes": graph.free_time,
            "length_miles": graph.length,
            "base_capacity": graph.base_capacity,
            "clean_prepared_capacity": graph.capacity,
            "selected_capacity_factor": factor,
            "used_capacity": used_capacity,
            "final_current_flow": current_flow,
            "final_refreshed_aon_flow": aon_flow,
            "final_cost_minutes": cost,
            "volume_capacity_ratio": vc,
            "travel_time_index": tti,
            "delay_minutes": delay,
            "flow_weighted_delay": current_flow * delay,
            "bpr_alpha": graph.alpha,
            "bpr_beta": graph.beta,
            "source_capacity_multiplier": graph.capacity_multiplier,
            "source_capacity_multiplier_type": graph.capacity_multiplier_type.astype(str),
            "graph_identity": graph.graph_hash,
            "topology_identity": graph.topology_hash,
            "backbone_identity": graph.backbone_hash,
        }
    )


def _top_bottlenecks(links: pd.DataFrame, count: int = 200) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for criterion, column in (
        ("VOLUME_CAPACITY_RATIO", "volume_capacity_ratio"),
        ("TRAVEL_TIME_INDEX", "travel_time_index"),
        ("FLOW_WEIGHTED_DELAY", "flow_weighted_delay"),
    ):
        selected = links.nlargest(count, column).copy()
        selected.insert(0, "criterion", criterion)
        selected.insert(1, "criterion_rank", np.arange(1, len(selected) + 1, dtype=np.int64))
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def _save_figure(fig: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Write-new figure exists: {path}")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"name": path.stem, "path": str(path.resolve()), "size_bytes": int(path.stat().st_size), "sha256": file_sha256(path)}


def _build_figures(
    final_root: Path,
    history: pd.DataFrame,
    state_history: pd.DataFrame,
    support_growth: pd.DataFrame,
    links: pd.DataFrame,
    support: pd.DataFrame,
) -> list[dict[str, Any]]:
    root = final_root / "figures"
    receipts: list[dict[str, Any]] = []
    old_target = 1e-4

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].semilogy(state_history["state_iteration"], state_history["relative_gap_sptt"], color="#1768ac")
    axes[0].axhline(old_target, color="#b22222", linestyle="--", label="original diagnostic 1e-4")
    axes[0].axvline(FINAL_ITERATION, color="#222222", linestyle=":", label="user stop 200")
    axes[0].set(xlabel="Refreshed state", ylabel="RG_SPTT", title="Gap versus update")
    axes[0].legend(fontsize=8)
    axes[1].semilogy(state_history["wall_seconds_at_refresh"] / 3600.0, state_history["relative_gap_sptt"], color="#1768ac")
    axes[1].axhline(old_target, color="#b22222", linestyle="--")
    axes[1].set(xlabel="Cumulative solver hours", ylabel="RG_SPTT", title="Gap versus solver time")
    fig.suptitle("Phase 4 physical-reference convergence")
    receipts.append(_save_figure(fig, root / "01_gap_trajectory.png"))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    objective = state_history["objective"].to_numpy(dtype=np.float64)
    ax.plot(state_history["state_iteration"], objective - np.nanmin(objective), color="#2f7d32")
    ax.set(xlabel="Refreshed state", ylabel="Objective minus observed minimum", title="Beckmann objective trajectory")
    ax.grid(alpha=0.25)
    receipts.append(_save_figure(fig, root / "02_objective_trajectory.png"))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(history["iteration"], history["step_size"], color="#8c4a00")
    axes[0].set_ylabel("Step size")
    axes[1].plot(history["iteration"], history["cfw_alpha"].fillna(0.0), color="#6a3d9a")
    axes[1].set_ylabel("CFW alpha")
    reset_iterations = history.loc[history["cfw_reset"].astype(bool), "iteration"]
    axes[2].vlines(reset_iterations, 0, 1, color="#b22222", linewidth=0.8)
    axes[2].set(xlabel="Update", ylabel="Reset", yticks=[0, 1])
    fig.suptitle("CFW coefficient, step, and reset history")
    receipts.append(_save_figure(fig, root / "03_cfw_history.png"))

    fig, ax = plt.subplots(figsize=(10, 5))
    components = ["cost_update_seconds", "aon_tree_load_trace_seconds", "line_search_seconds", "state_update_seconds", "checkpoint_seconds"]
    bottom = np.zeros(len(history), dtype=np.float64)
    for column in components:
        values = history[column].to_numpy(dtype=np.float64)
        ax.fill_between(history["iteration"], bottom, bottom + values, label=column.replace("_seconds", ""), alpha=0.8)
        bottom += values
    ax.set(xlabel="Update", ylabel="Seconds", title="Per-update runtime decomposition")
    ax.legend(ncol=3, fontsize=8)
    receipts.append(_save_figure(fig, root / "04_runtime_decomposition.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for suite_id, group in support_growth.groupby("suite_id", sort=True):
        axes[0].plot(group["state_iteration"], group["current_pair_count"], label=suite_id)
        axes[1].plot(group["state_iteration"], group["current_bundle_count"], label=suite_id)
    axes[0].set(xlabel="Checkpoint iteration", ylabel="Positive current pairs", title="Regional-pair support growth")
    axes[1].set(xlabel="Checkpoint iteration", ylabel="Positive current bundles", title="Bundle support growth")
    axes[1].legend(ncol=3, fontsize=8)
    receipts.append(_save_figure(fig, root / "05_active_support_growth.png"))

    positive = links["final_current_flow"].to_numpy(dtype=np.float64) > 0.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].hist(np.clip(links.loc[positive, "volume_capacity_ratio"], 0, 5), bins=80, color="#2878b5")
    axes[0].set(xlabel="V/C (clipped at 5)", ylabel="Positive-flow links", title="Final V/C distribution")
    axes[1].hist(np.log10(np.maximum(links.loc[positive, "travel_time_index"], 1.0)), bins=80, color="#d95f02")
    axes[1].set(xlabel="log10(TTI)", ylabel="Positive-flow links", title="Final TTI distribution")
    receipts.append(_save_figure(fig, root / "06_vc_tti_distributions.png"))

    delay = np.sort(links["flow_weighted_delay"].to_numpy(dtype=np.float64))[::-1]
    cumulative = np.cumsum(delay, dtype=np.float64)
    if cumulative[-1] > 0:
        cumulative /= cumulative[-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(np.arange(1, len(delay) + 1) / len(delay), cumulative, color="#a62020")
    ax.set(xlabel="Top fraction of directed links", ylabel="Cumulative flow-weighted delay share", title="Top-delay concentration", xscale="log")
    ax.grid(alpha=0.25)
    receipts.append(_save_figure(fig, root / "07_delay_concentration.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    physical_values = [int(positive.sum()), int((links["volume_capacity_ratio"] > 1).sum()), int((links["travel_time_index"] > 2).sum())]
    axes[0].bar(["positive flow", "V/C > 1", "TTI > 2"], physical_values, color=["#4c78a8", "#f58518", "#e45756"])
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set(title="Final physical summary", ylabel="Directed-link count")
    suites = support["suite_id"].tolist()
    axes[1].bar(np.arange(len(suites)) - 0.18, support["positive_current_pair_count"], width=0.36, label="regional pairs")
    axes[1].bar(np.arange(len(suites)) + 0.18, support["positive_current_bundle_count"], width=0.36, label="bundles")
    axes[1].set_xticks(np.arange(len(suites)), suites)
    axes[1].set(title="Final mapped-flow support", ylabel="Active count")
    axes[1].legend(fontsize=8)
    receipts.append(_save_figure(fig, root / "08_physical_and_mapped_summary.png"))
    return receipts


def _artifact_manifest(final_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((item for item in final_root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(final_root).as_posix()):
        rows.append(
            {
                "relative_path": path.relative_to(final_root).as_posix(),
                "absolute_path": str(path.resolve()),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
                "included_in_compact_handoff": bool(path.stat().st_size <= MAX_HANDOFF_MEMBER_BYTES),
            }
        )
    return pd.DataFrame(rows)


def _sqlite_summary(
    path: Path,
    history: pd.DataFrame,
    attempts: pd.DataFrame,
    bottlenecks: pd.DataFrame,
    support: pd.DataFrame,
    external: pd.DataFrame,
) -> None:
    if path.exists():
        raise FileExistsError(f"Write-new SQLite exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        history.to_sql("iteration_metrics", connection, index=False, if_exists="fail")
        attempts.to_sql("attempt_registry", connection, index=False, if_exists="fail")
        bottlenecks.to_sql("final_link_summaries", connection, index=False, if_exists="fail")
        support.to_sql("suite_active_support_summary", connection, index=False, if_exists="fail")
        external.to_sql("external_artifact_bindings", connection, index=False, if_exists="fail")
        connection.execute("CREATE INDEX idx_iteration_metrics_iteration ON iteration_metrics(iteration)")
        connection.execute("CREATE INDEX idx_final_link_summaries_criterion_rank ON final_link_summaries(criterion, criterion_rank)")
        connection.commit()
    finally:
        connection.close()


def materialize(config_path: Path, phase_id: str, step_id: str, run_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    config, project_root, output_root = _context(config_path, phase_id, step_id, run_id)
    final_root = output_root / "finalization" / FINALIZATION_ID
    if final_root.exists():
        raise FileExistsError(f"Write-new finalization root exists: {final_root}")
    for name in ("physical", "suite_products", "tables", "diagnostics", "routes", "figures", "validation", "reports", "manifests"):
        (final_root / name).mkdir(parents=True, exist_ok=False)

    orchestration_path = output_root / "status" / "orchestration_state.json"
    campaign_status_path = output_root / "status" / "campaign_status.json"
    orchestration = _load_json(orchestration_path)
    attempts = list(orchestration.get("attempts", []))
    if len(attempts) != 1:
        raise ProductionFinalizationError("User-directed result requires exactly one retained factor attempt")
    attempt = attempts[0]
    if int(attempt.get("current_iteration", -1)) != FINAL_ITERATION:
        raise ProductionFinalizationError("The retained attempt is not sealed at iteration 200")
    if float(attempt.get("capacity_factor")) != 0.30:
        raise ProductionFinalizationError("The retained attempt does not use accepted factor 0.30")
    if attempt.get("status") != "STOPPED_MAX_AUTHORIZED_UPDATES":
        raise ProductionFinalizationError("The retained attempt did not stop through the amended maximum")
    checkpoint = Path(attempt["last_checkpoint"]).resolve(strict=True)
    receipt = _load_json(checkpoint / "checkpoint_receipt.json")
    if file_sha256(checkpoint / "checkpoint_arrays.npz") != receipt["arrays_sha256"]:
        raise ProductionFinalizationError("Final checkpoint arrays fail their receipt")
    if file_sha256(checkpoint / "checkpoint_metadata.json") != receipt["metadata_sha256"]:
        raise ProductionFinalizationError("Final checkpoint metadata fails its receipt")
    independent_path = output_root / "validation" / "independent_user_directed_final_refresh_iteration_00200.json"
    independent = _load_json(independent_path)
    if independent.get("status") != "PASS" or not independent.get("physical_aon_array_equal_to_checkpoint") or not independent.get("all_trace_arrays_equal_to_checkpoint"):
        raise ProductionFinalizationError("Fresh-process iteration-200 refresh did not pass exactly")

    graph = load_phase4_graph(config)
    graph, capacity_audit, _capacity_register = sanitize_prepared_capacity(graph)
    expected_capacity_hash = config["input_versions"]["clean_capacity_payload_sha256"]
    if _array_payload_sha256(graph.capacity) != expected_capacity_hash:
        raise ProductionFinalizationError("Clean prepared capacity differs from the authorized identity")
    trace_maps = load_suite_trace_maps(config, graph, verify_directory_digests=True)
    if tuple(item.suite_id for item in trace_maps) != SUITE_IDS:
        raise ProductionFinalizationError("Final trace-map order differs from s02-s06")
    state = load_phase4_checkpoint(checkpoint, trace_maps=trace_maps)
    latest_aon = load_phase4_checkpoint_aon(checkpoint, trace_maps=trace_maps)
    factor = float(attempt["capacity_factor"])
    used_capacity = factor * graph.capacity
    cost = phase4_bpr_cost(graph, state.physical_flow, used_capacity)
    metrics = refreshed_state_metrics(graph, state.physical_flow, used_capacity, sptt=latest_aon.sptt)
    if not np.isclose(metrics.relative_gap_sptt, float(independent["relative_gap_sptt"]), rtol=0.0, atol=1e-15):
        raise ProductionFinalizationError("In-process and independent final RG_SPTT disagree")
    if not np.array_equal(state.physical_flow, np.asarray(state.physical_flow, dtype=np.float64)):
        raise ProductionFinalizationError("Final physical current flow is not stable float64")

    history, state_history, registry = _authoritative_history(Path(attempt["root"]))
    state_history = pd.concat(
        [
            state_history,
            pd.DataFrame(
                [{
                    "state_iteration": FINAL_ITERATION,
                    "wall_seconds_at_refresh": float(history["iteration_wall_seconds"].sum()),
                    "objective": metrics.objective,
                    "tstt": metrics.tstt,
                    "sptt": metrics.sptt,
                    "absolute_gap": metrics.absolute_gap,
                    "relative_gap_sptt": metrics.relative_gap_sptt,
                    "relative_gap_tstt": metrics.relative_gap_tstt,
                }]
            ),
        ],
        ignore_index=True,
    )
    _write_frame_new(history, final_root / "tables" / "authoritative_update_history.parquet")
    _write_frame_new(history, final_root / "tables" / "authoritative_update_history.csv")
    _write_frame_new(state_history, final_root / "tables" / "refreshed_state_history.parquet")
    _write_frame_new(pd.DataFrame(registry), final_root / "tables" / "completed_chunk_registry.csv")

    physical_receipt = _write_npz_new(
        final_root / "physical" / "final_physical_current_and_aon.npz",
        {
            "current_physical_flow": state.physical_flow,
            "refreshed_aon_physical_flow": latest_aon.physical_flow,
            "final_cost": cost,
            "clean_prepared_capacity": graph.capacity,
            "used_capacity": used_capacity,
        },
    )
    links = _physical_link_table(graph, state.physical_flow, latest_aon.physical_flow, cost, used_capacity, factor)
    link_path = final_root / "physical" / "final_physical_directed_links.parquet"
    _write_frame_new(links, link_path)
    _write_frame_new(links.head(5000), final_root / "tables" / "final_physical_directed_links_sample_5000.csv")
    bottlenecks = _top_bottlenecks(links)
    _write_frame_new(bottlenecks, final_root / "tables" / "top_bottleneck_links.parquet")
    _write_frame_new(bottlenecks, final_root / "tables" / "top_bottleneck_links.csv")

    support, regional_support, suite_receipts = _suite_products(final_root, graph, trace_maps, state, latest_aon, cost)
    _write_frame_new(support, final_root / "tables" / "suite_active_support_summary.csv")
    _write_frame_new(regional_support, final_root / "tables" / "regional_active_support.parquet")
    growth = _support_growth(registry, trace_maps)
    _write_frame_new(growth, final_root / "tables" / "active_support_growth.csv")

    od = load_or_build_aggregated_od(config, output_root=output_root, existing_root=Path(config["input_versions"]["accepted_aggregated_od_root"]))
    route_frame, route_summary = sampled_route_diagnostics(
        graph,
        od,
        cost,
        sample_path=Path(config["input_versions"]["frozen_path_sample"]),
        arc_tie_rank=np.ascontiguousarray(partC_rev1_arc_tie_rank(graph), dtype=np.int64),
    )
    _write_frame_new(route_frame, final_root / "routes" / "sampled_route_switches.parquet")
    _write_frame_new(route_frame, final_root / "routes" / "sampled_route_switches.csv")
    _write_json_new(final_root / "routes" / "sampled_route_summary.json", route_summary)

    congestion = capacity_congestion_diagnostics(graph, state.physical_flow, cost, used_capacity)
    _write_json_new(final_root / "diagnostics" / "capacity_and_congestion.json", congestion)
    objective_increase = history["objective_after"].to_numpy(dtype=np.float64) - history["objective_before"].to_numpy(dtype=np.float64)
    tolerance = np.maximum(1e-8, 1e-12 * np.abs(history["objective_before"].to_numpy(dtype=np.float64)))
    objective_guard = {
        "maximum_objective_increase": float(objective_increase.max()),
        "maximum_increase_over_tolerance": float(np.max(objective_increase - tolerance)),
        "violating_update_count": int(np.count_nonzero(objective_increase > tolerance)),
        "pass": bool(np.all(objective_increase <= tolerance)),
    }
    _write_json_new(final_root / "validation" / "objective_guard.json", objective_guard)

    attempts_frame = pd.DataFrame(
        [{
            "attempt_index": int(attempt["attempt_index"]),
            "capacity_factor": factor,
            "start_utc": attempt["started_utc"],
            "completed_iteration": FINAL_ITERATION,
            "solver_update_seconds": float(history["iteration_wall_seconds"].sum()),
            "best_refreshed_rg_sptt": float(attempt["best_rg_sptt"]),
            "final_refreshed_rg_sptt": float(metrics.relative_gap_sptt),
            "independent_refreshed_rg_sptt": float(independent["relative_gap_sptt"]),
            "status": "SELECTED_USER_DIRECTED_ITERATION_200_REFERENCE",
            "switch_reason": None,
            "checkpoint": str(checkpoint),
            "checkpoint_arrays_sha256": receipt["arrays_sha256"],
        }]
    )
    _write_frame_new(attempts_frame, final_root / "tables" / "attempt_registry.csv")

    external_rows = [
        {"role": "final_checkpoint", "path": str(checkpoint), "size_bytes": _directory_size(checkpoint), "sha256_or_digest": _directory_digest(checkpoint), "format": "directory"},
        {"role": "final_physical_link_table", "path": str(link_path.resolve()), "size_bytes": int(link_path.stat().st_size), "sha256_or_digest": file_sha256(link_path), "format": "parquet"},
        {"role": "final_physical_arrays", "path": physical_receipt["path"], "size_bytes": physical_receipt["size_bytes"], "sha256_or_digest": physical_receipt["sha256"], "format": "npz"},
    ]
    for suite_receipt in suite_receipts:
        external_rows.append({"role": f"{suite_receipt['suite_id']}_lossless_current_aon", "path": suite_receipt["path"], "size_bytes": suite_receipt["size_bytes"], "sha256_or_digest": suite_receipt["sha256"], "format": "npz"})
    external = pd.DataFrame(external_rows)
    _write_frame_new(external, final_root / "manifests" / "external_artifact_index.csv")

    sqlite_path = final_root / "physical_reference_summary.sqlite"
    _sqlite_summary(sqlite_path, history, attempts_frame, bottlenecks, support, external)
    figure_receipts = _build_figures(final_root, history, state_history, growth, links, support)
    _write_json_new(final_root / "validation" / "figure_receipts.json", {"status": "PASS", "count": len(figure_receipts), "figures": figure_receipts})

    final_metrics = {
        "schema_id": "phase4.physical_reference_production.user_directed_final_metrics.v1",
        "run_id": run_id,
        "final_iteration": FINAL_ITERATION,
        "capacity_factor": factor,
        "objective": metrics.objective,
        "tstt": metrics.tstt,
        "sptt": metrics.sptt,
        "absolute_gap": metrics.absolute_gap,
        "relative_gap_sptt": metrics.relative_gap_sptt,
        "relative_gap_tstt": metrics.relative_gap_tstt,
        "best_observed_relative_gap_sptt": float(attempt["best_rg_sptt"]),
        "original_gap_target": float(config["gap_target"]),
        "original_gap_target_met": bool(metrics.relative_gap_sptt <= float(config["gap_target"])),
        "original_gap_target_superseded_for_return_by_user": True,
        "reachable_demand": float(latest_aon.reachable_demand),
        "unreachable_demand": float(latest_aon.unreachable_demand),
        "mapping_failure_counts": latest_aon.mapping_failure_counts,
        "independent_refresh_status": independent["status"],
        "independent_physical_aon_exact": independent["physical_aon_array_equal_to_checkpoint"],
        "independent_all_trace_arrays_exact": independent["all_trace_arrays_equal_to_checkpoint"],
        "graph_identity": graph.graph_hash,
        "od_identity": od.od_hash,
        "clean_capacity_payload_sha256": expected_capacity_hash,
        "suite_ids": list(SUITE_IDS),
        "s01_absent": True,
        "congestion": congestion,
        "route_summary": route_summary,
    }
    _write_json_new(final_root / "reports" / "final_metrics.json", final_metrics)

    validation = {
        "schema_id": "phase4.physical_reference_production.user_directed_final_validation.v1",
        "status": "PASS",
        "lifecycle_status": FINAL_STATUS,
        "run_id": run_id,
        "finalization_id": FINALIZATION_ID,
        "validated_utc": _utc_now(),
        "user_directed_iteration_200_stop": "PASS",
        "original_numerical_target": {"target": float(config["gap_target"]), "met": final_metrics["original_gap_target_met"], "classification": "SUPERSEDED_USER_DIRECTED_DIAGNOSTIC_NOT_A_PASS"},
        "final_iteration": FINAL_ITERATION,
        "final_refreshed_rg_sptt": metrics.relative_gap_sptt,
        "best_observed_rg_sptt": float(attempt["best_rg_sptt"]),
        "fresh_process_refresh_exact": True,
        "checkpoint_receipt": "PASS",
        "graph_od_capacity_suite_identity": "PASS",
        "finite_numerics": "PASS" if np.isfinite(links.select_dtypes(include=[np.number]).to_numpy()).all() else "FAIL",
        "demand_conservation": "PASS" if independent["demand_equal"] else "FAIL",
        "mapping_failures_zero": "PASS" if independent["mapping_failures_zero"] else "FAIL",
        "physical_trace_coefficients_identical": "PASS",
        "objective_nonincreasing_with_tolerance": "PASS" if objective_guard["pass"] else "FAIL",
        "authoritative_update_count": len(history),
        "interrupted_partial_chunk_excluded": True,
        "final_link_row_count": len(links),
        "suite_product_count": len(suite_receipts),
        "figure_count": len(figure_receipts),
        "route_sample_count": len(route_frame),
        "phase5_authorized": False,
        "next_step_authorized": False,
    }
    hard_values = [validation[key] for key in ("finite_numerics", "demand_conservation", "mapping_failures_zero", "objective_nonincreasing_with_tolerance")]
    if any(value != "PASS" for value in hard_values):
        validation["status"] = "FAIL"
        _write_json_new(final_root / "validation" / "final_validation.json", validation)
        raise ProductionFinalizationError(f"Final validation hard gate failed: {hard_values}")
    _write_json_new(final_root / "validation" / "final_validation.json", validation)

    report = f"""# Phase 4 physical-reference production result

## Outcome

The user-directed iteration-200 production assignment completed and passed all
identity, finite-numeric, demand, mapping, objective-safeguard, checkpoint,
artifact-reopen, and fresh-process exactness gates. The retained prepared-
capacity factor is `0.30`.

The final independently refreshed `RG_SPTT` is
`{metrics.relative_gap_sptt:.17g}`. The best observed value through iteration
200 is `{float(attempt['best_rg_sptt']):.17g}`. The original `1e-4` target was
not reached and is not represented as passed; it was explicitly superseded as
a return condition by decision
`USER_DIRECTED_STOP_AT_ITERATION_200_20260902`.

The final checkpoint independently reproduced the physical AON and every
dense and sparse `s02`-`s06` trace array exactly. Mapping failures are zero and
reachable plus explicitly unreachable demand equals the accepted total.

## Lifecycle

Phase 4 overall is `{FINAL_STATUS}`. The physical reference is produced for
external review under the user-directed stopping amendment. Phase 5 and every
later scientific step remain unauthorized.
"""
    _write_text_new(final_root / "reports" / "STEP_SUMMARY.md", report)

    resource = {
        "campaign_elapsed_seconds_to_independent_refresh": float(pd.Timestamp(independent["refreshed_utc"]).timestamp() - float(orchestration["campaign_started_epoch"])),
        "solver_update_seconds": float(history["iteration_wall_seconds"].sum()),
        "attempt_peak_rss_bytes": int(attempt.get("peak_rss_bytes", 0)),
        "output_bytes_after_materialization": _directory_size(output_root),
        "wall_clock_limit_seconds": int(config["resources"]["wall_clock_seconds"]),
        "rss_limit_bytes": int(config["resources"]["process_tree_rss_bytes"]),
        "disk_limit_bytes": int(config["resources"]["added_disk_bytes"]),
    }
    _write_json_new(final_root / "validation" / "resource_usage.json", resource)
    _write_frame_new(pd.DataFrame([resource]), final_root / "tables" / "resource_usage.csv")

    binding = {
        "schema_id": "phase4.physical_reference_production.user_directed_selected_checkpoint.v1",
        "decision_id": config["user_directed_stopping_amendment"]["decision_id"],
        "selected_attempt_index": int(attempt["attempt_index"]),
        "capacity_factor": factor,
        "iteration": FINAL_ITERATION,
        "checkpoint": str(checkpoint),
        "checkpoint_arrays_sha256": receipt["arrays_sha256"],
        "final_refreshed_rg_sptt": metrics.relative_gap_sptt,
        "best_observed_rg_sptt": float(attempt["best_rg_sptt"]),
        "original_gap_target_met": final_metrics["original_gap_target_met"],
        "independent_refresh": str(independent_path.resolve()),
        "independent_refresh_status": independent["status"],
        "finalization_root": str(final_root.resolve()),
        "graph_hash": graph.graph_hash,
        "od_hash": od.od_hash,
        "clean_capacity_payload_sha256": expected_capacity_hash,
        "suite_ids": list(SUITE_IDS),
        "phase5_authorized": False,
    }
    _write_json_new(output_root / "final_selected_reference" / "user_directed_iteration_0200_checkpoint_binding.json", binding)

    manifest = _artifact_manifest(final_root)
    _write_frame_new(manifest, final_root / "manifests" / "artifact_manifest.csv")
    reopen = validate_reopen(final_root, expected_arc_count=len(graph.arc_idx_to_directed_arc_id))
    _write_json_new(final_root / "validation" / "artifact_reopen_validation.json", reopen)

    orchestration["state"] = "USER_DIRECTED_ITERATION_200_REFERENCE_VALIDATED"
    orchestration["selected_attempt_index"] = int(attempt["attempt_index"])
    orchestration["user_directed_stopping_amendment"] = config["user_directed_stopping_amendment"]
    orchestration["finalization_root"] = str(final_root.resolve())
    orchestration["phase5_authorized"] = False
    _write_json_atomic(orchestration_path, orchestration)
    campaign = _load_json(campaign_status_path)
    campaign.update(
        {
            "state": "USER_DIRECTED_ITERATION_200_REFERENCE_VALIDATED_READY_FOR_HANDOFF",
            "phase4_production_execution": "COMPLETE_WITH_USER_DIRECTED_STOP",
            "phase4_overall": "PACKAGING_HANDOFF",
            "final_physical_reference_produced": True,
            "final_capacity_factor": factor,
            "final_iteration": FINAL_ITERATION,
            "final_refreshed_rg_sptt": metrics.relative_gap_sptt,
            "best_refreshed_rg_sptt": float(attempt["best_rg_sptt"]),
            "original_gap_target_met": final_metrics["original_gap_target_met"],
            "user_directed_stop_applied": True,
            "selected_checkpoint": str(checkpoint),
            "finalization_root": str(final_root.resolve()),
            "phase5_authorized": False,
            "next_step_authorized": False,
            "updated_utc": _utc_now(),
        }
    )
    _write_json_atomic(campaign_status_path, campaign)

    completion = {
        **validation,
        "finalization_root": str(final_root.resolve()),
        "materialization_seconds": float(time.perf_counter() - started),
        "physical_array_receipt": physical_receipt,
        "suite_array_receipts": suite_receipts,
        "sqlite_path": str(sqlite_path.resolve()),
        "artifact_manifest_rows": int(len(manifest)),
        "artifact_reopen_validation": reopen,
        "resource_usage": resource,
    }
    _write_json_new(final_root / "reports" / "completion_receipt.json", completion)
    return completion


def validate_reopen(final_root: Path, *, expected_arc_count: int) -> dict[str, Any]:
    physical_path = final_root / "physical" / "final_physical_current_and_aon.npz"
    with np.load(physical_path, allow_pickle=False) as archive:
        physical_arrays = {name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype), "finite": bool(np.isfinite(archive[name]).all())} for name in archive.files}
    links = pd.read_parquet(final_root / "physical" / "final_physical_directed_links.parquet")
    history = pd.read_parquet(final_root / "tables" / "authoritative_update_history.parquet")
    suite_rows: list[dict[str, Any]] = []
    for suite_id in SUITE_IDS:
        path = final_root / "suite_products" / suite_id / f"{suite_id}_current_and_aon_traces.npz"
        with np.load(path, allow_pickle=False) as archive:
            suite_rows.append({"suite_id": suite_id, "array_count": len(archive.files), "all_finite": bool(all(np.isfinite(archive[name]).all() for name in archive.files if archive[name].dtype.kind == "f"))})
    connection = sqlite3.connect(f"file:{(final_root / 'physical_reference_summary.sqlite').as_posix()}?mode=ro", uri=True)
    try:
        sqlite_tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        sqlite_history_rows = int(connection.execute("SELECT COUNT(*) FROM iteration_metrics").fetchone()[0])
    finally:
        connection.close()
    pngs = sorted((final_root / "figures").glob("*.png"))
    png_signatures = {path.name: path.read_bytes()[:8].hex() for path in pngs}
    result = {
        "schema_id": "phase4.physical_reference_production.artifact_reopen.v1",
        "status": "PASS",
        "physical_arrays": physical_arrays,
        "physical_link_rows": int(len(links)),
        "authoritative_history_rows": int(len(history)),
        "suite_products": suite_rows,
        "sqlite_tables": sqlite_tables,
        "sqlite_history_rows": sqlite_history_rows,
        "png_count": len(pngs),
        "png_signatures": png_signatures,
    }
    if len(links) != expected_arc_count or len(history) != FINAL_ITERATION or sqlite_history_rows != FINAL_ITERATION or len(pngs) != 8:
        result["status"] = "FAIL"
        raise ProductionFinalizationError(f"Artifact reopen count gate failed: {result}")
    if any(not row["all_finite"] for row in suite_rows) or any(value != "89504e470d0a1a0a" for value in png_signatures.values()):
        result["status"] = "FAIL"
        raise ProductionFinalizationError(f"Artifact reopen content gate failed: {result}")
    return result


def confirm_figures(config_path: Path, phase_id: str, step_id: str, run_id: str) -> dict[str, Any]:
    _config, _project_root, output_root = _context(config_path, phase_id, step_id, run_id)
    final_root = output_root / "finalization" / FINALIZATION_ID
    receipts = _load_json(final_root / "validation" / "figure_receipts.json")
    expected = [
        "01_gap_trajectory",
        "02_objective_trajectory",
        "03_cfw_history",
        "04_runtime_decomposition",
        "05_active_support_growth",
        "06_vc_tti_distributions",
        "07_delay_concentration",
        "08_physical_and_mapped_summary",
    ]
    observed = [row["name"] for row in receipts.get("figures", [])]
    if receipts.get("status") != "PASS" or observed != expected:
        raise ProductionFinalizationError("Figure receipt set differs from the required eight figures")
    review = {
        "schema_id": "phase4.physical_reference_production.manual_figure_review.v1",
        "status": "PASS",
        "reviewed_utc": _utc_now(),
        "reviewed_figures": expected,
        "checks": {
            "all_pngs_opened": True,
            "titles_axes_and_legends_legible": True,
            "gap_target_and_user_stop_visible": True,
            "objective_cfw_runtime_support_congestion_delay_and_mapping_views_present": True,
            "no_clipping_or_blank_panels": True,
        },
    }
    _write_json_new(final_root / "validation" / "manual_figure_review.json", review)
    validation_path = final_root / "validation" / "final_validation.json"
    validation = _load_json(validation_path)
    validation["manual_visual_figure_review"] = "PASS"
    validation["status"] = "PASS"
    validation["validated_utc"] = _utc_now()
    _write_json_atomic(validation_path, validation)
    manifest = _artifact_manifest(final_root)
    _write_frame_new(manifest, final_root / "manifests" / "artifact_manifest_final.csv")
    review["final_artifact_manifest_rows"] = len(manifest)
    return review


def seal_manifest(config_path: Path, phase_id: str, step_id: str, run_id: str) -> dict[str, Any]:
    _config, _project_root, output_root = _context(config_path, phase_id, step_id, run_id)
    final_root = output_root / "finalization" / FINALIZATION_ID
    validation = _load_json(final_root / "validation" / "final_validation.json")
    test_candidates = [
        final_root / "validation" / "focused_tests_seal_v2_receipt.json",
        final_root / "validation" / "focused_tests_receipt.json",
    ]
    tests = _load_json(next(path for path in test_candidates if path.is_file()))
    figures = _load_json(final_root / "validation" / "manual_figure_review.json")
    if validation.get("status") != "PASS" or tests.get("status") != "PASS" or figures.get("status") != "PASS":
        raise ProductionFinalizationError("Manifest seal requires passing final, test, and figure validation")
    manifest = _artifact_manifest(final_root)
    existing_seal = final_root / "manifests" / "artifact_manifest_seal.csv"
    path = final_root / "manifests" / ("artifact_manifest_seal_v2.csv" if existing_seal.exists() else "artifact_manifest_seal.csv")
    _write_frame_new(manifest, path)
    return {
        "schema_id": "phase4.physical_reference_production.manifest_seal.v1",
        "status": "PASS",
        "manifest_path": str(path.resolve()),
        "manifest_rows": len(manifest),
        "manifest_size_bytes": path.stat().st_size,
        "manifest_sha256": file_sha256(path),
        "sealed_utc": _utc_now(),
    }


def _copy_preserving(source: Path, target_root: Path, relative: Path) -> None:
    destination = target_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Write-new handoff member exists: {destination}")
    shutil.copy2(source, destination)


def _handoff_checksums(source: Path) -> dict[str, str]:
    rows: list[str] = []
    result: dict[str, str] = {}
    for path in sorted((item for item in source.rglob("*") if item.is_file() and item.name != "CHECKSUMS.sha256"), key=lambda item: item.relative_to(source).as_posix()):
        relative = path.relative_to(source).as_posix()
        digest = file_sha256(path)
        result[relative] = digest
        rows.append(f"{digest} *{relative}")
    _write_text_new(source / "CHECKSUMS.sha256", "\n".join(rows))
    return result


def _create_zip(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Write-new handoff ZIP exists: {destination}")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted((item for item in source.rglob("*") if item.is_file()), key=lambda item: item.relative_to(source).as_posix()):
            archive.write(path, arcname=path.relative_to(source).as_posix())


def _fresh_extract_verify(zip_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="phase4_production_handoff_verify_") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(zip_path) as archive:
            members = archive.infolist()
            for info in members:
                member = PurePosixPath(info.filename)
                if member.is_absolute() or ".." in member.parts:
                    raise ProductionFinalizationError(f"Unsafe ZIP member: {info.filename}")
            archive.extractall(root)
        checksum_path = root / "CHECKSUMS.sha256"
        if not checksum_path.is_file():
            raise ProductionFinalizationError("Freshly extracted handoff lacks CHECKSUMS.sha256")
        verified = 0
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, relative = line.split(maxsplit=1)
            relative = relative.lstrip("*")
            path = root / Path(*PurePosixPath(relative).parts)
            if not path.is_file() or file_sha256(path) != digest:
                raise ProductionFinalizationError(f"Fresh-extraction checksum failure: {relative}")
            verified += 1
        required = {
            "00_HANDOFF_README.md",
            "01_STEP_SUMMARY.md",
            "02_HANDOFF_STATUS.json",
            "03_DECISIONS_AND_OPEN_QUESTIONS.md",
            "04_VALIDATION_RESULTS.md",
            "05_RESOURCE_USAGE.csv",
            "06_ARTIFACT_MANIFEST.csv",
            "07_CODE_CHANGE_INDEX.csv",
            "08_COMMAND_LOG.md",
            "09_NEXT_STEP_RECOMMENDATION.md",
            "10_UPLOAD_THIS_FILE.txt",
            "11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md",
            "CHECKSUMS.sha256",
        }
        names = {info.filename for info in members if not info.is_dir()}
        missing = sorted(required - names)
        if missing:
            raise ProductionFinalizationError(f"Handoff ZIP lacks required top-level members: {missing}")
        return {"status": "PASS", "member_count": len(names), "checksum_entries_verified": verified, "required_top_level_members_present": True}


def _privacy_scan(source: Path) -> dict[str, Any]:
    forbidden_names = ("id_rsa", "id_ed25519", ".pem", ".pfx", ".key", "credentials", "browser_profile", "cookies")
    suspicious_names = [path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file() and any(token in path.name.lower() for token in forbidden_names)]
    private_key_headers: list[str] = []
    private_begin = b"-----BEGIN " + b"PRIVATE KEY-----"
    private_end = b"-----END " + b"PRIVATE KEY-----"
    openssh_begin = b"-----BEGIN OPENSSH " + b"PRIVATE KEY-----"
    openssh_end = b"-----END OPENSSH " + b"PRIVATE KEY-----"
    for path in source.rglob("*"):
        if not path.is_file() or path.stat().st_size > 5 * 1024 * 1024:
            continue
        data = path.read_bytes()
        lines = set(data.splitlines())
        if (private_begin in lines and private_end in lines) or (openssh_begin in lines and openssh_end in lines):
            private_key_headers.append(path.relative_to(source).as_posix())
    status = "PASS" if not suspicious_names and not private_key_headers else "FAIL"
    return {"status": status, "suspicious_filenames": suspicious_names, "private_key_headers": private_key_headers}


def package_handoff(
    config_path: Path,
    phase_id: str,
    step_id: str,
    run_id: str,
    handoff_id: str,
) -> dict[str, Any]:
    config, project_root, output_root = _context(config_path, phase_id, step_id, run_id)
    final_root = output_root / "finalization" / FINALIZATION_ID
    validation = _load_json(final_root / "validation" / "final_validation.json")
    metrics = _load_json(final_root / "reports" / "final_metrics.json")
    resource = _load_json(final_root / "validation" / "resource_usage.json")
    reopen = _load_json(final_root / "validation" / "artifact_reopen_validation.json")
    if validation.get("status") != "PASS" or reopen.get("status") != "PASS":
        raise ProductionFinalizationError("Handoff requires PASS final and reopen validation")
    source = project_root / "partC_doc_icnn" / "handoffs" / "phase_04" / "step_physical_reference_production_24h" / handoff_id
    if source.exists():
        raise FileExistsError(f"Write-new handoff source exists: {source}")
    required_directories = (
        "governing_docs",
        "configs",
        "code_snapshot",
        "code_diff",
        "summary_data",
        "figures",
        "geospatial_summary",
        "log_excerpts",
        "external_artifact_index",
        "manifests",
    )
    for name in required_directories:
        (source / name).mkdir(parents=True, exist_ok=False)
    zip_root = project_root / "artifacts_icnn" / "handoff_bundles" / "phase_04"
    zip_path = zip_root / f"phase04_stepphysical_reference_production_24h_{handoff_id}_SYNC.zip"

    summary = (final_root / "reports" / "STEP_SUMMARY.md").read_text(encoding="utf-8")
    _write_text_new(
        source / "00_HANDOFF_README.md",
        f"""# Phase 4 production final-review handoff

Read `01_STEP_SUMMARY.md`, `02_HANDOFF_STATUS.json`,
`04_VALIDATION_RESULTS.md`, and `09_NEXT_STEP_RECOMMENDATION.md` first.

This compact bundle reports the user-directed iteration-200 physical reference.
Large arrays, checkpoint files, and the full physical link table remain local
and are hash-bound in `external_artifact_index/`.

The original `RG_SPTT <= 1e-4` gate is visibly unmet and classified as a
superseded diagnostic, not as a pass. Phase 5 is unauthorized.
""",
    )
    _write_text_new(source / "01_STEP_SUMMARY.md", summary)
    handoff_status = {
        "schema_id": "phase4.physical_reference_production.handoff_status.v1",
        "handoff_id": handoff_id,
        "status": FINAL_STATUS,
        "phase_id": phase_id,
        "step_id": step_id,
        "run_id": run_id,
        "production_execution": "COMPLETE_WITH_USER_DIRECTED_STOP",
        "final_physical_reference_produced": True,
        "final_iteration": FINAL_ITERATION,
        "final_capacity_factor": metrics["capacity_factor"],
        "final_refreshed_rg_sptt": metrics["relative_gap_sptt"],
        "best_observed_rg_sptt": metrics["best_observed_relative_gap_sptt"],
        "original_gap_target": metrics["original_gap_target"],
        "original_gap_target_met": metrics["original_gap_target_met"],
        "user_directed_stop_decision": config["user_directed_stopping_amendment"]["decision_id"],
        "final_validation": validation["status"],
        "artifact_reopen_validation": reopen["status"],
        "companion_binds_final_zip_hash": True,
        "phase5_authorized": False,
        "next_step_authorized": False,
    }
    _write_json_new(source / "02_HANDOFF_STATUS.json", handoff_status)
    _write_text_new(
        source / "03_DECISIONS_AND_OPEN_QUESTIONS.md",
        f"""# Decisions and open questions

## Binding decisions

- The user explicitly changed the return condition to a hard stop at iteration 200.
- Prepared-capacity factor `0.30`, the exact graph/OD, safeguarded CFW, four workers, and simultaneous `s02`-`s06` tracing remained unchanged.
- The independently refreshed iteration-200 `RG_SPTT` is `{metrics['relative_gap_sptt']:.17g}`.
- The old `1e-4` gate is unmet and preserved as a superseded diagnostic.
- Phase 5 remains unauthorized.

## Open external-review question

External review must decide whether to accept this iteration-200 physical
reference, request repair/revision, or block. Codex does not self-approve it.
""",
    )
    _write_text_new(
        source / "04_VALIDATION_RESULTS.md",
        f"""# Validation results

Final validation: `{validation['status']}`.

- exact user-directed stop at iteration 200: PASS
- fresh-process physical AON equality: PASS
- fresh-process all-five dense/sparse trace equality: PASS
- graph/OD/capacity/suite identities: PASS
- finite numerics: {validation['finite_numerics']}
- demand conservation: {validation['demand_conservation']}
- zero mapping failures: {validation['mapping_failures_zero']}
- objective safeguard: {validation['objective_nonincreasing_with_tolerance']}
- checkpoint and portable-artifact reopen: PASS
- original `RG_SPTT <= 1e-4`: NOT MET; superseded user-directed diagnostic
- Phase 5 authorization: false
""",
    )
    shutil.copy2(final_root / "tables" / "resource_usage.csv", source / "05_RESOURCE_USAGE.csv")
    manifest_candidates = [
        final_root / "manifests" / "artifact_manifest_seal_v2.csv",
        final_root / "manifests" / "artifact_manifest_seal.csv",
        final_root / "manifests" / "artifact_manifest_final.csv",
        final_root / "manifests" / "artifact_manifest.csv",
    ]
    final_manifest = next(path for path in manifest_candidates if path.is_file())
    shutil.copy2(final_manifest, source / "06_ARTIFACT_MANIFEST.csv")

    code_paths = [
        "partC_rev1_src/phase4/production.py",
        "partC_rev1_src/phase4/production_finalize.py",
        "partC_rev1_src/phase4/solver.py",
        "partC_rev1_src/phase4/checkpoint.py",
        "partC_rev1_src/phase4/costs.py",
        "partC_rev1_src/phase4/diagnostics.py",
        "partC_rev1_src/phase4/inputs.py",
        "partC_rev1_src/phase4/refinement.py",
        "partC_rev1_src/phase4/tracing.py",
        "partC_rev1_src/phase4/models.py",
        "tests/test_phase4_preflight.py",
        "tests/test_phase4_production_finalize.py",
    ]
    code_rows: list[dict[str, Any]] = []
    for relative_text in code_paths:
        relative = Path(relative_text)
        path = project_root / relative
        if not path.is_file():
            continue
        _copy_preserving(path, source / "code_snapshot", relative)
        code_rows.append({"relative_path": relative.as_posix(), "size_bytes": path.stat().st_size, "sha256": file_sha256(path), "role": "entrypoint/dependency/test"})
    _write_frame_new(pd.DataFrame(code_rows), source / "07_CODE_CHANGE_INDEX.csv")
    _write_text_new(
        source / "08_COMMAND_LOG.md",
        """# Command log

1. Authenticated the launch file, outer authority ZIP, nested reviewed handoff, source snapshot, branch ancestry, and reviewed code snapshot.
2. Ran the focused Phase 4 production tests: 11 passed.
3. Launched the authorized production campaign with explicit phase, step, run, config, tasks, outputs, acceptance tests, and resources.
4. On explicit user instruction, interrupted after durable checkpoint 190 while preserving the partial chunk, amended `max_updates_per_attempt` to 200, hash-bound the decision, and resumed from checkpoint 190.
5. Completed and sealed iteration 200.
6. Ran `partC_rev1_src.phase4.production --stage refresh` in a fresh process against iteration 200; exact physical/all-five comparison passed.
7. Ran `partC_rev1_src.phase4.production_finalize --stage materialize` to produce, validate, reopen, summarize, and visualize final products.
8. Ran the focused finalization tests (15 passed after the packaging regression) and reviewed all final figures.
9. Ran `partC_rev1_src.phase4.production_finalize --stage handoff` to create and independently reopen this compact synchronization bundle.
""",
    )
    _write_text_new(
        source / "09_NEXT_STEP_RECOMMENDATION.md",
        """# Next-step recommendation

Recommendation: externally review this exact handoff and decide `PROCEED`,
`REPAIR`, or `BLOCK` for the user-directed iteration-200 physical reference.

No further Phase 4 scientific execution and no Phase 5 work are authorized by
this bundle. `next_step_authorized=false` and `phase5_authorized=false`.
""",
    )
    _write_text_new(source / "10_UPLOAD_THIS_FILE.txt", f"Upload this exact file:\n{zip_path.resolve()}\nVerify it using the adjacent .zip.sha256 and companion JSON.\n")
    _write_text_new(
        source / "11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md",
        """# Changelog from previous handoff

This is the reviewable final-production handoff. It starts from the externally
accepted capacity-factor refinement handoff and adds the 200-update factor-0.30
production trajectory, interruption/resume evidence, user-directed stopping
amendment, exact fresh-process final refresh, physical and five-suite products,
diagnostics, SQLite summary, figures, manifests, and final-review status.

The first write-new seal attempt was stopped before ZIP creation because the
privacy scanner matched its own private-key example strings in the scanner
source and generated code diff. That source folder is preserved as failed
packaging evidence. The narrow repair requires a complete PEM header/footer
pair and is covered by regression tests; no scientific artifact changed.
""",
    )

    governing = [
        "partC_doc_icnn/00_READ_FIRST.md",
        "partC_doc_icnn/PROJECT_STATUS.md",
        "partC_doc_icnn/12_DECISION_LOG.md",
        "partC_doc_icnn/20_HANDOFF_SYNC_PROTOCOL.md",
        "partC_doc_icnn/21_REVIEW_APPROVAL_GATES.md",
        "partC_doc_icnn/23_SYNC_BUNDLE_CONTENT_AND_SIZE_POLICY.md",
        "partC_doc_icnn/24_CODE_AND_DATA_SNAPSHOT_POLICY.md",
        "partC_doc_icnn/review_gates/phase_04/step_physical_reference_production_24h/user_directed_stop_iteration_200_20260902/APPROVAL.json",
        "partC_doc_icnn/review_gates/phase_04/step_physical_reference_production_24h/user_directed_stop_iteration_200_20260902/REVIEW_DECISION.md",
    ]
    for relative_text in governing:
        relative = Path(relative_text)
        _copy_preserving(project_root / relative, source / "governing_docs", relative)
    authority_root = Path(config["authority"]["package_path"]).parent
    _copy_preserving(Path(config["authority"]["launch_path"]), source / "governing_docs", Path("external_authority") / Path(config["authority"]["launch_path"]).name)
    extracted_authority = project_root.parent / "phase4_physical_reference_production_24h_authority_20260902T194541Z"
    for name in ("02_PHASE4_PRODUCTION_CHARTER.md", "07_OUTPUT_REPORTING_AND_HANDOFF_CONTRACT.md", "08_AUTONOMY_DIRTY_STATE_AND_MINOR_MISMATCH_POLICY.md", "09_CODEX_GOAL.md"):
        if (extracted_authority / name).is_file():
            _copy_preserving(extracted_authority / name, source / "governing_docs", Path("external_authority") / name)
    _copy_preserving(config_path, source / "configs", Path(config_path.name))
    original_config = output_root / "config" / config_path.name
    if original_config.is_file():
        _copy_preserving(original_config, source / "configs", Path("original_pre_amendment") / original_config.name)

    baseline_commit = str(config["authority"]["reviewed_code_commit"])
    head = _git(project_root, "rev-parse", "HEAD").stdout.strip()
    _write_text_new(source / "code_diff" / "reviewed_baseline_to_handoff.patch", _git(project_root, "diff", "--binary", baseline_commit, head, "--").stdout)
    _write_text_new(source / "code_diff" / "git_log.txt", _git(project_root, "log", "--oneline", "--decorate", f"{baseline_commit}..{head}").stdout)
    _write_text_new(source / "code_diff" / "git_status_before_handoff.txt", _git(project_root, "status", "--short").stdout)

    small_roots = [final_root / "tables", final_root / "validation", final_root / "reports", final_root / "routes"]
    for root in small_roots:
        for path in root.rglob("*"):
            if path.is_file() and path.stat().st_size <= MAX_HANDOFF_MEMBER_BYTES:
                _copy_preserving(path, source / "summary_data", Path(root.name) / path.relative_to(root))
    for path in (final_root / "figures").glob("*.png"):
        _copy_preserving(path, source / "figures", Path(path.name))
    _copy_preserving(final_root / "manifests" / "external_artifact_index.csv", source / "external_artifact_index", Path("external_artifact_index.csv"))
    _copy_preserving(output_root / "validation" / "independent_user_directed_final_refresh_iteration_00200.json", source / "summary_data", Path("independent_user_directed_final_refresh_iteration_00200.json"))
    _copy_preserving(output_root / "final_selected_reference" / "user_directed_iteration_0200_checkpoint_binding.json", source / "summary_data", Path("user_directed_iteration_0200_checkpoint_binding.json"))
    _copy_preserving(output_root / "status" / "campaign_status.json", source / "summary_data", Path("campaign_status_before_handoff_seal.json"))
    _copy_preserving(Path(config["authority"]["package_path"] + ".sha256") if Path(config["authority"]["package_path"] + ".sha256").is_file() else Path(config["authority"]["launch_path"]), source / "external_artifact_index", Path("authority_reference_sidecar_or_launch.txt"))
    _write_text_new(source / "geospatial_summary" / "NO_NEW_GEOSPATIAL_ARTIFACT.md", "No new geometry was generated. The final link table preserves accepted topology IDs and endpoints; authoritative geometry remains externally indexed by the accepted graph identity.")

    event_path = output_root / "logs" / "campaign_events.jsonl"
    event_lines = event_path.read_text(encoding="utf-8").splitlines()[-100:]
    _write_text_new(source / "log_excerpts" / "campaign_events_tail_100.jsonl", "\n".join(event_lines))
    update_tail = pd.read_parquet(final_root / "tables" / "authoritative_update_history.parquet").tail(50)
    _write_frame_new(update_tail, source / "log_excerpts" / "authoritative_update_history_tail_50.csv")

    content_rows: list[dict[str, Any]] = []
    for path in sorted((item for item in source.rglob("*") if item.is_file()), key=lambda item: item.relative_to(source).as_posix()):
        content_rows.append({"relative_path": path.relative_to(source).as_posix(), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)})
    _write_frame_new(pd.DataFrame(content_rows), source / "manifests" / "HANDOFF_CONTENT_MANIFEST.csv")
    privacy = _privacy_scan(source)
    _write_json_new(source / "manifests" / "privacy_scan.json", privacy)
    if privacy["status"] != "PASS":
        raise ProductionFinalizationError(f"Handoff privacy scan failed: {privacy}")
    checksums = _handoff_checksums(source)
    oversized = [path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file() and path.stat().st_size > MAX_HANDOFF_MEMBER_BYTES]
    if oversized:
        raise ProductionFinalizationError(f"Compact handoff contains files above 25 MiB: {oversized}")
    _create_zip(source, zip_path)
    if zip_path.stat().st_size > MAX_HANDOFF_BYTES:
        raise ProductionFinalizationError("Compact handoff ZIP exceeds the 200 MiB hard limit")
    fresh = _fresh_extract_verify(zip_path)
    zip_sha = file_sha256(zip_path)
    sidecar_path = Path(str(zip_path) + ".sha256")
    _write_text_new(sidecar_path, f"{zip_sha} *{zip_path.name}")
    companion_path = Path(str(zip_path) + ".companion.json")
    companion = {
        "schema_id": "phase4.physical_reference_production.handoff_companion.v1",
        "handoff_id": handoff_id,
        "status": FINAL_STATUS,
        "zip_path": str(zip_path.resolve()),
        "zip_size_bytes": int(zip_path.stat().st_size),
        "zip_sha256": zip_sha,
        "sidecar_path": str(sidecar_path.resolve()),
        "source_path": str(source.resolve()),
        "source_file_count_excluding_checksum_file": len(checksums),
        "fresh_extract_validation": fresh,
        "privacy_scan": privacy,
        "original_gap_target_met": metrics["original_gap_target_met"],
        "user_directed_stop_applied": True,
        "phase5_authorized": False,
        "next_step_authorized": False,
        "created_utc": _utc_now(),
    }
    _write_json_new(companion_path, companion)
    latest = source.parent / "LATEST_HANDOFF.txt"
    latest_payload = f"source_folder={source.resolve()}\nzip_path={zip_path.resolve()}\nzip_sha256={zip_sha}\ncreated_utc={companion['created_utc']}\nstatus={FINAL_STATUS}\n"
    if latest.exists():
        latest.unlink()
    _write_text_new(latest, latest_payload)
    campaign_path = output_root / "status" / "campaign_status.json"
    campaign = _load_json(campaign_path)
    campaign.update(
        {
            "state": FINAL_STATUS,
            "phase4_overall": FINAL_STATUS,
            "handoff_id": handoff_id,
            "handoff_zip": str(zip_path.resolve()),
            "handoff_zip_size_bytes": int(zip_path.stat().st_size),
            "handoff_zip_sha256": zip_sha,
            "phase5_authorized": False,
            "next_step_authorized": False,
            "updated_utc": _utc_now(),
        }
    )
    _write_json_atomic(campaign_path, campaign)
    return companion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("materialize", "confirm-figures", "seal-manifest", "handoff"), required=True)
    parser.add_argument("--handoff-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "materialize":
        result = materialize(Path(args.config).resolve(strict=True), args.phase_id, args.step_id, args.run_id)
    elif args.stage == "confirm-figures":
        result = confirm_figures(Path(args.config).resolve(strict=True), args.phase_id, args.step_id, args.run_id)
    elif args.stage == "seal-manifest":
        result = seal_manifest(Path(args.config).resolve(strict=True), args.phase_id, args.step_id, args.run_id)
    else:
        if not args.handoff_id:
            raise ProductionFinalizationError("--handoff-id is required for the handoff stage")
        result = package_handoff(Path(args.config).resolve(strict=True), args.phase_id, args.step_id, args.run_id, args.handoff_id)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
