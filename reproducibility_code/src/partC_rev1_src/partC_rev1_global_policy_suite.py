"""Postcapacity global sparse policy suite for corrected regions."""

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

from .partC_rev1_full_regional_training import partC_rev1_find_latest_small_region_summary, partC_rev1_load_corrected_graph_from_summary
from .partC_rev1_graph_snapshot import PartCRev1GraphSnapshot
from .partC_rev1_hashing import partC_rev1_json_hash
from .partC_rev1_io import partC_rev1_read_json, partC_rev1_write_json, partC_rev1_write_table
from .partC_rev1_od_snapshot import PartCRev1ODSnapshot
from .partC_rev1_paths import partC_rev1_resolve_path
from .partC_rev1_progress_logging import PartCRev1ProgressReporter
from .partC_rev1_regional_training import PartCRev1RegionalTrainingConfig, partC_rev1_load_benchmark_flow, partC_rev1_read_route_witnesses
from .partC_rev1_runtime_guard import PARTC_REV1_DEFAULT_MAX_RUN_SECONDS, partC_rev1_resource_snapshot
from .partC_rev1_sparse_global_solver import (
    PartCRev1SparseGlobalConfig,
    partC_rev1_build_active_pairs_from_regions,
    partC_rev1_build_hybrid_graph,
    partC_rev1_conceptual_pair_estimate,
    partC_rev1_run_sparse_master,
)


PARTC_REV1_GLOBAL_POLICY_STAGE = "partC_rev1_global_policy_suite"
PARTC_REV1_GLOBAL_POLICIES = ("fftt_const_approximation", "best_surrogate", "best_nonconst_surrogate")
PARTC_REV1_NONCONST_MODELS = {"xgboost", "simple_nn", "icnn_shallow", "icnn_deep"}
PARTC_REV1_POTENTIAL_MODELS = {"fftt_const_approximation", "icnn_shallow", "icnn_deep"}


@dataclass(frozen=True, slots=True)
class PartCRev1GlobalPolicySuiteConfig:
    config_path: str
    max_iterations: int = 3
    pilot_origins: int = 0
    origin_batch_size: int = 16
    master_gap_tol: float = 1e-3
    pricing_gap_tol: float = 1e-3
    price_tol: float = 1e-6
    lazy_abs_tol: float = 1e-6
    lazy_rel_tol: float = 1e-4
    full_refresh_frequency: int = 2
    regional_cost_slope: float = 1e-6
    runtime_guard_seconds: float = PARTC_REV1_DEFAULT_MAX_RUN_SECONDS
    resume: bool = True


def _partC_rev1_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def partC_rev1_base_model_name(model_name: str) -> str:
    model = str(model_name)
    return model.removeprefix("retuned_")


def partC_rev1_model_is_potential(model_name: str) -> bool:
    return partC_rev1_base_model_name(model_name) in PARTC_REV1_POTENTIAL_MODELS


def _partC_rev1_best_row(rows: pd.DataFrame) -> pd.Series | None:
    if rows.empty:
        return None
    ranked = rows.copy()
    ranked["_wmape_rank"] = pd.to_numeric(ranked.get("wmape", math.inf), errors="coerce").fillna(math.inf)
    ranked["_rmse_rank"] = pd.to_numeric(ranked.get("rmse", math.inf), errors="coerce").fillna(math.inf)
    ranked["_runtime_rank"] = pd.to_numeric(ranked.get("runtime_seconds", math.inf), errors="coerce").fillna(math.inf)
    ranked["_size_rank"] = pd.to_numeric(ranked.get("file_size_bytes", math.inf), errors="coerce").fillna(math.inf)
    ranked = ranked.sort_values(["_wmape_rank", "_rmse_rank", "_runtime_rank", "_size_rank", "model"], kind="mergesort")
    return ranked.iloc[0]


def _partC_rev1_model_row_payload(row: pd.Series | None, *, fallback_model: str = "") -> dict[str, Any]:
    if row is None:
        return {
            "selected_model": fallback_model,
            "artifact_path": "",
            "wmape": None,
            "rmse": None,
            "mae": None,
            "pair_count": None,
            "pair_order_hash": "",
            "terminal_order_hash": "",
            "training_data_hash": "",
        }
    return {
        "selected_model": str(row.get("model", fallback_model)),
        "artifact_path": str(row.get("artifact_path", "")),
        "wmape": None if pd.isna(row.get("wmape")) else float(row.get("wmape")),
        "rmse": None if pd.isna(row.get("rmse")) else float(row.get("rmse")),
        "mae": None if pd.isna(row.get("mae")) else float(row.get("mae")),
        "pair_count": None if pd.isna(row.get("pair_count")) else int(row.get("pair_count")),
        "pair_order_hash": str(row.get("pair_order_hash", "")),
        "terminal_order_hash": str(row.get("terminal_order_hash", "")),
        "training_data_hash": str(row.get("training_data_hash", "")),
    }


def partC_rev1_select_policy_models(
    status: pd.DataFrame,
    selected: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    policy: str,
) -> pd.DataFrame:
    if policy not in PARTC_REV1_GLOBAL_POLICIES:
        raise ValueError(f"Unknown global policy: {policy}")
    selected_by_region = {str(row.region_uuid): row for row in selected.itertuples(index=False)}
    registry = registry.copy()
    registry["base_model"] = registry["model"].astype(str).map(partC_rev1_base_model_name)
    verified = registry.loc[registry["status"].astype(str).eq("verified")].copy()
    rows: list[dict[str, Any]] = []
    for status_row in status.itertuples(index=False):
        region_uuid = str(status_row.region_uuid)
        status_value = str(status_row.status)
        if status_value != "trained":
            rows.append(
                {
                    "policy": policy,
                    "region_uuid": region_uuid,
                    "training_status": status_value,
                    "policy_status": f"explicit_physical_fallback_{status_value}",
                    "uses_regional_transfer": False,
                    "selected_model": "explicit_physical_fallback",
                    "model_is_potential": True,
                    "fallback_reason": str(getattr(status_row, "reason", "")),
                    "status_pair_count": getattr(status_row, "pair_count", None),
                }
            )
            continue
        region_registry = verified.loc[verified["region_uuid"].astype(str).eq(region_uuid)]
        picked: pd.Series | None
        policy_status = "selected"
        fallback_reason = ""
        if policy == "fftt_const_approximation":
            picked = _partC_rev1_best_row(region_registry.loc[region_registry["model"].astype(str).eq("fftt_const_approximation")])
        elif policy == "best_surrogate":
            selected_row = selected_by_region.get(region_uuid)
            picked = None
            if selected_row is not None:
                picked = _partC_rev1_best_row(region_registry.loc[region_registry["model"].astype(str).eq(str(selected_row.model))])
            if picked is None:
                policy_status = "best_missing_fallback_const"
                fallback_reason = "selected-model row missing from registry"
                picked = _partC_rev1_best_row(region_registry.loc[region_registry["model"].astype(str).eq("fftt_const_approximation")])
        else:
            nonconst = region_registry.loc[region_registry["base_model"].isin(PARTC_REV1_NONCONST_MODELS)]
            picked = _partC_rev1_best_row(nonconst)
            if picked is None:
                policy_status = "nonconst_missing_fallback_const"
                fallback_reason = "no verified nonconstant model"
                picked = _partC_rev1_best_row(region_registry.loc[region_registry["model"].astype(str).eq("fftt_const_approximation")])
        payload = _partC_rev1_model_row_payload(picked, fallback_model="fftt_const_approximation")
        if not payload["selected_model"]:
            rows.append(
                {
                    "policy": policy,
                    "region_uuid": region_uuid,
                    "training_status": status_value,
                    "policy_status": "explicit_physical_fallback_no_verified_model",
                    "uses_regional_transfer": False,
                    "selected_model": "explicit_physical_fallback",
                    "model_is_potential": True,
                    "fallback_reason": "no verified model row",
                    "status_pair_count": getattr(status_row, "pair_count", None),
                }
            )
            continue
        model_name = str(payload["selected_model"])
        rows.append(
            {
                "policy": policy,
                "region_uuid": region_uuid,
                "training_status": status_value,
                "policy_status": policy_status,
                "uses_regional_transfer": True,
                "model_is_potential": partC_rev1_model_is_potential(model_name),
                "fallback_reason": fallback_reason,
                "status_pair_count": getattr(status_row, "pair_count", None),
                **payload,
            }
        )
    return pd.DataFrame(rows)


def partC_rev1_policy_explicit_graph(
    graph: PartCRev1GraphSnapshot,
    *,
    represented_regions: set[str],
    policy_hash: str,
    policy_name: str,
) -> tuple[PartCRev1GraphSnapshot, dict[str, Any]]:
    labels = np.asarray([graph.region_idx_to_label[int(code)] for code in graph.region_code], dtype=object)
    explicit_labels = sorted({str(label) for label in labels.tolist() if str(label) != "backbone" and str(label) not in represented_regions})
    label_to_code = {label: idx for idx, label in enumerate(graph.region_idx_to_label)}
    if "backbone" not in label_to_code:
        raise ValueError("Graph snapshot has no backbone label")
    new_region_code = graph.region_code.copy()
    explicit_mask = np.isin(labels, np.asarray(explicit_labels, dtype=object))
    new_region_code[explicit_mask] = int(label_to_code["backbone"])
    graph_hash = partC_rev1_json_hash(
        {
            "stage": PARTC_REV1_GLOBAL_POLICY_STAGE,
            "parent_graph_hash": graph.graph_hash,
            "policy_hash": policy_hash,
            "policy_name": policy_name,
            "explicit_region_count": len(explicit_labels),
            "explicit_arc_count": int(explicit_mask.sum()),
        }
    )
    policy_graph = PartCRev1GraphSnapshot(
        graph_hash=graph_hash,
        partition_hash=graph.partition_hash,
        node_idx_to_uid=graph.node_idx_to_uid.copy(),
        arc_idx_to_directed_arc_id=graph.arc_idx_to_directed_arc_id.copy(),
        arc_idx_to_link_uid=graph.arc_idx_to_link_uid.copy(),
        tail=graph.tail.copy(),
        head=graph.head.copy(),
        free_time=graph.free_time.copy(),
        base_capacity=graph.base_capacity.copy(),
        capacity=graph.capacity.copy(),
        capacity_multiplier=graph.capacity_multiplier.copy(),
        capacity_multiplier_type=graph.capacity_multiplier_type.copy(),
        alpha=graph.alpha.copy(),
        beta=graph.beta.copy(),
        length=graph.length.copy(),
        region_code=new_region_code,
        region_idx_to_label=list(graph.region_idx_to_label),
        csr_indptr=graph.csr_indptr.copy(),
        csr_head=graph.csr_head.copy(),
        csr_arc_idx=graph.csr_arc_idx.copy(),
        reverse_csr_indptr=graph.reverse_csr_indptr.copy(),
        reverse_csr_head=graph.reverse_csr_head.copy(),
        reverse_csr_arc_idx=graph.reverse_csr_arc_idx.copy(),
        metadata={
            **graph.metadata,
            "stage": PARTC_REV1_GLOBAL_POLICY_STAGE,
            "parent_graph_hash": graph.graph_hash,
            "policy_hash": policy_hash,
            "policy_name": policy_name,
            "explicit_backbone_fallback_region_count": len(explicit_labels),
            "explicit_backbone_fallback_arc_count": int(explicit_mask.sum()),
        },
    )
    return policy_graph, {
        "explicit_backbone_fallback_region_count": len(explicit_labels),
        "explicit_backbone_fallback_arc_count": int(explicit_mask.sum()),
        "represented_region_count": len(represented_regions),
    }


def partC_rev1_global_policy_common_config_hash(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    *,
    correction_summary: dict[str, Any],
    training_summary: dict[str, Any],
    config: PartCRev1GlobalPolicySuiteConfig,
) -> str:
    payload = {
        "stage": PARTC_REV1_GLOBAL_POLICY_STAGE,
        "corrected_graph_hash": graph.graph_hash,
        "topology_hash": graph.topology_hash,
        "backbone_hash": graph.backbone_hash,
        "partition_hash": graph.partition_hash,
        "assignment_graph_hash": graph.assignment_graph_hash,
        "od_hash": od.od_hash,
        "correction_hash": correction_summary.get("correction_hash"),
        "training_run_hash": training_summary.get("run_hash"),
        "non_model_settings": asdict(config),
        "policy_isolated_field": "regional_model_choice_only",
    }
    return partC_rev1_json_hash(payload)[:16]


def _partC_rev1_global_policy_md(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# partC_rev1 Global Policy Suite Handoff",
        "",
        f"Generated UTC: {_partC_rev1_utc_now()}",
        "",
        "## Status",
        "",
        f"- State: `{summary.get('state')}`",
        f"- Run hash: `{summary.get('run_hash')}`",
        f"- Corrected graph hash: `{summary.get('corrected_graph_hash')}`",
        f"- Training run hash: `{summary.get('training_run_hash')}`",
        "",
        "## Runs",
        "",
        "| Policy | Algorithm | State | Stop reason | Gap | Runtime seconds | Origins |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for run in summary.get("runs", []):
        run_summary = dict(run.get("run", {}))
        gap = run_summary.get("official_full_refresh_gap")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run.get("policy")),
                    str(run.get("algorithm")),
                    str(run_summary.get("state")),
                    str(run_summary.get("stop_reason")),
                    "" if gap is None else f"{float(gap):.6g}",
                    f"{float(run_summary.get('runtime_seconds', 0.0)):.3f}",
                    f"{run_summary.get('processed_origins')}/{run_summary.get('total_origins')}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Artifacts", "", f"- Summary JSON: `{summary.get('summary_path')}`"])
    for artifact, artifact_path in dict(summary.get("artifacts", {})).items():
        lines.append(f"- {artifact}: `{artifact_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _partC_rev1_run_policy_algorithm(
    policy_graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    *,
    active_pairs: pd.DataFrame,
    config: PartCRev1SparseGlobalConfig,
    run_dir: Path,
    policy: str,
    policy_hash: str,
    policy_graph_hash: str,
    corrected_graph_hash: str,
    training_run_hash: str,
    selection_path: Path,
    fallback_summary: dict[str, Any],
    potential_compatible: bool,
    global_policy_common_config_hash: str,
    reporter: PartCRev1ProgressReporter | None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = partC_rev1_build_hybrid_graph(policy_graph, od, active_pairs)
    result = partC_rev1_run_sparse_master(bundle, config=config, reporter=reporter)
    active_pair_history_path = run_dir / "partC_rev1_4E_active_pair_history.csv"
    convergence_path = run_dir / "partC_rev1_4E_convergence.csv"
    lazy_path = run_dir / "partC_rev1_4E_lazy_refresh_history.csv"
    pricing_path = run_dir / "partC_rev1_4E_pricing_candidates.csv"
    final_backbone_path = run_dir / "partC_rev1_4E_final_backbone_flow.parquet"
    final_pair_path = run_dir / "partC_rev1_4E_final_regional_pair_flow.parquet"
    partC_rev1_write_table(active_pairs, active_pair_history_path, index=False)
    partC_rev1_write_table(result["convergence"], convergence_path, index=False)
    partC_rev1_write_table(result["lazy_refresh"], lazy_path, index=False)
    partC_rev1_write_table(result["pricing"], pricing_path, index=False)
    arc_table = bundle.active_arc_table.copy()
    arc_table["flow"] = result["flow"]
    arc_table["cost"] = result["cost"]
    partC_rev1_write_table(arc_table.loc[arc_table["kind"].eq("backbone")], final_backbone_path, index=False)
    partC_rev1_write_table(arc_table.loc[arc_table["kind"].eq("regional_transfer")], final_pair_path, index=False)
    conceptual_estimate = partC_rev1_conceptual_pair_estimate(active_pairs)
    summary = {
        "stage": PARTC_REV1_GLOBAL_POLICY_STAGE,
        "created_utc": _partC_rev1_utc_now(),
        "config": asdict(config),
        "policy": policy,
        "policy_hash": policy_hash,
        "global_policy_common_config_hash": global_policy_common_config_hash,
        "algorithm": config.assignment_algorithm,
        "corrected_graph_hash": corrected_graph_hash,
        "policy_graph_hash": policy_graph_hash,
        "training_run_hash": training_run_hash,
        "model_selection_csv": str(selection_path),
        "fallback_summary": fallback_summary,
        "potential_compatible": potential_compatible,
        "hybrid_graph_hash": bundle.graph.graph_hash,
        "active_pair_count": int(len(active_pairs)),
        "candidate_superset_pairs": int(len(active_pairs)),
        "conceptual_pair_estimate_for_model_regions": int(conceptual_estimate),
        "memory_bytes_active_pairs": int(active_pairs.memory_usage(deep=True).sum()) if not active_pairs.empty else 0,
        "hybrid_node_count": int(len(bundle.graph.node_idx_to_uid)),
        "hybrid_arc_count": int(len(bundle.graph.arc_idx_to_directed_arc_id)),
        "backbone_arc_count": int(bundle.backbone_arc_mask.sum()),
        "connector_arc_count": int(bundle.connector_arc_mask.sum()),
        "regional_transfer_arc_count": int(bundle.regional_arc_mask.sum()),
        "model_region_count": int(active_pairs["region_uuid"].nunique()) if not active_pairs.empty else 0,
        "run": result["summary"],
        "artifacts": {
            "active_pair_history_csv": str(active_pair_history_path),
            "convergence_csv": str(convergence_path),
            "lazy_refresh_history_csv": str(lazy_path),
            "pricing_candidates_csv": str(pricing_path),
            "final_backbone_flow_parquet": str(final_backbone_path),
            "final_regional_pair_flow_parquet": str(final_pair_path),
        },
        "known_approximations": [
            "Phase 3 uses Phase 2 selected model metadata to choose regional policy coverage.",
            "Regions without a verified selected regional model are kept as explicit physical arcs in a run-local graph snapshot.",
            "The sparse master uses the existing tau0/slope regional cost adapter; nonpotential policies are run only with MSA.",
            "Pricing remains empirical and does not enumerate dense omitted conceptual pairs.",
        ],
    }
    summary_path = run_dir / "partC_rev1_4E_summary.json"
    summary["summary_path"] = str(summary_path)
    partC_rev1_write_json(summary, summary_path)
    return summary


def partC_rev1_run_global_policy_suite_stage(
    graph: PartCRev1GraphSnapshot,
    od: PartCRev1ODSnapshot,
    *,
    correction_summary: dict[str, Any],
    training_summary: dict[str, Any],
    benchmark_flow: np.ndarray,
    route_witnesses: pd.DataFrame,
    config: PartCRev1GlobalPolicySuiteConfig,
    output_dir: str | Path,
    reporter: PartCRev1ProgressReporter | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_root = partC_rev1_resolve_path(output_dir)
    artifacts = dict(training_summary.get("artifacts", {}))
    status = pd.read_csv(partC_rev1_resolve_path(artifacts["status_csv"]))
    selected = pd.read_csv(partC_rev1_resolve_path(artifacts["selected_models_csv"]))
    registry = pd.read_csv(partC_rev1_resolve_path(artifacts["model_registry_csv"]))
    run_hash = partC_rev1_json_hash(
        {
            "stage": PARTC_REV1_GLOBAL_POLICY_STAGE,
            "corrected_graph_hash": graph.graph_hash,
            "od_hash": od.od_hash,
            "training_run_hash": training_summary.get("run_hash"),
            "correction_hash": correction_summary.get("correction_hash"),
            "config": asdict(config),
        }
    )[:16]
    suite_root = out_root / f"partC_rev1_global_policy_runs_{run_hash}"
    suite_root.mkdir(parents=True, exist_ok=True)
    common_config_hash = partC_rev1_global_policy_common_config_hash(
        graph,
        od,
        correction_summary=correction_summary,
        training_summary=training_summary,
        config=config,
    )
    regional_cfg = PartCRev1RegionalTrainingConfig(
        config_path=config.config_path,
        max_active_pairs=int(training_summary.get("config", {}).get("max_active_pairs", 64)),
        top_k_per_source=int(training_summary.get("config", {}).get("top_k_per_source", 6)),
        regional_solver_iterations=int(training_summary.get("config", {}).get("regional_solver_iterations", 3)),
        regional_gap_target=float(training_summary.get("config", {}).get("regional_gap_target", 1e-3)),
        regional_origin_batch_size=int(training_summary.get("config", {}).get("regional_origin_batch_size", 8)),
        requested_samples_per_region=int(training_summary.get("config", {}).get("min_samples_per_region", 100)),
        pilot_samples_per_region=int(training_summary.get("config", {}).get("min_samples_per_region", 100)),
        pilot_region_count=0,
    )
    runs: list[dict[str, Any]] = []
    all_policy_rows: list[pd.DataFrame] = []
    for policy in PARTC_REV1_GLOBAL_POLICIES:
        policy_started = time.perf_counter()
        policy_rows = partC_rev1_select_policy_models(status, selected, registry, policy=policy)
        all_policy_rows.append(policy_rows)
        policy_hash = partC_rev1_json_hash(
            {
                "policy": policy,
                "training_run_hash": training_summary.get("run_hash"),
                "rows": policy_rows[["region_uuid", "selected_model", "policy_status", "uses_regional_transfer"]].to_dict("records"),
            }
        )[:16]
        policy_dir = suite_root / f"partC_rev1_policy_{policy}_hash-{policy_hash}"
        policy_dir.mkdir(parents=True, exist_ok=True)
        selection_path = policy_dir / "partC_rev1_global_policy_model_selection.csv"
        partC_rev1_write_table(policy_rows, selection_path, index=False)
        represented = set(policy_rows.loc[policy_rows["uses_regional_transfer"].astype(bool), "region_uuid"].astype(str))
        policy_graph, fallback_summary = partC_rev1_policy_explicit_graph(graph, represented_regions=represented, policy_hash=policy_hash, policy_name=policy)
        potential_compatible = bool(policy_rows.loc[policy_rows["uses_regional_transfer"].astype(bool), "model_is_potential"].all())
        algorithms = ["msa"] + (["fw", "cfw"] if potential_compatible else [])
        if reporter is not None:
            reporter.emit(
                "global_policy_started",
                policy=policy,
                policy_hash=policy_hash,
                potential_compatible=potential_compatible,
                algorithms=algorithms,
                **fallback_summary,
            )
        active_started = time.perf_counter()
        active_pairs_path = policy_dir / "partC_rev1_global_policy_active_pairs.csv"
        if config.resume and active_pairs_path.exists():
            active_pairs = pd.read_csv(active_pairs_path)
            active_source = "cached"
        else:
            active_pairs = partC_rev1_build_active_pairs_from_regions(
                policy_graph,
                od,
                benchmark_flow=benchmark_flow,
                route_witnesses=route_witnesses,
                completed_regions=[{"region_uuid": region} for region in sorted(represented)],
                regional_config=regional_cfg,
                reporter=reporter,
            )
            partC_rev1_write_table(active_pairs, active_pairs_path, index=False)
            active_source = "built"
        if reporter is not None:
            reporter.emit("global_policy_active_pairs_built", policy=policy, active_pair_count=int(len(active_pairs)), active_pair_source=active_source, active_pairs_csv=str(active_pairs_path), runtime_seconds=time.perf_counter() - active_started)
        for algorithm in algorithms:
            run_dir = policy_dir / f"partC_rev1_{policy}_{algorithm}"
            summary_path = run_dir / "partC_rev1_4E_summary.json"
            if config.resume and summary_path.exists():
                run_summary = partC_rev1_read_json(summary_path)
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                stage_config = PartCRev1SparseGlobalConfig(
                    config_path=config.config_path,
                    max_iterations=int(config.max_iterations),
                    pilot_origins=int(config.pilot_origins),
                    origin_batch_size=int(config.origin_batch_size),
                    master_gap_tol=float(config.master_gap_tol),
                    pricing_gap_tol=float(config.pricing_gap_tol),
                    price_tol=float(config.price_tol),
                    lazy_abs_tol=float(config.lazy_abs_tol),
                    lazy_rel_tol=float(config.lazy_rel_tol),
                    full_refresh_frequency=int(config.full_refresh_frequency),
                    regional_cost_slope=float(config.regional_cost_slope),
                    assignment_algorithm=algorithm,
                    runtime_guard_seconds=float(config.runtime_guard_seconds),
                )
                run_summary = _partC_rev1_run_policy_algorithm(
                    policy_graph,
                    od,
                    active_pairs=active_pairs,
                    config=stage_config,
                    run_dir=run_dir,
                    policy=policy,
                    policy_hash=policy_hash,
                    policy_graph_hash=policy_graph.graph_hash,
                    corrected_graph_hash=graph.graph_hash,
                    training_run_hash=str(training_summary.get("run_hash")),
                    selection_path=selection_path,
                    fallback_summary=fallback_summary,
                    potential_compatible=potential_compatible,
                    global_policy_common_config_hash=common_config_hash,
                    reporter=reporter,
                )
            runs.append(
                {
                    "policy": policy,
                    "algorithm": algorithm,
                    "policy_hash": policy_hash,
                    "global_policy_common_config_hash": common_config_hash,
                    "summary_path": str(summary_path),
                    "run": run_summary.get("run", {}),
                    "potential_compatible": potential_compatible,
                    "model_selection_csv": str(selection_path),
                    "active_pairs_csv": str(active_pairs_path),
                    **fallback_summary,
                }
            )
            if reporter is not None:
                reporter.emit("global_policy_algorithm_finished", policy=policy, algorithm=algorithm, summary_path=str(summary_path), run=run_summary.get("run", {}))
        if reporter is not None:
            resource = partC_rev1_resource_snapshot(started)
            reporter.emit("global_policy_finished", policy=policy, elapsed_seconds=time.perf_counter() - policy_started, memory_rss_mb=resource.memory_rss_mb, temp_disk_mb=resource.temp_disk_mb)
    all_policy_path = suite_root / "partC_rev1_global_policy_model_selection_all.csv"
    partC_rev1_write_table(pd.concat(all_policy_rows, ignore_index=True), all_policy_path, index=False)
    summary = {
        "stage": PARTC_REV1_GLOBAL_POLICY_STAGE,
        "created_utc": _partC_rev1_utc_now(),
        "state": "measured_complete" if all(dict(run.get("run", {})).get("state") == "measured_complete" for run in runs) else "measured_partial_projected",
        "run_hash": run_hash,
        "run_root": str(suite_root),
        "config": asdict(config),
        "corrected_graph_hash": graph.graph_hash,
        "topology_hash": graph.topology_hash,
        "backbone_hash": graph.backbone_hash,
        "assignment_graph_hash": graph.assignment_graph_hash,
        "corrected_partition_hash": graph.partition_hash,
        "od_hash": od.od_hash,
        "global_policy_common_config_hash": common_config_hash,
        "correction_hash": correction_summary.get("correction_hash"),
        "training_run_hash": training_summary.get("run_hash"),
        "candidate_region_count": int(len(status)),
        "trained_region_count": int(status["status"].astype(str).eq("trained").sum()),
        "policy_count": len(PARTC_REV1_GLOBAL_POLICIES),
        "run_count": len(runs),
        "runtime_seconds": time.perf_counter() - started,
        "runs": runs,
        "artifacts": {"all_policy_model_selection_csv": str(all_policy_path)},
    }
    summary_path = suite_root / "partC_rev1_global_policy_suite_summary.json"
    summary["summary_path"] = str(summary_path)
    partC_rev1_write_json(summary, summary_path)
    handoff_path = suite_root / "partC_rev1_global_policy_suite_handoff.md"
    _partC_rev1_global_policy_md(summary, handoff_path)
    summary["handoff_path"] = str(handoff_path)
    partC_rev1_write_json(summary, summary_path)
    return summary


def partC_rev1_load_global_policy_inputs(
    *,
    output_dir: str | Path,
    training_summary_path: str | Path | None = None,
    correction_summary_path: str | Path | None = None,
) -> tuple[PartCRev1GraphSnapshot, dict[str, Any], dict[str, Any]]:
    out_dir = partC_rev1_resolve_path(output_dir)
    training_path = partC_rev1_resolve_path(training_summary_path) if training_summary_path else partC_rev1_find_latest_full_training_summary(out_dir)
    correction_path = partC_rev1_resolve_path(correction_summary_path) if correction_summary_path else partC_rev1_find_latest_small_region_summary(out_dir)
    graph, correction_summary = partC_rev1_load_corrected_graph_from_summary(correction_path)
    training_summary = json.loads(training_path.read_text(encoding="utf-8"))
    if str(training_summary.get("corrected_graph_hash")) != graph.graph_hash:
        raise RuntimeError("Full regional training summary does not match the corrected graph summary")
    return graph, correction_summary, training_summary
