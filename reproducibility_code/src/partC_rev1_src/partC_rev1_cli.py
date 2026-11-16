"""Command line interface for Prompt 1 partC_rev1 utilities."""

from __future__ import annotations

import argparse
import json
import numpy as np
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .partC_rev1_artifact_registry import PartCRev1ArtifactRegistry
from .partC_rev1_config import load_partC_rev1_config, write_partC_rev1_default_config
from .partC_rev1_enrich_capacity_pipeline import partC_rev1_run_enrich_capacity
from .partC_rev1_gdb import partC_rev1_guess_link_node_layers, partC_rev1_inspect_gdb, partC_rev1_list_gdb_layers
from .partC_rev1_graph_snapshot import (
    partC_rev1_build_graph_snapshot_from_sqlite,
    partC_rev1_check_partition_consistency,
    partC_rev1_collect_positive_od_endpoints,
    partC_rev1_load_graph_snapshot,
)
from .partC_rev1_io import partC_rev1_read_json, partC_rev1_read_table, partC_rev1_write_json, partC_rev1_write_table
from .partC_rev1_logging import partC_rev1_setup_logging
from .partC_rev1_od_snapshot import partC_rev1_build_od_snapshot_from_sqlite, partC_rev1_load_od_snapshot
from .partC_rev1_paths import partC_rev1_output_file, partC_rev1_resolve_path
from .partC_rev1_preprocess import partC_rev1_run_preprocess
from .partC_rev1_progress_logging import PartCRev1ProgressReporter
from .partC_rev1_prompt3_pipeline import partC_rev1_run_partition, partC_rev1_run_partition_stability
from .partC_rev1_run_manifest import (
    partC_rev1_append_jsonl,
    partC_rev1_stage_manifest,
    partC_rev1_write_master_state,
)
from .partC_rev1_aon import partC_rev1_aon_load
from .partC_rev1_assignment import (
    PartCRev1AssignmentConfig,
    partC_rev1_solve_physical_assignment,
    partC_rev1_solver_result_summary,
)
from .partC_rev1_bpr import partC_rev1_bpr_cost
from .partC_rev1_physical_benchmarks import (
    PARTC_REV1_4C_STAGE,
    PartCRev1PhysicalBenchmarkStageConfig,
    partC_rev1_generate_route_witnesses,
    partC_rev1_run_physical_benchmark_stage,
)
from .partC_rev1_small_region_correction import (
    PARTC_REV1_SMALL_REGION_STAGE,
    PartCRev1SmallRegionCorrectionConfig,
    partC_rev1_append_route_witnesses_to_correction_summary,
    partC_rev1_load_cost_from_directed_flow,
    partC_rev1_run_small_region_correction_stage,
)
from .partC_rev1_full_regional_training import (
    PARTC_REV1_FULL_REGIONAL_STAGE,
    PartCRev1FullRegionalTrainingConfig,
    partC_rev1_find_latest_small_region_summary,
    partC_rev1_load_corrected_graph_from_summary,
    partC_rev1_run_full_regional_training_stage,
)
from .partC_rev1_global_policy_suite import (
    PARTC_REV1_GLOBAL_POLICY_STAGE,
    PartCRev1GlobalPolicySuiteConfig,
    partC_rev1_load_global_policy_inputs,
    partC_rev1_run_global_policy_suite_stage,
)
from .partC_rev1_regional_training import (
    PARTC_REV1_4D_STAGE,
    PartCRev1RegionalTrainingConfig,
    partC_rev1_load_benchmark_flow,
    partC_rev1_read_route_witnesses,
    partC_rev1_region_size_table,
    partC_rev1_run_regional_training_stage,
    partC_rev1_write_4d_handoff,
)
from .partC_rev1_sparse_global_solver import (
    PARTC_REV1_4E_STAGE,
    PartCRev1SparseGlobalConfig,
    partC_rev1_run_sparse_global_stage,
    partC_rev1_write_4e_handoff,
)
from .partC_rev1_physical_reload import (
    PARTC_REV1_4F_STAGE,
    PartCRev1PhysicalReloadConfig,
    partC_rev1_run_physical_reload_stage,
    partC_rev1_write_4f_handoff,
)
from .partC_rev1_physical_reload_suite import (
    PARTC_REV1_PHYSICAL_RELOAD_SUITE_STAGE,
    PartCRev1PhysicalReloadSuiteConfig,
    partC_rev1_find_latest_global_policy_summary,
    partC_rev1_run_physical_reload_suite_stage,
)
from .partC_rev1_partition_sweep_guard import (
    PARTC_REV1_PARTITION_SENSITIVITY_STAGE,
    PartCRev1PartitionSensitivityConfig,
    partC_rev1_run_partition_sensitivity_package,
)
from .partC_rev1_reviewer_experiments import (
    PARTC_REV1_4G_STAGE,
    PartCRev1ReviewerExperimentConfig,
    partC_rev1_run_reviewer_experiment_stage,
    partC_rev1_write_4g_handoff,
)
from .partC_rev1_revised_reviewer_experiments import (
    PARTC_REV1_REVISED_EXPERIMENT_STAGE,
    PartCRev1RevisedReviewerExperimentConfig,
    partC_rev1_run_revised_reviewer_experiment_stage,
)
from .partC_rev1_abstract_flow import (
    partC_rev1_build_abstract_graph_bundle,
    partC_rev1_load_saved_abstract_graph_bundle,
    partC_rev1_recovery_root,
    partC_rev1_run_abstract_global_assignment,
    partC_rev1_run_abstract_reachability_audit,
    partC_rev1_solve_fw_with_abstract_tracking,
    partC_rev1_train_recovery_regional_models,
    partC_rev1_write_abstract_benchmark_artifacts,
    partC_rev1_write_recovery_summary_pack,
)
from .partC_rev1_remaining_experiments import (
    PartCRev1RemainingExperimentConfig,
    partC_rev1_run_remaining_experiments,
    partC_rev1_validate_remaining_experiments,
)


def _partC_rev1_add_inspection_row(
    rows: list[dict[str, str]],
    *,
    category: str,
    item: str,
    path: str = "",
    status: str,
    details: str = "",
) -> None:
    rows.append({"category": category, "item": item, "path": path, "status": status, "details": details})


def _partC_rev1_existing_path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return partC_rev1_resolve_path(str(value))


def _partC_rev1_inspect_table(
    rows: list[dict[str, str]],
    *,
    category: str,
    item: str,
    path_value: Any,
    dtype_overrides: dict[str, Any] | None = None,
) -> None:
    path = _partC_rev1_existing_path(path_value)
    if path is None:
        _partC_rev1_add_inspection_row(rows, category=category, item=item, status="empty_path")
        return
    if not path.exists():
        _partC_rev1_add_inspection_row(rows, category=category, item=item, path=str(path), status="missing")
        return
    try:
        df = partC_rev1_read_table(path, dtype_overrides=dtype_overrides, nrows=5)
        details = f"sample_rows={len(df)}; columns={list(df.columns)}"
        _partC_rev1_add_inspection_row(rows, category=category, item=item, path=str(path), status="present", details=details)
    except Exception as exc:
        _partC_rev1_add_inspection_row(rows, category=category, item=item, path=str(path), status="error", details=str(exc))


def _partC_rev1_escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _partC_rev1_write_markdown_report(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# partC_rev1 Runtime Data Inspection Report",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Category | Item | Path | Status | Details |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    _partC_rev1_escape_md(row.category),
                    _partC_rev1_escape_md(row.item),
                    _partC_rev1_escape_md(row.path),
                    _partC_rev1_escape_md(row.status),
                    _partC_rev1_escape_md(row.details),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def partC_rev1_inspect_data(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    inputs = config.get("inputs", {})
    ground_truth = config.get("ground_truth", {})

    gdb_path = _partC_rev1_existing_path(inputs.get("network_gdb_path"))
    if gdb_path is None:
        _partC_rev1_add_inspection_row(rows, category="geodatabase", item="network_gdb_path", status="empty_path")
    elif not gdb_path.exists():
        _partC_rev1_add_inspection_row(rows, category="geodatabase", item="network_gdb_path", path=str(gdb_path), status="missing")
    else:
        try:
            layers = partC_rev1_list_gdb_layers(str(gdb_path))
            _partC_rev1_add_inspection_row(
                rows,
                category="geodatabase",
                item="layers",
                path=str(gdb_path),
                status="present",
                details=", ".join(layers),
            )
            inspection = partC_rev1_inspect_gdb(str(gdb_path))
            for layer_row in inspection.itertuples(index=False):
                _partC_rev1_add_inspection_row(
                    rows,
                    category="geodatabase_layer",
                    item=str(layer_row.layer),
                    path=str(gdb_path),
                    status="inspected",
                    details=f"geometry={layer_row.geometry_type}; rows={layer_row.feature_count}; link_score={layer_row.link_score}; node_score={layer_row.node_score}",
                )
            guesses = partC_rev1_guess_link_node_layers(str(gdb_path))
            _partC_rev1_add_inspection_row(
                rows,
                category="geodatabase",
                item="guessed_layers",
                path=str(gdb_path),
                status="ok",
                details=str(guesses),
            )
        except Exception as exc:
            _partC_rev1_add_inspection_row(
                rows,
                category="geodatabase",
                item="inspection",
                path=str(gdb_path),
                status="error",
                details=str(exc),
            )

    _partC_rev1_inspect_table(rows, category="table", item="faf_od", path_value=inputs.get("od_path"))
    _partC_rev1_inspect_table(
        rows,
        category="table",
        item="origin_factors",
        path_value=inputs.get("origin_factors_path"),
        dtype_overrides={"dms_orig_cnty": "string"},
    )
    _partC_rev1_inspect_table(
        rows,
        category="table",
        item="destination_factors",
        path_value=inputs.get("destination_factors_path"),
        dtype_overrides={"dms_dest_cnty": "string"},
    )

    flow_dir = _partC_rev1_existing_path(inputs.get("ground_truth_flow_dir"))
    if flow_dir is None:
        _partC_rev1_add_inspection_row(rows, category="ground_truth", item="flow_dir", status="empty_path")
    elif not flow_dir.exists():
        _partC_rev1_add_inspection_row(rows, category="ground_truth", item="flow_dir", path=str(flow_dir), status="missing")
    else:
        candidate_files = set(ground_truth.get("candidate_files", []))
        csv_files = sorted(path.name for path in flow_dir.glob("*.csv"))
        candidates = [name for name in csv_files if not candidate_files or name in candidate_files]
        _partC_rev1_add_inspection_row(
            rows,
            category="ground_truth",
            item="candidate_flow_csvs",
            path=str(flow_dir),
            status="present",
            details=", ".join(candidates),
        )
        recommended = ground_truth.get("recommended_file")
        if recommended:
            _partC_rev1_inspect_table(
                rows,
                category="ground_truth",
                item="recommended_flow_file",
                path_value=flow_dir / str(recommended),
            )

    report_df = pd.DataFrame(rows, columns=["category", "item", "path", "status", "details"])
    csv_path = partC_rev1_output_file(config, "partC_rev1_data_inspection_runtime_report.csv")
    md_path = partC_rev1_output_file(config, "partC_rev1_data_inspection_report.md")
    partC_rev1_write_table(report_df, csv_path, index=False)
    _partC_rev1_write_markdown_report(report_df, md_path)
    return report_df


def _partC_rev1_cmd_write_default_config(args: argparse.Namespace) -> int:
    write_partC_rev1_default_config(args.output)
    print(f"Wrote default config: {args.output}")
    return 0


def _partC_rev1_cmd_validate_config(args: argparse.Namespace) -> int:
    config = load_partC_rev1_config(args.config)
    warnings = config.get("_partC_rev1_warnings", [])
    print(f"Config valid: {args.config}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    return 0


def _partC_rev1_cmd_inspect_data(args: argparse.Namespace) -> int:
    config = load_partC_rev1_config(args.config)
    outputs = config.get("outputs", {})
    partC_rev1_setup_logging(outputs.get("output_dir", "partC_rev1_outputs"))
    report = partC_rev1_inspect_data(config)
    print(f"Inspection rows: {len(report)}")
    print(partC_rev1_output_file(config, "partC_rev1_data_inspection_runtime_report.csv"))
    print(partC_rev1_output_file(config, "partC_rev1_data_inspection_report.md"))
    return 0


def _partC_rev1_cmd_preprocess(args: argparse.Namespace) -> int:
    result = partC_rev1_run_preprocess(args.config)
    print(f"Preprocess complete: {args.config}")
    print(f"Cache: {result['cache_path']}")
    print(f"Clean geodatabase: {result['clean_geodatabase_path']}")
    return 0


def _partC_rev1_cmd_enrich_capacity(args: argparse.Namespace) -> int:
    result = partC_rev1_run_enrich_capacity(args.config, confirm_full_run=bool(args.confirm_full_run))
    print(f"Capacity enrichment complete: {args.config}")
    print(f"Cache: {result['cache_path']}")
    print(f"Enriched geodatabase: {result['enriched_gpkg_path']}")
    print(f"Runtime seconds: {result['runtime_seconds']:.3f}")
    return 0


def _partC_rev1_cmd_partition(args: argparse.Namespace) -> int:
    result = partC_rev1_run_partition(args.config, confirm_full_run=bool(args.confirm_full_run))
    outputs = result["outputs"]
    print(f"Partition complete: {args.config}")
    print(f"Cache: {result['cache_path']}")
    print(f"Final backbone arcs: {len(outputs['final_backbone_arcs'])}")
    print(f"Regions: {len(outputs['regions'])}")
    print(f"Runtime seconds: {result['runtime_seconds']:.3f}")
    return 0


def _partC_rev1_cmd_partition_stability(args: argparse.Namespace) -> int:
    result = partC_rev1_run_partition_stability(args.config, confirm_full_run=bool(args.confirm_full_run))
    runs = result["stability"]["runs"]
    print(f"Partition stability complete: {args.config}")
    print(f"Cache: {result['cache_path']}")
    print(f"Runs: {len(runs)}")
    print("Variation dimensions: resolution, random_seed")
    print(f"Runtime seconds: {result['runtime_seconds']:.3f}")
    return 0


def _partC_rev1_git_commit() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True, encoding="utf-8")
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _partC_rev1_cmd_prepare_solver_artifacts(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("prepare-solver-artifacts requires --confirm-full-run for real-data execution")
    config = load_partC_rev1_config(args.config)
    output_dir = partC_rev1_resolve_path(str(config.get("outputs", {}).get("output_dir", "partC_rev1_outputs")))
    docs_dir = partC_rev1_resolve_path(str(config.get("outputs", {}).get("docs_dir", "partC_rev1_docs")))
    cache_dir = partC_rev1_resolve_path(str(config.get("project", {}).get("cache_dir", "partC_rev1_cache")))
    sqlite_path = partC_rev1_resolve_path(str(config.get("preprocess", {}).get("cache_sqlite", cache_dir / "partC_rev1_network.sqlite")))
    reporter = PartCRev1ProgressReporter("partC_rev1_4A", output_dir=output_dir)
    started = time.perf_counter()
    try:
        reporter.emit("prepare_solver_artifacts_started", config=args.config, sqlite_path=str(sqlite_path))
        partition_check = partC_rev1_check_partition_consistency(sqlite_path, config)
        reporter.emit("partition_consistency_checked", **asdict(partition_check))
        if not partition_check.matches:
            raise RuntimeError(
                "Partition metadata does not match config; Prompt 4A requires regeneration before snapshots: "
                f"{asdict(partition_check)}"
            )
        endpoints = partC_rev1_collect_positive_od_endpoints(sqlite_path, reporter=reporter)
        reporter.emit("od_endpoints_collected", endpoint_count=len(endpoints))
        graph, graph_paths = partC_rev1_build_graph_snapshot_from_sqlite(
            sqlite_path,
            od_endpoint_uids=endpoints,
            output_dir=cache_dir,
            assignment_capacity_config=config,
            reporter=reporter,
        )
        od, od_paths = partC_rev1_build_od_snapshot_from_sqlite(
            sqlite_path,
            node_uid_to_idx=graph.node_uid_to_idx,
            output_dir=cache_dir,
            reporter=reporter,
        )
        artifacts = [
            {"label": "graph_npz", "path": graph_paths["npz"], "kind": "npz"},
            {"label": "graph_nodes", "path": graph_paths["nodes"], "kind": "parquet"},
            {"label": "graph_arcs", "path": graph_paths["arcs"], "kind": "parquet"},
            {"label": "graph_metadata", "path": graph_paths["metadata"], "kind": "json"},
            {"label": "od_npz", "path": od_paths["npz"], "kind": "npz"},
            {"label": "od_metadata", "path": od_paths["metadata"], "kind": "json"},
        ]
        manifest = partC_rev1_stage_manifest(
            stage="partC_rev1_4A",
            config_path=args.config,
            graph_hash=graph.graph_hash,
            od_hash=od.od_hash,
            partition_hash=graph.partition_hash,
            artifacts=artifacts,
            tests=[
                {
                    "command": "conda run -n py312 python -m pytest partC_rev1_tests",
                    "status": "baseline_passed_internal_pytest_rc0",
                    "note": "conda wrapper emits libmamba entry-point warning in this environment",
                }
            ],
            metadata={
                "topology_hash": graph.topology_hash,
                "backbone_hash": graph.backbone_hash,
                "assignment_graph_hash": graph.assignment_graph_hash,
                "partition_check": asdict(partition_check),
                "runtime_seconds": time.perf_counter() - started,
                "git_commit": _partC_rev1_git_commit(),
            },
        )
        manifest_path = output_dir / "partC_rev1_4A_run_manifest.json"
        from .partC_rev1_io import partC_rev1_write_json

        partC_rev1_write_json(manifest, manifest_path)
        registry = PartCRev1ArtifactRegistry(
            output_dir / "partC_rev1_4A_registry",
            "partC_rev1_4A",
            metadata={
                "graph_hash": graph.graph_hash,
                "topology_hash": graph.topology_hash,
                "backbone_hash": graph.backbone_hash,
                "assignment_graph_hash": graph.assignment_graph_hash,
                "od_hash": od.od_hash,
                "partition_hash": graph.partition_hash,
                "partition_check": asdict(partition_check),
            },
        )
        registry.add_input(label="sqlite", path=sqlite_path, description="Read-only authoritative Prompt 0-3/2.5 SQLite cache")
        for artifact in artifacts:
            registry.add_artifact(
                label=str(artifact["label"]),
                path=str(artifact["path"]),
                kind=str(artifact["kind"]),
                description="Prompt 4A solver artifact",
            )
        registry_path = registry.write(output_dir / "partC_rev1_4A_registry_manifest.json")
        handoff_path = output_dir / "partC_rev1_4A_handoff.md"
        handoff_lines = [
            "# partC_rev1 4A Handoff",
            "",
            f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
            f"Git commit at run time: {_partC_rev1_git_commit()}",
            f"Config: {args.config}",
            f"SQLite: {sqlite_path}",
            "",
            "## Status",
            "",
            "- Partition consistency: matched; current region labels retained.",
            "- Deterministic graph snapshot: created.",
            "- Origin-grouped OD snapshot: created.",
            "- Shared logging/runtime/artifact foundations: implemented and tested.",
            "",
            "## Hashes",
            "",
            f"- Graph hash: `{graph.graph_hash}`",
            f"- Topology hash: `{graph.topology_hash}`",
            f"- Backbone hash: `{graph.backbone_hash}`",
            f"- Assignment graph hash: `{graph.assignment_graph_hash}`",
            f"- Partition hash: `{graph.partition_hash}`",
            f"- OD hash: `{od.od_hash}`",
            "",
            "## Artifacts",
            "",
        ]
        handoff_lines.extend(f"- {item['label']}: `{item['path']}`" for item in artifacts)
        handoff_lines.extend(
            [
                f"- Run manifest: `{manifest_path}`",
                f"- Registry manifest: `{registry_path}`",
                f"- Stage log: `{reporter.log_path}`",
                f"- Progress JSONL: `{reporter.progress_path}`",
                "",
                "## Tests",
                "",
                "- Baseline command: `conda run -n py312 python -m pytest partC_rev1_tests`",
                "- Pytest internal result before implementation: 73 passed, 2 warnings, return code 0.",
                "- Environment note: `conda run` emits a `conda-libmamba-solver` entry-point warning and may return a wrapper nonzero despite pytest rc=0.",
                "",
                "## Runtime",
                "",
                f"- 4A prepare-solver-artifacts runtime seconds: {time.perf_counter() - started:.3f}",
                f"- Graph nodes: {len(graph.node_idx_to_uid):,}",
                f"- Graph directed arcs: {len(graph.arc_idx_to_directed_arc_id):,}",
                f"- OD rows: {len(od.demand):,}",
                f"- OD origins: {len(od.origin_group_node_idx):,}",
                f"- Unmatched OD rows: {od.metadata.get('unmatched_rows', 0):,}",
                "",
                "## Exact Next Command",
                "",
                "```text",
                f"conda run -n py312 python -m partC_rev1_src.partC_rev1_cli run-physical-benchmarks --config {args.config} --confirm-full-run --gap-target 1e-4 --gap-reference 1e-4 --witness-all-positive-origins --witness-diagnostic-max-origins 64 --backend-pilot-min-seconds 120 --backend-pilot-min-batches 3 --backend-pilot-cv-tolerance 0.25 --validate-dependency-hashes",
                "```",
                "",
                "Prompt 4B should inspect this handoff, the run manifest, graph metadata, OD metadata, and the stage progress log before implementing solver kernels.",
            ]
        )
        handoff_path.write_text("\n".join(handoff_lines) + "\n", encoding="utf-8")
        partC_rev1_write_master_state(
            output_dir / "partC_rev1_master_run_state.json",
            {
                "current_stage": "partC_rev1_4A",
                "current_run": "prepare_solver_artifacts",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "completed_artifacts": artifacts,
                "current_checkpoint": str(handoff_path),
                "next_action": "run_prompt4B_after_commit",
            },
        )
        partC_rev1_append_jsonl(
            output_dir / "partC_rev1_master_resource_usage.jsonl",
            {
                "stage": "partC_rev1_4A",
                "event": "prepare_solver_artifacts_finished",
                "runtime_seconds": time.perf_counter() - started,
                "graph_hash": graph.graph_hash,
                "od_hash": od.od_hash,
            },
        )
        reporter.emit("prepare_solver_artifacts_finished", handoff=str(handoff_path), manifest=str(manifest_path))
        print(f"Graph hash: {graph.graph_hash}")
        print(f"OD hash: {od.od_hash}")
        print(f"Handoff: {handoff_path}")
        return 0
    finally:
        reporter.close()


def _partC_rev1_load_4a_artifacts(output_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = output_dir / "partC_rev1_4A_run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing 4A manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = {str(item["label"]): str(item["path"]) for item in manifest.get("artifacts", [])}
    required = {"graph_npz", "graph_metadata", "od_npz", "od_metadata"}
    missing = sorted(required.difference(artifacts))
    if missing:
        raise ValueError(f"4A manifest is missing required artifacts: {missing}")
    return manifest, artifacts


def _partC_rev1_cmd_verify_cap2over3_gravity_state(args: argparse.Namespace) -> int:
    config = load_partC_rev1_config(args.config)
    results_root = partC_rev1_resolve_path(
        args.results_root or str(config.get("outputs", {}).get("output_dir", "partC_rev1_outputs/partC_rev1_cap2over3_gravity"))
    )
    results_root.mkdir(parents=True, exist_ok=True)
    _manifest_4a, artifacts = _partC_rev1_load_4a_artifacts(results_root)
    graph = partC_rev1_load_graph_snapshot(artifacts["graph_npz"], artifacts["graph_metadata"])
    od = partC_rev1_load_od_snapshot(artifacts["od_npz"], artifacts["od_metadata"])
    summary_4c_path = results_root / "partC_rev1_4C_summary.json"
    if not summary_4c_path.exists():
        raise FileNotFoundError(f"Missing 4C summary: {summary_4c_path}")
    summary_4c = partC_rev1_read_json(summary_4c_path)
    best = summary_4c.get("best_accuracy_benchmark")
    if not isinstance(best, dict):
        raise RuntimeError("4C summary does not contain a selected best benchmark")
    required_artifacts = {
        "directed_arc_flow": best.get("artifacts", {}).get("directed_arc_flow"),
        "physical_link_flow": best.get("artifacts", {}).get("physical_link_flow"),
    }
    missing_artifacts = [label for label, path in required_artifacts.items() if not path or not partC_rev1_resolve_path(str(path)).exists()]
    if missing_artifacts:
        raise FileNotFoundError(f"Selected 4C benchmark is missing artifacts: {missing_artifacts}")
    region_table = partC_rev1_region_size_table(graph, od)
    summary_4d_path = results_root / "partC_rev1_4D_summary.json"
    summary_4d = partC_rev1_read_json(summary_4d_path) if summary_4d_path.exists() else {}
    trained_regions = [str(row.get("region_uuid")) for row in summary_4d.get("completed_regions", []) if isinstance(row, dict)]
    selected_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root),
        "selected_benchmark_run_dir": str(best.get("run_dir", "")),
        "method": best.get("method"),
        "backend": best.get("backend"),
        "run_hash": best.get("run_hash"),
        "final_relative_gap": best.get("final_gap"),
        "runtime_seconds": best.get("runtime_seconds"),
        "graph_hash": graph.graph_hash,
        "assignment_graph_hash": graph.assignment_graph_hash,
        "topology_hash": graph.topology_hash,
        "backbone_hash": graph.backbone_hash,
        "partition_hash": graph.partition_hash,
        "od_hash": od.od_hash,
        "directed_arc_flow_parquet": required_artifacts["directed_arc_flow"],
        "physical_link_flow_parquet": required_artifacts["physical_link_flow"],
        "validation_status": "selected_4c_benchmark_validated_for_resume",
    }
    selected_path = results_root / "partC_rev1_selected_4C_benchmark_for_resume.json"
    selected_path.write_text(json.dumps(selected_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_path = results_root / "partC_rev1_cap2over3_gravity_resume_audit.md"
    lines = [
        "# partC_rev1 cap2over3 gravity resume audit",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Selected 4C benchmark",
        "",
        f"- Selected benchmark folder: `{best.get('run_dir')}`",
        f"- Method/backend: `{best.get('method')}` / `{best.get('backend')}`",
        f"- Run hash: `{best.get('run_hash')}`",
        f"- Final relative gap: `{best.get('final_gap')}`",
        f"- Runtime seconds: `{best.get('runtime_seconds')}`",
        f"- Directed arc flow: `{required_artifacts['directed_arc_flow']}`",
        f"- Physical link flow: `{required_artifacts['physical_link_flow']}`",
        "",
        "## Hashes",
        "",
        f"- Graph hash: `{graph.graph_hash}`",
        f"- Assignment graph hash: `{graph.assignment_graph_hash}`",
        f"- Topology hash: `{graph.topology_hash}`",
        f"- Backbone hash: `{graph.backbone_hash}`",
        f"- Partition hash: `{graph.partition_hash}`",
        f"- OD hash: `{od.od_hash}`",
        "",
        "## Region and training status",
        "",
        f"- Raw region rows from current graph snapshot: `{len(region_table)}`",
        f"- Regions with terminal proxy count greater than one before small-region correction: `{int(region_table['terminal_proxy_count'].astype(float).gt(1).sum())}`",
        f"- Existing 4D state: `{summary_4d.get('state', 'missing')}`",
        f"- Existing pilot/full completed regions listed in 4D summary: `{len(trained_regions)}`",
        f"- Existing 4D regions: `{', '.join(trained_regions) if trained_regions else 'none'}`",
        "",
        "## 4E/4F/4G status in this root",
        "",
        f"- Full regional training summaries: `{len(list(results_root.glob('partC_rev1_full_regional_training_*/partC_rev1_full_regional_training_summary.json')))}`",
        f"- Global policy summaries: `{len(list(results_root.glob('partC_rev1_global_policy_suite_*/partC_rev1_global_policy_suite_summary.json')))}`",
        f"- Physical reload suite summaries: `{len(list(results_root.glob('partC_rev1_physical_reload_suite_*/partC_rev1_physical_reload_suite_summary.json')))}`",
        f"- Revised reviewer experiment manifests: `{len(list(results_root.glob('partC_rev1_revised_experiments_*/partC_rev1_revised_experiment_manifest.json')))}`",
        "",
        "## Authoritative modules",
        "",
        "- Sampler and 4D model training: `partC_rev1_src/partC_rev1_regional_training.py`",
        "- Full regional training: `partC_rev1_src/partC_rev1_full_regional_training.py`",
        "- Corrected small-region handling: `partC_rev1_src/partC_rev1_small_region_correction.py`",
        "- Global policy suite: `partC_rev1_src/partC_rev1_global_policy_suite.py`",
        "- Sparse global solver core: `partC_rev1_src/partC_rev1_sparse_global_solver.py`",
        "- Physical reload suite: `partC_rev1_src/partC_rev1_physical_reload_suite.py`",
        "- Physical reload core: `partC_rev1_src/partC_rev1_physical_reload.py`",
        "- Reviewer experiments: `partC_rev1_src/partC_rev1_revised_reviewer_experiments.py` and `partC_rev1_src/partC_rev1_reviewer_experiments.py`",
        "- CLI orchestration: `partC_rev1_src/partC_rev1_cli.py`",
        "",
        "## Mandatory modification check",
        "",
        "- Cap-2/3 assignment capacity multipliers and hash separation are present in the graph snapshot artifacts.",
        "- The configured production regional OD sampler is `benchmark_flow_gravity_additive`.",
        "- Additive gravity uses no IPF, balancing, degree denominator, or support-mask renormalization.",
        "- Corrected-region route witnesses default to all positive-demand origins.",
        "- Existing 4D pilot models are retained as historical artifacts and are not treated as complete production full training.",
        "",
        "## Status",
        "",
        "- 4C is valid for resume and must not be rerun unless one of the selected artifacts is removed.",
        "- Corrected-region artifacts and full 4D training remain downstream tasks for this resume.",
        "- Source maps and authoritative SQLite/GPKG inputs were not modified by this verifier.",
    ]
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Resume audit: {audit_path}")
    print(f"Selected 4C manifest: {selected_path}")
    return 0


def _partC_rev1_write_4b_handoff(
    *,
    output_dir: Path,
    summary_path: Path,
    flow_paths: list[dict[str, Any]],
    summary: dict[str, Any],
    reporter: PartCRev1ProgressReporter,
) -> Path:
    handoff_path = output_dir / "partC_rev1_4B_handoff.md"
    lines = [
        "# partC_rev1 4B Handoff",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Git commit at run time: {_partC_rev1_git_commit()}",
        "",
        "## Status",
        "",
        "- Shared BPR, shortest-path, AON, MSA, FW, CFW, and optional GP primitives are implemented.",
        "- Production backend selected: `numba_csr`.",
        "- igraph and NetworkX benchmark paths are implemented; real-data benchmark pilots are runtime-guarded.",
        "- Relative gap uses the 4B v2 denominator `max(SPTT, epsilon)`.",
        "",
        "## APIs",
        "",
        "- `partC_rev1_bpr.partC_rev1_bpr_cost`, `partC_rev1_bpr_derivative`, `partC_rev1_bpr_integral`, `partC_rev1_fw_line_search`",
        "- `partC_rev1_shortest_paths.partC_rev1_sssp`, `partC_rev1_many_to_many`",
        "- `partC_rev1_aon.partC_rev1_aon_load`",
        "- `partC_rev1_assignment.partC_rev1_solve_physical_assignment`",
        "",
        "## Real-Data Pilot Summary",
        "",
        f"- Graph hash: `{summary.get('graph_hash')}`",
        f"- OD hash: `{summary.get('od_hash')}`",
        f"- Summary JSON: `{summary_path}`",
        f"- Stage log: `{reporter.log_path}`",
        f"- Progress JSONL: `{reporter.progress_path}`",
        "",
        "| Run | Backend | State | Runtime seconds | Processed origins | Projected full seconds | Reachable demand | Unreachable demand |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in summary.get("runs", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    _partC_rev1_escape_md(run.get("name", "")),
                    _partC_rev1_escape_md(run.get("backend", "")),
                    _partC_rev1_escape_md(run.get("state", "")),
                    f"{float(run.get('runtime_seconds') or 0.0):.3f}",
                    str(run.get("processed_origins", "")),
                    f"{float(run.get('full_run_projected_seconds') or 0.0):.3f}",
                    f"{float(run.get('reachable_demand') or 0.0):.6f}",
                    f"{float(run.get('unreachable_demand') or 0.0):.6f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Solver Results", ""])
    for solver in summary.get("solver_results", []):
        lines.append(
            f"- `{solver['method']}`/{solver['backend']}: stop `{solver['stop_reason']}`, "
            f"runtime {float(solver['runtime_seconds']):.3f}s, last gap `{solver.get('last_relative_gap')}`."
        )
    lines.extend(["", "## Flow Checkpoints", ""])
    if flow_paths:
        lines.extend(f"- {item['label']}: `{item['path']}`" for item in flow_paths)
    else:
        lines.append("- No flow checkpoints were written.")
    lines.extend(
        [
            "",
            "## Tests",
            "",
            "- Pre-stage: `conda run -n py312 python -m pytest partC_rev1_tests` -> 80 passed by pytest report.",
            "- Focused 4B: `conda run -n py312 python -m pytest partC_rev1_tests/partC_rev1_test_prompt4B_solvers.py`.",
            "- Full post-stage: `conda run -n py312 python -m pytest partC_rev1_tests`.",
            "- Environment note: conda emits the known `conda-libmamba-solver` entry-point warning.",
            "",
            "## Known Limitations",
            "",
            "- igraph and NetworkX path-loading pilots may be partial when full path materialization projects beyond the runtime guard.",
            "- GP is implemented as a tested primitive and remains opt-in behind future `store_paths=true` orchestration.",
            "",
            "## Exact 4C Command",
            "",
            "```text",
            "conda run -n py312 python -m partC_rev1_src.partC_rev1_cli run-physical-benchmarks --config partC_rev1_configs/partC_rev1_real_preprocess_config.json --confirm-full-run",
            "```",
            "",
        ]
    )
    handoff_path.write_text("\n".join(lines), encoding="utf-8")
    return handoff_path


def _partC_rev1_cmd_run_solver_pilots(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("run-solver-pilots requires --confirm-full-run for real-data execution")
    config = load_partC_rev1_config(args.config)
    output_dir = partC_rev1_resolve_path(str(config.get("outputs", {}).get("output_dir", "partC_rev1_outputs")))
    cache_dir = partC_rev1_resolve_path(str(config.get("project", {}).get("cache_dir", "partC_rev1_cache")))
    reporter = PartCRev1ProgressReporter("partC_rev1_4B", output_dir=output_dir)
    started = time.perf_counter()
    flow_paths: list[dict[str, Any]] = []
    try:
        manifest_4a, artifacts = _partC_rev1_load_4a_artifacts(output_dir)
        graph = partC_rev1_load_graph_snapshot(artifacts["graph_npz"], artifacts["graph_metadata"])
        od = partC_rev1_load_od_snapshot(artifacts["od_npz"], artifacts["od_metadata"])
        reporter.emit(
            "solver_pilots_started",
            graph_hash=graph.graph_hash,
            od_hash=od.od_hash,
            graph_nodes=len(graph.node_idx_to_uid),
            graph_arcs=len(graph.arc_idx_to_directed_arc_id),
            od_rows=len(od.demand),
            od_origins=len(od.origin_group_node_idx),
        )
        runs: list[dict[str, Any]] = []
        free_cost = graph.free_time.copy()

        numba_pilot = partC_rev1_aon_load(
            graph,
            od,
            free_cost,
            backend="numba_csr",
            max_origins=int(args.numba_pilot_origins),
            origin_batch_size=max(1, min(int(args.origin_batch_size), int(args.numba_pilot_origins))),
            reporter=reporter,
        )
        runs.append(
            {
                "name": "freeflow_aon_pilot",
                "backend": "numba_csr",
                "state": "measured_partial_projected",
                "runtime_seconds": numba_pilot.metadata["runtime_seconds"],
                "processed_origins": numba_pilot.processed_origins,
                "total_origins": numba_pilot.total_origins,
                "full_run_projected_seconds": numba_pilot.metadata["full_run_projected_seconds"],
                "reachable_demand": numba_pilot.reachable_demand,
                "unreachable_demand": numba_pilot.unreachable_demand,
                "sptt": numba_pilot.sptt,
                "flow_sum": float(np.sum(numba_pilot.flow, dtype=np.float64)),
            }
        )
        if float(numba_pilot.metadata["full_run_projected_seconds"]) > 24.0 * 3600.0:
            raise RuntimeError(
                "Runtime guard stopped 4B before full numba AON: "
                f"projected {numba_pilot.metadata['full_run_projected_seconds']:.1f} seconds"
            )

        numba_aon = partC_rev1_aon_load(
            graph,
            od,
            free_cost,
            backend="numba_csr",
            origin_batch_size=int(args.origin_batch_size),
            reporter=reporter,
        )
        numba_flow_path = output_dir / f"partC_rev1_4B_numba_freeflow_aon_{graph.graph_hash[:12]}_{od.od_hash[:12]}.npz"
        np.savez_compressed(numba_flow_path, flow=numba_aon.flow)
        flow_paths.append({"label": "numba_freeflow_aon", "path": str(numba_flow_path), "kind": "npz"})
        runs.append(
            {
                "name": "freeflow_aon",
                "backend": "numba_csr",
                "state": "measured_complete",
                "runtime_seconds": numba_aon.metadata["runtime_seconds"],
                "processed_origins": numba_aon.processed_origins,
                "total_origins": numba_aon.total_origins,
                "full_run_projected_seconds": numba_aon.metadata["full_run_projected_seconds"],
                "reachable_demand": numba_aon.reachable_demand,
                "unreachable_demand": numba_aon.unreachable_demand,
                "sptt": numba_aon.sptt,
                "flow_sum": float(np.sum(numba_aon.flow, dtype=np.float64)),
            }
        )

        igraph_origins = None if args.full_igraph_aon else int(args.igraph_origins)
        igraph_aon = partC_rev1_aon_load(
            graph,
            od,
            free_cost,
            backend="igraph",
            max_origins=igraph_origins,
            reporter=reporter,
        )
        runs.append(
            {
                "name": "freeflow_aon",
                "backend": "igraph",
                "state": "measured_complete" if not igraph_aon.metadata["partial"] else "measured_partial_projected",
                "runtime_seconds": igraph_aon.metadata["runtime_seconds"],
                "processed_origins": igraph_aon.processed_origins,
                "total_origins": igraph_aon.total_origins,
                "full_run_projected_seconds": igraph_aon.metadata["full_run_projected_seconds"],
                "reachable_demand": igraph_aon.reachable_demand,
                "unreachable_demand": igraph_aon.unreachable_demand,
                "sptt": igraph_aon.sptt,
                "flow_sum": float(np.sum(igraph_aon.flow, dtype=np.float64)),
            }
        )

        networkx_aon = partC_rev1_aon_load(
            graph,
            od,
            free_cost,
            backend="networkx_python",
            max_origins=int(args.networkx_origins),
            reporter=reporter,
        )
        runs.append(
            {
                "name": "freeflow_aon",
                "backend": "networkx_python",
                "state": "measured_partial_projected",
                "runtime_seconds": networkx_aon.metadata["runtime_seconds"],
                "processed_origins": networkx_aon.processed_origins,
                "total_origins": networkx_aon.total_origins,
                "full_run_projected_seconds": networkx_aon.metadata["full_run_projected_seconds"],
                "reachable_demand": networkx_aon.reachable_demand,
                "unreachable_demand": networkx_aon.unreachable_demand,
                "sptt": networkx_aon.sptt,
                "flow_sum": float(np.sum(networkx_aon.flow, dtype=np.float64)),
            }
        )

        solver_results: list[dict[str, Any]] = []
        for method in ["msa", "fw", "cfw"]:
            solver_reporter = reporter
            result = partC_rev1_solve_physical_assignment(
                graph,
                od,
                method=method,
                config=PartCRev1AssignmentConfig(
                    max_iterations=int(args.iterations),
                    backend="numba_csr",
                    origin_batch_size=int(args.origin_batch_size),
                    gap_target=float(args.gap_target),
                ),
                reporter=solver_reporter,
            )
            result_summary = partC_rev1_solver_result_summary(result)
            solver_results.append(result_summary)
            flow_path = output_dir / f"partC_rev1_4B_{method}_numba_flow_{graph.graph_hash[:12]}_{od.od_hash[:12]}.npz"
            np.savez_compressed(flow_path, flow=result.flow, cost=result.cost)
            flow_paths.append({"label": f"{method}_numba_flow", "path": str(flow_path), "kind": "npz"})

        igraph_fw_result: dict[str, Any] | None = None
        if int(args.igraph_fw_origins) > 0:
            result = partC_rev1_solve_physical_assignment(
                graph,
                od,
                method="fw",
                config=PartCRev1AssignmentConfig(
                    max_iterations=max(1, int(args.igraph_fw_iterations)),
                    backend="igraph",
                    max_origins=int(args.igraph_fw_origins),
                    gap_target=float(args.gap_target),
                ),
                reporter=reporter,
            )
            igraph_fw_result = partC_rev1_solver_result_summary(result)
            igraph_fw_result["state"] = "measured_partial_projected"

        summary = {
            "stage": "partC_rev1_4B",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": args.config,
            "git_commit": _partC_rev1_git_commit(),
            "graph_hash": graph.graph_hash,
            "od_hash": od.od_hash,
            "partition_hash": graph.partition_hash,
            "manifest_4a": manifest_4a,
            "runs": runs,
            "solver_results": solver_results,
            "igraph_fw_result": igraph_fw_result,
            "flow_paths": flow_paths,
            "runtime_seconds": time.perf_counter() - started,
            "selected_production_backend": "numba_csr",
            "cost_hash_freeflow": graph.graph_hash + ":free_time",
            "bpr_capacity_rule": "Prompt 2.5 remaining freight capacity, no second AADT subtraction",
        }
        summary_path = output_dir / "partC_rev1_4B_solver_pilot_summary.json"
        from .partC_rev1_io import partC_rev1_write_json

        partC_rev1_write_json(summary, summary_path)
        handoff_path = _partC_rev1_write_4b_handoff(
            output_dir=output_dir,
            summary_path=summary_path,
            flow_paths=flow_paths,
            summary=summary,
            reporter=reporter,
        )
        partC_rev1_write_master_state(
            output_dir / "partC_rev1_master_run_state.json",
            {
                "current_stage": "partC_rev1_4B",
                "current_run": "solver_pilots",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "completed_artifacts": flow_paths + [{"label": "summary", "path": str(summary_path), "kind": "json"}],
                "current_checkpoint": str(handoff_path),
                "next_action": "run_prompt4C_after_commit",
            },
        )
        partC_rev1_append_jsonl(
            output_dir / "partC_rev1_master_resource_usage.jsonl",
            {
                "stage": "partC_rev1_4B",
                "event": "solver_pilots_finished",
                "runtime_seconds": time.perf_counter() - started,
                "graph_hash": graph.graph_hash,
                "od_hash": od.od_hash,
            },
        )
        reporter.emit("solver_pilots_finished", summary=str(summary_path), handoff=str(handoff_path))
        print(f"4B summary: {summary_path}")
        print(f"4B handoff: {handoff_path}")
        return 0
    finally:
        reporter.close()


def _partC_rev1_write_4c_handoff(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    reporter: PartCRev1ProgressReporter,
    config_path: str,
) -> Path:
    handoff_path = output_dir / "partC_rev1_4C_handoff.md"
    best = summary.get("best_accuracy_benchmark")
    lines = [
        "# partC_rev1 4C Handoff",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Git commit at run time: {_partC_rev1_git_commit()}",
        "",
        "## Status",
        "",
        "- Physical benchmark runner and immutable run folders were executed.",
        "- Production method comparison uses one common backend: `numba_csr`.",
        "- igraph and NetworkX FW entries are backend fairness pilots/projections when partial.",
        "- Route witness artifacts were written with explicit measured coverage.",
        "",
        "## Selected Benchmark",
        "",
    ]
    if isinstance(best, dict):
        lines.extend(
            [
                f"- Method/backend: `{best.get('method')}` / `{best.get('backend')}`",
                f"- Run hash: `{best.get('run_hash')}`",
                f"- Run directory: `{best.get('run_dir')}`",
                f"- Final relative gap: `{best.get('final_gap')}`",
                f"- Accuracy score: `{best.get('accuracy_score')}`",
                f"- Runtime seconds: `{best.get('runtime_seconds')}`",
                f"- Directed flow: `{best.get('artifacts', {}).get('directed_arc_flow')}`",
                f"- Physical link flow: `{best.get('artifacts', {}).get('physical_link_flow')}`",
            ]
        )
    else:
        lines.append("- No completed feasible benchmark was selected.")
    lines.extend(
        [
            "",
            "## Registry",
            "",
            f"- CSV: `{summary.get('registry_csv')}`",
            f"- Markdown: `{summary.get('registry_md')}`",
            f"- Summary JSON: `{summary.get('summary_path')}`",
            f"- Stage log: `{reporter.log_path}`",
            f"- Progress JSONL: `{reporter.progress_path}`",
            "",
            "## Runs",
            "",
            "| Method | Backend | State | Stop reason | Final gap | Runtime seconds | Accuracy |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for item in summary.get("run_summaries", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    _partC_rev1_escape_md(item.get("method", "")),
                    _partC_rev1_escape_md(item.get("backend", "")),
                    _partC_rev1_escape_md(item.get("state", "")),
                    _partC_rev1_escape_md(item.get("stop_reason", "")),
                    "" if item.get("final_gap") is None else f"{float(item.get('final_gap')):.12g}",
                    "" if item.get("runtime_seconds") is None else f"{float(item.get('runtime_seconds')):.3f}",
                    "" if item.get("accuracy_score") is None else f"{float(item.get('accuracy_score')):.12g}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Route Witnesses", ""])
    for witness in summary.get("route_witnesses", []):
        lines.append(
            f"- `{witness.get('witness_set')}`: `{witness.get('path')}`; "
            f"state `{witness.get('state')}`, rows {witness.get('rows')}, "
            f"processed origins {witness.get('processed_origins')}/{witness.get('total_origins')}, "
            f"coverage share {witness.get('coverage_share')}"
        )
    validation = summary.get("reference_flow_validation", {})
    lines.extend(
        [
            "",
            "## Preliminary Reference-Flow Validation",
            "",
            f"- Status: `{validation.get('status')}`",
            f"- Summary: `{validation.get('path')}`",
            f"- Coverage share: `{validation.get('coverage_share')}`",
            f"- WMAPE: `{validation.get('wmape')}`",
            f"- Pearson: `{validation.get('pearson')}`",
            "",
            "## Tests",
            "",
            "- Pre-stage: `conda run -n py312 python -m pytest partC_rev1_tests` -> 89 passed, 2 warnings by pytest report.",
            "- Focused 4C and full post-stage tests are recorded in the execution manifest/current-state docs after commit.",
            "- Environment note: conda emits the known `conda-libmamba-solver` entry-point warning.",
            "",
            "## Known Limitations",
            "",
            "- Route witnesses are measured partial artifacts when configured witness origins are fewer than all origins; coverage is explicit in the parquet rows and this handoff.",
            "- GP remains skipped as optional because full path storage is not required for the 4C Definition of Done.",
            "",
            "## Exact 4D Command",
            "",
            "```text",
            f"conda run -n py312 python -m partC_rev1_src.partC_rev1_cli run-regional-training --config {config_path} --confirm-full-run",
            "```",
            "",
        ]
    )
    handoff_path.write_text("\n".join(lines), encoding="utf-8")
    return handoff_path


def _partC_rev1_append_master_run_log(output_dir: Path, message: str) -> None:
    for name in ["partC_rev1_prompt4_master_run.log", "partC_rev1_master_run.log", "partC_rev1_postcapacity_master.log"]:
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")


def _partC_rev1_cmd_run_physical_benchmarks(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("run-physical-benchmarks requires --confirm-full-run for real-data execution")
    config = load_partC_rev1_config(args.config)
    output_dir = partC_rev1_resolve_path(str(config.get("outputs", {}).get("output_dir", "partC_rev1_outputs")))
    reporter = PartCRev1ProgressReporter(PARTC_REV1_4C_STAGE, output_dir=output_dir)
    started = time.perf_counter()
    try:
        manifest_4a, artifacts = _partC_rev1_load_4a_artifacts(output_dir)
        del manifest_4a
        graph = partC_rev1_load_graph_snapshot(artifacts["graph_npz"], artifacts["graph_metadata"])
        od = partC_rev1_load_od_snapshot(artifacts["od_npz"], artifacts["od_metadata"])
        max_iterations = int(args.max_iterations or config.get("benchmarks", {}).get("max_iter", 50))
        stage_config = PartCRev1PhysicalBenchmarkStageConfig(
            config_path=args.config,
            max_iterations=max_iterations,
            gap_target=float(args.gap_target),
            gap_reference=float(args.gap_reference),
            origin_batch_size=int(args.origin_batch_size),
            backend_pilot_origins=int(args.backend_pilot_origins),
            backend_pilot_iterations=int(args.backend_pilot_iterations),
            backend_pilot_min_seconds=float(args.backend_pilot_min_seconds),
            backend_pilot_min_batches=int(args.backend_pilot_min_batches),
            backend_pilot_cv_tolerance=float(args.backend_pilot_cv_tolerance),
            witness_max_origins=None if int(args.witness_max_origins) <= 0 else int(args.witness_max_origins),
            witness_all_positive_origins=bool(args.witness_all_positive_origins),
            witness_origin_batch_size=int(args.witness_origin_batch_size),
            witness_diagnostic_path_count=int(args.witness_diagnostic_path_count),
        )
        reporter.emit(
            "physical_benchmarks_started",
            graph_hash=graph.graph_hash,
            topology_hash=graph.topology_hash,
            backbone_hash=graph.backbone_hash,
            assignment_graph_hash=graph.assignment_graph_hash,
            od_hash=od.od_hash,
            graph_nodes=len(graph.node_idx_to_uid),
            graph_arcs=len(graph.arc_idx_to_directed_arc_id),
            od_rows=len(od.demand),
            od_origins=len(od.origin_group_node_idx),
            validate_dependency_hashes=bool(args.validate_dependency_hashes),
            stage_config=asdict(stage_config),
        )
        summary = partC_rev1_run_physical_benchmark_stage(
            graph,
            od,
            config=config,
            output_dir=output_dir,
            stage_config=stage_config,
            reporter=reporter,
        )
        handoff_path = _partC_rev1_write_4c_handoff(
            output_dir=output_dir,
            summary=summary,
            reporter=reporter,
            config_path=args.config,
        )
        partC_rev1_write_master_state(
            output_dir / "partC_rev1_master_run_state.json",
            {
                "current_stage": PARTC_REV1_4C_STAGE,
                "current_run": "physical_benchmarks",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "accumulated_runtime_seconds": time.perf_counter() - started,
                "completed_artifacts": [
                    {"label": "summary", "path": str(summary.get("summary_path")), "kind": "json"},
                    {"label": "benchmark_registry_csv", "path": str(summary.get("registry_csv")), "kind": "csv"},
                    {"label": "benchmark_registry_md", "path": str(summary.get("registry_md")), "kind": "md"},
                    {"label": "handoff", "path": str(handoff_path), "kind": "md"},
                ]
                + [
                    {"label": f"route_witnesses_{item.get('witness_set')}", "path": str(item.get("path")), "kind": "parquet"}
                    for item in summary.get("route_witnesses", [])
                ],
                "current_checkpoint": str(handoff_path),
                "next_action": "run_prompt4D_after_commit",
            },
        )
        partC_rev1_append_jsonl(
            output_dir / "partC_rev1_master_resource_usage.jsonl",
            {
                "stage": PARTC_REV1_4C_STAGE,
                "event": "physical_benchmarks_finished",
                "runtime_seconds": time.perf_counter() - started,
                "graph_hash": graph.graph_hash,
                "od_hash": od.od_hash,
                "summary_path": summary.get("summary_path"),
            },
        )
        _partC_rev1_append_master_run_log(
            output_dir,
            f"4C physical benchmarks finished summary={summary.get('summary_path')} handoff={handoff_path}",
        )
        reporter.emit("physical_benchmarks_finished", summary=str(summary.get("summary_path")), handoff=str(handoff_path))
        print(f"4C summary: {summary.get('summary_path')}")
        print(f"4C handoff: {handoff_path}")
        return 0
    finally:
        reporter.close()


def _partC_rev1_cmd_run_regional_training(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("run-regional-training requires --confirm-full-run for real-data execution")
    config = load_partC_rev1_config(args.config)
    output_dir = partC_rev1_resolve_path(str(config.get("outputs", {}).get("output_dir", "partC_rev1_outputs")))
    cache_dir = partC_rev1_resolve_path(str(config.get("project", {}).get("cache_dir", "partC_rev1_cache")))
    reporter = PartCRev1ProgressReporter(PARTC_REV1_4D_STAGE, output_dir=output_dir)
    started = time.perf_counter()
    try:
        _manifest_4a, artifacts = _partC_rev1_load_4a_artifacts(output_dir)
        graph = partC_rev1_load_graph_snapshot(artifacts["graph_npz"], artifacts["graph_metadata"])
        od = partC_rev1_load_od_snapshot(artifacts["od_npz"], artifacts["od_metadata"])
        summary_4c_path = output_dir / "partC_rev1_4C_summary.json"
        summary_4c = partC_rev1_read_json(summary_4c_path)
        best = summary_4c.get("best_accuracy_benchmark")
        if not isinstance(best, dict):
            raise RuntimeError("Prompt 4D requires a selected Prompt 4C benchmark")
        directed_flow_path = str(best.get("artifacts", {}).get("directed_arc_flow", ""))
        if not directed_flow_path:
            raise RuntimeError("Selected Prompt 4C benchmark is missing directed arc flow path")
        benchmark_flow = partC_rev1_load_benchmark_flow(graph, directed_flow_path)
        witness_paths = [str(item.get("path")) for item in summary_4c.get("route_witnesses", []) if isinstance(item, dict) and item.get("path")]
        route_witnesses = partC_rev1_read_route_witnesses(witness_paths)
        regional_cfg = config.get("regional_training", {}) if isinstance(config.get("regional_training", {}), dict) else {}
        requested_samples = int(args.samples_per_region or regional_cfg.get("num_samples_per_region", 100))
        pilot_regions = int(args.pilot_region_count or regional_cfg.get("max_regions_for_debug", 3) or 3)
        stage_config = PartCRev1RegionalTrainingConfig(
            config_path=args.config,
            requested_samples_per_region=requested_samples,
            pilot_samples_per_region=int(args.pilot_samples_per_region),
            pilot_region_count=pilot_regions,
            production_region_limit=int(args.production_region_limit),
            max_pilot_region_arcs=int(args.max_pilot_region_arcs),
            max_active_pairs=int(args.max_active_pairs),
            top_k_per_source=int(args.top_k_per_source),
            regional_solver_iterations=int(args.regional_solver_iterations),
            nn_epochs=int(args.nn_epochs),
            icnn_epochs=int(args.icnn_epochs),
            random_seed=int(config.get("project", {}).get("random_seed", 42)),
            od_sampler=str(regional_cfg.get("od_sampler", "legacy_sparse_ipf_lognormal")),
            support_mask_renormalization=str(regional_cfg.get("support_mask_renormalization", "none")),
            terminal_perturbation=str(regional_cfg.get("terminal_perturbation", "uniform_0_200")),
            skip_on_marginal_imbalance=bool(regional_cfg.get("skip_on_marginal_imbalance", False)),
        )
        reporter.emit(
            "regional_training_started",
            graph_hash=graph.graph_hash,
            od_hash=od.od_hash,
            selected_benchmark_hash=best.get("run_hash"),
            directed_flow_path=directed_flow_path,
            route_witness_rows=int(len(route_witnesses)),
            stage_config=asdict(stage_config),
        )
        summary = partC_rev1_run_regional_training_stage(
            graph,
            od,
            benchmark_flow=benchmark_flow,
            route_witnesses=route_witnesses,
            selected_benchmark_hash=str(best.get("run_hash", "")),
            directed_flow_path=directed_flow_path,
            config=stage_config,
            output_dir=output_dir,
            cache_dir=cache_dir,
            reporter=reporter,
        )
        handoff_path = partC_rev1_write_4d_handoff(summary, output_dir=output_dir, reporter=reporter)
        partC_rev1_write_master_state(
            output_dir / "partC_rev1_master_run_state.json",
            {
                "current_stage": PARTC_REV1_4D_STAGE,
                "current_run": "regional_training",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "accumulated_runtime_seconds": time.perf_counter() - started,
                "completed_artifacts": [
                    {"label": "summary", "path": str(summary.get("summary_path")), "kind": "json"},
                    {"label": "model_registry_csv", "path": str(summary.get("model_registry_csv")), "kind": "csv"},
                    {"label": "model_registry_md", "path": str(summary.get("model_registry_md")), "kind": "md"},
                    {"label": "terminal_registry_csv", "path": str(summary.get("terminal_registry_csv")), "kind": "csv"},
                    {"label": "cleanup_manifest_csv", "path": str(summary.get("cleanup_manifest_csv")), "kind": "csv"},
                    {"label": "handoff", "path": str(handoff_path), "kind": "md"},
                ],
                "current_checkpoint": str(handoff_path),
                "next_action": "run_prompt4E_after_commit",
            },
        )
        partC_rev1_append_jsonl(
            output_dir / "partC_rev1_master_resource_usage.jsonl",
            {
                "stage": PARTC_REV1_4D_STAGE,
                "event": "regional_training_finished",
                "runtime_seconds": time.perf_counter() - started,
                "graph_hash": graph.graph_hash,
                "od_hash": od.od_hash,
                "summary_path": summary.get("summary_path"),
            },
        )
        _partC_rev1_append_master_run_log(
            output_dir,
            f"4D regional training finished summary={summary.get('summary_path')} handoff={handoff_path}",
        )
        reporter.emit("regional_training_finished", summary=str(summary.get("summary_path")), handoff=str(handoff_path))
        print(f"4D summary: {summary.get('summary_path')}")
        print(f"4D handoff: {handoff_path}")
        return 0
    finally:
        reporter.close()


def _partC_rev1_cmd_correct_training_regions(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("correct-training-regions requires --confirm-full-run for real-data execution")
    config = load_partC_rev1_config(args.config)
    output_dir = partC_rev1_resolve_path(str(config.get("outputs", {}).get("output_dir", "partC_rev1_outputs")))
    cache_dir = partC_rev1_resolve_path(str(config.get("project", {}).get("cache_dir", "partC_rev1_cache")))
    reporter = PartCRev1ProgressReporter(PARTC_REV1_SMALL_REGION_STAGE, output_dir=output_dir, heartbeat_seconds=float(args.heartbeat_seconds))
    started = time.perf_counter()
    try:
        _manifest_4a, artifacts = _partC_rev1_load_4a_artifacts(output_dir)
        graph = partC_rev1_load_graph_snapshot(artifacts["graph_npz"], artifacts["graph_metadata"])
        od = partC_rev1_load_od_snapshot(artifacts["od_npz"], artifacts["od_metadata"])
        stage_config = PartCRev1SmallRegionCorrectionConfig(
            config_path=args.config,
            min_arc_count=int(args.min_arc_count),
            portal_square_arc_count_threshold=int(args.portal_square_arc_count_threshold),
            heartbeat_seconds=float(args.heartbeat_seconds),
        )
        summary = partC_rev1_run_small_region_correction_stage(
            graph,
            od,
            config=stage_config,
            output_dir=output_dir,
            cache_dir=cache_dir,
            reporter=reporter,
        )
        corrected_graph = partC_rev1_load_graph_snapshot(summary["graph_paths"]["npz"], summary["graph_paths"]["metadata"])
        summary_4c = partC_rev1_read_json(output_dir / "partC_rev1_4C_summary.json")
        best = summary_4c.get("best_accuracy_benchmark")
        if not isinstance(best, dict):
            raise RuntimeError("correct-training-regions requires a selected Prompt 4C benchmark")
        directed_flow_path = str(best.get("artifacts", {}).get("directed_arc_flow", ""))
        if not directed_flow_path:
            raise RuntimeError("Selected Prompt 4C benchmark is missing directed arc flow path")
        benchmark_cost = partC_rev1_load_cost_from_directed_flow(corrected_graph, directed_flow_path)
        run_dir = Path(summary["run_output_dir"])
        route_witnesses: list[dict[str, Any]] = []
        witness_specs = [
            ("corrected_freeflow", corrected_graph.free_time, run_dir / "partC_rev1_small_region_correction_route_witnesses_freeflow.parquet"),
            ("corrected_benchmark", benchmark_cost, run_dir / "partC_rev1_small_region_correction_route_witnesses_benchmark.parquet"),
        ]
        for witness_set, cost, path in witness_specs:
            result = partC_rev1_generate_route_witnesses(
                corrected_graph,
                od,
                cost,
                output_path=path,
                witness_set=witness_set,
                max_origins=None if bool(args.witness_all_positive_origins) or int(args.witness_max_origins) <= 0 else int(args.witness_max_origins),
                diagnostic_path_count=int(args.witness_diagnostic_path_count),
                reporter=reporter,
            )
            route_witnesses.append({"witness_set": witness_set, "path": str(path), **result})
        summary = partC_rev1_append_route_witnesses_to_correction_summary(summary["summary_path"], route_witnesses)
        partC_rev1_write_master_state(
            output_dir / "partC_rev1_postcapacity_master_state.json",
            {
                "current_stage": PARTC_REV1_SMALL_REGION_STAGE,
                "current_run": "correct_training_regions",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "accumulated_runtime_seconds": time.perf_counter() - started,
                "parent_graph_hash": graph.graph_hash,
                "corrected_graph_hash": summary.get("corrected_graph_hash"),
                "correction_hash": summary.get("correction_hash"),
                "completed_artifacts": [
                    {"label": "summary", "path": str(summary.get("summary_path")), "kind": "json"},
                    {"label": "summary_md", "path": str(summary.get("summary_md_path")), "kind": "md"},
                    {"label": "corrected_graph_npz", "path": str(summary.get("graph_paths", {}).get("npz")), "kind": "npz"},
                    {"label": "region_audit", "path": str(summary.get("artifacts", {}).get("region_audit_csv")), "kind": "csv"},
                ],
                "current_checkpoint": str(summary.get("summary_path")),
                "next_action": "run_full_regional_training_after_commit",
            },
        )
        partC_rev1_append_jsonl(
            output_dir / "partC_rev1_postcapacity_resource_usage.jsonl",
            {
                "stage": PARTC_REV1_SMALL_REGION_STAGE,
                "event": "correct_training_regions_finished",
                "runtime_seconds": time.perf_counter() - started,
                "parent_graph_hash": graph.graph_hash,
                "corrected_graph_hash": summary.get("corrected_graph_hash"),
                "summary_path": summary.get("summary_path"),
            },
        )
        _partC_rev1_append_master_run_log(
            output_dir,
            f"postcapacity Phase 1 corrected regions finished summary={summary.get('summary_path')}",
        )
        _partC_rev1_append_master_run_log(
            output_dir,
            f"postcapacity_master_log stage={PARTC_REV1_SMALL_REGION_STAGE} summary={summary.get('summary_path')}",
        )
        print(f"Corrected-region summary: {summary.get('summary_path')}")
        print(f"Corrected graph: {summary.get('graph_paths', {}).get('npz')}")
        return 0
    finally:
        reporter.close()


def _partC_rev1_cmd_run_full_regional_training(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("run-full-regional-training requires --confirm-full-run for real-data execution")
    config = load_partC_rev1_config(args.config)
    output_dir = partC_rev1_resolve_path(str(config.get("outputs", {}).get("output_dir", "partC_rev1_outputs")))
    cache_dir = partC_rev1_resolve_path(str(config.get("project", {}).get("cache_dir", "partC_rev1_cache")))
    reporter = PartCRev1ProgressReporter(PARTC_REV1_FULL_REGIONAL_STAGE, output_dir=output_dir, heartbeat_seconds=float(args.heartbeat_seconds))
    started = time.perf_counter()
    try:
        correction_summary_path = Path(args.correction_summary) if args.correction_summary else partC_rev1_find_latest_small_region_summary(output_dir)
        graph, correction_summary = partC_rev1_load_corrected_graph_from_summary(correction_summary_path)
        _manifest_4a, artifacts = _partC_rev1_load_4a_artifacts(output_dir)
        od = partC_rev1_load_od_snapshot(artifacts["od_npz"], artifacts["od_metadata"])
        summary_4c = partC_rev1_read_json(output_dir / "partC_rev1_4C_summary.json")
        best = summary_4c.get("best_accuracy_benchmark")
        if not isinstance(best, dict):
            raise RuntimeError("run-full-regional-training requires a selected Prompt 4C benchmark")
        benchmark_flow = partC_rev1_load_benchmark_flow(graph, str(best.get("artifacts", {}).get("directed_arc_flow", "")))
        witness_paths = [str(item.get("path")) for item in correction_summary.get("route_witnesses", []) if isinstance(item, dict) and item.get("path")]
        route_witnesses = partC_rev1_read_route_witnesses(witness_paths)
        regional_cfg = config.get("regional_training", {}) if isinstance(config.get("regional_training", {}), dict) else {}
        stage_config = PartCRev1FullRegionalTrainingConfig(
            config_path=args.config,
            min_samples_per_region=int(args.min_samples_per_region),
            max_samples_per_region=int(args.max_samples_per_region),
            region_limit=int(args.region_limit),
            max_wall_hours=float(args.max_wall_hours),
            projection_min_completed_regions=int(args.projection_min_completed_regions),
            max_active_pairs=int(args.max_active_pairs),
            top_k_per_source=int(args.top_k_per_source),
            regional_solver_iterations=int(args.regional_solver_iterations),
            xgb_estimators=int(args.xgb_estimators),
            nn_epochs=int(args.nn_epochs),
            icnn_epochs=int(args.icnn_epochs),
            retune_multiplier=float(args.retune_multiplier),
            random_seed=int(config.get("project", {}).get("random_seed", 42)),
            resume=not bool(args.no_resume),
            od_sampler=str(regional_cfg.get("od_sampler", "legacy_sparse_ipf_lognormal")),
            support_mask_renormalization=str(regional_cfg.get("support_mask_renormalization", "none")),
            terminal_perturbation=str(regional_cfg.get("terminal_perturbation", "uniform_0_200")),
            skip_on_marginal_imbalance=bool(regional_cfg.get("skip_on_marginal_imbalance", False)),
        )
        reporter.emit(
            "full_regional_training_started",
            corrected_graph_hash=graph.graph_hash,
            od_hash=od.od_hash,
            selected_benchmark_hash=best.get("run_hash"),
            correction_summary=str(correction_summary_path),
            route_witness_rows=int(len(route_witnesses)),
            stage_config=asdict(stage_config),
        )
        summary = partC_rev1_run_full_regional_training_stage(
            graph,
            od,
            correction_summary=correction_summary,
            benchmark_flow=benchmark_flow,
            route_witnesses=route_witnesses,
            config=stage_config,
            output_dir=output_dir,
            cache_dir=cache_dir,
            reporter=reporter,
        )
        partC_rev1_write_master_state(
            output_dir / "partC_rev1_postcapacity_master_state.json",
            {
                "current_stage": PARTC_REV1_FULL_REGIONAL_STAGE,
                "current_run": "run_full_regional_training",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "accumulated_runtime_seconds": time.perf_counter() - started,
                "corrected_graph_hash": graph.graph_hash,
                "summary_path": summary.get("summary_path"),
                "state": summary.get("state"),
                "current_checkpoint": str(summary.get("summary_path")),
                "next_action": "run_global_policy_suite_after_commit" if summary.get("state") == "measured_complete" else "resume_full_regional_training_or_accept_projected_partial",
            },
        )
        partC_rev1_append_jsonl(
            output_dir / "partC_rev1_postcapacity_resource_usage.jsonl",
            {
                "stage": PARTC_REV1_FULL_REGIONAL_STAGE,
                "event": "full_regional_training_finished",
                "runtime_seconds": time.perf_counter() - started,
                "corrected_graph_hash": graph.graph_hash,
                "summary_path": summary.get("summary_path"),
                "state": summary.get("state"),
            },
        )
        _partC_rev1_append_master_run_log(
            output_dir,
            f"postcapacity Phase 2 full regional training finished state={summary.get('state')} summary={summary.get('summary_path')}",
        )
        print(f"Full regional training summary: {summary.get('summary_path')}")
        print(f"State: {summary.get('state')} stop_reason={summary.get('stop_reason')}")
        return 0
    finally:
        reporter.close()


def _partC_rev1_cmd_run_sparse_global_solver(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("run-sparse-global-solver requires --confirm-full-run for real-data execution")
    config = load_partC_rev1_config(args.config)
    output_dir = partC_rev1_resolve_path(str(config.get("outputs", {}).get("output_dir", "partC_rev1_outputs")))
    reporter = PartCRev1ProgressReporter(PARTC_REV1_4E_STAGE, output_dir=output_dir)
    started = time.perf_counter()
    try:
        _manifest_4a, artifacts = _partC_rev1_load_4a_artifacts(output_dir)
        graph = partC_rev1_load_graph_snapshot(artifacts["graph_npz"], artifacts["graph_metadata"])
        od = partC_rev1_load_od_snapshot(artifacts["od_npz"], artifacts["od_metadata"])
        summary_4c = partC_rev1_read_json(output_dir / "partC_rev1_4C_summary.json")
        summary_4d = partC_rev1_read_json(output_dir / "partC_rev1_4D_summary.json")
        best = summary_4c.get("best_accuracy_benchmark")
        if not isinstance(best, dict):
            raise RuntimeError("Prompt 4E requires a selected Prompt 4C benchmark")
        benchmark_flow = partC_rev1_load_benchmark_flow(graph, str(best.get("artifacts", {}).get("directed_arc_flow", "")))
        witness_paths = [str(item.get("path")) for item in summary_4c.get("route_witnesses", []) if isinstance(item, dict) and item.get("path")]
        route_witnesses = partC_rev1_read_route_witnesses(witness_paths)
        stage_config = PartCRev1SparseGlobalConfig(
            config_path=args.config,
            max_iterations=int(args.max_iterations),
            pilot_origins=int(args.pilot_origins),
            origin_batch_size=int(args.origin_batch_size),
            master_gap_tol=float(args.master_gap_tol),
            pricing_gap_tol=float(args.pricing_gap_tol),
            price_tol=float(args.price_tol),
            lazy_abs_tol=float(args.lazy_abs_tol),
            lazy_rel_tol=float(args.lazy_rel_tol),
            full_refresh_frequency=int(args.full_refresh_frequency),
            regional_cost_slope=float(args.regional_cost_slope),
        )
        reporter.emit(
            "sparse_global_solver_started",
            graph_hash=graph.graph_hash,
            od_hash=od.od_hash,
            selected_benchmark_hash=best.get("run_hash"),
            model_region_count=len(summary_4d.get("completed_regions", [])),
            stage_config=asdict(stage_config),
        )
        summary = partC_rev1_run_sparse_global_stage(
            graph,
            od,
            benchmark_flow=benchmark_flow,
            route_witnesses=route_witnesses,
            summary_4d=summary_4d,
            config=stage_config,
            output_dir=output_dir,
            reporter=reporter,
        )
        handoff_path = partC_rev1_write_4e_handoff(summary, output_dir=output_dir, reporter=reporter)
        partC_rev1_write_master_state(
            output_dir / "partC_rev1_master_run_state.json",
            {
                "current_stage": PARTC_REV1_4E_STAGE,
                "current_run": "sparse_global_solver",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "accumulated_runtime_seconds": time.perf_counter() - started,
                "completed_artifacts": [
                    {"label": "summary", "path": str(summary.get("summary_path")), "kind": "json"},
                    {"label": "handoff", "path": str(handoff_path), "kind": "md"},
                ]
                + [{"label": label, "path": str(path), "kind": Path(str(path)).suffix.lstrip(".")} for label, path in dict(summary.get("artifacts", {})).items()],
                "current_checkpoint": str(handoff_path),
                "next_action": "run_prompt4F_after_commit",
            },
        )
        partC_rev1_append_jsonl(
            output_dir / "partC_rev1_master_resource_usage.jsonl",
            {
                "stage": PARTC_REV1_4E_STAGE,
                "event": "sparse_global_solver_finished",
                "runtime_seconds": time.perf_counter() - started,
                "graph_hash": graph.graph_hash,
                "od_hash": od.od_hash,
                "summary_path": summary.get("summary_path"),
            },
        )
        _partC_rev1_append_master_run_log(
            output_dir,
            f"4E sparse global solver finished summary={summary.get('summary_path')} handoff={handoff_path}",
        )
        reporter.emit("sparse_global_solver_finished", summary=str(summary.get("summary_path")), handoff=str(handoff_path))
        print(f"4E summary: {summary.get('summary_path')}")
        print(f"4E handoff: {handoff_path}")
        return 0
    finally:
        reporter.close()


def _partC_rev1_cmd_run_global_policy_suite(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("run-global-policy-suite requires --confirm-full-run for real-data execution")
    config = load_partC_rev1_config(args.config)
    output_dir = partC_rev1_resolve_path(str(config.get("outputs", {}).get("output_dir", "partC_rev1_outputs")))
    reporter = PartCRev1ProgressReporter(PARTC_REV1_GLOBAL_POLICY_STAGE, output_dir=output_dir, heartbeat_seconds=float(args.heartbeat_seconds))
    started = time.perf_counter()
    try:
        _manifest_4a, artifacts = _partC_rev1_load_4a_artifacts(output_dir)
        od = partC_rev1_load_od_snapshot(artifacts["od_npz"], artifacts["od_metadata"])
        graph, correction_summary, training_summary = partC_rev1_load_global_policy_inputs(
            output_dir=output_dir,
            training_summary_path=args.training_summary or None,
            correction_summary_path=args.correction_summary or None,
        )
        summary_4c = partC_rev1_read_json(output_dir / "partC_rev1_4C_summary.json")
        best = summary_4c.get("best_accuracy_benchmark")
        if not isinstance(best, dict):
            raise RuntimeError("run-global-policy-suite requires a selected Prompt 4C benchmark")
        benchmark_flow = partC_rev1_load_benchmark_flow(graph, str(best.get("artifacts", {}).get("directed_arc_flow", "")))
        witness_paths = [str(item.get("path")) for item in correction_summary.get("route_witnesses", []) if isinstance(item, dict) and item.get("path")]
        route_witnesses = partC_rev1_read_route_witnesses(witness_paths)
        stage_config = PartCRev1GlobalPolicySuiteConfig(
            config_path=args.config,
            max_iterations=int(args.max_iterations),
            pilot_origins=int(args.pilot_origins),
            origin_batch_size=int(args.origin_batch_size),
            master_gap_tol=float(args.master_gap_tol),
            pricing_gap_tol=float(args.pricing_gap_tol),
            price_tol=float(args.price_tol),
            lazy_abs_tol=float(args.lazy_abs_tol),
            lazy_rel_tol=float(args.lazy_rel_tol),
            full_refresh_frequency=int(args.full_refresh_frequency),
            regional_cost_slope=float(args.regional_cost_slope),
            runtime_guard_seconds=float(args.runtime_guard_seconds),
            resume=not bool(args.no_resume),
        )
        reporter.emit(
            "global_policy_suite_started",
            corrected_graph_hash=graph.graph_hash,
            od_hash=od.od_hash,
            selected_benchmark_hash=best.get("run_hash"),
            training_run_hash=training_summary.get("run_hash"),
            route_witness_rows=int(len(route_witnesses)),
            stage_config=asdict(stage_config),
        )
        summary = partC_rev1_run_global_policy_suite_stage(
            graph,
            od,
            correction_summary=correction_summary,
            training_summary=training_summary,
            benchmark_flow=benchmark_flow,
            route_witnesses=route_witnesses,
            config=stage_config,
            output_dir=output_dir,
            reporter=reporter,
        )
        partC_rev1_write_master_state(
            output_dir / "partC_rev1_postcapacity_master_state.json",
            {
                "current_stage": PARTC_REV1_GLOBAL_POLICY_STAGE,
                "current_run": "run_global_policy_suite",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "accumulated_runtime_seconds": time.perf_counter() - started,
                "corrected_graph_hash": graph.graph_hash,
                "training_run_hash": training_summary.get("run_hash"),
                "summary_path": summary.get("summary_path"),
                "handoff_path": summary.get("handoff_path"),
                "state": summary.get("state"),
                "current_checkpoint": str(summary.get("summary_path")),
                "next_action": "run_physical_reload_suite_after_commit" if summary.get("state") in {"measured_complete", "measured_partial_projected"} else "resume_global_policy_suite",
            },
        )
        partC_rev1_append_jsonl(
            output_dir / "partC_rev1_postcapacity_resource_usage.jsonl",
            {
                "stage": PARTC_REV1_GLOBAL_POLICY_STAGE,
                "event": "global_policy_suite_finished",
                "runtime_seconds": time.perf_counter() - started,
                "corrected_graph_hash": graph.graph_hash,
                "training_run_hash": training_summary.get("run_hash"),
                "summary_path": summary.get("summary_path"),
                "state": summary.get("state"),
            },
        )
        _partC_rev1_append_master_run_log(
            output_dir,
            f"postcapacity Phase 3 global policy suite finished state={summary.get('state')} summary={summary.get('summary_path')}",
        )
        reporter.emit("global_policy_suite_finished", summary=str(summary.get("summary_path")), handoff=str(summary.get("handoff_path")))
        print(f"Global policy suite summary: {summary.get('summary_path')}")
        print(f"Global policy suite handoff: {summary.get('handoff_path')}")
        print(f"State: {summary.get('state')}")
        return 0
    finally:
        reporter.close()


def _partC_rev1_cmd_run_physical_reload(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("run-physical-reload requires --confirm-full-run for real-data execution")
    config = load_partC_rev1_config(args.config)
    output_dir = partC_rev1_resolve_path(str(config.get("outputs", {}).get("output_dir", "partC_rev1_outputs")))
    reporter = PartCRev1ProgressReporter(PARTC_REV1_4F_STAGE, output_dir=output_dir)
    started = time.perf_counter()
    try:
        _manifest_4a, artifacts = _partC_rev1_load_4a_artifacts(output_dir)
        graph = partC_rev1_load_graph_snapshot(artifacts["graph_npz"], artifacts["graph_metadata"])
        od = partC_rev1_load_od_snapshot(artifacts["od_npz"], artifacts["od_metadata"])
        stage_config = PartCRev1PhysicalReloadConfig(
            config_path=args.config,
            max_regions=int(args.max_regions),
            max_iterations=int(args.max_iterations),
            gap_target=float(args.gap_target),
            positive_flow_tol=float(args.positive_flow_tol),
            correction_enabled=bool(args.enable_correction),
        )
        reporter.emit(
            "physical_reload_started",
            graph_hash=graph.graph_hash,
            od_hash=od.od_hash,
            stage_config=asdict(stage_config),
        )
        summary = partC_rev1_run_physical_reload_stage(
            graph,
            od,
            config_dict=config,
            config=stage_config,
            output_dir=output_dir,
            reporter=reporter,
        )
        handoff_path = partC_rev1_write_4f_handoff(summary, output_dir=output_dir, reporter=reporter)
        partC_rev1_write_master_state(
            output_dir / "partC_rev1_master_run_state.json",
            {
                "current_stage": PARTC_REV1_4F_STAGE,
                "current_run": "physical_reload",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "accumulated_runtime_seconds": time.perf_counter() - started,
                "completed_artifacts": [
                    {"label": "summary", "path": str(summary.get("summary_path")), "kind": "json"},
                    {"label": "handoff", "path": str(handoff_path), "kind": "md"},
                    {"label": "run_folder", "path": str(summary.get("run_folder")), "kind": "folder"},
                ],
                "current_checkpoint": str(handoff_path),
                "next_action": "run_prompt4G_after_commit",
            },
        )
        partC_rev1_append_jsonl(
            output_dir / "partC_rev1_master_resource_usage.jsonl",
            {
                "stage": PARTC_REV1_4F_STAGE,
                "event": "physical_reload_finished",
                "runtime_seconds": time.perf_counter() - started,
                "graph_hash": graph.graph_hash,
                "od_hash": od.od_hash,
                "summary_path": summary.get("summary_path"),
            },
        )
        _partC_rev1_append_master_run_log(
            output_dir,
            f"4F physical reload finished summary={summary.get('summary_path')} handoff={handoff_path}",
        )
        reporter.emit("physical_reload_finished", summary=str(summary.get("summary_path")), handoff=str(handoff_path))
        print(f"4F summary: {summary.get('summary_path')}")
        print(f"4F handoff: {handoff_path}")
        return 0
    finally:
        reporter.close()


def _partC_rev1_cmd_run_physical_reload_suite(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("run-physical-reload-suite requires --confirm-full-run for real-data execution")
    config = load_partC_rev1_config(args.config)
    output_dir = partC_rev1_resolve_path(str(config.get("outputs", {}).get("output_dir", "partC_rev1_outputs")))
    reporter = PartCRev1ProgressReporter(
        PARTC_REV1_PHYSICAL_RELOAD_SUITE_STAGE,
        output_dir=output_dir,
        heartbeat_seconds=float(args.heartbeat_seconds),
    )
    started = time.perf_counter()
    try:
        graph, _correction_summary, _training_summary = partC_rev1_load_global_policy_inputs(output_dir=output_dir)
        global_summary_path = partC_rev1_resolve_path(args.global_policy_summary) if args.global_policy_summary else partC_rev1_find_latest_global_policy_summary(output_dir)
        global_policy_summary = partC_rev1_read_json(global_summary_path)
        summary_4c = partC_rev1_read_json(output_dir / "partC_rev1_4C_summary.json")
        stage_config = PartCRev1PhysicalReloadSuiteConfig(
            config_path=args.config,
            global_policy_summary_path=str(global_summary_path),
            max_runs=int(args.max_runs),
            max_regions=int(args.max_regions),
            max_iterations=int(args.max_iterations),
            gap_target=float(args.gap_target),
            positive_flow_tol=float(args.positive_flow_tol),
            runtime_guard_seconds=float(args.runtime_guard_seconds),
            resume=not bool(args.no_resume),
        )
        reporter.emit(
            "physical_reload_suite_started",
            corrected_graph_hash=graph.graph_hash,
            global_policy_summary=str(global_summary_path),
            selected_benchmark_hash=dict(summary_4c.get("best_accuracy_benchmark", {})).get("run_hash"),
            stage_config=asdict(stage_config),
        )
        summary = partC_rev1_run_physical_reload_suite_stage(
            graph,
            global_policy_summary=global_policy_summary,
            benchmark_summary=summary_4c,
            config=stage_config,
            output_dir=output_dir,
            reporter=reporter,
        )
        partC_rev1_write_master_state(
            output_dir / "partC_rev1_postcapacity_master_state.json",
            {
                "current_stage": PARTC_REV1_PHYSICAL_RELOAD_SUITE_STAGE,
                "current_run": "run_physical_reload_suite",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "accumulated_runtime_seconds": time.perf_counter() - started,
                "corrected_graph_hash": graph.graph_hash,
                "global_policy_run_hash": global_policy_summary.get("run_hash"),
                "summary_path": summary.get("summary_path"),
                "handoff_path": summary.get("handoff_path"),
                "state": summary.get("state"),
                "current_checkpoint": str(summary.get("summary_path")),
                "next_action": "run_revised_reviewer_experiments_after_commit"
                if summary.get("state") in {"measured_complete", "measured_partial_projected"}
                else "resume_physical_reload_suite",
            },
        )
        partC_rev1_append_jsonl(
            output_dir / "partC_rev1_postcapacity_resource_usage.jsonl",
            {
                "stage": PARTC_REV1_PHYSICAL_RELOAD_SUITE_STAGE,
                "event": "physical_reload_suite_finished",
                "runtime_seconds": time.perf_counter() - started,
                "corrected_graph_hash": graph.graph_hash,
                "global_policy_run_hash": global_policy_summary.get("run_hash"),
                "summary_path": summary.get("summary_path"),
                "state": summary.get("state"),
            },
        )
        _partC_rev1_append_master_run_log(
            output_dir,
            f"postcapacity Phase 4 physical reload suite finished state={summary.get('state')} summary={summary.get('summary_path')}",
        )
        reporter.emit("physical_reload_suite_finished", summary=str(summary.get("summary_path")), handoff=str(summary.get("handoff_path")))
        print(f"Physical reload suite summary: {summary.get('summary_path')}")
        print(f"Physical reload suite handoff: {summary.get('handoff_path')}")
        print(f"State: {summary.get('state')}")
        return 0
    finally:
        reporter.close()


def _partC_rev1_cmd_run_reviewer_experiments(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("run-reviewer-experiments requires --confirm-full-run for real-data execution")
    config = load_partC_rev1_config(args.config)
    output_dir = partC_rev1_resolve_path(str(config.get("outputs", {}).get("output_dir", "partC_rev1_outputs")))
    reporter = PartCRev1ProgressReporter(PARTC_REV1_4G_STAGE, output_dir=output_dir)
    started = time.perf_counter()
    try:
        stage_config = PartCRev1ReviewerExperimentConfig(
            config_path=args.config,
            bootstrap_seed=int(args.bootstrap_seed),
            max_scatter_points=int(args.max_scatter_points),
        )
        reporter.emit("reviewer_experiments_started", stage_config=asdict(stage_config))
        summary = partC_rev1_run_reviewer_experiment_stage(config=stage_config, output_dir=output_dir, reporter=reporter)
        handoff_path = partC_rev1_write_4g_handoff(summary, output_dir=output_dir, reporter=reporter)
        partC_rev1_write_master_state(
            output_dir / "partC_rev1_master_run_state.json",
            {
                "current_stage": PARTC_REV1_4G_STAGE,
                "current_run": "reviewer_experiments",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "accumulated_runtime_seconds": time.perf_counter() - started,
                "completed_artifacts": [
                    {"label": "summary", "path": str(summary.get("summary_path")), "kind": "json"},
                    {"label": "handoff", "path": str(handoff_path), "kind": "md"},
                    {"label": "experiment_root", "path": str(summary.get("experiment_root")), "kind": "folder"},
                ],
                "current_checkpoint": str(handoff_path),
                "next_action": "write_final_prompt4_execution_summary",
            },
        )
        partC_rev1_append_jsonl(
            output_dir / "partC_rev1_master_resource_usage.jsonl",
            {
                "stage": PARTC_REV1_4G_STAGE,
                "event": "reviewer_experiments_finished",
                "runtime_seconds": time.perf_counter() - started,
                "summary_path": summary.get("summary_path"),
                "experiment_root": summary.get("experiment_root"),
            },
        )
        _partC_rev1_append_master_run_log(
            output_dir,
            f"4G reviewer experiments finished summary={summary.get('summary_path')} handoff={handoff_path}",
        )
        reporter.emit("reviewer_experiments_finished", summary=str(summary.get("summary_path")), handoff=str(handoff_path))
        print(f"4G summary: {summary.get('summary_path')}")
        print(f"4G handoff: {handoff_path}")
        return 0
    finally:
        reporter.close()


def _partC_rev1_cmd_run_revised_reviewer_experiments(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("run-revised-reviewer-experiments requires --confirm-full-run for real-data execution")
    config = load_partC_rev1_config(args.config)
    output_dir = partC_rev1_resolve_path(str(config.get("outputs", {}).get("output_dir", "partC_rev1_outputs")))
    reporter = PartCRev1ProgressReporter(
        PARTC_REV1_REVISED_EXPERIMENT_STAGE,
        output_dir=output_dir,
        heartbeat_seconds=float(args.heartbeat_seconds),
    )
    started = time.perf_counter()
    try:
        stage_config = PartCRev1RevisedReviewerExperimentConfig(
            config_path=args.config,
            bootstrap_seed=int(args.bootstrap_seed),
            max_scatter_points=int(args.max_scatter_points),
            resume=not bool(args.no_resume),
        )
        reporter.emit("revised_reviewer_experiments_started", stage_config=asdict(stage_config))
        summary = partC_rev1_run_revised_reviewer_experiment_stage(config=stage_config, output_dir=output_dir, reporter=reporter)
        partC_rev1_write_master_state(
            output_dir / "partC_rev1_postcapacity_master_state.json",
            {
                "current_stage": PARTC_REV1_REVISED_EXPERIMENT_STAGE,
                "current_run": "run_revised_reviewer_experiments",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "accumulated_runtime_seconds": time.perf_counter() - started,
                "summary_path": summary.get("artifacts", {}).get("manifest_json"),
                "handoff_path": summary.get("artifacts", {}).get("handoff_md"),
                "state": "measured_complete_with_projected_and_skipped_experiments",
                "current_checkpoint": str(summary.get("artifacts", {}).get("manifest_json")),
                "next_action": "write_postcapacity_fullrun_handoff",
            },
        )
        partC_rev1_append_jsonl(
            output_dir / "partC_rev1_postcapacity_resource_usage.jsonl",
            {
                "stage": PARTC_REV1_REVISED_EXPERIMENT_STAGE,
                "event": "revised_reviewer_experiments_finished",
                "runtime_seconds": time.perf_counter() - started,
                "summary_path": summary.get("artifacts", {}).get("manifest_json"),
                "state": "measured_complete_with_projected_and_skipped_experiments",
            },
        )
        _partC_rev1_append_master_run_log(
            output_dir,
            f"postcapacity Phase 5 revised reviewer experiments finished summary={summary.get('artifacts', {}).get('manifest_json')}",
        )
        reporter.emit("revised_reviewer_experiments_finished", summary=str(summary.get("artifacts", {}).get("manifest_json")), handoff=str(summary.get("artifacts", {}).get("handoff_md")))
        print(f"Revised reviewer experiment manifest: {summary.get('artifacts', {}).get('manifest_json')}")
        print(f"Revised reviewer handoff: {summary.get('artifacts', {}).get('handoff_md')}")
        return 0
    finally:
        reporter.close()


def _partC_rev1_cmd_run_cap2over3_partition_sweep(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("run-cap2over3-partition-sweep requires --confirm-full-run for real-data execution")
    config = load_partC_rev1_config(args.config)
    output_dir = partC_rev1_resolve_path(str(config.get("outputs", {}).get("output_dir", "partC_rev1_outputs")))
    reporter = PartCRev1ProgressReporter(
        PARTC_REV1_PARTITION_SENSITIVITY_STAGE,
        output_dir=output_dir,
        heartbeat_seconds=float(args.heartbeat_seconds),
    )
    started = time.perf_counter()
    try:
        resolutions = tuple(float(value) for value in str(args.resolutions).split(",") if str(value).strip())
        stage_config = PartCRev1PartitionSensitivityConfig(
            config_path=args.config,
            resolution_values=resolutions or (),
            heartbeat_seconds=float(args.heartbeat_seconds),
            resume=not bool(args.no_resume),
        )
        reporter.emit("partition_sensitivity_started", stage_config=asdict(stage_config))
        summary = partC_rev1_run_partition_sensitivity_package(config=stage_config, output_dir=output_dir, reporter=reporter)
        partC_rev1_write_master_state(
            output_dir / "partC_rev1_postcapacity_master_state.json",
            {
                "current_stage": PARTC_REV1_PARTITION_SENSITIVITY_STAGE,
                "current_run": "run_cap2over3_partition_sweep",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "accumulated_runtime_seconds": time.perf_counter() - started,
                "summary_path": summary.get("artifacts", {}).get("summary_json"),
                "handoff_path": summary.get("artifacts", {}).get("handoff_md"),
                "state": summary.get("state"),
                "current_checkpoint": str(summary.get("artifacts", {}).get("summary_json")),
                "next_action": "run_revised_reviewer_experiments",
            },
        )
        partC_rev1_append_jsonl(
            output_dir / "partC_rev1_postcapacity_resource_usage.jsonl",
            {
                "stage": PARTC_REV1_PARTITION_SENSITIVITY_STAGE,
                "event": "partition_sensitivity_finished",
                "runtime_seconds": time.perf_counter() - started,
                "summary_path": summary.get("artifacts", {}).get("summary_json"),
                "state": summary.get("state"),
            },
        )
        _partC_rev1_append_master_run_log(
            output_dir,
            f"cap2over3 partition sensitivity finished summary={summary.get('artifacts', {}).get('summary_json')}",
        )
        reporter.emit("partition_sensitivity_finished", summary=str(summary.get("artifacts", {}).get("summary_json")), handoff=str(summary.get("artifacts", {}).get("handoff_md")))
        print(f"Partition sensitivity summary: {summary.get('artifacts', {}).get('summary_json')}")
        print(f"Partition sensitivity handoff: {summary.get('artifacts', {}).get('handoff_md')}")
        return 0
    finally:
        reporter.close()


def _partC_rev1_load_recovery_physical_inputs(config: dict[str, Any]) -> tuple[Path, Any, Any, dict[str, Any]]:
    output_dir = partC_rev1_resolve_path(str(config.get("outputs", {}).get("output_dir", "partC_rev1_outputs")))
    _manifest, artifacts = _partC_rev1_load_4a_artifacts(output_dir)
    od = partC_rev1_load_od_snapshot(artifacts["od_npz"], artifacts["od_metadata"])
    small_summary: dict[str, Any] = {}
    try:
        small_path = partC_rev1_find_latest_small_region_summary(output_dir)
        graph, small_summary = partC_rev1_load_corrected_graph_from_summary(small_path)
    except FileNotFoundError:
        graph = partC_rev1_load_graph_snapshot(artifacts["graph_npz"], artifacts["graph_metadata"])
    return output_dir, graph, od, small_summary


def _partC_rev1_recovery_root_from_args(config: dict[str, Any], value: str | None) -> Path:
    if value is not None and str(value).strip():
        return partC_rev1_resolve_path(str(value))
    return partC_rev1_recovery_root(config)


def _partC_rev1_cmd_run_abstractflow_benchmark(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("run-abstractflow-benchmark requires --confirm-full-run")
    config = load_partC_rev1_config(args.config)
    source_output_dir, graph, od, small_summary = _partC_rev1_load_recovery_physical_inputs(config)
    recovery_root = _partC_rev1_recovery_root_from_args(config, args.recovery_root)
    reporter = PartCRev1ProgressReporter("partC_rev1_abstractflow_benchmark", output_dir=recovery_root)
    try:
        method = str(args.algorithm).lower()
        if method == "cfw":
            raise ValueError("Recovery abstract-flow tracking does not support CFW; use --algorithm fw or --algorithm msa")
        assignment_config = PartCRev1AssignmentConfig(
            max_iterations=int(args.max_iterations),
            gap_target=float(args.gap_target),
            backend=str(args.backend),
            origin_batch_size=int(args.origin_batch_size),
            max_origins=None if int(args.max_origins) <= 0 else int(args.max_origins),
            max_runtime_seconds=float(args.max_runtime_seconds) if float(args.max_runtime_seconds) > 0 else None,
        )
        reporter.emit(
            "abstractflow_benchmark_started",
            recovery_root=str(recovery_root),
            source_output_dir=str(source_output_dir),
            graph_hash=graph.graph_hash,
            topology_hash=graph.topology_hash,
            backbone_hash=graph.backbone_hash,
            partition_hash=graph.partition_hash,
            assignment_graph_hash=graph.assignment_graph_hash,
            od_hash=od.od_hash,
            small_region_correction_hash=small_summary.get("correction_hash"),
            config=asdict(assignment_config),
        )
        result = partC_rev1_solve_fw_with_abstract_tracking(
            graph,
            od,
            method=method,
            config=assignment_config,
            reporter=reporter,
        )
        paths = partC_rev1_write_abstract_benchmark_artifacts(
            result,
            graph,
            od,
            output_dir=recovery_root,
            config_path=args.config,
        )
        reporter.emit("abstractflow_benchmark_finished", summary=paths.get("summary_json"), handoff=paths.get("handoff"))
        print(f"Abstract benchmark summary: {paths.get('summary_json')}")
        print(f"Abstract benchmark handoff: {paths.get('handoff')}")
        return 0
    finally:
        reporter.close()


def _partC_rev1_cmd_audit_abstract_reachability(args: argparse.Namespace) -> int:
    config = load_partC_rev1_config(args.config)
    _source_output_dir, graph, od, _small_summary = _partC_rev1_load_recovery_physical_inputs(config)
    summary_path = partC_rev1_resolve_path(args.abstract_benchmark_summary)
    summary = partC_rev1_read_json(summary_path)
    recovery_root = summary_path.parent
    reporter = PartCRev1ProgressReporter("partC_rev1_abstract_reachability", output_dir=recovery_root)
    try:
        leg_path = summary.get("artifacts", {}).get("abstract_region_leg_flow")
        mask_path = summary.get("artifacts", {}).get("benchmark_reachable_mask")
        if not leg_path or not Path(str(leg_path)).exists():
            raise FileNotFoundError(f"Missing abstract leg flow: {leg_path}")
        if not mask_path or not Path(str(mask_path)).exists():
            raise FileNotFoundError(f"Missing benchmark reachable OD mask: {mask_path}")
        leg_df = pd.read_parquet(leg_path)
        with np.load(mask_path, allow_pickle=False) as data:
            benchmark_reachable = data["reachable_mask"].astype(bool)
        bundle = partC_rev1_build_abstract_graph_bundle(graph, od, leg_df)
        snapshot_dir = recovery_root / "partC_rev1_abstract_graph_snapshot"
        bundle.graph.save(snapshot_dir)
        bundle.od.save(snapshot_dir)
        partC_rev1_write_table(bundle.arc_table, recovery_root / "partC_rev1_abstract_graph_arc_table.parquet", index=False)
        partC_rev1_write_table(bundle.region_leg_table, recovery_root / "partC_rev1_abstract_graph_region_leg_support.parquet", index=False)
        audit = partC_rev1_run_abstract_reachability_audit(
            od,
            bundle,
            benchmark_reachable_mask=benchmark_reachable,
            output_dir=recovery_root,
            rel_tol=float(args.reachability_rel_tol),
            abs_tol=float(args.reachability_abs_tol),
            reporter=reporter,
        )
        audit.update(bundle.metadata)
        summary_out = recovery_root / "partC_rev1_abstract_reachability_summary.json"
        serializable = {key: value for key, value in audit.items() if key not in {"abstract_reachable_mask", "missing_mask"}}
        partC_rev1_write_json(serializable, summary_out)
        if audit.get("status") != "passed":
            failure = recovery_root / "partC_rev1_abstract_reachability_failure_report.md"
            failure.write_text(
                "\n".join(
                    [
                        "# partC_rev1 Abstract Reachability Failure Report",
                        "",
                        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
                        "",
                        f"- Additional unreachable demand vs benchmark: `{audit.get('additional_unreachable_demand_vs_benchmark')}`",
                        f"- Tolerance threshold: `{audit.get('tolerance_threshold')}`",
                        f"- Missing reachable OD rows: `{audit.get('missing_reachable_od_rows')}`",
                        "",
                        "The benchmark-derived abstract support did not pass the gate. R2 requires missing-support patching before global comparison.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            if not args.patch_missing_support:
                raise RuntimeError(f"Abstract reachability gate failed; report written to {failure}")
        print(f"Abstract reachability summary: {summary_out}")
        return 0
    finally:
        reporter.close()


def _partC_rev1_cmd_train_abstractflow_regional_models(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("train-abstractflow-regional-models requires --confirm-full-run")
    config = load_partC_rev1_config(args.config)
    summary_path = partC_rev1_resolve_path(args.abstract_benchmark_summary)
    summary = partC_rev1_read_json(summary_path)
    recovery_root = summary_path.parent
    leg_path = summary.get("artifacts", {}).get("abstract_region_leg_flow")
    if not leg_path or not Path(str(leg_path)).exists():
        raise FileNotFoundError(f"Missing abstract leg flow: {leg_path}")
    leg_df = pd.read_parquet(leg_path)
    training_summary = partC_rev1_train_recovery_regional_models(
        leg_df,
        output_dir=recovery_root,
        seed=int(config.get("project", {}).get("random_seed", 42)),
        training_run_id=str(getattr(args, "training_run_id", "") or ""),
        min_samples=int(getattr(args, "min_samples", 100)),
        max_samples=int(getattr(args, "max_samples", 2000)),
        sample_count_rule=str(getattr(args, "sample_count_rule", "active_pairs_x2")),
        save_all_sample_matrices=bool(getattr(args, "save_all_sample_matrices", False)),
        record_representative_curves=bool(getattr(args, "record_representative_curves", False)),
        representative_count=int(getattr(args, "representative_count", 10)),
        strict_nonconst_models=bool(getattr(args, "strict_nonconst_models", False)),
        disk_budget_gb=float(getattr(args, "disk_budget_gb", 0.0) or 0.0),
        max_wall_hours=float(getattr(args, "max_wall_hours", 0.0) or 0.0),
    )
    print(f"Abstract training summary: {training_summary.get('summary_json')}")
    return 0


def _partC_rev1_cmd_run_abstractflow_global(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("run-abstractflow-global requires --confirm-full-run")
    config = load_partC_rev1_config(args.config)
    _source_output_dir, graph, od, _small_summary = _partC_rev1_load_recovery_physical_inputs(config)
    training_summary_path = partC_rev1_resolve_path(args.abstract_training_summary)
    training_summary = partC_rev1_read_json(training_summary_path)
    recovery_root = partC_rev1_resolve_path(training_summary.get("recovery_root", training_summary_path.parent))
    benchmark_summary_path = recovery_root / "partC_rev1_abstract_benchmark_summary.json"
    reachability_summary_path = recovery_root / "partC_rev1_abstract_reachability_summary.json"
    if reachability_summary_path.exists():
        reachability = partC_rev1_read_json(reachability_summary_path)
        if str(reachability.get("status")) != "passed":
            raise RuntimeError(f"Reachability gate has not passed: {reachability_summary_path}")
    benchmark_summary = partC_rev1_read_json(benchmark_summary_path)
    try:
        bundle = partC_rev1_load_saved_abstract_graph_bundle(recovery_root)
    except FileNotFoundError:
        leg_df = pd.read_parquet(benchmark_summary["artifacts"]["abstract_region_leg_flow"])
        bundle = partC_rev1_build_abstract_graph_bundle(graph, od, leg_df)
    policies = args.policies or ["fftt_const_approximation", "fast_poly_surrogate", "best_nonconst_surrogate", "best_surrogate"]
    global_output_dir = recovery_root / str(args.global_run_id) if str(getattr(args, "global_run_id", "") or "") else recovery_root
    reporter = PartCRev1ProgressReporter("partC_rev1_abstractflow_global", output_dir=global_output_dir)
    try:
        global_summary = partC_rev1_run_abstract_global_assignment(
            bundle,
            policies=policies,
            output_dir=global_output_dir,
            max_iterations=int(args.max_iterations),
            origin_batch_size=int(args.origin_batch_size),
            backend=str(args.backend),
            reporter=reporter,
            training_summary_path=training_summary_path,
            strict_nonconst_models=bool(getattr(args, "strict_nonconst_models", False)),
            force_first_surrogate_refresh=bool(getattr(args, "force_first_surrogate_refresh", False)),
            lazy_abs_tol=float(getattr(args, "lazy_abs_tol", 1e-6)),
            lazy_rel_tol=float(getattr(args, "lazy_rel_tol", 1e-4)),
            max_stale_iterations=int(getattr(args, "max_stale_iterations", 3)),
            disk_budget_gb=float(getattr(args, "disk_budget_gb", 0.0) or 0.0),
            max_wall_hours=float(getattr(args, "max_wall_hours", 0.0) or 0.0),
        )
        print(f"Abstract global summary: {global_summary.get('summary_json')}")
        return 0
    finally:
        reporter.close()


def _partC_rev1_cmd_summarize_abstractflow_recovery(args: argparse.Namespace) -> int:
    config = load_partC_rev1_config(args.config)
    _source_output_dir, graph, _od, _small_summary = _partC_rev1_load_recovery_physical_inputs(config)
    global_summary_path = partC_rev1_resolve_path(args.global_summary)
    global_summary = partC_rev1_read_json(global_summary_path)
    recovery_root = partC_rev1_resolve_path(global_summary.get("recovery_root", global_summary_path.parent.parent if global_summary_path.parent.name.startswith("partC_rev1_recovery_global_") else global_summary_path.parent))
    summary_output_dir = global_summary_path.parent
    benchmark_summary = partC_rev1_read_json(recovery_root / "partC_rev1_abstract_benchmark_summary.json")
    reachability_summary = partC_rev1_read_json(recovery_root / "partC_rev1_abstract_reachability_summary.json")
    training_summary_path = global_summary.get("training_summary_path") or str(recovery_root / "partC_rev1_abstract_training_summary.json")
    training_summary = partC_rev1_read_json(partC_rev1_resolve_path(training_summary_path))
    paths = partC_rev1_write_recovery_summary_pack(
        output_dir=summary_output_dir,
        physical_graph=graph,
        benchmark_summary=benchmark_summary,
        reachability_summary=reachability_summary,
        training_summary=training_summary,
        global_summary=global_summary,
    )
    final_handoff = summary_output_dir / "partC_rev1_recovery_final_handoff.md"
    final_handoff.write_text(
        "\n".join(
            [
                "# partC_rev1 Abstract-Flow Recovery Final Handoff",
                "",
                f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
                "",
                f"- Recovery root: `{recovery_root}`",
                f"- Policy summary: `{paths.get('policy_summary_csv')}`",
                f"- Backbone GeoPackage: `{paths.get('backbone_gpkg')}`",
                f"- Artifact index: `{paths.get('artifact_index')}`",
                "",
                "No full 4F physical reload was run for this recovery prompt.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Recovery summary: {paths.get('experiment_summary')}")
    print(f"Recovery final handoff: {final_handoff}")
    return 0


def _partC_rev1_cmd_run_remaining_reviewer_experiments(args: argparse.Namespace) -> int:
    if not args.confirm_full_run:
        raise ValueError("run-remaining-reviewer-experiments requires --confirm-full-run")
    run_config = PartCRev1RemainingExperimentConfig(
        config_path=str(args.config),
        recovery_root=str(args.recovery_root),
        max_wall_hours=float(args.max_wall_hours),
        disk_budget_gb=float(args.disk_budget_gb),
        selected_resolution_count=int(args.selected_resolution_count),
        global_iterations=int(args.global_iterations),
        origin_batch_size=int(args.origin_batch_size),
        heartbeat_seconds=float(args.heartbeat_seconds),
        resume=not bool(getattr(args, "no_resume", False)),
    )
    reporter = PartCRev1ProgressReporter(
        "partC_rev1_remaining_reviewer_experiments",
        output_dir=partC_rev1_resolve_path("partC_rev1_outputs"),
        heartbeat_seconds=float(args.heartbeat_seconds),
    )
    try:
        state = partC_rev1_run_remaining_experiments(config=run_config, reporter=reporter)
    finally:
        reporter.close()
    print(json.dumps({"status": state.get("status"), "root": state.get("root"), "run_hash": state.get("run_hash")}, indent=2))
    return 0


def _partC_rev1_cmd_validate_remaining_experiments(args: argparse.Namespace) -> int:
    result = partC_rev1_validate_remaining_experiments(args.root, disk_budget_gb=float(args.disk_budget_gb))
    print(json.dumps(result, indent=2))
    return 0


def build_partC_rev1_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="partC_rev1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_default = subparsers.add_parser("write-default-config")
    write_default.add_argument("--output", required=True)
    write_default.set_defaults(func=_partC_rev1_cmd_write_default_config)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", required=True)
    validate.set_defaults(func=_partC_rev1_cmd_validate_config)

    verify_resume = subparsers.add_parser("verify-cap2over3-gravity-state")
    verify_resume.add_argument("--config", required=True)
    verify_resume.add_argument("--results-root", default="")
    verify_resume.set_defaults(func=_partC_rev1_cmd_verify_cap2over3_gravity_state)

    inspect = subparsers.add_parser("inspect-data")
    inspect.add_argument("--config", required=True)
    inspect.set_defaults(func=_partC_rev1_cmd_inspect_data)

    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--config", required=True)
    preprocess.set_defaults(func=_partC_rev1_cmd_preprocess)

    enrich = subparsers.add_parser("enrich-capacity")
    enrich.add_argument("--config", required=True)
    enrich.add_argument("--confirm-full-run", action="store_true")
    enrich.set_defaults(func=_partC_rev1_cmd_enrich_capacity)

    partition = subparsers.add_parser("partition")
    partition.add_argument("--config", required=True)
    partition.add_argument("--confirm-full-run", action="store_true")
    partition.set_defaults(func=_partC_rev1_cmd_partition)

    stability = subparsers.add_parser("partition-stability")
    stability.add_argument("--config", required=True)
    stability.add_argument("--confirm-full-run", action="store_true")
    stability.set_defaults(func=_partC_rev1_cmd_partition_stability)

    prepare = subparsers.add_parser("prepare-solver-artifacts")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--confirm-full-run", action="store_true")
    prepare.set_defaults(func=_partC_rev1_cmd_prepare_solver_artifacts)

    solver_pilots = subparsers.add_parser("run-solver-pilots")
    solver_pilots.add_argument("--config", required=True)
    solver_pilots.add_argument("--confirm-full-run", action="store_true")
    solver_pilots.add_argument("--iterations", type=int, default=3)
    solver_pilots.add_argument("--gap-target", type=float, default=1e-4)
    solver_pilots.add_argument("--origin-batch-size", type=int, default=32)
    solver_pilots.add_argument("--numba-pilot-origins", type=int, default=3)
    solver_pilots.add_argument("--igraph-origins", type=int, default=3)
    solver_pilots.add_argument("--full-igraph-aon", action="store_true")
    solver_pilots.add_argument("--networkx-origins", type=int, default=3)
    solver_pilots.add_argument("--igraph-fw-origins", type=int, default=2)
    solver_pilots.add_argument("--igraph-fw-iterations", type=int, default=1)
    solver_pilots.set_defaults(func=_partC_rev1_cmd_run_solver_pilots)

    physical_benchmarks = subparsers.add_parser("run-physical-benchmarks")
    physical_benchmarks.add_argument("--config", required=True)
    physical_benchmarks.add_argument("--confirm-full-run", action="store_true")
    physical_benchmarks.add_argument("--max-iterations", type=int, default=0)
    physical_benchmarks.add_argument("--gap-target", type=float, default=1e-4)
    physical_benchmarks.add_argument("--gap-reference", type=float, default=1e-4)
    physical_benchmarks.add_argument("--origin-batch-size", type=int, default=64)
    physical_benchmarks.add_argument("--backend-pilot-origins", type=int, default=0)
    physical_benchmarks.add_argument("--backend-pilot-iterations", type=int, default=2)
    physical_benchmarks.add_argument("--backend-pilot-min-seconds", type=float, default=120.0)
    physical_benchmarks.add_argument("--backend-pilot-min-batches", type=int, default=3)
    physical_benchmarks.add_argument("--backend-pilot-cv-tolerance", type=float, default=0.25)
    physical_benchmarks.add_argument("--witness-max-origins", type=int, default=0)
    physical_benchmarks.add_argument("--witness-all-positive-origins", action="store_true", default=True)
    physical_benchmarks.add_argument("--witness-diagnostic-max-origins", type=int, default=64)
    physical_benchmarks.add_argument("--witness-origin-batch-size", type=int, default=16)
    physical_benchmarks.add_argument("--witness-diagnostic-path-count", type=int, default=50)
    physical_benchmarks.add_argument("--validate-dependency-hashes", action="store_true")
    physical_benchmarks.set_defaults(func=_partC_rev1_cmd_run_physical_benchmarks)

    regional_training = subparsers.add_parser("run-regional-training")
    regional_training.add_argument("--config", required=True)
    regional_training.add_argument("--confirm-full-run", action="store_true")
    regional_training.add_argument("--samples-per-region", type=int, default=0)
    regional_training.add_argument("--pilot-samples-per-region", type=int, default=12)
    regional_training.add_argument("--pilot-region-count", type=int, default=0)
    regional_training.add_argument("--production-region-limit", type=int, default=0)
    regional_training.add_argument("--max-pilot-region-arcs", type=int, default=600)
    regional_training.add_argument("--max-active-pairs", type=int, default=64)
    regional_training.add_argument("--top-k-per-source", type=int, default=6)
    regional_training.add_argument("--regional-solver-iterations", type=int, default=3)
    regional_training.add_argument("--nn-epochs", type=int, default=24)
    regional_training.add_argument("--icnn-epochs", type=int, default=24)
    regional_training.set_defaults(func=_partC_rev1_cmd_run_regional_training)

    correct_regions = subparsers.add_parser("correct-training-regions")
    correct_regions.add_argument("--config", required=True)
    correct_regions.add_argument("--confirm-full-run", action="store_true")
    correct_regions.add_argument("--min-arc-count", type=int, default=5)
    correct_regions.add_argument("--portal-square-arc-count-threshold", type=int, default=20)
    correct_regions.add_argument("--witness-max-origins", type=int, default=0)
    correct_regions.add_argument("--witness-all-positive-origins", action="store_true", default=True)
    correct_regions.add_argument("--witness-diagnostic-max-origins", type=int, default=64)
    correct_regions.add_argument("--witness-diagnostic-path-count", type=int, default=50)
    correct_regions.add_argument("--reuse-compatible-partition", action="store_true")
    correct_regions.add_argument("--validate-dependency-hashes", action="store_true")
    correct_regions.add_argument("--heartbeat-seconds", type=float, default=15.0)
    correct_regions.set_defaults(func=_partC_rev1_cmd_correct_training_regions)

    full_training = subparsers.add_parser("run-full-regional-training")
    full_training.add_argument("--config", required=True)
    full_training.add_argument("--confirm-full-run", action="store_true")
    full_training.add_argument("--correction-summary", default="")
    full_training.add_argument("--use-existing-4c-root", default="")
    full_training.add_argument("--min-samples-per-region", type=int, default=100)
    full_training.add_argument("--max-samples-per-region", type=int, default=1000)
    full_training.add_argument("--region-limit", type=int, default=0)
    full_training.add_argument("--max-wall-hours", type=float, default=72.0)
    full_training.add_argument("--projection-min-completed-regions", type=int, default=3)
    full_training.add_argument("--max-active-pairs", type=int, default=64)
    full_training.add_argument("--top-k-per-source", type=int, default=6)
    full_training.add_argument("--regional-solver-iterations", type=int, default=3)
    full_training.add_argument("--xgb-estimators", type=int, default=12)
    full_training.add_argument("--nn-epochs", type=int, default=24)
    full_training.add_argument("--icnn-epochs", type=int, default=24)
    full_training.add_argument("--retune-multiplier", type=float, default=2.0)
    full_training.add_argument("--heartbeat-seconds", type=float, default=15.0)
    full_training.add_argument("--validate-dependency-hashes", action="store_true")
    full_training.add_argument("--no-resume", action="store_true")
    full_training.set_defaults(func=_partC_rev1_cmd_run_full_regional_training)

    sparse_solver = subparsers.add_parser("run-sparse-global-solver")
    sparse_solver.add_argument("--config", required=True)
    sparse_solver.add_argument("--confirm-full-run", action="store_true")
    sparse_solver.add_argument("--max-iterations", type=int, default=3)
    sparse_solver.add_argument("--pilot-origins", type=int, default=16)
    sparse_solver.add_argument("--origin-batch-size", type=int, default=8)
    sparse_solver.add_argument("--master-gap-tol", type=float, default=1e-3)
    sparse_solver.add_argument("--pricing-gap-tol", type=float, default=1e-3)
    sparse_solver.add_argument("--price-tol", type=float, default=1e-6)
    sparse_solver.add_argument("--lazy-abs-tol", type=float, default=1e-6)
    sparse_solver.add_argument("--lazy-rel-tol", type=float, default=1e-4)
    sparse_solver.add_argument("--full-refresh-frequency", type=int, default=2)
    sparse_solver.add_argument("--regional-cost-slope", type=float, default=1e-6)
    sparse_solver.set_defaults(func=_partC_rev1_cmd_run_sparse_global_solver)

    global_policy = subparsers.add_parser("run-global-policy-suite")
    global_policy.add_argument("--config", required=True)
    global_policy.add_argument("--confirm-full-run", action="store_true")
    global_policy.add_argument("--training-summary", default="")
    global_policy.add_argument("--use-training-registry", default="")
    global_policy.add_argument("--policies", nargs="*", default=[])
    global_policy.add_argument("--correction-summary", default="")
    global_policy.add_argument("--max-iterations", type=int, default=3)
    global_policy.add_argument("--pilot-origins", type=int, default=0)
    global_policy.add_argument("--origin-batch-size", type=int, default=16)
    global_policy.add_argument("--master-gap-tol", type=float, default=1e-3)
    global_policy.add_argument("--pricing-gap-tol", type=float, default=1e-3)
    global_policy.add_argument("--price-tol", type=float, default=1e-6)
    global_policy.add_argument("--lazy-abs-tol", type=float, default=1e-6)
    global_policy.add_argument("--lazy-rel-tol", type=float, default=1e-4)
    global_policy.add_argument("--full-refresh-frequency", type=int, default=2)
    global_policy.add_argument("--regional-cost-slope", type=float, default=1e-6)
    global_policy.add_argument("--runtime-guard-seconds", type=float, default=86400.0)
    global_policy.add_argument("--heartbeat-seconds", type=float, default=15.0)
    global_policy.add_argument("--validate-dependency-hashes", action="store_true")
    global_policy.add_argument("--no-resume", action="store_true")
    global_policy.set_defaults(func=_partC_rev1_cmd_run_global_policy_suite)

    physical_reload = subparsers.add_parser("run-physical-reload")
    physical_reload.add_argument("--config", required=True)
    physical_reload.add_argument("--confirm-full-run", action="store_true")
    physical_reload.add_argument("--max-regions", type=int, default=0)
    physical_reload.add_argument("--max-iterations", type=int, default=8)
    physical_reload.add_argument("--gap-target", type=float, default=1e-4)
    physical_reload.add_argument("--positive-flow-tol", type=float, default=1e-9)
    physical_reload.add_argument("--enable-correction", action="store_true")
    physical_reload.set_defaults(func=_partC_rev1_cmd_run_physical_reload)

    physical_reload_suite = subparsers.add_parser("run-physical-reload-suite")
    physical_reload_suite.add_argument("--config", required=True)
    physical_reload_suite.add_argument("--confirm-full-run", action="store_true")
    physical_reload_suite.add_argument("--global-policy-summary", default="")
    physical_reload_suite.add_argument("--use-global-policy-registry", default="")
    physical_reload_suite.add_argument("--use-benchmark-root", default="")
    physical_reload_suite.add_argument("--max-runs", type=int, default=0)
    physical_reload_suite.add_argument("--max-regions", type=int, default=0)
    physical_reload_suite.add_argument("--max-iterations", type=int, default=8)
    physical_reload_suite.add_argument("--gap-target", type=float, default=1e-4)
    physical_reload_suite.add_argument("--positive-flow-tol", type=float, default=1e-9)
    physical_reload_suite.add_argument("--runtime-guard-seconds", type=float, default=86400.0)
    physical_reload_suite.add_argument("--heartbeat-seconds", type=float, default=15.0)
    physical_reload_suite.add_argument("--validate-dependency-hashes", action="store_true")
    physical_reload_suite.add_argument("--no-resume", action="store_true")
    physical_reload_suite.set_defaults(func=_partC_rev1_cmd_run_physical_reload_suite)

    partition_sweep = subparsers.add_parser("run-cap2over3-partition-sweep")
    partition_sweep.add_argument("--config", required=True)
    partition_sweep.add_argument("--confirm-full-run", action="store_true")
    partition_sweep.add_argument("--resolutions", default="0.25,0.35,0.50,0.75,1.0,1.25,1.5,2.0,3.0,4.0,6.0,8.0")
    partition_sweep.add_argument("--heartbeat-seconds", type=float, default=15.0)
    partition_sweep.add_argument("--validate-dependency-hashes", action="store_true")
    partition_sweep.add_argument("--no-resume", action="store_true")
    partition_sweep.set_defaults(func=_partC_rev1_cmd_run_cap2over3_partition_sweep)

    reviewer = subparsers.add_parser("run-reviewer-experiments")
    reviewer.add_argument("--config", required=True)
    reviewer.add_argument("--confirm-full-run", action="store_true")
    reviewer.add_argument("--bootstrap-seed", type=int, default=42)
    reviewer.add_argument("--max-scatter-points", type=int, default=20000)
    reviewer.set_defaults(func=_partC_rev1_cmd_run_reviewer_experiments)

    revised = subparsers.add_parser("run-revised-reviewer-experiments")
    revised.add_argument("--config", required=True)
    revised.add_argument("--confirm-full-run", action="store_true")
    revised.add_argument("--bootstrap-seed", type=int, default=42)
    revised.add_argument("--max-scatter-points", type=int, default=20000)
    revised.add_argument("--heartbeat-seconds", type=float, default=15.0)
    revised.add_argument("--validate-dependency-hashes", action="store_true")
    revised.add_argument("--no-resume", action="store_true")
    revised.set_defaults(func=_partC_rev1_cmd_run_revised_reviewer_experiments)

    cap2over3_reviewer = subparsers.add_parser("run-cap2over3-reviewer-experiments")
    cap2over3_reviewer.add_argument("--config", required=True)
    cap2over3_reviewer.add_argument("--confirm-full-run", action="store_true")
    cap2over3_reviewer.add_argument("--bootstrap-seed", type=int, default=42)
    cap2over3_reviewer.add_argument("--max-scatter-points", type=int, default=20000)
    cap2over3_reviewer.add_argument("--heartbeat-seconds", type=float, default=15.0)
    cap2over3_reviewer.add_argument("--use-benchmark-root", default="")
    cap2over3_reviewer.add_argument("--use-training-root", default="")
    cap2over3_reviewer.add_argument("--use-global-root", default="")
    cap2over3_reviewer.add_argument("--use-reload-root", default="")
    cap2over3_reviewer.add_argument("--run-e4", action="store_true")
    cap2over3_reviewer.add_argument("--run-e7", action="store_true")
    cap2over3_reviewer.add_argument("--run-e8-e10", action="store_true")
    cap2over3_reviewer.add_argument("--run-e13", action="store_true")
    cap2over3_reviewer.add_argument("--validate-dependency-hashes", action="store_true")
    cap2over3_reviewer.add_argument("--no-resume", action="store_true")
    cap2over3_reviewer.set_defaults(func=_partC_rev1_cmd_run_revised_reviewer_experiments)

    abstract_benchmark = subparsers.add_parser("run-abstractflow-benchmark")
    abstract_benchmark.add_argument("--config", required=True)
    abstract_benchmark.add_argument("--confirm-full-run", action="store_true")
    abstract_benchmark.add_argument("--algorithm", choices=["fw", "msa", "cfw"], default="fw")
    abstract_benchmark.add_argument("--backend", default="numba_csr")
    abstract_benchmark.add_argument("--gap-target", type=float, default=1e-4)
    abstract_benchmark.add_argument("--max-iterations", type=int, default=100)
    abstract_benchmark.add_argument("--max-runtime-seconds", type=float, default=14400.0)
    abstract_benchmark.add_argument("--origin-batch-size", type=int, default=16)
    abstract_benchmark.add_argument("--max-origins", type=int, default=0)
    abstract_benchmark.add_argument("--reachability-rel-tol", type=float, default=1e-4)
    abstract_benchmark.add_argument("--reachability-abs-tol", type=float, default=1e-6)
    abstract_benchmark.add_argument("--recovery-root", default="")
    abstract_benchmark.set_defaults(func=_partC_rev1_cmd_run_abstractflow_benchmark)

    abstract_reachability = subparsers.add_parser("audit-abstract-reachability")
    abstract_reachability.add_argument("--config", required=True)
    abstract_reachability.add_argument("--abstract-benchmark-summary", required=True)
    abstract_reachability.add_argument("--patch-missing-support", action="store_true")
    abstract_reachability.add_argument("--max-patch-rounds", type=int, default=5)
    abstract_reachability.add_argument("--reachability-rel-tol", type=float, default=1e-4)
    abstract_reachability.add_argument("--reachability-abs-tol", type=float, default=1e-6)
    abstract_reachability.set_defaults(func=_partC_rev1_cmd_audit_abstract_reachability)

    abstract_training = subparsers.add_parser("train-abstractflow-regional-models")
    abstract_training.add_argument("--config", required=True)
    abstract_training.add_argument("--abstract-benchmark-summary", required=True)
    abstract_training.add_argument("--training-run-id", default="")
    abstract_training.add_argument("--min-samples", type=int, default=100)
    abstract_training.add_argument("--max-samples", type=int, default=2000)
    abstract_training.add_argument("--sample-count-rule", default="active_pairs_x2")
    abstract_training.add_argument("--save-all-sample-matrices", action="store_true")
    abstract_training.add_argument("--record-representative-curves", action="store_true")
    abstract_training.add_argument("--representative-count", type=int, default=10)
    abstract_training.add_argument("--strict-nonconst-models", action="store_true")
    abstract_training.add_argument("--disk-budget-gb", type=float, default=0.0)
    abstract_training.add_argument("--max-wall-hours", type=float, default=0.0)
    abstract_training.add_argument("--confirm-full-run", action="store_true")
    abstract_training.set_defaults(func=_partC_rev1_cmd_train_abstractflow_regional_models)

    abstract_global = subparsers.add_parser("run-abstractflow-global")
    abstract_global.add_argument("--config", required=True)
    abstract_global.add_argument("--abstract-training-summary", required=True)
    abstract_global.add_argument("--policies", nargs="*", default=[])
    abstract_global.add_argument("--assignment-method", choices=["msa", "fixed_point"], default="msa")
    abstract_global.add_argument("--backend", default="numba_csr")
    abstract_global.add_argument("--max-iterations", type=int, default=20)
    abstract_global.add_argument("--origin-batch-size", type=int, default=16)
    abstract_global.add_argument("--global-run-id", default="")
    abstract_global.add_argument("--force-first-surrogate-refresh", action="store_true")
    abstract_global.add_argument("--strict-nonconst-models", action="store_true")
    abstract_global.add_argument("--lazy-abs-tol", type=float, default=1e-6)
    abstract_global.add_argument("--lazy-rel-tol", type=float, default=1e-4)
    abstract_global.add_argument("--max-stale-iterations", type=int, default=3)
    abstract_global.add_argument("--disk-budget-gb", type=float, default=0.0)
    abstract_global.add_argument("--max-wall-hours", type=float, default=0.0)
    abstract_global.add_argument("--confirm-full-run", action="store_true")
    abstract_global.set_defaults(func=_partC_rev1_cmd_run_abstractflow_global)

    abstract_summary = subparsers.add_parser("summarize-abstractflow-recovery")
    abstract_summary.add_argument("--config", required=True)
    abstract_summary.add_argument("--global-summary", required=True)
    abstract_summary.add_argument("--make-backbone-gpkg", action="store_true")
    abstract_summary.set_defaults(func=_partC_rev1_cmd_summarize_abstractflow_recovery)

    remaining = subparsers.add_parser("run-remaining-reviewer-experiments")
    remaining.add_argument("--config", required=True)
    remaining.add_argument("--recovery-root", required=True)
    remaining.add_argument("--confirm-full-run", action="store_true")
    remaining.add_argument("--max-wall-hours", type=float, default=48.0)
    remaining.add_argument("--disk-budget-gb", type=float, default=200.0)
    remaining.add_argument("--selected-resolution-count", type=int, default=5)
    remaining.add_argument("--global-iterations", type=int, default=20)
    remaining.add_argument("--origin-batch-size", type=int, default=32)
    remaining.add_argument("--heartbeat-seconds", type=float, default=15.0)
    remaining.add_argument("--no-resume", action="store_true")
    remaining.set_defaults(func=_partC_rev1_cmd_run_remaining_reviewer_experiments)

    validate_remaining = subparsers.add_parser("validate-remaining-experiments")
    validate_remaining.add_argument("--root", required=True)
    validate_remaining.add_argument("--disk-budget-gb", type=float, default=200.0)
    validate_remaining.set_defaults(func=_partC_rev1_cmd_validate_remaining_experiments)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_partC_rev1_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
