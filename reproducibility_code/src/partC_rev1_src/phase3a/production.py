"""Selected-method Phase 3 production orchestration for all accepted suites.

This module deliberately reuses the reviewed Phase 3A scientific kernels.  It
adds only all-suite authority binding, write-new checkpoints, normalized
production exports, and portfolio assembly inputs.  It never runs Phase 4.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import shutil
import subprocess
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from partC_rev1_src.partC_rev1_hashing import partC_rev1_file_sha256

from .authority import resolve_authority_paths, verify_authority
from .bundles import validate_bundle_closure
from .loaders import load_graph, load_suite
from .operations import (
    FULL_TASK2,
    _build_region_oracle,
    construct_geo_candidate,
    evaluate_candidate_od_batch,
    finalize_candidate_topology,
    reproduce_topologies,
)
from .resources import directory_bytes, process_tree_rss
from .schemas import stable_object_id, validate_candidate_matrix


CANDIDATE_ID = "geo_gc_1000m"
METHOD_ID = "geo_gc_1000m_structural_seed_v1"
SUITE_ORDER = ("s03", "s04", "s05", "s02", "s06", "s01")
MANDATORY_TOPOLOGY_FILES = (
    "bacn_membership.parquet",
    "bacn_summary.parquet",
    "bundle_membership.parquet",
    "bundle_summary.parquet",
    "suppressed_backbone_arcs.parquet",
    "racn_membership.parquet",
    "racn_summary.parquet",
    "racn_restricted_witnesses.parquet",
    "consolidated_terminals.parquet",
    "terminal_fine_members.parquet",
    "regional_pair_support_contract.json",
    "physical_to_candidate_lookup.npz",
    "tracer_interface.json",
    "structural_internal_charges.parquet",
    "finalist_spec.json",
)


class Phase3ProductionError(RuntimeError):
    """Raised when a selected-method production hard gate fails."""


def required_suite_ids(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the explicitly required production portfolio.

    The original all-suite authority did not carry ``required_suite_ids``;
    retaining ``SUITE_ORDER`` as the fallback keeps that reviewed execution
    contract byte-for-byte compatible.  Later scope amendments must be
    explicit and are validated as a partition with ``excluded_suite_ids``.
    """

    values = tuple(str(value) for value in config.get("required_suite_ids", SUITE_ORDER))
    if not values or len(values) != len(set(values)):
        raise Phase3ProductionError("Required production suite IDs must be nonempty and unique")
    return values


def excluded_suite_ids(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return explicitly excluded accepted suites, if an amendment supplies them."""

    values = tuple(str(value) for value in config.get("excluded_suite_ids", ()))
    if len(values) != len(set(values)):
        raise Phase3ProductionError("Excluded suite IDs must be unique")
    return values


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_production_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("phase_id") != "phase_03" or payload.get("step_id") != "all_suite_acnhn":
        raise Phase3ProductionError("Production config has the wrong explicit phase/step identity")
    if tuple(payload.get("suite_order", ())) != SUITE_ORDER:
        raise Phase3ProductionError("Production suite order differs from the external decision")
    if not payload.get("tasks") or not payload.get("expected_outputs") or not payload.get("acceptance_tests"):
        raise Phase3ProductionError("Production config requires tasks, outputs, and acceptance tests")
    if not payload.get("campaign_id") or not payload.get("run_id") or not payload.get("handoff_id"):
        raise Phase3ProductionError("Production config requires explicit campaign, run, and handoff IDs")
    if payload.get("random_seed") != 42:
        raise Phase3ProductionError("Production config changed the frozen deterministic seed")
    if payload.get("method", {}).get("method_id") != METHOD_ID:
        raise Phase3ProductionError("Production config changed the frozen winning method")
    if payload.get("phase_boundaries", {}).get("phase4_authorized") is not False:
        raise Phase3ProductionError("Phase 4 must remain explicitly unauthorized")
    if payload.get("storage", {}).get("dense_terminal_square_prohibited") is not True:
        raise Phase3ProductionError("Dense terminal-square allocation must remain prohibited")
    suites = payload.get("suites", [])
    accepted_suite_ids = {str(row.get("suite_short_id")) for row in suites}
    if accepted_suite_ids != {f"s{value:02d}" for value in range(1, 7)}:
        raise Phase3ProductionError("Production config must bind exactly six suites")
    required = required_suite_ids(payload)
    excluded = excluded_suite_ids(payload)
    if set(required).intersection(excluded):
        raise Phase3ProductionError("Required and excluded suite IDs overlap")
    if set(required).union(excluded) != accepted_suite_ids:
        raise Phase3ProductionError(
            "Required and excluded suite IDs must partition the six accepted suites"
        )
    if "s03" not in required:
        raise Phase3ProductionError("The accepted s03 production reference must remain required")
    if excluded and not payload.get("scope_amendment", {}).get("decision_id"):
        raise Phase3ProductionError("An excluded suite requires an explicit scope-amendment decision")
    validate_candidate_matrix(payload["candidate_matrix_path"])
    for key in ("project_root", "accepted_phase2_output_root"):
        Path(str(payload[key])).resolve(strict=True)
    Path(str(payload["authorization_package"]["s03_external_result_path"])).resolve(strict=True)
    for key, value in payload["input_versions"].items():
        if key.endswith(("_npz", "_parquet", "_json", "_sqlite")):
            Path(str(value)).resolve(strict=True)
    payload["config_path"] = str(source)
    return payload


def suite_record(config: Mapping[str, Any], suite_short_id: str) -> dict[str, Any]:
    matches = [dict(row) for row in config["suites"] if str(row["suite_short_id"]) == str(suite_short_id)]
    if len(matches) != 1:
        raise Phase3ProductionError(f"Unknown or duplicated suite: {suite_short_id}")
    return matches[0]


def operation_config(config: Mapping[str, Any], suite_short_id: str) -> dict[str, Any]:
    record = suite_record(config, suite_short_id)
    # Resolve deterministically before PREPARE creates the write-new roots.
    run_root = Path(str(config["output_root"])).resolve(strict=False)
    temp_root = Path(str(config["temp_root"])).resolve(strict=False)
    accepted = {
        "suite_id": record["phase2_suite_id"],
        "graph_identity": config["input_versions"]["graph_identity"],
        "od_identity": config["input_versions"]["od_identity"],
        "membership_hash": record["membership_hash"],
        "node_membership_hash": record["node_membership_hash"],
        "structural_signature": record["structural_signature"],
    }
    return {
        "phase_id": "phase_03",
        "step_id": "all_suite_acnhn",
        "suite_id": suite_short_id,
        "phase2_suite_id": record["phase2_suite_id"],
        "campaign_id": config["campaign_id"],
        "run_id": config["run_id"],
        "project_root": config["project_root"],
        "accepted_phase2_output_root": config["accepted_phase2_output_root"],
        "accepted_identities": accepted,
        "input_versions": dict(config["input_versions"]),
        "output_root": str(run_root / "operations" / suite_short_id),
        "temp_root": str(temp_root / suite_short_id),
        "candidate_matrix_path": config["candidate_matrix_path"],
        "execution_scope": FULL_TASK2,
        "production_write_support_summary": True,
        "random_seed": int(config["random_seed"]),
        "phase_boundaries": dict(config["phase_boundaries"]),
    }


def prepare_campaign(config: Mapping[str, Any], config_path: str | Path) -> dict[str, Any]:
    run_root = Path(str(config["output_root"]))
    temp_root = Path(str(config["temp_root"]))
    if run_root.exists() or temp_root.exists():
        raise Phase3ProductionError("Prepare is write-new-only; output or temp root already exists")
    package = Path(str(config["authorization_package"]["path"])).resolve(strict=True)
    if package.stat().st_size != int(config["authorization_package"]["size_bytes"]):
        raise Phase3ProductionError("Authorization package size changed")
    if partC_rev1_file_sha256(package) != str(config["authorization_package"]["sha256"]):
        raise Phase3ProductionError("Authorization package SHA-256 changed")
    run_root.mkdir(parents=True)
    temp_root.mkdir(parents=True)
    for name in (
        "operations", "suites", "receipts", "authority", "database", "geopackage",
        "tables", "figures", "styles", "evidence", "handoff",
    ):
        (run_root / name).mkdir()
    extract_root = temp_root / "authority_package"
    extract_root.mkdir()
    with zipfile.ZipFile(package, "r") as archive:
        infos = archive.infolist()
        if len(infos) != int(config["authorization_package"]["entry_count"]):
            raise Phase3ProductionError("Authorization package entry count changed")
        for info in infos:
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise Phase3ProductionError(f"Unsafe authorization package member: {info.filename}")
        archive.extractall(extract_root)
        if archive.testzip() is not None:
            raise Phase3ProductionError("Authorization package CRC validation failed")
    authority_dest = run_root / "authority"
    for source in sorted(extract_root.glob("*.md")):
        shutil.copy2(source, authority_dest / source.name)
    for source in (extract_root / "config").glob("*"):
        if source.is_file():
            target = authority_dest / "config" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in ("PACKAGE_INFO.json", "MANIFEST.csv", "CHECKSUMS.sha256"):
        shutil.copy2(extract_root / name, authority_dest / name)
    shutil.copy2(Path(config_path).resolve(strict=True), authority_dest / "executed_phase3_all_suite_config.json")
    shutil.copy2(Path(str(config["candidate_matrix_path"])).resolve(strict=True), authority_dest / "phase3_selected_candidate_matrix.csv")
    git_snapshot = _git_snapshot(Path(str(config["project_root"])))
    if git_snapshot["branch"] != config["expected_branch"]:
        raise Phase3ProductionError("Repository branch differs from the authorized branch")
    status = {
        "schema_id": "phase3.all_suite_run_status.v1",
        "phase_id": config["phase_id"],
        "step_id": config["step_id"],
        "campaign_id": config["campaign_id"],
        "run_id": config["run_id"],
        "status": "EXECUTING",
        "started_utc": utc_now(),
        "suite_order": list(SUITE_ORDER),
        "completed_suites": [],
        "all_six_suites_built": False,
        "winning_method": METHOD_ID,
        "phase4_authorized": False,
        "capacity_multiplier_tuned": False,
        "cost_calibration_performed": False,
        "model_training_performed": False,
        "acnhn_equilibrium_run": False,
        "authority_package_sha256": config["authorization_package"]["sha256"],
        "git_snapshot": git_snapshot,
        "resource_budget": dict(config["resources"]),
    }
    _write_json_new(run_root / "RUN_STATUS.json", status)
    receipt = {
        "schema_id": "phase3.production_receipt.v1",
        "task_id": "PREPARE-AUTHORITY",
        "status": "PASS",
        "completed_utc": utc_now(),
        "authorization_package_sha256": config["authorization_package"]["sha256"],
        "extracted_entry_count": len(list(extract_root.rglob("*"))),
        "output_root": str(run_root),
        "temp_root": str(temp_root),
        "git_snapshot": git_snapshot,
    }
    _write_json_new(run_root / "receipts" / "PREPARE-AUTHORITY.json", receipt)
    return receipt


def reuse_s03(config: Mapping[str, Any]) -> dict[str, Any]:
    began = time.perf_counter()
    run_root = Path(str(config["output_root"])).resolve(strict=True)
    receipt_path = run_root / "receipts" / "S03-REUSE.json"
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing.get("status") != "PASS":
            raise Phase3ProductionError("Existing s03 reuse receipt is not a passing checkpoint")
        _verify_existing_export(config, "s03", run_root / "suites" / "s03" / "parquet")
        _mark_suite_complete(run_root, "s03")
        return existing
    op_config = operation_config(config, "s03")
    op_root = Path(op_config["output_root"])
    temp_suite = Path(op_config["temp_root"])
    op_root.mkdir(parents=True, exist_ok=True)
    temp_suite.mkdir(parents=True, exist_ok=True)
    extract_root = Path(str(config["temp_root"])).resolve(strict=True) / "authority_package"
    manifest_path = extract_root / str(config["authorization_package"]["s03_manifest_member"])
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1863:
        raise Phase3ProductionError("s03 reuse manifest row count changed")
    linked = 0
    for row in rows:
        source = Path(str(row["absolute_path"])).resolve(strict=True)
        if source.stat().st_size != int(row["size_bytes"]) or partC_rev1_file_sha256(source) != row["sha256"]:
            raise Phase3ProductionError(f"s03 reuse source changed: {source}")
        target = op_root.joinpath(*PurePosixPath(row["relative_path"]).parts)
        _link_or_verify(source, target)
        linked += 1
    evaluation_source = Path(
        str(config["authorization_package"]["s03_external_result_path"])
    ).resolve(strict=True)
    if partC_rev1_file_sha256(evaluation_source) != config["common_evaluation"]["s03_result_sha256"]:
        raise Phase3ProductionError("Accepted s03 evaluation result changed")
    result_target = op_root / "round4_evaluation" / "candidates" / CANDIDATE_ID / "candidate_od_results.parquet"
    _link_or_verify(evaluation_source, result_target)
    sample_source = extract_root / str(config["authorization_package"]["od_sample_member"])
    paths_source = extract_root / str(config["authorization_package"]["physical_paths_member"])
    _verify_common_evaluation_inputs(config, sample_source, paths_source)
    _link_or_verify(sample_source, op_root / "round4_evaluation" / "od_sample.parquet")
    _link_or_verify(paths_source, op_root / "round4_evaluation" / "physical_paths.parquet")
    if not _operation_complete(op_root, "S03-REPRODUCE"):
        reproduce_receipt = reproduce_topologies(
            op_config,
            {"task_id": "S03-REPRODUCE", "operation": "reproduce_topologies", "candidate_ids": [CANDIDATE_ID]},
        )
    else:
        reproduction_path = op_root / "receipts" / "S03-REPRODUCE.json"
        reproduce_receipt = json.loads(reproduction_path.read_text(encoding="utf-8"))
        reproduce_receipt["receipt_path"] = str(reproduction_path)
    finalist = _finalist_dir(op_root)
    if not (finalist / "regional_pair_support_summary.parquet").is_file():
        enumerate_regional_pair_support(
            op_config,
            finalist,
            visualization_cap=int(config["storage"]["geopackage_regional_pair_visualization_cap_per_suite"]),
        )
    export = export_suite(config, "s03")
    receipt = {
        "schema_id": "phase3.production_receipt.v1",
        "task_id": "S03-REUSE",
        "status": "PASS",
        "completed_utc": utc_now(),
        "manifest_rows_verified": len(rows),
        "hash_identical_relocated_files": linked,
        "candidate_result_sha256": partC_rev1_file_sha256(result_target),
        "reproduction_receipt": reproduce_receipt["receipt_path"],
        "production_directory": export["production_directory"],
        "elapsed_seconds": time.perf_counter() - began,
    }
    _write_json_new(receipt_path, receipt)
    _mark_suite_complete(run_root, "s03")
    return receipt


def build_suite(config: Mapping[str, Any], suite_short_id: str) -> dict[str, Any]:
    if suite_short_id == "s03" or suite_short_id not in SUITE_ORDER:
        raise Phase3ProductionError("build_suite accepts only non-s03 authorized suites")
    began = time.perf_counter()
    run_root = Path(str(config["output_root"])).resolve(strict=True)
    prefix = suite_short_id.upper()
    receipt_path = run_root / "receipts" / f"{prefix}-COMPLETE.json"
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing.get("status") != "PASS":
            raise Phase3ProductionError(f"Existing {suite_short_id} completion receipt is not a pass")
        _verify_existing_export(config, suite_short_id, run_root / "suites" / suite_short_id / "parquet")
        _mark_suite_complete(run_root, suite_short_id)
        return existing
    op_config = operation_config(config, suite_short_id)
    op_root = Path(op_config["output_root"])
    temp_suite = Path(op_config["temp_root"])
    op_root.mkdir(parents=True, exist_ok=True)
    temp_suite.mkdir(parents=True, exist_ok=True)
    if not _operation_complete(op_root, f"{prefix}-BUILD"):
        construct_geo_candidate(
            op_config,
            {"task_id": f"{prefix}-BUILD", "operation": "construct_geo_candidate", "candidate_ids": [CANDIDATE_ID]},
        )
    if not _operation_complete(op_root, f"{prefix}-FINALIZE"):
        finalize_candidate_topology(
            op_config,
            {
                "task_id": f"{prefix}-FINALIZE",
                "operation": "finalize_candidate_topology",
                "substage": "full_topology",
                "candidate_ids": [CANDIDATE_ID],
            },
        )
    extract_root = Path(str(config["temp_root"])).resolve(strict=True) / "authority_package"
    sample_source = extract_root / str(config["authorization_package"]["od_sample_member"])
    paths_source = extract_root / str(config["authorization_package"]["physical_paths_member"])
    _verify_common_evaluation_inputs(config, sample_source, paths_source)
    sample_target = op_root / "round4_evaluation" / "od_sample.parquet"
    paths_target = op_root / "round4_evaluation" / "physical_paths.parquet"
    if not sample_target.exists():
        _link_or_copy_new(sample_source, sample_target)
    if not paths_target.exists():
        _link_or_copy_new(paths_source, paths_target)
    if not _operation_complete(op_root, f"{prefix}-EVALUATE"):
        common = config["common_evaluation"]
        evaluate_candidate_od_batch(
            op_config,
            {
                "task_id": f"{prefix}-EVALUATE",
                "operation": "evaluate_candidate_od_batch",
                "substage": "projected_scalar_lifted",
                "candidate_ids": [CANDIDATE_ID],
                "sample_path": str(sample_target),
                "physical_paths_path": str(paths_target),
                "minimum_unique_od_count": 200,
                "bounded_path_induced_scalar": False,
                "lazy_state_release_scope": common["lazy_state_release_scope"],
                "regional_oracle_cache_max_regions": common["regional_oracle_cache_max_regions"],
                "retain_regional_query_cache": common["retain_regional_query_cache"],
                "scalar_target_batch_size": common["scalar_target_batch_size"],
            },
        )
    if not _operation_complete(op_root, f"{prefix}-REPRODUCE"):
        reproduce_topologies(
            op_config,
            {"task_id": f"{prefix}-REPRODUCE", "operation": "reproduce_topologies", "candidate_ids": [CANDIDATE_ID]},
        )
    production_root = run_root / "suites" / suite_short_id / "parquet"
    if not production_root.exists():
        export_suite(config, suite_short_id)
    receipt = {
        "schema_id": "phase3.production_receipt.v1",
        "task_id": f"{prefix}-COMPLETE",
        "status": "PASS",
        "suite_short_id": suite_short_id,
        "phase2_suite_id": suite_record(config, suite_short_id)["phase2_suite_id"],
        "completed_utc": utc_now(),
        "production_directory": str(production_root),
        "elapsed_seconds": time.perf_counter() - began,
        "phase4_executed": False,
    }
    _write_json_new(receipt_path, receipt)
    _mark_suite_complete(run_root, suite_short_id)
    return receipt


def enumerate_regional_pair_support(
    op_config: Mapping[str, Any], finalist_dir: str | Path, *, visualization_cap: int = 5000
) -> pd.DataFrame:
    """Enumerate exact pair counts without retaining a global pair registry."""

    destination = Path(finalist_dir) / "regional_pair_support_summary.parquet"
    if destination.exists():
        return pd.read_parquet(destination)
    paths = resolve_authority_paths(op_config)
    graph = load_graph(paths)
    suite = load_suite(paths, graph)
    terminals = pd.read_parquet(Path(finalist_dir) / "consolidated_terminals.parquet")
    terminal_members = pd.read_parquet(Path(finalist_dir) / "terminal_fine_members.parquet")
    uid_to_index = {str(value): index for index, value in enumerate(graph.node_uid.tolist())}
    fine_global = {
        str(terminal_id): tuple(uid_to_index[str(uid)] for uid in group["partC_rev1_node_uid"].astype(str))
        for terminal_id, group in terminal_members.groupby("terminal_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    grouped = list(terminals.groupby("region_id", sort=True))
    base_cap, extra_cap = divmod(max(0, visualization_cap), max(1, len(grouped)))
    started = time.perf_counter()
    for position, (region, region_terminals) in enumerate(grouped, 1):
        region_started = time.perf_counter()
        region_cap = base_cap + (1 if position <= extra_cap else 0)
        oracle = _build_region_oracle(
            graph, suite, CANDIDATE_ID, str(region), terminals, fine_global, retain_query_cache=False
        )
        sources = sorted(
            region_terminals.loc[
                region_terminals["can_exit_region"].astype(bool), "terminal_id"
            ].astype(str).tolist()
        )
        pair_count = 0
        region_samples: list[dict[str, Any]] = []
        for source_id in sources:
            witnesses = oracle.query(source_id)
            pair_count += len(witnesses)
            for witness in witnesses:
                if len(region_samples) >= region_cap:
                    break
                region_samples.append(
                    {
                        "suite_id": suite.suite_id,
                        "candidate_id": CANDIDATE_ID,
                        "region_id": str(region),
                        "pair_id": str(witness.pair_id),
                        "source_terminal_id": str(witness.source_terminal_id),
                        "target_terminal_id": str(witness.target_terminal_id),
                        "direct_fftt_s": float(witness.fftt_s),
                    }
                )
        sample_rows.extend(region_samples)
        rows.append(
            {
                "suite_id": suite.suite_id,
                "candidate_id": CANDIDATE_ID,
                "region_id": str(region),
                "region_node_count": int(oracle.support.graph.node_count),
                "region_arc_count": int(oracle.support.graph.arc_count),
                "terminal_count": int(len(region_terminals)),
                "source_terminal_count": int(len(sources)),
                "target_terminal_count": int(region_terminals["can_enter_region"].astype(bool).sum()),
                "supported_pair_count": int(pair_count),
                "enumeration_seconds": float(time.perf_counter() - region_started),
                "representation_mode": "EXACT_SPARSE_LAZY",
                "dense_terminal_square_allocated": False,
            }
        )
        if position % 100 == 0 or position == len(grouped):
            print(
                f"PHASE3_SUPPORT suite={suite.suite_id} regions={position}/{len(grouped)} "
                f"pairs={sum(row['supported_pair_count'] for row in rows)} "
                f"elapsed_s={time.perf_counter() - started:.3f}",
                flush=True,
            )
    result = pd.DataFrame(rows).sort_values("region_id", kind="mergesort").reset_index(drop=True)
    _write_parquet_new(destination, result)
    if visualization_cap > 0 and sample_rows:
        sample = (
            pd.DataFrame(sample_rows)
            .sort_values(["region_id", "pair_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        _write_parquet_new(Path(finalist_dir) / "regional_pair_visualization_sample.parquet", sample)
    del graph, suite
    gc.collect()
    return result


def sample_regional_pairs(
    op_config: Mapping[str, Any], finalist_dir: str | Path, *, visualization_cap: int
) -> pd.DataFrame:
    """Create a deterministic region-stratified QGIS subset without dense support."""

    destination = Path(finalist_dir) / "regional_pair_visualization_sample.parquet"
    if destination.exists():
        return pd.read_parquet(destination)
    if visualization_cap <= 0:
        raise Phase3ProductionError("Regional-pair visualization cap must be positive")
    paths = resolve_authority_paths(op_config)
    graph = load_graph(paths)
    suite = load_suite(paths, graph)
    terminals = pd.read_parquet(Path(finalist_dir) / "consolidated_terminals.parquet")
    terminal_members = pd.read_parquet(Path(finalist_dir) / "terminal_fine_members.parquet")
    uid_to_index = {str(value): index for index, value in enumerate(graph.node_uid.tolist())}
    fine_global = {
        str(terminal_id): tuple(uid_to_index[str(uid)] for uid in group["partC_rev1_node_uid"].astype(str))
        for terminal_id, group in terminal_members.groupby("terminal_id", sort=False)
    }
    grouped = list(terminals.groupby("region_id", sort=True))
    base_cap, extra_cap = divmod(visualization_cap, max(1, len(grouped)))
    rows: list[dict[str, Any]] = []
    for position, (region, region_terminals) in enumerate(grouped, 1):
        region_cap = base_cap + (1 if position <= extra_cap else 0)
        if region_cap <= 0:
            continue
        oracle = _build_region_oracle(
            graph, suite, CANDIDATE_ID, str(region), terminals, fine_global, retain_query_cache=False
        )
        sources = sorted(
            region_terminals.loc[
                region_terminals["can_exit_region"].astype(bool), "terminal_id"
            ].astype(str).tolist()
        )
        region_count = 0
        for source_id in sources:
            for witness in oracle.query(source_id):
                rows.append(
                    {
                        "suite_id": suite.suite_id,
                        "candidate_id": CANDIDATE_ID,
                        "region_id": str(region),
                        "pair_id": str(witness.pair_id),
                        "source_terminal_id": str(witness.source_terminal_id),
                        "target_terminal_id": str(witness.target_terminal_id),
                        "direct_fftt_s": float(witness.fftt_s),
                    }
                )
                region_count += 1
                if region_count >= region_cap:
                    break
            if region_count >= region_cap:
                break
        if position % 250 == 0 or position == len(grouped):
            print(
                f"PHASE3_PAIR_SAMPLE suite={suite.suite_id} regions={position}/{len(grouped)} rows={len(rows)}",
                flush=True,
            )
    if not rows:
        raise Phase3ProductionError("Regional-pair visualization sampling produced no rows")
    result = pd.DataFrame(rows).sort_values(["region_id", "pair_id"], kind="mergesort").reset_index(drop=True)
    if len(result) > visualization_cap:
        raise Phase3ProductionError("Regional-pair visualization sample exceeded its deterministic cap")
    _write_parquet_new(destination, result)
    del graph, suite
    gc.collect()
    return result


def export_suite(config: Mapping[str, Any], suite_short_id: str) -> dict[str, Any]:
    run_root = Path(str(config["output_root"])).resolve(strict=True)
    op_config = operation_config(config, suite_short_id)
    op_root = Path(op_config["output_root"]).resolve(strict=True)
    finalist = _finalist_dir(op_root)
    result_source = op_root / "round4_evaluation" / "candidates" / CANDIDATE_ID / "candidate_od_results.parquet"
    reproduction_source = op_root / "validation" / "topology_reproduction.json"
    for name in MANDATORY_TOPOLOGY_FILES:
        if not (finalist / name).is_file():
            raise Phase3ProductionError(f"Suite {suite_short_id} lacks {name}")
    if not result_source.is_file() or not reproduction_source.is_file():
        raise Phase3ProductionError(f"Suite {suite_short_id} lacks evaluation or reproduction evidence")
    support_summary = finalist / "regional_pair_support_summary.parquet"
    if not support_summary.is_file():
        enumerate_regional_pair_support(
            op_config,
            finalist,
            visualization_cap=int(config["storage"]["geopackage_regional_pair_visualization_cap_per_suite"]),
        )
    sample_path = finalist / "regional_pair_visualization_sample.parquet"
    if not sample_path.is_file():
        sample_regional_pairs(
            op_config,
            finalist,
            visualization_cap=int(config["storage"]["geopackage_regional_pair_visualization_cap_per_suite"]),
        )
    production = run_root / "suites" / suite_short_id / "parquet"
    if production.exists():
        return _verify_existing_export(config, suite_short_id, production)
    work = _new_export_staging(Path(op_config["temp_root"]))
    for source in sorted(finalist.rglob("*")):
        if source.is_file():
            _link_or_copy_new(source, work / source.relative_to(finalist))
    _link_or_copy_new(result_source, work / "candidate_od_results.parquet")
    shutil.copy2(reproduction_source, work / "reproduction.json")
    paths = resolve_authority_paths(op_config)
    authority = verify_authority(
        paths, verify_large_signatures=False, expected_identity=op_config["accepted_identities"]
    )
    graph = load_graph(paths)
    suite = load_suite(paths, graph)
    nodes, connectors, static_links = _normalized_online_graph(
        config, suite_short_id, graph, suite, work
    )
    _write_parquet_new(work / "acnhn_nodes.parquet", nodes)
    _write_parquet_new(work / "terminal_connectors.parquet", connectors)
    _write_parquet_new(work / "acnhn_static_links.parquet", static_links)
    support = pd.read_parquet(work / "regional_pair_support_summary.parquet")
    results = pd.read_parquet(work / "candidate_od_results.parquet")
    validation = _validate_production_suite(
        config, suite_short_id, graph, suite, work, nodes, connectors, static_links, support, results
    )
    _write_json_new(work / "validation.json", validation)
    metadata = _suite_metadata(
        config, suite_short_id, graph, suite, work, nodes, connectors, static_links, support, results
    )
    _write_json_new(work / "suite_metadata.json", metadata)
    resources = _suite_resources(op_root)
    _write_json_new(work / "resource_usage.json", resources)
    manifest = _artifact_manifest(work)
    _write_csv_new(work / "artifact_manifest.csv", manifest)
    del graph, suite, nodes, connectors, static_links, support, results
    gc.collect()
    production.parent.mkdir(parents=True, exist_ok=True)
    os.replace(work, production)
    return {
        "status": "PASS",
        "suite_short_id": suite_short_id,
        "production_directory": str(production),
        "authority_status": authority["status"],
        "artifact_count": len(manifest),
        "metadata": metadata,
    }


def _new_export_staging(temp_suite: Path) -> Path:
    temp_suite.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 1000):
        path = temp_suite / f"production_export_attempt_{attempt:03d}"
        if not path.exists():
            path.mkdir()
            return path
    raise Phase3ProductionError(f"No free export staging directory remains under {temp_suite}")


def _verify_existing_export(
    config: Mapping[str, Any], suite_short_id: str, production: Path
) -> dict[str, Any]:
    manifest_path = production / "artifact_manifest.csv"
    if not manifest_path.is_file():
        raise Phase3ProductionError(f"Existing suite export has no artifact manifest: {production}")
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8-sig", newline="")))
    declared = {str(row["relative_path"]) for row in rows} | {"artifact_manifest.csv"}
    actual = {path.relative_to(production).as_posix() for path in production.rglob("*") if path.is_file()}
    failures = []
    for row in rows:
        path = production.joinpath(*PurePosixPath(str(row["relative_path"])).parts)
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
            failures.append(f"missing_or_size:{row['relative_path']}")
        elif partC_rev1_file_sha256(path) != str(row["sha256"]):
            failures.append(f"sha256:{row['relative_path']}")
    if actual != declared:
        failures.append("artifact_closure")
    if failures:
        raise Phase3ProductionError(f"Existing suite export failed validation: {failures[:10]}")
    metadata = json.loads((production / "suite_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("suite_short_id") != suite_short_id or metadata.get("method_id") != METHOD_ID:
        raise Phase3ProductionError("Existing suite export metadata differs from the selected method")
    return {
        "status": "PASS",
        "suite_short_id": suite_short_id,
        "production_directory": str(production),
        "authority_status": "PASS_REUSED_VERIFIED_EXPORT",
        "artifact_count": len(rows),
        "metadata": metadata,
    }


def _normalized_online_graph(
    config: Mapping[str, Any],
    suite_short_id: str,
    graph: Any,
    suite: Any,
    production: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    record = suite_record(config, suite_short_id)
    membership = pd.read_parquet(production / "bacn_membership.parquet")
    bacn_summary = pd.read_parquet(production / "bacn_summary.parquet")
    racn_membership = pd.read_parquet(production / "racn_membership.parquet")
    racn_summary = pd.read_parquet(production / "racn_summary.parquet")
    terminals = pd.read_parquet(production / "consolidated_terminals.parquet")
    terminal_members = pd.read_parquet(production / "terminal_fine_members.parquet")
    structural = pd.read_parquet(production / "structural_internal_charges.parquet")
    bundle_summary = pd.read_parquet(production / "bundle_summary.parquet")
    coordinates = pd.DataFrame(
        {
            "partC_rev1_node_uid": graph.node_uid.astype(str),
            "longitude": graph.lon,
            "latitude": graph.lat,
            "state_fips": graph.state_fips.astype(str),
            "physical_weak_component_id": graph.physical_weak_component.astype(np.int64),
        }
    )
    bacn_coords = (
        membership[["bacn_id", "partC_rev1_node_uid"]]
        .merge(coordinates, on="partC_rev1_node_uid", how="left", validate="many_to_one")
        .sort_values(["bacn_id", "partC_rev1_node_uid"], kind="mergesort")
        .groupby("bacn_id", sort=True, as_index=False)
        .agg(longitude=("longitude", "mean"), latitude=("latitude", "mean"), state_fips=("state_fips", "first"))
    )
    bacn_nodes = (
        bacn_summary.merge(bacn_coords, on="bacn_id", validate="one_to_one")
        .merge(structural[["bacn_id", "structural_charge_s"]], on="bacn_id", validate="one_to_one")
        .rename(columns={"bacn_id": "node_id"})
    )
    bacn_nodes = pd.DataFrame(
        {
            "node_id": bacn_nodes["node_id"].astype(str),
            "node_type": "BACN",
            "terminal_type": None,
            "parent_object_id": None,
            "region_id": None,
            "member_count": bacn_nodes["member_count"].astype(np.int64),
            "physical_weak_component_id": bacn_nodes["physical_weak_component_id"].astype(np.int64),
            "structural_charge_s": bacn_nodes["structural_charge_s"].astype(float),
            "longitude": bacn_nodes["longitude"].astype(float),
            "latitude": bacn_nodes["latitude"].astype(float),
            "state_fips": bacn_nodes["state_fips"].astype(str),
        }
    )
    if racn_membership.empty:
        racn_nodes = bacn_nodes.iloc[0:0].copy()
    else:
        racn_coords = (
            racn_membership[["racn_id", "physical_portal_node_uid"]]
            .rename(columns={"physical_portal_node_uid": "partC_rev1_node_uid"})
            .merge(coordinates, on="partC_rev1_node_uid", how="left", validate="many_to_one")
            .sort_values(["racn_id", "partC_rev1_node_uid"], kind="mergesort")
            .groupby("racn_id", sort=True, as_index=False)
            .agg(longitude=("longitude", "mean"), latitude=("latitude", "mean"), state_fips=("state_fips", "first"))
        )
        merged = racn_summary.merge(racn_coords, on="racn_id", validate="one_to_one")
        racn_nodes = pd.DataFrame(
            {
                "node_id": merged["racn_id"].astype(str),
                "node_type": "RACN",
                "terminal_type": None,
                "parent_object_id": None,
                "region_id": None,
                "member_count": merged["member_count"].astype(np.int64),
                "physical_weak_component_id": merged["physical_weak_component_id"].astype(np.int64),
                "structural_charge_s": 0.0,
                "longitude": merged["longitude"].astype(float),
                "latitude": merged["latitude"].astype(float),
                "state_fips": merged["state_fips"].astype(str),
            }
        )
    terminal_coords = (
        terminal_members[["terminal_id", "partC_rev1_node_uid"]]
        .merge(coordinates, on="partC_rev1_node_uid", how="left", validate="many_to_one")
        .sort_values(["terminal_id", "partC_rev1_node_uid"], kind="mergesort")
        .groupby("terminal_id", sort=True, as_index=False)
        .agg(longitude=("longitude", "mean"), latitude=("latitude", "mean"), state_fips=("state_fips", "first"))
    )
    terminal_merged = terminals.merge(terminal_coords, on="terminal_id", validate="one_to_one")
    declared_racn_parents = set(
        terminals.loc[terminals["terminal_type"] == "RACN", "parent_object_id"].astype(str)
    )
    summarized_racn_ids = set(racn_nodes["node_id"].astype(str))
    singleton_control_ids = declared_racn_parents.difference(summarized_racn_ids)
    if singleton_control_ids:
        control_members = (
            terminals.loc[
                (terminals["terminal_type"] == "RACN")
                & terminals["parent_object_id"].astype(str).isin(singleton_control_ids),
                ["terminal_id", "parent_object_id"],
            ]
            .merge(
                terminal_members[["terminal_id", "partC_rev1_node_uid"]],
                on="terminal_id",
                how="left",
                validate="one_to_many",
            )
            .drop_duplicates(["parent_object_id", "partC_rev1_node_uid"])
            .merge(coordinates, on="partC_rev1_node_uid", how="left", validate="many_to_one")
        )
        control_summary = (
            control_members.groupby("parent_object_id", sort=True, as_index=False)
            .agg(
                member_count=("partC_rev1_node_uid", "nunique"),
                longitude=("longitude", "mean"),
                latitude=("latitude", "mean"),
                state_fips=("state_fips", "first"),
                physical_weak_component_id=("physical_weak_component_id", "first"),
                weak_component_count=("physical_weak_component_id", "nunique"),
            )
        )
        if (
            set(control_summary["parent_object_id"].astype(str)) != singleton_control_ids
            or not (control_summary["member_count"] == 1).all()
            or not (control_summary["weak_component_count"] == 1).all()
        ):
            raise Phase3ProductionError("Terminal-declared singleton R-ACN control closure is invalid")
        control_nodes = pd.DataFrame(
            {
                "node_id": control_summary["parent_object_id"].astype(str),
                "node_type": "RACN",
                "terminal_type": None,
                "parent_object_id": None,
                "region_id": None,
                "member_count": control_summary["member_count"].astype(np.int64),
                "physical_weak_component_id": control_summary["physical_weak_component_id"].astype(np.int64),
                "structural_charge_s": 0.0,
                "longitude": control_summary["longitude"].astype(float),
                "latitude": control_summary["latitude"].astype(float),
                "state_fips": control_summary["state_fips"].astype(str),
            }
        )
        racn_nodes = pd.concat([racn_nodes, control_nodes], ignore_index=True)
    terminal_nodes = pd.DataFrame(
        {
            "node_id": terminal_merged["terminal_id"].astype(str),
            "node_type": "TERMINAL",
            "terminal_type": terminal_merged["terminal_type"].astype(str),
            "parent_object_id": terminal_merged["parent_object_id"].astype(str),
            "region_id": terminal_merged["region_id"].astype(str),
            "member_count": (
                terminal_merged["fine_member_offset_end"] - terminal_merged["fine_member_offset_begin"]
            ).astype(np.int64),
            "physical_weak_component_id": -1,
            "structural_charge_s": 0.0,
            "longitude": terminal_merged["longitude"].astype(float),
            "latitude": terminal_merged["latitude"].astype(float),
            "state_fips": terminal_merged["state_fips"].astype(str),
        }
    )
    nodes = pd.concat([bacn_nodes, racn_nodes, terminal_nodes], ignore_index=True)
    nodes.insert(0, "candidate_id", CANDIDATE_ID)
    nodes.insert(0, "resolution", float(record["resolution"]))
    nodes.insert(0, "phase2_suite_id", str(record["phase2_suite_id"]))
    nodes.insert(0, "suite_short_id", suite_short_id)
    nodes.insert(0, "phase3_run_id", str(config["run_id"]))
    nodes = nodes.sort_values(["node_type", "node_id"], kind="mergesort").reset_index(drop=True)
    if nodes["node_id"].duplicated().any() or nodes[["longitude", "latitude"]].isna().any().any():
        raise Phase3ProductionError("Normalized online node registry is invalid")

    connector_rows: list[dict[str, Any]] = []
    for row in terminals.loc[terminals["terminal_type"].isin(["BACN", "RACN"])].itertuples(index=False):
        terminal_id, parent = str(row.terminal_id), str(row.parent_object_id)
        if bool(row.can_exit_region):
            connector_rows.append(
                _connector_row(suite, terminal_id, parent, str(row.region_id), "parent_to_region")
            )
        if bool(row.can_enter_region):
            connector_rows.append(
                _connector_row(suite, terminal_id, parent, str(row.region_id), "region_to_parent")
            )
    connectors = pd.DataFrame(connector_rows).sort_values("link_id", kind="mergesort").reset_index(drop=True)
    bundles = pd.DataFrame(
        {
            "link_id": bundle_summary["bundle_id"].astype(str),
            "link_type": "BACKBONE_BUNDLE",
            "tail_node_id": bundle_summary["tail_bacn_id"].astype(str),
            "head_node_id": bundle_summary["head_bacn_id"].astype(str),
            "region_id": None,
            "direction_role": "directed_backbone_bundle",
            "member_object_count": bundle_summary["member_arc_count"].astype(np.int64),
            "direct_base_fftt_s": bundle_summary["direct_diagnostic_fftt_s"].astype(float),
            "minimum_witness_arc_id": bundle_summary["minimum_witness_arc_id"].astype(str),
        }
    )
    static_links = pd.concat([bundles, connectors], ignore_index=True)
    charge = dict(zip(nodes["node_id"].astype(str), nodes["structural_charge_s"].astype(float), strict=True))
    static_links["tail_half_charge_s"] = static_links["tail_node_id"].map(charge).fillna(0.0) * 0.5
    static_links["head_half_charge_s"] = static_links["head_node_id"].map(charge).fillna(0.0) * 0.5
    static_links["structural_seed_fftt_s"] = (
        static_links["direct_base_fftt_s"]
        + static_links["tail_half_charge_s"]
        + static_links["head_half_charge_s"]
    )
    static_links.insert(0, "candidate_id", CANDIDATE_ID)
    static_links.insert(0, "resolution", float(record["resolution"]))
    static_links.insert(0, "phase2_suite_id", str(record["phase2_suite_id"]))
    static_links.insert(0, "suite_short_id", suite_short_id)
    static_links.insert(0, "phase3_run_id", str(config["run_id"]))
    static_links = static_links.sort_values(["link_type", "link_id"], kind="mergesort").reset_index(drop=True)
    node_ids = set(nodes["node_id"].astype(str))
    if static_links["link_id"].duplicated().any() or not set(static_links["tail_node_id"]).issubset(node_ids) or not set(static_links["head_node_id"]).issubset(node_ids):
        raise Phase3ProductionError("Normalized static-link registry has duplicate IDs or missing endpoints")
    return nodes, connectors, static_links


def _connector_row(
    suite: Any, terminal_id: str, parent: str, region_id: str, direction: str
) -> dict[str, Any]:
    if direction == "parent_to_region":
        tail, head = parent, terminal_id
    else:
        tail, head = terminal_id, parent
    return {
        "link_id": stable_object_id(
            "terminal_connector",
            suite_id=suite.suite_id,
            candidate_id=CANDIDATE_ID,
            members=[tail, head],
            ordered_members=True,
            signature={"direction": direction},
        ),
        "link_type": "TERMINAL_CONNECTOR",
        "tail_node_id": tail,
        "head_node_id": head,
        "region_id": region_id,
        "direction_role": direction,
        "member_object_count": 1,
        "direct_base_fftt_s": 0.0,
        "minimum_witness_arc_id": None,
    }


def _validate_production_suite(
    config: Mapping[str, Any],
    suite_short_id: str,
    graph: Any,
    suite: Any,
    production: Path,
    nodes: pd.DataFrame,
    connectors: pd.DataFrame,
    static_links: pd.DataFrame,
    support: pd.DataFrame,
    results: pd.DataFrame,
) -> dict[str, Any]:
    membership = pd.read_parquet(production / "bacn_membership.parquet")
    bacn_summary = pd.read_parquet(production / "bacn_summary.parquet")
    bundles = pd.read_parquet(production / "bundle_membership.parquet")
    suppressed = pd.read_parquet(production / "suppressed_backbone_arcs.parquet")
    racn = pd.read_parquet(production / "racn_membership.parquet")
    terminals = pd.read_parquet(production / "consolidated_terminals.parquet")
    validate_bundle_closure(suite.is_final_backbone_arc, graph.arc_id, bundles, suppressed)
    expected_nodes = set(graph.node_uid[suite.is_final_backbone_node].astype(str))
    checks = {
        "bacn_partition_exact": (
            not membership["partC_rev1_node_uid"].duplicated().any()
            and set(membership["partC_rev1_node_uid"].astype(str)) == expected_nodes
        ),
        "gc_pairwise_diameter_at_most_1000m": bool(
            bacn_summary["production_pairwise_valid"].all()
            and (bacn_summary["exact_geodesic_diameter_m"] <= 1000.0 + 1e-6).all()
        ),
        "bundle_suppressed_arc_closure": len(bundles) + len(suppressed) == int(suite.is_final_backbone_arc.sum()),
        "racn_membership_unique": racn.empty or not racn["physical_portal_node_uid"].duplicated().any(),
        "racn_parent_nodes_complete": set(
            terminals.loc[terminals["terminal_type"] == "RACN", "parent_object_id"].astype(str)
        ).issubset(set(nodes.loc[nodes["node_type"] == "RACN", "node_id"].astype(str))),
        "terminal_ids_unique": not terminals["terminal_id"].duplicated().any(),
        "centroid_non_through": not terminals.loc[
            terminals["terminal_type"].isin(["ORIGIN_CENTROID", "DESTINATION_CENTROID"]),
            "centroid_through_allowed",
        ].any(),
        "support_all_regions": len(support) == len(suite.region_ids) and set(support["region_id"].astype(str)) == set(suite.region_ids),
        "support_exact_sparse_lazy": bool(
            (support["representation_mode"] == "EXACT_SPARSE_LAZY").all()
            and not support["dense_terminal_square_allocated"].any()
            and int(support["supported_pair_count"].sum()) > 0
        ),
        "online_node_ids_unique": not nodes["node_id"].duplicated().any(),
        "static_link_ids_unique": not static_links["link_id"].duplicated().any(),
        "connector_count_nonzero": len(connectors) > 0,
        "common_200_od_direct_structural": (
            len(results) == 400
            and results["sample_od_id"].nunique() == 200
            and results.groupby("diagnostic_mode").size().to_dict() == {"direct": 200, "structural": 200}
        ),
        "forward_projection_complete": bool(
            results["projected_coarse_path_id"].notna().all()
            and results["projected_generic_fftt_s"].notna().all()
        ),
        "phase4_not_executed": config["phase_boundaries"]["phase4_authorized"] is False,
    }
    failed = [key for key, value in checks.items() if not bool(value)]
    if failed:
        raise Phase3ProductionError(f"Suite {suite_short_id} validation failed: {failed}")
    return {
        "schema_id": "phase3.production_suite_validation.v1",
        "status": "PASS",
        "suite_short_id": suite_short_id,
        "phase2_suite_id": suite.suite_id,
        "checks": checks,
        "forward_projection_is_hard_gate": True,
        "reverse_liftability_is_hard_gate": False,
        "accepted_phase0_2_artifacts_modified": False,
        "validated_utc": utc_now(),
    }


def _suite_metadata(
    config: Mapping[str, Any],
    suite_short_id: str,
    graph: Any,
    suite: Any,
    production: Path,
    nodes: pd.DataFrame,
    connectors: pd.DataFrame,
    static_links: pd.DataFrame,
    support: pd.DataFrame,
    results: pd.DataFrame,
) -> dict[str, Any]:
    record = suite_record(config, suite_short_id)
    bacn_summary = pd.read_parquet(production / "bacn_summary.parquet")
    racn_summary = pd.read_parquet(production / "racn_summary.parquet")
    bundle_summary = pd.read_parquet(production / "bundle_summary.parquet")
    membership = pd.read_parquet(production / "bacn_membership.parquet")
    bacn_count = int(len(bacn_summary))
    backbone_nodes = int(suite.is_final_backbone_node.sum())
    actual_nodes = int(len(nodes))
    split_nodes = actual_nodes + bacn_count
    actual_static = int(len(static_links))
    exact_pairs = int(support["supported_pair_count"].sum())
    return {
        "schema_id": "phase3.production_suite_metadata.v1",
        "phase3_run_id": config["run_id"],
        "suite_short_id": suite_short_id,
        "phase2_suite_id": record["phase2_suite_id"],
        "resolution": float(record["resolution"]),
        "candidate_id": CANDIDATE_ID,
        "method_id": METHOD_ID,
        "physical_graph_node_count": graph.node_count,
        "physical_graph_directed_arc_count": graph.arc_count,
        "final_backbone_node_count": backbone_nodes,
        "final_backbone_arc_count": int(suite.is_final_backbone_arc.sum()),
        "bacn_member_count": int(len(membership)),
        "bacn_count": bacn_count,
        "bacn_raw_reduction_fraction": 1.0 - bacn_count / backbone_nodes,
        "bacn_max_pairwise_diameter_m": float(bacn_summary["exact_geodesic_diameter_m"].max()),
        "racn_count": int((nodes["node_type"] == "RACN").sum()),
        "summarized_fixed_rule_racn_count": int(len(racn_summary)),
        "singleton_ineligible_racn_control_count": int(
            (nodes["node_type"] == "RACN").sum() - len(racn_summary)
        ),
        "terminal_count": int((nodes["node_type"] == "TERMINAL").sum()),
        "bundle_count": int(len(bundle_summary)),
        "terminal_connector_count": int(len(connectors)),
        "actual_online_node_count": actual_nodes,
        "split_equivalent_online_node_count": split_nodes,
        "actual_static_link_count": actual_static,
        "split_equivalent_static_link_count": actual_static + bacn_count,
        "exact_regional_pair_count": exact_pairs,
        "full_logical_link_count": actual_static + exact_pairs,
        "regional_support_representation": "EXACT_SPARSE_LAZY",
        "regional_support_region_count": int(len(support)),
        "direct_od_rows": int((results["diagnostic_mode"] == "direct").sum()),
        "structural_od_rows": int((results["diagnostic_mode"] == "structural").sum()),
        "unique_od_count": int(results["sample_od_id"].nunique()),
        "phase4_executed": False,
        "generated_utc": utc_now(),
    }


def _suite_resources(op_root: Path) -> dict[str, Any]:
    receipts = []
    for path in sorted((op_root / "receipts").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        receipts.append(
            {
                "task_id": payload.get("task_id"),
                "operation": payload.get("operation"),
                "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
                "peak_process_tree_rss_bytes": int(payload.get("peak_process_tree_rss_bytes", 0)),
                "peak_process_count": int(payload.get("peak_process_count", 0)),
                "added_disk_bytes": int(payload.get("added_disk_bytes", 0)),
                "status": payload.get("status"),
            }
        )
    return {
        "schema_id": "phase3.production_suite_resources.v1",
        "status": "PASS" if receipts and all(row["status"] == "PASS" for row in receipts) else "PASS_WITH_REUSE",
        "operations": receipts,
        "elapsed_seconds_sum": sum(row["elapsed_seconds"] for row in receipts),
        "peak_process_tree_rss_bytes": max((row["peak_process_tree_rss_bytes"] for row in receipts), default=0),
        "peak_process_count": max((row["peak_process_count"] for row in receipts), default=0),
        "added_disk_bytes_sum": sum(row["added_disk_bytes"] for row in receipts),
    }


def _artifact_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "artifact_manifest.csv"):
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": partC_rev1_file_sha256(path),
            }
        )
    if not rows:
        raise Phase3ProductionError("Production suite artifact manifest would be empty")
    return pd.DataFrame(rows)


def _verify_common_evaluation_inputs(
    config: Mapping[str, Any], sample_path: Path, paths_path: Path
) -> None:
    if partC_rev1_file_sha256(sample_path) != config["common_evaluation"]["od_sample_sha256"]:
        raise Phase3ProductionError("Common 200-OD sample hash changed")
    if partC_rev1_file_sha256(paths_path) != config["common_evaluation"]["physical_paths_sha256"]:
        raise Phase3ProductionError("Common physical-path hash changed")
    sample = pd.read_parquet(sample_path)
    paths = pd.read_parquet(paths_path)
    if len(sample) != 200 or sample["origin_idx"].nunique() != 40 or len(paths) != 200:
        raise Phase3ProductionError("Common evaluation inputs do not contain 200 ODs and 40 origins")
    if set(sample["sample_od_id"].astype(str)) != set(paths["sample_od_id"].astype(str)):
        raise Phase3ProductionError("Common OD sample and physical paths do not close")


def _operation_complete(op_root: Path, task_id: str) -> bool:
    path = op_root / "receipts" / f"{task_id}.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or int(payload.get("meaningful_output_count", 0)) <= 0:
        raise Phase3ProductionError(f"Existing operation receipt is not a verified pass: {path}")
    for row in payload.get("output_manifest", []):
        artifact = Path(str(row["path"]))
        if not artifact.is_file() or artifact.stat().st_size != int(row["size_bytes"]):
            raise Phase3ProductionError(f"Completed operation artifact is missing or resized: {artifact}")
        if partC_rev1_file_sha256(artifact) != str(row["sha256"]):
            raise Phase3ProductionError(f"Completed operation artifact hash changed: {artifact}")
    return True


def _finalist_dir(op_root: Path) -> Path:
    path = op_root / "round3_finalists" / "candidates" / CANDIDATE_ID
    return path.resolve(strict=True)


def _mark_suite_complete(run_root: Path, suite_short_id: str) -> None:
    status_path = run_root / "RUN_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    completed = list(status.get("completed_suites", []))
    if suite_short_id not in completed:
        completed.append(suite_short_id)
    completed.sort(key=SUITE_ORDER.index)
    status["completed_suites"] = completed
    status["all_six_suites_built"] = set(completed) == set(SUITE_ORDER)
    required = tuple(str(value) for value in status.get("required_suite_ids", SUITE_ORDER))
    status["all_required_production_suites_built"] = set(required).issubset(completed)
    status["updated_utc"] = utc_now()
    _replace_json(status_path, status)


def _link_or_copy_new(source: Path, target: Path) -> str:
    source = source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise Phase3ProductionError(f"Write-new-only target exists: {target}")
    try:
        os.link(source, target)
        return "HARDLINK"
    except OSError:
        shutil.copy2(source, target)
        return "COPY"


def _link_or_verify(source: Path, target: Path) -> str:
    source = source.resolve(strict=True)
    if target.exists():
        if not target.is_file() or target.stat().st_size != source.stat().st_size:
            raise Phase3ProductionError(f"Existing resumed target differs in type or size: {target}")
        if partC_rev1_file_sha256(target) != partC_rev1_file_sha256(source):
            raise Phase3ProductionError(f"Existing resumed target differs by SHA-256: {target}")
        return "VERIFIED_EXISTING"
    return _link_or_copy_new(source, target)


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Phase3ProductionError(f"Write-new-only JSON path exists: {path}")
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise Phase3ProductionError(f"Atomic status temporary path exists: {temporary}")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or len(frame) == 0:
        raise Phase3ProductionError(f"Parquet path exists or frame is empty: {path}")
    frame.to_parquet(path, index=False)


def _write_csv_new(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or len(frame) == 0:
        raise Phase3ProductionError(f"CSV path exists or frame is empty: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def _git_snapshot(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), "-c", "core.longpaths=true", *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout

    return {
        "branch": run("branch", "--show-current").strip(),
        "head": run("rev-parse", "HEAD").strip(),
        "status_porcelain": run("status", "--porcelain=v1", "-uall").splitlines(),
        "binary_diff_line_count": len(run("diff", "--binary").splitlines()),
        "untracked": run("ls-files", "--others", "--exclude-standard").splitlines(),
    }


def resource_snapshot(run_root: Path, temp_root: Path) -> dict[str, Any]:
    rss, process_count = process_tree_rss(os.getpid())
    return {
        "captured_utc": utc_now(),
        "process_tree_rss_bytes": int(rss),
        "process_count": int(process_count),
        "run_root_bytes": directory_bytes(run_root),
        "temp_root_bytes": directory_bytes(temp_root),
        "drive_free_bytes": int(shutil.disk_usage(run_root).free),
    }
