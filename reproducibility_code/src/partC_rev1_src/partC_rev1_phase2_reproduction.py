"""Independent Phase 2 portfolio replay and artifact-closure validation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pandas as pd

from .partC_rev1_hashing import partC_rev1_dataframe_records_hash, partC_rev1_json_hash
from .partC_rev1_partition_igraph import partC_rev1_run_leiden_partition
from .partC_rev1_phase2_split_mapping import (
    PHASE_ID,
    STEP_ID,
    ResourceTracker,
    _build_portals,
    _correct_small_regions,
    _incident_registry,
    _initial_arc_membership,
    _region_summary,
    _sha256,
    _stable_region_ids,
    _write_json,
    build_initial_backbone,
    input_signatures,
    load_authoritative_inputs,
    load_config,
)


def _validate_json(payload_path: Path, schema_path: Path) -> None:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(payload, schema)


def _artifact_closure(config: dict[str, Any], output_root: Path) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    expected = set(config["expected_suite_outputs"])
    suite_manifest = pd.read_csv(output_root / "portfolio" / "suite_manifest.csv")
    all_expected = True
    all_manifest_hashes = True
    all_keys = True
    all_refs = True
    for suite in suite_manifest.itertuples(index=False):
        suite_dir = output_root / "suites" / str(suite.suite_id)
        actual = {path.name for path in suite_dir.iterdir() if path.is_file()}
        missing = sorted(expected - actual)
        extras = sorted(actual - expected)
        all_expected &= not missing and not extras
        manifest = pd.read_csv(suite_dir / "artifact_manifest.csv")
        manifest_ok = True
        for artifact in manifest.itertuples(index=False):
            path = suite_dir / str(artifact.relative_path)
            manifest_ok &= path.is_file() and path.stat().st_size == int(artifact.size_bytes) and _sha256(path) == str(artifact.sha256)
        all_manifest_hashes &= manifest_ok
        arc_membership = pd.read_parquet(suite_dir / "directed_arc_membership.parquet")
        nodes = pd.read_parquet(suite_dir / "node_incident_registry.parquet")
        regions = pd.read_parquet(suite_dir / "retained_regions.parquet")
        portals = pd.read_parquet(suite_dir / "portals.parquet")
        roles = pd.read_parquet(suite_dir / "centroid_roles.parquet")
        terminals = pd.read_parquet(suite_dir / "terminals.parquet")
        keys_ok = (
            len(arc_membership) == int(config["input_versions"]["expected_arc_count"])
            and arc_membership["directed_arc_id"].is_unique
            and len(nodes) == int(config["input_versions"]["expected_node_count"])
            and nodes["partC_rev1_node_uid"].is_unique
            and regions["region_uuid"].is_unique
            and portals["portal_id"].is_unique
            and roles["centroid_role_id"].is_unique
            and terminals["terminal_id"].is_unique
        )
        retained = set(regions["region_uuid"].astype(str))
        refs_ok = (
            set(arc_membership.loc[~arc_membership["is_final_backbone"], "suite_label"].astype(str)).issubset(retained)
            and set(portals["region_uuid"].astype(str)).issubset(retained)
            and set(roles["region_uuid"].astype(str)).issubset(retained)
            and set(terminals["region_uuid"].astype(str)).issubset(retained)
        )
        all_keys &= bool(keys_ok)
        all_refs &= bool(refs_ok)
        rows.append(
            {
                "suite_id": str(suite.suite_id),
                "missing_expected_files": missing,
                "unexpected_files": extras,
                "artifact_manifest_hashes_pass": bool(manifest_ok),
                "stable_keys_pass": bool(keys_ok),
                "referential_integrity_pass": bool(refs_ok),
            }
        )
    return {
        "all_suite_file_sets_exact": all_expected,
        "all_suite_manifest_hashes_pass": all_manifest_hashes,
        "all_stable_keys_pass": all_keys,
        "all_referential_integrity_pass": all_refs,
    }, rows


def run_reproduction(config: dict[str, Any], schema_root: Path) -> dict[str, Any]:
    output_root = Path(config["output_root"]).resolve()
    portfolio_dir = output_root / "portfolio"
    target = portfolio_dir / "reproduction_validation.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite reproduction evidence: {target}")
    tracker = ResourceTracker.begin()
    before = input_signatures(config)
    _validate_json(Path(config["_config_path"]), schema_root / "phase2_execution_config.schema.json")
    _validate_json(portfolio_dir / "portfolio_summary.json", schema_root / "phase2_portfolio_summary.schema.json")
    closure_checks, closure_rows = _artifact_closure(config, output_root)
    suite_manifest = pd.read_csv(portfolio_dir / "suite_manifest.csv").sort_values("resolution", kind="mergesort")
    for suite in suite_manifest.itertuples(index=False):
        suite_dir = output_root / "suites" / str(suite.suite_id)
        _validate_json(suite_dir / "suite_identity.json", schema_root / "phase2_suite_identity.schema.json")
        _validate_json(suite_dir / "validation.json", schema_root / "phase2_suite_validation.schema.json")

    nodes, arcs, origin_nodes, destination_nodes = load_authoritative_inputs(config, tracker)
    del origin_nodes, destination_nodes
    partition_nodes, initial_arcs, initial_identity = build_initial_backbone(nodes, arcs, config)
    nonbackbone = initial_arcs.loc[~initial_arcs["is_final_backbone"].fillna(False).astype(bool)].copy()
    replay_rows: list[dict[str, Any]] = []
    all_match = True
    for expected in suite_manifest.itertuples(index=False):
        partition = partC_rev1_run_leiden_partition(
            partition_nodes,
            nonbackbone,
            {"backbone_partition": config["backbone_partition"]},
            resolution=float(expected.resolution),
            random_seed=int(config["random_seed"]),
        )
        stable_partition, _ = _stable_region_ids(partition["node_partition"])
        labels, _arc_type, _post_partition, node_primary = _initial_arc_membership(initial_arcs, partition_nodes, stable_partition)
        registry_before = _incident_registry(nodes, initial_arcs, labels, node_primary)
        corrected, audit, _ = _correct_small_regions(labels, registry_before, initial_arcs, config["small_region_rule"])
        corrected_regions = set(audit.loc[audit["too_small_region"], "region_uuid"].astype(str))
        corrected_primary = node_primary.astype(object).copy()
        corrected_primary[np.isin(corrected_primary, list(corrected_regions))] = "backbone"
        corrected_primary[corrected_primary == ""] = "backbone"
        registry = _incident_registry(nodes, initial_arcs, corrected, corrected_primary.astype(str))
        retained = _region_summary(corrected, registry, initial_arcs)
        portals = _build_portals(str(expected.suite_id), nodes, initial_arcs, corrected, registry)
        final = corrected == "backbone"
        replay_membership = pd.DataFrame(
            {
                "directed_arc_id": initial_arcs["directed_arc_id"].astype(str),
                "suite_label": corrected,
                "is_final_backbone": final,
            }
        )
        membership_hash = partC_rev1_dataframe_records_hash(
            replay_membership, ["directed_arc_id", "suite_label", "is_final_backbone"]
        )
        node_hash = partC_rev1_dataframe_records_hash(
            registry, ["partC_rev1_node_uid", "primary_partition_region_uuid", "incident_labels"]
        )
        structural = partC_rev1_json_hash(
            {
                "membership_hash": membership_hash,
                "node_membership_hash": node_hash,
                "region_count": len(retained),
                "final_backbone_arc_count": int(final.sum()),
                "portal_count": len(portals),
            }
        )
        matches = (
            membership_hash == str(expected.membership_hash)
            and node_hash == str(expected.node_membership_hash)
            and structural == str(expected.structural_signature)
            and len(retained) == int(expected.retained_region_count)
            and int(final.sum()) == int(expected.final_backbone_directed_arc_count)
        )
        all_match &= matches
        replay_rows.append(
            {
                "suite_id": str(expected.suite_id),
                "resolution": float(expected.resolution),
                "membership_hash": membership_hash,
                "node_membership_hash": node_hash,
                "structural_signature": structural,
                "retained_region_count": len(retained),
                "final_backbone_directed_arc_count": int(final.sum()),
                "matches_accepted_suite": bool(matches),
            }
        )
        tracker.sample()
    after = input_signatures(config)
    checks = {
        **closure_checks,
        "config_schema_pass": True,
        "portfolio_schema_pass": True,
        "all_suite_schemas_pass": True,
        "all_six_replays_exact": bool(all_match and len(replay_rows) == 6),
        "initial_backbone_identity_exact": initial_identity["initial_backbone_membership_hash"]
        == "8196d407b2777b27f8e572dd2127cb48838b50db1de0ef16556054ecc0796eab",
        "input_invariance": before == after,
        "phase_3_authorized_false": not bool(config.get("phase_3_authorized")),
    }
    payload = {
        "schema_version": "1.0",
        "phase_id": PHASE_ID,
        "step_id": STEP_ID,
        "campaign_id": config["campaign_id"],
        "run_id": config["run_id"],
        "task_id": "P2-VALIDATE",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "suite_closure": closure_rows,
        "replays": replay_rows,
        "resource_use": {
            "wall_clock_seconds": tracker.elapsed_seconds,
            "peak_process_tree_rss_bytes": tracker.peak_rss_bytes,
        },
        "input_signatures_before": before,
        "input_signatures_after": after,
        "phase_3_authorized": False,
    }
    _write_json(target, payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"Phase 2 reproduction validation failed: {checks}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--schema-root", required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--confirm-real-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if (args.phase_id, args.step_id, args.run_id, args.task_id) != (
        PHASE_ID,
        STEP_ID,
        config["run_id"],
        "P2-VALIDATE",
    ):
        raise ValueError("Validation command identity mismatch")
    if args.task_id not in config["tasks"] or not args.confirm_real_run:
        raise ValueError("Explicit P2-VALIDATE authorization and --confirm-real-run are required")
    result = run_reproduction(config, Path(args.schema_root).resolve())
    print(json.dumps({"status": result["status"], "replay_count": len(result["replays"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
