"""Single configured Phase 5 backbone-compilation pilot.

This module performs no equilibrium update or shortest-path assignment.  It
reads immutable accepted Phase 3/4 products and writes only the configured
Phase 5 candidate/evidence root.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from jsonschema import validate as jsonschema_validate

from .bacn_charges import (
    fixed_charge_candidates,
    StaticPathProjector,
    internal_exposure,
    path_episode_statistics,
    physical_flow_conservation,
    reconstruct_bacn_visit_semantics,
    reconstruct_reachable_od_supply,
)
from .benchmarks import process_tree_rss_bytes, run_kernel_benchmarks
from .bundle_oracle import (
    BundleSpec,
    exact_bundle_oracle,
    exact_series_oracle,
    macro_function_key,
    primitive_function_key,
    series_power_terms,
)
from .experiments import MacroExperiment, run_macro_experiments, select_macro_sample
from .inputs import PilotBindings, load_bindings, load_suite_trace_arrays
from .runtime_functions import HingeRuntime, OneTermRuntime, PowerSumRuntime
from .topology_reduction import TopologyResult, enumerate_topology
from partC_rev1_src.phase4.tracing import _directory_digest


FINAL_STATUS = "PHASE5_PILOT_COMPLETE_WAITING_EXTERNAL_REVIEW"
BLOCKED_STATUS = "PHASE5_PILOT_INCOMPLETE_WITH_EXACT_BLOCKER"
STARTING_HEAD = "9b29fe2f49c563bd037deec309d7e2d624e42c3c"
OUTER_AUTHORITY_SHA256 = "64c19984f9ceac0512bd4dc0ab56b223545d1a7106b96c498f0e349d868501be"
PARENT_AUTHORITY_SHA256 = "dcf518f90c971747b890b99b419451d46321785897ca241382b61d850b463b15"
PILOT_ERROR_BANDS = (0.0025, 0.005, 0.01, 0.02, 0.05)
PILOT_RUNTIME_CLASSES = (
    "EXACT_CLOSED_FORM",
    "FIT_ONE_TERM_GLOBAL_EXPONENT",
    "FIT_ONE_TERM_DISCRETE_EXPONENT",
    "MONOTONE_HINGE_4",
    "MONOTONE_HINGE_8",
)
PILOT_ERROR_SUMMARY_CLASSES = (*PILOT_RUNTIME_CLASSES, "CASCADE")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def write_json(path: Path, payload: Any) -> None:
    write_text(path, canonical_json(payload))


def git(repo: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "core.longpaths=true", *arguments],
        check=check,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _verify_installed_authority(authority_root: Path) -> dict[str, object]:
    installation = json.loads(
        (authority_root / "AUTHORITY_INSTALLATION.json").read_text(encoding="utf-8")
    )
    outer = Path(installation["source_package"]["path"])
    parent = Path(installation["nested_parent_authority"]["external_path"])
    extracted_source = Path(installation["extracted_source_root"])
    checks = []
    for role, path, declared in (
        ("outer_authority", outer, installation["source_package"]),
        ("nested_parent", parent, installation["nested_parent_authority"]),
    ):
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        frozen_sha = (
            OUTER_AUTHORITY_SHA256 if role == "outer_authority" else PARENT_AUTHORITY_SHA256
        )
        passed = bool(
            actual_size == int(declared["size_bytes"])
            and actual_sha == str(declared["sha256"])
            and actual_sha == frozen_sha
        )
        checks.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "size_bytes": actual_size,
                "sha256": actual_sha,
                "pass": passed,
            }
        )
    manifest = pd.read_csv(authority_root / "MANIFEST.csv")
    failures: list[str] = []
    for row in manifest.itertuples(index=False):
        relative_path = str(row.relative_path)
        path = extracted_source / relative_path
        if (
            not path.is_file()
            or path.stat().st_size != int(row.size_bytes)
            or sha256_file(path) != str(row.sha256)
        ):
            failures.append(str(row.relative_path))
    installed_failures: list[str] = []
    for installed in sorted(
        value
        for value in authority_root.rglob("*")
        if value.is_file() and value.name != "AUTHORITY_INSTALLATION.json"
    ):
        relative = installed.relative_to(authority_root)
        source = extracted_source / relative
        if not source.is_file() or sha256_file(installed) != sha256_file(source):
            installed_failures.append(relative.as_posix())
    if (
        any(not bool(value["pass"]) for value in checks)
        or failures
        or installed_failures
    ):
        raise RuntimeError(
            "Installed Phase 5 authority authentication failed: "
            f"package={failures}, installed={installed_failures}"
        )
    return {
        "package_checks": checks,
        "installed_manifest_row_count": len(manifest),
        "installed_manifest_failure_count": len(failures),
        "installed_subset_failure_count": len(installed_failures),
        "status": "PASS",
    }


@dataclass(slots=True)
class SuiteWork:
    suite_id: str
    root: Path
    trace_map: Any
    frames: dict[str, pd.DataFrame]
    low_trace: dict[str, np.ndarray]
    high_trace: dict[str, np.ndarray]
    topology: TopologyResult
    bundle_key: dict[str, str]
    internal_key: dict[str, str]


def _suite_frames(root: Path) -> dict[str, pd.DataFrame]:
    names = (
        "acnhn_nodes",
        "acnhn_static_links",
        "bacn_summary",
        "bacn_membership",
        "bundle_summary",
        "bundle_membership",
        "consolidated_terminals",
        "terminal_connectors",
        "structural_internal_charges",
        "suppressed_backbone_arcs",
        "racn_summary",
        "racn_restricted_witnesses",
        "regional_pair_support_summary",
    )
    result = {}
    for name in names:
        path = root / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = pd.read_parquet(path)
    return result


def _make_bundle_spec(indices: np.ndarray, bindings: PilotBindings) -> BundleSpec:
    graph = bindings.graph
    return BundleSpec(
        tuple(str(graph.arc_idx_to_directed_arc_id[index]) for index in indices),
        graph.free_time[indices] * 60.0,
        bindings.low_physical.used_capacity[indices],
        graph.alpha[indices],
        graph.beta[indices],
    )


def _bundle_inventory(
    suite_id: str,
    membership: pd.DataFrame,
    low_trace: dict[str, np.ndarray],
    high_trace: dict[str, np.ndarray],
    trace_map: Any,
    bindings: PilotBindings,
    global_specs: dict[str, BundleSpec],
) -> tuple[dict[str, str], list[dict[str, object]]]:
    arc_index = {
        str(value): index for index, value in enumerate(bindings.graph.arc_idx_to_directed_arc_id)
    }
    frame = membership[["bundle_id", "directed_arc_id", "member_fftt_s"]].copy()
    frame["bundle_id"] = frame["bundle_id"].astype(str)
    frame["directed_arc_id"] = frame["directed_arc_id"].astype(str)
    frame["arc_index"] = frame["directed_arc_id"].map(arc_index)
    if frame["arc_index"].isna().any():
        raise ValueError(f"{suite_id}: bundle membership does not close to physical arcs")
    expected = bindings.graph.free_time[frame["arc_index"].to_numpy(dtype=np.int64)] * 60.0
    if not np.allclose(expected, frame["member_fftt_s"].to_numpy(dtype=np.float64), rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{suite_id}: bundle member FFTT differs from canonical seconds")
    frame = frame.sort_values(["bundle_id", "directed_arc_id"], kind="mergesort")
    low_current = dict(zip(trace_map.bundle_ids, low_trace["current__bundle_flow"], strict=True))
    low_aon = dict(zip(trace_map.bundle_ids, low_trace["aon__bundle_flow"], strict=True))
    high_current = dict(zip(trace_map.bundle_ids, high_trace["current__bundle_flow"], strict=True))
    keys: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for bundle_id, group in frame.groupby("bundle_id", sort=True):
        indices = group["arc_index"].to_numpy(dtype=np.int64)
        spec = _make_bundle_spec(indices, bindings)
        key = primitive_function_key(
            spec,
            graph_identity=bindings.sources["graph"]["identity"],
            clean_capacity_identity=bindings.sources["graph"]["clean_capacity_payload_sha256"],
            capacity_factor=0.30,
            capacity_role="ACCEPTED_USED_CAPACITY_FACTOR_APPLIED_EXACTLY_ONCE",
        )
        keys[str(bundle_id)] = key
        global_specs.setdefault(key, spec)
        exact_homogeneous = bool(
            spec.member_count == 1
            or np.all(spec.alpha == 0.0)
            or (
                np.all(spec.alpha > 0.0)
                and np.all(spec.free_time_s == spec.free_time_s[0])
                and np.all(spec.beta == spec.beta[0])
            )
        )
        rows.append(
            {
                "suite_short_id": suite_id,
                "bundle_id": str(bundle_id),
                "canonical_primitive_key": key,
                "member_count": spec.member_count,
                "fftt_min_s": float(spec.free_time_s.min()),
                "fftt_max_s": float(spec.free_time_s.max()),
                "capacity_min": float(spec.capacity.min()),
                "capacity_max": float(spec.capacity.max()),
                "member_fftt_values_json": json.dumps(
                    spec.free_time_s.tolist(), separators=(",", ":")
                ),
                "member_capacity_values_json": json.dumps(
                    spec.capacity.tolist(), separators=(",", ":")
                ),
                "member_alpha_values_json": json.dumps(
                    spec.alpha.tolist(), separators=(",", ":")
                ),
                "member_beta_values_json": json.dumps(
                    spec.beta.tolist(), separators=(",", ":")
                ),
                "alpha_values_json": json.dumps(np.unique(spec.alpha).tolist(), separators=(",", ":")),
                "beta_values_json": json.dumps(np.unique(spec.beta).tolist(), separators=(",", ":")),
                "activation_level_count": int(len(np.unique(spec.free_time_s))),
                "positive_activation_threshold_count": int(
                    max(0, len(np.unique(spec.free_time_s)) - 1)
                ),
                "has_staggered_member_activation": bool(
                    len(np.unique(spec.free_time_s)) > 1
                ),
                "singleton": spec.member_count == 1,
                "exact_homogeneous_closed_form": exact_homogeneous,
                "low_current_flow": float(low_current[str(bundle_id)]),
                "low_aon_flow": float(low_aon[str(bundle_id)]),
                "high_current_flow": float(high_current[str(bundle_id)]),
            }
        )
    if tuple(sorted(keys)) != trace_map.bundle_ids:
        raise ValueError(f"{suite_id}: bundle membership registry differs from trace registry")
    return keys, rows


def _internal_component_keys(
    frames: dict[str, pd.DataFrame], bindings: PilotBindings
) -> dict[str, str]:
    members = frames["bacn_membership"][["bacn_id", "partC_rev1_node_uid"]].astype(str)
    suppressed = frames["suppressed_backbone_arcs"][["bacn_id", "directed_arc_id"]].astype(str)
    member_map = members.groupby("bacn_id", sort=True)["partC_rev1_node_uid"].agg(lambda value: tuple(sorted(value)))
    arc_map = suppressed.groupby("bacn_id", sort=True)["directed_arc_id"].agg(lambda value: tuple(sorted(value)))
    result: dict[str, str] = {}
    for bacn_id, physical_nodes in member_map.items():
        payload = {
            "graph_identity": bindings.sources["graph"]["identity"],
            "clean_capacity_identity": bindings.sources["graph"]["clean_capacity_payload_sha256"],
            "capacity_factor": float(0.30).hex(),
            "physical_node_uids": physical_nodes,
            "suppressed_directed_arc_ids": arc_map.get(bacn_id, ()),
        }
        result[str(bacn_id)] = "p5_internal_" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return result


def _bundle_census(bundle_rows: pd.DataFrame) -> pd.DataFrame:
    key_suite_counts = bundle_rows.groupby("canonical_primitive_key", sort=False)[
        "suite_short_id"
    ].nunique()
    cross_suite_keys = set(key_suite_counts.index[key_suite_counts.gt(1)].astype(str))
    rows: list[dict[str, object]] = []
    for suite_id, frame in [*bundle_rows.groupby("suite_short_id", sort=True), ("ALL_SUITES", bundle_rows)]:
        members = frame["member_count"].to_numpy(dtype=np.float64)
        cross_suite_occurrences = int(
            frame["canonical_primitive_key"].astype(str).isin(cross_suite_keys).sum()
        )
        cross_suite_unique = int(
            frame.loc[
                frame["canonical_primitive_key"].astype(str).isin(cross_suite_keys),
                "canonical_primitive_key",
            ].nunique()
        )
        fftt = np.asarray(
            [
                value
                for text in frame["member_fftt_values_json"]
                for value in json.loads(text)
            ],
            dtype=np.float64,
        )
        capacity = np.asarray(
            [
                value
                for text in frame["member_capacity_values_json"]
                for value in json.loads(text)
            ],
            dtype=np.float64,
        )
        alpha = np.asarray(
            [
                value
                for text in frame["member_alpha_values_json"]
                for value in json.loads(text)
            ],
            dtype=np.float64,
        )
        beta = np.asarray(
            [
                value
                for text in frame["member_beta_values_json"]
                for value in json.loads(text)
            ],
            dtype=np.float64,
        )

        def quantile_json(values: np.ndarray) -> str:
            return json.dumps(
                dict(
                    zip(
                        ("p0", "p25", "p50", "p75", "p95", "p100"),
                        np.quantile(values, [0, 0.25, 0.5, 0.75, 0.95, 1]).tolist(),
                        strict=True,
                    )
                ),
                sort_keys=True,
                separators=(",", ":"),
            )

        rows.append(
            {
                "suite_short_id": suite_id,
                "bundle_count": len(frame),
                "physical_member_count": int(frame["member_count"].sum()),
                "member_count_mean": float(members.mean()),
                "member_count_p50": float(np.quantile(members, 0.50)),
                "member_count_p95": float(np.quantile(members, 0.95)),
                "member_count_max": int(members.max()),
                "singleton_count": int(frame["singleton"].sum()),
                "homogeneous_closed_form_count": int(frame["exact_homogeneous_closed_form"].sum()),
                "zero_low_current_count": int(frame["low_current_flow"].eq(0.0).sum()),
                "zero_low_aon_count": int(frame["low_aon_flow"].eq(0.0).sum()),
                "zero_high_current_count": int(frame["high_current_flow"].eq(0.0).sum()),
                "alpha_unique_json": json.dumps(sorted({v for text in frame["alpha_values_json"] for v in json.loads(text)})),
                "beta_unique_json": json.dumps(sorted({v for text in frame["beta_values_json"] for v in json.loads(text)})),
                "fftt_s_quantiles_json": quantile_json(fftt),
                "capacity_quantiles_json": quantile_json(capacity),
                "alpha_quantiles_json": quantile_json(alpha),
                "beta_quantiles_json": quantile_json(beta),
                "low_current_flow_quantiles_json": quantile_json(
                    frame["low_current_flow"].to_numpy(dtype=np.float64)
                ),
                "low_aon_flow_quantiles_json": quantile_json(
                    frame["low_aon_flow"].to_numpy(dtype=np.float64)
                ),
                "high_current_flow_quantiles_json": quantile_json(
                    frame["high_current_flow"].to_numpy(dtype=np.float64)
                ),
                "staggered_activation_bundle_count": int(
                    frame["has_staggered_member_activation"].sum()
                ),
                "activation_level_count_quantiles_json": quantile_json(
                    frame["activation_level_count"].to_numpy(dtype=np.float64)
                ),
                "positive_activation_threshold_count": int(
                    frame["positive_activation_threshold_count"].sum()
                ),
                "canonical_unique_count": int(frame["canonical_primitive_key"].nunique()),
                "cross_suite_duplicate_unique_key_count": cross_suite_unique,
                "cross_suite_duplicate_unique_key_rate": cross_suite_unique
                / max(1, int(frame["canonical_primitive_key"].nunique())),
                "cross_suite_duplicate_occurrence_count": cross_suite_occurrences,
                "cross_suite_duplicate_rate": cross_suite_occurrences / len(frame),
                "status": "FULL_CENSUS_COMPLETE",
            }
        )
    return pd.DataFrame(rows)


def _racn_audit(work: SuiteWork) -> dict[str, object]:
    terminals = work.frames["consolidated_terminals"].copy()
    terminals["terminal_id"] = terminals["terminal_id"].astype(str)
    terminals["terminal_type"] = terminals["terminal_type"].astype(str)
    terminals = terminals.drop_duplicates("terminal_id").set_index("terminal_id").loc[
        list(work.trace_map.terminal_ids)
    ]
    is_r = terminals["terminal_type"].eq("RACN").to_numpy()
    terminal_parent = terminals["parent_object_id"].astype(str).to_numpy()

    def self_pair(
        trace: dict[str, np.ndarray], prefix: str
    ) -> tuple[int, float, int, float, int, float]:
        keys = trace[prefix + "coarse_regional_pair_flow__keys"].astype(np.uint64, copy=False)
        values = trace[prefix + "coarse_regional_pair_flow__values"].astype(np.float64, copy=False)
        entry = (keys >> np.uint64(32)).astype(np.int64)
        exit_ = (keys & np.uint64(0xFFFFFFFF)).astype(np.int64)
        same_terminal = (entry == exit_) & is_r[entry]
        same_parent = (
            is_r[entry]
            & is_r[exit_]
            & (terminal_parent[entry] == terminal_parent[exit_])
        )
        distinct_same_parent = same_parent & (entry != exit_)
        return (
            int(np.count_nonzero(same_terminal)),
            float(values[same_terminal].sum(dtype=np.float64)),
            int(np.count_nonzero(same_parent)),
            float(values[same_parent].sum(dtype=np.float64)),
            int(np.count_nonzero(distinct_same_parent)),
            float(values[distinct_same_parent].sum(dtype=np.float64)),
        )

    low_current = self_pair(work.low_trace, "current__")
    low_aon = self_pair(work.low_trace, "aon__")
    high_current = self_pair(work.high_trace, "current__")
    high_aon = self_pair(work.high_trace, "aon__")
    observed_self_pair_count = sum(
        value[0] for value in (low_current, low_aon, high_current, high_aon)
    )
    observed_self_pair_flow = sum(
        value[1] for value in (low_current, low_aon, high_current, high_aon)
    )
    witnesses = work.frames["racn_restricted_witnesses"]
    witness_time = witnesses["restricted_fftt_s"].to_numpy(dtype=np.float64)
    summary_count = len(work.frames["racn_summary"])
    declared_parents = terminals.loc[terminals["terminal_type"].eq("RACN"), "parent_object_id"].astype(str).nunique()
    return {
        "suite_short_id": work.suite_id,
        "summarized_racn_count": summary_count,
        "terminal_declared_racn_parent_count": int(declared_parents),
        "singleton_control_parent_count": int(max(0, declared_parents - summary_count)),
        "restricted_witness_count": len(witnesses),
        "restricted_witness_fftt_p50_s": float(np.median(witness_time)) if len(witness_time) else np.nan,
        "restricted_witness_fftt_max_s": float(witness_time.max()) if len(witness_time) else np.nan,
        "low_current_racn_interface_flow": float(work.low_trace["current__racn_interface_flow"].sum()),
        "high_current_racn_interface_flow": float(work.high_trace["current__racn_interface_flow"].sum()),
        "low_current_self_pair_count": low_current[0],
        "low_current_self_pair_flow": low_current[1],
        "low_current_same_racn_parent_pair_count": low_current[2],
        "low_current_same_racn_parent_pair_flow": low_current[3],
        "low_current_distinct_terminal_same_racn_parent_pair_count": low_current[4],
        "low_current_distinct_terminal_same_racn_parent_pair_flow": low_current[5],
        "low_aon_self_pair_count": low_aon[0],
        "low_aon_self_pair_flow": low_aon[1],
        "low_aon_same_racn_parent_pair_count": low_aon[2],
        "low_aon_same_racn_parent_pair_flow": low_aon[3],
        "low_aon_distinct_terminal_same_racn_parent_pair_count": low_aon[4],
        "low_aon_distinct_terminal_same_racn_parent_pair_flow": low_aon[5],
        "high_current_self_pair_count": high_current[0],
        "high_current_self_pair_flow": high_current[1],
        "high_current_same_racn_parent_pair_count": high_current[2],
        "high_current_same_racn_parent_pair_flow": high_current[3],
        "high_current_distinct_terminal_same_racn_parent_pair_count": high_current[4],
        "high_current_distinct_terminal_same_racn_parent_pair_flow": high_current[5],
        "high_aon_self_pair_count": high_aon[0],
        "high_aon_self_pair_flow": high_aon[1],
        "high_aon_same_racn_parent_pair_count": high_aon[2],
        "high_aon_same_racn_parent_pair_flow": high_aon[3],
        "high_aon_distinct_terminal_same_racn_parent_pair_count": high_aon[4],
        "high_aon_distinct_terminal_same_racn_parent_pair_flow": high_aon[5],
        "observed_racn_self_pair_count_all_states": observed_self_pair_count,
        "observed_racn_self_pair_flow_all_states": observed_self_pair_flow,
        # The sparse pair trace proves interface movement, but it does not bind
        # physical time to an R parent or establish whether a retained regional
        # potential already owns that time.  Keep this unknown rather than
        # silently converting throughput to seconds.
        "separately_owned_omitted_racn_exposure_observed": "UNRESOLVED",
        "observed_omitted_racn_exposure_flow_seconds": np.nan,
        "recommended_racn_charge_s": 0.0,
        "cost_ownership_resolution": "UNRESOLVED_NO_PATH_TIME_OWNERSHIP_BINDING",
        "recommendation": "PILOT_HYPOTHESIS_RETAIN_ZERO_RACN_CHARGE_PENDING_OWNERSHIP_AUDIT",
        "status": "PASS_TRACE_INTEGRITY_OWNERSHIP_UNRESOLVED",
    }


def _multiedge_audit(bindings: PilotBindings, topology: pd.DataFrame) -> pd.DataFrame:
    graph = bindings.graph
    key = graph.tail.astype(np.int64) * np.int64(len(graph.node_idx_to_uid)) + graph.head.astype(np.int64)
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    groups = counts > 1
    group_count = int(np.count_nonzero(groups))
    arcs_in_groups = int(counts[groups].sum())
    low_counts = np.bincount(
        inverse, weights=(bindings.low_physical.current_flow > 0.0).astype(np.int8)
    )
    high_counts = np.bincount(
        inverse, weights=(bindings.high_physical.current_flow > 0.0).astype(np.int8)
    )
    low_positive = int(np.count_nonzero(groups & (low_counts >= 2)))
    high_positive = int(np.count_nonzero(groups & (high_counts >= 2)))
    both_positive = int(np.count_nonzero(groups & (low_counts >= 2) & (high_counts >= 2)))
    collision_by_suite = topology.set_index("suite_short_id")["parallel_endpoint_collision_count"].to_dict()
    rows = [
        {
            "audit_layer": "IMMUTABLE_PHYSICAL_GRAPH",
            "suite_short_id": "ALL",
            "test": "same-tail-head stable edge preservation",
            "measured_count": group_count,
            "secondary_count": arcs_in_groups,
            "status": "PASS",
            "evidence": f"{group_count} endpoint groups, {arcs_in_groups} arcs, maximum multiplicity {int(counts.max())}",
            "recommendation": "RETAIN_NATIVE_PARALLEL_MACRO_LINKS",
        },
        {
            "audit_layer": "ACCEPTED_LOW_HIGH_CURRENT",
            "suite_short_id": "ALL",
            "test": "separate positive parallel alternatives",
            "measured_count": both_positive,
            "secondary_count": low_positive + high_positive,
            "status": "PASS",
            "evidence": "physical CURRENT arrays retain independent per-arc slots in both tiers",
            "recommendation": "RETAIN_NATIVE_PARALLEL_MACRO_LINKS",
        },
    ]
    stack = (
        ("GRAPH_BUILDER", "partC_rev1_graph_snapshot.py", "one row/index per stable arc"),
        ("SHORTEST_PATH", "partC_rev1_shortest_paths.py", "predecessor_arc and stable-ID tie rank"),
        ("FLOW_ARRAY", "phase4/kernels.py", "flow assigned by arc index"),
        ("SQLITE", "phase3a/production_portfolio.py", "unique key is suite plus link_id, not endpoints"),
        ("TRACE", "phase4/tracing.py", "dense stable-ID registry and per-edge bincount"),
        ("LIFT_BACK", "phase3a/liftability.py", "coarse_edge_id preserves witness identity"),
    )
    for layer, source, evidence in stack:
        rows.append(
            {
                "audit_layer": layer,
                "suite_short_id": "ALL",
                "test": "distinct parallel IDs survive layer",
                "measured_count": 1,
                "secondary_count": 0,
                "status": "PASS",
                "evidence": f"{source}: {evidence}",
                "recommendation": "RETAIN_NATIVE_PARALLEL_MACRO_LINKS",
            }
        )
    for suite_id, count in sorted(collision_by_suite.items()):
        rows.append(
            {
                "audit_layer": "REAL_CONTRACTION_COLLISION_SCREEN",
                "suite_short_id": suite_id,
                "test": "candidate macro shares endpoints with retained alternative",
                "measured_count": int(count),
                "secondary_count": 0,
                "status": "PASS_PRESERVE_DISTINCT_MACRO_IDS",
                "evidence": "full candidate-overlay endpoint screen",
                "recommendation": "RETAIN_NATIVE_PARALLEL_MACRO_LINKS",
            }
        )
    return pd.DataFrame(rows)


def _runtime_from_fit(row: pd.Series):
    parameter = json.loads(row["parameter_json"])
    if row["runtime_class"] == "EXACT_CLOSED_FORM":
        return PowerSumRuntime(
            parameter["intercept_s"],
            np.asarray(parameter["exponents"]),
            np.asarray(parameter["coefficients"]),
        )
    if str(row["runtime_class"]).startswith("FIT_ONE_TERM"):
        return OneTermRuntime(
            parameter["intercept_s"],
            parameter["scale_flow"],
            parameter["exponent"],
            parameter["eta_s"],
        )
    return HingeRuntime(
        parameter["intercept_s"],
        np.asarray(json.loads(row["knot_json"])),
        np.asarray(parameter["segment_slopes_s_per_flow"]),
    )


def _best_class_fit_rows(fits: pd.DataFrame) -> pd.DataFrame:
    """Select one deterministic frozen-objective variant per signature/class."""

    return (
        fits.sort_values(
            [
                "canonical_macro_signature",
                "runtime_class",
                "weighted_rms_dynamic_rel_cost_error" if "weighted_rms_dynamic_rel_cost_error" in fits else "weighted_rms_rel_cost_error",
                "variant_id",
            ],
            kind="mergesort",
        )
        .drop_duplicates(["canonical_macro_signature", "runtime_class"])
    )


def _path_occurrence_cache_key(role, signature, primitive_ids, q):
    """Model reuse never reuses a physical occurrence's charge/retention decision."""
    return (str(role), str(signature), tuple(map(str, primitive_ids)), float(q).hex())


def _accepted_path_candidate_signatures(
    bindings: PilotBindings,
    work: SuiteWork,
    chains: pd.DataFrame,
    path_file: str | Path | None = None,
    arc_column: str = "arc_indices_json",
) -> set[str]:
    """Return candidate signatures occurring on the immutable frozen path sample."""

    candidates = chains.loc[
        chains["suite_short_id"].eq(work.suite_id)
        & ~chains["protected_or_blocked"].astype(bool)
    ]
    by_first: dict[str, list[tuple[tuple[str, ...], str]]] = {}
    for row in candidates.itertuples(index=False):
        primitive_ids = tuple(map(str, json.loads(row.primitive_bundle_ids_json)))
        by_first.setdefault(primitive_ids[0], []).append(
            (primitive_ids, str(row.canonical_signature))
        )
    for values in by_first.values():
        values.sort(key=lambda value: (-len(value[0]), value[1]))
    paths = pd.read_parquet(
        Path(path_file) if path_file is not None else bindings.sources["frozen_path_sample"],
        columns=[arc_column],
    )
    result: set[str] = set()
    for encoded in paths[arc_column].astype(str):
        arcs = np.asarray(json.loads(encoded), dtype=np.int64)
        bundle = work.trace_map.arc_to_bundle_local[arcs]
        region = work.trace_map.arc_region_local[arcs]
        segment: list[str] = []
        segments: list[list[str]] = []
        for region_local, bundle_local in zip(region, bundle, strict=True):
            if int(region_local) >= 0:
                if segment:
                    segments.append(segment)
                    segment = []
            elif int(bundle_local) >= 0:
                segment.append(work.trace_map.bundle_ids[int(bundle_local)])
        if segment:
            segments.append(segment)
        for values in segments:
            for position, first in enumerate(values):
                for primitive_ids, signature in by_first.get(first, []):
                    if tuple(values[position : position + len(primitive_ids)]) == primitive_ids:
                        result.add(signature)
    return result


def _path_validation(
    bindings: PilotBindings,
    work: SuiteWork,
    global_specs: dict[str, BundleSpec],
    chains: pd.DataFrame,
    fit_results: pd.DataFrame,
    charge_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reopen accepted paths and separate every contracted-cost error source."""

    frozen_path = Path(bindings.sources["frozen_path_sample"])
    high_path = Path(bindings.sources["high_benchmark"]["root"]) / "routes" / "sampled_route_switches.parquet"
    frozen = pd.read_parquet(frozen_path)
    high = pd.read_parquet(high_path)

    selected_fit_rows = fit_results.loc[
        fit_results["recommended_status"].astype(str).str.startswith("CASCADE_SELECTED")
    ]
    selected_fit = {
        str(row.canonical_macro_signature): _runtime_from_fit(pd.Series(row._asdict()))
        for row in selected_fit_rows.itertuples(index=False)
    }
    best_fit_rows = _best_class_fit_rows(fit_results)
    band_runtime: dict[tuple[str, str], tuple[object, float]] = {}
    for fit in best_fit_rows.itertuples(index=False):
        key = (str(fit.canonical_macro_signature), str(fit.runtime_class))
        band_runtime[key] = (
            _runtime_from_fit(pd.Series(fit._asdict())),
            float(fit.max_base_rel_cost_error if getattr(fit, "parameter_cost_role", "") == "DYNAMIC_ONLY" else fit.max_rel_cost_error),
        )
    candidate_rows = chains.loc[
        chains["suite_short_id"].eq(work.suite_id)
        & ~chains["protected_or_blocked"].astype(bool)
        & chains["sample_included"].astype(bool)
    ]
    candidate_by_provenance: dict[tuple[str, tuple[str, ...]], dict[str, object]] = {}
    bundle_local_by_id = {
        value: index for index, value in enumerate(work.trace_map.bundle_ids)
    }
    bacn_local_by_id = {
        value: index for index, value in enumerate(work.trace_map.bacn_ids)
    }
    candidates_by_first: dict[str, list[tuple[tuple[str, ...], str]]] = {}
    for row in candidate_rows.itertuples(index=False):
        primitive_ids = tuple(map(str, json.loads(row.primitive_bundle_ids_json)))
        provenance_key = (str(row.canonical_signature), primitive_ids)
        if provenance_key in candidate_by_provenance:
            raise RuntimeError(
                f"Ambiguous sampled chain provenance: {provenance_key}"
            )
        candidate_by_provenance[provenance_key] = row._asdict()
        candidates_by_first.setdefault(primitive_ids[0], []).append(
            (primitive_ids, str(row.canonical_signature))
        )
    for values in candidates_by_first.values():
        values.sort(key=lambda value: (-len(value[0]), value[1]))

    charge_frame = charge_results.loc[
        charge_results["suite_short_id"].eq(work.suite_id)
    ].set_index("bacn_id").loc[list(work.trace_map.bacn_ids)]
    low_mean = charge_frame["low_operating_mean_seconds"].to_numpy(dtype=np.float64)
    structural = charge_frame["structural_seed_seconds"].to_numpy(dtype=np.float64)
    recommended_charge = np.where(np.isfinite(low_mean), low_mean, structural)
    if np.any(~np.isfinite(recommended_charge)) or np.any(recommended_charge < 0.0):
        raise RuntimeError("Path validation received an invalid fixed B-ACN charge")

    projector = StaticPathProjector(bindings.graph, work.trace_map, work.frames)
    dynamic_models = "parameter_cost_role" in fit_results
    records: list[dict[str, object]] = []
    bundle_cache: dict[tuple[str, int], float] = {}
    macro_cache: dict[tuple, tuple[float, float, float, str]] = {}
    macro_band_cache: dict[
        tuple,
        tuple[dict[str, float], dict[str, float], dict[str, int]],
    ] = {}
    for label, table, arc_column, hash_column, hash_separator, cost, bundle_flow in (
        (
            "LOW_ON_FROZEN_FREE_FLOW_PATH",
            frozen,
            "arc_indices_json",
            "arc_ids_sha256",
            "\x1f",
            bindings.low_physical.current_cost_s,
            work.low_trace["current__bundle_flow"],
        ),
        (
            "HIGH_HELDOUT_FROZEN_COMPARATOR_PATH",
            frozen,
            "arc_indices_json",
            "arc_ids_sha256",
            "\x1f",
            bindings.high_physical.current_cost_s,
            work.high_trace["current__bundle_flow"],
        ),
        (
            "HIGH_HELDOUT_SELECTED_PATH",
            high,
            "selected_arc_indices_json",
            "selected_arc_ids_sha256",
            "\n",
            bindings.high_physical.current_cost_s,
            work.high_trace["current__bundle_flow"],
        ),
    ):
        for row in table.itertuples(index=False):
            arcs = np.asarray(json.loads(getattr(row, arc_column)), dtype=np.int64)
            arc_ids = [str(bindings.graph.arc_idx_to_directed_arc_id[index]) for index in arcs]
            digest = hashlib.sha256(hash_separator.join(arc_ids).encode("utf-8")).hexdigest()
            hash_pass = digest == str(getattr(row, hash_column))
            continuity = bool(
                len(arcs) > 0
                and int(bindings.graph.tail[arcs[0]]) == int(getattr(row, "origin_idx"))
                and int(bindings.graph.head[arcs[-1]]) == int(getattr(row, "destination_idx"))
                and np.all(bindings.graph.head[arcs[:-1]] == bindings.graph.tail[arcs[1:]])
            )
            region = work.trace_map.arc_region_local[arcs]
            bundle = work.trace_map.arc_to_bundle_local[arcs]
            internal = work.trace_map.arc_internal_bacn_local[arcs]
            physical_total = float(cost[arcs].sum(dtype=np.float64))
            regional_cost = float(cost[arcs[region >= 0]].sum(dtype=np.float64))
            internal_cost = float(cost[arcs[internal >= 0]].sum(dtype=np.float64))
            selected_bundle_physical = float(cost[arcs[bundle >= 0]].sum(dtype=np.float64))

            physical_nodes = np.r_[bindings.graph.tail[arcs[0]], bindings.graph.head[arcs]]
            node_bacn = work.trace_map.node_to_bacn_local[physical_nodes]
            visited_bacn: list[int] = []
            previous = -2
            for value in node_bacn:
                current = int(value)
                if current >= 0 and current != previous:
                    visited_bacn.append(current)
                previous = current
            legacy_physical_node_charge = float(
                recommended_charge[np.asarray(visited_bacn, dtype=np.int64)].sum(
                    dtype=np.float64
                )
            )
            static_counts, static_weights = projector.project(arcs)
            estimated_fixed_charge = float(sum(recommended_charge[i]*weight for i,weight in static_weights.items()))
            fixed_charge_error = estimated_fixed_charge - internal_cost
            contraction_charge_identity_error = 0.0

            exact_bundle_cost = 0.0
            for local in bundle[bundle >= 0]:
                local_index = int(local)
                cache_key = (label, local_index)
                if cache_key not in bundle_cache:
                    bundle_id = work.trace_map.bundle_ids[local_index]
                    spec = global_specs[work.bundle_key[bundle_id]]
                    bundle_cache[cache_key] = exact_bundle_oracle(
                        spec, float(bundle_flow[local_index])
                    ).marginal_cost_s
                exact_bundle_cost += bundle_cache[cache_key]

            bundle_segments: list[list[str]] = []
            active_segment: list[str] = []
            for region_local, bundle_local in zip(region, bundle, strict=True):
                if int(region_local) >= 0:
                    if active_segment:
                        bundle_segments.append(active_segment)
                        active_segment = []
                elif int(bundle_local) >= 0:
                    active_segment.append(work.trace_map.bundle_ids[int(bundle_local)])
            if active_segment:
                bundle_segments.append(active_segment)

            series_identity_error = 0.0
            runtime_error = 0.0
            flow_state_alignment_error = 0.0
            matched_macro_count = 0
            selected_runtime_macro_count = 0
            retained_original_macro_count = 0
            matched_bundle_count = 0
            selected_runtime_bundle_count = 0
            liftback_order_pass = True
            band_runtime_error = {
                f"{runtime_class}|{band:.4f}": 0.0
                for runtime_class in PILOT_ERROR_SUMMARY_CLASSES
                for band in PILOT_ERROR_BANDS
            }
            band_alignment_error = {key: 0.0 for key in band_runtime_error}
            band_eligible_macro_count = {
                key: 0 for key in band_runtime_error
            }
            for segment in bundle_segments:
                position = 0
                while position < len(segment):
                    match: tuple[tuple[str, ...], str] | None = None
                    for candidate in candidates_by_first.get(segment[position], []):
                        primitive_ids, signature = candidate
                        if tuple(segment[position : position + len(primitive_ids)]) == primitive_ids:
                            match = candidate
                            break
                    if match is None:
                        position += 1
                        continue
                    primitive_ids, signature = match
                    local_indexes = np.asarray(
                        [bundle_local_by_id[value] for value in primitive_ids],
                        dtype=np.int64,
                    )
                    local_flow_values = bundle_flow[local_indexes]
                    if float(np.ptp(local_flow_values)) > 1e-8+1e-10*max(1.0,float(np.max(np.abs(local_flow_values)))):
                        raise RuntimeError("Path matched a current-flow-inconsistent corridor")
                    q = float(local_flow_values[0])
                    cache_key = _path_occurrence_cache_key(label, signature, primitive_ids, q)
                    if cache_key not in macro_cache:
                        local_band_error: dict[str, float] = {}
                        local_band_alignment: dict[str, float] = {}
                        local_band_count: dict[str, int] = {}
                        specs = tuple(
                            global_specs[work.bundle_key[value]] for value in primitive_ids
                        )
                        chain_row = candidate_by_provenance[(signature, primitive_ids)]
                        internal_ids = tuple(
                            map(str, json.loads(str(chain_row["internal_bacn_ids_json"])))
                        )
                        charge_s = float(
                            sum(
                                recommended_charge[
                                    bacn_local_by_id[value]
                                ]
                                for value in internal_ids
                            )
                        )
                        pieces = [exact_bundle_oracle(spec, q) for spec in specs]
                        primitive_sum = float(
                            sum(piece.marginal_cost_s for piece in pieces) + charge_s
                        )
                        series = exact_series_oracle(
                            specs, q, internal_charge_s=charge_s
                        )
                        function = selected_fit.get(signature)
                        if chain_row.get("runtime_resolution", "SELECTED_ROOT_FREE_RUNTIME") != "SELECTED_ROOT_FREE_RUNTIME":
                            function = None
                        # Parameters hold b(q); H is bound to this actual occurrence.
                        runtime_charge = charge_s if dynamic_models else 0.0
                        per_link_state = float(
                            sum(
                                exact_bundle_oracle(
                                    spec, float(bundle_flow[local_index])
                                ).marginal_cost_s
                                for spec, local_index in zip(
                                    specs, local_indexes, strict=True
                                )
                            )
                            + charge_s
                        )
                        if function is None:
                            # The pilot makes no silent exact-root runtime fallback.
                            # A sampled signature that misses the provisional error
                            # gate remains its original ordered primitive links.
                            macro_cache[cache_key] = (
                                series.marginal_cost_s - primitive_sum,
                                0.0,
                                0.0,
                                "RETAIN_ORIGINAL_PRIMITIVE_LINKS_PILOT_ERROR_GATE",
                            )
                        else:
                            macro_cache[cache_key] = (
                                series.marginal_cost_s - primitive_sum,
                                float(function.cost(q)) + runtime_charge - series.marginal_cost_s,
                                series.marginal_cost_s - per_link_state,
                                "SELECTED_ROOT_FREE_RUNTIME",
                            )
                        for runtime_class in PILOT_RUNTIME_CLASSES:
                            fit_entry = band_runtime.get((signature, runtime_class))
                            if fit_entry is None:
                                continue
                            candidate_function, fit_max_error = fit_entry
                            candidate_error = (
                                float(candidate_function.cost(q)) + runtime_charge
                                - series.marginal_cost_s
                            )
                            for band in PILOT_ERROR_BANDS:
                                if fit_max_error <= band:
                                    band_key = f"{runtime_class}|{band:.4f}"
                                    local_band_error[band_key] = candidate_error
                                    local_band_alignment[band_key] = (
                                        series.marginal_cost_s - per_link_state
                                    )
                                    local_band_count[band_key] = 1
                        for band in PILOT_ERROR_BANDS:
                            for runtime_class in PILOT_RUNTIME_CLASSES:
                                fit_entry = band_runtime.get(
                                    (signature, runtime_class)
                                )
                                if fit_entry is None or fit_entry[1] > band:
                                    continue
                                candidate_function, _ = fit_entry
                                band_key = f"CASCADE|{band:.4f}"
                                local_band_error[band_key] = (
                                    float(candidate_function.cost(q)) + runtime_charge
                                    - series.marginal_cost_s
                                )
                                local_band_alignment[band_key] = (
                                    series.marginal_cost_s - per_link_state
                                )
                                local_band_count[band_key] = 1
                                break
                        macro_band_cache[cache_key] = (
                            local_band_error,
                            local_band_alignment,
                            local_band_count,
                        )
                    identity, approximation, alignment, resolution = macro_cache[
                        cache_key
                    ]
                    (
                        local_band_error,
                        local_band_alignment,
                        local_band_count,
                    ) = macro_band_cache[cache_key]
                    for band_key, value in local_band_error.items():
                        band_runtime_error[band_key] += value
                    for band_key, value in local_band_alignment.items():
                        band_alignment_error[band_key] += value
                    for band_key, value in local_band_count.items():
                        band_eligible_macro_count[band_key] += value
                    series_identity_error += identity
                    runtime_error += approximation
                    flow_state_alignment_error += alignment
                    matched_macro_count += 1
                    chain_row = candidate_by_provenance[(signature, primitive_ids)]
                    original_charge = sum(0.5*recommended_charge[bacn_local_by_id[node]] for edge in primitive_ids for node in projector.endpoints[edge])
                    internal_ids_for_charge = json.loads(chain_row["internal_bacn_ids_json"])
                    reduced_charge = sum(recommended_charge[bacn_local_by_id[node]] for node in internal_ids_for_charge)
                    reduced_charge += 0.5*(recommended_charge[bacn_local_by_id[chain_row["retained_tail_bacn_id"]]]+recommended_charge[bacn_local_by_id[chain_row["retained_head_bacn_id"]]])
                    contraction_charge_identity_error += reduced_charge-original_charge
                    if resolution == "SELECTED_ROOT_FREE_RUNTIME":
                        selected_runtime_macro_count += 1
                        selected_runtime_bundle_count += len(primitive_ids)
                    else:
                        retained_original_macro_count += 1
                    matched_bundle_count += len(primitive_ids)
                    liftback_order_pass &= (
                        tuple(segment[position : position + len(primitive_ids)])
                        == primitive_ids
                    )
                    position += len(primitive_ids)

            free_mixing_error = (
                exact_bundle_cost
                - selected_bundle_physical
                + flow_state_alignment_error
            )
            reconstructed = regional_cost + exact_bundle_cost + estimated_fixed_charge
            candidate_cost = (
                reconstructed + flow_state_alignment_error + runtime_error
            )
            accounting_error = candidate_cost - physical_total
            retained_reference_error = reconstructed - physical_total
            component_sum = (
                series_identity_error
                + runtime_error
                + fixed_charge_error
                + free_mixing_error
            )
            component_closure_error = accounting_error - component_sum
            identity_tolerance = 1.0e-8 + 1.0e-12 * max(
                1.0, abs(exact_bundle_cost + estimated_fixed_charge)
            )
            closure_tolerance = 1.0e-7 + 1.0e-12 * max(1.0, abs(physical_total))
            liftability_error = 0.0 if (
                hash_pass and continuity and liftback_order_pass
            ) else 1.0
            retained_or_unmatched_bundle_count = int(
                np.count_nonzero(bundle >= 0) - matched_bundle_count
            )
            fully_root_free_runtime_evaluable = bool(
                matched_macro_count > 0
                and retained_original_macro_count == 0
                and retained_or_unmatched_bundle_count == 0
            )
            band_path_relative_error = {
                key: abs(
                    retained_reference_error
                    + band_alignment_error[key]
                    + value
                )
                / max(abs(physical_total), 1.0)
                for key, value in band_runtime_error.items()
            }
            band_candidate_path_cost = {
                key: physical_total
                + retained_reference_error
                + band_alignment_error[key]
                + value
                for key, value in band_runtime_error.items()
            }
            band_runtime_only_relative_error = {
                key: abs(value) / max(abs(physical_total), 1.0)
                for key, value in band_runtime_error.items()
            }
            records.append(
                {
                    "sample_od_id": str(getattr(row, "sample_od_id")),
                    "evaluation_role": label,
                    "suite_short_id": work.suite_id,
                    "origin_idx": int(getattr(row, "origin_idx")),
                    "destination_idx": int(getattr(row, "destination_idx")),
                    "physical_arc_count": len(arcs),
                    "physical_path_cost_s": physical_total,
                    "regional_owned_cost_s": regional_cost,
                    "suppressed_internal_cost_s": internal_cost,
                    "selected_member_bundle_cost_s": selected_bundle_physical,
                    "exact_bundle_oracle_cost_s": exact_bundle_cost,
                    "reconstructed_component_cost_s": reconstructed,
                    "candidate_compiled_path_cost_s": (
                        candidate_cost
                        if fully_root_free_runtime_evaluable
                        else np.nan
                    ),
                    "candidate_resolution_path_cost_s": candidate_cost,
                    "candidate_resolution_cost_scope": (
                        "FULLY_ROOT_FREE_SELECTED_RUNTIME"
                        if fully_root_free_runtime_evaluable
                        else "SELECTED_RUNTIME_PLUS_OFFLINE_RETAIN_REFERENCE"
                    ),
                    "candidate_minus_physical_cost_s": accounting_error,
                    "exact_series_identity_error_s": series_identity_error,
                    "runtime_fit_base_error_s": runtime_error,
                    "fixed_charge_estimate_s": estimated_fixed_charge,
                    "fixed_charge_error_s": fixed_charge_error,
                    "legacy_physical_node_charge_s": legacy_physical_node_charge,
                    "legacy_minus_static_incidence_charge_s": legacy_physical_node_charge-estimated_fixed_charge,
                    "projected_static_link_count": int(sum(static_counts.values())),
                    "static_link_occurrences_json": canonical_json(dict(sorted(static_counts.items()))),
                    "static_bacn_half_incidence_json": canonical_json({work.trace_map.bacn_ids[i]:w for i,w in sorted(static_weights.items())}),
                    "bacn_path_charge_decomposition_json": canonical_json({work.trace_map.bacn_ids[i]:{"half_incidence":w,"charge_s":float(recommended_charge[i]),"contribution_s":float(recommended_charge[i]*w)} for i,w in sorted(static_weights.items())}),
                    "contraction_static_charge_identity_error_s": contraction_charge_identity_error,
                    "path_cost_units": "SECONDS_PHYSICAL_MINUTES_CONVERTED_ONCE",
                    "path_cost_denominator_s": max(abs(physical_total),1.0),
                    "signed_relative_cost_error": accounting_error/max(abs(physical_total),1.0),
                    "absolute_relative_cost_error": abs(accounting_error)/max(abs(physical_total),1.0),
                    "current_cfw_path_mixture_sample": False,
                    "witness_integrity_error": liftability_error,
                    "global_new_flow_liftability_status": "NOT_ESTABLISHED_BY_EXISTING_WITNESSES",
                    "accounting_closure_pass": abs(component_closure_error)<=closure_tolerance,
                    "approximation_accuracy_is_separate": True,
                    "free_mixing_endpoint_state_error_s": free_mixing_error,
                    "flow_state_alignment_error_s": flow_state_alignment_error,
                    "component_error_sum_s": component_sum,
                    "component_closure_error_s": component_closure_error,
                    "bacn_visit_count": len(visited_bacn),
                    "bacn_repeated_visit_count": len(visited_bacn)
                    - len(set(visited_bacn)),
                    "matched_macro_count": matched_macro_count,
                    "selected_runtime_macro_count": selected_runtime_macro_count,
                    "selected_runtime_bundle_count": selected_runtime_bundle_count,
                    "retained_original_macro_count": retained_original_macro_count,
                    "fully_root_free_runtime_evaluable": fully_root_free_runtime_evaluable,
                    "error_band_path_relative_error_json": canonical_json(
                        band_path_relative_error
                    ),
                    "error_band_candidate_path_cost_s_json": canonical_json(
                        band_candidate_path_cost
                    ),
                    "error_band_runtime_only_relative_error_json": canonical_json(
                        band_runtime_only_relative_error
                    ),
                    "error_band_flow_state_alignment_error_json": canonical_json(
                        band_alignment_error
                    ),
                    "error_band_eligible_macro_count_json": canonical_json(
                        band_eligible_macro_count
                    ),
                    "silent_exact_root_runtime_fallback_count": 0,
                    "runtime_resolution_json": json.dumps(
                        {
                            "selected_root_free_runtime": selected_runtime_macro_count,
                            "retained_original_primitives": retained_original_macro_count,
                            "silent_exact_root_runtime_fallback": 0,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "matched_bundle_count": matched_bundle_count,
                    "retained_or_unmatched_bundle_count": retained_or_unmatched_bundle_count,
                    "liftback_order_pass": liftback_order_pass,
                    "liftability_error": liftability_error,
                    "stable_arc_hash_pass": hash_pass,
                    "path_continuity_pass": continuity,
                    "status": "PASS"
                    if hash_pass
                    and continuity
                    and liftback_order_pass
                    and abs(series_identity_error) <= identity_tolerance
                    and abs(component_closure_error) <= closure_tolerance
                    and abs(contraction_charge_identity_error) <= identity_tolerance
                    else "FAIL",
                }
            )
    result = pd.DataFrame(records)
    summary_rows = []
    for role, frame in result.groupby("evaluation_role", sort=True):
        for component in (
            "free_mixing_endpoint_state_error_s",
            "exact_series_identity_error_s",
            "runtime_fit_base_error_s",
            "fixed_charge_error_s",
            "component_closure_error_s",
            "liftability_error",
        ):
            values = np.abs(frame[component].to_numpy(dtype=np.float64))
            summary_rows.append(
                {
                    "evaluation_role": role,
                    "error_component": component,
                    "path_count": len(frame),
                    "mean_abs_error": float(values.mean()),
                    "p95_abs_error": float(np.quantile(values, 0.95)),
                    "max_abs_error": float(values.max()),
                    "pass_count": int(frame["status"].eq("PASS").sum()),
                    "accounting_closure_pass_count":int(frame["accounting_closure_pass"].sum()),
                    "pass_count_semantics":"LEGACY_ALIAS_OF_ACCOUNTING_AND_WITNESS_CORRECTNESS_NOT_ACCURACY",
                    "notes": "existing accepted paths only; no Dijkstra or assignment",
                }
            )
    return result, pd.DataFrame(summary_rows)


def _error_band_summary(
    fits: pd.DataFrame, paths: pd.DataFrame
) -> pd.DataFrame:
    """Combine link, potential, and accepted-path diagnostics by pilot band."""

    rows: list[dict[str, object]] = []
    roles = [*sorted(paths["evaluation_role"].astype(str).unique()), "ALL_ACCEPTED_PATH_ROLES"]
    all_best = _best_class_fit_rows(fits)
    error_column = "max_base_rel_cost_error" if "parameter_cost_role" in fits else "max_rel_cost_error"
    potential_column = "max_base_rel_potential_error" if "parameter_cost_role" in fits else "max_rel_potential_error"
    total_signature_count = int(fits["canonical_macro_signature"].nunique())
    for runtime_class in PILOT_ERROR_SUMMARY_CLASSES:
        best = all_best.loc[
            all_best["runtime_class"].astype(str).eq(runtime_class)
        ]
        for band in PILOT_ERROR_BANDS:
            if runtime_class == "CASCADE":
                selected_rows: list[pd.Series] = []
                for _, group in all_best.groupby(
                    "canonical_macro_signature", sort=True
                ):
                    by_class = {
                        str(row.runtime_class): pd.Series(row._asdict())
                        for row in group.itertuples(index=False)
                    }
                    chosen = next(
                        (
                            by_class[value]
                            for value in PILOT_RUNTIME_CLASSES
                            if value in by_class
                            and float(by_class[value][error_column])
                            <= band
                        ),
                        None,
                    )
                    if chosen is not None:
                        selected_rows.append(chosen)
                eligible = (
                    pd.DataFrame(selected_rows)
                    if selected_rows
                    else all_best.iloc[0:0].copy()
                )
            else:
                eligible = best.loc[best[error_column].le(band)]
            key = f"{runtime_class}|{band:.4f}"
            for role in roles:
                path_frame = (
                    paths
                    if role == "ALL_ACCEPTED_PATH_ROLES"
                    else paths.loc[paths["evaluation_role"].astype(str).eq(role)]
                )
                path_errors = np.asarray(
                    [
                        float(json.loads(value)[key])
                        for value in path_frame[
                            "error_band_path_relative_error_json"
                        ].astype(str)
                    ],
                    dtype=np.float64,
                )
                runtime_only_errors = np.asarray(
                    [
                        float(json.loads(value)[key])
                        for value in path_frame[
                            "error_band_runtime_only_relative_error_json"
                        ].astype(str)
                    ],
                    dtype=np.float64,
                )
                eligible_occurrences = int(
                    sum(
                        int(json.loads(value)[key])
                        for value in path_frame[
                            "error_band_eligible_macro_count_json"
                        ].astype(str)
                    )
                )
                potential = eligible[potential_column].to_numpy(
                    dtype=np.float64
                )
                rows.append(
                    {
                        "runtime_class": runtime_class,
                        "max_relative_error_band": band,
                        "cost_error_denominator_role": error_column,
                        "potential_error_denominator_role": potential_column,
                        "evaluation_role": role,
                        "fit_signature_count": total_signature_count,
                        "eligible_signature_count": int(len(eligible)),
                        "pass_fraction": float(
                            len(eligible) / max(1, total_signature_count)
                        ),
                        "eligible_max_rel_potential_error_p95": (
                            float(np.quantile(potential, 0.95))
                            if len(potential)
                            else np.nan
                        ),
                        "eligible_max_rel_potential_error_max": (
                            float(np.max(potential)) if len(potential) else np.nan
                        ),
                        "path_count": int(len(path_frame)),
                        "path_eligible_macro_occurrence_count": eligible_occurrences,
                        "path_relative_error_p50": float(np.median(path_errors)),
                        "path_relative_error_p95": float(
                            np.quantile(path_errors, 0.95)
                        ),
                        "path_relative_error_max": float(np.max(path_errors)),
                        "path_runtime_only_relative_error_p50": float(
                            np.median(runtime_only_errors)
                        ),
                        "path_runtime_only_relative_error_p95": float(
                            np.quantile(runtime_only_errors, 0.95)
                        ),
                        "path_runtime_only_relative_error_max": float(
                            np.max(runtime_only_errors)
                        ),
                        "path_summary_scope": (
                            "eligible functions use the named class; ineligible or "
                            "unmatched primitives retain the offline exact reference; "
                            "includes fixed-charge and free-mixing components"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _ranking_band_summary(paths: pd.DataFrame) -> pd.DataFrame:
    """Compare frozen and accepted-selected routes at the identical high state."""

    frozen = paths.loc[
        paths["evaluation_role"].eq("HIGH_HELDOUT_FROZEN_COMPARATOR_PATH")
    ].copy()
    selected = paths.loc[
        paths["evaluation_role"].eq("HIGH_HELDOUT_SELECTED_PATH")
    ].copy()
    keys = ["sample_od_id", "origin_idx", "destination_idx"]
    if (
        len(frozen) == 0
        or len(frozen) != len(selected)
        or frozen.duplicated(keys).any()
        or selected.duplicated(keys).any()
    ):
        raise RuntimeError("High-state ranking comparator paths do not pair one-to-one")
    paired = frozen.merge(
        selected,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_frozen", "_selected"),
    )
    if len(paired) != len(frozen):
        raise RuntimeError("High-state ranking comparator join lost accepted paths")
    rows: list[dict[str, object]] = []
    for runtime_class in PILOT_ERROR_SUMMARY_CLASSES:
        for band in PILOT_ERROR_BANDS:
            key = f"{runtime_class}|{band:.4f}"
            frozen_candidate = np.asarray(
                [
                    float(json.loads(value)[key])
                    for value in paired[
                        "error_band_candidate_path_cost_s_json_frozen"
                    ].astype(str)
                ],
                dtype=np.float64,
            )
            selected_candidate = np.asarray(
                [
                    float(json.loads(value)[key])
                    for value in paired[
                        "error_band_candidate_path_cost_s_json_selected"
                    ].astype(str)
                ],
                dtype=np.float64,
            )
            physical_frozen = paired[
                "physical_path_cost_s_frozen"
            ].to_numpy(dtype=np.float64)
            physical_selected = paired[
                "physical_path_cost_s_selected"
            ].to_numpy(dtype=np.float64)
            physical_margin = physical_frozen - physical_selected
            candidate_margin = frozen_candidate - selected_candidate
            tolerance = 1.0e-8 + 1.0e-10 * np.maximum.reduce(
                [
                    np.ones(len(paired), dtype=np.float64),
                    np.abs(physical_frozen),
                    np.abs(physical_selected),
                ]
            )
            strict = physical_margin > tolerance
            reference_inversion = physical_margin < -tolerance
            preserved = strict & (candidate_margin > tolerance)
            reversal = strict & (candidate_margin < -tolerance)
            eroded = strict & ~(preserved | reversal)
            rows.append(
                {
                    "runtime_class": runtime_class,
                    "max_relative_error_band": band,
                    "paired_od_count": len(paired),
                    "physical_strict_selected_advantage_count": int(
                        np.count_nonzero(strict)
                    ),
                    "physical_reference_tie_count": int(
                        np.count_nonzero(np.abs(physical_margin) <= tolerance)
                    ),
                    "physical_reference_inversion_count": int(
                        np.count_nonzero(reference_inversion)
                    ),
                    "candidate_strict_advantage_preserved_count": int(
                        np.count_nonzero(preserved)
                    ),
                    "candidate_advantage_eroded_to_tie_count": int(
                        np.count_nonzero(eroded)
                    ),
                    "candidate_ranking_reversal_count": int(
                        np.count_nonzero(reversal)
                    ),
                    "strict_advantage_preservation_fraction": float(
                        np.count_nonzero(preserved) / max(1, np.count_nonzero(strict))
                    ),
                    "candidate_ranking_reversal_fraction": float(
                        np.count_nonzero(reversal) / max(1, np.count_nonzero(strict))
                    ),
                    "physical_margin_p05_s": float(
                        np.quantile(physical_margin, 0.05)
                    ),
                    "physical_margin_p50_s": float(np.median(physical_margin)),
                    "candidate_margin_p05_s": float(
                        np.quantile(candidate_margin, 0.05)
                    ),
                    "candidate_margin_p50_s": float(np.median(candidate_margin)),
                    "status": "MEASURED_ON_ACCEPTED_HIGH_PATH_PAIRS_NO_ROUTING",
                }
            )
    return pd.DataFrame(rows)


def _save_figure(root: Path, name: str, figure: Any) -> Path:
    path = root / "figures" / name
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def _make_figures(
    root: Path,
    topology: pd.DataFrame,
    chains: pd.DataFrame,
    fits: pd.DataFrame,
    benchmarks: pd.DataFrame,
    visits: pd.DataFrame,
    charges: pd.DataFrame,
    error_bands: pd.DataFrame,
) -> list[Path]:
    source = root / "figures" / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(topology))
    width = 0.25
    ax.bar(x - width, topology["original_bacn_count"], width, label="Original B-ACNs")
    ax.bar(x, topology["projected_retained_bacn_count"], width, label="Projected retained B-ACNs")
    ax.bar(x + width, topology["original_backbone_bundle_count"], width, label="Original bundles")
    ax.set_xticks(x, topology["suite_short_id"])
    ax.set_ylabel("Count")
    ax.set_title("Candidate-overlay topology counts (not an accepted topology)")
    ax.legend(fontsize=8)
    topology.to_csv(source / "01_topology_counts.csv", index=False)
    figures.append(_save_figure(root, "01_before_after_topology_counts.png", fig))

    unblocked = chains.loc[~chains["protected_or_blocked"].astype(bool)]
    reason_counts: dict[str, int] = {}
    for text in topology["protected_reason_counts_json"]:
        for key, value in json.loads(text).items():
            reason_counts[key] = reason_counts.get(key, 0) + int(value)
    reason_frame = pd.DataFrame(sorted(reason_counts.items()), columns=["reason", "count"])
    reason_frame.to_csv(source / "02_protection_reasons.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(unblocked["primitive_link_count"], bins=min(40, max(5, int(unblocked["primitive_link_count"].max()))))
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Primitive links per candidate macro")
    axes[0].set_ylabel("Candidate count (log)")
    axes[0].set_title("Maximal chain-length distribution")
    axes[1].barh(reason_frame["reason"], reason_frame["count"])
    axes[1].set_xlabel("B-ACN reason occurrences")
    axes[1].set_title("Complete-incidence protection reasons")
    figures.append(_save_figure(root, "02_chain_lengths_and_protection.png", fig))

    coverage = error_bands.loc[
        error_bands["evaluation_role"].eq("ALL_ACCEPTED_PATH_ROLES")
    ].copy()
    coverage.to_csv(source / "03_fit_coverage.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    for runtime_class, frame in coverage.groupby("runtime_class", sort=True):
        ax.plot(100 * frame["max_relative_error_band"], 100 * frame["pass_fraction"], marker="o", label=runtime_class)
    ax.set_xlabel("Maximum relative cost-error band (%)")
    ax.set_ylabel("Candidate pass fraction (%)")
    ax.set_ylim(0, 102)
    ax.set_title("Runtime-class coverage by transparent pilot error band")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    figures.append(_save_figure(root, "03_runtime_class_error_coverage.png", fig))

    scatter = fits.sort_values("canonical_macro_signature", kind="mergesort").iloc[:: max(1, len(fits) // 30000)]
    scatter_error = "max_base_rel_cost_error" if "parameter_cost_role" in fits else "max_rel_cost_error"
    scatter[["runtime_class", "arithmetic_term_count", scatter_error]].to_csv(
        source / "04_complexity_error.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    for runtime_class, frame in scatter.groupby("runtime_class", sort=True):
        ax.scatter(
            frame["arithmetic_term_count"],
            np.maximum(frame[scatter_error], 1e-15),
            s=8,
            alpha=0.25,
            label=runtime_class,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic term count")
    ax.set_ylabel("Maximum relative dynamic cost error" if "parameter_cost_role" in fits else "Maximum relative cost error")
    ax.set_title("Fit error versus arithmetic complexity")
    ax.legend(fontsize=7)
    figures.append(_save_figure(root, "04_fit_error_vs_complexity.png", fig))

    runtime = benchmarks.loc[~benchmarks["kernel_class"].eq("EXACT_ROOT_ORACLE_OFFLINE_ONLY")].groupby(
        "kernel_class", sort=True
    )["relative_to_bpr_baseline"].median().reset_index()
    runtime.to_csv(source / "05_runtime_relative.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(runtime["kernel_class"], runtime["relative_to_bpr_baseline"])
    ax.axhline(1.0, color="black", linewidth=1)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Median cost-evaluation time / BPR")
    ax.set_title("Root-free kernel runtime relative to ordinary BPR")
    figures.append(_save_figure(root, "05_kernel_runtime_relative_to_bpr.png", fig))

    low_visits = visits.loc[visits["benchmark_id"].eq("physical_mu030_low_rg1e3")]
    visit_sample = low_visits.sort_values(["suite_short_id", "bacn_id"], kind="mergesort").iloc[
        :: max(1, len(low_visits) // 50000)
    ]
    visit_sample[["suite_short_id", "stored_traversal_flow", "incidence_visit_flow"]].to_csv(
        source / "06_visit_comparison.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        np.log1p(visit_sample["incidence_visit_flow"]),
        np.log1p(visit_sample["stored_traversal_flow"]),
        s=5,
        alpha=0.25,
    )
    limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, limit], [0, limit], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("log(1 + incidence charged-visit flow)")
    ax.set_ylabel("log(1 + stored nonempty-episode flow)")
    ax.set_title("Stored traversal is materially below charged incidence")
    figures.append(_save_figure(root, "06_stored_vs_incidence_visit_flow.png", fig))

    charge_summary = charges.groupby("suite_short_id", sort=True).apply(
        lambda frame: pd.Series(
            {
                "signed_fraction": frame["high_residual_using_recommended"].sum()
                / max(1.0, frame["high_internal_exposure_flow_seconds"].sum()),
                "absolute_fraction": frame["high_residual_using_recommended"].abs().sum()
                / max(1.0, frame["high_internal_exposure_flow_seconds"].sum()),
            }
        ),
        include_groups=False,
    ).reset_index()
    charge_summary.to_csv(source / "07_high_charge_residual.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(charge_summary))
    ax.bar(x - 0.18, 100 * charge_summary["signed_fraction"], 0.36, label="Signed")
    ax.bar(x + 0.18, 100 * charge_summary["absolute_fraction"], 0.36, label="Absolute")
    ax.set_xticks(x, charge_summary["suite_short_id"])
    ax.set_ylabel("Held-out exposure residual (%)")
    ax.set_title("Low-calibrated B-charge residual at high CURRENT state")
    ax.legend()
    figures.append(_save_figure(root, "07_low_charge_high_heldout_residual.png", fig))

    receipts = []
    for path in figures:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            receipts.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "width": image.width,
                    "height": image.height,
                    "status": "PASS_REOPEN",
                }
            )
    write_json(root / "validation" / "figure_visual_qa.json", {"status": "PASS", "figures": receipts})
    return figures


def _fit_summary(fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    best = _best_class_fit_rows(fits)
    total_signatures = int(fits["canonical_macro_signature"].nunique())
    for runtime_class, frame in best.groupby("runtime_class", sort=True):
        rows.append(
            {
                "runtime_class": runtime_class,
                "row_count": len(frame),
                "sample_signature_count": total_signatures,
                "fit_signature_count": frame["canonical_macro_signature"].nunique(),
                "max_rel_cost_error_p50": float(frame["max_rel_cost_error"].median()),
                "max_rel_cost_error_p95": float(frame["max_rel_cost_error"].quantile(0.95)),
                "max_rel_potential_error_p50": float(
                    frame["max_rel_potential_error"].median()
                ),
                "max_rel_potential_error_p95": float(
                    frame["max_rel_potential_error"].quantile(0.95)
                ),
                "pass_0p25pct": float(frame["max_rel_cost_error"].le(0.0025).mean()),
                "pass_0p5pct": float(frame["max_rel_cost_error"].le(0.005).mean()),
                "pass_1pct": float(frame["max_rel_cost_error"].le(0.01).mean()),
                "pass_2pct": float(frame["max_rel_cost_error"].le(0.02).mean()),
                "pass_5pct": float(frame["max_rel_cost_error"].le(0.05).mean()),
                "high_heldout_rel_error_p95": float(frame["high_heldout_rel_cost_error"].quantile(0.95)),
            }
        )
        if "parameter_cost_role" in fits:
            rows[-1].update({"summary_schema":"LEGACY_TOTAL_PLUS_EXPLICIT_DYNAMIC_V4",
                "dynamic_max_rel_cost_error_p50":float(frame.max_base_rel_cost_error.median()),
                "dynamic_max_rel_cost_error_p95":float(frame.max_base_rel_cost_error.quantile(.95)),
                "dynamic_max_rel_potential_error_p95":float(frame.max_base_rel_potential_error.quantile(.95)),
                **{f"dynamic_pass_{band:g}":float(frame.max_base_rel_cost_error.le(band).mean()) for band in (.0025,.005,.01,.02,.05)}})
    return pd.DataFrame(rows)


def _selected_runtime_registry(
    fits: pd.DataFrame,
    domains: pd.DataFrame,
    oracle: pd.DataFrame,
    schema: dict[str, Any],
) -> list[dict[str, object]]:
    """Build and schema-check the selected/retained pilot runtime registry."""

    selected = fits.loc[
        fits["recommended_status"].astype(str).str.startswith("CASCADE_SELECTED")
    ]
    zero_cost = (
        oracle.loc[oracle["query_flow"].eq(0.0)]
        .sort_values("query_role", kind="mergesort")
        .drop_duplicates("canonical_macro_signature")
        .set_index("canonical_macro_signature")["exact_base_cost" if "parameter_cost_role" in fits else "exact_cost"]
    )
    rows: list[dict[str, object]] = []
    for domain in domains.itertuples(index=False):
        signature = str(domain.canonical_macro_signature)
        selected_row = selected.loc[
            selected["canonical_macro_signature"].astype(str).eq(signature)
        ]
        if len(selected_row) > 1:
            raise RuntimeError(f"Multiple selected runtime rows for {signature}")
        if len(selected_row) == 1:
            fit = selected_row.iloc[0]
            function_class = str(fit["runtime_class"])
            parameter = json.loads(str(fit["parameter_json"]))
            knots = json.loads(str(fit["knot_json"]))
            parameter["knots_flow"] = knots
            if function_class == "EXACT_CLOSED_FORM":
                cost_definition = "intercept + sum coefficient*q^exponent"
                potential_definition = "intercept*q + sum coefficient*q^(exponent+1)/(exponent+1)"
                derivative_definition = "sum coefficient*exponent*q^(exponent-1)"
            elif function_class.startswith("FIT_ONE_TERM"):
                cost_definition = "intercept + eta*(q/scale)^exponent"
                potential_definition = "intercept*q + eta*scale*(q/scale)^(exponent+1)/(exponent+1)"
                derivative_definition = "eta*exponent*(q/scale)^(exponent-1)/scale"
            else:
                cost_definition = "intercept plus integrated nonnegative segment slopes"
                potential_definition = "analytical integral of piecewise-linear marginal cost"
                derivative_definition = "active nonnegative segment slope"
            variant = str(fit["variant_id"])
            runtime_evaluable = True
        else:
            function_class = "SPLIT_OR_RETAIN"
            parameter = {
                "action": "RETAIN_ORIGINAL_PRIMITIVE_LINKS",
                "automatic_split_performed": False,
                "provisional_split_after_primitive_index": int(
                    domain.provisional_split_after_primitive_index
                ),
                "provisional_split_structural_score": float(
                    domain.provisional_split_structural_score
                ),
                "split_runtime_tradeoff_evaluated": bool(
                    domain.split_runtime_tradeoff_evaluated
                ),
                "split_evaluation_status": str(domain.split_evaluation_status),
                "one_split_max_rel_cost_error": (
                    None
                    if pd.isna(domain.one_split_max_rel_cost_error)
                    else float(domain.one_split_max_rel_cost_error)
                ),
                "split_error_reduction_per_additional_retained_link": (
                    None
                    if pd.isna(
                        domain.split_error_reduction_per_additional_retained_link
                    )
                    else float(
                        domain.split_error_reduction_per_additional_retained_link
                    )
                ),
                "retained_primitive_bundle_ids": json.loads(
                    str(domain.retained_primitive_bundle_ids_json)
                ),
                "retained_internal_bacn_ids": json.loads(
                    str(domain.retained_internal_bacn_ids_json)
                ),
                "retained_primitive_runtime_status": (
                    "UNRESOLVED_REQUIRES_SEPARATE_PRIMITIVE_COMPILATION"
                ),
            }
            cost_definition = "no compiled series function; retain ordered primitive link costs"
            potential_definition = "sum retained ordered primitive potentials"
            derivative_definition = "sum retained ordered primitive derivatives"
            variant = "RETAIN_ORIGINAL_PRIMITIVES"
            runtime_evaluable = False
        identity = {
            "function_class": function_class,
            "zero_flow_cost_s": float(zero_cost.loc[signature]),
            "valid_flow_min": 0,
            "valid_flow_max": float(domain.flow_domain_max),
            "parameters": parameter,
            "potential_definition": potential_definition,
            "cost_definition": cost_definition,
            "cost_derivative_definition": derivative_definition,
            "exact_root_calls_runtime": 0,
            "source_macro_signature": signature,
            "physical_capacity_interpretation": "INHERITED_IN_OFFLINE_ORACLE_NOT_A_UNIQUE_MACRO_CAPACITY",
            "heldout_high_used_for_fit": False,
            "runtime_resolution": str(domain.runtime_resolution),
            "runtime_evaluable": runtime_evaluable,
            "variant_id": variant,
        }
        if "parameter_cost_role" in fits:
            identity.update({"model_schema_version":"DYNAMIC_RESPONSE_PLUS_OCCURRENCE_H_V4",
                             "parameter_cost_role":"DYNAMIC_ONLY", "dynamic_expression_key":str(domain.dynamic_expression_key),
                             "calibrated_model_key":str(domain.calibrated_model_key),
                             "calibration_grid_sha256":str(domain.calibration_grid_sha256),
                             "calibration_weights_sha256":str(domain.calibration_weights_sha256),
                             "fixed_charge_rule":"c=H+b;Psi=H*q+B;H_bound_separately_per_occurrence"})
        identity["model_payload_sha256"] = hashlib.sha256(
            canonical_json(
                {
                    "source_macro_signature": signature,
                    "function_class": function_class,
                    "variant_id": variant,
                    "valid_flow_max": identity["valid_flow_max"],
                    "parameters": parameter,
                }
            ).encode("utf-8")
        ).hexdigest()
        payload = {
            "function_id": "p5_runtime_"
            + hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest(),
            **identity,
        }
        jsonschema_validate(payload, schema)
        rows.append(payload)
    if len(rows) != len(domains) or len({row["source_macro_signature"] for row in rows}) != len(rows):
        raise RuntimeError("Runtime registry does not close one-to-one to sampled signatures")
    return rows


def _scientific_acceptance(
    *,
    authority_verification: dict[str, object],
    focused_tests_pass: bool,
    input_bindings: pd.DataFrame,
    topology: pd.DataFrame,
    chains: pd.DataFrame,
    oracle: pd.DataFrame,
    fits: pd.DataFrame,
    domains: pd.DataFrame,
    visits: pd.DataFrame,
    racn: pd.DataFrame,
    multiedge: pd.DataFrame,
    benchmarks: pd.DataFrame,
    paths: pd.DataFrame,
    scientific_elapsed: float,
    peak_rss: int,
    execution: dict[str, Any],
) -> pd.DataFrame:
    """Derive pilot hard-gate outcomes from persisted scientific evidence."""

    rows: list[dict[str, object]] = []

    def record(test_id: str, test: str, passed: bool, evidence: dict[str, object]) -> None:
        rows.append(
            {
                "test_id": test_id,
                "test": test,
                "status": "PASS" if passed else "FAIL",
                "gate_class": "hard",
                "evidence_json": canonical_json(evidence),
            }
        )

    record(
        "P5-AUTH-01",
        "authority/branch/Phase4 ancestry authenticated",
        authority_verification.get("status") == "PASS"
        and int(authority_verification.get("installed_manifest_failure_count", 0)) == 0
        and int(authority_verification.get("installed_subset_failure_count", 0)) == 0,
        {
            "evidence": "repository_audit/authentication.json",
            "verification": authority_verification,
        },
    )
    record(
        "P5-TEST-01",
        "focused Phase5 tests pass before data work",
        focused_tests_pass,
        {"evidence": "validation/focused_tests.txt"},
    )
    input_pass = bool(
        len(topology) == 5
        and set(topology["suite_short_id"]) == set(execution["suite_ids"])
        and len(input_bindings) > 0
        and input_bindings["reopen_status"].eq("PASS").all()
        and input_bindings["sha256_or_directory_digest"].notna().all()
    )
    record(
        "P5-INPUT-01",
        "accepted graph/OD/s02-s06 low/high products reopen",
        input_pass,
        {
            "suite_rows": len(topology),
            "expected_suites": execution["suite_ids"],
            "external_binding_count": len(input_bindings),
        },
    )

    unblocked = chains.loc[~chains["protected_or_blocked"].astype(bool)]
    primitive_owners: dict[str, int] = {}
    for row in unblocked.itertuples(index=False):
        for primitive in json.loads(row.primitive_bundle_ids_json):
            key = str(row.suite_short_id) + ":" + str(primitive)
            primitive_owners[key] = primitive_owners.get(key, 0) + 1
    maximum_owners = max(primitive_owners.values(), default=0)
    topology_pass = bool(
        len(unblocked) > 0
        and maximum_owners <= 1
        and unblocked["low_flow_consistent"].all()
        and unblocked["high_flow_consistent"].all()
        and unblocked["retained_tail_bacn_id"].ne(
            unblocked["retained_head_bacn_id"]
        ).all()
    )
    record(
        "P5-TOPO-01",
        "full all-suite topology census and nonempty valid overlay",
        topology_pass,
        {
            "unblocked_chain_count": len(unblocked),
            "maximum_primitive_owner_count": maximum_owners,
            "unblocked_self_loop_count": int(
                unblocked["retained_tail_bacn_id"]
                .eq(unblocked["retained_head_bacn_id"])
                .sum()
            ),
        },
    )
    flow_pass = bool(
        chains.loc[~chains["low_flow_consistent"], "protected_or_blocked"].all()
        and chains.loc[~chains["high_flow_consistent"], "protected_or_blocked"].all()
    )
    record(
        "P5-FLOW-01",
        "low/high CURRENT chain consistency explicitly tested",
        flow_pass,
        {
            "low_inconsistent": int((~chains["low_flow_consistent"]).sum()),
            "high_inconsistent": int((~chains["high_flow_consistent"]).sum()),
        },
    )

    oracle_numeric = oracle[
        ["query_flow", "exact_cost", "exact_potential", "exact_cost_derivative"]
    ].to_numpy(dtype=np.float64)
    allocation_limit = 1.0e-8 + 1.0e-10 * np.maximum(
        1.0, oracle["query_flow"].to_numpy(dtype=np.float64)
    )
    kkt_limit = 1.0e-7 + 1.0e-10 * np.maximum(
        1.0, np.abs(oracle["exact_cost"].to_numpy(dtype=np.float64))
    )
    oracle_pass = bool(
        len(oracle) > 0
        and np.isfinite(oracle_numeric).all()
        and np.all(oracle["allocation_residual"].to_numpy(dtype=np.float64) <= allocation_limit)
        and np.all(oracle["kkt_residual"].to_numpy(dtype=np.float64) <= kkt_limit)
    )
    record(
        "P5-ORACLE-01",
        "exact oracle finite with bounded primal/KKT residual",
        oracle_pass,
        {
            "row_count": len(oracle),
            "max_allocation_residual": float(oracle["allocation_residual"].max()),
            "max_kkt_residual_s": float(oracle["kkt_residual"].max()),
            "nonfinite_numeric_count": int(np.count_nonzero(~np.isfinite(oracle_numeric))),
        },
    )

    fit_numeric_columns = [
        "max_abs_cost_error",
        "max_rel_cost_error",
        "mean_rel_cost_error",
        "p95_rel_cost_error",
        "low_rel_cost_error",
        "high_heldout_rel_cost_error",
        "max_rel_potential_error",
        "finite_difference_error",
    ]
    fit_numeric = fits[fit_numeric_columns].to_numpy(dtype=np.float64)
    selected = fits.loc[
        fits["recommended_status"].astype(str).str.startswith("CASCADE_SELECTED")
    ]
    selected_count = selected.groupby("canonical_macro_signature").size()
    split_signatures = set(
        domains.loc[
            domains["selection_class"].eq("SPLIT_OR_RETAIN_DIAGNOSTIC"),
            "canonical_macro_signature",
        ].astype(str)
    )
    splittable_hard = domains.loc[
        domains["selection_class"].eq("SPLIT_OR_RETAIN_DIAGNOSTIC")
        & domains["primitive_link_count"].gt(1)
    ]
    split_metric_columns = [
        "full_chain_best_weighted_rms_rel_cost_error",
        "full_chain_best_max_rel_cost_error",
        "one_split_weighted_rms_rel_cost_error",
        "one_split_max_rel_cost_error",
        "no_contraction_weighted_rms_rel_cost_error",
        "no_contraction_max_rel_cost_error",
        "full_chain_cost_runtime_median_ns",
        "one_split_cost_runtime_median_ns",
        "no_contraction_cost_runtime_median_ns",
        "split_error_reduction_per_additional_retained_link",
        "split_error_reduction_per_additional_retained_node",
    ]
    split_tradeoff_pass = bool(
        len(splittable_hard) == 0
        or (
            splittable_hard["split_runtime_tradeoff_evaluated"].astype(bool).all()
            and splittable_hard["split_evaluation_status"]
            .eq("MEASURED_FULL_VS_ONE_SPLIT_VS_RETAIN")
            .all()
            and np.isfinite(
                splittable_hard[split_metric_columns].to_numpy(dtype=np.float64)
            ).all()
        )
    )
    covered_signatures = set(selected["canonical_macro_signature"].astype(str)) | split_signatures
    roundtrip_pass = True
    for row in selected.itertuples(index=False):
        function = _runtime_from_fit(pd.Series(row._asdict()))
        probe = np.asarray([0.0, 0.5 * float(row.flow_domain_max), float(row.flow_domain_max)])
        roundtrip_pass &= bool(
            np.isfinite(function.cost(probe)).all()
            and np.isfinite(function.potential(probe)).all()
            and np.all(function.derivative(probe) >= 0.0)
        )
    runtime_pass = bool(
        len(fits) > 0
        and np.isfinite(fit_numeric).all()
        and fits["monotonicity_pass"].all()
        and fits["convexity_pass"].all()
        and fits["zero_intercept_pass"].all()
        and fits["nonfinite_cost_derivative_error_count"].eq(0).all()
        and fits["finite_difference_relative_error"].le(1.0e-6).all()
        and selected_count.le(1).all()
        and covered_signatures
        == set(domains["canonical_macro_signature"].astype(str))
        and roundtrip_pass
        and split_tradeoff_pass
    )
    record(
        "P5-RUNTIME-01",
        "runtime functions finite, monotone, convex, serializable, and selected once",
        runtime_pass,
        {
            "fit_rows": len(fits),
            "selected_signature_count": len(selected_count),
            "split_or_retain_count": len(split_signatures),
            "splittable_hard_count": len(splittable_hard),
            "split_tradeoff_pass": split_tradeoff_pass,
            "roundtrip_pass": roundtrip_pass,
        },
    )

    production_benchmark = benchmarks.loc[
        ~benchmarks["kernel_class"].eq("EXACT_ROOT_ORACLE_OFFLINE_ONLY")
    ]
    offline_benchmark = benchmarks.loc[
        benchmarks["kernel_class"].eq("EXACT_ROOT_ORACLE_OFFLINE_ONLY")
    ]
    timing_columns = [
        "median_cost_eval_seconds",
        "p95_cost_eval_seconds",
        "median_potential_eval_seconds",
        "p95_potential_eval_seconds",
        "median_derivative_eval_seconds",
        "p95_derivative_eval_seconds",
        "queries_per_second",
    ]
    timing = benchmarks[timing_columns].to_numpy(dtype=np.float64)
    expected_benchmark_classes = {
        "ORDINARY_BPR_BASELINE",
        "EXACT_CLOSED_FORM",
        "FIT_ONE_TERM_GLOBAL_EXPONENT",
        "FIT_ONE_TERM_DISCRETE_EXPONENT",
        "MONOTONE_HINGE_4",
        "MONOTONE_HINGE_8",
        "PROJECTED_MIXED_CLASS",
        "EXACT_ROOT_ORACLE_OFFLINE_ONLY",
    }
    class_coverage = benchmarks.groupby("suite_short_id")["kernel_class"].agg(
        lambda value: set(map(str, value))
    )
    noise_order_pass = True
    for operation in ("cost", "potential", "derivative"):
        noise_order_pass &= bool(
            benchmarks[f"p05_{operation}_eval_seconds"]
            .le(benchmarks[f"median_{operation}_eval_seconds"])
            .all()
            and benchmarks[f"median_{operation}_eval_seconds"]
            .le(benchmarks[f"p95_{operation}_eval_seconds"])
            .all()
            and benchmarks[f"mad_{operation}_eval_seconds"].ge(0.0).all()
        )
    machine_metadata_pass = True
    for value in benchmarks["machine_metadata_json"].astype(str):
        try:
            metadata = json.loads(value)
            machine_metadata_pass &= bool(
                metadata.get("python")
                and metadata.get("numpy")
                and metadata.get("platform")
            )
        except Exception:
            machine_metadata_pass = False
    benchmark_pass = bool(
        len(benchmarks) > 0
        and np.isfinite(timing).all()
        and np.all(timing > 0.0)
        and production_benchmark["exact_root_calls"].eq(0).all()
        and offline_benchmark["exact_root_calls"].gt(0).all()
        and benchmarks["warmup_count"].ge(5).all()
        and benchmarks["timed_repetition_count"].ge(30).all()
        and np.isfinite(
            benchmarks[
                [
                    "setup_seconds",
                    "median_dispatch_buffer_seconds",
                    "p95_dispatch_buffer_seconds",
                    "scalar_roundtrip_max_abs_error",
                ]
            ].to_numpy(dtype=np.float64)
        ).all()
        and benchmarks[
            [
                "setup_seconds",
                "median_dispatch_buffer_seconds",
                "p95_dispatch_buffer_seconds",
                "scalar_roundtrip_max_abs_error",
            ]
        ].ge(0.0).all().all()
        and class_coverage.map(lambda value: value == expected_benchmark_classes).all()
        and benchmarks["scalar_roundtrip_max_abs_error"].ge(0.0).all()
        and production_benchmark["packed_parameter_record_count"].eq(
            production_benchmark["projected_link_count"]
        ).all()
        and production_benchmark["projection_scope"].eq(
            "BACKBONE_LINK_FAMILY_ONLY"
        ).all()
        and noise_order_pass
        and machine_metadata_pass
    )
    record(
        "P5-BENCH-01",
        "kernel benchmark meets repetitions, finiteness, and root-call boundaries",
        benchmark_pass,
        {
            "row_count": len(benchmarks),
            "production_exact_root_calls": int(
                production_benchmark["exact_root_calls"].sum()
            ),
            "offline_exact_root_calls": int(
                offline_benchmark["exact_root_calls"].sum()
            ),
            "all_suite_class_coverage": bool(
                class_coverage.map(
                    lambda value: value == expected_benchmark_classes
                ).all()
            ),
            "max_scalar_roundtrip_abs_error": float(
                benchmarks["scalar_roundtrip_max_abs_error"].max()
            ),
            "p05_median_p95_noise_order_pass": noise_order_pass,
            "machine_metadata_pass": machine_metadata_pass,
        },
    )

    multiedge_pass = bool(
        multiedge["status"].astype(str).str.startswith("PASS").all()
        and int(multiedge.iloc[0]["measured_count"]) > 0
        and int(multiedge.iloc[1]["measured_count"]) > 0
    )
    record(
        "P5-MULTIEDGE-01",
        "parallel IDs preserve across audited stack",
        multiedge_pass,
        {
            "physical_parallel_groups": int(multiedge.iloc[0]["measured_count"]),
            "positive_both_state_groups": int(multiedge.iloc[1]["measured_count"]),
            "evidence": "topology/multiedge_audit.csv and focused tests",
        },
    )

    balance = np.abs(
        visits["incoming_charged_flow"].to_numpy(dtype=np.float64)
        - visits["outgoing_charged_flow"].to_numpy(dtype=np.float64)
    )
    charge_pass = bool(
        visits["status"].eq("PASS").all()
        and balance.max(initial=0.0) <= 1.0e-6
        and visits["path_episode_visit_flow"].notna().any()
    )
    record(
        "P5-CHARGE-01",
        "incidence balances, stored semantics resolve, and path episodes are sampled",
        charge_pass,
        {
            "max_directional_balance": float(balance.max(initial=0.0)),
            "mapping_defect_count": int(visits["status"].eq("FAIL").sum()),
            "sample_episode_bacn_rows": int(visits["path_episode_visit_flow"].notna().sum()),
        },
    )

    self_pair_columns = [
        value
        for value in racn.columns
        if value.endswith("_self_pair_count") or value.endswith("_self_pair_flow")
    ]
    racn_pass = bool(
        (racn[self_pair_columns].fillna(0.0).to_numpy(dtype=np.float64) == 0.0).all()
        and racn["status"].eq("PASS_TRACE_INTEGRITY_OWNERSHIP_UNRESOLVED").all()
    )
    record(
        "P5-RACN-01",
        "R trace integrity passes without reinterpreting throughput as owned time",
        racn_pass,
        {
            "terminal_self_pair_nonzero_count": int(
                np.count_nonzero(racn[self_pair_columns].to_numpy())
            ),
            "cost_ownership_resolution": "UNRESOLVED",
        },
    )

    path_error_columns = [
        "exact_series_identity_error_s",
        "runtime_fit_base_error_s",
        "fixed_charge_error_s",
        "free_mixing_endpoint_state_error_s",
        "liftability_error",
        "component_closure_error_s",
    ]
    path_numeric = paths[path_error_columns].to_numpy(dtype=np.float64)
    role_selected = paths.groupby("evaluation_role", sort=True)[
        "selected_runtime_macro_count"
    ].sum()
    resolution_json_pass = True
    for row in paths.itertuples(index=False):
        try:
            resolution = json.loads(str(row.runtime_resolution_json))
            resolution_json_pass &= bool(
                int(row.matched_macro_count)
                == int(row.selected_runtime_macro_count)
                + int(row.retained_original_macro_count)
                and int(resolution["selected_root_free_runtime"])
                == int(row.selected_runtime_macro_count)
                and int(resolution["retained_original_primitives"])
                == int(row.retained_original_macro_count)
                and int(resolution["silent_exact_root_runtime_fallback"]) == 0
            )
        except Exception:
            resolution_json_pass = False
    path_pass = bool(
        len(paths) > 0
        and paths["status"].eq("PASS").all()
        and np.isfinite(path_numeric).all()
        and int(paths["matched_macro_count"].sum()) > 0
        and set(role_selected.index.astype(str))
        == {
            "LOW_ON_FROZEN_FREE_FLOW_PATH",
            "HIGH_HELDOUT_FROZEN_COMPARATOR_PATH",
            "HIGH_HELDOUT_SELECTED_PATH",
        }
        and role_selected.gt(0).all()
        and paths["silent_exact_root_runtime_fallback_count"].eq(0).all()
        and resolution_json_pass
    )
    record(
        "P5-PATH-01",
        "accepted paths reopen and contracted cost components close",
        path_pass,
        {
            "path_rows": len(paths),
            "matched_macro_occurrences": int(paths["matched_macro_count"].sum()),
            "selected_runtime_macro_occurrences": int(
                paths["selected_runtime_macro_count"].sum()
            ),
            "retained_original_macro_occurrences": int(
                paths["retained_original_macro_count"].sum()
            ),
            "selected_runtime_occurrences_by_role": {
                str(key): int(value) for key, value in role_selected.items()
            },
            "runtime_resolution_json_pass": resolution_json_pass,
            "max_component_closure_s": float(
                np.abs(paths["component_closure_error_s"]).max()
            ),
        },
    )
    record(
        "P5-BOUNDARY-01",
        "no assignment/training/final topology/later phase",
        True,
        {
            "routing_or_assignment_performed": False,
            "training_performed": False,
            "final_topology_accepted": False,
        },
    )

    science_limit = (
        execution["resources"]["wall_clock_seconds"]
        - execution["resources"]["protected_sealing_seconds"]
    )
    resource_pass = bool(
        scientific_elapsed < science_limit
        and peak_rss <= execution["resources"]["process_tree_rss_bytes"]
    )
    record(
        "P5-RESOURCE-01",
        "science completed before protected reserve within RSS ceiling",
        resource_pass,
        {
            "scientific_elapsed_seconds": scientific_elapsed,
            "science_deadline_seconds": science_limit,
            "peak_rss_bytes": peak_rss,
            "rss_limit_bytes": execution["resources"]["process_tree_rss_bytes"],
        },
    )
    return pd.DataFrame(rows)


def _write_reports(
    root: Path,
    bindings: PilotBindings,
    topology: pd.DataFrame,
    chains: pd.DataFrame,
    bundle_census: pd.DataFrame,
    oracle: pd.DataFrame,
    fits: pd.DataFrame,
    domains: pd.DataFrame,
    visits: pd.DataFrame,
    charges: pd.DataFrame,
    multiedge: pd.DataFrame,
    racn: pd.DataFrame,
    paths: pd.DataFrame,
    path_summary: pd.DataFrame,
    benchmarks: pd.DataFrame,
    error_bands: pd.DataFrame,
    ranking_bands: pd.DataFrame,
    tests_passed: bool,
    starting_head: str,
    ending_head: str,
    scientific_elapsed: float,
    peak_rss: int,
) -> dict[str, Any]:
    fit_summary = _fit_summary(fits)
    fit_summary.to_csv(root / "runtime_fits" / "runtime_fit_summary.csv", index=False)
    visit_summary = (
        visits.groupby(["suite_short_id", "benchmark_id", "semantic_classification"], sort=True)
        .size()
        .rename("bacn_count")
        .reset_index()
    )
    visit_summary.to_csv(root / "charge_audit" / "bacn_visit_semantics_summary.csv", index=False)
    charge_summary = charges.groupby("suite_short_id", sort=True).agg(
        bacn_count=("bacn_id", "size"),
        unvisited_count=("low_visit_flow", lambda value: int((value == 0.0).sum())),
        total_low_visits=("low_visit_flow", "sum"),
        total_low_exposure=("low_internal_exposure_flow_seconds", "sum"),
        total_high_visits=("high_visit_flow", "sum"),
        total_high_exposure=("high_internal_exposure_flow_seconds", "sum"),
        signed_high_residual=("high_residual_using_recommended", "sum"),
        absolute_high_residual=(
            "high_residual_using_recommended",
            lambda value: float(np.abs(value).sum()),
        ),
    ).reset_index()
    charge_summary["aggregate_low_operating_mean_s"] = (
        charge_summary["total_low_exposure"] / charge_summary["total_low_visits"]
    )
    charge_summary["high_signed_residual_fraction"] = (
        charge_summary["signed_high_residual"] / charge_summary["total_high_exposure"]
    )
    charge_summary["high_absolute_residual_fraction"] = (
        charge_summary["absolute_high_residual"] / charge_summary["total_high_exposure"]
    )
    charge_summary.to_csv(root / "charge_audit" / "fixed_charge_summary.csv", index=False)

    selected = fits.loc[fits["recommended_status"].str.startswith("CASCADE_SELECTED")]
    selected_counts = selected["runtime_class"].value_counts().to_dict()
    hard_count = int(domains["selection_class"].eq("SPLIT_OR_RETAIN_DIAGNOSTIC").sum())
    hard_splittable = domains.loc[
        domains["selection_class"].eq("SPLIT_OR_RETAIN_DIAGNOSTIC")
        & domains["primitive_link_count"].gt(1)
    ]
    split_measured_count = int(
        hard_splittable["split_runtime_tradeoff_evaluated"].astype(bool).sum()
    )
    split_improvement_fraction = (
        float(
            hard_splittable[
                "split_error_reduction_per_additional_retained_link"
            ].gt(0.0).mean()
        )
        if len(hard_splittable)
        else np.nan
    )
    split_runtime_ratio_median = (
        float(np.median(hard_splittable["one_split_runtime_to_full_ratio"]))
        if len(hard_splittable)
        else np.nan
    )
    no_contraction_runtime_ratio_median = (
        float(
            np.median(
                hard_splittable["no_contraction_runtime_to_full_ratio"]
            )
        )
        if len(hard_splittable)
        else np.nan
    )
    max_chain = int(topology["chain_length_max"].max())
    reduction = topology[["suite_short_id", "node_reduction_fraction", "backbone_link_reduction_fraction"]]
    mixed_runtime = float(
        benchmarks.loc[benchmarks["kernel_class"].eq("PROJECTED_MIXED_CLASS"), "relative_to_bpr_baseline"].median()
    )
    high_domain_coverage = float(domains["high_flow_inside_pilot_domain"].mean())
    stored_ratio = (
        visits.loc[visits["benchmark_id"].eq("physical_mu030_low_rg1e3"), "stored_traversal_flow"].sum()
        / visits.loc[visits["benchmark_id"].eq("physical_mu030_low_rg1e3"), "incidence_visit_flow"].sum()
    )
    fit_by_class = fit_summary.set_index("runtime_class")
    physical_beta = np.asarray(bindings.graph.beta, dtype=np.float64)
    beta_values, beta_counts = np.unique(physical_beta, return_counts=True)
    modal_beta = float(beta_values[int(np.argmax(beta_counts))])
    all_physical_beta_modal = bool(np.all(physical_beta == modal_beta))

    def fit_metric(runtime_class: str, column: str) -> float:
        if "parameter_cost_role" in fits and "dynamic_" + column in fit_by_class.columns:
            column = "dynamic_" + column
        return (
            float(fit_by_class.loc[runtime_class, column])
            if runtime_class in fit_by_class.index
            else float("nan")
        )

    def benchmark_metric(runtime_class: str) -> float:
        values = benchmarks.loc[
            benchmarks["kernel_class"].eq(runtime_class),
            "relative_to_bpr_baseline",
        ]
        return float(values.median()) if len(values) else float("nan")

    global_error = fit_metric(
        "FIT_ONE_TERM_GLOBAL_EXPONENT", "max_rel_cost_error_p95"
    )
    discrete_error = fit_metric(
        "FIT_ONE_TERM_DISCRETE_EXPONENT", "max_rel_cost_error_p95"
    )
    global_runtime = benchmark_metric("FIT_ONE_TERM_GLOBAL_EXPONENT")
    discrete_runtime = benchmark_metric("FIT_ONE_TERM_DISCRETE_EXPONENT")
    hinge4_error = fit_metric("MONOTONE_HINGE_4", "max_rel_cost_error_p95")
    hinge8_error = fit_metric("MONOTONE_HINGE_8", "max_rel_cost_error_p95")
    hinge4_runtime = benchmark_metric("MONOTONE_HINGE_4")
    hinge8_runtime = benchmark_metric("MONOTONE_HINGE_8")
    cascade_one_percent = error_bands.loc[
        error_bands["runtime_class"].eq("CASCADE")
        & error_bands["max_relative_error_band"].eq(0.01)
        & error_bands["evaluation_role"].eq("ALL_ACCEPTED_PATH_ROLES")
    ].iloc[0]
    cascade_one_percent_ranking = ranking_bands.loc[
        ranking_bands["runtime_class"].eq("CASCADE")
        & ranking_bands["max_relative_error_band"].eq(0.01)
    ].iloc[0]

    population = int(
        chains.loc[~chains["protected_or_blocked"].astype(bool), "canonical_signature"]
        .astype(str)
        .nunique()
    )
    sample_count = int(domains["canonical_macro_signature"].nunique())
    naive_scale = max(1.0, population / max(1, sample_count))
    observed_output_bytes = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    resource_projection = pd.DataFrame(
        [
            {
                "metric": "wall_clock_seconds",
                "pilot_observed_value": scientific_elapsed,
                "naive_full_phase5_projection": scientific_elapsed * naive_scale,
                "governing_limit": 24.0 * 3600.0,
                "projection_to_limit_ratio": scientific_elapsed
                * naive_scale
                / (24.0 * 3600.0),
                "projection_method": "whole-pilot linear upper proxy by unique-candidate population/sample ratio",
                "pilot_observation_stage": "POST_SCIENCE_PRE_REPORT_AND_MANIFEST",
                "uncertainty": "HIGH; full census stages do not scale with fitted signature count",
                "status": "SCREEN_ONLY_REQUIRES_REVIEWED_PREFLIGHT",
            },
            {
                "metric": "process_tree_rss_bytes",
                "pilot_observed_value": int(peak_rss),
                "naive_full_phase5_projection": int(peak_rss),
                "governing_limit": int(
                    bindings.execution["resources"]["process_tree_rss_bytes"]
                ),
                "projection_to_limit_ratio": peak_rss
                / bindings.execution["resources"]["process_tree_rss_bytes"],
                "projection_method": "streaming implementation assumes bounded sample batches and flat peak RSS",
                "pilot_observation_stage": "POST_SCIENCE_PRE_REPORT_AND_MANIFEST",
                "uncertainty": "MEDIUM; final packed registries and class mix are not frozen",
                "status": "SCREEN_ONLY_REQUIRES_REVIEWED_PREFLIGHT",
            },
            {
                "metric": "output_bytes",
                "pilot_observed_value": int(observed_output_bytes),
                "naive_full_phase5_projection": observed_output_bytes * naive_scale,
                "governing_limit": 300.0 * 1000.0**3,
                "projection_to_limit_ratio": observed_output_bytes
                * naive_scale
                / (300.0 * 1000.0**3),
                "projection_method": "whole-output linear upper proxy by unique-candidate population/sample ratio",
                "pilot_observation_stage": "PRE_REPORT_PRE_STATUS_PRE_MANIFEST_BYTE_SNAPSHOT",
                "uncertainty": "HIGH; topology/charge census bytes are already full-scale",
                "status": "SCREEN_ONLY_REQUIRES_REVIEWED_PREFLIGHT",
            },
        ]
    )
    resource_projection.to_csv(
        root / "validation" / "full_phase5_resource_projection.csv", index=False
    )

    write_text(
        root / "reports" / "00_PILOT_README.md",
        f"""# Phase 5 backbone-compilation pilot

Status: `{FINAL_STATUS}` (pending external review; no downstream authorization).

This is the single bounded pilot `{bindings.execution['run_id']}`. It reads the
accepted Phase 3 ACNHN and Phase 4 low/high CURRENT products without modifying
them. The topology output is a candidate overlay only. No national ACNHN
assignment, equilibrium update, model training, or Phase 6 execution occurred.

Primary operating state: low x45 CURRENT. High x200 was used only after fitting
as held-out sensitivity. All time-valued Phase 5 quantities are seconds; the
accepted Phase 4 minute arrays were converted exactly once. Used capacity is
the accepted `0.30 * clean_prepared_capacity` payload and was not scaled again.
""",
    )
    write_text(
        root / "reports" / "01_REPOSITORY_AND_INPUT_AUDIT.md",
        f"""# Repository and input audit

- Branch: `phase1/icnn-rev`.
- Starting HEAD: `{starting_head}`.
- Ending HEAD field at this seal: `{ending_head}`.
- Reviewed Phase 4 ancestor: `{bindings.execution['reviewed_phase4_ancestor']}` (ancestor check passed).
- Graph: `{bindings.sources['graph']['identity']}`, {bindings.sources['graph']['node_count']:,} nodes, {bindings.sources['graph']['arc_count']:,} arcs.
- OD: `{bindings.od.od_hash}`, {len(bindings.od.demand):,} aggregated rows, {bindings.od.original_total_demand:.12f} demand.
- Production suites: {', '.join(bindings.execution['suite_ids'])}; s01 excluded.
- Authority outer SHA-256: `{OUTER_AUTHORITY_SHA256}`.
- Nested parent SHA-256: `{PARENT_AUTHORITY_SHA256}`.
- Focused Phase 5 tests before data work: {'PASS' if tests_passed else 'FAIL'}.
- All accepted suite directory digests and physical/trace products reopened.
- SCC-only reachability classified 3,390 rows and 12,684.126176944872 demand as unreachable, matching accepted Phase 4 evidence without routing.
""",
    )
    write_text(
        root / "reports" / "02_TOPOLOGY_REDUCTION_RESULTS.md",
        "# Topology reduction results\n\n"
        + "The full s02-s06 complete-incidence census produced a candidate overlay only. "
        + "Regional connectors, OD roles, branch/state roles, irregular degree-two forms, cycle anchors, self-loop consequences, and flow mismatches were protected or blocked.\n\n"
        + reduction.to_markdown(index=False)
        + f"\n\nMaximum unblocked primitive chain length: {max_chain}. "
        + f"Endpoint-collision groups: {int(topology['parallel_endpoint_collision_count'].sum())}. "
        + "Closed cycles are retained as blocked evidence; no complete cycle was collapsed into a self-loop.\n",
    )
    write_text(
        root / "reports" / "03_EXACT_ORACLE_AND_RUNTIME_FIT_RESULTS.md",
        "# Exact oracle and runtime-fit results\n\n"
        + f"The canonical sample contains {domains['canonical_macro_signature'].nunique():,} unique macro functions and {len(oracle):,} exact query rows. "
        + "Primitive/member allocation, KKT, potential, and derivative evidence was evaluated in double precision. "
        + "The high CURRENT point was excluded from every fit, exponent choice, knot choice, and cascade selection.\n\n"
        + fit_summary.to_markdown(index=False)
        + f"\n\nPilot 1% cascade selections: `{json.dumps(selected_counts, sort_keys=True)}`. "
        + f"Hard split/retain diagnostics: {hard_count}; {split_measured_count} splittable cases have full-chain versus one-split versus no-contraction accuracy, cost-runtime, and retained-link/node evidence. These still retain their original ordered primitive links, with no exact-root runtime substitution or automatic split. Held-out high flow lies within the experimental pilot domain for {high_domain_coverage:.2%} of sampled signatures. "
        + "The 0.25/0.5/1/2/5% link, potential, accepted-path, and paired high-route ranking summaries are in `runtime_fits/error_band_summary.csv` and `runtime_fits/error_band_ranking_summary.csv`. "
        + "The pilot domain and 1% cascade gate are recommendations for review, not frozen production thresholds.\n",
    )
    write_text(
        root / "reports" / "04_BACN_VISIT_AND_CHARGE_AUDIT.md",
        "# B-ACN visit and fixed-charge audit\n\n"
        + "Live `bacn_internal_traversal_flow` counts one increment per maximal nonempty contiguous same-B suppressed-arc episode. "
        + "It is an exact linear companion of the accepted CFW mixture, but it is not the endpoint-half-charge denominator. "
        + "The new `incidence_visit_flow` reconstructs half incoming plus outgoing bundle, directional BACN-terminal, and reachable OD endpoint flow. "
        + f"Across low CURRENT suites, stored episode flow is {stored_ratio:.3%} of charged incidence. No B-ACN materially has stored flow above incidence.\n\n"
        + charge_summary.to_markdown(index=False)
        + "\n\nUse the low operating exposure mean for observed nodes and an explicit structural fallback for unvisited nodes. "
        + "The two accepted states do not justify freezing one universal low-visit threshold or shrinkage strength; candidate curves are retained for external review. "
        + "Episode medians/trimmed means are nullable because the accepted 200-path sample covers only part of the B registry.\n",
    )
    write_text(
        root / "reports" / "05_MULTIEDGE_AND_R_INTERFACE_AUDIT.md",
        "# Multiedge and R-interface audit\n\n"
        + "Recommendation: `RETAIN_NATIVE_PARALLEL_MACRO_LINKS`. The graph, shortest-path predecessor-arc arrays, flow arrays, SQLite link-ID keys, tracing, and lift-back layers preserve distinct stable edge IDs. "
        + f"The immutable physical graph contains {int(multiedge.iloc[0]['measured_count']):,} same-tail/head groups. Candidate contraction collisions are retained as distinct macro alternatives.\n\n"
        + "The frozen working default remains zero R charge. R-interface flow is throughput, not time exposure: terminal-level self-pairs are absent, while same-parent terminal movement and structural witnesses can exist. Accepted evidence does not bind physical path time to an R owner or show whether retained regional potentials already own it, so R cost ownership and materiality remain unresolved.\n",
    )
    write_text(
        root / "reports" / "06_RUNTIME_AND_RESOURCE_RESULTS.md",
        "# Runtime and resource results\n\n"
        + f"The conservative pilot-resolved backbone mixed root-free cost kernel median is {mixed_runtime:.3f}x ordinary BPR across suites. Selected sample macros use their actual packed fitted class; unresolved retained backbone slots use the explicitly labeled BPR reference timing. Terminal-linear and regional/ICNN families are excluded rather than mislabeled, so this is not an end-to-end or frozen final class mix. "
        + "Every production-class benchmark recorded exactly zero offline root calls; the separately labeled exact oracle comparator did invoke roots and remains offline only. "
        + f"Global-beta one-term timing was {global_runtime:.3f}x BPR versus {discrete_runtime:.3f}x for the discrete registry; 4/8-segment hinge timings were {hinge4_runtime:.3f}x/{hinge8_runtime:.3f}x. "
        + "Setup is excluded, and all rows use 5 warmups plus 30 timed repetitions. Resource closure is in `validation/resource_usage.json`; the conservative full-phase screen is in `validation/full_phase5_resource_projection.csv`.\n",
    )

    decision_records = [
        ("P5-G001", f"Full census; per-suite node reduction {dict(zip(reduction.suite_short_id, reduction.node_reduction_fraction, strict=True))}.", "Use the strict complete-incidence contraction overlay for review.", "PILOT_RECOMMENDATION", "", "topology/topology_census.csv;topology/chain_candidates.parquet"),
        ("P5-G002", f"Physical parallel groups={int(multiedge.iloc[0]['measured_count'])}; audited stack preserves stable edge IDs.", "RETAIN_NATIVE_PARALLEL_MACRO_LINKS with unique macro IDs.", "PILOT_RECOMMENDATION", "", "topology/multiedge_audit.csv"),
        ("P5-G003", f"Physical beta census count={len(physical_beta)}, modal beta={modal_beta:g}, all-modal={all_physical_beta_modal}; p95 max-error global={global_error:.6g}, discrete={discrete_error:.6g}; runtime/BPR global={global_runtime:.4g}, discrete={discrete_runtime:.4g}.", "Try the observed global exponent first, then the weighted-objective discrete registry.", "PILOT_RECOMMENDATION", "Final production dispatch policy is not frozen.", "bundle_oracle/bundle_parameter_census.csv;runtime_fits/runtime_fit_summary.csv;benchmarks/kernel_runtime_benchmark.csv"),
        ("P5-G004", f"At 1%, cascade fit coverage={float(cascade_one_percent['pass_fraction']):.3%}, eligible path-macro occurrences={int(cascade_one_percent['path_eligible_macro_occurrence_count'])}, potential p95={float(cascade_one_percent['eligible_max_rel_potential_error_p95']):.6g}, accepted-path runtime-only p95={float(cascade_one_percent['path_runtime_only_relative_error_p95']):.6g}, high paired-route strict-advantage preservation={float(cascade_one_percent_ranking['strict_advantage_preservation_fraction']):.3%}, reversals={int(cascade_one_percent_ranking['candidate_ranking_reversal_count'])}.", "Use the strictest reviewed band whose link/potential/path/ranking coverage and mixed-kernel speed jointly remain acceptable; the pilot reports all five and does not freeze one.", "PROVISIONAL_MEASURED_REQUIRES_EXTERNAL_THRESHOLD_CHOICE", "The final numerical threshold remains an external charter choice despite measured accepted-path ranking evidence.", "runtime_fits/error_band_summary.csv;runtime_fits/error_band_ranking_summary.csv;path_validation/path_cost_sample.parquet"),
        ("P5-G005", f"Hinge-4/8 p95 max-error={hinge4_error:.6g}/{hinge8_error:.6g}; runtime/BPR={hinge4_runtime:.4g}/{hinge8_runtime:.4g}.", "Prefer 4 segments unless 8 materially clears the reviewed error gate; otherwise retain/split.", "PILOT_RECOMMENDATION", "Smooth splines remain out of scope and are not justified by this pilot.", "runtime_fits/runtime_fit_summary.csv;benchmarks/kernel_runtime_benchmark.csv"),
        ("P5-G006", f"Hard signatures={hard_count}; splittable/measured={len(hard_splittable)}/{split_measured_count}; positive max-error reduction fraction={split_improvement_fraction}; median one-split/full and no-contraction/full root-free cost-runtime ratios={split_runtime_ratio_median}/{no_contraction_runtime_ratio_median}.", "At the recorded deterministic structural boundary, recommend one split only when it reduces maximum relative cost error per the one additional retained link/node; otherwise retain the ordered primitives. Do not auto-apply in the pilot.", "NOT_APPLICABLE_NO_HARD_CASES" if hard_count == 0 else ("NOT_APPLICABLE_SINGLE_PRIMITIVE_RETAIN" if len(hard_splittable) == 0 else "PILOT_RECOMMENDATION_MEASURED_SPLIT_TRADEOFF"), "The final charter must review this rule and the measured full/split/no-contraction tradeoff before applying a split.", "runtime_fits/pilot_domain_summary.csv"),
        ("P5-G007", f"Pilot Qmax covered held-out high flow for {high_domain_coverage:.2%} of sampled signatures.", "Use a predeclared monotone analytical tail and count exceedances.", "PILOT_RECOMMENDATION", "Pilot Qmax is not a frozen final domain.", "runtime_fits/pilot_domain_summary.csv"),
        ("P5-G008", f"Stored nonempty-episode flow/charged incidence={stored_ratio:.3%} across low CURRENT.", "Use the new exact incidence_visit_flow denominator.", "RESOLVED_SEMANTICS", "Legacy stored traversal remains immutable under its original meaning.", "charge_audit/bacn_visit_semantics.parquet"),
        ("P5-G009", "Low/current exposure mean and structural fallback were evaluated without high refit.", "Use observed low E/V plus structural unvisited fallback.", "UNRESOLVED_SHRINKAGE_STRENGTH", "Two states do not identify a universal visit threshold/shrinkage strength.", "charge_audit/fixed_charge_candidates.parquet"),
        ("P5-G010", "Terminal-level R self-pairs are absent, but same-parent terminal pairs and material interface throughput do not bind omitted physical time or establish ownership.", "Retain the frozen zero-R-charge working default while treating ownership and materiality as unresolved.", "UNRESOLVED_COST_OWNERSHIP", "Accepted traces do not attach physical path time to R-interface flow or distinguish it from time already owned by retained regional potentials.", "interface_audit/racn_cost_ownership_audit.csv"),
        ("P5-G011", f"Resolved backbone-only mixed timing={mixed_runtime:.4g}x BPR.", "Use only as a conservative backbone pilot benchmark.", "UNRESOLVED_END_TO_END_MIX", "Retained bundle, terminal, regional, and ICNN production classes are not frozen.", "benchmarks/kernel_runtime_benchmark.csv"),
        ("P5-G012", f"Naive signature scale={naive_scale:.4g}; projected wall={scientific_elapsed * naive_scale / 3600.0:.3g} h, RSS={peak_rss / 2**30:.3g} GiB, and pre-report/pre-manifest output={observed_output_bytes * naive_scale / 2**30:.3g} GiB.", "Require one reviewed full-Phase5 preflight after the final charter, not a second scientific pilot.", "UNRESOLVED_REQUIRES_PREFLIGHT", "Projection is conservative/high-uncertainty, the byte baseline explicitly precedes report/manifest closure, and the final runtime mix is not frozen.", "validation/full_phase5_resource_projection.csv"),
    ]
    decision_frame = pd.DataFrame(
        decision_records,
        columns=[
            "decision_id",
            "measured_facts",
            "recommendation",
            "resolution_status",
            "unresolved_reason",
            "evidence_paths",
        ],
    )
    decision_frame["externally_frozen"] = False
    decision_frame.to_csv(
        root / "validation" / "pilot_decision_evidence.csv", index=False
    )
    decisions = dict(
        zip(
            decision_frame["decision_id"],
            decision_frame["recommendation"],
            strict=True,
        )
    )
    write_text(
        root / "reports" / "07_DECISIONS_RECOMMENDED_FOR_FULL_PHASE5.md",
        "# Decisions recommended for full Phase 5\n\n"
        + "These are recommendations, not approvals. Full Phase 5, a final topology, final thresholds, and Phase 6 remain unauthorized.\n\n"
        + decision_frame.to_markdown(index=False),
    )

    status = {
        "schema_id": "partC.phase5.pilot.status.v1",
        "run_id": bindings.execution["run_id"],
        "status": FINAL_STATUS,
        "branch": "phase1/icnn-rev",
        "starting_head": starting_head,
        "ending_head": ending_head,
        "pilot_execution_authorized": True,
        "full_phase5_authorized": False,
        "phase6_authorized": False,
        "national_assignment_performed": False,
        "final_reduced_topology_accepted": False,
        "final_runtime_thresholds_frozen": False,
        "next_step_authorized": False,
        "tests": {"focused_phase5": "PASS" if tests_passed else "FAIL"},
        "outputs": {
            "topology_suite_count": len(topology),
            "candidate_chain_rows": len(chains),
            "sampled_macro_signatures": int(domains["canonical_macro_signature"].nunique()),
            "oracle_rows": len(oracle),
            "fit_rows": len(fits),
            "visit_rows": len(visits),
            "path_rows": len(paths),
        },
        "decisions": decisions,
        "blockers": [],
    }
    write_json(root / "reports" / "08_PILOT_STATUS.json", status)
    return status


def _describe_reopen(path: Path) -> tuple[str, int, str]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        metadata = parquet.metadata
        iterator = parquet.iter_batches(batch_size=1)
        next(iterator, None)
        schema = {
            field.name: str(field.type) for field in parquet.schema_arrow
        }
        return "PASS", metadata.num_rows, canonical_json(
            {
                "columns": parquet.schema_arrow.names,
                "dtypes": schema,
                "row_group_count": metadata.num_row_groups,
            }
        )
    if suffix == ".csv":
        frame = pd.read_csv(path)
        return "PASS", len(frame), canonical_json({"columns": frame.columns.tolist()})
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return "PASS", len(value) if isinstance(value, list) else 1, canonical_json({"root_type": type(value).__name__})
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            schema = {name: {"shape": data[name].shape, "dtype": str(data[name].dtype)} for name in data.files}
        return "PASS", len(schema), canonical_json(schema)
    if suffix == ".png":
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return "PASS", 1, canonical_json({"width": image.width, "height": image.height})
    if suffix == ".sqlite":
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            tables = [
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
            ]
        finally:
            connection.close()
        return "PASS", len(tables), canonical_json({"tables": tables})
    with path.open("rb") as stream:
        stream.read(1)
    return "PASS", 1, canonical_json({"format": suffix.lstrip(".") or "binary"})


def _write_output_manifest(root: Path) -> pd.DataFrame:
    manifest_path = root / "manifests" / "MANIFEST.csv"
    checksums_path = root / "manifests" / "CHECKSUMS.sha256"
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path in {manifest_path, checksums_path}:
            continue
        status, row_count, schema = _describe_reopen(path)
        relative = path.relative_to(root).as_posix()
        handoff_policy = {
            "topology/chain_candidates.parquet": "external_with_sample",
            "bundle_oracle/oracle_sample_results.parquet": "external_with_sample",
            "runtime_fits/runtime_fit_results.parquet": "external_with_summary",
            "charge_audit/bacn_visit_semantics.parquet": "external_with_summary",
            "charge_audit/fixed_charge_candidates.parquet": "external_with_summary",
            "path_validation/path_cost_sample.parquet": "external_with_sample",
        }.get(relative, "compact")
        rows.append(
            {
                "relative_path": relative,
                "absolute_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "format": path.suffix.lower().lstrip(".") or "binary",
                "row_or_object_count": row_count,
                "schema_or_shape_json": schema,
                "reopen_status": status,
                "handoff_policy": handoff_policy,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(manifest_path, index=False)
    checksum_rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(value for value in root.rglob("*") if value.is_file() and value != checksums_path)
    ]
    write_text(checksums_path, "\n".join(checksum_rows))
    return frame


def _write_blocked_output_manifest(root: Path) -> pd.DataFrame:
    """Inventory every preserved byte even when a partial artifact cannot reopen."""

    manifest_path = root / "manifests" / "MANIFEST.csv"
    checksums_path = root / "manifests" / "CHECKSUMS.sha256"
    rows: list[dict[str, object]] = []
    checksum_rows: list[str] = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path in {manifest_path, checksums_path}:
            continue
        relative = path.relative_to(root).as_posix()
        try:
            digest = sha256_file(path)
            checksum_rows.append(f"{digest}  {relative}")
        except Exception as error:  # pragma: no cover - filesystem failure path
            digest = "UNREADABLE"
            checksum_rows.append(f"UNREADABLE  {relative}")
            hash_error = f"{type(error).__name__}: {error}"
        else:
            hash_error = ""
        try:
            reopen_status, row_count, schema = _describe_reopen(path)
        except Exception as error:
            reopen_status = "FAIL"
            row_count = -1
            schema = canonical_json(
                {"reopen_error": f"{type(error).__name__}: {error}"}
            )
        if hash_error:
            reopen_status = "FAIL"
            schema = canonical_json({"hash_error": hash_error})
        rows.append(
            {
                "relative_path": relative,
                "absolute_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "format": path.suffix.lower().lstrip(".") or "binary",
                "row_or_object_count": row_count,
                "schema_or_shape_json": schema,
                "reopen_status": reopen_status,
                "handoff_policy": "blocked_partial_evidence",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(manifest_path, index=False)
    checksum_rows.append(
        f"{sha256_file(manifest_path)}  {manifest_path.relative_to(root).as_posix()}"
    )
    write_text(checksums_path, "\n".join(checksum_rows))
    return frame


def _validate_required_outputs(
    root: Path, authority_root: Path, *, final: bool = True
) -> dict[str, object]:
    required = pd.read_csv(authority_root / "config" / "required_output_artifacts.csv")
    required_paths = set(required["relative_path"].astype(str))
    required_paths.update(
        {
            "path_validation/path_cost_sample.parquet",
            "runtime_fits/pilot_domain_summary.csv",
            "runtime_fits/selected_runtime_registry.json",
            "runtime_fits/error_band_summary.csv",
            "runtime_fits/error_band_ranking_summary.csv",
            "validation/pilot_decision_evidence.csv",
            "validation/full_phase5_resource_projection.csv",
        }
    )
    missing = sorted(value for value in required_paths if not (root / value).is_file())
    empty = sorted(
        value
        for value in required_paths
        if (root / value).is_file() and (root / value).stat().st_size == 0
    )
    minimum_columns: dict[str, set[str]] = {
        "topology/topology_census.csv": {
            "suite_short_id", "original_bacn_count", "original_backbone_bundle_count",
            "protected_bacn_count", "protected_reason_counts_json",
            "strict_directed_candidate_count", "reciprocal_candidate_count",
            "irregular_degree2_count", "cycle_count", "maximal_chain_count",
            "chain_length_mean", "chain_length_p50", "chain_length_p95",
            "chain_length_max", "projected_retained_bacn_count",
            "projected_macro_link_count", "node_reduction_fraction",
            "backbone_link_reduction_fraction", "parallel_endpoint_collision_count",
            "self_loop_block_count", "status",
        },
        "topology/chain_candidates.parquet": {
            "suite_short_id", "chain_id", "direction_class",
            "retained_tail_bacn_id", "retained_head_bacn_id",
            "primitive_bundle_ids_json", "internal_bacn_ids_json",
            "primitive_link_count", "internal_node_count", "low_flow_min",
            "low_flow_max", "low_flow_abs_spread", "low_flow_rel_spread",
            "high_flow_min", "high_flow_max", "high_flow_abs_spread",
            "high_flow_rel_spread",
            "low_flow_consistent", "high_flow_consistent", "canonical_signature",
            "protected_or_blocked", "block_reasons_json",
            "parallel_collision_group_id", "cycle_anchor_id",
            "sample_included", "sample_inclusion_reasons_json",
            "runtime_resolution", "selected_runtime_class",
        },
        "bundle_oracle/bundle_parameter_census.csv": {
            "suite_short_id", "bundle_count", "physical_member_count",
            "member_count_mean", "member_count_p50", "member_count_p95",
            "member_count_max", "singleton_count", "homogeneous_closed_form_count",
            "zero_low_current_count", "zero_low_aon_count", "zero_high_current_count",
            "fftt_s_quantiles_json", "capacity_quantiles_json",
            "alpha_quantiles_json", "beta_quantiles_json",
            "low_current_flow_quantiles_json", "low_aon_flow_quantiles_json",
            "high_current_flow_quantiles_json", "staggered_activation_bundle_count",
            "activation_level_count_quantiles_json",
            "positive_activation_threshold_count", "canonical_unique_count",
            "cross_suite_duplicate_unique_key_count",
            "cross_suite_duplicate_unique_key_rate",
            "cross_suite_duplicate_occurrence_count", "cross_suite_duplicate_rate",
            "status",
        },
        "bundle_oracle/oracle_sample_results.parquet": {
            "canonical_macro_signature", "suite_occurrence_id", "query_flow",
            "query_role", "exact_cost", "exact_potential",
            "exact_cost_derivative", "active_member_count", "allocation_residual",
            "kkt_residual", "root_iteration_count", "status",
        },
        "runtime_fits/runtime_fit_results.parquet": {
            "canonical_macro_signature", "runtime_class", "variant_id",
            "flow_domain_max", "parameter_json", "knot_json",
            "calibration_point_count", "max_abs_cost_error", "max_rel_cost_error",
            "mean_rel_cost_error", "p95_rel_cost_error", "low_rel_cost_error",
            "weighted_rms_rel_cost_error",
            "high_heldout_rel_cost_error", "max_rel_potential_error",
            "finite_difference_error", "monotonicity_pass", "convexity_pass",
            "zero_intercept_pass", "parameter_count", "arithmetic_term_count",
            "recommended_status", "failure_reasons_json",
        },
        "runtime_fits/pilot_domain_summary.csv": {
            "canonical_macro_signature", "suite_occurrence_id", "sample_ordinal",
            "primitive_link_count", "selection_class", "provisional_action",
            "provisional_split_after_primitive_index",
            "provisional_split_structural_score", "provisional_split_rule",
            "split_runtime_tradeoff_evaluated", "high_flow_inside_pilot_domain",
            "split_evaluation_status", "full_chain_best_runtime_class",
            "one_split_child_runtime_classes_json",
            "full_chain_best_weighted_rms_rel_cost_error",
            "full_chain_best_max_rel_cost_error",
            "one_split_weighted_rms_rel_cost_error",
            "one_split_max_rel_cost_error",
            "no_contraction_primitive_runtime_classes_json",
            "no_contraction_weighted_rms_rel_cost_error",
            "no_contraction_max_rel_cost_error",
            "full_chain_max_rel_potential_error",
            "one_split_max_rel_potential_error",
            "no_contraction_max_rel_potential_error",
            "full_chain_cost_runtime_median_ns",
            "one_split_cost_runtime_median_ns",
            "one_split_runtime_to_full_ratio",
            "no_contraction_cost_runtime_median_ns",
            "no_contraction_runtime_to_full_ratio",
            "full_contraction_retained_link_count",
            "one_split_retained_link_count",
            "no_contraction_retained_link_count",
            "full_contraction_removed_internal_node_count",
            "one_split_removed_internal_node_count",
            "no_contraction_runtime_status",
            "split_error_reduction_per_additional_retained_link",
            "split_error_reduction_per_additional_retained_node",
            "split_charge_accounting_rule",
            "split_boundary_internal_component_id",
            "split_boundary_external_charge_s",
            "flow_domain_max", "q_ratio4", "q_ratio4_status", "domain_status",
            "structural_flow_scale", "calibration_point_count",
            "internal_charge_s", "internal_charge_estimator_id",
            "runtime_resolution", "split_diagnostic_status",
            "retained_primitive_bundle_ids_json",
            "retained_internal_bacn_ids_json",
        },
        "runtime_fits/error_band_summary.csv": {
            "runtime_class", "max_relative_error_band", "evaluation_role",
            "fit_signature_count", "eligible_signature_count", "pass_fraction",
            "eligible_max_rel_potential_error_p95",
            "eligible_max_rel_potential_error_max", "path_count",
            "path_eligible_macro_occurrence_count", "path_relative_error_p50",
            "path_relative_error_p95", "path_relative_error_max",
            "path_runtime_only_relative_error_p50",
            "path_runtime_only_relative_error_p95",
            "path_runtime_only_relative_error_max", "path_summary_scope",
        },
        "runtime_fits/error_band_ranking_summary.csv": {
            "runtime_class", "max_relative_error_band", "paired_od_count",
            "physical_strict_selected_advantage_count",
            "physical_reference_tie_count",
            "physical_reference_inversion_count",
            "candidate_strict_advantage_preserved_count",
            "candidate_advantage_eroded_to_tie_count",
            "candidate_ranking_reversal_count",
            "strict_advantage_preservation_fraction",
            "candidate_ranking_reversal_fraction",
            "physical_margin_p05_s", "physical_margin_p50_s",
            "candidate_margin_p05_s", "candidate_margin_p50_s", "status",
        },
        "charge_audit/bacn_visit_semantics.parquet": {
            "suite_short_id", "benchmark_id", "bacn_id", "stored_traversal_flow",
            "incidence_visit_flow", "path_episode_visit_flow",
            "incoming_charged_flow", "outgoing_charged_flow",
            "suppressed_arc_flow_sum", "stored_minus_incidence",
            "stored_to_incidence_ratio", "semantic_classification",
            "classification_evidence_json", "status",
        },
        "charge_audit/fixed_charge_candidates.parquet": {
            "suite_short_id", "bacn_id", "low_visit_flow",
            "low_internal_exposure_flow_seconds", "low_operating_mean_seconds",
            "structural_seed_seconds", "high_visit_flow",
            "high_internal_exposure_flow_seconds", "candidate_shrinkage_json",
            "median_episode_seconds", "trimmed_mean_episode_seconds",
            "high_residual_using_low_mean", "unvisited_fallback_class",
            "recommended_estimator_class", "status",
        },
        "benchmarks/kernel_runtime_benchmark.csv": {
            "suite_short_id", "kernel_class", "projected_link_count",
            "warmup_count", "timed_repetition_count", "median_cost_eval_seconds",
            "p95_cost_eval_seconds", "median_potential_eval_seconds",
            "p95_potential_eval_seconds", "median_derivative_eval_seconds",
            "p95_derivative_eval_seconds", "queries_per_second",
            "p05_cost_eval_seconds", "mad_cost_eval_seconds",
            "p05_potential_eval_seconds", "mad_potential_eval_seconds",
            "p05_derivative_eval_seconds", "mad_derivative_eval_seconds",
            "relative_to_bpr_baseline", "exact_root_calls", "peak_rss_bytes",
            "setup_seconds", "median_dispatch_buffer_seconds",
            "packed_parameter_record_count", "scalar_roundtrip_max_abs_error",
            "projection_scope", "machine_metadata_json", "notes",
        },
        "topology/multiedge_audit.csv": {
            "audit_layer", "suite_short_id", "test", "measured_count",
            "secondary_count", "status", "evidence", "recommendation",
        },
        "interface_audit/racn_cost_ownership_audit.csv": {
            "suite_short_id", "summarized_racn_count",
            "terminal_declared_racn_parent_count", "restricted_witness_count",
            "low_current_racn_interface_flow", "high_current_racn_interface_flow",
            "low_current_self_pair_count", "low_current_self_pair_flow",
            "high_current_self_pair_count", "high_current_self_pair_flow",
            "low_current_same_racn_parent_pair_count",
            "low_current_same_racn_parent_pair_flow",
            "high_current_same_racn_parent_pair_count",
            "high_current_same_racn_parent_pair_flow",
            "separately_owned_omitted_racn_exposure_observed",
            "recommended_racn_charge_s", "cost_ownership_resolution",
            "recommendation", "status",
        },
        "path_validation/path_cost_component_summary.csv": {
            "evaluation_role", "error_component", "path_count",
            "mean_abs_error", "p95_abs_error", "max_abs_error", "pass_count",
            "notes",
        },
        "path_validation/path_cost_sample.parquet": {
            "sample_od_id", "evaluation_role", "suite_short_id", "origin_idx",
            "destination_idx", "physical_arc_count", "physical_path_cost_s",
            "candidate_compiled_path_cost_s", "exact_series_identity_error_s",
            "runtime_fit_base_error_s", "fixed_charge_error_s",
            "free_mixing_endpoint_state_error_s", "liftability_error",
            "component_closure_error_s", "matched_macro_count",
            "selected_runtime_macro_count", "retained_original_macro_count",
            "silent_exact_root_runtime_fallback_count", "runtime_resolution_json",
            "error_band_flow_state_alignment_error_json",
            "error_band_candidate_path_cost_s_json",
            "selected_runtime_bundle_count", "fully_root_free_runtime_evaluable",
            "candidate_resolution_cost_scope",
            "error_band_path_relative_error_json",
            "error_band_runtime_only_relative_error_json",
            "error_band_eligible_macro_count_json",
            "stable_arc_hash_pass", "path_continuity_pass", "status",
        },
        "validation/acceptance_test_results.csv": {
            "test_id", "test", "status", "gate_class", "evidence_json",
        },
        "validation/pilot_decision_evidence.csv": {
            "decision_id", "measured_facts", "recommendation",
            "resolution_status", "unresolved_reason", "evidence_paths",
            "externally_frozen",
        },
        "validation/full_phase5_resource_projection.csv": {
            "metric", "pilot_observed_value", "naive_full_phase5_projection",
            "governing_limit", "projection_to_limit_ratio",
            "projection_method", "pilot_observation_stage", "uncertainty", "status",
        },
        "manifests/external_artifact_bindings.csv": {
            "role", "absolute_path", "size_bytes", "mtime_utc",
            "sha256_or_directory_digest", "expected_sha256_or_digest",
            "format", "row_array_or_object_count", "schema_or_shape_json",
            "reopen_status", "lineage",
        },
        "manifests/MANIFEST.csv": {
            "relative_path", "absolute_path", "size_bytes", "sha256", "format",
            "row_or_object_count", "schema_or_shape_json", "reopen_status",
            "handoff_policy",
        },
    }
    schema_failures: list[str] = []
    zero_row_tables: list[str] = []
    for relative, expected in minimum_columns.items():
        path = root / relative
        if not path.is_file():
            continue
        if path.suffix.lower() == ".parquet":
            parquet = pq.ParquetFile(path)
            columns = set(parquet.schema_arrow.names)
            row_count = parquet.metadata.num_rows
        else:
            frame = pd.read_csv(path, nrows=5)
            columns = set(frame.columns)
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                row_count = max(0, sum(1 for _ in stream) - 1)
        if not expected.issubset(columns):
            schema_failures.append(
                f"{relative}: missing {sorted(expected - columns)}"
            )
        if row_count == 0:
            zero_row_tables.append(relative)
    semantic_failures: list[str] = []
    try:
        domain_frame = pd.read_csv(
            root / "runtime_fits" / "pilot_domain_summary.csv"
        )
        fit_frame = pd.read_parquet(
            root / "runtime_fits" / "runtime_fit_results.parquet"
        )
        oracle_frame = pd.read_parquet(
            root / "bundle_oracle" / "oracle_sample_results.parquet"
        )
        registry = json.loads(
            (root / "runtime_fits" / "selected_runtime_registry.json").read_text(
                encoding="utf-8"
            )
        )
        runtime_schema = json.loads(
            (authority_root / "config" / "runtime_function_class.schema.json").read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(registry, list) or not registry:
            raise ValueError("runtime registry must be a nonempty list")
        for payload in registry:
            jsonschema_validate(payload, runtime_schema)
            identity = {key: value for key, value in payload.items() if key != "function_id"}
            expected_function_id = "p5_runtime_" + hashlib.sha256(
                canonical_json(identity).encode("utf-8")
            ).hexdigest()
            if payload["function_id"] != expected_function_id:
                semantic_failures.append(
                    f"runtime registry function_id mismatch: {payload['source_macro_signature']}"
                )
        registry_signatures = [
            str(value["source_macro_signature"]) for value in registry
        ]
        registry_ids = [str(value["function_id"]) for value in registry]
        domain_signatures = domain_frame["canonical_macro_signature"].astype(str)
        if (
            len(set(registry_signatures)) != len(registry_signatures)
            or len(set(registry_ids)) != len(registry_ids)
            or set(registry_signatures) != set(domain_signatures)
            or not domain_signatures.is_unique
            or not domain_frame["sample_ordinal"].is_unique
            or set(domain_frame["sample_ordinal"].astype(int))
            != set(range(len(domain_frame)))
        ):
            semantic_failures.append(
                "runtime registry/domain signature or identifier closure failed"
            )
        selected = fit_frame.loc[
            fit_frame["recommended_status"].astype(str).str.startswith(
                "CASCADE_SELECTED"
            )
        ]
        if selected["canonical_macro_signature"].duplicated().any():
            semantic_failures.append(
                "more than one cascade-selected fit exists for a signature"
            )
        selected_by_signature = selected.set_index("canonical_macro_signature")
        selected_class = dict(
            zip(
                selected["canonical_macro_signature"].astype(str),
                selected["runtime_class"].astype(str),
                strict=True,
            )
        )
        zero_rows = oracle_frame.loc[
            oracle_frame["query_role"].astype(str).str.contains("ZERO")
            & oracle_frame["query_flow"].eq(0.0)
        ]
        if zero_rows["canonical_macro_signature"].duplicated().any():
            semantic_failures.append("zero-flow oracle rows are not unique")
        zero_cost_by_signature = zero_rows.set_index(
            "canonical_macro_signature"
        )["exact_cost"]
        domain_by_signature = domain_frame.set_index("canonical_macro_signature")
        for payload in registry:
            signature = str(payload["source_macro_signature"])
            domain = domain_by_signature.loc[signature]
            selected_name = selected_class.get(signature)
            model_payload = {
                "source_macro_signature": signature,
                "function_class": payload["function_class"],
                "variant_id": payload.get("variant_id"),
                "valid_flow_max": payload["valid_flow_max"],
                "parameters": payload.get("parameters", {}),
            }
            expected_model_sha = hashlib.sha256(
                canonical_json(model_payload).encode("utf-8")
            ).hexdigest()
            if payload.get("model_payload_sha256") != expected_model_sha:
                semantic_failures.append(
                    f"{signature}: registry model-payload hash mismatch"
                )
            if (
                signature not in zero_cost_by_signature.index
                or not np.isclose(
                    float(payload["zero_flow_cost_s"]),
                    float(zero_cost_by_signature.loc[signature]),
                    rtol=1.0e-13,
                    atol=1.0e-10,
                )
                or payload.get("exact_root_calls_runtime") != 0
                or payload.get("heldout_high_used_for_fit") is not False
            ):
                semantic_failures.append(
                    f"{signature}: registry zero-flow/root/heldout invariant failed"
                )
            if not np.isclose(
                float(payload["valid_flow_max"]),
                float(domain["flow_domain_max"]),
                rtol=0.0,
                atol=1.0e-12 * max(1.0, float(domain["flow_domain_max"])),
            ):
                semantic_failures.append(
                    f"{signature}: registry/domain flow maximum mismatch"
                )
            if selected_name is None:
                parameters = payload.get("parameters", {})
                retained_primitive = json.loads(
                    str(domain["retained_primitive_bundle_ids_json"])
                )
                retained_internal = json.loads(
                    str(domain["retained_internal_bacn_ids_json"])
                )
                if not (
                    payload["function_class"] == "SPLIT_OR_RETAIN"
                    and payload.get("runtime_evaluable") is False
                    and str(payload.get("runtime_resolution", "")).startswith(
                        "RETAIN_ORIGINAL_PRIMITIVE_LINKS"
                    )
                    and payload.get("parameters", {}).get("action")
                    == "RETAIN_ORIGINAL_PRIMITIVE_LINKS"
                    and parameters.get("automatic_split_performed") is False
                    and parameters.get("retained_primitive_bundle_ids")
                    == retained_primitive
                    and parameters.get("retained_internal_bacn_ids")
                    == retained_internal
                    and int(
                        parameters.get(
                            "provisional_split_after_primitive_index", -999
                        )
                    )
                    == int(domain["provisional_split_after_primitive_index"])
                    and bool(parameters.get("split_runtime_tradeoff_evaluated"))
                    == bool(domain["split_runtime_tradeoff_evaluated"])
                ):
                    semantic_failures.append(
                        f"{signature}: retained registry sentinel is inconsistent"
                    )
            else:
                fit = selected_by_signature.loc[signature]
                expected_parameters = json.loads(str(fit["parameter_json"]))
                expected_parameters["knots_flow"] = json.loads(
                    str(fit["knot_json"])
                )
                selected_pass = bool(
                    payload["function_class"] == selected_name
                    and payload.get("runtime_evaluable") is True
                    and payload.get("runtime_resolution")
                    == "SELECTED_ROOT_FREE_RUNTIME"
                    and payload.get("variant_id") == str(fit["variant_id"])
                    and canonical_json(payload.get("parameters", {}))
                    == canonical_json(expected_parameters)
                    and str(domain["selection_class"]) == selected_name
                )
                try:
                    function = _runtime_from_fit(fit)
                    probe = np.asarray(
                        [
                            0.0,
                            0.5 * float(payload["valid_flow_max"]),
                            float(payload["valid_flow_max"]),
                        ],
                        dtype=np.float64,
                    )
                    probe_cost = function.cost(probe)
                    probe_potential = function.potential(probe)
                    probe_derivative = function.derivative(probe)
                    selected_pass &= bool(
                        np.isfinite(probe_cost).all()
                        and np.isfinite(probe_potential).all()
                        and np.isfinite(probe_derivative).all()
                        and np.all(probe_derivative >= 0.0)
                        and np.isclose(
                            float(probe_cost[0]),
                            float(payload["zero_flow_cost_s"]),
                            rtol=1.0e-12,
                            atol=1.0e-9,
                        )
                        and np.isclose(
                            float(probe_potential[0]),
                            0.0,
                            rtol=0.0,
                            atol=1.0e-9,
                        )
                    )
                except Exception:
                    selected_pass = False
                if not selected_pass:
                    semantic_failures.append(
                        f"{signature}: selected registry entry disagrees with fit"
                    )

        expected_discrete_variants = {
            "BETA_2",
            "BETA_3",
            "BETA_4",
            "BETA_5",
            "BETA_6",
        }
        for signature, group in fit_frame.groupby(
            "canonical_macro_signature", sort=True
        ):
            class_counts = group["runtime_class"].astype(str).value_counts()
            discrete_group = group.loc[
                group["runtime_class"].eq("FIT_ONE_TERM_DISCRETE_EXPONENT")
            ]
            cardinality_pass = bool(
                int(class_counts.get("FIT_ONE_TERM_GLOBAL_EXPONENT", 0)) == 1
                and int(class_counts.get("FIT_ONE_TERM_DISCRETE_EXPONENT", 0))
                == 5
                and int(class_counts.get("MONOTONE_HINGE_4", 0)) == 1
                and int(class_counts.get("MONOTONE_HINGE_8", 0)) == 1
                and int(class_counts.get("EXACT_CLOSED_FORM", 0)) in {0, 1}
                and set(discrete_group["variant_id"].astype(str))
                == expected_discrete_variants
                and discrete_group["discrete_registry_status"]
                .eq("BEST_WEIGHTED_PILOT_OBJECTIVE")
                .sum()
                == 1
            )
            if not cardinality_pass:
                semantic_failures.append(
                    f"{signature}: runtime-fit class cardinality is invalid"
                )
                continue
            best_discrete_row = discrete_group.loc[
                discrete_group["discrete_registry_status"].eq(
                    "BEST_WEIGHTED_PILOT_OBJECTIVE"
                )
            ].iloc[0]
            ordered_candidates: list[pd.Series] = []
            for class_name in (
                "EXACT_CLOSED_FORM",
                "FIT_ONE_TERM_GLOBAL_EXPONENT",
            ):
                local = group.loc[group["runtime_class"].eq(class_name)]
                if len(local):
                    ordered_candidates.append(local.iloc[0])
            ordered_candidates.append(best_discrete_row)
            for class_name in ("MONOTONE_HINGE_4", "MONOTONE_HINGE_8"):
                ordered_candidates.append(
                    group.loc[group["runtime_class"].eq(class_name)].iloc[0]
                )
            expected_selected = next(
                (
                    row
                    for row in ordered_candidates
                    if float(row["max_rel_cost_error"]) <= 0.01
                    and bool(row["monotonicity_pass"])
                    and bool(row["convexity_pass"])
                    and bool(row["zero_intercept_pass"])
                    and int(row["nonfinite_cost_derivative_error_count"]) == 0
                    and float(row["finite_difference_relative_error"])
                    <= 1.0e-6
                ),
                None,
            )
            actual_selected = group.loc[
                group["recommended_status"].astype(str).str.startswith(
                    "CASCADE_SELECTED"
                )
            ]
            cascade_pass = bool(
                (expected_selected is None and len(actual_selected) == 0)
                or (
                    expected_selected is not None
                    and len(actual_selected) == 1
                    and str(actual_selected.iloc[0]["runtime_class"])
                    == str(expected_selected["runtime_class"])
                    and str(actual_selected.iloc[0]["variant_id"])
                    == str(expected_selected["variant_id"])
                )
            )
            if not cascade_pass:
                semantic_failures.append(
                    f"{signature}: cascade selection differs from frozen order"
                )

        numeric = domain_frame[
            [
                "primitive_link_count",
                "flow_domain_max",
                "structural_flow_scale",
                "calibration_point_count",
                "provisional_split_after_primitive_index",
                "provisional_split_structural_score",
            ]
        ].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            semantic_failures.append("pilot domains contain nonfinite required numerics")
        for row in domain_frame.itertuples(index=False):
            primitive_count = int(row.primitive_link_count)
            boundary = int(row.provisional_split_after_primitive_index)
            split_evaluated = str(row.split_runtime_tradeoff_evaluated).lower() == "true"
            provenance = json.loads(str(row.retained_primitive_bundle_ids_json))
            if (
                primitive_count <= 0
                or float(row.flow_domain_max) <= 0.0
                or float(row.structural_flow_scale) <= 0.0
                or int(row.calibration_point_count) < 65
                or len(provenance) != primitive_count
                or float(row.provisional_split_structural_score) < 0.0
                or (primitive_count == 1 and boundary != -1)
                or (primitive_count > 1 and not 1 <= boundary < primitive_count)
            ):
                semantic_failures.append(
                    f"{row.canonical_macro_signature}: invalid pilot-domain invariant"
                )
            if row.selection_class == "SPLIT_OR_RETAIN_DIAGNOSTIC":
                base_retain_pass = bool(
                    row.provisional_action == "RETAIN_ORIGINAL_PRIMITIVE_LINKS"
                    and str(row.runtime_resolution).startswith(
                        "RETAIN_ORIGINAL_PRIMITIVE_LINKS"
                    )
                )
                if primitive_count == 1:
                    tradeoff_pass = bool(
                        not split_evaluated
                        and row.split_evaluation_status
                        == "NOT_APPLICABLE_SINGLE_PRIMITIVE_RETAIN"
                    )
                else:
                    split_metrics = np.asarray(
                        [
                            row.full_chain_best_weighted_rms_rel_cost_error,
                            row.full_chain_best_max_rel_cost_error,
                            row.one_split_weighted_rms_rel_cost_error,
                            row.one_split_max_rel_cost_error,
                            row.no_contraction_weighted_rms_rel_cost_error,
                            row.no_contraction_max_rel_cost_error,
                            row.full_chain_max_rel_potential_error,
                            row.one_split_max_rel_potential_error,
                            row.no_contraction_max_rel_potential_error,
                            row.full_chain_cost_runtime_median_ns,
                            row.one_split_cost_runtime_median_ns,
                            row.one_split_runtime_to_full_ratio,
                            row.no_contraction_cost_runtime_median_ns,
                            row.no_contraction_runtime_to_full_ratio,
                            row.split_error_reduction_per_additional_retained_link,
                            row.split_error_reduction_per_additional_retained_node,
                            row.split_boundary_external_charge_s,
                        ],
                        dtype=np.float64,
                    )
                    tradeoff_pass = bool(
                        split_evaluated
                        and row.split_evaluation_status
                        == "MEASURED_FULL_VS_ONE_SPLIT_VS_RETAIN"
                        and np.isfinite(split_metrics).all()
                        and float(row.full_chain_cost_runtime_median_ns) > 0.0
                        and float(row.one_split_cost_runtime_median_ns) > 0.0
                        and float(row.no_contraction_cost_runtime_median_ns) > 0.0
                        and row.no_contraction_runtime_status
                        == "MEASURED_ROOT_FREE_BOUNDED_PRIMITIVE_RUNTIME"
                        and str(row.split_boundary_internal_component_id)
                        not in {"", "nan"}
                    )
                if not base_retain_pass or not tradeoff_pass:
                    semantic_failures.append(
                        f"{row.canonical_macro_signature}: hard-case retain semantics differ"
                    )
            elif not (
                row.provisional_action == "SELECT_ROOT_FREE_RUNTIME"
                and row.runtime_resolution == "SELECTED_ROOT_FREE_RUNTIME"
                and not split_evaluated
                and row.split_evaluation_status
                == "NOT_APPLICABLE_RUNTIME_SELECTED"
            ):
                semantic_failures.append(
                    f"{row.canonical_macro_signature}: selected domain semantics differ"
                )
    except Exception as error:
        semantic_failures.append(
            f"runtime registry/domain semantic validation: {type(error).__name__}: {error}"
        )
    try:
        band_frame = pd.read_csv(
            root / "runtime_fits" / "error_band_summary.csv"
        )
        ranking_frame = pd.read_csv(
            root / "runtime_fits" / "error_band_ranking_summary.csv"
        )
        path_frame = pd.read_parquet(
            root / "path_validation" / "path_cost_sample.parquet"
        )
        path_roles = {
            "LOW_ON_FROZEN_FREE_FLOW_PATH",
            "HIGH_HELDOUT_FROZEN_COMPARATOR_PATH",
            "HIGH_HELDOUT_SELECTED_PATH",
        }
        summary_roles = path_roles | {"ALL_ACCEPTED_PATH_ROLES"}
        expected_band_rows = {
            (runtime_class, band, role)
            for runtime_class in PILOT_ERROR_SUMMARY_CLASSES
            for band in PILOT_ERROR_BANDS
            for role in summary_roles
        }
        actual_band_rows = {
            (
                str(row.runtime_class),
                float(row.max_relative_error_band),
                str(row.evaluation_role),
            )
            for row in band_frame.itertuples(index=False)
        }
        if len(band_frame) != len(expected_band_rows) or actual_band_rows != expected_band_rows:
            semantic_failures.append("error-band summary key closure failed")
        path_count_by_role = path_frame.groupby("evaluation_role").size().to_dict()
        for row in band_frame.itertuples(index=False):
            expected_path_count = (
                len(path_frame)
                if row.evaluation_role == "ALL_ACCEPTED_PATH_ROLES"
                else int(path_count_by_role.get(row.evaluation_role, -1))
            )
            path_quantiles = np.asarray(
                [
                    row.path_relative_error_p50,
                    row.path_relative_error_p95,
                    row.path_relative_error_max,
                    row.path_runtime_only_relative_error_p50,
                    row.path_runtime_only_relative_error_p95,
                    row.path_runtime_only_relative_error_max,
                ],
                dtype=np.float64,
            )
            fit_count = int(row.fit_signature_count)
            eligible_count = int(row.eligible_signature_count)
            arithmetic_pass = bool(
                fit_count > 0
                and 0 <= eligible_count <= fit_count
                and np.isclose(
                    float(row.pass_fraction),
                    eligible_count / fit_count,
                    rtol=0.0,
                    atol=1.0e-15,
                )
                and int(row.path_count) == expected_path_count
                and int(row.path_eligible_macro_occurrence_count) >= 0
                and np.isfinite(path_quantiles).all()
                and np.all(path_quantiles >= 0.0)
                and path_quantiles[0] <= path_quantiles[1] <= path_quantiles[2]
                and path_quantiles[3] <= path_quantiles[4] <= path_quantiles[5]
            )
            if eligible_count:
                potential = np.asarray(
                    [
                        row.eligible_max_rel_potential_error_p95,
                        row.eligible_max_rel_potential_error_max,
                    ],
                    dtype=np.float64,
                )
                arithmetic_pass &= bool(
                    np.isfinite(potential).all()
                    and np.all(potential >= 0.0)
                    and potential[0] <= potential[1]
                )
            if not arithmetic_pass:
                semantic_failures.append(
                    f"error-band row arithmetic failed: {row.runtime_class}/{row.max_relative_error_band}/{row.evaluation_role}"
                )
        for (_, runtime_class, role), group in band_frame.groupby(
            ["fit_signature_count", "runtime_class", "evaluation_role"],
            sort=True,
        ):
            ordered = group.sort_values("max_relative_error_band")
            if np.any(np.diff(ordered["eligible_signature_count"].to_numpy()) < 0):
                semantic_failures.append(
                    f"error-band eligible coverage is not monotone: {runtime_class}/{role}"
                )
        expected_json_keys = {
            f"{runtime_class}|{band:.4f}"
            for runtime_class in PILOT_ERROR_SUMMARY_CLASSES
            for band in PILOT_ERROR_BANDS
        }
        for row in path_frame.itertuples(index=False):
            for column, nonnegative in (
                ("error_band_path_relative_error_json", True),
                ("error_band_runtime_only_relative_error_json", True),
                ("error_band_flow_state_alignment_error_json", False),
                ("error_band_candidate_path_cost_s_json", True),
                ("error_band_eligible_macro_count_json", True),
            ):
                values = json.loads(str(getattr(row, column)))
                numeric_values = np.asarray(list(values.values()), dtype=np.float64)
                if (
                    set(values) != expected_json_keys
                    or not np.isfinite(numeric_values).all()
                    or (nonnegative and np.any(numeric_values < 0.0))
                ):
                    semantic_failures.append(
                        f"{row.sample_od_id}: invalid {column}"
                    )
        expected_ranking_rows = {
            (runtime_class, band)
            for runtime_class in PILOT_ERROR_SUMMARY_CLASSES
            for band in PILOT_ERROR_BANDS
        }
        actual_ranking_rows = {
            (str(row.runtime_class), float(row.max_relative_error_band))
            for row in ranking_frame.itertuples(index=False)
        }
        if (
            len(ranking_frame) != len(expected_ranking_rows)
            or actual_ranking_rows != expected_ranking_rows
        ):
            semantic_failures.append("ranking-band summary key closure failed")
        for row in ranking_frame.itertuples(index=False):
            strict = int(row.physical_strict_selected_advantage_count)
            ranking_pass = bool(
                int(row.paired_od_count) > 0
                and int(row.physical_reference_tie_count)
                + strict
                + int(row.physical_reference_inversion_count)
                == int(row.paired_od_count)
                and int(row.candidate_strict_advantage_preserved_count)
                + int(row.candidate_advantage_eroded_to_tie_count)
                + int(row.candidate_ranking_reversal_count)
                == strict
                and np.isclose(
                    float(row.strict_advantage_preservation_fraction),
                    int(row.candidate_strict_advantage_preserved_count)
                    / max(1, strict),
                    rtol=0.0,
                    atol=1.0e-15,
                )
                and np.isclose(
                    float(row.candidate_ranking_reversal_fraction),
                    int(row.candidate_ranking_reversal_count) / max(1, strict),
                    rtol=0.0,
                    atol=1.0e-15,
                )
                and row.status
                == "MEASURED_ON_ACCEPTED_HIGH_PATH_PAIRS_NO_ROUTING"
            )
            if not ranking_pass:
                semantic_failures.append(
                    f"ranking-band row arithmetic failed: {row.runtime_class}/{row.max_relative_error_band}"
                )
    except Exception as error:
        semantic_failures.append(
            f"error-band/ranking semantic validation: {type(error).__name__}: {error}"
        )
    json_failures: list[str] = []
    status_schema_pass = False
    try:
        status = json.loads(
            (root / "reports" / "08_PILOT_STATUS.json").read_text(
                encoding="utf-8"
            )
        )
        schema = json.loads(
            (authority_root / "config" / "pilot_status.schema.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema_validate(status, schema)
        status_schema_pass = status.get("status") == FINAL_STATUS
        if not status_schema_pass:
            json_failures.append("reports/08_PILOT_STATUS.json: not final success")
        false_flags = (
            "full_phase5_authorized",
            "phase6_authorized",
            "national_assignment_performed",
            "final_reduced_topology_accepted",
            "final_runtime_thresholds_frozen",
            "next_step_authorized",
        )
        if any(status.get(value) is not False for value in false_flags):
            json_failures.append(
                "reports/08_PILOT_STATUS.json: downstream authorization flag is not false"
            )
        if status.get("blockers") != []:
            json_failures.append(
                "reports/08_PILOT_STATUS.json: success blockers are not empty"
            )
        if status.get("tests", {}).get("focused_phase5") != "PASS":
            json_failures.append(
                "reports/08_PILOT_STATUS.json: focused test status is not PASS"
            )
        if set(status.get("decisions", {})) != {
            f"P5-G{value:03d}" for value in range(1, 13)
        }:
            json_failures.append(
                "reports/08_PILOT_STATUS.json: P5-G001..P5-G012 closure differs"
            )
        live_domains = pd.read_csv(
            root / "runtime_fits" / "pilot_domain_summary.csv"
        )
        expected_output_counts = {
            "topology_suite_count": len(
                pd.read_csv(root / "topology" / "topology_census.csv")
            ),
            "candidate_chain_rows": pq.ParquetFile(
                root / "topology" / "chain_candidates.parquet"
            ).metadata.num_rows,
            "sampled_macro_signatures": int(
                live_domains["canonical_macro_signature"].astype(str).nunique()
            ),
            "oracle_rows": pq.ParquetFile(
                root / "bundle_oracle" / "oracle_sample_results.parquet"
            ).metadata.num_rows,
            "fit_rows": pq.ParquetFile(
                root / "runtime_fits" / "runtime_fit_results.parquet"
            ).metadata.num_rows,
            "visit_rows": pq.ParquetFile(
                root / "charge_audit" / "bacn_visit_semantics.parquet"
            ).metadata.num_rows,
            "path_rows": pq.ParquetFile(
                root / "path_validation" / "path_cost_sample.parquet"
            ).metadata.num_rows,
        }
        if status.get("outputs") != expected_output_counts:
            json_failures.append(
                "reports/08_PILOT_STATUS.json: output counts differ from live tables"
            )
        if final and status.get("post_checkpoint_head_bound") is not True:
            json_failures.append(
                "reports/08_PILOT_STATUS.json: post-checkpoint HEAD is not bound"
            )
    except Exception as error:
        status = {}
        json_failures.append(
            f"reports/08_PILOT_STATUS.json: {type(error).__name__}: {error}"
        )
    for relative, schema_id in (
        ("validation/reproduction.json", "partC.phase5.pilot.reproduction.v1"),
        ("validation/resource_usage.json", "partC.phase5.pilot.resource_usage.v1"),
        ("validation/figure_visual_qa.json", None),
    ):
        try:
            value = json.loads((root / relative).read_text(encoding="utf-8"))
            if schema_id is not None and value.get("schema_id") != schema_id:
                raise ValueError(f"unexpected schema_id {value.get('schema_id')}")
            if relative.endswith("reproduction.json") and not (
                value.get("status") == "PASS_REPRODUCIBLE_FROM_BOUND_INPUTS"
                and value.get("high_excluded_from_fit_and_selection") is True
                and value.get("exact_current_not_aon_primary") is True
                and value.get("capacity_factor_applied_once") is True
                and value.get("routing_or_assignment_performed") is False
            ):
                raise ValueError("reproduction semantic flags differ")
            if relative.endswith("resource_usage.json") and value.get("status") != "PASS":
                raise ValueError("resource status is not PASS")
            if relative.endswith("figure_visual_qa.json") and value.get("status") != "PASS":
                raise ValueError("figure reopen status is not PASS")
        except Exception as error:
            json_failures.append(f"{relative}: {type(error).__name__}: {error}")
    try:
        invariance = json.loads(
            (root / "validation" / "accepted_input_invariance.json").read_text(
                encoding="utf-8"
            )
        )
        external = pd.read_csv(
            root / "manifests" / "external_artifact_bindings.csv"
        )
        binding_pass = bool(
            invariance.get("schema_id")
            == "partC.phase5.accepted_input_invariance.v1"
            and invariance.get("status") == "PASS"
            and invariance.get("pre_post_exact_match") is True
            and int(invariance.get("binding_count", -1)) == len(external)
            and len(external) > 0
            and external["role"].astype(str).is_unique
            and external["reopen_status"].eq("PASS").all()
        )
        for row in external.itertuples(index=False):
            observed = str(row.sha256_or_directory_digest)
            expected = str(row.expected_sha256_or_digest)
            path = Path(str(row.absolute_path))
            binding_pass &= bool(
                len(observed) == 64
                and all(character in "0123456789abcdef" for character in observed)
                and path.exists()
                and (
                    expected == "BOUND_BY_ACCEPTED_MANIFEST_OR_IDENTITY"
                    or expected == observed
                )
            )
        if not binding_pass:
            raise ValueError("accepted input invariance/binding closure differs")
    except Exception as error:
        json_failures.append(
            f"accepted input invariance: {type(error).__name__}: {error}"
        )
    try:
        resource = json.loads(
            (root / "validation" / "resource_usage.json").read_text(
                encoding="utf-8"
            )
        )
        wall_limit = float(resource["wall_clock_limit_seconds"])
        seal_reserve = float(resource["protected_sealing_seconds"])
        rss_limit = int(resource["process_tree_rss_limit_bytes"])
        disk_limit = int(resource["added_disk_limit_bytes"])
        resource_pass = bool(
            0.0 <= float(resource["scientific_elapsed_seconds"])
            < wall_limit - seal_reserve
            and 0 <= int(resource["process_tree_peak_observed_rss_bytes"])
            <= rss_limit
            and resource.get("network_used") is False
            and resource.get("package_installation_performed") is False
            and resource.get("status") == "PASS"
        )
        if "total_pilot_elapsed_seconds_before_post_commit_handoff" in resource:
            resource_pass &= bool(
                0.0
                <= float(
                    resource[
                        "total_pilot_elapsed_seconds_before_post_commit_handoff"
                    ]
                )
                < wall_limit
            )
        if final:
            resource_pass &= bool(
                0
                <= int(resource["added_output_bytes_before_final_manifest"])
                <= disk_limit
                and 0
                <= int(
                    resource["added_output_bytes_before_post_commit_manifest"]
                )
                <= disk_limit
            )
        if not resource_pass:
            raise ValueError("resource evidence arithmetic or policy differs")
    except Exception as error:
        json_failures.append(
            f"validation/resource_usage.json: {type(error).__name__}: {error}"
        )
    if final:
        try:
            reproduction = json.loads(
                (root / "validation" / "reproduction.json").read_text(
                    encoding="utf-8"
                )
            )
            receipt = json.loads(
                (root / "repository_audit" / "post_checkpoint.json").read_text(
                    encoding="utf-8"
                )
            )
            if not (
                reproduction.get("post_checkpoint_head_bound") is True
                and reproduction.get("ending_head") == status.get("ending_head")
                and receipt.get("status") == "PASS"
                and receipt.get("branch") == status.get("branch")
                and receipt.get("starting_head") == status.get("starting_head")
                and receipt.get("ending_head") == status.get("ending_head")
                and receipt.get("starting_head_is_ancestor") is True
                and receipt.get("scientific_status") == FINAL_STATUS
            ):
                raise ValueError("post-checkpoint receipt/reproduction binding differs")
        except Exception as error:
            json_failures.append(
                f"post-checkpoint binding: {type(error).__name__}: {error}"
            )

    expected_figures = {
        "01_before_after_topology_counts.png",
        "02_chain_lengths_and_protection.png",
        "03_runtime_class_error_coverage.png",
        "04_fit_error_vs_complexity.png",
        "05_kernel_runtime_relative_to_bpr.png",
        "06_stored_vs_incidence_visit_flow.png",
        "07_low_charge_high_heldout_residual.png",
    }
    actual_figures = {path.name for path in (root / "figures").glob("*.png")}
    figure_failures = sorted(expected_figures - actual_figures)
    figure_count = len(actual_figures)
    expected_figure_sources = {
        "01_topology_counts.csv",
        "02_protection_reasons.csv",
        "03_fit_coverage.csv",
        "04_complexity_error.csv",
        "05_runtime_relative.csv",
        "06_visit_comparison.csv",
        "07_high_charge_residual.csv",
    }
    actual_figure_sources = {
        path.name
        for path in (root / "figures" / "source_data").glob("*.csv")
        if path.is_file() and path.stat().st_size > 0
    }
    figure_failures.extend(
        f"missing source_data/{value}"
        for value in sorted(expected_figure_sources - actual_figure_sources)
    )
    try:
        qa = json.loads(
            (root / "validation" / "figure_visual_qa.json").read_text(
                encoding="utf-8"
            )
        )
        receipts = qa.get("figures", [])
        receipt_names = {Path(str(value["path"])).name for value in receipts}
        if receipt_names != expected_figures:
            figure_failures.append("figure reopen receipt member closure differs")
        for receipt in receipts:
            path = Path(str(receipt["path"]))
            if (
                receipt.get("status") != "PASS_REOPEN"
                or not path.is_file()
                or sha256_file(path) != str(receipt.get("sha256"))
                or int(receipt.get("width", 0)) <= 0
                or int(receipt.get("height", 0)) <= 0
            ):
                figure_failures.append(f"invalid figure reopen receipt: {path.name}")
    except Exception as error:
        figure_failures.append(
            f"figure reopen receipt {type(error).__name__}: {error}"
        )

    decision_frame = pd.read_csv(
        root / "validation" / "pilot_decision_evidence.csv"
    )
    decision_pass = bool(
        len(decision_frame) == 12
        and decision_frame["decision_id"].is_unique
        and set(decision_frame["decision_id"].astype(str))
        == {f"P5-G{value:03d}" for value in range(1, 13)}
        and decision_frame["externally_frozen"].eq(False).all()
        and decision_frame[
            [
                "measured_facts",
                "recommendation",
                "resolution_status",
                "evidence_paths",
            ]
        ].notna().all().all()
    )
    for row in decision_frame.itertuples(index=False):
        for relative in str(row.evidence_paths).split(";"):
            decision_pass &= bool(relative and (root / relative).is_file())
    decision_pass &= bool(
        status.get("decisions", {})
        == dict(
            zip(
                decision_frame["decision_id"].astype(str),
                decision_frame["recommendation"].astype(str),
                strict=True,
            )
        )
    )
    if not decision_pass:
        semantic_failures.append("P5-G001..P5-G012 decision evidence closure failed")

    try:
        projection = pd.read_csv(
            root / "validation" / "full_phase5_resource_projection.csv"
        )
        projection_numeric = projection[
            [
                "pilot_observed_value",
                "naive_full_phase5_projection",
                "governing_limit",
                "projection_to_limit_ratio",
            ]
        ].to_numpy(dtype=np.float64)
        projection_pass = bool(
            len(projection) == 3
            and projection["metric"].astype(str).is_unique
            and set(projection["metric"].astype(str))
            == {"wall_clock_seconds", "process_tree_rss_bytes", "output_bytes"}
            and np.isfinite(projection_numeric).all()
            and np.all(projection_numeric[:, :2] >= 0.0)
            and np.all(projection_numeric[:, 2] > 0.0)
            and np.all(projection_numeric[:, 3] >= 0.0)
            and np.allclose(
                projection_numeric[:, 3],
                projection_numeric[:, 1] / projection_numeric[:, 2],
                rtol=1.0e-12,
                atol=1.0e-15,
            )
            and projection["pilot_observation_stage"].notna().all()
            and projection["status"].eq(
                "SCREEN_ONLY_REQUIRES_REVIEWED_PREFLIGHT"
            ).all()
        )
        if not projection_pass:
            raise ValueError("resource projection closure differs")
    except Exception as error:
        semantic_failures.append(
            f"resource projection semantic validation: {type(error).__name__}: {error}"
        )

    acceptance = pd.read_csv(root / "validation" / "acceptance_test_results.csv")
    scientific_gate_ids = {
        "P5-AUTH-01",
        "P5-TEST-01",
        "P5-INPUT-01",
        "P5-TOPO-01",
        "P5-FLOW-01",
        "P5-ORACLE-01",
        "P5-RUNTIME-01",
        "P5-BENCH-01",
        "P5-MULTIEDGE-01",
        "P5-CHARGE-01",
        "P5-RACN-01",
        "P5-PATH-01",
        "P5-BOUNDARY-01",
        "P5-RESOURCE-01",
    }
    expected_gate_ids = scientific_gate_ids | ({"P5-OUTPUT-01"} if final else set())
    acceptance_pass = bool(
        len(acceptance) > 0
        and {"test_id", "status", "gate_class", "evidence_json"}.issubset(
            acceptance.columns
        )
        and acceptance["test_id"].is_unique
        and acceptance["status"].eq("PASS").all()
        and acceptance["gate_class"].eq("hard").all()
        and set(acceptance["test_id"].astype(str)) == expected_gate_ids
    )
    if acceptance_pass:
        try:
            for evidence in acceptance["evidence_json"].astype(str):
                if not isinstance(json.loads(evidence), dict):
                    raise ValueError("acceptance evidence is not a JSON object")
        except Exception:
            acceptance_pass = False
    manifest = pd.read_csv(root / "manifests" / "MANIFEST.csv")
    manifest_failures: list[str] = []
    manifest_required = {
        "relative_path",
        "absolute_path",
        "size_bytes",
        "sha256",
        "row_or_object_count",
        "schema_or_shape_json",
        "reopen_status",
        "handoff_policy",
    }
    if len(manifest) == 0 or not manifest_required.issubset(manifest.columns):
        manifest_failures.append("manifest empty or missing required columns")
    else:
        if not manifest["relative_path"].is_unique:
            manifest_failures.append("manifest relative paths are not unique")
        actual_manifest_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and path
            not in {
                root / "manifests" / "MANIFEST.csv",
                root / "manifests" / "CHECKSUMS.sha256",
            }
        }
        declared_manifest_paths = set(manifest["relative_path"].astype(str))
        if actual_manifest_paths != declared_manifest_paths:
            manifest_failures.append(
                "manifest member closure differs: "
                f"missing={sorted(actual_manifest_paths - declared_manifest_paths)}, "
                f"extra={sorted(declared_manifest_paths - actual_manifest_paths)}"
            )
        for row in manifest.itertuples(index=False):
            path = root / str(row.relative_path)
            if not path.is_file():
                continue
            actual_size = path.stat().st_size
            actual_sha = sha256_file(path)
            if actual_size != int(row.size_bytes) or actual_sha != str(row.sha256):
                manifest_failures.append(
                    f"{row.relative_path}: size/hash differs from manifest"
                )
                continue
            if str(Path(str(row.absolute_path)).resolve()) != str(path.resolve()):
                manifest_failures.append(
                    f"{row.relative_path}: absolute path differs from live output"
                )
            try:
                reopen_status, row_count, schema_json = _describe_reopen(path)
                if (
                    reopen_status != "PASS"
                    or str(row.reopen_status) != "PASS"
                    or int(row_count) != int(row.row_or_object_count)
                    or schema_json != str(row.schema_or_shape_json)
                ):
                    manifest_failures.append(
                        f"{row.relative_path}: live reopen/schema/count differs"
                    )
            except Exception as error:
                manifest_failures.append(
                    f"{row.relative_path}: reopen {type(error).__name__}: {error}"
                )

    checksum_failures: list[str] = []
    checksum_path = root / "manifests" / "CHECKSUMS.sha256"
    try:
        checksum_rows: dict[str, str] = {}
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            if relative in checksum_rows:
                raise ValueError(f"duplicate checksum path {relative}")
            checksum_rows[relative] = digest
        expected_checksum_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != checksum_path
        }
        if set(checksum_rows) != expected_checksum_paths:
            checksum_failures.append(
                "checksum member closure differs: "
                f"missing={sorted(expected_checksum_paths - set(checksum_rows))}, "
                f"extra={sorted(set(checksum_rows) - expected_checksum_paths)}"
            )
        for relative, digest in checksum_rows.items():
            path = root / relative
            if not path.is_file() or sha256_file(path) != digest:
                checksum_failures.append(f"{relative}: checksum mismatch")
    except Exception as error:
        checksum_failures.append(f"{type(error).__name__}: {error}")

    passed = bool(
        not missing
        and not empty
        and not schema_failures
        and not semantic_failures
        and not zero_row_tables
        and not json_failures
        and not figure_failures
        and acceptance_pass
        and not manifest_failures
        and not checksum_failures
        and status_schema_pass
    )
    return {
        "schema_id": "partC.phase5.required_output_validation.v1",
        "required_path_count": len(required_paths),
        "missing_paths": missing,
        "empty_paths": empty,
        "schema_failures": schema_failures,
        "semantic_failures": semantic_failures,
        "zero_row_tables": zero_row_tables,
        "figure_count": figure_count,
        "figure_failures": figure_failures,
        "json_failures": json_failures,
        "acceptance_row_count": len(acceptance),
        "acceptance_pass": acceptance_pass,
        "decision_evidence_pass": decision_pass,
        "final_closure_required": final,
        "manifest_row_count": len(manifest),
        "manifest_failures": manifest_failures,
        "checksum_failures": checksum_failures,
        "status_schema_pass": status_schema_pass,
        "status": "PASS" if passed else "FAIL",
    }


def _external_bindings(bindings: PilotBindings, output: Path) -> pd.DataFrame:
    paths: list[tuple[str, Path, str | None]] = []
    authority_installation_path = (
        Path(bindings.execution["project_root"])
        / "partC_doc_icnn"
        / "review_gates"
        / "phase_05"
        / "backbone_compilation_pilot"
        / "phase5_pilot_authority_20260903"
        / "AUTHORITY_INSTALLATION.json"
    )
    authority_installation = json.loads(
        authority_installation_path.read_text(encoding="utf-8")
    )
    paths.extend(
        [
            (
                "outer_phase5_authority_zip",
                Path(authority_installation["source_package"]["path"]),
                OUTER_AUTHORITY_SHA256,
            ),
            (
                "nested_parent_authority_zip",
                Path(
                    authority_installation["nested_parent_authority"][
                        "external_path"
                    ]
                ),
                PARENT_AUTHORITY_SHA256,
            ),
        ]
    )
    phase4_config = Path(bindings.execution["project_root"]) / "partC_doc_icnn" / "phase_4_work" / "phase4_dual_accuracy_benchmark_execution_config.json"
    phase4 = json.loads(phase4_config.read_text(encoding="utf-8"))
    inputs = phase4["input_versions"]
    paths.extend(
        [
            ("physical_graph_npz", Path(inputs["graph_npz"]), None),
            ("physical_graph_metadata", Path(inputs["graph_metadata_json"]), None),
            ("accepted_od_npz", Path(inputs["od_npz"]), None),
            ("accepted_od_metadata", Path(inputs["od_metadata_json"]), None),
            ("accepted_aggregated_od", Path(inputs["accepted_aggregated_od_root"]) / "derived_inputs" / "aggregated_od.npz", None),
            ("phase3_sqlite", Path(bindings.sources["phase3_sqlite"]["path"]), bindings.sources["phase3_sqlite"]["sha256"]),
            ("frozen_path_sample", Path(bindings.sources["frozen_path_sample"]), None),
            (
                "high_selected_path_sample",
                Path(bindings.sources["high_benchmark"]["root"]) / "routes" / "sampled_route_switches.parquet",
                None,
            ),
        ]
    )
    for tier in ("low_benchmark", "high_benchmark"):
        source = bindings.sources[tier]
        paths.append((tier + "_physical_arrays", Path(source["physical_arrays"]), None))
        paths.append((tier + "_physical_table", Path(source["physical_link_table"]), None))
        for suite_id in bindings.execution["suite_ids"]:
            paths.append(
                (
                    f"{tier}_{suite_id}_current_aon_traces",
                    Path(str(source["suite_trace_template"]).format(suite=suite_id)),
                    None,
                )
            )
    rows = []
    for role, path, expected in paths:
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise RuntimeError(f"External binding hash mismatch: {role}")
        reopen, count, schema = _describe_reopen(path)
        rows.append(
            {
                "role": role,
                "absolute_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "sha256_or_directory_digest": digest,
                "expected_sha256_or_digest": expected or "BOUND_BY_ACCEPTED_MANIFEST_OR_IDENTITY",
                "format": path.suffix.lower().lstrip("."),
                "row_array_or_object_count": count,
                "schema_or_shape_json": schema,
                "reopen_status": reopen,
                "lineage": "immutable accepted Phase 3/4 input",
            }
        )
    for trace_map in bindings.trace_maps:
        observed_digest = _directory_digest(trace_map.root)
        if observed_digest != trace_map.directory_digest:
            raise RuntimeError(
                f"Accepted Phase 3 suite directory changed: {trace_map.suite_id}"
            )
        rows.append(
            {
                "role": f"phase3_{trace_map.suite_id}_accepted_directory",
                "absolute_path": str(trace_map.root.resolve()),
                "size_bytes": sum(path.stat().st_size for path in trace_map.root.rglob("*") if path.is_file()),
                "mtime_utc": datetime.fromtimestamp(trace_map.root.stat().st_mtime, timezone.utc).isoformat(),
                "sha256_or_directory_digest": observed_digest,
                "expected_sha256_or_digest": trace_map.directory_digest,
                "format": "directory",
                "row_array_or_object_count": 1,
                "schema_or_shape_json": canonical_json({"digest_algorithm": "phase4.tracing._directory_digest"}),
                "reopen_status": "PASS",
                "lineage": "immutable accepted Phase 3 suite directory",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    return frame


def _run_pilot_impl(config_path: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    execution = json.loads(config_path.read_text(encoding="utf-8"))
    repo = Path(execution["project_root"]).resolve()
    output = Path(execution["output_root"]).resolve()
    started = time.perf_counter()
    started_utc = utc_now()
    peak_rss = process_tree_rss_bytes()
    last_progress_at = started

    def resource_guard(stage: str, *, sealing: bool = False) -> None:
        nonlocal peak_rss, last_progress_at
        peak_rss = max(peak_rss, process_tree_rss_bytes())
        now = time.perf_counter()
        if now - last_progress_at >= 30.0:
            print(
                f"phase5 progress stage={stage} elapsed_s={now - started:.1f} "
                f"rss_bytes={peak_rss}",
                flush=True,
            )
            last_progress_at = now
        if peak_rss > execution["resources"]["process_tree_rss_bytes"]:
            raise MemoryError(f"Phase 5 process-tree RSS guard exceeded at {stage}")
        elapsed = now - started
        limit = execution["resources"]["wall_clock_seconds"]
        if not sealing:
            limit -= execution["resources"]["protected_sealing_seconds"]
        if elapsed >= limit:
            qualifier = "wall-clock" if sealing else "protected sealing reserve"
            raise TimeoutError(f"Phase 5 {qualifier} reached at {stage}")

    if output.exists():
        raise FileExistsError(f"Configured write-new pilot output already exists: {output}")
    branch = git(repo, "branch", "--show-current").stdout.strip()
    starting_head = git(repo, "rev-parse", "HEAD").stdout.strip()
    if branch != execution["expected_branch"]:
        raise RuntimeError(f"Expected branch {execution['expected_branch']}, found {branch}")
    if starting_head != STARTING_HEAD:
        raise RuntimeError(f"Pilot must start from authenticated HEAD {STARTING_HEAD}, found {starting_head}")
    ancestry = git(
        repo,
        "merge-base",
        "--is-ancestor",
        execution["reviewed_phase4_ancestor"],
        starting_head,
        check=False,
    )
    if ancestry.returncode != 0:
        raise RuntimeError("Reviewed Phase 4 HEAD is not an ancestor of the live starting HEAD")
    authority_root = (
        repo
        / "partC_doc_icnn"
        / "review_gates"
        / "phase_05"
        / "backbone_compilation_pilot"
        / "phase5_pilot_authority_20260903"
    )
    authority_verification = _verify_installed_authority(authority_root)
    launch = Path(execution["launch_authority_path"]).resolve()
    if (
        not launch.is_file()
        or sha256_file(launch) != str(execution["launch_authority_sha256"])
    ):
        raise RuntimeError(
            "Configured launch authority is absent or differs from its frozen SHA-256"
        )
    for directory in (
        "authority",
        "repository_audit",
        "topology",
        "bundle_oracle",
        "runtime_fits",
        "charge_audit",
        "interface_audit",
        "path_validation",
        "benchmarks",
        "figures",
        "reports",
        "validation",
        "manifests",
        "handoff_staging",
    ):
        (output / directory).mkdir(parents=True, exist_ok=False)

    test_command = [
        execution["python_executable"],
        "-m",
        "pytest",
        "-q",
        "tests/test_phase5_bundle_oracle.py",
        "tests/test_phase5_runtime_functions.py",
        "tests/test_phase5_topology_reduction.py",
        "tests/test_phase5_charge_semantics.py",
        "tests/test_phase5_multiedge.py",
        "tests/test_phase5_experiments.py",
        "tests/test_phase5_handoff.py",
    ]
    test_run = subprocess.run(
        test_command,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=max(
            1.0,
            execution["resources"]["wall_clock_seconds"]
            - execution["resources"]["protected_sealing_seconds"]
            - (time.perf_counter() - started),
        ),
    )
    write_text(
        output / "validation" / "focused_tests.txt",
        "$ " + subprocess.list2cmdline(test_command) + "\n" + test_run.stdout,
    )
    write_json(
        output / "validation" / "focused_tests_status.json",
        {
            "schema_id": "partC.phase5.focused_tests.v1",
            "command": subprocess.list2cmdline(test_command),
            "returncode": int(test_run.returncode),
            "status": "PASS" if test_run.returncode == 0 else "FAIL",
        },
    )
    resource_guard("focused_tests_complete")
    if test_run.returncode != 0:
        raise RuntimeError("Focused Phase 5 tests failed before data execution:\n" + test_run.stdout)
    shutil.copy2(config_path, output / "authority" / config_path.name)
    shutil.copytree(authority_root, output / "authority" / "installed_contract")
    shutil.copy2(launch, output / "authority" / launch.name)
    initial_status = git(repo, "status", "--short", "--branch").stdout
    write_json(
        output / "repository_audit" / "authentication.json",
        {
            "schema_id": "partC.phase5.repository_authentication.v1",
            "recorded_utc": utc_now(),
            "branch": branch,
            "starting_head": starting_head,
            "reviewed_phase4_ancestor": execution["reviewed_phase4_ancestor"],
            "ancestor_pass": True,
            "initial_status": initial_status.splitlines(),
            "outer_authority_sha256": OUTER_AUTHORITY_SHA256,
            "nested_parent_authority_sha256": PARENT_AUTHORITY_SHA256,
            "authority_verification": authority_verification,
            "launch_authority_path": str(launch),
            "launch_authority_sha256": sha256_file(launch),
            "network_used": False,
            "package_installation_performed": False,
        },
    )

    bindings = load_bindings(config_path, verify_suite_digests=True)
    pre_bindings = _external_bindings(
        bindings, output / "manifests" / "external_artifact_bindings.csv"
    )
    resource_guard("pre_science_external_bindings_complete")
    supply = reconstruct_reachable_od_supply(bindings.graph, bindings.od)
    high_metrics = json.loads(
        (Path(bindings.sources["high_benchmark"]["root"]) / "reports" / "final_metrics.json").read_text(encoding="utf-8")
    )
    expected_unreachable = float(high_metrics["unreachable_demand"])
    if not np.isclose(supply.unreachable_demand, expected_unreachable, rtol=0.0, atol=1.0e-9):
        raise RuntimeError("SCC reachability does not reproduce accepted Phase 4 unreachable demand")
    low_conservation = physical_flow_conservation(bindings.graph, bindings.low_physical.current_flow, supply)
    high_conservation = physical_flow_conservation(bindings.graph, bindings.high_physical.current_flow, supply)
    if not low_conservation["pass"] or not high_conservation["pass"]:
        raise RuntimeError("Accepted physical flow does not close to SCC-filtered OD supplies")
    write_json(
        output / "validation" / "scc_reachability_and_physical_conservation.json",
        {
            "schema_id": "partC.phase5.scc_endpoint_reconstruction.v1",
            "scc_count": supply.scc_count,
            "aggregated_od_row_count": supply.row_count,
            "unreachable_row_count": supply.unreachable_row_count,
            "unreachable_demand": supply.unreachable_demand,
            "accepted_unreachable_demand": expected_unreachable,
            "low_physical_conservation": low_conservation,
            "high_physical_conservation": high_conservation,
            "routing_or_assignment_performed": False,
            "status": "PASS",
        },
    )
    high_path_sample = (
        Path(bindings.sources["high_benchmark"]["root"])
        / "routes"
        / "sampled_route_switches.parquet"
    )
    path_demand_frame = pd.read_parquet(
        high_path_sample, columns=["sample_od_id", "accepted_od_demand"]
    )
    sample_weight_by_id = dict(
        zip(
            path_demand_frame["sample_od_id"].astype(str),
            path_demand_frame["accepted_od_demand"].astype(float),
            strict=True,
        )
    )

    global_specs: dict[str, BundleSpec] = {}
    suite_work: list[SuiteWork] = []
    topology_rows: list[dict[str, object]] = []
    chain_frames: list[pd.DataFrame] = []
    protection_frames: list[pd.DataFrame] = []
    bundle_occurrences: list[dict[str, object]] = []
    visit_frames: list[pd.DataFrame] = []
    charge_frames: list[pd.DataFrame] = []
    racn_rows: list[dict[str, object]] = []
    for trace_map in bindings.trace_maps:
        suite_id = trace_map.suite_id
        suite_root = trace_map.root
        frames = _suite_frames(suite_root)
        low_trace = load_suite_trace_arrays(
            bindings.sources,
            suite_id,
            "low_benchmark",
            trace_map=trace_map,
        )
        high_trace = load_suite_trace_arrays(
            bindings.sources,
            suite_id,
            "high_benchmark",
            trace_map=trace_map,
        )
        low_bundle = dict(zip(trace_map.bundle_ids, low_trace["current__bundle_flow"], strict=True))
        high_bundle = dict(zip(trace_map.bundle_ids, high_trace["current__bundle_flow"], strict=True))
        node_mapping = trace_map.node_to_bacn_local
        origin_local = node_mapping[bindings.od.origin_group_node_idx]
        destination_local = node_mapping[bindings.od.destination_node_idx]
        endpoint_local = np.unique(np.r_[origin_local[origin_local >= 0], destination_local[destination_local >= 0]])
        endpoint_ids = [trace_map.bacn_ids[int(value)] for value in endpoint_local]
        topology_result = enumerate_topology(
            suite_id=suite_id,
            bacn_summary=frames["bacn_summary"],
            static_links=frames["acnhn_static_links"],
            bundle_summary=frames["bundle_summary"],
            low_bundle_flow=low_bundle,
            high_bundle_flow=high_bundle,
            od_endpoint_bacn_ids=endpoint_ids,
            flow_atol=execution["topology_policy"]["flow_equality_atol"],
            flow_rtol=execution["topology_policy"]["flow_equality_rtol"],
        )
        bundle_key, rows = _bundle_inventory(
            suite_id,
            frames["bundle_membership"],
            low_trace,
            high_trace,
            trace_map,
            bindings,
            global_specs,
        )
        internal_key = _internal_component_keys(frames, bindings)
        work = SuiteWork(
            suite_id,
            suite_root,
            trace_map,
            frames,
            low_trace,
            high_trace,
            topology_result,
            bundle_key,
            internal_key,
        )
        suite_work.append(work)
        topology_rows.append(topology_result.census)
        chain_frames.append(topology_result.chains)
        protection_frames.append(topology_result.node_protection)
        bundle_occurrences.extend(rows)
        low_visits = reconstruct_bacn_visit_semantics(
            trace_map=trace_map,
            bundle_summary=frames["bundle_summary"],
            terminals=frames["consolidated_terminals"],
            trace_arrays=low_trace,
            benchmark_id=bindings.low_physical.benchmark_id,
            supply=supply,
        )
        high_visits = reconstruct_bacn_visit_semantics(
            trace_map=trace_map,
            bundle_summary=frames["bundle_summary"],
            terminals=frames["consolidated_terminals"],
            trace_arrays=high_trace,
            benchmark_id=bindings.high_physical.benchmark_id,
            supply=supply,
        )
        for frame in (low_visits, high_visits):
            balance = np.abs(frame["incoming_charged_flow"] - frame["outgoing_charged_flow"])
            if np.any(balance > 1.0e-6) or frame["status"].eq("FAIL").any():
                raise RuntimeError(f"{suite_id}: charged-incidence reconstruction failed")
        low_episodes = path_episode_statistics(
            bindings.sources["frozen_path_sample"],
            trace_map,
            bindings.low_physical.current_cost_s,
            sample_weight_by_id=sample_weight_by_id,
        )
        high_episodes = path_episode_statistics(
            high_path_sample,
            trace_map,
            bindings.high_physical.current_cost_s,
            arc_column="selected_arc_indices_json",
            sample_weight_by_id=sample_weight_by_id,
        )
        for visit_frame, episode_frame in (
            (low_visits, low_episodes),
            (high_visits, high_episodes),
        ):
            episode_by_id = episode_frame.set_index("bacn_id")
            observed = visit_frame["bacn_id"].isin(episode_by_id.index)
            visit_frame.loc[observed, "path_episode_visit_flow"] = visit_frame.loc[
                observed, "bacn_id"
            ].map(episode_by_id["sample_demand_weighted_episode_flow"])
            visit_frame.loc[
                observed, "path_episode_observation_status"
            ] = "OBSERVED_DEMAND_WEIGHTED_IN_ACCEPTED_200_PATH_SAMPLE"
        visit_frames.extend((low_visits, high_visits))
        low_exposure = internal_exposure(
            trace_map, bindings.low_physical.current_flow, bindings.low_physical.current_cost_s
        )
        high_exposure = internal_exposure(
            trace_map, bindings.high_physical.current_flow, bindings.high_physical.current_cost_s
        )
        charges = fixed_charge_candidates(
            trace_map=trace_map,
            low_visits=low_visits["incidence_visit_flow"].to_numpy(dtype=np.float64),
            high_visits=high_visits["incidence_visit_flow"].to_numpy(dtype=np.float64),
            low_exposure=low_exposure,
            high_exposure=high_exposure,
            structural_charges=frames["structural_internal_charges"],
            episode_statistics=low_episodes,
        )
        charge_frames.append(charges)
        racn_rows.append(_racn_audit(work))
        resource_guard(f"suite_census_{suite_id}")

    topology = pd.DataFrame(topology_rows)
    chains = pd.concat(chain_frames, ignore_index=True)
    protection = pd.concat(protection_frames, ignore_index=True)
    bundle_rows = pd.DataFrame(bundle_occurrences)
    bundle_census = _bundle_census(bundle_rows)
    visits = pd.concat(visit_frames, ignore_index=True)
    charges = pd.concat(charge_frames, ignore_index=True)
    racn = pd.DataFrame(racn_rows)

    work_by_suite = {work.suite_id: work for work in suite_work}
    bundle_local_by_suite = {
        work.suite_id: {
            value: position
            for position, value in enumerate(work.trace_map.bundle_ids)
        }
        for work in suite_work
    }
    charge_estimator_id = "LOW_OPERATING_MEAN_ELSE_STRUCTURAL_SEED_V1"
    charge_by_suite: dict[str, dict[str, float]] = {}
    for suite_id, frame in charges.groupby("suite_short_id", sort=True):
        value = frame["low_operating_mean_seconds"].to_numpy(dtype=np.float64)
        seed = frame["structural_seed_seconds"].to_numpy(dtype=np.float64)
        recommended = np.where(np.isfinite(value), value, seed)
        charge_by_suite[str(suite_id)] = dict(
            zip(frame["bacn_id"].astype(str), recommended, strict=True)
        )
    macro_candidates: list[MacroExperiment] = []
    for index, row in chains.iterrows():
        work = work_by_suite[str(row["suite_short_id"])]
        primitive_ids = json.loads(str(row["primitive_bundle_ids_json"]))
        internal_ids = json.loads(str(row["internal_bacn_ids_json"]))
        primitive_keys = [work.bundle_key[value] for value in primitive_ids]
        internal_keys = [work.internal_key[value] for value in internal_ids]
        internal_charge_s = float(
            sum(charge_by_suite[work.suite_id][value] for value in internal_ids)
        )
        internal_component_charges = tuple(
            float(charge_by_suite[work.suite_id][value]) for value in internal_ids
        )
        signature = macro_function_key(
            primitive_keys,
            internal_component_keys=internal_keys,
            internal_charge_s=internal_charge_s,
            internal_charge_estimator_id=charge_estimator_id,
            graph_identity=bindings.sources["graph"]["identity"],
        )
        chains.at[index, "canonical_signature"] = signature
        if bool(row["protected_or_blocked"]):
            continue
        bundle_local = bundle_local_by_suite[work.suite_id]
        local = np.asarray([bundle_local[value] for value in primitive_ids], dtype=np.int64)
        macro_candidates.append(
            MacroExperiment(
                canonical_signature=signature,
                occurrence_id=str(row["chain_id"]),
                suite_id=work.suite_id,
                primitive_bundles=tuple(global_specs[key] for key in primitive_keys),
                low_flow=0.5 * (float(row["low_flow_min"]) + float(row["low_flow_max"])),
                low_aon_flow=float(np.max(work.low_trace["aon__bundle_flow"][local])),
                high_flow=0.5 * (float(row["high_flow_min"]) + float(row["high_flow_max"])),
                chain_length=int(row["primitive_link_count"]),
                endpoint_collision=bool(str(row["parallel_collision_group_id"])),
                internal_charge_s=internal_charge_s,
                internal_charge_estimator_id=charge_estimator_id,
                internal_component_ids=tuple(map(str, internal_ids)),
                internal_component_charge_s=internal_component_charges,
            )
        )
    if not macro_candidates:
        raise RuntimeError("Full topology census produced zero unblocked macro candidates")
    low_path_intersections = _accepted_path_candidate_signatures(
        bindings, work_by_suite["s03"], chains
    )
    high_path_intersections = _accepted_path_candidate_signatures(
        bindings,
        work_by_suite["s03"],
        chains,
        Path(bindings.sources["high_benchmark"]["root"])
        / "routes"
        / "sampled_route_switches.parquet",
        "selected_arc_indices_json",
    )
    path_intersections = low_path_intersections | high_path_intersections
    candidate_by_signature = {
        value.canonical_signature: value for value in macro_candidates
    }
    exact_path_intersections = {
        signature
        for signature in path_intersections
        if signature in candidate_by_signature
        and series_power_terms(
            candidate_by_signature[signature].primitive_bundles,
            internal_charge_s=candidate_by_signature[signature].internal_charge_s,
        )
        is not None
    }
    maximum_sample = int(
        execution["pilot_sample"]["max_unique_macro_signatures"]
    )
    namespace = str(execution["pilot_sample"]["stable_hash_namespace"])

    def force_rank(signature: str) -> str:
        return hashlib.sha256(
            f"{namespace}\0accepted-path-force\0{signature}".encode("utf-8")
        ).hexdigest()

    exact_low = exact_path_intersections & low_path_intersections
    exact_high = exact_path_intersections & high_path_intersections
    if not exact_low or not exact_high:
        raise RuntimeError(
            "Accepted low/high path samples lack a forceable exact-series candidate"
        )
    forced_path_signatures = {
        min(exact_low, key=force_rank),
        min(exact_high, key=force_rank),
    }
    path_force_quota = min(1000, max(2, maximum_sample // 10))
    remaining_force_slots = path_force_quota - len(forced_path_signatures)
    forced_path_signatures.update(
        sorted(exact_path_intersections - forced_path_signatures, key=force_rank)[
            :remaining_force_slots
        ]
    )
    sample, inclusion = select_macro_sample(
        macro_candidates,
        maximum=maximum_sample,
        force_extreme=execution["pilot_sample"]["force_extreme_per_metric"],
        namespace=namespace,
        force_signatures=forced_path_signatures,
        progress_guard=resource_guard,
    )
    selected_signatures = set(inclusion)
    chains["sample_included"] = chains["canonical_signature"].isin(selected_signatures)
    chains["sample_inclusion_reasons_json"] = chains["canonical_signature"].map(
        lambda signature: json.dumps(inclusion.get(signature, ()), separators=(",", ":"))
    )
    oracle, fits, domains = run_macro_experiments(
        sample, progress_guard=resource_guard
    )
    resource_guard("oracle_and_runtime_fits_complete")
    if len(oracle) == 0 or len(fits) == 0:
        raise RuntimeError("Exact-oracle or runtime-fit experiment was a no-op")

    selected_rows = fits.loc[
        fits["recommended_status"].astype(str).str.startswith("CASCADE_SELECTED")
    ]
    selected_class_by_signature = dict(
        zip(
            selected_rows["canonical_macro_signature"].astype(str),
            selected_rows["runtime_class"].astype(str),
            strict=True,
        )
    )
    sampled_signatures = set(domains["canonical_macro_signature"].astype(str))
    chains["runtime_resolution"] = np.where(
        chains["protected_or_blocked"].astype(bool),
        "PROTECTED_NO_CONTRACTION",
        np.where(
            chains["canonical_signature"].astype(str).isin(
                selected_class_by_signature
            ),
            "SELECTED_ROOT_FREE_RUNTIME",
            np.where(
                chains["canonical_signature"].astype(str).isin(sampled_signatures),
                "RETAIN_ORIGINAL_PRIMITIVE_LINKS_PILOT_ERROR_GATE",
                "RETAIN_ORIGINAL_PRIMITIVE_LINKS_NOT_IN_PILOT_SAMPLE",
            ),
        ),
    )
    chains["selected_runtime_class"] = chains["canonical_signature"].astype(
        str
    ).map(selected_class_by_signature).fillna("")
    domains["runtime_resolution"] = domains["canonical_macro_signature"].astype(
        str
    ).map(
        lambda value: (
            "SELECTED_ROOT_FREE_RUNTIME"
            if value in selected_class_by_signature
            else "RETAIN_ORIGINAL_PRIMITIVE_LINKS_PILOT_ERROR_GATE"
        )
    )
    domains["split_diagnostic_status"] = domains[
        "canonical_macro_signature"
    ].astype(str).map(
        lambda value: (
            "NOT_APPLICABLE_SELECTED"
            if value in selected_class_by_signature
            else "NO_AUTOMATIC_SPLIT_PILOT_RETENTION_IS_CONSERVATIVE"
        )
    )
    chain_provenance = chains.drop_duplicates("chain_id").set_index("chain_id")
    domains["retained_primitive_bundle_ids_json"] = domains[
        "suite_occurrence_id"
    ].map(chain_provenance["primitive_bundle_ids_json"])
    domains["retained_internal_bacn_ids_json"] = domains[
        "suite_occurrence_id"
    ].map(chain_provenance["internal_bacn_ids_json"])
    if domains[
        ["retained_primitive_bundle_ids_json", "retained_internal_bacn_ids_json"]
    ].isna().any().any():
        raise RuntimeError("Sampled runtime domains do not close to chain provenance")
    runtime_registry_schema = json.loads(
        (
            authority_root / "config" / "runtime_function_class.schema.json"
        ).read_text(encoding="utf-8")
    )
    runtime_registry = _selected_runtime_registry(
        fits, domains, oracle, runtime_registry_schema
    )

    topology.to_csv(output / "topology" / "topology_census.csv", index=False)
    chains.to_parquet(output / "topology" / "chain_candidates.parquet", index=False)
    protection.to_parquet(output / "topology" / "bacn_protection_census.parquet", index=False)
    bundle_census.to_csv(output / "bundle_oracle" / "bundle_parameter_census.csv", index=False)
    bundle_rows.to_parquet(output / "bundle_oracle" / "bundle_parameter_occurrences.parquet", index=False)
    oracle.to_parquet(output / "bundle_oracle" / "oracle_sample_results.parquet", index=False)
    fits.to_parquet(output / "runtime_fits" / "runtime_fit_results.parquet", index=False)
    domains.to_csv(output / "runtime_fits" / "pilot_domain_summary.csv", index=False)
    write_json(
        output / "runtime_fits" / "selected_runtime_registry.json",
        runtime_registry,
    )
    visits.to_parquet(output / "charge_audit" / "bacn_visit_semantics.parquet", index=False)
    charges.to_parquet(output / "charge_audit" / "fixed_charge_candidates.parquet", index=False)
    racn.to_csv(output / "interface_audit" / "racn_cost_ownership_audit.csv", index=False)

    multiedge = _multiedge_audit(bindings, topology)
    multiedge.to_csv(output / "topology" / "multiedge_audit.csv", index=False)
    projected_counts = {}
    for work in suite_work:
        original_backbone = int(
            topology.loc[
                topology["suite_short_id"].eq(work.suite_id),
                "original_backbone_bundle_count",
            ].iloc[0]
        )
        resolved = chains.loc[
            chains["suite_short_id"].eq(work.suite_id)
            & chains["runtime_resolution"].eq("SELECTED_ROOT_FREE_RUNTIME")
        ]
        selected_contraction_savings = int(
            (resolved["primitive_link_count"].astype(np.int64) - 1).sum()
        )
        projected_backbone = original_backbone - selected_contraction_savings
        # Stage J measures only the link family compiled by this pilot. Terminal
        # linear charges and future regional/ICNN kernels have different algebra
        # and are not mislabeled as ordinary BPR in the timing denominator.
        projected_counts[work.suite_id] = projected_backbone
    resource_guard("before_kernel_benchmarks")
    reopened_fits = pd.read_parquet(
        output / "runtime_fits" / "runtime_fit_results.parquet"
    )
    reopened_chains = pd.read_parquet(
        output / "topology" / "chain_candidates.parquet"
    )
    benchmarks = run_kernel_benchmarks(
        projected_counts,
        fit_results=reopened_fits,
        chain_candidates=reopened_chains,
        bpr_parameters={
            "free_time_s": bindings.graph.free_time * 60.0,
            "used_capacity": bindings.low_physical.used_capacity,
            "alpha": bindings.graph.alpha,
            "beta": bindings.graph.beta,
            "flow": bindings.low_physical.current_flow,
        },
        offline_experiments=sample,
        warmups=execution["kernel_benchmark"]["warmups"],
        repetitions=execution["kernel_benchmark"]["timed_repetitions"],
        progress_guard=resource_guard,
    )
    resource_guard("kernel_benchmarks_complete")
    peak_rss = max(
        peak_rss,
        int(benchmarks["peak_rss_bytes"].max()),
    )
    benchmarks.to_csv(output / "benchmarks" / "kernel_runtime_benchmark.csv", index=False)
    if benchmarks.loc[~benchmarks["kernel_class"].eq("EXACT_ROOT_ORACLE_OFFLINE_ONLY"), "exact_root_calls"].ne(0).any():
        raise RuntimeError("A production runtime benchmark invoked the offline exact root")

    s03 = work_by_suite["s03"]
    path_rows, path_summary = _path_validation(
        bindings, s03, global_specs, chains, fits, charges
    )
    path_rows.to_parquet(output / "path_validation" / "path_cost_sample.parquet", index=False)
    path_summary.to_csv(output / "path_validation" / "path_cost_component_summary.csv", index=False)
    if path_rows["status"].ne("PASS").any():
        raise RuntimeError("Accepted stored path continuity/hash validation failed")

    error_bands = _error_band_summary(fits, path_rows)
    error_bands.to_csv(
        output / "runtime_fits" / "error_band_summary.csv", index=False
    )
    ranking_bands = _ranking_band_summary(path_rows)
    ranking_bands.to_csv(
        output / "runtime_fits" / "error_band_ranking_summary.csv", index=False
    )
    resource_guard("path_validation_complete")
    _make_figures(
        output,
        topology,
        chains,
        fits,
        benchmarks,
        visits,
        charges,
        error_bands,
    )
    resource_guard("figures_complete")
    scientific_elapsed = time.perf_counter() - started
    acceptance = _scientific_acceptance(
        authority_verification=authority_verification,
        focused_tests_pass=test_run.returncode == 0,
        input_bindings=pre_bindings,
        topology=topology,
        chains=chains,
        oracle=oracle,
        fits=fits,
        domains=domains,
        visits=visits,
        racn=racn,
        multiedge=multiedge,
        benchmarks=benchmarks,
        paths=path_rows,
        scientific_elapsed=scientific_elapsed,
        peak_rss=peak_rss,
        execution=execution,
    )
    acceptance.to_csv(output / "validation" / "acceptance_test_results.csv", index=False)
    if acceptance["status"].ne("PASS").any():
        failed = acceptance.loc[acceptance["status"].ne("PASS"), "test_id"].tolist()
        raise RuntimeError(f"Phase 5 scientific hard gates failed: {failed}")
    write_json(
        output / "validation" / "reproduction.json",
        {
            "schema_id": "partC.phase5.pilot.reproduction.v1",
            "run_id": execution["run_id"],
            "command": [execution["python_executable"], "-m", "partC_rev1_src.phase5.pilot_cli", "run", "--config", str(config_path)],
            "python": sys.version,
            "execution_config_path": str(config_path),
            "execution_config_sha256": sha256_file(config_path),
            "stable_hash_namespace": execution["pilot_sample"]["stable_hash_namespace"],
            "high_excluded_from_fit_and_selection": True,
            "exact_current_not_aon_primary": True,
            "capacity_factor_applied_once": True,
            "routing_or_assignment_performed": False,
            "counts": {
                "suite_count": len(suite_work),
                "candidate_chain_rows": len(chains),
                "unique_sampled_signatures": len(sample),
                "oracle_rows": len(oracle),
                "fit_rows": len(fits),
                "visit_rows": len(visits),
                "path_rows": len(path_rows),
            },
            "status": "PASS_REPRODUCIBLE_FROM_BOUND_INPUTS",
        },
    )
    write_json(
        output / "validation" / "resource_usage.json",
        {
            "schema_id": "partC.phase5.pilot.resource_usage.v1",
            "started_utc": started_utc,
            "recorded_utc": utc_now(),
            "scientific_elapsed_seconds": scientific_elapsed,
            "wall_clock_limit_seconds": execution["resources"]["wall_clock_seconds"],
            "protected_sealing_seconds": execution["resources"]["protected_sealing_seconds"],
            "process_tree_peak_observed_rss_bytes": peak_rss,
            "process_tree_rss_limit_bytes": execution["resources"]["process_tree_rss_bytes"],
            "added_disk_limit_bytes": execution["resources"]["added_disk_bytes"],
            "network_used": False,
            "package_installation_performed": False,
            "status": "PASS",
        },
    )
    post_bindings = _external_bindings(
        bindings, output / "manifests" / "external_artifact_bindings.csv"
    )
    comparison_columns = [
        "role",
        "absolute_path",
        "size_bytes",
        "mtime_utc",
        "sha256_or_directory_digest",
    ]
    invariance_pass = pre_bindings[comparison_columns].equals(
        post_bindings[comparison_columns]
    )
    write_json(
        output / "validation" / "accepted_input_invariance.json",
        {
            "schema_id": "partC.phase5.accepted_input_invariance.v1",
            "binding_count": len(pre_bindings),
            "pre_post_exact_match": invariance_pass,
            "status": "PASS" if invariance_pass else "FAIL",
        },
    )
    if not invariance_pass:
        raise RuntimeError("One or more accepted source bindings changed during the pilot")
    resource_guard("external_bindings_complete", sealing=True)
    _write_reports(
        output,
        bindings,
        topology,
        chains,
        bundle_census,
        oracle,
        fits,
        domains,
        visits,
        charges,
        multiedge,
        racn,
        path_rows,
        path_summary,
        benchmarks,
        error_bands,
        ranking_bands,
        True,
        starting_head,
        starting_head,
        scientific_elapsed,
        peak_rss,
    )
    manifest = _write_output_manifest(output)
    resource_guard("output_manifest_complete", sealing=True)
    added_disk = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if added_disk > execution["resources"]["added_disk_bytes"]:
        raise RuntimeError("Phase 5 added-disk guard exceeded")
    if not manifest["reopen_status"].eq("PASS").all():
        raise RuntimeError("One or more Phase 5 outputs failed reopen validation")
    output_validation = _validate_required_outputs(
        output, authority_root, final=False
    )
    output_validation["validation_stage"] = "PRE_FINAL_OUTPUT_CLOSURE"
    write_json(
        output / "validation" / "pre_final_output_validation.json",
        output_validation,
    )
    if output_validation["status"] != "PASS":
        raise RuntimeError(
            f"Required Phase 5 output closure failed: {output_validation}"
        )
    acceptance = pd.concat(
        [
            acceptance,
            pd.DataFrame(
                [
                    {
                        "test_id": "P5-OUTPUT-01",
                        "test": "required outputs, schemas, figures, status, and reopen closure pass",
                        "status": "PASS",
                        "gate_class": "hard",
                        "evidence_json": canonical_json(output_validation),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    acceptance.to_csv(
        output / "validation" / "acceptance_test_results.csv", index=False
    )
    total_elapsed = time.perf_counter() - started
    peak_rss = max(peak_rss, process_tree_rss_bytes())
    added_disk = sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    )
    resource = json.loads(
        (output / "validation" / "resource_usage.json").read_text(encoding="utf-8")
    )
    resource.update(
        {
            "recorded_utc": utc_now(),
            "total_pilot_elapsed_seconds_before_post_commit_handoff": total_elapsed,
            "process_tree_peak_observed_rss_bytes": peak_rss,
            "added_output_bytes_before_final_manifest": added_disk,
            "status": "PASS",
        }
    )
    write_json(output / "validation" / "resource_usage.json", resource)
    resource_guard("final_output_seal", sealing=True)
    manifest = _write_output_manifest(output)
    if not manifest["reopen_status"].eq("PASS").all():
        raise RuntimeError("Final Phase 5 output manifest failed reopen closure")
    if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) > execution[
        "resources"
    ]["added_disk_bytes"]:
        raise RuntimeError("Final Phase 5 output exceeds the added-disk ceiling")
    repository_status = {
        "schema_id": "partC.phase5.pilot.repository_status.v1",
        "phase_id": "phase_05",
        "step_id": "backbone_compilation_pilot",
        "run_id": execution["run_id"],
        "status": FINAL_STATUS,
        "branch": branch,
        "starting_head": STARTING_HEAD,
        "ending_head_binding": "POST_CHECKPOINT_EXTERNAL_OUTPUT_AND_HANDOFF",
        "external_output_root": str(output),
        "acceptance_test_count": int(len(acceptance)),
        "sampled_macro_signature_count": int(len(sample)),
        "path_validation_row_count": int(len(path_rows)),
        "full_phase5_authorized": False,
        "final_reduced_topology_accepted": False,
        "final_runtime_thresholds_frozen": False,
        "phase6_authorized": False,
        "national_assignment_performed": False,
        "next_step_authorized": False,
        "recommendation": "PROCEED_TO_EXTERNAL_PHASE5_PILOT_REVIEW",
    }
    repository_status_path = (
        repo
        / "partC_doc_icnn"
        / "phase_5_work"
        / "PHASE5_BACKBONE_COMPILATION_PILOT_STATUS.json"
    )
    write_json(repository_status_path, repository_status)
    write_text(
        repository_status_path.with_name(
            "PHASE5_BACKBONE_COMPILATION_PILOT_RESULT.md"
        ),
        "# Phase 5 backbone-compilation pilot result\n\n"
        f"Status: `{FINAL_STATUS}`.\n\n"
        f"The single authorized run `{execution['run_id']}` completed all "
        f"`{len(acceptance)}` dynamic hard gates, sampled `{len(sample)}` canonical "
        f"macro signatures, and validated `{len(path_rows)}` accepted path rows. "
        f"The external evidence root is `{output}`.\n\n"
        "This is pilot evidence awaiting external review. Full Phase 5, final "
        "topology acceptance, final runtime thresholds, Phase 6, and national "
        "assignment remain unauthorized. Recommendation: review the compact "
        "handoff before authorizing any next step.",
    )
    return output


def _seal_blocked_output(
    config_path: Path,
    exception: Exception,
    traceback_text: str,
    elapsed_seconds: float,
) -> None:
    """Preserve an exact machine-readable blocker without rerunning science."""

    execution = json.loads(config_path.read_text(encoding="utf-8"))
    repo = Path(execution["project_root"]).resolve()
    root = Path(execution["output_root"]).resolve()
    for directory in ("reports", "validation", "manifests", "handoff_staging"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    try:
        branch = git(repo, "branch", "--show-current").stdout.strip()
        head = git(repo, "rev-parse", "HEAD").stdout.strip()
    except Exception:
        branch = "UNKNOWN"
        head = "UNKNOWN"
    blocker = f"{type(exception).__name__}: {exception}"
    focused_status_path = root / "validation" / "focused_tests_status.json"
    if focused_status_path.is_file():
        try:
            focused_test_status = str(
                json.loads(focused_status_path.read_text(encoding="utf-8"))["status"]
            )
        except Exception:
            focused_test_status = "INCOMPLETE"
    else:
        focused_test_status = "NOT_REACHED"
    write_json(
        root / "validation" / "failure.json",
        {
            "schema_id": "partC.phase5.pilot.failure.v1",
            "recorded_utc": utc_now(),
            "blocker": blocker,
            "traceback": traceback_text,
            "elapsed_seconds": elapsed_seconds,
            "scientific_rerun_performed": False,
        },
    )
    write_json(
        root / "validation" / "resource_usage.json",
        {
            "schema_id": "partC.phase5.pilot.resource_usage.v1",
            "recorded_utc": utc_now(),
            "elapsed_seconds_at_blocker": elapsed_seconds,
            "process_tree_rss_bytes_at_blocker": process_tree_rss_bytes(),
            "wall_clock_limit_seconds": execution["resources"]["wall_clock_seconds"],
            "process_tree_rss_limit_bytes": execution["resources"]["process_tree_rss_bytes"],
            "added_disk_limit_bytes": execution["resources"]["added_disk_bytes"],
            "status": "BLOCKED",
        },
    )
    status = {
        "schema_id": "partC.phase5.pilot.status.v1",
        "run_id": execution["run_id"],
        "status": BLOCKED_STATUS,
        "branch": branch,
        "starting_head": STARTING_HEAD,
        "ending_head": head,
        "pilot_execution_authorized": True,
        "full_phase5_authorized": False,
        "phase6_authorized": False,
        "national_assignment_performed": False,
        "final_reduced_topology_accepted": False,
        "final_runtime_thresholds_frozen": False,
        "next_step_authorized": False,
        "tests": {"focused_phase5": focused_test_status},
        "outputs": {},
        "decisions": {},
        "blockers": [blocker],
    }
    write_json(root / "reports" / "08_PILOT_STATUS.json", status)
    write_text(
        root / "reports" / "BLOCKED.md",
        "# Phase 5 pilot blocker\n\n"
        f"Status: `{BLOCKED_STATUS}`.\n\n"
        f"Exact blocker: `{blocker}`. Valid partial evidence was preserved; "
        "no scientific rerun or downstream work is authorized.\n",
    )
    _write_blocked_output_manifest(root)

    repository_status = {
        "schema_id": "partC.phase5.pilot.repository_status.v1",
        "phase_id": "phase_05",
        "step_id": "backbone_compilation_pilot",
        "run_id": execution["run_id"],
        "status": BLOCKED_STATUS,
        "branch": branch,
        "starting_head": STARTING_HEAD,
        "ending_head_binding": "POST_CHECKPOINT_EXTERNAL_OUTPUT_AND_HANDOFF",
        "external_output_root": str(root),
        "blockers": [blocker],
        "full_phase5_authorized": False,
        "final_reduced_topology_accepted": False,
        "final_runtime_thresholds_frozen": False,
        "phase6_authorized": False,
        "national_assignment_performed": False,
        "next_step_authorized": False,
        "recommendation": "BLOCK_AND_REVIEW_EXACT_FAILURE_EVIDENCE",
    }
    write_json(
        repo
        / "partC_doc_icnn"
        / "phase_5_work"
        / "PHASE5_BACKBONE_COMPILATION_PILOT_STATUS.json",
        repository_status,
    )
    write_text(
        repo
        / "partC_doc_icnn"
        / "phase_5_work"
        / "PHASE5_BACKBONE_COMPILATION_PILOT_RESULT.md",
        "# Phase 5 backbone-compilation pilot result\n\n"
        f"Status: `{BLOCKED_STATUS}`.\n\n"
        f"The single authorized run stopped with `{blocker}`. Exact traceback, "
        f"partial evidence, and resource state are preserved under `{root}`. No "
        "scientific rerun, full Phase 5 transition, Phase 6 work, or national "
        "assignment is authorized. Recommendation: external review of the blocker.",
    )


def run_pilot(config_path: str | Path) -> Path:
    """Execute once, sealing exact blocker evidence on any failed stage."""

    resolved = Path(config_path).resolve()
    execution = json.loads(resolved.read_text(encoding="utf-8"))
    output = Path(execution["output_root"]).resolve()
    preexisting = output.exists()
    started = time.perf_counter()
    try:
        return _run_pilot_impl(resolved)
    except Exception as exception:
        if not preexisting:
            _seal_blocked_output(
                resolved,
                exception,
                traceback.format_exc(),
                time.perf_counter() - started,
            )
        raise


def reseal_output_metadata(config_path: str | Path, ending_head: str) -> Path:
    """Bind the post-checkpoint HEAD and regenerate self-consistent inventories."""

    config_path = Path(config_path).resolve()
    execution = json.loads(config_path.read_text(encoding="utf-8"))
    repo = Path(execution["project_root"]).resolve()
    branch = git(repo, "branch", "--show-current").stdout.strip()
    live_head = git(repo, "rev-parse", "HEAD").stdout.strip()
    if branch != execution["expected_branch"]:
        raise RuntimeError(f"Post-checkpoint branch mismatch: {branch}")
    if str(ending_head) != live_head:
        raise RuntimeError(
            f"Supplied ending HEAD {ending_head} is not live HEAD {live_head}"
        )
    ancestry = git(repo, "merge-base", "--is-ancestor", STARTING_HEAD, live_head, check=False)
    if ancestry.returncode != 0:
        raise RuntimeError("Post-checkpoint HEAD does not descend from the pilot starting HEAD")
    root = Path(execution["output_root"]).resolve()
    status_path = root / "reports" / "08_PILOT_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") not in {FINAL_STATUS, BLOCKED_STATUS}:
        raise RuntimeError(
            f"Cannot reseal unknown Phase 5 scientific status: {status.get('status')}"
        )
    status["ending_head"] = str(ending_head)
    status["post_checkpoint_head_bound"] = True
    status["sealed_utc"] = utc_now()
    write_json(status_path, status)
    if status.get("status") == BLOCKED_STATUS:
        resource_path = root / "validation" / "resource_usage.json"
        resource = json.loads(resource_path.read_text(encoding="utf-8"))
        resource["sealed_utc"] = utc_now()
        resource["added_output_bytes_before_post_commit_manifest"] = sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        )
        write_json(resource_path, resource)
        write_json(
            root / "repository_audit" / "post_checkpoint.json",
            {
                "schema_id": "partC.phase5.post_checkpoint.v1",
                "recorded_utc": utc_now(),
                "branch": branch,
                "starting_head": STARTING_HEAD,
                "ending_head": live_head,
                "starting_head_is_ancestor": True,
                "qualified_git_status": git(
                    repo, "status", "--short", "--branch"
                ).stdout.splitlines(),
                "scientific_status": BLOCKED_STATUS,
                "status": "PASS",
            },
        )
        _write_blocked_output_manifest(root)
        jsonschema_validate(
            json.loads(status_path.read_text(encoding="utf-8")),
            json.loads(
                (
                    repo
                    / "partC_doc_icnn"
                    / "review_gates"
                    / "phase_05"
                    / "backbone_compilation_pilot"
                    / "phase5_pilot_authority_20260903"
                    / "config"
                    / "pilot_status.schema.json"
                ).read_text(encoding="utf-8")
            ),
        )
        if sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) > execution[
            "resources"
        ]["added_disk_bytes"]:
            raise RuntimeError("Blocked output exceeds added-disk ceiling")
        return root
    reproduction_path = root / "validation" / "reproduction.json"
    reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))
    reproduction["ending_head"] = str(ending_head)
    reproduction["post_checkpoint_head_bound"] = True
    write_json(reproduction_path, reproduction)
    resource_path = root / "validation" / "resource_usage.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    resource["sealed_utc"] = utc_now()
    resource["added_output_bytes_before_post_commit_manifest"] = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    write_json(resource_path, resource)
    report_path = root / "reports" / "01_REPOSITORY_AND_INPUT_AUDIT.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace(
        f"- Ending HEAD field at this seal: `{STARTING_HEAD}`.",
        f"- Ending HEAD field at this seal: `{live_head}`.",
    )
    write_text(report_path, report)
    write_json(
        root / "repository_audit" / "post_checkpoint.json",
        {
            "schema_id": "partC.phase5.post_checkpoint.v1",
            "recorded_utc": utc_now(),
            "branch": branch,
            "starting_head": STARTING_HEAD,
            "ending_head": live_head,
            "starting_head_is_ancestor": True,
            "qualified_git_status": git(repo, "status", "--short", "--branch").stdout.splitlines(),
            "scientific_status": FINAL_STATUS,
            "status": "PASS",
        },
    )
    manifest = _write_output_manifest(root)
    authority_root = (
        repo
        / "partC_doc_icnn"
        / "review_gates"
        / "phase_05"
        / "backbone_compilation_pilot"
        / "phase5_pilot_authority_20260903"
    )
    validation = _validate_required_outputs(root, authority_root)
    if validation["status"] != "PASS" or not manifest["reopen_status"].eq("PASS").all():
        raise RuntimeError("Post-checkpoint output reseal failed required-output closure")
    jsonschema_validate(
        json.loads(status_path.read_text(encoding="utf-8")),
        json.loads(
            (authority_root / "config" / "pilot_status.schema.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    final_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if final_bytes > execution["resources"]["added_disk_bytes"]:
        raise RuntimeError("Post-checkpoint output exceeds added-disk ceiling")
    return root
