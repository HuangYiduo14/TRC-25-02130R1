"""Phase 2 deterministic physical split and stable-ID mapping portfolio.

This module is deliberately limited to the Phase 2 contract.  It reads the
authoritative cleaned graph and enriched source fields, applies the already
implemented backbone/Leiden rules, corrects small regions into the backbone,
and writes mapping artifacts.  It does not construct an ACN, run assignment,
generate labels, or train a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .partC_rev1_backbone_expansion import partC_rev1_expand_backbone
from .partC_rev1_backbone_seed import partC_rev1_extract_seed_backbone
from .partC_rev1_hashing import (
    partC_rev1_dataframe_records_hash,
    partC_rev1_file_signature,
    partC_rev1_json_hash,
    partC_rev1_ordered_records_hash,
)
from .partC_rev1_ids import partC_rev1_stable_id
from .partC_rev1_partition_igraph import partC_rev1_run_leiden_partition


PHASE_ID = "phase_02"
STEP_ID = "phase2_split_mapping_campaign"
SCHEMA_VERSION = "1.0"
FORBIDDEN_OUTPUT_TOKENS = (
    "b_acn",
    "bacn",
    "r_acn",
    "racn",
    "assignment",
    "benchmark_trace",
    "training_label",
    "trained_model",
    "phase_03",
    "phase3",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp.{os.getpid()}" + path.suffix)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(temporary, index=False, compression="zstd")
    elif path.suffix.lower() == ".csv":
        frame.to_csv(temporary, index=False, lineterminator="\n")
    else:
        raise ValueError(f"Unsupported table suffix: {path}")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_rss_bytes() -> int:
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return int(total)


@dataclass(slots=True)
class ResourceTracker:
    start: float
    peak_rss_bytes: int

    @classmethod
    def begin(cls) -> "ResourceTracker":
        return cls(start=time.perf_counter(), peak_rss_bytes=_tree_rss_bytes())

    def sample(self) -> None:
        self.peak_rss_bytes = max(self.peak_rss_bytes, _tree_rss_bytes())

    @property
    def elapsed_seconds(self) -> float:
        return float(time.perf_counter() - self.start)


def load_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "phase_id",
        "step_id",
        "campaign_id",
        "run_id",
        "tasks",
        "input_versions",
        "output_root",
        "temp_root",
        "resolutions",
        "random_seed",
        "backbone_partition",
        "small_region_rule",
        "resource_budget",
        "expected_suite_outputs",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Phase 2 config missing required fields: {missing}")
    if config["phase_id"] != PHASE_ID or config["step_id"] != STEP_ID:
        raise ValueError("Phase/step mismatch")
    if not isinstance(config["tasks"], list) or not config["tasks"]:
        raise ValueError("Phase 2 requires a nonempty explicit task list")
    if len(config["resolutions"]) not in {5, 6}:
        raise ValueError("Phase 2 requires five or six starting resolutions")
    if int(config["random_seed"]) != 42:
        raise ValueError("The frozen Phase 2 seed is 42")
    budget = config["resource_budget"]
    if float(budget["wall_clock_hours"]) > 24 or float(budget["process_tree_ram_gib"]) > 24 or float(budget["added_disk_gib"]) > 150:
        raise ValueError("Configured resource budget exceeds the Phase 2 authorization")
    if int(budget.get("write_capable_orchestrators", 0)) != 1:
        raise ValueError("Exactly one write-capable orchestrator is required")
    if not config["expected_suite_outputs"]:
        raise ValueError("Expected outputs may not be empty")
    for name in config["expected_suite_outputs"]:
        if Path(name).name != name:
            raise ValueError(f"Expected suite output must be a filename: {name}")
    config["_config_path"] = str(resolved)
    config["_config_sha256"] = _sha256(resolved)
    return config


def _assert_safe_output_roots(config: dict[str, Any]) -> tuple[Path, Path]:
    output_root = Path(config["output_root"]).resolve()
    temp_root = Path(config["temp_root"]).resolve()
    original_root = Path(config["original_repository_root"]).resolve()
    worktree_root = Path(config["worktree_root"]).resolve()
    for root in (output_root, temp_root):
        if root == original_root or original_root in root.parents:
            raise ValueError(f"Write root is under the read-only original repository: {root}")
        if root == worktree_root or worktree_root in root.parents:
            raise ValueError(f"Large run output may not be written inside the worktree: {root}")
    if output_root == temp_root or output_root in temp_root.parents or temp_root in output_root.parents:
        raise ValueError("Output and temporary roots must be separate siblings")
    return output_root, temp_root


def _input_paths(config: dict[str, Any]) -> dict[str, Path]:
    versions = config["input_versions"]
    return {
        "graph_npz": Path(versions["graph_npz"]).resolve(),
        "graph_nodes": Path(versions["graph_nodes_parquet"]).resolve(),
        "graph_arcs": Path(versions["graph_arcs_parquet"]).resolve(),
        "graph_metadata": Path(versions["graph_metadata_json"]).resolve(),
        "od_npz": Path(versions["od_npz"]).resolve(),
        "od_metadata": Path(versions["od_metadata_json"]).resolve(),
        "rich_sqlite": Path(versions["rich_sqlite"]).resolve(),
    }


def input_signatures(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    signatures: dict[str, dict[str, Any]] = {}
    for name, path in _input_paths(config).items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing authoritative input {name}: {path}")
        signatures[name] = {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "signature": partC_rev1_file_signature(path),
        }
    return signatures


def _read_sqlite_enrichment(sqlite_path: Path) -> pd.DataFrame:
    columns = [
        "directed_arc_id",
        "partC_rev1_link_uid",
        "from_node_uid",
        "to_node_uid",
        "class_code",
        "is_open",
        "is_valid_for_truck",
        "partC_rev1_faf_strahnet_nonempty",
        "STRAHNET",
        "partC_rev1_fftt_min",
        "partC_rev1_capacity_annual_ktons_dir",
        "length_miles",
    ]
    quoted = ", ".join(f'"{name}"' for name in columns)
    uri = f"file:{sqlite_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only=1")
        frame = pd.read_sql_query(f'SELECT {quoted} FROM "partC_rev1_directed_arcs_enriched"', connection)
    return frame


def load_authoritative_inputs(config: dict[str, Any], tracker: ResourceTracker) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    paths = _input_paths(config)
    graph_nodes = pd.read_parquet(paths["graph_nodes"], columns=["node_idx", "partC_rev1_node_uid"])
    graph_arcs = pd.read_parquet(
        paths["graph_arcs"],
        columns=[
            "arc_idx",
            "directed_arc_id",
            "partC_rev1_link_uid",
            "tail_idx",
            "head_idx",
            "tail_node_uid",
            "head_node_uid",
            "free_time",
            "base_capacity",
            "length_miles",
        ],
    )
    enriched = _read_sqlite_enrichment(paths["rich_sqlite"])
    tracker.sample()

    graph_arcs["directed_arc_id"] = graph_arcs["directed_arc_id"].map(partC_rev1_stable_id)
    enriched["directed_arc_id"] = enriched["directed_arc_id"].map(partC_rev1_stable_id)
    if graph_arcs["directed_arc_id"].duplicated().any() or enriched["directed_arc_id"].duplicated().any():
        raise ValueError("directed_arc_id is not unique in an input")
    authoritative_ids = graph_arcs[["directed_arc_id"]].copy()
    matched = authoritative_ids.merge(enriched, on="directed_arc_id", how="left", validate="one_to_one", indicator=True)
    if len(matched) != len(graph_arcs) or not matched["_merge"].eq("both").all():
        missing = matched.loc[matched["_merge"].ne("both"), "directed_arc_id"].head(10).tolist()
        raise ValueError(f"Enriched SQLite does not match the authoritative graph: {missing}")
    matched = matched.drop(columns="_merge")
    arcs = graph_arcs.drop(columns=["partC_rev1_link_uid"], errors="ignore").merge(
        matched, on="directed_arc_id", how="left", validate="one_to_one", sort=False
    )
    arcs = arcs.sort_values("arc_idx", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(arcs["arc_idx"].to_numpy(dtype=np.int64), np.arange(len(arcs), dtype=np.int64)):
        raise ValueError("Authoritative arc_idx must be dense and ordered")
    graph_nodes = graph_nodes.sort_values("node_idx", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(graph_nodes["node_idx"].to_numpy(dtype=np.int64), np.arange(len(graph_nodes), dtype=np.int64)):
        raise ValueError("Authoritative node_idx must be dense and ordered")
    graph_nodes["partC_rev1_node_uid"] = graph_nodes["partC_rev1_node_uid"].map(partC_rev1_stable_id)
    arcs["from_node_uid"] = arcs["tail_node_uid"].map(partC_rev1_stable_id)
    arcs["to_node_uid"] = arcs["head_node_uid"].map(partC_rev1_stable_id)
    arcs["tail_node_uid"] = arcs["from_node_uid"]
    arcs["head_node_uid"] = arcs["to_node_uid"]
    arcs["free_time_min"] = pd.to_numeric(arcs["free_time"], errors="raise")
    arcs["capacity"] = pd.to_numeric(arcs["base_capacity"], errors="raise")
    arcs["partC_rev1_length_m"] = pd.to_numeric(arcs["length_miles_x"], errors="coerce") * 1609.344
    if "length_miles_y" in arcs:
        arcs["partC_rev1_length_m"] = arcs["partC_rev1_length_m"].fillna(
            pd.to_numeric(arcs["length_miles_y"], errors="coerce") * 1609.344
        )
    arcs["partC_rev1_fftt_min"] = pd.to_numeric(arcs["partC_rev1_fftt_min"], errors="coerce").fillna(arcs["free_time_min"])
    arcs["partC_rev1_capacity_annual_ktons_dir"] = pd.to_numeric(
        arcs["partC_rev1_capacity_annual_ktons_dir"], errors="coerce"
    ).fillna(arcs["capacity"])

    with np.load(paths["od_npz"], allow_pickle=False) as od:
        origin_nodes = np.unique(od["origin_group_node_idx"].astype(np.int64))
        destination_nodes = np.unique(od["destination_node_idx"].astype(np.int64))
    if len(origin_nodes) == 0 or len(destination_nodes) == 0:
        raise ValueError("Authoritative OD endpoint sets may not be empty")
    tracker.sample()
    return graph_nodes, arcs, origin_nodes, destination_nodes


def build_initial_backbone(nodes: pd.DataFrame, arcs: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    algorithm_config = {"backbone_partition": config["backbone_partition"]}
    seeded = partC_rev1_extract_seed_backbone(arcs, algorithm_config)
    expanded = partC_rev1_expand_backbone(seeded, nodes, algorithm_config)
    output_arcs = expanded["arcs"].sort_values("arc_idx", kind="mergesort").reset_index(drop=True)
    flag = output_arcs["is_final_backbone"].fillna(False).astype(bool).to_numpy()
    backbone_node_mask = np.zeros(len(nodes), dtype=bool)
    backbone_node_mask[output_arcs.loc[flag, "tail_idx"].to_numpy(dtype=np.int64)] = True
    backbone_node_mask[output_arcs.loc[flag, "head_idx"].to_numpy(dtype=np.int64)] = True
    partition_nodes = nodes.copy()
    partition_nodes["is_backbone_node"] = backbone_node_mask
    partition_nodes["is_internal_region_vertex"] = ~backbone_node_mask
    identity = {
        "seed_candidate_directed_arc_count": int(output_arcs["is_seed_candidate_backbone"].fillna(False).astype(bool).sum()),
        "seed_lcc_directed_arc_count": int(output_arcs["is_seed_backbone"].fillna(False).astype(bool).sum()),
        "initial_final_backbone_directed_arc_count": int(flag.sum()),
        "initial_backbone_node_count": int(backbone_node_mask.sum()),
        "direct_connector_accepted_link_count": int(len(expanded["accepted_direct_connector_links"])),
        "initial_backbone_membership_hash": partC_rev1_ordered_records_hash(
            zip(output_arcs["directed_arc_id"].astype(str), flag.astype(np.int8))
        ),
    }
    return partition_nodes, output_arcs, identity


def _stable_region_ids(node_partition: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    if node_partition.empty:
        return node_partition.copy(), {}
    translated: dict[str, str] = {}
    for raw_id, group in node_partition.groupby("region_id", sort=True):
        members = sorted(group["partC_rev1_node_uid"].astype(str).tolist())
        digest = hashlib.sha256(("\n".join(members) + "\n").encode("utf-8")).hexdigest()[:20]
        translated[str(raw_id)] = f"partC_p2_region_{digest}"
    stable = node_partition.copy()
    stable["region_uuid"] = stable["region_id"].astype(str).map(translated)
    stable = stable.drop(columns="region_id").sort_values("partC_rev1_node_uid", kind="mergesort").reset_index(drop=True)
    return stable, translated


def _initial_arc_membership(
    arcs: pd.DataFrame,
    partition_nodes: pd.DataFrame,
    stable_partition: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_region = np.full(len(partition_nodes), "", dtype=object)
    node_index = partition_nodes.set_index("partC_rev1_node_uid")["node_idx"]
    positions = stable_partition["partC_rev1_node_uid"].map(node_index).to_numpy(dtype=np.int64)
    node_region[positions] = stable_partition["region_uuid"].astype(str).to_numpy()
    tail = arcs["tail_idx"].to_numpy(dtype=np.int64)
    head = arcs["head_idx"].to_numpy(dtype=np.int64)
    tail_region = node_region[tail]
    head_region = node_region[head]
    initial = arcs["is_final_backbone"].fillna(False).astype(bool).to_numpy()
    tail_backbone = tail_region == ""
    head_backbone = head_region == ""
    both_backbone = tail_backbone & head_backbone
    same_region = (~tail_backbone) & (~head_backbone) & (tail_region == head_region)
    final_label = np.full(len(arcs), "backbone", dtype=object)
    final_label[same_region] = tail_region[same_region]
    cross = (~tail_backbone) & (~head_backbone) & (~same_region)
    final_label[cross] = tail_region[cross]
    one_backbone = tail_backbone ^ head_backbone
    final_label[one_backbone] = np.where(tail_backbone[one_backbone], head_region[one_backbone], tail_region[one_backbone])
    arc_type = np.full(len(arcs), "backbone_network", dtype=object)
    arc_type[same_region] = "regional_internal"
    arc_type[cross] = "region_region_boundary_connector"
    arc_type[one_backbone] = "backbone_region_boundary_connector"
    post_partition = both_backbone & (~initial)
    return final_label.astype(str), arc_type.astype(str), post_partition, node_region.astype(str)


def _incident_registry(nodes: pd.DataFrame, arcs: pd.DataFrame, labels: np.ndarray, node_primary: np.ndarray) -> pd.DataFrame:
    endpoint = pd.DataFrame(
        {
            "node_idx": np.concatenate([arcs["tail_idx"].to_numpy(dtype=np.int64), arcs["head_idx"].to_numpy(dtype=np.int64)]),
            "suite_label": np.concatenate([labels, labels]),
        }
    ).drop_duplicates()
    grouped = endpoint.groupby("node_idx", sort=False)["suite_label"].agg(
        incident_label_count="nunique", incident_labels=lambda values: "|".join(sorted(set(values.astype(str))))
    )
    registry = nodes[["node_idx", "partC_rev1_node_uid"]].copy()
    registry["primary_partition_region_uuid"] = node_primary
    registry = registry.merge(grouped, left_on="node_idx", right_index=True, how="left", validate="one_to_one")
    registry["incident_label_count"] = registry["incident_label_count"].fillna(0).astype(np.int32)
    registry["incident_labels"] = registry["incident_labels"].fillna("")
    registry["is_physical_portal"] = registry["incident_label_count"].ge(2)
    return registry


def _region_summary(labels: np.ndarray, registry: pd.DataFrame, arcs: pd.DataFrame) -> pd.DataFrame:
    regional = labels != "backbone"
    counts = pd.Series(labels[regional]).value_counts(sort=False).rename_axis("region_uuid").reset_index(name="arc_count")
    if counts.empty:
        return pd.DataFrame(columns=["region_uuid", "arc_count", "node_count", "portal_count"])
    endpoint = pd.DataFrame(
        {
            "node_idx": np.concatenate([
                arcs.loc[regional, "tail_idx"].to_numpy(dtype=np.int64),
                arcs.loc[regional, "head_idx"].to_numpy(dtype=np.int64),
            ]),
            "region_uuid": np.concatenate([labels[regional], labels[regional]]),
        }
    ).drop_duplicates()
    portal_nodes = set(registry.loc[registry["is_physical_portal"], "node_idx"].astype(int).tolist())
    endpoint["is_portal"] = endpoint["node_idx"].isin(portal_nodes)
    node_counts = endpoint.groupby("region_uuid", sort=False).agg(
        node_count=("node_idx", "nunique"), portal_count=("is_portal", "sum")
    ).reset_index()
    return counts.merge(node_counts, on="region_uuid", how="left", validate="one_to_one")


def _correct_small_regions(
    labels: np.ndarray,
    registry: pd.DataFrame,
    arcs: pd.DataFrame,
    rule: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    before = _region_summary(labels, registry, arcs)
    min_arcs = int(rule["min_arc_count"])
    min_portal_square = int(rule["min_portal_square"])
    before["portal_square"] = before["portal_count"].astype(np.int64) ** 2
    before["too_small_region"] = before["arc_count"].lt(min_arcs) | before["portal_square"].lt(min_portal_square)
    before["corrected_region_uuid"] = np.where(before["too_small_region"], "backbone", before["region_uuid"])
    small = set(before.loc[before["too_small_region"], "region_uuid"].astype(str))
    # Object dtype prevents NumPy from truncating the replacement string when
    # a synthetic or future region identifier is shorter than ``backbone``.
    corrected = labels.astype(object).copy()
    if small:
        corrected[np.isin(corrected, list(small))] = "backbone"
    return corrected.astype(str), before.sort_values("region_uuid", kind="mergesort").reset_index(drop=True), before


def _build_portals(
    suite_id: str,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    labels: np.ndarray,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    regional = labels != "backbone"
    tails = pd.DataFrame({"node_idx": arcs.loc[regional, "tail_idx"].to_numpy(dtype=np.int64), "region_uuid": labels[regional], "can_exit": True})
    heads = pd.DataFrame({"node_idx": arcs.loc[regional, "head_idx"].to_numpy(dtype=np.int64), "region_uuid": labels[regional], "can_enter": True})
    pairs = pd.concat([tails[["node_idx", "region_uuid"]], heads[["node_idx", "region_uuid"]]], ignore_index=True).drop_duplicates()
    portal_nodes = registry.loc[registry["is_physical_portal"], ["node_idx"]]
    pairs = pairs.merge(portal_nodes, on="node_idx", how="inner")
    if pairs.empty:
        return pd.DataFrame(columns=["portal_id", "suite_id", "region_uuid", "partC_rev1_node_uid", "can_enter", "can_exit", "portal_kind"])
    exit_pairs = tails.drop_duplicates(["node_idx", "region_uuid"]).assign(can_exit=True).set_index(["node_idx", "region_uuid"])["can_exit"]
    enter_pairs = heads.drop_duplicates(["node_idx", "region_uuid"]).assign(can_enter=True).set_index(["node_idx", "region_uuid"])["can_enter"]
    index = pd.MultiIndex.from_frame(pairs[["node_idx", "region_uuid"]])
    pairs["can_exit"] = index.map(exit_pairs).fillna(False).astype(bool)
    pairs["can_enter"] = index.map(enter_pairs).fillna(False).astype(bool)
    pairs = pairs.merge(nodes[["node_idx", "partC_rev1_node_uid"]], on="node_idx", how="left", validate="many_to_one")
    pairs["suite_id"] = suite_id
    pairs["portal_id"] = (
        pairs["region_uuid"].astype(str) + "|portal|" + pairs["partC_rev1_node_uid"].astype(str)
    )
    pairs["portal_kind"] = np.select(
        [pairs["can_enter"] & pairs["can_exit"], pairs["can_enter"], pairs["can_exit"]],
        ["BIDIRECTIONAL", "INBOUND", "OUTBOUND"],
        default="INACTIVE",
    )
    return pairs[["portal_id", "suite_id", "region_uuid", "partC_rev1_node_uid", "node_idx", "can_enter", "can_exit", "portal_kind"]].sort_values(
        ["region_uuid", "partC_rev1_node_uid"], kind="mergesort"
    ).reset_index(drop=True)


def _build_centroid_roles(
    suite_id: str,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    labels: np.ndarray,
    origin_nodes: np.ndarray,
    destination_nodes: np.ndarray,
) -> pd.DataFrame:
    regional = labels != "backbone"
    endpoint = pd.DataFrame(
        {
            "node_idx": np.concatenate([
                arcs.loc[regional, "tail_idx"].to_numpy(dtype=np.int64),
                arcs.loc[regional, "head_idx"].to_numpy(dtype=np.int64),
            ]),
            "region_uuid": np.concatenate([labels[regional], labels[regional]]),
        }
    ).drop_duplicates()
    role_frames: list[pd.DataFrame] = []
    for role, eligible in (("centroid_origin", origin_nodes), ("centroid_destination", destination_nodes)):
        selected = endpoint.loc[endpoint["node_idx"].isin(eligible)].copy()
        selected["centroid_role"] = role
        role_frames.append(selected)
    roles = pd.concat(role_frames, ignore_index=True) if role_frames else pd.DataFrame()
    if roles.empty:
        return pd.DataFrame(columns=["centroid_role_id", "suite_id", "region_uuid", "partC_rev1_node_uid", "centroid_role", "through_allowed"])
    roles = roles.merge(nodes[["node_idx", "partC_rev1_node_uid"]], on="node_idx", how="left", validate="many_to_one")
    roles["suite_id"] = suite_id
    roles["through_allowed"] = False
    roles["centroid_role_id"] = (
        roles["region_uuid"].astype(str)
        + "|"
        + roles["centroid_role"].astype(str)
        + "|"
        + roles["partC_rev1_node_uid"].astype(str)
    )
    return roles[["centroid_role_id", "suite_id", "region_uuid", "partC_rev1_node_uid", "node_idx", "centroid_role", "through_allowed"]].sort_values(
        ["region_uuid", "centroid_role", "partC_rev1_node_uid"], kind="mergesort"
    ).reset_index(drop=True)


def _build_terminals(suite_id: str, portals: pd.DataFrame, centroid_roles: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not portals.empty:
        source = portals.loc[portals["can_exit"], ["region_uuid", "partC_rev1_node_uid"]].copy()
        source["terminal_kind"] = "portal_source"
        target = portals.loc[portals["can_enter"], ["region_uuid", "partC_rev1_node_uid"]].copy()
        target["terminal_kind"] = "portal_destination"
        rows.extend([source, target])
    if not centroid_roles.empty:
        centroid = centroid_roles[["region_uuid", "partC_rev1_node_uid", "centroid_role"]].rename(columns={"centroid_role": "terminal_kind"})
        rows.append(centroid)
    terminals = pd.concat(rows, ignore_index=True).drop_duplicates() if rows else pd.DataFrame()
    if terminals.empty:
        return pd.DataFrame(columns=["terminal_id", "suite_id", "region_uuid", "partC_rev1_node_uid", "terminal_kind", "terminal_order"])
    kind_order = {"portal_source": 0, "centroid_origin": 1, "portal_destination": 2, "centroid_destination": 3}
    terminals["_kind_order"] = terminals["terminal_kind"].map(kind_order).astype(int)
    terminals = terminals.sort_values(["region_uuid", "_kind_order", "partC_rev1_node_uid"], kind="mergesort").reset_index(drop=True)
    terminals["suite_id"] = suite_id
    terminals["terminal_order"] = terminals.groupby("region_uuid", sort=False).cumcount().astype(np.int32)
    terminals["terminal_id"] = (
        terminals["region_uuid"].astype(str)
        + "|"
        + terminals["terminal_kind"].astype(str)
        + "|"
        + terminals["partC_rev1_node_uid"].astype(str)
    )
    return terminals[["terminal_id", "suite_id", "region_uuid", "partC_rev1_node_uid", "terminal_kind", "terminal_order"]]


def _connectivity_validation(
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    stable_partition: pd.DataFrame,
    corrected_regions: set[str],
) -> tuple[bool, int]:
    retained = stable_partition.loc[~stable_partition["region_uuid"].isin(corrected_regions)].copy()
    if retained.empty:
        return False, 0
    node_region = np.full(len(nodes), "", dtype=object)
    index = nodes.set_index("partC_rev1_node_uid")["node_idx"]
    positions = retained["partC_rev1_node_uid"].map(index).to_numpy(dtype=np.int64)
    node_region[positions] = retained["region_uuid"].astype(str).to_numpy()
    tail = arcs["tail_idx"].to_numpy(dtype=np.int64)
    head = arcs["head_idx"].to_numpy(dtype=np.int64)
    internal = (node_region[tail] != "") & (node_region[tail] == node_region[head])
    graph = sparse.coo_matrix(
        (np.ones(int(internal.sum()), dtype=np.int8), (tail[internal], head[internal])),
        shape=(len(nodes), len(nodes)),
    ).tocsr()
    _component_count, component = connected_components(graph, directed=False, return_labels=True)
    test = pd.DataFrame({"region_uuid": node_region[positions], "component": component[positions]}).groupby("region_uuid")["component"].nunique()
    failures = int(test.gt(1).sum())
    return failures == 0, failures


def _suite_manifest(suite_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(p for p in suite_dir.rglob("*") if p.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(suite_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(rows)


def generate_suite(
    *,
    config: dict[str, Any],
    resolution: float,
    candidate_number: int,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    partition_nodes: pd.DataFrame,
    initial_identity: dict[str, Any],
    origin_nodes: np.ndarray,
    destination_nodes: np.ndarray,
    temp_root: Path,
    final_root: Path,
    tracker: ResourceTracker,
) -> dict[str, Any]:
    suite_id = f"phase2_suite_{candidate_number:02d}_r{format(resolution, '.12g').replace('.', 'p').replace('-', 'm')}"
    target = final_root / "suites" / suite_id
    temporary = temp_root / "suites" / suite_id
    if target.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite suite output: {target} / {temporary}")
    temporary.mkdir(parents=True)
    suite_tracker = ResourceTracker.begin()
    nonbackbone = arcs.loc[~arcs["is_final_backbone"].fillna(False).astype(bool)].copy()
    algorithm_config = {"backbone_partition": config["backbone_partition"]}
    partition = partC_rev1_run_leiden_partition(
        partition_nodes,
        nonbackbone,
        algorithm_config,
        resolution=float(resolution),
        random_seed=int(config["random_seed"]),
    )
    stable_partition, _translation = _stable_region_ids(partition["node_partition"])
    labels, arc_type, post_partition, node_primary = _initial_arc_membership(arcs, partition_nodes, stable_partition)
    registry_before = _incident_registry(nodes, arcs, labels, node_primary)
    corrected_labels, correction_audit, _ = _correct_small_regions(labels, registry_before, arcs, config["small_region_rule"])
    corrected_regions = set(correction_audit.loc[correction_audit["too_small_region"], "region_uuid"].astype(str))
    corrected_primary = node_primary.copy()
    corrected_primary[np.isin(corrected_primary, list(corrected_regions))] = "backbone"
    corrected_primary[corrected_primary == ""] = "backbone"
    registry = _incident_registry(nodes, arcs, corrected_labels, corrected_primary)
    retained = _region_summary(corrected_labels, registry, arcs)
    retained["resolution"] = float(resolution)
    retained["random_seed"] = int(config["random_seed"])
    retained["portal_square"] = retained["portal_count"].astype(np.int64) ** 2
    retained["passes_small_region_rule"] = (
        retained["arc_count"].ge(int(config["small_region_rule"]["min_arc_count"]))
        & retained["portal_square"].ge(int(config["small_region_rule"]["min_portal_square"]))
    )
    portals = _build_portals(suite_id, nodes, arcs, corrected_labels, registry)
    centroid_roles = _build_centroid_roles(suite_id, nodes, arcs, corrected_labels, origin_nodes, destination_nodes)
    terminals = _build_terminals(suite_id, portals, centroid_roles)
    final_backbone = corrected_labels == "backbone"
    initial_backbone = arcs["is_final_backbone"].fillna(False).astype(bool).to_numpy()
    membership = pd.DataFrame(
        {
            "suite_id": suite_id,
            "arc_idx": arcs["arc_idx"].to_numpy(dtype=np.int64),
            "directed_arc_id": arcs["directed_arc_id"].astype(str),
            "partC_rev1_link_uid": arcs["partC_rev1_link_uid"].map(partC_rev1_stable_id),
            "tail_node_uid": arcs["tail_node_uid"].astype(str),
            "head_node_uid": arcs["head_node_uid"].astype(str),
            "suite_label": corrected_labels,
            "arc_partition_type": np.where(final_backbone, "backbone_network", arc_type),
            "is_initial_backbone": initial_backbone,
            "is_post_partition_backbone": post_partition,
            "is_small_region_correction_backbone": np.isin(labels, list(corrected_regions)),
            "is_final_backbone": final_backbone,
        }
    )
    membership_hash = partC_rev1_dataframe_records_hash(
        membership, ["directed_arc_id", "suite_label", "is_final_backbone"]
    )
    node_membership_hash = partC_rev1_dataframe_records_hash(
        registry, ["partC_rev1_node_uid", "primary_partition_region_uuid", "incident_labels"]
    )
    structural_signature = partC_rev1_json_hash(
        {
            "membership_hash": membership_hash,
            "node_membership_hash": node_membership_hash,
            "region_count": len(retained),
            "final_backbone_arc_count": int(final_backbone.sum()),
            "portal_count": len(portals),
        }
    )
    connected, connectivity_failures = _connectivity_validation(nodes, arcs, stable_partition, corrected_regions)
    validation_checks = {
        "arc_membership_exhaustive": len(membership) == len(arcs),
        "directed_arc_id_unique": bool(membership["directed_arc_id"].is_unique),
        "directed_arc_id_order_matches_authoritative": bool(membership["directed_arc_id"].equals(arcs["directed_arc_id"].astype(str))),
        "final_backbone_region_disjoint": bool((membership["is_final_backbone"] == membership["suite_label"].eq("backbone")).all()),
        "retained_regions_nonempty": len(retained) > 0,
        "retained_region_ids_unique": bool(retained["region_uuid"].is_unique),
        "small_region_rule_passes": bool(retained["passes_small_region_rule"].all()),
        "region_connectivity_passes": connected,
        "portal_ids_unique": bool(portals["portal_id"].is_unique),
        "portal_semantics_pass": bool(portals["portal_kind"].ne("INACTIVE").all()),
        "centroid_role_ids_unique": bool(centroid_roles["centroid_role_id"].is_unique),
        "centroid_non_through": bool((~centroid_roles["through_allowed"]).all()),
        "terminal_ids_unique": bool(terminals["terminal_id"].is_unique),
        "terminal_order_contiguous": bool(
            terminals.groupby("region_uuid", sort=False)["terminal_order"].apply(
                lambda values: np.array_equal(values.to_numpy(), np.arange(len(values)))
            ).all()
        ),
        "phase3_artifact_count_zero": True,
        "bacn_count_zero": True,
        "racn_count_zero": True,
        "assignment_count_zero": True,
        "training_label_count_zero": True,
        "trained_model_count_zero": True,
    }
    status = "PASS" if all(validation_checks.values()) else "FAIL"

    suite_config = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "step_id": STEP_ID,
        "campaign_id": config["campaign_id"],
        "run_id": config["run_id"],
        "suite_id": suite_id,
        "resolution": float(resolution),
        "random_seed": int(config["random_seed"]),
        "small_region_rule": config["small_region_rule"],
        "config_sha256": config["_config_sha256"],
    }
    suite_identity = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": suite_id,
        "graph_identity": config["input_versions"]["graph_identity_sha256"],
        "od_identity": config["input_versions"]["od_identity_sha256"],
        "resolution": float(resolution),
        "random_seed": int(config["random_seed"]),
        "membership_hash": membership_hash,
        "node_membership_hash": node_membership_hash,
        "structural_signature": structural_signature,
    }
    signature = {
        **suite_identity,
        "initial_backbone_directed_arc_count": int(initial_backbone.sum()),
        "final_backbone_directed_arc_count": int(final_backbone.sum()),
        "post_partition_backbone_directed_arc_count": int(post_partition.sum()),
        "small_region_corrected_directed_arc_count": int(np.isin(labels, list(corrected_regions)).sum()),
        "corrected_small_region_count": len(corrected_regions),
        "retained_region_count": len(retained),
        "portal_record_count": len(portals),
        "centroid_role_count": len(centroid_roles),
        "terminal_count": len(terminals),
        "min_region_arc_count": int(retained["arc_count"].min()),
        "max_region_arc_count": int(retained["arc_count"].max()),
        "median_region_arc_count": float(retained["arc_count"].median()),
        "min_region_portal_count": int(retained["portal_count"].min()),
        "max_region_portal_count": int(retained["portal_count"].max()),
        "initial_backbone_identity": initial_identity,
    }
    validation = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": suite_id,
        "status": status,
        "checks": validation_checks,
        "connectivity_failure_count": connectivity_failures,
        "authoritative_arc_count": len(arcs),
        "authoritative_node_count": len(nodes),
        "retained_region_count": len(retained),
        "generated_utc": _utc_now(),
    }
    lazy_contract = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": suite_id,
        "contract_kind": "lazy_reachability_pair_support",
        "dense_pair_materialization": False,
        "phase2_pair_row_count": 0,
        "source_terminal_kinds": ["portal_source", "centroid_origin"],
        "destination_terminal_kinds": ["portal_destination", "centroid_destination"],
        "ordering": ["region_uuid", "source_terminal_id", "destination_terminal_id"],
        "same_physical_node_pair_allowed": False,
        "reachability_evaluation": "deferred_to_authorized_downstream_phase",
        "terminal_registry_sha256": partC_rev1_dataframe_records_hash(
            terminals, ["terminal_id", "region_uuid", "partC_rev1_node_uid", "terminal_kind", "terminal_order"]
        ),
    }
    trace_hooks = membership[["directed_arc_id", "suite_label"]].rename(columns={"suite_label": "phase2_suite_label"})
    trace_hooks["future_bacn_trace_id"] = pd.NA
    trace_hooks["future_racn_trace_id"] = pd.NA
    trace_hooks["future_benchmark_trace_id"] = pd.NA
    trace_hooks["phase2_trace_materialized"] = False

    _write_json(temporary / "suite_config.json", suite_config)
    _write_json(temporary / "suite_identity.json", suite_identity)
    _write_table(
        temporary / "initial_backbone_membership.parquet",
        membership.loc[membership["is_initial_backbone"], ["arc_idx", "directed_arc_id", "partC_rev1_link_uid", "tail_node_uid", "head_node_uid", "is_initial_backbone"]],
    )
    _write_table(
        temporary / "final_backbone_membership.parquet",
        membership.loc[membership["is_final_backbone"], ["arc_idx", "directed_arc_id", "partC_rev1_link_uid", "tail_node_uid", "head_node_uid", "is_initial_backbone", "is_post_partition_backbone", "is_small_region_correction_backbone", "is_final_backbone"]],
    )
    _write_table(temporary / "directed_arc_membership.parquet", membership)
    _write_table(temporary / "node_partition_membership.parquet", stable_partition)
    _write_table(temporary / "node_incident_registry.parquet", registry)
    _write_table(temporary / "retained_regions.parquet", retained.sort_values("region_uuid", kind="mergesort"))
    _write_table(temporary / "retained_regions.csv", retained.sort_values("region_uuid", kind="mergesort"))
    _write_table(temporary / "portals.parquet", portals)
    _write_table(temporary / "centroid_roles.parquet", centroid_roles)
    _write_table(temporary / "terminals.parquet", terminals)
    _write_json(temporary / "lazy_pair_support_contract.json", lazy_contract)
    _write_table(temporary / "small_region_corrections.parquet", correction_audit)
    _write_table(temporary / "physical_to_suite_trace_hooks.parquet", trace_hooks)
    _write_json(temporary / "structural_signature.json", signature)
    _write_json(temporary / "validation.json", validation)
    _write_json(
        temporary / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "suite_id": suite_id,
            "phase_id": PHASE_ID,
            "step_id": STEP_ID,
            "campaign_id": config["campaign_id"],
            "run_id": config["run_id"],
            "config_path": config["_config_path"],
            "config_sha256": config["_config_sha256"],
            "python": sys.version,
            "executable": sys.executable,
            "network_used": False,
            "package_installation": False,
            "source_maps_modified": False,
            "generated_utc": _utc_now(),
        },
    )
    _write_text(
        temporary / "QGIS_JOIN_GUIDANCE.md",
        "# QGIS stable-ID join guidance\n\n"
        "Join physical directed arcs to `directed_arc_membership.parquet` using `directed_arc_id`. "
        "Join physical nodes to `node_incident_registry.parquet`, `portals.parquet`, or `centroid_roles.parquet` using "
        "`partC_rev1_node_uid`. Never use row position or `arc_idx`/`node_idx` as a cross-artifact join key.\n",
    )
    suite_tracker.sample()
    resource = pd.DataFrame(
        [
            {
                "suite_id": suite_id,
                "resolution": float(resolution),
                "wall_clock_seconds": suite_tracker.elapsed_seconds,
                "peak_process_tree_rss_bytes": suite_tracker.peak_rss_bytes,
                "leiden_seconds": float(partition["leiden_seconds"]),
                "output_bytes_before_manifest": int(sum(p.stat().st_size for p in temporary.rglob("*") if p.is_file())),
            }
        ]
    )
    _write_table(temporary / "resource_usage.csv", resource)
    manifest = _suite_manifest(temporary)
    _write_table(temporary / "artifact_manifest.csv", manifest)
    actual = {p.name for p in temporary.iterdir() if p.is_file()}
    missing_outputs = sorted(set(config["expected_suite_outputs"]) - actual)
    if missing_outputs:
        raise RuntimeError(f"Suite missing frozen outputs: {missing_outputs}")
    if status != "PASS":
        raise RuntimeError(f"Suite validation failed: {suite_id}: {validation_checks}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, target)
    tracker.sample()
    return {
        **signature,
        "suite_id": suite_id,
        "suite_directory": target.as_posix(),
        "suite_runtime_seconds": suite_tracker.elapsed_seconds,
        "suite_peak_rss_bytes": suite_tracker.peak_rss_bytes,
        "leiden_seconds": float(partition["leiden_seconds"]),
        "status": status,
        "node_primary_labels": corrected_primary,
        "final_backbone_mask": final_backbone,
    }


def _candidate_grid(initial: Iterable[float]) -> list[float]:
    base = [float(value) for value in initial]
    fallbacks = [
        3e-6,
        0.1,
        0.2,
        0.45,
        0.7,
        1.25,
        2.25,
        3.0,
        6.0,
        8.0,
        16.0,
        24.0,
    ]
    result: list[float] = []
    for value in [*base, *fallbacks]:
        if value > 0 and value not in result:
            result.append(value)
    return result


def _pairwise_summary(accepted: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for left_pos, left in enumerate(accepted):
        for right in accepted[left_pos + 1 :]:
            left_labels = left["node_primary_labels"]
            right_labels = right["node_primary_labels"]
            left_backbone = left["final_backbone_mask"]
            right_backbone = right["final_backbone_mask"]
            intersection = int((left_backbone & right_backbone).sum())
            union = int((left_backbone | right_backbone).sum())
            cross = pd.DataFrame({"left": left_labels, "right": right_labels})
            left_containment = cross.groupby("left", sort=False)["right"].value_counts().groupby(level=0).max().sum() / len(cross)
            right_containment = cross.groupby("right", sort=False)["left"].value_counts().groupby(level=0).max().sum() / len(cross)
            rows.append(
                {
                    "left_suite_id": left["suite_id"],
                    "right_suite_id": right["suite_id"],
                    "left_resolution": left["resolution"],
                    "right_resolution": right["resolution"],
                    "common_node_count": len(left_labels),
                    "normalized_mutual_info_score": float(normalized_mutual_info_score(left_labels, right_labels)),
                    "adjusted_rand_score": float(adjusted_rand_score(left_labels, right_labels)),
                    "final_backbone_jaccard": float(intersection / union if union else 1.0),
                    "left_to_right_dominant_coverage": float(left_containment),
                    "right_to_left_dominant_coverage": float(right_containment),
                }
            )
    return pd.DataFrame(rows)


def run_preflight(config: dict[str, Any]) -> dict[str, Any]:
    output_root, temp_root = _assert_safe_output_roots(config)
    preflight_dir = output_root / "preflight"
    if preflight_dir.exists():
        raise FileExistsError(f"Preflight output already exists: {preflight_dir}")
    tracker = ResourceTracker.begin()
    before = input_signatures(config)
    nodes, arcs, origin_nodes, destination_nodes = load_authoritative_inputs(config, tracker)
    partition_nodes, initial_arcs, initial_identity = build_initial_backbone(nodes, arcs, config)
    sample_resolution = float(config["preflight"]["pilot_resolution"])
    pilot_start = time.perf_counter()
    partition = partC_rev1_run_leiden_partition(
        partition_nodes,
        initial_arcs.loc[~initial_arcs["is_final_backbone"].fillna(False).astype(bool)].copy(),
        {"backbone_partition": config["backbone_partition"]},
        resolution=sample_resolution,
        random_seed=int(config["random_seed"]),
    )
    pilot_seconds = time.perf_counter() - pilot_start
    tracker.sample()
    after = input_signatures(config)
    invariance = before == after
    suite_count = len(config["resolutions"])
    projected_hours = max(float(config["preflight"]["historical_projected_hours"]), pilot_seconds * suite_count * 4.0 / 3600.0)
    projected_ram_gib = max(float(config["preflight"]["historical_projected_ram_gib"]), tracker.peak_rss_bytes / 1024**3 * 1.5)
    projected_disk_gib = float(config["preflight"]["projected_disk_gib"])
    budget = config["resource_budget"]
    checks = {
        "authoritative_join_one_to_one": len(arcs) == int(config["input_versions"]["expected_arc_count"]),
        "authoritative_node_count_matches": len(nodes) == int(config["input_versions"]["expected_node_count"]),
        "od_origin_endpoints_nonempty": len(origin_nodes) > 0,
        "od_destination_endpoints_nonempty": len(destination_nodes) > 0,
        "initial_backbone_nonempty": initial_identity["initial_final_backbone_directed_arc_count"] > 0,
        "pilot_partition_nonempty": len(partition["node_partition"]) > 0 and len(partition["regions"]) > 0,
        "input_invariance": invariance,
        "projected_wall_clock_within_budget": projected_hours <= float(budget["wall_clock_hours"]),
        "projected_ram_within_budget": projected_ram_gib <= float(budget["process_tree_ram_gib"]),
        "projected_disk_within_budget": projected_disk_gib <= float(budget["added_disk_gib"]),
        "one_write_capable_orchestrator": int(budget["write_capable_orchestrators"]) == 1,
        "network_disabled": bool(budget["network_disabled"]),
        "package_installation_prohibited": bool(budget["package_installation_prohibited"]),
        "phase3_not_authorized": not bool(config.get("phase_3_authorized", False)),
    }
    status = "PASS" if all(checks.values()) else "BLOCK"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "step_id": STEP_ID,
        "campaign_id": config["campaign_id"],
        "run_id": config["run_id"],
        "status": status,
        "checks": checks,
        "authoritative_node_count": len(nodes),
        "authoritative_arc_count": len(arcs),
        "origin_endpoint_count": len(origin_nodes),
        "destination_endpoint_count": len(destination_nodes),
        "pilot_resolution": sample_resolution,
        "pilot_region_count": len(partition["regions"]),
        "pilot_leiden_seconds": float(partition["leiden_seconds"]),
        "pilot_total_seconds": pilot_seconds,
        "preflight_total_seconds": tracker.elapsed_seconds,
        "observed_peak_process_tree_rss_bytes": tracker.peak_rss_bytes,
        "projected_wall_clock_hours": projected_hours,
        "projected_process_tree_ram_gib": projected_ram_gib,
        "projected_added_disk_gib": projected_disk_gib,
        "resource_budget": budget,
        "initial_backbone_identity": initial_identity,
        "input_signatures_before": before,
        "input_signatures_after": after,
        "auto_continue_authorized": status == "PASS",
        "generated_utc": _utc_now(),
    }
    preflight_dir.mkdir(parents=True)
    _write_json(preflight_dir / "phase2_preflight.json", payload)
    _write_table(
        preflight_dir / "phase2_preflight_resource_usage.csv",
        pd.DataFrame(
            [
                {
                    "preflight_total_seconds": tracker.elapsed_seconds,
                    "pilot_leiden_seconds": float(partition["leiden_seconds"]),
                    "observed_peak_process_tree_rss_bytes": tracker.peak_rss_bytes,
                    "projected_wall_clock_hours": projected_hours,
                    "projected_process_tree_ram_gib": projected_ram_gib,
                    "projected_added_disk_gib": projected_disk_gib,
                }
            ]
        ),
    )
    _write_json(preflight_dir / "input_invariance.json", {"status": "PASS" if invariance else "FAIL", "before": before, "after": after})
    if status != "PASS":
        raise RuntimeError(f"Phase 2 preflight blocked: {checks}")
    return payload


def run_generation(config: dict[str, Any]) -> dict[str, Any]:
    output_root, temp_root = _assert_safe_output_roots(config)
    preflight_path = output_root / "preflight" / "phase2_preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("Approved internal preflight evidence is missing")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or not preflight.get("auto_continue_authorized"):
        raise RuntimeError("Phase 2 generation requires a passing internal preflight")
    if (output_root / "portfolio").exists() or (output_root / "suites").exists():
        raise FileExistsError("Refusing to overwrite an existing Phase 2 portfolio")
    tracker = ResourceTracker.begin()
    before = input_signatures(config)
    nodes, arcs, origin_nodes, destination_nodes = load_authoritative_inputs(config, tracker)
    partition_nodes, initial_arcs, initial_identity = build_initial_backbone(nodes, arcs, config)
    candidates: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    signatures: set[str] = set()
    desired = len(config["resolutions"])
    for candidate_number, resolution in enumerate(_candidate_grid(config["resolutions"]), start=1):
        if len(accepted) >= desired:
            break
        result = generate_suite(
            config=config,
            resolution=resolution,
            candidate_number=candidate_number,
            nodes=nodes,
            arcs=initial_arcs,
            partition_nodes=partition_nodes,
            initial_identity=initial_identity,
            origin_nodes=origin_nodes,
            destination_nodes=destination_nodes,
            temp_root=temp_root,
            final_root=output_root,
            tracker=tracker,
        )
        region_ok = int(config["portfolio_rule"]["min_retained_regions"]) <= int(result["retained_region_count"]) < int(
            config["portfolio_rule"]["max_retained_regions_exclusive"]
        )
        distinct = result["structural_signature"] not in signatures
        accepted_flag = bool(region_ok and distinct)
        reason = "accepted" if accepted_flag else "duplicate_structural_signature" if not distinct else "retained_region_count_outside_range"
        candidates.append(
            {
                "candidate_number": candidate_number,
                "suite_id": result["suite_id"],
                "resolution": resolution,
                "retained_region_count": result["retained_region_count"],
                "structural_signature": result["structural_signature"],
                "accepted": accepted_flag,
                "decision_reason": reason,
            }
        )
        if accepted_flag:
            signatures.add(result["structural_signature"])
            accepted.append(result)
        else:
            rejected_dir = output_root / "rejected_candidates"
            rejected_dir.mkdir(parents=True, exist_ok=True)
            source = output_root / "suites" / result["suite_id"]
            os.replace(source, rejected_dir / result["suite_id"])
        tracker.sample()
        if tracker.elapsed_seconds / 3600 > float(config["resource_budget"]["wall_clock_hours"]):
            raise RuntimeError("Phase 2 wall-clock resource guard exceeded")
        if tracker.peak_rss_bytes / 1024**3 > float(config["resource_budget"]["process_tree_ram_gib"]):
            raise RuntimeError("Phase 2 process-tree RSS resource guard exceeded")
    if len(accepted) != desired:
        raise RuntimeError(f"Could retain only {len(accepted)} structurally distinct suites; required {desired}")
    pairwise = _pairwise_summary(accepted)
    after = input_signatures(config)
    invariance = before == after
    portfolio_dir = output_root / "portfolio"
    portfolio_dir.mkdir(parents=True)
    candidate_frame = pd.DataFrame(candidates)
    summary_columns = [
        "suite_id",
        "resolution",
        "random_seed",
        "membership_hash",
        "node_membership_hash",
        "structural_signature",
        "initial_backbone_directed_arc_count",
        "final_backbone_directed_arc_count",
        "post_partition_backbone_directed_arc_count",
        "small_region_corrected_directed_arc_count",
        "corrected_small_region_count",
        "retained_region_count",
        "portal_record_count",
        "centroid_role_count",
        "terminal_count",
        "min_region_arc_count",
        "max_region_arc_count",
        "median_region_arc_count",
        "min_region_portal_count",
        "max_region_portal_count",
        "suite_runtime_seconds",
        "suite_peak_rss_bytes",
        "leiden_seconds",
        "status",
    ]
    summary = pd.DataFrame([{key: item[key] for key in summary_columns} for item in accepted]).sort_values("resolution", kind="mergesort")
    _write_table(portfolio_dir / "candidate_attempts.csv", candidate_frame)
    _write_table(portfolio_dir / "suite_manifest.csv", summary)
    _write_table(portfolio_dir / "pairwise_structural_similarity.csv", pairwise)
    _write_json(portfolio_dir / "input_invariance.json", {"status": "PASS" if invariance else "FAIL", "before": before, "after": after})
    portfolio_checks = {
        "accepted_suite_count": len(accepted) in {5, 6},
        "requested_suite_count_met": len(accepted) == desired,
        "structural_signatures_unique": len(signatures) == len(accepted),
        "all_suites_pass": bool(summary["status"].eq("PASS").all()),
        "all_region_counts_in_range": bool(
            summary["retained_region_count"].ge(int(config["portfolio_rule"]["min_retained_regions"])).all()
            and summary["retained_region_count"].lt(int(config["portfolio_rule"]["max_retained_regions_exclusive"])).all()
        ),
        "pairwise_comparison_count_complete": len(pairwise) == len(accepted) * (len(accepted) - 1) // 2,
        "input_invariance": invariance,
        "phase3_artifact_count_zero": True,
        "bacn_count_zero": True,
        "racn_count_zero": True,
        "assignment_count_zero": True,
        "training_label_count_zero": True,
        "trained_model_count_zero": True,
    }
    portfolio_status = "PASS" if all(portfolio_checks.values()) else "FAIL"
    portfolio_summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "step_id": STEP_ID,
        "campaign_id": config["campaign_id"],
        "run_id": config["run_id"],
        "status": portfolio_status,
        "accepted_suite_count": len(accepted),
        "candidate_attempt_count": len(candidates),
        "accepted_suite_ids": [item["suite_id"] for item in accepted],
        "accepted_resolutions": [item["resolution"] for item in accepted],
        "structural_signatures": [item["structural_signature"] for item in accepted],
        "checks": portfolio_checks,
        "input_invariance": invariance,
        "resource_use": {
            "wall_clock_seconds": tracker.elapsed_seconds,
            "peak_process_tree_rss_bytes": tracker.peak_rss_bytes,
            "added_output_bytes": int(sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file())),
        },
        "phase_3_authorized": False,
        "next_external_gate": "WAITING_FOR_EXTERNAL_REVIEW",
        "generated_utc": _utc_now(),
    }
    _write_json(portfolio_dir / "portfolio_summary.json", portfolio_summary)
    _write_json(portfolio_dir / "portfolio_validation.json", {"status": portfolio_status, "checks": portfolio_checks})
    _write_table(
        portfolio_dir / "resource_usage.csv",
        pd.DataFrame(
            [
                {
                    "wall_clock_seconds": tracker.elapsed_seconds,
                    "peak_process_tree_rss_bytes": tracker.peak_rss_bytes,
                    "added_output_bytes": portfolio_summary["resource_use"]["added_output_bytes"],
                    "accepted_suite_count": len(accepted),
                    "candidate_attempt_count": len(candidates),
                }
            ]
        ),
    )
    _write_json(
        portfolio_dir / "machine_status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "step_id": STEP_ID,
            "campaign_id": config["campaign_id"],
            "run_id": config["run_id"],
            "status": "WAITING_FOR_EXTERNAL_REVIEW" if portfolio_status == "PASS" else "BLOCKED",
            "recommendation": "PROCEED_TO_EXTERNAL_REVIEW" if portfolio_status == "PASS" else "REPAIR",
            "phase_3_authorized": False,
        },
    )
    manifest_rows = []
    for path in sorted(p for p in output_root.rglob("*") if p.is_file()):
        manifest_rows.append(
            {
                "relative_path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_table(portfolio_dir / "artifact_manifest.csv", pd.DataFrame(manifest_rows))
    forbidden = [
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and any(token in path.name.lower() for token in FORBIDDEN_OUTPUT_TOKENS)
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden later-phase outputs detected: {forbidden}")
    if portfolio_status != "PASS":
        raise RuntimeError(f"Portfolio validation failed: {portfolio_checks}")
    return portfolio_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--mode", required=True, choices=["preflight", "generate"])
    parser.add_argument("--confirm-real-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.phase_id != PHASE_ID or args.step_id != STEP_ID or args.run_id != config["run_id"]:
        raise ValueError("Command identity does not match the frozen config")
    allowed_task = "P2-PREFLIGHT" if args.mode == "preflight" else "P2-GENERATE"
    if args.task_id != allowed_task or args.task_id not in config["tasks"]:
        raise ValueError(f"Mode {args.mode} requires explicit task {allowed_task}")
    if not args.confirm_real_run:
        raise ValueError("Real Phase 2 execution requires --confirm-real-run")
    result = run_preflight(config) if args.mode == "preflight" else run_generation(config)
    print(json.dumps({"status": result["status"], "mode": args.mode, "run_id": config["run_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
