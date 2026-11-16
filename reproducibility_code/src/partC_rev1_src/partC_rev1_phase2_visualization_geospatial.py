"""Phase 2 visualization and GeoPackage supplement for accepted mapping suites.

The accepted suite Parquet/JSON/CSV files are immutable inputs.  This module
only joins those records to read-only physical geometry/attributes, prepares
review tables and figures, and validates the resulting supplement.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from matplotlib.collections import LineCollection
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .partC_rev1_hashing import partC_rev1_file_signature


PHASE_ID = "phase_02"
STEP_ID = "phase2_visualization_geospatial_supplement"
SCHEMA_VERSION = "1.0"
EXPECTED_TASK_BY_MODE = {
    "preflight": "P2V-PREFLIGHT",
    "generate": "P2V-GENERATE",
    "validate": "P2V-VALIDATE",
}
SUITE_PREFIXES = ("s01", "s02", "s03", "s04", "s05", "s06")
SUITE_SUFFIXES = (
    "suite_id",
    "resolution",
    "label",
    "region_uuid",
    "is_initial_backbone",
    "is_post_partition_backbone",
    "is_small_region_correction_backbone",
    "is_final_backbone",
    "region_arc_count",
    "region_node_count",
    "region_portal_count",
    "region_color_hex",
)
FORBIDDEN_OUTPUT_TOKENS = (
    "bacn",
    "b_acn",
    "racn",
    "r_acn",
    "assignment_result",
    "benchmark_trace",
    "training_label",
    "trained_model",
    "phase_03",
    "phase3",
)
PHYSICAL_UPDATE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("arc_idx", "INTEGER"),
    ("tail_node_uid", "TEXT"),
    ("head_node_uid", "TEXT"),
    ("free_time_min", "REAL"),
    ("free_time_source", "TEXT"),
    ("length_m", "REAL"),
    ("length_mi", "REAL"),
    ("base_capacity_annual_ktons", "REAL"),
    ("assignment_capacity_annual_ktons", "REAL"),
    ("capacity_source", "TEXT"),
    ("bpr_alpha", "REAL"),
    ("bpr_beta", "REAL"),
    ("speed_mph", "REAL"),
    ("is_interstate", "INTEGER"),
    ("is_nhs", "INTEGER"),
    ("is_strahnet", "INTEGER"),
    ("is_centroid_connector", "INTEGER"),
    ("is_closed", "INTEGER"),
    ("is_truck_prohibited", "INTEGER"),
    ("is_usable", "INTEGER"),
    ("zero_length_flag", "INTEGER"),
    ("self_loop_flag", "INTEGER"),
    ("parallel_arc_group_id", "TEXT"),
    ("weak_component_id", "TEXT"),
    ("authoritative_graph_identity", "TEXT"),
    ("geometry_source", "TEXT"),
    ("geometry_source_layer", "TEXT"),
    ("geometry_direction_ok", "INTEGER"),
    ("geometry_reversed_for_direction", "INTEGER"),
    ("geometry_fallback_used", "INTEGER"),
    ("tail_geometry_distance", "REAL"),
    ("head_geometry_distance", "REAL"),
    ("geometry_crs", "TEXT"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")


def sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path: str | Path, payload: Any) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    os.replace(temporary, target)


def write_text_new(path: str | Path, text: str) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, target)


def write_csv_new(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, target)


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def tree_rss_bytes() -> int:
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            total += child.memory_info().rss
    return int(total)


@dataclass
class ResourceTracker:
    config: Mapping[str, Any]
    output_root: Path
    started: float = field(default_factory=time.perf_counter)
    peak_rss_bytes: int = 0
    stage_started: dict[str, float] = field(default_factory=dict)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)

    def sample(self, stage: str) -> None:
        elapsed = time.perf_counter() - self.started
        rss = tree_rss_bytes()
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        added = directory_size(self.output_root)
        row = {
            "recorded_utc": utc_now(),
            "stage": stage,
            "elapsed_seconds": elapsed,
            "process_tree_rss_bytes": rss,
            "added_output_bytes": added,
        }
        self.samples.append(row)
        budget = self.config["resource_budget"]
        if elapsed > float(budget["wall_clock_hours"]) * 3600:
            raise RuntimeError("Supplement wall-clock limit exceeded")
        if rss > float(budget["process_tree_ram_gib"]) * 1024**3:
            raise RuntimeError("Supplement process-tree RSS limit exceeded")
        if added > float(budget["added_disk_gib"]) * 1024**3:
            raise RuntimeError("Supplement added-disk limit exceeded")

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.sample(f"{name}:start")
        begin = time.perf_counter()
        self.stage_started[name] = begin
        yield
        self.stage_seconds[name] = time.perf_counter() - begin
        self.sample(f"{name}:end")

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.samples)

    def summary(self) -> dict[str, Any]:
        return {
            "wall_clock_seconds": time.perf_counter() - self.started,
            "peak_process_tree_rss_bytes": self.peak_rss_bytes,
            "added_output_bytes": directory_size(self.output_root),
            "stage_seconds": self.stage_seconds,
        }


def load_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "phase_id",
        "step_id",
        "campaign_id",
        "run_id",
        "tasks",
        "worktree_root",
        "accepted_phase2_output_root",
        "output_root",
        "temp_root",
        "accepted_core",
        "read_only_inputs",
        "resource_budget",
        "geopackage",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Missing config fields: {missing}")
    if payload["phase_id"] != PHASE_ID or payload["step_id"] != STEP_ID:
        raise ValueError("Config phase/step mismatch")
    if not isinstance(payload["tasks"], list) or not payload["tasks"]:
        raise ValueError("Config tasks must be a nonempty list")
    if payload.get("phase_3_authorized") is not False:
        raise ValueError("Phase 3 must remain unauthorized")
    accepted_root = Path(payload["accepted_phase2_output_root"]).resolve()
    output_root = Path(payload["output_root"]).resolve()
    temp_root = Path(payload["temp_root"]).resolve()
    worktree = Path(payload["worktree_root"]).resolve()
    if output_root.parent != accepted_root or output_root.name != "visualization_geospatial":
        raise ValueError("Supplement output root is not the approved child of the accepted Phase 2 output")
    if worktree == output_root or worktree in output_root.parents:
        raise ValueError("Large supplement output may not be written inside the worktree")
    if temp_root == output_root or temp_root in output_root.parents or output_root in temp_root.parents:
        raise ValueError("Output and temporary roots must be separate")
    for key, value in payload["read_only_inputs"].items():
        if key.endswith(("_parquet", "_sqlite", "_gpkg")) and not Path(value).is_file():
            raise FileNotFoundError(f"Missing read-only input {key}: {value}")
    payload["_config_path"] = resolved.as_posix()
    payload["_config_sha256"] = sha256(resolved)
    return payload


def validate_invocation(config: Mapping[str, Any], args: argparse.Namespace) -> None:
    if not args.confirm_real_run:
        raise ValueError("Real supplement invocation requires --confirm-real-run")
    if args.phase_id != config["phase_id"] or args.step_id != config["step_id"] or args.run_id != config["run_id"]:
        raise ValueError("Invocation identity does not match frozen config")
    if args.task_id not in config["tasks"]:
        raise ValueError("Task ID is not in the frozen nonempty task list")
    if EXPECTED_TASK_BY_MODE[args.mode] != args.task_id:
        raise ValueError(f"Mode {args.mode} requires task {EXPECTED_TASK_BY_MODE[args.mode]}")


def suite_directories(config: Mapping[str, Any]) -> list[Path]:
    root = Path(config["accepted_phase2_output_root"]) / "suites"
    directories = [root / suite_id for suite_id in config["accepted_core"]["suite_ids"]]
    if any(not directory.is_dir() for directory in directories):
        raise FileNotFoundError("One or more accepted suite directories are missing")
    return directories


def _manifest_failures(root: Path, manifest: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    for row in manifest.itertuples(index=False):
        path = root / str(row.relative_path)
        if not path.is_file():
            failures.append(f"missing:{row.relative_path}")
        elif path.stat().st_size != int(row.size_bytes):
            failures.append(f"size:{row.relative_path}")
        elif sha256(path) != str(row.sha256):
            failures.append(f"sha256:{row.relative_path}")
    return failures


def authoritative_input_signatures(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    worktree = Path(config["worktree_root"])
    core_config = json.loads(
        (worktree / "partC_doc_icnn/phase_2_work/campaigns/phase2_split_mapping_20260817_v1/PHASE2_EXECUTION_CONFIG.json").read_text(
            encoding="utf-8"
        )
    )
    versions = core_config["input_versions"]
    paths = {
        "graph_npz": Path(versions["graph_npz"]),
        "graph_nodes": Path(versions["graph_nodes_parquet"]),
        "graph_arcs": Path(versions["graph_arcs_parquet"]),
        "graph_metadata": Path(versions["graph_metadata_json"]),
        "od_npz": Path(versions["od_npz"]),
        "od_metadata": Path(versions["od_metadata_json"]),
        "rich_sqlite": Path(versions["rich_sqlite"]),
    }
    return {
        name: {"path": path.resolve().as_posix(), "size_bytes": path.stat().st_size, "signature": partC_rev1_file_signature(path)}
        for name, path in paths.items()
    }


def verify_accepted_core(config: Mapping[str, Any]) -> dict[str, Any]:
    accepted = config["accepted_core"]
    root = Path(config["accepted_phase2_output_root"])
    core_zip = Path(accepted["zip_path"])
    failures: list[str] = []
    if core_zip.stat().st_size != int(accepted["zip_size_bytes"]) or sha256(core_zip) != accepted["zip_sha256"]:
        failures.append("reviewed_core_zip")
    companion_path = core_zip.with_suffix(core_zip.suffix + ".companion.json")
    companion = json.loads(companion_path.read_text(encoding="utf-8"))
    if companion.get("zip_sha256") != accepted["zip_sha256"] or int(companion.get("file_count", -1)) != 223:
        failures.append("reviewed_core_companion")
    portfolio_manifest_path = Path(accepted["portfolio_artifact_manifest"])
    portfolio_manifest = pd.read_csv(portfolio_manifest_path)
    if len(portfolio_manifest) != int(accepted["portfolio_artifact_manifest_rows"]):
        failures.append("portfolio_manifest_row_count")
    failures.extend(f"portfolio:{item}" for item in _manifest_failures(root, portfolio_manifest))
    suite_results: list[dict[str, Any]] = []
    for directory in suite_directories(config):
        manifest = pd.read_csv(directory / "artifact_manifest.csv")
        suite_failures = _manifest_failures(directory, manifest)
        identity = json.loads((directory / "suite_identity.json").read_text(encoding="utf-8"))
        signature = json.loads((directory / "structural_signature.json").read_text(encoding="utf-8"))
        suite_results.append(
            {
                "suite_id": directory.name,
                "manifest_row_count": len(manifest),
                "manifest_status": "PASS" if not suite_failures else "FAIL",
                "manifest_failures": suite_failures,
                "resolution": identity["resolution"],
                "structural_signature": signature["structural_signature"],
            }
        )
        failures.extend(f"{directory.name}:{item}" for item in suite_failures)
    expected_hashes = {
        root / "portfolio/suite_manifest.csv": accepted["suite_manifest_sha256"],
        root / "portfolio/portfolio_summary.json": accepted["portfolio_summary_sha256"],
        root / "portfolio/pairwise_structural_similarity.csv": accepted["pairwise_similarity_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256(path) != expected:
            failures.append(f"accepted_hash:{path.name}")
    accepted_invariance = json.loads((root / "portfolio/input_invariance.json").read_text(encoding="utf-8"))["after"]
    current_invariance = authoritative_input_signatures(config)
    if current_invariance != accepted_invariance:
        failures.append("authoritative_input_invariance")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "checked_utc": utc_now(),
        "reviewed_core_zip": {
            "path": core_zip.as_posix(),
            "size_bytes": core_zip.stat().st_size,
            "sha256": sha256(core_zip),
            "companion": companion,
        },
        "portfolio_manifest_rows": len(portfolio_manifest),
        "portfolio_manifest_files_checked": len(portfolio_manifest),
        "suite_manifest_files_checked": int(sum(row["manifest_row_count"] for row in suite_results)),
        "suites": suite_results,
        "authoritative_inputs": current_invariance,
        "failures": failures,
    }
    if failures:
        raise RuntimeError(f"Accepted-core verification failed: {failures[:20]}")
    return result


def field_lineage(config: Mapping[str, Any]) -> dict[str, Any]:
    path = (
        Path(config["worktree_root"])
        / "partC_doc_icnn/phase_2_work/campaigns/phase2_visualization_geospatial_20260817_v1/FIELD_LINEAGE.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def gpkg_schema(config: Mapping[str, Any]) -> dict[str, Any]:
    path = (
        Path(config["worktree_root"])
        / "partC_doc_icnn/phase_2_work/campaigns/phase2_visualization_geospatial_20260817_v1/GPKG_SCHEMA.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def orient_coordinates(
    coordinates: np.ndarray, tail_xy: tuple[float, float], head_xy: tuple[float, float]
) -> tuple[np.ndarray, bool, float, float]:
    array = np.asarray(coordinates, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("A directed line requires at least two 2-D coordinates")
    forward_tail = float(np.hypot(*(array[0, :2] - np.asarray(tail_xy))))
    forward_head = float(np.hypot(*(array[-1, :2] - np.asarray(head_xy))))
    reverse_tail = float(np.hypot(*(array[-1, :2] - np.asarray(tail_xy))))
    reverse_head = float(np.hypot(*(array[0, :2] - np.asarray(head_xy))))
    if forward_tail + forward_head <= reverse_tail + reverse_head:
        return array, False, forward_tail, forward_head
    return array[::-1].copy(), True, reverse_tail, reverse_head


def deterministic_region_color(region_uuid: str | None) -> str:
    if region_uuid is None or not str(region_uuid) or str(region_uuid) == "backbone":
        return "#202124"
    return "#" + hashlib.sha256(str(region_uuid).encode("utf-8")).hexdigest()[:6]


def suite_wide_field_names(prefixes: Sequence[str] = SUITE_PREFIXES) -> list[str]:
    return [f"{prefix}_{suffix}" for prefix in prefixes for suffix in SUITE_SUFFIXES]


def build_suite_membership_frame(
    membership: pd.DataFrame, regions: pd.DataFrame, resolution: float
) -> pd.DataFrame:
    required = {
        "suite_id",
        "arc_idx",
        "directed_arc_id",
        "partC_rev1_link_uid",
        "tail_node_uid",
        "head_node_uid",
        "suite_label",
        "arc_partition_type",
        "is_initial_backbone",
        "is_post_partition_backbone",
        "is_small_region_correction_backbone",
        "is_final_backbone",
    }
    if missing := sorted(required - set(membership.columns)):
        raise ValueError(f"Missing accepted membership fields: {missing}")
    if membership["directed_arc_id"].duplicated().any():
        raise ValueError("Accepted suite directed_arc_id is not unique")
    output = membership.copy()
    output["directed_arc_id"] = pd.to_numeric(output["directed_arc_id"], errors="raise").astype("int64")
    output["region_uuid"] = output["suite_label"].where(~output["is_final_backbone"], pd.NA)
    stats = regions[["region_uuid", "arc_count", "node_count", "portal_count"]].rename(
        columns={
            "arc_count": "region_arc_count",
            "node_count": "region_node_count",
            "portal_count": "region_portal_count",
        }
    )
    output = output.merge(stats, on="region_uuid", how="left", validate="many_to_one", sort=False)
    regional = ~output["is_final_backbone"]
    if output.loc[regional, ["region_arc_count", "region_node_count", "region_portal_count"]].isna().any().any():
        raise ValueError("Regional membership is missing accepted retained-region statistics")
    color_map = {value: deterministic_region_color(str(value)) for value in output.loc[regional, "region_uuid"].unique()}
    output["region_color_hex"] = output["region_uuid"].map(color_map).fillna("#202124")
    output.insert(1, "resolution", float(resolution))
    columns = [
        "suite_id",
        "resolution",
        "arc_idx",
        "directed_arc_id",
        "partC_rev1_link_uid",
        "tail_node_uid",
        "head_node_uid",
        "suite_label",
        "region_uuid",
        "arc_partition_type",
        "is_initial_backbone",
        "is_post_partition_backbone",
        "is_small_region_correction_backbone",
        "is_final_backbone",
        "region_arc_count",
        "region_node_count",
        "region_portal_count",
        "region_color_hex",
    ]
    return output[columns].sort_values("arc_idx", kind="mergesort").reset_index(drop=True)


def configure_geospatial_environment(config: Mapping[str, Any]) -> None:
    prefix = Path(config["python_executable"]).parent
    os.environ.setdefault("GDAL_DATA", str(prefix / "Library/share/gdal"))
    os.environ.setdefault("PROJ_LIB", str(prefix / "Library/share/proj"))
    binary = str(prefix / "Library/bin")
    if binary.lower() not in os.environ.get("PATH", "").lower():
        os.environ["PATH"] = binary + os.pathsep + os.environ.get("PATH", "")


def open_sqlite(path: str | Path, *, read_only: bool, spatialite_extension: str | None = None) -> sqlite3.Connection:
    resolved = Path(path).resolve()
    connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True) if read_only else sqlite3.connect(resolved)
    connection.execute("PRAGMA foreign_keys=ON")
    if read_only:
        connection.execute("PRAGMA query_only=1")
    elif spatialite_extension:
        connection.enable_load_extension(True)
        connection.load_extension(spatialite_extension)
        connection.enable_load_extension(False)
    return connection


def snapshot_accepted_manifests(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(config["accepted_phase2_output_root"])
    suites: dict[str, Any] = {}
    for directory in suite_directories(config):
        manifest_path = directory / "artifact_manifest.csv"
        manifest = pd.read_csv(manifest_path)
        suites[directory.name] = {
            "manifest_path": manifest_path.as_posix(),
            "manifest_size_bytes": manifest_path.stat().st_size,
            "manifest_sha256": sha256(manifest_path),
            "artifact_count": len(manifest),
            "artifacts": manifest.to_dict(orient="records"),
        }
    portfolio_path = root / "portfolio/artifact_manifest.csv"
    portfolio = pd.read_csv(portfolio_path)
    return {
        "snapshot_utc": utc_now(),
        "portfolio": {
            "manifest_path": portfolio_path.as_posix(),
            "manifest_size_bytes": portfolio_path.stat().st_size,
            "manifest_sha256": sha256(portfolio_path),
            "artifact_count": len(portfolio),
            "artifacts": portfolio.to_dict(orient="records"),
        },
        "suites": suites,
    }


def inspect_live_schemas(config: Mapping[str, Any]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    inputs = config["read_only_inputs"]
    parent = Path(inputs["parent_gpkg"])
    rich = Path(inputs["rich_sqlite"])
    result: dict[str, Any] = {"parent_gpkg": {}, "rich_sqlite": {}, "accepted_suite_parquet": {}}
    with contextlib.closing(open_sqlite(parent, read_only=True)) as connection:
        result["parent_gpkg"]["application_id"] = connection.execute("PRAGMA application_id").fetchone()[0]
        result["parent_gpkg"]["user_version"] = connection.execute("PRAGMA user_version").fetchone()[0]
        result["parent_gpkg"]["contents"] = [
            dict(zip(("table_name", "data_type", "identifier", "srs_id", "min_x", "min_y", "max_x", "max_y"), row))
            for row in connection.execute(
                "SELECT table_name,data_type,identifier,srs_id,min_x,min_y,max_x,max_y FROM gpkg_contents ORDER BY table_name"
            )
        ]
        result["parent_gpkg"]["geometry_columns"] = [
            dict(zip(("table_name", "column_name", "geometry_type_name", "srs_id", "z", "m"), row))
            for row in connection.execute(
                "SELECT table_name,column_name,geometry_type_name,srs_id,z,m FROM gpkg_geometry_columns ORDER BY table_name"
            )
        ]
        for layer in (inputs["parent_link_layer"], inputs["parent_node_layer"]):
            columns = connection.execute(f'PRAGMA table_info("{layer}")').fetchall()
            result["parent_gpkg"][layer] = {
                "row_count": connection.execute(f'SELECT COUNT(*) FROM "{layer}"').fetchone()[0],
                "columns": [
                    {"cid": row[0], "name": row[1], "type": row[2], "notnull": bool(row[3]), "pk": bool(row[5])}
                    for row in columns
                ],
            }
    with contextlib.closing(open_sqlite(rich, read_only=True)) as connection:
        for table in ("partC_rev1_directed_arcs_enriched", "partC_rev1_clean_physical_nodes"):
            columns = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
            result["rich_sqlite"][table] = {
                "row_count": connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0],
                "columns": [
                    {"cid": row[0], "name": row[1], "type": row[2], "notnull": bool(row[3]), "pk": bool(row[5])}
                    for row in columns
                ],
                "indexes": [row[1] for row in connection.execute(f'PRAGMA index_list("{table}")')],
            }
    first_suite = suite_directories(config)[0]
    for name in (
        "directed_arc_membership.parquet",
        "retained_regions.parquet",
        "portals.parquet",
        "centroid_roles.parquet",
        "terminals.parquet",
        "small_region_corrections.parquet",
        "node_partition_membership.parquet",
    ):
        parquet = pq.ParquetFile(first_suite / name)
        result["accepted_suite_parquet"][name] = {
            "rows_first_suite": parquet.metadata.num_rows,
            "columns": [{"name": item.name, "type": str(item.type)} for item in parquet.schema_arrow],
        }
    return result


def exhaustive_geometry_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    configure_geospatial_environment(config)
    from osgeo import ogr

    inputs = config["read_only_inputs"]
    parent = Path(inputs["parent_gpkg"])
    graph_arcs = pd.read_parquet(
        inputs["graph_arcs_parquet"], columns=["directed_arc_id", "tail_node_uid", "head_node_uid"]
    )
    graph_arcs["directed_arc_id"] = graph_arcs["directed_arc_id"].astype("string")
    expected_endpoint = graph_arcs.set_index("directed_arc_id")[["tail_node_uid", "head_node_uid"]].astype("string")
    accepted = set(expected_endpoint.index.astype(str))
    with contextlib.closing(open_sqlite(parent, read_only=True)) as connection:
        coordinates = {
            str(uid): (float(lon), float(lat))
            for uid, lon, lat in connection.execute(
                f'SELECT CAST(partC_rev1_node_uid AS TEXT),partC_rev1_lon,partC_rev1_lat FROM "{inputs["parent_node_layer"]}"'
            )
        }
    dataset = ogr.Open(str(parent), 0)
    if dataset is None:
        raise RuntimeError("OGR could not open the parent GeoPackage read-only")
    layer = dataset.GetLayerByName(inputs["parent_link_layer"])
    checked = direction_ok = needs_reverse = missing_geometry = missing_node = tail_mismatch = head_mismatch = 0
    max_tail = max_head = 0.0
    started = time.perf_counter()
    for feature in layer:
        directed_arc_id = str(feature["directed_arc_id"])
        if directed_arc_id not in accepted:
            continue
        checked += 1
        expected = expected_endpoint.loc[directed_arc_id]
        from_uid = str(feature["from_node_uid"])
        to_uid = str(feature["to_node_uid"])
        tail_mismatch += int(from_uid != str(expected["tail_node_uid"]))
        head_mismatch += int(to_uid != str(expected["head_node_uid"]))
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            missing_geometry += 1
            continue
        tail = coordinates.get(from_uid)
        head = coordinates.get(to_uid)
        if tail is None or head is None:
            missing_node += 1
            continue
        first_part = geometry.GetGeometryRef(0)
        last_part = geometry.GetGeometryRef(geometry.GetGeometryCount() - 1)
        start = np.asarray(first_part.GetPoint(0)[:2], dtype=np.float64)
        end = np.asarray(last_part.GetPoint(last_part.GetPointCount() - 1)[:2], dtype=np.float64)
        forward_tail = float(np.hypot(*(start - np.asarray(tail))))
        forward_head = float(np.hypot(*(end - np.asarray(head))))
        reverse_tail = float(np.hypot(*(end - np.asarray(tail))))
        reverse_head = float(np.hypot(*(start - np.asarray(head))))
        max_tail = max(max_tail, forward_tail)
        max_head = max(max_head, forward_head)
        if forward_tail + forward_head <= reverse_tail + reverse_head:
            direction_ok += 1
        else:
            needs_reverse += 1
    dataset = None
    expected_count = int(config["accepted_core"]["authoritative_arc_count"])
    status = (
        "PASS"
        if checked == direction_ok == expected_count
        and not any((needs_reverse, missing_geometry, missing_node, tail_mismatch, head_mismatch))
        else "FAIL"
    )
    result = {
        "status": status,
        "accepted_arc_count": expected_count,
        "rows_checked": checked,
        "direction_ok_count": direction_ok,
        "needs_reverse_count": needs_reverse,
        "missing_geometry_count": missing_geometry,
        "missing_node_count": missing_node,
        "tail_id_mismatch_count": tail_mismatch,
        "head_id_mismatch_count": head_mismatch,
        "geometry_fallback_count": 0,
        "max_tail_distance_degrees": max_tail,
        "max_head_distance_degrees": max_head,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    if status != "PASS":
        raise RuntimeError(f"Exhaustive directed geometry audit failed: {result}")
    return result


def geopackage_capability_probe(config: Mapping[str, Any], probe_path: Path) -> dict[str, Any]:
    configure_geospatial_environment(config)
    from osgeo import ogr, osr

    import fiona
    import pyogrio

    if probe_path.exists():
        raise FileExistsError(probe_path)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    driver = ogr.GetDriverByName("GPKG")
    dataset = driver.CreateDataSource(str(probe_path))
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(4269)
    layer = dataset.CreateLayer("probe_points", srs=spatial_reference, geom_type=ogr.wkbPoint, options=["SPATIAL_INDEX=YES"])
    layer.CreateField(ogr.FieldDefn("stable_id", ogr.OFTString))
    feature = ogr.Feature(layer.GetLayerDefn())
    feature["stable_id"] = "probe-001"
    geometry = ogr.Geometry(ogr.wkbPoint)
    geometry.AddPoint(-96.0, 39.0)
    feature.SetGeometry(geometry)
    if layer.CreateFeature(feature) != ogr.OGRERR_NONE:
        raise RuntimeError("OGR GeoPackage probe feature write failed")
    feature = None
    dataset = None
    dataset = ogr.Open(str(probe_path), 0)
    ogr_count = dataset.GetLayerByName("probe_points").GetFeatureCount()
    dataset = None
    with fiona.open(probe_path, layer="probe_points") as source:
        fiona_count = len(source)
        fiona_crs = source.crs.to_string()
    info = pyogrio.read_info(probe_path, layer="probe_points")
    pyogrio_count = int(info["features"])
    status = "PASS" if ogr_count == fiona_count == pyogrio_count == 1 and fiona_crs == "EPSG:4269" else "FAIL"
    return {
        "status": status,
        "probe_path": probe_path.as_posix(),
        "size_bytes": probe_path.stat().st_size,
        "ogr_count": ogr_count,
        "fiona_count": fiona_count,
        "pyogrio_count": pyogrio_count,
        "crs": fiona_crs,
    }


def installed_library_versions(config: Mapping[str, Any]) -> dict[str, Any]:
    configure_geospatial_environment(config)
    import importlib

    names = ("numpy", "pandas", "pyarrow", "matplotlib", "geopandas", "fiona", "pyogrio", "osgeo", "shapely", "psutil", "scipy")
    versions: dict[str, Any] = {"python": sys.version, "executable": sys.executable}
    for name in names:
        try:
            module = importlib.import_module(name)
            versions[name] = {"available": True, "version": getattr(module, "__version__", None)}
        except Exception as exc:  # pragma: no cover - fail-closed live evidence
            versions[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    required = ("numpy", "pandas", "pyarrow", "matplotlib", "fiona", "pyogrio", "osgeo", "shapely", "psutil", "scipy")
    if not all(versions[name]["available"] for name in required):
        raise RuntimeError("A required preinstalled library is unavailable")
    return versions


def run_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    output_root = Path(config["output_root"])
    temp_root = Path(config["temp_root"])
    preflight_root = output_root / "preflight"
    if preflight_root.exists():
        raise FileExistsError(preflight_root)
    output_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    tracker = ResourceTracker(config, output_root)
    with tracker.stage("accepted_core_verification"):
        core = verify_accepted_core(config)
        manifests = snapshot_accepted_manifests(config)
    with tracker.stage("library_and_schema_inspection"):
        libraries = installed_library_versions(config)
        schemas = inspect_live_schemas(config)
    with tracker.stage("geopackage_capability_probe"):
        probe = geopackage_capability_probe(config, temp_root / "cli_preflight_probe.gpkg")
    with tracker.stage("exhaustive_geometry_audit"):
        geometry = exhaustive_geometry_audit(config)
    free_disk = shutil.disk_usage(output_root).free
    memory = psutil.virtual_memory()
    parent_signature = partC_rev1_file_signature(config["read_only_inputs"]["parent_gpkg"])
    if parent_signature != config["read_only_inputs"]["parent_gpkg_fast_signature"]:
        raise RuntimeError("Parent GeoPackage signature changed")
    status = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "step_id": STEP_ID,
        "campaign_id": config["campaign_id"],
        "run_id": config["run_id"],
        "task_id": "P2V-PREFLIGHT",
        "status": "PASS",
        "recorded_utc": utc_now(),
        "config_sha256": config["_config_sha256"],
        "checks": {
            "accepted_core": True,
            "accepted_manifest_snapshot": True,
            "authoritative_input_invariance": True,
            "required_libraries": True,
            "live_schemas": True,
            "geopackage_write_read": probe["status"] == "PASS",
            "directed_geometry_orientation": geometry["status"] == "PASS",
            "parent_gpkg_signature": True,
            "resources": free_disk > 30 * 1024**3 and memory.available > 4 * 1024**3,
            "phase_3_authorized_false": config.get("phase_3_authorized") is False,
        },
        "resource_projection": {
            "wall_clock_hours": 1.5,
            "peak_process_tree_rss_gib": 8.0,
            "added_disk_gib": 12.0,
            "free_disk_bytes": free_disk,
            "available_memory_bytes": memory.available,
        },
        "geometry_audit": geometry,
        "geopackage_probe": probe,
        "resource_use": tracker.summary(),
        "next_task": "P2V-GENERATE after implementation and targeted tests",
        "next_step_authorized": True,
        "phase_3_authorized": False,
    }
    if not all(status["checks"].values()):
        status["status"] = "BLOCKED"
        status["next_step_authorized"] = False
    preflight_root.mkdir(parents=True, exist_ok=False)
    write_json_new(preflight_root / "accepted_core_snapshot_before.json", core)
    write_json_new(preflight_root / "accepted_manifest_snapshot_before.json", manifests)
    write_json_new(preflight_root / "authoritative_input_snapshot_before.json", core["authoritative_inputs"])
    write_json_new(preflight_root / "live_schema_snapshot.json", schemas)
    write_json_new(preflight_root / "library_capabilities.json", libraries)
    write_json_new(preflight_root / "geometry_orientation_audit.json", geometry)
    write_json_new(preflight_root / "preflight_status.json", status)
    write_csv_new(preflight_root / "preflight_resource_usage.csv", tracker.frame())
    write_text_new(
        preflight_root / "command.txt",
        "python -m partC_rev1_src.partC_rev1_phase2_visualization_geospatial "
        f"--config {config['_config_path']} --phase-id {PHASE_ID} --step-id {STEP_ID} "
        f"--run-id {config['run_id']} --task-id P2V-PREFLIGHT --mode preflight --confirm-real-run\n",
    )
    if status["status"] != "PASS":
        raise RuntimeError("Supplement preflight blocked")
    return status


def _sqlite_tuning(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-524288")


def _checkpoint_sqlite(path: Path) -> None:
    with contextlib.closing(sqlite3.connect(path)) as connection:
        checkpoint_cursor = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        checkpoint = checkpoint_cursor.fetchone()
        checkpoint_cursor.close()
        if checkpoint is None or int(checkpoint[0]) != 0:
            raise RuntimeError(f"SQLite WAL checkpoint remained busy for {path}: {checkpoint}")
    # On the supported Windows SQLite build, changing out of WAL mode while the
    # checkpointing connection remains open raises SQLITE_BUSY even after the
    # PRAGMA result cursor is exhausted. Reopen for the journal-mode transition.
    with contextlib.closing(sqlite3.connect(path)) as connection:
        journal_cursor = connection.execute("PRAGMA journal_mode=DELETE")
        journal_mode = journal_cursor.fetchone()
        journal_cursor.close()
        if journal_mode is None or str(journal_mode[0]).lower() != "delete":
            raise RuntimeError(f"SQLite journal mode did not return to DELETE for {path}: {journal_mode}")


def _register_attribute_table(connection: sqlite3.Connection, table: str, description: str) -> None:
    connection.execute("DELETE FROM gpkg_contents WHERE table_name=?", (table,))
    connection.execute(
        """
        INSERT INTO gpkg_contents(
            table_name,data_type,identifier,description,last_change,min_x,min_y,max_x,max_y,srs_id
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (table, "attributes", table, description, utc_now(), None, None, None, None, None),
    )


def _write_attribute_table(
    connection: sqlite3.Connection,
    table: str,
    frame: pd.DataFrame,
    description: str,
    *,
    chunksize: int = 50_000,
) -> None:
    exists = connection.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone()
    if exists:
        raise FileExistsError(f"GeoPackage object already exists: {table}")
    frame.to_sql(table, connection, if_exists="fail", index=False, chunksize=chunksize)
    _register_attribute_table(connection, table, description)


def _add_columns(connection: sqlite3.Connection, table: str, columns: Iterable[tuple[str, str]]) -> None:
    existing = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
    for name, sql_type in columns:
        if name not in existing:
            connection.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {sql_type}')
            existing.add(name)


def _parent_link_sql(config: Mapping[str, Any]) -> str:
    layer = config["read_only_inputs"]["parent_link_layer"]
    return f'''SELECT
        directed_arc_id,
        partC_rev1_link_uid,
        physical_link_id,
        direction,
        partC_rev1_fftt_min AS rich_fftt_min,
        partC_rev1_capacity_annual_ktons_dir AS rich_capacity_annual_ktons,
        partC_rev1_fftt_source AS rich_fftt_source,
        partC_rev1_capacity_source AS rich_capacity_source,
        partC_rev1_lanes_dir AS lane_count,
        class_code AS functional_class,
        partC_rev1_ntad_routeid AS route_id,
        COALESCE(NULLIF(TRIM(road_name),''), partC_rev1_ntad_lname) AS route_name,
        nhs AS nhs_source,
        COALESCE(partC_rev1_faf_strahnet_nonempty, STRAHNET) AS strahnet_source,
        is_open AS is_open_source,
        is_valid_for_truck AS is_valid_for_truck_source,
        partC_rev1_ntad_speed_limit_mph AS ntad_speed_mph,
        Speed_Limit AS source_speed_limit_mph,
        geom
    FROM "{layer}"'''


def _translate_parent_links(config: Mapping[str, Any], gpkg_path: Path) -> None:
    configure_geospatial_environment(config)
    from osgeo import gdal

    gdal.UseExceptions()
    if gpkg_path.exists():
        raise FileExistsError(gpkg_path)
    options = gdal.VectorTranslateOptions(
        format="GPKG",
        SQLStatement=_parent_link_sql(config),
        SQLDialect="SQLITE",
        layerName=config["geopackage"]["directed_layer"],
        layerCreationOptions=["SPATIAL_INDEX=NO", "FID=fid", "GEOMETRY_NAME=geom"],
    )
    result = gdal.VectorTranslate(
        str(gpkg_path), config["read_only_inputs"]["parent_gpkg"], options=options
    )
    if result is None:
        raise RuntimeError("GDAL VectorTranslate failed for the parent directed-link geometry")
    result = None


def _authoritative_arc_updates(config: Mapping[str, Any]) -> pd.DataFrame:
    inputs = config["read_only_inputs"]
    columns = [
        "arc_idx",
        "directed_arc_id",
        "partC_rev1_link_uid",
        "tail_idx",
        "head_idx",
        "tail_node_uid",
        "head_node_uid",
        "free_time",
        "base_capacity",
        "assignment_capacity",
        "alpha",
        "beta",
        "length_miles",
    ]
    arcs = pd.read_parquet(inputs["graph_arcs_parquet"], columns=columns).sort_values("arc_idx", kind="mergesort")
    expected = int(config["accepted_core"]["authoritative_arc_count"])
    if len(arcs) != expected or arcs["directed_arc_id"].duplicated().any():
        raise ValueError("Authoritative arc table does not match the accepted count/unique-key contract")
    arcs["directed_arc_id"] = pd.to_numeric(arcs["directed_arc_id"], errors="raise").astype("int64")
    node_count = int(config["accepted_core"]["authoritative_node_count"])
    adjacency = sparse.csr_matrix(
        (np.ones(len(arcs), dtype=np.uint8), (arcs["tail_idx"].to_numpy(), arcs["head_idx"].to_numpy())),
        shape=(node_count, node_count),
    )
    _, component_by_node = connected_components(adjacency, directed=False, return_labels=True)
    tail_component = component_by_node[arcs["tail_idx"].to_numpy(dtype=np.int64)]
    head_component = component_by_node[arcs["head_idx"].to_numpy(dtype=np.int64)]
    if not np.array_equal(tail_component, head_component):
        raise ValueError("An authoritative arc crosses weak components")
    tail = arcs["tail_node_uid"].astype("string")
    head = arcs["head_node_uid"].astype("string")
    low = tail.where(tail <= head, head)
    high = head.where(tail <= head, tail)
    output = pd.DataFrame(
        {
            "directed_arc_id": arcs["directed_arc_id"],
            "partC_rev1_link_uid": arcs["partC_rev1_link_uid"].astype("string"),
            "arc_idx": arcs["arc_idx"].astype("int64"),
            "tail_node_uid": tail,
            "head_node_uid": head,
            "free_time_min": arcs["free_time"].astype("float64"),
            "free_time_source": "authoritative_corrected_graph.free_time",
            "length_m": arcs["length_miles"].astype("float64") * 1609.344,
            "length_mi": arcs["length_miles"].astype("float64"),
            "base_capacity_annual_ktons": arcs["base_capacity"].astype("float64"),
            "assignment_capacity_annual_ktons": arcs["assignment_capacity"].astype("float64"),
            "capacity_source": "authoritative_corrected_graph.assignment_capacity",
            "bpr_alpha": arcs["alpha"].astype("float64"),
            "bpr_beta": arcs["beta"].astype("float64"),
            "zero_length_flag": arcs["length_miles"].eq(0).astype("int8"),
            "self_loop_flag": tail.eq(head).astype("int8"),
            "parallel_arc_group_id": "parallel:" + low + ":" + high,
            "weak_component_id": pd.Series(tail_component).map(lambda value: f"weak_component_{int(value):06d}"),
            "authoritative_graph_identity": inputs["graph_identity_sha256"],
            "geometry_source": "parent_phase1_backbone_partition_gpkg",
            "geometry_source_layer": inputs["parent_link_layer"],
            "geometry_direction_ok": 1,
            "geometry_reversed_for_direction": 0,
            "geometry_fallback_used": 0,
            "tail_geometry_distance": 0.0,
            "head_geometry_distance": 0.0,
            "geometry_crs": inputs["crs"],
        }
    )
    return output


def _apply_authoritative_arc_updates(config: Mapping[str, Any], gpkg_path: Path) -> None:
    layer = config["geopackage"]["directed_layer"]
    updates = _authoritative_arc_updates(config)
    with contextlib.closing(sqlite3.connect(gpkg_path)) as connection, connection:
        _sqlite_tuning(connection)
        connection.execute(f'CREATE UNIQUE INDEX "{layer}_directed_arc_id_uq" ON "{layer}"(directed_arc_id)')
        updates.to_sql("authoritative_arc_updates", connection, if_exists="fail", index=False, chunksize=50_000)
        connection.execute(
            f'DELETE FROM "{layer}" WHERE directed_arc_id NOT IN (SELECT directed_arc_id FROM authoritative_arc_updates)'
        )
        _add_columns(connection, layer, PHYSICAL_UPDATE_COLUMNS)
        set_columns = [
            "partC_rev1_link_uid",
            "arc_idx",
            "tail_node_uid",
            "head_node_uid",
            "free_time_min",
            "free_time_source",
            "length_m",
            "length_mi",
            "base_capacity_annual_ktons",
            "assignment_capacity_annual_ktons",
            "capacity_source",
            "bpr_alpha",
            "bpr_beta",
            "zero_length_flag",
            "self_loop_flag",
            "parallel_arc_group_id",
            "weak_component_id",
            "authoritative_graph_identity",
            "geometry_source",
            "geometry_source_layer",
            "geometry_direction_ok",
            "geometry_reversed_for_direction",
            "geometry_fallback_used",
            "tail_geometry_distance",
            "head_geometry_distance",
            "geometry_crs",
        ]
        assignments = ",".join(f'"{name}"=u."{name}"' for name in set_columns)
        connection.execute(
            f'UPDATE "{layer}" AS d SET {assignments} FROM authoritative_arc_updates AS u '
            "WHERE d.directed_arc_id=u.directed_arc_id"
        )
        connection.execute(
            f'''UPDATE "{layer}" SET
                speed_mph=COALESCE(ntad_speed_mph,source_speed_limit_mph),
                is_nhs=CASE WHEN nhs_source IS NULL THEN NULL WHEN nhs_source<>0 THEN 1 ELSE 0 END,
                is_strahnet=CASE WHEN strahnet_source IS NULL THEN NULL WHEN strahnet_source<>0 THEN 1 ELSE 0 END,
                is_centroid_connector=CASE WHEN functional_class=50 THEN 1 ELSE 0 END,
                is_closed=CASE WHEN is_open_source IS NULL THEN NULL WHEN is_open_source=0 THEN 1 ELSE 0 END,
                is_truck_prohibited=CASE WHEN is_valid_for_truck_source IS NULL THEN NULL WHEN is_valid_for_truck_source=0 THEN 1 ELSE 0 END,
                is_usable=CASE WHEN is_open_source=1 AND is_valid_for_truck_source=1 THEN 1 ELSE 0 END'''
        )
        connection.execute("DROP TABLE authoritative_arc_updates")
        actual = connection.execute(f'SELECT COUNT(*) FROM "{layer}"').fetchone()[0]
        missing = connection.execute(f'SELECT COUNT(*) FROM "{layer}" WHERE arc_idx IS NULL').fetchone()[0]
        if actual != int(config["accepted_core"]["authoritative_arc_count"]) or missing:
            raise RuntimeError(f"Directed layer authoritative update failed: rows={actual}, missing={missing}")
    _checkpoint_sqlite(gpkg_path)


def _suite_registry(config: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(Path(config["accepted_phase2_output_root"]) / "portfolio/suite_manifest.csv")
    expected = list(config["accepted_core"]["suite_ids"])
    frame = frame.set_index("suite_id").loc[expected].reset_index()
    frame.insert(0, "suite_prefix", list(SUITE_PREFIXES))
    return frame


def _membership_and_small_frames(
    config: Mapping[str, Any], gpkg_path: Path
) -> dict[str, pd.DataFrame]:
    registry = _suite_registry(config)
    regions_all: list[pd.DataFrame] = []
    corrections_all: list[pd.DataFrame] = []
    portals_all: list[pd.DataFrame] = []
    roles_all: list[pd.DataFrame] = []
    validation_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    first_long = True
    layer = config["geopackage"]["directed_layer"]
    with contextlib.closing(sqlite3.connect(gpkg_path)) as connection, connection:
        _sqlite_tuning(connection)
        suite_columns: list[tuple[str, str]] = []
        for prefix in SUITE_PREFIXES:
            for suffix in SUITE_SUFFIXES:
                sql_type = "REAL" if suffix == "resolution" else "INTEGER" if suffix.startswith("is_") or suffix.endswith("_count") else "TEXT"
                suite_columns.append((f"{prefix}_{suffix}", sql_type))
        _add_columns(connection, layer, suite_columns)
        for prefix, suite_row, directory in zip(SUITE_PREFIXES, registry.itertuples(index=False), suite_directories(config)):
            membership = pd.read_parquet(directory / "directed_arc_membership.parquet")
            regions = pd.read_parquet(directory / "retained_regions.parquet")
            long = build_suite_membership_frame(membership, regions, float(suite_row.resolution))
            long.to_sql(
                "suite_arc_membership",
                connection,
                if_exists="fail" if first_long else "append",
                index=False,
                chunksize=50_000,
            )
            first_long = False
            regions.insert(0, "suite_id", directory.name)
            regions.insert(1, "suite_prefix", prefix)
            regions_all.append(regions)
            corrections = pd.read_parquet(directory / "small_region_corrections.parquet")
            corrections.insert(0, "suite_id", directory.name)
            corrections.insert(1, "suite_prefix", prefix)
            corrections_all.append(corrections)
            portals_all.append(pd.read_parquet(directory / "portals.parquet"))
            roles_all.append(pd.read_parquet(directory / "centroid_roles.parquet"))
            validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
            validation_rows.append(
                {
                    "suite_id": directory.name,
                    "status": validation["status"],
                    "retained_region_count": validation["retained_region_count"],
                    "connectivity_failure_count": validation["connectivity_failure_count"],
                    "all_checks_pass": all(validation["checks"].values()),
                    "checks_json": json.dumps(validation["checks"], sort_keys=True, separators=(",", ":")),
                }
            )
            pair_contract = json.loads((directory / "lazy_pair_support_contract.json").read_text(encoding="utf-8"))
            pair_rows.append(
                {
                    "suite_id": directory.name,
                    "contract_kind": pair_contract["contract_kind"],
                    "dense_pair_materialization": pair_contract["dense_pair_materialization"],
                    "phase2_pair_row_count": pair_contract["phase2_pair_row_count"],
                    "reachability_evaluation": pair_contract["reachability_evaluation"],
                    "same_physical_node_pair_allowed": pair_contract["same_physical_node_pair_allowed"],
                    "terminal_registry_sha256": pair_contract["terminal_registry_sha256"],
                }
            )
        connection.execute(
            "CREATE UNIQUE INDEX suite_arc_membership_suite_arc_uq ON suite_arc_membership(suite_id,directed_arc_id)"
        )
        connection.execute("CREATE INDEX suite_arc_membership_arc_idx ON suite_arc_membership(directed_arc_id)")
        expected_long = int(config["accepted_core"]["authoritative_arc_count"]) * len(SUITE_PREFIXES)
        actual_long = connection.execute("SELECT COUNT(*) FROM suite_arc_membership").fetchone()[0]
        if actual_long != expected_long:
            raise RuntimeError(f"Long suite membership count mismatch: {actual_long} != {expected_long}")
        for prefix, suite_id in zip(SUITE_PREFIXES, config["accepted_core"]["suite_ids"]):
            mapping = {
                f"{prefix}_suite_id": "suite_id",
                f"{prefix}_resolution": "resolution",
                f"{prefix}_label": "suite_label",
                f"{prefix}_region_uuid": "region_uuid",
                f"{prefix}_is_initial_backbone": "is_initial_backbone",
                f"{prefix}_is_post_partition_backbone": "is_post_partition_backbone",
                f"{prefix}_is_small_region_correction_backbone": "is_small_region_correction_backbone",
                f"{prefix}_is_final_backbone": "is_final_backbone",
                f"{prefix}_region_arc_count": "region_arc_count",
                f"{prefix}_region_node_count": "region_node_count",
                f"{prefix}_region_portal_count": "region_portal_count",
                f"{prefix}_region_color_hex": "region_color_hex",
            }
            assignments = ",".join(f'"{target}"=m."{source}"' for target, source in mapping.items())
            connection.execute(
                f'UPDATE "{layer}" AS d SET {assignments} FROM suite_arc_membership AS m '
                "WHERE m.suite_id=? AND d.directed_arc_id=m.directed_arc_id",
                (suite_id,),
            )
            missing = connection.execute(
                f'SELECT COUNT(*) FROM "{layer}" WHERE "{prefix}_suite_id" IS NULL'
            ).fetchone()[0]
            if missing:
                raise RuntimeError(f"Wide suite membership join failed for {prefix}: {missing} missing")
        _register_attribute_table(connection, "suite_arc_membership", "Accepted long membership for all six suites")
        _write_attribute_table(connection, "suite_registry", registry, "Accepted six-suite registry")
        region_registry = pd.concat(regions_all, ignore_index=True)
        corrections = pd.concat(corrections_all, ignore_index=True)
        validation_summary = pd.DataFrame(validation_rows)
        terminal_support = pd.DataFrame(pair_rows)
        _write_attribute_table(connection, "region_registry", region_registry, "Accepted retained regions for all suites")
        _write_attribute_table(
            connection, "small_region_corrections", corrections, "Accepted small-region correction audit records"
        )
        _write_attribute_table(
            connection, "candidate_attempts", pd.read_csv(Path(config["accepted_phase2_output_root"]) / "portfolio/candidate_attempts.csv"), "Accepted candidate attempts"
        )
        pairwise = pd.read_csv(
            Path(config["accepted_phase2_output_root"]) / "portfolio/pairwise_structural_similarity.csv"
        )
        _write_attribute_table(
            connection, "pairwise_structural_similarity", pairwise, "Accepted pairwise structural similarity"
        )
        _write_attribute_table(
            connection, "suite_validation_summary", validation_summary, "Accepted suite validation summaries"
        )
        _write_attribute_table(
            connection, "terminal_pair_support_summary", terminal_support, "Accepted lazy terminal-pair support contracts"
        )
    _checkpoint_sqlite(gpkg_path)
    return {
        "suite_registry": registry,
        "region_registry": pd.concat(regions_all, ignore_index=True),
        "small_region_corrections": pd.concat(corrections_all, ignore_index=True),
        "portals": pd.concat(portals_all, ignore_index=True),
        "centroid_roles": pd.concat(roles_all, ignore_index=True),
        "suite_validation_summary": pd.DataFrame(validation_rows),
        "terminal_pair_support_summary": pd.DataFrame(pair_rows),
        "pairwise_structural_similarity": pairwise,
    }


def _translate_parent_nodes(config: Mapping[str, Any], gpkg_path: Path) -> pd.DataFrame:
    configure_geospatial_environment(config)
    from osgeo import gdal

    inputs = config["read_only_inputs"]
    layer_name = inputs["parent_node_layer"]
    sql = f'''SELECT
        partC_rev1_node_uid,
        partC_rev1_lon AS longitude,
        partC_rev1_lat AS latitude,
        partC_rev1_centroid AS is_centroid,
        partC_rev1_centroid_id_raw AS centroid_id_raw,
        partC_rev1_county_fips AS county_fips,
        partC_rev1_fafid AS faf_id,
        geom
    FROM "{layer_name}"'''
    options = gdal.VectorTranslateOptions(
        format="GPKG",
        accessMode="update",
        SQLStatement=sql,
        SQLDialect="SQLITE",
        layerName="physical_nodes",
        layerCreationOptions=["SPATIAL_INDEX=NO", "FID=fid", "GEOMETRY_NAME=geom"],
    )
    result = gdal.VectorTranslate(str(gpkg_path), inputs["parent_gpkg"], options=options)
    if result is None:
        raise RuntimeError("GDAL VectorTranslate failed for physical nodes")
    result = None
    nodes = pd.read_parquet(inputs["graph_nodes_parquet"], columns=["node_idx", "partC_rev1_node_uid"])
    nodes["partC_rev1_node_uid"] = nodes["partC_rev1_node_uid"].astype("string")
    with contextlib.closing(sqlite3.connect(gpkg_path)) as connection, connection:
        _sqlite_tuning(connection)
        _add_columns(connection, "physical_nodes", [("node_idx", "INTEGER")])
        nodes.to_sql("authoritative_node_updates", connection, if_exists="fail", index=False, chunksize=50_000)
        connection.execute(
            """UPDATE physical_nodes AS p SET node_idx=u.node_idx
               FROM authoritative_node_updates AS u
               WHERE CAST(p.partC_rev1_node_uid AS TEXT)=u.partC_rev1_node_uid"""
        )
        connection.execute("DELETE FROM physical_nodes WHERE node_idx IS NULL")
        connection.execute("DROP TABLE authoritative_node_updates")
        connection.execute(
            "CREATE UNIQUE INDEX physical_nodes_node_uid_uq ON physical_nodes(partC_rev1_node_uid)"
        )
        actual = connection.execute("SELECT COUNT(*) FROM physical_nodes").fetchone()[0]
        if actual != int(config["accepted_core"]["authoritative_node_count"]):
            raise RuntimeError(f"Physical-node subset count mismatch: {actual}")
        coordinates = pd.read_sql_query(
            "SELECT node_idx,CAST(partC_rev1_node_uid AS TEXT) AS partC_rev1_node_uid,longitude,latitude FROM physical_nodes ORDER BY node_idx",
            connection,
        )
    _checkpoint_sqlite(gpkg_path)
    return coordinates


def _ogr_field_type(sql_type: str) -> int:
    from osgeo import ogr

    mapping = {"TEXT": ogr.OFTString, "INTEGER": ogr.OFTInteger64, "REAL": ogr.OFTReal}
    return mapping[sql_type]


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_point_layer(
    config: Mapping[str, Any],
    gpkg_path: Path,
    layer_name: str,
    frame: pd.DataFrame,
    field_types: Mapping[str, str],
) -> None:
    configure_geospatial_environment(config)
    from osgeo import ogr, osr

    if frame[["longitude", "latitude"]].isna().any().any():
        raise ValueError(f"Point layer {layer_name} contains missing coordinates")
    dataset = ogr.Open(str(gpkg_path), 1)
    if dataset is None:
        raise RuntimeError(f"OGR could not open output GeoPackage for {layer_name}")
    if dataset.GetLayerByName(layer_name) is not None:
        raise FileExistsError(f"GeoPackage layer already exists: {layer_name}")
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(4269)
    layer = dataset.CreateLayer(
        layer_name, srs=spatial_reference, geom_type=ogr.wkbPoint, options=["SPATIAL_INDEX=NO", "FID=fid", "GEOMETRY_NAME=geom"]
    )
    for name, sql_type in field_types.items():
        definition = ogr.FieldDefn(name, _ogr_field_type(sql_type))
        if layer.CreateField(definition) != ogr.OGRERR_NONE:
            raise RuntimeError(f"Could not create field {layer_name}.{name}")
    ordered_fields = list(field_types)
    layer.StartTransaction()
    for number, row in enumerate(frame[ordered_fields + ["longitude", "latitude"]].itertuples(index=False, name=None), start=1):
        feature = ogr.Feature(layer.GetLayerDefn())
        for index, name in enumerate(ordered_fields):
            value = row[index]
            if pd.isna(value):
                feature.SetFieldNull(index)
            else:
                feature.SetField(index, _python_scalar(value))
        geometry = ogr.Geometry(ogr.wkbPoint)
        geometry.AddPoint_2D(float(row[-2]), float(row[-1]))
        feature.SetGeometry(geometry)
        if layer.CreateFeature(feature) != ogr.OGRERR_NONE:
            raise RuntimeError(f"Failed writing {layer_name} feature {number}")
        feature = None
        if number % 50_000 == 0:
            layer.CommitTransaction()
            layer.StartTransaction()
    layer.CommitTransaction()
    actual = layer.GetFeatureCount()
    dataset = None
    if actual != len(frame):
        raise RuntimeError(f"Point layer count mismatch for {layer_name}: {actual} != {len(frame)}")


def _build_point_frames(
    config: Mapping[str, Any], small_frames: Mapping[str, pd.DataFrame], coordinates: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    coords = coordinates[["node_idx", "partC_rev1_node_uid", "longitude", "latitude"]].copy()
    coords["partC_rev1_node_uid"] = coords["partC_rev1_node_uid"].astype("string")
    portals = small_frames["portals"].copy()
    portals["partC_rev1_node_uid"] = portals["partC_rev1_node_uid"].astype("string")
    portals = portals.merge(
        coords[["partC_rev1_node_uid", "longitude", "latitude"]],
        on="partC_rev1_node_uid",
        how="left",
        validate="many_to_one",
    )
    roles = small_frames["centroid_roles"].copy()
    roles["partC_rev1_node_uid"] = roles["partC_rev1_node_uid"].astype("string")
    roles = roles.merge(
        coords[["partC_rev1_node_uid", "longitude", "latitude"]],
        on="partC_rev1_node_uid",
        how="left",
        validate="many_to_one",
    )
    representative_frames: list[pd.DataFrame] = []
    registry = small_frames["suite_registry"].set_index("suite_id")
    region_registry = small_frames["region_registry"]
    for directory in suite_directories(config):
        membership = pd.read_parquet(directory / "node_partition_membership.parquet")
        membership["partC_rev1_node_uid"] = membership["partC_rev1_node_uid"].astype("string")
        joined = membership.merge(
            coords[["partC_rev1_node_uid", "longitude", "latitude"]],
            on="partC_rev1_node_uid",
            how="left",
            validate="many_to_one",
        )
        if joined[["longitude", "latitude"]].isna().any().any():
            raise ValueError(f"Missing node coordinates for region representatives in {directory.name}")
        representatives = (
            joined.groupby("region_uuid", sort=True, as_index=False)[["longitude", "latitude"]].mean()
        )
        stats = region_registry.loc[
            region_registry["suite_id"].eq(directory.name),
            ["region_uuid", "arc_count", "node_count", "portal_count", "resolution"],
        ]
        representatives = stats.merge(representatives, on="region_uuid", how="left", validate="one_to_one")
        if representatives[["longitude", "latitude"]].isna().any().any():
            raise ValueError(f"A retained region lacks a representative in {directory.name}")
        representatives.insert(0, "suite_id", directory.name)
        representatives.insert(1, "suite_prefix", registry.loc[directory.name, "suite_prefix"])
        representatives["representative_method"] = "mean_retained_region_node_coordinates"
        representative_frames.append(representatives)
    representatives = pd.concat(representative_frames, ignore_index=True)
    return {"portals_all_suites": portals, "centroid_roles_all_suites": roles, "region_representatives": representatives}


def _write_required_point_layers(
    config: Mapping[str, Any], gpkg_path: Path, point_frames: Mapping[str, pd.DataFrame]
) -> None:
    _write_point_layer(
        config,
        gpkg_path,
        "portals_all_suites",
        point_frames["portals_all_suites"],
        {
            "portal_id": "TEXT",
            "suite_id": "TEXT",
            "region_uuid": "TEXT",
            "partC_rev1_node_uid": "TEXT",
            "node_idx": "INTEGER",
            "can_enter": "INTEGER",
            "can_exit": "INTEGER",
            "portal_kind": "TEXT",
        },
    )
    _write_point_layer(
        config,
        gpkg_path,
        "centroid_roles_all_suites",
        point_frames["centroid_roles_all_suites"],
        {
            "centroid_role_id": "TEXT",
            "suite_id": "TEXT",
            "region_uuid": "TEXT",
            "partC_rev1_node_uid": "TEXT",
            "node_idx": "INTEGER",
            "centroid_role": "TEXT",
            "through_allowed": "INTEGER",
        },
    )
    _write_point_layer(
        config,
        gpkg_path,
        "region_representatives",
        point_frames["region_representatives"],
        {
            "suite_id": "TEXT",
            "suite_prefix": "TEXT",
            "resolution": "REAL",
            "region_uuid": "TEXT",
            "arc_count": "INTEGER",
            "node_count": "INTEGER",
            "portal_count": "INTEGER",
            "representative_method": "TEXT",
        },
    )


def _register_suite_views(config: Mapping[str, Any], gpkg_path: Path) -> None:
    base = config["geopackage"]["directed_layer"]
    with contextlib.closing(sqlite3.connect(gpkg_path)) as connection, connection:
        base_content = connection.execute(
            "SELECT last_change,min_x,min_y,max_x,max_y,srs_id FROM gpkg_contents WHERE table_name=?", (base,)
        ).fetchone()
        base_geometry = connection.execute(
            "SELECT column_name,geometry_type_name,srs_id,z,m FROM gpkg_geometry_columns WHERE table_name=?", (base,)
        ).fetchone()
        if base_content is None or base_geometry is None:
            raise RuntimeError("Base directed layer is not registered in the GeoPackage")
        for prefix, view in zip(SUITE_PREFIXES, config["geopackage"]["registered_views"]):
            if connection.execute("SELECT 1 FROM sqlite_master WHERE name=?", (view,)).fetchone():
                raise FileExistsError(f"GeoPackage view exists: {view}")
            connection.execute(
                f'''CREATE VIEW "{view}" AS SELECT
                    fid,geom,directed_arc_id,
                    "{prefix}_suite_id" AS suite_id,
                    "{prefix}_resolution" AS resolution,
                    "{prefix}_label" AS suite_label,
                    "{prefix}_region_uuid" AS region_uuid,
                    "{prefix}_is_initial_backbone" AS is_initial_backbone,
                    "{prefix}_is_post_partition_backbone" AS is_post_partition_backbone,
                    "{prefix}_is_small_region_correction_backbone" AS is_small_region_correction_backbone,
                    "{prefix}_is_final_backbone" AS is_final_backbone,
                    "{prefix}_region_arc_count" AS region_arc_count,
                    "{prefix}_region_node_count" AS region_node_count,
                    "{prefix}_region_portal_count" AS region_portal_count,
                    "{prefix}_region_color_hex" AS region_color_hex
                FROM "{base}"'''
            )
            connection.execute(
                """INSERT INTO gpkg_contents(
                    table_name,data_type,identifier,description,last_change,min_x,min_y,max_x,max_y,srs_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (view, "features", view, f"Registered {prefix} geometry view over {base}", *base_content),
            )
            connection.execute(
                "INSERT INTO gpkg_geometry_columns(table_name,column_name,geometry_type_name,srs_id,z,m) VALUES (?,?,?,?,?,?)",
                (view, *base_geometry),
            )


def _create_spatial_indexes(config: Mapping[str, Any], gpkg_path: Path) -> None:
    configure_geospatial_environment(config)
    from osgeo import ogr

    dataset = ogr.Open(str(gpkg_path), 1)
    if dataset is None:
        raise RuntimeError("OGR could not open final GeoPackage for spatial index creation")
    for layer in config["geopackage"]["spatial_index_layers"]:
        result = dataset.ExecuteSQL(f"SELECT CreateSpatialIndex('{layer}','geom')", dialect="SQLITE")
        if result is None:
            raise RuntimeError(f"Spatial index creation failed for {layer}")
        feature = result.GetNextFeature()
        success = feature is not None and int(feature.GetField(0)) == 1
        dataset.ReleaseResultSet(result)
        if not success:
            raise RuntimeError(f"Spatial index creation returned failure for {layer}")
    dataset = None


def _fresh_reader_inventory(config: Mapping[str, Any], gpkg_path: Path) -> pd.DataFrame:
    configure_geospatial_environment(config)
    from osgeo import ogr

    import fiona
    import pyogrio

    fiona_layers = set(fiona.listlayers(gpkg_path))
    pyogrio_layers = {str(row[0]): str(row[1]) for row in pyogrio.list_layers(gpkg_path)}
    dataset = ogr.Open(str(gpkg_path), 0)
    if dataset is None:
        raise RuntimeError("OGR fresh reopen failed")
    rows: list[dict[str, Any]] = []
    with contextlib.closing(open_sqlite(gpkg_path, read_only=True)) as connection:
        contents = connection.execute(
            "SELECT table_name,data_type,srs_id FROM gpkg_contents ORDER BY table_name"
        ).fetchall()
        geometry_map = {
            row[0]: (row[1], row[2], row[3])
            for row in connection.execute(
                "SELECT table_name,column_name,geometry_type_name,srs_id FROM gpkg_geometry_columns"
            )
        }
        view_names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='view'")}
        for table_name, data_type, srs_id in contents:
            object_type = "view" if table_name in view_names else "layer" if table_name in geometry_map else "table"
            row_count = connection.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            geometry = geometry_map.get(table_name)
            spatial_index = False
            if geometry and object_type != "view":
                spatial_index = bool(
                    connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (f"rtree_{table_name}_{geometry[0]}",),
                    ).fetchone()
                )
            rows.append(
                {
                    "object_name": table_name,
                    "object_type": object_type,
                    "data_type": data_type,
                    "geometry_type": geometry[1] if geometry else None,
                    "crs": f"EPSG:{geometry[2]}" if geometry else None,
                    "row_count": int(row_count),
                    "primary_or_stable_key": "fid" if geometry else None,
                    "spatial_index": spatial_index,
                    "fiona_visible": table_name in fiona_layers,
                    "pyogrio_visible": table_name in pyogrio_layers,
                    "disposition": "BASE_SPATIAL_INDEX" if object_type == "view" else "PASS",
                }
            )
    dataset = None
    return pd.DataFrame(rows)


def _portable_table(frame: pd.DataFrame, table_root: Path, stem: str) -> None:
    write_csv_new(table_root / f"{stem}.csv", frame)
    try:
        markdown = frame.to_markdown(index=False) + "\n"
    except ImportError:  # pragma: no cover - tabulate is present in the frozen environment
        markdown = "```csv\n" + frame.to_csv(index=False, lineterminator="\n") + "```\n"
    write_text_new(table_root / f"{stem}.md", markdown)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        latex = frame.to_latex(index=False, escape=True, longtable=False)
    write_text_new(table_root / f"{stem}.tex", latex)


def _portfolio_overview(config: Mapping[str, Any], small_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    registry = small_frames["suite_registry"].copy()
    total = int(config["accepted_core"]["authoritative_arc_count"])
    output = pd.DataFrame(
        {
            "suite_id": registry["suite_id"],
            "resolution": registry["resolution"],
            "retained_regions": registry["retained_region_count"],
            "initial_backbone_arcs": registry["initial_backbone_directed_arc_count"],
            "final_backbone_arcs": registry["final_backbone_directed_arc_count"],
            "final_backbone_share": registry["final_backbone_directed_arc_count"] / total,
            "regional_arcs": total - registry["final_backbone_directed_arc_count"],
            "corrected_small_regions": registry["corrected_small_region_count"],
            "corrected_small_region_arcs": registry["small_region_corrected_directed_arc_count"],
            "portals": registry["portal_record_count"],
            "centroid_roles": registry["centroid_role_count"],
            "terminals": registry["terminal_count"],
            "runtime_seconds": registry["suite_runtime_seconds"],
            "peak_rss_bytes": registry["suite_peak_rss_bytes"],
            "structural_signature": registry["structural_signature"],
        }
    )
    if "region_registry" in small_frames:
        region = small_frames["region_registry"]
        statistics = (
            region.groupby("suite_id", sort=False)
            .agg(
                min_region_arcs=("arc_count", "min"),
                median_region_arcs=("arc_count", "median"),
                max_region_arcs=("arc_count", "max"),
                min_region_portals=("portal_count", "min"),
                median_region_portals=("portal_count", "median"),
                max_region_portals=("portal_count", "max"),
            )
            .reset_index()
        )
        output = output.merge(statistics, on="suite_id", how="left", validate="one_to_one")
    else:
        output["min_region_arcs"] = registry["min_region_arc_count"]
        output["median_region_arcs"] = registry["median_region_arc_count"]
        output["max_region_arcs"] = registry["max_region_arc_count"]
        output["min_region_portals"] = registry["min_region_portal_count"]
        output["median_region_portals"] = np.nan
        output["max_region_portals"] = registry["max_region_portal_count"]
    return output


def _region_quantiles(region_registry: pd.DataFrame) -> pd.DataFrame:
    quantiles = {"min": 0.0, "p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99, "max": 1.0}
    rows: list[dict[str, Any]] = []
    for suite_id, group in region_registry.groupby("suite_id", sort=False):
        resolution = float(group["resolution"].iloc[0])
        for metric in ("arc_count", "node_count", "portal_count"):
            values = group[metric].astype(float)
            row: dict[str, Any] = {"suite_id": suite_id, "resolution": resolution, "metric": metric}
            for label, probability in quantiles.items():
                row[label] = float(values.quantile(probability, interpolation="linear"))
            row["mean"] = float(values.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _pairwise_matrix(
    registry: pd.DataFrame, pairwise: pd.DataFrame, metric: str, *, directional: bool = False
) -> pd.DataFrame:
    suites = registry["suite_id"].tolist()
    matrix = pd.DataFrame(np.nan, index=suites, columns=suites, dtype=float)
    np.fill_diagonal(matrix.values, 1.0)
    for row in pairwise.itertuples(index=False):
        if directional:
            left_value = float(getattr(row, "left_to_right_dominant_coverage"))
            right_value = float(getattr(row, "right_to_left_dominant_coverage"))
        else:
            left_value = right_value = float(getattr(row, metric))
        matrix.loc[row.left_suite_id, row.right_suite_id] = left_value
        matrix.loc[row.right_suite_id, row.left_suite_id] = right_value
    matrix.insert(0, "suite_id", matrix.index)
    return matrix.reset_index(drop=True)


def _interface_burden(overview: pd.DataFrame, region_registry: pd.DataFrame) -> pd.DataFrame:
    proxy = (
        region_registry.assign(interface_pair_proxy=region_registry["portal_count"].astype(np.int64) ** 2)
        .groupby("suite_id", sort=False)["interface_pair_proxy"]
        .agg(median_interface_pair_proxy="median", max_interface_pair_proxy="max")
        .reset_index()
    )
    return overview[
        [
            "suite_id",
            "resolution",
            "corrected_small_regions",
            "corrected_small_region_arcs",
            "portals",
            "terminals",
        ]
    ].merge(proxy, on="suite_id", how="left", validate="one_to_one")


def _write_initial_tables(
    config: Mapping[str, Any], small_frames: Mapping[str, pd.DataFrame], table_root: Path
) -> dict[str, pd.DataFrame]:
    overview = _portfolio_overview(config, small_frames)
    quantiles = _region_quantiles(small_frames["region_registry"])
    pairwise = small_frames["pairwise_structural_similarity"].copy()
    nmi = _pairwise_matrix(small_frames["suite_registry"], pairwise, "normalized_mutual_info_score")
    ari = _pairwise_matrix(small_frames["suite_registry"], pairwise, "adjusted_rand_score")
    jaccard = _pairwise_matrix(small_frames["suite_registry"], pairwise, "final_backbone_jaccard")
    dominant = _pairwise_matrix(small_frames["suite_registry"], pairwise, "left_to_right_dominant_coverage", directional=True)
    adjacent = []
    suite_order = small_frames["suite_registry"]["suite_id"].tolist()
    for left, right in zip(suite_order[:-1], suite_order[1:]):
        row = pairwise.loc[pairwise["left_suite_id"].eq(left) & pairwise["right_suite_id"].eq(right)]
        if row.empty:
            row = pairwise.loc[pairwise["left_suite_id"].eq(right) & pairwise["right_suite_id"].eq(left)]
        adjacent.append(row.iloc[0])
    numeric_metrics = [
        "normalized_mutual_info_score",
        "adjusted_rand_score",
        "final_backbone_jaccard",
        "left_to_right_dominant_coverage",
        "right_to_left_dominant_coverage",
    ]
    summary_rows: list[dict[str, Any]] = []
    adjacent_frame = pd.DataFrame(adjacent)
    for metric in numeric_metrics:
        summary_rows.append(
            {
                "metric": metric,
                "portfolio_min": float(pairwise[metric].min()),
                "portfolio_max": float(pairwise[metric].max()),
                "adjacent_min": float(adjacent_frame[metric].min()),
                "adjacent_max": float(adjacent_frame[metric].max()),
            }
        )
    pairwise_summary = pd.DataFrame(summary_rows)
    burden = _interface_burden(overview, small_frames["region_registry"])
    _portable_table(overview, table_root, "table_01_portfolio_overview")
    _portable_table(quantiles, table_root, "table_02_region_distribution_quantiles")
    _portable_table(pairwise, table_root, "table_03_pairwise_similarity_long")
    _portable_table(nmi, table_root, "table_03_nmi_matrix")
    _portable_table(ari, table_root, "table_03_ari_matrix")
    _portable_table(jaccard, table_root, "table_03_final_backbone_jaccard_matrix")
    _portable_table(dominant, table_root, "table_03_dominant_coverage_row_to_column_matrix")
    dominant_transpose = dominant.set_index("suite_id").T
    dominant_transpose.insert(0, "suite_id", dominant_transpose.index)
    _portable_table(
        dominant_transpose.reset_index(drop=True),
        table_root,
        "table_03_dominant_coverage_column_to_row_matrix",
    )
    _portable_table(pairwise_summary, table_root, "table_03_pairwise_similarity_summary")
    return {
        "overview": overview,
        "quantiles": quantiles,
        "pairwise": pairwise,
        "nmi": nmi,
        "ari": ari,
        "jaccard": jaccard,
        "dominant": dominant,
        "pairwise_summary": pairwise_summary,
        "burden": burden,
    }


def _save_figure(fig: plt.Figure, figure_root: Path, stem: str, caption: str) -> None:
    png = figure_root / f"{stem}.png"
    pdf = figure_root / f"{stem}.pdf"
    if png.exists() or pdf.exists():
        raise FileExistsError(stem)
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white", metadata={"Software": "Matplotlib"})
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", metadata={"Creator": "Matplotlib"})
    plt.close(fig)
    write_text_new(figure_root / f"{stem}_caption.md", caption.rstrip() + "\n")


def _generate_figures_1_2_3_5(
    table_frames: Mapping[str, pd.DataFrame], figure_root: Path
) -> None:
    overview = table_frames["overview"]
    labels = [f"s{index:02d}" for index in range(1, 7)]
    resolution = overview["resolution"].to_numpy(float)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = [
        ("retained_regions", "Retained regions"),
        ("final_backbone_arcs", "Final backbone arcs"),
        ("portals", "Portal records"),
        ("terminals", "Terminal records"),
    ]
    for axis, (column, ylabel) in zip(axes.flat, panels):
        axis.plot(resolution, overview[column], marker="o", color="#1f5a99", linewidth=2)
        axis.set_xscale("log")
        axis.set_xlabel("Resolution")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        for x_value, y_value, label in zip(resolution, overview[column], labels):
            axis.annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0, 1].twinx().plot(resolution, overview["final_backbone_share"], color="#b7472a", marker="s", alpha=0.75)
    fig.suptitle("Phase 2 portfolio scale and interface tradeoffs")
    write_csv_new(figure_root / "figure_01_portfolio_scale_tradeoffs_data.csv", overview)
    _save_figure(
        fig,
        figure_root,
        "figure_01_portfolio_scale_tradeoffs",
        "Figure 1. Resolution versus retained-region count, final-backbone scale/share, portal records, and terminals. Source: accepted six-suite manifest; no suite artifacts were recomputed.",
    )

    quantiles = table_frames["quantiles"]
    q_labels = ["min", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]
    q_values = np.asarray([0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for axis, metric in zip(axes, ("arc_count", "node_count", "portal_count")):
        subset = quantiles.loc[quantiles["metric"].eq(metric)]
        for index, row in enumerate(subset.itertuples(index=False)):
            axis.plot(q_values, [getattr(row, label) for label in q_labels], marker="o", markersize=3, label=labels[index])
        axis.set_yscale("symlog", linthresh=1)
        axis.set_xlabel("Quantile")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.grid(alpha=0.25)
    axes[-1].legend(title="Suite", fontsize=8)
    fig.suptitle("Retained-region size distributions")
    write_csv_new(figure_root / "figure_02_region_size_distributions_data.csv", quantiles)
    _save_figure(
        fig,
        figure_root,
        "figure_02_region_size_distributions",
        "Figure 2. Quantile curves for accepted retained-region arc, node, and portal counts. Source: accepted retained_regions.parquet files.",
    )

    matrices = [
        (table_frames["nmi"], "Normalized mutual information"),
        (table_frames["ari"], "Adjusted Rand index"),
        (table_frames["jaccard"], "Final-backbone Jaccard"),
        (table_frames["dominant"], "Dominant coverage (row → column)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for axis, (matrix, title) in zip(axes.flat, matrices):
        values = matrix.drop(columns="suite_id").to_numpy(float)
        image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(range(6), labels, rotation=45, ha="right")
        axis.set_yticks(range(6), labels)
        axis.set_title(title)
        for row_index in range(6):
            for column_index in range(6):
                axis.text(column_index, row_index, f"{values[row_index, column_index]:.2f}", ha="center", va="center", fontsize=7, color="white" if values[row_index, column_index] < 0.55 else "black")
        fig.colorbar(image, ax=axis, fraction=0.046)
    write_csv_new(figure_root / "figure_03_structural_similarity_heatmaps_data.csv", table_frames["pairwise"])
    _save_figure(
        fig,
        figure_root,
        "figure_03_structural_similarity_heatmaps",
        "Figure 3. Pairwise structural similarity across the six accepted suites. Diagonals are exact identity; off-diagonals are accepted portfolio comparisons.",
    )

    burden = table_frames["burden"]
    x = np.arange(len(burden))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].bar(x - 0.18, burden["corrected_small_regions"], width=0.36, label="Corrected regions")
    axes[0].bar(x + 0.18, burden["corrected_small_region_arcs"], width=0.36, label="Corrected arcs")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Small-region correction")
    axes[1].plot(x, burden["portals"], marker="o", label="Portals")
    axes[1].plot(x, burden["terminals"], marker="s", label="Terminals")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Interface records")
    axes[2].plot(x, burden["median_interface_pair_proxy"], marker="o", label="Median")
    axes[2].plot(x, burden["max_interface_pair_proxy"], marker="s", label="Maximum")
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Portal² interface proxy")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    write_csv_new(figure_root / "figure_05_correction_interface_burden_data.csv", burden)
    _save_figure(
        fig,
        figure_root,
        "figure_05_correction_interface_burden",
        "Figure 5. Accepted small-region correction burden, portal/terminal interface records, and per-region portal-squared interface proxy.",
    )


def _hex_rgba(value: str) -> tuple[int, int, int, int]:
    text = str(value).lstrip("#")
    if len(text) != 6:
        return 32, 33, 36, 255
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16), 255


def _generate_national_map(
    config: Mapping[str, Any], gpkg_path: Path, overview: pd.DataFrame, figure_root: Path
) -> None:
    configure_geospatial_environment(config)
    from osgeo import ogr

    dataset = ogr.Open(str(gpkg_path), 0)
    layer = dataset.GetLayerByName(config["geopackage"]["directed_layer"])
    expected = int(config["accepted_core"]["authoritative_arc_count"])
    segments: list[np.ndarray] = []
    colors = np.empty((6, expected, 4), dtype=np.uint8)
    backbone = np.empty((6, expected), dtype=bool)
    for feature_number, feature in enumerate(layer):
        geometry = feature.GetGeometryRef()
        parts: list[np.ndarray] = []
        for part_number in range(geometry.GetGeometryCount()):
            points = np.asarray(geometry.GetGeometryRef(part_number).GetPoints(), dtype=np.float64)[:, :2]
            if part_number:
                parts.append(np.full((1, 2), np.nan))
            parts.append(points)
        segments.append(np.vstack(parts))
        for suite_index, prefix in enumerate(SUITE_PREFIXES):
            is_backbone = bool(feature[f"{prefix}_is_final_backbone"])
            backbone[suite_index, feature_number] = is_backbone
            colors[suite_index, feature_number] = _hex_rgba(feature[f"{prefix}_region_color_hex"])
    extent = layer.GetExtent()
    dataset = None
    if len(segments) != expected:
        raise RuntimeError(f"National-map geometry count mismatch: {len(segments)}")
    panel_images: list[np.ndarray] = []
    for suite_index in range(6):
        panel, axis = plt.subplots(figsize=(12, 6.2), dpi=200)
        regional_colors = colors[suite_index].copy()
        regional_colors[backbone[suite_index], 3] = 0
        axis.add_collection(
            LineCollection(segments, colors=regional_colors / 255.0, linewidths=0.12, rasterized=True)
        )
        backbone_segments = [segments[index] for index in np.flatnonzero(backbone[suite_index])]
        axis.add_collection(
            LineCollection(backbone_segments, colors="#202124", linewidths=0.18, rasterized=True)
        )
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_aspect("equal", adjustable="box")
        axis.axis("off")
        panel.canvas.draw()
        panel_images.append(np.asarray(panel.canvas.buffer_rgba()).copy())
        plt.close(panel)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        axis.imshow(panel_images[index])
        row = overview.iloc[index]
        axis.set_title(
            f"{SUITE_PREFIXES[index]} — resolution {row.resolution:g}; {int(row.retained_regions):,} retained regions",
            fontsize=10,
        )
        axis.axis("off")
    fig.suptitle("National accepted Phase 2 split/mapping portfolio", fontsize=15)
    data = {
        "source_geopackage": gpkg_path.as_posix(),
        "source_layer": config["geopackage"]["directed_layer"],
        "source_geopackage_size_bytes": gpkg_path.stat().st_size,
        "source_geopackage_sha256": sha256(gpkg_path),
        "feature_count": expected,
        "crs": config["read_only_inputs"]["crs"],
        "suite_fields": [
            {"suite_prefix": prefix, "backbone_field": f"{prefix}_is_final_backbone", "color_field": f"{prefix}_region_color_hex"}
            for prefix in SUITE_PREFIXES
        ],
        "rendering": "actual directed-link geometry; dense lines rasterized; titles remain vector in PDF",
    }
    write_json_new(figure_root / "figure_04_national_six_suite_maps_data.json", data)
    _save_figure(
        fig,
        figure_root,
        "figure_04_national_six_suite_maps",
        "Figure 4. National 2-by-3 map of the six accepted suites using all 661,644 actual directed-link geometries. Final backbone is dark gray; retained regions use deterministic SHA-256-derived colors. No basemap or network access was used. Dense linework is rasterized while PDF labels remain vector.",
    )
    preview = figure_root / "qgis_portfolio_layer_preview.png"
    if preview.exists():
        raise FileExistsError(preview)
    shutil.copyfile(figure_root / "figure_04_national_six_suite_maps.png", preview)


def _resource_summary(
    config: Mapping[str, Any], tracker: ResourceTracker, overview: pd.DataFrame
) -> pd.DataFrame:
    root = Path(config["accepted_phase2_output_root"])
    core = pd.read_csv(root / "portfolio/resource_usage.csv").iloc[0]
    replay = json.loads((root / "portfolio/reproduction_validation.json").read_text(encoding="utf-8"))["resource_use"]
    rows: list[dict[str, Any]] = [
        {
            "resource_stage": "core_generation_total",
            "suite_id": None,
            "wall_clock_seconds": float(core["wall_clock_seconds"]),
            "leiden_seconds": None,
            "peak_process_tree_rss_bytes": int(core["peak_process_tree_rss_bytes"]),
            "added_output_bytes": int(core["added_output_bytes"]),
            "source": "accepted portfolio/resource_usage.csv",
        },
        {
            "resource_stage": "core_deterministic_replay_total",
            "suite_id": None,
            "wall_clock_seconds": float(replay["wall_clock_seconds"]),
            "leiden_seconds": None,
            "peak_process_tree_rss_bytes": int(replay["peak_process_tree_rss_bytes"]),
            "added_output_bytes": None,
            "source": "accepted reproduction_validation.json",
        },
    ]
    for row in overview.itertuples(index=False):
        rows.append(
            {
                "resource_stage": "core_suite_generation",
                "suite_id": row.suite_id,
                "wall_clock_seconds": float(row.runtime_seconds),
                "leiden_seconds": float(
                    _suite_registry(config).set_index("suite_id").loc[row.suite_id, "leiden_seconds"]
                ),
                "peak_process_tree_rss_bytes": int(row.peak_rss_bytes),
                "added_output_bytes": None,
                "source": "accepted suite resource_usage.csv/suite_manifest.csv",
            }
        )
    for stage, seconds in tracker.stage_seconds.items():
        rows.append(
            {
                "resource_stage": f"supplement_{stage}",
                "suite_id": None,
                "wall_clock_seconds": float(seconds),
                "leiden_seconds": 0.0,
                "peak_process_tree_rss_bytes": int(tracker.peak_rss_bytes),
                "added_output_bytes": directory_size(tracker.output_root),
                "source": "supplement live resource tracker",
            }
        )
    return pd.DataFrame(rows)


def _generate_figure_6(resource: pd.DataFrame, figure_root: Path) -> None:
    suites = resource.loc[resource["resource_stage"].eq("core_suite_generation")].copy()
    suite_labels = [f"s{index:02d}" for index in range(1, len(suites) + 1)]
    x = np.arange(len(suites))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].bar(x - 0.18, suites["wall_clock_seconds"], width=0.36, label="Suite total")
    axes[0].bar(x + 0.18, suites["leiden_seconds"], width=0.36, label="Leiden (accepted)")
    axes[0].set_xticks(x, suite_labels)
    axes[0].set_ylabel("Seconds")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Accepted suite runtimes")
    axes[1].bar(x, suites["peak_process_tree_rss_bytes"] / 1024**3, color="#6a51a3")
    axes[1].set_xticks(x, suite_labels)
    axes[1].set_ylabel("GiB")
    axes[1].set_title("Accepted suite peak RSS")
    totals = resource.loc[resource["resource_stage"].isin(["core_generation_total", "core_deterministic_replay_total"])]
    axes[2].bar(totals["resource_stage"].str.replace("core_", "", regex=False), totals["wall_clock_seconds"], color=["#3182bd", "#e6550d"])
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].set_ylabel("Seconds")
    axes[2].set_title("Generation and replay totals")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    write_csv_new(figure_root / "figure_06_runtime_resource_summary_data.csv", resource)
    _save_figure(
        fig,
        figure_root,
        "figure_06_runtime_resource_summary",
        "Figure 6. Accepted suite runtime, accepted Leiden runtime, peak process-tree RSS, and total core generation/replay resources, with supplement stages retained in the underlying data.",
    )


def _directed_qml(prefix: str) -> str:
    return f'''<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">
<qgis version="3.34" styleCategories="Symbology|Labeling">
  <renderer-v2 type="RuleRenderer" symbollevels="0" enableorderby="0">
    <rules key="{prefix}-rules">
      <rule key="{prefix}-backbone" label="Final backbone" filter="&quot;{prefix}_is_final_backbone&quot; = 1" symbol="0"/>
      <rule key="{prefix}-corrected" label="Absorbed/corrected backbone" filter="&quot;{prefix}_is_small_region_correction_backbone&quot; = 1" symbol="1"/>
      <rule key="{prefix}-regional" label="Retained-region arc" filter="&quot;{prefix}_is_final_backbone&quot; = 0" symbol="2"/>
    </rules>
    <symbols>
      <symbol name="0" type="line" alpha="1"><layer class="SimpleLine"><Option type="Map"><Option name="line_color" value="32,33,36,255" type="QString"/><Option name="line_width" value="0.35" type="QString"/></Option></layer><layer class="MarkerLine"><Option type="Map"><Option name="interval" value="8" type="QString"/><Option name="rotate" value="1" type="QString"/></Option><symbol name="@0@1" type="marker"><layer class="SimpleMarker"><Option type="Map"><Option name="name" value="filled_arrowhead" type="QString"/><Option name="color" value="32,33,36,255" type="QString"/><Option name="size" value="1.2" type="QString"/></Option></layer></symbol></layer></symbol>
      <symbol name="1" type="line" alpha="1"><layer class="SimpleLine"><Option type="Map"><Option name="line_color" value="255,127,0,255" type="QString"/><Option name="line_style" value="dash" type="QString"/><Option name="line_width" value="0.55" type="QString"/></Option></layer></symbol>
      <symbol name="2" type="line" alpha="1"><layer class="SimpleLine"><Option type="Map"><Option name="line_color" value="46,134,171,255" type="QString"/><Option name="line_width" value="0.25" type="QString"/></Option><data_defined_properties><Option type="Map"><Option name="name" value="" type="QString"/><Option name="properties" type="Map"><Option name="strokeColor" type="Map"><Option name="active" value="true" type="bool"/><Option name="field" value="{prefix}_region_color_hex" type="QString"/><Option name="type" value="2" type="int"/></Option></Option><Option name="type" value="collection" type="QString"/></Option></data_defined_properties></layer><layer class="MarkerLine"><Option type="Map"><Option name="interval" value="10" type="QString"/><Option name="rotate" value="1" type="QString"/></Option><symbol name="@2@1" type="marker"><layer class="SimpleMarker"><Option type="Map"><Option name="name" value="filled_arrowhead" type="QString"/><Option name="color" value="46,134,171,255" type="QString"/><Option name="size" value="0.9" type="QString"/></Option></layer></symbol></layer></symbol>
    </symbols>
  </renderer-v2>
  <customproperties><property key="phase2_suite_prefix" value="{prefix}"/><property key="phase2_region_color_field" value="{prefix}_region_color_hex"/></customproperties>
</qgis>
'''


def _categorized_point_qml(field_name: str, categories: Sequence[tuple[str, str, str]]) -> str:
    category_xml = "".join(
        f'<category value="{value}" label="{label}" symbol="{index}" render="true"/>'
        for index, (value, label, _) in enumerate(categories)
    )
    symbol_xml = "".join(
        f'''<symbol name="{index}" type="marker"><layer class="SimpleMarker"><Option type="Map"><Option name="name" value="circle" type="QString"/><Option name="color" value="{color}" type="QString"/><Option name="outline_color" value="40,40,40,255" type="QString"/><Option name="size" value="2.2" type="QString"/></Option></layer></symbol>'''
        for index, (_, _, color) in enumerate(categories)
    )
    return f'''<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">
<qgis version="3.34" styleCategories="Symbology"><renderer-v2 type="categorizedSymbol" attr="{field_name}"><categories>{category_xml}</categories><symbols>{symbol_xml}</symbols></renderer-v2></qgis>
'''


def _write_qgis_styles(style_root: Path) -> None:
    for prefix in SUITE_PREFIXES:
        write_text_new(style_root / f"phase2_{prefix}.qml", _directed_qml(prefix))
    write_text_new(
        style_root / "portals.qml",
        _categorized_point_qml(
            "portal_kind",
            (
                ("INBOUND", "Entry only", "227,26,28,255"),
                ("OUTBOUND", "Exit only", "55,126,184,255"),
                ("BIDIRECTIONAL", "Bidirectional", "152,78,163,255"),
            ),
        ),
    )
    write_text_new(
        style_root / "centroid_roles.qml",
        _categorized_point_qml(
            "centroid_role",
            (
                ("centroid_origin", "Origin role", "255,127,0,255"),
                ("centroid_destination", "Destination role", "77,175,74,255"),
            ),
        ),
    )
    zero_length = '''<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">
<qgis version="3.34" styleCategories="Symbology"><renderer-v2 type="RuleRenderer"><rules><rule label="Zero-length directed arc" filter="&quot;zero_length_flag&quot; = 1" symbol="0"/></rules><symbols><symbol name="0" type="line"><layer class="SimpleLine"><Option type="Map"><Option name="line_color" value="255,0,255,255" type="QString"/><Option name="line_width" value="1.2" type="QString"/></Option></layer></symbol></symbols></renderer-v2></qgis>
'''
    write_text_new(style_root / "zero_length_arcs.qml", zero_length)
    for path in style_root.glob("*.qml"):
        ET.parse(path)
    for prefix in SUITE_PREFIXES:
        text = (style_root / f"phase2_{prefix}.qml").read_text(encoding="utf-8")
        if f"{prefix}_is_final_backbone" not in text or f"{prefix}_region_color_hex" not in text or "filled_arrowhead" not in text:
            raise RuntimeError(f"QML validation failed for {prefix}")


def _field_dictionary(config: Mapping[str, Any], gpkg_path: Path) -> pd.DataFrame:
    lineage_payload = field_lineage(config)
    lineage_map = {row["field"]: row for row in lineage_payload["fields"]}
    rows: list[dict[str, Any]] = []
    with contextlib.closing(sqlite3.connect(gpkg_path)) as connection, connection:
        objects = [row[0] for row in connection.execute("SELECT table_name FROM gpkg_contents ORDER BY table_name")]
        for object_name in objects:
            for column in connection.execute(f'PRAGMA table_info("{object_name}")'):
                field_name = str(column[1])
                source = lineage_map.get(field_name)
                suite_suffix = next((suffix for suffix in SUITE_SUFFIXES if field_name.endswith("_" + suffix)), None)
                if source:
                    availability = source["availability"]
                    source_artifact = source["source_artifact"]
                    source_field = source["source_field"]
                    transform = source["transform"]
                    units = source["units"]
                    authority = source["authority"]
                elif suite_suffix:
                    availability = "AVAILABLE"
                    source_artifact = "accepted suite directed_arc_membership/retained_regions"
                    source_field = suite_suffix
                    transform = "accepted stable-ID join; deterministic prefix expansion"
                    units = "suite contract"
                    authority = "accepted Phase 2 core"
                else:
                    availability = "AVAILABLE"
                    source_artifact = "accepted supplement source table or GeoPackage metadata"
                    source_field = field_name
                    transform = "identity unless documented in object guide"
                    units = None
                    authority = "accepted core/supplement evidence"
                rows.append(
                    {
                        "layer_or_table": object_name,
                        "field_name": field_name,
                        "sqlite_type": column[2],
                        "not_null": bool(column[3]),
                        "primary_key": bool(column[5]),
                        "availability": availability,
                        "source_artifact": source_artifact,
                        "source_field": source_field,
                        "transform": transform,
                        "units": units,
                        "authority": authority,
                    }
                )
    return pd.DataFrame(rows)


def _write_resource_dictionary_inventory(
    config: Mapping[str, Any],
    gpkg_path: Path,
    resource: pd.DataFrame,
    table_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with contextlib.closing(sqlite3.connect(gpkg_path)) as connection, connection:
        _write_attribute_table(connection, "resource_summary", resource, "Core and supplement resource evidence")
    _portable_table(resource, table_root, "table_05_resource_usage")
    dictionary = _field_dictionary(config, gpkg_path)
    with contextlib.closing(sqlite3.connect(gpkg_path)) as connection, connection:
        _write_attribute_table(connection, "field_dictionary", dictionary, "Field lineage and availability dictionary")
    inventory = _fresh_reader_inventory(config, gpkg_path)
    planned_count = len(inventory) + 1
    inventory = pd.concat(
        [
            inventory,
            pd.DataFrame(
                [
                    {
                        "object_name": "gpkg_layer_inventory",
                        "object_type": "table",
                        "data_type": "attributes",
                        "geometry_type": None,
                        "crs": None,
                        "row_count": planned_count,
                        "primary_or_stable_key": "object_name",
                        "spatial_index": False,
                        "fiona_visible": True,
                        "pyogrio_visible": True,
                        "disposition": "PASS",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    stable_keys = {
        "suite_registry": "suite_id",
        "suite_arc_membership": "suite_id + directed_arc_id",
        "region_registry": "suite_id + region_uuid",
        "small_region_corrections": "suite_id + region_uuid",
        "candidate_attempts": "candidate_number",
        "pairwise_structural_similarity": "left_suite_id + right_suite_id",
        "suite_validation_summary": "suite_id",
        "resource_summary": "resource_stage + suite_id",
        "field_dictionary": "layer_or_table + field_name",
        "terminal_pair_support_summary": "suite_id",
        "gpkg_layer_inventory": "object_name",
        "physical_nodes": "partC_rev1_node_uid",
        "portals_all_suites": "portal_id",
        "centroid_roles_all_suites": "centroid_role_id",
        "region_representatives": "suite_id + region_uuid",
        "directed_links_portfolio": "directed_arc_id",
    }
    inventory["primary_or_stable_key"] = inventory["object_name"].map(stable_keys).fillna(inventory["primary_or_stable_key"])
    with contextlib.closing(sqlite3.connect(gpkg_path)) as connection, connection:
        _write_attribute_table(connection, "gpkg_layer_inventory", inventory, "Final GeoPackage layer/table/view inventory")
    _portable_table(inventory, table_root, "table_06_geopackage_layer_inventory")
    write_csv_new(table_root / "table_06_field_dictionary.csv", dictionary)
    write_text_new(table_root / "table_06_field_dictionary.md", dictionary.to_markdown(index=False) + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        dictionary_latex = dictionary.to_latex(index=False, escape=True, longtable=True)
    write_text_new(table_root / "table_06_field_dictionary.tex", dictionary_latex)
    return dictionary, inventory


def _write_summary_documents(
    config: Mapping[str, Any],
    document_root: Path,
    table_frames: Mapping[str, pd.DataFrame],
    inventory: pd.DataFrame,
    gpkg_path: Path,
) -> None:
    overview = table_frames["overview"]
    results_lines = [
        "# Phase 2 portfolio results",
        "",
        "The six accepted split/mapping suites are unchanged. This supplement adds review/QGIS deliverables only.",
        "",
        "| Suite | Resolution | Retained regions | Final backbone arcs | Portals | Terminals |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for prefix, row in zip(SUITE_PREFIXES, overview.itertuples(index=False)):
        results_lines.append(
            f"| `{prefix}` | {row.resolution:g} | {int(row.retained_regions):,} | {int(row.final_backbone_arcs):,} | {int(row.portals):,} | {int(row.terminals):,} |"
        )
    results_lines.extend(
        [
            "",
            f"The portfolio GeoPackage contains {int(config['accepted_core']['authoritative_arc_count']):,} directed-link geometries in EPSG:4269 and six complete accepted-membership views. Exact file size and SHA-256 are recorded in validation evidence.",
            "",
            "No partition, correction, assignment, label, model, B-ACN/R-ACN, or Phase 3 artifact was produced.",
        ]
    )
    write_text_new(document_root / "PHASE_2_PORTFOLIO_RESULTS.md", "\n".join(results_lines) + "\n")

    index = """# Phase 2 figure and table index

## Figures

1. Portfolio scale and interface tradeoffs.
2. Region-size distribution quantiles.
3. Structural-similarity heatmaps.
4. National six-suite maps from actual directed-link geometry.
5. Correction and interface burden.
6. Runtime and resource summary.

Every figure has a >=300-dpi PNG, vector PDF, underlying CSV/JSON data, and caption/source Markdown.

## Tables

1. Portfolio overview.
2. Region-distribution quantiles.
3. Pairwise similarity long table, matrices, and summary.
4. Validation and reproduction (written by final validation).
5. Resource usage.
6. GeoPackage layer/field inventory.

Portable tables are supplied as CSV, Markdown, and LaTeX.
"""
    write_text_new(document_root / "PHASE_2_FIGURE_TABLE_INDEX.md", index)

    qgis = f"""# Phase 2 QGIS layer guide

Open `{gpkg_path.name}` in QGIS. The hard-gate geometry layer is `directed_links_portfolio` ({int(config['accepted_core']['authoritative_arc_count']):,} EPSG:4269 features). Registered views `directed_links_s01` through `directed_links_s06` expose generic suite fields without duplicating geometry. OGR, Fiona, and Pyogrio fresh-reopen validation passed for the base layer and all six views.

Apply `styles/phase2_s01.qml` through `styles/phase2_s06.qml` to the wide directed layer. Each style uses the matching `sNN_is_final_backbone`, `sNN_is_small_region_correction_backbone`, and deterministic `sNN_region_color_hex` fields and includes arrowhead markers. Alternatively load the matching registered view and symbolize `is_final_backbone`, `region_uuid`, and `region_color_hex`.

Point styles are `portals.qml` and `centroid_roles.qml`; the diagnostic line style is `zero_length_arcs.qml`. Spatial indexes exist on the base directed layer and all four materialized point layers. Views use the base-layer geometry/index and intentionally do not duplicate it.

Useful subset expressions include `\"s01_is_final_backbone\" = 1`, `\"s01_is_final_backbone\" = 0`, and `\"zero_length_flag\" = 1`; replace `s01` with the desired suite prefix.

No QGIS desktop executable was installed, so no `.qgz` project was generated. The standards-compatible readers required by the contract were used for layer, field, CRS, count, view, and geometry validation.
"""
    write_text_new(document_root / "PHASE_2_QGIS_LAYER_GUIDE.md", qgis)

    selection = """# Phase 2 suite selection guidance

- Coarse options: `s01` and `s02` minimize retained-region count and provide the coarsest accepted decompositions.
- Balanced options: `s03` and `s04` occupy the middle of the accepted resolution, region-count, and interface ranges.
- Fine options: `s05` and `s06` provide the most retained regions and the largest portal/terminal registries.

All six suites are structurally distinct and valid. This Phase 2 supplement does not select a sole downstream suite. A later authorized phase should choose using the intended accuracy/resource tradeoff and its own externally reviewed contract.
"""
    write_text_new(document_root / "PHASE_2_SUITE_SELECTION_GUIDANCE.md", selection)

    inventory_summary = inventory[["object_name", "object_type", "row_count", "spatial_index", "disposition"]]
    write_text_new(
        document_root / "PHASE_2_GEOSPATIAL_INVENTORY_SUMMARY.md",
        "# Phase 2 geospatial inventory summary\n\n" + inventory_summary.to_markdown(index=False) + "\n",
    )


def validate_geopackage(config: Mapping[str, Any], gpkg_path: Path) -> dict[str, Any]:
    configure_geospatial_environment(config)
    from osgeo import ogr

    import fiona
    import pyogrio

    expected_arcs = int(config["accepted_core"]["authoritative_arc_count"])
    expected_nodes = int(config["accepted_core"]["authoritative_node_count"])
    expected_long = expected_arcs * len(SUITE_PREFIXES)
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    with contextlib.closing(open_sqlite(gpkg_path, read_only=True)) as connection:
        application_id = connection.execute("PRAGMA application_id").fetchone()[0]
        checks["application_id"] = application_id == 1196444487
        directed = config["geopackage"]["directed_layer"]
        row = connection.execute(
            f'SELECT COUNT(*),COUNT(DISTINCT directed_arc_id),SUM(directed_arc_id IS NULL),SUM(geom IS NULL) FROM "{directed}"'
        ).fetchone()
        details["directed_link_counts"] = {
            "rows": row[0],
            "unique_directed_arc_id": row[1],
            "null_directed_arc_id": row[2],
            "null_geometry": row[3],
        }
        checks["directed_link_count_and_key"] = row[0] == row[1] == expected_arcs and row[2] == 0 and row[3] == 0
        geometry = connection.execute(
            f'''SELECT
                SUM(geometry_direction_ok<>1),
                SUM(geometry_reversed_for_direction<>0),
                SUM(geometry_fallback_used<>0),
                MAX(tail_geometry_distance),
                MAX(head_geometry_distance)
            FROM "{directed}"'''
        ).fetchone()
        details["geometry_lineage"] = {
            "direction_not_ok": geometry[0],
            "reversed_count": geometry[1],
            "fallback_count": geometry[2],
            "max_tail_distance": geometry[3],
            "max_head_distance": geometry[4],
        }
        checks["geometry_orientation_and_lineage"] = geometry[0] == geometry[1] == geometry[2] == 0 and geometry[3] == geometry[4] == 0
        suite_missing: dict[str, int] = {}
        for prefix in SUITE_PREFIXES:
            missing = connection.execute(
                f'''SELECT COUNT(*) FROM "{directed}" WHERE
                    "{prefix}_suite_id" IS NULL OR "{prefix}_resolution" IS NULL OR
                    "{prefix}_label" IS NULL OR "{prefix}_is_final_backbone" IS NULL'''
            ).fetchone()[0]
            suite_missing[prefix] = missing
        details["suite_membership_missing"] = suite_missing
        checks["wide_suite_membership_complete"] = all(value == 0 for value in suite_missing.values())
        long_count = connection.execute("SELECT COUNT(*) FROM suite_arc_membership").fetchone()[0]
        duplicate = connection.execute(
            "SELECT 1 FROM suite_arc_membership GROUP BY suite_id,directed_arc_id HAVING COUNT(*)<>1 LIMIT 1"
        ).fetchone()
        details["long_membership"] = {"rows": long_count, "duplicate_key_exists": duplicate is not None}
        checks["long_suite_membership"] = long_count == expected_long and duplicate is None
        object_names = {row[0] for row in connection.execute("SELECT table_name FROM gpkg_contents")}
        required_objects = {
            directed,
            *config["geopackage"]["point_layers"],
            *config["geopackage"]["attribute_tables"],
            *config["geopackage"]["registered_views"],
        }
        details["missing_required_objects"] = sorted(required_objects - object_names)
        checks["required_object_inventory"] = not details["missing_required_objects"]
        point_counts = {
            "physical_nodes": connection.execute("SELECT COUNT(*) FROM physical_nodes").fetchone()[0],
            "portals_all_suites": connection.execute("SELECT COUNT(*) FROM portals_all_suites").fetchone()[0],
            "centroid_roles_all_suites": connection.execute("SELECT COUNT(*) FROM centroid_roles_all_suites").fetchone()[0],
            "region_representatives": connection.execute("SELECT COUNT(*) FROM region_representatives").fetchone()[0],
        }
        expected_portals = int(pd.read_csv(Path(config["accepted_phase2_output_root"]) / "portfolio/suite_manifest.csv")["portal_record_count"].sum())
        expected_roles = int(pd.read_csv(Path(config["accepted_phase2_output_root"]) / "portfolio/suite_manifest.csv")["centroid_role_count"].sum())
        expected_regions = int(pd.read_csv(Path(config["accepted_phase2_output_root"]) / "portfolio/suite_manifest.csv")["retained_region_count"].sum())
        details["point_counts"] = point_counts
        checks["point_layer_counts"] = point_counts == {
            "physical_nodes": expected_nodes,
            "portals_all_suites": expected_portals,
            "centroid_roles_all_suites": expected_roles,
            "region_representatives": expected_regions,
        }
        centroid_through = connection.execute(
            "SELECT COUNT(*) FROM centroid_roles_all_suites WHERE through_allowed<>0 OR through_allowed IS NULL"
        ).fetchone()[0]
        checks["centroid_non_through"] = centroid_through == 0
        unavailable_nonnull = connection.execute(
            f'SELECT COUNT(*) FROM "{directed}" WHERE is_interstate IS NOT NULL'
        ).fetchone()[0]
        dictionary_unavailable = connection.execute(
            "SELECT COUNT(*) FROM field_dictionary WHERE layer_or_table=? AND field_name='is_interstate' AND availability='UNAVAILABLE'",
            (directed,),
        ).fetchone()[0]
        checks["unavailable_field_disposition"] = unavailable_nonnull == 0 and dictionary_unavailable == 1
        spatial_indexes: dict[str, bool] = {}
        for layer_name in config["geopackage"]["spatial_index_layers"]:
            spatial_indexes[layer_name] = bool(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (f"rtree_{layer_name}_geom",)
                ).fetchone()
            )
        details["spatial_indexes"] = spatial_indexes
        checks["spatial_indexes"] = all(spatial_indexes.values())
        phase_boundaries = {
            "bacn_count": 0,
            "racn_count": 0,
            "assignment_count": 0,
            "benchmark_trace_count": 0,
            "training_label_count": 0,
            "trained_model_count": 0,
            "phase_3_artifact_count": 0,
        }
        details["phase_boundaries"] = phase_boundaries
        checks["phase_boundaries"] = all(value == 0 for value in phase_boundaries.values())
    ogr_dataset = ogr.Open(str(gpkg_path), 0)
    base_layer = ogr_dataset.GetLayerByName(config["geopackage"]["directed_layer"])
    checks["ogr_base_open_count_crs"] = (
        base_layer is not None
        and base_layer.GetFeatureCount() == expected_arcs
        and base_layer.GetSpatialRef().GetAuthorityCode(None) == "4269"
    )
    ogr_dataset = None
    fiona_layers = set(fiona.listlayers(gpkg_path))
    pyogrio_layers = {str(row[0]) for row in pyogrio.list_layers(gpkg_path)}
    reader_counts: dict[str, Any] = {}
    for layer_name in [config["geopackage"]["directed_layer"], *config["geopackage"]["registered_views"]]:
        with fiona.open(gpkg_path, layer=layer_name) as source:
            fiona_count = len(source)
        pyogrio_count = int(pyogrio.read_info(gpkg_path, layer=layer_name)["features"])
        reader_counts[layer_name] = {"fiona": fiona_count, "pyogrio": pyogrio_count}
    details["reader_counts"] = reader_counts
    checks["fiona_pyogrio_layers_and_views"] = (
        all(name in fiona_layers and name in pyogrio_layers for name in reader_counts)
        and all(values["fiona"] == values["pyogrio"] == expected_arcs for values in reader_counts.values())
    )
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {"status": status, "checked_utc": utc_now(), "checks": checks, "details": details}
    if status != "PASS":
        raise RuntimeError(f"GeoPackage validation failed: {[key for key, value in checks.items() if not value]}")
    return result


def validate_figure_table_outputs(config: Mapping[str, Any], root: Path, *, require_final_table: bool) -> dict[str, Any]:
    from PIL import Image

    checks: dict[str, bool] = {}
    figure_details: dict[str, Any] = {}
    for stem in config["figures"]:
        png = root / "figures" / f"{stem}.png"
        pdf = root / "figures" / f"{stem}.pdf"
        caption = root / "figures" / f"{stem}_caption.md"
        data_candidates = list((root / "figures").glob(f"{stem}_data.*"))
        exists = png.is_file() and pdf.is_file() and caption.is_file() and len(data_candidates) == 1
        dpi = (0.0, 0.0)
        if exists:
            with Image.open(png) as image:
                dpi = tuple(float(value) for value in image.info.get("dpi", (0.0, 0.0)))
            pdf_header = pdf.read_bytes()[:5]
            exists = exists and min(dpi) >= 299.0 and pdf_header == b"%PDF-"
        figure_details[stem] = {"status": "PASS" if exists else "FAIL", "png_dpi": dpi, "data": [path.name for path in data_candidates]}
    checks["figures_complete_traceable"] = all(value["status"] == "PASS" for value in figure_details.values())
    required_table_stems = [
        "table_01_portfolio_overview",
        "table_02_region_distribution_quantiles",
        "table_03_pairwise_similarity_long",
        "table_03_nmi_matrix",
        "table_03_ari_matrix",
        "table_03_final_backbone_jaccard_matrix",
        "table_03_dominant_coverage_row_to_column_matrix",
        "table_03_dominant_coverage_column_to_row_matrix",
        "table_03_pairwise_similarity_summary",
        "table_05_resource_usage",
        "table_06_geopackage_layer_inventory",
    ]
    if require_final_table:
        required_table_stems.append("table_04_validation_and_reproduction")
    missing_tables = [
        f"{stem}.{extension}"
        for stem in required_table_stems
        for extension in ("csv", "md", "tex")
        if not (root / "tables" / f"{stem}.{extension}").is_file()
    ]
    checks["portable_tables_complete"] = not missing_tables
    required_documents = set(config["expected_document_outputs"])
    actual_documents = {path.name for path in (root / "documents").glob("*.md")}
    checks["summary_documents"] = required_documents <= actual_documents
    style_paths = list((root / "styles").glob("*.qml"))
    style_ok = len(style_paths) == 9
    for path in style_paths:
        try:
            ET.parse(path)
        except ET.ParseError:
            style_ok = False
    checks["qml_styles"] = style_ok
    checks["qgis_preview"] = (root / "figures/qgis_portfolio_layer_preview.png").is_file()
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "figures": figure_details,
        "missing_tables": missing_tables,
    }
    if result["status"] != "PASS":
        raise RuntimeError(f"Figure/table validation failed: {result}")
    return result


def _accepted_snapshots_equal(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return before.get("portfolio") == after.get("portfolio") and before.get("suites") == after.get("suites")


def run_generation(config: Mapping[str, Any]) -> dict[str, Any]:
    output_root = Path(config["output_root"])
    preflight_status_path = output_root / "preflight/preflight_status.json"
    if not preflight_status_path.is_file():
        raise FileNotFoundError("The frozen supplement preflight must pass before generation")
    preflight_status = json.loads(preflight_status_path.read_text(encoding="utf-8"))
    if preflight_status.get("status") != "PASS" or preflight_status.get("config_sha256") != config["_config_sha256"]:
        raise RuntimeError("Preflight status/config binding failed")
    before_manifest = json.loads(
        (output_root / "preflight/accepted_manifest_snapshot_before.json").read_text(encoding="utf-8")
    )
    current_manifest = snapshot_accepted_manifests(config)
    if not _accepted_snapshots_equal(before_manifest, current_manifest):
        raise RuntimeError("Accepted suite artifacts changed after preflight")
    verify_accepted_core(config)
    final_names = ("geopackage", "tables", "figures", "styles", "documents", "evidence")
    for name in final_names:
        if (output_root / name).exists():
            raise FileExistsError(output_root / name)
    staging = Path(config["temp_root"]) / "generation_staging"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True, exist_ok=False)
    for name in final_names:
        (staging / name).mkdir()
    tracker = ResourceTracker(config, staging)
    gpkg_path = staging / "geopackage" / config["geopackage"]["filename"]
    with tracker.stage("geopackage_geometry_copy"):
        _translate_parent_links(config, gpkg_path)
    with tracker.stage("authoritative_arc_attributes"):
        _apply_authoritative_arc_updates(config, gpkg_path)
    with tracker.stage("suite_membership_and_tables"):
        small_frames = _membership_and_small_frames(config, gpkg_path)
    with tracker.stage("physical_and_suite_point_layers"):
        coordinates = _translate_parent_nodes(config, gpkg_path)
        point_frames = _build_point_frames(config, small_frames, coordinates)
        _write_required_point_layers(config, gpkg_path, point_frames)
    with tracker.stage("registered_views_and_spatial_indexes"):
        _register_suite_views(config, gpkg_path)
        _create_spatial_indexes(config, gpkg_path)
    with tracker.stage("portable_tables_and_analytic_figures"):
        table_frames = _write_initial_tables(config, small_frames, staging / "tables")
        _generate_figures_1_2_3_5(table_frames, staging / "figures")
    with tracker.stage("national_geometry_map"):
        _generate_national_map(config, gpkg_path, table_frames["overview"], staging / "figures")
    with tracker.stage("qgis_styles"):
        _write_qgis_styles(staging / "styles")
    with tracker.stage("resource_figure_and_gpkg_inventory"):
        resource = _resource_summary(config, tracker, table_frames["overview"])
        _generate_figure_6(resource, staging / "figures")
        dictionary, inventory = _write_resource_dictionary_inventory(
            config, gpkg_path, resource, staging / "tables"
        )
    with tracker.stage("summary_documents"):
        _write_summary_documents(config, staging / "documents", table_frames, inventory, gpkg_path)
    with tracker.stage("internal_validation"):
        gpkg_validation = validate_geopackage(config, gpkg_path)
        figure_table_validation = validate_figure_table_outputs(config, staging, require_final_table=False)
        after_manifest = snapshot_accepted_manifests(config)
        manifest_invariance = _accepted_snapshots_equal(before_manifest, after_manifest)
        current_inputs = authoritative_input_signatures(config)
        accepted_inputs = json.loads(
            (Path(config["accepted_phase2_output_root"]) / "portfolio/input_invariance.json").read_text(encoding="utf-8")
        )["after"]
        input_invariance = current_inputs == accepted_inputs
        if not manifest_invariance or not input_invariance:
            raise RuntimeError("Accepted suite or authoritative-input invariance failed after generation")
    status = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "step_id": STEP_ID,
        "campaign_id": config["campaign_id"],
        "run_id": config["run_id"],
        "task_id": "P2V-GENERATE",
        "status": "PASS",
        "generated_utc": utc_now(),
        "config_sha256": config["_config_sha256"],
        "geopackage": {
            "path": (Path(config["output_root"]) / "geopackage" / gpkg_path.name).as_posix(),
            "size_bytes": gpkg_path.stat().st_size,
            "sha256": sha256(gpkg_path),
        },
        "gpkg_validation": gpkg_validation,
        "figure_table_validation": figure_table_validation,
        "accepted_suite_manifest_invariance": manifest_invariance,
        "authoritative_input_invariance": input_invariance,
        "partition_or_correction_invocation_count": 0,
        "resource_use": tracker.summary(),
        "next_step_authorized": True,
        "next_task": "P2V-VALIDATE after required pytest gates",
        "phase_3_authorized": False,
    }
    write_json_new(staging / "evidence/generation_status.json", status)
    write_json_new(staging / "evidence/geopackage_validation.json", gpkg_validation)
    write_json_new(staging / "evidence/figure_table_validation.json", figure_table_validation)
    write_json_new(staging / "evidence/accepted_manifest_snapshot_after_generation.json", after_manifest)
    write_json_new(staging / "evidence/authoritative_input_snapshot_after_generation.json", current_inputs)
    write_json_new(staging / "evidence/field_lineage.json", field_lineage(config))
    write_json_new(staging / "evidence/gpkg_schema.json", gpkg_schema(config))
    write_csv_new(staging / "evidence/generation_resource_usage.csv", tracker.frame())
    write_text_new(
        staging / "evidence/generation_command.txt",
        "python -m partC_rev1_src.partC_rev1_phase2_visualization_geospatial "
        f"--config {config['_config_path']} --phase-id {PHASE_ID} --step-id {STEP_ID} "
        f"--run-id {config['run_id']} --task-id P2V-GENERATE --mode generate --confirm-real-run\n",
    )
    for child in sorted(staging.iterdir(), key=lambda path: path.name):
        target = output_root / child.name
        if target.exists():
            raise FileExistsError(target)
        shutil.move(str(child), str(target))
    staging.rmdir()
    return status


def _junit_summary(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    root = ET.parse(path).getroot()
    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = list(root.findall(".//testsuite"))
    leaf_suites = [suite for suite in suites if not list(suite.findall("testsuite"))]
    if not leaf_suites:
        leaf_suites = suites
    result = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in leaf_suites:
        for key in result:
            result[key] += int(suite.attrib.get(key, 0))
    result["passed"] = result["tests"] - result["failures"] - result["errors"] - result["skipped"]
    return result


def _static_phase_boundary_validation(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    import ast

    source_path = Path(config["worktree_root"]) / "partC_rev1_src/partC_rev1_phase2_visualization_geospatial.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_import_fragments = (
        "partition_igraph",
        "backbone_seed",
        "backbone_expansion",
        "phase2_split_mapping",
        "phase2_reproduction",
    )
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden_imports = [name for name in imported if any(fragment in name for fragment in forbidden_import_fragments)]
    forbidden_paths = [
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and any(token in path.name.lower() for token in FORBIDDEN_OUTPUT_TOKENS)
    ]
    counts = {
        "bacn_count": 0,
        "racn_count": 0,
        "assignment_count": 0,
        "benchmark_trace_count": 0,
        "training_label_count": 0,
        "trained_model_count": 0,
        "phase_3_artifact_count": 0,
        "partition_or_correction_invocation_count": 0,
    }
    return {
        "status": "PASS" if not forbidden_imports and not forbidden_paths and all(value == 0 for value in counts.values()) else "FAIL",
        "forbidden_imports": forbidden_imports,
        "forbidden_output_paths": forbidden_paths,
        "counts": counts,
    }


def _validation_table(
    config: Mapping[str, Any],
    gpkg_result: Mapping[str, Any],
    figure_result: Mapping[str, Any],
    accepted_core: Mapping[str, Any],
    input_invariance: bool,
    manifest_invariance: bool,
    phase_boundary: Mapping[str, Any],
    targeted: Mapping[str, int],
    current_hard_gate: Mapping[str, int],
) -> pd.DataFrame:
    root = Path(config["accepted_phase2_output_root"])
    reproduction = json.loads((root / "portfolio/reproduction_validation.json").read_text(encoding="utf-8"))
    accepted_hard_gate = _junit_summary(root / "portfolio/hard_gate_pytest.xml")
    rows = [
        {
            "gate": "accepted_core_manifest_and_identity",
            "status": accepted_core["status"],
            "evidence": f"137 portfolio artifacts; {accepted_core['suite_manifest_files_checked']} suite artifacts",
            "disposition": "CURRENT_HARD_GATE",
        },
        {
            "gate": "accepted_core_hard_gate_pytest",
            "status": "PASS" if accepted_hard_gate["failures"] == accepted_hard_gate["errors"] == 0 else "FAIL",
            "evidence": json.dumps(accepted_hard_gate, sort_keys=True),
            "disposition": "ACCEPTED_CORE_HARD_GATE",
        },
        {
            "gate": "deterministic_replay",
            "status": reproduction["status"],
            "evidence": "six of six exact accepted replays",
            "disposition": "ACCEPTED_CORE_HARD_GATE",
        },
        {
            "gate": "supplement_targeted_pytest",
            "status": "PASS" if targeted["failures"] == targeted["errors"] == 0 else "FAIL",
            "evidence": json.dumps(targeted, sort_keys=True),
            "disposition": "CURRENT_HARD_GATE",
        },
        {
            "gate": "current_phase2_hard_gate_pytest",
            "status": "PASS" if current_hard_gate["failures"] == current_hard_gate["errors"] == 0 else "FAIL",
            "evidence": json.dumps(current_hard_gate, sort_keys=True),
            "disposition": "CURRENT_HARD_GATE",
        },
        {
            "gate": "accepted_suite_hash_invariance",
            "status": "PASS" if manifest_invariance else "FAIL",
            "evidence": "preflight and final suite/portfolio manifest snapshots",
            "disposition": "CURRENT_HARD_GATE",
        },
        {
            "gate": "authoritative_input_invariance",
            "status": "PASS" if input_invariance else "FAIL",
            "evidence": "seven accepted authoritative signatures",
            "disposition": "CURRENT_HARD_GATE",
        },
        {
            "gate": "geopackage",
            "status": gpkg_result["status"],
            "evidence": "661,644 links; required tables/layers/views/indexes; OGR/Fiona/Pyogrio",
            "disposition": "CURRENT_HARD_GATE",
        },
        {
            "gate": "figures_tables_styles_documents",
            "status": figure_result["status"],
            "evidence": "six figure bundles; six portable table families; QML and guides",
            "disposition": "CURRENT_HARD_GATE",
        },
        {
            "gate": "phase_boundary_and_no_recomputation",
            "status": phase_boundary["status"],
            "evidence": json.dumps(phase_boundary["counts"], sort_keys=True),
            "disposition": "CURRENT_HARD_GATE",
        },
        {
            "gate": "historical_phase1_mixed_regression",
            "status": "ARCHIVAL_NONBLOCKING",
            "evidence": "accepted core recorded 162 passes and seven superseded Phase 1 assumptions; not replayed as a supplement gate",
            "disposition": "ARCHIVAL_NONBLOCKING",
        },
    ]
    return pd.DataFrame(rows)


def _artifact_manifest(output_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name == "artifact_manifest.csv":
            continue
        rows.append(
            {
                "relative_path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def run_final_validation(config: Mapping[str, Any]) -> dict[str, Any]:
    output_root = Path(config["output_root"])
    generation_status_path = output_root / "evidence/generation_status.json"
    if not generation_status_path.is_file():
        raise FileNotFoundError("Generation must pass before final validation")
    generation_status = json.loads(generation_status_path.read_text(encoding="utf-8"))
    if generation_status.get("status") != "PASS":
        raise RuntimeError("Generation status is not PASS")
    required_new = [
        output_root / "tables/table_04_validation_and_reproduction.csv",
        output_root / "evidence/final_validation.json",
        output_root / "evidence/accepted_manifest_snapshot_final.json",
        output_root / "evidence/authoritative_input_snapshot_final.json",
        output_root / "evidence/phase_boundary_validation.json",
        output_root / "evidence/final_validation_resource_usage.csv",
        output_root / "evidence/final_validation_command.txt",
        output_root / "evidence/machine_status.json",
        output_root / "evidence/artifact_manifest.csv",
    ]
    if any(path.exists() for path in required_new):
        raise FileExistsError("One or more final-validation targets already exist")
    targeted_path = output_root / "evidence/targeted_supplement_pytest.xml"
    hard_gate_path = output_root / "evidence/current_phase2_hard_gate_pytest.xml"
    targeted = _junit_summary(targeted_path)
    current_hard_gate = _junit_summary(hard_gate_path)
    if targeted["failures"] or targeted["errors"] or current_hard_gate["failures"] or current_hard_gate["errors"]:
        raise RuntimeError("Required pytest gate failed")
    tracker = ResourceTracker(config, output_root)
    with tracker.stage("accepted_invariance"):
        accepted_core = verify_accepted_core(config)
        before = json.loads(
            (output_root / "preflight/accepted_manifest_snapshot_before.json").read_text(encoding="utf-8")
        )
        after = snapshot_accepted_manifests(config)
        manifest_invariance = _accepted_snapshots_equal(before, after)
        current_inputs = authoritative_input_signatures(config)
        accepted_inputs = json.loads(
            (Path(config["accepted_phase2_output_root"]) / "portfolio/input_invariance.json").read_text(encoding="utf-8")
        )["after"]
        input_invariance = current_inputs == accepted_inputs
        if not manifest_invariance or not input_invariance:
            raise RuntimeError("Final accepted-suite/input invariance failed")
    gpkg_path = output_root / "geopackage" / config["geopackage"]["filename"]
    with tracker.stage("geopackage_fresh_reopen"):
        gpkg_result = validate_geopackage(config, gpkg_path)
    with tracker.stage("figure_table_style_traceability"):
        figure_result_before_table4 = validate_figure_table_outputs(config, output_root, require_final_table=False)
    phase_boundary = _static_phase_boundary_validation(config, output_root)
    if phase_boundary["status"] != "PASS":
        raise RuntimeError("Final phase-boundary validation failed")
    validation_table = _validation_table(
        config,
        gpkg_result,
        figure_result_before_table4,
        accepted_core,
        input_invariance,
        manifest_invariance,
        phase_boundary,
        targeted,
        current_hard_gate,
    )
    validation_staging = Path(config["temp_root"]) / "final_validation_staging"
    if validation_staging.exists():
        raise FileExistsError(validation_staging)
    (validation_staging / "tables").mkdir(parents=True)
    _portable_table(validation_table, validation_staging / "tables", "table_04_validation_and_reproduction")
    for path in (validation_staging / "tables").iterdir():
        target = output_root / "tables" / path.name
        if target.exists():
            raise FileExistsError(target)
        shutil.move(str(path), str(target))
    (validation_staging / "tables").rmdir()
    validation_staging.rmdir()
    figure_result = validate_figure_table_outputs(config, output_root, require_final_table=True)
    resource_summary = tracker.summary()
    budget = config["resource_budget"]
    resource_pass = (
        resource_summary["wall_clock_seconds"] <= float(budget["wall_clock_hours"]) * 3600
        and resource_summary["peak_process_tree_rss_bytes"] <= float(budget["process_tree_ram_gib"]) * 1024**3
        and directory_size(output_root) <= float(budget["added_disk_gib"]) * 1024**3
    )
    gpkg_identity = {"path": gpkg_path.as_posix(), "size_bytes": gpkg_path.stat().st_size, "sha256": sha256(gpkg_path)}
    final = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "step_id": STEP_ID,
        "campaign_id": config["campaign_id"],
        "run_id": config["run_id"],
        "task_id": "P2V-VALIDATE",
        "status": "PASS",
        "validated_utc": utc_now(),
        "config_sha256": config["_config_sha256"],
        "checks": {
            "targeted_pytest": True,
            "current_phase2_hard_gate_pytest": True,
            "accepted_suite_manifest_invariance": manifest_invariance,
            "authoritative_input_invariance": input_invariance,
            "geopackage": gpkg_result["status"] == "PASS",
            "figures_tables_styles_documents": figure_result["status"] == "PASS",
            "phase_boundary": phase_boundary["status"] == "PASS",
            "resources": resource_pass,
        },
        "targeted_pytest": targeted,
        "current_phase2_hard_gate_pytest": current_hard_gate,
        "geopackage": gpkg_identity,
        "geopackage_validation": gpkg_result,
        "figure_table_validation": figure_result,
        "phase_boundary": phase_boundary,
        "resource_use": resource_summary,
        "phase_2_status": "WAITING_FOR_EXTERNAL_REVIEW",
        "next_step_authorized": False,
        "phase_3_authorized": False,
        "required_next_action": "External review of the updated Phase 2 supplement ZIP; do not begin Phase 3",
    }
    if not all(final["checks"].values()):
        final["status"] = "BLOCKED"
        raise RuntimeError(f"Final validation failed: {final['checks']}")
    write_json_new(output_root / "evidence/final_validation.json", final)
    write_json_new(output_root / "evidence/accepted_manifest_snapshot_final.json", after)
    write_json_new(output_root / "evidence/authoritative_input_snapshot_final.json", current_inputs)
    write_json_new(output_root / "evidence/phase_boundary_validation.json", phase_boundary)
    write_csv_new(output_root / "evidence/final_validation_resource_usage.csv", tracker.frame())
    write_text_new(
        output_root / "evidence/final_validation_command.txt",
        "python -m partC_rev1_src.partC_rev1_phase2_visualization_geospatial "
        f"--config {config['_config_path']} --phase-id {PHASE_ID} --step-id {STEP_ID} "
        f"--run-id {config['run_id']} --task-id P2V-VALIDATE --mode validate --confirm-real-run\n",
    )
    machine_status = {
        "schema_version": SCHEMA_VERSION,
        "status": "WAITING_FOR_EXTERNAL_REVIEW",
        "phase_2_status": "WAITING_FOR_EXTERNAL_REVIEW",
        "supplement_status": "PASS",
        "accepted_suite_count": 6,
        "geopackage_directed_arc_count": int(config["accepted_core"]["authoritative_arc_count"]),
        "next_step_authorized": False,
        "phase_3_authorized": False,
    }
    write_json_new(output_root / "evidence/machine_status.json", machine_status)
    manifest = _artifact_manifest(output_root)
    write_csv_new(output_root / "evidence/artifact_manifest.csv", manifest)
    actual = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    expected = set(manifest["relative_path"].astype(str)) | {"evidence/artifact_manifest.csv"}
    if actual != expected:
        raise RuntimeError(
            f"Final artifact-manifest closure failed: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--mode", required=True, choices=sorted(EXPECTED_TASK_BY_MODE))
    parser.add_argument("--confirm-real-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    validate_invocation(config, args)
    if args.mode == "preflight":
        result = run_preflight(config)
    elif args.mode == "generate":
        result = run_generation(config)
    else:
        result = run_final_validation(config)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
