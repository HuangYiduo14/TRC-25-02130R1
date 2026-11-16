"""Bounded read-only Phase 1 Step 1.1 authoritative-input validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
import zipfile
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import jsonschema
import numpy as np
import psutil
import pyarrow.parquet as pq

from .partC_rev1_hashing import partC_rev1_file_sha256, partC_rev1_file_signature
from .partC_rev1_phase1_input_contract import (
    PartCRev1AuthoritativeInputManifest,
    PartCRev1InputIdentityError,
    PartCRev1InputPathError,
    PartCRev1InputSchemaError,
    PartCRev1InputSignatureError,
    PartCRev1InputValidationError,
    PartCRev1Phase1DataRoots,
    PartCRev1ResourceLimitError,
    PartCRev1ValidationReport,
    partC_rev1_assert_phase1_destination,
    partC_rev1_load_authoritative_input_manifest,
    partC_rev1_load_phase1_data_roots,
    partC_rev1_open_sqlite_strict_readonly,
)


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ALLOWED_ZIP_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
_FULL_MEMBER_ALLOWLIST = frozenset({"origin_group_node_idx.npy", "origin_offsets.npy"})
_FULL_MEMBER_LIMIT = 1024 * 1024
_EXPECTED_TASKS = {
    "validate_registry_and_manifest",
    "verify_reviewed_signatures",
    "inspect_npz_bounded",
    "sample_parquet_stable_ids",
    "verify_sqlite_gpkg_readonly",
    "verify_corrected_partition_registry",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_exact(handle: Any, size: int) -> bytes:
    if size < 0:
        raise PartCRev1InputSchemaError("Unbounded or negative member reads are prohibited")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    value = b"".join(chunks)
    if len(value) != size:
        raise PartCRev1InputSchemaError(
            "NPZ member ended before the configured bounded read completed",
            details={"requested": size, "received": len(value)},
        )
    return value


@dataclass(frozen=True, slots=True)
class PartCRev1NPYHeader:
    member: str
    version: tuple[int, int]
    shape: tuple[int, ...]
    fortran_order: bool
    dtype: str
    dtype_kind: str
    itemsize: int
    header_bytes: int
    data_bytes: int
    compressed_bytes: int
    uncompressed_bytes: int


class PartCRev1BoundedNPZReader:
    """Inspect NPZ central-directory entries, NPY headers, and fixed samples."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_members: Iterable[str] | None = None,
        max_prefix_count: int = 128,
        full_member_allowlist: Iterable[str] = _FULL_MEMBER_ALLOWLIST,
        full_member_limit: int = _FULL_MEMBER_LIMIT,
    ) -> None:
        self.path = Path(path)
        self.max_prefix_count = int(max_prefix_count)
        self.full_member_allowlist = frozenset(str(item) for item in full_member_allowlist)
        self.full_member_limit = int(full_member_limit)
        if self.max_prefix_count <= 0 or self.full_member_limit <= 0:
            raise PartCRev1InputSchemaError("NPZ bounds must be positive", path=self.path)
        self._archive = zipfile.ZipFile(self.path, mode="r")
        infos = self._archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            self._archive.close()
            raise PartCRev1InputSchemaError("NPZ contains duplicate member names", path=self.path)
        self._infos = {item.filename: item for item in infos}
        for info in infos:
            pure = PurePosixPath(info.filename)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or "\\" in info.filename
                or info.is_dir()
                or not info.filename.endswith(".npy")
            ):
                self._archive.close()
                raise PartCRev1InputSchemaError("NPZ contains an unsafe member path", path=info.filename)
            if info.flag_bits & 0x1:
                self._archive.close()
                raise PartCRev1InputSchemaError("Encrypted NPZ members are prohibited", path=info.filename)
            if info.compress_type not in _ALLOWED_ZIP_COMPRESSION:
                self._archive.close()
                raise PartCRev1InputSchemaError(
                    "Unsupported NPZ compression method", path=info.filename, details={"compress_type": info.compress_type}
                )
        if expected_members is not None:
            expected = set(expected_members)
            actual = set(names)
            if actual != expected:
                self._archive.close()
                raise PartCRev1InputSchemaError(
                    "NPZ member set mismatch",
                    path=self.path,
                    details={"missing": sorted(expected - actual), "unexpected": sorted(actual - expected)},
                )
        self.bytes_requested = 0
        self._headers: dict[str, PartCRev1NPYHeader] = {}

    def __enter__(self) -> "PartCRev1BoundedNPZReader":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        self._archive.close()

    @property
    def member_names(self) -> tuple[str, ...]:
        return tuple(self._infos)

    def header(self, member: str) -> PartCRev1NPYHeader:
        if member in self._headers:
            return self._headers[member]
        if member not in self._infos:
            raise PartCRev1InputSchemaError("Requested NPZ member is missing", path=member)
        info = self._infos[member]
        with self._archive.open(info, mode="r") as handle:
            try:
                version = np.lib.format.read_magic(handle)
                if version not in {(1, 0), (2, 0), (3, 0)}:
                    raise PartCRev1InputSchemaError(
                        "Unsupported NPY header version", path=member, details={"version": version}
                    )
                shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)
            except (EOFError, ValueError, OSError) as exc:
                raise PartCRev1InputSchemaError(f"Invalid NPY header for {member}: {exc}", path=member) from exc
            header_bytes = int(handle.tell())
        dtype = np.dtype(dtype)
        if dtype.hasobject:
            raise PartCRev1InputSchemaError("Object dtype is prohibited in authoritative NPZ members", path=member)
        if any(not isinstance(item, int) or item < 0 for item in shape):
            raise PartCRev1InputSchemaError("Invalid NPY shape", path=member, details={"shape": shape})
        elements = math.prod(shape)
        data_bytes = elements * dtype.itemsize
        if data_bytes < 0 or data_bytes > sys.maxsize:
            raise PartCRev1InputSchemaError("NPY byte-size calculation overflow", path=member)
        if info.file_size - header_bytes != data_bytes:
            raise PartCRev1InputSchemaError(
                "NPY header/member-size inconsistency",
                path=member,
                details={"header_bytes": header_bytes, "data_bytes": data_bytes, "member_bytes": info.file_size},
            )
        result = PartCRev1NPYHeader(
            member=member,
            version=version,
            shape=tuple(shape),
            fortran_order=bool(fortran_order),
            dtype=dtype.str,
            dtype_kind=dtype.kind,
            itemsize=int(dtype.itemsize),
            header_bytes=header_bytes,
            data_bytes=data_bytes,
            compressed_bytes=int(info.compress_size),
            uncompressed_bytes=int(info.file_size),
        )
        self._headers[member] = result
        return result

    def inspect_all_headers(self) -> list[PartCRev1NPYHeader]:
        return [self.header(name) for name in self.member_names]

    def read_prefix(self, member: str, prefix_count: int) -> np.ndarray:
        count = int(prefix_count)
        if count < 0 or count > self.max_prefix_count:
            raise PartCRev1InputSchemaError(
                "NPZ prefix request exceeds the configured bound",
                path=member,
                details={"requested": count, "maximum": self.max_prefix_count},
            )
        header = self.header(member)
        if len(header.shape) != 1:
            raise PartCRev1InputSchemaError("Bounded prefix reads require a vector member", path=member)
        if count > header.shape[0]:
            raise PartCRev1InputSchemaError("NPZ prefix request exceeds vector length", path=member)
        byte_count = count * header.itemsize
        with self._archive.open(self._infos[member], mode="r") as handle:
            _read_exact(handle, header.header_bytes)
            raw = _read_exact(handle, byte_count)
        self.bytes_requested += byte_count
        return np.frombuffer(raw, dtype=np.dtype(header.dtype), count=count).copy()

    def read_full_member(self, member: str) -> np.ndarray:
        if member not in self.full_member_allowlist:
            raise PartCRev1InputSchemaError("Full NPZ member read is not allowlisted", path=member)
        header = self.header(member)
        if len(header.shape) != 1:
            raise PartCRev1InputSchemaError("Allowlisted full NPZ member must be a vector", path=member)
        if header.data_bytes > self.full_member_limit:
            raise PartCRev1InputSchemaError(
                "Allowlisted full NPZ member exceeds the 1 MiB ceiling",
                path=member,
                details={"data_bytes": header.data_bytes, "limit": self.full_member_limit},
            )
        with self._archive.open(self._infos[member], mode="r") as handle:
            _read_exact(handle, header.header_bytes)
            raw = _read_exact(handle, header.data_bytes)
        self.bytes_requested += header.data_bytes
        return np.frombuffer(raw, dtype=np.dtype(header.dtype)).reshape(header.shape).copy()

    def read_member_bytes(self, member: str, size: int | None) -> bytes:
        if size is None or int(size) < 0:
            raise PartCRev1InputSchemaError("Unbounded NPZ member reads are prohibited", path=member)
        if int(size) > self.max_prefix_count:
            raise PartCRev1InputSchemaError("Raw NPZ member read exceeds configured byte bound", path=member)
        with self._archive.open(self._infos[member], mode="r") as handle:
            value = _read_exact(handle, int(size))
        self.bytes_requested += len(value)
        return value


def _file_state(path: Path) -> dict[str, Any]:
    value = path.stat()
    return {
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "modified_utc": datetime.fromtimestamp(value.st_mtime, tz=timezone.utc).isoformat(),
    }


def _database_state(path: Path) -> dict[str, Any]:
    sidecars: dict[str, dict[str, Any]] = {}
    for candidate in sorted(path.parent.glob(path.name + "*")):
        if candidate == path or not candidate.is_file():
            continue
        sidecars[candidate.name] = _file_state(candidate)
    return {"main": _file_state(path), "sidecars": sidecars}


def _check_magic(path: Path, role: str) -> None:
    with path.open("rb") as handle:
        first = handle.read(16)
        if role in {"graph_npz", "od_npz"} and first[:2] != b"PK":
            raise PartCRev1InputSchemaError("NPZ ZIP magic mismatch", role=role, path=path)
        if role in {"graph_nodes_parquet", "graph_arcs_parquet"}:
            if first[:4] != b"PAR1":
                raise PartCRev1InputSchemaError("Parquet leading magic mismatch", role=role, path=path)
            handle.seek(-4, os.SEEK_END)
            if handle.read(4) != b"PAR1":
                raise PartCRev1InputSchemaError("Parquet trailing magic mismatch", role=role, path=path)
        if role in {"rich_sqlite", "parent_gpkg"} and first != b"SQLite format 3\x00":
            raise PartCRev1InputSchemaError("SQLite file magic mismatch", role=role, path=path)


def _validate_npy_contract(header: PartCRev1NPYHeader, contract: Mapping[str, Any]) -> None:
    expected_shape = tuple(int(item) for item in contract["shape"])
    if header.shape != expected_shape:
        raise PartCRev1InputSchemaError(
            "NPY member shape mismatch", path=header.member, details={"expected": expected_shape, "actual": header.shape}
        )
    expected_dtype = str(contract["dtype"])
    actual_dtype = np.dtype(header.dtype)
    if expected_dtype == "string":
        matches = actual_dtype.kind in {"U", "S"}
    else:
        matches = actual_dtype == np.dtype(expected_dtype)
    if not matches:
        raise PartCRev1InputSchemaError(
            "NPY member dtype mismatch", path=header.member, details={"expected": expected_dtype, "actual": header.dtype}
        )


def _validate_od_sample_vectors(
    origins: np.ndarray,
    destinations: np.ndarray,
    demand: np.ndarray,
    origin_groups: np.ndarray,
    origin_offsets: np.ndarray,
    *,
    prefix_count: int,
    node_count: int,
    od_row_count: int,
    origin_count: int,
) -> None:
    if not (len(origins) == len(destinations) == len(demand) == prefix_count) or prefix_count <= 0:
        raise PartCRev1InputSchemaError("OD prefix vector lengths differ or form a no-op")
    if not np.all(np.isfinite(demand)) or np.any(demand <= 0):
        raise PartCRev1InputSchemaError("OD demand prefix contains nonpositive/nonfinite values")
    pairs = list(zip(origins.tolist(), destinations.tolist()))
    if pairs != sorted(pairs):
        raise PartCRev1InputSchemaError("OD pair prefix violates lexicographic vector order")
    if np.any(origins < 0) or np.any(destinations < 0) or np.any(origins >= node_count) or np.any(destinations >= node_count):
        raise PartCRev1InputSchemaError("OD prefix contains out-of-range graph node indices")
    if len(origin_groups) != origin_count or len(origin_offsets) != origin_count + 1:
        raise PartCRev1InputSchemaError("OD origin-group dimensions mismatch")
    if origin_offsets[0] != 0 or origin_offsets[-1] != od_row_count or np.any(np.diff(origin_offsets) <= 0):
        raise PartCRev1InputSchemaError("OD origin offsets are invalid")
    if np.any(np.diff(origin_groups) <= 0) or np.any(origin_groups < 0) or np.any(origin_groups >= node_count):
        raise PartCRev1InputSchemaError("OD origin-group indices are invalid")


def _bounded_parquet_sample(path: Path, contract: Mapping[str, Any], sample_rows: int) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    expected_rows = int(contract["row_count"])
    if parquet.metadata.num_rows != expected_rows:
        raise PartCRev1InputSchemaError(
            "Parquet row-count mismatch", path=path, details={"expected": expected_rows, "actual": parquet.metadata.num_rows}
        )
    required = dict(contract["columns"])
    schema = parquet.schema_arrow
    missing = sorted(set(required) - set(schema.names))
    if missing:
        raise PartCRev1InputSchemaError("Parquet required columns are missing", path=path, details={"missing": missing})
    actual_types = {name: str(schema.field(name).type) for name in required}
    wrong = {name: {"expected": required[name], "actual": actual_types[name]} for name in required if actual_types[name] != required[name]}
    if wrong:
        raise PartCRev1InputSchemaError("Parquet column type mismatch", path=path, details=wrong)
    batches = parquet.iter_batches(batch_size=sample_rows, columns=list(required), use_threads=False)
    try:
        batch = next(batches)
    except StopIteration as exc:
        raise PartCRev1InputSchemaError("Parquet sample is empty", path=path) from exc
    sample = batch.to_pydict()
    if batch.num_rows != sample_rows:
        raise PartCRev1InputSchemaError("Parquet sample did not return the configured row count", path=path)
    for key in contract["stable_keys"]:
        values = sample[key]
        normalized = [str(item).strip() if item is not None else "" for item in values]
        if not all(normalized) or len(normalized) != len(set(normalized)):
            raise PartCRev1InputSchemaError("Sampled stable key is blank or duplicated", path=path, details={"key": key})
    index_key = str(contract["index_key"])
    if sample[index_key] != list(range(sample_rows)):
        raise PartCRev1InputSchemaError("Parquet vector-order prefix mismatch", path=path, details={"key": index_key})
    return {
        "rows": int(parquet.metadata.num_rows),
        "row_groups": int(parquet.metadata.num_row_groups),
        "schema": actual_types,
        "sample": sample,
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    escaped = table.replace('"', '""')
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{escaped}")')}


def _validate_database(path: Path, contract: Mapping[str, Any], *, gpkg: bool) -> dict[str, Any]:
    before = _database_state(path)
    result: dict[str, Any] = {}
    with closing(partC_rev1_open_sqlite_strict_readonly(path)) as connection:
        if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
            raise PartCRev1InputSchemaError("query_only changed during database validation", path=path)
        if gpkg:
            catalog_objects = {
                str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
            }
            required_catalog = {"gpkg_contents", "gpkg_geometry_columns", "gpkg_ogr_contents"}
            missing_catalog = sorted(required_catalog - catalog_objects)
            if missing_catalog:
                raise PartCRev1InputSchemaError(
                    "GeoPackage catalog metadata is missing", path=path, details={"missing": missing_catalog}
                )
            application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
            user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if application_id != int(contract["application_id"]) or user_version != int(contract["user_version"]):
                raise PartCRev1InputIdentityError(
                    "GeoPackage application/user version mismatch",
                    path=path,
                    details={"application_id": application_id, "user_version": user_version},
                )
            contents = {
                str(row[0]): {"data_type": row[1], "srs_id": row[2]}
                for row in connection.execute("SELECT table_name,data_type,srs_id FROM gpkg_contents")
            }
            geometry_types = {
                str(row[0]): str(row[1])
                for row in connection.execute("SELECT table_name,geometry_type_name FROM gpkg_geometry_columns")
            }
            feature_counts = {
                str(row[0]): int(row[1])
                for row in connection.execute(
                    "SELECT table_name,feature_count FROM gpkg_ogr_contents WHERE feature_count IS NOT NULL"
                )
            }
            for layer, layer_contract in contract["layers"].items():
                if layer not in contents or int(contents[layer]["srs_id"]) != int(layer_contract["srs_id"]):
                    raise PartCRev1InputSchemaError("GeoPackage layer/CRS mismatch", path=path, details={"layer": layer})
                actual_geometry = geometry_types.get(layer, "")
                if actual_geometry.casefold() != str(layer_contract["geometry_type"]).casefold():
                    raise PartCRev1InputSchemaError(
                        "GeoPackage layer geometry type mismatch",
                        path=path,
                        details={"layer": layer, "expected": layer_contract["geometry_type"], "actual": actual_geometry},
                    )
                if feature_counts.get(layer) != int(layer_contract["feature_count"]):
                    raise PartCRev1InputSchemaError(
                        "GeoPackage layer feature-count metadata mismatch",
                        path=path,
                        details={
                            "layer": layer,
                            "expected": layer_contract["feature_count"],
                            "actual": feature_counts.get(layer),
                        },
                    )
                columns = _columns(connection, layer)
                missing = sorted(set(layer_contract["stable_keys"]) - columns)
                if missing:
                    raise PartCRev1InputSchemaError("GeoPackage stable key missing", path=path, details={"layer": layer, "missing": missing})
                key = str(layer_contract["stable_keys"][0]).replace('"', '""')
                escaped_layer = layer.replace('"', '""')
                rows = connection.execute(
                    f'SELECT "{key}" FROM "{escaped_layer}" WHERE "{key}" IS NOT NULL LIMIT 64'
                ).fetchall()
                values = [str(row[0]).strip() for row in rows]
                if not values or not all(values) or len(values) != len(set(values)):
                    raise PartCRev1InputSchemaError("GeoPackage stable-ID sample failed", path=path, details={"layer": layer})
            result.update(
                {
                    "application_id": application_id,
                    "user_version": user_version,
                    "contents": contents,
                    "geometry_types": geometry_types,
                    "feature_counts": feature_counts,
                }
            )
        else:
            objects = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
            }
            for table, table_contract in contract["tables"].items():
                if table not in objects:
                    raise PartCRev1InputSchemaError("Required SQLite table is missing", path=path, details={"table": table})
                columns = _columns(connection, table)
                missing = sorted(set(table_contract["required_columns"]) - columns)
                if missing:
                    raise PartCRev1InputSchemaError(
                        "Required SQLite columns are missing", path=path, details={"table": table, "missing": missing}
                    )
            arc_rows = connection.execute(
                "SELECT directed_arc_id,from_node_uid,to_node_uid FROM partC_rev1_directed_arcs_enriched "
                "WHERE directed_arc_id IS NOT NULL LIMIT 64"
            ).fetchall()
            arc_ids = [str(row[0]).strip() for row in arc_rows]
            if len(arc_ids) != 64 or len(arc_ids) != len(set(arc_ids)) or not all(arc_ids):
                raise PartCRev1InputSchemaError("SQLite directed-arc stable-ID sample failed", path=path)
            od_rows = connection.execute(
                "SELECT origin_node_uid,destination_node_uid,demand_solver_unit FROM partC_rev1_od_centroid "
                "WHERE origin_node_uid IS NOT NULL AND destination_node_uid IS NOT NULL LIMIT 64"
            ).fetchall()
            if len(od_rows) != 64:
                raise PartCRev1InputSchemaError("SQLite OD stable-ID sample failed", path=path)
            result.update({"required_objects": sorted(contract["tables"]), "stable_id_sample_rows": 64})
    after = _database_state(path)
    if before != after:
        raise PartCRev1InputIdentityError(
            "SQLite/GeoPackage main file or sidecar state changed during read-only validation",
            path=path,
            details={"before": before, "after": after},
        )
    result["pre_state"] = before
    result["post_state"] = after
    return result


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def _write_outputs(
    output_dir: Path,
    report: PartCRev1ValidationReport,
    run_manifest: Mapping[str, Any],
    resolved_inputs: Mapping[str, Path],
    command: Sequence[str],
) -> None:
    status = {
        "phase_id": run_manifest["phase_id"],
        "step_id": run_manifest["step_id"],
        "run_id": run_manifest["run_id"],
        "status": report.status,
        "next_step_authorized": False,
        "scientific_execution_performed": False,
    }
    _write_json(output_dir / "partC_rev1_step1_1_status.json", status)
    _write_json(output_dir / "partC_rev1_step1_1_run_manifest.json", run_manifest)
    _write_json(output_dir / "partC_rev1_step1_1_resolved_inputs.json", {key: str(value) for key, value in resolved_inputs.items()})
    _write_json(output_dir / "partC_rev1_step1_1_validation.json", report.to_dict())
    summary = (
        "# Phase 1 Step 1.1 bounded validation\n\n"
        f"- Status: `{report.status}`\n"
        f"- Validation level: `{report.validation_level}`\n"
        f"- Checks: `{len(report.checks)}`\n"
        f"- Diagnostics: `{len(report.diagnostics)}`\n"
        "- Scientific execution: `false`\n"
        "- Next step authorized: `false`\n"
    )
    (output_dir / "partC_rev1_step1_1_validation_summary.md").write_text(summary, encoding="utf-8", newline="\n")
    with (output_dir / "partC_rev1_step1_1_diagnostics.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in report.diagnostics:
            handle.write(json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
    with (output_dir / "partC_rev1_step1_1_resource_usage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(report.resource_usage))
        writer.writeheader()
        writer.writerow(report.resource_usage)
    (output_dir / "partC_rev1_step1_1_command_log.md").write_text(
        "# Step 1.1 command log\n\n```text\n" + " ".join(command) + "\n```\n",
        encoding="utf-8",
        newline="\n",
    )
    artifact_path = output_dir / "partC_rev1_step1_1_artifact_manifest.csv"
    output_files = [item for item in sorted(output_dir.iterdir()) if item.is_file() and item != artifact_path]
    with artifact_path.open("w", encoding="utf-8", newline="") as handle:
        columns = ["artifact_id", "path", "format", "size_bytes", "sha256", "status"]
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for item in output_files:
            writer.writerow(
                {
                    "artifact_id": item.stem,
                    "path": str(item),
                    "format": item.suffix.lstrip("."),
                    "size_bytes": item.stat().st_size,
                    "sha256": partC_rev1_file_sha256(item),
                    "status": "COMPLETE" if report.status == "PASS" else "FAILED",
                }
            )


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "-c", "core.longpaths=true", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _load_config(path: Path, schema_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, jsonschema.exceptions.ValidationError, jsonschema.exceptions.SchemaError) as exc:
        raise PartCRev1InputSchemaError(f"Invalid Step 1.1 smoke configuration: {exc}", path=path) from exc
    if payload.get("schema_version") != "1.0":
        raise PartCRev1InputSchemaError("Unsupported Step 1.1 smoke config schema version", path=path)
    tasks = payload.get("tasks", [])
    if set(tasks) != _EXPECTED_TASKS or len(tasks) != len(_EXPECTED_TASKS):
        raise PartCRev1InputSchemaError("Smoke task list must be the exact nonempty approved task set", path=path)
    return payload


def partC_rev1_validate_authoritative_inputs(
    *,
    phase_id: str,
    step_id: str,
    run_id: str,
    validation_level: str,
    registry_path: str | Path,
    registry_schema_path: str | Path,
    manifest_path: str | Path,
    manifest_schema_path: str | Path,
    config_path: str | Path,
    config_schema_path: str | Path,
    output_root: str | Path,
    temp_root: str | Path,
    command: Sequence[str] = (),
) -> tuple[PartCRev1ValidationReport, Path]:
    if phase_id != "01" or step_id != "1.1E" or validation_level not in {"REGISTRY", "METADATA", "SAMPLE"}:
        raise PartCRev1InputSchemaError("Explicit phase, step, or validation level is invalid")
    if not _RUN_ID.fullmatch(run_id):
        raise PartCRev1InputSchemaError("Run ID is empty or unsafe")
    started = time.monotonic()
    process = psutil.Process()
    report = PartCRev1ValidationReport(validation_level=validation_level)
    roots = partC_rev1_load_phase1_data_roots(registry_path, registry_schema_path)
    manifest = partC_rev1_load_authoritative_input_manifest(manifest_path, manifest_schema_path)
    config = _load_config(Path(config_path), Path(config_schema_path))
    if config["phase_id"] != phase_id or config["step_id"] != step_id or config["validation_level"] != validation_level:
        raise PartCRev1InputIdentityError("CLI phase/step/level does not match the approved smoke config")
    resolved_inputs = manifest.resolve_inputs(roots)
    output_base = partC_rev1_assert_phase1_destination(output_root, roots, "output")
    temp_base = partC_rev1_assert_phase1_destination(temp_root, roots, "temp")
    if output_base != roots.phase1_output_root or temp_base != roots.phase1_temp_root:
        raise PartCRev1InputPathError("CLI output/temp roots must equal the approved registry roots")
    output_dir = partC_rev1_assert_phase1_destination(
        output_base / "phase_01" / "step_1_1" / run_id, roots, "output"
    )
    temp_dir = partC_rev1_assert_phase1_destination(
        temp_base / "phase_01" / "step_1_1" / run_id, roots, "temp"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    temp_dir.mkdir(parents=True, exist_ok=False)

    budget = config["resource_budget"]
    peak_rss = process.memory_info().rss
    initial_disk = _directory_bytes(output_dir) + _directory_bytes(temp_dir)

    def guard() -> None:
        nonlocal peak_rss
        elapsed = time.monotonic() - started
        peak_rss = max(peak_rss, process.memory_info().rss)
        added_disk = _directory_bytes(output_dir) + _directory_bytes(temp_dir) - initial_disk
        if elapsed > float(budget["hard_wall_seconds"]):
            raise PartCRev1ResourceLimitError("Step 1.1 wall-clock budget exceeded", details={"elapsed": elapsed})
        if peak_rss > int(budget["hard_rss_bytes"]):
            raise PartCRev1ResourceLimitError("Step 1.1 RSS budget exceeded", details={"peak_rss": peak_rss})
        if added_disk > int(budget["hard_added_disk_bytes"]):
            raise PartCRev1ResourceLimitError("Step 1.1 disk budget exceeded", details={"added_disk": added_disk})

    pre_states = {role: _file_state(path) for role, path in resolved_inputs.items()}
    report.resolved_inputs = {role: str(path) for role, path in resolved_inputs.items()}
    report.add_check("REGISTRY_AND_MANIFEST", "PASS", role_count=len(resolved_inputs), task_count=len(config["tasks"]))

    items = {str(item["role"]): item for item in manifest.inputs}
    if validation_level in {"METADATA", "SAMPLE"}:
        for role in sorted(resolved_inputs):
            path = resolved_inputs[role]
            item = items[role]
            if not path.is_file():
                raise PartCRev1InputPathError("Authoritative input is not a regular file", role=role, path=path)
            state = _file_state(path)
            if state["size_bytes"] != int(item["size_bytes"]):
                raise PartCRev1InputIdentityError(
                    "Authoritative input size mismatch", role=role, path=path, details={"expected": item["size_bytes"], "actual": state["size_bytes"]}
                )
            _check_magic(path, role)
            signature = partC_rev1_file_signature(path)
            expected_signature = f"{item['signature']['algorithm']}:{item['signature']['value']}"
            if signature != expected_signature:
                raise PartCRev1InputSignatureError(
                    "Authoritative input reviewed signature mismatch",
                    role=role,
                    path=path,
                    details={"expected": expected_signature, "actual": signature},
                )
            report.add_check("REVIEWED_SIGNATURE", "PASS", role=role, signature=signature)
            guard()

        graph_meta = json.loads(resolved_inputs["graph_metadata"].read_text(encoding="utf-8"))
        od_meta = json.loads(resolved_inputs["od_metadata"].read_text(encoding="utf-8"))
        for role, payload in (("graph_metadata", graph_meta), ("od_metadata", od_meta)):
            required_fields = set(items[role]["required"]["fields"])
            missing_fields = sorted(required_fields - set(payload))
            if missing_fields:
                raise PartCRev1InputSchemaError(
                    "Authoritative metadata fields are missing",
                    role=role,
                    path=resolved_inputs[role],
                    details={"missing": missing_fields},
                )
        identities = manifest.payload["identities"]
        comparisons = {
            "topology": graph_meta.get("topology_hash"),
            "parent_assignment_graph": graph_meta.get("parent_assignment_graph_hash", graph_meta.get("parent_graph_hash")),
            "corrected_graph": graph_meta.get("assignment_graph_hash"),
            "od": od_meta.get("od_hash"),
            "parent_partition": graph_meta.get("parent_partition_hash"),
            "correction": graph_meta.get("correction_hash"),
            "corrected_partition": graph_meta.get("partition_hash"),
        }
        for key, actual in comparisons.items():
            if actual != identities[key]:
                raise PartCRev1InputIdentityError(
                    "Authoritative identity-chain mismatch", role=key, details={"expected": identities[key], "actual": actual}
                )
        report.add_check("IDENTITY_CHAIN", "PASS", identities=comparisons)

    if validation_level == "SAMPLE":
        sample_rows = int(config["sample_limits"]["parquet_rows"])
        prefix_count = int(config["sample_limits"]["npz_prefix_values"])
        graph_contract = items["graph_npz"]["required"]
        with PartCRev1BoundedNPZReader(
            resolved_inputs["graph_npz"],
            expected_members=graph_contract["members"],
            max_prefix_count=prefix_count,
        ) as graph_reader:
            graph_headers = graph_reader.inspect_all_headers()
            for header in graph_headers:
                _validate_npy_contract(header, graph_contract["members"][header.member])
            graph_node_uids = graph_reader.read_prefix("node_idx_to_uid.npy", prefix_count).astype(str)
            graph_arc_ids = graph_reader.read_prefix("arc_idx_to_directed_arc_id.npy", prefix_count).astype(str)
            graph_tail = graph_reader.read_prefix("tail.npy", prefix_count)
            graph_head = graph_reader.read_prefix("head.npy", prefix_count)
            if np.any(graph_tail < 0) or np.any(graph_head < 0) or np.any(graph_tail >= 346373) or np.any(graph_head >= 346373):
                raise PartCRev1InputSchemaError("Graph endpoint prefix contains out-of-range node indices")
            graph_npz_evidence = {
                "headers": [asdict(item) for item in graph_headers],
                "bytes_requested": graph_reader.bytes_requested,
            }

        od_contract = items["od_npz"]["required"]
        with PartCRev1BoundedNPZReader(
            resolved_inputs["od_npz"],
            expected_members=od_contract["members"],
            max_prefix_count=prefix_count,
        ) as od_reader:
            od_headers = od_reader.inspect_all_headers()
            for header in od_headers:
                _validate_npy_contract(header, od_contract["members"][header.member])
            origins = od_reader.read_prefix("origin_node_idx.npy", prefix_count)
            destinations = od_reader.read_prefix("destination_node_idx.npy", prefix_count)
            demand = od_reader.read_prefix("demand.npy", prefix_count)
            origin_groups = od_reader.read_full_member("origin_group_node_idx.npy")
            origin_offsets = od_reader.read_full_member("origin_offsets.npy")
            _validate_od_sample_vectors(
                origins,
                destinations,
                demand,
                origin_groups,
                origin_offsets,
                prefix_count=prefix_count,
                node_count=346373,
                od_row_count=41637922,
                origin_count=3225,
            )
            od_npz_evidence = {
                "headers": [asdict(item) for item in od_headers],
                "bytes_requested": od_reader.bytes_requested,
                "full_members": ["origin_group_node_idx.npy", "origin_offsets.npy"],
            }
        report.add_check("BOUNDED_NPZ", "PASS", graph=graph_npz_evidence, od=od_npz_evidence)
        guard()

        node_sample = _bounded_parquet_sample(
            resolved_inputs["graph_nodes_parquet"], items["graph_nodes_parquet"]["required"], sample_rows
        )
        arc_sample = _bounded_parquet_sample(
            resolved_inputs["graph_arcs_parquet"], items["graph_arcs_parquet"]["required"], sample_rows
        )
        if [str(item) for item in node_sample["sample"]["partC_rev1_node_uid"]] != graph_node_uids.tolist():
            raise PartCRev1InputIdentityError("Graph node stable-ID NPZ/Parquet prefix mismatch")
        if [str(item) for item in arc_sample["sample"]["directed_arc_id"]] != graph_arc_ids.tolist():
            raise PartCRev1InputIdentityError("Graph arc stable-ID NPZ/Parquet prefix mismatch")
        report.add_check(
            "PARQUET_STABLE_IDS",
            "PASS",
            node_rows=node_sample["rows"],
            arc_rows=arc_sample["rows"],
            sample_rows=sample_rows,
        )
        guard()

        sqlite_evidence = _validate_database(
            resolved_inputs["rich_sqlite"], items["rich_sqlite"]["required"], gpkg=False
        )
        gpkg_evidence = _validate_database(
            resolved_inputs["parent_gpkg"], items["parent_gpkg"]["required"], gpkg=True
        )
        report.add_check("SQLITE_GPKG_STRICT_READONLY", "PASS", sqlite=sqlite_evidence, gpkg=gpkg_evidence)
        guard()

        terminal_path = resolved_inputs["corrected_terminal_registry"]
        with terminal_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            expected_columns = items["corrected_terminal_registry"]["required"]["columns"]
            if reader.fieldnames != expected_columns:
                raise PartCRev1InputSchemaError("Corrected terminal-registry header mismatch", path=terminal_path)
            terminal_rows = list(reader)
        regions = [str(row["region_uuid"]).strip() for row in terminal_rows]
        if len(regions) != 2242 or len(regions) != len(set(regions)) or not all(regions):
            raise PartCRev1InputSchemaError("Corrected terminal-registry stable region IDs are invalid", path=terminal_path)
        graph_regions = set(str(item) for item in graph_meta["region_labels"] if str(item) != "backbone")
        if set(regions) != graph_regions:
            raise PartCRev1InputIdentityError("Corrected terminal-registry regions do not match graph metadata")
        report.add_check(
            "CORRECTED_PARTITION_TERMINALS",
            "PASS",
            rows=len(regions),
            registry_sha256=partC_rev1_file_sha256(terminal_path),
        )
        report.add_check("INHERITED_UNRESOLVED_CATEGORIES", "PASS_REPORTED_ONLY", facts=manifest.payload["inherited_facts"])
        guard()

    post_states = {role: _file_state(path) for role, path in resolved_inputs.items()}
    if pre_states != post_states:
        raise PartCRev1InputIdentityError(
            "An authoritative input size or modification time changed during validation",
            details={"before": pre_states, "after": post_states},
        )
    report.add_check("AUTHORITATIVE_INPUT_INVARIANCE", "PASS", before=pre_states, after=post_states)
    guard()
    report.finish("PASS")
    elapsed = time.monotonic() - started
    report.resource_usage = {
        "run_id": run_id,
        "wall_seconds": round(elapsed, 6),
        "peak_rss_bytes": int(peak_rss),
        "added_disk_bytes_before_reports": int(_directory_bytes(output_dir) + _directory_bytes(temp_dir) - initial_disk),
        "worker_count": 1,
    }
    run_manifest = {
        "schema_version": "1.0",
        "phase_id": phase_id,
        "step_id": step_id,
        "run_id": run_id,
        "mode": "bounded_read_only_smoke",
        "code_commit": _git_head(roots.phase1_worktree_root),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": partC_rev1_file_sha256(config_path),
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_fingerprint": manifest.manifest_fingerprint,
        "tasks": config["tasks"],
        "task_count": len(config["tasks"]),
        "expected_outputs": config["expected_outputs"],
        "acceptance_tests": config["acceptance_tests"],
        "resource_budget": budget,
        "started_at": report.started_at,
        "finished_at": report.finished_at,
        "status": "PASS",
        "resume_policy": "not_resumable_bounded_smoke",
    }
    _write_outputs(output_dir, report, run_manifest, resolved_inputs, command)
    final_disk = _directory_bytes(output_dir) + _directory_bytes(temp_dir) - initial_disk
    if final_disk > int(budget["hard_added_disk_bytes"]):
        raise PartCRev1ResourceLimitError("Final Step 1.1 report disk budget exceeded", details={"added_disk": final_disk})
    return report, output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validation-level", required=True, choices=("REGISTRY", "METADATA", "SAMPLE"))
    parser.add_argument("--registry", required=True)
    parser.add_argument("--registry-schema", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-schema", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-schema", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--temp-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _parser().parse_args(arguments)
        report, output_dir = partC_rev1_validate_authoritative_inputs(
            phase_id=args.phase_id,
            step_id=args.step_id,
            run_id=args.run_id,
            validation_level=args.validation_level,
            registry_path=args.registry,
            registry_schema_path=args.registry_schema,
            manifest_path=args.manifest,
            manifest_schema_path=args.manifest_schema,
            config_path=args.config,
            config_schema_path=args.config_schema,
            output_root=args.output_root,
            temp_root=args.temp_root,
            command=[sys.executable, "-m", __name__, *arguments],
        )
        print(json.dumps({"status": report.status, "output_dir": str(output_dir)}, sort_keys=True))
        return 0
    except PartCRev1ResourceLimitError as exc:
        print(json.dumps(exc.diagnostic.to_dict(), sort_keys=True), file=sys.stderr)
        return 3
    except PartCRev1InputValidationError as exc:
        print(json.dumps(exc.diagnostic.to_dict(), sort_keys=True), file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {"code": "PARTC_REV1_UNEXPECTED_ERROR", "severity": "ERROR", "message": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
