"""Accepted Phase 1/2 identity resolution and immutable input verification."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from partC_rev1_src.partC_rev1_hashing import partC_rev1_file_sha256, partC_rev1_file_signature

from .models import AuthorityPaths
from .schemas import canonical_json_bytes


EXPECTED_GRAPH_IDENTITY = "15a6fc40f38992504d74e14079a5d5e3aa2b7812faa16c33b859ed053bd7c31c"
EXPECTED_OD_IDENTITY = "7d95a193d5b858bc9ac9077279161038de8f79f1782399bdee1fa91129be0e5a"
EXPECTED_S03_SUITE_ID = "phase2_suite_03_r1"
EXPECTED_S03_MEMBERSHIP = "e6eb8344ce9d5f702638c47ee91a7dc44305a919433b48ad7f26bbb73075f10e"
EXPECTED_S03_NODE_MEMBERSHIP = "92b502c34bdb58037f5e9ce22c24cc7063027d4f55c71f5f97bf7250a19bbbc5"
EXPECTED_S03_STRUCTURAL_SIGNATURE = "958a67472531815497be119fff10430c8b9fd553769c0cf6e1319070e57fe59c"
EXPECTED_PHASE2_SUPPLEMENT_HEAD = "2134390865ff77ecc8dc2487b6cec86ed1065004"


class Phase3AAuthorityError(RuntimeError):
    pass


def load_execution_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Phase3AAuthorityError(f"Execution config is not strict UTF-8 JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise Phase3AAuthorityError("Execution config must be a JSON object")
    for key, expected in (
        ("phase_id", "phase_03a"),
        ("step_id", "bacn_method_selection"),
        ("suite_id", "s03"),
    ):
        if payload.get(key) != expected:
            raise Phase3AAuthorityError(f"Execution config {key} must equal {expected!r}")
    if not payload.get("campaign_id") or not payload.get("run_id"):
        raise Phase3AAuthorityError("Execution config requires nonempty campaign_id and run_id")
    return payload


def resolve_phase2_suite_id(config: Mapping[str, Any]) -> str:
    """Resolve the accepted Phase 2 suite without changing the legacy s03 default."""

    value = str(config.get("phase2_suite_id", EXPECTED_S03_SUITE_ID)).strip()
    if not value.startswith("phase2_suite_"):
        raise Phase3AAuthorityError(f"Invalid accepted Phase 2 suite ID: {value!r}")
    return value


def resolve_authority_paths(config: Mapping[str, Any]) -> AuthorityPaths:
    project_root = Path(str(config["project_root"])).resolve(strict=True)
    phase2_root = Path(str(config["accepted_phase2_output_root"])).resolve(strict=True)
    phase2_suite_id = resolve_phase2_suite_id(config)
    suite_root = phase2_root / "suites" / phase2_suite_id
    if not suite_root.is_dir():
        raise Phase3AAuthorityError(f"Accepted Phase 2 suite root is missing: {suite_root}")
    input_versions = config.get("input_versions")
    if not isinstance(input_versions, Mapping):
        # The final Task 1 config copies the accepted paths explicitly.  During
        # bootstrap, resolve them from the accepted Phase 2 producer config.
        producer_config = project_root / "partC_doc_icnn" / "phase_2_work" / "campaigns" / "phase2_split_mapping_20260817_v1" / "PHASE2_EXECUTION_CONFIG.json"
        producer = json.loads(producer_config.read_text(encoding="utf-8"))
        input_versions = producer["input_versions"]
    return AuthorityPaths(
        project_root=project_root,
        phase2_output_root=phase2_root,
        suite_root=suite_root.resolve(strict=True),
        graph_npz=Path(str(input_versions["graph_npz"])).resolve(strict=True),
        graph_nodes_parquet=Path(str(input_versions["graph_nodes_parquet"])).resolve(strict=True),
        graph_arcs_parquet=Path(str(input_versions["graph_arcs_parquet"])).resolve(strict=True),
        graph_metadata_json=Path(str(input_versions["graph_metadata_json"])).resolve(strict=True),
        od_npz=Path(str(input_versions["od_npz"])).resolve(strict=True),
        od_metadata_json=Path(str(input_versions["od_metadata_json"])).resolve(strict=True),
        rich_sqlite=Path(str(input_versions["rich_sqlite"])).resolve(strict=True),
    )


def verify_authority(
    paths: AuthorityPaths,
    *,
    verify_large_signatures: bool = True,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    suite_identity_path = paths.suite_root / "suite_identity.json"
    suite_identity = json.loads(suite_identity_path.read_text(encoding="utf-8"))
    suite_id = str(paths.suite_root.name)
    supplied = dict(expected_identity or {})
    default_s03 = suite_id == EXPECTED_S03_SUITE_ID
    expected = {
        "suite_id": str(supplied.get("suite_id", supplied.get("phase2_suite_id", suite_id))),
        "graph_identity": str(supplied.get("graph_identity", supplied.get("graph", EXPECTED_GRAPH_IDENTITY))),
        "od_identity": str(supplied.get("od_identity", supplied.get("od", EXPECTED_OD_IDENTITY))),
    }
    optional_expected = {
        "membership_hash": supplied.get(
            "membership_hash", EXPECTED_S03_MEMBERSHIP if default_s03 else None
        ),
        "node_membership_hash": supplied.get(
            "node_membership_hash", EXPECTED_S03_NODE_MEMBERSHIP if default_s03 else None
        ),
        "structural_signature": supplied.get(
            "structural_signature", EXPECTED_S03_STRUCTURAL_SIGNATURE if default_s03 else None
        ),
    }
    expected.update({key: str(value) for key, value in optional_expected.items() if value is not None})
    mismatches = {key: {"expected": value, "actual": suite_identity.get(key)} for key, value in expected.items() if suite_identity.get(key) != value}
    if mismatches:
        raise Phase3AAuthorityError(f"Accepted Phase 2 suite identity mismatch: {mismatches}")
    manifest_path = paths.suite_root / "artifact_manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    manifest_failures: list[str] = []
    for row in manifest:
        artifact = paths.suite_root / str(row["relative_path"])
        if not artifact.is_file():
            manifest_failures.append(f"missing:{row['relative_path']}")
            continue
        if artifact.stat().st_size != int(row["size_bytes"]):
            manifest_failures.append(f"size:{row['relative_path']}")
            continue
        if partC_rev1_file_sha256(artifact) != str(row["sha256"]):
            manifest_failures.append(f"sha256:{row['relative_path']}")
    if manifest_failures:
        raise Phase3AAuthorityError(f"Accepted Phase 2 suite manifest failed: {manifest_failures[:10]}")
    graph_meta = json.loads(paths.graph_metadata_json.read_text(encoding="utf-8"))
    od_meta = json.loads(paths.od_metadata_json.read_text(encoding="utf-8"))
    if graph_meta.get("graph_hash") != EXPECTED_GRAPH_IDENTITY:
        raise Phase3AAuthorityError("Graph metadata identity differs from accepted Phase 1/2 authority")
    if od_meta.get("od_hash") != EXPECTED_OD_IDENTITY:
        raise Phase3AAuthorityError("OD metadata identity differs from accepted Phase 1/2 authority")
    input_signatures: dict[str, dict[str, Any]] = {}
    for name, path in (
        ("graph_npz", paths.graph_npz),
        ("graph_nodes_parquet", paths.graph_nodes_parquet),
        ("graph_arcs_parquet", paths.graph_arcs_parquet),
        ("graph_metadata_json", paths.graph_metadata_json),
        ("od_npz", paths.od_npz),
        ("od_metadata_json", paths.od_metadata_json),
        ("rich_sqlite", paths.rich_sqlite),
    ):
        record = {"path": str(path), "size_bytes": int(path.stat().st_size)}
        if verify_large_signatures:
            record["reviewed_signature"] = partC_rev1_file_signature(path)
        input_signatures[name] = record
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "suite_identity": suite_identity,
        "suite_identity_sha256": partC_rev1_file_sha256(suite_identity_path),
        "suite_manifest_sha256": partC_rev1_file_sha256(manifest_path),
        "suite_manifest_member_count": len(manifest),
        "graph_identity": EXPECTED_GRAPH_IDENTITY,
        "od_identity": EXPECTED_OD_IDENTITY,
        "input_signatures": input_signatures,
        "accepted_artifacts_modified": False,
    }


def repository_inventory(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve(strict=True)
    head = _git(root, "rev-parse", "HEAD").strip()
    branch = _git(root, "branch", "--show-current").strip()
    status = _git(root, "status", "--short", "--branch").splitlines()
    return {
        "project_root": str(root),
        "git_head": head,
        "branch": branch,
        "git_status": status,
    }


def config_sha256(config: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(dict(config))).hexdigest()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "-c", "core.longpaths=true", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout
