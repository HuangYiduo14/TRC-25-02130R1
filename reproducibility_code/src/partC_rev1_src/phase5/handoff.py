"""Compact, write-new Phase 5 synchronization-handoff builder."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .pilot import (
    BLOCKED_STATUS,
    FINAL_STATUS,
    OUTER_AUTHORITY_SHA256,
    PARENT_AUTHORITY_SHA256,
    STARTING_HEAD,
    _describe_reopen,
    _validate_required_outputs,
    canonical_json,
    git,
    sha256_file,
    write_json,
    write_text,
)


MAX_MEMBER_BYTES = 25 * 1024 * 1024
TARGET_ZIP_BYTES = 100 * 1024 * 1024
HARD_ZIP_BYTES = 200 * 1024 * 1024
SAFE_HANDOFF_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validated_handoff_id(value: str) -> str:
    token = str(value)
    if not SAFE_HANDOFF_ID.fullmatch(token):
        raise ValueError(
            "handoff_id must be a 1-128 character alphanumeric/underscore/hyphen token"
        )
    return token


def _contained(parent: Path, *parts: str) -> Path:
    root = parent.resolve()
    result = root.joinpath(*parts).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Handoff target escapes its declared parent: {result}")
    return result


def _qualified_git_state(repo: Path, live_head: str) -> dict[str, object]:
    if live_head == STARTING_HEAD:
        raise RuntimeError("A distinct Phase 5 checkpoint commit is required before handoff")
    lines = git(
        repo, "status", "--porcelain=v1", "--untracked-files=all"
    ).stdout.splitlines()
    tracked_dirty = [line for line in lines if not line.startswith("?? ")]
    task_prefixes = (
        "partC_rev1_src/phase5/",
        "tests/test_phase5_",
        "partC_doc_icnn/phase_5_work/",
        "partC_doc_icnn/review_gates/phase_05/",
    )
    task_untracked = [
        line
        for line in lines
        if line.startswith("?? ")
        and line[3:].replace("\\", "/").startswith(task_prefixes)
    ]
    if tracked_dirty or task_untracked:
        raise RuntimeError(
            "Checkpoint tree is not qualified for handoff: "
            f"tracked={tracked_dirty}, task_untracked={task_untracked}"
        )
    changed = set(
        git(repo, "diff", "--name-only", STARTING_HEAD, live_head).stdout.splitlines()
    )
    required = {
        "partC_rev1_src/phase5/pilot.py",
        "partC_rev1_src/phase5/handoff.py",
        "partC_doc_icnn/phase_5_work/phase5_backbone_compilation_pilot_execution_config.json",
        "partC_doc_icnn/phase_5_work/PHASE5_BACKBONE_COMPILATION_PILOT_STATUS.json",
        "partC_doc_icnn/phase_5_work/PHASE5_BACKBONE_COMPILATION_PILOT_RESULT.md",
    }
    missing = sorted(required - changed)
    if missing:
        raise RuntimeError(f"Checkpoint commit omits required Phase 5 records: {missing}")
    return {
        "status_lines": lines,
        "tracked_dirty_count": len(tracked_dirty),
        "task_owned_untracked_count": len(task_untracked),
        "checkpoint_changed_path_count": len(changed),
        "status": "QUALIFIED_UNRELATED_UNTRACKED_ALLOWED",
    }


def _verified_post_checkpoint_binding(
    output: Path,
    *,
    branch: str,
    live_head: str,
    scientific_status: str,
    status: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = output / "repository_audit" / "post_checkpoint.json"
    if not receipt_path.is_file():
        raise RuntimeError("Post-checkpoint receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not (
        status.get("post_checkpoint_head_bound") is True
        and status.get("ending_head") == live_head
        and receipt.get("branch") == branch
        and receipt.get("starting_head") == STARTING_HEAD
        and receipt.get("ending_head") == live_head
        and receipt.get("starting_head_is_ancestor") is True
        and receipt.get("scientific_status") == scientific_status
        and receipt.get("status") == "PASS"
    ):
        raise RuntimeError("Post-checkpoint status/receipt binding is incomplete")
    return receipt


def _code_snapshot_paths(repo: Path) -> list[Path]:
    return [
        *sorted((repo / "partC_rev1_src" / "phase5").glob("*.py")),
        *sorted((repo / "tests").glob("test_phase5_*.py")),
        repo / "partC_rev1_src" / "phase4" / "inputs.py",
        repo / "partC_rev1_src" / "phase4" / "models.py",
        repo / "partC_rev1_src" / "phase4" / "tracing.py",
        repo / "partC_rev1_src" / "phase3a" / "diagnostics.py",
        repo / "partC_rev1_src" / "phase3a" / "liftability.py",
        repo / "partC_rev1_src" / "partC_rev1_shortest_paths.py",
    ]


def _validate_supersession(repo: Path, handoff_id: str | None) -> None:
    if handoff_id is None:
        return
    source = _contained(
        repo,
        "partC_doc_icnn",
        "handoffs",
        "phase_05",
        "step_backbone_compilation_pilot",
        handoff_id,
    )
    bundle = _contained(
        repo,
        "artifacts_icnn",
        "handoff_bundles",
        "phase_05",
        f"phase05_stepbackbone_compilation_pilot_{handoff_id}_SYNC.zip",
    )
    if not source.is_dir() or not bundle.is_file() or not Path(str(bundle) + ".sha256").is_file():
        raise FileNotFoundError(
            f"Superseded handoff is not an intact local package: {handoff_id}"
        )


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree_files(source: Path, destination: Path, patterns: Iterable[str]) -> None:
    for pattern in patterns:
        for path in sorted(source.glob(pattern)):
            if path.is_file():
                _copy(path, destination / path.relative_to(source))


def _stable_sample(frame: pd.DataFrame, maximum: int, key_columns: list[str]) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy()
    keys = frame[key_columns].astype(str).agg("\x1f".join, axis=1)
    ranks = keys.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    return frame.loc[ranks.sort_values(kind="mergesort").index[:maximum]].sort_values(
        key_columns, kind="mergesort"
    )


def _write_samples(output: Path, destination: Path) -> list[dict[str, object]]:
    specifications = (
        ("topology/chain_candidates.parquet", 5000, ["suite_short_id", "chain_id"]),
        ("bundle_oracle/oracle_sample_results.parquet", 5000, ["canonical_macro_signature", "query_flow", "query_role"]),
        ("runtime_fits/runtime_fit_results.parquet", 5000, ["canonical_macro_signature", "runtime_class", "variant_id"]),
        ("charge_audit/bacn_visit_semantics.parquet", 5000, ["suite_short_id", "benchmark_id", "bacn_id"]),
        ("charge_audit/fixed_charge_candidates.parquet", 5000, ["suite_short_id", "bacn_id"]),
        ("path_validation/path_cost_sample.parquet", 5000, ["evaluation_role", "sample_od_id"]),
    )
    receipts: list[dict[str, object]] = []
    for relative, maximum, keys in specifications:
        source = output / relative
        frame = pd.read_parquet(source)
        sample = _stable_sample(frame, maximum, keys)
        name = relative.replace("/", "__").replace(".parquet", "__sample.csv")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(target, index=False)
        receipts.append(
            {
                "source": relative,
                "source_rows": len(frame),
                "sample_rows": len(sample),
                "sample_method": "lowest_sha256_of_declared_stable_key",
                "key_columns": keys,
                "sample_path": target.name,
            }
        )
    write_json(destination / "SAMPLE_METHODS.json", receipts)
    return receipts


def _index_external_output(output: Path, destination: Path) -> pd.DataFrame:
    """Create a tolerant, byte-authenticated index suitable for blocked evidence."""

    rows: list[dict[str, object]] = []
    for path in sorted(value for value in output.rglob("*") if value.is_file()):
        relative = path.relative_to(output).as_posix()
        try:
            digest = sha256_file(path)
        except Exception as error:  # pragma: no cover - filesystem failure path
            digest = "UNREADABLE"
            reopen = f"HASH_FAIL:{type(error).__name__}:{error}"
            count = -1
            schema = "{}"
        else:
            try:
                _, count, schema = _describe_reopen(path)
                reopen = "PASS"
            except Exception as error:
                reopen = f"FAIL:{type(error).__name__}:{error}"
                count = -1
                schema = "{}"
        rows.append(
            {
                "relative_path": relative,
                "absolute_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "format": path.suffix.lower().lstrip(".") or "binary",
                "row_or_object_count": count,
                "schema_or_shape_json": schema,
                "reopen_status": reopen,
            }
        )
    frame = pd.DataFrame(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


def _copy_blocked_partial_evidence(output: Path, destination: Path) -> list[str]:
    """Copy every small reviewable partial artifact and leave large bytes indexed."""

    copied: list[str] = []
    allowed = {".md", ".txt", ".json", ".csv", ".png", ".sha256", ".log"}
    roots = {"reports", "validation", "repository_audit", "manifests", "figures"}
    for path in sorted(value for value in output.rglob("*") if value.is_file()):
        relative = path.relative_to(output)
        if (
            relative.parts
            and relative.parts[0] in roots
            and path.suffix.lower() in allowed
            and path.stat().st_size <= MAX_MEMBER_BYTES
        ):
            _copy(path, destination / relative)
            copied.append(relative.as_posix())
    return copied


def _privacy_scan(root: Path) -> dict[str, object]:
    patterns = {
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
        "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    }
    text_suffixes = {
        ".md", ".txt", ".json", ".jsonl", ".csv", ".py", ".toml", ".yaml", ".yml",
        ".patch", ".diff", ".log", ".sql", ".ps1", ".sha256", ".rst",
        ".ini", ".cfg", ".lock", ".bat", ".cmd",
    }
    reviewed_binary_suffixes = {".png", ".pdf"}
    findings: list[dict[str, str]] = []
    scanned = 0
    forbidden_name_tokens = (
        "id_rsa", "id_ed25519", ".pem", ".pfx", ".key", "credentials",
        "browser_profile", "cookies",
    )
    unresolved = re.compile(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}")
    for candidate in sorted(root.rglob("*")):
        attributes = int(getattr(candidate.lstat(), "st_file_attributes", 0))
        if candidate.is_symlink() or attributes & 0x400:
            findings.append(
                {
                    "path": candidate.relative_to(root).as_posix(),
                    "pattern": "symlink_or_reparse_point",
                }
            )
            continue
        if not candidate.is_file():
            continue
        path = candidate
        relative = path.relative_to(root).as_posix()
        lowered_components = [value.lower() for value in Path(relative).parts]
        if any(
            token in component
            for component in lowered_components
            for token in forbidden_name_tokens
        ):
            findings.append({"path": relative, "pattern": "suspicious_filename"})
        if path.suffix.lower() in reviewed_binary_suffixes:
            continue
        if path.suffix.lower() not in text_suffixes:
            findings.append({"path": relative, "pattern": "unreviewed_binary_type"})
            continue
        scanned += 1
        try:
            text = path.read_bytes().decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            findings.append({"path": relative, "pattern": "invalid_utf8"})
            continue
        if any(
            (ord(character) < 32 and character not in "\t\r\n")
            or 0x7F <= ord(character) <= 0x9F
            for character in text
        ):
            findings.append({"path": relative, "pattern": "forbidden_control_character"})
        if relative.split("/", 1)[0] not in {"code_snapshot", "governing_docs"} and unresolved.search(text):
            findings.append({"path": relative, "pattern": "unresolved_template_token"})
        for name, pattern in patterns.items():
            if pattern.search(text):
                findings.append({"path": relative, "pattern": name})
    return {
        "schema_id": "partC.phase5.handoff.privacy_scan.v1",
        "text_files_scanned": scanned,
        "findings": findings,
        "status": "PASS" if not findings else "FAIL",
    }


def _internal_manifest(root: Path) -> pd.DataFrame:
    excluded = {
        "06_ARTIFACT_MANIFEST.csv",
        "CHECKSUMS.sha256",
    }
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "format": path.suffix.lower().lstrip(".") or "binary",
            }
        )
    return pd.DataFrame(rows)


def _write_internal_checksums(root: Path) -> list[str]:
    checksum = root / "CHECKSUMS.sha256"
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(value for value in root.rglob("*") if value.is_file() and value != checksum)
    ]
    write_text(checksum, "\n".join(rows))
    return rows


def _verify_extracted_checksums(root: Path) -> None:
    lines = (root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines()
    for line in lines:
        digest, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Fresh-extraction checksum failure: {relative}")


def _verify_extracted_manifest(root: Path) -> None:
    manifest = pd.read_csv(root / "06_ARTIFACT_MANIFEST.csv")
    required = {"relative_path", "size_bytes", "sha256", "format"}
    if len(manifest) == 0 or not required.issubset(manifest.columns):
        raise RuntimeError("Fresh-extraction handoff manifest is empty or incomplete")
    declared = set(manifest["relative_path"].astype(str))
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path
        not in {root / "06_ARTIFACT_MANIFEST.csv", root / "CHECKSUMS.sha256"}
    }
    if declared != actual or not manifest["relative_path"].is_unique:
        raise RuntimeError("Fresh-extraction handoff manifest member closure failed")
    for row in manifest.itertuples(index=False):
        path = root / str(row.relative_path)
        if (
            not path.is_file()
            or path.stat().st_size != int(row.size_bytes)
            or sha256_file(path) != str(row.sha256)
        ):
            raise RuntimeError(
                f"Fresh-extraction handoff manifest mismatch: {row.relative_path}"
            )


def _zip_write(source: Path, destination: Path) -> list[str]:
    members = sorted(value for value in source.rglob("*") if value.is_file())
    oversized = [
        path.relative_to(source).as_posix()
        for path in members
        if path.stat().st_size > MAX_MEMBER_BYTES
    ]
    if oversized:
        raise RuntimeError(f"Handoff members exceed 25 MiB: {oversized}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in members:
            archive.write(path, path.relative_to(source).as_posix())
    if destination.stat().st_size > TARGET_ZIP_BYTES:
        raise RuntimeError(
            "Compact handoff exceeds the unmodified 100 MiB target; split or obtain approval"
        )
    with zipfile.ZipFile(destination, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Compact handoff ZIP failed CRC validation")
        archived = sorted(archive.namelist())
    expected = sorted(path.relative_to(source).as_posix() for path in members)
    if archived != expected:
        raise RuntimeError("Compact handoff ZIP member closure failed")
    with tempfile.TemporaryDirectory(prefix="phase5_handoff_verify_") as temporary:
        extracted = Path(temporary) / "extracted"
        with zipfile.ZipFile(destination, "r") as archive:
            archive.extractall(extracted)
        _verify_extracted_checksums(extracted)
        _verify_extracted_manifest(extracted)
    return archived


def _build_blocked_handoff(
    *,
    execution: dict[str, Any],
    repo: Path,
    output: Path,
    branch: str,
    live_head: str,
    handoff_id: str,
    status: dict[str, Any],
    resource: dict[str, Any],
    git_state: dict[str, object],
    supersedes_handoff_id: str | None,
) -> Path:
    """Seal exact failure evidence without claiming success or rerunning science."""

    if (
        status.get("status") != BLOCKED_STATUS
        or status.get("ending_head") != live_head
        or not status.get("blockers")
        or resource.get("status") != "BLOCKED"
        or not (output / "validation" / "failure.json").is_file()
    ):
        raise RuntimeError("Blocked pilot evidence is not eligible for a blocked handoff")
    parent = _contained(
        repo,
        "partC_doc_icnn",
        "handoffs",
        "phase_05",
        "step_backbone_compilation_pilot",
    )
    bundle_parent = _contained(
        repo, "artifacts_icnn", "handoff_bundles", "phase_05"
    )
    final_source = _contained(parent, handoff_id)
    building_source = _contained(parent, f".{handoff_id}.building")
    zip_path = _contained(
        bundle_parent,
        f"phase05_stepbackbone_compilation_pilot_{handoff_id}_SYNC.zip",
    )
    building_zip = _contained(bundle_parent, f".{zip_path.name}.building")
    checksum_path = Path(str(zip_path) + ".sha256")
    companion_path = Path(str(zip_path) + ".companion.json")
    building_checksum = Path(str(checksum_path) + ".building")
    building_companion = Path(str(companion_path) + ".building")
    latest_path = parent / "LATEST_HANDOFF.txt"
    collisions = [
        value
        for value in (
            final_source,
            building_source,
            zip_path,
            building_zip,
            checksum_path,
            companion_path,
            building_checksum,
            building_companion,
        )
        if value.exists()
    ]
    if collisions:
        raise FileExistsError(f"Write-new blocked handoff target exists: {collisions}")
    building_source.mkdir(parents=True, exist_ok=False)
    source = building_source
    for directory in (
        "governing_docs",
        "configs",
        "code_snapshot",
        "code_diff",
        "summary_data",
        "figures",
        "geospatial_summary",
        "log_excerpts",
        "external_artifact_index",
        "internal_validation",
    ):
        (source / directory).mkdir(parents=True, exist_ok=False)
    authority_root = (
        repo
        / "partC_doc_icnn"
        / "review_gates"
        / "phase_05"
        / "backbone_compilation_pilot"
        / "phase5_pilot_authority_20260903"
    )
    for name in (
        "00_READ_FIRST.md",
        "PROJECT_STATUS.md",
        "12_DECISION_LOG.md",
        "20_HANDOFF_SYNC_PROTOCOL.md",
        "21_REVIEW_APPROVAL_GATES.md",
        "23_SYNC_BUNDLE_CONTENT_AND_SIZE_POLICY.md",
        "24_CODE_AND_DATA_SNAPSHOT_POLICY.md",
    ):
        _copy(repo / "partC_doc_icnn" / name, source / "governing_docs" / "repository" / name)
    _copy_tree_files(
        authority_root,
        source / "governing_docs" / "phase5_authority",
        ("*.md", "*.json", "*.csv", "*.sha256"),
    )
    _copy_tree_files(
        authority_root / "config", source / "configs" / "authority", ("*",)
    )
    _copy(Path(execution["execution_config_path"]) if execution.get("execution_config_path") else repo / "partC_doc_icnn" / "phase_5_work" / "phase5_backbone_compilation_pilot_execution_config.json", source / "configs" / "phase5_backbone_compilation_pilot_execution_config.json")
    for path in _code_snapshot_paths(repo):
        if path.is_file():
            _copy(path, source / "code_snapshot" / path.relative_to(repo))
    write_text(
        source / "code_diff" / "phase5_checkpoint.patch",
        git(repo, "diff", "--no-ext-diff", STARTING_HEAD, live_head).stdout,
    )
    write_text(
        source / "code_diff" / "diff_stat.txt",
        git(repo, "diff", "--stat", STARTING_HEAD, live_head).stdout,
    )
    write_text(
        source / "code_diff" / "git_log.txt",
        git(repo, "log", "--oneline", f"{STARTING_HEAD}..{live_head}").stdout,
    )
    copied_partial = _copy_blocked_partial_evidence(
        output, source / "summary_data" / "blocked_output"
    )
    _index_external_output(
        output,
        source / "external_artifact_index" / "blocked_output_inventory.csv",
    )
    for name in (
        "PHASE5_BACKBONE_COMPILATION_PILOT_RESULT.md",
        "PHASE5_BACKBONE_COMPILATION_PILOT_STATUS.json",
    ):
        path = repo / "partC_doc_icnn" / "phase_5_work" / name
        if path.is_file():
            _copy(path, source / "summary_data" / "repository_records" / name)
    write_text(
        source / "geospatial_summary" / "NO_GEOSPATIAL_OUTPUT.md",
        "# No geospatial output\n\nGeospatial work was outside this pilot scope.",
    )
    write_text(
        source / "figures" / "NO_COMPLETE_FIGURE_SET.md",
        "# No complete figure set\n\nThe pilot stopped at the exact blocker before success closure.",
    )
    write_text(
        source / "external_artifact_index" / "OUTPUT_ROOT.txt", str(output)
    )
    focused_log = output / "validation" / "focused_tests.txt"
    if focused_log.is_file():
        _copy(focused_log, source / "log_excerpts" / "focused_tests.txt")
    else:
        write_text(
            source / "log_excerpts" / "TESTS_NOT_REACHED.md",
            "# Focused tests not reached\n\nThe exact blocker occurred before a focused-test log was produced.",
        )
    write_json(
        source / "internal_validation" / "BLOCKED_PARTIAL_COPY.json",
        {
            "copied_file_count": len(copied_partial),
            "copied_relative_paths": copied_partial,
            "full_output_index": "external_artifact_index/blocked_output_inventory.csv",
            "status": "PASS_TOLERANT_PARTIAL_EVIDENCE",
        },
    )
    handoff_status = {
        "schema_id": "partC.phase5.handoff.status.v1",
        "handoff_id": handoff_id,
        "run_id": execution["run_id"],
        "status": "WAITING_FOR_EXTERNAL_REVIEW",
        "scientific_status": BLOCKED_STATUS,
        "branch": branch,
        "starting_head": STARTING_HEAD,
        "ending_head": live_head,
        "supersedes_handoff_id": supersedes_handoff_id,
        "blockers": status["blockers"],
        "next_step_authorized": False,
        "full_phase5_authorized": False,
        "final_reduced_topology_accepted": False,
        "final_runtime_thresholds_frozen": False,
        "phase6_authorized": False,
        "national_assignment_performed": False,
    }
    write_text(
        source / "00_HANDOFF_README.md",
        "# Upload this blocked Phase 5 pilot handoff\n\n"
        f"Upload `{zip_path.name}` for review of the exact blocker.",
    )
    write_text(
        source / "01_STEP_SUMMARY.md",
        "# Phase 5 backbone-compilation pilot blocked summary\n\n"
        "## Objective and exact inputs\n\n"
        f"The single authorized run `{execution['run_id']}` was the bounded s02-s06 "
        "backbone-compilation pilot. It used low-primary benchmark "
        f"`{execution['primary_benchmark']['benchmark_id']}`, held-out benchmark "
        f"`{execution['heldout_benchmark']['benchmark_id']}`, authority hashes "
        f"`{OUTER_AUTHORITY_SHA256}` / `{PARENT_AUTHORITY_SHA256}`, config "
        f"SHA-256 `{sha256_file(Path(execution.get('execution_config_path', repo / 'partC_doc_icnn' / 'phase_5_work' / 'phase5_backbone_compilation_pilot_execution_config.json')))}`, "
        f"and checkpoint HEAD `{live_head}` on `{branch}`.\n\n"
        "## Executed work and changes\n\n"
        "The configured focused tests and explicit pilot command were attempted once. "
        "Phase 5 source/tests/configuration and the exact checkpoint diff are included; "
        "accepted Phase 3/4 scientific artifacts remained read only. No rerun, assignment, "
        "training, network use, package installation, or Phase 6 work occurred.\n\n"
        "## Verified blocker and partial outputs\n\n"
        f"The run stopped with: `{status['blockers'][0]}`. A tolerant inventory indexed "
        f"every preserved output byte, and `{len(copied_partial)}` small reviewable partial "
        "artifacts were copied. This package makes no success or scientific-decision claim.\n\n"
        "## Validation and resources\n\n"
        f"Focused-test status is `{status.get('tests', {}).get('focused_phase5', 'INCOMPLETE')}`. "
        f"Elapsed time at the blocker was `{resource.get('elapsed_seconds_at_blocker', 'NA')}` s "
        f"and observed process-tree RSS was `{resource.get('process_tree_rss_bytes_at_blocker', 'NA')}` bytes. "
        "The traceback, blocker record, resource evidence, post-checkpoint receipt, partial "
        "artifact index, and code patch are included.\n\n"
        "## Inference and recommendation\n\n"
        "No scientific inference is advanced from incomplete evidence. Recommendation: BLOCK "
        "and externally review the exact failure before authorizing any repair or next step. "
        "Full Phase 5, final topology/runtime thresholds, Phase 6, and national assignment remain unauthorized.",
    )
    write_json(source / "02_HANDOFF_STATUS.json", handoff_status)
    write_text(
        source / "03_DECISIONS_AND_OPEN_QUESTIONS.md",
        "# Decisions and open questions\n\nNo scientific decision is advanced. The exact blocker requires external review.",
    )
    write_text(
        source / "04_VALIDATION_RESULTS.md",
        "# Validation results\n\nSuccess acceptance was not claimed. See `summary_data/blocked_output/validation/failure.json`.",
    )
    pd.DataFrame([resource]).to_csv(source / "05_RESOURCE_USAGE.csv", index=False)
    changes = git(repo, "diff", "--name-status", STARTING_HEAD, live_head).stdout.splitlines()
    pd.DataFrame(
        [
            {
                "status": line.split("\t", 1)[0],
                "repository_path": line.split("\t", 1)[1] if "\t" in line else "",
            }
            for line in changes
        ],
        columns=["status", "repository_path"],
    ).to_csv(source / "07_CODE_CHANGE_INDEX.csv", index=False)
    write_text(
        source / "08_COMMAND_LOG.md",
        "# Command log\n\n"
        f"- `{execution['python_executable']} -m pytest -q <focused Phase 5 test list>` "
        f"(status: `{status.get('tests', {}).get('focused_phase5', 'INCOMPLETE')}`; exact log or not-reached marker included).\n"
        f"- `{execution['python_executable']} -m partC_rev1_src.phase5.pilot_cli run "
        f"--config {execution.get('execution_config_path', 'phase5 execution config')}` (attempted exactly once; blocked).\n"
        f"- `git -C {repo} -c core.longpaths=true commit ...` (checkpoint bound to `{live_head}`).\n"
        "No scientific rerun, network use, package installation, assignment, or training occurred.",
    )
    write_text(
        source / "09_NEXT_STEP_RECOMMENDATION.md",
        "# Next-step recommendation\n\nReview the exact blocker. No next step is authorized.",
    )
    write_text(source / "10_UPLOAD_THIS_FILE.txt", str(zip_path))
    write_text(
        source / "11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md",
        (
            f"# Changelog\n\nThis blocked handoff supersedes `{supersedes_handoff_id}`; "
            "the exact repair delta is captured by the checkpoint patch and blocker evidence."
            if supersedes_handoff_id
            else "# Changelog\n\nInitial blocked handoff for this authorized pilot run; it supersedes no Phase 5 handoff."
        ),
    )
    write_json(source / "internal_validation" / "QUALIFIED_GIT_STATUS.json", git_state)
    privacy = _privacy_scan(source)
    write_json(source / "internal_validation" / "PRIVACY_SCAN.json", privacy)
    if privacy["status"] != "PASS":
        raise RuntimeError(f"Blocked handoff privacy scan failed: {privacy['findings']}")
    _internal_manifest(source).to_csv(
        source / "06_ARTIFACT_MANIFEST.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    _write_internal_checksums(source)
    members = _zip_write(source, building_zip)
    zip_sha = sha256_file(building_zip)
    write_text(building_checksum, f"{zip_sha}  {zip_path.name}")
    companion = {
        "schema_id": "partC.phase5.handoff.companion.v1",
        "created_utc": _utc_now(),
        "handoff_id": handoff_id,
        "source_folder": str(final_source),
        "zip_path": str(zip_path),
        "zip_size_bytes": building_zip.stat().st_size,
        "zip_sha256": zip_sha,
        "zip_member_count": len(members),
        "fresh_extraction_checksum_status": "PASS",
        "privacy_scan_status": privacy["status"],
        "branch": branch,
        "starting_head": STARTING_HEAD,
        "ending_head": live_head,
        "run_id": execution["run_id"],
        "scientific_status": BLOCKED_STATUS,
        "blockers": status["blockers"],
        "supersedes_handoff_id": supersedes_handoff_id,
        "outer_authority_sha256": OUTER_AUTHORITY_SHA256,
        "parent_authority_sha256": PARENT_AUTHORITY_SHA256,
        "external_output_root": str(output),
        "test_status": status.get("tests", {}).get(
            "focused_phase5", "INCOMPLETE"
        ),
        "resource_status": resource["status"],
        "qualified_git_status": git_state["status"],
        "next_step_authorized": False,
        "full_phase5_authorized": False,
        "final_reduced_topology_accepted": False,
        "final_runtime_thresholds_frozen": False,
        "phase6_authorized": False,
        "national_assignment_performed": False,
        "status": "WAITING_FOR_EXTERNAL_REVIEW",
    }
    write_json(building_companion, companion)
    final_source.parent.mkdir(parents=True, exist_ok=True)
    source.replace(final_source)
    building_zip.replace(zip_path)
    building_checksum.replace(checksum_path)
    building_companion.replace(companion_path)
    write_text(
        latest_path,
        f"source_folder={final_source}\nzip_path={zip_path}\nzip_sha256={zip_sha}\n"
        f"created_utc={companion['created_utc']}\nstatus=WAITING_FOR_EXTERNAL_REVIEW",
    )
    return zip_path


def build_handoff(
    config_path: str | Path,
    *,
    handoff_id: str,
    ending_head: str,
    supersedes_handoff_id: str | None = None,
) -> Path:
    handoff_id = _validated_handoff_id(handoff_id)
    supersedes_handoff_id = (
        _validated_handoff_id(supersedes_handoff_id)
        if supersedes_handoff_id is not None
        else None
    )
    if supersedes_handoff_id == handoff_id:
        raise ValueError("A handoff cannot supersede itself")
    config_path = Path(config_path).resolve()
    execution = json.loads(config_path.read_text(encoding="utf-8"))
    repo = Path(execution["project_root"]).resolve()
    output = Path(execution["output_root"]).resolve()
    branch = git(repo, "branch", "--show-current").stdout.strip()
    live_head = git(repo, "rev-parse", "HEAD").stdout.strip()
    if branch != execution["expected_branch"] or live_head != str(ending_head):
        raise RuntimeError("Handoff must bind the live authorized branch and exact ending HEAD")
    if git(repo, "merge-base", "--is-ancestor", STARTING_HEAD, live_head, check=False).returncode != 0:
        raise RuntimeError("Handoff ending HEAD does not descend from the pilot starting HEAD")
    git_state = _qualified_git_state(repo, live_head)
    _validate_supersession(repo, supersedes_handoff_id)
    authority_root = (
        repo / "partC_doc_icnn" / "review_gates" / "phase_05"
        / "backbone_compilation_pilot" / "phase5_pilot_authority_20260903"
    )
    status = json.loads((output / "reports" / "08_PILOT_STATUS.json").read_text(encoding="utf-8"))
    resource = json.loads((output / "validation" / "resource_usage.json").read_text(encoding="utf-8"))
    if status.get("status") not in {FINAL_STATUS, BLOCKED_STATUS}:
        raise RuntimeError(f"Unknown scientific status: {status.get('status')}")
    _verified_post_checkpoint_binding(
        output,
        branch=branch,
        live_head=live_head,
        scientific_status=str(status["status"]),
        status=status,
    )
    if status.get("status") == BLOCKED_STATUS:
        return _build_blocked_handoff(
            execution=execution,
            repo=repo,
            output=output,
            branch=branch,
            live_head=live_head,
            handoff_id=handoff_id,
            status=status,
            resource=resource,
            git_state=git_state,
            supersedes_handoff_id=supersedes_handoff_id,
        )
    output_validation = _validate_required_outputs(output, authority_root)
    acceptance = pd.read_csv(output / "validation" / "acceptance_test_results.csv")
    if (
        output_validation["status"] != "PASS"
        or status["status"] != FINAL_STATUS
        or status["ending_head"] != live_head
        or not acceptance["status"].eq("PASS").all()
        or resource["status"] != "PASS"
        or status.get("next_step_authorized") is not False
    ):
        raise RuntimeError("Post-checkpoint pilot evidence is not eligible for a success handoff")

    source_parent = _contained(
        repo,
        "partC_doc_icnn",
        "handoffs",
        "phase_05",
        "step_backbone_compilation_pilot",
    )
    bundle_parent = _contained(
        repo, "artifacts_icnn", "handoff_bundles", "phase_05"
    )
    final_source = _contained(source_parent, handoff_id)
    source = _contained(source_parent, f".{handoff_id}.building")
    zip_path = _contained(
        bundle_parent,
        f"phase05_stepbackbone_compilation_pilot_{handoff_id}_SYNC.zip",
    )
    building_zip = _contained(bundle_parent, f".{zip_path.name}.building")
    checksum_path = Path(str(zip_path) + ".sha256")
    companion_path = Path(str(zip_path) + ".companion.json")
    building_checksum = Path(str(checksum_path) + ".building")
    building_companion = Path(str(companion_path) + ".building")
    latest_path = source_parent / "LATEST_HANDOFF.txt"
    collisions = [
        value
        for value in (
            final_source,
            source,
            zip_path,
            building_zip,
            checksum_path,
            companion_path,
            building_checksum,
            building_companion,
        )
        if value.exists()
    ]
    if collisions:
        raise FileExistsError(f"Write-new handoff target exists: {collisions}")
    source.mkdir(parents=True, exist_ok=False)
    for directory in (
        "governing_docs", "configs", "code_snapshot", "code_diff", "summary_data",
        "figures", "geospatial_summary", "log_excerpts", "external_artifact_index",
        "internal_validation",
    ):
        (source / directory).mkdir(parents=True, exist_ok=False)

    reports = sorted((output / "reports").glob("*"))
    for path in reports:
        if path.is_file():
            _copy(path, source / "summary_data" / "reports" / path.name)
    for name in (
        "00_READ_FIRST.md", "PROJECT_STATUS.md", "12_DECISION_LOG.md",
        "20_HANDOFF_SYNC_PROTOCOL.md", "21_REVIEW_APPROVAL_GATES.md",
        "23_SYNC_BUNDLE_CONTENT_AND_SIZE_POLICY.md",
        "24_CODE_AND_DATA_SNAPSHOT_POLICY.md",
    ):
        path = repo / "partC_doc_icnn" / name
        _copy(path, source / "governing_docs" / "repository" / name)
    _copy_tree_files(authority_root, source / "governing_docs" / "phase5_authority", ("*.md", "*.json", "*.csv", "*.sha256"))
    _copy_tree_files(authority_root / "config", source / "configs" / "authority", ("*",))
    _copy(config_path, source / "configs" / config_path.name)

    for path in _code_snapshot_paths(repo):
        if path.is_file():
            _copy(path, source / "code_snapshot" / path.relative_to(repo))
    diff = git(repo, "diff", "--no-ext-diff", STARTING_HEAD, live_head).stdout
    write_text(source / "code_diff" / "phase5_checkpoint.patch", diff or "# No textual diff")
    write_text(source / "code_diff" / "diff_stat.txt", git(repo, "diff", "--stat", STARTING_HEAD, live_head).stdout)
    write_text(source / "code_diff" / "git_log.txt", git(repo, "log", "--oneline", f"{STARTING_HEAD}..{live_head}").stdout)

    _copy_tree_files(output, source / "summary_data" / "external_summaries", ("**/*.csv", "validation/*.json"))
    _copy(
        output / "runtime_fits" / "selected_runtime_registry.json",
        source
        / "summary_data"
        / "external_summaries"
        / "runtime_fits"
        / "selected_runtime_registry.json",
    )
    write_json(
        source / "internal_validation" / "EXTERNAL_OUTPUT_CLOSURE.json",
        output_validation,
    )
    _write_samples(output, source / "summary_data" / "representative_samples")
    _copy_tree_files(output / "figures", source / "figures", ("*.png", "source_data/*.csv"))
    write_text(source / "geospatial_summary" / "NO_GEOSPATIAL_OUTPUT.md", "# No geospatial output\n\nGeospatial output was outside this Phase 5 pilot scope.")
    _copy(output / "validation" / "focused_tests.txt", source / "log_excerpts" / "focused_tests.txt")
    _copy(output / "manifests" / "external_artifact_bindings.csv", source / "external_artifact_index" / "external_artifact_bindings.csv")
    _copy(output / "manifests" / "MANIFEST.csv", source / "external_artifact_index" / "full_output_MANIFEST.csv")
    _copy(output / "manifests" / "CHECKSUMS.sha256", source / "external_artifact_index" / "full_output_CHECKSUMS.sha256")

    result_doc = repo / "partC_doc_icnn" / "phase_5_work" / "PHASE5_BACKBONE_COMPILATION_PILOT_RESULT.md"
    status_doc = repo / "partC_doc_icnn" / "phase_5_work" / "PHASE5_BACKBONE_COMPILATION_PILOT_STATUS.json"
    for path in (result_doc, status_doc):
        if path.is_file():
            _copy(path, source / "summary_data" / "repository_records" / path.name)

    topology = pd.read_csv(output / "topology" / "topology_census.csv")
    fit_summary = pd.read_csv(output / "runtime_fits" / "runtime_fit_summary.csv")
    charge_summary = pd.read_csv(output / "charge_audit" / "fixed_charge_summary.csv")
    decision_evidence = pd.read_csv(
        output / "validation" / "pilot_decision_evidence.csv"
    )
    unresolved_decisions = decision_evidence.loc[
        decision_evidence["resolution_status"]
        .astype(str)
        .str.contains("UNRESOLVED|PROVISIONAL"),
        "decision_id",
    ].astype(str).tolist()
    headline = {
        "candidate_chain_count": int(topology["maximal_chain_count"].sum()),
        "sampled_macro_signatures": int(status["outputs"]["sampled_macro_signatures"]),
        "path_rows": int(status["outputs"]["path_rows"]),
        "acceptance_test_count": len(acceptance),
    }
    write_text(
        source / "00_HANDOFF_README.md",
        "# Upload this Phase 5 pilot handoff\n\n"
        f"Upload `{zip_path.name}` for external review. Status is `{FINAL_STATUS}`; "
        "full Phase 5, final topology/runtime thresholds, Phase 6, and national assignment remain unauthorized.",
    )
    write_text(
        source / "01_STEP_SUMMARY.md",
        "# Phase 5 backbone-compilation pilot summary\n\n"
        "## Objective and authorized scope\n\n"
        f"The single configured pilot `{execution['run_id']}` evaluated backbone contraction, "
        "exact offline bundle/series references, root-free candidate runtimes, charge semantics, "
        "existing accepted paths, and projected kernels for suites s02-s06. It did not run a "
        "new assignment, train a model, accept a final topology, freeze thresholds, or start Phase 6.\n\n"
        "## Exact inputs and code\n\n"
        f"Execution used config `{config_path}` (SHA-256 `{sha256_file(config_path)}`), low-primary "
        f"benchmark `{execution['primary_benchmark']['benchmark_id']}` at iteration "
        f"`{execution['primary_benchmark']['state_iteration']}`, and held-out benchmark "
        f"`{execution['heldout_benchmark']['benchmark_id']}` at iteration "
        f"`{execution['heldout_benchmark']['state_iteration']}`. The outer/parent authority hashes "
        f"are `{OUTER_AUTHORITY_SHA256}` / `{PARENT_AUTHORITY_SHA256}`. Code is bound to branch "
        f"`{branch}`, starting HEAD `{STARTING_HEAD}`, and checkpoint HEAD `{live_head}`; the "
        "repository-relative snapshot, tests, config, and exact patch are included.\n\n"
        "## Commands and changes\n\n"
        f"The configured Python `{execution['python_executable']}` ran the focused Phase 5 pytest "
        "set, the explicit `pilot_cli run` entrypoint once, the checkpoint commit, metadata "
        "reseal, and this handoff builder. Phase 5 source, tests, execution records, reports, "
        "manifests, and governance status were added or updated; accepted Phase 3/4 inputs were read only.\n\n"
        "## Verified facts and outputs\n\n"
        f"Headline counts are `{canonical_json(headline)}`. All `{len(acceptance)}` declared hard "
        "gates pass, focused tests pass, accepted inputs are invariant, and the external output "
        "manifest/checksum/reopen closure passes. Detailed metrics and all twelve decision records "
        "are copied under `summary_data/`.\n\n"
        "## Resources\n\n"
        f"Observed scientific wall time was `{resource.get('scientific_elapsed_seconds', 'NA')}` s, "
        f"peak process-tree RSS was `{resource.get('process_tree_peak_observed_rss_bytes', 'NA')}` "
        f"bytes, and the sealed-output byte measure was "
        f"`{resource.get('added_output_bytes_before_post_commit_manifest', resource.get('added_output_bytes_before_final_manifest', 'NA'))}`. "
        "No network use or package installation occurred.\n\n"
        "## Inference, unresolved questions, and recommendation\n\n"
        f"The pilot evidence supports bounded recommendations but leaves `{', '.join(unresolved_decisions)}` "
        "explicitly unresolved/provisional. These are inferences for external review, not frozen "
        "decisions. Recommendation: review this package and issue an explicit proceed, repair, or "
        "block decision. No next step is authorized.",
    )
    handoff_status = {
        "schema_id": "partC.phase5.handoff.status.v1",
        "handoff_id": handoff_id,
        "run_id": execution["run_id"],
        "status": "WAITING_FOR_EXTERNAL_REVIEW",
        "scientific_status": FINAL_STATUS,
        "branch": branch,
        "starting_head": STARTING_HEAD,
        "ending_head": live_head,
        "supersedes_handoff_id": supersedes_handoff_id,
        "next_step_authorized": False,
        "full_phase5_authorized": False,
        "final_reduced_topology_accepted": False,
        "final_runtime_thresholds_frozen": False,
        "phase6_authorized": False,
        "national_assignment_performed": False,
        "blockers": [],
    }
    write_json(source / "02_HANDOFF_STATUS.json", handoff_status)
    _copy(output / "reports" / "07_DECISIONS_RECOMMENDED_FOR_FULL_PHASE5.md", source / "03_DECISIONS_AND_OPEN_QUESTIONS.md")
    write_text(
        source / "04_VALIDATION_RESULTS.md",
        "# Validation results\n\n"
        f"All {len(acceptance)} dynamic hard-gate rows pass. Required-output validation, "
        "status-schema validation, input pre/post invariance, source reopen, figure reopen, "
        "path component closure, and exact root-call boundaries pass. See `summary_data/" 
        "external_summaries/validation/` and `external_artifact_index/` for machine-readable evidence.",
    )
    pd.DataFrame([resource]).to_csv(source / "05_RESOURCE_USAGE.csv", index=False)
    changes = git(repo, "diff", "--name-status", STARTING_HEAD, live_head).stdout.splitlines()
    change_rows = []
    for line in changes:
        parts = line.split("\t", 1)
        change_rows.append({"status": parts[0], "repository_path": parts[1] if len(parts) > 1 else ""})
    pd.DataFrame(change_rows, columns=["status", "repository_path"]).to_csv(source / "07_CODE_CHANGE_INDEX.csv", index=False)
    write_text(
        source / "08_COMMAND_LOG.md",
        "# Command log\n\n"
        f"- `{execution['python_executable']} -m pytest -q tests/test_phase5_*.py` (focused set; exact command in log excerpt)\n"
        f"- `{execution['python_executable']} -m partC_rev1_src.phase5.pilot_cli run --config {config_path}`\n"
        f"- `git -C {repo} -c core.longpaths=true commit ...`\n"
        f"- `{execution['python_executable']} -m partC_rev1_src.phase5.pilot_cli reseal --config {config_path} --ending-head {live_head}`\n"
        "No network, package installation, national assignment, or scientific rerun occurred.",
    )
    write_text(
        source / "09_NEXT_STEP_RECOMMENDATION.md",
        "# Next-step recommendation\n\nRecommendation: review the pilot and issue an explicit approval, repair, or block decision. "
        "No next step is authorized by this handoff. Full Phase 5 and Phase 6 remain false.",
    )
    write_text(source / "10_UPLOAD_THIS_FILE.txt", str(zip_path.resolve()))
    write_text(
        source / "11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md",
        (
            f"# Changelog\n\nThis handoff supersedes `{supersedes_handoff_id}`. "
            "The exact code and evidence delta is captured by the checkpoint patch and artifact manifests."
            if supersedes_handoff_id
            else "# Changelog\n\nInitial handoff for the authorized Phase 5 backbone-compilation pilot; it supersedes no Phase 5 handoff."
        ),
    )
    write_json(
        source / "internal_validation" / "QUALIFIED_GIT_STATUS.json", git_state
    )

    privacy = _privacy_scan(source)
    write_json(source / "internal_validation" / "PRIVACY_SCAN.json", privacy)
    if privacy["status"] != "PASS":
        raise RuntimeError(f"Handoff privacy scan failed: {privacy['findings']}")
    manifest = _internal_manifest(source)
    manifest.to_csv(source / "06_ARTIFACT_MANIFEST.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    _write_internal_checksums(source)
    members = _zip_write(source, building_zip)
    zip_sha = sha256_file(building_zip)
    write_text(building_checksum, f"{zip_sha}  {zip_path.name}")
    companion = {
        "schema_id": "partC.phase5.handoff.companion.v1",
        "created_utc": _utc_now(),
        "handoff_id": handoff_id,
        "source_folder": str(final_source.resolve()),
        "zip_path": str(zip_path.resolve()),
        "zip_size_bytes": building_zip.stat().st_size,
        "zip_sha256": zip_sha,
        "target_zip_limit_bytes": TARGET_ZIP_BYTES,
        "hard_zip_limit_bytes": HARD_ZIP_BYTES,
        "zip_member_count": len(members),
        "fresh_extraction_checksum_status": "PASS",
        "privacy_scan_status": privacy["status"],
        "branch": branch,
        "starting_head": STARTING_HEAD,
        "ending_head": live_head,
        "run_id": execution["run_id"],
        "scientific_status": FINAL_STATUS,
        "supersedes_handoff_id": supersedes_handoff_id,
        "outer_authority_sha256": OUTER_AUTHORITY_SHA256,
        "parent_authority_sha256": PARENT_AUTHORITY_SHA256,
        "external_output_root": str(output),
        "external_output_manifest_sha256": sha256_file(output / "manifests" / "MANIFEST.csv"),
        "external_output_manifest_rows": len(pd.read_csv(output / "manifests" / "MANIFEST.csv")),
        "test_status": "PASS",
        "resource_status": resource["status"],
        "qualified_git_status": git_state["status"],
        "next_step_authorized": False,
        "full_phase5_authorized": False,
        "final_reduced_topology_accepted": False,
        "final_runtime_thresholds_frozen": False,
        "phase6_authorized": False,
        "national_assignment_performed": False,
        "status": "WAITING_FOR_EXTERNAL_REVIEW",
    }
    write_json(building_companion, companion)
    source.replace(final_source)
    building_zip.replace(zip_path)
    building_checksum.replace(checksum_path)
    building_companion.replace(companion_path)
    write_text(
        latest_path,
        f"source_folder={final_source.resolve()}\nzip_path={zip_path.resolve()}\n"
        f"zip_sha256={zip_sha}\ncreated_utc={companion['created_utc']}\n"
        "status=WAITING_FOR_EXTERNAL_REVIEW",
    )
    return zip_path
