"""Compact manifest, checksum, text-policy, and fresh-extraction guards."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .partC_rev1_handoff_text import partC_rev1_validate_handoff_text_tree


_TEXT_SUFFIXES = {".md", ".txt", ".json", ".csv", ".py", ".patch", ".sha256", ".toml", ".yaml", ".yml"}
_PROHIBITED_BINARY_SUFFIXES = {
    ".sqlite",
    ".sqlite3",
    ".gpkg",
    ".npz",
    ".npy",
    ".parquet",
    ".feather",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
}
_ALLOWED_BINARY_SUFFIXES = {".png", ".pdf"}
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
)
_MAX_INCLUDED_FILE_BYTES = 25 * 1024 * 1024
_HARD_ZIP_BYTES = 200 * 1024 * 1024


class PartCRev1HandoffError(RuntimeError):
    """Raised when a compact handoff is unsafe, incomplete, or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> PurePosixPath:
    pure = PurePosixPath(value)
    if (
        not pure.parts
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in value
        or any(":" in part for part in pure.parts)
    ):
        raise PartCRev1HandoffError(f"Unsafe handoff relative path: {value!r}")
    return pure


def _all_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise PartCRev1HandoffError(f"Handoff source contains a symlink/reparse entry: {item}")
        if item.is_file():
            files.append(item)
    return files


def _policy_for(relative: str) -> dict[str, Any]:
    if relative == "08_COMMAND_LOG.md" or relative.endswith("COMMAND_LOG.md"):
        policy_class = "command_log"
    elif relative.startswith("code_diff/") or relative.endswith(".patch"):
        policy_class = "diff_or_patch"
    elif relative.startswith("code_snapshot/") or relative.startswith("configs/"):
        policy_class = "code_snapshot"
    elif re.match(r"^(?:0[0-9]|1[0-1])_", relative):
        policy_class = "template_control"
    else:
        policy_class = "plain_document"
    return {
        "relative_path": relative,
        "policy_class": policy_class,
        "encoding": "utf-8",
        "template_rendered": policy_class == "template_control",
        "forbid_powershell_variables": policy_class == "template_control" and relative != "08_COMMAND_LOG.md",
    }


def _scan_text_and_binary(root: Path, files: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    classifications: list[dict[str, Any]] = []
    class_by_path: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        size = path.stat().st_size
        if size > _MAX_INCLUDED_FILE_BYTES:
            raise PartCRev1HandoffError(f"Included handoff file exceeds 25 MiB: {relative}")
        if suffix in _PROHIBITED_BINARY_SUFFIXES:
            raise PartCRev1HandoffError(f"Prohibited data/checkpoint binary in compact handoff: {relative}")
        if suffix in _TEXT_SUFFIXES or path.name in {"Dockerfile", "LICENSE"}:
            policy = _policy_for(relative)
            classifications.append(policy)
            class_by_path[relative] = str(policy["policy_class"])
            try:
                text = path.read_text(encoding="utf-8", errors="strict")
            except (OSError, UnicodeError) as exc:
                raise PartCRev1HandoffError(f"Invalid UTF-8 handoff text: {relative}") from exc
            for pattern in _SECRET_PATTERNS:
                if pattern.search(text):
                    raise PartCRev1HandoffError(f"Potential secret found in handoff payload: {relative}")
        elif suffix in _ALLOWED_BINARY_SUFFIXES:
            class_by_path[relative] = "binary_included"
        else:
            raise PartCRev1HandoffError(f"Unclassified handoff file type: {relative}")
    if classifications:
        partC_rev1_validate_handoff_text_tree(
            root, {"schema_version": "1.0", "files": classifications}
        )
    return classifications, class_by_path


def partC_rev1_build_manifest(
    source_root: str | Path,
    *,
    provenance: str = "phase1_completion_campaign",
) -> list[dict[str, Any]]:
    """Build a deterministic compact-file manifest after text/secret/binary scans."""

    root = Path(source_root).resolve(strict=True)
    if not root.is_dir():
        raise PartCRev1HandoffError("Handoff source root must be a directory")
    files = _all_files(root)
    if not files:
        raise PartCRev1HandoffError("Handoff source may not be empty")
    _, class_by_path = _scan_text_and_binary(root, files)
    return [
        {
            "role": "payload",
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
            "provenance": str(provenance),
            "classification": class_by_path[path.relative_to(root).as_posix()],
            "source_status": "PRESENT",
            "extracted_status": "PENDING",
        }
        for path in files
    ]


def partC_rev1_write_checksums(
    destination: str | Path, manifest: Sequence[Mapping[str, Any]]
) -> Path:
    """Write stable SHA-256 lines for non-self-control manifest rows."""

    path = Path(destination).resolve(strict=False)
    if not path.parent.is_dir() or path.exists():
        raise PartCRev1HandoffError("Checksum destination parent must exist and destination must be new")
    rows = sorted(manifest, key=lambda item: str(item["relative_path"]))
    if not rows:
        raise PartCRev1HandoffError("Cannot write checksums for an empty manifest")
    lines: list[str] = []
    for row in rows:
        relative = str(row["relative_path"])
        _safe_relative(relative)
        digest = str(row["sha256"])
        if digest in {"SELF", "SELF_REFERENTIAL_CONTROL"}:
            continue
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise PartCRev1HandoffError(f"Invalid manifest SHA-256 for {relative}")
        lines.append(f"{digest}  {relative}")
    encoded = ("\n".join(lines) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def partC_rev1_verify_member_closure(
    root: str | Path,
    manifest: Sequence[Mapping[str, Any]],
    *,
    control_paths: Sequence[str] = ("CHECKSUMS.sha256", "MANIFEST.csv"),
) -> list[dict[str, Any]]:
    """Require exact path closure, allowing only named self-control files."""

    base = Path(root).resolve(strict=True)
    rows = list(manifest)
    if not rows:
        raise PartCRev1HandoffError("Manifest may not be empty")
    declared = [str(row["relative_path"]) for row in rows]
    if len(declared) != len(set(declared)):
        raise PartCRev1HandoffError("Manifest contains duplicate paths")
    for value in declared:
        _safe_relative(value)
    actual = {path.relative_to(base).as_posix() for path in _all_files(base)}
    allowed_controls = {str(_safe_relative(value)) for value in control_paths}
    missing = set(declared) - actual
    extra = actual - set(declared) - allowed_controls
    if missing or extra:
        raise PartCRev1HandoffError(
            f"Handoff member closure mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    results: list[dict[str, Any]] = []
    for row in rows:
        relative = str(row["relative_path"])
        path = base / Path(*PurePosixPath(relative).parts)
        if int(row["size_bytes"]) != path.stat().st_size or str(row["sha256"]) != _sha256(path):
            raise PartCRev1HandoffError(f"Handoff manifest size/hash mismatch: {relative}")
        results.append({"relative_path": relative, "status": "PASS"})
    return results


def _verify_checksum_file(root: Path, name: str = "CHECKSUMS.sha256") -> int:
    path = root / name
    if not path.is_file():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise PartCRev1HandoffError("Malformed CHECKSUMS.sha256 line")
        relative = str(_safe_relative(match.group(2)))
        target = root / Path(*PurePosixPath(relative).parts)
        if not target.is_file() or _sha256(target) != match.group(1):
            raise PartCRev1HandoffError(f"Checksum mismatch after extraction: {relative}")
        count += 1
    if count == 0:
        raise PartCRev1HandoffError("CHECKSUMS.sha256 is empty")
    return count


def _validate_zip_members(archive: zipfile.ZipFile) -> list[str]:
    names = [item.filename for item in archive.infolist() if not item.is_dir()]
    if not names or len(names) != len(set(names)):
        raise PartCRev1HandoffError("ZIP must contain unique nonempty file members")
    for info in archive.infolist():
        _safe_relative(info.filename.rstrip("/"))
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise PartCRev1HandoffError(f"ZIP symlink member is prohibited: {info.filename}")
        if info.file_size > _MAX_INCLUDED_FILE_BYTES:
            raise PartCRev1HandoffError(f"ZIP member exceeds 25 MiB: {info.filename}")
    if archive.testzip() is not None:
        raise PartCRev1HandoffError("ZIP CRC test failed")
    return names


def partC_rev1_verify_zip_and_fresh_extraction(
    zip_path: str | Path,
    manifest: Sequence[Mapping[str, Any]],
    temp_root: str | Path,
    *,
    classification_manifest: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify safe ZIP members and the independent freshly extracted tree."""

    archive_path = Path(zip_path).resolve(strict=True)
    if archive_path.stat().st_size > _HARD_ZIP_BYTES:
        raise PartCRev1HandoffError("Compact handoff ZIP exceeds the 200 MiB hard limit")
    temporary = Path(temp_root).resolve(strict=True)
    extraction = temporary / f"{archive_path.stem}_fresh_{os.getpid()}"
    if extraction.exists():
        raise PartCRev1HandoffError(f"Fresh extraction target already exists: {extraction}")
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = _validate_zip_members(archive)
        extraction.mkdir(parents=False, exist_ok=False)
        for info in archive.infolist():
            pure = _safe_relative(info.filename.rstrip("/"))
            destination = extraction / Path(*pure.parts)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, destination.open("xb") as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
    closure = partC_rev1_verify_member_closure(extraction, manifest)
    checksum_count = _verify_checksum_file(extraction)
    if classification_manifest is None:
        files = _all_files(extraction)
        classifications, _ = _scan_text_and_binary(extraction, files)
        text_results = partC_rev1_validate_handoff_text_tree(
            extraction, {"schema_version": "1.0", "files": classifications}
        ) if classifications else []
    else:
        text_results = partC_rev1_validate_handoff_text_tree(extraction, classification_manifest)
    return {
        "status": "PASS",
        "zip_path": archive_path.as_posix(),
        "zip_size_bytes": archive_path.stat().st_size,
        "zip_sha256": _sha256(archive_path),
        "member_count": len(names),
        "manifest_verified_count": len(closure),
        "checksum_verified_count": checksum_count,
        "text_verified_count": len(text_results),
        "fresh_extraction_root": extraction.as_posix(),
    }
