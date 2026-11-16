"""Artifact registry helpers for Prompt 4 immutable run folders."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from .partC_rev1_hashing import partC_rev1_file_sha256
from .partC_rev1_io import partC_rev1_write_json
from .partC_rev1_paths import partC_rev1_ensure_directory, partC_rev1_resolve_path


def _partC_rev1_package_versions() -> dict[str, str | None]:
    packages = ["numpy", "scipy", "pandas", "pyarrow", "networkx", "igraph", "geopandas", "psutil"]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _partC_rev1_git_metadata() -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status or "",
    }


def partC_rev1_environment_metadata() -> dict[str, Any]:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _partC_rev1_package_versions(),
        "git": _partC_rev1_git_metadata(),
    }


@dataclass(slots=True)
class PartCRev1ArtifactRegistry:
    """Collect and write a run manifest for a prefixed artifact folder."""

    base_dir: str | Path
    stage: str
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(init=False)
    inputs: list[dict[str, Any]] = field(init=False)
    dependencies: list[dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        if not self.stage.startswith("partC_rev1_"):
            raise ValueError(f"stage must start with 'partC_rev1_': {self.stage}")
        self.base_dir = partC_rev1_ensure_directory(partC_rev1_resolve_path(self.base_dir))
        if not Path(self.base_dir).name.startswith("partC_rev1_"):
            raise ValueError(f"artifact folder must start with 'partC_rev1_': {self.base_dir}")
        self.artifacts: list[dict[str, Any]] = []
        self.inputs: list[dict[str, Any]] = []
        self.dependencies: list[dict[str, Any]] = []

    def add_input(self, *, label: str, path: str | Path, description: str = "", sha256: str | None = None) -> None:
        resolved = partC_rev1_resolve_path(path)
        digest = sha256
        if digest is None and resolved.exists() and resolved.is_file() and resolved.stat().st_size <= 20 * 1024 * 1024:
            digest = partC_rev1_file_sha256(resolved)
        self.inputs.append(
            {
                "label": label,
                "path": str(resolved),
                "exists": resolved.exists(),
                "sha256": digest,
                "description": description,
            }
        )

    def add_artifact(
        self,
        *,
        label: str,
        path: str | Path,
        kind: str,
        description: str = "",
        graph_hash: str | None = None,
        od_hash: str | None = None,
        partition_hash: str | None = None,
        config_hash: str | None = None,
        completion_state: str = "complete",
        sha256: str | None = None,
    ) -> None:
        resolved = partC_rev1_resolve_path(path)
        if not resolved.name.startswith("partC_rev1_"):
            raise ValueError(f"artifact filename must start with 'partC_rev1_': {resolved}")
        digest = sha256
        if digest is None and resolved.exists() and resolved.is_file() and resolved.stat().st_size <= 20 * 1024 * 1024:
            digest = partC_rev1_file_sha256(resolved)
        self.artifacts.append(
            {
                "label": label,
                "path": str(resolved),
                "kind": kind,
                "exists": resolved.exists(),
                "size_bytes": resolved.stat().st_size if resolved.exists() and resolved.is_file() else None,
                "sha256": digest,
                "graph_hash": graph_hash,
                "od_hash": od_hash,
                "partition_hash": partition_hash,
                "config_hash": config_hash,
                "completion_state": completion_state,
                "description": description,
            }
        )

    def add_dependency(self, *, label: str, artifact_label: str, required_state: str = "complete") -> None:
        self.dependencies.append({"label": label, "artifact_label": artifact_label, "required_state": required_state})

    def validate_dependencies(self) -> list[str]:
        by_label = {artifact["label"]: artifact for artifact in self.artifacts}
        errors: list[str] = []
        for dependency in self.dependencies:
            artifact = by_label.get(dependency["artifact_label"])
            if artifact is None:
                errors.append(f"Missing dependency artifact: {dependency['artifact_label']}")
                continue
            if not artifact.get("exists"):
                errors.append(f"Dependency does not exist: {dependency['artifact_label']}")
            if artifact.get("completion_state") != dependency.get("required_state", "complete"):
                errors.append(
                    f"Dependency state mismatch for {dependency['artifact_label']}: "
                    f"{artifact.get('completion_state')} != {dependency.get('required_state', 'complete')}"
                )
        return errors

    def manifest(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "metadata": self.metadata,
            "environment": partC_rev1_environment_metadata(),
            "inputs": self.inputs,
            "artifacts": self.artifacts,
            "dependencies": self.dependencies,
            "dependency_errors": self.validate_dependencies(),
        }

    def write(self, path: str | Path) -> Path:
        manifest_path = partC_rev1_resolve_path(path)
        if not manifest_path.name.startswith("partC_rev1_"):
            raise ValueError(f"manifest filename must start with 'partC_rev1_': {manifest_path}")
        partC_rev1_write_json(self.manifest(), manifest_path)
        return manifest_path
