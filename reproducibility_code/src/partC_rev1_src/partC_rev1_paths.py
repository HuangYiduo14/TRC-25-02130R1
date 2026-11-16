"""Path helpers for partC_rev1."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .partC_rev1_constants import PARTC_REV1_PREFIX


def partC_rev1_resolve_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    """Resolve a config path relative to *base_dir* or the current working directory."""

    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    root = Path.cwd() if base_dir is None else Path(base_dir).expanduser()
    return (root / raw).resolve()


def partC_rev1_ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def partC_rev1_assert_prefixed_path_name(path: str | Path, *, label: str) -> None:
    """Require created artifact files/folders to use the partC_rev1_ prefix."""

    name = Path(path).name
    if not name.startswith(PARTC_REV1_PREFIX):
        raise ValueError(f"{label} must start with {PARTC_REV1_PREFIX!r}: {path}")


def partC_rev1_ensure_configured_folders(config: Mapping[str, object]) -> dict[str, Path]:
    project = config.get("project", {})
    outputs = config.get("outputs", {})
    if not isinstance(project, Mapping):
        raise ValueError("config['project'] must be a mapping")
    if not isinstance(outputs, Mapping):
        raise ValueError("config['outputs'] must be a mapping")

    keys = {
        "source_code_dir": project.get("source_code_dir", "partC_rev1_src"),
        "tests_dir": project.get("tests_dir", "partC_rev1_tests"),
        "config_dir": project.get("config_dir", "partC_rev1_configs"),
        "cache_dir": project.get("cache_dir", "partC_rev1_cache"),
        "output_dir": outputs.get("output_dir", project.get("output_dir", "partC_rev1_outputs")),
        "docs_dir": outputs.get("docs_dir", project.get("docs_dir", "partC_rev1_docs")),
    }

    resolved: dict[str, Path] = {}
    for key, value in keys.items():
        if value is None or str(value).strip() == "":
            raise ValueError(f"Configured folder {key!r} is empty")
        partC_rev1_assert_prefixed_path_name(str(value), label=key)
        resolved[key] = partC_rev1_ensure_directory(partC_rev1_resolve_path(str(value)))
    return resolved


def partC_rev1_output_file(config: Mapping[str, object], filename: str) -> Path:
    outputs = config.get("outputs", {})
    if not isinstance(outputs, Mapping):
        raise ValueError("config['outputs'] must be a mapping")
    project = config.get("project", {})
    docs_default = "partC_rev1_docs"
    if isinstance(project, Mapping):
        docs_default = str(project.get("docs_dir", docs_default))
    if Path(filename).suffix.lower() == ".md":
        output_dir = outputs.get("docs_dir", docs_default)
        partC_rev1_assert_prefixed_path_name(str(output_dir), label="outputs.docs_dir")
    else:
        output_dir = outputs.get("output_dir", "partC_rev1_outputs")
        partC_rev1_assert_prefixed_path_name(str(output_dir), label="outputs.output_dir")
    partC_rev1_assert_prefixed_path_name(filename, label="output filename")
    return partC_rev1_ensure_directory(partC_rev1_resolve_path(str(output_dir))) / filename
