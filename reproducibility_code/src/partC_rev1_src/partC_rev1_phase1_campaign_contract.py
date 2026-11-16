"""Fail-closed execution contract for the approved Phase 1 campaign."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import jsonschema


_APPROVAL_RELATIVE = Path(
    "partC_doc_icnn/review_gates/phase_01/completion_campaign/"
    "phase1_completion_preflight_20260812T041155Z/APPROVAL.json"
)
_BINDINGS_RELATIVE = Path(
    "partC_doc_icnn/phase_1_work/completion_campaign/preflight/"
    "phase1_completion_campaign_20260812_v1/PHASE1_COMPLETION_APPROVAL_BINDINGS.json"
)
_RUN_MATRIX_RELATIVE = Path(
    "partC_doc_icnn/phase_1_work/completion_campaign/preflight/"
    "phase1_completion_campaign_20260812_v1/PHASE1_COMPLETION_RUN_MATRIX.json"
)
_RESOURCE_POLICY_RELATIVE = Path(
    "partC_doc_icnn/phase_1_work/completion_campaign/preflight/"
    "phase1_completion_campaign_20260812_v1/PHASE1_COMPLETION_RESOURCE_POLICY.json"
)
_CONFIG_SCHEMA_RELATIVE = Path(
    "partC_doc_icnn/phase_1_work/completion_campaign/schemas/phase1_campaign_config.schema.json"
)
_RUN_MATRIX_SCHEMA_RELATIVE = Path(
    "partC_doc_icnn/phase_1_work/completion_campaign/schemas/phase1_campaign_run_matrix.schema.json"
)
_REGISTRY_RELATIVE = Path("partC_rev1_cache/phase1_local/phase1_data_roots.local.json")
_MANIFEST_RELATIVE = Path("partC_doc_icnn/phase_1_work/manifests/phase1_authoritative_inputs.v1.json")


class PartCRev1CampaignContractError(ValueError):
    """Base error for an incomplete, implicit, or mismatched campaign request."""


class PartCRev1ApprovalBindingError(PartCRev1CampaignContractError):
    """Raised when an approval-bound value differs from the reviewed value."""


class PartCRev1CampaignPathError(PartCRev1CampaignContractError):
    """Raised before an unsafe or unapproved destination can be created."""


class PartCRev1NoOpError(PartCRev1CampaignContractError):
    """Raised when a task list or meaningful output set is empty."""


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PartCRev1CampaignContractError(f"Duplicate JSON key is prohibited: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PartCRev1CampaignContractError(f"Cannot read strict UTF-8 JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PartCRev1CampaignContractError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_schema(payload: Mapping[str, Any], schema_path: Path, label: str) -> None:
    schema = _strict_json(schema_path.resolve(strict=True))
    try:
        validator = jsonschema.Draft202012Validator(schema)
        validator.check_schema(schema)
        errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.absolute_path))
    except jsonschema.SchemaError as exc:
        raise PartCRev1CampaignContractError(f"Invalid {label} schema: {exc.message}") from exc
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.absolute_path) or "$"
        raise PartCRev1CampaignContractError(f"{label} schema validation failed at {location}: {first.message}")


def _repository_from_config(config: Mapping[str, Any]) -> Path:
    value = config.get("repository_root")
    if not isinstance(value, str) or not value.strip():
        raise PartCRev1CampaignContractError("repository_root must be explicit and nonblank")
    try:
        return Path(value).resolve(strict=True)
    except OSError as exc:
        raise PartCRev1CampaignContractError(f"repository_root does not resolve: {value}") from exc


def partC_rev1_load_campaign_config(
    path: str | Path, schema_path: str | Path | None = None
) -> dict[str, Any]:
    """Load the explicit campaign config and reject missing/default execution."""

    config_path = Path(path).resolve(strict=True)
    payload = _strict_json(config_path)
    repository = _repository_from_config(payload)
    schema = repository / _CONFIG_SCHEMA_RELATIVE if schema_path is None else Path(schema_path)
    _validate_schema(payload, schema.resolve(strict=True), "campaign config")
    if not payload.get("step_ids") or not payload.get("run_id") or not payload.get("campaign_id"):
        raise PartCRev1CampaignContractError("Campaign, run, and nonempty step IDs are mandatory")
    execution = payload.get("execution", {})
    if execution.get("phase_2_authorized") or execution.get("scientific_real_data_execution_authorized"):
        raise PartCRev1ApprovalBindingError("Frozen Phase 1 config may not authorize Phase 2 or real scientific execution")
    if execution.get("network_enabled") or execution.get("package_installation_enabled") or execution.get("environment_repair_enabled"):
        raise PartCRev1ApprovalBindingError("Network/package/environment mutation is not approved")
    return payload


def partC_rev1_load_run_matrix(
    path: str | Path, schema_path: str | Path | None = None, *, repository_root: str | Path | None = None
) -> dict[str, Any]:
    """Load a versioned run matrix with a nonempty, unique mandatory task list."""

    matrix_path = Path(path).resolve(strict=True)
    payload = _strict_json(matrix_path)
    if repository_root is None:
        candidate = Path(__file__).resolve().parents[1]
    else:
        candidate = Path(repository_root).resolve(strict=True)
    schema = candidate / _RUN_MATRIX_SCHEMA_RELATIVE if schema_path is None else Path(schema_path)
    _validate_schema(payload, schema.resolve(strict=True), "campaign run matrix")
    partC_rev1_require_nonempty_tasks(payload)
    all_tasks = list(payload["mandatory"]) + list(payload.get("optional", []))
    task_ids = [str(item["task_id"]) for item in all_tasks]
    if len(task_ids) != len(set(task_ids)):
        raise PartCRev1NoOpError("Campaign run matrix task IDs must be globally unique")
    return payload


def partC_rev1_require_nonempty_tasks(
    run_matrix: Mapping[str, Any], task_ids: Sequence[str] | None = None
) -> tuple[Mapping[str, Any], ...]:
    mandatory = run_matrix.get("mandatory")
    if not isinstance(mandatory, Sequence) or isinstance(mandatory, (str, bytes)) or not mandatory:
        raise PartCRev1NoOpError("Mandatory task list must be nonempty")
    tasks = list(mandatory)
    for task in tasks:
        if not isinstance(task, Mapping) or not str(task.get("task_id", "")).strip():
            raise PartCRev1NoOpError("Every mandatory task requires an explicit task_id")
        outputs = task.get("expected_outputs")
        if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)) or not outputs:
            raise PartCRev1NoOpError(f"Task {task.get('task_id')} requires nonempty expected_outputs")
    if task_ids is None:
        return tuple(tasks)
    requested = [str(value) for value in task_ids]
    if not requested or any(not value.strip() for value in requested) or len(requested) != len(set(requested)):
        raise PartCRev1NoOpError("Requested task IDs must be explicit, nonempty, and unique")
    by_id = {str(task["task_id"]): task for task in tasks}
    missing = [value for value in requested if value not in by_id]
    if missing:
        raise PartCRev1NoOpError(f"Requested task IDs are absent from the mandatory matrix: {missing}")
    return tuple(by_id[value] for value in requested)


def _git(repository: Path, *arguments: str) -> str:
    command = ["git", "-C", str(repository), "-c", "core.longpaths=true", *arguments]
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise PartCRev1ApprovalBindingError(f"Git approval check failed: {' '.join(arguments)}") from exc


def partC_rev1_verify_approval_bindings(
    config_path: str | Path,
    run_matrix_path: str | Path,
    approval_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify exact approval, hash, ancestry, task, and immutable-input bindings."""

    resolved_config = Path(config_path).resolve(strict=True)
    config = partC_rev1_load_campaign_config(resolved_config)
    repository = _repository_from_config(config)
    resolved_matrix = Path(run_matrix_path).resolve(strict=True)
    matrix = partC_rev1_load_run_matrix(resolved_matrix, repository_root=repository)
    approval_file = repository / _APPROVAL_RELATIVE if approval_path is None else Path(approval_path)
    approval = _strict_json(approval_file.resolve(strict=True))
    bindings = _strict_json((repository / _BINDINGS_RELATIVE).resolve(strict=True))
    if not approval.get("completion_campaign_execution_authorized"):
        raise PartCRev1ApprovalBindingError("Completion-campaign execution approval is absent")
    if approval.get("phase_2_authorized") or approval.get("scientific_real_data_execution_authorized"):
        raise PartCRev1ApprovalBindingError("Approval boundary unexpectedly enables prohibited work")
    exact_pairs = {
        "campaign_id": (config.get("campaign_id"), approval.get("campaign_id")),
        "run_id": (config.get("run_id"), approval.get("run_id")),
        "branch": (config.get("branch"), approval.get("approved_branch")),
        "output_root": (Path(str(config.get("output_root"))).as_posix(), approval.get("approved_output_root")),
        "temp_root": (Path(str(config.get("temp_root"))).as_posix(), approval.get("approved_temp_root")),
    }
    for name, (actual, expected) in exact_pairs.items():
        if actual != expected:
            raise PartCRev1ApprovalBindingError(f"Approval-bound {name} mismatch: {actual!r} != {expected!r}")
    expected_hashes = approval["approved_config_hashes"]
    if _sha256(resolved_config) != expected_hashes["execution_config_sha256"]:
        raise PartCRev1ApprovalBindingError("Execution-config SHA-256 mismatch")
    if _sha256(resolved_matrix) != approval["approved_run_matrix_hash"]:
        raise PartCRev1ApprovalBindingError("Run-matrix SHA-256 mismatch")
    resource_path = (repository / _RESOURCE_POLICY_RELATIVE).resolve(strict=True)
    if _sha256(resource_path) != expected_hashes["resource_policy_sha256"]:
        raise PartCRev1ApprovalBindingError("Resource-policy SHA-256 mismatch")
    prompt = (repository / approval["approved_goal_prompt"]["path"]).resolve(strict=True)
    if prompt.stat().st_size != int(approval["approved_goal_prompt"]["size_bytes"]) or _sha256(prompt) != approval["approved_goal_prompt"]["sha256"]:
        raise PartCRev1ApprovalBindingError("Approved Goal prompt byte identity mismatch")
    for section in ("authority_package", "step_1_1_reviewed_zip"):
        item = bindings[section]
        path = (repository / item["path"]).resolve(strict=True)
        if path.stat().st_size != int(item["size_bytes"]) or _sha256(path) != item["sha256"]:
            raise PartCRev1ApprovalBindingError(f"{section} byte identity mismatch")
    required = [str(value) for value in approval["mandatory_task_ids"]]
    observed = [str(item["task_id"]) for item in matrix["mandatory"]]
    if observed != required:
        raise PartCRev1ApprovalBindingError("Mandatory task order or closure differs from approval")
    if bindings["input_fingerprint"]["value"] != approval["approved_input_hashes"]["campaign_input_fingerprint"]:
        raise PartCRev1ApprovalBindingError("Campaign input fingerprint mismatch")
    if _git(repository, "branch", "--show-current") != approval["approved_branch"]:
        raise PartCRev1ApprovalBindingError("Current branch differs from approval")
    _git(repository, "merge-base", "--is-ancestor", approval["approved_starting_head"], "HEAD")
    return {
        "status": "PASS",
        "campaign_id": config["campaign_id"],
        "run_id": config["run_id"],
        "mandatory_task_count": len(observed),
        "config_sha256": _sha256(resolved_config),
        "run_matrix_sha256": _sha256(resolved_matrix),
        "resource_policy_sha256": _sha256(resource_path),
        "input_fingerprint": bindings["input_fingerprint"]["value"],
        "approval_commit": _git(
            repository,
            "log",
            "--format=%H",
            "--grep=^phase1: record completion campaign execution approval$",
            "-1",
        ),
        "current_head": _git(repository, "rev-parse", "HEAD"),
        "phase_2_authorized": False,
    }


def _has_reparse_component(path: Path, stop: Path) -> bool:
    current = path
    while True:
        if current.exists():
            info = current.lstat()
            attributes = int(getattr(info, "st_file_attributes", 0))
            if current.is_symlink() or (attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))):
                return True
        if current == stop or current.parent == current:
            return False
        current = current.parent


def _within(child: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath((os.path.normcase(str(child)), os.path.normcase(str(parent)))) == os.path.normcase(str(parent))
    except ValueError:
        return False


def partC_rev1_resolve_campaign_output(
    config: Mapping[str, Any], relative_path: str, *, create_parent: bool = False
) -> Path:
    """Resolve only a relative path beneath the exact approved output root."""

    repository = _repository_from_config(config)
    raw = str(relative_path)
    pure = PurePosixPath(raw)
    if not raw.strip() or pure.is_absolute() or ".." in pure.parts or "\\" in raw or ":" in raw or pure == PurePosixPath("."):
        raise PartCRev1CampaignPathError(f"Unsafe campaign output relative path: {relative_path!r}")
    output_value = config.get("output_root")
    if not isinstance(output_value, str) or not output_value.strip():
        raise PartCRev1CampaignPathError("output_root must be explicit")
    output_root = Path(output_value).resolve(strict=True)
    registry = _strict_json((repository / _REGISTRY_RELATIVE).resolve(strict=True))
    manifest = _strict_json((repository / _MANIFEST_RELATIVE).resolve(strict=True))
    original_root = Path(registry["original_repository_root"]).resolve(strict=True)
    data_roots = {
        "corrected_graph_root": Path(registry["authoritative_inputs"]["corrected_graph"]["root"]).resolve(strict=True),
        "od_root": Path(registry["authoritative_inputs"]["od"]["root"]).resolve(strict=True),
        "original_repository_root": original_root,
    }
    forbidden = {repository, original_root}
    forbidden.update((data_roots[item["root_key"]] / item["relative_path"]).resolve(strict=True).parent for item in manifest["inputs"])
    if any(_within(output_root, root) for root in forbidden):
        raise PartCRev1CampaignPathError("Approved output root overlaps a checkout or authoritative-input directory")
    destination = (output_root / Path(*pure.parts)).resolve(strict=False)
    if not _within(destination, output_root) or destination == output_root:
        raise PartCRev1CampaignPathError("Campaign output escapes the exact approved root")
    if _has_reparse_component(destination.parent, output_root):
        raise PartCRev1CampaignPathError("Campaign output path contains reparse ambiguity")
    if create_parent:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _has_reparse_component(destination.parent, output_root):
            raise PartCRev1CampaignPathError("Created campaign output parent contains reparse ambiguity")
    return destination


def partC_rev1_guard_meaningful_outputs(
    paths: Sequence[str | Path], *, minimum_file_count: int = 1, minimum_total_bytes: int = 1
) -> tuple[Path, ...]:
    """Reject a no-op, missing output, empty file, or zero-byte output set."""

    if minimum_file_count <= 0 or minimum_total_bytes <= 0:
        raise PartCRev1NoOpError("Meaningful-output minima must be positive")
    if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)) or not paths:
        raise PartCRev1NoOpError("Meaningful output list must be nonempty")
    files: list[Path] = []
    for value in paths:
        candidate = Path(value).resolve(strict=True)
        if candidate.is_file():
            files.append(candidate)
        elif candidate.is_dir():
            files.extend(item for item in sorted(candidate.rglob("*")) if item.is_file())
        else:
            raise PartCRev1NoOpError(f"Meaningful output is not a regular file or directory: {candidate}")
    if len(files) < minimum_file_count:
        raise PartCRev1NoOpError(f"Meaningful output file count {len(files)} is below {minimum_file_count}")
    if any(item.stat().st_size <= 0 for item in files):
        raise PartCRev1NoOpError("Meaningful outputs may not contain zero-byte files")
    total = sum(item.stat().st_size for item in files)
    if total < minimum_total_bytes:
        raise PartCRev1NoOpError(f"Meaningful output bytes {total} are below {minimum_total_bytes}")
    return tuple(files)
