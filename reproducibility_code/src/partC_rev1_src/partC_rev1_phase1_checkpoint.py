"""Atomic checkpoint and exact-resume support for the Phase 1 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

from .partC_rev1_phase1_campaign_contract import (
    PartCRev1CampaignContractError,
    partC_rev1_load_campaign_config,
    partC_rev1_load_run_matrix,
    partC_rev1_resolve_campaign_output,
    partC_rev1_verify_approval_bindings,
)


_SCHEMA_RELATIVE = Path(
    "partC_doc_icnn/phase_1_work/completion_campaign/schemas/phase1_checkpoint.schema.json"
)
_RUN_MATRIX_RELATIVE = Path(
    "partC_doc_icnn/phase_1_work/completion_campaign/preflight/"
    "phase1_completion_campaign_20260812_v1/PHASE1_COMPLETION_RUN_MATRIX.json"
)
_RESOURCE_POLICY_RELATIVE = Path(
    "partC_doc_icnn/phase_1_work/completion_campaign/preflight/"
    "phase1_completion_campaign_20260812_v1/PHASE1_COMPLETION_RESOURCE_POLICY.json"
)
_BINDINGS_RELATIVE = Path(
    "partC_doc_icnn/phase_1_work/completion_campaign/preflight/"
    "phase1_completion_campaign_20260812_v1/PHASE1_COMPLETION_APPROVAL_BINDINGS.json"
)


class PartCRev1CheckpointError(PartCRev1CampaignContractError):
    """Base checkpoint error."""


class PartCRev1CheckpointIntegrityError(PartCRev1CheckpointError):
    """Raised for payload, sidecar, canonicalization, or schema corruption."""


class PartCRev1CheckpointCompatibilityError(PartCRev1CheckpointError):
    """Raised when a resume identity differs from the checkpoint."""


@dataclass(frozen=True, slots=True)
class PartCRev1CampaignCheckpoint:
    path: Path
    sidecar_path: Path
    payload: Mapping[str, Any]
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "sidecar_path": self.sidecar_path.as_posix(),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "payload": json.loads(json.dumps(self.payload, allow_nan=False)),
        }


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PartCRev1CheckpointIntegrityError(f"Duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PartCRev1CheckpointIntegrityError(f"Invalid strict UTF-8 checkpoint JSON in {label}") from exc
    if not isinstance(value, dict):
        raise PartCRev1CheckpointIntegrityError(f"Checkpoint payload must be an object: {label}")
    return value


def _canonical(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PartCRev1CheckpointIntegrityError("Checkpoint payload must be finite canonical JSON") from exc


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validate_payload(payload: Mapping[str, Any], schema_path: str | Path | None) -> None:
    schema_file = (
        _repository_root() / _SCHEMA_RELATIVE if schema_path is None else Path(schema_path)
    ).resolve(strict=True)
    schema = _strict_json_bytes(schema_file.read_bytes(), str(schema_file))
    try:
        validator = jsonschema.Draft202012Validator(schema)
        validator.check_schema(schema)
        errors = sorted(validator.iter_errors(dict(payload)), key=lambda item: list(item.absolute_path))
    except jsonschema.SchemaError as exc:
        raise PartCRev1CheckpointIntegrityError(f"Invalid checkpoint schema: {exc.message}") from exc
    if errors:
        first = errors[0]
        location = ".".join(str(value) for value in first.absolute_path) or "$"
        raise PartCRev1CheckpointIntegrityError(
            f"Checkpoint schema validation failed at {location}: {first.message}"
        )


def partC_rev1_write_atomic_checkpoint(
    destination: str | Path,
    payload: Mapping[str, Any],
    schema_path: str | Path | None = None,
) -> PartCRev1CampaignCheckpoint:
    """Write canonical payload and sidecar atomically, then reopen both."""

    target = Path(destination).resolve(strict=False)
    if not target.parent.is_dir():
        raise PartCRev1CheckpointError("Checkpoint parent must exist after caller path validation")
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if target.exists() or sidecar.exists():
        raise PartCRev1CheckpointError(f"Refusing to overwrite checkpoint or sidecar: {target}")
    _validate_payload(payload, schema_path)
    encoded = _canonical(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    payload_tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
    sidecar_tmp = sidecar.with_name(sidecar.name + f".tmp.{os.getpid()}")
    if payload_tmp.exists() or sidecar_tmp.exists():
        raise PartCRev1CheckpointError("Checkpoint temporary path already exists")
    try:
        with payload_tmp.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        with sidecar_tmp.open("xb") as handle:
            handle.write((digest + "\n").encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(payload_tmp, target)
        os.replace(sidecar_tmp, sidecar)
    finally:
        for item in (payload_tmp, sidecar_tmp):
            if item.exists():
                item.unlink()
    return partC_rev1_read_verified_checkpoint(target, schema_path)


def partC_rev1_read_verified_checkpoint(
    path: str | Path, schema_path: str | Path | None = None
) -> PartCRev1CampaignCheckpoint:
    target = Path(path).resolve(strict=True)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if not sidecar.is_file():
        raise PartCRev1CheckpointIntegrityError(f"Checkpoint SHA-256 sidecar is missing: {sidecar}")
    raw = target.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    try:
        declared = sidecar.read_text(encoding="ascii", errors="strict").strip()
    except (OSError, UnicodeError) as exc:
        raise PartCRev1CheckpointIntegrityError("Checkpoint sidecar is not strict ASCII") from exc
    if len(declared) != 64 or any(value not in "0123456789abcdef" for value in declared):
        raise PartCRev1CheckpointIntegrityError("Checkpoint sidecar is not a lowercase SHA-256")
    if observed != declared:
        raise PartCRev1CheckpointIntegrityError(
            f"Checkpoint payload/sidecar mismatch: observed {observed}, declared {declared}"
        )
    payload = _strict_json_bytes(raw, str(target))
    if _canonical(payload) != raw:
        raise PartCRev1CheckpointIntegrityError("Checkpoint payload is not canonical compact sorted-key JSON")
    _validate_payload(payload, schema_path)
    return PartCRev1CampaignCheckpoint(
        path=target,
        sidecar_path=sidecar,
        payload=payload,
        sha256=observed,
        size_bytes=len(raw),
    )


def partC_rev1_validate_resume_compatibility(
    checkpoint: PartCRev1CampaignCheckpoint, expected: Mapping[str, Any]
) -> PartCRev1CampaignCheckpoint:
    """Require exact equality for every supplied approval and artifact identity."""

    if not isinstance(expected, Mapping) or not expected:
        raise PartCRev1CheckpointCompatibilityError("Resume expectations must be explicit and nonempty")
    for key, expected_value in expected.items():
        if key not in checkpoint.payload:
            raise PartCRev1CheckpointCompatibilityError(f"Checkpoint lacks resume field: {key}")
        actual = checkpoint.payload[key]
        if key == "artifact_hashes":
            if not isinstance(expected_value, Mapping) or dict(actual) != dict(expected_value):
                raise PartCRev1CheckpointCompatibilityError("Completed artifact hashes differ from checkpoint")
        elif actual != expected_value:
            raise PartCRev1CheckpointCompatibilityError(
                f"Resume identity mismatch for {key}: expected {expected_value!r}, observed {actual!r}"
            )
    return checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "-c", "core.longpaths=true", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_report(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise PartCRev1CheckpointError(f"Refusing to overwrite self-test report: {path}")
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
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


def _self_test(config_path: Path, task_id: str) -> dict[str, Any]:
    if task_id != "S2-T02":
        raise PartCRev1CheckpointError("Checkpoint self-test is bound only to S2-T02")
    config = partC_rev1_load_campaign_config(config_path)
    repository = Path(config["repository_root"]).resolve(strict=True)
    matrix_path = (repository / _RUN_MATRIX_RELATIVE).resolve(strict=True)
    matrix = partC_rev1_load_run_matrix(matrix_path, repository_root=repository)
    task = next((item for item in matrix["mandatory"] if item["task_id"] == task_id), None)
    if task is None:
        raise PartCRev1CheckpointError("S2-T02 is absent from the approved run matrix")
    binding = partC_rev1_verify_approval_bindings(config_path, matrix_path)
    bindings = json.loads((repository / _BINDINGS_RELATIVE).read_text(encoding="utf-8"))
    resource_path = (repository / _RESOURCE_POLICY_RELATIVE).resolve(strict=True)
    report_path = partC_rev1_resolve_campaign_output(
        config, "stage_02/checkpoint_resume_smoke.json", create_parent=True
    )
    checkpoint_root = Path(config["checkpoint"]["directory"]).resolve(strict=True)
    temp_root = Path(config["temp_root"]).resolve(strict=True)
    fixture_root = temp_root / "checkpoint_self_test" / task_id
    fixture_root.mkdir(parents=True, exist_ok=False)
    current_head = _git_head(repository)
    common = {
        "schema_id": "partC_rev1_phase1_checkpoint",
        "schema_version": "1.0",
        "campaign_id": config["campaign_id"],
        "run_id": config["run_id"],
        "branch": config["branch"],
        "stage_id": "step_1_4",
        "task_id": task_id,
        "completed_task_ids": ["S1-T01", "S1-T02", "S2-T01"],
        "failed_task_ids": [],
        "pending_task_ids": ["S2-T02", "S3-T01", "S3-T02", "S4-T01", "S5-T01", "S5-T02"],
        "approved_preflight_code_commit": config["approval_baseline_code_commit"],
        "authority_package_sha256": bindings["authority_package"]["sha256"],
        "step_1_1_reviewed_zip_sha256": bindings["step_1_1_reviewed_zip"]["sha256"],
        "input_fingerprint": bindings["input_fingerprint"]["value"],
        "execution_config_sha256": binding["config_sha256"],
        "run_matrix_sha256": binding["run_matrix_sha256"],
        "resource_policy_sha256": _sha256(resource_path),
        "output_root": Path(config["output_root"]).as_posix(),
        "stage_code_commit": current_head,
        "artifact_hashes": {"stage_01_commit": current_head},
        "repair_count": 0,
        "resource_maxima": {
            "rss_bytes": 1,
            "output_added_bytes": 1,
            "temp_added_bytes": 1,
            "elapsed_seconds": 1,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    interrupted_payload = {**common, "state": "INTERRUPTED_AFTER_COMPLETED_TASK"}
    interrupted = partC_rev1_write_atomic_checkpoint(
        checkpoint_root / "stage_02_self_test_interrupted.json", interrupted_payload
    )
    expected = {
        key: interrupted_payload[key]
        for key in (
            "campaign_id",
            "run_id",
            "branch",
            "approved_preflight_code_commit",
            "authority_package_sha256",
            "step_1_1_reviewed_zip_sha256",
            "input_fingerprint",
            "execution_config_sha256",
            "run_matrix_sha256",
            "resource_policy_sha256",
            "output_root",
            "stage_id",
            "stage_code_commit",
            "artifact_hashes",
        )
    }
    partC_rev1_validate_resume_compatibility(interrupted, expected)
    mismatch_payload = fixture_root / "mismatch.json"
    mismatch_sidecar = mismatch_payload.with_suffix(".json.sha256")
    shutil.copy2(interrupted.path, mismatch_payload)
    shutil.copy2(interrupted.sidecar_path, mismatch_sidecar)
    mismatch_sidecar.write_text("0" * 64 + "\n", encoding="ascii", newline="\n")
    mismatch_rejected = False
    try:
        partC_rev1_read_verified_checkpoint(mismatch_payload)
    except PartCRev1CheckpointIntegrityError:
        mismatch_rejected = True
    if not mismatch_rejected:
        raise PartCRev1CheckpointError("Deliberate checkpoint mismatch was not rejected")
    mismatch_payload.unlink()
    mismatch_sidecar.unlink()
    fixture_root.rmdir()
    fixture_root.parent.rmdir()
    resumed_payload = {
        **common,
        "state": "RESUME_COMPATIBLE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    resumed = partC_rev1_write_atomic_checkpoint(
        checkpoint_root / "stage_02_self_test_resumed.json", resumed_payload
    )
    partC_rev1_validate_resume_compatibility(resumed, expected)
    report = {
        "schema_version": "1.0",
        "status": "PASS",
        "campaign_id": config["campaign_id"],
        "run_id": config["run_id"],
        "stage_id": "step_1_4",
        "task_id": task_id,
        "atomic_write_read_passed": True,
        "compatible_resume_passed": True,
        "deliberate_hash_mismatch_rejected": mismatch_rejected,
        "temporary_mismatch_fixture_removed": not fixture_root.exists(),
        "interrupted_checkpoint": interrupted.to_dict(),
        "resumed_checkpoint": resumed.to_dict(),
        "phase_2_authorized": False,
        "scientific_real_data_execution_performed": False,
    }
    _atomic_report(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--self-test", action="store_true", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.self_test:
        raise PartCRev1CheckpointError("Only the explicit --self-test CLI is approved")
    _self_test(Path(args.config).resolve(strict=True), str(args.task_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
