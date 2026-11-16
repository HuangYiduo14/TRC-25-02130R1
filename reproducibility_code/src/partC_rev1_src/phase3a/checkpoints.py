"""Manifest-consistent Phase 3A checkpoints using the validated Phase 1 API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from partC_rev1_src.partC_rev1_phase1_checkpoint import (
    PartCRev1CampaignCheckpoint,
    PartCRev1CheckpointCompatibilityError,
    partC_rev1_read_verified_checkpoint,
    partC_rev1_validate_resume_compatibility,
    partC_rev1_write_atomic_checkpoint,
)

from partC_rev1_src.partC_rev1_hashing import partC_rev1_file_sha256


CHECKPOINT_STATES = frozenset(
    {
        "NOT_STARTED", "RUNNING", "CHECKPOINTED", "COMPLETE_UNSEALED", "VALIDATED", "SEALED",
        "FAILED_RECOVERABLE", "FAILED_FINAL", "BLOCKED_REQUIRES_EXTERNAL_DECISION",
    }
)
RESUMABLE_STATES = frozenset({"CHECKPOINTED", "COMPLETE_UNSEALED", "VALIDATED", "FAILED_RECOVERABLE"})


class Phase3ACheckpointError(RuntimeError):
    pass


def write_phase3a_checkpoint(
    destination: str | Path,
    *,
    schema_path: str | Path,
    campaign_id: str,
    run_id: str,
    stage_id: str,
    candidate_id: str | None,
    state: str,
    input_hashes: Mapping[str, str],
    config_sha256: str,
    code_identity: str,
    completed_shard_ids: Sequence[str],
    artifact_records: Sequence[Mapping[str, Any]],
    resource_ledger: Mapping[str, Any],
    next_deterministic_action: str,
    recoverability_status: str,
) -> PartCRev1CampaignCheckpoint:
    if state not in CHECKPOINT_STATES:
        raise Phase3ACheckpointError(f"Unknown checkpoint state: {state}")
    payload = {
        "schema_id": "phase3a.checkpoint",
        "schema_version": "phase3a-1.0",
        "campaign_id": str(campaign_id),
        "run_id": str(run_id),
        "stage_id": str(stage_id),
        "candidate_id": None if candidate_id is None else str(candidate_id),
        "state": state,
        "input_hashes": dict(sorted((str(key), str(value)) for key, value in input_hashes.items())),
        "config_sha256": str(config_sha256),
        "code_identity": str(code_identity),
        "completed_shard_ids": sorted(set(map(str, completed_shard_ids))),
        "artifact_records": [dict(value) for value in artifact_records],
        "resource_ledger": dict(resource_ledger),
        "next_deterministic_action": str(next_deterministic_action),
        "recoverability_status": str(recoverability_status),
        "written_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    return partC_rev1_write_atomic_checkpoint(destination, payload, schema_path)


def resume_phase3a_checkpoint(
    path: str | Path,
    *,
    schema_path: str | Path,
    expected: Mapping[str, Any],
) -> PartCRev1CampaignCheckpoint:
    checkpoint = partC_rev1_read_verified_checkpoint(path, schema_path)
    if checkpoint.payload.get("state") not in RESUMABLE_STATES:
        raise Phase3ACheckpointError(f"Checkpoint state is not resumable: {checkpoint.payload.get('state')}")
    try:
        partC_rev1_validate_resume_compatibility(checkpoint, expected)
    except PartCRev1CheckpointCompatibilityError as exc:
        raise Phase3ACheckpointError(str(exc)) from exc
    for record in checkpoint.payload.get("artifact_records", []):
        artifact = Path(str(record["path"])).resolve(strict=True)
        if artifact.stat().st_size != int(record["size_bytes"]) or partC_rev1_file_sha256(artifact) != str(record["sha256"]):
            raise Phase3ACheckpointError(f"Checkpoint artifact changed: {artifact}")
    return checkpoint
