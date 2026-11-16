"""Approved-mode dispatcher for the reusable Phase 3A stage engine.

Handlers support either a reviewed full Task 2 campaign or the explicitly
bounded Task 1 corrective-continuation scope.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .schemas import write_json_new
from .operations import (
    BOUNDED_TASK1_CONTINUATION,
    FULL_TASK2,
    census_aug_witness_shard,
    construct_aug_candidate,
    construct_geo_candidate,
    evaluate_candidate_od_batch,
    finalize_candidate_topology,
    reproduce_topologies,
    seal_handoff,
    validate_campaign,
)


class Phase3AOrchestrationError(RuntimeError):
    pass


MODE_REQUIRED_TASK_PREFIX = {
    "round1_geo": "R1-",
    "round2_aug": "R2-",
    "round3_finalize": "R3-",
    "round4_evaluate": "R4-",
    "validate": "VAL-",
    "reproduce": "REP-",
    "handoff": "HANDOFF-",
}


def run_approved_mode(config: Mapping[str, Any], mode: str, tasks: Sequence[str]) -> dict[str, Any]:
    """Dispatch a nonempty frozen task batch after approval hash validation.

    The final Task 2 plan supplies an ordered ``task_contracts`` mapping. Each
    contract names one implemented operation primitive. This avoids implicit
    defaults and makes a missing operation a hard failure rather than a no-op.
    """

    scope = str(config.get("execution_scope", FULL_TASK2))
    if scope == FULL_TASK2:
        approval = config.get("task2_approval")
        if not isinstance(approval, Mapping) or not approval.get("authorized"):
            raise Phase3AOrchestrationError("Task 2 approval is absent")
    elif scope == BOUNDED_TASK1_CONTINUATION:
        authority = config.get("task1_continuation_authority")
        if not isinstance(authority, Mapping) or not authority.get("authorized"):
            raise Phase3AOrchestrationError("Bounded Task 1 continuation authority is absent")
        if authority.get("full_task2_authorized") or authority.get("phase3b_authorized") or authority.get("phase4_authorized"):
            raise Phase3AOrchestrationError("Bounded continuation authority crosses a frozen phase boundary")
    else:
        raise Phase3AOrchestrationError(f"Unknown Phase 3A execution scope: {scope}")
    contracts = config.get("task_contracts")
    if not isinstance(contracts, Mapping):
        raise Phase3AOrchestrationError("Approved execution config lacks task contracts")
    prefix = MODE_REQUIRED_TASK_PREFIX.get(str(mode))
    if prefix is None or not tasks or any(not str(task).startswith(prefix) for task in tasks):
        raise Phase3AOrchestrationError("Mode task list is empty or has an inconsistent prefix")
    output_root = Path(str(config["output_root"])).resolve(strict=True)
    completed: list[dict[str, Any]] = []
    for task_id in tasks:
        contract = contracts.get(task_id)
        if not isinstance(contract, Mapping):
            raise Phase3AOrchestrationError(f"Missing frozen task contract: {task_id}")
        operation = str(contract.get("operation", ""))
        handler = _OPERATIONS.get(operation)
        if handler is None:
            raise Phase3AOrchestrationError(f"Unknown/nonimplemented task operation: {operation}")
        scoped_contract = {**dict(contract), "task_id": str(task_id)}
        result = handler(config, scoped_contract)
        if result.get("status") != "PASS" or int(result.get("meaningful_output_count", 0)) <= 0:
            raise Phase3AOrchestrationError(f"Task failed or was a no-op: {task_id}")
        completed.append({"task_id": task_id, **result})
    receipt = {
        "schema_version": "phase3a-1.0",
        "mode": str(mode),
        "status": "PASS",
        "completed_tasks": completed,
        "meaningful_output_count": sum(int(value["meaningful_output_count"]) for value in completed),
    }
    write_json_new(output_root / str(mode) / f"batch_{tasks[0]}_{tasks[-1]}.json", receipt)
    return receipt


_OPERATIONS = {
    "construct_geo_candidate": construct_geo_candidate,
    "census_aug_witness_shard": census_aug_witness_shard,
    "construct_aug_candidate": construct_aug_candidate,
    "finalize_candidate_topology": finalize_candidate_topology,
    "evaluate_candidate_od_batch": evaluate_candidate_od_batch,
    "validate_campaign": validate_campaign,
    "reproduce_topologies": reproduce_topologies,
    "seal_handoff": seal_handoff,
}
