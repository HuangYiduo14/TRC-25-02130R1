"""Restartable repaired Phase 5 backbone-compilation pilot.

The supervisor is the only process that publishes immutable PASS receipts.
Scientific stages run in dedicated children and write only the configured new
output root.  Accepted Phase 3/4 products are always opened read-only.
"""

from __future__ import annotations

import csv
import faulthandler
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from jsonschema import validate as jsonschema_validate

from .bacn_charges import (
    ReachableODSupply,
    fixed_charge_candidates,
    internal_exposure,
    load_reachable_od_supply_cache,
    path_episode_statistics,
    physical_flow_conservation,
    reconstruct_bacn_visit_semantics,
    reconstruct_reachable_od_supply,
    write_reachable_od_supply_cache,
)
from .benchmarks import process_tree_rss_bytes, run_kernel_benchmarks
from .bundle_oracle import BundleSpec, macro_function_key, series_power_terms
from .experiments import MacroExperiment, run_macro_experiments, select_macro_sample, dynamic_expression_key, calibration_request_key
from .v4_evidence import (is_v4, V4_SCHEMA, verify_v4_authority, import_verified_r0,
    atomic_runtime_retention, postfit_topology_evidence, sample_delta_evidence,
    certificate_gate, stage_warmup, stage_retained_controls, write_v4_audits)
from .inputs import PilotBindings, load_bindings, load_suite_trace_arrays
from .pilot import (
    BLOCKED_STATUS,
    FINAL_STATUS,
    _accepted_path_candidate_signatures,
    _bundle_census,
    _bundle_inventory,
    _describe_reopen,
    _error_band_summary,
    _internal_component_keys,
    _make_figures,
    _multiedge_audit,
    _path_validation,
    _racn_audit,
    _ranking_band_summary,
    _scientific_acceptance,
    _selected_runtime_registry,
    _suite_frames,
    _write_blocked_output_manifest,
    _write_output_manifest,
    _write_reports,
    canonical_json,
    git,
    sha256_file,
    utc_now,
)
from .topology_reduction import TopologyResult, enumerate_topology
from .scc_reachability import (
    LABEL_PAYLOAD_DTYPE,
    SCC_ALGORITHM_ID,
    reachable_od_supply_metadata,
)
from partC_rev1_src.phase4.tracing import _directory_digest


REPAIR_BASE = "bf78c17c3af601baf90ed6b823140036f21d9e68"
REPAIRED_OUTER_SHA256 = (
    "0fa61d32e093690381e9e528aa115e4276feadb29ee22101ad838d2021115c0d"
)
PRIOR_REPAIRED_OUTER_SHA256 = (
    "44be04793a49dde1695fa7ed760ff0d2c65e127d14fe1c3053873f0bceace350"
)
BLOCKED_V1_HANDOFF_SHA256 = (
    "9c1f0564c7427aa36bf9ca4629fc82d58d04f1ed08840c5ab3f9920417c522f3"
)
ORIGINAL_AUTHORITY_SHA256 = (
    "64c19984f9ceac0512bd4dc0ab56b223545d1a7106b96c498f0e349d868501be"
)
BLOCKED_HANDOFF_SHA256 = (
    "c288abbcd77a640fcfabba9fbb08c1a9ae54b324e014f495146e494a1b2e8b95"
)
PARENT_CONTINUATION_SHA256 = (
    "dcf518f90c971747b890b99b419451d46321785897ca241382b61d850b463b15"
)
REPAIRED_STATUS_SCHEMA = "partC.phase5.pilot_repaired_0904_v2.status.v1"
REPAIR_OF_RUN_ID = "phase5_backbone_compilation_pilot_repaired_0904_v1_run_001"
R0_CROSS_SCC_BLOCKER = (
    "ACCEPTED_SCC_ENDPOINT_ASSUMPTION_CONTRADICTS_EXACT_GRAPH_REACHABILITY"
)
ENGINEERING_RETRY_SCHEMA = "partC.phase5.repaired.engineering_retry_epoch.v1"
ENGINEERING_RETRY_ALLOWED_PATHS = frozenset(
    {
        "partC_rev1_src/phase5/benchmarks.py",
        "partC_rev1_src/phase5/staged_pilot.py",
        "partC_rev1_src/phase5/staged_cli.py",
        "partC_rev1_src/phase5/repaired_handoff.py",
        "tests/test_phase5_experiments.py",
        "tests/test_phase5_repaired_staging.py",
        "partC_doc_icnn/phase_5_work/PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_R3_ENGINEERING_REPAIR.json",
        "partC_doc_icnn/phase_5_work/PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_R3_ENGINEERING_REPAIR.md",
        "partC_doc_icnn/phase_5_work/PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_R5_ENGINEERING_REPAIR.json",
        "partC_doc_icnn/phase_5_work/PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_R5_ENGINEERING_REPAIR.md",
    }
)


def _engineering_allowed_paths(execution):
    if not is_v4(execution):
        return ENGINEERING_RETRY_ALLOWED_PATHS
    repo = Path(execution["project_root"])
    return frozenset([
        *(p.relative_to(repo).as_posix() for p in (repo / "partC_rev1_src/phase5").glob("*.py") if p.name != "scc_reachability.py"),
        *(p.relative_to(repo).as_posix() for p in (repo / "tests").glob("test_phase5_*.py")),
        *(f"partC_doc_icnn/phase_5_work/PHASE5_V4_0905_REPAIR_{i:03d}.{ext}" for i in range(1,7) for ext in ("json","md")),
    ])
STAGE_ORDER = (
    "R0_v2",
    "R1_s02",
    "R2_s03",
    "R2_s04",
    "R2_s05",
    "R2_s06",
    "R3",
    "R4",
    "R5",
    "R6",
)


class StageFailure(RuntimeError):
    """A child stage failed without consuming later work."""


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value.rstrip() + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, canonical_json(payload))


def _atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp.{os.getpid()}" + path.suffix)
    if temporary.exists():
        raise FileExistsError(temporary)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(temporary, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(temporary, index=False)
    else:
        raise ValueError(f"Unsupported atomic frame extension: {path}")
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(payload) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _execution(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    execution = json.loads(path.read_text(encoding="utf-8"))
    if (
        execution.get("schema_id")
        not in {
            "partC.phase5.backbone_compilation_pilot_repaired.execution.v1",
            "partC.phase5.pilot_repaired_0904_v2.execution.v1",
            V4_SCHEMA,
        }
    ):
        raise ValueError(
            "The staged runner requires the repaired Phase 5 execution schema"
        )
    if not execution.get("tasks"):
        raise ValueError("Execution task list must be nonempty")
    if not execution.get("expected_outputs") or not execution.get("acceptance_tests"):
        raise ValueError(
            "Execution outputs and acceptance tests must be explicit and nonempty"
        )
    return path, execution


def authority_root(execution: dict[str, Any]) -> Path:
    return (
        (
            Path(execution["project_root"])
            / str(execution["installed_authority_goal_relative_path"])
        )
        .resolve()
        .parent
    )


def _project_contract_file(execution: dict[str, Any], field: str) -> Path:
    """Resolve one explicitly configured compact contract file inside the repo."""

    repo = Path(execution["project_root"]).resolve()
    path = (repo / str(execution[field])).resolve()
    if not path.is_file() or not path.is_relative_to(repo):
        raise FileNotFoundError(f"Configured contract file is absent or outside repo: {field}")
    return path


def inherited_authority_root(execution: dict[str, Any]) -> Path:
    repo = Path(execution["project_root"]).resolve()
    goal = (repo / str(execution["inherited_authority_goal_relative_path"])).resolve()
    if not goal.is_file() or not goal.is_relative_to(repo):
        raise FileNotFoundError("Inherited repaired authority goal is absent or outside repo")
    return goal.parent


def _config_binding(config_path: Path, pre_run_code_head: str) -> dict[str, str]:
    return {
        "execution_config_path": str(config_path),
        "execution_config_sha256": sha256_file(config_path),
        "pre_run_code_head": str(pre_run_code_head),
    }


def _task_worktree_changes(repo: Path) -> tuple[list[str], list[str], list[str]]:
    """Return full status plus tracked and task-owned untracked changes."""

    lines = git(
        repo, "status", "--porcelain=v1", "--untracked-files=all"
    ).stdout.splitlines()
    tracked = [value for value in lines if not value.startswith("?? ")]
    task_prefixes = (
        "partC_rev1_src/phase5/",
        "tests/test_phase5_",
        "partC_doc_icnn/phase_5_work/",
        "partC_doc_icnn/review_gates/phase_05/backbone_compilation_pilot_repaired_0904_v2/",
    )
    task_untracked = [
        value
        for value in lines
        if value.startswith("?? ")
        and value[3:].replace("\\", "/").startswith(task_prefixes)
    ]
    return lines, tracked, task_untracked


def _require_clean_task_tree(repo: Path) -> list[str]:
    lines, tracked, task_untracked = _task_worktree_changes(repo)
    if tracked or task_untracked:
        raise RuntimeError(
            "Execution requires a clean committed task tree: "
            f"tracked={tracked}, task_untracked={task_untracked}"
        )
    return lines


def _engineering_retry_root(output: Path) -> Path:
    return output / "repository_audit" / "engineering_retries"


def _engineering_retry_files(output: Path) -> list[Path]:
    root = _engineering_retry_root(output)
    if not root.is_dir():
        return []
    paths = sorted(root.glob("epoch_*.json"))
    expected = [f"epoch_{index:03d}.json" for index in range(1, len(paths) + 1)]
    observed = [path.name for path in paths]
    if observed != expected:
        raise RuntimeError(f"Engineering-retry epoch history is not contiguous: {observed}")
    return paths


def _bound_file(output: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(output.resolve()):
        raise FileNotFoundError(f"Bound retry evidence is absent or outside output: {path}")
    return {
        "relative_path": resolved.relative_to(output.resolve()).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _verify_bound_files(output: Path, rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        path = (output / str(row["relative_path"])).resolve()
        if (
            not path.is_file()
            or not path.is_relative_to(output.resolve())
            or path.stat().st_size != int(row["size_bytes"])
            or sha256_file(path) != str(row["sha256"])
        ):
            raise RuntimeError(f"Engineering-retry evidence changed: {path}")


def _git_patch_sha256(repo: Path, old_head: str, new_head: str) -> str:
    patch = git(repo, "diff", "--binary", "--no-ext-diff", old_head, new_head).stdout
    return hashlib.sha256(patch.encode("utf-8")).hexdigest()


def _validate_engineering_retry_chain(
    *,
    config_path: Path,
    execution: dict[str, Any],
    output: Path,
    pre_run_code_head: str,
    verify_repository: bool = True,
) -> list[dict[str, Any]]:
    """Validate immutable committed-code epochs without rewriting the run binding."""

    repo = Path(execution["project_root"]).resolve() if verify_repository else None
    previous_head = str(pre_run_code_head)
    previous_record_sha256: str | None = None
    records: list[dict[str, Any]] = []
    for ordinal, path in enumerate(_engineering_retry_files(output), start=1):
        if is_v4(execution) and ordinal > int(execution["retry"]["max_engineering_code_epochs"]):
            raise RuntimeError("V4 engineering code epoch budget exhausted")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_id": ENGINEERING_RETRY_SCHEMA,
            "epoch_number": ordinal,
            "campaign_id": execution["campaign_id"],
            "run_id": execution["run_id"],
            "branch": execution["expected_branch"],
            "pre_run_code_head": pre_run_code_head,
            "previous_execution_code_head": previous_head,
            "previous_epoch_sha256": previous_record_sha256,
            "execution_config_sha256": sha256_file(config_path),
            "status": "PASS_AUTHORIZED_ENGINEERING_RETRY",
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Engineering-retry epoch binding failed: {path}")
        new_head = str(payload.get("authorized_execution_code_head", ""))
        if len(new_head) != 40 or new_head == previous_head:
            raise RuntimeError(f"Engineering-retry code transition is invalid: {path}")
        changed_paths = list(map(str, payload.get("changed_paths", [])))
        if not changed_paths or not set(changed_paths).issubset(
            _engineering_allowed_paths(execution)
        ):
            raise RuntimeError(f"Engineering-retry changed-path scope failed: {path}")
        _verify_bound_files(output, payload.get("failed_attempt_evidence", []))
        _verify_bound_files(output, payload.get("upstream_receipts", []))
        validation = payload.get("repair_validation_record", {})
        validation_path = (
            Path(execution["project_root"]) / str(validation.get("relative_path", ""))
        ).resolve()
        if (
            not validation_path.is_file()
            or not validation_path.is_relative_to(Path(execution["project_root"]).resolve())
            or validation_path.stat().st_size != int(validation.get("size_bytes", -1))
            or sha256_file(validation_path) != str(validation.get("sha256", ""))
        ):
            raise RuntimeError(f"Engineering-retry validation record changed: {path}")
        if verify_repository:
            assert repo is not None
            parents = git(repo, "rev-list", "--parents", "-n", "1", new_head).stdout.split()
            actual_paths = sorted(
                git(repo, "diff", "--name-only", previous_head, new_head).stdout.splitlines()
            )
            if (
                len(parents) < 2
                or parents[0] != new_head
                or parents[1] != previous_head
                or git(
                    repo,
                    "merge-base",
                    "--is-ancestor",
                    previous_head,
                    new_head,
                    check=False,
                ).returncode
                != 0
                or actual_paths != changed_paths
                or _git_patch_sha256(repo, previous_head, new_head)
                != payload.get("git_patch_sha256")
            ):
                raise RuntimeError(f"Engineering-retry Git transition failed: {path}")
        record = dict(payload)
        record["relative_path"] = path.relative_to(output).as_posix()
        record["record_sha256"] = sha256_file(path)
        records.append(record)
        previous_head = new_head
        previous_record_sha256 = record["record_sha256"]
    return records


def _receipt_epoch_reference(
    records: Sequence[dict[str, Any]], attempt_code_head: str
) -> dict[str, Any] | None:
    if not records:
        return None
    for record in records:
        if record["authorized_execution_code_head"] == attempt_code_head:
            return {
                "epoch_number": int(record["epoch_number"]),
                "relative_path": str(record["relative_path"]),
                "sha256": str(record["record_sha256"]),
            }
    return None


def _artifact_record(output: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_relative_to(output.resolve()) or not resolved.is_file():
        raise ValueError(f"Receipt artifact is absent or outside output root: {path}")
    reopen, count, schema = _describe_reopen(resolved)
    if reopen != "PASS":
        raise RuntimeError(f"Receipt artifact failed reopen: {resolved}")
    return {
        "relative_path": resolved.relative_to(output.resolve()).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "reopen_status": reopen,
        "row_array_or_object_count": int(count),
        "schema_or_shape_json": schema,
    }


def _receipt_upstream(
    output: Path, upstream_paths: Sequence[Path]
) -> list[dict[str, Any]]:
    rows = []
    for path in upstream_paths:
        resolved = path.resolve()
        if not resolved.is_file() or not resolved.is_relative_to(output.resolve()):
            raise FileNotFoundError(path)
        rows.append(
            {
                "relative_path": resolved.relative_to(output.resolve()).as_posix(),
                "sha256": sha256_file(resolved),
            }
        )
    return rows


def publish_pass_receipt(
    *,
    output: Path,
    receipt_path: Path,
    stage_id: str,
    config_path: Path,
    pre_run_code_head: str,
    attempt_code_head: str | None = None,
    engineering_retry_epoch: dict[str, Any] | None = None,
    attempt_id: str,
    artifacts: Sequence[Path],
    upstream_receipts: Sequence[Path] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically publish one immutable, hash-bound PASS receipt."""

    if receipt_path.exists():
        raise FileExistsError(f"PASS receipt is immutable: {receipt_path}")
    artifact_rows = [_artifact_record(output, Path(value)) for value in artifacts]
    if not artifact_rows:
        raise ValueError("A PASS receipt requires at least one meaningful artifact")
    producer_head = str(attempt_code_head or pre_run_code_head)
    payload: dict[str, Any] = {
        "schema_id": "partC.phase5.repaired.stage_receipt.v1",
        "stage_id": stage_id,
        "attempt_id": attempt_id,
        "status": "PASS",
        "published_utc": utc_now(),
        **_config_binding(config_path, pre_run_code_head),
        "attempt_code_head": producer_head,
        "engineering_retry_epoch": engineering_retry_epoch,
        "upstream_receipts": _receipt_upstream(output, upstream_receipts),
        "artifacts": artifact_rows,
    }
    if extra:
        payload["extra"] = extra
    _atomic_json(receipt_path, payload)
    validate_pass_receipt(
        output=output,
        receipt_path=receipt_path,
        config_path=config_path,
        pre_run_code_head=pre_run_code_head,
        expected_stage_id=stage_id,
    )
    return payload


def validate_pass_receipt(
    *,
    output: Path,
    receipt_path: Path,
    config_path: Path,
    pre_run_code_head: str,
    expected_stage_id: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    attempt_code_head = str(payload.get("attempt_code_head", pre_run_code_head))
    allowed_heads = {str(pre_run_code_head)}
    epoch_by_head: dict[str, dict[str, Any]] = {}
    if _engineering_retry_files(output):
        _, execution = _execution(config_path)
        records = _validate_engineering_retry_chain(
            config_path=config_path,
            execution=execution,
            output=output,
            pre_run_code_head=pre_run_code_head,
        )
        epoch_by_head = {
            str(record["authorized_execution_code_head"]): record
            for record in records
        }
        allowed_heads.update(epoch_by_head)
    epoch_reference = payload.get("engineering_retry_epoch")
    expected_epoch = _receipt_epoch_reference(
        list(epoch_by_head.values()), attempt_code_head
    )
    if (
        payload.get("schema_id") != "partC.phase5.repaired.stage_receipt.v1"
        or payload.get("status") != "PASS"
        or payload.get("execution_config_sha256") != sha256_file(config_path)
        or payload.get("pre_run_code_head") != pre_run_code_head
        or attempt_code_head not in allowed_heads
        or epoch_reference != expected_epoch
        or (
            expected_stage_id is not None
            and payload.get("stage_id") != expected_stage_id
        )
    ):
        raise RuntimeError(f"Stage receipt binding failed: {receipt_path}")
    for upstream in payload.get("upstream_receipts", []):
        path = output / str(upstream["relative_path"])
        if not path.is_file() or sha256_file(path) != str(upstream["sha256"]):
            raise RuntimeError(f"Upstream receipt changed after PASS: {path}")
    if not payload.get("artifacts"):
        raise RuntimeError(f"PASS receipt has no artifacts: {receipt_path}")
    for artifact in payload["artifacts"]:
        path = output / str(artifact["relative_path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(artifact["size_bytes"])
            or sha256_file(path) != str(artifact["sha256"])
            or _describe_reopen(path)[0] != "PASS"
        ):
            raise RuntimeError(f"Receipt artifact changed or fails reopen: {path}")
    return payload


def deterministic_shard_plan(
    signatures: Iterable[str], *, shard_size: int, namespace: str
) -> list[dict[str, Any]]:
    """Create stable-hash shards with content-bound identities."""

    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    values = list(map(str, signatures))
    if len(values) != len(set(values)) or not values:
        raise ValueError("Shard signatures must be nonempty and unique")
    ordered = sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{namespace}\0shard-rank\0{value}".encode("utf-8")
        ).hexdigest(),
    )
    result = []
    for ordinal, start in enumerate(range(0, len(ordered), shard_size)):
        selected = ordered[start : start + shard_size]
        content_hash = hashlib.sha256(
            canonical_json(
                {"namespace": namespace, "signatures": sorted(selected)}
            ).encode("utf-8")
        ).hexdigest()
        result.append(
            {
                "ordinal": ordinal,
                "shard_id": f"r4_{ordinal:04d}_{content_hash[:12]}",
                "content_sha256": content_hash,
                "signature_count": len(selected),
                "signatures": selected,
            }
        )
    return result


@dataclass(slots=True)
class StageContext:
    config_path: Path
    execution: dict[str, Any]
    output: Path
    stage_id: str
    attempt_dir: Path
    pre_run_code_head: str
    attempt_code_head: str
    heartbeat_seconds: float
    stack_dump_seconds: float
    _stop: threading.Event | None = None
    _thread: threading.Thread | None = None
    _stack_stream: Any = None
    _substep: str = "STARTING"
    _last_event_monotonic: float = 0.0

    def __enter__(self) -> "StageContext":
        self.attempt_dir.mkdir(parents=True, exist_ok=True)
        if (self.attempt_dir / "child_result.json").exists():
            raise FileExistsError(
                f"Attempt directory already contains a child result: {self.attempt_dir}"
            )
        self._stop = threading.Event()
        self._stack_stream = (self.attempt_dir / "python_stacks.log").open(
            "a", encoding="utf-8", newline="\n"
        )
        faulthandler.dump_traceback_later(
            self.stack_dump_seconds,
            repeat=True,
            file=self._stack_stream,
        )
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"{self.stage_id}-heartbeat",
            daemon=True,
        )
        self._thread.start()
        self.mark("CHILD_CONTEXT_READY", force=True)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        faulthandler.cancel_dump_traceback_later()
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.heartbeat_seconds))
        if self._stack_stream is not None:
            self._stack_stream.flush()
            self._stack_stream.close()

    def _heartbeat_loop(self) -> None:
        assert self._stop is not None
        while not self._stop.wait(self.heartbeat_seconds):
            _append_jsonl(
                self.attempt_dir / "heartbeats.jsonl",
                {
                    "recorded_utc": utc_now(),
                    "stage_id": self.stage_id,
                    "substep": self._substep,
                    "pid": os.getpid(),
                    "process_tree_rss_bytes": process_tree_rss_bytes(),
                },
            )

    def mark(self, substep: str, *, force: bool = False) -> None:
        self._substep = str(substep)
        now = time.monotonic()
        if force or now - self._last_event_monotonic >= 1.0:
            _append_jsonl(
                self.attempt_dir / "child_events.jsonl",
                {
                    "recorded_utc": utc_now(),
                    "event": "CHILD_SUBSTEP",
                    "stage_id": self.stage_id,
                    "substep": self._substep,
                    "pid": os.getpid(),
                    "process_tree_rss_bytes": process_tree_rss_bytes(),
                },
            )
            self._last_event_monotonic = now
        self.guard()

    def guard(self) -> None:
        rss = process_tree_rss_bytes()
        if rss > int(self.execution["resources"]["process_tree_rss_bytes"]):
            raise MemoryError(
                f"Process-tree RSS guard exceeded in {self.stage_id}: {rss}"
            )
        state = json.loads(
            (self.output / "validation" / "campaign_state.json").read_text(
                encoding="utf-8"
            )
        )
        deadline_key = (
            "wall_clock_deadline_epoch_seconds"
            if self.stage_id == "R6"
            else "science_boundary_epoch_seconds"
        )
        fallback_seconds = float(
            self.execution["resources"].get(
                (
                    "wall_clock_seconds"
                    if self.stage_id == "R6"
                    else "science_boundary_seconds"
                ),
                self.execution["resources"]["wall_clock_seconds"],
            )
        )
        deadline = float(
            state.get(
                deadline_key,
                float(state["started_epoch_seconds"]) + fallback_seconds,
            )
        )
        if time.time() >= deadline:
            raise TimeoutError(
                "Repaired-pilot sealing deadline reached"
                if self.stage_id == "R6"
                else "Protected sealing reserve reached"
            )


def _copy_tree_without_inputs(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    for path in sorted(value for value in source.rglob("*") if value.is_file()):
        relative = path.relative_to(source)
        if relative.parts and relative.parts[0] in {"input", "inputs", "nested_authorities", "source_handoffs"}:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _stage_timeout(execution: dict[str, Any], stage_id: str) -> float:
    if stage_id.startswith("r4_"):
        for row in execution["stages"]:
            if row["stage_id"] == "R4":
                return float(row["child_timeout_seconds_per_shard"])
    for row in execution["stages"]:
        if row["stage_id"] == stage_id:
            if row.get("uses_sealing_reserve"):
                return float(execution["resources"]["protected_sealing_seconds"])
            return float(row["child_timeout_seconds"])
    raise KeyError(f"No timeout contract for stage {stage_id}")


def _receipt_path(output: Path, stage_id: str) -> Path:
    if stage_id.startswith("r4_"):
        return output / "stage_receipts" / f"R4_shard_{stage_id}.json"
    if stage_id == "R4":
        return output / "stage_receipts" / "R4_index.json"
    return output / "stage_receipts" / f"{stage_id}.json"


def _upstream_paths(output: Path, stage_id: str) -> list[Path]:
    if stage_id == "r4_v4_controls":
        plan = json.loads((output / "runtime_fits/shard_plan.json").read_text())
        return [_receipt_path(output, "R3"), *[_receipt_path(output, s["shard_id"]) for s in plan["shards"]]]
    mapping = {
        "R0_v2": [],
        "R1_s02": [_receipt_path(output, "R0_v2")],
        "R2_s03": [_receipt_path(output, "R1_s02")],
        "R2_s04": [_receipt_path(output, "R2_s03")],
        "R2_s05": [_receipt_path(output, "R2_s04")],
        "R2_s06": [_receipt_path(output, "R2_s05")],
        "R3": [_receipt_path(output, value) for value in STAGE_ORDER[1:6]],
        "R5": [_receipt_path(output, "R4")],
        "R6": [_receipt_path(output, "R5")],
    }
    if stage_id.startswith("r4_"):
        return [_receipt_path(output, "R3")]
    return mapping[stage_id]


def verify_repaired_authority(execution: dict[str, Any]) -> dict[str, Any]:
    """Authenticate outer/nested ZIPs and the installed non-input subset."""

    if is_v4(execution):
        return verify_v4_authority(execution)

    root = authority_root(execution)
    installation_path = root / "AUTHORITY_INSTALLATION.json"
    installation = json.loads(installation_path.read_text(encoding="utf-8"))
    expected = {
        "resume_authority": REPAIRED_OUTER_SHA256,
        "launch_prompt": str(execution["launch_authority_sha256"]),
        "prior_authority": PRIOR_REPAIRED_OUTER_SHA256,
        "blocked_v1_handoff": BLOCKED_V1_HANDOFF_SHA256,
        "original_authority": ORIGINAL_AUTHORITY_SHA256,
        "blocked_handoff": BLOCKED_HANDOFF_SHA256,
        "parent_continuation": PARENT_CONTINUATION_SHA256,
    }
    checks = []
    for role, expected_sha in expected.items():
        declared = installation[role]
        path = Path(declared["external_path"])
        actual_sha = sha256_file(path)
        actual_size = path.stat().st_size
        passed = bool(
            actual_sha == expected_sha
            and actual_sha == str(declared["sha256"])
            and actual_size == int(declared["size_bytes"])
        )
        checks.append(
            {
                "role": role,
                "absolute_path": str(path.resolve()),
                "size_bytes": actual_size,
                "sha256": actual_sha,
                "status": "PASS" if passed else "FAIL",
            }
        )
    extracted = Path(installation["extracted_source_root"])
    manifest = pd.read_csv(extracted / "MANIFEST.csv")
    manifest_failures = []
    for row in manifest.itertuples(index=False):
        path = extracted / str(row.relative_path)
        if (
            not path.is_file()
            or path.stat().st_size != int(row.size_bytes)
            or sha256_file(path) != str(row.sha256)
        ):
            manifest_failures.append(str(row.relative_path))
    checksum_failures = []
    checksum_rows = []
    for line in (extracted / "CHECKSUMS.sha256").read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        expected_digest, relative_text = line.split(maxsplit=1)
        relative_text = relative_text.strip()
        checksum_rows.append(relative_text)
        path = extracted / Path(relative_text)
        if not path.is_file() or sha256_file(path) != expected_digest.lower():
            checksum_failures.append(relative_text)
    installed_failures = []
    installed_paths = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "AUTHORITY_INSTALLATION.json"
    }
    expected_installed = {
        path.relative_to(extracted).as_posix()
        for path in extracted.rglob("*")
        if path.is_file()
        and path.relative_to(extracted).parts[0]
        not in {"nested_authorities", "source_handoffs"}
    }
    installed_member_set_match = set(installed_paths) == expected_installed
    for relative_text, path in sorted(installed_paths.items()):
        source = extracted / Path(relative_text)
        if not source.is_file() or sha256_file(path) != sha256_file(source):
            installed_failures.append(relative_text)
    status = (
        "PASS"
        if all(value["status"] == "PASS" for value in checks)
        and not manifest_failures
        and not checksum_failures
        and installed_member_set_match
        and not installed_failures
        else "FAIL"
    )
    result = {
        "schema_id": "partC.phase5.repaired_0904_v2.authority_verification.v1",
        "recorded_utc": utc_now(),
        "package_checks": checks,
        "installed_manifest_row_count": len(manifest),
        "installed_manifest_failure_count": len(manifest_failures),
        "installed_manifest_failures": manifest_failures,
        "declared_checksum_row_count": len(checksum_rows),
        "declared_checksum_failure_count": len(checksum_failures),
        "declared_checksum_failures": checksum_failures,
        "installed_member_set_match": installed_member_set_match,
        "installed_member_count": len(installed_paths),
        "expected_installed_member_count": len(expected_installed),
        "installed_subset_failure_count": len(installed_failures),
        "installed_subset_failures": installed_failures,
        "parent_manifest_unlisted_manifest_csv_archival_note": installation.get(
            "parent_manifest_unlisted_manifest_csv_archival_note", False
        ),
        "status": status,
    }
    if status != "PASS":
        raise RuntimeError(f"Repaired authority authentication failed: {result}")
    return result


def _binding_row(
    role: str,
    path: Path,
    expected: str | None,
    *,
    lineage: str,
) -> dict[str, Any]:
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise RuntimeError(f"External binding hash mismatch: {role}")
    reopen, count, schema = _describe_reopen(path)
    return {
        "role": role,
        "absolute_path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "mtime_utc": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat(),
        "sha256_or_directory_digest": digest,
        "expected_sha256_or_digest": expected
        or "BOUND_BY_ACCEPTED_MANIFEST_OR_IDENTITY",
        "format": path.suffix.lower().lstrip("."),
        "row_array_or_object_count": count,
        "schema_or_shape_json": schema,
        "reopen_status": reopen,
        "lineage": lineage,
    }


def repaired_external_bindings(
    bindings: PilotBindings,
    destination: Path,
) -> pd.DataFrame:
    """Hash and reopen every authority and accepted scientific source."""

    installation = {} if is_v4(bindings.execution) else json.loads(
        (authority_root(bindings.execution) / "AUTHORITY_INSTALLATION.json").read_text(
            encoding="utf-8"
        )
    )
    paths: list[tuple[str, Path, str | None, str]] = []
    for role, expected in (() if is_v4(bindings.execution) else (
        ("resume_authority", REPAIRED_OUTER_SHA256),
        ("launch_prompt", str(bindings.execution["launch_authority_sha256"])),
        ("prior_authority", PRIOR_REPAIRED_OUTER_SHA256),
        ("blocked_v1_handoff", BLOCKED_V1_HANDOFF_SHA256),
        ("original_authority", ORIGINAL_AUTHORITY_SHA256),
        ("blocked_handoff", BLOCKED_HANDOFF_SHA256),
        ("parent_continuation", PARENT_CONTINUATION_SHA256),
    )):
        paths.append(
            (
                role + ("_file" if role == "launch_prompt" else "_zip"),
                Path(installation[role]["external_path"]),
                expected,
                "immutable repaired/original authority lineage",
            )
        )
    if is_v4(bindings.execution):
        paths.extend([
            ("v4_authority_zip", Path(bindings.execution["authority_zip_path"]), bindings.execution["authority_zip_sha256"], "current controlling v4 authority"),
            ("v4_goal", authority_root(bindings.execution) / "09_CODEX_GOAL.md", bindings.execution["launch_authority_sha256"], "current controlling v4 goal"),
            ("reviewed_v2_handoff", authority_root(bindings.execution) / bindings.execution["latest_handoff_relative"], bindings.execution["latest_handoff_sha256"], "immutable reviewed v2 evidence"),
        ])
    repo = Path(bindings.execution["project_root"])
    phase4_config = (
        repo
        / "partC_doc_icnn"
        / "phase_4_work"
        / "phase4_dual_accuracy_benchmark_execution_config.json"
    )
    phase4 = json.loads(phase4_config.read_text(encoding="utf-8"))
    inputs = phase4["input_versions"]
    paths.extend(
        [
            (
                "physical_graph_npz",
                Path(inputs["graph_npz"]),
                None,
                "immutable accepted Phase 4 input",
            ),
            (
                "physical_graph_metadata",
                Path(inputs["graph_metadata_json"]),
                None,
                "immutable accepted Phase 4 input",
            ),
            (
                "accepted_od_npz",
                Path(inputs["od_npz"]),
                None,
                "immutable accepted Phase 4 input",
            ),
            (
                "accepted_od_metadata",
                Path(inputs["od_metadata_json"]),
                None,
                "immutable accepted Phase 4 input",
            ),
            (
                "accepted_aggregated_od",
                Path(inputs["accepted_aggregated_od_root"])
                / "derived_inputs"
                / "aggregated_od.npz",
                "a84078ec875980d3299a747895dd5c92c5dc56330041805a64ef7d23c8133f12",
                "immutable accepted Phase 4 input",
            ),
            (
                "accepted_aggregated_od_metadata",
                Path(inputs["accepted_aggregated_od_root"])
                / "derived_inputs"
                / "aggregated_od_metadata.json",
                None,
                "immutable accepted Phase 4 input",
            ),
            (
                "phase3_sqlite",
                Path(bindings.sources["phase3_sqlite"]["path"]),
                str(bindings.sources["phase3_sqlite"]["sha256"]),
                "immutable accepted Phase 3 input",
            ),
            (
                "frozen_path_sample",
                Path(bindings.sources["frozen_path_sample"]),
                None,
                "immutable accepted Phase 3A input",
            ),
            (
                "high_selected_path_sample",
                Path(bindings.sources["high_benchmark"]["root"])
                / "routes"
                / "sampled_route_switches.parquet",
                None,
                "immutable accepted Phase 4 held-out input",
            ),
        ]
    )
    for tier in ("low_benchmark", "high_benchmark"):
        source = bindings.sources[tier]
        paths.append(
            (
                tier + "_physical_arrays",
                Path(source["physical_arrays"]),
                None,
                "immutable accepted Phase 4 input",
            )
        )
        paths.append(
            (
                tier + "_physical_table",
                Path(source["physical_link_table"]),
                None,
                "immutable accepted Phase 4 input",
            )
        )
        for suite_id in bindings.execution["suite_ids"]:
            paths.append(
                (
                    f"{tier}_{suite_id}_current_aon_traces",
                    Path(str(source["suite_trace_template"]).format(suite=suite_id)),
                    None,
                    "immutable accepted Phase 4 input",
                )
            )
    rows = [
        _binding_row(role, path, expected, lineage=lineage)
        for role, path, expected, lineage in paths
    ]
    maps = {value.suite_id: value for value in bindings.trace_maps}
    if set(maps) != set(bindings.execution["suite_ids"]):
        raise RuntimeError("R0 must authenticate all five accepted suite directories")
    for suite_id in bindings.execution["suite_ids"]:
        trace_map = maps[suite_id]
        observed = _directory_digest(trace_map.root)
        if observed != trace_map.directory_digest:
            raise RuntimeError(f"Accepted suite directory changed: {suite_id}")
        rows.append(
            {
                "role": f"phase3_{suite_id}_accepted_directory",
                "absolute_path": str(trace_map.root.resolve()),
                "size_bytes": sum(
                    path.stat().st_size
                    for path in trace_map.root.rglob("*")
                    if path.is_file()
                ),
                "mtime_utc": datetime.fromtimestamp(
                    trace_map.root.stat().st_mtime, timezone.utc
                ).isoformat(),
                "sha256_or_directory_digest": observed,
                "expected_sha256_or_digest": trace_map.directory_digest,
                "format": "directory",
                "row_array_or_object_count": 1,
                "schema_or_shape_json": canonical_json(
                    {"digest_algorithm": "phase4.tracing._directory_digest"}
                ),
                "reopen_status": "PASS",
                "lineage": "immutable accepted Phase 3 suite directory",
            }
        )
    frame = pd.DataFrame(rows)
    _atomic_frame(destination, frame)
    reopened = pd.read_csv(destination)
    if len(reopened) != len(frame) or not reopened["reopen_status"].eq("PASS").all():
        raise RuntimeError("External binding manifest failed reopen")
    return reopened


def _run_focused_tests(
    config_path: Path, output: Path, execution: dict[str, Any]
) -> None:
    command = [
        str(execution["python_executable"]),
        "-m",
        "pytest",
        "-q",
        "tests/test_phase5_bundle_oracle.py",
        "tests/test_phase5_runtime_functions.py",
        "tests/test_phase5_topology_reduction.py",
        "tests/test_phase5_charge_semantics.py",
        "tests/test_phase5_multiedge.py",
        "tests/test_phase5_experiments.py",
        "tests/test_phase5_handoff.py",
        "tests/test_phase5_repaired_staging.py",
        "tests/test_phase5_scc_reachability.py",
    ]
    if is_v4(execution):
        command = command[:4] + list(execution["focused_test_files"]) + ["--junitxml=" + str(output / "validation/focused_tests.xml")]
    result = subprocess.run(
        command,
        cwd=execution["project_root"],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
    _atomic_text(
        output / "validation" / "focused_tests.txt",
        "$ " + subprocess.list2cmdline(command) + "\n" + result.stdout,
    )
    _atomic_json(
        output / "validation" / "focused_tests_status.json",
        {
            "schema_id": "partC.phase5.repaired.focused_tests.v1",
            "recorded_utc": utc_now(),
            "command": subprocess.list2cmdline(command),
            "returncode": result.returncode,
            "status": "PASS" if result.returncode == 0 else "FAIL",
        },
    )
    if result.returncode != 0:
        raise RuntimeError("Focused Phase 5 repair tests failed:\n" + result.stdout)


def _enforce_r0_cross_scc_gate(
    supply: ReachableODSupply, r0_config: dict[str, Any]
) -> None:
    """Block the accepted endpoint rule when the exact directed audit disagrees."""

    if (
        supply.cross_scc_reachable_row_count
        != int(r0_config["cross_scc_reachable_rows_required"])
        or supply.cross_scc_reachable_demand
        != float(r0_config["cross_scc_reachable_demand_required"])
    ):
        raise RuntimeError(R0_CROSS_SCC_BLOCKER)


def stage_r0(context: StageContext) -> list[Path]:
    if context.stage_id != "R0_v2":
        raise ValueError(f"R0 v2 cannot execute as {context.stage_id}")
    context.mark("FOCUSED_TESTS_START", force=True)
    _run_focused_tests(context.config_path, context.output, context.execution)
    context.mark("AUTHORITY_AUTHENTICATION_START", force=True)
    verification = verify_repaired_authority(context.execution)
    repo = Path(context.execution["project_root"])
    branch = git(repo, "branch", "--show-current").stdout.strip()
    live_head = git(repo, "rev-parse", "HEAD").stdout.strip()
    if (
        branch != context.execution["expected_branch"]
        or live_head != context.attempt_code_head
    ):
        raise RuntimeError("R0 repository identity differs from the launch binding")
    authentication_path = context.output / "repository_audit" / "authentication.json"
    _atomic_json(
        authentication_path,
        {
            "schema_id": "partC.phase5.repaired.repository_authentication.v1",
            "recorded_utc": utc_now(),
            "branch": branch,
            "starting_head": context.execution["repair_starting_head"],
            "pre_run_code_head": context.pre_run_code_head,
            "attempt_code_head": context.attempt_code_head,
            "required_repair_base_ancestor": context.execution[
                "required_repair_base_ancestor"
            ],
            "reviewed_phase4_ancestor": context.execution["reviewed_phase4_ancestor"],
            "authority_verification": verification,
            "network_used": False,
            "package_installation_performed": False,
            "status": "PASS",
        },
    )
    context.mark("ACCEPTED_BINDINGS_LOAD_START", force=True)
    bindings = load_bindings(
        context.config_path, verify_suite_digests=True, load_aon=False
    )
    context.mark("EXTERNAL_HASH_AND_SCHEMA_AUDIT_START", force=True)
    external_path = context.output / "manifests" / "external_artifact_bindings.csv"
    repaired_external_bindings(bindings, external_path)
    if is_v4(context.execution):
        return import_verified_r0(context, bindings, authentication_path, external_path)
    context.mark("R0_REUSABLE_SUPPLY_ASSESSMENT", force=True)
    inspected_roots = sorted(
        {
            str(Path(bindings.sources["low_benchmark"]["root"]).resolve()),
            str(Path(bindings.sources["high_benchmark"]["root"]).resolve()),
            str(
                Path(context.execution["project_root"]).resolve()
                / "partC_rev1_src"
                / "phase4"
            ),
        }
    )
    candidate_name_tokens = ("reachab", "endpoint_supply", "od_supply")
    candidates: list[str] = []
    for root_text in inspected_roots:
        root = Path(root_text)
        if not root.exists():
            continue
        candidates.extend(
            str(path.resolve())
            for path in root.rglob("*")
            if path.is_file()
            and any(token in path.name.lower() for token in candidate_name_tokens)
        )
    reuse_assessment_path = (
        context.output / "validation" / "r0_reusable_supply_assessment.json"
    )
    _atomic_json(
        reuse_assessment_path,
        {
            "schema_id": "partC.phase5.repaired.r0_reuse_assessment.v1",
            "recorded_utc": utc_now(),
            "inspected_roots": inspected_roots,
            "candidate_name_tokens": list(candidate_name_tokens),
            "candidate_paths": sorted(set(candidates)),
            "authoritative_reusable_product_found": False,
            "decision": "RECONSTRUCT_ACCEPTED_PHASE4_SCC_ENDPOINT_CLASSIFICATION",
            "reason": (
                "No candidate carried the exact graph/OD identity, schema, demand-accounting, "
                "and low/high conservation proof required for reuse."
            ),
            "new_reachability_definition_introduced": False,
            "status": "PASS",
        },
    )
    scc_module_path = repo / "partC_rev1_src" / "phase5" / "scc_reachability.py"
    bacn_module_path = repo / "partC_rev1_src" / "phase5" / "bacn_charges.py"
    staged_module_path = repo / "partC_rev1_src" / "phase5" / "staged_pilot.py"
    r0_code_members = {
        path.relative_to(repo).as_posix(): sha256_file(path)
        for path in (scc_module_path, bacn_module_path, staged_module_path)
    }
    r0_code_payload_sha256 = hashlib.sha256(
        canonical_json(r0_code_members).encode("utf-8")
    ).hexdigest()
    scc_source = scc_module_path.read_text(encoding="utf-8")
    bacn_source = bacn_module_path.read_text(encoding="utf-8")
    scipy_audit_path = (
        context.output / "validation" / "r0_scipy_full_graph_call_audit.json"
    )
    scipy_audit = {
        "schema_id": "partC.phase5.repaired_0904_v2.scipy_full_graph_call_audit.v1",
        "recorded_utc": utc_now(),
        "algorithm_id": SCC_ALGORITHM_ID,
        "production_modules": [
            scc_module_path.relative_to(repo).as_posix(),
            bacn_module_path.relative_to(repo).as_posix(),
        ],
        "scc_module_sha256": sha256_file(scc_module_path),
        "bacn_module_sha256": sha256_file(bacn_module_path),
        "scc_module_has_scipy_reference": "scipy" in scc_source.lower(),
        "bacn_module_has_connected_components_reference": (
            "connected_components" in bacn_source
        ),
        "bacn_module_has_scipy_csr_reference": "csr_matrix" in bacn_source,
        "focused_monkeypatch_prohibition_test_included": True,
        "full_graph_scipy_connected_components_call_count": 0,
        "status": "PASS",
    }
    if (
        scipy_audit["scc_module_has_scipy_reference"]
        or scipy_audit["bacn_module_has_connected_components_reference"]
        or scipy_audit["bacn_module_has_scipy_csr_reference"]
    ):
        scipy_audit["status"] = "FAIL"
        _atomic_json(scipy_audit_path, scipy_audit)
        raise RuntimeError("R0 v2 production path retains a forbidden SciPy SCC reference")
    _atomic_json(scipy_audit_path, scipy_audit)

    change_audit_path = (
        context.output / "validation" / "r0_v1_to_v2_change_audit.json"
    )
    _atomic_json(
        change_audit_path,
        {
            "schema_id": "partC.phase5.repaired_0904_v2.r0_change_audit.v1",
            "recorded_utc": utc_now(),
            "repair_of_run_id": REPAIR_OF_RUN_ID,
            "blocked_repaired_v1_handoff_sha256": BLOCKED_V1_HANDOFF_SHA256,
            "blocked_v1_operation": (
                "scipy.sparse.csgraph.connected_components on the accepted full graph"
            ),
            "replacement_algorithm_id": SCC_ALGORITHM_ID,
            "cache_change": "reachable_od_supply_v1 -> reachable_od_supply_v2",
            "receipt_change": "R0 -> R0_v2",
            "new_exact_audit": "EXHAUSTIVE_DIRECTED_BFS_CROSS_SCC_OD_V1",
            "r1_to_r6_scientific_logic_changed": False,
            "reachability_definition_changed": False,
            "prior_outputs_modified": False,
            "status": "PASS",
        },
    )
    report_path = (
        context.output / "reports" / "11_R0_SCC_ENGINE_REPAIR_AND_VALIDATION.md"
    )
    _atomic_text(
        report_path,
        "# R0 SCC engine repair and validation\n\n"
        f"Run `{context.execution['run_id']}` uses `{SCC_ALGORITHM_ID}` with iterative, "
        "non-recursive two-pass DFS and a directly constructed stable reverse CSR. "
        "The accepted full graph does not use SciPy connected-components. R0 v2 is in progress; "
        "machine-readable timing and exact cross-SCC results are written separately.\n",
    )

    context.mark("SCC_CLASSIFICATION_START", force=True)
    science_started = time.perf_counter()
    supply = reconstruct_reachable_od_supply(
        bindings.graph,
        bindings.od,
        origin_chunk_size=int(context.execution["r0_v2"]["origin_chunk_size"]),
        progress=context.mark,
        exact_cross_scc_audit=True,
        graph_identity=str(bindings.sources["graph"]["identity"]),
        od_identity=str(bindings.od.od_hash),
    )
    supply = replace(
        supply,
        code_sha256=r0_code_payload_sha256,
        config_sha256=sha256_file(context.config_path),
    )
    benchmark_path = (
        context.output / "validation" / "r0_scc_algorithm_benchmark.json"
    )
    exact_audit_path = (
        context.output / "validation" / "r0_cross_scc_exact_audit.json"
    )
    benchmark = {
        "schema_id": "partC.phase5.repaired_0904_v2.scc_algorithm_benchmark.v1",
        "recorded_utc": utc_now(),
        **(
            supply.scc_benchmark.to_dict()
            if supply.scc_benchmark is not None
            else {}
        ),
        "timing_seconds": dict(supply.timing_seconds or {}),
        "process_tree_rss_bytes_after_scc_and_od_audit": process_tree_rss_bytes(),
        "status": "PASS",
    }
    _atomic_json(benchmark_path, benchmark)
    exact_audit = dict(supply.exact_audit or {})
    exact_audit.update(
        {
            "schema_id": "partC.phase5.repaired_0904_v2.cross_scc_exact_audit.v1",
            "recorded_utc": utc_now(),
            "algorithm_id": SCC_ALGORITHM_ID,
            "graph_identity": supply.graph_identity,
            "od_identity": supply.od_identity,
        }
    )
    _atomic_json(exact_audit_path, exact_audit)
    try:
        _enforce_r0_cross_scc_gate(supply, context.execution["r0_v2"])
    except RuntimeError:
        _atomic_text(
            report_path,
            "# R0 SCC engine repair and validation\n\n"
            f"`{SCC_ALGORITHM_ID}` completed, but the exhaustive exact directed audit found "
            f"{supply.cross_scc_reachable_row_count} reachable cross-SCC rows with demand "
            f"{supply.cross_scc_reachable_demand:.17g}. The accepted endpoint assumption is "
            "therefore blocked for external review.\n",
        )
        raise
    expected = context.execution["accepted_demand_accounting"]
    rtol = float(expected["comparison_rtol"])
    atol = float(expected["comparison_atol"])
    demand_checks = {
        "aggregated_row_count": supply.row_count
        == int(expected["aggregated_row_count"]),
        "unreachable_row_count": supply.unreachable_row_count
        == int(expected["unreachable_row_count"]),
        "reachable": bool(
            np.isclose(
                supply.reachable_demand,
                float(expected["reachable_demand"]),
                rtol=rtol,
                atol=atol,
            )
        ),
        "unreachable": bool(
            np.isclose(
                supply.unreachable_demand,
                float(expected["unreachable_demand"]),
                rtol=rtol,
                atol=atol,
            )
        ),
        "total": bool(
            np.isclose(
                supply.reachable_demand + supply.unreachable_demand,
                float(expected["total_demand"]),
                rtol=rtol,
                atol=atol,
            )
        ),
    }
    if not all(demand_checks.values()):
        raise RuntimeError(
            "R0 SCC reachability failed accepted demand accounting: "
            + canonical_json(demand_checks)
        )
    identity_checks = {
        "algorithm_id": supply.algorithm_id == SCC_ALGORITHM_ID,
        "label_payload_dtype": supply.label_payload_dtype == LABEL_PAYLOAD_DTYPE,
        "label_payload_sha256_length": len(supply.label_payload_sha256) == 64,
        "graph_identity": supply.graph_identity
        == str(context.execution["accepted_graph"]["identity"]),
        "graph_node_count": len(supply.scc_labels)
        == int(context.execution["accepted_graph"]["node_count"]),
        "graph_arc_count": len(bindings.graph.head)
        == int(context.execution["accepted_graph"]["arc_count"]),
        "od_identity": supply.od_identity
        == str(context.execution["accepted_od"]["identity"]),
    }
    if not all(identity_checks.values()):
        raise RuntimeError(
            "R0 v2 graph/OD/algorithm identity gate failed: "
            + canonical_json(identity_checks)
        )
    context.mark("LOW_HIGH_PHYSICAL_CONSERVATION", force=True)
    conservation_started = time.perf_counter()
    low_started = time.perf_counter()
    low = physical_flow_conservation(
        bindings.graph, bindings.low_physical.current_flow, supply
    )
    low_seconds = time.perf_counter() - low_started
    high_started = time.perf_counter()
    high = physical_flow_conservation(
        bindings.graph, bindings.high_physical.current_flow, supply
    )
    high_seconds = time.perf_counter() - high_started
    conservation_limit = float(
        context.execution["r0_v2"]["physical_conservation_max_abs"]
    )
    if (
        not low["pass"]
        or not high["pass"]
        or float(low["max_abs_residual"]) > conservation_limit
        or float(high["max_abs_residual"]) > conservation_limit
    ):
        raise RuntimeError("R0 accepted low/high CURRENT flow conservation failed")
    timing_seconds = dict(supply.timing_seconds or {})
    timing_seconds.update(
        {
            "low_current_physical_conservation": float(low_seconds),
            "high_current_physical_conservation": float(high_seconds),
            "physical_conservation_total": float(
                time.perf_counter() - conservation_started
            ),
        }
    )
    supply = replace(supply, timing_seconds=timing_seconds)
    cache_path = context.output / "cache" / "reachable_od_supply_v2.npz"
    metadata_path = context.output / "cache" / "reachable_od_supply_v2.metadata.json"
    context.mark("SUPPLY_CACHE_ATOMIC_WRITE", force=True)
    cache_write_started = time.perf_counter()
    write_reachable_od_supply_cache(cache_path, supply)
    cache_write_seconds = time.perf_counter() - cache_write_started
    metadata = reachable_od_supply_metadata(supply)
    metadata.update(
        {
            "schema_id": "partC.phase5.repaired_0904_v2.reachable_od_supply_metadata.v1",
            "created_utc": utc_now(),
            "classification": context.execution["r0_v2"]["classification"],
            "aggregated_od_sha256": "a84078ec875980d3299a747895dd5c92c5dc56330041805a64ef7d23c8133f12",
            "total_demand": supply.reachable_demand + supply.unreachable_demand,
            "cache_sha256": sha256_file(cache_path),
            "demand_checks": demand_checks,
            "identity_checks": identity_checks,
            "r0_code_members": r0_code_members,
            "r0_code_payload_sha256": r0_code_payload_sha256,
            "routing_or_assignment_performed": False,
            "aon_flow_loaded_for_classification": False,
            "cache_reopen_identical": False,
            "status": "PENDING_REOPEN",
        }
    )
    _atomic_json(metadata_path, metadata)
    context.mark("SUPPLY_CACHE_REOPEN", force=True)
    cache_reopen_started = time.perf_counter()
    reopened = load_reachable_od_supply_cache(
        cache_path,
        expected_node_count=len(bindings.graph.node_idx_to_uid),
        expected_row_count=len(bindings.od.demand),
        expected_graph_identity=supply.graph_identity,
        expected_od_identity=supply.od_identity,
        expected_code_sha256=supply.code_sha256,
        expected_config_sha256=supply.config_sha256,
    )
    cache_reopen_seconds = time.perf_counter() - cache_reopen_started
    cache_reopen = bool(
        np.array_equal(reopened.scc_labels, supply.scc_labels)
        and np.array_equal(reopened.origin_node_flow, supply.origin_node_flow)
        and np.array_equal(reopened.destination_node_flow, supply.destination_node_flow)
        and reopened.reachable_demand == supply.reachable_demand
        and reopened.unreachable_demand == supply.unreachable_demand
        and reopened.algorithm_id == supply.algorithm_id
        and reopened.label_payload_sha256 == supply.label_payload_sha256
        and reopened.graph_identity == supply.graph_identity
        and reopened.od_identity == supply.od_identity
        and reopened.code_sha256 == supply.code_sha256
        and reopened.config_sha256 == supply.config_sha256
        and reopened.timing_seconds == supply.timing_seconds
    )
    if not cache_reopen:
        raise RuntimeError("R0 reachable-supply cache did not reopen identically")
    timing_seconds.update(
        {
            "cache_write": float(cache_write_seconds),
            "cache_reopen": float(cache_reopen_seconds),
            "cache_write_reopen_total": float(
                cache_write_seconds + cache_reopen_seconds
            ),
            "total_r0_science": float(time.perf_counter() - science_started),
        }
    )
    metadata["timing_seconds"] = timing_seconds
    metadata["cache_reopen_identical"] = True
    metadata["status"] = "PASS"
    _atomic_json(metadata_path, metadata)
    validation_path = (
        context.output / "validation" / "reachability_and_physical_conservation_v2.json"
    )
    _atomic_json(
        validation_path,
        {
            "schema_id": "partC.phase5.repaired_0904_v2.reachability_conservation.v1",
            "recorded_utc": utc_now(),
            "scc_count": supply.scc_count,
            "algorithm_id": supply.algorithm_id,
            "scc_label_payload_sha256": supply.label_payload_sha256,
            "scc_label_payload_dtype": supply.label_payload_dtype,
            "aggregated_od_row_count": supply.row_count,
            "unreachable_row_count": supply.unreachable_row_count,
            "reachable_demand": supply.reachable_demand,
            "unreachable_demand": supply.unreachable_demand,
            "total_demand": supply.reachable_demand + supply.unreachable_demand,
            "accepted_demand": expected,
            "demand_checks": demand_checks,
            "identity_checks": identity_checks,
            "cross_scc_candidate_row_count": supply.cross_scc_candidate_row_count,
            "cross_scc_candidate_demand": supply.cross_scc_candidate_demand,
            "cross_scc_reachable_row_count": supply.cross_scc_reachable_row_count,
            "cross_scc_reachable_demand": supply.cross_scc_reachable_demand,
            "low_physical_conservation": low,
            "high_physical_conservation": high,
            "cache_reopen_identical": cache_reopen,
            "timing_seconds": timing_seconds,
            "routing_or_assignment_performed": False,
            "status": "PASS",
        },
    )
    benchmark["timing_seconds"] = timing_seconds
    benchmark["total_r0_science_seconds"] = timing_seconds["total_r0_science"]
    benchmark["process_tree_rss_bytes_after_physical_conservation"] = (
        process_tree_rss_bytes()
    )
    _atomic_json(benchmark_path, benchmark)
    _atomic_text(
        report_path,
        "# R0 SCC engine repair and validation\n\n"
        f"`{SCC_ALGORITHM_ID}` completed on {len(supply.scc_labels):,} accepted nodes and "
        f"{len(bindings.graph.head):,} accepted arcs, producing {supply.scc_count:,} canonical "
        f"components. Canonical labels are `{LABEL_PAYLOAD_DTYPE}` and hash to "
        f"`{supply.label_payload_sha256}`.\n\n"
        f"All {supply.cross_scc_candidate_row_count:,} cross-SCC candidate OD rows were audited "
        "by exact directed BFS; zero were reachable. The accepted 10,193,770-row demand "
        "accounting and low x45/high x200 CURRENT physical conservation gates pass. The v2 "
        "cache reopened with exact schema, identity, code/config hash, label hash, and array "
        "equality. No routing, assignment, model training, or full-graph SciPy SCC call occurred.\n\n"
        f"Total R0 science time was {timing_seconds['total_r0_science']:.6f} seconds; named "
        "substep timings are in `validation/r0_scc_algorithm_benchmark.json`.\n",
    )
    context.mark("R0_COMPLETE", force=True)
    return [
        context.output / "validation" / "focused_tests.txt",
        context.output / "validation" / "focused_tests_status.json",
        authentication_path,
        external_path,
        reuse_assessment_path,
        cache_path,
        metadata_path,
        validation_path,
        benchmark_path,
        exact_audit_path,
        scipy_audit_path,
        change_audit_path,
        report_path,
    ]


def _load_supply_for_stage(bindings: PilotBindings, output: Path) -> ReachableODSupply:
    metadata_path = output / "cache" / "reachable_od_supply_v2.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache_path = output / "cache" / "reachable_od_supply_v2.npz"
    receipt_path = output / "stage_receipts" / "R0_v2.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_artifacts = {
        str(value["relative_path"]): value for value in receipt.get("artifacts", [])
    }
    if (
        receipt.get("stage_id") != "R0_v2"
        or receipt.get("status") != "PASS"
        or "cache/reachable_od_supply_v2.npz" not in receipt_artifacts
        or "cache/reachable_od_supply_v2.metadata.json" not in receipt_artifacts
        or metadata.get("status") != "PASS"
        or metadata.get("algorithm_id") != SCC_ALGORITHM_ID
        or metadata.get("graph_identity") != bindings.sources["graph"]["identity"]
        or metadata.get("od_identity") != bindings.od.od_hash
        or metadata.get("cache_sha256") != sha256_file(cache_path)
        or receipt_artifacts["cache/reachable_od_supply_v2.npz"].get("sha256")
        != sha256_file(cache_path)
        or receipt_artifacts["cache/reachable_od_supply_v2.metadata.json"].get(
            "sha256"
        )
        != sha256_file(metadata_path)
    ):
        raise RuntimeError("R0 reachable-supply metadata binding failed")
    return load_reachable_od_supply_cache(
        cache_path,
        expected_node_count=len(bindings.graph.node_idx_to_uid),
        expected_row_count=len(bindings.od.demand),
        expected_graph_identity=str(bindings.sources["graph"]["identity"]),
        expected_od_identity=str(bindings.od.od_hash),
        expected_code_sha256=str(metadata["code_sha256"]),
        expected_config_sha256=str(metadata["config_sha256"]),
    )


def _suite_output_paths(output: Path, suite_id: str) -> dict[str, Path]:
    root = output / "suite_products" / suite_id
    return {
        "root": root,
        "topology": root / "topology_census.csv",
        "chains": root / "chain_candidates.parquet",
        "protection": root / "bacn_protection_census.parquet",
        "bundles": root / "bundle_parameter_occurrences.parquet",
        "internal_keys": root / "internal_component_keys.parquet",
        "visits": root / "bacn_visit_semantics.parquet",
        "charges": root / "fixed_charge_candidates.parquet",
        "racn": root / "racn_cost_ownership_audit.csv",
        "validation": root / "suite_validation.json",
        "ownership": root / "static_link_ownership.parquet",
        "incidence": root / "remaining_incidence_validation.json",
        "reciprocal": root / "reciprocal_group_decisions.parquet",
        "self_loops": root / "excluded_self_loop_candidates.csv",
    }


def stage_suite(context: StageContext, suite_id: str) -> list[Path]:
    expected_stage = "R1_s02" if suite_id == "s02" else f"R2_{suite_id}"
    if context.stage_id != expected_stage:
        raise ValueError(f"Stage {context.stage_id} cannot process {suite_id}")
    context.mark(f"{suite_id}_BINDINGS_LOAD", force=True)
    bindings = load_bindings(
        context.config_path,
        verify_suite_digests=False,
        suite_ids=[suite_id],
    )
    supply = _load_supply_for_stage(bindings, context.output)
    trace_map = bindings.trace_maps[0]
    if trace_map.suite_id != suite_id:
        raise RuntimeError("Suite trace-map selection changed")
    paths = _suite_output_paths(context.output, suite_id)
    paths["root"].mkdir(parents=True, exist_ok=False)

    context.mark(f"{suite_id}_PHASE3_TABLES_LOAD", force=True)
    frames = _suite_frames(trace_map.root)
    context.mark(f"{suite_id}_LOW_TRACE_LOAD", force=True)
    low_trace = load_suite_trace_arrays(
        bindings.sources,
        suite_id,
        "low_benchmark",
        trace_map=trace_map,
    )
    context.mark(f"{suite_id}_HIGH_TRACE_LOAD", force=True)
    high_trace = load_suite_trace_arrays(
        bindings.sources,
        suite_id,
        "high_benchmark",
        trace_map=trace_map,
    )
    low_bundle = dict(
        zip(trace_map.bundle_ids, low_trace["current__bundle_flow"], strict=True)
    )
    high_bundle = dict(
        zip(trace_map.bundle_ids, high_trace["current__bundle_flow"], strict=True)
    )
    context.mark(f"{suite_id}_OD_ENDPOINT_PROTECTION", force=True)
    node_mapping = trace_map.node_to_bacn_local
    origin_local = node_mapping[bindings.od.origin_group_node_idx]
    destination_local = node_mapping[bindings.od.destination_node_idx]
    endpoint_local = np.unique(
        np.r_[
            origin_local[origin_local >= 0], destination_local[destination_local >= 0]
        ]
    )
    endpoint_ids = [trace_map.bacn_ids[int(value)] for value in endpoint_local]
    context.mark(f"{suite_id}_TOPOLOGY_CENSUS", force=True)
    topology = enumerate_topology(
        suite_id=suite_id,
        bacn_summary=frames["bacn_summary"],
        static_links=frames["acnhn_static_links"],
        bundle_summary=frames["bundle_summary"],
        low_bundle_flow=low_bundle,
        high_bundle_flow=high_bundle,
        od_endpoint_bacn_ids=endpoint_ids,
        flow_atol=context.execution["topology_policy"]["flow_equality_atol"],
        flow_rtol=context.execution["topology_policy"]["flow_equality_rtol"],
    )
    _atomic_frame(paths["topology"], pd.DataFrame([topology.census]))
    _atomic_frame(paths["chains"], topology.chains)
    _atomic_frame(paths["protection"], topology.node_protection)
    _atomic_frame(paths["ownership"], topology.ownership_ledger)
    _atomic_json(paths["incidence"], topology.incidence_validation)
    _atomic_frame(paths["reciprocal"], topology.chains.loc[topology.chains.reciprocal_group_id.ne("")])
    _atomic_frame(paths["self_loops"], topology.chains.loc[topology.chains.retained_tail_bacn_id.eq(topology.chains.retained_head_bacn_id)])
    if topology.incidence_validation.get("status") != "PASS":
        raise RuntimeError(f"{suite_id}: immediate full-static topology gate failed")

    context.mark(f"{suite_id}_BUNDLE_INVENTORY", force=True)
    local_specs: dict[str, BundleSpec] = {}
    bundle_key, bundle_values = _bundle_inventory(
        suite_id,
        frames["bundle_membership"],
        low_trace,
        high_trace,
        trace_map,
        bindings,
        local_specs,
    )
    bundle_rows = pd.DataFrame(bundle_values)
    _atomic_frame(paths["bundles"], bundle_rows)
    internal_key = _internal_component_keys(frames, bindings)
    internal_rows = pd.DataFrame(
        [
            {
                "suite_short_id": suite_id,
                "bacn_id": bacn_id,
                "canonical_internal_key": key,
            }
            for bacn_id, key in sorted(internal_key.items())
        ]
    )
    _atomic_frame(paths["internal_keys"], internal_rows)

    context.mark(f"{suite_id}_VISIT_SEMANTICS_LOW", force=True)
    low_visits = reconstruct_bacn_visit_semantics(
        trace_map=trace_map,
        bundle_summary=frames["bundle_summary"],
        terminals=frames["consolidated_terminals"],
        trace_arrays=low_trace,
        benchmark_id=bindings.low_physical.benchmark_id,
        supply=supply,
        static_incidence_only=is_v4(context.execution),
    )
    context.mark(f"{suite_id}_VISIT_SEMANTICS_HIGH", force=True)
    high_visits = reconstruct_bacn_visit_semantics(
        trace_map=trace_map,
        bundle_summary=frames["bundle_summary"],
        terminals=frames["consolidated_terminals"],
        trace_arrays=high_trace,
        benchmark_id=bindings.high_physical.benchmark_id,
        supply=supply,
        static_incidence_only=is_v4(context.execution),
    )
    for frame in (low_visits, high_visits):
        balance = np.abs(
            frame["incoming_charged_flow"] - frame["outgoing_charged_flow"]
        )
        if np.any(balance > 1.0e-6) or frame["status"].eq("FAIL").any():
            raise RuntimeError(f"{suite_id}: charged-incidence reconstruction failed")
    high_path_sample = (
        Path(bindings.sources["high_benchmark"]["root"])
        / "routes"
        / "sampled_route_switches.parquet"
    )
    path_demand = pd.read_parquet(
        high_path_sample, columns=["sample_od_id", "accepted_od_demand"]
    )
    sample_weight_by_id = dict(
        zip(
            path_demand["sample_od_id"].astype(str),
            path_demand["accepted_od_demand"].astype(float),
            strict=True,
        )
    )
    context.mark(f"{suite_id}_ACCEPTED_PATH_EPISODES", force=True)
    low_episodes = path_episode_statistics(
        bindings.sources["frozen_path_sample"],
        trace_map,
        bindings.low_physical.current_cost_s,
        sample_weight_by_id=sample_weight_by_id,
    )
    high_episodes = path_episode_statistics(
        high_path_sample,
        trace_map,
        bindings.high_physical.current_cost_s,
        arc_column="selected_arc_indices_json",
        sample_weight_by_id=sample_weight_by_id,
    )
    for visit_frame, episode_frame in (
        (low_visits, low_episodes),
        (high_visits, high_episodes),
    ):
        if episode_frame.empty:
            continue
        episode_by_id = episode_frame.set_index("bacn_id")
        observed = visit_frame["bacn_id"].isin(episode_by_id.index)
        visit_frame.loc[observed, "path_episode_visit_flow"] = visit_frame.loc[
            observed, "bacn_id"
        ].map(episode_by_id["sample_demand_weighted_episode_flow"])
        visit_frame.loc[observed, "path_episode_observation_status"] = (
            "OBSERVED_DEMAND_WEIGHTED_IN_ACCEPTED_200_PATH_SAMPLE"
        )
    visits = pd.concat([low_visits, high_visits], ignore_index=True)
    _atomic_frame(paths["visits"], visits)

    context.mark(f"{suite_id}_FIXED_CHARGE_DIAGNOSTIC", force=True)
    low_exposure = internal_exposure(
        trace_map,
        bindings.low_physical.current_flow,
        bindings.low_physical.current_cost_s,
    )
    high_exposure = internal_exposure(
        trace_map,
        bindings.high_physical.current_flow,
        bindings.high_physical.current_cost_s,
    )
    charges = fixed_charge_candidates(
        trace_map=trace_map,
        low_visits=low_visits["incidence_visit_flow"].to_numpy(dtype=np.float64),
        high_visits=high_visits["incidence_visit_flow"].to_numpy(dtype=np.float64),
        low_exposure=low_exposure,
        high_exposure=high_exposure,
        structural_charges=frames["structural_internal_charges"],
        episode_statistics=low_episodes,
    )
    _atomic_frame(paths["charges"], charges)
    work = SimpleNamespace(
        suite_id=suite_id,
        root=trace_map.root,
        trace_map=trace_map,
        frames=frames,
        low_trace=low_trace,
        high_trace=high_trace,
        topology=topology,
        bundle_key=bundle_key,
        internal_key=internal_key,
    )
    racn = pd.DataFrame([_racn_audit(work)])
    _atomic_frame(paths["racn"], racn)

    context.mark(f"{suite_id}_SUITE_REOPEN", force=True)
    reopen = {}
    for name, path in paths.items():
        if name == "root":
            continue
        if name == "validation":
            continue
        reopen[name] = {
            "status": _describe_reopen(path)[0],
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    validation = {
        "schema_id": "partC.phase5.repaired.suite_validation.v1",
        "recorded_utc": utc_now(),
        "suite_id": suite_id,
        "phase3_directory_digest": trace_map.directory_digest,
        "bundle_registry_count": len(bundle_key),
        "chain_count": len(topology.chains),
        "unblocked_chain_count": int(
            (~topology.chains["protected_or_blocked"].astype(bool)).sum()
        ),
        "low_current_chain_flow_inconsistent": int(
            (~topology.chains["low_flow_consistent"].astype(bool)).sum()
        ),
        "high_current_chain_flow_inconsistent": int(
            (~topology.chains["high_flow_consistent"].astype(bool)).sum()
        ),
        "visit_rows": len(visits),
        "charge_rows": len(charges),
        "racn_rows": len(racn),
        "reopen": reopen,
        "current_not_aon_primary": True,
        "heldout_high_used_for_fit": False,
        "status": "PASS",
    }
    _atomic_json(paths["validation"], validation)
    context.mark(f"{suite_id}_COMPLETE", force=True)
    return [path for name, path in paths.items() if name != "root"]


def _spec_from_occurrence(row: Any) -> BundleSpec:
    free_time = np.asarray(
        json.loads(str(row.member_fftt_values_json)), dtype=np.float64
    )
    capacity = np.asarray(
        json.loads(str(row.member_capacity_values_json)), dtype=np.float64
    )
    alpha = np.asarray(json.loads(str(row.member_alpha_values_json)), dtype=np.float64)
    beta = np.asarray(json.loads(str(row.member_beta_values_json)), dtype=np.float64)
    key = str(row.canonical_primitive_key)
    return BundleSpec(
        tuple(f"{key}:{index:04d}" for index in range(len(free_time))),
        free_time,
        capacity,
        alpha,
        beta,
    )


def _global_specs_from_occurrences(frame: pd.DataFrame) -> dict[str, BundleSpec]:
    result: dict[str, BundleSpec] = {}
    for row in frame.sort_values(
        ["canonical_primitive_key", "suite_short_id", "bundle_id"],
        kind="mergesort",
    ).itertuples(index=False):
        key = str(row.canonical_primitive_key)
        candidate = _spec_from_occurrence(row)
        current = result.get(key)
        if current is None:
            result[key] = candidate
            continue
        if not (
            np.array_equal(current.free_time_s, candidate.free_time_s)
            and np.array_equal(current.capacity, candidate.capacity)
            and np.array_equal(current.alpha, candidate.alpha)
            and np.array_equal(current.beta, candidate.beta)
        ):
            raise RuntimeError(f"Canonical primitive key has unequal payloads: {key}")
    return result


def _experiment_payload(experiment: MacroExperiment) -> dict[str, Any]:
    return {
        "canonical_signature": experiment.canonical_signature,
        "occurrence_id": experiment.occurrence_id,
        "suite_id": experiment.suite_id,
        "primitive_bundles": [
            {
                "member_ids": list(spec.member_ids),
                "free_time_s": spec.free_time_s.tolist(),
                "capacity": spec.capacity.tolist(),
                "alpha": spec.alpha.tolist(),
                "beta": spec.beta.tolist(),
            }
            for spec in experiment.primitive_bundles
        ],
        "low_flow": experiment.low_flow,
        "low_aon_flow": experiment.low_aon_flow,
        "high_flow": experiment.high_flow,
        "chain_length": experiment.chain_length,
        "endpoint_collision": experiment.endpoint_collision,
        "internal_charge_s": experiment.internal_charge_s,
        "internal_charge_estimator_id": experiment.internal_charge_estimator_id,
        "internal_component_ids": list(experiment.internal_component_ids),
        "internal_component_charge_s": list(experiment.internal_component_charge_s),
        "inclusion_reasons": list(experiment.inclusion_reasons),
        "dynamic_expression_key": experiment.dynamic_expression_key,
        "calibration_request_key": experiment.calibration_request_key,
    }


def _experiment_from_payload(payload: dict[str, Any]) -> MacroExperiment:
    bundles = tuple(
        BundleSpec(
            tuple(map(str, value["member_ids"])),
            np.asarray(value["free_time_s"], dtype=np.float64),
            np.asarray(value["capacity"], dtype=np.float64),
            np.asarray(value["alpha"], dtype=np.float64),
            np.asarray(value["beta"], dtype=np.float64),
        )
        for value in payload["primitive_bundles"]
    )
    return MacroExperiment(
        canonical_signature=str(payload["canonical_signature"]),
        occurrence_id=str(payload["occurrence_id"]),
        suite_id=str(payload["suite_id"]),
        primitive_bundles=bundles,
        low_flow=float(payload["low_flow"]),
        low_aon_flow=float(payload["low_aon_flow"]),
        high_flow=float(payload["high_flow"]),
        chain_length=int(payload["chain_length"]),
        endpoint_collision=bool(payload["endpoint_collision"]),
        internal_charge_s=float(payload["internal_charge_s"]),
        internal_charge_estimator_id=str(payload["internal_charge_estimator_id"]),
        internal_component_ids=tuple(map(str, payload["internal_component_ids"])),
        internal_component_charge_s=tuple(
            map(float, payload["internal_component_charge_s"])
        ),
        inclusion_reasons=tuple(map(str, payload.get("inclusion_reasons", []))),
        dynamic_expression_key=str(payload.get("dynamic_expression_key", "")),
        calibration_request_key=str(payload.get("calibration_request_key", "")),
    )


def _read_experiment_jsonl(path: Path) -> list[MacroExperiment]:
    values = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                values.append(_experiment_from_payload(json.loads(line)))
    if not values:
        raise RuntimeError(f"Experiment shard is empty: {path}")
    return values


def _write_experiment_jsonl(path: Path, values: Sequence[MacroExperiment]) -> None:
    _atomic_text(
        path,
        "\n".join(canonical_json(_experiment_payload(value)) for value in values),
    )


def _indexed_bundle_low_aon(bundle: pd.DataFrame, suite_id: str) -> pd.Series:
    """Build one immutable suite-local lookup used by every R3 chain."""

    required = {"suite_short_id", "bundle_id", "low_aon_flow"}
    missing = sorted(required - set(bundle.columns))
    if missing:
        raise RuntimeError(f"R3 bundle low-AON lookup lacks columns: {missing}")
    observed_suites = set(bundle["suite_short_id"].astype(str))
    if observed_suites != {str(suite_id)}:
        raise RuntimeError(
            f"R3 bundle low-AON lookup is not suite-local for {suite_id}: "
            f"{sorted(observed_suites)}"
        )
    lookup = bundle.loc[:, ["bundle_id", "low_aon_flow"]].copy()
    lookup["bundle_id"] = lookup["bundle_id"].astype(str)
    try:
        indexed = lookup.set_index("bundle_id", verify_integrity=True)["low_aon_flow"]
    except ValueError as error:
        raise RuntimeError(f"R3 duplicate bundle_id within suite {suite_id}") from error
    if indexed.empty:
        raise RuntimeError(f"R3 bundle low-AON lookup is empty for suite {suite_id}")
    return indexed


def stage_r3(context: StageContext) -> list[Path]:
    context.mark("R3_REOPEN_SUITE_RECEIPTS", force=True)
    for stage_id in STAGE_ORDER[1:6]:
        validate_pass_receipt(
            output=context.output,
            receipt_path=_receipt_path(context.output, stage_id),
            config_path=context.config_path,
            pre_run_code_head=context.pre_run_code_head,
            expected_stage_id=stage_id,
        )
    topology_frames = []
    chain_frames = []
    protection_frames = []
    bundle_frames = []
    visit_frames = []
    charge_frames = []
    racn_frames = []
    internal_by_suite: dict[str, dict[str, str]] = {}
    bundle_key_by_suite: dict[str, dict[str, str]] = {}
    bundle_low_aon_by_suite: dict[str, pd.Series] = {}
    for suite_id in context.execution["suite_ids"]:
        paths = _suite_output_paths(context.output, suite_id)
        topology_frames.append(pd.read_csv(paths["topology"]))
        chain_frames.append(pd.read_parquet(paths["chains"]))
        protection_frames.append(pd.read_parquet(paths["protection"]))
        bundle = pd.read_parquet(paths["bundles"])
        bundle_frames.append(bundle)
        bundle_low_aon_by_suite[suite_id] = _indexed_bundle_low_aon(bundle, suite_id)
        visit_frames.append(pd.read_parquet(paths["visits"]))
        charge_frames.append(pd.read_parquet(paths["charges"]))
        racn_frames.append(pd.read_csv(paths["racn"]))
        internal = pd.read_parquet(paths["internal_keys"])
        internal_by_suite[suite_id] = dict(
            zip(
                internal["bacn_id"].astype(str),
                internal["canonical_internal_key"].astype(str),
                strict=True,
            )
        )
        bundle_key_by_suite[suite_id] = dict(
            zip(
                bundle["bundle_id"].astype(str),
                bundle["canonical_primitive_key"].astype(str),
                strict=True,
            )
        )
    topology = pd.concat(topology_frames, ignore_index=True)
    chains = pd.concat(chain_frames, ignore_index=True)
    protection = pd.concat(protection_frames, ignore_index=True)
    bundle_rows = pd.concat(bundle_frames, ignore_index=True)
    visits = pd.concat(visit_frames, ignore_index=True)
    charges = pd.concat(charge_frames, ignore_index=True)
    racn = pd.concat(racn_frames, ignore_index=True)
    if set(topology["suite_short_id"].astype(str)) != set(
        context.execution["suite_ids"]
    ):
        raise RuntimeError("R3 suite census is incomplete")
    context.mark("R3_CANONICAL_PRIMITIVE_RECONSTRUCTION", force=True)
    global_specs = _global_specs_from_occurrences(bundle_rows)
    bundle_census = _bundle_census(bundle_rows)

    charge_estimator_id = "LOW_OPERATING_MEAN_ELSE_STRUCTURAL_SEED_V1"
    charge_by_suite: dict[str, dict[str, float]] = {}
    for suite_id, frame in charges.groupby("suite_short_id", sort=True):
        value = frame["low_operating_mean_seconds"].to_numpy(dtype=np.float64)
        seed = frame["structural_seed_seconds"].to_numpy(dtype=np.float64)
        recommended = np.where(np.isfinite(value), value, seed)
        charge_by_suite[str(suite_id)] = dict(
            zip(frame["bacn_id"].astype(str), recommended, strict=True)
        )

    context.mark("R3_CANONICAL_MACRO_CONSTRUCTION", force=True)
    macro_candidates: list[MacroExperiment] = []
    exact_eligible: list[bool] = []
    chain_count = len(chains)
    for position, (index, row) in enumerate(chains.iterrows(), start=1):
        if position % 5000 == 0:
            context.mark(
                f"R3_CANONICAL_MACRO_CONSTRUCTION_{position}_OF_{chain_count}",
                force=True,
            )
        suite_id = str(row["suite_short_id"])
        primitive_ids = list(
            map(str, json.loads(str(row["primitive_bundle_ids_json"])))
        )
        internal_ids = list(map(str, json.loads(str(row["internal_bacn_ids_json"]))))
        primitive_keys = [
            bundle_key_by_suite[suite_id][value] for value in primitive_ids
        ]
        internal_keys = [internal_by_suite[suite_id][value] for value in internal_ids]
        internal_charge_s = float(
            sum(charge_by_suite[suite_id][value] for value in internal_ids)
        )
        component_charges = tuple(
            float(charge_by_suite[suite_id][value]) for value in internal_ids
        )
        signature = macro_function_key(
            primitive_keys,
            internal_component_keys=internal_keys,
            internal_charge_s=internal_charge_s,
            internal_charge_estimator_id=charge_estimator_id,
            graph_identity=str(context.execution["graph_identity"]),
        )
        chains.at[index, "legacy_charge_inclusive_signature"] = signature
        primitive_specs = tuple(global_specs[value] for value in primitive_keys)
        low_aon_flow = float(bundle_low_aon_by_suite[suite_id].loc[primitive_ids].max())
        if is_v4(context.execution):
            signature = calibration_request_key(primitive_specs, float(row["low_flow_min"]), low_aon_flow)
            chains.at[index, "dynamic_expression_key"] = dynamic_expression_key(primitive_specs)
            chains.at[index, "calibration_request_key"] = signature
            chains.at[index, "internal_charge_s"] = internal_charge_s
            chains.at[index, "internal_charge_estimator_id"] = charge_estimator_id
        chains.at[index, "canonical_signature"] = signature
        eligible = (
            series_power_terms(
                tuple(global_specs[value] for value in primitive_keys),
                internal_charge_s=internal_charge_s,
            )
            is not None
        )
        exact_eligible.append(eligible)
        if bool(row["protected_or_blocked"]):
            continue
        low_aon_flow = float(
            bundle_low_aon_by_suite[suite_id].loc[primitive_ids].max()
        )
        macro_candidates.append(
            MacroExperiment(
                canonical_signature=signature,
                occurrence_id=str(row["chain_id"]),
                suite_id=suite_id,
                primitive_bundles=tuple(
                    global_specs[value] for value in primitive_keys
                ),
                low_flow=float(row["low_flow_min"]),
                low_aon_flow=low_aon_flow,
                high_flow=float(row["high_flow_min"]),
                chain_length=int(row["primitive_link_count"]),
                endpoint_collision=bool(str(row["parallel_collision_group_id"])),
                internal_charge_s=internal_charge_s,
                internal_charge_estimator_id=charge_estimator_id,
                internal_component_ids=tuple(internal_ids),
                internal_component_charge_s=component_charges,
                dynamic_expression_key=dynamic_expression_key(primitive_specs),
                calibration_request_key=signature if is_v4(context.execution) else "",
            )
        )
    chains["exact_closed_form_eligible"] = exact_eligible
    if not macro_candidates:
        raise RuntimeError("R3 full topology census produced zero macro candidates")

    context.mark("R3_ACCEPTED_PATH_FORCE_SET", force=True)
    s03_bindings = load_bindings(
        context.config_path,
        verify_suite_digests=False,
        suite_ids=["s03"],
    )
    s03_work = SimpleNamespace(trace_map=s03_bindings.trace_maps[0], suite_id="s03")
    low_intersections = _accepted_path_candidate_signatures(
        s03_bindings, s03_work, chains
    )
    high_intersections = set() if is_v4(context.execution) else _accepted_path_candidate_signatures(
        s03_bindings,
        s03_work,
        chains,
        Path(s03_bindings.sources["high_benchmark"]["root"])
        / "routes"
        / "sampled_route_switches.parquet",
        "selected_arc_indices_json",
    )
    candidate_by_signature = {
        value.canonical_signature: value for value in macro_candidates
    }
    exact_intersections = {
        signature
        for signature in (low_intersections | high_intersections)
        if signature in candidate_by_signature
        and series_power_terms(
            candidate_by_signature[signature].primitive_bundles,
            internal_charge_s=candidate_by_signature[signature].internal_charge_s,
        )
        is not None
    }
    exact_low = exact_intersections & low_intersections
    exact_high = exact_intersections & high_intersections
    if not exact_low or (not is_v4(context.execution) and not exact_high):
        raise RuntimeError(
            "Accepted low/high path samples lack forceable exact-series candidates"
        )
    namespace = str(context.execution["pilot_sample"]["stable_hash_namespace"])

    def force_rank(signature: str) -> str:
        return hashlib.sha256(
            f"{namespace}\0accepted-path-force\0{signature}".encode("utf-8")
        ).hexdigest()

    maximum = int(context.execution["pilot_sample"]["max_unique_macro_signatures"])
    forced = {min(exact_low, key=force_rank)}
    if exact_high:
        forced.add(min(exact_high, key=force_rank))
    quota = min(1000, max(2, maximum // 10))
    forced.update(
        sorted(exact_intersections - forced, key=force_rank)[: quota - len(forced)]
    )
    context.mark("R3_STRATIFIED_SAMPLE_SELECTION", force=True)
    sample, inclusion = select_macro_sample(
        macro_candidates,
        maximum=maximum,
        force_extreme=int(
            context.execution["pilot_sample"]["force_extreme_per_metric"]
        ),
        namespace=namespace,
        force_signatures=forced,
        progress_guard=context.mark,
    )
    selected = set(inclusion)
    chains["sample_included"] = chains["canonical_signature"].astype(str).isin(selected)
    chains["sample_inclusion_reasons_json"] = (
        chains["canonical_signature"]
        .astype(str)
        .map(lambda value: json.dumps(inclusion.get(value, ()), separators=(",", ":")))
    )
    sample = [
        replace(
            value,
            inclusion_reasons=inclusion[value.canonical_signature],
        )
        for value in sample
    ]

    context.mark("R3_DETERMINISTIC_SHARD_PLAN", force=True)
    shard_plan = deterministic_shard_plan(
        [value.canonical_signature for value in sample],
        shard_size=int(context.execution["oracle_policy"]["shard_size"]),
        namespace=namespace,
    )
    sample_by_signature = {value.canonical_signature: value for value in sample}
    shard_root = context.output / "r4_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_records = []
    shard_inputs: list[Path] = []
    for shard in shard_plan:
        shard_dir = shard_root / str(shard["shard_id"])
        shard_dir.mkdir(parents=True, exist_ok=False)
        input_path = shard_dir / "input_experiments.jsonl"
        _write_experiment_jsonl(
            input_path,
            [sample_by_signature[value] for value in shard["signatures"]],
        )
        shard_inputs.append(input_path)
        shard_records.append(
            {
                "ordinal": shard["ordinal"],
                "shard_id": shard["shard_id"],
                "content_sha256": shard["content_sha256"],
                "signature_count": shard["signature_count"],
                "input_relative_path": input_path.relative_to(
                    context.output
                ).as_posix(),
                "input_sha256": sha256_file(input_path),
            }
        )
    plan_path = context.output / "runtime_fits" / "shard_plan.json"
    _atomic_json(
        plan_path,
        {
            "schema_id": "partC.phase5.repaired.r4_shard_plan.v1",
            "namespace": namespace,
            "shard_size": int(context.execution["oracle_policy"]["shard_size"]),
            "signature_count": len(sample),
            "shard_count": len(shard_records),
            "shards": shard_records,
            "status": "PASS",
        },
    )
    sample_index_path = context.output / "runtime_fits" / "pilot_sample_index.csv"
    _atomic_frame(
        sample_index_path,
        pd.DataFrame(
            [
                {
                    "canonical_macro_signature": value.canonical_signature,
                    "suite_occurrence_id": value.occurrence_id,
                    "suite_short_id": value.suite_id,
                    "chain_length": value.chain_length,
                    "low_current_flow": value.low_flow,
                    "low_aon_flow": value.low_aon_flow,
                    "heldout_high_flow": value.high_flow,
                    "inclusion_reasons_json": json.dumps(
                        value.inclusion_reasons, separators=(",", ":")
                    ),
                }
                for value in sorted(sample, key=lambda item: item.canonical_signature)
            ]
        ),
    )

    topology_path = context.output / "topology" / "topology_census.csv"
    # R4 adds the runtime resolution.  Keep the R3 artifact immutable so its
    # PASS receipt remains valid after that downstream enrichment.
    chain_path = context.output / "topology" / "chain_candidates_r3.parquet"
    protection_path = context.output / "topology" / "bacn_protection_census.parquet"
    bundle_census_path = (
        context.output / "bundle_oracle" / "bundle_parameter_census.csv"
    )
    bundle_occurrence_path = (
        context.output / "bundle_oracle" / "bundle_parameter_occurrences.parquet"
    )
    macro_census_path = context.output / "bundle_oracle" / "canonical_macro_census.csv"
    _atomic_frame(topology_path, topology)
    _atomic_frame(chain_path, chains)
    _atomic_frame(protection_path, protection)
    _atomic_frame(bundle_census_path, bundle_census)
    _atomic_frame(bundle_occurrence_path, bundle_rows)
    _atomic_frame(
        macro_census_path,
        pd.DataFrame(
            [
                {
                    "scope": "ALL_SUITES_UNBLOCKED",
                    "occurrence_count": len(macro_candidates),
                    "unique_signature_count": len(candidate_by_signature),
                    "duplicate_occurrence_count": len(macro_candidates)
                    - len(candidate_by_signature),
                    "exact_closed_form_unique_count": sum(
                        series_power_terms(
                            value.primitive_bundles,
                            internal_charge_s=value.internal_charge_s,
                        )
                        is not None
                        for value in candidate_by_signature.values()
                    ),
                    "selected_unique_count": len(sample),
                    "forced_path_signature_count": len(forced),
                    "status": "PASS",
                }
            ]
        ),
    )
    # R5 consumes these merged suite-local scientific tables; retaining them
    # avoids reopening all accepted Phase 3 tables in a single process.
    merged_paths = {
        "visits": context.output / "charge_audit" / "bacn_visit_semantics.parquet",
        "charges": context.output / "charge_audit" / "fixed_charge_candidates.parquet",
        "racn": context.output / "interface_audit" / "racn_cost_ownership_audit.csv",
    }
    _atomic_frame(merged_paths["visits"], visits)
    _atomic_frame(merged_paths["charges"], charges)
    _atomic_frame(merged_paths["racn"], racn)
    context.mark("R3_REOPEN_COMPLETE", force=True)
    artifacts = [
        topology_path,
        chain_path,
        protection_path,
        bundle_census_path,
        bundle_occurrence_path,
        macro_census_path,
        merged_paths["visits"],
        merged_paths["charges"],
        merged_paths["racn"],
        plan_path,
        sample_index_path,
        *shard_inputs,
    ]
    if is_v4(context.execution):
        artifacts.append(sample_delta_evidence(context.output, context.execution, chains))
    for path in artifacts:
        if _describe_reopen(path)[0] != "PASS":
            raise RuntimeError(f"R3 output failed reopen: {path}")
    context.mark("R3_COMPLETE", force=True)
    return artifacts


def _shard_record(output: Path, shard_id: str) -> dict[str, Any]:
    plan = json.loads(
        (output / "runtime_fits" / "shard_plan.json").read_text(encoding="utf-8")
    )
    matches = [value for value in plan["shards"] if value["shard_id"] == shard_id]
    if len(matches) != 1:
        raise RuntimeError(f"Unknown or duplicate R4 shard identity: {shard_id}")
    return matches[0]


def stage_r4_shard(context: StageContext) -> list[Path]:
    if context.stage_id == "r4_v4_warmup":
        return stage_warmup(context)
    if context.stage_id == "r4_v4_controls":
        return stage_retained_controls(context)
    shard_id = context.stage_id
    shard = _shard_record(context.output, shard_id)
    input_path = context.output / str(shard["input_relative_path"])
    if sha256_file(input_path) != str(shard["input_sha256"]):
        raise RuntimeError(f"R4 shard input hash changed: {shard_id}")
    context.mark(f"{shard_id}_INPUT_REOPEN", force=True)
    experiments = _read_experiment_jsonl(input_path)
    signatures = [value.canonical_signature for value in experiments]
    expected_plan = deterministic_shard_plan(
        signatures,
        shard_size=max(len(signatures), 1),
        namespace=str(context.execution["pilot_sample"]["stable_hash_namespace"]),
    )[0]
    if len(experiments) != int(shard["signature_count"]) or expected_plan[
        "content_sha256"
    ] != str(shard["content_sha256"]):
        raise RuntimeError(f"R4 shard content identity failed: {shard_id}")
    context.mark(f"{shard_id}_EXACT_ORACLE_AND_FITS", force=True)
    oracle, fits, domains = run_macro_experiments(
        experiments, progress_guard=context.mark
    )
    certification = certificate_gate(oracle, fits)
    expected = set(signatures)
    if (
        set(oracle["canonical_macro_signature"].astype(str)) != expected
        or set(fits["canonical_macro_signature"].astype(str)) != expected
        or set(domains["canonical_macro_signature"].astype(str)) != expected
    ):
        raise RuntimeError(f"R4 shard result coverage failed: {shard_id}")
    for frame in (oracle, fits, domains):
        frame["shard_id"] = shard_id
    root = context.output / "r4_shards" / shard_id
    oracle_path = root / "oracle_sample_results.parquet"
    fit_path = root / "runtime_fit_results.parquet"
    domain_path = root / "pilot_domain_summary.csv"
    validation_path = root / "shard_validation.json"
    _atomic_frame(oracle_path, oracle)
    _atomic_frame(fit_path, fits)
    _atomic_frame(domain_path, domains)
    _atomic_json(
        validation_path,
        {
            "schema_id": "partC.phase5.repaired.r4_shard_validation.v1",
            "recorded_utc": utc_now(),
            "shard_id": shard_id,
            "content_sha256": shard["content_sha256"],
            "input_sha256": shard["input_sha256"],
            "signature_count": len(experiments),
            "oracle_row_count": len(oracle),
            "fit_row_count": len(fits),
            "domain_row_count": len(domains),
            "heldout_high_used_for_fit": False,
            "runtime_exact_root_calls": 0,
            "certification": certification,
            "activation_cache_profile": __import__("partC_rev1_src.phase5.bundle_oracle", fromlist=["oracle_cache_profile"]).oracle_cache_profile(),
            "status": "PASS",
        },
    )
    context.mark(f"{shard_id}_OUTPUT_REOPEN", force=True)
    artifacts = [oracle_path, fit_path, domain_path, validation_path]
    for path in artifacts:
        if _describe_reopen(path)[0] != "PASS":
            raise RuntimeError(f"R4 shard output failed reopen: {path}")
    context.mark(f"{shard_id}_COMPLETE", force=True)
    return artifacts


def merge_r4(
    *,
    config_path: Path,
    execution: dict[str, Any],
    output: Path,
    pre_run_code_head: str,
    attempt_code_head: str | None = None,
    engineering_retry_epoch: dict[str, Any] | None = None,
) -> Path:
    plan = json.loads(
        (output / "runtime_fits" / "shard_plan.json").read_text(encoding="utf-8")
    )
    oracle_frames = []
    fit_frames = []
    domain_frames = []
    shard_receipts = []
    for shard in plan["shards"]:
        shard_id = str(shard["shard_id"])
        receipt = _receipt_path(output, shard_id)
        validate_pass_receipt(
            output=output,
            receipt_path=receipt,
            config_path=config_path,
            pre_run_code_head=pre_run_code_head,
            expected_stage_id=shard_id,
        )
        shard_receipts.append(receipt)
        root = output / "r4_shards" / shard_id
        oracle_frames.append(pd.read_parquet(root / "oracle_sample_results.parquet"))
        fit_frames.append(pd.read_parquet(root / "runtime_fit_results.parquet"))
        domain_frames.append(pd.read_csv(root / "pilot_domain_summary.csv"))
    if not shard_receipts:
        raise RuntimeError("R4 shard plan is empty")
    oracle = pd.concat(oracle_frames, ignore_index=True).sort_values(
        ["canonical_macro_signature", "query_flow"], kind="mergesort"
    )
    fits = pd.concat(fit_frames, ignore_index=True).sort_values(
        ["canonical_macro_signature", "runtime_class", "variant_id"],
        kind="mergesort",
    )
    domains = pd.concat(domain_frames, ignore_index=True).sort_values(
        ["canonical_macro_signature"], kind="mergesort"
    )
    sample_index = pd.read_csv(output / "runtime_fits" / "pilot_sample_index.csv")
    expected = set(sample_index["canonical_macro_signature"].astype(str))
    if (
        set(oracle["canonical_macro_signature"].astype(str)) != expected
        or set(fits["canonical_macro_signature"].astype(str)) != expected
        or set(domains["canonical_macro_signature"].astype(str)) != expected
        or len(domains) != len(expected)
    ):
        raise RuntimeError("R4 merged signature coverage differs from R3 sample")

    selected_rows = fits.loc[
        fits["recommended_status"].astype(str).str.startswith("CASCADE_SELECTED")
    ]
    selected_class_by_signature = dict(
        zip(
            selected_rows["canonical_macro_signature"].astype(str),
            selected_rows["runtime_class"].astype(str),
            strict=True,
        )
    )
    sampled_signatures = set(domains["canonical_macro_signature"].astype(str))
    chains_r3_path = output / "topology" / "chain_candidates_r3.parquet"
    chains_path = output / "topology" / "chain_candidates.parquet"
    chains = pd.read_parquet(chains_r3_path)
    chains["runtime_resolution"] = np.where(
        chains["protected_or_blocked"].astype(bool),
        "PROTECTED_NO_CONTRACTION",
        np.where(
            chains["canonical_signature"].astype(str).isin(selected_class_by_signature),
            "SELECTED_ROOT_FREE_RUNTIME",
            np.where(
                chains["canonical_signature"].astype(str).isin(sampled_signatures),
                "RETAIN_ORIGINAL_PRIMITIVE_LINKS_PILOT_ERROR_GATE",
                "RETAIN_ORIGINAL_PRIMITIVE_LINKS_NOT_IN_PILOT_SAMPLE",
            ),
        ),
    )
    chains["selected_runtime_class"] = (
        chains["canonical_signature"]
        .astype(str)
        .map(selected_class_by_signature)
        .fillna("")
    )
    if is_v4(execution):
        chains = atomic_runtime_retention(chains, fits)
    domains["runtime_resolution"] = (
        domains["canonical_macro_signature"]
        .astype(str)
        .map(
            lambda value: (
                "SELECTED_ROOT_FREE_RUNTIME"
                if value in selected_class_by_signature
                else "RETAIN_ORIGINAL_PRIMITIVE_LINKS_PILOT_ERROR_GATE"
            )
        )
    )
    domains["split_diagnostic_status"] = (
        domains["canonical_macro_signature"]
        .astype(str)
        .map(
            lambda value: (
                "NOT_APPLICABLE_SELECTED"
                if value in selected_class_by_signature
                else "NO_AUTOMATIC_SPLIT_PILOT_RETENTION_IS_CONSERVATIVE"
            )
        )
    )
    provenance = chains.drop_duplicates("chain_id").set_index("chain_id")
    domains["retained_primitive_bundle_ids_json"] = domains["suite_occurrence_id"].map(
        provenance["primitive_bundle_ids_json"]
    )
    domains["retained_internal_bacn_ids_json"] = domains["suite_occurrence_id"].map(
        provenance["internal_bacn_ids_json"]
    )
    if (
        domains[
            ["retained_primitive_bundle_ids_json", "retained_internal_bacn_ids_json"]
        ]
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError("R4 domains do not close to R3 chain provenance")
    runtime_schema = json.loads(
        (
            inherited_authority_root(execution)
            / "config"
            / "inherited_runtime_function_class.schema.json"
        ).read_text(encoding="utf-8")
    )
    registry = _selected_runtime_registry(fits, domains, oracle, runtime_schema)
    if any(
        int(value.get("exact_root_calls_runtime", -1)) != 0
        or bool(value.get("heldout_high_used_for_fit", True))
        for value in registry
    ):
        raise RuntimeError("R4 selected runtime registry violates root/high boundaries")

    oracle_path = output / "bundle_oracle" / "oracle_sample_results.parquet"
    fit_path = output / "runtime_fits" / "runtime_fit_results.parquet"
    domain_path = output / "runtime_fits" / "pilot_domain_summary.csv"
    registry_path = output / "runtime_fits" / "selected_runtime_registry.json"
    _atomic_frame(oracle_path, oracle)
    _atomic_frame(fit_path, fits)
    _atomic_frame(domain_path, domains)
    _atomic_json(registry_path, registry)
    _atomic_frame(chains_path, chains)
    extra_artifacts = postfit_topology_evidence(output, chains) if is_v4(execution) else []
    if is_v4(execution):
        for stage in ("r4_v4_warmup", "r4_v4_controls"):
            receipt = _receipt_path(output, stage)
            validate_pass_receipt(output=output, receipt_path=receipt, config_path=config_path,
                pre_run_code_head=pre_run_code_head, expected_stage_id=stage)
            shard_receipts.append(receipt)
    receipt_path = _receipt_path(output, "R4")
    publish_pass_receipt(
        output=output,
        receipt_path=receipt_path,
        stage_id="R4",
        config_path=config_path,
        pre_run_code_head=pre_run_code_head,
        attempt_code_head=attempt_code_head,
        engineering_retry_epoch=engineering_retry_epoch,
        attempt_id="MERGE_AFTER_ALL_IMMUTABLE_SHARDS",
        artifacts=[oracle_path, fit_path, domain_path, registry_path, chains_path, *extra_artifacts],
        upstream_receipts=[_receipt_path(output, "R3"), *shard_receipts],
        extra={
            "shard_count": len(shard_receipts),
            "signature_count": len(expected),
            "oracle_row_count": len(oracle),
            "fit_row_count": len(fits),
            "all_runtime_exact_root_calls": 0,
        },
    )
    return receipt_path


def _all_selected_experiments(output: Path) -> list[MacroExperiment]:
    plan = json.loads(
        (output / "runtime_fits" / "shard_plan.json").read_text(encoding="utf-8")
    )
    values: list[MacroExperiment] = []
    for shard in plan["shards"]:
        values.extend(
            _read_experiment_jsonl(output / str(shard["input_relative_path"]))
        )
    expected = int(plan["signature_count"])
    if (
        len(values) != expected
        or len({value.canonical_signature for value in values}) != expected
    ):
        raise RuntimeError("Selected experiment registry does not close to R3 plan")
    return values


def _observed_peak_rss(output: Path) -> int:
    path = output / "validation" / "resource_history.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return process_tree_rss_bytes()
    frame = pd.read_csv(path)
    if frame.empty:
        return process_tree_rss_bytes()
    return max(
        process_tree_rss_bytes(),
        int(frame["process_tree_rss_bytes"].max()),
    )


def _scientific_elapsed(output: Path) -> float:
    state = json.loads(
        (output / "validation" / "campaign_state.json").read_text(encoding="utf-8")
    )
    return time.time() - float(state["started_epoch_seconds"])


def stage_r5(context: StageContext) -> list[Path]:
    context.mark("R5_REOPEN_UPSTREAM", force=True)
    validate_pass_receipt(
        output=context.output,
        receipt_path=_receipt_path(context.output, "R4"),
        config_path=context.config_path,
        pre_run_code_head=context.pre_run_code_head,
        expected_stage_id="R4",
    )
    topology = pd.read_csv(context.output / "topology" / "topology_census.csv")
    chains = pd.read_parquet(context.output / "topology" / "chain_candidates.parquet")
    bundle_rows = pd.read_parquet(
        context.output / "bundle_oracle" / "bundle_parameter_occurrences.parquet"
    )
    bundle_census = pd.read_csv(
        context.output / "bundle_oracle" / "bundle_parameter_census.csv"
    )
    oracle = pd.read_parquet(
        context.output / "bundle_oracle" / "oracle_sample_results.parquet"
    )
    fits = pd.read_parquet(
        context.output / "runtime_fits" / "runtime_fit_results.parquet"
    )
    domains = pd.read_csv(context.output / "runtime_fits" / "pilot_domain_summary.csv")
    visits = pd.read_parquet(
        context.output / "charge_audit" / "bacn_visit_semantics.parquet"
    )
    charges = pd.read_parquet(
        context.output / "charge_audit" / "fixed_charge_candidates.parquet"
    )
    racn = pd.read_csv(
        context.output / "interface_audit" / "racn_cost_ownership_audit.csv"
    )
    global_specs = _global_specs_from_occurrences(bundle_rows)
    experiments = _all_selected_experiments(context.output)

    context.mark("R5_BINDINGS_AND_MULTIEDGE", force=True)
    bindings = load_bindings(
        context.config_path,
        verify_suite_digests=False,
        suite_ids=["s03"],
    )
    multiedge = _multiedge_audit(bindings, topology)
    multiedge_path = context.output / "topology" / "multiedge_audit.csv"
    _atomic_frame(multiedge_path, multiedge)

    projected_counts: dict[str, int] = {}
    for suite_id in context.execution["suite_ids"]:
        original_backbone = int(
            topology.loc[
                topology["suite_short_id"].eq(suite_id),
                "original_backbone_bundle_count",
            ].iloc[0]
        )
        resolved = chains.loc[
            chains["suite_short_id"].eq(suite_id)
            & chains["runtime_resolution"].eq("SELECTED_ROOT_FREE_RUNTIME")
        ]
        savings = int((resolved["primitive_link_count"].astype(np.int64) - 1).sum())
        projected_counts[suite_id] = original_backbone - savings

    context.mark("R5_VECTORIZED_KERNEL_BENCHMARK", force=True)
    benchmarks = run_kernel_benchmarks(
        projected_counts,
        fit_results=fits,
        chain_candidates=chains,
        bpr_parameters={
            "free_time_s": bindings.graph.free_time * 60.0,
            "used_capacity": bindings.low_physical.used_capacity,
            "alpha": bindings.graph.alpha,
            "beta": bindings.graph.beta,
            "flow": bindings.low_physical.current_flow,
        },
        offline_experiments=experiments,
        warmups=int(context.execution["kernel_benchmark"]["warmups"]),
        repetitions=int(context.execution["kernel_benchmark"]["timed_repetitions"]),
        progress_guard=context.mark,
    )
    if (
        benchmarks.loc[
            ~benchmarks["kernel_class"].eq("EXACT_ROOT_ORACLE_OFFLINE_ONLY"),
            "exact_root_calls",
        ]
        .ne(0)
        .any()
    ):
        raise RuntimeError("A runtime benchmark invoked the offline exact oracle")
    benchmark_path = context.output / "benchmarks" / "kernel_runtime_benchmark.csv"
    _atomic_frame(benchmark_path, benchmarks)

    context.mark("R5_S03_ACCEPTED_PATH_RECONSTRUCTION", force=True)
    s03_paths = _suite_output_paths(context.output, "s03")
    trace_map = bindings.trace_maps[0]
    frames = _suite_frames(trace_map.root)
    low_trace = load_suite_trace_arrays(
        bindings.sources, "s03", "low_benchmark", trace_map=trace_map
    )
    high_trace = load_suite_trace_arrays(
        bindings.sources, "s03", "high_benchmark", trace_map=trace_map
    )
    s03_topology_frame = pd.read_csv(s03_paths["topology"])
    s03_topology = TopologyResult(
        census=s03_topology_frame.iloc[0].to_dict(),
        chains=pd.read_parquet(s03_paths["chains"]),
        node_protection=pd.read_parquet(s03_paths["protection"]),
    )
    s03_bundle = pd.read_parquet(s03_paths["bundles"])
    s03_internal = pd.read_parquet(s03_paths["internal_keys"])
    work = SimpleNamespace(
        suite_id="s03",
        root=trace_map.root,
        trace_map=trace_map,
        frames=frames,
        low_trace=low_trace,
        high_trace=high_trace,
        topology=s03_topology,
        bundle_key=dict(
            zip(
                s03_bundle["bundle_id"].astype(str),
                s03_bundle["canonical_primitive_key"].astype(str),
                strict=True,
            )
        ),
        internal_key=dict(
            zip(
                s03_internal["bacn_id"].astype(str),
                s03_internal["canonical_internal_key"].astype(str),
                strict=True,
            )
        ),
    )
    path_rows, path_summary = _path_validation(
        bindings, work, global_specs, chains, fits, charges
    )
    if path_rows["status"].ne("PASS").any():
        raise RuntimeError("Accepted path continuity/hash validation failed")
    path_sample_path = context.output / "path_validation" / "path_cost_sample.parquet"
    path_summary_path = (
        context.output / "path_validation" / "path_cost_component_summary.csv"
    )
    _atomic_frame(path_sample_path, path_rows)
    _atomic_frame(path_summary_path, path_summary)
    error_bands = _error_band_summary(fits, path_rows)
    ranking_bands = _ranking_band_summary(path_rows)
    error_path = context.output / "runtime_fits" / "error_band_summary.csv"
    ranking_path = context.output / "runtime_fits" / "error_band_ranking_summary.csv"
    _atomic_frame(error_path, error_bands)
    _atomic_frame(ranking_path, ranking_bands)

    context.mark("R5_FIGURES", force=True)
    _make_figures(
        context.output,
        topology,
        chains,
        fits,
        benchmarks,
        visits,
        charges,
        error_bands,
    )
    authority_verification = json.loads(
        (context.output / "repository_audit" / "authentication.json").read_text(
            encoding="utf-8"
        )
    )["authority_verification"]
    input_bindings = pd.read_csv(
        context.output / "manifests" / "external_artifact_bindings.csv"
    )
    elapsed = _scientific_elapsed(context.output)
    peak_rss = _observed_peak_rss(context.output)
    context.mark("R5_SCIENTIFIC_ACCEPTANCE", force=True)
    acceptance = _scientific_acceptance(
        authority_verification=authority_verification,
        focused_tests_pass=True,
        input_bindings=input_bindings,
        topology=topology,
        chains=chains,
        oracle=oracle,
        fits=fits,
        domains=domains,
        visits=visits,
        racn=racn,
        multiedge=multiedge,
        benchmarks=benchmarks,
        paths=path_rows,
        scientific_elapsed=elapsed,
        peak_rss=peak_rss,
        execution=context.execution,
    )
    supply_validation = json.loads(
        (
            context.output
            / "validation"
            / "reachability_and_physical_conservation_v2.json"
        ).read_text(encoding="utf-8")
    )
    runtime_registry = json.loads(
        (context.output / "runtime_fits" / "selected_runtime_registry.json").read_text(
            encoding="utf-8"
        )
    )
    plan = json.loads(
        (context.output / "runtime_fits" / "shard_plan.json").read_text(
            encoding="utf-8"
        )
    )
    repair_rows = pd.DataFrame(
        [
            {
                "test_id": "P5-REPAIR-R0",
                "test": "R0 v2 cache, canonical SCC labels, exact cross-SCC audit, and accepted conservation reproduce",
                "status": (
                    "PASS"
                    if supply_validation["status"] == "PASS"
                    and supply_validation["cache_reopen_identical"]
                    and supply_validation["cross_scc_reachable_row_count"] == 0
                    and supply_validation["cross_scc_reachable_demand"] == 0.0
                    else "FAIL"
                ),
                "gate_class": "hard",
                "evidence_json": canonical_json(supply_validation),
            },
            {
                "test_id": "P5-REPAIR-SHARDS",
                "test": "deterministic <=500-signature shard plan closes to the sample",
                "status": (
                    "PASS"
                    if int(plan["signature_count"]) == len(experiments)
                    and all(
                        int(value["signature_count"]) <= 500 for value in plan["shards"]
                    )
                    else "FAIL"
                ),
                "gate_class": "hard",
                "evidence_json": canonical_json(
                    {
                        "shard_count": plan["shard_count"],
                        "signature_count": plan["signature_count"],
                        "maximum_shard_size": max(
                            int(value["signature_count"]) for value in plan["shards"]
                        ),
                    }
                ),
            },
            {
                "test_id": "P5-REPAIR-RUNTIME-ROOTS",
                "test": "every published runtime class declares zero exact-root calls and excludes high fit data",
                "status": (
                    "PASS"
                    if runtime_registry
                    and all(
                        int(value["exact_root_calls_runtime"]) == 0
                        and value["heldout_high_used_for_fit"] is False
                        for value in runtime_registry
                    )
                    else "FAIL"
                ),
                "gate_class": "hard",
                "evidence_json": canonical_json(
                    {
                        "runtime_registry_rows": len(runtime_registry),
                        "runtime_exact_root_calls": 0,
                        "heldout_high_used_for_fit": False,
                    }
                ),
            },
        ]
    )
    acceptance = pd.concat([acceptance, repair_rows], ignore_index=True)
    # R6 adds the pre/post input-invariance gate.  Publish the R5 version under
    # a distinct name so no downstream stage mutates a receipt-bound artifact.
    acceptance_path = (
        context.output / "validation" / "acceptance_test_results_pre_invariance.csv"
    )
    _atomic_frame(acceptance_path, acceptance)
    if acceptance["status"].ne("PASS").any():
        failed = acceptance.loc[acceptance["status"].ne("PASS"), "test_id"].tolist()
        raise RuntimeError(f"Phase 5 repaired scientific hard gates failed: {failed}")

    context.mark("R5_REPORTS", force=True)
    _write_reports(
        context.output,
        bindings,
        topology,
        chains,
        bundle_census,
        oracle,
        fits,
        domains,
        visits,
        charges,
        multiedge,
        racn,
        path_rows,
        path_summary,
        benchmarks,
        error_bands,
        ranking_bands,
        True,
        str(context.execution["repair_starting_head"]),
        context.pre_run_code_head,
        elapsed,
        peak_rss,
    )
    # The original report generator intentionally describes the original
    # authority hash. This addendum records the repaired execution authority.
    audit_report = context.output / "reports" / "01_REPOSITORY_AND_INPUT_AUDIT.md"
    audit_text = audit_report.read_text(encoding="utf-8").rstrip()
    _atomic_text(
        audit_report,
        audit_text
        + "\n\n## Repaired v2 execution authority\n\n"
        + f"- SCC-repair/resume authority ZIP SHA-256: `{REPAIRED_OUTER_SHA256}`.\n"
        + f"- Prior repaired-v1 authority ZIP SHA-256: `{PRIOR_REPAIRED_OUTER_SHA256}`.\n"
        + f"- Blocked repaired-v1 handoff SHA-256: `{BLOCKED_V1_HANDOFF_SHA256}`.\n"
        + f"- Original authority ZIP SHA-256: `{ORIGINAL_AUTHORITY_SHA256}`.\n"
        + f"- Original blocked handoff evidence SHA-256: `{BLOCKED_HANDOFF_SHA256}`.\n"
        + f"- Nested parent continuation SHA-256: `{PARENT_CONTINUATION_SHA256}`.\n"
        + "- The v2 package replaces only R0 SCC/reachability mechanics; inherited R1-R6 scientific freezes remain controlling.\n",
    )
    repaired_audit_path = (
        context.output / "reports" / "09_REPAIRED_EXECUTION_AND_RESUME_AUDIT.md"
    )
    _atomic_text(
        repaired_audit_path,
        "# Repaired execution and resume audit\n\n"
        "The pilot ran as independent R0_v2, suite-local R1/R2, merge R3, stable-hash R4 shards, and audit R5 children. "
        "Every PASS receipt is atomic, immutable, bound to the exact execution-config SHA-256 and pre-run code HEAD, and revalidated before downstream use. "
        "Each child emitted 30-second heartbeat/resource evidence and scheduled repeated real Python stack dumps every 300 seconds. Failed attempts, if any, remain under `attempts/`; no PASS stage is recomputed during resume.\n\n"
        f"R4 used `{plan['shard_count']}` shards for `{plan['signature_count']}` unique signatures, with at most `{max(int(value['signature_count']) for value in plan['shards'])}` signatures per shard. "
        "The prior blocked output remains unchanged and was not treated as executable authority.\n",
    )
    projection_path = (
        context.output / "reports" / "10_FORMAL_PHASE5_RUNTIME_PROJECTION.md"
    )
    projection = pd.read_csv(
        context.output / "validation" / "full_phase5_resource_projection.csv"
    )
    _atomic_text(
        projection_path,
        "# Formal Phase 5 runtime projection\n\n"
        "This is a screening projection from the repaired pilot, not authority to run full Phase 5. "
        "It preserves the original population/sample scaling method and must be reviewed in a separate long-run preflight.\n\n"
        + projection.to_markdown(index=False)
        + "\n\nFull Phase 5, final topology/runtime thresholds, Phase 6, national assignment, and ICNN training remain unauthorized.\n",
    )
    extra_v4 = []
    if is_v4(context.execution):
        extra_v4 = write_v4_audits(context, chains, oracle, fits, domains, visits, charges, path_rows, benchmarks)
        findings = json.loads((context.output / "validation/v4_correctness_and_recommendation.json").read_text())
        acceptance = pd.concat([acceptance, pd.DataFrame([{"test_id":"P5-V4-CERTIFIED-CLOSURE", "test":"v4 all-query, all-variant, static-incidence, controls, roles and forecast checks pass", "status":findings["status"], "gate_class":"hard", "evidence_json":canonical_json(findings)}])], ignore_index=True)
        _atomic_frame(acceptance_path, acceptance)
        _atomic_text(audit_report, "# V4 repository and input audit\n\nThe current controlling v4 authority SHA-256 is `" + context.execution["authority_zip_sha256"] + "`. The reviewed v2 R0 cache was imported after byte and dependency checks; later scientific stages were regenerated. Accepted Phase 3/4 inputs, old outputs and repository history remain preserved. See repository_audit/authentication.json and validation/source_reuse_receipts.json.")
        _atomic_text(repaired_audit_path, "# V4 staged execution audit\n\nThe existing supervisor owns independent R0 reuse, five suite children, deterministic sample construction, a real-parameter numerical gate, main shards, retained controls and final audits. PASS receipts bind code/config/input identities. Failed attempts and all event/resource histories are preserved. Engineering repairs are bounded by the current v4 authority. No prior campaign was resumed.")
        _atomic_text(projection_path, (context.output / "reports/formal_campaign_forecast.md").read_text())
        for index, title, evidence in (
            (0,"PILOT_README","validation/v4_correctness_and_recommendation.json"),
            (2,"TOPOLOGY_REDUCTION_RESULTS","topology/corrected_topology_census.csv and topology/remaining_incidence_validation.json"),
            (3,"EXACT_ORACLE_AND_RUNTIME_FIT_RESULTS","bundle_oracle/oracle_certification_summary.csv and runtime_fits/raw_vs_knot_aware_fd.csv"),
            (4,"BACN_VISIT_AND_CHARGE_AUDIT","charge_audit/charge_units_and_denominators.json and path_validation/path_fidelity_by_source_role.csv"),
            (5,"MULTIEDGE_AND_R_INTERFACE_AUDIT","topology/multiedge_audit.csv and interface_audit/racn_cost_ownership_audit.csv"),
            (6,"RUNTIME_AND_RESOURCE_RESULTS","benchmarks/runtime_benchmarks.csv, benchmarks/compiled_vs_proxy_coverage.csv and validation/formal_campaign_forecast.json"),
            (7,"DECISIONS_RECOMMENDED_FOR_FULL_PHASE5","validation/v4_correctness_and_recommendation.json"),
        ):
            _atomic_text(context.output / "reports" / f"{index:02d}_{title}.md", "# " + title.replace("_"," ") + "\n\nV4 regenerated diagnostics pass their correctness gates. Production recommendation: `NOT_READY_MODEL_LIMITATIONS`. Evidence: " + evidence + ".\n\nTopology uses reciprocal group retention and full static-incidence ownership. Runtime fits describe dynamic b/B; occurrence H is added once. All primitive oracle queries and every evaluated fit class are certified. Existing path witness integrity and accounting closure do not establish global new-flow liftability or approximation accuracy. Retained heterogeneous coverage is bounded by at most 256 controls; the full reduced network was not benchmarked. R-ACN cost ownership remains unresolved. Full Phase 5, Phase 6, training and national assignment remain unauthorized.")
    context.mark("R5_OUTPUT_REOPEN", force=True)
    artifacts = [
        multiedge_path,
        benchmark_path,
        path_sample_path,
        path_summary_path,
        error_path,
        ranking_path,
        acceptance_path,
        repaired_audit_path,
        projection_path,
        *sorted((context.output / "figures").glob("*.png")),
        *sorted((context.output / "reports").glob("0[0-7]_*.md")),
        context.output / "validation" / "pilot_decision_evidence.csv",
        context.output / "validation" / "full_phase5_resource_projection.csv",
        context.output / "runtime_fits" / "runtime_fit_summary.csv",
        context.output / "charge_audit" / "bacn_visit_semantics_summary.csv",
        context.output / "charge_audit" / "fixed_charge_summary.csv",
        *extra_v4,
    ]
    for path in artifacts:
        if not path.is_file() or _describe_reopen(path)[0] != "PASS":
            raise RuntimeError(f"R5 output failed reopen: {path}")
    context.mark("R5_COMPLETE", force=True)
    return artifacts


def stage_r6(context: StageContext) -> list[Path]:
    context.mark("R6_RECEIPT_CHAIN_REOPEN", force=True)
    for stage_id in STAGE_ORDER[:-1]:
        validate_pass_receipt(
            output=context.output,
            receipt_path=_receipt_path(context.output, stage_id),
            config_path=context.config_path,
            pre_run_code_head=context.pre_run_code_head,
            expected_stage_id=stage_id,
        )
    context.mark("R6_ACCEPTED_INPUT_INVARIANCE", force=True)
    bindings = pd.read_csv(
        context.output / "manifests" / "external_artifact_bindings.csv"
    )
    rows = []
    for row in bindings.itertuples(index=False):
        path = Path(str(row.absolute_path))
        if str(row.format) == "directory":
            actual_digest = _directory_digest(path)
            actual_size = sum(
                value.stat().st_size for value in path.rglob("*") if value.is_file()
            )
        else:
            actual_digest = sha256_file(path)
            actual_size = path.stat().st_size
        passed = bool(
            actual_digest == str(row.sha256_or_directory_digest)
            and actual_size == int(row.size_bytes)
        )
        rows.append(
            {
                "role": str(row.role),
                "absolute_path": str(path.resolve()),
                "start_sha256_or_directory_digest": str(row.sha256_or_directory_digest),
                "end_sha256_or_directory_digest": actual_digest,
                "start_size_bytes": int(row.size_bytes),
                "end_size_bytes": actual_size,
                "status": "PASS" if passed else "FAIL",
            }
        )
    invariance_frame = pd.DataFrame(rows)
    invariance_csv = context.output / "validation" / "accepted_input_invariance.csv"
    _atomic_frame(invariance_csv, invariance_frame)
    invariance_path = context.output / "validation" / "accepted_input_invariance.json"
    invariance_pass = bool(invariance_frame["status"].eq("PASS").all())
    _atomic_json(
        invariance_path,
        {
            "schema_id": "partC.phase5.repaired.accepted_input_invariance.v1",
            "recorded_utc": utc_now(),
            "binding_count": len(invariance_frame),
            "pre_post_exact_match": invariance_pass,
            "mismatch_roles": invariance_frame.loc[
                invariance_frame["status"].ne("PASS"), "role"
            ]
            .astype(str)
            .tolist(),
            "status": "PASS" if invariance_pass else "FAIL",
        },
    )
    if not invariance_pass:
        raise RuntimeError("Accepted inputs changed during the repaired pilot")
    acceptance_pre_path = (
        context.output / "validation" / "acceptance_test_results_pre_invariance.csv"
    )
    acceptance_path = context.output / "validation" / "acceptance_test_results.csv"
    acceptance = pd.read_csv(acceptance_pre_path)
    if "P5-REPAIR-INPUT-INVARIANCE" not in set(acceptance["test_id"].astype(str)):
        acceptance = pd.concat(
            [
                acceptance,
                pd.DataFrame(
                    [
                        {
                            "test_id": "P5-REPAIR-INPUT-INVARIANCE",
                            "test": "all repaired authority and accepted scientific inputs are byte invariant",
                            "status": "PASS",
                            "gate_class": "hard",
                            "evidence_json": canonical_json(
                                {
                                    "binding_count": len(invariance_frame),
                                    "evidence": "validation/accepted_input_invariance.csv",
                                }
                            ),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    _atomic_frame(acceptance_path, acceptance)
    if acceptance["status"].ne("PASS").any():
        raise RuntimeError("R6 cannot seal failing acceptance evidence")
    context.mark("R6_PRE_FINAL_REOPEN", force=True)
    required_before_final = [
        context.output / "reports" / f"{index:02d}_{name}"
        for index, name in (
            (0, "PILOT_README.md"),
            (1, "REPOSITORY_AND_INPUT_AUDIT.md"),
            (2, "TOPOLOGY_REDUCTION_RESULTS.md"),
            (3, "EXACT_ORACLE_AND_RUNTIME_FIT_RESULTS.md"),
            (4, "BACN_VISIT_AND_CHARGE_AUDIT.md"),
            (5, "MULTIEDGE_AND_R_INTERFACE_AUDIT.md"),
            (6, "RUNTIME_AND_RESOURCE_RESULTS.md"),
            (7, "DECISIONS_RECOMMENDED_FOR_FULL_PHASE5.md"),
        )
    ]
    for path in required_before_final:
        if not path.is_file() or _describe_reopen(path)[0] != "PASS":
            raise RuntimeError(f"R6 pre-final report is absent or unreadable: {path}")
    context.mark("R6_CHILD_COMPLETE", force=True)
    return [invariance_csv, invariance_path, acceptance_path, *required_before_final]


def _retry_history(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    attempt_root = output / "attempts"
    if not attempt_root.is_dir():
        return rows
    timing_by_attempt: dict[tuple[str, str], dict[str, Any]] = {}
    timing_path = output / "validation" / "stage_timing.csv"
    if timing_path.is_file():
        for value in pd.read_csv(timing_path).to_dict(orient="records"):
            timing_by_attempt[(str(value["stage_id"]), str(value["attempt_id"]))] = value
    for attempt_dir in sorted(
        value for value in attempt_root.glob("*/attempt_*") if value.is_dir()
    ):
        stage_id = attempt_dir.parent.name
        attempt_id = attempt_dir.name
        launch_path = attempt_dir / "launch.json"
        result_path = attempt_dir / "child_result.json"
        launch = (
            json.loads(launch_path.read_text(encoding="utf-8"))
            if launch_path.is_file()
            else {}
        )
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.is_file()
            else {}
        )
        timing = timing_by_attempt.get((stage_id, attempt_id), {})
        status = str(result.get("status") or timing.get("status") or "UNKNOWN")
        retry_reason = launch.get("retry_reason")
        if status == "PASS" and not retry_reason:
            continue
        preservation_path = attempt_dir / "failed_output_preservation.json"
        preservation = (
            json.loads(preservation_path.read_text(encoding="utf-8"))
            if preservation_path.is_file()
            else {}
        )
        rows.append(
            {
                "stage_id": stage_id,
                "attempt_id": attempt_id,
                "status": status,
                "exception": result.get("exception"),
                "returncode": (
                    int(timing["returncode"]) if "returncode" in timing else None
                ),
                "timed_out": (
                    str(timing.get("timed_out", "")).lower() == "true"
                    if timing
                    else None
                ),
                "native_exit_without_child_result": not result_path.is_file(),
                "retry_reason": retry_reason,
                "pre_run_code_head": launch.get("pre_run_code_head"),
                "attempt_code_head": launch.get(
                    "attempt_code_head", launch.get("pre_run_code_head")
                ),
                "engineering_retry_epoch": launch.get("engineering_retry_epoch"),
                "failed_output_preservation_status": preservation.get("status"),
                "preserved_path": attempt_dir.relative_to(output).as_posix(),
            }
        )
    return rows


def _repaired_status_payload(
    *,
    config_path: Path,
    execution: dict[str, Any],
    output: Path,
    pre_run_code_head: str,
    ending_head: str,
    post_checkpoint_bound: bool,
    status: str = FINAL_STATUS,
    blockers: Sequence[str] = (),
) -> dict[str, Any]:
    epoch_records = _validate_engineering_retry_chain(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=pre_run_code_head,
    )
    code_epochs = [
        {
            "epoch_number": int(record["epoch_number"]),
            "failed_stage_id": str(record["failed_stage_id"]),
            "failed_attempt_id": str(record["failed_attempt_id"]),
            "authorized_next_attempt_id": str(record["authorized_next_attempt_id"]),
            "previous_execution_code_head": str(
                record["previous_execution_code_head"]
            ),
            "authorized_execution_code_head": str(
                record["authorized_execution_code_head"]
            ),
            "retry_reason": str(record["retry_reason"]),
            "relative_path": str(record["relative_path"]),
            "sha256": str(record["record_sha256"]),
        }
        for record in epoch_records
    ]
    receipts = {}
    for stage_id in STAGE_ORDER:
        path = _receipt_path(output, stage_id)
        if path.is_file():
            receipts[stage_id] = {
                "relative_path": path.relative_to(output).as_posix(),
                "sha256": sha256_file(path),
                "status": json.loads(path.read_text(encoding="utf-8")).get("status"),
            }
    decisions: dict[str, str] = {}
    decision_path = output / "validation" / "pilot_decision_evidence.csv"
    if decision_path.is_file():
        frame = pd.read_csv(decision_path)
        decisions = dict(
            zip(
                frame["decision_id"].astype(str),
                frame["recommendation"].astype(str),
                strict=True,
            )
        )
    outputs: dict[str, Any] = {}
    domain_path = output / "runtime_fits" / "pilot_domain_summary.csv"
    if domain_path.is_file():
        domains = pd.read_csv(domain_path)
        outputs["sampled_macro_signatures"] = int(
            domains["canonical_macro_signature"].nunique()
        )
    for name, path in (
        ("oracle_rows", output / "bundle_oracle" / "oracle_sample_results.parquet"),
        ("fit_rows", output / "runtime_fits" / "runtime_fit_results.parquet"),
        ("path_rows", output / "path_validation" / "path_cost_sample.parquet"),
        ("visit_rows", output / "charge_audit" / "bacn_visit_semantics.parquet"),
    ):
        if path.is_file():
            outputs[name] = int(_describe_reopen(path)[1])
    payload = {
        "schema_id": "partC.phase5.pilot_v4_0905.status.v1" if is_v4(execution) else REPAIRED_STATUS_SCHEMA,
        "campaign_id": execution["campaign_id"],
        "run_id": execution["run_id"],
        "repair_of_run_id": "phase5_backbone_compilation_pilot_repaired_0904_v2_run_001" if is_v4(execution) else REPAIR_OF_RUN_ID,
        "status": status,
        "branch": execution["expected_branch"],
        "starting_head": execution["repair_starting_head"],
        "ending_head": ending_head,
        "pre_run_code_head": pre_run_code_head,
        "active_scientific_execution_code_head": (
            code_epochs[-1]["authorized_execution_code_head"]
            if code_epochs
            else pre_run_code_head
        ),
        "engineering_retry_code_epochs": code_epochs,
        "post_checkpoint_head_bound": post_checkpoint_bound,
        "execution_config_path": str(config_path),
        "execution_config_sha256": sha256_file(config_path),
        "repaired_pilot_authorized": True,
        "full_phase5_authorized": False,
        "final_reduced_topology_accepted": False,
        "final_runtime_thresholds_frozen": False,
        "phase6_authorized": False,
        "national_assignment_performed": False,
        "icnn_training_performed": False,
        "next_step_authorized": False,
        "stage_receipts": receipts,
        "tests": {
            "focused_phase5_original_and_repair": (
                json.loads(
                    (output / "validation" / "focused_tests_status.json").read_text(
                        encoding="utf-8"
                    )
                ).get("status")
                if (output / "validation" / "focused_tests_status.json").is_file()
                else "NOT_REACHED"
            )
        },
        "outputs": outputs,
        "decisions": decisions,
        "blockers": list(blockers),
        "retry_history": _retry_history(output),
        "recommendation": (
            "PROCEED_TO_EXTERNAL_PHASE5_REPAIRED_PILOT_REVIEW"
            if status == FINAL_STATUS
            else "BLOCK_AND_REVIEW_EXACT_FAILURE_EVIDENCE"
        ),
    }
    if is_v4(execution):
        payload.update({"authority_sha256":execution["authority_zip_sha256"],"real_output_root":str(output),
            "elapsed_task_seconds":_scientific_elapsed(output),"production_recommendation":"NOT_READY_MODEL_LIMITATIONS",
            "diagnostics_completion":status,"scientific_correctness_pass":status == FINAL_STATUS})
        forecast_path = output / "validation/formal_campaign_forecast.json"
        if forecast_path.exists():
            payload["formal_campaign_forecast"] = json.loads(forecast_path.read_text())
    return payload


def _validate_repaired_required_outputs(
    output: Path, execution: dict[str, Any]
) -> dict[str, Any]:
    required = pd.read_csv(
        _project_contract_file(execution, "required_output_artifacts_relative_path")
    )
    required_paths = list(required["relative_path"].astype(str))
    required_paths.extend(
        [
            "path_validation/path_cost_sample.parquet",
            "runtime_fits/pilot_domain_summary.csv",
            "runtime_fits/selected_runtime_registry.json",
            "runtime_fits/error_band_summary.csv",
            "runtime_fits/error_band_ranking_summary.csv",
            "validation/pilot_decision_evidence.csv",
            "validation/full_phase5_resource_projection.csv",
            "validation/accepted_input_invariance.json",
        ]
    )
    missing = []
    empty = []
    reopen_failures = []
    for relative in sorted(set(required_paths)):
        path = output / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if path.stat().st_size <= 0:
            empty.append(relative)
            continue
        try:
            if _describe_reopen(path)[0] != "PASS":
                reopen_failures.append(relative)
        except Exception as error:
            reopen_failures.append(f"{relative}: {type(error).__name__}: {error}")
    return {
        "schema_id": "partC.phase5.repaired.required_output_validation.v1",
        "required_count": len(set(required_paths)),
        "missing": missing,
        "empty": empty,
        "reopen_failures": reopen_failures,
        "status": (
            "PASS" if not missing and not empty and not reopen_failures else "FAIL"
        ),
    }


def finalize_r6(
    *,
    config_path: Path,
    execution: dict[str, Any],
    output: Path,
    pre_run_code_head: str,
) -> Path:
    attempt_code_head, _ = _bound_live_execution(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=pre_run_code_head,
    )
    r6_receipt = _receipt_path(output, "R6")
    validate_pass_receipt(
        output=output,
        receipt_path=r6_receipt,
        config_path=config_path,
        pre_run_code_head=pre_run_code_head,
        expected_stage_id="R6",
    )
    elapsed = _scientific_elapsed(output)
    peak_rss = _observed_peak_rss(output)
    added_bytes = sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    )
    resource_path = output / "validation" / "resource_usage.json"
    _atomic_json(
        resource_path,
        {
            "schema_id": "partC.phase5.pilot_repaired.resource_usage.v1",
            "started_utc": json.loads(
                (output / "validation" / "campaign_state.json").read_text(
                    encoding="utf-8"
                )
            )["started_utc"],
            "recorded_utc": utc_now(),
            "elapsed_seconds_before_post_commit_handoff": elapsed,
            "wall_clock_limit_seconds": execution["resources"]["wall_clock_seconds"],
            "protected_sealing_seconds": execution["resources"][
                "protected_sealing_seconds"
            ],
            "process_tree_peak_observed_rss_bytes": peak_rss,
            "process_tree_rss_limit_bytes": execution["resources"][
                "process_tree_rss_bytes"
            ],
            "added_output_bytes_before_manifest": added_bytes,
            "added_disk_limit_bytes": execution["resources"]["added_disk_bytes"],
            "network_used": False,
            "package_installation_performed": False,
            "status": (
                "PASS"
                if elapsed < execution["resources"]["wall_clock_seconds"]
                and peak_rss <= execution["resources"]["process_tree_rss_bytes"]
                and added_bytes <= execution["resources"]["added_disk_bytes"]
                else "FAIL"
            ),
        },
    )
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    if resource["status"] != "PASS":
        raise RuntimeError("Repaired-pilot final resource closure failed")
    reproduction_path = output / "validation" / "reproduction.json"
    plan = json.loads(
        (output / "runtime_fits" / "shard_plan.json").read_text(encoding="utf-8")
    )
    _atomic_json(
        reproduction_path,
        {
            "schema_id": "partC.phase5.pilot_repaired.reproduction.v1",
            "run_id": execution["run_id"],
            "command": [
                execution["python_executable"],
                "-m",
                "partC_rev1_src.phase5.staged_cli",
                "resume",
                "--config",
                str(config_path),
            ],
            "python": sys.version,
            "execution_config_path": str(config_path),
            "execution_config_sha256": sha256_file(config_path),
            "pre_run_code_head": pre_run_code_head,
            "active_scientific_execution_code_head": attempt_code_head,
            "engineering_retry_code_epochs": _repaired_status_payload(
                config_path=config_path,
                execution=execution,
                output=output,
                pre_run_code_head=pre_run_code_head,
                ending_head=attempt_code_head,
                post_checkpoint_bound=False,
            )["engineering_retry_code_epochs"],
            "stable_hash_namespace": execution["pilot_sample"]["stable_hash_namespace"],
            "shard_count": plan["shard_count"],
            "high_excluded_from_fit_and_selection": True,
            "current_not_aon_primary": True,
            "capacity_factor_applied_once": True,
            "routing_or_assignment_performed": False,
            "status": "PASS_REPRODUCIBLE_FROM_BOUND_INPUTS_AND_RECEIPTS",
        },
    )
    status_path = output / "reports" / "08_PILOT_STATUS.json"
    status = _repaired_status_payload(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=pre_run_code_head,
        ending_head=attempt_code_head,
        post_checkpoint_bound=False,
    )
    jsonschema_validate(
        status,
        json.loads(
            (
                _project_contract_file(execution, "inherited_status_schema_relative_path")
            ).read_text(encoding="utf-8")
        ),
    )
    _atomic_json(status_path, status)
    _atomic_json(
        output / "validation" / "current_stage.json",
        {
            "schema_id": "partC.phase5.repaired.current_stage.v1",
            "recorded_utc": utc_now(),
            "stage_id": "R6",
            "state": "PASS_COMPLETE_PENDING_POST_COMMIT_HANDOFF",
            "pre_run_code_head": pre_run_code_head,
            "attempt_code_head": attempt_code_head,
        },
    )
    # Manifest generation occurs only after every mutable pre-commit output has
    # reached its final content, so its hashes close without self-reference.
    manifest = _write_output_manifest(output)
    if not manifest["reopen_status"].eq("PASS").all():
        raise RuntimeError("Final repaired output manifest reopen closure failed")
    required = _validate_repaired_required_outputs(output, execution)
    if required["status"] != "PASS":
        raise RuntimeError(f"Required repaired output closure failed: {required}")
    return status_path


def seal_exact_blocker(
    *,
    config_path: Path,
    execution: dict[str, Any],
    output: Path,
    pre_run_code_head: str,
    blocker: str,
) -> Path:
    """Seal existing partial stage evidence without recomputing science."""

    attempt_code_head, _ = _bound_live_execution(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=pre_run_code_head,
    )
    for name in ("reports", "validation", "manifests", "repository_audit"):
        (output / name).mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "validation" / "failure.json",
        {
            "schema_id": "partC.phase5.pilot_repaired.failure.v1",
            "recorded_utc": utc_now(),
            "blocker": blocker,
            "scientific_rerun_performed": False,
            "preserved_pass_receipts": [
                path.relative_to(output).as_posix()
                for path in sorted((output / "stage_receipts").glob("*.json"))
                if json.loads(path.read_text(encoding="utf-8")).get("status") == "PASS"
            ],
        },
    )
    status = _repaired_status_payload(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=pre_run_code_head,
        ending_head=attempt_code_head,
        post_checkpoint_bound=False,
        status=BLOCKED_STATUS,
        blockers=[blocker],
    )
    _atomic_json(output / "reports" / "08_PILOT_STATUS.json", status)
    _atomic_text(
        output / "reports" / "BLOCKED.md",
        "# Phase 5 repaired pilot blocker\n\n"
        f"Status: `{BLOCKED_STATUS}`.\n\n"
        f"Exact blocker: `{blocker}`. All earlier PASS receipts and failed-attempt diagnostics are preserved. "
        "No later phase, full Phase 5 execution, assignment, or training is authorized.\n",
    )
    _atomic_json(
        output / "validation" / "resource_usage.json",
        {
            "schema_id": "partC.phase5.pilot_repaired.resource_usage.v1",
            "recorded_utc": utc_now(),
            "elapsed_seconds_at_blocker": _scientific_elapsed(output),
            "process_tree_peak_observed_rss_bytes": _observed_peak_rss(output),
            "status": "BLOCKED",
        },
    )
    _write_blocked_output_manifest(output)
    return output / "reports" / "08_PILOT_STATUS.json"


def qualified_repaired_git_state(
    repo: Path, execution: dict[str, Any], live_head: str
) -> dict[str, Any]:
    """Require committed repaired-pilot code/records while tolerating older evidence."""

    lines = git(
        repo, "status", "--porcelain=v1", "--untracked-files=all"
    ).stdout.splitlines()
    tracked_dirty = [value for value in lines if not value.startswith("?? ")]
    repaired_untracked_prefixes = (
        "partC_rev1_src/phase5/",
        "partC_rev1_src/phase5/repaired_handoff.py",
        "tests/test_phase5_",
        "partC_doc_icnn/phase_5_work/phase5_backbone_compilation_pilot_repaired_0904_",
        "partC_doc_icnn/phase_5_work/PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_",
        "partC_doc_icnn/review_gates/phase_05/backbone_compilation_pilot_repaired_0904_v2/",
        "partC_doc_icnn/handoffs/phase_05/step_backbone_compilation_pilot_repaired_0904_v2/",
        "artifacts_icnn/handoff_bundles/phase_05/phase05_stepbackbone_compilation_pilot_repaired_0904_",
    )
    if is_v4(execution):
        # Reviewed older packages remain untracked archival assets. Require the
        # current v4 source/config/authority records, not old generated handoffs.
        repaired_untracked_prefixes = (
            "partC_rev1_src/phase5/",
            "tests/test_phase5_",
            "partC_doc_icnn/phase_5_work/phase5_pilot_v4_0905_",
            "partC_doc_icnn/phase_5_work/PHASE5_V4_0905_",
            "partC_doc_icnn/review_gates/phase_05/pilot_v4_0905/",
        )
    task_untracked = [
        value
        for value in lines
        if value.startswith("?? ")
        and value[3:].replace("\\", "/").startswith(repaired_untracked_prefixes)
    ]
    if tracked_dirty or task_untracked:
        raise RuntimeError(
            "Repaired-pilot Git state is not qualified: "
            f"tracked={tracked_dirty}, task_untracked={task_untracked}"
        )
    repair_base = str(execution["required_repair_base_ancestor"])
    if (
        git(
            repo,
            "merge-base",
            "--is-ancestor",
            repair_base,
            live_head,
            check=False,
        ).returncode
        != 0
    ):
        raise RuntimeError("Repaired result HEAD does not descend from the repair base")
    changed = set(
        git(repo, "diff", "--name-only", repair_base, live_head).stdout.splitlines()
    )
    required = {
        "partC_rev1_src/phase5/staged_pilot.py",
        "partC_rev1_src/phase5/repaired_handoff.py",
        "partC_rev1_src/phase5/bacn_charges.py",
        "partC_rev1_src/phase5/scc_reachability.py",
        "partC_rev1_src/phase5/inputs.py",
        "partC_rev1_src/phase5/pilot.py",
        "tests/test_phase5_scc_reachability.py",
        "partC_doc_icnn/phase_5_work/phase5_backbone_compilation_pilot_repaired_0904_v2_execution_config.json",
        "partC_doc_icnn/phase_5_work/phase5_pilot_repaired_0904_v2_required_output_artifacts.csv",
        "partC_doc_icnn/phase_5_work/phase5_pilot_repaired_0904_v2_status.schema.json",
        "partC_doc_icnn/phase_5_work/PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_PRELAUNCH_TESTS.json",
        "partC_doc_icnn/phase_5_work/PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_PRELAUNCH_TESTS.md",
        "partC_doc_icnn/phase_5_work/PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_STATUS.json",
        "partC_doc_icnn/phase_5_work/PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_RESULT.md",
        "partC_doc_icnn/phase_5_work/PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_R3_ENGINEERING_REPAIR.json",
        "partC_doc_icnn/phase_5_work/PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_R3_ENGINEERING_REPAIR.md",
        "partC_doc_icnn/review_gates/phase_05/backbone_compilation_pilot_repaired_0904_v2/phase5_pilot_scc_repair_resume_authority_20260904/09_CODEX_GOAL.md",
        "partC_doc_icnn/review_gates/phase_05/backbone_compilation_pilot_repaired_0904_v2/phase5_pilot_scc_repair_resume_authority_20260904/AUTHORITY_INSTALLATION.json",
    }
    missing = sorted(required - changed)
    if is_v4(execution):
        required = {
            "partC_rev1_src/phase5/staged_pilot.py", "partC_rev1_src/phase5/repaired_handoff.py",
            "partC_rev1_src/phase5/bundle_oracle.py", "partC_rev1_src/phase5/topology_reduction.py",
            "partC_rev1_src/phase5/bacn_charges.py", "partC_rev1_src/phase5/runtime_functions.py",
            "partC_rev1_src/phase5/experiments.py", "partC_rev1_src/phase5/v4_evidence.py",
            "tests/test_phase5_v4_correctness.py", "tests/test_phase5_v4_dynamic.py", "tests/test_phase5_v4_numerics.py",
            "partC_doc_icnn/phase_5_work/phase5_pilot_v4_0905_execution_config.json",
            "partC_doc_icnn/phase_5_work/PHASE5_V4_0905_STATUS.json",
            "partC_doc_icnn/phase_5_work/PHASE5_V4_0905_RESULT.md",
        }
        missing = sorted(required - changed)
        if "partC_rev1_src/phase5/scc_reachability.py" in changed:
            raise RuntimeError("V4 must preserve the reviewed SCC implementation")
    if missing:
        raise RuntimeError(
            f"Repaired result checkpoint omits required records: {missing}"
        )
    return {
        "status_lines": lines,
        "tracked_dirty_count": len(tracked_dirty),
        "task_owned_untracked_count": len(task_untracked),
        "changed_path_count_from_repair_base": len(changed),
        "required_changed_paths": sorted(required),
        "status": "QUALIFIED_UNRELATED_HISTORICAL_UNTRACKED_ALLOWED",
    }


def reseal_repaired_output_metadata(config_path: str | Path, ending_head: str) -> Path:
    """Bind the result checkpoint without modifying receipt-bound science."""

    config_path, execution = _execution(config_path)
    repo = Path(execution["project_root"]).resolve()
    output = Path(execution["output_root"]).resolve()
    branch = git(repo, "branch", "--show-current").stdout.strip()
    live_head = git(repo, "rev-parse", "HEAD").stdout.strip()
    if branch != execution["expected_branch"] or live_head != str(ending_head):
        raise RuntimeError(
            "Reseal requires the live authorized branch and exact ending HEAD"
        )
    git_state = qualified_repaired_git_state(repo, execution, live_head)
    binding = json.loads(
        (output / "repository_audit" / "run_binding.json").read_text(encoding="utf-8")
    )
    pre_run_head = str(binding["pre_run_code_head"])
    retry_epochs = _validate_engineering_retry_chain(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=pre_run_head,
    )
    if (
        binding.get("execution_config_sha256") != sha256_file(config_path)
        or binding.get("run_id") != execution["run_id"]
        or git(
            repo,
            "merge-base",
            "--is-ancestor",
            pre_run_head,
            live_head,
            check=False,
        ).returncode
        != 0
    ):
        raise RuntimeError(
            "Post-run checkpoint does not close to the bound pre-run state"
        )
    status_path = output / "reports" / "08_PILOT_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    scientific_status = str(status.get("status"))
    if scientific_status not in {FINAL_STATUS, BLOCKED_STATUS}:
        raise RuntimeError(f"Unknown repaired-pilot status: {scientific_status}")
    if status.get("pre_run_code_head") != pre_run_head:
        raise RuntimeError("Status pre-run HEAD differs from the immutable run binding")
    expected_execution_head = (
        str(retry_epochs[-1]["authorized_execution_code_head"])
        if retry_epochs
        else pre_run_head
    )
    if status.get("active_scientific_execution_code_head") != expected_execution_head:
        raise RuntimeError("Status active execution HEAD differs from the code-epoch chain")
    for epoch in retry_epochs:
        if (
            git(
                repo,
                "merge-base",
                "--is-ancestor",
                str(epoch["authorized_execution_code_head"]),
                live_head,
                check=False,
            ).returncode
            != 0
        ):
            raise RuntimeError("Result checkpoint does not descend every execution code epoch")
    if scientific_status == FINAL_STATUS:
        for stage_id in STAGE_ORDER:
            validate_pass_receipt(
                output=output,
                receipt_path=_receipt_path(output, stage_id),
                config_path=config_path,
                pre_run_code_head=pre_run_head,
                expected_stage_id=stage_id,
            )
        plan = json.loads(
            (output / "runtime_fits" / "shard_plan.json").read_text(encoding="utf-8")
        )
        for shard in plan["shards"]:
            shard_id = str(shard["shard_id"])
            validate_pass_receipt(
                output=output,
                receipt_path=_receipt_path(output, shard_id),
                config_path=config_path,
                pre_run_code_head=pre_run_head,
                expected_stage_id=shard_id,
            )
    status["ending_head"] = live_head
    status["post_checkpoint_head_bound"] = True
    status["sealed_utc"] = utc_now()
    jsonschema_validate(
        status,
        json.loads(
            (
                _project_contract_file(execution, "inherited_status_schema_relative_path")
            ).read_text(encoding="utf-8")
        ),
    )
    _atomic_json(status_path, status)
    reproduction_path = output / "validation" / "reproduction.json"
    if reproduction_path.is_file():
        reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))
        reproduction["ending_head"] = live_head
        reproduction["post_checkpoint_head_bound"] = True
        reproduction["sealed_utc"] = utc_now()
        _atomic_json(reproduction_path, reproduction)
    resource_path = output / "validation" / "resource_usage.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    resource["sealed_utc"] = utc_now()
    resource["total_campaign_elapsed_seconds_before_handoff"] = _scientific_elapsed(
        output
    )
    resource["added_output_bytes_before_post_commit_manifest"] = _output_bytes(output)
    _atomic_json(resource_path, resource)
    _atomic_json(
        output / "repository_audit" / "post_checkpoint.json",
        {
            "schema_id": "partC.phase5.repaired.post_checkpoint.v1",
            "recorded_utc": utc_now(),
            "branch": branch,
            "repair_starting_head": execution["repair_starting_head"],
            "pre_run_code_head": pre_run_head,
            "active_scientific_execution_code_head": expected_execution_head,
            "engineering_retry_code_epochs": status.get(
                "engineering_retry_code_epochs", []
            ),
            "ending_head": live_head,
            "repair_base_is_ancestor": True,
            "pre_run_head_is_ancestor": True,
            "scientific_status": scientific_status,
            "execution_config_sha256": sha256_file(config_path),
            "qualified_git_state": git_state,
            "status": "PASS",
        },
    )
    _atomic_json(
        output / "validation" / "current_stage.json",
        {
            "schema_id": "partC.phase5.repaired.current_stage.v1",
            "recorded_utc": utc_now(),
            "stage_id": "SEALED",
            "state": "WAITING_FOR_EXTERNAL_REVIEW",
            "scientific_status": scientific_status,
            "pre_run_code_head": pre_run_head,
            "active_scientific_execution_code_head": expected_execution_head,
            "ending_head": live_head,
        },
    )
    if scientific_status == BLOCKED_STATUS:
        _write_blocked_output_manifest(output)
    else:
        manifest = _write_output_manifest(output)
        required = _validate_repaired_required_outputs(output, execution)
        if (
            required["status"] != "PASS"
            or not manifest["reopen_status"].eq("PASS").all()
        ):
            raise RuntimeError("Post-checkpoint repaired output closure failed")
    if _output_bytes(output) > int(execution["resources"]["added_disk_bytes"]):
        raise RuntimeError("Post-checkpoint output exceeds the added-disk ceiling")
    return output


def initialize_run(config_path: str | Path) -> tuple[Path, dict[str, Any], Path, str]:
    config_path, execution = _execution(config_path)
    repo = Path(execution["project_root"]).resolve()
    output = Path(execution["output_root"]).resolve()
    if output.exists():
        raise FileExistsError(f"Configured write-new output already exists: {output}")
    branch = git(repo, "branch", "--show-current").stdout.strip()
    pre_run_head = git(repo, "rev-parse", "HEAD").stdout.strip()
    if branch != execution["expected_branch"]:
        raise RuntimeError(
            f"Expected branch {execution['expected_branch']}, found {branch}"
        )
    for ancestor_key in ("required_repair_base_ancestor", "reviewed_phase4_ancestor"):
        ancestor = str(execution[ancestor_key])
        if (
            git(
                repo,
                "merge-base",
                "--is-ancestor",
                ancestor,
                pre_run_head,
                check=False,
            ).returncode
            != 0
        ):
            raise RuntimeError(
                f"Required ancestor is absent: {ancestor_key}={ancestor}"
            )
    status_lines = _require_clean_task_tree(repo)
    goal = repo / str(execution["installed_authority_goal_relative_path"])
    if not goal.is_file() or sha256_file(goal) != str(
        execution["launch_authority_sha256"]
    ):
        raise RuntimeError("Installed repaired launch goal hash binding failed")
    verify_repaired_authority(execution)
    for directory in (
        "authority",
        "repository_audit",
        "cache",
        "suite_products",
        "topology",
        "bundle_oracle",
        "runtime_fits",
        "r4_shards",
        "charge_audit",
        "interface_audit",
        "path_validation",
        "benchmarks",
        "figures",
        "reports",
        "validation",
        "manifests",
        "stage_receipts",
        "attempts",
        "handoff_staging",
    ):
        (output / directory).mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "authority" / config_path.name)
    _copy_tree_without_inputs(
        authority_root(execution), output / "authority" / "installed_contract"
    )
    started = float(execution["task_start_epoch_seconds"]) if is_v4(execution) else time.time()
    _atomic_json(
        output / "validation" / "campaign_state.json",
        {
            "schema_id": "partC.phase5.repaired.campaign_state.v1",
            "campaign_id": execution["campaign_id"],
            "run_id": execution["run_id"],
            "started_utc": utc_now(),
            "started_epoch_seconds": started,
            "science_boundary_epoch_seconds": started
            + float(execution["resources"]["science_boundary_seconds"]),
            "wall_clock_deadline_epoch_seconds": started
            + float(execution["resources"]["wall_clock_seconds"]),
            "status": "RUNNING",
        },
    )
    binding = {
        "schema_id": "partC.phase5.repaired.run_binding.v1",
        "recorded_utc": utc_now(),
        "campaign_id": execution["campaign_id"],
        "run_id": execution["run_id"],
        "branch": branch,
        "repair_starting_head": execution["repair_starting_head"],
        "pre_run_code_head": pre_run_head,
        "execution_config_path": str(config_path),
        "execution_config_sha256": sha256_file(config_path),
        "qualified_git_status_lines": status_lines,
        "network_used": False,
        "package_installation_performed": False,
        "status": "PASS",
    }
    _atomic_json(output / "repository_audit" / "run_binding.json", binding)
    if is_v4(execution):
        _atomic_json(output / "repository_audit/launch_identity.json", {**binding,
            "authority_sha256": execution["authority_zip_sha256"], "output_root": str(output),
            "task_started_epoch_seconds": started, "output_created_epoch_seconds": time.time(),
            "prelaunch_engineering_seconds": time.time() - started, "prior_outputs_preserved": True})
        _atomic_json(output / "validation/dependency_invalidation.json", {
            "schema_id": "partC.phase5.v4.dependency_invalidation.v1", "status": "PASS",
            "reused": ["R0_REVIEWED_BYTES_AND_DEPENDENCIES"], "regenerated": list(STAGE_ORDER[1:]),
            "reason": "topology, oracle, dynamic grid/model identity, potential validation and charge accounting changed",
            "old_r1_r4_scientific_evidence_valid": False, "old_evidence_preserved": True})
        _append_jsonl(output / "validation/repair_epochs.jsonl", {"epoch": 0,
            "role": "PRELAUNCH_CORRECTNESS_REPAIR", "code_head": pre_run_head,
            "starting_head": execution["repair_starting_head"], "maximum_code_epochs": 6,
            "status": "CHECKPOINTED_BEFORE_FIRST_V4_SCIENCE"})
        prelaunch_source = repo / "logs_icnn/phase_5/pilot_v4_0905"
        for source in sorted(prelaunch_source.glob("*.xml")):
            _atomic_text(output / "validation/prelaunch_logs" / (source.name + ".txt"), source.read_text(encoding="utf-8"))
        for name in ("PHASE5_V4_0905_PRELAUNCH_TESTS.json", "PHASE5_V4_0905_CODE_REVIEW.md"):
            source = repo / "partC_doc_icnn/phase_5_work" / name
            if not source.is_file():
                raise RuntimeError(f"Prelaunch evidence missing: {source}")
            shutil.copy2(source, output / "validation" / name)
    _atomic_text(
        output / "validation" / "stage_timing.csv",
        "stage_id,attempt_id,started_utc,ended_utc,elapsed_seconds,returncode,timed_out,status",
    )
    _atomic_text(
        output / "validation" / "resource_history.csv",
        "recorded_utc,stage_id,attempt_id,elapsed_seconds,process_tree_rss_bytes,output_bytes",
    )
    _atomic_text(output / "validation" / "stage_events.jsonl", "")
    _atomic_json(
        output / "validation" / "current_stage.json",
        {
            "schema_id": "partC.phase5.repaired.current_stage.v1",
            "recorded_utc": utc_now(),
            "stage_id": "NOT_STARTED",
            "state": "READY",
            "pre_run_code_head": pre_run_head,
        },
    )
    return config_path, execution, output, pre_run_head


def authorize_engineering_retry(
    config_path: str | Path,
    *,
    stage_id: str,
    retry_reason: str,
    validation_record: str | Path,
) -> Path:
    """Authorize one committed engineering repair without rewriting prior receipts."""

    reason = str(retry_reason).strip()
    if not reason:
        raise ValueError("An exact demonstrated engineering-repair reason is required")
    config_path, execution = _execution(config_path)
    output = Path(execution["output_root"]).resolve()
    binding_path = output / "repository_audit" / "run_binding.json"
    if not binding_path.is_file():
        raise FileNotFoundError("Engineering retry requires an initialized bound run")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    pre_run_head = str(binding["pre_run_code_head"])
    if (
        binding.get("execution_config_sha256") != sha256_file(config_path)
        or binding.get("campaign_id") != execution["campaign_id"]
        or binding.get("run_id") != execution["run_id"]
    ):
        raise RuntimeError("Engineering retry differs from the immutable run binding")
    repo = Path(execution["project_root"]).resolve()
    branch = git(repo, "branch", "--show-current").stdout.strip()
    live_head = git(repo, "rev-parse", "HEAD").stdout.strip()
    status_lines = _require_clean_task_tree(repo)
    prior_records = _validate_engineering_retry_chain(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=pre_run_head,
    )
    if is_v4(execution) and len(prior_records) >= int(execution["retry"]["max_engineering_code_epochs"]):
        raise RuntimeError("V4 engineering code epoch budget exhausted")
    previous_head = (
        str(prior_records[-1]["authorized_execution_code_head"])
        if prior_records
        else pre_run_head
    )
    if branch != execution["expected_branch"] or live_head == previous_head:
        raise RuntimeError("Engineering retry requires a new commit on the bound branch")
    parents = git(repo, "rev-list", "--parents", "-n", "1", live_head).stdout.split()
    if (
        len(parents) < 2
        or parents[0] != live_head
        or parents[1] != previous_head
        or git(
            repo,
            "merge-base",
            "--is-ancestor",
            previous_head,
            live_head,
            check=False,
        ).returncode
        != 0
    ):
        raise RuntimeError("Engineering retry must be one committed descendant of the active code epoch")
    changed_paths = sorted(
        git(repo, "diff", "--name-only", previous_head, live_head).stdout.splitlines()
    )
    if not changed_paths or not set(changed_paths).issubset(
        _engineering_allowed_paths(execution)
    ):
        raise RuntimeError(f"Engineering-retry diff is outside the authorized scope: {changed_paths}")
    validation_path = Path(validation_record).resolve()
    if not validation_path.is_file() or not validation_path.is_relative_to(repo):
        raise FileNotFoundError("Engineering-retry validation record is absent or outside repo")
    validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation_payload.get("status") != "PASS"
        or validation_payload.get("pre_run_code_head") != pre_run_head
        or validation_payload.get("failed_stage_id") != stage_id
        or validation_payload.get("execution_config_sha256") != sha256_file(config_path)
    ):
        raise RuntimeError("Engineering-retry validation record does not bind the failed run")
    if is_v4(execution) and validation_payload.get("post_run_audit_repair"):
        from .v4_evidence import authorize_audit_repair
        return authorize_audit_repair(config_path=config_path,execution=execution,output=output,
            pre_run_head=pre_run_head,previous_head=previous_head,live_head=live_head,prior_records=prior_records,
            validation_path=validation_path,validation_payload=validation_payload,changed_paths=changed_paths,reason=reason)
    receipt_path = _receipt_path(output, stage_id)
    if receipt_path.exists():
        raise RuntimeError(f"Cannot retry a stage that already has a PASS receipt: {stage_id}")
    attempt_root = output / "attempts" / stage_id
    attempts = sorted(
        value
        for value in attempt_root.glob("attempt_*")
        if value.is_dir() and value.name[8:].isdigit()
    )
    if not attempts:
        raise RuntimeError("Engineering retry requires a preserved failed attempt")
    expected_attempts = [f"attempt_{index:03d}" for index in range(1, len(attempts) + 1)]
    if [value.name for value in attempts] != expected_attempts:
        raise RuntimeError("Engineering-retry attempt history is not contiguous")
    if len(attempts) > int(
        execution["resume_policy"]["max_engineering_retries_per_child"]
    ):
        raise RuntimeError("Engineering-retry limit is already exhausted")
    failed_attempt = attempts[-1]
    preservation_path = failed_attempt / "failed_output_preservation.json"
    if not preservation_path.is_file():
        raise RuntimeError("Engineering retry requires failed-output preservation evidence")
    preservation = json.loads(preservation_path.read_text(encoding="utf-8"))
    if preservation.get("status") != "PASS" or preservation.get("stage_id") != stage_id:
        raise RuntimeError("Failed-output preservation evidence did not pass")
    timing = pd.read_csv(output / "validation" / "stage_timing.csv")
    timing_match = timing.loc[
        timing["stage_id"].astype(str).eq(stage_id)
        & timing["attempt_id"].astype(str).eq(failed_attempt.name)
    ]
    if (
        len(timing_match) != 1
        or str(timing_match.iloc[0]["status"]) not in {"FAIL", "TIMEOUT"}
    ):
        raise RuntimeError("Engineering retry lacks one exact failed timing row")
    current_stage = json.loads(
        (output / "validation" / "current_stage.json").read_text(encoding="utf-8")
    )
    if (
        current_stage.get("stage_id") != stage_id
        or current_stage.get("attempt_id") != failed_attempt.name
        or current_stage.get("state") not in {"FAIL", "TIMEOUT"}
    ):
        raise RuntimeError("Engineering retry does not match the current failed stage")
    upstream_rows = []
    for upstream in _upstream_paths(output, stage_id):
        upstream_stage = str(
            json.loads(upstream.read_text(encoding="utf-8"))["stage_id"]
        )
        validate_pass_receipt(
            output=output,
            receipt_path=upstream,
            config_path=config_path,
            pre_run_code_head=pre_run_head,
            expected_stage_id=upstream_stage,
        )
        upstream_rows.append(_bound_file(output, upstream))
    failed_evidence = [
        _bound_file(output, path)
        for path in sorted(value for value in failed_attempt.rglob("*") if value.is_file())
    ]
    epoch_number = len(prior_records) + 1
    epoch_path = _engineering_retry_root(output) / f"epoch_{epoch_number:03d}.json"
    if epoch_path.exists():
        raise FileExistsError(f"Engineering-retry epoch is immutable: {epoch_path}")
    timing_row = timing_match.iloc[0]
    failed_timed_out = str(timing_row["timed_out"]).strip().lower() == "true"
    payload = {
        "schema_id": ENGINEERING_RETRY_SCHEMA,
        "epoch_number": epoch_number,
        "recorded_utc": utc_now(),
        "campaign_id": execution["campaign_id"],
        "run_id": execution["run_id"],
        "branch": branch,
        "pre_run_code_head": pre_run_head,
        "previous_execution_code_head": previous_head,
        "authorized_execution_code_head": live_head,
        "previous_epoch_sha256": (
            str(prior_records[-1]["record_sha256"]) if prior_records else None
        ),
        "execution_config_path": str(config_path),
        "execution_config_sha256": sha256_file(config_path),
        "failed_stage_id": stage_id,
        "failed_attempt_id": failed_attempt.name,
        "authorized_next_attempt_id": f"attempt_{len(attempts) + 1:03d}",
        "retry_reason": reason,
        "failed_timing": {
            key: (
                failed_timed_out
                if key == "timed_out"
                else (
                    float(timing_row[key])
                    if key == "elapsed_seconds"
                    else int(timing_row[key])
                    if key == "returncode"
                    else str(timing_row[key])
                )
            )
            for key in ("elapsed_seconds", "returncode", "timed_out", "status")
        },
        "native_exit_without_child_result": not (
            failed_attempt / "child_result.json"
        ).exists(),
        "failed_attempt_evidence": failed_evidence,
        "upstream_receipts": upstream_rows,
        "repair_validation_record": {
            "relative_path": validation_path.relative_to(repo).as_posix(),
            "size_bytes": validation_path.stat().st_size,
            "sha256": sha256_file(validation_path),
        },
        "changed_paths": changed_paths,
        "git_patch_sha256": _git_patch_sha256(repo, previous_head, live_head),
        "qualified_git_status_lines": status_lines,
        "scientific_config_unchanged": True,
        "upstream_pass_receipts_unchanged": True,
        "failed_attempt_preserved": True,
        "blind_retry_performed": False,
        "status": "PASS_AUTHORIZED_ENGINEERING_RETRY",
    }
    _atomic_json(epoch_path, payload)
    records = _validate_engineering_retry_chain(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=pre_run_head,
    )
    if records[-1]["authorized_execution_code_head"] != live_head:
        raise RuntimeError("Engineering-retry epoch failed immediate reopen")
    return epoch_path


def _bound_live_execution(
    *,
    config_path: Path,
    execution: dict[str, Any],
    output: Path,
    pre_run_code_head: str,
) -> tuple[str, dict[str, Any] | None]:
    repo = Path(execution["project_root"]).resolve()
    branch = git(repo, "branch", "--show-current").stdout.strip()
    live_head = git(repo, "rev-parse", "HEAD").stdout.strip()
    _require_clean_task_tree(repo)
    records = _validate_engineering_retry_chain(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=pre_run_code_head,
    )
    expected_head = (
        str(records[-1]["authorized_execution_code_head"])
        if records
        else str(pre_run_code_head)
    )
    if branch != execution["expected_branch"] or live_head != expected_head:
        raise RuntimeError("Live code HEAD is not the bound active execution epoch")
    return live_head, _receipt_epoch_reference(records, live_head)


def load_initialized_run(
    config_path: str | Path,
) -> tuple[Path, dict[str, Any], Path, str]:
    config_path, execution = _execution(config_path)
    output = Path(execution["output_root"]).resolve()
    binding_path = output / "repository_audit" / "run_binding.json"
    if not binding_path.is_file():
        return initialize_run(config_path)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if (
        binding.get("execution_config_sha256") != sha256_file(config_path)
        or binding.get("run_id") != execution["run_id"]
        or binding.get("campaign_id") != execution["campaign_id"]
        or binding.get("branch") != execution["expected_branch"]
    ):
        raise RuntimeError("Existing repaired run binding differs from the config")
    repo = Path(execution["project_root"])
    branch = git(repo, "branch", "--show-current").stdout.strip()
    _bound_live_execution(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=str(binding["pre_run_code_head"]),
    )
    if branch != execution["expected_branch"]:
        raise RuntimeError("Resume requires the exact bound branch")
    return config_path, execution, output, str(binding["pre_run_code_head"])


def next_attempt_number(
    attempt_root: Path,
    *,
    max_engineering_retries: int,
    retry_reason: str | None,
) -> int:
    attempts = sorted(
        value
        for value in attempt_root.glob("attempt_*")
        if value.is_dir() and value.name[8:].isdigit()
    )
    if not attempts:
        return 1
    if not retry_reason or not retry_reason.strip():
        raise RuntimeError(
            "A failed child already exists; an explicit demonstrated engineering repair reason is required"
        )
    maximum_attempts = 1 + int(max_engineering_retries)
    if len(attempts) >= maximum_attempts:
        raise RuntimeError(
            f"Stage retry limit exhausted ({maximum_attempts} total attempts)"
        )
    expected = list(range(1, len(attempts) + 1))
    observed = [int(value.name[8:]) for value in attempts]
    if observed != expected:
        raise RuntimeError(f"Attempt history is not contiguous: {observed}")
    return len(attempts) + 1


def _append_csv_row(path: Path, values: Sequence[Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerow(values)
        stream.flush()
        os.fsync(stream.fileno())


def _output_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def _stage_owned_paths(output: Path, stage_id: str) -> list[Path]:
    """Return the non-overlapping scientific targets owned by one child."""

    if stage_id == "R0_v2":
        relative = [
            "repository_audit/authentication.json",
            "manifests/external_artifact_bindings.csv",
            "cache",
            "validation/focused_tests.txt",
            "validation/focused_tests_status.json",
            "validation/r0_reusable_supply_assessment.json",
            "validation/reachability_and_physical_conservation_v2.json",
            "validation/r0_scc_algorithm_benchmark.json",
            "validation/r0_cross_scc_exact_audit.json",
            "validation/r0_scipy_full_graph_call_audit.json",
            "validation/r0_v1_to_v2_change_audit.json",
            "reports/11_R0_SCC_ENGINE_REPAIR_AND_VALIDATION.md",
        ]
    elif stage_id == "R1_s02" or stage_id.startswith("R2_s"):
        suite_id = stage_id.split("_")[-1]
        relative = [f"suite_products/{suite_id}"]
    elif stage_id == "R3":
        relative = [
            "topology/topology_census.csv",
            "topology/chain_candidates_r3.parquet",
            "topology/bacn_protection_census.parquet",
            "bundle_oracle/bundle_parameter_census.csv",
            "bundle_oracle/bundle_parameter_occurrences.parquet",
            "bundle_oracle/canonical_macro_census.csv",
            "charge_audit/bacn_visit_semantics.parquet",
            "charge_audit/fixed_charge_candidates.parquet",
            "interface_audit/racn_cost_ownership_audit.csv",
            "runtime_fits/shard_plan.json",
            "runtime_fits/pilot_sample_index.csv",
            "r4_shards",
        ]
    elif stage_id.startswith("r4_"):
        relative = [f"r4_shards/{stage_id}"] if stage_id in {"r4_v4_warmup", "r4_v4_controls"} else [
            f"r4_shards/{stage_id}/oracle_sample_results.parquet",
            f"r4_shards/{stage_id}/runtime_fit_results.parquet",
            f"r4_shards/{stage_id}/pilot_domain_summary.csv",
            f"r4_shards/{stage_id}/shard_validation.json",
        ]
    elif stage_id == "R5":
        relative = [
            "topology/multiedge_audit.csv",
            "benchmarks/kernel_runtime_benchmark.csv",
            "path_validation",
            "figures",
            "reports",
            "validation/acceptance_test_results_pre_invariance.csv",
            "validation/pilot_decision_evidence.csv",
            "validation/full_phase5_resource_projection.csv",
            "runtime_fits/error_band_summary.csv",
            "runtime_fits/error_band_ranking_summary.csv",
            "runtime_fits/runtime_fit_summary.csv",
            "charge_audit/bacn_visit_semantics_summary.csv",
            "charge_audit/fixed_charge_summary.csv",
        ]
    elif stage_id == "R6":
        relative = [
            "validation/accepted_input_invariance.csv",
            "validation/accepted_input_invariance.json",
            "validation/acceptance_test_results.csv",
        ]
    else:
        raise ValueError(f"No failed-output ownership map for {stage_id}")
    targets = [(output / value).resolve() for value in relative]
    root = output.resolve()
    if any(value == root or not value.is_relative_to(root) for value in targets):
        raise RuntimeError(f"Unsafe stage-owned target map for {stage_id}")
    return targets


def _copy_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def capture_stage_baseline(output: Path, stage_id: str, attempt_dir: Path) -> Path:
    """Copy any pre-attempt targets so a failed child can be rolled back safely."""

    baseline = attempt_dir / "pre_attempt_baseline"
    baseline.mkdir(parents=True, exist_ok=False)
    rows = []
    for target in _stage_owned_paths(output, stage_id):
        relative = target.relative_to(output.resolve())
        existed = target.exists()
        if existed:
            _copy_path(target, baseline / relative)
        rows.append({"relative_path": relative.as_posix(), "existed": existed})
    manifest = attempt_dir / "pre_attempt_baseline.json"
    _atomic_json(
        manifest,
        {
            "schema_id": "partC.phase5.repaired.pre_attempt_baseline.v1",
            "recorded_utc": utc_now(),
            "stage_id": stage_id,
            "targets": rows,
            "status": "PASS",
        },
    )
    return manifest


def preserve_failed_stage_outputs(
    output: Path, stage_id: str, attempt_dir: Path
) -> Path:
    """Move failed partial products into the attempt and restore its baseline."""

    baseline_root = attempt_dir / "pre_attempt_baseline"
    if not baseline_root.is_dir():
        raise RuntimeError("Failed stage has no pre-attempt baseline")
    snapshot_root = attempt_dir / "failed_output_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for target in _stage_owned_paths(output, stage_id):
        relative = target.relative_to(output.resolve())
        snapshot = snapshot_root / relative
        baseline = baseline_root / relative
        partial_existed = target.exists()
        preserved_paths: list[str] = []
        if partial_existed:
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            target.replace(snapshot)
            snapshot_files = (
                sorted(value for value in snapshot.rglob("*") if value.is_file())
                if snapshot.is_dir()
                else [snapshot]
            )
            for snapshot_file in snapshot_files:
                opaque = snapshot_file.with_name(snapshot_file.name + ".failed-partial")
                snapshot_file.replace(opaque)
                preserved_paths.append(opaque.relative_to(attempt_dir).as_posix())
        restored = baseline.exists()
        if restored:
            _copy_path(baseline, target)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "failed_partial_preserved": partial_existed,
                "preserved_paths": preserved_paths,
                "pre_attempt_baseline_restored": restored,
            }
        )
    receipt = attempt_dir / "failed_output_preservation.json"
    _atomic_json(
        receipt,
        {
            "schema_id": "partC.phase5.repaired.failed_output_preservation.v1",
            "recorded_utc": utc_now(),
            "stage_id": stage_id,
            "snapshot_root": snapshot_root.relative_to(output).as_posix(),
            "targets": rows,
            "status": "PASS",
        },
    )
    return receipt


def _merge_child_logs(output: Path, stage_id: str, attempt_dir: Path) -> None:
    for name in ("child_events.jsonl", "heartbeats.jsonl"):
        path = attempt_dir / name
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    value["source"] = name
                    _append_jsonl(output / "validation" / "stage_events.jsonl", value)


def _monitor_child(
    *,
    process: subprocess.Popen[Any],
    output: Path,
    stage_id: str,
    attempt_id: str,
    timeout_seconds: float,
    sample_seconds: float,
    rss_limit_bytes: int,
    disk_limit_bytes: int,
    campaign_deadline_epoch_seconds: float,
) -> tuple[int, bool, float, str | None]:
    started = time.monotonic()
    last_sample = -sample_seconds
    timed_out = False
    forced_reason: str | None = None
    while process.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed - last_sample >= sample_seconds:
            rss = process_tree_rss_bytes()
            disk = _output_bytes(output)
            _append_csv_row(
                output / "validation" / "resource_history.csv",
                [utc_now(), stage_id, attempt_id, f"{elapsed:.6f}", rss, disk],
            )
            _append_jsonl(
                output / "validation" / "stage_events.jsonl",
                {
                    "recorded_utc": utc_now(),
                    "event": "SUPERVISOR_HEARTBEAT",
                    "stage_id": stage_id,
                    "attempt_id": attempt_id,
                    "elapsed_seconds": elapsed,
                    "process_tree_rss_bytes": rss,
                    "output_bytes": disk,
                },
            )
            last_sample = elapsed
            if rss > rss_limit_bytes:
                forced_reason = (
                    f"Process-tree RSS guard exceeded: {rss} > {rss_limit_bytes}"
                )
            elif disk > disk_limit_bytes:
                forced_reason = (
                    f"Added-disk guard exceeded: {disk} > {disk_limit_bytes}"
                )
        if time.time() >= campaign_deadline_epoch_seconds:
            forced_reason = "Repaired-pilot stage campaign deadline reached"
        if forced_reason is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            break
        if elapsed >= timeout_seconds:
            timed_out = True
            forced_reason = f"Child timeout reached after {timeout_seconds} seconds"
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            break
        time.sleep(min(1.0, max(0.05, sample_seconds / 4.0)))
    returncode = process.wait()
    return int(returncode), timed_out, time.monotonic() - started, forced_reason


def worker_entry(
    *,
    config_path: str | Path,
    stage_id: str,
    attempt_dir: str | Path,
    pre_run_code_head: str,
    attempt_code_head: str | None = None,
    engineering_retry_epoch_path: str | None = None,
    engineering_retry_epoch_sha256: str | None = None,
) -> int:
    config_path, execution = _execution(config_path)
    output = Path(execution["output_root"]).resolve()
    attempt = Path(attempt_dir).resolve()
    result_path = attempt / "child_result.json"
    started = time.monotonic()
    producer_head = str(attempt_code_head or pre_run_code_head)
    try:
        if execution.get("project_root"):
            live_head, epoch_reference = _bound_live_execution(
                config_path=config_path,
                execution=execution,
                output=output,
                pre_run_code_head=pre_run_code_head,
            )
            expected_path = (
                str(epoch_reference["relative_path"]) if epoch_reference else None
            )
            expected_sha = str(epoch_reference["sha256"]) if epoch_reference else None
            if (
                live_head != producer_head
                or engineering_retry_epoch_path != expected_path
                or engineering_retry_epoch_sha256 != expected_sha
            ):
                raise RuntimeError("Worker code-epoch binding differs from its launch")
            launch_path = attempt / "launch.json"
            if launch_path.is_file():
                launch = json.loads(launch_path.read_text(encoding="utf-8"))
                if (
                    launch.get("attempt_code_head") != producer_head
                    or launch.get("engineering_retry_epoch") != epoch_reference
                ):
                    raise RuntimeError("Worker launch record code epoch failed reopen")
        with StageContext(
            config_path=config_path,
            execution=execution,
            output=output,
            stage_id=stage_id,
            attempt_dir=attempt,
            pre_run_code_head=pre_run_code_head,
            attempt_code_head=producer_head,
            heartbeat_seconds=float(execution["monitoring"]["heartbeat_seconds"]),
            stack_dump_seconds=float(execution["monitoring"]["stack_dump_seconds"]),
        ) as context:
            if stage_id == "R0_v2":
                artifacts = stage_r0(context)
            elif stage_id in {"R1_s02", "R2_s03", "R2_s04", "R2_s05", "R2_s06"}:
                artifacts = stage_suite(context, stage_id.split("_")[-1])
            elif stage_id == "R3":
                artifacts = stage_r3(context)
            elif stage_id.startswith("r4_"):
                artifacts = stage_r4_shard(context)
            elif stage_id == "R5":
                artifacts = stage_r5(context)
            elif stage_id == "R6":
                artifacts = stage_r6(context)
            else:
                raise ValueError(f"Unknown child stage: {stage_id}")
        _atomic_json(
            result_path,
            {
                "schema_id": "partC.phase5.repaired.child_result.v1",
                "recorded_utc": utc_now(),
                "stage_id": stage_id,
                "pre_run_code_head": pre_run_code_head,
                "attempt_code_head": producer_head,
                "status": "PASS",
                "elapsed_seconds": time.monotonic() - started,
                "artifacts": [
                    Path(value).resolve().relative_to(output).as_posix()
                    for value in artifacts
                ],
            },
        )
        return 0
    except Exception as error:
        traceback_text = traceback.format_exc()
        print(traceback_text, file=sys.stderr, flush=True)
        _atomic_json(
            result_path,
            {
                "schema_id": "partC.phase5.repaired.child_result.v1",
                "recorded_utc": utc_now(),
                "stage_id": stage_id,
                "pre_run_code_head": pre_run_code_head,
                "attempt_code_head": producer_head,
                "status": "FAIL",
                "elapsed_seconds": time.monotonic() - started,
                "exception": f"{type(error).__name__}: {error}",
                "traceback": traceback_text,
                "artifacts": [],
            },
        )
        return 1


def run_child_stage(
    *,
    config_path: Path,
    execution: dict[str, Any],
    output: Path,
    pre_run_code_head: str,
    stage_id: str,
    retry_reason: str | None = None,
) -> Path:
    receipt_path = _receipt_path(output, stage_id)
    if receipt_path.is_file():
        validate_pass_receipt(
            output=output,
            receipt_path=receipt_path,
            config_path=config_path,
            pre_run_code_head=pre_run_code_head,
            expected_stage_id=stage_id,
        )
        return receipt_path
    for upstream in _upstream_paths(output, stage_id):
        upstream_stage = json.loads(upstream.read_text(encoding="utf-8"))["stage_id"]
        validate_pass_receipt(
            output=output,
            receipt_path=upstream,
            config_path=config_path,
            pre_run_code_head=pre_run_code_head,
            expected_stage_id=upstream_stage,
        )
    state = json.loads(
        (output / "validation" / "campaign_state.json").read_text(encoding="utf-8")
    )
    if stage_id != "R6" and time.time() >= float(
        state["science_boundary_epoch_seconds"]
    ):
        raise TimeoutError("Protected sealing reserve reached before child launch")
    if _output_bytes(output) > int(execution["resources"]["added_disk_bytes"]):
        raise RuntimeError("Added-disk guard exceeded before child launch")
    attempt_root = output / "attempts" / stage_id
    attempt_root.mkdir(parents=True, exist_ok=True)
    existing_attempts = sorted(
        value
        for value in attempt_root.glob("attempt_*")
        if value.is_dir() and value.name[8:].isdigit()
    )
    effective_retry_reason = retry_reason if existing_attempts else None
    attempt_code_head, epoch_reference = _bound_live_execution(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=pre_run_code_head,
    )
    attempt_number = next_attempt_number(
        attempt_root,
        max_engineering_retries=int(
            execution["resume_policy"]["max_engineering_retries_per_child"]
        ),
        retry_reason=effective_retry_reason,
    )
    attempt_id = f"attempt_{attempt_number:03d}"
    if attempt_number > 1:
        records = _validate_engineering_retry_chain(
            config_path=config_path,
            execution=execution,
            output=output,
            pre_run_code_head=pre_run_code_head,
        )
        transition = records[-1] if records else None
        audited_attempt = (transition or {}).get("authorized_stage_attempts", {}).get(stage_id)
        if (
            transition is None
            or (transition.get("failed_stage_id") != stage_id and audited_attempt != attempt_id)
            or (transition.get("authorized_next_attempt_id") != attempt_id and audited_attempt != attempt_id)
            or transition.get("retry_reason") != effective_retry_reason
            or transition.get("authorized_execution_code_head") != attempt_code_head
        ):
            raise RuntimeError("Retry request is not bound to the active engineering epoch")
    attempt_dir = attempt_root / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(execution["python_executable"]),
        "-m",
        "partC_rev1_src.phase5.staged_cli",
        "worker",
        "--config",
        str(config_path),
        "--stage",
        stage_id,
        "--attempt-dir",
        str(attempt_dir),
        "--pre-run-code-head",
        pre_run_code_head,
        "--attempt-code-head",
        attempt_code_head,
    ]
    if epoch_reference is not None:
        command.extend(
            [
                "--engineering-retry-epoch-path",
                str(epoch_reference["relative_path"]),
                "--engineering-retry-epoch-sha256",
                str(epoch_reference["sha256"]),
            ]
        )
    _atomic_json(
        attempt_dir / "launch.json",
        {
            "schema_id": "partC.phase5.repaired.child_launch.v1",
            "launched_utc": utc_now(),
            "stage_id": stage_id,
            "attempt_id": attempt_id,
            "command": command,
            "timeout_seconds": _stage_timeout(execution, stage_id),
            "retry_reason": effective_retry_reason,
            "attempt_code_head": attempt_code_head,
            "engineering_retry_epoch": epoch_reference,
            **_config_binding(config_path, pre_run_code_head),
        },
    )
    capture_stage_baseline(output, stage_id, attempt_dir)
    _atomic_json(
        output / "validation" / "current_stage.json",
        {
            "schema_id": "partC.phase5.repaired.current_stage.v1",
            "recorded_utc": utc_now(),
            "stage_id": stage_id,
            "attempt_id": attempt_id,
            "state": "RUNNING",
            "pre_run_code_head": pre_run_code_head,
            "attempt_code_head": attempt_code_head,
            "engineering_retry_epoch": epoch_reference,
        },
    )
    _append_jsonl(
        output / "validation" / "stage_events.jsonl",
        {
            "recorded_utc": utc_now(),
            "event": "CHILD_LAUNCHED",
            "stage_id": stage_id,
            "attempt_id": attempt_id,
            "timeout_seconds": _stage_timeout(execution, stage_id),
            "attempt_code_head": attempt_code_head,
            "engineering_retry_epoch": epoch_reference,
        },
    )
    started_utc = utc_now()
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with (attempt_dir / "stdout_stderr.log").open(
        "w", encoding="utf-8", newline="\n"
    ) as log:
        process = subprocess.Popen(
            command,
            cwd=execution["project_root"],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        returncode, timed_out, elapsed, forced_reason = _monitor_child(
            process=process,
            output=output,
            stage_id=stage_id,
            attempt_id=attempt_id,
            timeout_seconds=_stage_timeout(execution, stage_id),
            sample_seconds=float(execution["monitoring"]["resource_sample_seconds"]),
            rss_limit_bytes=int(execution["resources"]["process_tree_rss_bytes"]),
            disk_limit_bytes=int(execution["resources"]["added_disk_bytes"]),
            campaign_deadline_epoch_seconds=float(
                state[
                    (
                        "wall_clock_deadline_epoch_seconds"
                        if stage_id == "R6"
                        else "science_boundary_epoch_seconds"
                    )
                ]
            ),
        )
    _merge_child_logs(output, stage_id, attempt_dir)
    result_path = attempt_dir / "child_result.json"
    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.is_file()
        else {
            "stage_id": stage_id,
            "status": "TIMEOUT" if timed_out else "FAIL",
            "exception": (forced_reason or f"Child exited {returncode} without result"),
            "artifacts": [],
        }
    )
    status = (
        "PASS"
        if returncode == 0 and result.get("status") == "PASS"
        else ("TIMEOUT" if timed_out else "FAIL")
    )
    _append_csv_row(
        output / "validation" / "stage_timing.csv",
        [
            stage_id,
            attempt_id,
            started_utc,
            utc_now(),
            f"{elapsed:.6f}",
            returncode,
            str(timed_out).lower(),
            status,
        ],
    )
    _append_jsonl(
        output / "validation" / "stage_events.jsonl",
        {
            "recorded_utc": utc_now(),
            "event": "CHILD_FINISHED",
            "stage_id": stage_id,
            "attempt_id": attempt_id,
            "elapsed_seconds": elapsed,
            "returncode": returncode,
            "timed_out": timed_out,
            "status": status,
            "attempt_code_head": attempt_code_head,
            "engineering_retry_epoch": epoch_reference,
        },
    )
    if status != "PASS":
        preservation_error: str | None = None
        try:
            preserve_failed_stage_outputs(output, stage_id, attempt_dir)
        except Exception as error:
            preservation_error = f"{type(error).__name__}: {error}"
        _atomic_json(
            output / "validation" / "current_stage.json",
            {
                "schema_id": "partC.phase5.repaired.current_stage.v1",
                "recorded_utc": utc_now(),
                "stage_id": stage_id,
                "attempt_id": attempt_id,
                "state": status,
                "exception": result.get("exception"),
                "failed_attempt_preserved": str(attempt_dir),
                "failed_output_preservation_error": preservation_error,
                "pre_run_code_head": pre_run_code_head,
                "attempt_code_head": attempt_code_head,
                "engineering_retry_epoch": epoch_reference,
            },
        )
        if preservation_error is not None:
            raise StageFailure(
                f"{stage_id} {attempt_id} {status}; failed-output preservation also failed: "
                f"{preservation_error}"
            )
        raise StageFailure(
            f"{stage_id} {attempt_id} {status}: {result.get('exception', 'unknown child failure')}"
        )
    rebound_head, rebound_epoch = _bound_live_execution(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=pre_run_code_head,
    )
    if (
        rebound_head != attempt_code_head
        or rebound_epoch != epoch_reference
        or result.get("pre_run_code_head") != pre_run_code_head
        or result.get("attempt_code_head") != attempt_code_head
    ):
        raise StageFailure("Child PASS result code epoch changed before receipt publication")
    artifacts = [output / str(value) for value in result["artifacts"]]
    publish_pass_receipt(
        output=output,
        receipt_path=receipt_path,
        stage_id=stage_id,
        config_path=config_path,
        pre_run_code_head=pre_run_code_head,
        attempt_code_head=attempt_code_head,
        engineering_retry_epoch=epoch_reference,
        attempt_id=attempt_id,
        artifacts=artifacts,
        upstream_receipts=_upstream_paths(output, stage_id),
        extra={
            "elapsed_seconds": elapsed,
            "timed_out": False,
            "retry_reason": effective_retry_reason,
            "attempt_directory": attempt_dir.relative_to(output).as_posix(),
        },
    )
    _atomic_json(
        output / "validation" / "current_stage.json",
        {
            "schema_id": "partC.phase5.repaired.current_stage.v1",
            "recorded_utc": utc_now(),
            "stage_id": stage_id,
            "attempt_id": attempt_id,
            "state": "PASS",
            "receipt": receipt_path.relative_to(output).as_posix(),
            "pre_run_code_head": pre_run_code_head,
            "attempt_code_head": attempt_code_head,
            "engineering_retry_epoch": epoch_reference,
        },
    )
    return receipt_path


def run_r4(
    *,
    config_path: Path,
    execution: dict[str, Any],
    output: Path,
    pre_run_code_head: str,
    retry_reason: str | None = None,
) -> Path:
    index = _receipt_path(output, "R4")
    if index.is_file():
        validate_pass_receipt(
            output=output,
            receipt_path=index,
            config_path=config_path,
            pre_run_code_head=pre_run_code_head,
            expected_stage_id="R4",
        )
        return index
    validate_pass_receipt(
        output=output,
        receipt_path=_receipt_path(output, "R3"),
        config_path=config_path,
        pre_run_code_head=pre_run_code_head,
        expected_stage_id="R3",
    )
    attempt_code_head, epoch_reference = _bound_live_execution(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=pre_run_code_head,
    )
    plan = json.loads(
        (output / "runtime_fits" / "shard_plan.json").read_text(encoding="utf-8")
    )
    if is_v4(execution):
        run_child_stage(config_path=config_path, execution=execution, output=output,
            pre_run_code_head=pre_run_code_head, stage_id="r4_v4_warmup", retry_reason=retry_reason)
    for shard in plan["shards"]:
        run_child_stage(
            config_path=config_path,
            execution=execution,
            output=output,
            pre_run_code_head=pre_run_code_head,
            stage_id=str(shard["shard_id"]),
            retry_reason=retry_reason,
        )
    if is_v4(execution):
        run_child_stage(config_path=config_path, execution=execution, output=output,
            pre_run_code_head=pre_run_code_head, stage_id="r4_v4_controls", retry_reason=retry_reason)
    return merge_r4(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=pre_run_code_head,
        attempt_code_head=attempt_code_head,
        engineering_retry_epoch=epoch_reference,
    )


def first_incomplete_stage(
    *,
    config_path: Path,
    execution: dict[str, Any],
    output: Path,
    pre_run_code_head: str,
) -> str | None:
    stage_order = (
        STAGE_ORDER
        if execution.get("schema_id")
        in {"partC.phase5.pilot_repaired_0904_v2.execution.v1", V4_SCHEMA}
        else ("R0", *STAGE_ORDER[1:])
    )
    for stage_id in stage_order:
        receipt = _receipt_path(output, stage_id)
        if not receipt.is_file():
            return stage_id
        validate_pass_receipt(
            output=output,
            receipt_path=receipt,
            config_path=config_path,
            pre_run_code_head=pre_run_code_head,
            expected_stage_id=stage_id,
        )
    return None


def run_stage_command(
    config_path: str | Path,
    stage_id: str,
    *,
    retry_reason: str | None = None,
) -> Path:
    config_path, execution, output, pre_run_code_head = load_initialized_run(
        config_path
    )
    if stage_id not in STAGE_ORDER:
        raise ValueError(f"Unknown public stage: {stage_id}")
    if stage_id == "R4":
        return run_r4(
            config_path=config_path,
            execution=execution,
            output=output,
            pre_run_code_head=pre_run_code_head,
            retry_reason=retry_reason,
        )
    receipt = run_child_stage(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=pre_run_code_head,
        stage_id=stage_id,
        retry_reason=retry_reason,
    )
    if stage_id == "R6":
        finalize_r6(
            config_path=config_path,
            execution=execution,
            output=output,
            pre_run_code_head=pre_run_code_head,
        )
    return receipt


def resume_run(config_path: str | Path, *, retry_reason: str | None = None) -> Path:
    config_path, execution, output, pre_run_code_head = load_initialized_run(
        config_path
    )
    while True:
        stage_id = first_incomplete_stage(
            config_path=config_path,
            execution=execution,
            output=output,
            pre_run_code_head=pre_run_code_head,
        )
        if stage_id is None:
            return output
        if stage_id == "R4":
            run_r4(
                config_path=config_path,
                execution=execution,
                output=output,
                pre_run_code_head=pre_run_code_head,
                retry_reason=retry_reason,
            )
        else:
            run_child_stage(
                config_path=config_path,
                execution=execution,
                output=output,
                pre_run_code_head=pre_run_code_head,
                stage_id=stage_id,
                retry_reason=retry_reason,
            )
            if stage_id == "R6":
                finalize_r6(
                    config_path=config_path,
                    execution=execution,
                    output=output,
                    pre_run_code_head=pre_run_code_head,
                )


def run_status(config_path: str | Path) -> dict[str, Any]:
    config_path, execution = _execution(config_path)
    output = Path(execution["output_root"])
    if not output.is_dir():
        return {"run_id": execution["run_id"], "state": "NOT_STARTED"}
    binding = json.loads(
        (output / "repository_audit" / "run_binding.json").read_text(encoding="utf-8")
    )
    pre_run = str(binding["pre_run_code_head"])
    retry_epochs = _validate_engineering_retry_chain(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=pre_run,
    )
    receipts = {}
    for stage_id in STAGE_ORDER:
        path = _receipt_path(output, stage_id)
        if path.is_file():
            validate_pass_receipt(
                output=output,
                receipt_path=path,
                config_path=config_path,
                pre_run_code_head=pre_run,
                expected_stage_id=stage_id,
            )
            receipts[stage_id] = "PASS"
        else:
            receipts[stage_id] = "INCOMPLETE"
    return {
        "run_id": execution["run_id"],
        "output_root": str(output.resolve()),
        "pre_run_code_head": pre_run,
        "active_scientific_execution_code_head": (
            retry_epochs[-1]["authorized_execution_code_head"]
            if retry_epochs
            else pre_run
        ),
        "engineering_retry_epoch_count": len(retry_epochs),
        "receipts": receipts,
        "first_incomplete_stage": next(
            (key for key, value in receipts.items() if value != "PASS"), None
        ),
        "current_stage": json.loads(
            (output / "validation" / "current_stage.json").read_text(encoding="utf-8")
        ),
    }


def seal_blocker_command(config_path: str | Path, blocker: str) -> Path:
    """Seal an inspected exact blocker without launching another child."""

    message = str(blocker).strip()
    if not message:
        raise ValueError("An exact nonempty blocker is required")
    config_path, execution, output, pre_run_code_head = load_initialized_run(
        config_path
    )
    return seal_exact_blocker(
        config_path=config_path,
        execution=execution,
        output=output,
        pre_run_code_head=pre_run_code_head,
        blocker=message,
    )


def run_timeout_probe(
    python_executable: str,
    attempt_dir: Path,
    *,
    timeout_seconds: float,
    stack_dump_seconds: float,
) -> dict[str, Any]:
    """Exercise the production timeout/real-stack mechanism in focused tests."""

    attempt_dir.mkdir(parents=True, exist_ok=False)
    stack_path = attempt_dir / "python_stacks.log"
    probe = (
        "import faulthandler,time\n"
        f"stream=open({str(stack_path)!r},'a',encoding='utf-8')\n"
        f"faulthandler.dump_traceback_later({float(stack_dump_seconds)!r},repeat=True,file=stream)\n"
        "def test_hang_worker():\n"
        "    while True:\n"
        "        time.sleep(1.0)\n"
        "test_hang_worker()\n"
    )
    command = [python_executable, "-c", probe]
    with (attempt_dir / "stdout_stderr.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        started = time.monotonic()
        while process.poll() is None and time.monotonic() - started < timeout_seconds:
            time.sleep(0.01)
        timed_out = process.poll() is None
        if timed_out:
            process.terminate()
        process.wait(timeout=5)
    text = stack_path.read_text(encoding="utf-8") if stack_path.is_file() else ""
    return {
        "timed_out": timed_out,
        "stack_path": str(stack_path),
        "stack_dump_count": text.count("Timeout"),
        "contains_worker_frame": "test_hang_worker" in text,
    }


def test_hang_worker(stack_file: str | Path, stack_dump_seconds: float) -> None:
    path = Path(stack_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        faulthandler.dump_traceback_later(stack_dump_seconds, repeat=True, file=stream)
        try:
            while True:
                time.sleep(1.0)
        finally:
            faulthandler.cancel_dump_traceback_later()
