"""Compact synchronization handoff for the repaired staged Phase 5 pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

import pandas as pd
from .v4_evidence import is_v4

from .handoff import (
    HARD_ZIP_BYTES,
    MAX_MEMBER_BYTES,
    TARGET_ZIP_BYTES,
    _contained,
    _copy,
    _copy_tree_files,
    _internal_manifest,
    _privacy_scan,
    _stable_sample,
    _validated_handoff_id,
    _write_internal_checksums,
    _zip_write,
)
from .pilot import (
    BLOCKED_STATUS,
    FINAL_STATUS,
    _describe_reopen,
    canonical_json,
    git,
    sha256_file,
    utc_now,
    write_json,
    write_text,
)
from .staged_pilot import (
    BLOCKED_HANDOFF_SHA256,
    BLOCKED_V1_HANDOFF_SHA256,
    ORIGINAL_AUTHORITY_SHA256,
    PARENT_CONTINUATION_SHA256,
    PRIOR_REPAIRED_OUTER_SHA256,
    REPAIRED_OUTER_SHA256,
    REPAIR_OF_RUN_ID,
    STAGE_ORDER,
    _execution,
    _receipt_path,
    _validate_repaired_required_outputs,
    _validate_engineering_retry_chain,
    authority_root,
    qualified_repaired_git_state,
    validate_pass_receipt,
)


def _git_diagnostic_text(repo: Path, *arguments: str) -> str:
    """Capture archival Git text without depending on the Windows code page."""
    completed = subprocess.run(
        ["git", "-C", str(repo), "-c", "core.longpaths=true", *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.decode("utf-8", errors="replace")


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


def _copy_small_evidence(
    output: Path, destination: Path, *, roots: Iterable[str]
) -> list[str]:
    allowed = {
        ".md",
        ".txt",
        ".json",
        ".jsonl",
        ".csv",
        ".png",
        ".sha256",
        ".log",
    }
    copied: list[str] = []
    for root_name in roots:
        root = output / root_name
        if not root.exists():
            continue
        for path in sorted(value for value in root.rglob("*") if value.is_file()):
            failed_partial = path.suffix.lower() == ".failed-partial"
            content_suffix = (
                Path(path.stem).suffix.lower()
                if failed_partial
                else path.suffix.lower()
            )
            if (
                content_suffix not in allowed
                or path.stat().st_size > MAX_MEMBER_BYTES
            ):
                continue
            relative = path.relative_to(output)
            if failed_partial:
                relative_without_marker = relative.with_suffix("")
            else:
                relative_without_marker = relative
            target_relative = (
                relative_without_marker.with_suffix(
                    relative_without_marker.suffix + ".txt"
                )
                if content_suffix == ".jsonl"
                else relative_without_marker
            )
            _copy(path, destination / target_relative)
            copied.append(
                relative.as_posix()
                + (
                    " -> " + target_relative.as_posix()
                    if relative != target_relative
                    else ""
                )
            )
    return copied


def _copy_complete_installed_authority(
    source: Path, destination: Path, *, omit_archives: bool = False
) -> dict[str, Any]:
    """Copy and hash-close every installed authority member.

    The installed authority is the compact, authenticated contract snapshot;
    its larger source ZIPs remain externally hash-bound under the package size
    policy.  Suffix filtering is unsafe here because the authority's own
    checksum file also binds Python and diagnostic log evidence.
    """

    if not source.is_dir():
        raise FileNotFoundError(f"Installed authority root is absent: {source}")
    source_files = sorted(value for value in source.rglob("*") if value.is_file())
    excluded = []
    if omit_archives:
        excluded = [{"relative_path":p.relative_to(source).as_posix(), "size_bytes":p.stat().st_size,"sha256":sha256_file(p),"policy":"EXTERNALLY_HASH_BOUND_OLD_ARCHIVE_NOT_RECURSIVELY_EMBEDDED"} for p in source_files if p.suffix.lower()==".zip"]
        source_files = [p for p in source_files if p.suffix.lower()!=".zip"]
    if not source_files:
        raise RuntimeError("Installed authority snapshot is empty")
    expected = {
        path.relative_to(source).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in source_files
    }
    for path in source_files:
        _copy(path, destination / path.relative_to(source))
    actual_files = sorted(
        value for value in destination.rglob("*") if value.is_file()
    )
    actual = {
        path.relative_to(destination).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in actual_files
    }
    expected_paths = set(expected)
    actual_paths = set(actual)
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    hash_or_size_mismatches = sorted(
        path
        for path in expected_paths & actual_paths
        if expected[path] != actual[path]
    )
    status = (
        "PASS"
        if not missing and not extra and not hash_or_size_mismatches
        else "FAIL"
    )
    result: dict[str, Any] = {
        "schema_id": "partC.phase5.repaired.installed_authority_copy_closure.v1",
        "source_root": str(source.resolve()),
        "source_file_count": len(expected),
        "copied_file_count": len(actual),
        "source_total_bytes": sum(
            int(value["size_bytes"]) for value in expected.values()
        ),
        "copied_total_bytes": sum(
            int(value["size_bytes"]) for value in actual.values()
        ),
        "missing": missing,
        "extra": extra,
        "hash_or_size_mismatches": hash_or_size_mismatches,
        "status": status,
        "excluded_external_archive_members": excluded,
    }
    if status != "PASS":
        raise RuntimeError(f"Installed authority copy closure failed: {result}")
    return result


def _write_bounded_review_sample(frame, target, maximum, keys):
    """Bound review bytes while keeping full columns and essential path witnesses."""
    if frame.empty:
        raise RuntimeError("A required representative source table is empty")
    priority = frame.iloc[:0]
    if {"evaluation_role", "absolute_relative_cost_error"}.issubset(frame.columns):
        ranked = frame.sort_values(
            ["absolute_relative_cost_error", *keys],
            ascending=[False, *([True] * len(keys))], kind="mergesort",
        )
        priority = pd.concat([
            ranked.head(20), ranked.groupby("evaluation_role", sort=True).head(1)
        ]).drop_duplicates(keys).sort_values(keys, kind="mergesort")
    mandatory_keys = pd.MultiIndex.from_frame(priority[keys])
    remaining = frame.loc[~pd.MultiIndex.from_frame(frame[keys]).isin(mandatory_keys)]
    count = min(maximum, len(frame))
    minimum = max(1, len(priority))
    if count < minimum:
        raise RuntimeError("Review row limit would omit mandatory path witnesses")
    iterations = 0
    while True:
        sample = pd.concat([
            priority, _stable_sample(remaining, count - len(priority), keys)
        ], ignore_index=True)
        sample.to_csv(target, index=False)
        iterations += 1
        size = target.stat().st_size
        if size <= MAX_MEMBER_BYTES:
            return sample, {
                "sample_size_bytes": size, "member_byte_limit": MAX_MEMBER_BYTES,
                "byte_bound_iterations": iterations,
                "mandatory_worst_and_source_role_rows": len(priority),
                "all_source_columns_preserved": True,
            }
        if count <= minimum:
            raise RuntimeError("Mandatory review witnesses alone exceed the member byte limit")
        count = max(minimum, min(count - 1, int(count * MAX_MEMBER_BYTES / size * .90)))


def _write_representative_samples(
    output: Path, destination: Path
) -> list[dict[str, Any]]:
    specifications = (
        (
            "topology/chain_candidates.parquet",
            5000,
            ["suite_short_id", "chain_id"],
        ),
        (
            "bundle_oracle/oracle_sample_results.parquet",
            5000,
            ["canonical_macro_signature", "query_flow", "query_role"],
        ),
        (
            "runtime_fits/runtime_fit_results.parquet",
            5000,
            ["canonical_macro_signature", "runtime_class", "variant_id"],
        ),
        (
            "charge_audit/bacn_visit_semantics.parquet",
            5000,
            ["suite_short_id", "benchmark_id", "bacn_id"],
        ),
        (
            "charge_audit/fixed_charge_candidates.parquet",
            5000,
            ["suite_short_id", "bacn_id"],
        ),
        (
            "path_validation/path_cost_sample.parquet",
            5000,
            ["evaluation_role", "sample_od_id"],
        ),
    )
    receipts = []
    destination.mkdir(parents=True, exist_ok=True)
    for relative, maximum, keys in specifications:
        source = output / relative
        frame = pd.read_parquet(source)
        missing = sorted(set(keys) - set(frame.columns))
        if missing:
            raise RuntimeError(
                f"Representative sample keys are absent for {relative}: {missing}"
            )
        target = destination / relative.replace("/", "__").replace(
            ".parquet", "__sample.csv"
        )
        sample, byte_metadata = _write_bounded_review_sample(frame, target, maximum, keys)
        receipts.append(
            {
                "source": relative,
                "source_rows": len(frame),
                "sample_rows": len(sample),
                "key_columns": keys,
                "sample_method": "lowest_sha256_of_declared_stable_key_with_byte_bound; path samples force worst20 and worst-per-source-role",
                "sample_path": target.name,
                "sample_sha256": sha256_file(target),
                **byte_metadata,
            }
        )
    write_json(destination / "SAMPLE_METHODS.json", receipts)
    return receipts


def _verify_external_output_closure(output: Path) -> dict[str, Any]:
    manifest_path = output / "manifests" / "MANIFEST.csv"
    checksums_path = output / "manifests" / "CHECKSUMS.sha256"
    manifest = pd.read_csv(manifest_path)
    required = {"relative_path", "size_bytes", "sha256", "reopen_status"}
    if manifest.empty or not required.issubset(manifest.columns):
        raise RuntimeError("External output manifest is empty or incomplete")
    if not manifest["relative_path"].is_unique:
        raise RuntimeError("External output manifest paths are not unique")
    declared = set(manifest["relative_path"].astype(str))
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path not in {manifest_path, checksums_path}
    }
    failures = []
    for row in manifest.itertuples(index=False):
        path = output / str(row.relative_path)
        if (
            not path.is_file()
            or path.stat().st_size != int(row.size_bytes)
            or sha256_file(path) != str(row.sha256)
            or str(row.reopen_status) != "PASS"
            or _describe_reopen(path)[0] != "PASS"
        ):
            failures.append(str(row.relative_path))
    checksum_failures = []
    checksum_members = set()
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksum_members.add(relative)
        path = output / relative
        if not path.is_file() or sha256_file(path) != digest:
            checksum_failures.append(relative)
    expected_checksum_members = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path != checksums_path
    }
    status = (
        "PASS"
        if declared == actual
        and not failures
        and checksum_members == expected_checksum_members
        and not checksum_failures
        else "FAIL"
    )
    result = {
        "schema_id": "partC.phase5.repaired.external_output_closure.v1",
        "manifest_rows": len(manifest),
        "manifest_member_set_match": declared == actual,
        "manifest_failures": failures,
        "checksum_member_set_match": checksum_members == expected_checksum_members,
        "checksum_failures": checksum_failures,
        "manifest_sha256": sha256_file(manifest_path),
        "checksums_sha256": sha256_file(checksums_path),
        "status": status,
    }
    if status != "PASS":
        raise RuntimeError(f"External output closure failed: {result}")
    return result


def _copy_repository_records(repo: Path, destination: Path) -> None:
    names = (
        "PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_STATUS.json",
        "PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_RESULT.md",
        "PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_PRELAUNCH_TESTS.json",
        "PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_PRELAUNCH_TESTS.md",
        "PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_R3_ENGINEERING_REPAIR.json",
        "PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_R3_ENGINEERING_REPAIR.md",
        "PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_R5_ENGINEERING_REPAIR.json",
        "PHASE5_BACKBONE_COMPILATION_PILOT_REPAIRED_0904_V2_R5_ENGINEERING_REPAIR.md",
    )
    source_root = repo / "partC_doc_icnn" / "phase_5_work"
    for name in names:
        path = source_root / name
        if path.is_file():
            _copy(path, destination / name)
    for path in sorted(source_root.glob("PHASE5_V4_0905_*")):
        if path.is_file():
            _copy(path, destination / path.name)


def build_repaired_handoff(
    config_path: str | Path, *, handoff_id: str, ending_head: str
) -> Path:
    """Build exactly one write-new repaired synchronization handoff."""

    handoff_id = _validated_handoff_id(handoff_id)
    config_path, execution = _execution(config_path)
    repo = Path(execution["project_root"]).resolve()
    output = Path(execution["output_root"]).resolve()
    branch = git(repo, "branch", "--show-current").stdout.strip()
    live_head = git(repo, "rev-parse", "HEAD").stdout.strip()
    if branch != execution["expected_branch"] or live_head != str(ending_head):
        raise RuntimeError("Handoff must bind the live branch and exact result HEAD")
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
            raise RuntimeError("Handoff result HEAD does not descend every execution epoch")
    status = json.loads(
        (output / "reports" / "08_PILOT_STATUS.json").read_text(encoding="utf-8")
    )
    resource = json.loads(
        (output / "validation" / "resource_usage.json").read_text(encoding="utf-8")
    )
    scientific_status = str(status.get("status"))
    if scientific_status not in {FINAL_STATUS, BLOCKED_STATUS}:
        raise RuntimeError(f"Unknown repaired scientific status: {scientific_status}")
    post = json.loads(
        (output / "repository_audit" / "post_checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    if not (
        status.get("post_checkpoint_head_bound") is True
        and status.get("ending_head") == live_head
        and status.get("pre_run_code_head") == pre_run_head
        and post.get("branch") == branch
        and post.get("pre_run_code_head") == pre_run_head
        and post.get("ending_head") == live_head
        and post.get("scientific_status") == scientific_status
        and post.get("status") == "PASS"
    ):
        raise RuntimeError("Repaired post-checkpoint binding is incomplete")
    output_closure = _verify_external_output_closure(output)
    if scientific_status == FINAL_STATUS:
        required = _validate_repaired_required_outputs(output, execution)
        acceptance = pd.read_csv(output / "validation" / "acceptance_test_results.csv")
        if (
            required["status"] != "PASS"
            or acceptance.empty
            or not acceptance["status"].eq("PASS").all()
            or resource.get("status") != "PASS"
        ):
            raise RuntimeError("Successful repaired output is not handoff-eligible")
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
    else:
        acceptance = pd.DataFrame()
        required = {"status": "NOT_APPLICABLE_BLOCKED"}
        if not status.get("blockers") or resource.get("status") != "BLOCKED":
            raise RuntimeError("Blocked repaired output lacks exact blocker closure")

    source_parent = _contained(
        repo,
        "partC_doc_icnn",
        "handoffs",
        "phase_05",
        "step_pilot_v4_0905" if is_v4(execution) else "step_backbone_compilation_pilot_repaired_0904_v2",
    )
    bundle_parent = _contained(repo, "artifacts_icnn", "handoff_bundles", "phase_05")
    final_source = _contained(source_parent, handoff_id)
    building_source = _contained(source_parent, f".{handoff_id}.building")
    zip_path = _contained(
        bundle_parent,
        f"phase05_steppilot_v4_0905_{handoff_id}_SYNC.zip" if is_v4(execution) else f"phase05_stepbackbone_compilation_pilot_repaired_0904_v2_{handoff_id}_SYNC.zip",
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
        raise FileExistsError(f"Write-new repaired handoff target exists: {collisions}")
    building_source.mkdir(parents=True, exist_ok=False)
    source = building_source
    for name in (
        "governing_docs",
        "configs",
        "code_snapshot",
        "code_diff",
        "summary_data",
        "representative_samples",
        "figures",
        "geospatial_summary",
        "log_excerpts",
        "external_artifact_index",
        "internal_validation",
    ):
        (source / name).mkdir(parents=True, exist_ok=False)
    write_text(
        source / "geospatial_summary" / "NOT_APPLICABLE.md",
        "# Geospatial summary\n\n"
        "Not applicable to the authorized Phase 5 backbone-compilation pilot repair. "
        "No geospatial supplement or new geospatial product was authorized or produced.\n",
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
        _copy(
            repo / "partC_doc_icnn" / name,
            source / "governing_docs" / "repository" / name,
        )
    repaired_authority = authority_root(execution)
    authority_copy_closure = _copy_complete_installed_authority(
        repaired_authority,
        source / "governing_docs" / "repaired_phase5_authority",
        omit_archives=is_v4(execution),
    )
    _copy(config_path, source / "configs" / config_path.name)
    for field in (
        "required_output_artifacts_relative_path",
        "inherited_status_schema_relative_path",
    ):
        configured = (repo / str(execution[field])).resolve()
        if not configured.is_file() or not configured.is_relative_to(repo):
            raise FileNotFoundError(f"Configured handoff contract file is absent: {field}")
        _copy(configured, source / "configs" / configured.name)
    for path in _code_snapshot_paths(repo):
        if path.is_file():
            _copy(path, source / "code_snapshot" / path.relative_to(repo))
    repair_base = str(execution["repair_starting_head"])
    write_text(
        source / "code_diff" / "phase5_repaired_checkpoint.patch",
        _git_diagnostic_text(repo, "diff", "--no-ext-diff", repair_base, live_head),
    )
    write_text(
        source / "code_diff" / "diff_stat.txt",
        _git_diagnostic_text(repo, "diff", "--stat", repair_base, live_head),
    )
    write_text(
        source / "code_diff" / "git_log.txt",
        _git_diagnostic_text(repo, "log", "--oneline", f"{repair_base}..{live_head}"),
    )
    changes = _git_diagnostic_text(
        repo, "diff", "--name-status", repair_base, live_head
    ).splitlines()
    pd.DataFrame(
        [
            {
                "status": value.split("\t", 1)[0],
                "repository_path": value.split("\t", 1)[1] if "\t" in value else "",
            }
            for value in changes
        ],
        columns=["status", "repository_path"],
    ).to_csv(source / "07_CODE_CHANGE_INDEX.csv", index=False)

    copied = _copy_small_evidence(
        output,
        source / "summary_data" / "external_output",
        roots=(
            "reports",
            "validation",
            "repository_audit",
            "manifests",
            "cache",
            "stage_receipts",
            "topology",
            "bundle_oracle",
            "runtime_fits",
            "charge_audit",
            "interface_audit",
            "path_validation",
            "benchmarks",
            "suite_products",
            "r4_shards",
        ),
    )
    attempt_logs = _copy_small_evidence(
        output,
        source / "log_excerpts",
        roots=("attempts",),
    )
    _copy_repository_records(repo, source / "summary_data" / "repository_records")
    _copy_tree_files(
        output / "figures", source / "figures", ("*.png", "source_data/*.csv")
    )
    if scientific_status == FINAL_STATUS:
        samples = _write_representative_samples(
            output, source / "representative_samples"
        )
    else:
        samples = []
        write_text(
            source / "representative_samples" / "NOT_AVAILABLE.md",
            "# Representative samples unavailable\n\nThe repaired pilot stopped at the exact recorded blocker.",
        )
    _copy(
        output / "manifests" / "MANIFEST.csv",
        source / "external_artifact_index" / "full_output_MANIFEST.csv",
    )
    _copy(
        output / "manifests" / "CHECKSUMS.sha256",
        source / "external_artifact_index" / "full_output_CHECKSUMS.sha256",
    )
    write_json(
        source / "internal_validation" / "EXTERNAL_OUTPUT_CLOSURE.json",
        output_closure,
    )
    write_json(
        source / "internal_validation" / "REPAIRED_REQUIRED_OUTPUTS.json",
        required,
    )
    write_json(
        source / "internal_validation" / "COMPACT_COPY_INDEX.json",
        {
            "copied_small_output_file_count": len(copied),
            "copied_attempt_log_file_count": len(attempt_logs),
            "representative_sample_count": len(samples),
            "full_parquet_products_external_only": True,
            "external_output_root": str(output),
            "status": "PASS",
        },
    )
    write_json(
        source / "internal_validation" / "AUTHORITY_COPY_CLOSURE.json",
        authority_copy_closure,
    )
    write_json(source / "internal_validation" / "QUALIFIED_GIT_STATUS.json", git_state)

    reachability_path = (
        output / "validation" / "reachability_and_physical_conservation_v2.json"
    )
    reachability = (
        json.loads(reachability_path.read_text(encoding="utf-8"))
        if reachability_path.is_file()
        else {}
    )
    receipt_count = len(list((output / "stage_receipts").glob("*.json")))
    headline = {
        "reachable_demand": reachability.get("reachable_demand"),
        "unreachable_demand": reachability.get("unreachable_demand"),
        "total_demand": reachability.get("total_demand"),
        "scc_count": reachability.get("scc_count"),
        "scc_label_payload_sha256": reachability.get("scc_label_payload_sha256"),
        "cross_scc_candidate_row_count": reachability.get(
            "cross_scc_candidate_row_count"
        ),
        "cross_scc_reachable_row_count": reachability.get(
            "cross_scc_reachable_row_count"
        ),
        "stage_receipt_count": receipt_count,
        "sampled_macro_signatures": status.get("outputs", {}).get(
            "sampled_macro_signatures"
        ),
        "acceptance_test_count": len(acceptance),
    }
    write_text(
        source / "00_HANDOFF_README.md",
        "# Upload this repaired Phase 5 pilot v2 handoff\n\n"
        f"Upload `{zip_path.name}` for external review. Scientific status is "
        f"`{scientific_status}`. No later step is authorized.",
    )
    write_text(
        source / "01_STEP_SUMMARY.md",
        "# Repaired Phase 5 backbone-compilation pilot v2 summary\n\n"
        "## Authorized objective and scope\n\n"
        f"The write-new run `{execution['run_id']}` replaced the stalled repaired-v1 SCC path "
        "with the authorized iterative Numba Kosaraju engine and exact cross-SCC audit while preserving topology, "
        "oracle, fit, and acceptance definitions. It used production suites s02-s06, low x^45 "
        "CURRENT as primary, high x^200 only as held-out validation, and capacity factor 0.30 "
        "already applied once. No assignment, model training, Phase 6 work, or full Phase 5 "
        "campaign was performed.\n\n"
        "## Authority, code, and result\n\n"
        f"Authority ZIP hashes are SCC-repair/resume `{REPAIRED_OUTER_SHA256}`, prior repaired-v1 "
        f"`{PRIOR_REPAIRED_OUTER_SHA256}`, original `{ORIGINAL_AUTHORITY_SHA256}`, repaired-v1 blocked evidence "
        f"`{BLOCKED_V1_HANDOFF_SHA256}`, original blocked evidence `{BLOCKED_HANDOFF_SHA256}`, and parent "
        f"`{PARENT_CONTINUATION_SHA256}`. The config SHA-256 is `{sha256_file(config_path)}`; "
        f"the pre-run implementation HEAD is `{pre_run_head}`, the controlled engineering-retry "
        f"epoch count is `{len(retry_epochs)}`, and result HEAD is `{live_head}`.\n\n"
        "## Evidence and headline facts\n\n"
        f"Scientific status is `{scientific_status}`. Headline evidence is "
        f"`{canonical_json(headline)}`. All completed stage receipts, attempts, heartbeats, "
        "actual stack logs, resource history, compact summaries, representative samples, code "
        "snapshot/diff, and the full external byte index are included.\n\n"
        "## Review boundary\n\n"
        "This handoff recommends external review only. Full Phase 5, final topology/runtime "
        "thresholds, Phase 6, national ACNHN assignment, and ICNN training remain unauthorized.",
    )
    handoff_status = {
        "schema_id": "partC.phase5.repaired_0904_v2.handoff.status.v1",
        "handoff_id": handoff_id,
        "campaign_id": execution["campaign_id"],
        "run_id": execution["run_id"],
        "repair_of_run_id": REPAIR_OF_RUN_ID,
        "status": "WAITING_FOR_EXTERNAL_REVIEW",
        "scientific_status": scientific_status,
        "branch": branch,
        "repair_starting_head": repair_base,
        "pre_run_code_head": pre_run_head,
        "ending_head": live_head,
        "engineering_retry_code_epochs": status.get(
            "engineering_retry_code_epochs", []
        ),
        "installed_authority_copy_status": authority_copy_closure["status"],
        "installed_authority_copy_file_count": authority_copy_closure[
            "copied_file_count"
        ],
        "blockers": status.get("blockers", []),
        "next_step_authorized": False,
        "full_phase5_authorized": False,
        "final_reduced_topology_accepted": False,
        "final_runtime_thresholds_frozen": False,
        "phase6_authorized": False,
        "national_assignment_performed": False,
        "icnn_training_performed": False,
    }
    write_json(source / "02_HANDOFF_STATUS.json", handoff_status)
    decision_source = output / "reports" / "07_DECISIONS_RECOMMENDED_FOR_FULL_PHASE5.md"
    if decision_source.is_file():
        _copy(decision_source, source / "03_DECISIONS_AND_OPEN_QUESTIONS.md")
    else:
        write_text(
            source / "03_DECISIONS_AND_OPEN_QUESTIONS.md",
            "# Decisions and open questions\n\nNo scientific decision is advanced from incomplete evidence.",
        )
    write_text(
        source / "04_VALIDATION_RESULTS.md",
        "# Validation results\n\n"
        + (
            f"All `{len(acceptance)}` declared acceptance rows pass. Receipt, input-invariance, "
            "runtime-root, output-reopen, external-manifest, and fresh-extraction closure pass."
            if scientific_status == FINAL_STATUS
            else "Success was not claimed. The exact blocker and every earlier PASS receipt are preserved."
        ),
    )
    pd.DataFrame([resource]).to_csv(source / "05_RESOURCE_USAGE.csv", index=False)
    write_text(
        source / "08_COMMAND_LOG.md",
        "# Command log\n\n"
        f"- `{execution['python_executable']} -m pytest -q <original and repaired Phase 5 focused tests>`\n"
        f"- `{execution['python_executable']} -m partC_rev1_src.phase5.staged_cli resume --config {config_path}`\n"
        "- Two `authorize-retry` records separately bound the preserved R3 native failure and R5 benchmark-attribution failure to committed, tested engineering repairs before their respective `attempt_002` runs.\n"
        f"- `git -C {repo} -c core.longpaths=true commit ...` (pre-run, controlled engineering-repair, and result checkpoints only)\n"
        f"- `{execution['python_executable']} -m partC_rev1_src.phase5.staged_cli reseal --config {config_path} --ending-head {live_head}`\n"
        "- No network use or package installation occurred.\n",
    )
    write_text(
        source / "09_NEXT_STEP_RECOMMENDATION.md",
        "# Next-step recommendation\n\nExternal review should issue an explicit PROCEED, REPAIR, or BLOCK decision. No next step is authorized by this package.",
    )
    write_text(source / "10_UPLOAD_THIS_FILE.txt", str(zip_path.resolve()))
    write_text(
        source / "11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md",
        "# Changelog from repaired-v1 blocker\n\n"
        f"This is the authorized SCC repair and resume of `{REPAIR_OF_RUN_ID}`. The prior "
        "authorities and blocked handoffs remain immutable evidence; this package does not rewrite "
        "or silently supersede those records. The R3 native failure, R5 benchmark-attribution "
        "failure, their separate authorized retries, and the terminal R5 hard-gate blocker are "
        "all preserved and code-epoch bound.",
    )

    if is_v4(execution):
        summary = (repo / "partC_doc_icnn/phase_5_work/PHASE5_V4_0905_RESULT.md").read_text(encoding="utf-8")
        write_text(source / "01_STEP_SUMMARY.md", summary)
        write_text(source / "00_HANDOFF_README.md", "# Upload the v4 pilot synchronization ZIP\n\n" + f"Scientific status: `{scientific_status}`. Production recommendation: `NOT_READY_MODEL_LIMITATIONS`. The current v4 authority is included; the older reviewed ZIP is indexed by hash and deliberately not embedded. Upload `{zip_path.name}` for external review.")
        write_text(source / "08_COMMAND_LOG.md", "# Actual v4 command evidence\n\nExact launched commands are in each attempts/*/launch.json; ordered stage timings, events, resource samples and retry epochs are included. The existing staged CLI operated the new v4 output root. See repository result report for checkpoint commands. No old campaign was resumed.")
        write_text(source / "11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md", "# Changes from reviewed v2\n\nThe working v2 SCC/cache is preserved and imported with verified dependencies. V4 regenerates topology with reciprocal atomicity and static-incidence closure, certifies primitive primal/KKT/FW conditions in activation-relative coordinates, fits dynamic functions separately from fixed charges, corrects knot-aware validation, and reports static path charge incidence, bounded retained controls and full-universe engineering forecasts. All old failed evidence is preserved.")
        handoff_status.update({"schema_id":"partC.phase5.v4.handoff.status.v1", "repair_of_run_id":status["repair_of_run_id"],
            "authority_sha256":execution["authority_zip_sha256"],"production_recommendation":status["production_recommendation"],"external_output_root":str(output)})
        write_json(source / "02_HANDOFF_STATUS.json",handoff_status)
        write_json(source / "governing_docs/EXTERNALLY_BOUND_ARCHIVES.json",authority_copy_closure["excluded_external_archive_members"])
    privacy = _privacy_scan(source)
    write_json(source / "internal_validation" / "PRIVACY_SCAN.json", privacy)
    if privacy["status"] != "PASS":
        raise RuntimeError(
            f"Repaired handoff privacy scan failed: {privacy['findings']}"
        )
    manifest = _internal_manifest(source)
    manifest.to_csv(
        source / "06_ARTIFACT_MANIFEST.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    _write_internal_checksums(source)
    members = _zip_write(source, building_zip)
    zip_sha = sha256_file(building_zip)
    write_text(building_checksum, f"{zip_sha}  {zip_path.name}")
    companion = {
        "schema_id": "partC.phase5.repaired_0904_v2.handoff.companion.v1",
        "created_utc": utc_now(),
        "handoff_id": handoff_id,
        "source_folder": str(final_source.resolve()),
        "zip_path": str(zip_path.resolve()),
        "zip_size_bytes": building_zip.stat().st_size,
        "zip_sha256": zip_sha,
        "target_zip_limit_bytes": TARGET_ZIP_BYTES,
        "hard_zip_limit_bytes": HARD_ZIP_BYTES,
        "zip_member_count": len(members),
        "fresh_extraction_checksum_status": "PASS",
        "fresh_extraction_manifest_status": "PASS",
        "privacy_scan_status": privacy["status"],
        "branch": branch,
        "repair_starting_head": repair_base,
        "pre_run_code_head": pre_run_head,
        "ending_head": live_head,
        "engineering_retry_code_epochs": status.get(
            "engineering_retry_code_epochs", []
        ),
        "campaign_id": execution["campaign_id"],
        "run_id": execution["run_id"],
        "repair_of_run_id": REPAIR_OF_RUN_ID,
        "scientific_status": scientific_status,
        "blockers": status.get("blockers", []),
        "scc_repair_resume_authority_sha256": REPAIRED_OUTER_SHA256,
        "prior_repaired_authority_sha256": PRIOR_REPAIRED_OUTER_SHA256,
        "original_authority_sha256": ORIGINAL_AUTHORITY_SHA256,
        "blocked_repaired_v1_handoff_sha256": BLOCKED_V1_HANDOFF_SHA256,
        "original_blocked_handoff_sha256": BLOCKED_HANDOFF_SHA256,
        "parent_continuation_sha256": PARENT_CONTINUATION_SHA256,
        "installed_authority_copy_status": authority_copy_closure["status"],
        "installed_authority_copy_file_count": authority_copy_closure[
            "copied_file_count"
        ],
        "execution_config_sha256": sha256_file(config_path),
        "external_output_root": str(output),
        "external_output_manifest_sha256": output_closure["manifest_sha256"],
        "external_output_manifest_rows": output_closure["manifest_rows"],
        "resource_status": resource.get("status"),
        "qualified_git_status": git_state["status"],
        "next_step_authorized": False,
        "full_phase5_authorized": False,
        "final_reduced_topology_accepted": False,
        "final_runtime_thresholds_frozen": False,
        "phase6_authorized": False,
        "national_assignment_performed": False,
        "icnn_training_performed": False,
        "status": "WAITING_FOR_EXTERNAL_REVIEW",
    }
    if is_v4(execution):
        companion.update({"schema_id":"partC.phase5.v4.handoff.companion.v1", "repair_of_run_id":status["repair_of_run_id"],
            "authority_sha256":execution["authority_zip_sha256"],"reviewed_v2_handoff_sha256":execution["latest_handoff_sha256"],
            "production_recommendation":status["production_recommendation"],
            "elapsed_task_seconds":resource.get("total_campaign_elapsed_seconds_before_handoff"),
            "formal_campaign_forecast":status.get("formal_campaign_forecast")})
    write_json(building_companion, companion)
    final_source.parent.mkdir(parents=True, exist_ok=True)
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
