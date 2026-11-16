"""Explicit compact handoff packaging; no science, Git operations or discovery.

Only caller-declared, hash-bound task-owned files are copied. Large native data
remain in the explicitly supplied external-artifact index. A populated pre-seal
audit and an external final receipt form an acyclic two-stage sealing record.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
import zipfile

PHASE = "PACNHN_V1"
START = 1790304598
HARD_DEADLINE = START+52*3600
MAX_ZIP = 200_000_000
TARGET_ZIP = 100_000_000
MAX_MEMBER = 25_000_000
OWNED = ("partC_rev1_src/pacnhn_v1/", "partC_doc_icnn/pacnhn_v1/")
REQUIRED_IDS = set("C01 C02 C03 C04 N01 N02 N03 N04 N05 N06 N07 N08 N09 N10 R01 R02 R03 M01 M02 M03 M04 M05 T01 T02 T03 T04 T05 T06 T07 T08 T09 T10 A01 A02 A03 A04 A05 A06 E01 E02 E03 P01 P02 P03 P04 P05 P06 H01 H02 H03 H04".split())
ROOT_REPORTS = {"00_HANDOFF_README.md", "01_STEP_SUMMARY.md", "02_HANDOFF_STATUS.json",
    "03_DECISIONS_AND_OPEN_QUESTIONS.md", "04_VALIDATION_RESULTS.md", "05_RESOURCE_USAGE.csv",
    "06_ARTIFACT_MANIFEST.csv", "07_CODE_CHANGE_INDEX.csv", "08_COMMAND_LOG.md",
    "09_NEXT_STEP_RECOMMENDATION.md", "11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md"}
SECTIONS = {"governing_docs", "configs", "code_snapshot", "code_diff", "summary_data", "figures",
    "geospatial_summary", "log_excerpts", "external_artifact_index"}
GENERATED = {"10_UPLOAD_THIS_FILE.txt", "SEALING_PROTOCOL.md", "INCLUDE_BINDINGS.json", "MANIFEST_SHA256.csv"}
ALLOWED_SUFFIXES = {".md", ".txt", ".json", ".csv", ".tsv", ".py", ".tex", ".bib", ".sty", ".cls",
    ".png", ".pdf", ".svg", ".html", ".css", ".diff", ".patch", ".sha256", ".log", ".parquet", ".gpkg"}
EXCLUDED_PARTS = {".git", ".venv", "venv", "env", "environments", "node_modules", "site-packages", "__pycache__",
    "cache", "caches", ".cache", ".mplconfig", "toolchain", "checkpoints", "initial_witnesses", "full_paths",
    "solver_paths", "path_sets", "witness_forests", "browser_profiles", "credentials", ".ssh", ".aws", ".azure"}
EXCLUDED_NAMES = {".env", "auth.json", "credentials.json", "id_rsa", "id_ed25519", "model.json", "model.ubj",
    "common_graph.parquet", "common_od.parquet", "raw_graph.csv", "national_graph.csv", "full_paths.csv"}
CORE_ROLES = {"SUMMARY", "STATUS", "CONTRACT", "CONFIG", "CODE", "TEST", "FINAL_AUDIT", "FINAL_EVIDENCE", "BULK_INDEX", "HARDWARE_METADATA"}
BULK_FIELDS = {"native_path", "size_bytes", "mtime_ns", "format", "schema", "sha256", "role", "reason_external", "producer", "consumer", "sample_path", "sample_method", "row_count", "key_uniqueness", "summary", "missing_nonfinite"}
FINAL_STATUSES = {"PASS", "PASS_WITH_LIMITATIONS", "PARTIAL_WITH_LIMITATIONS", "FAIL", "BLOCKED", "NOT_APPLICABLE_WITH_REASON"}


class HandoffRejected(ValueError):
    """An indexed, preserved attempt is not a deliverable handoff."""
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code; self.details = details or {}


def reject(code, message, **details):
    raise HandoffRejected(code, message, details)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def digest(path, guard=lambda: None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            guard(); h.update(chunk)
    return h.hexdigest()


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        reject("UNSAFE_PATH", "Use nonempty forward-slash relative paths", value=value)
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in {"", ".", ".."} for part in path.parts):
        reject("UNSAFE_PATH", "Noncanonical or traversing relative path", value=value)
    reserved = {"CON", "PRN", "AUX", "NUL", *("COM"+str(n) for n in range(1, 10)), *("LPT"+str(n) for n in range(1, 10))}
    if any(part.rstrip(" .") != part or part.split(".")[0].upper() in reserved or any(ord(c)<32 or c in '<>"|?*' for c in part) for part in path.parts):
        reject("UNSAFE_PATH", "Unsafe Windows member name", value=value)
    return value


def bound_path(root, relative):
    relative = safe_relative(relative); root = Path(root).resolve(strict=True)
    candidate = root/relative
    # Reject every symlink/junction component, even if it resolves back inside.
    current = root
    for part in PurePosixPath(relative).parts:
        current = current/part
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            reject("LINK_SOURCE", "Symlink/junction source is not admitted", path=str(current))
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        reject("MISSING_SOURCE", "Explicit included source is missing", path=str(candidate))
    if not resolved.is_relative_to(root) or not resolved.is_file():
        reject("OUTSIDE_ALLOWED_ROOT", "Source is not a regular file inside its declared root", path=str(resolved))
    return resolved


def permitted_payload(relative, source_root, role):
    path = PurePosixPath(relative); parts = {p.lower() for p in path.parts}
    if parts & EXCLUDED_PARTS or path.name.lower() in EXCLUDED_NAMES:
        reject("EXCLUDED_PAYLOAD", "Environment/cache/checkpoint/raw-path payload is excluded", path=relative)
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        reject("EXCLUDED_PAYLOAD", "Bulk native arrays, models, executables and nested archives must remain indexed locally", path=relative)
    if path.suffix.lower() in {".parquet", ".gpkg"} and role not in {"SUMMARY_DATA", "GEOSPATIAL_SUMMARY"}:
        reject("EXCLUDED_PAYLOAD", "Binary table/geometry payload requires an explicit compact-summary role", path=relative)
    if source_root == "REPO" and not relative.startswith(OWNED):
        reject("UNOWNED_REPOSITORY_PATH", "Only new task-owned code/docs may be copied from the repository", path=relative)


def prefix_admitted(relative, prefixes):
    return any(relative == p or relative.startswith(p.rstrip("/")+"/") for p in prefixes)


def request_index(plan, roots):
    result = []
    for row in plan.get("includes", []):
        source = roots.get(row.get("source_root"))
        relative = row.get("source_relative_path", "")
        actual_size = None
        if source and isinstance(relative, str):
            try:
                p = bound_path(source, relative)
                actual_size = p.stat().st_size
            except (OSError, HandoffRejected):
                pass
        result.append({"source_root": row.get("source_root"), "source_relative_path": relative,
            "archive_path": row.get("archive_path"), "expected_sha256": row.get("sha256"),
            "expected_size_bytes": row.get("size_bytes"), "observed_size_bytes": actual_size,
            "role": row.get("role"), "disposition": "REQUESTED_EXPLICITLY_NOT_SILENTLY_OMITTED"})
    return result


def _bound_clock_json(binding, label):
    if not isinstance(binding, dict) or not binding.get("path") or not re.fullmatch(r"[a-fA-F0-9]{64}", str(binding.get("sha256", ""))):
        reject("CONTINUATION_BINDING", label+" requires an explicit path and SHA-256")
    path = Path(binding["path"]).resolve(strict=True)
    actual = digest(path)
    if actual != binding["sha256"].lower():
        reject("CONTINUATION_BINDING", label+" bytes changed")
    return read_json(path), {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def _finite_epoch(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        reject("CONTINUATION_CLOCK", label+" must be a finite epoch")
    return float(value)


def publication_window(config, *, campaign_config_sha256, stage4_clock, clock_amendment, deadline_epoch, now=None):
    """Authenticate a separate continuation and its actual four-hour final window.

    Stdlib only so both the publication reader and callerless packager use the
    same admission. Original campaign and checklist records are never changed.
    """
    from .continuation import validate_continuation
    observed = time.time() if now is None else now
    authority = validate_continuation(config, action="publication", now=observed)
    if authority is None:
        reject("CONTINUATION_BINDING", "A publication extension requires recorded user authority")
    clock, clock_binding = _bound_clock_json(stage4_clock, "Stage4 clock")
    canonical_clock = Path(config["output_root"]).resolve(strict=True)/"STAGE4_CLOCK.json"
    supplied_clock = Path(stage4_clock["path"])
    if (supplied_clock.absolute() != canonical_clock or supplied_clock.is_symlink()
            or (hasattr(supplied_clock, "is_junction") and supplied_clock.is_junction())
            or Path(clock_binding["path"]) != canonical_clock):
        reject("CONTINUATION_CLOCK", "Reuse canonical OUT/STAGE4_CLOCK.json; alternate Stage4 clocks cannot reset the window")
    supplement, supplement_binding = _bound_clock_json(clock_amendment, "C01 amendment")
    identity = {"phase_id": config["phase_id"], "run_id": config["run_id"],
        "continuation_approval_sha256": authority["approval_sha256"],
        "continuation_clock_sha256": authority["clock_sha256"],
        "campaign_config_sha256": campaign_config_sha256,
        "original_contract_sha256": config["contract_sha256"]}
    if any(clock.get(k) != v or supplement.get(k) != v for k, v in identity.items()):
        reject("CONTINUATION_BINDING", "Stage4/C01 records must bind the exact continuation")
    if clock.get("schema") != "PACNHN_STAGE4_CLOCK_V1":
        reject("CONTINUATION_CLOCK", "Explicit versioned Stage4 clock required")
    start = _finite_epoch(clock.get("stage_start_epoch"), "Stage4 start")
    end = _finite_epoch(clock.get("stage_deadline_epoch"), "Stage4 deadline")
    requested = _finite_epoch(deadline_epoch, "Requested final deadline")
    if not authority["start_epoch"] <= start <= observed or end != min(start+14400, authority["hard_deadline_epoch"]):
        reject("CONTINUATION_CLOCK", "Stage4 must retain its actual start and at-most-four-hour ceiling")
    if not start < requested <= end or observed >= requested:
        reject("CONTINUATION_CLOCK", "Requested publication/packaging deadline is expired or exceeds Stage4")
    if (supplement.get("schema") != "PACNHN_C01_CLOCK_AMENDMENT_V1"
            or supplement.get("requirement_id") != "C01"
            or supplement.get("original_start_epoch") != START
            or supplement.get("original_hard_deadline_epoch") != HARD_DEADLINE
            or supplement.get("original_52h_requirement_met") is not False
            or not isinstance(supplement.get("disclosure"), str) or not supplement["disclosure"].strip()):
        reject("C01_DISCLOSURE", "C01 must disclose original52h incompletion and retain the old clock")
    disclosure = ("The original 52-hour campaign deadline was not met for the complete assignment/publication deliverable. "
        "Its original start and deadline remain unchanged in the historical record. A separately recorded, user-authorized "
        "24-hour continuation covers the remaining work, with at most12hours for assignments and at most4hours from the "
        "actual publication-stage start for publication, QA and handoff. The intervening interruption is not erased or "
        "reported as computation, and the original C01 clock requirement is not an unqualified PASS.")
    return {"authority": authority, "stage4_clock": clock_binding, "clock_amendment": supplement_binding,
        "stage_start_epoch": start, "stage_deadline_epoch": end, "deadline_epoch": requested,
        "original_hard_deadline_epoch": HARD_DEADLINE, "original_52h_requirement_met": False,
        "disclosure": disclosure}


def _handoff_window(plan, roots, *, now=None):
    """Callerless core authenticates the actual campaign config, not a flag."""
    path_value = plan.get("campaign_config_path")
    if not path_value:
        if plan.get("deadline_epoch", HARD_DEADLINE) > HARD_DEADLINE:
            reject("CLOCK_CHANGED", "Extended handoff requires an exact campaign_config_path")
        return None
    config_path = Path(path_value).resolve(strict=True)
    if not config_path.is_relative_to(roots["REPO"]) and not config_path.is_relative_to(roots["OUT"]):
        reject("CAMPAIGN_CONFIG_IDENTITY", "Handoff config must be within the declared repository/output roots")
    if digest(config_path) != plan.get("campaign_config_sha256"):
        reject("CAMPAIGN_CONFIG_IDENTITY", "Handoff continuation config SHA differs")
    config = read_json(config_path)
    if config.get("run_id") != plan.get("run_id") or config.get("original_start_epoch") != START:
        reject("CAMPAIGN_CONFIG_IDENTITY", "Handoff continuation config/run differs")
    from .continuation import validate_continuation
    authority = validate_continuation(config, action="handoff", now=now)
    if authority is None:
        if plan.get("deadline_epoch", HARD_DEADLINE) > HARD_DEADLINE:
            reject("CLOCK_CHANGED", "No authenticated continuation permits an extended handoff")
        return None
    member = plan.get("clock_amendment_member")
    match = [row for row in plan.get("includes", []) if row.get("archive_path") == member]
    if not member or len(match) != 1 or match[0].get("source_root") not in roots:
        reject("C01_DISCLOSURE", "An explicitly included C01 amendment member is required")
    row = match[0]
    amendment_path = bound_path(roots[row["source_root"]], row["source_relative_path"])
    window = publication_window(config, campaign_config_sha256=plan["campaign_config_sha256"],
        stage4_clock=plan.get("stage4_clock"),
        clock_amendment={"path": str(amendment_path), "sha256": row.get("sha256")},
        deadline_epoch=plan.get("deadline_epoch"), now=now)
    window["campaign_config_path"] = str(config_path)
    window["clock_amendment_member"] = member
    continuation = config["continuation"]
    window["required_authority_bindings"] = [
        {"path": str(config_path), "sha256": plan["campaign_config_sha256"]},
        {"path": config["contract_path"], "sha256": config["contract_sha256"]},
        *[{"path": continuation[key+"_path"], "sha256": continuation[key+"_sha256"]}
          for key in ("approval", "clock", "original_config", "training_result")],
        window["stage4_clock"], window["clock_amendment"]]
    return window


def validate_plan(plan, roots, guard, *, window=None):
    needed = {"schema", "phase_id", "step_id", "run_id", "handoff_id", "campaign_config_sha256", "original_start_epoch", "deadline_epoch",
        "tasks", "expected_outputs", "acceptance_tests", "resource_budget", "approved_output_prefixes",
        "includes", "final_audit_member", "bulk_index_member", "status_member", "publication_disposition", "standalone_assets"}
    if needed-set(plan) or plan.get("schema") != "pacnhn.handoff.manifest.v1" or plan.get("phase_id") != PHASE or plan.get("step_id") != "handoff":
        reject("INCOMPLETE_EXECUTION_CONTRACT", "Explicit PACNHN handoff manifest is incomplete")
    ceiling = HARD_DEADLINE if window is None else window["stage_deadline_epoch"]
    if plan["original_start_epoch"] != START or not START < plan["deadline_epoch"] <= ceiling:
        reject("CLOCK_CHANGED", "Original campaign clock cannot be reset or extended")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", plan["handoff_id"]):
        reject("UNSAFE_HANDOFF_ID", "handoff_id must be a simple versioned name")
    for name in ["run_id", "tasks", "expected_outputs", "acceptance_tests", "resource_budget", "approved_output_prefixes", "includes"]:
        if not plan[name]:
            reject("EMPTY_EXECUTION_FIELD", "No implicit or empty execution field", field=name)
    if not {"VALIDATE_INPUTS", "BUILD_COMPACT_ZIP", "VERIFY_ARCHIVE"}.issubset(plan["tasks"]):
        reject("UNDECLARED_TASK", "All three packaging tasks must be explicit")
    limits = {"target_zip_bytes": TARGET_ZIP, "hard_zip_bytes": MAX_ZIP, "max_member_bytes": MAX_MEMBER}
    for key, ceiling in list(limits.items()):
        value = plan["resource_budget"].get(key, ceiling)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= ceiling:
            reject("SIZE_POLICY", "Requested size limit exceeds current policy or is invalid", field=key)
        limits[key] = value
    if limits["target_zip_bytes"] > limits["hard_zip_bytes"]:
        reject("SIZE_POLICY", "Target ZIP limit cannot exceed hard limit")
    prefixes = [safe_relative(p) for p in plan["approved_output_prefixes"]]
    compression = plan.get("compression", "DEFLATE")
    if compression not in {"DEFLATE", "LZMA"}:
        reject("COMPRESSION_POLICY", "Supported explicit ZIP methods are DEFLATE and LZMA")
    sources = []; seen = set(); roles = set()
    for row in plan["includes"]:
        guard()
        fields = {"source_root", "source_relative_path", "archive_path", "sha256", "size_bytes", "role"}
        if fields-set(row) or row["source_root"] not in roots:
            reject("INVALID_INCLUDE", "Every included file needs an explicit root/path/member/hash/size/role")
        relative, member = safe_relative(row["source_relative_path"]), safe_relative(row["archive_path"])
        role = row["role"]
        if not isinstance(role, str) or not role:
            reject("INVALID_INCLUDE", "File role must be nonempty")
        permitted_payload(relative, row["source_root"], role)
        permitted_payload(member, "OUT", role)
        if row["source_root"] == "OUT" and not prefix_admitted(relative, prefixes):
            reject("UNAPPROVED_OUTPUT", "OUT evidence is outside the explicit allowlist", path=relative)
        if row["source_root"] == "REPO":
            if relative.startswith(OWNED[0]) and member != "code_snapshot/"+relative:
                reject("SOURCE_LAYOUT", "Code/test snapshots must preserve exact repository-relative paths", path=relative)
            if relative.startswith(OWNED[1]) and not any(member == p+relative for p in ["governing_docs/", "configs/", "documents/"]):
                reject("SOURCE_LAYOUT", "Document/config snapshots must preserve exact repository-relative paths", path=relative)
        folded = member.casefold()
        if folded in seen or folded in {x.casefold() for x in GENERATED}:
            reject("DUPLICATE_MEMBER", "Duplicate, case-colliding or generated-reserved ZIP member", member=member)
        seen.add(folded); roles.add(role)
        if not re.fullmatch(r"[a-fA-F0-9]{64}", str(row["sha256"])) or isinstance(row["size_bytes"], bool) or not isinstance(row["size_bytes"], int) or row["size_bytes"] < 0:
            reject("INVALID_IDENTITY", "Every included source needs a valid exact size/SHA-256", member=member)
        path = bound_path(roots[row["source_root"]], relative)
        size = path.stat().st_size
        if size > limits["max_member_bytes"]:
            reject("MEMBER_TOO_LARGE", "Explicit file exceeds the per-member cap; preserve it locally and revise the declared artifact index/storage layout, never silently omit", member=member, size_bytes=size, cap_bytes=limits["max_member_bytes"])
        if size != row["size_bytes"] or digest(path, guard).lower() != row["sha256"].lower():
            reject("SOURCE_IDENTITY_CHANGED", "Explicit source size/hash differs", member=member)
        sources.append({**row, "sha256": row["sha256"].lower(), "path": path})
    members = {r["archive_path"] for r in sources}
    if ROOT_REPORTS-members or CORE_ROLES-roles:
        reject("MISSING_REQUIRED_INVENTORY", "Report-first inventory or required role missing", missing_reports=sorted(ROOT_REPORTS-members), missing_roles=sorted(CORE_ROLES-roles))
    missing_sections = {s for s in SECTIONS if not any(m.startswith(s+"/") for m in members)}
    if missing_sections:
        reject("MISSING_REQUIRED_INVENTORY", "Every required section needs evidence or an explicit not-applicable marker", missing_sections=sorted(missing_sections))
    by_member = {r["archive_path"]: r for r in sources}
    if not re.fullmatch(r"[a-fA-F0-9]{64}", str(plan["campaign_config_sha256"])) or not any(row["role"] == "CONFIG" and row["sha256"] == plan["campaign_config_sha256"].lower() for row in sources):
        reject("CAMPAIGN_CONFIG_IDENTITY", "The exact hash-bound live campaign configuration must itself be an included CONFIG member")
    for key, role in [("final_audit_member", "FINAL_AUDIT"), ("bulk_index_member", "BULK_INDEX"), ("status_member", "STATUS")]:
        if plan[key] not in by_member or by_member[plan[key]]["role"] != role:
            reject("MISSING_BOUND_GATE", "Required gate/index must be an explicitly included member with the exact role", field=key)
    if window is not None:
        for required in window["required_authority_bindings"]:
            target = Path(required["path"]).resolve(strict=True)
            if not any(row["path"] == target and row["sha256"] == required["sha256"] for row in sources):
                reject("MISSING_CONTINUATION_EVIDENCE", "Continuation authority/config/clock/supplement must be included exactly", path=str(target))
    if plan["status_member"] != "02_HANDOFF_STATUS.json":
        reject("STATUS_LAYOUT", "Canonical top-level handoff status is required")
    return sources, by_member, limits, compression


def read_bulk_index(row):
    with row["path"].open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream); fields = set(reader.fieldnames or [])
        required = BULK_FIELDS
        if required-fields:
            reject("BULK_INDEX_SCHEMA", "Explicit bulk index lacks required schema/identity/summary fields", missing=sorted(required-fields))
        rows = list(reader)
    indexed = {}
    for item in rows:
        key = item["native_path"]
        if not key or key in indexed or not re.fullmatch(r"[a-fA-F0-9]{64}", item["sha256"]):
            reject("BULK_INDEX_IDENTITY", "Bulk index has empty/duplicate paths or invalid hash")
        try:
            size = int(item["size_bytes"]); mtime = int(item["mtime_ns"])
        except ValueError:
            reject("BULK_INDEX_IDENTITY", "Bulk size/mtime must be exact integers")
        if size < 0 or mtime < 0 or not all(item[k].strip() for k in ["format", "schema", "role", "reason_external", "producer", "consumer", "sample_method", "row_count", "key_uniqueness", "summary", "missing_nonfinite"]):
            reject("BULK_INDEX_IDENTITY", "Bulk artifact identity/schema/exclusion rationale is incomplete")
        indexed[key] = {"sha256": item["sha256"].lower(), "size_bytes": size}
    # This packager authenticates the supplied index, not huge external arrays.
    # Their actual hash verification is a separately bound final-source audit.
    return indexed


def validate_status(status, run_id):
    if status.get("status") != "WAITING_FOR_EXTERNAL_REVIEW" or status.get("phase_id") != PHASE or status.get("run_id") != run_id:
        reject("UNSAFE_FINAL_STATUS", "Final handoff state must await external review for this exact run")
    if status.get("next_campaign_authorized") is not False or status.get("next_step_authorized") is not False:
        reject("UNSAFE_FINAL_STATUS", "Both next-step and next-campaign authorization must be explicitly false")
    if status.get("external_acceptance") not in {None, False, "NOT_RECORDED", "PENDING"}:
        reject("UNSAFE_FINAL_STATUS", "A new handoff cannot assert external acceptance")


def validate_audit(audit, by_member, external, roots, run_id, *, window=None):
    if audit.get("phase_id") != PHASE or audit.get("run_id") != run_id:
        reject("FINAL_AUDIT_IDENTITY", "Final completion audit is for another campaign")
    requirements = audit.get("requirements", [])
    ids = [r.get("id") for r in requirements]
    if len(ids) != len(set(ids)) or set(ids) != REQUIRED_IDS:
        reject("FINAL_AUDIT_COVERAGE", "All51 original requirement IDs need a populated disposition")
    known = {}
    for name, row in by_member.items():
        value = {"sha256": row["sha256"], "size_bytes": row["size_bytes"]}
        known[name] = value; known[str(row["path"])] = value
        known[row["source_root"]+"/"+row["source_relative_path"]] = value
        if row["source_root"] == "REPO":
            relative = row["source_relative_path"]
            if relative.startswith(OWNED[0]): known["SRC/"+relative[len(OWNED[0]):]] = value
            if relative.startswith(OWNED[1]): known["DOC/"+relative[len(OWNED[1]):]] = value
    for key, value in external.items():
        if key in known and known[key] != value:
            reject("CONFLICTING_EVIDENCE_IDENTITY", "Included and external identities disagree", path=key)
        known[key] = value
    pending = []; counts = {}; scientific_counts = {}
    audit_ceiling = HARD_DEADLINE if window is None else window["stage_deadline_epoch"]
    for requirement in requirements:
        item = requirement.get("final_audit", {}); state = item.get("status")
        if state not in FINAL_STATUSES:
            reject("FINAL_AUDIT_UNPOPULATED", "Every requirement needs a final measured/limited/failure disposition", requirement=requirement["id"])
        epoch = item.get("audited_epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, (int, float)) or not START <= epoch <= audit_ceiling or not item.get("reviewer") or not item.get("disposition_reason"):
            reject("FINAL_AUDIT_UNPOPULATED", "Requirement audit lacks actual epoch/reviewer/rationale", requirement=requirement["id"])
        bindings = item.get("evidence_bindings", [])
        if not bindings:
            reject("FINAL_AUDIT_UNPOPULATED", "Intent/tests alone cannot close a requirement without bound evidence", requirement=requirement["id"])
        for binding in bindings:
            target = known.get(binding.get("path"))
            if target is None or target["sha256"] != str(binding.get("sha256", "")).lower() or target["size_bytes"] != binding.get("bytes"):
                reject("FINAL_AUDIT_UNBOUND_EVIDENCE", "Audit evidence is not an included exact file or explicitly indexed native artifact", requirement=requirement["id"], evidence=binding.get("path"))
        if window is not None and requirement["id"] == "C01":
            if state not in {"PASS_WITH_LIMITATIONS", "PARTIAL_WITH_LIMITATIONS", "FAIL", "BLOCKED"}:
                reject("C01_DISCLOSURE", "Original52h C01 cannot be an unqualified PASS or not-applicable")
            amendment = by_member[window["clock_amendment_member"]]
            if not any(binding.get("sha256") == amendment["sha256"]
                       and binding.get("bytes") == amendment["size_bytes"] for binding in bindings):
                reject("C01_DISCLOSURE", "C01 must cite the included exact clock-amendment supplement")
        is_pending = str(item["disposition_reason"]).startswith("PENDING_SEAL_ONLY:")
        if is_pending:
            if requirement["id"] not in {"H03", "H04"} or state != "PARTIAL_WITH_LIMITATIONS":
                reject("INVALID_PENDING_SEAL", "Only archive verification/stop rows may await this acyclic seal")
            pending.append(requirement["id"])
        else:
            scientific_counts[state] = scientific_counts.get(state, 0)+1
        counts[state] = counts.get(state, 0)+1
    summary = audit.get("final_audit_summary", {})
    if summary.get("unaudited_rows") != 0 or summary.get("required_rows") != len(REQUIRED_IDS) or summary.get("next_campaign_authorized") is not False:
        reject("FINAL_AUDIT_SUMMARY", "Final audit summary is not populated or disagrees with requirement coverage")
    expected_counts = {"passed_rows": counts.get("PASS", 0), "qualified_rows": sum(counts.get(s, 0) for s in ["PASS_WITH_LIMITATIONS", "PARTIAL_WITH_LIMITATIONS", "NOT_APPLICABLE_WITH_REASON"]), "failed_or_blocked_rows": counts.get("FAIL", 0)+counts.get("BLOCKED", 0)}
    if any(summary.get(key) != value for key, value in expected_counts.items()):
        reject("FINAL_AUDIT_SUMMARY", "Summary disposition counts disagree with populated rows", expected_counts=expected_counts)
    return {"requirement_count": len(requirements), "preseal_disposition_counts": counts, "scientific_disposition_counts": scientific_counts, "pending_seal_only_ids": pending,
            "science_acceptance_inferred": False}


def validate_assets(plan, by_member):
    disposition = plan["publication_disposition"]
    if disposition == "UNAVAILABLE_WITH_EXPLICIT_EVIDENCE":
        if plan["standalone_assets"] or not any(r["role"] == "PUBLICATION_DISPOSITION" for r in by_member.values()):
            reject("PUBLICATION_DISPOSITION", "Unavailable publication needs an explicit included disposition and no claimed asset inventory")
        return []
    if disposition != "AVAILABLE":
        reject("PUBLICATION_DISPOSITION", "Publication availability must be explicit")
    expected_ids = {"figure"+str(i) for i in range(3, 9)}|{"table"+str(i) for i in range(1, 7)}
    assets = plan["standalone_assets"]
    if len(assets) != 12 or {a.get("asset_id") for a in assets} != expected_ids:
        reject("ASSET_INVENTORY", "All12 standalone folders must be explicitly declared")
    if len({a.get("archive_prefix", "").casefold() for a in assets}) != 12:
        reject("ASSET_INVENTORY", "Each standalone asset needs its own distinct folder")
    validated = []
    for asset in assets:
        prefix = safe_relative(asset["archive_prefix"]).rstrip("/")
        script = "build_table.py" if asset["asset_id"].startswith("table") else "plot.py"
        common = [script, "README.md", "data_dictionary.csv", "provenance.json", "asset.csv", "csv_parts.csv", "read_csv_hashes.csv", "RENDER_RESULT.json"]
        common += ["table.csv", "columns.csv", "table.tex"] if script == "build_table.py" else ["panels.csv", "panel_status.csv"]
        missing = [name for name in common if prefix+"/"+name not in by_member]
        if missing:
            reject("ASSET_INVENTORY", "Standalone folder is missing essential members", asset=asset["asset_id"], missing=missing)
        with by_member[prefix+"/asset.csv"]["path"].open(encoding="utf-8-sig", newline="") as stream:
            identity = list(csv.DictReader(stream))
        if len(identity) != 1 or identity[0].get("asset_id") != asset["asset_id"]:
            reject("ASSET_INVENTORY", "Standalone asset identity disagrees with declared folder", asset=asset["asset_id"])
        with by_member[prefix+"/read_csv_hashes.csv"]["path"].open(encoding="utf-8-sig", newline="") as stream:
            read_rows = list(csv.DictReader(stream))
        if not read_rows or len({r.get("file") for r in read_rows}) != len(read_rows):
            reject("ASSET_READ_AUDIT", "Standalone scientific read audit is empty/duplicated", asset=asset["asset_id"])
        for read in read_rows:
            name = safe_relative(read["file"])
            if "/" in name or not name.endswith(".csv"):
                reject("ASSET_READ_AUDIT", "Scientific inputs must remain local sibling CSVs")
            source = by_member.get(prefix+"/"+name)
            if source is None or source["sha256"] != read["sha256"].lower():
                reject("OMITTED_SCIENTIFIC_CSV", "A read scientific CSV was omitted or changed", asset=asset["asset_id"], file=name)
        result = read_json(by_member[prefix+"/RENDER_RESULT.json"]["path"])
        if result.get("status") not in {"RENDERED", "UNAVAILABLE_NO_RENDERABLE_OBSERVATIONS"}:
            reject("ASSET_RENDER_STATUS", "Missing or unrecognized actual asset disposition", asset=asset["asset_id"])
        outputs = result.get("outputs", [])
        if result["status"] == "RENDERED" and (not any(n.endswith(".png") for n in outputs) or not any(n.endswith(".pdf") for n in outputs)):
            reject("ASSET_RENDER_STATUS", "Rendered asset needs actual PNG and PDF outputs")
        for name in outputs:
            if "/" in safe_relative(name) or prefix+"/"+name not in by_member:
                reject("ASSET_INVENTORY", "Declared actual render output is absent", asset=asset["asset_id"], file=name)
        validated.append({"asset_id": asset["asset_id"], "status": result["status"], "read_csv_files": len(read_rows), "outputs": len(outputs)})
    return validated


def copy_bound(row, destination, guard):
    destination.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(); total = 0
    with row["path"].open("rb") as src, destination.open("xb") as dst:
        before = os.fstat(src.fileno())
        for chunk in iter(lambda: src.read(1024*1024), b""):
            guard(); dst.write(chunk); h.update(chunk); total += len(chunk)
        after = os.fstat(src.fileno())
    if total != row["size_bytes"] or h.hexdigest() != row["sha256"] or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        reject("SOURCE_CHANGED_DURING_COPY", "Bound source changed while copied", member=row["archive_path"])


def verify_archive(archive, manifest_rows, manifest_digest, guard=lambda: None):
    expected = {row["relative_path"] for row in manifest_rows}|{"MANIFEST_SHA256.csv"}
    with zipfile.ZipFile(archive, "r") as zipped:
        infos = zipped.infolist(); names = [info.filename for info in infos]
        if len(names) != len(set(n.casefold() for n in names)) or set(names) != expected:
            reject("ARCHIVE_MEMBER_SET", "Archive has missing/extra/duplicate or case-colliding members")
        for info in infos:
            safe_relative(info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
                reject("ARCHIVE_MEMBER_TYPE", "Directory/symlink/encrypted archive entry is not admitted")
        guard()
        bad = zipped.testzip()
        if bad is not None:
            reject("ARCHIVE_CRC", "ZIP CRC test failed", member=bad)
        expected_by_name = {r["relative_path"]: r for r in manifest_rows}
        for info in infos:
            guard(); h = hashlib.sha256(); count = 0
            with zipped.open(info.filename) as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    guard(); h.update(chunk); count += len(chunk)
            if info.filename == "MANIFEST_SHA256.csv":
                expected_hash = manifest_digest
            else:
                row = expected_by_name[info.filename]; expected_hash = row["sha256"]
                if count != row["size_bytes"]:
                    reject("ARCHIVE_MEMBER_SIZE", "Archive uncompressed member size differs", member=info.filename)
            if h.hexdigest() != expected_hash:
                reject("ARCHIVE_MEMBER_SHA", "Archive reopened member hash differs", member=info.filename)
    return {"testzip": "PASS", "member_sha256": "PASS", "exact_member_set": "PASS", "member_count": len(expected)}


def create_handoff(manifest_path, *, repo_root, output_root, guard, now=time.time):
    """Bounded packager API; CLI supplies the live campaign Context guard.

    Unit tests supply isolated synthetic roots and a synthetic clock. This does
    not alter the live campaign clock or provide a CLI clock bypass.
    """
    plan = read_json(manifest_path); roots = {"REPO": Path(repo_root).resolve(strict=True), "OUT": Path(output_root).resolve(strict=True)}
    handoff_id = plan.get("handoff_id", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", str(handoff_id)):
        reject("UNSAFE_HANDOFF_ID", "A safe unique handoff_id is required before creating an attempt")
    attempt = roots["OUT"]/"handoff_attempts"/handoff_id
    if attempt.exists():
        reject("EXISTING_ATTEMPT", "Versioned attempts are immutable; use a new handoff_id")
    parent = roots["OUT"]/"handoff_attempts"
    if parent.exists() and (parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction())):
        reject("UNSAFE_OUTPUT", "Handoff parent cannot be a symlink/junction")
    parent.mkdir(exist_ok=True); attempt.mkdir(exist_ok=False)
    started = now(); plan_digest = digest(manifest_path)
    write_new_json(attempt/"REQUESTED_ARTIFACT_INDEX.json", request_index(plan, roots))
    window = None
    def bounded():
        guard()
        ceiling = HARD_DEADLINE if window is None else window["stage_deadline_epoch"]
        deadline = min(plan.get("deadline_epoch", ceiling), ceiling)
        if now() >= deadline:
            reject("DEADLINE", "Original campaign/packaging deadline reached")
    try:
        window = _handoff_window(plan, roots, now=now())
        bounded()
        sources, by_member, limits, compression = validate_plan(plan, roots, bounded, window=window)
        status = read_json(by_member[plan["status_member"]]["path"]); validate_status(status, plan["run_id"])
        external = read_bulk_index(by_member[plan["bulk_index_member"]])
        audit_row = by_member[plan["final_audit_member"]]
        audit_source = read_json(audit_row["path"])
        audit = validate_audit(audit_source, by_member, external, roots, plan["run_id"], window=window)
        assets = validate_assets(plan, by_member)
        if plan["publication_disposition"] == "UNAVAILABLE_WITH_EXPLICIT_EVIDENCE" and any(r["id"] in {"P02", "P03", "P04", "P05"} and r["final_audit"]["status"] == "PASS" for r in audit_source["requirements"]):
            reject("PUBLICATION_AUDIT_CONTRADICTION", "Unavailable publication cannot have unqualified PASS asset/PDF audit rows")
        stage = attempt/"payload"; stage.mkdir()
        for row in sorted(sources, key=lambda r: r["archive_path"]):
            bounded(); copy_bound(row, stage/row["archive_path"], bounded)
        binding_rows = [{k: row[k] for k in ["source_root", "source_relative_path", "archive_path", "sha256", "size_bytes", "role"]} for row in sources]
        write_new_json(stage/"INCLUDE_BINDINGS.json", {"manifest_sha256": plan_digest, "files": binding_rows,
            "phase_id": PHASE, "run_id": plan["run_id"], "campaign_config_sha256": plan["campaign_config_sha256"],
            "original_start_epoch": START, "deadline_epoch": plan["deadline_epoch"],
            "continuation_window": window,
            "declared_tasks": plan["tasks"], "acceptance_tests": plan["acceptance_tests"], "resource_budget": plan["resource_budget"],
            "external_artifact_index_member": plan["bulk_index_member"], "no_implicit_discovery_or_omission": True})
        archive_name = handoff_id+"_SYNC.zip"
        (stage/"10_UPLOAD_THIS_FILE.txt").write_text("Upload "+archive_name+" after verifying the adjacent FINAL_SEAL_RECEIPT.json and .sha256 companion.\nStatus: WAITING_FOR_EXTERNAL_REVIEW. No next campaign is authorized.\n", encoding="utf-8")
        method_note = "Standard ZIP DEFLATE compression." if compression == "DEFLATE" else "ZIP LZMA method14: use Python zipfile or a compatible extractor such as7-Zip; Windows Explorer compatibility is not asserted."
        (stage/"SEALING_PROTOCOL.md").write_text("# Two-stage sealing record\n\nThis package embeds a populated pre-seal completion audit. Only explicitly marked H03/H04 archive-verification rows may await the seal. Scientific limitations are recorded separately; no pending seal is presented as a scientific time-limit result.\n\nThe adjacent FINAL_SEAL_RECEIPT.json binds the actual final ZIP SHA-256/byte count, exact reopened member inventory, CRC/SHA verification, embedded audit hash and any seal-only rows closed by that verification. The embedded audit is not rewritten after ZIP construction. This avoids circular self-hashes. MANIFEST_SHA256.csv binds every other ZIP member; its own hash is bound by the external final receipt and whole-ZIP hash.\n\nThe receipt verifies packaging, not scientific correctness or external acceptance. Final review state is WAITING_FOR_EXTERNAL_REVIEW; next_step_authorized=false and next_campaign_authorized=false.\n\n"+method_note+"\n", encoding="utf-8")
        rows = []
        for path in sorted(stage.rglob("*")):
            if not path.is_file(): continue
            bounded(); size = path.stat().st_size
            if size > limits["max_member_bytes"]:
                reject("GENERATED_MEMBER_TOO_LARGE", "Generated manifest/evidence exceeds the per-member cap", member=path.relative_to(stage).as_posix(), size_bytes=size)
            rows.append({"relative_path": path.relative_to(stage).as_posix(), "size_bytes": size, "sha256": digest(path, bounded)})
        manifest_file = stage/"MANIFEST_SHA256.csv"
        with manifest_file.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["relative_path", "size_bytes", "sha256"]); writer.writeheader(); writer.writerows(rows)
        if manifest_file.stat().st_size > limits["max_member_bytes"]:
            reject("GENERATED_MEMBER_TOO_LARGE", "Per-member manifest itself exceeds cap")
        manifest_sha = digest(manifest_file, bounded)
        candidate = attempt/(handoff_id+".candidate.zip")
        method = zipfile.ZIP_DEFLATED if compression == "DEFLATE" else zipfile.ZIP_LZMA
        options = {"compresslevel": 9} if compression == "DEFLATE" else {}
        with zipfile.ZipFile(candidate, "x", compression=method, allowZip64=True, **options) as zipped:
            for row in rows:
                bounded(); zipped.write(stage/row["relative_path"], row["relative_path"])
            zipped.write(manifest_file, "MANIFEST_SHA256.csv")
        if candidate.stat().st_size > limits["hard_zip_bytes"]:
            reject("ZIP_TOO_LARGE", "Complete candidate exceeds the hard cap; candidate and requested artifact index are preserved. Revise compression/storage or explicit numbered-bundle design; no member was silently dropped", candidate=str(candidate), size_bytes=candidate.stat().st_size, cap_bytes=limits["hard_zip_bytes"])
        verification = verify_archive(candidate, rows, manifest_sha, bounded)
        for row in sources:
            if row["path"].stat().st_size != row["size_bytes"] or digest(row["path"], bounded) != row["sha256"]:
                reject("SOURCE_CHANGED_BEFORE_SEAL", "Included source changed after snapshot; final seal rejected", member=row["archive_path"])
        if digest(manifest_path, bounded) != plan_digest:
            reject("MANIFEST_CHANGED_BEFORE_SEAL", "Explicit include manifest changed during packaging")
        if window is not None and _handoff_window(plan, roots, now=now()) != window:
            reject("CONTINUATION_CHANGED_BEFORE_SEAL", "Continuation authority/window changed during packaging")
        final = attempt/archive_name
        if final.exists() or final.parent.resolve() != attempt.resolve():
            reject("UNSAFE_OUTPUT", "Final ZIP must be new and inside this exact attempt")
        final_sha = digest(candidate, bounded)
        final_size = candidate.stat().st_size
        bounded()
        receipt = {"schema": "pacnhn.final_seal.v1", "status": "WAITING_FOR_EXTERNAL_REVIEW", "phase_id": PHASE,
            "run_id": plan["run_id"], "handoff_id": handoff_id, "archive_path": str(final), "archive_sha256": final_sha,
            "archive_bytes": final_size, "target_zip_bytes": limits["target_zip_bytes"],
            "target_size_met": final_size <= limits["target_zip_bytes"], "hard_limit_pass": True,
            "max_member_bytes": limits["max_member_bytes"], "compression": compression, "verification": verification,
            "member_manifest_sha256": manifest_sha, "member_manifest_self_hash_policy": "External receipt and whole-archive hash bind the manifest itself",
            "embedded_final_audit_member": plan["final_audit_member"], "embedded_final_audit_sha256": audit_row["sha256"],
            "populated_requirement_count": audit["requirement_count"], "preseal_disposition_counts": audit["preseal_disposition_counts"], "scientific_disposition_counts": audit["scientific_disposition_counts"],
            "closed_pending_seal_only_ids": audit["pending_seal_only_ids"], "assets": assets,
            "external_bulk_artifact_count": len(external), "external_bulk_contents_rehashed_by_packager": False,
            "source_snapshot_unchanged": True, "include_manifest_sha256": plan_digest,
            "original_start_epoch": START, "packaging_started_epoch": started, "completed_epoch": now(),
            "continuation_window": window,
            "next_step_authorized": False, "next_campaign_authorized": False, "external_acceptance": "NOT_RECORDED",
            "scientific_acceptance_inferred": False, "supersedes_handoff_id": plan.get("supersedes_handoff_id")}
        candidate.rename(final)
        with final.with_suffix(".zip.sha256").open("x", encoding="ascii") as stream:
            stream.write(final_sha+"  "+final.name+"\n")
        # This receipt is the final marker. A ZIP without it is never deliverable.
        write_new_json(attempt/"FINAL_SEAL_RECEIPT.json", receipt)
        return receipt
    except BaseException as exc:
        failure = {"status": "REJECTED_NOT_A_FINAL_HANDOFF", "handoff_id": handoff_id,
            "code": getattr(exc, "code", type(exc).__name__), "message": str(exc), "details": getattr(exc, "details", {}),
            "include_manifest_sha256": plan_digest, "recorded_epoch": now(), "no_member_silently_omitted": True,
            "requested_artifact_index": "REQUESTED_ARTIFACT_INDEX.json", "attempt_preserved": True,
            "next_campaign_authorized": False}
        if not (attempt/"REJECTION.json").exists(): write_new_json(attempt/"REJECTION.json", failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Exact live PACNHN campaign configuration")
    parser.add_argument("--manifest", required=True, help="Explicit compact-file inclusion manifest")
    args = parser.parse_args()
    # Imported only by the real CLI; the packaging core/tests use stdlib only.
    from .control import Context, sha
    ctx = Context(args.config, step_id="handoff")
    plan = read_json(args.manifest)
    if plan.get("run_id") != ctx.c["run_id"] or plan.get("campaign_config_sha256") != sha(args.config):
        reject("CAMPAIGN_CONFIG_IDENTITY", "Manifest must bind the exact current campaign config/run")
    if ctx.c["original_start_epoch"] != START or ctx.c["hard_deadline_epoch"] != HARD_DEADLINE:
        reject("CLOCK_CHANGED", "Original campaign identity must remain preserved")
    from .gates import require_component
    require_component(ctx, "handoff")
    ctx.bind(args.manifest, "EXPLICIT_COMPACT_INCLUDE_MANIFEST")
    result = create_handoff(args.manifest, repo_root=ctx.repo, output_root=ctx.root,
        guard=lambda: ctx.guard(plan["deadline_epoch"]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
