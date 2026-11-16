"""Small synthetic packaging regressions; never use campaign data or clocks.

These tests create temporary bytes only. Their execution is separately gated by
the campaign controller and must not overlap an active protected national job.
"""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from partC_rev1_src.pacnhn_v1 import handoff as h


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value, encoding="utf-8")


def make_case(tmp_path):
    repo, out = tmp_path/"repo", tmp_path/"out"
    repo.mkdir(); out.mkdir()
    includes = []

    def include(relative, member, role, content, source="OUT"):
        path = (out if source == "OUT" else repo)/relative
        put(path, content)
        row = dict(source_root=source, source_relative_path=relative, archive_path=member,
                   sha256=h.digest(path), size_bytes=path.stat().st_size, role=role)
        includes.append(row)
        return row

    for name in sorted(h.ROOT_REPORTS):
        role = "STATUS" if name == "02_HANDOFF_STATUS.json" else "SUMMARY"
        content = dict(phase_id=h.PHASE, run_id="synthetic", status="WAITING_FOR_EXTERNAL_REVIEW",
                       next_step_authorized=False, next_campaign_authorized=False,
                       external_acceptance="NOT_RECORDED") if role == "STATUS" else "Synthetic fixture only.\n"
        include("reports/"+name, name, role, content)
    for section in sorted(h.SECTIONS):
        include("reports/"+section+".md", section+"/NOT_APPLICABLE.md", "EXPLICIT_DISPOSITION", "Synthetic fixture: no measured content.\n")
    include(h.OWNED[0]+"tiny.py", "code_snapshot/"+h.OWNED[0]+"tiny.py", "CODE", "VALUE = 1\n", "REPO")
    include(h.OWNED[0]+"tests/test_tiny.py", "code_snapshot/"+h.OWNED[0]+"tests/test_tiny.py", "TEST", "def test_tiny():\n    assert True\n", "REPO")
    include(h.OWNED[1]+"APPROVED_PLAN.md", "governing_docs/"+h.OWNED[1]+"APPROVED_PLAN.md", "CONTRACT", "Synthetic contract.\n", "REPO")
    include("reports/config.json", "configs/config.json", "CONFIG", {"synthetic": True})
    evidence = include("reports/evidence.json", "summary_data/evidence.json", "FINAL_EVIDENCE", {"synthetic": True, "science_run": False})
    include("reports/hardware.json", "summary_data/hardware.json", "HARDWARE_METADATA", {"synthetic": True})
    include("reports/publication.json", "summary_data/publication.json", "PUBLICATION_DISPOSITION", {"status": "UNAVAILABLE_SYNTHETIC_FIXTURE"})
    include("reports/bulk.csv", "external_artifact_index/bulk.csv", "BULK_INDEX", ",".join(sorted(h.BULK_FIELDS))+"\n")
    binding = dict(path=evidence["archive_path"], sha256=evidence["sha256"], bytes=evidence["size_bytes"], role="SYNTHETIC_ONLY")
    rows = []
    for name in sorted(h.REQUIRED_IDS):
        pending = name in {"H03", "H04"}
        rows.append(dict(id=name, final_audit=dict(status="PARTIAL_WITH_LIMITATIONS", audited_epoch=h.START+10,
            reviewer="synthetic test", disposition_reason="PENDING_SEAL_ONLY: archive test" if pending else "Synthetic fixture limitation, not campaign completion.",
            evidence_bindings=[binding], unresolved=[])))
    audit = dict(phase_id=h.PHASE, run_id="synthetic", requirements=rows,
        final_audit_summary=dict(unaudited_rows=0, required_rows=51, passed_rows=0, qualified_rows=51,
                                failed_or_blocked_rows=0, next_campaign_authorized=False))
    include("reports/audit.json", "summary_data/final_audit.json", "FINAL_AUDIT", audit)
    plan = dict(schema="pacnhn.handoff.manifest.v1", phase_id=h.PHASE, step_id="handoff", run_id="synthetic", campaign_config_sha256=h.digest(out/"reports/config.json"),
        handoff_id="synthetic_v1", original_start_epoch=h.START, deadline_epoch=h.START+300,
        tasks=["VALIDATE_INPUTS", "BUILD_COMPACT_ZIP", "VERIFY_ARCHIVE"], expected_outputs=["verified synthetic archive"],
        acceptance_tests=["reopened SHA-256"], resource_budget=dict(target_zip_bytes=1000000, hard_zip_bytes=2000000, max_member_bytes=1000000),
        approved_output_prefixes=["reports"], includes=includes, final_audit_member="summary_data/final_audit.json",
        bulk_index_member="external_artifact_index/bulk.csv", status_member="02_HANDOFF_STATUS.json",
        publication_disposition="UNAVAILABLE_WITH_EXPLICIT_EVIDENCE", standalone_assets=[])
    manifest = out/"manifest.json"; put(manifest, plan)
    return repo, out, manifest, plan, audit


def refresh(plan, root, role):
    row = next(r for r in plan["includes"] if r["role"] == role)
    path = root/row["source_relative_path"]
    row.update(sha256=h.digest(path), size_bytes=path.stat().st_size)


def execute(case):
    repo, out, manifest, plan, _ = case
    put(manifest, plan)
    return h.create_handoff(manifest, repo_root=repo, output_root=out, guard=lambda: None, now=lambda: h.START+20)


def assert_rejected(case, code):
    with pytest.raises(h.HandoffRejected) as caught:
        execute(case)
    assert caught.value.code == code
    attempt = case[1]/"handoff_attempts"/case[3]["handoff_id"]
    assert json.loads((attempt/"REJECTION.json").read_text())["code"] == code
    assert (attempt/"REQUESTED_ARTIFACT_INDEX.json").exists()
    assert not (attempt/"FINAL_SEAL_RECEIPT.json").exists()


@pytest.mark.parametrize("compression", ["DEFLATE", "LZMA"])
def test_complete_synthetic_archive_reopens_and_preserves_source_layout(tmp_path, compression):
    case = make_case(tmp_path); case[3]["compression"] = compression
    receipt = execute(case)
    archive = Path(receipt["archive_path"])
    assert h.digest(archive) == receipt["archive_sha256"]
    assert receipt["verification"]["member_sha256"] == "PASS"
    assert receipt["populated_requirement_count"] == 51
    assert receipt["closed_pending_seal_only_ids"] == ["H03", "H04"]
    assert sum(receipt["scientific_disposition_counts"].values()) == 49
    assert receipt["scientific_acceptance_inferred"] is False
    assert receipt["external_acceptance"] == "NOT_RECORDED"
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert "code_snapshot/"+h.OWNED[0]+"tiny.py" in z.namelist()
        assert "MANIFEST_SHA256.csv" in z.namelist()
        assert json.loads(z.read("summary_data/final_audit.json"))["requirements"] == case[4]["requirements"]


@pytest.mark.parametrize("name", ["../a.csv", "C:/a.csv", "/a.csv", "a\\b.csv", "a//b.csv", "a/./b.csv", "CON.txt", "a?.csv", "a .", "a|b.csv"])
def test_nonportable_or_escaping_member_names_rejected(name):
    with pytest.raises(h.HandoffRejected): h.safe_relative(name)


def test_pending_seal_cannot_hide_missing_science(tmp_path):
    case = make_case(tmp_path); audit = case[4]
    audit["requirements"][0]["final_audit"]["disposition_reason"] = "PENDING_SEAL_ONLY: invalid scientific row"
    put(case[1]/"reports/audit.json", audit); refresh(case[3], case[1], "FINAL_AUDIT")
    assert_rejected(case, "INVALID_PENDING_SEAL")


@pytest.mark.parametrize("alter,code", [("missing", "FINAL_AUDIT_COVERAGE"), ("unaudited", "FINAL_AUDIT_UNPOPULATED"), ("unbound", "FINAL_AUDIT_UNBOUND_EVIDENCE"), ("counts", "FINAL_AUDIT_SUMMARY")])
def test_populated_bound_audit_required(tmp_path, alter, code):
    case = make_case(tmp_path); audit = case[4]
    if alter == "missing": audit["requirements"].pop()
    if alter == "unaudited": audit["requirements"][0]["final_audit"]["status"] = "NOT_AUDITED"
    if alter == "unbound": audit["requirements"][0]["final_audit"]["evidence_bindings"][0]["path"] = "undeclared.json"
    if alter == "counts": audit["final_audit_summary"]["qualified_rows"] = 0
    put(case[1]/"reports/audit.json", audit); refresh(case[3], case[1], "FINAL_AUDIT")
    assert_rejected(case, code)


@pytest.mark.parametrize("field,value", [("next_step_authorized", True), ("next_campaign_authorized", True), ("external_acceptance", "ACCEPTED")])
def test_external_acceptance_and_next_campaign_never_self_authorized(tmp_path, field, value):
    case = make_case(tmp_path); path = case[1]/"reports/02_HANDOFF_STATUS.json"
    status = h.read_json(path); status[field] = value; put(path, status); refresh(case[3], case[1], "STATUS")
    assert_rejected(case, "UNSAFE_FINAL_STATUS")


def test_source_identity_mismatch_is_indexed_not_silently_skipped(tmp_path):
    case = make_case(tmp_path); put(case[1]/"reports/evidence.json", "Changed fixture\n")
    assert_rejected(case, "SOURCE_IDENTITY_CHANGED")


def test_member_cap_rejects_complete_attempt_with_artifact_index(tmp_path):
    case = make_case(tmp_path); case[3]["resource_budget"]["max_member_bytes"] = 8
    assert_rejected(case, "MEMBER_TOO_LARGE")


def test_archive_hard_cap_preserves_complete_candidate(tmp_path):
    case = make_case(tmp_path); case[3]["resource_budget"].update(target_zip_bytes=32, hard_zip_bytes=32)
    assert_rejected(case, "ZIP_TOO_LARGE")
    assert (case[1]/"handoff_attempts/synthetic_v1/synthetic_v1.candidate.zip").exists()


def test_existing_attempt_is_never_overwritten(tmp_path):
    case = make_case(tmp_path); receipt = execute(case); sha = receipt["archive_sha256"]
    with pytest.raises(h.HandoffRejected, match="immutable"): execute(case)
    assert h.digest(receipt["archive_path"]) == sha


@pytest.mark.parametrize("path", ["cache/a.json", "checkpoints/a.json", "full_paths/a.csv", "a.npz", "a.zip", "common_graph.parquet"])
def test_prohibited_payloads_not_allowed_by_role(path):
    with pytest.raises(h.HandoffRejected): h.permitted_payload(path, "OUT", "SUMMARY_DATA")


def test_archive_reopen_detects_extra_or_changed_member(tmp_path):
    payload = b"abc"; checksum = hashlib.sha256(payload).hexdigest()
    rows = [dict(relative_path="a.txt", size_bytes=3, sha256=checksum)]
    manifest = b"synthetic manifest"; manifest_sha = hashlib.sha256(manifest).hexdigest()
    path = tmp_path/"changed.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("a.txt", b"bad"); z.writestr("MANIFEST_SHA256.csv", manifest)
    with pytest.raises(h.HandoffRejected) as caught: h.verify_archive(path, rows, manifest_sha)
    assert caught.value.code == "ARCHIVE_MEMBER_SHA"
    with zipfile.ZipFile(path, "a") as z: z.writestr("extra.txt", b"extra")
    with pytest.raises(h.HandoffRejected) as caught: h.verify_archive(path, rows, manifest_sha)
    assert caught.value.code == "ARCHIVE_MEMBER_SET"


def test_missing_required_section_rejected(tmp_path):
    case = make_case(tmp_path)
    case[3]["includes"] = [r for r in case[3]["includes"] if not r["archive_path"].startswith("geospatial_summary/")]
    assert_rejected(case, "MISSING_REQUIRED_INVENTORY")


def test_unavailable_publication_cannot_claim_unqualified_pdf_pass(tmp_path):
    case = make_case(tmp_path); audit = case[4]
    next(r for r in audit["requirements"] if r["id"] == "P05")["final_audit"]["status"] = "PASS"
    audit["final_audit_summary"].update(passed_rows=1, qualified_rows=50)
    put(case[1]/"reports/audit.json", audit); refresh(case[3], case[1], "FINAL_AUDIT")
    assert_rejected(case, "PUBLICATION_AUDIT_CONTRADICTION")
