"""Synthetic continuation identity/window tests; no scientific input or execution."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from partC_rev1_src.pacnhn_v1 import continuation as con
from partC_rev1_src.pacnhn_v1 import control


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def authority(tmp_path, monkeypatch):
    now = con._AUTHORIZATION_EPOCH + 100
    monkeypatch.setattr(con.time, "time", lambda: now)
    root = tmp_path / "run"
    contract = tmp_path / "contract.md"
    contract.write_text("synthetic approved immutable contract", encoding="utf8")
    config = dict(phase_id="PACNHN_V1", run_id="synthetic_run", original_start_epoch=1790304598,
        hard_deadline_epoch=1790491798, output_root=str(root), contract_path=str(contract),
        contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(),
        tasks=["preflight", "assignments", "publication", "handoff", "training", "reference_open", "support"],
        input_versions={"synthetic": "metadata-only"}, expected_outputs=["synthetic"], acceptance_tests=["synthetic"],
        resources={"tree_rss_bytes": 512*1024**2, "minimum_available_bytes": 0},
        assignment={"RG_SPTT": .001}, topology={"capacity_multiplier": .3}, training={"seconds": 86400})
    original_path = tmp_path / "original.json"
    original_hash = put(original_path, config)
    registry_path = root / "training/REGISTRY_FREEZE.json"
    registry_hash = put(registry_path, {"synthetic": "fixed registry bytes; not a scientific model"})
    result_path = root / "training/RESULT.json"
    result_hash = put(result_path, dict(status="STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE",
        registry=str(registry_path), registry_sha256=registry_hash))
    approval_path = tmp_path / "APPROVAL.json"
    approval = dict(schema="pacnhn.continuation.approval.v1", phase_id=config["phase_id"], run_id=config["run_id"],
        decision="AUTHORIZE_TIME_EXTENSION_24H", authority_kind="explicit_user_message",
        user_statement="I hereby give additional 24 hours (from now). You can keep working.",
        authorization_turn_id="01a0e3ff-a4af-79d1-b5be-72c4affce7c3", authorization_epoch=1790531577,
        hard_deadline_epoch=1790617977, additional_seconds=86400,
        original_start_epoch=1790304598, original_hard_deadline_epoch=1790491798,
        original_config_path=str(original_path), original_config_sha256=original_hash,
        original_contract_sha256=config["contract_sha256"], frozen_registry_sha256=registry_hash,
        allowed_actions=sorted(con._ALLOWED_ACTIONS), stage3_max_seconds=43200,
        publication_max_seconds=14400, publication_reserve_seconds=14400, retraining_authorized=False,
        scientific_settings_change_authorized=False, next_campaign_authorized=False,
        automatic_technical_gate_continuation=True)
    approval_hash = put(approval_path, approval)
    clock_path = tmp_path / "CLOCK.json"
    clock = dict(schema="pacnhn.continuation.clock.v1", phase_id=config["phase_id"], run_id=config["run_id"],
        authorization_epoch=1790531577, continuation_start_epoch=1790531577, hard_deadline_epoch=1790617977,
        science_deadline_epoch=1790603577, additional_seconds=86400, stage3_max_seconds=43200,
        publication_max_seconds=14400, publication_reserve_seconds=14400,
        approval_path=str(approval_path), approval_sha256=approval_hash,
        original_start_epoch=1790304598, original_hard_deadline_epoch=1790491798, clock_reset_allowed=False)
    clock_hash = put(clock_path, clock)
    spec = dict(schema="pacnhn.continuation.config.v1", approval_path=str(approval_path), approval_sha256=approval_hash,
        clock_path=str(clock_path), clock_sha256=clock_hash, original_config_path=str(original_path),
        original_config_sha256=original_hash, training_result_path=str(result_path), training_result_sha256=result_hash,
        registry_path=str(registry_path), registry_sha256=registry_hash)
    config["continuation"] = spec
    pins = {k: spec[k] for k in con._EXPECTED_PINS}
    monkeypatch.setattr(con, "_EXPECTED_PINS", pins)
    config_path = tmp_path / "continued.json"
    put(config_path, config)
    return dict(config=config, path=config_path, original_path=original_path, approval=approval, clock=clock,
                pins=pins, now=now, tmp=tmp_path)


def test_absent_continuation_preserves_legacy_and_does_not_open_files(monkeypatch):
    c = {"hard_deadline_epoch": 12.}
    monkeypatch.setattr(con, "_sha", lambda *_: pytest.fail("Legacy must not hash extension assets"))
    assert con.validate_continuation(c, action="training", now=999.) is None
    assert con.effective_deadline(c, action="support") == 12.


@pytest.mark.parametrize("action", sorted(con._ALLOWED_ACTIONS))
def test_valid_authority_normalized_and_action_deadline(authority, action):
    c = authority["config"]; before = copy.deepcopy(c)
    r = con.validate_continuation(c, action=action)
    assert r["start_epoch"] == 1790531577
    assert r["stage3_max_seconds"] == 43200 and r["publication_max_seconds"] == 14400
    assert r["approval_sha256"] == c["continuation"]["approval_sha256"]
    assert r["clock_sha256"] == c["continuation"]["clock_sha256"]
    assert r["original_52h_requirement_met"] is False
    assert con.effective_deadline(c, action=action) == (1790603577 if action in con._SCIENCE_ACTIONS else 1790617977)
    assert c == before


@pytest.mark.parametrize("action", ["training", "train", "pilot", "reference", "reference_open", "reference_i95", "support", "witness-audit", "topology", "unknown"])
def test_unauthorized_action_fails(authority, action):
    with pytest.raises(ValueError, match="not authorized"):
        con.validate_continuation(authority["config"], action=action)


@pytest.mark.parametrize("name", ["approval", "clock", "original_config", "training_result", "registry"])
def test_missing_bound_file_fails(authority, name):
    Path(authority["config"]["continuation"][name + "_path"]).unlink()
    with pytest.raises(FileNotFoundError):con.validate_continuation(authority["config"])


@pytest.mark.parametrize("name", ["approval", "clock", "original_config", "training_result", "registry"])
def test_tampered_bound_bytes_fail(authority, name):
    path = Path(authority["config"]["continuation"][name + "_path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="bound file changed"):
        con.validate_continuation(authority["config"])


def test_self_declared_forged_hash_chain_is_not_a_trust_root(authority):
    c = authority["config"]; path = Path(c["continuation"]["approval_path"])
    forged = dict(authority["approval"], authorization_epoch=authority["now"])
    c["continuation"]["approval_sha256"] = put(path, forged)
    with pytest.raises(ValueError, match="pinned authority"):
        con.validate_continuation(c)


@pytest.mark.parametrize("field", ["assignment", "topology", "training", "resources", "hard_deadline_epoch", "new_unapproved_field"])
def test_full_original_configuration_equality_prevents_science_changes(authority, field):
    c = authority["config"]; c[field] = {"changed": True}
    with pytest.raises(ValueError, match="original scientific configuration"):
        con.validate_continuation(c)


@pytest.mark.parametrize("value", [None, {}, {"schema": "unapproved"}])
def test_present_but_invalid_extension_cannot_fall_back_to_legacy(value):
    with pytest.raises(ValueError):con.validate_continuation({"continuation": value})


@pytest.mark.parametrize("field,value", [
    ("continuation_start_epoch", 1790531578), ("authorization_epoch", 1790531578),
    ("hard_deadline_epoch", 1790617978), ("science_deadline_epoch", 1790617977),
    ("additional_seconds", 90000), ("clock_reset_allowed", True),
    ("stage3_max_seconds", 86400), ("publication_max_seconds", 18000),
    ("publication_reserve_seconds", 0), ("original_hard_deadline_epoch", 1790617977)])
def test_semantic_reset_or_expansion_rejected_even_with_test_clock_pin(authority, field, value):
    clock = dict(authority["clock"], **{field: value});c = authority["config"]
    digest = put(Path(c["continuation"]["clock_path"]), clock)
    c["continuation"]["clock_sha256"] = digest;authority["pins"]["clock_sha256"] = digest
    with pytest.raises(ValueError):con.validate_continuation(c)


@pytest.mark.parametrize("now", [1790617977, 1790617978])
def test_hard_expiry_inclusive(authority, now):
    with pytest.raises(TimeoutError):con.validate_continuation(authority["config"], action="publication", now=now)


def test_science_reserve_and_publication_window_are_distinct(authority):
    with pytest.raises(TimeoutError):con.validate_continuation(authority["config"], action="assignments", now=1790603577)
    for action in ("analysis", "publication", "handoff"):
        assert con.validate_continuation(authority["config"], action=action, now=1790603577)
    with pytest.raises(ValueError):con.validate_continuation(authority["config"], now=1790531576)
    with pytest.raises(ValueError):con.validate_continuation(authority["config"], now=float("nan"))


def test_contract_mutation_rejected(authority):
    Path(authority["config"]["contract_path"]).write_text("changed", encoding="utf8")
    with pytest.raises(ValueError, match="bound file changed"):con.validate_continuation(authority["config"])


def test_context_caches_bulk_identity_and_explicit_actions_without_step_mutation(authority, monkeypatch):
    ctx = control.Context(authority["path"], step_id="publication")
    monkeypatch.setattr(con, "_sha", lambda *_: pytest.fail("Guard must not rehash bulk files"))
    r = ctx.require_action("analysis")
    assert r is ctx.continuation and ctx.step_id == "publication"
    ctx.guard()
    assert (ctx.root / "resources" / f"publication_{control.os.getpid()}.json").exists()
    with pytest.raises(ValueError):ctx.require_action("training")


def test_context_guard_uses_explicit_action_cutoff_and_preserves_earlier_deadline(authority, monkeypatch):
    ctx = control.Context(authority["path"], step_id="publication")
    ctx.require_action("assignments")
    monkeypatch.setattr(con.time, "time", lambda: 1790603577)
    with pytest.raises(TimeoutError):ctx.guard()
    ctx.require_action("analysis");ctx.guard()
    with pytest.raises(TimeoutError):ctx.guard(deadline=1790603576)
    assert ctx.c["hard_deadline_epoch"] == 1790491798


@pytest.mark.parametrize("action", [None, "", False, 0, []])
def test_context_rejects_missing_action_without_widening_science_cutoff(authority, action):
    ctx = control.Context(authority["path"], step_id="assignments")
    with pytest.raises(ValueError, match="explicit nonempty action"):
        ctx.require_action(action)
    assert ctx._continuation_action == "assignments" and ctx.step_id == "assignments"


def test_context_guard_detects_config_and_cached_file_mutation(authority):
    ctx = control.Context(authority["path"], step_id="publication")
    ctx.c["assignment"]["RG_SPTT"] = .1
    with pytest.raises(ValueError, match="configuration mutated"):ctx.guard()
    ctx.c["assignment"]["RG_SPTT"] = .001
    path = Path(ctx.c["continuation"]["clock_path"]);path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="file changed"):ctx.require_action("publication")


def test_context_legacy_deadline_and_return_are_unchanged(authority):
    ctx = control.Context(authority["original_path"], step_id="training")
    assert ctx.continuation is None and ctx.require_action("unknown_legacy_action") is None
    with pytest.raises(TimeoutError, match="Original campaign"):ctx.guard()


def test_context_denies_training_before_creating_output_artifacts(authority):
    with pytest.raises(ValueError, match="not authorized"):
        control.Context(authority["path"], step_id="training")
    assert not (Path(authority["config"]["output_root"])/"resources").exists()


def test_production_pins_are_not_derived_from_configuration():
    expected = "96db48d27571a9d9a26459c8d17777a91b1799ba9109a9bd3ba03cc476af00e7"
    assert con._EXPECTED_PINS["approval_sha256"] == expected
    c = {"continuation": {"schema": "pacnhn.continuation.config.v1"}}
    for prefix in ("approval", "clock", "original_config", "training_result", "registry"):
        c["continuation"][prefix + "_path"] = "untrusted.json"
        c["continuation"][prefix + "_sha256"] = "0"*64
    with pytest.raises(ValueError, match="pinned authority"):con.validate_continuation(c)
