"""Authenticate the explicitly approved, fixed 2026-09-27 operational extension.

Standard library only: publication/handoff may use this without scientific imports.
No clock, source, configuration or evidence file is written by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from types import MappingProxyType


_EXPECTED_PINS = MappingProxyType({
    "approval_sha256": "96db48d27571a9d9a26459c8d17777a91b1799ba9109a9bd3ba03cc476af00e7",
    "clock_sha256": "ee910649e1a606b73c5d444753ea66118aed420ee69b6d3597b5986aec34ff38",
    "original_config_sha256": "6713495e409887278e2af73761606b460e70fff87fa48a62d0aceeea640afdfc",
    "training_result_sha256": "d615c75eaa4b8f88d6d01e50d000a77580d1b099d2eb38c00820f3a661560adb",
    "registry_sha256": "cf114b7b9f0a3fba9741a7095beefea5238affaf9f21c2ca37d4dc584c4a01fa",
})
_AUTHORIZATION_EPOCH = 1790531577
_HARD_DEADLINE_EPOCH = 1790617977
_ORIGINAL_START_EPOCH = 1790304598
_ORIGINAL_HARD_DEADLINE_EPOCH = 1790491798
_ALLOWED_ACTIONS = frozenset({"preflight", "inference", "assignments", "analysis", "publication", "handoff"})
_SCIENCE_ACTIONS = frozenset({"preflight", "inference", "assignments"})
_CONFIG_KEYS = frozenset({"schema", *(name for prefix in
    ("approval", "clock", "original_config", "training_result", "registry")
    for name in (prefix + "_path", prefix + "_sha256"))})


def config_identity(config):
    """Canonical identity for detecting mutation of an already admitted mapping."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf8")).hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _file_state(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def _same_path(left, right):
    return Path(left).resolve() == Path(right).resolve()


def _reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def authorized_deadline(record, action=None, now=None):
    """Choose an action cutoff from an already authenticated record; no file I/O."""
    current = time.time() if now is None else now
    _require(_finite(current), "Continuation observation time must be finite")
    if action is not None:
        _require(action in _ALLOWED_ACTIONS and action in record["allowed_actions"],
                 "Action is not authorized by the continuation: " + str(action))
    _require(current >= record["start_epoch"], "Continuation cannot precede the user authorization")
    deadline = record["science_deadline_epoch"] if action in _SCIENCE_ACTIONS else record["hard_deadline_epoch"]
    if current >= deadline:
        raise TimeoutError("Fixed continuation action deadline reached: " + str(action))
    return float(deadline)


def validate_continuation(config, action=None, now=None):
    """Return authenticated normalized authority, or None for untouched legacy.

    Production pins are code constants, not supplied by the configuration. Tests
    may monkeypatch _EXPECTED_PINS for isolated synthetic files; no config field
    or environment variable provides an alternate trust root.
    """
    if "continuation" not in config:
        return None
    spec = config["continuation"]
    _require(isinstance(spec, dict) and set(spec) == _CONFIG_KEYS,
             "Continuation requires the exact authenticated configuration schema")
    _require(spec["schema"] == "pacnhn.continuation.config.v1", "Unknown continuation schema")
    for name, expected in _EXPECTED_PINS.items():
        _require(spec.get(name) == expected, "Continuation differs from pinned authority: " + name)
    files = {}

    def bound(path, expected, *, decode=True):
        p = Path(path).resolve(strict=True)
        before = _file_state(p)
        if decode:
            raw = p.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
        else:
            digest = _sha(p)
        _require(digest == expected, "Continuation bound file changed: " + str(p))
        _require(_file_state(p) == before, "Continuation input changed during authentication: " + str(p))
        files[str(p)] = dict(sha256=digest, **before)
        return json.loads(raw, parse_constant=_reject_constant) if decode else None

    approval = bound(spec["approval_path"], spec["approval_sha256"])
    clock = bound(spec["clock_path"], spec["clock_sha256"])
    original = bound(spec["original_config_path"], spec["original_config_sha256"])
    original_view = {k: v for k, v in config.items() if k != "continuation"}
    _require("continuation" not in original and original_view == original,
             "Continuation may not change any original scientific configuration field")
    _require(approval["schema"] == "pacnhn.continuation.approval.v1"
             and clock["schema"] == "pacnhn.continuation.clock.v1", "Invalid authority/clock schema")
    _require(approval["decision"] == "AUTHORIZE_TIME_EXTENSION_24H"
             and approval["authority_kind"] == "explicit_user_message"
             and approval["user_statement"] == "I hereby give additional 24 hours (from now). You can keep working."
             and approval["authorization_turn_id"] == "01a0e3ff-a4af-79d1-b5be-72c4affce7c3",
             "Continuation lacks the exact user authorization")
    for value in (approval, clock):
        _require(value["phase_id"] == config["phase_id"] == "PACNHN_V1"
                 and value["run_id"] == config["run_id"], "Continuation run/phase mismatch")
        _require(value["authorization_epoch"] == _AUTHORIZATION_EPOCH
                 and value["hard_deadline_epoch"] == _HARD_DEADLINE_EPOCH
                 and value["additional_seconds"] == 86400
                 and value["hard_deadline_epoch"] - value["authorization_epoch"] == 86400,
                 "Continuation window is fixed to the original user receipt; reset prohibited")
        _require(value["original_start_epoch"] == config["original_start_epoch"] == _ORIGINAL_START_EPOCH
                 and value["original_hard_deadline_epoch"] == config["hard_deadline_epoch"] == _ORIGINAL_HARD_DEADLINE_EPOCH,
                 "Original campaign clocks must remain unchanged")
        _require(value["stage3_max_seconds"] == 43200 and value["publication_max_seconds"] == 14400
                 and value["publication_reserve_seconds"] == 14400, "Stage/reserve limits differ from authority")
    _require(clock["continuation_start_epoch"] == _AUTHORIZATION_EPOCH
             and clock["science_deadline_epoch"] == _HARD_DEADLINE_EPOCH - 14400
             and clock["clock_reset_allowed"] is False, "Continuation clock reset/reserve mismatch")
    _require(set(approval["allowed_actions"]) == _ALLOWED_ACTIONS
             and len(approval["allowed_actions"]) == len(_ALLOWED_ACTIONS), "Continuation action scope changed")
    _require(approval["retraining_authorized"] is False
             and approval["scientific_settings_change_authorized"] is False
             and approval["next_campaign_authorized"] is False
             and approval["automatic_technical_gate_continuation"] is True, "Continuation scope is not time-only")
    _require(_same_path(approval["original_config_path"], spec["original_config_path"])
             and approval["original_config_sha256"] == spec["original_config_sha256"], "Original configuration binding differs")
    _require(_same_path(clock["approval_path"], spec["approval_path"])
             and clock["approval_sha256"] == spec["approval_sha256"], "Clock binds another approval")
    _require(approval["original_contract_sha256"] == config["contract_sha256"], "Original contract binding differs")
    bound(config["contract_path"], config["contract_sha256"], decode=False)
    _require(approval["frozen_registry_sha256"] == spec["registry_sha256"], "Approval binds another frozen registry")
    result = bound(spec["training_result_path"], spec["training_result_sha256"])
    output = Path(config["output_root"])
    _require(_same_path(spec["training_result_path"], output / "training/RESULT.json")
             and _same_path(spec["registry_path"], output / "training/REGISTRY_FREEZE.json")
             and _same_path(result["registry"], spec["registry_path"]), "Frozen training paths differ")
    _require(result["status"] == "STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE"
             and result["registry_sha256"] == spec["registry_sha256"], "Continuation requires completed frozen training")
    bound(spec["registry_path"], spec["registry_sha256"], decode=False)
    normalized = dict(phase_id=config["phase_id"], run_id=config["run_id"],
        approval_path=str(Path(spec["approval_path"]).resolve()), approval_sha256=spec["approval_sha256"],
        clock_path=str(Path(spec["clock_path"]).resolve()), clock_sha256=spec["clock_sha256"],
        start_epoch=_AUTHORIZATION_EPOCH, hard_deadline_epoch=_HARD_DEADLINE_EPOCH,
        science_deadline_epoch=_HARD_DEADLINE_EPOCH - 14400,
        stage3_max_seconds=43200, publication_max_seconds=14400, publication_reserve_seconds=14400,
        original_hard_deadline_epoch=_ORIGINAL_HARD_DEADLINE_EPOCH,
        allowed_actions=tuple(approval["allowed_actions"]), config_identity=config_identity(config),
        identity_files=files, original_52h_requirement_met=False)
    authorized_deadline(normalized, action=action, now=now)
    return normalized


def effective_deadline(config, action=None):
    record = validate_continuation(config, action=action)
    return float(config["hard_deadline_epoch"]) if record is None else authorized_deadline(record, action=action)


def check_cached_identity(config, record):
    """Cheap guard of immutable admitted inputs; never rehash registry per tick.

    A changed stat signature is rejected, not silently re-admitted. Full hashes
    are authenticated at Context construction and by each independent validator.
    """
    _require(config_identity(config) == record["config_identity"], "Admitted continuation configuration mutated")
    for path, identity in record["identity_files"].items():
        _require(_file_state(path) == {k: identity[k] for k in ("bytes", "mtime_ns", "ctime_ns")},
                 "Admitted continuation file changed: " + path)
