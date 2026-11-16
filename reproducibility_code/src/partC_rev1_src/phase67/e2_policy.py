"""Executable invariants for the explicit post-audit E2 continuation."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

START = 1788883986.0
DEADLINE = 1788970386.0
MIN_RESERVE = 5400.0
EPISODE = 'p67_postaudit_e2_0908'
REPAIR_CEILING = 8


def science_left(now: float, forecast: float = 0.0) -> float:
    if not math.isfinite(forecast) or forecast < 0 or not math.isfinite(now):
        raise ValueError('invalid finalization forecast or clock')
    return DEADLINE - max(MIN_RESERVE, 1.5 * forecast) - now


def admitted(now: float, *, forecast: float = 0.0, job_seconds: float = 0.0,
             repair_pass: bool, epoch_frozen: bool, audit_plan_frozen: bool,
             integrity_pass: bool) -> bool:
    if not math.isfinite(job_seconds) or job_seconds < 0:
        raise ValueError('negative job forecast')
    return (repair_pass and not epoch_frozen and audit_plan_frozen and integrity_pass
            and science_left(now, forecast) > job_seconds)


def validate_roots(parent: str | Path, child: str | Path) -> None:
    p, c = Path(parent).resolve(), Path(child).resolve()
    if p == c or p in c.parents or c in p.parents:
        raise ValueError('sealed parent and writable continuation must be disjoint roots')


def allowed_model_role(role: str, *, use: str, final_freeze: bool) -> bool:
    if use == 'train':
        return role in {'TRAIN', 'TRAIN_old', 'TRAIN_new', 'E2_TRAIN'}
    if use == 'select':
        return role in {'VALID', 'VALID_old', 'PATCH_VALID', 'E2_VALID'}
    if use == 'report':
        return final_freeze
    raise ValueError('unknown use')


def seed_for(region: str, role: str, slot: int, redraw: int = 0) -> int:
    if slot < 0 or redraw < 0:
        raise ValueError('negative seed field')
    data = json.dumps([EPISODE, 'AUDIT2_FIXED_20260908', region, role, slot, redraw],
                      separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'big')


def continuation_identity(start: float, deadline: float, repair_ceiling: int) -> None:
    if start != START or deadline != DEADLINE or repair_ceiling != REPAIR_CEILING:
        raise ValueError('no fresh 24h clock or blanket repair allowance')


def old_holdout_is_new_final_test(role: str) -> bool:
    return False
