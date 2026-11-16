"""Pure, metadata-only XGB quota proposals; no files, models or configuration writes.

The caller authenticates the supplied snapshots and persists the returned complete
manifest in a NEW configuration revision. Byte counts include UBJ plus all native
parameter buffers, not the compressed NPZ size. This is an allocation forecast,
not a model quality, measured resident-memory or successful-training claim.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re

SCHEMA = 'RECOVERY_XGB_QUOTA_REALLOCATION_V1'
SUITES = ('s02', 's03', 's04', 's05', 's06')
SUITE_CAP = 2_989_862_912
REGION_CAP = 256 * 1024**2
MIN_QUOTA = 192 * 1024
EXPECTED_REGIONS = 10_442
DEPTHS = (3, 5, 7)
MAX_ROUNDS = 128


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Exact lowercase SHA-256 required')
    return value


def _binding(value):
    if not isinstance(value, dict) or not isinstance(value.get('path'), str) or not value['path']:
        raise ValueError('Explicit authenticated metadata path and digest required')
    return dict(path=value['path'], sha256=_digest(value.get('sha256')))


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid integer '+name)
    return value


def _key(value):
    if not isinstance(value, str) or not re.fullmatch(r's0[2-6]:\d{6}', value):
        raise ValueError('Invalid active region key')
    return value


def _score(value):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _anchor(candidate):
    if candidate is None:
        return None
    if not isinstance(candidate, dict):
        raise ValueError('Invalid selected/latest anchor')
    raw = _integer(candidate.get('uncompressed_booster_bytes'), 'raw UBJ bytes')
    native = _integer(candidate.get('native_parameter_bytes'), 'native parameter bytes')
    total = _integer(candidate.get('uncompressed_model_bytes'), 'total model bytes', 1)
    if total != raw + native or total > REGION_CAP:
        raise ValueError('Raw UBJ and native buffers must equal complete bounded model bytes')
    rounds = _integer(candidate.get('boosting_rounds_total'), 'rounds')
    depth = candidate.get('max_depth')
    if rounds > MAX_ROUNDS or depth not in DEPTHS or type(depth) is not int:
        raise ValueError('Existing XGB round/depth limits cannot change')
    train_id = candidate.get('trained_dataset_identity')
    _digest(train_id)
    return dict(model_sha256=_digest(candidate.get('model_sha256')),
                uncompressed_booster_bytes=raw, native_parameter_bytes=native,
                uncompressed_model_bytes=total, boosting_rounds_total=rounds,
                max_depth=depth, trained_dataset_identity=train_id)


def observation_from_state(region_key, state, *, source_binding, twofold_trial=None):
    """Normalize an already authenticated STATE dict without opening its paths.

Unknown measurements are explicit, not estimates from compressed file sizes.
The full original state digest binds fields retained by this compact projection.
"""
    _key(region_key)
    selected = state.get('selected', {}).get('XGBOOST')
    audit = (selected or {}).get('audit', {})
    return dict(region_key=region_key, source_binding=_binding(source_binding),
        current_train_dataset_identity=state.get('dataset_identity'),
        current_valid_dataset_identity=state.get('valid_dataset_identity'),
        requires_current_dataset_refresh=state.get('requires_current_dataset_refresh', False),
        selected=copy.deepcopy(selected), latest=copy.deepcopy(state.get('xgb_latest')),
        audit_model_sha256=audit.get('model_sha256'),
        audit_train_dataset_identity=audit.get('train_dataset_identity'),
        audit_valid_dataset_identity=audit.get('valid_dataset_identity'),
        train_wmape_percent=audit.get('TRAIN', {}).get('WMAPE_percent'),
        valid_wmape_percent=audit.get('VALID', {}).get('WMAPE_percent'),
        recorded_target_met=audit.get('target_met'),
        twofold_trial=copy.deepcopy(twofold_trial))


def headroom(anchor, quota, current_train_identity, *, increment=8):
    """The unchanged legacy _xgb_block capacity rule, reported prospectively."""
    if anchor is None:
        return dict(status='NO_ANCHOR', append_rounds=None, restart_depth=None)
    rounds, size, depth = (anchor['boosting_rounds_total'],
                          anchor['uncompressed_model_bytes'], anchor['max_depth'])
    if rounds == 0:
        return dict(status='ZERO_ROUND_ANCHOR', append_rounds=0, restart_depth=None)
    append = max(0, min(increment, MAX_ROUNDS-rounds,
                      math.floor(rounds*quota/(1.5*size))-rounds))
    if append:
        return dict(status='APPEND', append_rounds=append, restart_depth=None)
    next_depth = DEPTHS[min(DEPTHS.index(depth)+1, len(DEPTHS)-1)]
    projected = size*increment/max(1, rounds)*2**(next_depth-depth)*1.5
    if projected > quota:
        next_depth = depth
    if next_depth != depth:
        return dict(status='NEW_DEPTH', append_rounds=0, restart_depth=next_depth)
    if anchor['trained_dataset_identity'] != current_train_identity:
        return dict(status='CHANGED_TRAIN_RESTART', append_rounds=0, restart_depth=depth)
    return dict(status='NO_NEW_TREE', append_rounds=0, restart_depth=None)


def _measured(obs):
    """Do not donate or prioritize from missing, stale or contradictory metadata."""
    try:
        selected, latest = _anchor(obs.get('selected')), _anchor(obs.get('latest'))
    except ValueError as exc:
        return None, 'INVALID_OR_MISSING_BYTE_MEASUREMENTS: '+str(exc)
    if selected is None or latest is None:
        return None, 'MISSING_SELECTED_OR_LATEST'
    train_id = obs.get('current_train_dataset_identity')
    valid_id = obs.get('current_valid_dataset_identity')
    try:
        _digest(train_id); _digest(valid_id)
    except ValueError:
        return None, 'MISSING_CURRENT_COHORT_IDENTITY'
    if (obs.get('requires_current_dataset_refresh') is not False
            or obs.get('audit_train_dataset_identity') != train_id
            or obs.get('audit_valid_dataset_identity') != valid_id
            or obs.get('audit_model_sha256') != selected['model_sha256']):
        return None, 'STALE_OR_UNBOUND_SELECTED_AUDIT'
    if not _score(obs.get('train_wmape_percent')) or not _score(obs.get('valid_wmape_percent')):
        return None, 'UNDEFINED_OR_INVALID_NATIVE_METRICS'
    qualified = obs['train_wmape_percent'] < 1.
    if type(obs.get('recorded_target_met')) is not bool or obs['recorded_target_met'] != qualified:
        return None, 'CONTRADICTORY_TARGET_FLAG'
    # A 10% raw-byte reserve protects both retained candidates. 3x resident RAM
    # remains the unchanged separate suite forecast; these are raw quota units.
    floor = max(MIN_QUOTA, (11*max(selected['uncompressed_model_bytes'],
                                 latest['uncompressed_model_bytes'])+9)//10)
    if floor > REGION_CAP:
        raise ValueError('Retained model plus declared safety exceeds per-region cap')
    return dict(selected=selected, latest=latest, floor=floor, qualified=qualified,
                train_id=train_id), None


def allocate_quotas(previous_manifest, observations, *, prior_binding, expected_keys,
                    multiplier=2, diagnostic_keys=()):
    """Return (complete_new_manifest, complete_override_map), without side effects.

Only measured, current TRAIN-qualified regions donate unused allocations. All
other regions keep their existing allocation before recipient increases. Explicit
diagnostic keys have reserved first priority; remaining requests sort by smallest
unmet native TRAIN error, then stable key. Requests are all-or-nothing 2x/4x of
the ORIGINAL allocation (subject to 256 MiB), never another exponential doubling.
4x requires a bound 2x technical trial under the exact preceding manifest.
"""
    prior_binding = _binding(prior_binding)
    if type(multiplier) is not int or multiplier not in (2, 4):
        raise ValueError('Only sequential 2x then 4x amendments are authorized')
    expected = list(expected_keys)
    if len(expected) != EXPECTED_REGIONS or len(set(expected)) != len(expected):
        raise ValueError('Complete distinct 10442-region frozen population required')
    for key in expected:
        _key(key)
    if {k.split(':')[0] for k in expected} != set(SUITES):
        raise ValueError('All five suites must remain represented')
    prior = copy.deepcopy(previous_manifest)
    if prior.get('per_suite_cap_bytes') != SUITE_CAP or prior.get('resident_multiplier') != 3.:
        raise ValueError('Original suite memory cap and resident factor are immutable')
    _binding(dict(path=prior.get('source_budget_path'), sha256=prior.get('source_budget_sha256')))
    old_rows = prior.get('rows', [])
    keys = [_key(row['region_key']) for row in old_rows]
    if len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise ValueError('Previous allocation does not exactly cover the frozen support')
    old = {}; old_sums = {sid: 0 for sid in SUITES}
    for row in old_rows:
        quota = _integer(row.get('raw_quota_bytes'), 'quota', 1)
        _integer(row.get('pairs'), 'active pairs', 1)
        if quota > REGION_CAP:
            raise ValueError('Previous quota exceeds per-region cap')
        baseline = _integer(row.get('baseline_raw_quota_bytes', quota), 'original quota', 1)
        if baseline > REGION_CAP:
            raise ValueError('Original quota exceeds per-region cap')
        row['baseline_raw_quota_bytes'] = baseline
        old[row['region_key']] = row
        old_sums[row['region_key'].split(':')[0]] += quota
    if old_sums != prior.get('suite_sums') or any(x > SUITE_CAP for x in old_sums.values()):
        raise ValueError('Prior complete-suite accounting differs or exceeds cap')
    observations = list(observations)
    indexed = {}
    for obs in observations:
        key = _key(obs['region_key'])
        if key not in old or key in indexed:
            raise ValueError('Foreign or duplicate region observation')
        _binding(obs.get('source_binding'))
        indexed[key] = obs
    diagnostics = list(diagnostic_keys)
    if len(set(diagnostics)) != len(diagnostics) or not set(diagnostics) <= set(indexed):
        raise ValueError('Explicit unique diagnostic keys need authenticated observations')
    diagnostic_rank = {k: i for i, k in enumerate(diagnostics)}
    rows = copy.deepcopy(old); decisions = []; candidates = []; measured = {}
    for key in sorted(old):
        row = rows[key]; obs = indexed.get(key)
        info, reason = _measured(obs) if obs is not None else (None, 'NO_AUTHENTICATED_OBSERVATION')
        decision = dict(region_key=key, previous_quota_bytes=row['raw_quota_bytes'],
                        reason=reason, proposed_multiplier=None)
        if info:
            measured[key] = info
            decision.update(retained_model_floor_bytes=info['floor'],
                selected_model_sha256=info['selected']['model_sha256'],
                latest_model_sha256=info['latest']['model_sha256'],
                source_binding=_binding(obs['source_binding']),
                train_wmape_percent=obs['train_wmape_percent'],
                qualified=info['qualified'])
            if info['qualified']:
                row['raw_quota_bytes'] = info['floor']
                decision['reason'] = 'QUALIFIED_RETAIN_SELECTED_AND_LATEST_WITH_10_PERCENT_RESERVE'
            else:
                row['raw_quota_bytes'] = max(row['raw_quota_bytes'], info['floor'])
                decision['reason'] = 'UNRESOLVED_ALLOCATION_RETAINED'
                permitted = True
                if multiplier == 4:
                    trial = obs.get('twofold_trial') or {}
                    permitted = (old[key].get('allocation_multiplier') == 2
                        and trial.get('allocation_sha256') == prior_binding['sha256']
                        and trial.get('status') in {'FITTED', 'RESOURCE_REJECTED', 'NO_HEADROOM'})
                    if permitted:
                        _binding(trial.get('result_binding'))
                        updates = _integer(trial.get('new_boosting_rounds'), 'trial actual rounds')
                        if trial['status'] == 'FITTED' and updates < 1:
                            raise ValueError('A zero-tree trial cannot claim fitting')
                    else:
                        decision['reason'] = 'FOURFOLD_REQUIRES_BOUND_TWOFOLD_TRIAL'
                target = min(REGION_CAP, multiplier*row['baseline_raw_quota_bytes'])
                increment = obs.get('requested_round_increment', 8)
                if type(increment) is not int or increment not in (8, 16, 32):
                    raise ValueError('Only approved 8/16/32 round increments allowed')
                current_room = headroom(info['latest'], row['raw_quota_bytes'], info['train_id'], increment=increment)
                prospective = headroom(info['latest'], target, info['train_id'], increment=increment)
                decision['requested_quota_bytes'] = target
                decision['requested_round_increment'] = increment
                decision['latest_requested_headroom'] = prospective
                if permitted and target > row['raw_quota_bytes']:
                    if current_room['status'] in {'NEW_DEPTH', 'CHANGED_TRAIN_RESTART'} or current_room['append_rounds'] == increment:
                        decision['reason'] = 'EXISTING_QUOTA_HAS_REQUESTED_HEADROOM'
                    elif prospective['status'] in {'NO_NEW_TREE', 'ZERO_ROUND_ANCHOR', 'NO_ANCHOR'}:
                        decision['reason'] = 'NO_NEW_TREE_AT_REQUESTED_QUOTA'
                    else:
                        candidates.append((key, target, decision))
        decisions.append(decision)
    sums = {sid: sum(r['raw_quota_bytes'] for k, r in rows.items() if k.startswith(sid+':'))
            for sid in SUITES}
    if any(x > SUITE_CAP for x in sums.values()):
        raise ValueError('Protected retained-model allocations exceed unchanged suite cap')
    candidates.sort(key=lambda item: (item[0] not in diagnostic_rank,
        diagnostic_rank.get(item[0], 0), indexed[item[0]]['train_wmape_percent'], item[0]))
    for rank, (key, target, decision) in enumerate(candidates, 1):
        sid = key.split(':')[0]; needed = target-rows[key]['raw_quota_bytes']
        decision.update(priority_rank=rank, diagnostic_reserved=key in diagnostic_rank)
        if sums[sid]+needed <= SUITE_CAP:
            rows[key]['raw_quota_bytes'] = target
            rows[key]['allocation_multiplier'] = multiplier
            sums[sid] += needed
            decision.update(reason='GRANTED', proposed_multiplier=multiplier)
        else:
            decision['reason'] = 'INSUFFICIENT_UNUSED_SAME_SUITE_CAPACITY'
    for decision in decisions:
        key = decision['region_key']; quota = rows[key]['raw_quota_bytes']
        decision['new_quota_bytes'] = quota
        if key in measured:
            info = measured[key]
            decision['selected_headroom'] = headroom(info['selected'], quota, info['train_id'])
            decision['latest_headroom'] = headroom(info['latest'], quota, info['train_id'])
    canonical_observations = json.dumps(observations, sort_keys=True, separators=(',', ':'), allow_nan=False)
    manifest = dict(schema='RECOVERY_XGB_COMPLETE_SUITE_QUOTAS_V1', allocator_schema=SCHEMA,
        source_budget_path=prior['source_budget_path'], source_budget_sha256=prior['source_budget_sha256'],
        per_suite_cap_bytes=SUITE_CAP, resident_multiplier=3., suite_sums=sums,
        rows=[rows[key] for key in sorted(rows)], prior_allocation=prior_binding,
        requested_multiplier=multiplier, diagnostic_keys=diagnostics,
        observation_snapshot_sha256=hashlib.sha256(canonical_observations.encode()).hexdigest(),
        observation_count=len(observations), decisions=decisions,
        unused_suite_capacity={sid: SUITE_CAP-sums[sid] for sid in SUITES},
        policy='Measured current-qualified donors only; protect selected/latest raw bytes plus 10%; '
               'explicit diagnostics first, then smallest unmet native TRAIN WMAPE; '
               '2x/4x original quotas, unchanged 128 rounds/depth 3,5,7; no cross-suite transfer.',
        actual_models_loaded=False, production_config_applied=False, scientific_success_claim=False)
    return manifest, {row['region_key']: row['raw_quota_bytes'] for row in manifest['rows']}
