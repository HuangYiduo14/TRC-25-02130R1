"""Fixed-cohort recovery in a new run, reusing immutable PACNHN model primitives.

No discovery or execution at import. The caller supplies the authenticated new
Context, explicit regions and deadline. TRAIN/VALID are the only readable roles.
An old checkpoint is a read dependency, never a writable continuation root.
"""
from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
import traceback
import uuid

import numpy as np
import torch

from ..pacnhn_v1 import training as legacy
from ..pacnhn_v1.control import array_hash, read_json, sha, write_json, _atomic_replace
from ..pacnhn_v1.models import (MonotoneICNN, Power4Model, PortableICNN,
    ICNN_SCALED, fit_power4_cost, load_model, save_model, update_icnn)
from ..pacnhn_v1.regional import load_region_graph

SCHEMA = "PACNHN_RECOVERY_TRAINING_V1"
FAMILIES = ("ICNN", "XGBOOST")
WIDTHS = (32, 64, 128)
RATES = (.001, .0003, .0001)


def _new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    write_json(path, value)


def _roots(ctx):
    new, old = Path(ctx.root).resolve(), Path(ctx.source_root).resolve()
    if new == old or new.is_relative_to(old) or old.is_relative_to(new):
        raise ValueError("Recovery output and immutable source must be disjoint roots")
    return new, old


def require_training_open(ctx):
    new, _ = _roots(ctx)
    for marker in (new/'training'/'MODEL_SELECTION_FREEZE.json',
                   new/'training'/'REGISTRY_FREEZE.json', new/'holdout'/'TEST_OPENED.json'):
        if marker.exists():
            raise PermissionError('Recovery training is sealed: '+str(marker))


def _read_path(ctx, path):
    new, old = _roots(ctx)
    path = Path(path).resolve(strict=True)
    if not (path.is_relative_to(old) or path.is_relative_to(new)):
        raise ValueError("Scientific input escapes the explicitly bound old/new run")
    return path


def _bound(ctx, path, role, expected=None):
    path = _read_path(ctx, path)
    if (path.is_relative_to(Path(ctx.source_root).resolve())
            and path.name in {'REQUEST_INDEX.json', 'REGION_IDENTITY.json', 'STATE.json', 'NATIVE_SCALE.json'}
            and hasattr(ctx, 'source_expected')
            and os.path.normcase(str(path)) not in ctx.source_expected):
        raise ValueError('Old regional control metadata lacks a checkpoint-authenticated hash pin')
    digest = ctx.bind(path, role, expected)
    return path, digest


def region_key(record):
    sid, index = record['suite_id'], record['region_index']
    if sid not in {'s02', 's03', 's04', 's05', 's06'} or type(index) is not int or index < 0:
        raise ValueError("Invalid recovery region")
    return f'{sid}:{index:06d}'


def region_root(ctx, record, *, source=False):
    new, old = _roots(ctx)
    region_key(record)
    base = old if source else new
    result = base/'training'/'regions'/record['suite_id']/f"{record['region_index']:06d}"
    if not result.resolve().is_relative_to(base):
        raise ValueError('Regional path alias escapes its owned root')
    return result


def settings(ctx):
    """The old mathematical configuration is explicit; recovery changes scheduling."""
    r = dict(ctx.c.get('recovery', {}))
    path = r.get('training_config_path')
    digest = r.get('training_config_sha256')
    if not path or not digest:
        raise ValueError("Explicit source training config path/hash required")
    ctx.bind(path, 'IMMUTABLE_TRAINING_CONFIG', digest)
    c = read_json(path)
    if (c.get('train_target_wmape_percent') != 1. or tuple(c.get('icnn_widths', ())) != WIDTHS
            or c.get('icnn_coefficient_parameterization') != ICNN_SCALED
            or c.get('icnn_updates_per_block') != 250):
        raise ValueError("Unchanged ICNN definition, widths, native target and 250-update blocks required")
    rates = tuple(r.get('icnn_learning_rates', RATES))
    if rates != RATES:
        raise ValueError("Only the approved learning-rate ladder is supported")
    result = dict(c)
    result.update(recovery_initial_blocks=int(r.get('initial_icnn_blocks', 4)),
        recovery_extra_blocks=int(r.get('extra_icnn_blocks', 4)),
        recovery_min_gain_pp=float(r.get('min_gain_absolute_pp', .01)),
        recovery_min_gain_relative=float(r.get('min_gain_relative', .01)),
        recovery_xgb_blocks=int(r.get('xgb_blocks_per_visit', 1)),
        recovery_region_seconds=float(r.get('region_seconds', 300.)),
        recovery_learning_rates=rates,
        recovery_valid_degradation_pp=float(r.get('valid_degradation_absolute_pp', .02)),
        recovery_valid_degradation_relative=float(r.get('valid_degradation_relative', .02)),
        recovery_family_seconds=float(r.get('maximum_region_worker_seconds', 1200)),
        recovery_xgb_round_increments=tuple(r.get('xgb_round_increments', (8, 16, 32))))
    if (not 1 <= result['recovery_initial_blocks'] <= 4 or not 0 <= result['recovery_extra_blocks'] <= 4
            or not 0 <= result['recovery_xgb_blocks'] <= 4
            or not 0 < result['recovery_region_seconds'] <= 1800
            or not 0 < result['recovery_family_seconds'] <= 1200
            or result['recovery_xgb_round_increments'] != (8, 16, 32)
            or any(not math.isfinite(result[k]) or result[k] < 0 for k in
                ('recovery_min_gain_pp', 'recovery_min_gain_relative',
                 'recovery_valid_degradation_pp', 'recovery_valid_degradation_relative'))):
        raise ValueError("Invalid bounded recovery scheduling settings")
    quota_manifest = r.get('xgb_quota_manifest')
    if quota_manifest:
        path, _ = _bound(ctx, quota_manifest['path'], 'COMPLETE_XGB_SUITE_ALLOCATION', quota_manifest['sha256'])
        allocation = read_json(path)
        prior_path, _ = _bound(ctx, allocation['source_budget_path'], 'ORIGINAL_XGB_SUITE_BUDGET', allocation['source_budget_sha256'])
        original = read_json(prior_path)
        cap = allocation['per_suite_cap_bytes']
        if cap != original['per_suite_raw_booster_bytes']:
            raise ValueError('XGB suite quota amendment cannot raise the original complete-suite cap')
        rows = allocation['rows']; mapping = {row['region_key']: row['raw_quota_bytes'] for row in rows}
        if len(mapping) != len(rows) or mapping != r.get('xgb_region_quotas'):
            raise ValueError('XGB config differs from its complete allocation manifest')
        sums = {}
        for key, quota in mapping.items():
            if type(quota) is not int or not 0 < quota <= c['xgb_max_projected_model_bytes']:
                raise ValueError('Invalid individual XGB quota')
            sid = key.split(':')[0]; sums[sid] = sums.get(sid, 0)+quota
        if sums != allocation['suite_sums'] or any(value > cap for value in sums.values()):
            raise ValueError('XGB allocation exceeds the unchanged complete-suite budget')
    return result


def _selected_entries(ctx, root, region, role, check):
    """Authenticate all role/input/receipt bindings; never open blind TEST contents."""
    if role not in {'TRAIN', 'VALID'}:
        raise PermissionError("Training recovery cannot open TEST")
    identity_path, _ = _bound(ctx, root/'REGION_IDENTITY.json', 'REGIONAL_LABEL_IDENTITY')
    identity = read_json(identity_path)
    if (identity['graph_sha256'] != region.graph_sha256 or identity['graph_identity'] != region.identity
            or identity['support_identity'] != region.support_identity):
        raise ValueError("Training label graph/support identity mismatch")
    index_path, index_sha = _bound(ctx, root/'REQUEST_INDEX.json', 'TRAIN_VALID_REQUEST_INDEX')
    index = read_json(index_path)
    for input_id, row in index.items():
        if row['role'] != role or row.get('selected_label') is None:
            continue
        check()
        canonical = hashlib.sha256(f'{region.identity}|{role}|{input_id}'.encode()).hexdigest()
        if row['sample_id'] != canonical:
            raise ValueError("Canonical sample identity mismatch")
        request_path, _ = _bound(ctx, row['request_metadata'], 'FROZEN_'+role+'_REQUEST', row['request_metadata_sha256'])
        _bound(ctx, row['input_file'], 'FROZEN_'+role+'_INPUT', row['input_file_sha256'])
        request = read_json(request_path)
        expected = dict(sample_id=canonical, role=role, input_id=input_id,
            graph_identity=region.identity, support_identity=region.support_identity,
            input_file_sha256=row['input_file_sha256'])
        if any(request.get(k) != v for k, v in expected.items()):
            raise ValueError("Frozen request metadata does not bind the selected role/input")
        selected = row['selected_label']
        label_path, _ = _bound(ctx, selected['path'], 'ADMITTED_'+role+'_LABEL', selected['sha256'])
        receipt_path, _ = _bound(ctx, selected['receipt'], 'ADMITTED_'+role+'_RECEIPT', selected['receipt_sha256'])
        receipt = read_json(receipt_path)
        required = dict(sample_id=canonical, role=role, input_identity=input_id,
            graph_identity=region.identity, graph_sha256=region.graph_sha256,
            support_identity=region.support_identity, input_file_sha256=row['input_file_sha256'],
            label_sha256=selected['sha256'], admitted=True, fresh_certificate=True)
        if any(receipt.get(k) != v for k, v in required.items()) or receipt.get('quality') not in {'A', 'B'}:
            raise ValueError("Selected label admission/identity changed")
        # Discard potentially large iteration histories before retaining this row.
        compact = {k: receipt.get(k) for k in ('sample_id', 'role', 'input_identity', 'quality',
            'graph_identity', 'support_identity', 'reported_state_iteration', 'admitted')}
        compact.update(label_path=str(label_path), label_sha256=selected['sha256'],
            receipt_path=str(receipt_path), receipt_sha256=selected['receipt_sha256'])
        del receipt
        yield input_id, label_path, compact, identity, index_sha
    if sha(index_path) != index_sha:
        raise ValueError("Dataset index changed during loading")


def _load_role(ctx, root, region, role, check):
    inputs, costs, records = [], [], []
    c0 = identity = index_sha = None
    for input_id, path, record, identity, index_sha in _selected_entries(ctx, root, region, role, check):
        with np.load(path, allow_pickle=False) as z:
            q, tau, baseline = z['u'].copy(), z['tau'].copy(), z['c0'].copy()
        if (q.shape != (region.m,) or tau.shape != q.shape or baseline.shape != q.shape
                or not all(np.isfinite(a).all() for a in (q, tau, baseline))
                or any(np.any(a < 0) for a in (q, tau, baseline))
                or array_hash({'u': q}) != input_id or not np.allclose(baseline, region.c0, rtol=1e-12, atol=1e-10)
                or (c0 is not None and not np.array_equal(c0, baseline))):
            raise ValueError("Nonfinite, changed or mismatched full native label")
        c0 = baseline; inputs.append(q); costs.append(tau); records.append(record)
    if not inputs:
        raise ValueError('No admitted labels in requested role: '+role)
    u, tau = np.stack(inputs), np.stack(costs)
    return dict(u=u, tau=tau, c0=c0, records=records, identity=identity, role=role,
        dataset_identity=array_hash({'u': u, 'tau': tau, 'c0': c0}), unique_rows=len(u),
        train_cost_mean=float(tau.mean()) if role == 'TRAIN' else None, request_index_sha256=index_sha)


def load_training_cohorts(ctx, record, *, labels_root=None, region=None, deadline=None,
                          read_only=False, selection_freeze_path=None, selection_freeze_sha256=None):
    """Read current cohorts under fitting or explicit post-freeze authority.

    Evaluation may read TRAIN/VALID after the fitting deadline, but only under
    the separately admitted holdout clock and immutable global selection. This
    mode grants no fitting, label generation or reselection authority.
    """
    if type(read_only) is not bool:raise ValueError('Explicit boolean read-only mode required')
    if read_only:
        from .labels import require_selection_freeze
        canonical=Path(ctx.root)/'training/MODEL_SELECTION_FREEZE.json'
        if selection_freeze_path is None or Path(selection_freeze_path).resolve()!=canonical.resolve():
            raise PermissionError('Read-only cohorts require the canonical selection freeze')
        if not isinstance(selection_freeze_sha256,str) or len(selection_freeze_sha256)!=64:
            raise PermissionError('Read-only cohorts require an explicit selection-freeze hash')
        ctx.require_action('holdout');ctx.guard(deadline=deadline)
        frozen,_=require_selection_freeze(ctx,canonical,selection_freeze_sha256)
        opening=Path(ctx.root)/'holdout/TEST_OPENED.json'
        ctx.bind(opening,'POSTFREEZE_READONLY_COHORT_AUTHORITY')
        opened=read_json(opening)
        if (opened.get('selection_freeze_sha256')!=selection_freeze_sha256
                or opened.get('epoch',-math.inf)<frozen['created_epoch']):
            raise PermissionError('Read-only cohorts require the matching post-selection TEST opening')
    else:
        if selection_freeze_path is not None or selection_freeze_sha256 is not None:
            raise ValueError('Selection-freeze authority is only valid in explicit read-only mode')
        ctx.require_action('training')
    check = lambda: ctx.guard(deadline=deadline)
    check()
    graph, _ = _bound(ctx, record['path'], 'FROZEN_REGIONAL_GRAPH', record['sha256'])
    region = region or load_region_graph(graph, expected_sha256=record['sha256'])
    if labels_root is None:
        overlay = ctx.c.get('recovery', {}).get('training_overlays', {}).get(region_key(record))
        if overlay:
            root = _read_path(ctx, overlay['path'])
            _bound(ctx, root/'REQUEST_INDEX.json', 'SEALED_RECOVERY_DATA_OVERLAY', overlay['sha256'])
        else:
            root = region_root(ctx, record, source=True)/'labels'
    else:
        root = _read_path(ctx, labels_root)
    train = _load_role(ctx, root, region, 'TRAIN', check)
    valid = _load_role(ctx, root, region, 'VALID', check)
    if train['request_index_sha256'] != valid['request_index_sha256']:
        raise ValueError("TRAIN/VALID were not loaded from one frozen index")
    return train, valid, region


def _initial_state(ctx, record, target):
    if (target/'STATE.json').exists():
        state = read_json(target/'STATE.json')
        if state.get('recovery_schema') != SCHEMA or state.get('recovery_region_key') != region_key(record):
            raise ValueError('Foreign recovery state')
        return state
    old = region_root(ctx, record, source=True)/'STATE.json'
    state = dict(visits=0, dataset_identity=None, eligible={f: [] for f in FAMILIES},
        selected={}, optimizers={}, blocks_by_width={}, xgb_latest=None,
        candidate_counter=0, retired=[], failures=[])
    if old.exists():
        _, digest = _bound(ctx, old, 'INHERITED_REGION_STATE')
        prior = read_json(old)
        for key in ('selected', 'eligible', 'optimizers', 'blocks_by_width', 'candidate_counter', 'xgb_latest'):
            state[key] = copy.deepcopy(prior.get(key, state.get(key)))
        state['inherited_state'] = dict(path=str(old), sha256=digest)
        selected = state['selected'].get('ICNN')
        if selected:
            checkpoint = Path(selected['model_path']).parent/'CHECKPOINT.json'
            _, digest = _bound(ctx, checkpoint, 'SELECTED_OPTIMIZER_METADATA')
            saved = read_json(checkpoint)
            if (saved.get('candidate_id') != selected['candidate_id'] or saved.get('model_sha256') != selected['model_sha256']
                    or saved.get('model_path') != selected['model_path']):
                raise ValueError('Selected optimizer metadata model mismatch')
            optimizer = saved['optimizer']
            if (optimizer['candidate_id'] != selected['candidate_id']
                    or Path(optimizer['path']).resolve() != (checkpoint.parent/'optimizer.pt').resolve()):
                raise ValueError('Optimizer belongs to another candidate')
            state['optimizers'][str(selected['width'])] = copy.deepcopy(optimizer)
            state['selected_optimizer_origin'] = dict(path=str(checkpoint), sha256=digest)
    state.update(recovery_schema=SCHEMA, recovery_region_key=region_key(record),
        recovery_visits=0, recovery_blocks=0, recovery_failures=[], recovery_history=[],
        inherited_candidates_preserved=True, TEST_opened=False)
    return state


def target_met(candidate, train_id, valid_id):
    audit = (candidate or {}).get('audit', {})
    value = audit.get('TRAIN', {}).get('WMAPE_percent')
    other = audit.get('VALID', {}).get('WMAPE_percent')
    return (audit.get('train_dataset_identity') == train_id and audit.get('valid_dataset_identity') == valid_id
        and type(value) in (int, float) and math.isfinite(value) and 0 <= value < 1.
        and type(other) in (int, float) and math.isfinite(other) and other >= 0)


def _validate_audit(candidate, train, valid):
    audit = candidate.get('audit', {})
    if audit.get('train_dataset_identity') != train['dataset_identity'] or audit.get('valid_dataset_identity') != valid['dataset_identity']:
        raise ValueError('Selected audit is not current TRAIN/VALID')
    for role in ('TRAIN', 'VALID'):
        metric = audit[role]['WMAPE_percent']
        if type(metric) not in (int, float) or not math.isfinite(metric) or metric < 0:
            raise ValueError('Invalid native cost WMAPE')
    if bool(audit['target_met']) != (audit['TRAIN']['WMAPE_percent'] < 1.):
        raise ValueError('Selected target flag contradicts strict native TRAIN target')


def _audit_select(ctx, root, family, state, train, valid, config, check):
    for candidate in state['eligible'][family]:
        check()
        _bound(ctx, candidate['model_path'], 'ELIGIBLE_'+family+'_MODEL', candidate['model_sha256'])
    elapsed = legacy.audit_and_select(root, family, state, train, valid, config, check)
    if family in state['selected']:
        _validate_audit(state['selected'][family], train, valid)
    return elapsed


def _torch_save(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with tmp.open('xb') as stream:
        torch.save(value, stream)
        stream.flush(); os.fsync(stream.fileno())
    _atomic_replace(tmp, path)


def icnn_block(ctx, root, state, train, scale, config, *, width, learning_rate, deadline, fresh=False):
    """Exact optimizer continuation, with explicit post-restore LR and no unit reset."""
    if width not in WIDTHS or learning_rate not in RATES:
        raise ValueError('Unapproved width or learning rate')
    require_training_open(ctx)
    key = str(width); previous = None if fresh else state['optimizers'].get(key)
    check = lambda: ctx.guard(deadline=deadline)
    check(); started = time.perf_counter()
    model = (MonotoneICNN(train['c0'], scale, width=width, seed=config['seed'], device='cpu')
        if previous else legacy._initialize_icnn(train, scale, config, 'cpu', width))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(config['seed']+width)
    total_updates = 0
    if previous:
        path, _ = _bound(ctx, previous['path'], 'RESUMED_ICNN_OPTIMIZER', previous['sha256'])
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (saved.get('schema') != legacy.OPTIMIZER_SCHEMA or saved.get('coefficient_parameterization') != ICNN_SCALED
                or saved['width'] != width or not np.array_equal(saved['scale'].numpy(), scale)
                or not np.array_equal(saved['model']['c0'].numpy(), train['c0'])):
            raise ValueError('Optimizer architecture/native units incompatible; no implicit migration')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        model.coefficient_parameterization = ICNN_SCALED
        model.calibration = saved.get('TRAIN_calibration')
        generator.set_state(saved['generator']); total_updates = int(saved['total_updates'])
        if total_updates <= 0:
            raise ValueError('Initialization-only optimizer cannot be presented as a resumed fit')
        # Saved Adam param_groups contain its previous LR: apply the policy AFTER restore.
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        model_path = Path(previous['path']).parent/'model.npz'
        checkpoint_path, _ = _bound(ctx, Path(previous['path']).parent/'CHECKPOINT.json', 'RESUMED_ICNN_CHECKPOINT')
        checkpoint = read_json(checkpoint_path)
        if (checkpoint.get('optimizer', {}).get('sha256') != previous['sha256']
                or checkpoint.get('candidate_id') != previous['candidate_id']):
            raise ValueError('Optimizer/checkpoint provenance mismatch')
        _bound(ctx, model_path, 'RESUMED_ICNN_MODEL', checkpoint['model_sha256'])
        portable = load_model(model_path)
        exported = portable.export_arrays()
        if not isinstance(portable, PortableICNN) or any(not np.array_equal(value, exported[name])
                for name, value in model.export_arrays().items()):
            raise ValueError('Optimizer parameters differ from their immutable inference checkpoint')
    updates = 0; last = None
    for _ in range(config['icnn_updates_per_block']):
        if time.time() >= deadline:
            break
        check()
        indices = torch.randint(len(train['u']), (min(config['icnn_batch_rows'], len(train['u'])),), generator=generator).numpy()
        last = update_icnn(model, optimizer, train['u'][indices], train['tau'][indices], train_cost_mean=train['train_cost_mean'])
        updates += 1
    if not updates:
        return dict(status='TIME_LIMIT_BEFORE_UPDATE', seconds=time.perf_counter()-started, updates=0)
    candidate, directory = legacy._candidate(root, state, 'ICNN', model, width=width,
        updates_in_block=updates, total_updates=total_updates+updates, device='cpu',
        trained_dataset_identity=train['dataset_identity'], last_batch=last,
        training_block_index=sum(state['blocks_by_width'].values())+1,
        coefficient_parameterization=ICNN_SCALED, TRAIN_calibration=model.calibration,
        recovery_learning_rate=learning_rate, optimizer_source=copy.deepcopy(previous),
        recovery_fresh_start=previous is None)
    state_path = directory/'optimizer.pt'
    _torch_save(state_path, dict(schema=legacy.OPTIMIZER_SCHEMA, coefficient_parameterization=ICNN_SCALED,
        TRAIN_calibration=model.calibration, model=model.state_dict(), optimizer=optimizer.state_dict(),
        generator=generator.get_state(), width=width, scale=torch.tensor(scale), total_updates=total_updates+updates))
    state['optimizers'][key] = dict(path=str(state_path), sha256=sha(state_path), candidate_id=candidate['candidate_id'])
    state['blocks_by_width'][key] = state['blocks_by_width'].get(key, 0)+1
    state['recovery_blocks'] = state.get('recovery_blocks', 0)+1
    _new(directory/'CHECKPOINT.json', dict(candidate, optimizer=state['optimizers'][key]))
    return dict(status='UPDATED', seconds=time.perf_counter()-started, updates=updates,
        width=width, learning_rate=learning_rate, candidate_id=candidate['candidate_id'],
        resume=previous is not None, total_updates=total_updates+updates)


def _normal_record(record, region, candidate, train, valid, family, *, analytic=False):
    audit = candidate['audit']
    _validate_audit(candidate, train, valid)
    digest = candidate['model_sha256']
    return dict(suite_id=record['suite_id'], region_index=record['region_index'], family=family,
        status='AVAILABLE', model_path=candidate['model_path'], model_sha256=digest,
        artifact_hashes={candidate['model_path']: digest}, model_kind=candidate['model_kind'],
        pair_keys_identity=array_hash({'pair_keys': region.pair_keys}), graph_sha256=region.graph_sha256,
        pairs=len(region.pair_keys),
        c0_identity=array_hash({'c0': train['c0']}), train_rows=train['unique_rows'], valid_rows=valid['unique_rows'],
        train_wmape_percent=audit['TRAIN']['WMAPE_percent'], valid_wmape_percent=audit['VALID']['WMAPE_percent'],
        current_train_dataset_identity=train['dataset_identity'], current_valid_dataset_identity=valid['dataset_identity'],
        audit_train_dataset_identity=audit['train_dataset_identity'], audit_valid_dataset_identity=audit['valid_dataset_identity'],
        audit_model_sha256=digest, target_met=audit['TRAIN']['WMAPE_percent'] < 1.,
        fit_sanity_status='PASS', fit_sanity_scope='HASH_BOUND_FINITE_FULL_NATIVE_TRAIN_VALID_METRICS',
        property_status='PASS' if analytic else 'PENDING', property_model_sha256=digest if analytic else None,
        property_scope='EXACT_NONNEGATIVE_ANALYTIC_POWER4_COEFFICIENTS' if analytic else 'REQUIRES_INDEPENDENT_PROPERTY_AUDIT',
        candidate_id=candidate.get('candidate_id'), width=candidate.get('width'), total_updates=candidate.get('total_updates'),
        uncompressed_model_bytes=candidate.get('uncompressed_model_bytes'),
        uncompressed_booster_bytes=candidate.get('uncompressed_booster_bytes'),
        native_parameter_bytes=candidate.get('native_parameter_bytes'),
        forecast_resident_booster_bytes=(3*int(candidate['uncompressed_model_bytes'])
            if family == 'XGBOOST' and candidate.get('uncompressed_model_bytes') is not None else None),
        memory_forecast_basis='THREE_TIMES_RAW_UBJ_PLUS_NATIVE_ARRAY_BYTES' if family == 'XGBOOST' else None,
        neural_branch=audit.get('neural_branch'), audit=audit, TEST_opened=False)


def _baselines(ctx, root, record, region, state, train, valid, scale, config, check):
    records = []
    for family, model in (('POWER4', fit_power4_cost(train['u'], train['tau'], train['c0'], scale)),
            ('FFTT', Power4Model(train['c0'], np.ones(region.m), np.zeros(region.m)))):
        check()
        path = root/'baselines'/family/(train['dataset_identity']+'.npz')
        if path.exists():
            old = load_model(path)
            if any(not np.array_equal(a, old.export_arrays()[k]) for k, a in model.export_arrays().items()):
                raise ValueError('Immutable analytic model mismatch')
        else:
            save_model(model, path, metadata={'trained_dataset_identity': train['dataset_identity'], 'recovery_family': family})
        candidate = dict(model_path=str(path), model_sha256=sha(path), model_kind=model.kind,
            audit=dict(train_dataset_identity=train['dataset_identity'], valid_dataset_identity=valid['dataset_identity'],
                TRAIN=legacy._evaluate(model, train, config['audit_batch_rows'], check),
                VALID=legacy._evaluate(model, valid, config['audit_batch_rows'], check)))
        candidate['audit']['target_met'] = candidate['audit']['TRAIN']['WMAPE_percent'] < 1.
        records.append(_normal_record(record, region, candidate, train, valid, family, analytic=True))
        state[family.lower()] = candidate
    return records


def _useful_gain(before, after, config):
    return before-after >= max(config['recovery_min_gain_pp'], abs(before)*config['recovery_min_gain_relative'])


def tranche_progress(before, after, config):
    """Spending rule only; never changes target qualification or candidate selection."""
    if before is None:
        return dict(productive=False, reason='FIRST_FIT_WITHOUT_GAIN_BASELINE', train_gain_pp=None, valid_degradation_pp=None)
    gain = before[0]-after[0]
    degradation = after[1]-before[1]
    useful = _useful_gain(before[0], after[0], config)
    valid = degradation <= max(config['recovery_valid_degradation_pp'], abs(before[1])*config['recovery_valid_degradation_relative'])
    return dict(productive=bool(useful and valid), reason='USEFUL_GAIN_WITH_VALID_GUARD' if useful and valid else 'UNPRODUCTIVE_TRANCHE',
        train_gain_pp=gain, valid_degradation_pp=degradation)


def _metrics(candidate):
    if not candidate:
        return None
    return (candidate['audit']['TRAIN']['WMAPE_percent'], candidate['audit']['VALID']['WMAPE_percent'])


def record_cohort_epoch(state, train_id, valid_id, index_sha256):
    """New data gets a new comparison epoch, not refunded work or failure budgets."""
    identity = (train_id, valid_id)
    epochs = state.setdefault('recovery_cohort_epochs', [])
    if epochs and (epochs[-1]['train_dataset_identity'], epochs[-1]['valid_dataset_identity']) == identity:
        if epochs[-1]['request_index_sha256'] != index_sha256:
            # Request-only changes do not buy a new optimization allowance.
            epochs[-1].setdefault('equivalent_population_index_versions', []).append(index_sha256)
        return False
    histories = {}
    for family, policy in state.get('recovery_family_policy', {}).items():
        histories[family] = copy.deepcopy(policy)
        policy['unproductive_tranches'] = 0
        policy['cohort_tranches'] = 0
        policy.pop('base_width', None)
        policy.pop('inactive_diagnostic', None)
        # elapsed_seconds, failures/invariant_failure, evaluated_tranches and
        # fresh_initialization_used deliberately remain cumulative.
    epochs.append(dict(epoch_number=len(epochs)+1, recorded_epoch=time.time(),
        train_dataset_identity=train_id, valid_dataset_identity=valid_id, request_index_sha256=index_sha256,
        prior_family_policies=histories, cumulative_budgets_preserved=True,
        comparison_reset_reason='INITIAL_FIXED_COHORT' if not epochs else 'EXPLICIT_NEW_DATASET_IDENTITY'))
    return True


def amplitude_boundary_diagnostic(model, train, batch, check):
    """One-sided native MAE derivative at Gamma=0; diagnostic, never a floor."""
    if not isinstance(model, PortableICNN) or model.amplitude != 0:
        return dict(applicable=False, reason='NOT_ZERO_AMPLITUDE', fresh_start_permitted=False)
    slope = 0.; basis = 0.
    for offset in range(0, len(train['u']), batch):
        check()
        u = train['u'][offset:offset+batch]; y = train['tau'][offset:offset+batch]
        residual = model.cost(u)-y; direction = model.unit_neural_cost(u)
        if not np.isfinite(direction).all() or np.any(direction < 0):
            raise FloatingPointError('Invalid zero-amplitude diagnostic direction')
        slope += float(np.where(residual > 0, direction, np.where(residual < 0, -direction, np.abs(direction))).sum())
        basis += float(direction.sum())
    return dict(applicable=True, one_sided_native_L1_derivative=slope, unit_neural_increment_sum=basis,
        fresh_start_permitted=bool(basis > 0), positive_amplitude_is_local_descent=bool(slope < 0),
        interpretation='Diagnostic at this saved shape only; fresh initialization has independent TRAIN calibration; no amplitude floor')


def _transactional_block(ctx, root, family, state, train, valid, config, check, operation):
    """Publish eligibility only after a full current-cohort audit; keep failed files."""
    keys = ('selected', 'eligible', 'optimizers', 'blocks_by_width', 'xgb_latest', 'xgb_blocks', 'recovery_blocks', 'retired')
    before = {k: copy.deepcopy(state.get(k)) for k in keys}
    try:
        outcome = operation()
        _audit_select(ctx, root, family, state, train, valid, config, check)
        return outcome
    except Exception:
        for key, value in before.items():
            if value is None:
                state.pop(key, None)
            else:
                state[key] = value
        # candidate_counter is deliberately not rolled back; immutable files stay.
        raise


def _fit_family(ctx, target, record, region, state, train, valid, scale, config, family, end, blocks):
    require_training_open(ctx)
    policy = state.setdefault('recovery_family_policy', {}).setdefault(family,
        dict(elapsed_seconds=0., unproductive_tranches=0, evaluated_tranches=0, failures=0))
    if target_met(state['selected'].get(family), train['dataset_identity'], valid['dataset_identity']):
        return [dict(family=family, status='PRESERVED_TARGET_MET', updates=0, rounds=0)]
    if policy['elapsed_seconds'] >= config['recovery_family_seconds'] or policy['unproductive_tranches'] >= 2:
        return [dict(family=family, status='STOPPED_SPENDING_RULE', policy=copy.deepcopy(policy), updates=0, rounds=0)]
    if policy['failures'] >= 2 or policy.get('invariant_failure'):
        return [dict(family=family, status='STOPPED_REPEATED_OR_INVARIANT_FAILURE', updates=0, rounds=0)]
    begin = time.perf_counter()
    family_end = min(end, time.time()+config['recovery_family_seconds']-policy['elapsed_seconds'])
    check = lambda: ctx.guard(deadline=family_end)
    outcomes = []; before = _metrics(state['selected'].get(family)); committed_work = 0
    def record_progress():
        nonlocal before, committed_work
        after = _metrics(state['selected'].get(family))
        work = sum(int(x.get('updates', 0))+int(x.get('rounds', 0)) for x in outcomes)
        if after is None or work <= committed_work:
            return None
        progress = tranche_progress(before, after, config)
        policy['evaluated_tranches'] += 1
        policy['cohort_tranches'] = policy.get('cohort_tranches', 0)+1
        policy['unproductive_tranches'] = (0 if before is None or progress['productive'] else policy['unproductive_tranches']+1)
        outcomes.append(dict(family=family, status='TRANCHE_EVALUATED', **progress))
        before = after; committed_work = work
        return progress
    try:
        if family == 'ICNN':
            selected = state['selected'].get(family, {})
            phase = policy.get('cohort_tranches', policy['evaluated_tranches'])
            base_width = policy.setdefault('base_width', selected.get('width') or WIDTHS[0])
            width = WIDTHS[min(WIDTHS.index(base_width)+phase//3, len(WIDTHS)-1)]
            rate = RATES[phase % len(RATES)]
            count = config['recovery_initial_blocks'] if blocks is None else int(blocks)
            if not 1 <= count <= 8:
                raise ValueError('Explicit ICNN visit block count must be 1..8')
            fresh = False
            if selected and not policy.get('inactive_diagnostic'):
                model_path, _ = _bound(ctx, selected['model_path'], 'INACTIVE_BRANCH_DIAGNOSTIC', selected['model_sha256'])
                diagnostic = amplitude_boundary_diagnostic(load_model(model_path), train, config['audit_batch_rows'], check)
                policy['inactive_diagnostic'] = diagnostic
            # Use the compatible saved state first. A single later fresh attempt
            # is available after an unproductive observed tranche at Gamma zero.
            if (policy['unproductive_tranches'] and not policy.get('fresh_initialization_used')
                    and policy.get('inactive_diagnostic', {}).get('fresh_start_permitted')):
                fresh = True; policy['fresh_initialization_used'] = True
            limit = count
            for index in range(count+config['recovery_extra_blocks']):
                if index >= limit or time.time() >= family_end:
                    break
                operation = lambda: icnn_block(ctx, target, state, train, scale, config, width=width,
                    learning_rate=rate, deadline=family_end, fresh=fresh and index == 0)
                outcome = _transactional_block(ctx, target, family, state, train, valid, config, check, operation)
                outcomes.append(dict(family=family, **outcome))
                write_json(target/'STATE.json', state)
                if target_met(state['selected'].get(family), train['dataset_identity'], valid['dataset_identity']):
                    break
                if index+1 == count:
                    progress = record_progress()
                    if progress and progress['productive']:
                        limit += config['recovery_extra_blocks']
        else:
            quotas = ctx.c.get('recovery', {}).get('xgb_region_quotas', {})
            quota = record.get('xgb_raw_budget_bytes', quotas.get(region_key(record)))
            if type(quota) is not int or quota <= 0 or quota > config['xgb_max_projected_model_bytes']:
                raise ValueError('Explicit bounded per-region XGB suite allocation required')
            increment = config['recovery_xgb_round_increments'][min(policy.get('cohort_tranches', policy['evaluated_tranches']), 2)]
            xconfig = dict(config, xgb_max_projected_model_bytes=quota, xgb_rounds_per_block=increment)
            if state.get('xgb_latest'):
                row = state['xgb_latest']; _bound(ctx, row['model_path'], 'RESUMED_XGB_MODEL', row['model_sha256'])
            for _ in range(config['recovery_xgb_blocks']):
                check()
                operation = lambda: legacy._xgb_block(target, state, train, scale, region, xconfig, 'cpu', family_end, check)
                outcome = _transactional_block(ctx, target, family, state, train, valid, config, check, operation)
                outcomes.append(dict(family=family, **dict(outcome, status=outcome.get('status', 'UPDATED'))))
                write_json(target/'STATE.json', state)
                if not outcome.get('rounds') or target_met(state['selected'].get(family), train['dataset_identity'], valid['dataset_identity']):
                    break
        work = any(x.get('updates', 0) or x.get('rounds', 0) for x in outcomes)
        record_progress()
        if not work:
            outcomes.append(dict(family=family, status='NO_FITTING_PERFORMED', updates=0, rounds=0))
    except Exception as exc:
        policy['failures'] += 1
        policy['invariant_failure'] = isinstance(exc, (ValueError, ArithmeticError)) and not isinstance(exc, FloatingPointError)
        failure = dict(family=family, status='TIME_LIMIT' if isinstance(exc, TimeoutError) else
            ('NUMERICAL_FAILURE' if isinstance(exc, (FloatingPointError, ArithmeticError)) else 'STATE_OR_RUNTIME_FAILURE'),
            error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        state['recovery_failures'].append(failure); outcomes.append(failure)
    finally:
        policy['elapsed_seconds'] += time.perf_counter()-begin
    return outcomes


def run_region(ctx, record, *, deadline, blocks=None, labels_root=None, labels_sha256=None):
    """One bounded fixed-cohort visit. Failures preserve prior selected checkpoints."""
    ctx.require_action('training'); require_training_open(ctx); ctx.guard(deadline=deadline)
    config = settings(ctx); target = region_root(ctx, record)
    labels_root = labels_root if labels_root is not None else record.get('labels_root')
    labels_sha256 = labels_sha256 if labels_sha256 is not None else record.get('labels_sha256')
    if (labels_root is None) != (labels_sha256 is None):
        raise ValueError('Explicit new-cohort path and index SHA must be supplied together')
    if labels_root is not None:
        labels_root = _read_path(ctx, labels_root)
        _bound(ctx, labels_root/'REQUEST_INDEX.json', 'EXPLICIT_NEW_TRAIN_VALID_INDEX', labels_sha256)
    target.mkdir(parents=True, exist_ok=True)
    state = _initial_state(ctx, record, target)
    visit = state['recovery_visits']+1
    while (target/'visits'/f'{visit:06d}.json').exists():
        visit += 1
    started = time.time(); end = min(float(deadline), started+config['recovery_region_seconds'])
    check = lambda: ctx.guard(deadline=end)
    intent = dict(schema=SCHEMA, region_key=region_key(record), start_epoch=started, deadline_epoch=end,
        phase_id=ctx.c['phase_id'], run_id=ctx.c['run_id'], visit=visit, TEST_opened=False,
        tasks=['AUTHENTICATE_FIXED_TRAIN_VALID', 'PRESERVE_OR_FIT', 'CURRENT_NATIVE_AUDIT'],
        expected_outputs=['new checkpoint/state or explicit preservation/failure', 'portable chosen records'],
        acceptance=['no source writes', 'same cohort selection', 'strict TRAIN <1', 'finite full-coordinate costs'],
        explicit_labels_root=str(labels_root) if labels_root is not None else None,
        explicit_labels_index_sha256=labels_sha256)
    _new(target/'intents'/f'{visit:06d}.json', intent)
    timings = {}; outcomes = []; records = []; status = 'INCOMPLETE'
    try:
        t = time.perf_counter(); train, valid, region = load_training_cohorts(ctx, record, labels_root=labels_root, deadline=end)
        timings['dataset_load'] = time.perf_counter()-t
        record_cohort_epoch(state, train['dataset_identity'], valid['dataset_identity'], train.get('request_index_sha256'))
        source_scale = region_root(ctx, record, source=True)/'NATIVE_SCALE.json'
        if source_scale.exists():
            _, scale_sha = _bound(ctx, source_scale, 'FROZEN_NATIVE_SCALE')
            scale_record = read_json(source_scale)
            if scale_record['graph_sha256'] != region.graph_sha256:
                raise ValueError('Native scale graph mismatch')
            scale = np.asarray(scale_record['scale'], dtype=np.float64)
        else:
            scale, scale_record = legacy.native_scale(region, record, config)
            scale_sha = None
        if scale.shape != (region.m,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Invalid native scale')
        scale_path = target/'NATIVE_SCALE.json'
        if not scale_path.exists():
            _new(scale_path, dict(scale_record, scale=scale, source_sha256=scale_sha))
        elif not np.array_equal(np.asarray(read_json(scale_path)['scale']), scale):
            raise ValueError('Recovery scale changed')
        state.update(dataset_identity=train['dataset_identity'], valid_dataset_identity=valid['dataset_identity'],
            training_rows=train['unique_rows'], valid_rows=valid['unique_rows'], requires_current_dataset_refresh=False)
        t = time.perf_counter()
        for family in FAMILIES:
            _audit_select(ctx, target, family, state, train, valid, config, check)
        records = _baselines(ctx, target, record, region, state, train, valid, scale, config, check)
        timings['initial_audit_and_baselines'] = time.perf_counter()-t
        both = all(target_met(state['selected'].get(f), train['dataset_identity'], valid['dataset_identity']) for f in FAMILIES)
        if both:
            status = 'PRESERVED_BOTH_TARGETS_MET_NO_FITTING'
        else:
            for family in FAMILIES:
                t = time.perf_counter()
                outcomes.extend(_fit_family(ctx, target, record, region, state, train, valid, scale, config, family, end, blocks))
                timings[family.lower()+'_fit_and_audit'] = time.perf_counter()-t
            failed = any(x.get('status') in {'TIME_LIMIT', 'NUMERICAL_FAILURE', 'STATE_OR_RUNTIME_FAILURE'} for x in outcomes)
            status = 'PARTIAL_PRESERVED_PRIOR_CHECKPOINTS' if failed else 'RECOVERY_VISIT_COMPLETE'
        for family in FAMILIES:
            candidate = state['selected'].get(family)
            if candidate:
                records.append(_normal_record(record, region, candidate, train, valid, family))
        state['portable_records'] = records
    except Exception as exc:
        status = 'TIME_LIMIT' if isinstance(exc, TimeoutError) else 'FAILED_PRESERVED_PRIOR_CHECKPOINTS'
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        state['recovery_failures'].append(dict(epoch=time.time(), visit=visit, **error))
        outcomes.append(dict(status=status, error=error))
    finally:
        state['recovery_visits'] = visit; state['status'] = status
        result = dict(schema=SCHEMA, region_key=region_key(record), status=status, visit=visit,
            start_epoch=started, end_epoch=time.time(), deadline_epoch=end, elapsed_seconds=time.time()-started,
            timings=timings, outcomes=outcomes, records=records, TEST_opened=False,
            trained_updates=sum(x.get('updates', 0) for x in outcomes),
            new_xgb_rounds=sum(x.get('rounds', 0) for x in outcomes), source_root=str(ctx.source_root))
        state['recovery_history'].append({k: result[k] for k in ('visit', 'status', 'start_epoch', 'end_epoch', 'trained_updates', 'new_xgb_rounds')})
        write_json(target/'STATE.json', state)
        _new(target/'visits'/f'{visit:06d}.json', result)
        ctx.flush_bindings()
    return result


def _worker(payload):
    from threadpoolctl import threadpool_limits
    from .context import RecoveryContext
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    ctx = RecoveryContext(payload['config_path'], step_id='training')
    with threadpool_limits(limits=1):
        return run_region(ctx, payload['record'], deadline=payload['deadline'])


def run(ctx, *, deadline, region_keys=None, max_workers=4):
    """One fair pass over explicit support keys; repeated passes are caller-owned."""
    ctx.require_action('training'); require_training_open(ctx); ctx.guard(deadline=deadline)
    if type(max_workers) is not int or not 1 <= max_workers <= 4:
        raise ValueError('Recovery supports one to four bounded workers')
    support_path, digest = _bound(ctx, Path(ctx.source_root)/'SUPPORT_FREEZE.json', 'FROZEN_REGIONAL_SUPPORT',
        ctx.c.get('recovery', {}).get('support_sha256'))
    support = read_json(support_path)
    if support.get('status') != 'PASS':
        raise ValueError('Frozen support is not admitted')
    available = {region_key(r): r for s in support['suites'] for r in s['region_records']}
    keys = list(region_keys) if region_keys is not None else sorted(k for k in available if k.startswith('s03:'))
    if not keys or len(set(keys)) != len(keys) or any(k not in available for k in keys):
        raise ValueError('Nonempty unique explicitly supported region task list required')
    _roots(ctx)
    root = Path(ctx.root)/'training'/'passes'/str(time.time_ns())
    if not root.resolve().is_relative_to(Path(ctx.root).resolve()):
        raise ValueError('Pass output alias escapes new run')
    root.mkdir(parents=True, exist_ok=False)
    _new(root/'INTENT.json', dict(schema=SCHEMA, config_path=str(ctx.path), config_sha256=sha(ctx.path),
        region_keys=keys, max_workers=max_workers, deadline_epoch=deadline,
        support_sha256=digest, TEST_opened=False, scheduling='ONE_FAIR_PASS_FIXED_CURRENT_TRAIN_VALID'))
    results = []; dispatched = set(); pool = None
    try:
        if max_workers == 1:
            for key in keys:
                ctx.guard(deadline=deadline)
                dispatched.add(key)
                row = run_region(ctx, available[key], deadline=deadline)
                _new(root/(key.replace(':', '_')+'.json'), row); results.append(row)
        else:
            pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context('spawn'))
            pending = {}; iterator = iter(keys)
            while True:
                while len(pending) < max_workers:
                    key = next(iterator, None)
                    if key is None:
                        break
                    ctx.guard(deadline=deadline)
                    payload = dict(config_path=str(ctx.path), record=available[key], deadline=deadline)
                    pending[pool.submit(_worker, payload)] = key; dispatched.add(key)
                if not pending:
                    break
                done, _ = wait(pending, timeout=.5, return_when=FIRST_COMPLETED)
                ctx.guard(deadline=deadline)
                for future in done:
                    key = pending.pop(future)
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = dict(region_key=key, status='WORKER_FAILED', error_type=type(exc).__name__, error=str(exc), records=[])
                    _new(root/(key.replace(':', '_')+'.json'), row); results.append(row)
            pool.shutdown(wait=True); pool = None
    except Exception as exc:
        _new(root/'FAILURE.json', dict(type=type(exc).__name__, message=str(exc), epoch=time.time()))
    finally:
        if pool is not None:
            legacy._stop_own_pool(pool)
        completed = {r['region_key'] for r in results}
        for key in keys:
            if key not in completed:
                results.append(dict(region_key=key, status='INTERRUPTED' if key in dispatched else 'UNATTEMPTED', records=[]))
        summary = dict(schema=SCHEMA, status='PASS_COMPLETE_TASK_ACCOUNTING', completed_visits=len(completed),
            requested_regions=len(keys), deadline_epoch=deadline, TEST_opened=False,
            regions=[{k: r[k] for k in ('region_key', 'status')} for r in results],
            records=[record for result in results for record in result.get('records', [])],
            scientific_success_claim=False, output_directory=str(root))
        _new(root/'RESULT.json', summary)
        ctx.flush_bindings()
    return summary
