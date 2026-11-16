"""Isolated v5 nested TRAIN-size experiment; never changes production selections.

The explicit preparation freezes 16 regions and 160 tasks. Every fraction starts
fresh and calibrates on that TRAIN subset only. TEST is a separate post-freeze
action. Importing this module reads no campaign input and launches no work.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from ..pacnhn_v1.control import array_hash, read_json, sha

SCHEMA = 'PACNHN_RECOVERY24_NESTED_ABLATION_V1'
FRACTIONS = (.1, .25, .5, .7, 1.)
FAMILIES = ('ICNN', 'XGBOOST')
STRATA = ('large', 'small', 'good', 'bad')
FIT_POLICY = dict(icnn_width=32, icnn_updates=1000, icnn_learning_rate=.001,
                  icnn_audit_interval=250, xgb_rounds=32, xgb_depth=3,
                  xgb_learning_rate=.05, batch_rows=128, threads=1)


def _new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _root(ctx):
    root = ctx.output_path('ablation')
    if root.resolve().is_relative_to(Path(ctx.source_root).resolve()):
        raise ValueError('Ablation outputs must be outside immutable source')
    return root


def _settings(ctx):
    c = ctx.c.get('ablation')
    if not isinstance(c, dict) or c.get('enabled') is not True:
        raise PermissionError('Explicit ablation configuration required')
    if (c.get('seconds') != 1800 or c.get('regions_per_stratum') != 4
            or tuple(c.get('fractions', ())) != FRACTIONS
            or c.get('fit_policy') != FIT_POLICY or type(c.get('seed')) is not int):
        raise ValueError('Declared 16-region/160-fit/1800-second ablation policy required')
    digest = c.get('selection_metadata_sha256')
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError('Explicit pre-experiment selection metadata SHA256 required')
    return c


def select_regions(rows):
    """Size then pre-experiment ICNN VALID ranks; disjoint, stable ID ties.

    Only the explicitly named scalar fields are read. Neither TEST nor TRAIN
    target attainment is a stratum-selection criterion.
    """
    eligible = []
    seen = set()
    for row in rows:
        if row.get('suite_id') != 's03' or row.get('family') != 'ICNN' or row.get('status') != 'AVAILABLE':
            continue
        index, pairs = int(row['region_index']), int(row['pairs'])
        value = float(row['VALID_WMAPE_percent'])
        if index < 0 or pairs < 1 or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid pre-experiment stratum metadata')
        if index in seen:
            raise ValueError('Duplicate s03 ICNN region')
        seen.add(index)
        eligible.append(dict(suite_id='s03', region_index=index, pairs=pairs,
                             selection_VALID_WMAPE_percent=value))
    if len(eligible) < 16:
        raise ValueError('At least 16 independently identified eligible regions required')
    selected = []
    keys = [lambda r: (-r['pairs'], r['region_index']),
            lambda r: (r['pairs'], r['region_index']),
            lambda r: (r['selection_VALID_WMAPE_percent'], r['region_index']),
            lambda r: (-r['selection_VALID_WMAPE_percent'], r['region_index'])]
    for stratum, key in zip(STRATA, keys):
        chosen = sorted(eligible, key=key)[:4]
        selected.extend(dict(r, stratum=stratum) for r in chosen)
        used = {r['region_index'] for r in chosen}
        eligible = [r for r in eligible if r['region_index'] not in used]
    return selected


def nested_indices(input_ids, seed, region_key):
    ids = list(input_ids)
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError('Nonempty unique ordered TRAIN input identities required')
    order = sorted(range(len(ids)), key=lambda i: (
        hashlib.sha256(f'{seed}|{region_key}|{ids[i]}'.encode()).digest(), ids[i]))
    return {fraction: np.asarray(order[:max(1, math.floor(fraction * len(order)))], dtype=np.int64)
            for fraction in FRACTIONS}


def subset_train(train, indices):
    if train.get('role') != 'TRAIN':
        raise PermissionError('Only TRAIN data may calibrate a fraction')
    ix = np.asarray(indices)
    if ix.ndim != 1 or not ix.size or ix.dtype.kind not in 'iu' or len(set(ix.tolist())) != len(ix):
        raise ValueError('Unique nonempty subset indices required')
    u, tau = np.asarray(train['u']), np.asarray(train['tau'])
    if u.ndim != 2 or u.shape != tau.shape or ix.min() < 0 or ix.max() >= len(u):
        raise ValueError('Invalid subset shape or index')
    c0 = np.asarray(train['c0'])
    q, y = u[ix].copy(), tau[ix].copy()
    if c0.shape != (u.shape[1],) or any(not np.isfinite(a).all() or np.any(a < 0) for a in (q, y, c0)):
        raise ValueError('Finite nonnegative full-coordinate TRAIN data required')
    if y.sum() <= 0:
        raise ValueError('Native TRAIN WMAPE denominator must be positive')
    records = [train['records'][int(i)] for i in ix]
    return dict(u=q, tau=y, c0=c0.copy(), records=records, role='TRAIN',
                dataset_identity=array_hash({'u': q, 'tau': y, 'c0': c0}),
                unique_rows=len(q), train_cost_mean=float(y.mean()))


def subset_scale(train):
    """Per-coordinate TRAIN mean flow, with 1% positive-median floor.

    If the entire subset has zero flow, unit native flow is a declared numerical
    scale convention, not an invented observation. c0 remains graph-derived.
    """
    if train.get('role') != 'TRAIN':
        raise PermissionError('Scaling must use TRAIN only')
    q = np.asarray(train['u'], dtype=np.float64)
    if q.ndim != 2 or not q.size or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError('Invalid TRAIN flow matrix')
    mean = q.mean(axis=0)
    positive = mean[mean > 0]
    floor = .01 * float(np.median(positive)) if positive.size else 1.
    scale = np.maximum(mean, floor)
    return scale, dict(policy='SUBSET_TRAIN_MEAN_FLOW_POSITIVE_MEDIAN_FLOOR',
                       train_dataset_identity=train['dataset_identity'], floor=floor,
                       all_zero_flow_unit_convention=not bool(positive.size),
                       VALID_or_TEST_used=False)


def _code_bindings(ctx):
    from . import training_recovery
    from ..pacnhn_v1 import models, training, regional, control
    paths = [Path(__file__), Path(__file__).with_name('context.py'),
             Path(__file__).with_name('labels.py'), Path(training_recovery.__file__), Path(models.__file__),
             Path(training.__file__), Path(regional.__file__), Path(control.__file__)]
    return {str(p.resolve()): ctx.bind(p, 'ABLATION_SOURCE') for p in paths}


def prepare(ctx):
    """Metadata-only task freeze; no model, label target or TEST access."""
    ctx.require_action('training'); ctx.guard()
    c = _settings(ctx)
    _input_freeze(ctx)
    source = Path(ctx.source_root) / 'training' / 'LEARNING_RESULTS.csv'
    metadata_sha = ctx.bind(source, 'PRE_EXPERIMENT_VALID_AND_SIZE_METADATA', c.get('selection_metadata_sha256'))
    with source.open(encoding='utf-8-sig', newline='') as stream:
        selected = select_regions(csv.DictReader(stream))
    support_path = Path(ctx.source_root) / 'SUPPORT_FREEZE.json'
    support_sha = ctx.bind(support_path, 'FROZEN_REGION_SUPPORT')
    support = read_json(support_path)
    if support.get('status') != 'PASS':
        raise ValueError('Admitted frozen regional support required')
    records = {int(r['region_index']): r for s in support['suites'] if s['suite_id'] == 's03'
               for r in s['region_records']}
    for row in selected:
        row['record'] = records[row['region_index']]
        row['region_key'] = f"s03:{row['region_index']:06d}"
    tasks = [dict(task_id=f"{r['region_key'].replace(':', '_')}_{int(f*100):03d}_{family}",
                  region_key=r['region_key'], stratum=r['stratum'], fraction=f, family=family)
             for r in selected for f in FRACTIONS for family in FAMILIES]
    plan = dict(schema=SCHEMA, phase_id=ctx.c['phase_id'], run_id=ctx.c['run_id'],
                config_sha256=ctx.config_sha256, seed=c['seed'], policy=FIT_POLICY,
                selection_metadata=dict(path=str(source), sha256=metadata_sha),
                support_sha256=support_sha, regions=selected, tasks=tasks,
                TEST_used=False, no_production_registry_mutation=True,
                code_sha256=_code_bindings(ctx), prepared_epoch=time.time())
    path = _root(ctx) / 'PLAN.json'
    _new(path, plan)
    return dict(path=str(path), sha256=sha(path), tasks=len(tasks))


def fit_case(train, valid, *, family, seed, check, structural_zero_mask, structural_zero_evidence):
    """Fresh initialization and fixed fitting allowance for one fraction.

    ICNN selection uses the qualified-TRAIN pool, if any, then fixed VALID. Every case
    completes 1000 updates regardless of target attainment; no TEST argument.
    """
    import torch
    from ..pacnhn_v1.models import (fit_xgboost_cost, XGB_CENTERED, update_icnn, ICNN_SCALED)
    from ..pacnhn_v1.training import _initialize_icnn, _evaluate
    if train.get('role') != 'TRAIN' or valid.get('role') != 'VALID':
        raise PermissionError('Explicit disjoint TRAIN and VALID roles required')
    train_ids = {r['input_identity'] for r in train['records']}
    valid_ids = {r['input_identity'] for r in valid['records']}
    if train_ids & valid_ids:
        raise ValueError('TRAIN/VALID input leakage')
    if not np.array_equal(train['c0'], valid['c0']):
        raise ValueError('Fixed native cost baseline mismatch')
    torch.set_num_threads(1)
    check(); scale, calibration = subset_scale(train)
    started = time.perf_counter()
    if family == 'ICNN':
        cfg = dict(seed=seed, icnn_coefficient_parameterization=ICNN_SCALED,
                   icnn_initial_neural_cost_share=.001)
        model = _initialize_icnn(train, scale, cfg, 'cpu', FIT_POLICY['icnn_width'])
        initial = array_hash(model.export_arrays())
        optimizer = torch.optim.Adam(model.parameters(), lr=FIT_POLICY['icnn_learning_rate'])
        generator = torch.Generator().manual_seed(seed + FIT_POLICY['icnn_width'])
        history = []; best = None
        for update in range(1, FIT_POLICY['icnn_updates'] + 1):
            check()
            ix = torch.randint(len(train['u']), (min(FIT_POLICY['batch_rows'], len(train['u'])),),
                               generator=generator).numpy()
            update_icnn(model, optimizer, train['u'][ix], train['tau'][ix],
                        train_cost_mean=train['train_cost_mean'])
            if update % FIT_POLICY['icnn_audit_interval'] == 0:
                portable = model.export_model()
                tr = _evaluate(portable, train, 128, check)
                va = _evaluate(portable, valid, 128, check)
                history.append(dict(updates=update, TRAIN=tr, VALID=va))
                key = (tr['WMAPE_percent'] >= 1., va['WMAPE_percent'], update)
                if best is None or key < best[0]:
                    best = (key, portable, update, tr, va)
        _, selected, count, tr, va = best
        details = dict(initial_model_array_sha256=initial, actual_updates=FIT_POLICY['icnn_updates'],
                       selected_updates=count, history=history, TRAIN_calibration=model.calibration,
                       selection='TRAIN_QUALIFIED_POOL_IF_ANY_THEN_MIN_FIXED_VALID; EARLIEST_TIE')
    elif family == 'XGBOOST':
        selected = fit_xgboost_cost(train['u'], train['tau'], train['c0'], scale,
            seed=seed, n_estimators=FIT_POLICY['xgb_rounds'], max_depth=FIT_POLICY['xgb_depth'],
            learning_rate=FIT_POLICY['xgb_learning_rate'], device='cpu', n_jobs=1,
            structural_zero_mask=structural_zero_mask, structural_zero_evidence=structural_zero_evidence,
            log_center_policy=XGB_CENTERED, train_dataset_identity=train['dataset_identity'], guard=check)
        rounds = selected.metadata['boosting_rounds_total']
        if selected.booster is not None and rounds != FIT_POLICY['xgb_rounds']:
            raise RuntimeError('Truncated XGBoost fit cannot be called a complete ablation case')
        tr = _evaluate(selected, train, 128, check); va = _evaluate(selected, valid, 128, check)
        details = dict(actual_boosting_rounds=rounds, all_structural_zero=selected.booster is None,
                       fresh_log_center=True, selection='FIXED_32_ROUND_FRESH_MODEL', history=[])
    else:
        raise ValueError('Only the two declared ablation model families are supported')
    check()
    return selected, dict(status='AVAILABLE', family=family, TRAIN=tr, VALID=va,
        scale=calibration, scale_sha256=array_hash({'scale': scale}), fresh_initialization=True,
        TRAIN_identity=train['dataset_identity'], VALID_identity=valid['dataset_identity'],
        training_rows=len(train['u']), target_met=tr['WMAPE_percent'] < 1.,
        fitting_seconds=time.perf_counter()-started, **details)


def _read_plan(ctx):
    c = _settings(ctx); path = _root(ctx) / 'PLAN.json'
    ctx.bind(path, 'FROZEN_ABLATION_PLAN')
    plan = read_json(path)
    if (plan.get('schema') != SCHEMA or plan.get('config_sha256') != ctx.config_sha256
            or len(plan.get('tasks', [])) != 160 or plan.get('policy') != FIT_POLICY
            or plan.get('seed') != c['seed']):
        raise ValueError('Foreign or changed ablation plan')
    for source, digest in plan['code_sha256'].items():
        ctx.bind(source, 'FROZEN_ABLATION_SOURCE', digest)
    return plan, path


def _input_freeze(ctx):
    binding = ctx.c['ablation'].get('holdout_input_freeze')
    if not isinstance(binding, dict) or not binding.get('path') or not binding.get('sha256'):
        raise PermissionError('New holdout input identities must freeze before ablation fitting')
    path = Path(binding['path']).resolve(strict=True)
    if path != (Path(ctx.root) / 'holdout' / 'INPUT_FREEZE.json').resolve():
        raise PermissionError('Canonical new-run holdout input freeze required')
    ctx.bind(path, 'PROSPECTIVE_HOLDOUT_INPUT_FREEZE', binding['sha256'])
    data = read_json(path)
    if (data.get('status') != 'FROZEN_BEFORE_FITTING' or data.get('targets_opened') is not False
            or not isinstance(data.get('created_epoch'), (int, float))
            or not math.isfinite(data['created_epoch']) or data['created_epoch'] > time.time()):
        raise PermissionError('Invalid prospective input-freeze metadata')
    if (Path(ctx.root) / 'holdout' / 'TEST_OPENED.json').exists():
        raise PermissionError('Ablation fitting is prohibited after TEST was opened')
    return binding


def _fit_clock(ctx, root, plan_path, deadline):
    path = root / 'RUN_CLOCK.json'
    if not path.exists():
        start = time.time()
        _new(path, dict(start_epoch=start, deadline_epoch=min(start + 1800., float(deadline),
            float(ctx.c['recovery']['s03_deadline_epoch'])), plan_sha256=sha(plan_path)))
        _new(root / 'CLOCK_BINDING.json', dict(clock_sha256=sha(path)))
    expected = read_json(root / 'CLOCK_BINDING.json')['clock_sha256']
    ctx.bind(path, 'ORIGINAL_ABLATION_CLOCK', expected)
    clock = read_json(path)
    if (clock['plan_sha256'] != sha(plan_path) or clock['deadline_epoch'] > clock['start_epoch'] + 1800.
            or not clock['start_epoch'] <= clock['deadline_epoch']
            or clock['deadline_epoch'] > ctx.c['recovery']['s03_deadline_epoch']):
        raise ValueError('Changed or invalid ablation clock')
    return path, clock, min(float(deadline), clock['deadline_epoch'])


def run(ctx, deadline, *, pilot_only=False):
    """Run in the original 1800-second ablation window; optional two-case pilot.

    Pilot tasks are retained as part of the 160, not repeated. A retry never resets
    the clock. Completed outcomes remain immutable; failures stay explicit.
    """
    from .training_recovery import load_training_cohorts
    from ..pacnhn_v1.training import _structural_zeros
    from ..pacnhn_v1.models import save_model
    ctx.require_action('training'); ctx.guard(deadline=deadline)
    plan, plan_path = _read_plan(ctx); root = _root(ctx)
    _input_freeze(ctx)
    if (root / 'FIT_FREEZE.json').exists():
        raise FileExistsError('Ablation already frozen; no post-TEST fitting')
    clock_path, clock, stop = _fit_clock(ctx, root, plan_path, deadline)
    check = lambda: ctx.guard(deadline=stop)
    # Largest-region 100% fits are a conservative pilot of each family's cost.
    first = plan['regions'][0]['region_key']
    ordered = sorted(plan['tasks'], key=lambda t: (not (t['region_key'] == first and t['fraction'] == 1.),
                                                  t['region_key'], t['fraction'], t['family']))
    records = {r['region_key']: r for r in plan['regions']}
    current = None; train = valid = region = indices = zero = None
    newly_run = []; timed_out = False
    for task in ordered:
        destination = root / 'fits' / task['task_id']; result_path = destination / 'RESULT.json'
        if result_path.exists():
            prior = read_json(result_path)
            if prior['task'] != task or prior['plan_sha256'] != sha(plan_path):
                raise ValueError('Foreign existing ablation outcome')
            if prior.get('model'):
                ctx.bind(prior['model']['path'], 'SAVED_ABLATION_MODEL', prior['model']['sha256'])
            continue
        if pilot_only and len(newly_run) >= 2:
            break
        started = time.time()
        try:
            check()
            if current != task['region_key']:
                entry = records[task['region_key']]
                train, valid, region = load_training_cohorts(ctx, entry['record'], deadline=stop)
                ids = [r['input_identity'] for r in train['records']]
                indices = nested_indices(ids, plan['seed'], task['region_key'])
                zero = _structural_zeros(region)
                freeze_path = root / 'cohorts' / (task['region_key'].replace(':', '_') + '.json')
                freeze = dict(region_key=task['region_key'], TRAIN_identity=train['dataset_identity'],
                    VALID_identity=valid['dataset_identity'], request_index_sha256=train['request_index_sha256'],
                    ordered_TRAIN_ids=ids, VALID_ids=[r['input_identity'] for r in valid['records']],
                    subsets={str(f): [ids[int(i)] for i in ix] for f, ix in indices.items()},
                    graph_sha256=region.graph_sha256, pair_keys_identity=region.support_identity)
                if freeze_path.exists():
                    if read_json(freeze_path) != freeze:
                        raise ValueError('Ablation cohorts changed after first fraction')
                else:
                    _new(freeze_path, freeze)
                current = task['region_key']
            fraction = subset_train(train, indices[task['fraction']])
            model, detail = fit_case(fraction, valid, family=task['family'], seed=plan['seed'], check=check,
                structural_zero_mask=zero, structural_zero_evidence='FROZEN_GRAPH_ZERO_COST_DIRECTED_REACHABILITY')
            check()
            model_binding = save_model(model, destination / 'model.npz', metadata={
                'experiment': SCHEMA, 'task_id': task['task_id'], 'production_registry': False,
                'TEST_used': False, 'fit_TRAIN_identity': fraction['dataset_identity']})
            outcome = dict(detail, model=model_binding)
        except Exception as exc:
            outcome = dict(status='TIME_LIMIT' if isinstance(exc, TimeoutError) else 'FAILED',
                           error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        outcome.update(task=task, plan_sha256=sha(plan_path), started_epoch=started,
                       elapsed_seconds=time.time() - started, TEST_used=False)
        _new(result_path, outcome); newly_run.append(outcome)
        if len(newly_run) == 2 and not (root / 'PILOT_FORECAST.json').exists():
            observed = {r['task']['family']: r['elapsed_seconds'] for r in newly_run}
            _new(root / 'PILOT_FORECAST.json', dict(observed_case_seconds=observed,
                conservative_160_fit_seconds=80 * sum(observed.values()),
                remaining_seconds=max(0., stop - time.time()),
                interpretation='Largest-region full-TRAIN pilot; estimate, not guaranteed duration.',
                requires_root_attention=80 * sum(observed.values()) > 1800.))
        if outcome['status'] == 'TIME_LIMIT':
            timed_out = True
            break
    outcomes = []
    for task in plan['tasks']:
        path = root / 'fits' / task['task_id'] / 'RESULT.json'
        outcomes.append(dict(task=task, status=read_json(path)['status'], result_path=str(path),
                             result_sha256=sha(path)) if path.exists() else dict(task=task,
                             status='UNATTEMPTED_TIME_LIMIT' if timed_out else 'UNATTEMPTED'))
    completed = sum(r['status'] == 'AVAILABLE' for r in outcomes)
    terminal = timed_out or all(r['status'] != 'UNATTEMPTED' for r in outcomes)
    if terminal:
        _read_plan(ctx)
        _new(root / 'FIT_FREEZE.json', dict(schema=SCHEMA, status='ALL_160_DISPOSITIONS_FROZEN',
            phase_id=ctx.c['phase_id'], run_id=ctx.c['run_id'], plan_sha256=sha(plan_path),
            clock_sha256=sha(clock_path), completed=completed, outcomes=outcomes,
            ablation_complete=completed == 160, total=160,
            frozen_epoch=time.time(), TEST_used=False))
    receipt = dict(schema=SCHEMA, completed=completed, total=160, pilot_only=bool(pilot_only),
                   all_dispositions_frozen=terminal, all_fits_frozen=terminal,
                   ablation_complete=completed == 160, outcomes=outcomes,
                   clock_sha256=sha(clock_path), deadline_epoch=stop)
    _new(root / 'passes' / f'{time.time_ns()}.json', receipt)
    ctx.flush_bindings()
    return receipt


def _evaluation_authority(ctx, global_binding):
    """Authenticate both freezes before importing a target-capable loader."""
    root = _root(ctx)
    plan, plan_path = _read_plan(ctx)
    fit_path = root / 'FIT_FREEZE.json'
    fit_sha = ctx.bind(fit_path, 'ALL_ABLATION_MODEL_SELECTIONS_FROZEN')
    frozen = read_json(fit_path)
    if (frozen.get('status') != 'ALL_160_DISPOSITIONS_FROZEN' or frozen.get('total') != 160
            or frozen.get('plan_sha256') != sha(plan_path)):
        raise PermissionError('All 160 ablation dispositions must freeze before TEST')
    path = Path(global_binding['path']).resolve(strict=True)
    digest = ctx.bind(path, 'GLOBAL_SELECTION_FREEZE_BEFORE_TEST', global_binding['sha256'])
    global_freeze = read_json(path)
    included = global_freeze.get('ablation_fit_freeze', {})
    if (Path(included.get('path', '')).resolve() != fit_path.resolve()
            or included.get('sha256') != fit_sha
            or global_freeze.get('phase_id') != ctx.c['phase_id']
            or global_freeze.get('run_id') != ctx.c['run_id']):
        raise PermissionError('Global freeze does not authenticate this exact ablation selection')
    if (not path.is_relative_to(Path(ctx.root).resolve()) or global_freeze.get('status') != 'FROZEN'
            or global_freeze.get('test_targets_opened') is not False
            or not isinstance(global_freeze.get('created_epoch'), (int, float))
            or not math.isfinite(global_freeze['created_epoch'])
            or global_freeze['created_epoch'] < frozen['frozen_epoch']
            or global_freeze.get('holdout_input_freeze') != ctx.c['ablation']['holdout_input_freeze']):
        raise PermissionError('Global selection/ablation/input-freeze chronology or population differs')
    allowed = {'AVAILABLE', 'FAILED', 'TIME_LIMIT', 'UNATTEMPTED_TIME_LIMIT'}
    if len(frozen.get('outcomes', [])) != 160 or any(r['status'] not in allowed for r in frozen['outcomes']):
        raise ValueError('Ablation freeze requires explicit terminal dispositions')
    count = sum(r['status'] == 'AVAILABLE' for r in frozen['outcomes'])
    if frozen.get('completed') != count or frozen.get('ablation_complete') is not (count == 160):
        raise ValueError('Ablation completion count or claim differs from actual dispositions')
    if [r['task'] for r in frozen['outcomes']] != plan['tasks']:
        raise ValueError('Ablation freeze must cover the complete ordered task population')
    return plan, frozen, path, digest


def summarize_test(rows):
    """Unweighted regional median/IQR; missing observations are never zero."""
    summaries = []
    for stratum in STRATA:
        for family in FAMILIES:
            for fraction in FRACTIONS:
                group = [r for r in rows if r['stratum'] == stratum and r['family'] == family
                         and r['fraction'] == fraction]
                finite = [r for r in group if r.get('TEST_WMAPE_percent') is not None]
                values = np.asarray([r['TEST_WMAPE_percent'] / 100. for r in finite])
                if values.size and (not np.isfinite(values).all() or np.any(values < 0)):
                    raise ValueError('Invalid native TEST error')
                quantiles = np.quantile(values, [.25, .5, .75]).tolist() if values.size else [None] * 3
                summaries.append(dict(stratum=stratum, model_family=family, split='test',
                    training_fraction=fraction, median_WMAPE=quantiles[1], q25_WMAPE=quantiles[0],
                    q75_WMAPE=quantiles[2], region_seed_rows=len(finite), requested_regions=4,
                    admitted_TEST_inputs=sum(r.get('admitted_TEST_inputs', 0) for r in finite),
                    requested_TEST_inputs=4 * 48, metric_units='fraction; plot_as_percent',
                    unavailable_fit_regions=sum(r.get('fit_status', 'AVAILABLE') != 'AVAILABLE' for r in group),
                    no_admitted_label_regions=sum(r.get('TEST_status') == 'NO_ADMITTED_LABELS' for r in group),
                    complete_region_coverage=len(finite) == 4))
    return summaries


def _csv_new(path, rows):
    if not rows:
        raise ValueError('A scientific export cannot be empty')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def evaluate(ctx, holdout_manifest):
    """Evaluate frozen subset fits on root's same 48-input regional holdouts.

    holdout_manifest is the binding {path,sha256} for MODEL_SELECTION_FREEZE.
    labels.load_holdout_dataset authenticates input/label chronology and receipts.
    This function cannot select models or alter any fit/production registry.
    """
    ctx.require_action('holdout'); ctx.guard()
    plan, frozen, global_path, global_sha = _evaluation_authority(ctx, holdout_manifest)
    from .labels import load_holdout_dataset
    from ..pacnhn_v1.models import load_model
    from ..pacnhn_v1.training import _evaluate
    root = _root(ctx) / 'evaluation'
    root.mkdir(parents=True, exist_ok=False)
    _new(root / 'INTENT.json', dict(schema=SCHEMA, global_selection_freeze=holdout_manifest,
        fit_freeze_sha256=sha(_root(ctx) / 'FIT_FREEZE.json'), started_epoch=time.time(),
        tasks=plan['tasks'], selection_allowed=False))
    rows = []; current = None; test = None; dataset_bindings = []
    for item in frozen['outcomes']:
        ctx.guard()
        task = item['task']
        row = dict(task_id=task['task_id'], region_key=task['region_key'], stratum=task['stratum'],
            family=task['family'], fraction=task['fraction'], fit_status=item['status'],
            requested_TEST_inputs=48)
        fitted = None
        if item.get('result_path'):
            ctx.bind(item['result_path'], 'FROZEN_ABLATION_FIT_RESULT', item['result_sha256'])
            fitted = read_json(item['result_path'])
            if fitted['task'] != task or fitted.get('status') != item['status']:
                raise ValueError('Frozen fit result identity changed')
        if item['status'] != 'AVAILABLE':
            row.update(TEST_status='UNAVAILABLE_FIT', TEST_WMAPE_percent=None,
                       admitted_TEST_inputs=0, fit_error=None if fitted is None else fitted.get('error'))
            rows.append(row); _new(root / (task['task_id'] + '.json'), row)
            continue
        if fitted is None:
            raise ValueError('Available fit has no authenticated outcome')
        if current != task['region_key']:
            test = load_holdout_dataset(ctx, task['region_key'], freeze_path=global_path, freeze_sha256=global_sha)
            if test.get('role') != 'TEST' or test.get('requested_inputs') != 48:
                raise ValueError('Ablation requires the declared same 48-request blind cohort')
            current = task['region_key']
            dataset_bindings.append(dict(region_key=current, dataset_identity=test.get('dataset_identity'),
                source_manifest=test.get('source_manifest'), admitted_inputs=test['admitted_inputs']))
        row.update(TRAIN_WMAPE_percent=fitted['TRAIN']['WMAPE_percent'],
            VALID_WMAPE_percent=fitted['VALID']['WMAPE_percent'], TRAIN_identity=fitted['TRAIN_identity'],
            VALID_identity=fitted['VALID_identity'], training_rows=fitted['training_rows'],
            model_sha256=fitted['model']['sha256'], requested_TEST_inputs=48,
            admitted_TEST_inputs=test['admitted_inputs'], TEST_dataset_identity=test.get('dataset_identity'))
        if test['admitted_inputs'] == 0:
            row.update(TEST_status='NO_ADMITTED_LABELS', TEST_WMAPE_percent=None)
        else:
            ctx.bind(fitted['model']['path'], 'FROZEN_ABLATION_MODEL', fitted['model']['sha256'])
            model = load_model(fitted['model']['path'])
            if not np.array_equal(model.c0, test['c0']) or len(test['u']) != test['admitted_inputs']:
                raise ValueError('Ablation TEST native coordinates/cohort count mismatch')
            result = _evaluate(model, test, 128, ctx.guard)
            row.update(TEST_status='MEASURED', TEST_WMAPE_percent=result['WMAPE_percent'],
                       TEST_absolute_error_sum=result['absolute_error_sum'],
                       TEST_true_cost_sum=result['true_cost_sum'], TEST_coordinates=result['coordinates'])
            del model
        rows.append(row)
        _new(root / (task['task_id'] + '.json'), row)
    _read_plan(ctx)
    _csv_new(root / 'REGION_FRACTION_OBSERVATIONS.csv', rows)
    _csv_new(root / 'FIGURE5_SUMMARIES.csv', summarize_test(rows))
    receipt = dict(schema=SCHEMA, status='COMPLETE_WITH_EXPLICIT_LABEL_COVERAGE',
        fit_cases=160, available_fits=frozen['completed'], ablation_complete=frozen['ablation_complete'],
        unavailable_fits=160-frozen['completed'], measured_cases=sum(r['TEST_status'] == 'MEASURED' for r in rows),
        distinct_requested_regions=16, requested_TEST_inputs=16 * 48,
        global_selection_freeze=holdout_manifest, datasets=dataset_bindings,
        production_registry_changed=False, TEST_used_for_selection=False,
        output_sha256={p.name: sha(p) for p in root.glob('*.csv')})
    _new(root / 'RESULT.json', receipt); ctx.flush_bindings()
    return receipt
