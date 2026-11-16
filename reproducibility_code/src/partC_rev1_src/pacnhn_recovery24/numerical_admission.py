"""Guarded, read-only model property admission for fixed recovery portfolios.

Only declared synthetic probes are evaluated. No TRAIN/VALID/TEST target is read,
no model is fitted, and no historical M02 action or clock is invoked.
"""
from __future__ import annotations

from collections import Counter
import gc
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from ..pacnhn_v1.control import array_hash, read_json, sha
from .labels import native_json
from .portfolio import KINDS, _digest, _sha, output_path

SCHEMA = 'PACNHN_RECOVERY24_NUMERICAL_ADMISSION_V1'


def _new(path, data):
    # Normalize and serialize before opening, including NumPy bool_/scalars.
    text = json.dumps(native_json(data), sort_keys=True, indent=2, allow_nan=False) + '\n'
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)


def stress_inputs(model):
    m = len(model.c0)
    x = np.vstack([np.full(m, v) for v in (0., 1e-12, 1e-6, 1., 10.)]
                  + [np.resize(np.array([0., .25, 1., 4.]), m)])
    return x * model.scale


def audit_power4(model, *, guard=lambda: None, fftt=False):
    from ..pacnhn_v1.models import Power4Model, POWER_KIND
    if type(model) is not Power4Model or model.kind != POWER_KIND:
        raise TypeError('Exact direct-cost Power4 model required')
    arrays = model.export_arrays()
    if (any(a.dtype != np.float64 or not np.isfinite(a).all() for a in arrays.values())
            or np.any(model.c0 < 0) or np.any(model.scale <= 0) or np.any(model.eta < 0)
            or (fftt and np.any(model.eta != 0))):
        raise ValueError('Invalid analytic parameters or nonconstant FFTT')
    rebuilt = Power4Model(**arrays)
    inputs = stress_inputs(model); checks = []
    for u in inputs:
        guard(); value, cost = model.value_cost(u)
        x = u / model.scale
        expected = model.c0 + model.eta * x**4
        integral = float(u @ model.c0 + np.sum(model.eta * model.scale * x**5 / 5))
        h = model.hess_vec(u, np.ones(len(u)))
        exact_h = 4 * model.eta * x**3 / model.scale
        rv, rc = rebuilt.value_cost(u)
        checks.append(bool(np.array_equal(cost, expected) and value == integral
                           and np.array_equal(h, exact_h) and np.array_equal(rc, cost)
                           and rv == value and np.isfinite(value) and value >= 0
                           and np.isfinite(cost).all() and np.all(cost >= model.c0)))
    guard()
    return dict(status='PASS' if all(checks) else 'FAIL', checks=checks,
                dimension=len(model.c0), kind=model.kind, stress_points=len(inputs),
                analytic_scope='Exact nonnegative eta/c0 and positive scale; separable convex integral',
                numerical_scope='Finite declared synthetic probes and exact export reconstruction',
                has_potential=True, accuracy_evaluated=False)


def audit_xgboost(model, *, structural_zero_mask, guard=lambda: None):
    from ..pacnhn_v1.models import XGBoostCostModel, XGB_KIND
    if not isinstance(model, XGBoostCostModel) or model.kind != XGB_KIND or model.has_potential:
        raise TypeError('Exact cost-only XGBoost interface required')
    zero = np.asarray(structural_zero_mask, bool)
    if not np.array_equal(zero, model.structural_zero_mask):
        raise ValueError('XGBoost structural zeros differ from exact graph proof')
    guard(); inputs = stress_inputs(model); batch = model.cost(inputs)
    if batch.shape != inputs.shape or batch.dtype != np.float64:
        raise ValueError('Cost vector shape/dtype/order differs')
    if (not np.isfinite(batch).all() or np.any(batch[:, ~zero] <= 0)
            or np.any(batch[:, zero] != 0)):
        raise FloatingPointError('Nonpositive/nonfinite XGBoost stress costs')
    singles = []
    for u in inputs:
        guard(); singles.append(model.cost(u))
    singles = np.stack(singles)
    # Save/load raw booster in RAM only: no source/model payload is copied to disk.
    guard(); booster = None
    if model.booster is not None:
        from xgboost import Booster
        booster = Booster(model_file=bytearray(model.booster.save_raw(raw_format='ubj')))
        booster.set_param({'device': 'cpu', 'nthread': 1})
    rebuilt = XGBoostCostModel(booster, model.c0.copy(), model.scale.copy(),
        structural_zero_mask=zero.copy(), log_center=model.log_center.copy(), metadata=dict(model.metadata))
    guard(); exported = rebuilt.cost(inputs)
    passed = bool(np.array_equal(batch, singles) and np.array_equal(batch, exported))
    del rebuilt, booster
    guard()
    return dict(status='PASS' if passed else 'FAIL', dimension=len(model.c0), kind=model.kind,
        stress_points=len(inputs), batch_single_order_exact=bool(np.array_equal(batch, singles)),
        raw_export_cost_exact=bool(np.array_equal(batch, exported)),
        stress_input_identity=array_hash({'u':inputs}), structural_zero_count=int(zero.sum()),
        positivity=True, has_potential=False,
        numerical_scope='Finite stress points, complete ordered outputs, batch/single and raw-UBJ reproduction',
        analytic_scope='Positive exp transform on nonzero outputs; exact graph-proven zeros',
        convergence_guarantee=False, accuracy_evaluated=False)


def structural_zeros(region, guard):
    """The existing exact zero-cost directed-path proof, with explicit guards."""
    from ..pacnhn_v1.oracle import aon
    guard(); edges = (region.base == 0) & (region.k4 == 0)
    if not edges.any():
        return region.pair_source == region.pair_target
    od, order = region.demand(np.zeros(region.m))
    result = aon(region.network, od, np.zeros(len(edges)), enabled=edges,
                 workers=1, hash_capacity=16, guard=guard)
    zero = np.empty(region.m, bool); zero[order] = np.isfinite(result['distances'])
    guard(); return zero


def _archive_size(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(not n.endswith('.npy') for n in names):
            raise ValueError('Unexpected/duplicate model NPZ members')
        return sum(i.file_size for i in archive.infolist())


def _source_binding(ctx):
    from ..pacnhn_v1 import final_icnn_guarantees, models, regional, oracle, training
    from . import labels
    paths = [Path(__file__), Path(final_icnn_guarantees.__file__), Path(models.__file__),
             Path(regional.__file__), Path(oracle.__file__), Path(training.__file__), Path(labels.__file__)]
    return {str(p.resolve()):ctx.bind(p, 'MODEL_ADMISSION_SOURCE') for p in paths}


def audit_records(ctx, records, *, deadline):
    """Return exact model-SHA -> admission report; failed/unrun rows stay explicit.

    The caller must apply PASS only to the exact model digest, graph and ordered
    pair binding in ``record_bindings``. This action never selects a model.
    """
    from ..pacnhn_v1.final_icnn_guarantees import audit_model
    from ..pacnhn_v1.models import load_model
    from ..pacnhn_v1.regional import load_region_graph
    auth = ctx.require_action('portfolio')
    end = min(float(deadline), auth['science_deadline_epoch'])
    ctx.guard(deadline=end)
    if not records:
        raise ValueError('Numerical admission requires nonempty explicit records')
    source = _source_binding(ctx)
    support_path = Path(ctx.source_root)/'SUPPORT_FREEZE.json'
    support_sha = ctx.bind(support_path, 'IMMUTABLE_SUPPORT')
    support = read_json(support_path)
    if support.get('status') != 'PASS':
        raise ValueError('Frozen support is not admitted')
    by_region = {(s['suite_id'], r['region_index']):r for s in support['suites'] for r in s['region_records']}
    grouped = {}; keys = set()
    for rec in records:
        if rec.get('status') != 'AVAILABLE':
            continue
        key = (rec['suite_id'], rec['region_index'], rec['family'])
        if key in keys or key[:2] not in by_region or key[2] not in (*KINDS, 'FFTT') or not _sha(rec.get('model_sha256')):
            raise ValueError('Unexpected, duplicate or unbound model record')
        keys.add(key); grouped.setdefault(rec['model_sha256'], []).append(rec)
    if not grouped:
        raise ValueError('No available declared model can be audited')
    started = time.time(); root = output_path(ctx, 'admission')/str(time.time_ns())
    root.mkdir(parents=True, exist_ok=False)
    intent = dict(schema=SCHEMA, phase_id=ctx.c['phase_id'], run_id=ctx.c['run_id'],
        step_id='MODEL_NUMERICAL_ADMISSION', tasks=['AUTHENTICATE_NATIVE_MODELS','SYNTHETIC_PROPERTY_PROBES','BIND_ADMISSION'],
        expected_outputs=['per-model reports','RESULT.json'],
        acceptance_tests=['exact_model_graph_coordinate_identity','model_properties','no_target_or_selection_access'],
        resources=ctx.c['resources'], started_epoch=started, deadline_epoch=end,
        records_identity=_digest(native_json(records)), model_count=len(grouped), record_count=len(keys),
        sources=source, support_freeze_sha256=support_sha, target_values_read=False)
    _new(root/'INTENT.json', intent)
    reports = {}; region = None; current = None; zero = None
    check = lambda:ctx.guard(deadline=end-5.)
    for digest, group in grouped.items():
        t = time.time(); model = None
        bindings = []
        try:
            check()
            for rec in group:
                key = (rec['suite_id'], rec['region_index']); frozen = by_region[key]
                if rec['graph_sha256'] != frozen['sha256']:
                    raise ValueError('Model graph differs from frozen support')
                if current != key:
                    ctx.bind(frozen['path'], 'IMMUTABLE_REGION_GRAPH', frozen['sha256'])
                    region = load_region_graph(frozen['path'], expected_sha256=frozen['sha256'])
                    current = key; zero = None
                if rec['pair_keys_identity'] != array_hash({'pair_keys':region.pair_keys}):
                    raise ValueError('Model ordered output coordinates differ')
                ctx.bind(rec['model_path'], 'MODEL_UNDER_NUMERICAL_ADMISSION', digest)
                for p, h in rec.get('artifact_hashes', {}).items():
                    ctx.bind(p, 'MODEL_ARTIFACT', h)
                raw_bytes = _archive_size(rec['model_path'])
                if raw_bytes > 256*1024**2 + 16*1024**2:
                    raise MemoryError('Model exceeds declared per-region uncompressed envelope')
                if model is None:
                    model = load_model(rec['model_path'])
                expected = KINDS['POWER4' if rec['family']=='FFTT' else rec['family']]
                if rec['model_kind'] != expected or model.kind != expected or not np.array_equal(model.c0, region.c0):
                    raise ValueError('Model family or exact frozen c0 differs')
                bindings.append(dict(suite_id=rec['suite_id'], region_index=rec['region_index'], family=rec['family'],
                    graph_sha256=frozen['sha256'], pair_keys_identity=rec['pair_keys_identity'],
                    model_path=rec['model_path'], model_sha256=digest, uncompressed_archive_bytes=raw_bytes))
                check()
            families = {r['family'] for r in group}
            if model.kind == KINDS['ICNN']:
                detail = audit_model(model, guard=check)
            elif model.kind == KINDS['POWER4']:
                detail = audit_power4(model, guard=check, fftt='FFTT' in families)
            else:
                # A shared byte-identical model across regions must prove zeros for every graph.
                for rec in group:
                    frozen = by_region[(rec['suite_id'],rec['region_index'])]
                    if current != (rec['suite_id'],rec['region_index']):
                        region = load_region_graph(frozen['path'], expected_sha256=frozen['sha256']);current=(rec['suite_id'],rec['region_index'])
                        zero = None
                    if zero is None:zero = structural_zeros(region, check)
                    if not np.array_equal(zero,model.structural_zero_mask):raise ValueError('Structural zero proof differs')
                detail = audit_xgboost(model, structural_zero_mask=zero, guard=check)
            check()
            for rec in group:ctx.bind(rec['model_path'], 'MODEL_AFTER_NUMERICAL_ADMISSION', digest)
        except (TimeoutError, MemoryError, OSError, ImportError) as exc:
            detail = dict(status='NUMERIC_INCONCLUSIVE', error_type=type(exc).__name__, error=str(exc),
                          interpretation='Operational limit or unavailable runtime, not a proved model defect')
        except Exception as exc:
            detail = dict(status='FAIL', error_type=type(exc).__name__, error=str(exc),
                          interpretation='Authentication, structural or resolved numerical failure')
        finally:
            del model
        report = dict(native_json(detail))
        report.update(schema=SCHEMA, model_sha256=digest, record_bindings=bindings,
            requested_records=[dict(suite_id=r['suite_id'],region_index=r['region_index'],family=r['family']) for r in group],
            started_epoch=t, elapsed_seconds=time.time()-t,
            deadline_epoch=end, TEST_used=False, accuracy_evaluated=False)
        report_path=root/(digest+'.json');_new(report_path,report)
        report['report_path']=str(report_path);report['report_sha256']=sha(report_path);reports[digest]=report
    del region
    gc.collect()
    for path, digest in source.items():ctx.bind(path, 'UNCHANGED_ADMISSION_SOURCE', digest)
    ctx.flush_bindings()
    # Wall expiry cannot silently turn a partial/inconclusive run into complete PASS.
    terminal=dict(schema=SCHEMA,status='COMPLETE_WITH_EXPLICIT_MODEL_DISPOSITIONS',
        reports=reports, model_count=len(grouped), record_count=len(keys), started_epoch=started,
        elapsed_seconds=time.time()-started,deadline_epoch=end,
        completed_within_deadline=time.time()<=end,status_counts=dict(Counter(r['status'] for r in reports.values())),
        source_bindings=source, support_freeze_sha256=support_sha, TEST_used=False, fitting_performed=False)
    if time.time()>end:terminal['status']='TIME_LIMIT_WITH_PRESERVED_MODEL_DISPOSITIONS'
    _new(root/'RESULT.json',terminal)
    return reports
