"""Frozen whole-region recovery portfolios; old graphs remain read-only.

No fitting, label admission, TEST selection or per-iteration fallback occurs here.
All cost equations and serialized model readers come from the immutable v1 code.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import psutil

from partC_rev1_src.pacnhn_v1.control import array_hash, read_json, sha, write_json
from partC_rev1_src.pacnhn_v1.solver import PolynomialObjective

SUITES = ('s02', 's03', 's04', 's05', 's06')
FAMILIES = ('FFTT', 'POWER4', 'ICNN', 'XGBOOST')
LABELS = {'FFTT': 'FFTT', 'POWER4': 'POWER4_WITH_FFTT',
          'ICNN': 'ICNN_HYBRID', 'XGBOOST': 'XGB_HYBRID'}
KINDS = {'POWER4': 'DIRECT_COST_POWER4', 'ICNN': 'MONOTONE_ICNN_POWER4',
         'XGBOOST': 'XGBOOST_LOG_COST'}
SCHEMA = 'PACNHN_RECOVERY24_PORTFOLIO_V1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(x in '0123456789abcdef' for x in value)


def output_path(ctx, relative):
    """Every caller-owned write is below NEW root, never the old source tree."""
    root = Path(ctx.root).resolve()
    source = Path(ctx.source_root).resolve(strict=True)
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError('Source and output roots must be disjoint')
    target = (root / relative).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError('Output escapes new campaign root')
    return target


def _identity_current(record, role):
    current = record.get('current_' + role + '_dataset_identity')
    return (_sha(current) and record.get('audit_' + role + '_dataset_identity') == current
            and record.get('audit_model_sha256') == record.get('model_sha256'))


def model_eligible(record, *, require_train=False):
    if not record or record.get('status') != 'AVAILABLE':
        return False
    family = record.get('family')
    if family not in KINDS or record.get('model_kind') != KINDS[family]:
        return False
    digest = record.get('model_sha256')
    if not (_sha(digest) and record.get('model_path')
            and record.get('property_status') == 'PASS'
            and record.get('property_model_sha256') == digest
            and _sha(record.get('pair_keys_identity')) and _sha(record.get('graph_sha256'))):
        return False
    artifacts = record.get('artifact_hashes')
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    normalized = {str(Path(p).resolve()): h for p, h in artifacts.items()}
    if normalized.get(str(Path(record['model_path']).resolve())) != digest or not all(_sha(h) for h in artifacts.values()):
        return False
    if not _identity_current(record, 'valid') or not _finite(record.get('valid_wmape_percent')):
        return False
    if not _identity_current(record, 'train') or not _finite(record.get('train_wmape_percent')):
        return False
    if require_train:
        return (_identity_current(record, 'train') and _finite(record.get('train_wmape_percent'))
                and record['train_wmape_percent'] < 1.)
    return True


def choose_region(requested, candidates):
    """Pure decision on scalar TRAIN/VALID metadata. Missing evidence => FFTT.

    Learned-family fallback compares POWER4 against FFTT on current VALID. FFTT
    wins ties. The POWER4 arm retains valid Power4 regardless of baseline rank.
    The requested learned model requires the unchanged strict TRAIN
    target plus current audited VALID and a model-hash-bound property PASS.
    """
    if requested not in FAMILIES:
        raise ValueError('Unknown requested family')
    if requested == 'FFTT':
        return 'FFTT', 'REQUESTED_BASELINE'
    primary = candidates.get(requested)
    if requested == 'POWER4':
        return (('POWER4', 'VALID_REQUESTED_POWER4') if model_eligible(primary)
                else ('FFTT', 'MISSING_OR_INVALID_REQUESTED_POWER4'))
    if requested in ('ICNN', 'XGBOOST') and model_eligible(primary, require_train=True):
        return requested, 'CURRENT_TRAIN_QUALIFIED_PROPERTY_PASS'
    power, fftt = candidates.get('POWER4'), candidates.get('FFTT')
    valid_id = power.get('current_valid_dataset_identity') if power else None
    fftt_comparable = (fftt and fftt.get('status') == 'AVAILABLE'
        and fftt.get('current_valid_dataset_identity') == valid_id
        and fftt.get('audit_valid_dataset_identity') == valid_id
        and _sha(valid_id) and _finite(fftt.get('valid_wmape_percent')))
    if model_eligible(power) and fftt_comparable and power['valid_wmape_percent'] < fftt['valid_wmape_percent']:
        return 'POWER4', 'LOWER_SAME_CURRENT_VALID_ERROR_THAN_FFTT'
    if not model_eligible(power) or not fftt_comparable:
        return 'FFTT', 'NO_VALID_COMPARABLE_POWER4_EVIDENCE'
    return 'FFTT', 'FFTT_VALID_ERROR_NO_WORSE_THAN_POWER4'


def _record_summary(record):
    if record is None:
        return None
    keys = ('status', 'model_sha256', 'train_wmape_percent', 'valid_wmape_percent',
            'current_train_dataset_identity', 'current_valid_dataset_identity',
            'audit_train_dataset_identity', 'audit_valid_dataset_identity',
            'audit_model_sha256', 'property_status', 'property_model_sha256')
    return {k: record[k] for k in keys if k in record}


def make_portfolio(support, records):
    """Pure, testable construction on an authenticated support object."""
    if support.get('status') != 'PASS':
        raise ValueError('Passed frozen support required')
    suites = support.get('suites', [])
    if len(suites) != len(SUITES) or {x['suite_id'] for x in suites} != set(SUITES):
        raise ValueError('Exact five-suite support required')
    indexed = {}
    needed = {(s['suite_id'], r['region_index']) for s in suites for r in s['region_records']}
    graphs = {(s['suite_id'], r['region_index']): r['sha256'] for s in suites for r in s['region_records']}
    for record in records:
        key = (record['suite_id'], record['region_index'], record['family'])
        if (key in indexed or key[2] not in FAMILIES or key[:2] not in needed
                or type(key[1]) is not int or key[1] < 0):
            raise ValueError('Unexpected or duplicate model record')
        if record.get('graph_sha256') not in (None, graphs[key[:2]]):
            raise ValueError('Record refers to another physical region')
        indexed[key] = record
    result = []
    for suite in sorted(suites, key=lambda s: SUITES.index(s['suite_id'])):
        sid = suite['suite_id']; region_rows = suite['region_records']
        ids = [x['region_index'] for x in region_rows]
        if len(set(ids)) != len(ids) or len(ids) != suite['active_regions']:
            raise ValueError('Duplicate/missing active support regions')
        if sum(r['pairs'] for r in region_rows) != suite['active_pairs']:
            raise ValueError('Active pair accounting differs')
        for requested in FAMILIES:
            chosen = []
            for region in sorted(region_rows, key=lambda r: r['region_index']):
                rid = region['region_index']
                candidates = {f: indexed.get((sid, rid, f)) for f in FAMILIES}
                component, reason = choose_region(requested, candidates)
                model = candidates.get(component) if component != 'FFTT' else None
                if model is not None and model['graph_sha256'] != region['sha256']:
                    raise ValueError('Chosen model uses a different physical region')
                row = dict(region_index=rid, pairs=region['pairs'], component=component,
                           reason=reason, graph_sha256=region['sha256'],
                           evidence={f: _record_summary(candidates[f]) for f in FAMILIES})
                if model is not None:
                    keys = ('model_path', 'model_sha256', 'artifact_hashes', 'model_kind',
                            'pair_keys_identity', 'graph_sha256', 'property_status',
                            'property_model_sha256', 'uncompressed_model_bytes',
                            'uncompressed_booster_bytes', 'forecast_resident_booster_bytes')
                    row['model'] = {k: model[k] for k in keys if k in model}
                chosen.append(row)
            identity = _digest(dict(suite_id=sid, requested_family=requested,
                                    graph_sha256=suite['graph_sha256'], regions=chosen))
            result.append(dict(suite_id=sid, requested_family=requested,
                effective_method=LABELS[requested], solver='MSA' if requested == 'XGBOOST' else 'FW',
                has_potential=requested != 'XGBOOST', graph_sha256=suite['graph_sha256'],
                graph_identity=suite['graph_identity'], OD_identity=suite['OD_identity'],
                regions=chosen, active_pairs=suite['active_pairs'], active_regions=len(chosen),
                component_counts=dict(Counter(r['component'] for r in chosen)),
                portfolio_identity=identity, fixed_across_scenarios=True))
    return result


def build_portfolios(ctx, records, *, records_binding=None):
    ctx.require_action('portfolio'); ctx.guard()
    destination = output_path(ctx, 'portfolios/PORTFOLIOS.json')
    if destination.exists():
        raise FileExistsError('Frozen portfolios already exist; no reselection')
    if not records_binding:
        raise ValueError('Exact normalized record source binding required')
    ctx.bind(records_binding['path'], 'CURRENT_TRAIN_VALID_MODEL_RECORDS', records_binding['sha256'])
    source_records = read_json(records_binding['path'])
    if source_records != records:
        raise ValueError('Provided records differ from bound normalized metadata')
    support_path = Path(ctx.source_root) / 'SUPPORT_FREEZE.json'
    expected = {str(Path(p).resolve()): h for p, h in ctx.c['source_bindings'].items()}
    support_sha = ctx.bind(support_path, 'FROZEN_SUPPORT', expected[str(support_path.resolve())])
    support = read_json(support_path)
    portfolios = make_portfolio(support, records)
    ctx.guard()
    report = dict(schema=SCHEMA, phase_id=ctx.c['phase_id'], run_id=ctx.c['run_id'],
        source_root=str(Path(ctx.source_root).resolve()), support_freeze_sha256=support_sha,
        records_binding=records_binding, policies=dict(train_target_percent=1.,
        strict=True, fallback='LEARNED_ARMS:LOWER_CURRENT_VALID_POWER4_OR_FFTT; TIE_OR_MISSING=FFTT',
        power4_arm='VALID_POWER4; MISSING_OR_INVALID=FFTT',
        TEST_used=False, dynamic_fallback=False), portfolios=portfolios)
    report['identity'] = _digest(report)
    write_json(destination, report); ctx.flush_bindings()
    return report


class FrozenComposite:
    """Exact whole-region substitution, with portfolio-wide solver semantics."""
    def __init__(self, base, k4, regional_indices, blocks, *, cost_only=False):
        self.has_potential = not cost_only
        self.base = np.asarray(base, np.float64).copy()
        self.k4 = np.asarray(k4, np.float64).copy()
        self.regional_indices = np.asarray(regional_indices, np.int64)
        if (self.base.shape != self.k4.shape or self.base.ndim != 1
                or len(np.unique(self.regional_indices)) != len(self.regional_indices)
                or np.any(self.regional_indices < 0) or np.any(self.regional_indices >= len(self.base))):
            raise ValueError('Invalid regional coordinate universe')
        self.blocks = []; covered = []
        for indices, model in blocks:
            indices = np.asarray(indices, np.int64)
            if (indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices)
                    or not np.isin(indices, self.regional_indices).all()):
                raise ValueError('Invalid or nonregional model block')
            if self.has_potential and not model.has_potential:
                raise ValueError('Cost-only component in potential portfolio')
            if len(model.c0) != len(indices) or not np.array_equal(model.c0, self.base[indices]):
                raise ValueError('Regional c0/order differs')
            covered.extend(indices.tolist()); self.blocks.append((indices.copy(), model))
        if len(set(covered)) != len(covered):
            raise ValueError('Overlapping whole-region components')
        # Unreplaced coordinates remain exact FFTT; selected blocks replace them once.
        if covered:
            self.base[covered] = 0.; self.k4[covered] = 0.
        self.polynomial = PolynomialObjective(self.base, self.k4)

    def cost(self, x):
        cost = self.polynomial.cost(x)
        for indices, model in self.blocks:
            cost[indices] = model.cost(np.asarray(x)[indices])
        if not np.isfinite(cost).all() or (cost < 0).any():
            raise ValueError('Invalid actual portfolio cost; no clipping or fallback')
        return cost

    def value_cost(self, x):
        if not self.has_potential:
            raise TypeError('XGB-designated portfolio is cost-only')
        value, cost = self.polynomial.value_cost(x)
        for indices, model in self.blocks:
            v, c = model.value_cost(np.asarray(x)[indices]); value += float(v); cost[indices] = c
        if not np.isfinite(value) or not np.isfinite(cost).all() or (cost < 0).any():
            raise ValueError('Invalid actual portfolio value/cost')
        return value, cost

    def hess_vec(self, x, direction):
        if not self.has_potential:
            raise TypeError('Cost-only portfolio has no potential Hessian')
        result = self.polynomial.hessian_diag(x) * direction
        for indices, model in self.blocks:
            result[indices] = model.hess_vec(np.asarray(x)[indices], np.asarray(direction)[indices])
        return result


def validate_graph_arrays(a, common, report, od_identity, graph_identity):
    bcount = int(a['backbone_count'])
    if bcount != 87582 or report['backbone_count'] != bcount:
        raise ValueError('Frozen B count differs')
    for key in ('tail', 'head', 'edge_ids', 'base', 'k4', 'closed_mask'):
        if not np.array_equal(a[key][:bcount], common[key][:bcount]):
            raise ValueError('Frozen P/AC backbone differs: ' + key)
    if report['OD_identity'] != od_identity or report['graph_identity'] != graph_identity:
        raise ValueError('Frozen graph/OD identity differs')
    if (len(a['pair_keys']) != report['active_pairs']
            or len(np.unique(a['pair_keys'])) != len(a['pair_keys'])
            or len(a['pair_region']) != len(a['pair_keys'])):
        raise ValueError('Regional pair universe differs')
    if len(a['base']) != len(a['tail']) or len(a['closed_mask']) != len(a['tail']):
        raise ValueError('Graph field dimensions differ')
    if np.any(a['closed_mask'][bcount:]):
        raise ValueError('Represented I95 closes non-backbone edges')
    m = len(a['pair_keys'])
    if ('c0' not in a or len(a['c0']) != m or
            not np.array_equal(a['base'][bcount:bcount+m], a['c0']) or
            np.any(a['k4'][bcount:bcount+m] != 0)):
        raise ValueError('Regional FFTT cost ownership differs')


def load_objective(ctx, suite_id, requested_family, portfolio, *, deadline):
    from partC_rev1_src.pacnhn_v1.topology import load_common
    from partC_rev1_src.pacnhn_v1.oracle import Network
    from partC_rev1_src.pacnhn_v1.models import load_model
    ctx.require_action('assignments'); ctx.guard(deadline=deadline)
    source = Path(ctx.source_root)
    payload = dict(portfolio); identity = payload.pop('identity', None)
    if (portfolio.get('schema') != SCHEMA or identity != _digest(payload)
            or portfolio.get('phase_id') != ctx.c['phase_id'] or portfolio.get('run_id') != ctx.c['run_id']
            or Path(portfolio.get('source_root', '')).resolve() != source.resolve()):
        raise ValueError('Unverified portfolio identity/authority')
    if suite_id not in SUITES or requested_family not in FAMILIES:
        raise ValueError('Unknown portfolio coordinates')
    choices = [p for p in portfolio['portfolios']
               if p['suite_id'] == suite_id and p['requested_family'] == requested_family]
    if len(choices) != 1:
        raise ValueError('Missing/duplicate portfolio')
    chosen = choices[0]
    if chosen['solver'] != ('MSA' if requested_family == 'XGBOOST' else 'FW'):
        raise ValueError('Portfolio solver semantics changed')
    for name in ('RESULT.json', 'common_graph.npz', 'common_od.npz', 'partitions.npz'):
        ctx.bind(source / 'topology' / name, 'FROZEN_COMMON_P_INPUT')
    graph_p, od, common, parts = load_common(source)
    del graph_p, parts
    ctx.guard(deadline=deadline)
    support_path = source / 'SUPPORT_FREEZE.json'
    ctx.bind(support_path, 'FROZEN_SUPPORT', portfolio['support_freeze_sha256'])
    support = read_json(support_path)
    reports = [s for s in support['suites'] if s['suite_id'] == suite_id]
    if len(reports) != 1:
        raise ValueError('Frozen suite support missing')
    report = reports[0]
    if read_json(source / 'acnhn' / suite_id / 'REPORT.json') != report:
        raise ValueError('AC report differs from frozen support')
    if report['graph_sha256'] != chosen['graph_sha256']:
        raise ValueError('Portfolio graph changed')
    graph_path = source / 'acnhn' / suite_id / 'graph.npz'
    ctx.bind(graph_path, 'FROZEN_AC_GRAPH', report['graph_sha256'])
    with np.load(graph_path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    graph = Network(a['tail'], a['head'], a['blocked'], a['edge_ids'], a['node_ids'])
    validate_graph_arrays(a, common, report, od.identity, graph.identity)
    del common
    regions = {r['region_index']: r for r in chosen['regions']}
    if len(regions) != len(chosen['regions']) or set(regions) != set(np.unique(a['pair_region']).tolist()):
        raise ValueError('Portfolio region coverage differs')
    regional_meta = {r['region_index']: r for r in report['region_records']}
    xgb = [r['model'] for r in chosen['regions'] if r['component'] == 'XGBOOST']
    if xgb:
        forecast = 0
        for record in xgb:
            value = record.get('forecast_resident_booster_bytes')
            if value is None:
                raw = record.get('uncompressed_model_bytes')
                if type(raw) is not int or raw <= 0:
                    raise ValueError('Missing complete XGBoost memory forecast')
                value = 3 * raw
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid XGBoost memory forecast')
            forecast += value
        reserve = ctx.c['assignment']['graph_runtime_reserve_bytes']
        available = psutil.virtual_memory().available - ctx.c['resources']['minimum_available_bytes']
        tree_left = ctx.c['resources']['tree_rss_bytes'] - psutil.Process().memory_info().rss
        if forecast + reserve > min(available, tree_left):
            raise MemoryError('Frozen portfolio XGBoost resident forecast exceeds guarded RAM')
    bcount = int(a['backbone_count']); blocks = []
    for rid, row in sorted(regions.items()):
        ctx.guard(deadline=deadline)
        positions = np.flatnonzero(a['pair_region'] == rid)
        if row['pairs'] != len(positions) or row['graph_sha256'] != regional_meta[rid]['sha256']:
            raise ValueError('Region ownership/support differs')
        if row['component'] == 'FFTT':
            continue
        rec = row['model']; component = row['component']
        if rec['pair_keys_identity'] != array_hash({'pair_keys': a['pair_keys'][positions]}):
            raise ValueError('Model coordinate order differs')
        if rec['model_kind'] != KINDS[component] or rec['property_model_sha256'] != rec['model_sha256'] or rec['property_status'] != 'PASS':
            raise ValueError('Frozen model property/kind binding differs')
        ctx.bind(rec['model_path'], 'FROZEN_PORTFOLIO_' + component, rec['model_sha256'])
        artifacts = rec.get('artifact_hashes')
        if not artifacts or str(Path(rec['model_path']).resolve()) not in {str(Path(p).resolve()) for p in artifacts}:
            raise ValueError('Missing selected model artifact commitment')
        for path, digest in artifacts.items():
            ctx.bind(path, 'FROZEN_MODEL_ARTIFACT', digest)
        model = load_model(rec['model_path'])
        if model.kind != KINDS[component] or not np.array_equal(model.c0, a['c0'][positions]):
            raise ValueError('Loaded component kind/c0 differs')
        blocks.append((bcount + positions, model))
    objective = FrozenComposite(a['base'], a['k4'], np.arange(bcount, bcount + len(a['pair_keys'])),
                                blocks, cost_only=requested_family == 'XGBOOST')
    ctx.guard(deadline=deadline)
    return graph, od, a, objective
