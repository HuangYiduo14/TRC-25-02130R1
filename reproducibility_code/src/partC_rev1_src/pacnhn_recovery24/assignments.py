"""Finite 46-case recovery matrix with explicit old-result reuse and new writes.

The historical solver is imported unchanged. Reuse is a recorded disposition,
not a new timing measurement, and never changes an old result or its certificate.
"""
from __future__ import annotations

import gc
import math
from pathlib import Path
import time

import numpy as np

from partC_rev1_src.pacnhn_v1.control import array_hash, read_json, sha, write_json
from partC_rev1_src.pacnhn_v1.solver import solve, converged
from .portfolio import SUITES, FAMILIES, LABELS, SCHEMA, _digest, _sha, load_objective, output_path, validate_graph_arrays

ADMITTED = {'CONVERGED', 'TIME_LIMIT', 'STAGNATION'}


def matrix():
    rows = []
    for solver in ('CFW', 'FW', 'MSA'):
        for scenario in ('open', 'i95'):
            rows.append(dict(case_id=f'P_{solver}_{scenario}', network='P', suite_id=None,
                             family=None, effective_method='P/' + solver, solver=solver,
                             scenario=scenario, reuse_candidate=True))
    for sid in SUITES:
        for family in FAMILIES:
            for scenario in ('open', 'i95'):
                rows.append(dict(case_id=f'AC_{sid}_{family}_{scenario}', network='AC',
                    suite_id=sid, family=family, effective_method=LABELS[family],
                    solver='MSA' if family == 'XGBOOST' else 'FW', scenario=scenario,
                    reuse_candidate=family == 'FFTT'))
    return rows


def _bound_metadata(ctx, path, role):
    """Root supplies immutable baseline metadata pins; payloads have native identities."""
    path = Path(path).resolve(strict=True)
    bindings = {str(Path(p).resolve()): value for p, value in ctx.c.get('source_bindings', {}).items()}
    if str(path) not in bindings:
        raise ValueError('Baseline metadata absent from explicit source bindings: ' + str(path))
    ctx.bind(path, role, bindings[str(path)])
    return read_json(path)


def read_portfolios(ctx, path):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(Path(ctx.root).resolve()):
        raise ValueError('Portfolio freeze must belong to this new campaign')
    digest = ctx.bind(path, 'FROZEN_RECOVERY_PORTFOLIOS')
    data = read_json(path); payload = dict(data); identity = payload.pop('identity', None)
    if (data.get('schema') != SCHEMA or identity != _digest(payload)
            or data.get('phase_id') != ctx.c['phase_id'] or data.get('run_id') != ctx.c['run_id']
            or Path(data.get('source_root', '')).resolve() != Path(ctx.source_root).resolve()):
        raise ValueError('Portfolio freeze identity differs')
    keys = [(p['suite_id'], p['requested_family']) for p in data['portfolios']]
    if len(keys) != 20 or set(keys) != {(s, f) for s in SUITES for f in FAMILIES}:
        raise ValueError('Not the complete five-suite/four-family portfolio freeze')
    return data, digest


def query_cap(row, settings, remaining_seconds, remaining_rows):
    """Reserve declared initial caps for remaining jobs before granting extra time.

    This is allocation arithmetic, not an iteration-to-gap forecast. Initial caps
    include input/model loading. No query cap exceeds the historical 2700 seconds.
    """
    base = settings['cap_seconds']
    reserve = float(settings['sealing_reserve_seconds'])
    maximum = float(settings['per_query_seconds'])
    if not 0 < maximum <= 2700 or not math.isfinite(reserve) or reserve < 0 or not math.isfinite(remaining_seconds):
        raise ValueError('Invalid complete-query budget')
    for family in ('POWER4', 'ICNN', 'XGBOOST'):
        value = base[family]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError('Invalid declared family cap')
    own = float(base[row['family']])
    protected = sum(float(base[r['family']]) for r in remaining_rows if not r['reuse_candidate'])
    available = max(0., float(remaining_seconds) - reserve)
    # Do not starve the later cases when remaining wall is below the initial sum.
    total = own + protected
    if available < total:
        return own * available / total if total else 0.
    released = available - total
    requested_extra = float(settings.get('released_slack_fraction', 0.)) * released
    if not 0 <= settings.get('released_slack_fraction', 0.) <= 1:
        raise ValueError('Invalid slack fraction')
    return min(maximum, own + requested_extra)


def validate_reused_metadata(row, result, aggregate_row, expected_graph, tolerance):
    if result != aggregate_row:
        raise ValueError('Historical individual result differs from bound aggregate')
    old_case = result.get('case', {})
    for key in ('case_id', 'network', 'suite_id', 'family', 'solver', 'scenario'):
        if old_case.get(key) != row.get(key):
            raise ValueError('Historical case/solver differs: ' + key)
    if (result.get('status') != 'CONVERGED' or result.get('final_certificate_fresh') is not True
            or result.get('graph_identity') != expected_graph
            or result.get('nominal_tolerance') != tolerance
            or result.get('method') != row['solver']):
        raise ValueError('Historical result is not the required converged control')
    certificate = result.get('certificate') or {}
    rg = certificate.get('RG_SPTT')
    if (certificate.get('fresh') is not True or not _sha(certificate.get('state_hash'))
            or (rg is not None and (not math.isfinite(rg) or rg > tolerance))
            or not converged(certificate, tolerance)):
        raise ValueError('Historical certificate invalid')


def validate_native_state(result, flow, served, reference_served, closed_mask, *, closed):
    if (flow.ndim != 1 or len(flow) != len(closed_mask) or not np.isfinite(flow).all()
            or np.any(flow < 0)):
        raise ValueError('Invalid native final flow')
    cert = result.get('certificate') or {}
    if cert.get('state_hash') != array_hash({'flow': flow}):
        raise ValueError('Certificate belongs to another native state')
    if (served.dtype != np.bool_ or reference_served.dtype != np.bool_
            or not np.array_equal(served, reference_served)):
        raise ValueError('Exact within-scenario served OD membership differs')
    if closed and np.any(flow[np.asarray(closed_mask, bool)] != 0):
        raise ValueError('Closed represented backbone has positive flow')


def _artifacts(ctx, directory):
    result = {}
    for name in ('RESULT.json', 'final_flow.npz', 'coverage.npz', 'COVERAGE.json',
                 'HISTORY.json', 'CERTIFICATE.json', 'STAGES.json', 'certified_flow.npz'):
        path = Path(directory) / name
        if path.exists():
            result[name] = dict(path=str(path.resolve()), sha256=ctx.bind(path, 'ASSIGNMENT_NATIVE_' + name),
                                bytes=path.stat().st_size)
    return result


def _common(ctx):
    from partC_rev1_src.pacnhn_v1.topology import load_common
    base = Path(ctx.source_root) / 'topology'
    _bound_metadata(ctx, base / 'RESULT.json', 'IMMUTABLE_P_TOPOLOGY')
    for name in ('common_graph.npz', 'common_od.npz', 'partitions.npz'):
        ctx.bind(base / name, 'IMMUTABLE_COMMON_P_PAYLOAD')
    # Recomputes full graph, OD and partition array identities against bound RESULT.
    return load_common(ctx.source_root)


def _scenario_reference(ctx, scenario, *, original_od=None):
    from partC_rev1_src.pacnhn_v1.oracle import Demand
    root = Path(ctx.source_root) / 'references' / scenario
    result = _bound_metadata(ctx, root / 'RESULT.json', 'IMMUTABLE_P_REFERENCE')
    if result.get('status') != 'CONVERGED' or not result.get('final_certificate_fresh'):
        raise ValueError('Frozen reference is not converged/certified')
    cert = result.get('certificate') or {}
    if cert.get('fresh') is not True or not converged(cert, 1e-4):
        raise ValueError('Reference residual exceeds frozen tolerance')
    ctx.bind(root / 'coverage.npz', 'IMMUTABLE_REFERENCE_SERVED_MASK')
    with np.load(root / 'coverage.npz', allow_pickle=False) as z:
        served = z['serve']
    if original_od is None:
        _, original_od, _, _ = _common(ctx)
    if served.dtype != np.bool_ or served.shape != original_od.serve.shape:
        raise ValueError('Reference served mask dimensions/type differ')
    actual = Demand(original_od.origins, original_od.offsets, original_od.destinations,
                    original_od.demand, served)
    if actual.identity != result['OD_identity']:
        raise ValueError('Reference mask is not its certified exact OD population')
    return result, served


def reuse_case(ctx, row, aggregate, support, *, deadline, common=None):
    ctx.require_action('assignments')
    if not row['reuse_candidate']:
        raise ValueError('Learned/changed portfolio cannot use baseline reuse path')
    ctx.guard(deadline=deadline)
    source = Path(ctx.source_root)
    if common is None:
        common = _common(ctx)
    graph_p, od, p_arrays, _ = common
    path = source / 'assignments' / row['case_id']
    result = read_json(path / 'RESULT.json')
    found = [r for r in aggregate if r.get('case', {}).get('case_id') == row['case_id']]
    if len(found) != 1:
        raise ValueError('Historical aggregate missing/duplicate case')
    if row['network'] == 'P':
        report = _bound_metadata(ctx, source / 'topology' / 'RESULT.json', 'IMMUTABLE_P_TOPOLOGY')
        if report['graph_identity'] != graph_p.identity:
            raise ValueError('Reused P graph identity differs')
        arrays = p_arrays
    else:
        report = next(x for x in support['suites'] if x['suite_id'] == row['suite_id'])
        graph_path = source / 'acnhn' / row['suite_id'] / 'graph.npz'
        ctx.bind(graph_path, 'IMMUTABLE_AC_GRAPH', report['graph_sha256'])
        from partC_rev1_src.pacnhn_v1.oracle import Network
        with np.load(graph_path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        graph = Network(arrays['tail'], arrays['head'], arrays['blocked'], arrays['edge_ids'], arrays['node_ids'])
        validate_graph_arrays(arrays, p_arrays, report, od.identity, graph.identity)
    validate_reused_metadata(row, result, found[0], report['graph_identity'], ctx.c['assignment']['RG_SPTT'])
    reference, reference_served = _scenario_reference(ctx, row['scenario'], original_od=od)
    if result['OD_identity'] != reference['OD_identity']:
        raise ValueError('Historical served OD identity differs from reference')
    closed_mask = arrays['closed_mask']
    with np.load(path / 'final_flow.npz', allow_pickle=False) as z:
        flow = z['flow']
    with np.load(path / 'coverage.npz', allow_pickle=False) as z:
        served = z['serve']
    validate_native_state(result, flow, served, reference_served, closed_mask, closed=row['scenario'] == 'i95')
    history = read_json(path / 'HISTORY.json')
    if (not history or history[-1]['state_hash'] != result['certificate']['state_hash']
            or history[-1]['iteration'] != result['reported_state_iteration']):
        raise ValueError('Historical history/certificate state differs')
    artifacts = _artifacts(ctx, path)
    ctx.guard(deadline=deadline)
    return dict(result, case=row, execution_status='REUSED_VERIFIED_IMMUTABLE_RESULT',
                reused_from=str(path), native_artifacts=artifacts,
                historical_query_elapsed_seconds=result['elapsed_seconds'],
                reuse_basis='Identical graph/cost/OD/mask/solver/tolerance; native state and served membership checked',
                new_scientific_execution=False)


def run_case(ctx, row, portfolios, portfolio_sha256, *, deadline, cap_seconds):
    admission = ctx.require_action('assignments')
    if row['reuse_candidate'] or not 0 < cap_seconds <= 2700:
        raise ValueError('New query requires an explicit bounded non-reuse case')
    started = time.time(); end = min(deadline, started + cap_seconds, admission['science_deadline_epoch'])
    target = output_path(ctx, 'assignments/' + row['case_id'])
    if target.exists():
        raise FileExistsError('Existing query evidence cannot be silently overwritten or restarted')
    intent = dict(case=row, phase_id=ctx.c['phase_id'], run_id=ctx.c['run_id'],
        tasks=['LOAD_FROZEN_PORTFOLIO', 'COLD_FEASIBLE_INITIALIZATION', 'SOLVE', 'FRESH_CERTIFICATE'],
        expected_outputs=['RESULT.json', 'final_flow.npz', 'coverage.npz', 'HISTORY.json'],
        acceptance_tests=['B_OD_MASK_INVARIANCE', 'EXACT_SERVED_MEMBERSHIP', 'CONSERVATION', 'FRESH_STATE'],
        resources=ctx.c['resources'], portfolio_sha256=portfolio_sha256,
        complete_query_start_epoch=started, deadline_epoch=end, cap_seconds=cap_seconds,
        initialization='COLD_FEASIBLE_AON', dynamic_fallback=False)
    write_json(output_path(ctx, 'intents/' + row['case_id'] + '.json'), intent)
    try:
        ctx.guard(deadline=end)
        graph, od, arrays, objective = load_objective(ctx, row['suite_id'], row['family'], portfolios, deadline=end)
    except (TimeoutError, MemoryError) as exc:
        result = dict(case=row, status='TIME_LIMIT_DURING_INPUT_LOAD' if isinstance(exc, TimeoutError) else 'UNAVAILABLE_CURRENT_RAM',
                      final_certificate_fresh=False, elapsed_seconds=time.time() - started,
                      complete_query_start_epoch=started, deadline_epoch=end, intent=intent,
                      error_type=type(exc).__name__, error=str(exc), new_scientific_execution=False)
        write_json(target / 'RESULT.json', result)
        return result
    intent.update(graph_identity=graph.identity, original_OD_identity=od.identity)
    write_json(output_path(ctx, 'intents/' + row['case_id'] + '.json'), intent)
    enabled = ~arrays['closed_mask'] if row['scenario'] == 'i95' else np.ones(len(graph.tail), bool)
    result = solve(graph, od, objective, target, method=row['solver'], run_start_epoch=started,
                   deadline_epoch=end, tolerance=ctx.c['assignment']['RG_SPTT'], ctx=ctx,
                   enabled=enabled, workers=ctx.c['resources']['regional_workers'], capture_witness=False)
    result.update(case=row, intent=intent, execution_status='EXECUTED_RECOVERY_PORTFOLIO',
                  portfolio_sha256=portfolio_sha256, new_scientific_execution=True)
    if result.get('final_certificate_fresh'):
        reference, reference_served = _scenario_reference(ctx, row['scenario'], original_od=od)
        with np.load(target / 'coverage.npz', allow_pickle=False) as z:
            served = z['serve']
        with np.load(target / 'final_flow.npz', allow_pickle=False) as z:
            flow = z['flow']
        validate_native_state(result, flow, served, reference_served, arrays['closed_mask'], closed=row['scenario'] == 'i95')
        if result['OD_identity'] != reference['OD_identity']:
            raise ValueError('New assignment served OD identity differs')
    write_json(target / 'RESULT.json', result)
    # RESULT cannot contain its own digest. The caller's aggregate indexes it.
    result['native_artifacts'] = _artifacts(ctx, target)
    del graph, od, arrays, objective
    gc.collect(); ctx.flush_bindings()
    return result


def run(ctx, portfolios_path, *, deadline):
    admission = ctx.require_action('assignments')
    deadline = min(deadline, admission['science_deadline_epoch'])
    ctx.guard(deadline=deadline)
    if output_path(ctx, 'ASSIGNMENT_MATRIX.json').exists():
        raise FileExistsError('Existing matrix requires explicit recovery; never overwrite its evidence')
    start = time.time(); portfolios, portfolio_sha = read_portfolios(ctx, portfolios_path)
    source = Path(ctx.source_root)
    support = _bound_metadata(ctx, source / 'SUPPORT_FREEZE.json', 'IMMUTABLE_SUPPORT')
    if portfolios['support_freeze_sha256'] != sha(source / 'SUPPORT_FREEZE.json'):
        raise ValueError('Portfolio/support mismatch')
    aggregate = _bound_metadata(ctx, source / 'ASSIGNMENT_RESULTS.json', 'IMMUTABLE_COMPARISON_OUTCOMES')
    rows = matrix(); write_json(output_path(ctx, 'ASSIGNMENT_MATRIX.json'), rows)
    results = []
    common = _common(ctx)
    # Validate all immutable baselines first; this is metadata/native-state read work.
    for row in [r for r in rows if r['reuse_candidate']]:
        t = time.time()
        result = reuse_case(ctx, row, aggregate, support, deadline=deadline, common=common)
        result['reuse_admission_elapsed_seconds'] = time.time() - t
        target = output_path(ctx, 'assignments/' + row['case_id'] + '/RESULT.json')
        if target.exists():
            raise FileExistsError('Existing reuse receipt cannot be replaced')
        write_json(target, result); results.append(result)
    del common
    gc.collect()
    new_rows = [r for r in rows if not r['reuse_candidate']]
    for index, row in enumerate(new_rows):
        remaining = deadline - time.time()
        cap = query_cap(row, ctx.c['assignment'], remaining, new_rows[index+1:])
        if cap <= 0 or remaining <= ctx.c['assignment']['sealing_reserve_seconds']:
            result = dict(case=row, status='UNRUN_STAGE_TIME_LIMIT', final_certificate_fresh=False,
                          new_scientific_execution=False, elapsed_seconds=0.)
            write_json(output_path(ctx, 'assignments/' + row['case_id'] + '/RESULT.json'), result)
        else:
            result = run_case(ctx, row, portfolios, portfolio_sha, deadline=deadline,
                              cap_seconds=cap)
        results.append(result)
        write_json(output_path(ctx, 'ASSIGNMENT_RESULTS.json'), results)
    by_id = {r['case']['case_id']: r for r in results}
    if len(by_id) != 46 or set(by_id) != {r['case_id'] for r in rows}:
        raise ValueError('Incomplete computational matrix accounting')
    results = [by_id[r['case_id']] for r in rows]
    references = []
    for scenario in ('open', 'i95'):
        ref, _ = _scenario_reference(ctx, scenario)
        references.append(dict(ref, case=dict(case_id='P_REFERENCE_' + scenario, network='P',
            suite_id=None, family=None, solver='CFW', scenario=scenario),
            assignment_role='REFERENCE_SELF', execution_status='REUSED_FROZEN_REFERENCE',
            native_artifacts=_artifacts(ctx, source / 'references' / scenario), new_scientific_execution=False))
    write_json(output_path(ctx, 'ASSIGNMENT_RESULTS.json'), results)
    write_json(output_path(ctx, 'REPORTING_RESULTS.json'), references + results)
    terminal = dict(status='COMPLETE_WITH_RECORDED_OUTCOMES', phase_id=ctx.c['phase_id'],
        run_id=ctx.c['run_id'], rows=46, reporting_rows=48, reused_control_or_FFTT_rows=16,
        elapsed_seconds=time.time() - start, started_epoch=start, deadline_epoch=deadline,
        portfolio_sha256=portfolio_sha, statuses=dict(__import__('collections').Counter(r['status'] for r in results)),
        all_converged=all(r['status'] == 'CONVERGED' for r in results),
        note='Matrix completeness is separate from convergence, TRAIN qualification and learned-family coverage')
    write_json(output_path(ctx, 'ASSIGNMENT_STAGE_RESULT.json'), terminal)
    ctx.flush_bindings()
    return references + results
