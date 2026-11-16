"""Read-only native assignment evidence for the recovery publication.

No solver/model/label execution. The old audited metric and certificate readers
are reused, while this module authenticates the new 46-case/48-row ledger.
"""
from __future__ import annotations

from collections import Counter
import gc
import json
import math
import os
import hashlib
import stat
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd

from partC_rev1_src.pacnhn_v1.control import array_hash, read_json, sha
from partC_rev1_src.pacnhn_v1.metrics import (
    admit_assignment, backbone_frame, compare_native, load_frozen_keys, GEO_COLUMNS)
from partC_rev1_src.pacnhn_v1.final_assignment_metrics import independent_metrics, delta_statistics
from .assignments import matrix, read_portfolios
from . import assignment_reconciliation as repair
from .portfolio import SUITES, LABELS, validate_graph_arrays

SCHEMA = 'PACNHN_RECOVERY24_ASSIGNMENT_PUBLICATION_DATA_V1'
B_COUNT = 87582
KEY_COUNT = 200
SCENARIOS = {'open': 'OPEN', 'i95': 'REPRESENTED_I95'}
ADMITTED = {'CONVERGED', 'TIME_LIMIT', 'STAGNATION'}
UNAVAILABLE = {'TIME_LIMIT_DURING_INPUT_LOAD', 'UNAVAILABLE_CURRENT_RAM', 'UNRUN_STAGE_TIME_LIMIT', 'FAILED'}
BOUNDARY = ('Before graph/model load through final flow/trace/support exports; '
            'excludes final RESULT/STAGES/HISTORY writes and post-run admission')
LIMITATIONS = [
    'Fresh saved certificate/state identities are checked; no solver, shortest-path, conservation or model replay is performed.',
    'TIME_LIMIT and STAGNATION with fresh certificates are comparisons, not convergence claims.',
    'The common native 87,582 backbone coordinates and OPEN-frozen 200 IDs are retained, including zero-reference and closed links.',
    'Represented I95 omits hidden backbone-interior arcs; changes combine demand loss and redistribution, not pure rerouting.',
    'Demand and original source-row multiplicity reuse the authenticated exact OD membership audit; current masks are checked independently.',
    'Historical reused query times are separate from new measured query wall; no repeated execution or speed guarantee is implied.',
    'Network partition counts are frozen metadata; this export does not claim a new partition-stability experiment.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _new(path, value):
    # Serialize before opening: an invalid number must not create a success file.
    data = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(data)


def _csv(frame, path):
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        frame.to_csv(handle, index=False, float_format='%.17g')


def admitted(row):
    return row.get('status') in ADMITTED and row.get('final_certificate_fresh') is True


def case_metadata(case):
    reference = case['case_id'] == 'P_REFERENCE_' + case['scenario']
    role = 'REFERENCE_SELF' if reference else 'INDEPENDENT_CONTROL' if case['network'] == 'P' else 'SURROGATE'
    method = 'P/REFERENCE' if reference else 'P/' + case['solver'] if case['network'] == 'P' else case['suite_id'] + '/' + LABELS[case['family']]
    return dict(case_id=case['case_id'], method_id=method, network=case['network'],
        suite_id=case.get('suite_id'), family=case.get('family'), scenario=SCENARIOS[case['scenario']],
        solver=case['solver'], effective_method=method.split('/')[-1], assignment_role=role,
        reference_case_id='P_REFERENCE_' + case['scenario'])


def validate_ledger(rows, planned, computational, terminal, *, phase_id, run_id, revision=None):
    expected = repair.revised_matrix(matrix(), revision)
    require(planned == expected, 'Explicit 46-case matrix changed')
    require(terminal.get('status') == 'COMPLETE_WITH_RECORDED_OUTCOMES'
        and terminal.get('phase_id') == phase_id and terminal.get('run_id') == run_id
        and terminal.get('rows') == 46 and terminal.get('reporting_rows') == 48,
        'Assignment stage is not the current complete-accounting terminal')
    by_id = {r['case']['case_id']: r for r in rows}
    refs = {'P_REFERENCE_open', 'P_REFERENCE_i95'}
    require(len(rows) == len(by_id) == 48 and set(by_id) == refs | {r['case_id'] for r in expected},
        'Missing/extra/duplicate reporting case')
    require(len(computational) == 46 and computational == [by_id[r['case_id']] for r in expected],
        'Reporting and computational ledgers differ')
    for case in expected:
        row = by_id[case['case_id']]
        require(row['case'] == case and row['status'] in ADMITTED | UNAVAILABLE | (repair.UNAVAILABLE if revision is not None else set()), 'Case or terminal disposition differs')
        reused = case['reuse_candidate']
        if reused:
            require(row.get('execution_status') == 'REUSED_VERIFIED_IMMUTABLE_RESULT'
                and row.get('new_scientific_execution') is False and row['status'] == 'CONVERGED',
                'Historical control/FFTT reuse is not explicit')
        elif admitted(row):
            require(row.get('execution_status') == ('EXECUTED_FROZEN_PORTFOLIO_MEMORY_REPAIR'
                if case['case_id'] in repair.MAPPING.values() and revision is not None else 'EXECUTED_RECOVERY_PORTFOLIO')
                and row.get('new_scientific_execution') is True, 'Changed portfolio cannot be relabeled as reuse')
    for scenario in SCENARIOS:
        row = by_id['P_REFERENCE_' + scenario]
        require(row['case'] == dict(case_id='P_REFERENCE_' + scenario, network='P', suite_id=None,
            family=None, solver='CFW', scenario=scenario) and row.get('assignment_role') == 'REFERENCE_SELF'
            and row.get('execution_status') == 'REUSED_FROZEN_REFERENCE'
            and row.get('new_scientific_execution') is False and admitted(row) and row['status'] == 'CONVERGED',
            'Frozen reference-self row is missing or reinterpreted')
    require(terminal.get('statuses') == dict(Counter(r['status'] for r in computational))
        and terminal.get('all_converged') is all(r['status'] == 'CONVERGED' for r in computational),
        'Terminal convergence/count claims differ from cases')
    return [by_id['P_REFERENCE_' + s] for s in SCENARIOS] + [by_id[c['case_id']] for c in expected]


class Reader:
    def __init__(self, ctx, deadline, *, administrative_lineage=()):
        self.ctx, self.deadline, self.bindings = ctx, deadline, {}
        # Only the exact admission returned by successful repair.resolve is admitted.
        require(len(administrative_lineage) <= 1, 'Only reconciliation admission metadata may be external')
        self.administrative_lineage = {str(Path(item['path'])): dict(item) for item in administrative_lineage}

    def check(self):
        self.ctx.guard(deadline=self.deadline)

    def path(self, path, role, expected=None, *, pinned=False):
        self.check()
        p = Path(path).resolve(strict=True)
        require(p.is_relative_to(Path(self.ctx.root).resolve()) or p.is_relative_to(Path(self.ctx.source_root).resolve())
            or p.is_relative_to(Path(self.ctx.repo).resolve()), 'Input outside explicit campaign/source/repository')
        if pinned:
            pins = getattr(self.ctx, 'source_expected', {})
            pins = {**{os.path.normcase(str(Path(k).resolve())): v for k, v in self.ctx.c.get('source_bindings', {}).items()}, **pins}
            require(os.path.normcase(str(p)) in pins, 'Immutable metadata lacks a pre-recovery/CP pin: ' + str(p))
            expected = pins[os.path.normcase(str(p))] if expected is None else expected
        digest = self.ctx.bind(p, role, expected)
        prior = self.bindings.get(str(p))
        require(prior is None or prior['sha256'] == digest, 'Read input changed')
        self.bindings[str(p)] = dict(path=str(p), sha256=digest, bytes=p.stat().st_size, role=role)
        return p

    def administrative_json(self, item, role):
        """Record one explicitly authenticated reconciliation-admission JSON.

        This exception never applies to path(), json() or npz(). The caller's
        exact one-file allowlist comes from the independently resolved revision.
        """
        self.check()
        require(role == 'ASSIGNMENT_RECONCILIATION_ADMISSION', 'Unapproved administrative role')
        supplied = Path(item['path'])
        require(supplied.is_absolute() and supplied.resolve(strict=True) == supplied,
                'Administrative metadata path must be canonical and absolute')
        require(self.administrative_lineage.get(str(supplied)) == item,
                'Unlisted or substituted administrative metadata binding')
        require(supplied.name == 'RECONCILIATION_ADMISSION.json', 'Only reconciliation-admission JSON permitted')
        for node in (supplied, *supplied.parents):
            mode = node.lstat()
            require(not node.is_symlink() and not (getattr(mode, 'st_file_attributes', 0)
                & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024)), 'Administrative metadata alias/reparse path')
        expected = item.get('sha256')
        require(isinstance(expected, str) and len(expected) == 64
                and all(c in '0123456789abcdef' for c in expected), 'Mandatory exact administrative SHA256')
        def identity(st): return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        before = supplied.stat()
        require(stat.S_ISREG(before.st_mode) and before.st_size <= 32 * 1024**2, 'Bounded regular JSON required')
        with supplied.open('rb') as handle:
            first = os.fstat(handle.fileno()); raw = handle.read(32 * 1024**2 + 1); last = os.fstat(handle.fileno())
        after = supplied.stat()
        require(identity(before) == identity(first) == identity(last) == identity(after)
                and len(raw) == before.st_size, 'Administrative metadata changed while reading')
        require(hashlib.sha256(raw).hexdigest() == expected, 'Administrative metadata SHA mismatch')
        if 'bytes' in item: require(item['bytes'] == len(raw), 'Administrative metadata size mismatch')
        data = json.loads(raw, object_pairs_hook=repair._pairs, parse_constant=repair._reject_constant)
        require(isinstance(data, dict) and data.get('schema') == 'RECOVERY24_ASSIGNMENT_RECONCILIATION_ADMISSION_V1'
                and data.get('phase_id') == self.ctx.c['phase_id'] and data.get('run_id') == self.ctx.c['run_id']
                and data.get('output_relative') == repair.LEDGERS
                and data.get('tasks') == ['RECONCILE_TWO_MEMORY_REPAIR_ATTEMPTS'], 'Administrative lineage schema/scope differs')
        require(self.ctx.bind(supplied, role, expected) == expected, 'Administrative binding differs')
        prior = self.bindings.get(str(supplied))
        require(prior is None or prior['sha256'] == expected, 'Administrative input changed')
        self.bindings[str(supplied)] = dict(path=str(supplied), sha256=expected, bytes=len(raw), role=role)
        return data

    def json(self, path, role, expected=None, *, pinned=False):
        p = self.path(path, role, expected, pinned=pinned)
        require(p.stat().st_size <= 32 * 1024**2, 'Metadata exceeds bounded JSON reader')
        return read_json(p)

    def npz(self, path, fields, role, expected=None):
        p = self.path(path, role, expected)
        with zipfile.ZipFile(p) as archive:
            entries = archive.infolist()
            require(len(entries) == len({e.filename for e in entries}), 'Duplicate NPZ member')
            selected = entries if fields is None else [e for e in entries if e.filename in {f + '.npy' for f in fields}]
            require(sum(e.file_size for e in selected) <= 512 * 1024**2, 'Graph/flow payload exceeds bounded reader')
        with np.load(p, allow_pickle=False) as arrays:
            result = {k: arrays[k] for k in (arrays.files if fields is None else fields)}
        self.check()
        return result

    def unchanged(self):
        for row in list(self.bindings.values()):
            self.check()
            require(sha(row['path']) == row['sha256'], 'Input changed after reading: ' + row['path'])


def _native_bindings(reader, row, directory):
    bindings = row.get('native_artifacts') or {}
    for name in ('RESULT.json', 'final_flow.npz', 'coverage.npz', 'HISTORY.json'):
        require(name in bindings, 'Admitted native artifact missing: ' + name)
        binding = bindings[name]
        require(Path(binding['path']).resolve() == (directory / name).resolve(), 'Native artifact path substitution')
        p = reader.path(binding['path'], 'CERTIFIED_NATIVE_' + name, binding['sha256'])
        require(p.stat().st_size == binding['bytes'], 'Native artifact size changed')


def validate_result_relation(row, stored, receipt):
    """Account for deliberate aggregate-only fields without discarding science."""
    case = row['case']; reference = case_metadata(case)['assignment_role'] == 'REFERENCE_SELF'
    if reference:
        extra = {'case', 'assignment_role', 'execution_status', 'native_artifacts', 'new_scientific_execution'}
        require({k: v for k, v in row.items() if k not in extra} == stored, 'Reference row differs from frozen result')
        return
    aggregate_only = {'native_artifacts'}
    if case['reuse_candidate']:
        require(receipt == row, 'New reuse receipt differs from complete ledger')
        extra = {'case', 'execution_status', 'reused_from', 'native_artifacts', 'historical_query_elapsed_seconds',
            'reuse_basis', 'new_scientific_execution', 'reuse_admission_elapsed_seconds'}
        require({k: v for k, v in row.items() if k not in extra} == {k: v for k, v in stored.items() if k != 'case'},
            'Historical native scientific result differs from reuse ledger')
        for key in ('case_id', 'network', 'suite_id', 'family', 'solver', 'scenario'):
            require(stored['case'].get(key) == case.get(key), 'Reused native case changed')
        require(row['historical_query_elapsed_seconds'] == stored['elapsed_seconds'], 'Reused query time changed')
    else:
        require({k: v for k, v in row.items() if k not in aggregate_only} == stored and receipt == stored,
            'New result differs from complete ledger')


def validate_semantics(row, result, history, *, original_start, assignment_deadline):
    case = row['case']; reference = case_metadata(case)['assignment_role'] == 'REFERENCE_SELF'
    potential = case.get('family') != 'XGBOOST'
    require(result['method'] == case['solver'] and result['objective_has_potential'] is potential,
        'Solver/objective semantics changed')
    tolerance = 1e-4 if reference else 1e-3
    require(result['nominal_tolerance'] == tolerance, 'Nominal tolerance changed')
    cert = result['certificate']
    require(math.isclose(cert['arithmetic_allowance'], 1e-10 * max(1., abs(cert['TSTT']), abs(cert['SPTT'])),
        rel_tol=1e-12, abs_tol=1e-12), 'Certificate arithmetic allowance changed')
    require(history and history[-1]['iteration'] == result['reported_state_iteration']
        and history[-1]['state_hash'] == cert['state_hash'], 'History is not the reported fresh state')
    start, end, elapsed = (result[k] for k in ('complete_query_start_epoch', 'deadline_epoch', 'elapsed_seconds'))
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (start, end, elapsed))
        and 0 < end - start <= (10800 if reference else 2700) + 1e-6 and elapsed >= 0, 'Invalid complete query timing')
    if row.get('new_scientific_execution'):
        require(start >= original_start and end <= assignment_deadline, 'New query outside original recovery assignment budget')
        intent = result['intent']
        require(intent['complete_query_start_epoch'] == start and intent['deadline_epoch'] == end
            and intent['initialization'] == 'COLD_FEASIBLE_AON' and intent.get('dynamic_fallback') is False,
            'Query timing/initialization/fixed portfolio intent differs')


def timing_row(row):
    meta = case_metadata(row['case'])
    # Input-loading failures are still new measured attempts, even when no
    # scientific solver iteration ran. Reuse is a case disposition, not a flag.
    reused = meta['assignment_role'] == 'REFERENCE_SELF' or row['case'].get('reuse_candidate', False)
    measured = row.get('elapsed_seconds')
    start, end = row.get('complete_query_start_epoch'), row.get('deadline_epoch')
    has_interval = start is not None and measured is not None
    return dict(**meta, status=row['status'], execution_status=row.get('execution_status', 'NOT_EXECUTED'),
        category='COMPLETE_QUERY', seconds=measured, start_epoch=start,
        end_epoch=start + measured if has_interval else None,
        declared_deadline_epoch=end, cap_seconds=end - start if end is not None and start is not None else None,
        time_kind='HISTORICAL_MEASURED_WALL' if reused and has_interval else 'MEASURED_WALL' if has_interval else 'NO_MEASURED_QUERY',
        accounting_role='HISTORICAL_CONTEXT_ONLY' if reused else 'DISJOINT_INTERVAL' if has_interval else 'NOT_ADDITIVE',
        reuse_admission_seconds=row.get('reuse_admission_elapsed_seconds'),
        timing_boundary=(row.get('intent') or {}).get('timing_boundary',BOUNDARY),
        query_work_scope='PRE_SOLVE_REJECTION' if has_interval and not reused and row.get('new_scientific_execution') is False
            else 'NO_QUERY_PERFORMED' if not has_interval else 'RECORDED_QUERY',
        attempt_elapsed_seconds=row.get('attempt_elapsed_seconds'),
        attempt_start_epoch=row.get('disposition_recorded_epoch'),attempt_end_epoch=row.get('attempt_end_epoch'),
        post_query_seconds=row.get('post_query_seconds'),
        attempt_cost_scope=('QUERY_PLUS_POST_QUERY_VERIFICATION_RELEASE_AND_HASHING' if has_interval else 'ADMINISTRATIVE_UNRUN_DISPOSITION_ONLY') if row.get('attempt_elapsed_seconds') is not None
            else 'FULL_ATTEMPT_NOT_SEPARATELY_MEASURED',
        attempt_cost_accounting='ALTERNATIVE_TOTAL_INCLUDES_QUERY; DO_NOT_ADD_TO_QUERY_OR_STAGE_WALL',
        timing_end_scope='RECONSTRUCTED_FROM_RECORDED_START_PLUS_ELAPSED; NOT_PROCESS_EXIT' if has_interval else 'UNAVAILABLE',
        observed_cap_overrun_seconds=max(0., measured-(end-start)) if has_interval and end is not None else None)


def validate_time_intervals(rows):
    intervals = sorted((r['start_epoch'], r['end_epoch']) for r in rows if r['accounting_role'] == 'DISJOINT_INTERVAL')
    require(all(a <= b and (i == 0 or a >= intervals[i-1][1]) for i, (a, b) in enumerate(intervals)),
        'New complete-query intervals overlap; cannot add them')


def validate_membership(proof, open_mask, closed_mask):
    require(proof.get('status') == 'PASS' and proof.get('schema') == 'PACNHN_FROZEN_REFERENCE_I95_MEMBERSHIP_V1'
        and proof.get('demand_units') == 'ANNUAL_KTON' and proof.get('demand_renormalized') is False,
        'Invalid immutable membership proof')
    require(open_mask.dtype == closed_mask.dtype == np.dtype(bool)
        and open_mask.shape == closed_mask.shape == (proof['aggregate_OD_rows'],), 'OD membership dimensions/order differ')
    known = {r['population']: r for r in proof['populations']}
    names = ('COMMON_SERVED', 'LOST_CLOSED_VS_OPEN', 'GAINED_CLOSED_VS_OPEN', 'NEITHER_SERVED')
    require(len(known) == len(proof['populations']) == 4 and set(known) == set(names), 'Membership proof population ledger differs')
    for name in names:
        mask = (open_mask & closed_mask if name == names[0] else open_mask & ~closed_mask if name == names[1]
            else ~open_mask & closed_mask if name == names[2] else ~open_mask & ~closed_mask)
        require(int(mask.sum()) == known[name]['aggregate_OD_rows']
            and array_hash({'membership': mask}) == known[name]['membership_mask_identity'], 'Exact membership population changed')
    require(known[names[2]]['aggregate_OD_rows'] == 0 and proof.get('closed_served_subset_of_open') is True,
        'Represented closed served population is not a subset of OPEN')
    require(sum(r['original_OD_row_multiplicity'] for r in known.values()) == proof['original_OD_row_multiplicity']
        and math.isclose(math.fsum(r['demand'] for r in known.values()), proof['total_original_demand'], abs_tol=1e-7, rel_tol=1e-12),
        'Membership demand/source multiplicity does not reconcile')
    for scenario, mask in (('open', open_mask), ('i95', closed_mask)):
        require(int(mask.sum()) == proof['scenarios'][scenario]['served_rows'], 'Scenario served rows differ')
    return [dict(known[name], demand_basis='AUTHENTICATED_EXACT_FROZEN_OD_AUDIT; CURRENT_MASK_RECHECKED') for name in names]


def _graphs(reader, source, topo, support):
    from partC_rev1_src.pacnhn_v1.oracle import Network
    common = reader.npz(source / 'topology/common_graph.npz', None, 'IMMUTABLE_COMMON_GRAPH')
    require(array_hash(common) == topo['graph_array_identity'], 'Full common graph arrays differ')
    graphs, networks = {}, []
    for sid in ('P',) + SUITES:
        reader.check()
        report = topo if sid == 'P' else next(s for s in support['suites'] if s['suite_id'] == sid)
        arrays = common if sid == 'P' else reader.npz(source / 'acnhn' / sid / 'graph.npz', None,
            'IMMUTABLE_AC_GRAPH', report['graph_sha256'])
        graph = Network(arrays['tail'], arrays['head'], arrays['blocked'], arrays['edge_ids'], arrays['node_ids'])
        require(graph.identity == report['graph_identity'], 'Frozen graph identity differs')
        if sid != 'P':
            validate_graph_arrays(arrays, common, report, topo['OD_identity'], graph.identity)
        graphs[sid] = dict(identity=graph.identity, edge_ids=arrays['edge_ids'].copy(), closed_mask=arrays['closed_mask'].copy())
        require(graphs[sid]['closed_mask'].dtype == bool and not graphs[sid]['closed_mask'][B_COUNT:].any(), 'Invalid represented closure mask')
        partition = None if sid == 'P' else next(p for p in topo['partitions'] if p['suite_id'] == sid)
        networks.append(dict(suite_id=sid, resolution=None if sid == 'P' else partition['resolution'],
            nodes=len(graph.node_ids), edges=len(graph.tail), backbone_links=B_COUNT,
            regional_links=topo['residual_physical_arc_count'] if sid == 'P' else report['active_pairs'],
            regions=None if sid == 'P' else partition['region_count'], active_regions=report.get('active_regions'),
            active_pairs=report.get('active_pairs'), graph_identity=graph.identity, OD_identity=topo['OD_identity'],
            represented_closed_B_count=int(graphs[sid]['closed_mask'].sum()),
            partition_stability_scope='FROZEN_SOURCE_COUNTS_NO_NEW_PARTITION_EXPERIMENT'))
        del graph
        if sid != 'P':
            del arrays
    fixed = {k: common[k][:B_COUNT].copy() for k in ('edge_ids', 'base', 'k4', 'closed_mask')}
    del common
    gc.collect()
    return graphs, networks, fixed


def evaluate_assignments(ctx, assignment_results_path, *, deadline, assignment_revision=None):
    """Authenticate terminal REPORTING_RESULTS.json and create report_data once.

    Use the publication action budget. All 48 rows remain represented; unavailable
    states have empty metrics. No labels, predictions or optimization are run.
    """
    auth = ctx.require_action('publication')
    end = min(float(deadline), float(auth['science_deadline_epoch']))
    ctx.guard(end)
    root, source = Path(ctx.root).resolve(), Path(ctx.source_root).resolve()
    layout = repair.resolve(ctx, assignment_revision); revision = layout['revision']
    reader = Reader(ctx, end, administrative_lineage=[revision['admission']] if revision else ()); reader.check()
    require(Path(assignment_results_path).resolve() == layout['paths']['reporting'], 'Exact admitted 48-row ledger required')
    output = layout['metrics_root']
    require(not output.exists(), 'Existing reporting evidence cannot be overwritten')
    output.mkdir(parents=True)
    started = time.time()
    _new(output / 'INTENT.json', dict(schema=SCHEMA, phase_id=ctx.c['phase_id'], run_id=ctx.c['run_id'],
        step_id='NATIVE_ASSIGNMENT_REPORT_DATA', config_path=str(ctx.path), config_sha256=ctx.config_sha256,
        start_epoch=started, deadline_epoch=end, tasks=['AUTHENTICATE48_ROWS', 'VERIFY_NATIVE_CERTIFICATES',
        'RECOMPUTE_FULL_B_AND_FIXED_KEYS', 'COMPARE_REPRESENTED_I95', 'EXPORT_READABLE_DATA'],
        expected_outputs=['table2.csv', 'population_metrics.csv', 'backbone_links.csv', 'native_backbone_vectors.csv',
            'convergence.csv', 'networks.csv', 'timing.csv', 'i95_paired.csv', 'i95_links.csv', 'served_membership.csv', 'MANIFEST.json'],
        acceptance_tests=['EXACT48_ACCOUNTING', 'FULL87582_AND_FIXED200', 'CURRENT_NATIVE_BINDINGS', 'NO_MODEL_OR_SOLVER_EXECUTION'],
        resources=ctx.c['resources'], source_root=str(source)))
    try:
        reader.path(ctx.path, 'EXPLICIT_REPORTING_CONFIGURATION', ctx.config_sha256)
        if assignment_revision is not None:
            reader.path(assignment_revision['path'], 'ASSIGNMENT_REVISION', assignment_revision['sha256'])
            reader.administrative_json(revision['admission'], 'ASSIGNMENT_RECONCILIATION_ADMISSION')
            for item in [revision['runner_result'], *revision['originals'].values()]:
                reader.path(item['path'], 'PRESERVED_ASSIGNMENT_LINEAGE', item['sha256'])
        terminal = reader.json(layout['paths']['terminal'], 'TERMINAL_ASSIGNMENT_STAGE')
        rows = validate_ledger(reader.json(assignment_results_path, 'COMPLETE_REPORTING_LEDGER'),
            reader.json(layout['paths']['matrix'], 'FROZEN46_CASE_MATRIX'),
            reader.json(layout['paths']['computational'], 'COMPUTATIONAL46_LEDGER'), terminal,
            phase_id=ctx.c['phase_id'], run_id=ctx.c['run_id'], revision=revision)
        portfolios, portfolio_sha = read_portfolios(ctx, root / 'portfolios/PORTFOLIOS.json')
        reader.path(root / 'portfolios/PORTFOLIOS.json', 'FROZEN_PAIRED_PORTFOLIOS', portfolio_sha)
        require(terminal['portfolio_sha256'] == portfolio_sha, 'Terminal portfolio freeze differs')
        for module in (__file__, repair.__file__, Path(__file__).with_name('assignments.py'), Path(__file__).with_name('portfolio.py'),
            Path(__file__).parents[1] / 'pacnhn_v1/metrics.py', Path(__file__).parents[1] / 'pacnhn_v1/final_assignment_metrics.py'):
            reader.path(module, 'REPORTING_SOURCE')
        topo = reader.json(source / 'topology/RESULT.json', 'FROZEN_TOPOLOGY', pinned=True)
        support = reader.json(source / 'SUPPORT_FREEZE.json', 'FROZEN_SUPPORT', portfolios['support_freeze_sha256'])
        require(topo['status'] == support['status'] == 'PASS' and topo['backbone_count'] == B_COUNT
            and len(support['suites']) == len(SUITES) and {s['suite_id'] for s in support['suites']} == set(SUITES), 'Frozen topology/support scope differs')
        witness = reader.json(source / 'COMPLETE_WITNESS_AUDIT.json', 'COMPLETE_OD_WITNESS', pinned=True)
        require(witness['status'] == 'PASS' and witness['support_freeze_sha256'] == portfolios['support_freeze_sha256'], 'Witness/support binding differs')
        proof_root = source / 'audits/reference_i95_membership_v1'
        proof = reader.json(proof_root / 'REPORT.json', 'EXACT_REFERENCE_OD_PROOF', pinned=True)
        proof_sources = reader.json(proof_root / 'INPUT_MANIFEST.json', 'REFERENCE_OD_PROOF_SOURCES', pinned=True)
        require(proof['common_OD_identity'] == topo['OD_identity'] and proof['graph_identity'] == topo['graph_identity'], 'OD proof graph/population differs')
        proof_bindings = {str(Path(r['path']).resolve()): r['sha256'] for r in proof_sources['bindings']}
        for p in [source / 'topology/common_od.npz'] + [source / 'references' / s / n for s in SCENARIOS for n in ('RESULT.json', 'coverage.npz', 'final_flow.npz')]:
            reader.path(p, 'UNCHANGED_EXACT_OD_PROOF_INPUT', proof_bindings[str(p)])
        historical_rows = reader.json(source / 'ASSIGNMENT_RESULTS.json', 'FROZEN_HISTORICAL_COMPARISONS', pinned=True)
        historical = {r['case']['case_id']: r for r in historical_rows}
        require(len(historical) == len(historical_rows), 'Duplicate historical outcome')
        graphs, networks, fixed = _graphs(reader, source, topo, support)
        backbone = backbone_frame(pd.read_parquet(reader.path(source / 'topology/backbone.parquet', 'NATIVE_B_GEOGRAPHY'),
            columns=['link_id'] + GEO_COLUMNS))
        require(np.array_equal(backbone.link_id.to_numpy(str), fixed['edge_ids']), 'Geographic B IDs/order differ')
        references, native, table, populations, curves, timings = {}, {}, [], [], [], []
        keys = None
        for row in rows:
            reader.check(); case = row['case']; meta = case_metadata(case); key = case['case_id']; scenario = case['scenario']
            meta['logical_case_id'] = repair.logical_id(row)
            reference = meta['assignment_role'] == 'REFERENCE_SELF'
            receipt_directory, intent_path = repair.case_paths(root, row, revision)
            directory = source / 'references' / scenario if reference else source / 'assignments' / key if case['reuse_candidate'] else receipt_directory
            receipt = None if reference else reader.json(receipt_directory / 'RESULT.json', 'CURRENT_CASE_DISPOSITION')
            timings.append(dict(timing_row(row), logical_case_id=repair.logical_id(row), attempt_disposition='CURRENT_LOGICAL_CASE'))
            if not admitted(row):
                require(not reference and {k:v for k,v in row.items() if k != 'native_artifacts'} == receipt,
                    'Unavailable case differs from saved disposition')
                table.append(dict(**meta, status=row['status'], certificate_fresh=False,
                    execution_status=row.get('execution_status', 'NOT_EXECUTED'), comparison_available=False,
                    elapsed_seconds=row.get('elapsed_seconds'), wmape_backbone_percent=None, wmape_key_percent=None,
                    backbone_cost_error_percent=None, note=row.get('error', 'No admitted final state')))
                continue
            _native_bindings(reader, row, directory)
            graph = graphs['P' if case['network'] == 'P' else case['suite_id']]
            flow, stored, served, _ = admit_assignment(directory, expected_graph_identity=graph['identity'],
                expected_edge_ids=graph['edge_ids'], expected_OD_identity=proof['scenario_OD_identities'][scenario])
            validate_result_relation(row, stored, receipt)
            if not reference and case['reuse_candidate']:
                require(historical.get(key) == stored, 'Reused control/FFTT is not its pre-recovery frozen result')
            history = reader.json(directory / 'HISTORY.json', 'FRESH_CERTIFICATE_HISTORY')
            validate_semantics(row, stored, history, original_start=ctx.c['original_start_epoch'],
                assignment_deadline=ctx.c['action_deadlines']['assignments'])
            require(served.shape == (proof['aggregate_OD_rows'],), 'Served-mask aggregate population differs')
            if scenario == 'i95':
                require(np.all(flow[graph['closed_mask']] == 0), 'Represented closed native flow is nonzero')
            if row.get('new_scientific_execution'):
                intent = reader.json(intent_path, 'NEW_QUERY_INTENT')
                require(intent == stored['intent'] and intent['phase_id'] == ctx.c['phase_id'] and intent['run_id'] == ctx.c['run_id']
                    and intent['portfolio_sha256'] == stored['portfolio_sha256'] == portfolio_sha
                    and intent['original_OD_identity'] == topo['OD_identity'] and intent['graph_identity'] == graph['identity'],
                    'New query authority/portfolio/common OD differs')
            bflow = flow[:B_COUNT].copy(); del flow
            if reference:
                references[scenario] = (bflow, served, stored)
                if scenario == 'open':
                    keyroot = source / 'metrics/keys'
                    keymeta = reader.json(keyroot / 'KEY_SELECTION_FREEZE.json', 'OPEN_REFERENCE_ONLY_KEY_FREEZE', pinned=True)
                    for binding in keymeta['source_bindings']:
                        reader.path(binding['path'], 'IMMUTABLE_KEY_SELECTION_SOURCE', binding['sha256'])
                    reader.path(keyroot / 'key_links_200.csv', 'FROZEN200_KEYS', keymeta['key_file_sha256'])
                    reader.path(keyroot / 'open_backbone_reference.csv', 'FROZEN_KEY_REFERENCE_VECTOR', keymeta['reference_file_sha256'])
                    keys, checked = load_frozen_keys(keyroot, backbone, bflow, reference_id='P_REFERENCE_open')
                    require(checked == keymeta and keymeta['prediction_errors_used'] is False
                        and keymeta['key_selection_role'] == 'REFERENCE_ONLY' and keymeta['full_backbone_population_filtered'] is False,
                        'Frozen key selection scope differs')
            ref, refserved, refresult = references[scenario]
            require(np.array_equal(served, refserved), 'Exact within-scenario served population changed')
            stats, per = compare_native(backbone, ref, bflow, keys, scenario=SCENARIOS[scenario],
                reference_case_id='P_REFERENCE_' + scenario, case_id=key, method_id=meta['method_id'], assignment_role=meta['assignment_role'],
                reference_serve=refserved, predicted_serve=served, closed_mask=fixed['closed_mask'], base=fixed['base'], k4=fixed['k4'])
            independent, _ = independent_metrics(ref, bflow, np.isin(fixed['edge_ids'], keys.link_id), fixed['base'], fixed['k4'])
            for stat in stats:
                for field, value in independent[stat['population']].items():
                    actual = stat[field]
                    require(actual == value if value is None or isinstance(value, str) else math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-10),
                        'Independent metric arithmetic differs: ' + field)
            by_population = {s['population']: s for s in stats}; full = by_population['BACKBONE']; subset = by_population['KEY_LINKS']
            table.append(dict(**meta, status=stored['status'], certificate_fresh=True, comparison_available=True,
                execution_status=row['execution_status'], RG_SPTT=stored['certificate']['RG_SPTT'],
                residual_kind='FW_GAP' if stored['objective_has_potential'] else 'VI_RESIDUAL',
                elapsed_seconds=stored['elapsed_seconds'], nominal_tolerance=stored['nominal_tolerance'],
                query_cap_seconds=stored['deadline_epoch']-stored['complete_query_start_epoch'],
                wmape_backbone_percent=full['wmape_percent'], wmape_key_percent=subset['wmape_percent'],
                wmape_backbone_status=full['wmape_status'], wmape_key_status=subset['wmape_status'],
                backbone_cost_error_percent=full['backbone_cost_error_percent'], backbone_cost_status=full['backbone_cost_status'],
                reference_backbone_cost=full['reference_backbone_cost'], predicted_backbone_cost=full['predicted_backbone_cost'],
                full_B_links=B_COUNT, key_links=KEY_COUNT, reference_zero_link_count=full['reference_zero_link_count'],
                reference_zero_positive_prediction_count=full['reference_zero_positive_prediction_count'],
                served_OD_membership_equal=True, reference_status=refresult['status'], reference_RG_SPTT=refresult['certificate']['RG_SPTT'],
                reported_state_iteration=stored['reported_state_iteration'],
                timing_boundary=(stored.get('intent') or {}).get('timing_boundary',BOUNDARY)))
            populations.extend(stats)
            per = per.rename(columns={'lambda_reference': 'reference_flow', 'lambda_estimate': 'predicted_flow', 'is_key_link': 'is_key'})
            for column in GEO_COLUMNS:
                per[column] = backbone[column].to_numpy()
            per.to_csv(output / 'backbone_links.csv', index=False, mode='a' if native else 'x', header=not native, float_format='%.17g')
            native[key] = bflow
            curves.extend(dict(case_id=key, method_id=meta['method_id'], scenario=meta['scenario'], iteration=h['iteration'],
                elapsed_seconds=h['elapsed_seconds'], RG_SPTT=h['RG_SPTT'], raw_signed_gap=h['raw_signed_gap'],
                state_hash=h['state_hash'], timing_kind=timings[-1]['time_kind'], residual_kind=table[-1]['residual_kind']) for h in history)
            del per, served
        membership = validate_membership(proof, references['open'][1], references['i95'][1])
        if revision is not None:
            failures = reader.json(revision['preserved_failures']['path'], 'PRESERVED_FAILED_QUERY_TIMING', revision['preserved_failures']['sha256'])
            timings.extend(dict(timing_row(item['outcome']), logical_case_id=item['logical_case_id'],
                attempt_disposition='PRESERVED_ORIGINAL_FAILED_ATTEMPT') for item in failures)
        validate_time_intervals(timings)
        known = {r['population']: r for r in membership}
        coverage = dict(lost_OD_rows=known['LOST_CLOSED_VS_OPEN']['aggregate_OD_rows'],
            gained_OD_rows=known['GAINED_CLOSED_VS_OPEN']['aggregate_OD_rows'],
            lost_OD_demand=known['LOST_CLOSED_VS_OPEN']['demand'], gained_OD_demand=known['GAINED_CLOSED_VS_OPEN']['demand'],
            open_closed_OD_membership_equal=bool(np.array_equal(references['open'][1], references['i95'][1])),
            demand_units='ANNUAL_KTON', demand_renormalized=False)
        methods = {}
        for row in rows:
            methods.setdefault(case_metadata(row['case'])['method_id'], {})[row['case']['scenario']] = row['case']['case_id']
        pairs = []; wrote_pairs = False
        for method, cases in methods.items():
            reader.check(); require(set(cases) == set(SCENARIOS), 'Unpaired reporting method')
            pair = dict(method_id=method, open_case_id=cases['open'], closed_case_id=cases['i95'], **coverage)
            if not all(k in native for k in cases.values()):
                pairs.append(dict(pair, comparison_available=False, status='UNAVAILABLE_PAIRED_NATIVE_STATE'))
                continue
            oo, cc = native[cases['open']], native[cases['i95']]
            pairs.append(dict(pair, comparison_available=True, status='VERIFIED_PAIRED_NATIVE_STATE', **delta_statistics(oo, cc)))
            frame = backbone[['link_id', 'native_index'] + GEO_COLUMNS].copy()
            frame['method_id'] = method; frame['open_flow'] = oo; frame['closed_flow'] = cc
            frame['delta'] = cc-oo; frame['is_closed'] = fixed['closed_mask']
            frame.to_csv(output / 'i95_links.csv', mode='a' if wrote_pairs else 'x', header=not wrote_pairs, index=False, float_format='%.17g')
            wrote_pairs = True
        require(len(pairs) == 24, 'Incomplete paired method accounting')
        wide = backbone[['link_id', 'native_index'] + GEO_COLUMNS].copy()
        wide['is_key'] = np.isin(fixed['edge_ids'], keys.link_id); wide['represented_I95_closed'] = fixed['closed_mask']
        for key, values in native.items():
            wide['flow__' + key] = values
        for name, frame in [('table2', pd.DataFrame(table)), ('population_metrics', pd.DataFrame(populations)),
            ('native_backbone_vectors', wide), ('convergence', pd.DataFrame(curves)), ('timing', pd.DataFrame(timings)),
            ('networks', pd.DataFrame(networks)), ('i95_paired', pd.DataFrame(pairs)), ('served_membership', pd.DataFrame(membership))]:
            reader.check(); _csv(frame, output / (name + '.csv'))
        _new(output / 'I95_OD_COVERAGE.json', coverage)
        reader.unchanged(); reader.check()
        _new(output / 'PROVENANCE.json', dict(schema=SCHEMA, bindings=list(reader.bindings.values()),
            assignment_revision=assignment_revision, portfolio_sha256=portfolio_sha, common_backbone_order_identity=array_hash({'link_id': fixed['edge_ids']}),
            immutable_reference_membership_proof_sha256=reader.bindings[str(proof_root / 'REPORT.json')]['sha256'],
            limitations=LIMITATIONS, units={'flow': 'ANNUAL_KTON', 'backbone_cost': 'ANNUAL_KTON_SECONDS', 'WMAPE': 'PERCENT', 'RG_SPTT': 'FRACTION'},
            model_inference=False, label_access=False, solver_replay=False))
        _new(output / 'RESULT.json', dict(schema=SCHEMA, status='PASS_WITH_EXPLICIT_CASE_DISPOSITIONS',
            phase_id=ctx.c['phase_id'], run_id=ctx.c['run_id'], reporting_rows=48, computational_cases=46,
            assignment_revision=assignment_revision, timing_attempt_rows=len(timings),
            timing_row_scope='48 logical reporting dispositions plus two preserved original failures' if revision else '48 logical reporting dispositions',
            non_reference_timing_dispositions=sum(r['assignment_role']!='REFERENCE_SELF' for r in timings),
            separately_measured_full_attempt_rows=sum(r['attempt_elapsed_seconds'] is not None for r in timings),
            separately_measured_full_attempt_seconds=math.fsum(r['attempt_elapsed_seconds'] for r in timings if r['attempt_elapsed_seconds'] is not None),
            separately_measured_post_query_seconds=math.fsum(r['post_query_seconds'] for r in timings if r['post_query_seconds'] is not None),
            attempt_cost_scope='Known measured subset; full attempt totals include native query seconds and must not be added to query or stage wall',
            available_native_cases=len(native), unavailable_cases=48-len(native), full_B_count=B_COUNT, key_count=KEY_COUNT,
            paired_method_rows=len(pairs), statuses=dict(Counter(r['status'] for r in rows)),
            all_computational_cases_converged=terminal['all_converged'],
            historical_query_seconds=math.fsum(r['seconds'] for r in timings if r['time_kind'] == 'HISTORICAL_MEASURED_WALL'),
            new_complete_query_seconds=math.fsum(r['seconds'] for r in timings if r['accounting_role'] == 'DISJOINT_INTERVAL'),
            started_epoch=started, elapsed_seconds=time.time()-started, deadline_epoch=end, limitations=LIMITATIONS))
        artifacts = []
        for path in sorted(output.iterdir()):
            reader.check()
            if path.is_file():
                artifacts.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
        ctx.flush_bindings(); reader.check()
        manifest = dict(schema=SCHEMA, status='VERIFIED_REPORT_DATA', files=artifacts,
            reporting_rows=48, assignment_revision=assignment_revision, numerical_execution=False, export_elapsed_seconds=time.time()-started)
        _new(output / 'MANIFEST.json', manifest)
        return dict(manifest, manifest_path=str(output / 'MANIFEST.json'), manifest_sha256=sha(output / 'MANIFEST.json'))
    except Exception as exc:
        _new(output / 'FAILURE.json', dict(schema=SCHEMA, status='FAILED_PRESERVED_PARTIAL_REPORT_DATA',
            error_type=type(exc).__name__, error=str(exc), epoch=time.time(), deadline_epoch=end))
        raise
