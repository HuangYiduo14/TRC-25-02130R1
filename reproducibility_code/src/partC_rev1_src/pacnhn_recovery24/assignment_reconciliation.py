"""Explicit two-attempt memory-repair lineage; metadata only, no automatic retries.

Original ledgers/receipts remain immutable. Native replacement case IDs are never
renamed: the revision matrix and logical_case_id field express the mapping.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import time

PORTFOLIO_SHA = 'dc963991270ef3819fd518bec4df1181eb493778657605aab75f6fef6a7e227e'
BASE = 'assignment_revisions/xgb_s03_memory_v1'
LEDGERS = BASE + '/reporting_v2/ledgers'
METRICS = 'report_data_revisions/xgb_s03_memory_v2'
TIMING = 'report_timing/xgb_s03_memory_v2'
MAPPING = {f'AC_s03_XGBOOST_{s}': f'AC_s03_XGBOOST_{s}_memory_v1' for s in ('open', 'i95')}
FILES = dict(matrix='ASSIGNMENT_MATRIX.json', computational='ASSIGNMENT_RESULTS.json',
             reporting='REPORTING_RESULTS.json', terminal='ASSIGNMENT_STAGE_RESULT.json')
NATIVE = {'RESULT.json', 'final_flow.npz', 'coverage.npz', 'HISTORY.json', 'CERTIFICATE.json'}
NATIVE_ALLOWED = NATIVE | {'COVERAGE.json', 'STAGES.json', 'certified_flow.npz'}
ADMITTED = {'CONVERGED', 'TIME_LIMIT', 'STAGNATION'}
UNAVAILABLE = {'TIME_LIMIT_DURING_INPUT_LOAD', 'UNAVAILABLE_CURRENT_RAM', 'UNRUN_STAGE_TIME_LIMIT',
    'TIME_LIMIT_DURING_REPAIR', 'TIME_LIMIT_DURING_FINAL_VALIDATION', 'FAILED_MEMORY_REPAIR',
    'FAILED_MODEL_RELEASE', 'FAILED_RESOURCE_GUARD', 'FAILED_INTEGRITY_GUARD', 'UNRUN_PRIOR_SAFETY_FAILURE'}
UNRUN = {'UNRUN_STAGE_TIME_LIMIT', 'UNRUN_PRIOR_SAFETY_FAILURE'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    p = Path(path).resolve(strict=True)
    return dict(path=str(p), sha256=digest(p), bytes=p.stat().st_size)


def read_bound(ctx, item, canonical=None):
    p = Path(item['path']).resolve(strict=True)
    if canonical is not None:
        require(p == Path(canonical).resolve(), 'Unexpected metadata path: ' + str(p))
    require(p.is_file(), 'Metadata must be a regular file')
    ctx.guard()
    require(ctx.bind(p, 'ASSIGNMENT_REVISION_METADATA', item['sha256']) == item['sha256'], 'Metadata SHA mismatch')
    if 'bytes' in item:
        require(p.stat().st_size == item['bytes'], 'Metadata size mismatch')
    require(p.stat().st_size <= 64 * 1024**2, 'Metadata exceeds bounded JSON reader')
    raw = p.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == item['sha256'], 'Metadata changed while reading')
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_reject_constant)


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate metadata key: ' + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def _same_binding(left, right):
    return (Path(left['path']).resolve() == Path(right['path']).resolve()
            and left['sha256'] == right['sha256'])


def revised_matrix(original, revision):
    if revision is None:
        return original
    rows = copy.deepcopy(original)
    require({r['logical_case_id']: r['attempt_case_id'] for r in revision['attempts']} == MAPPING,
            'Only the frozen two-case repair mapping is permitted')
    for case in rows:
        case['case_id'] = MAPPING.get(case['case_id'], case['case_id'])
    return rows


def logical_id(row):
    native = row['case']['case_id']
    inverse = {v: k for k, v in MAPPING.items()}
    if native in inverse:
        require(row.get('logical_case_id') == inverse[native], 'Repair logical identity missing')
        return inverse[native]
    require(row.get('logical_case_id', native) == native, 'Unexpected logical remapping')
    return native


def case_paths(root, row, revision):
    key = row['case']['case_id']
    if key in MAPPING.values():
        require(revision is not None, 'Repair result requires explicit revision authority')
        entry = next(x for x in revision['attempts'] if x['attempt_case_id'] == key)
        return Path(entry['result']['path']).parent, Path(entry['intent']['path'])
    return Path(root) / 'assignments' / key, Path(root) / 'intents' / (key + '.json')


def _validated(ctx, admission):
    """Independently rebuild the exact revision from immutable native metadata."""
    root = Path(ctx.root).resolve()
    require(admission.get('schema') == 'RECOVERY24_ASSIGNMENT_RECONCILIATION_ADMISSION_V1'
            and admission.get('phase_id') == ctx.c['phase_id'] and admission.get('run_id') == ctx.c['run_id'],
            'Explicit reconciliation authority required')
    require(_same_binding(admission['campaign_config'], dict(path=str(ctx.path), sha256=ctx.config_sha256)),
            'Frozen campaign configuration changed')
    read_bound(ctx, admission['campaign_config'], ctx.path)
    require(admission.get('output_relative') == LEDGERS and admission.get('tasks') == ['RECONCILE_TWO_MEMORY_REPAIR_ATTEMPTS']
            and admission.get('acceptance_tests') and admission.get('expected_outputs'), 'Explicit bounded reconciliation scope required')
    own = str(Path(__file__).resolve())
    require(admission.get('source_bindings', {}).get(own) == digest(__file__), 'Unadmitted reconciliation source')
    for path, sha in admission['source_bindings'].items():
        ctx.guard(); require(digest(path) == sha, 'Reconciliation dependency changed')
    originals = admission['originals']
    require(set(originals) == set(FILES), 'All four original ledgers required')
    old = {key: read_bound(ctx, originals[key], root / name) for key, name in FILES.items()}
    expected_ids = [f'P_{solver}_{s}' for solver in ('CFW', 'FW', 'MSA') for s in ('open', 'i95')]
    expected_ids += [f'AC_{suite}_{family}_{s}' for suite in ('s02', 's03', 's04', 's05', 's06')
                     for family in ('FFTT', 'POWER4', 'ICNN', 'XGBOOST') for s in ('open', 'i95')]
    require([c['case_id'] for c in old['matrix']] == expected_ids, 'Original matrix changed')
    require(len(old['computational']) == 46 and [r['case'] for r in old['computational']] == old['matrix'], 'Original46 mismatch')
    require(len(old['reporting']) == 48 and old['reporting'][2:] == old['computational']
            and [r['case']['case_id'] for r in old['reporting'][:2]] == ['P_REFERENCE_open', 'P_REFERENCE_i95'], 'Original48 mismatch')
    t = old['terminal']
    require(t.get('status') == 'COMPLETE_WITH_RECORDED_OUTCOMES' and t.get('phase_id') == ctx.c['phase_id']
            and t.get('run_id') == ctx.c['run_id'] and t.get('rows') == 46 and t.get('reporting_rows') == 48
            and t.get('statuses') == dict(Counter(r['status'] for r in old['computational']))
            and t.get('all_converged') is all(r['status'] == 'CONVERGED' for r in old['computational']), 'Original terminal/counts mismatch')
    require(admission['portfolio']['sha256'] == t['portfolio_sha256'] == PORTFOLIO_SHA, 'Frozen portfolio changed')
    # Portfolio payload is large; its exact SHA is authenticated by the Context once.
    pp = Path(admission['portfolio']['path']).resolve(strict=True)
    require(pp == root / 'portfolios/PORTFOLIOS.json'
            and ctx.bind(pp, 'EXACT_UNCHANGED_PORTFOLIO', PORTFOLIO_SHA) == PORTFOLIO_SHA, 'Portfolio path/SHA changed')
    freeze = read_bound(ctx, admission['selection_freeze'], root / 'training/MODEL_SELECTION_FREEZE.json')
    require(freeze.get('status') == 'FROZEN' and freeze.get('config_sha256') == ctx.config_sha256
            and freeze.get('phase_id') == ctx.c['phase_id'] and freeze.get('run_id') == ctx.c['run_id']
            and freeze.get('test_targets_opened') is False and _same_binding(freeze['portfolio'], admission['portfolio']), 'Selection identity changed')
    runner = read_bound(ctx, admission['runner_result'], root / BASE / 'RUN_RESULT.json')
    require(runner.get('schema') == 'RECOVERY24_ASSIGNMENT_MEMORY_REPAIR_RUN_V1'
            and runner.get('status') == 'COMPLETE_WITH_RECORDED_OUTCOMES'
            and runner.get('phase_id') == ctx.c['phase_id'] and runner.get('run_id') == ctx.c['run_id']
            and runner.get('error') is None and runner.get('all_attempts_accounted') is True
            and runner.get('originals_unchanged_verified') is True and runner.get('originals_changed') is False
            and runner.get('clocks_reset') is False and runner.get('original_write_operations') is False
            and runner.get('scientific_success_claim') is False
            and runner.get('original_assignment_deadline_epoch') == ctx.c['action_deadlines']['assignments'],
            'Repair runner is not terminal with preserved original authority')
    require(_same_binding(runner['admission'], admission['runner_admission']), 'Explicit runner authority differs')
    runner_admission = read_bound(ctx, admission['runner_admission'])
    require(_same_binding(runner_admission['config'], admission['campaign_config'])
            and runner_admission['originals'] == originals
            and _same_binding(runner_admission['portfolio'], admission['portfolio'])
            and _same_binding(runner_admission['selection_freeze'], admission['selection_freeze']),
            'Runner admission lineage differs')
    closed = read_bound(ctx, runner['owned_lock_closed'], root / BASE / 'NATIONAL_LOCK_CLOSED.json')
    require(closed.get('schema') == 'RECOVERY24_NATIONAL_MEMORY_REPAIR_LOCK_V1'
            and _same_binding(closed['admission'], admission['runner_admission'])
            and Path(closed['output_root']).resolve() == root / BASE,
            'Owned lock closure does not bind this repair')
    for key in ('portfolio', 'selection_freeze'):
        require(_same_binding(runner[key], admission[key]), 'Runner lineage differs: ' + key)
    require(runner['originals'] == originals and runner['attempts'] == admission['attempts'], 'Runner/admission records differ')
    attempts = admission['attempts']
    require(len(attempts) == 2 and {x['logical_case_id']: x['attempt_case_id'] for x in attempts} == MAPPING,
            'Exactly the two declared repairs required')
    old_by_id = {r['case']['case_id']: r for r in old['computational']}
    replacement = {}; prior_failures = []
    for item in attempts:
        logical, native = item['logical_case_id'], item['attempt_case_id']
        previous = old_by_id[logical]; previous_path = root / 'assignments' / logical / 'RESULT.json'
        previous_binding = binding(previous_path); saved = read_bound(ctx, previous_binding, previous_path)
        require({k:v for k,v in previous.items() if k != 'native_artifacts'} == saved
                and saved.get('status') == 'UNAVAILABLE_CURRENT_RAM' and saved.get('new_scientific_execution') is False
                and saved.get('final_certificate_fresh') is False
                and saved.get('error') == 'Frozen portfolio XGBoost resident forecast exceeds guarded RAM'
                and saved['intent']['portfolio_sha256'] == PORTFOLIO_SHA, 'Original failure is not the admitted RAM forecast failure')
        directory = root / BASE / 'attempts' / native
        result = read_bound(ctx, item['result'], directory / 'RESULT.json')
        intent = read_bound(ctx, item['intent'], root / BASE / 'intents' / (native + '.json'))
        expected_case = dict(previous['case'], case_id=native)
        require(result['case'] == expected_case and result.get('logical_case_id') == logical and result.get('intent') == intent,
                'Native repair case/intent mismatch')
        require(result.get('status') in ADMITTED | UNAVAILABLE and intent.get('portfolio_sha256') == PORTFOLIO_SHA
                and intent.get('phase_id') == ctx.c['phase_id'] and intent.get('run_id') == ctx.c['run_id']
                and intent.get('initialization') == 'COLD_FEASIBLE_AON' and intent.get('dynamic_fallback') is False
                and intent.get('model_or_fallback_change') is False
                and _same_binding(intent['admission'], admission['runner_admission'])
                and _same_binding(intent['config'], admission['campaign_config'])
                and intent.get('solver') == 'MSA' and intent.get('objective_has_potential') is False,
                'Repair outcome/authority changed')
        start, end = intent['complete_query_start_epoch'], intent['deadline_epoch']
        require(result.get('complete_query_start_epoch') == start and result.get('deadline_epoch') == end,
                'Native result and immutable intent timing differ')
        if result['status'] in UNRUN:
            require(start is None and end is None and result.get('complete_query_end_epoch') is None
                    and result.get('elapsed_seconds') == 0 and result.get('new_scientific_execution') is False,
                    'Unrun disposition falsely reports query execution')
        else:
            elapsed = result.get('elapsed_seconds'); finished = result.get('complete_query_end_epoch')
            require(all(type(v) in (int,float) and math.isfinite(v) for v in (start,end,elapsed,finished))
                    and start >= max(ctx.c['original_start_epoch'], t['started_epoch'] + t['elapsed_seconds'])
                    and 0 < end-start <= 2700.000001 and elapsed >= 0
                    and math.isclose(finished-start, elapsed, rel_tol=1e-12, abs_tol=1e-6)
                    and end <= ctx.c['action_deadlines']['assignments'], 'Repair query exceeded original assignment authority')
        recorded, attempt_end, attempt_elapsed, post = (result.get(k) for k in
            ('disposition_recorded_epoch','attempt_end_epoch','attempt_elapsed_seconds','post_query_seconds'))
        require(all(type(v) in (int,float) and math.isfinite(v) for v in (recorded,attempt_end,attempt_elapsed,post))
                and attempt_elapsed >= 0 and post >= 0
                and math.isclose(attempt_end-recorded,attempt_elapsed,rel_tol=1e-12,abs_tol=1e-6),
                'Full attempt timing is not independently preserved')
        if start is not None:
            require(start == recorded and attempt_end >= finished and attempt_elapsed >= elapsed
                    and math.isclose(post,attempt_end-finished,rel_tol=1e-12,abs_tol=1e-6),
                    'Query and post-query accounting are inconsistent')
        else:
            require(post == 0, 'Unrun administrative disposition is not post-query work')
        verification_end = result.get('post_query_verification_deadline_epoch')
        if verification_end is not None:
            require(type(verification_end) in (int,float) and math.isfinite(verification_end)
                    and verification_end <= ctx.c['action_deadlines']['assignments'],
                    'Post-query verification exceeds original assignment deadline')
        provenance = result['memory_repair_provenance']
        require(_same_binding(provenance['original_failed_result'], previous_binding)
                and _same_binding(provenance['admission'], admission['runner_admission'])
                and _same_binding(provenance['evaluator'], runner_admission['evaluator'])
                and provenance.get('policy') == runner_admission['evaluator_policy']
                and provenance.get('no_model_or_fallback_change') is True
                and provenance.get('original_assignment_deadline_epoch') == ctx.c['action_deadlines']['assignments'],
                'Prior failure, evaluator or original clock linkage mismatch')
        if result.get('final_certificate_fresh') is True:
            require(result.get('new_scientific_execution') is True
                    and result.get('execution_status') == 'EXECUTED_FROZEN_PORTFOLIO_MEMORY_REPAIR'
                    and result.get('portfolio_sha256') == PORTFOLIO_SHA
                    and result.get('status') in ADMITTED and result.get('native_validation_passed') is True and verification_end is not None
                    and result.get('graph_identity') == intent['graph_identity']
                    and result.get('original_OD_identity') == intent['original_OD_identity'],
                    'Fresh repair is not explicit new validated execution')
            artifacts = item.get('native_artifacts', {})
            require(NATIVE.issubset(artifacts) and set(artifacts).issubset(NATIVE_ALLOWED),
                    'All certified native artifact bindings required; unknown names forbidden')
            for name, value in artifacts.items():
                require(Path(value['path']).resolve() == directory / name and len(value['sha256']) == 64
                        and type(value['bytes']) is int and value['bytes'] > 0, 'Native artifact identity malformed')
            require(_same_binding(artifacts['RESULT.json'], item['result']), 'Native RESULT binding differs')
        else:
            require(result.get('final_certificate_fresh') is False and result['status'] in UNAVAILABLE
                    and not item.get('native_artifacts'), 'Uncertified result mislabeled usable')
        row = copy.deepcopy(result)
        if item.get('native_artifacts'): row['native_artifacts'] = item['native_artifacts']
        replacement[logical] = row
        prior_failures.append(dict(logical_case_id=logical, result=previous_binding, outcome=previous))
    computational = [replacement.get(r['case']['case_id'], r) for r in old['computational']]
    reporting = old['reporting'][:2] + computational
    matrix = revised_matrix(old['matrix'], dict(attempts=attempts))
    terminal = dict(status='COMPLETE_WITH_RECORDED_OUTCOMES', phase_id=ctx.c['phase_id'], run_id=ctx.c['run_id'],
        rows=46, reporting_rows=48, portfolio_sha256=PORTFOLIO_SHA,
        statuses=dict(Counter(r['status'] for r in computational)),
        all_converged=all(r['status']=='CONVERGED' for r in computational),
        revision='EXACT_TWO_ATTEMPT_MEMORY_REPAIR', original_stage=originals['terminal'], repair_runner=admission['runner_result'],
        original_failed_attempts_retained=True, all_attempt_rows=50, non_reference_disposition_rows=48,
        all_attempt_rows_scope='48 original reporting dispositions including two reference-self rows, plus two retry dispositions; not 50 solver executions',
        timing_scope='Original stage wall and separate repair intervals retained; no combined fabricated stage wall')
    return dict(matrix=matrix, computational=computational, reporting=reporting, terminal=terminal,
                prior_failures=prior_failures, all_attempts=old['reporting']+[replacement[k] for k in MAPPING])


def reconcile(ctx, admission_path, *, admission_sha256, deadline):
    ctx.require_action('publication'); ctx.guard(deadline)
    root=Path(ctx.root).resolve()
    require(not any((root/n).exists() for n in ('ADVANCE_CONTROLLER_LOCK.json','NATIONAL_TIMING_ACTIVE.json')),
            'No reconciliation during live scientific activity')
    admitted=dict(path=str(Path(admission_path).resolve()),sha256=admission_sha256)
    admission=read_bound(ctx,admitted)
    end=admission['deadline_epoch']
    require(type(end) in (int,float) and math.isfinite(end) and time.time()<end<=min(deadline,ctx.deadline('publication')),
            'Metadata reconciliation deadline invalid')
    data=_validated(ctx,admission); output=root/LEDGERS
    ctx.guard(end); output.mkdir(parents=True,exist_ok=False)
    def new(name,value):
        with (output/name).open('x',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    new('INTENT.json',dict(admission=admitted,started_epoch=time.time(),deadline_epoch=end,source_sha256=digest(__file__)))
    for key,name in FILES.items():new(name,data[key])
    new('ALL_ATTEMPTS.json',data['all_attempts'])
    new('PRESERVED_FAILURES.json',data['prior_failures'])
    # Reauthenticate small source/receipt inputs after writes; failures preserve the new directory without a PASS manifest.
    require(_validated(ctx,admission)==data,'Source outcomes changed during reconciliation');ctx.guard(end)
    record=dict(schema='RECOVERY24_ASSIGNMENT_REVISION_V1',status='VERIFIED_EXACT_TWO_CASE_REVISION',
        phase_id=ctx.c['phase_id'],run_id=ctx.c['run_id'],campaign_config=admission['campaign_config'],
        admission=admitted,attempts=admission['attempts'],originals=admission['originals'],
        portfolio=admission['portfolio'],selection_freeze=admission['selection_freeze'],runner_result=admission['runner_result'],
        files={key:binding(output/name) for key,name in FILES.items()},
        all_attempts=binding(output/'ALL_ATTEMPTS.json'),preserved_failures=binding(output/'PRESERVED_FAILURES.json'),
        logical_computational_rows=46,logical_reporting_rows=48,all_attempt_rows=50,non_reference_disposition_rows=48,
        original_rows_unchanged=44,reference_rows_unchanged=2,scientific_execution=False,
        metrics_relative=METRICS,timing_relative=TIMING)
    new('REVISION.json',record);return binding(output/'REVISION.json')


def resolve(ctx, revision_binding=None):
    root=Path(ctx.root).resolve()
    if revision_binding is None:
        return dict(revision=None,paths={k:root/n for k,n in FILES.items()},metrics_root=root/'report_data',timing_root=root/'report_timing/v1')
    record=read_bound(ctx,revision_binding,root/LEDGERS/'REVISION.json')
    require(record.get('schema')=='RECOVERY24_ASSIGNMENT_REVISION_V1' and record.get('status')=='VERIFIED_EXACT_TWO_CASE_REVISION',
            'Verified explicit two-case revision required')
    admission=read_bound(ctx,record['admission']); expected=_validated(ctx,admission)
    require(record['attempts']==admission['attempts'] and record['originals']==admission['originals']
            and record['portfolio']==admission['portfolio'] and record['runner_result']==admission['runner_result']
            and record['selection_freeze']==admission['selection_freeze']
            and record['campaign_config']==admission['campaign_config']
            and record.get('phase_id')==ctx.c['phase_id'] and record.get('run_id')==ctx.c['run_id']
            and record.get('logical_computational_rows')==46 and record.get('logical_reporting_rows')==48
            and record.get('all_attempt_rows')==50 and record.get('non_reference_disposition_rows')==48
            and record.get('original_rows_unchanged')==44 and record.get('reference_rows_unchanged')==2
            and record.get('scientific_execution') is False, 'Revision lineage/count substitution')
    require(set(record['files'])==set(FILES),'Complete revised ledger map required')
    for key,name in FILES.items():
        require(read_bound(ctx,record['files'][key],root/LEDGERS/name)==expected[key],'Revised ledger changed: '+key)
    require(read_bound(ctx,record['all_attempts'],root/LEDGERS/'ALL_ATTEMPTS.json')==expected['all_attempts']
            and read_bound(ctx,record['preserved_failures'],root/LEDGERS/'PRESERVED_FAILURES.json')==expected['prior_failures'],
            'Original failed attempts lost or changed')
    require(record['metrics_relative']==METRICS and record['timing_relative']==TIMING,'Versioned output namespace changed')
    return dict(revision=record,paths={k:Path(v['path']).resolve() for k,v in record['files'].items()},
                metrics_root=root/METRICS,timing_root=root/TIMING)
