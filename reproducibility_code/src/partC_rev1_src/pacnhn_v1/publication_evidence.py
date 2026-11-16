"""Explicit metadata supplements for publication; never loads scientific arrays.

Import has no side effects. The nine canonical publication CSVs are unchanged.
Caller supplies authenticated continuation pins and an explicit supplement map.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

SCHEMA = 'PACNHN_PUBLICATION_EVIDENCE_V1'
KEYS = ('training_result', 'training_clock', 'finalization_clock',
        'finalization_forecast', 'test_summary', 'test_tail_report',
        'm02_result', 'm02_seal', 't08_result', 't08_intent',
        't08_manifest', 't08_seal')
MAXIMUM_BYTES = 16 * 1024**2


def _require(value, message):
    if not value:
        raise ValueError(message)


def _number(value, label, *, nonnegative=True):
    _require(not isinstance(value, bool) and isinstance(value, (int, float)), label+' is not numeric')
    value = float(value)
    _require(math.isfinite(value) and (not nonnegative or value >= 0), label+' is invalid')
    return value


def _count(value, label):
    n = _number(value, label)
    _require(n.is_integer(), label+' is not an integer')
    return int(n)


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=0.) or a == b


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'Duplicate metadata key')
            result[key] = value
        return result
    return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON '+x)))


def _read(binding, guard):
    _require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}, 'Explicit path/SHA binding required')
    digest = binding['sha256']
    _require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Invalid SHA256')
    _require(isinstance(binding['path'], str) and bool(binding['path']), 'Invalid path')
    p = Path(binding['path']).resolve(strict=True)
    guard(); _require(p.is_file() and p.stat().st_size <= MAXIMUM_BYTES, 'Oversize or non-file supplement')
    with p.open('rb') as f:
        raw = f.read(MAXIMUM_BYTES+1)
    guard(); _require(len(raw) <= MAXIMUM_BYTES and _digest(raw) == digest, 'Supplement bytes differ')
    return _json(raw), dict(path=str(p), sha256=digest, bytes=len(raw))


def _same_binding(a, b):
    return (a.get('sha256') == b.get('sha256') and
            Path(a['path']).resolve() == Path(b['path']).resolve())


def training_facts(result, clock, final_clock, forecast, records):
    _require(result['status'] == 'STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE', 'Training is not terminal')
    _require(result['clock'] == clock, 'Original training clock differs')
    start = _number(clock['stage_start_epoch'], 'stage start')
    end = _number(result['finalization_ended_epoch'], 'terminal time')
    deadline = _number(clock['stage_deadline_epoch'], 'stage deadline')
    _require(deadline-start == _number(clock['stage_seconds'], 'stage allowance') and start <= end, 'Invalid training interval')
    rows = []; seen = set()
    for r in result['coverage']:
        key = (r['suite_id'], r['family'])
        _require(key not in seen, 'Duplicate training coverage'); seen.add(key)
        e, a, q = (_count(r[k], k) for k in ('expected', 'available', 'qualified'))
        _require(0 <= q <= a <= e, 'Impossible training counts')
        rows.append(dict(suite_id=key[0], family=key[1], expected=e, available=a, qualified=q))
    facts = dict(elapsed_seconds=end-start, unused_seconds=max(0., deadline-end),
        overrun_seconds=max(0., end-deadline), coverage=rows,
        all_s03_qualified=result.get('first_all_s03_qualified') is not None,
        finalization_seconds=None, forecast_reserve_seconds=None,
        timing_interpretation='Elapsed wall includes interruption and finalization; not active fitting time')
    if final_clock is not None:
        expected = dict(path=result['finalization_clock_path'], sha256=result['finalization_clock_sha256'])
        _require(_same_binding(expected, records['finalization_clock']), 'Finalization clock identity differs')
        _require(final_clock['stage_start_epoch'] == start and final_clock['stage_deadline_epoch'] == deadline, 'Finalization clock stage differs')
        fs = _number(final_clock['finalization_started_epoch'], 'finalization start')
        _require(start <= fs <= end and fs == result['finalization_started_epoch'], 'Finalization chronology differs')
        _require(_close(end-fs, _number(result['finalization_elapsed_seconds'], 'finalization elapsed')), 'Finalization elapsed differs')
        facts['finalization_seconds'] = end-fs
    if forecast is not None:
        _require(forecast['stage_deadline_epoch'] == deadline, 'Forecast deadline differs')
        reserve = _number(forecast['reserve_seconds'], 'forecast reserve')
        _require(_close(deadline-_number(forecast['science_deadline_epoch'], 'science cutoff'), reserve), 'Forecast reserve/cutoff differs')
        facts['forecast_reserve_seconds'] = reserve
    return facts


def test_facts(summary, registry_sha):
    _require(summary['registry_sha256'] == registry_sha, 'TEST registry differs')
    _require(summary['test_results_used_for_selection'] is False, 'TEST selection separation absent')
    regions = summary['results']; seen = set(); suites = {}; groups = {}
    _require(_count(summary['regions_audited'], 'audited regions') == len(regions), 'TEST disposition count differs')
    for row in regions:
        key = (row['suite_id'], _count(row['region_index'], 'region'))
        _require(key not in seen and row['registry_sha256'] == registry_sha, 'Duplicate region or registry mismatch')
        seen.add(key)
        admitted = _count(row['admitted_test_inputs'], 'admitted TEST inputs')
        requested = _count(row['requested_test_inputs'], 'requested TEST inputs')
        _require(admitted <= requested, 'Admitted TEST exceeds requested')
        suite = suites.setdefault(key[0], dict(regions=0, regions_with_metrics=0, admitted_inputs=0, requested_inputs=0))
        suite['regions'] += 1; suite['admitted_inputs'] += admitted; suite['requested_inputs'] += requested
        models = row['models']; suite['regions_with_metrics'] += bool(models); families = set()
        for model in models:
            family = model['family']; _require(family not in families, 'Duplicate family/region TEST metric'); families.add(family)
            error = _number(model['absolute_error_sum'], 'absolute error')
            truth = _number(model['true_cost_sum'], 'true cost')
            coordinates = _count(model['coordinates'], 'coordinates')
            inputs = _count(model['unique_inputs'], 'unique inputs')
            _require(inputs == admitted and inputs > 0 and coordinates >= inputs, 'TEST metric population differs')
            value = model.get('WMAPE_percent')
            if truth > 0:
                value = _number(value, 'regional WMAPE')
                _require(_close(value, 100*error/truth), 'Recorded regional TEST WMAPE differs from sums')
            else:
                _require(value is None or (error == 0 and _number(value, 'zero WMAPE') == 0), 'Undefined regional WMAPE')
            groups.setdefault((key[0], family), []).append(dict(region_index=key[1], absolute_error_sum=error,
                true_cost_sum=truth, coordinates=coordinates, inputs=inputs, WMAPE_percent=value))
    aggregates = []
    for (suite, family), rows in sorted(groups.items()):
        error = math.fsum(r['absolute_error_sum'] for r in rows); truth = math.fsum(r['true_cost_sum'] for r in rows)
        _require(math.isfinite(error) and math.isfinite(truth), 'Pooled TEST sums overflowed')
        percentages = [r['WMAPE_percent'] for r in rows if r['WMAPE_percent'] is not None]
        top = sorted(rows, key=lambda r: (-r['absolute_error_sum'], r['region_index']))[:5]
        top5 = [dict(r, share_of_family_absolute_error_percent=100*r['absolute_error_sum']/error if error else 0.,
                     share_of_common_true_cost_percent=100*r['true_cost_sum']/truth if truth else None) for r in top]
        pooled=100*error/truth if truth else None
        _require(pooled is None or math.isfinite(pooled), 'Pooled TEST percentage overflowed')
        aggregates.append(dict(suite_id=suite, family=family, regions=len(rows), finite_regions=len(percentages),
            pooled_native_cost_WMAPE_percent=pooled,
            region_median_WMAPE_percent=statistics.median(percentages) if percentages else None,
            region_max_WMAPE_percent=max(percentages) if percentages else None,
            absolute_error_sum=error, true_cost_sum=truth, coordinates=sum(r['coordinates'] for r in rows),
            inputs=sum(r['inputs'] for r in rows), top5=top5,
            top5_absolute_error_share_percent=100*math.fsum(r['absolute_error_sum'] for r in top)/error if error else 0.))
    for s in suites.values(): s['missing_inputs'] = s['requested_inputs']-s['admitted_inputs']
    return dict(dispositions_complete=summary.get('complete') is True, suite_coverage=suites, aggregates=aggregates)


def validate_tail_report(report, facts):
    groups = {(r['suite_id'], r['family']): r for r in facts['aggregates']}; seen = set()
    for supplied in report['families']:
        key = ('s03', supplied['family']); _require(key not in seen and key in groups, 'Tail family differs'); seen.add(key)
        expected = groups[key]
        _require(len(supplied['top5']) == len(expected['top5']), 'Tail row count differs')
        for a, b in zip(supplied['top5'], expected['top5']):
            _require(a['region_index'] == b['region_index'], 'Tail rank/region differs')
            for k in ('WMAPE_percent', 'absolute_error_sum', 'share_of_family_absolute_error_percent', 'share_of_common_true_cost_percent'):
                _require((a[k] is None and b[k] is None) or (a[k] is not None and b[k] is not None and _close(_number(a[k], k), b[k])), 'Tail value differs: '+k)
        _require(_close(_number(supplied['top5_absolute_error_share_percent'], 'tail share'), expected['top5_absolute_error_share_percent']), 'Tail total differs')
    _require(seen == {k for k in groups if k[0] == 's03'}, 'Tail report family coverage differs')


def audit_facts(documents, records, registry_sha, training_sha, expected):
    result = {}
    m = documents.get('m02_result')
    if m is not None:
        _require(m['schema'] == 'PACNHN_FINAL_ICNN_GUARANTEES_RESULT_V1' and m['registry_binding']['sha256'] == registry_sha, 'Property audit selection differs')
        _require(_count(m['expected'], 'expected models') == expected, 'Property audit coverage differs')
        counts = {k:_count(v, 'model disposition') for k,v in m['dispositions'].items()}
        _require(sum(counts.values()) == expected, 'Property disposition ledger incomplete')
        _require(all(m[k] is False for k in ('accuracy_evaluated','training_target_changed','model_selection_changed','TEST_opened')), 'Property audit changed scope')
        if m['all_selected_constraints_and_declared_checks_pass']:
            _require(counts == {'PASS':expected}, 'Property PASS contradicts dispositions')
        seal = documents.get('m02_seal')
        verified = seal is not None
        if verified:
            _require(seal.get('status') == 'VERIFIED' and seal.get('result_sha256') == records['m02_result']['sha256'], 'Property audit seal differs')
        result['properties'] = dict(status=m['status'], expected=expected, dispositions=counts,
            seal_verified=verified, scope=m['finite_numerical_scope'], accuracy_evaluated=False)
        properties = result['properties']
        if 'documentary_recovery' in m:
            _require(isinstance(m['documentary_recovery'], bool), 'Documentary recovery flag is not Boolean')
            properties['documentary_recovery'] = m['documentary_recovery']
        if 'completed' in m or 'uncompleted' in m:
            _require('completed' in m and 'uncompleted' in m, 'Property completion counts must be paired')
            completed = _count(m['completed'], 'completed property checks')
            uncompleted = _count(m['uncompleted'], 'uncompleted property checks')
            _require(completed+uncompleted == expected, 'Property completion counts differ')
            properties.update(completed=completed, uncompleted=uncompleted)
        for key in ('original_shared_start_epoch', 'original_audit_deadline_epoch'):
            if key in m:
                properties[key] = _number(m[key], key)
        if 'terminal_reason' in m:
            _require(isinstance(m['terminal_reason'], str) and bool(m['terminal_reason'].strip()), 'Invalid property terminal reason')
            properties['terminal_reason'] = m['terminal_reason']
        if properties.get('documentary_recovery'):
            _require(m['status'] in {'PARTIAL_WITH_LIMITATIONS','FAIL'}, 'Documentary recovery cannot claim successful numerical audit')
            _require(properties.get('completed') == 0 and properties.get('uncompleted') == expected,
                     'Documentary recovery cannot establish completed model checks')
            _require(m['all_selected_constraints_and_declared_checks_pass'] is False
                     and set(counts).issubset({'INVALID_RESULT_METADATA','UNATTEMPTED'}),
                     'Documentary recovery dispositions imply numerical checks')
            _require(properties.get('terminal_reason') == 'OPERATIONAL_SERIALIZATION_FAILURE_AND_ORIGINAL_600_SECOND_BUDGET_EXHAUSTED',
                     'Unknown documentary terminal reason')
            start = properties.get('original_shared_start_epoch')
            end = properties.get('original_audit_deadline_epoch')
            _require(start is not None and end is not None and end-start == 600,
                     'Documentary recovery changed original 600-second window')
            properties['original_audit_allowance_seconds'] = 600
    t = documents.get('t08_result')
    if t is not None:
        intent = documents.get('t08_intent'); _require(intent is not None, 'Training screen requires bound intent')
        _require(intent['config']['training_result']['sha256'] == training_sha, 'Training screen source differs')
        e = _count(t['expected_regions'], 'expected screened regions'); done = _count(t['completed_regions'], 'completed screens')
        reused = _count(t['reused_regions'], 'reused screens'); fresh = _count(t['refreshed_regions'], 'refreshed screens')
        _require(e == expected and 0 <= done <= e and reused+fresh == done, 'Training screen coverage differs')
        _require(all(t[k] is False for k in ('TEST_targets_opened','new_labels_or_fits','selected_models_changed','exact_physical_cost_truth_claim')), 'Training screen scope differs')
        seal_flag_recorded = documents.get('t08_seal') is not None
        if seal_flag_recorded:
            manifest = documents.get('t08_manifest'); _require(manifest is not None and documents['t08_seal'].get('verified') is True, 'Training screen sealing evidence missing')
            entries = {r['relative_path']:r for r in manifest['artifacts']}
            for name,key in [('RESULT.json','t08_result'),('INTENT.json','t08_intent')]:
                _require(entries.get(name,{}).get('sha256') == records[key]['sha256'], 'Training screen manifest identity differs')
        maximum = t.get('maximum_bound_percent')
        if maximum is not None:maximum = _number(maximum, 'maximum fixed-label bound')
        positive = _count(t['positive_bound_regions'], 'positive bound regions'); above = _count(t['above_1_percent_regions'], 'above1 regions')
        _require(0 <= above <= positive <= done, 'Impossible bound counts')
        result['training_limits'] = dict(status=t['status'], expected=e, completed=done,reused=reused,refreshed=fresh,
            positive_bound_regions=positive,above_1_percent_regions=above,maximum_bound_percent=maximum,
            seal_verified=False,seal_flag_recorded=seal_flag_recorded,
            archive_linkage='NOT_ESTABLISHED_BY_METADATA_ONLY_ADAPTER',
            manifest_identity_verified=seal_flag_recorded)
    return result


def narrative(facts):
    """Scientific prose only; machine paths, epochs and orchestration stay provenance."""
    sections = dict(learning=[], timing=[], resolution=[], validation=[])
    t = facts.get('training')
    if t:
        text = (f"The regional data and training stage elapsed {t['elapsed_seconds']/3600:.6f} h, including interruptions and finalization rather than solely active fitting. "
                f"It finished with {t['unused_seconds']:.3f} s of its original allowance unused and {t['overrun_seconds']:.3f} s of overrun.")
        if t['forecast_reserve_seconds'] is not None:text += f" The forecast reserved {t['forecast_reserve_seconds']:.3f} s for finalization."
        if t['finalization_seconds'] is not None:text += f" Measured finalization took {t['finalization_seconds']:.3f} s."
        text += " An unused allowance is not additional training or evidence that the accuracy target was met."
        sections['timing'].append(text)
        rows = [r for r in t['coverage'] if r['suite_id']=='s03' and r['family']!='FFTT']
        sections['learning'].append('Current s03 model availability and TRAIN qualification are separate: '+'; '.join(
            f"{r['family']}: {r['available']:,}/{r['expected']:,} available, {r['qualified']:,} below 1% TRAIN cost WMAPE" for r in rows)+'. '
            +('The complete s03 qualification milestone was recorded.' if t['all_s03_qualified'] else 'The complete s03 qualification milestone was not reached.'))
        other = [r for r in t['coverage'] if r['suite_id']!='s03']
        sections['resolution'].append('Additional learned training is gated on complete s03 qualification; analytic FFTT assignment eligibility is separate. Other-suite model coverage is '+'; '.join(
            f"{r['suite_id']}/{r['family']}: {r['available']:,}/{r['expected']:,}" for r in other)+'. Actual assignment outcomes are reported separately.')
    else:sections['timing'].append('Authenticated terminal training timing metadata is unavailable; no elapsed or unused training allowance is inferred.')
    tests = facts.get('test')
    if tests:
        for suite,s in sorted(tests['suite_coverage'].items()):
            sections['learning'].append(f"{suite} held-out coverage: {s['regions_with_metrics']:,}/{s['regions']:,} regions have model metrics, with {s['admitted_inputs']:,}/{s['requested_inputs']:,} requested regional inputs admitted and {s['missing_inputs']:,} missing. Disposition completeness does not replace missing input coverage.")
        for a in tests['aggregates']:
            pooled='undefined' if a['pooled_native_cost_WMAPE_percent'] is None else f"{a['pooled_native_cost_WMAPE_percent']:.6f}%"
            median='undefined' if a['region_median_WMAPE_percent'] is None else f"{a['region_median_WMAPE_percent']:.6f}%"
            sections['learning'].append(f"{a['suite_id']}/{a['family']} TEST cost WMAPE is {pooled} pooled versus {median} as the unweighted regional median ({a['regions']:,} measured regions, {a['inputs']:,} regional-input observations, {a['coordinates']:,} cost coordinates). Its five largest absolute-error regions contribute {a['top5_absolute_error_share_percent']:.6f}% of total absolute error.")
        sections['learning'].append('Pooled error uses the sum of absolute native cost errors divided by the sum of true native costs, including zero-truth-row errors in the numerator. It is not an average of regional percentages, a trip-weighted statistic, or national backbone-flow error. Family observations repeat their admitted cohorts and are not independent samples; missing values are not zero errors.')
    else:sections['learning'].append('Authenticated sealed held-out aggregate metadata is unavailable; no pooled error or omitted-input count is inferred.')
    audits = facts.get('audits',{})
    m = audits.get('properties')
    if m:
        sections['validation'].append(f"Finite implementation checks record an outcome of {str(m['status']).lower().replace('_',' ')} for an expected {m['expected']:,} selected ICNNs, with dispositions "+'; '.join(f'{k}: {v:,}' for k,v in sorted(m['dispositions'].items()))+'. '
            +('A matching verification seal is recorded. ' if m['seal_verified'] else 'A matching verification seal is unavailable; this evidence remains qualified. ')
            +'These are declared finite boundary/stress, derivative and Hessian-vector checks, not a global floating-point theorem or fit/generalization test. The global intended-model guarantees remain conditional on constrained parameters and the mathematical proof.')
    else:sections['validation'].append('Final all-selected ICNN implementation-check outcomes are unavailable; the mathematical derivation and synthetic tests do not establish completed all-model numerical validation.')
    t = audits.get('training_limits')
    if t:
        maximum='unavailable' if t['maximum_bound_percent'] is None else f"{t['maximum_bound_percent']:.8g}%"
        sections['validation'].append(f"The current-TRAIN monotonicity screen records an outcome of {str(t['status']).lower().replace('_',' ')} and completed {t['completed']:,}/{t['expected']:,} regions ({t['reused']:,} exact-current reused screens and {t['refreshed']:,} refreshed screens); {t['positive_bound_regions']:,} had positive fixed-label lower bounds and {t['above_1_percent_regions']:,} exceeded 1%, with maximum {maximum}. "
            +'The diagnostic outcome is reported from bound result metadata; archive linkage is not established by these supplemental inputs. '
            +'Finite screened pairs do not prove absence of other conflicts, attainable accuracy, exact physical truth or coordinatewise marginal-cost bounds. Global label-uncertainty and zero distributions are not newly established by this screen.')
    else:sections['validation'].append('Final current-TRAIN conflict/uncertainty audit outcomes are unavailable; no completeness or attainable-error conclusion is inferred.')
    return sections


def load_evidence(spec, *, training_result_sha256, registry_sha256, guard=lambda: None):
    _require(isinstance(spec,dict) and set(spec)==set(KEYS), 'Explicit complete supplement map required; use null for unavailable metadata')
    documents={}; records={}
    for key in KEYS:
        if spec[key] is not None: documents[key],records[key]=_read(spec[key],guard)
    result=documents.get('training_result'); facts={}
    if result is not None:
        _require(records['training_result']['sha256']==training_result_sha256 and result['registry_sha256']==registry_sha256, 'Continuation-pinned training/selection differs')
    if documents.get('training_clock') is not None:
        _require(result is not None, 'Training clock requires terminal result')
        facts['training']=training_facts(result,documents['training_clock'],documents.get('finalization_clock'),documents.get('finalization_forecast'),records)
    elif any(k in documents for k in ('finalization_clock','finalization_forecast')):
        raise ValueError('Finalization metadata requires original training clock')
    if 'test_summary' in documents:
        _require(result is not None and result.get('TEST_opened_only_after_freeze') is True, 'Post-selection TEST authority missing')
        _require(_same_binding(result['test_disposition'],records['test_summary']), 'Sealed TEST summary identity differs')
        facts['test']=test_facts(documents['test_summary'],registry_sha256)
    if 'test_tail_report' in documents:
        _require('test' in facts, 'Tail report needs sealed TEST summary'); validate_tail_report(documents['test_tail_report'],facts['test'])
    for child,parent in [('m02_seal','m02_result'),('t08_intent','t08_result'),('t08_manifest','t08_result'),('t08_seal','t08_result')]:
        _require(child not in documents or parent in documents, child+' lacks result')
    if 'm02_result' in documents or 't08_result' in documents:
        _require(result is not None, 'Audit result requires terminal training')
        expected=[r['expected'] for r in result['coverage'] if r['suite_id']=='s03' and r['family']=='ICNN']
        _require(len(expected)==1,'Expected ICNN population missing')
        facts['audits']=audit_facts(documents,records,registry_sha256,training_result_sha256,_count(expected[0],'ICNN population'))
    return dict(schema=SCHEMA,bindings=records,missing=[k for k in KEYS if k not in records],facts=facts,
        paragraphs=narrative(facts),provenance_only=dict(training_result_sha256=training_result_sha256,registry_sha256=registry_sha256),
        code_sha256=_digest(Path(__file__).read_bytes()))


def snapshot_evidence(root, evidence, guard=lambda: None):
    """New local metadata snapshot plus provenance; never edits original sources."""
    folder=Path(root)/'manuscript_evidence'; folder.mkdir(parents=True,exist_ok=False)
    record=dict(evidence); snapshots={}
    for key,binding in evidence['bindings'].items():
        _,current=_read({k:binding[k] for k in ('path','sha256')},guard)
        raw=Path(current['path']).read_bytes(); _require(_digest(raw)==current['sha256'],'Source changed while snapshotting')
        p=folder/(key+'.json')
        with p.open('xb') as f:f.write(raw)
        snapshots[key]=dict(relative_path='manuscript_evidence/'+p.name,sha256=current['sha256'],bytes=len(raw))
    record['snapshots']=snapshots
    with (folder/'PROVENANCE.json').open('x',encoding='utf8',newline='\n') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
    return dict(relative_path='manuscript_evidence/PROVENANCE.json',sha256=_digest((folder/'PROVENANCE.json').read_bytes()))


def verify_snapshot(root, binding, *, evidence=None, guard=lambda: None):
    """Called before/after rendering and compilation with extraction-pinned SHA."""
    base=Path(root).resolve();p=(base/binding['relative_path']).resolve()
    _require(p.is_relative_to(base) and p==base/'manuscript_evidence/PROVENANCE.json','Noncanonical evidence provenance path')
    record,_=_read(dict(path=str(p),sha256=binding['sha256']),guard)
    _require(record['schema']==SCHEMA and record['code_sha256']==_digest(Path(__file__).read_bytes()),'Evidence adapter source changed')
    if evidence is not None:
        _require({k:v for k,v in record.items() if k!='snapshots'}==evidence,'Supplement identities/facts changed after extraction')
    for key,row in record['snapshots'].items():
        q=(base/row['relative_path']).resolve()
        _require(q==base/'manuscript_evidence'/(key+'.json'),'Noncanonical snapshot path')
        _,actual=_read(dict(path=str(q),sha256=row['sha256']),guard)
        _require(actual['bytes']==row['bytes'] and row['sha256']==record['bindings'][key]['sha256'],'Snapshot identity differs')
    _require(set(record['snapshots'])==set(record['bindings']),'Snapshot coverage differs')
    return record
