"""Synthetic metadata only; no campaign models, arrays or held-out targets."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from partC_rev1_src.pacnhn_v1 import publication_evidence as e


def fixture(tmp_path):
    spec={k:None for k in e.KEYS};registry='a'*64
    def write(key,value):
        p=tmp_path/(key+'.json');p.write_text(json.dumps(value),encoding='utf8')
        spec[key]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()};return spec[key]
    def region(i,error,truth,requested=3,admitted=2):
        return dict(suite_id='s03',region_index=i,registry_sha256=registry,requested_test_inputs=requested,
            admitted_test_inputs=admitted,models=[dict(family='ICNN',model_sha256=str(i)*64,absolute_error_sum=error,
                true_cost_sum=truth,WMAPE_percent=100*error/truth if truth else None,coordinates=4,unique_inputs=admitted)])
    summary=dict(registry_sha256=registry,test_results_used_for_selection=False,regions_audited=3,complete=True,
        results=[region(0,1,10),region(1,30,100),dict(suite_id='s02',region_index=0,registry_sha256=registry,requested_test_inputs=0,admitted_test_inputs=0,models=[])])
    tb=write('test_summary',summary)
    clock=dict(stage_start_epoch=100.,stage_deadline_epoch=1100.,stage_seconds=1000.)
    write('training_clock',clock)
    fc=dict(stage_start_epoch=100.,stage_deadline_epoch=1100.,finalization_started_epoch=600.)
    fcb=write('finalization_clock',fc)
    write('finalization_forecast',dict(stage_deadline_epoch=1100.,science_deadline_epoch=500.,reserve_seconds=600.))
    training=dict(status='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE',registry_sha256=registry,clock=clock,
        finalization_ended_epoch=900.,finalization_started_epoch=600.,finalization_elapsed_seconds=300.,
        finalization_clock_path=fcb['path'],finalization_clock_sha256=fcb['sha256'],test_disposition=tb,
        TEST_opened_only_after_freeze=True,first_all_s03_qualified=None,
        coverage=[dict(suite_id='s03',family='ICNN',expected=2,available=2,qualified=1),
                  dict(suite_id='s02',family='FFTT',expected=1,available=1,qualified=0)])
    rb=write('training_result',training)
    pins=dict(training_result_sha256=rb['sha256'],registry_sha256=registry)
    return spec,pins,write


def test_actual_sums_counts_times_and_missing_audits(tmp_path):
    spec,pins,_=fixture(tmp_path);d=e.load_evidence(spec,**pins)
    assert d['facts']['training']['elapsed_seconds']==800
    assert d['facts']['training']['unused_seconds']==200
    assert d['facts']['training']['forecast_reserve_seconds']==600
    assert d['facts']['training']['finalization_seconds']==300
    row=d['facts']['test']['aggregates'][0]
    assert row['pooled_native_cost_WMAPE_percent']==pytest.approx(3100/110)
    assert row['region_median_WMAPE_percent']==20
    assert d['facts']['test']['suite_coverage']['s03']['missing_inputs']==2
    assert 'm02_result' in d['missing'] and 't08_result' in d['missing']
    prose=' '.join(x for rows in d['paragraphs'].values() for x in rows)
    for forbidden in [str(tmp_path),'M02','T08','C01','root','epoch']:
        assert forbidden not in prose
    assert 'not reached' in prose and 'outcomes are unavailable' in prose


def test_explicit_all_missing_is_qualified_not_noop_pass():
    d=e.load_evidence(dict.fromkeys(e.KEYS),training_result_sha256='a'*64,registry_sha256='b'*64)
    assert len(d['missing'])==len(e.KEYS) and not d['facts']
    assert any('unavailable' in x for x in d['paragraphs']['learning'])


@pytest.mark.parametrize('bad',[{},None,{'extra':None}])
def test_implicit_or_incomplete_supplement_spec_rejected(bad):
    with pytest.raises(ValueError):e.load_evidence(bad,training_result_sha256='a'*64,registry_sha256='b'*64)


@pytest.mark.parametrize('mutation',['result_pin','registry_pin','clock','test_hash','test_registry','selection','count','negative','metric','duplicate_region','forecast'])
def test_authentication_and_numeric_inconsistency_fail_closed(tmp_path,mutation):
    spec,pins,write=fixture(tmp_path)
    if mutation=='result_pin':pins['training_result_sha256']='c'*64
    elif mutation=='registry_pin':pins['registry_sha256']='b'*64
    elif mutation=='clock':
        value=json.loads(Path(spec['training_clock']['path']).read_text());value['stage_start_epoch']=101;write('training_clock',value)
    elif mutation=='forecast':
        value=json.loads(Path(spec['finalization_forecast']['path']).read_text());value['reserve_seconds']=601;write('finalization_forecast',value)
    elif mutation=='test_hash':Path(spec['test_summary']['path']).write_text('{}')
    else:
        s=json.loads(Path(spec['test_summary']['path']).read_text())
        if mutation=='test_registry':s['registry_sha256']='c'*64
        elif mutation=='selection':s['test_results_used_for_selection']=True
        elif mutation=='count':s['regions_audited']=2
        elif mutation=='duplicate_region':s['results'][1]['region_index']=0
        elif mutation=='negative':s['results'][0]['models'][0]['absolute_error_sum']=-1
        elif mutation=='metric':s['results'][0]['models'][0]['WMAPE_percent']=99
        write('test_summary',s)
        r=json.loads(Path(spec['training_result']['path']).read_text());r['test_disposition']=spec['test_summary'];write('training_result',r);pins['training_result_sha256']=spec['training_result']['sha256']
    with pytest.raises(ValueError):e.load_evidence(spec,**pins)


def test_zero_truth_errors_stay_in_pooled_numerator_and_no_zero_denominator_guess(tmp_path):
    spec,pins,write=fixture(tmp_path)
    s=json.loads(Path(spec['test_summary']['path']).read_text());s['results'][0]['models'][0].update(true_cost_sum=0,absolute_error_sum=10,WMAPE_percent=None)
    d=e.test_facts(s,pins['registry_sha256']);a=d['aggregates'][0]
    assert a['pooled_native_cost_WMAPE_percent']==40 and a['region_median_WMAPE_percent']==30
    s['results'][1]['models'][0].update(true_cost_sum=0,WMAPE_percent=None)
    assert e.test_facts(s,pins['registry_sha256'])['aggregates'][0]['pooled_native_cost_WMAPE_percent'] is None


def test_zero_truth_error_can_not_make_pooled_percentage_overflow_silently(tmp_path):
    spec,pins,_=fixture(tmp_path);s=json.loads(Path(spec['test_summary']['path']).read_text())
    s['results'][0]['models'][0].update(true_cost_sum=0,absolute_error_sum=1e308,WMAPE_percent=None)
    s['results'][1]['models'][0].update(true_cost_sum=1,absolute_error_sum=0,WMAPE_percent=0)
    with pytest.raises(ValueError,match='overflowed'):e.test_facts(s,pins['registry_sha256'])


def test_bound_tail_report_must_match_full_sealed_sums(tmp_path):
    spec,pins,write=fixture(tmp_path);facts=e.load_evidence(spec,**pins)['facts']['test'];a=facts['aggregates'][0]
    report={'families':[{'family':'ICNN','top5':a['top5'],'top5_absolute_error_share_percent':a['top5_absolute_error_share_percent']}]}
    write('test_tail_report',report);e.load_evidence(spec,**pins)
    report['families'][0]['top5'][0]['share_of_family_absolute_error_percent']+=1
    write('test_tail_report',report)
    with pytest.raises(ValueError):e.load_evidence(spec,**pins)


def audit_fixture(tmp_path):
    spec,pins,write=fixture(tmp_path)
    m=dict(schema='PACNHN_FINAL_ICNN_GUARANTEES_RESULT_V1',registry_binding={'sha256':pins['registry_sha256']},expected=2,
        dispositions={'PASS':1,'NUMERIC_INCONCLUSIVE':1},status='PARTIAL_WITH_LIMITATIONS',all_selected_constraints_and_declared_checks_pass=False,
        accuracy_evaluated=False,training_target_changed=False,model_selection_changed=False,TEST_opened=False,
        finite_numerical_scope='FINITE_SYNTHETIC_SAMPLING_NOT_GLOBAL_OR_ACCURACY_PROOF')
    write('m02_result',m);write('m02_seal',dict(status='VERIFIED',result_sha256=spec['m02_result']['sha256']))
    t=dict(status='PASS',expected_regions=2,completed_regions=2,reused_regions=1,refreshed_regions=1,
        positive_bound_regions=1,above_1_percent_regions=0,maximum_bound_percent=.1,
        TEST_targets_opened=False,new_labels_or_fits=False,selected_models_changed=False,exact_physical_cost_truth_claim=False)
    write('t08_result',t);write('t08_intent',dict(config={'training_result':spec['training_result']}))
    write('t08_manifest',dict(artifacts=[dict(relative_path='RESULT.json',sha256=spec['t08_result']['sha256']),dict(relative_path='INTENT.json',sha256=spec['t08_intent']['sha256'])]))
    write('t08_seal',dict(verified=True))
    return spec,pins,write


def test_partial_property_and_training_screen_counts_are_not_global_proofs(tmp_path):
    spec,pins,_=audit_fixture(tmp_path);d=e.load_evidence(spec,**pins)
    assert d['facts']['audits']['properties']['dispositions']['NUMERIC_INCONCLUSIVE']==1
    assert d['facts']['audits']['training_limits']['completed']==2
    prose=' '.join(d['paragraphs']['validation'])
    assert 'NUMERIC_INCONCLUSIVE: 1' in prose and 'not a global' in prose and 'do not prove' in prose


@pytest.mark.parametrize('change',['m_selection','m_count','m_seal','m_false_pass','t_intent','t_count','t_manifest','t_scope'])
def test_audit_bindings_counts_and_scope_fail_closed(tmp_path,change):
    spec,pins,write=audit_fixture(tmp_path)
    key={'m_selection':'m02_result','m_count':'m02_result','m_seal':'m02_seal','m_false_pass':'m02_result',
         't_intent':'t08_intent','t_count':'t08_result','t_manifest':'t08_manifest','t_scope':'t08_result'}[change]
    d=json.loads(Path(spec[key]['path']).read_text())
    if change=='m_selection':d['registry_binding']['sha256']='b'*64
    elif change=='m_count':d['dispositions']['PASS']=2
    elif change=='m_seal':d['result_sha256']='b'*64
    elif change=='m_false_pass':d['all_selected_constraints_and_declared_checks_pass']=True
    elif change=='t_intent':d['config']['training_result']['sha256']='b'*64
    elif change=='t_count':d['completed_regions']=1
    elif change=='t_manifest':d['artifacts'][0]['sha256']='b'*64
    elif change=='t_scope':d['TEST_targets_opened']=True
    write(key,d)
    with pytest.raises(ValueError):e.load_evidence(spec,**pins)


def test_missing_seals_do_not_become_verified(tmp_path):
    spec,pins,_=audit_fixture(tmp_path);spec['m02_seal']=None;spec['t08_seal']=None
    d=e.load_evidence(spec,**pins)
    assert all(not r['seal_verified'] for r in d['facts']['audits'].values())


def documentary_fixture(tmp_path):
    spec,pins,write=audit_fixture(tmp_path)
    m=json.loads(Path(spec['m02_result']['path']).read_text())
    m.update(documentary_recovery=True,completed=0,uncompleted=2,
        dispositions={'INVALID_RESULT_METADATA':1,'UNATTEMPTED':1},
        terminal_reason='OPERATIONAL_SERIALIZATION_FAILURE_AND_ORIGINAL_600_SECOND_BUDGET_EXHAUSTED',
        original_shared_start_epoch=1000.,original_audit_deadline_epoch=1600.)
    write('m02_result',m)
    write('m02_seal',dict(status='VERIFIED',result_sha256=spec['m02_result']['sha256']))
    return spec,pins,write,m


def test_documentary_zero_completion_and_original_window_propagate(tmp_path):
    spec,pins,_,_=documentary_fixture(tmp_path)
    m=e.load_evidence(spec,**pins)['facts']['audits']['properties']
    assert m['documentary_recovery'] is True and m['completed']==0 and m['uncompleted']==2
    assert m['original_shared_start_epoch']==1000 and m['original_audit_deadline_epoch']==1600
    assert m['original_audit_allowance_seconds']==600
    assert m['terminal_reason']=='OPERATIONAL_SERIALIZATION_FAILURE_AND_ORIGINAL_600_SECOND_BUDGET_EXHAUSTED'


@pytest.mark.parametrize('change',[dict(documentary_recovery='true'),dict(completed=1,uncompleted=1),
    dict(uncompleted=1),dict(terminal_reason='TIME_LIMIT'),dict(original_audit_deadline_epoch=1601),
    dict(original_shared_start_epoch=float('nan')),dict(dispositions={'PASS':1,'UNATTEMPTED':1}),dict(status='PASS')])
def test_documentary_inconsistencies_rejected(tmp_path,change):
    spec,pins,write,m=documentary_fixture(tmp_path);m.update(change)
    write('m02_result',m);write('m02_seal',dict(status='VERIFIED',result_sha256=spec['m02_result']['sha256']))
    with pytest.raises(ValueError):e.load_evidence(spec,**pins)


def test_legacy_property_schema_does_not_invent_documentary_counts(tmp_path):
    spec,pins,_=audit_fixture(tmp_path)
    m=e.load_evidence(spec,**pins)['facts']['audits']['properties']
    assert all(k not in m for k in ['documentary_recovery','completed','uncompleted','terminal_reason','original_audit_deadline_epoch'])


def test_cross_run_t08_seal_flag_never_establishes_archive_linkage(tmp_path):
    spec,pins,write=audit_fixture(tmp_path)
    write('t08_seal',dict(verified=True,zip_sha256='f'*64,other_run=True))
    d=e.load_evidence(spec,**pins);a=d['facts']['audits']['training_limits']
    assert a['status']=='PASS' and a['completed']==2
    assert a['seal_flag_recorded'] and a['manifest_identity_verified']
    assert not a['seal_verified'] and a['archive_linkage']=='NOT_ESTABLISHED_BY_METADATA_ONLY_ADAPTER'
    assert any('archive linkage is not established' in x for x in d['paragraphs']['validation'])


def test_failed_audit_status_remains_visible_despite_counts_or_seal(tmp_path):
    spec,pins,write=audit_fixture(tmp_path)
    t=json.loads(Path(spec['t08_result']['path']).read_text());t['status']='FAIL';write('t08_result',t)
    spec['t08_seal']=None;spec['t08_manifest']=None
    d=e.load_evidence(spec,**pins)
    assert any('outcome of fail' in x for x in d['paragraphs']['validation'])


def test_snapshot_exact_bytes_and_cross_stage_mutation_rejection(tmp_path):
    spec,pins,_=fixture(tmp_path);d=e.load_evidence(spec,**pins);root=tmp_path/'bundle'
    binding=e.snapshot_evidence(root,d);record=e.verify_snapshot(root,binding,evidence=d)
    assert set(record['snapshots'])==set(d['bindings'])
    for key,row in record['snapshots'].items():assert (root/row['relative_path']).read_bytes()==Path(d['bindings'][key]['path']).read_bytes()
    with pytest.raises(FileExistsError):e.snapshot_evidence(root,d)
    row=next(iter(record['snapshots'].values()));(root/row['relative_path']).write_text('{}')
    with pytest.raises(ValueError):e.verify_snapshot(root,binding,evidence=d)


def test_snapshot_rejects_changed_source_or_facts(tmp_path):
    spec,pins,_=fixture(tmp_path);d=e.load_evidence(spec,**pins);root=tmp_path/'bundle'
    binding=e.snapshot_evidence(root,d);altered=copy.deepcopy(d);altered['facts']['training']['unused_seconds']=999
    with pytest.raises(ValueError):e.verify_snapshot(root,binding,evidence=altered)
    Path(spec['training_clock']['path']).write_text('{}')
    with pytest.raises(ValueError):e.load_evidence(spec,**pins)


def test_snapshot_traversal_rejected(tmp_path):
    spec,pins,_=fixture(tmp_path);d=e.load_evidence(spec,**pins);root=tmp_path/'bundle';b=e.snapshot_evidence(root,d)
    b['relative_path']='../outside.json'
    with pytest.raises(ValueError):e.verify_snapshot(root,b,evidence=d)


@pytest.mark.parametrize('raw',['{"x":1,"x":2}','{"x":NaN}'])
def test_ambiguous_json_rejected(tmp_path,raw):
    p=tmp_path/'bad.json';p.write_text(raw)
    with pytest.raises(ValueError):e._read(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()),lambda:None)
