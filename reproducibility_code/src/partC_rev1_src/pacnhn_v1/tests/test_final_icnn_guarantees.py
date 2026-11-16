"""Only synthetic models/metadata. Never discovers campaign data or starts an audit."""
import copy
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from partC_rev1_src.pacnhn_v1 import final_icnn_guarantees as g
from partC_rev1_src.pacnhn_v1.models import MonotoneICNN,PortableICNN,save_model


def toy(m=7,units=1.,amplitude=3.):
    scale=np.geomspace(1e-5,1e5,m)
    model=MonotoneICNN(np.linspace(0,3,m)*units,scale,eta=np.linspace(0,.4,m)*units,
        width=5,seed=83,amplitude=amplitude*units)
    return model.export_model()


@pytest.mark.parametrize('m',[1,2,7,31])
@pytest.mark.parametrize('amplitude',[0.,3.])
def test_finite_properties_real_portable_synthetic(m,amplitude):
    result=g.audit_model(toy(m,amplitude=amplitude))
    assert result['status']=='PASS',result['failed_checks']
    assert all(x['passed'] for x in result['checks'])
    assert len(result['finite_differences'])==3
    assert all(len(x['steps'])==3 for x in result['finite_differences'])
    assert result['accuracy_evaluated'] is False


@pytest.mark.parametrize('units',[1e-8,1e8])
def test_tolerances_are_cost_unit_scaled(units):
    original=g.audit_model(toy());scaled=g.audit_model(toy(units=units))
    assert original['status']==scaled['status']=='PASS'
    a=original['finite_differences'][0]['steps'][0]['gradient']['tolerance']
    b=scaled['finite_differences'][0]['steps'][0]['gradient']['tolerance']
    assert b/a==pytest.approx(units,rel=1e-9)


def test_all_zero_and_linear_models_are_not_strict_convexity_claims():
    model=toy(amplitude=0);a=model.export_arrays();a['c0'][:]=0;a['eta'][:]=0
    result=g.audit_model(PortableICNN(a))
    assert result['status']=='PASS' and result['inactive_amplitude']
    assert all(r.get('quadratic',0)==0 for r in result['checks'])


def test_roundoff_dominated_curvature_is_inconclusive_not_false_pass():
    model=MonotoneICNN([1e30],[1.],eta=[1.],width=5,seed=9,amplitude=0.).export_model()
    result=g.audit_model(model)
    assert result['status']=='NUMERIC_INCONCLUSIVE'
    assert all(not s['hvp']['signal_resolved'] for d in result['finite_differences'] for s in d['steps'])
    assert all(name.startswith('FD_HVP') for name in result['failed_checks'])


def test_signal_resolved_fd_only_mismatch_is_hard_failure(monkeypatch):
    # Corrupt every reconstructed instance identically, so export/symmetry/PSD
    # still pass. Only independent cost finite differences expose the error.
    original=PortableICNN.hess_vec
    monkeypatch.setattr(PortableICNN,'hess_vec',lambda self,u,v:1.25*original(self,u,v))
    result=g.audit_model(toy())
    assert result['status']=='FAIL'
    assert result['failed_checks'] and all(x.startswith('FD_HVP') for x in result['failed_checks'])
    assert any(d['hvp_disposition']=='FAIL' for d in result['finite_differences'])


def test_incorrect_zero_hvp_with_resolved_cost_curvature_is_hard_failure(monkeypatch):
    model=MonotoneICNN([0.],[1.],eta=[1.],width=5,seed=9,amplitude=0.).export_model()
    monkeypatch.setattr(PortableICNN,'hess_vec',lambda self,u,v:np.zeros_like(v))
    result=g.audit_model(model)
    assert result['status']=='FAIL'
    assert result['failed_checks'] and all(x.startswith('FD_HVP') for x in result['failed_checks'])
    assert all(d['hvp_disposition']=='FAIL' for d in result['finite_differences'])
    assert all(s['hvp']['signal_resolved'] for d in result['finite_differences'] for s in d['steps'])


def test_negative_hessian_is_a_failure_not_fit_inconclusive(monkeypatch):
    model=toy();original=model.hess_vec
    monkeypatch.setattr(model,'hess_vec',lambda u,v:-original(u,v)-v/model.scale**2)
    result=g.audit_model(model)
    assert result['status']=='FAIL'
    assert any(x.startswith('quadratic_') for x in result['failed_checks'])


def test_nonsymmetric_hvp_is_detected(monkeypatch):
    model=toy();original=model.hess_vec
    def corrupt(u,v):
        h=original(u,v);h[0]+=v[1]/(model.scale[0]*model.scale[1]);return h
    monkeypatch.setattr(model,'hess_vec',corrupt)
    result=g.audit_model(model)
    assert result['status']=='FAIL'
    assert any('symmetry' in x for x in result['failed_checks'])


def test_value_gradient_mismatch_is_not_hidden_by_one_good_step(monkeypatch):
    model=toy();original=model.value_cost
    def corrupt(u):
        value,cost=original(u);return value+float(np.sum(np.asarray(u)/model.scale)),cost
    monkeypatch.setattr(model,'value_cost',corrupt)
    result=g.audit_model(model)
    assert result['status']!='PASS'
    assert any(x.startswith('FD_gradient') for x in result['failed_checks'])


def test_guard_interrupts_before_any_new_model_call():
    calls=[]
    def guard():
        calls.append(1)
        if len(calls)==3:raise TimeoutError('synthetic cap')
    with pytest.raises(TimeoutError,match='synthetic cap'):g.audit_model(toy(),guard=guard)
    assert len(calls)==3


def test_fd_requires_adjacent_accuracy_and_refinement():
    def row(error,bound=0,passed=True):return {'gradient':{'pass':passed,'absolute_error':error,'roundoff_bound':bound}}
    assert not g._fd_pair_accept([row(1),row(3),row(9)],'gradient')
    assert not g._fd_pair_accept([row(1),row(.1,passed=False),row(.01)],'gradient')
    assert g._fd_pair_accept([row(1),row(.1),row(.01)],'gradient')
    assert g._fd_pair_accept([row(0),row(.1,bound=.1)],'gradient')


def test_fd_disposition_separates_unresolved_signal_from_resolved_error():
    def row(error,passed,resolved):return {'hvp':dict(absolute_error=error,roundoff_bound=0.,
        signal_resolved=resolved,**{'pass':passed})}
    assert g.fd_disposition([row(1,False,False)]*3,'hvp')=='NUMERIC_INCONCLUSIVE'
    assert g.fd_disposition([row(1,False,False),row(1,False,True),row(1,False,False)],'hvp')=='FAIL'
    assert g.fd_disposition([row(.1,True,True),row(1,True,True),row(10,True,True)],'hvp')=='FAIL'
    assert g.fd_disposition([row(1,True,True),row(.1,True,True)],'hvp')=='PASS'


def inventory_fixture(tmp_path):
    records=[];manifest=[];bindings=[];selected=[]
    for i in range(3):
        p=tmp_path/f'{i}.npz';digest=str(i)*64
        records.append(dict(suite_id='s03',family='ICNN',region_index=i,status='AVAILABLE',model_path=str(p),
            model_sha256=digest,model_kind='MONOTONE_ICNN_POWER4',artifact_hashes={str(p):digest}))
        manifest.append(dict(suite_id='s03',region_index=str(i),model_path=str(p),model_sha256=digest,recorded_compressed_bytes='100'))
        bindings.append(dict(path=str(p),sha256=digest,role='FROZEN_ICNN_COST_MODEL'))
        selected.append(dict(suite_id='s03',family='ICNN',region_index=str(i),status='AVAILABLE',model_sha256=digest))
    bindings.append(dict(path=str(tmp_path/'power4.npz'),sha256='d'*64,role='FROZEN_POWER4_COST_MODEL'))
    return manifest,records,bindings,selected


def test_inventory_filters_old_family_bindings_and_keeps_exact_ids(tmp_path):
    data=inventory_fixture(tmp_path)
    result=g.reconcile_inventory(*data,expected_count=3)
    assert [r['region_index'] for r in result]==[0,1,2]


@pytest.mark.parametrize('which',[0,1,2,3])
def test_inventory_rejects_each_independent_hash_disagreement(tmp_path,which):
    data=inventory_fixture(tmp_path);key='sha256' if which==2 else 'model_sha256'
    data[which][0][key]='f'*64
    with pytest.raises(ValueError):g.reconcile_inventory(*data,expected_count=3)


@pytest.mark.parametrize('which',[0,1,2,3])
def test_inventory_rejects_duplicates(tmp_path,which):
    data=inventory_fixture(tmp_path);data[which].append(copy.deepcopy(data[which][0]))
    with pytest.raises(ValueError):g.reconcile_inventory(*data,expected_count=3)


def test_missing_available_population_cannot_shrink_expected_denominator(tmp_path):
    data=inventory_fixture(tmp_path)
    with pytest.raises(ValueError):g.reconcile_inventory(*data,expected_count=1499)
    data[1][0]['status']='UNAVAILABLE_STALE_CURRENT_DATASET'
    with pytest.raises(ValueError):g.reconcile_inventory(*data,expected_count=3)


def test_companion_targets_cannot_enter_model_only_audit(tmp_path):
    data=inventory_fixture(tmp_path)
    data[1][0]['artifact_hashes'][str(tmp_path/'TEST.npz')]='a'*64
    with pytest.raises(ValueError,match='self-contained'):g.reconcile_inventory(*data,expected_count=3)


def stage3_fixture():
    rows=[]
    for solver in ('CFW','FW','MSA'):
        for s in ('open','i95'):rows.append(dict(case_id=f'P_{solver}_{s}',priority='PRIMARY'))
    for family in ('FFTT','POWER4','XGBOOST','ICNN'):
        for s in ('open','i95'):rows.append(dict(case_id=f'AC_s03_{family}_{s}',priority='PRIMARY'))
    result=dict(status='COMPLETE_WITH_RECORDED_OUTCOMES',primary_rows=14,rows=14)
    outcomes=[dict(case=copy.deepcopy(r),status='TIME_LIMIT') for r in rows]
    return result,rows,outcomes


def test_stage3_terminal_accepts_explicit_censoring_not_running():
    data=stage3_fixture();g.validate_stage3(*data)
    data[2][0]['status']='RUNNING'
    with pytest.raises(ValueError):g.validate_stage3(*data)


@pytest.mark.parametrize('mutation',['missing','duplicate','bad_primary','wrong_case'])
def test_stage3_exact_fourteen_case_coverage(mutation):
    result,matrix,outcomes=stage3_fixture()
    if mutation=='missing':outcomes.pop()
    elif mutation=='duplicate':outcomes[-1]=outcomes[0]
    elif mutation=='bad_primary':matrix[-1]['priority']='ADDITIONAL'
    else:outcomes[0]['case']['extra']='wrong'
    with pytest.raises(ValueError):g.validate_stage3(result,matrix,outcomes)


def test_metadata_write_is_exclusive_and_json_rejects_duplicates(tmp_path):
    p=tmp_path/'record.json';g.new_json(p,{'value':1})
    with pytest.raises(FileExistsError):g.new_json(p,{'value':2})
    q=tmp_path/'bad.json';q.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError):g.read_json(q)


def test_model_identity_checked_before_load_and_uncompressed_cap(tmp_path):
    model=MonotoneICNN([1.,2.],[.2,3.],width=32,seed=2).export_model()
    p=tmp_path/'synthetic.npz';save_model(model,p)
    row=dict(model_path=str(p),model_sha256=g.file_sha(p),recorded_compressed_bytes=p.stat().st_size,artifact_hashes={})
    assert isinstance(g._load_one(row,lambda:None),PortableICNN)
    row['model_sha256']='f'*64
    with pytest.raises(ValueError,match='bytes differ'):g._load_one(row,lambda:None)


def test_pilot_order_is_deterministic_without_repeating_models():
    rows=[dict(region_index=i,recorded_compressed_bytes=i+1) for i in range(100)]
    ordered=g.pilot_order(rows)
    assert [r['region_index'] for r in ordered[:8]]==[99,98,97,0,96,1,2,3]
    assert len(ordered)==len({r['region_index'] for r in ordered})==100


@pytest.mark.parametrize('states,forecast,remaining,allowed',[
    (['PASS']*8,10.,20.,True),
    (['PASS']*7+['NUMERIC_INCONCLUSIVE'],10.,20.,True),
    (['NUMERIC_INCONCLUSIVE']*8,10.,20.,True),
    (['PASS']*7+['FAIL'],10.,20.,False),
    (['PASS']*7+['TIME_LIMIT'],10.,20.,False),
    (['PASS']*7+['NUMERIC_INCONCLUSIVE'],21.,20.,False),
])
def test_pilot_preserves_inconclusive_but_continues_independent_models(states,forecast,remaining,allowed):
    decision=g.pilot_continuation([{'status':s} for s in states],forecast,remaining)
    assert decision['continue_main'] is allowed
    if allowed and 'NUMERIC_INCONCLUSIVE' in states:assert decision['status']=='COMPLETE_WITH_NUMERICAL_INCONCLUSIVES'


def test_owned_cleanup_never_creates_or_kills_unrelated_processes():
    class Child:
        def __init__(self):self.calls=[]
        def poll(self):return None
        def terminate(self):self.calls.append('terminate')
        def wait(self,timeout):
            self.calls.append(('wait',timeout))
            if len(self.calls)==2:raise subprocess.TimeoutExpired('synthetic',timeout)
        def kill(self):self.calls.append('kill')
    child=Child();g._retire_owned(child)
    assert child.calls==['terminate',('wait',3),'kill',('wait',3)]


def admission_fixture(tmp_path,monkeypatch):
    from partC_rev1_src.pacnhn_v1 import continuation,handoff
    root=tmp_path/'out';root.mkdir();now=2000.
    def put(name,value):
        p=root/name;g.new_json(p,value);return dict(path=str(p),sha256=g.file_sha(p))
    training=put('training_result.json',{'synthetic':True});registry=put('registry.json',{'synthetic':True})
    pins=dict(training_result_path=training['path'],training_result_sha256=training['sha256'],registry_path=registry['path'],registry_sha256=registry['sha256'])
    campaign=put('campaign.json',dict(run_id='synthetic',output_root=str(root),continuation=pins))
    stage4=put('STAGE4_CLOCK.json',{'synthetic':True})
    amendment=put('amendment.json',{'synthetic':True})
    shared=put('final_audits/AUDIT_CLOCK.json',dict(schema='PACNHN_FINAL_AUDIT_CLOCK_V1',phase_id='PACNHN_V1',run_id='synthetic',
        stage4_clock_sha256=stage4['sha256'],start_epoch=1900.,deadline_epoch=3700.))
    clock=put('STAGE3_CLOCK.json',{'original_stage_start_epoch':1000.,'deadline_epoch':1500.})
    terminal,matrix,outcomes=stage3_fixture();terminal['clock']=g.read_json(clock['path'])
    sources={rel:{'path':str(Path(g.__file__).parent/rel),'sha256':g.file_sha(Path(g.__file__).parent/rel)} for rel in g.REQUIRED_SOURCES}
    pre=put('preflight.json',dict(status='PASS',exit_code=0,inputs_stable_during_tests=True,config_sha256=campaign['sha256'],
        source_snapshot={v['path']:v['sha256'] for v in sources.values()}))
    cfg=dict(schema=g.SCHEMA,phase_id='PACNHN_V1',run_id='synthetic',step_id='M02_FINAL_GUARANTEES',
        tasks=g.TASKS,expected_outputs=g.OUTPUTS,acceptance_tests=['synthetic'],numerics=g.PARAMETERS,budget=g.BUDGET,
        expected_models=1499,suite_id='s03',family='ICNN',campaign_config=campaign,stage4_clock=stage4,clock_amendment=amendment,
        training_result=training,registry=registry,
        deadline_epoch=4000.,shared_audit_clock=shared,output_root=str(root/'audits/new'),stage3_clock=clock,
        stage3_result=put('STAGE3_RESULT.json',terminal),stage3_matrix=put('matrix.json',matrix),stage3_outcomes=put('outcomes.json',outcomes),
        source_bindings=sources,preflight=pre,proof_bindings=[put('proof.json',{'proof':'synthetic'}),put('proof_receipt.json',{'proof':'synthetic'})],other_owned_processes=[])
    cp=put('audit_config.json',cfg)
    ap=put('admission.json',dict(schema='PACNHN_M02_ADMISSION_V1',decision='ADMIT_POST_STAGE3_PROPERTY_AUDIT',config_sha256=cp['sha256'],
        preflight_sha256=pre['sha256'],phase_id='PACNHN_V1',run_id='synthetic'))
    calls=[]
    def auth(c,action,now):calls.append(action);return {'synthetic':True}
    monkeypatch.setattr(continuation,'validate_continuation',auth)
    monkeypatch.setattr(handoff,'publication_window',lambda *a,**k:dict(stage_start_epoch=1800.,stage_deadline_epoch=4000.,deadline_epoch=4000.))
    return root,cfg,cp,ap,calls,now


def test_admission_requires_analysis_authority_and_shared_nonreset_clock(tmp_path,monkeypatch):
    root,cfg,cp,ap,calls,now=admission_fixture(tmp_path,monkeypatch)
    _,record=g.admit(cp['path'],cp['sha256'],ap['path'],now=now)
    assert calls==['analysis'] and record['audit_deadline_epoch']==2500.
    assert not (root/'audits/new').exists()
    with pytest.raises(TimeoutError):g.admit(cp['path'],cp['sha256'],ap['path'],now=2470.)


@pytest.mark.parametrize('mutation',['budget','source','shared_clock','stage3','config_hash','preflight'])
def test_admission_rejects_drift_without_creating_output(tmp_path,monkeypatch,mutation):
    root,cfg,cp,ap,_,now=admission_fixture(tmp_path,monkeypatch)
    if mutation=='config_hash':cp['sha256']='f'*64
    elif mutation=='source':cfg['source_bindings']['models.py']['sha256']='e'*64
    elif mutation=='budget':cfg['budget']=dict(g.BUDGET,seconds=601)
    elif mutation=='shared_clock':cfg['shared_audit_clock']['sha256']='a'*64
    elif mutation=='stage3':cfg['stage3_result']['sha256']='b'*64
    else:cfg['preflight']['sha256']='c'*64
    if mutation not in ('config_hash',):
        Path(cp['path']).write_text(json.dumps(cfg));cp['sha256']=g.file_sha(cp['path'])
        a=g.read_json(ap['path']);a['config_sha256']=cp['sha256'];Path(ap['path']).write_text(json.dumps(a))
    with pytest.raises((ValueError,TimeoutError)):g.admit(cp['path'],cp['sha256'],ap['path'],now=now)
    assert not (root/'audits/new').exists()


@pytest.mark.parametrize('mutation',['alternate','reset','wrong_stage4','long_window'])
def test_shared_clock_cannot_be_replaced_reset_or_rebound(tmp_path,monkeypatch,mutation):
    root,cfg,cp,ap,_,now=admission_fixture(tmp_path,monkeypatch)
    p=Path(cfg['shared_audit_clock']['path']);clock=g.read_json(p)
    if mutation=='alternate':p=root/'alternate.json'
    elif mutation=='reset':clock['start_epoch']=2100.;clock['deadline_epoch']=3900.
    elif mutation=='wrong_stage4':clock['stage4_clock_sha256']='f'*64
    else:clock['deadline_epoch']=3701.
    p.write_text(json.dumps(clock));cfg['shared_audit_clock']={'path':str(p),'sha256':g.file_sha(p)}
    Path(cp['path']).write_text(json.dumps(cfg));cp['sha256']=g.file_sha(cp['path'])
    a=g.read_json(ap['path']);a['config_sha256']=cp['sha256'];Path(ap['path']).write_text(json.dumps(a))
    with pytest.raises(ValueError):g.admit(cp['path'],cp['sha256'],ap['path'],now=now)


@pytest.mark.parametrize('failure',['rss','peak','available','birth'])
def test_resource_guards_fail_before_more_work(monkeypatch,failure):
    import psutil
    from types import SimpleNamespace
    class Proc:
        pid=123
        def memory_info(self):return SimpleNamespace(rss=g.BUDGET['process_rss_bytes']+1 if failure=='rss' else 100,
            peak_wset=g.BUDGET['process_rss_bytes']+1 if failure=='peak' else 100)
        def children(self,recursive):return []
        def create_time(self):return 100.
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=1 if failure=='available' else 10**12))
    monkeypatch.setattr(psutil,'Process',lambda pid:Proc())
    others=[dict(pid=456,create_time=101.)] if failure=='birth' else []
    with pytest.raises((MemoryError,ValueError)):g.resource_observation(Proc(),others)


def test_noop_or_worker_failure_cannot_become_complete_property_pass(tmp_path):
    import time
    manifest=tmp_path/'inventory.csv'
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['region_index','model_sha256']);w.writeheader()
        w.writerows(dict(region_index=i,model_sha256='a'*64) for i in range(1499))
    out=tmp_path/'out';out.mkdir()
    c=dict(model_manifest=dict(path=str(manifest),sha256=g.file_sha(manifest)),source_bindings={},proof_bindings=[],registry={})
    result=g.seal_outputs(out,c,supervisor=dict(worker_exit_code=0,status='FINISHED',deadline_epoch=time.time()+60,shared_start_epoch=time.time()))
    assert result['dispositions']=={'UNATTEMPTED':1499}
    assert result['all_selected_constraints_and_declared_checks_pass'] is False
    assert result['status']=='PARTIAL_WITH_LIMITATIONS'
    assert g.read_json(out/'SEAL_RECEIPT.json')['status']=='VERIFIED'


def test_expired_sealing_preserves_outputs_without_verified_receipt(tmp_path):
    import time
    out=tmp_path/'out';out.mkdir();g.new_json(out/'WORKER_RESULT.json',{'status':'FAIL'})
    with pytest.raises(TimeoutError):g.seal_outputs(out,{},supervisor={'deadline_epoch':time.time()-1})
    assert (out/'WORKER_RESULT.json').exists() and not (out/'SEAL_RECEIPT.json').exists()


def test_native_numpy_diagnostics_serialize_without_type_loss(tmp_path):
    path=tmp_path/'result.json'
    g.new_json(path,dict(checks=[dict(passed=np.bool_(False),count=np.int64(3),error=np.float64(.25))]))
    assert g.read_json(path)==dict(checks=[dict(passed=False,count=3,error=.25)])


@pytest.mark.parametrize('value',[float('nan'),np.float64('inf'),object()])
def test_invalid_serialization_does_not_leave_empty_native_result(tmp_path,value):
    path=tmp_path/'result.json'
    with pytest.raises((ValueError,TypeError)):g.new_json(path,dict(value=value))
    assert not path.exists()


def test_empty_native_result_is_preserved_as_failed_disposition(tmp_path):
    import time
    manifest=tmp_path/'inventory.csv'
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['region_index','model_sha256']);w.writeheader()
        w.writerows(dict(region_index=i,model_sha256='a'*64) for i in range(1499))
    out=tmp_path/'out';(out/'model_results').mkdir(parents=True)
    corrupt=out/'model_results/000000.json';corrupt.write_bytes(b'')
    c=dict(model_manifest=dict(path=str(manifest),sha256=g.file_sha(manifest)),source_bindings={},proof_bindings=[],registry={})
    result=g.seal_outputs(out,c,supervisor=dict(worker_exit_code=1,status='FINISHED',deadline_epoch=time.time()+60,shared_start_epoch=time.time()))
    assert result['dispositions']=={'INVALID_RESULT_METADATA':1,'UNATTEMPTED':1498}
    assert result['all_selected_constraints_and_declared_checks_pass'] is False
    assert corrupt.read_bytes()==b''


@pytest.mark.parametrize('mutation',['training_result','registry','preflight_config'])
def test_root_admission_cannot_substitute_frozen_selection_or_preflight_config(tmp_path,monkeypatch,mutation):
    root,cfg,cp,ap,_,now=admission_fixture(tmp_path,monkeypatch)
    if mutation=='preflight_config':
        p=Path(cfg['preflight']['path']);value=g.read_json(p);value['config_sha256']='f'*64
        p.write_text(json.dumps(value));cfg['preflight']['sha256']=g.file_sha(p)
    else:
        p=root/(mutation+'_alternate.json');g.new_json(p,{'self_consistent_but_not_frozen':True})
        cfg[mutation]={'path':str(p),'sha256':g.file_sha(p)}
    Path(cp['path']).write_text(json.dumps(cfg));cp['sha256']=g.file_sha(cp['path'])
    a=g.read_json(ap['path']);a['config_sha256']=cp['sha256'];a['preflight_sha256']=cfg['preflight']['sha256']
    Path(ap['path']).write_text(json.dumps(a))
    with pytest.raises(ValueError):g.admit(cp['path'],cp['sha256'],ap['path'],now=now)
