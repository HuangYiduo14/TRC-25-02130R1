"""Synthetic finalization regressions; no fitting, oracle, or TEST array reads."""
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from partC_rev1_src.pacnhn_v1.control import array_hash,read_json,sha,write_json

STAGED = Path(__file__).with_name('training.py')
if STAGED.exists():
    spec=importlib.util.spec_from_file_location('partC_rev1_src.pacnhn_v1._training_reassessment_staged',STAGED)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
else:
    from partC_rev1_src.pacnhn_v1 import training as module


@pytest.fixture
def case(tmp_path,monkeypatch):
    root=tmp_path/'training';regional=root/'regions/s03/000000';labels=regional/'labels'
    labels.mkdir(parents=True)
    graph=tmp_path/'graph.npz';graph.write_bytes(b'immutable synthetic graph')
    write_json(tmp_path/'SUPPORT_FREEZE.json',{'status':'PASS','synthetic':True})
    deadline=time.time()+10000
    write_json(root/'TRAINING_CLOCK.json',{'stage_deadline_epoch':deadline,'stage_start_epoch':deadline-86400})
    config=dict(audit_batch_rows=1,train_target_wmape_percent=1.,eligible_pool_policy='UNCHANGED',
        finalization_minimum_reserve_seconds=10.,region_slice_seconds=1.,final_evaluation_reserve_seconds=15.)
    region=SimpleNamespace(graph_sha256=sha(graph),identity='graph-id',support_identity='support-id',
        c0=np.array([1.]),pair_keys=np.array([11],np.uint64),m=1)
    record=dict(key='s03:000000',suite_id='s03',region_index=0,path=str(graph),sha256=sha(graph),pairs=1,graph_file_bytes=graph.stat().st_size)
    monkeypatch.setattr(module,'load_region_graph',lambda *a,**kw:region)
    datasets={}
    entries={}
    for role,n in [('TRAIN',3),('VALID',2)]:
        u=np.arange(1,n+1,dtype=float).reshape(-1,1);tau=np.full_like(u,2.)
        datasets[role]=dict(u=u,tau=tau,c0=region.c0,unique_rows=n,dataset_identity=role+'-current',
            identity=dict(graph_sha256=region.graph_sha256,graph_identity=region.identity,support_identity=region.support_identity))
        for i in range(n):
            f=labels/(role+str(i)+'.bin');f.write_bytes((role+str(i)).encode())
            r=labels/(role+str(i)+'.json');write_json(r,{'role':role,'admitted':True})
            entries[role+str(i)]=dict(role=role,selected_label=dict(path=str(f),sha256=sha(f),receipt=str(r),receipt_sha256=sha(r)))
    entries['blind']=dict(role='TEST',selected_label={'path':'DO_NOT_OPEN_TEST','sha256':'sealed','receipt':'DO_NOT_OPEN_TEST_RECEIPT','receipt_sha256':'sealed'})
    write_json(labels/'REQUEST_INDEX.json',entries)
    write_json(labels/'REGION_IDENTITY.json',datasets['TRAIN']['identity'])
    seen_roles=[]
    def loader(path,*,role):
        assert role in ('TRAIN','VALID'),'TEST must remain sealed'
        seen_roles.append(role)
        for row in read_json(Path(path)/'REQUEST_INDEX.json').values():
            if row['role']==role:
                selected=row['selected_label']
                assert sha(selected['path'])==selected['sha256']
                assert sha(selected['receipt'])==selected['receipt_sha256']
        return datasets[role]
    monkeypatch.setattr(module,'load_admitted_dataset',loader)
    loaded=[]
    class Model:
        kind='SYNTHETIC_COST';c0=np.array([1.]);amplitude=.25
        def cost(self,u):return np.full_like(u,2.1)
        def unit_neural_cost(self,u):return np.full_like(u,.4)
    monkeypatch.setattr(module,'load_model',lambda path:(loaded.append(str(path)),Model())[1])
    def forbidden(*a,**kw):raise AssertionError('Fitting or selection is forbidden')
    for name in ['fit_power4_cost','fit_xgboost_cost','update_icnn','audit_and_select']:
        monkeypatch.setattr(module,name,forbidden)
    metric=dict(WMAPE_percent=0.,amortized_seconds_per_query=.001,batch_inference_seconds=.002,unique_inputs=2)
    old=dict(train_dataset_identity='TRAIN-before-last-admission',valid_dataset_identity='VALID-before-last-admission',
        TRAIN=metric,VALID=metric,model_load_seconds=.01,model_hash_seconds=.001)
    selected={}
    for family in module.FAMILIES:
        model=regional/(family+'.npz');model.write_bytes(family.encode())
        audit=dict(copy.deepcopy(old),model_sha256=sha(model))
        audit_path=regional/(family+'_prior.json');write_json(audit_path,audit)
        selected[family]=dict(candidate_id=family+'_selected',model_path=str(model),model_sha256=sha(model),
            model_kind=Model.kind,audit=audit,audit_path=str(audit_path),total_updates=250,trained_dataset_identity='TRAIN-actual-fit')
    power=regional/'POWER4.npz';power.write_bytes(b'POWER4')
    power4=dict(copy.deepcopy(old),model_path=str(power),model_sha256=sha(power),model_kind=Model.kind)
    state=dict(selected=selected,power4=power4,visits=4,dataset_identity=old['train_dataset_identity'],
        valid_dataset_identity=old['valid_dataset_identity'],audited_request_index_sha256='before-last-admission',
        training_rows=2,valid_rows=2,label_counts={'TRAIN':3,'VALID':2,'TEST':1},
        last_timing=dict(dataset_load=.01,input_load=.001,graph_file_bytes=graph.stat().st_size),
        eligible={'ICNN':[{'candidate_id':'unselected_better_VALID','model_path':'DO_NOT_LOAD_ORPHAN'}]})
    write_json(regional/'STATE.json',state)
    ctx=SimpleNamespace(root=tmp_path,c={'suite_ids':['s03']},guard=lambda **kw:None)
    return SimpleNamespace(root=root,regional=regional,labels=labels,states={record['key']:state},record=record,
        config=config,deadline=deadline,ctx=ctx,support_sha=sha(tmp_path/'SUPPORT_FREEZE.json'),
        datasets=datasets,seen_roles=seen_roles,loaded=loaded,Model=Model,region=region)


def reassess(c):
    return module.reassess_selected_before_freeze(c.ctx,c.root,[c.record],c.states,c.support_sha,c.config,c.deadline)


def test_exact_selections_reassessed_without_fit_reselection_or_test(case):
    c=case;before=copy.deepcopy(c.states);state_sha=sha(c.regional/'STATE.json')
    refreshed=reassess(c)[c.record['key']]
    assert c.states==before and sha(c.regional/'STATE.json')==state_sha
    assert c.seen_roles==['TRAIN','VALID'] and len(c.loaded)==3
    for family,source in before[c.record['key']]['selected'].items():
        new=refreshed['selected'][family]
        assert (new['candidate_id'],new['model_sha256'],new['model_path'])==(source['candidate_id'],source['model_sha256'],source['model_path'])
        assert sha(new['model_path'])==source['model_sha256']
        assert new['audit']['TRAIN']['coordinates']==3 and new['audit']['VALID']['coordinates']==2
        assert new['audit']['TRAIN']['WMAPE_percent']==pytest.approx(5.)
        assert not new['audit']['target_met']
        receipt=read_json(new['reassessment_receipt_path'])
        assert receipt['prior_audit']==source['audit'] and not receipt['model_fitted'] and not receipt['selection_changed']
        assert receipt['model_sha256_before']==receipt['model_sha256_after']==source['model_sha256']
        assert receipt['current_population']['admitted_counts']=={'TRAIN':3,'VALID':2,'TEST':1}
    branch=refreshed['selected']['ICNN']['audit']['neural_branch']
    assert branch['train_dataset_identity']=='TRAIN-current'
    assert branch['observed_increment_absolute_sum']==pytest.approx(.3)
    assert branch['nonzero_prediction_coordinates']==3
    assert refreshed['power4']['fit_train_dataset_identity']=='TRAIN-before-last-admission'
    assert not (c.root/'TEST_OPENING_AUTHORIZATION.json').exists()
    # Freeze accepts valid reassessed models even when the TRAIN target is missed.
    registry=module.freeze_registry(c.ctx,c.root,[c.record],{c.record['key']:refreshed},c.support_sha,c.config)
    rows=[r for r in registry['models'] if r['family']!='FFTT']
    assert all(r['status']=='AVAILABLE' and r['target_met'] is False for r in rows)
    assert next(r for r in rows if r['family']=='POWER4')['fit_train_dataset_identity']=='TRAIN-before-last-admission'


def test_exact_receipt_resume_reuses_metrics_not_models(case,monkeypatch):
    c=case;first=reassess(c);paths=[s['reassessment_receipt_path'] for s in first[c.record['key']]['selected'].values()]
    monkeypatch.setattr(module,'load_model',lambda p:pytest.fail('Exact completed receipt must be reused'))
    second=reassess(c)
    assert [s['reassessment_receipt_path'] for s in second[c.record['key']]['selected'].values()]==paths
    assert second[c.record['key']]['power4']['fit_train_dataset_identity']=='TRAIN-before-last-admission'


@pytest.mark.parametrize('failure_stage',['VALID','neural'])
def test_timeout_never_admits_partial_metrics(case,monkeypatch,failure_stage):
    c=case;original=module._evaluate
    if failure_stage=='VALID':
        def evaluate(model,data,*args):
            if data['dataset_identity']=='VALID-current':raise TimeoutError('synthetic VALID timeout')
            return original(model,data,*args)
        monkeypatch.setattr(module,'_evaluate',evaluate)
    else:
        monkeypatch.setattr(module,'_neural_branch_audit',lambda *a:(_ for _ in ()).throw(TimeoutError('synthetic branch timeout')))
    refreshed=reassess(c);icnn=refreshed[c.record['key']]['selected']['ICNN']
    assert icnn['final_reassessment_status']=='UNAVAILABLE_REASSESSMENT_TIMEOUT'
    assert icnn['audit']==c.states[c.record['key']]['selected']['ICNN']['audit']
    receipt=read_json(icnn['reassessment_receipt_path'])
    assert 'audit' not in receipt
    registry=module.freeze_registry(c.ctx,c.root,[c.record],refreshed,c.support_sha,c.config)
    row=next(r for r in registry['models'] if r['family']=='ICNN')
    assert row['status']=='UNAVAILABLE_REASSESSMENT_TIMEOUT' and not row['target_met']
    assert not (c.root/'TEST_OPENING_AUTHORIZATION.json').exists()


def test_hash_change_is_unavailable_and_selection_intent_cannot_change(case):
    c=case;source=c.states[c.record['key']]['selected']['ICNN']
    Path(source['model_path']).write_bytes(b'changed')
    refreshed=reassess(c)
    assert refreshed[c.record['key']]['selected']['ICNN']['final_reassessment_status']=='UNAVAILABLE_REASSESSMENT_FAILED'
    assert all(Path(p).name!='ICNN.npz' for p in c.loaded)
    c.states[c.record['key']]['selected']['ICNN']['candidate_id']='different_selection'
    with pytest.raises(ValueError,match='Frozen prior selection'):
        reassess(c)


def test_protected_freeze_budget_prevents_new_numeric_work(case,monkeypatch):
    c=case
    c.config['finalization_minimum_reserve_seconds']=20000.
    monkeypatch.setattr(module,'load_region_graph',lambda *a,**kw:pytest.fail('No numeric load after reassessment cutoff'))
    refreshed=reassess(c)
    assert refreshed[c.record['key']]['selected']['ICNN']['final_reassessment_status']=='UNAVAILABLE_REASSESSMENT_TIMEOUT'
    assert not c.loaded and not c.seen_roles
    assert read_json(c.root/'SELECTED_REASSESSMENT_SELECTION.json')['stage_deadline_epoch']==c.deadline


def test_fresh_sources_need_no_reassessment_even_if_cutoff_expires(case,monkeypatch):
    c=case;state=c.states[c.record['key']]
    state.update(dataset_identity='TRAIN-current',valid_dataset_identity='VALID-current',
        audited_request_index_sha256=sha(c.labels/'REQUEST_INDEX.json'))
    for source in state['selected'].values():
        source['audit'].update(train_dataset_identity='TRAIN-current',valid_dataset_identity='VALID-current')
    state['power4'].update(train_dataset_identity='TRAIN-current',valid_dataset_identity='VALID-current')
    c.config['finalization_minimum_reserve_seconds']=20000.
    monkeypatch.setattr(module,'load_region_graph',lambda *a,**kw:pytest.fail('Fresh sources must skip reassessment'))
    assert reassess(c)==c.states and not c.loaded and not c.seen_roles


def test_forecast_separately_budgets_inference_branch_load_and_protected_work(case):
    c=case;forecast=module.finalization_forecast([c.record],c.states,c.config)
    for key in ['projected_reassessment_TRAIN_VALID_prediction_seconds','projected_reassessment_ICNN_branch_seconds',
            'projected_reassessment_model_load_and_hash_seconds','projected_reassessment_data_and_graph_load_seconds']:
        assert forecast[key]>0
    assert forecast['reserve_seconds']==pytest.approx(forecast['protected_freeze_and_TEST_seconds']+forecast['selected_reassessment_allowance_seconds'])


def test_existing_registry_never_reassessed(case):
    c=case;write_json(c.root/'REGISTRY_FREEZE.json',{'already':'frozen'})
    with pytest.raises(ValueError,match='unfrozen selection'):
        reassess(c)
    assert not c.loaded and not c.seen_roles


def test_completed_receipt_resume_after_reassessment_cutoff_is_metadata_only(case,monkeypatch):
    c=case;first=reassess(c)
    cutoff=read_json(c.root/'SELECTED_REASSESSMENT_SELECTION.json')['work_deadline_epoch']
    monkeypatch.setattr(module.time,'time',lambda:cutoff+1.)
    monkeypatch.setattr(module,'load_region_graph',lambda *a,**kw:pytest.fail('No array/graph reopening for completed receipt'))
    monkeypatch.setattr(module,'load_model',lambda *a:pytest.fail('No inference after cutoff'))
    assert reassess(c)==first


def test_changed_receipt_population_requires_new_audit_and_missing_valid_is_unavailable(case):
    c=case;reassess(c)
    entries=read_json(c.labels/'REQUEST_INDEX.json')
    entries={k:v for k,v in entries.items() if v['role']!='VALID'}
    write_json(c.labels/'REQUEST_INDEX.json',entries)
    refreshed=reassess(c)
    for source in refreshed[c.record['key']]['selected'].values():
        assert source['final_reassessment_status']=='UNAVAILABLE_REASSESSMENT_FAILED'
        assert read_json(source['reassessment_receipt_path'])['error']=='Current TRAIN and VALID must both be nonempty'
    assert not (c.root/'TEST_OPENING_AUTHORIZATION.json').exists()
