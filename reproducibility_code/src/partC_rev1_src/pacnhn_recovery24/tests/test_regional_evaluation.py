"""Synthetic metadata/arrays only. No old or new campaign data are opened."""
import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from partC_rev1_src.pacnhn_recovery24 import regional_evaluation as e

D='d'*64;T='a'*64;V='b'*64;TEST='c'*64


def dataset(role='TRAIN',truth=2.):
    return dict(u=np.array([[0.],[1.]]),tau=np.full((2,1),truth),c0=np.array([2.]),
                role=role,dataset_identity={'TRAIN':T,'VALID':V,'TEST':TEST}[role])


def audit():
    metric=dict(WMAPE_percent=0.,absolute_error_sum=0.,true_cost_sum=4.,coordinates=2,
                unique_inputs=2,batch_inference_seconds=.25)
    return dict(model_sha256=D,train_dataset_identity=T,valid_dataset_identity=V,
                TRAIN=copy.deepcopy(metric),VALID=copy.deepcopy(metric))


def model_record(family='ICNN',index=0):
    return dict(suite_id='s03',region_index=index,family=family,status='AVAILABLE',
        model_path='model.npz',model_sha256=D,graph_sha256=D,pair_keys_identity=D,
        current_train_dataset_identity=T,current_valid_dataset_identity=V,
        audit_train_dataset_identity=T,audit_valid_dataset_identity=V,audit_model_sha256=D,
        property_status='PASS',audit=audit())


def test_native_full_coordinate_reduction_and_batching():
    calls=[];d=dataset(truth=4.)
    r=e.native_metrics(e._FFTT(d['c0']),d,check=lambda:calls.append(1),batch=1)
    assert r['absolute_error_sum']==4 and r['true_cost_sum']==8
    assert r['WMAPE_percent']==50 and r['coordinates']==2 and len(calls)==3


def test_zero_truth_explicitly_undefined_not_zero_accuracy():
    d=dataset(truth=0.)
    r=e.native_metrics(e._FFTT(d['c0']),d,check=lambda:None)
    assert r['WMAPE_percent'] is None and r['status']=='UNDEFINED_ZERO_TRUTH'
    assert r['absolute_error_sum']==4 and r['true_cost_sum']==0


def test_empty_admissions_preserved():
    d=dataset();d['u']=d['tau']=np.empty((0,1))
    r=e.native_metrics(e._FFTT(d['c0']),d,check=lambda:None)
    assert r['status']=='NO_ADMITTED_LABELS' and r['coordinates']==0


@pytest.mark.parametrize('pred',[np.array([2.,2.]),np.array([[-1.],[-1.]]),np.array([[np.nan],[2.]])])
def test_invalid_predictions_never_clipped(pred):
    model=SimpleNamespace(c0=np.array([2.]),cost=lambda ignored:pred)
    with pytest.raises(ArithmeticError):e.native_metrics(model,dataset(),check=lambda:None)


def test_exact_c0_and_deadline_required():
    with pytest.raises(ValueError):e.native_metrics(e._FFTT([2.00000000001]),dataset(),check=lambda:None)
    with pytest.raises(TimeoutError):e.native_metrics(e._FFTT([2.]),dataset(),check=lambda:(_ for _ in ()).throw(TimeoutError()))


def test_native_sum_audit_reuse_reports_zero_new_inference_time():
    entry=e.descriptor(model_record(),family='ICNN',source='frozen')
    result=e.reusable_metric(entry,'TRAIN',T)
    assert result['WMAPE_percent']==0 and result['inference_seconds']==0
    assert result['historical_inference_seconds']==.25
    assert result['metric_evidence'].startswith('REUSED')
    assert e.reusable_metric(entry,'TEST',TEST) is None


@pytest.mark.parametrize('mutation',['model','train','missing_sum','bad_ratio','nonfinite','count'])
def test_audit_reuse_never_accepts_stale_or_inconsistent_sums(mutation):
    entry=e.descriptor(model_record(),family='ICNN',source='frozen');a=entry['saved_audit']
    if mutation=='model':a['model_sha256']='e'*64
    if mutation=='train':a['train_dataset_identity']='e'*64
    if mutation=='missing_sum':a['TRAIN'].pop('absolute_error_sum')
    if mutation=='bad_ratio':a['TRAIN']['WMAPE_percent']=2.
    if mutation=='nonfinite':a['TRAIN']['true_cost_sum']=float('inf')
    if mutation=='count':a['TRAIN']['coordinates']=False
    assert e.reusable_metric(entry,'TRAIN',T) is None


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'new';old=tmp_path/'old';root.mkdir();old.mkdir()
    ctx=SimpleNamespace(root=root,source_root=old,path=root/'config.json',config_sha256=D,
        c=dict(phase_id='P',run_id='R',resources={}),require_action=lambda action:None,
        guard=lambda **kwargs:None,flush_bindings=lambda:None)
    ctx.bind=lambda path,role,expected=None: e.sha(path) if Path(path).exists() else expected or D
    records=[dict(suite_id='s03',region_index=i,pairs=1,sha256=D) for i in range(2)]
    normalized=[model_record(f,i) for i in range(1) for f in e.FAMILIES]
    models=e.index_models(normalized,records)
    portfolios={(r['suite_id'],r['region_index'],f):dict(region_index=r['region_index'],component='FFTT',
        graph_sha256=D,reason='FROZEN_FALLBACK') for r in records for f in e.FAMILIES}
    input_rows={'s03:000000':dict(region_key='s03:000000',requested_inputs=48,record=records[0])}
    monkeypatch.setattr(e,'_baseline',lambda *args:({f:e.descriptor(model_record(f),family=f,source='original') for f in e.FAMILIES},None))
    return ctx,records,models,portfolios,input_rows


def evaluate_args(values):
    ctx,records,models,portfolios,input_rows=values
    return dict(models=models,portfolios=portfolios,input_rows=input_rows,freeze_path='freeze',freeze_sha256=D,deadline=9999999999.)


def test_exact_audits_avoid_training_payload_load_and_test_is_fresh(tmp_path,monkeypatch):
    values=fixture(tmp_path,monkeypatch);ctx,records,*_=values;calls=[]
    def cohorts(*args,need_training,**kwargs):
        assert not need_training
        return {'TEST':dataset('TEST')},{'TEST':'AVAILABLE'},[]
    monkeypatch.setattr(e,'_cohorts',cohorts)
    monkeypatch.setattr(e,'_load_frozen_model',lambda *args: (calls.append(1) or e._FFTT([2.])))
    rows,_=e.evaluate_region(ctx,records[0],**evaluate_args(values))
    assert len(rows)==36
    assert sum(r.get('metric_evidence')=='REUSED_HASH_AND_CURRENT_DATASET_BOUND_NATIVE_AUDIT' for r in rows)==24
    assert sum(r.get('metric_evidence')=='FRESH_POSTFREEZE_EVALUATION' for r in rows)==2
    assert sum(r.get('metric_evidence')=='REUSED_IDENTICAL_MODEL_AND_COHORT_WITHIN_EVALUATION' for r in rows)==10
    assert all(r['inference_seconds']==0 for r in rows if r.get('metric_evidence','').startswith('REUSED'))
    assert len(calls)==1 # all synthetic serialized variants share the same actual model hash
    assert all(r['target_met'] is None for r in rows if r['role']!='TRAIN')
    assert all(r['train_target_met'] for r in rows)


def test_changed_baseline_train_requires_same_current_cohort(tmp_path,monkeypatch):
    values=fixture(tmp_path,monkeypatch);ctx,records,*_=values
    old=e.descriptor(model_record(),family='ICNN',source='old');old['saved_audit']['train_dataset_identity']='e'*64
    monkeypatch.setattr(e,'_baseline',lambda *args:({'ICNN':old},None))
    def cohorts(*args,need_training,**kwargs):
        assert need_training
        return {r:dataset(r) for r in e.ROLES},{r:'AVAILABLE' for r in e.ROLES},[]
    monkeypatch.setattr(e,'_cohorts',cohorts);monkeypatch.setattr(e,'_load_frozen_model',lambda *args:e._FFTT([2.]))
    rows,_=e.evaluate_region(ctx,records[0],**evaluate_args(values))
    r=next(r for r in rows if (r['version'],r['family'],r['role'])==('baseline','ICNN','TRAIN'))
    assert r['metric_evidence']=='FRESH_POSTFREEZE_EVALUATION' and r['dataset_identity']==T


def test_loaded_current_identity_cannot_drift_after_selection(tmp_path,monkeypatch):
    values=fixture(tmp_path,monkeypatch);ctx,records,*_=values
    d=dataset();d['dataset_identity']='e'*64
    monkeypatch.setattr(e,'_cohorts',lambda *args,**kwargs:({'TRAIN':d},{},[]))
    with pytest.raises(ValueError,match='differs'):e.evaluate_region(ctx,records[0],**evaluate_args(values))


def test_missing_model_is_a_row_and_portfolio_does_not_switch(tmp_path,monkeypatch):
    values=list(fixture(tmp_path,monkeypatch));ctx,records,models,portfolios,_=values
    del models[('s03',0,'ICNN')]
    monkeypatch.setattr(e,'_cohorts',lambda *args,**kwargs:({'TEST':dataset('TEST')},{},[]))
    monkeypatch.setattr(e,'_load_frozen_model',lambda *args:e._FFTT([2.]))
    rows,_=e.evaluate_region(ctx,records[0],**evaluate_args(values))
    missing=next(r for r in rows if (r['version'],r['family'],r['role'])==('recovered','ICNN','TEST'))
    assert missing['status']=='UNAVAILABLE_MODEL' and 'WMAPE_percent' not in missing
    portfolio=next(r for r in rows if (r['version'],r['family'],r['role'])==('portfolio','ICNN','TEST'))
    assert portfolio['fallback_component']=='FFTT' and portfolio['status']=='MEASURED'


def run_fixture(tmp_path,monkeypatch):
    values=fixture(tmp_path,monkeypatch);ctx,records,models,portfolios,inputs=values
    freeze=ctx.root/'training/MODEL_SELECTION_FREEZE.json';freeze.parent.mkdir();freeze.write_text('{}')
    opened=ctx.root/'holdout/TEST_OPENED.json';opened.parent.mkdir();opened.write_text(json.dumps(dict(epoch=2.,selection_freeze_sha256=D)))
    normalized=ctx.root/'training/NORMALIZED_RECORDS.json';normalized.write_text(json.dumps(list(models.values())))
    portfolio=ctx.root/'portfolios.json';portfolio.write_text('{}')
    frozen=dict(created_epoch=1.,model_records={'path':str(normalized)},portfolio={'path':str(portfolio)})
    monkeypatch.setattr(e,'require_selection_freeze',lambda *args:(frozen,dict(rows=list(inputs.values()))))
    monkeypatch.setattr(e,'support_records',lambda ignored:records)
    monkeypatch.setattr(e,'index_portfolios',lambda *args:portfolios)
    from partC_rev1_src.pacnhn_recovery24 import labels
    def parallel(ctx,worker,jobs,deadline,max_workers):
        assert max_workers==4
        return [dict(region_key=job[1]['region_key'],rows=list(e.unavailable_rows(job[1]['record'],'UNCOMPLETED_TIME_LIMIT',inputs)),
                     evidence={},elapsed_seconds=1.) for job in jobs]
    monkeypatch.setattr(labels,'_parallel',parallel)
    return ctx,freeze


def test_full_streamed_accounting_and_missing_regions(tmp_path,monkeypatch):
    ctx,freeze=run_fixture(tmp_path,monkeypatch)
    result=e.run(ctx,freeze_path=freeze,freeze_sha256=D,deadline=9999999999.,national_timing_active=False)
    assert result['regions']==2 and result['row_count']==72 and result['status'].startswith('TIME_LIMIT')
    out=ctx.root/'evaluation/regional_v1'
    with (out/'learning.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==72 and sum(r['status']=='UNREQUESTED_TEST' for r in rows)==12
    assert all(not r['WMAPE_percent'] for r in rows)
    assert result['model_fitting'] is False and result['model_selection_changed'] is False
    with pytest.raises(FileExistsError):e.run(ctx,freeze_path=freeze,freeze_sha256=D,deadline=9999999999.,national_timing_active=False)


def test_preopening_denied_before_output(tmp_path,monkeypatch):
    ctx,freeze=run_fixture(tmp_path,monkeypatch)
    (ctx.root/'holdout/TEST_OPENED.json').unlink()
    with pytest.raises(FileNotFoundError):e.run(ctx,freeze_path=freeze,freeze_sha256=D,deadline=9999999999.,national_timing_active=False)
    assert not (ctx.root/'evaluation').exists()


def test_missing_requested_worker_result_is_not_unrequested(tmp_path,monkeypatch):
    ctx,freeze=run_fixture(tmp_path,monkeypatch)
    from partC_rev1_src.pacnhn_recovery24 import labels
    monkeypatch.setattr(labels,'_parallel',lambda *args,**kwargs:[])
    e.run(ctx,freeze_path=freeze,freeze_sha256=D,deadline=9999999999.,national_timing_active=False)
    with (ctx.root/'evaluation/regional_v1/learning.csv').open() as stream:rows=list(csv.DictReader(stream))
    requested=[r for r in rows if r['region_index']=='0' and r['role']=='TEST']
    assert len(requested)==12 and all(r['status']=='UNCOMPLETED_MISSING_WORKER_RESULT' for r in requested)
    assert all(r['requested_inputs']=='48' for r in requested)


def test_accounting_cannot_publish_completion_after_explicit_deadline(tmp_path,monkeypatch):
    ctx,freeze=run_fixture(tmp_path,monkeypatch);calls=[]
    def guard(**kwargs):
        calls.append(kwargs['deadline'])
        if len(calls)>1:raise TimeoutError('Accounting deadline reached')
    ctx.guard=guard
    with pytest.raises(TimeoutError):
        e.run(ctx,freeze_path=freeze,freeze_sha256=D,deadline=9999999999.,national_timing_active=False)
    assert len(calls)==2 and not (ctx.root/'evaluation/regional_v1/RESULT.json').exists()
    assert (ctx.root/'evaluation/regional_v1/INTENT.json').exists()


@pytest.mark.parametrize('active',[True,None,0,'false'])
def test_no_national_concurrency_requires_explicit_false(tmp_path,monkeypatch,active):
    ctx,freeze=run_fixture(tmp_path,monkeypatch)
    with pytest.raises(PermissionError):e.run(ctx,freeze_path=freeze,freeze_sha256=D,deadline=9999999999.,national_timing_active=active)
    assert not (ctx.root/'evaluation').exists()


def test_frozen_model_record_identity_rejection():
    support=[dict(suite_id='s03',region_index=0,sha256=D)]
    row=model_record();row['audit_model_sha256']='a'*64
    with pytest.raises(ValueError,match='hash'):e.index_models([row],support)


def test_portfolio_requires_exact_whole_region_family_coverage():
    support=[dict(suite_id='s03',region_index=0,sha256=D)]
    with pytest.raises(ValueError,match='exactly cover'):e.index_portfolios({'portfolios':[]},support)


def test_zero_native_gamma_disclosed_without_predictions():
    entry=e.descriptor(model_record(),family='ICNN',source='frozen')
    model=SimpleNamespace(arrays={'amplitude':np.array(0.)},cost=lambda _:pytest.fail('No predictions'))
    result=e.neural_metadata(entry,model)
    assert result['neural_amplitude']==0 and result['neural_amplitude_is_zero'] is True
    assert 'STRUCTURAL_ONLY' in result['neural_activity_scope']


def test_saved_gamma_requires_same_model_hash_and_does_not_claim_output_activity():
    entry=e.descriptor(model_record(),family='ICNN',source='frozen')
    entry['saved_audit']['neural_branch']={'amplitude':1e-100,'structurally_active':True}
    result=e.neural_metadata(entry)
    assert result['neural_amplitude']==1e-100 and result['neural_amplitude_is_zero'] is False
    entry['saved_audit']['model_sha256']='e'*64
    assert e.neural_metadata(entry)['neural_amplitude'] is None


@pytest.mark.parametrize('amplitude',[float('nan'),float('inf'),-1.])
def test_invalid_gamma_is_not_misreported_as_active(amplitude):
    entry=e.descriptor(model_record(),family='ICNN',source='frozen')
    with pytest.raises(ValueError):e.neural_metadata(entry,SimpleNamespace(arrays={'amplitude':np.array(amplitude)}))


def test_worker_binding_ledger_is_per_region_not_quadratic(monkeypatch):
    from partC_rev1_src.pacnhn_recovery24 import context
    saved=[]
    class Ctx:
        def __init__(self,*args,**kwargs):self.bindings={'config':'fixed'}
        def require_action(self,action):assert action=='holdout'
        def flush_bindings(self):saved.append(dict(self.bindings))
    monkeypatch.setattr(context,'RecoveryContext',Ctx)
    monkeypatch.setattr(e,'_CONTEXTS',{})
    def frozen(ctx,*args):ctx.bindings['freeze']='fixed'
    monkeypatch.setattr(e,'require_selection_freeze',frozen)
    def evaluate(ctx,record,**kwargs):
        ctx.bindings['region_'+str(record['region_index'])]='bound'
        return [],{}
    monkeypatch.setattr(e,'evaluate_region',evaluate)
    for i in range(3):
        e._worker('synthetic',dict(record={'suite_id':'s03','region_index':i},models={},portfolios={},
            input_rows={},freeze_path='freeze',freeze_sha256=D),9999999999.)
    assert all(len(r)==3 for r in saved)
    assert [set(r)-{'config','freeze'} for r in saved]==[{'region_0'},{'region_1'},{'region_2'}]


def test_cohort_reader_forwards_postfreeze_authority_and_restores_holdout(tmp_path,monkeypatch):
    values=fixture(tmp_path,monkeypatch);ctx,records,*_=values;actions=[]
    ctx.require_action=lambda action:actions.append(action)
    def loader(c,record,**kwargs):
        assert kwargs['read_only'] is True and kwargs['selection_freeze_path']=='freeze'
        assert kwargs['selection_freeze_sha256']==D
        return dataset('TRAIN'),dataset('VALID'),None
    monkeypatch.setattr(e,'load_training_cohorts',loader)
    cohorts,_,_=e._cohorts(ctx,records[0],{},'freeze',D,lambda:None,9999999999.)
    assert set(cohorts)=={'TRAIN','VALID'} and actions==['holdout']
