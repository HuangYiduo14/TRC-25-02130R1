"""Mocked budget/failure provenance checks; no optimizer or oracle is executed."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from partC_rev1_src.phase_n8.common import digest,read_json,save_npz,sha,write_json
from partC_rev1_src.phase_n8_r4 import learning,learning_data
from partC_rev1_src.phase_n8_r4 import learning_artifacts as artifacts


def fake_context(root):
    bound=[]
    def bind(path,role,expected=None):
        actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('hash mismatch')
        bound.append(Path(path).name);return actual
    return SimpleNamespace(root=root,started=1000.,deadline=1600.,c={'compute_cutoff_epoch':20000.},
        guard=lambda **kwargs:None,bind=bind,bound=bound)


@pytest.mark.parametrize('interrupt',[False,True])
def test_d00_partial_handoff_survives_cutoff_or_interruption(tmp_path,monkeypatch,interrupt):
    ctx=fake_context(tmp_path);clock=[ctx.started];calls=[];closed=[]
    monkeypatch.setattr(learning_data.time,'time',lambda:clock[0])
    def guard(deadline=None):
        if clock[0]>=min(ctx.deadline,deadline or ctx.deadline):raise TimeoutError('mock deadline')
    ctx.guard=guard
    monkeypatch.setattr(learning_data,'RegionalSources',lambda _:SimpleNamespace(close=lambda:closed.append(True)))
    strata=['large','small','good_VALID','difficult_TRAIN_VALID']
    selected=pd.DataFrame([dict(region_id=f'r{i}',stratum=strata[i//4],rank_within_stratum=i%4+1,
        m_active=2,primary_E2_additional_updates=0) for i in range(16)])
    monkeypatch.setattr(learning_data,'select_regions',lambda *_:(selected,selected.copy(),selected[['region_id']]))
    def prepare_region(*args,**kwargs):
        row=args[2];calls.append(row['region_id'])
        if len(calls)==2:raise KeyboardInterrupt('mock hard interruption')
        clock[0]=1001. if interrupt else 1541.
        return dict(row,status='PREPARED')
    monkeypatch.setattr(learning_data,'prepare_region',prepare_region)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):learning_data.prepare(ctx)
    else:
        learning_data.prepare(ctx)
        assert calls==['r0']
    saved=read_json(tmp_path/'learning/PREPARE.json')
    assert saved['status']=='PARTIAL_PREPARATION' and not saved['preparation_complete']
    assert [r['region_id'] for r in saved['prepared_regions']]==['r0']
    missing=saved['preparation_failures']
    assert {r['region_id'] for r in missing}=={f'r{i}' for i in range(1,16)}
    expected='NOT_PREPARED_AT_CHECKPOINT' if interrupt else 'NOT_RUN_PREPARATION_BUDGET'
    assert all(r['status']==expected for r in missing)
    assert closed==[True]
    record=read_json(tmp_path/'learning/LEARNING_CLOCK.json')
    assert record['stage_start_epoch']==1000. and record['stage_deadline_epoch']==15400.


def tiny_prepared_case(tmp_path,monkeypatch):
    ctx=fake_context(tmp_path);path=tmp_path/'learning/regions/r1';path.mkdir(parents=True)
    arrays=dict(u=np.arange(40,dtype=float).reshape(20,2),t=np.full((20,2),10.),c0=np.full(2,10.),scale=np.ones(2))
    save_npz(path/'train.npz',arrays)
    pd.DataFrame(dict(input_hash=[f'h{i}' for i in range(20)],family=['RADIAL']*20)).to_csv(path/'train_rows.csv',index=False)
    subset=dict(fraction=.1,unique_n=2,subset_input_hash=digest(['h0','h1']))
    region=dict(region_id='r1',stratum='large',subsets=[subset],train_file_sha256=sha(path/'train.npz'),
        train_rows_sha256=sha(path/'train_rows.csv'),calibration_input_hash=digest(['h0','h1']))
    monkeypatch.setattr(artifacts,'weighted_power4_fit',lambda u,t,scale,c0:np.zeros_like(c0))
    return ctx,arrays,region,subset,path/'fits/seed_0_fraction_010'


def test_nn_initialization_failure_preserves_analytic_readiness(tmp_path,monkeypatch):
    ctx,arrays,region,subset,dest=tiny_prepared_case(tmp_path,monkeypatch)
    def fail_init(*args,**kwargs):raise ValueError('mock NN initialization failure')
    def forbidden_update(*args,**kwargs):raise AssertionError('No optimizer update may run')
    monkeypatch.setattr(learning,'make_model',fail_init);monkeypatch.setattr(learning,'update',forbidden_update)
    with pytest.raises(ValueError,match='mock NN initialization'):learning.fit(ctx,region,0,.1,{})
    fields=artifacts.mark_neural_failure(dest,'ValueError');families=fields['model_readiness']
    for family in ('FFTT','POWER4'):
        assert families[family]['evaluation_ready']
        assert families[family]['arrays_sha256']==sha(dest/families[family]['arrays_file'])
    assert not families['SKIP_ICNN32']['evaluation_ready']
    assert fields['unique_n']==2 and fields['subset_input_hash']==subset['subset_input_hash']


def frozen_case(tmp_path,monkeypatch,*,failed):
    ctx,arrays,region,subset,dest=tiny_prepared_case(tmp_path,monkeypatch);dest.mkdir(parents=True)
    artifacts.prepare_analytic_artifacts(dest,arrays,region,0,.1,subset)
    checkpoint=dest/'SELECTED_SKIP_ICNN32.npz';save_npz(checkpoint,dict(c0=arrays['c0']))
    write_json(dest/'CHECKPOINT.json',dict(selected_update=0,selected_VALID_score=.25,selected_arrays_sha256=sha(checkpoint)))
    artifacts.finalize_neural_readiness(dest,0,.25,'COMPLETE')
    fit=dict(region_id='r1',stratum='large',seed=0,fraction=.1,status='FAILED' if failed else 'COMPLETE')
    if failed:fit.update(artifacts.mark_neural_failure(dest,'RuntimeError'))
    else:
        fit.update(optimizer_updates=0,selected_update=0)
        fit.update(artifacts.readiness_fields(dest,fit_status='COMPLETE'))
    return ctx,arrays,dest,fit


def test_ready_models_keeps_analytics_and_excludes_failed_nn(tmp_path,monkeypatch):
    ctx,arrays,dest,fit=frozen_case(tmp_path,monkeypatch,failed=True)
    models=list(artifacts.ready_models(ctx,fit,arrays['c0']))
    assert [item[0] for item in models]==['FFTT','POWER4']
    assert 'SELECTED_SKIP_ICNN32.npz' not in ctx.bound and 'CHECKPOINT.json' not in ctx.bound
    for _,_,_,meta in models:
        assert meta['fit_status']=='ANALYTIC_COMPLETE' and meta['parent_NN_fit_status']=='FAILED'
        assert meta['optimizer_updates']==meta['selected_update']==0
    neural=read_json(dest/'MODEL_READINESS.json')['families']['SKIP_ICNN32']
    assert neural['parent_NN_fit_status']=='FAILED' and neural['diagnostic_arrays_sha256'] is not None


@pytest.mark.parametrize('filename',['MODEL_READINESS.json','FFTT.npz','POWER4_RELATIVE.npz','SELECTED_SKIP_ICNN32.npz','CHECKPOINT.json'])
def test_ready_models_rejects_mutated_frozen_artifact(tmp_path,monkeypatch,filename):
    ctx,arrays,dest,fit=frozen_case(tmp_path,monkeypatch,failed=False)
    path=dest/filename;path.write_bytes(path.read_bytes()+b'\nchanged')
    with pytest.raises(ValueError,match='hash mismatch'):list(artifacts.ready_models(ctx,fit,arrays['c0']))


def test_promotion_rejects_arrays_other_than_validated_checkpoint(tmp_path,monkeypatch):
    ctx,arrays,dest,fit=frozen_case(tmp_path,monkeypatch,failed=False)
    path=dest/'SELECTED_SKIP_ICNN32.npz';save_npz(path,dict(c0=arrays['c0']+1.))
    with pytest.raises(ValueError,match='differ from validated checkpoint'):
        artifacts.finalize_neural_readiness(dest,0,.25,'COMPLETE')
