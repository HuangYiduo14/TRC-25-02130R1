"""Only tiny metadata and mocked public stages; no scientific job imports."""
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from partC_rev1_src.pacnhn_recovery24 import execute as e


def new_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream)


class NoSuchProcess(Exception):pass
class AccessDenied(Exception):pass


def resources(rss=10,available=1000,child=None):
    proc=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=rss),
                         children=lambda **kwargs:[] if child is None else [child])
    return SimpleNamespace(Process=lambda:proc,virtual_memory=lambda:SimpleNamespace(available=available),
                           NoSuchProcess=NoSuchProcess,AccessDenied=AccessDenied)


class Context:
    def __init__(self,root):
        self.root=root;self.path=root/'config.json';self.action=None
        self.c=dict(phase_id='PACNHN_RECOVERY24',run_id='toy',
                    resources=dict(tree_rss_bytes=100,minimum_available_bytes=100),
                    expected_outputs=['receipts'],acceptance_tests=['scoped'])
        new_json(self.path,self.c);self.config_sha256=e.binding(self.path)['sha256']
        self.end=time.time()+30000;self.calls=[]
    def require_action(self,action):self.action=action;self.calls.append(('action',action))
    def deadline(self):return self.end
    def guard(self,deadline=None):
        if time.time()>=min(self.end,self.end if deadline is None else deadline):raise TimeoutError('expired')
    def bind(self,path,role,expected=None):
        value=e.binding(path)['sha256']
        if expected is not None and expected!=value:raise ValueError('changed '+role)
        return value
    def flush_bindings(self):return None


@pytest.fixture
def wrapper(tmp_path,monkeypatch):
    ctx=Context(tmp_path);source=tmp_path/'stage.py';source.write_text('before')
    monkeypatch.setattr(e,'source_paths',lambda action:[source])
    return ctx,source


def test_environment_disables_current_and_child_bytecode(monkeypatch,tmp_path):
    monkeypatch.setattr(sys,'dont_write_bytecode',False)
    for key in ('PYTHONDONTWRITEBYTECODE','NUMBA_CACHE_DIR','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        monkeypatch.setenv(key,'old')
    e.configure_environment({'output_root':str(tmp_path)})
    assert sys.dont_write_bytecode and os.environ['PYTHONDONTWRITEBYTECODE']=='1'
    assert os.environ['NUMBA_CACHE_DIR']==str(tmp_path/'cache')
    assert not (tmp_path/'cache').exists()


@pytest.mark.parametrize('action',['primary','other','assignments','metrics'])
def test_ordinary_stage_never_resets_deadline(action):
    assert e.stage_limits(action,30000,1000,{})==dict(action_deadline_epoch=30000.,stage_deadline_epoch=30000.)


def test_source_closure_includes_shared_disk_guard_without_unrelated_stages():
    names=[path.name for path in e.source_paths('primary')]
    assert names==['execute.py','context.py','disk_accounting.py','campaign.py']
    for action in e.ACTIONS:
        assert 'disk_accounting.py' in [path.name for path in e.source_paths(action)]
    assert 'publication.py' not in names and 'THEORETICAL_MODEL.md' not in names


def test_exact_original_reserved_substage_limits():
    assert e.stage_limits('selection',30000,1000,{})['stage_deadline_epoch']==2800
    assert e.stage_limits('selection',30000,20000,{})['stage_deadline_epoch']==15600
    assert e.stage_limits('holdout',30000,1000,{})['stage_deadline_epoch']==26400
    assert e.stage_limits('ablation',30000,1000,{'ablation':{'deadline_epoch':20000}})['stage_deadline_epoch']==20000
    r=e.stage_limits('evaluation',30000,1000,{})
    assert r['regional_evaluation_deadline_epoch']==4600 and r['ablation_evaluation_deadline_epoch']==30000


@pytest.mark.parametrize('value',[float('inf'),float('nan')])
def test_nonfinite_deadlines_rejected(value):
    with pytest.raises(ValueError):e.stage_limits('holdout',value,1,{})
    with pytest.raises(ValueError):e.stage_limits('ablation',30000,1,{'ablation':{'deadline_epoch':value}})


def test_samples_include_children_and_zero_minimum_is_preserved():
    child=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=5))
    sampler=e.ResourceSampler(resources(child=child,available=0),{'tree_rss_bytes':20,'minimum_available_bytes':1})
    sampler.sample();sampler.psutil=resources(available=500);sampler.sample()
    assert sampler.snapshot()['peak_process_tree_rss_bytes']==15
    assert sampler.snapshot()['minimum_available_bytes']==0
    with pytest.raises(MemoryError):sampler.check()


def test_inaccessible_owned_child_is_explicit_not_complete_sampling():
    def denied():raise AccessDenied()
    child=SimpleNamespace(memory_info=denied)
    sampler=e.ResourceSampler(resources(child=child),{'tree_rss_bytes':20,'minimum_available_bytes':1})
    sampler.sample()
    assert sampler.snapshot()['child_access_denied']==1
    with pytest.raises(RuntimeError):sampler.check()


def test_resource_sampler_stops_and_joins():
    s=e.ResourceSampler(resources(),{'tree_rss_bytes':100,'minimum_available_bytes':1},interval=.001)
    s.start();result=s.finish()
    assert result['monitor_joined'] and result['samples']>=2 and not s.thread.is_alive()


def test_wrapper_records_partial_outcome_without_convergence_claim(wrapper,monkeypatch):
    ctx,source=wrapper
    monkeypatch.setattr(e,'dispatch',lambda *args:{'status':'TIME_LIMIT_COMPLETE_ACCOUNTING'})
    result,path=e.execute_stage(ctx,'primary',new_json=new_json,psutil_module=resources())
    assert result['status']=='RETURNED_WITH_EXPLICIT_OUTCOMES'
    assert result['result_status']=='TIME_LIMIT_COMPLETE_ACCOUNTING' and not result['scientific_success_claim']
    assert result['config']['sha256']==ctx.config_sha256
    assert result['resource_samples']['monitor_joined']
    assert result['preparation']['ended_epoch']==result['started_epoch']
    assert result['ended_epoch']<=result['terminal_receipt_started_epoch']
    assert json.loads(path.read_text())==result


@pytest.mark.parametrize('changed',['config','source'])
def test_changed_entry_identity_fails_and_keeps_original_binding(wrapper,monkeypatch,changed):
    ctx,source=wrapper;old_config=ctx.config_sha256;old_source=e.binding(source)['sha256']
    def mutate(*args):
        (ctx.path if changed=='config' else source).write_text('changed')
        return {'status':'RETURNED'}
    monkeypatch.setattr(e,'dispatch',mutate)
    result,path=e.execute_stage(ctx,'primary',new_json=new_json,psutil_module=resources())
    assert result['status']=='FAILED_PRESERVED_OUTPUTS' and result['error']['type']=='ValueError'
    assert result['config']['sha256']==old_config and result['sources'][str(source)]==old_source
    assert path.exists()


def test_unrelated_source_is_not_part_of_active_stage_guard(wrapper,monkeypatch):
    ctx,source=wrapper;unrelated=ctx.root/'publication.py';unrelated.write_text('old')
    def work(*args):unrelated.write_text('new');return {'status':'RETURNED'}
    monkeypatch.setattr(e,'dispatch',work)
    result,_=e.execute_stage(ctx,'primary',new_json=new_json,psutil_module=resources())
    assert not result['error'] and str(unrelated) not in result['sources']
    assert 'publication' not in e.MODULES['primary']


def test_expired_reserved_stage_does_not_dispatch(wrapper,monkeypatch):
    ctx,_=wrapper;ctx.end=time.time()+10
    monkeypatch.setattr(e,'dispatch',lambda *args:pytest.fail('No late selection'))
    result,_=e.execute_stage(ctx,'selection',new_json=new_json,psutil_module=resources())
    assert result['error']['type']=='TimeoutError'


def test_sampled_memory_breach_does_not_dispatch(wrapper,monkeypatch):
    ctx,_=wrapper;monkeypatch.setattr(e,'dispatch',lambda *args:pytest.fail('No work after resource breach'))
    result,_=e.execute_stage(ctx,'primary',new_json=new_json,psutil_module=resources(rss=101))
    assert result['error']['type']=='MemoryError'
    assert result['resource_samples']['sampled_rss_limit_exceeded']


def test_action_drift_is_detected(wrapper,monkeypatch):
    ctx,_=wrapper
    def wrong(*args):ctx.require_action('training');return {'status':'RETURNED'}
    monkeypatch.setattr(e,'dispatch',wrong)
    result,_=e.execute_stage(ctx,'evaluation',new_json=new_json,psutil_module=resources())
    assert result['error']['type']=='ValueError' and result['action_at_return']=='training'


def test_flush_failure_is_not_a_success_receipt(wrapper,monkeypatch):
    ctx,_=wrapper;monkeypatch.setattr(e,'dispatch',lambda *args:{'status':'RETURNED'})
    def fail():raise OSError('metadata write failed')
    ctx.flush_bindings=fail
    result,path=e.execute_stage(ctx,'primary',new_json=new_json,psutil_module=resources())
    assert result['error']['type']=='OSError' and path.exists()


@pytest.mark.parametrize('action',[a for a in e.ACTIONS if a!='assignments'])
def test_active_national_timing_rejects_other_actions_before_import(tmp_path,monkeypatch,action):
    ctx=Context(tmp_path);new_json(tmp_path/'NATIONAL_TIMING_ACTIVE.json',{'pid':999})
    monkeypatch.setattr(e,'_module',lambda *args:pytest.fail('No imports after active lock'))
    with pytest.raises(PermissionError):e.dispatch(ctx,action,{'stage_deadline_epoch':ctx.end},tmp_path,new_json)


@pytest.mark.parametrize('fail',[False,True])
def test_owned_assignment_lock_archives_on_return_or_failure(tmp_path,monkeypatch,fail):
    ctx=Context(tmp_path);folder=tmp_path/'receipt';folder.mkdir()
    def run(*args,**kwargs):
        assert (tmp_path/'NATIONAL_TIMING_ACTIVE.json').exists()
        if fail:raise OSError('query failure')
        return {'status':'COMPLETE'}
    monkeypatch.setattr(e,'_module',lambda name:SimpleNamespace(run=run))
    if fail:
        with pytest.raises(OSError):e.dispatch(ctx,'assignments',{'stage_deadline_epoch':ctx.end},folder,new_json)
    else:assert e.dispatch(ctx,'assignments',{'stage_deadline_epoch':ctx.end},folder,new_json)['status']=='COMPLETE'
    assert not (tmp_path/'NATIONAL_TIMING_ACTIVE.json').exists()
    assert (folder/('NATIONAL_TIMING_FAILED.json' if fail else 'NATIONAL_TIMING_COMPLETED.json')).exists()


def test_foreign_timing_lock_never_moved(tmp_path,monkeypatch):
    ctx=Context(tmp_path);active=tmp_path/'NATIONAL_TIMING_ACTIVE.json';new_json(active,{'pid':999})
    with pytest.raises(FileExistsError):e.dispatch(ctx,'assignments',{'stage_deadline_epoch':ctx.end},tmp_path,new_json)
    assert json.loads(active.read_text())=={'pid':999}


def test_evaluation_forwards_two_deadlines_and_rechecks_holdout(tmp_path,monkeypatch):
    ctx=Context(tmp_path);ctx.require_action('holdout')
    (tmp_path/'training').mkdir();new_json(tmp_path/'training/MODEL_SELECTION_FREEZE.json',{'status':'FROZEN'})
    calls=[]
    def run(context,**kwargs):calls.append(kwargs);return {'status':'PARTIAL'}
    def evaluate(context,freeze):
        assert context.action=='holdout' and freeze['sha256']==e.binding(freeze['path'])['sha256']
        return {'status':'ACCOUNTED'}
    monkeypatch.setattr(e,'_module',lambda name:SimpleNamespace(run=run) if name=='regional_evaluation' else SimpleNamespace(evaluate=evaluate))
    limits=e.stage_limits('evaluation',ctx.end,time.time(),{})
    result=e.dispatch(ctx,'evaluation',limits,tmp_path,new_json)
    assert calls[0]['deadline']==limits['regional_evaluation_deadline_epoch']
    assert calls[0]['national_timing_active'] is False and calls[0]['max_workers']==4
    assert result['ablation_evaluation']['status']=='ACCOUNTED'
