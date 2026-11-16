"""Mocked parent process/deadline regressions; no workers, fits, or TEST reads."""
import copy
from concurrent.futures import Future
import importlib.util
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from partC_rev1_src.pacnhn_v1.control import read_json,sha,write_json
from partC_rev1_src.pacnhn_v1 import gates

STAGED=Path(__file__).with_name('training.py')
if STAGED.exists():
    spec=importlib.util.spec_from_file_location('partC_rev1_src.pacnhn_v1._training_resource_staged',STAGED)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
else:
    from partC_rev1_src.pacnhn_v1 import training as module

REPO=Path(r'C:/Users/yiduo2/PycharmProjects/doe_solver_v3_phase1')


@pytest.fixture
def setup_case(tmp_path,monkeypatch):
    def build(*,stage_remaining=5000.,retained=True):
        config=module.read_training_config(REPO/'partC_doc_icnn/pacnhn_v1/training_config_v2.json')
        config_path=tmp_path/'config.json';write_json(config_path,config)
        root=tmp_path/'training';now=time.time();deadline=now+stage_remaining
        clock=dict(original_campaign_start_epoch=deadline-86500,stage_start_epoch=deadline-86400,
            stage_seconds=86400,stage_deadline_epoch=deadline,output_bytes_before_training=0)
        write_json(root/'TRAINING_CLOCK.json',clock)
        for name in ['SUPPORT_FREEZE.json','COMPLETE_WITNESS_AUDIT.json']:
            write_json(tmp_path/name,{'synthetic':True})
        resources=dict(added_disk_bytes=10**12,minimum_available_bytes=3*1024**3,tree_rss_bytes=24*1024**3)
        ctx=SimpleNamespace(root=tmp_path,path=config_path,started=now,
            c=dict(phase_id='PACNHN_V1',run_id='synthetic_resource_recovery',suite_ids=['s03'],
                original_start_epoch=clock['original_campaign_start_epoch'],hard_deadline_epoch=now+10000,
                maximum_cumulative_stage_deadlines=[deadline-100,deadline+1000],resources=resources,acceptance_tests=['mock']),
            peak_rss=4*1024**3,minimum_available=2*1024**3,
            guard=lambda **kw:None,bind=lambda *a,**kw:None,flush_bindings=lambda:None)
        regions=[dict(suite_id='s03',region_index=i,key=f's03:{i:06d}',pairs=1) for i in range(2)]
        prior_intent=root/'intents/original.json'
        write_json(prior_intent,dict(phase_id=ctx.c['phase_id'],run_id=ctx.c['run_id'],clock=clock,
            config_sha256=sha(config_path),support_freeze_sha256=sha(tmp_path/'SUPPORT_FREEZE.json')))
        model=root/'regions/s03/000000/checkpoint.npz';model.parent.mkdir(parents=True);model.write_bytes(b'prior selected checkpoint')
        state=dict(visits=1,dataset_identity='TRAIN-old',valid_dataset_identity='VALID-old',
            selected={'ICNN':dict(candidate_id='selected_1',model_path=str(model),model_sha256=sha(model),
                audit={'TRAIN':{'WMAPE_percent':.5},'VALID':{'WMAPE_percent':.5}})},
            label_counts={'TRAIN':64,'VALID':24,'TEST':48},fixed_role_progress={'complete':True})
        controller=dict(states={regions[0]['key']:state},scheduler={},slices=1,
            status='RESOURCE_PAUSED',config_sha256=sha(config_path),source_intent=str(prior_intent))
        if retained:
            write_json(root/'CONTROLLER_STATE.json',controller)
            write_json(model.parent/'STATE.json',state)
        monkeypatch.setattr(gates,'require_component',lambda *a,**kw:None)
        monkeypatch.setattr(module,'gate_inputs',lambda *a:({},regions))
        monkeypatch.setattr(module,'freeze_xgb_memory_budget',lambda *a:{})
        return SimpleNamespace(ctx=ctx,root=root,config=config,config_path=config_path,clock=clock,
            regions=regions,model=model,controller=controller,prior_intent=prior_intent)
    return build


def forbid_training(monkeypatch):
    def fail(*a,**kw):pytest.fail('Finalization-only recovery attempted training/labels/device or worker work')
    for name in ['choose_device','concurrency_gate','ProcessPoolExecutor','generate_labels','fit_power4_cost',
            'fit_xgboost_cost','update_icnn','freeze_xgb_memory_budget']:
        monkeypatch.setattr(module,name,fail)


def test_parent_ram_pause_cleans_owned_pending_worker_and_preserves_current_state(setup_case,monkeypatch):
    c=setup_case();clock_sha=sha(c.root/'TRAINING_CLOCK.json');model_sha=sha(c.model)
    submitted=[];stopped=[];guard_state={'raise':False}
    def guard(**kw):
        if guard_state['raise']:raise MemoryError('Available system RAM guard')
    c.ctx.guard=guard
    monkeypatch.setattr(module,'choose_device',lambda *a:'cpu')
    monkeypatch.setattr(module,'concurrency_gate',lambda *a:1)
    class Pool:
        def __init__(self,**kw):self._processes={7:SimpleNamespace(pid=99999)}
        def submit(self,fn,payload):
            submitted.append(payload['record']['key'])
            region_root=Path(payload['region_root']);state=copy.deepcopy(c.controller['states']['s03:000000'])
            state['visits']=2;write_json(region_root/'STATE.json',state)
            write_json(region_root/'labels/REQUEST_INDEX.json',{'new_admitted_label':{'metadata_only':True}})
            return Future()
    monkeypatch.setattr(module,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(module,'wait',lambda *a,**kw:(guard_state.update(raise_=True) or guard_state.__setitem__('raise',True) or (set(),set())))
    monkeypatch.setattr(module,'_stop_own_pool',lambda pool:stopped.append(pool))
    for name in ['reassess_selected_before_freeze','freeze_registry','final_test_audit','complete_frozen_training']:
        monkeypatch.setattr(module,name,lambda *a,**kw:pytest.fail('Resource pause entered finalization'))
    result=module.run(c.ctx,c.config_path)
    assert result['status']=='RESOURCE_PAUSED' and len(submitted)==len(stopped)==1
    assert result['reason']=='Available system RAM guard' and result['pid']==os.getpid()
    assert result['owned_worker_cleanup']['completed'] and result['owned_worker_cleanup']['owned_pids']==[99999]
    assert result['owned_worker_cleanup']['completed_epoch']<=result['completed_epoch']
    assert result['source_intent_sha256']==sha(result['source_intent'])
    assert result['context_start_epoch']==c.ctx.started and result['training_clock_sha256']==clock_sha
    saved=read_json(c.root/'CONTROLLER_STATE.json')
    assert saved['status']=='RESOURCE_PAUSED' and saved['stop_reason']=='PARENT_MEMORY_GUARD'
    recovered=saved['states'][submitted[0]]
    assert recovered['visits']==2 and recovered['requires_current_dataset_refresh']
    assert recovered['label_counts']['TEST']==48
    assert sha(c.model)==model_sha and sha(c.root/'TRAINING_CLOCK.json')==clock_sha
    assert read_json(saved['resource_pause_receipt_path'])==result
    assert not result['registry_frozen'] and not result['TEST_opened']
    assert not (c.root/'REGISTRY_FREEZE.json').exists() and not (c.root/'RESULT.json').exists()


@pytest.mark.parametrize('where',['startup','device','journal'])
def test_resource_pause_covers_parent_exception_boundaries(setup_case,monkeypatch,where):
    c=setup_case();calls=[]
    def memory(*a,**kw):raise MemoryError('Synthetic parent '+where+' RAM guard')
    if where=='startup':monkeypatch.setattr(module,'gate_inputs',memory)
    elif where=='device':monkeypatch.setattr(module,'choose_device',memory)
    else:
        write_json(c.root/'SELECTED_REASSESSMENT_SELECTION.json',{'synthetic_journal':True})
        monkeypatch.setattr(module,'reassess_selected_before_freeze',memory)
    monkeypatch.setattr(module,'ProcessPoolExecutor',lambda **kw:pytest.fail('No worker pool expected'))
    result=module.run(c.ctx,c.config_path)
    assert result['status']=='RESOURCE_PAUSED' and result['owned_worker_cleanup']['mode']=='NO_ACTIVE_POOL_AT_EXCEPTION_BOUNDARY'
    assert result['no_new_registry_or_TEST_during_pause']
    assert result['prior_controller_source_intent']==str(c.prior_intent)
    assert result['prior_controller_source_intent_sha256']==sha(c.prior_intent)
    assert result['current_training_config_sha256']==sha(c.config_path)
    if where in ('startup','journal'):
        assert result['source_intent'] is None and result['source_intent_sha256'] is None
        assert result['source_intent_scope']=='NO_CURRENT_INVOCATION_INTENT_CREATED'
    else:
        assert result['source_intent']!=str(c.prior_intent)
        assert result['source_intent_sha256']==sha(result['source_intent'])
        assert read_json(result['source_intent'])['pilot'] is False
    assert not (c.root/'REGISTRY_FREEZE.json').exists() and not (c.root/'TEST_OPENING_AUTHORIZATION.json').exists()


def test_late_same_clock_restart_finalizes_without_training_or_pilots(setup_case,monkeypatch):
    c=setup_case(stage_remaining=300.);clock_sha=sha(c.root/'TRAINING_CLOCK.json');model_sha=sha(c.model)
    forbid_training(monkeypatch);steps=[]
    def assess(ctx,root,regions,states,support_sha,config,deadline):
        steps.append('reassess');assert deadline==c.clock['stage_deadline_epoch']
        assert states['s03:000000']['selected']['ICNN']['candidate_id']=='selected_1'
        return states
    def freeze(ctx,root,regions,states,support_sha,config,*,deadline):
        steps.append('freeze');assert (root/'FINALIZATION_CLOCK.json').exists()
        return {'frozen_fixture':True}
    def finish(ctx,root,registry,clock,**kw):
        steps.append('finish');assert clock==c.clock and kw['resumed']
        return {'status':'SYNTHETIC_BOUNDED_FINALIZATION_FINISHED'}
    monkeypatch.setattr(module,'reassess_selected_before_freeze',assess)
    monkeypatch.setattr(module,'freeze_registry',freeze)
    monkeypatch.setattr(module,'complete_frozen_training',finish)
    assert module.run(c.ctx,c.config_path)['status']=='SYNTHETIC_BOUNDED_FINALIZATION_FINISHED'
    assert steps==['reassess','freeze','finish']
    assert sha(c.root/'TRAINING_CLOCK.json')==clock_sha and sha(c.model)==model_sha
    intent=read_json(next((c.root/'finalization_recovery_intents').glob('*.json')))
    assert not intent['new_training_or_labels'] and not intent['new_device_or_concurrency_pilots']


def test_dynamic_forecast_science_cutoff_also_uses_finalization_only(setup_case,monkeypatch):
    c=setup_case(stage_remaining=1500.);forbid_training(monkeypatch)
    monkeypatch.setattr(module,'finalization_forecast',lambda *a:{'reserve_seconds':1600.})
    monkeypatch.setattr(module,'finalize_retained_controller',lambda *a:{'status':'FINALIZATION_ONLY'})
    assert module.run(c.ctx,c.config_path)['status']=='FINALIZATION_ONLY'


@pytest.mark.parametrize('change',['config','clock','support','empty'])
def test_late_recovery_rejects_changed_identity_or_empty_selection(setup_case,monkeypatch,change):
    c=setup_case(stage_remaining=300.);forbid_training(monkeypatch)
    if change=='config':
        record=read_json(c.root/'CONTROLLER_STATE.json');record['config_sha256']='wrong';write_json(c.root/'CONTROLLER_STATE.json',record)
    elif change in ('clock','support'):
        intent=read_json(c.prior_intent)
        if change=='clock':intent['clock']['stage_deadline_epoch']+=1
        else:intent['support_freeze_sha256']='wrong'
        write_json(c.prior_intent,intent)
    else:
        controller=read_json(c.root/'CONTROLLER_STATE.json');controller['states']['s03:000000']['selected']={}
        write_json(c.root/'CONTROLLER_STATE.json',controller)
        write_json(c.model.parent/'STATE.json',controller['states']['s03:000000'])
    with pytest.raises(ValueError):module.run(c.ctx,c.config_path)
    assert not (c.root/'REGISTRY_FREEZE.json').exists() and not (c.root/'RESULT.json').exists()


@pytest.mark.parametrize('hard',[False,True])
def test_expired_unfrozen_stage_is_metadata_only_incomplete(setup_case,monkeypatch,hard):
    c=setup_case(stage_remaining=-1.);clock_sha=sha(c.root/'TRAINING_CLOCK.json');model_sha=sha(c.model)
    if hard:c.ctx.c['hard_deadline_epoch']=time.time()-.5
    monkeypatch.setattr(module,'_run_training',lambda *a,**kw:pytest.fail('Expired recovery entered scientific controller'))
    result=module.run(c.ctx,c.config_path)
    assert result['status']==('CAMPAIGN_HARD_DEADLINE_EXPIRED' if hard else 'STAGE2_EXPIRED_BEFORE_REGISTRY_FREEZE')
    assert result['metadata_only'] and not result['completed_stage2'] and not result['new_model_availability_claimed']
    assert not (c.root/'REGISTRY_FREEZE.json').exists() and not (c.root/'RESULT.json').exists()
    assert not (c.root/'TEST_OPENING_AUTHORIZATION.json').exists()
    assert sha(c.model)==model_sha and sha(c.root/'TRAINING_CLOCK.json')==clock_sha


def test_existing_registry_after_stage_expiry_keeps_authenticated_completion_path(setup_case,monkeypatch):
    c=setup_case(stage_remaining=-1.)
    write_json(c.root/'REGISTRY_FREEZE.json',{'synthetic':'existing frozen registry'})
    forbid_training(monkeypatch);called=[]
    monkeypatch.setattr(module,'complete_frozen_training',lambda *a,**kw:called.append(kw) or {'status':'EXISTING_REGISTRY_RECOVERY'})
    assert module.run(c.ctx,c.config_path)['status']=='EXISTING_REGISTRY_RECOVERY' and called[0]['resumed']
    assert not (c.root/'finalization_expiry_receipts').exists()


def test_late_freeze_reaching_original_stage_deadline_records_incomplete(setup_case,monkeypatch):
    c=setup_case(stage_remaining=300.);forbid_training(monkeypatch)
    monkeypatch.setattr(module,'reassess_selected_before_freeze',lambda ctx,root,regions,states,*a:states)
    def freeze(*a,**kw):
        monkeypatch.setattr(module.time,'time',lambda:c.clock['stage_deadline_epoch']+1.)
        raise TimeoutError('Original Stage2 deadline during freeze')
    monkeypatch.setattr(module,'freeze_registry',freeze)
    result=module.run(c.ctx,c.config_path)
    assert result['status']=='STAGE2_EXPIRED_BEFORE_REGISTRY_FREEZE' and not result['completed_stage2']
    assert not (c.root/'RESULT.json').exists() and not (c.root/'REGISTRY_FREEZE.json').exists()


def test_late_restart_without_retained_controller_is_not_success(setup_case,monkeypatch):
    c=setup_case(stage_remaining=300.,retained=False);forbid_training(monkeypatch)
    with pytest.raises(TimeoutError,match='no retained controller'):
        module.run(c.ctx,c.config_path)
    assert not (c.root/'REGISTRY_FREEZE.json').exists() and not (c.root/'RESULT.json').exists()


def test_late_finalization_memory_pause_binds_this_invocation_recovery_intent(setup_case,monkeypatch):
    c=setup_case(stage_remaining=300.);forbid_training(monkeypatch)
    def memory(*a,**kw):raise MemoryError('Synthetic reassessment RAM guard')
    monkeypatch.setattr(module,'reassess_selected_before_freeze',memory)
    result=module.run(c.ctx,c.config_path)
    assert result['status']=='RESOURCE_PAUSED' and result['source_intent']!=str(c.prior_intent)
    intent=read_json(result['source_intent'])
    assert intent['step_id']=='FINALIZATION_ONLY_RECOVERY' and intent['pilot'] is False
    assert intent['original_source_intent']==str(c.prior_intent) and intent['clock']==c.clock
    assert result['prior_controller_source_intent']==str(c.prior_intent)
    assert not result['registry_frozen'] and not result['TEST_opened']


def test_early_pilot_memory_failure_cannot_overwrite_production_controller(setup_case,monkeypatch):
    c=setup_case();before=sha(c.root/'CONTROLLER_STATE.json');clock_sha=sha(c.root/'TRAINING_CLOCK.json')
    def memory(*a,**kw):raise MemoryError('Synthetic pilot startup RAM guard')
    monkeypatch.setattr(module,'gate_inputs',memory)
    result=module.run(c.ctx,c.config_path,pilot=True,pilot_id='isolated_early_failure',pilot_regions=['s03:000000'],pilot_seconds=60.)
    assert result['status']=='RESOURCE_PAUSED' and result['source_intent'] is None
    pilot_root=c.root/'pilots/isolated_early_failure'
    assert Path(result['controller_state_path'])==pilot_root/'CONTROLLER_STATE.json'
    assert read_json(pilot_root/'CONTROLLER_STATE.json')['status']=='RESOURCE_PAUSED'
    assert sha(c.root/'CONTROLLER_STATE.json')==before and sha(c.root/'TRAINING_CLOCK.json')==clock_sha
    assert result['training_clock_path']==str(c.root/'TRAINING_CLOCK.json')
    with pytest.raises(ValueError,match='safe explicit ID'):
        module.run(c.ctx,c.config_path,pilot=True,pilot_id='../escape',pilot_regions=['s03:000000'],pilot_seconds=60.)
