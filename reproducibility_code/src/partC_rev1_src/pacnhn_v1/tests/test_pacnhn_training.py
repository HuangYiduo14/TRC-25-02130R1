"""Bounded controller tests with synthetic arrays only; no campaign invocation."""
import copy
import csv
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from partC_rev1_src.pacnhn_v1.control import array_hash,read_json,save_npz,sha,write_json
from partC_rev1_src.pacnhn_v1.models import fit_xgboost_cost
from partC_rev1_src.pacnhn_v1.training import (
    _state,_icnn_block,_xgb_block,audit_and_select,monotonic_conflicts,native_scale,
    qualified,read_training_config,schedule_region,stage_clock,
    compact_state,export_curves,export_learning_csv,final_test_audit,freeze_registry,
    freeze_xgb_memory_budget,gate_inputs,region_slice,
    pilot_checkpoint_ready,pilot_progress,schedule_pilot_region,
    selected_learning_metadata,
    finalize_test_with_receipt,
    _requests_for_region,fixed_role_progress,
    complete_frozen_training,ensure_finalization_clock,
)


def settings():
    path=Path(__file__).resolve().parents[3]/"partC_doc_icnn/pacnhn_v1/training_config.json"
    return read_training_config(path)


def dataset(u,tau,c0):
    u=np.asarray(u,float);tau=np.asarray(tau,float);c0=np.asarray(c0,float)
    return dict(u=u,tau=tau,c0=c0,unique_rows=len(u),train_cost_mean=float(tau.mean()),
        dataset_identity=array_hash(dict(u=u,tau=tau,c0=c0)),
        records=[dict(input_identity=array_hash({'u':row})) for row in u])


def test_training_clock_never_resets_or_exceeds_original_cumulative_bound(tmp_path):
    ctx=SimpleNamespace(root=tmp_path,c=dict(original_start_epoch=time.time()-20,
        maximum_cumulative_stage_deadlines=[time.time()+100,time.time()+500],hard_deadline_epoch=time.time()+1000))
    c=settings();first=stage_clock(ctx,c);second=stage_clock(ctx,c)
    assert first==second and first['stage_deadline_epoch']==ctx.c['maximum_cumulative_stage_deadlines'][1]
    first['stage_deadline_epoch']+=1
    write_json(tmp_path/'training/TRAINING_CLOCK.json',first)
    with pytest.raises(ValueError,match='reset'):
        stage_clock(ctx,c)


def test_training_fixed_scaling_floor_and_capacity_fallback():
    c=settings()
    region=SimpleNamespace(baseline_open=np.array([1e-15,100.,0.]),baseline_i95=np.array([0.,200.,0.]),graph_sha256='g')
    scale,evidence=native_scale(region,{},c)
    np.testing.assert_allclose(scale,[1.,200.,1.])
    assert evidence['floor']==1.
    region.baseline_open[:]=0;region.baseline_i95[:]=0
    with pytest.raises(ValueError,match='capacity'):
        native_scale(region,{},c)
    scale,evidence=native_scale(region,dict(physical_capacity_reference=500.,physical_capacity_reference_provenance='actual used capacity'),c)
    np.testing.assert_array_equal(scale,[5.,5.,5.])


def test_training_disjoint_monotonicity_lower_bound():
    train=dataset([[0.],[1.],[2.],[3.]],[[4.],[2.],[3.],[2.]],[1.])
    evidence=monotonic_conflicts(train,np.ones(1),np.array([7],np.uint64))
    assert evidence['fixed_label_L1_error_lower_bound']==3.
    assert evidence['fixed_label_WMAPE_percent_lower_bound']==100*3/11
    assert evidence['violating_disjoint_pairs']==2


def test_training_qualification_rechecks_growing_train_and_valid(tmp_path,monkeypatch):
    import partC_rev1_src.pacnhn_v1.training as module
    root=tmp_path/'region';root.mkdir();state=_state(root/'STATE.json');c=settings()
    class Constant:
        def __init__(self,value):self.value=value
        def cost(self,u):return np.full_like(u,self.value)
    paths=[]
    for i,value in enumerate([3.,2.]):
        p=root/f'{i}.bin';p.write_text(str(value));paths.append(p)
        state['eligible']['XGBOOST'].append(dict(candidate_id=str(i),width=0,model_path=str(p),model_sha256=sha(p)))
    monkeypatch.setattr(module,'load_model',lambda p:Constant(float(Path(p).read_text())))
    train=dataset([[1.],[2.]],[[3.],[3.]],[1.]);valid=dataset([[3.]],[[2.]],[1.])
    audit_and_select(root,'XGBOOST',state,train,valid,c,lambda:None)
    assert state['selected']['XGBOOST']['candidate_id']=='0'
    grown=dataset([[1.],[2.],[4.]],[[2.],[2.],[2.]],[1.])
    audit_and_select(root,'XGBOOST',state,grown,valid,c,lambda:None)
    assert state['selected']['XGBOOST']['candidate_id']=='1'
    assert state['selected']['XGBOOST']['audit']['TRAIN']['WMAPE_percent']==0.
    changed_valid=dataset([[3.],[4.]],[[3.],[3.]],[1.])
    audit_and_select(root,'XGBOOST',state,grown,changed_valid,c,lambda:None)
    assert state['selected']['XGBOOST']['audit']['valid_dataset_identity']==changed_valid['dataset_identity']
    assert len(list((root/'audits/XGBOOST').glob('*.json')))==6


def test_training_resume_preserves_optimizer_and_prior_checkpoints(tmp_path):
    torch.set_num_threads(1);c=settings();c.update(icnn_updates_per_block=2,icnn_widths=[3],icnn_batch_rows=4)
    train=dataset([[.1,.2],[.3,.4],[.5,.6],[.7,.8]],[[2.,3.],[2.1,3.2],[2.3,3.5],[2.7,3.8]],[2.,3.])
    state=_state(tmp_path/'STATE.json')
    first=_icnn_block(tmp_path,state,train,np.ones(2),c,'cpu',time.time()+30,lambda:None)
    previous=copy.deepcopy(state['optimizers']['3'])
    second=_icnn_block(tmp_path,state,train,np.ones(2),c,'cpu',time.time()+30,lambda:None)
    assert first['updates']==second['updates']==2
    assert Path(previous['path']).exists() and sha(previous['path'])==previous['sha256']
    current=torch.load(state['optimizers']['3']['path'],weights_only=True)
    assert current['total_updates']==4
    assert len(state['eligible']['ICNN'])==2


def test_training_xgb_complexity_restart_and_deadline_callback(tmp_path):
    pytest.importorskip('xgboost');c=settings();c.update(xgb_rounds_per_block=2,xgb_max_rounds_per_candidate=2,xgb_max_depth=1,xgb_depth_candidates=[1,2])
    train=dataset([[.1],[.3],[.5],[.7]],[[2.],[2.1],[2.3],[2.7]],[2.])
    region=SimpleNamespace(base=np.array([1.]),k4=np.array([1.]),pair_source=np.array([0]),pair_target=np.array([1]),identity='toy')
    state=_state(tmp_path/'STATE.json')
    for _ in range(2):_xgb_block(tmp_path,state,train,np.ones(1),region,c,'cpu',time.time()+30,lambda:None)
    assert state['xgb_latest']['boosting_rounds_total']==2
    assert state['xgb_latest']['restart_reason']=='PREDECLARED_COMPLEXITY_LIMIT_RETRAIN_ON_CURRENT_TRAIN'
    assert len(state['eligible']['XGBOOST'])==2
    assert [r['max_depth'] for r in state['eligible']['XGBOOST']]==[1,2]
    expired=fit_xgboost_cost(train['u'],train['tau'],train['c0'],np.ones(1),n_estimators=2,deadline_epoch=time.time()-1)
    assert expired.metadata['boosting_rounds_total']==0


def test_training_schedule_s03_first_then_one_in_four_enrichment():
    c=settings();regions=[dict(suite_id=sid,region_index=0,key=sid+':000000') for sid in ['s03','s04','s02','s05','s06']]
    states={};scheduler={}
    assert schedule_region(regions,states,scheduler,c)['suite_id']=='s03'
    states['s03:000000']=dict(visits=1,dataset_identity='train',valid_dataset_identity='valid',last_visit_epoch=1.,
        fixed_role_progress=dict(complete=True),
        selected={f:dict(audit=dict(train_dataset_identity='train',valid_dataset_identity='valid',target_met=True)) for f in ['ICNN','XGBOOST']})
    assert qualified(states['s03:000000'])
    order=[schedule_region(regions,states,scheduler,c)['suite_id'] for _ in range(4)]
    assert order==['s03','s04','s04','s04']


def _scheduler_state(*,resolved,meets=True,train_rows=64,valid_rows=24,test_rows=48,last=1.):
    return dict(visits=1,dataset_identity='train',valid_dataset_identity='valid',last_visit_epoch=last,
        training_rows=train_rows,valid_rows=valid_rows,
        fixed_role_progress=dict(complete=resolved,roles={'TEST':dict(admitted=test_rows)}),
        selected={f:dict(audit=dict(train_dataset_identity='train',valid_dataset_identity='valid',target_met=meets)) for f in ['ICNN','XGBOOST']})


def test_scheduler_pending_fixed_cohorts_precede_persistently_unmet_accuracy():
    c=settings();regions=[dict(suite_id='s03',region_index=i,key=f's03:{i:06d}') for i in [0,3,9]]
    states={regions[0]['key']:_scheduler_state(resolved=False,train_rows=45,valid_rows=14,test_rows=0,last=10.),
        regions[1]['key']:_scheduler_state(resolved=False,valid_rows=22,last=20.),
        regions[2]['key']:_scheduler_state(resolved=True,meets=False,last=0.)}
    assert qualified(states[regions[0]['key']])  # Accuracy is a distinct metric.
    scheduler={}
    assert schedule_region(regions,states,scheduler,c)['region_index']==0
    assert schedule_region(regions,states,scheduler,c,exclude_keys=[regions[0]['key']])['region_index']==3
    # Pending fixed cohorts in flight cannot leave a free worker idle.
    assert schedule_region(regions,states,scheduler,c,exclude_keys=[r['key'] for r in regions[:2]])['region_index']==9
    assert not scheduler.get('first_all_s03_qualified')


def test_scheduler_milestone_requires_resolved_roles_and_drained_current_audits():
    c=settings();regions=[dict(suite_id='s03',region_index=i,key=f's03:{i:06d}') for i in [0,3]]
    states={r['key']:_scheduler_state(resolved=False) for r in regions};scheduler={}
    schedule_region(regions,states,scheduler,c)
    assert not scheduler.get('first_all_s03_qualified')
    for state in states.values():state['fixed_role_progress']['complete']=True
    assert schedule_region(regions,states,scheduler,c,exclude_keys=[regions[0]['key']]) is None
    assert not scheduler.get('first_all_s03_qualified')
    assert schedule_region(regions,states,scheduler,c) is not None
    assert scheduler['first_all_s03_qualified']>0
    # It remains a historical latch if later enriched TRAIN no longer qualifies.
    states[regions[0]['key']]['selected']['ICNN']['audit']['target_met']=False
    stamp=scheduler['first_all_s03_qualified'];schedule_region(regions,states,scheduler,c)
    assert scheduler['first_all_s03_qualified']==stamp


def test_scheduler_additional_suite_pending_cohorts_and_active_exclusions():
    c=settings();regions=[dict(suite_id=sid,region_index=i,key=f'{sid}:{i:06d}') for sid,i in [('s03',0),('s04',0),('s04',1),('s02',0)]]
    states={r['key']:_scheduler_state(resolved=True) for r in regions}
    states['s04:000000']=_scheduler_state(resolved=False,train_rows=45,valid_rows=14,test_rows=0,last=20.)
    states['s04:000001']=_scheduler_state(resolved=True,meets=False,last=1.)
    scheduler=dict(first_all_s03_qualified=1.,postqualification_slices=1)
    assert schedule_region(regions,states,scheduler,c)['key']=='s04:000000'
    scheduler['postqualification_slices']=1
    assert schedule_region(regions,states,scheduler,c,exclude_keys=['s04:000000'])['key']=='s04:000001'
    states['s02:000000']['fixed_role_progress']['complete']=False
    scheduler['postqualification_slices']=1
    assert schedule_region(regions,states,scheduler,c,exclude_keys=['s04:000000','s04:000001'])['key']=='s02:000000'


def test_owned_pause_request_preserves_clock_and_never_finalizes(tmp_path):
    from partC_rev1_src.pacnhn_v1.training import owned_stop_request,pause_without_finalization
    ctx=SimpleNamespace(c={'run_id':'test_run'})
    clock={'stage_deadline_epoch':time.time()+1000,'stage_start_epoch':time.time()-100}
    write_json(tmp_path/'TRAINING_CLOCK.json',clock);clock_sha=sha(tmp_path/'TRAINING_CLOCK.json')
    controller=dict(states={'s03:000000':{'visits':1}},slices=1)
    request=dict(phase_id='PACNHN_V1',run_id='test_run',action='PAUSE_FOR_REPAIR',target_pid=os.getpid(),
        request_id='test_pause',reason='bounded control regression',stage_deadline_epoch=clock['stage_deadline_epoch'])
    write_json(tmp_path/'STOP_REQUEST.json',dict(request,target_pid=-1))
    assert owned_stop_request(ctx,tmp_path,controller,clock) is None
    write_json(tmp_path/'STOP_REQUEST.json',request)
    bound=owned_stop_request(ctx,tmp_path,controller,clock)
    result=pause_without_finalization(tmp_path,controller,clock,bound,tmp_path/'INTENT.json')
    assert result['status']=='PAUSED_FOR_REPAIR' and not result['TEST_opened'] and not result['registry_frozen']
    assert sha(tmp_path/'TRAINING_CLOCK.json')==clock_sha
    assert owned_stop_request(ctx,tmp_path,controller,clock) is None
    assert not (tmp_path/'REGISTRY_FREEZE.json').exists() and not (tmp_path/'TEST_OPENING_AUTHORIZATION.json').exists()
    assert read_json(tmp_path/'CONTROLLER_STATE.json')['states']==controller['states']


def test_training_run_drains_owned_pause_without_new_submission_or_test(tmp_path,monkeypatch):
    from concurrent.futures import Future
    import partC_rev1_src.pacnhn_v1.training as module
    import partC_rev1_src.pacnhn_v1.gates as gates
    config=settings();config_path=tmp_path/'config.json';write_json(config_path,config)
    for name in ['SUPPORT_FREEZE.json','COMPLETE_WITNESS_AUDIT.json']:write_json(tmp_path/name,{'synthetic':True})
    now=time.time();ctx=SimpleNamespace(root=tmp_path,path=config_path,started=now,
        c=dict(phase_id='PACNHN_V1',run_id='test_run',suite_ids=['s03'],original_start_epoch=now-10,
            maximum_cumulative_stage_deadlines=[now+100,now+5000],hard_deadline_epoch=now+6000,
            resources={'added_disk_bytes':10**12},acceptance_tests=['synthetic']),
        bind=lambda *a,**kw:None,guard=lambda **kw:None,flush_bindings=lambda:None)
    clock=stage_clock(ctx,config);clock_sha=sha(tmp_path/'training/TRAINING_CLOCK.json')
    regions=[dict(suite_id='s03',region_index=i,key=f's03:{i:06d}',pairs=1) for i in [0,1]]
    monkeypatch.setattr(gates,'require_component',lambda *a,**kw:None)
    monkeypatch.setattr(module,'gate_inputs',lambda ctx:({},regions))
    monkeypatch.setattr(module,'freeze_xgb_memory_budget',lambda *a,**kw:{})
    monkeypatch.setattr(module,'choose_device',lambda *a,**kw:'cpu')
    monkeypatch.setattr(module,'concurrency_gate',lambda *a,**kw:1)
    for name in ['freeze_registry','final_test_audit','complete_frozen_training','ensure_finalization_clock']:
        monkeypatch.setattr(module,name,lambda *a,**kw:pytest.fail('Pause entered scientific finalization'))
    submitted=[]
    class Pool:
        def __init__(self,**kwargs):pass
        def submit(self,fn,payload):
            submitted.append(payload['record']['key']);assert len(submitted)==1
            write_json(tmp_path/'training/STOP_REQUEST.json',dict(phase_id='PACNHN_V1',run_id='test_run',
                action='PAUSE_FOR_REPAIR',target_pid=os.getpid(),request_id='while_worker_pending',reason='test',
                stage_deadline_epoch=clock['stage_deadline_epoch']))
            future=Future();future.set_result(_scheduler_state(resolved=True));return future
        def shutdown(self,**kwargs):pass
    monkeypatch.setattr(module,'ProcessPoolExecutor',Pool)
    result=module.run(ctx,config_path)
    assert result['status']=='PAUSED_FOR_REPAIR' and submitted==['s03:000000']
    saved=read_json(tmp_path/'training/CONTROLLER_STATE.json')
    assert saved['slices']==1 and saved['states']['s03:000000']['visits']==1
    assert sha(tmp_path/'training/TRAINING_CLOCK.json')==clock_sha
    assert not (tmp_path/'training/REGISTRY_FREEZE.json').exists()
    assert not (tmp_path/'training/TEST_OPENING_AUTHORIZATION.json').exists()


def test_training_requires_exact_witness_support_binding(tmp_path):
    write_json(tmp_path/'SUPPORT_FREEZE.json',{'status':'PASS'})
    write_json(tmp_path/'COMPLETE_WITNESS_AUDIT.json',{'status':'PASS','support_freeze_sha256':'different'})
    with pytest.raises(ValueError,match='exact frozen support'):
        gate_inputs(SimpleNamespace(root=tmp_path))


def test_training_memory_budget_is_positive_proportional_and_frozen(tmp_path,monkeypatch):
    import partC_rev1_src.pacnhn_v1.training as module
    write_json(tmp_path/'SUPPORT_FREEZE.json',{'frozen':'fixture'})
    ctx=SimpleNamespace(root=tmp_path,c={'suite_ids':['s03'],'resources':{
        'tree_rss_bytes':24*1024**3,'minimum_available_bytes':3*1024**3}})
    monkeypatch.setattr(module.psutil,'virtual_memory',lambda:SimpleNamespace(available=14*1024**3))
    regions=[{'suite_id':'s03','pairs':1},{'suite_id':'s03','pairs':3}]
    budget=freeze_xgb_memory_budget(ctx,regions,settings())
    assert budget['per_suite_raw_booster_bytes']==3*1024**3
    assert all(r['xgb_raw_budget_bytes']<=settings()['xgb_max_projected_model_bytes'] for r in regions)
    monkeypatch.setattr(module.psutil,'virtual_memory',lambda:SimpleNamespace(available=8*1024**3))
    assert freeze_xgb_memory_budget(ctx,regions,settings())==budget


def test_training_v2_quota_floor_preserves_original_budget_and_record(tmp_path,monkeypatch):
    import partC_rev1_src.pacnhn_v1.training as module
    write_json(tmp_path/'SUPPORT_FREEZE.json',{'frozen':'fixture'})
    ctx=SimpleNamespace(root=tmp_path,c={'suite_ids':['s03'],'resources':{
        'tree_rss_bytes':24*1024**3,'minimum_available_bytes':3*1024**3}})
    monkeypatch.setattr(module.psutil,'virtual_memory',lambda:SimpleNamespace(available=14*1024**3))
    regions=[dict(suite_id='s03',key='small',pairs=1),dict(suite_id='s03',key='large',pairs=73946)]
    original=freeze_xgb_memory_budget(ctx,regions,settings())
    p=tmp_path/'training/XGB_REGISTRY_MEMORY_BUDGET.json';digest=sha(p)
    config=settings();config['xgb_region_minimum_allocation_bytes']=196608
    monkeypatch.setattr(module.psutil,'virtual_memory',lambda:SimpleNamespace(available=6*1024**3))
    current=freeze_xgb_memory_budget(ctx,regions,config)
    assert sha(p)==digest and current['per_suite_raw_booster_bytes']==original['per_suite_raw_booster_bytes']
    allocation=read_json(current['allocation_path'])
    assert allocation['frozen_budget_sha256']==digest
    assert regions[0]['xgb_raw_budget_bytes']>=196608
    assert allocation['per_suite_allocated']['s03']<=original['per_suite_raw_booster_bytes']
    assert freeze_xgb_memory_budget(ctx,regions,config)==current


def test_training_v2_adapts_remaining_tree_quota_without_identical_restart(tmp_path):
    torch.set_num_threads(1)
    config=settings();config.update(xgb_rounds_per_block=4,xgb_max_depth=1,xgb_depth_candidates=[1],
        xgb_adaptive_quota_blocks=True,xgb_log_center_policy='TRAIN_OUTPUT_LOG_CENTER_V2')
    train=dataset([[0.],[1.],[2.],[3.]],[[2.],[2.],[2.],[2.]],[2.])
    region=SimpleNamespace(base=np.array([2.]),k4=np.array([0.]),pair_source=np.array([0]),pair_target=np.array([1]),identity='toy')
    state=_state(tmp_path/'STATE.json')
    _xgb_block(tmp_path,state,train,np.ones(1),region,config,'cpu',time.time()+30,lambda:None)
    first=copy.deepcopy(state['xgb_latest']);b=first['uncompressed_model_bytes']
    # The frozen1.5 projection admits exactly two more rounds, not four.
    config['xgb_max_projected_model_bytes']=int(np.ceil(1.5*b*6.01/4))
    outcome=_xgb_block(tmp_path,state,train,np.ones(1),region,config,'cpu',time.time()+30,lambda:None)
    assert outcome['rounds']==2 and state['xgb_latest']['boosting_rounds_total']==6
    assert state['xgb_latest']['restart_reason'] is None
    checkpoint=copy.deepcopy(state['xgb_latest'])
    config['xgb_max_projected_model_bytes']=checkpoint['uncompressed_model_bytes']
    outcome=_xgb_block(tmp_path,state,train,np.ones(1),region,config,'cpu',time.time()+30,lambda:None)
    assert outcome['rounds']==0 and 'BOUND' in outcome['status']
    assert state['xgb_latest']==checkpoint and sha(first['model_path'])==first['model_sha256']


def test_training_conditioning_probe_reuses_only_fixed_roles_and_original_clock(tmp_path,monkeypatch):
    import partC_rev1_src.pacnhn_v1.training as module
    import partC_rev1_src.pacnhn_v1.gates as gates
    torch.set_num_threads(1)
    config=read_training_config(Path(__file__).resolve().parents[3]/'partC_doc_icnn/pacnhn_v1/training_config_v2.json')
    config.update(initial_train=4,fixed_valid=2,blind_test=2,icnn_updates_per_block=2,
        icnn_widths=[3],xgb_rounds_per_block=2,xgb_depth_candidates=[1],conditioning_probe_icnn_blocks=1)
    config_path=tmp_path/'config.json';write_json(config_path,config)
    graph=tmp_path/'graph.npz'
    save_npz(graph,tail=np.array([0],np.int64),head=np.array([1],np.int64),node_ids=np.array(['P0','P1']),
        edge_ids=np.array(['toy']),base=np.array([2.]),k4=np.array([0.]),pair_source=np.array([0],np.int64),
        pair_target=np.array([1],np.int64),pair_keys=np.array([1],np.uint64),baseline_open=np.array([1.]),baseline_i95=np.array([2.]),c0=np.array([2.]))
    record=dict(path=str(graph),sha256=sha(graph),key='s03:000000',suite_id='s03',region_index=0,pairs=1,xgb_raw_budget_bytes=1024**2)
    def bind(path,role,expected=None):
        digest=sha(path)
        if expected:assert digest==expected
        return digest
    now=time.time()
    ctx=SimpleNamespace(root=tmp_path,started=now,c=dict(run_id='SYNTHETIC_TEST',phase_id='PACNHN_V1',suite_ids=['s03'],
        original_start_epoch=now-20,maximum_cumulative_stage_deadlines=[now+100,now+2000],hard_deadline_epoch=now+3000,resources={}),
        bind=bind,guard=lambda **kwargs:None,flush_bindings=lambda:None)
    source=tmp_path/'training/pilots/source'
    regional_root=source/'regions/s03/000000'
    region_slice(ctx,record,regional_root,config,'cpu',time.time()+60,30.)
    write_json(source/'PILOT_RESULT.json',dict(status='PILOT_PASS',TEST_opened=False))
    stage_clock(ctx,config);clock_sha=sha(tmp_path/'training/TRAINING_CLOCK.json')
    source_state_sha=sha(regional_root/'STATE.json')
    monkeypatch.setattr(gates,'require_component',lambda *a,**kw:None)
    monkeypatch.setattr(gates,'component_snapshot',lambda *a,**kw:{'synthetic':'fixed'})
    monkeypatch.setattr(module,'gate_inputs',lambda ctx:({},[record]))
    monkeypatch.setattr(module,'freeze_xgb_memory_budget',lambda *a,**kw:{})
    original=module.load_admitted_dataset;roles=[]
    def blinded(path,*,role):
        roles.append(role);assert role in {'TRAIN','VALID'}
        return original(path,role=role)
    monkeypatch.setattr(module,'load_admitted_dataset',blinded)
    result=module.conditioning_probe(ctx,config_path,source_pilot_id='source',probe_id='repair',region_keys=['s03:000000'],seconds=60.)
    assert result['status']=='PASS' and result['TEST_opened'] is False
    assert roles==['TRAIN','VALID']
    assert sha(tmp_path/'training/TRAINING_CLOCK.json')==clock_sha and sha(regional_root/'STATE.json')==source_state_sha
    row=result['rows'][0]
    assert row['initialization']['total_updates']==0 and not row['initialization']['eligible_for_trained_selection']
    assert row['selected']['ICNN']['total_updates']==2
    assert all(r['portable_roundtrip_exact'] for r in row['portable_checks'])
    assert not (tmp_path/'training/TEST_OPENING_AUTHORIZATION.json').exists()


def test_training_toy_handoff_freezes_before_test_and_exports_measured_curves(tmp_path):
    from partC_rev1_src.pacnhn_v1.regional import load_admitted_dataset
    torch.set_num_threads(1);root=tmp_path/'training';root.mkdir()
    graph=tmp_path/'graph.npz'
    save_npz(graph,tail=np.array([0],np.int64),head=np.array([1],np.int64),node_ids=np.array(['P0','P1']),
        edge_ids=np.array(['toy']),base=np.array([2.]),k4=np.array([0.]),pair_source=np.array([0],np.int64),
        pair_target=np.array([1],np.int64),pair_keys=np.array([1],np.uint64),baseline_open=np.array([1.]),baseline_i95=np.array([2.]),c0=np.array([2.]))
    record=dict(path=str(graph),sha256=sha(graph),key='s03:000000',suite_id='s03',region_index=0,pairs=1)
    def bind(path,role,expected=None):
        if expected:assert sha(path)==expected
    ctx=SimpleNamespace(root=tmp_path,c={'run_id':'SYNTHETIC_UNIT_TEST','suite_ids':['s03']},
        bind=bind,guard=lambda **kwargs:None)
    config=settings();config.update(initial_train=4,fixed_valid=2,blind_test=2,train_growth=2,
        icnn_updates_per_block=2,icnn_widths=[8],xgb_rounds_per_block=2,xgb_depth_candidates=[1])
    regional_root=root/'regions/s03/000000';started=time.time()
    state=region_slice(ctx,record,regional_root,config,'cpu',time.time()+60,30.)
    assert set(state['selected'])=={'ICNN','XGBOOST'}
    with pytest.raises(PermissionError,match='TEST'):
        load_admitted_dataset(regional_root/'labels',role='TEST')
    assert not (root/'TEST_OPENING_AUTHORIZATION.json').exists()
    registry=freeze_registry(ctx,root,[record],{record['key']:compact_state(state,regional_root)},'support-fixture',config)
    assert len(registry['models'])==4 and all(r['status']=='AVAILABLE' for r in registry['models'])
    frozen_sha=sha(root/'REGISTRY_FREEZE.json')
    final_test_audit(ctx,root,registry,time.time()+60)
    assert sha(root/'REGISTRY_FREEZE.json')==frozen_sha
    assert read_json(root/'TEST_OPENING_AUTHORIZATION.json')['registry_sha256']==frozen_sha
    result_sha=sha(root/'final_test/s03/000000.json')
    final_test_audit(ctx,root,registry,time.time()+60)
    assert sha(root/'final_test/s03/000000.json')==result_sha
    export_learning_csv(root,registry);export_curves(root,{'stage_start_epoch':started})
    with (root/'LEARNING_RESULTS.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==4 and all(r['test_rows']=='2' and r['test_opened']=='True' for r in rows)
    neural=next(r for r in rows if r['family']=='ICNN')
    assert neural['selected_optimizer_updates']=='2' and neural['selected_is_initialization']=='False'
    assert neural['neural_contribution_train_dataset_identity']==neural['train_dataset_identity']
    assert neural['observed_increment_absolute_sum']!='' and neural['structurally_active']!=''
    tree=next(r for r in rows if r['family']=='XGBOOST')
    assert tree['selected_boosting_rounds']=='2' and tree['selected_optimizer_updates']==''
    assert all(r['structurally_active']=='' for r in rows if r['family']!='ICNN')
    with (root/'TRAINING_CURVES.csv').open(newline='') as stream:curves=list(csv.DictReader(stream))
    assert {r['family'] for r in curves}=={'ICNN','XGBOOST'}
    assert all(r['evaluation_kind']=='NEW_CHECKPOINT' and float(r['elapsed_seconds'])>=0 for r in curves)


def _freshness_fixture(tmp_path):
    from partC_rev1_src.pacnhn_v1.regional import load_region_graph,generate_labels,load_admitted_dataset
    from partC_rev1_src.pacnhn_v1.models import fit_power4_cost,save_model
    from partC_rev1_src.pacnhn_v1.training import _evaluate
    graph=tmp_path/'graph.npz';root=tmp_path/'training';regional=root/'regions/s03/000000'
    save_npz(graph,tail=np.array([0],np.int64),head=np.array([1],np.int64),node_ids=np.array(['P0','P1']),
        edge_ids=np.array(['toy']),base=np.array([2.]),k4=np.array([0.]),pair_source=np.array([0],np.int64),
        pair_target=np.array([1],np.int64),pair_keys=np.array([1],np.uint64),baseline_open=np.array([1.]),baseline_i95=np.array([2.]),c0=np.array([2.]))
    region=load_region_graph(graph);config=settings();config.update(initial_train=4,fixed_valid=2,blind_test=2,train_growth=1)
    label_config=dict(phase_id='PACNHN_V1',run_id='SYNTHETIC_FRESHNESS',region_id='s03:000000',graph_path=str(graph),
        graph_sha256=sha(graph),deadline_epoch=time.time()+60,per_label_seconds=2.,max_iterations=5,relative_gap_tolerance=1e-6)
    def append(grow):
        requests=_requests_for_region(region,regional,config,grow=grow,reference_scale=np.ones(1))
        return generate_labels(label_config,regional/'labels',requests=requests)
    append(False)
    train=load_admitted_dataset(regional/'labels',role='TRAIN');valid=load_admitted_dataset(regional/'labels',role='VALID')
    model=fit_power4_cost(train['u'],train['tau'],train['c0'],np.ones(1));receipt=save_model(model,regional/'power.npz')
    state=dict(dataset_identity=train['dataset_identity'],valid_dataset_identity=valid['dataset_identity'],training_rows=4,valid_rows=2,
        power4=dict(model_path=receipt['path'],model_sha256=receipt['sha256'],model_kind=model.kind,
            train_dataset_identity=train['dataset_identity'],valid_dataset_identity=valid['dataset_identity'],
            TRAIN=_evaluate(model,train,8,lambda:None),VALID=_evaluate(model,valid,8,lambda:None)),
        audited_request_index_sha256=sha(regional/'labels/REQUEST_INDEX.json'),selected={},visits=1,
        fixed_role_progress=dict(complete=True),retired=[],failures=[])
    write_json(regional/'STATE.json',state)
    record=dict(path=str(graph),sha256=sha(graph),key='s03:000000',suite_id='s03',region_index=0,pairs=1)
    ctx=SimpleNamespace(root=tmp_path,c={'suite_ids':['s03']},guard=lambda **kw:None)
    return ctx,root,regional,state,record,config,append


def test_registry_freeze_rejects_label_admitted_after_last_state_commit(tmp_path):
    from partC_rev1_src.pacnhn_v1.training import recovered_current_data_state
    ctx,root,regional,state,record,config,append=_freshness_fixture(tmp_path)
    old_state_sha=sha(regional/'STATE.json');old_model_sha=state['power4']['model_sha256']
    append(True)  # Label/index committed, then worker dies before STATE changes.
    assert sha(regional/'STATE.json')==old_state_sha
    recovered=recovered_current_data_state(state,regional)
    assert recovered['requires_current_dataset_refresh'] is True and not qualified(recovered)
    registry=freeze_registry(ctx,root,[record],{record['key']:state},'synthetic_support',config)
    power=next(r for r in registry['models'] if r['family']=='POWER4')
    assert power['status']=='UNAVAILABLE_STALE_CURRENT_DATASET' and not power['target_met']
    assert power['train_rows']==5 and power['prior_audited_train_dataset_identity']==state['dataset_identity']
    assert power['current_train_dataset_identity']!=state['dataset_identity']
    assert sha(power['preserved_model_path'])==old_model_sha
    assert next(r for r in registry['models'] if r['family']=='FFTT')['status']=='AVAILABLE'
    assert not (root/'TEST_OPENING_AUTHORIZATION.json').exists()


def test_recovery_freshness_is_independent_of_controller_visit_counter(tmp_path):
    from partC_rev1_src.pacnhn_v1.training import recovered_current_data_state
    state=_scheduler_state(resolved=True);state.update(visits=9,retired_candidates=3,failure_count=2)
    write_json(tmp_path/'labels/REQUEST_INDEX.json',{'before':'audit'})
    state['audited_request_index_sha256']=sha(tmp_path/'labels/REQUEST_INDEX.json')
    assert qualified(recovered_current_data_state(state,tmp_path))
    # A controller exception may advance its visit count without a new audit.
    state['visits']+=1;write_json(tmp_path/'labels/REQUEST_INDEX.json',{'after':'new admitted label'})
    current=recovered_current_data_state(state,tmp_path)
    assert current['requires_current_dataset_refresh'] and not qualified(current)
    assert current['visits']==10 and current['retired_candidates']==3 and current['failure_count']==2
    c=settings();regions=[dict(suite_id='s03',region_index=i,key=f's03:{i:06d}') for i in [0,1]]
    states={regions[0]['key']:current,regions[1]['key']:_scheduler_state(resolved=True,meets=False,last=0.)}
    assert schedule_region(regions,states,{},c)['region_index']==0


def test_compact_recovered_state_preserves_blind_counts_and_finalization_forecast(tmp_path):
    from partC_rev1_src.pacnhn_v1.training import recovered_current_data_state,finalization_forecast
    write_json(tmp_path/'labels/REQUEST_INDEX.json',{'synthetic':'unchanged'})
    state=_scheduler_state(resolved=True)
    state.update(audited_request_index_sha256=sha(tmp_path/'labels/REQUEST_INDEX.json'),
        label_summary={'admitted_by_role':{'TRAIN':64,'VALID':24,'TEST':48}},
        last_timing={'dataset_load':.5,'input_load':.1,'graph_file_bytes':100})
    first=compact_state(state,tmp_path);second=recovered_current_data_state(first,tmp_path)
    third=recovered_current_data_state(second,tmp_path)
    assert first['label_counts']==second['label_counts']==third['label_counts']=={'TRAIN':64,'VALID':24,'TEST':48}
    regions=[{'graph_file_bytes':100}]
    before=finalization_forecast(regions,{'region':first},settings())
    after=finalization_forecast(regions,{'region':third},settings())
    assert before==after and after['projected_test_IO_seconds']>0


def test_existing_registry_and_test_resume_reject_changed_admission_commitment(tmp_path):
    from partC_rev1_src.pacnhn_v1.training import validate_registry_current_data
    ctx,root,regional,state,record,config,append=_freshness_fixture(tmp_path)
    registry=freeze_registry(ctx,root,[record],{record['key']:state},'synthetic_support',config)
    assert validate_registry_current_data(root,registry)
    digest=sha(root/'REGISTRY_FREEZE.json');append(True)
    with pytest.raises(ValueError,match='population changed'):
        freeze_registry(ctx,root,[record],{record['key']:state},'synthetic_support',config)
    with pytest.raises(ValueError,match='population changed'):
        final_test_audit(ctx,root,registry,time.time()+60)
    with pytest.raises(ValueError,match='population changed'):
        complete_frozen_training(ctx,root,registry,{},resumed=True)
    assert sha(root/'REGISTRY_FREEZE.json')==digest
    assert not (root/'TEST_OPENING_AUTHORIZATION.json').exists()
    legacy=dict(registry);legacy.pop('current_dataset_commitment')
    with pytest.raises(ValueError,match='lacks current admitted'):
        validate_registry_current_data(root,legacy)


def test_untrained_region_freezes_fftt_without_fabricated_dataset(tmp_path):
    ctx,root,regional,state,record,config,append=_freshness_fixture(tmp_path)
    # A different output root has the same physical region and no label directory.
    untouched=tmp_path/'untouched';untouched.mkdir()
    registry=freeze_registry(ctx,untouched,[record],{},'synthetic_support',config)
    assert [r['status'] for r in registry['models']]==['AVAILABLE','MISSING','MISSING','MISSING']
    assert all(r['train_rows']==r['valid_rows']==r['test_rows']==0 for r in registry['models'])
    assert registry['current_dataset_commitment']['regions'][record['key']]['request_index_sha256'] is None


def test_training_pilot_revisits_incomplete_regions_and_requires_all_selected():
    regions=[dict(key=key) for key in ('small','median','largest')]
    states={};attempts={}
    def ready():
        metric={'WMAPE_percent':5.,'unique_inputs':24}
        return dict(dataset_identity='train',valid_dataset_identity='valid',fixed_role_progress={'complete':True},selected={family:dict(
            model_path=f'{family}.npz',model_sha256=family,audit=dict(train_dataset_identity='train',
                valid_dataset_identity='valid',TRAIN=metric.copy(),VALID=metric.copy(),target_met=False))
            for family in ('ICNN','XGBOOST')})
    # First120s large-region slice can contain only TRAIN labels and no model.
    for key in ('small','median','largest'):
        row=schedule_pilot_region(regions,states,attempts,now=0.,deadline=600.)
        assert row['key']==key;attempts[key]=1
        states[key]=ready() if key!='largest' else {'visits':1,'status':'INCOMPLETE_REGION_SLICE'}
    progress=pilot_progress(regions,states,attempts)
    assert progress['regions_attempted']==3 and progress['regions_completed']==2
    assert not progress['all_required_complete']
    assert schedule_pilot_region(regions,states,attempts,now=120.,deadline=600.)['key']=='largest'
    assert schedule_pilot_region(regions,states,attempts,exclude_keys={'largest'},now=120.,deadline=600.) is None
    attempts['largest']+=1;states['largest']=ready();states['largest']['selected'].pop('XGBOOST')
    assert schedule_pilot_region(regions,states,attempts,now=240.,deadline=600.)['key']=='largest'
    assert schedule_pilot_region(regions,states,attempts,now=600.,deadline=600.) is None
    attempts['largest']+=1;states['largest']=ready()
    assert pilot_checkpoint_ready(states['largest']) and not qualified(states['largest'])
    assert pilot_progress(regions,states,attempts)['all_required_complete']
    assert schedule_pilot_region(regions,states,attempts,now=360.,deadline=600.) is None
    states['largest']['dataset_identity']='new_train'
    assert not pilot_checkpoint_ready(states['largest'])
    assert not pilot_progress(regions,states,attempts)['all_required_complete']


def test_training_pilot_never_passes_unattempted_or_hard_blocked_region():
    regions=[{'key':'large'}];states={'large':{'status':'BLOCKED_REGIONAL_INVARIANT'}}
    assert schedule_pilot_region(regions,states,{'large':1},now=0.,deadline=600.) is None
    progress=pilot_progress(regions,states,{'large':1})
    assert progress['regions_completed']==0 and not progress['all_required_complete']
    assert pilot_progress(regions,{}, {})['regions_attempted']==0


def test_training_selected_metadata_preserves_zero_and_never_infers_update_count(tmp_path):
    audit=dict(train_dataset_identity='fixed_current_train',neural_branch=dict(
        train_dataset_identity='fixed_current_train',observed_increment_absolute_sum=0.,
        maximum_native_cost_increment=0.,nonzero_prediction_coordinates=0,structurally_active=False))
    initial=selected_learning_metadata('ICNN',{'candidate_id':'ICNN_99999999','total_updates':0},audit)
    assert initial['selected_optimizer_updates']==0 and initial['selected_is_initialization'] is True
    assert initial['structurally_active'] is False and initial['observed_increment_absolute_sum']==0.
    assert initial['neural_contribution_train_dataset_identity']=='fixed_current_train'
    unknown=selected_learning_metadata('ICNN',{'candidate_id':'ICNN_99999999'},audit)
    assert unknown['selected_optimizer_updates'] is None and unknown['selected_is_initialization'] is None
    trained=selected_learning_metadata('ICNN',{'total_updates':250},audit)
    assert trained['selected_optimizer_updates']==250 and trained['selected_is_initialization'] is False
    tree=selected_learning_metadata('XGBOOST',{'boosting_rounds_total':32,'total_updates':999},audit)
    assert tree['selected_boosting_rounds']==32 and tree['selected_optimizer_updates'] is None
    assert tree['structurally_active'] is None
    assert all(v is None for v in selected_learning_metadata('POWER4',{'total_updates':1},audit).values())
    mismatched=copy.deepcopy(audit);mismatched['neural_branch']['train_dataset_identity']='other'
    with pytest.raises(ValueError,match='another TRAIN'):
        selected_learning_metadata('ICNN',{'total_updates':1},mismatched)
    export_learning_csv(tmp_path,{'models':[
        dict(suite_id='s03',region_index=0,family='ICNN',status='AVAILABLE',**initial),
        dict(suite_id='s03',region_index=0,family='XGBOOST',status='AVAILABLE',**tree)]})
    with (tmp_path/'LEARNING_RESULTS.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    assert rows[0]['observed_increment_absolute_sum']=='0.0' and rows[0]['structurally_active']=='False'
    assert rows[0]['selected_optimizer_updates']=='0' and rows[0]['selected_is_initialization']=='True'
    assert rows[1]['observed_increment_absolute_sum']=='' and rows[1]['selected_boosting_rounds']=='32'


def test_training_test_timeout_binds_registry_without_opening_labels(tmp_path,monkeypatch):
    import partC_rev1_src.pacnhn_v1.training as module
    registry={'selection_frozen_epoch':time.time()-10,'models':[],
        'current_dataset_commitment':{'schema':'PACNHN_CURRENT_DATASET_COMMITMENT_V1','regions':{}}}
    write_json(tmp_path/'REGISTRY_FREEZE.json',registry)
    monkeypatch.setattr(module,'final_test_audit',lambda *args:pytest.fail('Expired TEST labels must not open'))
    assert finalize_test_with_receipt(None,tmp_path,registry,time.time()-1) is None
    receipt=read_json(tmp_path/'FINAL_TEST_TIMEOUT.json')
    assert receipt['registry_sha256']==sha(tmp_path/'REGISTRY_FREEZE.json')
    assert receipt['status']=='INCOMPLETE_TIME_LIMIT' and receipt['VALID_not_substituted'] is True
    assert receipt['TEST_opening_authorization_exists'] is False
    assert not (tmp_path/'TEST_OPENING_AUTHORIZATION.json').exists()


def test_training_slow_fixed_labels_finish_before_any_train_growth(tmp_path,monkeypatch):
    import partC_rev1_src.pacnhn_v1.training as module
    config=settings();config.update(initial_train=2,fixed_valid=2,blind_test=2,train_growth=2)
    def samples(region,*,role,count,seed,start_index,existing_input_ids,reference_scale):
        rows=[]
        for i in range(start_index,start_index+count):
            q=np.array([{'TRAIN':100.,'VALID':200.,'TEST':300.}[role]+i])
            identity=array_hash({'u':q})
            rows.append(dict(role=role,sample_index=i,input_id=identity,sample_id=role+str(i),u=q))
        return rows
    monkeypatch.setattr(module,'sample_requests',samples)
    index={};path=tmp_path/'labels/REQUEST_INDEX.json';seen=[]
    for visit in range(6):
        # Even a caller asking to grow is held at the initial sealed cohort.
        requests=_requests_for_region(None,tmp_path,config,grow=True,reference_scale=np.ones(1))
        for r in requests:
            if r['input_id'] in index:continue
            directory=tmp_path/'labels/requests'/r['role']/r['sample_id']
            save_npz(directory/'input.npz',u=r['u'])
            write_json(directory/'REQUEST.json',{k:v for k,v in r.items() if k!='u'})
            index[r['input_id']]=dict(role=r['role'],sample_id=r['sample_id'],selected_label=None,
                input_file=str(directory/'input.npz'),request_metadata=str(directory/'REQUEST.json'))
        assert sum(row['role']=='TRAIN' for row in index.values())==2
        assert not fixed_role_progress(tmp_path,config,index)['complete']
        # A slow label budget admits only one request per visit, preserving all
        # other role assignments and admitting no synthetic numerical target.
        current=requests[0];seen.append(current['role'])
        index[current['input_id']]['selected_label']={'test_disposition':'MOCK_ADMITTED_METADATA_ONLY'}
        write_json(path,index)
    assert seen==['TRAIN','TRAIN','VALID','VALID','TEST','TEST']
    progress=fixed_role_progress(tmp_path,config)
    assert progress['complete'] and progress['admitted_total']==6 and progress['exhausted_total']==0
    growth=_requests_for_region(None,tmp_path,config,grow=True,reference_scale=np.ones(1))
    assert len(growth)==2 and all(row['role']=='TRAIN' and row['sample_index']>=2 for row in growth)
    # A sealed failure is explicit exhaustion, not a made-up usable label.
    failed=next(row for row in index.values() if row['role']=='VALID')
    failed['selected_label']=None
    for attempt in range(1,4):
        write_json(tmp_path/'labels/labels/VALID'/failed['sample_id']/f'attempt_{attempt:06d}'/'FAILED.json',{'status':'FAILED'})
    write_json(path,index)
    progress=fixed_role_progress(tmp_path,config)
    assert progress['complete'] and progress['roles']['VALID']['admitted']==1
    assert progress['roles']['VALID']['exhausted']==1 and progress['admitted_total']==5


@pytest.mark.parametrize('has_journal',[True,False])
def test_training_resume_after_registry_freeze_recovers_bound_stage_result(tmp_path,monkeypatch,has_journal):
    import partC_rev1_src.pacnhn_v1.training as module
    now=time.time();clock={'stage_start_epoch':now-100,'stage_deadline_epoch':now-1}
    registry={'selection_frozen_epoch':now-20,'models':[],'coverage':[],
        'current_dataset_commitment':{'schema':'PACNHN_CURRENT_DATASET_COMMITMENT_V1','regions':{}}}
    write_json(tmp_path/'REGISTRY_FREEZE.json',registry)
    if has_journal:
        write_json(tmp_path/'FINALIZATION_CLOCK.json',dict(stage_start_epoch=clock['stage_start_epoch'],
            stage_deadline_epoch=clock['stage_deadline_epoch'],finalization_started_epoch=now-30,
            start_basis='PERSISTED_BEFORE_FIRST_REGISTRY_FREEZE',pre_freeze_finalization_interval_known=True))
    monkeypatch.setattr(module,'final_test_audit',lambda *args:pytest.fail('Resume after deadline must not open TEST'))
    result=complete_frozen_training(None,tmp_path,registry,clock,
        controller={'slices':7,'scheduler':{},'source_intent':'original-intent.json'},resumed=True)
    assert result['status']=='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE' and result['clock']==clock
    assert result['registry_sha256']==sha(tmp_path/'REGISTRY_FREEZE.json')
    assert result['test_disposition']['sha256']==sha(tmp_path/'FINAL_TEST_TIMEOUT.json')
    assert result['test_disposition']['kind']=='FINAL_TEST_TIMEOUT.json'
    assert result['finalization_started_epoch']==(now-30 if has_journal else now-20)
    assert result['finalization_started_epoch']+result['finalization_elapsed_seconds']==pytest.approx(result['finalization_ended_epoch'])
    assert result['pre_freeze_finalization_interval_known'] is has_journal
    assert result['finalization_resumed'] and result['completed_after_stage_deadline']
    assert not (tmp_path/'TEST_OPENING_AUTHORIZATION.json').exists()
    if not has_journal:assert 'PRE_FREEZE_FINALIZATION_TIME_UNKNOWN' in result['finalization_start_basis']
    result_sha=sha(tmp_path/'RESULT.json');journal_sha=sha(tmp_path/'FINALIZATION_CLOCK.json')
    assert complete_frozen_training(None,tmp_path,registry,clock,resumed=True)==result
    assert sha(tmp_path/'RESULT.json')==result_sha and sha(tmp_path/'FINALIZATION_CLOCK.json')==journal_sha


def test_training_finalization_journal_precedes_freeze_and_never_resets(tmp_path):
    clock={'stage_start_epoch':time.time()-10,'stage_deadline_epoch':time.time()+60}
    first=ensure_finalization_clock(tmp_path,clock)
    registry={'selection_frozen_epoch':time.time()}
    assert first['finalization_started_epoch']<=registry['selection_frozen_epoch']
    assert first['pre_freeze_finalization_interval_known'] is True
    assert ensure_finalization_clock(tmp_path,clock,registry)==first
