"""Synthetic timer/receipt checks; never import solvers or read national arrays."""
import importlib.util
import sys
from pathlib import Path as _StagePath

# Bind these existing regressions to the staged copy in this isolated pytest
# process; this never changes the live source or the running training process.
_staged_path=_StagePath(__file__).with_name('timing.py')
if _staged_path.exists():
    import partC_rev1_src.pacnhn_v1 as _stage_parent
    _stage_spec=importlib.util.spec_from_file_location('partC_rev1_src.pacnhn_v1.timing',_staged_path)
    _stage_module=importlib.util.module_from_spec(_stage_spec)
    sys.modules[_stage_spec.name]=_stage_module
    _stage_spec.loader.exec_module(_stage_module)
    _stage_parent.timing=_stage_module

import json
import hashlib
from types import SimpleNamespace
from pathlib import Path

import pytest

from partC_rev1_src.pacnhn_v1.timing import (
    JobSpec,Snapshot,collect_job,collect_timing,conditional_forecasts,
    historical_rows,validate_disjoint,verify_live_process,worker_rows,stage_rows,aggregate_workers,export_timing,
    conditioning_rows,controller_rows,verify_training_process,
)


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def spec(tmp_path, full=True):
    return JobSpec('toy',tmp_path/'job',tmp_path/'intent.json','P','open','CFW',full,action='reference')


def history():
    return [dict(iteration=i,elapsed_seconds=t,fresh=True,value_cost_seconds=.1,aon_seconds=1.,
                 **(dict(line_search_seconds=.2,line_search=dict(calls=4)) if i<2 else {}))
            for i,t in enumerate([5.,8.,12.])]


def test_disjoint_intervals_are_not_nested_or_worker_sums():
    def row(a,b,name):
        return dict(task=name,start_epoch=a,end_epoch=b,seconds=b-a,
                    accounting_role='DISJOINT_INTERVAL',time_kind='MEASURED_WALL')
    rows=[row(1.,3.,'one'),row(3.,5.,'two')]
    rows.append(dict(rows[0],accounting_role='NESTED_DETAIL',seconds=100.,time_kind='WORKER_WALL'))
    assert validate_disjoint(rows)['seconds']==4.
    with pytest.raises(ValueError,match='Overlapping'):
        validate_disjoint(rows+[row(4.,6.,'overlap')])
    with pytest.raises(ValueError,match='Only measured'):
        validate_disjoint([dict(rows[0],time_kind='WORKER_WALL')])


def test_full_cycles_forecast_updates_not_time_to_gap(tmp_path):
    rows=conditional_forecasts(spec(tmp_path),history(),14.,'TIME_LIMIT',1000.)
    assert [r['updates'] for r in rows]==[50,100,300]
    assert rows[0]['seconds_low']==157.
    assert rows[0]['seconds_high']==207.
    assert rows[0]['measured_final_tail_seconds']==2.
    assert all(r['time_to_gap_claim'] is False for r in rows)
    partial=conditional_forecasts(spec(tmp_path,False),history(),14.,'CONVERGED',1000.)
    assert partial[0]['status']=='EXCLUDED_PARTIAL_ORIGIN_PILOT'
    assert partial[0]['seconds_low'] is None
    active=conditional_forecasts(spec(tmp_path),history(),14.,'IN_PROGRESS',1000.)
    assert active[0]['measured_final_tail_seconds'] is None


def test_pid_command_config_and_birth_all_bind_running(tmp_path):
    s=spec(tmp_path);config=tmp_path/'config.json'
    record=dict(state='RUNNING',action='reference',scenario='open',pid=42,start_epoch=100.)
    class Proc:
        def is_running(self):return True
        def cmdline(self):return ['python','-m',s.module,'--action','reference','--scenario','open','--config',str(config)]
        def cwd(self):return str(tmp_path)
        def create_time(self):return 99.
    assert verify_live_process(record,s,config,lambda pid:Proc())
    assert not verify_live_process(dict(record,start_epoch=1000),s,config,lambda pid:Proc())
    assert not verify_live_process(record,s,tmp_path/'other.json',lambda pid:Proc())
    assert not verify_live_process(dict(record,state='COMPLETE'),s,config,lambda pid:Proc())
    assert not verify_live_process(dict(record,action='pilot'),s,config,lambda pid:Proc())


def test_completed_ledger_remainder_does_not_claim_cpu_or_io(tmp_path):
    s=spec(tmp_path)
    put(s.intent,dict(run_start_epoch=100.,deadline_epoch=200.,full_coverage=True))
    put(s.path/'RESULT.json',dict(status='TIME_LIMIT',complete_query_start_epoch=100.,deadline_epoch=200.,elapsed_seconds=14.))
    put(s.path/'HISTORY.json',history())
    put(s.path/'STAGES.json',[dict(stage='FEASIBLE_INITIALIZATION_AND_OD_WITNESSES',seconds=2.)])
    rows,forecasts,state=collect_job(s,Snapshot(),config_path=tmp_path/'config.json')
    by={r['category']:r for r in rows}
    assert validate_disjoint(rows)['seconds']==14.
    assert by['UNATTRIBUTED_WITHIN_QUERY']['seconds']==pytest.approx(8.3)
    assert by['CPU_TIME']['seconds'] is None
    assert by['CHECKPOINT_IO']['status']=='NOT_ISOLATED'
    assert by['AON_REFRESH']['time_kind']=='MEASURED_WALL'
    assert all(r['accounting_role']=='NESTED_DETAIL' for r in rows[1:])


def test_stale_running_receipt_is_incomplete_not_live(tmp_path,monkeypatch):
    import partC_rev1_src.pacnhn_v1.timing as module
    s=spec(tmp_path);put(s.intent,dict(run_start_epoch=100.,deadline_epoch=200.))
    put(s.path/'HISTORY.json',history())
    monkeypatch.setattr(module,'verify_live_process',lambda *a:False)
    rows,forecasts,state=collect_job(s,Snapshot(),config_path=tmp_path/'config.json',process_records=[{}],now_epoch=150.)
    assert state['status']=='INCOMPLETE_UNVERIFIED' and rows[0]['seconds'] is None
    monkeypatch.setattr(module,'verify_live_process',lambda *a:True)
    rows,forecasts,state=collect_job(s,Snapshot(),config_path=tmp_path/'config.json',process_records=[{}],now_epoch=150.)
    assert state['status']=='IN_PROGRESS' and rows[0]['seconds']==50.


def test_historical_cumulative_prefixes_never_add(tmp_path):
    path=tmp_path/'old.csv'
    path.write_text('benchmark_id,update_wall_seconds,broader_campaign_elapsed_seconds,later_refresh_only_seconds,worker_count\n'
                    'x45,3353.9,,138.0,4\nx200,14614.3,17724.5,,4\n')
    rows=historical_rows(path,Snapshot())
    assert len(rows)==4 and all(r['accounting_role']=='NESTED_DETAIL' for r in rows)
    assert validate_disjoint(rows)['seconds']==0
    assert all('standalone query clock and matched hardware unknown' in r['timing_boundary'] for r in rows)


def test_missing_jobs_have_no_zero_duration_success(tmp_path):
    config=tmp_path/'campaign.json'
    put(config,dict(phase_id='PACNHN_V1',run_id='synthetic',output_root=str(tmp_path)))
    rows,forecasts,summary=collect_timing(config,now_epoch=100.)
    assert len(summary['jobs'])==2 and summary['status']=='PROVISIONAL'
    assert all(j['status']=='NOT_STARTED' for j in summary['jobs'])
    assert summary['additive_job_wall']['interval_count']==0
    assert all(r['seconds'] is None for r in rows)
    assert all(r['accounting_role']=='NESTED_DETAIL' for r in rows)
    assert all('note' in r and 'notes' not in r for r in rows)


def test_worker_walls_stay_nested_even_with_large_parallel_total(tmp_path):
    receipts=[]
    for index in range(2):
        path=tmp_path/f'training/regions/s03/{index:06d}/timing/visit_000001.json'
        put(path,dict(complete_slice_seconds=20.,input_load=1.,labels=5.,dataset_load=1.,
            power4=2.,training_icnn=3.,training_xgboost=3.,evaluation=1.))
        receipts.extend(worker_rows(path,Snapshot(),tmp_path))
    assert validate_disjoint(receipts)['seconds']==0
    assert all(r['time_kind']=='WORKER_WALL' for r in receipts)
    assert sum(r['seconds'] for r in receipts if r['category']=='WORKER_COMPLETE_SLICE')==40.


def test_full_od_pilot_requires_bound_membership_receipt(tmp_path):
    config=tmp_path/'campaign.json'
    put(config,dict(phase_id='PACNHN_V1',run_id='synthetic',output_root=str(tmp_path),suite_ids=['s03']))
    intent=tmp_path/'intents/pilot_AC_s03_open.json'
    root=tmp_path/'pilots/AC_s03_FFTT_open_fullOD'
    put(intent,dict(run_start_epoch=100.,deadline_epoch=200.,full_coverage=True))
    put(root/'RESULT.json',dict(status='PILOT_UPDATE_LIMIT',complete_query_start_epoch=100.,elapsed_seconds=14.))
    put(root/'HISTORY.json',history())
    snap=Snapshot();snap.json(root/'RESULT.json')
    put(root/'TIMING_PILOT.json',dict(full_coverage=True,exact_served_OD_membership=True,
        result_sha256=snap.identity(root/'RESULT.json')['source_sha256']))
    rows,forecasts,summary=collect_timing(config,now_epoch=120.)
    assert len(summary['jobs'])==3
    selected=[r for r in forecasts if r['task']=='pilot_AC_s03_FFTT_open_fullOD']
    assert len(selected)==3 and all(r['seconds_low'] is not None for r in selected)
    put(root/'TIMING_PILOT.json',dict(full_coverage=True,exact_served_OD_membership=True,result_sha256='tampered'))
    with pytest.raises(ValueError,match='binding'):
        collect_timing(config,now_epoch=120.)


def test_completed_training_parent_and_broad_spans_are_not_added_twice(tmp_path):
    clock=dict(stage_start_epoch=100.,stage_deadline_epoch=200.)
    put(tmp_path/'training/TRAINING_CLOCK.json',clock)
    put(tmp_path/'training/RESULT.json',dict(status='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE',clock=clock,
        finalization_started_epoch=170.,finalization_elapsed_seconds=5.))
    put(tmp_path/'topology/RESULT.json',dict(construction_complete_elapsed_seconds=12.))
    sc=dict(original_stage_start_epoch=180.,deadline_epoch=300.)
    put(tmp_path/'STAGE3_CLOCK.json',sc)
    put(tmp_path/'STAGE3_RESULT.json',dict(status='COMPLETE_WITH_RECORDED_OUTCOMES',clock=sc,elapsed_seconds=50.))
    rows,states=stage_rows(dict(original_start_epoch=10.),tmp_path,Snapshot(),[])
    by={r['category']:r for r in rows}
    assert by['OFFLINE_TRAINING_STAGE_WALL']['seconds']==75.
    assert by['OFFLINE_TRAINING_STAGE_WALL']['end_epoch']==175.
    assert by['OFFLINE_STAGE1_BROAD_WALL']['seconds']==90.
    assert by['ONLINE_ASSIGNMENT_STAGE3_BROAD_WALL']['seconds']==50.
    assert by['OFFLINE_TOPOLOGY_CONSTRUCTION']['start_epoch'] is None
    assert by['OFFLINE_TOPOLOGY_CONSTRUCTION']['end_epoch'] is None
    assert validate_disjoint(rows)['seconds']==75.


def test_training_clock_without_finalization_does_not_invent_terminal_time(tmp_path):
    clock=dict(stage_start_epoch=100.,stage_deadline_epoch=200.)
    put(tmp_path/'training/TRAINING_CLOCK.json',clock)
    rows,states=stage_rows(dict(original_start_epoch=10.),tmp_path,Snapshot(),[])
    training=rows[0]
    assert training['seconds'] is None and training['end_epoch'] is None
    assert training['status']=='INCOMPLETE_NO_TERMINAL_TIME'
    assert training['accounting_role']=='NESTED_DETAIL'
    put(tmp_path/'training/RESULT.json',dict(status='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE',clock=clock))
    rows,states=stage_rows(dict(original_start_epoch=10.),tmp_path,Snapshot(),[])
    assert rows[0]['status']=='MISSING_TERMINAL_FINALIZATION_CLOCK' and rows[0]['seconds'] is None


def test_overlapping_training_parent_remains_nested_with_explicit_reason(tmp_path):
    clock=dict(stage_start_epoch=100.,stage_deadline_epoch=200.)
    put(tmp_path/'training/TRAINING_CLOCK.json',clock)
    put(tmp_path/'training/RESULT.json',dict(status='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE',clock=clock,
        finalization_started_epoch=170.,finalization_elapsed_seconds=5.))
    query=dict(task='query',start_epoch=160.,end_epoch=180.,seconds=20.,
        accounting_role='DISJOINT_INTERVAL',time_kind='MEASURED_WALL')
    rows,states=stage_rows(dict(original_start_epoch=10.),tmp_path,Snapshot(),[query])
    assert rows[0]['accounting_role']=='NESTED_DETAIL'
    assert rows[0]['status']=='COMPLETED_OVERLAP_RETAINED_NESTED'
    assert states[0]['overlap_tasks']==['query']
    assert validate_disjoint([query]+rows)['seconds']==20.


def test_worker_aggregation_is_bounded_and_receipt_identity_based(tmp_path):
    paths=[]
    for index in range(3):
        path=tmp_path/f'training/regions/s03/{index%2:06d}/timing/visit_{index:06d}.json'
        put(path,dict(complete_slice_seconds=20.,input_load=1.,labels=5.,dataset_load=1.,
            power4=2.,training_icnn=3.,training_xgboost=3.,evaluation=1.))
        paths.append(path)
    rows,manifest,digest,raw=aggregate_workers(paths,Snapshot(),tmp_path,retain_individual=True)
    assert len(rows)==9 and len(raw)==27 and len(manifest['receipts'])==3
    complete=next(r for r in rows if r['category']=='WORKER_COMPLETE_SLICE')
    assert complete['seconds']==60. and complete['visit_count']==3 and complete['region_count']==2
    assert complete['worker_scope']=='PRODUCTION' and complete['source_sha256']==digest
    assert validate_disjoint(rows)['seconds']==0
    # Equal timer bytes are real distinct visits, but a repeated path is duplicate accounting.
    with pytest.raises(ValueError,match='Duplicate worker receipt'):
        aggregate_workers(paths+[paths[0]],Snapshot(),tmp_path)


def test_inference_open_and_i95_have_distinct_tasks_and_load_scopes(tmp_path):
    config=tmp_path/'campaign.json'
    put(config,dict(phase_id='PACNHN_V1',run_id='synthetic',output_root=str(tmp_path)))
    paths=[]
    for scenario in ('open','i95'):
        path=tmp_path/f'inference/{scenario}.json'
        put(path,dict(schema='PACNHN_INFERENCE_WALL_V1',full_model=True,registry_sha256='frozen',
            measurement='sequential_calls',repetitions=3,seconds=6.,family='ICNN',suite_id='s03',device='cpu',
            scenario=scenario,load_seconds=2.,baseline_validation_seconds=1.,complete_pilot_seconds=10.))
        paths.append(path)
    rows,_,_=collect_timing(config,inference_receipts=paths)
    inference=[r for r in rows if r['category']=='WHOLE_MODEL_INFERENCE']
    assert len({r['task'] for r in inference})==2
    assert {r['scenario'] for r in inference}=={'open','i95'}
    loads=[r for r in rows if r['category']=='INFERENCE_MODEL_AND_GRAPH_LOAD']
    assert len(loads)==2 and all(r['seconds']==2. and r['accounting_role']=='NESTED_DETAIL' for r in loads)


def test_worker_manifest_digest_binds_exported_bytes_and_raw_rows_are_separate(tmp_path):
    config=tmp_path/'campaign.json'
    put(config,dict(phase_id='PACNHN_V1',run_id='synthetic',output_root=str(tmp_path)))
    path=tmp_path/'training/regions/s03/000000/timing/visit_000001.json'
    put(path,dict(complete_slice_seconds=20.,input_load=1.,labels=5.,dataset_load=1.,
        power4=2.,training_icnn=3.,training_xgboost=3.,evaluation=1.))
    result=export_timing(SimpleNamespace(path=config,root=tmp_path),worker_receipts=[path],retain_individual_worker_rows=True)
    manifest=tmp_path/'analysis/WORKER_RECEIPT_MANIFEST.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==result['worker_receipt_manifest_sha256']
    assert (tmp_path/'analysis/timing_workers_raw.csv').exists()
    assert '_raw_worker_rows' not in result
    assert len(result['worker_receipt_manifest']['receipts'])==1


def conditioning_fixture(root):
    c=dict(phase_id='PACNHN_V1',run_id='toy',suite_ids=['s03'],output_root=str(root))
    clock=dict(stage_start_epoch=100.,stage_deadline_epoch=1000.)
    cp=root/'training/TRAINING_CLOCK.json';put(cp,clock)
    folder=root/'training/conditioning_probes/probe'
    ip=folder/'INTENT.json'
    put(ip,dict(phase_id=c['phase_id'],run_id=c['run_id'],action='conditioning-probe',probe_id='probe',
        started_epoch=120.,deadline_epoch=700.,original_training_clock=clock))
    result=dict(status='PASS',started_epoch=120.,ended_epoch=550.,deadline_epoch=700.,
        intent_sha256=hashlib.sha256(ip.read_bytes()).hexdigest(),
        original_training_clock_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
        rows=[dict(key='s03:000007',elapsed_seconds=425.,timings=[
            dict(family='ICNN',seconds=2.),dict(family='XGBOOST',seconds=399.639)])])
    put(folder/'RESULT.json',result)
    resource=root/'resources/training_17.json';put(resource,dict(pid=17,time=545.,elapsed_seconds=435.))
    return c,clock,folder,resource


def test_conditioning_probe_wall_prefix_and_overrun_are_nested(tmp_path):
    c,clock,folder,resource=conditioning_fixture(tmp_path)
    rows,states=conditioning_rows(c,tmp_path,Snapshot(),resource_receipts={'probe':resource})
    by={r['category']:r for r in rows}
    assert by['CONDITIONING_PROBE_WALL']['seconds']==430.
    assert by['CONDITIONING_CONTEXT_TO_INTENT']['seconds']==10.
    assert by['CONDITIONING_XGBOOST_FIT']['seconds']==399.639
    assert states[0]['result_status']=='PASS'
    assert validate_disjoint(rows)['seconds']==0.
    put(tmp_path/'training/RESULT.json',dict(status='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE',clock=clock,
        finalization_started_epoch=800.,finalization_elapsed_seconds=10.,finalization_ended_epoch=810.,
        finalization_resumed=True,finalization_wall_includes_restart_gaps=True,
        finalization_start_basis='PERSISTED_BEFORE_FIRST_REGISTRY_FREEZE',pre_freeze_finalization_interval_known=True))
    stage,states=stage_rows(c,tmp_path,Snapshot(),rows)
    assert validate_disjoint(rows+stage)['seconds']==710.
    assert stage[0]['status']=='COMPLETED_RECORDED_WALL'
    assert states[0]['finalization_provenance']['finalization_resumed'] is True


def test_probe_binding_missing_end_and_unknown_resource_fail_honestly(tmp_path):
    c,_,folder,_=conditioning_fixture(tmp_path)
    result=json.loads((folder/'RESULT.json').read_text());result['intent_sha256']='wrong'
    put(folder/'RESULT.json',result)
    with pytest.raises(ValueError,match='source binding'):
        conditioning_rows(c,tmp_path,Snapshot())
    (folder/'RESULT.json').unlink()
    rows,states=conditioning_rows(c,tmp_path,Snapshot())
    assert rows[0]['seconds'] is None and rows[0]['end_epoch'] is None
    assert rows[1]['status']=='MISSING_CONTEXT_RESOURCE_ASSOCIATION'
    assert states[0]['status']=='INCOMPLETE_NO_TERMINAL_TIME'
    with pytest.raises(ValueError,match='unknown conditioning probe'):
        conditioning_rows(c,tmp_path,Snapshot(),resource_receipts={'other':tmp_path/'resource.json'})


def test_interrupted_controller_observation_is_bound_not_terminal(tmp_path,monkeypatch):
    import partC_rev1_src.pacnhn_v1.timing as module
    c,clock,_,_=conditioning_fixture(tmp_path)
    resource=tmp_path/'audits/stop/training_42.json'
    put(resource,dict(pid=42,time=220.,elapsed_seconds=100.))
    report=tmp_path/'audits/stop/REPORT.json'
    put(report,dict(status='INTENTIONAL_PAUSE_FOR_SCHEDULER_REPAIR',phase_id=c['phase_id'],run_id=c['run_id'],
        original_clock=clock,stopped_parent_pid=42,all_stopped_processes_verified_absent=True,observed_epoch=250.,
        bindings=[dict(snapshot=str(resource),sha256=hashlib.sha256(resource.read_bytes()).hexdigest())]))
    monkeypatch.setattr(module,'verify_training_process',lambda *a:False)
    descriptor=dict(resource_path=resource,disposition_path=report)
    rows,states=controller_rows(c,tmp_path,Snapshot(),[descriptor],config_path=tmp_path/'config.json')
    assert rows[0]['seconds']==100.
    assert rows[1]['seconds'] is None and rows[1]['end_epoch'] is None
    assert states[0]['terminal_duration_lower_bound_seconds']==100.
    assert states[0]['terminal_duration_upper_bound_seconds']==130.
    assert states[0]['terminal_epoch'] is None
    assert validate_disjoint(rows)['seconds']==0.
    with pytest.raises(ValueError,match='Duplicate controller'):
        controller_rows(c,tmp_path,Snapshot(),[descriptor,descriptor],config_path=tmp_path/'config.json')
    put(resource,dict(pid=42,time=221.,elapsed_seconds=101.))
    with pytest.raises(ValueError,match='snapshot binding'):
        controller_rows(c,tmp_path,Snapshot(),[descriptor],config_path=tmp_path/'config.json')


def test_owned_pause_and_resumed_prefix_preserve_original_parent_wall(tmp_path,monkeypatch):
    import partC_rev1_src.pacnhn_v1.timing as module
    c,clock,_,_=conditioning_fixture(tmp_path)
    resource=tmp_path/'resources/training_42.json';put(resource,dict(pid=42,time=220.,elapsed_seconds=100.))
    pause=tmp_path/'training/pause_receipts/one.json'
    put(pause,dict(status='PAUSED_FOR_REPAIR',clock=clock,completed_epoch=240.,pause_request=dict(
        request=dict(phase_id=c['phase_id'],run_id=c['run_id'],target_pid=42))))
    resumed=tmp_path/'resources/training_43.json';put(resumed,dict(pid=43,time=400.,elapsed_seconds=50.))
    monkeypatch.setattr(module,'verify_training_process',lambda resource,*a:resource['pid']==43)
    rows,states=controller_rows(c,tmp_path,Snapshot(),[
        dict(resource_path=resource,disposition_path=pause),dict(resource_path=resumed)],config_path=tmp_path/'config.json')
    assert rows[1]['seconds']==120. and rows[1]['end_epoch']==240.
    assert rows[2]['status']=='RECORDED_PREFIX_RUNNING_VERIFIED' and rows[2]['seconds']==50.
    assert rows[3]['seconds'] is None  # No future/current-time terminal extrapolation.
    put(tmp_path/'training/RESULT.json',dict(status='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE',clock=clock,
        finalization_started_epoch=800.,finalization_elapsed_seconds=10.))
    stage,_=stage_rows(c,tmp_path,Snapshot(),rows)
    assert validate_disjoint(rows+stage)['seconds']==710.  # Includes the pause/gap once.


def test_training_parent_pid_reuse_worker_and_config_checks(tmp_path):
    resource=dict(pid=42,time=200.,elapsed_seconds=100.)
    config=tmp_path/'config.json'
    class Proc:
        def is_running(self):return True
        def cmdline(self):return ['python','-m','partC_rev1_src.pacnhn_v1.training','--action','train','--config',str(config)]
        def cwd(self):return str(tmp_path)
        def create_time(self):return 95.
    assert verify_training_process(resource,config,lambda pid:Proc())
    assert not verify_training_process(resource,tmp_path/'different.json',lambda pid:Proc())
    assert not verify_training_process(dict(resource,elapsed_seconds=1.),config,lambda pid:SimpleNamespace(
        is_running=lambda:True,cmdline=lambda:Proc().cmdline(),create_time=lambda:200.))
    class Worker(Proc):
        def cmdline(self):return ['python','-c','multiprocessing.spawn.spawn_main(...)']
    assert not verify_training_process(resource,config,lambda pid:Worker())


def test_collector_discovers_probe_and_retains_source_provenance(tmp_path):
    c,clock,folder,resource=conditioning_fixture(tmp_path)
    config=tmp_path/'config.json';put(config,c)
    ledger,_,summary=collect_timing(config,conditioning_resources={'probe':resource})
    assert len(summary['conditioning_probes'])==1
    assert any(r['category']=='CONDITIONING_XGBOOST_FIT' and r['seconds']==399.639 for r in ledger)
    assert {str((folder/'RESULT.json').resolve()),str(resource.resolve())}<={r['path'] for r in summary['input_bindings']}
    assert summary['additive_disjoint_wall']['seconds']==0.


def resource_pause_fixture(root):
    c,clock,_,_=conditioning_fixture(root)
    resource=root/'resources/training_73.json'
    put(resource,dict(pid=73,time=200.,elapsed_seconds=80.))
    intent=root/'training/intents/one.json'
    put(intent,dict(phase_id=c['phase_id'],run_id=c['run_id'],clock=clock,pilot=False))
    receipt=root/'training/resource_pause_receipts/one.json'
    record=dict(schema='PACNHN_RESOURCE_PAUSE_V1',status='RESOURCE_PAUSED',phase_id=c['phase_id'],
        run_id=c['run_id'],pid=73,clock=clock,context_start_epoch=120.,
        training_clock_path=str(root/'training/TRAINING_CLOCK.json'),
        training_clock_sha256=hashlib.sha256((root/'training/TRAINING_CLOCK.json').read_bytes()).hexdigest(),
        source_intent=str(intent),source_intent_sha256=hashlib.sha256(intent.read_bytes()).hexdigest(),
        error_type='MemoryError',reason='Available system RAM guard',completed_epoch=250.,
        owned_worker_cleanup=dict(completed=True,completed_epoch=249.,mode='TERMINATED_OWNED_POOL',owned_pids=[74,75]),
        TEST_opened=False,registry_frozen=False,no_new_registry_or_TEST_during_pause=True)
    put(receipt,record)
    return c,clock,resource,intent,receipt,record


def test_resource_pause_full_context_wall_remains_nested_and_binds_intent(tmp_path):
    c,clock,resource,intent,receipt,_=resource_pause_fixture(tmp_path)
    snapshot=Snapshot()
    rows,states=controller_rows(c,tmp_path,snapshot,[dict(resource_path=resource,disposition_path=receipt)],config_path=tmp_path/'config.json')
    assert rows[0]['seconds']==80. and rows[1]['seconds']==130.
    assert rows[1]['end_epoch']==250. and rows[1]['status']=='RESOURCE_PAUSED_RECORDED_CONTEXT_WALL'
    assert states[0]['source_intent_evidence']=='VERIFIED_IMMUTABLE_INTENT'
    assert states[0]['source_intent']['source_sha256']==hashlib.sha256(intent.read_bytes()).hexdigest()
    assert validate_disjoint(rows)['seconds']==0.
    put(tmp_path/'training/RESULT.json',dict(status='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE',clock=clock,
        finalization_started_epoch=800.,finalization_elapsed_seconds=10.))
    stage,_=stage_rows(c,tmp_path,Snapshot(),rows)
    assert validate_disjoint(stage+rows)['seconds']==710.


@pytest.mark.parametrize('change',[
    {'pid':99},{'run_id':'another-run'},{'training_clock_sha256':'wrong'},
    {'source_intent_sha256':'wrong'},{'context_start_epoch':119.},
    {'completed_epoch':199.},{'no_new_registry_or_TEST_during_pause':False},
    {'owned_worker_cleanup':dict(completed=False,completed_epoch=249.,mode='TERMINATED_OWNED_POOL')},
    {'owned_worker_cleanup':dict(completed=True,completed_epoch=251.,mode='TERMINATED_OWNED_POOL')},
])
def test_resource_pause_rejects_wrong_identity_incomplete_cleanup_and_false_clocks(tmp_path,change):
    c,_,resource,_,receipt,record=resource_pause_fixture(tmp_path)
    put(receipt,dict(record,**change))
    with pytest.raises(ValueError):
        controller_rows(c,tmp_path,Snapshot(),[dict(resource_path=resource,disposition_path=receipt)],config_path=tmp_path/'config.json')


def test_resource_pause_absent_intent_is_explicit_and_prior_registry_is_not_new_test(tmp_path):
    c,_,resource,_,receipt,record=resource_pause_fixture(tmp_path)
    record.update(source_intent=None,source_intent_sha256=None,TEST_opened=True,registry_frozen=True)
    put(receipt,record)
    rows,states=controller_rows(c,tmp_path,Snapshot(),[dict(resource_path=resource,disposition_path=receipt)],config_path=tmp_path/'config.json')
    assert rows[1]['seconds']==130.
    assert rows[1]['status']=='RESOURCE_PAUSED_RECORDED_CONTEXT_WALL_INTENT_UNAVAILABLE'
    assert states[0]['source_intent_evidence']=='ABSENT_RECORDED'
    assert states[0]['source_intent'] is None
    assert 'Existing TEST/registry state may predate' in rows[1]['note']


def test_resource_pause_missing_terminal_epoch_never_uses_last_sample(tmp_path):
    c,_,resource,_,receipt,record=resource_pause_fixture(tmp_path)
    record.pop('completed_epoch');put(receipt,record)
    with pytest.raises(KeyError,match='completed_epoch'):
        controller_rows(c,tmp_path,Snapshot(),[dict(resource_path=resource,disposition_path=receipt)],config_path=tmp_path/'config.json')


def test_resource_pause_bound_intent_must_describe_same_original_clock(tmp_path):
    c,clock,resource,intent,receipt,record=resource_pause_fixture(tmp_path)
    put(intent,dict(phase_id=c['phase_id'],run_id=c['run_id'],clock=dict(clock,stage_start_epoch=101.),pilot=False))
    record['source_intent_sha256']=hashlib.sha256(intent.read_bytes()).hexdigest();put(receipt,record)
    with pytest.raises(ValueError,match='original clock mismatch'):
        controller_rows(c,tmp_path,Snapshot(),[dict(resource_path=resource,disposition_path=receipt)],config_path=tmp_path/'config.json')
