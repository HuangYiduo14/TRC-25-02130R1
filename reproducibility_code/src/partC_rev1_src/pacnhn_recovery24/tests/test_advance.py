"""Tiny metadata fixtures and fake processes; no scientific imports or launches."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from partC_rev1_src.pacnhn_recovery24 import advance as a


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value));return a.binding(path)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    root=tmp_path/'new';root.mkdir();(root/'execution').mkdir();(root/'controller').mkdir();source=tmp_path/'stage.py';source.write_text('frozen')
    monkeypatch.setattr(a,'source_paths',lambda action:[source])
    cfg=dict(phase_id='PACNHN_RECOVERY24',run_id='toy',output_root=str(root),repository=str(tmp_path),hard_deadline_epoch=100000.,
             config_revision=1,recovery={'training_overlays':{},'xgb_region_quotas':{}},ablation={'deadline_epoch':30000.},
             action_deadlines={'training':30000.,'portfolio':50000.,'holdout':50000.,'assignments':80000.,'publication':90000.})
    config=save(root/'campaign_v1.json',cfg);save(root/'ACTIVE_CONFIG.json',config)
    frozen_rows=[dict(record={'suite_id':sid,'region_index':i}) for sid in ['s04','s02','s05','s06'] for i in range(32)]+[dict(record={'suite_id':'s03','region_index':1})]
    save(root/'holdout/INPUT_FREEZE.json',dict(status='FROZEN_BEFORE_FITTING',created_epoch=20.,requested_inputs=10,rows=frozen_rows))
    command=['python','-B','-m','partC_rev1_src.pacnhn_recovery24.execute','--config',config['path'],'--action','primary']
    admission=dict(schema=a.SCHEMA,phase_id=cfg['phase_id'],run_id='toy',output_root=str(root),initial_config=config,
        primary_process=dict(pid=777,create_time=10.,command=command,cwd=str(tmp_path)),python_path='python',repository=str(tmp_path),
        output_relative='controller/successors',hard_deadline_epoch=100000.,stages=list(a.STAGES),
        source_bindings={str(source):a.binding(source)['sha256']},root_honors_exclusive_controller_lock=True,
        helper_sha256=a.binding(a.__file__)['sha256'])
    def wrapper(action, conf, started, status='RETURNED_WITH_EXPLICIT_OUTCOMES'):
        folder=root/'execution'/f'{action}_{int(started)}';folder.mkdir()
        intent=dict(phase_id=cfg['phase_id'],run_id='toy',step_id=action,config=conf,started_epoch=started,sources=admission['source_bindings'])
        save(folder/'INTENT.json',intent)
        save(folder/'RESULT.json',dict(action=action,config=conf,started_epoch=started,ended_epoch=started+1,
            status=status,error=None if status=='RETURNED_WITH_EXPLICIT_OUTCOMES' else {'type':'ValueError'},
            secondary_errors=[],sources=admission['source_bindings']))
        return folder
    folder=wrapper('primary',config,80.)
    admission['primary_intent']=a.binding(folder/'INTENT.json')
    save(root/'training/PRIMARY_RESULT.json',dict(status='PRIMARY_TIME_LIMIT',TEST_used=False,config=config))
    def terminal(action,conf):
        if action=='other':
            old=a.read(conf['path']); revised=copy.deepcopy(old);revised.update(config_revision=2,supersedes_config_sha256=conf['sha256'],revision_reason='bounded overlays',revision_created_epoch=100.)
            revised['recovery']['training_overlays']={'s04:000001':{'path':'small metadata','sha256':'a'*64}}
            newconf=save(root/'campaign_v2.json',revised)
            save(root/'config/REVISION_002.json',dict(parent=conf,new=newconf))
            save(root/'ACTIVE_CONFIG.json',newconf)
            save(root/'training/OTHER_SUITES_RESULT.json',dict(status='ALL_FOUR_SUITE_ALLOCATIONS_ACCOUNTED',TEST_used=False,config=newconf,suites=[{'suite_id':s,'results':[{'job_key':f'{s}:{i:06d}','status':'UNATTEMPTED_TIME_LIMIT'} for i in range(32)]} for s in ['s04','s02','s05','s06']]))
        if action=='ablation':
            plan=save(root/'ablation/PLAN.json',dict(tasks=[{'task_id':str(i)} for i in range(160)]))
            save(root/'ablation/FIT_FREEZE.json',dict(status='ALL_160_DISPOSITIONS_FROZEN',total=160,TEST_used=False,
                plan_sha256=plan['sha256'],frozen_epoch=105.,outcomes=[dict(task={'task_id':str(i)},status='UNATTEMPTED_TIME_LIMIT') for i in range(160)]))
        if action=='selection':
            holdout=a.binding(root/'holdout/INPUT_FREEZE.json')
            portfolio=save(root/'portfolios/PORTFOLIOS.json',{'status':'FROZEN'})
            records=save(root/'training/NORMALIZED_RECORDS.json',[])
            disposition=save(root/'selection/one/REGION_DISPOSITIONS.json',{'metadata':True})
            freeze=save(root/'training/MODEL_SELECTION_FREEZE.json',dict(status='FROZEN',created_epoch=110.,test_targets_opened=False,
                config_path=conf['path'],config_sha256=conf['sha256'],holdout_input_freeze=holdout,portfolio=portfolio,
                model_records=records,ablation_fit_freeze=a.binding(root/'ablation/FIT_FREEZE.json'),region_dispositions=disposition))
            save(root/'selection/one/RESULT.json',dict(status='FROZEN_WITH_EXPLICIT_FALLBACK_PORTFOLIOS',selection=freeze))
        if action=='holdout':
            digest=a.binding(root/'training/MODEL_SELECTION_FREEZE.json')['sha256']
            save(root/'holdout/TEST_OPENED.json',dict(epoch=115.,selection_freeze_sha256=digest))
            save(root/'holdout/RESULT.json',dict(status='HOLDOUT_STAGE_FINISHED',selection_freeze_sha256=digest,targets_used_for_selection=False,requested_inputs=10,results=[{'region_key':f"{x['record']['suite_id']}:{x['record']['region_index']:06d}",'status':'PARTIAL_LABEL_COVERAGE'} for x in frozen_rows]))
        if action=='evaluation':
            save(root/'evaluation/regional_v1/RESULT.json',dict(status='TIME_LIMIT_COMPLETE_DISPOSITION_ACCOUNTING',regions=10442,row_count=10442*3*4*3))
            save(root/'ablation/evaluation/RESULT.json',dict(status='COMPLETE_WITH_EXPLICIT_LABEL_COVERAGE',fit_cases=160,TEST_used_for_selection=False))
        if action=='assignments':
            save(root/'ASSIGNMENT_STAGE_RESULT.json',dict(status='COMPLETE_WITH_RECORDED_OUTCOMES',rows=46,reporting_rows=48,all_converged=False))
            save(root/'ASSIGNMENT_RESULTS.json',[{'case':{'case_id':str(i)},'status':'TIME_LIMIT'} for i in range(46)])
            save(root/'ASSIGNMENT_MATRIX.json',[{'case_id':str(i)} for i in range(46)])
            save(root/'REPORTING_RESULTS.json',[{'case':{'case_id':i},'status':'TIME_LIMIT'} for i in list(map(str,range(46)))+['P_REFERENCE_open','P_REFERENCE_i95']])
        if action=='metrics':save(root/'report_data/MANIFEST.json',dict(status='VERIFIED_REPORT_DATA'))
    class Child:
        def __init__(self,pid,action):self.pid=pid;self.action=action;self.done=False;self.code=0
        def poll(self):return self.code if self.done else None
        def wait(self):assert self.done;return self.code
    class Runtime:
        def __init__(self):
            self.clock=100.;self.child=None;self.launches=[];self.primary_waits=2;self.killed=[]
            self.mutate=None;self.fail_action=None;self.expire_action=None;self.overlap=False;self.cleanup_success=True
        def now(self):return self.clock
        def describe(self,pid):
            if pid==os.getpid():return dict(pid=pid,create_time=5.,command=['controller'])
            if pid==777:return admission['primary_process'] if self.primary_waits>0 else None
            if self.child and self.child.pid==pid and not self.child.done:
                return dict(pid=pid,create_time=50.+pid,command=self.launches[-1])
            return None
        def sleep(self,seconds):
            self.clock+=seconds
            if self.primary_waits:self.primary_waits-=1;return
            if self.child and not self.child.done:
                child=self.child
                if self.expire_action==child.action:self.clock=100001.;return
                command=self.launches[-1];conf=a.binding(command[command.index('--config')+1])
                wrapper(child.action,conf,self.clock-2)
                terminal(child.action,conf)
                if self.mutate:self.mutate(child.action)
                child.done=True;child.code=3 if self.fail_action==child.action else 0
        def stages(self,root):return [{'pid':999}] if self.overlap else []
        def children(self,pid):return []
        def guard_metadata(self,ctx):pass
        def launch(self,command,cwd,log):
            assert self.primary_waits==0 and (self.child is None or self.child.done)
            self.launches.append(command);self.child=Child(800+len(self.launches),command[-1]);return self.child
        def terminate_owned(self,process,identity):
            assert identity['pid']==process.pid
            self.killed.append(process.pid);process.done=True;process.code=-1;return self.cleanup_success
    rt=Runtime()
    class Context:
        def __init__(self,path,step_id):self.c=a.read(path);self.action=step_id
        def deadline(self):return self.c['action_deadlines'][self.action]
        def guard(self,deadline=None):
            if rt.now()>=min(self.deadline(),deadline or self.deadline()):raise TimeoutError('expired original action')
    return SimpleNamespace(root=root,source=source,admission=admission,config=config,cfg=cfg,runtime=rt,Context=Context,wrapper=wrapper,terminal=terminal)


def run(f):return a.run(f.admission,runtime=f.runtime,context_factory=f.Context)


def test_all_sequential_successors_and_explicit_partial_outcomes(fixture):
    f=fixture;r=run(f)
    assert r['status']=='ALL_SUCCESSOR_STAGES_ACCOUNTED' and len(r['steps'])==8
    assert [cmd[-1] for cmd in f.runtime.launches]==list(a.STAGES)
    assert all(cmd[1:4]==['-B','-m','partC_rev1_src.pacnhn_recovery24.execute'] for cmd in f.runtime.launches)
    assert f.runtime.launches[0][-3].endswith('campaign_v1.json')
    assert all(cmd[-3].endswith('campaign_v2.json') for cmd in f.runtime.launches[1:])
    assert not r['scientific_success_claim'] and not (f.root/'ADVANCE_CONTROLLER_LOCK.json').exists()
    assert (f.root/'controller/successors/CONTROLLER_LOCK_CLOSED.json').exists()


def test_duplicate_controller_lock_never_overwritten_or_launched(fixture):
    f=fixture;path=f.root/'ADVANCE_CONTROLLER_LOCK.json';save(path,{'owner':'root'})
    with pytest.raises(FileExistsError):run(f)
    assert a.read(path)=={'owner':'root'} and not f.runtime.launches


def test_manual_stage_already_running_blocks_successors(fixture):
    f=fixture;f.runtime.overlap=True;r=run(f)
    assert r['status']=='STOPPED_PRESERVED_EVIDENCE' and not f.runtime.launches


@pytest.mark.parametrize('changed',['create_time','command'])
def test_pid_reuse_or_command_mismatch_never_admitted(fixture,changed):
    f=fixture;expected=copy.deepcopy(f.admission['primary_process']);expected[changed]=999 if changed=='create_time' else ['other']
    with pytest.raises(RuntimeError):a.identity_running(f.runtime,expected)


def test_nonzero_child_exit_stops_without_launching_next_stage(fixture):
    f=fixture;f.runtime.fail_action='ablation';r=run(f)
    assert r['status']=='STOPPED_PRESERVED_EVIDENCE'
    assert [c[-1] for c in f.runtime.launches]==['other','ablation']
    assert r['owned_child_closed']


def test_expired_child_terminates_only_owned_identity_and_stops(fixture):
    f=fixture;f.runtime.expire_action='other';r=run(f)
    assert r['error']['type']=='TimeoutError' and len(f.runtime.killed)==1
    assert len(f.runtime.launches)==1 and r['owned_child_closed']


def test_incomplete_owned_cleanup_keeps_exclusive_lock(fixture):
    f=fixture;f.runtime.expire_action='other';f.runtime.cleanup_success=False;r=run(f)
    assert not r['owned_child_closed'] and (f.root/'ADVANCE_CONTROLLER_LOCK.json').exists()


def test_wrapper_failure_receipt_stops_even_with_zero_exit(fixture):
    f=fixture
    def mutate(action):
        if action=='other':
            p=next((f.root/'execution').glob('other_*/RESULT.json'));row=a.read(p);row['error']={'message':'failure'};save(p,row)
    f.runtime.mutate=mutate;r=run(f)
    assert r['error']['type']=='RuntimeError' and len(f.runtime.launches)==1


def test_post_other_configuration_change_stops(fixture):
    f=fixture
    def mutate(action):
        if action=='ablation':save(f.root/'ACTIVE_CONFIG.json',f.config)
    f.runtime.mutate=mutate;r=run(f)
    assert 'Configuration changed' in r['error']['message'] and len(f.runtime.launches)==2


def test_changed_completed_prerequisite_is_not_silently_reused(fixture):
    f=fixture
    def mutate(action):
        if action=='other':save(f.root/'training/PRIMARY_RESULT.json',{'status':'rewritten'})
    f.runtime.mutate=mutate;r=run(f)
    assert r['status']=='STOPPED_PRESERVED_EVIDENCE' and len(f.runtime.launches)==1


def test_changed_source_stops_before_successor(fixture):
    f=fixture;f.source.write_text('changed');r=run(f)
    assert not f.runtime.launches and r['error']['type']=='ValueError'


@pytest.mark.parametrize('change',['deadline','input','foreign','unlinked'])
def test_active_revision_rejects_unauthorized_lineage(fixture,change):
    f=fixture;f.terminal('other',f.config);record=a.read(f.root/'campaign_v2.json')
    if change=='deadline':record['hard_deadline_epoch']+=1
    if change=='input':record['input_versions']={'new':True}
    if change in {'deadline','input'}:
        conf=save(f.root/'campaign_v2.json',record);save(f.root/'ACTIVE_CONFIG.json',conf)
    elif change=='foreign':save(f.root/'ACTIVE_CONFIG.json',save(f.root.parent/'foreign.json',record))
    else:save(f.root/'config/REVISION_002.json',{'new':f.config,'parent':f.config})
    with pytest.raises(ValueError):a.active_config(f.root,f.config)


@pytest.mark.parametrize('change',['count','unattempted','duplicate'])
def test_ablation_requires_all_unique_terminal_dispositions(fixture,change):
    f=fixture;f.terminal('ablation',f.config);p=f.root/'ablation/FIT_FREEZE.json';r=a.read(p)
    if change=='count':r['outcomes'].pop()
    if change=='unattempted':r['outcomes'][0]['status']='UNATTEMPTED'
    if change=='duplicate':r['outcomes'][0]['task']=r['outcomes'][1]['task']
    save(p,r)
    with pytest.raises(ValueError):a.terminal_metadata(f.root,'ablation',f.config)


def test_holdout_rejects_wrong_or_early_opening(fixture):
    f=fixture;f.terminal('ablation',f.config);f.terminal('selection',f.config);f.terminal('holdout',f.config)
    p=f.root/'holdout/TEST_OPENED.json';r=a.read(p);r['epoch']=1;save(p,r)
    with pytest.raises(ValueError):a.terminal_metadata(f.root,'holdout',f.config)


def test_missing_terminal_receipt_blocks_next_stage(fixture):
    f=fixture
    def mutate(action):
        if action=='selection':(f.root/'training/MODEL_SELECTION_FREEZE.json').unlink()
    f.runtime.mutate=mutate;r=run(f)
    assert r['status']=='STOPPED_PRESERVED_EVIDENCE' and [c[-1] for c in f.runtime.launches]==['other','ablation','selection']


def test_platform_launch_is_hidden_and_uses_argument_list(monkeypatch):
    calls=[];monkeypatch.setattr(a.subprocess,'Popen',lambda *args,**kw:calls.append((args,kw)))
    rt=a.Runtime();rt.launch(['python','-B','-m','module'],'workspace',None)
    assert calls[0][0][0]==['python','-B','-m','module']
    assert calls[0][1]['creationflags']==(a.subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)


@pytest.mark.parametrize('value',[True,1.0])
def test_config_type_changes_are_not_equal(value):
    assert a.invariant_config({'science':1})!=a.invariant_config({'science':value})


def test_current_admission_pins_helper_repository_interpreter_and_sources(fixture):
    f=fixture;ad=copy.deepcopy(f.admission);ad['source_bindings'][str(Path(a.__file__).resolve())]=a.binding(a.__file__)['sha256']
    p=f.root/'admission.json';b=save(p,ad)
    assert a.validate_admission(p,b['sha256'])['run_id']=='toy'
    ad['repository']=str(f.root);b=save(p,ad)
    with pytest.raises(ValueError,match='Repository'):a.validate_admission(p,b['sha256'])
    ad['repository']=str(f.root.parent);ad['python_path']='other_python';b=save(p,ad)
    with pytest.raises(ValueError,match='interpreter'):a.validate_admission(p,b['sha256'])


@pytest.fixture
def timeout_receipt():
    from partC_rev1_src.pacnhn_recovery24 import execute
    source=Path(execute.__file__).resolve();tree=a.ast.parse(source.read_text())
    fn=next(n for n in tree.body if isinstance(n,a.ast.FunctionDef) and n.name=='execute_stage')
    lines=[n.lineno for n in a.ast.walk(fn) if isinstance(n,a.ast.Call) and isinstance(n.func,a.ast.Attribute)
           and isinstance(n.func.value,a.ast.Name) and n.func.value.id=='ctx' and n.func.attr=='guard']
    record=dict(status='FAILED_PRESERVED_OUTPUTS',action_at_return='holdout',
        error=dict(type='TimeoutError',message='Approved recovery action/campaign deadline reached',
            traceback=f'  File "{source}", line {max(lines)}, in execute_stage\n    ctx.guard(deadline=limits["stage_deadline_epoch"])\n'),
        resource_samples=dict(samples=5,monitor_joined=True,child_access_denied=0,
            sampled_rss_limit_exceeded=False,sampled_available_floor_breached=False),secondary_errors=[],
        limits={'stage_deadline_epoch':1000.},ended_epoch=1000.5,result_status='HOLDOUT_STAGE_FINISHED')
    return record,dict(source_bindings={str(source):a.binding(source)['sha256']},hard_deadline_epoch=2000.),source,lines


def test_exact_postdispatch_timeout_is_narrowly_qualified(timeout_receipt):
    row,ad,_,_=timeout_receipt
    a.validate_postdispatch_timeout(row,'holdout',ad)
    assert row['status']=='FAILED_PRESERVED_OUTPUTS'  # Original failure is never rewritten.


@pytest.mark.parametrize('change',['wrong_line','other_error','secondary','rss','free','missing_samples','missing_return','too_late','hard_end','changed_source','action'])
def test_unproven_timeout_or_failed_resource_cannot_be_promoted(timeout_receipt,change):
    row,ad,source,lines=timeout_receipt
    if change=='wrong_line':row['error']['traceback']=row['error']['traceback'].replace(f'line {max(lines)}',f'line {min(lines)}')
    if change=='other_error':row['error']['type']='ValueError'
    if change=='secondary':row['secondary_errors']=[{'type':'ValueError'}]
    if change=='rss':row['resource_samples']['sampled_rss_limit_exceeded']=True
    if change=='free':row['resource_samples']['sampled_available_floor_breached']=True
    if change=='missing_samples':row['resource_samples']['samples']=0
    if change=='missing_return':row['result_status']=None
    if change=='too_late':row['ended_epoch']=1061.
    if change=='hard_end':ad['hard_deadline_epoch']=1000.
    if change=='changed_source':ad['source_bindings'][str(source)]='a'*64
    if change=='action':row['action_at_return']='training'
    with pytest.raises(RuntimeError):a.validate_postdispatch_timeout(row,'holdout',ad)


def test_statusless_ablation_return_still_requires_exact_postguard_proof(timeout_receipt):
    row,ad,_,_=timeout_receipt;row['action_at_return']='training';row['result_status']=None
    a.validate_postdispatch_timeout(row,'ablation',ad)


def test_holdout_missing_region_cannot_be_complete_accounting(fixture):
    f=fixture
    for action in ('ablation','selection','holdout'):f.terminal(action,f.config)
    p=f.root/'holdout/RESULT.json';r=a.read(p);r['results']=[];save(p,r)
    with pytest.raises(ValueError,match='Unaccounted'):a.terminal_metadata(f.root,'holdout',f.config)


def test_assignment_duplicate_cases_rejected_even_if_count_fields_claim_complete(fixture):
    f=fixture
    for action in ('ablation','selection','holdout','evaluation','assignments'):f.terminal(action,f.config)
    p=f.root/'ASSIGNMENT_RESULTS.json';r=a.read(p);r[0]=r[1];save(p,r)
    with pytest.raises(ValueError,match='duplicated'):a.terminal_metadata(f.root,'assignments',f.config)


def test_cleanup_requires_matching_actual_return_status_and_stage_chronology(fixture):
    f=fixture;terminal=f.root/'holdout/RESULT.json';save(terminal,{'status':'HOLDOUT_STAGE_FINISHED'})
    os.utime(terminal,(1000.,1000.));path=f.root/'wrapper.json'
    save(path,dict(ended_epoch=1001.,result_status='HOLDOUT_STAGE_FINISHED'))
    wrapper=dict(a.binding(path),administrative_disposition='VERIFIED_TERMINAL_ACCOUNTING_AFTER_WORK_DEADLINE')
    a.check_cleanup_chronology(f.root,'holdout',wrapper,999.)
    save(path,dict(ended_epoch=1001.,result_status='ANOTHER_TRUTHY_STATUS'))
    with pytest.raises(ValueError,match='disposition'):a.check_cleanup_chronology(f.root,'holdout',wrapper,999.)
    save(path,dict(ended_epoch=1001.,result_status='HOLDOUT_STAGE_FINISHED'));os.utime(terminal,(100.,100.))
    with pytest.raises(ValueError,match='chronology'):a.check_cleanup_chronology(f.root,'holdout',wrapper,999.)


def test_statusless_ablation_cleanup_needs_new_freeze_and_exact_none_schema(fixture):
    f=fixture;f.terminal('ablation',f.config);terminal=f.root/'ablation/FIT_FREEZE.json'
    r=a.read(terminal);r['frozen_epoch']=1000.;save(terminal,r);os.utime(terminal,(1000.,1000.))
    path=f.root/'wrapper.json';save(path,dict(ended_epoch=1001.,result_status=None,nested_result_statuses={}))
    wrapper=dict(a.binding(path),administrative_disposition='VERIFIED_TERMINAL_ACCOUNTING_AFTER_WORK_DEADLINE')
    a.terminal_metadata(f.root,'ablation',f.config);a.check_cleanup_chronology(f.root,'ablation',wrapper,999.)
    r['frozen_epoch']=990.;save(terminal,r);os.utime(terminal,(1000.,1000.))
    with pytest.raises(ValueError,match='predates'):a.check_cleanup_chronology(f.root,'ablation',wrapper,999.)


def test_live_observed_primary_descendant_blocks_launch(fixture):
    f=fixture;original=f.runtime.describe
    orphan=dict(pid=987,create_time=9.,command=['owned_worker'])
    f.runtime.children=lambda pid:[orphan]
    f.runtime.describe=lambda pid:orphan if pid==987 else original(pid)
    r=run(f)
    assert 'descendant' in r['error']['message'] and not f.runtime.launches


def test_live_successor_descendant_blocks_next_stage_and_keeps_lock(fixture):
    f=fixture;original=f.runtime.describe;orphan=dict(pid=987,create_time=9.,command=['owned_worker'])
    f.runtime.children=lambda pid:[] if pid==777 else [orphan]
    f.runtime.describe=lambda pid:orphan if pid==987 else original(pid)
    r=run(f)
    assert not r['owned_child_closed'] and len(f.runtime.launches)==1
    assert (f.root/'ADVANCE_CONTROLLER_LOCK.json').exists()


def test_other_suite_missing_disposition_stops(fixture):
    f=fixture;f.terminal('other',f.config);p=f.root/'training/OTHER_SUITES_RESULT.json';r=a.read(p);r['suites'][0]['results'].pop();save(p,r)
    with pytest.raises(ValueError,match='Unaccounted'):a.terminal_metadata(f.root,'other',r['config'])


def test_wrong_assignment_matrix_and_reference_ids_rejected(fixture):
    f=fixture
    for action in ('ablation','selection','holdout','evaluation','assignments'):f.terminal(action,f.config)
    p=f.root/'ASSIGNMENT_MATRIX.json';r=a.read(p);r[0]['case_id']='foreign';save(p,r)
    with pytest.raises(ValueError,match='matrix identities'):a.terminal_metadata(f.root,'assignments',f.config)
