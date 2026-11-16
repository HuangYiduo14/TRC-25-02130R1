"""One primary launcher: bounded J1 followed by automatic fair data/training cycles."""
import argparse,os,time,traceback,subprocess,multiprocessing,collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import pandas as pd
from .common import root,rd,read_json,write_json,append_json,sha,admit,left,elapsed,native_threads,lease,is_e2
from .source import Sources
from .resources import Monitor,worker_setup
from .data import revision
from .training import PRIMARY,ESCALATION,LEGACY,CONTROL
from .policy import active_budget,failed_fraction_below_gate

SOURCE=None
def initialize_worker(c):
    global SOURCE
    worker_setup(c,'campaign');SOURCE=Sources(c)

def worker(c,job):
    started=time.perf_counter();rid=job['region_id']
    try:
        with lease(c,rid,job['action']):
            if job['action']=='TRAIN':
                from .training import train_slice
                result=train_slice(c,SOURCE,rid,job['cycle'],first=job.get('first',False),kind=job.get('kind',PRIMARY),
                    max_updates=job.get('max_updates'),old_only=job.get('old_only',False))
            elif job['action']=='FREEZE':
                from .training import freeze_cycle
                result=freeze_cycle(c,SOURCE,rid,job['cycle'],old_only=job.get('old_only',False),final=job.get('final',False))
            elif job['action']=='EVALUATE':
                if is_e2(c):from .e2_evaluation import evaluate_region
                else:from .final_evaluation import evaluate_region
                result=evaluate_region(c,SOURCE,rid)
            elif job['action']=='GENERATE':
                from .generation import generate_job
                result=generate_job(c,SOURCE,job)
            else:raise ValueError('No implicit worker action')
        SOURCE.flush('campaign')
        return dict(ok=True,job=job,result=result,wall_seconds=time.perf_counter()-started)
    except Exception as e:
        trace=traceback.format_exc()
        error=dict(ok=False,job=job,error_type=type(e).__name__,error=repr(e),traceback=trace,wall_seconds=time.perf_counter()-started,unix=time.time(),
            shared_integrity_failure=is_e2(c) and ('shared_io.py' in trace or 'e2_forecast.py' in trace),
            io_evidence={k:getattr(e,k,None) for k in ['winerror','io_attempts','pending_path','pending_sha256','serialized_payload_sha256','replace_retry_elapsed_seconds']})
        write_json(rd(c,rid)/'failures'/f'{time.time_ns()}.json',error);SOURCE.flush('failure');return error

class Runner:
    def __init__(self,c,workers=2):
        self.c=c;self.count=workers;self.pool=ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker,initargs=(c,))
        self.running={};self.regions=set();self.done=0;self.times=collections.defaultdict(float);self.failures=[]
        self.registry=pd.read_parquet(root(c)/'control/source_region_registry.parquet').set_index('region_id')
    def available(self,rid):
        if rid in self.regions or len(self.running)>=self.count:return False
        large=int(self.registry.loc[rid,'m_active'])>=2048
        return not (self.running and (large or any(int(self.registry.loc[r,'m_active'])>=2048 for r in self.regions)))
    def submit(self,job):
        rid=job['region_id'];assert self.available(rid)
        f=self.pool.submit(worker,self.c,job);self.running[f]=job;self.regions.add(rid)
    def collect(self,timeout=1.):
        if not self.running:return []
        finished,_=wait(self.running,timeout=timeout,return_when=FIRST_COMPLETED);results=[]
        for f in finished:
            job=self.running.pop(f);self.regions.remove(job['region_id'])
            r=f.result();self.done+=1;self.times[job['action']]+=r['wall_seconds']
            append_json(root(self.c)/'control/job_events.jsonl',dict(job=job,ok=r['ok'],wall_seconds=r['wall_seconds'],unix=time.time(),
                summary={k:r.get('result',{}).get(k) for k in ['status','steps','cycle_epochs','published','AB','C','quarantined','last_slice_updates']}))
            if not r['ok']:
                self.failures.append(r)
                if r.get('shared_integrity_failure') or any(s in r.get('error','') for s in ['Source binding changed','Source changed during read','Unbound source','Cross-source/extension input collision','Pack manifest chain differs']):
                    write_json(root(self.c)/'control/STOP_SCIENCE.json',dict(reason='SHARED_CURRENT_INTEGRITY_FAILURE',event=r))
            results.append(r)
        return results
    def batch(self,jobs,*,time_limit=None):
        todo=collections.deque(jobs);results=[]
        while todo or self.running:
            if time_limit is not None and time.time()>=time_limit:
                todo.clear()
            if todo and todo[0]['action']=='TRAIN' and not admit(self.c,3):todo.clear()
            for _ in range(len(todo)):
                if len(self.running)>=self.count:break
                job=todo.popleft()
                if self.available(job['region_id']):self.submit(job)
                else:todo.append(job)
            got=self.collect();results.extend(got)
            if not todo and not self.running:break
        return results
    def close(self):self.pool.shutdown(wait=True,cancel_futures=False)

def fair_regions(c):
    from partC_rev1_src.phase7.data import fair_queue
    reg=pd.read_parquet(root(c)/'control/source_region_registry.parquet');return fair_queue(reg.loc[reg.m_active.gt(0)])

def status(c,rid,kind=PRIMARY):
    p=rd(c,rid)/'training'/kind/'status.json';return read_json(p) if p.exists() else None

def expansion_ready(role_counts,workers,now,next_check):
    """Drain admitted jobs before expansion; never wait for an accidental idle instant."""
    return workers<4 and role_counts['GENERATE']>=100 and role_counts['TRAIN']>=100 and now>=next_check

def cleanup_stale_leases(c):
    import psutil
    for p in (root(c)/'regions').glob('*/LEASE.json'):
        v=read_json(p);alive=False
        try:alive=abs(psutil.Process(v['pid']).create_time()-v.get('process_created_unix',0))<.01
        except psutil.NoSuchProcess:pass
        if alive:raise RuntimeError('An existing patch worker still owns '+str(p))
        write_json(root(c)/'control/recovered_leases'/f'{p.parent.name}_{time.time_ns()}.json',dict(original_path=str(p),lease=v,recovered_unix=time.time()))
        p.unlink() # Proven dead process's ephemeral lease only.

def recover_pool(c,runner,error):
    import psutil
    for child in psutil.Process().children(recursive=False):
        try:
            if any('multiprocessing.spawn' in a for a in child.cmdline()):child.terminate()
        except psutil.Error:pass
    runner.pool.shutdown(wait=False,cancel_futures=True)
    time.sleep(1.)
    cleanup_stale_leases(c)
    history=root(c)/'control/ENGINEERING_REPAIRS.json';old=read_json(history) if history.exists() else dict(epochs=[])
    epoch=len(old['epochs'])+1
    if epoch>c['execution']['engineering_repairs_max']:
        write_json(root(c)/'control/STOP_SCIENCE.json',dict(reason='ENGINEERING_REPAIR_BUDGET_EXHAUSTED',unix=time.time(),
            attempted_additional_repair=False,error=repr(error),completed_repair_epochs=len(old['epochs']),original_clock_preserved=True))
        progress=root(c)/'control/J23_PROGRESS.json';cycle=read_json(progress)['cycle'] if progress.exists() else 0
        write_json(root(c)/'control/SCIENCE_COMPLETE.json',dict(status='TECHNICAL_STOP_PRESERVE_PARTIAL',cycle=cycle,elapsed_seconds=elapsed(c),
            next='J4 diagnostic freeze, reopen and one handoff; no additional repair epoch authorized'))
        return False
    record=dict(epoch=epoch,unix=time.time(),cause='RECOVERABLE_NATIVE_WORKER_PROCESS_BOUNDARY',error=repr(error),
        recovery='Terminate owned surviving workers, preserve scientific bytes and restore exact saved Adam/RNG/shuffle in fresh single-thread spawned processes',
        original_clock_preserved=True,code_or_input_change=False,interrupted_jobs=list(runner.running.values()))
    old['epochs'].append(record);write_json(history,old)
    test=subprocess.run([c['execution']['python_executable'],'-m','pytest','-q','tests/test_phase67_patch.py',
        '--junitxml='+str(root(c)/'validation'/f'recovery_{epoch:02d}_tests.xml')],capture_output=True,text=True)
    from .common import atomic
    atomic(root(c)/'validation'/f'recovery_{epoch:02d}_tests.log',(test.stdout+test.stderr).encode())
    if test.returncode:raise RuntimeError('Native recovery regression checks failed')
    return True

def J1(c,runner,regions):
    done=root(c)/'control/J1_COMPLETE.json'
    if done.exists():return read_json(done)
    startfile=root(c)/'control/J1_STARTED.json'
    if not startfile.exists():write_json(startfile,dict(unix=time.time(),wall_cap_seconds=7200,training_cutoff_seconds=6900,comparison_freeze_reserve_seconds=300))
    started=read_json(startfile)['unix'];cutoff=started+6900
    first=[dict(action='TRAIN',region_id=r,cycle=0,first=True,old_only=True) for r in regions if status(c,r) is None]
    runner.batch(first,time_limit=cutoff)
    plan=read_json(root(c)/'control/audit_plan.json')
    while time.time()<cutoff and admit(c,120):
        jobs=[]
        for r in regions:
            s=status(c,r)
            if s is not None and s['status']=='TRAINING':jobs.append(dict(action='TRAIN',region_id=r,cycle=0,old_only=True))
        # Legacy continuation is distinct and matched to the primary's available budget.
        for r in plan['legacy_panel']:
            s=status(c,r);old=status(c,r,LEGACY);delta=(s['steps'] if s else 0)-(old['steps'] if old else 0)
            if delta>0 and (old is None or old['status']=='TRAINING'):
                jobs.append(dict(action='TRAIN',region_id=r,cycle=0,kind=LEGACY,old_only=True,max_updates=min(200,delta),first=old is None))
        if not jobs:break
        runner.batch(jobs,time_limit=cutoff)
        write_json(root(c)/'control/J1_PROGRESS.json',dict(elapsed=time.time()-started,first_attempted=sum(status(c,r) is not None for r in regions),
            jobs_completed=runner.done,failures=len(runner.failures),timing=dict(runner.times)))
    runner.batch([dict(action='FREEZE',region_id=r,cycle=0,old_only=True) for r in regions])
    stats=[status(c,r) for r in regions];present=[s for s in stats if s]
    result=dict(status='EXISTING_DATA_ONLY_COMPARISON_FROZEN',started_unix=started,ended_unix=time.time(),wall_seconds=time.time()-started,
        attempted=len(present),unattempted=[r for r,s in zip(regions,stats) if s is None],new_TRAIN_labels=0,
        complete_cycle=sum(s['status'] in ['PLATEAU_NEXT_DATA','CYCLE_CAP'] for s in present),partial_cycle=sum(s['status'] not in ['PLATEAU_NEXT_DATA','CYCLE_CAP'] for s in present),
        total_updates=sum(s['steps'] for s in present),failure_count=len(runner.failures),next='J2/J3 automatic incremental data and training')
    from .review_artifacts import existing_data_comparison
    result['comparison_report']=existing_data_comparison(c,regions)
    result['two_hour_cap_exceeded']=result['wall_seconds']>7200
    write_json(done,result);return result

def generation_schedule(c,regions,cycle):
    planpath=root(c)/'control/cycles'/f'cycle_{cycle:04d}_generation_plan.json'
    if planpath.exists():return read_json(planpath)['jobs']
    reg=pd.read_parquet(root(c)/'control/source_region_registry.parquet').set_index('region_id');targets={}
    audit_counts={r['region_id']:r['counts'] for r in read_json(root(c)/'validation/AUDIT2_INPUTS_FROZEN.json')['region_manifests']} if is_e2(c) else {}
    for r in regions:
        if int(reg.loc[r,'m_active'])<=0:raise ValueError('Inactive regions cannot enter the data schedule')
        p=rd(c,r)/'training/cycles'/f'cycle_{cycle-1:04d}'/'manifest.json'
        prev=read_json(p) if p.exists() else {};metrics=prev.get('metrics',{}).get(PRIMARY,{})
        vals=[v['MARE_equal_pairs'] for v in metrics.get('valid',{}).values() if v and v.get('MARE_equal_pairs') is not None]
        error=sum(vals)/len(vals) if vals else 0.
        negative=bool(metrics.get('train',{}).get('negative_cost_count',0)) or any(v.get('negative_cost_count',0) for v in metrics.get('valid',{}).values())
        factor=4 if error>.20 or negative else 2 if error>.05 else 1
        width=64 if ESCALATION in prev.get('files',{}) else 32
        targets[r]=dict(target=active_budget(int(reg.loc[r,'m_active']),factor,width),difficulty=factor,width=width,preceding_VALID_MARE=error)
    jobs=[]
    def add(r,role,pack,offset):
        name=f'c{cycle:04d}_{role}_p{pack:06d}_o{offset:03d}'
        jobs.append(dict(action='GENERATE',region_id=r,cycle=cycle,role=role,pack_index=pack,offset=offset,jobid=name))
    # Eight microbatch waves guarantee all-region opportunity before proportional extras.
    for wave in range(8):
        for r in regions:
            add(r,'TRAIN',0,wave*4)
            if cycle==1:
                if wave<2:add(r,'PATCH_VALID',0,wave*4)
                if is_e2(c):
                    if wave<2:
                        for audit_role in ['E2_AUDIT_OPERATING','E2_AUDIT_STRESS']:
                            if audit_counts[r].get(audit_role,0)>wave*4:add(r,audit_role,0,wave*4)
                elif wave<4:add(r,'PATCH_AUDIT_OPERATING' if wave<2 else 'PATCH_AUDIT_STRESS',0,(wave%2)*4)
    maxpacks=max(v['target']//32 for v in targets.values())
    for pack in range(1,maxpacks):
        for wave in range(8):
            for r in regions:
                if targets[r]['target']>pack*32:add(r,'TRAIN',pack,wave*4)
    write_json(planpath,dict(cycle=cycle,targets=targets,jobs=jobs,first_global_wave32=True,created_unix=time.time(),TEST_HIGH_AUDIT_errors_used=False))
    return jobs

def J23(c,runner,regions):
    progressfile=root(c)/'control/J23_PROGRESS.json';old=read_json(progressfile) if progressfile.exists() else {};cycle=old.get('cycle',1)
    plan=read_json(root(c)/'control/audit_plan.json');generation_failed=set(old.get('generation_failed_regions',[]));training_failed=set(old.get('training_failed_regions',[]))
    training_cursor=0;control_cursor=0;role_counts=collections.Counter();upgrade_done=runner.count>=4;promotion_check_after=0.
    generation_balance_override=False;last_progress=0.
    while admit(c,120):
        started=time.time();jobs=generation_schedule(c,regions,cycle)
        todo=collections.deque(j for j in jobs if j['region_id'] not in generation_failed and not (rd(c,j['region_id'])/'data_extension/jobs'/j['jobid']/'RESULT.json').exists())
        first_remaining={(j['region_id'],j['jobid']) for j in todo if j['pack_index']==0}
        cycle_start_steps={r:(status(c,r) or {}).get('steps',0) for r in regions}
        cycle_completed=False
        while admit(c,120) and (todo or runner.running or not cycle_completed):
            promotion_pending=not upgrade_done and expansion_ready(role_counts,runner.count,time.time(),promotion_check_after)
            generating=sum(j['action']=='GENERATE' for j in runner.running.values());training=sum(j['action']=='TRAIN' for j in runner.running.values())
            max_gen=runner.count//2;max_train=runner.count-max_gen
            # Measured conservation of at least four compute hours if generation dominates.
            generation_admit=admit(c,900) and not (root(c)/'control/STOP_GENERATION.json').exists() and (generation_balance_override or not (left(c)<14400 and runner.times['GENERATE']>runner.times['TRAIN']))
            if generation_admit and generating<max_gen and not cycle_completed and not promotion_pending:
                for _ in range(len(todo)):
                    if generating>=max_gen or len(runner.running)>=runner.count:break
                    job=todo.popleft();r=job['region_id']
                    if r in generation_failed:
                        first_remaining.discard((r,job['jobid']));continue
                    if job['pack_index']>0 and first_remaining:
                        todo.append(job)
                        # All unfinished first-wave jobs currently hold leases.
                        if len(first_remaining)<=sum(j['action']=='GENERATE' and j['pack_index']==0 for j in runner.running.values()):break
                        continue
                    if runner.available(r):runner.submit(job);generating+=1
                    else:todo.append(job)
            if not cycle_completed and not promotion_pending and (training<max_train or (not generation_admit and training<runner.count)):
                # Bounded same-initialization controls get an explicit share of training.
                if role_counts['TRAIN']%10==0:
                    for _ in range(8):
                        r=plan['matched_update_panel'][control_cursor%8];control_cursor+=1
                        if not runner.available(r):continue
                        main=status(c,r);control=status(c,r,CONTROL)
                        delta=(main['steps'] if main else 0)-(control['steps'] if control else 0)
                        if delta>0 and revision(c,r)['TRAIN_AB']>0:
                            runner.submit(dict(action='TRAIN',region_id=r,cycle=cycle,kind=CONTROL,max_updates=min(200,delta),first=control is None,old_only=True));training+=1;break
                for _ in range(len(regions)):
                    if len(runner.running)>=runner.count or (generation_admit and training>=max_train):break
                    r=regions[training_cursor%len(regions)];training_cursor+=1
                    if r in training_failed or not runner.available(r):continue
                    st=status(c,r);rev=revision(c,r)
                    if not st:job=dict(action='TRAIN',region_id=r,cycle=0,first=True,old_only=True)
                    elif st['cycle']<cycle or rev['TRAIN_AB']>st.get('new_TRAIN_AB',-1) or st['status']=='TRAINING':
                        # A changed immutable snapshot resumes weights/Adam and resets patience.
                        job=dict(action='TRAIN',region_id=r,cycle=cycle)
                    elif st['status'] in ['PLATEAU_NEXT_DATA','CYCLE_CAP']:
                        vm=st.get('last_metrics',{});score=vm.get('score')
                        es=status(c,r,ESCALATION)
                        if (score is None or score>.05) and es is None:
                            job=dict(action='TRAIN',region_id=r,cycle=cycle,kind=ESCALATION,first=True)
                        elif es and (es['cycle']<cycle or rev['TRAIN_AB']>es.get('new_TRAIN_AB',-1) or es['status']=='TRAINING'):
                            job=dict(action='TRAIN',region_id=r,cycle=cycle,kind=ESCALATION)
                        else:continue
                    else:continue
                    runner.submit(job);training+=1
            got=runner.collect(timeout=1.)
            for event in got:
                job=event['job'];r=job['region_id'];role_counts[job['action']]+=1
                if not event['ok']:
                    (generation_failed if job['action']=='GENERATE' else training_failed).add(r)
                    if job['action']=='GENERATE':first_remaining={pair for pair in first_remaining if pair[0]!=r}
                    continue
                result=event['result']
                if job['action']=='GENERATE':first_remaining.discard((r,job['jobid']))
                if job['action']=='GENERATE' and result['quarantined']:
                    generation_failed.add(r)
                if job['action']=='TRAIN' and not is_e2(c):
                    sp=rd(c,r)/'training'/job.get('kind',PRIMARY)/'status.json';record=read_json(sp);record['new_TRAIN_AB']=revision(c,r)['TRAIN_AB'];write_json(sp,record)
            if got:
                # Streaming certification timings include read/hash/pricing, and scale
                # the remaining final reopen forecast without rewalking source trees.
                history=root(c)/'control/finalization_measurements.json'
                measure=read_json(history) if history.exists() else dict(added_labels=0,reopen_seconds=0.,completed_generation_jobs=0)
                for event in got:
                    if event['ok'] and event['job']['action']=='GENERATE':
                        r=event['result'];measure['added_labels']+=r.get('new_certified',r['published']) if is_e2(c) else r['published'];measure['reopen_seconds']+=r['fresh_reopen_seconds'];measure['completed_generation_jobs']+=1
                write_json(history,measure)
                # 30 min population evaluation/report/seal allowance plus measured
                # serial new-label reopen; reserve() applies the required 1.5 factor.
                if is_e2(c):
                    from .e2_forecast import publish
                    publish(c,max(c['initial_finalization_forecast_seconds'],1800.+measure['reopen_seconds']),measure)
                else:write_json(root(c)/'control/finalization_forecast.json',dict(remaining_seconds=1800.+measure['reopen_seconds'],basis=measure,updated_unix=time.time()))
            if not failed_fraction_below_gate(len(generation_failed)):
                write_json(root(c)/'control/STOP_SCIENCE.json',dict(reason='NEW_DATA_REGIONAL_FAILURE_THRESHOLD',regions=sorted(generation_failed),count=len(generation_failed)));break
            if got or time.time()-last_progress>=1.:
                write_json(progressfile,dict(status='CYCLING',cycle=cycle,pending_generation_jobs=len(todo),active_jobs=list(runner.running.values()),
                    generation_failed_regions=sorted(generation_failed),training_failed_regions=sorted(training_failed),role_jobs=dict(role_counts),wall_seconds_by_role=dict(runner.times),
                    elapsed_seconds=elapsed(c),updated_unix=time.time()))
                last_progress=time.time()
            if not todo:
                trained=[r for r in regions if r not in training_failed]
                cycle_completed=all((s:=status(c,r)) is not None and s['cycle']==cycle and (s['steps']-cycle_start_steps[r]>=1000 or s['status'] in ['PLATEAU_NEXT_DATA','CYCLE_CAP']) for r in trained)
                if cycle_completed and not runner.running:break
                if cycle_completed:
                    while runner.running:runner.collect()
                    break
            if promotion_pending and not runner.running:
                hb=read_json(root(c)/'control/heartbeat.json')
                if hb['peak_rss_bytes']<12*1024**3 and hb['free_bytes']>4*1024**3:
                    write_json(root(c)/'control/WORKER_ADMISSION_4.json',dict(measured_heartbeat=hb,completed_role_jobs=dict(role_counts),
                        measured_wall_seconds_by_role=dict(runner.times),new_total_workers=4,unix=time.time(),drained_before_expansion=True))
                    old_times=runner.times.copy();runner.close();runner=Runner(c,4);runner.times.update(old_times);upgrade_done=True
                else:
                    promotion_check_after=time.time()+300
                    append_json(root(c)/'control/worker_admission_deferrals.jsonl',dict(unix=time.time(),heartbeat=hb,reason='MEASURED_RSS_OR_DISK_ADMISSION_BOUND'))
                continue
            if not generation_admit and not runner.running:
                # Complete the bounded matched-update controls before declaring an idle state.
                for r in plan['matched_update_panel']:
                    main=status(c,r);control=status(c,r,CONTROL)
                    delta=(main['steps'] if main else 0)-(control['steps'] if control else 0)
                    if delta>0 and revision(c,r)['TRAIN_AB']>0 and r not in training_failed and runner.available(r):
                        runner.submit(dict(action='TRAIN',region_id=r,cycle=cycle,kind=CONTROL,max_updates=min(200,delta),first=control is None,old_only=True))
                if runner.running:continue
                if todo and not (root(c)/'control/STOP_GENERATION.json').exists() and admit(c,900):
                    # All currently eligible optimizers have reached their actual terminal.
                    # Fresh data can now enable useful training, so do not idle on the share target.
                    generation_balance_override=True
                    append_json(root(c)/'control/generation_balance_exceptions.jsonl',dict(unix=time.time(),cycle=cycle,
                        reason='NO_ADMISSIBLE_MEANINGFUL_TRAINING_WITHOUT_FRESH_DATA',pending_generation_jobs=len(todo),remaining_science_seconds=left(c)))
                elif todo and (root(c)/'control/STOP_GENERATION.json').exists():
                    write_json(root(c)/'control/STOP_SCIENCE.json',dict(reason='DATA_DISK_BOUND_AND_NO_REMAINING_MEANINGFUL_OPTIMIZER_UPDATE',
                        unix=time.time(),cycle=cycle,pending_generation_jobs=len(todo),generation_stop=read_json(root(c)/'control/STOP_GENERATION.json')))
                    break
                elif todo and not admit(c,900):
                    write_json(root(c)/'control/STOP_SCIENCE.json',dict(reason='INSUFFICIENT_BUDGET_FOR_FRESH_DATA_AFTER_ALL_ADMISSIBLE_TRAINING',
                        unix=time.time(),cycle=cycle,pending_generation_jobs=len(todo),remaining_science_seconds=left(c)))
                    break
        # Drain before writing one immutable cycle snapshot per region.
        while runner.running:runner.collect()
        if not admit(c,120):break
        runner.batch([dict(action='FREEZE',region_id=r,cycle=cycle) for r in regions])
        write_json(root(c)/'control/cycles'/f'cycle_{cycle:04d}_COMPLETE.json',dict(cycle=cycle,generation_pending=len(todo),
            trained_transition='1000 new optimizer updates or actual cycle terminal; nonterminal snapshots remain partial',
            wall_seconds=time.time()-started,frozen_unix=time.time()))
        cycle+=1
    while runner.running:runner.collect()
    runner.close()
    write_json(root(c)/'control/SCIENCE_COMPLETE.json',dict(status='SCIENCE_CUTOFF_PRESERVE_PARTIAL',cycle=cycle,elapsed_seconds=elapsed(c),
        generation_failed_regions=sorted(generation_failed),training_failed_regions=sorted(training_failed),next='J4 final freeze, hidden audit, reopen, seal'))

def main():
    from .common import load_config
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',choices=['J1','J23','AUTO'],required=True);args=p.parse_args();c=load_config(args.config)
    native_threads();gate=read_json(root(c)/'validation/PRE_RUN_GATE.json')
    assert gate['status']=='PASS' and gate['config_sha256']==sha(args.config)
    for binding in gate['code_bindings']:
        assert sha(Path(c['execution']['repository_root'])/binding['path'])==binding['sha256']
    if is_e2(c):
        assert args.stage=='J23','E2 never reruns J1'
        from .e2_setup import verify_ready
        verify_ready(c,args.config,require_reentry=True)
        worker_setup(c,'E2_supervisor')
    else:assert (root(c)/'validation/J0_LABEL_AUDIT_PASS.json').exists() and (root(c)/'validation/PROTECTED_INPUTS_FROZEN.json').exists()
    c['execution_head']=gate['head'];c['effective_config_sha256']=gate['config_sha256'];monitor=Monitor(c,'CAMPAIGN').start()
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    guardian=subprocess.Popen([c['execution']['python_executable'],'-m','partC_rev1_src.phase67.resources','--config',args.config,'--pid',str(os.getpid())],creationflags=flags)
    regions=fair_regions(c);cleanup_stale_leases(c);runner=Runner(c,2)
    try:
        from concurrent.futures.process import BrokenProcessPool
        while True:
            try:
                if args.stage in ['J1','AUTO']:print(J1(c,runner,regions),flush=True)
                if args.stage=='J1':runner.close()
                else:
                    J23(c,runner,regions)
                    if is_e2(c):from .e2_finalize import finalize
                    else:from .finalize import finalize
                    finalize(c)
                break
            except BrokenProcessPool as error:
                if not recover_pool(c,runner,error):
                    if is_e2(c):from .e2_finalize import finalize
                    else:from .finalize import finalize
                    finalize(c);break
                runner=Runner(c,2)
    finally:monitor.close()
if __name__=='__main__':main()
