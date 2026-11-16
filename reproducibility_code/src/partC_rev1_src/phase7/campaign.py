"""Single coordinator, two disjoint CPU workers, explicit original-clock campaign."""
from __future__ import annotations
import argparse,collections,concurrent.futures,faulthandler,json,os,subprocess,threading,time,traceback,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from .common import load_config,root,region_out,read_json,write_json,append_json,elapsed,seconds_left,sha
from .data import build_index,initialize_registry,fair_queue

def worker_init(out):
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    global STACK_FILE
    STACK_FILE=open(Path(out)/'logs'/f'worker_{os.getpid()}_stacks.log','a',encoding='utf-8')
    faulthandler.enable(STACK_FILE);faulthandler.dump_traceback_later(300,repeat=True,file=STACK_FILE)

def run_worker(c,row,first,finalize=False):
    from .training import run_slice
    progress=root(c)/'logs'/f'worker_{os.getpid()}_current.json'
    write_json(progress,dict(region_id=row['region_id'],pid=os.getpid(),event='START',unix=time.time(),first_pass=first))
    result=run_slice(c,row,first,finalize_override=finalize)
    write_json(progress,dict(region_id=row['region_id'],pid=os.getpid(),event='RETURN',unix=time.time(),status=result['status']))
    return result

class RecoveringPool:
    """Recreate an abruptly broken native process pool; retries retain checkpoints."""
    def __init__(self,out):self.out=out;self.pool=None
    def __enter__(self):self.rebuild();return self
    def rebuild(self):
        if self.pool is not None:self.pool.shutdown(wait=True,cancel_futures=True)
        self.pool=concurrent.futures.ProcessPoolExecutor(max_workers=2,initializer=worker_init,initargs=(str(self.out),))
    def submit(self,*args,**kwargs):return self.pool.submit(*args,**kwargs)
    def __exit__(self,*args):self.pool.shutdown(wait=True,cancel_futures=True)

def summarize(c,f):
    records=[]
    for r in f.to_dict('records'):
        p=region_out(c,r['region_id'])/'status.json'
        if p.exists():
            s=read_json(p);r.update({k:v for k,v in s.items() if k not in ['pack','counts_AB','traceback']})
            for split,n in s.get('counts_AB',s.get('pack',{}).get('counts_AB',{})).items():r['n_AB_'+split]=n
            r['counts_verified']='counts_AB' in s or 'pack' in s
        records.append(r)
    table=pd.DataFrame(records)
    for ext in ['csv','parquet']:
        p=root(c)/('region_status.'+ext);tmp=p.with_name(p.name+'.pending')
        table.to_csv(tmp,index=False) if ext=='csv' else table.to_parquet(tmp,index=False)
        os.replace(tmp,p)
    active=table.m_active.gt(0);trained=active&table.steps.fillna(0).gt(0)
    evaluated=trained&table.get('native_runtime_valid',pd.Series(False,index=table.index)).fillna(False).eq(True)
    cov=dict(unix=time.time(),elapsed_seconds=elapsed(c),status_counts=table.status.value_counts().to_dict(),
        active_regions=1566,inactive_regions=281,active_pairs=80316,LOW_exposure=float(table.loc[active,'low_exposure'].sum()))
    for name,mask in [('attempted',trained),('evaluated',evaluated)]:
        cov[name]=dict(regions=int(mask.sum()),pairs=int(table.loc[mask,'m_active'].sum()),LOW_exposure=float(table.loc[mask,'low_exposure'].sum()))
    write_json(root(c)/'coverage_summary.json',cov)
    return table,cov

class Watchdog:
    def __init__(self,c):self.c=c;self.stop=threading.Event();self.resource_stop=False;self.peak=0;self.max_disk=0;self.thread=threading.Thread(target=self.loop,daemon=True)
    def start(self):self.thread.start()
    def sample(self):
        p=psutil.Process();procs=[p]+p.children(recursive=True);rss=0
        for q in procs:
            try:rss+=q.memory_info().rss
            except psutil.Error:pass
        from .guardian import tree_bytes
        out=root(self.c);added=tree_bytes(out);free=shutil.disk_usage(out).free
        self.peak=max(self.peak,rss);self.max_disk=max(self.max_disk,added)
        self.resource_stop=bool(rss>=self.c['execution']['rss_bytes'] or added>=self.c['execution']['disk_new_bytes']-self.c['execution']['disk_seal_reserve_bytes'] or free<self.c['execution']['disk_seal_reserve_bytes'])
        record=dict(unix=time.time(),elapsed_seconds=elapsed(self.c),rss_bytes=rss,peak_rss_bytes=self.peak,added_output_bytes=added,free_disk_bytes=free,resource_stop=self.resource_stop,pids=[q.pid for q in procs])
        append_json(out/'resource_timeline.jsonl',record);write_json(out/'heartbeat.json',record)
        if elapsed(self.c)>=28800:
            write_json(out/'HARD_DEADLINE_REACHED.json',record)
            for q in procs[1:]:
                try:q.terminate()
                except psutil.Error:pass
            self.resource_stop=True
    def loop(self):
        while not self.stop.is_set():
            try:self.sample()
            except Exception:append_json(root(self.c)/'logs/watchdog_errors.jsonl',dict(traceback=traceback.format_exc(),unix=time.time()))
            self.stop.wait(30)

def run(c):
    out=root(c);(out/'logs').mkdir(exist_ok=True);lock=out/'LAUNCH_LOCK.json'
    if lock.exists():
        prior=read_json(lock)
        if psutil.pid_exists(prior['pid']):raise RuntimeError('Overlapping campaign writer')
    write_json(lock,dict(pid=os.getpid(),unix=time.time(),start=c['task_started_unix']))
    assert (out/'validation/T0_PASS.json').exists(),'Integrated correctness and checkpoint required'
    gate=out/'validation/CURRENT_EXECUTION_GATE.json'
    receipt=read_json(gate if gate.exists() else out/'validation/T0_PASS.json')
    git=['git','-C',c['execution']['repository_root'],'-c','core.longpaths=true']
    head=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()
    assert receipt['pre_run_head']==head
    for p,h in receipt['code_hashes'].items():assert sha(Path(c['execution']['repository_root'])/p)==h
    assert receipt['config_sha256']==sha(c['_config_path'])
    watchdog=Watchdog(c);watchdog.start();first_done=set();terminal={}
    try:
        f=initialize_registry(c);rows=f.set_index('region_id',drop=False).to_dict('index');queue=fair_queue(f)
        pd.DataFrame([dict(position=i,region_id=r,stratum=rows[r]['stratum'],seed=rows[r]['seed']) for i,r in enumerate(queue)]).to_csv(out/'training_queue.csv',index=False)
        write_json(out/'validation/optional_data_size_diagnostics.json',dict(status='NOT_EXECUTED',reason='Priority remains full-data coverage and main fitting; no subset diagnostics are required for this first campaign',preselected_regions=[queue[i] for i in np.linspace(0,len(queue)-1,8,dtype=int)]))
        for rid in queue:
            s=region_out(c,rid)/'status.json'
            if s.exists():
                st=read_json(s)
                if st.get('attempts',0)>0 or st['status'] in ['INSUFFICIENT_INDEPENDENT_AB_DATA','DATA_CONTRACT_FAILURE']:first_done.add(rid)
                if st['status'] not in ['TRAINING','PENDING_PACK','PENDING_TRAIN']:terminal[rid]=st
        turn=0;last_report=0.;eval_rate=[]
        retries={}
        with RecoveringPool(out) as pool:
            while True:
                first=len(first_done)<len(queue)
                todo=[r for r in queue if r not in first_done] if first else [r for r in queue if r not in terminal]
                if not todo:break
                # At least 45 min, dynamically enlarged from measured pending evaluation and reopen cost.
                evaluated_times=[v.get('evaluation_seconds',0.) for v in terminal.values() if v.get('native_runtime_valid')]
                per_eval=max(1.,float(np.quantile(evaluated_times,.95))) if evaluated_times else 1.
                pending_models=max(0,len(first_done)-len(terminal))
                reserve=max(2700.,1.5*(pending_models*per_eval/2+300.))
                c['effective_reserve_seconds']=reserve
                if seconds_left(c,reserve)<5 or watchdog.resource_stop:break
                turn+=1;pending={};next_i=0
                append_json(out/'training_events.jsonl',dict(event='ROUND_BEGIN',round=turn,first_pass=first,queued=len(todo),first_done=len(first_done),reserve_seconds=reserve,unix=time.time()))
                while pending or next_i<len(todo):
                    while len(pending)<2 and next_i<len(todo) and seconds_left(c,reserve)>10 and not watchdog.resource_stop:
                        rid=todo[next_i];next_i+=1
                        append_json(out/'training_events.jsonl',dict(event='DISPATCH',region_id=rid,first_pass=first,round=turn,unix=time.time()))
                        pending[pool.submit(run_worker,c,rows[rid],first)]=rid
                    if not pending:break
                    done,_=concurrent.futures.wait(pending,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        if future not in pending:continue
                        rid=pending.pop(future)
                        try:result=future.result()
                        except concurrent.futures.process.BrokenProcessPool:
                            interrupted=[rid]+list(pending.values());pending.clear()
                            append_json(out/'training_events.jsonl',dict(event='NATIVE_POOL_INTERRUPTED',regions=interrupted,round=turn,first_pass=first,unix=time.time(),traceback=traceback.format_exc()))
                            pool.rebuild()
                            for interrupted_rid in interrupted:
                                retries[interrupted_rid]=retries.get(interrupted_rid,0)+1
                                if retries[interrupted_rid]<=1:
                                    pending[pool.submit(run_worker,c,rows[interrupted_rid],first)]=interrupted_rid
                                    append_json(out/'training_events.jsonl',dict(event='NATIVE_WORKER_RETRY',region_id=interrupted_rid,first_pass=first,unix=time.time(),checkpoint_preserved=True))
                                else:
                                    rd=region_out(c,interrupted_rid);prior=read_json(rd/'status.json') if (rd/'status.json').exists() else {}
                                    failure=dict(**{k:v for k,v in prior.items() if k not in ['status','region_id']},region_id=interrupted_rid,status='TRAINING_NUMERICAL_FAILURE',
                                        error='Repeated native process termination; scientific label defect is not established',stage='NATIVE_WORKER',retry_count=retries[interrupted_rid],unix=time.time())
                                    write_json(rd/'status.json',failure);write_json(rd/'native_worker_failure.json',failure);terminal[interrupted_rid]=failure;first_done.add(interrupted_rid)
                            continue
                        first_done.add(rid)
                        if result['status']!='TRAINING':terminal[rid]=result
                        append_json(out/'training_events.jsonl',dict(event='RETURN',round=turn,first_pass=first,unix=time.time(),**{k:v for k,v in result.items() if k not in ['pack','traceback']}))
                    if time.time()-last_report>=30:
                        _,cov=summarize(c,f);last_report=time.time();print(json.dumps(cov),flush=True)
                if next_i<len(todo):break
        write_json(out/'TRAINING_CLOSED.json',dict(unix=time.time(),elapsed_seconds=elapsed(c),first_attempts=len(first_done),terminal_regions=len(terminal),training_cutoff=c['hard_deadline_unix']-c.get('effective_reserve_seconds',2700),resource_stop=watchdog.resource_stop))
        # Freeze/evaluate saved partials; the trainer performs zero updates in this mode.
        pending_eval=[]
        for rid in queue:
            rd=region_out(c,rid);p=rd/'status.json'
            if p.exists() and read_json(p).get('steps',0)>0 and not (rd/'evaluation/manifest.json').exists() and read_json(p)['status']=='TRAINING':pending_eval.append(rid)
        with concurrent.futures.ProcessPoolExecutor(max_workers=2,initializer=worker_init,initargs=(str(out),)) as pool:
            futures={pool.submit(run_worker,c,rows[rid],False,True):rid for rid in pending_eval}
            for future in concurrent.futures.as_completed(futures):future.result()
        table,cov=summarize(c,f)
        write_json(out/'CAMPAIGN_EXECUTION_DONE.json',dict(status='READY_FOR_FINAL_REPORT_AND_REOPEN',coverage=cov,elapsed_seconds=elapsed(c),original_clock_preserved=True,next_phase_authorized=False))
        print('CAMPAIGN_EXECUTION_DONE',flush=True)
    finally:
        watchdog.stop.set();watchdog.thread.join(timeout=5);watchdog.sample()
        write_json(out/'LAUNCH_EXIT.json',dict(pid=os.getpid(),unix=time.time(),elapsed_seconds=elapsed(c)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--phase',required=True);parser.add_argument('--step',required=True);parser.add_argument('--run-id',required=True)
    a=parser.parse_args();c=load_config(a.config)
    assert a.phase==c['phase_id'] and a.step==c['step_id'] and a.run_id==c['execution']['run_id'];c['_config_path']=str(Path(a.config).resolve());run(c)
if __name__=='__main__':main()
