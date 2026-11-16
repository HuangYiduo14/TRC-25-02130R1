"""Gated fresh official assignments, with distinct IDs and retained pilot clocks.

Completed matching pilots may be reused by the campaign. A censored pilot stays
censored; a separately budgeted fresh official experiment has its own declared
run ID. All earlier pilot time remains charged to the original twelve-hour Goal.
"""
import argparse
import os
from pathlib import Path
import time
import traceback

import numpy as np
import psutil

from .common import read_json,write_json,save_npz,sha,code_identity
from .metrics import R3Context
from .supervise import run_supervised


def execute_assignment(ctx,job_id,scenario,start,deadline):
    from .native import Sources,reopen_demand,load_registry
    from .objective import Composite,UnsafeCost
    from .solver import solve
    manifest=read_json(ctx.root/'CAMPAIGN_MANIFEST.json')
    row=next(r for r in manifest['scenarios'] if r['scenario_id']==scenario)
    if row['r3_disposition']!='FRESH_OFFICIAL_FW_FROM_FROZEN_INITIAL_RULE':
        raise ValueError('Fresh official assignment was not declared in the frozen campaign')
    sources=Sources(ctx);suite=sources.suite(row['suite']);original,_,_,_=sources.original_od()
    od,balance,coverage=reopen_demand(ctx,suite,original)
    entries,registry=load_registry(sources,suite,row['registry_id'])
    objective=Composite(suite,entries,registry,guard=ctx.guard)
    identity=dict(graph=suite['graph'].identity,demand=od.identity,objective=objective.identity,
        support=suite['support_identity'],registry=registry['identity'],code=code_identity(ctx.repo)[0],
        scenario=scenario,solver='FW',tolerance=.001)
    if identity!=row['expected_assignment_identity']:
        raise ValueError('Official assignment differs from the frozen graph/demand/code/objective')
    clocks=dict(R2_start_epoch=ctx.c['start_epoch'],R2_deadline_epoch=ctx.c['deadline_epoch'],
        combined_deadline_epoch=ctx.c['start_epoch']+43200,run_start_epoch=start,run_deadline_epoch=deadline)
    dest=ctx.root/'conditional_R3/assignments'/job_id;dest.mkdir(parents=True,exist_ok=False)
    write_json(dest/'PRE_RUN_FREEZE.json',dict(identity=identity,clocks=clocks,coverage=coverage,
        campaign_manifest_sha256=sha(ctx.root/'CAMPAIGN_MANIFEST.json'),
        dispatcher_sha256=sha(__file__),pilot_provenance=row['pilot_job_id'],
        pilot_clock_not_reset=True,experiment_role='FRESH_OFFICIAL_R3_FW'))
    try:
        t=time.perf_counter();fixed=objective.evaluate(suite['fixed_x45'],np.ones(len(suite['graph'].tail)))
        write_json(dest/'FIXED_X45_INFERENCE.json',dict(elapsed_seconds=time.perf_counter()-t,
            objective=fixed[0],diagnostic=fixed[3],source_state_role='MAPPED_X45_NOT_A_FEASIBLE_WARM_START'))
        load=time.time()-start;solver_start=time.time()
        result=solve(suite['graph'],od,objective,balance,dest,identity=identity,clocks=clocks,ctx=ctx,tol=.001)
        result.update(solver_elapsed_seconds=time.time()-solver_start,pre_run_input_load_seconds=load)
    except UnsafeCost as exc:
        save_npz(dest/'offending_query.npz',dict(flow=exc.flow));write_json(dest/'SAFETY_STOP.json',exc.diagnostic)
        result=dict(stop_reason=exc.reason,accepted_iterations=0,elapsed_seconds=time.time()-start,
            state_hash=None,final_certificate=None,last_valid_certificate=None,final_certificate_fresh=False,
            final_certificate_status='NO_VALID_ACCEPTED_SOLVER_STATE',identity=identity,clocks=clocks,
            initialization='FIXED_X45_ADMISSION_STOP',represented_demand=od.total,
            nominal_tolerance=.001,solver='FW',solver_elapsed_seconds=0.,pre_run_input_load_seconds=time.time()-start)
    result.update(scenario=scenario,suite=row['suite'],job_id=job_id,coverage=coverage,r3_official_run=True,
        pilot_provenance=row['pilot_job_id'],all_pilot_time_charged_to_original_goal=True)
    write_json(dest/'INFERENCE_TIMINGS.json',objective.timings);write_json(dest/'RESULT.json',result)
    ctx.finish(job_id,dict(status='R3_ASSIGNMENT_EXECUTED',run=result))
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ['config','run-id','scenario','job-id']:p.add_argument('--'+name,required=True)
    p.add_argument('--cap-seconds',required=True,type=float);p.add_argument('--child',action='store_true')
    p.add_argument('--deadline',type=float);p.add_argument('--start',type=float)
    args=p.parse_args();c=read_json(args.config);root=Path(c['output_root'])
    start=args.start if args.child else time.time()
    deadline=args.deadline if args.child else min(start+args.cap_seconds,c['start_epoch']+43200-3600)
    ctx=R3Context(args.config,args.run_id,deadline)
    ctx.step='N8R3_OFFICIAL_ASSIGNMENT'
    manifest=read_json(root/'CAMPAIGN_MANIFEST.json')
    task=next(r for r in manifest['assignment_jobs'] if r['job_id']==args.job_id)
    if task['scenario_id']!=args.scenario or task['cap_seconds']!=args.cap_seconds:
        raise ValueError('Assignment task/cap differs from the frozen campaign')
    if args.child:
        try:execute_assignment(ctx,args.job_id,args.scenario,start,deadline);return 0
        except Exception as exc:
            write_json(root/'receipts'/(args.job_id+'_FAILURE.json'),dict(type=type(exc).__name__,
                error=str(exc),traceback=traceback.format_exc(),original_R2_start=c['start_epoch']));return 1
    receipt=root/'receipts'/(args.job_id+'_JOB.json')
    if receipt.exists():raise FileExistsError('Existing official run clock; do not relaunch')
    owner=root/'NUMERIC_OWNER.json'
    if owner.exists():
        prior=read_json(owner)
        if prior['status']=='ACTIVE' and any(psutil.pid_exists(prior.get(k,-1)) for k in ['supervisor_pid','child_pid']):
            raise RuntimeError('Another numeric job owns this Goal')
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:env[k]='1'
    env['NUMBA_CACHE_DIR']=str(root/'cache/numba_cache_dir');env['PYTHONDONTWRITEBYTECODE']='1'
    command=[c['python_executable'],'-B','-m','partC_rev1_src.phase_n8.campaign','--config',args.config,
        '--run-id',args.run_id,'--scenario',args.scenario,'--job-id',args.job_id,
        '--cap-seconds',str(args.cap_seconds),'--child','--start',str(start),'--deadline',str(deadline)]
    payload=dict(job_id=args.job_id,task='N8R3_FRESH_OFFICIAL_ASSIGNMENT',scenario=args.scenario,status='ACTIVE',
        supervisor_pid=os.getpid(),start_epoch=start,deadline_epoch=deadline,
        original_R2_start=c['start_epoch'],workers=1,command=command)
    write_json(receipt,payload);write_json(owner,payload)
    def began(pid):
        payload['child_pid']=pid;write_json(receipt,payload);write_json(owner,payload)
    result=run_supervised(command,env,root/'logs'/(args.job_id+'.log'),deadline=deadline,
        rss_cap=10*1024**3,on_start=began)
    payload.update(status='FINISHED',result=result);write_json(receipt,payload);write_json(owner,payload)
    print(payload);return result['returncode']


if __name__=='__main__':raise SystemExit(main())
