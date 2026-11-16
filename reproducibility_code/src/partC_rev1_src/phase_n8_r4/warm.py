"""At most one unchanged-objective trajectory from the verified analytic solution."""
import time
import numpy as np
from partC_rev1_src.phase_n8.common import read_json,read_npz,write_json,array_hash,sha
from partC_rev1_src.phase_n8.native import load_registry
from partC_rev1_src.phase_n8.objective import Composite
from partC_rev1_src.phase_n8.oracle import conservation
from partC_rev1_src.phase_n8.solver import solve,save_checkpoint
from .safety import ORIGINAL_RUNS
from .context import sources,reopen_suite,code_identity


def run(ctx,args):
    if args.scenario not in ORIGINAL_RUNS:raise ValueError('Unknown original portfolio')
    src=sources(ctx);s,od,balance,coverage=reopen_suite(ctx,src,'s03')
    entries,registry=load_registry(src,s,args.scenario)
    original=Composite(s,entries,registry,guard=ctx.guard)
    prior=ctx.prior/'conditional_R3/assignments/R3_A02_S03_ANALYTIC_VALID'
    result=read_json(prior/'RESULT.json');ctx.bind(prior/'RESULT.json','CERTIFIED_ANALYTIC_WARM_SOURCE')
    ctx.bind(prior/'final_flow.npz','CERTIFIED_ANALYTIC_WARM_FLOW',result['final_flow_sha256'])
    flow=read_npz(prior/'final_flow.npz')['flow']
    if array_hash({'flow':flow})!=result['state_hash']:raise ValueError('Warm source state changed')
    if not result['final_certificate_fresh'] or result['final_certificate']['RG_SPTT']>.001:
        raise ValueError('Warm source is not converged with a fresh certificate')
    for name,value in [('graph',s['graph'].identity),('demand',od.identity),('support',s['support_identity'])]:
        if result['identity'][name]!=value:raise ValueError('Warm source graph/support/demand mismatch')
    feasible=conservation(s['graph'],flow,balance,od.total)
    ch,members,inherited=code_identity(ctx.repo)
    identity=dict(graph=s['graph'].identity,demand=od.identity,support=s['support_identity'],objective=original.identity,
        registry=registry['identity'],scenario=args.scenario+'_UNCHANGED_ANALYTIC_WARM_DIAGNOSTIC',code=ch,solver='FW',tolerance=.001)
    clocks=dict(R4_start_epoch=ctx.c['start_epoch'],R4_deadline_epoch=ctx.c['deadline_epoch'],run_start_epoch=args.job_start,run_deadline_epoch=args.job_deadline)
    root=ctx.root/'runs'/args.job_id;root.mkdir(exist_ok=False)
    write_json(root/'PRE_RUN_FREEZE.json',dict(identity=identity,clocks=clocks,coverage=coverage,code_members=members,
        warm_source_state_hash=result['state_hash'],warm_source_path=str(prior),warm_conservation=feasible,
        historical_analytic_startup_seconds=result.get('elapsed_seconds'),unchanged_original_weights=True,
        interpretation='One tested trajectory; no global neural nonnegativity guarantee'))
    save_checkpoint(root,flow,0,identity,clocks,None,suffix='verified_analytic_warm_seed')
    started=time.time();out=solve(s['graph'],od,original,balance,root,identity=identity,clocks=clocks,ctx=ctx,tol=.001,resume=True)
    out.update(initialization='VERIFIED_CONVERGED_ANALYTIC_SAME_GRAPH_FLOW',source_model_unchanged=True,
        scenario=args.scenario+'_UNCHANGED_ANALYTIC_WARM_DIAGNOSTIC',warm_source_state_hash=result['state_hash'],
        historical_analytic_startup_seconds=result.get('elapsed_seconds'),source_load_seconds=started-args.job_start,
        solver_elapsed_seconds=time.time()-started,coverage=coverage,job_id=args.job_id,suite='s03')
    write_json(root/'RESULT.json',out);return dict(status='EXECUTED',result=out)
