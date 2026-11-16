"""Explicit, logged entrypoints for the approved P-ACNHN campaign."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .control import Context,read_json,write_json,sha,array_hash
from .gates import COMPONENTS,component_snapshot,topology_snapshot,require_component

def source_snapshot(ctx):
    return component_snapshot(ctx,'reference')

def preflight(ctx,extra_configs=()):
    if ctx.c.get('continuation'):
        ctx.require_action('preflight')
    root=ctx.root/'validation';root.mkdir(parents=True,exist_ok=True)
    stamp=str(time.time_ns());log=root/f'pytest_{stamp}.txt'
    cmd=[sys.executable,'-X','utf8','-B','-m','pytest',str(Path(__file__).parent/'tests'),'-q','-p','no:cacheprovider']
    start=time.time()
    before={k:component_snapshot(ctx,k) for k in COMPONENTS}
    before_tests={str(p):sha(p) for p in (Path(__file__).parent/'tests').glob('*.py')}
    before_topology=topology_snapshot(ctx)
    extra={str(Path(p).resolve()):sha(p) for p in extra_configs}
    with log.open('w',encoding='utf-8') as out:
        result=subprocess.run(cmd,cwd=ctx.repo,stdout=out,stderr=subprocess.STDOUT)
    after={k:component_snapshot(ctx,k) for k in COMPONENTS}
    after_tests={str(p):sha(p) for p in (Path(__file__).parent/'tests').glob('*.py')}
    stable=(before==after and before_tests==after_tests and before_topology==topology_snapshot(ctx)
            and extra=={p:sha(p) for p in extra})
    record=dict(status='PASS' if result.returncode==0 and stable else 'FAIL',exit_code=result.returncode,
        command=cmd,log=str(log),log_sha256=sha(log),elapsed_seconds=time.time()-start,
        source_snapshot=source_snapshot(ctx),config_sha256=sha(ctx.path),recorded_epoch=time.time(),
        component_snapshots=after,topology_snapshot=before_topology,extra_configs=extra,
        test_snapshot=before_tests,inputs_stable_during_tests=stable)
    if (ctx.root/'topology/RESULT.json').exists():
        topology=read_json(ctx.root/'topology/RESULT.json')
        if topology['status']!='PASS':raise ValueError('Topology validation incomplete')
        record['topology_identity']=topology['graph_identity']
    write_json(root/f'PREFLIGHT_{stamp}.json',record)
    write_json(root/'PREFLIGHT_LATEST.json',record)
    print(log.read_text(encoding='utf-8'),flush=True)
    if result.returncode or not stable:raise RuntimeError('Preflight tests failed or code/config/inputs changed during checks')
    return record

def require_preflight(ctx,component='reference',extra_config=None):
    return require_component(ctx,component,extra_config=extra_config)

def reference(ctx,scenario,pilot=False):
    require_preflight(ctx)
    from .topology import load_common
    from .oracle import Demand
    from .solver import solve,PolynomialObjective
    started=time.time()
    topology_report=read_json(ctx.root/'topology/RESULT.json')
    if topology_report['status']!='PASS':raise ValueError('Topology hard gate failed')
    graph,od,a,p=load_common(ctx.root)
    for name in ['common_graph.npz','common_od.npz','partitions.npz']:
        ctx.bind(ctx.root/'topology'/name,'COMMON_P_FROZEN_INPUT')
    if pilot:
        count=min(32,len(od.origins));end=od.offsets[count]
        od=Demand(od.origins[:count],od.offsets[:count+1],od.destinations[:end],od.demand[:end],od.serve[:end])
    enabled=~a['closed_mask'] if scenario=='i95' else np.ones(len(a['tail']),bool)
    target=ctx.root/('pilots' if pilot else 'references')/scenario
    seconds=600 if pilot else ctx.c['reference']['per_scenario_seconds']
    deadline=min(started+seconds,ctx.c['maximum_cumulative_stage_deadlines'][0])
    intent=dict(phase_id=ctx.c['phase_id'],step_id=ctx.step_id,run_id=ctx.c['run_id'],scenario=scenario,pilot=pilot,
        tasks=['LOAD_P','FEASIBLE_INITIALIZE','CFW_UPDATES','REGIONAL_TRACES','FRESH_CERTIFICATE'],
        expected_outputs=['final_flow.npz','final_traces.npz','initial_witnesses','RESULT.json'],
        acceptance_tests=ctx.c['acceptance_tests'],resources=ctx.c['resources'],source_snapshot=source_snapshot(ctx),
        graph_identity=graph.identity,OD_identity=od.identity,run_start_epoch=started,deadline_epoch=deadline,
        full_coverage=not pilot,selected_origin_groups=len(od.origins),input_bindings=ctx.bindings)
    write_json(ctx.root/'intents'/f'{"pilot" if pilot else "reference"}_{scenario}.json',intent)
    result=solve(graph,od,PolynomialObjective(a['base'],a['k4']),target,method='CFW',
        run_start_epoch=started,deadline_epoch=deadline,tolerance=ctx.c['reference']['RG_SPTT'],ctx=ctx,
        enabled=enabled,edge_regions=p['edge_region'],workers=ctx.c['resources']['regional_workers'],
        capture_witness=True,max_updates=2 if pilot else None)
    ctx.flush_bindings();return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--action',required=True,choices=['preflight','pilot','reference','support','witness-audit'])
    parser.add_argument('--scenario',choices=['open','i95'])
    parser.add_argument('--extra-config',action='append',default=[])
    args=parser.parse_args()
    step={'preflight':'preflight','pilot':'preflight','reference':'reference_'+str(args.scenario),'support':'support','witness-audit':'support'}[args.action]
    ctx=Context(args.config,step_id=step)
    if ctx.c.get('continuation'):
        ctx.require_action(args.action)
    os.environ['NUMBA_CACHE_DIR']=str(ctx.root/'cache/numba')
    ctx.guard()
    write_json(ctx.root/'processes'/f'{args.action}_{os.getpid()}.json',dict(pid=os.getpid(),action=args.action,scenario=args.scenario,start_epoch=time.time(),state='RUNNING'))
    try:
        if args.action=='preflight':result=preflight(ctx,args.extra_config)
        elif args.action in {'pilot','reference'}:
            if not args.scenario:raise ValueError('Explicit scenario required')
            result=reference(ctx,args.scenario,args.action=='pilot')
        elif args.action=='support':
            require_preflight(ctx,'support')
            from .support import build
            result=build(ctx)
        else:
            require_preflight(ctx,'support')
            from .witnesses import audit
            result=audit(ctx)
        write_json(ctx.root/'processes'/f'{args.action}_{os.getpid()}.json',dict(pid=os.getpid(),action=args.action,scenario=args.scenario,end_epoch=time.time(),state='COMPLETE'))
        print('COMPLETE '+args.action,flush=True)
    except BaseException as exc:
        write_json(ctx.root/'processes'/f'{args.action}_{os.getpid()}.json',dict(pid=os.getpid(),action=args.action,scenario=args.scenario,end_epoch=time.time(),state='FAILED',exception=type(exc).__name__,message=str(exc)))
        raise

if __name__=='__main__':main()
