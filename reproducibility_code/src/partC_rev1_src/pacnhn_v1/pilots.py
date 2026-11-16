"""Explicit full-demand FFTT AC pilots after support/witness admission."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
import numpy as np
from .control import Context, read_json, write_json, sha


def pilot(ctx, sid, scenario, *, seconds=600, updates=2):
    from .assignments import load_ac
    from .solver import solve
    if sid not in ctx.c['suite_ids'] or scenario not in {'open','i95'}:
        raise ValueError('Explicit admitted suite and scenario required')
    if seconds<=0 or seconds>600 or updates!=2:
        raise ValueError('Approved pilot bound: at most 600 seconds and two updates')
    support=read_json(ctx.root/'SUPPORT_FREEZE.json')
    witness=read_json(ctx.root/'COMPLETE_WITNESS_AUDIT.json')
    if support['status']!='PASS' or witness['status']!='PASS' or witness['support_freeze_sha256']!=sha(ctx.root/'SUPPORT_FREEZE.json'):
        raise ValueError('Frozen support and all-OD witness gates required')
    # This probe has explicit scientific inputs, tests and a fresh code snapshot.
    # It is a timing pilot, never admitted as a final comparison by metrics.py.
    from .cli import source_snapshot
    from .gates import require_component
    require_component(ctx,'pilot')
    source={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('assignments.py'),Path(__file__).with_name('objective.py')]}
    source.update(source_snapshot(ctx))
    start=time.time();deadline=min(start+seconds,ctx.c['maximum_cumulative_stage_deadlines'][0])
    graph,od,a,objective=load_ac(ctx,sid,'FFTT',{},deadline=deadline)
    target=ctx.root/'pilots'/f'AC_{sid}_FFTT_{scenario}_fullOD'
    enabled=~a['closed_mask'] if scenario=='i95' else np.ones(len(graph.tail),bool)
    intent=dict(phase_id=ctx.c['phase_id'],step_id='preflight',run_id=ctx.c['run_id'],
        suite_id=sid,scenario=scenario,family='FFTT',solver='FW',full_coverage=True,
        tasks=['LOAD_FIXED_AC','FULL_OD_FEASIBLE_INITIALIZATION','TWO_FW_UPDATES','FRESH_CERTIFICATES','TIMING_EXPORT'],
        input_versions={'support_freeze_sha256':sha(ctx.root/'SUPPORT_FREEZE.json'),
                        'AC_graph_sha256':sha(ctx.root/'acnhn'/sid/'graph.npz')},
        graph_identity=graph.identity,OD_identity=od.identity,selected_origin_groups=len(od.origins),
        resources=ctx.c['resources'],source_snapshot=source,run_start_epoch=start,deadline_epoch=deadline,
        expected_outputs=['RESULT.json','HISTORY.json','TIMING_PILOT.json'],
        acceptance_tests=['FULL_OD_MEMBERSHIP_EQUALS_P','NATIVE_B_IDENTITY','FRESH_CERTIFICATES'],
        inference_scope='FFTT_ONLY_NO_TRAINED_MODEL_COST_FORECAST',
        convergence_forecast='CONDITIONAL_ON_UNMEASURED_FUTURE_ITERATION_COUNT')
    write_json(ctx.root/'intents'/f'pilot_AC_{sid}_{scenario}.json',intent)
    result=solve(graph,od,objective,target,method='FW',run_start_epoch=start,deadline_epoch=deadline,
        tolerance=1e-15,ctx=ctx,enabled=enabled,workers=ctx.c['resources']['regional_workers'],max_updates=updates)
    with np.load(target/'coverage.npz',allow_pickle=False) as z:actual=z['serve']
    with np.load(ctx.root/'references'/scenario/'coverage.npz',allow_pickle=False) as z:expected=z['serve']
    if not np.array_equal(actual,expected):raise ValueError('Full-OD pilot reachability differs from P')
    if not result['final_certificate_fresh']:raise ValueError('Pilot has no certified state')
    history=read_json(target/'HISTORY.json')
    timings=dict(status='PASS',pilot=True,full_coverage=True,suite_id=sid,scenario=scenario,family='FFTT',
        complete_query_seconds=result['elapsed_seconds'],measured_origin_groups=len(od.origins),
        served_rows=int(actual.sum()),fresh_certificates=len(history),
        mean_AON_seconds=float(np.mean([r['aon_seconds'] for r in history])),
        mean_value_cost_seconds=float(np.mean([r['value_cost_seconds'] for r in history])),
        mean_measured_update_cycle_seconds=(float(np.mean(np.diff([r['elapsed_seconds'] for r in history]))) if len(history)>1 else None),
        residuals=[r['RG_SPTT'] for r in history],result_sha256=sha(target/'RESULT.json'),
        exact_served_OD_membership=True,inference_scope=intent['inference_scope'],
        convergence_forecast=intent['convergence_forecast'])
    write_json(target/'TIMING_PILOT.json',timings);ctx.flush_bindings();return timings


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--suite-id',required=True);parser.add_argument('--scenario',required=True,choices=['open','i95'])
    args=parser.parse_args();ctx=Context(args.config,step_id='preflight')
    os.environ['NUMBA_CACHE_DIR']=str(ctx.root/'cache/numba')
    pilot(ctx,args.suite_id,args.scenario)
