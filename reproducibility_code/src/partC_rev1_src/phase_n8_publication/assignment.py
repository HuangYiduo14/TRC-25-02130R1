"""One immutable objective per FW epoch; explicit feasible restarts on unsafe queries."""
from pathlib import Path
import time
import numpy as np
from partC_rev1_src.phase_n8.common import read_json,read_npz,write_json,save_npz,sha,array_hash
from partC_rev1_src.phase_n8.objective import validate_arrays
from partC_rev1_src.phase_n8.solver import solve,save_checkpoint
from partC_rev1_src.phase_n8.oracle import Demand,conservation
from .context import sources,reopen_suite,code_identity
from .overlay import apply_overlay
from .safe_objective import FixedSafeComposite
from .registry import load_portfolio,freeze_portfolio,POLY,FREE


def scenario_inputs(ctx,src,scenario):
    suite,od,balance,coverage=reopen_suite(ctx,src,'s03')
    if scenario.endswith('_CLOSED'):
        mask_record=read_json(ctx.root/'closure/MASK_FREEZE.json')
        ctx.bind(ctx.root/'closure/availability.npz','FROZEN_CLOSURE_AVAILABILITY',mask_record['mask_sha256'])
        mask=read_npz(ctx.root/'closure/availability.npz')['unavailable']
        suite['graph']=apply_overlay(suite['graph'],mask)
        path=ctx.root/'closure/closed_fftt_initial.npz';ctx.bind(path,'MASKED_FEASIBLE_INITIALIZATION')
        initial=read_npz(path)
        od=Demand(od.origins,od.offsets,od.destinations,od.demand,initial['serve'])
        balance=initial['balance'];coverage=read_json(ctx.root/'closure/CLOSED_REACHABILITY.json')
        if suite['graph'].identity!=coverage['graph_identity'] or od.identity!=coverage['demand_identity']:
            raise ValueError('Closure graph/demand differs from frozen feasible initialization')
    elif scenario.endswith('_OPEN'):
        path=ctx.root/'closure/open_fftt_initial.npz';ctx.bind(path,'OPEN_FEASIBLE_INITIALIZATION');initial=read_npz(path)
        if not np.array_equal(initial['serve'],od.serve) or not np.array_equal(initial['balance'],balance):
            raise ValueError('Open demand differs from frozen initialization')
    else:raise ValueError('Explicit OPEN/CLOSED scenario required')
    conservation(suite['graph'],initial['flow'],balance,od.total)
    if hasattr(suite['graph'],'unavailable') and np.any(initial['flow'][suite['graph'].unavailable]!=0):
        raise ValueError('Closed initialization includes unavailable flow')
    return suite,od,balance,coverage,initial['flow']


def pending_invalidations(ctx,current):
    """Completed paired states become obsolete if the common portfolio changes."""
    charged=set(current.get('charged_invalidated_jobs',[]));pending=[]
    for path in sorted((ctx.root/'runs').glob('*/RESULT.json')):
        row=read_json(path)
        if (row.get('scenario','').startswith('M1_') and row.get('final_registry_identity')==current['identity']
                and row['job_id'] not in charged):
            pending.append(dict(job_id=row['job_id'],seconds=float(row['elapsed_seconds']),path=str(path)))
    return pending


def downgrade(ctx,src,entries,current,failures,discarded_seconds,discarded_root,*,force_rescue=False,invalidated=()):
    chain_path=ctx.root/'registries/FALLBACK_CHAINS.json';ctx.bind(chain_path,'IMMUTABLE_FALLBACK_POLICY')
    policy=read_json(chain_path);by={e['region_id']:e for e in entries}
    failing={f['region_id'] for f in failures}
    if not failing or not failing.issubset(by):raise ValueError('Missing or unknown unsafe region')
    used=current['cumulative_recovery_seconds']+discarded_seconds+sum(r['seconds'] for r in invalidated)
    count=current['discovery_epochs']+1
    rescue=force_rescue or count>=policy['restart_epoch_cap'] or used>=policy['cumulative_recovery_cap_seconds']
    if rescue:failing|={rid for rid,e in by.items() if 'ICNN' in e['model_kind']}
    events=[];updated=[]
    for e in entries:
        rid=e['region_id']
        if rid not in failing:updated.append(e);continue
        chain=policy['chains'][rid];index=chain.index(e['model_kind'])
        if index+1>=len(chain):raise ArithmeticError('FFTT safety failed; no valid further fallback')
        chosen=POLY if rescue and 'ICNN' in e['model_kind'] else chain[index+1]
        row=policy['candidates'][rid][chosen]
        a=src.arrays(row['portable_path'],'PERMANENT_WHOLE_REGION_FALLBACK',row['portable_sha256']);validate_arrays(chosen,a)
        if not np.array_equal(a['c0'],e['arrays']['c0']):raise ValueError('Fallback native c0 mismatch')
        updated.append(dict(row,arrays=a))
        events.append(dict(region_id=rid,from_kind=e['model_kind'],to_kind=chosen,from_sha256=e['portable_sha256'],
            to_sha256=row['portable_sha256'],reason='DECLARED_ALL_NEURAL_RESCUE' if rescue else 'ABORTED_INVALID_QUERY'))
    number=current['epoch']+1
    registry=freeze_portfolio(ctx,f'M1_EPOCH_{number:03d}',updated,epoch=number,events=events)
    current=dict(current,name=registry['registry_id'],identity=registry['identity'],epoch=number,
        discovery_epochs=count,cumulative_recovery_seconds=used,rescue_used=current['rescue_used'] or rescue,
        permanently_downgraded_regions=sorted(set(current['permanently_downgraded_regions'])|failing),
        charged_invalidated_jobs=sorted(set(current.get('charged_invalidated_jobs',[]))|{r['job_id'] for r in invalidated}),
        invalidated_completed_runs=current.get('invalidated_completed_runs',[])+list(invalidated),
        discarded_epochs=current['discarded_epochs']+[dict(path=str(discarded_root),seconds=discarded_seconds,failures=failures)])
    write_json(ctx.root/'registries/M1_CURRENT.json',current)
    return updated,registry,current


def epoch_deadline(current,entries,begin,job_deadline,pending_seconds=0.):
    if current is None or current['rescue_used'] or not any('ICNN' in e['model_kind'] for e in entries):
        return job_deadline,False
    remaining=max(0.,7200.-current['cumulative_recovery_seconds']-pending_seconds)
    limited=min(job_deadline,begin+remaining)
    return limited,limited<job_deadline


def retain_fresh_state(root,result):
    if result['final_certificate_fresh']:return result
    if not result.get('last_certified_checkpoint'):return result
    old=Path(result['last_certified_checkpoint']);meta=read_json(old/'STATE.json')
    if meta['identity']!=result['identity'] or meta['certificate'] is None:
        raise ValueError('Last certificate is not for this fixed epoch')
    if sha(old/'flow.npz')!=meta['flow_file_sha256']:raise ValueError('Certified checkpoint hash differs')
    flow=read_npz(old/'flow.npz')['flow']
    if array_hash({'flow':flow})!=meta['certificate']['state_hash']:raise ValueError('Certified checkpoint state differs')
    # Preserve the exact terminal solver record before selecting its last certified state.
    write_json(root/'UNCERTIFIED_TERMINAL_RESULT.json',result)
    if (root/'final_flow.npz').exists():
        save_npz(root/'uncertified_terminal_flow.npz',read_npz(root/'final_flow.npz'))
    save_npz(root/'final_flow.npz',dict(flow=flow))
    result=dict(result,final_certificate=meta['certificate'],final_certificate_fresh=True,
        final_certificate_status='LAST_FEASIBLE_FRESH_CERTIFIED_STATE_SELECTED',state_hash=meta['state_hash'],
        accepted_iterations=meta['accepted_iteration'],final_flow_sha256=sha(root/'final_flow.npz'))
    write_json(root/'RESULT.json',result)
    return result


def run(ctx,args):
    if args.scenario not in {'M1_OPEN','M2_OPEN','M3_OPEN','M1_CLOSED','M2_CLOSED','M3_CLOSED'}:
        raise ValueError('Unknown explicit scientific scenario')
    if read_json(ctx.root/'registries/PREFLIGHT.json')['status']!='PASS':raise ValueError('Preflight gate incomplete')
    src=sources(ctx);suite,od,balance,coverage,seed=scenario_inputs(ctx,src,args.scenario)
    current=read_json(ctx.root/'registries/M1_CURRENT.json') if args.scenario.startswith('M1_') else None
    name=current['name'] if current else ('M2_POWER4' if args.scenario.startswith('M2_') else 'M3_FFTT')
    entries,registry=load_portfolio(ctx,src,suite,name)
    code,members,inherited=code_identity(ctx.repo)
    source_load_seconds=time.time()-args.job_start
    root=ctx.root/'runs'/args.job_id;root.mkdir(exist_ok=False);epoch_records=[]
    write_json(root/'PRE_RUN_FREEZE.json',dict(scenario=args.scenario,job_id=args.job_id,coverage=coverage,
        initial_registry=registry['identity'],graph_identity=suite['graph'].identity,demand_identity=od.identity,
        support_identity=suite['support_identity'],code_identity=code,code_members=members,
        original_start_epoch=ctx.c['start_epoch'],job_start_epoch=args.job_start,job_deadline_epoch=args.job_deadline,
        initial_flow_hash=array_hash({'flow':seed}),initialization='COMMON_FEASIBLE_FFTT_AON',workers=1,
        allowance_discarded_epochs_seconds=7200,max_discovery_epochs=6,method='ORDINARY_FW'))
    while True:
        ctx.guard();begin=time.time();epoch=root/f"epoch_{registry['epoch']:03d}";epoch.mkdir(exist_ok=False)
        objective=FixedSafeComposite(suite,entries,registry,guard=ctx.guard)
        identity=dict(graph=suite['graph'].identity,demand=od.identity,objective=objective.identity,
            support=suite['support_identity'],registry=registry['identity'],code=code,scenario=args.scenario,solver='FW',tolerance=.001)
        pending=pending_invalidations(ctx,current) if current else []
        limited_deadline,recovery_limited=epoch_deadline(current,entries,begin,args.job_deadline,sum(r['seconds'] for r in pending))
        clocks=dict(amendment_start_epoch=ctx.c['start_epoch'],amendment_deadline_epoch=ctx.c['deadline_epoch'],
            job_start_epoch=args.job_start,run_start_epoch=begin,run_deadline_epoch=limited_deadline)
        conservation(suite['graph'],seed,balance,od.total)
        save_checkpoint(epoch,seed,0,identity,clocks,None,suffix='epoch_restart_seed')
        result=solve(suite['graph'],od,objective,balance,epoch,identity=identity,clocks=clocks,ctx=ctx,tol=.001,resume=True)
        write_json(epoch/'INFERENCE_TIMINGS.json',objective.timings)
        duration=time.time()-begin
        epoch_records.append(dict(epoch=registry['epoch'],registry=registry['identity'],seconds=duration,
            stop_reason=result['stop_reason'],accepted_iterations=result['accepted_iterations'],path=str(epoch)))
        write_json(root/'EPOCHS.json',epoch_records)
        recovery_expired=recovery_limited and result['stop_reason']=='TIME_LIMIT'
        if result['stop_reason'] in {'STOP_NEGATIVE_COST','STOP_NONFINITE'} or recovery_expired:
            if current is None:raise ArithmeticError('Analytic fixed portfolio produced unsafe costs')
            safety=(dict(failing_regions=[dict(region_id=e['region_id'],reason='RECOVERY_ALLOWANCE_EXHAUSTED')
                for e in entries if 'ICNN' in e['model_kind']]) if recovery_expired else read_json(epoch/'SAFETY_STOP.json'))
            entries,registry,current=downgrade(ctx,src,entries,current,safety['failing_regions'],duration,epoch,
                force_rescue=recovery_expired,invalidated=pending)
            # The last accepted state remains feasible; no old objective or gap survives the reset.
            if (epoch/'state_at_stop.npz').exists():seed=read_npz(epoch/'state_at_stop.npz')['flow']
            elif recovery_expired and (epoch/'final_flow.npz').exists():seed=read_npz(epoch/'final_flow.npz')['flow']
            continue
        result=retain_fresh_state(epoch,result)
        final_path=epoch/'final_flow.npz'
        if final_path.exists():
            flow=read_npz(final_path)['flow'];conservation(suite['graph'],flow,balance,od.total)
            if hasattr(suite['graph'],'unavailable') and np.any(flow[suite['graph'].unavailable]!=0):
                raise ArithmeticError('Final unavailable edge carries flow')
            save_npz(root/'final_flow.npz',dict(flow=flow,edge_ids=suite['graph'].edge_ids))
        result.update(scenario=args.scenario,suite='s03',job_id=args.job_id,coverage=coverage,
            final_effective_registry=registry['registry_id'],final_registry_identity=registry['identity'],
            effective_composition=registry['composition'],final_epoch_seconds=duration,elapsed_seconds=time.time()-args.job_start,
            discarded_epoch_seconds=sum(r['seconds'] for r in epoch_records[:-1]),epochs=epoch_records,
            final_epoch_path=str(epoch),source_load_and_admission_seconds=source_load_seconds,
            final_flow_sha256=sha(root/'final_flow.npz') if (root/'final_flow.npz').exists() else None,
            initialization='FEASIBLE_FFTT_OR_PREVIOUS_EPOCH_ACCEPTED_FLOW; OBJECTIVE_STATE_RESET',
            phase_id='N8',step_id=ctx.c['step_id'],run_id=ctx.c['run_id'],model_weights_changed=False)
        write_json(root/'RESULT.json',result)
        return dict(status='EXECUTED',result=result)
