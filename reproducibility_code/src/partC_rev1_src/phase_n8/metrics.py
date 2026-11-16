"""Explicitly gated R3 metrics from frozen pilots and existing physical times.

This entry point cannot start a campaign until the package's exact fast-track
gate passes and the final matrix digest matches. It never runs a physical solve.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import psutil

from .common import Context, ResourceLimit, read_json, write_json, read_npz, sha, array_hash, code_identity
from .supervise import run_supervised


class R3Context(Context):
    def __init__(self, config, run_id, deadline):
        self.config_path=Path(config).resolve(strict=True)
        self.c=read_json(self.config_path)
        if self.c['run_id']!=run_id or self.c['deadline_epoch']!=self.c['start_epoch']+28800:
            raise ValueError('Wrong original R2 identity or clock')
        self.repo=Path(self.c['repository']).resolve(strict=True)
        self.root=Path(self.c['output_root']).resolve(strict=True)
        gate=read_json(self.root/'FAST_TRACK_GATE.json')
        report=read_json(self.root/'FAST_TRACK_GATE_INPUT.json')
        if not gate.get('R3_auto_start_allowed') or report['frozen_manifest_sha256']!=sha(self.root/'CAMPAIGN_MANIFEST.json'):
            raise PermissionError('Exact frozen same-Goal R3 gate has not authorized execution')
        manifest=read_json(self.root/'CAMPAIGN_MANIFEST.json')
        if manifest['original_R2_start_epoch']!=self.c['start_epoch'] or manifest['total_deadline_epoch']!=self.c['start_epoch']+43200:
            raise ValueError('Campaign clock was reset')
        for relative,expected in manifest.get('actual_code_files',{}).items():
            if sha(self.repo/relative)!=expected:
                raise ValueError('Code changed after the campaign gate: '+relative)
        self.step='N8R3_PATH_FLOW_METRICS'
        self.started=time.time();self.monotonic_started=time.monotonic()
        self.bindings={};self.peak_rss=0;self.last_resource=0
        self.task_deadline=min(deadline,self.c['start_epoch']+43200-3600)
        self.guard()

    def guard(self,deadline=None):
        cutoff=self.task_deadline if deadline is None else min(self.task_deadline,deadline)
        now=max(time.time(),self.started+time.monotonic()-self.monotonic_started)
        if now>=cutoff:raise TimeoutError('Original combined deadline/reporting reserve reached')
        proc=psutil.Process();rss=proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:rss+=child.memory_info().rss
            except psutil.NoSuchProcess:pass
        self.peak_rss=max(rss,self.peak_rss)
        if rss>=10*1024**3 or psutil.virtual_memory().available<3*1024**3:
            raise ResourceLimit('R3 process-tree/system-headroom guard')
        if now-self.last_resource>=15:
            added=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
            if added>=150*1024**3:raise ResourceLimit('R3 added-disk guard')
            with (self.root/'logs/R3_resource_usage.jsonl').open('a') as out:
                import json
                out.write(json.dumps(dict(epoch=now,elapsed_since_R2_start=now-self.c['start_epoch'],
                    rss_bytes=rss,added_bytes=added,task=self.step))+'\n')
            self.last_resource=now


def path_summary(frame, *, track, scenario, population):
    valid=np.isfinite(frame.D_phys_s)&np.isfinite(frame.D_surrogate_s)
    v=frame.loc[valid];d=v.demand.to_numpy(float)
    reference=v.D_phys_s.to_numpy(float);pred=v.D_surrogate_s.to_numpy(float)
    err=pred-reference;ae=np.abs(err);rel=ae/np.maximum(np.abs(reference),1.)
    denom=float(d@reference);demand=float(d.sum())
    result=dict(scenario=scenario,track=track,population=population,
        population_role='HISTORICAL_NOW_SEEN_SELECTED_COHORT_NOT_A_NATIONAL_ESTIMATE',
        cohort_rows=len(frame),valid_paired_rows=int(valid.sum()),cohort_demand=float(frame.demand.sum()),
        valid_paired_demand=demand,physical_unreachable_rows=int((~np.isfinite(frame.D_phys_s)).sum()),
        surrogate_unreachable_rows=int((~np.isfinite(frame.D_surrogate_s)).sum()),
        additional_surrogate_unreachable_rows=int((np.isfinite(frame.D_phys_s)&~np.isfinite(frame.D_surrogate_s)).sum()),
        zero_reference_rows=int((reference==0).sum()),WREL_denominator_demand_seconds=denom,
        WREL=float(d@ae/denom) if denom>0 else None,
        WMEANREL_1S=float(d@rel/demand) if demand>0 else None,
        demand_weighted_signed_bias_seconds=float(d@err/demand) if demand>0 else None,
        demand_weighted_absolute_error_seconds=float(d@ae/demand) if demand>0 else None,
        unweighted_mean_signed_error_seconds=float(err.mean()) if len(err) else None,
        status='MEASURED_BOTH_TRACKS_SEPARATELY')
    for q in [50,90,95,99,100]:
        result[f'absolute_error_seconds_p{q}']=float(np.percentile(ae,q)) if len(ae) else None
        result[f'absolute_relative_1s_p{q}']=float(np.percentile(rel,q)) if len(rel) else None
    w=v.ratio_weight.to_numpy(float);wd=float(w@reference)
    result['source_ratio_weight_WREL']=float(w@ae/wd) if wd>0 else None
    result['source_ratio_weight_definition']='Exact native ratio_weight; demand times source stratum expansion; cohort-specific, now seen'
    return result


def evaluate_case(ctx, scenario, job_id):
    from .native import Sources, reopen_demand, load_registry
    from .objective import Composite
    from .oracle import Demand,aon
    sources=Sources(ctx)
    matrix=read_json(ctx.root/'CAMPAIGN_MANIFEST.json')
    row=next(r for r in matrix['scenarios'] if r['scenario_id']==scenario)
    if row['r3_disposition'] not in ['REUSE_COMPLETED_MATCHING_PILOT','FRESH_OFFICIAL_FW_FROM_FROZEN_INITIAL_RULE']:
        raise ValueError('This scenario has no admitted assignment for metrics')
    pilot=Path(row['assignment_root'])
    result=read_json(pilot/'RESULT.json')
    bindings=read_json(ctx.root/'R3_ASSIGNMENT_RESULTS_FREEZE.json')
    expected_result_hash=bindings['results'][scenario]['sha256']
    if sha(pilot/'RESULT.json')!=expected_result_hash:
        raise ValueError('Assignment result changed after the pre-metrics freeze')
    if not result['final_certificate_fresh'] or result['state_hash'] is None:
        raise ValueError('No freshly certified state is available for solved-state metrics')
    if result['stop_reason'].startswith('STOP_'):
        raise ValueError('A safety-stopped scenario cannot silently resume path queries')
    suite=sources.suite(row['suite']);original,uids,_,_=sources.original_od()
    full_od,balance,coverage=reopen_demand(ctx,suite,original)
    entries,registry=load_registry(sources,suite,row['registry_id'])
    objective=Composite(suite,entries,registry,guard=ctx.guard)
    ident=dict(graph=suite['graph'].identity,demand=full_od.identity,objective=objective.identity,
        support=suite['support_identity'],registry=registry['identity'],code=code_identity(ctx.repo)[0],
        scenario=scenario,solver='FW',tolerance=.001)
    if ident!=result['identity']:raise ValueError('Pilot source/code/graph/demand/objective no longer matches')
    if sha(pilot/'final_flow.npz')!=result['final_flow_sha256']:
        raise ValueError('Frozen solved flow changed')
    solved=read_npz(pilot/'final_flow.npz')['flow']
    if array_hash({'flow':solved})!=result['state_hash']:
        raise ValueError('Flow and certificate state differ')
    base=sources.base/'phase_05/fidelity_0906/run_001'
    sources.bind(base/'sampling/frozen_od_sample.csv','FROZEN_456_SAMPLE',
        '7dcb9d2fad9d159fd299c66326ca5e44a4b7eef8fcff9d3a21e8955f801f28fd')
    sources.bind(base/'receipts/F2.json','ACCEPTED_LOW_PATH_RESULTS_RECEIPT')
    expected=next(a['sha256'] for a in read_json(base/'receipts/F2.json')['artifacts'] if a['relative_path']=='results/PHYSICAL_LOW.csv')
    sources.bind(base/'results/PHYSICAL_LOW.csv','EXISTING_PHYSICAL_X45_PATH_TIMES',expected)
    sample=pd.read_csv(base/'sampling/frozen_od_sample.csv',float_precision='round_trip')
    # The historical producer load_sample() used pandas' default parser. Its
    # persisted demand/weight columns reproduce that parser exactly, whereas
    # new calculations retain the authoritative sample's round-trip values.
    legacy_sample=pd.read_csv(base/'sampling/frozen_od_sample.csv')
    physical=pd.read_csv(base/'results/PHYSICAL_LOW.csv',float_precision='round_trip')
    if len(sample)!=456 or sample.stable_od_key.duplicated().any() or physical.stable_od_key.duplicated().any():
        raise ValueError('Frozen path population changed')
    sample=sample.sort_values(['origin_idx','destination_idx'],kind='stable').reset_index(drop=True)
    physical=physical.set_index('stable_od_key').loc[sample.stable_od_key].reset_index()
    legacy_sample=legacy_sample.set_index('stable_od_key').loc[sample.stable_od_key].reset_index()
    for field in ['accepted_od_row','origin_idx','destination_idx','source_set','N_h','n_h']:
        if not np.array_equal(physical[field].to_numpy(),sample[field].to_numpy()):
            raise ValueError('Physical/sample native field mismatch: '+field)
    parsing=[]
    for field in ['demand','ratio_weight']:
        canonical=sample[field].to_numpy(float);saved=physical[field].to_numpy(float)
        if not np.array_equal(saved,legacy_sample[field].to_numpy(float)):
            raise ValueError('Physical/sample historical parser mismatch: '+field)
        parsing.append(dict(field=field,exact_historical_parser_match=True,
            canonical_difference_rows=int(np.count_nonzero(saved!=canonical)),
            maximum_absolute_difference=float(np.max(np.abs(saved-canonical))),
            calculation_source='Original frozen sample read with float_precision=round_trip'))
    indices=sample.accepted_od_row.to_numpy(np.int64)
    groups_of_rows=np.searchsorted(original['origin_offsets'][1:],indices,side='right')
    if not np.array_equal(original['origin_group_node_idx'][groups_of_rows],sample.origin_idx.to_numpy(np.int64)):
        raise ValueError('Sample origin identity mismatch')
    if not np.array_equal(original['demand'][indices],sample.demand.to_numpy(float)):
        raise ValueError('Sample original demand mismatch')
    # Extract exact mapped rows, including empty physical-self paths and serve mask.
    groups,count=np.unique(groups_of_rows,return_counts=True)
    sampled=Demand(full_od.origins[groups],np.r_[0,np.cumsum(count)],full_od.destinations[indices],
                   full_od.demand[indices],full_od.serve[indices])
    dest=ctx.root/'conditional_R3'/job_id;dest.mkdir(parents=True,exist_ok=False)
    summaries=[];states=[];flows=[]
    for track,flow in [('FIXED_X45',suite['fixed_x45']),('SOLVED_X45_REFERENCE',solved)]:
        ctx.guard();started=time.perf_counter()
        value,cost,_,diagnostic=objective.evaluate(flow)
        query=aon(suite['graph'],sampled,cost,guard=ctx.guard,return_distances=True)
        data=sample.copy();data['D_phys_s']=physical.D_phys_s.to_numpy(float)
        data['D_surrogate_s']=query['distances'];data['scenario']=scenario;data['track']=track
        data['graph_identity']=ident['graph'];data['state_hash']=array_hash({'flow':flow})
        data['reference_result_sha256']=expected
        data.to_csv(dest/(track+'_PATH_ROWS.csv'),index=False)
        for cohort,subset in data.groupby('source_set',sort=True):
            item=path_summary(subset,track=track,scenario=scenario,population=cohort)
            item.update(assignment_stop_reason=result['stop_reason'],assignment_RG_SPTT=result['final_certificate']['RG_SPTT'],
                state_hash=array_hash({'flow':flow}),graph_identity=ident['graph'])
            summaries.append(item)
        # Exact unchanged support and native coordinate order. No member-flow duplication.
        for name,sl in [('PACKED_NATIVE_LOGICAL_EDGES',slice(0,suite['packed_count'])),
                        ('ACTIVE_REGIONAL_NATIVE_COORDINATES',slice(suite['packed_count'],suite['packed_count']+len(suite['pairs'])))]:
            ref=suite['fixed_x45'][sl];current=flow[sl];den=float(np.abs(ref).sum())
            flows.append(dict(scenario=scenario,track=track,coordinate_population=name,coordinates=len(ref),
                flow_NL1=float(np.abs(current-ref).sum()/den) if den>0 else None,
                reference_L1=den,zero_reference_coordinates=int((ref==0).sum()),
                zero_reference_spurious_absolute_flow=float(np.abs(current[ref==0]).sum()),
                reference_hash=array_hash({'flow':ref}),current_hash=array_hash({'flow':current}),
                attribution='Regional traversal is not deduplicated OD demand; native member variables stay distinct'))
        states.append(dict(track=track,scenario=scenario,state_hash=array_hash({'flow':flow}),
            query_seconds=query['elapsed_seconds'],total_track_seconds=time.perf_counter()-started,
            unique_sample_origins=len(sampled.origins),sample_rows=len(sample),identity=ident,
            reference_path=str(base/'results/PHYSICAL_LOW.csv'),reference_sha256=expected,
            physical_sample_serialization_reconciliation=parsing,
            assignment_stop_reason=result['stop_reason'],assignment_certificate=result['final_certificate'],
            diagnostic=diagnostic,fixed_input_is_feasible_warm_start=False if track=='FIXED_X45' else None,
            global_path_lifting='NOT_REQUIRED_NOT_CLAIMED',physical_assignment_executed=False))
    pd.DataFrame(summaries).to_csv(dest/'PATH_METRICS.csv',index=False)
    pd.DataFrame(flows).to_csv(dest/'FLOW_METRICS.csv',index=False)
    write_json(dest/'METRIC_STATES.json',states)
    return dict(status='R3_METRICS_COMPLETE',scenario=scenario,job_id=job_id,
        reused_pilot_id=row['pilot_job_id'] if row['r3_disposition']=='REUSE_COMPLETED_MATCHING_PILOT' else None,
        assignment_run_id=result['job_id'],assignment_result_sha256=expected_result_hash,
        assignment_stop_reason=result['stop_reason'],assignment_compute_seconds=result['elapsed_seconds'],
        metrics_tracks=2,path_rows_each=456,
        original_R2_start=ctx.c['start_epoch'],combined_deadline=ctx.c['start_epoch']+43200,
        source_bindings=list(ctx.bindings.values()))


def main():
    p=argparse.ArgumentParser()
    for name in ['config','run-id','scenario','job-id']:p.add_argument('--'+name,required=True)
    p.add_argument('--cap-seconds',required=True,type=float)
    p.add_argument('--child',action='store_true');p.add_argument('--deadline',type=float)
    args=p.parse_args();c=read_json(args.config);root=Path(c['output_root'])
    deadline=args.deadline if args.child else min(time.time()+args.cap_seconds,c['start_epoch']+43200-3600)
    ctx=R3Context(args.config,args.run_id,deadline)
    manifest=read_json(root/'CAMPAIGN_MANIFEST.json')
    task=next(r for r in manifest['metrics_jobs'] if r['job_id']==args.job_id)
    if task['scenario_id']!=args.scenario or task['cap_seconds']!=args.cap_seconds:
        raise ValueError('Metric task does not match the frozen nonempty campaign list')
    if args.child:
        try:
            result=evaluate_case(ctx,args.scenario,args.job_id);ctx.finish(args.job_id,result);return 0
        except Exception as exc:
            write_json(root/'receipts'/(args.job_id+'_FAILURE.json'),dict(type=type(exc).__name__,
                error=str(exc),traceback=traceback.format_exc(),original_R2_start=c['start_epoch']))
            return 1
    receipt=root/'receipts'/(args.job_id+'_JOB.json')
    if receipt.exists():raise FileExistsError('Original metric job clock exists; do not relaunch')
    owner=root/'NUMERIC_OWNER.json'
    if owner.exists():
        prior=read_json(owner)
        if prior['status']=='ACTIVE' and any(psutil.pid_exists(prior.get(k,-1)) for k in ['supervisor_pid','child_pid']):
            raise RuntimeError('Another numeric job owns this Goal')
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:env[k]='1'
    env['NUMBA_CACHE_DIR']=str(root/'cache/numba_cache_dir');env['PYTHONDONTWRITEBYTECODE']='1'
    command=[c['python_executable'],'-B','-m','partC_rev1_src.phase_n8.metrics','--config',args.config,
        '--run-id',args.run_id,'--scenario',args.scenario,'--job-id',args.job_id,
        '--cap-seconds',str(args.cap_seconds),'--child','--deadline',str(deadline)]
    payload=dict(job_id=args.job_id,task='N8R3_METRICS',scenario=args.scenario,status='ACTIVE',
        supervisor_pid=os.getpid(),start_epoch=time.time(),deadline_epoch=deadline,
        original_R2_start=c['start_epoch'],workers=1,command=command)
    write_json(receipt,payload);write_json(owner,payload)
    def began(pid):
        payload['child_pid']=pid;write_json(receipt,payload);write_json(owner,payload)
    result=run_supervised(command,env,root/'logs'/(args.job_id+'.log'),deadline=deadline,
        rss_cap=10*1024**3,on_start=began)
    payload.update(status='FINISHED',result=result);write_json(receipt,payload);write_json(owner,payload)
    print(payload);return result['returncode']


if __name__=='__main__':raise SystemExit(main())
