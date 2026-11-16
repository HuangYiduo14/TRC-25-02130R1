"""Frozen draw-unit and historical-cohort paths, with explicit weights and tails."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json, read_npz, write_json, sha, array_hash
from partC_rev1_src.phase_n8.oracle import Network, Demand, aon
from partC_rev1_src.phase_n8.native import load_registry
from partC_rev1_src.phase_n8.objective import Composite
from .context import sources, reopen_suite
from .safety import load_safe, SafeComposite
from .diagnostic_handoffs import progress


def summarize(frame, weight, role, *, bootstrap=False):
    valid=np.isfinite(frame.D_phys_s)&np.isfinite(frame.D_surrogate_s)
    v=frame.loc[valid];w=np.asarray(weight,float)[valid]
    ref=v.D_phys_s.to_numpy(float);pred=v.D_surrogate_s.to_numpy(float)
    err=pred-ref;ae=np.abs(err);rel=ae/np.maximum(np.abs(ref),1.)
    den=float(w@ref);sw=float(w.sum());numerator=w*ae
    result=dict(weight_definition=role,rows=len(frame),valid_paired_rows=int(valid.sum()),
        physical_unreachable_rows=int((~np.isfinite(frame.D_phys_s)).sum()),
        surrogate_unreachable_rows=int((~np.isfinite(frame.D_surrogate_s)).sum()),
        additional_surrogate_unreachable_rows=int((np.isfinite(frame.D_phys_s)&~np.isfinite(frame.D_surrogate_s)).sum()),
        WREL=float(numerator.sum()/den) if den>0 else None,
        WMEANREL_1S=float(w@rel/sw) if sw>0 else None,
        weighted_signed_bias_seconds=float(w@err/sw) if sw>0 else None,
        weighted_MAE_seconds=float(w@ae/sw) if sw>0 else None,
        WREL_reference_denominator=den,weight_sum=sw,zero_reference_rows=int((ref==0).sum()))
    for q in (0,50,90,95,99,100):
        for name,a in [('reference_seconds',ref),('absolute_error_seconds',ae),('relative_error_1s',rel)]:
            result[f'{name}_p{q}']=float(np.percentile(a,q)) if len(a) else None
    order=np.argsort(-numerator,kind='stable')
    for n in (1,5):
        result[f'top{n}_WREL_numerator_share']=float(numerator[order[:n]].sum()/numerator.sum()) if numerator.sum()>0 else 0.
        keep=order[n:];d=float(w[keep]@ref[keep])
        result[f'SENSITIVITY_EXCLUDING_TOP{n}_WREL']=float(numerator[keep].sum()/d) if d>0 else None
    if bootstrap and len(v):
        # PPS draws, including duplicates, are the independent sampling units.
        rng=np.random.default_rng(20260915);metrics=[]
        for _ in range(2000):
            take=rng.integers(0,len(v),len(v));d=ref[take].sum()
            metrics.append((ae[take].sum()/d if d>0 else np.nan,rel[take].mean()))
        b=np.asarray(metrics)
        for k,name in enumerate(('WREL','WMEANREL_1S')):
            result[name+'_bootstrap_draw_95_low']=float(np.nanquantile(b[:,k],.025))
            result[name+'_bootstrap_draw_95_high']=float(np.nanquantile(b[:,k],.975))
        result['bootstrap_replicates']=2000
        result['bootstrap_scope']='PAIRED_DRAW_UNITS_CONDITIONAL_ON_FINITE_PAIRS; coverage reported separately'
    return result


def sample_demand(original,full,frame):
    indices=frame.accepted_od_row.to_numpy(np.int64)
    groups_of_rows=np.searchsorted(original['origin_offsets'][1:],indices,side='right')
    if (np.diff(groups_of_rows)<0).any():raise ValueError('Sample must be stably grouped by original origin')
    if not np.array_equal(original['demand'][indices],frame.demand.to_numpy(float)):
        raise ValueError('Canonical sample demand changed')
    groups,count=np.unique(groups_of_rows,return_counts=True)
    return Demand(full.origins[groups],np.r_[0,np.cumsum(count)],full.destinations[indices],
                  full.demand[indices],full.serve[indices])


def bindings(ctx):
    from partC_rev1_src.phase5.inputs import load_bindings
    p=ctx.repo/'partC_doc_icnn/phase_5_work/phase5_pilot_v4_0905_execution_config.json'
    ctx.bind(p,'PHYSICAL_REFERENCE_SOURCE_LOADER')
    b=load_bindings(p,verify_suite_digests=False,suite_ids=['s03'],load_aon=False)
    for which in ('low_benchmark','high_benchmark'):
        ctx.bind(b.sources[which]['physical_arrays'],'ACCEPTED_'+which.upper()+'_ARRAYS')
    return b


def cases(ctx):
    manifest=read_json(ctx.prior/'CAMPAIGN_MANIFEST.json');rows=[]
    for r in manifest['scenarios']:
        if r['r3_disposition'] not in ('REUSE_COMPLETED_MATCHING_PILOT','FRESH_OFFICIAL_FW_FROM_FROZEN_INITIAL_RULE'):continue
        rows.append(dict(scenario=r['scenario_id'],suite=r['suite'],registry=r['registry_id'],root=r['assignment_root'],safe=False))
    for scenario,job in [('N8R4_PRIMARY_BOUND_SAFE_V1','B02_PRIMARY_BOUND_SAFE'),
                         ('N8R4_BEST_ANY_BOUND_SAFE_V1','B03_BEST_ANY_BOUND_SAFE'),
                         ('N8R4_NONCONSTANT_SOURCE_BOUND_SAFE_V1','B04_NONCONSTANT_BOUND_SAFE')]:
        rows.append(dict(scenario=scenario,suite='s03',registry=scenario,root=str(ctx.root/'runs'/job),safe=True))
    return rows


def missing_state(ctx,root):
    """Preserve actual terminal failure and saved evidence without re-certifying."""
    info=dict(status='MISSING_ASSIGNMENT_RESULT')
    exitpath=ctx.root/'jobs'/root.name/'EXIT.json'
    receipt=ctx.root/'receipts'/(root.name+'_RESULT.json')
    if exitpath.exists():
        e=read_json(exitpath);info.update(status=e.get('supervisor_stop') or 'NO_TERMINAL_SCIENTIFIC_RESULT',
            child_exit_code=e.get('exit_code'),complete_child_seconds=e.get('elapsed_seconds'),exit_path=str(exitpath),exit_sha256=sha(exitpath))
    if receipt.exists():
        r=read_json(receipt);info.update(task_status=r.get('status'),failure_type=r.get('error_type'),failure_detail=r.get('error'))
    journal=root/'fresh_certificates.jsonl'
    if journal.exists():
        last=None
        with journal.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():last=json.loads(line)
        info.update(last_saved_certificate=last,saved_certificate_journal=str(journal),
            saved_certificate_policy='Indexed diagnostic evidence only; no terminal admission or reconstructed AON')
    return info


def _run(ctx,args):
    dst=ctx.root/'path_metrics'
    design=read_json(ctx.root/'sampling/DESIGN_FREEZE.json')
    ctx.bind(ctx.root/'sampling/PPS_draws.csv','PREOUTCOME_PPS_DESIGN',design['sample_sha256'])
    pps=pd.read_csv(ctx.root/'sampling/PPS_draws.csv',float_precision='round_trip')
    pps['source_set']='FRESH_PPS_DRAW';pps['sample_role']='DEMAND_PPS_WITH_REPLACEMENT'
    src=sources(ctx);original,uids,reachable,meta=src.original_od()
    oldroot=src.base/'phase_05/fidelity_0906/run_001'
    ctx.bind(oldroot/'sampling/frozen_od_sample.csv','HISTORICAL_NOW_SEEN_456',
             '7dcb9d2fad9d159fd299c66326ca5e44a4b7eef8fcff9d3a21e8955f801f28fd')
    ctx.bind(oldroot/'results/PHYSICAL_LOW.csv','ACCEPTED_456_PHYSICAL_PATHS',
             'e17992c274ac7e47ea8fc0313cdc8b890188ec7c60ecbff72e9222f26b8755ea')
    old=pd.read_csv(oldroot/'sampling/frozen_od_sample.csv',float_precision='round_trip')
    old['sample_role']='HISTORICAL_NOW_SEEN';old['draw_id']=np.arange(1,len(old)+1)
    sample=pd.concat([old,pps],ignore_index=True).sort_values(['origin_idx','destination_idx','sample_role','draw_id'],kind='stable').reset_index(drop=True)
    if len(pps)!=2048 or len(old)!=456:raise ValueError('Frozen sample sizes changed')
    rows=cases(ctx);write_json(dst/'EVALUATION_FREEZE.json',dict(cases=rows,sample_sha256=design['sample_sha256'],
        historical_456_unchanged=True,tracks=['FIXED_X45','SOLVED_X45_REFERENCE'],frozen_epoch=time.time(),
        bootstrap='2000 independent paired PPS draw-unit resamples seed20260915; no second weighting'))
    b=bindings(ctx);g=b.graph
    # The accepted historical physical reference allows all physical vertices in transit.
    # Retain exactly that policy; ACNHN centroid blocking remains native and is audited separately.
    physical=Network(g.tail,g.head,np.zeros(len(g.node_idx_to_uid),bool),
        g.arc_idx_to_directed_arc_id.astype(str),g.node_idx_to_uid.astype(str))
    full=Demand(original['origin_group_node_idx'],original['origin_offsets'],original['destination_node_idx'],
                original['demand'],reachable)
    sampled=sample_demand(original,full,sample);start=time.perf_counter()
    query=aon(physical,sampled,b.low_physical.current_cost_s,guard=ctx.guard,return_distances=True)
    sample['D_phys_s']=query['distances']
    historical=pd.read_csv(oldroot/'results/PHYSICAL_LOW.csv',float_precision='round_trip').set_index('stable_od_key')
    oldquery=sample.loc[sample.sample_role.eq('HISTORICAL_NOW_SEEN')]
    reference=historical.loc[oldquery.stable_od_key].D_phys_s.to_numpy(float)
    if not np.allclose(oldquery.D_phys_s,reference,rtol=2e-13,atol=1e-7,equal_nan=True):
        raise ValueError('Fresh physical shortest times fail accepted historical reference reproduction')
    sample.to_csv(dst/'PHYSICAL_SAMPLE_ROWS.csv',index=False,float_format='%.17g')
    write_json(dst/'PHYSICAL_QUERY.json',dict(graph_identity=physical.identity,seconds=time.perf_counter()-start,
        reference='CURRENT_X45',cost_units='seconds',centroid_policy='EXACT_ACCEPTED_PHYSICAL_SOURCE_POLICY_NO_TRANSIT_BLOCK',
        historical456_max_absolute_difference=float(np.max(np.abs(oldquery.D_phys_s.to_numpy()-reference))),
        physical_assignment_executed=False))
    del physical,b
    summaries=[];states=[]
    for row in rows:
        ctx.guard();root=Path(row['root']);rp=root/'RESULT.json'
        progress(dst,'METRIC_STATES.json',states,active=dict(row,attempt_stage='SCENARIO_ADMISSION'))
        if not rp.exists():
            states.append(dict(**row,**missing_state(ctx,root)));progress(dst,'METRIC_STATES.json',states);continue
        result=read_json(rp);ctx.bind(rp,'FROZEN_ASSIGNMENT_FOR_PATHS')
        if not result.get('final_certificate_fresh') or result.get('stop_reason','').startswith('STOP_'):
            states.append(dict(**row,status='NO_ADMITTED_FRESH_STATE',stop_reason=result.get('stop_reason'),
                final_certificate_fresh=result.get('final_certificate_fresh'),
                diagnostic_final_RG_SPTT=(result.get('final_certificate') or {}).get('RG_SPTT')))
            progress(dst,'METRIC_STATES.json',states);continue
        fp=root/'final_flow.npz';expected=result.get('final_flow_sha256',result.get('current_flow_file_sha256'))
        ctx.bind(fp,'SAME_STATE_SOLVED_FLOW',expected);solved=read_npz(fp)['flow']
        if array_hash({'flow':solved})!=result['state_hash']:raise ValueError('Solved certificate/flow identity mismatch')
        s,od,balance,coverage=reopen_suite(ctx,src,row['suite'])
        entries,registry=load_safe(ctx,src,s,row['registry']) if row['safe'] else load_registry(src,s,row['registry'])
        objective=(SafeComposite if row['safe'] else Composite)(s,entries,registry,guard=ctx.guard)
        for name,actual in [('graph',s['graph'].identity),('demand',od.identity),('objective',objective.identity),('support',s['support_identity'])]:
            if result['identity'][name]!=actual:raise ValueError('Frozen assignment identity changed: '+name)
        sampled=sample_demand(original,od,sample)
        for track,flow in [('FIXED_X45',s['fixed_x45']),('SOLVED_X45_REFERENCE',solved)]:
            ctx.guard();progress(dst,'METRIC_STATES.json',states,active=dict(row,track=track,attempt_stage='PATH_QUERY'))
            start=time.perf_counter();value,cost,hvp,diagnostic=objective.evaluate(flow)
            query=aon(s['graph'],sampled,cost,guard=ctx.guard,return_distances=True)
            data=sample.copy();data['D_surrogate_s']=query['distances'];data['scenario']=row['scenario'];data['track']=track
            data['state_hash']=array_hash({'flow':flow});data['graph_identity']=s['graph'].identity
            data['assignment_stop_reason']=result['stop_reason'];data['assignment_RG_SPTT']=result['final_certificate']['RG_SPTT']
            data['signed_error_seconds']=data.D_surrogate_s-data.D_phys_s
            data['absolute_error_seconds']=np.abs(data.signed_error_seconds)
            data['relative_error_1s']=data.absolute_error_seconds/np.maximum(np.abs(data.D_phys_s),1.)
            data.to_csv(dst/(row['scenario']+'_'+track+'_PATH_ROWS.csv'),index=False,float_format='%.17g')
            for cohort,v in data.groupby('source_set',sort=True):
                designs=[(np.ones(len(v)),'EQUAL_PPS_DRAW_UNITS_NO_SECOND_DEMAND_WEIGHT',True)] if cohort=='FRESH_PPS_DRAW' else [
                    (v.demand.to_numpy(float),'RAW_ORIGINAL_OD_DEMAND',False),(v.ratio_weight.to_numpy(float),'SOURCE_RATIO_WEIGHT_DEMAND_TIMES_STRATUM_EXPANSION',False)]
                for w,role,boot in designs:
                    summaries.append(dict(scenario=row['scenario'],suite=row['suite'],track=track,cohort=cohort,
                        **summarize(v,w,role,bootstrap=boot)))
            states.append(dict(**row,status='PATHS_MEASURED',track=track,rows=len(data),query_seconds=query['elapsed_seconds'],
                complete_track_seconds=time.perf_counter()-start,state_hash=array_hash({'flow':flow}),
                graph_identity=s['graph'].identity,assignment_stop_reason=result['stop_reason'],
                assignment_RG_SPTT=result['final_certificate']['RG_SPTT']))
            pd.DataFrame(summaries).to_csv(dst/'PATH_METRICS.csv',index=False,float_format='%.17g')
            progress(dst,'METRIC_STATES.json',states)
            print('PATH',row['scenario'],track,'completed',len(data),'rows',flush=True)
        del objective,entries,s,od,solved
    progress(dst,'METRIC_STATES.json',states)
    measured=sum(r['status']=='PATHS_MEASURED' for r in states)
    return dict(status='EXECUTED' if measured else 'NO_ADMITTED_STATES',case_count=len(rows),measured_tracks=measured,
        paired_rows_each=len(sample),probability_draws=2048,historical_rows=456)


def run(ctx,args):
    from .diagnostic_handoffs import bounded
    if (ctx.root/'path_metrics').exists():raise FileExistsError('Path diagnostics already started; preserve prior outputs')
    plans=[dict(row,track=track) for row in cases(ctx) for track in ('FIXED_X45','SOLVED_X45_REFERENCE')]
    columns=['scenario','suite','track','cohort','weight_definition','rows','valid_paired_rows',
        'physical_unreachable_rows','surrogate_unreachable_rows','additional_surrogate_unreachable_rows',
        'WREL','WMEANREL_1S','weighted_signed_bias_seconds','weighted_MAE_seconds','WREL_reference_denominator',
        'weight_sum','zero_reference_rows','WREL_bootstrap_draw_95_low','WREL_bootstrap_draw_95_high',
        'WMEANREL_1S_bootstrap_draw_95_low','WMEANREL_1S_bootstrap_draw_95_high','bootstrap_replicates','bootstrap_scope',
        'top1_WREL_numerator_share','top5_WREL_numerator_share','SENSITIVITY_EXCLUDING_TOP1_WREL','SENSITIVITY_EXCLUDING_TOP5_WREL']
    columns += [f'{name}_p{q}' for name in ('reference_seconds','absolute_error_seconds','relative_error_1s') for q in (0,50,90,95,99,100)]
    sample_columns=['stable_od_key','accepted_od_row','origin_idx','destination_idx','demand','source_set','sample_role','draw_id','D_phys_s']
    return bounded(ctx,args,'path_metrics',plans,['scenario','track'],'METRIC_STATES.json',
        {'PATH_METRICS.csv':columns,'PHYSICAL_SAMPLE_ROWS.csv':sample_columns},_run,reserve=120)
