"""Audited physical flow recovery; unknown parallel allocations stay unknown."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,read_npz,write_json,sha,array_hash
from .context import sources
from .path_metrics import bindings,cases,missing_state
from .diagnostic_handoffs import progress


def mapping(ctx,src,sid,graph):
    root=src.p5/'suites'/sid
    ledger=src.frame(root/'physical_link_ownership.parquet',sid+'_PHYSICAL_OWNERSHIP')
    links=src.frame(root/'runtime_links.parquet',sid+'_NATIVE_PACKED_LINK_ORDER')
    src.bind(root/'counts.json',sid+'_ACCEPTED_EXHAUSTIVE_SERIES_INCIDENCE')
    incidence=read_json(root/'counts.json')['incidence']
    if incidence['status']!='PASS' or not incidence['exactly_one_owner'] or any(incidence[k]!=0 for k in
            ('dangling_surviving_incidence','removed_od_nodes','replacement_endpoint_conflicts')):
        raise ValueError('Accepted strict-series incidence certificate is not valid')
    if len(ledger)!=len(graph.tail) or ledger.physical_arc_idx.duplicated().any():raise ValueError('Physical ownership rows changed')
    ledger=ledger.sort_values('physical_arc_idx',kind='stable').reset_index(drop=True)
    if not np.array_equal(ledger.physical_arc_idx,np.arange(len(graph.tail))) or not np.array_equal(
        ledger.physical_arc_id.astype(str),graph.arc_idx_to_directed_arc_id.astype(str)):
        raise ValueError('Physical ownership ID/order mismatch')
    if not ledger.owner_count.eq(1).all():raise ValueError('Nonunique physical ownership')
    ix=pd.Index(links.link_id);logical=ix.get_indexer(ledger.final_link_id)
    role=np.full(len(ledger),'',object);known=logical>=0
    role[known]=links.role.to_numpy()[logical[known]]
    inter=ledger.ownership_role.eq('INTER_B').to_numpy()
    member_counts=ledger.loc[inter].groupby('original_owner_id').size()
    count=ledger.original_owner_id.map(member_counts).fillna(0).to_numpy(np.int64)
    direct=inter & (role=='PHYSICAL_MEMBER')
    if not np.array_equal(links.member_arc_idx.to_numpy()[logical[direct]].astype(np.int64),np.flatnonzero(direct)):
        raise ValueError('Explicit physical-member arc identity mismatch')
    singleton=inter & (count==1) & (role=='EXACT_PRIMITIVE')
    series=inter & (count==1) & (role=='MACRO')
    seq=src.frame(root/'macro_sequences.parquet',sid+'_STRICT_SERIES_SOURCE_SEQUENCE')
    if series.any():
        key=seq[['corridor_id','primitive_id','sequence_position']].drop_duplicates()
        if key.duplicated(['primitive_id']).any():raise ValueError('Primitive occurs in multiple macros')
        check=ledger.loc[series,['final_link_id','original_owner_id','sequence_position']].merge(key,
            left_on=['final_link_id','original_owner_id','sequence_position'],
            right_on=['corridor_id','primitive_id','sequence_position'],how='left',validate='one_to_one',indicator=True)
        if not check['_merge'].eq('both').all():raise ValueError('MACRO common-flow sequence not certified')
    recover=direct|singleton|series
    reason=np.select([direct,singleton,series,ledger.ownership_role.eq('REGION'),ledger.ownership_role.eq('SUPPRESSED_B'),inter & (count>1)],
        ['EXPLICIT_PHYSICAL_MEMBER','SINGLETON_EXACT_PRIMITIVE','SINGLETON_MEMBER_OF_CERTIFIED_STRICT_SERIES',
         'REGIONAL_INTERIOR_FLOW_NOT_RECOVERED','SUPPRESSED_B_INTERIOR_NOT_A_PHYSICAL_ZERO','UNKNOWN_PARALLEL_MEMBER_ALLOCATION'],
        default='NO_VERIFIED_PHYSICAL_RECOVERY')
    ref=np.full(len(ledger),np.nan);ref[recover]=links.low_current_flow.to_numpy(float)[logical[recover]]
    native=graph.arc_idx_to_directed_arc_id.astype(str)
    return pd.DataFrame(dict(physical_arc_id=native,physical_arc_idx=np.arange(len(native)),
        suite=sid,source_logical_id=ledger.final_link_id,logical_index=logical,ownership_role=ledger.ownership_role,
        original_owner_id=ledger.original_owner_id,member_count=count,sequence_position=ledger.sequence_position,
        source_logical_role=role,recoverable=recover,recovery_class_or_exclusion=reason,
        mapped_x45_logical_flow=ref))


def freeze(ctx,args):
    src=sources(ctx);b=bindings(ctx);g=b.graph
    dst=ctx.root/'physical_population';dst.mkdir(exist_ok=False)
    population=pd.DataFrame(dict(physical_arc_id=g.arc_idx_to_directed_arc_id.astype(str),physical_arc_idx=np.arange(len(g.tail)),
        tail_physical_uid=g.node_idx_to_uid[g.tail].astype(str),head_physical_uid=g.node_idx_to_uid[g.head].astype(str),
        x45_flow=b.low_physical.current_flow,x45_cost_seconds=b.low_physical.current_cost_s,
        x200_flow=b.high_physical.current_flow,x200_cost_seconds=b.high_physical.current_cost_s,
        free_time_seconds=g.free_time*60.,used_capacity=b.low_physical.used_capacity,alpha=g.alpha,beta=g.beta))
    common=np.ones(len(g.tail),bool);summary=[]
    for sid in ctx.c['active_suites']:
        ctx.guard();m=mapping(ctx,src,sid,g);m.to_csv(dst/(sid+'_recovery.csv'),index=False,float_format='%.17g')
        recover=m.recoverable.to_numpy();common &= recover
        population[sid+'_recoverable']=recover;population[sid+'_logical_index']=m.logical_index
        population[sid+'_reason']=m.recovery_class_or_exclusion
        # Check direct source mapping. A compiled macro's reference common-flow
        # projection need not equal each original physical x45 member flow;
        # preserve that discrepancy as evidence rather than silently changing it.
        delta=np.abs(m.mapped_x45_logical_flow.to_numpy()[recover]-b.low_physical.current_flow[recover])
        direct=recover & m.source_logical_role.ne('MACRO').to_numpy()
        direct_delta=np.abs(m.mapped_x45_logical_flow.to_numpy()[direct]-b.low_physical.current_flow[direct])
        allowed=1e-8*np.maximum(1.,b.low_physical.current_flow[direct])
        if np.any(direct_delta>allowed):raise ValueError('Direct source flow identity fails x45: '+sid)
        for reason,v in m.groupby('recovery_class_or_exclusion',sort=True):
            ii=v.physical_arc_idx.to_numpy(np.int64)
            summary.append(dict(suite=sid,class_or_reason=reason,physical_arc_count=len(v),
                reference_flow_sum=float(b.low_physical.current_flow[ii].sum()),
                reference_flow_cost_sum=float(b.low_physical.current_flow[ii]@b.low_physical.current_cost_s[ii]),
                max_source_recovery_absolute_error=float(delta.max(initial=0.)) if v.recoverable.all() else None))
    population['common_all_five_suites']=common
    population['s03_common_formulations']=population.s03_recoverable
    # This complete inventory freezes the population before any predicted flow is loaded.
    population.to_csv(ctx.root/'physical_comparison_population.csv',index=False,float_format='%.17g')
    population.to_parquet(dst/'population.parquet',index=False)
    pd.DataFrame(summary).to_csv(dst/'mapping_coverage.csv',index=False,float_format='%.17g')
    coverage=[]
    for label,mask in [('COMMON_ALL_FIVE_SUITES',common),('S03_COMMON_FORMULATIONS',population.s03_recoverable.to_numpy())]:
        coverage.append(dict(population=label,physical_arc_count=int(mask.sum()),all_physical_arc_count=len(mask),
            x45_flow_share=float(population.x45_flow.to_numpy()[mask].sum()/population.x45_flow.sum()),
            x45_flow_cost_share=float((population.x45_flow.to_numpy()[mask]@population.x45_cost_seconds.to_numpy()[mask])/
                (population.x45_flow.to_numpy()@population.x45_cost_seconds.to_numpy()))))
    pd.DataFrame(coverage).to_csv(dst/'COMMON_POPULATION_COVERAGE.csv',index=False,float_format='%.17g')
    record=dict(status='FROZEN_BEFORE_PREDICTED_FLOW_ERRORS',epoch=time.time(),graph_identity=g.graph_hash,
        population_sha256=sha(ctx.root/'physical_comparison_population.csv'),native_sha256=sha(dst/'population.parquet'),
        common_populations=coverage,no_parallel_allocation_invented=True,no_interior_zero_imputation=True,
        geometry='Physical vertices exported separately and joined by stable directed arc ID',
        interpretation='Recovered physical arc flows on declared subset; not a certificate of globally feasible physical lifting')
    write_json(dst/'POPULATION_FREEZE.json',record)
    return record


def _compare(ctx,args):
    dst=ctx.root/'physical_population';freeze_record=read_json(dst/'POPULATION_FREEZE.json')
    ctx.bind(dst/'population.parquet','PREOUTCOME_COMMON_PHYSICAL_POPULATION',freeze_record['native_sha256'])
    pop=pd.read_parquet(dst/'population.parquet');summaries=[];states=[]
    rows=cases(ctx)
    for method,job in [('CFW','C02_PHYSICAL_CFW'),('FW','C03_PHYSICAL_FW'),('MSA','C04_PHYSICAL_MSA')]:
        rows.append(dict(scenario='PHYSICAL_'+method,suite='PHYSICAL',root=str(ctx.root/'runs'/job)))
    for row in rows:
        ctx.guard();root=Path(row['root']);rp=root/'RESULT.json'
        progress(dst,'COMPARISON_STATES.json',states,active=dict(row,attempt_stage='CASE_ADMISSION_AND_FLOW_RECOVERY'))
        if not rp.exists():
            states.append(dict(**row,**missing_state(ctx,root)));progress(dst,'COMPARISON_STATES.json',states);continue
        r=read_json(rp);ctx.bind(rp,'FLOW_COMPARISON_SOURCE_RESULT')
        if not r.get('final_certificate_fresh') or r.get('stop_reason','').startswith('STOP_'):
            states.append(dict(**row,status='NO_ADMITTED_STATE',stop_reason=r.get('stop_reason'),
                final_certificate_fresh=r.get('final_certificate_fresh'),
                diagnostic_final_RG_SPTT=(r.get('final_certificate') or {}).get('RG_SPTT')))
            progress(dst,'COMPARISON_STATES.json',states);continue
        fp=root/'final_flow.npz';ctx.bind(fp,'FROZEN_FLOW_COMPARISON_STATE',r.get('final_flow_sha256',r.get('current_flow_file_sha256')))
        f=read_npz(fp)['flow'];sid=row['suite']
        certificate=r['final_certificate'];certificate_hash=certificate.get('state_hash',r.get('state_hash'))
        if not certificate_hash or array_hash({'flow':f})!=certificate_hash:raise ValueError('Comparison flow differs from its fresh certificate')
        if sid=='PHYSICAL':
            if r['identity']['graph_identity']!=freeze_record['graph_identity'] or len(f)!=len(pop):
                raise ValueError('Physical comparison graph/order differs from frozen population')
            if certificate.get('conservation_passed') is not True:raise ValueError('Physical comparison lacks final conservation evidence')
            recovered=f
        else:
            graph_path=ctx.prior/'graphs'/sid/'IDENTITY_AND_COVERAGE_SELF_OD_V2.json'
            if not graph_path.exists():graph_path=graph_path.with_name('IDENTITY_AND_COVERAGE.json')
            ctx.bind(graph_path,'FROZEN_NATIVE_GRAPH_ORDER_FOR_PHYSICAL_RECOVERY');graph_meta=read_json(graph_path)
            for identity_key,metadata_key in [('graph','graph_identity'),('demand','demand_identity'),('support','support_identity')]:
                if r['identity'][identity_key]!=graph_meta[metadata_key]:raise ValueError('Native physical-recovery identity differs: '+identity_key)
            if len(f)!=graph_meta['packed_links']+graph_meta['regional_links']+graph_meta['endpoint_attachment_links']:
                raise ValueError('Native physical-recovery flow dimension differs')
            index=pop[sid+'_logical_index'].to_numpy(np.int64);recover=pop[sid+'_recoverable'].to_numpy()
            recovered=np.full(len(pop),np.nan);recovered[recover]=f[index[recover]]
        masks=[('COMMON_ALL_FIVE_SUITES',pop.common_all_five_suites.to_numpy())]
        if sid in ('PHYSICAL','s03'):masks.append(('S03_COMMON_FORMULATIONS',pop.s03_common_formulations.to_numpy()))
        for name,mask in masks:
            data=pop.loc[mask,['physical_arc_id','physical_arc_idx','tail_physical_uid','head_physical_uid',
                'x45_flow','x45_cost_seconds','x200_flow','x200_cost_seconds','free_time_seconds','used_capacity','alpha','beta']].copy()
            prediction=recovered[mask];ref=data.x45_flow.to_numpy()
            if not np.isfinite(prediction).all() or (prediction<0).any():raise ValueError('Invalid recovered flow')
            difference=prediction-ref;den=np.maximum(np.abs(prediction),np.abs(ref))
            signed=np.divide(difference,den,out=np.zeros_like(den),where=den>0)
            physical_cost=data.free_time_seconds.to_numpy()*(1+data.alpha.to_numpy()*(prediction/data.used_capacity.to_numpy())**data.beta.to_numpy())
            data['scenario']=row['scenario'];data['population']=name;data['predicted_physical_flow']=prediction
            data['signed_bounded_difference']=signed;data['absolute_flow_error']=np.abs(difference)
            data['physical_cost_at_prediction_seconds']=physical_cost
            data.to_csv(dst/(row['scenario']+'_'+name+'_ARC_ROWS.csv'),index=False,float_format='%.17g')
            summaries.append(dict(scenario=row['scenario'],suite=sid,population=name,physical_arc_count=len(data),
                flow_WMAPE=float(np.abs(difference).sum()/np.abs(ref).sum()),flow_MAE=float(np.abs(difference).mean()),
                flow_RMSE=float(np.sqrt(np.mean(difference**2))),reference_flow_sum=float(ref.sum()),
                x200_sensitivity_flow_WMAPE=float(np.abs(prediction-data.x200_flow.to_numpy()).sum()/data.x200_flow.sum()),
                x45_x200_reference_WMAPE=float(np.abs(ref-data.x200_flow.to_numpy()).sum()/data.x200_flow.sum()),
                reference_physical_population_TSTT=float(ref@data.x45_cost_seconds.to_numpy()),
                predicted_physical_population_TSTT=float(prediction@physical_cost),
                physical_TSTT_units='annual_kton_seconds; physical BPR only; no fixed coarse charge',
                stop_reason=r['stop_reason'],RG_SPTT=r['final_certificate']['RG_SPTT']))
        states.append(dict(**row,status='MEASURED',stop_reason=r['stop_reason']))
        pd.DataFrame(summaries).to_csv(dst/'PHYSICAL_FLOW_METRICS.csv',index=False,float_format='%.17g')
        progress(dst,'COMPARISON_STATES.json',states)
    progress(dst,'COMPARISON_STATES.json',states)
    measured=sum(r['status']=='MEASURED' for r in states)
    return dict(status='EXECUTED' if measured else 'NO_ADMITTED_STATES',measured_cases=measured,common_population_frozen=True)


def compare(ctx,args):
    from .diagnostic_handoffs import bounded
    plans=cases(ctx)
    for method,job in [('CFW','C02_PHYSICAL_CFW'),('FW','C03_PHYSICAL_FW'),('MSA','C04_PHYSICAL_MSA')]:
        plans.append(dict(scenario='PHYSICAL_'+method,suite='PHYSICAL',root=str(ctx.root/'runs'/job)))
    columns=['scenario','suite','population','physical_arc_count','flow_WMAPE','flow_MAE','flow_RMSE',
        'reference_flow_sum','x200_sensitivity_flow_WMAPE','x45_x200_reference_WMAPE',
        'reference_physical_population_TSTT','predicted_physical_population_TSTT','physical_TSTT_units',
        'stop_reason','RG_SPTT']
    return bounded(ctx,args,'physical_population',plans,['scenario'],'COMPARISON_STATES.json',
        {'PHYSICAL_FLOW_METRICS.csv':columns},_compare,reserve=60)
