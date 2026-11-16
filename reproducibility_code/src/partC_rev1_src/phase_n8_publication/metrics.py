"""Native ACNHN-coordinate comparisons and explicitly separate closure redistribution."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,read_npz,write_json,sha,array_hash
from partC_rev1_src.phase_n8_r4.path_metrics import bindings
from .context import sources,reopen_suite
from .reference import project_backbone,B_ROLES

LABELS={'REFERENCE_X45':'Physical CFW reference','PHYSICAL_CFW':'Physical CFW','PHYSICAL_FW':'Physical FW',
    'PHYSICAL_MSA':'Physical MSA','M1':'ACNHN–best nonconstant with safety fallback',
    'M2':'ACNHN–power-4','M3':'ACNHN–FFTT','REFERENCE_CLOSED':'Matched physical CFW closure reference'}


def paired_reachability(abstract,physical,demand,multiplicity):
    abstract=np.asarray(abstract,bool);physical=np.asarray(physical,bool);demand=np.asarray(demand,float)
    multiplicity=np.asarray(multiplicity,np.int64)
    if not (abstract.shape==physical.shape==demand.shape==multiplicity.shape):raise ValueError('Paired OD order/dimension differs')
    rows=[]
    for label,mask in [('BOTH_REACHABLE',abstract&physical),('ABSTRACT_ONLY',abstract&~physical),
                       ('PHYSICAL_ONLY',physical&~abstract),('NEITHER_REACHABLE',~abstract&~physical),
                       ('ABSTRACT_SERVED_TOTAL',abstract),('PHYSICAL_SERVED_TOTAL',physical)]:
        rows.append(dict(population=label,aggregated_rows=int(mask.sum()),original_rows=int(multiplicity[mask].sum()),
            demand_annual_ktons=float(demand[mask].sum())))
    return dict(same_served_OD_membership=bool(np.array_equal(abstract,physical)),rows=rows,
        interpretation='Full native-backbone discrepancy includes differing served demand when memberships differ; redistribution includes lost demand and is not solely rerouting.')


def compare_vectors(reference,predicted,keys,base,k4):
    ref=np.asarray(reference,float);pred=np.asarray(predicted,float);key=np.asarray(keys,bool)
    if ref.shape!=pred.shape or ref.shape!=key.shape or not len(ref):raise ValueError('Comparison coordinate shape differs')
    if not np.isfinite(ref).all() or not np.isfinite(pred).all() or (ref<0).any() or (pred<0).any():raise ValueError('Invalid comparison flows')
    difference=pred-ref;absolute=abs(difference);scale=np.maximum(abs(ref),abs(pred))
    per=pd.DataFrame(dict(lambda_reference=ref,lambda_estimate=pred,signed_error=difference,absolute_error=absolute,
        signed_bounded_error=np.divide(difference,scale,out=np.zeros_like(scale),where=scale>0),
        signed_relative_error=np.divide(difference,ref,out=np.full(len(ref),np.nan),where=ref>0),
        absolute_relative_error=np.divide(absolute,ref,out=np.full(len(ref),np.nan),where=ref>0),is_key_link=key))
    den=math.fsum(ref);per['backbone_wmape_contribution_percent']=100*absolute/den if den else np.nan
    result=[]
    for population,mask in [('BACKBONE',np.ones(len(ref),bool)),('KEY_LINKS',key)]:
        r=ref[mask];p=pred[mask];d=p-r;numerator=math.fsum(abs(d));denominator=math.fsum(r)
        if not len(r):raise ValueError('Empty comparison population')
        rc=float(r@(base[mask]+k4[mask]*r**4));pc=float(p@(base[mask]+k4[mask]*p**4))
        result.append(dict(population=population,link_count=len(r),reference_flow_sum=denominator,absolute_error_sum=numerator,
            wmape_percent=100*numerator/denominator if denominator else None,
            wmape_status='DEFINED' if denominator else 'UNDEFINED_ZERO_REFERENCE_TOTAL',mae=numerator/len(r),
            rmse=float(np.sqrt(np.mean(d*d))),mean_signed_error=float(np.mean(d)),reference_zero_link_count=int((r==0).sum()),
            reference_zero_positive_prediction_count=int(((r==0)&(p>0)).sum()),reference_backbone_cost=rc,predicted_backbone_cost=pc,
            backbone_cost_error_percent=100*abs(pc-rc)/rc if rc else None,flow_units='ANNUAL_KTON',cost_units='ANNUAL_KTON_SECONDS'))
    return result,per


def admit(ctx,path,*,suite=None,graph_identity=None):
    path=Path(path);ctx.bind(path/'RESULT.json','SAVED_ASSIGNMENT_RESULT');r=read_json(path/'RESULT.json')
    if not r.get('final_certificate_fresh') or not r.get('final_certificate'):
        raise ValueError('No fresh final certificate: '+str(path))
    if r['stop_reason'] not in {'CONVERGED','CONVERGED_REPRESENTED_DEMAND_ONLY','TIME_LIMIT'}:
        raise ValueError('State not admitted: '+r['stop_reason'])
    fp=path/'final_flow.npz';ctx.bind(fp,'CERTIFIED_CURRENT_FLOW',r.get('final_flow_sha256',r.get('current_flow_file_sha256')))
    a=read_npz(fp);flow=a['flow'];cert=r['final_certificate']
    state_hash=array_hash({'flow':flow})
    if state_hash!=cert.get('state_hash',r.get('state_hash')):raise ValueError('Final flow/certificate state differs')
    required=np.asarray([cert[k] for k in ('raw_signed_gap','arithmetic_allowance','SPTT','RG_SPTT')],float)
    if not np.isfinite(required).all() or cert['arithmetic_allowance']<0 or cert['SPTT']<0:
        raise ValueError('Missing or invalid signed gap evidence')
    if cert['raw_signed_gap'] < -cert['arithmetic_allowance']:raise ValueError('Material negative gap cannot be admitted')
    expected_gap=cert['raw_signed_gap']/cert['SPTT'] if cert['SPTT']>0 else 0.
    if cert['SPTT']==0 and abs(cert['raw_signed_gap'])>cert['arithmetic_allowance']:
        raise ValueError('Nonzero gap has zero SPTT')
    if not np.isclose(cert['RG_SPTT'],expected_gap,rtol=1e-9,atol=1e-12):raise ValueError('Relative and raw gaps differ')
    if r['stop_reason'].startswith('CONVERGED') and cert['RG_SPTT']>.001:raise ValueError('Convergence label exceeds tolerance')
    if suite is not None:
        expected=graph_identity or suite['graph'].identity
        if r['identity']['graph']!=expected or r['identity']['support']!=suite['support_identity']:
            raise ValueError('Prediction graph/support differs from comparison')
        if len(flow)!=len(suite['graph'].edge_ids):raise ValueError('Prediction native dimension differs')
        if 'edge_ids' in a and not np.array_equal(a['edge_ids'],suite['graph'].edge_ids):raise ValueError('Prediction edge IDs reordered')
    return flow,r,state_hash


def final_new_case(ctx,scenario):
    matches=[]
    for p in (ctx.root/'runs').glob('*/RESULT.json'):
        r=read_json(p)
        if r.get('scenario')==scenario:matches.append((p.parent,r))
    if not matches:raise FileNotFoundError('Required assignment absent: '+scenario)
    if scenario.startswith('M1_'):
        current=read_json(ctx.root/'registries/M1_CURRENT.json')
        matches=[x for x in matches if x[1].get('final_registry_identity')==current['identity']]
        if not matches:raise ValueError('M1 open/closed final registry mismatch; final-registry rerun required: '+scenario)
    def start(item):
        row=item[1]
        return row['clocks']['run_start_epoch'] if 'clocks' in row else row['identity']['job_start_epoch']
    return max(matches,key=start)[0]


def history_rows(ctx,path,method,sid,network_state):
    path=Path(path);r=read_json(path/'RESULT.json')
    if 'epochs' in r:
        parts=[]
        for epoch in r['epochs']:
            p=Path(epoch['path'])/'HISTORY.json';ctx.bind(p,'FIXED_EPOCH_CONVERGENCE_HISTORY')
            f=pd.DataFrame(read_json(p))
            if len(f):
                f['epoch']=epoch['epoch'];f['registry_identity']=epoch['registry'];f['final_epoch']=str(Path(epoch['path']))==r['final_epoch_path'];parts.append(f)
        f=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    elif (path/'HISTORY.json').exists():
        ctx.bind(path/'HISTORY.json','REUSED_CONVERGENCE_HISTORY');f=pd.DataFrame(read_json(path/'HISTORY.json'))
        f['epoch']=0;f['final_epoch']=True
    elif (path/'fresh_certificates.jsonl').exists():
        ctx.bind(path/'fresh_certificates.jsonl','PHYSICAL_FRESH_CERTIFICATE_HISTORY')
        f=pd.read_json(path/'fresh_certificates.jsonl',lines=True);f['epoch']=0;f['final_epoch']=True
    else:return pd.DataFrame()
    f['method_id']=method;f['suite_id']=sid;f['network_state']=network_state
    f['job_id']=path.name;f['assignment_role']='FINAL_COMPARISON'
    if 'state_iteration' in f:
        f['iteration']=f.get('iteration',pd.Series(index=f.index,dtype=float)).fillna(f.state_iteration)
    f['registry_epoch']=f['epoch'];f['termination_status']=r['stop_reason']
    f['state_id']=f['state_hash'] if 'state_hash' in f else None
    f['certificate_fresh']=f['fresh'] if 'fresh' in f else True
    f['raw_gap']=f['raw_signed_gap'] if 'raw_signed_gap' in f else None
    f['time_scope']='FIXED_EPOCH_ELAPSED_SECONDS'
    return f


def job_timing(ctx,path,r):
    if path is None:return None,'REFERENCE_ITERATION_TIME_REPORTED_SEPARATELY'
    path=Path(path)
    candidates=[ctx.root/'jobs'/path.name/'EXIT.json',ctx.r4/'jobs'/path.name/'EXIT.json']
    for p in candidates:
        if p.exists():
            ctx.bind(p,'COMPLETE_SUPERVISED_JOB_TIME');return read_json(p)['elapsed_seconds'],'COMPLETE_SUPERVISED_PROCESS_WALL'
    p=ctx.prior/'receipts'/(path.name+'_JOB.json')
    if p.exists():
        ctx.bind(p,'COMPLETE_REUSED_SUPERVISED_JOB_TIME');return read_json(p)['result']['elapsed_seconds'],'COMPLETE_SUPERVISED_PROCESS_WALL'
    return r.get('elapsed_seconds'),'REPORTED_ASSIGNMENT_WALL_INCOMPLETE_BOUNDARY'


def run(ctx,args):
    src=sources(ctx);phys=bindings(ctx);dst=ctx.root/'metrics'
    manifest_path=ctx.prior/'CAMPAIGN_MANIFEST.json';ctx.bind(manifest_path,'REUSED_FFTT_SCENARIOS')
    fftt={r['suite']:r for r in read_json(manifest_path)['scenarios'] if r['scenario_id'].startswith('FFTT_')}
    summaries=[];states=[];histories=[];redistribution=[];closed_summaries=[]
    for sid in ctx.c['active_suites']:
        ctx.guard();suite,od,balance,coverage=reopen_suite(ctx,src,sid)
        rd=ctx.root/'reference'/sid;fr=pd.read_csv(rd/'backbone_reference.csv',float_precision='round_trip');ctx.bind(rd/'backbone_reference.csv','FROZEN_FULL_BACKBONE_REFERENCE')
        key_frame=pd.read_csv(rd/'key_links_200.csv',float_precision='round_trip')
        ctx.bind(rd/'key_links_200.csv','PREOUTCOME_FIXED_200_KEYS',read_json(rd/'KEY_SELECTION_FREEZE.json')['sha256'])
        bi=fr.native_index.to_numpy(np.int64);keys=fr.abstract_link_id.isin(key_frame.abstract_link_id).to_numpy()
        if len(set(fr.abstract_link_id))!=len(bi) or not np.array_equal(suite['graph'].edge_ids[bi],fr.abstract_link_id):
            raise ValueError('Reference backbone IDs/order differs')
        if int(keys.sum())!=200:raise ValueError('Frozen key count differs')
        expected_b=np.flatnonzero(suite['links'].role.isin(B_ROLES).to_numpy())
        if not np.array_equal(bi,expected_b):raise ValueError('Complete typed backbone population differs')
        ref=fr.lambda_reference.to_numpy();a=suite['packed'].arrays
        if not np.array_equal(ref,suite['fixed_x45'][bi]):raise ValueError('Reference values changed after freeze')
        base=(a['tau_s']+a['fixed_internal_H_s']+a['tail_half_charge_s']+a['head_half_charge_s'])[bi];k4=a['k4'][bi]
        expected=suite['packed'].evaluate(suite['fixed_x45'][:suite['packed_count']],'cost')[0][bi]
        if not np.allclose(base+k4*ref**4,expected,rtol=2e-14,atol=1e-9):raise ValueError('Backbone fixed charge/cost semantics differ')
        directory=dst/sid;directory.mkdir(exist_ok=False)
        fr[['suite_id','abstract_link_id','native_index','role','tail_longitude','tail_latitude','head_longitude','head_latitude',
            'tail_x','tail_y','head_x','head_y','longitude','latitude','x','y','geometry_source']].to_csv(directory/'backbone_geometry.csv',index=False,float_format='%.17g')
        coeff=fr[['abstract_link_id']].copy();coeff['base_seconds']=base;coeff['k4']=k4;coeff.to_csv(directory/'backbone_cost_coefficients.csv',index=False,float_format='%.17g')
        ctx.bind(Path(fftt[sid]['assignment_root'])/'RESULT.json','EXACT_COMPLETED_FFTT_RESULT',fftt[sid]['pilot_result_sha256'])
        coarse,record,state=admit(ctx,fftt[sid]['assignment_root'],suite=suite)
        cases=[('REFERENCE_X45',ref,None,dict(stop_reason='REFERENCE_X45',final_certificate={'RG_SPTT':ctx.c['reference_gap']},elapsed_seconds=None),fr.reference_id.iloc[0]),
               ('M3',coarse,Path(fftt[sid]['assignment_root']),record,state)]
        if sid=='s03':
            incidence_path=rd/'backbone_source_incidence.parquet';ctx.bind(incidence_path,'VALIDATED_FORWARD_INCIDENCE');incidence=pd.read_parquet(incidence_path)
            for method,job in [('PHYSICAL_CFW','C02_PHYSICAL_CFW'),('PHYSICAL_FW','C03_PHYSICAL_FW'),('PHYSICAL_MSA','C04_PHYSICAL_MSA')]:
                path=ctx.r4/'runs'/job;physical_flow,r,h=admit(ctx,path)
                if r['identity']['graph_identity']!=phys.graph.graph_hash:raise ValueError('Reused physical graph identity differs')
                projected,spread=project_backbone(incidence,suite['links'],physical_flow)
                spread.to_csv(directory/(method+'_series_projection_diagnostics.csv'),index=False,float_format='%.17g')
                cases.append((method,projected[bi],path,r,h))
            for method in ('M1','M2'):
                path=final_new_case(ctx,method+'_OPEN');flow,r,h=admit(ctx,path,suite=suite);cases.append((method,flow,path,r,h))
        opened={}
        for method,flow,path,r,h in cases:
            pred=flow if len(flow)==len(bi) else flow[bi];opened[method]=(flow,r,h,path)
            result,per=compare_vectors(ref,pred,keys,base,k4)
            per.insert(0,'abstract_link_id',fr.abstract_link_id);per.insert(0,'method_id',method);per.insert(0,'suite_id',sid)
            per['state_id']=h;per['reference_id']=fr.reference_id.iloc[0];per['graph_identity']=suite['graph'].identity
            per['mapping_identity']=fr.mapping_identity.iloc[0];per['model_registry_sha256']=r.get('final_registry_identity',r.get('identity',{}).get('registry','NOT_REGIONAL_MODEL'))
            per['flow_unit']='ANNUAL_KTON'
            per.to_csv(directory/(method+'_open_per_link.csv'),index=False,float_format='%.17g')
            online,timing_scope=job_timing(ctx,path,r)
            for x in result:
                x.update(suite_id=sid,method_id=method,method_label=LABELS[method],network_state='OPEN',
                    reference_id=fr.reference_id.iloc[0],state_id=h,graph_identity=suite['graph'].identity,mapping_identity=fr.mapping_identity.iloc[0],
                    status=r['stop_reason'],RG_SPTT=r['final_certificate']['RG_SPTT'],online_seconds=online,timing_scope=timing_scope,
                    represented_demand=od.total,original_demand=float(od.demand.sum()),
                    accepted_iterations=r.get('accepted_iterations'),effective_registry=r.get('final_registry_identity',r.get('identity',{}).get('registry')),
                    registry_identity=r.get('final_registry_identity',r.get('identity',{}).get('registry','NOT_REGIONAL_MODEL')))
                if method=='REFERENCE_X45' and any(x[k]!=0 for k in ('wmape_percent','mae','rmse','mean_signed_error','backbone_cost_error_percent')):
                    raise ValueError('Reference self-join metric is not exactly zero')
                summaries.append(x)
            states.append(dict(suite_id=sid,method_id=method,network_state='OPEN',path=str(path) if path else 'FIXED_X45',state_id=h,
                               fresh_certificate=True,status=r['stop_reason']))
            if path:histories.append(history_rows(ctx,path,method,sid,'OPEN'))
        if sid=='s03':
            # Keep discarded discovery epochs AND its accepted rescued state;
            # neither is the independently initialized final measurement.
            selected_m1=opened['M1'][3]
            for old_result in sorted((ctx.root/'runs').glob('*/RESULT.json')):
                if old_result.parent==selected_m1:continue
                rr=read_json(old_result)
                if rr.get('scenario')=='M1_OPEN':
                    old_history=history_rows(ctx,old_result.parent,'M1',sid,'OPEN')
                    old_history['assignment_role']='RECOVERY_CAMPAIGN';histories.append(old_history)
            mask=read_npz(ctx.root/'closure/availability.npz')['unavailable'];mask_b=mask[bi]
            closed_meta=read_json(ctx.root/'closure/CLOSED_REACHABILITY.json')
            closed_physical_path=final_new_case(ctx,'P_CLOSED');pf,pr,ph=admit(ctx,closed_physical_path)
            mask_record=read_json(ctx.root/'closure/MASK_FREEZE.json')
            if pr['identity']['base_graph_identity']!=phys.graph.graph_hash or pr['identity']['mask_sha256']!=mask_record['mask_sha256']:
                raise ValueError('Physical closure base graph or exact represented mask differs')
            p_membership=closed_physical_path/'closed_demand_membership.npz'
            ctx.bind(p_membership,'PHYSICAL_CLOSED_OD_MEMBERSHIP',pr['identity']['closed_membership_sha256'])
            a_membership=ctx.root/'closure/closed_fftt_initial.npz';ctx.bind(a_membership,'ABSTRACT_CLOSED_OD_MEMBERSHIP')
            paired=paired_reachability(read_npz(a_membership)['serve'],read_npz(p_membership)['serve'],od.demand,phys.od.original_row_multiplicity)
            pd.DataFrame(paired['rows']).to_csv(dst/'closed_OD_membership_comparison.csv',index=False,float_format='%.17g')
            write_json(dst/'CLOSED_OD_COMPARABILITY.json',paired)
            closed_ref,spread=project_backbone(incidence,suite['links'],pf);closed_ref=closed_ref[bi]
            if np.any(closed_ref[mask_b]!=0):raise ValueError('Matched closed projection does not have zero disabled coordinates')
            spread.to_csv(directory/'REFERENCE_CLOSED_series_projection_diagnostics.csv',index=False,float_format='%.17g')
            closed_reference=fr.copy();closed_reference['lambda_reference']=closed_ref;closed_reference['reference_id']=ph
            closed_reference['graph_identity']=closed_meta['graph_identity'];closed_reference.to_csv(directory/'closed_backbone_reference.csv',index=False,float_format='%.17g')
            closure_cases=[('REFERENCE_CLOSED',closed_ref,closed_physical_path,pr,ph)]
            for method in ('M1','M2','M3'):
                path=final_new_case(ctx,method+'_CLOSED');flow,r,h=admit(ctx,path,suite=suite,graph_identity=closed_meta['graph_identity'])
                if method=='M1' and r['final_registry_identity']!=opened[method][1]['final_registry_identity']:
                    raise ValueError('Paired M1 registry differs')
                if np.any(flow[mask]!=0):raise ValueError('Unavailable native flow nonzero')
                closure_cases.append((method,flow,path,r,h))
                old=opened[method][0];old_b=old[bi];new_b=flow[bi]
                for name,take in [('ALL_BACKBONE',np.ones(len(bi),bool)),('BACKBONE_EXCLUDING_CLOSED',~mask_b)]:
                    numerator=float(abs(new_b[take]-old_b[take]).sum());den=float(old_b[take].sum())
                    redistribution.append(dict(method_id=method,population=name,link_count=int(take.sum()),
                        absolute_change_sum=numerator,open_flow_sum=den,redistribution_percent=100*numerator/den if den else None,
                        interpretation='OPEN_TO_CLOSED_REDISTRIBUTION_NOT_REFERENCE_ACCURACY',
                        registry_identity=r.get('final_registry_identity')))
                pair_count=len(suite['pairs']);sl=slice(suite['packed_count'],suite['packed_count']+pair_count)
                pairs=suite['pairs'];nodes=suite['nodes'].set_index('node_id')
                regional=pd.DataFrame(dict(pair_id=pairs.pair_id,region_id=pairs.region_id,open_flow=old[sl],closed_flow=flow[sl]))
                for prefix,column in [('tail','source_terminal_id'),('head','target_terminal_id')]:
                    regional[prefix+'_longitude']=pairs[column].map(nodes.longitude);regional[prefix+'_latitude']=pairs[column].map(nodes.latitude)
                regional['signed_change']=regional.closed_flow-regional.open_flow
                regional.to_csv(directory/(method+'_regional_open_closed.csv'),index=False,float_format='%.17g')
                redistribution.append(dict(method_id=method,population='ACTIVE_REGIONAL_PAIRS',link_count=pair_count,
                    absolute_change_sum=float(abs(flow[sl]-old[sl]).sum()),open_flow_sum=float(old[sl].sum()),
                    redistribution_percent=100*float(abs(flow[sl]-old[sl]).sum())/float(old[sl].sum()) if float(old[sl].sum()) else None,
                    net_traversal_change=float(flow[sl].sum()-old[sl].sum()),interpretation='REGIONAL_TRAVERSAL_REDISTRIBUTION'))
                delta=fr[['abstract_link_id']].copy();delta['open_flow']=old_b;delta['closed_flow']=new_b;delta['signed_change']=new_b-old_b
                delta['unavailable']=mask_b;delta['suite_id']=sid;delta['method_id']=method
                delta['registry_hash']=r['final_registry_identity'];delta['mask_hash']=mask_record['mask_sha256']
                delta['flow_open']=old_b;delta['flow_closed']=new_b;delta['flow_delta']=new_b-old_b;delta['is_closed']=mask_b
                delta.to_csv(directory/(method+'_backbone_open_closed.csv'),index=False,float_format='%.17g')
            for method,flow,path,r,h in closure_cases:
                pred=flow if len(flow)==len(bi) else flow[bi];results,per=compare_vectors(closed_ref,pred,keys,base,k4)
                per.insert(0,'abstract_link_id',fr.abstract_link_id);per.insert(0,'method_id',method);per.insert(0,'suite_id',sid)
                per['state_id']=h;per['reference_id']=ph;per['graph_identity']=closed_meta['graph_identity']
                per['mapping_identity']=fr.mapping_identity.iloc[0];per['model_registry_sha256']=r.get('final_registry_identity','PHYSICAL_REFERENCE')
                per['flow_unit']='ANNUAL_KTON'
                per.to_csv(directory/(method+'_closed_per_link.csv'),index=False,float_format='%.17g')
                online,timing_scope=job_timing(ctx,path,r)
                for x in results:
                    x.update(suite_id=sid,method_id=method,method_label=LABELS[method],network_state='CLOSED',reference_id=ph,state_id=h,
                        graph_identity=closed_meta['graph_identity'],mapping_identity=fr.mapping_identity.iloc[0],
                        status=r['stop_reason'],RG_SPTT=r['final_certificate']['RG_SPTT'],online_seconds=online,timing_scope=timing_scope,
                        reference_status=pr['stop_reason'],reference_RG_SPTT=pr['final_certificate']['RG_SPTT'],
                        registry_identity=r.get('final_registry_identity','NOT_REGIONAL_MODEL'),
                        represented_demand=r.get('represented_demand'),original_demand=float(od.demand.sum()),
                        reference_represented_demand=pr['represented_demand'],served_OD_membership_equal=paired['same_served_OD_membership'],
                        comparison_demand_scope='COMMON_REACHABLE_MEMBERSHIP' if paired['same_served_OD_membership'] else 'DIFFERENT_REACHABLE_MEMBERSHIP_FULL_BACKBONE_DISCREPANCY')
                    closed_summaries.append(x)
                histories.append(history_rows(ctx,path,method,sid,'CLOSED'))
                states.append(dict(suite_id=sid,method_id=method,network_state='CLOSED',path=str(path),state_id=h,
                                   fresh_certificate=True,status=r['stop_reason']))
        pd.DataFrame(summaries).to_csv(dst/'open_summary_metrics.csv',index=False,float_format='%.17g')
    pd.DataFrame(closed_summaries).to_csv(dst/'closed_summary_metrics.csv',index=False,float_format='%.17g')
    pd.DataFrame(redistribution).to_csv(dst/'closure_redistribution.csv',index=False,float_format='%.17g')
    pd.concat([f for f in histories if len(f)],ignore_index=True).to_csv(dst/'all_fixed_epoch_histories.csv',index=False,float_format='%.17g')
    write_json(dst/'ADMITTED_STATES.json',states)
    result=dict(status='PASS',comparison_space='ACNHN_FORWARD_REFERENCE',predicted_physical_recovery_used=False,
        open_metric_rows=len(summaries),closed_metric_rows=len(closed_summaries),admitted_states=len(states),
        M1_open_closed_registry_equal=True,fixed200_used=True,reference_self_error_zero=True,
        all_original_backbone_zero_flow_rows_retained=True,closed_reference_status=pr['stop_reason'])
    write_json(dst/'RESULT.json',result)
    return result
