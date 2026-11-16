"""Low-discovery ranking, isolated SELECT decision and complete adverse summaries."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from partC_rev1_src.phase5.bacn_charges import StaticPathProjector
from partC_rev1_src.phase5.campaign_io import sha,frame_write,write_json,file_record
from .queries import paths_from_rows


def rank_B(ctx,b,t,frames,physical_rows,charges):
    sample=pd.read_csv(ctx.out/'sampling/frozen_od_sample.csv')
    old=sample.loc[sample.source_set.eq('DISCOVERY_OLD200')].set_index('source_sample_od_id')
    frozen=pd.read_parquet(b.sources['frozen_path_sample'])
    newpaths=paths_from_rows(ctx.out,physical_rows.loc[physical_rows.source_set.eq('DISCOVERY_OLD200')])
    ch=charges.set_index('bacn_id').loc[list(t.bacn_ids)]
    chi=ch.chi_s.to_numpy(float);scores=np.zeros(len(chi));observations=np.zeros(len(chi),np.int64)
    contributions=[];projector=StaticPathProjector(b.graph,t,frames)
    total=old.demand.sum()
    for role in ['ORIGINAL_FREE_FLOW_WITNESS','NEW_LOW_PHYSICAL_SHORTEST']:
        for ordinal,row in enumerate(frozen.itertuples(index=False)):
            selected=old.loc[str(row.sample_od_id)]
            arcs=np.asarray(json.loads(row.arc_indices_json),np.int64) if role=='ORIGINAL_FREE_FLOW_WITNESS' else newpaths[selected.stable_od_key]
            _,weights=projector.project(arcs)
            owner=t.arc_internal_bacn_local[arcs];valid=owner>=0
            exposed=np.bincount(owner[valid],weights=b.low_physical.current_cost_s[arcs[valid]],minlength=len(chi))
            visit=np.zeros(len(chi))
            for p,w in weights.items():visit[p]=w
            per=abs(visit*chi-exposed)
            w=float(selected.demand)/total/2.
            scores+=w*per;seen=np.flatnonzero((visit!=0)|(exposed!=0));observations[seen]+=1
            contributions.extend(dict(stable_od_key=selected.stable_od_key,source_role=role,bacn_id=t.bacn_ids[p],
                normalized_split_OD_weight=w,half_incidence=visit[p],chi_s=chi[p],suppressed_physical_time_s=exposed[p],
                absolute_charge_error_s=per[p],score_contribution=w*per[p]) for p in seen)
            if ordinal%64==0:ctx.mark(f'F3 low-discovery risk {role} {ordinal+1}/200')
    flags=set();flag_rows=[]
    for path in sorted((ctx.out/'results/configurations').glob('s03_LOW_*_PRIMARY_M0.csv')):
        d=pd.read_csv(path)
        for r in d.loc[d.source_set.eq('DISCOVERY_OLD200')&d.lift_status.eq('UNLIFTABLE')].itertuples(index=False):
            for p in json.loads(r.failed_B_ids_json):
                flags.add(p);flag_rows.append(dict(bacn_id=p,stable_od_key=r.stable_od_key,graph_variant=r.graph_variant,
                    source_set=r.source_set,state='LOW',failure_from_physical_nodes_json=r.failure_from_physical_nodes_json,
                    failure_to_physical_nodes_json=r.failure_to_physical_nodes_json))
    ranking=pd.DataFrame(dict(bacn_id=t.bacn_ids,low_discovery_score=scores,infeasible_discovery_transition=[p in flags for p in t.bacn_ids],
        discovery_role_observations=observations,chi_s=chi,E_L=ch.E_L.to_numpy(),V_L=ch.V_L.to_numpy(),structural_seed_s=ch.structural_seed_s.to_numpy()))
    ranking=ranking.sort_values(['infeasible_discovery_transition','low_discovery_score','bacn_id'],ascending=[False,False,True],kind='stable').reset_index(drop=True)
    ranking.insert(0,'rank',np.arange(1,len(ranking)+1));ranking['selection_source']='LOW_DISCOVERY_ONLY'
    frame_write(ctx.out/'refinement/low_frozen_B_ranking.csv',ranking)
    frame_write(ctx.out/'refinement/discovery_risk_contributions.parquet',pd.DataFrame(contributions))
    frame_write(ctx.out/'refinement/infeasible_discovery_B_transitions.csv',pd.DataFrame(flag_rows,columns=['bacn_id','stable_od_key','graph_variant','source_set','state','failure_from_physical_nodes_json','failure_to_physical_nodes_json']))
    return ranking


def weighted_quantile(values,weights,q):
    v=np.asarray(values,float);w=np.asarray(weights,float)
    mask=np.isfinite(v)&np.isfinite(w)&(w>0)
    if not mask.any():return None
    v,w=v[mask],w[mask];order=np.argsort(v,kind='stable');v,w=v[order],w[order]
    return float(v[min(int(np.searchsorted(np.cumsum(w),q*w.sum(),side='left')),len(v)-1)])


def request_selection_counts(requests,occurrences):
    counts=occurrences.calibration_request_id.dropna().value_counts()
    result=requests[['request_id','runtime_class','status','is_exact']].copy()
    result['emitted_link_occurrence_count']=result.request_id.map(counts).fillna(0).astype(np.int64)
    result['selected_for_emission']=result.emitted_link_occurrence_count.gt(0)
    result['numerical_acceptance']=result.status.eq('ACCEPT')
    if (result.selected_for_emission & ~result.numerical_acceptance).any():raise ValueError('Rejected request emitted as runtime model')
    return result


def metrics(group):
    result=dict(OD_count=len(group),new_unreachable_count=int(group.new_unreachable.sum()),
        unliftable_count=int(group.lift_status.eq('UNLIFTABLE').sum()),verified_lift_count=int(group.lift_status.eq('VERIFIED').sum()),
        unreachable_count=int((~group.reachable.astype(bool)).sum()),
        unliftable_proportion=float(group.lift_status.eq('UNLIFTABLE').mean()))
    weights=pd.to_numeric(group.ratio_weight).to_numpy(float)
    for prefix,column in [('distance','absolute_relative_distance_error'),('regret','regret'),('absolute_seconds','absolute_distance_error_s')]:
        v=pd.to_numeric(group[column],errors='coerce').to_numpy(float);mask=np.isfinite(v)
        result[prefix+'_finite_count']=int(mask.sum())
        result[prefix+'_unweighted_mean']=float(v[mask].mean()) if mask.any() else None
        result[prefix+'_weighted_mean']=float(np.average(v[mask],weights=weights[mask])) if mask.any() else None
        result[prefix+'_unweighted_p50']=float(np.quantile(v[mask],.5)) if mask.any() else None
        result[prefix+'_unweighted_p95']=float(np.quantile(v[mask],.95)) if mask.any() else None
        result[prefix+'_weighted_p95']=weighted_quantile(v,weights,.95)
        result[prefix+'_max']=float(v[mask].max()) if mask.any() else None
    result['regret_population']='CONDITIONAL_ON_VERIFIED_LIFTS; all unliftable/unreachable counts retained'
    result['weighting']='demand/(n_h/N_h); fixed stratified sample estimate, not an exact national total or confidence interval'
    return result


def screening(m):
    # Both weighted and unweighted p95 are reported; the weighted quantile governs.
    return bool(m['new_unreachable_count']==0 and m['unliftable_count']==0 and m['verified_lift_count']==m['OD_count'] and
        all(m[k] is not None and m[k]<=bound for k,bound in [
            ('distance_weighted_mean',.02),('distance_weighted_p95',.05),('distance_max',.2),
            ('regret_weighted_mean',.02),('regret_weighted_p95',.05),('regret_max',.2)]))


def candidate_summary(ctx,state='LOW',source_set=None):
    rows=[]
    for m in [0,16,64,256]:
        variant='MEMBER_B_R_FINE' if m==0 else 'RESTORED_B_R_FINE'
        for charge in ['PRIMARY','STRUCTURAL']:
            path=ctx.out/f'results/configurations/s03_{state}_{variant}_{charge}_M{m}.csv'
            if not path.exists():continue
            df=pd.read_csv(path)
            for role,g in df.groupby('source_set'):
                if source_set is not None and role!=source_set:continue
                result=metrics(g)
                rows.append(dict(restored_B_count=m,charge_variant=charge,state=state,source_set=role,
                    graph_variant='RESTORED_B_R_FINE',M0_alias_of='MEMBER_B_R_FINE' if m==0 else '',
                    LOW_SCREEN_PASS=screening(result) if state=='LOW' and role=='NEW_SELECT128' else None,**result))
    return pd.DataFrame(rows)


def freeze_selection(ctx):
    selection=candidate_summary(ctx,'LOW','NEW_SELECT128')
    frame_write(ctx.out/'refinement/LOW_SELECT_only_candidate_metrics.csv',selection)
    eligible=selection.loc[selection.charge_variant.eq('PRIMARY') & selection.LOW_SCREEN_PASS.astype(bool)]
    m=None if eligible.empty else int(eligible.restored_B_count.min())
    result=dict(provisional_smallest_candidate=m,readiness='MODEL_REVISION_REQUIRED' if m is None else 'PENDING_INDEPENDENT_AUDIT',
        decision_source='LOW_NEW_SELECT128_PRIMARY_ONLY',screen_quantile='Stratified demand-weighted empirical inverse-CDF p95',
        selection_metrics_sha256=sha(ctx.out/'refinement/LOW_SELECT_only_candidate_metrics.csv'),
        ladder_sha256=sha(ctx.out/'refinement/candidate_ladder_frozen.json'),sample_sha256=sha(ctx.out/'sampling/frozen_od_sample.csv'),
        all_candidates_preserved=True,HIGH_and_AUDIT_used_for_selection=False,production_adoption=False,phase6_authorized=False)
    path=ctx.out/'refinement/selection.json'
    if path.exists() and json.loads(path.read_text())!=result:raise ValueError('Attempt to alter frozen LOW selection')
    write_json(path,result)
    return result


def interface_impact(ctx,b,t,frames,ladder):
    terminals=frames['consolidated_terminals'];fine=frames['terminal_fine_members']
    counts=fine.groupby('terminal_id').size()
    member_frame=fine[['terminal_id','partC_rev1_node_uid']].copy()
    member_frame['partC_rev1_node_uid']=member_frame.partC_rev1_node_uid.astype(str)
    members_by_terminal=member_frame.groupby('terminal_id',sort=False).partC_rev1_node_uid.agg(list).to_dict()
    nodecounts=np.bincount(t.node_to_bacn_local[t.node_to_bacn_local>=0],minlength=len(t.bacn_ids))
    arccounts=np.bincount(t.arc_internal_bacn_local[t.arc_internal_bacn_local>=0],minlength=len(t.bacn_ids))
    bi={v:i for i,v in enumerate(t.bacn_ids)}
    rows=[]
    for m in [16,64,256]:
        selected=set(ladder[str(m)])
        changed=terminals.loc[terminals.parent_object_id.isin(selected)]
        for region,rg in changed.groupby('region_id'):
            orig=terminals.loc[terminals.region_id.eq(region)]
            src=int(orig.can_exit_region.sum());dst=int(orig.can_enter_region.sum())
            dsrc=int(sum((counts[r.terminal_id]-1)*bool(r.can_exit_region) for r in rg.itertuples()))
            ddst=int(sum((counts[r.terminal_id]-1)*bool(r.can_enter_region) for r in rg.itertuples()))
            for r in rg.itertuples(index=False):
                p=bi[r.parent_object_id]
                members=members_by_terminal[r.terminal_id]
                rows.append(dict(candidate_M=m,bacn_id=r.parent_object_id,region_id=region,terminal_id=r.terminal_id,
                    original_terminal_type=r.terminal_type,physical_B_nodes_retained=int(nodecounts[p]),suppressed_B_arcs_restored=int(arccounts[p]),
                    fine_portal_states=len(members),fine_physical_node_ids_json=json.dumps(members),
                    terminal_state_dimension_increase=max(0,len(members)-1),can_exit_region=bool(r.can_exit_region),can_enter_region=bool(r.can_enter_region),
                    region_original_source_states=src,region_original_target_states=dst,
                    region_proposed_source_states=src+dsrc,region_proposed_target_states=dst+ddst,
                    region_cartesian_pair_dimension_upper_bound_increase=(src+dsrc)*(dst+ddst)-src*dst,
                    feasible_pair_delta='NOT_ENUMERATED; requires representative interface preflight',
                    regional_objective_excludes_B_R_charges=True,production_accepted=False,phase6_authorized=False))
    p=ctx.out/'refinement/interface_impact.csv'
    bimpact=pd.DataFrame(rows);bimpact['dimension_scope']='B_RESTORATION_ONLY_VS_ACCEPTED_COARSE; see joint_interface_dimensions.csv for B plus fine R'
    frame_write(p,bimpact)
    rrows=[]
    for r in terminals.loc[terminals.terminal_type.eq('RACN')].itertuples(index=False):
        members=members_by_terminal[r.terminal_id]
        rrows.append(dict(racn_id=r.parent_object_id,region_id=r.region_id,terminal_id=r.terminal_id,
            fine_portal_states=len(members),fine_physical_node_ids_json=json.dumps(members),
            terminal_state_dimension_increase=max(0,len(members)-1),can_exit_region=bool(r.can_exit_region),can_enter_region=bool(r.can_enter_region),
            requirement='SAME_PHYSICAL_PORTAL_IDENTITY; no free switch between distinct R members',
            scope='ALL_FINE_R_DIAGNOSTICS; independent of restoration candidate',production_accepted=False,phase6_authorized=False))
    frame_write(ctx.out/'refinement/R_fine_interface_impact.csv',pd.DataFrame(rrows))
    dimensions=[]
    for m in [0,16,64,256]:
        selected=set(ladder[str(m)])
        d=terminals[['region_id','terminal_id','parent_object_id','terminal_type','can_exit_region','can_enter_region']].copy()
        d['fine_count']=d.terminal_id.map(counts).astype(np.int64)
        refined=d.terminal_type.eq('RACN')|d.parent_object_id.isin(selected)
        d['original_source_states']=d.can_exit_region.astype(np.int64)
        d['original_target_states']=d.can_enter_region.astype(np.int64)
        d['proposed_source_states']=d.original_source_states*np.where(refined,d.fine_count,1)
        d['proposed_target_states']=d.original_target_states*np.where(refined,d.fine_count,1)
        q=d.groupby('region_id')[['original_source_states','original_target_states','proposed_source_states','proposed_target_states']].sum().reset_index()
        q.insert(0,'candidate_M',m)
        q['cartesian_pair_dimension_upper_bound_increase']=q.proposed_source_states*q.proposed_target_states-q.original_source_states*q.original_target_states
        q['scope']='RESTORED_B_PLUS_FINE_R_VS_ACCEPTED_COARSE_TERMINALS'
        q['feasible_pair_delta']='NOT_ENUMERATED; requires representative interface preflight'
        q['phase6_authorized']=False;dimensions.append(q)
    frame_write(ctx.out/'refinement/joint_interface_dimensions.csv',pd.concat(dimensions,ignore_index=True))
    schema=ctx.out/'refinement/interface_delta.schema.json'
    write_json(schema,dict(schema_id='partC.phase5F.interface_delta.v1',identity=['candidate_M','region_id','terminal_id'],
        proposed_state='one role-correct state per fine physical portal at a restored B',
        R_requirement='Preserve exact physical portal identity wherever coarse-R minimizing routes are unliftable',
        pair_estimate='Cartesian upper bound; does not claim feasibility',
        tables=['interface_impact.csv (B-only rows)','R_fine_interface_impact.csv (every fine R terminal)','joint_interface_dimensions.csv (combined per-region dimensions)'],
        production_adoption=False,labels_generated=False,unchanged_theory='Exact coarse regional convex value with endogenous fine disaggregation'))
    return p


def final_report(ctx):
    request_counts=request_selection_counts(pd.read_csv(ctx.out/'reports/full_calibration_requests.csv'),
        pd.read_parquet(ctx.out/'reports/accepted_link_occurrence_metadata.parquet'))
    request_count_path=ctx.out/'reports/calibration_request_selection_counts.csv';frame_write(request_count_path,request_counts)
    sample=pd.read_csv(ctx.out/'sampling/frozen_od_sample.csv')
    expected=set(sample.stable_od_key);frames=[];coverage=[]
    files=sorted((ctx.out/'results/configurations').glob('*.csv'))
    if len(files)!=96:raise ValueError(f'Expected 96 hybrid graph/state/charge/candidate configurations, found {len(files)}')
    for path in files+[ctx.out/'results/PHYSICAL_LOW.csv',ctx.out/'results/PHYSICAL_HIGH.csv']:
        d=pd.read_csv(path)
        if len(d)!=456 or d.stable_od_key.duplicated().any() or set(d.stable_od_key)!=expected:
            raise ValueError('Incomplete graph/sample coverage: '+str(path))
        for role,g in d.groupby('source_set'):
            n=200 if role=='DISCOVERY_OLD200' else 128
            if len(g)!=n:raise ValueError('Source set row count differs')
            coverage.append(dict(suite_id=d.suite_id.iloc[0],graph_variant=d.graph_variant.iloc[0],state=d.state.iloc[0],
                charge_variant=d.charge_variant.iloc[0],M=int(d.restored_B_count.iloc[0]),source_set=role,rows=len(g),expected_rows=n,status='PASS'))
        frames.append(d)
    full=pd.concat(frames,ignore_index=True)
    if len(full)!=44688:raise ValueError('Main experiment scalar total differs')
    result=ctx.out/'results/shortest_path_scalar_metrics.csv';frame_write(result,full)
    liftcols=[c for c in full if c.startswith(('lift','failure','failed','physical_lift')) or c in ['stable_od_key','suite_id','graph_variant','state','charge_variant','restored_B_count','source_set','reachable','new_unreachable','D_phys_s','D_hybrid_s','D_lift_s','regret','graph_key']]
    lifts=ctx.out/'results/liftability_and_regret.csv';frame_write(lifts,full[liftcols])
    components=[c for c in full if c.endswith('_s') or c in ['stable_od_key','suite_id','graph_variant','state','charge_variant','restored_B_count','source_set','lift_status']]
    cp=ctx.out/'results/component_attribution.csv';frame_write(cp,full[components])
    summaries=[]
    for key,g in full.groupby(['suite_id','graph_variant','state','charge_variant','restored_B_count','source_set'],dropna=False):
        summaries.append(dict(zip(['suite_id','graph_variant','state','charge_variant','restored_B_count','source_set'],key),**metrics(g)))
    summary=pd.DataFrame(summaries);sp=ctx.out/'reports/shortest_path_error_summary.csv';frame_write(sp,summary)
    comparisons=pd.concat([candidate_summary(ctx,'LOW'),candidate_summary(ctx,'HIGH')],ignore_index=True)
    cf=ctx.out/'refinement/candidate_comparison.csv';frame_write(cf,comparisons)
    selection=json.loads((ctx.out/'refinement/selection.json').read_text());selected=selection['provisional_smallest_candidate']
    audit_pass=False
    if selected is not None:
        selected_audit=full.loc[full.suite_id.eq('s03')&full.state.eq('LOW')&full.charge_variant.eq('PRIMARY')&full.source_set.eq('NEW_AUDIT128')&
            full.restored_B_count.eq(selected)&full.graph_variant.eq('MEMBER_B_R_FINE' if selected==0 else 'RESTORED_B_R_FINE')]
        audit_pass=screening(metrics(selected_audit))
    readiness='READY_FOR_PHASE6_INTERFACE_REVIEW' if selected is not None and audit_pass else 'MODEL_REVISION_REQUIRED'
    cross=[]
    basecols=['suite_id','state','charge_variant','source_set','stable_od_key']
    for fine,coarse,label in [('COMPILED_B_R_FINE','COMPILED_B_R_COARSE','R_COARSE_MINUS_FINE'),
        ('COMPILED_B_R_FINE','EXACT_DYNAMIC_SHADOW_R_FINE','COMPILED_MINUS_SHADOW_FINE'),
        ('COMPILED_B_R_COARSE','EXACT_DYNAMIC_SHADOW_R_COARSE','COMPILED_MINUS_SHADOW_COARSE')]:
        a=full.loc[full.graph_variant.eq(fine),basecols+['D_hybrid_s']]
        z=full.loc[full.graph_variant.eq(coarse),basecols+['D_hybrid_s']]
        pair=a.merge(z,on=basecols,suffixes=('_a','_b'),validate='one_to_one')
        pair['distance_difference_s']=pair.D_hybrid_s_b-pair.D_hybrid_s_a if label.startswith('R_') else pair.D_hybrid_s_a-pair.D_hybrid_s_b
        pair['comparison']=label
        if label.startswith('R_') and (pair.distance_difference_s > 1e-7+1e-10*np.maximum(1.,abs(pair.D_hybrid_s_a))).any():
            raise ValueError('R coarse relaxation unexpectedly raises minimum distance')
        cross.append(pair)
    crossfile=ctx.out/'results/paired_minimum_distance_components.csv';frame_write(crossfile,pd.concat(cross,ignore_index=True))
    counts=pd.read_csv(ctx.out/'reports/node_link_count_breakdown.csv').set_index('suite_id')
    reviewed=pd.read_csv(Path(ctx.config['authority_root'])/'evidence/reconciled_topology_counts.csv').set_index('suite_id')
    for current,source in [('B_nodes','retained_B_nodes'),('R_nodes','racn_count'),('total_online_nodes','retained_nodes'),('emitted_static_links','actual_total_links'),('total_logical_links','total_logical_links')]:
        if not np.array_equal(counts[current],reviewed[source]):raise ValueError('Full count reconciliation differs: '+current)
    graphs=[json.loads(p.read_text()) for p in sorted((ctx.out/'graphs').glob('*/identity.json'))]
    gm=ctx.out/'graphs/variant_manifest.json';write_json(gm,dict(graphs=graphs,active_main_configurations=98,
        graph_identity_controls='topology + edge IDs + frozen costs + charges + fine roles + restoration set + input hashes',
        ownership='Every regional arc exactly once; all-B identity every physical arc exactly once'))
    frame_write(ctx.out/'validation/metric_coverage.csv',pd.DataFrame(coverage))
    receipt_paths=list((ctx.out/'receipts').glob('*.json'))+list((ctx.out/'batches').glob('*/*.json'))+list((ctx.out/'graphs').glob('*/receipt.json'))
    receipt_index=ctx.out/'validation/stage_receipts.json'
    write_json(receipt_index,dict(receipts=[file_record(ctx.out,p) for p in sorted(receipt_paths)],F5_receipt_bound_by_final_manifest=True,F6_seal_bound_by_companion=True))
    primary=summary.loc[summary.graph_variant.eq('COMPILED_B_R_FINE')&summary.charge_variant.eq('PRIMARY')&summary.source_set.eq('NEW_AUDIT128')]
    text='# Phase 5F fidelity and interface assessment\n\n'
    text+=f'All 98 required main graph/state/charge configurations and 44,688 scalar rows are complete. Scientific readiness: {readiness}. This is an internal research recommendation awaiting external review. Production is not accepted; Phase 6 remains unauthorized.\n\n'
    text+='The sample is 200 discovery ODs plus 128 new selection and 128 new audit ODs, frozen by eight demand rank strata and full SHA256 ordering. Discovery uses two declared LOW witness roles with each OD demand divided across the two roles. Only NEW_SELECT128 at LOW and PRIMARY charges selects a candidate. HIGH is reused sensitivity, not a pristine final test.\n\n'
    text+='All costs are seconds. CURRENT x45/x200 flows, already scaled capacities, original compiler products and accepted interfaces are immutable. Searches perform no demand loading. The complete source 600 witnesses (200 ODs times three roles; 1,200 rows for two charge registries) are preserved separately from new minimizing routes.\n\n'
    text+='Diagnostic graphs retain detailed regional arcs. Direction-separated terminal entry/exit ports preserve each accepted connector and ensure a fine portal cannot switch through a terminal without its prescribed parent transition. Fine R connections retain one physical portal identity. Coarse R permits only the approved contraction and records replaced connections. These diagnostic ports and any retained fine B/R state require interface review before labels are frozen.\n\n'
    text+='Primary compiled fine-R results on the new LOW/HIGH audit sample:\n\n'
    for r in primary.itertuples(index=False):
        text+=f'- {r.suite_id} {r.state}: weighted mean absolute distance error {r.distance_weighted_mean:.6%}; weighted p95 {r.distance_weighted_p95:.6%}; maximum {r.distance_max:.6%}; unliftable {r.unliftable_count}/{r.OD_count}.\n'
    text+='\n'
    text+=f'The provisional smallest LOW SELECT candidate is {selected if selected is not None else "NONE: no prescribed candidate passes"}. The frozen choice is never changed using AUDIT or HIGH. All M=0,16,64,256 and both unchanged charges are reported. All-B restoration is solely an identity validation, not production adoption.\n\n'
    text+='Verified lifts use exact directed suppressed-B paths and dynamic programming over member choices. Detailed regional arcs retain their selected physical endpoints. A different fine R portal cannot be connected for free in a physical lift; impossible paths are UNLIFTABLE, with failure states retained. Lifted regret is conditional on verified continuous paths and must not beat the physical optimum. Signed distance error, B charge mismatch, aggregation/alignment, fit contribution, R relaxation and route regret are reported separately.\n\n'
    text+='Before Phase 6, externally review the B/R state architecture and the complete adverse distributions. If bounded restoration does not pass, revise the physical-state architecture rather than clipping charges or adjusting capacity. Use interface_impact.csv for B restoration, R_fine_interface_impact.csv for every fine R terminal, joint_interface_dimensions.csv for combined regional dimension upper bounds, and interface_delta.schema.json for interpretation; exact feasible pair deltas remain a future preflight question.\n\n'
    text+='Proposed representative Phase 6 preflight (not authorized here): choose regions covering large/small portal counts, affected restored B boundaries, coarse-R lift failures, disconnected directed support and zero-CURRENT feasible pairs. Preserve conceptual feasible, observed and modeled registries separately. Solve the exact regional Beckmann value with endogenous fine disaggregation under Q lambda = lambda_bar, excluding external B/R charges; obtain refreshed coarse subgradients and test conservation, value hierarchy, finite-difference/subgradient inequalities, monotonicity, cyclic monotonicity and label stability. Freeze reviewed terminal states and local numerical tolerances before any labels or ICNN training. No labels were generated by this task.\n\n'
    text+='The graph diagnostics retain physical regional detail and do not establish final ICNN network speedup, national assignment speedup or global label accuracy. Weighted outputs are stratified sample estimates with no asserted national totals or confidence intervals.\n'
    report=ctx.out/'reports/readiness_and_next_phase.md';report.write_text(text,encoding='utf-8')
    status=ctx.out/'reports/F5_status.json'
    write_json(status,dict(status='VALIDATED_PENDING_F6_FRESH_REOPEN_AND_SEAL',scientific_readiness=readiness,
        main_scalar_rows=len(full),main_configurations=98,source_OD_count=456,all_candidates_preserved=True,
        provisional_smallest_candidate=selected,independent_LOW_audit_pass=audit_pass,
        production_accepted=False,phase6_authorized=False,icnn_training_performed=False,
        national_assignment_performed=False,physical_benchmark_rerun_performed=False,
        frozen_shortest_path_diagnostics_performed=True,source_compiler_rerun_performed=False,
        next_step_authorized=False,recommendation='EXTERNAL_FIDELITY_AND_INTERFACE_REVIEW_ONLY'))
    return [result,lifts,cp,sp,cf,crossfile,gm,ctx.out/'validation/metric_coverage.csv',receipt_index,report,status,request_count_path]
