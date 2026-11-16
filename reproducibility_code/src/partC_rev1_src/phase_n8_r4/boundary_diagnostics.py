"""Frozen physical-cost regional witnesses and explicitly conditional global lifts."""
from pathlib import Path
from types import SimpleNamespace
import time
import traceback
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,read_npz,write_json,sha,array_hash,ResourceLimit
from partC_rev1_src.phase_n8.oracle import Network,conservation,aon
from partC_rev1_src.phase_n8.native import load_registry
from partC_rev1_src.phase_n8.objective import Composite
from partC_rev1_src.phase6.oracle import Oracle
from partC_rev1_src.phase5_fidelity.lift import PhysicalLifter
from .context import sources,reopen_suite
from .safety import load_safe,SafeComposite,ANALYTIC
from .path_metrics import bindings,sample_demand
from .path_search import query
from .learning_data import RegionalSources
from .geography import database,NODE_TABLE

WITNESS_COLUMNS=['region_id','pair_id','native_coordinate','global_edge','national_coarse_pair_flow',
    'frozen_x45_physical_path_seconds','physical_arc_count','source_physical_uid','target_physical_uid',
    'frozen_physical_path_hash','source_terminal_id','target_terminal_id','source_terminal_type',
    'target_terminal_type','local_path_validated','centroid_nontransit_validated',
    'max_pair_flow_reconstruction_error','role']


def centroid_flags(ctx,g,dst):
    """Complete cleaned physical-node flag; OD endpoints are an incomplete subset."""
    path,con=database(ctx);before=path.stat()
    try:
        nodes=pd.read_sql_query('SELECT partC_rev1_node_uid,partC_rev1_centroid FROM '+NODE_TABLE,con)
    finally:con.close()
    if path.stat().st_mtime_ns!=before.st_mtime_ns:raise ValueError('Source node database changed')
    if nodes.partC_rev1_node_uid.duplicated().any() or not nodes.partC_rev1_centroid.isin([0,1]).all():
        raise ValueError('Ambiguous physical centroid flags')
    # The accepted graph loader stores UIDs as strings; use the same stable
    # integer-string representation on the SQLite side before the exact join.
    flag=nodes.set_index(nodes.partC_rev1_node_uid.astype(np.int64).astype(str)).partC_rev1_centroid.reindex(g.node_idx_to_uid.astype(str))
    if flag.isna().any():raise ValueError('Physical centroid flags missing accepted graph nodes')
    frame=pd.DataFrame(dict(physical_node_uid=g.node_idx_to_uid.astype(str),is_physical_centroid=flag.to_numpy(bool)))
    frame.to_csv(dst/'accepted_graph_centroid_flags.csv',index=False)
    write_json(dst/'CENTROID_SOURCE.json',dict(native_database=str(path),size_bytes=before.st_size,mtime_ns=before.st_mtime_ns,
        query='SELECT partC_rev1_node_uid,partC_rev1_centroid FROM '+NODE_TABLE,
        exported_rows_sha256=sha(dst/'accepted_graph_centroid_flags.csv'),graph_nodes=len(flag),centroids=int(flag.sum()),
        whole_database_hash_recomputed=False,universe='All accepted graph physical nodes, not only OD endpoints'))
    return flag.to_numpy(bool)


def centroid_audit(g,path,flags):
    interior=g.head[path[:-1]] if len(path)>1 else np.empty(0,np.int64)
    bad=interior[flags[interior]]
    return len(bad), '|'.join(g.node_idx_to_uid[np.unique(bad)].astype(str))


def choose_case(ctx):
    p=ctx.root/'runs/B02_PRIMARY_BOUND_SAFE/RESULT.json'
    if p.exists():
        r=read_json(p)
        if r.get('final_certificate_fresh') and not r.get('stop_reason','').startswith('STOP_'):
            return 'N8R4_PRIMARY_BOUND_SAFE_V1',p.parent,True
    return 'S03_ANALYTIC_VALID',ctx.prior/'conditional_R3/assignments/R3_A02_S03_ANALYTIC_VALID',False


def _run(ctx,args):
    dst=ctx.root/'boundary_diagnostics';dst.mkdir(exist_ok=False)
    src=sources(ctx);suite,od,balance,coverage=reopen_suite(ctx,src,'s03');b=bindings(ctx);g=b.graph;t=b.trace_maps[0]
    physical_centroid=centroid_flags(ctx,g,dst)
    scenario,root,is_safe=choose_case(ctx);r=read_json(root/'RESULT.json');ctx.bind(root/'RESULT.json','ADMITTED_DIAGNOSTIC_BASELINE')
    ctx.bind(root/'final_flow.npz','DIAGNOSTIC_BASELINE_FLOW',r['final_flow_sha256']);flow=read_npz(root/'final_flow.npz')['flow']
    if array_hash({'flow':flow})!=r['state_hash']:raise ValueError('Diagnostic baseline certificate mismatch')
    national_conservation=conservation(suite['graph'],flow,balance,od.total)
    entries,reg=load_safe(ctx,src,suite,scenario) if is_safe else load_registry(src,suite,ANALYTIC)
    objective=(SafeComposite if is_safe else Composite)(suite,entries,reg,guard=ctx.guard)
    if objective.identity!=r['identity']['objective']:raise ValueError('Diagnostic baseline objective identity changed')
    value,solved_cost,hvp,diag=objective.evaluate(flow)
    value,fixed_cost,hvp,fdiag=objective.evaluate(suite['fixed_x45'])
    source_rows=ctx.root/'path_metrics'/(scenario+'_SOLVED_X45_REFERENCE_PATH_ROWS.csv')
    ctx.bind(source_rows,'CURRENT_PAIRED_PATH_OUTCOMES_FOR_DECLARED_DIAGNOSTICS')
    rows=pd.read_csv(source_rows,float_precision='round_trip');historical=rows.loc[rows.sample_role.eq('HISTORICAL_NOW_SEEN')]
    selected={}
    def select(row,role):selected.setdefault(str(row.stable_od_key),dict(row=row,roles=[]))['roles'].append(role)
    named=historical.loc[historical.stable_od_key.eq('92018|262423')]
    if len(named)!=1:raise ValueError('Required named diagnostic OD absent')
    select(named.iloc[0],'REQUIRED_92018_262423')
    discovery=historical.loc[historical.source_set.eq('DISCOVERY_OLD200')].copy()
    discovery['demand_absolute_error']=discovery.demand*discovery.absolute_error_seconds
    select(discovery.sort_values(['demand_absolute_error','stable_od_key'],ascending=[False,True]).iloc[0],'DISCOVERY_MAX_DEMAND_ERROR_CONTRIBUTION')
    select(discovery.sort_values(['D_phys_s','stable_od_key'],ascending=[False,True]).iloc[0],'DISCOVERY_MAX_REFERENCE_TIME')
    ordinary=historical.loc[np.isfinite(historical.relative_error_1s)].copy()
    ordinary['median_distance']=abs(ordinary.relative_error_1s-ordinary.relative_error_1s.median())
    select(ordinary.sort_values(['median_distance','stable_od_key']).iloc[0],'ORDINARY_NEAREST_MEDIAN_RELATIVE_ERROR')
    for row in rows.loc[rows.sample_role.eq('DEMAND_PPS_WITH_REPLACEMENT')].sort_values('draw_id').head(32).itertuples(index=False):
        select(pd.Series(row._asdict()),'FIRST32_PREFROZEN_PPS_DRAWS')
    native_paths={};needed=set();native_rows=[]
    for key,item in selected.items():
        row=item['row'];idx=int(row.accepted_od_row);group=int(np.searchsorted(od.offsets[1:],idx,side='right'))
        distance,path=query(suite['graph'],solved_cost,od.origins[group],od.destinations[idx])
        if not np.isclose(distance,row.D_surrogate_s,rtol=1e-12,atol=1e-7):raise ValueError('Native diagnostic path does not reproduce paired time')
        native_paths[key]=path
        for order,edge in enumerate(path):
            edge=int(edge);kind='SUPERPORT_ZERO';region='';fixed=dynamic=regional=0.
            if edge<suite['packed_count']:
                link=suite['links'].iloc[edge];kind=str(link.role)
                fixed=float(link.fixed_internal_H_s+link.tail_half_charge_s+link.head_half_charge_s)
                dynamic=float(solved_cost[edge]-fixed)
            elif edge<suite['packed_count']+len(suite['pairs']):
                pair=suite['pairs'].iloc[edge-suite['packed_count']];kind='REGIONAL_PAIR';region=pair.region_id
                regional=float(solved_cost[edge]);needed.add(edge)
            native_rows.append(dict(stable_od_key=key,sequence=order,edge_index=edge,edge_id=suite['graph'].edge_ids[edge],
                kind=kind,region_id=region,cost_seconds=solved_cost[edge],fixed_B_charge_seconds=fixed,
                packed_dynamic_including_fftt_seconds=dynamic,regional_surrogate_seconds=regional,
                tail_node_id=suite['graph'].node_ids[suite['graph'].tail[edge]],head_node_id=suite['graph'].node_ids[suite['graph'].head[edge]]))
    pd.DataFrame(native_rows).to_csv(dst/'selected_native_path_edges.csv',index=False,float_format='%.17g')
    write_json(dst/'DIAGNOSTIC_SELECTION.json',dict(scenario=scenario,selection=[dict(stable_od_key=k,roles=v['roles']) for k,v in selected.items()],
        probability_component='FIRST32_PREDECLARED_DRAW_IDS; unique ODs may be fewer, draw multiplicity retained in source',
        selected_tails_are_not_population_estimates=True,source_sha256=sha(source_rows)))
    shadow=fixed_cost.copy();solved_shadow=solved_cost.copy();witnesses={};witness_rows=[];local_failures=[];regional_sources=RegionalSources(ctx)
    term_types=suite['terms'].drop_duplicates('terminal_id').set_index('terminal_id').terminal_type.to_dict()
    completed_regions=[]
    try:
        groups=suite['pairs'].groupby('region_id',sort=False).indices
        for ordinal,(rid,index) in enumerate(groups.items()):
            ctx.guard();desc,a,pairs,scales=regional_sources.source.local(rid)
            native=suite['pairs'].iloc[index]
            if not np.array_equal(native.pair_id,pairs.pair_id):raise ValueError('Local/national active pair order mismatch')
            oracle=Oracle(a);cost=b.low_physical.current_cost_s[a['global_arc_index']]
            distance,sn,tn,ptr,arcs=oracle.lmo(cost)
            pairflow=flow[suite['packed_count']+index]
            ids=oracle.register(sn,tn,ptr,arcs);max_pair_error=oracle.validate(pairflow,ids,pairflow)
            shadow[suite['packed_count']+index]=distance
            solved_shadow[suite['packed_count']+index]=distance
            for k in np.flatnonzero((pairflow>0)|np.isin(suite['packed_count']+index,list(needed))):
                local=arcs[ptr[k]:ptr[k+1]];physical=a['global_arc_index'][local]
                edge=int(suite['packed_count']+index[k]);source_node=int(a['global_node_index'][sn[k]]);target_node=int(a['global_node_index'][tn[k]])
                if edge in needed:witnesses[edge]=dict(arcs=physical.copy(),source=source_node,target=target_node,region_id=rid,pair_id=pairs.pair_id.iloc[k])
                witness_rows.append(dict(region_id=rid,pair_id=pairs.pair_id.iloc[k],native_coordinate=int(k),global_edge=edge,
                    national_coarse_pair_flow=float(pairflow[k]),frozen_x45_physical_path_seconds=float(distance[k]),
                    physical_arc_count=len(physical),source_physical_uid=str(g.node_idx_to_uid[source_node]),target_physical_uid=str(g.node_idx_to_uid[target_node]),
                    frozen_physical_path_hash=array_hash({'arcs':physical}),
                    source_terminal_id=native.source_terminal_id.iloc[k],target_terminal_id=native.target_terminal_id.iloc[k],
                    source_terminal_type=term_types[native.source_terminal_id.iloc[k]],target_terminal_type=term_types[native.target_terminal_id.iloc[k]],
                    local_path_validated=True,centroid_nontransit_validated=True,max_pair_flow_reconstruction_error=max_pair_error,
                    role='FROZEN_PHYSICAL_COST_SHORTEST_PATH_NOT_JOINT_REGIONAL_EQUILIBRIUM'))
            completed_regions.append(rid)
            if ordinal%100==0:
                print('LOCAL_WITNESS',ordinal+1,'of',len(groups),flush=True);regional_sources.flush_bindings()
                pd.DataFrame(witness_rows,columns=WITNESS_COLUMNS).to_csv(dst/'positive_coarse_pair_local_witnesses.csv',index=False,float_format='%.17g')
                write_json(dst/'LOCAL_WITNESS_PROGRESS.json',dict(completed_regions=completed_regions,total_regions=len(groups),
                    completed_rows=len(witness_rows),positive_pair_rows=sum(x['national_coarse_pair_flow']>0 for x in witness_rows),
                    remaining_regions=[r for r in groups if r not in set(completed_regions)]))
        regional_sources.flush_bindings()
    finally:
        pd.DataFrame(witness_rows,columns=WITNESS_COLUMNS).to_csv(dst/'positive_coarse_pair_local_witnesses.csv',index=False,float_format='%.17g')
        write_json(dst/'LOCAL_WITNESS_PROGRESS.json',dict(completed_regions=completed_regions,total_regions=len(groups),
            completed_rows=len(witness_rows),positive_pair_rows=sum(x['national_coarse_pair_flow']>0 for x in witness_rows),
            remaining_regions=[r for r in groups if r not in set(completed_regions)]))
        regional_sources.close()
    pd.DataFrame(witness_rows,columns=WITNESS_COLUMNS).to_csv(dst/'positive_coarse_pair_local_witnesses.csv',index=False,float_format='%.17g')
    if set(witnesses)!=needed:raise ValueError('Some selected regional path edges lack frozen-state witnesses')
    source_physical=src.base/'phase_05/fidelity_0906/run_001'
    source_table=pd.read_csv(source_physical/'results/PHYSICAL_LOW.csv',float_precision='round_trip').set_index('stable_od_key')
    physical_graph=Network(g.tail,g.head,np.zeros(len(g.node_idx_to_uid),bool),g.arc_idx_to_directed_arc_id.astype(str),g.node_idx_to_uid.astype(str))
    sequences=src.frame(src.p5/'suites/s03/macro_sequences.parquet','S03_NATIVE_MACRO_SEQUENCE_FOR_DIAGNOSTIC_LIFT')
    lifter=PhysicalLifter(b,t,{},suite['links'],sequences,'LOW')
    bundle_index={v:i for i,v in enumerate(t.bundle_ids)};macro_index={v:i for i,v in enumerate(lifter.native.link_id)}
    result_rows=[];physical_rows=[];lift_rows=[];regional_arcs=[];centroid_violations=[]
    for key,item in selected.items():
        ctx.guard();row=item['row'];origin=int(row.origin_idx);destination=int(row.destination_idx)
        if key in source_table.index:
            source=source_table.loc[key];p=source_physical/source.native_path_artifact
            receipt=p.with_name(p.name.replace('_paths.npz','.json'));ctx.bind(receipt,'EXACT_SOURCE_PHYSICAL_PATH_BATCH_RECEIPT')
            rr=read_json(receipt);binding=next(x for x in rr['artifacts'] if x['relative_path'].replace('\\','/')==p.relative_to(source_physical).as_posix())
            ctx.bind(p,'EXACT_FROZEN_PHYSICAL_REFERENCE_PATH',binding['sha256']);z=read_npz(p);q=int(source.path_row)
            if z['stable_od_keys'][q]!=key:raise ValueError('Frozen physical path OD identity changed')
            path=z['physical_arcs'][z['lift_ptr'][q]:z['lift_ptr'][q+1]];role='ORIGINAL_FROZEN_PHYSICAL_X45_PATH'
        else:
            distance,path=query(physical_graph,b.low_physical.current_cost_s,origin,destination);role='FRESH_QUERY_AT_FROZEN_PHYSICAL_X45'
        reference=float(b.low_physical.current_cost_s[path].sum())
        if not np.isclose(reference,row.D_phys_s,rtol=1e-12,atol=1e-7):raise ValueError('Physical path/cost/reference discrepancy')
        if len(path) and (g.tail[path[0]]!=origin or g.head[path[-1]]!=destination or np.any(g.head[path[:-1]]!=g.tail[path[1:]])):
            raise ValueError('Stored physical path discontinuity')
        for sequence,arc in enumerate(path):
            physical_rows.append(dict(stable_od_key=key,sequence=sequence,physical_arc_idx=int(arc),physical_arc_id=str(g.arc_idx_to_directed_arc_id[arc]),
                source_role=role,x45_flow=float(b.low_physical.current_flow[arc]),clean_prepared_capacity=float(g.capacity[arc]),
                used_capacity=float(b.low_physical.used_capacity[arc]),capacity_factor=.3,free_time_seconds=float(g.free_time[arc]*60),
                alpha=float(g.alpha[arc]),beta=float(g.beta[arc]),x45_cost_seconds=float(b.low_physical.current_cost_s[arc]),
                x45_volume_capacity_ratio=float(b.low_physical.current_flow[arc]/b.low_physical.used_capacity[arc]),
                physical_cost_units='seconds; native minutes multiplied once by60'))
        ops=[];refs=[];occurrences=[]
        for edge in native_paths[key]:
            edge=int(edge)
            if edge<suite['packed_count']:
                link=suite['links'].iloc[edge]
                if link.role=='PHYSICAL_MEMBER':ops.append(1);refs.append(int(link.member_arc_idx))
                elif link.role=='EXACT_PRIMITIVE':ops.append(2);refs.append(bundle_index[str(link.primitive_id)])
                elif link.role=='MACRO':ops.append(3);refs.append(macro_index[str(link.link_id)])
                elif link.role!='TERMINAL_CONNECTOR':raise ValueError('Unrecognized packed diagnostic lift role')
            elif edge in witnesses:
                w=witnesses[edge];occurrences.append(w['region_id'])
                # Preserve every occurrence. An independently selected local path
                # is one declared fine-state choice, not all admissible choices.
                if len(w['arcs']):
                    ops.extend([1]*len(w['arcs']));refs.extend(w['arcs'].tolist())
                else:ops.append(4);refs.append(w['source'])
                for k,arc in enumerate(w['arcs']):regional_arcs.append(dict(stable_od_key=key,occurrence=len(occurrences),
                    region_id=w['region_id'],pair_id=w['pair_id'],sequence=k,physical_arc_idx=int(arc),physical_arc_id=str(g.arc_idx_to_directed_arc_id[arc])))
        adapter=SimpleNamespace(op=np.asarray(ops,np.int8),ref=np.asarray(refs,np.int64))
        lifted,physical= lifter.lift(adapter,np.arange(len(ops),dtype=np.int64),origin,destination,reference)
        if lifted['lift_status']=='UNLIFTABLE':lifted['lift_status']='SELECTED_LOCAL_WITNESS_STITCH_FAILURE_NOT_EXHAUSTIVE_UNLIFTABILITY'
        ref_centroid_count,ref_centroid_ids=centroid_audit(g,path,physical_centroid)
        lift_centroid_count,lift_centroid_ids=centroid_audit(g,physical,physical_centroid)
        lift_centroid_status=('PASS' if lift_centroid_count==0 else 'CENTROID_TRANSIT_PRESENT') if lifted['lift_status']=='VERIFIED' else 'NO_VERIFIED_GLOBAL_PATH'
        for path_role,arc_sequence in [('PHYSICAL_REFERENCE',path),('SELECTED_GLOBAL_LIFT',physical)]:
            if len(arc_sequence)<2:continue
            interior=g.head[arc_sequence[:-1]]
            for pos in np.flatnonzero(physical_centroid[interior]):
                centroid_violations.append(dict(stable_od_key=key,path_role=path_role,path_node_position=int(pos+1),
                    physical_node_index=int(interior[pos]),physical_node_uid=str(g.node_idx_to_uid[interior[pos]]),
                    incoming_physical_arc_id=str(g.arc_idx_to_directed_arc_id[arc_sequence[pos]]),
                    outgoing_physical_arc_id=str(g.arc_idx_to_directed_arc_id[arc_sequence[pos+1]])))
        for sequence,arc in enumerate(physical):lift_rows.append(dict(stable_od_key=key,sequence=sequence,physical_arc_idx=int(arc),
            physical_arc_id=str(g.arc_idx_to_directed_arc_id[arc]),frozen_x45_cost_seconds=float(b.low_physical.current_cost_s[arc])))
        idx=int(row.accepted_od_row);group=int(np.searchsorted(od.offsets[1:],idx,side='right'))
        shadow_distance,shadow_path=query(suite['graph'],shadow,od.origins[group],od.destinations[idx])
        fixed_native_distance,_=query(suite['graph'],fixed_cost,od.origins[group],od.destinations[idx])
        solved_shadow_distance,_=query(suite['graph'],solved_shadow,od.origins[group],od.destinations[idx])
        result_rows.append(dict(stable_od_key=key,selection_roles='|'.join(item['roles']),D_phys_s=reference,D_native_s=float(row.D_surrogate_s),
            D_native_FIXED_X45_s=fixed_native_distance,D_frozen_regional_cost_native_solved_backbone_s=solved_shadow_distance,
            D_frozen_regional_cost_native_fixed_backbone_s=shadow_distance,regional_traversal_occurrences=len(occurrences),
            distinct_regions=len(set(occurrences)),repeated_regional_traversals=len(occurrences)-len(set(occurrences)),
            physical_reference_arc_count=len(path),source_physical_path_role=role,
            physical_reference_continuity_pass=True,selected_global_lift_continuity_pass=lifted['lift_status']=='VERIFIED',
            physical_reference_interior_centroid_occurrences=ref_centroid_count,physical_reference_interior_centroid_uids=ref_centroid_ids,
            physical_reference_centroid_policy='Accepted historical physical assignment permits transit; separate audit, no source change',
            selected_global_lift_centroid_audit=lift_centroid_status,selected_global_lift_interior_centroid_occurrences=lift_centroid_count,
            selected_global_lift_interior_centroid_uids=lift_centroid_ids,
            global_lift_scope='Declared independently cheapest regional witnesses + exact B/bundle member DP at frozen x45 costs; no equilibrium topology changes',
            **lifted))
        pd.DataFrame(result_rows).to_csv(dst/'selected_path_diagnostics.csv',index=False,float_format='%.17g')
        pd.DataFrame(physical_rows).to_csv(dst/'selected_physical_reference_path_arcs.csv',index=False,float_format='%.17g')
        pd.DataFrame(lift_rows,columns=['stable_od_key','sequence','physical_arc_idx','physical_arc_id',
            'frozen_x45_cost_seconds']).to_csv(dst/'verified_selected_global_lift_arcs.csv',index=False,float_format='%.17g')
        pd.DataFrame(regional_arcs,columns=['stable_od_key','occurrence','region_id','pair_id','sequence',
            'physical_arc_idx','physical_arc_id']).to_csv(dst/'selected_regional_witness_arcs.csv',index=False)
        pd.DataFrame(centroid_violations,columns=['stable_od_key','path_role','path_node_position','physical_node_index','physical_node_uid',
            'incoming_physical_arc_id','outgoing_physical_arc_id']).to_csv(dst/'interior_centroid_violations.csv',index=False)
        write_json(dst/'GLOBAL_DIAGNOSTIC_PROGRESS.json',dict(selected_ODs_planned=len(selected),
            completed_selected_ODs=len(result_rows),verified_selected_global_lifts=sum(x['lift_status']=='VERIFIED' for x in result_rows),
            selected_witness_stitch_failures=sum(x['lift_status']!='VERIFIED' for x in result_rows)))
    pd.DataFrame(result_rows).to_csv(dst/'selected_path_diagnostics.csv',index=False,float_format='%.17g')
    pd.DataFrame(physical_rows).to_csv(dst/'selected_physical_reference_path_arcs.csv',index=False,float_format='%.17g')
    pd.DataFrame(lift_rows,columns=['stable_od_key','sequence','physical_arc_idx','physical_arc_id',
        'frozen_x45_cost_seconds']).to_csv(dst/'verified_selected_global_lift_arcs.csv',index=False,float_format='%.17g')
    pd.DataFrame(regional_arcs,columns=['stable_od_key','occurrence','region_id','pair_id','sequence',
        'physical_arc_idx','physical_arc_id']).to_csv(dst/'selected_regional_witness_arcs.csv',index=False)
    pd.DataFrame(centroid_violations,columns=['stable_od_key','path_role','path_node_position','physical_node_index','physical_node_uid',
        'incoming_physical_arc_id','outgoing_physical_arc_id']).to_csv(dst/'interior_centroid_violations.csv',index=False)
    # Reused centroid exclusions stay outside the current accepted support.
    audit=src.base/'phase_06/s03_active_0906/run_001/validation/source_support_audit'
    for name in ('RESULT.json','centroid_nontransit_witnesses.json'):
        ctx.bind(audit/name,'REUSED_ACCEPTED_SUPPORT_EXCLUSION_EVIDENCE')
        write_json(dst/('REUSED_'+name),read_json(audit/name))
    result=dict(status='EXECUTED',scenario=scenario,national_conservation=national_conservation,
        local_positive_pair_witness_rows=sum(x['national_coarse_pair_flow']>0 for x in witness_rows),
        selected_distinct_ODs=len(selected),verified_selected_global_lifts=sum(x['lift_status']=='VERIFIED' for x in result_rows),
        completed_selected_ODs=len(result_rows),
        selected_lift_success_fraction_is_not_a_probability_estimate=True,
        selected_witness_stitch_failures=sum(x['lift_status']!='VERIFIED' for x in result_rows),
        all_OD_global_lifting_claim=False,regional_equilibrium_labels_generated=False,
        fixed_physical_cost_diagnostic_not_equilibrium=True,capacity_changes=0,active_support_changes=0)
    write_json(dst/'RESULT.json',result);return result


def run(ctx,args):
    if (ctx.root/'boundary_diagnostics').exists():
        raise FileExistsError('Boundary diagnostics already started; preserve existing outputs before explicit recovery')
    hard_deadline=ctx.deadline
    ctx.deadline=min(ctx.c['compute_cutoff_epoch'],hard_deadline or ctx.c['compute_cutoff_epoch'])-120
    try:return _run(ctx,args)
    except Exception as exc:
        dst=ctx.root/'boundary_diagnostics';dst.mkdir(exist_ok=True)
        selection=read_json(dst/'DIAGNOSTIC_SELECTION.json') if (dst/'DIAGNOSTIC_SELECTION.json').exists() else {}
        local=read_json(dst/'LOCAL_WITNESS_PROGRESS.json') if (dst/'LOCAL_WITNESS_PROGRESS.json').exists() else {}
        global_progress=read_json(dst/'GLOBAL_DIAGNOSTIC_PROGRESS.json') if (dst/'GLOBAL_DIAGNOSTIC_PROGRESS.json').exists() else {}
        result=dict(status='PARTIAL_DIAGNOSTICS' if isinstance(exc,(TimeoutError,ResourceLimit)) else 'FAIL',
            error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
            scenario=selection.get('scenario'),selected_distinct_ODs=len(selection.get('selection',[])),
            completed_selected_ODs=global_progress.get('completed_selected_ODs',0),
            verified_selected_global_lifts=global_progress.get('verified_selected_global_lifts',0),
            local_positive_pair_witness_rows=local.get('positive_pair_rows',0),
            completed_local_regions=len(local.get('completed_regions',[])),
            local_regions_planned=local.get('total_regions'),
            selected_witness_stitch_failures=global_progress.get('selected_witness_stitch_failures',0),
            missing_diagnostics_not_imputed=True,all_OD_global_lifting_claim=False,
            regional_equilibrium_labels_generated=False,capacity_changes=0,active_support_changes=0,
            actual_available_files=[p.name for p in dst.iterdir() if p.is_file()])
        write_json(dst/'RESULT.json',result);return result
    finally:ctx.deadline=hard_deadline
