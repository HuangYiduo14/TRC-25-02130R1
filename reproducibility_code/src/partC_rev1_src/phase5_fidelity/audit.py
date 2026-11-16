"""Complete source-population addenda, immutable bindings and held-state queries."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from partC_rev1_src.phase5.campaign_io import sha, frame_write, write_json
from partC_rev1_src.phase5.campaign_compile import spec_from_row
from partC_rev1_src.phase5.bundle_oracle import exact_series_oracle


def source_invariance(ctx, final=False):
    source_manifest = pd.read_csv(Path(ctx.config['source_extraction_root'])/'external_artifact_index/full_output_MANIFEST.csv')
    prior = pd.read_parquet(ctx.source/'validation/source_bindings.parquet')
    source = source_manifest[['absolute_path','size_bytes','sha256']].copy();source['role']='SOURCE_CAMPAIGN'
    bindings = pd.concat([source,prior[['absolute_path','size_bytes','sha256','role']]],ignore_index=True).drop_duplicates('absolute_path')
    rows=[]
    for i,r in enumerate(bindings.itertuples(index=False)):
        p=Path(r.absolute_path);st=p.stat()
        actual=sha(p)
        if st.st_size != int(r.size_bytes) or actual != r.sha256:
            raise ValueError('Authoritative input changed: '+str(p))
        rows.append(dict(absolute_path=str(p),role=r.role,size_bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=actual))
        if i%250==0:ctx.mark(f'{"Final" if final else "Initial"} source identity {i+1}/{len(bindings)}')
    path=ctx.out/'validation'/('source_invariance.json' if final else 'source_bindings_initial.csv')
    if final:
        initial=pd.read_csv(ctx.out/'validation/source_bindings_initial.csv')
        current=pd.DataFrame(rows)
        if not initial[['absolute_path','size_bytes','sha256']].equals(current[['absolute_path','size_bytes','sha256']]):
            raise ValueError('Initial/final source invariance differs')
        write_json(path,dict(status='PASS',bound_files=len(rows),all_sha256_unchanged=True,
            initial_manifest_sha256=sha(ctx.out/'validation/source_bindings_initial.csv'),
            original_ancestor=ctx.config['required_ancestor'],accepted_physical_benchmark_updates=0))
    else:frame_write(path,pd.DataFrame(rows))
    return path


def summarize(values):
    v=np.asarray(values,float);v=v[np.isfinite(v)]
    return dict(count=len(v),mean=float(v.mean()) if len(v) else None,
        p50=float(np.quantile(v,.5)) if len(v) else None,
        p95=float(np.quantile(v,.95)) if len(v) else None,
        p99=float(np.quantile(v,.99)) if len(v) else None,
        maximum=float(v.max()) if len(v) else None)


def count_addendum(ctx):
    records=[];accepted_rows=[]
    review=pd.read_csv(Path(ctx.config['authority_root'])/'evidence/reconciled_topology_counts.csv')
    for suite in ctx.config['suites']:
        base=ctx.source/'suites'/suite
        nodes=pd.read_parquet(base/'retained_nodes.parquet')
        links=pd.read_parquet(base/'runtime_links.parquet')
        if nodes.node_id.duplicated().any() or links.link_id.duplicated().any():raise ValueError('Stable-ID count duplication')
        kind=nodes.set_index('node_id').node_type
        connectors=links.loc[links.role.eq('TERMINAL_CONNECTOR')]
        tail=connectors.tail_node_id.map(kind);head=connectors.head_node_id.map(kind)
        classes=dict(B_nodes=int(nodes.node_type.eq('BACN').sum()),R_nodes=int(nodes.node_type.eq('RACN').sum()),
            terminal_origin=int(nodes.terminal_type.eq('ORIGIN_CENTROID').sum()),
            terminal_destination=int(nodes.terminal_type.eq('DESTINATION_CENTROID').sum()),
            terminal_B=int(nodes.terminal_type.eq('BACN').sum()),terminal_R=int(nodes.terminal_type.eq('RACN').sum()),
            terminal_other=int((nodes.node_type.eq('TERMINAL') & ~nodes.terminal_type.isin(['BACN','RACN','ORIGIN_CENTROID','DESTINATION_CENTROID'])).sum()),
            macro_links=int(links.role.eq('MACRO').sum()),exact_primitive_links=int(links.role.eq('EXACT_PRIMITIVE').sum()),
            physical_member_links=int(links.role.eq('PHYSICAL_MEMBER').sum()),
            B_terminal_connectors=int((tail.eq('BACN')|head.eq('BACN')).sum()),
            R_terminal_connectors=int((tail.eq('RACN')|head.eq('RACN')).sum()),
            other_links=int((~links.role.isin(['MACRO','EXACT_PRIMITIVE','PHYSICAL_MEMBER','TERMINAL_CONNECTOR'])).sum()))
        if classes['B_terminal_connectors']+classes['R_terminal_connectors']!=len(connectors):raise ValueError('Unclassified connector')
        contract=json.loads((base/'regional_interface_contract.json').read_text())
        counts=json.loads((base/'counts.json').read_text())
        # Exact logical-support count is copied from the unchanged interface contract.
        pair_count=contract.get('conceptual_regional_pair_count')
        if pair_count is None:
            pair_count=counts.get('conceptual_regional_pair_count')
        if pair_count is None:
            raise ValueError('Conceptual pair count field requires explicit schema reconciliation: '+json.dumps(contract))
        row=dict(suite_id=suite,**classes,total_online_nodes=len(nodes),emitted_static_links=len(links),
            conceptual_regional_links=int(pair_count),total_logical_links=len(links)+int(pair_count),
            interface_contract_sha256=sha(base/'regional_interface_contract.json'),
            zero_current_member_links=int((links.role.eq('PHYSICAL_MEMBER')&links.low_current_flow.eq(0)).sum()))
        records.append(row)
        ctx.mark('F0 full node/link count '+suite)
        # Full link occurrence map identifies every fitted request and exact class.
        models=pd.read_parquet(base/'dynamic_models.parquet')
        cols=[x for x in ['model_id','calibration_request_id','is_exact'] if x in models]
        meta=links[['suite_id','link_id','dynamic_model_id','role','runtime_class','domain_Q','low_current_flow','high_current_flow']].merge(models[cols],left_on='dynamic_model_id',right_on='model_id',validate='many_to_one')
        accepted_rows.append(meta)
    result=pd.DataFrame(records)
    p=ctx.out/'reports/node_link_count_breakdown.csv';frame_write(p,result)
    ap=ctx.out/'reports/accepted_link_occurrence_metadata.parquet';frame_write(ap,pd.concat(accepted_rows,ignore_index=True))
    return [p,ap]


def fit_population_addendum(ctx):
    requests=pd.read_parquet(ctx.source/'registries/calibration_requests.parquet')
    if requests.request_id.duplicated().any() or len(requests)!=23064 or requests.high_used.any():
        raise ValueError('Complete low-only calibration request population differs')
    reqfile=ctx.out/'reports/full_calibration_requests.csv';frame_write(reqfile,requests)
    ridx=pd.Index(requests.request_id)
    costs=np.zeros(len(requests));pots=np.zeros(len(requests));counts=np.zeros(len(requests),np.int64)
    summaries=[];all_probes=[]
    for i,p in enumerate(sorted((ctx.source/'compilation').glob('*/*/probes.parquet'))):
        q=pd.read_parquet(p)
        q['source_probe_file']=str(p.relative_to(ctx.source))
        all_probes.append(q)
        if i%8==0:ctx.mark(f'F0 reading full source probe shard {i+1}')
    full=pd.concat(all_probes,ignore_index=True)
    # Preflight requests are reused by subsequent stages. Identical grid rows are one query.
    identity=['request_id','query_flow']
    duplicate=full.duplicated(identity,keep=False)
    if duplicate.any():
        d=full.loc[duplicate]
        for name in ['exact_cost_s','exact_potential','signed_cost_relative_error','signed_potential_relative_error']:
            if (d.groupby(identity)[name].nunique()>1).any():raise ValueError('Conflicting reused source probe '+name)
        full=full.drop_duplicates(identity)
    full=full.merge(requests[['request_id','runtime_class','status','is_exact','domain_Q']],on='request_id',validate='many_to_one')
    full['cost_denominator_s']=np.maximum(1.,abs(full.exact_cost_s))
    full['potential_denominator_flow_s']=np.maximum(1.,abs(full.exact_potential))
    ids=ridx.get_indexer(full.request_id)
    if (ids<0).any():raise ValueError('Probe has missing request')
    np.maximum.at(costs,ids,abs(full.signed_cost_relative_error.to_numpy()))
    np.maximum.at(pots,ids,abs(full.signed_potential_relative_error.to_numpy()))
    np.add.at(counts,ids,1)
    if (counts==0).any() or not np.allclose(costs,requests.cost_error,rtol=1e-12,atol=1e-14) or not np.allclose(pots,requests.potential_error,rtol=1e-12,atol=1e-14):
        raise ValueError('Full grid summaries do not reproduce request extrema')
    pf=ctx.out/'reports/full_grid_probe_population.parquet';frame_write(pf,full)
    for (klass,status),g in requests.groupby(['runtime_class','status']):
        for metric in ['cost_error','potential_error']:
            summaries.append(dict(population='ALL_REQUESTS',runtime_class=klass,status=status,grid_role='TRAIN_PLUS_INDEPENDENT',metric=metric,**summarize(g[metric])))
    for role,mask in [('TRAIN',full.role.str.contains('TRAIN')),('INDEPENDENT',full.role.str.contains('INDEPENDENT'))]:
        for (klass,status),g in full.loc[mask].groupby(['runtime_class','status']):
            for metric in ['signed_cost_relative_error','signed_potential_relative_error']:
                summaries.append(dict(population='ALL_GRID_POINTS',runtime_class=klass,status=status,grid_role=role,metric='absolute_'+metric,**summarize(abs(g[metric]))))
    occurrences=pd.read_parquet(ctx.out/'reports/accepted_link_occurrence_metadata.parquet')
    linked=occurrences.merge(requests[['request_id','cost_error','potential_error','train_count','independent_count']],left_on='calibration_request_id',right_on='request_id',how='left',validate='many_to_one')
    linked['grid_policy']=np.where(linked.request_id.notna(),'BOUND_TO_FULL_REQUEST_GRID','EXACT_ANALYTICAL_FUNCTION_NO_FIT_GRID')
    of=ctx.out/'reports/full_accepted_link_error_population.parquet';frame_write(of,linked)
    for (suite,klass),g in linked.groupby(['suite_id','runtime_class']):
        for metric in ['cost_error','potential_error']:
            summaries.append(dict(population='ACCEPTED_LINK_OCCURRENCES',suite_id=suite,runtime_class=klass,status='EMITTED',grid_role='TRAIN_PLUS_INDEPENDENT' if g.request_id.notna().all() else 'EXACT_ANALYTICAL_NO_FIT_GRID',metric=metric,total_link_occurrences=len(g),**summarize(g[metric])))
    sp=ctx.out/'reports/full_fit_error_summary.csv';frame_write(sp,pd.DataFrame(summaries))
    old=pd.read_parquet(ctx.source/'validation/path_fidelity.parquet')
    if len(old)!=1200 or old.path_id.nunique()!=600 or old.sample_od_id.nunique()!=200:raise ValueError('Old witness role coverage failed')
    old['evidence_kind']='EXISTING_200_ODS_TIMES_3_PATH_ROLES_TIMES_2_CHARGES'
    oldfile=ctx.out/'reports/old600_witness_components.csv';frame_write(oldfile,old)
    return [reqfile,pf,sp,of,oldfile]


def held_state_fit_audit(ctx):
    source=Path(ctx.source_config['source_pilot_output_root'])
    bundles=pd.read_parquet(source/'bundle_oracle/bundle_parameter_occurrences.parquet')
    occurrence=bundles.set_index(['suite_short_id','bundle_id']).canonical_primitive_key.to_dict()
    seq=[];wanted=set()
    for suite in ctx.config['suites']:
        links=pd.read_parquet(ctx.source/f'suites/{suite}/runtime_links.parquet')
        links=links.loc[links.runtime_class.eq('FIT_POWER4')]
        sequences=pd.read_parquet(ctx.source/f'suites/{suite}/macro_sequences.parquet')
        sequences=sequences.loc[sequences.corridor_id.isin(links.link_id)]
        sequences=sequences.sort_values(['corridor_id','sequence_position'])
        macro={k:tuple(occurrence[(suite,p)] for p in v.primitive_id) for k,v in sequences.groupby('corridor_id',sort=False)}
        for r in links.itertuples(index=False):
            seq.append((suite,r,macro[r.link_id]));wanted.update(macro[r.link_id])
    specs={r.canonical_primitive_key:spec_from_row(r) for r in bundles.drop_duplicates('canonical_primitive_key').loc[lambda x:x.canonical_primitive_key.isin(wanted)].itertuples(index=False)}
    cache={};rows=[]
    for i,(suite,r,keys) in enumerate(seq):
        for state,q in [('LOW',r.low_current_flow),('HIGH',r.high_current_flow)]:
            key=(keys,float(q))
            if key not in cache:cache[key]=exact_series_oracle(tuple(specs[k] for k in keys),float(q))
            exact=cache[key]
            if not exact.certified:raise ArithmeticError('Uncertified held-state oracle')
            predicted=r.tau_s+r.k4*q**4;potential=r.tau_s*q+r.k4*q**5/5
            rows.append(dict(suite_id=suite,link_id=r.link_id,model_id=r.dynamic_model_id,state=state,q_current=q,
                domain_Q=r.domain_Q,out_of_domain=q>r.domain_Q,exact_cost_s=exact.marginal_cost_s,
                runtime_cost_s=predicted,exact_potential_flow_s=exact.potential_flow_seconds,runtime_potential_flow_s=potential,
                signed_cost_relative_error=(predicted-exact.marginal_cost_s)/max(1.,abs(exact.marginal_cost_s)),
                signed_potential_relative_error=(potential-exact.potential_flow_seconds)/max(1.,abs(exact.potential_flow_seconds)),
                oracle_certified=exact.certified,used_for_fit_or_domain_or_selection=False,
                high_role='REUSED_SENSITIVITY_NOT_PRISTINE_TEST' if state=='HIGH' else 'PRIMARY_CURRENT'))
        if i%256==0:ctx.mark(f'F1 offline held-state fitted occurrence {i+1}/{len(seq)}')
    result=pd.DataFrame(rows)
    if len(result)!=16564 or result.duplicated(['suite_id','link_id','state']).any():raise ValueError('Held-state fitted occurrence coverage failed')
    path=ctx.out/'reports/all_fitted_LOW_HIGH_point_queries.csv';frame_write(path,result)
    summary=[]
    for (suite,state),g in result.groupby(['suite_id','state']):
        for metric in ['signed_cost_relative_error','signed_potential_relative_error']:
            summary.append(dict(suite_id=suite,state=state,metric='absolute_'+metric,
                population='EVERY_ACCEPTED_FIT_POWER4_OCCURRENCE',out_of_domain_count=int(g.out_of_domain.sum()),**summarize(abs(g[metric]))))
    sp=ctx.out/'reports/held_state_fit_error_summary.csv';frame_write(sp,pd.DataFrame(summary))
    return [path,sp]
