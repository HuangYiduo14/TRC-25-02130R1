"""Freeze exact regional active pairs and build all ACNHNs from one P graph."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .control import save_npz,write_json,read_json,array_hash,sha
from .oracle import Network,Demand,aon,decode_keys,load_traces

def local_demand(source,target,values):
    source=np.asarray(source,np.int64);target=np.asarray(target,np.int64);values=np.asarray(values,float)
    order=np.argsort(source,kind='stable');origins,counts=np.unique(source[order],return_counts=True)
    return Demand(origins,np.r_[0,np.cumsum(counts)],target[order],values[order],np.ones(len(order),bool)),order

def aligned_trace(keys,trace):
    values=np.zeros(len(keys));at=np.searchsorted(keys,trace.keys)
    if np.any(at>=len(keys)) or not np.array_equal(keys[at],trace.keys):raise ValueError('State trace is outside frozen accepted support')
    values[at]=trace.values;return values

def build(ctx):
    from .topology import load_common
    graph,od,a,p=load_common(ctx.root)
    sidlist=ctx.c['suite_ids'];n=len(graph.node_ids);root=ctx.root
    common_region=p['edge_region']
    ref_roots=[root/'references'/scenario for scenario in ['open','i95']]
    reports=[read_json(x/'RESULT.json') for x in ref_roots]
    if not all(x['final_certificate_fresh'] for x in reports):raise ValueError('Reference state needs its own fresh certificate')
    ever=[];flows=[];coverage=[]
    for reference,report in zip(ref_roots,reports):
        if report['graph_identity']!=graph.identity or report['trace_count']!=len(sidlist):raise ValueError('Reference graph or trace suite identity differs')
        ctx.bind(reference/'RESULT.json','FROZEN_P_REFERENCE')
        ctx.bind(reference/'final_flow.npz','CERTIFIED_P_FLOW')
        ctx.bind(reference/'coverage.npz','REFERENCE_EXACT_OD_COVERAGE')
        ctx.bind(reference/'final_traces.npz','STATE_CONSISTENT_TRACES')
        ctx.bind(reference/'ever_active_pairs.npz','ACCEPTED_PATH_SUPPORT')
        with np.load(reference/'final_flow.npz',allow_pickle=False) as z:reference_flow=z['flow']
        if array_hash({'flow':reference_flow})!=report['certificate']['state_hash']:raise ValueError('Reference certificate belongs to another flow')
        with np.load(reference/'ever_active_pairs.npz',allow_pickle=False) as z:ever.append([z[f'keys_{i}'] for i in range(len(sidlist))])
        flows.append(load_traces(reference/'final_traces.npz'))
        with np.load(reference/'coverage.npz',allow_pickle=False) as z:coverage.append(z['serve'])
        reference_od=Demand(od.origins,od.offsets,od.destinations,od.demand,coverage[-1])
        if reference_od.identity!=report['OD_identity']:raise ValueError('Reference served OD identity mismatch')
        if len(flows[-1])!=len(sidlist) or len(ever[-1])!=len(sidlist):raise ValueError('Incomplete regional trace coverage')
    summaries=[]
    for suite,sid in enumerate(sidlist):
        ctx.guard(deadline=ctx.c['maximum_cumulative_stage_deadlines'][0])
        keys=np.union1d(ever[0][suite],ever[1][suite]);regions,sources,targets=decode_keys(keys,n)
        if not len(keys):raise ValueError('No active pairs in '+sid)
        if np.any(sources==targets):raise ValueError('Positive-cost cyclic regional segment should not be in shortest witness')
        baseline_open=aligned_trace(keys,flows[0][suite]);baseline_i95=aligned_trace(keys,flows[1][suite])
        all_c0=np.zeros(len(keys));native_index=np.empty(len(keys),np.int64);regional_records=[]
        regional_edge_ids=np.flatnonzero(common_region[suite]>=0)
        edge_order=regional_edge_ids[np.argsort(common_region[suite,regional_edge_ids],kind='stable')]
        edge_regions=common_region[suite,edge_order]
        for region in np.unique(regions):
            ctx.guard(deadline=ctx.c['maximum_cumulative_stage_deadlines'][0]);start=time.perf_counter()
            begin,end=np.searchsorted(regions,[region,region+1]);selection=slice(begin,end)
            lo,hi=np.searchsorted(edge_regions,[region,region+1]);edges=edge_order[lo:hi]
            if not len(edges):raise ValueError('Active pair without regional physical graph')
            nodes=np.unique(np.r_[graph.tail[edges],graph.head[edges]])
            tail=np.searchsorted(nodes,graph.tail[edges]);head=np.searchsorted(nodes,graph.head[edges])
            src=np.searchsorted(nodes,sources[selection]);dst=np.searchsorted(nodes,targets[selection])
            if np.any(src>=len(nodes)) or np.any(dst>=len(nodes)) or not np.array_equal(nodes[src],sources[selection]) or not np.array_equal(nodes[dst],targets[selection]):raise ValueError('Regional endpoint does not lift')
            local=Network(tail,head,np.zeros(len(nodes),bool),graph.edge_ids[edges],graph.node_ids[nodes])
            zero_od,order=local_demand(src,dst,np.zeros(end-begin))
            loaded=aon(local,zero_od,a['base'][edges],guard=lambda:ctx.guard(deadline=ctx.c['maximum_cumulative_stage_deadlines'][0]))
            c0=np.empty(end-begin);c0[order]=loaded['distances']
            if not np.isfinite(c0).all():raise ValueError('Frozen pair lacks legal directed P regional path')
            all_c0[selection]=c0;native_index[selection]=np.arange(end-begin)
            directory=root/'regions'/sid/f'{region:06d}';directory.mkdir(parents=True,exist_ok=False)
            arrays=dict(tail=tail,head=head,node_ids=local.node_ids,edge_ids=local.edge_ids,base=a['base'][edges],k4=a['k4'][edges],
                pair_source=src,pair_target=dst,pair_keys=keys[selection],baseline_open=baseline_open[selection],
                baseline_i95=baseline_i95[selection],c0=c0,P_edge_indices=edges,P_node_indices=nodes,
                physical_capacity_reference=np.array(np.median(a['capacity'][edges]),np.float64))
            save_npz(directory/'graph.npz',**arrays)
            record=dict(suite_id=sid,region_index=int(region),graph_identity=local.identity,
                input_identity=array_hash(arrays),path=str(directory/'graph.npz'),sha256=sha(directory/'graph.npz'),
                pairs=end-begin,nodes=len(nodes),edges=len(edges),all_pairs_lift=True,local_FFTT_seconds=time.perf_counter()-start,
                physical_capacity_reference=float(arrays['physical_capacity_reference']),
                physical_capacity_reference_provenance='Median of once-scaled accepted physical capacities on this exact regional edge set; annual-kton flow units')
            write_json(directory/'REGION.json',record);regional_records.append(record)
        backbone=np.flatnonzero(a['edge_role']==0);ports=np.flatnonzero(a['edge_role']==2)
        bcount=ctx.c['topology']['expected_backbone_count']
        if not np.array_equal(backbone,np.arange(bcount)):raise ValueError('Backbone native order changed')
        ids=np.r_[graph.edge_ids[backbone],np.array([f'pac_pair:{sid}:{key}' for key in keys]),graph.edge_ids[ports]]
        tail=np.r_[graph.tail[backbone],sources,graph.tail[ports]];head=np.r_[graph.head[backbone],targets,graph.head[ports]]
        compiled=Network(tail,head,graph.blocked,ids,graph.node_ids)
        base=np.r_[a['base'][backbone],all_c0,a['base'][ports]];k4=np.r_[a['k4'][backbone],np.zeros(len(keys)),a['k4'][ports]]
        closed=np.r_[a['closed_mask'][backbone],np.zeros(len(keys)+len(ports),bool)]
        target_root=root/'acnhn'/sid;target_root.mkdir(parents=True,exist_ok=False)
        save_npz(target_root/'graph.npz',tail=tail,head=head,blocked=graph.blocked,node_ids=graph.node_ids,edge_ids=ids,
            base=base,k4=k4,closed_mask=closed,pair_keys=keys,pair_region=regions,pair_source=sources,pair_target=targets,
            c0=all_c0,baseline_open=baseline_open,baseline_i95=baseline_i95,
            backbone_count=np.array(bcount),port_count=np.array(len(ports)))
        pairs=pd.DataFrame(dict(pair_key=keys,region_index=regions,native_index=native_index,
            source_node=sources,target_node=targets,c0=all_c0,baseline_open=baseline_open,baseline_i95=baseline_i95))
        pairs.to_parquet(target_root/'pairs.parquet',index=False)
        reach=[]
        for scenario,serve in zip(['open','i95'],coverage):
            enabled=~closed if scenario=='i95' else np.ones(len(ids),bool)
            result=aon(compiled,od,base,enabled=enabled,guard=lambda:ctx.guard(deadline=ctx.c['maximum_cumulative_stage_deadlines'][0]),workers=ctx.c['resources']['regional_workers'])
            actual=od.serve&np.isfinite(result['distances'])
            if not np.array_equal(actual,serve):
                save_npz(target_root/f'FEASIBILITY_FAILURE_{scenario}.npz',P_serve=serve,AC_serve=actual)
                raise ValueError(f'{sid}/{scenario} exact served OD membership differs: {int(np.sum(actual!=serve))}')
            save_npz(target_root/f'coverage_{scenario}.npz',serve=actual,shortest_cost=result['distances'])
            reach.append(dict(scenario=scenario,exact_OD_membership=True,served_rows=int(actual.sum()),served_demand=float(od.demand[actual].sum())))
        # Every stored P OD witness is a concatenation of unchanged B/port arcs and maximal
        # regional runs. Its exact runs are in the accepted witness union; each
        # surrogate edge independently has a legal regional lift (FFTT audit).
        report=dict(suite_id=sid,graph_identity=compiled.identity,graph_path=str(target_root/'graph.npz'),graph_sha256=sha(target_root/'graph.npz'),backbone_count=bcount,active_pairs=len(keys),active_regions=len(regional_records),
            backbone_identity=array_hash({'tail':tail[:bcount],'head':head[:bcount],'ids':ids[:bcount],'base':base[:bcount],'k4':k4[:bcount],'closed':closed[:bcount]}),
            OD_identity=od.identity,region_records=regional_records,coverage=reach,support_frozen_before_labels=True)
        write_json(target_root/'REPORT.json',report);summaries.append(report)
    if len({r['backbone_identity'] for r in summaries})!=1 or len({r['OD_identity'] for r in summaries})!=1:raise ValueError('Cross-resolution B/OD identity mismatch')
    write_json(root/'SUPPORT_FREEZE.json',dict(status='PASS',suites=summaries,source_references=reports))
    ctx.flush_bindings();return summaries
