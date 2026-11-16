"""Independent replay of stored OD path witnesses against all frozen supports."""
from __future__ import annotations
import numpy as np
from numba import njit
from .control import read_json,write_json,sha,array_hash

@njit(cache=True)
def audit_forest(pred,group_indices,origins,offsets,destinations,serve,tail,head,enabled,regions,keys,key_offsets,n):
    checked=0;segments=np.zeros(len(regions),np.int64)
    for local in range(len(group_indices)):
        group=group_indices[local];source=origins[group]
        if pred[local,source]!=-1:raise ValueError('Witness root mismatch')
        for row in range(offsets[group],offsets[group+1]):
            if not serve[row]:continue
            checked+=1;node=destinations[row]
            active=np.full(len(regions),-1,np.int64);exit_node=np.full(len(regions),-1,np.int64)
            hops=0
            while node!=source:
                edge=pred[local,node]
                if edge<0 or edge>=len(tail) or head[edge]!=node or not enabled[edge]:raise ValueError('Illegal OD witness arc')
                for suite in range(len(regions)):
                    region=regions[suite,edge]
                    if region!=active[suite]:
                        if active[suite]>=0:
                            key=(np.uint64(active[suite])*np.uint64(n)+np.uint64(node))*np.uint64(n)+np.uint64(exit_node[suite])
                            at=np.searchsorted(keys[key_offsets[suite]:key_offsets[suite+1]],key)+key_offsets[suite]
                            if at>=key_offsets[suite+1] or keys[at]!=key:raise ValueError('OD witness regional segment absent from AC support')
                            segments[suite]+=1
                        active[suite]=region;exit_node[suite]=node
                node=tail[edge];hops+=1
                if hops>n:raise ValueError('OD witness cycle')
            for suite in range(len(regions)):
                if active[suite]>=0:
                    key=(np.uint64(active[suite])*np.uint64(n)+np.uint64(source))*np.uint64(n)+np.uint64(exit_node[suite])
                    at=np.searchsorted(keys[key_offsets[suite]:key_offsets[suite+1]],key)+key_offsets[suite]
                    if at>=key_offsets[suite+1] or keys[at]!=key:raise ValueError('Origin regional segment absent from AC support')
                    segments[suite]+=1
    return checked,segments

def audit(ctx):
    from .topology import load_common
    graph,od,a,p=load_common(ctx.root)
    freeze=read_json(ctx.root/'SUPPORT_FREEZE.json')
    if freeze['status']!='PASS':raise ValueError('Support is not frozen')
    ctx.bind(ctx.root/'SUPPORT_FREEZE.json','FROZEN_SUPPORT_FOR_WITNESS_AUDIT')
    frozen={record['suite_id']:record for record in freeze['suites']}
    support=[]
    for sid in ctx.c['suite_ids']:
        ctx.bind(ctx.root/'acnhn'/sid/'graph.npz','FROZEN_AC_GRAPH',frozen[sid]['graph_sha256'])
        with np.load(ctx.root/'acnhn'/sid/'graph.npz',allow_pickle=False) as z:support.append(z['pair_keys'])
    keys=np.concatenate(support);offsets=np.r_[0,np.cumsum([len(k) for k in support])]
    records=[]
    for scenario in ['open','i95']:
        root=ctx.root/'references'/scenario
        manifest=read_json(root/'initial_witnesses/MANIFEST.json')
        ctx.bind(root/'initial_witnesses/MANIFEST.json','COMPLETE_INITIAL_OD_FOREST_MANIFEST')
        ctx.bind(root/'coverage.npz','WITNESS_EXACT_OD_COVERAGE')
        if manifest['graph_identity']!=graph.identity:raise ValueError('Forest belongs to another P graph')
        if manifest['OD_identity']!=od.identity:raise ValueError('Initial forest OD identity differs')
        with np.load(root/'coverage.npz',allow_pickle=False) as z:serve=z['serve']
        enabled=~a['closed_mask'] if scenario=='i95' else np.ones(len(graph.tail),bool)
        if manifest['enabled_identity']!=array_hash({'enabled':enabled}) or manifest['cost_identity']!=array_hash({'cost':a['base']}):raise ValueError('Initial forest mask or zero-flow cost differs')
        checked=0;segments=np.zeros(len(support),np.int64);groups=[]
        for record in manifest['forests']:
            ctx.guard(deadline=ctx.c['maximum_cumulative_stage_deadlines'][0])
            if sha(record['path'])!=record['sha256']:raise ValueError('Forest checksum changed')
            with np.load(record['path'],allow_pickle=False) as z:
                pred=z['predecessor'];index=z['origin_group_indices']
            count,parts=audit_forest(pred,index,od.origins,od.offsets,od.destinations,serve,
                graph.tail,graph.head,enabled,p['edge_region'],keys,offsets,len(graph.node_ids))
            checked+=count;segments+=parts;groups.extend(index.tolist())
        if groups!=list(range(len(od.origins))) or checked!=int(serve.sum()):raise ValueError('Incomplete OD forest coverage')
        records.append(dict(scenario=scenario,checked_OD_rows=checked,projected_segments=segments,
            all_five_resolutions=True,all_directed_P_arcs_and_masks_valid=True,all_segments_in_frozen_support=True))
    result=dict(status='PASS',checks=records,graph_identity=graph.identity,
        legal_edge_lifts='Independently verified by detailed local shortest-path solves in SUPPORT_FREEZE.json')
    result['support_freeze_sha256']=sha(ctx.root/'SUPPORT_FREEZE.json')
    write_json(ctx.root/'COMPLETE_WITNESS_AUDIT.json',result);ctx.flush_bindings();return result
