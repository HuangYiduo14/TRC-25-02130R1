"""Fresh-process reopen of every active graph, batch, path and scalar result."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .context import Context
from .graph import Graph
from .queries import array_hash
from partC_rev1_src.phase5.campaign_data import load_accepted
from partC_rev1_src.phase5.campaign_io import verify_receipt,write_json,sha


def verify(ctx):
    start=time.time();b=load_accepted(ctx.source_config,'s03')
    full=pd.read_csv(ctx.out/'results/shortest_path_scalar_metrics.csv')
    assert len(full)==44688 and not full.duplicated(['graph_key','stable_od_key']).any()
    sample=pd.read_csv(ctx.out/'sampling/frozen_od_sample.csv')
    assert len(sample)==456 and sample.stable_od_key.nunique()==456
    graph_count=batch_count=path_count=lift_count=0;max_native_closure=max_lift_closure=0.
    for key,rows in full.groupby('graph_key',sort=True):
        base=ctx.out/'graphs'/key
        verify_receipt(ctx.out,base/'receipt.json')
        identity=json.loads((base/'identity.json').read_text())
        metadata={k:v for k,v in identity.items() if k not in ['fingerprint','node_count','edge_count','role_counts','nonfinite_cost_count','negative_cost_count']}
        edges=pd.read_parquet(base/'edges.parquet')
        with np.load(base/'nodes_and_adjacency.npz',allow_pickle=False) as arrays:
            graph=Graph(arrays['nodes'],edges,metadata)
            if graph.fingerprint!=identity['fingerprint'] or not np.array_equal(graph.cost,arrays['cost_s']):raise ValueError('Fresh graph identity mismatch')
        physical_cost=(b.low_physical if identity['state']=='LOW' else b.high_physical).current_cost_s
        # One deterministic origin per configuration is rechecked in a fresh process.
        first=rows.sort_values(['origin_idx','stable_od_key']).iloc[0]
        source=graph.node_index[('p:' if identity['variant']=='PHYSICAL' else 'od_o:')+str(b.graph.node_idx_to_uid[int(first.origin_idx)])]
        target=graph.node_index[('p:' if identity['variant']=='PHYSICAL' else 'od_d:')+str(b.graph.node_idx_to_uid[int(first.destination_idx)])]
        distance,pred=graph.tree(source);native=graph.path(pred,source,target)
        if not np.isclose(distance[target],float(first.D_hybrid_s),rtol=1e-10,atol=1e-7) or array_hash(native)!=first.native_path_edge_indices_sha256:
            raise ValueError('Fresh deterministic search differs')
        for path,group in rows.groupby('native_path_artifact',sort=True):
            path=ctx.out/path;receipt=path.with_name(path.name.replace('_paths.npz','.json'))
            verify_receipt(ctx.out,receipt);batch_count+=1
            with np.load(path,allow_pickle=False) as arrays:
                for r in group.itertuples(index=False):
                    i=int(r.path_row)
                    ni=arrays['native_edges'][arrays['native_ptr'][i]:arrays['native_ptr'][i+1]]
                    pi=arrays['physical_arcs'][arrays['lift_ptr'][i]:arrays['lift_ptr'][i+1]]
                    if str(arrays['stable_od_keys'][i])!=r.stable_od_key or array_hash(ni)!=r.native_path_edge_indices_sha256 or array_hash(pi)!=r.physical_lift_indices_sha256:
                        raise ValueError('Path identity mismatch on fresh reopen')
                    if len(ni) and np.any(graph.head[ni[:-1]]!=graph.tail[ni[1:]]):raise ValueError('Native path discontinuity on reopen')
                    if bool(r.reachable):
                        diff=abs(float(graph.cost[ni].sum())-float(r.D_hybrid_s));max_native_closure=max(max_native_closure,diff)
                        if diff>1e-7+1e-10*max(1.,abs(float(r.D_hybrid_s))):raise ValueError('Native scalar closure failed on reopen')
                    if r.lift_status=='VERIFIED':
                        if len(pi):
                            if np.any(b.graph.head[pi[:-1]]!=b.graph.tail[pi[1:]]) or b.graph.tail[pi[0]]!=r.origin_idx or b.graph.head[pi[-1]]!=r.destination_idx:raise ValueError('Fresh physical lift continuity failed')
                        elif r.origin_idx!=r.destination_idx:raise ValueError('False empty lift')
                        value=float(physical_cost[pi].sum());difference=abs(value-float(r.D_lift_s));max_lift_closure=max(max_lift_closure,difference)
                        tolerance=1e-7+1e-10*max(1.,float(r.D_phys_s))
                        if difference>tolerance or value<float(r.D_phys_s)-tolerance:raise ValueError('Fresh lift cost/regret failure')
                        lift_count+=1
                    elif r.lift_status=='UNLIFTABLE':
                        if len(pi) or not pd.isna(r.regret) or not pd.isna(r.D_lift_s):raise ValueError('Unliftable path has fabricated lift or regret')
                    path_count+=1
        graph_count+=1;ctx.mark(f'Fresh reopen graph {graph_count}/98; {path_count}/44688 scalar paths verified')
    if graph_count!=98 or path_count!=44688:raise ValueError('Fresh reopen incomplete')
    # All stages before the seal have their own immutable, nonempty receipts.
    for name in ['F0','F1','F2','F3','F4','F5']:
        verify_receipt(ctx.out,ctx.out/'receipts'/f'{name}.json')
    with np.load(ctx.out/'sampling/frozen_current_states.npz',allow_pickle=False) as states:
        for label,physical in [('LOW',b.low_physical),('HIGH',b.high_physical)]:
            if not np.array_equal(states[label+'_cost_s'],physical.current_cost_s) or not np.array_equal(states[label+'_current_flow'],physical.current_flow):raise ValueError('Frozen accepted CURRENT changed')
    result=dict(status='PASS',fresh_process=True,graphs=graph_count,batches=batch_count,scalar_paths=path_count,
        verified_continuous_lifts=lift_count,deterministic_fresh_origins=graph_count,max_native_scalar_closure_s=max_native_closure,
        max_lift_scalar_closure_s=max_lift_closure,elapsed_seconds=time.time()-start,
        no_assignment=True,physical_states_unchanged=True,source_manifest_sha256=sha(ctx.out/'validation/source_bindings_initial.csv'))
    write_json(ctx.out/'validation/fresh_process_reopen.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();ctx=Context(args.config)
    try:print(json.dumps(verify(ctx),indent=2))
    finally:ctx.close()


if __name__=='__main__':main()
