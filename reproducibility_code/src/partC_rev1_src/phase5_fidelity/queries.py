"""Serial origin batches, complete scalar evidence and compact path arrays."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from partC_rev1_src.phase5.campaign_io import (
    frame_write, write_json, npz_write, sha, binding_hash,
    publish_receipt, verify_receipt, file_record,
)
from partC_rev1_src.phase5.bundle_oracle import get_exact_root_call_count


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array,dtype=np.int64).tobytes()).hexdigest()


def read_query_rows(path):
    """Recover the exact binary64 values originally written to the CSV."""
    return pd.read_csv(path,keep_default_na=False,float_precision='round_trip')


def verify_query_receipt(ctx,receipt,dependency):
    record=verify_receipt(ctx.out,receipt)
    if record['dependency_hash']==dependency:return record
    # Explicit, hash-bound engineering evidence only; no tolerance-based reuse.
    for path in sorted((ctx.out/'engineering_epochs').glob('epoch_*/receipt_compatibility.json')):
        evidence=json.loads(path.read_text())
        for item in evidence['receipts']:
            if (item['relative_path']==receipt.relative_to(ctx.out).as_posix()
                and item['receipt_sha256']==sha(receipt)
                and item['original_dependency_hash']==record['dependency_hash']
                and item['current_dependency_hash']==dependency):
                return record
    raise ValueError('Changed query dependency has no exact evidenced compatibility binding')


def persist_graph(ctx,graph):
    m=graph.metadata
    key='_'.join(str(m.get(x,'NONE')) for x in ['suite','variant','state','charge_variant'])
    key += '_M'+str(len(m.get('restored_B',[])))+'_'+graph.fingerprint[:12]
    root=ctx.out/'graphs'/key
    receipt=root/'receipt.json'
    if receipt.exists():
        verify_receipt(ctx.out,receipt,graph.fingerprint)
        return key
    start=time.time()
    frame_write(root/'edges.parquet',graph.edges)
    npz_write(root/'nodes_and_adjacency.npz',dict(nodes=graph.nodes,tail=graph.tail,head=graph.head,
        order=graph.order,indptr=graph.indptr,cost_s=graph.cost))
    paths=[root/'edges.parquet',root/'nodes_and_adjacency.npz']
    if hasattr(graph,'connector_ledger'):
        frame_write(root/'connector_ledger.csv',graph.connector_ledger);paths.append(root/'connector_ledger.csv')
    write_json(root/'identity.json',dict(**m,fingerprint=graph.fingerprint,node_count=len(graph.nodes),
        edge_count=len(graph.edges),role_counts=graph.edges.role.value_counts().to_dict(),
        nonfinite_cost_count=int((~np.isfinite(graph.cost)).sum()),negative_cost_count=int((graph.cost<0).sum())))
    paths.append(root/'identity.json')
    publish_receipt(ctx.out,receipt,'GRAPH',graph.fingerprint,paths,start)
    return key


def query_batches(ctx, graph, sample, b, lifter=None, physical_reference=None,
                  max_batches=None):
    key=persist_graph(ctx,graph)
    root=ctx.out/'batches'/key
    groups=list(sample.groupby('origin_idx',sort=True))
    physical=graph.metadata['variant']=='PHYSICAL'
    all_rows=[];batch_receipts=[]
    for batch,begin in enumerate(range(0,len(groups),32)):
        if max_batches is not None and batch>=max_batches:break
        current=groups[begin:begin+32]
        batch_sample=pd.concat([x[1] for x in current])
        dependency=ctx.dependency(ctx.out/'sampling/frozen_od_sample.csv',
            modules=['queries.py','kernels.py','lift.py'],
            extra=dict(graph=graph.fingerprint,rows=batch_sample.stable_od_key.tolist(),
                physical_reference=None if physical else {str(k):float(physical_reference[k]) for k in batch_sample.stable_od_key}))
        receipt=root/f'batch_{batch:03d}.json'
        scalar=root/f'batch_{batch:03d}.csv';pathsfile=root/f'batch_{batch:03d}_paths.npz'
        if receipt.exists():
            record=verify_query_receipt(ctx,receipt,dependency)
            all_rows.append(read_query_rows(scalar));batch_receipts.append(record)
            ctx.mark(f'Reused query {key} batch {batch+1}/{(len(groups)+31)//32}')
            continue
        start=time.time();rows=[];native_paths=[];lifted_paths=[];native_ptr=[0];lift_ptr=[0]
        roots_before=get_exact_root_call_count()
        tree_seconds=lift_seconds=0.
        for origin,subset in current:
            source_name=('p:' if physical else 'od_o:')+str(b.graph.node_idx_to_uid[origin])
            source=graph.node_index[source_name]
            ts=time.perf_counter();dist,pred=graph.tree(source);tree_seconds+=time.perf_counter()-ts
            for row in subset.itertuples(index=False):
                target=graph.node_index[('p:' if physical else 'od_d:')+str(b.graph.node_idx_to_uid[row.destination_idx])]
                distance=float(dist[target]);reachable=bool(np.isfinite(distance))
                path=graph.path(pred,source,target) if reachable else np.empty(0,np.int64)
                if reachable and not np.isclose(graph.cost[path].sum(),distance,rtol=1e-10,atol=1e-7):
                    raise ValueError('Native minimum path cost does not close')
                reference=distance if physical else float(physical_reference[row.stable_od_key])
                lift=np.empty(0,np.int64)
                if physical:
                    lift=graph.ref[path]
                    result=dict(lift_status='VERIFIED' if reachable else 'UNREACHABLE',D_lift_s=distance if reachable else None,regret=0. if reachable else None,
                        failure_layer=-1,failed_B_ids=[],failure_from_physical_nodes=[],failure_to_physical_nodes=[])
                elif reachable:
                    ts=time.perf_counter();result,lift=lifter.lift(graph,path,int(origin),int(row.destination_idx),reference)
                    lift_seconds+=time.perf_counter()-ts
                else:
                    result=dict(lift_status='UNREACHABLE',D_lift_s=None,regret=None,failure_layer=-1,failed_B_ids=[],failure_from_physical_nodes=[],failure_to_physical_nodes=[])
                native_paths.extend(path);lifted_paths.extend(lift);native_ptr.append(len(native_paths));lift_ptr.append(len(lifted_paths))
                subset_edges=graph.edges.iloc[path]
                fixed=float(subset_edges.fixed_s.sum());fit=float(subset_edges.fit_delta_s.sum())
                regional=float(subset_edges.loc[subset_edges.role.eq('REGIONAL_PHYSICAL'),'cost_s'].sum())
                restored=float(subset_edges.loc[subset_edges.role.eq('RESTORED_B_PHYSICAL'),'cost_s'].sum())
                data=dict(stable_od_key=row.stable_od_key,accepted_od_row=int(row.accepted_od_row),
                    origin_idx=int(origin),destination_idx=int(row.destination_idx),source_set=row.source_set,
                    stratum=int(row.stratum),N_h=int(row.N_h),n_h=int(row.n_h),demand=float(row.demand),ratio_weight=float(row.ratio_weight),
                    graph_key=key,graph_variant=graph.metadata['variant'],suite_id=graph.metadata['suite'],state=graph.metadata['state'],
                    charge_variant=graph.metadata.get('charge_variant','NONE'),restored_B_count=len(graph.metadata.get('restored_B',[])),
                    D_phys_s=reference,D_hybrid_s=distance if reachable else None,reachable=reachable,
                    new_unreachable=bool(np.isfinite(reference) and not reachable),
                    signed_distance_error_s=distance-reference if reachable else None,
                    absolute_distance_error_s=abs(distance-reference) if reachable else None,
                    absolute_relative_distance_error=abs(distance-reference)/max(1.,reference) if reachable else None,
                    fixed_B_charge_s=fixed,runtime_fit_term_s=fit,regional_dynamic_s=regional,restored_B_dynamic_s=restored,
                    native_path_edge_count=len(path),native_path_edge_indices_sha256=array_hash(path),
                    native_path_edge_ids_sha256=hashlib.sha256('\n'.join(subset_edges.edge_id).encode()).hexdigest(),
                    physical_lift_arc_count=len(lift),physical_lift_indices_sha256=array_hash(lift),
                    physical_lift_arc_ids_sha256=hashlib.sha256('\n'.join(b.graph.arc_idx_to_directed_arc_id[lift].astype(str)).encode()).hexdigest(),
                    out_of_domain_path_edges=int(subset_edges.out_of_domain.sum()),
                    native_path_artifact=str(pathsfile.relative_to(ctx.out)),path_row=len(rows),
                    high_role='REUSED_SENSITIVITY_NOT_PRISTINE_TEST' if graph.metadata['state']=='HIGH' else 'LOW_CURRENT',
                    **result)
                for field in ['failed_B_ids','failure_from_physical_nodes','failure_to_physical_nodes']:
                    data[field+'_json']=json.dumps(data.pop(field),separators=(',',':'))
                if not physical and result['lift_status']=='VERIFIED':
                    data['B_charge_error_vs_lift_s']=fixed+restored-result['lift_B_internal_s']
                    native_inter=distance-fixed-regional-restored
                    data['aggregation_and_alignment_vs_lift_s']=native_inter-fit-result['lift_inter_B_s']
                    data['regional_vs_lift_s']=regional-result['lift_regional_s']
                    data['lifted_route_regret_s']=result['D_lift_s']-reference
                    closure=distance-reference-(data['B_charge_error_vs_lift_s']+data['aggregation_and_alignment_vs_lift_s']+fit+data['regional_vs_lift_s']+data['lifted_route_regret_s'])
                    if abs(closure)>1e-7+1e-10*max(1.,abs(distance),abs(reference)):
                        raise ValueError('New-path signed component closure failed')
                    data['component_closure_s']=closure
                rows.append(data)
            ctx.mark(f'Query {key} origin {begin+len({r["origin_idx"] for r in rows})}/{len(groups)}')
        if get_exact_root_call_count()!=roots_before:raise ValueError('Oracle was invoked inside shortest-path/lift queries')
        frame_write(scalar,pd.DataFrame(rows))
        npz_write(pathsfile,dict(native_ptr=np.asarray(native_ptr,np.int64),native_edges=np.asarray(native_paths,np.int64),
            lift_ptr=np.asarray(lift_ptr,np.int64),physical_arcs=np.asarray(lifted_paths,np.int64),
            stable_od_keys=np.asarray([r['stable_od_key'] for r in rows])))
        publish_receipt(ctx.out,receipt,'QUERY_BATCH',dependency,[scalar,pathsfile],start,
            graph_fingerprint=graph.fingerprint,unique_origins=len(current),OD_rows=len(rows),
            tree_seconds=tree_seconds,lift_seconds=lift_seconds,online_oracle_calls=0)
        all_rows.append(pd.DataFrame(rows));batch_receipts.append(json.loads(receipt.read_text()))
    return pd.concat(all_rows,ignore_index=True),batch_receipts,key


def paths_from_rows(out,rows,kind='physical'):
    result={}
    for path,g in rows.groupby('native_path_artifact',sort=False):
        with np.load(out/path,allow_pickle=False) as data:
            ptr=data['lift_ptr' if kind=='physical' else 'native_ptr'];values=data['physical_arcs' if kind=='physical' else 'native_edges']
            for r in g.itertuples(index=False):
                i=int(r.path_row);result[r.stable_od_key]=values[ptr[i]:ptr[i+1]].copy()
    return result
