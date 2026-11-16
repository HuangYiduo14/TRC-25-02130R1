"""Scoped source-label replay and a frozen, non-TEST-selected hard-region docket."""
from pathlib import Path
import argparse,heapq,time,traceback,os
import numpy as np
import pandas as pd
from .common import root,rd,read_json,write_json,save_npz,sha,digest,input_hash,admit,native_threads
from .policy import seed_for
from .source import Sources

def panel_plan(c):
    p=root(c)/'control/audit_plan.json'
    if p.exists():return read_json(p)
    reg=pd.read_parquet(root(c)/'control/source_region_registry.parquet');active=reg.loc[reg.m_active.gt(0)].copy()
    active['patch_seed']=[seed_for(r,'panel') for r in active.region_id]
    metrics=pd.read_parquet(root(c)/'control/old_train_valid_metrics.parquet')
    metrics=metrics.loc[metrics.model.eq('ICNN') & metrics.family.eq('ALL') & metrics.grade.eq('ALL')]
    valid=metrics.loc[metrics.split.eq('VALID')].sort_values(['MARE_equal_pairs','region_id'],ascending=[False,True])
    train=metrics.loc[metrics.split.eq('TRAIN')].sort_values(['composite_loss','region_id'],ascending=[False,True])
    buckets=[g.sort_values(['patch_seed','region_id']).region_id.tolist() for _,g in active.groupby('stratum',sort=True)]
    stable=[b[i] for i in range(max(map(len,buckets))) for b in buckets if i<len(b)]
    groups=dict(top16_old_valid=valid.head(16).region_id.tolist(),top8_old_train_loss=train.head(8).region_id.tolist(),
        top8_low_exposure=active.sort_values(['low_exposure','region_id'],ascending=[False,True]).head(8).region_id.tolist(),dimension_stratified_8=stable[:8])
    audits=list(dict.fromkeys(sum(groups.values(),[])))
    legacy=list(dict.fromkeys(valid.head(8).region_id.tolist()+stable[:8]))
    for rid in stable:
        if len(legacy)>=16:break
        if rid not in legacy:legacy.append(rid)
    result=dict(audit_regions=audits,groups=groups,legacy_panel=legacy,matched_update_panel=legacy[:4]+[r for r in stable if r not in legacy[:4]][:4],
        precision_regions=valid.head(6).region_id.tolist(),precision_roles=['LOW','LARGEST_OLD_TRAIN_ROW_LOSS'],precision_samples_max=12,
        selection_source='OLD TRAIN/VALID + LOW metadata; never TEST/HIGH',plan_frozen_unix=time.time())
    assert len(audits)<=40 and len(legacy)==16 and len(result['matched_update_panel'])==8
    write_json(p,result);return result

def independent_price(a,z):
    costs=a['t0']*(1+a['alpha']*(z/a['capacity'])**a['beta'])
    value=float(np.sum(a['t0']*z*(1+a['alpha']/(a['beta']+1)*(z/a['capacity'])**a['beta'])))
    distances=[];n=len(a['blocked_through'])
    for g in range(len(a['source_offsets'])-1):
        starts=a['source_nodes'][a['source_offsets'][g]:a['source_offsets'][g+1]];ss=set(starts.tolist())
        d=np.full(n,np.inf);heap=[]
        for v in starts:d[v]=0.;heapq.heappush(heap,(0.,int(v)))
        while heap:
            dv,v=heapq.heappop(heap)
            if dv!=d[v]:continue
            if a['blocked_through'][v] and v not in ss:continue
            for arc in range(a['indptr'][v],a['indptr'][v+1]):
                w=int(a['head'][arc]);nd=dv+float(costs[arc])
                if nd<d[w]:d[w]=nd;heapq.heappush(heap,(nd,w))
        distances.append(d)
    tau=np.array([min(distances[a['choice_group'][j]][a['choice_target'][j]] for j in range(a['choice_offsets'][k],a['choice_offsets'][k+1])) for k in range(len(a['choice_offsets'])-1)])
    return costs,value,tau

def source_oracle(source,rid,a):
    from partC_rev1_src.phase6.oracle import Oracle
    o=Oracle(a);rel='regions/'+rid.removeprefix('partC_p2_region_')+'/labels'
    index=source.json6(rel+'/catalog_index.json')
    for seg in index['segments']:
        p=source.path6(rel+'/'+seg['relative_path']);assert sha(p)==seg['sha256']
        o.import_catalog(source.npz6(rel+'/'+seg['relative_path']))
    assert len(o.paths)==index['path_count']
    return o

def region_audit(c,rid):
    native_threads()
    from partC_rev1_src.phase6.oracle import Oracle,Result
    from partC_rev1_src.phase6.registry import make_choices
    from partC_rev1_src.phase7.model import frozen_eval
    from .model import PowerResidualPotential,weighted_power4_fit,calibrate_residual_unit
    import torch
    torch.set_num_threads(1)
    dest=rd(c,rid)/'audit';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'RESULT.json').exists():return read_json(dest/'RESULT.json')
    source=Sources(c);start=time.perf_counter()
    try:
        plan=panel_plan(c);pack=source.old_pack(rid);d,a,pairs,s=source.local(rid);old=source.old_arrays(rid)
        rel='regions/'+rid.removeprefix('partC_p2_region_')
        fm=source.npz6(rel+'/fine_members.npz');members={int(k):fm['local_nodes'][fm['offsets'][i]:fm['offsets'][i+1]] for i,k in enumerate(fm['terminal_indices'])}
        choices=make_choices(pairs.source_terminal_index.to_numpy(),pairs.target_terminal_index.to_numpy(),members)
        assert all(np.array_equal(a[k],v) for k,v in choices.items()),'Fine fiber omitted or reordered'
        fine=source.frame6(rel+'/LOW_native_fine_trace.parquet');fine=fine.loc[fine.coarse_key.isin(pairs.coarse_key)&fine.amount.gt(0)]
        mapping=pd.Index(pairs.coarse_key).get_indexer(fine.coarse_key)
        q=np.bincount(mapping,weights=fine.amount,minlength=len(pairs));assert np.allclose(q,pack.s['low'],rtol=1e-12,atol=1e-9)
        hierarchy=source.json6(rel+'/baseline_status.json')['hierarchy']
        assert hierarchy.get('policy_decision')!='LOCAL_ERROR'
        idx=pack.indices('TRAIN');u,t,L,U,eps=pack.fields(idx)
        row_losses=[];max_perturb=0.;saturation=[]
        for i in range(0,len(u),128):
            _,pred=frozen_eval(old,u[i:i+128]);row_losses.extend(np.sum(((pred-t[i:i+128])/pack.s['cost_scale'])**2*pack.s['pair_weight'],axis=1))
            logits=(u[i:i+128]/old['scale'])@old['A1'].T+old['b1']
            saturation.append(float(np.mean((logits>40)|(logits< -40))))
        losses=np.asarray(row_losses);rank=int(np.argmax(losses));leverage=int(np.argmax(np.max(u/pack.s['low'],axis=1)))
        random=int(np.random.default_rng(seed_for(rid,'audit_random')).integers(len(idx)))
        train_selected=list(dict.fromkeys([rank,leverage,random]))
        lowindex=int(idx[np.flatnonzero(np.all(u==pack.s['low'],axis=1))[0]])
        selected=list(dict.fromkeys([lowindex]+[int(idx[k]) for k in train_selected]))
        o=source_oracle(source,rid,a);assert np.allclose(o.c0,s['c0'],rtol=1e-12,atol=1e-10)
        replay=[];precision=[]
        for j in selected:
            ih=str(pack.rows.iloc[j].input_hash);rc=source.json6(rel+'/labels/receipts/'+ih+'.json')
            label=source.npz6(rel+'/labels/'+rc['payload_path'])
            assert input_hash(d['model_hash'],d['support_hash'],label['demand'])==ih
            assert np.array_equal(label['demand'],pack.a['inputs_native'][j]) and np.array_equal(label['tau'],pack.a['costs_native'][j])
            assert np.allclose(label['normalized_input'],label['demand']/s['input_scale'],rtol=1e-12,atol=1e-12)
            assert np.allclose(label['normalized_gradient'],s['input_scale']*(label['tau']-s['c0'])/float(s['Vscale']),rtol=1e-12,atol=1e-12)
            h=np.zeros(len(o.paths));h[label['path_ids']]=label['weights']
            fresh=o.certify(label['demand'],h,rc['scalars']['iterations'],time.perf_counter(),'PATCH_J0_REPRICE',0,o.lmo_count)
            costs,value,tau=independent_price(a,label['z'])
            assert np.allclose(fresh.z,label['z'],rtol=2e-12,atol=1e-9)
            assert np.allclose(tau,label['tau'],rtol=2e-12,atol=1e-9)
            assert np.isclose(value,rc['scalars']['value_upper'],rtol=2e-12,atol=1e-7)
            assert fresh.quality==rc['scalars']['quality']
            valid=pack.indices('VALID');vu,vt,vL,vU,ve=pack.fields(valid)
            supporting=fresh.supporting_intercept+vu@fresh.tau
            assert np.all(supporting<=vU+ve+fresh.roundoff_allowance+1e-8)
            record=dict(input_hash=ih,role='LOW' if j==lowindex else 'TRAIN',quality=fresh.quality,relative_gap=fresh.relative_gap,
                native_U=value,native_tau_max=float(tau.max()),min_tau_minus_c0=float(np.min(tau-s['c0'])),
                max_u_over_LOW=float(np.max(label['demand']/pack.s['low'])),max_z_over_capacity=float(np.max(label['z']/a['capacity'],initial=0)),
                repriced_max_tau_difference=float(np.max(abs(tau-label['tau']))),reconstructed_max_z_difference=float(np.max(abs(fresh.z-label['z']),initial=0)),
                raw_dual_gap=fresh.raw_gap,roundoff_allowance=fresh.roundoff_allowance,path_conservation_max=o.validate(label['demand'],label['path_ids'],label['weights']),
                gap_aware_VALID_supporting_planes_pass=True,source_payload_sha256=rc['payload_sha256'])
            replay.append(record)
            ex=dest/'examples'/ih;ex.mkdir(parents=True,exist_ok=True)
            used=label['path_ids'];cat=o.catalog();lookup={int(p):i for i,p in enumerate(cat['path_id'])};ci=np.array([lookup[int(p)] for p in used])
            paths=[cat['arcs'][cat['offsets'][i]:cat['offsets'][i+1]] for i in ci]
            save_npz(ex/'used_paths.npz',source_path_ids=used,pair=cat['pair'][ci],source=cat['source'][ci],target=cat['target'][ci],
                offsets=np.r_[0,np.cumsum([len(p) for p in paths])].astype(np.int64),arcs=np.concatenate(paths) if paths else np.empty(0,np.int64))
            save_npz(ex/'native_label.npz',**label);write_json(ex/'source_receipt.json',rc)
            if rid in plan['precision_regions'] and j in {lowindex,int(idx[rank])}:
                cold=Oracle(a);tight=cold.solve(label['demand'],target=1e-7,seconds=min(300,max(.01,86400-5400-(time.time()-c['task_started_unix']))),max_iter=8000,polish_sweeps=3,stop=lambda:not admit(c))
                overlap=max(tight.value_lower,fresh.value_lower)<=min(tight.value_upper,fresh.value_upper)+tight.roundoff_allowance+fresh.roundoff_allowance
                assert overlap,'Same-model cold/source value intervals contradict'
                precision.append(dict(input_hash=ih,target=1e-7,cold_gap=tight.relative_gap,source_gap=fresh.relative_gap,target_reached=tight.relative_gap<=1e-7,
                    cold_seconds=tight.seconds,cold_quality=tight.quality,interval_compatible=overlap,max_tau_change_seconds=float(np.max(abs(tight.tau-fresh.tau))),
                    gradient_error_not_certified_by_value_gap=True))
                save_npz(ex/'tight_result.npz',demand=tight.demand,z=tight.z,tau=tight.tau,path_ids=tight.path_ids,weights=tight.weights,**{'catalog_'+k:v for k,v in cold.catalog().items()})
                write_json(ex/'tight_scalars.json',tight.scalars())
        derived=np.maximum(np.maximum(np.quantile(u,.95,axis=0),pack.s['low']),1e-12)
        eta=weighted_power4_fit(u,t,derived,s['c0']);model=PowerResidualPotential(derived,s['c0'],eta,seed=seed_for(rid,'init',32))
        init=calibrate_residual_unit(model,u,t,U,pack.s['pair_weight'])
        result=dict(status='PASS',region_id=rid,m_active=len(pairs),same_model=True,full_fiber_choices_unchanged=True,Q_LOW_closure_max=float(np.max(abs(q-pack.s['low']))),
            conditional_physical_comparison=hierarchy,old_max_row_cost_loss=float(losses.max()),old_mean_row_cost_loss=float(losses.mean()),
            old_first_layer_saturated_fraction_mean=float(np.mean(saturation)),old_gain=float(pack.s['gain']),old_effective_gain=float(pack.s['gain']*pack.s['Vscale']),
            old_cost_denominator_min=float(pack.s['cost_scale'].min()),old_value_denominator=float(pack.s['value_scale']),
            derived_scale_min=float(derived.min()),derived_scale_max=float(derived.max()),safe_initialization=init,
            replay=replay,precision=precision,source_label_corrections=0,source_extremes_preserved=True,seconds=time.perf_counter()-start)
        save_npz(dest/'local_graph.npz',**a);save_npz(dest/'source_scales.npz',**s)
        source.flush('J0_audit');write_json(dest/'RESULT.json',result);return result
    except Exception as e:
        fail=dict(status='FAIL',region_id=rid,error=repr(e),traceback=traceback.format_exc(),seconds=time.perf_counter()-start)
        source.flush('J0_failure');write_json(dest/'FAILURE.json',fail);raise
    finally:source.close()

def main():
    from .common import load_config
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args();c=load_config(args.config)
    native_threads();plan=panel_plan(c);results=[]
    with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(region_audit,c,r):r for r in plan['audit_regions']}
        for f in as_completed(futures):
            result=f.result();results.append(result);print(result['region_id'],result['status'],round(result['seconds'],2),flush=True)
    report=dict(status='PASS',regions=len(results),samples=sum(len(r['replay']) for r in results),precision_samples=sum(len(r['precision']) for r in results),
        corrections=0,source_model_preserved=True,results=[dict(region_id=r['region_id'],seconds=r['seconds']) for r in results])
    write_json(root(c)/'validation/J0_LABEL_AUDIT_PASS.json',report);print(report,flush=True)
if __name__=='__main__':main()
