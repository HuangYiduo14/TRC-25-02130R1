"""Bounded calls to the unchanged regional oracle with namespaced certificates."""
import time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from .common import root,rd,read_json,write_json,save_npz,sha,admit,input_hash,append_json
from .data import Union,publish_batch,revision
from .sampling import train_plan32,operating_mask
from .metrics import predict_arrays

def preceding_models(c,source,rid,cycle,union):
    from .training import candidates
    p=rd(c,rid)/'training/cycles'/f'cycle_{max(0,cycle-1):04d}'/'manifest.json'
    if p.exists():
        meta=read_json(p);models={}
        for kind in ['FREEFLOW_LINEAR','NEW_POWER4_RELATIVE_NNLS','POWER4_SKIP_ICNN32','POWER4_SKIP_ICNN64_POWER5_HEAD']:
            if kind not in meta['files']:continue
            binding=meta['files'][kind];path=rd(c,rid)/binding['path'];assert sha(path)==binding['sha256']
            with np.load(path,allow_pickle=False) as z:models[kind]={k:z[k] for k in z.files}
        return models,sha(p)
    # Explicit partial J1 comparisons still get a deterministic source-only baseline.
    old_union=Union(c,source,rid,old_only=True);all_candidates=candidates(c,source,rid,old_union)
    return {k:v for k,v in all_candidates.items() if k in ['FREEFLOW_LINEAR','OLD_POWER4_NNLS','NEW_POWER4_RELATIVE_NNLS']},'SOURCE_ONLY_J1_PENDING'

def generate_job(c,source,job):
    from partC_rev1_src.phase6.storage import Store
    if (root(c)/'control/ALL_CHAMPIONS_FROZEN.json').exists():raise PermissionError('Generation forbidden after final champion freeze')
    rid=job['region_id'];role=job['role'];jobid=job['jobid'];dest=rd(c,rid)/'data_extension/jobs'/jobid
    if (dest/'RESULT.json').exists():return read_json(dest/'RESULT.json')
    start=time.perf_counter();d,a,pairs,s=source.local(rid)
    assert d['m_active']>0;union=Union(c,source,rid,old_only=False);roleclass='TRAIN' if role=='TRAIN' else 'VALID' if role=='PATCH_VALID' else 'AUDIT'
    if role=='TRAIN':
        models,binding=preceding_models(c,source,rid,job['cycle'],union)
        utrain=union.fields('TRAIN')[0]
        funcs=[lambda u,k=k,arr=arr:predict_arrays(k,arr,u) for k,arr in models.items()]
        scale=models.get('NEW_POWER4_RELATIVE_NNLS',{}).get('scale',union.s['scale'])
        planpath=train_plan32(c,source,rid,job['cycle'],job['pack_index'],union.ownership_hashes(),funcs,utrain,scale)
        plan=pd.read_parquet(planpath/'plan.parquet')
        with np.load(planpath/'inputs.npz',allow_pickle=False) as z:xs=z['u']
        subset=plan.iloc[job['offset']:job['offset']+4]
    else:
        planpath=rd(c,rid)/('protected_e2' if role.startswith('E2_AUDIT') else 'protected');protected=read_json(planpath/'FROZEN.json')
        assert sha(planpath/'inputs.npz')==protected['input_npz_sha256'] and sha(planpath/'plan.parquet')==protected['plan_sha256']
        plan=pd.read_parquet(planpath/'plan.parquet');subset=plan.loc[plan.split.eq(role)].iloc[job['offset']:job['offset']+4]
        with np.load(planpath/'inputs.npz',allow_pickle=False) as z:xs=z['u']
        binding='FROZEN_LOW_SUPPORT_ONLY'
    if len(subset)==0:raise ValueError('Empty planned generation job')
    namespace=('audit2_labels' if role.startswith('E2_AUDIT') else {'TRAIN':'train_labels','VALID':'valid_labels','AUDIT':'audit_labels'}[roleclass])
    store=Store(rd(c,rid)/'data_extension',d,a,namespace=namespace)
    assert np.allclose(store.oracle.c0,s['c0'],rtol=1e-12,atol=1e-10)
    published=rd(c,rid)/'data_extension/batches'/jobid/'manifest.json'
    if published.exists():
        # A worker can exit after durable batch publication but before linking the
        # revision or writing its job receipt. Reopen and attach exactly that batch;
        # do not append additional labels to an already immutable partial batch.
        manifest=read_json(published);reopen_start=time.perf_counter();recovered=[];attempts=[]
        for binding0 in manifest['receipts']:
            ih=Path(binding0['path']).stem
            result,receipt=store.get(ih,recertify=True);recovered.append((result,receipt))
            attempts.append(dict(input_hash=ih,disposition='RECOVERED_IMMUTABLE_BATCH',quality=result.quality))
        batch=publish_batch(c,rid,store,recovered,role,jobid)
        record=dict(status='COMPLETE' if batch['rows']==len(subset) else 'PARTIAL_PRESERVED',region_id=rid,role=role,jobid=jobid,
            planned=len(subset),published=batch['rows'],AB=batch['AB'],C=batch['C'],quarantined=0,seconds=time.perf_counter()-start,
            attempts=attempts,cycle=job.get('cycle',0),revision=revision(c,rid)['version'],fresh_reopen_seconds=time.perf_counter()-reopen_start,
            recovery='CERTIFIED_BATCH_REATTACHED_WITHOUT_RESOLVING_OR_DUPLICATION',unresolved_planned_inputs=len(subset)-batch['rows'],hidden_truth_exposed_to_scheduler=False)
        write_json(dest/'RESULT.json',record)
        return {k:v for k,v in record.items() if k!='attempts'}
    done=[];attempts=[];warm=None;reopen_seconds=0.
    for row in subset.to_dict('records'):
        if int(row.get('input_index',-1))<0:
            attempts.append(dict(input_hash=None,disposition='NO_UNIQUE_INPUT'));continue
        if not admit(c,2):
            attempts.append(dict(input_hash=row['input_hash'],disposition='DEFERRED_DEADLINE'));break
        u=xs[int(row['input_index'])];ih=input_hash(d['model_hash'],d['support_hash'],u);assert ih==row['input_hash']
        metadata=dict(phase_id='06_incremental_patch',step_id=c['step_id'],run_id=c['run_id'],cycle=job.get('cycle',0),
            split=role,family=row['family'],seed=str(row['seed']),regime=row['regime'],jobid=jobid,input_plan_sha256=sha(planpath/'plan.parquet'),
            preceding_candidates_sha256=binding,source_model_unchanged=True,capacity_already_applied_once=True,
            input_scale_hash=sha(source.path6('regions/'+rid.removeprefix('partC_p2_region_')+'/scales.npz')),
            execution_head=c.get('execution_head'),config_sha256=c.get('effective_config_sha256'),target_gap=1e-5)
        if store.exists(ih):result,receipt=store.get(ih,recertify=True);disposition='RECOVERED_EXISTING_PATCH_RECEIPT'
        else:
            progress_path=rd(c,rid)/('protected_e2/audit_solver_progress.jsonl' if role.startswith('E2_AUDIT') else 'protected/audit_solver_progress.jsonl' if roleclass=='AUDIT' else 'data_extension/solver_progress.jsonl')
            result=store.oracle.solve(u,target=1e-5,max_iter=2000,seconds=min(300,max(.01,86400-5400-(time.time()-c['task_started_unix']))),
                polish_sweeps=3,warm=warm,stop=lambda:not admit(c),
                progress=lambda r:append_json(progress_path,dict(jobid=jobid,input_hash=ih,**r)))
            disposition='NEW_CERTIFIED' if result.quality in {'A','B','C'} else 'QUARANTINED'
            if result.quality in {'A','B','C'}:receipt=store.publish(result,metadata,s)
            else:
                qp=dest/'quarantine'/ih;save_npz(qp.with_suffix('.npz'),demand=result.demand,z=result.z,tau=result.tau,path_ids=result.path_ids,weights=result.weights,**{'catalog_'+k:v for k,v in store.oracle.catalog().items()})
                write_json(qp.with_suffix('.json'),dict(metadata=metadata,scalars=result.scalars()))
        if result.quality in {'A','B','C'}:
            check_start=time.perf_counter();fresh,_=store.get(ih,recertify=True);reopen_seconds+=time.perf_counter()-check_start
            done.append((fresh,receipt));warm=result
        attempts.append(dict(input_hash=ih,disposition=disposition,quality=result.quality,seconds=result.seconds,iterations=result.iterations,
            relative_gap=result.relative_gap if roleclass!='AUDIT' else 'HIDDEN_UNTIL_FINAL_FREEZE'))
    batch=publish_batch(c,rid,store,done,role,jobid)
    record=dict(status='COMPLETE' if len(attempts)==len(subset) and all(x['disposition']!='DEFERRED_DEADLINE' for x in attempts) else 'PARTIAL',
        region_id=rid,role=role,jobid=jobid,planned=len(subset),published=len(done),AB=batch['AB'],C=batch['C'],
        quarantined=sum(r.get('quality')=='QUARANTINED' for r in attempts),seconds=time.perf_counter()-start,attempts=attempts,
        cycle=job.get('cycle',0),revision=revision(c,rid)['version'],fresh_reopen_seconds=reopen_seconds,hidden_truth_exposed_to_scheduler=False)
    record.update(new_certified=sum(r['disposition']=='NEW_CERTIFIED' for r in attempts),
                  recovered_existing=sum(r['disposition']=='RECOVERED_EXISTING_PATCH_RECEIPT' for r in attempts))
    write_json(dest/'RESULT.json',record)
    # Scheduler receives coverage only; audit values and errors remain segregated.
    return {k:v for k,v in record.items() if k!='attempts'}
