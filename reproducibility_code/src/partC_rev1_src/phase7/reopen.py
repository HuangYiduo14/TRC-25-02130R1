"""Fresh-process reopening of frozen models, prediction packs and metric evidence."""
from __future__ import annotations
import argparse,concurrent.futures,json,os,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import root,region_out,read_json,write_json,append_json,sha,load_config,elapsed
from .data import Pack
from .runtime import FrozenRegion
from .portable import frozen_eval
from .evaluation import metric,detailed_math
from .training import torch_load,make_model

def verify_one(c,rid,detailed=False):
    started=time.perf_counter();rd=region_out(c,rid)
    try:
        em=read_json(rd/'evaluation/manifest.json')
        for name,r in em['files'].items():
            if sha(rd/'evaluation'/name)!=r['sha256']:raise ValueError('Frozen evaluation file changed: '+name)
        if sha(rd/'pack/manifest.json')!=em['source_pack_manifest_sha256']:raise ValueError('Pack changed after evaluation')
        p=Pack(rd/'pack',verify=True);runtime=FrozenRegion(rd)
        frozen=read_json(rd/'CHECKPOINT_FROZEN.json')
        assert sha(rd/'best.pt')==frozen['best_sha256']
        best=torch_load(rd/'best.pt');last=torch_load(rd/'last.pt')
        assert best['pack_manifest_sha256']==last['pack_manifest_sha256']==sha(rd/'pack/manifest.json')
        assert em['steps']==last['steps'] and em['best_steps']==best['steps'] and best['steps']>0
        model=make_model(p);model.load_state_dict(best['model'])
        t0=time.perf_counter();query=runtime.query(p.s['low'][None,:]);query_seconds=time.perf_counter()-t0
        assert np.isfinite(query['potential']).all() and np.isfinite(query['cost']).all()
        zero=runtime.query(np.zeros((1,p.meta['m_active'])))
        assert np.allclose(zero['cost'],p.s['c0'],rtol=0,atol=1e-8)
        metrics=pd.read_parquet(rd/'evaluation/metrics.parquet');csv=pd.read_csv(rd/'evaluation/metrics.csv')
        assert len(metrics)==len(csv)>0
        assert not metrics.duplicated(['split','family','grade','model']).any()
        rows_checked=0;coordinates_checked=0
        for split in ['TRAIN','VALID','LOW_TEST','HIGH_PROJECTED']:
            idx=p.indices(split)
            if not len(idx):continue
            saved_idx=np.load(rd/'evaluation'/(split+'_pack_indices.npy'),allow_pickle=False)
            assert np.array_equal(idx,saved_idx)
            values=np.load(rd/'evaluation'/(split+'_potential.npy'),mmap_mode='r',allow_pickle=False)
            costs=np.load(rd/'evaluation'/(split+'_cost.npy'),mmap_mode='r',allow_pickle=False)
            assert costs.shape==(len(idx),p.meta['m_active']) and values.shape==(len(idx),)
            # Every prediction row is checked for finite native values. Deterministic
            # endpoints and middle are independently queried and metrically replayed.
            for start in range(0,len(idx),256):
                assert np.isfinite(values[start:start+256]).all() and np.isfinite(costs[start:start+256]).all()
            select=np.unique(np.array([0,len(idx)//2,len(idx)-1]));u,t,L,U,eps=p.fields(idx[select]);v,co=frozen_eval(runtime.a,u)
            assert np.allclose(co,costs[select],rtol=1e-9,atol=1e-8)
            assert np.allclose(v,values[select],rtol=1e-9,atol=1e-8)
            u,t,L,U,eps=p.fields(idx)
            replay=metric(values,costs,u,t,L,U,eps,p.s)
            recorded=metrics.loc[metrics.model.eq('ICNN')&metrics.split.eq(split)&metrics.family.eq('ALL')&metrics.grade.eq('ALL')].iloc[0]
            for k in ['MARE_equal_pairs','MARE_LOW_pair_weighted','RMSE_seconds','mean_interval_violation_relative','composite_loss']:
                assert np.isclose(replay[k],recorded[k],rtol=1e-12,atol=1e-12),k
            rows_checked+=len(idx);coordinates_checked+=costs.size
        detail=None
        if detailed:
            detail,_=detailed_math(model,p)
            timings=[]
            for _ in range(5):
                q=torch.tensor(p.s['low'][None,:],dtype=torch.float64,requires_grad=True)
                d=torch.tensor(p.s['scale'][None,:]/np.sqrt(p.meta['m_active']),dtype=torch.float64)
                t0=time.perf_counter();vv=model(q);gg=torch.autograd.grad(vv.sum(),q,create_graph=True)[0]
                torch.autograd.grad((gg*d).sum(),q)
                timings.append(time.perf_counter()-t0)
            detail['batch1_Torch_value_gradient_HVP_seconds']=float(np.median(timings))
        # Model card binds safe exports, source support, recipe, checkpoint and limitations.
        valid=metrics.loc[metrics.model.eq('ICNN')&metrics.split.eq('VALID')&metrics.family.eq('ALL')&metrics.grade.eq('ALL')].iloc[0]
        all_main=metrics.loc[metrics.model.eq('ICNN')&metrics.family.eq('ALL')&metrics.grade.eq('ALL')]
        card=dict(region_id=rid,architecture=c['model_and_training']['architecture'],m_active=p.meta['m_active'],parameter_count=64*p.meta['m_active']+1120,
            seed=p.meta['seed'],training_recipe=c['model_and_training'],native_dtype='float64',source_descriptor=p.meta['source_descriptor'],source_pair_order_hash=p.meta['source_pair_order_hash'],
            source_pack_manifest_sha256=sha(rd/'pack/manifest.json'),train_input_hash_binding=p.meta['train_input_hash_binding'],
            trained_steps=last['steps'],best_steps=best['steps'],best_epoch=best['epoch'],complete_epochs=last['epoch'],rows_seen=last['rows_seen'],unique_rows_seen=int(last['unique_seen'].sum()),
            train_seconds=last['train_seconds'],best_checkpoint_sha256=sha(rd/'best.pt'),last_checkpoint_sha256=sha(rd/'last.pt'),portable_sha256=sha(rd/'evaluation/portable.npz'),
            numerical_status='PASS',VALID_accuracy_screen=bool(valid.accuracy_screen),any_negative_total_costs=bool(all_main.negative_cost_count.gt(0).any()),
            any_freeflow_floor_violations=bool(all_main.floor_violation_count.gt(0).any()),deployment_authorized=False,
            unavailable_for_deployment_reason='National deployment is not authorized; negative total costs additionally disqualify any affected regional model. No costs were clipped.',
            domain='Finite nonnegative exact active-coordinate inputs; out-of-observed-TRAIN box is explicitly flagged. No clamping.',next_phase_authorized=False)
        write_json(rd/'model.json',card)
        return dict(region_id=rid,status='PASS',m_active=p.meta['m_active'],rows_checked=rows_checked,coordinates_checked=coordinates_checked,
            detailed_math=detail,portable_LOW_query_seconds=query_seconds,reopen_seconds=time.perf_counter()-started)
    except Exception as e:return dict(region_id=rid,status='FAIL',error=str(e),traceback=traceback.format_exc(),reopen_seconds=time.perf_counter()-started)

def init_worker(out):
    import faulthandler
    global STACK_FILE
    STACK_FILE=open(Path(out)/'logs'/f'reopen_worker_{os.getpid()}_stacks.log','a',encoding='utf-8')
    faulthandler.enable(STACK_FILE);faulthandler.dump_traceback_later(300,repeat=True,file=STACK_FILE)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)

def _run(c):
    assert (root(c)/'CAMPAIGN_EXECUTION_DONE.json').exists()
    manifest=pd.read_parquet(root(c)/'reports/model_manifest.parquet').sort_values('m_active')
    detailed=set(manifest.iloc[np.linspace(0,len(manifest)-1,min(12,len(manifest)),dtype=int)].region_id)
    for scale in ['Vscale','TRAIN_gain','effective_residual_scale','input_scale_min','input_scale_max']:
        s=manifest.sort_values(scale);detailed.update(s.iloc[[0,len(s)//2,len(s)-1]].region_id)
    results=[];started=time.perf_counter();todo=list(manifest.region_id);pending={};retries={}
    pool=concurrent.futures.ProcessPoolExecutor(max_workers=2,initializer=init_worker,initargs=(str(root(c)),))
    try:
        while todo or pending:
            while todo and len(pending)<2:
                rid=todo.pop(0);pending[pool.submit(verify_one,c,rid,rid in detailed)]=rid
            done,_=concurrent.futures.wait(pending,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                if future not in pending:continue
                rid=pending.pop(future)
                try:r=future.result()
                except concurrent.futures.process.BrokenProcessPool:
                    interrupted=[rid]+list(pending.values());pending.clear()
                    append_json(root(c)/'validation/fresh_reopen_native_recovery.jsonl',dict(unix=time.time(),regions=interrupted,traceback=traceback.format_exc()))
                    pool.shutdown(wait=True,cancel_futures=True)
                    pool=concurrent.futures.ProcessPoolExecutor(max_workers=2,initializer=init_worker,initargs=(str(root(c)),))
                    for rr in interrupted:
                        retries[rr]=retries.get(rr,0)+1
                        if retries[rr]<=1:todo.insert(0,rr)
                        else:
                            failure=dict(region_id=rr,status='FAIL',error='Repeated native process interruption during fresh read-only model verification')
                            results.append(failure);write_json(root(c)/'validation/reopen_regions'/(rr+'.json'),failure)
                    continue
                results.append(r);write_json(root(c)/'validation/reopen_regions'/(rid+'.json'),r)
                if len(results)%100==0:print(json.dumps(dict(reopened=len(results),failed=sum(r['status']!='PASS' for r in results),elapsed=time.perf_counter()-started)),flush=True)
    finally:pool.shutdown(wait=True,cancel_futures=True)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['detailed_math','traceback']} for r in results]).to_csv(root(c)/'validation/FRESH_REOPEN_REGIONS.csv',index=False)
    from .diagnostics import reopen as reopen_diagnostics
    diagnostic_reopens=reopen_diagnostics(c)
    result=dict(status='PASS' if all(r['status']=='PASS' for r in results) else 'FAIL',produced_model_count=len(manifest),reopened_models=len(results),failed_regions=[r for r in results if r['status']!='PASS'],
        covered_active_coordinates=int(manifest.m_active.sum()),native_prediction_coordinates=sum(r.get('coordinates_checked',0) for r in results),
        detailed_derivative_regions=sorted(detailed),total_portable_LOW_query_seconds=sum(r.get('portable_LOW_query_seconds',0) for r in results),
        diagnostic_reopened_models=len(diagnostic_reopens),diagnostic_reopens=diagnostic_reopens,
        fresh_process=True,no_source_oracle=True,seconds=time.perf_counter()-started,whole_campaign_elapsed_seconds=elapsed(c))
    write_json(root(c)/'validation/FRESH_REOPEN_RESULT.json',result);print(json.dumps(result))
    return result


def run(c):
    from .campaign import Watchdog
    watchdog=Watchdog(c);watchdog.start()
    try:return _run(c)
    finally:watchdog.stop.set();watchdog.thread.join(timeout=5);watchdog.sample()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);args=a.parse_args();run(load_config(args.config))
