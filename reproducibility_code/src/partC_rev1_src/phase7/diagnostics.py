"""Bounded nested TRAIN-subset probes; never select or alter the main cohort."""
from __future__ import annotations
import argparse,copy,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import load_config,root,region_out,read_json,write_json,sha,elapsed
from .data import Pack
from .model import train_scales,objective,export_arrays,frozen_eval
from .evaluation import validation_metrics,detailed_math,metric
from .training import make_model,new_state,check_valid,terminal_rule,torch_save,torch_load


def nested_indices(pack,fractions=(.25,.5)):
    """One seeded ordering; preserve canonical ZERO and LOW anchors in both sizes."""
    train=pack.indices('TRAIN');u=pack.a['inputs_native'];low=pack.s['low']
    mandatory=[int(i) for i in train if np.all(u[i]==0) or np.array_equal(u[i],low)]
    assert 1<=len(mandatory)<=2,'Expected canonical TRAIN ZERO/LOW anchors'
    rest=np.setdiff1d(train,mandatory)
    rng=np.random.default_rng(pack.meta['seed']);rest=rng.permutation(rest)
    order=np.r_[mandatory,rest].astype(int)
    return {f:order[:min(len(train),max(16,int(np.ceil(f*len(train)))))] for f in fractions}


class SubsetPack:
    def __init__(self,p,ids):
        self.path=p.path;self.meta=copy.deepcopy(p.meta);self.rows=p.rows;self.a=p.a;self.s={k:v.copy() for k,v in p.s.items()};self.ids=np.asarray(ids,int)
        u,t,L,U,eps=p.fields(self.ids)
        self.s.update(train_scales(u,t,U,self.s['scale'],self.s['c0'],float(self.s['Vscale']),self.s['low']))
        self.s.update(train_min=u.min(axis=0),train_max=u.max(axis=0))
    def indices(self,split):
        assert split in ['TRAIN','VALID'],'Diagnostics cannot read heldout TEST/HIGH'
        return self.ids if split=='TRAIN' else np.flatnonzero(self.rows.split.eq('VALID')&self.rows.quality.isin(['A','B']))
    def fields(self,ids):return tuple(np.asarray(self.a[n][ids]) for n in ['inputs_native','costs_native','lower_native','upper_native','allowance_native'])


def fit(c,p,ids,fraction,dest,whole_stop):
    assert not dest.exists();dest.mkdir(parents=True)
    start=time.perf_counter();stop=min(whole_stop,start+c['model_and_training']['optional_data_curve_seconds_per_fit'])
    sub=SubsetPack(p,ids);recipe=copy.deepcopy(c['model_and_training']);recipe['epochs_max']=recipe['optional_data_curve_epochs_max']
    model=make_model(sub);state=new_state(model,sub,recipe)
    opt=torch.optim.Adam(model.parameters(),lr=recipe['learning_rate'],betas=tuple(recipe['betas']),eps=recipe['eps'])
    gen=torch.Generator().manual_seed(sub.meta['seed']);n=len(ids);batch=64 if p.meta['m_active']>=1024 else 128
    ds=torch.tensor(sub.s['cost_scale']);w=torch.tensor(sub.s['pair_weight']);vs=float(sub.s['value_scale']);last_eval=.1
    while state['status']=='TRAINING':
        if time.perf_counter()+max(.1,2*state['last_step_seconds'])+max(.1,1.5*last_eval)>=stop:
            state['status']='DIAGNOSTIC_WALL_TIME_CAP';break
        if not state['order'].numel():state['order']=torch.randperm(n,generator=gen);state['cursor']=0
        end=min(n,state['cursor']+batch);positions=state['order'][state['cursor']:end].numpy();u,t,L,U,eps=sub.fields(ids[positions]);t0=time.perf_counter()
        tensors=[torch.tensor(a,dtype=torch.float64) for a in [u,t,L-eps,U+eps]]
        opt.zero_grad(set_to_none=True);loss,_=objective(model,*tensors,ds,w,vs)
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step()
        assert all(torch.isfinite(v).all() for v in model.parameters())
        dt=time.perf_counter()-t0;state['last_step_seconds']=dt;state['train_seconds']+=dt
        state['steps']+=1;state['rows_seen']+=len(positions);state['unique_seen'][positions]=True;state['cursor']=end
        if end==n:
            state['epoch']+=1;state['cursor']=0;state['order']=torch.empty(0,dtype=torch.int64)
            if state['epoch']==1 or state['epoch']%10==0:
                t0=time.perf_counter();check_valid(model,sub,opt,state,dest,True);last_eval=time.perf_counter()-t0
            state['status']=terminal_rule(state,recipe)
    if not state['steps']:raise RuntimeError('Diagnostic no-op')
    if state['cursor'] or state['last_scheduled_epoch']!=state['epoch']:check_valid(model,sub,opt,state,dest,False)
    state.update(model=model.state_dict(),optimizer=opt.state_dict(),rng_state=gen.get_state());torch_save(dest/'last.pt',state)
    best=torch_load(dest/'best.pt');model.load_state_dict(best['model']);check,a=detailed_math(model,sub)
    np.savez_compressed(dest/'portable.npz',**a);np.save(dest/'TRAIN_pack_indices.npy',ids);np.savez(dest/'subset_scales.npz',**sub.s)
    train=validation_metrics(model,sub,'TRAIN');valid=validation_metrics(model,sub,'VALID')
    idx=sub.indices('VALID');u,t,L,U,eps=sub.fields(idx);v,co=frozen_eval(a,u)
    np.savez_compressed(dest/'VALID_predictions.npz',indices=idx,potential=v,cost=co)
    result=dict(region_id=p.meta['region_id'],fraction=fraction,TRAIN_rows=n,full_TRAIN_rows=len(p.indices('TRAIN')),VALID_rows=len(idx),m_active=p.meta['m_active'],
        seed=p.meta['seed'],status=state['status'],steps=state['steps'],epochs=state['epoch'],best_steps=best['steps'],best_epoch=best['epoch'],
        rows_seen=state['rows_seen'],unique_rows_seen=int(state['unique_seen'].sum()),train_seconds=state['train_seconds'],wall_seconds=time.perf_counter()-start,
        train=train,valid=valid,mathematical_validation=check,source_pack_manifest_sha256=sha(p.path/'manifest.json'),
        subset_input_hashes=p.rows.iloc[ids].input_hash.tolist(),numerical_scalings='Recomputed on each TRAIN subset; source input_scale/c0/Vscale unchanged',
        main_model_altered=False,TEST_HIGH_accessed=False,comparison_note='Same architecture and regional initialization seed; data sizes, optimizer updates, convergence and TRAIN-derived gain/scales differ. This is supporting evidence, not causal proof.')
    write_json(dest/'result.json',result)
    write_json(dest/'manifest.json',dict(status='PASS',files={x.name:dict(sha256=sha(x),size_bytes=x.stat().st_size) for x in dest.iterdir() if x.is_file()}))
    return result


def _run(c,plan_path):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=root(c);assert (out/'CAMPAIGN_EXECUTION_DONE.json').exists(),'Main fitting has priority'
    plan=read_json(plan_path);assert plan['config_sha256']==sha(c['_config_path']) and plan['task_id']=='T3_OPTIONAL_DATA_SIZE'
    assert plan['phase_id']==c['phase_id'] and plan['step_id']==c['step_id'] and plan['run_id']==c['execution']['run_id']
    assert elapsed(c)+960+2700<28800
    for file,h in plan['code_hashes'].items():assert sha(Path(c['execution']['repository_root'])/file)==h
    prior=read_json(out/'validation/optional_data_size_diagnostics.json')
    assert plan['regions']==prior['preselected_regions'] and len(plan['regions'])<=8
    base=out/'diagnostics/nested_train_subsets';assert not base.exists();base.mkdir(parents=True)
    started=time.perf_counter();stop=started+960;results=[]
    for rid in plan['regions']:
        rd=region_out(c,rid);before={n:sha(rd/n) for n in ['best.pt','last.pt','evaluation/manifest.json']}
        p=Pack(rd/'pack',verify=True)
        for fraction,ids in nested_indices(p).items():
            if time.perf_counter()+65>=stop:break
            dest=base/rid/str(int(fraction*100));dest.parent.mkdir(exist_ok=True)
            try:results.append(fit(c,p,ids,fraction,dest,stop))
            except Exception as e:
                r=dict(region_id=rid,fraction=fraction,status='DIAGNOSTIC_FAILURE',error=str(e),traceback=traceback.format_exc());write_json(dest/'failure.json',r);results.append(r)
            print(json.dumps({k:v for k,v in results[-1].items() if k in ['region_id','fraction','status','epochs','steps','wall_seconds']}),flush=True)
        assert all(sha(rd/n)==h for n,h in before.items()),'Main checkpoint changed during diagnostic'
    final=dict(status='COMPLETED' if len(results)==16 and all('FAILURE' not in r['status'] for r in results) else 'COMPLETED_WITH_DIAGNOSTIC_EXCEPTIONS',
        regions=plan['regions'],fits=len(results),wall_seconds=time.perf_counter()-started,hard_seconds=960,results=results,
        main_checkpoints_unchanged=True,TEST_HIGH_accessed=False,whole_campaign_elapsed_seconds=elapsed(c))
    write_json(base/'DIAGNOSTIC_RESULT.json',final);return final


def run(c,plan_path):
    from .campaign import Watchdog
    watchdog=Watchdog(c);watchdog.start()
    try:return _run(c,plan_path)
    finally:watchdog.stop.set();watchdog.thread.join(timeout=5);watchdog.sample()


def reopen(c):
    base=root(c)/'diagnostics/nested_train_subsets';records=[]
    if not base.exists():return []
    for manifest in base.glob('*/*/manifest.json'):
        d=manifest.parent;r=read_json(d/'result.json');m=read_json(manifest)
        for n,h in m['files'].items():assert sha(d/n)==h['sha256']
        p=Pack(region_out(c,r['region_id'])/'pack',verify=False);ids=np.load(d/'TRAIN_pack_indices.npy',allow_pickle=False);sub=SubsetPack(p,ids)
        with np.load(d/'portable.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        with np.load(d/'VALID_predictions.npz',allow_pickle=False) as z:values={k:z[k] for k in z.files}
        u,t,L,U,eps=sub.fields(sub.indices('VALID'));v,co=frozen_eval(a,u)
        assert np.allclose(co,values['cost'],rtol=1e-9,atol=1e-8) and np.allclose(v,values['potential'],rtol=1e-9,atol=1e-8)
        actual=metric(v,co,u,t,L,U,eps,sub.s)
        assert np.isclose(actual['MARE_equal_pairs'],r['valid']['MARE_equal_pairs'],rtol=1e-12,atol=1e-12)
        model=make_model(sub);model.load_state_dict(torch_load(d/'best.pt')['model']);torch_load(d/'last.pt');check,_=detailed_math(model,sub)
        records.append(dict(region_id=r['region_id'],fraction=r['fraction'],status='PASS',coordinates=int(co.size),mathematical_validation=check))
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--plan',required=True);a=p.parse_args();c=load_config(a.config);c['_config_path']=str(Path(a.config).resolve());run(c,a.plan)
