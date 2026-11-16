"""Native errors and independent portable scalar-potential validation."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import torch
from .model import frozen_eval,export_arrays,separable_power4_eval
from .common import save_npz,write_json,sha

def metric(v,t,u,target,L,U,eps,s):
    err=t-target;ae=abs(err);e=ae/np.maximum(abs(target),1.);low=s['low']/s['low'].sum()
    gap=np.maximum(L-eps-v,0)+np.maximum(v-U-eps,0)
    res=err/s['cost_scale'];congestion=target-s['c0']
    below=np.maximum(s['c0']-t,0)
    cost_loss=float(np.mean((res*res)@s['pair_weight']))
    value_loss=float(np.mean((gap/float(s['value_scale']))**2))
    floor_loss=float(np.mean((below/s['cost_scale'])**2@s['pair_weight']))
    result=dict(n_rows=len(u),n_coordinates=t.size,MAE_seconds=float(ae.mean()),RMSE_seconds=float(np.sqrt((err*err).mean())),
        bias_seconds=float(err.mean()),MARE_equal_pairs=float(e.mean()),MARE_LOW_pair_weighted=float((e@low).mean()),
        p50_relative=float(np.quantile(e,.5)),p90_relative=float(np.quantile(e,.9)),p95_relative=float(np.quantile(e,.95)),max_relative=float(e.max()),max_seconds=float(ae.max()),
        mean_interval_violation_native=float(gap.mean()),max_interval_violation_native=float(gap.max()),
        mean_interval_violation_relative=float(np.mean(gap/np.maximum(abs(U),1.))),
        residual_cost_RMSE_scaled=float(np.sqrt((res*res).mean())),congestion_truth_RMS_seconds=float(np.sqrt((congestion*congestion).mean())),
        congestion_relative_MAE=float((ae/np.maximum(abs(congestion),s['cost_scale'])).mean()),
        residual_value_interval_RMSE_scaled=float(np.sqrt(value_loss)),floor_violation_count=int((t<s['c0']-1e-8).sum()),
        negative_cost_count=int((t < -1e-8).sum()),minimum_cost_seconds=float(t.min()),
        zero_demand_coordinate_count=int((u==0).sum()),zero_demand_MAE_seconds=float(ae[u==0].mean()) if np.any(u==0) else None,
        observed_train_box_outside_rows=int(np.any((u<s['train_min'])|(u>s['train_max']),axis=1).sum()),
        interval_width_mean=float(np.mean(U-L)),cost_loss=cost_loss,value_interval_loss=value_loss,floor_loss=floor_loss,
        composite_loss=cost_loss+.1*value_loss+.1*floor_loss)
    result['accuracy_screen']=bool(result['MARE_equal_pairs']<=.05 and result['MARE_LOW_pair_weighted']<=.05 and result['mean_interval_violation_relative']<=.01 and result['negative_cost_count']==0)
    if not np.isfinite(np.r_[v,t.ravel()]).all():raise FloatingPointError('Nonfinite prediction')
    return result

def predict(model,u,batch=256):
    vv=[];tt=[]
    for start in range(0,len(u),batch):
        v,t=model.value_cost(torch.as_tensor(np.asarray(u[start:start+batch]),dtype=torch.float64))
        vv.append(v.numpy());tt.append(t.numpy())
    return np.concatenate(vv),np.concatenate(tt)

def validation_metrics(model,pack,split='VALID'):
    idx=pack.indices(split);u,t,L,U,eps=pack.fields(idx)
    v,p=predict(model,u,64 if pack.meta['m_active']>=1024 else 256)
    return metric(v,p,u,t,L,U,eps,pack.s)

def detailed_math(model,pack):
    a=export_arrays(model);s=pack.s;rng=np.random.default_rng(8675309);m=len(s['scale'])
    train=pack.indices('TRAIN');native=np.asarray(pack.a['inputs_native'][train[:min(3,len(train))]])
    # Normalized interior directions control finite-difference scale without high-m Hessians.
    u=np.vstack([np.zeros(m),s['scale']*.25,native]);d=rng.normal(size=u.shape)*s['scale']/np.sqrt(m)
    tv,tc=predict(model,u);v,t,h=frozen_eval(a,u,d)
    potential_atol=max(1e-8,64*np.finfo(float).eps*float(a['Vscale']*a['gain'])*max(1.,abs(float(a['anchor_value']))))
    assert np.allclose(tv,v,rtol=1e-9,atol=potential_atol),'Portable potential differs'
    assert np.allclose(tc,t,rtol=1e-9,atol=1e-8),'Portable native cost differs'
    assert abs(v[0])<=potential_atol and np.allclose(t[0],s['c0'],rtol=0,atol=1e-8),'Zero anchor'
    uu=torch.tensor(u,requires_grad=True,dtype=torch.float64)
    value=model(uu);g=torch.autograd.grad(value.sum(),uu,create_graph=True)[0]
    th=torch.autograd.grad((g*torch.tensor(d)).sum(),uu)[0].detach().numpy()
    assert np.allclose(th,h,rtol=1e-9,atol=1e-8),'Portable HVP differs'
    assert np.all(a['W2']>=0) and np.all(a['v']>=0) and model.net.parameter_count==64*m+1120
    # Dimensionless residual Jensen, monotonicity and HVP symmetry tests.
    x1=rng.uniform(.1,.7,(3,m));x2=rng.uniform(.1,.7,(3,m));xmid=.3*x1+.7*x2
    def residual(x):
        with torch.no_grad():return model.net(torch.tensor(x,dtype=torch.float64),differentiable_anchor=False).numpy()
    jensen=float(np.max(residual(xmid)-.3*residual(x1)-.7*residual(x2)))
    assert jensen<=1e-10
    _,t1=frozen_eval(a,x1*s['scale']);_,t2=frozen_eval(a,x2*s['scale'])
    mono=np.sum((t1-t2)*(x1-x2)*s['scale'],axis=1)/float(a['Vscale']*a['gain'])
    assert mono.min()>=-1e-9
    dd=rng.normal(size=(1,m))/np.sqrt(m);ee=rng.normal(size=(1,m))/np.sqrt(m);probe=.3*s['scale'][None,:]
    _,_,hd=frozen_eval(a,probe,dd*s['scale']);_,_,he=frozen_eval(a,probe,ee*s['scale'])
    sym=abs(float(np.sum(ee*s['scale']*hd)-np.sum(dd*s['scale']*he)))/float(a['Vscale']*a['gain'])
    assert sym<=1e-9
    # Finite differences of normalized residual potential, including one-sided origin.
    base=np.full((1,m),.3);z=np.zeros((1,m));fd=[]
    _,bp=frozen_eval(a,base*s['scale']);exact=float(np.sum((bp-s['c0'])*s['scale']*dd))/float(a['Vscale']*a['gain'])
    for step in [1e-4,1e-5,1e-6]:
        central=float((residual(base+step*dd)-residual(base-step*dd))[0]/(2*step))
        fd.append(dict(step=step,estimate=central,error=abs(central-exact),passed=bool(np.isclose(central,exact,rtol=1e-6,atol=1e-8))))
    assert any(r['passed'] for r in fd),'Normalized directional finite differences failed'
    boundary=float(residual(z+1e-6*abs(dd))[0]/1e-6)
    assert boundary>=-1e-8
    return dict(status='PASS',m_active=m,parameters=model.net.parameter_count,max_native_cost_difference=float(np.max(abs(tc-t))),
        max_potential_difference=float(np.max(abs(tv-v))),potential_absolute_roundoff_allowance=potential_atol,max_hvp_difference=float(np.max(abs(th-h))),
        jensen_max_excess=jensen,monotonicity_min=float(mono.min()),hvp_symmetry_scaled=sym,normalized_finite_differences=fd,boundary_residual_slope=boundary,
        no_full_hessian=True,no_oracle=True),a

def evaluate_frozen(model,pack,directory,state):
    start=time.perf_counter();directory.mkdir(parents=True,exist_ok=True)
    assert state['steps']>0
    check,a=detailed_math(model,pack)
    save_npz(directory/'portable.npz',**a)
    rows=[];bench=[]
    for split in ['TRAIN','VALID','LOW_TEST','HIGH_PROJECTED']:
        idx=pack.indices(split)
        if not len(idx):continue
        u,target,L,U,eps=pack.fields(idx)
        t0=time.perf_counter();v,p=predict(model,u,64 if pack.meta['m_active']>=1024 else 256);torch_seconds=time.perf_counter()-t0
        pv=[];pp=[];t0=time.perf_counter()
        for j in range(0,len(u),128):
            vv,tt=frozen_eval(a,u[j:j+128]);pv.append(vv);pp.append(tt)
        portable_seconds=time.perf_counter()-t0;pv=np.concatenate(pv);pp=np.concatenate(pp)
        assert np.allclose(pv,v,rtol=1e-9,atol=check['potential_absolute_roundoff_allowance'])
        assert np.allclose(pp,p,rtol=1e-9,atol=1e-8)
        s=pack.s;bv,bt=separable_power4_eval(u,s['scale'],s['c0'],s['baseline_eta'])
        pred={'ICNN':(v,p),'FREEFLOW_LINEAR':(u@s['c0'],np.broadcast_to(s['c0'],target.shape)),'SEPARABLE_POWER4_NNLS':(bv,bt)}
        rr=pack.rows.iloc[idx].reset_index(drop=True)
        for name,(vv,tt) in pred.items():
            groups=[('ALL','ALL',np.arange(len(idx)))]+[(str(f),'ALL',np.flatnonzero(rr.family.eq(f))) for f in sorted(rr.family.unique())]+[('ALL',str(g),np.flatnonzero(rr.quality.eq(g))) for g in sorted(rr.quality.unique())]
            for family,grade,k in groups:
                rows.append(dict(region_id=pack.meta['region_id'],m_active=pack.meta['m_active'],LOW_exposure=float(s['low'].sum()),split=split,family=family,grade=grade,model=name,
                    **metric(vv[k],tt[k],u[k],target[k],L[k],U[k],eps[k],s)))
        # Full native predictions stay in external files. Identity rows are bound alongside them.
        np.save(directory/(split+'_potential.npy'),v);np.save(directory/(split+'_cost.npy'),p)
        np.save(directory/(split+'_pack_indices.npy'),idx)
        bench.append(dict(split=split,rows=len(idx),coordinates=int(p.size),torch_seconds=torch_seconds,portable_seconds=portable_seconds))
    pd.DataFrame(rows).to_parquet(directory/'metrics.parquet',index=False)
    pd.DataFrame(rows).to_csv(directory/'metrics.csv',index=False)
    # Batch-one value+gradient+HVP runtime on LOW anchor, with fixed thread mode.
    u=pack.s['low'][None,:];d=pack.s['scale'][None,:]/np.sqrt(len(pack.s['scale']))
    times=[]
    for _ in range(5):
        t0=time.perf_counter();frozen_eval(a,u,d);times.append(time.perf_counter()-t0)
    check['batch1_portable_value_gradient_hvp_seconds']=float(np.median(times))
    write_json(directory/'mathematical_validation.json',check)
    write_json(directory/'benchmarks.json',dict(device='cpu',dtype='float64',threads=1,batches=bench,batch1_HVP_seconds=float(np.median(times))))
    manifest=dict(status='PASS',region_id=pack.meta['region_id'],checkpoint_frozen_before_test=True,steps=state['steps'],best_epoch=state['best_epoch'],best_steps=state['best_steps'],
        source_pack_manifest_sha256=sha(pack.path/'manifest.json'),files={p.name:dict(sha256=sha(p),size_bytes=p.stat().st_size) for p in directory.iterdir() if p.is_file()},
        evaluation_seconds=time.perf_counter()-start)
    write_json(directory/'manifest.json',manifest)
    return manifest
