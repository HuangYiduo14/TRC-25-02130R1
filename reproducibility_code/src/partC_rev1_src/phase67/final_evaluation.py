"""Final-only heldout metrics and fresh physical/model reopening."""
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .common import root,rd,read_json,write_json,save_npz,sha,append_json
from .data import Union
from .metrics import predict_arrays,score_metric
from .sampling import operating_mask

def portable_math(kind,a,low):
    u=np.stack([np.zeros_like(low),.3*low,low]);direction=np.ones_like(u)*np.maximum(low,1e-12)/np.sqrt(len(low))
    v,t,h=predict_arrays(kind,a,u,direction)
    assert np.isfinite(v).all() and np.isfinite(t).all() and np.isfinite(h).all()
    assert abs(v[0])<=1e-7 and np.allclose(t[0],a['c0'],rtol=0,atol=1e-8)
    q=torch.tensor(u,dtype=torch.float64,requires_grad=True);c0=torch.tensor(a['c0']);base=q@c0
    if kind!='FREEFLOW_LINEAR':
        scale=torch.tensor(a['scale']);x=q/scale
        if 'ICNN' not in kind:
            value=base+(torch.tensor(a['eta'])*scale*x**5/5).sum(-1)
        else:
            arrays={k:torch.tensor(a[k],dtype=torch.float64) for k in ['A1','A2','b1','b2','W2','v','anchor_value','anchor_gradient']}
            assert (a['W2']>=0).all() and (a['v']>=0).all()
            l1=x@arrays['A1'].T+arrays['b1'];h1=F.softplus(l1,threshold=40.)
            l2=h1@arrays['W2'].T+x@arrays['A2'].T+arrays['b2'];h2=F.softplus(l2,threshold=40.)
            power=int(a.get('head_power',1));raw=(h2**power/power)@arrays['v']
            residual=raw-arrays['anchor_value']-x@arrays['anchor_gradient']
            if 'SKIP_ICNN' in kind:
                assert float(a['gate'])>=0 and (a['coef']>=0).all()
                eta=torch.tensor(a['coef']*a['eta_unit']);value=base+(eta*scale*x**5/5).sum(-1)+float(a['residual_unit']*a['gate'])*residual
            else:value=base+float(a['Vscale']*a['gain'])*residual
        gradient=torch.autograd.grad(value.sum(),q,create_graph=True)[0]
        hv=torch.autograd.grad((gradient*torch.tensor(direction)).sum(),q)[0]
    else:value=base;gradient=c0.expand_as(q);hv=torch.zeros_like(q)
    tv=value.detach().numpy();tt=gradient.detach().numpy();hh=hv.detach().numpy()
    atol=max(1e-8,64*np.finfo(float).eps*float(np.max(abs(v),initial=1.)))
    assert np.allclose(tv,v,rtol=1e-9,atol=atol),'Portable value differs'
    assert np.allclose(tt,t,rtol=1e-9,atol=1e-8),'Portable native cost differs'
    assert np.allclose(hh,h,rtol=1e-9,atol=1e-8),'Portable HVP differs'
    curvature=np.sum(h*direction,axis=1);allowance=1e-10*np.maximum(1,np.sum(abs(h*direction),axis=1))
    assert np.all(curvature>=-allowance),'Negative directional curvature'
    return dict(status='PASS',kind=kind,max_value_difference=float(np.max(abs(tv-v))),max_cost_difference=float(np.max(abs(tt-t))),
        max_HVP_difference=float(np.max(abs(hh-h))),minimum_directional_curvature=float(curvature.min()),full_Hessian_built=False,source_oracle_in_online_query=False)

def evaluate_region(c,source,rid):
    assert (root(c)/'control/ALL_CHAMPIONS_FROZEN.json').exists()
    dest=rd(c,rid)/'final_evaluation';done=dest/'RESULT.json'
    if done.exists():return read_json(done)
    started=time.perf_counter();union=Union(c,source,rid,final=True)
    frozen=read_json(rd(c,rid)/'training/cycles/final/manifest.json');roster={}
    for kind,b in frozen['files'].items():
        p=rd(c,rid)/b['path'];assert sha(p)==b['sha256']
        with np.load(p,allow_pickle=False) as z:roster[kind]={k:z[k] for k in z.files}
    append_json(root(c)/'control/protected_access_log.jsonl',dict(region_id=rid,unix=time.time(),action='FINAL_HELDOUT_LABEL_AND_PREDICTION_ACCESS',
        freeze_sha256=sha(root(c)/'control/ALL_CHAMPIONS_FROZEN.json')))
    mathrows=[portable_math(kind,a,union.s['low']) for kind,a in roster.items()]
    rows=[]
    splits=['TRAIN_old','TRAIN_new','VALID_old','PATCH_VALID','LOW_TEST','HIGH_PROJECTED','PATCH_AUDIT_OPERATING','PATCH_AUDIT_STRESS']
    for split in splits:
        fields=union.fields(split);u,t,L,U,eps,rr=fields
        if not len(u):continue
        operating=operating_mask(u,union.s['low'])
        masks=[('ALL','ALL',np.arange(len(u))),('OPERATING','ALL',np.flatnonzero(operating)),('STRESS','ALL',np.flatnonzero(~operating))]
        for regime,mask in [('OPERATING',operating),('STRESS',~operating)]:
            for family in rr.family.unique():masks.append((regime,str(family),np.flatnonzero(mask&rr.family.eq(family))))
        for kind,a in roster.items():
            fn=lambda q,k=kind,aa=a:predict_arrays(k,aa,q)
            for regime,family,ix in masks:
                if not len(ix):continue
                subset=tuple(x[ix] for x in fields[:5])+(rr.iloc[ix].reset_index(drop=True),)
                metric=score_metric(fn,subset,union.s,full=True,batch=64 if union.meta['m_active']>=1024 else 128)
                rows.append(dict(region_id=rid,m_active=union.meta['m_active'],LOW_exposure=float(union.s['low'].sum()),split=split,regime=regime,family=family,
                    model_kind=kind,selected_champion=kind==frozen['champion_kind'],national_readiness=False,**metric))
    # Fresh final physical certification, including C diagnostic labels.
    from partC_rev1_src.phase6.storage import Store
    d,a,pairs,s=source.local(rid);counts={};reopened=0;reopen_start=time.perf_counter()
    for namespace in ['train_labels','valid_labels','audit_labels']:
        path=rd(c,rid)/'data_extension'/namespace
        if not path.exists():continue
        store=Store(rd(c,rid)/'data_extension',d,a,namespace=namespace)
        for receipt in (path/'receipts').glob('*.json'):
            r,rc=store.get(receipt.stem,recertify=True);reopened+=1;counts[r.quality]=counts.get(r.quality,0)+1
            assert np.allclose(np.bincount(store.oracle.pool_arrays()[0][r.path_ids],weights=r.weights,minlength=d['m_active']),r.demand,rtol=2e-11,atol=1e-12)
    dest.mkdir(parents=True,exist_ok=True);f=pd.DataFrame(rows);f.to_parquet(dest/'metrics.parquet',index=False)
    write_json(dest/'math.json',dict(status='PASS',candidates=mathrows))
    champion=f.loc[f.model_kind.eq(frozen['champion_kind']) & f.regime.eq('ALL') & f.family.eq('ALL')]
    hidden=champion.loc[champion.split.str.startswith('PATCH_AUDIT')]
    invalid_hidden=bool(hidden.negative_cost_count.fillna(0).gt(0).any() or hidden.nonfinite.any()) if len(hidden) else None
    result=dict(status='PASS',region_id=rid,m_active=d['m_active'],candidate_count=len(roster),metric_rows=len(f),champion_kind=frozen['champion_kind'],
        champion_is_analytic=frozen['champion_is_analytic'],champion_invalid_on_hidden_audit=invalid_hidden,hidden_audit_causes_reselection=False,
        physical_labels_freshly_reopened=reopened,quality_counts=counts,reopen_seconds=time.perf_counter()-reopen_start,seconds=time.perf_counter()-started,
        metrics_sha256=sha(dest/'metrics.parquet'),math_sha256=sha(dest/'math.json'))
    write_json(done,result);return result
