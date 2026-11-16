"""Native candidate metrics and VALID-only scoring, with explicit analytic kinds."""
import math
import numpy as np
from partC_rev1_src.phase7.model import frozen_eval as legacy_eval,separable_power4_eval
from .portable import portable_eval
from .sampling import operating_mask

def predict_arrays(kind,a,u,direction=None):
    if kind in {'OLD_ICNN_FROZEN','LEGACY_ICNN_CONTINUED'}:return legacy_eval(a,u,direction)
    if 'SKIP_ICNN' in kind:return portable_eval(a,u,direction)
    if kind=='FREEFLOW_LINEAR':
        vals=(u@a['c0'],np.broadcast_to(a['c0'],u.shape))
        return vals if direction is None else (*vals,np.zeros_like(u))
    if kind in {'OLD_POWER4_NNLS','NEW_POWER4_RELATIVE_NNLS'}:
        vals=separable_power4_eval(u,a['scale'],a['c0'],a['eta'])
        return vals if direction is None else (*vals,4*a['eta']*(u/a['scale'])**3*np.asarray(direction)/a['scale'])
    raise ValueError('Unknown explicit model_kind '+kind)

def huber_np(x):
    a=abs(x);return np.where(a<=.1,.5*x*x,.1*(a-.05))

def score_metric(fn,fields,s,batch=128,full=False):
    u,t,L,U,eps,rows=fields;n=len(u);m=len(s['c0'])
    if not n:return None
    low=s['low']/s['low'].sum();w=s['pair_weight'];sums=np.zeros(14);neg=0;floor=0;minimum=math.inf
    coordinates=0;zero_count=0;zero_ae=0.;parts=[];potential_gaps=[];worst=(-1.,0,0)
    for i in range(0,n,batch):
        q=u[i:i+batch];truth=t[i:i+batch];v,p=fn(q)
        if not np.isfinite(v).all() or not np.isfinite(p).all():
            return dict(n_rows=n,nonfinite=True,eligible=False,negative_cost_count=neg,MARE_equal_pairs=None,MARE_LOW_pair_weighted=None,mean_interval_violation_relative=None)
        e=p-truth;ae=abs(e);relative=ae/np.maximum(abs(truth),1.)
        wi,wk=np.unravel_index(np.argmax(relative),relative.shape)
        if relative[wi,wk]>worst[0]:worst=(float(relative[wi,wk]),int(i+wi),int(wk))
        gap=np.maximum(L[i:i+batch]-eps[i:i+batch]-v,0)+np.maximum(v-U[i:i+batch]-eps[i:i+batch],0)
        dg=np.maximum(np.maximum(abs(truth),s['c0']),1.);dr=np.maximum(abs(truth-s['c0']),.05*dg)
        dv=np.maximum(np.maximum(abs(U[i:i+batch]),q@s['c0']),1.)
        floor_e=np.maximum(s['c0']-p,0)
        robusts=[huber_np(e/dg)@w,huber_np(e/dr)@w,huber_np(gap/dv),(floor_e/np.maximum(s['c0'],1.))**2@w]
        oldg=((e/s['cost_scale'])**2)@w;oldv=(gap/float(s['value_scale']))**2;oldf=(floor_e/s['cost_scale'])**2@w
        vals=[relative.mean(axis=1),relative@low,gap/np.maximum(abs(U[i:i+batch]),1.),ae.mean(axis=1),(e*e).mean(axis=1),e.mean(axis=1),oldg,oldv,oldf,*robusts,ae.mean(axis=1)/np.maximum(np.mean(abs(truth-s['c0']),axis=1),1.)]
        sums+=np.array([float(x.sum()) for x in vals]);coordinates+=p.size;neg+=int((p < -1e-8).sum());floor+=int((p<s['c0']-1e-8).sum());minimum=min(minimum,float(p.min()))
        zeros=q==0;zero_count+=int(zeros.sum());zero_ae+=float(ae[zeros].sum())
        if full:parts.append(relative.ravel());potential_gaps.extend(gap.tolist())
    sums/=n
    result=dict(n_rows=n,n_coordinates=coordinates,MARE_equal_pairs=sums[0],MARE_LOW_pair_weighted=sums[1],mean_interval_violation_relative=sums[2],
        MAE_seconds=sums[3],RMSE_seconds=math.sqrt(sums[4]),raw_MSE_seconds2=sums[4],bias_seconds=sums[5],
        legacy_cost_loss=sums[6],legacy_value_loss=sums[7],legacy_floor_loss=sums[8],legacy_composite=sums[6]+.1*sums[7]+.1*sums[8],
        robust_total=sums[9],robust_congestion=sums[10],robust_interval=sums[11],robust_floor=sums[12],robust_objective=.75*sums[9]+.25*sums[10]+.1*sums[11]+sums[12],
        congestion_relative_MAE=sums[13],negative_cost_count=neg,floor_violation_count=floor,minimum_cost_seconds=minimum,
        zero_demand_coordinate_count=zero_count,zero_demand_MAE_seconds=zero_ae/zero_count if zero_count else None,nonfinite=False,eligible=neg==0,
        worst_row_index=worst[1],worst_coordinate=worst[2],worst_input_hash=str(rows.iloc[worst[1]].input_hash))
    result['score']=.5*(sums[0]+sums[1])+.1*sums[2]
    result['accuracy_screen']=bool(sums[0]<=.05 and sums[1]<=.05 and sums[2]<=.01 and neg==0)
    if full:
        allrel=np.concatenate(parts);result.update(p50_relative=float(np.quantile(allrel,.5)),p95_relative=float(np.quantile(allrel,.95)),max_relative=float(allrel.max()),
            mean_interval_violation_native=float(np.mean(potential_gaps)),max_interval_violation_native=float(np.max(potential_gaps)))
    return {k:(v.item() if isinstance(v,np.generic) else v) for k,v in result.items()}

def validation_candidate(kind,a,union):
    fn=lambda u:predict_arrays(kind,a,u);splits={}
    for split in ['VALID_old','PATCH_VALID']:
        v=score_metric(fn,union.fields(split),union.s)
        if v is not None:splits[split]=v
    train=score_metric(fn,union.fields('TRAIN'),union.s)
    guards=np.stack([np.zeros_like(union.s['low']),union.s['low'],2*union.s['low'],np.maximum(union.s['low'],union.s['train_max'])])
    gv,gt=fn(guards);guard_ok=bool(np.isfinite(gv).all() and np.isfinite(gt).all() and (gt>=-1e-8).all())
    eligible=bool(train and train['eligible'] and splits and all(v['eligible'] for v in splits.values()) and guard_ok)
    score=float(np.mean([v['score'] for v in splits.values()])) if eligible else None
    return dict(model_kind=kind,score=score,eligible=eligible,valid=splits,train=train,guard_eligible=guard_ok,
        guard_minimum_cost=float(np.min(gt)) if np.isfinite(gt).all() else None,selection_source='VALID_old and PATCH_VALID equally weighted when present')

def choose_candidate(candidates):
    good=[(v['score'],k) for k,v in candidates.items() if v['eligible'] and v['score'] is not None and math.isfinite(v['score'])]
    if not good:return None
    best=min(s for s,k in good);ties=[(s,k) for s,k in good if s-best<=1e-12*max(1.,abs(best))]
    def order(k):return 0 if k=='FREEFLOW_LINEAR' else 1 if 'ICNN' not in k else 3 if '64' in k else 2
    return min(ties,key=lambda sk:(order(sk[1]),sk[1]))[1]
