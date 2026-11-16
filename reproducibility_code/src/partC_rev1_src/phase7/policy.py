"""Small tested policy functions. These do not implement the full campaign scheduler."""
from __future__ import annotations
import hashlib, json, math
import numpy as np

QUALITY={'A','B'}
SPLITS={'TRAIN','VALID','LOW_TEST','HIGH_PROJECTED'}

def region_seed(region_id: str, master: int=20260908) -> int:
    b=f'{master}|P7_S03_ACTIVE_ICNN_V1|{region_id}|seed0'.encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8],'big')%(2**63-1)

def eligible(meta,final_state):
    if meta['m_active']==0:return False,'NO_ACTIVE_LOW_PAIRS'
    if final_state!='HEALTHY_BASELINE_READY':return False,'SOURCE_NOT_HEALTHY'
    # descriptor.region_status can be historical ACTIVE_BASELINE_PENDING; ignore it.
    return True,'ELIGIBLE'

def assert_split_ownership(records):
    owner={}
    for row in records:
        if row['quality'] not in {'A','B','C'}:raise ValueError('Unknown source quality')
        if row['split'] not in SPLITS:raise ValueError('Unknown canonical split')
        key=(row['model_hash'],row['support_hash'],row['input_hash'])
        if key in owner:
            raise ValueError('Duplicate canonical input: audit aliases without multiplying samples')
        owner[key]=row['split']
    return True

def manifest_counts(rows):
    return {s:sum(r['quality'] in QUALITY and r['split']==s for r in rows) for s in sorted(SPLITS)}

def stage_limits(start,now,remaining_eval_forecast=0.):
    hard=start+28800.
    reserve=max(2700.,1.5*max(0.,remaining_eval_forecast))
    cutoff=hard-reserve
    return {'hard':hard,'cutoff':cutoff,'reserve':reserve,'can_train':now<cutoff}

def result_status(n_active,n_terminal_trained,n_failed,n_untrained):
    if n_terminal_trained+n_failed+n_untrained!=n_active:raise ValueError('Coverage does not close')
    if n_untrained>0:return 'PHASE7_8H_PARTIAL_COHORT_READY_FOR_REVIEW'
    if n_failed>0:return 'PHASE7_COMPLETE_WITH_REGION_EXCEPTIONS_WAITING_EXTERNAL_REVIEW'
    return 'PHASE7_COMPLETE_WAITING_EXTERNAL_REVIEW'

def input_box_flag(u,lo,hi):
    u=np.asarray(u);return bool(np.any((u<lo)|(u>hi)))

def metrics(cost,target,low,values=None,lower=None,upper=None):
    cost=np.asarray(cost,float);target=np.asarray(target,float);low=np.asarray(low,float)
    if cost.shape!=target.shape or cost.ndim!=2 or cost.shape[1]!=len(low):raise ValueError('Shape')
    e=abs(cost-target)/np.maximum(abs(target),1.)
    w=low/low.sum()
    out={'MARE_equal_pairs':float(e.mean()),'MARE_LOW_pair_weighted':float((e@w).mean()),
         'p95_relative':float(np.quantile(e,.95)),'max_relative':float(e.max()),
         'MAE_seconds':float(abs(cost-target).mean()),'max_seconds':float(abs(cost-target).max()),
         'negative_cost_count':int((cost < -1e-8).sum())}
    if values is not None:
        gap=np.maximum(lower-values,0)+np.maximum(values-upper,0)
        out['mean_interval_violation_relative']=float(np.mean(gap/np.maximum(abs(upper),1.)))
    return out
