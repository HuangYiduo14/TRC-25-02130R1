"""Pure deterministic planning helpers. No external execution or hidden clocks."""
from __future__ import annotations
import hashlib,math


def active_budget(m:int, difficulty:int=1, width:int=32)->int:
    if m<0 or difficulty not in (1,2,4) or width not in (32,64):raise ValueError('Bad region size/difficulty/width')
    if m==0:return 0
    proxy=(2*width+1)*m+width*width+3*width+1
    base=32*math.ceil(max(64,proxy/32)/32)
    return difficulty*base


def seed_for(*parts, master=20260908)->int:
    text='|'.join([str(master),*[str(p) for p in parts]])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big') & ((1<<63)-1)


def failed_fraction_below_gate(failed:int,active:int=1566)->bool:
    if failed<0 or active<1 or failed>active:raise ValueError('Bad counts')
    return 20*failed < active


def can_admit(elapsed:float, forecast:float, finalization_forecast:float=0.,
              hard:float=86400., minimum_reserve:float=5400.)->bool:
    if min(elapsed,forecast,finalization_forecast)<0:raise ValueError('Negative time')
    reserve=max(minimum_reserve,1.5*finalization_forecast)
    return elapsed+forecast < hard-reserve


def validation_score(metric:dict)->float:
    vals=[metric[k] for k in ['MARE_equal_pairs','MARE_LOW_pair_weighted','mean_interval_violation_relative']]
    if not all(math.isfinite(x) and x>=0 for x in vals):return math.inf
    if metric.get('negative_cost_count',0)>0 or not metric.get('structural_valid',True):return math.inf
    return .5*(vals[0]+vals[1])+.1*vals[2]


def select_champion(metrics:dict)->str|None:
    """Caller supplies VALID-only metrics; return an explicit type, never fake ICNN."""
    valid=[(validation_score(v),k) for k,v in metrics.items()]
    good=[(s,k) for s,k in valid if math.isfinite(s)]
    if not good:return None
    best=min(s for s,k in good)
    tied=[(s,k) for s,k in good if s-best <= 1e-12*max(1.,abs(best))]
    def simplicity(k):
        name=k.upper()
        if 'FREEFLOW' in name:return 0
        if 'POWER4' in name and 'ICNN' not in name:return 1
        if '64' in name:return 3
        return 2
    return min(tied,key=lambda sk:(simplicity(sk[1]),sk[1]))[1]


def candidate_origin_ownership(existing_owner:str|None,new_owner:str)->str:
    if existing_owner is None:return 'NEW'
    if existing_owner=='TRAIN' and new_owner=='TRAIN':return 'REUSE_NOT_NEW'
    return 'REDRAW_PROTECTED_OR_ALIAS'
