"""Frozen heldouts, immutable input plans and TRAIN-only candidate disagreement."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from partC_rev1_src.phase6.sampling import draw,PACK
from .common import root,rd,read_json,write_json,save_npz,sha,digest,input_hash
from .policy import seed_for,active_budget

def operating_mask(u,low):
    u=np.asarray(u);total=u.sum(axis=1)/low.sum()
    return (np.all((u>=0)&(u<=2*low*(1+1e-14)),axis=1)&(total>=.25-1e-14)&(total<=2+1e-14))|np.all(u==0,axis=1)

def bounded(low,src,tgt,rng,family):
    gamma=float(rng.uniform(.5,1.5));m=len(low)
    if family=='RADIAL':ratios=np.full(m,gamma)
    elif family in ('SOURCE','TARGET'):
        _,inv=np.unique(src if family=='SOURCE' else tgt,return_inverse=True)
        ratios=gamma*np.exp(rng.normal(0,.35,inv.max()+1))[inv]
    elif family=='SPARSE':
        ratios=gamma*np.exp(rng.normal(0,.25,m));ratios[rng.random(m)<.2]=0.
        if not np.any(ratios):ratios[rng.integers(m)]=gamma
    else:ratios=gamma*np.exp(rng.normal(0,.35,m))
    u=low*np.clip(ratios,0,2);ratio=float(u.sum()/low.sum())
    if ratio<.25:u=low+(.75/(1-ratio))*(u-low)
    u[u==0]=0.
    assert operating_mask(u[None,:],low)[0]
    return u

def freeze_heldouts(c,source,rid):
    dest=rd(c,rid)/'protected';done=dest/'FROZEN.json'
    if done.exists():return read_json(done)
    pack=source.old_pack(rid);d,a,pairs,s=source.local(rid)
    existing=set(pack.rows.input_hash);xs=[];rows=[];attempts=[]
    src=pairs.source_terminal_index.to_numpy();tgt=pairs.target_terminal_index.to_numpy();low=pack.s['low']
    for split,count in [('PATCH_VALID',8),('PATCH_AUDIT_OPERATING',8),('PATCH_AUDIT_STRESS',8)]:
        for slot in range(count):
            family=PACK[slot*4%32] if split=='PATCH_AUDIT_STRESS' else ['RADIAL','PAIR','SOURCE','TARGET','SPARSE'][slot%5]
            for redraw in range(9):
                seed=seed_for(d['model_class'],d['model_hash'],d['support_hash'],rid,0,split,family,slot,redraw)
                rng=np.random.default_rng(seed)
                u,tag=draw(low,src,tgt,family,rng,s['bottleneck_masks']) if split=='PATCH_AUDIT_STRESS' else (bounded(low,src,tgt,rng,family),'LOW_BOUNDED')
                ih=input_hash(d['model_hash'],d['support_hash'],u)
                attempts.append(dict(split=split,slot=slot,redraw=redraw,input_hash=ih,collision=ih in existing))
                if ih not in existing:
                    existing.add(ih);rows.append(dict(region_id=rid,input_index=len(xs),split=split,family=family,slot=slot,seed=str(seed),redraw=redraw,input_hash=ih,
                        model_hash=d['model_hash'],support_hash=d['support_hash'],regime='OPERATING' if operating_mask(u[None,:],low)[0] else 'STRESS',grammar=tag))
                    xs.append(u);break
            else:raise ValueError('Unable to freeze distinct protected input after 8 redraws')
    save_npz(dest/'inputs.npz',u=np.asarray(xs));pd.DataFrame(rows).to_parquet(dest/'plan.parquet',index=False)
    result=dict(status='FROZEN_INPUTS_ONLY',region_id=rid,rows=len(rows),counts=pd.DataFrame(rows).split.value_counts().to_dict(),frozen_unix=time.time(),
        input_npz_sha256=sha(dest/'inputs.npz'),plan_sha256=sha(dest/'plan.parquet'),input_hashes=[r['input_hash'] for r in rows],
        no_labels_or_predictions_read=True,adaptive_sampling_not_started=True,collision_attempts=attempts)
    write_json(done,result);return result

def row_probabilities(u,low,families):
    operating=operating_mask(u,low);families=np.asarray(families,dtype=str);w=np.zeros(len(u))
    both=operating.any() and (~operating).any();counts={}
    for mask,name,mass in [(operating,'OPERATING',.75),(~operating,'STRESS',.25)]:
        n=int(mask.sum());counts[name]=n
        if not n:continue
        mass=mass if both else 1.;present=np.unique(families[mask])
        for family in present:
            take=mask&(families==family);w[take]=mass/(len(present)*int(take.sum()))
    assert len(u)>0 and (w>0).all() and np.isclose(w.sum(),1.)
    return w,dict(class_counts=counts,class_mass={'OPERATING':float(w[operating].sum()),'STRESS':float(w[~operating].sum())},families=sorted(str(f) for f in set(families)),all_rows_available=True)

def train_plan32(c,source,rid,cycle,pack_index,existing,models,train_u,model_scale):
    """models are frozen preceding-cycle native callables, with no heldout truth."""
    dest=rd(c,rid)/'data_extension/plans'/f'cycle_{cycle:04d}_pack_{pack_index:06d}'
    if (dest/'FROZEN.json').exists():
        manifest=read_json(dest/'FROZEN.json');assert sha(dest/'inputs.npz')==manifest['inputs_sha256'] and sha(dest/'plan.parquet')==manifest['plan_sha256']
        return dest
    d,a,pairs,s=source.local(rid);low=pairs.LOW_CURRENT.to_numpy(float);src=pairs.source_terminal_index.to_numpy();tgt=pairs.target_terminal_index.to_numpy()
    protected=read_json(rd(c,rid)/'protected/FROZEN.json');seen=set(existing)|set(protected['input_hashes']);xs=[];rows=[];attempts=[]
    # Exact inherited grammar, half of its 32 roles each pack; rotate across packs.
    inherited=[PACK[(2*j+pack_index%2)%32] for j in range(16)]
    for slot in range(32):
        mode='INHERITED' if slot<16 else 'LOW_BOUNDED' if slot<24 else 'TRAIN_ADAPTIVE'
        family=inherited[slot] if slot<16 else ['RADIAL','PAIR','SOURCE','TARGET','SPARSE'][(slot-16)%5]
        absolute_slot=pack_index*32+slot
        candidates=[]
        for poolslot in range(4 if mode=='TRAIN_ADAPTIVE' else 1):
            for redraw in range(9):
                seed=seed_for(d['model_class'],d['model_hash'],d['support_hash'],rid,cycle,mode,family,absolute_slot,poolslot,redraw)
                rng=np.random.default_rng(seed)
                if mode=='INHERITED':u,tag=draw(low,src,tgt,family,rng,s['bottleneck_masks'])
                elif mode=='TRAIN_ADAPTIVE' and poolslot%2:
                    grammar=PACK[(absolute_slot+poolslot)%32];u,tag=draw(low,src,tgt,grammar,rng,s['bottleneck_masks'])
                else:u,tag=bounded(low,src,tgt,rng,family),'LOW_BOUNDED'
                ih=input_hash(d['model_hash'],d['support_hash'],u)
                attempts.append(dict(slot=absolute_slot,poolslot=poolslot,redraw=redraw,input_hash=ih,duplicate=ih in seen))
                if ih not in seen:
                    candidates.append((u,ih,seed,redraw,tag));break
        if not candidates:
            rows.append(dict(region_id=rid,slot=absolute_slot,input_index=-1,disposition='NO_UNIQUE_INPUT',split='TRAIN',family=mode+'_'+family));continue
        scores=np.zeros(len(candidates))
        if mode=='TRAIN_ADAPTIVE':
            pool=np.stack([v[0] for v in candidates]);pred=[fn(pool)[1] for fn in models]
            scale=np.maximum(s['c0'],1.)
            disagreement=np.max(np.stack(pred),axis=0)-np.min(np.stack(pred),axis=0) if len(pred)>1 else np.zeros_like(pool)
            scores=np.sqrt(np.mean((disagreement/scale)**2,axis=1))
            # Squared Euclidean nearest-neighbor distance in frozen model coordinates.
            xn=np.asarray(train_u)/model_scale;qn=pool/model_scale
            distances=np.maximum(np.sum(qn*qn,axis=1)[:,None]+np.sum(xn*xn,axis=1)[None,:]-2*qn@xn.T,0.)
            scores+=np.sqrt(np.min(distances,axis=1)/len(low))
            if not np.isfinite(scores).all():raise ValueError('Nonfinite TRAIN-only adaptive score')
        k=int(np.argmax(scores));u,ih,seed,redraw,tag=candidates[k];seen.add(ih)
        rows.append(dict(region_id=rid,cycle=cycle,pack_index=pack_index,slot=absolute_slot,input_index=len(xs),disposition='NEW_INPUT',split='TRAIN',family=mode+'_'+family,
            regime='OPERATING' if operating_mask(u[None,:],low)[0] else 'STRESS',seed=str(seed),redraw=redraw,input_hash=ih,
            adaptive_score=float(scores[k]),adaptive_pool_size=len(candidates),model_hash=d['model_hash'],support_hash=d['support_hash'],grammar=tag))
        xs.append(u)
    save_npz(dest/'inputs.npz',u=np.asarray(xs).reshape(-1,len(low)));pd.DataFrame(rows).to_parquet(dest/'plan.parquet',index=False)
    write_json(dest/'FROZEN.json',dict(status='FROZEN_TRAIN_INPUTS',region_id=rid,cycle=cycle,pack_index=pack_index,rows=len(xs),
        inputs_sha256=sha(dest/'inputs.npz'),plan_sha256=sha(dest/'plan.parquet'),source='preceding frozen candidates and TRAIN only',attempts=attempts,frozen_unix=time.time()))
    return dest
