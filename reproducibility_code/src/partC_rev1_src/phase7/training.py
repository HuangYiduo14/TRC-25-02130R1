"""Exact resumable float64 Adam slices. Only TRAIN and VALID reach this module."""
from __future__ import annotations
import io,time,traceback
from pathlib import Path
import numpy as np
import torch
from .common import root,region_out,write_json,read_json,atomic,append_json,sha,seconds_left
from .data import Pack,pack_region
from .model import RegionalPotential,objective
from .evaluation import validation_metrics,evaluate_frozen

VERIFIED={}
def get_pack(path):
    path=Path(path);key=str(path)
    meta=read_json(path/'manifest.json')
    identity=(sha(path/'manifest.json'),tuple((n,(path/n).stat().st_size,(path/n).stat().st_mtime_ns) for n in meta['files']))
    verify=VERIFIED.get(key)!=identity
    p=Pack(path,verify=verify);VERIFIED[key]=identity
    return p

def torch_save(path,obj):
    b=io.BytesIO();torch.save(obj,b);atomic(path,b.getvalue())
def torch_load(path):return torch.load(path,map_location='cpu',weights_only=True)
def make_model(pack):
    s=pack.s
    return RegionalPotential(s['scale'].copy(),s['c0'].copy(),float(s['Vscale']),float(s['gain']),seed=pack.meta['seed'])
def new_state(model,pack,recipe):
    gen=torch.Generator().manual_seed(pack.meta['seed'])
    n=len(pack.indices('TRAIN'))
    return dict(model=model.state_dict(),optimizer=None,rng_state=gen.get_state(),order=torch.empty(0,dtype=torch.int64),cursor=0,epoch=0,
        steps=0,rows_seen=0,unique_seen=torch.zeros(n,dtype=torch.bool),best_score=None,best_epoch=None,best_steps=0,
        plateau_best=None,bad_checks=0,lr_bad_checks=0,screen_checks=0,last_scheduled_epoch=-1,
        train_seconds=0.,validation_seconds=0.,load_seconds=0.,attempts=0,status='TRAINING',last_step_seconds=.005,
        pack_manifest_sha256=sha(pack.path/'manifest.json'),recipe=recipe)

def terminal_rule(state,recipe):
    if state['epoch']>=recipe['epochs_max']:return 'TRAINED_EPOCH_LIMIT'
    if state['epoch']>=recipe['epochs_min_for_terminal_early_stop']:
        if state['screen_checks']>=2:return 'TRAINED_VALIDATION_TARGET'
        if state['bad_checks']>=recipe['early_stop_patience_checks']:return 'TRAINED_EARLY_STOP'
    if state['train_seconds']>=recipe['region_train_seconds_max']:return 'TRAINED_PARTIAL_COMPUTE_CAP'
    return 'TRAINING'

def check_valid(model,pack,opt,state,directory,scheduled):
    t0=time.perf_counter();v=validation_metrics(model,pack)
    train=validation_metrics(model,pack,'TRAIN')
    state['validation_seconds']+=time.perf_counter()-t0
    score=v['composite_loss']
    if state['best_score'] is None or score<state['best_score']:
        state.update(best_score=score,best_epoch=state['epoch'],best_steps=state['steps'])
        torch_save(directory/'best.pt',dict(model=model.state_dict(),epoch=state['epoch'],steps=state['steps'],score=score,pack_manifest_sha256=state['pack_manifest_sha256']))
    if scheduled:
        recipe=state['recipe'];ref=state['plateau_best']
        improved=ref is None or score<ref*(1-recipe['plateau_relative_improvement'])
        if improved:state.update(plateau_best=score,bad_checks=0,lr_bad_checks=0)
        else:state['bad_checks']+=1;state['lr_bad_checks']+=1
        if state['lr_bad_checks']>=recipe['lr_plateau_checks']:
            for group in opt.param_groups:group['lr']=max(recipe['lr_min'],group['lr']*recipe['lr_factor'])
            state['lr_bad_checks']=0
        state['screen_checks']=state['screen_checks']+1 if v['accuracy_screen'] else 0
        state['last_scheduled_epoch']=state['epoch']
    curve=dict(epoch=state['epoch'],fractional_epoch=state['epoch']+state['cursor']/len(pack.indices('TRAIN')),steps=state['steps'],rows_seen=state['rows_seen'],
        unique_rows_seen=int(state['unique_seen'].sum()),lr=opt.param_groups[0]['lr'],train_seconds=state['train_seconds'],validation_seconds=state['validation_seconds'],
        scheduled=scheduled,train=train,valid=v)
    append_json(directory/'learning_curves.jsonl',curve)
    state['last_valid']=v;state['last_train']=train

def run_slice(c,row,first,*,max_steps_override=None,finalize_override=False):
    """Worker writes only its exclusively assigned regional directory."""
    torch.set_num_threads(1);torch.set_num_interop_threads(1) if torch.get_num_interop_threads()!=1 else None
    started=time.perf_counter();rid=row['region_id'];directory=region_out(c,rid);directory.mkdir(parents=True,exist_ok=True)
    stage='PACK';state=None
    try:
        meta=pack_region(c,row)
        if not meta['eligible']:
            result=dict(region_id=rid,status='INSUFFICIENT_INDEPENDENT_AB_DATA',steps=0,attempts=0,pack=meta)
            write_json(directory/'status.json',result);return result
        stage='LOAD';t0=time.perf_counter();pack=get_pack(directory/'pack');recipe=c['model_and_training'];model=make_model(pack)
        state=torch_load(directory/'last.pt') if (directory/'last.pt').exists() else new_state(model,pack,recipe)
        assert state['pack_manifest_sha256']==sha(pack.path/'manifest.json') and state['recipe']==recipe
        model.load_state_dict(state['model']);opt=torch.optim.Adam(model.parameters(),lr=recipe['learning_rate'],betas=tuple(recipe['betas']),eps=recipe['eps'],weight_decay=0.)
        if state['optimizer'] is not None:opt.load_state_dict(state['optimizer'])
        gen=torch.Generator();gen.set_state(state['rng_state']);train_idx=pack.indices('TRAIN');n=len(train_idx)
        batch=recipe['batch_size_m_ge_1024'] if meta['m_active']>=1024 else recipe['batch_size']
        denom=torch.tensor(pack.s['cost_scale']);weight=torch.tensor(pack.s['pair_weight']);vs=float(pack.s['value_scale'])
        state['load_seconds']+=time.perf_counter()-t0
        limit_steps=recipe['first_slice_steps'] if first else recipe['subsequent_slice_steps']
        if first:limit_steps=max(0,limit_steps-state['steps'])
        if max_steps_override is not None:limit_steps=min(limit_steps,max_steps_override)
        limit_seconds=recipe['first_slice_train_seconds'] if first else recipe['subsequent_slice_train_seconds']
        if first:limit_seconds=max(0.,limit_seconds-state['train_seconds'])
        start_steps=state['steps'];start_seconds=state['train_seconds'];state['attempts']+=1;stage='TRAIN'
        while state['steps']-start_steps<limit_steps and state['train_seconds']-start_seconds<limit_seconds and not finalize_override:
            forecast=max(.02,2*state['last_step_seconds'])
            if seconds_left(c, float(c.get('effective_reserve_seconds',2700.)))<=forecast+3:
                state['status']='TRAINED_PARTIAL_TIME_LIMIT' if state['steps'] else 'PENDING_TRAIN';break
            if state['train_seconds']+forecast>recipe['region_train_seconds_max']:
                state['status']='TRAINED_PARTIAL_COMPUTE_CAP';break
            if terminal_rule(state,recipe)!='TRAINING':state['status']=terminal_rule(state,recipe);break
            if state['order'].numel()==0:
                state['order']=torch.randperm(n,generator=gen);state['cursor']=0
            stop=min(n,state['cursor']+batch);positions=state['order'][state['cursor']:stop].numpy();idx=train_idx[positions]
            t0=time.perf_counter();u,t,L,U,eps=pack.fields(idx)
            tensors=[torch.tensor(a,dtype=torch.float64) for a in [u,t,L-eps,U+eps]]
            opt.zero_grad(set_to_none=True)
            loss,_=objective(model,*tensors,denom,weight,vs)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite composite TRAIN loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['gradient_norm_clip'],error_if_nonfinite=True)
            opt.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite model after update')
            dt=time.perf_counter()-t0;state['last_step_seconds']=dt;state['train_seconds']+=dt
            state['steps']+=1;state['rows_seen']+=len(idx);state['unique_seen'][positions]=True;state['cursor']=stop
            if stop==n:
                state['epoch']+=1;state['cursor']=0;state['order']=torch.empty(0,dtype=torch.int64)
                if state['epoch']==1 or state['epoch']%recipe['validation_every_epochs']==0:
                    check_valid(model,pack,opt,state,directory,True)
                    state['status']=terminal_rule(state,recipe)
                    if state['status']!='TRAINING':break
        if finalize_override and state['status']=='TRAINING':state['status']='TRAINED_PARTIAL_TIME_LIMIT'
        # Every saved slice gets a VALID checkpoint; scheduled patience is not advanced by cutoffs.
        if state['steps']>0 and not (state['cursor']==0 and state['last_scheduled_epoch']==state['epoch']):
            check_valid(model,pack,opt,state,directory,False)
        elif state['steps']>0 and state['best_score'] is None:check_valid(model,pack,opt,state,directory,False)
        state.update(model=model.state_dict(),optimizer=opt.state_dict(),rng_state=gen.get_state())
        torch_save(directory/'last.pt',state)
        result={k:state[k] for k in ['status','steps','epoch','rows_seen','best_score','best_epoch','best_steps','train_seconds','validation_seconds','load_seconds','attempts']}
        result.update(region_id=rid,m_active=meta['m_active'],unique_rows_seen=int(state['unique_seen'].sum()),slice_wall_seconds=time.perf_counter()-started,
            pack_seconds=meta['pack_seconds'],counts_AB=meta['counts_AB'],alias_count=meta['alias_count'],eligible=True,
            accuracy_adequacy=bool(state.get('last_valid',{}).get('accuracy_screen',False)),native_runtime_valid=None)
        if state['status']!='TRAINING' and state['steps']>0:
            stage='EXPORT_EVALUATION';best=torch_load(directory/'best.pt');model.load_state_dict(best['model'])
            # This immutable marker precedes any LOW_TEST/HIGH access by evaluation.
            write_json(directory/'CHECKPOINT_FROZEN.json',dict(best_sha256=sha(directory/'best.pt'),best_steps=best['steps'],best_epoch=best['epoch'],unix=time.time(),resume_forbidden_after_heldout=True))
            evaluation=evaluate_frozen(model,pack,directory/'evaluation',state)
            result.update(native_runtime_valid=True,evaluation_seconds=evaluation['evaluation_seconds'])
        write_json(directory/'status.json',result);return result
    except Exception as e:
        status='DATA_CONTRACT_FAILURE' if stage=='PACK' else 'EXPORT_VALIDATION_FAILURE' if stage=='EXPORT_EVALUATION' else 'TRAINING_NUMERICAL_FAILURE'
        if isinstance(e,TimeoutError):status='PENDING_PACK' if stage=='PACK' else 'TRAINED_PARTIAL_TIME_LIMIT'
        error=dict(region_id=rid,status=status,stage=stage,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),
            model_hash=row['model_hash'],support_hash=row['support_hash'],steps=state['steps'] if state else 0,attempts=state['attempts'] if state else 0,
            preserved_last_checkpoint=str(directory/'last.pt') if (directory/'last.pt').exists() else None,unix=time.time())
        write_json(directory/f'failure_{int(time.time()*1000)}.json',error);write_json(directory/'status.json',error);return error
