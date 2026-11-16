"""Long, resumable float64 training cycles with validation-selected typed candidates."""
import time,copy,math,io,traceback
from pathlib import Path
import numpy as np
import torch
from .common import root,rd,read_json,write_json,save_npz,sha,digest,admit
from .policy import seed_for
from .model import PowerResidualPotential,weighted_power4_fit,calibrate_residual_unit,export_arrays,robust_loss
from .metrics import validation_candidate,choose_candidate,predict_arrays
from .sampling import row_probabilities
from .data import Union
from partC_rev1_src.phase7.training import torch_load
from .common import atomic,is_e2
from partC_rev1_src.phase7.model import RegionalPotential,export_arrays as legacy_arrays

PRIMARY='POWER4_SKIP_ICNN32'
ESCALATION='POWER4_SKIP_ICNN64_POWER5_HEAD'
LEGACY='LEGACY_ICNN_CONTINUED'
CONTROL='SAME_UPDATE_OLD_DATA_CONTROL'

def torch_save(path,obj):
    payload=io.BytesIO();torch.save(obj,payload);atomic(path,payload.getvalue())

def copies(state):return {k:v.detach().clone() for k,v in state.items()}
def arrays_for(kind,model):return legacy_arrays(model) if kind==LEGACY else export_arrays(model)
def actual_kind(kind):return PRIMARY if kind==CONTROL else kind

def build_model(c,source,rid,kind,union,directory):
    old=union.old;s=old.s;idx=old.indices('TRAIN');u,t,L,U,eps=old.fields(idx)
    if kind==LEGACY:
        model=RegionalPotential(s['scale'].copy(),s['c0'].copy(),float(s['Vscale']),float(s['gain']),seed=old.meta['seed'])
        original=source.path7('regions/'+rid.removeprefix('partC_p2_region_')+'/best.pt')
        model.load_state_dict(torch_load(original)['model'])
        initialization=dict(source_kind='OLD_ICNN_FROZEN',source_best_sha256=sha(original),original_scales_preserved=True)
    else:
        width=64 if kind==ESCALATION else 32;power=5 if kind==ESCALATION else 1
        frozen=rd(c,rid)/'training/model_scales.npz'
        if frozen.exists():
            with np.load(frozen,allow_pickle=False) as z:scales={k:z[k] for k in z.files}
        else:
            scale=np.maximum(np.maximum(np.quantile(u,.95,axis=0),s['low']),1e-12);eta=weighted_power4_fit(u,t,scale,s['c0'])
            scales=dict(scale=scale,eta0=eta,eta_unit=np.maximum(np.maximum(eta,.01*s['c0']),1.))
            save_npz(frozen,**scales)
        failures=[]
        for attempt in range(2):
            model=PowerResidualPotential(scales['scale'],s['c0'],scales['eta0'],eta_unit=scales['eta_unit'],width=width,head_power=power,
                seed=seed_for(rid,'init',width,attempt))
            try:
                initialization=calibrate_residual_unit(model,u,t,U,s['pair_weight'],batch=64 if old.meta['m_active']>=1024 else 128)
                initialization.update(reinitializations=attempt,failures=failures,scale_source='OLD A/B TRAIN q95 plus LOW, frozen',scale_sha256=sha(frozen))
                break
            except ValueError as e:
                failures.append(repr(e))
                if attempt==1:raise
        assert sum(p.numel() for p in model.parameters())==(129*old.meta['m_active']+4289 if width==64 else 65*old.meta['m_active']+1121)
    directory.mkdir(parents=True,exist_ok=True)
    if kind==CONTROL:
        original=rd(c,rid)/'training'/PRIMARY/'init.pt'
        assert original.exists(),'Same-initialization control requires preserved primary initialization'
        model.load_state_dict(torch_load(original)['model']);initialization.update(same_initialization_sha256=sha(original))
    if not (directory/'init.pt').exists():
        torch_save(directory/'init.pt',dict(model=copies(model.state_dict()),kind=actual_kind(kind)))
        save_npz(directory/'initialized.npz',**arrays_for(kind,model));write_json(directory/'initialization.json',initialization)
    else:model.load_state_dict(torch_load(directory/'init.pt')['model'])
    return model

def new_state(model,kind,seed):
    rng=torch.Generator().manual_seed(seed)
    return dict(kind=kind,model=copies(model.state_dict()),optimizer=None,rng_state=rng.get_state(),order=torch.empty(0,dtype=torch.int64),cursor=0,
        steps=0,rows_seen=0,train_seconds=0.,check_seconds=0.,attempts=0,cycle=-1,cycle_steps=0,cycle_rows=0,cycle_epochs=0.,
        dataset_hash=None,n_train=0,unique_seen=torch.empty(0,dtype=torch.bool),best_model=None,best_score=None,
        best_steps=0,last_check_steps=-1,patience=0,lr_patience=0,plateau_score=None,plateau_train=None,status='TRAINING',last_step_seconds=.001)

def terminal(state,recipe):
    if state['cycle_epochs']>=recipe['cycle_max_effective_epochs'] or state['cycle_steps']>=recipe['cycle_max_updates']:return 'CYCLE_CAP'
    if state['cycle_epochs']>=recipe['minimum_effective_epochs_before_plateau'] and state['cycle_steps']>=recipe['minimum_updates_before_plateau'] and state['patience']>=recipe['plateau_checks']:
        return 'PLATEAU_NEXT_DATA'
    return 'TRAINING'

def check(model,kind,union,opt,state,directory,scheduled,c):
    start=time.perf_counter();a=arrays_for(kind,model);metrics=validation_candidate(actual_kind(kind),a,union)
    score=metrics['score'];train=metrics['train'];recipe=c['model_training']
    if score is not None and (state['best_score'] is None or score<state['best_score']):
        state.update(best_score=score,best_steps=state['steps'],best_model=copies(model.state_dict()),best_metrics=metrics)
    if scheduled:
        ref=state['plateau_score'];tr=state['plateau_train'];nowtr=train.get('robust_objective') if train else None
        improved=score is not None and (ref is None or score<ref*(1-recipe['plateau_relative_improvement']))
        train_improved=nowtr is not None and (tr is None or nowtr<tr*(1-recipe['plateau_relative_improvement']))
        if improved:state.update(plateau_score=score,patience=0,lr_patience=0)
        else:
            state['lr_patience']+=1
            state['patience']=0 if train_improved else state['patience']+1
        if train_improved:state['plateau_train']=nowtr
        if state['lr_patience']>=recipe['lr_plateau_checks']:
            for group in opt.param_groups:group['lr']=max(recipe['lr_min'],group['lr']*recipe['lr_factor'])
            state['lr_patience']=0
        state['last_check_steps']=state['steps']
    state['last_metrics']=metrics;state['check_seconds']+=time.perf_counter()-start
    from .common import append_json
    append_json(directory/'learning_curves.jsonl',dict(unix=time.time(),kind=kind,cycle=state['cycle'],steps=state['steps'],cycle_steps=state['cycle_steps'],
        cycle_effective_epochs=state['cycle_epochs'],cycle_rows_consumed=state['cycle_rows'],n_TRAIN=state['n_train'],dataset_hash=state['dataset_hash'],
        row_probabilities=state.get('row_probability_summary'),unique_TRAIN_seen=int(state['unique_seen'].sum()),lr=opt.param_groups[0]['lr'],
        train_seconds=state['train_seconds'],scheduled=scheduled,patience=state['patience'],metrics=metrics))

def train_slice(c,source,rid,cycle,*,first=False,kind=PRIMARY,max_updates=None,old_only=False):
    if (root(c)/'control/ALL_CHAMPIONS_FROZEN.json').exists():raise PermissionError('Training forbidden after final champion freeze')
    directory=rd(c,rid)/'training'/kind;directory.mkdir(parents=True,exist_ok=True)
    union=Union(c,source,rid,old_only=old_only or kind==CONTROL)
    model=build_model(c,source,rid,kind,union,directory) if not (directory/'last.pt').exists() else None
    state=torch_load(directory/'last.pt') if (directory/'last.pt').exists() else new_state(model,kind,seed_for(rid,'shuffle',actual_kind(kind)))
    if model is None:
        # Construct from frozen arrays without repeating q95/NNLS/Gamma calibration.
        if kind==LEGACY:model=RegionalPotential(union.s['scale'].copy(),union.s['c0'].copy(),float(union.s['Vscale']),float(union.s['gain']),seed=union.old.meta['seed'])
        else:
            sd=state['model'];width=sd['net.A1'].shape[0]
            model=PowerResidualPotential(sd['scale'],sd['c0'],sd['coef']*sd['eta_unit'],eta_unit=sd['eta_unit'],residual_unit=float(sd['residual_unit']),width=width,head_power=5 if kind==ESCALATION else 1)
    model.load_state_dict(state['model']);recipe=c['model_training']
    opt=torch.optim.Adam(model.parameters(),lr=recipe['lr'])
    if state['optimizer'] is not None:opt.load_state_dict(state['optimizer'])
    rng=torch.Generator();rng.set_state(state['rng_state'])
    u,t,L,U,eps,rows=union.fields('TRAIN');n=len(u);assert n>=16
    probs,weight_record=row_probabilities(u,union.s['low'],rows.family.to_numpy());prob=torch.tensor(probs,dtype=torch.float64)
    change=state['dataset_hash']!=union.version_hash or state['cycle']!=cycle
    if change:
        if state['cycle']>=0 and state['best_model'] is not None:
            saved=directory/'snapshots'/f'cycle_{state["cycle"]:04d}_dataset_{state["dataset_hash"][:16]}.pt'
            if not saved.exists():torch_save(saved,dict(model=state['best_model'],steps=state['best_steps'],dataset_hash=state['dataset_hash'],score=state['best_score']))
        prior_best=state['best_model']
        state.update(cycle=cycle,cycle_steps=0,cycle_rows=0,cycle_epochs=0.,dataset_hash=union.version_hash,n_train=n,
            unique_seen=torch.zeros(n,dtype=torch.bool),order=torch.empty(0,dtype=torch.int64),cursor=0,patience=0,lr_patience=0,
            plateau_score=None,plateau_train=None,status='TRAINING',best_score=None,best_model=None,last_check_steps=-1,row_probability_summary=weight_record)
        for group in opt.param_groups:group['lr']=recipe['lr']
        # Both current and preceding best compete on the newly available VALID pool.
        last=copies(model.state_dict())
        if prior_best is not None:model.load_state_dict(prior_best);check(model,kind,union,opt,state,directory,False,c);model.load_state_dict(last)
        check(model,kind,union,opt,state,directory,False,c)
    limit=50 if first else recipe.get('slice_updates',200)
    if max_updates is not None:limit=min(limit,max_updates)
    seconds=5 if first else 30.;start_steps=state['steps'];start_seconds=state['train_seconds'];state['attempts']+=1
    batch=64 if union.meta['m_active']>=1024 else 128;w=torch.tensor(union.s['pair_weight'],dtype=torch.float64)
    while state['steps']-start_steps<limit and state['train_seconds']-start_seconds<seconds:
        if not admit(c,max(.1,2*state['last_step_seconds'])+2):state['status']='SLICE_RESOURCE_CUTOFF';break
        if kind!=CONTROL and terminal(state,recipe)!='TRAINING':state['status']=terminal(state,recipe);break
        if state['order'].numel()==0:
            # Persistent weighted draw order implements the exact declared mixture.
            state['order']=torch.multinomial(prob,n,replacement=True,generator=rng);state['cursor']=0
        end=min(n,state['cursor']+batch);idx=state['order'][state['cursor']:end].numpy()
        start=time.perf_counter();tensors=[torch.tensor(v,dtype=torch.float64) for v in [u[idx],t[idx],L[idx]-eps[idx],U[idx]+eps[idx]]]
        opt.zero_grad(set_to_none=True);loss,_=robust_loss(model,*tensors,w)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite robust TRAIN loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step()
        if kind!=LEGACY:model.project_parameters()
        if any(not torch.isfinite(v).all() for v in model.parameters()):raise FloatingPointError('Nonfinite candidate parameters')
        dt=time.perf_counter()-start;state['last_step_seconds']=dt;state['train_seconds']+=dt
        state['steps']+=1;state['rows_seen']+=len(idx);state['cycle_steps']+=1;state['cycle_rows']+=len(idx);state['cycle_epochs']=state['cycle_rows']/n
        state['unique_seen'][idx]=True;state['cursor']=end
        if end==n:state['order']=torch.empty(0,dtype=torch.int64);state['cursor']=0
        if state['steps']%100==0 or (state['cycle_epochs']>=1 and state['last_check_steps']<0):check(model,kind,union,opt,state,directory,True,c)
    if state['last_check_steps']!=state['steps']:check(model,kind,union,opt,state,directory,False,c)
    if state['status']=='TRAINING':state['status']=terminal(state,recipe) if kind!=CONTROL else 'CONTROL_PARTIAL'
    state.update(model=copies(model.state_dict()),optimizer=opt.state_dict(),rng_state=rng.get_state())
    torch_save(directory/'last.pt',state)
    chosen=state['best_model'] if state['best_model'] is not None else state['model'];model.load_state_dict(chosen)
    save_npz(directory/'best.npz',**arrays_for(kind,model))
    torch_save(directory/'best.pt',dict(model=chosen,steps=state['best_steps'] if state['best_model'] is not None else state['steps'],eligible=state['best_model'] is not None,dataset_hash=state['dataset_hash']))
    status={k:state[k] for k in ['status','cycle','steps','rows_seen','cycle_steps','cycle_rows','cycle_epochs','n_train','dataset_hash','best_score','best_steps','train_seconds','check_seconds','attempts','patience']}
    status.update(region_id=rid,model_kind=actual_kind(kind),training_variant=kind,best_eligible=state['best_model'] is not None,
        best_portable_sha256=sha(directory/'best.npz'),last_checkpoint_sha256=sha(directory/'last.pt'),row_probability_summary=weight_record,
        last_metrics=state['last_metrics'],best_metrics=state.get('best_metrics'),old_data_only=old_only or kind==CONTROL,
        last_slice_updates=state['steps']-start_steps,last_slice_optimizer_seconds=state['train_seconds']-start_seconds,updated_unix=time.time(),
        new_TRAIN_AB=0 if old_only or kind==CONTROL else union.rev['TRAIN_AB'],
        LBFGS='OMITTED: bounded float64 Adam with explicit parameter projection; no smooth-constraint line-search implementation used')
    write_json(directory/'status.json',status);return status

def candidates(c,source,rid,union):
    s=union.s
    result={'OLD_ICNN_FROZEN':source.old_arrays(rid),'FREEFLOW_LINEAR':{'c0':s['c0']},
        'OLD_POWER4_NNLS':dict(scale=s['scale'],c0=s['c0'],eta=s['baseline_eta'])}
    frozen=rd(c,rid)/'training/model_scales.npz'
    if frozen.exists():
        with np.load(frozen,allow_pickle=False) as z:scale=z['scale']
    else:
        u=union.old.fields(union.old.indices('TRAIN'))[0];scale=np.maximum(np.maximum(np.quantile(u,.95,axis=0),s['low']),1e-12)
    u,t,*_=union.fields('TRAIN');eta=weighted_power4_fit(u,t,scale,s['c0'])
    result['NEW_POWER4_RELATIVE_NNLS']=dict(scale=scale,c0=s['c0'],eta=eta)
    for kind in [PRIMARY,ESCALATION,LEGACY]:
        p=rd(c,rid)/'training'/kind/'best.npz'
        if p.exists():
            with np.load(p,allow_pickle=False) as z:result[kind]={k:z[k] for k in z.files}
    return result

def freeze_cycle(c,source,rid,cycle,*,old_only=False,final=False):
    dest=rd(c,rid)/'training/cycles'/('final' if final else f'cycle_{cycle:04d}')
    if (dest/'manifest.json').exists():return read_json(dest/'manifest.json')
    union=Union(c,source,rid,old_only=old_only);roster=candidates(c,source,rid,union);metrics={};files={}
    math_evidence={};failures={}
    for kind,a in roster.items():
        p=dest/(kind+'.npz');save_npz(p,**a);files[kind]=dict(path=p.relative_to(rd(c,rid)).as_posix(),sha256=sha(p))
        if is_e2(c):
            try:
                from .final_evaluation import portable_math
                math_evidence[kind]=portable_math(kind,a,union.s['low'])
                metrics[kind]=validation_candidate(kind,a,union)
            except (AssertionError,ValueError,FloatingPointError,RuntimeError) as error:
                failures[kind]=dict(error_type=type(error).__name__,error=repr(error),traceback=traceback.format_exc(),
                    portable_sha256=files[kind]['sha256'],permitted_inputs_only=True,thresholds_unchanged=True)
                metrics[kind]=dict(eligible=False,score=None,train={},valid={},candidate_math_failure=failures[kind])
        else:metrics[kind]=validation_candidate(kind,a,union)
    champion=choose_candidate(metrics)
    result=dict(status='FROZEN_CYCLE_CANDIDATES',region_id=rid,cycle=cycle,dataset_hash=union.version_hash,revision=union.rev,
        m_active=union.meta['m_active'],model_hash=union.meta['source_descriptor']['model_hash'],support_hash=union.meta['source_descriptor']['support_hash'],pair_order_hash=union.meta['source_pair_order_hash'],
        old_data_only=old_only,new_TRAIN_AB=0 if old_only else union.rev['TRAIN_AB'],metrics=metrics,files=files,
        champion_kind=champion,standalone_ICNN_kind=PRIMARY,champion_is_analytic=bool(champion and 'ICNN' not in champion),
        frozen_unix=time.time(),final_selection=final,TEST_HIGH_AUDIT_used=False,
        candidate_math=math_evidence,candidate_failures=failures)
    write_json(dest/'manifest.json',result);return result
