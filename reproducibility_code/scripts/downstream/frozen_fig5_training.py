"""MANUSCRIPT12H fixed-design Figure5 runner. No work at import.

The SOURCE reader has no output methods; old REC root is a read namespace only.
The writer owns only this new run's explicit experiment directory. No TEST method
is callable in fitting mode. Original model/data files are never modified.
"""
from pathlib import Path
import argparse, hashlib, json, math, os, shutil, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp

SCHEMA='MANUSCRIPT12H_FIG5_V1'
POLICY=dict(icnn_width=32,icnn_updates=1000,icnn_learning_rate=.001,
    icnn_audit_interval=250,xgb_rounds=32,xgb_depth=3,xgb_learning_rate=.05,
    batch_rows=128,threads=1)
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def binding(path):
    p=Path(path).resolve(strict=True);return dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_new(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def atomic_json(path,value):
    p=Path(path);temp=p.with_name(p.name+'.'+str(time.time_ns())+'.tmp')
    write_new(temp,value);os.replace(temp,p)
def bound(b):
    p=Path(b['path']).resolve(strict=True)
    if not p.is_file() or digest(p)!=b['sha256']:raise ValueError('Changed bound input: '+str(p))
    if 'bytes' in b and p.stat().st_size!=b['bytes']:raise ValueError('Changed bound size')
    return p
def task_key(t):return (t['region_key'],float(t['fraction']),t['family'])
def config(path):
    p=Path(path).resolve(strict=True);c=read(p)
    if c['schema']!=SCHEMA or c['phase_id']!='MANUSCRIPT12H':raise ValueError('Explicit phase/config required')
    if c['policy']!=POLICY or c['seed']!=20260928:raise ValueError('Frozen fit definition changed')
    a=read(bound(c['authority']))
    if not a['training_authorized'] or a['phase_id']!=c['phase_id']:raise PermissionError('Training authority missing')
    if c['science_deadline_epoch']!=a['science_deadline_epoch'] or c['deadline_epoch']!=a['deadline_epoch']:raise ValueError('Original clock changed')
    if not a['start_epoch']<c['fit_deadline_epoch']<c['science_deadline_epoch']:raise ValueError('No evaluation reserve')
    if c['workers']>3 or c['xgb_workers']>2 or c['xgb_threads'] not in (1,2):raise ValueError('Concurrency not bounded')
    if c['resources']['rss_bytes']>24*2**30 or c['resources']['min_available_bytes']<3*2**30:raise ValueError('Unsafe RAM policy')
    root=Path(c['output_root']).resolve();new_run=Path(c['authority']['path']).resolve().parent.parent
    if not root.is_relative_to(new_run/'work/training'):raise ValueError('Output outside owned new run')
    for src,sha in c['source_bindings'].items():
        if digest(src)!=sha:raise ValueError('Changed implementation: '+src)
    plan=read(bound(c['plan']))
    if plan['policy']!=POLICY or plan['seed']!=c['seed']:raise ValueError('Frozen source plan changed')
    if len(plan['tasks'])!=160 or len({task_key(t) for t in plan['tasks']})!=160 or len(plan['regions'])!=16:raise ValueError('Nonexact frozen task set')
    if sorted(c['cohorts'])!=sorted(r['region_key'] for r in plan['regions']):raise ValueError('Cohort universe changed')
    if str(Path(c['repository']).resolve()) not in sys.path:sys.path.insert(0,str(Path(c['repository']).resolve()))
    return c,plan

class Guard:
    def __init__(self,c,end):
        self.c=c;self.end=min(float(end),c['science_deadline_epoch']);self.last=0;self.peak=0;self.minfree=2**63-1
    def __call__(self,*,time_check=True):
        import psutil
        now=time.time()
        if (Path(self.c['output_root'])/'STOP_REQUESTED.json').exists():raise RuntimeError('Owned controller requested stop; finite checkpoints retained')
        if time_check and now>=self.end:raise TimeoutError('Explicit new-campaign action deadline reached')
        if 0<=now-self.last<2:return
        proc=psutil.Process();rss=proc.memory_info().rss
        owned={proc.pid:proc}
        marker=str(Path(self.c['authority']['path']).resolve().parent.parent).lower().replace('\\','/')
        for candidate in psutil.process_iter(['pid','cmdline']):
            try:
                args=' '.join(candidate.info.get('cmdline') or []).lower().replace('\\','/')
                if marker in args:owned[candidate.pid]=candidate
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        for parent in list(owned.values()):
            try:
                for child in parent.children(recursive=True):owned[child.pid]=child
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        rss=0
        for child in owned.values():
            try:rss+=child.memory_info().rss
            except psutil.NoSuchProcess:pass
        free=psutil.virtual_memory().available
        self.peak=max(self.peak,rss);self.minfree=min(self.minfree,free);self.last=now
        if rss>self.c['resources']['rss_bytes'] or free<self.c['resources']['min_available_bytes']:raise MemoryError('Guarded process-tree/free RAM exceeded')
        disk=shutil.disk_usage(Path(self.c['output_root']).parent).free
        if disk<self.c['resources']['min_disk_free_bytes']:raise OSError('Disk floor reached')
        if self.c['resources']['baseline_disk_free_bytes']-disk>self.c['resources']['max_added_bytes']:raise OSError('Conservative added-disk allowance exceeded')

class SourceReader:
    """Read-only compatibility adapter; root/source_root are immutable INPUT roots."""
    def __init__(self,c,guard):
        self.request=c;self.guard_fn=guard;self.path=bound(c['labels_config']);self.config_sha256=c['labels_config']['sha256'];self.c=read(self.path)
        self.root=Path(self.c['output_root']).resolve();self.source_root=Path(self.c['source_root']).resolve()
        self.bindings={};self.source_expected={}
        pins=self.c.get('checkpoint_metadata_pins')
        if pins:
            data=read(bound(pins));self.source_expected={os.path.normcase(str(Path(k).resolve())):v for k,v in data['bindings'].items()}
        for b in c['cohorts'].values():
            for item in [b['index'],b['identity']]:self.source_expected[os.path.normcase(str(Path(item['path']).resolve()))]=item['sha256']
    def require_action(self,action):
        if action!='training':raise PermissionError('TRAIN/VALID reader cannot enter TEST')
        self.guard_fn()
    def guard(self,deadline=None):self.guard_fn()
    def bind(self,path,role,expected=None):
        self.guard_fn()
        p=Path(path).resolve(strict=True)
        allowed=[self.root,self.source_root,Path(self.request['repository']).resolve()]
        if not any(p.is_relative_to(r) for r in allowed):raise PermissionError('Source path outside explicit immutable inputs')
        if role.startswith(('HELDOUT','BLIND','FROZEN_TEST','ADMITTED_TEST')):raise PermissionError('TEST read prohibited')
        st=p.stat();stamp=(st.st_size,st.st_mtime_ns)
        prior=self.bindings.get(str(p))
        h=prior['sha256'] if prior and prior['stamp']==stamp else digest(p)
        if (p.stat().st_size,p.stat().st_mtime_ns)!=stamp:raise ValueError('Source changed during read')
        pin=self.source_expected.get(os.path.normcase(str(p)))
        if (expected is not None and h!=expected) or (pin is not None and h!=pin):raise ValueError('Source SHA differs: '+str(p))
        if prior and prior['sha256']!=h:raise ValueError('Previously read source changed')
        self.bindings[str(p)]=dict(sha256=h,stamp=stamp,role=role);return h

def load_cohorts(c,entry,guard):
    from partC_rev1_src.pacnhn_recovery24.training_recovery import load_training_cohorts
    reader=SourceReader(c,guard)
    frozen=c['cohorts'][entry['region_key']]
    bound(frozen['index']);bound(frozen['identity'])
    train,valid,region=load_training_cohorts(reader,entry['record'],labels_root=Path(frozen['index']['path']).parent,deadline=guard.end)
    if frozen.get('TRAIN_identity') and train['dataset_identity']!=frozen['TRAIN_identity']:raise ValueError('Historical full TRAIN changed')
    if frozen.get('VALID_identity') and valid['dataset_identity']!=frozen['VALID_identity']:raise ValueError('Historical VALID changed')
    return train,valid,region,reader.bindings

def _model_save(model,path,metadata):
    from partC_rev1_src.pacnhn_v1.models import save_model
    p=Path(path);tmp=p.with_name(p.stem+'.'+str(time.time_ns())+'.npz')
    save_model(model,tmp,metadata=metadata);os.replace(tmp,p)
    return binding(p)
def _torch_save(value,path):
    import torch
    p=Path(path);tmp=p.with_name(p.name+'.'+str(time.time_ns())+'.tmp')
    torch.save(value,tmp);os.replace(tmp,p);return binding(p)
def selection_key(tr,va,updates):return (tr['WMAPE_percent']>=1.,va['WMAPE_percent'],updates)
def fit_icnn(train,valid,seed,check,soft_end,folder,resume=None):
    import numpy as np,torch
    from partC_rev1_src.pacnhn_recovery24.ablation import subset_scale
    from partC_rev1_src.pacnhn_v1.models import MonotoneICNN,ICNN_SCALED,update_icnn,load_model
    from partC_rev1_src.pacnhn_v1.training import _initialize_icnn,_evaluate
    torch.set_num_threads(1);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    scale,calibration=subset_scale(train);width=POLICY['icnn_width']
    optimizer_state=None;updates=0;history=[];best=None
    if resume:
        if resume['TRAIN_identity']!=train['dataset_identity'] or resume['VALID_identity']!=valid['dataset_identity']:raise ValueError('Resume cohort mismatch')
        saved=torch.load(bound(resume['optimizer']),weights_only=True,map_location='cpu')
        if saved['seed']!=seed or not np.array_equal(saved['scale'].numpy(),scale):raise ValueError('Resume calibration/seed mismatch')
        model=MonotoneICNN(train['c0'],scale,width=width,seed=seed,device='cpu')
        model.load_state_dict(saved['model']);model.coefficient_parameterization=ICNN_SCALED;model.calibration=saved['calibration']
        optimizer_state=saved['optimizer'];updates=saved['updates'];history=resume['history'];best=resume['best']
    else:model=_initialize_icnn(train,scale,dict(seed=seed,icnn_coefficient_parameterization=ICNN_SCALED,icnn_initial_neural_cost_share=.001),'cpu',width)
    opt=torch.optim.Adam(model.parameters(),lr=POLICY['icnn_learning_rate'])
    gen=torch.Generator().manual_seed(seed+width)
    if optimizer_state is not None:opt.load_state_dict(optimizer_state);gen.set_state(saved['generator'])
    latest=None;generation=int(resume.get('generation',0)) if resume else 0
    def checkpoint(at,allow_selection=True):
        nonlocal best,latest,generation
        generation+=1;slot=generation%2
        portable=model.export_model()
        # Closing audit may finish after the soft slice; hard science/resource guard still applies.
        tr=_evaluate(portable,train,128,check);va=_evaluate(portable,valid,128,check)
        history.append(dict(updates=at,TRAIN=tr,VALID=va,selection_eligible=allow_selection))
        if allow_selection and (best is None or selection_key(tr,va,at)<tuple(best['key'])):
            best_slot=1-int(Path(best['model']['path']).stem.rsplit('_',1)[1]) if best else 0
            b=_model_save(portable,folder/f'best_{best_slot}.npz',dict(experiment=SCHEMA,TEST_used=False,updates=at))
            best=dict(key=selection_key(tr,va,at),model=b,updates=at,TRAIN=tr,VALID=va)
        latest_model=_model_save(portable,folder/f'latest_{slot}.npz',dict(experiment=SCHEMA,TEST_used=False,updates=at))
        optimizer=_torch_save(dict(model=model.state_dict(),optimizer=opt.state_dict(),generator=gen.get_state(),updates=at,seed=seed,scale=torch.tensor(scale),calibration=model.calibration),folder/f'optimizer_{slot}.pt')
        latest=dict(schema=SCHEMA,family='ICNN',generation=generation,updates=at,history=history,best=best,latest_model=latest_model,optimizer=optimizer,TRAIN_identity=train['dataset_identity'],VALID_identity=valid['dataset_identity'],scale=calibration,latest_TRAIN=tr,latest_VALID=va)
        atomic_json(folder/'CHECKPOINT.json',latest)
        with (folder/'CHECKPOINT_HISTORY.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=time.time(),updates=at,latest_model=latest_model,optimizer=optimizer),allow_nan=False)+'\n')
    start=updates
    while updates<POLICY['icnn_updates'] and time.time()<soft_end:
        check();ix=torch.randint(len(train['u']),(min(128,len(train['u'])),),generator=gen).numpy()
        update_icnn(model,opt,train['u'][ix],train['tau'][ix],train_cost_mean=train['train_cost_mean'])
        updates+=1
        if updates%250==0:checkpoint(updates)
    if latest is None or latest['updates']!=updates:checkpoint(updates,allow_selection=updates%250==0)
    if best is None:
        # An incomplete prefix is reportable but is not a 250-update selection checkpoint.
        chosen=dict(model=latest['latest_model'],updates=updates,TRAIN=latest['latest_TRAIN'],VALID=latest['latest_VALID'])
        selection='FINITE_PARTIAL_PREFIX_NO_SCHEDULED_SELECTION_CHECKPOINT'
    else:chosen=best;selection='TRAIN_QUALIFIED_POOL_IF_ANY_THEN_MIN_FIXED_VALID; EARLIEST_TIE'
    return dict(status='AVAILABLE' if updates==1000 else 'PARTIAL_TIME_LIMIT',model=chosen['model'],TRAIN=chosen['TRAIN'],VALID=chosen['VALID'],actual_updates=updates,selected_updates=chosen['updates'],new_updates=updates-start,target_met=chosen['TRAIN']['WMAPE_percent']<1,selection=selection,checkpoint=binding(folder/'CHECKPOINT.json'),TRAIN_identity=train['dataset_identity'],VALID_identity=valid['dataset_identity'],fresh_initialization=resume is None)

def fit_xgb(train,valid,seed,check,soft_end,folder,zero,evidence,threads=2,resume=None):
    import numpy as np
    from xgboost.callback import TrainingCallback
    from partC_rev1_src.pacnhn_recovery24.ablation import subset_scale
    from partC_rev1_src.pacnhn_v1.models import fit_xgboost_cost,XGBoostCostModel,XGB_CENTERED,log_center_hash,load_model
    from partC_rev1_src.pacnhn_v1.training import _evaluate
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    scale,calibration=subset_scale(train);initial=None;start=0
    if resume:
        if resume['TRAIN_identity']!=train['dataset_identity'] or resume['VALID_identity']!=valid['dataset_identity']:raise ValueError('Resume cohort mismatch')
        initial=load_model(bound(resume['model']));start=int(initial.metadata['boosting_rounds_total'])
        if not 0<start<=32:raise ValueError('Invalid resumable round count')
    center=initial.log_center.copy() if initial is not None else np.zeros(len(scale))
    if initial is None:center[~zero]=np.mean(np.log(train['tau'][:,~zero]),axis=0)
    params=dict(objective='reg:squarederror',tree_method='hist',multi_strategy='one_output_per_tree',n_estimators=32,max_depth=3,learning_rate=.05,random_state=seed,device='cpu',n_jobs=threads,base_score=0.0)
    latest=dict(resume) if resume else {}
    def snapshot(booster):
        rounds=booster.num_boosted_rounds()
        meta=dict(label_target='log_native_cost_minus_frozen_TRAIN_output_center',inverse_transform='exp(frozen_output_center + tree_margin)',training_rows=len(train['u']),all_input_cost_monotonicity=False,parameters=params,structural_zero_evidence=evidence,structural_zero_coordinates=np.flatnonzero(zero).tolist(),boosting_rounds_total=rounds,continued_from_existing_model=initial is not None,log_center_policy=XGB_CENTERED,log_center_train_dataset_identity=train['dataset_identity'],log_center_sha256=log_center_hash(center),log_center_bytes=int(center.nbytes),log_center_frozen_across_TRAIN_growth=True,native_api_dtype='float64',tree_internal_dtype='float32',has_potential=False)
        model=XGBoostCostModel(booster,train['c0'],scale,structural_zero_mask=zero,log_center=center,metadata=meta)
        committed=latest.get('model')
        slot=1-int(Path(committed['path']).stem.rsplit('_',1)[1]) if committed else 0
        b=_model_save(model,folder/f'latest_{slot}.npz',dict(experiment=SCHEMA,TEST_used=False))
        latest.clear()
        latest.update(schema=SCHEMA,family='XGBOOST',model=b,rounds=rounds,TRAIN_identity=train['dataset_identity'],VALID_identity=valid['dataset_identity'],scale=calibration)
        atomic_json(folder/'CHECKPOINT.json',latest)
        with (folder/'CHECKPOINT_HISTORY.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=time.time(),rounds=rounds,model=b))+'\n')
    class DurableBudget(TrainingCallback):
        def before_iteration(self,model,epoch,evals_log):
            check()
            return time.time()>=soft_end
        def after_iteration(self,model,epoch,evals_log):
            snapshot(model)
            check()
            return time.time()>=soft_end
    if start==32:result=initial
    else:
        result=fit_xgboost_cost(train['u'],train['tau'],train['c0'],scale,seed=seed,n_estimators=32-start,max_depth=3,learning_rate=.05,n_jobs=threads,device='cpu',structural_zero_mask=zero,structural_zero_evidence=evidence,log_center_policy=XGB_CENTERED,train_dataset_identity=train['dataset_identity'],initial_model=initial,callbacks=[DurableBudget()])
        # Callback objects are runtime controls, not serializable mathematical parameters.
        result.metadata['parameters']=params
    rounds=int(result.metadata['boosting_rounds_total'])
    if result.booster is not None and not 0<rounds<=32:raise ValueError('No finite completed XGB round')
    tr=_evaluate(result,train,128,check);va=_evaluate(result,valid,128,check)
    committed=latest.get('model') or (resume.get('model') if resume else None)
    final_slot=1-int(Path(committed['path']).stem.rsplit('_',1)[1]) if committed else rounds%2
    b=_model_save(result,folder/f'latest_{final_slot}.npz',dict(experiment=SCHEMA,TEST_used=False))
    latest.update(schema=SCHEMA,family='XGBOOST',model=b,rounds=rounds,TRAIN_identity=train['dataset_identity'],VALID_identity=valid['dataset_identity'],scale=calibration,TRAIN=tr,VALID=va)
    atomic_json(folder/'CHECKPOINT.json',latest)
    return dict(status='AVAILABLE' if rounds==32 or result.booster is None else 'PARTIAL_TIME_LIMIT',model=b,TRAIN=tr,VALID=va,actual_boosting_rounds=rounds,new_rounds=rounds-start,target_met=tr['WMAPE_percent']<1,all_structural_zero=result.booster is None,selection='FIXED_32_ROUND_FRESH_MODEL' if rounds==32 else 'FINITE_COMPLETED_ROUND_PREFIX',checkpoint=binding(folder/'CHECKPOINT.json'),TRAIN_identity=train['dataset_identity'],VALID_identity=valid['dataset_identity'],fresh_initialization=resume is None)

def task_worker(config_path,task,attempt,end):
    sys.dont_write_bytecode=True
    c,plan=config(config_path);root=Path(c['output_root']);folder=root/'fits'/task['task_id'];folder.mkdir(parents=True,exist_ok=True)
    os.environ['NUMBA_CACHE_DIR']=c['runtime_cache'];os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    guard=Guard(c,c['science_deadline_epoch']-120);started=time.time()
    write_new(folder/'attempts'/f'{attempt:03d}'/'INTENT.json',dict(task=task,start_epoch=started,soft_end_epoch=end,config=binding(config_path),TEST_used=False))
    try:
        if any((root.parent/name).exists() for name in ['TEST_OPENED.json','EXPERIMENT_SELECTION_FREEZE.json']) or (root/'FIT_FREEZE.json').exists():raise PermissionError('Selections already frozen')
        entry=next(r for r in plan['regions'] if r['region_key']==task['region_key'])
        tr,va,region,inputs=load_cohorts(c,entry,guard)
        from partC_rev1_src.pacnhn_recovery24.ablation import nested_indices,subset_train
        from partC_rev1_src.pacnhn_v1.training import _structural_zeros
        idx=nested_indices([r['input_identity'] for r in tr['records']],c['seed'],task['region_key'])[task['fraction']]
        subset=subset_train(tr,idx)
        cohort=dict(full_TRAIN_identity=tr['dataset_identity'],TRAIN_identity=subset['dataset_identity'],VALID_identity=va['dataset_identity'],ordered_subset_ids=[r['input_identity'] for r in subset['records']],VALID_ids=[r['input_identity'] for r in va['records']],request_index_sha256=tr['request_index_sha256'],graph_sha256=region.graph_sha256,pair_keys_identity=region.support_identity,training_rows=len(subset['u']),VALID_rows=len(va['u']),source_bindings=inputs)
        if (folder/'COHORT.json').exists():
            prior=read(folder/'COHORT.json')
            for k in cohort:
                if k!='source_bindings' and cohort[k]!=prior[k]:raise ValueError('Attempt changed immutable cohort: '+k)
        else:write_new(folder/'COHORT.json',cohort)
        resume=read(folder/'CHECKPOINT.json') if (folder/'CHECKPOINT.json').exists() else None
        if task['family']=='ICNN':out=fit_icnn(subset,va,c['seed'],guard,end,folder,resume)
        else:out=fit_xgb(subset,va,c['seed'],guard,end,folder,_structural_zeros(region),'FROZEN_GRAPH_ZERO_COST_DIRECTED_REACHABILITY',c['xgb_threads'],resume)
    except Exception as e:
        out=dict(status='TIME_LIMIT' if isinstance(e,TimeoutError) else 'FAILED',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
        if (folder/'CHECKPOINT.json').exists():out['finite_checkpoint_preserved']=binding(folder/'CHECKPOINT.json')
    out.update(task=task,attempt=attempt,started_epoch=started,elapsed_seconds=time.time()-started,config_sha256=digest(config_path),TEST_used=False,peak_RSS_bytes=guard.peak,minimum_available_bytes=guard.minfree)
    p=folder/'attempts'/f'{attempt:03d}'/'RESULT.json';write_new(p,out)
    atomic_json(folder/'LATEST_RESULT.json',out)
    return dict(task=task,status=out['status'],result=binding(p),checkpoint=out.get('checkpoint') or out.get('finite_checkpoint_preserved'))

def order_tasks(tasks,regions):
    pairs={r['region_key']:r['pairs'] for r in regions}
    # Family partition is used by admission. Ordering never uses TEST or observed errors.
    return sorted(tasks,key=lambda t:(t['family']=='XGBOOST',pairs[t['region_key']]>100, t['fraction'],pairs[t['region_key']],t['region_key']))

def run(config_path,admission_path):
    c,plan=config(config_path);admit=read(admission_path)
    if not admit['status'].startswith('PASS') or admit['config']['sha256']!=digest(config_path) or admit['runner']['sha256']!=digest(__file__):raise PermissionError('Root technical admission does not bind config/source')
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=True)
    if any((root.parent/name).exists() for name in ['TEST_OPENED.json','EXPERIMENT_SELECTION_FREEZE.json']):raise PermissionError('No fitting after global selection freeze')
    if (root/'RUN_INTENT.json').exists() or (root/'FIT_FREEZE.json').exists():raise FileExistsError('Explicit new controller attempt required')
    write_new(root/'RUN_INTENT.json',dict(schema=SCHEMA,config=binding(config_path),admission=binding(admission_path),started_epoch=time.time(),tasks=plan['tasks'],checkpoint_policy='Only NEW-run latest/best slots are atomically replaced; immutable attempt receipts and hash history retained. Historical source checkpoints never altered.',TEST_targets_opened=False))
    pending=order_tasks(plan['tasks'],plan['regions']);attempts={t['task_id']:0 for t in pending};outcomes={};active={}
    guard=Guard(c,c['fit_deadline_epoch']);error=None
    with ProcessPoolExecutor(max_workers=c['workers'],mp_context=mp.get_context('spawn')) as pool:
        try:
            while (pending or active) and time.time()<c['fit_deadline_epoch']:
                guard()
                while pending and len(active)<c['workers']:
                    active_x=sum(t['family']=='XGBOOST' for t in active.values())
                    at=next((i for i,t in enumerate(pending) if t['family']!='XGBOOST' or active_x<c['xgb_workers']),None)
                    if at is None:break
                    t=pending.pop(at);attempts[t['task_id']]+=1
                    end=min(time.time()+c['attempt_seconds'],c['fit_deadline_epoch']-90)
                    if end<=time.time():pending.insert(0,t);break
                    active[pool.submit(task_worker,str(config_path),t,attempts[t['task_id']],end)]=t
                if not active:break
                done,_=wait(active,timeout=2,return_when=FIRST_COMPLETED)
                for fut in done:
                    t=active.pop(fut)
                    try:o=fut.result()
                    except Exception as e:o=dict(task=t,status='WORKER_FAILED',error_type=type(e).__name__,error=str(e))
                    outcomes[t['task_id']]=o
                    if o['status'] in {'PARTIAL_TIME_LIMIT','TIME_LIMIT'} and o.get('checkpoint') and attempts[t['task_id']]<c['max_attempts_per_task'] and time.time()+120<c['fit_deadline_epoch']:pending.append(t)
                atomic_json(root/'PROGRESS.json',dict(epoch=time.time(),completed=sum(o['status']=='AVAILABLE' for o in outcomes.values()),terminal=len(outcomes),active=[t['task_id'] for t in active.values()],pending=len(pending),peak_RSS_bytes=guard.peak))
        except Exception as e:
            error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
            write_new(root/'STOP_REQUESTED.json',dict(epoch=time.time(),reason=error))
        # Bounded cleanup after expensive work stops; own worker PIDs only.
        if active:
            _,unfinished=wait(active,timeout=90)
            if unfinished:
                import psutil
                if not (root/'STOP_REQUESTED.json').exists():write_new(root/'STOP_REQUESTED.json',dict(epoch=time.time(),reason='Worker exceeded fit closure allowance'))
                killed=[]
                for worker in list(pool._processes.values()):
                    try:
                        proc=psutil.Process(worker.pid);birth=proc.create_time();children=proc.children(recursive=True)
                        for ch in children:ch.kill()
                        proc.kill();killed.append(dict(pid=worker.pid,birth_epoch=birth))
                    except psutil.NoSuchProcess:pass
                write_new(root/'OWNED_WORKER_TERMINATION.json',dict(epoch=time.time(),owned_workers=killed,reason='No continuation past bounded fit cleanup'))
        for fut,t in list(active.items()):
            try:outcomes[t['task_id']]=fut.result()
            except Exception as e:outcomes[t['task_id']]=dict(task=t,status='WORKER_FAILED',error_type=type(e).__name__,error=str(e))
    for t in plan['tasks']:outcomes.setdefault(t['task_id'],dict(task=t,status='UNATTEMPTED_TIME_LIMIT'))
    rows=[outcomes[t['task_id']] for t in plan['tasks']]
    freeze=dict(schema=SCHEMA,status='ALL_160_DISPOSITIONS_FROZEN',config=binding(config_path),plan=c['plan'],frozen_epoch=time.time(),outcomes=rows,total=160,completed=sum(r['status']=='AVAILABLE' for r in rows),TEST_used=False,error=error,peak_RSS_bytes=guard.peak,minimum_available_bytes=guard.minfree)
    # Every selected artifact is bound freshly; latest rotating slots stop changing here.
    for row in rows:
        folder=root/'fits'/row['task']['task_id']
        if (folder/'CHECKPOINT.json').exists():row['checkpoint']=binding(folder/'CHECKPOINT.json')
        if row.get('checkpoint'):
            checkpoint=read(bound(row['checkpoint']))
            chosen=checkpoint.get('best') or checkpoint
            item=chosen.get('model') or checkpoint.get('latest_model')
            if item:
                row['model']=binding(bound(item))
                row['partial_checkpoint_without_completed_attempt']=row['status']!='AVAILABLE'
    config(config_path)
    write_new(root/'FIT_FREEZE.json',freeze);print(json.dumps(dict(status=freeze['status'],completed=freeze['completed'],total=160)))
    return freeze

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--admission',required=True);a=p.parse_args()
    run(a.config,a.admission)
if __name__=='__main__':main()
