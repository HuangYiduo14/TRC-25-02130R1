"""Versioned recovery labels and prospectively frozen independent holdout inputs.

Original requests, attempt slots and payloads are read-only. New attempts carry
their cumulative ordinal. TEST targets are unavailable before model selection.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import copy,csv,json,math,multiprocessing,time,traceback,zlib
from pathlib import Path
import numpy as np
from ..pacnhn_v1.control import array_hash,read_json,save_npz,sha,write_json
from ..pacnhn_v1.regional import sample_requests
from .context import RecoveryContext
from .regional_solver import solve_label,load_region_graph

_WORKER_CONTEXTS={}

def native_json(v):
    if isinstance(v,dict):return {str(k):native_json(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [native_json(x) for x in v]
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,Path):return str(v)
    return v

def new_json(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    write_json(path,native_json(value))

def require_training_labels_open(ctx):
    for rel in ('training/MODEL_SELECTION_FREEZE.json','training/REGISTRY_FREEZE.json','holdout/TEST_OPENED.json'):
        if (ctx.root/rel).exists():raise PermissionError('TRAIN/VALID labels frozen by '+rel)

def begin_attempt(folder,input_id,role,attempt):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    new_json(folder/'START.json',dict(epoch=time.time(),input_id=input_id,role=role,attempt=attempt,status='ATTEMPT_SLOT_CONSUMED'))

def validate_saved_label(ctx,receipt,label_path,*,region,input_id,role,sample_id,q):
    expected=dict(input_identity=input_id,role=role,sample_id=sample_id,graph_identity=region.identity,
        graph_sha256=region.graph_sha256,support_identity=region.support_identity,fresh_certificate=True)
    if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('Saved label role/input/graph/certificate mismatch')
    if receipt.get('admitted')!=(receipt.get('quality') in {'A','B'}):raise ValueError('Saved label admission mismatch')
    ctx.bind(label_path,'RESUMED_NATIVE_LABEL',receipt['label_sha256'])
    with np.load(label_path,allow_pickle=False) as data:
        u,tau,c0=data['u'].copy(),data['tau'].copy(),data['c0'].copy()
    if (not np.array_equal(u,q) or array_hash({'u':u})!=input_id or not np.allclose(c0,region.c0,rtol=1e-12,atol=1e-12)
            or tau.shape!=q.shape or not np.isfinite(tau).all() or np.any(tau<0)):
        raise ValueError('Saved label native arrays differ')
    return tau

def key(record):return f"{record['suite_id']}:{int(record['region_index']):06d}"

def support_records(ctx):
    path=ctx.source_root/'SUPPORT_FREEZE.json';ctx.bind(path,'IMMUTABLE_SUPPORT')
    support=read_json(path)
    if support.get('status')!='PASS':raise ValueError('Frozen support failed')
    rows=[r for s in support['suites'] for r in s['region_records']]
    if len(rows)!=10442 or {r['suite_id'] for r in rows}!={'s02','s03','s04','s05','s06'}:
        raise ValueError('Exact five-suite active region inventory required')
    return sorted(rows,key=key)

def sampling_scale(region,record):
    baseline=np.maximum(region.baseline_open,region.baseline_i95)
    positive=baseline[baseline>0]
    reference=float(np.median(positive)) if len(positive) else record.get('physical_capacity_reference')
    if reference is None or not math.isfinite(reference) or reference<=0:
        raise ValueError('Positive physical baseline/capacity sampling reference required')
    if not len(positive) and not record.get('physical_capacity_reference_provenance'):
        raise ValueError('Missing physical capacity reference provenance')
    return np.maximum(baseline,.01*reference)

def _old_index(ctx,record):
    root=ctx.source_root/'training/regions'/record['suite_id']/f"{record['region_index']:06d}"/'labels'
    path=root/'REQUEST_INDEX.json'
    if path.exists():
        ctx.bind(path,'SEALED_SOURCE_ROLE_INDEX');return root,read_json(path)
    return root,{}

def freeze_holdout_inputs(ctx):
    ctx.require_action('preflight');ctx.guard()
    if (ctx.root/'training/REGISTRY_FREEZE.json').exists():raise PermissionError('Input freeze must precede selection')
    destination=ctx.root/'holdout/INPUT_FREEZE.json'
    if destination.exists():
        frozen=read_json(destination)
        if frozen['config_sha256']!=ctx.config_sha256:raise ValueError('Holdout inputs belong to another config')
        return destination
    all_rows=support_records(ctx);chosen=[];c=ctx.c['holdout']
    for sid in ctx.c['suite_ids']:
        group=[r for r in all_rows if r['suite_id']==sid]
        if sid=='s03':chosen.extend(group)
        else:
            ordered=sorted(group,key=lambda r:(int(r['pairs']),int(r['region_index'])))
            indices=np.rint(np.linspace(0,len(ordered)-1,c['other_regions_per_suite'])).astype(int)
            if len(set(indices))!=c['other_regions_per_suite']:raise ValueError('Invalid unique holdout stratum indices')
            chosen.extend(ordered[int(i)] for i in indices)
    entries=[]
    for at,record in enumerate(chosen):
        ctx.guard();ctx.bind(record['path'],'FROZEN_REGIONAL_GRAPH',record['sha256'])
        region=load_region_graph(record['path'],expected_sha256=record['sha256'])
        _,old_index=_old_index(ctx,record)
        count=c['s03_inputs_per_region'] if record['suite_id']=='s03' else c['other_inputs_per_region']
        requests=sample_requests(region,role='TEST',count=count,seed=c['seed'],existing_input_ids=set(old_index),reference_scale=sampling_scale(region,record))
        folder=ctx.root/'holdout/inputs'/record['suite_id']/f"{record['region_index']:06d}"
        data_path=folder/'INPUTS.npz';meta_path=folder/'INPUTS.json'
        if meta_path.exists():
            prior=read_json(meta_path)
            if prior['input_ids']!=[r['input_id'] for r in requests] or sha(data_path)!=prior['data_sha256']:
                raise ValueError('Previously saved holdout input prefix changed')
        else:
            if data_path.exists():raise FileExistsError('Unsealed input payload needs explicit recovery: '+str(data_path))
            save_npz(data_path,u=np.stack([r['u'] for r in requests]))
            new_json(meta_path,dict(region_key=key(record),role='TEST',seed=c['seed'],requested_inputs=count,
                data_path=str(data_path),data_sha256=sha(data_path),input_ids=[r['input_id'] for r in requests],
                requests=[{k:v for k,v in r.items() if k!='u'} for r in requests],
                graph_identity=region.identity,support_identity=region.support_identity,graph_sha256=record['sha256'],
                disjoint_from_all_source_roles=True,source_index_sha256=sha(_old_index_path(ctx,record)) if old_index else None))
        entries.append(dict(record=record,region_key=key(record),requested_inputs=count,metadata_path=str(meta_path),metadata_sha256=sha(meta_path)))
        if at%50==0:write_json(ctx.root/'holdout/INPUT_FREEZE_PROGRESS.json',dict(completed=len(entries),expected=len(chosen),epoch=time.time()))
    counts={sid:sum(r['requested_inputs'] for r in entries if r['record']['suite_id']==sid) for sid in ctx.c['suite_ids']}
    if counts['s03']!=71952 or sum(counts.values())!=72976:raise ValueError('Approved holdout request counts differ')
    new_json(destination,dict(schema='RECOVERY_HOLDOUT_INPUT_FREEZE_V1',status='FROZEN_BEFORE_FITTING',created_epoch=time.time(),
        config_sha256=ctx.config_sha256,seed=c['seed'],rows=entries,counts=counts,requested_inputs=sum(counts.values()),targets_opened=False))
    ctx.flush_bindings();return destination

def _old_index_path(ctx,record):
    return ctx.source_root/'training/regions'/record['suite_id']/f"{record['region_index']:06d}"/'labels/REQUEST_INDEX.json'

def gap_progress(record):
    """Use measured certificate times, never iteration count as elapsed time."""
    history=record.get('history',[]);duration=float(record['complete_label_elapsed_seconds'])
    before=[float(r['certificate']['RG_SPTT']) for r in history if float(r.get('elapsed_seconds',math.inf))<=duration/2
            and r['certificate'].get('RG_SPTT') is not None]
    after=[float(r['certificate']['RG_SPTT']) for r in history if float(r.get('elapsed_seconds',-1))>duration/2
           and r['certificate'].get('RG_SPTT') is not None]
    if not before or not after or min(before)<=0:return None
    return (min(before)-min(after))/min(before)

def _save_attempt(folder,result,extra):
    folder=Path(folder)
    started=read_json(folder/'START.json')
    if (started['input_id']!=result['record']['input_identity'] or started['role']!=extra['role']
            or started['attempt']!=extra['attempt'] or any(p.name!='START.json' for p in folder.iterdir())):
        raise ValueError('Unowned or previously written attempt directory')
    receipt=copy.deepcopy(result['record']);history=receipt.pop('history')
    payload={k:v for k,v in result.items() if k!='record'}
    encoded=json.dumps(native_json(history),separators=(',',':'),allow_nan=False).encode('utf-8')
    payload['history_json_zlib']=np.frombuffer(zlib.compress(encoded,6),dtype=np.uint8)
    label=folder/'label.npz';save_npz(label,**payload)
    receipt.update(extra,label_sha256=sha(label),history_storage='label.npz:history_json_zlib',history_rows=len(history),
        latter_half_best_gap_relative_improvement=gap_progress(result['record']))
    path=folder/'RESULT.json';new_json(path,receipt)
    return dict(path=str(label),sha256=receipt['label_sha256'],receipt=str(path),receipt_sha256=sha(path))

def _retry_region(config,record,tasks,deadline):
    ctx=RecoveryContext(config,step_id='labels');ctx.require_action('labels')
    require_training_labels_open(ctx)
    check=lambda:ctx.guard(deadline=deadline)
    source,index=_old_index(ctx,record);identity_path=source/'REGION_IDENTITY.json'
    ctx.bind(identity_path,'SOURCE_REGIONAL_LABEL_IDENTITY');identity=read_json(identity_path)
    ctx.bind(record['path'],'REGIONAL_LABEL_GRAPH',record['sha256']);region=load_region_graph(record['path'],expected_sha256=record['sha256'])
    folder=ctx.root/'labels/overlays'/record['suite_id']/f"{record['region_index']:06d}"
    if (folder/'RESULT.json').exists():return read_json(folder/'RESULT.json')
    c=ctx.c['label_recovery'];dispositions=[]
    for item in tasks:
        input_id=item['input_id'];row=index[input_id]
        if row['role'] not in {'TRAIN','VALID'} or row['selected_label'] is not None:raise ValueError('Invalid recovery retry target')
        ctx.bind(row['input_file'],'FROZEN_RECOVERY_INPUT',row['input_file_sha256'])
        ctx.bind(row['request_metadata'],'FROZEN_RECOVERY_REQUEST',row['request_metadata_sha256'])
        with np.load(row['input_file'],allow_pickle=False) as data:q=data['u'].copy()
        if array_hash({'u':q})!=input_id:raise ValueError('Input identity changed')
        historical=list((source/'labels'/row['role']/row['sample_id']).glob('attempt_*'))
        if len(historical)!=int(item['attempt_slots']):raise ValueError('Cumulative historical attempt slots changed')
        attempts=[];previous=None
        for ordinal,seconds in enumerate(c['attempt_seconds']):
            if ordinal and (previous is None or previous['status']!='TIME_LIMIT' or gap_progress(previous) is None
                            or gap_progress(previous)<c['best_gap_relative_improvement']):break
            if time.time()+seconds+2>deadline:break
            check();attempt=len(historical)+ordinal+1
            attempt_folder=folder/'labels'/row['role']/row['sample_id']/f'attempt_{attempt:06d}'
            if attempt_folder.exists():raise ValueError('Interrupted recovery attempt requires explicit resume disposition')
            try:
                require_training_labels_open(ctx);begin_attempt(attempt_folder,input_id,row['role'],attempt)
                result=solve_label(region,q,max_seconds=seconds,max_iterations=c['iterations'],relative_gap_tolerance=c['grade_A'],
                    deadline_epoch=deadline,guard=check,method=c['method'],cfw_reset_on_bound=c['cfw_reset_on_bound'],iterate_selection=c['iterate_selection'])
                previous=result['record']
                selected=_save_attempt(attempt_folder,result,dict(sample_id=row['sample_id'],role=row['role'],attempt=attempt,
                    input_file_sha256=row['input_file_sha256'],source_attempt_slots=len(historical),recovery_attempt=ordinal+1,
                    phase_id=ctx.c['phase_id'],run_id=ctx.c['run_id'],source_index_sha256=sha(source/'REQUEST_INDEX.json')))
                attempts.append(dict(attempt=attempt,admitted=previous['admitted'],quality=previous['quality'],result=selected))
                if previous['admitted']:row['selected_label']=selected;break
            except (TimeoutError,MemoryError,OSError) as exc:
                if not (attempt_folder/'FAILED.json').exists():new_json(attempt_folder/'FAILED.json',dict(status='RUNTIME_FAILURE',error_type=type(exc).__name__,error=str(exc),attempt=attempt))
                attempts.append(dict(attempt=attempt,status='RUNTIME_FAILURE',error=str(exc)));break
        dispositions.append(dict(input_id=input_id,role=row['role'],attempts=attempts,admitted=row['selected_label'] is not None,
            status='ADMITTED' if row['selected_label'] is not None else 'LABEL_UNRESOLVED'))
    new_json(folder/'REGION_IDENTITY.json',identity);new_json(folder/'REQUEST_INDEX.json',index)
    receipt=dict(region_key=key(record),status='SEALED_DATA_OVERLAY',path=str(folder),sha256=sha(folder/'REQUEST_INDEX.json'),
        source_index_sha256=sha(source/'REQUEST_INDEX.json'),requested=len(tasks),admitted=sum(r['admitted'] for r in dispositions),dispositions=dispositions)
    new_json(folder/'RESULT.json',receipt);ctx.flush_bindings();return receipt

def _parallel(ctx,worker,jobs,deadline,max_workers=4):
    if type(max_workers) is not int or not 1<=max_workers<=4:
        raise ValueError('One to four regional workers required')
    results=[];jobs=list(jobs);running={};position=0
    pool=ProcessPoolExecutor(max_workers=max_workers,mp_context=multiprocessing.get_context('spawn'))
    def job_key(job):
        record=job[1]
        return record.get('region_key') or key(record.get('record',record))
    try:
        exhausted=False
        while running or not exhausted:
            # Workers sample their own RSS; only this parent sees the aggregate
            # of every worker. Do not dispatch after a resource/deadline breach.
            if time.time()<deadline:ctx.guard(deadline=deadline)
            while len(running)<max_workers and not exhausted:
                if position==len(jobs):exhausted=True;break
                job=jobs[position];position+=1
                if time.time()>=deadline:results.append(dict(status='UNATTEMPTED_TIME_LIMIT',job_key=job_key(job)));continue
                running[pool.submit(worker,*job,deadline)]=job
            if not running:continue
            done,_=wait(running,timeout=.5,return_when=FIRST_COMPLETED)
            # Workers own their exact deadlines; parent continues collecting their
            # terminal receipts instead of abandoning partial on-disk products.
            for future in done:
                job=running.pop(future)
                try:results.append(future.result())
                except Exception as exc:results.append(dict(status='WORKER_FAILURE',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),job=job[1]))
            write_json(ctx.root/'progress'/f'{ctx.action}.json',dict(completed=len(results),active=len(running),epoch=time.time()))
        pool.shutdown(wait=True);pool=None
    except (TimeoutError,MemoryError,OSError) as exc:
        results.extend(dict(status='INTERRUPTED_RESOURCE_LIMIT',job_key=job_key(job),error_type=type(exc).__name__,error=str(exc)) for job in running.values())
        results.extend(dict(status='UNATTEMPTED_RESOURCE_LIMIT',job_key=job_key(job),error_type=type(exc).__name__,error=str(exc)) for job in jobs[position:])
    finally:
        if pool is not None:
            from ..pacnhn_v1.training import _stop_own_pool
            _stop_own_pool(pool)
    return results

def run_missing_retries(ctx,*,deadline,max_workers=4):
    ctx.require_action('labels');require_training_labels_open(ctx)
    clock=ctx.root/'labels/RETRY_CLOCK.json'
    if not clock.exists():new_json(clock,dict(start_epoch=time.time(),deadline_epoch=min(deadline,ctx.deadline(),time.time()+ctx.c['label_recovery']['max_elapsed_seconds'])))
    deadline=min(deadline,ctx.deadline(),read_json(clock)['deadline_epoch'])
    docket=Path(ctx.c['diagnosis_root'])/'failure_diagnosis/MISSING_INITIAL_INPUTS.csv'
    ctx.bind(docket,'PREDECLARED_MISSING_INPUT_DOCKET');regions={key(r):r for r in support_records(ctx)};groups={}
    with docket.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['role'] in {'TRAIN','VALID'}:groups.setdefault(f"s03:{int(row['region_index']):06d}",[]).append(row)
    jobs=[(str(ctx.path),regions[k],sorted(v,key=lambda r:(r['any_full_20s_BEST_CFW_attempt']=='True',r['role'],r['input_id']))) for k,v in sorted(groups.items())]
    if sum(len(v) for v in groups.values())!=232:raise ValueError('Missing TRAIN/VALID docket changed')
    results=_parallel(ctx,_retry_region,jobs,deadline,max_workers)
    report=dict(status='RETRY_STAGE_FINISHED',results=results,overlays={r['region_key']:{'path':r['path'],'sha256':r['sha256']} for r in results if r.get('status')=='SEALED_DATA_OVERLAY'})
    new_json(ctx.root/'labels/RETRY_RESULT.json',report);ctx.flush_bindings();return report

def prepare_initial_training(ctx,record,*,deadline):
    """Bounded first-fit labels for a previously untrained other-suite region."""
    ctx.require_action('training');ctx.guard(deadline=deadline)
    require_training_labels_open(ctx)
    if record['suite_id']=='s03' or _old_index_path(ctx,record).exists():
        raise ValueError('Initial generation is only for genuinely untrained additional-suite regions')
    if (ctx.root/'training/MODEL_SELECTION_FREEZE.json').exists() or (ctx.root/'holdout/TEST_OPENED.json').exists():
        raise PermissionError('Cannot generate training labels after model selection')
    folder=ctx.root/'labels/initial'/record['suite_id']/f"{record['region_index']:06d}"
    if (folder/'RESULT.json').exists():return read_json(folder/'RESULT.json')
    ctx.bind(record['path'],'INITIAL_TRAINING_GRAPH',record['sha256'])
    region=load_region_graph(record['path'],expected_sha256=record['sha256']);scale=sampling_scale(region,record)
    holdout=read_json(ctx.root/'holdout/INPUT_FREEZE.json');existing=set()
    for item in holdout['rows']:
        if item['region_key']==key(record):
            ctx.bind(item['metadata_path'],'INDEPENDENT_TEST_INPUT_IDS',item['metadata_sha256'])
            existing.update(read_json(item['metadata_path'])['input_ids'])
    requests=[];c=ctx.c['training_config']
    for role,count in (('TRAIN',c['initial_train']),('VALID',c['fixed_valid'])):
        part=sample_requests(region,role=role,count=count,seed=ctx.c['recovery']['seed'],existing_input_ids=existing,reference_scale=scale)
        requests.extend(part);existing.update(r['input_id'] for r in part)
    index={}
    for request in requests:
        directory=folder/'requests'/request['role']/request['sample_id'];data=directory/'input.npz';meta=directory/'REQUEST.json'
        if data.exists() or meta.exists():raise FileExistsError('Incomplete initial request freeze must be preserved and diagnosed')
        save_npz(data,u=request['u']);metadata={k:v for k,v in request.items() if k!='u'};metadata['input_file_sha256']=sha(data)
        new_json(meta,metadata)
        index[request['input_id']]=dict(sample_id=request['sample_id'],role=request['role'],input_file=str(data),input_file_sha256=sha(data),
            request_metadata=str(meta),request_metadata_sha256=sha(meta),selected_label=None)
    identity=dict(phase_id=ctx.c['phase_id'],run_id=ctx.c['run_id'],region_id=key(record),graph_identity=region.identity,
                  graph_sha256=region.graph_sha256,support_identity=region.support_identity)
    new_json(folder/'REGION_IDENTITY.json',identity)
    new_json(folder/'INITIAL_ROLE_FREEZE.json',dict(epoch=time.time(),rows=index,TEST_targets_opened=False,config_sha256=ctx.config_sha256))
    # Interleave roles so a short suite allocation cannot consume all of VALID's
    # opportunity on TRAIN labels. Inputs and roles remain fixed before solving.
    byrole={role:[r for r in requests if r['role']==role] for role in ('TRAIN','VALID')}
    ordered=[]
    for at in range(max(map(len,byrole.values()))):
        for role in ('TRAIN','VALID'):
            if at<len(byrole[role]):ordered.append(byrole[role][at])
    dispositions=[]
    for request in ordered:
        if time.time()+.1>=deadline:
            dispositions.append(dict(input_id=request['input_id'],role=request['role'],status='UNATTEMPTED_TIME_LIMIT'));continue
        row=index[request['input_id']]
        try:
            attempt_folder=folder/'labels'/request['role']/request['sample_id']/'attempt_000001'
            require_training_labels_open(ctx);begin_attempt(attempt_folder,request['input_id'],request['role'],1)
            result=solve_label(region,request['u'],max_seconds=c['per_label_seconds'],max_iterations=c['label_retry_max_iterations'],
                relative_gap_tolerance=c['label_target_relative_gap'],deadline_epoch=deadline,guard=lambda:ctx.guard(deadline=deadline),
                method=c['label_retry_solver_method'],cfw_reset_on_bound=c['label_retry_cfw_reset_on_bound'],iterate_selection=c['label_retry_iterate_selection'])
            selected=_save_attempt(attempt_folder,result,
                dict(sample_id=request['sample_id'],role=request['role'],attempt=1,input_file_sha256=row['input_file_sha256'],
                     phase_id=ctx.c['phase_id'],run_id=ctx.c['run_id']))
            if result['record']['admitted']:row['selected_label']=selected
            dispositions.append(dict(input_id=request['input_id'],role=request['role'],status=result['record']['status'],admitted=result['record']['admitted']))
        except (TimeoutError,MemoryError,OSError) as exc:
            dispositions.append(dict(input_id=request['input_id'],role=request['role'],status='RUNTIME_LIMIT',error=str(exc),admitted=False))
    new_json(folder/'REQUEST_INDEX.json',index)
    counts={role:sum(row['role']==role and row['selected_label'] is not None for row in index.values()) for role in ('TRAIN','VALID')}
    result=dict(region_key=key(record),path=str(folder),sha256=sha(folder/'REQUEST_INDEX.json'),counts=counts,dispositions=dispositions,
        status='ADMITTED_NONEMPTY_TRAIN_VALID' if min(counts.values())>0 else 'INSUFFICIENT_LABEL_COVERAGE')
    new_json(folder/'RESULT.json',result);ctx.flush_bindings();return result

def _holdout_context(config):
    """Reuse authority checks without retaining earlier regions' audit bindings."""
    cache_key=(str(config),'holdout')
    if cache_key not in _WORKER_CONTEXTS:_WORKER_CONTEXTS[cache_key]=RecoveryContext(config,step_id='holdout')
    ctx=_WORKER_CONTEXTS[cache_key]
    if hasattr(ctx,'_holdout_shared_bindings'):
        ctx.bindings=dict(ctx._holdout_shared_bindings)
    ctx.require_action('holdout')
    opened_path=ctx.root/'holdout/TEST_OPENED.json'
    ctx.bind(opened_path,'FROZEN_TEST_OPENING')
    opened=read_json(opened_path)
    freeze,_=require_selection_freeze(ctx,opened['selection_freeze_path'],opened['selection_freeze_sha256'])
    if opened['epoch']<freeze['created_epoch']:raise PermissionError('TEST opening precedes selection')
    if not hasattr(ctx,'_holdout_shared_bindings'):
        ctx._holdout_shared_bindings=dict(ctx.bindings)
    return ctx


def _holdout_region(config,entry,deadline):
    ctx=_holdout_context(config)
    record=entry['record'];ctx.bind(entry['metadata_path'],'FROZEN_TEST_INPUT_METADATA',entry['metadata_sha256'])
    metadata=read_json(entry['metadata_path']);ctx.bind(metadata['data_path'],'FROZEN_TEST_INPUTS',metadata['data_sha256'])
    with np.load(metadata['data_path'],allow_pickle=False) as data:inputs=data['u'].copy()
    ctx.bind(record['path'],'FROZEN_TEST_GRAPH',record['sha256']);region=load_region_graph(record['path'],expected_sha256=record['sha256'])
    folder=ctx.root/'holdout/results'/record['suite_id']/f"{record['region_index']:06d}"
    if (folder/'RESULT.json').exists():return read_json(folder/'RESULT.json')
    c=ctx.c['holdout'];rows=[];admitted_u=[];admitted_tau=[];admitted_ids=[];dataset_c0=None
    for index,(q,request) in enumerate(zip(inputs,metadata['requests'])):
        if array_hash({'u':q})!=request['input_id']:raise ValueError('Frozen holdout input changed')
        attempt_folder=folder/'attempts'/request['sample_id']
        if time.time()+.05>=deadline:
            rows.append(dict(input_id=request['input_id'],status='UNATTEMPTED_TIME_LIMIT',admitted=False));continue
        if (attempt_folder/'RESULT.json').exists():
            receipt=read_json(attempt_folder/'RESULT.json');label_path=attempt_folder/'label.npz'
            tau=validate_saved_label(ctx,receipt,label_path,region=region,input_id=request['input_id'],role='TEST',sample_id=request['sample_id'],q=q)
            with np.load(label_path,allow_pickle=False) as data:current_c0=data['c0'].copy()
            selected=dict(path=str(label_path),sha256=receipt['label_sha256'],receipt=str(attempt_folder/'RESULT.json'),receipt_sha256=sha(attempt_folder/'RESULT.json'))
        elif attempt_folder.exists():
            rows.append(dict(input_id=request['input_id'],status='INTERRUPTED_ATTEMPT_SLOT',admitted=False));continue
        else:
            try:
                begin_attempt(attempt_folder,request['input_id'],'TEST',1)
                result=solve_label(region,q,max_seconds=c['per_label_seconds'],max_iterations=c['max_iterations'],relative_gap_tolerance=1e-6,
                    deadline_epoch=deadline,guard=lambda:ctx.guard(deadline=deadline),method=c['solver'],
                    cfw_reset_on_bound=c['cfw_reset_on_bound'],iterate_selection=c['iterate_selection'])
                receipt=result['record'];tau=result['tau'];current_c0=result['c0']
                selected=_save_attempt(attempt_folder,result,dict(sample_id=request['sample_id'],role='TEST',attempt=1,
                    input_file_sha256=metadata['data_sha256'],phase_id=ctx.c['phase_id'],run_id=ctx.c['run_id']))
            except (TimeoutError,MemoryError,OSError) as exc:
                rows.append(dict(input_id=request['input_id'],status='RUNTIME_LIMIT',admitted=False,error=str(exc)));continue
        rows.append(dict(input_id=request['input_id'],status=receipt['status'],quality=receipt['quality'],admitted=receipt['admitted'],selected_label=selected))
        if receipt['admitted']:
            if dataset_c0 is not None and not np.array_equal(dataset_c0,current_c0):raise ValueError('Heldout exact native c0 differs across labels')
            dataset_c0=current_c0.copy();admitted_u.append(q);admitted_tau.append(tau);admitted_ids.append(request['input_id'])
    dataset_path=folder/'DATASET.npz'
    u=np.stack(admitted_u) if admitted_u else np.empty((0,region.m));tau=np.stack(admitted_tau) if admitted_tau else np.empty((0,region.m))
    if dataset_path.exists():raise FileExistsError('Unsealed holdout aggregate already exists')
    if dataset_c0 is None:dataset_c0=region.c0.copy()
    save_npz(dataset_path,u=u,tau=tau,c0=dataset_c0)
    report=dict(region_key=key(record),role='TEST',status='COMPLETE' if len(admitted_u)==len(inputs) else 'PARTIAL_LABEL_COVERAGE',
        requested_inputs=len(inputs),admitted_inputs=len(admitted_u),input_ids=admitted_ids,dispositions=rows,
        dataset_path=str(dataset_path),dataset_sha256=sha(dataset_path),dataset_identity=array_hash({'u':u,'tau':tau,'c0':dataset_c0}),
        input_metadata_sha256=entry['metadata_sha256'],identity=dict(graph_identity=region.identity,support_identity=region.support_identity,graph_sha256=region.graph_sha256))
    new_json(folder/'RESULT.json',report);ctx.flush_bindings();return report

def require_selection_freeze(ctx,freeze_path,freeze_sha256):
    path=Path(freeze_path).resolve(strict=True)
    if not path.is_relative_to(ctx.root):raise PermissionError('Selection freeze must belong to recovery run')
    cached=getattr(ctx,'_selection_freeze_verified',None)
    if cached and cached['path']==str(path) and cached['sha256']==freeze_sha256:
        if all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==stamp for p,stamp in cached['stats'].items()):
            return cached['freeze'],cached['inputs']
    ctx.bind(path,'GLOBAL_MODEL_SELECTION_FREEZE',freeze_sha256);freeze=read_json(path)
    if freeze.get('status')!='FROZEN' or freeze.get('test_targets_opened') is not False:raise PermissionError('Valid pre-TEST selection freeze required')
    if (freeze.get('schema')!='RECOVERY_GLOBAL_MODEL_SELECTION_FREEZE_V1' or freeze.get('phase_id')!=ctx.c['phase_id']
            or freeze.get('run_id')!=ctx.c['run_id']):raise PermissionError('Foreign global selection freeze')
    ctx.bind(freeze['config_path'],'SELECTION_EXECUTION_CONFIGURATION',freeze['config_sha256'])
    if Path(freeze['config_path']).resolve()!=ctx.path or freeze['config_sha256']!=ctx.config_sha256:
        raise PermissionError('Selection freeze and runtime configurations differ')
    for field,relative in (('model_records','training/NORMALIZED_RECORDS.json'),('portfolio','portfolios/PORTFOLIOS.json')):
        binding=freeze[field]
        if Path(binding['path']).resolve()!=(ctx.root/relative).resolve():raise ValueError('Foreign frozen '+field)
        ctx.bind(binding['path'],'FROZEN_'+field.upper(),binding['sha256'])
    portfolio=read_json(freeze['portfolio']['path'])
    if (portfolio.get('records_binding')!=freeze['model_records'] or portfolio.get('phase_id')!=ctx.c['phase_id']
            or portfolio.get('run_id')!=ctx.c['run_id']):raise ValueError('Portfolio does not bind frozen model selections')
    ablation=freeze['ablation_fit_freeze'];ctx.bind(ablation['path'],'ABLATION_FIT_FREEZE',ablation['sha256'])
    if Path(ablation['path']).resolve()!=(ctx.root/'ablation/FIT_FREEZE.json').resolve():raise ValueError('Foreign ablation fit freeze')
    ablation_record=read_json(ablation['path'])
    if (len(ablation_record.get('outcomes',[]))!=160 or ablation_record.get('frozen_epoch',math.inf)>freeze['created_epoch']
            or ablation_record.get('TEST_used') is not False):raise ValueError('Incomplete ablation selection dispositions')
    input_binding=freeze['holdout_input_freeze'];ctx.bind(input_binding['path'],'BLIND_INPUT_FREEZE',input_binding['sha256'])
    inputs=read_json(input_binding['path'])
    if inputs['created_epoch']>freeze['created_epoch']:raise ValueError('Input freeze chronology invalid')
    files=[str(path),freeze['config_path'],freeze['model_records']['path'],freeze['portfolio']['path'],ablation['path'],input_binding['path']]
    ctx._selection_freeze_verified={'path':str(path),'sha256':freeze_sha256,'freeze':freeze,'inputs':inputs,
        'stats':{p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in files}}
    return freeze,inputs

def run_holdouts(ctx,*,freeze_path,freeze_sha256,deadline,max_workers=4):
    ctx.require_action('holdout');freeze,inputs=require_selection_freeze(ctx,freeze_path,freeze_sha256)
    opened=ctx.root/'holdout/TEST_OPENED.json'
    if not opened.exists():new_json(opened,dict(epoch=time.time(),selection_freeze_path=str(freeze_path),selection_freeze_sha256=freeze_sha256))
    else:
        if read_json(opened)['selection_freeze_sha256']!=freeze_sha256:raise PermissionError('TEST was opened against another model freeze')
    jobs=[(str(ctx.path),entry) for entry in inputs['rows']]
    results=_parallel(ctx,_holdout_region,jobs,min(deadline,ctx.deadline()),max_workers)
    report=dict(status='HOLDOUT_STAGE_FINISHED',selection_freeze_sha256=freeze_sha256,requested_inputs=inputs['requested_inputs'],
        admitted_inputs=sum(r.get('admitted_inputs',0) for r in results),results=results,targets_used_for_selection=False)
    new_json(ctx.root/'holdout/RESULT.json',report);ctx.flush_bindings();return report

def load_holdout_dataset(ctx,region_key,*,freeze_path,freeze_sha256):
    freeze,inputs=require_selection_freeze(ctx,freeze_path,freeze_sha256)
    opened=read_json(ctx.root/'holdout/TEST_OPENED.json')
    if opened['selection_freeze_sha256']!=freeze_sha256 or opened['epoch']<freeze['created_epoch']:raise PermissionError('Invalid TEST opening chronology')
    sid,rid=region_key.split(':');path=ctx.root/'holdout/results'/sid/rid/'RESULT.json';ctx.bind(path,'HELDOUT_LABEL_DISPOSITION')
    report=read_json(path);entries={r['region_key']:r for r in inputs['rows']}
    if report['input_metadata_sha256']!=entries[region_key]['metadata_sha256']:raise ValueError('Heldout population mismatch')
    ctx.bind(report['dataset_path'],'ADMITTED_HELDOUT_DATASET',report['dataset_sha256'])
    with np.load(report['dataset_path'],allow_pickle=False) as data:arrays={k:data[k].copy() for k in ('u','tau','c0')}
    if array_hash(arrays)!=report['dataset_identity'] or len(arrays['u'])!=report['admitted_inputs']:raise ValueError('Heldout dataset identity mismatch')
    return dict(**arrays,dataset_identity=report['dataset_identity'],role='TEST',requested_inputs=report['requested_inputs'],
        admitted_inputs=report['admitted_inputs'],input_ids=report['input_ids'],identity=report['identity'],source_manifest=str(path))
