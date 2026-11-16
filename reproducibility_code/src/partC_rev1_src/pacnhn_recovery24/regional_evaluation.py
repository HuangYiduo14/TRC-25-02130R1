"""Post-freeze regional comparisons on identical cohorts; never fit or select.

Baseline means the original checkpoint selection, evaluated on the SAME current
cohort as the recovered and frozen portfolio versions. Old opened TEST labels are
never read. Missing/unrequested/undefined rows remain separate from metric sums.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import csv
import math
from pathlib import Path
import time
import zipfile

import numpy as np

from ..pacnhn_v1.control import array_hash, read_json, sha
from ..pacnhn_v1.models import load_model
from .labels import new_json, require_selection_freeze, load_holdout_dataset, support_records, key
from .training_recovery import load_training_cohorts, _bound

SCHEMA='RECOVERY_REGIONAL_EVALUATION_V1'
FAMILIES=('FFTT','POWER4','ICNN','XGBOOST')
VERSIONS=('baseline','recovered','portfolio')
ROLES=('TRAIN','VALID','TEST')
FIELDS=('suite_id','region_index','family','version','role','status','WMAPE_percent',
    'absolute_error_sum','true_cost_sum','coordinates','admitted_inputs','requested_inputs',
    'dataset_identity','model_sha256','component_identity','target_met','train_target_met',
    'fallback_component','fallback_reason','property_status','model_source','error_type','error',
    'inference_seconds','metric_evidence','historical_inference_seconds','neural_amplitude',
    'neural_amplitude_is_zero','neural_activity_scope','TEST_used_for_selection')


class _FFTT:
    def __init__(self,c0):self.c0=np.asarray(c0,dtype=np.float64).copy()
    def cost(self,u):return np.broadcast_to(self.c0,np.asarray(u).shape).copy()


def native_metrics(model,dataset,*,check,batch=128):
    """Full native coordinate sums; a zero truth denominator remains undefined."""
    u,tau,c0=(np.asarray(dataset[k],dtype=np.float64) for k in ('u','tau','c0'))
    if (u.ndim!=2 or tau.shape!=u.shape or c0.shape!=(u.shape[1],)
        or not all(np.isfinite(a).all() and not (a<0).any() for a in (u,tau,c0))
        or not np.array_equal(np.asarray(model.c0),c0)):
        raise ValueError('Model/cohort native shape, finiteness or exact c0 mismatch')
    if type(batch) is not int or batch<1:raise ValueError('Positive evaluation batch required')
    if not len(u):
        return dict(status='NO_ADMITTED_LABELS',WMAPE_percent=None,absolute_error_sum=0.,
                    true_cost_sum=0.,coordinates=0,admitted_inputs=0,inference_seconds=0.)
    error=truth=seconds=0.;coordinates=0
    for start in range(0,len(u),batch):
        check();tick=time.perf_counter();pred=np.asarray(model.cost(u[start:start+batch]))
        seconds+=time.perf_counter()-tick;actual=tau[start:start+batch]
        if pred.shape!=actual.shape or not np.isfinite(pred).all() or (pred<0).any():
            raise ArithmeticError('Invalid native predictions; no clipping or fallback')
        error+=float(np.abs(pred-actual).sum());truth+=float(actual.sum());coordinates+=actual.size
    if not math.isfinite(error) or not math.isfinite(truth):raise ArithmeticError('Native metric sums overflowed')
    wmape=None if truth==0 else 100.*error/truth
    if wmape is not None and not math.isfinite(wmape):raise ArithmeticError('Native WMAPE overflowed')
    check()
    return dict(status='MEASURED' if wmape is not None else 'UNDEFINED_ZERO_TRUTH',
                WMAPE_percent=wmape,absolute_error_sum=error,true_cost_sum=truth,
                coordinates=int(coordinates),admitted_inputs=len(u),inference_seconds=seconds)


def _sha(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Exact frozen model hash required')
    return value


def descriptor(candidate,*,family,source):
    if not candidate:return None
    return dict(model_path=candidate['model_path'],model_sha256=_sha(candidate['model_sha256']),
        family=family,source=source,property_status=candidate.get('property_status','NOT_ESTABLISHED'),
        graph_sha256=candidate.get('graph_sha256'),pair_keys_identity=candidate.get('pair_keys_identity'),
        saved_audit=copy.deepcopy(candidate.get('audit') or
            dict(train_dataset_identity=candidate.get('train_dataset_identity'),
                 valid_dataset_identity=candidate.get('valid_dataset_identity'),
                 model_sha256=candidate.get('model_sha256'),TRAIN=candidate.get('TRAIN'),VALID=candidate.get('VALID'))))


def reusable_metric(entry,role,current_identity):
    """Only exact model/current dataset-bound complete native sums may be reused."""
    if entry is None or role not in ('TRAIN','VALID') or not current_identity:return None
    audit=entry.get('saved_audit') or {};metric=audit.get(role)
    if (audit.get('model_sha256')!=entry['model_sha256']
        or audit.get(role.lower()+'_dataset_identity')!=current_identity or not isinstance(metric,dict)):
        return None
    error,truth=metric.get('absolute_error_sum'),metric.get('true_cost_sum')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (error,truth)):
        return None
    count,rows=metric.get('coordinates'),metric.get('unique_inputs')
    if type(count) is not int or type(rows) is not int or count<=0 or rows<=0:return None
    value=None if truth==0 else 100.*error/truth
    recorded=metric.get('WMAPE_percent')
    if value is None:
        if recorded is not None:return None
    elif (type(recorded) not in (int,float) or not math.isfinite(recorded)
          or not math.isfinite(value) or not math.isclose(value,recorded,rel_tol=1e-12,abs_tol=1e-12)):
        return None
    return dict(status='MEASURED' if value is not None else 'UNDEFINED_ZERO_TRUTH',
        WMAPE_percent=value,absolute_error_sum=error,true_cost_sum=truth,coordinates=count,
        admitted_inputs=rows,dataset_identity=current_identity,inference_seconds=0.,
        historical_inference_seconds=metric.get('batch_inference_seconds'),
        metric_evidence='REUSED_HASH_AND_CURRENT_DATASET_BOUND_NATIVE_AUDIT')


def neural_metadata(entry,model=None):
    """Expose native Gamma without subtracting full and Power4 predictions.

    Positive Gamma is structural activity, not proof of nonzero contribution on
    every input. Zero Gamma is exactly Power4-equivalent for this model hash.
    """
    if entry is None or entry['family']!='ICNN':return {}
    value=None;scope='NOT_AVAILABLE'
    if model is not None:
        arrays=getattr(model,'arrays',{})
        if 'amplitude' in arrays:
            raw=np.asarray(arrays['amplitude'])
            if raw.size!=1:raise ValueError('ICNN native amplitude must be scalar')
            value=float(raw.item());scope='EXACT_HASH_LOADED_NATIVE_GAMMA_STRUCTURAL_ONLY'
    if value is None:
        audit=entry.get('saved_audit') or {}
        if audit.get('model_sha256')==entry['model_sha256']:
            value=(audit.get('neural_branch') or {}).get('amplitude')
            if value is not None:scope='EXACT_HASH_BOUND_SAVED_NATIVE_GAMMA_STRUCTURAL_ONLY'
    if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<0):
        raise ValueError('Invalid authenticated ICNN native amplitude')
    return dict(neural_amplitude=value,neural_amplitude_is_zero=None if value is None else value==0.,
                neural_activity_scope=scope)


def index_models(records,support):
    expected={(r['suite_id'],r['region_index']):r for r in support};result={}
    for row in records:
        identity=(row['suite_id'],row['region_index'],row['family'])
        if identity[:2] not in expected or identity[2] not in FAMILIES or identity in result:
            raise ValueError('Foreign or duplicate frozen model record')
        record=expected[identity[:2]]
        if row.get('graph_sha256')!=record['sha256']:raise ValueError('Frozen record graph mismatch')
        for role in ('train','valid'):
            if (not row.get('current_'+role+'_dataset_identity')
                or row.get('audit_'+role+'_dataset_identity')!=row['current_'+role+'_dataset_identity']):
                raise ValueError('Frozen model audit is not current on both roles')
        if row.get('audit_model_sha256')!=row.get('model_sha256'):
            raise ValueError('Frozen model audit hash mismatch')
        result[identity]=row
    return result


def index_portfolios(portfolio,support):
    expected={(r['suite_id'],r['region_index']):r for r in support};result={}
    for item in portfolio['portfolios']:
        sid,family=item['suite_id'],item['requested_family']
        if family not in FAMILIES:raise ValueError('Unknown portfolio family')
        for row in item['regions']:
            identity=(sid,row['region_index'],family)
            if identity in result or identity[:2] not in expected:raise ValueError('Duplicate/foreign portfolio region')
            if (row['graph_sha256']!=expected[identity[:2]]['sha256'] or row['component'] not in FAMILIES
                or row['component'] not in (family,'POWER4','FFTT')):
                raise ValueError('Portfolio component/graph mismatch')
            if row['component']!='FFTT':descriptor(row['model'],family=row['component'],source='portfolio')
            result[identity]=row
    if set(result)!={(sid,rid,f) for sid,rid in expected for f in FAMILIES}:
        raise ValueError('Frozen portfolios must exactly cover every requested family and active region')
    return result


def _baseline(ctx,record):
    path=Path(ctx.source_root)/'training/regions'/record['suite_id']/f"{record['region_index']:06d}"/'STATE.json'
    if not path.exists():return {},None
    path,digest=_bound(ctx,path,'CHECKPOINT_PINNED_BASELINE_SELECTION')
    state=read_json(path)
    result={family:descriptor(state.get('selected',{}).get(family),family=family,source='original_selected')
            for family in ('ICNN','XGBOOST')}
    result['POWER4']=descriptor(state.get('power4'),family='POWER4',source='original_power4')
    return result,dict(path=str(path),sha256=digest)


def _load_frozen_model(ctx,entry,check):
    path=Path(entry['model_path']).resolve(strict=True)
    if not (path.is_relative_to(Path(ctx.root).resolve()) or path.is_relative_to(Path(ctx.source_root).resolve())):
        raise ValueError('Frozen model escapes authenticated source/output roots')
    check();ctx.bind(path,'FROZEN_EVALUATED_MODEL',entry['model_sha256'])
    # Reject compression bombs before np.load; measured process RAM remains ctx.guard-owned.
    with zipfile.ZipFile(path) as archive:
        total=sum(i.file_size for i in archive.infolist())
        if total>272*1024**2:raise MemoryError('Frozen model exceeds bounded raw NPZ envelope')
    model=load_model(path);check()
    return model


def _cohorts(ctx,record,inputs,freeze_path,freeze_sha256,check,deadline,*,need_training=True):
    datasets={};dispositions={};bindings=[]
    if need_training:
        try:
            check()
            try:
                train,valid,region=load_training_cohorts(ctx,record,deadline=deadline,read_only=True,
                    selection_freeze_path=freeze_path,selection_freeze_sha256=freeze_sha256)
            finally:
                ctx.require_action('holdout')
            datasets.update(TRAIN=train,VALID=valid)
            dispositions.update(TRAIN='AVAILABLE',VALID='AVAILABLE')
        except TimeoutError:raise
        except Exception as exc:
            dispositions.update({r:dict(status='UNAVAILABLE_COHORT',error_type=type(exc).__name__,error=str(exc)) for r in ('TRAIN','VALID')})
    region_key=key(record);entry=inputs.get(region_key)
    if entry is None:
        dispositions['TEST']='UNREQUESTED_TEST'
    else:
        # Only the new-root canonical results path is checked; old TEST is never consulted.
        result_path=Path(ctx.root)/'holdout/results'/record['suite_id']/f"{record['region_index']:06d}"/'RESULT.json'
        if not result_path.exists():dispositions['TEST']='UNAVAILABLE_HELDOUT_RESULT'
        else:
            try:
                check();test=load_holdout_dataset(ctx,region_key,freeze_path=freeze_path,freeze_sha256=freeze_sha256)
                if (test.get('role')!='TEST' or test['requested_inputs']!=entry['requested_inputs']
                    or test['identity']['graph_sha256']!=record['sha256']):
                    raise ValueError('New holdout graph/role/request population mismatch')
                datasets['TEST']=test;dispositions['TEST']='AVAILABLE'
                bindings.append(dict(path=str(result_path),sha256=ctx.bind(result_path,'NEW_HELDOUT_RESULT')))
            except TimeoutError:raise
            except Exception as exc:
                dispositions['TEST']=dict(status='UNAVAILABLE_HELDOUT_DATASET',error_type=type(exc).__name__,error=str(exc))
    return datasets,dispositions,bindings


def evaluate_region(ctx,record,*,models,portfolios,input_rows,freeze_path,freeze_sha256,deadline):
    """Called only after public run's global freeze/opening admission."""
    check=lambda:ctx.guard(deadline=deadline)
    check();baseline,baseline_binding=_baseline(ctx,record)
    descriptors={};prefix=(record['suite_id'],record['region_index'])
    current={}
    for role in ('train','valid'):
        values={row['current_'+role+'_dataset_identity'] for k,row in models.items() if k[:2]==prefix}
        if len(values)>1:raise ValueError('Frozen regional models disagree on current cohort')
        current[role.upper()]=next(iter(values),None)
    for family in FAMILIES:
        descriptors[('baseline',family)]=baseline.get(family)
        recovered=models.get((*prefix,family))
        descriptors[('recovered',family)]=descriptor(recovered,family=family,source='frozen_recovered')
        picked=portfolios[(*prefix,family)]
        descriptors[('portfolio',family)]=(None if picked['component']=='FFTT' else
            descriptor(picked['model'],family=picked['component'],source='frozen_portfolio'))
        component=models.get((*prefix,picked['component']))
        if component and (picked['component']=='FFTT' or component['model_sha256']==picked['model']['model_sha256']):
            descriptors[('portfolio',family)]=descriptor(component,family=picked['component'],source='frozen_portfolio')
    # FFTT is the same exact native c0 in every version; reuse its frozen current
    # audit when available, without inventing a serialized baseline model hash.
    fftt=descriptors[('recovered','FFTT')]
    if fftt:descriptors[('baseline','FFTT')]=fftt
    reusable={}
    for identity,entry in descriptors.items():
        for role in ('TRAIN','VALID'):
            result=reusable_metric(entry,role,current[role])
            if result is not None:reusable[(*identity,role)]=result
    need_training=any((*identity,role) not in reusable for identity,entry in descriptors.items()
        for role in ('TRAIN','VALID') if entry is not None or identity[1]=='FFTT')
    datasets,dispositions,bindings=_cohorts(ctx,record,input_rows,freeze_path,freeze_sha256,check,deadline,
                                           need_training=need_training)
    for role in ('TRAIN','VALID'):
        if role in datasets and current[role] is not None and datasets[role]['dataset_identity']!=current[role]:
            raise ValueError('Loaded current cohort differs from immutable selection audit')
    cache={};rows=[];neural={}
    for entry in descriptors.values():
        if entry and entry['family']=='ICNN':neural[entry['model_sha256']]=neural_metadata(entry)
    for version in VERSIONS:
        for family in FAMILIES:
            entry=descriptors[(version,family)]
            picked=portfolios[(*prefix,family)]
            component=picked['component'] if version=='portfolio' else family
            analytic=(component=='FFTT')
            model=None;model_error=None
            token=('ANALYTIC_FFTT' if analytic else None if entry is None else entry['model_sha256'])
            for role in ROLES:
                dataset=datasets.get(role)
                requested=(input_rows.get(key(record),{}).get('requested_inputs',0) if role=='TEST' else None)
                row=dict(suite_id=record['suite_id'],region_index=record['region_index'],family=family,
                    version=version,role=role,status=None,requested_inputs=requested,
                    model_sha256=None if analytic or entry is None else entry['model_sha256'],
                    component_identity=token,fallback_component=component,
                    fallback_reason=picked['reason'] if version=='portfolio' else None,
                    property_status='ANALYTIC_FFTT' if analytic else None if entry is None else entry['property_status'],
                    model_source='exact_native_c0' if analytic else None if entry is None else entry['source'],
                    target_met=False,train_target_met=False,TEST_used_for_selection=False)
                if (version,family,role) in reusable:
                    row.update(reusable[(version,family,role)])
                elif dataset is None:
                    disposition=dispositions.get(role,'UNAVAILABLE_COHORT')
                    row.update(disposition if isinstance(disposition,dict) else dict(status=disposition))
                elif token is None:
                    row.update(status='UNAVAILABLE_MODEL',dataset_identity=dataset['dataset_identity'],
                               admitted_inputs=len(dataset['u']),coordinates=0)
                else:
                    row.update(dataset_identity=dataset['dataset_identity'],admitted_inputs=len(dataset['u']))
                    cache_key=(token,role,dataset['dataset_identity'])
                    cached=cache_key in cache
                    if not cached:
                        try:
                            check()
                            if analytic:model=_FFTT(dataset['c0'])
                            elif model_error is not None:raise model_error
                            elif model is None:
                                model=_load_frozen_model(ctx,entry,check)
                                if entry['family']=='ICNN':neural[token]=neural_metadata(entry,model)
                            cache[cache_key]=native_metrics(model,dataset,check=check)
                            cache[cache_key]['metric_evidence']='FRESH_POSTFREEZE_EVALUATION'
                        except TimeoutError:raise
                        except Exception as exc:
                            if model is None:model_error=exc
                            cache[cache_key]=dict(status='INVALID_MODEL_OR_EVALUATION',error_type=type(exc).__name__,error=str(exc))
                    row.update(cache[cache_key])
                    if cached and row.get('metric_evidence')=='FRESH_POSTFREEZE_EVALUATION':
                        row['metric_evidence']='REUSED_IDENTICAL_MODEL_AND_COHORT_WITHIN_EVALUATION'
                        row['inference_seconds']=0.
                    if analytic:row['component_identity']='ANALYTIC_FFTT:'+array_hash({'c0':np.asarray(dataset['c0'])})
                rows.append(row)
            del model
    for row in rows:
        row.update(neural.get(row['model_sha256'],{}))
        tr=next(r for r in rows if (r['version'],r['family'],r['role'])==(row['version'],row['family'],'TRAIN'))
        qualified=tr['status']=='MEASURED' and tr['WMAPE_percent']<1.
        row['train_target_met']=qualified
        row['target_met']=qualified if row['role']=='TRAIN' else None
    return rows,dict(region_key=key(record),baseline_binding=baseline_binding,holdout_bindings=bindings,
        dataset_identities={**current,**{role:d['dataset_identity'] for role,d in datasets.items()}},
        statuses=dict(Counter(r['status'] for r in rows)),selection_changed=False)


def unavailable_rows(record,status,input_rows,exc=None):
    for version in VERSIONS:
        for family in FAMILIES:
            for role in ROLES:
                yield dict(suite_id=record['suite_id'],region_index=record['region_index'],family=family,
                    version=version,role=role,status=status,requested_inputs=input_rows.get(key(record),{}).get('requested_inputs',0) if role=='TEST' else None,
                    error_type=type(exc).__name__ if exc else None,error=str(exc) if exc else None,
                    TEST_used_for_selection=False,target_met=False,train_target_met=False)


_CONTEXTS={}


def _worker(config,payload,deadline):
    from .context import RecoveryContext
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(1)
    if config not in _CONTEXTS:_CONTEXTS[config]=RecoveryContext(config,step_id='holdout')
    ctx=_CONTEXTS[config];ctx.require_action('holdout')
    # Keep process-local authority/cache reuse without quadratic cumulative
    # per-region input manifests. Each region retains its own complete bindings.
    if hasattr(ctx,'_evaluation_authority_bindings'):
        ctx.bindings=dict(ctx._evaluation_authority_bindings)
    require_selection_freeze(ctx,payload['freeze_path'],payload['freeze_sha256'])
    if not hasattr(ctx,'_evaluation_authority_bindings'):
        ctx._evaluation_authority_bindings=dict(ctx.bindings)
    record=payload['record'];started=time.time()
    try:
        with threadpool_limits(limits=1):
            rows,evidence=evaluate_region(ctx,record,models=payload['models'],portfolios=payload['portfolios'],
                input_rows=payload['input_rows'],freeze_path=payload['freeze_path'],
                freeze_sha256=payload['freeze_sha256'],deadline=deadline)
    except Exception as exc:
        status='UNCOMPLETED_TIME_LIMIT' if isinstance(exc,TimeoutError) else 'REGION_EVALUATION_FAILED'
        rows=list(unavailable_rows(record,status,payload['input_rows'],exc))
        evidence=dict(region_key=key(record),error_type=type(exc).__name__,error=str(exc))
    ctx.flush_bindings()
    return dict(region_key=key(record),rows=rows,evidence=evidence,elapsed_seconds=time.time()-started)


def run(ctx,*,freeze_path,freeze_sha256,deadline,national_timing_active,
        output_relative='evaluation/regional_v1',max_workers=4):
    """One bounded streamed export; every frozen support key remains accounted."""
    ctx.require_action('holdout');ctx.guard(deadline=deadline)
    if national_timing_active is not False:raise PermissionError('Caller must establish no concurrent national timing')
    if type(max_workers) is not int or not 1<=max_workers<=4:raise ValueError('One to four bounded regional workers required')
    canonical=Path(ctx.root)/'training/MODEL_SELECTION_FREEZE.json'
    if Path(freeze_path).resolve()!=canonical.resolve():raise PermissionError('Canonical global model selection freeze required')
    freeze,inputs=require_selection_freeze(ctx,freeze_path,freeze_sha256)
    opened_path=Path(ctx.root)/'holdout/TEST_OPENED.json'
    opened=read_json(opened_path)
    if opened['selection_freeze_sha256']!=freeze_sha256 or opened['epoch']<freeze['created_epoch']:
        raise PermissionError('TEST opening must follow this exact immutable selection freeze')
    ctx.bind(opened_path,'AUTHORIZED_TEST_OPENING')
    records=support_records(ctx);input_rows={r['region_key']:r for r in inputs['rows']}
    if len(input_rows)!=len(inputs['rows']):raise ValueError('Duplicate requested holdout population')
    frozen_models=read_json(freeze['model_records']['path']);models=index_models(frozen_models,records)
    portfolio=read_json(freeze['portfolio']['path']);portfolios=index_portfolios(portfolio,records)
    destination=(Path(ctx.root)/output_relative).resolve()
    if not destination.is_relative_to(Path(ctx.root).resolve()) or destination==Path(ctx.root).resolve():
        raise ValueError('Evaluation output escapes the new run')
    destination.mkdir(parents=True,exist_ok=False)
    started=time.time();timed_out=False;counts=Counter();row_count=0;ledger=[]
    source_files=[Path(__file__),Path(__file__).with_name('training_recovery.py'),Path(__file__).with_name('labels.py')]
    sources={str(p):ctx.bind(p,'REGIONAL_EVALUATION_SOURCE') for p in source_files}
    new_json(destination/'INTENT.json',dict(schema=SCHEMA,phase_id=ctx.c['phase_id'],run_id=ctx.c['run_id'],
        started_epoch=started,deadline_epoch=deadline,config_path=str(ctx.path),config_sha256=ctx.config_sha256,
        max_workers=max_workers,no_concurrent_national_timing_attested=True,
        selection_freeze=dict(path=str(freeze_path),sha256=freeze_sha256),
        tasks=[key(r) for r in records],sources=sources,resources=ctx.c['resources'],
        expected_outputs=['learning.csv','difficult_regions.csv','REGION_LEDGER.json','RESULT.json'],
        acceptance=['all support keys accounted','identical cohort comparisons','no fitting or selection','explicit metric missingness']))
    from .labels import _parallel
    jobs=[];grouped=defaultdict(dict)
    for identity,row in models.items():grouped[identity[:2]][identity]=row
    for record in records:
        prefix=(record['suite_id'],record['region_index']);region_key=key(record)
        regional_models=grouped.get(prefix,{})
        if not regional_models and region_key not in input_rows:continue
        payload=dict(record=record,region_key=region_key,models=regional_models,
            portfolios={(*prefix,f):portfolios[(*prefix,f)] for f in FAMILIES},
            input_rows={region_key:input_rows[region_key]} if region_key in input_rows else {},
            freeze_path=str(freeze_path),freeze_sha256=freeze_sha256)
        jobs.append((str(ctx.path),payload))
    jobs.sort(key=lambda job:(-job[1]['record']['pairs'],job[1]['region_key']))
    # The pilot consumes the same clock and its outcomes are reused, not repeated.
    # Leave60s for streamed accounting; no model/label work occurs after this end.
    numerical_end=float(deadline)-60.
    pilot=_parallel(ctx,_worker,jobs[:max_workers],numerical_end,max_workers=max_workers) if jobs else []
    durations=[r['elapsed_seconds'] for r in pilot if 'elapsed_seconds' in r]
    new_json(destination/'FORECAST.json',dict(completed_pilot_regions=len(durations),pilot_regions=len(pilot),
        elapsed_region_seconds=durations,remaining_regions=max(0,len(jobs)-len(pilot)),
        estimated_remaining_wall_seconds=(max(durations)*max(0,len(jobs)-len(pilot))/max_workers if durations else None),
        forecast_basis='Largest-pair regions; estimate only; cached metadata and cohorts differ.',
        remaining_seconds=max(0.,numerical_end-time.time()),original_deadline_epoch=deadline))
    outputs=pilot+(_parallel(ctx,_worker,jobs[max_workers:],numerical_end,max_workers=max_workers) if len(jobs)>max_workers else [])
    results={}
    for item in outputs:
        identity=item.get('region_key') or item.get('job_key') or (item.get('job') or {}).get('region_key')
        if identity is None or identity in results:raise ValueError('Worker accounting duplicate or unknown region')
        results[identity]=item
    with (destination/'learning.csv').open('x',encoding='utf-8',newline='') as stream, (destination/'difficult_regions.csv').open('x',encoding='utf-8',newline='') as hard:
        writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader()
        difficult=csv.DictWriter(hard,fieldnames=('suite_id','region_index','family','status','TRAIN_WMAPE_percent',
            'VALID_WMAPE_percent','TEST_WMAPE_percent','train_target_met','model_sha256','portfolio_component','portfolio_reason'))
        difficult.writeheader()
        for ordinal,record in enumerate(records):
            if ordinal%64==0:ctx.guard(deadline=deadline)
            item=results.get(key(record))
            if item is not None and 'rows' in item:
                rows,evidence=item['rows'],item['evidence']
            else:
                requested_job=key(record) in input_rows or (record['suite_id'],record['region_index']) in grouped
                status=(item.get('status','REGION_EVALUATION_FAILED') if item else
                        'UNCOMPLETED_MISSING_WORKER_RESULT' if requested_job else 'UNAVAILABLE_COHORT')
                rows=list(unavailable_rows(record,status,input_rows));evidence=dict(region_key=key(record),worker_disposition=item)
                if item is None and not requested_job:
                    for row in rows:
                        if row['role']=='TEST':row['status']='UNREQUESTED_TEST'
            timed_out|=any('TIME_LIMIT' in r['status'] or 'RESOURCE_LIMIT' in r['status'] for r in rows)
            ledger.append(evidence)
            for row in rows:
                writer.writerow(row);counts[row['status']]+=1;row_count+=1
            for family in ('ICNN','XGBOOST'):
                selected={r['role']:r for r in rows if r['version']=='recovered' and r['family']==family}
                choice=portfolios[(record['suite_id'],record['region_index'],family)]
                difficult.writerow(dict(suite_id=record['suite_id'],region_index=record['region_index'],family=family,
                    status=selected['TRAIN']['status'],TRAIN_WMAPE_percent=selected['TRAIN'].get('WMAPE_percent'),
                    VALID_WMAPE_percent=selected['VALID'].get('WMAPE_percent'),TEST_WMAPE_percent=selected['TEST'].get('WMAPE_percent'),
                    train_target_met=selected['TRAIN']['train_target_met'],model_sha256=selected['TRAIN'].get('model_sha256'),
                    portfolio_component=choice['component'],portfolio_reason=choice['reason']))
    if row_count!=len(records)*len(VERSIONS)*len(FAMILIES)*len(ROLES):raise RuntimeError('Incomplete regional row accounting')
    ctx.guard(deadline=deadline)
    for p,digest in sources.items():ctx.bind(p,'UNCHANGED_EVALUATION_SOURCE',digest)
    require_selection_freeze(ctx,freeze_path,freeze_sha256)
    new_json(destination/'REGION_LEDGER.json',ledger)
    result=dict(schema=SCHEMA,status='COMPLETE_ACCOUNTING_WITH_SCOPE_LIMITATIONS' if not timed_out else 'TIME_LIMIT_COMPLETE_DISPOSITION_ACCOUNTING',
        regions=len(records),row_count=row_count,status_counts=dict(counts),elapsed_seconds=time.time()-started,
        max_workers=max_workers,measured_worker_seconds=sum(r.get('elapsed_seconds',0.) for r in outputs),
        native_audit_reuse=True,
        deadline_epoch=deadline,selection_freeze=dict(path=str(freeze_path),sha256=freeze_sha256),
        semantics='Baseline original selection on same CURRENT cohort. Portfolio whole-region choice fixed before TEST. '
                  'All native sums exclude unavailable observations; zero truth remains undefined. '
                  'TRAIN target flags apply only to TRAIN, not VALID/TEST.',
        model_fitting=False,model_selection_changed=False,old_TEST_opened=False,TEST_used_for_selection=False,
        outputs={p.name:dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in destination.iterdir() if p.is_file()})
    ctx.guard(deadline=deadline)
    new_json(destination/'RESULT.json',result);ctx.flush_bindings();return result
