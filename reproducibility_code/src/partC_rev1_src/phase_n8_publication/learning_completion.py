"""Complete only the missing frozen eight; reuse all forty existing fits unchanged."""
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,write_json,read_npz,sha
from partC_rev1_src.phase_n8_r4.learning import fit
from partC_rev1_src.phase_n8_r4.learning_artifacts import mark_neural_failure,ready_models
from partC_rev1_src.phase_n8_r4.learning_data import RegionalSources
from partC_rev1_src.phase6.common import input_hash
from .cost_metrics import error_rows,regional_aggregate


def frozen_truths(ctx,source,episode,rid):
    union=source.union(rid);desc=union.meta['source_descriptor'];cohorts=source.evaluation(union)
    dest=episode/'learning/regions'/rid/'post_freeze_evaluation'
    if (dest/'FROZEN.json').exists():
        freeze=read_json(dest/'FROZEN.json');ctx.bind(dest/'FROZEN.json','FROZEN_LEARNING_QUERY_PLAN')
        ctx.bind(episode/'learning/ALL_EXPERIMENT_MODELS_FROZEN.json','MODELS_FROZEN_BEFORE_QUERIES',freeze['all_experiment_models_frozen_sha256'])
        ctx.bind(dest/'frozen_inputs.npz','FROZEN_QUERY_INPUTS',freeze['input_sha256'])
        ctx.bind(dest/'frozen_rows.csv','FROZEN_QUERY_IDENTITIES',freeze['row_sha256'])
        xs=read_npz(dest/'frozen_inputs.npz')['u'];planned=pd.read_csv(dest/'frozen_rows.csv',dtype={'input_hash':str})
        if planned.input_hash.duplicated().any():raise ValueError('Duplicate learning query')
        attempts=read_json(dest/'ATTEMPTS.json') if (dest/'ATTEMPTS.json').exists() else []
        if attempts:ctx.bind(dest/'ATTEMPTS.json','FROZEN_LABEL_ADMISSION_OUTCOMES')
        by={(a['cohort'],a['slot']):a for a in attempts}
        if len(by)!=len(attempts):raise ValueError('Duplicate learning label attempt')
        new={};store=dest/'certified_labels'
        for row in planned.itertuples(index=False):
            attempt=by.get((row.cohort,row.slot))
            if attempt is None or not attempt.get('admitted_for_model_evaluation',False):continue
            if attempt['input_hash']!=row.input_hash:raise ValueError('Attempt/query differs')
            rp=store/'receipts'/(row.input_hash+'.json');ctx.bind(rp,'CERTIFIED_LOCAL_LABEL_RECEIPT');receipt=read_json(rp)
            if (receipt['model_hash'],receipt['support_hash'])!=(desc['model_hash'],desc['support_hash']):raise ValueError('Label model/support changed')
            payload=(store/receipt['payload_path']).resolve()
            if not payload.is_relative_to(store.resolve()):raise ValueError('Label payload escapes frozen directory')
            ctx.bind(payload,'CERTIFIED_LOCAL_LABEL_PAYLOAD',receipt['payload_sha256']);a=read_npz(payload);scalar=receipt['scalars']
            if scalar['quality'] not in {'A','B'} or not np.array_equal(a['demand'],xs[row.input_index]):raise ValueError('Label admission/input differs')
            if input_hash(desc['model_hash'],desc['support_hash'],a['demand'])!=row.input_hash:raise ValueError('Label input hash mismatch')
            new.setdefault(row.cohort,[]).append((a['demand'],a['tau'],scalar['value_lower'],scalar['value_upper'],scalar['roundoff_allowance'],
                                                  dict(input_hash=row.input_hash,quality=scalar['quality'])))
        for cohort,items in new.items():
            cohorts[cohort]=(*[np.asarray([x[k] for x in items]) for k in range(5)],pd.DataFrame([x[5] for x in items]))
    return union,cohorts


def direct_all_frozen_costs(ctx,old_fits,new_fits,deadline):
    from types import SimpleNamespace
    source=RegionalSources(ctx);dst=ctx.root/'learning/cost_wmape';dst.mkdir(exist_ok=False)
    rows={};inventory={};failure=None;completed=[]
    summary_columns=['region_id','stratum','cohort','family','seed','fraction','unique_n','fit_status',
        'optimizer_updates','selected_update','fit_reused','cost_abs_error_sum','true_cost_sum','n_records',
        'n_coordinates','MARE_equal_pairs','MARE_LOW_pair_weighted','negative_coordinates',
        'minimum_cost_seconds','cost_wmape_percent','cost_wmape_status']
    pd.DataFrame(columns=summary_columns).to_csv(dst/'region_fraction_cost_wmape.csv',index=False)
    def flush(rid,parts):
        if not parts:return
        f=pd.concat(parts,ignore_index=True)
        path=dst/(rid.removeprefix('partC_p2_region_')+'_query_rows.csv');f.to_csv(path,index=False,float_format='%.17g')
        inventory[rid]=dict(region_id=rid,path=path.name,sha256=sha(path),rows=len(f))
        rows[rid]=regional_aggregate(f,['region_id','stratum','cohort','family','seed','fraction','unique_n',
            'fit_status','optimizer_updates','selected_update','fit_reused'])
        pd.concat(list(rows.values()),ignore_index=True).to_csv(dst/'region_fraction_cost_wmape.csv',index=False,float_format='%.17g')
        write_json(dst/'ROW_INVENTORY.json',list(inventory.values()))
    try:
        for episode,fits in [(ctx.r4,old_fits),(ctx.root,new_fits)]:
            for rid in sorted({r['region_id'] for r in fits}):
                ctx.guard(deadline=deadline)
                union,cohorts=frozen_truths(ctx,source,episode,rid);parts=[]
                for result in [r for r in fits if r['region_id']==rid]:
                    for family,kind,arrays,meta in ready_models(SimpleNamespace(root=episode,bind=ctx.bind),result,union.s['c0']):
                        for cohort,fields in cohorts.items():
                            ctx.guard(deadline=deadline)
                            data=dict(zip(['u','t','L','U','eps'],fields[:5]));data['low']=union.s['low']
                            f=error_rows(kind,arrays,data,fields[5],rid,cohort,stratum=result['stratum'],family=family,
                                seed=result['seed'],fraction=result['fraction'],unique_n=result['unique_n'],
                                fit_reused=episode==ctx.r4,**meta)
                            f['source_quality']=fields[5].quality.to_numpy();parts.append(f);flush(rid,parts)
                completed.append(rid)
                source.flush_bindings();print('LEARNING_COST_WMAPE',rid,'cohorts',len(cohorts),flush=True)
    except Exception as exc:
        failure=dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        source.close();write_json(dst/'ROW_INVENTORY.json',list(inventory.values()))
    expected=sorted({r['region_id'] for r in old_fits+new_fits})
    result=dict(status='PASS' if failure is None and set(completed)==set(expected) else 'PARTIAL_COST_EXPORT',
        regions=len(inventory),completed_regions=completed,missing_or_incomplete_regions=sorted(set(expected)-set(completed)),
        row_files=list(inventory.values()),failure=failure,direct_frozen_model_inference=True,training_repeated=False)
    write_json(dst/'RESULT.json',result)
    return result


def run(ctx,args):
    import torch
    out=ctx.root/'learning';old=ctx.r4/'learning'
    for name in ('PREPARE.json','STUDY_FREEZE.json','FIT_RESULTS.json','selected_regions_16.csv'):
        ctx.bind(old/name,'IMMUTABLE_R4_LEARNING_SOURCE')
    prepared=read_json(old/'PREPARE.json');old_plan=read_json(old/'STUDY_FREEZE.json');old_fits=read_json(old/'FIT_RESULTS.json')
    if len(old_fits)!=40 or any(r['status']!='COMPLETE' for r in old_fits):raise ValueError('The forty reusable fits differ')
    old_regions={r['region_id'] for r in old_fits};missing=[r for r in prepared['prepared_regions'] if r['region_id'] not in old_regions]
    if len(missing)!=8 or any(r['rank_within_stratum'] not in (3,4) for r in missing):raise ValueError('Frozen missing-eight docket differs')
    if (out/'STUDY_FREEZE.json').exists():raise FileExistsError('Learning stage already started; original clock must be reconciled')
    deadline=min(args.job_start+10800,args.job_deadline,ctx.c['compute_cutoff_epoch'])
    plan=dict(old_plan);plan.update(status='V5_COMPLETION_PLAN_FROZEN',selected_region_ids=[r['region_id'] for r in missing],
        prepared_selected_region_ids=[r['region_id'] for r in missing],selected_region_strata={r['region_id']:r['stratum'] for r in missing},
        failed_preparation_regions=[],seeds=[0],stage_deadline_epoch=deadline,training_deadline_epoch=deadline-2700,
        frozen_epoch=time.time(),intended_total_regions=16,reused_region_count=8,reused_fit_count=40,completion_region_count=8,
        original_protocol_sha256=sha(old/'STUDY_FREEZE.json'),metric_reporting='DIRECT_COST_WMAPE; VALID_SELECTION_POLICY_UNCHANGED')
    for key in ('runtime_pilot_choices','selected_runtime_choice','pilot_sha256','incomplete_runtime_pilot_regions'):plan.pop(key,None)
    write_json(out/'STUDY_FREEZE.json',plan)
    write_json(out/'LEARNING_CLOCK.json',dict(cap_seconds=10800,original_stage_start_epoch=args.job_start,stage_deadline_epoch=deadline))
    shutil.copy2(old/'selected_regions_16.csv',out/'selected_regions_16.csv')
    for r in missing:
        ctx.guard(deadline=deadline);source=old/'regions'/r['region_id'];dest=out/'regions'/r['region_id'];dest.mkdir(parents=True,exist_ok=False)
        files={'train.npz':r['train_file_sha256'],'train_rows.csv':r['train_rows_sha256']}
        for split in r['validation_splits']:
            binding=r['validation_artifacts'][split];files[split+'.npz']=binding['arrays_sha256'];files[split+'_rows.csv']=binding['rows_sha256']
        files['PREPARED.json']=ctx.bind(source/'PREPARED.json','FROZEN_REGION_PREPARATION')
        for name,expected in files.items():
            ctx.bind(source/name,'FROZEN_PREPARED_INPUT',expected);shutil.copy2(source/name,dest/name)
            if sha(dest/name)!=expected:raise ValueError('Prepared input copy changed')
    missing.sort(key=lambda r:(r['rank_within_stratum'],r['stratum']))
    docket=[(r,fraction) for fraction in old_plan['fractions'] for r in missing]
    results=[dict(region_id=r['region_id'],stratum=r['stratum'],seed=0,fraction=fraction,status='NOT_STARTED') for r,fraction in docket]
    write_json(out/'FIT_RESULTS.json',results);torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    for i,(r,fraction) in enumerate(docket):
        if time.time()>=plan['training_deadline_epoch']:results[i]['status']='NOT_RUN_BUDGET'
        else:
            try:results[i]=fit(ctx,r,0,fraction,plan)
            except Exception as exc:
                dest=out/'regions'/r['region_id']/'fits'/f'seed_0_fraction_{round(fraction*100):03d}'
                failed=dict(results[i],status='FAILED',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
                failed.update(mark_neural_failure(dest,type(exc).__name__))
                if (dest/'CHECKPOINT.json').exists():failed['last_saved_checkpoint']=read_json(dest/'CHECKPOINT.json')
                if dest.exists():write_json(dest/'FAILURE.json',failed)
                results[i]=failed
        write_json(out/'FIT_RESULTS.json',results)
    write_json(out/'ALL_EXPERIMENT_MODELS_FROZEN.json',dict(status='EXPERIMENT_MODELS_FROZEN',results=results,
        frozen_epoch=time.time(),study_plan_sha256=sha(out/'STUDY_FREEZE.json'),stage_deadline_epoch=deadline,
        immutable_reused_freeze_path=str(old/'ALL_EXPERIMENT_MODELS_FROZEN.json'),
        immutable_reused_freeze_sha256=ctx.bind(old/'ALL_EXPERIMENT_MODELS_FROZEN.json','UNCHANGED_FORTY_MODEL_FREEZE'),
        native_national_registries_unchanged=True))
    write_json(out/'COMBINED_FIT_INVENTORY.json',[dict(r,reused=True) for r in old_fits]+[dict(r,reused=False) for r in results])
    from .learning_evaluate import evaluate
    evaluation=evaluate(ctx,plan,missing,results,deadline-600)
    coverage=[]
    for episode,reused in [(ctx.r4,True),(ctx.root,False)]:
        path=episode/'learning/new_label_coverage.csv'
        if path.exists():
            ctx.bind(path,'ALL_SIXTEEN_FROZEN_REGION_LABEL_COVERAGE');frame=pd.read_csv(path)
            frame['fit_reused']=reused;coverage.append(frame)
    if coverage:pd.concat(coverage,ignore_index=True).to_csv(out/'combined_label_coverage.csv',index=False,float_format='%.17g')
    costs=direct_all_frozen_costs(ctx,old_fits,results,deadline-60)
    complete=all(r['status']=='COMPLETE' for r in results) and evaluation['status']=='EXECUTED' and costs['status']=='PASS'
    return dict(status='EXECUTED' if complete else 'PARTIAL_LEARNING_COMPLETION',
        reused_fits=40,new_completed_fits=sum(r['status']=='COMPLETE' for r in results),intended_regions=16,
        evaluation=evaluation,costs=costs,shared_stage_seconds=time.time()-args.job_start)
