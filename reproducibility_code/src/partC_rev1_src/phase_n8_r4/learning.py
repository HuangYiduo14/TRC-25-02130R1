"""Bounded, frozen nested-data experiment; diagnostic weights never enter portfolios."""
from __future__ import annotations
import copy
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
import torch
from partC_rev1_src.phase_n8.common import read_json,write_json,read_npz,save_npz,sha,digest,array_hash
from partC_rev1_src.phase67.model import PowerResidualPotential,calibrate_residual_unit,weighted_power4_fit,robust_loss,export_arrays
from partC_rev1_src.phase67.portable import predict_arrays
from partC_rev1_src.phase67.sampling import row_probabilities
from .learning_data import seed,RegionalSources
from .learning_artifacts import prepare_analytic_artifacts,finalize_neural_readiness,readiness_fields,mark_neural_failure


def make_model(a,rid,replicate):
    model=PowerResidualPotential(a['scale'],a['c0'],a['eta_calibration'],eta_unit=a['eta_unit'],width=32,
        head_power=1,seed=seed('N8R4_INIT',rid,replicate))
    idx=a['calibration_indices']
    calibration=calibrate_residual_unit(model,a['u'][idx],a['t'][idx],a['U'][idx],a['pair_weight'])
    return model,calibration


def update(model,opt,a,indices):
    u,t,L,U=[torch.tensor(x,dtype=torch.float64) for x in (a['u'][indices],a['t'][indices],a['L'][indices]-a['eps'][indices],a['U'][indices]+a['eps'][indices])]
    w=torch.tensor(a['pair_weight'],dtype=torch.float64)
    opt.zero_grad(set_to_none=True);loss,pieces=robust_loss(model,u,t,L,U,w)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite original robust training loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
    opt.step();model.project_parameters()
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite trained parameter')
    return float(loss.detach())


def error_rows(kind,arr,a,rows,rid,cohort,**extra):
    frames=[]
    for begin in range(0,len(a['u']),128):
        stop=min(begin+128,len(a['u']));u=a['u'][begin:stop];truth=a['t'][begin:stop]
        value,pred=predict_arrays(kind,arr,u)
        if not np.isfinite(pred).all() or not np.isfinite(value).all():raise FloatingPointError('Nonfinite diagnostic prediction')
        diff=pred-truth;ae=abs(diff);relative=ae/np.maximum(abs(truth),1.)
        gap=np.maximum(a['L'][begin:stop]-a['eps'][begin:stop]-value,0.)+np.maximum(value-a['U'][begin:stop]-a['eps'][begin:stop],0.)
        w=a.get('low',np.ones(u.shape[1]));w=w/w.sum()
        frame=pd.DataFrame(dict(region_id=rid,cohort=cohort,input_hash=rows.input_hash.iloc[begin:stop].to_numpy(),
            n_coordinates=u.shape[1],MARE_equal_pairs=relative.mean(axis=1),MARE_LOW_pair_weighted=relative@w,
            MAE_seconds=ae.mean(axis=1),MSE_seconds2=(diff**2).mean(axis=1),bias_seconds=diff.mean(axis=1),
            maximum_relative_error=relative.max(axis=1),minimum_cost_seconds=pred.min(axis=1),
            negative_coordinates=(pred < -1e-8).sum(axis=1),below_fftt_coordinates=(pred<arr['c0']-1e-8).sum(axis=1),
            interval_violation_relative=gap/np.maximum(abs(a['U'][begin:stop]),1.),
            total_query_flow=u.sum(axis=1)))
        for k,v in extra.items():frame[k]=v
        frames.append(frame)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def validate(model,path,prep,a):
    arr=export_arrays(model);result=[]
    for split in prep['validation_splits']:
        data=read_npz(path/(split+'.npz'));data['low']=a['low']
        rows=pd.read_csv(path/(split+'_rows.csv'),dtype={'input_hash':str})
        result.append(error_rows('POWER4_SKIP_ICNN32',arr,data,rows,prep['region_id'],split))
    frame=pd.concat(result,ignore_index=True)
    summaries=frame.groupby('cohort')[['MARE_equal_pairs','MARE_LOW_pair_weighted','interval_violation_relative']].mean()
    score=float((.5*(summaries.MARE_equal_pairs+summaries.MARE_LOW_pair_weighted)+.1*summaries.interval_violation_relative).mean())
    return arr,score,int(frame.negative_coordinates.sum()),float(frame.MARE_equal_pairs.mean())


def pilot(ctx,prepared,stage_deadline):
    out=ctx.root/'learning';timings=[]
    for r in prepared['prepared_regions']:
        row=dict(region_id=r['region_id'],stratum=r['stratum'],rank_within_stratum=r['rank_within_stratum'],
            dimensions=None,batch=None,target_batch=None,pilot_steps=0,seconds=None,seconds_per_step=None,
            pilot_accuracy_used_for_selection=False,status='NOT_RUN_BUDGET')
        try:
            if time.time()<stage_deadline-2400:
                ctx.guard(deadline=stage_deadline);path=out/'regions'/r['region_id']
                ctx.bind(path/'train.npz','PREPARED_TIMING_PILOT_ARRAYS',r['train_file_sha256'])
                a=read_npz(path/'train.npz')
                model,cal=make_model(a,r['region_id'],0);opt=torch.optim.Adam(model.parameters(),lr=.001,
                    betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False)
                n=r['calibration_n'];rng=np.random.default_rng(seed('N8R4_TIMING',r['region_id']))
                target_batch=64 if len(a['c0'])>=1024 else 128
                begin=time.perf_counter()
                for step in range(32):
                    ctx.guard(deadline=stage_deadline-2400)
                    idx=rng.integers(n,size=min(target_batch,n));update(model,opt,a,idx);row['pilot_steps']=step+1
                elapsed=time.perf_counter()-begin
                row.update(status='COMPLETE',dimensions=len(a['c0']),batch=min(target_batch,n),target_batch=target_batch,
                    seconds=elapsed,seconds_per_step=elapsed/32)
        except Exception as exc:
            row.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        timings.append(row)
        pd.DataFrame(timings).to_csv(out/'runtime_pilot.csv',index=False,float_format='%.17g')
    frame=pd.DataFrame(timings) if timings else pd.DataFrame(columns=['region_id','stratum','rank_within_stratum',
        'dimensions','batch','target_batch','pilot_steps','seconds','seconds_per_step','status'])
    frame.to_csv(out/'runtime_pilot.csv',index=False,float_format='%.17g')
    # Small calibration batches understate full fitting cost; scale linearly and add 50% margin.
    rate=pd.to_numeric(frame.seconds_per_step)*np.maximum(1.,pd.to_numeric(frame.target_batch)/pd.to_numeric(frame.batch))
    available=stage_deadline-time.time()-2400
    choices=[]
    for count,seeds in [(16,2),(16,1),(8,1)]:
        mask=np.ones(len(frame),bool) if count==16 else frame.rank_within_stratum.le(2).to_numpy()
        measurable=bool(int(mask.sum())==count and frame.loc[mask,'status'].eq('COMPLETE').all() and np.isfinite(rate[mask]).all())
        forecast=float((rate[mask]*4000*5*seeds).sum()*1.5+120*count) if measurable else None
        fits=forecast is not None and forecast<=available
        choices.append(dict(regions=count,seeds=seeds,forecast_seconds=forecast,available_training_seconds=available,
            forecast_coverage_complete=measurable,fits=fits))
        if fits:break
    choice=choices[-1]
    if not choice['fits']:
        # Preserve predeclared minimum; honest partial fits if hardware cannot finish it.
        choice=dict(choice,budget_warning='MINIMUM_8_REGIONS_ONE_SEED_FORECAST_EXCEEDS_BUDGET_OR_UNAVAILABLE; partial execution possible')
    take=frame if choice['regions']==16 else frame.loc[frame.rank_within_stratum.le(2)]
    docket=pd.read_csv(out/'selected_regions_16.csv')
    selected_docket=docket if choice['regions']==16 else docket.loc[docket.rank_within_stratum.le(2)]
    plan=dict(status='FROZEN_BEFORE_CURVE_OUTCOMES',selected_region_ids=selected_docket.region_id.tolist(),
        prepared_selected_region_ids=take.region_id.tolist(),selected_region_strata=selected_docket.set_index('region_id').stratum.to_dict(),
        failed_preparation_regions=[r for r in prepared['preparation_failures'] if r['region_id'] in set(selected_docket.region_id)],
        seeds=list(range(choice['seeds'])),fractions=[.1,.25,.5,.7,1.],updates_per_fit=4000,
        learning_rate=.001,optimizer='Adam, original robust_loss, parameter projection, grad-norm clip10',
        optimizer_name='Adam',dtype='float64',gradient_norm_clip=10.,adam_weight_decay=0.,
        adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,adam_amsgrad=False,explicit_L1_L2_regularization=False,
        batch_dimension_threshold=1024,batch_large_dimension=64,batch_small_dimension=128,
        sampling_with_replacement=True,sampling_operating_probability_when_both=.75,
        sampling_stress_probability_when_both=.25,sampling_within_class='equal family probabilities then uniform within each family',
        inverse_sampling_probability_correction=False,huber_delta=.1,
        total_cost_loss_weight=.75,congestion_loss_weight=.25,value_interval_loss_weight=.1,below_c0_penalty_weight=1.,
        pair_weight_policy='0.5/m + 0.5*LOW/sum(LOW)',
        parameter_projection='eta and gate nonnegative without an upper gate clamp; recurrent ICNN/head weights use nonnegative parameterization',
        source_architecture='two32 softplus skip-ICNN, linear head',
        initialization='same complete calibration-initialized model for every fraction within region/seed',
        validation_checks_every_updates=500,validation_selection='minimum fixed VALID score among finite checkpoints; negative counts retained',
        new_evaluation_policy='After all experiment models freeze: 8 current bounded operating inputs and 8 radial stress inputs at2.25-4xLOW; deterministic seeds and source-collision checks; no adaptive selection',
        runtime_pilot_choices=choices,selected_runtime_choice=choice,
        incomplete_runtime_pilot_regions=frame.loc[frame.status.ne('COMPLETE'),'region_id'].tolist(),
        pilot_sha256=sha(out/'runtime_pilot.csv'),
        training_deadline_epoch=stage_deadline-2400,stage_deadline_epoch=stage_deadline,frozen_epoch=time.time())
    write_json(out/'STUDY_FREEZE.json',plan);return plan


def fit(ctx,r,replicate,fraction,plan):
    path=ctx.root/'learning/regions'/r['region_id']
    ctx.bind(path/'train.npz','PREPARED_DIAGNOSTIC_TRAIN_ARRAYS',r['train_file_sha256'])
    ctx.bind(path/'train_rows.csv','PREPARED_DIAGNOSTIC_TRAIN_ROWS',r['train_rows_sha256'])
    a=read_npz(path/'train.npz')
    rows=pd.read_csv(path/'train_rows.csv',dtype={'input_hash':str})
    subset=next(x for x in r['subsets'] if x['fraction']==fraction);n=subset['unique_n']
    dest=path/'fits'/f'seed_{replicate}_fraction_{round(fraction*100):03d}';dest.mkdir(parents=True,exist_ok=False)
    started=time.time();prepare_analytic_artifacts(dest,a,r,replicate,fraction,subset)
    model,calibration=make_model(a,r['region_id'],replicate)
    for split in r['validation_splits']:
        binding=r['validation_artifacts'][split]
        ctx.bind(path/(split+'.npz'),'FIXED_DIAGNOSTIC_VALID_ARRAYS',binding['arrays_sha256'])
        ctx.bind(path/(split+'_rows.csv'),'FIXED_DIAGNOSTIC_VALID_ROWS',binding['rows_sha256'])
    init=export_arrays(model);init_hash=array_hash(init)
    initial_path=path/f'initial_seed_{replicate}.npz'
    if initial_path.exists():
        if array_hash(read_npz(initial_path))!=init_hash:raise ValueError('Initialization changed across fractions')
    else:save_npz(initial_path,init)
    probs,probmeta=row_probabilities(a['u'][:n],a['low'],rows.family.iloc[:n].to_numpy())
    gen=torch.Generator().manual_seed(seed('N8R4_SHUFFLE',r['region_id'],replicate))
    opt=torch.optim.Adam(model.parameters(),lr=plan['learning_rate'],betas=(plan['adam_beta1'],plan['adam_beta2']),
        eps=plan['adam_epsilon'],weight_decay=plan['adam_weight_decay'],amsgrad=plan['adam_amsgrad'])
    exposure=np.zeros(n,np.int64);history=[];best_score=np.inf;best=None;best_step=None;steps=0;status='COMPLETE'
    begin_optim=time.perf_counter();optimizer_seconds=0.;validation_seconds=0.;batch_losses=[]
    for step in range(0,4001):
        if step%500==0:
            ctx.guard(deadline=plan['stage_deadline_epoch']);t=time.perf_counter()
            arr,score,negative,mean=validate(model,path,r,a)
            validation_seconds+=time.perf_counter()-t
            if score<best_score:
                best_score=score;best=arr;best_step=step
                save_npz(dest/'SELECTED_SKIP_ICNN32.npz',best)
            save_npz(dest/'LAST_SKIP_ICNN32.npz',arr)
            pd.DataFrame(dict(input_hash=rows.input_hash.iloc[:n].to_numpy(),exposure_count=exposure)).to_csv(dest/'exposure_counts.csv',index=False)
            write_json(dest/'CHECKPOINT.json',dict(status='IN_PROGRESS',last_saved_update=step,selected_update=best_step,
                selected_VALID_score=best_score,selected_arrays_sha256=sha(dest/'SELECTED_SKIP_ICNN32.npz'),
                last_arrays_sha256=sha(dest/'LAST_SKIP_ICNN32.npz'),optimizer_seconds=optimizer_seconds,
                rows_consumed=int(exposure.sum()),unique_rows_consumed=int(np.count_nonzero(exposure))))
            history.append(dict(region_id=r['region_id'],stratum=r['stratum'],seed=replicate,fraction=fraction,unique_n=n,
                updates=step,validation_score=score,validation_MARE_equal_pairs=mean,negative_validation_coordinates=negative,
                elapsed_seconds=time.time()-started,optimizer_seconds=optimizer_seconds))
            pd.DataFrame(history).to_csv(dest/'optimization_curve.csv',index=False,float_format='%.17g')
            pd.DataFrame(batch_losses).to_csv(dest/'training_batch_losses.csv',index=False,float_format='%.17g')
            print(f'LEARNING {r["stratum"]} {r["region_id"]} seed={replicate} n={n} updates={step} VALID={mean:.6g}',flush=True)
        if step==4000:break
        if time.time()>=plan['training_deadline_epoch']:
            status='TIME_LIMIT';break
        if step%20==0:ctx.guard(deadline=plan['stage_deadline_epoch'])
        t=time.perf_counter();batch=64 if len(a['c0'])>=1024 else 128
        idx=torch.multinomial(torch.tensor(probs,dtype=torch.float64),batch,replacement=True,generator=gen).numpy()
        loss=update(model,opt,a,idx);optimizer_seconds+=time.perf_counter()-t
        exposure+=np.bincount(idx,minlength=n);steps=step+1
        batch_losses.append(dict(updates=steps,robust_training_batch_loss=float(loss),
            cumulative_row_exposures=int(exposure.sum()),unique_rows_exposed=int(np.count_nonzero(exposure)),
            optimizer_seconds=optimizer_seconds))
    # Preserve the final iterate separately; only the selected VALID checkpoint is admitted.
    last=export_arrays(model);save_npz(dest/'FINAL_ITERATE_SKIP_ICNN32.npz',last)
    pd.DataFrame(batch_losses).to_csv(dest/'training_batch_losses.csv',index=False,float_format='%.17g')
    if best is None:raise RuntimeError('No finite validation checkpoint')
    finalize_neural_readiness(dest,best_step,best_score,status)
    pd.DataFrame(dict(input_hash=rows.input_hash.iloc[:n].to_numpy(),exposure_count=exposure)).to_csv(dest/'exposure_counts.csv',index=False)
    result=dict(status=status,region_id=r['region_id'],stratum=r['stratum'],seed=replicate,fraction=fraction,unique_n=n,
        optimizer_updates=steps,rows_consumed=int(exposure.sum()),unique_rows_consumed=int(np.count_nonzero(exposure)),
        initialization_array_hash=init_hash,calibration=calibration,calibration_input_hash=r['calibration_input_hash'],
        subset_input_hash=subset['subset_input_hash'],selected_update=best_step,selected_VALID_score=best_score,
        selected_arrays_sha256=sha(dest/'SELECTED_SKIP_ICNN32.npz'),power4_arrays_sha256=sha(dest/'POWER4_RELATIVE.npz'),
        final_iterate_sha256=sha(dest/'FINAL_ITERATE_SKIP_ICNN32.npz'),final_iterate_optimizer_updates=steps,
        final_iterate_evaluation_policy='Diagnostic final iterate only; selected fixed-VALID checkpoint governs evaluation',
        optimizer_seconds=optimizer_seconds,validation_seconds=validation_seconds,elapsed_seconds=time.time()-started,
        sampling=probmeta,model_path=str(dest),diagnostic_only_not_national=True)
    result.update(readiness_fields(dest,fit_status=status))
    write_json(dest/'RESULT.json',result);return result


def study(ctx):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    out=ctx.root/'learning'
    ctx.bind(out/'PREPARE.json','FROZEN_LEARNING_PREPARATION_HANDOFF')
    ctx.bind(out/'LEARNING_CLOCK.json','ORIGINAL_SHARED_LEARNING_STAGE_CLOCK')
    prepared=read_json(out/'PREPARE.json');clock=read_json(out/'LEARNING_CLOCK.json')
    if clock['cap_seconds']!=14400 or not ctx.c['start_epoch']<=clock['stage_start_epoch']<=ctx.started:
        raise ValueError('Invalid original learning-stage clock')
    if clock['stage_deadline_epoch']!=min(clock['stage_start_epoch']+14400,ctx.c['compute_cutoff_epoch']):
        raise ValueError('Learning-stage deadline changed or reset')
    if (out/'runtime_pilot.csv').exists() or (out/'STUDY_FREEZE.json').exists() or (out/'FIT_RESULTS.json').exists():
        raise FileExistsError('Learning study already started; explicit remaining-budget recovery must preserve prior outputs')
    deadline=min(clock['stage_deadline_epoch'],ctx.c['compute_cutoff_epoch'],ctx.deadline or np.inf)
    plan=pilot(ctx,prepared,deadline);selected=set(plan['selected_region_ids']);results=[]
    prepared_rows=[r for r in prepared['prepared_regions'] if r['region_id'] in selected]
    for failed in plan['failed_preparation_regions']:
        for fraction in plan['fractions']:
            for replicate in plan['seeds']:
                results.append(dict(failed,seed=replicate,fraction=fraction,model_training_executed=False))
    # Interleave strata and fractions so a limit is visible across the whole design.
    prepared_rows.sort(key=lambda r:(r['rank_within_stratum'],plan['selected_region_ids'].index(r['region_id'])))
    for fraction in plan['fractions']:
        for replicate in plan['seeds']:
            for r in prepared_rows:
                if time.time()>=plan['training_deadline_epoch']:
                    results.append(dict(status='NOT_RUN_BUDGET',region_id=r['region_id'],stratum=r['stratum'],seed=replicate,fraction=fraction));continue
                try:result=fit(ctx,r,replicate,fraction,plan)
                except Exception as exc:
                    result=dict(status='FAILED',region_id=r['region_id'],stratum=r['stratum'],seed=replicate,fraction=fraction,
                        error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
                    dest=out/'regions'/r['region_id']/'fits'/f'seed_{replicate}_fraction_{round(fraction*100):03d}'
                    result.update(mark_neural_failure(dest,type(exc).__name__))
                    checkpoint=dest/'CHECKPOINT.json'
                    if checkpoint.exists():result.update(last_saved_checkpoint=read_json(checkpoint),retained_checkpoint_path=str(dest),
                        checkpoint_scope='Diagnostic partial work retained; failed fit is not admitted as a completed fit')
                    if dest.exists():write_json(dest/'FAILURE.json',result)
                results.append(result);write_json(out/'FIT_RESULTS.json',results)
    write_json(out/'FIT_RESULTS.json',results)
    freeze=dict(status='EXPERIMENT_MODELS_FROZEN',results=results,frozen_epoch=time.time(),study_plan_sha256=sha(out/'STUDY_FREEZE.json'),
        native_national_registries_unchanged=True,stage_deadline_epoch=deadline)
    write_json(out/'ALL_EXPERIMENT_MODELS_FROZEN.json',freeze)
    from .learning_evaluate import evaluate
    evaluation=evaluate(ctx,plan,prepared_rows,results,deadline)
    missing=[]
    for failed in plan['failed_preparation_regions']:
        for cohort in ('NEW_OPERATING','NEW_STRESS'):
            for slot in range(8):missing.append(dict(region_id=failed['region_id'],stratum=failed['stratum'],cohort=cohort,slot=slot,
                status='NOT_RUN_PREPARATION_UNAVAILABLE',preparation_status=failed['status'],quality=None,input_hash=None,input_generated=False))
    if missing:
        coverage=pd.read_csv(out/'new_label_coverage.csv')
        pd.concat([coverage,pd.DataFrame(missing)],ignore_index=True).to_csv(out/'new_label_coverage.csv',index=False,float_format='%.17g')
        policy=pd.read_csv(out/'fresh_evaluation_planned_coverage.csv')
        pd.concat([policy,pd.DataFrame(missing)[['region_id','stratum','cohort','slot']]],ignore_index=True).to_csv(out/'fresh_evaluation_planned_coverage.csv',index=False)
        evaluation.update(planned_regions=len(selected),failed_preparation_regions=len(plan['failed_preparation_regions']),
            fresh_planned=len(selected)*16,coverage_records_including_failed_preparation=len(coverage)+len(missing))
        write_json(out/'EVALUATION_RESULT.json',evaluation)
    all_complete=bool(results) and all(r['status']=='COMPLETE' for r in results) and evaluation['status']=='EXECUTED'
    return dict(status='EXECUTED' if all_complete else 'PARTIAL_LEARNING_STUDY',fits=len(results),complete=sum(r['status']=='COMPLETE' for r in results),
        partial=sum(r['status']=='TIME_LIMIT' for r in results),failed=sum(r['status']=='FAILED' for r in results),evaluation=evaluation)
