"""Post-freeze diagnostic labels and explicit reused/fresh learning-curve evaluation."""
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,write_json,read_npz,save_npz,sha
from partC_rev1_src.phase6.common import input_hash
from partC_rev1_src.phase6.storage import Store
from partC_rev1_src.phase67.sampling import bounded,operating_mask
from .learning_data import RegionalSources,seed
from .learning import error_rows
from .learning_artifacts import ready_models

COVERAGE_COLUMNS=['region_id','stratum','cohort','slot','status','quality','input_hash']
SUMMARY_COLUMNS=['region_id','stratum','cohort','family','seed','fraction','unique_n','status','fit_status',
    'n_evaluation_rows','n_coordinate_observations','MARE_equal_pairs','MARE_LOW_pair_weighted']


def new_inputs(ctx,source,union,destination):
    desc,arrays,pairs,scales=source.source.local(union.rid)
    low=union.s['low'];src=pairs.source_terminal_index.to_numpy();tgt=pairs.target_terminal_index.to_numpy()
    seen=union.ownership_hashes();rows=[];xs=[]
    for cohort in ('NEW_OPERATING','NEW_STRESS'):
        for slot in range(8):
            family=['RADIAL','PAIR','SOURCE','TARGET','SPARSE'][slot%5] if cohort=='NEW_OPERATING' else 'PREFROZEN_RADIAL_2P25_TO_4_LOW'
            for redraw in range(9):
                rng=np.random.default_rng(seed('N8R4_POST_MODEL_FREEZE',union.rid,cohort,slot,redraw))
                u=bounded(low,src,tgt,rng,family) if cohort=='NEW_OPERATING' else low*rng.uniform(2.25,4.)
                ih=input_hash(desc['model_hash'],desc['support_hash'],u)
                if ih not in seen:break
            else:raise ValueError('Unable to obtain distinct post-freeze diagnostic input')
            seen.add(ih);xs.append(u)
            op=bool(operating_mask(u[None,:],low)[0])
            if op!=(cohort=='NEW_OPERATING'):raise ValueError('New evaluation regime disagrees with frozen policy')
            rows.append(dict(input_hash=ih,input_index=len(xs)-1,cohort=cohort,slot=slot,family=family,redraw=redraw,
                native_support_hash=desc['support_hash'],model_hash=desc['model_hash']))
    destination.mkdir(parents=True,exist_ok=False)
    save_npz(destination/'frozen_inputs.npz',dict(u=np.stack(xs)))
    pd.DataFrame(rows).to_csv(destination/'frozen_rows.csv',index=False)
    write_json(destination/'FROZEN.json',dict(status='FROZEN_INPUTS_BEFORE_LABELS',frozen_epoch=time.time(),
        all_experiment_models_frozen_sha256=sha(ctx.root/'learning/ALL_EXPERIMENT_MODELS_FROZEN.json'),
        input_sha256=sha(destination/'frozen_inputs.npz'),row_sha256=sha(destination/'frozen_rows.csv'),
        source_ownership_collision_count=0,source_weights_unchanged=True))
    return desc,arrays,scales,np.stack(xs),rows


def _evaluate(ctx,plan,prepared,results,deadline):
    out=ctx.root/'learning';source=RegionalSources(ctx);all_coverage=[];all_summary=[];files=[]
    try:
        for ordinal,r in enumerate(prepared):
            ctx.guard(deadline=deadline);rid=r['region_id'];union=source.union(rid)
            cohorts=source.evaluation(union)
            region_path=out/'regions'/rid;dest=region_path/'post_freeze_evaluation'
            desc,arrays,scales,xs,planned=new_inputs(ctx,source,union,dest)
            store=Store(dest,desc,arrays,namespace='certified_labels')
            new_fields={};attempts=[]
            for i,row in enumerate(planned):
                # Keep a strict shared stage deadline and a fair maximum per fresh label.
                remaining_points=(len(prepared)-ordinal-1)*16+len(planned)-i
                remaining=max(0.,deadline-time.time()-120)
                cap=min(30.,remaining/max(remaining_points,1))
                if cap<.5:
                    attempts.append(dict(**row,status='NOT_RUN_STAGE_BUDGET',quality=None))
                    write_json(dest/'ATTEMPTS.json',attempts)
                    pd.DataFrame(all_coverage+[dict(region_id=rid,stratum=r['stratum'],**item) for item in attempts]).to_csv(
                        out/'new_label_coverage.csv',index=False,float_format='%.17g')
                    continue
                ctx.guard(deadline=deadline);end=min(deadline-60,time.time()+cap)
                metadata=dict(phase_id='N8',step_id='N8R4_NUM',run_id=ctx.c['run_id'],split=row['cohort'],
                    family=row['family'],jobid='N8R4_LEARNING_EVALUATION',regime=row['cohort'],
                    input_plan_sha256=sha(dest/'frozen_rows.csv'),experiment_models_already_frozen=True,
                    source_model_unchanged=True,capacity_already_applied_once=True,target_gap=1e-5)
                try:
                    fresh=None
                    label=store.oracle.solve(xs[i],target=1e-5,max_iter=2000,seconds=cap,polish_sweeps=3,
                        stop=lambda:time.time()>=end)
                    if label.quality in {'A','B','C'}:
                        receipt=store.publish(label,metadata,scales)
                        fresh,_=store.get(row['input_hash'],recertify=True)
                        disposition='CERTIFIED_AND_REOPENED'
                        if fresh.quality in {'A','B'}:
                            new_fields.setdefault(row['cohort'],[]).append((xs[i],fresh.tau,fresh.value_lower,fresh.value_upper,
                                fresh.roundoff_allowance,dict(row,quality=fresh.quality)))
                    else:
                        disposition='QUARANTINED';save_npz(dest/(row['input_hash']+'_quarantined.npz'),dict(u=label.demand,z=label.z,tau=label.tau))
                        write_json(dest/(row['input_hash']+'_quarantined.json'),label.scalars())
                    attempts.append(dict(**row,status=disposition,quality=fresh.quality if fresh is not None else label.quality,
                        solver_quality=label.quality,reopened_quality=fresh.quality if fresh is not None else None,
                        admitted_for_model_evaluation=fresh is not None and fresh.quality in {'A','B'},seconds=label.seconds,iterations=label.iterations,
                        relative_gap=label.relative_gap,allocated_cap_seconds=cap))
                except Exception as exc:
                    attempts.append(dict(**row,status='FAILED',error=repr(exc),quality=None,allocated_cap_seconds=cap))
                write_json(dest/'ATTEMPTS.json',attempts)
                pd.DataFrame(all_coverage+[dict(region_id=rid,stratum=r['stratum'],**item) for item in attempts]).to_csv(
                    out/'new_label_coverage.csv',index=False,float_format='%.17g')
            for cohort,items in new_fields.items():
                cohorts[cohort]=(*[np.asarray([x[k] for x in items]) for k in range(5)],pd.DataFrame([x[5] for x in items]))
            for row in attempts:all_coverage.append(dict(region_id=rid,stratum=r['stratum'],**row))
            fit_results=[x for x in results if x['region_id']==rid and x.get('evaluation_artifact_path')]
            region_rows=[]
            for fit in fit_results:
                for family,kind,arr,meta in ready_models(ctx,fit,union.s['c0']):
                    for cohort,fields in cohorts.items():
                        ctx.guard(deadline=deadline)
                        a=dict(zip(['u','t','L','U','eps'],fields[:5]));a['low']=union.s['low']
                        try:
                            frame=error_rows(kind,arr,a,fields[5],rid,cohort,stratum=r['stratum'],family=family,
                                seed=fit['seed'],fraction=fit['fraction'],unique_n=fit['unique_n'],
                                **meta)
                            region_rows.append(frame)
                            stats=frame[['MARE_equal_pairs','MARE_LOW_pair_weighted','MAE_seconds','MSE_seconds2','bias_seconds','interval_violation_relative']].mean().to_dict()
                            all_summary.append(dict(region_id=rid,stratum=r['stratum'],cohort=cohort,family=family,seed=fit['seed'],
                                fraction=fit['fraction'],unique_n=fit['unique_n'],n_evaluation_rows=len(frame),
                                n_coordinate_observations=int(frame.n_coordinates.sum()),negative_coordinates=int(frame.negative_coordinates.sum()),
                                status='MEASURED',**stats,**meta))
                        except FloatingPointError as exc:
                            all_summary.append(dict(region_id=rid,stratum=r['stratum'],cohort=cohort,family=family,seed=fit['seed'],
                                fraction=fit['fraction'],unique_n=fit['unique_n'],status='NONFINITE_DIAGNOSTIC_PREDICTION',error=str(exc),**meta))
                        # Preserve every completed cohort before the next guard, model,
                        # or label failure can interrupt the region.
                        if region_rows:
                            p=dest/'row_level_errors.csv'
                            pd.concat(region_rows,ignore_index=True).to_csv(p,index=False,float_format='%.17g')
                            if str(p) not in files:files.append(str(p))
                        pd.DataFrame(all_summary).to_csv(out/'learning_curve_summary.csv',index=False,float_format='%.17g')
            if region_rows:
                p=dest/'row_level_errors.csv';pd.concat(region_rows,ignore_index=True).to_csv(p,index=False,float_format='%.17g')
                if str(p) not in files:files.append(str(p))
            source.flush_bindings()
            pd.DataFrame(all_coverage).to_csv(out/'new_label_coverage.csv',index=False,float_format='%.17g')
            (pd.DataFrame(all_summary) if all_summary else pd.DataFrame(columns=SUMMARY_COLUMNS)).to_csv(out/'learning_curve_summary.csv',index=False,float_format='%.17g')
        result=dict(status='EXECUTED',regions=len(prepared),fresh_planned=len(prepared)*16,fresh_attempt_records=len(all_coverage),
            fresh_AB=sum(x.get('quality') in {'A','B'} for x in all_coverage),row_level_error_files=files,
            source_portfolios_mutated=False,learning_causality='Fixed-update, shared-calibration policy; deterministic extreme strata; finite seeds and reused heldouts disclosed')
        write_json(out/'EVALUATION_RESULT.json',result);return result
    finally:source.close()


def evaluate(ctx,plan,prepared,results,deadline):
    """Finalize explicit missing coverage even when the stage deadline interrupts work."""
    out=ctx.root/'learning'
    policy=[dict(region_id=r['region_id'],stratum=r['stratum'],cohort=cohort,slot=slot)
        for r in prepared for cohort in ('NEW_OPERATING','NEW_STRESS') for slot in range(8)]
    pd.DataFrame(policy,columns=['region_id','stratum','cohort','slot']).to_csv(out/'fresh_evaluation_planned_coverage.csv',index=False)
    pd.DataFrame(columns=COVERAGE_COLUMNS).to_csv(out/'new_label_coverage.csv',index=False)
    pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(out/'learning_curve_summary.csv',index=False)
    if not prepared:
        result=dict(status='NO_PREPARED_REGIONS',regions=0,fresh_planned=0,fresh_attempt_records=0,fresh_AB=0,
            row_level_error_files=[],source_portfolios_mutated=False,missing_labels_not_imputed=True)
        write_json(out/'EVALUATION_RESULT.json',result);return result
    try:return _evaluate(ctx,plan,prepared,results,deadline-120)
    except Exception as exc:
        records=[]
        for r in prepared:
            d=out/'regions'/r['region_id']/'post_freeze_evaluation'
            existing=read_json(d/'ATTEMPTS.json') if (d/'ATTEMPTS.json').exists() else []
            done={(x['cohort'],x['slot']):x for x in existing}
            frozen=pd.read_csv(d/'frozen_rows.csv').to_dict('records') if (d/'frozen_rows.csv').exists() else []
            named={(x['cohort'],x['slot']):x for x in frozen}
            for cohort in ('NEW_OPERATING','NEW_STRESS'):
                for slot in range(8):
                    key=(cohort,slot)
                    if key in done:item=done[key]
                    else:
                        item=dict(named.get(key,dict(cohort=cohort,slot=slot,input_hash=None)),
                            status='NOT_RUN_STAGE_BUDGET' if isinstance(exc,TimeoutError) else 'NOT_RUN_EVALUATION_INTERRUPTED',
                            quality=None,input_generated=key in named,interruption_type=type(exc).__name__)
                    records.append(dict(region_id=r['region_id'],stratum=r['stratum'],**item))
        pd.DataFrame(records).to_csv(out/'new_label_coverage.csv',index=False,float_format='%.17g')
        files=[str(p) for p in (out/'regions').glob('*/post_freeze_evaluation/row_level_errors.csv')]
        result=dict(status='PARTIAL_EVALUATION',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
            regions=len(prepared),fresh_planned=len(policy),fresh_attempt_records=len(records),
            fresh_AB=sum(x.get('quality') in {'A','B'} for x in records),row_level_error_files=files,
            source_portfolios_mutated=False,missing_labels_not_imputed=True)
        write_json(out/'EVALUATION_RESULT.json',result)
        return result
