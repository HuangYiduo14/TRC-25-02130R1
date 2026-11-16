"""Atomic preparation handoffs and independently frozen diagnostic model readiness."""
from pathlib import Path
import time
import numpy as np
from partC_rev1_src.phase_n8.common import read_json,write_json,read_npz,save_npz,sha
from partC_rev1_src.phase67.model import weighted_power4_fit


def persist_preparation(out,selected,prepared,failures,work_deadline,*,final=False,budget_exhausted=False):
    """Every checkpoint accounts for the complete predeclared region docket."""
    completed={r['region_id'] for r in prepared}|{r['region_id'] for r in failures}
    pending=[dict(region_id=r['region_id'],stratum=r['stratum'],rank_within_stratum=r['rank_within_stratum'],
        status='NOT_RUN_PREPARATION_BUDGET' if budget_exhausted else 'NOT_PREPARED_AT_CHECKPOINT',
        substitution=False) for r in selected if r['region_id'] not in completed]
    unavailable=failures+pending
    result=dict(status='PASS' if len(prepared)==len(selected) else 'PARTIAL_PREPARATION',
        regions=len(prepared),selected_regions=len(selected),preparation_failures=unavailable,
        strata=['large','small','good_VALID','difficult_TRAIN_VALID'],fractions=[.1,.25,.5,.7,1.],
        updates_per_fit=4000,target_seeds=2,minimum_seeds=1,
        budget_pilot_policy='32 updates per selected region, runtime only; estimate target then 1 seed then top2/stratum; freeze before curve outcomes',
        budget_including_pilot_labels_training_seconds=14400,label_and_export_reserve_seconds=2400,
        statistical_scope='Deterministic size/difficulty strata, not a random sample of all regions',
        model_family='POWER4_SKIP_ICNN32; exact current two-layer32 softplus potential, robust_loss unchanged',
        national_portfolio_substitution=False,prepared_regions=prepared,prepared_epoch=time.time(),
        preparation_loop_finalized=final,preparation_complete=len(prepared)==len(selected),
        preparation_work_deadline_epoch=work_deadline,preparation_finalization_reserve_seconds=60,
        checkpoint_scope='Usable partial handoff; unavailable regions retain their actual failure or unstarted disposition')
    write_json(Path(out)/'PREPARE.json',result)
    write_json(Path(out)/'PREPARATION_PROGRESS.json',dict(prepared=prepared,failures=unavailable,
        preparation_loop_finalized=final))
    return result


def prepare_analytic_artifacts(dest,a,r,replicate,fraction,subset):
    """Freeze analytic artifacts before any neural initialization can fail."""
    dest=Path(dest)
    kinds={'FFTT':'FREEFLOW_LINEAR','POWER4':'NEW_POWER4_RELATIVE_NNLS','SKIP_ICNN32':'POWER4_SKIP_ICNN32'}
    record=dict(region_id=r['region_id'],stratum=r['stratum'],seed=replicate,fraction=fraction,
        unique_n=subset['unique_n'],subset_input_hash=subset['subset_input_hash'],
        train_file_sha256=r['train_file_sha256'],train_rows_sha256=r['train_rows_sha256'],
        calibration_input_hash=r['calibration_input_hash'],
        diagnostic_only_not_national=True,label_certification='Separate postfreeze A/B label checks; readiness is not label certification',
        families={family:dict(model_kind=kind,status='NOT_READY',evaluation_ready=False,
            arrays_file=None,arrays_sha256=None) for family,kind in kinds.items()})
    write_json(dest/'MODEL_READINESS.json',record)

    def ready(family,filename,arrays,basis):
        if any(not np.isfinite(np.asarray(v)).all() for v in arrays.values()):raise ValueError('Nonfinite analytic artifact')
        if not np.array_equal(arrays['c0'],a['c0']) or np.any(arrays['c0']<0):raise ValueError('Analytic free-flow identity/floor changed')
        if family=='POWER4' and (np.any(arrays['scale']<=0) or np.any(arrays['eta']<0)):
            raise ValueError('Invalid convex power4 scale/coefficient')
        save_npz(dest/filename,arrays)
        record['families'][family].update(status='READY_ANALYTIC',evaluation_ready=True,
            arrays_file=filename,arrays_sha256=sha(dest/filename),readiness_basis=basis)
        write_json(dest/'MODEL_READINESS.json',record)

    ready('FFTT','FFTT.npz',dict(c0=a['c0']),
        'Exact finite nonnegative prepared c0; analytic constant derivative; no model fitting')
    n=subset['unique_n']
    power=dict(c0=a['c0'],scale=a['scale'],eta=weighted_power4_fit(a['u'][:n],a['t'][:n],a['scale'],a['c0']))
    ready('POWER4','POWER4_RELATIVE.npz',power,
        'Relative NNLS on the declared nested TRAIN subset; finite nonnegative coefficients and positive common scale')
    return record


def finalize_neural_readiness(dest,selected_update,selected_VALID_score,fit_status):
    if fit_status not in {'COMPLETE','TIME_LIMIT'}:raise ValueError('Failed neural fit cannot be admitted')
    if not isinstance(selected_update,(int,np.integer)) or selected_update<0 or not np.isfinite(selected_VALID_score):
        raise ValueError('Missing finite validation selection evidence')
    dest=Path(dest);record=read_json(dest/'MODEL_READINESS.json');path=dest/'SELECTED_SKIP_ICNN32.npz'
    arrays=read_npz(path)
    if not arrays or any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite neural checkpoint arrays')
    checkpoint=read_json(dest/'CHECKPOINT.json')
    if checkpoint['selected_update']!=selected_update or checkpoint['selected_VALID_score']!=selected_VALID_score:
        raise ValueError('Neural selection does not match persisted validation checkpoint')
    if sha(path)!=checkpoint['selected_arrays_sha256']:
        raise ValueError('Selected arrays differ from validated checkpoint')
    record['families']['SKIP_ICNN32'].update(status='READY_SELECTED_CHECKPOINT',evaluation_ready=True,
        arrays_file=path.name,arrays_sha256=sha(path),selected_update=int(selected_update),selected_VALID_score=float(selected_VALID_score),
        validation_checkpoint_sha256=sha(dest/'CHECKPOINT.json'),parent_NN_fit_status=fit_status,
        readiness_basis='Finite fixed-VALID-selected checkpoint from a normally finalized complete/time-limited fit; negative costs remain observable')
    write_json(dest/'MODEL_READINESS.json',record)
    return record


def readiness_fields(dest,*,fit_status):
    """Copy declared readiness into the all-model freeze; never infer it from files."""
    dest=Path(dest);path=dest/'MODEL_READINESS.json';record=read_json(path)
    neural=record['families']['SKIP_ICNN32']
    if fit_status not in {'COMPLETE','TIME_LIMIT'} and neural['evaluation_ready']:
        neural.update(status='FAILED_NOT_ADMITTED',evaluation_ready=False,
            diagnostic_arrays_file=neural.get('arrays_file'),diagnostic_arrays_sha256=neural.get('arrays_sha256'),
            arrays_file=None,arrays_sha256=None,parent_NN_fit_status=fit_status)
        write_json(path,record)
    return dict(evaluation_artifact_path=str(dest),model_readiness_sha256=sha(path),
        model_readiness=record['families'],unique_n=record['unique_n'],subset_input_hash=record['subset_input_hash'])


def mark_neural_failure(dest,error_type):
    dest=Path(dest);path=dest/'MODEL_READINESS.json'
    if not path.exists():return {}
    record=read_json(path)
    neural=record['families']['SKIP_ICNN32']
    diagnostic=dict(diagnostic_arrays_file=neural.get('arrays_file'),diagnostic_arrays_sha256=neural.get('arrays_sha256'))
    record['families']['SKIP_ICNN32'].update(status='FAILED_NOT_ADMITTED',evaluation_ready=False,
        arrays_file=None,arrays_sha256=None,error_type=error_type,parent_NN_fit_status='FAILED',
        readiness_basis='Failed fit checkpoints are retained diagnostics only; no neural evaluation admission',**diagnostic)
    write_json(path,record)
    return readiness_fields(dest,fit_status='FAILED')


def ready_models(ctx,fit,source_c0):
    """Yield only explicitly frozen families, independently of parent NN success."""
    if not fit.get('evaluation_artifact_path'):return
    dest=Path(fit['evaluation_artifact_path']).resolve()
    if (ctx.root/'learning/regions').resolve() not in dest.parents:raise ValueError('Diagnostic artifact path outside the study')
    manifest=dest/'MODEL_READINESS.json'
    ctx.bind(manifest,'FROZEN_DIAGNOSTIC_FAMILY_READINESS',fit['model_readiness_sha256'])
    record=read_json(manifest)
    for key in ('region_id','stratum','seed','fraction','unique_n','subset_input_hash'):
        if record[key]!=fit[key]:raise ValueError('Diagnostic family identity/subset changed: '+key)
    if record['families']!=fit['model_readiness']:raise ValueError('Family readiness differs from all-model freeze')
    expected={'FFTT':('FREEFLOW_LINEAR','FFTT.npz','READY_ANALYTIC'),
        'POWER4':('NEW_POWER4_RELATIVE_NNLS','POWER4_RELATIVE.npz','READY_ANALYTIC'),
        'SKIP_ICNN32':('POWER4_SKIP_ICNN32','SELECTED_SKIP_ICNN32.npz','READY_SELECTED_CHECKPOINT')}
    if set(record['families'])!=set(expected):raise ValueError('Diagnostic model family population changed')
    for family,(kind,filename,status) in expected.items():
        readiness=record['families'][family]
        if not readiness['evaluation_ready']:continue
        if family=='SKIP_ICNN32' and fit['status'] not in {'COMPLETE','TIME_LIMIT'}:
            raise ValueError('Failed NN fit was incorrectly marked evaluation ready')
        if (readiness['model_kind'],readiness['arrays_file'],readiness['status'])!=(kind,filename,status):
            raise ValueError('Untyped or inconsistent diagnostic family readiness')
        ctx.bind(dest/filename,'FROZEN_DIAGNOSTIC_MODEL',readiness['arrays_sha256'])
        if family=='SKIP_ICNN32':
            ctx.bind(dest/'CHECKPOINT.json','FROZEN_DIAGNOSTIC_VALIDATION_CHECKPOINT',readiness['validation_checkpoint_sha256'])
        arrays=read_npz(dest/filename)
        if not np.array_equal(arrays['c0'],source_c0):raise ValueError('Diagnostic model/source free-flow identity changed')
        yield family,kind,arrays,dict(family_readiness_status=status,arrays_sha256=readiness['arrays_sha256'],
            fit_status=fit['status'] if family=='SKIP_ICNN32' else 'ANALYTIC_COMPLETE',
            parent_NN_fit_status=fit['status'],optimizer_updates=fit['optimizer_updates'] if family=='SKIP_ICNN32' else 0,
            selected_update=fit['selected_update'] if family=='SKIP_ICNN32' else 0)
