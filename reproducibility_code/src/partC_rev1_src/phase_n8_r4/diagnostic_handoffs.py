"""Bounded diagnostic tables retain their complete planned coverage on interruption."""
from pathlib import Path
import traceback
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,write_json,ResourceLimit


def _merge(plans,keys,observed):
    complete=[]
    for plan in plans:
        matches=[row for row in observed if all(row.get(k)==plan[k] for k in keys)]
        if not matches and 'track' in keys:
            matches=[row for row in observed if 'track' not in row and row.get('scenario')==plan['scenario']]
        if len(matches)>1:raise ValueError('Duplicate diagnostic coverage disposition')
        complete.append(dict(plan,**{k:v for k,v in matches[0].items() if k not in plan}) if matches else dict(plan,status='NOT_STARTED'))
    return complete


def progress(dst,ledger_name,observed,*,active=None):
    frozen=read_json(Path(dst)/'PLANNED_COVERAGE.json')
    rows=list(observed)
    if active is not None:rows.append(dict(active,status='RUNNING'))
    write_json(Path(dst)/ledger_name,_merge(frozen['plans'],frozen['keys'],rows))


def bounded(ctx,args,folder,plans,keys,ledger_name,schemas,operation,*,reserve=120):
    dst=ctx.root/folder
    protected=[dst/ledger_name,dst/'RESULT.json']+[dst/name for name in schemas]
    if any(p.exists() for p in protected):
        raise FileExistsError('Diagnostic already started; preserve its outputs before explicit remaining-budget recovery')
    dst.mkdir(exist_ok=True)
    identities=[tuple(row[k] for k in keys) for row in plans]
    if not plans or len(set(identities))!=len(plans):raise ValueError('Empty or duplicate diagnostic plan')
    for name,columns in schemas.items():pd.DataFrame(columns=columns).to_csv(dst/name,index=False)
    write_json(dst/ledger_name,[dict(row,status='NOT_STARTED') for row in plans])
    write_json(dst/'PLANNED_COVERAGE.json',dict(plans=plans,keys=keys,job_id=args.job_id,
        finalization_reserve_seconds=reserve,unvisited_cases_are_not_successes=True))
    old_deadline=ctx.deadline
    ctx.deadline=min(ctx.c['compute_cutoff_epoch'],old_deadline or ctx.c['compute_cutoff_epoch'])-reserve
    error=None;result=dict(status='INTERRUPTED_BEFORE_RETURN')
    try:result=operation(ctx,args)
    except Exception as exc:
        error=exc
        result=dict(status='PARTIAL_DIAGNOSTIC' if isinstance(exc,(TimeoutError,ResourceLimit)) else 'FAIL',
            error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        ctx.deadline=old_deadline
        observed=read_json(dst/ledger_name)
        complete=_merge(plans,keys,observed)
        for row in complete:
            if row['status']=='NOT_STARTED':
                row['status']='NOT_RUN_BUDGET' if isinstance(error,TimeoutError) else 'NOT_RUN_RESOURCE' if isinstance(error,ResourceLimit) else 'NOT_RUN_INTERRUPTED' if error else 'NOT_PRODUCED'
            elif row['status']=='RUNNING':
                row['status']='INTERRUPTED_BUDGET' if isinstance(error,TimeoutError) else 'INTERRUPTED_RESOURCE' if isinstance(error,ResourceLimit) else 'FAILED_DIAGNOSTIC'
                row['error_type']=type(error).__name__ if error else 'INCOMPLETE_RETURN'
                row['error']=str(error) if error else 'Active unit did not receive a terminal disposition'
        write_json(dst/ledger_name,complete)
        measured=sum(row['status'] in {'MEASURED','PATHS_MEASURED'} for row in complete)
        if measured!=len(plans) and result.get('status')=='EXECUTED':result['status']='PARTIAL_DIAGNOSTIC'
        result.update(planned_units=len(plans),measured_units=measured,missing_units=len(plans)-measured,
            coverage_ledger=ledger_name,missing_observations_not_imputed=True)
        write_json(dst/'RESULT.json',result)
    return result
