"""Numerical admission and one-way model/portfolio freeze before new TEST targets."""
from __future__ import annotations
import copy,time,math
from pathlib import Path
from collections import Counter
from .campaign import binding,collect_records
from .labels import new_json,require_selection_freeze
from ..pacnhn_v1.control import read_json,sha


def apply_admissions(records,reports):
    """A model SHA alone is insufficient: prove its exact region/order association."""
    result=[]
    for source in records:
        row=copy.deepcopy(source);report=reports.get(row['model_sha256'])
        if report and report.get('model_sha256')!=row['model_sha256']:
            raise ValueError('Admission report refers to a different model')
        expected={k:row[k] for k in ('suite_id','region_index','family','graph_sha256','pair_keys_identity','model_path','model_sha256')}
        associations=[] if report is None else report.get('record_bindings',[])
        bound=any(all(value.get(k)==v for k,v in expected.items()) for value in associations)
        status=report.get('status','NUMERIC_INCONCLUSIVE') if report else 'NUMERIC_INCONCLUSIVE'
        if status=='PASS' and not bound:raise ValueError('Property PASS lacks exact model/region/native-order binding')
        row['property_status']=status
        row['property_model_sha256']=row['model_sha256']
        if report:
            row['property_report']={'path':report['report_path'],'sha256':report['report_sha256']}
            row['property_record_bound']=bound
        else:row['property_record_bound']=False
        result.append(row)
    return result


def authenticate_admissions(ctx,reports):
    for digest,report in reports.items():
        if report.get('model_sha256')!=digest:raise ValueError('Admission map/model identity differs')
        path=Path(report['report_path']).resolve(strict=True)
        if not path.is_relative_to(ctx.root/'admission'):raise ValueError('Foreign property report')
        ctx.bind(path,'EXACT_MODEL_PROPERTY_REPORT',report['report_sha256'])
        saved=read_json(path)
        if saved!={k:v for k,v in report.items() if k not in {'report_path','report_sha256'}}:
            raise ValueError('Admission transport metadata differs from saved report')


def validate_ablation_dispositions(ctx,frozen,plan,plan_sha):
    from .ablation import SCHEMA
    allowed={'AVAILABLE','FAILED','TIME_LIMIT','UNATTEMPTED_TIME_LIMIT'}
    if (frozen.get('schema')!=SCHEMA or frozen.get('status')!='ALL_160_DISPOSITIONS_FROZEN'
            or frozen.get('phase_id')!=ctx.c['phase_id'] or frozen.get('run_id')!=ctx.c['run_id']
            or frozen.get('plan_sha256')!=plan_sha or frozen.get('total')!=160 or frozen.get('TEST_used') is not False):
        raise ValueError('Invalid or foreign ablation freeze')
    rows=frozen.get('outcomes',[]);tasks=plan.get('tasks',[])
    if (len(rows)!=160 or len(tasks)!=160 or len({t['task_id'] for t in tasks})!=160
            or [r.get('task') for r in rows]!=tasks or any(r.get('status') not in allowed for r in rows)):
        raise ValueError('Ablation needs the exact160 distinct terminal tasks')
    count=sum(r['status']=='AVAILABLE' for r in rows)
    if frozen.get('completed')!=count or frozen.get('ablation_complete') is not (count==160):
        raise ValueError('Ablation completion claim differs from terminal outcomes')
    epoch=frozen.get('frozen_epoch')
    if not isinstance(epoch,(int,float)) or not math.isfinite(epoch) or epoch>time.time():
        raise ValueError('Invalid ablation freeze chronology')


def run(ctx,*,deadline):
    from . import inheritance
    from .numerical_admission import audit_records
    from .portfolio import build_portfolios
    ctx.require_action('portfolio');ctx.guard(deadline=deadline)
    for rel in ('training/MODEL_SELECTION_FREEZE.json','training/REGISTRY_FREEZE.json','holdout/TEST_OPENED.json'):
        if (ctx.root/rel).exists():raise FileExistsError('Selection has already frozen or TEST opened')
    ablation=ctx.root/'ablation/FIT_FREEZE.json'
    data=read_json(ablation)
    from .ablation import _read_plan
    plan,plan_path=_read_plan(ctx)
    validate_ablation_dispositions(ctx,data,plan,sha(plan_path))
    ctx.bind(ablation,'FROZEN_ALL160_ABLATION_DISPOSITIONS')
    for row in data['outcomes']:
        if row.get('result_path'):
            ctx.bind(row['result_path'],'PRE_TEST_ABLATION_FIT_RECEIPT',row['result_sha256'])
            saved=read_json(row['result_path'])
            if saved.get('task')!=row['task'] or saved.get('status')!=row['status']:
                raise ValueError('Ablation terminal disposition differs from saved result')
            if row['status']=='AVAILABLE':
                model=saved['model'];ctx.bind(model['path'],'PRE_TEST_ABLATION_SELECTED_MODEL',model['sha256'])
        elif row['status']=='AVAILABLE':raise ValueError('Available ablation model lacks fit evidence')
    records,ledger=collect_records(ctx)
    folder=ctx.root/'selection'/str(time.time_ns());folder.mkdir(parents=True,exist_ok=False)
    new_json(folder/'INTENT.json',dict(phase_id=ctx.c['phase_id'],step_id='ADMIT_AND_FREEZE',run_id=ctx.c['run_id'],
        config=binding(ctx.path),started_epoch=time.time(),deadline_epoch=deadline,tasks=[dict(suite_id=r['suite_id'],region_index=r['region_index'],family=r['family'],model_sha256=r['model_sha256']) for r in records],
        expected_outputs=['property disposition for each exact model','fixed20 suite/method portfolios','single pre-TEST selection freeze'],
        acceptance_tests=['exact graph/order/model admission','same-current-VALID baseline comparison','TRAIN strictly below1 for learned primary','no TEST selection'],
        inherited_selection_policy='ONLY_MISSING_REGION_FAMILY_WITH_EXACT_OLD_FINAL_CURRENT_COHORT_PROOF',
        inheritance_source=binding(inheritance.__file__),resources=ctx.c['resources']))
    records,inherited=inheritance.inherit_selected(ctx,records,deadline=min(deadline-60,time.time()+600),max_workers=4)
    new_json(folder/'INHERITANCE.json',inherited)
    by_region={}
    for row in inherited['dispositions']:by_region.setdefault(row['region_key'],[]).append(row)
    for row in ledger:row['selected_only_inheritance']=by_region.get(row['region_key'],[])
    new_json(folder/'PRE_ADMISSION_RECORDS.json',records);new_json(folder/'REGION_DISPOSITIONS.json',ledger)
    reports=audit_records(ctx,records,deadline=deadline-10)
    authenticate_admissions(ctx,reports)
    normalized=apply_admissions(records,reports)
    # Never silently replace the record source after the portfolio refers to it.
    target=ctx.root/'training/NORMALIZED_RECORDS.json';new_json(target,normalized)
    portfolios=build_portfolios(ctx,normalized,records_binding=binding(target))
    created=time.time()
    freeze=dict(schema='RECOVERY_GLOBAL_MODEL_SELECTION_FREEZE_V1',status='FROZEN',phase_id=ctx.c['phase_id'],run_id=ctx.c['run_id'],
        created_epoch=created,test_targets_opened=False,config_path=str(ctx.path),config_sha256=ctx.config_sha256,
        holdout_input_freeze=ctx.c['recovery']['holdout_input_freeze'],ablation_fit_freeze=binding(ablation),model_records=binding(target),
        portfolio=binding(ctx.root/'portfolios/PORTFOLIOS.json'),
        selection_source_bindings={str(Path(__file__)):sha(__file__),str(Path(inheritance.__file__)):sha(inheritance.__file__)},
        selected_only_inheritance=binding(folder/'INHERITANCE.json'),
        admission_status_counts=dict(Counter(r['status'] for r in reports.values())),region_dispositions=binding(folder/'REGION_DISPOSITIONS.json'))
    freeze_path=ctx.root/'training/MODEL_SELECTION_FREEZE.json';new_json(freeze_path,freeze)
    require_selection_freeze(ctx,freeze_path,sha(freeze_path))
    result=dict(status='FROZEN_WITH_EXPLICIT_FALLBACK_PORTFOLIOS',selection=binding(freeze_path),available_records=len(normalized),
        admission_status_counts=freeze['admission_status_counts'],method_components=[dict(suite_id=p['suite_id'],method=p['effective_method'],counts=p['component_counts']) for p in portfolios['portfolios']],
        TEST_used=False,epoch=time.time(),completed_before_deadline=time.time()<=deadline)
    new_json(folder/'RESULT.json',result);ctx.flush_bindings();return result
