"""Fixed 160-cell sample-fraction and separate 32-cell enrichment computation.

No figure/table renderer. The original final manuscript fit functions are reused.
New inputs/labels/models come only from the current portable run.
"""
from pathlib import Path
import time
import numpy as np
from partC_rev1_src.pacnhn_v1.control import read_json,write_json,array_hash,sha
from partC_rev1_src.pacnhn_v1.regional import sample_requests,generate_labels,load_admitted_dataset
from partC_rev1_src.pacnhn_recovery24.ablation import nested_indices,subset_train
from partC_rev1_src.pacnhn_v1.training import _structural_zeros
from partC_rev1_src.pacnhn_v1.models import load_model
import frozen_fig5_training as frozen


def additional_training(ctx,record,train,valid,region,folder):
    from run_downstream import region_root,scale_for
    old=read_json(region_root(ctx,record)/'labels/REQUEST_INDEX.json')
    c=ctx.c['enrichment']
    requests=sample_requests(region,role='TRAIN',count=c['additional_train_per_region'],seed=c['seed'],
        start_index=4,existing_input_ids=set(old),reference_scale=scale_for(region,record))
    cfg=dict(phase_id='PACNHN_V1',run_id=ctx.c['run_id'],region_id=f'{record["suite_id"]}:{record["region_index"]:06d}',
        graph_path=record['path'],graph_sha256=record['sha256'],deadline_epoch=ctx.deadline,
        per_label_seconds=c['attempt_seconds'][0],max_iterations=20000,relative_gap_tolerance=1e-6,
        solver_method='CFW',retry_solver_method='CFW',cfw_reset_on_bound=True,retry_cfw_reset_on_bound=True,
        iterate_selection='BEST_CERTIFIED_RG',retry_iterate_selection='BEST_CERTIFIED_RG',max_attempts_per_input=len(c['attempt_seconds']))
    reports=[generate_labels(cfg,folder/'labels',requests=requests,guard=ctx.guard)]
    for seconds in c['attempt_seconds'][1:]:
        index=read_json(folder/'labels/REQUEST_INDEX.json')
        missing=[]
        for row in index.values():
            if row['selected_label'] is None:
                metadata=read_json(row['request_metadata'])
                with np.load(row['input_file'],allow_pickle=False) as z:missing.append(dict(metadata,u=z['u'].copy()))
        if not missing:break
        cfg['per_label_seconds']=seconds
        reports.append(generate_labels(cfg,folder/'labels',requests=missing,guard=ctx.guard))
    extension=load_admitted_dataset(folder/'labels',role='TRAIN')
    u=np.concatenate([train['u'],extension['u']],axis=0)
    tau=np.concatenate([train['tau'],extension['tau']],axis=0)
    if not np.array_equal(train['c0'],extension['c0']):raise ValueError('Extension c0 differs')
    merged=dict(train,u=u,tau=tau,records=train['records']+extension['records'],unique_rows=len(u),
        dataset_identity=array_hash(dict(u=u,tau=tau,c0=train['c0'])),train_cost_mean=float(tau.mean()))
    write_json(folder/'ADDITIONAL_LABEL_DISPOSITIONS.json',dict(requested=len(requests),admitted=len(extension['u']),
        missing=len(requests)-len(extension['u']),reports=reports,original_TRAIN_unchanged=True,VALID_TEST_unchanged=True))
    return merged


def run(ctx,*,enriched):
    from run_downstream import support_rows,resolve,load_graph,cohorts,load_test,metric
    _,records=support_rows(ctx)
    c=ctx.c['sample_fraction']
    metadata_path=resolve(c['cohort_metadata'])
    ctx.bind(metadata_path,'FROZEN_MANUSCRIPT_16_REGION_IDS')
    metadata=read_json(metadata_path)
    regions=metadata['regions']
    if len(regions)!=16 or len({r['region_key'] for r in regions})!=16 or any(r['suite_id']!='s03' for r in regions):
        raise ValueError('Exact 16-region declared cohort required')
    available={(r['suite_id'],r['region_index']):r for r in records}
    if any((r['suite_id'],r['region_index']) not in available for r in regions):
        raise ValueError('Frozen manuscript region absent from freshly regenerated support; never silently renumber')
    experiment='enrichment' if enriched else 'sample_fractions'
    root=ctx.root/'experiments'/experiment
    root.mkdir(parents=True,exist_ok=False)
    fractions=[1.] if enriched else list(c['fractions'])
    if not enriched and fractions!=[.1,.25,.5,.7,1.]:raise ValueError('Frozen five fractions required')
    tasks=[dict(region_key=r['region_key'],stratum=r['stratum'],fraction=f,family=family,
        task_id=f'{r["region_key"].replace(":","_")}_{int(f*100):03d}_{family}')
        for r in regions for f in fractions for family in ('ICNN','XGBOOST')]
    write_json(root/'PLAN.json',dict(tasks=tasks,seed=c['seed'],policy=frozen.POLICY,
        cohort_metadata_sha256=sha(metadata_path),TEST_used=False,experiment=experiment,
        source=frozen.binding(frozen.__file__)))
    outcomes=[]
    for chosen in regions:
        record=available[(chosen['suite_id'],chosen['region_index'])]
        region=load_graph(ctx,record)
        train,valid=cohorts(ctx,record)
        folder=root/chosen['region_key'].replace(':','_')
        folder.mkdir()
        if enriched:
            train=additional_training(ctx,record,train,valid,region,folder)
        indices=nested_indices([r['input_identity'] for r in train['records']],c['seed'],chosen['region_key'])
        for fraction in fractions:
            # Enrichment appends to the original natural order; baseline uses seeded prefixes.
            subset=train if enriched else subset_train(train,indices[fraction])
            for family in ('ICNN','XGBOOST'):
                ctx.guard()
                task=next(t for t in tasks if t['region_key']==chosen['region_key'] and t['fraction']==fraction and t['family']==family)
                dest=folder/task['task_id'];dest.mkdir()
                end=min(ctx.deadline-60,time.time()+c['per_fit_seconds'])
                if family=='ICNN':
                    result=frozen.fit_icnn(subset,valid,c['seed'],ctx.guard,end,dest)
                else:
                    result=frozen.fit_xgb(subset,valid,c['seed'],ctx.guard,end,dest,
                        _structural_zeros(region),'FROZEN_GRAPH_ZERO_COST_DIRECTED_REACHABILITY',threads=1)
                row=dict(task=task,result=result,current_pairs=record['pairs'],historical_pairs=chosen['pairs'],
                    TRAIN_rows=len(subset['u']),VALID_rows=len(valid['u']),TEST_used=False)
                outcomes.append(row)
                write_json(dest/'FIT_RESULT.json',row)
                write_json(root/'FIT_DISPOSITIONS.json',outcomes)
    # No TEST labels are read until every scheduled fit has an immutable selection.
    write_json(root/'SELECTION_FREEZE.json',dict(status='ALL_EXPERIMENT_SELECTIONS_FROZEN',
        outcomes_sha256=sha(root/'FIT_DISPOSITIONS.json'),no_more_training=True,TEST_used_for_selection=False))
    evaluations=[]
    for row in outcomes:
        key=row['task']['region_key'];sid,rid=key.split(':')
        record=available[(sid,int(rid))]
        dataset=load_test(ctx,record)
        model=row['result']['model']
        ctx.bind(model['path'],'EXPERIMENT_FROZEN_SELECTED_MODEL',model['sha256'])
        evaluations.append(dict(task=row['task'],fit_status=row['result']['status'],TEST=metric(load_model(model['path']),dataset,ctx)))
        write_json(root/'TEST_EVALUATIONS.json',evaluations)
    expected=32 if enriched else 160
    if len(outcomes)!=expected or len(evaluations)!=expected:
        raise ValueError('Incomplete planned experiment task coverage')
    if any(r['result']['status']!='AVAILABLE' for r in outcomes):
        raise ValueError('One or more fits are partial; true finite checkpoints and metrics preserved')
    return dict(status='COMPLETE',experiment=experiment,planned_fits=expected,completed=len(outcomes),evaluated=len(evaluations),
        TEST_used_for_selection=False,original_national_portfolios_unchanged=True)
