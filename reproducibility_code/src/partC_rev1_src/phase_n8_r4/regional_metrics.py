"""Fresh native predictions on now-seen certified E2 batches; no selection changes."""
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,read_npz,write_json,sha,digest
from partC_rev1_src.phase_n8.native import load_registry
from partC_rev1_src.phase67.data import FIELDS
from partC_rev1_src.phase6.common import input_hash
from .context import sources
from .safety import load_safe,REGISTRIES,ANALYTIC
from .learning import error_rows
from .learning_data import RegionalSources

SUMMARY_COLUMNS=['region_id','m_active','cohort','registry','model_kind','rows','coordinate_observations',
    'negative_coordinates','below_fftt_coordinates','minimum_cost_seconds','maximum_relative_error','status',
    'MARE_equal_pairs','MARE_LOW_pair_weighted','MAE_seconds','MSE_seconds2','bias_seconds','interval_violation_relative']
COVERAGE_COLUMNS=['region_id','m_active','cohort','status','source_rows','eligible_AB_rows','evaluated_rows','excluded_quality_rows']
PREDICTION_COLUMNS=['region_id','cohort','registry','status']


def run(ctx,args):
    if (ctx.root/'regional_metrics').exists():raise FileExistsError('Regional evaluation already has persisted outputs; preserve them for explicit recovery')
    original_deadline=ctx.deadline
    ctx.deadline=min(original_deadline or ctx.c['compute_cutoff_epoch'],ctx.c['compute_cutoff_epoch'])-120
    try:
        return _run(ctx,args)
    except Exception as exc:
        dst=ctx.root/'regional_metrics'
        if not dst.exists():raise
        result_path=dst/'RESULT.json'
        result=read_json(result_path) if result_path.exists() else dict(active_regions=1566,registries=8,
            evaluated_regions=0,missing_regions=1566,missing_active_coordinates=80316,row_observation_files=[],
            coverage_semantics='No completed source/prediction counts are inferred from absent evidence')
        result.update(status='PARTIAL_REGIONAL_EVALUATION',error_type=type(exc).__name__,error=str(exc),
            traceback=traceback.format_exc(),scientific_source_mutations=0)
        write_json(result_path,result)
        return result
    finally:
        ctx.deadline=original_deadline


def _run(ctx,args):
    dst=ctx.root/'regional_metrics';dst.mkdir(exist_ok=False)
    pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(dst/'region_registry_metrics.csv',index=False)
    pd.DataFrame(columns=COVERAGE_COLUMNS).to_csv(dst/'coverage.csv',index=False)
    pd.DataFrame(columns=PREDICTION_COLUMNS).to_csv(dst/'model_prediction_coverage.csv',index=False)
    write_json(dst/'M28_DISPOSITION.json',dict(status='NOT_EXECUTED',
        cohort='TRAIN_E2_SEEN_NOT_TEST',reason='Named diagnostic follows the bounded all-region evaluation; absent rows are not zero error'))
    src=sources(ctx);suite=src.suite('s03');models={}
    for safe,original in REGISTRIES.items():
        entries,reg=load_registry(src,suite,original);models[original]={e['region_id']:e for e in entries}
        entries,reg=load_safe(ctx,src,suite,safe);models[safe]={e['region_id']:e for e in entries}
    entries,reg=load_registry(src,suite,ANALYTIC);models[ANALYTIC]={e['region_id']:e for e in entries}
    pairs=suite['pairs'];groups=pairs.groupby('region_id',sort=False)
    models['FFTT']={rid:dict(model_kind='FREEFLOW_LINEAR',arrays=dict(c0=r.c0_seconds.to_numpy(float))) for rid,r in groups}
    low={rid:r.LOW_CURRENT.to_numpy(float) for rid,r in groups}
    idx=src.r1/'tables/exact_frozen_audit_batches.csv';src.bind(idx,'EXACT_NOW_SEEN_E2_BATCH_INDEX')
    index=pd.read_csv(idx);index=index.loc[index.role.isin(['E2_AUDIT_OPERATING','E2_AUDIT_STRESS'])]
    regional=RegionalSources(ctx)
    write_json(dst/'EVALUATION_FREEZE.json',dict(status='FROZEN',registries=list(models),
        source_index_sha256=sha(idx),evaluation_roles=sorted(index.role.unique())+['LOW_TEST','HIGH_PROJECTED'],
        old_pack_source_index_sha256=sha(regional.e2/'control/p7_source_manifest.parquet'),
        old_test_roles='REUSED_NOW_SEEN; canonical protected roles and A/B eligibility unchanged',
        experiment_models_excluded=True,new_national_selection=False,source_weights_changed=False))
    summary=[];coverage=[];parts=[];part_no=0;part_rows=0;files=[]
    def flush():
        nonlocal parts,part_no,part_rows
        if not parts:return
        p=dst/f'current_E2_row_observations_{part_no:03d}.csv'
        pd.concat(parts,ignore_index=True).to_csv(p,index=False,float_format='%.17g')
        files.append(dict(path=p.name,sha256=sha(p),rows=part_rows));parts=[];part_rows=0;part_no+=1
    try:
        for ordinal,(rid,native) in enumerate(groups):
            ctx.guard();region_index=index.loc[index.region_id.eq(rid)];cohorts={};excluded={};source_counts={}
            for row in region_index.itertuples(index=False):
                p=Path(row.manifest_path);ctx.bind(p,'CURRENT_CERTIFIED_NOW_SEEN_E2_BATCH',row.expected_sha256)
                manifest=read_json(p);ctx.bind(p.parent/'native.npz','CURRENT_E2_NATIVE_ARRAYS',manifest['native_sha256'])
                ctx.bind(p.parent/'rows.parquet','CURRENT_E2_ROW_IDENTITIES',manifest['rows_sha256'])
                a=read_npz(p.parent/'native.npz');rr=pd.read_parquet(p.parent/'rows.parquet')
                if manifest['support_hash']!=native.support_hash.iloc[0]:raise ValueError('Current region support mismatch')
                for i,ih in enumerate(rr.input_hash):
                    if input_hash(manifest['model_hash'],manifest['support_hash'],a[FIELDS[0]][i])!=ih:
                        raise ValueError('Current audit native input hash mismatch')
                take=np.flatnonzero(rr.quality.isin(['A','B']).to_numpy())
                rejected=len(rr)-len(take)
                if len(take)!=int(row.rows_AB) or rejected!=int(row.rows_nonAB):
                    raise ValueError('Frozen E2 quality denominators changed')
                source_counts[row.role]=source_counts.get(row.role,0)+len(rr)
                excluded[row.role]=excluded.get(row.role,0)+rejected
                if len(take):cohorts.setdefault(row.role,[]).append((*[a[k][take] for k in FIELDS],rr.iloc[take].reset_index(drop=True)))
            pack=regional.source.old_pack(rid)
            if pack.meta['source_pair_order_hash']!=digest(native.pair_id.tolist()) or pack.meta['source_descriptor']['support_hash']!=native.support_hash.iloc[0]:
                raise ValueError('Protected old-test native support/order mismatch')
            if not np.array_equal(pack.s['low'],low[rid]) or not np.array_equal(pack.s['c0'],native.c0_seconds.to_numpy(float)):
                raise ValueError('Protected old-test LOW or free-flow identity mismatch')
            for split in ('LOW_TEST','HIGH_PROJECTED'):
                take=pack.indices(split);rr=pack.rows.iloc[take].copy().reset_index(drop=True);aa=pack.fields(take)
                for k,ih in enumerate(rr.input_hash):
                    if input_hash(pack.meta['source_descriptor']['model_hash'],pack.meta['source_descriptor']['support_hash'],aa[0][k])!=ih:
                        raise ValueError('Protected old-test input hash mismatch')
                excluded[split]=int((pack.rows.split.eq(split)&~pack.rows.quality.isin(['A','B'])).sum())
                source_counts[split]=int(pack.rows.split.eq(split).sum())
                if source_counts[split]!=len(take)+excluded[split]:raise ValueError('Old-test quality denominators changed')
                if len(take):cohorts[split]=[(*aa,rr)]
            for cohort in ('E2_AUDIT_OPERATING','E2_AUDIT_STRESS','LOW_TEST','HIGH_PROJECTED'):
                items=cohorts.get(cohort,[])
                if not items:
                    coverage.append(dict(region_id=rid,m_active=len(native),cohort=cohort,status='NO_CERTIFIED_AB_ROWS',
                        source_rows=source_counts.get(cohort,0),eligible_AB_rows=0,evaluated_rows=0,
                        excluded_quality_rows=excluded.get(cohort,0)));continue
                aa=[np.concatenate([x[k] for x in items]) for k in range(5)]
                rr=pd.concat([x[5] for x in items],ignore_index=True)
                if rr.input_hash.duplicated().any():raise ValueError('Repeated batch input in same audit cohort')
                a=dict(zip(['u','t','L','U','eps'],aa));a['low']=low[rid]
                if source_counts[cohort]!=len(rr)+excluded[cohort]:raise ValueError('Cohort quality coverage is not exhaustive')
                coverage.append(dict(region_id=rid,m_active=len(native),cohort=cohort,status='CERTIFIED_AB_REOPENED',
                    source_rows=source_counts[cohort],eligible_AB_rows=len(rr),evaluated_rows=len(rr),
                    excluded_quality_rows=excluded[cohort]))
                for registry,entries in models.items():
                    e=entries[rid]
                    frame=error_rows(e['model_kind'],e['arrays'],a,rr,rid,'REUSED_NOW_SEEN_'+cohort,
                        registry=registry,model_kind=e['model_kind'])
                    frame['source_family']=rr.family.to_numpy();frame['source_quality']=rr.quality.to_numpy()
                    frame['source_split']=cohort
                    frame['native_support_hash']=native.support_hash.iloc[0]
                    parts.append(frame);part_rows+=len(frame)
                    means=frame[['MARE_equal_pairs','MARE_LOW_pair_weighted','MAE_seconds','MSE_seconds2','bias_seconds','interval_violation_relative']].mean().to_dict()
                    summary.append(dict(region_id=rid,m_active=len(native),cohort='REUSED_NOW_SEEN_'+cohort,registry=registry,
                        model_kind=e['model_kind'],rows=len(frame),coordinate_observations=int(frame.n_coordinates.sum()),
                        negative_coordinates=int(frame.negative_coordinates.sum()),below_fftt_coordinates=int(frame.below_fftt_coordinates.sum()),
                        minimum_cost_seconds=float(frame.minimum_cost_seconds.min()),maximum_relative_error=float(frame.maximum_relative_error.max()),
                        status='MEASURED',**means))
            if part_rows>=12000:flush()
            if ordinal%100==0:
                regional.flush_bindings()
                print('REGIONAL_CURRENT',ordinal+1,'of',len(groups),flush=True)
                pd.DataFrame(coverage,columns=COVERAGE_COLUMNS).to_csv(dst/'coverage.csv',index=False)
                pd.DataFrame(summary,columns=SUMMARY_COLUMNS).to_csv(dst/'region_registry_metrics.csv',index=False,float_format='%.17g')
        flush()
        pd.DataFrame(coverage,columns=COVERAGE_COLUMNS).to_csv(dst/'coverage.csv',index=False)
        pd.DataFrame(summary,columns=SUMMARY_COLUMNS).to_csv(dst/'region_registry_metrics.csv',index=False,float_format='%.17g')
    finally:
        flush()
        seen_coverage={(r['region_id'],r['cohort']) for r in coverage}
        for rid,native in groups:
            for cohort in ('E2_AUDIT_OPERATING','E2_AUDIT_STRESS','LOW_TEST','HIGH_PROJECTED'):
                if (rid,cohort) not in seen_coverage:
                    coverage.append(dict(region_id=rid,m_active=len(native),cohort=cohort,
                        status='NOT_REOPENED_EVALUATION_INTERRUPTED',evaluated_rows=0))
        measured={(r['region_id'],r['cohort'],r['registry']) for r in summary}
        model_coverage=[]
        for c in coverage:
            for registry in models:
                cohort='REUSED_NOW_SEEN_'+c['cohort']
                status='MEASURED' if (c['region_id'],cohort,registry) in measured else (
                    'NO_CERTIFIED_AB_ROWS' if c['status']=='NO_CERTIFIED_AB_ROWS' else 'NOT_PREDICTED_EVALUATION_INTERRUPTED')
                model_coverage.append(dict(region_id=c['region_id'],cohort=cohort,registry=registry,status=status))
        pd.DataFrame(coverage,columns=COVERAGE_COLUMNS).to_csv(dst/'coverage.csv',index=False)
        pd.DataFrame(summary,columns=SUMMARY_COLUMNS).to_csv(dst/'region_registry_metrics.csv',index=False,float_format='%.17g')
        pd.DataFrame(model_coverage,columns=PREDICTION_COLUMNS).to_csv(dst/'model_prediction_coverage.csv',index=False)
        source_coverage=pd.DataFrame(coverage,columns=COVERAGE_COLUMNS)
        e2coverage=source_coverage.loc[source_coverage.cohort.str.startswith('E2_AUDIT_')]
        source_per_region=e2coverage.groupby('region_id').evaluated_rows.sum()
        write_json(dst/'RESULT.json',dict(status='PARTIAL_PENDING_COMPLETE_EVALUATION_AND_M28',active_regions=len(groups),
            evaluated_regions=int(source_per_region.gt(0).sum()),missing_regions=int(source_per_region.eq(0).sum()),
            missing_active_coordinates=int(e2coverage.loc[e2coverage.region_id.isin(source_per_region.index[source_per_region.eq(0)])].drop_duplicates('region_id').m_active.sum()),
            registries=len(models),row_observation_files=files,now_seen_not_fresh_test=True,
            coverage_semantics='Source rows reopened: source_rows=eligible_AB_rows+excluded_quality_rows. evaluated_rows is a legacy alias for eligible_AB_rows, not completed predictions; completed registry-specific predictions separately in model_prediction_coverage.csv',
            scientific_source_mutations=0))
        regional.close()
    # The named m28 training case is a seen TRAIN diagnostic, never a fresh test.
    regional=RegionalSources(ctx)
    try:
        rid='partC_p2_region_0ce745b1cfa26af63aab';union=regional.union(rid);fields=union.fields('TRAIN_e2')
        if len(fields[0]):
            a=dict(zip(['u','t','L','U','eps'],fields[:5]));a['low']=union.s['low'];m28=[]
            for registry,entries in models.items():
                e=entries[rid];m28.append(error_rows(e['model_kind'],e['arrays'],a,fields[5],rid,'TRAIN_E2_SEEN_NOT_TEST',registry=registry,model_kind=e['model_kind']))
            pd.concat(m28,ignore_index=True).to_csv(dst/'m28_current_TRAIN_e2_rows.csv',index=False,float_format='%.17g')
            write_json(dst/'M28_DISPOSITION.json',dict(status='EXECUTED',cohort='TRAIN_E2_SEEN_NOT_TEST',
                source_rows=len(fields[0]),registries=len(models),
                observation_file='m28_current_TRAIN_e2_rows.csv',observation_sha256=sha(dst/'m28_current_TRAIN_e2_rows.csv')))
        else:write_json(dst/'M28_DISPOSITION.json',dict(status='NO_SOURCE_TRAIN_E2_ROWS',cohort='TRAIN_E2_SEEN_NOT_TEST'))
    finally:regional.close()
    frame=pd.DataFrame(coverage);e2=frame.loc[frame.cohort.str.startswith('E2_AUDIT_')]
    per=e2.groupby('region_id').evaluated_rows.sum()
    result=dict(status='EXECUTED',active_regions=len(groups),evaluated_regions=int(per.gt(0).sum()),
        missing_regions=int(per.eq(0).sum()),missing_active_coordinates=int(e2.loc[e2.region_id.isin(per.index[per.eq(0)])].drop_duplicates('region_id').m_active.sum()),
        E2_source_rows=int(e2.source_rows.sum()),E2_source_AB_rows=int(e2.eligible_AB_rows.sum()),
        E2_excluded_nonAB_rows=int(e2.excluded_quality_rows.sum()),
        all_cohort_source_AB_rows=int(frame.eligible_AB_rows.sum()),
        coverage_by_cohort=frame.groupby('cohort').agg(evaluated_rows=('evaluated_rows','sum'),evaluated_regions=('evaluated_rows',lambda v:int((v>0).sum()))).reset_index().to_dict('records'),
        registries=len(models),row_observation_files=files,
        now_seen_not_fresh_test=True,scientific_source_mutations=0)
    write_json(dst/'RESULT.json',result);return result
