"""Historical disjoint counters, bounded missing scope, and current complete job clocks."""
from pathlib import Path
import sqlite3
import json
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,write_json,sha
from .context import sources
from .stability import SUITE_DIRS


def union_seconds(intervals):
    total=0.;end=None
    for a,b in sorted(intervals):
        if b<a:raise ValueError('Reversed accounting interval')
        total+=max(0.,b-max(a,end)) if end is not None else b-a
        end=max(end,b) if end is not None else b
    return total


def run(ctx,args):
    src=sources(ctx);base=src.base;dst=ctx.root/'accounting';dst.mkdir(exist_ok=False)
    records=[];missing=[]
    def add(stage,identity,seconds,path,measure,**extra):
        if seconds is None or not np.isfinite(float(seconds)) or float(seconds)<0:raise ValueError('Invalid accounting timer')
        records.append(dict(stage=stage,record_id=identity,seconds=float(seconds),source_path=str(path),
            source_sha256=ctx.bind(path,'OFFLINE_ACCOUNTING_SOURCE'),measure=measure,
            CPU_seconds_estimated=False,**extra))
    p=ctx.repo.parent/'doe_solver_v3/partC_rev1_outputs/partC_rev1_preprocess_summary.csv'
    row=pd.read_csv(p).iloc[0]
    for item in str(row['runtime by stage']).split(';'):
        key,seconds=item.strip().split('=');add('PREPROCESSING',key,float(seconds),p,'DISJOINT_SEQUENTIAL_STAGE_WALL',
            inclusion='MEASURED_SUBTOTAL_NOT_COMPLETE_ELAPSED')
    missing.append(dict(stage='PREPROCESSING',status='INCOMPLETE',missing_scope='Capacity enrichment, earlier preparation, untimed bookkeeping and later repairs'))
    p=base/'phase_02/split_mapping_campaign/phase2_split_mapping_20260817_v1_run_001/portfolio/suite_manifest.csv'
    table=pd.read_csv(p);active=set(SUITE_DIRS.values())
    for row in table.loc[table.suite_id.isin(active)].itertuples(index=False):
        add('PHASE2_ACTIVE_SUITES',row.suite_id,row.suite_runtime_seconds,p,'PER_SUITE_ELAPSED',
            nested_leiden_seconds=float(row.leiden_seconds),nested_timer_not_added=True,structural_signature=row.structural_signature)
    p=src.p3/'tables/table_06_phase3_resource_usage.csv';table=pd.read_csv(p)
    for number,row in table.loc[table.resource_source.eq('cli_task')&table.suite_short_id.isin(ctx.c['active_suites'])].iterrows():
        add('PHASE3_ACTIVE_SUITE_TASKS',str(row.task_id)+'_row'+str(number),row.elapsed_seconds,p,'PARENT_TASK_ELAPSED',
            status=str(row.status),suite=str(row.suite_short_id),operation=str(row.operation),
            interpretation='s03 reuse/copy is not original construction; failed attempts retained')
    missing.append(dict(stage='PHASE3_ORIGINAL_S03_METHOD_SELECTION',status='NOT_ESTABLISHED_AS_WINNER_ONLY_COST',
        missing_scope='Multi-candidate 3A search and winner construction cannot be attributed by copying whole campaign timer; s03 Phase3 reuse time is not construction'))
    intervals=[]
    for p in sorted((src.p5/'stage_receipts').glob('C[0-7].json')):
        r=read_json(p);intervals.append((r['started_unix'],r['ended_unix']))
        add('PHASE5_RUNTIME_CAMPAIGN',r['stage_id'],r['elapsed_seconds'],p,'PARENT_STAGE_ELAPSED',
            started_epoch=r['started_unix'],ended_epoch=r['ended_unix'],child_shards_excluded=True,status=r['status'])
    write_json(dst/'PHASE5_INTERVAL_UNION.json',dict(seconds=union_seconds(intervals),intervals=len(intervals),
        scope='Compilation, validation and reporting campaign stages; inspect stage names; child shard timers not added'))
    # One final cumulative counter per region and distinct model generation.
    p7=base/'phase_07/s03_active_8h_v1/run_001'
    for ordinal,p in enumerate(sorted((p7/'regions').glob('*/status.json'))):
        if ordinal%100==0:ctx.guard()
        r=read_json(p)
        for field in ('train_seconds','validation_seconds','load_seconds','pack_seconds','evaluation_seconds'):
            if field in r:add('P7_ORIGINAL_'+field.upper(),p.parent.name+'_'+field,r[field],p,'FINAL_CUMULATIVE_WORKER_WALL',
                summed_worker_time_not_CPU_or_elapsed=True)
    e2=base/'phase_07/p67_postaudit_e2_0908/run_001';v1=base/'phase_07/p67_patch24h_0908/run_001'
    parent_path=e2/'control/parent_source_manifest.parquet'
    ctx.bind(parent_path,'SEALED_V1_COUNTER_AND_LABEL_MANIFEST',
        '6eadbb9cbd08aaa38f8e4e38ab3426b0059f53cc13d29ca5875681c029514348')
    parent=pd.read_parquet(parent_path)
    if not parent.relative_path.is_unique:raise ValueError('Ambiguous sealed V1 source identity')
    parent_hash=parent.set_index('relative_path').sha256.to_dict()
    migration_path=e2/'validation/E2_MIGRATION_PASS.json';ctx.bind(migration_path,'INHERITED_COUNTER_MIGRATION_LINEAGE')
    migration=read_json(migration_path)
    if migration['status']!='PASS' or migration['mutable_hardlinks'] or migration['v1_writes']:
        raise ValueError('Unverified inherited counter lineage')
    ctx.bind(ctx.repo/'partC_rev1_src/phase67/training.py','COUNTER_SCOPE_SOURCE_CODE_TRAIN_AND_CHECK_ARE_DISJOINT')
    for ordinal,p in enumerate(sorted((e2/'regions').glob('*/training/*/status.json'))):
        if ordinal%100==0:ctx.guard()
        r=read_json(p)
        relative=p.relative_to(e2).as_posix();vp=v1/relative
        if relative in parent_hash:
            ctx.bind(vp,'SEALED_V1_FINAL_CUMULATIVE_COUNTER',parent_hash[relative]);prior=read_json(vp)
        elif vp.exists():raise ValueError('Unsealed same-path V1 counter cannot be subtracted')
        else:prior=None
        for field,scope in [('train_seconds','OPTIMIZER'),('check_seconds','VALIDATION_CHECK')]:
            if field not in r:
                missing.append(dict(stage='P67_'+scope,status='COUNTER_UNAVAILABLE',missing_scope=str(p)+' / '+field));continue
            if prior is not None and field in prior:
                old=float(prior[field]);new=float(r[field])
                if new<old-1e-6:
                    missing.append(dict(stage='P67_'+scope,status='AMBIGUOUS_DECREASED_COUNTER_EXCLUDED',
                        missing_scope=str(p)+' / '+field,prior_value=old,current_value=new,
                        prior_sha256=ctx.bind(vp,'AMBIGUOUS_PRIOR_COUNTER'),current_sha256=ctx.bind(p,'AMBIGUOUS_CURRENT_COUNTER')));continue
                add('P67_V1_'+scope,p.relative_to(e2).as_posix(),old,vp,'FINAL_CUMULATIVE_WORKER_WALL',
                    source_field=field,summed_worker_time_not_CPU_or_elapsed=True)
                add('P67_E2_'+scope+'_INCREMENT',p.relative_to(e2).as_posix(),max(0.,new-old),p,'DIFFERENCE_OF_BOUND_CUMULATIVE_WORKER_WALL',
                    source_field=field,inherited_counter_sha256=ctx.bind(vp,'V1_COUNTER_TO_SUBTRACT'),summed_worker_time_not_CPU_or_elapsed=True)
            else:
                prefix='P67_E2_ONLY_' if prior is None else 'P67_V1_E2_UNSEPARATED_'
                add(prefix+scope,p.relative_to(e2).as_posix(),r[field],p,'FINAL_CUMULATIVE_WORKER_WALL',
                    source_field=field,summed_worker_time_not_CPU_or_elapsed=True)
    v1_index_relative='reports/result_audit/generation_allocation_all_jobs.parquet';v1_index=v1/v1_index_relative
    ctx.bind(v1_index,'SEALED_V1_LABEL_JOB_INDEX',parent_hash[v1_index_relative])
    v1_jobs=pd.read_parquet(v1_index);v1_jobs=v1_jobs.loc[v1_jobs.job_result_present.eq(True)].copy()
    if v1_jobs.source_result_path.duplicated().any() or v1_jobs.duplicated(['region_id','jobid']).any():
        raise ValueError('V1 label receipt identities are not disjoint')
    v1_hashes=set()
    for ordinal,row in enumerate(v1_jobs.itertuples(index=False)):
        if ordinal%100==0:ctx.guard()
        relative=Path(row.source_result_path)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe V1 indexed receipt path')
        relative=relative.as_posix();p=v1/relative
        if parent_hash.get(relative)!=row.source_result_sha256:raise ValueError('V1 label index and sealed parent disagree')
        ctx.bind(p,'SEALED_V1_LABEL_JOB_RECEIPT',row.source_result_sha256);r=read_json(p)
        if row.source_result_sha256 in v1_hashes:raise ValueError('Duplicated V1 label receipt payload')
        v1_hashes.add(row.source_result_sha256)
        add('P67_V1_CATALOGUED_LABEL_WORKER_JOBS',str(row.region_id)+'/'+str(row.jobid),r['seconds'],p,
            'DISJOINT_JOB_WORKER_WALL',worker_jobs_may_overlap_in_elapsed=True,nested_solver_seconds_not_added=True)
    sealed_job_paths={p for p in parent_hash if '/data_extension/jobs/' in p and p.endswith('/RESULT.json')}
    if sealed_job_paths!=set(v1_jobs.source_result_path.map(lambda p:Path(p).as_posix())):
        raise ValueError('V1 label receipt index does not cover the sealed receipt family')
    missing.append(dict(stage='P67_V1_LABEL_GENERATION',status='CATALOGUED_WORKER_WALL_SUBTOTAL',
        missing_scope='All sealed completed job receipts are covered; unfinished work and work outside this receipt family remain unknown'))
    index=e2/'reports/E2_generation_coverage/SOURCE_BINDINGS.csv';ctx.bind(index,'E2_JOB_RECEIPT_INDEX')
    catalog=pd.read_csv(index);seen=set();attempts=[]
    for ordinal,row in enumerate(catalog.loc[catalog.role.eq('E2_JOB_COUNTERS')].itertuples(index=False)):
        if ordinal%100==0:ctx.guard()
        p=Path(row.path);ctx.bind(p,'E2_JOB_COUNTERS',row.sha256);r=read_json(p)
        if row.sha256 in v1_hashes:raise ValueError('Inherited V1 label receipt would be counted again as E2 work')
        key=(r.get('region_id'),r.get('jobid'),row.sha256)
        if key in seen:continue
        seen.add(key)
        add('E2_LABEL_WORKER_JOBS',str(key),r['seconds'],p,'DISJOINT_JOB_WORKER_WALL',
            recovered_existing=r.get('recovered_existing'),new_certified=r.get('new_certified'),
            worker_jobs_may_overlap_in_elapsed=True,nested_solver_seconds_not_added=True)
        for number,a in enumerate(r.get('attempts',[])):
            attempts.append(dict(region_id=r.get('region_id'),jobid=r.get('jobid'),attempt=number,source_job_sha256=row.sha256,
                status=a.get('status',a.get('disposition')),solver_seconds=a.get('seconds'),
                counted_as_new_solver_work=a.get('status',a.get('disposition')) in ('NEW_CERTIFIED','QUARANTINED')))
    pd.DataFrame(attempts).to_csv(dst/'E2_label_solver_attempt_timers_nested_not_added.csv',index=False,float_format='%.17g')
    # P6 result catalogs may include copied resumes. Bind exact payload hash once.
    p6_seen={}
    for name in ('s03_active_0906','s03_active_resume_0907'):
        root=base/'phase_06'/name/'run_001';db=root/'control/enrichment.sqlite'
        if not db.exists():continue
        ctx.bind(db,'READ_ONLY_P6_JOB_CATALOG');con=sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True)
        try:
            con.execute('PRAGMA query_only=ON');jobs=con.execute('SELECT round_index,wave,region_id,start_slot,result_path FROM completed_batches').fetchall()
        finally:con.close()
        for ordinal,(round_index,wave,rid,slot,path) in enumerate(jobs):
            if ordinal%100==0:ctx.guard()
            p=Path(path);p=p if p.is_absolute() else root/p
            if not p.exists():missing.append(dict(stage='P6_LABEL_JOBS',status='MISSING_RESULT',missing_scope=str(p)));continue
            digest=ctx.bind(p,'COMPLETED_P6_JOB_RESULT')
            job_key=(rid,round_index,wave,slot)
            record_id=f'{rid}/{round_index}/{wave}/{slot}/{digest}'
            if job_key in p6_seen:
                previous=p6_seen[job_key]
                if previous is not None and previous['sha256']!=digest:
                    records[:]=[item for item in records if not (item['stage']=='P6_CATALOGUED_LABEL_WORKER_JOBS' and item['record_id']==previous['record_id'])]
                    missing.append(dict(stage='P6_LABEL_JOBS',status='AMBIGUOUS_RESUMED_JOB_EXCLUDED_FROM_SUBTOTAL',
                        missing_scope=str(job_key)+' has different original/resume payloads; neither counted as disjoint work'))
                    p6_seen[job_key]=None
                continue
            p6_seen[job_key]=dict(sha256=digest,record_id=record_id);r=read_json(p)
            if 'seconds' in r:add('P6_CATALOGUED_LABEL_WORKER_JOBS',record_id,r['seconds'],p,
                'DISJOINT_HASH_DEDUPLICATED_JOB_WORKER_WALL',worker_jobs_may_overlap_in_elapsed=True,
                original_and_resume_share_campaign_clock=True)
    missing.append(dict(stage='P6_LABEL_HISTORY',status='MEASURED_LOWER_BOUND',
        missing_scope='Catalogued jobs do not necessarily cover initial seed/projection labels; promotion or recertification timers are not original solve time'))
    # Reopen exact historical physical timing authority, without rerunning it.
    p=ctx.prior/'validation/PHYSICAL_TIMING_EVIDENCE.json';ctx.bind(p,'R2_VERIFIED_PHYSICAL_TIMING_AUDIT');audit=read_json(p)
    for row in audit['paths'].values():ctx.bind(row['path'],'HISTORICAL_PHYSICAL_TIMING_SOURCE',row['sha256'])
    updates=pd.read_csv(audit['paths']['update']['path'])
    updates.to_csv(dst/'historical_physical_authoritative_update_history.csv',index=False,float_format='%.17g')
    write_json(dst/'HISTORICAL_PHYSICAL_TIMING_AUDIT.json',audit)
    p=ctx.prior/'tables/physical_benchmark_timing.csv';ctx.bind(p,'ACCEPTED_COMPACT_PHYSICAL_TIMING_TABLE')
    historical=pd.read_csv(p)
    historical['update_wall_seconds_scope']='Cumulative recorded iteration_wall_seconds, including checkpoint/refresh work inside each iteration; not pure update/kernel time'
    historical['later_refresh_only_seconds_scope']='Separate later zero-update frozen-state refresh; not assignment runtime'
    historical.to_csv(dst/'historical_physical_benchmark_timing.csv',index=False,float_format='%.17g')
    # Complete child clocks include source loading and startup, not just solver loops.
    current=[]
    for p in sorted((ctx.root/'jobs').glob('*/EXIT.json')):
        r=read_json(p);current.append(dict(job_id=r['job_id'],task=r['task'],started_epoch=r['start_epoch'],ended_epoch=r['ended_epoch'],
            elapsed_seconds=r['elapsed_seconds'],exit_code=r['exit_code'],peak_rss_bytes=r['peak_process_tree_rss_bytes'],
            supervisor_stop=r.get('supervisor_stop'),source_path=str(p),source_sha256=sha(p)))
    pd.DataFrame(current).to_csv(dst/'R4_complete_job_clock_ledger.csv',index=False,float_format='%.17g')
    frame=pd.DataFrame(records)
    if frame.duplicated(['stage','record_id']).any():raise ValueError('Duplicate accounting identity')
    frame.to_csv(dst/'offline_disjoint_components.csv',index=False,float_format='%.17g')
    frame.groupby(['stage','measure'],as_index=False).seconds.sum().to_csv(dst/'offline_stage_subtotals.csv',index=False,float_format='%.17g')
    pd.DataFrame(missing).to_csv(dst/'offline_missing_scope.csv',index=False)
    result=dict(status='EXECUTED',component_records=len(records),missing_scope_records=len(missing),
        complete_offline_elapsed_cost='NOT_ESTABLISHED',complete_CPU_GPU_hours='NOT_ESTABLISHED',
        worker_wall_not_added_to_parent_elapsed=True,inherited_training_not_double_counted=True,
        amortization='Complete offline intercepts unknown; only explicitly labelled online-query slopes or conditional scenarios may be plotted; no confident full break-even claim')
    write_json(dst/'RESULT.json',result);return result
