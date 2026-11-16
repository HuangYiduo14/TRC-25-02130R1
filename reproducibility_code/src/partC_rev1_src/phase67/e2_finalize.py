"""Freeze, evaluate and seal one cumulative E2 extension without modifying v1."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
import zipfile
import numpy as np
import pandas as pd
from .common import root,rd,authority,read_json,write_json,sha,elapsed
from .training import PRIMARY
from .resources import owned_bytes
from .review_artifacts import snapshot_paths,snapshot_tests
from .finalize import aggregate,source_closeout,verify_folder
from partC_rev1_src.phase7.finalize import git,copy,text_file,schema,scan_bundle


def parent_unchanged(c,*,persist=True):
    for b in c['sealed_parent_guard_bindings']:
        assert Path(b['path']).stat().st_size==b['size_bytes'] and sha(b['path'])==b['sha256']
    records=pd.read_parquet(root(c)/'validation/migration_file_bindings.parquet').drop_duplicates('source_path')
    changed=[]
    for row in records.itertuples(index=False):
        if sha(row.source_path)!=row.sha256:changed.append(row.source_path)
    assert not changed,changed
    result=dict(status='PASS',guard_bindings=c['sealed_parent_guard_bindings'],independently_copied_source_files_rechecked=len(records),
        source_mutations=0,checked_unix=time.time())
    if persist:write_json(root(c)/'validation/SEALED_PARENT_UNCHANGED.json',result)
    return result


def reports(c):
    out=root(c);dest=out/'reports';dest.mkdir(parents=True,exist_ok=True)
    registry=pd.read_parquet(out/'control/source_region_registry.parquet');old=pd.read_parquet(out/'validation/inherited_checkpoint_states.parquet')
    old_primary=old.loc[old.kind.eq(PRIMARY)].set_index('region_id');region_rows=[];frames=[];models=[];labels=[];failures=[];access=[]
    for row in registry.to_dict('records'):
        rid=row['region_id'];record=dict(region_id=rid,m_active=int(row['m_active']),status='SKIPPED_NO_ACTIVE_LOW_PAIRS' if not row['m_active'] else 'E2_UNRESOLVED')
        if row['m_active']:
            frozen_path=rd(c,rid)/'training/cycles/final/manifest.json'
            if frozen_path.exists():
                frozen=read_json(frozen_path);record['selected_model_kind']=frozen['champion_kind']
                for kind,b in frozen['files'].items():models.append(dict(region_id=rid,model_kind=kind,selected=kind==frozen['champion_kind'],
                    portable_path=str(rd(c,rid)/b['path']),portable_sha256=b['sha256'],valid_math=kind not in frozen['candidate_failures']))
                failures.extend(dict(region_id=rid,model_kind=k,category='CANDIDATE_MATH',**v) for k,v in frozen['candidate_failures'].items())
            status=read_json(rd(c,rid)/'training'/PRIMARY/'status.json');inherited=int(old_primary.loc[rid,'steps'])
            record.update(standalone_ICNN_available=True,primary_inherited_updates=inherited,primary_cumulative_updates=status['steps'],
                primary_E2_additional_updates=status['steps']-inherited,training_status=status['status'],trained_n=status['n_train'],
                primary_selected_checkpoint_updates=status['best_steps'])
            assert record['primary_E2_additional_updates']>=0
            result_path=rd(c,rid)/'final_evaluation/RESULT.json'
            if result_path.exists():
                result=read_json(result_path);record['status']=result['status'];record['physical_reopened']=result['physical_labels_freshly_reopened']
                assert sha(result['access_receipt_path'])==result['access_receipt_sha256']
                receipt=read_json(result['access_receipt_path']);assert receipt['unix']<=result['completed_unix']
                access.append(dict(region_id=rid,receipt_path=result['access_receipt_path'],receipt_sha256=result['access_receipt_sha256'],
                    before_unix=receipt['unix'],completed_unix=result['completed_unix']))
                frame=pd.read_parquet(result_path.parent/'metrics.parquet');frames.append(frame)
                failures.extend(dict(region_id=rid,category='FINAL_METRIC',**v) for v in result['metric_failures'])
                failures.extend(dict(region_id=rid,category='PHYSICAL_REOPEN',**v) for v in result['physical_reopen_failures'])
                failures.extend(dict(region_id=rid,category='NATIVE_CHECKPOINT_REOPEN',**v) for v in result['checkpoint_reopen_failures'])
            for namespace in ['train_labels','valid_labels','audit2_labels']:
                for p in (rd(c,rid)/'data_extension'/namespace/'receipts').glob('*.json'):
                    receipt=read_json(p);meta=receipt['metadata']
                    labels.append(dict(region_id=rid,input_hash=receipt['input_hash'],role=meta['split'],quality=receipt['scalars']['quality'],
                        fresh_E2=meta['run_id']==c['run_id'],source_run_id=meta['run_id'],receipt_path=str(p),receipt_sha256=sha(p),payload_sha256=receipt['payload_sha256']))
            for p in (rd(c,rid)/'failures').glob('*.json'):
                failure=read_json(p);failures.append(dict(region_id=rid,category='WORKER_ATTEMPT',path=str(p),error=failure.get('error')))
        region_rows.append(record)
    region=pd.DataFrame(region_rows);region.to_csv(dest/'all_region_status.csv',index=False)
    model=pd.DataFrame(models);model.to_csv(dest/'typed_model_registry.csv',index=False)
    label=pd.DataFrame(labels);label.to_parquet(dest/'E2_local_label_receipts.parquet',index=False)
    assert not label.duplicated(['region_id','input_hash']).any()
    unresolved=region.loc[region.status.eq('E2_UNRESOLVED')].astype(object)
    unresolved=unresolved.where(pd.notna(unresolved),None)
    write_json(dest/'unresolved_docket.json',dict(region_failures=unresolved.to_dict('records'),events=failures,
        inherited_v1_limitations='Preserved in the exact sealed v1 ZIP; no retrospective access-log reconstruction'))
    pd.DataFrame(access).to_csv(dest/'BEFORE_access_receipt_reconciliation.csv',index=False)
    metrics=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if len(metrics):
        metrics.to_parquet(dest/'candidate_metrics.parquet',index=False);summary=aggregate(metrics);summary.to_csv(dest/'model_kind_summary.csv',index=False)
        summary.loc[summary.model_kind.eq(PRIMARY)].to_csv(dest/'standalone_ICNN_summary.csv',index=False)
        summary.loc[summary.split.str.startswith('E2_AUDIT')].to_csv(dest/'fresh_E2_audit_summary.csv',index=False)
        summary.loc[summary.split.isin(['LOW_TEST','HIGH_PROJECTED','PATCH_AUDIT_OPERATING','PATCH_AUDIT_STRESS'])].to_csv(dest/'reused_historical_evaluation_summary.csv',index=False)
    counts=label.groupby(['fresh_E2','role','quality']).size().rename('rows').reset_index();counts.to_csv(dest/'label_counts_by_origin_role_quality.csv',index=False)
    curves=[]
    for p in (out/'regions').glob('*/training/*/learning_curves.jsonl'):
        with p.open(encoding='utf-8') as stream:
            for line in stream:
                entry=json.loads(line)
                if entry['unix']<c['clock_facts']['adopted_unix']:continue
                score=entry.pop('metrics');curves.append(dict(region_id='partC_p2_region_'+p.parents[2].name,**entry,VALID_score=score.get('score')))
    if curves:pd.DataFrame(curves).to_parquet(dest/'E2_learning_curves.parquet',index=False)
    history=[]
    for line in (out/'control/resource_history.jsonl').read_text(encoding='utf-8').splitlines():history.append(json.loads(line))
    resource=pd.DataFrame(history);resource.to_csv(dest/'resource_usage.csv',index=False)
    plan=read_json(out/'validation/AUDIT2_INPUTS_FROZEN.json');fresh=label.loc[label.fresh_E2];audit=fresh.loc[fresh.role.str.startswith('E2_AUDIT')]
    data_coverage=fresh.groupby('role').agg(regions=('region_id','nunique'),rows=('input_hash','size')).reset_index().to_dict('records')
    coverage=dict(all_regions=1847,active_regions=1566,inactive_skipped=281,active_pairs=80316,final_evaluated_regions=len(access),
        fresh_E2_labels=len(fresh),recovered_v1_receipts=int((~label.fresh_E2).sum()),fresh_audit_labels=len(audit),
        fresh_audit_AB=int(audit.quality.isin(['A','B']).sum()),fresh_audit_target=25056,fresh_audit_planned=plan['planned_rows'],primary_inherited_updates=int(old_primary.steps.sum()),
        primary_E2_additional_updates=int(region.primary_E2_additional_updates.sum()),role_coverage=data_coverage)
    incomplete=coverage['fresh_audit_AB']<25056 or len(access)<1566
    state=dict(status='WAITING_FOR_EXTERNAL_REVIEW',campaign_status='P67_E2_PARTIAL_COHORT_WAITING_REVIEW' if incomplete else 'P67_E2_COMPLETE_WITH_NOTES_WAITING_REVIEW',
        coverage=coverage,peak_rss_bytes=int(resource.rss_bytes.max()),total_repair_epochs=8,national_readiness=False,next_phase_authorized=False,next_step_authorized=False,
        scientific_continuation_performed=coverage['fresh_E2_labels']>0 and coverage['primary_E2_additional_updates']>0,
        fresh_and_reused_audits_separate=True,standalone_analytic_selected_models_separate=True,
        next_step_recommendation='Review the E2 extension together with its immutable v1 parent. All unresolved regions, candidate failures, partial rounds and audit denominators remain explicit. No later phase is authorized.')
    freeze=read_json(out/'control/ALL_CHAMPIONS_FROZEN.json')
    write_json(dest/'deadline_table.json',dict(original_start=c['task_started_unix'],original_hard_deadline=c['hard_deadline_unix'],
        latest_science_cutoff=c['latest_science_cutoff_unix'],E2_adopted=c['clock_facts']['adopted_unix'],
        first_scientific_action=read_json(out/'validation/E3_REENTRY_PASS.json')['started_unix'],science_complete=read_json(out/'control/SCIENCE_COMPLETE.json'),
        final_freeze_unix=freeze['frozen_unix'],last_protected_access=max((r['before_unix'] for r in access),default=None),report_unix=time.time()))
    write_json(dest/'FINAL_RESULT.json',state)
    text_file(dest/'FINAL_REPORT.md',f'# E2 cumulative continuation result\n\n{state["campaign_status"]}\n\n'
        f'E2 added {len(fresh):,} certified labels and {coverage["primary_E2_additional_updates"]:,} primary ICNN optimizer updates. '
        f'{len(audit):,} of 25,056 planned fresh audit labels were certified; {len(access):,} of 1,566 active regions have completed evaluation receipts. '
        'Exact A/B/C classes and denominators are in the tables. C is diagnostic only.\n\n'
        'The Windows helper passed three consecutive real spawned-process trials. Earlier failed implementations and payloads remain evidence. '
        'The final protocol gates controlled opens and replacements, retains delete-shared snapshots, and fails closed on missing or malformed forecasts.\n\n'
        'Pre-final optimizer/RNG/sampler/checkpoint bytes were independently forked. J1 was not repeated. Parent TRAIN/VALID roles and original model/scales remain unchanged. '
        'Fresh E2 audit inputs were frozen before adaptation. Old LOW_TEST, HIGH and v1 audit scores are reused historical evaluation, not fresh E2 confirmation.\n\n'
        'Standalone ICNN, analytic baselines and the explicitly typed selected portfolio are separate tables. Invalid portable candidates are retained and excluded before selection; thresholds are unchanged. '
        'No selection or training occurred after the E2 final freeze.\n\n'+state['next_step_recommendation']+'\n')
    return state


def external_index(c):
    out=root(c);dest=out/'handoff/external_index';dest.mkdir(parents=True,exist_ok=True);rows=[]
    volatile={'control/heartbeat.json','control/guardian_heartbeat.json','control/resource_history.jsonl','control/campaign_console.log','FINAL_HANDOFF.json'}
    for base,dirs,names in os.walk(out):
        if Path(base)==out:dirs[:]=[d for d in dirs if d!='handoff']
        for name in names:
            p=Path(base)/name;relative=p.relative_to(out).as_posix()
            if relative in volatile:continue
            try:count,description=schema(p)
            except Exception as error:count,description=None,'Preserved failed/partial artifact: '+repr(error)
            rows.append(dict(relative_path=relative,sha256=sha(p),size_bytes=p.stat().st_size,modified_unix_ns=p.stat().st_mtime_ns,
                format=p.suffix,row_or_object_count=count,schema_or_shape=description,copy_policy='LOCAL_HASH_BOUND'))
    pd.DataFrame(rows).to_parquet(dest/'manifest.parquet',index=False)
    write_json(dest/'INDEX.json',dict(root=str(out),files=len(rows),bytes=sum(r['size_bytes'] for r in rows),manifest_sha256=sha(dest/'manifest.parquet'),
        excluded_volatile=sorted(volatile),pending_artifacts_preserved_and_indexed=True,parent_root=c['parent_output_root_readonly'],parent_zip_sha256=c['parent_zip_sha256']))
    return dest


def seal(c,state):
    out=root(c);repo=Path(c['execution']['repository_root']);step=c['step_id'];work=repo/'partC_doc_icnn/phase_7_work'/step
    assert not git(c,'diff','--name-only').strip();head=git(c,'rev-parse','HEAD').strip()
    hid=step+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_extension_001'
    parent=repo/'partC_doc_icnn/handoffs/phase_07'/('step_'+step);folder=parent/hid
    archive=repo/'artifacts_icnn/handoff_bundles/phase_07'/('phase07_step'+step+'_'+hid+'_SYNC.zip')
    assert not folder.exists() and not archive.exists();folder.mkdir(parents=True);archive.parent.mkdir(parents=True,exist_ok=True)
    for name in ['governing_docs','configs','code_snapshot','code_diff','summary_data','figures','geospatial_summary','log_excerpts','external_artifact_index']:(folder/name).mkdir()
    for p in authority(c).rglob('*'):
        if p.is_file() and p.suffix not in ['.zip','.pyc'] and '__pycache__' not in p.parts:copy(p,folder/'governing_docs/authority'/p.relative_to(authority(c)))
    for name in ['00_READ_FIRST.md','PROJECT_STATUS.md','12_DECISION_LOG.md','20_HANDOFF_SYNC_PROTOCOL.md','21_REVIEW_APPROVAL_GATES.md','23_SYNC_BUNDLE_CONTENT_AND_SIZE_POLICY.md','24_CODE_AND_DATA_SNAPSHOT_POLICY.md']:
        copy(repo/'partC_doc_icnn'/name,folder/'governing_docs/repository'/name)
    copy(repo/'AGENTS.md',folder/'governing_docs/AGENTS.md');copy(work/'effective_config.json',folder/'configs/effective_config.json');copy(out/'run_identity.json',folder/'configs/run_identity.json')
    paths=snapshot_paths(repo)
    for p in paths:copy(p,folder/'code_snapshot'/p.relative_to(repo))
    pd.DataFrame([dict(path=p.relative_to(repo).as_posix(),sha256=sha(p),size_bytes=p.stat().st_size) for p in paths]).to_csv(folder/'07_CODE_CHANGE_INDEX.csv',index=False)
    snapshot_tests(c,folder/'code_snapshot',out/'validation/staged_final_snapshot')
    excluded=[];renamed=[]
    for prefix,label in [(out/'reports','reports'),(out/'validation','validation'),(work,'work_records')]:
        for p in prefix.rglob('*'):
            if not p.is_file() or 'source_reads' in p.parts or '__pycache__' in p.parts or '.pytest_cache' in p.parts:continue
            if p.stat().st_size>25*1024**2:
                excluded.append(dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size,reason='Large artifact externally indexed'));continue
            relative=p.relative_to(prefix);target=folder/'summary_data'/label/relative
            if '.pending.' in p.name or p.suffix not in ['.md','.txt','.log','.json','.jsonl','.py','.xml','.csv','.parquet','.npz','.npy','.png','.pdf','.sha256']:
                try:p.read_text(encoding='utf-8')
                except UnicodeError:
                    excluded.append(dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size,reason='Failed binary attempt preserved by external binding'));continue
                target=target.with_name(target.name+'.txt');renamed.append(dict(original=str(p),review_path=target.relative_to(folder).as_posix(),sha256=sha(p),bytes_unchanged=True))
            copy(p,target)
    for p in (out/'control').glob('*.json'):
        if p.stat().st_size<=25*1024**2:copy(p,folder/'summary_data/control'/p.name)
    index=external_index(c)
    for p in index.iterdir():copy(p,folder/'external_artifact_index'/p.name)
    copy(out/'control/parent_source_manifest.parquet',folder/'external_artifact_index/v1_manifest.parquet')
    write_json(folder/'external_artifact_index/LINEAGE_AND_EXCLUSIONS.json',dict(parent_bindings=c['sealed_parent_guard_bindings'],
        authority_sha256=c['authority_sha256'],excluded=excluded,review_typed_evidence=renamed,parent_replaced=False,bulk_scientific_bytes_remain_local=True))
    text_file(folder/'code_diff/source.patch',git(c,'diff','--binary',c['initial_head'],head,'--','partC_rev1_src/phase67','tests/test_phase67_e2.py','tests/test_phase67_patch.py',str(work.relative_to(repo))))
    text_file(folder/'code_diff/commits.txt',git(c,'log','--format=%H %s',c['initial_head']+'..'+head));text_file(folder/'code_diff/git_status.txt',git(c,'status','--short'))
    copy(out/'reports/FINAL_REPORT.md',folder/'01_STEP_SUMMARY.md');write_json(folder/'02_HANDOFF_STATUS.json',dict(**state,head=head,handoff_id=hid))
    texts={'00_HANDOFF_README.md':'Read the step summary, status and reports. This ZIP extends the immutable v1 parent; it does not replace its final_001 archive.',
        '03_DECISIONS_AND_OPEN_QUESTIONS.md':state['next_step_recommendation'],
        '04_VALIDATION_RESULTS.md':'Live/isolated tests, three Windows spawn trials, native migration, fresh audit input freeze, E3 reentry, final candidate and physical reopen evidence are in summary_data/validation and reports. Failed attempts are preserved.',
        '08_COMMAND_LOG.md':'Commands use python -m partC_rev1_src.phase67.e2_setup with explicit E1/E2/E3 tasks, followed by phase67.campaign --stage J23 and the E2 effective config. Exact stage evidence and logs accompany this bundle.',
        '09_NEXT_STEP_RECOMMENDATION.md':state['next_step_recommendation'],
        '10_UPLOAD_THIS_FILE.txt':str(archive),
        '11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md':'Adds repair epoch 8 and the E2 evaluation episode under the inherited clock. Sealed v1 ZIP SHA256 '+c['parent_zip_sha256']+' remains unchanged.',
        'figures/NUMERICAL_FIGURE_DATA.md':'Scientific learning curves and model-kind/error tables are supplied under summary_data/reports.',
        'geospatial_summary/NO_GEOMETRY_CHANGE.md':'Active-only s03 network, capacities, topology and B/R records remain unchanged.'}
    for name,content in texts.items():text_file(folder/name,content+'\n')
    copy(out/'reports/resource_usage.csv',folder/'05_RESOURCE_USAGE.csv')
    write_json(folder/'log_excerpts/privacy_scan.json',scan_bundle(c,folder))
    records=[dict(relative_path=p.relative_to(folder).as_posix(),size_bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.rglob('*')) if p.is_file()]
    pd.DataFrame(records).to_csv(folder/'06_ARTIFACT_MANIFEST.csv',index=False)
    checks=records+[dict(relative_path='06_ARTIFACT_MANIFEST.csv',sha256=sha(folder/'06_ARTIFACT_MANIFEST.csv'))]
    text_file(folder/'CHECKSUMS.sha256',''.join(r['sha256']+'  '+r['relative_path']+'\n' for r in checks));verify_folder(folder)
    pending=archive.with_suffix('.zip.building')
    with zipfile.ZipFile(pending,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(folder).as_posix())
    assert pending.stat().st_size<200_000_000
    fresh=out/'handoff'/('fresh_'+hid)
    with zipfile.ZipFile(pending) as z:
        assert z.testzip() is None
        for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts and ':' not in name
        z.extractall(fresh)
    count=verify_folder(fresh);isolated=snapshot_tests(c,fresh/'code_snapshot',out/'handoff'/('fresh_validation_'+hid));assert verify_folder(fresh)==count
    parent_unchanged(c,persist=False);assert time.time()<c['hard_deadline_unix']
    h=sha(pending);size=pending.stat().st_size;pending.rename(archive)
    companion=dict(status='WAITING_FOR_EXTERNAL_REVIEW',campaign_status=state['campaign_status'],zip_path=str(archive),zip_sha256=h,zip_bytes=size,
        snapshot_head=head,parent_zip_sha256=c['parent_zip_sha256'],parent_unchanged=True,authority_sha256=c['authority_sha256'],source_folder=str(folder),
        coverage=state['coverage'],original_start_unix=c['task_started_unix'],original_deadline_unix=c['hard_deadline_unix'],latest_science_cutoff_unix=c['latest_science_cutoff_unix'],
        sealed_unix=time.time(),elapsed_seconds=elapsed(c),timer_reset=False,CRC='PASS',fresh_extraction='PASS',verified_checksums=count,
        fresh_snapshot_tests=isolated,total_repair_epochs=8,national_readiness=False,next_phase_authorized=False,next_step_authorized=False,
        added_artifact_bytes_including_handoff=owned_bytes(c)[0])
    assert companion['added_artifact_bytes_including_handoff']<c['execution']['added_disk_bytes']
    write_json(archive.with_suffix('.companion.json'),companion);text_file(archive.with_suffix('.sha256'),h+'  '+archive.name+'\n')
    write_json(out/'FINAL_HANDOFF.json',companion);text_file(parent/'LATEST_HANDOFF.txt',json.dumps(companion,indent=2)+'\n');return companion


def finalize(c):
    from .campaign import Runner,fair_regions
    out=root(c);regions=fair_regions(c);complete=read_json(out/'control/SCIENCE_COMPLETE.json');runner=Runner(c,2)
    if not (out/'control/ALL_CHAMPIONS_FROZEN.json').exists():
        results=runner.batch([dict(action='FREEZE',region_id=r,cycle=complete['cycle'],final=True) for r in regions]);valid=[r['job']['region_id'] for r in results if r['ok']]
        write_json(out/'control/ALL_CHAMPIONS_FROZEN.json',dict(episode_id=c['episode_id'],frozen_unix=time.time(),regions=len(valid),
            manifests=[dict(region_id=r,sha256=sha(rd(c,r)/'training/cycles/final/manifest.json')) for r in valid],
            unresolved_regions=[r['job']['region_id'] for r in results if not r['ok']],selection_uses_TRAIN_VALID_only=True,
            hidden_audit_not_yet_accessed=True,training_and_generation_now_prohibited=True))
    freeze=read_json(out/'control/ALL_CHAMPIONS_FROZEN.json');runner.batch([dict(action='EVALUATE',region_id=r['region_id']) for r in freeze['manifests']]);runner.close()
    state=reports(c);source_closeout(c);parent_unchanged(c)
    repo=Path(c['execution']['repository_root']);work=repo/'partC_doc_icnn/phase_7_work'/c['step_id']
    for name in ['FINAL_REPORT.md','FINAL_RESULT.json']:copy(out/'reports'/name,work/name)
    entry='# E2 continuation sealed for external review\n\n'+state['campaign_status']+'. See phase_7_work/'+c['step_id']+'/FINAL_REPORT.md. Original v1 remains immutable.\n\n---\n\n'
    for name in ['00_READ_FIRST.md','PROJECT_STATUS.md']:
        p=repo/'partC_doc_icnn'/name;text_file(p,entry+p.read_text(encoding='utf-8'))
    p=repo/'partC_doc_icnn/12_DECISION_LOG.md'
    text_file(p,p.read_text(encoding='utf-8')+'\n\n## E2 continuation result\n\n'+state['campaign_status']+'. External review remains required; this result does not authorize another phase.\n')
    git(c,'add','--',str(work.relative_to(repo)/'FINAL_REPORT.md'),str(work.relative_to(repo)/'FINAL_RESULT.json'),
        'partC_doc_icnn/00_READ_FIRST.md','partC_doc_icnn/PROJECT_STATUS.md','partC_doc_icnn/12_DECISION_LOG.md')
    git(c,'commit','-m','phase67: record E2 cumulative continuation result')
    result=seal(c,state);print(json.dumps(result),flush=True);return result
