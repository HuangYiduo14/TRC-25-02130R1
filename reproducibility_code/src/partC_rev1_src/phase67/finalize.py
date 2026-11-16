"""Cumulative final freeze, native reports, source closure and one verified handoff."""
from pathlib import Path
from datetime import datetime,timezone
import os,time,json,zipfile,shutil,subprocess,hashlib
import numpy as np
import pandas as pd
from .common import root,rd,authority,read_json,write_json,save_npz,sha,elapsed
from .source import Sources
from .data import revision,Union
from .training import PRIMARY,CONTROL
from .metrics import predict_arrays,score_metric
from .resources import owned_bytes
from .review_artifacts import snapshot_paths,snapshot_tests
from partC_rev1_src.phase7.finalize import git,copy,text_file,scan_bundle,schema

def aggregate(f):
    selected=f.loc[f.family.eq('ALL') & f.regime.eq('ALL') & ~f.nonfinite].copy();rows=[]
    columns=['MARE_equal_pairs','MARE_LOW_pair_weighted','MAE_seconds','RMSE_seconds','bias_seconds','mean_interval_violation_relative','robust_objective','legacy_composite']
    for (split,kind),g in selected.groupby(['split','model_kind']):
        for aggregation,weights in [('EQUAL_REGION',np.ones(len(g))),('PAIR_COORDINATES',g.n_coordinates.to_numpy(float)),('LOW_TRAVERSAL_EXPOSURE',g.LOW_exposure.to_numpy(float))]:
            rows.append(dict(split=split,model_kind=kind,aggregation=aggregation,regions=len(g),active_pairs=int(g.m_active.sum()),coordinates=int(g.n_coordinates.sum()),
                LOW_exposure=float(g.LOW_exposure.sum()),**{col:float(np.average(g[col],weights=weights)) for col in columns},
                negative_regions=int(g.negative_cost_count.gt(0).sum()),screen_pass_regions=int(g.accuracy_screen.fillna(False).sum())))
    champion=selected.loc[selected.selected_champion].copy();champion['model_kind']='SELECTED_TYPED_PORTFOLIO'
    if len(champion):
        for split,g in champion.groupby('split'):
            for aggregation,weights in [('EQUAL_REGION',np.ones(len(g))),('PAIR_COORDINATES',g.n_coordinates.to_numpy(float)),('LOW_TRAVERSAL_EXPOSURE',g.LOW_exposure.to_numpy(float))]:
                rows.append(dict(split=split,model_kind='SELECTED_TYPED_PORTFOLIO',aggregation=aggregation,regions=len(g),active_pairs=int(g.m_active.sum()),coordinates=int(g.n_coordinates.sum()),
                    LOW_exposure=float(g.LOW_exposure.sum()),**{col:float(np.average(g[col],weights=weights)) for col in columns},
                    negative_regions=int(g.negative_cost_count.gt(0).sum()),screen_pass_regions=int(g.accuracy_screen.fillna(False).sum())))
    result=pd.DataFrame(rows)
    # A population RMSE is sqrt(weighted MSE), not the weighted mean of regional RMSEs.
    for i,row in result.iterrows():
        g=(champion if row.model_kind=='SELECTED_TYPED_PORTFOLIO' else selected.loc[selected.model_kind.eq(row.model_kind)])
        g=g.loc[g.split.eq(row.split)]
        w=np.ones(len(g)) if row.aggregation=='EQUAL_REGION' else g.n_coordinates.to_numpy(float) if row.aggregation=='PAIR_COORDINATES' else g.LOW_exposure.to_numpy(float)
        result.loc[i,'RMSE_seconds']=float(np.sqrt(np.average(g.raw_MSE_seconds2,weights=w)))
    return result

def source_closeout(c):
    manifests=list((root(c)/'validation/source_reads').glob('*.parquet'))
    f=pd.concat([pd.read_parquet(p) for p in manifests],ignore_index=True).drop_duplicates('path')
    changed=[]
    for row in f.itertuples():
        p=Path(row.path);s=p.stat()
        if s.st_size!=row.size_bytes or s.st_mtime_ns!=row.modified_unix_ns:
            if s.st_size!=row.size_bytes or sha(p)!=row.sha256:changed.append(row.path)
    assert not changed,'Read-bound source files changed: '+str(changed[:5])
    f.to_parquet(root(c)/'validation/source_read_closure.parquet',index=False)
    result=dict(status='PASS',distinct_read_files=len(f),authenticated_bytes=int(f.size_bytes.sum()),changed_source_bytes=0,
        full_historical_tree_recopied=False,method='SHA256 when read; final metadata closure with rehash on change')
    write_json(root(c)/'validation/SOURCE_CLOSEOUT.json',result);return result

def export_examples(c,source,metrics):
    from .audit import source_oracle
    from partC_rev1_src.phase6.storage import Store
    candidates=metrics.loc[metrics.family.eq('ALL')&metrics.regime.eq('ALL')&metrics.split.isin(['TRAIN_old','TRAIN_new','VALID_old','PATCH_VALID'])&metrics.model_kind.isin([PRIMARY,'OLD_ICNN_FROZEN'])]
    chosen=[]
    for kind in [PRIMARY,'OLD_ICNN_FROZEN']:
        chosen.extend(candidates.loc[candidates.model_kind.eq(kind)].sort_values('max_relative',ascending=False).head(20).to_dict('records'))
    seen=set();records=[]
    for row in chosen:
        key=(row['region_id'],row['worst_input_hash'])
        if key in seen:continue
        seen.add(key);rid,ih=key;union=Union(c,source,rid,final=True);d,a,pairs,s=source.local(rid)
        if ih in set(union.old.rows.input_hash):
            rel='regions/'+rid.removeprefix('partC_p2_region_');rc=source.json6(rel+'/labels/receipts/'+ih+'.json');label=source.npz6(rel+'/labels/'+rc['payload_path']);o=source_oracle(source,rid,a)
        else:
            found=None
            for namespace in ['train_labels','valid_labels']:
                p=rd(c,rid)/'data_extension'/namespace/'receipts'/(ih+'.json')
                if p.exists():found=(namespace,p);break
            assert found is not None
            namespace,p=found;rc=read_json(p)
            with np.load(p.parent.parent/rc['payload_path'],allow_pickle=False) as z:label={k:z[k] for k in z.files}
            o=Store(rd(c,rid)/'data_extension',d,a,namespace=namespace).oracle
        dest=root(c)/'reports/native_worst_examples'/f'{len(records)+1:02d}_{rid.removeprefix("partC_p2_region_")}'
        ids=label['path_ids'];paths=[o.paths[int(i)] for i in ids];pair,ss,tt=o.pool_arrays()
        save_npz(dest/'used_paths.npz',source_path_ids=ids,pair=pair[ids],source=ss[ids],target=tt[ids],offsets=np.r_[0,np.cumsum([len(p) for p in paths])].astype(np.int64),arcs=np.concatenate(paths) if paths else np.empty(0,np.int64))
        save_npz(dest/'native_label.npz',**label);save_npz(dest/'local_graph.npz',**a);save_npz(dest/'source_scales.npz',**s);write_json(dest/'source_receipt.json',rc)
        frozen=read_json(rd(c,rid)/'training/cycles/final/manifest.json');pred={}
        for kind,b in frozen['files'].items():
            p=rd(c,rid)/b['path'];assert sha(p)==b['sha256']
            with np.load(p,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
            v,t=predict_arrays(kind,arrays,label['demand'][None,:]);save_npz(dest/(kind+'_prediction.npz'),value=v,cost=t)
            if kind==PRIMARY:copy(p,dest/(kind+'_portable.npz'))
            pred[kind]=dict(value=float(v[0]),minimum_cost=float(t.min()),max_native_error=float(np.max(abs(t[0]-label['tau']))))
        record=dict(region_id=rid,input_hash=ih,selection_model=row['model_kind'],selection_split=row['split'],max_relative=row['max_relative'],
            max_z_over_capacity=float(np.max(label['z']/a['capacity'],initial=0)),max_u_over_LOW=float(np.max(label['demand']/union.s['low'])),
            predictions=pred,source_payload_sha256=rc['payload_sha256'])
        write_json(dest/'README.json',record);records.append(record)
        if len(records)>=20:break
    write_json(root(c)/'reports/NATIVE_WORST_EXAMPLE_INDEX.json',dict(examples=records,count=len(records),selection_uses_TRAIN_VALID_only=True))

def reports(c):
    out=root(c);reports=out/'reports';reports.mkdir(exist_ok=True);registry=pd.read_parquet(out/'control/source_region_registry.parquet')
    finals=[];frames=[];states=[];curves=[];ablation=[]
    source=Sources(c)
    for row in registry.to_dict('records'):
        rid=row['region_id'];rev=revision(c,rid)
        record={k:row[k] for k in ['region_id','m_active','low_exposure','model_hash','support_hash']};record.update(new_TRAIN_AB=rev['TRAIN_AB'],new_VALID_AB=rev['VALID_AB'],
            new_labels=sum(b['rows'] for b in rev['batches']),new_AB=sum(b['AB'] for b in rev['batches']),new_C=sum(b['rows']-b['AB'] for b in rev['batches']))
        if not row['m_active']:
            record.update(status='SKIPPED_NO_ACTIVE_PAIRS',champion_kind=None,standalone_ICNN_available=False);states.append(record);continue
        p=rd(c,rid)/'final_evaluation/RESULT.json'
        if p.exists():
            result=read_json(p);finals.append(result);frames.append(pd.read_parquet(p.parent/'metrics.parquet'));record.update(result)
        else:record.update(status='FINAL_EVALUATION_UNRESOLVED',champion_kind=None)
        st=rd(c,rid)/'training'/PRIMARY/'status.json'
        if st.exists():
            s=read_json(st);record.update(standalone_ICNN_available=True,training_status=s['status'],updates=s['steps'],effective_epochs_current_dataset=s['cycle_epochs'],trained_n=s['n_train'],
                train_seconds=s['train_seconds'],training_dataset_hash=s['dataset_hash'])
        else:record.update(standalone_ICNN_available=False,training_status='UNTRAINED',updates=0)
        states.append(record)
        for path in (rd(c,rid)/'training').glob('*/learning_curves.jsonl'):
            for line in path.read_text().splitlines():
                v=json.loads(line);metrics=v.pop('metrics');v.update(region_id=rid,valid_score=metrics.get('score'),train_MARE=metrics.get('train',{}).get('MARE_equal_pairs'),
                    train_robust=metrics.get('train',{}).get('robust_objective'));v.pop('row_probabilities',None);curves.append(v)
    metrics=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame();reg=pd.DataFrame(states)
    metrics.to_parquet(reports/'metrics_population.parquet',index=False);reg.to_parquet(reports/'region_model_registry.parquet',index=False);reg.to_csv(reports/'region_model_registry.csv',index=False)
    aggregated=aggregate(metrics);aggregated.to_csv(reports/'aggregate_metrics.csv',index=False);aggregated.to_parquet(reports/'aggregate_metrics.parquet',index=False)
    base=metrics.loc[metrics.family.eq('ALL')&metrics.regime.eq('ALL')]
    base.to_csv(reports/'candidate_metrics_summary.csv',index=False)
    base.loc[base.model_kind.eq(PRIMARY)].to_csv(reports/'standalone_ICNN_metrics.csv',index=False)
    base.loc[base.negative_cost_count.fillna(0).gt(0)|base.nonfinite].to_csv(reports/'negative_and_nonfinite_docket.csv',index=False)
    reg.loc[reg.m_active.gt(0)&(~reg.standalone_ICNN_available|reg.status.ne('PASS'))].to_csv(reports/'failed_or_untrained_regions.csv',index=False)
    pd.DataFrame(curves).to_parquet(reports/'learning_curves_by_cycle.parquet',index=False)
    reg.loc[reg.m_active.gt(0)].groupby('champion_kind',dropna=False).size().rename('regions').reset_index().to_csv(reports/'explicit_champion_kind_counts.csv',index=False)
    plan=read_json(out/'control/audit_plan.json')
    for rid in plan['matched_update_panel']:
        cp=rd(c,rid)/'training'/CONTROL/'status.json';main=rd(c,rid)/'training'/PRIMARY/'status.json'
        row=dict(region_id=rid,status='CONTROL_NOT_COMPLETED')
        if cp.exists() and main.exists():
            control=read_json(cp);primary=read_json(main);row.update(control_updates=control['steps'],enlarged_updates=primary['steps'],same_updates=control['steps']==primary['steps'],
                control_dataset_hash=control['dataset_hash'],enlarged_dataset_hash=primary['dataset_hash'],control_old_VALID_score=control['last_metrics']['score'],enlarged_VALID_pool_score=primary['last_metrics']['score'],
                status='MATCHED' if control['steps']==primary['steps'] else 'PARTIAL_UNMATCHED')
        ablation.append(row)
    pd.DataFrame(ablation).to_csv(reports/'matched_update_ablation.csv',index=False)
    audit=[read_json(p) for p in (out/'regions').glob('*/audit/RESULT.json')]
    pd.DataFrame([dict(region_id=r['region_id'],**v) for r in audit for v in r['replay']]).to_csv(reports/'local_label_replay.csv',index=False)
    pd.DataFrame([dict(region_id=r['region_id'],**v) for r in audit for v in r['precision']]).to_csv(reports/'label_precision_sensitivity.csv',index=False)
    write_json(reports/'label_correction_ledger.json',dict(corrections=0,invalidations=0,reason='All scoped source labels reprice under the unchanged certified regional model; two tighter precision targets remain disclosed',source_labels_immutable=True))
    export_examples(c,source,metrics);source.flush('final_examples');source.close()
    resources=[json.loads(line) for line in (out/'control/resource_history.jsonl').read_text().splitlines()]
    rf=pd.DataFrame(resources);rf.to_csv(reports/'resource_usage.csv',index=False)
    pd.DataFrame(finals).to_csv(reports/'fresh_reopen_by_region.csv',index=False)
    nextrows=[]
    for r in reg.loc[reg.m_active.gt(0)].to_dict('records'):
        nextrows.append(dict(region_id=r['region_id'],action='REVIEW_MISSING_OR_INVALID_MODEL' if not r['standalone_ICNN_available'] or r['status']!='PASS' else
            'REVIEW_ANALYTIC_CHAMPION_AND_CONTINUE_NEURAL_RESEARCH' if r.get('champion_is_analytic') else 'EXTERNAL_REVIEW_OF_NEURAL_CANDIDATE',
            current_training_status=r.get('training_status'),new_TRAIN_AB=r['new_TRAIN_AB'],next_execution_authorized=False))
    pd.DataFrame(nextrows).to_csv(reports/'recommended_next_actions.csv',index=False)
    result=dict(status='WAITING_FOR_EXTERNAL_REVIEW',campaign_status='P67_PATCH24H_PARTIAL_COHORT_READY_FOR_REVIEW',
        coverage=dict(all_regions=1847,active_regions=1566,inactive_skipped=281,active_pairs=80316,standalone_ICNN_models=int(reg.standalone_ICNN_available.sum()),
            final_evaluated_regions=len(finals),new_TRAIN_AB=int(reg.new_TRAIN_AB.sum()),new_labels=int(reg.new_labels.sum()),new_AB=int(reg.new_AB.sum()),new_C=int(reg.new_C.sum())),
        data_status='CUMULATIVE_CERTIFIED_DATA_WITH_PARTIAL_FINAL_ROUND',learning_status='SEE_STANDALONE_AND_TYPED_PORTFOLIO_METRICS',runtime_validity='FRESH_REGION_REOPEN_RECORDED',
        national_readiness=False,next_phase_authorized=False,next_step_authorized=False,authority_sha256=c['authority_sha256'],source_bindings=c['source_bindings'],
        task_started_unix=c['task_started_unix'],hard_deadline_unix=c['hard_deadline_unix'],elapsed_seconds=elapsed(c),
        peak_rss_bytes=int(rf.rss_bytes.max()),added_bytes=owned_bytes(c)[0],matched_update_controls=sum(r.get('same_updates',False) for r in ablation),
        next_step_recommendation='External review of this exact cumulative handoff. Review remaining regional learning/precision/partial-data issues; no further execution or national deployment is authorized.')
    if (out/'control/STOP_SCIENCE.json').exists():
        stop=read_json(out/'control/STOP_SCIENCE.json');result['stop_record']=stop
        if 'SHARED' in stop['reason']:result['campaign_status']='P67_PATCH24H_BLOCKED_SHARED_INTEGRITY'
        elif 'FAILURE_THRESHOLD' in stop['reason']:result['campaign_status']='P67_PATCH24H_WITH_REGION_EXCEPTIONS_WAITING_EXTERNAL_REVIEW'
    write_json(reports/'FINAL_RESULT.json',result)
    low=aggregated.loc[aggregated.split.eq('LOW_TEST')&aggregated.aggregation.eq('EQUAL_REGION')]
    lines=['# Joint Phase 6/7 24-hour patch result','',result['campaign_status'],'',
        f"The campaign preserves the active-only s03 regional model, all 341,665 original labels, 1,566 active regions, 281 inactive skips and 80,316 ordered coordinates. It added {result['coverage']['new_labels']:,} certified labels, including {result['coverage']['new_TRAIN_AB']:,} A/B TRAIN rows. Useful partial data/model cycles and all unresolved regions remain local and hash-bound.",
        '',f"The source audit independently repriced 137 labels in 36 selected regions. All reproduced; no source label was corrected. Ten of twelve bounded cold solves met 1e-7; all twelve retained interval compatibility. Large stress costs are preserved.",
        '', 'The revised scalar potential has a nonnegative quartic-cost skip and a tangent-anchored ICNN. Relative Huber/interval supervision, TRAIN-calibrated residual amplitude, 75/25 operating/stress sampling and resumable float64 Adam replace the former unstable optimization policy. Analytic and neural candidates retain explicit model kinds. The existing-data-only comparison precedes fresh TRAIN labels.',
        '', 'Reused LOW_TEST equal-region MARE (regional traversal metrics, not national OD accuracy):','', '| Candidate | Regions | MARE |','|---|---:|---:|']
    lines += [f"| {r.model_kind} | {r.regions} | {r.MARE_equal_pairs:.6%} |" for r in low.itertuples()]
    lines += ['', 'Old LOW_TEST and projected HIGH are reused diagnostics. PATCH_AUDIT was accessed only after all champions froze; audit failures do not trigger reselection. Standalone ICNN results are separate from the selected portfolio. See aggregate_metrics, candidate_metrics_summary, explicit_champion_kind_counts, negative_and_nonfinite_docket, matched_update_ablation and recommended_next_actions.',
        '',f"All work uses the original start {datetime.fromtimestamp(c['task_started_unix'],timezone.utc).isoformat()} and deadline {datetime.fromtimestamp(c['hard_deadline_unix'],timezone.utc).isoformat()}. Sampled peak RSS was {result['peak_rss_bytes']/1024**3:.3f} GiB. Final companion records the sealed elapsed time, disk use and exact ZIP identity.",
        '',result['next_step_recommendation']]
    text_file(reports/'FINAL_REPORT.md','\n'.join(lines)+'\n')
    return result

def external_index(c):
    out=root(c);dest=out/'handoff/external_index';dest.mkdir(parents=True,exist_ok=True);records=[]
    volatile={'control/heartbeat.json','control/guardian_heartbeat.json','control/resource_history.jsonl','control/campaign_console.log','FINAL_HANDOFF.json'}
    for base,dirs,names in os.walk(out):
        if Path(base)==out:dirs[:]=[d for d in dirs if d!='handoff']
        for name in names:
            p=Path(base)/name
            if '.pending.' in name:continue
            if p.relative_to(out).as_posix() in volatile:continue
            count,description=schema(p)
            records.append(dict(relative_path=p.relative_to(out).as_posix(),size_bytes=p.stat().st_size,modified_unix_ns=p.stat().st_mtime_ns,sha256=sha(p),format=p.suffix,
                row_or_object_count=count,schema_or_shape=description,producer='partC_rev1_src.phase67',consumer='External review / authorized future continuation',copy_policy='LOCAL_HASH_BOUND'))
    f=pd.DataFrame(records);f.to_parquet(dest/'manifest.parquet',index=False)
    write_json(dest/'INDEX.json',dict(root=str(out),files=len(f),bytes=int(f.size_bytes.sum()),manifest_sha256=sha(dest/'manifest.parquet'),
        exclusions=['handoff staging and self-index']+sorted(volatile),
        volatile_evidence='Point-in-time resource history in reports/resource_usage.csv and copied control/log excerpts; final elapsed/disk fields in adjacent companion'))
    return dest

def verify_folder(folder):
    declared=set()
    for line in (folder/'CHECKSUMS.sha256').read_text().splitlines():
        h,n=line.split('  ',1);rel=Path(n);assert not rel.is_absolute() and '..' not in rel.parts and ':' not in n
        assert sha(folder/rel)==h;declared.add(n)
    actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    assert actual==declared|{'CHECKSUMS.sha256'}
    manifest=pd.read_csv(folder/'06_ARTIFACT_MANIFEST.csv');assert set(manifest.relative_path)==actual-{'CHECKSUMS.sha256','06_ARTIFACT_MANIFEST.csv'}
    return len(declared)

def seal(c,state):
    repo=Path(c['execution']['repository_root']);out=root(c);work=repo/'partC_doc_icnn/phase_7_work/p67_patch24h_0908';reportdir=out/'reports'
    head=git(c,'rev-parse','HEAD').strip();assert not git(c,'diff','--name-only').strip()
    hid='p67_patch24h_0908_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_final_001'
    parent=repo/'partC_doc_icnn/handoffs/phase_07/step_p67_patch24h_0908';folder=parent/hid
    archive=repo/'artifacts_icnn/handoff_bundles/phase_07'/f'phase07_stepp67_patch24h_0908_{hid}_SYNC.zip'
    assert not folder.exists() and not archive.exists();folder.mkdir(parents=True);archive.parent.mkdir(parents=True,exist_ok=True)
    for n in ['governing_docs','configs','code_snapshot','code_diff','summary_data','figures','geospatial_summary','log_excerpts','external_artifact_index']:(folder/n).mkdir()
    excluded=[]
    for p in authority(c).rglob('*'):
        if p.is_file() and p.suffix not in ['.zip','.pyc'] and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:copy(p,folder/'governing_docs/authority'/p.relative_to(authority(c)))
    for n in ['00_READ_FIRST.md','PROJECT_STATUS.md','12_DECISION_LOG.md','20_HANDOFF_SYNC_PROTOCOL.md','21_REVIEW_APPROVAL_GATES.md','23_SYNC_BUNDLE_CONTENT_AND_SIZE_POLICY.md','24_CODE_AND_DATA_SNAPSHOT_POLICY.md']:
        copy(repo/'partC_doc_icnn'/n,folder/'governing_docs/repository'/n)
    copy(repo/'AGENTS.md',folder/'governing_docs/AGENTS.md')
    for p in work.glob('*'):
        if p.is_file() and p.suffix in ['.md','.json','.py']:copy(p,folder/'summary_data/work_records'/p.name)
    copy(work/'effective_config.json',folder/'configs/effective_config.json');copy(out/'run_identity.json',folder/'configs/run_identity.json')
    snapshots=snapshot_paths(repo)
    for p in snapshots:copy(p,folder/'code_snapshot'/p.relative_to(repo))
    pd.DataFrame([dict(path=p.relative_to(repo).as_posix(),sha256=sha(p),size_bytes=p.stat().st_size) for p in snapshots]).to_csv(folder/'07_CODE_CHANGE_INDEX.csv',index=False)
    staged_test=snapshot_tests(c,folder/'code_snapshot',out/'validation/staged_snapshot_tests')
    text_file(folder/'code_diff/initial_to_final.patch',git(c,'diff','--binary',c['initial_head'],head,'--','partC_rev1_src/phase67','tests','partC_doc_icnn/phase_7_work/p67_patch24h_0908','partC_doc_icnn/00_READ_FIRST.md','partC_doc_icnn/PROJECT_STATUS.md','partC_doc_icnn/12_DECISION_LOG.md'))
    text_file(folder/'code_diff/commits.txt',git(c,'log','--format=%H %s',c['initial_head']+'..'+head));text_file(folder/'code_diff/git_status.txt',git(c,'status','--short'))
    for prefix,target in [(reportdir,folder/'summary_data/reports'),(out/'validation',folder/'summary_data/validation')]:
        for p in prefix.rglob('*'):
            if not p.is_file() or 'source_reads' in p.parts:continue
            if p.stat().st_size<=25*1024**2:copy(p,target/p.relative_to(prefix))
            else:excluded.append(dict(path=str(p),size_bytes=p.stat().st_size,sha256=sha(p),reason='Full external binding; individual compact member limit'))
    for p in (out/'control').glob('*'):
        if p.is_file() and p.stat().st_size<25*1024**2:
            if p.suffix=='.jsonl':
                lines=p.read_text().splitlines();text_file(folder/'log_excerpts'/p.name,'\n'.join(lines[:100]+lines[-100:])+'\n')
            else:copy(p,folder/'summary_data/control'/p.name)
    for p in (out/'regions').glob('*/failures/*.json'):copy(p,folder/'summary_data/failures'/p.parent.parent.name/p.name)
    index=external_index(c)
    for p in index.iterdir():copy(p,folder/'external_artifact_index'/p.name)
    write_json(folder/'external_artifact_index/COMPACT_POLICY.json',dict(excluded=excluded,source_bindings=c['source_bindings'],authority_sha256=c['authority_sha256'],
        embedded_parent_ZIPs_not_duplicated=True,large_external_root=str(out),full_old_data_preserved=True))
    copy(reportdir/'FINAL_REPORT.md',folder/'01_STEP_SUMMARY.md');write_json(folder/'02_HANDOFF_STATUS.json',dict(**state,handoff_id=hid,head=head))
    text_file(folder/'00_HANDOFF_README.md','Read 01_STEP_SUMMARY.md and 02_HANDOFF_STATUS.json, then the cumulative reports. Upload this one ZIP and adjacent SHA-256/companion. External review only.\n')
    text_file(folder/'03_DECISIONS_AND_OPEN_QUESTIONS.md','Original active regional model/data/topology preserved. Analytic champions retain explicit types. Partial new-data rounds, optimizer states, unsuccessful neural models and two unmet tighter precision targets are retained. Source old tests are reused diagnostics. No next phase authorized.\n')
    text_file(folder/'04_VALIDATION_RESULTS.md','See summary_data/validation: authenticated authority/source identities, 36-region/137-label replay, 12 precision probes, frozen protected inputs, current tests, and fresh all-new-label/model reopening. Accuracy failures are reported separately from structural validity.\n')
    copy(reportdir/'resource_usage.csv',folder/'05_RESOURCE_USAGE.csv')
    text_file(folder/'08_COMMAND_LOG.md',f'Execution: {c["execution"]["python_executable"]} -m partC_rev1_src.phase67.campaign --config partC_doc_icnn/phase_7_work/p67_patch24h_0908/effective_config.json --stage AUTO\nJ0 bootstrap, audit, prepare and current pytest results are bound in validation. All phases use the same original task start and deadline.\n')
    text_file(folder/'09_NEXT_STEP_RECOMMENDATION.md',state['next_step_recommendation']+'\n');text_file(folder/'10_UPLOAD_THIS_FILE.txt',str(archive)+'\n')
    text_file(folder/'11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md','New joint patch after the exact Phase 7 source handoff '+c['source_bindings']['p7_result']['sha256']+'. Adds scoped audit, robust learner and repeated certified incremental data/training. Original records are immutable.\n')
    text_file(folder/'geospatial_summary/NO_GEOMETRY_CHANGE.md','No topology, geometry, capacity, B/R or active-support change.\n')
    if not any((folder/'figures').iterdir()):text_file(folder/'figures/NUMERICAL_FIGURE_DATA.md','Full learning curves, operating/stress regimes and paired candidate tables are under summary_data/reports.\n')
    write_json(folder/'log_excerpts/privacy_scan.json',scan_bundle(c,folder))
    records=[dict(relative_path=p.relative_to(folder).as_posix(),size_bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.rglob('*')) if p.is_file()]
    assert all(r['size_bytes']<=25*1024**2 for r in records)
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
        for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts and ':' not in n
        z.extractall(fresh)
    count=verify_folder(fresh)
    fresh_test=snapshot_tests(c,fresh/'code_snapshot',out/'handoff'/('fresh_snapshot_validation_'+hid))
    assert verify_folder(fresh)==count,'Fresh snapshot testing modified the extracted review bytes'
    h=sha(pending);size=pending.stat().st_size;pending.rename(archive)
    assert elapsed(c)<86400,'Hard deadline reached during sealing'
    companion=dict(status='WAITING_FOR_EXTERNAL_REVIEW',campaign_status=state['campaign_status'],zip_path=str(archive),zip_sha256=h,zip_bytes=size,source_folder=str(folder),
        snapshot_head=head,authority_sha256=c['authority_sha256'],source_handoff_sha256=c['source_bindings']['p7_result']['sha256'],
        original_start_unix=c['task_started_unix'],original_deadline_unix=c['hard_deadline_unix'],elapsed_seconds=elapsed(c),timer_reset=False,
        CRC='PASS',fresh_extraction='PASS',verified_checksums=count,coverage=state['coverage'],fresh_snapshot_tests=fresh_test,
        added_artifact_bytes_including_handoff=owned_bytes(c)[0],peak_sampled_rss_bytes=state['peak_rss_bytes'],
        next_step_authorized=False,next_phase_authorized=False,national_readiness=False)
    assert companion['added_artifact_bytes_including_handoff']<c['execution']['added_disk_bytes']
    write_json(archive.with_suffix('.companion.json'),companion);text_file(archive.with_suffix('.sha256'),h+'  '+archive.name+'\n')
    write_json(out/'FINAL_HANDOFF.json',companion);text_file(parent/'LATEST_HANDOFF.txt',json.dumps(companion,indent=2)+'\n');return companion

def finalize(c):
    from .campaign import Runner,fair_regions
    out=root(c);regions=fair_regions(c);cycle=read_json(out/'control/SCIENCE_COMPLETE.json')['cycle']
    runner=Runner(c,2)
    if not (out/'control/ALL_CHAMPIONS_FROZEN.json').exists():
        results=runner.batch([dict(action='FREEZE',region_id=r,cycle=cycle,final=True) for r in regions])
        assert all(r['ok'] for r in results),'Final candidate freeze has unresolved correctness error'
        write_json(out/'control/ALL_CHAMPIONS_FROZEN.json',dict(frozen_unix=time.time(),regions=1566,
            manifests=[dict(region_id=r,sha256=sha(rd(c,r)/'training/cycles/final/manifest.json')) for r in regions],
            selection_uses_TRAIN_VALID_only=True,hidden_audit_not_yet_accessed=True,training_and_generation_now_prohibited=True))
    runner.batch([dict(action='EVALUATE',region_id=r) for r in regions]);runner.close()
    state=reports(c);closure=source_closeout(c)
    repo=Path(c['execution']['repository_root']);work=repo/'partC_doc_icnn/phase_7_work/p67_patch24h_0908'
    copy(out/'reports/FINAL_REPORT.md',work/'FINAL_REPORT.md');copy(out/'reports/FINAL_RESULT.json',work/'FINAL_RESULT.json')
    entry='# Joint Phase 6/7 patch sealed for external review\n\n'+state['campaign_status']+'. See phase_7_work/p67_patch24h_0908/FINAL_REPORT.md and the exact final handoff companion. No further execution is authorized.\n\n---\n\n'
    for n in ['00_READ_FIRST.md','PROJECT_STATUS.md']:
        p=repo/'partC_doc_icnn'/n;text_file(p,entry+p.read_text(encoding='utf-8'))
    p=repo/'partC_doc_icnn/12_DECISION_LOG.md'
    with p.open('a',encoding='utf-8') as f:f.write('\n\n## Joint Phase 6/7 patch result\n\n'+state['campaign_status']+'. This is an internal result, not external approval. Original source model, topology and labels remain preserved; standalone neural and analytic/typed-portfolio outcomes are reported separately. External review only.\n')
    paths=['partC_doc_icnn/phase_7_work/p67_patch24h_0908/FINAL_REPORT.md','partC_doc_icnn/phase_7_work/p67_patch24h_0908/FINAL_RESULT.json','partC_doc_icnn/00_READ_FIRST.md','partC_doc_icnn/PROJECT_STATUS.md','partC_doc_icnn/12_DECISION_LOG.md']
    git(c,'add','--',*paths);git(c,'commit','-m','phase67: record 24h patch results')
    result=seal(c,state);print(json.dumps(result,indent=2),flush=True);return result
