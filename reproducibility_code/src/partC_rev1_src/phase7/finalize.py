"""One compact review handoff, fresh extraction, exact hashes and no next phase."""
from __future__ import annotations
import argparse,ast,hashlib,io,json,os,shutil,subprocess,time,zipfile,re
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .common import root,region_out,read_json,write_json,sha,load_config,elapsed
from .guardian import tree_bytes

def git(c,*args):
    p=subprocess.run(['git','-C',c['execution']['repository_root'],'-c','core.longpaths=true',*args],capture_output=True,text=True,encoding='utf-8',check=True)
    return p.stdout
def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
def text_file(path,text):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')

def schema(path):
    ext=path.suffix.lower()
    if ext=='.npy':
        a=np.load(path,mmap_mode='r',allow_pickle=False);return int(a.shape[0]) if a.ndim else 1,json.dumps(dict(shape=list(a.shape),dtype=str(a.dtype)))
    if ext=='.npz':
        with np.load(path,allow_pickle=False) as z:return len(z.files),json.dumps({k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype)) for k in z.files})
    if ext=='.parquet':
        import pyarrow.parquet as pq
        meta=pq.read_metadata(path);return meta.num_rows,str(meta.schema)
    if ext in ['.csv','.jsonl']:
        with path.open(encoding='utf-8') as f:head=f.readline();count=sum(1 for _ in f)
        return count+(ext=='.jsonl'),head.strip()[:500]
    if ext=='.json':
        obj=read_json(path);return len(obj) if hasattr(obj,'__len__') else 1,type(obj).__name__
    if ext=='.sqlite':
        import sqlite3
        con=sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True)
        tables=[r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        details={}
        for table in tables:
            name='"'+table.replace('"','""')+'"'
            details[table]=dict(rows=con.execute('SELECT COUNT(*) FROM '+name).fetchone()[0],columns=con.execute('PRAGMA table_info('+name+')').fetchall())
        con.close();return sum(r['rows'] for r in details.values()),json.dumps(details)
    return None,'Trusted local checkpoint; weights_only loading' if ext=='.pt' else ext

def external_index(c):
    out=root(c);dest=out/'manifests';dest.mkdir(exist_ok=True);records=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file() or dest in p.parents or 'preview_reports' in p.parts or p.name.endswith('.pending'):continue
        count,description=schema(p)
        records.append(dict(relative_path=p.relative_to(out).as_posix(),absolute_path=str(p),size_bytes=p.stat().st_size,modified_unix_ns=p.stat().st_mtime_ns,
            sha256=sha(p),format=p.suffix,row_or_object_count=count,schema_or_shape=description,producer='partC_rev1_src.phase7',consumer='External Phase 7 review only',copy_policy='LOCAL_HASH_BOUND'))
    f=pd.DataFrame(records);f.to_parquet(dest/'external_artifact_manifest.parquet',index=False)
    f.to_csv(dest/'external_artifact_manifest.csv',index=False)
    write_json(dest/'external_INDEX.json',dict(root=str(out),files=len(f),bytes=int(f.size_bytes.sum()),manifest_path=str(dest/'external_artifact_manifest.parquet'),manifest_sha256=sha(dest/'external_artifact_manifest.parquet'),
        exclusions='This manifest/index itself and preview reports; source Phase 6 remains separately bound by its exact reviewed inventories.'))
    return f

def overall_status(registry,models):
    active=registry.loc[registry.m_active.gt(0)];partial=active.status.str.contains('PARTIAL').any();pending=active.status.str.startswith('PENDING').any() or active.status.eq('TRAINING').any()
    if partial or pending:return 'PHASE7_8H_PARTIAL_COHORT_READY_FOR_REVIEW'
    if len(models)<1566 or active.status.str.contains('FAILURE').any():return 'PHASE7_COMPLETE_WITH_REGION_EXCEPTIONS_WAITING_EXTERNAL_REVIEW'
    return 'PHASE7_COMPLETE_WAITING_EXTERNAL_REVIEW'

def prepare_summary(c):
    out=root(c);reports=out/'reports';registry=pd.read_parquet(reports/'all_region_registry.parquet');models=pd.read_parquet(reports/'model_manifest.parquet')
    resources=[]
    for name in ['resource_timeline.jsonl','guardian_resource_timeline.jsonl']:
        if (out/name).exists():resources.extend(json.loads(s) for s in (out/name).read_text().splitlines())
    resource_table=pd.DataFrame(resources).sort_values('unix')
    resource_table.to_parquet(reports/'resource_usage.parquet',index=False);resource_table.to_csv(reports/'resource_usage.csv',index=False)
    cov=read_json(reports/'coverage_summary.json');reopen=read_json(out/'validation/FRESH_REOPEN_RESULT.json')
    if reopen['status']!='PASS':raise ValueError('Fresh reopen failed; must preserve and explicitly repair affected scope before seal')
    assert read_json(out/'validation/FINAL_CAMPAIGN_AUDIT.json')['status']=='PASS'
    assert read_json(out/'validation/FIGURE_VISUAL_QA.json')['status']=='PASS'
    ag=pd.read_parquet(reports/'aggregate_metrics.parquet');headline=ag.loc[ag.model.eq('ICNN')&ag.family.eq('ALL')&ag.grade.eq('ALL')]
    metric_table=pd.read_parquet(reports/'metrics_by_region_split.parquet')
    main_metrics=metric_table.loc[metric_table.model.eq('ICNN')]
    valid_screens=int(main_metrics.loc[main_metrics.split.eq('VALID'),'accuracy_screen'].sum())
    negative_regions=main_metrics.loc[main_metrics.negative_cost_count.gt(0),'region_id'].nunique()
    worst=main_metrics.sort_values('MARE_equal_pairs',ascending=False).groupby('split',sort=False).head(10)
    worst.to_parquet(reports/'worst_regional_errors.parquet',index=False);worst.to_csv(reports/'worst_regional_errors.csv',index=False)
    low_summary=ag.loc[ag.split.eq('LOW_TEST')&ag.family.eq('ALL')&ag.grade.eq('ALL')&ag.aggregation.eq('REGION_UNWEIGHTED')].set_index('model')
    counts=pd.read_parquet(reports/'source_split_quality_totals.parquet');runtime=pd.read_parquet(reports/'resource_usage.parquet')
    docket=pd.read_parquet(reports/'next_action_docket.parquet');status=overall_status(registry,models)
    benchmark=pd.read_parquet(reports/'inference_benchmarks.parquet')
    active=registry.loc[registry.m_active.gt(0)]
    timing=dict(device='cpu',dtype='float64',workers=2,native_threads_per_worker=1,
        measured_model_count=len(models),covered_active_coordinates=int(models.m_active.sum()),all_evaluated_scenario_coordinates=int(benchmark.coordinates.sum()),
        Torch_all_saved_scenarios_potential_and_gradient_seconds=float(benchmark.torch_seconds.sum()),portable_all_saved_scenarios_potential_and_gradient_seconds=float(benchmark.portable_seconds.sum()),
        portable_all_models_one_LOW_query_seconds=reopen['total_portable_LOW_query_seconds'],
        sum_optimizer_seconds=float(active.train_seconds.sum()),sum_TRAIN_VALID_check_seconds=float(active.validation_seconds.sum()),sum_pack_seconds=float(active.pack_seconds.sum()),
        region_optimizer_seconds_p50=float(active.train_seconds.median()),region_optimizer_seconds_p95=float(active.train_seconds.quantile(.95)),region_optimizer_seconds_max=float(active.train_seconds.max()),
        portable_batch1_value_gradient_HVP_seconds_p50=float(benchmark.drop_duplicates('region_id').batch1_HVP_seconds.median()),
        portable_batch1_value_gradient_HVP_seconds_p95=float(benchmark.drop_duplicates('region_id').batch1_HVP_seconds.quantile(.95)),
        interpretation='Checkpoint-accounted task-local kernel timings; sums are serial-equivalent measured CPU work. Discarded in-flight optimizer work after native interruption is not reconstructed; whole-task elapsed includes it. No national assignment speedup is inferred.')
    write_json(reports/'RUNTIME_SUMMARY.json',timing)
    diagnostics_path=out/'diagnostics/nested_train_subsets/DIAGNOSTIC_RESULT.json'
    diagnostics=read_json(diagnostics_path) if diagnostics_path.exists() else dict(status='NOT_EXECUTED',fits=0,wall_seconds=0)
    expected=c['source_bindings']['expected_counts'];observed=counts.groupby('quality')['count'].sum().to_dict()
    if (root(c)/'TRAINING_CLOSED.json').exists():trainclosed=read_json(root(c)/'TRAINING_CLOSED.json')
    else:trainclosed={}
    split_ab=counts.loc[counts.quality.isin(['A','B'])].groupby('split')['count'].sum().to_dict()
    if registry.counts_verified.fillna(False).sum()==1566:
        assert observed=={'A':314826,'B':26767,'C':72}
        assert counts.groupby('split')['count'].sum().to_dict()=={'HIGH_PROJECTED':1554,'LOW_TEST':12358,'TRAIN':315807,'VALID':11946}
    state=dict(status='WAITING_FOR_EXTERNAL_REVIEW',campaign_status=status,phase_id='07',step_id=c['step_id'],run_id=c['execution']['run_id'],
        authority_sha256=c['authority_sha256'],source_handoff_sha256=c['source_handoff_sha256'],source_phase6_status='APPROVED_COMPLETE_WITH_NOTES',
        start_unix=c['task_started_unix'],hard_deadline_unix=c['hard_deadline_unix'],actual_elapsed_seconds=elapsed(c),timer_reset=False,
        pre_run_head=read_json(out/'validation/T0_PASS.json')['pre_run_head'],initial_head=c['initial_head'],training_closed=trainclosed,
        coverage=cov,source_grade_counts_observed=observed,source_independent_AB_split_counts=split_ab,source_labels_generated=0,
        peak_sampled_training_tree_RSS_bytes=int(runtime.rss_bytes.max()),max_sampled_added_output_bytes=int(runtime.added_output_bytes.max()),
        input_dimension='Exact ordered m_active; no full support padding',architecture='Tangent-anchored 2x32 scalar-potential ICNN',
        fixed_topology=dict(nodes=72724,static_links=143603,regional_links=80316,logical_links=223919,inactive_regions_without_model=281),
        frozen_model_parameters=int((64*models.m_active+1120).sum()),learning_accuracy='See exact VALID/LOW_TEST/HIGH cohort metrics; execution completeness does not imply accuracy',
        frozen_VALID_screen_pass_regions=valid_screens,frozen_VALID_screen_fail_regions=len(models)-valid_screens,regions_with_observed_materially_negative_total_cost=negative_regions,
        national_readiness=False,national_assignment_performed=False,phase6_generation_performed=False,B_restoration=False,inactive_pair_fallback=False,
        next_step_authorized=False,next_phase_authorized=False,source_known_comparator_warnings=9,source_conditional_comparator_premises=252,
        engineering_repair_epochs=3,current_execution_gate=read_json(out/'validation/CURRENT_EXECUTION_GATE.json'),export_interface_gate=read_json(out/'validation/T4_EXPORT_GATE.json'),
        diagnostics=dict(status=diagnostics['status'],fits=diagnostics['fits'],wall_seconds=diagnostics['wall_seconds'],main_checkpoints_unchanged=diagnostics.get('main_checkpoints_unchanged')),
        runtime_summary=timing,fresh_reopen=reopen,next_step_recommendation='External review of this exact handoff. Consider the evidence-ranked epoch/data/optimizer/label-precision docket; no continuation or new data is authorized.')
    write_json(reports/'FINAL_RESULT.json',state)
    errorlines=[]
    for r in headline.itertuples():
        errorlines.append(f'| {r.split} | {r.aggregation} | {r.covered_regions} | {r.covered_active_pairs} | {r.covered_LOW_exposure_fraction:.4%} | {100*r.MARE_equal_pairs:.5g}% | {100*r.MARE_LOW_pair_weighted:.5g}% | {r.RMSE_seconds:.6g} | {100*r.mean_interval_violation_relative:.5g}% |')
    flags=docket.recommendations.str.split(';').explode().value_counts().to_dict()
    allsteps=registry.loc[registry.m_active.gt(0),'steps'].fillna(0).sum()
    text=(f'# Phase 7 s03 ICNN campaign result\n\n{status}; WAITING_FOR_EXTERNAL_REVIEW.\n\n'
        f'Trained and evaluated cohort: {cov["trained_evaluated_regions"]}/1566 active regions, {cov["trained_evaluated_active_pairs"]}/80316 active pairs '
        f'and {cov["trained_exposure_fraction"]:.4%} of LOW regional traversal exposure. All 281 inactive regions have no ICNN. '
        f'There were {int(allsteps):,} optimizer updates across attempted regions. Absent or invalid models are excluded from errors, never zero-filled.\n\n'
        f'Empirical accuracy remains inadequate for a general deployment claim: {valid_screens}/1566 frozen checkpoints pass the VALID research screen. '
        f'Region-unweighted LOW_TEST equal-pair MARE is {100*low_summary.loc["ICNN","MARE_equal_pairs"]:.4f}% for ICNN, '
        f'{100*low_summary.loc["FREEFLOW_LINEAR","MARE_equal_pairs"]:.4f}% for freeflow and {100*low_summary.loc["SEPARABLE_POWER4_NNLS","MARE_equal_pairs"]:.4f}% for the TRAIN-fit power-4 baseline. '
        f'{negative_regions} models produce materially negative total costs on at least one observed evaluated input and are additionally unsuitable for deployment. '
        'Large error tails are retained in worst_regional_errors and the full metric tables. The selected best-composite checkpoint screen is reported separately from the last optimizer-state screen.\n\n'
        f'The supplied authority is `{c["authority_sha256"]}`. Phase 6 is accepted with notes under reviewed ZIP `{c["source_handoff_sha256"]}` '
        f'and checkpoint `{c["source_bindings"]["expected_final_head"]}`. Its 341,665 coarse labels remain read-only; no label or assignment was generated. '
        f'Observed independent A/B counts by canonical split: {json.dumps(split_ab,sort_keys=True)}. C labels are excluded from fitting and selection.\n\n'
        'Every model uses the exact ordered active dimension, signed input skips, nonnegative hidden/output weights and the scalar tangent anchor. '
        'Costs are the native derivative of the potential. Source input_scale, c0 and Vscale are preserved; all additional scales, gain and power-4 coefficients use TRAIN only. '
        'No independent cost head, clipping, inactive fallback, B restoration or topology change was introduced.\n\n'
        'The fixed canonical TRAIN/VALID/LOW_TEST/HIGH ownership and aliases were audited. VALID composite loss selected checkpoints; '
        'LOW_TEST and projected HIGH were evaluated only after best-checkpoint freeze. Projected HIGH is reused sensitivity, and LOW itself is a training anchor. '
        'Finite A/B label gaps certify potential intervals, not exact marginal coordinates.\n\n'
        'Errors below are among the named trained/evaluated cohort with each split. Weighted summaries use LOW traversal exposure, which is not deduplicated national OD demand.\n\n'
        '| Split | Aggregation | Regions | Pairs | LOW exposure coverage | Equal-pair MARE | LOW-pair MARE | Regional RMSE summary (seconds) | Mean interval violation |\n'
        '|---|---|---:|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(errorlines)+'\n\n'
        'Complete region/split/family/grade metrics include native MAE, RMSE, p50/p90/p95/max errors, signed bias, interval violations, congestion residuals, '
        'floor and negative-cost counts, zero-demand coordinates and observed-domain flags. Regional tail summaries are not pooled-coordinate percentiles. '
        'FREEFLOW_LINEAR and TRAIN-fit SEPARABLE_POWER4_NNLS are reported on identical rows, with paired ICNN-minus-baseline deltas.\n\n'
        f'Coverage-first queue and per-slice receipts document first attempts before longer fitting. Terminal status counts: {json.dumps(cov["status_counts"],sort_keys=True)}. '
        'Last/best weights, Adam, RNG, shuffle/cursor, epochs, updates and unique-row exposure are preserved locally. One update alone is never described as completed training.\n\n'
        f'Training closed at elapsed {trainclosed.get("elapsed_seconds",float("nan")):.2f} seconds. At report preparation the whole-task elapsed time is {elapsed(c):.2f} seconds '
        f'from UTC {datetime.fromtimestamp(c["task_started_unix"],timezone.utc).isoformat()}; hard deadline is {datetime.fromtimestamp(c["hard_deadline_unix"],timezone.utc).isoformat()}. '
        f'Peak sampled training process-tree RSS: {state["peak_sampled_training_tree_RSS_bytes"]/2**30:.3f} GiB; maximum sampled added output: {state["max_sampled_added_output_bytes"]/2**30:.3f} GiB. '
        'All implementation, testing, data packing, training, repair, reporting and sealing share that original clock. The companion records final elapsed time and added disk.\n\n'
        'The installed psutil disk-space function failed before the first valid heartbeat. Engineering epoch 1 substituted the standard-library disk-space API '
        'and added an external guardian for the live workers. The brief initial resource-evidence gap and original tracebacks are preserved. The training admission clock never reset. '
        'Engineering epoch 2 followed a native Windows python312.dll access violation while resolving a source path. Its underlying native cause is unproven. '
        'The two interrupted, incomplete packs were archived and repacked; 1,151 saved first attempts were retained. Source paths now use a once-resolved root plus strict lexical checks and the same exact reviewed path/hash membership. '
        'A later native interruption during checkpoint serialization was recovered automatically by that pool, without restarting the coordinator or changing code. '
        'The two affected regions resumed; the three duplicated saved validation checks on the collateral region have exactly equal TRAIN/VALID metrics. Raw logs remain intact and replay duplicates are marked and excluded from trend weighting. '
        'Both original and repaired code, Windows crash evidence and dependency scope are preserved. '
        'Engineering epoch 3 separated the existing portable kernel from the Torch training-module import path. Its function AST is identical, a clean subprocess queries its value/gradient/HVP with Torch imports blocked, and all-model reopening exercises the independent module. No weights or equations changed. '
        'CPU float64, two workers and one native thread were retained. No package, environment, driver, network or CUDA change was made.\n\n'
        f'Checkpoint-accounted optimizer work summed to {timing["sum_optimizer_seconds"]:.3f} seconds; TRAIN/VALID checking summed to {timing["sum_TRAIN_VALID_check_seconds"]:.3f} seconds. '
        'Discarded in-flight work is not reconstructed in these sums; it remains included in the whole-task elapsed clock. '
        f'The full saved-scenario potential/gradient passes covered {timing["all_evaluated_scenario_coordinates"]:,} coordinates: {timing["Torch_all_saved_scenarios_potential_and_gradient_seconds"]:.3f} seconds in Torch and '
        f'{timing["portable_all_saved_scenarios_potential_and_gradient_seconds"]:.3f} seconds in portable NumPy/SciPy kernels. '
        f'A fresh LOW query across all produced models covered {timing["covered_active_coordinates"]:,} active coordinates in {timing["portable_all_models_one_LOW_query_seconds"]:.6f} summed seconds. '
        'These are measured kernel workloads; they do not estimate a national assignment speedup. Batch-one value/gradient/HVP timings and dimension-specific details are supplied separately.\n\n'
        f'Fresh-process verification reopened all {reopen["reopened_models"]} produced model/metric packs and checked {reopen["native_prediction_coordinates"]:,} prediction coordinates for finiteness, '
        'replayed reported primary metrics, queried every frozen model and repeated detailed derivative checks across dimension representatives. '
        'Portable value/gradient/HVP checks, structural/Jensen convexity, monotonicity, symmetry, finite differences and zero/freeflow anchors are included. '
        'T0 had 47 production tests plus 35 supplied reference tests; repair 1 had 48 and repair 2 had 50 production tests. Final and extracted-snapshot test receipts supply the current count.\n\n'
        f'Evidence-based recommendation flags: {json.dumps(flags,sort_keys=True)}. Last-20%-versus-prior-20% TRAIN/VALID changes are saved when enough checks exist. '
        'Continuing improvement at a compute limit supports an epochs candidate; a low TRAIN/high heldout gap suggests data-coverage review with model/regularization alternatives; '
        'high plateaus or a materially stronger separable baseline point to capacity or optimization review. Grade/interval associations can motivate label-precision review, '
        f'but do not isolate it causally. Optional nested data-size diagnostics: {diagnostics["status"]}, {diagnostics["fits"]} fits, {diagnostics["wall_seconds"]:.3f} seconds. '
        'The eight regions were fixed by the original dimension-interleaved queue before errors were examined; this sample covers four dimension strata and does not represent the largest stratum. '
        'The 25% and 50% subsets are nested, preserve ZERO/LOW, use only A/B TRAIN, keep the same regional initialization seed and architecture, and compare on VALID only. '
        'Different optimizer updates, convergence and subset-derived gain/scales remain explicit confounders. Main checkpoints are unchanged and no causal data-volume claim is made.\n\n'
        'National readiness remains false. The nine known source-path warnings, 252 conditional mapped-comparison premises, active-support pruning and global lifting/distance limitations '
        'are preserved. The provisional 5% VALID cost/1% interval screen is a research summary, not national shortest-path validation. '
        'The next permitted action is external review. Resume/data-expansion tables are recommendations only; no later phase or new Phase 6 samples are authorized.\n')
    text_file(reports/'FINAL_REPORT.md',text)
    work=Path(c['execution']['repository_root'])/'partC_doc_icnn/phase_7_work/s03_active_8h_v1'
    copy(reports/'FINAL_REPORT.md',work/'FINAL_REPORT.md');copy(reports/'FINAL_RESULT.json',work/'FINAL_RESULT.json')
    return state

def scan_bundle(c,source):
    # Reuse the inspected existing scanner's exact function without importing the
    # historical scientific pilot dependency graph. The whole file is snapshotted.
    file=Path(c['execution']['repository_root'])/'partC_rev1_src/phase5/handoff.py'
    tree=ast.parse(file.read_text(encoding='utf-8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_privacy_scan')
    env={'Path':Path,'re':re};exec(compile(ast.Module(body=[node],type_ignores=[]),str(file),'exec'),env)
    result=env['_privacy_scan'](source);remaining=[];typed=[]
    import xml.etree.ElementTree as ET
    for r in result['findings']:
        p=source/r['path']
        if r['pattern']=='unreviewed_binary_type' and p.suffix in ['.npz','.npy','.parquet']:
            count,desc=schema(p);typed.append(dict(**r,rows=count,schema=desc,status='PASS'))
        elif r['pattern']=='unreviewed_binary_type' and p.suffix=='.xml':
            ET.parse(p);typed.append(dict(**r,status='PASS'))
        else:remaining.append(r)
    result.update(findings=remaining,typed_format_reopens=typed,status='PASS' if not remaining else 'FAIL',reused_scanner_sha256=sha(file))
    if remaining:raise ValueError('Bundle privacy/text scan failed: '+str(remaining))
    return result

def verify_bundle_folder(folder):
    checks=[]
    for line in (folder/'CHECKSUMS.sha256').read_text().splitlines():
        h,p=line.split('  ',1)
        assert (folder/p).resolve().is_relative_to(folder.resolve())
        assert sha(folder/p)==h;checks.append(p)
    actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    assert actual==set(checks)|{'CHECKSUMS.sha256'}
    f=pd.read_csv(folder/'06_ARTIFACT_MANIFEST.csv')
    assert set(f.relative_path)==actual-{'CHECKSUMS.sha256','06_ARTIFACT_MANIFEST.csv'}
    assert all((folder/r.relative_path).stat().st_size==r.size_bytes for r in f.itertuples())
    return len(checks)

def bundle(c):
    out=root(c);repo=Path(c['execution']['repository_root']);reports=out/'reports';state=read_json(reports/'FINAL_RESULT.json')
    assert state['status']=='WAITING_FOR_EXTERNAL_REVIEW' and not state['next_phase_authorized']
    assert not git(c,'diff','--name-only').strip(),'Tracked work must be committed before final seal'
    head=git(c,'rev-parse','HEAD').strip();assert git(c,'log','-1','--format=%s').strip()=='phase7: record eight-hour training and evaluation result'
    hid='active_8h_v1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_final_001'
    parent=repo/'partC_doc_icnn/handoffs/phase_07/step_s03_active_8h_v1';source=parent/hid
    archive=repo/'artifacts_icnn/handoff_bundles/phase_07'/('phase07_steps03_active_8h_v1_'+hid+'_SYNC.zip')
    assert not source.exists() and not archive.exists();source.mkdir(parents=True);archive.parent.mkdir(parents=True,exist_ok=True)
    for n in ['governing_docs','configs','code_snapshot','code_diff','summary_data','figures','geospatial_summary','log_excerpts','external_artifact_index']:(source/n).mkdir()
    ext=external_index(c);copy(out/'manifests/external_artifact_manifest.parquet',source/'external_artifact_index/phase7_artifact_manifest.parquet');copy(out/'manifests/external_INDEX.json',source/'external_artifact_index/phase7_INDEX.json')
    auth=repo/c['execution']['authority_install_relative']/'authority'
    for p in auth.rglob('*'):
        if p.is_file() and p.suffix not in ['.zip','.pyc'] and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:
            copy(p,source/'governing_docs/authority'/p.relative_to(auth))
    for name in ['00_READ_FIRST.md','PROJECT_STATUS.md','12_DECISION_LOG.md','20_HANDOFF_SYNC_PROTOCOL.md','21_REVIEW_APPROVAL_GATES.md','23_SYNC_BUNDLE_CONTENT_AND_SIZE_POLICY.md','24_CODE_AND_DATA_SNAPSHOT_POLICY.md']:
        copy(repo/'partC_doc_icnn'/name,source/'governing_docs/repository'/name)
    if (repo/'AGENTS.md').exists():copy(repo/'AGENTS.md',source/'governing_docs/AGENTS.md')
    for p in (repo/'partC_doc_icnn/phase_7_work/s03_active_8h_v1').glob('*'):
        if p.is_file() and p.suffix in ['.md','.json','.py']:copy(p,source/'summary_data/work_records'/p.name)
    copy(out/'run_identity.json',source/'configs/run_identity.json');copy(out/'environment.json',source/'configs/environment.json')
    cp=repo/'partC_doc_icnn/phase_7_work/s03_active_8h_v1/effective_config.json';copy(cp,source/'configs/effective_config.json')
    snapshots=list((repo/'partC_rev1_src/phase7').glob('*.py'))+[repo/'partC_rev1_src/phase6/common.py',repo/'partC_rev1_src/phase6/__init__.py',repo/'partC_rev1_src/phase5/handoff.py',repo/'partC_rev1_src/__init__.py']+list((repo/'tests').glob('test_phase7_*.py'))
    snapshots=[p for p in snapshots if p.exists()]
    for p in snapshots:copy(p,source/'code_snapshot'/p.relative_to(repo))
    # Preserve the precise original coordinator alongside the resource-only repair.
    original_head=read_json(out/'validation/T0_PASS.json')['pre_run_head']
    text_file(source/'code_diff/original_execution_coordinator.py',git(c,'show',original_head+':partC_rev1_src/phase7/campaign.py'))
    # The synthetic tests resolve their explicit recipe relative to the snapshot.
    for p in (auth/'config').glob('*.json'):copy(p,source/'code_snapshot'/p.relative_to(repo))
    pd.DataFrame([dict(relative_path=p.relative_to(repo).as_posix(),sha256=sha(p),size_bytes=p.stat().st_size) for p in snapshots]).to_csv(source/'07_CODE_CHANGE_INDEX.csv',index=False)
    text_file(source/'code_diff/initial_to_final.patch',git(c,'diff','--binary',c['initial_head'],head,'--','partC_rev1_src/phase7','tests/test_phase7_campaign.py','tests/test_phase7_mathematics.py','tests/test_phase7_reporting.py','partC_doc_icnn/phase_7_work/s03_active_8h_v1','partC_doc_icnn/00_READ_FIRST.md','partC_doc_icnn/PROJECT_STATUS.md','partC_doc_icnn/12_DECISION_LOG.md'))
    text_file(source/'code_diff/commits.txt',git(c,'log','--format=%H %s',c['initial_head']+'..'+head));text_file(source/'code_diff/git_status.txt',git(c,'status','--short'))
    excluded=[]
    for p in reports.rglob('*'):
        if not p.is_file():continue
        target=source/('figures' if p.parent.name=='figures' else 'summary_data/reports')/p.name
        if p.stat().st_size<=25*2**20:copy(p,target)
        else:
            excluded.append(dict(path=str(p),size_bytes=p.stat().st_size,sha256=sha(p),reason='Full table remains external under member size policy'))
            if p.suffix=='.csv':
                pd.read_csv(p,nrows=2000).to_csv(target.with_name(target.stem+'_first2000.csv'),index=False)
    # Keep full all-region exception traces, validation controls and source witnesses.
    for p in (out/'validation').rglob('*'):
        if p.is_file() and 'reopen_regions' not in p.parts and p.stat().st_size<25*2**20:copy(p,source/'summary_data/validation'/p.relative_to(out/'validation'))
    for rd in (out/'regions').iterdir():
        if not rd.is_dir():continue
        for p in rd.glob('failure_*.json'):copy(p,source/'summary_data/region_failures'/rd.name/p.name)
    for name in ['training_queue.csv','coverage_summary.json','TRAINING_CLOSED.json','CAMPAIGN_EXECUTION_DONE.json','LAUNCH_LOCK.json','LAUNCH_EXIT.json','GUARDIAN_EXIT.json']:
        if (out/name).exists():copy(out/name,source/'summary_data/control'/name)
    # Full event/curve logs stay external; first/last events and all watchdog errors are compact evidence.
    for p in [out/'training_events.jsonl',out/'logs/watchdog_errors.jsonl']:
        if p.exists():
            lines=p.read_text().splitlines();selected=lines if len(lines)<400 else lines[:200]+lines[-200:]
            text_file(source/'log_excerpts'/p.name,'\n'.join(selected)+'\n')
    for p in (out/'logs').glob('*stacks.log'):
        if p.stat().st_size:
            lines=p.read_text().splitlines();text_file(source/'log_excerpts'/p.name,'\n'.join(lines[-400:])+'\n')
    diag=out/'diagnostics/nested_train_subsets'
    if diag.exists():
        for p in diag.rglob('*'):
            if p.is_file() and p.suffix not in ['.pt'] and p.stat().st_size<25*2**20:copy(p,source/'summary_data/diagnostics'/p.relative_to(diag))
    models=pd.read_parquet(reports/'model_manifest.parquet').sort_values('m_active')
    reps=list(dict.fromkeys(models.iloc[np.linspace(0,len(models)-1,min(6,len(models)),dtype=int)].region_id))
    for rid in reps:
        rd=region_out(c,rid);dest=source/'summary_data/representative_models'/rid
        for p in [rd/'model.json',rd/'evaluation/portable.npz',rd/'evaluation/mathematical_validation.json',rd/'pack/manifest.json',rd/'pack/scales.npz',rd/'pack/active_pairs.parquet']:
            copy(p,dest/('pack_manifest.json' if p.name=='manifest.json' else p.name))
        rows=pd.read_parquet(rd/'pack/rows.parquet');keep=[]
        for split in ['TRAIN','VALID','LOW_TEST','HIGH_PROJECTED']:
            ids=np.flatnonzero(rows.split.eq(split)&rows.quality.isin(['A','B']));keep.extend(ids[:2])
        keep=np.asarray(keep,int);sample={n:np.asarray(np.load(rd/'pack'/(n+'.npy'),mmap_mode='r',allow_pickle=False)[keep]) for n in ['inputs_native','costs_native','lower_native','upper_native','allowance_native']}
        from .model import frozen_eval
        with np.load(rd/'evaluation/portable.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        v,t=frozen_eval(a,sample['inputs_native']);sample.update(predicted_potential=v,predicted_cost=t)
        np.savez_compressed(dest/'native_prediction_sample.npz',**sample);rows.iloc[keep].to_csv(dest/'native_prediction_sample_rows.csv',index=False)
    write_json(source/'external_artifact_index/COMPACT_POLICY.json',dict(external_tables=excluded,representative_regions=reps,source_authority_zip=str(auth/'input'/c['source_bindings']['source_zip']),source_authority_sha256=c['source_handoff_sha256'],
        source_output_root=c['execution']['source_root'],source_metadata_index=str(out/'source_index.sqlite'),source_metadata_index_sha256=sha(out/'source_index.sqlite'),
        original_authority_manifest_domain='governing_docs/authority CHECKSUMS and MANIFEST refer to the intact supplied ZIP, whose embedded parent stays external here; this handoff has its own complete root manifest.'))
    copy(reports/'FINAL_REPORT.md',source/'01_STEP_SUMMARY.md')
    write_json(source/'02_HANDOFF_STATUS.json',dict(**state,handoff_id=hid,snapshot_head=head,one_final_zip=True))
    text_file(source/'00_HANDOFF_README.md','Read 01_STEP_SUMMARY.md, 02_HANDOFF_STATUS.json and summary_data/reports/. Upload this one ZIP with its adjacent SHA-256 and companion JSON. No further execution is authorized.\n')
    text_file(source/'03_DECISIONS_AND_OPEN_QUESTIONS.md','The user accepted the local Phase 6 model with notes and authorized this single campaign. The evidence-ranked next_action_docket distinguishes epochs, data, capacity/optimization and label precision. These are recommendations for external review only. National readiness remains false.\n')
    text_file(source/'04_VALIDATION_RESULTS.md','See summary_data/validation for T0, both repairs and final JUnit, source bindings, independent native witnesses and fresh model/metric reopening. Every trained model was structurally and numerically checked. The source data were verified as read; no repeated Phase 6 pricing or full tree recertification was performed. The initial resource-heartbeat gap and native Windows worker crash are preserved in log_excerpts, REPAIR_001.md and REPAIR_002.md.\n')
    copy(reports/'resource_usage.csv',source/'05_RESOURCE_USAGE.csv')
    text_file(source/'08_COMMAND_LOG.md',f'Explicit campaign command: `{c["execution"]["python_executable"]} -u -m partC_rev1_src.phase7.campaign --config partC_doc_icnn/phase_7_work/s03_active_8h_v1/effective_config.json --phase 07 --step s03_active_8h_v1 --run-id {c["execution"]["run_id"]}`.\n\n'
        'T0 ran the supplied reference tests plus tests/test_phase7_campaign.py and test_phase7_mathematics.py. Guardian --config --pid monitors the exact live coordinator. The repaired campaign resumed with the same command and unchanged original start/deadline. '
        f'T3 command: `{c["execution"]["python_executable"]} -m partC_rev1_src.phase7.diagnostics --config partC_doc_icnn/phase_7_work/s03_active_8h_v1/effective_config.json --plan partC_doc_icnn/phase_7_work/s03_active_8h_v1/DIAGNOSTIC_PLAN.json`. '
        'T4 uses reporting, reopen, audit and finalize modules with the same explicit config. Final pytest runs all three test_phase7 files. Full event history and exact configurations are externally hash-bound.\n')
    text_file(source/'09_NEXT_STEP_RECOMMENDATION.md',state['next_step_recommendation']+'\n')
    text_file(source/'10_UPLOAD_THIS_FILE.txt',str(archive)+'\n')
    text_file(source/'11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md','This is the first Phase 7 handoff. The accepted Phase 6 handoff remains immutable at SHA-256 '+c['source_handoff_sha256']+'. The new campaign adds existing-data training and evaluation only.\n')
    text_file(source/'geospatial_summary/NO_GEOMETRY_CHANGE.md','No geometry, network links, nodes, active support, B/R nodes or path rules changed. Phase 6 topology and comparator caveats are inherited unchanged.\n')
    write_json(source/'log_excerpts/privacy_and_format_scan.json',scan_bundle(c,source))
    records=[dict(relative_path=p.relative_to(source).as_posix(),size_bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(source.rglob('*')) if p.is_file()]
    assert all(r['size_bytes']<=25*2**20 for r in records)
    pd.DataFrame(records).to_csv(source/'06_ARTIFACT_MANIFEST.csv',index=False)
    checks=records+[dict(relative_path='06_ARTIFACT_MANIFEST.csv',sha256=sha(source/'06_ARTIFACT_MANIFEST.csv'))]
    text_file(source/'CHECKSUMS.sha256',''.join(r['sha256']+'  '+r['relative_path']+'\n' for r in checks))
    assert verify_bundle_folder(source)==len(checks)
    tmp=archive.with_suffix('.zip.building')
    with zipfile.ZipFile(tmp,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(source.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(source).as_posix())
    assert tmp.stat().st_size<200_000_000
    extraction=out/'manifests'/('fresh_'+hid)
    with zipfile.ZipFile(tmp) as z:
        assert z.testzip() is None
        for n in z.namelist():assert (extraction/n).resolve().is_relative_to(extraction.resolve())
        z.extractall(extraction)
    assert verify_bundle_folder(extraction)==len(checks)
    env=dict(os.environ);env.update(PYTHONPATH=str(extraction/'code_snapshot'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    xml=out/'manifests/fresh_snapshot_tests.xml'
    test=subprocess.run([c['execution']['python_executable'],'-m','pytest','-q','tests','--junitxml='+str(xml)],cwd=extraction/'code_snapshot',env=env,capture_output=True,text=True,encoding='utf-8')
    text_file(xml.with_suffix('.log'),test.stdout+test.stderr)
    assert test.returncode==0,test.stdout+test.stderr
    assert elapsed(c)<28800,'Hard whole-campaign deadline reached during sealing'
    h=sha(tmp);size=tmp.stat().st_size;tmp.rename(archive)
    companion=dict(status='WAITING_FOR_EXTERNAL_REVIEW',campaign_status=state['campaign_status'],zip_path=str(archive),zip_sha256=h,zip_bytes=size,source_folder=str(source),
        snapshot_head=head,final_head=head,pre_run_head=state['pre_run_head'],authority_sha256=c['authority_sha256'],source_handoff_sha256=c['source_handoff_sha256'],
        config_sha256=sha(cp),verified_checksums=len(checks),fresh_extraction='PASS',CRC='PASS',fresh_snapshot_tests='PASS',fresh_snapshot_tests_xml_sha256=sha(xml),
        coverage=state['coverage'],whole_campaign_elapsed_seconds=elapsed(c),original_start_unix=c['task_started_unix'],original_hard_deadline_unix=c['hard_deadline_unix'],timer_reset=False,
        output_and_handoff_added_bytes=tree_bytes(out)+tree_bytes(source)+size,target_100_MB_met=size<=100_000_000,hard_200_MB_met=True,one_final_zip=True,next_step_authorized=False,next_phase_authorized=False,national_readiness=False)
    write_json(archive.with_suffix('.companion.json'),companion);text_file(archive.with_suffix('.sha256'),h+'  '+archive.name+'\n')
    text_file(parent/'LATEST_HANDOFF.txt',json.dumps(companion,indent=2)+'\n');write_json(out/'FINAL_HANDOFF.json',companion)
    print(json.dumps(companion,indent=2));return companion

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',choices=['summary','seal'],required=True);a=p.parse_args();c=load_config(a.config)
    prepare_summary(c) if a.stage=='summary' else bundle(c)
