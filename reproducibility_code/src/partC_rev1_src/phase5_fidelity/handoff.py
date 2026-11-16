"""Compact complete-scalar synchronization bundle and independent ZIP verification."""
import argparse
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
from partC_rev1_src.phase5.campaign_io import sha,write_json,frame_write,file_record,git,verify_receipt
from partC_rev1_src.phase5.handoff import _privacy_scan


def build(config_path):
    config=json.loads(Path(config_path).read_text());repo=Path(config['repository_root']);out=Path(config['output_root'])
    fresh=json.loads((out/'validation/fresh_process_reopen.json').read_text())
    if fresh['status']!='PASS' or fresh['scalar_paths']!=44688:raise ValueError('Fresh full reopen is required before seal')
    for name in ['F0','F1','F2','F3','F4','F5']:verify_receipt(out,out/'receipts'/f'{name}.json')
    final_head=git(repo,'rev-parse','HEAD')
    if git(repo,'log','-1','--format=%s')!='phase5: record fidelity audit and phase6 readiness decision':raise ValueError('Required result commit is missing')
    status=json.loads((out/'reports/F5_status.json').read_text())
    # The current F5 status is immutable evidence. The final F6 record adds the seal state.
    status.update(status='P5_FIDELITY_COMPLETE_WAITING_EXTERNAL_REVIEW',review_status='WAITING_FOR_EXTERNAL_REVIEW',
        pre_run_commit=json.loads((out/'validation/pre_run_checkpoint.json').read_text())['commit'],result_commit=final_head,
        authority_zip_sha256=config['authority_zip_sha256'],reviewed_source_zip_sha256=config['source_zip_sha256'],
        whole_task_elapsed_seconds_before_seal=time.time()-config['task_started_unix'])
    write_json(out/'reports/F6_final_status.json',status)
    write_json(out/'reports/final_status.json',status)
    manifest=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and 'manifests' not in p.relative_to(out).parts:
            r=file_record(out,p);r['absolute_path']=str(p);r['format']=p.suffix;r['mtime_ns']=p.stat().st_mtime_ns
            if p.suffix=='.parquet':
                import pyarrow.parquet as pq
                pf=pq.ParquetFile(p);r['row_count']=pf.metadata.num_rows;r['schema']=str(pf.schema_arrow)
            elif p.suffix=='.npz':
                with np.load(p,allow_pickle=False) as a:r['schema']=json.dumps({k:dict(dtype=str(a[k].dtype),shape=list(a[k].shape)) for k in a.files})
            manifest.append(r)
    mf=out/'manifests/full_output_manifest.csv';frame_write(mf,pd.DataFrame(manifest))
    handoff_id='fidelity_0906_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_final_001'
    parent=repo/'partC_doc_icnn/handoffs/phase_05/step_fidelity_0906';source=parent/handoff_id;staging=parent/(handoff_id+'.building')
    archive_root=repo/'artifacts_icnn/handoff_bundles/phase_05';archive=archive_root/('phase05_stepfidelity_0906_'+handoff_id+'_SYNC.zip')
    if source.exists() or staging.exists() or archive.exists():raise FileExistsError('Handoff target must be write-new')
    staging.mkdir(parents=True);archive_root.mkdir(parents=True,exist_ok=True)
    def copy(p,d):
        d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
    for section in ['governing_docs','configs','code_snapshot','code_diff','summary_data','figures','geospatial_summary','log_excerpts','external_artifact_index']:(staging/section).mkdir()
    for p in sorted(Path(config['authority_root']).rglob('*')):
        if p.is_file() and p.suffix not in ['.zip','.pyc']:copy(p,staging/'governing_docs/authority'/p.relative_to(config['authority_root']))
    for name in ['00_READ_FIRST.md','PROJECT_STATUS.md','12_DECISION_LOG.md','20_HANDOFF_SYNC_PROTOCOL.md','21_REVIEW_APPROVAL_GATES.md','23_SYNC_BUNDLE_CONTENT_AND_SIZE_POLICY.md','24_CODE_AND_DATA_SNAPSHOT_POLICY.md']:
        copy(repo/'partC_doc_icnn'/name,staging/'governing_docs'/name)
    for p in (repo/config['work_records_relative']).glob('*'):
        if p.is_file():copy(p,staging/'configs'/p.name)
    copy(Path(config_path),staging/'configs/execution.json')
    snapshot=sorted((repo/'partC_rev1_src/phase5_fidelity').glob('*.py'))+[repo/'tests/test_phase5_fidelity.py']
    snapshot += [repo/p for p in ['partC_rev1_src/partC_rev1_shortest_paths.py','partC_rev1_src/phase4/inputs.py','partC_rev1_src/phase4/models.py','partC_rev1_src/phase4/tracing.py']]
    snapshot += sorted((repo/'partC_rev1_src/phase5').glob('*.py'))
    for p in snapshot:copy(p,staging/'code_snapshot'/p.relative_to(repo))
    patch=subprocess.check_output(['git','-C',str(repo),'-c','core.longpaths=true','diff',config['required_ancestor'],final_head,'--',
        'partC_rev1_src/phase5_fidelity','tests/test_phase5_fidelity.py',config['work_records_relative'],
        'partC_doc_icnn/review_gates/phase_05/fidelity_0906','partC_doc_icnn/00_READ_FIRST.md','partC_doc_icnn/PROJECT_STATUS.md','partC_doc_icnn/12_DECISION_LOG.md'])
    (staging/'code_diff/fidelity.patch').write_bytes(patch)
    (staging/'code_diff/git_diff_stat.txt').write_text(git(repo,'diff','--stat',config['required_ancestor'],final_head)+'\n')
    (staging/'code_diff/git_status.txt').write_text(git(repo,'status','--short')+'\n')
    for p in sorted((out/'engineering_epochs').glob('epoch_*/*.py')):
        copy(p,staging/'code_snapshot/engineering_epochs'/p.relative_to(out/'engineering_epochs'))
    copy(mf,staging/'external_artifact_index/full_output_manifest.csv')
    write_json(staging/'external_artifact_index/upstream_bindings.json',dict(authority_sha256=config['authority_zip_sha256'],
        source_zip_sha256=config['source_zip_sha256'],source_output_root=str(config['source_output_root']),
        full_source_invariance= json.loads((out/'validation/source_invariance.json').read_text()),
        graph_and_path_arrays='Full local files bound by full_output_manifest.csv; no national graph duplicate in ZIP'))
    included=[];split=[]
    # All scalar result tables are included completely, split by bytes where necessary.
    for p in sorted(out.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(out)
        if rel.parts[0] in ['graphs','batches','projections','manifests']:
            if rel.as_posix()!='graphs/variant_manifest.json':continue
        if p.suffix not in ['.csv','.json','.md','.log','.txt','.xml']:continue
        if rel.parts[0]=='results' and len(rel.parts)>1 and rel.parts[1]=='configurations':continue
        target=staging/'summary_data'/rel
        if p.stat().st_size<=24_000_000:
            copy(p,target);included.append(rel.as_posix())
        elif p.suffix=='.csv':
            # Byte-bounded complete row chunks, never a prefix-only review sample.
            count=0;members=[]
            for i,chunk in enumerate(pd.read_csv(p,chunksize=8000,float_precision='round_trip')):
                dest=target.with_name(target.stem+f'_part_{i:03d}.csv');frame_write(dest,chunk)
                if dest.stat().st_size>25*2**20:raise ValueError('Full scalar CSV chunk exceeds member limit')
                count+=len(chunk);members.append(str(dest.relative_to(staging)))
            split.append(dict(source=str(rel),full_rows=count,included_complete_parts=members,source_sha256=sha(p)))
        else:raise ValueError('Essential summary exceeds member size; explicit complete segmentation required: '+str(rel))
    write_json(staging/'summary_data/complete_scalar_table_index.json',dict(included_full_tables=included,split_complete_tables=split,no_adverse_rows_dropped=True))
    # Compact, deterministic path samples are reconstructed from real native/lift arrays.
    full=pd.read_csv(out/'results/shortest_path_scalar_metrics.csv')
    picks=pd.concat([full.sort_values('absolute_relative_distance_error',ascending=False).head(16),
        full.loc[full.lift_status.eq('UNLIFTABLE')].sort_values(['graph_key','stable_od_key']).groupby('graph_variant',sort=True).head(2),
        full.sort_values(['graph_key','stable_od_key']).groupby(['graph_variant','state'],sort=True).head(1)]).drop_duplicates(['graph_key','stable_od_key'])
    raw=[]
    for r in picks.itertuples(index=False):
        with np.load(out/r.native_path_artifact,allow_pickle=False) as a:
            i=int(r.path_row);native=a['native_edges'][a['native_ptr'][i]:a['native_ptr'][i+1]];physical=a['physical_arcs'][a['lift_ptr'][i]:a['lift_ptr'][i+1]]
        edge_ids=pd.read_parquet(out/'graphs'/r.graph_key/'edges.parquet',columns=['edge_id']).edge_id.to_numpy()
        raw.append(dict(graph_key=r.graph_key,stable_od_key=r.stable_od_key,state=r.state,source_set=r.source_set,lift_status=r.lift_status,
            native_edge_ids=edge_ids[native].tolist(),physical_arc_indices=physical.tolist(),native_hash=r.native_path_edge_indices_sha256,
            physical_hash=r.physical_lift_indices_sha256,failure_from=r.failure_from_physical_nodes_json,failure_to=r.failure_to_physical_nodes_json))
    write_json(staging/'summary_data/raw_path_samples.json',dict(method='16 worst distance errors plus deterministic unliftable and per-variant/state cases',samples=raw))
    summary=(out/'reports/readiness_and_next_phase.md').read_text()
    headers={'00_HANDOFF_README.md':'# Phase 5F final synchronization evidence\n\nRead 01_STEP_SUMMARY.md, 02_HANDOFF_STATUS.json and summary_data/reports/shortest_path_error_summary.csv. Complete scalar populations are included, segmented where required; large graphs and paths are hash-bound externally.\n',
        '01_STEP_SUMMARY.md':summary,'03_DECISIONS_AND_OPEN_QUESTIONS.md':'Only the user may accept production state or authorize Phase 6. Review the B/R state architecture, adverse distances, unliftable paths and all candidate results. Fixed charges are not clipped; capacities and CURRENT benchmarks are unchanged.\n',
        '04_VALIDATION_RESULTS.md':f'Complete source/new regressions and supplied reference tests pass. Fresh process reopens {fresh["graphs"]} active graphs, {fresh["batches"]} batches, {fresh["scalar_paths"]} paths; all source hashes remain unchanged. See complete logs and machine receipts.\n',
        '08_COMMAND_LOG.md':'Entrypoints: python -m partC_rev1_src.phase5_fidelity.run --config <configs/execution.json> --stage F0 through F5, followed by phase5_fidelity.verify and phase5_fidelity.handoff with --config. Exact pre-run and result commits are in status and companion. No assignment, loading, refitting, package install or network command was used.\n',
        '09_NEXT_STEP_RECOMMENDATION.md':'External review only. Scientific readiness: '+status['scientific_readiness']+'. Production, Phase 6 labels, ICNN training and national assignment remain unauthorized.\n',
        '10_UPLOAD_THIS_FILE.txt':str(archive)+'\n',
        '11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md':'The completed source compiler is preserved. This task adds complete metric/count summaries, frozen OD sampling, true frozen-state shortest paths, exact physical lifts/regret, low-discovery B restoration diagnostics and interface recommendations. It does not rerun C0-C7.\n'}
    for name,text in headers.items():(staging/name).write_text(text,encoding='utf-8')
    write_json(staging/'02_HANDOFF_STATUS.json',status)
    copy(out/'validation/resource_history.csv',staging/'05_RESOURCE_USAGE.csv')
    frame_write(staging/'07_CODE_CHANGE_INDEX.csv',pd.DataFrame([dict(relative_path=str(p.relative_to(repo)),sha256=sha(p)) for p in snapshot]))
    (staging/'figures/NO_FIGURES.md').write_text('Complete quantitative evidence is delivered as CSV/JSON; no scientific figure or manuscript result was changed.\n')
    (staging/'geospatial_summary/NO_GPKG.md').write_text('Accepted geospatial/topology products remain immutable. Proposed interface changes are listed by exact stable B/region/terminal/fine-node IDs in interface_impact.csv.\n')
    scan=_privacy_scan(staging);remaining=[];resolved=[]
    for finding in scan['findings']:
        p=staging/finding['path']
        if finding['pattern']=='unreviewed_binary_type' and p.suffix=='.xml':
            import xml.etree.ElementTree as ET
            ET.fromstring(p.read_text());resolved.append(dict(**finding,disposition='XML_PARSE_PASS'))
        else:remaining.append(finding)
    scan.update(findings=remaining,format_reopen_evidence=resolved,status='PASS' if not remaining else 'FAIL')
    write_json(staging/'log_excerpts/privacy_scan.json',scan)
    if remaining:raise ValueError('Privacy/text scan failed: '+json.dumps(remaining))
    payload=[file_record(staging,p) for p in sorted(staging.rglob('*')) if p.is_file()]
    if any(r['size_bytes']>25*2**20 for r in payload):raise ValueError('Included file exceeds 25MiB')
    frame_write(staging/'06_ARTIFACT_MANIFEST.csv',pd.DataFrame(payload))
    checks=payload+[file_record(staging,staging/'06_ARTIFACT_MANIFEST.csv')]
    (staging/'CHECKSUMS.sha256').write_text(''.join(r['sha256']+'  '+r['relative_path']+'\n' for r in checks))
    temp=archive.with_suffix('.zip.building')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(staging.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(staging).as_posix())
    if temp.stat().st_size>200*2**20:raise ValueError('ZIP exceeds hard synchronization limit')
    extract=out.parent/'seal_verification'/handoff_id
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
        for n in z.namelist():
            if not (extract/n).resolve().is_relative_to(extract.resolve()):raise ValueError('Unsafe ZIP member')
        z.extractall(extract)
    for r in checks:
        p=extract/r['relative_path']
        if p.stat().st_size!=r['size_bytes'] or sha(p)!=r['sha256']:raise ValueError('Fresh ZIP extraction checksum mismatch')
    if {p.relative_to(extract).as_posix() for p in extract.rglob('*') if p.is_file()}!={r['relative_path'] for r in checks}|{'CHECKSUMS.sha256'}:raise ValueError('Fresh ZIP member closure failed')
    elapsed=time.time()-config['task_started_unix']
    added=sum(p.stat().st_size for root in [out.parent,staging,Path(config['authority_root']),repo/config['work_records_relative'],repo.parent/'p5f0906_input'] for p in root.rglob('*') if p.is_file())+temp.stat().st_size
    if elapsed>=config['resources']['base_seconds'] or added>=config['resources']['added_disk_bytes']:raise ValueError('Final resource guard')
    digest=sha(temp);temp.rename(archive);staging.rename(source)
    Path(str(archive)+'.sha256').write_text(digest+'  '+archive.name+'\n')
    resource=pd.read_csv(out/'validation/resource_history.csv')
    companion=dict(status,zip_path=str(archive),zip_sha256=digest,zip_size_bytes=archive.stat().st_size,
        companion_path=str(archive)+'.companion.json',source_folder=str(source),handoff_id=handoff_id,
        fresh_zip_checksums=len(checks),fresh_zip_manifest_closure='PASS',privacy_scan='PASS',
        whole_task_elapsed_seconds=elapsed,added_bytes_including_ZIP_and_fresh_extraction=added,
        peak_sampled_process_tree_RSS_bytes=int(resource.process_tree_rss_bytes.max()),
        next_step_authorized=False,review_status='WAITING_FOR_EXTERNAL_REVIEW')
    write_json(Path(str(archive)+'.companion.json'),companion)
    write_json(parent/'LATEST_HANDOFF.txt',dict(zip_path=str(archive),sha256=digest,companion=companion['companion_path'],source_folder=str(source),status=companion['status']))
    return companion


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    print(json.dumps(build(args.config),indent=2))


if __name__=='__main__':main()
