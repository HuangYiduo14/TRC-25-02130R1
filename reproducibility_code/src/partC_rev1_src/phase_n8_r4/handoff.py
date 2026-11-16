"""Source verification and compact report-first packaging after numerical work."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,write_json,sha
from .reporting import REPORT_NAMES,publication_qa
from .code_dependencies import dependencies


def source_verification(ctx,args):
    root=ctx.root;records={};receipt_files=[]
    for path in sorted((root/'receipts').glob('*_SOURCE_BINDINGS.json')):
        receipt_files.append(dict(path=str(path),sha256=sha(path)))
        for row in read_json(path):
            key=(row['native_path'],row['sha256'])
            if key not in records:records[key]=dict(row,used_by_receipts=[])
            records[key]['used_by_receipts'].append(path.name)
    rows=[];failures=[]
    for i,row in enumerate(records.values()):
        if i%100==0:ctx.guard()
        p=Path(row['native_path']);internal=p==root or root in p.parents
        scoped_code=ctx.repo/'partC_rev1_src/phase_n8_r4' in p.parents
        fixed_R4=internal and (p.relative_to(root).parts[0] in {
            'runs','registries','physical_population','sampling','geography','regional_geography',
            'regional_metrics','learning','stability','boundary_diagnostics','path_metrics'} or
            p.relative_to(root).as_posix() in {'regional_safety_transform.csv','SAFE_SCENARIO_FREEZE.json','physical_comparison_population.csv'})
        scientific_immutable=(not internal and not scoped_code) or fixed_R4
        exists=p.is_file();actual=sha(p) if exists and scientific_immutable else None
        status=('PASS' if actual==row['sha256'] else 'HASH_MISMATCH' if exists else 'MISSING') if scientific_immutable else 'VERSIONED_R4_DERIVED_BINDING'
        if scientific_immutable and status!='PASS':failures.append(dict(path=str(p),expected=row['sha256'],actual=actual,status=status))
        rows.append(dict(native_path=str(p),bound_sha256=row['sha256'],size_bytes=row['size_bytes'],role=row['role'],
            source_scope='IMMUTABLE_R4_SCIENTIFIC_OUTPUT' if fixed_R4 else 'IMMUTABLE_INHERITED_SOURCE' if scientific_immutable else 'VERSIONED_R4_DERIVED_SOURCE',
            final_verification=status,final_sha256=actual,used_by_receipts='|'.join(row['used_by_receipts'])))
    pd.DataFrame(rows).to_csv(root/'source_index.csv',index=False)
    result=dict(status='PASS' if not failures else 'FAIL',immutable_source_records=sum(r['source_scope']=='IMMUTABLE_INHERITED_SOURCE' for r in rows),
        immutable_R4_scientific_output_records=sum(r['source_scope']=='IMMUTABLE_R4_SCIENTIFIC_OUTPUT' for r in rows),
        all_bound_records=len(rows),failures=failures,receipt_manifests=receipt_files,
        scope='Every consumed immutable inherited byte binding and completed R4 scientific-output binding; intentionally versioned report/release revisions retain historical hashes separately and current release QA binds final bytes',
        no_claim_to_hash_unconsumed_national_database=True)
    write_json(root/'validation/FINAL_SOURCE_VERIFICATION.json',result)
    return dict(status=result['status'],source_records=len(rows),failures=len(failures))


def _code_dependencies(repo):
    roots=list((repo/'partC_rev1_src/phase_n8_r4').rglob('*.py'))
    roots.append(repo/'tests/test_phase4_preflight.py')
    return dependencies(repo,roots)


def _npz_schema(path):
    fields=[]
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith('.npy'):continue
            with z.open(name) as f:
                version=np.lib.format.read_magic(f)
                shape,fortran,dtype=np.lib.format._read_array_header(f,version)
                fields.append(dict(array=name[:-4],shape=list(shape),dtype=str(dtype),fortran=bool(fortran)))
    return json.dumps(fields,separators=(',',':'))


def artifact_index(ctx,args):
    root=ctx.root;rows=[]
    excluded_top={'toolchain','cache','handoff_staging'}
    for i,p in enumerate(sorted(root.rglob('*'))):
        if not p.is_file():continue
        rel=p.relative_to(root)
        if rel.parts[0] in excluded_top or any(x in rel.parts for x in ('.mplcache','__pycache__')):continue
        if rel.as_posix() in ('artifact_manifest.csv','artifact_index.csv') or p.suffix=='.zip':continue
        if i%100==0:ctx.guard()
        schema='';suffix=p.suffix.lower()
        if suffix=='.npz':
            try:schema=_npz_schema(p)
            except Exception as e:schema='SCHEMA_UNAVAILABLE: '+type(e).__name__+': '+str(e)
        elif suffix=='.parquet':
            import pyarrow.parquet as pq
            meta=pq.ParquetFile(p);schema=json.dumps(dict(rows=meta.metadata.num_rows,schema=str(meta.schema_arrow)))
        elif suffix=='.csv':
            with p.open(encoding='utf-8-sig',newline='') as f:header=next(csv.reader(f),[])
            schema=json.dumps(dict(columns=header))
        rows.append(dict(relative_path=rel.as_posix(),native_path=str(p),size_bytes=p.stat().st_size,
            mtime_ns=p.stat().st_mtime_ns,sha256=sha(p),format=suffix[1:],schema=schema,
            native_storage='Full local artifact; compact inclusion is separately declared by bundle_manifest.csv'))
    frame=pd.DataFrame(rows);frame.to_csv(root/'artifact_manifest.csv',index=False)
    frame.to_csv(root/'artifact_index.csv',index=False)
    code=[]
    for p in _code_dependencies(ctx.repo):
        code.append(dict(relative_path=p.relative_to(ctx.repo).as_posix(),native_path=str(p),size_bytes=p.stat().st_size,
            sha256=sha(p),role='NEW_R4_CODE' if ctx.repo/'partC_rev1_src/phase_n8_r4' in p.parents else 'UNCHANGED_DIRECT_OR_TRANSITIVE_LOCAL_DEPENDENCY'))
    pd.DataFrame(code).to_csv(root/'code_snapshot_manifest.csv',index=False)
    return dict(status='EXECUTED',artifact_files=len(rows),artifact_bytes=int(frame.size_bytes.sum()),code_files=len(code))


def verify_delivery_commit(ctx,commit):
    """Bind the final owned code/docs, actual Git tree, and report copies."""
    root=ctx.root;repo=ctx.repo;docs=Path(ctx.c['documentation_root'])
    sync=read_json(root/'receipts/DOCUMENTATION_SYNC.json')
    if sync.get('review_status')!='WAITING_FOR_EXTERNAL_REVIEW' or sync.get('source_handoff_status_sha256')!=sha(root/'02_HANDOFF_STATUS.json'):
        raise ValueError('Final documentation sync is absent or stale')
    report_records={r['name']:r for r in sync.get('report_copies',[])}
    if set(report_records)!=set(REPORT_NAMES):raise ValueError('Incomplete final report copies')
    for name,row in report_records.items():
        if row['source_sha256']!=sha(root/name) or row['copied_sha256']!=sha(docs/name) or sha(root/name)!=sha(docs/name):
            raise ValueError('Versioned report differs from final source: '+name)
    owned={p.relative_to(repo).as_posix() for p in (repo/'partC_rev1_src/phase_n8_r4').rglob('*.py')}
    owned|={(docs/name).relative_to(repo).as_posix() for name in REPORT_NAMES+['AUTHORITY_POINTER.json']}
    owned|={r['relative_path'] for r in sync['files'] if r['status']=='UPDATED'}
    listed={r['relative_path']:r for r in commit.get('files',[])}
    if set(listed)!=owned:raise ValueError('Commit receipt omits or adds final owned files')
    def git(*args):
        return subprocess.run(['git','-C',str(repo),'-c','core.longpaths=true',*args],
            text=True,capture_output=True,check=True).stdout.strip()
    head=git('rev-parse','HEAD')
    if not commit.get('commit') or head!=commit['commit']:raise ValueError('Commit receipt does not identify current HEAD')
    tree=git('rev-parse',head+'^{tree}')
    if tree!=commit.get('tree'):raise ValueError('Commit tree identity missing or different')
    git('merge-base','--is-ancestor',ctx.c['initial_head'],head)
    changed=set(git('diff','--name-only',ctx.c['initial_head'],head).splitlines())
    if changed!=owned:raise ValueError('Committed change scope differs from final owned file inventory')
    for relative,row in listed.items():
        if sha(repo/relative)!=row['sha256']:raise ValueError('Owned working file changed after commit receipt: '+relative)
        blob=git('rev-parse',head+':'+relative)
        if blob!=row.get('git_blob_oid') or git('hash-object','--path='+relative,str(repo/relative))!=blob:
            raise ValueError('Owned current file does not match committed blob: '+relative)
    diff_path=root/'receipts/FINAL_SCOPED_DIFF.patch'
    if not diff_path.is_file() or sha(diff_path)!=commit.get('scoped_diff_sha256'):
        raise ValueError('Scoped review diff absent or changed')
    actual_diff=subprocess.run(['git','-C',str(repo),'-c','core.longpaths=true','diff','--binary','--full-index',
        ctx.c['initial_head'],head],capture_output=True,check=True).stdout
    if hashlib.sha256(actual_diff).hexdigest()!=commit['scoped_diff_sha256']:
        raise ValueError('Scoped review diff does not match the actual initial-to-final Git tree change')
    return dict(status='PASS',commit=head,tree=tree,owned_files=len(owned),
        documentation_sync_sha256=sha(root/'receipts/DOCUMENTATION_SYNC.json'),scoped_diff_sha256=sha(diff_path))


def package(ctx,args):
    root=ctx.root;state=read_json(root/'02_HANDOFF_STATUS.json')
    if state.get('review_status')!='WAITING_FOR_EXTERNAL_REVIEW' or not state.get('publication_assets_complete'):
        raise ValueError('Final reports and verified publication assets required before packaging')
    qa=publication_qa(root)
    if not all(qa[k] for k in ('pdf_compiled','visual_QA','isolated_replot_QA')):
        raise ValueError('Publication QA does not bind the current release bytes')
    if (root/'ACTIVE_SCIENCE_CHILD.json').exists():
        active=read_json(root/'ACTIVE_SCIENCE_CHILD.json')
        if active.get('job_id')!=args.job_id:raise ValueError('Another owned child remains active')
    if read_json(root/'validation/FINAL_SOURCE_VERIFICATION.json')['status']!='PASS':raise ValueError('Source integrity not passed')
    commit=read_json(root/'receipts/COMMIT_RECEIPT.json')
    write_json(root/'receipts/FINAL_COMMIT_VERIFICATION.json',verify_delivery_commit(ctx,commit))
    staging=root/'handoff_staging'/args.job_id;staging.mkdir(parents=True,exist_ok=False)
    included={};exclusions=[]
    def copy(path,relative=None,reason='REVIEW_EVIDENCE'):
        p=Path(path);rel=relative or p.relative_to(root).as_posix()
        if Path(rel).is_absolute() or '..' in Path(rel).parts:raise ValueError('Unsafe bundle relative path')
        if rel in included:return
        if p.stat().st_size>=25_000_000:raise ValueError('Oversize essential review member requires sharding: '+str(p))
        dst=staging/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        included[rel]=dict(relative_path=rel,size_bytes=dst.stat().st_size,sha256=sha(dst),source_path=str(p),inclusion_reason=reason)
    def csv_shards(path,folder):
        p=Path(path);fields=None;writer=None;stream=None;current_bytes=0;number=0;count=0;shards=[]
        target=staging/folder;target.mkdir(parents=True,exist_ok=True)
        with p.open(encoding='utf-8-sig',newline='') as source:
            reader=csv.reader(source);fields=next(reader)
            for row in reader:
                if stream is None or current_bytes>=18_000_000:
                    if stream is not None:stream.close();shards.append(dict(path=output,rows=count))
                    output=target/f'{p.stem}_{number:03d}.csv';number+=1;count=0;current_bytes=0
                    stream=output.open('w',encoding='utf-8',newline='');writer=csv.writer(stream);writer.writerow(fields)
                writer.writerow(row);count+=1
                # UTF-8 bytes upper bound, including quoting and separators.
                current_bytes+=sum(len(v.encode('utf-8'))+v.count('"')+3 for v in row)+2
                if count%50000==0:ctx.guard()
        if stream is not None:stream.close();shards.append(dict(path=output,rows=count))
        for item in shards:
            pth=item['path'];rel=pth.relative_to(staging).as_posix()
            if pth.stat().st_size>=25_000_000:raise ValueError('CSV shard exceeds member cap')
            included[rel]=dict(relative_path=rel,size_bytes=pth.stat().st_size,sha256=sha(pth),source_path=str(p),inclusion_reason='LOSSLESS_NATIVE_CSV_SHARD')
        return [dict(relative_path=x['path'].relative_to(staging).as_posix(),rows=x['rows'],
            source_path=str(p),source_sha256=sha(p),schema='|'.join(fields),shard_sha256=sha(x['path'])) for x in shards]
    for name in REPORT_NAMES:copy(root/name,reason='REPORT_FIRST_REQUIRED')
    required=['experiment_inventory.csv','run_matrix.csv','figure_inventory.csv','table_inventory.csv','claim_to_result.csv',
        'source_index.csv','reviewer_crosswalk.csv','reviewer_response_actual.csv','regional_safety_transform.csv','coverage.csv',
        'time_budget_ledger.csv','data_access_and_validation_status.csv','artifact_manifest.csv','artifact_index.csv','code_snapshot_manifest.csv',
        'resolved_config.json','ORIGINAL_CLOCK.json','INTAKE_RECEIPT.json','SAFE_SCENARIO_FREEZE.json']
    for name in required:
        p=root/name
        if p.stat().st_size>=25_000_000:
            index=csv_shards(p,'large_inventory_shards/'+p.stem)
            dest=staging/name;pd.DataFrame(index).to_csv(dest,index=False)
            included[name]=dict(relative_path=name,size_bytes=dest.stat().st_size,sha256=sha(dest),source_path=str(p),
                inclusion_reason='EXPLICIT_LOSSLESS_SHARD_INDEX; full rows supplied in referenced bundle members')
        else:copy(p,reason='REQUIRED_INVENTORY_OR_CONFIGURATION')
    population=root/'physical_comparison_population.csv'
    if population.stat().st_size>=25_000_000:
        index=csv_shards(population,'physical_comparison_population_shards')
        dst=staging/'physical_comparison_population.csv';pd.DataFrame(index).to_csv(dst,index=False)
        included[dst.name]=dict(relative_path=dst.name,size_bytes=dst.stat().st_size,sha256=sha(dst),source_path=str(population),
            inclusion_reason='EXPLICIT_LOSSLESS_SHARD_INDEX; native full table unchanged and every row included in shards')
    else:copy(population,reason='REQUIRED_PHYSICAL_POPULATION')
    plot_inputs={}
    for folder in sorted((root/'numerical_release').glob('fig_*')):
        if not folder.is_dir():continue
        records=pd.read_csv(folder/'read_csv_hashes.csv').to_dict('records')
        if not records:raise ValueError('Missing observed standalone plotting inputs')
        for row in records:
            p=folder/row['file']
            if p.parent!=folder or sha(p)!=row['sha256']:raise ValueError('Plot input audit is stale')
            plot_inputs[p.relative_to(root).as_posix()]=row['sha256']
    for p in sorted((root/'numerical_release').rglob('*')):
        if p.is_file() and not any(x in p.parts for x in ('.mplcache','__pycache__')):
            rel=p.relative_to(root).as_posix();folder=p.parent.name
            # S4 aggregate intervals/bars also require their per-draw/per-OD observations.
            underlying_observations=folder=='fig_S4' and p.name.startswith('all_paired_path_observations_')
            auxiliary_figure=folder.startswith('fig_') and rel not in plot_inputs and not underlying_observations
            auxiliary_table=folder.startswith('table_') and p.name not in ('table.csv','table_metadata.csv')
            if p.suffix=='.csv' and p.stat().st_size>=2_000_000 and (auxiliary_figure or auxiliary_table):
                exclusions.append(dict(path=str(p),reason='UNPLOTTED_AUXILIARY_OBSERVATIONS_NATIVE_HASH_INDEXED; every observed plotting input and table display CSV remains included',
                    sha256=sha(p),size_bytes=p.stat().st_size))
                continue
            if p.suffix not in ('.aux','.bbl','.blg','.out','.synctex','.xdv'):copy(p,reason='COMPLETE_STANDALONE_NUMERICAL_RELEASE')
    for rel,expected in plot_inputs.items():
        if rel not in included or included[rel]['sha256']!=expected:raise ValueError('A plotted CSV was omitted or changed')
    proof=staging/'PLOTTING_INPUT_COMPLETENESS.json'
    write_json(proof,dict(status='PASS',observed_plotting_input_files=len(plot_inputs),
        all_observed_plotting_inputs_included=True,bindings=plot_inputs,
        auxiliary_exclusions_do_not_change_plot_scripts_inputs_or_figures=True))
    included[proof.name]=dict(relative_path=proof.name,size_bytes=proof.stat().st_size,sha256=sha(proof),
        source_path='GENERATED_CURRENT_INPUT_COMPLETENESS_PROOF',inclusion_reason='NO_PLOTTED_CSV_EXTERNAL_ONLY')
    for folder in ('audits','receipts','logs','documentation_backups','registries','sampling','tables','validation','accounting'):
        for p in sorted((root/folder).rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(root)
            if any(x in rel.parts for x in ('local_pytest','physical_pytest')):continue
            if 'figure_isolation' in rel.parts:
                if any(x.endswith('_ALONE') for x in rel.parts) and p.name!='runtime_file_access.json':continue
            if 'pdf_render' in rel.parts and p.suffix=='.png':continue
            if p.suffix in ('.npz','.npy','.parquet','.pkl','.sqlite','.exe','.dll','.pyc','.zip'):continue
            if p.stat().st_size>=25_000_000:
                exclusions.append(dict(path=str(p),reason='LARGE_NATIVE_EVIDENCE_HASH_INDEXED; plotted copies remain in numerical_release'));continue
            copy(p)
    for folder in ('physical_population','geography','regional_geography','stability','boundary_diagnostics','regional_metrics','learning','path_metrics'):
        for p in sorted((root/folder).rglob('*')):
            if not p.is_file() or p.suffix not in ('.json','.csv','.md'):continue
            if p.suffix=='.csv' and p.stat().st_size>=2_000_000:continue
            if p.stat().st_size>=25_000_000:continue
            copy(p,reason='COMPACT_SCIENTIFIC_STATUS_COVERAGE_AND_COMPUTE; larger native tables indexed or included with plots')
    for p in sorted((root/'jobs').rglob('*')):
        if p.is_file() and p.suffix in ('.json','.log','.py') and p.name!='LIVE.json':
            if p.stat().st_size<25_000_000:copy(p,reason='EXACT_TASK_LOG_OR_CLOCK')
    for p in sorted((root/'runs').glob('*/*')):
        if p.is_file() and p.suffix in ('.json','.jsonl','.csv') and p.name!='node_balance.csv':
            if p.stat().st_size<25_000_000:copy(p,reason='ASSIGNMENT_SUMMARY_FRESH_CERTIFICATES_AND_HISTORIES')
    for row in pd.read_csv(root/'code_snapshot_manifest.csv').to_dict('records'):
        p=Path(row['native_path'])
        if sha(p)!=row['sha256']:raise ValueError('Code changed after snapshot')
        copy(p,'code/'+row['relative_path'],row['role'])
    for row in commit.get('files',[]):
        p=ctx.repo/row['relative_path']
        if sha(p)!=row['sha256']:raise ValueError('Owned file changed after commit receipt: '+str(p))
        if row['relative_path'].startswith('partC_doc_icnn/'):
            copy(p,'repository_changes/'+row['relative_path'],'EXACT_SCOPED_DOCUMENTATION_CHANGE')
    for p in sorted(ctx.authority.glob('[0-9][0-9]_*.md')):copy(p,'authority/'+p.name,'GOVERNING_NUMBERED_CONTRACT')
    for p in (ctx.authority/'CHECKSUMS.sha256',ctx.authority/'reviewer_crosswalk.csv'):copy(p,'authority/'+p.name,'AUTHORITY_BINDING')
    for name in ('PROMPT_BODY.md','RUN_GUIDE.md'):copy(ctx.authority/name,'authority/'+name,'GOVERNING_EXECUTION_INSTRUCTION')
    for p in sorted((ctx.authority/'scripts').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            copy(p,'authority/scripts/'+p.relative_to(ctx.authority/'scripts').as_posix(),'EXACT_EXECUTED_AUTHORITY_TEST_OR_HELPER')
    readme=staging/'BUNDLE_STORAGE_NOTES.md'
    readme.write_text('# Compact storage conventions\n\nThe root physical_comparison_population.csv is explicitly a lossless shard index because the native table exceeds the member limit. Its referenced CSV shards contain every original row in original order and preserve field values. The full native source remains unchanged at its indexed path/hash. Other oversize required inventories use the same declared convention.\n\nEvery plotted CSV remains beside its plot.py and is included in full; none is external-only. Native arrays, experimental weights/checkpoints, package caches, compiler binaries and full national databases remain local and hash-indexed. See artifact_index.csv for native format/schema/hash and bundle_manifest.csv for exact inclusion.\n\nNo external publication or next scientific round is authorized.\n',encoding='utf-8')
    included[readme.name]=dict(relative_path=readme.name,size_bytes=readme.stat().st_size,sha256=sha(readme),source_path='GENERATED_STORAGE_NOTE',inclusion_reason='EXPLICIT_STORAGE_CONVENTION')
    with readme.open('a',encoding='utf-8') as f:
        f.write('\nLarge auxiliary observation CSVs that were not read by any figure, and large table-support exports other than table.csv/table_metadata.csv, remain unchanged in the full native numerical_release and are explicitly indexed in explicit_large_evidence_exclusions.csv. Their native FILE_HASHES/PROVENANCE entries are historical full-release inventories, not assertions of compact inclusion. PLOTTING_INPUT_COMPLETENESS.json proves every observed drawing input is present with its exact hash. All actual table display CSVs, TeX and scripts remain included.\n\nThis ZIP uses lossless LZMA compression (ZIP method 14). Extract using Python 3: python -m zipfile -e ARCHIVE.zip NEW_OUTPUT_DIRECTORY, or a compatible extractor such as 7-Zip. Windows Explorer compatibility is not asserted.\n')
    included[readme.name].update(size_bytes=readme.stat().st_size,sha256=sha(readme))
    pd.DataFrame(exclusions,columns=['path','reason','sha256','size_bytes']).to_csv(staging/'explicit_large_evidence_exclusions.csv',index=False)
    p=staging/'explicit_large_evidence_exclusions.csv';included[p.name]=dict(relative_path=p.name,size_bytes=p.stat().st_size,sha256=sha(p),source_path='GENERATED_EXCLUSION_LEDGER',inclusion_reason='NO_SILENT_OMISSION')
    manifest=pd.DataFrame(included.values());manifest.to_csv(staging/'bundle_manifest.csv',index=False)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive=root.parent/f'N8R4_NUMERICAL_HANDOFF_{stamp}.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_LZMA,allowZip64=True) as z:
        for name in REPORT_NAMES:z.write(staging/name,name)
        for p in sorted(staging.rglob('*')):
            if not p.is_file() or p.relative_to(staging).as_posix() in REPORT_NAMES:continue
            ctx.guard();z.write(p,p.relative_to(staging).as_posix())
    if archive.stat().st_size>200_000_000:raise ValueError('Main handoff exceeds 200 MB; preserve candidate and revise compact packaging without dropping plot CSVs')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC verification failed')
        actual=set(z.namelist());expected=set(included)|{'bundle_manifest.csv'}
        if actual!=expected:raise ValueError('Archive member inventory differs')
        for row in included.values():
            with z.open(row['relative_path']) as stream:
                h=hashlib.file_digest(stream,'sha256').hexdigest()
            if h!=row['sha256']:raise ValueError('Archive payload hash differs')
    ctx.guard();completed=time.time()
    result=dict(status='PASS',archive=str(archive),sha256=sha(archive),size_bytes=archive.stat().st_size,
        members=len(included)+1,member_cap_bytes=25_000_000,target_bytes=100_000_000,hard_cap_bytes=200_000_000,
        source_manifest=str(staging/'bundle_manifest.csv'),full_plotted_CSVs_included=True,report_first=True,
        compression='ZIP_LZMA_METHOD_14',extraction='Python 3: python -m zipfile -e ARCHIVE.zip NEW_OUTPUT_DIRECTORY',
        completed_epoch=completed,total_R4_elapsed_seconds=completed-ctx.c['start_epoch'],
        hard_deadline_epoch=ctx.c['deadline_epoch'],hard_deadline_met=completed<ctx.c['deadline_epoch'],
        review_status='WAITING_FOR_EXTERNAL_REVIEW',next_round_authorized=False)
    write_json(root/'HANDOFF_RECEIPT.json',result)
    archive.with_suffix('.zip.sha256').write_text(result['sha256']+'  '+archive.name+'\n',encoding='ascii')
    return result
