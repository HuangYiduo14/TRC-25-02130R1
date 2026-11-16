"""Bounded final reports, code/data snapshots and one independently checked ZIP."""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import numpy as np
import pandas as pd

from .common import read_json,write_json,sha,code_identity
from .collect import gather,records_csv
from .reports import verify_sources,markdown_table,fmt


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),'-c','core.longpaths=true',*args],text=True)


def context(config,run_id):
    c=read_json(config)
    if c['run_id']!=run_id or c['deadline_epoch']!=c['start_epoch']+28800:
        raise ValueError('Finalization requires original explicit phase/run clock')
    root=Path(c['output_root']);repo=Path(c['repository'])
    deadline=c['start_epoch']+(43200 if (root/'R3_ACTIVATION.json').exists() else 28800)
    if time.time()>=deadline:raise TimeoutError('Original whole-Goal deadline reached')
    owner=read_json(root/'NUMERIC_OWNER.json')
    if owner['status']=='ACTIVE':raise RuntimeError('Finish requires the numeric owner to be stopped')
    return c,root,repo


def write_final_reports(c,root,repo):
    data=gather(root,repo);preservation=verify_sources(root,repo,data)
    activated=(root/'R3_ACTIVATION.json').exists()
    phase='N8R3' if activated else 'N8R2'
    rows=[r for r in data['runs'] if r['phase']==phase]
    converged=[r for r in rows if str(r['status']).startswith('CONVERGED')]
    timeout=[r for r in rows if r['status']=='TIME_LIMIT']
    rejected=[r for r in rows if r['status']=='SAFETY_REJECTED' or str(r['status']).startswith('STOP_')]
    unrun=[r for r in rows if 'UNRUN' in str(r['status']) or 'BLOCKED' in str(r['status'])]
    metric=pd.read_csv(root/'tables/path_metrics.csv')
    measured=int(metric.status.eq('MEASURED_BOTH_TRACKS_SEPARATELY').sum()) if 'status' in metric else 0
    expected=6*(len(rows)-len(rejected)) if activated else 0
    complete=not timeout and not unrun and (not activated or measured==expected)
    status=('R2_R3_COMBINED_COMPLETE_WITH_LIMITATIONS' if complete else 'R2_R3_COMBINED_PARTIAL_WITH_REPORT') if activated else 'R2_COMPLETE_AWAITING_REVIEW'
    table=markdown_table(['Scenario','Disposition','Updates','Fresh RG','Elapsed min','Origin'],[
        [r['scenario'],r['status'],r.get('accepted_iterations','—'),fmt(r.get('RG_SPTT'),7),
         fmt(r['elapsed_seconds']/60,2) if r.get('elapsed_seconds') is not None else '—',r.get('reuse_or_new','R2 pilot')] for r in rows])
    elapsed=time.time()-c['start_epoch']
    final=f'''# Final N8 review

**{status}** — WAITING_FOR_EXTERNAL_REVIEW.

R3 auto-started: **{activated}**. At the final scientific boundary, {len(converged)} scenario results reached fresh tolerance, {len(timeout)} were time-limited, {len(rejected)} were safety-rejected, and {len(unrun)} were unrun or blocked. These are separate from campaign completion.

{table}
Original Goal start: {c['start_utc']}. Elapsed at this report: {elapsed/3600:.3f} hours. R2 retains its original eight-hour cap; activated R3 retains the original twelve-hour total deadline. Numerical work is stopped before final packaging. No future step is authorized by this report. R4 remains NOT_AUTHORIZED.

## Runtime and physical reference

The immutable R2 checkpoint contains pilot observations, per-scenario base/high estimates, the full campaign forecast and exact gate. Read `RUNTIME_ESTIMATE.md`, `PHYSICAL_TIMING_AUDIT.md` and `R2_CHECKPOINT/FAST_TRACK_GATE.json`. Full-demand FFTT pilot timings are measured assignments; safety-stop timings are not assignments to tolerance. Unsupported forecasts remain unknown.

Physical x45 has 3,353.9471747006755 recorded update seconds (55.899 minutes). x200 has 14,614.275030500488 update seconds (4.060 hours) and 17,724.460236787796 broader elapsed seconds (4.923 hours). x45's 138.0435530000832-second later refresh is separate. The physical work used four workers; these experiments used one. Neither a complete standalone x45 elapsed boundary nor matched historical hardware is established.

## Scientific coverage and limitations

All final mappings preserve 12,654,353.578361887 represented demand units across 10,190,380 aggregated ODs and 3,225 origins: 99.899865% of original demand. Inherited physical unreachability accounts for the remainder; there is no additional final active-graph unreachability. Original inputs, active support, native model parameters, compiled functions and production modules remain preserved. Source closure status: {preservation['status']} for {preservation['checked_followed_sources']:,} followed files.

The fixed portfolios were not reselected. Every negative/nonfinite actual query remains a failed scenario with its full offending state. Below-FFTT/extrapolation, missing audits, m28, large models with zero extra E2 updates and now-seen audit populations retain their limitations. A small surrogate gap is not physical accuracy. See `LIMITATIONS_AND_OPEN_ITEMS.md` and the inherited-evidence tables.

## Review and delivery

Review the readable reports, full failed/unrun run matrix, signed certificates, timing forecast/gate, source preservation, exact code snapshots, tests and commit receipt. The final archive SHA-256, size and independent reopening checks are recorded in the external delivery receipt. No report can self-approve the next scientific step.
'''
    (root/'01_EXECUTIVE_REVIEW.md').write_text(final,encoding='utf-8')
    (root/'00_READ_FIRST.md').write_text('# Read first\n\n'+status+' — WAITING_FOR_EXTERNAL_REVIEW.\n\n'+
        'Start with `01_EXECUTIVE_REVIEW.md`. The immutable `R2_CHECKPOINT` preserves the complete reports, forecast and gate saved before any R3 activation. Root tables contain final phase-labeled results. Read the standalone physical timing audit and scientific limitations. The final ZIP receipt contains the exact upload path, SHA-256 and verification. R4 and any new campaign remain unauthorized.\n',encoding='utf-8')
    state=dict(status=status,review_status='WAITING_FOR_EXTERNAL_REVIEW',phase_id='N8',run_id=c['run_id'],
        R2_status=read_json(root/'R2_CHECKPOINT/02_HANDOFF_STATUS.json')['status'],R3_auto_started=activated,
        converged_scenarios=len(converged),time_limited_scenarios=len(timeout),safety_rejected_scenarios=len(rejected),
        unrun_or_blocked_scenarios=len(unrun),measured_path_population_tracks=measured,
        next_step_authorized=False,R4_authorized=False,original_R2_start=c['start_epoch'],
        R2_deadline=c['deadline_epoch'],combined_deadline=c['start_epoch']+43200,
        final_science_report_epoch=time.time(),source_preservation=preservation['status'])
    write_json(root/'02_HANDOFF_STATUS.json',state)
    runstate=read_json(root/'RUN_STATE.json');runstate.update(state);runstate['status']='WAITING_FOR_EXTERNAL_REVIEW'
    write_json(root/'RUN_STATE.json',runstate)
    if activated:
        valid=metric[metric.status.eq('MEASURED_BOTH_TRACKS_SEPARATELY')]
        accuracy=markdown_table(['Scenario','Population','Track','WREL','WMEANREL 1s','Paired rows'],[
            [r['scenario'],r['population'],r['track'],fmt(r.get('WREL'),5),fmt(r.get('WMEANREL_1S'),5),r.get('valid_paired_rows','Unknown')]
            for r in valid.to_dict('records')])
        (root/'R3_CAMPAIGN_RESULTS.md').write_text(final+'\n## Path and flow evidence\n\n'+
            f'{measured} cohort/track summaries were measured. Both 128-row cohorts and the historical 200-row cohort remain separately labeled. Full 456-row query tables and exact native flow comparisons are retained for each executed metric case. Rejected/unrun rows remain explicit. See tables/path_metrics.csv and tables/flow_metrics.csv. Ratios below are unitless; multiply by 100 for percentages. These fixed cohorts are now seen and do not establish national unbiased accuracy.\n\n'+accuracy,encoding='utf-8')
        write_json(root/'R3_STATUS.json',state)
    write_json(root/'FINAL_SCIENTIFIC_BOUNDARY.json',dict(epoch=time.time(),elapsed_seconds=time.time()-c['start_epoch'],
        numeric_owner=read_json(root/'NUMERIC_OWNER.json'),current_HEAD=git(repo,'rev-parse','HEAD').strip(),state=state))
    return data,state


def living_documents(c,root,repo,state):
    doc=Path(c['documentation_root']);doc.mkdir(parents=True,exist_ok=True)
    selected=list(root.glob('*.md'))+[root/n for n in ['02_HANDOFF_STATUS.json','FAST_TRACK_GATE.json','FAST_TRACK_GATE_INPUT.json',
        'CAMPAIGN_MANIFEST.json','resolved_config.json','RUN_STATE.json','FILE_PRESERVATION_REPORT.json']]
    for p in selected:shutil.copy2(p,doc/p.name)
    (doc/'ROADMAP.md').write_text('# N8 roadmap at this handoff\n\nR1 is accepted for the launched R2 scope with its scientific limitations. R2 implementation/pilots are recorded in this version. R3 activation and results are governed only by the frozen exact gate and final status in this directory. All subsequent scientific work requires an explicit review decision bound to the final handoff. R4 remains NOT_AUTHORIZED. Historical phase records and source candidates remain unchanged.\n',encoding='utf-8')
    header=f'''# Current N8 handoff — external review required

The user launched authority `ICNN_N8R2_20260915_v1_AUTO`, ZIP SHA-256 `1ac206269ee83cde276b2e098946615633baecb95dc5836a7d1a7e39652a47e3`, accepting R1 for R2 with limitations and conditionally authorizing R3 through the exact twelve-hour total gate. Current terminal status: **{state['status']}**. R3 auto-started: **{state['R3_auto_started']}**.

Read [the versioned N8 review](phase_N8/N8R2/{c['run_id']}/00_READ_FIRST.md), its frozen gate/forecast, source/test/commit evidence and final handoff receipt. Original models, active support and physical references remain immutable. No next scientific step is self-approved; **R4 remains NOT_AUTHORIZED**. The original Goal clock and all failed/time-limited/unrun rows are preserved.

## Historical records below — preserved as provenance

---

'''
    ledger=[]
    for name in ['00_READ_FIRST.md','PROJECT_STATUS.md','12_DECISION_LOG.md']:
        relative='partC_doc_icnn/'+name;p=repo/relative
        tracked=git(repo,'diff','--',relative).strip();staged=git(repo,'diff','--cached','--',relative).strip()
        if tracked or staged:
            patch=root/'proposed_patches'/(name+'.prepend.md');patch.parent.mkdir(exist_ok=True);patch.write_text(header,encoding='utf-8')
            ledger.append(dict(path=str(p),action='PROPOSED_ONLY_USER_CHANGED_FILE',proposal=str(patch)));continue
        old=p.read_bytes();backup=root/'backups/living_docs'/name;backup.parent.mkdir(parents=True,exist_ok=True)
        if backup.exists():raise FileExistsError('Prior living-doc preimage exists; no duplicate prepend')
        backup.write_bytes(old);p.write_bytes(header.encode()+old)
        if not p.read_bytes().endswith(old):raise ValueError('Historical document bytes were not preserved')
        ledger.append(dict(path=str(p),action='NARROW_PREPEND',before_sha256=sha(backup),after_sha256=sha(p),backup=str(backup)))
    records_csv(root/'tables/living_document_changes.csv',ledger)


def source_closure(repo):
    todo=list((repo/'partC_rev1_src/phase_n8').rglob('*.py'));seen=set()
    def resolve(module):
        p=repo.joinpath(*module.split('.'))
        return p.with_suffix('.py') if p.with_suffix('.py').is_file() else p/'__init__.py' if (p/'__init__.py').is_file() else None
    while todo:
        p=todo.pop().resolve()
        if p in seen:continue
        seen.add(p);relative=p.relative_to(repo)
        module='.'.join(relative.with_suffix('').parts)
        package=module.split('.')[:-1]
        for k in range(1,len(relative.parts)-1):
            parent=repo.joinpath(*relative.parts[:k+1])/'__init__.py'
            if parent.exists() and parent not in seen:todo.append(parent)
        top=repo/'partC_rev1_src/__init__.py'
        if top not in seen:todo.append(top)
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[]
            if isinstance(node,ast.Import):names=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    base=package[:len(package)-node.level+1]
                    name='.'.join(base+([node.module] if node.module else []))
                else:name=node.module or ''
                names=[name]+[name+'.'+n.name for n in node.names]
            for name in names:
                if name.startswith('partC_rev1_src'):
                    q=resolve(name)
                    if q and q not in seen:todo.append(q)
    seen.add(repo/'partC_rev1_src/phase5_fidelity/graph.py')
    return sorted(seen)


def plots(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=pd.read_csv(root/'tables/convergence_history.csv')
    fig,ax=plt.subplots(figsize=(9,5.3))
    for (phase,scenario),g in history.groupby(['phase','scenario'],sort=False):
        valid=g.RG_SPTT.notna()&(g.RG_SPTT>0)
        ax.semilogy(g.loc[valid,'elapsed_seconds']/60,g.loc[valid,'RG_SPTT'],marker='.',label=f'{phase} {scenario}')
    ax.axhline(.001,color='black',linestyle='--',linewidth=1,label='Nominal tolerance')
    ax.set(xlabel='Elapsed since this run started (minutes)',ylabel='Fresh signed relative gap / SPTT',title='Full represented-demand assignment trajectories')
    ax.grid(True,alpha=.2);ax.legend(fontsize=7,ncol=2);fig.tight_layout()
    dest=root/'figures';dest.mkdir(exist_ok=True);fig.savefig(dest/'convergence_vs_wall.png',dpi=180);plt.close(fig)
    f=pd.read_csv(root/'tables/runtime_forecast.csv')
    colors=['#257a63' if str(s).startswith('CONVERGED') else '#c58627' if s=='TIME_LIMIT' else '#a44242' for s in f.stop_reason]
    fig,ax=plt.subplots(figsize=(9,5.3));ax.barh(f.scenario,f.observed_end_to_end_seconds/60,color=colors)
    ax.invert_yaxis();ax.set(xlabel='Observed process end-to-end time (minutes)',title='R2 pilot duration by terminal outcome')
    ax.text(0,-.13,'Green: fresh tolerance   Amber: time limit   Red: unsafe cost (not a completed assignment)',transform=ax.transAxes,fontsize=8)
    ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(dest/'pilot_wall_times.png',dpi=180);plt.close(fig)


def copy_file(source,target):
    source=Path(source);target=Path(target)
    if source.stat().st_size>25_000_000:raise ValueError('Compact payload member exceeds 25 MB: '+str(source))
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)


def artifact_schema(path):
    p=Path(path)
    if p.suffix=='.npz':
        arrays=[]
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                with z.open(name) as f:
                    version=np.lib.format.read_magic(f)
                    if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
                    elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
                    else:raise ValueError('Unexpected NPZ array header version')
                arrays.append(dict(key=name[:-4],shape=shape,dtype=str(dtype),fortran_order=fortran))
        return json.dumps(arrays,separators=(',',':'))
    if p.suffix=='.csv':
        with p.open(encoding='utf-8-sig') as f:return f.readline().strip()
    if p.suffix=='.parquet':
        import pyarrow.parquet as pq
        return str(pq.read_schema(p))
    return p.suffix.lstrip('.').upper() or 'TEXT'


def snapshot(c,root,repo):
    state=read_json(root/'02_HANDOFF_STATUS.json');combined=state['R3_auto_started']
    hid=('N8R2R3' if combined else 'N8R2')+'_20260915T140337Z_final_001'
    dest=repo/'partC_doc_icnn/handoffs/phase_N8/step_N8R2'/hid
    dest.mkdir(parents=True,exist_ok=False)
    zip_path=repo/'artifacts_icnn/handoff_bundles/phase_N8'/('phaseN8_stepN8R2_'+hid+'_SYNC.zip')
    if zip_path.exists():raise FileExistsError('Final ZIP already exists')
    write_json(root/'PACKAGING_PLAN.json',dict(handoff_id=hid,source_folder=str(dest),zip_path=str(zip_path),status='FROZEN_PACKAGING_PATHS'))
    for directory in ['governing_docs','configs','code_snapshot','code_diff','summary_data','figures','geospatial_summary','log_excerpts','external_artifact_index']:
        (dest/directory).mkdir()
    for p in root.glob('*.md'):copy_file(p,dest/p.name)
    for p in root.glob('*.json'):copy_file(p,dest/p.name)
    for directory in ['tables','validation','receipts','inherited_evidence','R2_CHECKPOINT','configs']:
        for p in (root/directory).rglob('*'):
            if not p.is_file():continue
            # Include concise test controls/receipts; toy checkpoint payloads stay indexed locally.
            if directory=='validation' and any(x.startswith('T0') for x in p.relative_to(root/directory).parts[:-1]):continue
            target=dest/p.relative_to(root) if directory!='receipts' else dest/'summary_data'/p.relative_to(root)
            copy_file(p,target)
    for p in (root/'figures').glob('*.png'):copy_file(p,dest/'figures'/p.name)
    authority=Path(c['authority_root'])
    for p in authority.glob('[0-9][0-9]_*.md'):copy_file(p,dest/'governing_docs/authority'/p.name)
    for p in (authority/'config').glob('*.json'):copy_file(p,dest/'configs/authority'/p.name)
    for name in ['03_ALGORITHM_AND_THEORY_FREEZE.md','04_EXPERIMENT_DESIGN_FREEZE.md']:
        copy_file(authority/'references/r1'/name,dest/'governing_docs/R1'/name)
    for p in (authority/'references/r1/proposals').glob('*.json'):copy_file(p,dest/'configs/frozen_registries'/p.name)
    copy_file(authority/'references/r1/derived/INDEX.json',dest/'external_artifact_index/R1_ACTIVE_FFTT_INDEX.json')
    for name in ['20_HANDOFF_SYNC_PROTOCOL.md','21_REVIEW_APPROVAL_GATES.md','23_SYNC_BUNDLE_CONTENT_AND_SIZE_POLICY.md','24_CODE_AND_DATA_SNAPSHOT_POLICY.md']:
        copy_file(repo/'partC_doc_icnn'/name,dest/'governing_docs'/name)
    for p in source_closure(repo):copy_file(p,dest/'code_snapshot'/p.relative_to(repo))
    write_json(dest/'code_snapshot/CODE_SNAPSHOT_MANIFEST.json',[
        dict(path=str(p.relative_to(dest/'code_snapshot')).replace('\\','/'),sha256=sha(p),bytes=p.stat().st_size)
        for p in (dest/'code_snapshot').rglob('*.py')])
    print(json.dumps(dict(snapshot=str(dest),zip_path=str(zip_path)),indent=2))
    return dest


def payload_details(c,root,repo,dest):
    external=[]
    for p in root.rglob('*'):
        if not p.is_file() or any(x in ['cache','backups','R2_CHECKPOINT'] for x in p.relative_to(root).parts):continue
        if p.suffix not in ['.json','.csv','.npz','.md','.parquet','.png','.log','.jsonl']:continue
        external.append(dict(native_path=str(p),relative_path=str(p.relative_to(root)),size_bytes=p.stat().st_size,
            sha256=sha(p),schema=artifact_schema(p),retention='Original local artifact retained; bulk arrays/checkpoint bodies indexed'))
    records_csv(dest/'external_artifact_index/OWNED_OUTPUT_INDEX.csv',external)
    # Exact followed source index includes model/graph files without duplicating national data.
    copy_file(root/'tables/source_bindings.csv',dest/'external_artifact_index/NATIVE_SOURCE_BINDINGS.csv')
    for p in (root/'runs').glob('*/SAFETY_STOP.json'):
        target=dest/'summary_data/safety_queries'/p.parent.name
        copy_file(p,target/p.name)
        if (p.parent/'offending_query.npz').exists():copy_file(p.parent/'offending_query.npz',target/'offending_query.npz')
        d=read_json(p)
        if d.get('portable_path'):copy_file(d['portable_path'],target/Path(d['portable_path']).name)
    for p in (root/'conditional_R3').glob('*/PATH_METRICS.csv') if (root/'conditional_R3').exists() else []:
        for q in p.parent.iterdir():
            if q.is_file():copy_file(q,dest/'summary_data/conditional_R3'/p.parent.name/q.name)
    for p in (root/'runs').glob('*/HISTORY.json'):
        for name in ['HISTORY.json','RESULT.json','PRE_RUN_FREEZE.json','STAGES.json','INFERENCE_TIMINGS.json','FIXED_X45_INFERENCE.json']:
            q=p.parent/name
            if q.exists():copy_file(q,dest/'summary_data/runs'/p.parent.name/name)
    for p in (root/'conditional_R3/assignments').glob('*/RESULT.json') if (root/'conditional_R3/assignments').exists() else []:
        for name in ['RESULT.json','HISTORY.json','PRE_RUN_FREEZE.json','STAGES.json','FIXED_X45_INFERENCE.json']:
            q=p.parent/name
            if q.exists():copy_file(q,dest/'summary_data/official_assignments'/p.parent.name/name)
    for p in (root/'logs').glob('*.log'):
        text=p.read_text(errors='replace');lines=text.splitlines()
        # Preserve all concise test/error output; suppress duplicate huge JSON payload prints.
        selected=[line for line in lines if len(line)<2500]
        (dest/'log_excerpts'/p.name).write_text('\n'.join(selected)+'\n',encoding='utf-8')
    (dest/'geospatial_summary/NOT_APPLICABLE.md').write_text('No new topology or geospatial artifact was generated. Accepted native graph identities, role counts and coverage are in tables/demand_coverage.csv.\n')
    copy_file(root/'01_EXECUTIVE_REVIEW.md',dest/'01_STEP_SUMMARY.md')
    copy_file(root/'00_READ_FIRST.md',dest/'00_HANDOFF_README.md')
    copy_file(root/'LIMITATIONS_AND_OPEN_ITEMS.md',dest/'03_DECISIONS_AND_OPEN_QUESTIONS.md')
    copy_file(root/'R2_IMPLEMENTATION_AND_TESTS.md',dest/'04_VALIDATION_RESULTS.md')
    copy_file(root/'tables/resource_usage.csv',dest/'05_RESOURCE_USAGE.csv')
    (dest/'09_NEXT_STEP_RECOMMENDATION.md').write_text('REVIEW the exact handoff and its scientific limitations. Further scientific execution requires an explicit user decision bound to this archive. No automatic slow campaign or R4 work is authorized.\n')
    (dest/'10_UPLOAD_THIS_FILE.txt').write_text(read_json(root/'PACKAGING_PLAN.json')['zip_path']+'\nThe external delivery receipt holds archive SHA-256 and reopening evidence.\n')
    (dest/'11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md').write_text('Prior reviewed R1 ZIP SHA-256: 9de63d07ac3e6b26060fe3f99c183802d557e4bef2c0afffd8c9bb84de2d6432. This handoff adds the new native coupled solver, safe multigraph loading, explicit full-demand mapping/coverage, actual pilots, timing forecast, physical timing reconciliation and exact conditional R3 decision/results. R1 source selections and historical evidence remain immutable.\n')
    commands=['# Actual supervised command log','']
    for p in sorted((root/'receipts').glob('*_JOB.json')):
        j=read_json(p);commands+=['## '+j['job_id'],'',f"Task {j['task']}; original Goal start {j['original_R2_start']}; run start {j['start_epoch']}; deadline {j['deadline_epoch']}.",'','```json',json.dumps(j['command'],indent=2),'```','']
    (dest/'08_COMMAND_LOG.md').write_text('\n'.join(commands),encoding='utf-8')
    records_csv(dest/'07_CODE_CHANGE_INDEX.csv',[dict(path=str(p.relative_to(repo)),sha256=sha(p),size_bytes=p.stat().st_size,
        role='NEW_OWNED_CODE' if 'phase_n8' in p.parts else 'UNCHANGED_LOCAL_IMPORT_OR_RULE_DEPENDENCY') for p in source_closure(repo)])


def scoped_commit(c,root,repo):
    if git(repo,'diff','--cached','--name-only').strip():raise RuntimeError('Pre-existing staged changes must be preserved')
    paths=[]
    for directory in [repo/'partC_rev1_src/phase_n8',Path(c['documentation_root']),
            repo/'partC_doc_icnn/review_gates/phase_N8/step_N8R2/N8R2_authority_20260915']:
        paths.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    changes=pd.read_csv(root/'tables/living_document_changes.csv')
    for row in changes.to_dict('records'):
        if row['action']=='NARROW_PREPEND' and sha(row['path'])==row['after_sha256']:paths.append(Path(row['path']))
    relative=sorted({str(p.relative_to(repo)).replace('\\','/') for p in paths})
    before=git(repo,'rev-parse','HEAD').strip()
    git(repo,'add','--',*relative)
    staged=sorted(git(repo,'diff','--cached','--name-only').splitlines())
    if not set(staged).issubset(relative):raise RuntimeError('Unexpected staged path; no commit made')
    outcome=subprocess.run(['git','-C',str(repo),'-c','core.longpaths=true','commit','-m',
        'Implement N8 native coupled assignment and record bounded campaign evidence'],capture_output=True,text=True)
    after=git(repo,'rev-parse','HEAD').strip()
    receipt=dict(status='COMMITTED' if outcome.returncode==0 else 'COMMIT_FAILED_EVIDENCE_PRESERVED',
        before_HEAD=before,after_HEAD=after,branch=git(repo,'branch','--show-current').strip(),
        staged_explicit_paths=staged,command_returncode=outcome.returncode,stdout=outcome.stdout,stderr=outcome.stderr,
        original_HEAD=c['initial_head'],push_executed=False,all_paths_explicit_and_owned=True,
        post_commit_porcelain=git(repo,'status','--porcelain=v1','--untracked-files=normal'),epoch=time.time())
    write_json(root/'COMMIT_RECEIPT.json',receipt)
    print(json.dumps(dict(status=receipt['status'],HEAD=after,paths=len(staged)),indent=2))


def seal(c,root,repo):
    plan=read_json(root/'PACKAGING_PLAN.json');dest=Path(plan['source_folder']);zip_path=Path(plan['zip_path'])
    validation=root/'validation/FINAL_SNAPSHOT_TESTS.json'
    if not validation.exists() or read_json(validation)['returncode']!=0:
        raise ValueError('A successful independent snapshot test receipt is required before sealing')
    payload_details(c,root,repo,dest)
    copy_file(root/'COMMIT_RECEIPT.json',dest/'COMMIT_RECEIPT.json')
    copy_file(validation,dest/'validation/FINAL_SNAPSHOT_TESTS.json')
    copy_file(root/'logs/FINAL_SNAPSHOT_TESTS.log',dest/'log_excerpts/FINAL_SNAPSHOT_TESTS.log')
    (dest/'04_VALIDATION_RESULTS.md').write_text((dest/'04_VALIDATION_RESULTS.md').read_text()+
        '\n## Final independent source snapshot check\n\nThe bundled code snapshot was imported from its own directory and the scoped tests passed. See validation/FINAL_SNAPSHOT_TESTS.json and the full snapshot test log. Archive closure is checked separately by the external delivery receipt after this payload is frozen.\n')
    (dest/'code_diff/owned_commit.patch').write_text(git(repo,'diff','--binary',c['initial_head'],'HEAD','--',
        'partC_rev1_src/phase_n8','partC_doc_icnn/phase_N8/N8R2',
        'partC_doc_icnn/review_gates/phase_N8/step_N8R2','partC_doc_icnn/00_READ_FIRST.md',
        'partC_doc_icnn/PROJECT_STATUS.md','partC_doc_icnn/12_DECISION_LOG.md'),encoding='utf-8')
    (dest/'code_diff/final_git_status.txt').write_text(git(repo,'status','--porcelain=v1','--untracked-files=normal'),encoding='utf-8')
    write_ledger=[]
    for p in [*root.rglob('*'),*(repo/'partC_rev1_src/phase_n8').rglob('*'),*Path(c['documentation_root']).rglob('*')]:
        if p.is_file() and 'cache' not in p.parts and '__pycache__' not in p.parts:
            write_ledger.append(dict(path=str(p),action='OWNED_CREATE_OR_VERSIONED_OUTPUT',size_bytes=p.stat().st_size,
                sha256=sha(p),mtime_ns=p.stat().st_mtime_ns))
    records_csv(root/'tables/write_ledger.csv',write_ledger);copy_file(root/'tables/write_ledger.csv',dest/'tables/write_ledger.csv')
    records_csv(dest/'06_ARTIFACT_MANIFEST.csv',[dict(path=str(p.relative_to(dest)).replace('\\','/'),
        size_bytes=p.stat().st_size,sha256=sha(p),role='HANDOFF_PAYLOAD') for p in sorted(dest.rglob('*')) if p.is_file()])
    members=[p for p in sorted(dest.rglob('*')) if p.is_file()]
    checks=''.join(sha(p)+'  '+str(p.relative_to(dest)).replace('\\','/')+'\n' for p in members)
    (dest/'CHECKSUMS.sha256').write_text(checks,encoding='utf-8')
    members.append(dest/'CHECKSUMS.sha256')
    if any(p.stat().st_size>25_000_000 for p in members):raise ValueError('Member exceeds compact policy')
    zip_path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in members:z.write(p,str(p.relative_to(dest)).replace('\\','/'))
    if zip_path.stat().st_size>200_000_000:raise ValueError('ZIP exceeds hard bundle limit')
    expected={str(p.relative_to(dest)).replace('\\','/'):sha(p) for p in members}
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failed')
        if len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(expected):raise ValueError('ZIP membership mismatch')
        for name,digest in expected.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('ZIP member hash mismatch: '+name)
        actual_lines=z.read('CHECKSUMS.sha256').decode().splitlines()
        if len(actual_lines)!=len(expected)-1:raise ValueError('Checksum closure count differs')
        for line in actual_lines:
            digest,name=line.split('  ',1)
            if expected[name]!=digest:raise ValueError('Root checksum does not match payload')
    receipt=dict(status='SEALED_WAITING_FOR_EXTERNAL_REVIEW',handoff_id=plan['handoff_id'],zip_path=str(zip_path),
        zip_sha256=sha(zip_path),zip_bytes=zip_path.stat().st_size,source_folder=str(dest),payload_members=len(members),
        full_CRC_pass=True,exact_membership_pass=True,every_payload_hash_pass=True,root_checksum_closure_pass=True,
        independently_reopened=True,source_snapshot_tests_pass=True,
        commit_HEAD=git(repo,'rev-parse','HEAD').strip(),branch=git(repo,'branch','--show-current').strip(),
        final_git_porcelain=git(repo,'status','--porcelain=v1','--untracked-files=normal'),
        elapsed_total_seconds=time.time()-c['start_epoch'],original_R2_start=c['start_epoch'],sealed_epoch=time.time(),
        next_step_authorized=False,R4_authorized=False)
    write_json(root/'DELIVERY_RECEIPT.json',receipt)
    zip_path.with_suffix('.sha256').write_text(receipt['zip_sha256']+'  '+zip_path.name+'\n')
    write_json(zip_path.with_suffix('.receipt.json'),receipt)
    (dest.parent/'LATEST_HANDOFF.txt').write_text(str(dest)+'\n'+str(zip_path)+'\n'+receipt['zip_sha256']+'\nWAITING_FOR_EXTERNAL_REVIEW\n')
    print(json.dumps(receipt,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--action',choices=['reports','commit','snapshot','seal'],required=True);args=p.parse_args()
    c,root,repo=context(args.config,args.run_id)
    if args.action=='reports':
        data,state=write_final_reports(c,root,repo);plots(root);living_documents(c,root,repo,state)
        print(json.dumps(state,indent=2))
    elif args.action=='commit':scoped_commit(c,root,repo)
    elif args.action=='snapshot':snapshot(c,root,repo)
    else:seal(c,root,repo)


if __name__=='__main__':raise SystemExit(main())
