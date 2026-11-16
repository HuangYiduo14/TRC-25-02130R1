"""Report-first closeout, scoped source commit and verified compact handoff."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,write_json,sha
from .context import code_identity


def git(ctx,*args):
    command=['git','-C',str(ctx.repo),'-c','core.longpaths=true',*args]
    p=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
    if p.returncode:raise RuntimeError('Git operation failed: '+p.stderr[-4000:])
    return p.stdout.strip()


def reports(ctx):
    root=ctx.root;publication=root/'publication_bundle'
    from .final_evidence import publication_gate,inventories
    publication_proof=publication_gate(ctx);inventory=inventories(ctx)
    visual=read_json(root/'validation/VISUAL_QA.json')
    if visual['status']!='PASS' or visual['main_pdf_sha256']!=sha(publication/'main_numerical.pdf'):
        raise ValueError('Final actual visual inspection has not passed for this PDF')
    automated=read_json(publication/'AUTOMATED_VERIFICATION.json');metrics=read_json(root/'metrics/RESULT.json')
    if automated['arithmetic']['status']!='PASS' or metrics['status']!='PASS':raise ValueError('Final numerical publication gates incomplete')
    current=read_json(root/'registries/M1_CURRENT.json');portfolio=read_json(root/'registries'/(current['name']+'.json'))
    fits=read_json(root/'learning/COMBINED_FIT_INVENTORY.json');learning=read_json(root/'receipts/C02_LEARNING_COMPLETION_RESULT.json')
    regional=read_json(root/'regional_metrics/RESULT.json')
    opened=pd.read_csv(root/'metrics/open_summary_metrics.csv',float_precision='round_trip')
    closed=pd.read_csv(root/'metrics/closed_summary_metrics.csv',float_precision='round_trip')
    response=pd.read_csv(root/'metrics/closure_redistribution.csv',float_precision='round_trip')
    tests=read_json(root/'validation/TESTS.json')
    rows=opened.loc[opened.suite_id.eq('s03')&opened.population.eq('BACKBONE')]
    lines=['# Executive review: numerical publication completion','',
        'The authorized amendment has executed its bounded experiments and produced the numerical-only manuscript and seventeen self-contained CSV publication folders. The terminal handoff awaits external review; no next round is authorized.','',
        '## Main assignment results','',
        '| Method | Full-backbone WMAPE (%) | Backbone cost error (%) | Fresh gap | Complete job wall (s) | State |',
        '|---|---:|---:|---:|---:|---|']
    for r in rows.itertuples():
        seconds='Unavailable' if pd.isna(r.online_seconds) else format(r.online_seconds,'.3f')
        lines.append('|'+r.method_label+'|'+format(r.wmape_percent,'.6f')+'|'+format(r.backbone_cost_error_percent,'.6f')+'|'+format(r.RG_SPTT,'.8g')+'|'+seconds+'|'+r.status+'|')
    lines+=['','All KPIs compare native ACNHN coordinates with forward-mapped physical CURRENT reference flow. The reference self-row is actually recomputed and zero; the independent physical CFW control remains distinct. Full backbone reference-zero coordinates remain included. All200 key IDs were frozen before predictions and retained for the closed comparison.','',
        '## Safety and closure','',
        'Final effective portfolio: '+json.dumps(portfolio['composition'],sort_keys=True)+'. Whole-region fallback used '+str(current['discovery_epochs'])+' discovery epochs; rescue_used='+str(current['rescue_used'])+'. Discarded queries, feasible restart states and immutable epoch identities remain in the evidence. The final open/closed M1 registry matches.','',
        'All three abstract represented I-95 closure assignments actually executed. The scenario removes 1,105 matched represented physical source arcs / 869 abstract coordinates, explicitly omits 608 suppressed backbone-interior arcs, and retains 55 healthy physical alternatives inside disabled macros. Original demand, additional unreachable rows/demand, and zero masked flows are audited. Closed-reference accuracy and open-to-closed redistribution are separate tables.','']
    witness=read_json(root/'registries/all_recorded_witnesses/RESULT.json')
    if witness['status']!='PASS' or witness['final_registry_identity']!=current['identity']:raise ValueError('Final saved-query replay gate incomplete')
    lines+=['The first preflight omitted the three recorded warm-start failure cases. A clearly labeled supplemental audit authenticated all twelve recorded cold/warm/trial witnesses (eight unique vectors), then replayed the original and final frozen portfolios. The final portfolio passed every recorded vector; the audit is not backdated and does not imply global nonnegativity of the abandoned neural models.','']
    control=closed.loc[closed.method_id.eq('REFERENCE_CLOSED')&closed.population.eq('BACKBONE')].iloc[0]
    paired=read_json(root/'metrics/CLOSED_OD_COMPARABILITY.json')
    lines+=['Physical/abstract closed served-OD membership equal: '+str(paired['same_served_OD_membership'])+'. Both row and demand totals and their exact overlap are in metrics/closed_OD_membership_comparison.csv. If membership differs, closed-reference WMAPE is full-backbone discrepancy including different served demand, not common-served-OD accuracy. Redistribution includes lost demand as well as rerouting.','']
    lines+=['Matched physical closure control: '+control.status+', fresh gap '+format(control.RG_SPTT,'.8g')+', complete supervisor wall '+format(control.online_seconds,'.3f')+' seconds. Its approximate-reference status is preserved.','',
        '## Regional fidelity and learning','',
        str(sum(r['status']=='COMPLETE' for r in fits))+' of 80 neural fit slots are complete; 40 previous fits were reused unchanged. Only the missing frozen eight regions were trained. The original nested fractions, seed0, 4,000-update budget and fixed VALID selection are preserved. Learning completion status: '+learning['status']+'. Regional evaluation status: '+regional['status']+'. Cost WMAPE is direct native prediction/truth arithmetic, not renamed MARE.','',
        'The operating/stress audits omit native m=28 and m=39 rows where certified sources are absent; the reused low-flow and high projected test cohorts retain their separate coverage denominators. All fresh-label attempts, severe/missing cases, now-seen cohorts and actual selected-versus-executed counters are retained. These limits restrict scientific conclusions even when the document is complete.','',
        '## Timing and pilot audit','',
        'Historical FFTT pilot fresh-assignment estimates ranged from 422.959–561.401 seconds at the base estimate to 634.439–842.101 seconds at the high estimate across the five suites. The earlier analytic VALID mixture estimated 1,490.897–2,885.384 seconds and is not the new full-power-4 method. Unsafe original neural proposals had no valid tolerance forecast. Exact pilot evidence and estimator scope are in the publication supporting table and private source bindings.','',
        'Physical x45: 3,353.947175 seconds cumulative iteration wall, with a separate later138.043553-second zero-update refresh. Physical x200: 14,614.275031 seconds iteration wall and17,724.460237 seconds broader campaign. Complete standalone historical cold clocks are unknown. Current complete supervisor timings use one worker; new abstract initializations were prepared in separately timed preflight, while physical controls initialize inside their jobs. No matched cold-start or equal-accuracy speedup is claimed.','',
        'Offline parent elapsed and summed worker wall are kept separate. Cold-start amortization is marked not comparable because initialization boundaries and attributable offline timing differ. No repeated-query crossing is claimed, and missing construction/training cost is not zero-imputed.','',
        'Execution qualifications are recorded in [EXECUTION_NOTES.md](EXECUTION_NOTES.md). The initial run matrix declared the job caps, but a separate zero-hour forecast was not recorded. The first standalone cap-based forecast was late; the three-hour checkpoint uses its actual timestamp. Earlier job snapshots retained task namespaces plus the clean source commit; later regression and final handoff snapshots include the full static dependency closure. No timing or preflight evidence is backdated.','',
        '## Deliverables and verification','',
        '- publication_bundle/main_numerical.pdf and its numerical/supplement/macros TeX sources.',
        '- publication_bundle/numerical_data/INDEX.html and INDEX.csv; figure3–figure8, figureS1–figureS4, table1–table6 and tableS1.',
        '- Every folder has local CSVs, standalone drawing/build code, PDF/PNG, README, dictionary and provenance; tables include table.tex.',
        '- Every folder regenerated alone at two distinct working directories with cleared PYTHONPATH, runtime file-access auditing, equal input rows/hashes and identical PNGs.',
        '- Independent authority CSV arithmetic, computed reference zero, signed-gap admission and fixed closed keys passed. Mandatory regression status: '+tests['status']+'.',
        '- Final numerical PDF compiled twice; every final numerical page/panel was visually checked. Detailed record: validation/VISUAL_QA.json and visual_QA.md.',
        '- The optional v5 numerical-replacement review copy labels its unrevised nonnumerical sections; it is not a fully revised manuscript.','',
        '## Limits and exact next action','',
        'Projected approximate reference; fixed active support; scenario reachability loss; represented-only closure; missing/severe local labels; now-seen audit data; incomplete attributable offline timing; synthetic annual-cost interpretation; no matched independent traffic validation. Errors and time-limited outcomes are retained without favorable-value targets.','',
        '**Next action: externally review this handoff. next_round_authorized=false. Stop here.**','']
    (root/'01_EXECUTIVE_REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')
    scientific_complete=learning['status']=='EXECUTED' and regional['status']=='EXECUTED' and inventory['primary_matrix_complete']
    status=dict(status='WAITING_FOR_EXTERNAL_REVIEW',authority_id=ctx.c['authority_id'],phase_id='N8',step_id=ctx.c['step_id'],run_id=ctx.c['run_id'],
        completion_state='COMPLETE_NUMERICAL_SECTION_AWAITING_REVIEW' if inventory['primary_matrix_complete'] else 'PARTIAL_WITH_PAPER_ASSETS',
        branch=git(ctx,'branch','--show-current'),reviewed_source_base_commit=git(ctx,'rev-parse','HEAD'),
        immutable_source_seal=read_json(root/'IMMUTABLE_SOURCE_SEAL.json'),
        commit_status='PENDING_SCOPED_COMMIT_SEE_EXTERNAL_RECEIPT',commit_receipt_absolute=str(root/'COMMIT_RECEIPT.json'),
        canonical_output_root=str(root),committed_document_policy='This precommit mirror cannot contain its own commit hash; exact final commit is in the external receipt and canonical status.',
        authority_archive_sha256=ctx.c['authority_sha256'],source_model_registry='N8_PROPOSED_NONCONSTANT_VALID_V1',
        source_model_registry_identity=witness['original_registry_identity'],final_model_registry=current['name'],
        final_model_registry_identity=current['identity'],primary_matrix_complete=inventory['primary_matrix_complete'],
        experiment_complete=inventory['experiment_complete'],publication_verification=publication_proof,
        scientific_cutoff_epoch=ctx.c['compute_cutoff_epoch'],final_scientific_caps_seconds=ctx.c['per_run_caps_seconds'],
        document_complete=True,scientific_execution_complete=scientific_complete,scientific_limitations_retained=True,
        I95_assignments_actually_executed=True,matched_physical_closure_executed=True,matched_physical_closure_status=control.status,
        closed_physical_abstract_served_OD_membership_equal=paired['same_served_OD_membership'],
        fallback_final_composition=portfolio['composition'],fallback_rescue_used=current['rescue_used'],
        M1_open_closed_registry_equal=True,learning_status=learning['status'],regional_status=regional['status'],
        completed_neural_fits=sum(r['status']=='COMPLETE' for r in fits),reused_fits=40,publication_folders=17,
        main_pdf_sha256=sha(publication/'main_numerical.pdf'),original_start_epoch=ctx.c['start_epoch'],
        completed_evidence_epoch=time.time(),elapsed_seconds=time.time()-ctx.c['start_epoch'],deadline_epoch=ctx.c['deadline_epoch'],
        next_round_authorized=False,next_action='EXTERNAL_REVIEW_ONLY',commit_receipt='COMMIT_RECEIPT.json')
    write_json(root/'02_HANDOFF_STATUS.json',status)
    (root/'00_READ_FIRST.md').write_text('# Numerical publication handoff\n\nRead [the executive review](01_EXECUTIVE_REVIEW.md), then [machine-readable status](02_HANDOFF_STATUS.json). Open [the numerical PDF](publication_bundle/main_numerical.pdf) and [all standalone data, code and figures](publication_bundle/numerical_data/INDEX.html). Scientific limitations, partial labels, portfolio degeneration and timing boundaries remain explicit.\n\nThis amendment is finished and awaits external review. No next round is authorized. The original authority ZIP, branch and earlier source/result/model assets were preserved. The complete publication is embedded in this handoff instead of a redundant nested archive.\n',encoding='utf-8')
    doc=Path(ctx.c['documentation_root'])
    for name in ('00_READ_FIRST.md','01_EXECUTIVE_REVIEW.md','02_HANDOFF_STATUS.json'):shutil.copy2(root/name,doc/name)
    mirror=doc/'00_READ_FIRST.md'
    mirror.write_text('# Repository documentation mirror\n\nThe canonical [numerical handoff]('+root.as_posix()+'/00_READ_FIRST.md) includes the PDF, data index, source seal, and final postcommit receipt. This committed mirror records the reviewed precommit state; it cannot contain its own commit hash.\n',encoding='utf-8')
    marker='<!-- N8R4_V5_PUBLICATION_FINAL '+ctx.c['run_id']+' -->'
    for name in ('00_READ_FIRST.md','PROJECT_STATUS.md','12_DECISION_LOG.md'):
        p=ctx.repo/'partC_doc_icnn'/name;old=p.read_text(encoding='utf-8-sig')
        if not old.startswith(marker):
            prefix=marker+'\n# Publication amendment: waiting for external review\n\nThe current numerical publication handoff is `phase_N8/N8R4_V5_PUBLICATION/'+ctx.c['run_id']+'`. Its final status supersedes the earlier in-progress amendment entry below. The seventeen-folder CSV release and numerical PDF are complete; scientific limitations remain explicit. No next round is authorized. Read the run-local executive review and status.\n\n'
            p.write_text(prefix+old,encoding='utf-8')
    return status


def commit_owned(ctx):
    if git(ctx,'branch','--show-current')!='phase1/icnn-rev':raise ValueError('Requested branch changed')
    if git(ctx,'diff','--cached','--name-only'):raise ValueError('Existing staged changes must be preserved; scoped commit cannot proceed')
    scope=['partC_rev1_src/phase_n8_publication','partC_doc_icnn/phase_N8/N8R4_V5_PUBLICATION/'+ctx.c['run_id'],
        'partC_doc_icnn/00_READ_FIRST.md','partC_doc_icnn/PROJECT_STATUS.md','partC_doc_icnn/12_DECISION_LOG.md']
    git(ctx,'diff','--check','--',*scope)
    before=git(ctx,'rev-parse','HEAD');git(ctx,'add','--',*scope)
    staged=git(ctx,'diff','--cached','--name-only').splitlines()
    if not staged or any(not any(p==s or p.startswith(s+'/') for s in scope) for p in staged):raise ValueError('Staged scope differs from owned task paths')
    git(ctx,'commit','-m','Complete native-coordinate publication amendment and represented I-95 study')
    after=git(ctx,'rev-parse','HEAD')
    result=dict(status='COMMITTED',before_head=before,commit=after,branch='phase1/icnn-rev',scoped_files=staged,
        remote_push=False,preserved_initial_untracked=read_json(ctx.root/'INTAKE_RECEIPT.json')['initial_status_porcelain'],
        final_status_porcelain=git(ctx,'status','--short','--untracked-files=normal'))
    write_json(ctx.root/'COMMIT_RECEIPT.json',result);return result


def run(ctx,args):
    from .final_evidence import source_seal
    seal=source_seal(ctx)
    status=reports(ctx)
    try:commit=commit_owned(ctx)
    except Exception as exc:
        # Never repair an index/branch conflict by resetting, stashing or broad staging.
        commit=dict(status='NOT_COMMITTED',reason=str(exc),branch=git(ctx,'branch','--show-current'),commit=git(ctx,'rev-parse','HEAD'));write_json(ctx.root/'COMMIT_RECEIPT.json',commit)
    status.update(commit=commit.get('commit'),commit_status=commit['status'],immutable_source_seal=seal)
    write_json(ctx.root/'02_HANDOFF_STATUS.json',status)
    root=ctx.root;stage=root/'handoff_staging'/args.job_id;stage.mkdir(parents=True,exist_ok=False);members=[]
    def add(path,name=None):
        p=Path(path);relative=name or p.relative_to(root).as_posix()
        if p.stat().st_size>25_000_000:raise ValueError('Handoff member >25 MB: '+relative)
        dest=stage/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        members.append(dict(relative_path=relative,sha256=sha(dest),size_bytes=dest.stat().st_size))
    for p in root.iterdir():
        if p.is_file() and p.suffix in ('.md','.json','.csv') and p.name!='ACTIVE_SCIENCE_CHILD.json':add(p)
    # Include the compact active contract, configuration and mathematical tests;
    # historical bulk reference archives remain authenticated native inputs.
    contract_files=[p for p in ctx.authority.iterdir() if p.is_file() and (p.suffix=='.md' or p.name=='MANIFEST_SHA256.csv')]
    for folder in ('config','tests','scripts'):
        contract_files.extend(p for p in (ctx.authority/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for p in sorted(contract_files):add(p,'authority/'+p.relative_to(ctx.authority).as_posix())
    for folder in ('publication_bundle','registries','closure','reference','metrics','regional_metrics','receipts'):
        for p in sorted((root/folder).rglob('*')):
            if not p.is_file() or '.mplcache' in p.parts:continue
            if p.suffix.lower() in ('.aux','.log','.out','.synctex','.toc'):continue
            if folder in ('reference','metrics','regional_metrics') and p.suffix in ('.npz','.parquet'):continue
            if folder=='publication_bundle' and p.name in ('main_numerical.bbl','main_numerical.blg'):continue
            if p.stat().st_size>25_000_000:
                if folder=='publication_bundle':raise ValueError('Required publication member exceeds 25 MB: '+str(p))
                continue # Indexed native evidence; required public inputs are always bundled.
            add(p)
    for job in (root/'jobs').iterdir():
        if not job.is_dir():continue
        if job.name==args.job_id:continue # Closing supervisor receipt is issued after sealing, outside the archive.
        for name in ('START.json','EXIT.json','CODE_SNAPSHOT.json','RUN_MATRIX.csv','RESOLVED_CONFIG.json','stdout.log','stderr.log'):
            p=job/name
            if p.exists():add(p)
    for run in (root/'runs').iterdir():
        if not run.is_dir():continue
        for p in run.rglob('*'):
            if p.is_file() and p.name in ('RESULT.json','PRE_RUN_FREEZE.json','HISTORY.json','EPOCHS.json','SAFETY_STOP.json','INFERENCE_TIMINGS.json','OD_COVERAGE.csv','fresh_certificates.jsonl','iteration_history.csv'):
                add(p)
    for name in ('FIT_RESULTS.json','COMBINED_FIT_INVENTORY.json','STUDY_FREEZE.json','LEARNING_CLOCK.json','ALL_EXPERIMENT_MODELS_FROZEN.json','selected_regions_16.csv','combined_label_coverage.csv','EVALUATION_RESULT.json'):
        p=root/'learning'/name
        if p.exists():add(p)
    for p in (root/'learning/cost_wmape').glob('*.json'):add(p)
    for p in (root/'validation').rglob('*'):
        if not p.is_file():continue
        if any('_alone_' in part or part.startswith('cwd_') or part=='.mplcache' for part in p.parts):continue
        if p.name in ('TESTS.json','VISUAL_QA.json','visual_QA.md','ISOLATION_RESULT.json','PDF_BUILD.json','RESULT.json','CHECKS.csv','INDEPENDENT_SUMMARIES.csv','PDF_TEXT_EXTRACTION.json','layout_warnings.txt') or p.name.endswith('_RESULT.json') or p.suffix=='.log':add(p)
    # The supervisor/Context continue to append to the live resource log during
    # sealing. Bundle and index a finite snapshot, never a stale live-file hash.
    resource_snapshot=root/'logs/SEALED_RESOURCE_USAGE.jsonl'
    if resource_snapshot.exists():raise FileExistsError('A resource snapshot already exists; use an explicitly reconciled reseal')
    shutil.copy2(root/'logs/resource_usage.jsonl',resource_snapshot);add(resource_snapshot)
    identity,code,_=code_identity(ctx.repo)
    for relative in code:add(ctx.repo/relative,'code_snapshot/'+relative)
    write_json(stage/'CODE_SNAPSHOT_MANIFEST.json',dict(identity=identity,files=code,commit=commit))
    # Inventory every persisted scientific file, while reproducible rendering
    # scratch/cache/toolchain copies are explicitly outside native evidence scope.
    native=[]
    excluded={'cache','toolchain','handoff_staging','publication_bundle','documentation_backups'}
    for p in root.rglob('*'):
        if not p.is_file():continue
        relative=p.relative_to(root)
        if relative.parts[0] in excluded or (relative.parts[0]=='validation' and ('publication' in relative.parts or 'pytest' in relative.parts)):continue
        if len(relative.parts)>1 and relative.parts[0]=='jobs' and relative.parts[1]==args.job_id:continue
        if relative.as_posix() in ('ACTIVE_SCIENCE_CHILD.json','logs/resource_usage.jsonl'):continue
        ctx.guard();native.append(dict(relative_path=relative.as_posix(),native_path=str(p),size_bytes=p.stat().st_size,sha256=sha(p)))
    pd.DataFrame(native).to_csv(stage/'NATIVE_ARTIFACT_INDEX.csv',index=False)
    write_json(stage/'NATIVE_INDEX_SCOPE.json',dict(excluded_reproducible_scopes=sorted(excluded|{'validation/publication','validation/pytest'}),
        required_publication_files='Bundled directly with their local provenance; no required plot CSV is omitted.',source_assets='Exact immutable input paths and hashes in receipts and private provenance',
        closing_live_exclusions=['logs/resource_usage.jsonl','jobs/'+args.job_id,'ACTIVE_SCIENCE_CHILD.json'],
        frozen_resource_log='logs/SEALED_RESOURCE_USAGE.jsonl',closing_supervisor_receipt='Issued after sealing and retained beside this archive'))
    # Rebuild the final member manifest after adding the closeout indices.
    final=[]
    for p in sorted(stage.rglob('*')):
        if p.is_file():
            if p.stat().st_size>25_000_000:raise ValueError('Generated manifest/index exceeds 25 MB: '+str(p))
            final.append(dict(relative_path=p.relative_to(stage).as_posix(),sha256=sha(p),size_bytes=p.stat().st_size))
    pd.DataFrame(final).to_csv(stage/'MANIFEST_SHA256.csv',index=False)
    if (stage/'MANIFEST_SHA256.csv').stat().st_size>25_000_000:raise ValueError('Final member manifest exceeds 25 MB')
    archive=root.parent/('N8R4_V5_PUBLICATION_HANDOFF_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_LZMA,allowZip64=True) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():ctx.guard();z.write(p,p.relative_to(stage).as_posix())
    if archive.stat().st_size>200_000_000:raise ValueError('Handoff exceeds hard 200 MB limit; preserve attempt and reseal within declared scope')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Handoff CRC verification failed')
        for row in final:
            if hashlib.sha256(z.read(row['relative_path'])).hexdigest()!=row['sha256']:raise ValueError('Handoff manifest hash mismatch')
    result=dict(status='WAITING_FOR_EXTERNAL_REVIEW',archive_path=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        manifest_payload_members=len(final),archive_members=len(final)+1,commit=commit,next_round_authorized=False,document_complete=status['document_complete'],
        scientific_execution_complete=status['scientific_execution_complete'],elapsed_seconds=time.time()-ctx.c['start_epoch'])
    write_json(root/'HANDOFF_RECEIPT.json',result)
    archive.with_suffix('.zip.sha256').write_text(result['archive_sha256']+'  '+archive.name+'\n',encoding='ascii')
    return result
