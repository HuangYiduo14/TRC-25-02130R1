"""Evidence-derived progress and terminal reports; never approves later work."""
from pathlib import Path
import csv
import json
import os
import time
from datetime import datetime,timezone
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,write_json,sha


REPORT_NAMES=['00_READ_FIRST.md','01_EXECUTIVE_REVIEW.md','02_HANDOFF_STATUS.json',
    '03_RESULTS_AND_NUMERICAL_CLAIMS.md','04_REVIEWER_RESPONSE_NUMERICAL.md',
    '05_LIMITATIONS_AND_FAILURES.md','06_NEXT_METHOD_SECTION_EDITS.md','07_RUNTIME_AND_OFFLINE_ACCOUNTING.md']


def utc(epoch):return datetime.fromtimestamp(epoch,timezone.utc).isoformat()


def publication_qa(root):
    release=root/'numerical_release';visual_path=root/'validation/VISUAL_QA.json';pdf_path=root/'validation/CURRENT_PDF_BUILD.json'
    visual=read_json(visual_path) if visual_path.exists() else {};pdf=read_json(pdf_path) if pdf_path.exists() else {}
    actual_pdf=release/'main_numerical.pdf'
    pdf_hash=sha(actual_pdf) if actual_pdf.is_file() else None
    input_bindings=pdf.get('manuscript_input_bindings',[])
    pdf_pass=bool(pdf_hash and pdf_hash==pdf.get('pdf_sha256') and input_bindings and
        all((release/r['relative_path']).is_file() and sha(release/r['relative_path'])==r['sha256'] for r in input_bindings))
    visual_pass=bool(pdf_pass and visual.get('status')=='PASS' and visual.get('pdf_sha256')==pdf_hash and
        visual.get('all_manuscript_pages_inspected') is True and visual.get('inspected_page_count')==pdf.get('pages'))
    inspected=visual.get('figure_png_bindings',{})
    expected_pngs={p.relative_to(release).as_posix():p for p in release.glob('fig_*/panel_*.png')}
    visual_pass=visual_pass and bool(expected_pngs) and set(inspected)==set(expected_pngs)
    if visual_pass:visual_pass=all(sha(p)==inspected[name] for name,p in expected_pngs.items())
    candidates=[]
    for p in (root/'validation/figure_isolation').glob('*/ISOLATION_RESULT.json'):
        r=read_json(p)
        if r.get('status')=='PASS' and r.get('all_ten_figures') and r.get('current_release_bindings'):
            candidates.append((r.get('completed_epoch',0),p,r))
    isolation_pass=False;isolation_path=None
    if candidates:
        _,isolation_path,record=max(candidates,key=lambda x:x[0])
        isolation_pass=all((release/r['relative_path']).is_file() and sha(release/r['relative_path'])==r['sha256']
            for r in record['current_release_bindings'])
    return dict(pdf_compiled=pdf_pass,visual_QA=bool(visual_pass),isolated_replot_QA=isolation_pass,
        isolation_receipt=str(isolation_path) if isolation_path else None,current_pdf_sha256=pdf_hash)


def table(frame,columns):
    def cell(v):
        if isinstance(v,float):return '' if pd.isna(v) else f'{v:.9g}'
        return str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(columns)+' |','|'+'---|'*len(columns)]+
        ['| '+' | '.join(cell(row.get(c,'')) for c in columns)+' |' for row in frame.to_dict('records')])+'\n'


def main_findings(ctx):
    path=ctx.root/'numerical_release/numerical_facts.csv'
    if not path.exists():return ''
    binding=ctx.bind(path,'CURRENT_SOURCE_BOUND_MANUSCRIPT_FACTS_FOR_REPORT_FIRST_SUMMARY')
    facts=pd.read_csv(path,keep_default_na=False).set_index('fact_id').value
    def n(key):return format(float(facts[key]),'.6g')
    failed=int(float(facts['LiftCompletedODs']))-int(float(facts['LiftContinuityVerified']))
    text='## Main scientific findings\n\n'
    text+=('- The primary bound-safe **TIME_LIMIT** snapshot has represented-demand PPS path WREL **'+n('PrimaryPPSWrel')+
        '%**, with paired-draw 95% bootstrap interval ['+n('PrimaryPPSLow')+', '+n('PrimaryPPSHigh')+
        ']%. Its mean absolute relative error with a one-second denominator floor is '+n('PrimaryPPSWmeanrel')+
        '% across '+n('PrimaryPPSPairs')+' finite paired draw units. These distinct estimands preserve duplicate draws and apply no second demand weighting. A small optimization gap does not certify physical accuracy.\n')
    text+=('- The exact-recoverable intersection across all five suites contains '+n('CommonArcs')+' of '+n('PhysicalArcs')+
        ' physical directed arcs, carrying '+format(100*float(facts['CommonFlowShare']),'.6g')+'% of reference arc-flow and '+
        format(100*float(facts['CommonFlowCostShare']),'.6g')+'% of reference flow-cost. This restricted population does not cover all national physical arcs and is not a served-OD fraction.\n')
    text+=('- The nested learning study completed '+n('LearningCompleteFits')+'/'+n('LearningPlannedFits')+' fits across '+n('LearningRegions')+
        ' regions and '+n('LearningSeeds')+' seed. Of '+n('FreshLabelPlanned')+' fresh label attempts, '+n('FreshLabelAB')+
        ' met A/B quality. The m28 operating cohort admitted only '+n('FreshMtwentyEightOperatingAB')+'/'+n('FreshMtwentyEightOperatingPlanned')+
        '; its stress cohort admitted '+n('FreshMtwentyEightStressAB')+'/'+n('FreshMtwentyEightStressPlanned')+
        '. VALID selected initialization for '+n('LearningInitializationSelectedFits')+' fits in '+n('LearningInitializationSelectedRegions')+
        ' region despite executed updates. This small study does not establish monotone or national learning gains.\n')
    text+=('- Local positive-flow witnesses cover '+n('PositiveLocalWitnesses')+' coarse pairs, but only '+n('LiftContinuityVerified')+'/'+
        n('LiftCompletedODs')+' selected distinct-OD stitches were verified, including '+n('LiftVerifiedSelfODs')+' empty self-OD path. The other '+
        str(failed)+' selected witness stitches failed. This is neither a probability-sample lift rate nor proof that every alternative fine path fails.\n')
    text+=('- The Virginia–Florida I-95 intervention is **UNSUPPORTED_MAPPING**. Complete historical offline elapsed time, CPU/GPU totals, full amortization and independent observed-traffic validation remain unestablished.\n\n')
    return text+'Numerical values above are drawn from `numerical_release/numerical_facts.csv` (SHA-256 `'+binding+'`), whose rows identify exact sources, units, denominators and cohorts.\n\n'


def collect(root,final=False):
    matrix=pd.read_csv(root/'run_matrix.csv',keep_default_na=False);rows=[]
    for original in matrix.to_dict('records'):
        job=original['job_id'];p=root/'jobs'/job;exitpath=p/'EXIT.json';rpath=root/'receipts'/(job+'_RESULT.json')
        result=read_json(rpath) if rpath.exists() else {};native=result.get('result') or {}
        exitrecord=read_json(exitpath) if exitpath.exists() else {}
        if exitrecord:
            if exitrecord.get('supervisor_stop'):status=exitrecord['supervisor_stop']
            elif exitrecord['exit_code']!=0:status='FAILED'
            else:status=native.get('stop_reason',result.get('status','EXIT_ZERO_NO_RESULT'))
        elif (p/'START.json').exists():status='ACTIVE'
        else:status='NOT_RUN_BUDGET' if final else 'PLANNED'
        certificate=native.get('final_certificate') or {}
        rows.append(dict(original,status=status,complete_child_seconds=exitrecord.get('elapsed_seconds'),
            start_epoch=exitrecord.get('start_epoch'),ended_epoch=exitrecord.get('ended_epoch'),
            exit_code=exitrecord.get('exit_code'),supervisor_stop=exitrecord.get('supervisor_stop'),
            peak_process_tree_rss_bytes=exitrecord.get('peak_process_tree_rss_bytes'),
            fresh_RG_SPTT=certificate.get('RG_SPTT'),accepted_updates=native.get('accepted_iterations'),
            result_path=str(rpath) if rpath.exists() else '',result_sha256=sha(rpath) if rpath.exists() else '',
            exit_path=str(exitpath) if exitpath.exists() else '',exit_sha256=sha(exitpath) if exitpath.exists() else '',
            failure_detail=result.get('error','')))
    return pd.DataFrame(rows)


METHOD_EDITS='''# Required later method, theory and literature edits

These are recorded future edits; no next round is authorized.

1. Replace the obsolete encoder/decoder and one-hidden-layer description with the implemented two-layer width-32 softplus skip-ICNN and linear head. Describe the tangent-anchored scalar potential and its full coupled gradient and Hessian-vector product.
2. Distinguish convexity/integrability from componentwise nonnegative gradients. Original frozen weights produced reproduced negative costs. If the separately named bound-safe potential is adopted later, state its weight-only coefficient, assumptions, numerical guard and fractional floor; do not call it the original portfolio.
3. Replace obsolete GP path-set, symmetric-difference, pi/pi-star and approximate diagonal-Jacobian notation with the actual fixed-potential FW and safeguarded-CFW algorithms, feasible AON initialization and fresh signed certificate. Reconcile the nonbasic-path/difference-set definitions and the legacy L_delta and Gamma symbols explicitly.
4. Define the accepted active regional support and directional terminal roles, replacing or precisely defining legacy hyperedge terminology. Explain B-node suppression, compiled series/parallel mappings and OD wrappers. Counts at P3, P5 and N8 are different objects.
5. Replace any universal fine-path lift or physical optimality claim with the measured local feasibility, coarse conservation, selected stitching and physical comparison scopes. Revisit the regional-to-national error inequality; current numerical evidence is not its proof.
6. Use partition for the Leiden construction; distinguish raw partition, final ownership correction and changing backbone. Revise method Figure 1 using actual terminal/centroid roles and a clear legend.
7. Revisit MFD motivation and the requested Liu/Yin, Liu/Meidani and other assignment/surrogate literature. Correct bibliography typos, DOI/details and duplicated agency references in a separate literature revision. No unverified citation is supplied here.
8. Keep independent observed counts, cross-year validation and actual disruption evidence as missing unless qualified data are obtained. Static annual freight does not model staging, hours-of-service, parking, time windows or dynamic interactions.
9. Retire the old emissions/COPERT/GREET and Baltimore extrapolation claims from this numerical section. Any later emissions analysis needs its own current model and evidence; GREET must be described as a model.
10. Public deposition remains a user action with source permissions and an actual repository URL. This task delivers a local review archive only.
'''


def refresh(ctx,args,final=False):
    root=ctx.root;now=time.time();jobs=collect(root,final)
    jobs.to_csv(root/'experiment_inventory.csv',index=False,float_format='%.17g')
    matrix=pd.read_csv(root/'run_matrix.csv',keep_default_na=False)
    matrix['status']=matrix.job_id.map(jobs.set_index('job_id').status)
    matrix.to_csv(root/'run_matrix.csv',index=False)
    clock=jobs[['job_id','task','status','cap_seconds','complete_child_seconds','start_epoch','ended_epoch',
        'exit_code','peak_process_tree_rss_bytes','exit_path','exit_sha256']].copy()
    clock['original_R4_start_epoch']=ctx.c['start_epoch'];clock['R4_science_cutoff_epoch']=ctx.c['compute_cutoff_epoch']
    clock['R4_hard_deadline_epoch']=ctx.c['deadline_epoch']
    clock.to_csv(root/'time_budget_ledger.csv',index=False,float_format='%.17g')
    scenario_path=root/'tables/SCENARIO_RESULTS.csv'
    scenarios=pd.read_csv(scenario_path) if scenario_path.exists() else jobs.loc[jobs.scenario.ne(''),
        ['scenario','status','complete_child_seconds','fresh_RG_SPTT','accepted_updates']].copy()
    coverage=[]
    for p,scope in [(root/'physical_population/COMMON_POPULATION_COVERAGE.csv','EXACT_RECOVERABLE_PHYSICAL_POPULATION'),
        (root/'regional_metrics/coverage.csv','REGIONAL_SOURCE_COHORTS'),
        (root/'regional_metrics/model_prediction_coverage.csv','REGISTRY_COMPLETED_PREDICTIONS'),
        (root/'learning/new_label_coverage.csv','NEW_POSTFREEZE_LOCAL_LABELS')]:
        if p.exists():
            f=pd.read_csv(p,keep_default_na=False);f.insert(0,'coverage_scope',scope)
            f['coverage_source_path']=str(p);f['coverage_source_sha256']=sha(p);coverage.append(f)
    if coverage:pd.concat(coverage,ignore_index=True).to_csv(root/'coverage.csv',index=False)
    else:pd.DataFrame([dict(coverage_scope='PENDING',status='NO_COMPLETED_COVERAGE_EXPORT')]).to_csv(root/'coverage.csv',index=False)
    qa=publication_qa(root)
    assets=bool(final and qa['visual_QA'] and qa['isolated_replot_QA'] and qa['pdf_compiled'])
    if assets:
        for name in ('figure_inventory.csv','table_inventory.csv'):
            inventory=pd.read_csv(root/name,keep_default_na=False)
            inventory['status']='VALIDATED_AND_VISUALLY_INSPECTED'
            inventory['visual_qa_sha256']=sha(root/'validation/VISUAL_QA.json')
            inventory.to_csv(root/name,index=False)
    source_path=root/'validation/FINAL_SOURCE_VERIFICATION.json'
    source_status=read_json(source_path).get('status','UNKNOWN') if source_path.exists() else 'FINAL_VERIFICATION_PENDING'
    active=root/'ACTIVE_SCIENCE_CHILD.json'
    active_record=read_json(active) if active.exists() else {}
    own_reporting_lock=bool(active_record.get('job_id')==args.job_id and
        active_record.get('child_pid')==os.getpid() and active_record.get('task')==ctx.task and ctx.task.startswith('REPORT'))
    priority_order=['C02_PHYSICAL_CFW','C03_PHYSICAL_FW','C04_PHYSICAL_MSA',
        'D00_LEARNING_PREPARE','D01_LEARNING_STUDY','B08_CURRENT_REGIONAL_METRICS',
        'C05_PATH_METRICS','C06_PHYSICAL_FLOW_COMPARE','E02_PARTITION_STABILITY',
        'E03_BOUNDARY_PATH_DIAGNOSTICS','E05_REGIONAL_GEOGRAPHY','E04_OFFLINE_ACCOUNTING',
        'F01_BUILD_RELEASE','F02_RENDER_RELEASE','F03_COMPILE_PDF']
    pending=set(jobs.loc[jobs.status.eq('PLANNED'),'job_id'])
    next_job=next((job for job in priority_order if job in pending),None)
    next_step=('External review of the final handoff; no next scientific round is authorized.' if final else
        ('Await '+active_record['job_id']+'; then ' if active_record and not own_reporting_lock else '')+
        ('execute '+next_job+' within the unchanged original clock and its explicit budget.' if next_job else
         'complete current evidence validation and final review packaging within the unchanged original clock.'))
    state=dict(authority_id=ctx.c['authority_id'],authority_sha256=ctx.c['authority_sha256'],phase_id='N8',step_id='N8R4_NUM',
        run_id=ctx.c['run_id'],execution='PARTIAL_WITH_LIMITATIONS' if final else 'IN_PROGRESS',
        review_status='WAITING_FOR_EXTERNAL_REVIEW' if final else 'IN_PROGRESS',publication_assets_complete=assets,
        reviewer_numerical_evidence='MIXED_ADDRESSED_PARTIAL_NOT_AVAILABLE_AND_DEFERRED',
        scientific_claims_supported='LIMITED_TO_EXACT_REPORTED_SCENARIOS_POPULATIONS_AND_CERTIFICATES',
        source_integrity=source_status,next_round_authorized=False,
        next_step_recommendation=next_step,
        original_R4_start_epoch=ctx.c['start_epoch'],science_cutoff_epoch=ctx.c['compute_cutoff_epoch'],
        hard_deadline_epoch=ctx.c['deadline_epoch'],status_written_epoch=now,elapsed_total_seconds=now-ctx.c['start_epoch'],
        heavy_job_active=active_record.get('job_id') if active_record and not own_reporting_lock else None,
        manuscript_pdf='numerical_release/main_numerical.pdf' if (root/'numerical_release/main_numerical.pdf').exists() else None,
        **qa)
    if final and active.exists() and not own_reporting_lock:raise RuntimeError('Cannot write terminal reports with another child active')
    write_json(root/'02_HANDOFF_STATUS.json',state);write_json(root/'RUN_STATE.json',state)
    intro=f'Updated {utc(now)}. Original R4 start {utc(ctx.c["start_epoch"])}; elapsed {(now-ctx.c["start_epoch"])/3600:.3f} hours.\n\n'
    (root/'00_READ_FIRST.md').write_text('# N8R4 numerical review\n\n'+intro+
        ('Status: **WAITING_FOR_EXTERNAL_REVIEW**. No next round is authorized.\n\n' if final else 'Status: **IN_PROGRESS**. This is a maintained progress report, not a final handoff.\n\n')+
        'Read `01_EXECUTIVE_REVIEW.md`, `02_HANDOFF_STATUS.json`, `03_RESULTS_AND_NUMERICAL_CLAIMS.md`, then the reviewer response and limitations. '
        'The numerical manuscript and standalone figure/table folders are under `numerical_release/`. '
        'Exact job status/time/failure records are in `experiment_inventory.csv` and `time_budget_ledger.csv`. '
        'Source hashes, coverage and claim crosswalks define what the archive supports. Original negative failures, time-limited states and unsupported closure are retained.\n',encoding='utf-8')
    compact=jobs[['job_id','status','complete_child_seconds','fresh_RG_SPTT','accepted_updates']]
    summary='# Executive review\n\n'+intro
    summary+='The round extends the accepted R2/R3 evidence with separately named fixed-safe potentials, contemporary physical controls, paired path/flow diagnostics, a bounded local learning experiment and a numerical release. s01 is excluded. Source models, accepted supports, baseline capacities and physical x45/x200 references remain protected.\n\n'
    findings=main_findings(ctx);summary+=findings
    summary+='## Current experiment outcomes\n\n'+table(compact,compact.columns.tolist())
    summary+='\nA fresh small optimization gap is distinct from physical approximation error. TIME_LIMIT is not convergence. Original cold neural failures occur before the first accepted update. Their unchanged warm diagnostics and the fixed-safe variants have separate identities.\n\n'
    closure=root/'geography/I95_CLOSURE_STATUS.json'
    if closure.exists():
        cl=read_json(closure);summary+=f'I-95 disposition: **{cl["status"]}**. The intended physical closure cannot be represented faithfully while preserving the required compiled representation. No substitute logical cut was run.\n\n'
    summary+='Complete offline elapsed cost and full break-even remain unestablished when historical scopes are missing. Local observation candidates are not independent validation until qualified and matched. Method/literature edits and public deposition remain separate work.\n'
    (root/'01_EXECUTIVE_REVIEW.md').write_text(summary,encoding='utf-8')
    results='# Results and numerical claims\n\n'+intro+findings+table(scenarios,scenarios.columns.tolist()[:12])
    results+='\n## Claim boundaries\n\n- Original frozen neural portfolios and new fixed-safe variants are distinct scalar potentials.\n- Reported path errors specify FIXED_X45 or SOLVED_X45_REFERENCE, state, cohort and weights. PPS duplicate draws stay in the denominator; no second demand weighting is applied.\n- Physical WMAPE uses only the frozen exact-recoverable common physical-arc population; unrecovered interiors are not imputed zeros.\n- Regional E2 and old tests are reused now-seen diagnostics; postfreeze learning labels have separate coverage.\n- The preserved 32.403893% TRAIN-e2 headline is the national equal-region standalone NN mean, not m28\'s own error. Leave-m28-out is sensitivity only.\n- Selected global stitches are diagnostics, not a probability lift rate or universal fine-network optimality proof.\n\nEvery final numerical sentence and panel is mapped through `claim_to_result.csv`; full figure observations remain in local drawing CSVs.\n'
    (root/'03_RESULTS_AND_NUMERICAL_CLAIMS.md').write_text(results,encoding='utf-8')
    acceptable=jobs.status.isin(['PASS','EXECUTED','CONVERGED','CONVERGED_REPRESENTED_DEMAND_ONLY','ACTIVE','PLANNED'])
    failures=jobs.loc[~acceptable]
    limitations='# Limitations and failures\n\n'+intro
    if len(failures):limitations+=table(failures,['job_id','status','complete_child_seconds','fresh_RG_SPTT','failure_detail'])+'\n'
    limitations+='## Scientific scope\n\nThe primary physical reference is an accepted finite-gap model state, not observed truth. x200 is sensitivity. Reused diagnostics cannot be relabeled new held-out validation. Exact common-population physical recovery does not cover all physical arcs or certify global lifting. Fixed-safe attenuation establishes only the declared fractional gradient bound under its assumptions, not accuracy or the full free-flow floor.\n\nThe I-95 membership audit identifies hidden compiled movements; a faithful counterfactual remains unsupported. No emissions/modal/Baltimore extrapolation is supplied. Static annual freight omits operational staging, hours-of-service, delivery windows and dynamic interactions. Independent truck counts, cross-year and actual event validation remain unestablished. Full historical offline/CPU costs and complete amortization are not invented from partial worker timers.\n\nA complete figure/PDF asset bundle does not close every reviewer concern or approve the next scientific round.\n'
    (root/'05_LIMITATIONS_AND_FAILURES.md').write_text(limitations,encoding='utf-8')
    (root/'06_NEXT_METHOD_SECTION_EDITS.md').write_text(METHOD_EDITS,encoding='utf-8')
    runtime='# Runtime and offline accounting\n\n'+intro
    runtime+=f'Science cutoff: {utc(ctx.c["compute_cutoff_epoch"])}. Hard delivery deadline: {utc(ctx.c["deadline_epoch"])}. The final four hours are reserved for reporting.\n\n'
    runtime+=table(clock,['job_id','status','cap_seconds','complete_child_seconds','peak_process_tree_rss_bytes'])
    runtime+='\nNew complete supervised-job wall timers start before code snapshotting and child launch and include input, execution and exit work. The legacy inventory column complete_child_seconds refers to this supervised-job scope, not a separately measured child-process interval. Solver, update, initialization and refresh timers retain their explicit nested scopes. Historical x45/x200 cumulative recorded iteration-wall times include checkpoint/refresh work inside the recorded iterations; they are not pure update/kernel clocks or contemporary complete assignment runtimes. The later zero-update x45 refresh is separate. Serial one-worker controls use the same physical backend/input policy; historical production used four workers, its hardware identity is not established by the targeted metadata, and complete x45 cold elapsed time remains unknown.\n\n'
    study_path=root/'learning/STUDY_FREEZE.json'
    if study_path.exists():
        study=read_json(study_path)
        runtime+='## Frozen learning runtime forecast and observed execution\n\n'
        runtime+=table(pd.DataFrame(study['runtime_pilot_choices']),
            ['regions','seeds','forecast_seconds','available_training_seconds','fits','forecast_coverage_complete'])
        runtime+='\nThese are pre-outcome training-runtime forecasts from the saved pilot, not observed completion times or confidence bounds. '
        runtime+='The declared minimum eight-region, one-seed study was frozen with an explicit forecast-exceeds-budget warning. '
        runtime+='Its region/seed docket was not expanded after observing the faster realized execution. '
        runtime+='Exact policy, forecast and pilot hash: learning/STUDY_FREEZE.json (SHA-256 '+sha(study_path)+').\n\n'
        prep_path=root/'jobs/D00_LEARNING_PREPARE/START.json';study_exit=root/'jobs/D01_LEARNING_STUDY/EXIT.json'
        if prep_path.exists() and study_exit.exists():
            prep=read_json(prep_path);ended=read_json(study_exit)
            runtime+='The complete supervised preparation-to-study span, including the launch gap and postfreeze label evaluation, was '
            runtime+=format(ended['ended_epoch']-prep['start_epoch'],'.6f')+' seconds. '
            runtime+='The separate D00 and D01 job intervals and peak process-tree memory are listed above. '
            runtime+='The conservative training forecasts exceeded even this broader observed complete-stage span; the recorded forecasts remain unchanged.\n\n'
    ap=root/'accounting/offline_stage_subtotals.csv'
    if ap.exists():runtime+='## Disjoint measured offline components\n\n'+table(pd.read_csv(ap),['stage','measure','seconds'])+'\n'
    runtime+='Worker wall is not CPU time or critical-path elapsed. Nested intervals and inherited/resumed counters are not added twice. Complete offline intercepts are unknown; online-only repeated-query arithmetic is not a full break-even estimate.\n'
    (root/'07_RUNTIME_AND_OFFLINE_ACCOUNTING.md').write_text(runtime,encoding='utf-8')
    write_reviewer_response(root,now)
    return dict(status='EXECUTED',terminal=final,publication_assets_complete=assets,job_records=len(jobs))


def write_reviewer_response(root,now):
    cross=pd.read_csv(root/'reviewer_crosswalk.csv',keep_default_na=False);rows=[]
    evidence={
        'R1_overall':('PARTIAL','audits/DATA_PROVENANCE_AND_OBSERVATIONS.md','Static annual demand and background capacity are documented; operational time dynamics remain outside scope.'),
        'R1_1':('PARTIAL','numerical_release/numerical_section.tex','The numerical revision uses explicit regional potentials, active coordinates and directed terminal roles. A manuscript-wide replacement or definition of the old hyperedge terminology remains a method edit.'),
        'R1_2':('PARTIAL','regional_geography/RESULT.json','New numerical maps use true road geometry and fine-member portal/centroid locations with legends. The original method Figure 1 is unchanged and remains on the revision list.'),
        'R1_3':('PARTIAL','boundary_diagnostics/RESULT.json','Current evidence records terminal-role transitions, selected local witnesses and explicit backbone/regional stitching. Formal entrance/exit-selection definitions still require the method revision; no new boundary rule was introduced.'),
        'R1_4':('DEFERRED_METHOD','06_NEXT_METHOD_SECTION_EDITS.md','The legacy pi and pi-star path-set notation is listed for replacement with the actual fixed-potential FW formulation. The old method definitions were not rewritten in this numerical-only round.'),
        'R1_5':('DEFERRED_METHOD','06_NEXT_METHOD_SECTION_EDITS.md','The legacy nonbasic-path and difference-set description remains a method edit. Numerical execution uses the frozen feasible-flow/AON algorithm and does not claim that the old path-set exposition has been repaired.'),
        'R1_6':('DEFERRED_METHOD','06_NEXT_METHOD_SECTION_EDITS.md','Definitions of legacy L_delta and Gamma require reconciliation with the implemented objective and solver. Current numerical units, loss terms, caps and certificates are stated separately.'),
        'R1_7':('DEFERRED_LITERATURE','06_NEXT_METHOD_SECTION_EDITS.md','The requested Liu/Yin and Liu/Meidani discussion is explicitly retained for a literature revision; no unverified bibliographic detail or comparison is supplied.'),
        'R1_8':('PARTIAL','stability/RESULT.json','The numerical revision calls the Leiden construction a partition and distinguishes raw partition from corrected ownership. Terminology elsewhere in the method manuscript remains to be revised.'),
        'R2_major_1':('PARTIAL','audits/DATA_PROVENANCE_AND_OBSERVATIONS.md','Consumed release/base-year and filtering stages are bound locally; county-factor release year is not established.'),
        'R2_major_2':('PARTIAL','tables/SCENARIO_RESULTS.csv','Bounded MSA/FW/safeguarded-CFW outcomes and censored runs are reported; bush/origin methods are unrun.'),
        'R2_major_3':('NOT_AVAILABLE','data_access_and_validation_status.csv','Available candidate observations are unqualified; no independent cross-year or event validation is claimed.'),
        'R2_major_4':('DEFERRED_METHOD','06_NEXT_METHOD_SECTION_EDITS.md','A formal MFD connection is not established by these experiments. The motivation and theoretical interpretation remain on the later method/theory revision list.'),
        'R2_major_5':('PARTIAL','path_metrics/PATH_METRICS.csv','Fixed/solved path errors, tails and physical comparison coverage are measured; no regional-to-national error theorem follows.'),
        'R2_major_6':('PARTIAL','validation/native_safety_derivative_tests.csv','Full scalar-potential derivatives and separately named safety transform are tested; the old method/theory section still needs revision.'),
        'R2_major_7':('DEFERRED_LITERATURE','06_NEXT_METHOD_SECTION_EDITS.md','A balanced assignment/surrogate literature revision remains outstanding. The numerical comparison is limited to the implemented controls and makes no superiority claim over unrun methods.'),
        'R2_major_8':('ADDRESSED_WITH_SCOPE','stability/RESULT.json','Accepted s02-s06 constant-regional FFTT resolution and ownership stability are measured; trained all-resolution accuracy is not claimed.'),
        'R2_major_9':('PARTIAL','tables/SCENARIO_RESULTS.csv','Fresh signed same-state certificates, complete curves, failures and time limits are retained with native value/gradient/HVP tests. The obsolete approximate diagonal-Jacobian and method/theory exposition still require the explicitly deferred correction.'),
        'R2_major_10':('PARTIAL','accounting/RESULT.json','Disjoint retained scopes and missing history are reported; complete offline elapsed/CPU totals and full break-even remain unknown.'),
        'R2_major_11':('PARTIAL','boundary_diagnostics/RESULT.json','Positive local witnesses, conservation and selected global stitching are separated from universal liftability/optimality.'),
        'R2_minor_1':('DEFERRED_BIBLIOGRAPHY','06_NEXT_METHOD_SECTION_EDITS.md','The reported author typo, Liu DOI and incomplete reference details are recorded for separate bibliography verification. This round does not invent replacement citation details.'),
        'R2_minor_2':('PARTIAL','audits/DATA_PROVENANCE_AND_OBSERVATIONS.md','Numerical provenance uses the consumed source; full manuscript citation cleanup is deferred.'),
        'R2_minor_3':('ADDRESSED_WITH_SCOPE','learning/STUDY_FREEZE.json','The pre-outcome study freeze records architecture, optimizer, loss, sampling, normalization, configured updates and seeds. Actual fits, labels and completed predictions have separate evidence and are not inferred from this plan.'),
        'R2_minor_4':('DEFERRED_METHOD','06_NEXT_METHOD_SECTION_EDITS.md','The symmetric-difference definition belongs to the legacy path-set method description and remains a named correction. It is not silently reused as a definition of the current solver.'),
        'R2_minor_5':('ADDRESSED_WITH_SCOPE','learning/FIT_RESULTS.json','Nested unique-data budgets are distinct from fixed-data optimization; incomplete fits and labels remain explicit.'),
        'R2_minor_6':('OMITTED_WITH_REASON','06_NEXT_METHOD_SECTION_EDITS.md','Unsupported legacy emissions results are omitted; no replacement emissions experiment was authorized.'),
        'R2_minor_7':('PARTIAL','numerical_release/REPRODUCE.md','A local standalone numerical release is delivered; public deposition and URL remain a future user action.'),
        'R2_minor_8':('DEFERRED_METHOD','06_NEXT_METHOD_SECTION_EDITS.md','Legacy emissions claims are omitted; GREET terminology and bibliography are recorded for later revision.')}
    for item in cross.to_dict('records'):
        cid=item['comment_id']
        disposition,path,response=evidence.get(cid,('DEFERRED_METHOD','06_NEXT_METHOD_SECTION_EDITS.md',
            'The numerical section uses current operational definitions and scoped evidence; full terminology, notation, theory, figure-1 or literature revision remains later work.'))
        exists=(root/path).is_file()
        intended_response=response
        if not exists:
            disposition='PENDING_OR_UNAVAILABLE'
            response='Required current evidence is not available; this response makes no completion claim for the requested numerical work. See the intended response and named evidence path for the remaining scope.'
        if cid in ('R1_3','R2_major_11') and exists:
            diagnostics=read_json(root/path)
            local=int(diagnostics.get('local_positive_pair_witness_rows',0))
            completed=int(diagnostics.get('completed_selected_ODs',0));verified=int(diagnostics.get('verified_selected_global_lifts',0))
            response=(f'Actual diagnostic status: {diagnostics.get("status","UNKNOWN")}; completed positive-flow local witness rows: {local}; '
                f'completed selected OD diagnostics: {completed}; verified selected global stitches: {verified}. '
                'Uncompleted diagnostics are unavailable, not valid or invalid outcomes. National represented-demand conservation has its own same-state certificates. '
                'Formal entrance/exit definitions and universal liftability/optimality claims remain method/theory work.')
            disposition='PARTIAL' if local or completed else 'PENDING_OR_UNAVAILABLE'
        if cid=='R2_minor_7' and exists:
            qa=publication_qa(root)
            ready=all(qa[k] for k in ('pdf_compiled','visual_QA','isolated_replot_QA'))
            response=('A local standalone numerical release has passed current isolation, PDF and visual checks; public deposition and URL remain a future user action.'
                if ready else 'Local release assets have been extracted, but complete current isolation/PDF/visual QA is not yet established. Public deposition and URL remain a future user action.')
            disposition='PARTIAL' if ready else 'PENDING_OR_UNAVAILABLE'
        if cid=='R2_minor_5' and exists:
            fits=read_json(root/'learning/FIT_RESULTS.json');ep=root/'learning/EVALUATION_RESULT.json'
            evaluation=read_json(ep) if ep.exists() else {}
            complete=sum(r['status']=='COMPLETE' for r in fits)
            summary_path=root/'learning/learning_curve_summary.csv'
            measured=set();required={(r['region_id'],r['seed'],r['fraction'],family,cohort)
                for r in fits for family in ('FFTT','POWER4','SKIP_ICNN32') for cohort in ('NEW_OPERATING','NEW_STRESS')}
            if summary_path.exists():
                summary=pd.read_csv(summary_path)
                columns=['region_id','seed','fraction','family','cohort']
                if set(columns+['status','n_evaluation_rows']).issubset(summary.columns):
                    eligible=summary.loc[summary.status.eq('MEASURED')&summary.n_evaluation_rows.eq(8)]
                    if not eligible.duplicated(columns).any():measured=set(eligible[columns].itertuples(index=False,name=None))
            if (complete!=len(fits) or not fits or evaluation.get('status')!='EXECUTED' or
                evaluation.get('fresh_AB',0)!=evaluation.get('fresh_planned') or not required.issubset(measured)):
                disposition='PARTIAL'
            response+=f' Actual completed fits: {complete}/{len(fits)}; certified postfreeze A/B inputs: {evaluation.get("fresh_AB",0)}/{evaluation.get("fresh_planned","not yet finalized")}; evaluation status: {evaluation.get("status","not yet finalized")}; fully measured required fresh fit/family/cohort cells: {len(required & measured)}/{len(required)}.'
        if cid=='R2_major_5' and exists:
            path_metrics=pd.read_csv(root/path)
            path_units=len(path_metrics[['scenario','track']].drop_duplicates())
            physical_path=root/'physical_population/PHYSICAL_FLOW_METRICS.csv'
            physical_units=len(pd.read_csv(physical_path)[['scenario','population']].drop_duplicates()) if physical_path.exists() else 0
            response=(f'Completed path scenario/track units with exported cohort estimators: {path_units}; '
                f'completed physical scenario/population comparisons: {physical_units}. '
                'The frozen recoverable-arc coverage is separate from completed prediction coverage. Unvisited, rejected and interrupted units '
                'remain in their full planned ledgers; finite paired paths define each reported error denominator. '
                'No regional-to-national error theorem or universal physical lifting claim follows.')
            disposition='PARTIAL' if path_units or physical_units else 'PENDING_OR_UNAVAILABLE'
        additional=[]
        if cid in ('R2_minor_3','R2_minor_5'):
            for relative in ('learning/FIT_RESULTS.json','learning/EVALUATION_RESULT.json','learning/learning_curve_summary.csv','learning/new_label_coverage.csv'):
                p=root/relative
                additional.append(dict(path=relative,status='AVAILABLE' if p.is_file() else 'PENDING_OR_UNAVAILABLE',sha256=sha(p) if p.is_file() else None))
        if cid=='R2_major_9':
            for relative in ('validation/native_safety_derivative_tests.csv','tables/FRESH_GAP_HISTORY.csv','06_NEXT_METHOD_SECTION_EDITS.md'):
                p=root/relative
                additional.append(dict(path=relative,status='AVAILABLE' if p.is_file() else 'PENDING_OR_UNAVAILABLE',sha256=sha(p) if p.is_file() else None))
        if cid=='R2_major_5':
            for relative in ('physical_population/COMMON_POPULATION_COVERAGE.csv','physical_population/PHYSICAL_FLOW_METRICS.csv',
                             'physical_population/COMPARISON_STATES.json','path_metrics/METRIC_STATES.json'):
                p=root/relative
                additional.append(dict(path=relative,status='AVAILABLE' if p.is_file() else 'PENDING_OR_UNAVAILABLE',sha256=sha(p) if p.is_file() else None))
        rows.append(dict(comment_id=cid,reviewer_question=item['reviewer_question'],actual_disposition=disposition,
            evidence_path=path,evidence_sha256=sha(root/path) if exists else '',actual_response=response,
            intended_scope_response=intended_response,
            additional_evidence=json.dumps(additional,separators=(',',':')),
            further_work_authorized=False,original_requested_assets=item['assets']))
    frame=pd.DataFrame(rows);frame.to_csv(root/'reviewer_response_actual.csv',index=False)
    (root/'04_REVIEWER_RESPONSE_NUMERICAL.md').write_text('# Numerical reviewer response\n\nUpdated '+utc(now)+
        '. Each row reports actual evidence and its remaining scope; a compiled PDF is not approval.\n\n'+
        table(frame,['comment_id','reviewer_question','actual_disposition','evidence_path','actual_response']),encoding='utf-8')
