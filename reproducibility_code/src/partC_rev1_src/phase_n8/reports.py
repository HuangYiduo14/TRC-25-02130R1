"""Explicit report-first R2 checkpoint and exact package fast-track evaluation."""
import argparse
import datetime
import importlib.util
import json
import math
from pathlib import Path
import shutil
import time

import pandas as pd

from .common import Context,read_json,write_json,sha,code_identity
from .collect import gather,records_csv


def fmt(value,digits=3):
    if value is None:return 'Unknown / not applicable'
    return f'{value:,.{digits}f}'


def markdown_table(headers,rows):
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    lines+=['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows]
    return '\n'.join(lines)+'\n'


def verify_sources(root,repo,data):
    rows=[]
    config=read_json(root/'resolved_config.json')
    deadline=config['start_epoch']+(43200 if (root/'R3_ACTIVATION.json').exists() else 28800)
    for i,r in enumerate(data['source_bindings']):
        if i%64==0 and time.time()>=deadline:raise TimeoutError('Original reporting deadline reached during source closure')
        p=Path(r['native_path']);actual=sha(p) if p.exists() else None
        rows.append(dict(native_path=str(p),expected_sha256=r['sha256'],actual_sha256=actual,
            status='UNCHANGED' if actual==r['sha256'] else 'CHANGED_OR_MISSING',size_bytes=r['size_bytes']))
    records_csv(root/'tables/source_preservation.csv',rows)
    import subprocess
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),'-c','core.longpaths=true',*args],text=True).strip()
    original=read_json(root/'receipts/INITIAL_RECONCILIATION.json')
    status=git('status','--porcelain=v1','--untracked-files=normal')
    tracked=git('diff','--name-only').splitlines()
    staged=git('diff','--cached','--name-only').splitlines()
    allowed={'partC_doc_icnn/00_READ_FIRST.md','partC_doc_icnn/PROJECT_STATUS.md','partC_doc_icnn/12_DECISION_LOG.md'}
    clean_science=not(set(tracked)-allowed) and not staged
    record=dict(status='PASS' if all(r['status']=='UNCHANGED' for r in rows) and clean_science else 'FAIL',
        checked_followed_sources=len(rows),checked_bytes=sum(r['size_bytes'] for r in rows),
        changed_followed_sources=[r for r in rows if r['status']!='UNCHANGED'],
        current_branch=git('branch','--show-current'),current_HEAD=git('rev-parse','HEAD'),
        initial_HEAD=original['head'],initial_branch=original['branch'],current_porcelain=status,
        initial_porcelain=original['status_porcelain'],unexpected_tracked_changes=sorted(set(tracked)-allowed),
        staged_paths=staged,scope='Actual followed source hashes and scoped Git state; not an all-filesystem hash audit',
        no_source_model_graph_or_production_code_write=True,epoch=time.time())
    if record['current_branch']!='phase1/icnn-rev':record['status']='FAIL'
    write_json(root/'FILE_PRESERVATION_REPORT.json',record)
    return record


def campaign_manifest(root,repo,data):
    definitions=read_json(root/'SCENARIO_DEFINITION_FREEZE.json')['scenarios']
    estimates={r['scenario']:r for r in data['forecasts']}
    rows=[];assignments=[];metrics=[];remaining=[];fresh_base=[];fresh_high=[]
    for n,d in enumerate(definitions,1):
        f=estimates[d['scenario_id']];r=read_json(root/'runs'/d['pilot_job_id']/'RESULT.json')
        row=dict(d,expected_assignment_identity=r['identity'],pilot_result_sha256=sha(root/'runs'/d['pilot_job_id']/'RESULT.json'))
        if f['gate_status']=='COMPLETED_MATCHING':
            row.update(r3_disposition='REUSE_COMPLETED_MATCHING_PILOT',official_job_id=d['pilot_job_id'],
                assignment_root=str(root/'runs'/d['pilot_job_id']),official_run_cap_seconds=900)
            remaining.append(0.)
        elif f['gate_status']=='SAFETY_REJECTED':
            row.update(r3_disposition='SAFETY_REJECTED',official_job_id=None,assignment_root=None,official_run_cap_seconds=None)
            remaining.append(0.)
        else:
            job=f'R3_A{n:02d}_{d["scenario_id"]}'
            cap=f['proposed_official_run_cap_seconds']
            row.update(r3_disposition='FRESH_OFFICIAL_FW_FROM_FROZEN_INITIAL_RULE',official_job_id=job,
                assignment_root=str(root/'conditional_R3/assignments'/job),official_run_cap_seconds=cap)
            assignments.append(dict(job_id=job,scenario_id=d['scenario_id'],cap_seconds=cap,
                nonempty_task='One full represented-demand ordinary FW experiment',
                expected_outputs=['RESULT.json','HISTORY.json','final_flow.npz','fresh same-state certificate or explicit missing-certificate status'],
                acceptance=['unchanged identities','same-objective oracle','conservation','fresh tolerance or explicit bounded stop']))
            remaining.append(f['fresh_assignment_high_seconds'])
        if f['gate_status']!='SAFETY_REJECTED':
            job=f'R3_M{n:02d}_{d["scenario_id"]}'
            metrics.append(dict(job_id=job,scenario_id=d['scenario_id'],cap_seconds=600,
                nonempty_task='All 456 original sampled ODs in FIXED_X45 and SOLVED_X45_REFERENCE, plus exact common-coordinate flow metrics',
                expected_outputs=['PATH_METRICS.csv','FLOW_METRICS.csv','METRIC_STATES.json','both 456-row path tables'],
                acceptance=['bound physical x45 result','exact cohort identities','explicit unreachable denominators','unchanged source models']))
            remaining.append(f['metrics_high_seconds'])
            fresh_base.append(f['fresh_assignment_base_seconds']);fresh_high.append(f['fresh_assignment_high_seconds'])
        row['forecast_gate_status']=f['gate_status'];row['source_mutations_authorized']=False;rows.append(row)
    high=None if any(x is None for x in remaining) else sum(remaining)+600.
    totalfreshbase=None if any(x is None for x in fresh_base) else sum(fresh_base)
    totalfreshhigh=None if any(x is None for x in fresh_high) else sum(fresh_high)
    c=read_json(root/'resolved_config.json')
    from .finish import source_closure
    files={str(p.relative_to(repo)).replace('\\','/'):sha(p) for p in source_closure(repo)}
    m=dict(schema='N8_R3_COMPLETE_FROZEN_MATRIX_V1',phase_id='N8',step_id='N8R3',
        campaign_id='N8R3_conditional_20260915',original_R2_start_epoch=c['start_epoch'],
        total_deadline_epoch=c['start_epoch']+43200,finalization_reserve_seconds=3600,
        solver='ORDINARY_FW',nominal_tolerance=.001,numeric_workers=1,nested_BLAS_threads=1,
        admission='CONDITIONAL_EXACT_PACKAGE_GATE_ONLY',scenarios=rows,assignment_jobs=assignments,metrics_jobs=metrics,
        expected_outputs=['separate R2 and R3 statuses','all failed/unrun rows','same-state certificates','two path tracks','one final combined handoff'],
        acceptance_tests=['LATEST_CURRENT_UNIT_TESTS','NATIVE_PARITY','PHYSICAL_TIMING_RECONCILIATION','PHYSICAL_SAMPLE_PRECHECK','SOURCE_PRESERVATION'],
        actual_code_files=files,core_code_identity=code_identity(repo)[0],
        high_remaining_work_seconds=high,remaining_validation_and_control_allowance_seconds=600,
        fresh_core_assignment_base_seconds=totalfreshbase,fresh_core_assignment_high_seconds=totalfreshhigh,
        pilot_reuse='Only completed exact matches; censored pilots retain their clocks/results and have distinct fresh official experiment IDs if admitted',
        source_versions=c['input_versions'],max_process_tree_RSS_bytes=10*1024**3,max_added_disk_bytes=150*1024**3,
        frozen_epoch=time.time(),optional_experiments=[],R4_authorized=False,slow_campaign_auto_authorized=False)
    write_json(root/'CAMPAIGN_MANIFEST.json',m)
    return m


def write_r2_reports(root,repo,data,manifest,preservation):
    forecasts=data['forecasts'];core=[r for r in data['runs'] if r['phase']=='N8R2']
    converged=[r for r in core if r['status'].startswith('CONVERGED')]
    rejected=[r for r in core if r['status'].startswith('STOP_')]
    censored=[r for r in core if r['status']=='TIME_LIMIT']
    table=markdown_table(['Scenario','Pilot stop','Updates','Fresh RG','Observed minutes','Assignment base / high minutes'],[
        [f['scenario'],f['stop_reason'],f['accepted_iterations'],fmt(f['achieved_RG'],7),fmt(f['observed_end_to_end_seconds']/60,2),
         (fmt(f['fresh_assignment_base_seconds']/60,2)+' / '+fmt(f['fresh_assignment_high_seconds']/60,2)) if f['fresh_assignment_base_seconds'] is not None else 'Unknown / not admitted'] for f in forecasts])
    (root/'R2_PILOT_RESULTS.md').write_text('# N8R2 pilot results\n\n'+
        f'All nine frozen full-demand scenarios were attempted. {len(converged)} reached fresh tolerance, {len(censored)} stopped for time, and {len(rejected)} stopped on actual unsafe costs. The separate 32-OD fixture completed its integration check; it is excluded from national runtime extrapolation.\n\n'+table+
        '\nEvery full-demand assignment uses all 3,225 original origins and the frozen reachable represented population: 10,190,380 aggregated ODs, 41,632,692 original OD rows and 12,654,353.578361887 demand units. Original full-demand totals are 10,193,770 aggregates, 41,637,922 rows and 12,667,037.704538835 demand units. The remaining inherited unreachable population is reported, not zero-filled.\n\n'+
        'Caps were fixed at 900 seconds before outcomes. The solver can stop earlier to preserve checkpoint/certificate time. TIME_LIMIT is not convergence. Every final result links the accepted state hash, fresh certificate or explicit absence, immutable run clock, graph, demand, code and registry. Uncertified unsafe states retain their own identity and cannot be used as certified final equilibria.\n\n'+
        'See tables/run_matrix.csv, convergence_history.csv, stage_timings.csv, demand_coverage.csv and safety_exceptions.csv. Native predecessor/member/centroid and whole-vector model tests precede these runs. V1 mapping profiles remain historical and are not used as current certificates.\n',encoding='utf-8')
    timingrows=[[f['scenario'],fmt(f['cold_input_load_seconds'],2),fmt(f['full_AON_p50_seconds'],2),
        fmt(f['warm_iteration_p50_seconds'],2),fmt(f['value_gradient_p50_seconds'],4),fmt(f['full_HVP_p50_seconds'],4)] for f in forecasts]
    (root/'RUNTIME_ESTIMATE.md').write_text('# Pilot-based ACNHN assignment runtime\n\n'+table+
        '\n## Evidence and boundaries\n\n'+markdown_table(['Scenario','Process setup s','Full AON median s','Warm update median s','Value/gradient median s','Full HVP median s'],timingrows)+
        '\nHardware: Intel Core i7-13700, 16 physical/24 logical processors, Windows 11, about 31.7 GiB OS-visible memory. Runs use one numeric worker and one nested BLAS/Numba thread, with a stricter 10 GiB numeric-tree guard and at least 3 GiB system memory headroom. No packages changed. Input setup includes authentication, per-process loading and native fixed-state inference. The OS file cache was not flushed.\n\n'+
        'Observed end-to-end timing starts before child imports and input loading. Solver time includes feasible initialization, accepted updates, full same-state certificates and checkpoints. The update-window field measures from the first fresh initial-state certificate to the final fresh certificate. Final AON/refresh and inference call timings are explicit in the tables; unmeasured subcomponents are not replaced by zero.\n\n'+
        'Completed assignments use their observed tolerance time and a 1.5-times repeat planning estimate. Censored trajectories fit up to seven post-initial log gaps against iteration and log(iteration+1), requiring at least five points. Nondecreasing, noisy or strongly disagreeing fits leave the high convergence estimate null. Supported fits use the larger iteration forecast, a 1.5 iteration margin and a 1.5 p90 timing margin. These are engineering heuristics, not convergence guarantees. Cap selection adds 180 seconds for checkpoint/refresh scheduling. Safety-rejected scenarios have no assignment-to-tolerance estimate.\n\n'+
        f'Fresh executable core assignments total {fmt(manifest["fresh_core_assignment_base_seconds"],1)} seconds base and {fmt(manifest["fresh_core_assignment_high_seconds"],1)} seconds high, excluding path metrics and reporting. After exact completed-pilot reuse, conservative remaining R3 work is {fmt(manifest["high_remaining_work_seconds"],1)} seconds, including both metric tracks and 600 seconds for control/validation. The exact gate additionally charges all elapsed time since original R2 start and 3,600 seconds of remaining finalization. See FAST_TRACK_GATE.json for the actual total and decision.\n\n'+
        'Both path tracks are conservatively budgeted as two complete origin sweeps plus process setup, measured inference and I/O, multiplied by 1.5; the fixed 456-OD population is smaller than the full-origin workload. These costs are kept separate from solver-update timing. A tiny imposed time cap never substitutes for a convergence estimate.\n\n'+
        'The physical x45 reference has 3,353.9471747006755 seconds of recorded update work; its independent later refresh alone is 138.0435530000832 seconds. x200 records 14,614.275030500488 update seconds and 17,724.460236787796 broader elapsed seconds. Historical hardware is not fully established, and that benchmark used four workers. A matched-hardware speedup is not claimed. See PHYSICAL_TIMING_AUDIT.md.\n',encoding='utf-8')
    native=pd.read_csv(root/'tables/native_derivative_cohorts.csv')
    latest=sorted([r for r in data['resources'] if r['task']=='unit_tests'],key=lambda r:r['start_epoch'])[-1]
    (root/'R2_IMPLEMENTATION_AND_TESTS.md').write_text('# N8R2 implementation and validation\n\n'+
        'New scoped adapters, coupled objective/gradient/HVP, settlement-safe oracle, feasible ordinary FW, exact-state certificates, external budget supervision, immutable checkpoints and authenticated continuation segments are implemented. Existing production modules and source models/graphs are imported read-only.\n\n'+
        f'The latest scoped regression job is {latest["job_id"]}, return code {latest["returncode"]}. Its log records the actual test count. Native validation covers 4,697 unique models and {len(native)} derivative cases across all five s03 registries, including m=1, typical, largest and every m=28 case. Maximum potential/gradient relative difference is {native.potential_gradient_relative_error.max():.6g}; maximum gradient/HVP relative difference is {native.gradient_HVP_relative_error.max():.6g}. Exact scalar/native value, gradient and HVP parity passes.\n\n'+
        'The test ladder covers full coupling and charge accounting, directed/parallel/near-tie/zero-cost paths, source predecessor safety, centroid nontransit, duplicate targets and empty groups, inherited unreachable demand, zero SPTT/curvature, analytic FW optima, monotonic potential, stale signed certificates, actual nonfinite/negative rejection, atomic writes, time/resource watchdog termination and a real authenticated restart with preserved earlier files and clocks. Optional CFW is not implemented and is not required for ordinary FW.\n\n'+
        'Preserved engineering evidence: B01 stopped at the original single-endpoint adapter assumption; B02 retains every native attachment. The self-OD mapping correction retains initial artifacts and labels obsolete profiles. T06 had 40 passes and one resource-fixture assertion that incorrectly assumed a single Windows process; the watchdog had correctly terminated its helper descendant. T07 verifies every owned terminated PID is gone. No scientific coefficient, source graph, model or membership was changed to repair a test.\n\n'+
        f'Source rehash verification: {preservation["status"]}, {preservation["checked_followed_sources"]:,} followed native files ({preservation["checked_bytes"]:,} bytes), with {len(preservation["changed_followed_sources"])} changed followed files. Original branch/HEAD and pre-existing untracked files are recorded. Current scientific identity: {data["current_core_code_identity"]}.\n\n'+
        'Full job accounting separates 18 native timed jobs from negligible deterministic tests and the read-only timing audit; every attempt and failure remains in tables/resource_usage.csv and tables/tests.csv. A no-op is not treated as a successful scientific pilot. All scientific assignments have nonempty full demand and actual loading work.\n',encoding='utf-8')
    return dict(converged=len(converged),time_limited=len(censored),safety_stopped=len(rejected))


def front_reports(root,counts,status,gate=None):
    decision='Exact fast-track evaluation pending' if gate is None else gate['decision']
    readme=f'''# N8R2 review package

Status: **{status}**. {counts['converged']} full-demand pilots reached fresh tolerance; {counts['time_limited']} were time-limited; {counts['safety_stopped']} stopped on actual unsafe costs. No numerical stop is relabeled as convergence.

Start with `01_EXECUTIVE_REVIEW.md`, `R2_PILOT_RESULTS.md`, `RUNTIME_ESTIMATE.md`, `PHYSICAL_TIMING_AUDIT.md`, and `FAST_TRACK_GATE.json`. Source/code/run identities, complete coverage, raw signed certificates and failed/unrun rows are in the machine tables. `ALGORITHM_IMPLEMENTATION_NOTES.md` describes the new adapter and solver; `LIMITATIONS_AND_OPEN_ITEMS.md` retains scientific limitations.

Gate decision: **{decision}**. Original R2 start is 2026-09-15T14:03:37Z. R2 has an eight-hour total cap. Conditional combined R2+R3 has the original twelve-hour total cap, ending 2026-09-16T02:03:37Z. R4 remains NOT_AUTHORIZED.

The final ZIP is created only after readable reports, exact snapshot manifests, commit/status evidence and source closure. Its size and SHA-256 belong in the external delivery receipt. This report is not a self-approval or an authorization to extend the campaign.
'''
    (root/'00_READ_FIRST.md').write_text(readme,encoding='utf-8')
    (root/'01_EXECUTIVE_REVIEW.md').write_text(readme.replace('# N8R2 review package','# Executive review')+'''
## Key interpretation

Full-demand s03 FFTT reached RG 0.000726661 in 460.277 seconds with 9 accepted updates. The separate analytic pilot retained a fresh RG 0.005377961 at its TIME_LIMIT after 16 updates and 760.302 seconds. These are real assignment measurements; the 32-OD fixture is not used to extrapolate national timing. The frozen primary neural portfolio stopped on a -141.066-second actual cost after feasible full-demand initialization. Its complete coupled query is retained.

The physical x45 benchmark recorded 3,353.947 seconds of cumulative update work (55.899 minutes). x200 recorded 14,614.275 update seconds (4.060 hours) and 17,724.460 broader elapsed seconds (4.923 hours). x45's later 138.044-second certificate refresh is a separate measurement. Standalone x45 end-to-end time and matched historical hardware remain unknown.

The scientific population covers 99.899865% of original demand, with inherited physical unreachability explicit and no additional active-graph unreachable demand in the final mapping. The solver retains exact source support/weights and all unsuccessful scenarios. Scientific accuracy and global neural safety remain separate from optimization correctness.
''',encoding='utf-8')
    write_json(root/'02_HANDOFF_STATUS.json',dict(status=status,review_status='NOT_EXTERNALLY_REVIEWED',
        counts=counts,next_step_authorized=False,R3_auto_started=False,R4_authorized=False,
        original_R2_start=1789481017,exact_gate_decision=decision))


def plan_report(root,manifest,gate=None):
    rows=[[r['scenario_id'],r['r3_disposition'],r['official_job_id'] or 'Rejected; row retained',
        r['official_run_cap_seconds'] if r['official_run_cap_seconds'] is not None else 'Not executable'] for r in manifest['scenarios']]
    planned=markdown_table(['Scenario','Frozen disposition','Official/canonical run ID','Cap seconds'],rows)
    passed=gate is not None and gate['R3_auto_start_allowed']
    text='# R3 campaign plan\n\n'+('The exact fast-track passed. The user\'s conditional same-Goal authorization permits direct execution under the original twelve-hour total deadline.\n\n' if passed else
        'This is a frozen conditional matrix. If the exact fast-track fails, no R3 science is launched; the alternative below is a proposal requiring review.\n\n')+planned
    text+='''
## Fixed outputs and execution

All nine scenario rows remain present. Completed exact-matching pilots can be reused. A censored pilot keeps its original result and clock; any declared fresh official experiment uses a distinct ID, the identical frozen initialization/solver/graph/demand/objective, and a predeclared cap. All pilot time is already charged to the original Goal. This is not a reset or continuation of an expired pilot measurement.

After assignments, bind the exact result states before metrics. Each admitted scenario receives the full historical 456-OD sample for FIXED_X45 and SOLVED_X45_REFERENCE, with the two 128-row cohorts and old 200-row cohort separate. Preserve safety-rejected and unrun metric rows. Compare only exact common packed and regional flow coordinates. No global lifting, new reference solve, topology/capacity change, support expansion, model reselection, labels, training, optional tighter solve, or R4 task is scheduled.

One numeric job and one worker run at a time. The numeric child-tree cap is 10 GiB with at least 3 GiB system headroom. Combined science admits no work beyond original T0 + 11 hours, retaining at least one hour for final reports/ZIP. A run stops at fresh tolerance, its own cap, resource limit or explicit failure. An activated campaign can return partials and never extends itself to 48 hours.

## Alternative review proposal, capped at 48 hours

If fast-track fails, the unlaunched standalone proposal has a 172,800-second total ceiling, including a 7,200-second finalization reserve. Its immutable matrix contains the same nine rows, with unsafe scenarios retained as rejected. Each executable fresh ordinary-FW assignment has at most 14,400 seconds; there are at most nine, so assignment caps total at most 129,600 seconds. Both metric tracks, source/identity closure and diagnostics have a shared 14,400-second ceiling, leaving at least 21,600 seconds of contingency inside the total. Time-to-tolerance remains unknown for any unsupported trajectory; these are resource ceilings, not promised convergence times. Exact standalone run IDs/start/deadline are to be bound by the later approval; none is started by this proposal.

The next review must bind the final R2 handoff SHA-256. Source weights/support and physical references remain immutable. R4 is not authorized by either route.
'''
    (root/'R3_CAMPAIGN_PLAN.md').write_text(text,encoding='utf-8')


def checkpoint(config,run_id):
    ctx=Context(config,'REPORT',run_id);root=ctx.root;repo=ctx.repo
    if (root/'FAST_TRACK_GATE.json').exists():raise FileExistsError('Frozen gate exists; preserve it and resume its recorded state')
    data=gather(root,repo)
    if len(data['forecasts'])!=9:raise ValueError('Every frozen pilot needs an actual terminal result before gate evaluation')
    preservation=verify_sources(root,repo,data)
    manifest=campaign_manifest(root,repo,data)
    counts=write_r2_reports(root,repo,data,manifest,preservation)
    front_reports(root,counts,'R2_COMPLETE_CHECKPOINT_PENDING_EXACT_GATE')
    plan_report(root,manifest)
    resources=data['resources'];unit=sorted([r for r in resources if r['task']=='unit_tests'],key=lambda r:r['start_epoch'])[-1]
    known_repaired={'B01_s03','T06_control_and_metrics'}
    untriaged=[r['job_id'] for r in resources if r['returncode'] not in (0,None) and r['job_id'] not in known_repaired]
    tests_ok=unit['returncode']==0 and all(r['status'].startswith('PASS') for r in data['tests'] if not r['test_id'].startswith('T0'))
    core=[r for r in data['runs'] if r['phase']=='N8R2']
    identity_ok=all(r['code_hash']==data['current_core_code_identity'] for r in core)
    enough=all((r['accepted_iterations']>=5 or r['status'].startswith('CONVERGED')) for r in core if r['mandatory'])
    statuses={f['scenario']:dict(status=f['gate_status'],evidence=str(root/'runs'/f['job_id']/'RESULT.json'),
        canonical_run_id=f['job_id']) for f in data['forecasts']}
    core_tol=all(f['gate_status'] in ['COMPLETED_MATCHING','FORECAST_SUPPORTED','SAFETY_REJECTED'] and not f.get('high_exceeds_per_run_ceiling',False) for f in data['forecasts'])
    reports=read_json(Path(ctx.c['authority_root'])/'config/output_contract.json')['reports']
    inp=dict(implementation_passed=bool(tests_ok),same_objective_and_oracle_passed=bool(tests_ok and identity_ok),
        coverage_accounted=all(r['invalid_mapping_rows']==0 for r in data['coverage']),source_preserved=preservation['status']=='PASS',
        no_untriaged_shared_defects=not untriaged,untriaged_shared_defects=untriaged,
        full_demand_evidence_sufficient=enough,core_expected_to_reach_tolerance=core_tol,
        pretransition_reports_saved=all((root/p).exists() for p in reports),
        resources_ok=all(r['no_orphan_child'] is True and (r['peak_numeric_child_tree_rss_bytes'] or 0)<10*1024**3 for r in resources),
        frozen_manifest_sha256=sha(root/'CAMPAIGN_MANIFEST.json'),forecast_method='Observed completions or larger supported geometric/power-law fit with >=1.5 margins; full-demand throughput; null if unsupported',
        elapsed_seconds=time.time()-ctx.c['start_epoch'],high_remaining_campaign_seconds=manifest['high_remaining_work_seconds'],
        scenario_statuses=statuses,original_R2_start_epoch=ctx.c['start_epoch'],
        total_deadline_epoch=ctx.c['start_epoch']+43200,finalization_reserve_seconds=3600,
        evidence_verifier='Actual source rehash, current core identities, complete run/state records and scoped tests')
    write_json(root/'FAST_TRACK_GATE_INPUT.json',inp)
    script=Path(ctx.c['authority_root'])/'scripts/n8_budget_gate.py'
    spec=importlib.util.spec_from_file_location('verified_n8_gate',script);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    configgate=read_json(Path(ctx.c['authority_root'])/'config/fast_track_gate.json')
    gate=mod.evaluate(inp,configgate);write_json(root/'FAST_TRACK_GATE.json',gate)
    write_json(root/'receipts/FAST_TRACK_GATE_BINDING.json',dict(helper_path=str(script),helper_sha256=sha(script),
        input_sha256=sha(root/'FAST_TRACK_GATE_INPUT.json'),output_sha256=sha(root/'FAST_TRACK_GATE.json'),
        manifest_sha256=sha(root/'CAMPAIGN_MANIFEST.json')))
    status='R2_COMPLETE_CHECKPOINT_FOR_AUTHORIZED_R3' if gate['R3_auto_start_allowed'] else 'R2_COMPLETE_AWAITING_REVIEW'
    front_reports(root,counts,status,gate);plan_report(root,manifest,gate)
    state=read_json(root/'RUN_STATE.json');state.update(R2_status=status,R3_auto_started=False,R4_authorized=False,
        next_step_authorized=False,gate_decision=gate['decision'],status='READY_FOR_CONDITIONAL_R3' if gate['R3_auto_start_allowed'] else 'WAITING_FOR_EXTERNAL_REVIEW')
    write_json(root/'RUN_STATE.json',state)
    snapshot=root/'R2_CHECKPOINT';snapshot.mkdir(exist_ok=False)
    for name in reports+['FAST_TRACK_GATE.json','FAST_TRACK_GATE_INPUT.json','CAMPAIGN_MANIFEST.json','RUNTIME_FORECAST_DETAIL.json',
                         'FILE_PRESERVATION_REPORT.json','resolved_config.json','INTAKE_RECEIPT.json','RUN_STATE.json']:
        shutil.copy2(root/name,snapshot/name)
    shutil.copytree(root/'tables',snapshot/'tables')
    entries=[dict(path=str(p.relative_to(snapshot)).replace('\\','/'),sha256=sha(p),size_bytes=p.stat().st_size)
        for p in snapshot.rglob('*') if p.is_file()]
    write_json(snapshot/'CHECKPOINT_MANIFEST.json',dict(status=status,files=entries,original_R2_start=ctx.c['start_epoch'],created_epoch=time.time()))
    print(json.dumps(dict(gate=gate,counts=counts,R2_checkpoint=str(snapshot)),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--action',choices=['r2_checkpoint'],required=True);args=p.parse_args()
    checkpoint(args.config,args.run_id)


if __name__=='__main__':raise SystemExit(main())
