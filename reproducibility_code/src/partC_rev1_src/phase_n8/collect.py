"""Read-only collection of actual run evidence into compact machine tables."""
import argparse
import csv
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from .common import Context,read_json,write_json,sha,code_identity
from .forecast import assignment_forecast,METHOD


def records_csv(path,rows,empty_columns=('status',)):
    pd.DataFrame(rows,columns=None if rows else list(empty_columns)).to_csv(path,index=False)


def gather(root,repo):
    root=Path(root);repo=Path(repo)
    definition=read_json(root/'SCENARIO_DEFINITION_FREEZE.json')['scenarios']
    runs=[];history=[];stages=[];safety=[];forecasts=[];identities=[];resources=[];tests=[]
    for job in sorted((root/'receipts').glob('*_JOB.json')):
        j=read_json(job);o=j.get('result',{})
        resources.append(dict(job_id=j['job_id'],task=j['task'],status=j['status'],
            start_epoch=j['start_epoch'],deadline_epoch=j['deadline_epoch'],
            cap_seconds=j.get('cap_seconds',j['deadline_epoch']-j['start_epoch']),
            elapsed_seconds=o.get('elapsed_seconds'),returncode=o.get('returncode'),
            termination_reason=o.get('reason'),peak_numeric_child_tree_rss_bytes=o.get('peak_process_tree_rss'),
            no_orphan_child=o.get('no_orphan_child'),workers=j.get('workers'),
            original_R2_start=j['original_R2_start'],
            R2_native_timed_job=j['task'] in ['build','reconcile_self_od','native_tests','pilot'],
            tiny_fixture_or_read_only_audit=j['task'] in ['unit_tests','timing']))
        if j['task']=='unit_tests':
            text=(root/'logs'/(j['job_id']+'.log')).read_text(errors='replace')
            found=re.search(r'(\d+) passed(?: in ([0-9.]+)s)?',text)
            tests.append(dict(test_id=j['job_id'],status='PASS' if o.get('returncode')==0 else 'FAIL_PRESERVED_HISTORY',
                passed_count=int(found.group(1)) if found else None,elapsed_seconds=o.get('elapsed_seconds'),
                evidence=str(root/'logs'/(j['job_id']+'.log')),
                interpretation='Earlier failed fixture retained; final current suite is determined by latest unit job'))
    for sc in definition:
        runroot=root/'runs'/sc['pilot_job_id'];p=runroot/'RESULT.json'
        if not p.exists():
            runs.append(dict(phase='N8R2',scenario=sc['scenario_id'],job_id=sc['pilot_job_id'],status='UNRUN_OR_CODE_BLOCKED',
                mandatory=sc['mandatory'],reason='No completed result; inspect exact failure receipt'))
            continue
        r=read_json(p);h=read_json(runroot/'HISTORY.json') if (runroot/'HISTORY.json').exists() else []
        st=read_json(runroot/'STAGES.json') if (runroot/'STAGES.json').exists() else []
        inf=read_json(runroot/'INFERENCE_TIMINGS.json')
        f=assignment_forecast(r,h,st,inf);forecasts.append(f)
        runs.append(run_row(r,sc,'N8R2',str(p)))
        identities.append(dict(r['identity'],suite=sc['suite'],
            registry_id=sc['registry_id'],active_regions=r['coverage']['active_regions'],
            regional_coordinates=r['coverage']['regional_links'],packed_links=r['coverage']['packed_links'],
            mapping_sha256=r['coverage']['mapping_sha256'],result_sha256=sha(p)))
        collect_details(runroot,r,'N8R2',history,stages,safety)
    micro=root/'runs/P00_micro_s03/RESULT.json'
    if micro.exists():
        r=read_json(micro);runs.append(run_row(r,dict(mandatory=False),'N8R2_MICRO',str(micro)))
    if (root/'CAMPAIGN_MANIFEST.json').exists():
        m=read_json(root/'CAMPAIGN_MANIFEST.json')
        active=(root/'R3_ACTIVATION.json').exists()
        for sc in m['scenarios']:
            role=sc['r3_disposition'];path=Path(sc['assignment_root'])/'RESULT.json' if sc.get('assignment_root') else None
            if active and path and path.exists():
                r=read_json(path);row=run_row(r,sc,'N8R3',str(path))
                row['reuse_or_new']=role;row['official_experiment_id']=sc['experiment_id']
                runs.append(row)
                if role=='FRESH_OFFICIAL_FW_FROM_FROZEN_INITIAL_RULE':collect_details(path.parent,r,'N8R3',history,stages,safety)
            else:runs.append(dict(phase='N8R3',scenario=sc['scenario_id'],job_id=sc.get('official_job_id'),
                status='SAFETY_REJECTED' if role=='SAFETY_REJECTED' else 'UNRUN_NOT_AUTHORIZED' if not active else 'UNRUN',
                reuse_or_new=role,mandatory=sc['mandatory'],official_experiment_id=sc['experiment_id']))
    coverage=[]
    for sid in ['s02','s03','s04','s05','s06']:
        p=root/'graphs'/sid/'IDENTITY_AND_COVERAGE_SELF_OD_V2.json'
        if not p.exists():p=root/'graphs'/sid/'IDENTITY_AND_COVERAGE.json'
        d=read_json(p)
        for item in d['coverage']:
            coverage.append(dict(**item,graph_identity=d['graph_identity'],demand_identity=d['demand_identity'],
                nodes=d['nodes'],packed_links=d['packed_links'],regional_links=d['regional_links'],
                endpoint_attachment_links=d['endpoint_attachment_links'],active_regions=d['active_regions'],
                invalid_mapping_rows=d['invalid_mapping_rows'],full_original_OD_count=10193770,
                full_original_row_count=41637922,full_original_demand=12667037.704538835,
                coverage_metadata_path=str(p),coverage_metadata_sha256=sha(p)))
    for p in ['native_model_validation.csv','native_derivative_cohorts.csv']:
        table=pd.read_csv(root/'tables'/p)
        column='structural_order_hash_scale_eligibility' if p=='native_model_validation.csv' else 'derivative_status'
        tests.append(dict(test_id=p,status='PASS' if table[column].astype(str).str.startswith('PASS').all() else 'FAIL',
            passed_count=len(table),elapsed_seconds=None,evidence=str(root/'tables'/p)))
    tests.append(dict(test_id='NATIVE_PARITY',status='PASS' if all(x['status']=='PASS_PARITY_AND_DERIVATIVES' for x in read_json(root/'validation/NATIVE_PARITY.json')) else 'FAIL',
        passed_count=5,evidence=str(root/'validation/NATIVE_PARITY.json')))
    tests.append(dict(test_id='PHYSICAL_TIMING_RECONCILIATION',status=read_json(root/'receipts/A01_physical_timing_RESULT.json')['status'],
        evidence=str(root/'validation/PHYSICAL_TIMING_EVIDENCE.json')))
    tests.append(dict(test_id='PHYSICAL_SAMPLE_PRECHECK',status=read_json(root/'validation/PHYSICAL_SAMPLE_PRECHECK_V2.json')['status'],
        evidence=str(root/'validation/PHYSICAL_SAMPLE_PRECHECK_V2.json')))
    source={}
    for p in (root/'receipts').glob('*_SOURCE_BINDINGS.json'):
        for item in read_json(p):
            key=item['native_path']
            if key in source and source[key]['sha256']!=item['sha256']:
                raise ValueError('Conflicting actual-use source hash: '+key)
            source[key]=item
    for p in (root/'receipts').glob('R3*_RESULT.json'):
        for item in read_json(p).get('source_bindings',[]):
            if item['native_path'] in source and source[item['native_path']]['sha256']!=item['sha256']:
                raise ValueError('R3 source hash changed')
            source[item['native_path']]=item
    paths=[];flows=[]
    for p in sorted((root/'conditional_R3').glob('*/PATH_METRICS.csv')) if (root/'conditional_R3').exists() else []:
        paths.extend(pd.read_csv(p).to_dict('records'))
    for p in sorted((root/'conditional_R3').glob('*/FLOW_METRICS.csv')) if (root/'conditional_R3').exists() else []:
        flows.extend(pd.read_csv(p).to_dict('records'))
    done={r['scenario'] for r in paths}
    for sc in definition:
        if sc['scenario_id'] not in done:
            result=next((r for r in runs if r['phase']=='N8R2' and r['scenario']==sc['scenario_id']),{})
            reason='SAFETY_REJECTED' if str(result.get('status','')).startswith('STOP_') else 'NOT_EXECUTED_R3_PENDING_GATE_OR_REVIEW'
            for track in ['FIXED_X45','SOLVED_X45_REFERENCE']:
                for cohort in ['NEW_AUDIT128','NEW_SELECT128','DISCOVERY_OLD200']:
                    paths.append(dict(scenario=sc['scenario_id'],track=track,population=cohort,status=reason,WREL=None,WMEANREL_1S=None))
                flows.append(dict(scenario=sc['scenario_id'],track=track,status=reason,flow_NL1=None))
    tables=dict(run_matrix=runs,convergence_history=history,stage_timings=stages,runtime_forecast=forecasts,
        demand_coverage=coverage,safety_exceptions=safety,tests=tests,source_bindings=list(source.values()),
        resource_usage=resources,path_metrics=paths,flow_metrics=flows,source_registry_graph_identities=identities)
    for name,rows in tables.items():
        flat=[{k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows]
        records_csv(root/'tables'/(name+'.csv'),flat)
    write_json(root/'RUNTIME_FORECAST_DETAIL.json',dict(method=METHOD,scenarios=forecasts))
    return dict(runs=runs,forecasts=forecasts,resources=resources,tests=tests,source_bindings=list(source.values()),
        coverage=coverage,current_core_code_identity=code_identity(repo)[0])


def run_row(r,sc,phase,path):
    cert=r.get('final_certificate') or {}
    return dict(phase=phase,scenario=r['scenario'],suite=r['suite'],job_id=r['job_id'],status=r['stop_reason'],
        mandatory=sc['mandatory'],solver=r['solver'],nominal_tolerance=r['nominal_tolerance'],
        initialization=r['initialization'],accepted_iterations=r['accepted_iterations'],
        elapsed_seconds=r['elapsed_seconds'],run_start_epoch=r['clocks']['run_start_epoch'],
        run_deadline_epoch=r['clocks']['run_deadline_epoch'],original_R2_start=r['clocks']['R2_start_epoch'],
        graph_hash=r['identity']['graph'],model_registry_hash=r['identity']['registry'],demand_hash=r['identity']['demand'],
        code_hash=r['identity']['code'],objective_hash=r['identity']['objective'],state_hash=r['state_hash'],
        final_certificate_fresh=r['final_certificate_fresh'],final_certificate_status=r['final_certificate_status'],
        B=cert.get('objective'),TSTT=cert.get('TSTT'),SPTT=cert.get('SPTT'),raw_signed_gap=cert.get('raw_signed_gap'),
        RG_SPTT=cert.get('RG_SPTT'),represented_demand=r['represented_demand'],
        population='FROZEN_MICRO_32' if phase=='N8R2_MICRO' else 'FULL_GRAPH_REPRESENTED_DEMAND',
        full_denominator_coverage_fraction=None if phase=='N8R2_MICRO' else .99899865094959,
        result_path=path,result_sha256=sha(path))


def collect_details(root,result,phase,history,stages,safety):
    shared=dict(phase=phase,scenario=result['scenario'],job_id=result['job_id'])
    if (root/'HISTORY.json').exists():
        for row in read_json(root/'HISTORY.json'):
            history.append(dict(**shared,**row))
            for column,label in [('aon_seconds','FULL_AON_CERTIFICATE'),('value_gradient_seconds','ACCEPTED_STATE_VALUE_GRAD'),
                                 ('line_search_seconds','COMPOSITE_LINE_SEARCH'),('checkpoint_seconds','CERTIFIED_CHECKPOINT_IO')]:
                if row.get(column) is not None:stages.append(dict(**shared,iteration=row['iteration'],stage=label,seconds=row[column]))
            if row.get('below_fftt_coordinates',0) or row.get('packed_extrapolated_links',0):
                safety.append(dict(**shared,kind='FINITE_DOMAIN_OR_FLOOR_DIAGNOSTIC',iteration=row['iteration'],
                    **{k:v for k,v in row.items() if k in ['state_hash','below_fftt_coordinates','maximum_below_fftt_fraction',
                      'maximum_below_fftt_seconds','below_fftt_traversal_flow','flow_weighted_below_fftt_deficit',
                      'maximum_native_scale_ratio','packed_extrapolated_links','packed_max_domain_ratio']}))
    if (root/'STAGES.json').exists():
        for row in read_json(root/'STAGES.json'):stages.append(dict(**shared,**row))
    for i,row in enumerate(read_json(root/'INFERENCE_TIMINGS.json')):
        stages.append(dict(**shared,stage=row['operation'],call_index=i,seconds=row['seconds']))
    if (root/'SAFETY_STOP.json').exists():
        safety.append(dict(**shared,kind=result['stop_reason'],**read_json(root/'SAFETY_STOP.json')))
    if (root/'FIXED_X45_INFERENCE.json').exists():
        row=read_json(root/'FIXED_X45_INFERENCE.json')
        safety.append(dict(**shared,kind='FIXED_X45_DIAGNOSTIC',**row['diagnostic']))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--run-id',required=True);p.add_argument('--collection-id',required=True)
    args=p.parse_args();ctx=Context(args.config,'REPORT',args.run_id)
    data=gather(ctx.root,ctx.repo)
    write_json(ctx.root/'receipts'/(args.collection_id+'.json'),dict(status='COLLECTED_ACTUAL_EVIDENCE',
        original_R2_start=ctx.c['start_epoch'],epoch=time.time(),run_rows=len(data['runs']),
        forecast_rows=len(data['forecasts']),unique_followed_sources=len(data['source_bindings'])))
    print(dict(run_rows=len(data['runs']),forecast_rows=len(data['forecasts']),sources=len(data['source_bindings'])))


if __name__=='__main__':raise SystemExit(main())
