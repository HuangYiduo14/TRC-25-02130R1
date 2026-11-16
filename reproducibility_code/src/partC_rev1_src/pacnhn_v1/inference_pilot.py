"""Bounded sequential inference on feasible, frozen whole-AC baseline states.

No work occurs on import. This is a Stage 3 measurement, not an assignment or
time-to-gap forecast. The measured calls include the fixed B cost field and all
regional models. Initialization/load/validation are separate wall-time scopes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import math
import numpy as np

from .control import Context,read_json,write_json,sha,array_hash

FAMILIES=('FFTT','POWER4','ICNN','XGBOOST')
SCENARIOS=('open','i95')
RETRYABLE_STATUSES=('RESOURCE_LIMIT','TIME_LIMIT')


def _retry_id(value):
    if not isinstance(value,str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',value) is None:
        raise ValueError('Explicit safe versioned retry ID required')
    return value


def _case_key(row):
    key=(row['suite_id'],row['family'],row['scenario'])
    if key[0] not in ('s02','s03','s04','s05','s06') or key[1] not in FAMILIES or key[2] not in SCENARIOS:
        raise ValueError('Invalid inference case identity')
    return key


def _inside(root,path):
    path=Path(path).resolve();root=Path(root).resolve()
    if not path.is_relative_to(root):raise ValueError('Inference artifact escapes its owned directory')
    return path


def _metadata(path,bindings,expected=None):
    """Authenticate the same bytes that are decoded, without model/array reads."""
    path=Path(path).resolve();raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if expected is not None and digest!=expected:raise ValueError('Inference metadata hash differs: '+str(path))
    if str(path) in bindings and bindings[str(path)]!=digest:
        raise ValueError('Inference metadata changed during validation: '+str(path))
    bindings[str(path)]=digest
    return json.loads(raw)


def _check_bindings(bindings):
    for path,digest in bindings.items():
        if sha(path)!=digest:raise ValueError('Inference predecessor/input changed: '+path)


def _validate_case(root,target,row,registry_sha,clock,support_sha,bindings,*,retry=None):
    """Verify one persisted outcome and its exact local intent/optional receipt."""
    target=_inside(root,target);target=_inside(root/'inference_pilots',target);key=_case_key(row)
    result_path=target/'RESULT.json'
    if _metadata(result_path,bindings)!=row:raise ValueError('Inference ledger differs from exact case outcome')
    intent=_metadata(target/'INTENT.json',bindings,row['intent_sha256'])
    if (_case_key(intent)!=key or row.get('registry_sha256')!=registry_sha
            or intent['input_versions']['registry_sha256']!=registry_sha
            or intent['input_versions']['support_sha256']!=support_sha or intent['stage3_clock']!=clock):
        raise ValueError('Inference case differs from frozen registry/support/Stage3 clock')
    reference_path=root/'references'/key[2]/'RESULT.json'
    _metadata(reference_path,bindings,intent['input_versions']['reference_result_sha256'])
    if retry is not None and intent.get('retry')!=retry:
        raise ValueError('Inference case retry lineage differs')
    if retry is None and intent.get('retry') is not None:
        raise ValueError('Original inference case unexpectedly claims a retry')
    if row['status']=='PASS':
        receipt_path=target/'INFERENCE_WALL.json'
        if _inside(target,row['inference_receipt'])!=receipt_path:
            raise ValueError('Inference receipt points outside its exact case')
        receipt=_metadata(receipt_path,bindings,row['inference_receipt_sha256'])
        if (receipt.get('schema')!='PACNHN_INFERENCE_WALL_V1' or receipt.get('status')!='PASS'
                or _case_key(receipt)!=key or receipt['registry_sha256']!=registry_sha
                or receipt['intent_sha256']!=row['intent_sha256'] or receipt.get('full_model') is not True
                or receipt.get('includes_fixed_B') is not True or receipt.get('scope')!='WHOLE_AC_COST_FIELD'
                or receipt.get('measurement')!='sequential_calls'
                or not 3<=receipt['repetitions']<=5
                or receipt['repetitions']!=row['repetitions']
                or receipt['repetitions']!=intent['requested_repetitions']
                or not math.isfinite(receipt['seconds']) or receipt['seconds']<0):
            raise ValueError('Invalid same-case whole-field inference receipt')
    elif 'inference_receipt' in row or (target/'INFERENCE_WALL.json').exists():
        raise ValueError('Non-PASS inference outcome may not expose a successful receipt')
    return dict(result_path=str(result_path),result_sha256=bindings[str(result_path.resolve())],
                intent_path=str(target/'INTENT.json'),intent_sha256=row['intent_sha256'])


def resolve_inference_results(root, *,registry_sha256):
    """Resolve complete, authenticated retry batches; never promote partial work.

    The original ledger is immutable. Returned results add only resolution
    provenance; history keeps all prior dispositions. Receipt paths are verified
    local files. No registry parsing, model loading, TEST or inference occurs.
    """
    root=Path(root).resolve();base=_inside(root,root/'inference_pilots');bindings={}
    clock=_metadata(root/'STAGE3_CLOCK.json',bindings)
    _metadata(root/'SUPPORT_FREEZE.json',bindings);support_sha=bindings[str((root/'SUPPORT_FREEZE.json').resolve())]
    original_path=base/'RESULTS.json';original=_metadata(original_path,bindings)
    original_sha=bindings[str(original_path.resolve())];current={};history=[]
    for row in original:
        key=_case_key(row)
        if key in current:raise ValueError('Duplicate original inference case')
        if row['status']=='UNRUN_STAGE_TIME_LIMIT':
            info=dict(result_path=str(original_path),result_sha256=original_sha)
        else:
            info=_validate_case(root,base.joinpath(*key),row,registry_sha256,clock,support_sha,bindings)
        current[key]=dict(row, resolved_attempt_id='ORIGINAL',resolved_result_path=info['result_path'],
                         resolved_result_sha256=info['result_sha256'])
        history.append(dict(case=list(key),attempt_id='ORIGINAL',status=row['status'],**info))
    intents=[];incomplete=[];seen_sequences=set()
    for directory in sorted((base/'retries').glob('*')):
        if not directory.is_dir():raise ValueError('Unexpected file in inference retry inventory')
        _inside(base,directory);_retry_id(directory.name)
        if not (directory/'RETRY_INTENT.json').exists():
            incomplete.append(dict(retry_id=directory.name,status='INCOMPLETE_NO_RETRY_INTENT'));continue
        intent=_metadata(directory/'RETRY_INTENT.json',bindings)
        seq=intent['sequence']
        if not isinstance(seq,int) or isinstance(seq,bool) or seq<1 or seq in seen_sequences:
            raise ValueError('Duplicate/invalid inference retry sequence')
        seen_sequences.add(seq);intents.append((seq,directory,intent))
    for sequence,directory,intent in sorted(intents):
        if (intent.get('schema')!='PACNHN_INFERENCE_RETRY_V1' or intent['retry_id']!=directory.name
                or intent['original_results_sha256']!=original_sha or intent['registry_sha256']!=registry_sha256
                or intent['stage3_clock']!=clock or intent['stage3_clock_sha256']!=bindings[str((root/'STAGE3_CLOCK.json').resolve())]
                or intent['support_sha256']!=support_sha):
            raise ValueError('Retry changes frozen original inputs/clock')
        preflight_path=directory/'PREFLIGHT.json'
        preflight=_metadata(preflight_path,bindings,intent['preflight_sha256'])
        if (preflight.get('status')!='PASS' or preflight['config_sha256']!=intent['config_sha256']
                or preflight['component_snapshots']['inference']!=intent['source_snapshot']):
            raise ValueError('Retry lacks its exact passing technical gate')
        requested={};prior_keys=set()
        for prior in intent['predecessors']:
            key=_case_key(prior)
            if key in prior_keys or key not in current or key[:2]!=('s03','XGBOOST'):
                raise ValueError('Invalid/duplicate XGBoost retry predecessor')
            prior_keys.add(key);previous=current[key]
            if (previous['status'] not in RETRYABLE_STATUSES
                    or prior['result_path']!=previous['resolved_result_path']
                    or prior['result_sha256']!=previous['resolved_result_sha256']
                    or prior['attempt_id']!=previous['resolved_attempt_id']):
                raise ValueError('Retry predecessor is stale, successful or changed')
            requested[key]=prior
        if not requested:raise ValueError('Empty retry batch')
        ledger_path=directory/'RESULTS.json';rows=_metadata(ledger_path,bindings) if ledger_path.exists() else []
        by_key={_case_key(row):row for row in rows}
        if len(by_key)!=len(rows) or not set(by_key)<=set(requested):raise ValueError('Retry ledger case inventory differs')
        completion_path=directory/'COMPLETION.json';complete=completion_path.exists()
        # Also authenticate a finished case written immediately before a failed
        # ledger update. Its existence never makes an incomplete batch eligible.
        available={}
        for key,prior in requested.items():
            target=directory.joinpath(*key);path=target/'RESULT.json'
            if not path.exists():
                if key in by_key:raise ValueError('Retry ledger points to an absent result')
                continue
            row=by_key.get(key) or _metadata(path,bindings)
            retry=dict(retry_id=directory.name,batch_intent_sha256=bindings[str((directory/'RETRY_INTENT.json').resolve())],
                       predecessor=prior)
            info=_validate_case(root,target,row,registry_sha256,clock,support_sha,bindings,retry=retry)
            case_intent=_metadata(target/'INTENT.json',bindings)
            if (case_intent['source_snapshot']!=intent['source_snapshot']
                    or case_intent['config_sha256']!=intent['config_sha256']):
                raise ValueError('Retry case source/config differs from admitted batch')
            available[key]=(row,info)
            history.append(dict(case=list(key),attempt_id=directory.name,status=row['status'],
                                batch_complete=complete,**info))
        if not complete:
            incomplete.append(dict(retry_id=directory.name,status='INCOMPLETE_NOT_SELECTED',completed_cases=len(available)))
            continue
        completion=_metadata(completion_path,bindings)
        expected=[dict(case=list(key),**available[key][1]) for key in sorted(available)]
        if (completion.get('schema')!='PACNHN_INFERENCE_RETRY_COMPLETION_V1'
                or completion.get('status')!='COMPLETE' or completion['retry_id']!=directory.name
                or set(by_key)!=set(requested) or set(available)!=set(requested)
                or completion['retry_intent_sha256']!=bindings[str((directory/'RETRY_INTENT.json').resolve())]
                or completion['results_sha256']!=bindings[str(ledger_path.resolve())]
                or completion['outcomes']!=expected):
            raise ValueError('Retry completion is partial or does not bind exact outcomes')
        for key,(row,info) in available.items():
            current[key]=dict(row,resolved_attempt_id=directory.name,resolved_result_path=info['result_path'],
                             resolved_result_sha256=info['result_sha256'])
    _check_bindings(bindings)
    return dict(results=[current[key] for key in sorted(current)],history=history,incomplete_retries=incomplete,
                bindings=bindings,max_sequence=max(seen_sequences,default=0))


def pure_family_available(registry,sid,family):
    rows=[r for r in registry['coverage'] if r['suite_id']==sid and r['family']==family]
    if len(rows)!=1:raise ValueError('Missing/duplicate frozen family coverage')
    if rows[0].get('complete_family') is not True:return False
    models=[r for r in registry['models'] if r['suite_id']==sid and r['family']==family]
    if not models or len({r['region_index'] for r in models})!=len(models):
        raise ValueError('Empty/duplicate purported complete family')
    return all(r['status']=='AVAILABLE' for r in models)


def projected_baseline(p_graph,p_arrays,ac_graph,ac_arrays,p_flow,scenario, *,expected_backbone_count=87582):
    """Exact B and port flows plus actual source-trace regional pair coordinates."""
    if scenario not in SCENARIOS:raise ValueError('Explicit scenario required')
    b=int(ac_arrays['backbone_count']);m=len(ac_arrays['pair_keys'])
    if b!=expected_backbone_count:raise ValueError('Frozen native backbone count differs')
    if not np.array_equal(p_graph.node_ids,ac_graph.node_ids) or not np.array_equal(p_graph.blocked,ac_graph.blocked):
        raise ValueError('Common node/centroid-role identity differs')
    if not np.array_equal(np.flatnonzero(p_arrays['edge_role']==0),np.arange(b)):
        raise ValueError('Physical B coordinate order differs')
    for key in ('edge_ids','tail','head'):
        if not np.array_equal(getattr(p_graph,key)[:b],getattr(ac_graph,key)[:b]):
            raise ValueError('Native B identity differs: '+key)
    for key in ('base','k4','closed_mask'):
        if not np.array_equal(p_arrays[key][:b],ac_arrays[key][:b]):
            raise ValueError('Native B cost/mask differs: '+key)
    ports=np.flatnonzero(p_arrays['edge_role']==2)
    if len(ac_graph.tail)!=b+m+len(ports) or int(ac_arrays['port_count'])!=len(ports):
        raise ValueError('Port/region coordinate dimensions differ')
    for key in ('edge_ids','tail','head'):
        if not np.array_equal(getattr(p_graph,key)[ports],getattr(ac_graph,key)[b+m:]):
            raise ValueError('Port stable-ID/endpoints differ: '+key)
    if np.any(ac_arrays['closed_mask'][b:]):raise ValueError('Closure changed a regional/port coordinate')
    flow=np.asarray(p_flow,float)
    baseline=np.asarray(ac_arrays['baseline_'+scenario],float)
    if flow.shape!=p_graph.tail.shape or baseline.shape!=(m,):raise ValueError('Baseline shape mismatch')
    x=np.concatenate((flow[:b],baseline,flow[ports]))
    if not np.isfinite(x).all() or np.any(x<0):raise ValueError('Invalid projected baseline flow')
    if scenario=='i95' and (np.any(flow[p_arrays['closed_mask']]!=0) or np.any(x[ac_arrays['closed_mask']]!=0)):
        raise ValueError('Masked coordinate has positive baseline flow')
    x.setflags(write=False)
    return x


def measure_calls(objective,flow, *,repetitions,deadline,guard):
    """Sum separately timed complete sequential calls; validation is outside.

    One objective call is not preemptible. Guards run before/after every call,
    and an observed-call reserve prevents knowingly starting an over-budget call.
    On timeout only the completed measurement prefix is retained by the caller.
    """
    if not isinstance(repetitions,int) or not 3<=repetitions<=5:
        raise ValueError('Require 3–5 sequential whole-field calls')
    identity=array_hash({'flow':flow});measurements=[];first_cost=None;first_value=None
    outcome='PASS';error=None
    for index in range(repetitions):
        try:
            guard(deadline=deadline)
            if measurements and time.time()+1.25*max(r['seconds'] for r in measurements)+1>=deadline:
                raise TimeoutError('Insufficient remaining time for another measured full-field call')
            epoch=time.time();tick=time.perf_counter()
            if objective.has_potential:
                value,cost=objective.value_cost(flow)
            else:
                value=None;cost=objective.cost(flow)
            duration=time.perf_counter()-tick;end=time.time()
            cost=np.asarray(cost,float)
            if cost.shape!=flow.shape or not np.isfinite(cost).all() or np.any(cost<0):
                raise ValueError('Invalid whole-field cost shape/value')
            if value is not None and (not math.isfinite(float(value)) or value<0):
                raise ValueError('Invalid whole-field potential')
            if array_hash({'flow':flow})!=identity:raise ValueError('Objective mutated the frozen baseline')
            if first_cost is None:first_cost=cost.copy();first_value=value
            elif not np.array_equal(first_cost,cost) or value!=first_value:
                raise ValueError('Repeated frozen model inference changed predictions')
            measurements.append(dict(call_index=index,start_epoch=epoch,end_epoch=end,seconds=duration,
                cost_min=float(cost.min()),cost_max=float(cost.max()),potential=None if value is None else float(value)))
            guard(deadline=deadline)
        except (TimeoutError,MemoryError) as exc:
            outcome='TIME_LIMIT' if isinstance(exc,TimeoutError) else 'RESOURCE_LIMIT'
            error=dict(type=type(exc).__name__,message=str(exc));break
    return dict(status=outcome,requested_repetitions=repetitions,repetitions=len(measurements),
        seconds=math.fsum(r['seconds'] for r in measurements),calls=measurements,
        flow_identity=identity,cost_identity=None if first_cost is None else array_hash({'cost':first_cost}),
        call_kind='VALUE_COST' if objective.has_potential else 'COST_ONLY',error=error,
        clock_boundary='Sum of complete sequential CompositeObjective calls only; includes B and all regional '
          'models, excludes load, projection, guards and output validation. No batched amortization.',
        budget_policy='Cooperative guards before/after calls; no new call if observed duration reserve cannot fit. '
          'An individual call cannot be preempted; any overrun is explicitly recorded, never a completed in-cap pilot.')


def _load_feasible_baseline(ctx,sid,scenario,graph,od,a,deadline):
    from .topology import load_common
    from .metrics import admit_assignment
    from .oracle import Demand,conservation,load_traces
    from .support import aligned_trace
    ctx.guard(deadline=deadline)
    p_graph,p_od,p_arrays,_=load_common(ctx.root)
    if od.identity!=p_od.identity:raise ValueError('AC/P original demand identity differs')
    source=ctx.root/'references'/scenario
    p_flow,result,serve,bindings=admit_assignment(source,expected_graph_identity=p_graph.identity,
        expected_edge_ids=p_graph.edge_ids)
    for binding in bindings:ctx.bind(binding['path'],'FROZEN_P_INFERENCE_BASELINE',binding['sha256'])
    support=read_json(ctx.root/'SUPPORT_FREEZE.json')
    if result!=support['source_references'][SCENARIOS.index(scenario)]:
        raise ValueError('P baseline result differs from the support construction source')
    served=Demand(od.origins,od.offsets,od.destinations,od.demand,serve)
    if served.identity!=result['OD_identity']:raise ValueError('Baseline served-demand certificate differs')
    coverage=ctx.root/'acnhn'/sid/f'coverage_{scenario}.npz'
    ctx.bind(coverage,'EXACT_AC_REACHABILITY')
    with np.load(coverage,allow_pickle=False) as z:ac_serve=z['serve']
    if not np.array_equal(ac_serve,serve):raise ValueError('Exact AC/P served membership differs')
    traces_path=source/'final_traces.npz';ctx.bind(traces_path,'EXACT_STATE_REGIONAL_TRACES')
    traces=load_traces(traces_path);suite_index=ctx.c['suite_ids'].index(sid)
    if len(traces)!=len(ctx.c['suite_ids']):raise ValueError('Reference trace suite count differs')
    baseline=aligned_trace(a['pair_keys'],traces[suite_index])
    if not np.array_equal(baseline,a['baseline_'+scenario]):
        raise ValueError('Regional baseline is not the exact final P trace')
    x=projected_baseline(p_graph,p_arrays,graph,a,p_flow,scenario)
    balance=served.balance(len(graph.node_ids))
    residual=conservation(graph,x,balance,served.total)
    ctx.guard(deadline=deadline)
    return x,dict(graph_identity=graph.identity,original_OD_identity=od.identity,
        served_OD_identity=served.identity,served_rows=int(serve.sum()),served_demand=served.total,
        reference_state_hash=result['certificate']['state_hash'],reference_status=result['status'],
        reference_RG_SPTT=result['certificate']['RG_SPTT'],conservation=residual,
        baseline_origin='Native P B and exact port IDs; regional pair values are keyed final-state P traces.',
        exact_served_OD_membership=True,baseline_flow_identity=array_hash({'flow':x}))


def _one(ctx,registry,registry_sha,clock,sid,family,scenario, *,repetitions,seconds,retry=None):
    from .assignments import load_ac,action_deadline
    from .gates import component_snapshot
    started=time.time();bound=action_deadline(ctx,'inference')
    deadline=min(started+seconds,clock['deadline_epoch'],bound)
    target=ctx.root/'inference_pilots'/sid/family/scenario
    if retry is not None:
        target=ctx.root/'inference_pilots/retries'/_retry_id(retry['retry_id'])/sid/family/scenario
        target=_inside(ctx.root,target)
    if target.exists():raise FileExistsError('Inference pilot already exists; preserve and review it before any rerun')
    target.mkdir(parents=True)
    intent=dict(schema='PACNHN_INFERENCE_INTENT_V1',phase_id=ctx.c['phase_id'],step_id='assignments',
        run_id=ctx.c['run_id'],config_path=str(ctx.path),config_sha256=sha(ctx.path),suite_id=sid,
        family=family,scenario=scenario,requested_repetitions=repetitions,device='cpu',
        task_list=['LOAD_FROZEN_PURE_FAMILY','PROJECT_CERTIFIED_P_BASELINE','VERIFY_EXACT_SERVED_CONSERVATION',
                   'MEASURE_SEQUENTIAL_WHOLE_AC_COST_FIELD','VERIFY_UNCHANGED_INPUT'],
        expected_outputs=['INTENT.json','RESULT.json','INFERENCE_WALL.json_IF_PASS'],
        acceptance_tests=['frozen_registry_and_support','pure_family_complete','exact_native_B_and_ports',
            'keyed_reference_trace_baseline','conservation','finite_nonnegative_cost','unchanged_input','within_budget'],
        input_versions=dict(ctx.c['input_versions'],registry_sha256=registry_sha,
            support_sha256=sha(ctx.root/'SUPPORT_FREEZE.json'),
            reference_result_sha256=sha(ctx.root/'references'/scenario/'RESULT.json')),
        resources=ctx.c['resources'],source_snapshot=component_snapshot(ctx,'inference'),
        complete_pilot_start_epoch=started,deadline_epoch=deadline,cap_seconds=seconds,
        stage3_clock=clock,scope='WHOLE_AC_COST_FIELD_INCLUDING_FIXED_B',
        full_coverage=True,initialization='EXACT_FEASIBLE_PROJECTED_P_REFERENCE_NO_ASSIGNMENT')
    if retry is not None:intent['retry']=retry
    write_json(target/'INTENT.json',intent)
    timings={};baseline_evidence=None;measurement=None
    try:
        if not pure_family_available(registry,sid,family):
            result=dict(status='UNAVAILABLE_INCOMPLETE_PURE_FAMILY',repetitions=0)
        else:
            ctx.guard(deadline=deadline);tick=time.perf_counter()
            graph,od,a,objective=load_ac(ctx,sid,family,registry,deadline=deadline)
            timings['input_and_model_load_seconds']=time.perf_counter()-tick
            tick=time.perf_counter()
            x,baseline_evidence=_load_feasible_baseline(ctx,sid,scenario,graph,od,a,deadline)
            timings['baseline_projection_validation_seconds']=time.perf_counter()-tick
            measurement=measure_calls(objective,x,repetitions=repetitions,deadline=deadline,guard=ctx.guard)
            result=dict(status=measurement['status'],repetitions=measurement['repetitions'])
    except (TimeoutError,MemoryError) as exc:
        result=dict(status='TIME_LIMIT' if isinstance(exc,TimeoutError) else 'RESOURCE_LIMIT',repetitions=0,
            error_type=type(exc).__name__,error=str(exc))
    except Exception as exc:
        write_json(target/'RESULT.json',dict(status='FAIL',error_type=type(exc).__name__,error=str(exc),
            complete_pilot_start_epoch=started,deadline_epoch=deadline,elapsed_seconds=time.time()-started,
            timings=timings,baseline=baseline_evidence,suite_id=sid,family=family,scenario=scenario,
            registry_sha256=registry_sha,intent_sha256=sha(target/'INTENT.json')))
        raise
    result.update(suite_id=sid,family=family,scenario=scenario,timings=timings,baseline=baseline_evidence,
        measurement=measurement,complete_pilot_start_epoch=started,deadline_epoch=deadline,
        elapsed_seconds=time.time()-started,registry_sha256=registry_sha,intent_sha256=sha(target/'INTENT.json'))
    if result['status']=='PASS':
        if measurement['repetitions']!=repetitions or time.time()>deadline:
            result['status']='TIME_LIMIT'
        else:
            receipt=dict(schema='PACNHN_INFERENCE_WALL_V1',status='PASS',full_model=True,
                measurement='sequential_calls',family=family,suite_id=sid,scenario=scenario,device='cpu',
                repetitions=repetitions,seconds=measurement['seconds'],registry_sha256=registry_sha,
                scope='WHOLE_AC_COST_FIELD',includes_fixed_B=True,call_kind=measurement['call_kind'],
                load_seconds=timings['input_and_model_load_seconds'],
                baseline_validation_seconds=timings['baseline_projection_validation_seconds'],
                complete_pilot_seconds=result['elapsed_seconds'],complete_pilot_start_epoch=started,
                deadline_epoch=deadline,cap_seconds=seconds,baseline=baseline_evidence,calls=measurement['calls'],
                per_call_seconds=[r['seconds'] for r in measurement['calls']],
                call_boundary=measurement['clock_boundary'],budget_policy=measurement['budget_policy'],
                inference_device_policy='Current assignment loader returns portable NumPy/SciPy ICNN/POWER4 and CPU XGBoost.',
                source_bindings=list(ctx.bindings.values()),intent_sha256=result['intent_sha256'],
                estimator_scope='Measured whole-field cost latency at one feasible baseline; no future assignment convergence claim.')
            write_json(target/'INFERENCE_WALL.json',receipt)
            result['inference_receipt']=str(target/'INFERENCE_WALL.json')
            result['inference_receipt_sha256']=sha(target/'INFERENCE_WALL.json')
    write_json(target/'RESULT.json',result);ctx.flush_bindings()
    return result


def _run_retry(ctx,registry,registry_sha,clock,*,suite_ids,scenarios,families,repetitions,seconds,
               retry_id,retry_reason,preflight):
    """One new immutable batch; original ledgers and all failed attempts stay."""
    from .gates import component_snapshot
    retry_id=_retry_id(retry_id)
    if tuple(suite_ids)!=('s03',) or tuple(families)!=('XGBOOST',):
        raise ValueError('Versioned retry scope is s03 XGBOOST only')
    if not isinstance(retry_reason,str) or not retry_reason.strip():raise ValueError('Explicit retry reason required')
    target=_inside(ctx.root,ctx.root/'inference_pilots/retries'/retry_id)
    if target.exists():raise FileExistsError('Retry ID already exists; every attempt must remain preserved')
    resolved=resolve_inference_results(ctx.root,registry_sha256=registry_sha)
    current={_case_key(row):row for row in resolved['results']};predecessors=[]
    for scenario in scenarios:
        key=('s03','XGBOOST',scenario);prior=current.get(key)
        if prior is None or prior['status'] not in RETRYABLE_STATUSES:
            raise ValueError('Retry requires an existing RESOURCE_LIMIT/TIME_LIMIT same-case predecessor')
        predecessors.append(dict(suite_id=key[0],family=key[1],scenario=key[2],
            attempt_id=prior['resolved_attempt_id'],result_path=prior['resolved_result_path'],
            result_sha256=prior['resolved_result_sha256']))
    ctx.guard(deadline=clock['deadline_epoch'])
    preflight_path=ctx.root/'validation/PREFLIGHT_LATEST.json';raw=preflight_path.read_bytes()
    if json.loads(raw)!=preflight:raise ValueError('Passing inference preflight changed before retry')
    sources=component_snapshot(ctx,'inference')
    if (preflight.get('status')!='PASS' or preflight['config_sha256']!=sha(ctx.path)
            or preflight['component_snapshots']['inference']!=sources):
        raise ValueError('Retry requires fresh exact inference source/config preflight')
    target.mkdir(parents=True,exist_ok=False)
    with (target/'PREFLIGHT.json').open('xb') as stream:stream.write(raw)
    intent=dict(schema='PACNHN_INFERENCE_RETRY_V1',phase_id=ctx.c['phase_id'],run_id=ctx.c['run_id'],
        retry_id=retry_id,sequence=resolved['max_sequence']+1,reason=retry_reason.strip(),
        started_epoch=time.time(),registry_sha256=registry_sha,config_sha256=sha(ctx.path),
        original_results_sha256=resolved['bindings'][str((ctx.root/'inference_pilots/RESULTS.json').resolve())],
        stage3_clock=clock,stage3_clock_sha256=sha(ctx.root/'STAGE3_CLOCK.json'),
        support_sha256=sha(ctx.root/'SUPPORT_FREEZE.json'),predecessors=predecessors,
        source_snapshot=sources,preflight_sha256=hashlib.sha256(raw).hexdigest(),
        requested_repetitions=repetitions,per_pilot_seconds=seconds,
        policy='Exact frozen inputs; RESOURCE_LIMIT/TIME_LIMIT retry only; no selection, training or TEST; '
               'whole batch eligible only after immutable completion; original Stage3 clock is unchanged.')
    write_json(target/'RETRY_INTENT.json',intent);intent_sha=sha(target/'RETRY_INTENT.json');rows=[]
    try:
        for prior in predecessors:
            ctx.guard(deadline=clock['deadline_epoch'])
            retry=dict(retry_id=retry_id,batch_intent_sha256=intent_sha,predecessor=prior)
            row=_one(ctx,registry,registry_sha,clock,prior['suite_id'],prior['family'],prior['scenario'],
                     repetitions=repetitions,seconds=seconds,retry=retry)
            rows.append(row);write_json(target/'RESULTS.json',rows)
        _check_bindings(resolved['bindings'])
        if component_snapshot(ctx,'inference')!=sources or sha(ctx.path)!=intent['config_sha256']:
            raise ValueError('Retry source/config changed during measurement')
        outcomes=[]
        for row in rows:
            case=target.joinpath(*_case_key(row))
            outcomes.append(dict(case=list(_case_key(row)),result_path=str((case/'RESULT.json').resolve()),
                result_sha256=sha(case/'RESULT.json'),intent_path=str((case/'INTENT.json').resolve()),
                intent_sha256=sha(case/'INTENT.json')))
        # Validate all emitted cases while the batch is still ineligible. A
        # metadata failure must not leave an apparently committed retry.
        resolve_inference_results(ctx.root,registry_sha256=registry_sha)
        write_json(target/'COMPLETION.json',dict(schema='PACNHN_INFERENCE_RETRY_COMPLETION_V1',status='COMPLETE',
            retry_id=retry_id,retry_intent_sha256=intent_sha,results_sha256=sha(target/'RESULTS.json'),
            outcomes=sorted(outcomes,key=lambda r:r['case']),completed_epoch=time.time()))
        # The public resolver independently authenticates completion and refuses
        # promotion if any input/outcome/receipt is partial or inconsistent.
        resolve_inference_results(ctx.root,registry_sha256=registry_sha)
        return rows
    except BaseException as exc:
        write_json(target/'FAILURE.json',dict(status='INCOMPLETE_RETRY_NOT_SELECTED',retry_id=retry_id,
            error_type=type(exc).__name__,error=str(exc),completed_cases=len(rows),recorded_epoch=time.time(),
            original_stage3_clock_sha256=intent['stage3_clock_sha256']))
        raise


def run(ctx, *,suite_ids=('s03',),scenarios=SCENARIOS,families=FAMILIES,repetitions=3,per_pilot_seconds=180.,
        retry_id=None,retry_reason=None):
    from .assignments import stage_clock,validate_gates
    from .gates import require_component
    if not suite_ids or not scenarios or not families or len(set(suite_ids))!=len(suite_ids):
        raise ValueError('Explicit nonempty, unique pilot selection required')
    if any(s not in ctx.c['suite_ids'] or s=='s01' for s in suite_ids) or any(s not in SCENARIOS for s in scenarios):
        raise ValueError('Unapproved suite/scenario')
    if any(f not in FAMILIES for f in families) or len(set(families))!=len(families) or len(set(scenarios))!=len(scenarios):
        raise ValueError('Unknown/duplicate inference cases')
    if not isinstance(repetitions,int) or not 3<=repetitions<=5 or not math.isfinite(per_pilot_seconds) or not 0<per_pilot_seconds<=180:
        raise ValueError('Require 3–5 calls and a positive at-most-180-second pilot cap')
    if retry_id is not None:
        _retry_id(retry_id)
        if not (ctx.root/'STAGE3_CLOCK.json').is_file():raise ValueError('Retry requires the original persisted Stage3 clock')
        if tuple(suite_ids)!=('s03',) or tuple(families)!=('XGBOOST',):raise ValueError('Versioned retry scope is s03 XGBOOST only')
    elif retry_reason is not None:raise ValueError('Retry reason requires an explicit versioned retry ID')
    clock=stage_clock(ctx,action='inference')  # Validation and every pilot consume this same Stage 3 clock.
    preflight=require_component(ctx,'inference');registry=validate_gates(ctx,deadline=clock['deadline_epoch'])
    registry_sha=sha(ctx.root/'training/REGISTRY_FREEZE.json')
    if retry_id is not None:
        return _run_retry(ctx,registry,registry_sha,clock,suite_ids=suite_ids,scenarios=scenarios,families=families,
                          repetitions=repetitions,seconds=per_pilot_seconds,retry_id=retry_id,
                          retry_reason=retry_reason,preflight=preflight)
    index_path=ctx.root/'inference_pilots/RESULTS.json'
    rows=read_json(index_path) if index_path.exists() else []
    old_keys={(r['suite_id'],r['family'],r['scenario']) for r in rows}
    if len(old_keys)!=len(rows):raise ValueError('Existing inference ledger contains duplicate cases')
    requested={(sid,family,scenario) for sid in suite_ids for family in families for scenario in scenarios}
    if old_keys & requested:raise ValueError('Requested inference case is already recorded; preserve its measurement')
    new_rows=[]
    for sid in suite_ids:
        for family in families:
            for scenario in scenarios:
                if time.time()>=clock['deadline_epoch']:
                    row=dict(suite_id=sid,family=family,scenario=scenario,status='UNRUN_STAGE_TIME_LIMIT',repetitions=0)
                    rows.append(row);new_rows.append(row)
                    continue
                row=_one(ctx,registry,registry_sha,clock,sid,family,scenario,
                    repetitions=repetitions,seconds=per_pilot_seconds)
                rows.append(row);new_rows.append(row)
                write_json(index_path,rows)
    write_json(index_path,rows)
    return new_rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--suite-id',action='append',required=True)
    parser.add_argument('--scenario',action='append',choices=SCENARIOS)
    parser.add_argument('--family',action='append',choices=FAMILIES)
    parser.add_argument('--repetitions',type=int,default=3);parser.add_argument('--seconds',type=float,default=180.)
    parser.add_argument('--retry-id');parser.add_argument('--retry-reason')
    args=parser.parse_args()
    run(Context(args.config,step_id='assignments'),suite_ids=args.suite_id,scenarios=args.scenario or SCENARIOS,
        families=args.family or FAMILIES,repetitions=args.repetitions,per_pilot_seconds=args.seconds,
        retry_id=args.retry_id,retry_reason=args.retry_reason)
