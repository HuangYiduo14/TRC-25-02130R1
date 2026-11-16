"""Bounded Stage 3 queries against the frozen P graph and regional registry."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import math
import numpy as np
import psutil
from .control import Context,read_json,write_json,sha,array_hash
from .oracle import Network,Demand
from .solver import PolynomialObjective,solve
from .objective import CompositeObjective

def load_ac(ctx,sid,family,registry, *,deadline=None):
    from .topology import load_common
    from .models import load_model,ICNN_KIND,POWER_KIND,XGB_KIND
    ctx.guard(deadline=deadline)
    _,od,common,_=load_common(ctx.root)
    ctx.guard(deadline=deadline)
    root=ctx.root/'acnhn'/sid
    report=read_json(root/'REPORT.json')
    support=read_json(ctx.root/'SUPPORT_FREEZE.json')
    frozen=[r for r in support['suites'] if r['suite_id']==sid]
    if support['status']!='PASS' or len(frozen)!=1 or frozen[0]!=report:
        raise ValueError('AC report differs from the witness-bound support freeze')
    ctx.bind(ctx.root/'SUPPORT_FREEZE.json','FROZEN_NATIVE_REGIONAL_SUPPORT')
    ctx.bind(root/'graph.npz','FROZEN_AC_GRAPH',report['graph_sha256'])
    with np.load(root/'graph.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    graph=Network(a['tail'],a['head'],a['blocked'],a['edge_ids'],a['node_ids'])
    if graph.identity!=report['graph_identity'] or od.identity!=report['OD_identity']:raise ValueError('AC graph/OD freeze differs')
    bcount=int(a['backbone_count']);m=len(a['pair_keys']);blocks=[]
    if bcount!=87582 or report['backbone_count']!=bcount:raise ValueError('Common B count differs')
    for key in ['tail','head','edge_ids','base','k4','closed_mask']:
        if not np.array_equal(a[key][:bcount],common[key][:bcount]):raise ValueError('AC/P fixed B differs: '+key)
    del common
    if family!='FFTT':
        records=[r for r in registry['models'] if r['suite_id']==sid and r['family']==family]
        found={r['region_index']:r for r in records}
        needed=np.unique(a['pair_region'])
        if len(found)!=len(records) or set(found)!=set(needed):raise ValueError('Pure family has missing/duplicate regional models')
        if family=='XGBOOST':
            forecast=registry['XGB_per_suite_memory_forecast'][sid]['forecast_resident_booster_bytes']
            reserve=registry['XGB_memory_budget']['graph_runtime_reserve_bytes']
            available=psutil.virtual_memory().available-ctx.c['resources']['minimum_available_bytes']
            tree_remaining=ctx.c['resources']['tree_rss_bytes']-psutil.Process().memory_info().rss
            if forecast+reserve>min(available,tree_remaining):
                raise MemoryError('Frozen complete XGBoost suite forecast exceeds currently available guarded RAM')
        for region in needed:
            ctx.guard(deadline=deadline)
            record=found[int(region)]
            if record['status']!='AVAILABLE':raise ValueError('Regional model unavailable')
            positions=np.flatnonzero(a['pair_region']==region)
            pair_identity=array_hash({'pair_keys':a['pair_keys'][positions]})
            if pair_identity!=record['pair_keys_identity']:raise ValueError('Regional model coordinate order mismatch')
            region_meta=read_json(ctx.root/'regions'/sid/f'{region:06d}'/'REGION.json')
            if record['graph_sha256']!=region_meta['sha256']:raise ValueError('Regional model uses another detailed graph')
            ctx.bind(record['model_path'],'FROZEN_'+family+'_COST_MODEL',record['model_sha256'])
            for path,digest in record['artifact_hashes'].items():
                if sha(path)!=digest:raise ValueError('Model artifact identity differs')
            model=load_model(record['model_path'])
            expected_kind={'ICNN':ICNN_KIND,'POWER4':POWER_KIND,'XGBOOST':XGB_KIND}[family]
            if model.kind!=expected_kind or record['model_kind']!=expected_kind:
                raise ValueError('Registry/serialized model does not implement the named family')
            if not np.array_equal(model.c0,a['c0'][positions]):raise ValueError('Regional baseline cost/order differs')
            blocks.append((bcount+positions,model))
            ctx.guard(deadline=deadline)
    obj=CompositeObjective(a['base'],a['k4'],np.arange(bcount,bcount+m),blocks,family=family)
    return graph,od,a,obj

def matrix():
    rows=[]
    for method in ['CFW','FW','MSA']:
        for scenario in ['open','i95']:
            rows.append(dict(case_id=f'P_{method}_{scenario}',network='P',suite_id=None,family=None,solver=method,scenario=scenario,priority='PRIMARY'))
    for family in ['FFTT','POWER4','XGBOOST','ICNN']:
        for scenario in ['open','i95']:
            rows.append(dict(case_id=f'AC_s03_{family}_{scenario}',network='AC',suite_id='s03',family=family,
                solver='MSA' if family=='XGBOOST' else 'FW',scenario=scenario,priority='PRIMARY'))
    return rows

def validate_gates(ctx, *,deadline=None):
    from .registry_projection import read_stage3_registry
    guard=getattr(ctx,'guard',lambda **kwargs:None)
    guard(deadline=deadline)
    support=read_json(ctx.root/'SUPPORT_FREEZE.json');witness=read_json(ctx.root/'COMPLETE_WITNESS_AUDIT.json')
    if support['status']!='PASS' or witness['status']!='PASS' or witness['support_freeze_sha256']!=sha(ctx.root/'SUPPORT_FREEZE.json'):
        raise ValueError('Topology and complete witness gates required')
    registry_path=ctx.root/'training/REGISTRY_FREEZE.json'
    finished_path=ctx.root/'training/RESULT.json';finished=read_json(finished_path)
    if finished['status']!='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE':
        raise ValueError('Completed training stage is not bound to this frozen registry')
    guard(deadline=deadline)
    registry=read_stage3_registry(registry_path,finished['registry_sha256'],
        guard=lambda:guard(deadline=deadline))
    guard(deadline=deadline)
    if registry['schema']!='PACNHN_REGISTRY_V1' or registry['support_freeze_sha256']!=sha(ctx.root/'SUPPORT_FREEZE.json'):
        raise ValueError('Regional registry is not frozen against current support')
    if finished['registry_sha256']!=sha(registry_path):
        raise ValueError('Completed training stage is not bound to this frozen registry')
    test_authorization=ctx.root/'training/TEST_OPENING_AUTHORIZATION.json'
    dispositions=[p for p in [ctx.root/'training/FINAL_TEST_SUMMARY.json',ctx.root/'training/FINAL_TEST_TIMEOUT.json'] if p.exists()]
    if not dispositions:
        raise ValueError('Final TEST completion or bounded-timeout disposition is required')
    for disposition in dispositions:
        if read_json(disposition).get('registry_sha256')!=sha(registry_path):
            raise ValueError('Final TEST disposition is not bound to this frozen registry')
        ctx.bind(disposition,'FINAL_BLIND_TEST_DISPOSITION')
    if test_authorization.exists():
        if read_json(test_authorization)['registry_sha256']!=sha(registry_path):
            raise ValueError('Final TEST selection freeze identity differs')
        ctx.bind(test_authorization,'BLIND_TEST_SELECTION_FREEZE')
    else:
        timeout=ctx.root/'training/FINAL_TEST_TIMEOUT.json'
        if (ctx.root/'training/FINAL_TEST_SUMMARY.json').exists() or not timeout.exists() or read_json(timeout).get('TEST_opening_authorization_exists') is not False:
            raise ValueError('TEST authorization is absent without an explicit never-opened timeout')
    ctx.bind(finished_path,'COMPLETED_TRAINING_STAGE')
    ctx.bind(registry_path,'FROZEN_REGIONAL_REGISTRY')
    guard(deadline=deadline)
    return registry

def _admit_action(ctx,action):
    """Use the Context's authenticated continuation without rehashing bulk inputs."""
    if action not in ('assignments','inference'):
        raise ValueError('Stage3 requires an explicit assignment or inference action')
    admit=getattr(ctx,'require_action',None)
    if admit is not None:return admit(action)
    # Lightweight legacy test contexts have no Context methods. A configured
    # continuation still goes through the exact same authentication helper.
    from .continuation import validate_continuation
    return validate_continuation(ctx.c,action=action)


def action_deadline(ctx,action):
    continuation=_admit_action(ctx,action)
    return (continuation['science_deadline_epoch'] if continuation is not None
        else ctx.c['hard_deadline_epoch'])


def run_case(ctx,row,registry,stage_deadline):
    from .topology import load_common
    from .gates import component_snapshot
    start=time.time();bound=action_deadline(ctx,'assignments')
    deadline=min(start+ctx.c['assignment']['per_query_seconds'],stage_deadline,bound)
    case_id=row['case_id'];target=ctx.root/'assignments'/case_id
    if (target/'RESULT.json').exists():
        existing=read_json(target/'RESULT.json')
        if not existing.get('case',{}).get('case_id')==case_id:raise ValueError('Existing assignment identity differs')
        return existing
    if target.exists():raise RuntimeError('Interrupted assignment needs explicit checkpoint repair, not restart')
    loading_intent=dict(**row,phase_id=ctx.c['phase_id'],step_id='assignments',run_id=ctx.c['run_id'],
        tasks=['LOAD_FROZEN_INPUTS','FEASIBLE_INITIALIZATION','SOLVE','FRESH_CERTIFICATE'],
        expected_outputs=['RESULT.json','final_flow.npz','HISTORY.json'],resources=ctx.c['resources'],
        acceptance_tests=['fixed_B_OD_masks','conservation','fresh_certificates'],
        registry_sha256=sha(ctx.root/'training/REGISTRY_FREEZE.json'),source_snapshot=component_snapshot(ctx,'assignments'),
        complete_query_start_epoch=start,deadline_epoch=deadline,initialization='COLD_FEASIBLE_AON')
    write_json(ctx.root/'intents'/f'{case_id}.json',loading_intent)
    try:
        ctx.guard(deadline=deadline)
        if row['network']=='P':
            graph,od,a,_=load_common(ctx.root);objective=PolynomialObjective(a['base'],a['k4'])
        else:graph,od,a,objective=load_ac(ctx,row['suite_id'],row['family'],registry,deadline=deadline)
        ctx.guard(deadline=deadline)
    except (TimeoutError,MemoryError) as exc:
        target.mkdir(parents=True,exist_ok=False)
        outcome=dict(case=row,status='TIME_LIMIT_DURING_INPUT_LOAD' if isinstance(exc,TimeoutError) else 'UNAVAILABLE_CURRENT_RAM',
            final_certificate_fresh=False,elapsed_seconds=time.time()-start,complete_query_start_epoch=start,
            deadline_epoch=deadline,intent=loading_intent,error_type=type(exc).__name__,error=str(exc))
        write_json(target/'RESULT.json',outcome);ctx.flush_bindings();return outcome
    enabled=~a['closed_mask'] if row['scenario']=='i95' else np.ones(len(graph.tail),bool)
    intent=dict(loading_intent,graph_identity=graph.identity,OD_identity=od.identity)
    write_json(ctx.root/'intents'/f'{case_id}.json',intent)
    result=solve(graph,od,objective,target,method=row['solver'],run_start_epoch=start,
        deadline_epoch=deadline,tolerance=ctx.c['assignment']['RG_SPTT'],ctx=ctx,enabled=enabled,
        workers=ctx.c['resources']['regional_workers'],capture_witness=False)
    # Admit only the exact same served OD membership as the matching P reference.
    with np.load(ctx.root/'references'/row['scenario']/'coverage.npz',allow_pickle=False) as z:reference_serve=z['serve']
    if (target/'coverage.npz').exists():
        with np.load(target/'coverage.npz',allow_pickle=False) as z:actual_serve=z['serve']
        if not np.array_equal(actual_serve,reference_serve):raise ArithmeticError('Assignment changed exact served OD membership')
    result['case']=row;result['intent']=intent;result['independent_cold_control']=row['network']=='P'
    write_json(target/'RESULT.json',result);ctx.flush_bindings();return result

def stage_clock(ctx, *,action='assignments'):
    continuation=_admit_action(ctx,action)
    now=time.time()
    clock_path=ctx.root/'STAGE3_CLOCK.json'
    if clock_path.exists():
        clock=read_json(clock_path)
        start=clock['original_stage_start_epoch']
        if (not isinstance(start,(int,float)) or isinstance(start,bool) or not math.isfinite(start)
                or not ctx.c['original_start_epoch']<=start<=now):
            raise ValueError('Original Stage3 clock changed; reset prohibited')
        if clock.get('schema')=='PACNHN_STAGE3_CLOCK_CONTINUATION_V1':
            if continuation is None:raise ValueError('Stage3 continuation clock requires its authenticated authority')
            expected=_continuation_stage_clock(ctx,start,continuation)
            if clock!=expected:raise ValueError('Original Stage3 continuation clock/authority changed; reset prohibited')
        elif clock.get('schema') is not None:
            raise ValueError('Unknown Stage3 clock schema')
        elif clock['deadline_epoch']!=min(start+12*3600,ctx.c['maximum_cumulative_stage_deadlines'][2]):
            raise ValueError('Original Stage3 clock changed; reset prohibited')
        elif continuation is not None:
            # An earlier Stage3 run cannot acquire a new 12-hour budget merely
            # by changing configuration. Preserve its exact two-field clock.
            if start>=continuation['start_epoch'] or clock['deadline_epoch']>continuation['science_deadline_epoch']:
                raise ValueError('Legacy Stage3 clock conflicts with continuation; reset prohibited')
    else:
        start=now
        if continuation is None:
            clock=dict(original_stage_start_epoch=start,deadline_epoch=min(start+12*3600,ctx.c['maximum_cumulative_stage_deadlines'][2]))
        else:
            clock=_continuation_stage_clock(ctx,start,continuation)
            if now>=clock['deadline_epoch']:
                raise TimeoutError('No Stage3 time remains before the protected publication reserve')
        write_json(clock_path,clock)
    return clock


def _continuation_stage_clock(ctx,start,continuation):
    if not continuation['start_epoch']<=start:
        raise ValueError('Stage3 continuation start precedes the authorization')
    maximum=continuation['stage3_max_seconds']
    if maximum!=12*3600 or continuation['science_deadline_epoch']>continuation['hard_deadline_epoch']-4*3600:
        raise ValueError('Stage3 cap/publication reserve differs from the approved bounds')
    return dict(schema='PACNHN_STAGE3_CLOCK_CONTINUATION_V1',phase_id=ctx.c['phase_id'],run_id=ctx.c['run_id'],
        original_stage_start_epoch=start,deadline_epoch=min(start+maximum,continuation['science_deadline_epoch']),
        continuation_start_epoch=continuation['start_epoch'],continuation_hard_deadline_epoch=continuation['hard_deadline_epoch'],
        continuation_science_deadline_epoch=continuation['science_deadline_epoch'],stage3_max_seconds=maximum,
        continuation_approval_sha256=continuation['approval_sha256'],continuation_clock_sha256=continuation['clock_sha256'])


def run(ctx):
    from .gates import require_component
    # The first inference/assignment invocation starts the shared Stage3 clock;
    # its gates and registry loading consume the same budget as the queries.
    clock=stage_clock(ctx,action='assignments')
    require_component(ctx,'assignments')
    registry=validate_gates(ctx,deadline=clock['deadline_epoch'])
    deadline=clock['deadline_epoch'];rows=matrix()
    # All five frozen graph supports permit FFTT. Trained extra-resolution rows
    # require complete pure-family coverage in the frozen training registry.
    coverage={(r['suite_id'],r['family']):r for r in registry['coverage']}
    for sid in ctx.c['training']['additional_suite_order']:
        for family in ['FFTT','POWER4','XGBOOST','ICNN']:
            if family!='FFTT' and not coverage.get((sid,family),{}).get('complete_family',False):continue
            for scenario in ['open','i95']:
                rows.append(dict(case_id=f'AC_{sid}_{family}_{scenario}',network='AC',suite_id=sid,family=family,
                    solver='MSA' if family=='XGBOOST' else 'FW',scenario=scenario,priority='ADDITIONAL'))
    write_json(ctx.root/'ASSIGNMENT_MATRIX.json',rows);results=[]
    for row in rows:
        if time.time()>=deadline:
            results.append(dict(case=row,status='UNRUN_STAGE_TIME_LIMIT'));continue
        if row['network']=='AC' and row['family']!='FFTT' and not coverage.get((row['suite_id'],row['family']),{}).get('complete_family',False):
            results.append(dict(case=row,status='UNAVAILABLE_INCOMPLETE_PURE_FAMILY'));continue
        result=run_case(ctx,row,registry,deadline);results.append(result)
        write_json(ctx.root/'ASSIGNMENT_RESULTS.json',results)
    write_json(ctx.root/'ASSIGNMENT_RESULTS.json',results)
    write_json(ctx.root/'STAGE3_RESULT.json',dict(status='COMPLETE_WITH_RECORDED_OUTCOMES',clock=clock,
        elapsed_seconds=time.time()-clock['original_stage_start_epoch'],rows=len(results),primary_rows=14))
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--step-id',required=True,choices=['assignments']);args=parser.parse_args()
    run(Context(args.config,step_id=args.step_id))
