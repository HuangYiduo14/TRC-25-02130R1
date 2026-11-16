"""Matched contemporary physical controls using the inherited numerical backend."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import time
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,write_json,save_npz,array_hash,sha
from partC_rev1_src.phase4.inputs import load_phase4_graph,load_or_build_aggregated_od
from .physical_solver import Phase4SolverConfig,solve_assignment
from .context import code_identity,sources
from .code_dependencies import physical_code_bindings
from partC_rev1_src.phase5.scc_reachability import load_reachable_od_supply_cache


def run(ctx,args):
    if args.scenario not in {'CFW','FW','MSA'}:raise ValueError('Explicit physical algorithm required')
    test=read_json(ctx.root/'validation/CURRENT_PHYSICAL_TESTS.json')
    if test['status']!='PASS':raise ValueError('Current physical controls require passed integration tests')
    if test['tested_code']!=physical_code_bindings(ctx.repo):raise ValueError('Physical dependency closure changed after tests')
    for relative,expected in test['tested_code'].items():
        if sha(ctx.repo/relative)!=expected:raise ValueError('Physical code changed after its scoped tests')
    p4=ctx.repo/'partC_doc_icnn/phase_4_work/phase4_dual_accuracy_benchmark_execution_config.json'
    src=sources(ctx)
    index=ctx.prior/'tables/source_bindings.csv'
    # This exact final index is a verified payload of the accepted R2/R3 ZIP,
    # distinct from its earlier R2_CHECKPOINT index.
    ctx.bind(index,'ACCEPTED_R2R3_FINAL_SOURCE_INDEX',
        '65371e90874bd85776932f4d2444e4e07076e0f1c050b0d8368b6039c318ddeb')
    accepted_rows=pd.read_csv(index,usecols=['native_path','sha256'],dtype=str)
    accepted_rows['native_path']=accepted_rows.native_path.map(lambda p:str(Path(p).resolve()))
    def accepted(path,role):
        path=Path(path).resolve(strict=True)
        candidates=set(accepted_rows.loc[accepted_rows.native_path.eq(str(path)),'sha256'])
        if str(path) in src.expected:candidates.add(src.expected[str(path)])
        if len(candidates)!=1:raise ValueError('Accepted physical source hash unavailable or conflicting: '+role)
        return ctx.bind(path,role,next(iter(candidates)))
    accepted(p4,'ACCEPTED_PHYSICAL_INPUT_CONFIG');c=read_json(p4);inp=c['input_versions']
    # The existing aggregate loader does not consume the original raw OD metadata.
    for k in ('graph_npz','graph_metadata_json'):
        accepted(inp[k],'ACCEPTED_PHYSICAL_'+k.upper())
    agg=Path(inp['accepted_aggregated_od_root'])/'derived_inputs'
    am=read_json(agg/'aggregated_od_metadata.json')
    accepted(agg/'aggregated_od_metadata.json','ACCEPTED_AGGREGATED_OD_METADATA')
    accepted(agg/'aggregated_od.npz','ACCEPTED_AGGREGATED_OD')
    accepted(agg/'original_to_aggregated_od_index.npz','ACCEPTED_OD_MULTIPLICITY_MAP')
    ctx.bind(agg/'aggregated_od.npz','ACCEPTED_AGGREGATED_OD',am['aggregate_npz_sha256'])
    ctx.bind(agg/'original_to_aggregated_od_index.npz','ACCEPTED_OD_MULTIPLICITY_MAP',am['mapping_npz_sha256'])
    graph=load_phase4_graph(c)
    clean_hash=hashlib.sha256(np.ascontiguousarray(graph.capacity,dtype=np.float64).tobytes()).hexdigest()
    if clean_hash!=inp['clean_capacity_payload_sha256']:
        raise ValueError('Loaded prepared capacity differs from accepted clean CURRENT payload')
    od=load_or_build_aggregated_od(c,output_root=ctx.root,existing_root=inp['accepted_aggregated_od_root'])
    supply_path=ctx.repo.parent/(ctx.repo.name+'_outputs')/'phase_05/pilot_v4_0905/run_001/cache/reachable_od_supply_v2.npz'
    sm=read_json(supply_path.with_suffix('.metadata.json'))
    accepted(supply_path.with_suffix('.metadata.json'),'ACCEPTED_PHYSICAL_REACHABILITY_METADATA')
    accepted(supply_path,'ACCEPTED_PHYSICAL_REACHABILITY')
    ctx.bind(supply_path,'ACCEPTED_PHYSICAL_REACHABILITY',sm['cache_sha256'])
    if sm['aggregated_od_sha256']!=am['aggregate_npz_sha256'] or sm['cross_scc_reachable_row_count']!=0:
        raise ValueError('Physical reachable-demand source proof differs')
    supply=load_reachable_od_supply_cache(supply_path,expected_graph_identity=inp['graph_identity'],
        expected_od_identity=inp['od_identity'],expected_node_count=len(graph.node_idx_to_uid),expected_row_count=len(od.demand))
    ctx.guard();root=ctx.root/'runs'/args.job_id;root.mkdir(exist_ok=False)
    code,members,inherited=code_identity(ctx.repo)
    cfg=Phase4SolverConfig(method=args.scenario,capacity_multiplier=.3,max_iterations=2**31-1,minimum_iterations=1,
        gap_target=.001,worker_count=1,max_origins=None,max_runtime_seconds=max(1,args.job_deadline-time.time()-120),
        rss_hard_bytes=ctx.c['max_process_tree_RSS_bytes'],checkpoint_directory=root/'checkpoints',
        final_checkpoint=True,checkpoint_compressed=True,iteration_record_path=root/'iterations.jsonl')
    config_record=asdict(cfg)
    for key in ('checkpoint_directory','iteration_record_path'):config_record[key]=str(config_record[key])
    freeze=dict(phase_id='N8',step_id='N8R4_NUM',run_id=ctx.c['run_id'],job_id=args.job_id,
        graph_identity=graph.graph_hash,od_identity=od.od_hash,physical_arcs=len(graph.tail),physical_nodes=len(graph.node_idx_to_uid),
        OD_aggregated_rows=len(od.demand),original_rows=od.original_row_count,origins=len(od.origin_group_node_idx),
        total_demand=float(od.demand.sum()),capacity_multiplier=.3,capacity_multiplier_applications=1,
        clean_prepared_capacity_payload_sha256=clean_hash,
        physical_cost_units='MINUTES_NATIVE; EXPORTED_COSTS_SECONDS_ONCE_X60',
        scenario=args.scenario,solver_config=config_record,code_identity=code,code_members=members,
        tested_scientific_dependency_code=test['tested_code'],physical_test_receipt_sha256=sha(ctx.root/'validation/CURRENT_PHYSICAL_TESTS.json'),
        clocks=dict(R4_start=ctx.c['start_epoch'],job_start=args.job_start,job_deadline=args.job_deadline),
        no_parallel_heavy_jobs=True,initialization='FEASIBLE_FREEFLOW_AON_COLD',
        matched_backend='phase4 kernels, costs, line search and CFW alpha; scoped fresh stopping only',
        diagnostic_trace_maps='NONE_FOR_ALL_THREE_CONTROLS; no regional trace overhead',
        iteration_cap_role='INTEGER_SAFETY_CEILING_NOT_A_FIXED_ITERATION_EXPERIMENT')
    write_json(root/'PRE_RUN_FREEZE.json',freeze)
    source_load=time.time()-args.job_start
    ctx.guard();result=solve_assignment(graph,od,trace_maps=[],config=cfg,run_metadata=freeze);ctx.guard()
    flow=result.flow;cost_s=result.cost*60.;state_hash=array_hash({'flow':flow})
    signed=float(flow@result.cost-result.latest_aon.sptt)
    allowance=1e-10*max(1.,abs(float(flow@result.cost)),abs(result.latest_aon.sptt))
    if signed < -allowance:raise ArithmeticError('Material negative signed physical gap')
    loaded=float(result.latest_aon.physical_flow@result.cost)
    if abs(loaded-result.latest_aon.sptt)>allowance:raise ArithmeticError('Physical loaded SPTT inconsistent')
    outgoing=np.bincount(graph.tail,weights=flow,minlength=len(graph.node_idx_to_uid))
    incoming=np.bincount(graph.head,weights=flow,minlength=len(graph.node_idx_to_uid))
    balance=supply.origin_node_flow-supply.destination_node_flow
    residual=outgoing-incoming-balance;maximum=float(np.max(np.abs(residual),initial=0.))
    tolerance=1e-10*max(1.,supply.reachable_demand)
    feasible=maximum<=tolerance and np.isclose(supply.reachable_demand,result.latest_aon.reachable_demand,rtol=1e-12,atol=1e-8)
    pd.DataFrame(dict(physical_node_uid=graph.node_idx_to_uid.astype(str),represented_OD_net_supply=balance,
        outgoing_flow=outgoing,incoming_flow=incoming,residual=residual)).to_csv(root/'node_balance.csv',index=False,float_format='%.17g')
    write_json(root/'FEASIBILITY.json',dict(passed=bool(feasible),maximum_absolute_residual=maximum,
        absolute_tolerance=tolerance,max_incident_scaled_residual=float(np.max(np.abs(residual)/np.maximum(1.,outgoing+incoming+abs(balance)),initial=0.)),
        reachable_demand=supply.reachable_demand,unreachable_demand=supply.unreachable_demand,source_cache_sha256=sha(supply_path)))
    if not feasible:raise ArithmeticError('Physical current-flow conservation or represented-demand identity failed')
    save_npz(root/'final_flow.npz',dict(flow=flow,cost_seconds=cost_s))
    rows=[x.to_dict() for x in result.histories]
    pd.DataFrame(rows).to_csv(root/'iteration_history.csv',index=False,float_format='%.17g')
    final=dict(state_hash=state_hash,RG_SPTT=signed/result.latest_aon.sptt,TSTT=float(flow@cost_s),SPTT=result.latest_aon.sptt*60.,
        raw_signed_gap=signed*60.,arithmetic_allowance=allowance*60.,fresh=True,time_units='SECONDS',
        objective=result.final_metrics.objective*60.,conservation_passed=True,maximum_node_balance_residual=maximum)
    status='CONVERGED_REPRESENTED_DEMAND_ONLY' if final['RG_SPTT']<=.001 else 'TIME_LIMIT'
    record=dict(stop_reason=status,native_stop_reason=result.stop_reason,scenario='PHYSICAL_'+args.scenario,
        accepted_iterations=len(rows),final_certificate=final,final_certificate_fresh=True,
        represented_demand=result.latest_aon.reachable_demand,unreachable_demand=result.latest_aon.unreachable_demand,
        total_original_demand=float(od.demand.sum()),source_load_seconds=source_load,
        solver_elapsed_seconds=result.runtime_seconds,initialization_seconds=result.initialization_seconds,
        final_refresh_seconds=result.final_refresh_seconds,elapsed_seconds=time.time()-args.job_start,
        current_flow_file_sha256=sha(root/'final_flow.npz'),peak_rss_bytes=result.peak_rss_bytes,
        native_cost_units='minutes',exported_cost_units='seconds',worker_count=1,identity=freeze,
        matched_contemporary_control=True,historical_x45_unchanged=True)
    write_json(root/'RESULT.json',record)
    return dict(status='EXECUTED',result=record)
