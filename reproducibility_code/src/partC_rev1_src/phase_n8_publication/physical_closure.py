"""Matched represented-subset physical CFW closure with exact directed OD coverage."""
from dataclasses import replace,asdict
import hashlib
import time
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,read_npz,write_json,save_npz,sha,array_hash,digest
from partC_rev1_src.phase_n8.oracle import Network,Demand,aon,conservation
from partC_rev1_src.phase_n8_r4.physical_solver import Phase4SolverConfig,solve_assignment
from partC_rev1_src.phase_n8_r4.path_metrics import bindings
from .context import sources,code_identity
from .overlay import apply_overlay


def physical_overlay(graph,unavailable):
    mask=np.asarray(unavailable,bool)
    if mask.shape!=graph.tail.shape:raise ValueError('Physical closure mask dimension differs')
    forward=graph.csr_arc_idx[~mask[graph.csr_arc_idx]]
    reverse=graph.reverse_csr_arc_idx[~mask[graph.reverse_csr_arc_idx]]
    n=len(graph.node_idx_to_uid)
    fi=np.r_[0,np.cumsum(np.bincount(graph.tail[forward],minlength=n))].astype(np.int64)
    ri=np.r_[0,np.cumsum(np.bincount(graph.head[reverse],minlength=n))].astype(np.int64)
    identity=digest(dict(base_graph=graph.graph_hash,mask=array_hash({'unavailable':mask}),
                         policy='MATCHED_REPRESENTED_I95_SOURCE_ARCS_FILTERED_CSR'))
    return replace(graph,graph_hash=identity,csr_arc_idx=forward,csr_indptr=fi,csr_head=graph.head[forward],
        reverse_csr_arc_idx=reverse,reverse_csr_indptr=ri,reverse_csr_head=graph.tail[reverse],
        metadata=dict(graph.metadata,base_graph_identity=graph.graph_hash,assignment_graph_hash=identity,
                      closure_overlay_identity=identity,unavailable_count=int(mask.sum())))


def directed_closed_membership(graph,od,unavailable,guard=None):
    # Physical backend has no centroid transit exclusion; reproduce that policy explicitly.
    base=Network(graph.tail,graph.head,np.zeros(len(graph.node_idx_to_uid),bool),
                 graph.arc_idx_to_directed_arc_id.astype(str),graph.node_idx_to_uid.astype(str))
    masked=apply_overlay(base,unavailable)
    all_od=Demand(od.origin_group_node_idx,od.origin_offsets,od.destination_node_idx,od.demand,np.ones(len(od.demand),bool))
    loading=aon(masked,all_od,graph.free_time,guard=guard,return_distances=True)
    serve=np.isfinite(loading['distances'])
    served=Demand(all_od.origins,all_od.offsets,all_od.destinations,all_od.demand,serve)
    balance=served.balance(len(graph.node_idx_to_uid))
    conservation(masked,loading['flow'],balance,served.total)
    if np.any(loading['flow'][unavailable]!=0):raise ArithmeticError('Masked physical membership load traverses removed arc')
    return serve,balance,loading['elapsed_seconds']


def run(ctx,args):
    if args.scenario!='P_CLOSED':raise ValueError('Explicit matched physical closure scenario required')
    test=read_json(ctx.root/'validation/TESTS.json')
    if test['status']!='PASS':raise ValueError('Current preflight tests did not pass')
    src=sources(ctx);b=bindings(ctx);base=b.graph;od=b.od
    for key in ('graph_npz','graph_metadata_json'):
        src.bind(src.p4['input_versions'][key],'ACCEPTED_PHYSICAL_'+key)
    raw,_,baseline_serve,_=src.original_od()
    if not np.array_equal(raw['demand'],od.demand):raise ValueError('Physical OD wrapper differs from exact baseline ledger')
    freeze=read_json(ctx.root/'closure/MASK_FREEZE.json')
    if freeze['status']!='EXACT_REPRESENTED_MASK_EQUIVALENCE_PASS':raise ValueError('Physical matched-mask gate incomplete')
    ctx.bind(ctx.root/'closure/availability.npz','EXACT_MATCHED_PHYSICAL_MASK',freeze['mask_sha256'])
    mask=read_npz(ctx.root/'closure/availability.npz')['physical_unavailable']
    if int(mask.sum())!=1105:raise ValueError('Matched physical source count differs')
    graph=physical_overlay(base,mask)
    clean_hash=hashlib.sha256(np.ascontiguousarray(graph.capacity,dtype=np.float64).tobytes()).hexdigest()
    if clean_hash!=src.p4['input_versions']['clean_capacity_payload_sha256']:
        raise ValueError('Physical clean capacity changed')
    root=ctx.root/'runs'/args.job_id;root.mkdir(exist_ok=False)
    serve,balance,reach_seconds=directed_closed_membership(graph,od,mask,ctx.guard)
    if np.any(serve & ~baseline_serve):raise ValueError('Removing edges increased physical reachability')
    ledger=[]
    for name,which in [('BASELINE_UNREACHABLE',~baseline_serve),('ADDITIONAL_CLOSED_UNREACHABLE',baseline_serve&~serve),('CLOSED_REACHABLE',serve)]:
        ledger.append(dict(classification=name,aggregated_rows=int(which.sum()),
            original_rows=int(od.original_row_multiplicity[which].sum()),demand=float(od.demand[which].sum())))
    pd.DataFrame(ledger).to_csv(root/'OD_COVERAGE.csv',index=False,float_format='%.17g')
    save_npz(root/'closed_demand_membership.npz',dict(serve=serve,balance=balance,
        original_row_multiplicity=od.original_row_multiplicity))
    code,members,_=code_identity(ctx.repo)
    cfg=Phase4SolverConfig(method='CFW',capacity_multiplier=.3,max_iterations=2**31-1,minimum_iterations=1,
        gap_target=.001,worker_count=1,max_origins=None,max_runtime_seconds=max(1,args.job_deadline-time.time()-180),
        rss_hard_bytes=ctx.c['max_process_tree_RSS_bytes'],checkpoint_directory=root/'checkpoints',
        final_checkpoint=True,checkpoint_compressed=True,iteration_record_path=root/'iterations.jsonl')
    cfg_json=asdict(cfg)
    for key in ('checkpoint_directory','iteration_record_path'):cfg_json[key]=str(cfg_json[key])
    identity=dict(phase_id='N8',step_id=ctx.c['step_id'],run_id=ctx.c['run_id'],job_id=args.job_id,
        base_graph_identity=base.graph_hash,graph_identity=graph.graph_hash,od_identity=od.od_hash,
        scenario='MATCHED_REPRESENTED_I95_PHYSICAL_SUBSET',mask_sha256=freeze['mask_sha256'],
        matched_source_arcs=int(mask.sum()),suppressed_interiors_omitted=608,healthy_macro_alternatives_retained=55,
        physical_arcs=len(graph.tail),physical_nodes=len(graph.node_idx_to_uid),capacity_multiplier=.3,
        clean_prepared_capacity_payload_sha256=clean_hash,capacity_multiplier_applications=1,
        original_total_demand=float(od.demand.sum()),served_closed_demand=float(od.demand[serve].sum()),
        demand_renormalized=False,closed_membership_sha256=sha(root/'closed_demand_membership.npz'),
        original_start_epoch=ctx.c['start_epoch'],job_start_epoch=args.job_start,job_deadline_epoch=args.job_deadline,
        solver_config=cfg_json,code_identity=code,code_members=members,initialization='FRESH_MASKED_FREEFLOW_AON',
        no_regional_trace_output=True,physical_cost_units='MINUTES_NATIVE_EXPORT_SECONDS_ONCE_X60')
    write_json(root/'PRE_RUN_FREEZE.json',identity)
    source_load=time.time()-args.job_start
    result=solve_assignment(graph,od,trace_maps=[],config=cfg,run_metadata=identity)
    flow=result.flow;cost_seconds=result.cost*60
    if np.any(flow[mask]!=0) or np.any(result.latest_aon.physical_flow[mask]!=0):
        raise ArithmeticError('Masked physical CFW flow is nonzero')
    out=np.bincount(graph.tail,weights=flow,minlength=len(balance));inc=np.bincount(graph.head,weights=flow,minlength=len(balance))
    residual=out-inc-balance;maximum=float(abs(residual).max());demand=float(od.demand[serve].sum())
    if maximum>1e-10*max(1.,demand) or not np.isclose(result.latest_aon.reachable_demand,demand,rtol=1e-12,atol=1e-8):
        raise ArithmeticError('Physical closed CURRENT conservation/coverage fails')
    tstt=float(flow@cost_seconds);sptt=result.latest_aon.sptt*60;signed=tstt-sptt
    allowance=1e-10*max(1.,abs(tstt),abs(sptt))
    if signed < -allowance or abs(float(result.latest_aon.physical_flow@cost_seconds)-sptt)>allowance:
        raise ArithmeticError('Material signed physical gap defect')
    cert=dict(state_hash=array_hash({'flow':flow}),RG_SPTT=signed/sptt,TSTT=tstt,SPTT=sptt,
        raw_signed_gap=signed,arithmetic_allowance=allowance,fresh=True,time_units='SECONDS',
        objective=result.final_metrics.objective*60,conservation_passed=True,maximum_node_balance_residual=maximum)
    save_npz(root/'final_flow.npz',dict(flow=flow,cost_seconds=cost_seconds,physical_arc_ids=graph.arc_idx_to_directed_arc_id.astype(str)))
    pd.DataFrame([r.to_dict() for r in result.histories]).to_csv(root/'iteration_history.csv',index=False,float_format='%.17g')
    pd.DataFrame(dict(physical_node_uid=graph.node_idx_to_uid.astype(str),closed_OD_net_supply=balance,
        outgoing_flow=out,incoming_flow=inc,residual=residual)).to_csv(root/'node_balance.csv',index=False,float_format='%.17g')
    record=dict(stop_reason='CONVERGED_REPRESENTED_DEMAND_ONLY' if cert['RG_SPTT']<=.001 else 'TIME_LIMIT',
        native_stop_reason=result.stop_reason,scenario='P_CLOSED',accepted_iterations=len(result.histories),
        final_certificate=cert,final_certificate_fresh=True,identity=identity,
        final_flow_sha256=sha(root/'final_flow.npz'),represented_demand=demand,coverage=ledger,
        elapsed_seconds=time.time()-args.job_start,source_load_and_membership_seconds=source_load,
        exact_membership_loading_seconds=reach_seconds,solver_elapsed_seconds=result.runtime_seconds,
        initialization_seconds=result.initialization_seconds,final_refresh_seconds=result.final_refresh_seconds,
        peak_rss_bytes=result.peak_rss_bytes,worker_count=1,masked_coordinates_all_zero=True,
        interpretation='Matched represented-subset closed reference; 608 suppressed B-interior arcs omitted')
    write_json(root/'RESULT.json',record)
    return dict(status='EXECUTED',result=record)
