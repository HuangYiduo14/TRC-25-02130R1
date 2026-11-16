"""Feasible FW, safeguarded CFW and cost-only MSA with fresh certificates."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
from partC_rev1_src.partC_rev1_assignment import partC_rev1_cfw_alpha
from partC_rev1_src.phase4.models import SparseFlowVector
from .oracle import Demand,aon,conservation,combine_traces,save_traces
from .control import write_json,save_npz,array_hash

class PolynomialObjective:
    has_potential=True
    def __init__(self,base,k4):
        self.base=np.asarray(base,np.float64);self.k4=np.asarray(k4,np.float64)
        if self.base.shape!=self.k4.shape or not np.isfinite(self.base).all() or not np.isfinite(self.k4).all() or (self.base<0).any() or (self.k4<0).any():raise ValueError('Invalid fixed polynomial costs')
    def cost(self,x):return self.base+self.k4*np.asarray(x)**4
    def value_cost(self,x):
        x=np.asarray(x,np.float64);increment=self.k4*x**4
        return float(x@(self.base+increment/5)),self.base+increment
    def hessian_diag(self,x):return 4*self.k4*np.asarray(x)**3


def certificate(flow,cost,linear,value, *,has_potential):
    tstt=float(flow@cost);sptt=float(linear['sptt']);gap=tstt-sptt
    allowance=1e-10*max(1.,abs(tstt),abs(sptt))
    if gap < -allowance or abs(linear['loaded_sptt']-sptt)>allowance:raise ArithmeticError('Invalid signed gap or oracle')
    return dict(state_hash=array_hash({'flow':flow}),TSTT=tstt,SPTT=sptt,raw_signed_gap=gap,
        RG_SPTT=gap/sptt if sptt>0 else None,arithmetic_allowance=allowance,value=value,
        interpretation='CONVEX_POTENTIAL_FW_GAP' if has_potential else 'POINTWISE_VI_RESIDUAL_NO_POTENTIAL',fresh=True)

def converged(cert,tol):
    return cert['RG_SPTT']<=tol if cert['RG_SPTT'] is not None else abs(cert['raw_signed_gap'])<=cert['arithmetic_allowance']

def line_search(objective,x,sight,value,cost,guard):
    direction=sight-x;d0=float(cost@direction)
    if d0>=0:return 0.,value,dict(derivative0=d0,calls=0)
    v,c=objective.value_cost(sight);d1=float(c@direction)
    if d1<=0:return 1.,v,dict(derivative0=d0,derivative1=d1,calls=1)
    lo=0.;hi=1.
    for it in range(48):
        guard();alpha=(lo+hi)/2;trial=(1-alpha)*x+alpha*sight
        v,c=objective.value_cost(trial);derivative=float(c@direction)
        if not np.isfinite(c).all() or (c<0).any():raise ValueError('Invalid trial costs')
        if derivative>0:hi=alpha
        else:lo=alpha
        if hi-lo<=1e-11:break
    if v>value+1e-10*max(1.,abs(value)):raise ArithmeticError('Potential increased during line search')
    return alpha,v,dict(derivative0=d0,derivative1=d1,derivative_final=derivative,calls=it+2)

def solve(graph,od,objective,root, *,method,run_start_epoch,deadline_epoch,tolerance,ctx,
          enabled=None,edge_regions=None,workers=1,batch_origins=16,
          capture_witness=False,max_updates=None):
    """A complete query starts before graph/model input loading, at run_start_epoch.

    The last certified state is retained separately if a later refresh hits its
    cap. Time-limited output never borrows a certificate from another flow.
    """
    if method not in {'FW','CFW','MSA'}:raise ValueError('Explicit supported solver required')
    if method!='MSA' and not objective.has_potential:raise ValueError('Cost-only model requires MSA')
    if method=='CFW' and not hasattr(objective,'hessian_diag'):raise ValueError('Current CFW supports separable polynomial benchmark only')
    if od.total<=0:raise ValueError('Zero demand is not a national experiment')
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    enabled=np.ones(len(graph.tail),bool) if enabled is None else np.asarray(enabled,bool)
    iterations=0;history=[];stages=[];latest_cert=None;certified_flow=None;certified_traces=None
    flow=None;traces=[];ever=[];previous_sight=None;previous_traces=None;previous_direction=None;previous_step=None
    reason='RUNNING';last_aon_seconds=0.;last_line_seconds=0.
    def guard():ctx.guard(deadline=deadline_epoch)
    def evaluate(x):
        if objective.has_potential:value,cost=objective.value_cost(x)
        else:value,cost=None,objective.cost(x)
        if not np.isfinite(cost).all() or np.any(cost<0):raise ValueError('Nonfinite or negative model cost')
        if value is not None and not np.isfinite(value):raise ValueError('Nonfinite potential')
        return value,cost
    def load(cost,witness=False):
        return aon(graph,od,cost,edge_regions=edge_regions,enabled=enabled,guard=guard,
            workers=workers,batch_origins=batch_origins,witness_root=root/'initial_witnesses' if witness else None)
    try:
        guard();t=time.perf_counter();_,cost=evaluate(np.zeros(len(graph.tail)))
        linear=load(cost,capture_witness)
        initial_serve=od.serve.copy();served=initial_serve&np.isfinite(linear['distances'])
        od=Demand(od.origins,od.offsets,od.destinations,od.demand,served)
        if od.total<=0:raise ValueError('No OD demand reachable')
        balance=od.balance(len(graph.node_ids));flow=linear['flow'];traces=linear['traces']
        conservation(graph,flow,balance,od.total)
        ever=[v.keys.copy() for v in traces]
        save_npz(root/'coverage.npz',serve=served,initial_serve=initial_serve,initial_shortest_cost=linear['distances'])
        stages.append(dict(stage='FEASIBLE_INITIALIZATION_AND_OD_WITNESSES',seconds=time.perf_counter()-t))
        write_json(root/'COVERAGE.json',dict(aggregated_rows=len(served),served_rows=int(served.sum()),
            served_demand=od.total,unreachable_demand=float(od.demand[initial_serve&~served].sum()),
            demand_renormalized=False,OD_identity=od.identity))
        while True:
            guard();t=time.perf_counter();value,cost=evaluate(flow);eval_seconds=time.perf_counter()-t
            t=time.perf_counter();linear=load(cost);last_aon_seconds=time.perf_counter()-t
            if abs(linear['reachable_demand']-od.total)>1e-10*max(1.,od.total):raise ArithmeticError('Frozen served demand coverage changed')
            cert=certificate(flow,cost,linear,value,has_potential=objective.has_potential)
            residual=conservation(graph,flow,balance,od.total)
            if np.any(flow[~enabled]!=0):raise ArithmeticError('Positive masked flow')
            latest_cert=cert;certified_flow=flow.copy();certified_traces=traces
            row=dict(iteration=iterations,elapsed_seconds=time.time()-run_start_epoch,**cert,**residual,
                value_cost_seconds=eval_seconds,aon_seconds=last_aon_seconds)
            history.append(row)
            save_npz(root/'certified_flow.npz',flow=flow,cost=cost,serve=served)
            save_traces(root/'certified_traces.npz',traces)
            write_json(root/'CERTIFICATE.json',cert);write_json(root/'HISTORY.json',history)
            write_json(root/'PROGRESS.json',dict(iteration=iterations,RG_SPTT=cert['RG_SPTT'],
                elapsed_seconds=time.time()-run_start_epoch,deadline_epoch=deadline_epoch))
            print(f'{method} {root.name} update={iterations} RG={cert["RG_SPTT"]} elapsed={time.time()-run_start_epoch:.2f}s',flush=True)
            if converged(cert,tolerance):reason='CONVERGED';break
            if max_updates is not None and iterations>=max_updates:reason='PILOT_UPDATE_LIMIT';break
            # Leave the current certified state intact if a complete next
            # update+refresh cannot reasonably fit the query clock.
            if time.time()+1.4*last_aon_seconds+max(last_line_seconds,2*eval_seconds)+5>=deadline_epoch:
                reason='TIME_LIMIT';break
            sight=linear['flow'];sight_traces=linear['traces'];beta=0.;cfw_meta={}
            fw_direction=sight-flow
            if method=='CFW' and previous_sight is not None and previous_step<1-1e-8:
                beta,cfw_meta=partC_rev1_cfw_alpha(previous_sight,flow,fw_direction,previous_step,
                    previous_direction,objective.hessian_diag(flow),denominator_tol=1e-12,delta_cfw=1e-4)
                if beta>0:
                    sight=beta*previous_sight+(1-beta)*linear['flow']
                    sight_traces=combine_traces(1-beta,previous_traces,linear['traces'])
                if float(cost@(sight-flow))>=-1e-9:
                    beta=0.;sight=linear['flow'];sight_traces=linear['traces'];cfw_meta['reset']='non_descent'
            t=time.perf_counter()
            if method=='MSA':alpha=1./(iterations+2);search={}
            else:
                alpha,_,search=line_search(objective,flow,sight,value,cost,guard)
                if alpha==0 and beta>0:
                    beta=0.;sight=linear['flow'];sight_traces=linear['traces'];cfw_meta['reset']='zero_step'
                    alpha,_,search=line_search(objective,flow,sight,value,cost,guard)
            last_line_seconds=time.perf_counter()-t
            if alpha==0:reason='STAGNATION';break
            new_flow=(1-alpha)*flow+alpha*sight
            conservation(graph,new_flow,balance,od.total)
            if method!='MSA':
                new_value,_=evaluate(new_flow)
                if new_value>value+1e-10*max(1.,abs(value)):raise ArithmeticError('Accepted potential increase')
            previous_sight=sight;previous_traces=sight_traces;previous_direction=sight-flow;previous_step=alpha
            flow=new_flow;traces=combine_traces(alpha,traces,sight_traces)
            ever=[np.union1d(k,v.keys) for k,v in zip(ever,sight_traces)]
            iterations+=1
            row.update(step=alpha,cfw_beta=beta,cfw_details=cfw_meta,line_search=search,line_search_seconds=last_line_seconds)
            save_npz(root/'accepted_flow.npz',flow=flow,iteration=np.array(iterations))
            save_traces(root/'accepted_traces.npz',traces)
            write_json(root/'HISTORY.json',history)
    except TimeoutError as exc:
        reason='TIME_LIMIT';write_json(root/'TIME_LIMIT.json',dict(message=str(exc),epoch=time.time()))
    except Exception as exc:
        write_json(root/'FAILED.json',dict(type=type(exc).__name__,message=str(exc),iteration=iterations))
        raise
    # Freeze the actual reported benchmark state and its own fresh certificate.
    final_flow=certified_flow if certified_flow is not None else flow
    final_traces=certified_traces if certified_traces is not None else traces
    if final_flow is not None:save_npz(root/'final_flow.npz',flow=final_flow)
    save_traces(root/'final_traces.npz',final_traces)
    save_npz(root/'ever_active_pairs.npz',**{f'keys_{i}':v for i,v in enumerate(ever)})
    final=dict(status=reason,method=method,accepted_updates=iterations,reported_state_iteration=history[-1]['iteration'] if history else None,
        reported_state='LAST_FRESH_CERTIFIED_STATE' if latest_cert else 'UNCERTIFIED_INITIALIZATION',
        certificate=latest_cert,elapsed_seconds=time.time()-run_start_epoch,
        complete_query_start_epoch=run_start_epoch,deadline_epoch=deadline_epoch,
        nominal_tolerance=tolerance,graph_identity=graph.identity,OD_identity=od.identity,
        objective_has_potential=objective.has_potential,trace_count=len(final_traces),
        ever_active_pairs=[len(k) for k in ever],final_certificate_fresh=latest_cert is not None)
    write_json(root/'RESULT.json',final);write_json(root/'STAGES.json',stages);write_json(root/'HISTORY.json',history)
    return final
