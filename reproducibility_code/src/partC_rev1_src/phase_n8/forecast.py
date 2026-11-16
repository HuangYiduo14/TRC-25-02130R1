"""Observed assignment boundaries and conservative, explicit planning estimates."""
import math
import numpy as np


METHOD = dict(schema='N8_FORECAST_METHOD_V1',
    complete='Observed fresh-tolerance assignment; repeat high = 1.5 times observed end-to-end',
    censored='Fit last up to 7 positive post-initial signed gaps against iteration and log(iteration+1); compare geometric and power-law predictions',
    minimum_post_initial_gaps=5, maximum_log_fit_residual=.35,
    maximum_competing_remaining_iteration_ratio=2., multiplier=1.5,
    unknown='Nondecreasing/noisy/short or materially disagreeing fits give a null convergence high; time caps are never a tolerance forecast',
    remaining_metrics='1.5 times (observed cold load + 2 full-origin p90 loading passes + 2 p90 cost queries + 10 seconds I/O)',
    finalization_reserve_seconds=3600,uncertainty='Engineering planning, not a convergence guarantee or statistical confidence interval')


def trajectory_estimate(history,tol=.001):
    usable=[r for r in history if r['iteration']>0 and r.get('RG_SPTT') is not None and r['RG_SPTT']>tol]
    last=usable[-7:]
    if len(last)<5:return dict(status='UNKNOWN_SHORT_TRAJECTORY',base_remaining_iterations=None,high_remaining_iterations=None)
    k=np.array([r['iteration'] for r in last],float);g=np.log([r['RG_SPTT'] for r in last])
    fits=[]
    for method,x in [('geometric',k),('power_law',np.log1p(k))]:
        slope,intercept=np.polyfit(x,g,1);residual=float(np.max(np.abs(g-(slope*x+intercept))))
        if slope>=0:
            remaining=None
        else:
            crossing=(math.log(tol)-intercept)/slope
            if method=='power_law':crossing=math.expm1(crossing) if crossing<20 else math.inf
            remaining=max(1,math.ceil(crossing-k[-1])) if math.isfinite(crossing) else None
        fits.append(dict(method=method,slope=float(slope),intercept=float(intercept),max_log_residual=residual,remaining_iterations=remaining))
    values=[f['remaining_iterations'] for f in fits]
    if any(x is None for x in values) or any(f['max_log_residual']>.35 for f in fits):
        return dict(status='UNKNOWN_FLAT_OR_NOISY_TRAJECTORY',fits=fits,base_remaining_iterations=None,high_remaining_iterations=None)
    ratio=max(values)/max(1,min(values))
    if ratio>2:
        return dict(status='UNKNOWN_COMPETING_FITS_DISAGREE',fits=fits,competing_ratio=ratio,base_remaining_iterations=None,high_remaining_iterations=None)
    return dict(status='SUPPORTED_HEURISTIC',fits=fits,competing_ratio=ratio,
        base_remaining_iterations=max(values),high_remaining_iterations=math.ceil(1.5*max(values)))


def assignment_forecast(result,history,stages,inference):
    aons=[r['aon_seconds'] for r in history]
    grads=[r['seconds'] for r in inference if r['operation']=='VALUE_GRAD']
    hvps=[r['seconds'] for r in inference if r['operation']=='VALUE_GRAD_HVP']
    updates=np.diff([r['elapsed_seconds'] for r in history]).tolist()
    def quantile(a,q):return float(np.percentile(a,q)) if len(a) else None
    warm=quantile(updates[-7:],50);p90=quantile(updates[-7:],90)
    initial=sum(r['seconds'] for r in stages if r['stage']=='FEASIBLE_AON_INITIALIZATION')
    r=dict(scenario=result['scenario'],job_id=result['job_id'],stop_reason=result['stop_reason'],
        accepted_iterations=result['accepted_iterations'],fresh=result['final_certificate_fresh'],
        achieved_RG=(result.get('final_certificate') or {}).get('RG_SPTT'),
        observed_end_to_end_seconds=result['elapsed_seconds'],
        observed_solver_seconds=result.get('solver_elapsed_seconds'),
        observed_update_window_seconds=float(sum(updates)) if updates else 0.,
        update_window_definition='First fresh initial-state certificate to final certificate; includes FW updates, later loading and intervening checkpoints',
        cold_input_load_seconds=result.get('pre_run_input_load_seconds'),
        feasible_initialization_seconds=initial if stages else None,
        full_AON_p50_seconds=quantile(aons,50),full_AON_p90_seconds=quantile(aons,90),
        final_certificate_AON_seconds=aons[-1] if aons else None,
        warm_iteration_p50_seconds=warm,warm_iteration_p90_seconds=p90,
        value_gradient_p50_seconds=quantile(grads,50),value_gradient_p90_seconds=quantile(grads,90),
        full_HVP_p50_seconds=quantile(hvps,50),value_gradient_calls=len(grads),full_HVP_calls=len(hvps),
        censored=not result['stop_reason'].startswith('CONVERGED'),
        remaining_tolerance_base_seconds=None,remaining_tolerance_high_seconds=None,
        fresh_assignment_base_seconds=None,fresh_assignment_high_seconds=None,
        assignment_solver_base_seconds=None,assignment_solver_high_seconds=None,
        metrics_base_seconds=None,metrics_high_seconds=None,gate_status='UNRESOLVED',
        graph=result['identity']['graph'],demand=result['identity']['demand'],registry=result['identity']['registry'],
        support=result['identity']['support'],code=result['identity']['code'],represented_demand=result['represented_demand'],
        total_OD_count=10193770,represented_OD_count=10190380,represented_origins=3225,
        workers=1,nested_BLAS_threads=1,estimate_type='OBSERVED_OR_HEURISTIC_NOT_A_CONVERGENCE_GUARANTEE')
    if result['stop_reason'].startswith('CONVERGED') and result['final_certificate_fresh']:
        r.update(gate_status='COMPLETED_MATCHING',forecast_method='OBSERVED_COMPLETION',
            remaining_tolerance_base_seconds=0.,remaining_tolerance_high_seconds=0.,
            fresh_assignment_base_seconds=result['elapsed_seconds'],fresh_assignment_high_seconds=1.5*result['elapsed_seconds'],
            assignment_solver_base_seconds=result['solver_elapsed_seconds'],assignment_solver_high_seconds=1.5*result['solver_elapsed_seconds'])
    elif result['stop_reason'] in ['STOP_NEGATIVE_COST','STOP_NONFINITE']:
        r.update(gate_status='SAFETY_REJECTED',forecast_method='ACTUAL_COST_FAIL_STOP_NO_TOLERANCE_CLAIM',censored=False)
    else:
        estimate=trajectory_estimate(history);r['trajectory_fit']=estimate;r['forecast_method']=estimate['status']
        if estimate['status']=='SUPPORTED_HEURISTIC' and warm is not None:
            base=estimate['base_remaining_iterations']*warm
            high=estimate['high_remaining_iterations']*max(p90,warm)*1.5
            r.update(gate_status='FORECAST_SUPPORTED',remaining_tolerance_base_seconds=base,remaining_tolerance_high_seconds=high,
                fresh_assignment_base_seconds=result['elapsed_seconds']+base,
                fresh_assignment_high_seconds=1.5*result['elapsed_seconds']+high,
                assignment_solver_base_seconds=result['solver_elapsed_seconds']+base,
                assignment_solver_high_seconds=1.5*result['solver_elapsed_seconds']+high)
    if aons:
        r['metrics_base_seconds']=result['pre_run_input_load_seconds']+2*quantile(aons,50)+2*(quantile(grads,50) or 0)+10
        r['metrics_high_seconds']=1.5*(result['pre_run_input_load_seconds']+2*quantile(aons,90)+2*(quantile(grads,90) or 0)+10)
    if r['fresh_assignment_high_seconds'] is not None:
        r['proposed_official_run_cap_seconds']=min(14400,max(900,math.ceil((r['fresh_assignment_high_seconds']+180)/300)*300))
        r['high_exceeds_per_run_ceiling']=r['fresh_assignment_high_seconds']+180>14400
    else:r['proposed_official_run_cap_seconds']=None if r['gate_status']=='SAFETY_REJECTED' else 14400
    return r
