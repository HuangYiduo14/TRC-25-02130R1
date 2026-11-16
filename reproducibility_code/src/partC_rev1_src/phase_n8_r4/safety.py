"""Frozen weight-only registries, complete saved-failure replay and native tests."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,write_json,read_npz,sha,digest,array_hash
from partC_rev1_src.phase_n8.native import load_registry
from partC_rev1_src.phase_n8.objective import Composite,UnsafeCost,validate_arrays
from partC_rev1_src.phase67.portable import predict_arrays
from .safety_bound import plan,transformed_copy
from .context import sources,reopen_suite

REGISTRIES={
    'N8R4_PRIMARY_BOUND_SAFE_V1':'E2_PRIMARY_FROZEN',
    'N8R4_BEST_ANY_BOUND_SAFE_V1':'E2_BEST_ANY_FROZEN',
    'N8R4_NONCONSTANT_SOURCE_BOUND_SAFE_V1':'N8_PROPOSED_NONCONSTANT_VALID_V1',
}
ORIGINAL_RUNS={
    'E2_PRIMARY_FROZEN':'P03_S03_PRIMARY',
    'E2_BEST_ANY_FROZEN':'P08_S03_BEST_ANY',
    'N8_PROPOSED_NONCONSTANT_VALID_V1':'P09_S03_NONCONSTANT_VALID',
}
ANALYTIC='N8_PROPOSED_ANALYTIC_SAFE_VALID_V1'
FORMULA='TWO_SOFTPLUS_LINEAR_HEAD_WEIGHT_ONLY_LOWER_BOUND_V1_AUTHORITY_4947c04e8e77'


def torch_check(kind,a,u,d):
    """Independent autograd of the implemented potential, including its softplus threshold."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    t={k:torch.tensor(v,dtype=torch.float64) for k,v in a.items()}
    ut=torch.tensor(u,dtype=torch.float64,requires_grad=True);dt=torch.tensor(d,dtype=torch.float64)
    x=ut/t['scale'];h1=F.softplus(x@t['A1'].T+t['b1'],threshold=40)
    h2=F.softplus(h1@t['W2'].T+x@t['A2'].T+t['b2'],threshold=40)
    power=int(a.get('head_power',1))
    residual=(h2**power/power)@t['v']-t['anchor_value']-x@t['anchor_gradient']
    if 'SKIP_ICNN' in kind:
        eta=t['coef']*t['eta_unit'];amp=t['residual_unit']*t['gate']
        val=ut@t['c0']+(eta*t['scale']*x**5/5).sum()+amp*residual
    else:val=ut@t['c0']+t['Vscale']*t['gain']*residual
    grad=torch.autograd.grad(val,ut,create_graph=True)[0]
    hvp=torch.autograd.grad((grad*dt).sum(),ut)[0]
    got=[val.detach().numpy(),grad.detach().numpy(),hvp.detach().numpy()]
    expected=predict_arrays(kind,a,u,d)
    for x,y in zip(got,expected):np.testing.assert_allclose(x,y,rtol=2e-10,atol=2e-7)
    return {k:float(np.max(np.abs(x-y))) for k,x,y in zip(('value_abs_error','gradient_max_abs_error','hvp_max_abs_error'),got,expected)}


class SafeComposite(Composite):
    def __init__(self,suite,entries,registry,guard=None):
        super().__init__(suite,entries,registry,guard)
        self.bound=np.empty(self.m)
        for e in entries:self.bound[e['start']:e['stop']]=e['cost_lower_bound']

    def evaluate(self,flow,direction=None,*,enforce_safety=True):
        value,cost,hvp,diag=super().evaluate(flow,direction,enforce_safety=enforce_safety)
        rc=cost[self.np:self.np+self.m]
        allowance=1e-8+1e-11*np.maximum(1.,np.abs(self.c0))
        if (rc<self.bound-allowance).any():
            raise UnsafeCost('STOP_BOUND_VIOLATION',dict(minimum_cost_minus_bound=float((rc-self.bound).min())),flow)
        diag=dict(diag,global_neural_nonnegativity_certified=True,
            certificate_scope='MATHEMATICAL_LINEAR_HEAD_BOUND_WITH_CONSERVATIVE_COEFFICIENT_AND_FINITE_QUERY_GUARDS',
            minimum_regional_cost_seconds=float(rc.min()),minimum_cost_minus_derived_bound=float((rc-self.bound).min()),
            bound_margin_fraction=.01,registry_variant=self.registry['registry_id'])
        return value,cost,hvp,diag


def derived_entry(source,fallback=None):
    kind=source['model_kind'];a=source['arrays'];validate_arrays(kind,a)
    original_array_hash=array_hash(a)
    unsupported='ICNN' in kind and float(np.asarray(a.get('head_power',1)))!=1
    row={k:v for k,v in source.items() if k!='arrays'}
    if unsupported:
        if fallback is None:raise ValueError('Unsupported head without authenticated analytic registry entry')
        if fallback['model_kind'] not in {'FREEFLOW_LINEAR','OLD_POWER4_NNLS','NEW_POWER4_RELATIVE_NNLS'}:
            raise ValueError('Predeclared analytic fallback is not analytic')
        if any(source[k]!=fallback[k] for k in ('region_id','pair_order_hash','support_hash','start','stop')):
            raise ValueError('Fallback native identity mismatch')
        e=dict(fallback);b=a['c0'].copy();rho=None;status='UNSUPPORTED_NONLINEAR_HEAD_TYPED_ANALYTIC_FALLBACK'
        amp=0.;ell=None;negative=np.zeros_like(b)
    elif 'ICNN' in kind:
        bound=plan(a,kind,.01);rho=bound.rho;e=dict(source,arrays=transformed_copy(a,kind,rho))
        b=bound.cost_lower_bound;status=bound.status;amp=bound.source_amplitude
        ell=bound.raw_gradient_lower;negative=bound.native_negative_increment_bound
    else:
        e=dict(source);b=a['c0'].copy();rho=1.;status='UNCHANGED_ANALYTIC';amp=0.;ell=None;negative=np.zeros_like(b)
    validate_arrays(e['model_kind'],e['arrays'])
    np.testing.assert_array_equal(e['arrays']['c0'],a['c0'])
    if np.any(b < .01*a['c0']-(1e-10+1e-12*a['c0'])):raise ArithmeticError('Conservative safety bound misses promised margin')
    ea=e['arrays'];ek=e['model_kind']
    eta=(ea['coef']*ea['eta_unit'] if 'SKIP_ICNN' in ek else ea.get('eta',np.zeros_like(b)))
    residual_active='ICNN' in ek and float(ea['residual_unit']*ea['gate'] if 'SKIP_ICNN' in ek else ea['Vscale']*ea['gain'])>0
    effective=('BOUND_SAFE_'+ek if residual_active else ('ANALYTIC_POWER4' if np.any(eta>0) else 'FREEFLOW_LINEAR'))
    row.update(source_kind=kind,source_portable_sha256=source['portable_sha256'],source_arrays_sha256=original_array_hash,
        effective_source_kind=ek,effective_kind=effective,effective_portable_path=e['portable_path'],
        effective_portable_sha256=e['portable_sha256'],transformed_arrays_sha256=array_hash(ea),
        rho=rho,transformation_status=status,margin_fraction=.01,formula=FORMULA,
        minimum_lower_bound_seconds=float(b.min()),minimum_c0_seconds=float(a['c0'].min()),
        eta_positive_coordinates=int(np.count_nonzero(eta)),eta_min=float(eta.min()),eta_max=float(eta.max()),
        source_residual_amplitude=float(amp),neural_residual_active=bool(residual_active),
        dimensions=len(b),strong_attenuation=bool(rho is not None and rho<.1),
        source_head_power=int(np.asarray(a.get('head_power',1))),new_mathematical_variant=True)
    e.update(cost_lower_bound=b,transformation=row)
    if original_array_hash!=array_hash(a):raise ValueError('Source arrays changed by transformation')
    return e,row,ell,negative


def probe_entry(e):
    a=e['arrays'];kind=e['model_kind'];m=len(a['c0']);s=a.get('scale',np.ones(m))
    rng=np.random.default_rng(20260915+m)
    zero=np.zeros(m);u=s*rng.uniform(.05,.4,m);d=s*rng.uniform(-.3,.3,m)
    v0,g0,h0=predict_arrays(kind,a,zero,np.ones(m))
    np.testing.assert_allclose(g0,a['c0'],rtol=1e-10,atol=1e-8)
    if abs(float(v0))>1e-7:raise ArithmeticError('Nonzero anchor')
    val,g,h=predict_arrays(kind,a,u,d)
    eps=1e-4;vp,gp=predict_arrays(kind,a,u+eps*d);vm,gm=predict_arrays(kind,a,u-eps*d)
    np.testing.assert_allclose((vp-vm)/(2*eps),float(g@d),rtol=2e-5,atol=1e-4)
    np.testing.assert_allclose((gp-gm)/(2*eps),h,rtol=1e-3,atol=5e-5)
    q=float(d@h)
    if q < -1e-8*max(1.,float(np.linalg.norm(d)*np.linalg.norm(h))):raise ArithmeticError('Negative directional curvature')
    if np.min(g-e['cost_lower_bound']) < -1e-7:raise ArithmeticError('Observed derived-bound violation')
    return dict(region_id=e['region_id'],model_kind=kind,dimensions=m,value_directional_error=float(abs((vp-vm)/(2*eps)-g@d)),
        hvp_fd_max_error=float(np.max(np.abs((gp-gm)/(2*eps)-h))),directional_curvature=q,
        zero_anchor_gradient_max_error=float(np.max(abs(g0-a['c0']))),status='PASS')


def freeze(ctx):
    src=sources(ctx);suite,od,balance,coverage=reopen_suite(ctx,src,'s03')
    fallback,areg=load_registry(src,suite,ANALYTIC);fallback={e['region_id']:e for e in fallback}
    all_rows=[];tests=[];replays=[];scenarios=[]
    dst=ctx.root/'registries'/'bound_safe_v1';dst.mkdir(exist_ok=False)
    failure_csv=ctx.root/'validation'/'original_failure_queries';failure_csv.mkdir(exist_ok=False)
    for safe_name,original_name in REGISTRIES.items():
        ctx.guard();entries,reg=load_registry(src,suite,original_name)
        original=Composite(suite,entries,reg,guard=ctx.guard)
        prior=ctx.prior/'runs'/ORIGINAL_RUNS[original_name]
        stop=read_json(prior/'SAFETY_STOP.json');ctx.bind(prior/'SAFETY_STOP.json','ORIGINAL_NEURAL_SAFETY_STOP')
        ctx.bind(prior/'offending_query.npz','COMPLETE_ORIGINAL_SAVED_QUERY',stop['offending_query_sha256'])
        flow=read_npz(prior/'offending_query.npz')['flow']
        if array_hash({'flow':flow})!=stop['input_state_hash']:raise ValueError('Saved global flow identity mismatch')
        val,cost,hvp,diag=original.evaluate(flow,np.ones(len(flow)),enforce_safety=False)
        if not np.isclose(cost[stop['edge_index']],stop['cost_seconds'],rtol=1e-12,atol=1e-9):raise ArithmeticError('Original failure did not reproduce')
        offender=next(e for e in entries if e['region_id']==stop['region_id'])
        if offender['portable_sha256']!=stop['portable_sha256']:raise ValueError('Original offender hash changed')
        start=suite['packed_count']+offender['start'];end=suite['packed_count']+offender['stop'];u=flow[start:end]
        if start+stop['native_coordinate']!=stop['edge_index']:raise ValueError('Full-state to native-coordinate mapping mismatch')
        parity=torch_check(offender['model_kind'],offender['arrays'],u,np.ones(len(u)))
        safe_entries=[];registry_rows=[];coordinates=[]
        for i,e in enumerate(entries):
            if i%32==0:ctx.guard()
            derived,row,ell,neg=derived_entry(e,fallback[e['region_id']]);test=probe_entry(derived)
            test['registry_id']=safe_name;tests.append(test)
            row['registry_id']=safe_name
            exposure=suite['pairs'].iloc[e['start']:e['stop']].LOW_CURRENT.to_numpy(float)
            row['fixed_x45_coordinate_flow_sum']=float(exposure.sum())
            row['fixed_x45_c0_flow_seconds']=float(exposure@e['arrays']['c0'])
            safe_entries.append(derived);registry_rows.append(row);all_rows.append(row)
            native=suite['pairs'].iloc[e['start']:e['stop']]
            coordinates.append(pd.DataFrame(dict(registry_id=safe_name,region_id=e['region_id'],
                pair_id=native.pair_id.to_numpy(),native_index=native.native_index.to_numpy(),c0_seconds=e['arrays']['c0'],
                lower_bound_seconds=derived['cost_lower_bound'],negative_increment_bound_seconds=neg,
                raw_gradient_lower=np.full(len(neg),np.nan) if ell is None else ell)))
        payload=dict(registry_id=safe_name,source_registry=original_name,source_registry_identity=reg['identity'],
            analytic_fallback_registry_identity=areg['identity'],formula=FORMULA,margin_fraction=.01,
            float_policy=dict(longdouble_bits=np.finfo(np.longdouble).bits,mantissa_bits=np.finfo(np.longdouble).nmant,
                inner_bound_cushion=1e-12,negative_increment_relative_cushion=1e-12,nextafter_toward_zero=True),
            frozen_epoch=time.time(),source_weights_unchanged=True,selection_inputs='FROZEN_WEIGHTS_C0_NATIVE_SUPPORT_ONLY',
            entries=registry_rows,graph_identity=suite['graph'].identity,demand_identity=od.identity,support_identity=suite['support_identity'])
        payload['identity']=digest(payload)
        write_json(dst/(safe_name+'.json'),payload)
        new=SafeComposite(suite,safe_entries,payload,guard=ctx.guard)
        sv,sc,sh,sd=new.evaluate(flow,np.ones(len(flow)))
        de=next(e for e in safe_entries if e['region_id']==stop['region_id'])
        new_parity=torch_check(de['model_kind'],de['arrays'],u,np.ones(len(u))) if 'ICNN' in de['model_kind'] else {}
        pd.DataFrame(dict(native_index=np.arange(len(u)),pair_id=suite['pairs'].iloc[offender['start']:offender['stop']].pair_id.to_numpy(),
            flow=u,c0_seconds=offender['arrays']['c0'],native_scale=offender['arrays']['scale'],
            original_cost_seconds=cost[start:end],safe_cost_seconds=sc[start:end],
            original_hvp=hvp[start:end],safe_hvp=sh[start:end],derived_lower_bound_seconds=de['cost_lower_bound'])).to_csv(
                failure_csv/(safe_name+'.csv'),index=False,float_format='%.17g')
        replay=dict(original_registry=original_name,new_registry=safe_name,status='REPRODUCED_ORIGINAL_FAILURE_AND_SAFE_QUERY',
            saved_full_flow_entries=len(flow),saved_state_sha256=stop['input_state_hash'],source_model_sha256=offender['portable_sha256'],
            region_id=offender['region_id'],dimensions=len(u),native_coordinate=stop['native_coordinate'],
            packed_count=suite['packed_count'],region_start=offender['start'],global_edge=stop['edge_index'],
            own_coordinate_flow=float(u[stop['native_coordinate']]),nonzero_other_coordinates=int(np.count_nonzero(u)),
            original_cost_seconds=float(cost[stop['edge_index']]),safe_cost_seconds=float(sc[stop['edge_index']]),
            source_region_max_native_scale_ratio=float(np.max(u/offender['arrays']['scale'])),
            original_portfolio_max_native_scale_ratio=diag['maximum_native_scale_ratio'],
            original_complete_value=val,safe_complete_value=sv,original_full_hvp_hash=array_hash({'hvp':hvp}),safe_full_hvp_hash=array_hash({'hvp':sh}),
            original_parity=parity,safe_parity=new_parity,safe_query_diagnostic=sd,rho=de['transformation']['rho'])
        replays.append(replay)
        # Exact objective content identity is separate from portfolio labels and freeze timestamps.
        content=digest(dict(graph=suite['graph'].identity,demand=od.identity,support=suite['support_identity'],
            packed=array_hash(suite['packed'].arrays),models=[(e['region_id'],e['model_kind'],array_hash(e['arrays'])) for e in safe_entries],solver='FW',tol=.001))
        scenarios.append(dict(scenario_id=safe_name,registry_file=str(dst/(safe_name+'.json')),registry_identity=payload['identity'],
            complete_objective_graph_od_solver_identity=content,objective_identity=new.identity,dimensions=len(suite['pairs']),regions=len(entries)))
        coord=pd.concat(coordinates,ignore_index=True)
        for n,begin in enumerate(range(0,len(coord),20000)):
            coord.iloc[begin:begin+20000].to_csv(dst/(safe_name+f'_coordinate_bounds_{n:02d}.csv'),index=False,float_format='%.17g')
        print(safe_name,'frozen:',len(entries),'regions; saved native negative',replay['original_cost_seconds'],'safe',replay['safe_cost_seconds'],flush=True)
    pd.DataFrame(all_rows).to_csv(ctx.root/'regional_safety_transform.csv',index=False,float_format='%.17g')
    pd.DataFrame(tests).to_csv(ctx.root/'validation'/'native_safety_derivative_tests.csv',index=False,float_format='%.17g')
    write_json(ctx.root/'validation'/'ORIGINAL_FAILURE_REPLAY.json',replays)
    result=dict(status='PASS',scenarios=scenarios,original_failure_replays=3,native_derivative_cases=len(tests),coverage=coverage,
        frozen_before_new_national_outcomes=True,next_round_authorized=False)
    write_json(ctx.root/'SAFE_SCENARIO_FREEZE.json',result)
    return result


def load_safe(ctx,src,suite,name):
    if name not in REGISTRIES:raise ValueError('Unknown named safe registry')
    freeze=read_json(ctx.root/'SAFE_SCENARIO_FREEZE.json')
    sc=next(x for x in freeze['scenarios'] if x['scenario_id']==name)
    path=Path(sc['registry_file']);ctx.bind(path,'FIXED_R4_SAFETY_REGISTRY')
    reg=read_json(path);identity=reg.pop('identity')
    if digest(reg)!=identity or identity!=sc['registry_identity']:raise ValueError('Frozen safety registry changed')
    reg['identity']=identity
    originals,source_registry=load_registry(src,suite,REGISTRIES[name])
    if source_registry['identity']!=reg['source_registry_identity']:raise ValueError('Source registry changed')
    fallback,areg=load_registry(src,suite,ANALYTIC);fallback={e['region_id']:e for e in fallback}
    if areg['identity']!=reg['analytic_fallback_registry_identity']:raise ValueError('Analytic fallback source changed')
    frozen={e['region_id']:e for e in reg['entries']};entries=[]
    for source in originals:
        e,row,ell,neg=derived_entry(source,fallback[source['region_id']]);r=frozen[source['region_id']]
        for k in ('source_portable_sha256','transformed_arrays_sha256','rho','transformation_status','pair_order_hash','support_hash'):
            if r[k]!=row[k]:raise ValueError('Safety transform no longer reproduces: '+k)
        entries.append(e)
    return entries,reg
