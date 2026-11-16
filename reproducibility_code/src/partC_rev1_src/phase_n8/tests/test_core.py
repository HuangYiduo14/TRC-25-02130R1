"""Independent numerical/oracle and process-lifecycle regressions for N8R2."""
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import psutil
import pytest
from scipy.special import expit

from partC_rev1_src.phase5.campaign_runtime import PackedRuntime
from partC_rev1_src.phase67.portable import predict_arrays
from partC_rev1_src.phase_n8.common import array_hash, read_json, write_json
from partC_rev1_src.phase_n8.oracle import Network, Demand, aon, conservation, load_range
from partC_rev1_src.phase_n8.objective import Composite, UnsafeCost, validate_arrays
from partC_rev1_src.phase_n8.solver import (certificate, line_search, save_checkpoint,
    reopen_checkpoint, solve, meets_tolerance)
from partC_rev1_src.phase_n8.supervise import run_supervised


def graph(tail, head, n=None, blocked=None):
    n = n or int(max(max(tail), max(head))+1)
    return Network(np.array(tail), np.array(head), np.zeros(n, bool) if blocked is None else blocked,
                   np.asarray([f"edge{i:03d}" for i in range(len(tail))]),
                   np.asarray([f"node{i}" for i in range(n)]))


def od(origins=(0,), offsets=(0, 1), destinations=(1,), demands=(3,), serve=None):
    return Demand(np.array(origins), np.array(offsets), np.array(destinations),
                  np.array(demands), np.ones(len(demands), bool) if serve is None else serve)


def packed(base, k, *, charge=None):
    n = len(base)
    zero = np.zeros(n)
    return PackedRuntime(dict(tau_s=np.array(base, float), k4=np.array(k, float),
        fixed_internal_H_s=zero if charge is None else np.array(charge, float),
        tail_half_charge_s=zero, head_half_charge_s=zero, structural_fixed_s=zero,
        domain_Q=np.full(n, np.inf), is_fitted=np.zeros(n, bool)))


def objective(g, base, k, c0=(), entries=None, charge=None):
    suite = dict(graph=g, packed=packed(base, k, charge=charge), packed_count=len(base),
                 pairs=pd.DataFrame(dict(c0_seconds=np.array(c0))), support_identity="fixture")
    return Composite(suite, entries or [], dict(identity="fixture", registry_id="fixture"))


def nn(m=3, legacy=False, power=1):
    rng = np.random.default_rng(419)
    a = dict(c0=np.arange(m, dtype=float)+10, scale=np.arange(m, dtype=float)+2,
             A1=rng.normal(0, .2, (4, m)), A2=rng.normal(0, .2, (4, m)),
             W2=rng.uniform(0, .2, (4, 4)), b1=rng.normal(0, .1, 4), b2=rng.normal(0, .1, 4),
             v=rng.uniform(.1, .4, 4))
    h1=np.logaddexp(0,a['b1']); l2=a['W2']@h1+a['b2'];h2=np.logaddexp(0,l2)
    k=expit(l2)*a['v']*h2**(power-1)
    a.update(anchor_value=np.array(np.sum(a['v']*h2**power/power)),
             anchor_gradient=(expit(a['b1'])*(k@a['W2']))@a['A1']+k@a['A2'])
    if legacy:
        a.update(Vscale=np.array(3.), gain=np.array(.4))
    else:
        a.update(coef=np.full(m,.03),eta_unit=np.array(2.),residual_unit=np.array(2.),
                 gate=np.array(.5),head_power=np.array(power))
    return a


@pytest.mark.parametrize("kind", ["FREEFLOW_LINEAR", "OLD_POWER4_NNLS", "NEW_POWER4_RELATIVE_NNLS",
                                  "OLD_ICNN_FROZEN", "POWER4_SKIP_ICNN32"])
@pytest.mark.parametrize("m", [1, 3, 28])
def test_native_derivatives(kind, m):
    a = nn(m, legacy=kind=="OLD_ICNN_FROZEN")
    if kind == "FREEFLOW_LINEAR": a = dict(c0=a['c0'])
    elif "ICNN" not in kind: a = dict(c0=a['c0'],scale=a['scale'],eta=np.full(m,.25))
    validate_arrays(kind,a)
    u=np.arange(m,dtype=float)*.02+1.
    d=np.cos(np.arange(m))
    eps=1e-4
    val,c,h=predict_arrays(kind,a,u,d)
    vp,cp=predict_arrays(kind,a,u+eps*d)
    vm,cm=predict_arrays(kind,a,u-eps*d)
    assert np.isclose((vp-vm)/(2*eps),c@d,rtol=2e-6,atol=2e-7)
    assert np.allclose((cp-cm)/(2*eps),h,rtol=2e-5,atol=2e-7)
    assert d@h >= -1e-10


def test_zero_anchor_one_sided_and_head5():
    for power in [1,5]:
        a=nn(power=power);u=np.zeros(3);d=np.array([1.,.3,.1]);eps=1e-6
        v,c,h=predict_arrays('POWER4_SKIP_ICNN32',a,u,d)
        vp,cp=predict_arrays('POWER4_SKIP_ICNN32',a,u+eps*d)
        assert abs(v)<1e-12
        assert np.allclose(c,a['c0'],atol=1e-12)
        assert np.isclose((vp-v)/eps,c@d,rtol=1e-6)
        assert np.allclose((cp-c)/eps,h,rtol=2e-5,atol=1e-7)


@pytest.mark.parametrize('mutation',["scale","shape","negative_weight","nonfinite","anchor"])
def test_invalid_model_boundary(mutation):
    a=nn()
    if mutation=='scale':a['scale'][0]=0
    if mutation=='shape':a['A1']=a['A1'][:,:2]
    if mutation=='negative_weight':a['W2'][0,0]=-1
    if mutation=='nonfinite':a['v'][0]=np.nan
    if mutation=='anchor':a['anchor_gradient']+=1
    with pytest.raises(ValueError):validate_arrays('POWER4_SKIP_ICNN32',a)


def test_full_coupling_hvp_and_repeated_region_accounting():
    g=graph([0,1,2],[1,2,3]);a=nn(2)
    e=dict(model_kind='POWER4_SKIP_ICNN32',arrays=a,start=0,stop=2)
    o=objective(g,[1.],[.02],a['c0'],[e],charge=[7.])
    f=np.array([1.,2.,3.]);d=np.array([.2,.1,-.3]);eps=1e-4
    val,c,h,_=o.evaluate(f,d)
    vref,cref,href=predict_arrays(e['model_kind'],a,f[1:],d[1:])
    assert np.isclose(val,8*f[0]+.02*f[0]**5/5+vref)
    assert np.allclose(c[1:],cref) and np.allclose(h[1:],href)
    assert abs(predict_arrays(e['model_kind'],a,f[1:],np.array([1.,0.]))[2][1])>1e-8
    vp,cp,_,_=o.evaluate(f+eps*d);vm,cm,_,_=o.evaluate(f-eps*d)
    assert np.isclose((vp-vm)/(2*eps),c@d,rtol=1e-6)
    assert np.allclose((cp-cm)/(2*eps),h,rtol=1e-5,atol=1e-8)


def test_zero_cost_scc_and_source_predecessor_safety():
    g=graph([0,0,1,2,2,1],[1,2,2,1,3,0]);d=od(destinations=[3],demands=[7])
    r=aon(g,d,np.array([0.,0.,0.,0.,1.,0.]),return_distances=True,batch_origins=1)
    assert r['sptt']==7 and r['loaded_sptt']==7
    assert r['flow'][5]==0
    conservation(g,r['flow'],d.balance(4),7)


def test_parallel_member_and_exact_near_tie():
    g=graph([0,0],[1,1]);d=od()
    r=aon(g,d,np.array([1.+1e-12,1.]))
    assert np.array_equal(r['flow'],[0,3])
    assert r['sptt']==3
    exact=aon(g,d,np.ones(2))
    assert np.array_equal(exact['flow'],[3,0])


def test_centroid_nontransit_and_unreachable():
    g=graph([0,1,0],[1,2,2],blocked=np.array([False,True,False]))
    d=od(destinations=[2])
    r=aon(g,d,np.array([0.,0.,5.]),return_distances=True)
    assert r['sptt']==15
    g2=graph([0,1],[1,2],blocked=np.array([False,True,False]))
    r=aon(g2,d,np.zeros(2),return_distances=True)
    assert r['reachable_demand']==0 and np.isinf(r['distances'][0])


def test_duplicate_mapped_destinations_and_empty_origin_group():
    g=graph([0,1],[1,2]);d=od(origins=[0,1],offsets=[0,0,2],destinations=[2,2],demands=[2,3])
    r=aon(g,d,np.ones(2),return_distances=True)
    assert r['sptt']==5 and np.array_equal(r['flow'],[0,5])
    conservation(g,r['flow'],d.balance(3),5)


def test_zero_sptt_and_signed_negative_gap():
    f=np.array([1.]);c=np.array([0.]);h=array_hash({'flow':f})
    cert=certificate(f,c,dict(sptt=0.,loaded_sptt=0.),0.,h)
    assert cert['RG_SPTT'] is None and meets_tolerance(cert,.001)
    with pytest.raises(ArithmeticError):certificate(f,np.array([1.]),dict(sptt=2.,loaded_sptt=2.),1.,h)
    tiny=certificate(f,np.array([1.]),dict(sptt=1.+1e-12,loaded_sptt=1.+1e-12),1.,h)
    assert tiny['raw_signed_gap']<0 and tiny['roundoff_negative']


def test_constant_cost_search_and_zero_curvature():
    g=graph([0,0],[1,1]);o=objective(g,[1.,2.],[0.,0.])
    f=np.array([0.,3.]);s=np.array([3.,0.]);v,c,_,_=o.evaluate(f)
    alpha,_,_,_=line_search(o,f,s,v,c)
    assert alpha==1
    assert np.array_equal(o.evaluate(f,np.ones(2))[2],np.zeros(2))


class Guard:
    def guard(self,deadline=None):
        if deadline is not None and time.time()>=deadline:raise TimeoutError('fixture deadline')


def test_feasible_fw_known_parallel_optimum(tmp_path):
    g=graph([0,0],[1,1]);d=od();o=objective(g,[1.,1.],[1.,16.]);t=time.time()
    r=solve(g,d,o,d.balance(2),tmp_path,identity={'case':'known'},
            clocks=dict(run_start_epoch=t,run_deadline_epoch=t+180),ctx=Guard(),tol=1e-10)
    assert r['stop_reason']=='CONVERGED'
    assert r['final_certificate_fresh'] and r['accepted_iterations']==1
    with np.load(tmp_path/'final_flow.npz') as z:assert np.allclose(z['flow'],[2.,1.],atol=1e-8)
    h=read_json(tmp_path/'HISTORY.json')
    assert all(a['objective']>=b['objective']-1e-8 for a,b in zip(h,h[1:]))


def test_timeout_keeps_own_fresh_certificate(tmp_path):
    g=graph([0,0],[1,1]);d=od();o=objective(g,[1.,1.],[1.,16.]);t=time.time()
    r=solve(g,d,o,d.balance(2),tmp_path,identity={'case':'timeout'},
            clocks=dict(run_start_epoch=t,run_deadline_epoch=t+30),ctx=Guard())
    assert r['stop_reason']=='TIME_LIMIT' and r['final_certificate_fresh']
    assert r['accepted_iterations']==0 and r['final_certificate']['RG_SPTT']>.001


def test_checkpoint_hash_clock_and_stale_certificate(tmp_path):
    f=np.array([1.,2.]);ident={'graph':'x'};clocks={'start':100,'deadline':200}
    cert={'state_hash':array_hash({'flow':f})}
    save_checkpoint(tmp_path,f,0,ident,clocks,cert)
    loaded,meta=reopen_checkpoint(tmp_path,ident,clocks)
    assert np.array_equal(loaded,f) and meta['clocks']==clocks
    with pytest.raises(ValueError):reopen_checkpoint(tmp_path,ident,dict(start=101,deadline=201))
    with pytest.raises(ValueError):reopen_checkpoint(tmp_path,{'graph':'changed'},clocks)
    with pytest.raises(ValueError):save_checkpoint(tmp_path,f+1,1,ident,clocks,cert)
    ptr=read_json(tmp_path/'LATEST_CHECKPOINT.json');p=Path(ptr['path'])/'flow.npz'
    with p.open('ab') as out:out.write(b'changed')
    with pytest.raises(ValueError):reopen_checkpoint(tmp_path,ident,clocks)


def test_atomic_windows_write_and_external_watchdog(tmp_path):
    path=tmp_path/'state.json'
    for i in range(20):write_json(path,{'iteration':i})
    assert read_json(path)['iteration']==19
    assert not list(tmp_path.glob('*.pending.*'))
    result=run_supervised([sys.executable,'-c','import time; time.sleep(30)'],os.environ.copy(),
          tmp_path/'child.log',deadline=time.time()+.8,rss_cap=1024**3,available_floor=0)
    assert result['reason']=='WATCHDOG_TIME_LIMIT' and result['no_orphan_child']
    assert not psutil.pid_exists(result['pid'])


def test_negative_nonfinite_actual_costs_are_not_clipped():
    g=graph([0,1],[1,2]);a=nn(1);a['c0']=np.array([0.]);a['anchor_gradient']+=100
    # Deliberately invalid anchor must be rejected before runtime.
    with pytest.raises(ValueError):validate_arrays('POWER4_SKIP_ICNN32',a)
    with pytest.raises(ValueError):aon(g,od(destinations=[2]),np.array([0.,-1.]))
    with pytest.raises(ValueError):aon(g,od(destinations=[2]),np.array([0.,np.inf]))
    o=objective(g,[1.,1.],[1e300,1e300])
    with pytest.raises(UnsafeCost) as e:o.evaluate(np.array([1e10,1e10]))
    assert e.value.reason=='STOP_NONFINITE' and e.value.flow is not None


def test_invalid_maps_and_directed_graph():
    with pytest.raises(ValueError):graph([0],[4],n=2)
    g=graph([1],[0]);r=aon(g,od(),np.ones(1),return_distances=True)
    assert r['reachable_demand']==0
    with pytest.raises(ValueError):Demand([0],[1,2],[1,1],[1,1],[True,True])


def test_multiple_native_origin_attachments_are_one_way():
    # Native origin terminals 0/1, destination terminal 2; OD ports 3/4.
    # The one original OD can use either native origin role, but ports do not transit.
    from partC_rev1_src.phase_n8.native import attach_roles
    g=graph([0,1],[2,2]);s=dict(graph=g,
        nodes=pd.DataFrame(dict(terminal_type=['ORIGIN_CENTROID','ORIGIN_CENTROID','DESTINATION_CENTROID'])),
        fixed_x45=np.zeros(2))
    ports=dict(port_node_ids=np.array(['origin_port','destination_port']),
               port_edge_ids=np.array(['attach0','attach1','attach2']),
               port_tail=np.array([3,3,2]),port_head=np.array([0,1,4]))
    attach_roles(s,ports)
    r=aon(s['graph'],od(origins=[3],destinations=[4]),np.array([5.,1.,0.,0.,0.]))
    assert r['sptt']==3 and r['flow'][1]==3 and r['flow'][0]==0
    conservation(s['graph'],r['flow'],od(origins=[3],destinations=[4]).balance(5),3)


def test_exact_physical_self_od_uses_empty_path_without_support_promotion():
    from partC_rev1_src.phase_n8.native import mapped_demand
    original=dict(origin_group_node_idx=np.array([2,3]),origin_offsets=np.array([0,2,3]),
                  destination_node_idx=np.array([2,3,2]),demand=np.array([7.,5.,2.]))
    om=np.array([-1,-1,0,1]);dm=np.array([-1,-1,2,3])
    d=mapped_demand(original,om,dm,np.ones(3,bool))
    assert np.array_equal(d.destinations,[0,3,2])
    # Different physical IDs do not become empty paths even if coarse parents coincide.
    g=graph([0,1],[3,2],n=4)
    result=aon(g,d,np.array([1.,2.]))
    assert result['reachable_demand']==14 and result['sptt']==9
    assert np.array_equal(result['flow'],[5.,2.])
