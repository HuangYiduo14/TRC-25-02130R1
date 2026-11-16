import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from partC_rev1_src.pacnhn_recovery24 import assignments as a
from partC_rev1_src.pacnhn_recovery24.portfolio import FrozenComposite, SCHEMA, _digest
from partC_rev1_src.pacnhn_v1.control import array_hash, sha, write_json
from partC_rev1_src.pacnhn_v1.oracle import Network, Demand
from partC_rev1_src.pacnhn_recovery24.tests.test_portfolio import ToyPotential, ToyCost, support, make_portfolio


def settings():
    return dict(cap_seconds={'ICNN':1500,'POWER4':900,'XGBOOST':600,'FFTT':300},
                sealing_reserve_seconds=600,per_query_seconds=2700,released_slack_fraction=.25,
                RG_SPTT=.001)


class Context:
    def __init__(self,tmp_path):
        self.root=tmp_path/'new';self.source_root=tmp_path/'old'
        self.root.mkdir();self.source_root.mkdir();self.end=time.time()+120
        self.c=dict(phase_id='PACNHN_RECOVERY24',run_id='toy',assignment=settings(),
                    resources={'regional_workers':1},source_bindings={})
    def require_action(self,action):
        assert action=='assignments';return {'science_deadline_epoch':self.end}
    def guard(self,deadline=None):
        if time.time()>=min(self.end,deadline or self.end):raise TimeoutError('expired')
    def bind(self,path,role,expected=None):
        actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('changed')
        return actual
    def flush_bindings(self):pass


def tiny_graph():
    return Network(np.array([0,0]),np.array([1,1]),np.zeros(2,bool),
                   np.array(['a','b']),np.array(['o','d']))


def tiny_od():
    return Demand(np.array([0]),np.array([0,1]),np.array([1]),np.array([2.]),np.array([True]))


def test_exact_matrix_and_solver_semantics():
    rows=a.matrix()
    assert len(rows)==46 and len({r['case_id'] for r in rows})==46
    assert sum(r['network']=='P' for r in rows)==6
    assert sum(r['reuse_candidate'] for r in rows)==16
    assert sum(not r['reuse_candidate'] for r in rows)==30
    assert all(r['solver']=='MSA' for r in rows if r['family']=='XGBOOST')
    assert {r['scenario'] for r in rows}=={'open','i95'}
    for sid in a.SUITES:assert sum(r['suite_id']==sid for r in rows)==8


def test_budget_protects_all_later_cases_and_bounds_slack():
    rows=[r for r in a.matrix() if not r['reuse_candidate']]
    cfg=settings();initial=sum(cfg['cap_seconds'][r['family']] for r in rows)
    assert initial==30000
    cap=a.query_cap(rows[0],cfg,32400,rows[1:])
    assert cap==cfg['cap_seconds'][rows[0]['family']]+450
    assert a.query_cap(rows[0],cfg,60600,rows[1:])<=2700
    half=a.query_cap(rows[0],cfg,15600,rows[1:])
    assert half==cfg['cap_seconds'][rows[0]['family']]/2
    assert a.query_cap(rows[0],cfg,500,rows[1:])==0


@pytest.mark.parametrize('key,value',[('per_query_seconds',2701),('per_query_seconds',0),
    ('sealing_reserve_seconds',-1),('released_slack_fraction',2)])
def test_bad_budgets_rejected(key,value):
    cfg=settings();cfg[key]=value;row=[x for x in a.matrix() if not x['reuse_candidate']][0]
    with pytest.raises(ValueError):a.query_cap(row,cfg,100000,[])


def reused():
    row=a.matrix()[0];flow=np.array([1.,1.])
    cert=dict(fresh=True,state_hash=array_hash({'flow':flow}),RG_SPTT=.0001,
              raw_signed_gap=.001,arithmetic_allowance=1e-10)
    return row,dict(case={k:v for k,v in row.items() if k!='reuse_candidate'},status='CONVERGED',
                    final_certificate_fresh=True,graph_identity='graph',nominal_tolerance=.001,
                    method='CFW',certificate=cert),flow


@pytest.mark.parametrize('key,value',[('status','TIME_LIMIT'),('method','MSA'),
    ('graph_identity','changed'),('nominal_tolerance',.01),('final_certificate_fresh',False)])
def test_reuse_cannot_accept_other_query(key,value):
    row,result,_=reused();a.validate_reused_metadata(row,result,copy.deepcopy(result),'graph',.001)
    result[key]=value
    with pytest.raises(ValueError):a.validate_reused_metadata(row,result,copy.deepcopy(result),'graph',.001)


def test_reuse_checks_aggregate_and_zero_reference_certificate():
    row,r,_=reused();other=copy.deepcopy(r);other['case']['solver']='FW'
    with pytest.raises(ValueError):a.validate_reused_metadata(row,r,other,'graph',.001)
    r['certificate'].update(RG_SPTT=None,raw_signed_gap=0.)
    a.validate_reused_metadata(row,r,copy.deepcopy(r),'graph',.001)
    r['certificate']['raw_signed_gap']=1.
    with pytest.raises(ValueError):a.validate_reused_metadata(row,r,copy.deepcopy(r),'graph',.001)


def test_native_certificate_exact_membership_and_closed_zero():
    _,r,flow=reused();served=np.array([True,False]);closed=np.array([False,False])
    a.validate_native_state(r,flow,served,served.copy(),closed,closed=True)
    with pytest.raises(ValueError):a.validate_native_state(r,flow,served,~served,closed,closed=False)
    with pytest.raises(ValueError):a.validate_native_state(r,flow+.1,served,served,closed,closed=False)
    with pytest.raises(ValueError):a.validate_native_state(r,flow,served,served,np.array([True,False]),closed=True)
    with pytest.raises(ValueError):a.validate_native_state(r,flow,served.astype(int),served,closed,closed=False)


def test_portfolio_self_identity_and_campaign_binding(tmp_path):
    ctx=Context(tmp_path)
    data=dict(schema=SCHEMA,phase_id=ctx.c['phase_id'],run_id='toy',source_root=str(ctx.source_root),
              portfolios=make_portfolio(support(),[]))
    data['identity']=_digest(data);p=ctx.root/'PORTFOLIOS.json';write_json(p,data)
    assert a.read_portfolios(ctx,p)[0]==data
    data['portfolios'][0]['regions'][0]['component']='XGBOOST';write_json(p,data)
    with pytest.raises(ValueError):a.read_portfolios(ctx,p)


@pytest.mark.parametrize('error,status',[(MemoryError('RAM'),'UNAVAILABLE_CURRENT_RAM'),
    (TimeoutError('cap'),'TIME_LIMIT_DURING_INPUT_LOAD')])
def test_input_failures_get_honest_new_disposition(tmp_path,monkeypatch,error,status):
    ctx=Context(tmp_path);p=ctx.source_root/'keep';p.write_bytes(b'old bytes')
    before=p.read_bytes();row=next(r for r in a.matrix() if r['family']=='ICNN')
    def fail(*args,**kwargs):raise error
    monkeypatch.setattr(a,'load_objective',fail)
    result=a.run_case(ctx,row,{},'a'*64,deadline=ctx.end,cap_seconds=30)
    assert result['status']==status and not result['final_certificate_fresh']
    assert p.read_bytes()==before and list(ctx.source_root.iterdir())==[p]
    with pytest.raises(FileExistsError):a.run_case(ctx,row,{},'a'*64,deadline=ctx.end,cap_seconds=30)


def test_uncertified_timeout_does_not_require_native_flow(tmp_path,monkeypatch):
    ctx=Context(tmp_path);row=next(r for r in a.matrix() if r['family']=='ICNN')
    graph,od=tiny_graph(),tiny_od()
    monkeypatch.setattr(a,'load_objective',lambda *args,**kwargs:(graph,od,{'closed_mask':np.zeros(2,bool)},object()))
    def timeout(*args,**kwargs):
        root=Path(args[3]);root.mkdir(parents=True);return dict(status='TIME_LIMIT',final_certificate_fresh=False)
    monkeypatch.setattr(a,'solve',timeout)
    result=a.run_case(ctx,row,{},'a'*64,deadline=ctx.end,cap_seconds=30)
    assert result['status']=='TIME_LIMIT' and not result['final_certificate_fresh']
    assert not (ctx.root/'assignments'/row['case_id']/'final_flow.npz').exists()


@pytest.mark.parametrize('method,cost_only',[('FW',False),('MSA',True)])
def test_real_solver_with_fixed_tiny_portfolio(tmp_path,method,cost_only):
    ctx=Context(tmp_path);graph=tiny_graph();od=tiny_od()
    model=(ToyCost if cost_only else ToyPotential)([1.,1.],[1.,1.])
    obj=FrozenComposite([1.,1.],[0.,0.],[0,1],[(np.array([0,1]),model)],cost_only=cost_only)
    p=ctx.source_root/'keep';p.write_bytes(b'unchanged');before=sha(p)
    root=ctx.root/method;start=time.time()
    result=a.solve(graph,od,obj,root,method=method,run_start_epoch=start,deadline_epoch=start+90,
                   tolerance=1e-8,ctx=ctx,workers=1,max_updates=5)
    assert result['status']=='CONVERGED' and result['final_certificate_fresh']
    with np.load(root/'final_flow.npz',allow_pickle=False) as z:flow=z['flow']
    np.testing.assert_allclose(flow,[1.,1.],atol=1e-8)
    assert result['certificate']['state_hash']==array_hash({'flow':flow})
    expected='POINTWISE_VI_RESIDUAL_NO_POTENTIAL' if cost_only else 'CONVEX_POTENTIAL_FW_GAP'
    assert result['certificate']['interpretation']==expected
    assert (result['certificate']['value'] is None)==cost_only
    assert sha(p)==before and list(ctx.source_root.iterdir())==[p]


def test_existing_matrix_is_not_overwritten(tmp_path):
    ctx=Context(tmp_path);p=ctx.root/'ASSIGNMENT_MATRIX.json';p.write_text('preserve')
    with pytest.raises(FileExistsError):a.run(ctx,ctx.root/'nonexistent',deadline=ctx.end)
    assert p.read_text()=='preserve'


def test_reference_served_population_reconstructed_from_original_od(tmp_path):
    ctx=Context(tmp_path);root=ctx.source_root/'references/open';root.mkdir(parents=True)
    od=tiny_od();result=dict(status='CONVERGED',final_certificate_fresh=True,
        certificate=dict(fresh=True,RG_SPTT=.00001),OD_identity=od.identity)
    p=root/'RESULT.json';write_json(p,result);ctx.c['source_bindings'][str(p)]=sha(p)
    np.savez(root/'coverage.npz',serve=np.array([True]))
    assert a._scenario_reference(ctx,'open',original_od=od)[0]==result
    np.savez(root/'coverage.npz',serve=np.array([False]))
    with pytest.raises(ValueError,match='exact OD'):a._scenario_reference(ctx,'open',original_od=od)


def test_reuse_real_saved_tiny_p_state_without_source_writes(tmp_path):
    ctx=Context(tmp_path);graph,od=tiny_graph(),tiny_od();row=a.matrix()[0]
    directory=ctx.source_root/'assignments'/row['case_id'];directory.mkdir(parents=True)
    topo=ctx.source_root/'topology/RESULT.json';write_json(topo,dict(graph_identity=graph.identity))
    ctx.c['source_bindings'][str(topo)]=sha(topo)
    _,result,flow=reused();result.update(graph_identity=graph.identity,OD_identity=od.identity,
        reported_state_iteration=1,elapsed_seconds=1.)
    write_json(directory/'RESULT.json',result)
    write_json(directory/'HISTORY.json',[dict(state_hash=result['certificate']['state_hash'],iteration=1)])
    np.savez(directory/'final_flow.npz',flow=flow);np.savez(directory/'coverage.npz',serve=od.serve)
    ref=ctx.source_root/'references/open';ref.mkdir(parents=True)
    rr=dict(result);rr['certificate']=dict(result['certificate'],RG_SPTT=.00001)
    write_json(ref/'RESULT.json',rr);np.savez(ref/'coverage.npz',serve=od.serve)
    ctx.c['source_bindings'][str(ref/'RESULT.json')]=sha(ref/'RESULT.json')
    before={str(p):sha(p) for p in ctx.source_root.rglob('*') if p.is_file()}
    common=(graph,od,{'closed_mask':np.zeros(2,bool)}, {})
    got=a.reuse_case(ctx,row,[result],support(),deadline=ctx.end,common=common)
    assert got['execution_status']=='REUSED_VERIFIED_IMMUTABLE_RESULT'
    assert not got['new_scientific_execution'] and got['historical_query_elapsed_seconds']==1.
    assert {str(p):sha(p) for p in ctx.source_root.rglob('*') if p.is_file()}==before


def test_full_46_case_accounting_and_48_reporting_rows(tmp_path,monkeypatch):
    ctx=Context(tmp_path);ctx.end=time.time()+32400
    monkeypatch.setattr(a,'read_portfolios',lambda *x:({'support_freeze_sha256':'s'},'p'))
    monkeypatch.setattr(a,'_bound_metadata',lambda *x:[])
    monkeypatch.setattr(a,'sha',lambda p:'s')
    monkeypatch.setattr(a,'_common',lambda c:())
    monkeypatch.setattr(a,'reuse_case',lambda c,r,*x,**kw:dict(case=r,status='CONVERGED',new_scientific_execution=False))
    caps=[]
    def query(c,row,*args,**kwargs):
        caps.append(kwargs['cap_seconds']);return dict(case=row,status='TIME_LIMIT',final_certificate_fresh=True)
    monkeypatch.setattr(a,'run_case',query)
    monkeypatch.setattr(a,'_scenario_reference',lambda *x,**kw:({'status':'CONVERGED'},np.ones(1,bool)))
    monkeypatch.setattr(a,'_artifacts',lambda *x:{})
    rows=a.run(ctx,ctx.root/'portfolio',deadline=ctx.end)
    assert len(rows)==48 and len(caps)==30 and all(0<c<=2700 for c in caps)
    terminal=json.loads((ctx.root/'ASSIGNMENT_STAGE_RESULT.json').read_text())
    assert terminal['rows']==46 and not terminal['all_converged']
    assert terminal['statuses']=={'CONVERGED':16,'TIME_LIMIT':30}
