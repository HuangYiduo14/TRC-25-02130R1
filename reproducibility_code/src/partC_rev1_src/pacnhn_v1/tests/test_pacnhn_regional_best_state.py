"""Synthetic regional output-selection regressions using the installed package.

Historical exact-source default equivalence is separately preserved in the
source-bound external repair evidence. Permanent tests have no output-root or
historical-source runtime dependency.
"""
import copy
import math
import numpy as np
import pytest
from partC_rev1_src.pacnhn_v1 import regional as M

def graph(m,tmp_path,zero_cost=False):
 p=tmp_path/'toy.npz'
 m.save_npz(p,tail=np.array([0,0,0],np.int64),head=np.array([1,1,1],np.int64),node_ids=np.array(['p0','p1']),edge_ids=np.array(['a','b','c']),base=np.array([1.,1.3,1.7])*(not zero_cost),k4=np.array([.8,.2,.1])*(not zero_cost),pair_source=np.array([0],np.int64),pair_target=np.array([1],np.int64),pair_keys=np.array([1],np.uint64),baseline_open=np.array([3.]),baseline_i95=np.array([4.]))
 return m.load_region_graph(p)

def run(m,g,**kwargs):
 return m.solve_label(g,[3.],max_seconds=30,max_iterations=6,relative_gap_tolerance=1e-30,**kwargs)

def clean_record(rec):
 rec=copy.deepcopy(rec)
 ignore={'solver_source_sha256','nested_direction_seconds','nested_line_search_seconds','complete_label_elapsed_seconds','complete_label_start_epoch','deadline_epoch','iterate_selection','selected_state_iteration','selected_state_hash','selection_basis','last_certified_iteration','last_certified_state_hash','last_certified_certificate','last_certified_cost_change','finite_certified_selection_candidates','best_state_snapshot_updates','nested_snapshot_seconds','additional_retained_state_vector_bytes'}
 return {k:v for k,v in rec.items() if k not in ignore}

@pytest.mark.parametrize('method,reset',[('FW',False),('CFW',False),('CFW',True)])
def test_default_is_explicit_last_certified(tmp_path,method,reset):
 g=graph(M,tmp_path)
 new=run(M,g,method=method,cfw_reset_on_bound=reset)
 explicit=run(M,g,method=method,cfw_reset_on_bound=reset,iterate_selection='LAST_CERTIFIED')
 assert clean_record(new['record'])==clean_record(explicit['record'])
 for k in ('flow','tau','edge_cost','c0','u','pair_keys'):
  np.testing.assert_array_equal(new[k],explicit[k])
 assert new['record']['additional_retained_state_vector_bytes']==0

@pytest.mark.parametrize('method,reset',[('FW',False),('CFW',True)])
def test_best_selection_matches_actual_fresh_oracle_and_bounds(tmp_path,method,reset):
 g=graph(M,tmp_path);out=run(M,g,method=method,cfw_reset_on_bound=reset,iterate_selection='BEST_CERTIFIED_RG');r=out['record']
 best=min(r['history'],key=lambda row:row['certificate']['RG_SPTT'])
 assert r['certificate']==best['certificate']
 assert r['reported_state_iteration']==r['selected_state_iteration']==best['iteration']
 assert r['last_certified_iteration']==r['history'][-1]['iteration']
 value,cost=M.PolynomialObjective(g.base,g.k4).value_cost(out['flow'])
 od,order=g.demand(out['u']);aon=M._load(g,od,cost,lambda:None)
 cert=M.certificate(out['flow'],cost,aon,value,has_potential=True)
 residual=M.conservation(g.network,out['flow'],od.balance(2),od.total)
 assert r['certificate']==dict(cert,**residual)
 assert r['selected_state_hash']==M.array_hash({'flow':out['flow']})
 np.testing.assert_array_equal(out['edge_cost'],cost)
 np.testing.assert_array_equal(out['tau'][order],aon['distances'])
 assert r['last_cost_change']==best['max_cost_change_since_previous_certificate']
 assert r['potential_upper']==value
 assert r['potential_error_bound']==max(0.,cert['raw_signed_gap'])+cert['arithmetic_allowance']
 assert r['potential_lower']==max(0.,value-r['potential_error_bound'],float(out['c0']@out['u']))
 assert r['additional_retained_state_vector_bytes']==out['flow'].nbytes+out['edge_cost'].nbytes+out['tau'].nbytes

# Controlled certificate values test selection boundaries. Actual-state certificate
# reproduction is tested independently above; these mocks do not assert real gaps.
def controlled_gaps(monkeypatch,gaps):
 original=M.certificate;seen=[]
 def cert(*args,**kwargs):
  c=original(*args,**kwargs);i=len(seen);c['RG_SPTT']=gaps[min(i,len(gaps)-1)];seen.append(c);return c
 monkeypatch.setattr(M,'certificate',cert);return seen

def test_earlier_better_state_not_last_and_no_aliases(tmp_path,monkeypatch):
 g=graph(M,tmp_path);seen=controlled_gaps(monkeypatch,[.1,5e-6,2e-5,4e-5])
 out=run(M,g,iterate_selection='BEST_CERTIFIED_RG');r=out['record'];h=r['history'][1]
 assert r['selected_state_iteration']==1<r['last_certified_iteration']
 assert r['quality']=='B' and r['admitted'] and r['status']=='ITERATION_LIMIT'
 assert r['accepted_updates']==6
 assert r['certificate']==h['certificate'] and r['certificate'] is not h['certificate']
 selected_hash=r['certificate']['state_hash'];selected_flow=out['flow'].copy()
 h['certificate']['state_hash']='mutated history';seen[1]['RG_SPTT']=99
 assert r['certificate']['state_hash']==selected_hash and r['certificate']['RG_SPTT']==5e-6
 np.testing.assert_array_equal(out['flow'],selected_flow)
 assert not np.shares_memory(out['flow'],out['edge_cost'])
 assert not np.shares_memory(out['u'],out['tau'])

@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_gap_never_selected(tmp_path,monkeypatch,bad):
 g=graph(M,tmp_path);controlled_gaps(monkeypatch,[bad,.2,.1,.3])
 out=run(M,g,iterate_selection='BEST_CERTIFIED_RG')
 assert math.isfinite(out['record']['certificate']['RG_SPTT'])
 assert out['record']['selected_state_iteration']!=0

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_no_finite_certificate_is_failure(tmp_path,monkeypatch,bad):
 g=graph(M,tmp_path);controlled_gaps(monkeypatch,[bad])
 with pytest.raises(FloatingPointError,match='No finite conserved certificate'):
  run(M,g,iterate_selection='BEST_CERTIFIED_RG')

@pytest.mark.parametrize('policy',['LAST_CERTIFIED','BEST_CERTIFIED_RG'])
def test_timeout_after_uncertified_accepted_update(tmp_path,monkeypatch,policy):
 g=graph(M,tmp_path);original=M._load;calls=0
 def load(*args,**kwargs):
  nonlocal calls
  calls+=1
  if calls==3:raise TimeoutError('Interrupted next certificate AON')
  return original(*args,**kwargs)
 monkeypatch.setattr(M,'_load',load)
 out=run(M,g,iterate_selection=policy);r=out['record']
 assert r['status']=='TIME_LIMIT' and r['accepted_updates']==1
 assert r['reported_state_iteration']==r['last_certified_iteration']==0
 np.testing.assert_array_equal(out['flow'],[3.,0.,0.])
 assert r['certificate']['state_hash']==M.array_hash({'flow':out['flow']})

@pytest.mark.parametrize('policy',['LAST_CERTIFIED','BEST_CERTIFIED_RG'])
def test_no_initial_certificate_timeout_is_failure(tmp_path,monkeypatch,policy):
 g=graph(M,tmp_path)
 monkeypatch.setattr(M,'_load',lambda *a,**k:(_ for _ in ()).throw(TimeoutError('initial timeout')))
 with pytest.raises(TimeoutError):run(M,g,iterate_selection=policy)

def test_timeout_keeps_earlier_best_complete_snapshot(tmp_path,monkeypatch):
 g=graph(M,tmp_path);controlled_gaps(monkeypatch,[.1,5e-6,4e-5])
 original=M._load;calls=0
 def load(*args,**kwargs):
  nonlocal calls
  calls+=1
  if calls==5:raise TimeoutError('next refresh')
  return original(*args,**kwargs)
 monkeypatch.setattr(M,'_load',load)
 out=run(M,g,iterate_selection='BEST_CERTIFIED_RG');r=out['record']
 assert r['status']=='TIME_LIMIT' and r['accepted_updates']==3
 assert r['selected_state_iteration']==1 and r['last_certified_iteration']==2
 assert r['certificate']==r['history'][1]['certificate']
 assert r['certificate']['state_hash']==M.array_hash({'flow':out['flow']})
 assert r['last_cost_change']==r['history'][1]['max_cost_change_since_previous_certificate']

def test_A_termination_preserved(tmp_path,monkeypatch):
 g=graph(M,tmp_path);controlled_gaps(monkeypatch,[.1,5e-7])
 out=M.solve_label(g,[3.],max_seconds=30,max_iterations=6,iterate_selection='BEST_CERTIFIED_RG')
 r=out['record'];assert r['status']=='CONVERGED' and r['quality']=='A'
 assert r['selected_state_iteration']==r['last_certified_iteration']==r['accepted_updates']==1

@pytest.mark.parametrize('q,zero_cost',[(0.,False),(3.,True)])
def test_zero_denominator_has_honest_A_state(tmp_path,q,zero_cost):
 g=graph(M,tmp_path,zero_cost=zero_cost)
 out=M.solve_label(g,[q],max_seconds=30,max_iterations=6,iterate_selection='BEST_CERTIFIED_RG')
 r=out['record'];assert r['quality']=='A' and r['status']=='CONVERGED'
 assert r['certificate']['RG_SPTT'] is None
 assert r['selection_basis']=='ZERO_SPTT_ARITHMETIC_ZERO_GAP'
 assert r['selected_state_iteration']==r['accepted_updates']==0

def test_equal_gaps_keep_first_exact_snapshot(tmp_path,monkeypatch):
 g=graph(M,tmp_path);controlled_gaps(monkeypatch,[.1])
 out=run(M,g,iterate_selection='BEST_CERTIFIED_RG')
 assert out['record']['selected_state_iteration']==0
 np.testing.assert_array_equal(out['flow'],[3.,0.,0.])

def test_invariant_failure_never_salvages_prior_snapshot(tmp_path,monkeypatch):
 g=graph(M,tmp_path);original=M.conservation;calls=0
 def check(*args,**kwargs):
  nonlocal calls
  calls+=1
  if calls>=3:raise ArithmeticError('synthetic conservation failure')
  return original(*args,**kwargs)
 monkeypatch.setattr(M,'conservation',check)
 with pytest.raises(ArithmeticError,match='conservation failure'):
  run(M,g,iterate_selection='BEST_CERTIFIED_RG')

@pytest.mark.parametrize('policy',['BEST','',None,True])
def test_explicit_policy_validation(tmp_path,policy):
 g=graph(M,tmp_path)
 with pytest.raises(ValueError,match='iterate_selection'):run(M,g,iterate_selection=policy)


@pytest.mark.parametrize('method,reset',[('FW',False),('CFW',False),('CFW',True)])
@pytest.mark.parametrize('policy',['LAST_CERTIFIED','BEST_CERTIFIED_RG'])
def test_exact_parallel_equilibrium_analytic_fixture(tmp_path,method,reset,policy):
 # tau_a=1+x_a^4, tau_b=2, q=2 gives x=(1,1), tau=2 and W=3.2.
 # This independent analytic fixture replaces a permanent historical-source
 # dependency, while external exact-source trajectory evidence is preserved.
 p=tmp_path/'analytic.npz'
 M.save_npz(p,tail=np.array([0,0],np.int64),head=np.array([1,1],np.int64),
  node_ids=np.array(['p0','p1']),edge_ids=np.array(['a','b']),
  base=np.array([1.,2.]),k4=np.array([1.,0.]),
  pair_source=np.array([0],np.int64),pair_target=np.array([1],np.int64),
  pair_keys=np.array([1],np.uint64),baseline_open=np.array([2.]),baseline_i95=np.array([3.]))
 g=M.load_region_graph(p)
 result=M.solve_label(g,[2.],max_seconds=30,max_iterations=20,
  method=method,cfw_reset_on_bound=reset,iterate_selection=policy)
 r=result['record']
 assert r['status']=='CONVERGED' and r['quality']=='A' and r['admitted']
 np.testing.assert_allclose(result['flow'],[1.,1.],atol=1e-9,rtol=0)
 np.testing.assert_allclose(result['tau'],[2.],atol=1e-9,rtol=0)
 np.testing.assert_array_equal(result['c0'],[1.])
 assert abs(r['potential_upper']-3.2)<1e-9
 assert r['selected_state_hash']==M.array_hash({'flow':result['flow']})
 assert r['reported_state_iteration']==r['selected_state_iteration']
 assert r['potential_lower']<=3.2+1e-9 and r['potential_upper']>=3.2-1e-9
