import time
import numpy as np
import pytest
from partC_rev1_src.pacnhn_v1.oracle import Network,Demand
from partC_rev1_src.pacnhn_v1.solver import PolynomialObjective,solve

class ToyContext:
    def guard(self,deadline):
        if time.time()>=deadline:raise TimeoutError()

@pytest.mark.parametrize('method',['FW','CFW','MSA'])
def test_parallel_analytic_equilibrium(method,tmp_path):
    g=Network(np.array([0,0]),np.array([1,1]),np.array([False,False]),np.array(['a','b']),np.array(['o','d']))
    od=Demand(np.array([0]),np.array([0,1]),np.array([1]),np.array([2.]),np.array([True]))
    obj=PolynomialObjective([1.,1.],[1.,1.]);start=time.time()
    r=solve(g,od,obj,tmp_path/method,method=method,run_start_epoch=start,deadline_epoch=start+60,tolerance=1e-8,ctx=ToyContext(),max_updates=10)
    assert r['status']=='CONVERGED';assert r['certificate']['interpretation']=='CONVEX_POTENTIAL_FW_GAP'
    with np.load(tmp_path/method/'final_flow.npz') as z:np.testing.assert_allclose(z['flow'],[1.,1.],atol=1e-8)

def test_cost_only_msa_has_no_potential_claim(tmp_path):
    class CostOnly:
        has_potential=False
        def cost(self,x):return 1.+np.asarray(x)**4
    g=Network(np.array([0,0]),np.array([1,1]),np.array([False,False]),np.array(['a','b']),np.array(['o','d']))
    od=Demand(np.array([0]),np.array([0,1]),np.array([1]),np.array([2.]),np.array([True]));start=time.time()
    r=solve(g,od,CostOnly(),tmp_path/'msa',method='MSA',run_start_epoch=start,deadline_epoch=start+60,tolerance=1e-8,ctx=ToyContext())
    assert r['certificate']['value'] is None
    assert r['certificate']['interpretation']=='POINTWISE_VI_RESIDUAL_NO_POTENTIAL'
    with pytest.raises(ValueError,match='Cost-only'):
        solve(g,od,CostOnly(),tmp_path/'fw',method='FW',run_start_epoch=start,deadline_epoch=start+60,tolerance=1e-8,ctx=ToyContext())
