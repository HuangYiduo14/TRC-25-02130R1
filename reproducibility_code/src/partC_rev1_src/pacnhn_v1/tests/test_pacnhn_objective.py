import numpy as np
import pytest
from partC_rev1_src.pacnhn_v1.models import Power4Model
from partC_rev1_src.pacnhn_v1.objective import CompositeObjective

def test_composite_replaces_regional_potential_without_double_charge():
    model=Power4Model([3.,4.],[2.,1.],[.7,.2])
    obj=CompositeObjective([1.,99.,99.,0.],[.1,0.,0.,0.],[1,2],[(np.array([1,2]),model)],family='POWER4')
    x=np.array([2.,3.,4.,2.]);v,c=obj.value_cost(x)
    mv,mc=model.value_cost(x[1:3])
    assert v==pytest.approx(2.+.1*2**5/5+mv)
    np.testing.assert_allclose(c,[2.6,*mc,0.])
    eps=1e-5
    fd=np.array([(obj.value_cost(x+eps*np.eye(4)[j])[0]-obj.value_cost(x-eps*np.eye(4)[j])[0])/(2*eps) for j in range(4)])
    np.testing.assert_allclose(c,fd,rtol=1e-8)
    with pytest.raises(ValueError,match='exactly one'):
        CompositeObjective([1.,2.,3.],[0.,0.,0.],[1,2],[(np.array([1]),Power4Model([1.],[1.],[0.]))],family='ICNN')

def test_cost_only_composite_rejects_invented_potential():
    class Direct:
        has_potential=False
        def cost(self,x):return np.array([2.+x.sum(),4.+x[0]])
    obj=CompositeObjective([1.,99.,99.],[0.,0.,0.],[1,2],[(np.array([1,2]),Direct())],family='XGBOOST')
    np.testing.assert_array_equal(obj.cost(np.array([1.,2.,3.])),[1.,7.,6.])
    with pytest.raises(TypeError,match='no potential'):obj.value_cost(np.ones(3))
