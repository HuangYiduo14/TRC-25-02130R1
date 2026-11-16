"""Distinct estimands and PPS duplicate draw units are scientific contracts."""
import numpy as np
import pandas as pd
import pytest
from partC_rev1_src.phase_n8.oracle import Demand
from partC_rev1_src.phase_n8_r4.path_metrics import summarize,sample_demand


def test_ratio_and_mean_relative_are_not_interchanged():
    f=pd.DataFrame(dict(D_phys_s=[1.,100.],D_surrogate_s=[2.,100.]))
    r=summarize(f,np.ones(2),'PPS_EQUAL_DRAWS')
    assert r['WREL']==pytest.approx(1/101)
    assert r['WMEANREL_1S']==.5
    assert r['top1_WREL_numerator_share']==1.
    assert r['SENSITIVITY_EXCLUDING_TOP1_WREL']==0.


def test_duplicate_pps_draws_keep_multiplicity_without_second_demand_weight():
    f=pd.DataFrame(dict(D_phys_s=[1.,1.,100.],D_surrogate_s=[2.,2.,100.]))
    r=summarize(f,np.ones(3),'PPS_EQUAL_DRAWS')
    assert r['WREL']==pytest.approx(2/102)
    assert r['WMEANREL_1S']==pytest.approx(2/3)
    assert r['valid_paired_rows']==3


def test_sample_mapping_preserves_duplicate_destinations_and_self_aliases():
    original=dict(origin_offsets=np.array([0,2,4]),demand=np.array([2.,3.,4.,5.]))
    full=Demand([10,20],[0,2,4],[10,12,20,22],[2.,3.,4.,5.],[True]*4)
    frame=pd.DataFrame(dict(accepted_od_row=[0,1,1,3],demand=[2.,3.,3.,5.]))
    sample=sample_demand(original,full,frame)
    np.testing.assert_array_equal(sample.origins,[10,20])
    np.testing.assert_array_equal(sample.offsets,[0,3,4])
    np.testing.assert_array_equal(sample.destinations,[10,12,12,22])
    assert sample.total==13.
