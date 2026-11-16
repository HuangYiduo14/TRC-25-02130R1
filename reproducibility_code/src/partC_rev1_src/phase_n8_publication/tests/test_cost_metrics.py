import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8_publication.metrics import compare_vectors
from partC_rev1_src.phase_n8_publication.cost_metrics import error_rows,regional_aggregate


def test_true_cost_wmape_is_ratio_of_sums_not_mean_percentage():
    a=dict(u=np.array([[0.,1.],[1.,0.]]),t=np.array([[1.,1.],[100.,100.]]),L=np.zeros(2),U=np.ones(2)*1000,eps=np.zeros(2),low=np.ones(2))
    rows=pd.DataFrame(dict(input_hash=['a','b']))
    frame=error_rows('FREEFLOW_LINEAR',dict(c0=np.array([2.,2.])),a,rows,'region','cohort')
    summary=regional_aggregate(frame,['region_id','cohort'])
    assert summary.cost_abs_error_sum.iloc[0]==198
    assert summary.true_cost_sum.iloc[0]==202
    assert summary.cost_wmape_percent.iloc[0]==100*198/202
    assert not np.isclose(summary.cost_wmape_percent.iloc[0],frame.cost_wmape_query_percent.mean())


def test_forward_metric_retains_reference_zero_and_closed_key_zero():
    summary,per=compare_vectors([100,50,0],[110,40,7],[True,False,True],np.ones(3),np.zeros(3))
    assert summary[0]['wmape_percent']==18
    assert summary[1]['wmape_percent']==17
    assert per.signed_bounded_error.iloc[2]==1 and np.isnan(per.signed_relative_error.iloc[2])
    assert summary[0]['reference_zero_positive_prediction_count']==1


def test_reference_pipeline_self_zero_and_no_extra_charges():
    ref=np.array([5.,0.,9.]);base=np.array([2.,3.,4.]);k4=np.array([.1,.2,.3])
    summary,per=compare_vectors(ref,ref,[True,True,False],base,k4)
    assert all(x['wmape_percent']==x['mae']==x['rmse']==x['backbone_cost_error_percent']==0 for x in summary)
    assert summary[0]['reference_backbone_cost']==float(ref@(base+k4*ref**4))
    assert np.array_equal(per.signed_bounded_error,np.zeros(3))
