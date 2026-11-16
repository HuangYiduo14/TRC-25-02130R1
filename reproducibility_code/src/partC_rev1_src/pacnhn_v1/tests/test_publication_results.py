import pandas as pd
import pytest
from partC_rev1_src.pacnhn_v1.publication_results import scientific_results


def inputs():
    rows=[]
    for scenario in ['OPEN','REPRESENTED_I95']:
        for i, method in enumerate(['P/REFERENCE','P/CFW','P/FW','P/MSA',
                                   's03/FFTT','s03/POWER4','s03/XGBOOST','s03/ICNN']):
            physical=method.startswith('P/')
            rows.append(dict(case_id=method+'_'+scenario,method_id=method,network='P' if physical else 'AC',
                suite_id=None if physical else 's03',scenario=scenario,
                assignment_role='REFERENCE_SELF' if i==0 else 'INDEPENDENT_QUERY',status='CONVERGED',
                wmape_backbone_percent=float(i)+(1 if scenario!='OPEN' and i else 0),
                wmape_key_percent=float(8-i) if i else 0.,certificate_fresh=True,
                served_OD_membership_equal=True,elapsed_seconds=10.*i))
    return pd.DataFrame(rows),pd.DataFrame(dict(method_id=['P/REFERENCE']*3,
        link_id=['a','b','c'],delta=[8.,-3.,0.]))


def test_excludes_reference_identity_and_reports_independent_controls():
    methods,i95=inputs();result=scientific_results(methods,i95)
    assert 'from 1% to 3%' in result['assignments'][1]
    assert 'from 0%' not in result['assignments'][1]
    assert 'FFTT had the smallest full-backbone error' in result['assignments'][2]
    assert 'ICNN the smallest fixed-key error' in result['assignments'][2]
    assert '14 converged primary' in result['timing'][0]
    assert '8 annual thousand tons' in result['i95'][-1]
    assert '3 annual thousand tons' in result['i95'][-1]
    assert 'FFTT +1 percentage points' in result['i95'][-2]


def test_missing_and_uncertified_rows_do_not_become_zero_error():
    methods,i95=inputs()
    methods=methods.loc[~methods.method_id.eq('s03/FFTT')].copy()
    mask=methods.method_id.eq('s03/ICNN');methods.loc[mask,'certificate_fresh']=False
    methods.loc[mask,'status']='TIME_LIMIT'
    result=scientific_results(methods,i95)
    assert 'FFTT: unavailable' in result['assignments'][0]
    assert 'ICNN: TIME_LIMIT (no admitted flow-error comparison)' in result['assignments'][0]
    assert 'Among the 2 reported' in result['assignments'][2]
    assert 'ICNN +1' not in ' '.join(result['i95'])


def test_other_resolutions_do_not_enter_primary_results():
    methods,i95=inputs();extra=methods.loc[methods.method_id.eq('s03/FFTT')].copy()
    extra['method_id']='s04/FFTT';extra['case_id']='s04_'+extra.scenario;extra['suite_id']='s04'
    extra['wmape_backbone_percent']=0.;extra['elapsed_seconds']=1.
    combined=pd.concat([methods,extra],ignore_index=True)
    assert scientific_results(methods,i95)==scientific_results(combined,i95)


def test_duplicate_methods_rejected():
    methods,i95=inputs();extra=methods.iloc[[0]].copy();extra['case_id']='different'
    with pytest.raises(ValueError,match='Duplicate'):
        scientific_results(pd.concat([methods,extra],ignore_index=True),i95)


def test_invalid_change_population_rejected():
    methods,i95=inputs();i95.loc[1,'link_id']='a'
    with pytest.raises(ValueError,match='Duplicate'):
        scientific_results(methods,i95)
    i95.loc[1,'link_id']='b';i95.loc[1,'delta']=float('inf')
    with pytest.raises(ValueError,match='Nonfinite'):
        scientific_results(methods,i95)
