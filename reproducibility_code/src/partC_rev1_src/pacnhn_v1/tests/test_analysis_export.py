import pytest
from partC_rev1_src.pacnhn_v1.analysis_export import method_row,validate_outcome_matrix
from partC_rev1_src.pacnhn_v1.assignments import matrix


def test_primary_queries_are_independent_and_include_direct_cost_xgb():
    rows=matrix()
    assert len(rows)==14 and len({r['case_id'] for r in rows})==14
    assert len([r for r in rows if r['network']=='P'])==6
    assert {r['family'] for r in rows if r['network']=='AC'}=={'FFTT','POWER4','XGBOOST','ICNN'}
    assert all(r['solver']=='MSA' for r in rows if r['family']=='XGBOOST')
    assert not any(r['suite_id']=='s01' for r in rows)


@pytest.mark.parametrize('potential,kind',[(True,'FW_GAP'),(False,'VI_RESIDUAL')])
def test_publication_preserves_nonzero_control_error_and_residual_interpretation(potential,kind):
    case=dict(case_id='cold',method_id='P/CFW',network='P',scenario='open',assignment_role='INDEPENDENT_CONTROL')
    result=dict(certificate={'RG_SPTT':.002},status='TIME_LIMIT',method='CFW' if potential else 'MSA',
        elapsed_seconds=2699,final_certificate_fresh=True,objective_has_potential=potential,
        nominal_tolerance=.001,deadline_epoch=4000,complete_query_start_epoch=1300,reported_state_iteration=100)
    ref=dict(status='CONVERGED',certificate={'RG_SPTT':.00009})
    rows=[dict(population=p,wmape_percent=v,served_OD_membership_equal=True,
               reference_zero_link_count=6,reference_zero_positive_prediction_count=2)
          for p,v in [('BACKBONE',1.5),('KEY_LINKS',2.1)]]
    row=method_row(case,result,ref,rows)
    assert row['wmape_backbone_percent']==1.5
    assert row['wmape_key_percent']==2.1
    assert row['residual_kind']==kind
    assert row['query_cap_seconds']==2700
    assert row['reference_status']=='CONVERGED' and row['status']=='TIME_LIMIT'
    assert row['reference_case_id']=='P_REFERENCE_open'


def test_unavailable_rows_must_remain_in_complete_planned_case_ledger():
    planned=matrix()
    outcomes=[dict(case=dict(r),status='UNAVAILABLE_INCOMPLETE_PURE_FAMILY') for r in planned]
    validate_outcome_matrix(planned,outcomes)
    with pytest.raises(ValueError):validate_outcome_matrix(planned,outcomes[:-1])
    with pytest.raises(ValueError):validate_outcome_matrix(planned,outcomes+[outcomes[0]])
    outcomes[-1]['case']['solver']='CFW'
    with pytest.raises(ValueError):validate_outcome_matrix(planned,outcomes)
