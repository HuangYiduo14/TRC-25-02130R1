from partC_rev1_src.pacnhn_recovery24.campaign import family_retry_allowed
import pytest


@pytest.mark.parametrize('policy,reason',[
    ({'unproductive_tranches':2},'TWO_UNPRODUCTIVE_TRANCHES'),
    ({'elapsed_seconds':1200},'CUMULATIVE_WORKER_TIME_EXHAUSTED'),
    ({'failures':2},'REPEATED_FAILURE_AFTER_RETRY'),
    ({'invariant_failure':True},'INVARIANT_FAILURE')])
def test_stop_gates_are_not_reset_by_a_new_pass(policy,reason):
    state={'recovery_family_policy':{'ICNN':policy}}
    assert family_retry_allowed(state,'ICNN',200000)==(False,reason)


def test_xgb_known_no_tree_is_not_a_fitting_task():
    state={'dataset_identity':'same','xgb_latest':dict(boosting_rounds_total=128,
        uncompressed_model_bytes=100000,max_depth=7,trained_dataset_identity='same')}
    assert family_retry_allowed(state,'XGBOOST',100000)[0] is False
    # The approved changed-cohort restart is scientifically a new fit, so allow.
    state['dataset_identity']='new'
    assert family_retry_allowed(state,'XGBOOST',100000)[0] is True


def test_next_approved_xgb_rounds_require_actual_capacity():
    state={'dataset_identity':'same','xgb_latest':dict(boosting_rounds_total=32,
        uncompressed_model_bytes=100000,max_depth=7,trained_dataset_identity='same')}
    assert family_retry_allowed(state,'XGBOOST',100000)[0] is False
    assert family_retry_allowed(state,'XGBOOST',200000)[0] is True
