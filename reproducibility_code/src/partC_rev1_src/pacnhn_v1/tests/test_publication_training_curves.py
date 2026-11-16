import copy
import json

import pandas as pd
import pytest

from partC_rev1_src.pacnhn_v1.publication_training_curves import select_first_candidate_evaluations


def row(candidate='ICNN_1', elapsed=1., **kw):
    value=dict(suite_id='s03',region_index=0,family='ICNN',candidate_id=candidate,block=1,
               elapsed_seconds=elapsed,train_wmape=2.,valid_wmape=3.,train_dataset_identity='TRAIN1',
               valid_dataset_identity='VALID1',evaluation_kind='NEW_CHECKPOINT',audit_path=candidate+'_'+str(elapsed))
    value.update(kw)
    return value


def test_different_valid_same_candidate_keeps_all_raw_selects_first():
    frame=pd.DataFrame([row(elapsed=4,valid_dataset_identity='VALID2'),row(elapsed=2),
                        row(elapsed=5,evaluation_kind='REEVALUATION_ON_GROWING_TRAIN')])
    before=copy.deepcopy(frame)
    plotted,ledger=select_first_candidate_evaluations(frame)
    pd.testing.assert_frame_equal(frame,before)
    assert len(plotted)==1 and plotted.iloc[0].valid_dataset_identity=='VALID1'
    assert ledger['counts']==dict(raw_observations=3,new_checkpoint_observations=2,plotted_candidates=1,
                                 repeated_new_observations=1,retained_non_new_observations=1)
    assert len(ledger['observations'])==3 and ledger['observations'][0]['valid_dataset_identity']=='VALID2'
    assert plotted.iloc[0].block==1 and plotted.iloc[0].checkpoint_ordinal==1
    json.dumps(ledger,allow_nan=False)


def test_same_block_distinct_capacity_candidates_receive_ordinals():
    frame=pd.DataFrame([row('ICNN_2',3),row('ICNN_1',2),row('ICNN_3',3)])
    plotted,_=select_first_candidate_evaluations(frame)
    assert plotted.candidate_id.tolist()==['ICNN_1','ICNN_2','ICNN_3']
    assert plotted.block.tolist()==[1,1,1] and plotted.checkpoint_ordinal.tolist()==[1,2,3]
    shuffled,_=select_first_candidate_evaluations(frame.iloc[::-1])
    assert shuffled.candidate_id.tolist()==plotted.candidate_id.tolist()


def test_identical_duplicate_tie_is_deterministic():
    r=row();plotted,ledger=select_first_candidate_evaluations(pd.DataFrame([r,r]))
    assert plotted.source_observation_index.tolist()==[0]
    assert ledger['observations'][1]['selected_source_observation_index']==0


@pytest.mark.parametrize('change',[dict(valid_wmape=4.),dict(valid_dataset_identity='VALID2'),dict(audit_path='other'),dict(block=2)])
def test_contradictory_candidate_time_tie_rejected(change):
    with pytest.raises(ValueError,match='Contradictory'):
        select_first_candidate_evaluations(pd.DataFrame([row(),row(**change)]))


def test_family_region_and_suite_each_separate_candidate_identity():
    rows=[row(),row(family='XGBOOST'),row(region_index=2),row(suite_id='s04')]
    plotted,ledger=select_first_candidate_evaluations(pd.DataFrame(rows))
    assert len(plotted)==4 and plotted.checkpoint_ordinal.tolist()==[1]*4
    assert ledger['counts']['repeated_new_observations']==0


@pytest.mark.parametrize('elapsed',[float('nan'),float('inf'),-1.,'nonsense'])
def test_bad_new_chronology_rejected(elapsed):
    with pytest.raises((ValueError,TypeError)):
        select_first_candidate_evaluations(pd.DataFrame([row(elapsed=elapsed)]))


def test_empty_and_no_new_frames_have_explicit_zero_plot_counts():
    for frame in [pd.DataFrame([row()]).iloc[:0],pd.DataFrame([row(evaluation_kind='REEVALUATION_ON_GROWING_TRAIN')])]:
        plotted,ledger=select_first_candidate_evaluations(frame)
        assert plotted.empty and ledger['counts']['plotted_candidates']==0
        assert len(ledger['observations'])==len(frame)


def test_undefined_metric_remains_missing_not_zero_and_json_ledger_safe():
    plotted,ledger=select_first_candidate_evaluations(pd.DataFrame([row(valid_wmape=float('nan'))]))
    assert pd.isna(plotted.iloc[0].valid_wmape)
    assert ledger['observations'][0]['valid_wmape'] is None
    json.dumps(ledger,allow_nan=False)


def test_missing_identity_or_reserved_columns_rejected():
    with pytest.raises(ValueError,match='Missing candidate identity'):
        select_first_candidate_evaluations(pd.DataFrame([row(candidate='')]))
    with pytest.raises(ValueError,match='reserved'):
        select_first_candidate_evaluations(pd.DataFrame([row(checkpoint_ordinal=4)]))


def test_sanitized_publication_columns_without_audit_path_and_extra_metadata():
    frame=pd.DataFrame([row(),row(elapsed=2,valid_dataset_identity='VALID2')]).drop(columns=['audit_path'])
    frame['public_note']=['first','later']
    plotted,ledger=select_first_candidate_evaluations(frame)
    assert 'audit_path' not in plotted and all('audit_path' not in r for r in ledger['observations'])
    assert len(plotted)==1 and ledger['observations'][1]['public_note']=='later'
    assert ledger['counts']['raw_observations']==2
