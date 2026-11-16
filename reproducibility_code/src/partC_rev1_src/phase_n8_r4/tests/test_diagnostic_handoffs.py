"""Interrupted diagnostics preserve completed records and the full missing docket."""
from types import SimpleNamespace
import pandas as pd
import pytest
from partC_rev1_src.phase_n8.common import read_json,write_json
from partC_rev1_src.phase_n8_r4.diagnostic_handoffs import bounded,progress


def test_interrupted_track_preserves_schema_and_unvisited_coverage(tmp_path):
    ctx=SimpleNamespace(root=tmp_path,deadline=1000.,c={'compute_cutoff_epoch':2000.})
    plans=[dict(scenario=s,track=t) for s in ('a','b') for t in ('fixed','solved')]
    def work(ctx,args):
        assert ctx.deadline==880.
        write_json(tmp_path/'probe/STATES.json',[dict(scenario='a',track='fixed',status='PATHS_MEASURED',rows=5)])
        raise TimeoutError('mock stop after first track')
    result=bounded(ctx,SimpleNamespace(job_id='mock'),'probe',plans,['scenario','track'],'STATES.json',
        {'METRICS.csv':['scenario','track','WREL']},work)
    assert result['status']=='PARTIAL_DIAGNOSTIC'
    assert result['measured_units']==1 and result['missing_units']==3
    assert ctx.deadline==1000.
    table=pd.read_csv(tmp_path/'probe/METRICS.csv')
    assert table.empty and list(table)==['scenario','track','WREL']
    states=read_json(tmp_path/'probe/STATES.json')
    assert len(states)==4 and states[0]['rows']==5
    assert all(row['status']=='NOT_RUN_BUDGET' for row in states[1:])
    before=(tmp_path/'probe/RESULT.json').read_bytes()
    with pytest.raises(FileExistsError):
        bounded(ctx,SimpleNamespace(job_id='retry'),'probe',plans,['scenario','track'],'STATES.json',
            {'METRICS.csv':['scenario','track','WREL']},work)
    assert (tmp_path/'probe/RESULT.json').read_bytes()==before


def test_case_rejection_applies_to_every_track_without_admitting_data(tmp_path):
    ctx=SimpleNamespace(root=tmp_path,deadline=1000.,c={'compute_cutoff_epoch':2000.})
    plans=[dict(scenario='a',track=t) for t in ('fixed','solved')]
    def work(ctx,args):
        write_json(tmp_path/'probe/STATES.json',[dict(scenario='a',status='NO_ADMITTED_FRESH_STATE',
            stop_reason='STOP_NEGATIVE_COST',final_certificate_fresh=False)])
        return dict(status='NO_ADMITTED_STATES')
    result=bounded(ctx,SimpleNamespace(job_id='mock'),'probe',plans,['scenario','track'],'STATES.json',
        {'METRICS.csv':['scenario','track','WREL']},work)
    states=read_json(tmp_path/'probe/STATES.json')
    assert result['measured_units']==0 and len(states)==2
    assert all(row['stop_reason']=='STOP_NEGATIVE_COST' and not row['final_certificate_fresh'] for row in states)


def test_identity_failure_is_hard_failure_and_distinguishes_attempted_unit(tmp_path):
    ctx=SimpleNamespace(root=tmp_path,deadline=1000.,c={'compute_cutoff_epoch':2000.})
    plans=[dict(scenario=s) for s in ('a','b','c')]
    def work(ctx,args):
        progress(tmp_path/'probe','STATES.json',[dict(scenario='a',status='MEASURED')],
            active=dict(scenario='b',attempt_stage='CASE_ADMISSION'))
        live=read_json(tmp_path/'probe/STATES.json')
        assert [r['status'] for r in live]==['MEASURED','RUNNING','NOT_STARTED']
        raise ValueError('source identity mismatch')
    result=bounded(ctx,SimpleNamespace(job_id='mock'),'probe',plans,['scenario'],'STATES.json',
        {'METRICS.csv':['scenario','WREL']},work)
    assert result['status']=='FAIL'
    states=read_json(tmp_path/'probe/STATES.json')
    assert [r['status'] for r in states]==['MEASURED','FAILED_DIAGNOSTIC','NOT_RUN_INTERRUPTED']
    assert states[1]['attempt_stage']=='CASE_ADMISSION'
    assert states[1]['error']=='source identity mismatch'
