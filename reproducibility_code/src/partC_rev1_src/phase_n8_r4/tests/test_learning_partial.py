"""Failure persistence tests use no optimizer steps or regional oracle calls."""
from types import SimpleNamespace
import time
import pandas as pd
from partC_rev1_src.phase_n8_r4 import learning,learning_evaluate
from partC_rev1_src.phase_n8.common import read_json


def test_failed_runtime_pilots_preserve_docket_and_unknown_forecast(tmp_path,monkeypatch):
    out=tmp_path/'learning';out.mkdir()
    rows=[dict(region_id=f'r{i}',stratum=f's{i//4}',rank_within_stratum=i%4+1,train_file_sha256='mock') for i in range(16)]
    pd.DataFrame(rows).to_csv(out/'selected_regions_16.csv',index=False)
    monkeypatch.setattr(learning,'read_npz',lambda path:{})
    def fail(*args):raise ValueError('deliberate timing-pilot failure')
    monkeypatch.setattr(learning,'make_model',fail)
    ctx=SimpleNamespace(root=tmp_path,guard=lambda **kw:None,bind=lambda *args:None)
    plan=learning.pilot(ctx,dict(prepared_regions=rows,preparation_failures=[]),time.time()+10000)
    saved=pd.read_csv(out/'runtime_pilot.csv')
    assert len(saved)==16 and saved.status.eq('FAILED').all()
    assert saved.error.str.contains('deliberate').all()
    assert len(plan['selected_region_ids'])==8
    assert all(r['forecast_seconds'] is None and not r['fits'] for r in plan['runtime_pilot_choices'])
    assert read_json(out/'STUDY_FREEZE.json')['selected_region_ids']==plan['selected_region_ids']


def test_no_prepared_regions_keeps_schemaful_missing_evidence(tmp_path):
    (tmp_path/'learning').mkdir()
    ctx=SimpleNamespace(root=tmp_path)
    result=learning_evaluate.evaluate(ctx,{},[],[],time.time()+100)
    assert result['status']=='NO_PREPARED_REGIONS' and result['fresh_AB']==0
    coverage=pd.read_csv(tmp_path/'learning/new_label_coverage.csv')
    summary=pd.read_csv(tmp_path/'learning/learning_curve_summary.csv')
    planned=pd.read_csv(tmp_path/'learning/fresh_evaluation_planned_coverage.csv')
    assert coverage.empty and {'region_id','quality','status'}<=set(coverage)
    assert summary.empty and {'stratum','cohort','status'}<=set(summary)
    assert planned.empty and {'region_id','cohort','slot'}<=set(planned)
