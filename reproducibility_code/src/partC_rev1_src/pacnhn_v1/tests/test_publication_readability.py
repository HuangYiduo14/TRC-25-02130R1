"""Display omissions remain explicit and retain every scientific observation."""
import textwrap
from pathlib import Path
import pandas as pd
from partC_rev1_src.pacnhn_v1 import publication as p


def test_timing_table_preserves_nested_observations_and_censored_query(monkeypatch):
    data={key:pd.DataFrame(columns=columns) for key,columns in p.SCHEMAS.items()}
    data['timing']=pd.DataFrame([
        dict(task='case',category='ONLINE_ASSIGNMENT_COMPLETE_QUERY',seconds=2700.,status='TIME_LIMIT',timing_boundary='cold query'),
        dict(task='case',category='CPU_TIME',seconds=None,status='UNAVAILABLE',timing_boundary='not measured'),
        dict(task='old',category='HISTORICAL_REFRESH_ONLY',seconds=4.,status='CONTEXT_ONLY',timing_boundary='refresh only')])
    before=data['timing'].copy(deep=True);assets={}
    def capture(root,asset_id,frames,records,**kwargs):
        assets[asset_id]=(frames,kwargs['table']);return Path('not-written')
    monkeypatch.setattr(p,'asset_folder',capture)
    p.build_tables(Path('unused'),data,[])
    frames,table=assets['table6']
    assert list(table.status)==['TIME_LIMIT','CONTEXT_ONLY']
    assert list(frames['observations_display_policy.csv'].displayed)==[True,False,True]
    pd.testing.assert_frame_equal(frames['observations_timing.csv'],before)
    pd.testing.assert_frame_equal(data['timing'],before)


def test_regional_caption_fits_existing_readable_area():
    assert len(textwrap.fill(p.CAPTIONS['table3'],160,break_long_words=False).splitlines())<=5
