"""Bounded geographic audits keep incomplete populations out of measured results."""
from types import SimpleNamespace
import pandas as pd
import pytest
from partC_rev1_src.phase_n8.common import read_json,write_json
from partC_rev1_src.phase_n8_r4 import stability,regional_geography,boundary_diagnostics
from partC_rev1_src.phase_n8_r4.diagnostic_handoffs import progress


def context(root):
    return SimpleNamespace(root=root,deadline=1000.,c={'compute_cutoff_epoch':2000.,
        'active_suites':['s02','s03','s04','s05','s06']})


def test_stability_stop_cannot_report_a_smaller_intersection(monkeypatch,tmp_path):
    def stop(ctx,args):
        progress(tmp_path/'stability','STABILITY_STATES.json',[dict(unit='inputs_s02',status='MEASURED')],
            active=dict(unit='inputs_s03'))
        raise TimeoutError('bounded stop before all five inputs')
    monkeypatch.setattr(stability,'_run',stop)
    result=stability.run(context(tmp_path),SimpleNamespace(job_id='test'))
    assert result['status']=='PARTIAL_DIAGNOSTIC' and result['measured_units']==1
    states={r['unit']:r['status'] for r in read_json(tmp_path/'stability/STABILITY_STATES.json')}
    assert len(states)==42 and states['inputs_s03']=='INTERRUPTED_BUDGET'
    assert states['five_suite_common_population']=='NOT_RUN_BUDGET'
    assert pd.read_csv(tmp_path/'stability/partition_stability.csv').empty
    labels=pd.read_csv(tmp_path/'stability/common_final_regional_arc_labels.csv')
    assert labels.empty and list(labels)==['physical_arc_id','s02','s03','s04','s05','s06']
    assert 'N8_runtime_nodes' in pd.read_csv(tmp_path/'stability/resolution_counts.csv')


def test_geography_stop_preserves_completed_region_and_missing_role(monkeypatch,tmp_path):
    selected=pd.DataFrame([dict(region_id='a',figure_role='large'),dict(region_id='b',figure_role='good_VALID')])
    monkeypatch.setattr(regional_geography,'selected_regions',lambda ctx:selected)
    def stop(ctx,args,docket):
        dst=tmp_path/'regional_geography'
        pd.DataFrame([dict(region_id='a',figure_role='large',mean_x=1.,mean_y=2.)],
            columns=regional_geography.SUMMARY_COLUMNS).to_csv(dst/'region_geography_summary.csv',index=False)
        progress(dst,'GEOGRAPHY_STATES.json',[dict(region_id='a',status='MEASURED')],active=dict(region_id='b'))
        raise TimeoutError('second geometry incomplete')
    monkeypatch.setattr(regional_geography,'_run',stop)
    result=regional_geography.run(context(tmp_path),SimpleNamespace(job_id='test'))
    assert result['status']=='PARTIAL_DIAGNOSTIC' and result['measured_units']==1
    states=read_json(tmp_path/'regional_geography/GEOGRAPHY_STATES.json')
    assert [r['status'] for r in states]==['MEASURED','INTERRUPTED_BUDGET']
    summary=pd.read_csv(tmp_path/'regional_geography/region_geography_summary.csv')
    assert summary.region_id.tolist()==['a']
    assert 'terminal_type' in pd.read_csv(tmp_path/'regional_geography/true_portal_and_centroid_points.csv')


@pytest.mark.parametrize('error,status',[(TimeoutError('time cap'),'PARTIAL_DIAGNOSTICS'),
    (ValueError('source identity mismatch'),'FAIL')])
def test_boundary_preserves_prefix_but_rejects_identity_failures(monkeypatch,tmp_path,error,status):
    def stop(ctx,args):
        dst=tmp_path/'boundary_diagnostics';dst.mkdir()
        write_json(dst/'GLOBAL_DIAGNOSTIC_PROGRESS.json',dict(completed_selected_ODs=1,verified_selected_global_lifts=1))
        (dst/'verified_selected_global_lift_arcs.csv').write_text('stable_od_key,physical_arc_id\nx,7\n')
        raise error
    monkeypatch.setattr(boundary_diagnostics,'_run',stop)
    ctx=context(tmp_path);result=boundary_diagnostics.run(ctx,SimpleNamespace(job_id='test'))
    assert result['status']==status and result['completed_selected_ODs']==1
    assert result['verified_selected_global_lifts']==1 and ctx.deadline==1000.
    assert pd.read_csv(tmp_path/'boundary_diagnostics/verified_selected_global_lift_arcs.csv').physical_arc_id.tolist()==[7]


@pytest.mark.parametrize('kind',['ownership','heatmap'])
def test_empty_geographic_panel_becomes_explicit_disposition(tmp_path,kind):
    from partC_rev1_src.phase_n8_r4.release_data import Release
    release=Release(SimpleNamespace(root=tmp_path,c={'run_id':'test'}))
    data=release.csv('fig_S2','empty.csv',pd.DataFrame(columns=['x','y','suite']),'Empty verified schema')
    release.panel('fig_S2','a',kind,data,'Incomplete geography','x','y','suite')
    spec=release.specs['fig_S2'][0]
    assert spec['kind']=='status' and spec['data']=='status_a.csv'
    assert pd.read_csv(tmp_path/'numerical_release/fig_S2/empty.csv').empty
    disposition=pd.read_csv(tmp_path/'numerical_release/fig_S2/status_a.csv')
    assert disposition.status.tolist()==['NOT_MEASURED']
