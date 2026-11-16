"""Synthetic-only publication contracts, preservation and visible plot semantics."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from partC_rev1_src.pacnhn_recovery24 import publication as p
from partC_rev1_src.pacnhn_recovery24 import publication_geometry as g
from partC_rev1_src.pacnhn_recovery24.publication_templates import plot


def learning():
    rows=[]
    for i,(error,truth) in enumerate([(1.,10.),(3.,0.),(np.nan,np.nan)]):
        for role in ['TRAIN','VALID','TEST']:
            rows.append(dict(suite_id='s03',region_index=i,family='ICNN',version='recovered',role=role,
                WMAPE_percent=(100*error/truth if truth>0 else np.nan),absolute_error_sum=error,true_cost_sum=truth,
                coordinates=2 if np.isfinite(error) else 0,admitted_inputs=1 if np.isfinite(error) else 0,
                requested_inputs=48,dataset_identity=role+str(i),model_sha256='a'*64,target_met=i==0,
                fallback_component=None,fallback_reason=None,status='AVAILABLE' if np.isfinite(error) else 'UNAVAILABLE'))
    return pd.DataFrame(rows)


def data():
    methods=[]
    for scenario in ['OPEN','REPRESENTED_I95']:
        for family in ['REFERENCE','CFW','FW','MSA']:
            methods.append(dict(case_id='P_'+family+'_'+scenario,network='P',suite_id=None,family=family,
                scenario=scenario,effective_method='P/'+family,solver=family,status='CONVERGED',RG_SPTT=.0005,
                elapsed_seconds=1.,wmape_backbone_percent=0.,wmape_key_percent=0.,backbone_cost_error_percent=0.,
                assignment_role='REFERENCE_SELF' if family=='REFERENCE' else 'INDEPENDENT_CONTROL',execution_status='REUSED'))
        for sid in ['s02','s03','s04','s05','s06']:
            for family in ['ICNN','XGBOOST','POWER4','FFTT']:
                methods.append(dict(case_id=sid+'_'+family+'_'+scenario,network='AC',suite_id=sid,family=family,
                    scenario=scenario,effective_method=sid+'/'+family,solver='MSA' if family=='XGBOOST' else 'FW',
                    status='CONVERGED',RG_SPTT=.0007,elapsed_seconds=2.,wmape_backbone_percent=3.,
                    wmape_key_percent=4.,backbone_cost_error_percent=5.,assignment_role='SURROGATE',execution_status='NEW'))
    m=pd.DataFrame(methods)
    geom=pd.DataFrame([dict(tail_longitude=-90.,head_longitude=-89.,tail_latitude=35.,head_latitude=36.)])
    links=[]
    for r in m.loc[m.scenario.eq('OPEN')&(m.network.eq('P')|m.suite_id.eq('s03'))].itertuples():
        links.append(dict(geom.iloc[0],case_id=r.case_id,link_id='B0',native_index=0,reference_flow=1.,predicted_flow=2.,signed_bounded_error=.5,is_key=True))
    summary=[];obs=[]
    for stratum in ['large','small','good','bad']:
        for family in ['ICNN','XGBOOST']:
            for fraction in [.1,.25,.5,.7,1.]:
                summary.append(dict(stratum=stratum,model_family=family,training_fraction=fraction,
                    median_WMAPE=.02,q25_WMAPE=.01,q75_WMAPE=.03,region_seed_rows=3))
                for i in range(4):
                    obs.append(dict(task_id=str(len(obs)),region_key='s03:'+str(i),stratum=stratum,family=family,
                        fraction=fraction,fit_status='AVAILABLE' if i else 'FAILED',TEST_status='MEASURED' if i else 'UNAVAILABLE_FIT',
                        TEST_WMAPE_percent=2. if i else np.nan,requested_TEST_inputs=48,admitted_TEST_inputs=40 if i else 0))
    counts=pd.DataFrame([dict(suite_id=s,resolution=i,regions=10+i,portal_nodes=5+i,active_regions=5,
                             active_pairs=15,mean_physical_arcs=7.) for i,s in enumerate(['s02','s03','s04','s05','s06'],1)])
    region=pd.concat([geom.assign(suite_id='s03',region_index=i,stratum=s,geometry_role='PHYSICAL_LINK',native_id='R'+str(i))
                      for i,s in enumerate(['large','good','bad'])],ignore_index=True)
    maps=pd.concat([geom.assign(link_id='B0',edge_kind='BACKBONE',flow=f,scenario=s,is_closed=True)
                    for s,f in [('open',1.),('i95',0.)]],ignore_index=True)
    return dict(methods=m,backbone_links=pd.DataFrame(links),learning=p.normalize_learning(learning()),learning_observations=learning(),
        networks=counts[['suite_id','resolution']],settings=pd.DataFrame([dict(setting='Native flow',value='annual thousand tons')]),
        timing=pd.DataFrame([dict(case_id='P_CFW_OPEN',seconds=1,status='CONVERGED',timing_boundary='Complete query',time_kind='HISTORICAL')]),
        resolution_counts=counts,partition_stability=pd.DataFrame([dict(suite_a='s02',suite_b='s03',ARI=.5,NMI_arithmetic=.6,common_population_n=3,population='CURRENT')]),
        boundary_jaccard=pd.DataFrame([dict(suite_a='s02',suite_b='s03',boundary_Jaccard=.4,population='CURRENT')]),
        convergence=pd.DataFrame([dict(case_id='P_CFW_OPEN',iteration=1,elapsed_seconds=1.,RG_SPTT=.0005)]),
        ablation_summary=pd.DataFrame(summary),ablation_observations=pd.DataFrame(obs),region_geometry=region,
        region_locations=region.assign(geometry_role='REGION_LOCATION',label=['large','good','bad']),
        primary_map_flows=maps,closure_geometry=geom.assign(link_id='B0'),
        served_membership=pd.DataFrame([dict(population='LOST_CLOSED_VS_OPEN',aggregate_OD_rows=2,original_OD_row_multiplicity=3,demand=4.)]))


def test_pooled_errors_include_zero_truth_and_do_not_average_medians():
    result=p.learning_summary(p.normalize_learning(learning())).iloc[0]
    assert result.pooled_TEST_percent==40
    assert result.median_TEST_percent==10
    assert result.pooled_regions==2 and result.measured_regions==1
    assert result.requested_TEST_inputs==144 and result.admitted_TEST_inputs==2
    assert result.fallback_regions==0


@pytest.mark.parametrize('change',['duplicate','missing','identity'])
def test_role_or_model_identity_conflicts_reject(change):
    frame=learning()
    if change=='duplicate':frame=pd.concat([frame,frame.iloc[[0]]])
    if change=='missing':frame=frame.iloc[1:]
    if change=='identity':frame.loc[0,'model_sha256']='b'*64
    with pytest.raises(ValueError):p.normalize_learning(frame)


def test_half_missing_and_negative_aggregate_reject():
    frame=p.normalize_learning(learning());frame.loc[0,'TEST_true_cost_sum']=np.nan
    with pytest.raises(ValueError):p.learning_summary(frame)
    frame.loc[0,'TEST_true_cost_sum']=-1
    with pytest.raises(ValueError):p.learning_summary(frame)


def test_fallback_count_is_real_analytic_replacement_not_missing_component():
    frame=p.normalize_learning(learning());frame['version']='portfolio'
    frame['selected_family']=['ICNN','POWER4',None]
    summary=p.learning_summary(frame).iloc[0]
    assert summary.fallback_regions==1 and summary.unavailable_component_regions==1


def test_nonfinite_supplied_sums_fail_closed():
    frame=p.normalize_learning(learning());frame.loc[0,'TEST_absolute_error_sum']=np.inf
    with pytest.raises(ValueError):p.learning_summary(frame)


def test_optional_training_seconds_require_explicit_worker_scope():
    frame=pd.DataFrame({'recorded_training_seconds':[12.,np.nan]})
    with pytest.raises(ValueError):p.validate_training_timing(frame)
    frame['recorded_training_scope']='RECOVERY_FAMILY_FIT_AND_AUDIT_WORKER_SECONDS'
    p.validate_training_timing(frame)
    frame.loc[0,'recorded_training_scope']='CPU_SECONDS'
    with pytest.raises(ValueError):p.validate_training_timing(frame)


@pytest.mark.parametrize('error,truth',[(1e308,1e308),(1e308,1e-308)])
def test_finite_individual_sums_cannot_overflow_pooled_result(error,truth):
    frame=p.normalize_learning(learning())
    frame['TEST_absolute_error_sum']=0.;frame.loc[0,'TEST_absolute_error_sum']=error;frame['TEST_true_cost_sum']=truth
    with pytest.raises(ValueError):p.learning_summary(frame)


def test_optional_gamma_metadata_is_retained_and_structural_only():
    frame=learning();frame['neural_amplitude']=frame.region_index.map({0:0.,1:2.,2:np.nan})
    frame['neural_amplitude_is_zero']=frame.region_index.map({0:True,1:False})
    frame['neural_activity_scope']='RECORDED_SCALAR_ONLY'
    normalized=p.normalize_learning(frame);r=p.learning_summary(normalized).iloc[0]
    assert (r.ICNN_zero_Gamma,r.ICNN_positive_Gamma,r.ICNN_unknown_Gamma)==(1,1,1)
    assert normalized.neural_activity_scope.eq('RECORDED_SCALAR_ONLY').all()
    d=data();d['learning']=normalized;text=' '.join(p.scientific_paragraphs(d))
    assert 'positive amplitude is structural metadata, not measured predictive benefit' in text
    normalized.loc[0,'neural_amplitude_is_zero']=False
    with pytest.raises(ValueError):p.learning_summary(normalized)


def test_nine_standalone_folders_keep_all_dispositions_and_physical_rows(tmp_path):
    d=data();original=d['methods'].copy(deep=True)
    p.build_tables(tmp_path,d,{})
    p.build_figures(tmp_path,d,{})
    assert {x.name for x in (tmp_path/'numerical_data').iterdir()}==set(p.ASSETS)
    table=pd.read_csv(tmp_path/'numerical_data/table2/table.csv')
    assert len(table)==8 and table.effective_method.str.startswith('P/').sum()==4
    underlying=pd.read_csv(tmp_path/'numerical_data/table2/observations_all_48_methods.csv')
    assert len(underlying)==48 and len(underlying.loc[underlying.network.eq('P')])==8
    pd.testing.assert_frame_equal(original,d['methods'])
    for name in ['figure3','figure4','figure5','figure6','figure7']:
        folder=tmp_path/'numerical_data'/name
        assert len(pd.read_csv(folder/'panels.csv'))==4
        script=(folder/'plot.py').read_text()
        assert '# EMBED_LOCAL_LOGICAL_CSV_SUPPORT_HERE' not in script
        p.portable.static_renderer_check(folder/'plot.py')
    observed=pd.read_csv(tmp_path/'numerical_data/figure5/observations_all_160_fits.csv')
    assert len(observed)==160 and observed.fit_status.eq('FAILED').sum()==40


def test_v5_table_no_new_package_and_breaks_literal_ids():
    out=p.v5_table(pd.DataFrame({'identity':['a'*64]}),[('identity','Frozen identity',1)],'Title','tab:test')
    assert r'\allowbreak{}' in out
    assert 'landscape' not in out and 'a'*13 not in out


def test_baseline_change_boundary_and_safe_paths(tmp_path):
    files={'numerical example.tex':'old numerical','model.tex':'fixed theory','supplement.tex':'fixed supplement'}
    records=[]
    for name,text in files.items():
        path=tmp_path/name;path.write_text(text);records.append(dict(relative_path=name,sha256=p.sha(path)))
    (tmp_path/'numerical example.tex').write_text('new numerical')
    assert p.verify_preservation(tmp_path,records)['all_other_existing_members_byte_identical']
    (tmp_path/'model.tex').write_text('changed')
    with pytest.raises(ValueError):p.verify_preservation(tmp_path,records)
    for bad in ['../escape','C:/absolute','']:
        with pytest.raises(ValueError):p.safe_member(tmp_path,bad)


def test_rendered_and_copied_asset_changes_reject(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ASSETS',['figure3']);monkeypatch.setattr(p,'code_bindings',lambda:{'source':'fixed'})
    folder=tmp_path/'numerical_data/figure3';folder.mkdir(parents=True);image=folder/'panel_a.png';image.write_bytes(b'image')
    result=dict(status='REPRODUCED_VISUAL_QA_PENDING',config_sha256='config',code_sha256={'source':'fixed'},
                assets=[dict(asset='figure3',native={'outputs':{'panel_a.png':p.sha(image)}})])
    (tmp_path/'REPRODUCTION_RESULT.json').write_text(json.dumps(result))
    p.verify_reproduction(tmp_path,'config')
    records=[dict(relative_path=image.relative_to(tmp_path).as_posix(),sha256=p.sha(image))]
    p.verify_bundle_assets(tmp_path,records);image.write_bytes(b'changed')
    with pytest.raises(ValueError):p.verify_reproduction(tmp_path,'config')
    with pytest.raises(ValueError):p.verify_bundle_assets(tmp_path,records)


def test_scientific_prose_native_counts_not_orchestration():
    text=' '.join(p.scientific_paragraphs(data()))
    assert '2 aggregate OD rows (3 original rows)' in text
    assert '120/160' in text and 'without renormalization' in text
    assert 'C01' not in text and 'deadline_epoch' not in text


def test_exact_uid_and_logical_role_coordinate_mapping():
    physical=pd.DataFrame(dict(physical_node_uid=['u1','u2'],partC_rev1_lon=[-90.,-91.],partC_rev1_lat=[35.,36.]))
    b=pd.DataFrame(dict(tail_node_id=['b'],head_node_id=['b'],tail_longitude=[-89.],head_longitude=[-89.],tail_latitude=[34.],head_latitude=[34.]))
    xy=g.common_coordinates(['u1','b','n8_od_origin:u2','n8_od_destination:u1'],physical,b)
    assert xy.tolist()==[[-90.,35.],[-89.,34.],[-91.,36.],[-90.,35.]]
    with pytest.raises(ValueError):g.common_coordinates(['missing'],physical,b)
    physical.loc[1,'physical_node_uid']='u1'
    with pytest.raises(ValueError):g.common_coordinates(['u1'],physical,b)


def test_stability_label_permutation_and_common_population():
    score,cross=g.information([1,1,2,2],[9,9,8,8])
    assert score['ARI']==pytest.approx(1) and score['NMI_arithmetic']==pytest.approx(1)
    assert cross.observations.sum()==4
    inter=pd.DataFrame(dict(suite_id=['s02','s03'],node_id=['a','b']))
    scores,_,j=g.stability_frames(['s02','s03'],np.array([[1,1,-1],[2,2,-1]]),[True,True,False],inter)
    assert scores.common_population_n.eq(2).all() and j.loc[j.suite_a!=j.suite_b,'boundary_Jaccard'].iloc[0]==0
    with pytest.raises(ValueError):g.stability_frames(['s02','s03'],np.array([[1,-1],[2,2]]),[True,True],inter)


def test_segments_reject_out_of_range_not_silent_wrong_endpoint():
    xy=np.array([[-90.,35.],[-89.,36.]])
    assert g.segments([0],[0],[1],xy).iloc[0].tolist()==[-90.,35.,-89.,36.]
    with pytest.raises(ValueError):g.segments([1],[0],[1],xy)


def test_geometry_export_uses_fresh_native_state_and_preserves_inputs(tmp_path):
    source=tmp_path/'old';source.mkdir();out=tmp_path/'new';out.mkdir()
    class Ctx:
        root=out;source_root=source
        c={'phase_id':'TEST','run_id':'toy','suite_ids':['s02','s03','s04','s05','s06'],
           'resolutions':dict(zip(['s02','s03','s04','s05','s06'],[.2,.3,.4,.5,.6]))}
        def require_action(self,action):assert action=='publication'
        def guard(self,*args):pass
        def bind(self,path,role,expected=None):
            digest=p.sha(path)
            if expected is not None:assert digest==expected
            return digest
        def output_path(self,relative):return self.root/relative
        def flush_bindings(self):pass
    ctx=Ctx();inputs={}
    def save(name,value):
        path=source/(name+'.json');path.write_text(json.dumps(value));inputs[name]={'path':str(path),'sha256':p.sha(path)};return path
    def parquet(name,frame):
        path=source/(name+'.parquet');frame.to_parquet(path,index=False);inputs[name]={'path':str(path),'sha256':p.sha(path)}
    graph=dict(tail=np.array([0,0,1,2]),head=np.array([0,1,2,0]),node_ids=np.array(['b0','u1','u2']),
               edge_ids=np.array(['B0','R0','R1','R2']),edge_role=np.array([0,1,1,1]),backbone_count=np.array(1))
    path=source/'common_graph.npz';np.savez(path,**graph);inputs['common_graph']={'path':str(path),'sha256':p.sha(path)}
    path=source/'partitions.npz';np.savez(path,suite_ids=np.array(ctx.c['suite_ids']),edge_region=np.tile([-1,0,1,2],(5,1)))
    inputs['partitions']={'path':str(path),'sha256':p.sha(path)}
    parquet('backbone',pd.DataFrame(dict(tail_node_id=['b0'],head_node_id=['b0'],tail_longitude=[-90.],head_longitude=[-90.],tail_latitude=[35.],head_latitude=[35.])))
    parquet('interfaces',pd.DataFrame([dict(suite_id=s,region_index=i,node_index=i,node_id=graph['node_ids'][i]) for s in ctx.c['suite_ids'] for i in range(3)]))
    parquet('regions',pd.DataFrame([dict(suite_id=s,region_index=i,physical_arc_count=1) for s in ctx.c['suite_ids'] for i in range(3)]))
    parquet('od_endpoints',pd.DataFrame(dict(road_node_idx=[1],physical_node_uid=['u1'])))
    path=source/'coordinates.csv';pd.DataFrame(dict(physical_node_uid=['u1','u2'],partC_rev1_lon=[-89.,-88.],partC_rev1_lat=[36.,37.])).to_csv(path,index=False)
    inputs['physical_coordinates']={'path':str(path),'sha256':p.sha(path)}
    ac=source/'acnhn/s03/graph.npz';ac.parent.mkdir(parents=True)
    np.savez(ac,tail=graph['tail'],head=graph['head'],edge_ids=graph['edge_ids'],backbone_count=1,pair_keys=np.arange(3),closed_mask=[True,False,False,False])
    save('support_freeze',dict(status='PASS',suites=[dict(suite_id=s,graph_sha256=p.sha(ac),region_records=[dict(pairs=1) for _ in range(3)]) for s in ctx.c['suite_ids']]))
    save('ablation_plan',dict(regions=[dict(region_index=i%3,stratum=s) for s in ['large','small','good','bad'] for i in range(4)]))
    flow=np.array([1.,2.,3.,4.]);path=source/'flow.npz';np.savez(path,flow=flow)
    save('reporting_results',[dict(case={'case_id':'AC_s03_ICNN_open'},status='CONVERGED',final_certificate_fresh=True,
        native_artifacts={'final_flow.npz':{'path':str(path),'sha256':p.sha(path)}},certificate={'state_hash':g.array_hash({'flow':flow})}),
        dict(case={'case_id':'AC_s03_ICNN_i95'},status='UNAVAILABLE',final_certificate_fresh=False)])
    cfg=out/'geometry.json';cfg.write_text(json.dumps(dict(phase_id='TEST',run_id='toy',inputs=inputs,output_relative='geometry')))
    before={str(x):p.sha(x) for x in source.rglob('*') if x.is_file()}
    result=g.export(ctx,cfg,deadline=1e12)
    assert result['map_dispositions'][0]['geometry_flow_available'] is True
    assert result['map_dispositions'][1]['geometry_flow_available'] is False
    assert len(pd.read_csv(out/'geometry/primary_i95_map_flows.csv'))==4
    assert len(pd.read_csv(out/'geometry/closure_geometry.csv'))==1
    assert before=={str(x):p.sha(x) for x in source.rglob('*') if x.is_file()}


@pytest.mark.parametrize('kind',['region_map','location_map','flow_map','iqr'])
def test_new_plot_kinds_draw_synthetic_canvas_and_hide_map_axes(kind):
    d=data();spec=dict(title='Synthetic',kind=kind,x='',y='',group='',xlabel='',ylabel='')
    if kind in ['region_map','location_map']:
        frame=d['region_geometry'] if kind=='region_map' else d['region_locations']
    elif kind=='flow_map':
        frame=d['primary_map_flows'].copy();frame.loc[1,'edge_kind']='REGIONAL';frame.loc[1,'flow']=-2
        spec.update(y='flow',colorbar='native flow')
    else:
        frame=d['ablation_summary'].loc[lambda f:f.stratum.eq('large')]
        spec.update(x='training_fraction',y='median_WMAPE',group='model_family')
    fig,ax=plot.plt.subplots();plot.draw(ax,plot.eligible(frame,spec),spec);fig.canvas.draw()
    if 'map' in kind:assert not ax.axison
    if kind=='iqr':assert [v.get_text() for v in ax.get_xticklabels()]==['10','25','50','70','100']
    plot.plt.close(fig)
