"""Synthetic independent metrics audit checks; never reads campaign arrays."""
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from partC_rev1_src.pacnhn_v1 import final_assignment_metrics as a


def binding(path):
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8');return binding(path)


def npz(path,**values):
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**values);return binding(path)


def matrix():
    rows=[]
    for solver in ('CFW','FW','MSA'):
        for scenario in a.SCENARIOS:
            rows.append(dict(case_id=f'P_{solver}_{scenario}',network='P',suite_id=None,family=None,solver=solver,scenario=scenario,priority='PRIMARY'))
    for sid in ('s03','s04','s02','s05','s06'):
        for family in (('FFTT','POWER4','XGBOOST','ICNN') if sid=='s03' else ('FFTT',)):
            for scenario in a.SCENARIOS:
                rows.append(dict(case_id=f'AC_{sid}_{family}_{scenario}',network='AC',suite_id=sid,family=family,
                    solver='MSA' if family=='XGBOOST' else 'FW',scenario=scenario,priority='PRIMARY' if sid=='s03' else 'ADDITIONAL'))
    return rows


def case(family=None,solver='CFW',reference=False,scenario='open'):
    return dict(case_id='case',network='AC' if family else 'P',family=family,suite_id='s03' if family else None,
        scenario=scenario,solver=solver,assignment_role='REFERENCE_SELF' if reference else 'SURROGATE' if family else 'INDEPENDENT_CONTROL')


def certificate(flow,c,*,start=100.,status='CONVERGED'):
    potential=c['family']!='XGBOOST';ref=c['assignment_role']=='REFERENCE_SELF'
    return dict(status=status,final_certificate_fresh=True,reported_state='LAST_FRESH_CERTIFIED_STATE',reported_state_iteration=2,
        graph_identity='graph',OD_identity='od_'+c['scenario'],objective_has_potential=potential,
        nominal_tolerance=1e-4 if ref else 1e-3,method=c['solver'],complete_query_start_epoch=start,
        deadline_epoch=start+(10800 if ref else 2700),elapsed_seconds=25.,
        independent_cold_control=c['network']=='P',intent=dict(initialization='COLD_FEASIBLE_AON',complete_query_start_epoch=start,deadline_epoch=start+2700),
        certificate=dict(fresh=True,state_hash=a.array_hash({'flow':flow}),TSTT=100.001,SPTT=100.,raw_signed_gap=100.001-100.,
            arithmetic_allowance=100.001e-10,RG_SPTT=(100.001-100.)/100.,value=50. if potential else None,
            interpretation='CONVEX_POTENTIAL_FW_GAP' if potential else 'POINTWISE_VI_RESIDUAL_NO_POTENTIAL'))


def verify(result,flow,c):
    return a.verify_certificate(result,flow,graph_identity='graph',scenario_OD_identity='od_'+c['scenario'],case=c,
        stage3_clock=dict(original_stage_start_epoch=100.,deadline_epoch=43300.))


def test_exact_matrix_and_missing_unavailable_rows():
    rows=matrix();out=[dict(case=r,status='UNRUN_STAGE_TIME_LIMIT') for r in rows]
    assert len(a.planned_cases(rows,out)[0])==22
    with pytest.raises(ValueError,match='Missing/duplicate'):a.planned_cases(rows,out[:-1])
    changed=copy.deepcopy(rows);changed[0]['solver']='FW'
    with pytest.raises(ValueError,match='matrix differs'):a.planned_cases(changed,out)


@pytest.mark.parametrize('reference,predicted,wmape,zeros,newpositive',[
    ([0,2,0,3],[4,2,0,2],100.,2,1),([0,0,0,0],[0,2,0,0],None,4,1),([0,2,0,3],[0,2,0,3],0.,2,0)])
def test_ratio_of_sums_retains_zero_references(reference,predicted,wmape,zeros,newpositive):
    r,p=np.array(reference,float),np.array(predicted,float)
    stats,per=a.independent_metrics(r,p,np.array([1,1,0,0],bool),np.ones(4),np.zeros(4))
    assert stats['BACKBONE']['wmape_percent']==wmape
    assert stats['BACKBONE']['reference_zero_link_count']==zeros
    assert stats['BACKBONE']['reference_zero_positive_prediction_count']==newpositive
    assert np.isnan(per['signed_relative_error'][0])
    assert per['signed_bounded_error'][2]==0
    if wmape is None:assert np.isnan(per['backbone_wmape_contribution_percent']).all()


def test_independent_control_is_not_forced_to_reference_self():
    r=np.array([1.,2.]);p=np.array([2.,3.])
    result,_=a.independent_metrics(r,p,np.ones(2,bool),np.ones(2),np.zeros(2))
    assert result['BACKBONE']['wmape_percent']==pytest.approx(200/3)


@pytest.mark.parametrize('family,solver,reference',[ (None,'CFW',True),(None,'MSA',False),('XGBOOST','MSA',False),('ICNN','FW',False)])
def test_reference_cap_and_msa_interpretations(family,solver,reference):
    c=case(family,solver,reference);flow=np.ones(4);r=certificate(flow,c)
    checked=verify(r,flow,c)
    assert checked['declared_cap_seconds']==(10800 if reference else 2700)
    assert checked['residual_kind']==('VI_RESIDUAL' if family=='XGBOOST' else 'FW_GAP')


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(final_certificate_fresh=False),lambda r:r['certificate'].update(state_hash='changed'),
    lambda r:r.update(graph_identity='changed'),lambda r:r.update(OD_identity='changed'),
    lambda r:r['certificate'].update(RG_SPTT=.1),lambda r:r['certificate'].update(raw_signed_gap=-1.),
    lambda r:r.update(deadline_epoch=99999.),lambda r:r.update(complete_query_start_epoch=99.),
    lambda r:r.update(nominal_tolerance=.1),lambda r:r.update(objective_has_potential=False),
    lambda r:r['intent'].update(initialization='WARM_START')])
def test_certificate_rejects_false_fresh_state_or_scope(mutation):
    c=case();flow=np.ones(4);r=certificate(flow,c);mutation(r)
    with pytest.raises(ValueError):verify(r,flow,c)


def test_measured_overrun_disclosed_without_censoring():
    c=case();flow=np.ones(4);r=certificate(flow,c,status='TIME_LIMIT');r['elapsed_seconds']=2701.2
    assert verify(r,flow,c)['observed_complete_wall_overrun_seconds']==pytest.approx(1.2)


def test_delta_is_traversal_not_demand():
    d=a.delta_statistics(np.array([10.,5.,0.]),np.array([0.,8.,2.]))
    assert d['positive_traversal_change']==5 and d['negative_traversal_change_magnitude']==10 and d['net_traversal_change']==-5
    assert 'not original OD loss' in d['interpretation']


def fixture(tmp_path,monkeypatch):
    from partC_rev1_src.pacnhn_v1 import handoff
    monkeypatch.setattr(a,'B_COUNT',4);monkeypatch.setattr(a,'KEY_COUNT',2)
    root=tmp_path/'campaign';root.mkdir();now=time.time();stage_start=now-5000
    clock=dict(original_stage_start_epoch=stage_start,deadline_epoch=stage_start+43200)
    put(root/'STAGE3_CLOCK.json',clock);put(root/'STAGE3_RESULT.json',dict(status='COMPLETE_WITH_RECORDED_OUTCOMES',clock=clock,elapsed_seconds=3000,rows=22,primary_rows=14))
    ids=np.array(['b0','b1','b2','b3']);base=np.ones(4);k4=np.zeros(4);closed=np.array([0,1,0,0],bool)
    backbone=pd.DataFrame(dict(link_id=ids,tail_longitude=[-80.]*4,tail_latitude=[30.]*4,head_longitude=[-79.]*4,head_latitude=[31.]*4))
    (root/'topology').mkdir();backbone.to_parquet(root/'topology/backbone.parquet')
    npz(root/'topology/common_graph.npz',edge_ids=ids,closed_mask=closed,base=base,k4=k4)
    npz(root/'topology/common_od.npz',demand=np.ones(3))
    put(root/'topology/RESULT.json',dict(status='PASS',OD_identity='common_od',graph_identity='graph'))
    reports=[]
    for sid in ('s02','s03','s04','s05','s06'):
        b=npz(root/'acnhn'/sid/'graph.npz',edge_ids=ids,closed_mask=closed,base=base,k4=k4)
        report=dict(suite_id=sid,active_regions=2,graph_identity='graph',OD_identity='common_od',graph_sha256=b['sha256'])
        put(root/'acnhn'/sid/'REPORT.json',report);reports.append(report)
    support=put(root/'SUPPORT_FREEZE.json',dict(status='PASS',suites=reports))
    put(root/'COMPLETE_WITNESS_AUDIT.json',dict(status='PASS',support_freeze_sha256=support['sha256']))
    coverage=[dict(suite_id=s,family=f,expected=2,available=2 if s=='s03' or f=='FFTT' else 0,
        qualified=0,complete_family=(s=='s03' or f=='FFTT')) for s in ('s02','s03','s04','s05','s06') for f in ('FFTT','POWER4','ICNN','XGBOOST')]
    put(root/'training/RESULT.json',dict(status='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE',registry_sha256='registry',coverage=coverage))
    rows=matrix();cases=[dict(case_id='P_REFERENCE_'+s,network='P',suite_id=None,family=None,solver='CFW',scenario=s,assignment_role='REFERENCE_SELF') for s in a.SCENARIOS]
    cases += [dict(r,assignment_role='INDEPENDENT_CONTROL' if r['network']=='P' else 'SURROGATE') for r in rows]
    flows={'open':np.array([0.,2.,3.,0.]),'i95':np.array([0.,0.,4.,0.])}
    serves={'open':np.array([1,1,0],bool),'i95':np.array([1,0,0],bool)}
    results={};outcomes=[]
    for c in cases:
        ref=c['assignment_role']=='REFERENCE_SELF';s=c['scenario'];f=flows[s].copy()
        if not ref:f[2]+=.5
        directory=root/('references' if ref else 'assignments')/(s if ref else c['case_id'])
        r=certificate(f,c,start=stage_start+10);r['certificate']['state_hash']=a.array_hash({'flow':f})
        if not ref:
            r['case']={k:v for k,v in c.items() if k!='assignment_role'}
            r['intent'].update(registry_sha256='registry',phase_id='PACNHN_V1',run_id='synthetic',OD_identity='common_od')
            put(root/'intents'/f"{c['case_id']}.json",r['intent']);outcomes.append(r)
        put(directory/'RESULT.json',r);npz(directory/'final_flow.npz',flow=f);npz(directory/'coverage.npz',serve=serves[s])
        put(directory/'HISTORY.json',[dict(iteration=2,state_hash=r['certificate']['state_hash'])]);results[c['case_id']]=(r,f)
    put(root/'ASSIGNMENT_MATRIX.json',rows);put(root/'ASSIGNMENT_RESULTS.json',outcomes)
    keydir=root/'metrics/keys';keydir.mkdir(parents=True)
    keys=pd.DataFrame(dict(link_id=ids[1:3],native_index=[1,2],lambda_reference=flows['open'][1:3]));keys.to_csv(keydir/'key_links_200.csv',index=False)
    full=pd.DataFrame(dict(link_id=ids,lambda_reference=flows['open']));full.to_csv(keydir/'open_backbone_reference.csv',index=False)
    km=dict(schema='pacnhn.fixed_keys.v1',key_ids=ids[1:3].tolist(),key_file_sha256=binding(keydir/'key_links_200.csv')['sha256'],
        reference_file_sha256=binding(keydir/'open_backbone_reference.csv')['sha256'],prediction_errors_used=False,
        reference_id='P_REFERENCE_open',reference_scenario='OPEN',target=2,backbone_count=4,key_selection_role='REFERENCE_ONLY',
        full_backbone_population_filtered=False,source_bindings=[binding(root/'references/open/final_flow.npz')],
        backbone_order_identity=a.array_hash({'link_id':ids}),open_reference_flow_identity=a.array_hash({'flow':flows['open']}))
    put(keydir/'KEY_SELECTION_FREEZE.json',km)
    analysis=root/'analysis';analysis.mkdir();keys.to_csv(analysis/'keys.csv',index=False)
    wide=backbone.copy();wide['native_index']=np.arange(4);wide['is_key']=[False,True,True,False];wide['represented_I95_closed']=closed
    methods=[];pop=[];long=[];paired={}
    keymask=np.array([0,1,1,0],bool)
    for c in cases:
        r,f=results[c['case_id']];s=c['scenario'];stats,per=a.independent_metrics(flows[s],f,keymask,base,k4)
        ident=dict(scenario=a.SCENARIOS[s],reference_case_id='P_REFERENCE_'+s,case_id=c['case_id'],method_id=a.method_id(c),assignment_role=c['assignment_role'],backbone_order_identity=km['backbone_order_identity'],served_OD_membership_equal=True,comparison_space='COMMON_NATIVE_BACKBONE',key_origin_scenario='OPEN',closed_backbone_count=int(closed.sum()) if s=='i95' else 0)
        for p,v in stats.items():pop.append(dict(v,population=p,flow_units='ANNUAL_KTON',cost_units='ANNUAL_KTON_SECONDS',**ident))
        frame=backbone.copy();frame['native_index']=np.arange(4);frame['reference_flow']=flows[s];frame['predicted_flow']=f
        for k,v in per.items():frame[k]=v
        frame['is_key']=keymask;frame['is_closed']=closed if s=='i95' else False
        for k,v in ident.items():frame[k]=v
        long.append(frame);wide['flow__'+c['case_id']]=f
        methods.append(dict(case_id=c['case_id'],method_id=a.method_id(c),network=c['network'],suite_id=c['suite_id'],scenario=a.SCENARIOS[s],status=r['status'],solver=c['solver'],RG_SPTT=r['certificate']['RG_SPTT'],elapsed_seconds=25.,wmape_backbone_percent=stats['BACKBONE']['wmape_percent'],wmape_key_percent=stats['KEY_LINKS']['wmape_percent'],reference_case_id='P_REFERENCE_'+s,assignment_role=c['assignment_role'],certificate_fresh=True,residual_kind='VI_RESIDUAL' if c['family']=='XGBOOST' else 'FW_GAP',nominal_tolerance=r['nominal_tolerance'],query_cap_seconds=r['deadline_epoch']-r['complete_query_start_epoch'],timing_boundary=a.TIMING_BOUNDARY,reference_status='CONVERGED',reference_RG_SPTT=r['certificate']['RG_SPTT'],served_OD_membership_equal=True,reported_state_iteration=2,reference_zero_link_count=stats['BACKBONE']['reference_zero_link_count'],reference_zero_positive_prediction_count=stats['BACKBONE']['reference_zero_positive_prediction_count']))
        paired.setdefault(a.method_id(c),{})[s]=f
    pd.DataFrame(methods).to_csv(analysis/'methods.csv',index=False);pd.DataFrame(pop).to_csv(analysis/'metric_populations.csv',index=False)
    pd.concat(long).to_csv(analysis/'backbone_links.csv',index=False);wide.to_csv(analysis/'native_backbone_vectors.csv',index=False)
    odcoverage=dict(open_closed_OD_membership_equal=False,open_served_OD_rows=2,closed_served_OD_rows=1,lost_OD_rows=1,gained_OD_rows=0,lost_OD_demand=1.,gained_OD_demand=0.,demand_units='ANNUAL_KTON',demand_renormalized=False)
    put(analysis/'I95_OD_COVERAGE.json',odcoverage);frames=[]
    for method,pair in paired.items():
        f=backbone.copy();f['method_id']=method;f['open_flow']=pair['open'];f['closed_flow']=pair['i95'];f['delta']=pair['i95']-pair['open'];f['is_closed']=closed
        for k in ('open_closed_OD_membership_equal','lost_OD_rows','gained_OD_rows','lost_OD_demand','gained_OD_demand'):f[k]=odcoverage[k]
        frames.append(f)
    pd.concat(frames).to_csv(analysis/'I95.csv',index=False)
    masks={'COMMON_SERVED':np.array([1,0,0],bool),'LOST_CLOSED_VS_OPEN':np.array([0,1,0],bool),'GAINED_CLOSED_VS_OPEN':np.zeros(3,bool),'NEITHER_SERVED':np.array([0,0,1],bool)}
    proof=put(tmp_path/'proof.json',dict(status='PASS',schema='PACNHN_FROZEN_REFERENCE_I95_MEMBERSHIP_V1',run_id='synthetic',demand_renormalized=False,common_OD_identity='common_od',scenario_OD_identities={s:'od_'+s for s in a.SCENARIOS},aggregate_OD_rows=3,closed_served_subset_of_open=True,represented_closure='synthetic',populations=[dict(population=k,aggregate_OD_rows=int(v.sum()),demand=float(v.sum()),membership_mask_identity=a.array_hash({'membership':v})) for k,v in masks.items()]))
    proof_sources=put(tmp_path/'proof_sources.json',dict(bindings=[binding(root/'topology/common_od.npz')]+[binding(root/'references'/s/n) for s in a.SCENARIOS for n in ('RESULT.json','coverage.npz','final_flow.npz')]))
    sourcefiles=[binding(p) for p in root.rglob('*') if p.is_file()]
    put(analysis/'PROVENANCE.json',dict(status='PASS',schema='PACNHN_ANALYSIS_V1',inputs=[],outputs=[binding(p) for p in analysis.iterdir() if p.is_file()]))
    sourcefiles.append(binding(analysis/'PROVENANCE.json'));inputs=put(tmp_path/'inputs.json',dict(files=sourcefiles))
    src=Path(a.__file__).parent;names=['final_assignment_metrics.py','final_training_limits.py','control.py','continuation.py','handoff.py','metrics.py','analysis_export.py','topology.py','tests/test_final_assignment_metrics.py']
    source_inventory=put(tmp_path/'sources.json',dict(files=[binding(src/n) for n in names]))
    campaign=put(tmp_path/'campaign.json',dict(run_id='synthetic'))
    cfg=dict(schema='PACNHN_FINAL_ASSIGNMENT_METRICS_CONFIG_V1',phase_id='PACNHN_V1',step_id='final_assignment_metrics',run_id='synthetic',tasks=sorted(a.TASKS),input_versions={'synthetic':True},expected_outputs=['RESULT.json'],acceptance_tests=['synthetic'],resources=dict(wall_seconds=120,single_process_rss_bytes=1024**3,tree_rss_bytes=24*1024**3,minimum_available_bytes=3*1024**3),campaign_config=campaign,stage4_clock={'path':'synthetic','sha256':'stage4'},clock_amendment={},deadline_epoch=now+1000,source_inventory=source_inventory,input_inventory=inputs,reference_membership_proof=proof,reference_membership_sources=proof_sources,output_dir=str(root/'audits/test'))
    cfg['audit_clock']=put(root/'final_audits/ASSIGNMENT_METRICS_CLOCK.json',dict(schema='PACNHN_FINAL_ASSIGNMENT_AUDIT_CLOCK_V1',phase_id='PACNHN_V1',run_id='synthetic',stage4_clock_sha256='stage4',start_epoch=now-10,deadline_epoch=now+1190))
    cb=put(tmp_path/'config.json',cfg);tr=put(tmp_path/'test_receipt.json',dict(status='PASS',exit_code=0,source_sha256=binding(src/'final_assignment_metrics.py')['sha256'],test_sha256=binding(Path(__file__))['sha256']))
    admission=put(tmp_path/'admission.json',dict(schema='PACNHN_FINAL_ASSIGNMENT_METRICS_ADMISSION_V1',decision='ADMIT_FINAL_ASSIGNMENT_METRICS_AUDIT',phase_id='PACNHN_V1',run_id='synthetic',config_sha256=cb['sha256'],source_inventory_sha256=source_inventory['sha256'],stage4_clock_sha256='stage4',audit_clock_sha256=cfg['audit_clock']['sha256'],test_receipt=tr))
    class Context:
        def __init__(self,*args,**kwargs):self.root=root
        def require_action(self,action):assert action=='analysis'
        def guard(self,**kwargs):pass
    class Budget:
        peak_rss=100;peak_tree_rss=100;minimum_available=4*1024**3
        def __init__(self,*args,**kwargs):pass
        def __call__(self):pass
    monkeypatch.setattr(a,'Context',Context);monkeypatch.setattr(a,'Budget',Budget)
    monkeypatch.setattr(handoff,'publication_window',lambda *args,**kwargs:dict(deadline_epoch=now+1000,stage_start_epoch=now-100,stage_deadline_epoch=now+13400))
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):monkeypatch.setenv(name,'1')
    return root,cb,admission


def test_complete_24_case_export_recomputed_and_sealed(tmp_path,monkeypatch):
    root,cfg,admit=fixture(tmp_path,monkeypatch)
    result=a.run(cfg['path'],admission_path=admit['path'],admission_sha256=admit['sha256'])
    assert result['status']=='COMPLETE_WITH_SCOPE_LIMITATIONS',result
    assert result['admitted_native_cases']==24
    ledger=pd.read_csv(root/'audits/test/CASE_LEDGER.csv')
    assert len(ledger)==24 and ledger.disposition.eq('VERIFIED_NATIVE_COMPARISON').all()
    assert (root/'audits/test/SEAL.json').exists()


def test_authenticated_source_change_fails_with_full_planned_ledger(tmp_path,monkeypatch):
    root,cfg,admit=fixture(tmp_path,monkeypatch)
    path=root/'references/open/final_flow.npz';path.write_bytes(b'changed')
    result=a.run(cfg['path'],admission_path=admit['path'],admission_sha256=admit['sha256'])
    assert result['status']=='FAIL' and 'Source bytes changed' in result['error']['message']
    assert len(pd.read_csv(root/'audits/test/CASE_LEDGER.csv'))==24


def test_metadata_identities_are_not_ignored():
    frame=pd.DataFrame(dict(method_id=['wrong']))
    with pytest.raises(ValueError,match='identity'):a.constant(frame,'method_id','right')
    with pytest.raises(ValueError,match='boolean'):a.constant(pd.DataFrame(dict(served=['false'])),'served',True)


def test_requested_npz_fields_exclude_unrelated_payload(tmp_path):
    path=tmp_path/'coverage.npz';npz(path,serve=np.ones(3,bool),initial_shortest_cost=np.ones(20))
    source=a.Sources(a.Reader(),[binding(path)])
    assert set(source.npz(path,['serve']))=={'serve'}


@pytest.mark.parametrize('change',[dict(deadline_epoch=1401),dict(start_epoch=201),dict(stage4_clock_sha256='other')])
def test_original_cumulative_audit_clock_is_not_reset(tmp_path,change):
    c=dict(schema='PACNHN_FINAL_ASSIGNMENT_AUDIT_CLOCK_V1',phase_id='PACNHN_V1',run_id='r',stage4_clock_sha256='s',start_epoch=100.,deadline_epoch=1300.)
    c.update(change)
    with pytest.raises(ValueError):a.audit_clock_window(c,root=tmp_path,binding={'path':str(tmp_path/'final_audits/ASSIGNMENT_METRICS_CLOCK.json')},run_id='r',stage4_binding={'sha256':'s'},window={'stage_start_epoch':50.,'stage_deadline_epoch':1400.},now=200.)


def rebind_fixture(cfg,admit,root):
    """Authenticate intentionally wrong synthetic exports to test semantic gates."""
    config=json.loads(Path(cfg['path']).read_text());ap=Path(admit['path'])
    provenance=root/'analysis/PROVENANCE.json';pv=json.loads(provenance.read_text())
    pv['outputs']=[binding(Path(r['path'])) for r in pv['outputs']];put(provenance,pv)
    idx=Path(config['input_inventory']['path']);rows=json.loads(idx.read_text())
    rows['files']=[binding(Path(r['path'])) for r in rows['files'] if Path(r['path']).exists()];config['input_inventory']=put(idx,rows)
    cb=put(Path(cfg['path']),config);av=json.loads(ap.read_text());av['config_sha256']=cb['sha256']
    return cb,put(ap,av)


@pytest.mark.parametrize('filename,column,value',[
    ('methods.csv','wmape_backbone_percent',99.),('methods.csv','residual_kind','VI_RESIDUAL'),
    ('metric_populations.csv','flow_units','OTHER'),('backbone_links.csv','signed_error',99.),
    ('I95.csv','lost_OD_rows',99),('keys.csv','lambda_reference',99.)])
def test_authenticated_wrong_export_fails_semantic_checks(tmp_path,monkeypatch,filename,column,value):
    root,cfg,admit=fixture(tmp_path,monkeypatch)
    path=root/'analysis'/filename;frame=pd.read_csv(path);frame.loc[0,column]=value;frame.to_csv(path,index=False)
    cfg,admit=rebind_fixture(cfg,admit,root)
    result=a.run(cfg['path'],admission_path=admit['path'],admission_sha256=admit['sha256'])
    assert result['status']=='FAIL',result
    assert 'Source bytes changed' not in result['error']['message']


def test_uncertified_initialization_timeout_is_explicit_unavailable(tmp_path,monkeypatch):
    root,cfg,admit=fixture(tmp_path,monkeypatch);key='P_FW_open'
    path=root/'ASSIGNMENT_RESULTS.json';outcomes=json.loads(path.read_text())
    previous=next(r for r in outcomes if r['case']['case_id']==key)
    replacement=dict(case=previous['case'],status='TIME_LIMIT',final_certificate_fresh=False)
    put(path,[replacement if r is previous else r for r in outcomes]);put(root/'assignments'/key/'RESULT.json',replacement)
    for filename in ('final_flow.npz','coverage.npz','HISTORY.json'):(root/'assignments'/key/filename).unlink()
    for filename in ('methods.csv','metric_populations.csv','backbone_links.csv','native_backbone_vectors.csv','I95.csv'):
        p=root/'analysis'/filename;frame=pd.read_csv(p)
        if filename=='methods.csv':
            frame.loc[frame.case_id==key,'status']='TIME_LIMIT';frame.loc[frame.case_id==key,'certificate_fresh']=False
            for column in ('wmape_backbone_percent','wmape_key_percent'):frame.loc[frame.case_id==key,column]=np.nan
        elif filename=='native_backbone_vectors.csv':frame=frame.drop(columns=['flow__'+key])
        elif filename=='I95.csv':frame=frame.loc[frame.method_id!='P/FW']
        else:frame=frame.loc[frame.case_id!=key]
        frame.to_csv(p,index=False)
    cfg,admit=rebind_fixture(cfg,admit,root)
    result=a.run(cfg['path'],admission_path=admit['path'],admission_sha256=admit['sha256'])
    assert result['status']=='COMPLETE_WITH_SCOPE_LIMITATIONS',result
    assert result['admitted_native_cases']==23
    assert result['case_dispositions'][key]=='VERIFIED_UNAVAILABLE_OR_UNRUN'
