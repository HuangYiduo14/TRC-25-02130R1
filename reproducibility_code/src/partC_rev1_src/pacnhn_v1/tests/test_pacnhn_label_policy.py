"""Synthetic label policy tests against installed modules; no campaign data."""
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[3]
CONFIG=REPO/'partC_doc_icnn/pacnhn_v1/training_config_v3.json'

@pytest.fixture(scope="module")
def modules():
    from partC_rev1_src.pacnhn_v1 import regional, training
    return regional, training

def config():
    return json.loads(CONFIG.read_text())

def toy(r,tmp_path):
    path=tmp_path/'graph.npz'
    r.save_npz(path,tail=np.array([0],np.int64),head=np.array([1],np.int64),
        node_ids=np.array(['P0','P1']),edge_ids=np.array(['edge']),base=np.ones(1),k4=np.zeros(1),
        pair_source=np.array([0],np.int64),pair_target=np.array([1],np.int64),pair_keys=np.array([1],np.uint64),
        baseline_open=np.ones(1),baseline_i95=np.ones(1)*2)
    return r.load_region_graph(path,expected_sha256=r.sha(path))

def label_config(r,t,graph):
    return dict(phase_id='PACNHN_V1',run_id='synthetic',region_id='s03:000000',graph_path=graph.source_path,
        graph_sha256=graph.graph_sha256,deadline_epoch=time.time()+60,per_label_seconds=20.,
        relative_gap_tolerance=1e-6,**t.label_solver_settings(config()))

def test_v3_exact_six_key_change_and_unchanged_clock(modules):
    _,t=modules;old=json.loads((REPO/'partC_doc_icnn/pacnhn_v1/training_config_v2.json').read_text());new=config()
    changed={k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    assert changed=={'label_solver_method','label_retry_solver_method','label_retry_max_iterations',
        'label_cfw_reset_on_bound','label_retry_cfw_reset_on_bound','max_attempts_per_input'}
    assert new['stage_seconds']==old['stage_seconds']==86400
    assert new['per_label_seconds']==old['per_label_seconds']==20
    assert t.read_training_config(CONFIG)==new
    assert t.label_solver_settings(old)==dict(solver_method='FW',retry_solver_method='FW',max_iterations=2000,
        retry_max_iterations=2000,cfw_reset_on_bound=False,retry_cfw_reset_on_bound=False,max_attempts_per_input=3,
        iterate_selection="LAST_CERTIFIED",retry_iterate_selection="LAST_CERTIFIED")

@pytest.mark.parametrize('key,value',[
    ('label_solver_method','INVALID'),('label_retry_solver_method','MSA'),
    ('label_max_iterations',True),('label_retry_max_iterations',0),('label_retry_max_iterations',20.5),
    ('max_attempts_per_input',False),('label_cfw_reset_on_bound',1),('label_retry_cfw_reset_on_bound','true')])
def test_invalid_training_policy_rejected_before_execution(modules,tmp_path,key,value):
    _,t=modules;c=config();c[key]=value;path=tmp_path/'invalid.json';path.write_text(json.dumps(c))
    with pytest.raises(ValueError):t.read_training_config(path)

@pytest.mark.parametrize('key,value',[
    ('solver_method','MSA'),('retry_solver_method','INVALID'),('max_iterations',True),
    ('retry_max_iterations',0),('retry_max_iterations',1.2),('cfw_reset_on_bound',1),
    ('retry_cfw_reset_on_bound','true'),('max_attempts_per_input',0)])
def test_invalid_adapter_policy_writes_nothing(modules,tmp_path,key,value):
    r,t=modules;g=toy(r,tmp_path);c=label_config(r,t,g);c[key]=value
    request=r.sample_requests(g,role='TRAIN',count=1,seed=7)[0]
    with pytest.raises(ValueError):r.generate_labels(c,tmp_path/'output',requests=[request])
    assert not (tmp_path/'output').exists()

@pytest.mark.parametrize('role',['TRAIN','VALID','TEST'])
def test_fourth_dispatch_preserves_all_attempt_slots_and_fifth_is_blocked(modules,tmp_path,monkeypatch,role):
    r,t=modules;g=toy(r,tmp_path);c=label_config(r,t,g);request=r.sample_requests(g,role=role,count=1,seed=7)[0]
    root=tmp_path/'labels';calls=[]
    def fail(region,u,**kwargs):calls.append(kwargs);raise RuntimeError('synthetic rejected episode')
    monkeypatch.setattr(r,'solve_label',fail)
    first=r.generate_labels(c,root,requests=[request]);assert first['attempted']==1
    assert calls[0]['method']=='FW' and calls[0]['max_iterations']==2000 and not calls[0]['cfw_reset_on_bound']
    cohort='blind_TEST' if role=='TEST' else role
    sample=root/'labels'/cohort/request['sample_id']
    # Both a terminal failed slot and an unfinished slot count globally.
    (sample/'attempt_000002').mkdir();(sample/'attempt_000002/FAILED.json').write_text('{"old":true}')
    (sample/'attempt_000003').mkdir();(sample/'attempt_000003/PARTIAL.bin').write_bytes(b'old partial evidence')
    frozen={str(p):r.sha(p) for p in sample.rglob('*') if p.is_file()}
    fourth=r.generate_labels(c,root,requests=[request]);assert fourth['attempted']==1
    assert calls[-1]['method']=='CFW' and calls[-1]['max_iterations']==20000 and calls[-1]['cfw_reset_on_bound'] is True
    record=r.read_json(sample/'attempt_000004/FAILED.json');policy=record['solver_policy']
    assert policy['attempt_number_including_prior_failures']==4 and policy['max_attempts_per_input']==4
    assert policy['resolved_method']=='CFW' and policy['resolved_max_iterations']==20000
    assert policy['resolved_cfw_reset_on_bound'] is True and policy['prior_attempts_preserved']
    assert {p:r.sha(p) for p in frozen}==frozen
    fifth=r.generate_labels(c,root,requests=[request])
    assert fifth['status']=='ATTEMPT_LIMIT_NO_EXECUTION' and fifth['attempted']==0 and fifth['attempt_limited']==1
    assert len(calls)==2 and not (sample/'attempt_000005').exists()
    assert all(p.parent.name==cohort for p in (root/'labels').glob('*/*'))
    if role=='TEST':
        with pytest.raises(PermissionError):r.load_admitted_dataset(root,role='TEST')

def test_success_receipt_resolved_policy_and_selected_label_reuse(modules,tmp_path,monkeypatch):
    r,t=modules;g=toy(r,tmp_path);c=label_config(r,t,g);root=tmp_path/'labels'
    request=r.sample_requests(g,role='TRAIN',count=1,seed=7)[0]
    def good(region,u,**kwargs):
        return dict(u=np.asarray(u),tau=np.ones(1),flow=np.zeros(1),record=dict(status='CONVERGED',admitted=True,
            quality='A',certificate={'RG_SPTT':0.},solver='SYNTHETIC',complete_label_elapsed_seconds=.001))
    monkeypatch.setattr(r,'solve_label',good)
    result=r.generate_labels(c,root,requests=[request]);record=r.read_json(result['rows'][0]['receipt'])
    assert record['solver_policy']['resolved_method']=='FW' and record['solver_policy']['resolved_max_iterations']==2000
    assert not record['solver_policy']['resolved_cfw_reset_on_bound']
    index_before=r.read_json(root/'REQUEST_INDEX.json');selected=next(iter(index_before.values()))['selected_label']
    monkeypatch.setattr(r,'solve_label',lambda *a,**k:pytest.fail('admitted label must not regenerate'))
    again=r.generate_labels(c,root,requests=[request]);assert again['reused']==1 and again['attempted']==0
    assert next(iter(r.read_json(root/'REQUEST_INDEX.json').values()))['selected_label']==selected

def test_default_adapter_preserves_fw_same_cap_without_retry_settings(modules,tmp_path,monkeypatch):
    r,t=modules;g=toy(r,tmp_path);c=label_config(r,t,g)
    for key in ('solver_method','retry_solver_method','retry_max_iterations','cfw_reset_on_bound','retry_cfw_reset_on_bound','max_attempts_per_input'):c.pop(key)
    calls=[]
    def fail(region,u,**kwargs):calls.append(kwargs);raise RuntimeError('deliberate')
    monkeypatch.setattr(r,'solve_label',fail);req=r.sample_requests(g,role='TRAIN',count=1,seed=7)
    for _ in range(2):r.generate_labels(c,tmp_path/'labels',requests=req)
    assert [(v['method'],v['max_iterations'],v['cfw_reset_on_bound']) for v in calls]==[('FW',2000,False)]*2

def test_region_slice_forwards_original_deadline_and_explicit_policy(modules,tmp_path,monkeypatch):
    r,t=modules;g=toy(r,tmp_path);c=config();capture={};deadline=time.time()+60
    ctx=SimpleNamespace(c={'run_id':'synthetic'},bind=lambda *a:None,guard=lambda **k:None)
    record=dict(path=str(g.source_path),sha256=g.graph_sha256,key='s03:000000')
    monkeypatch.setattr(t,'load_region_graph',lambda *a,**k:g)
    monkeypatch.setattr(t,'native_scale',lambda *a:(np.ones(1),{'graph_sha256':g.graph_sha256}))
    monkeypatch.setattr(t,'_requests_for_region',lambda *a,**k:[{'sentinel':True}])
    def generated(settings,*args,**kwargs):
        capture.update(settings);return {'status':'BLOCKED_REGIONAL_INVARIANT'}
    monkeypatch.setattr(t,'generate_labels',generated)
    monkeypatch.setattr(t,'load_admitted_dataset',lambda *a,**k:pytest.fail('no labels/models needed in forwarding test'))
    state=t.region_slice(ctx,record,tmp_path/'region',c,'cpu',deadline,120)
    assert state['status']=='BLOCKED_REGIONAL_INVARIANT'
    for k,v in t.label_solver_settings(c).items():assert capture[k]==v
    assert capture['deadline_epoch']<=deadline and capture['per_label_seconds']==20 and capture['relative_gap_tolerance']==1e-6
    assert capture['graph_sha256']==g.graph_sha256
