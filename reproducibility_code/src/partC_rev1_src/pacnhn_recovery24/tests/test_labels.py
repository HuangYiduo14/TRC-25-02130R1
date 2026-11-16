import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from partC_rev1_src.pacnhn_recovery24.labels import gap_progress,native_json,new_json,require_selection_freeze,sampling_scale

def test_progress_uses_time_and_best_certificates():
    record={'complete_label_elapsed_seconds':20,'history':[
        {'elapsed_seconds':1,'certificate':{'RG_SPTT':.8}},
        {'elapsed_seconds':9,'certificate':{'RG_SPTT':.5}},
        {'elapsed_seconds':11,'certificate':{'RG_SPTT':.3}},
        {'elapsed_seconds':19,'certificate':{'RG_SPTT':.7}}]}
    assert gap_progress(record)==pytest.approx(.4)
    record['history'][1].pop('elapsed_seconds')
    assert gap_progress(record)==pytest.approx(.625)

def test_no_unmeasured_time_escalation():
    r={'complete_label_elapsed_seconds':20,'history':[{'iteration':10,'certificate':{'RG_SPTT':.01}}]}
    assert gap_progress(r) is None

def test_numpy_boolean_writer_and_append_only(tmp_path):
    path=tmp_path/'r.json';new_json(path,{'good':np.bool_(True),'n':np.int64(3),'a':np.array([1.])})
    assert json.loads(path.read_text())=={'good':True,'n':3,'a':[1.]}
    with pytest.raises(FileExistsError):new_json(path,{})

def test_sampling_reference_is_unchanged_native_rule():
    region=SimpleNamespace(baseline_open=np.array([0.,3.]),baseline_i95=np.array([0.,5.]))
    np.testing.assert_array_equal(sampling_scale(region,{}),[.05,5.])
    region.baseline_open[:]=0;region.baseline_i95[:]=0
    with pytest.raises(ValueError):sampling_scale(region,{})
    np.testing.assert_array_equal(sampling_scale(region,{'physical_capacity_reference':200,'physical_capacity_reference_provenance':'fixed physical capacity'}),[2.,2.])

def test_no_test_access_before_valid_selection_freeze(tmp_path):
    from partC_rev1_src.pacnhn_v1.control import sha
    p=tmp_path/'freeze.json';new_json(p,{'status':'IN_PROGRESS','test_targets_opened':False})
    ctx=SimpleNamespace(root=tmp_path,bind=lambda path,role,expected:sha(path))
    with pytest.raises(PermissionError):require_selection_freeze(ctx,p,sha(p))

@pytest.mark.parametrize('marker',['training/MODEL_SELECTION_FREEZE.json','training/REGISTRY_FREEZE.json','holdout/TEST_OPENED.json'])
def test_training_label_changes_stop_at_any_freeze_marker(tmp_path,marker):
    from partC_rev1_src.pacnhn_recovery24.labels import require_training_labels_open
    new_json(tmp_path/marker,{})
    with pytest.raises(PermissionError):require_training_labels_open(SimpleNamespace(root=tmp_path))

def test_attempt_slot_is_durable_before_a_solver_result(tmp_path):
    from partC_rev1_src.pacnhn_recovery24.labels import begin_attempt
    path=tmp_path/'attempt_000005';begin_attempt(path,'a'*64,'TRAIN',5)
    assert json.loads((path/'START.json').read_text())['attempt']==5
    with pytest.raises(FileExistsError):begin_attempt(path,'a'*64,'TRAIN',5)


def test_parallel_parent_resource_failure_accounts_every_unstarted_job(tmp_path,monkeypatch):
    from partC_rev1_src.pacnhn_recovery24 import labels
    from partC_rev1_src.pacnhn_v1 import training
    import time
    stopped=[]
    pool=object()
    monkeypatch.setattr(labels,'ProcessPoolExecutor',lambda **kw:pool)
    monkeypatch.setattr(training,'_stop_own_pool',lambda value:stopped.append(value))
    def fail(**kwargs):raise MemoryError('aggregate worker RSS')
    ctx=SimpleNamespace(root=tmp_path,action='labels',guard=fail)
    jobs=[('config',{'suite_id':'s03','region_index':i}) for i in (1,2)]
    result=labels._parallel(ctx,None,jobs,time.time()+60,2)
    assert stopped==[pool]
    assert [r['job_key'] for r in result]==['s03:000001','s03:000002']
    assert {r['status'] for r in result}=={'UNATTEMPTED_RESOURCE_LIMIT'}

def test_complete_label_roundtrip_and_role_tampering(tmp_path):
    from partC_rev1_src.pacnhn_v1.control import sha
    from partC_rev1_src.pacnhn_v1.tests.test_pacnhn_regional_best_state import graph
    from partC_rev1_src.pacnhn_recovery24 import regional_solver as solver
    from partC_rev1_src.pacnhn_recovery24.labels import begin_attempt,_save_attempt,validate_saved_label
    region=graph(solver,tmp_path);q=np.array([3.])
    result=solver.solve_label(region,q,max_seconds=30,max_iterations=6,method='CFW',iterate_selection='BEST_CERTIFIED_RG')
    region.c0=result['c0'].copy()
    folder=tmp_path/'attempt_000001';begin_attempt(folder,result['record']['input_identity'],'TEST',1)
    saved=_save_attempt(folder,result,dict(role='TEST',sample_id='sample',attempt=1))
    ctx=SimpleNamespace(bind=lambda path,role,expected:sha(path))
    receipt=json.loads(Path(saved['receipt']).read_text())
    tau=validate_saved_label(ctx,receipt,saved['path'],region=region,input_id=result['record']['input_identity'],role='TEST',sample_id='sample',q=q)
    np.testing.assert_array_equal(tau,result['tau'])
    receipt['role']='TRAIN'
    with pytest.raises(ValueError):validate_saved_label(ctx,receipt,saved['path'],region=region,input_id=result['record']['input_identity'],role='TEST',sample_id='sample',q=q)

def test_timing_instrumentation_is_only_solver_logic_delta():
    folder=Path(__file__).parents[1]
    original=(folder.parent/'pacnhn_v1/regional.py').read_text(encoding='utf-8')
    copied=(folder/'regional_solver.py').read_text(encoding='utf-8')
    expected=original.replace('from .control import','from partC_rev1_src.pacnhn_v1.control import').replace(
        'from .oracle import','from partC_rev1_src.pacnhn_v1.oracle import').replace(
        'from .solver import','from partC_rev1_src.pacnhn_v1.solver import').replace(
        'history.append(dict(iteration=iteration, certificate=latest,',
        'history.append(dict(iteration=iteration, elapsed_seconds=time.perf_counter()-mono_started, certificate=latest,')
    assert copied==expected

@pytest.mark.parametrize('method,reset',[('FW',False),('CFW',False),('CFW',True)])
def test_timed_copy_preserves_complete_numerical_state(tmp_path,method,reset):
    from partC_rev1_src.pacnhn_v1.tests.test_pacnhn_regional_best_state import graph
    from partC_rev1_src.pacnhn_v1 import regional as old
    from partC_rev1_src.pacnhn_recovery24 import regional_solver as new
    region=graph(old,tmp_path)
    kwargs=dict(max_seconds=30,max_iterations=6,relative_gap_tolerance=1e-30,method=method,
                cfw_reset_on_bound=reset,iterate_selection='BEST_CERTIFIED_RG')
    a=old.solve_label(region,[3.],**kwargs)
    b=new.solve_label(new.load_region_graph(region.source_path),[3.],**kwargs)
    for name in ('u','tau','c0','flow','edge_cost','pair_keys'):np.testing.assert_array_equal(a[name],b[name])
    assert a['record']['certificate']==b['record']['certificate']
    times=[r['elapsed_seconds'] for r in b['record']['history']]
    assert times==sorted(times) and min(times)>=0
