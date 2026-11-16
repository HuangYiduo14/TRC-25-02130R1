import copy
import json
from pathlib import Path
import time

import numpy as np
import pytest

from partC_rev1_src.pacnhn_recovery24 import numerical_admission as n
from partC_rev1_src.pacnhn_v1.control import array_hash, sha, write_json
from partC_rev1_src.pacnhn_v1.final_icnn_guarantees import audit_model
from partC_rev1_src.pacnhn_v1.models import (
    Power4Model, MonotoneICNN, save_model, fit_xgboost_cost, XGBoostCostModel,
)


@pytest.mark.parametrize('eta',[0.,.2])
def test_power4_exact_integral_export_and_optional_fftt(eta):
    model=Power4Model([0.,2.],[.1,100.],[eta,eta])
    result=n.audit_power4(model)
    assert result['status']=='PASS' and len(result['checks'])==6
    if eta:
        with pytest.raises(ValueError):n.audit_power4(model,fftt=True)
    else:assert n.audit_power4(model,fftt=True)['status']=='PASS'


def test_incorrect_power_cost_is_not_admitted(monkeypatch):
    model=Power4Model([1.],[2.],[.3]);original=model.value_cost
    monkeypatch.setattr(model,'value_cost',lambda u:(original(u)[0],original(u)[1]+.1))
    assert n.audit_power4(model)['status']=='FAIL'


@pytest.mark.parametrize('amplitude',[0.,1.])
def test_old_icnn_pure_checks_are_reused(amplitude):
    model=MonotoneICNN([0.,2.],[.1,100.],eta=[.1,.2],width=5,seed=4,amplitude=amplitude).export_model()
    result=audit_model(model)
    assert result['status']=='PASS'
    assert result['finite_differences'] and result['accuracy_evaluated'] is False


def test_numpy_boolean_normalization_and_serialize_before_open(tmp_path):
    p=tmp_path/'report.json';n._new(p,{'passed':np.bool_(True),'number':np.int64(3),'x':np.array([1.])})
    assert json.loads(p.read_text())=={'passed':True,'number':3,'x':[1.]}
    bad=tmp_path/'bad.json'
    with pytest.raises(ValueError):n._new(bad,{'value':np.float64(np.nan)})
    assert not bad.exists()
    with pytest.raises(FileExistsError):n._new(p,{'changed':True})


def xgb_toy():
    u=np.array([[0.,0.],[1.,2.],[2.,1.],[3.,3.]])
    return fit_xgboost_cost(u,1.+.1*u,np.ones(2),np.ones(2),n_estimators=2,max_depth=1,
        device='cpu',n_jobs=1,structural_zero_mask=np.zeros(2,bool),structural_zero_evidence='SYNTHETIC_NO_ZERO_PATH')


def test_xgb_cost_only_positive_full_order_and_raw_export():
    model=xgb_toy();result=n.audit_xgboost(model,structural_zero_mask=np.zeros(2,bool))
    assert result['status']=='PASS' and result['raw_export_cost_exact'] and result['batch_single_order_exact']
    assert not result['has_potential'] and not result['convergence_guarantee']
    with pytest.raises(ValueError):n.audit_xgboost(model,structural_zero_mask=np.ones(2,bool))


def test_xgb_structural_zero_model_is_exact_cost_only():
    model=XGBoostCostModel(None,[0.,0.],[1.,1.],structural_zero_mask=np.ones(2,bool),metadata={'structural_zero_evidence':'SYNTHETIC_ALL_ZERO'})
    result=n.audit_xgboost(model,structural_zero_mask=np.ones(2,bool))
    assert result['status']=='PASS' and result['structural_zero_count']==2


def test_guard_interrupts_before_model_evaluation():
    model=Power4Model([1.],[1.],[1.])
    def stop():raise TimeoutError('synthetic')
    with pytest.raises(TimeoutError):n.audit_power4(model,guard=stop)


class Context:
    def __init__(self,tmp_path):
        self.root=tmp_path/'new';self.source_root=tmp_path/'old'
        self.root.mkdir();self.source_root.mkdir()
        self.c={'phase_id':'PACNHN_RECOVERY24','run_id':'toy','resources':{}}
        self.end=time.time()+120
    def require_action(self,action):
        assert action=='portfolio';return {'science_deadline_epoch':self.end}
    def guard(self,deadline=None):
        if time.time()>=min(self.end,deadline or self.end):raise TimeoutError('clock')
    def bind(self,path,role,expected=None):
        digest=sha(path)
        if expected is not None and digest!=expected:raise ValueError('source identity')
        return digest
    def flush_bindings(self):pass


def fixture(tmp_path,kind='POWER4'):
    ctx=Context(tmp_path);graph=ctx.source_root/'graph.npz'
    np.savez(graph,tail=np.array([0],np.int64),head=np.array([1],np.int64),node_ids=np.array(['o','d']),
        edge_ids=np.array(['e']),base=np.array([2.]),k4=np.array([0.]),pair_source=np.array([0],np.int64),
        pair_target=np.array([1],np.int64),pair_keys=np.array([1],np.uint64),baseline_open=np.array([1.]),
        baseline_i95=np.array([1.]),c0=np.array([2.]))
    if kind=='POWER4':model=Power4Model([2.],[1.],[.2])
    else:model=MonotoneICNN([2.],[1.],eta=[.2],width=5,seed=1).export_model()
    binding=save_model(model,ctx.source_root/'model.npz')
    frozen=dict(region_index=7,sha256=sha(graph),path=str(graph))
    write_json(ctx.source_root/'SUPPORT_FREEZE.json',{'status':'PASS','suites':[{'suite_id':'s03','region_records':[frozen]}]})
    rec=dict(suite_id='s03',region_index=7,family=kind,status='AVAILABLE',model_path=binding['path'],
        model_sha256=binding['sha256'],model_kind=model.kind,graph_sha256=frozen['sha256'],
        pair_keys_identity=array_hash({'pair_keys':np.array([1],np.uint64)}),artifact_hashes={binding['path']:binding['sha256']})
    return ctx,rec


@pytest.mark.parametrize('family',['POWER4','ICNN'])
def test_actual_record_identity_and_no_old_source_writes(tmp_path,family):
    ctx,rec=fixture(tmp_path,family);before={str(p):sha(p) for p in ctx.source_root.iterdir()}
    got=n.audit_records(ctx,[rec],deadline=ctx.end)[rec['model_sha256']]
    assert got['status']=='PASS' and got['record_bindings'][0]['graph_sha256']==rec['graph_sha256']
    assert got['model_sha256']==rec['model_sha256'] and not got['TEST_used']
    assert sha(got['report_path'])==got['report_sha256']
    assert {str(p):sha(p) for p in ctx.source_root.iterdir()}==before


@pytest.mark.parametrize('field,value',[('pair_keys_identity','a'*64),('graph_sha256','b'*64),('model_kind','OTHER')])
def test_graph_coordinate_kind_mismatch_fails_closed(tmp_path,field,value):
    ctx,rec=fixture(tmp_path);rec[field]=value
    assert n.audit_records(ctx,[rec],deadline=ctx.end)[rec['model_sha256']]['status']=='FAIL'


def test_changed_model_bytes_are_not_admitted(tmp_path):
    ctx,rec=fixture(tmp_path);p=Path(rec['model_path']);p.write_bytes(p.read_bytes()+b'changed')
    assert n.audit_records(ctx,[rec],deadline=ctx.end)[rec['model_sha256']]['status']=='FAIL'


def test_inconclusive_icnn_disposition_is_preserved(tmp_path,monkeypatch):
    from partC_rev1_src.pacnhn_v1 import final_icnn_guarantees as old
    ctx,rec=fixture(tmp_path,'ICNN')
    monkeypatch.setattr(old,'audit_model',lambda *args,**kwargs:dict(status='NUMERIC_INCONCLUSIVE',accuracy_evaluated=False))
    assert n.audit_records(ctx,[rec],deadline=ctx.end)[rec['model_sha256']]['status']=='NUMERIC_INCONCLUSIVE'


def test_runtime_timeout_is_not_model_failure_or_pass(tmp_path,monkeypatch):
    from partC_rev1_src.pacnhn_v1 import final_icnn_guarantees as old
    ctx,rec=fixture(tmp_path,'ICNN')
    def expire(*args,**kwargs):raise TimeoutError('declared cap')
    monkeypatch.setattr(old,'audit_model',expire)
    got=n.audit_records(ctx,[rec],deadline=ctx.end)[rec['model_sha256']]
    assert got['status']=='NUMERIC_INCONCLUSIVE' and got['error_type']=='TimeoutError'


def test_empty_duplicate_and_unavailable_only_are_not_success(tmp_path):
    ctx,rec=fixture(tmp_path)
    with pytest.raises(ValueError):n.audit_records(ctx,[],deadline=ctx.end)
    with pytest.raises(ValueError):n.audit_records(ctx,[rec,rec],deadline=ctx.end)
    rec['status']='UNAVAILABLE'
    with pytest.raises(ValueError):n.audit_records(ctx,[rec],deadline=ctx.end)
