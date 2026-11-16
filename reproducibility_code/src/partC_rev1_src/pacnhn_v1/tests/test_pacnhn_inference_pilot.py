"""Synthetic inference measurement checks, with no national or learned models."""
from types import SimpleNamespace
import time

import numpy as np
import pytest
from partC_rev1_src.pacnhn_v1.control import write_json,read_json

from partC_rev1_src.pacnhn_v1.inference_pilot import (
    measure_calls,projected_baseline,pure_family_available,
)
from partC_rev1_src.pacnhn_v1.oracle import Network,Demand,conservation


def fixture():
    graph=Network(np.array([1,2,0,3]),np.array([2,3,1,4]),np.array([True,False,False,False,True]),
        np.array(['B1','R1','PORT_O','PORT_D']),np.array(['O','A','B','C','D']))
    ac=Network(graph.tail.copy(),graph.head.copy(),graph.blocked.copy(),
        np.array(['B1','PAIR1','PORT_O','PORT_D']),graph.node_ids.copy())
    pa=dict(edge_role=np.array([0,1,2,2]),base=np.array([1.,2.,0.,0.]),k4=np.zeros(4),closed_mask=np.zeros(4,bool))
    aa=dict(backbone_count=np.array(1),port_count=np.array(2),pair_keys=np.array([13],np.uint64),
        base=pa['base'].copy(),k4=pa['k4'].copy(),closed_mask=pa['closed_mask'].copy(),
        baseline_open=np.array([2.]),baseline_i95=np.array([2.]))
    return graph,pa,ac,aa


def test_exact_trace_projection_conserves_served_od_and_preserves_input():
    graph,pa,ac,aa=fixture();original=np.full(4,2.)
    flow=projected_baseline(graph,pa,ac,aa,original,'open',expected_backbone_count=1)
    od=Demand(np.array([0]),np.array([0,1]),np.array([4]),np.array([2.]),np.array([True]))
    evidence=conservation(ac,flow,od.balance(5),od.total)
    assert evidence['max_absolute_node_residual']==0.
    assert not flow.flags.writeable and original.flags.writeable
    np.testing.assert_array_equal(flow,original)
    aa['baseline_open'][0]=3.
    bad=projected_baseline(graph,pa,ac,aa,original,'open',expected_backbone_count=1)
    with pytest.raises(ArithmeticError):conservation(ac,bad,od.balance(5),od.total)


def test_projection_rejects_port_relabel_and_closed_flow():
    graph,pa,ac,aa=fixture()
    ac.edge_ids[-1]='OTHER_D'
    with pytest.raises(ValueError,match='Port stable-ID'):
        projected_baseline(graph,pa,ac,aa,np.full(4,2.),'open',expected_backbone_count=1)
    graph,pa,ac,aa=fixture();pa['closed_mask'][0]=True;aa['closed_mask'][0]=True
    with pytest.raises(ValueError,match='Masked'):
        projected_baseline(graph,pa,ac,aa,np.full(4,2.),'i95',expected_backbone_count=1)
    aa['closed_mask'][1]=True
    with pytest.raises(ValueError,match='regional/port'):
        projected_baseline(graph,pa,ac,aa,np.zeros(4),'i95',expected_backbone_count=1)


def test_potential_and_cost_only_use_correct_full_field_interfaces():
    flow=np.array([2.,3.,0.]);flow.setflags(write=False)
    class Potential:
        has_potential=True
        def value_cost(self,x):return float(x.sum()),np.ones_like(x)
        def cost(self,x):raise AssertionError('Potential pilot must measure value_cost')
    class CostOnly:
        has_potential=False
        def cost(self,x):return np.ones_like(x)
        def value_cost(self,x):raise AssertionError('XGBoost has no potential')
    for objective,kind in ((Potential(),'VALUE_COST'),(CostOnly(),'COST_ONLY')):
        result=measure_calls(objective,flow,repetitions=3,deadline=time.time()+30,guard=lambda **k:None)
        assert result['status']=='PASS' and result['repetitions']==3 and result['call_kind']==kind
        assert result['seconds']==pytest.approx(sum(r['seconds'] for r in result['calls']))
        assert bool(result['cost_identity']) and all(r['cost_min']==1. for r in result['calls'])


def test_mutation_negative_cost_and_changing_prediction_are_hard_failures():
    class Bad:
        has_potential=False
        def __init__(self,kind):self.kind=kind;self.n=0
        def cost(self,x):
            self.n+=1
            if self.kind=='mutate':x[0]+=1
            return np.full_like(x,-1. if self.kind=='negative' else self.n)
    for kind,match in [('mutate','mutated'),('negative','Invalid whole-field'),('changing','changed predictions')]:
        with pytest.raises(ValueError,match=match):
            measure_calls(Bad(kind),np.ones(2),repetitions=3,deadline=time.time()+30,guard=lambda **k:None)


def test_timeout_retains_only_completed_calls_and_does_not_claim_pass():
    class Objective:
        has_potential=False
        def cost(self,x):return np.ones_like(x)
    checks=[]
    def guard(**kwargs):
        checks.append(1)
        if len(checks)==2:raise TimeoutError('Synthetic after-call deadline')
    result=measure_calls(Objective(),np.ones(2),repetitions=3,deadline=time.time()+30,guard=guard)
    assert result['status']=='TIME_LIMIT' and result['repetitions']==1 and result['seconds']>0
    assert result['requested_repetitions']==3 and result['error']['type']=='TimeoutError'


def test_pure_family_requires_exact_coverage_and_available_models():
    registry=dict(coverage=[dict(suite_id='s03',family='ICNN',complete_family=True)],
        models=[dict(suite_id='s03',family='ICNN',region_index=0,status='AVAILABLE')])
    assert pure_family_available(registry,'s03','ICNN')
    registry['models'][0]['status']='MISSING'
    assert not pure_family_available(registry,'s03','ICNN')
    registry['models'].append(registry['models'][0].copy())
    with pytest.raises(ValueError,match='duplicate'):pure_family_available(registry,'s03','ICNN')


def test_intent_precedes_model_load_and_receipt_has_whole_field_scope(tmp_path,monkeypatch):
    import partC_rev1_src.pacnhn_v1.inference_pilot as module
    import partC_rev1_src.pacnhn_v1.assignments as assignments
    import partC_rev1_src.pacnhn_v1.gates as gates
    config=tmp_path/'config.json';write_json(config,{'synthetic':True})
    write_json(tmp_path/'SUPPORT_FREEZE.json',{'synthetic':True})
    write_json(tmp_path/'references/open/RESULT.json',{'synthetic':True})
    ctx=SimpleNamespace(root=tmp_path,path=config,bindings={},flush_bindings=lambda:None,
        guard=lambda **kwargs:None,c=dict(phase_id='PACNHN_V1',run_id='synthetic',hard_deadline_epoch=time.time()+300,
            input_versions={'synthetic':'one_edge'},resources={'synthetic':True}))
    registry=dict(coverage=[dict(suite_id='s03',family='FFTT',complete_family=True)],
        models=[dict(suite_id='s03',family='FFTT',region_index=0,status='AVAILABLE')])
    class Objective:
        has_potential=True
        def value_cost(self,x):return float(x.sum()),np.ones_like(x)
    def load(*args,**kwargs):
        intent=read_json(tmp_path/'inference_pilots/s03/FFTT/open/INTENT.json')
        assert intent['task_list'] and intent['acceptance_tests'] and kwargs['deadline']==intent['deadline_epoch']
        return None,None,{},Objective()
    monkeypatch.setattr(assignments,'load_ac',load)
    monkeypatch.setattr(gates,'component_snapshot',lambda *a:{'synthetic':'code_hash'})
    flow=np.array([1.,2.]);flow.setflags(write=False)
    monkeypatch.setattr(module,'_load_feasible_baseline',lambda *a:(flow,dict(exact_served_OD_membership=True)))
    result=module._one(ctx,registry,'frozen_registry',dict(deadline_epoch=time.time()+250),'s03','FFTT','open',
        repetitions=3,seconds=180.)
    assert result['status']=='PASS'
    receipt=read_json(result['inference_receipt'])
    assert receipt['schema']=='PACNHN_INFERENCE_WALL_V1' and receipt['full_model']
    assert receipt['scope']=='WHOLE_AC_COST_FIELD' and receipt['includes_fixed_B']
    assert receipt['measurement']=='sequential_calls' and receipt['repetitions']==3
    assert receipt['load_seconds']>=0 and receipt['complete_pilot_seconds']>=receipt['seconds']
