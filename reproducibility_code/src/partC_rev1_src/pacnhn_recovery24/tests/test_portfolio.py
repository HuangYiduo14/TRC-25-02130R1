import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from partC_rev1_src.pacnhn_recovery24.portfolio import (
    FAMILIES, SUITES, KINDS, FrozenComposite, choose_region, make_portfolio,
    model_eligible, output_path, validate_graph_arrays, build_portfolios,
)
from partC_rev1_src.pacnhn_v1.control import sha

H = 'a' * 64
V = 'b' * 64
M = 'c' * 64


def record(family='ICNN', **changes):
    d = dict(suite_id='s03', region_index=7, family=family, status='AVAILABLE',
             model_path='/synthetic/model.npz', model_sha256=M,
             artifact_hashes={'/synthetic/model.npz': M}, model_kind=KINDS.get(family),
             pair_keys_identity=H, graph_sha256=H, train_wmape_percent=.5,
             valid_wmape_percent=2., current_train_dataset_identity=H,
             current_valid_dataset_identity=V, audit_train_dataset_identity=H,
             audit_valid_dataset_identity=V, audit_model_sha256=M,
             property_status='PASS', property_model_sha256=M)
    d.update(changes)
    return d


def support():
    return dict(status='PASS', suites=[dict(suite_id=s, graph_sha256=H,
        graph_identity=H, OD_identity=V, active_regions=2, active_pairs=2,
        region_records=[dict(region_index=i, pairs=1, sha256=H) for i in (7, 99)]) for s in SUITES])


@pytest.mark.parametrize('family', ['ICNN', 'XGBOOST'])
def test_primary_strict_qualification(family):
    assert choose_region(family, {family: record(family)})[0] == family
    assert choose_region(family, {family: record(family, train_wmape_percent=1.)})[0] == 'FFTT'


@pytest.mark.parametrize('field,value', [
    ('property_status', 'FAIL'), ('property_model_sha256', H),
    ('audit_model_sha256', H), ('audit_train_dataset_identity', V),
    ('audit_valid_dataset_identity', H), ('train_wmape_percent', float('nan')),
    ('valid_wmape_percent', float('inf')), ('status', 'UNAVAILABLE'),
    ('artifact_hashes', {}), ('model_kind', 'SOME_OTHER_MODEL'),
])
def test_stale_or_invalid_primary_falls_back(field, value):
    assert not model_eligible(record(**{field: value}), require_train=True)
    assert choose_region('ICNN', {'ICNN': record(**{field: value})})[0] == 'FFTT'


def test_power_fftt_comparison_same_valid_and_ties():
    p = record('POWER4', valid_wmape_percent=2., train_wmape_percent=10.)
    f = record('FFTT', valid_wmape_percent=3.)
    assert choose_region('ICNN', {'POWER4': p, 'FFTT': f})[0] == 'POWER4'
    assert choose_region('POWER4', {'POWER4': p, 'FFTT': f})[0] == 'POWER4'
    f['valid_wmape_percent'] = 2.
    assert choose_region('POWER4', {'POWER4': p, 'FFTT': f})[0] == 'POWER4'
    assert choose_region('ICNN', {'POWER4': p, 'FFTT': f})[0] == 'FFTT'
    f['valid_wmape_percent'] = 3.; f['current_valid_dataset_identity'] = H
    assert choose_region('ICNN', {'POWER4': p, 'FFTT': f})[0] == 'FFTT'
    assert choose_region('POWER4', {'POWER4': p})[0] == 'POWER4'
    p['property_status'] = 'FAIL'
    assert choose_region('POWER4', {'POWER4': p})[0] == 'FFTT'


def test_test_metrics_never_select_components():
    rec = record('ICNN', train_wmape_percent=2., TEST_wmape_percent=0.)
    assert choose_region('ICNN', {'ICNN': rec})[0] == 'FFTT'
    rec.update(train_wmape_percent=.1, TEST_wmape_percent=1000.)
    assert choose_region('ICNN', {'ICNN': rec})[0] == 'ICNN'


def test_complete_all_fftt_portfolios_preserve_sparse_ids():
    portfolios = make_portfolio(support(), [])
    assert len(portfolios) == 20
    assert all(p['component_counts'] == {'FFTT': 2} for p in portfolios)
    assert all([r['region_index'] for r in p['regions']] == [7, 99] for p in portfolios)
    assert all(p['solver'] == ('MSA' if p['requested_family'] == 'XGBOOST' else 'FW') for p in portfolios)
    assert all(p['fixed_across_scenarios'] for p in portfolios)


@pytest.mark.parametrize('mutator', [
    lambda s: s['suites'].pop(),
    lambda s: s['suites'][0]['region_records'].append(s['suites'][0]['region_records'][0]),
    lambda s: s['suites'][0].update(active_pairs=4),
])
def test_invalid_support_rejected(mutator):
    s = support(); mutator(s)
    with pytest.raises(ValueError): make_portfolio(s, [])


def test_duplicate_and_wrong_region_model_rejected():
    r = record()
    with pytest.raises(ValueError): make_portfolio(support(), [r, r])
    with pytest.raises(ValueError): make_portfolio(support(), [record(region_index=8)])
    with pytest.raises(ValueError): make_portfolio(support(), [record(graph_sha256=V)])


class ToyPotential:
    has_potential = True
    def __init__(self, c0, eta): self.c0=np.asarray(c0); self.eta=np.asarray(eta)
    def cost(self, u): return self.c0+self.eta*np.asarray(u)**4
    def value_cost(self, u):
        u=np.asarray(u); return float(u@(self.c0+self.eta*u**4/5)), self.cost(u)
    def hess_vec(self, u, d): return 4*self.eta*np.asarray(u)**3*d


class ToyCost(ToyPotential):
    has_potential=False
    def value_cost(self, u): raise AssertionError('Cost-only evaluator must not be called as potential')


def test_cost_charge_ownership_and_analytic_gradient():
    base=np.array([2.,3.,4.,0.]);k4=np.array([.1,0.,0.,0.]);u=np.array([2.,1.,3.,0.])
    model=ToyPotential([3.],[.5])
    obj=FrozenComposite(base,k4,[1,2],[(np.array([1]),model)])
    value,cost=obj.value_cost(u)
    assert value == pytest.approx(2*2+.1*2**5/5 +3+.5/5 +4*3)
    np.testing.assert_allclose(cost,[3.6,3.5,4.,0.])
    for i in range(3):
        e=np.eye(4)[i]*1e-5
        assert (obj.value_cost(u+e)[0]-obj.value_cost(u-e)[0])/2e-5 == pytest.approx(cost[i],rel=1e-7)
    np.testing.assert_array_equal(base,[2,3,4,0]);np.testing.assert_array_equal(k4,[.1,0,0,0])


def test_mixed_xgb_power_fftt_is_always_cost_only():
    obj=FrozenComposite([2.,3.,4.],[0.,0.,0.],[0,1,2],
        [(np.array([0]),ToyCost([2.],[.1])),(np.array([1]),ToyPotential([3.],[.2]))],cost_only=True)
    np.testing.assert_allclose(obj.cost(np.ones(3)),[2.1,3.2,4.])
    assert not obj.has_potential
    with pytest.raises(TypeError): obj.value_cost(np.ones(3))
    with pytest.raises(TypeError): obj.hess_vec(np.ones(3),np.ones(3))
    with pytest.raises(ValueError): FrozenComposite([2.],[0.],[0],[(np.array([0]),ToyCost([2.],[1.]))])
    assert not FrozenComposite([1.],[0.],[0],[],cost_only=True).has_potential


@pytest.mark.parametrize('indices,c0', [([0,0],[1.,1.]),([2],[1.]),([0],[2.])])
def test_bad_component_coordinates_rejected(indices,c0):
    with pytest.raises(ValueError): FrozenComposite([1.,1.],[0.,0.],[0,1],[(np.array(indices),ToyPotential(c0,np.zeros(len(c0))))])


def test_runtime_invalid_cost_never_falls_back():
    m=ToyPotential([1.],[np.inf]);obj=FrozenComposite([1.],[0.],[0],[(np.array([0]),m)])
    with pytest.raises(ValueError): obj.cost(np.ones(1))


def test_paths_cannot_write_old_root_or_escape(tmp_path):
    old=tmp_path/'old';old.mkdir();new=tmp_path/'new';new.mkdir()
    ctx=SimpleNamespace(root=new,source_root=old)
    assert output_path(ctx,'a/b.json') == new/'a/b.json'
    with pytest.raises(ValueError): output_path(ctx,'../old/file.json')
    with pytest.raises(ValueError): output_path(SimpleNamespace(root=old/'new',source_root=old),'a')


def graph_fixture():
    n=87584
    a=dict(backbone_count=np.array(87582),tail=np.zeros(n,dtype=np.int32),head=np.ones(n,dtype=np.int32),
           edge_ids=np.arange(n),base=np.ones(n),k4=np.zeros(n),closed_mask=np.zeros(n,bool),
           pair_keys=np.array([1,3],np.uint64),pair_region=np.array([7,99]),c0=np.ones(2))
    common={k:v.copy() for k,v in a.items()};report=dict(backbone_count=87582,OD_identity='od',graph_identity='g',active_pairs=2)
    return a,common,report


@pytest.mark.parametrize('field',['tail','head','edge_ids','base','k4','closed_mask'])
def test_each_frozen_backbone_field_checked(field):
    a,c,r=graph_fixture();validate_graph_arrays(a,c,r,'od','g')
    a[field][0]=True if field=='closed_mask' else a[field][0]+1
    with pytest.raises(ValueError,match='backbone'):validate_graph_arrays(a,c,r,'od','g')


def test_graph_od_pair_and_mask_boundaries():
    a,c,r=graph_fixture()
    with pytest.raises(ValueError):validate_graph_arrays(a,c,r,'wrong','g')
    a['closed_mask'][-1]=True
    with pytest.raises(ValueError):validate_graph_arrays(a,c,r,'od','g')
    a['closed_mask'][-1]=False;a['pair_keys'][1]=1
    with pytest.raises(ValueError):validate_graph_arrays(a,c,r,'od','g')


def test_builder_requires_bound_records_and_never_changes_source(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    sp=old/'SUPPORT_FREEZE.json';sp.write_text(json.dumps(support()))
    rp=new/'records.json';rp.write_text('[]');before=sp.read_bytes()
    class C:
        root=new;source_root=old
        c={'phase_id':'PACNHN_RECOVERY24','run_id':'toy','source_bindings':{str(sp):sha(sp)}}
        def require_action(self, action): assert action=='portfolio'
        def guard(self,deadline=None): pass
        def bind(self,path,role,expected=None):
            h=sha(path)
            if expected is not None: assert h==expected
            return h
        def flush_bindings(self): pass
    result=build_portfolios(C(),[],records_binding={'path':str(rp),'sha256':sha(rp)})
    assert len(result['portfolios'])==20 and sp.read_bytes()==before
    assert sorted(p.name for p in old.iterdir())==['SUPPORT_FREEZE.json']
    with pytest.raises(FileExistsError):build_portfolios(C(),[],records_binding={'path':str(rp),'sha256':sha(rp)})
