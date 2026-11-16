"""Certified-state observers must reject adjacent or mutated numerical states."""
from types import SimpleNamespace
import numpy as np
import pytest
from partC_rev1_src.pacnhn_v1.control import array_hash,save_npz,write_json
from partC_rev1_src.pacnhn_v1.stability import capture,summarize


def fixture(tmp_path):
    ctx=SimpleNamespace(root=tmp_path,c={'topology':{'expected_backbone_count':2}})
    source=tmp_path/'references/i95';earlier=np.array([2.,4.,8.])
    certificate=dict(fresh=True,state_hash=array_hash({'flow':earlier}),RG_SPTT=.002)
    save_npz(source/'certified_flow.npz',flow=earlier)
    write_json(source/'CERTIFICATE.json',certificate)
    write_json(source/'HISTORY.json',[dict(state_hash=certificate['state_hash'],iteration=3,elapsed_seconds=10.)])
    return ctx,source,earlier,certificate


def test_observer_rejects_adjacent_atomic_states_and_missing_history(tmp_path):
    ctx,source,earlier,certificate=fixture(tmp_path)
    save_npz(source/'certified_flow.npz',flow=earlier+1)
    assert capture(ctx,'i95')['status']=='ADJACENT_ATOMIC_UPDATES_RETRY'
    save_npz(source/'certified_flow.npz',flow=earlier)
    write_json(source/'HISTORY.json',[])
    assert capture(ctx,'i95')['status']=='HISTORY_WRITE_PENDING_RETRY'
    assert not list((tmp_path/'audits').glob('**/state_*.npz'))


def test_stability_is_final_denominator_change_not_convergence_claim(tmp_path):
    ctx,source,earlier,certificate=fixture(tmp_path)
    receipt=capture(ctx,'i95',minimum_interval_seconds=0)
    assert receipt['status']=='CERTIFIED_SNAPSHOT'
    assert capture(ctx,'i95',minimum_interval_seconds=0)['status']=='ALREADY_CAPTURED'
    final=np.array([3.,3.,10.]);save_npz(source/'final_flow.npz',flow=final)
    write_json(source/'RESULT.json',dict(final_certificate_fresh=True,
        certificate=dict(state_hash=array_hash({'flow':final}),RG_SPTT=.001),
        reported_state_iteration=8,elapsed_seconds=30.,status='TIME_LIMIT'))
    report=summarize(ctx,'i95');b,p=report['comparisons']
    assert report['reference_status']=='TIME_LIMIT'
    assert b['change_percent']==pytest.approx(100*2/6)
    assert p['change_percent']==pytest.approx(100*4/16)
    assert b['comparison_seconds']==20.
    assert 'not a bound' in report['interpretation']
    save_npz(receipt['flow_path'],flow=earlier+1)
    with pytest.raises(ValueError,match='snapshot changed'):summarize(ctx,'i95')
