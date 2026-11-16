from pathlib import Path
from types import SimpleNamespace
import importlib.util
import numpy as np
import pytest
from partC_rev1_src.phase_n8.common import write_json,read_npz,array_hash,sha
from partC_rev1_src.phase_n8.solver import save_checkpoint
from partC_rev1_src.phase_n8_publication.assignment import retain_fresh_state,epoch_deadline,downgrade
from partC_rev1_src.phase_n8_publication.physical_closure import physical_overlay,directed_closed_membership
from partC_rev1_src.phase4.aon import run_exact_aon,AONExecutionConfig


def test_recovery_deadline_cannot_borrow_final_run_time():
    current=dict(rescue_used=False,cumulative_recovery_seconds=7000)
    assert epoch_deadline(current,[dict(model_kind='POWER4_SKIP_ICNN32')],1000,6400)==(1200,True)
    assert epoch_deadline(dict(current,rescue_used=True),[dict(model_kind='NEW_POWER4_RELATIVE_NNLS')],1200,6400)==(6400,False)
    assert epoch_deadline(current,[dict(model_kind='POWER4_SKIP_ICNN32')],1000,1100)==(1100,False)
    assert epoch_deadline(current,[dict(model_kind='POWER4_SKIP_ICNN32')],1000,6400,pending_seconds=150)==(1050,True)


def test_timeout_selects_only_authenticated_same_epoch_certificate(tmp_path):
    identity={'objective':'new'};clocks={'run_start_epoch':1};flow=np.array([3.,2.])
    cert=dict(state_hash=array_hash({'flow':flow}),RG_SPTT=.01,fresh=True)
    saved=save_checkpoint(tmp_path,flow,2,identity,clocks,cert)
    result=dict(final_certificate_fresh=False,last_certified_checkpoint=saved,identity=identity,stop_reason='TIME_LIMIT')
    selected=retain_fresh_state(tmp_path,result)
    assert selected['stop_reason']=='TIME_LIMIT' and selected['final_certificate_fresh']
    assert np.array_equal(read_npz(tmp_path/'final_flow.npz')['flow'],flow)
    with pytest.raises(ValueError,match='this fixed epoch'):
        retain_fresh_state(tmp_path,dict(result,identity={'objective':'different'}))


def test_batched_whole_region_downgrades_keep_source_and_share_new_identity(tmp_path):
    reg=tmp_path/'registries';reg.mkdir();entries=[];chains={};candidates={}
    arrays={'c0':np.array([2.,3.]),'scale':np.ones(2),'eta':np.array([.2,.3])}
    for i,rid in enumerate(['a','b']):
        row=dict(region_id=rid,model_kind='POWER4_SKIP_ICNN32',portable_sha256='old'+rid,portable_path='old',
            pair_order_hash='pairs'+rid,support_hash='support'+rid,start=2*i,stop=2*i+2,arrays={'c0':arrays['c0']})
        entries.append(row);chains[rid]=['POWER4_SKIP_ICNN32','NEW_POWER4_RELATIVE_NNLS','FREEFLOW_LINEAR']
        candidates[rid]={'NEW_POWER4_RELATIVE_NNLS':dict({k:v for k,v in row.items() if k!='arrays'},
            model_kind='NEW_POWER4_RELATIVE_NNLS',portable_sha256='poly'+rid,portable_path='poly')}
    write_json(reg/'FALLBACK_CHAINS.json',dict(chains=chains,candidates=candidates,restart_epoch_cap=6,cumulative_recovery_cap_seconds=7200))
    ctx=SimpleNamespace(root=tmp_path,bind=lambda *a:None);src=SimpleNamespace(arrays=lambda *a:arrays)
    current=dict(epoch=0,discovery_epochs=0,cumulative_recovery_seconds=10,rescue_used=False,
                 permanently_downgraded_regions=[],discarded_epochs=[])
    new,portfolio,state=downgrade(ctx,src,entries,current,[{'region_id':'a'},{'region_id':'b'}],12,tmp_path/'aborted')
    assert all(e['model_kind']=='NEW_POWER4_RELATIVE_NNLS' for e in new)
    assert all(e['model_kind']=='POWER4_SKIP_ICNN32' for e in entries)
    assert state['epoch']==1 and state['cumulative_recovery_seconds']==22
    assert len(portfolio['events'])==2 and state['identity']==portfolio['identity']


def test_physical_csr_overlay_matches_exact_od_membership_without_zero_capacity():
    p=Path(__file__).resolve().parents[3]/'tests/test_phase4_preflight.py'
    spec=importlib.util.spec_from_file_location('closure_physical_fixture',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    graph=m._graph();od=m._od();mask=np.zeros(len(graph.tail),bool);mask[2]=True
    closed=physical_overlay(graph,mask)
    serve,balance,_=directed_closed_membership(closed,od,mask)
    actual=run_exact_aon(closed,od,closed.free_time,config=AONExecutionConfig(worker_count=1))
    assert np.array_equal(serve,[True,False,True])
    assert actual.reachable_demand==7 and actual.unreachable_demand==3
    assert actual.physical_flow[2]==0
    assert np.array_equal(closed.capacity,graph.capacity) and np.all(closed.capacity>0)
    net=np.bincount(closed.tail,weights=actual.physical_flow,minlength=8)-np.bincount(closed.head,weights=actual.physical_flow,minlength=8)
    assert np.array_equal(net,balance)
