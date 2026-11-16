"""Persistent workers keep shared authority plus only the current region inputs."""
import json
from types import SimpleNamespace
import pytest
from partC_rev1_src.pacnhn_recovery24 import labels


def fixture_worker(tmp_path,monkeypatch):
    root=tmp_path/'run';(root/'holdout').mkdir(parents=True)
    opened=root/'holdout/TEST_OPENED.json'
    opened.write_text(json.dumps(dict(epoch=20,selection_freeze_path=str(root/'freeze.json'),selection_freeze_sha256='f'*64)))
    ctx=SimpleNamespace(root=root,bindings={'config':{'sha256':'config-digest'}},actions=[])
    def bind(path,role,expected=None):
        ctx.bindings[str(path)]={'role':role,'sha256':expected or 'measured'}
    ctx.bind=bind;ctx.require_action=ctx.actions.append
    created=[]
    def factory(*args,**kwargs):created.append((args,kwargs));return ctx
    verified=[]
    def verify(context,path,digest):
        verified.append((path,digest));context.bind(path,'FREEZE',digest)
        return {'created_epoch':10},{}
    monkeypatch.setattr(labels,'RecoveryContext',factory)
    monkeypatch.setattr(labels,'require_selection_freeze',verify)
    monkeypatch.setattr(labels,'_WORKER_CONTEXTS',{})
    return ctx,created,verified,opened


def test_many_regions_keep_authority_without_accumulating_prior_region_inputs(tmp_path,monkeypatch):
    ctx,created,verified,_=fixture_worker(tmp_path,monkeypatch)
    for n in range(100):
        current=labels._holdout_context('fixed-config')
        assert current is ctx and 'config' in ctx.bindings
        assert all(not key.startswith('region_') for key in ctx.bindings)
        assert len(ctx.bindings)==3
        ctx.bind('region_'+str(n),'CURRENT_REGION',str(n))
    assert len(created)==1 and len(verified)==100
    assert ctx.actions==['holdout']*100
    assert len(ctx._holdout_shared_bindings)==3


def test_each_region_revalidates_freeze_after_reset(tmp_path,monkeypatch):
    ctx,_,_,_=fixture_worker(tmp_path,monkeypatch)
    labels._holdout_context('fixed-config');ctx.bind('region_1','CURRENT_REGION')
    def fail(*args):raise ValueError('Frozen portfolio changed')
    monkeypatch.setattr(labels,'require_selection_freeze',fail)
    with pytest.raises(ValueError,match='Frozen portfolio changed'):
        labels._holdout_context('fixed-config')
    assert 'region_1' not in ctx.bindings


def test_invalid_test_chronology_never_caches_shared_authority(tmp_path,monkeypatch):
    ctx,_,_,opened=fixture_worker(tmp_path,monkeypatch)
    value=json.loads(opened.read_text());value['epoch']=5;opened.write_text(json.dumps(value))
    with pytest.raises(PermissionError,match='precedes selection'):
        labels._holdout_context('fixed-config')
    assert not hasattr(ctx,'_holdout_shared_bindings')
