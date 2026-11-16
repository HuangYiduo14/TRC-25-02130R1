"""Clock/action regressions only: no national graph, model or assignment runs."""
import copy
from types import SimpleNamespace

import pytest

from partC_rev1_src.pacnhn_v1 import assignments as a
from partC_rev1_src.pacnhn_v1 import inference_pilot as ip
from partC_rev1_src.pacnhn_v1.control import read_json,write_json,sha


def authority():
    return dict(start_epoch=100000.,hard_deadline_epoch=186400.,science_deadline_epoch=172000.,
        stage3_max_seconds=43200,publication_max_seconds=14400,approval_sha256='approval-a',clock_sha256='clock-a')


def context(tmp_path,auth):
    actions=[]
    def admit(action):
        assert action in ('assignments','inference')
        actions.append(action)
        return copy.deepcopy(auth)
    config=dict(phase_id='PACNHN_V1',run_id='synthetic',original_start_epoch=100.,hard_deadline_epoch=90000.,
        maximum_cumulative_stage_deadlines=[10000.,50000.,80000.,90000.],suite_ids=['s03'],
        assignment=dict(per_query_seconds=2700,RG_SPTT=.001),training=dict(additional_suite_order=[]),
        input_versions={'synthetic':'metadata_only'},resources={'synthetic':True})
    path=tmp_path/'config.json';write_json(path,config)
    return SimpleNamespace(root=tmp_path,path=path,c=config,require_action=admit,actions=actions,
        guard=lambda **kwargs:None,bindings={},flush_bindings=lambda:None)


def test_legacy_clock_keeps_original_two_field_schema_and_deadline(tmp_path,monkeypatch):
    ctx=context(tmp_path,None);monkeypatch.setattr(a.time,'time',lambda:20000.)
    first=a.stage_clock(ctx);before=(tmp_path/'STAGE3_CLOCK.json').read_bytes()
    assert first==dict(original_stage_start_epoch=20000.,deadline_epoch=63200.)
    monkeypatch.setattr(a.time,'time',lambda:21000.)
    assert a.stage_clock(ctx)==first and (tmp_path/'STAGE3_CLOCK.json').read_bytes()==before


def test_continuation_first_inference_starts_shared_twelve_hour_clock(tmp_path,monkeypatch):
    ctx=context(tmp_path,authority());monkeypatch.setattr(a.time,'time',lambda:100300.)
    first=a.stage_clock(ctx,action='inference');before=sha(tmp_path/'STAGE3_CLOCK.json')
    assert first['original_stage_start_epoch']==100300. and first['deadline_epoch']==143500.
    assert first['continuation_approval_sha256']=='approval-a' and first['continuation_clock_sha256']=='clock-a'
    assert first['deadline_epoch']>ctx.c['hard_deadline_epoch']
    monkeypatch.setattr(a.time,'time',lambda:101000.)
    assert a.stage_clock(ctx,action='assignments')==first
    assert sha(tmp_path/'STAGE3_CLOCK.json')==before and ctx.actions==['inference','assignments']


def test_late_start_clamps_to_publication_reserve(tmp_path,monkeypatch):
    ctx=context(tmp_path,authority());monkeypatch.setattr(a.time,'time',lambda:160000.)
    clock=a.stage_clock(ctx)
    assert clock['deadline_epoch']==172000.==authority()['hard_deadline_epoch']-14400


@pytest.mark.parametrize('field,value',[('deadline_epoch',180000.),('continuation_clock_sha256','other'),
    ('continuation_approval_sha256','other'),('stage3_max_seconds',50000)])
def test_reopen_rejects_clock_or_authority_mutation(tmp_path,monkeypatch,field,value):
    ctx=context(tmp_path,authority());monkeypatch.setattr(a.time,'time',lambda:100300.)
    clock=a.stage_clock(ctx);clock[field]=value;write_json(tmp_path/'STAGE3_CLOCK.json',clock)
    with pytest.raises(ValueError,match='clock/authority changed'):a.stage_clock(ctx)


def test_new_continuation_cannot_reset_a_historical_stage3_clock(tmp_path,monkeypatch):
    ctx=context(tmp_path,authority());monkeypatch.setattr(a.time,'time',lambda:110000.)
    historical=dict(original_stage_start_epoch=20000.,deadline_epoch=63200.)
    write_json(tmp_path/'STAGE3_CLOCK.json',historical);before=sha(tmp_path/'STAGE3_CLOCK.json')
    assert a.stage_clock(ctx)==historical and sha(tmp_path/'STAGE3_CLOCK.json')==before


def test_continuation_clock_cannot_reopen_without_authority(tmp_path,monkeypatch):
    ctx=context(tmp_path,authority());monkeypatch.setattr(a.time,'time',lambda:100300.)
    a.stage_clock(ctx);ctx.require_action=lambda action:None
    with pytest.raises(ValueError,match='requires its authenticated authority'):a.stage_clock(ctx)


def test_expired_science_budget_does_not_create_clock(tmp_path,monkeypatch):
    ctx=context(tmp_path,authority());monkeypatch.setattr(a.time,'time',lambda:172000.)
    with pytest.raises(TimeoutError,match='publication reserve'):a.stage_clock(ctx)
    assert not (tmp_path/'STAGE3_CLOCK.json').exists()


def test_explicit_action_rejection_precedes_clock_or_output(tmp_path,monkeypatch):
    ctx=context(tmp_path,authority())
    def deny(action):raise ValueError('Action not approved')
    ctx.require_action=deny
    with pytest.raises(ValueError,match='not approved'):a.run(ctx)
    with pytest.raises(ValueError,match='not approved'):ip.run(ctx)
    assert not (tmp_path/'STAGE3_CLOCK.json').exists()
    with pytest.raises(ValueError,match='explicit assignment or inference'):a.stage_clock(ctx,action='training')


def test_lightweight_context_fallback_still_authenticates_continuation(tmp_path,monkeypatch):
    from partC_rev1_src.pacnhn_v1 import continuation
    ctx=context(tmp_path,authority());del ctx.require_action;ctx.c['continuation']={'synthetic':'must authenticate'}
    seen=[]
    def validate(config,action=None):seen.append((config,action));return authority()
    monkeypatch.setattr(continuation,'validate_continuation',validate)
    assert a.action_deadline(ctx,'inference')==172000.
    assert seen==[(ctx.c,'inference')]


@pytest.mark.parametrize('module,action',[(a,'assignments'),(ip,'inference')])
def test_clock_starts_before_technical_gate_or_registry_loading(tmp_path,monkeypatch,module,action):
    from partC_rev1_src.pacnhn_v1 import gates
    ctx=context(tmp_path,authority());monkeypatch.setattr(a.time,'time',lambda:100300.)
    def gate(context,component):
        assert ctx.actions==[action] and read_json(tmp_path/'STAGE3_CLOCK.json')['original_stage_start_epoch']==100300.
        raise ValueError('Synthetic stop before registry/numerics')
    monkeypatch.setattr(gates,'require_component',gate)
    monkeypatch.setattr(a,'validate_gates',lambda *args,**kwargs:pytest.fail('Registry read before technical gate'))
    with pytest.raises(ValueError,match='Synthetic stop'):module.run(ctx)


@pytest.mark.parametrize('now,expected',[(101000.,101180.),(171900.,172000.)])
def test_inference_direct_entry_uses_effective_bound_and_keeps_180_second_cap(tmp_path,monkeypatch,now,expected):
    from partC_rev1_src.pacnhn_v1 import gates
    ctx=context(tmp_path,authority());monkeypatch.setattr(ip.time,'time',lambda:now)
    write_json(tmp_path/'SUPPORT_FREEZE.json',{})
    write_json(tmp_path/'references/open/RESULT.json',{})
    monkeypatch.setattr(gates,'component_snapshot',lambda *args:{'synthetic':'only'})
    registry=dict(coverage=[dict(suite_id='s03',family='ICNN',complete_family=False)],models=[])
    result=ip._one(ctx,registry,'frozen',dict(deadline_epoch=180000.),'s03','ICNN','open',repetitions=3,seconds=180.)
    assert result['status']=='UNAVAILABLE_INCOMPLETE_PURE_FAMILY' and result['deadline_epoch']==expected
    intent=read_json(tmp_path/'inference_pilots/s03/ICNN/open/INTENT.json')
    assert intent['cap_seconds']==180. and intent['complete_pilot_start_epoch']==now
    assert ctx.actions==['inference']


@pytest.mark.parametrize('now,expected',[(101000.,103700.),(171900.,172000.)])
def test_assignment_direct_entry_uses_effective_bound_and_preserves_45_min_cap(tmp_path,monkeypatch,now,expected):
    from partC_rev1_src.pacnhn_v1 import gates,topology
    ctx=context(tmp_path,authority());monkeypatch.setattr(a.time,'time',lambda:now)
    write_json(tmp_path/'training/REGISTRY_FREEZE.json',{})
    monkeypatch.setattr(gates,'component_snapshot',lambda *args:{'synthetic':'only'})
    def unavailable(*args):raise MemoryError('Synthetic guard before graph access')
    monkeypatch.setattr(topology,'load_common',unavailable)
    result=a.run_case(ctx,a.matrix()[0],{},stage_deadline=180000.)
    assert result['status']=='UNAVAILABLE_CURRENT_RAM' and result['deadline_epoch']==expected
    assert ctx.c['assignment']['per_query_seconds']==2700 and ctx.actions==['assignments']


def test_direct_query_entries_cannot_bypass_action_admission(tmp_path):
    ctx=context(tmp_path,authority())
    def deny(action):raise ValueError('Action not approved')
    ctx.require_action=deny
    with pytest.raises(ValueError,match='not approved'):
        a.run_case(ctx,a.matrix()[0],{},stage_deadline=180000.)
    with pytest.raises(ValueError,match='not approved'):
        ip._one(ctx,{},'frozen',dict(deadline_epoch=180000.),'s03','FFTT','open',repetitions=3,seconds=180.)
    assert not (tmp_path/'assignments').exists() and not (tmp_path/'inference_pilots').exists()


def test_registry_load_has_deadline_guards_before_and_after(tmp_path,monkeypatch):
    from partC_rev1_src.pacnhn_v1 import registry_projection as rp
    ctx=context(tmp_path,None);seen=[]
    support=tmp_path/'SUPPORT_FREEZE.json';write_json(support,dict(status='PASS'))
    write_json(tmp_path/'COMPLETE_WITNESS_AUDIT.json',dict(status='PASS',support_freeze_sha256=sha(support)))
    registry=tmp_path/'training/REGISTRY_FREEZE.json';write_json(registry,{'synthetic':True})
    write_json(tmp_path/'training/RESULT.json',dict(status='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE',registry_sha256=sha(registry)))
    def read(path,expected_sha256,*,guard):
        assert path==registry and expected_sha256==sha(registry)
        seen.append('registry')
        return {'synthetic':True}
    def guard(**kwargs):
        assert kwargs['deadline']==12345.
        seen.append('guard')
        if 'registry' in seen:raise MemoryError('Synthetic post-registry resource limit')
    ctx.guard=guard;monkeypatch.setattr(rp,'read_stage3_registry',read)
    with pytest.raises(MemoryError,match='post-registry'):a.validate_gates(ctx,deadline=12345.)
    assert seen==['guard','guard','registry','guard']
