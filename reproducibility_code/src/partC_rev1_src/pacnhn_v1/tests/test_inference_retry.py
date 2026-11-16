"""Metadata-only retry/lineage tests. No real models, graphs, labels or TEST."""
import copy
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from partC_rev1_src.pacnhn_v1 import inference_pilot as m
from partC_rev1_src.pacnhn_v1 import assignments as a
from partC_rev1_src.pacnhn_v1 import gates
from partC_rev1_src.pacnhn_v1.control import read_json, write_json, sha


@pytest.fixture
def env(tmp_path, monkeypatch):
    config=tmp_path/'config.json';write_json(config,dict(synthetic=True))
    write_json(tmp_path/'SUPPORT_FREEZE.json',dict(status='PASS',synthetic=True))
    write_json(tmp_path/'training/REGISTRY_FREEZE.json',dict(synthetic='immutable model registry'))
    registry_sha=sha(tmp_path/'training/REGISTRY_FREEZE.json')
    clock=dict(original_stage_start_epoch=time.time()-60,deadline_epoch=time.time()+600)
    write_json(tmp_path/'STAGE3_CLOCK.json',clock)
    for scenario in m.SCENARIOS:write_json(tmp_path/'references'/scenario/'RESULT.json',dict(scenario=scenario))
    sources={'synthetic_inference.py':'fixed-source-sha'}
    preflight=dict(status='PASS',config_sha256=sha(config),component_snapshots={'inference':sources})
    write_json(tmp_path/'validation/PREFLIGHT_LATEST.json',preflight)
    calls=[]
    ctx=SimpleNamespace(root=tmp_path,path=config,c=dict(phase_id='PACNHN_V1',run_id='synthetic',suite_ids=['s03'],
        input_versions={'synthetic':'metadata'},resources={},hard_deadline_epoch=clock['deadline_epoch']),
        guard=lambda **kw:None,bindings={},flush_bindings=lambda:None)
    monkeypatch.setattr(a,'stage_clock',lambda ctx,action:read_json(tmp_path/'STAGE3_CLOCK.json'))
    monkeypatch.setattr(a,'validate_gates',lambda *args,**kw:{'synthetic':'projected registry interface'})
    monkeypatch.setattr(gates,'require_component',lambda *args,**kw:preflight)
    monkeypatch.setattr(gates,'component_snapshot',lambda *args:sources)

    def emit(target,family,scenario,status,*,retry=None,repetitions=3):
        target.mkdir(parents=True,exist_ok=False)
        intent=dict(schema='PACNHN_INFERENCE_INTENT_V1',suite_id='s03',family=family,scenario=scenario,
            input_versions=dict(registry_sha256=registry_sha,support_sha256=sha(tmp_path/'SUPPORT_FREEZE.json'),
                reference_result_sha256=sha(tmp_path/'references'/scenario/'RESULT.json')),
            config_sha256=sha(config),source_snapshot=sources,stage3_clock=clock,requested_repetitions=repetitions)
        if retry is not None:intent['retry']=retry
        write_json(target/'INTENT.json',intent)
        row=dict(suite_id='s03',family=family,scenario=scenario,status=status,
                 repetitions=repetitions if status=='PASS' else 0,registry_sha256=registry_sha,
                 intent_sha256=sha(target/'INTENT.json'))
        if status=='PASS':
            receipt=dict(schema='PACNHN_INFERENCE_WALL_V1',suite_id='s03',family=family,scenario=scenario,status='PASS',
                registry_sha256=registry_sha,intent_sha256=row['intent_sha256'],full_model=True,includes_fixed_B=True,
                scope='WHOLE_AC_COST_FIELD',measurement='sequential_calls',repetitions=repetitions,seconds=.003)
            write_json(target/'INFERENCE_WALL.json',receipt)
            row.update(inference_receipt=str(target/'INFERENCE_WALL.json'),inference_receipt_sha256=sha(target/'INFERENCE_WALL.json'))
        write_json(target/'RESULT.json',row);return row

    original=[emit(tmp_path/'inference_pilots/s03/XGBOOST'/s,'XGBOOST',s,'RESOURCE_LIMIT') for s in m.SCENARIOS]
    original.append(emit(tmp_path/'inference_pilots/s03/ICNN/open','ICNN','open','PASS'))
    write_json(tmp_path/'inference_pilots/RESULTS.json',original)
    statuses={s:'PASS' for s in m.SCENARIOS}
    def one(ctx,registry,digest,stage,sid,family,scenario,*,repetitions,seconds,retry=None):
        assert digest==registry_sha and stage==clock and retry is not None
        calls.append((scenario,retry));status=statuses[scenario]
        if isinstance(status,Exception):raise status
        return emit(tmp_path/'inference_pilots/retries'/retry['retry_id']/sid/family/scenario,
                    family,scenario,status,retry=retry,repetitions=repetitions)
    monkeypatch.setattr(m,'_one',one)
    before={str(p):p.read_bytes() for p in (tmp_path/'inference_pilots').rglob('*') if p.is_file()}
    before[str(tmp_path/'STAGE3_CLOCK.json')]=(tmp_path/'STAGE3_CLOCK.json').read_bytes()
    return SimpleNamespace(ctx=ctx,registry_sha=registry_sha,clock=clock,preflight=preflight,calls=calls,
                           statuses=statuses,before=before,emit=emit,original=original)


def retry(e, **kw):
    return m.run(e.ctx,suite_ids=('s03',),families=('XGBOOST',),scenarios=m.SCENARIOS,
                 retry_id=kw.pop('retry_id','projection_v1'),retry_reason='Synthetic explicit operational repair',**kw)


def resolved(e):return m.resolve_inference_results(e.ctx.root,registry_sha256=e.registry_sha)


def test_versioned_retry_preserves_originals_clock_and_selects_exact_complete_cases(env):
    rows=retry(env);r=resolved(env)
    assert len(rows)==2 and all(x['status']=='PASS' for x in rows)
    assert all(Path(p).read_bytes()==raw for p,raw in env.before.items())
    assert [x['resolved_attempt_id'] for x in r['results'] if x['family']=='XGBOOST']==['projection_v1']*2
    assert next(x for x in r['results'] if x['family']=='ICNN')['resolved_attempt_id']=='ORIGINAL'
    assert len(r['history'])==5 and not r['incomplete_retries']
    assert len({(x['suite_id'],x['family'],x['scenario']) for x in r['results']})==3
    for row in rows:assert '/retries/projection_v1/' in row['inference_receipt'].replace('\\','/')


def test_partial_batch_retains_case_files_but_promotes_nothing(env):
    env.statuses['i95']=TimeoutError('Synthetic remaining Stage3 budget expired')
    with pytest.raises(TimeoutError):retry(env)
    r=resolved(env)
    assert [x['status'] for x in r['results'] if x['family']=='XGBOOST']==['RESOURCE_LIMIT']*2
    assert r['incomplete_retries'][0]['status']=='INCOMPLETE_NOT_SELECTED'
    assert (env.ctx.root/'inference_pilots/retries/projection_v1/s03/XGBOOST/open/INFERENCE_WALL.json').exists()
    assert not (env.ctx.root/'inference_pilots/retries/projection_v1/COMPLETION.json').exists()


def test_nonpass_complete_retry_keeps_honest_status_and_can_have_later_version(env):
    env.statuses.update(open='TIME_LIMIT',i95='RESOURCE_LIMIT');retry(env)
    env.statuses.update(open='PASS',i95='PASS');retry(env,retry_id='projection_v2')
    r=resolved(env)
    assert all(x['resolved_attempt_id']=='projection_v2' for x in r['results'] if x['family']=='XGBOOST')
    assert len(r['history'])==7
    intent=read_json(env.ctx.root/'inference_pilots/retries/projection_v2/RETRY_INTENT.json')
    assert all(p['attempt_id']=='projection_v1' for p in intent['predecessors'])


def test_retry_existing_id_and_successful_predecessors_are_rejected(env):
    retry(env)
    with pytest.raises(FileExistsError):retry(env)
    with pytest.raises(ValueError,match='RESOURCE_LIMIT/TIME_LIMIT'):retry(env,retry_id='v2')
    assert not (env.ctx.root/'inference_pilots/retries/v2').exists()


@pytest.mark.parametrize("bad", ["../external", "a/b", "a\\b", ".", "", None, 42])
def test_retry_id_is_safe_and_explicit(env,bad):
    with pytest.raises(ValueError):m._retry_id(bad)


@pytest.mark.parametrize('families,suites',[(('ICNN',),('s03',)),(('XGBOOST',),('s04',)),(('XGBOOST','ICNN'),('s03',))])
def test_retry_cannot_broaden_family_or_suite(env,families,suites):
    with pytest.raises(ValueError):
        m.run(env.ctx,suite_ids=suites,families=families,retry_id='v1',retry_reason='explicit')
    assert not env.calls


def test_retry_requires_existing_clock_and_never_creates_replacement(env):
    (env.ctx.root/'STAGE3_CLOCK.json').unlink()
    with pytest.raises(ValueError,match='original persisted'):retry(env)
    assert not env.calls and not (env.ctx.root/'STAGE3_CLOCK.json').exists()


def test_fresh_technical_gate_precedes_any_retry_creation(env,monkeypatch):
    def reject(*args):raise ValueError('Code differs from reviewed preflight')
    monkeypatch.setattr(gates,'require_component',reject)
    with pytest.raises(ValueError,match='preflight'):retry(env)
    assert not env.calls and not (env.ctx.root/'inference_pilots/retries').exists()


def test_retry_requires_unchanged_preflight_bytes_and_source(env):
    path=env.ctx.root/'validation/PREFLIGHT_LATEST.json'
    write_json(path,dict(env.preflight,status='FAIL'))
    with pytest.raises(ValueError,match='preflight changed'):retry(env)
    assert not env.calls


@pytest.mark.parametrize('what',['result','intent','receipt','ledger','support','clock','reference','completion','preflight','predecessor','original_ledger'])
def test_resolver_rejects_changed_bound_metadata(env,what):
    retry(env);root=env.ctx.root;b=root/'inference_pilots/retries/projection_v1';case=b/'s03/XGBOOST/open'
    if what=='result':path=case/'RESULT.json';obj=read_json(path);obj['status']='TIME_LIMIT'
    elif what=='intent':path=case/'INTENT.json';obj=read_json(path);obj['scenario']='i95'
    elif what=='receipt':path=case/'INFERENCE_WALL.json';obj=read_json(path);obj['seconds']=100
    elif what=='ledger':path=b/'RESULTS.json';obj=read_json(path);obj.pop()
    elif what=='support':path=root/'SUPPORT_FREEZE.json';obj={'different':'support'}
    elif what=='clock':path=root/'STAGE3_CLOCK.json';obj=dict(env.clock,deadline_epoch=env.clock['deadline_epoch']+1)
    elif what=='reference':path=root/'references/open/RESULT.json';obj={'different':'reference'}
    elif what=='completion':path=b/'COMPLETION.json';obj=read_json(path);obj['outcomes'].pop()
    elif what=='preflight':path=b/'PREFLIGHT.json';obj=read_json(path);obj['status']='FAIL'
    elif what=='predecessor':path=b/'RETRY_INTENT.json';obj=read_json(path);obj['predecessors'][0]['result_path']='outside.json'
    else:path=root/'inference_pilots/RESULTS.json';obj=read_json(path);obj.reverse()
    write_json(path,obj)
    with pytest.raises(ValueError):resolved(env)


def test_original_success_receipt_cannot_point_to_external_file_even_with_matching_hash(env,tmp_path):
    row=copy.deepcopy(env.original[-1]);source=Path(row['inference_receipt']);external=tmp_path/'external_receipt.json'
    external.write_bytes(source.read_bytes());row['inference_receipt']=str(external)
    ledger=copy.deepcopy(env.original);ledger[-1]=row
    write_json(env.ctx.root/'inference_pilots/RESULTS.json',ledger)
    write_json(source.parent/'RESULT.json',row)
    with pytest.raises(ValueError,match='escapes'):resolved(env)


def test_resolver_does_not_promote_orphan_without_complete_ledger(env):
    retry(env);directory=env.ctx.root/'inference_pilots/retries/projection_v1'
    (directory/'COMPLETION.json').unlink();(directory/'RESULTS.json').unlink()
    r=resolved(env)
    assert len(r['incomplete_retries'])==1
    assert all(x['resolved_attempt_id']=='ORIGINAL' for x in r['results'])


def test_original_run_still_rejects_existing_cases_without_touching_outputs(env):
    with pytest.raises(ValueError,match='already recorded'):
        m.run(env.ctx,suite_ids=('s03',),families=('XGBOOST',))
    assert not env.calls and all(Path(p).read_bytes()==raw for p,raw in env.before.items())
