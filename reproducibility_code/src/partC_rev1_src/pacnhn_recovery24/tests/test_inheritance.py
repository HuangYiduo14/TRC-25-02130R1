"""Tiny opaque byte fixtures: no model loading, label decoding, or fitting."""
import copy
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
from partC_rev1_src.pacnhn_recovery24 import inheritance as i


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');return sha(p)


class Context:
    def __init__(self,root,source):
        self.root=root;self.source_root=source;self.bindings={};self.source_expected={};self.c={'recovery':{}};self.actions=[]
    def require_action(self,action):self.actions.append(action)
    def guard(self,deadline=None):
        if deadline is not None and time.time()>=deadline:raise TimeoutError('budget')
    def bind(self,p,role,expected=None):
        p=Path(p).resolve();actual=sha(p)
        if expected is not None and actual!=expected:raise ValueError('expected digest')
        pin=self.source_expected.get(os.path.normcase(str(p)))
        if pin is not None and actual!=pin:raise ValueError('pin')
        st=p.stat();self.bindings[str(p)]=dict(path=str(p),sha256=actual,bytes=st.st_size,mtime_ns=st.st_mtime_ns,role=role)
        return actual
    def pin(self,p):self.source_expected[os.path.normcase(str(p.resolve()))]=sha(p)


@pytest.fixture
def fixture(tmp_path):
    old=tmp_path/'old';root=tmp_path/'new';old.mkdir();root.mkdir();ctx=Context(root,old)
    graph=old/'regions/s03/000001/graph.npz';graph.parent.mkdir(parents=True);graph.write_bytes(b'opaque graph')
    supportrow=dict(suite_id='s03',region_index=1,path=str(graph),sha256=sha(graph),pairs=3)
    sp=old/'SUPPORT_FREEZE.json';save(sp,dict(status='PASS',suites=[dict(region_records=[supportrow])]))
    base=old/'training/regions/s03/000001/labels';sources={};index={}
    for role in ('TRAIN','VALID','TEST'):
        input_id=hashlib.sha256(role.encode()).hexdigest()
        sample=hashlib.sha256(f'g|{role}|{input_id}'.encode()).hexdigest();leaf=base/'labels'/role/sample/'attempt_000001'
        label=leaf/'label.npz';receipt=leaf/'RESULT.json';leaf.mkdir(parents=True)
        label.write_bytes(('opaque '+role).encode());save(receipt,{'role':role,'admitted':True})
        index[input_id]=dict(sample_id=sample,role=role,selected_label=dict(path=str(label),sha256=sha(label),receipt=str(receipt),receipt_sha256=sha(receipt)))
        if role!='TEST':sources.update({str(label):sha(label),str(receipt):sha(receipt)})
    ip=base/'REQUEST_INDEX.json';rp=base/'REGION_IDENTITY.json';save(ip,index)
    save(rp,dict(region_id='s03:000001',graph_sha256=sha(graph),support_identity='s',graph_identity='g'))
    pop=dict(request_index_path=str(ip),request_index_sha256=sha(ip),region_identity_path=str(rp),region_identity_sha256=sha(rp),
        admitted_counts=dict(TRAIN=1,VALID=1,TEST=1),TRAIN=dict(dataset_identity='a'*64,unique_rows=1),
        VALID=dict(dataset_identity='b'*64,unique_rows=1),source_files=sources,TEST_arrays_opened=False)
    rows=[]
    for family in i.FAMILIES:
        mp=old/'models'/f'{family}.npz';mp.parent.mkdir(exist_ok=True);mp.write_bytes(('opaque '+family).encode())
        rows.append(dict(suite_id='s03',region_index=1,family=family,status='AVAILABLE',model_path=str(mp),model_sha256=sha(mp),
            artifact_hashes={str(mp):sha(mp)},model_kind=i.KINDS[family],graph_sha256=sha(graph),pair_keys_identity='c'*64,pairs=3,
            train_rows=1,valid_rows=1,current_train_dataset_identity='a'*64,current_valid_dataset_identity='b'*64,
            train_dataset_identity='a'*64,valid_dataset_identity='b'*64,TRAIN_WMAPE_percent=.2,VALID_WMAPE_percent=.4,
            uncompressed_model_bytes=100,native_parameter_bytes=40,uncompressed_booster_bytes=60,parameters={'total_updates':250},target_met=True))
    rows[0].pop('TRAIN_WMAPE_percent');rows[0].pop('VALID_WMAPE_percent')
    reg=dict(schema='PACNHN_REGISTRY_V1',support_freeze_sha256=sha(sp),models=rows,test_opened_before_selection=False,
        current_dataset_commitment=dict(schema='PACNHN_CURRENT_DATASET_COMMITMENT_V1',regions={'s03:000001':pop},TEST_arrays_opened=False))
    regpath=old/'training/REGISTRY_FREEZE.json';csv=old/'training/LEARNING_RESULTS.csv';csv.write_bytes(b'opaque immutable CSV lineage')
    resultpath=old/'training/RESULT.json'
    def reseal():
        save(regpath,reg);save(resultpath,dict(status='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE',registry=str(regpath),registry_sha256=sha(regpath),learning_csv_sha256=sha(csv)))
        for p in (sp,resultpath,ip,rp):ctx.pin(p)
    reseal();save(root/'training/PRIMARY_RESULT.json',dict(status='PRIMARY_TIME_LIMIT'))
    return SimpleNamespace(ctx=ctx,reg=reg,rows=rows,pop=pop,support=supportrow,base=base,index=index,reseal=reseal,path=regpath)


def run(f,new=(),**kwargs):return i.inherit_selected(f.ctx,list(new),deadline=time.time()+15,**kwargs)


def test_complete_selected_only_proof_never_loads_arrays_or_test(fixture,monkeypatch):
    f=fixture;before={str(p):sha(p) for p in f.ctx.source_root.rglob('*') if p.is_file()};opened=[];original=Path.open
    def guarded(p,*args,**kwargs):
        assert 'TEST' not in p.parts,'TEST target/receipt opened'
        opened.append(str(p));return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    rows,evidence=run(f)
    assert len(rows)==4 and all(r['property_status']=='PENDING' for r in rows)
    assert all(r['c0_identity'] is None for r in rows)
    assert next(r for r in rows if r['family']=='FFTT')['valid_wmape_percent'] is None
    assert next(r for r in rows if r['family']=='XGBOOST')['forecast_resident_booster_bytes']==300
    assert evidence['fitting_performed'] is False and evidence['TEST_values_read'] is False
    assert evidence['inherited_records']==4 and f.ctx.actions==['portfolio']
    monkeypatch.setattr(Path,'open',original)
    assert before=={str(p):sha(p) for p in f.ctx.source_root.rglob('*') if p.is_file()}


def test_new_wins_per_family_even_when_region_partial(fixture):
    new=dict(fixture.rows[2],current_train_dataset_identity='a'*64,current_valid_dataset_identity='b'*64,custom='retained')
    rows,e=run(fixture,[new]);assert len(rows)==4 and e['inherited_records']==3
    assert next(r for r in rows if r['family']=='ICNN')==new
    assert 'property_status' not in new


def test_changed_new_cohort_blocks_old_missing_families(fixture):
    new=dict(fixture.rows[2],current_train_dataset_identity='d'*64)
    rows,e=run(fixture,[new]);assert rows==[new] and all(r['status']=='NOT_INHERITED' for r in e['dispositions'])


def test_changed_overlay_is_not_old_cohort(fixture):
    fixture.ctx.c['recovery']['training_overlays']={'s03:000001':{'path':str(fixture.ctx.root/'labels/overlay'),'sha256':'a'*64}}
    rows,e=run(fixture);assert not rows and all('overlay' in r['reason'] for r in e['dispositions'])


def test_producer_input_id_and_sample_id_are_distinct(fixture):
    assert all(input_id!=r['sample_id'] for input_id,r in fixture.index.items())
    assert len(run(fixture)[0])==4
    first=next(iter(fixture.index));fixture.index[first]['sample_id']=first
    save(fixture.base/'REQUEST_INDEX.json',fixture.index)
    fixture.pop['request_index_sha256']=sha(fixture.base/'REQUEST_INDEX.json');fixture.reseal()
    fixture.ctx.bindings={}  # New synthetic authority, so test the canonical-ID guard rather than prior-byte pinning.
    rows,e=run(fixture);assert not rows
    assert all('canonical' in r['reason'] for r in e['dispositions'])


@pytest.mark.parametrize('field,value',[('TRAIN_WMAPE_percent',1.),('TRAIN_WMAPE_percent',float('inf')),
    ('valid_dataset_identity','d'*64),('pair_keys_identity',None),('graph_sha256','d'*64),('train_rows',2),('model_kind','different')])
def test_unsafe_selected_metadata_rejected(fixture,field,value):
    row=copy.deepcopy(fixture.rows[2]);row[field]=value
    with pytest.raises(ValueError):i.normalize(row,fixture.pop,fixture.support,{})


def test_power4_above_target_still_preserved(fixture):
    fixture.rows[1]['TRAIN_WMAPE_percent']=50.;fixture.reseal()
    rows,_=run(fixture);assert next(r for r in rows if r['family']=='POWER4')['train_wmape_percent']==50.


@pytest.mark.parametrize('which',['index','label','receipt','model'])
def test_changed_bytes_rejected_without_substitution(fixture,which):
    if which=='index':p=fixture.base/'REQUEST_INDEX.json'
    elif which=='model':p=Path(fixture.rows[2]['model_path'])
    else:p=next(Path(p) for p in fixture.pop['source_files'] if p.endswith('label.npz' if which=='label' else 'RESULT.json'))
    p.write_bytes(p.read_bytes()+b'changed');rows,e=run(fixture)
    if which=='model':assert len(rows)==3 and not any(r['family']=='ICNN' for r in rows)
    else:assert not rows
    assert any(r['status']=='NOT_INHERITED' for r in e['dispositions'])


def test_committed_sources_cannot_include_test(fixture):
    test=next(r for r in fixture.index.values() if r['role']=='TEST')['selected_label']
    fixture.pop['source_files'][test['path']]=test['sha256'];fixture.reseal()
    rows,e=run(fixture);assert not rows and all('population' in r['reason'] for r in e['dispositions'])


def test_current_population_count_discrepancy(fixture):
    fixture.pop['admitted_counts']['TRAIN']=2;fixture.reseal()
    assert not run(fixture)[0]


def test_missing_population_uses_implicit_fftt_without_invented_metrics(fixture):
    fixture.pop['TRAIN']['dataset_identity']=None;fixture.reseal()
    rows,e=run(fixture);assert not rows and len(e['dispositions'])==4
    assert all(r['status']=='NO_CURRENT_TRAIN_VALID_PROOF_IMPLICIT_FFTT' for r in e['dispositions'])


def test_unmet_learned_fallback_does_not_discard_valid_power4(fixture):
    fixture.rows[2]['TRAIN_WMAPE_percent']=5.;fixture.reseal()
    rows,e=run(fixture);assert {r['family'] for r in rows}=={'FFTT','POWER4','XGBOOST'}
    assert any('target is unmet' in r.get('reason','') for r in e['dispositions'])


@pytest.mark.parametrize('rel',['training/MODEL_SELECTION_FREEZE.json','holdout/TEST_OPENED.json','training/REGISTRY_FREEZE.json','NATIONAL_TIMING_ACTIVE.json'])
def test_forbidden_stage_fails_before_source_access(fixture,rel,monkeypatch):
    save(fixture.ctx.root/rel,{})
    monkeypatch.setattr(i,'_pinned',lambda *a:pytest.fail('source accessed'))
    with pytest.raises(PermissionError):run(fixture)


def test_primary_terminal_required(fixture,monkeypatch):
    (fixture.ctx.root/'training/PRIMARY_RESULT.json').unlink()
    monkeypatch.setattr(i,'_pinned',lambda *a:pytest.fail('source accessed'))
    with pytest.raises(PermissionError):run(fixture)


def test_independent_final_result_pin_required(fixture):
    fixture.ctx.source_expected.pop(os.path.normcase(str(fixture.ctx.source_root/'training/RESULT.json')))
    with pytest.raises(ValueError,match='pin'):run(fixture)


def test_registry_stream_rejects_digest_and_duplicate(fixture):
    with pytest.raises(ValueError,match='bytes differ'):i.project_registry(fixture.path,'f'*64,{'s03:000001'})
    fixture.reg['models'].append(fixture.rows[0]);fixture.reseal()
    with pytest.raises(ValueError,match='Duplicate'):i.project_registry(fixture.path,sha(fixture.path),{'s03:000001'})


def test_projection_skips_other_population_without_losing_hash(fixture):
    fixture.reg['current_dataset_commitment']['regions']['s03:000009']=copy.deepcopy(fixture.pop);fixture.reseal()
    projected=i.project_registry(fixture.path,sha(fixture.path),{'s03:000001'})
    assert set(projected['current_dataset_commitment']['regions'])=={'s03:000001'}


def test_memory_forecast_never_uses_compressed_size(fixture):
    row=copy.deepcopy(fixture.rows[3]);row['uncompressed_model_bytes']=None;row['model_file_bytes']=10
    with pytest.raises(ValueError,match='raw-memory'):i.normalize(row,fixture.pop,fixture.support,{})


def test_hash_timeout_and_hash_failure_close_readers(fixture,monkeypatch):
    active=[];real=i.hash_file
    def fail(p,digest,**kw):
        active.append(str(p))
        try:time.sleep(.01);raise ValueError('fixture mismatch')
        finally:active.remove(str(p))
    monkeypatch.setattr(i,'hash_file',fail)
    with i.HashPool(fixture.ctx,time.time()+2,2) as pool:
        with pytest.raises(ValueError):pool.authenticate(fixture.pop['source_files'])
    assert not active
    with pytest.raises(TimeoutError):real(next(iter(fixture.pop['source_files'])),'a'*64,stop=threading.Event(),deadline=0)


@pytest.mark.parametrize('workers',[0,5,True])
def test_worker_bound(fixture,workers):
    with pytest.raises(ValueError):run(fixture,max_workers=workers)


def test_new_duplicate_and_conflicting_current_cohorts(fixture):
    with pytest.raises(ValueError,match='Duplicate'):run(fixture,[fixture.rows[2]]*2)
    second=dict(fixture.rows[3],current_valid_dataset_identity='d'*64)
    with pytest.raises(ValueError,match='disagree'):run(fixture,[fixture.rows[2],second])
