"""Source-authenticated metadata projection; no model or scientific data load."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from partC_rev1_src.pacnhn_v1 import registry_projection as rp


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture():
    rows = []
    for region, status in ((0, 'AVAILABLE'), (1, 'MISSING')):
        rows.append(dict(suite_id='s03', family='XGBOOST', region_index=region,
            status=status, pair_keys_identity='pair'+str(region), graph_sha256='graph',
            artifact_hashes={'one.npz': 'artifact'}, model_path='one.npz',
            model_sha256='model', model_kind='cost', train_dataset_identity='train',
            current_train_dataset_identity='current', selected_receipt_sha256='receipt',
            TRAIN_WMAPE_percent=17, selected_audit={'large': [1, 2, 3]},
            note='Quoted "value", slash\\, brace } and newline\n are data'))
    return dict(schema='PACNHN_REGISTRY_V1', support_freeze_sha256='support',
        selection_frozen_epoch=1., models=rows,
        coverage=[dict(suite_id='s03', family='XGBOOST', expected=2, available=1,
            qualified=0, complete_family=False)],
        current_dataset_commitment={'regions': {'s03:000000': {'deep': ['large']*8}}},
        XGB_memory_budget={'graph_runtime_reserve_bytes': 200},
        XGB_per_suite_memory_forecast={'s03': {'forecast_resident_booster_bytes': 400}})


def save(tmp_path, value, *, newline='\n'):
    path = tmp_path/'registry.json'
    path.write_bytes((json.dumps(value, indent=2, ensure_ascii=False)+'\n').replace('\n', newline).encode())
    return path


@pytest.mark.parametrize('newline', ['\n', '\r\n'])
def test_projection_keeps_every_row_runtime_field_and_scalar_identity(tmp_path, newline):
    source = fixture();path = save(tmp_path, source, newline=newline)
    observed = rp.read_stage3_registry(path, digest(path))
    assert set(observed) == set(source)-{'current_dataset_commitment'}
    assert len(observed['models']) == len(source['models']) == 2
    for original, projected in zip(source['models'], observed['models']):
        expected = {k: v for k, v in original.items() if k in rp.MODEL_FIELDS or
                    k.endswith('_identity') or k.endswith('_sha256')}
        assert projected == expected
    assert observed['models'][1]['status'] == 'MISSING'
    for key in set(observed)-{'models'}:
        assert observed[key] == source[key]


def test_excluded_metadata_tampering_is_detected_by_full_hash(tmp_path):
    source = fixture();path = save(tmp_path, source);expected = digest(path)
    source['current_dataset_commitment']['regions']['s03:000000']['deep'][0] = 'tampered'
    save(tmp_path, source)
    with pytest.raises(ValueError, match='SHA-256 differs'):
        rp.read_stage3_registry(path, expected)


@pytest.mark.parametrize('sha', [None, '', 'old', 'a'*63, 'A'*64])
def test_source_digest_required(tmp_path, sha):
    path = save(tmp_path, fixture())
    with pytest.raises(ValueError, match='exact registry SHA-256'):
        rp.read_stage3_registry(path, sha)


@pytest.mark.parametrize('kind', ['model', 'root', 'nested', 'coverage'])
def test_duplicates_rejected(tmp_path, kind):
    source = fixture()
    if kind == 'model':source['models'].append(source['models'][0])
    if kind == 'coverage':source['coverage'].append(source['coverage'][0])
    path = save(tmp_path, source)
    if kind in ('root', 'nested'):
        raw = path.read_text()
        line = ('  "schema": "PACNHN_REGISTRY_V1",\n' if kind == 'root'
                else '      "model_sha256": "model",\n')
        path.write_text(raw.replace(line, line+line, 1), encoding='utf8')
    with pytest.raises(ValueError, match='Duplicate'):
        rp.read_stage3_registry(path, digest(path))


@pytest.mark.parametrize('field,value', [('expected', 3), ('available', 2),
    ('complete_family', True), ('expected', True)])
def test_coverage_cannot_change_model_availability(tmp_path, field, value):
    source = fixture();source['coverage'][0][field] = value
    path = save(tmp_path, source)
    with pytest.raises(ValueError, match='coverage differs'):
        rp.read_stage3_registry(path, digest(path))


@pytest.mark.parametrize('mode', ['compact', 'indent4', 'trailing', 'truncated', 'comma'])
def test_unsupported_or_malformed_format_fails_closed(tmp_path, mode):
    path = save(tmp_path, fixture())
    if mode == 'compact':path.write_text(json.dumps(fixture()), encoding='utf8')
    if mode == 'indent4':path.write_text(json.dumps(fixture(), indent=4)+'\n', encoding='utf8')
    if mode == 'trailing':path.write_bytes(path.read_bytes()+b'junk')
    if mode == 'truncated':path.write_bytes(path.read_bytes()[:-10])
    if mode == 'comma':path.write_text(path.read_text().replace('    }\n  ],', '    },\n  ],', 1))
    with pytest.raises(ValueError):rp.read_stage3_registry(path, digest(path))


def test_bound_record_size(tmp_path, monkeypatch):
    path = save(tmp_path, fixture());monkeypatch.setattr(rp, 'MAX_VALUE_BYTES', 50)
    with pytest.raises(ValueError, match='bounded reader limit'):
        rp.read_stage3_registry(path, digest(path))


def test_guard_can_stop_loading(tmp_path):
    path = save(tmp_path, fixture())
    def guard():raise TimeoutError('expired')
    with pytest.raises(TimeoutError, match='expired'):
        rp.read_stage3_registry(path, digest(path), guard=guard)


def test_file_mutation_before_open_rejected(tmp_path):
    path = save(tmp_path, fixture());expected = digest(path);calls = []
    def guard():
        calls.append(1)
        if len(calls) == 1:path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError, match='changed before reading'):
        rp.read_stage3_registry(path, expected, guard=guard)


def test_file_mutation_during_read_rejected(tmp_path):
    path = save(tmp_path, fixture());expected = digest(path);calls = []
    def guard():
        calls.append(1)
        if len(calls) == 2:path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):rp.read_stage3_registry(path, expected, guard=guard)


def test_empty_and_missing_groups_are_not_invented(tmp_path):
    source = fixture();source['models'] = [];source['coverage'] = []
    path = save(tmp_path, source)
    assert rp.read_stage3_registry(path, digest(path))['models'] == []
    source['coverage'] = [dict(suite_id='s03',family='XGBOOST',expected=2,available=0,complete_family=False)]
    path = save(tmp_path, source)
    with pytest.raises(ValueError, match='coverage differs'):
        rp.read_stage3_registry(path, digest(path))


def test_gate_uses_projection_and_keeps_full_external_hash_bindings(tmp_path, monkeypatch):
    from partC_rev1_src.pacnhn_v1 import assignments as a
    from partC_rev1_src.pacnhn_v1.control import write_json, sha
    support = tmp_path/'SUPPORT_FREEZE.json';write_json(support, dict(status='PASS'))
    write_json(tmp_path/'COMPLETE_WITNESS_AUDIT.json',dict(status='PASS',support_freeze_sha256=sha(support)))
    source = fixture();source['support_freeze_sha256'] = sha(support)
    registry = tmp_path/'training/REGISTRY_FREEZE.json';write_json(registry, source)
    write_json(tmp_path/'training/RESULT.json', dict(status='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE',registry_sha256=sha(registry)))
    for name in ['TEST_OPENING_AUTHORIZATION.json', 'FINAL_TEST_SUMMARY.json']:
        write_json(tmp_path/'training'/name, dict(registry_sha256=sha(registry)))
    original = a.read_json
    def no_full_registry(path):
        assert Path(path) != registry
        return original(path)
    monkeypatch.setattr(a, 'read_json', no_full_registry)
    bindings = [];guards = []
    ctx = SimpleNamespace(root=tmp_path, bind=lambda p,r:bindings.append((p,r)),
                          guard=lambda **kw:guards.append(kw))
    result = a.validate_gates(ctx, deadline=123.)
    assert len(result['models']) == 2 and 'current_dataset_commitment' not in result
    assert (registry, 'FROZEN_REGIONAL_REGISTRY') in bindings
    assert guards and all(r == {'deadline': 123.} for r in guards)
