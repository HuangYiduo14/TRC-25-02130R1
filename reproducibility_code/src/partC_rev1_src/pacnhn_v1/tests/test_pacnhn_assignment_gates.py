"""Prevent assignments from bypassing current selection and blind-test evidence."""
from types import SimpleNamespace
import pytest
from partC_rev1_src.pacnhn_v1.control import read_json,write_json,sha
from partC_rev1_src.pacnhn_v1.assignments import validate_gates


@pytest.mark.parametrize('disposition',['FINAL_TEST_SUMMARY.json','FINAL_TEST_TIMEOUT.json'])
def test_assignment_requires_hash_bound_training_and_test_disposition(tmp_path,disposition):
    bindings=[];ctx=SimpleNamespace(root=tmp_path,bind=lambda p,r:bindings.append((p,r)))
    support=tmp_path/'SUPPORT_FREEZE.json';write_json(support,dict(status='PASS'))
    write_json(tmp_path/'COMPLETE_WITNESS_AUDIT.json',dict(status='PASS',support_freeze_sha256=sha(support)))
    registry=tmp_path/'training/REGISTRY_FREEZE.json'
    write_json(registry,dict(schema='PACNHN_REGISTRY_V1',support_freeze_sha256=sha(support)))
    write_json(tmp_path/'training/RESULT.json',dict(status='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE',registry_sha256=sha(registry)))
    write_json(tmp_path/'training/TEST_OPENING_AUTHORIZATION.json',dict(registry_sha256=sha(registry)))
    with pytest.raises(ValueError,match='disposition is required'):validate_gates(ctx)
    path=tmp_path/'training'/disposition;write_json(path,dict(registry_sha256='old-registry'))
    with pytest.raises(ValueError,match='not bound'):validate_gates(ctx)
    write_json(path,dict(registry_sha256=sha(registry)))
    assert validate_gates(ctx)==read_json(registry)
    assert (path,'FINAL_BLIND_TEST_DISPOSITION') in bindings
    authorization=tmp_path/'training/TEST_OPENING_AUTHORIZATION.json'
    authorization.unlink()
    with pytest.raises(ValueError,match='explicit never-opened timeout'):validate_gates(ctx)
    write_json(path,dict(registry_sha256=sha(registry),TEST_opening_authorization_exists=False))
    if disposition=='FINAL_TEST_TIMEOUT.json':
        assert validate_gates(ctx)==read_json(registry)
    else:
        with pytest.raises(ValueError,match='explicit never-opened timeout'):validate_gates(ctx)
    write_json(authorization,dict(registry_sha256=sha(registry)))
    write_json(tmp_path/'training/RESULT.json',dict(status='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE',registry_sha256='old'))
    with pytest.raises(ValueError,match='Completed training'):validate_gates(ctx)
