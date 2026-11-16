import copy
import pytest
from partC_rev1_src.pacnhn_recovery24.selection import apply_admissions


def fixture():
    row=dict(suite_id='s03',region_index=1,family='ICNN',graph_sha256='g',pair_keys_identity='p',model_path='model.npz',model_sha256='m')
    report=dict(status='PASS',model_sha256='m',record_bindings=[copy.deepcopy(row)],report_path='report.json',report_sha256='r')
    return row,report


def test_property_pass_is_exact_region_order_and_model_bound():
    row,report=fixture()
    output=apply_admissions([row],{'m':report})[0]
    assert output['property_status']=='PASS' and output['property_record_bound']
    assert 'property_status' not in row


@pytest.mark.parametrize('field',['region_index','family','graph_sha256','pair_keys_identity','model_path','model_sha256'])
def test_property_pass_cannot_be_transferred(field):
    row,report=fixture();report['record_bindings'][0][field]='different'
    with pytest.raises(ValueError,match='lacks exact'):apply_admissions([row],{'m':report})


def test_unrun_and_failure_are_explicit_not_success():
    row,report=fixture()
    assert apply_admissions([row],{})[0]['property_status']=='NUMERIC_INCONCLUSIVE'
    report['status']='FAIL';report['record_bindings']=[]
    assert apply_admissions([row],{'m':report})[0]['property_status']=='FAIL'


def test_admission_file_body_and_hash_must_match_transport(tmp_path):
    from types import SimpleNamespace
    from partC_rev1_src.pacnhn_recovery24.selection import authenticate_admissions
    from partC_rev1_src.pacnhn_recovery24.labels import new_json
    from partC_rev1_src.pacnhn_v1.control import sha
    path=tmp_path/'admission/one.json';body={'model_sha256':'m','status':'FAIL'};new_json(path,body)
    def bind(p,role,expected):
        if sha(p)!=expected:raise ValueError('SHA mismatch')
    ctx=SimpleNamespace(root=tmp_path,bind=bind)
    report={**body,'report_path':str(path),'report_sha256':sha(path)}
    authenticate_admissions(ctx,{'m':report})
    report['status']='PASS'
    with pytest.raises(ValueError,match='transport'):authenticate_admissions(ctx,{'m':report})


def test_ablation_freeze_rejects_duplicate_or_nonterminal_tasks():
    from types import SimpleNamespace
    from partC_rev1_src.pacnhn_recovery24.selection import validate_ablation_dispositions
    from partC_rev1_src.pacnhn_recovery24.ablation import SCHEMA
    tasks=[{'task_id':str(i)} for i in range(160)];plan={'tasks':tasks}
    ctx=SimpleNamespace(c={'phase_id':'phase','run_id':'run'})
    frozen=dict(schema=SCHEMA,status='ALL_160_DISPOSITIONS_FROZEN',phase_id='phase',run_id='run',
        plan_sha256='p',total=160,TEST_used=False,completed=0,ablation_complete=False,frozen_epoch=0,
        outcomes=[{'task':t,'status':'UNATTEMPTED_TIME_LIMIT'} for t in tasks])
    validate_ablation_dispositions(ctx,frozen,plan,'p')
    frozen['outcomes'][1]['task']=tasks[0]
    with pytest.raises(ValueError,match='distinct terminal'):validate_ablation_dispositions(ctx,frozen,plan,'p')
