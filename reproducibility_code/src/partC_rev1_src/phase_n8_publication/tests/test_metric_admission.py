from types import SimpleNamespace
import numpy as np
import pytest
from partC_rev1_src.phase_n8.common import write_json,save_npz,array_hash,sha
from partC_rev1_src.phase_n8_publication.metrics import admit,final_new_case


def test_fresh_flag_does_not_admit_nonfinite_or_false_convergence(tmp_path):
    ctx=SimpleNamespace(bind=lambda *a:None);flow=np.array([2.,0.]);save_npz(tmp_path/'final_flow.npz',{'flow':flow})
    cert=dict(state_hash=array_hash({'flow':flow}),raw_signed_gap=1.,arithmetic_allowance=1e-8,SPTT=100.,RG_SPTT=.01)
    record=dict(final_certificate_fresh=True,final_certificate=cert,stop_reason='TIME_LIMIT',final_flow_sha256=sha(tmp_path/'final_flow.npz'))
    write_json(tmp_path/'RESULT.json',record);assert admit(ctx,tmp_path)[2]==cert['state_hash']
    write_json(tmp_path/'RESULT.json',dict(record,stop_reason='CONVERGED'))
    with pytest.raises(ValueError,match='Convergence label'):admit(ctx,tmp_path)
    write_json(tmp_path/'RESULT.json',dict(record,final_certificate=dict(cert,RG_SPTT=None)))
    with pytest.raises(ValueError,match='invalid signed gap'):admit(ctx,tmp_path)


def test_physical_case_selection_uses_physical_job_clock(tmp_path):
    for name,start in [('first',3),('second',4)]:
        write_json(tmp_path/'runs'/name/'RESULT.json',dict(scenario='P_CLOSED',identity=dict(job_start_epoch=start)))
    assert final_new_case(SimpleNamespace(root=tmp_path),'P_CLOSED').name=='second'
