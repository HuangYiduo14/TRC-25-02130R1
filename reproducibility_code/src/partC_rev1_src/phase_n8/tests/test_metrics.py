import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from partC_rev1_src.phase_n8.metrics import path_summary,R3Context
from partC_rev1_src.phase_n8.common import write_json


def test_path_metrics_population_denominators_and_zero_reference():
    f=pd.DataFrame(dict(D_phys_s=[10.,0.,20.,np.inf],D_surrogate_s=[12.,3.,np.inf,5.],
        demand=[2.,1.,4.,1.],ratio_weight=[4.,3.,8.,2.]))
    r=path_summary(f,track='FIXED_X45',scenario='fixture',population='known')
    assert r['valid_paired_rows']==2 and r['cohort_rows']==4
    assert r['WREL']==pytest.approx(7./20.)
    assert r['WMEANREL_1S']==pytest.approx((.4+3)/3)
    assert r['additional_surrogate_unreachable_rows']==1
    assert r['physical_unreachable_rows']==1 and r['zero_reference_rows']==1
    z=path_summary(f.iloc[[1]],track='FIXED_X45',scenario='fixture',population='zero')
    assert z['WREL'] is None and z['WMEANREL_1S']==3.


def test_r3_entry_rejects_false_or_unbound_gate(tmp_path):
    t=time.time();config=tmp_path/'config.json'
    write_json(config,dict(run_id='test',start_epoch=t,deadline_epoch=t+28800,
        repository=str(tmp_path),output_root=str(tmp_path)))
    write_json(tmp_path/'FAST_TRACK_GATE.json',dict(R3_auto_start_allowed=False))
    write_json(tmp_path/'FAST_TRACK_GATE_INPUT.json',dict(frozen_manifest_sha256='0'*64))
    with pytest.raises(PermissionError):R3Context(config,'test',t+50)
    write_json(tmp_path/'CAMPAIGN_MANIFEST.json',dict(original_R2_start_epoch=t,total_deadline_epoch=t+43200))
    write_json(tmp_path/'FAST_TRACK_GATE.json',dict(R3_auto_start_allowed=True))
    with pytest.raises(PermissionError):R3Context(config,'test',t+50)
