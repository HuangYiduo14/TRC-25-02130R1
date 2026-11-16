"""A real resumed FW update, with authenticated untouched prior evidence."""
import time
import numpy as np
import pytest

from .test_core import graph, od, objective, Guard
from partC_rev1_src.phase_n8.common import read_json, write_json, sha
from partC_rev1_src.phase_n8.solver import save_checkpoint, certificate
from partC_rev1_src.phase_n8.oracle import aon
from partC_rev1_src.phase_n8.common import array_hash
from partC_rev1_src.phase_n8.restart import resume_segment


def test_real_resume_preserves_prior_segment_history_and_clock(tmp_path):
    g=graph([0,0],[1,1]);d=od();o=objective(g,[1.,1.],[1.,16.]);f=np.array([3.,0.])
    v,c,_,_=o.evaluate(f);linear=aon(g,d,c)
    cert=certificate(f,c,linear,v,array_hash({'flow':f}))
    started=time.time();clocks=dict(run_start_epoch=started,run_deadline_epoch=started+180)
    ident=dict(graph=g.identity,tolerance=1e-10)
    prior=tmp_path/'prior';new=tmp_path/'continuation'
    save_checkpoint(prior,f,0,ident,clocks,cert)
    write_json(prior/'HISTORY.json',[dict(iteration=0,certificate=cert)])
    originals={str(p):sha(p) for p in prior.rglob('*') if p.is_file()}
    result=resume_segment(g,d,o,d.balance(2),prior,new,identity=ident,clocks=clocks,ctx=Guard(),tol=1e-10)
    assert result['stop_reason']=='CONVERGED' and result['accepted_iterations']==1
    assert result['clocks']==clocks and result['final_certificate_fresh']
    assert len(read_json(new/'CUMULATIVE_HISTORY.json'))==3
    assert originals=={str(p):sha(p) for p in prior.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        resume_segment(g,d,o,d.balance(2),prior,new,identity=ident,clocks=clocks,ctx=Guard(),tol=1e-10)
    with pytest.raises(TimeoutError):
        resume_segment(g,d,o,d.balance(2),prior,tmp_path/'expired',identity=ident,
          clocks=dict(run_start_epoch=started,run_deadline_epoch=started-1),ctx=Guard())
