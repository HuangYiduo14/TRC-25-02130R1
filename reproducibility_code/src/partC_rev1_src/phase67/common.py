"""Explicit new campaign identity, lexical boundaries and immutable receipts."""
from pathlib import Path
import os, time, json, math
from contextlib import contextmanager
from partC_rev1_src.phase7.common import sha,append_json
from .shared_io import read_json,write_json,atomic,save_npz
from partC_rev1_src.phase6.common import input_hash,arrays_hash,digest

def root(c): return Path(c['execution']['output_root'])
def rd(c,rid): return root(c)/'regions'/rid.removeprefix('partC_p2_region_')
def authority(c): return Path(c['execution']['repository_root'])/c['execution']['authority_install_relative']/'authority'
def is_e2(c):return c.get('episode_id')=='p67_postaudit_e2_0908'
def elapsed(c):
    utc=time.time()-c['task_started_unix']
    if is_e2(c):
        import psutil
        if abs(psutil.boot_time()-c['clock_facts']['boot_time'])>=1.:return utc
    return max(utc,time.monotonic()-c['task_started_monotonic'])
def reserve(c):
    if is_e2(c):
        from .e2_forecast import read
        return max(5400.,1.5*read(c)['remaining_seconds'])
    p=root(c)/'control/finalization_forecast.json'
    forecast=read_json(p).get('remaining_seconds',0.) if p.exists() else 0.
    return max(5400.,1.5*forecast)
def left(c):
    value=86400.-reserve(c)-elapsed(c)
    return min(value,c['latest_science_cutoff_unix']-time.time(),c['hard_deadline_unix']-reserve(c)-time.time()) if is_e2(c) else value
def admit(c,forecast=0.):
    if not math.isfinite(forecast) or forecast<0:raise ValueError('Invalid job forecast')
    if is_e2(c):
        if not c.get('_e2_admission_verified',False):return False
        if not (root(c)/'validation/AUDIT2_INPUTS_FROZEN.json').is_file():return False
        if (root(c)/'control/ALL_CHAMPIONS_FROZEN.json').exists():return False
        if any((root(c)/'control/stop_requests').glob('*.json')):return False
    return left(c)>forecast and not (root(c)/'control/STOP_SCIENCE.json').exists()
def load_config(path):
    c=read_json(path)
    assert c['phase_id']=='06_07' and c['step_id'] in ['p67_patch24h_0908','p67_postaudit_e2_0908'] and c['task_list']
    assert c['run_id']==c['execution']['run_id'] and c['execution']['hard_seconds']==86400
    assert c['hard_deadline_unix']==c['task_started_unix']+86400
    assert c['execution']['source_overwrite'] is False and c['execution']['network'] is False
    assert c['device']=='cpu' and c['native_threads']==1
    if is_e2(c):
        from .e2_policy import continuation_identity,validate_roots
        continuation_identity(c['task_started_unix'],c['hard_deadline_unix'],c['execution']['engineering_repairs_max'])
        validate_roots(c['parent_output_root_readonly'],root(c))
        assert c['run_id']=='p67_postaudit_e2_0908_run_001' and c['latest_science_cutoff_unix']==1788964986.
        assert c['execution']['new_clock_explicitly_authorized'] is False
    return c
def native_threads():
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
def file_binding(path):
    p=Path(path);s=p.stat();return dict(path=str(p),sha256=sha(p),size_bytes=s.st_size,modified_unix_ns=s.st_mtime_ns)
@contextmanager
def lease(c,rid,kind):
    p=rd(c,rid)/'LEASE.json';p.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        import psutil
        with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid(),process_created_unix=psutil.Process().create_time(),region_id=rid,kind=kind,unix=time.time()),f)
        yield
    finally:
        # Only remove this process's ephemeral lease, never scientific artifacts.
        if p.exists() and read_json(p).get('pid')==os.getpid():p.unlink()
