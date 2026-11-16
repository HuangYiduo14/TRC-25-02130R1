"""Explicit campaign identity, atomic files and fixed original-clock guards."""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
from partC_rev1_src.phase6.common import input_hash, arrays_hash, digest

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def atomic(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.pending.{os.getpid()}')
    with tmp.open('wb') as f: f.write(data);f.flush();os.fsync(f.fileno())
    for i in range(10):
        try: os.replace(tmp,path);return
        except PermissionError:
            if i==9: raise
            time.sleep(.02*(i+1))
def write_json(path,obj):
    atomic(path,(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n').encode())
def append_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f: f.write(json.dumps(obj,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def save_npz(path,**arrays):
    import io
    assert all(np.asarray(v).dtype.kind!='O' for v in arrays.values())
    b=io.BytesIO();np.savez_compressed(b,**arrays);atomic(path,b.getvalue())
def load_config(path):
    c=read_json(path)
    assert c['phase_id']=='07' and c['step_id']=='s03_active_8h_v1' and c['task_list']
    assert c['execution']['hard_seconds']==28800 and c['hard_deadline_unix']==c['task_started_unix']+28800
    assert c['execution']['source_write'] is False and c['execution']['phase6_generation'] is False
    assert c['device']=='cpu' and c['workers']==2 and c['native_threads']==1
    return c
def elapsed(c):
    # Both clocks are persisted; use the stricter clock, including across resume.
    return max(time.time()-c['task_started_unix'],time.monotonic()-c['task_started_monotonic'])
def seconds_left(c,reserve=2700.): return 28800.-max(2700.,reserve)-elapsed(c)
def root(c): return Path(c['execution']['output_root'])
def region_out(c,rid): return root(c)/'regions'/rid.removeprefix('partC_p2_region_')
def assert_training_time(c,forecast=0.,reserve=2700.):
    if seconds_left(c,reserve)<=forecast: raise TimeoutError('Original Phase 7 training cutoff reached')
