"""Filesystem-only tests of exact control AST; no scientific imports or data."""
import ast
import errno
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid
import pytest


@pytest.fixture
def control():
    source=Path(__file__).resolve().parents[1]/'control.py'
    tree=ast.parse(source.read_text())
    names={'json_default','_atomic_replace','write_json','save_npz'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==names
    clock=SimpleNamespace(now=0.,sleeps=[])
    def sleep(seconds):
        assert seconds>=0
        clock.sleeps.append(seconds);clock.now+=seconds
    fake_np=SimpleNamespace(integer=type('FakeInt',(),{}),floating=type('FakeFloat',(),{}),
        ndarray=type('FakeArray',(),{}),savez_compressed=lambda f,**a:f.write(b'NPZ_SYNTHETIC_BYTES'))
    namespace=dict(Path=Path,json=json,uuid=SimpleNamespace(uuid4=uuid.uuid4),np=fake_np,
        os=SimpleNamespace(replace=os.replace,getpid=os.getpid),
        time=SimpleNamespace(monotonic=lambda:clock.now,sleep=sleep))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    return SimpleNamespace(**namespace,clock=clock)


def winerror(code):
    error=PermissionError(errno.EACCES,'synthetic Windows replace denial')
    error.winerror=code
    return error


def test_normal_json_bytes_and_success_consumes_temp(control,tmp_path):
    p=tmp_path/'a'/'state.json';value={'integer':1,'path':Path('portable'),'string':'value'}
    assert control.write_json(p,value) is None
    expected=(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n').replace('\n',os.linesep).encode('utf8')
    assert p.read_bytes()==expected
    assert not list(p.parent.glob(p.name+'.tmp.*')) and not control.clock.sleeps


@pytest.mark.parametrize('code',[5,32,33])
def test_transient_replace_then_success_is_atomic_and_same_temp(control,tmp_path,code):
    p=tmp_path/'state.json';p.write_bytes(b'original-destination');seen=[]
    real=control.os.replace
    def replace(source,destination):
        source=Path(source);seen.append(source)
        assert p.read_bytes()==b'original-destination'
        assert json.loads(source.read_text())=={'revision':2}
        if len(seen)<=2:raise winerror(code)
        return real(source,destination)
    control.os.replace=replace
    control.write_json(p,{'revision':2})
    assert len(seen)==3 and len(set(seen))==1
    assert json.loads(p.read_text())=={'revision':2}
    assert control.clock.sleeps==[.01,.02] and not seen[0].exists()


@pytest.mark.parametrize('code',[5,32,33])
def test_permanent_allowlisted_failure_is_bounded_and_preserves_both_files(control,tmp_path,code):
    p=tmp_path/'state.json';p.write_bytes(b'original');attempts=[];error=winerror(code)
    def replace(src,dst):attempts.append(Path(src));raise error
    control.os.replace=replace
    with pytest.raises(PermissionError) as caught:control.write_json(p,{'new':True})
    assert caught.value is error and len(attempts)==8 and len(set(attempts))==1
    assert sum(control.clock.sleeps)==pytest.approx(.81)
    assert p.read_bytes()==b'original' and json.loads(attempts[0].read_text())=={'new':True}


@pytest.mark.parametrize('error',[PermissionError(errno.EACCES,'no Windows code'),FileNotFoundError(errno.ENOENT,'missing'),OSError(errno.ENOSPC,'disk full')])
def test_nontransient_error_is_immediate(control,tmp_path,error):
    p=tmp_path/'state.json';p.write_bytes(b'original');calls=[]
    def replace(src,dst):calls.append(src);raise error
    control.os.replace=replace
    with pytest.raises(type(error)) as caught:control.write_json(p,{'new':1})
    assert caught.value is error and len(calls)==1 and control.clock.sleeps==[]
    assert p.read_bytes()==b'original' and Path(calls[0]).exists()


def test_nonallowlisted_windows_error_is_immediate(control,tmp_path):
    error=winerror(112);calls=[]
    def replace(src,dst):calls.append(src);raise error
    control.os.replace=replace
    with pytest.raises(PermissionError):control.write_json(tmp_path/'state.json',{'new':1})
    assert len(calls)==1 and control.clock.sleeps==[]


def test_monotonic_budget_prevents_late_retry(control,tmp_path):
    calls=[];error=winerror(32)
    def replace(src,dst):
        calls.append(src);control.clock.now=.999;raise error
    control.os.replace=replace
    with pytest.raises(PermissionError):control.write_json(tmp_path/'state.json',{'new':1})
    assert len(calls)==1 and sum(control.clock.sleeps)==pytest.approx(.001)
    assert control.clock.now==1.


def test_consecutive_failures_retain_distinct_temp_evidence(control,tmp_path):
    p=tmp_path/'state.json';p.write_bytes(b'original')
    def replace(src,dst):raise winerror(5)
    control.os.replace=replace
    for revision in (1,2):
        with pytest.raises(PermissionError):control.write_json(p,{'revision':revision})
    temps=list(tmp_path.glob('state.json.tmp.*'))
    assert len(temps)==2 and sorted(json.loads(t.read_text())['revision'] for t in temps)==[1,2]
    assert p.read_bytes()==b'original'


def test_exclusive_temp_collision_does_not_overwrite_evidence(control,tmp_path):
    p=tmp_path/'state.json';p.write_bytes(b'original');control.uuid.uuid4=lambda:SimpleNamespace(hex='fixed')
    temporary=tmp_path/f'state.json.tmp.{os.getpid()}.fixed';temporary.write_bytes(b'prior-failed-temp')
    with pytest.raises(FileExistsError):control.write_json(p,{'revision':2})
    assert temporary.read_bytes()==b'prior-failed-temp' and p.read_bytes()==b'original'
    assert control.clock.sleeps==[]


def test_serialization_failure_keeps_destination_without_new_temp(control,tmp_path):
    p=tmp_path/'state.json';p.write_bytes(b'original')
    with pytest.raises(ValueError):control.write_json(p,{'nonfinite':float('nan')})
    assert p.read_bytes()==b'original' and list(tmp_path.glob('state.json.tmp.*'))==[]


def test_partial_write_failure_keeps_old_destination_and_partial_temp(control,tmp_path,monkeypatch):
    p=tmp_path/'state.json';p.write_bytes(b'original');real_open=Path.open
    class FailingStream:
        def __init__(self,path,args,kwargs):self.stream=real_open(path,*args,**kwargs)
        def __enter__(self):return self
        def write(self,value):self.stream.write(value[:5]);self.stream.flush();raise OSError(errno.ENOSPC,'synthetic full disk')
        def __exit__(self,*args):self.stream.close()
    def open_path(path,*args,**kwargs):
        if args and args[0]=='x' and '.tmp.' in path.name:return FailingStream(path,args,kwargs)
        return real_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',open_path)
    with pytest.raises(OSError):control.write_json(p,{'revision':2})
    temps=list(tmp_path.glob('state.json.tmp.*'))
    assert p.read_bytes()==b'original' and len(temps)==1 and temps[0].read_bytes()
    assert control.clock.sleeps==[]


def test_npz_wrapper_uses_same_retry_and_exclusive_temp(control,tmp_path):
    p=tmp_path/'state.npz';p.write_bytes(b'original');real=control.os.replace;calls=[]
    def replace(src,dst):
        calls.append(src)
        assert p.read_bytes()==b'original'
        if len(calls)==1:raise winerror(32)
        return real(src,dst)
    control.os.replace=replace;control.save_npz(p,synthetic='not-an-array')
    assert p.read_bytes()==b'NPZ_SYNTHETIC_BYTES' and len(calls)==2 and calls[0]==calls[1]
    assert not list(tmp_path.glob('state.npz.tmp.*'))


def test_npz_serializer_failure_keeps_destination_and_temp(control,tmp_path):
    p=tmp_path/'state.npz';p.write_bytes(b'original')
    def fail(stream,**arrays):stream.write(b'partial-npz');raise ValueError('synthetic serializer failure')
    control.np.savez_compressed=fail
    with pytest.raises(ValueError):control.save_npz(p,synthetic=1)
    temps=list(tmp_path.glob('state.npz.tmp.*'))
    assert p.read_bytes()==b'original' and len(temps)==1 and temps[0].read_bytes()==b'partial-npz'
    assert control.clock.sleeps==[]


@pytest.mark.skipif(os.name!='nt',reason='Real Windows file sharing semantics')
def test_windows_real_sharing_lock_released_then_atomic_replace(control,tmp_path):
    import ctypes
    from ctypes import wintypes
    import threading
    import time
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
        wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    target=tmp_path/'locked_destination.json';target.write_bytes(b'original')
    temporary=tmp_path/'owned_temporary.json';temporary.write_bytes(b'complete-new-state')
    # Share reads/writes, deliberately omit FILE_SHARE_DELETE needed by replacement.
    handle=kernel.CreateFileW(str(target),0x80000000,0x1|0x2,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    held=[handle];mutex=threading.Lock();release=threading.Event();cleanup_errors=[];observed=[]
    def close():
        with mutex:
            if held[0] is not None:
                if not kernel.CloseHandle(held[0]):cleanup_errors.append(ctypes.get_last_error())
                held[0]=None
    def release_later():
        release.wait(.05);close()
    real_replace=os.replace
    def replace(src,dst):
        try:return real_replace(src,dst)
        except OSError as exc:
            observed.append(getattr(exc,'winerror',None))
            assert target.read_bytes()==b'original' and temporary.read_bytes()==b'complete-new-state'
            raise
    control.os.replace=replace
    control.time.monotonic=time.monotonic;control.time.sleep=time.sleep
    worker=threading.Thread(target=release_later,daemon=True)
    started=time.monotonic()
    try:
        worker.start();control._atomic_replace(temporary,target)
        assert time.monotonic()-started<2.
        assert observed and all(code in (5,32,33) for code in observed)
        assert target.read_bytes()==b'complete-new-state' and not temporary.exists()
    finally:
        release.set()
        if worker.ident is not None:worker.join(timeout=1.)
        close()
    assert not worker.is_alive() and not cleanup_errors
