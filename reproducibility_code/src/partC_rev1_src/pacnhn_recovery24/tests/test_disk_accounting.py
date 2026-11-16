"""Tiny metadata-only fixtures; no campaign paths, payload copies or science."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import stat
from types import SimpleNamespace
import threading
import time

import pytest

from partC_rev1_src.pacnhn_recovery24 import disk_accounting as d


@pytest.fixture(autouse=True)
def empty_cache(monkeypatch):
    monkeypatch.setattr(d, '_CACHE', {})


@pytest.fixture
def roots(tmp_path):
    new, old = tmp_path/'new', tmp_path/'old'
    new.mkdir();old.mkdir()
    return new, old


def limits(**changes):
    return dict(dict(new_output_bytes=100, baseline_run_bytes=200, added_disk_bytes=300), **changes)


def check(roots, now, **kwargs):
    return d.check_output_disk(roots[0], source_root=roots[1], resources=limits(**kwargs), now=now)


def test_real_small_tree_counts_only_new_and_preserves_census_epoch(roots):
    new, old = roots
    (new/'a').write_bytes(b'abc');(new/'nested').mkdir();(new/'nested'/'b').write_bytes(b'12345')
    (old/'untouched').write_bytes(b'x'*500)
    first=check(roots,100.)
    assert first['bytes']==8 and not first['cache_reused'] and first['census_epoch']==100.
    (new/'later').write_bytes(b'12')
    cached=check(roots,699.999)
    assert cached['bytes']==8 and cached['cache_reused'] and cached['census_epoch']==100.
    refreshed=check(roots,700.)
    assert refreshed['bytes']==10 and not refreshed['cache_reused'] and refreshed['census_epoch']==700.
    assert (old/'untouched').read_bytes()==b'x'*500


def test_repeated_hits_do_not_renew_expiry(roots,monkeypatch):
    calls=[];monkeypatch.setattr(d,'_scan_output_bytes',lambda p:calls.append(p) or 10)
    for now in (100.,200.,300.,699.):check(roots,now)
    assert len(calls)==1
    check(roots,700.);assert len(calls)==2


def test_clock_rollback_since_last_call_forces_scan(roots,monkeypatch):
    calls=[];monkeypatch.setattr(d,'_scan_output_bytes',lambda p:calls.append(p) or len(calls))
    check(roots,100.);check(roots,500.)
    result=check(roots,400.)
    assert len(calls)==2 and result['bytes']==2 and result['census_epoch']==400.
    assert not result['cache_reused']


@pytest.mark.parametrize('change',[{'new_output_bytes':9},{'added_disk_bytes':209},{'baseline_run_bytes':291}])
def test_cache_hits_apply_current_stricter_limits(roots,monkeypatch,change):
    calls=[];monkeypatch.setattr(d,'_scan_output_bytes',lambda p:calls.append(p) or 10)
    check(roots,100.)
    with pytest.raises(OSError,match='disk guard'):check(roots,101.,**change)
    assert len(calls)==1


def test_failed_admission_does_not_cache_a_decision(roots,monkeypatch):
    calls=[];monkeypatch.setattr(d,'_scan_output_bytes',lambda p:calls.append(p) or 10)
    with pytest.raises(OSError):check(roots,100.,new_output_bytes=9)
    assert check(roots,101.)['cache_reused'] and len(calls)==1


def test_root_isolation_and_canonical_alias(roots,tmp_path,monkeypatch):
    other=tmp_path/'newer';other.mkdir();calls=[]
    monkeypatch.setattr(d,'_scan_output_bytes',lambda p:calls.append(p) or 10)
    check(roots,100.)
    alias=roots[0]/'nested'/'..';(roots[0]/'nested').mkdir()
    d.check_output_disk(alias,source_root=roots[1],resources=limits(),now=101.)
    check((other,roots[1]),101.)
    assert calls==[roots[0].resolve(),other.resolve()]


@pytest.mark.parametrize('relation',['equal','new_in_old','old_in_new'])
def test_source_overlap_rejected_before_scan(roots,monkeypatch,relation):
    new,old=roots
    if relation=='equal':new=old
    elif relation=='new_in_old':new=old/'child';new.mkdir()
    else:old=new/'child';old.mkdir()
    monkeypatch.setattr(d,'_scan_output_bytes',lambda _:pytest.fail('No source census'))
    with pytest.raises(ValueError):check((new,old),100.)


def test_scan_failure_never_uses_stale_count(roots,monkeypatch):
    monkeypatch.setattr(d,'_scan_output_bytes',lambda _:10);check(roots,100.)
    def failed(_):raise PermissionError('incomplete census')
    monkeypatch.setattr(d,'_scan_output_bytes',failed)
    with pytest.raises(PermissionError):check(roots,700.)
    assert not d._CACHE


def test_threaded_first_call_performs_one_scan(roots,monkeypatch):
    calls=[];barrier=threading.Barrier(4)
    def scan(p):calls.append(p);time.sleep(.03);return 10
    monkeypatch.setattr(d,'_scan_output_bytes',scan)
    def worker(_):barrier.wait();return check(roots,100.)
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(worker,range(4)))
    assert len(calls)==1 and sum(not r['cache_reused'] for r in rows)==1


@pytest.mark.parametrize('now',[True,float('nan'),float('inf'),'100'])
def test_bad_epoch_rejected(roots,now):
    with pytest.raises(ValueError):check(roots,now)


@pytest.mark.parametrize('value',[True,-1,1.5,None])
def test_invalid_limit_rejected_even_from_cache(roots,value):
    check(roots,100.)
    with pytest.raises(ValueError):check(roots,101.,new_output_bytes=value)


def test_returned_metadata_cannot_mutate_cached_measurement(roots):
    first=check(roots,100.);first['bytes']=900;first['census_epoch']=1000
    second=check(roots,101.)
    assert second['bytes']==0 and second['census_epoch']==100.


def test_reparse_entry_is_rejected_without_following_target(roots,monkeypatch):
    class Entry:
        path='synthetic_reparse'
        def stat(self,*,follow_symlinks):
            assert follow_symlinks is False
            return SimpleNamespace(st_mode=stat.S_IFDIR,st_file_attributes=1024,st_size=0)
    class Scan:
        def __enter__(self):return iter([Entry()])
        def __exit__(self,*args):return False
    monkeypatch.setattr(d.os,'scandir',lambda _:Scan())
    with pytest.raises(OSError,match='reparse'):check(roots,100.)
    assert not d._CACHE


def test_actual_symlink_rejected_if_os_allows(roots):
    new,old=roots;(old/'payload').write_bytes(b'old')
    try:os.symlink(old/'payload',new/'alias')
    except OSError:pytest.skip('Windows symlink creation privilege unavailable')
    with pytest.raises(OSError,match='symlink'):check(roots,100.)


@pytest.fixture
def actual_guard(roots, monkeypatch):
    """Exercise installed Context methods with tiny roots and synthetic resources."""
    from partC_rev1_src.pacnhn_recovery24 import context as c
    clock=[100.];writes=[];scans=[]
    observed=dict(rss=20,available=1000,free=1000,resource_samples=0)
    class Process:
        def memory_info(self):
            observed['resource_samples']+=1
            return SimpleNamespace(rss=observed['rss'])
        def children(self,recursive=True):return []
    monkeypatch.setattr(c,'time',SimpleNamespace(time=lambda:clock[0]))
    monkeypatch.setattr(c.psutil,'Process',Process)
    monkeypatch.setattr(c.psutil,'virtual_memory',lambda:SimpleNamespace(available=observed['available']))
    monkeypatch.setattr(c.shutil,'disk_usage',lambda _:SimpleNamespace(free=observed['free']))
    monkeypatch.setattr(c,'write_json',lambda path,obj:writes.append((path,dict(obj))))
    monkeypatch.setattr(d,'_scan_output_bytes',lambda path:scans.append(path) or 10)
    def make(**changes):
        ctx=c.RecoveryContext.__new__(c.RecoveryContext)
        ctx.root,ctx.source_root=roots
        ctx.action=ctx.step_id='training';ctx.started=90.
        ctx.last_sample=ctx.last_disk_sample=0.;ctx.new_output_bytes=ctx.peak_rss=0
        ctx.minimum_available=2**63-1
        resources=dict(limits(),tree_rss_bytes=100,minimum_available_bytes=100,minimum_output_free_bytes=100)
        resources.update(changes)
        ctx.c=dict(action_deadlines={'training':5000.},hard_deadline_epoch=5000.,resources=resources)
        return ctx
    return SimpleNamespace(make=make,clock=clock,writes=writes,scans=scans,observed=observed)


def test_context_instances_reuse_census_and_keep_original_timestamp(actual_guard):
    g=actual_guard;first=g.make();first.guard()
    g.clock[0]=300.;second=g.make();second.guard()
    assert len(g.scans)==1
    assert first.last_disk_sample==second.last_disk_sample==100.
    assert second.new_output_bytes==10
    assert g.writes[-1][1]['disk_census_reused'] is True
    assert g.writes[-1][1]['last_disk_census_epoch']==100.


@pytest.mark.parametrize('change',[{'new_output_bytes':9},{'added_disk_bytes':209}])
def test_context_stricter_new_configuration_uses_current_disk_limit(actual_guard,change):
    g=actual_guard;g.make().guard();g.clock[0]=101.
    with pytest.raises(OSError,match='disk guard'):g.make(**change).guard()
    assert len(g.scans)==1


def test_context_expiry_and_rollback_do_not_hide_behind_sample_throttle(actual_guard):
    g=actual_guard;ctx=g.make();ctx.guard()
    g.clock[0]=699.;ctx.guard();assert len(g.scans)==1
    g.clock[0]=704.;ctx.guard();assert len(g.scans)==2 and ctx.last_disk_sample==704.
    g.clock[0]=703.;ctx.guard();assert len(g.scans)==3 and ctx.last_disk_sample==703.


@pytest.mark.parametrize('field,value,error',[
    ('rss',101,MemoryError),('available',99,MemoryError),('free',99,OSError)])
def test_context_cached_census_does_not_bypass_live_resources(actual_guard,field,value,error):
    g=actual_guard;ctx=g.make();ctx.guard()
    g.clock[0]=106.;g.observed[field]=value
    with pytest.raises(error):ctx.guard()
    assert len(g.scans)==1


def test_context_five_second_throttle_preserved_but_deadline_always_checked(actual_guard):
    g=actual_guard;ctx=g.make();ctx.guard();g.clock[0]=102.;ctx.guard()
    assert g.observed['resource_samples']==1 and len(g.writes)==1
    with pytest.raises(TimeoutError):ctx.guard(deadline=101.)
    assert g.observed['resource_samples']==1
