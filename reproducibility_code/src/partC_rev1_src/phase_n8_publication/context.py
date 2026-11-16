"""Explicit N8R4 clock, one scientific child, source bindings and resource receipts."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import psutil
from partC_rev1_src.phase_n8.common import (read_json, write_json, sha, digest,
    ResourceLimit, code_identity as inherited_code_identity)


class Context:
    def __init__(self, config, task, run_id, *, deadline=None):
        self.config_path = Path(config).resolve(strict=True)
        self.c = read_json(self.config_path)
        c = self.c
        for key in ('phase_id','step_id','run_id','tasks','input_versions','expected_outputs','acceptance_tests'):
            if not c.get(key): raise ValueError('Incomplete execution contract: '+key)
        if (c['phase_id'],c['step_id'],c['run_id']) != ('N8','N8R4_PUBLICATION_COMPLETION',run_id):
            raise ValueError('Wrong phase/step/run')
        if c['deadline_epoch'] != c['start_epoch']+85500 or c['compute_cutoff_epoch'] != c['start_epoch']+71100:
            raise ValueError('Original clock/reserve changed')
        clock = read_json(c['original_clock_file'])
        if clock['original_start_epoch'] != c['start_epoch']: raise ValueError('Original clock mismatch')
        if set(c['active_suites']) != {'s02','s03','s04','s05','s06'}: raise ValueError('Scope mismatch')
        if task not in c['tasks']: raise ValueError('Explicit authorized task required')
        self.repo = Path(c['repository']).resolve(strict=True)
        self.root = Path(c['output_root']).resolve(strict=True)
        if self.root == self.repo or self.repo in self.root.parents: raise ValueError('External output required')
        self.prior = Path(c['prior_R2R3_root']).resolve(strict=True)
        self.r4 = Path(c['prior_R4_root']).resolve(strict=True)
        self.authority = Path(c['authority_root']).resolve(strict=True)
        self.task = self.step = task
        self.deadline = deadline
        self.started = time.time()
        self.monotonic_started = time.monotonic()
        self.bindings = {}
        self.peak_rss = 0
        self.last_resource = 0
        self.last_disk_check = 0
        self.added = 0
        self.guard()

    def guard(self, deadline=None):
        cutoff = self.c['deadline_epoch'] if self.task in {'PUBLICATION','VERIFY','HANDOFF'} else self.c['compute_cutoff_epoch']
        for d in (self.deadline,deadline):
            if d is not None: cutoff=min(cutoff,d)
        now=max(time.time(),self.started+time.monotonic()-self.monotonic_started)
        if now>=cutoff: raise TimeoutError('Original N8R4 science/task/report deadline reached')
        p=psutil.Process();rss=p.memory_info().rss
        for child in p.children(recursive=True):
            try:rss+=child.memory_info().rss
            except psutil.NoSuchProcess:pass
        self.peak_rss=max(self.peak_rss,rss)
        available=psutil.virtual_memory().available
        if rss>=self.c['max_process_tree_RSS_bytes'] or available<self.c['system_available_floor_bytes']:
            raise ResourceLimit('N8R4 process RSS/system headroom guard')
        if now-self.last_disk_check>=60:
            self.added=sum(f.stat().st_size for f in self.root.rglob('*') if f.is_file())
            self.last_disk_check=now
            if self.added>=self.c['max_added_disk_bytes']:raise ResourceLimit('N8R4 added disk guard')
        if now-self.last_resource>=15:
            cpu=p.cpu_times()
            row=dict(epoch=now,elapsed_since_R4_start=now-self.c['start_epoch'],pid=p.pid,task=self.task,
                rss_bytes=rss,system_available_bytes=available,added_disk_bytes=self.added,
                process_user_cpu_seconds=cpu.user,process_system_cpu_seconds=cpu.system,gpu_used=False)
            with (self.root/'logs/resource_usage.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            self.last_resource=now

    def bind(self,path,role,expected=None):
        p=Path(path).resolve(strict=True);key=str(p)
        if key not in self.bindings:
            st=p.stat();self.bindings[key]=dict(native_path=key,role=role,sha256=sha(p),size_bytes=st.st_size,mtime_ns=st.st_mtime_ns)
        row=self.bindings[key]
        if expected is not None and row['sha256']!=expected:raise ValueError('Source hash mismatch: '+key)
        return row['sha256']

    def finish(self,name,result):
        # Completion/resource-stop evidence must still be saved after a job deadline.
        for row in self.bindings.values():
            p=Path(row['native_path'])
            stat=p.stat()
            if (stat.st_mtime_ns!=row['mtime_ns'] or stat.st_size!=row['size_bytes']) and sha(p)!=row['sha256']:raise ValueError('Source mutated: '+str(p))
        result.update(task=self.task,started_epoch=self.started,ended_epoch=time.time(),elapsed_seconds=time.time()-self.started,
            peak_rss_bytes=self.peak_rss,source_mutations=0,original_R4_start_epoch=self.c['start_epoch'])
        write_json(self.root/'receipts'/(name+'_RESULT.json'),result)
        write_json(self.root/'receipts'/(name+'_SOURCE_BINDINGS.json'),list(self.bindings.values()))
        print(json.dumps({k:result[k] for k in ('task','status','elapsed_seconds','peak_rss_bytes','error_type','error') if k in result},allow_nan=False),flush=True)
        return result


def sources(ctx):
    """Read inherited R1 bindings using their original reference bundle, not R4 policy."""
    from partC_rev1_src.phase_n8.native import Sources
    prior_config=read_json(ctx.prior/'resolved_config.json')
    old_authority=Path(prior_config['authority_root']).resolve(strict=True)
    ctx.bind(ctx.prior/'resolved_config.json','IMMUTABLE_R2_NATIVE_REFERENCE_CONFIGURATION')
    proxy=SimpleNamespace(c=dict(ctx.c,authority_root=str(old_authority)),repo=ctx.repo,root=ctx.root,guard=ctx.guard,bind=ctx.bind)
    return Sources(proxy)


def reopen_suite(ctx,src,sid):
    from partC_rev1_src.phase_n8.native import reopen_demand
    if sid not in ctx.c['active_suites']:raise ValueError('Inactive suite')
    suite=src.suite(sid)
    original,uids,reachable,odmeta=src.original_od()
    graph_root=ctx.prior/'graphs'/sid
    meta_path=graph_root/'IDENTITY_AND_COVERAGE_SELF_OD_V2.json'
    if not meta_path.exists():meta_path=graph_root/'IDENTITY_AND_COVERAGE.json'
    meta=read_json(meta_path)
    ctx.bind(meta_path,'FROZEN_R2_GRAPH_COVERAGE')
    ctx.bind(graph_root/meta.get('mapping_file','mapping.npz'),'FROZEN_R2_GRAPH_MAPPING',meta['mapping_sha256'])
    od,balance,coverage=reopen_demand(SimpleNamespace(root=ctx.prior),suite,original)
    return suite,od,balance,coverage


def code_identity(repo):
    from partC_rev1_src.phase_n8_r4.code_dependencies import dependencies
    repo=Path(repo)
    inherited,records=inherited_code_identity(repo)
    for namespace in ('phase_n8_r4','phase_n8_publication'):
        for p in sorted((Path(repo)/'partC_rev1_src'/namespace).rglob('*.py')):
            records[p.relative_to(repo).as_posix()]=sha(p)
    roots=[repo/p for p in records]+[repo/'tests/test_phase4_preflight.py']
    records.update({p.relative_to(repo).as_posix():sha(p) for p in dependencies(repo,roots)})
    return digest(records),records,inherited
