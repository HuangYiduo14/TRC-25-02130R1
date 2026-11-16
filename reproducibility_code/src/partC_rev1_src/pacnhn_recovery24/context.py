"""Explicit recovery authority, read-only source bindings and new-output guards."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import psutil
from partC_rev1_src.pacnhn_v1.control import read_json, write_json, sha
from .disk_accounting import check_output_disk

class RecoveryContext:
    def __init__(self, config, *, step_id):
        self.path=Path(config).resolve(strict=True)
        self.config_sha256=sha(self.path)
        self.c=read_json(self.path)
        if self.c.get('phase_id')!='PACNHN_RECOVERY24' or step_id not in self.c.get('tasks',[]):
            raise ValueError('Explicit approved recovery phase and nonempty action required')
        for key in ['run_id','input_versions','expected_outputs','acceptance_tests','authority','resources']:
            if not self.c.get(key):raise ValueError('Missing recovery contract: '+key)
        if self.c['authority'].get('user_approved') is not True:
            raise PermissionError('User approval required')
        self.root=Path(self.c['output_root']).resolve(strict=True)
        self.source_root=Path(self.c['source_root']).resolve(strict=True)
        self.repo=Path(self.c['repository']).resolve(strict=True)
        if self.root==self.source_root or self.root.is_relative_to(self.source_root) or self.source_root.is_relative_to(self.root):
            raise ValueError('Recovery outputs and immutable source must be disjoint')
        contract=Path(self.c['contract_path']).resolve(strict=True)
        if sha(contract)!=self.c['contract_sha256']:raise ValueError('Approved recovery contract changed')
        inventory=Path(self.c['inventory_path']).resolve(strict=True)
        if sha(inventory)!=self.c['inventory_sha256'] or read_json(inventory)['status']!='PASS_LIGHTWEIGHT_RECOVERY_INVENTORY':
            raise ValueError('Lightweight state inventory is missing or changed')
        self.step_id=step_id;self.action=step_id;self.started=time.time();self.bindings={}
        self.source_expected={os.path.normcase(str(Path(k).resolve())):v for k,v in self.c.get('source_bindings',{}).items()}
        pin_record=self.c.get('checkpoint_metadata_pins')
        if pin_record:
            pin_path=Path(pin_record['path'])
            if sha(pin_path)!=pin_record['sha256']:raise ValueError('Checkpoint metadata pins changed')
            self.source_expected.update({os.path.normcase(str(Path(k).resolve())):v for k,v in read_json(pin_path)['bindings'].items()})
        self.last_sample=0.;self.peak_rss=0;self.minimum_available=2**63-1
        self.last_disk_sample=0.;self.new_output_bytes=0
        self.bind(self.path,'EXPLICIT_RECOVERY_CONFIGURATION',self.config_sha256)
        self.bind(contract,'USER_APPROVED_RECOVERY_POLICY',self.c['contract_sha256'])
        self.bind(inventory,'PRE_RECOVERY_LIGHTWEIGHT_INVENTORY',self.c['inventory_sha256'])

    def require_action(self, action):
        if action not in self.c['tasks']:raise PermissionError('Unapproved recovery action: '+str(action))
        if sha(self.path)!=self.config_sha256:raise ValueError('Running configuration changed; new admission required')
        self.action=action
        return {'science_deadline_epoch':self.deadline(action),'hard_deadline_epoch':self.c['hard_deadline_epoch']}

    def deadline(self, action=None):
        action=action or self.action
        end=self.c['action_deadlines'].get(action)
        if end is None:raise ValueError('No explicit budget for action '+action)
        if action=='manuscript' or action=='handoff' and (self.root/'MANUSCRIPT_CLOCK.json').exists():
            clock=self.root/'MANUSCRIPT_CLOCK.json'
            if not clock.exists():raise PermissionError('Manuscript stage clock must be explicitly opened after campaign')
            record=read_json(clock)
            return min(float(record['deadline_epoch']),float(end))
        return min(float(end),float(self.c['hard_deadline_epoch']))

    def output_path(self, relative):
        target=(self.root/relative).resolve()
        if not target.is_relative_to(self.root):raise ValueError('Write outside new recovery root')
        return target

    def bind(self,path,role,expected=None):
        p=Path(path).resolve(strict=True); before=p.stat()
        if not p.is_file():raise ValueError('A bound input must be a file')
        prior=self.bindings.get(str(p))
        if prior and prior['bytes']==before.st_size and prior['mtime_ns']==before.st_mtime_ns:
            digest=prior['sha256']
        else:
            digest=sha(p);after=p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Input changed while hashing')
        frozen=self.source_expected.get(os.path.normcase(str(p)))
        if frozen is not None and digest!=frozen:raise ValueError('Pre-recovery input changed: '+str(p))
        if expected is not None and digest!=expected:raise ValueError('Input SHA256 mismatch: '+str(p))
        if prior and prior['sha256']!=digest:raise ValueError('Previously bound input changed: '+str(p))
        self.bindings[str(p)]={'path':str(p),'role':role,'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'sha256':digest}
        return digest

    def guard(self,deadline=None):
        now=time.time();limit=self.deadline()
        if deadline is not None:limit=min(limit,float(deadline))
        if now>=limit:raise TimeoutError('Approved recovery action/campaign deadline reached')
        if 0<=now-self.last_sample<5:return
        proc=psutil.Process();rss=proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:rss+=child.memory_info().rss
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        available=psutil.virtual_memory().available
        self.peak_rss=max(self.peak_rss,rss);self.minimum_available=min(self.minimum_available,available)
        resource=self.c['resources']
        if rss>resource['tree_rss_bytes']:raise MemoryError('Recovery process-tree RAM guard')
        if available<resource['minimum_available_bytes']:raise MemoryError('Recovery available-RAM guard')
        free=shutil.disk_usage(self.root).free
        if free<resource['minimum_output_free_bytes']:raise OSError('Recovery disk free-space guard')
        # Reuse process-local measurements, never resource-admission decisions.
        # Current config limits are checked even when the census is cached.
        census=check_output_disk(self.root,source_root=self.source_root,resources=resource,now=now)
        self.new_output_bytes=census['bytes'];self.last_disk_sample=census['census_epoch']
        self.last_sample=now
        write_json(self.output_path('resources')/f'{self.step_id}_{os.getpid()}.json',{
            'pid':os.getpid(),'action':self.action,'epoch':now,'peak_rss_bytes':self.peak_rss,
            'minimum_available_bytes':self.minimum_available,'free_disk_bytes':free,
            'last_observed_new_output_bytes':self.new_output_bytes,'last_disk_census_epoch':self.last_disk_sample,
            'disk_census_reused':census['cache_reused'],'elapsed_seconds':now-self.started})

    def flush_bindings(self):
        target=self.output_path('manifests')/f'inputs_{self.step_id}_{os.getpid()}_{time.time_ns()}.json'
        write_json(target,list(self.bindings.values()))
        return target

Context=RecoveryContext
