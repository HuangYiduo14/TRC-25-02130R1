"""Explicit campaign identity, source bindings and bounded resource controls."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
import uuid
from datetime import datetime, timezone
import numpy as np
import psutil
from .continuation import validate_continuation, authorized_deadline, check_cached_identity

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'partC_doc_icnn/pacnhn_v1/campaign.json'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

def json_default(v):
    if isinstance(v, (np.integer, np.floating)): return v.item()
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, Path): return str(v)
    raise TypeError(type(v).__name__)

def _atomic_replace(temporary, destination):
    """Retry Windows sharing/access denial only; keep failed source and old target.

    WinError5 can also represent a permanent ACL failure: it is only potentially
    transient, so both attempt and monotonic-time bounds always apply. No generic
    PermissionError/EACCES retry and no delete/write-in-place fallback are used.
    The retry budget bounds our waits/attempts, not time spent inside an OS call.
    """
    delays = (.01, .02, .04, .08, .16, .25, .25)
    retry_deadline = time.monotonic() + 1.
    for attempt in range(len(delays)+1):
        try:
            os.replace(temporary, destination)
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == len(delays):
                raise
            remaining = retry_deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delays[attempt], remaining))
            if time.monotonic() >= retry_deadline:
                raise


def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False, default=json_default)+'\n'
    tmp = p.with_name(p.name + '.tmp.' + str(os.getpid()) + '.' + uuid.uuid4().hex)
    with tmp.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    _atomic_replace(tmp, p)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save_npz(path, **arrays):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid())+'.'+uuid.uuid4().hex)
    with tmp.open('xb') as stream: np.savez_compressed(stream, **arrays)
    _atomic_replace(tmp,p)

def array_hash(arrays):
    h=hashlib.sha256()
    for key,value in sorted(arrays.items()):
        a=np.ascontiguousarray(value)
        if a.dtype.hasobject: raise ValueError('Object arrays prohibited')
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

class Context:
    def __init__(self, config, *, step_id):
        self.path=Path(config).resolve(strict=True); self.c=read_json(self.path)
        if self.c['phase_id']!='PACNHN_V1' or not step_id or step_id not in self.c['tasks']:
            raise ValueError('Explicit approved phase/step required')
        if not self.c['run_id'] or not self.c['input_versions'] or not self.c['expected_outputs'] or not self.c['acceptance_tests']:
            raise ValueError('Incomplete execution contract')
        self.continuation = validate_continuation(self.c, action=step_id)
        self._continuation_action = step_id
        self.step_id=step_id; self.root=Path(self.c['output_root']); self.root.mkdir(parents=True,exist_ok=True)
        self.repo=REPO; self.started=time.time(); self.last_sample=0.; self.bindings={}
        self.peak_rss=0; self.minimum_available=2**63-1
        if sha(self.c['contract_path'])!=self.c['contract_sha256']: raise ValueError('Approved contract changed')

    def require_action(self, action):
        """Select an explicitly allowed continuation action without changing step_id."""
        if self.continuation is not None:
            if not isinstance(action, str) or not action:
                raise ValueError('Continuation requires an explicit nonempty action')
            check_cached_identity(self.c, self.continuation)
            authorized_deadline(self.continuation, action=action)
            self._continuation_action = action
        return self.continuation

    def bind(self,path,role,expected=None):
        p=Path(path).resolve(strict=True); stat=p.stat(); digest=sha(p)
        if expected and digest!=expected: raise ValueError('Input identity mismatch: '+str(p))
        prior=self.bindings.get(str(p))
        if prior and prior['sha256']!=digest: raise ValueError('Source changed during task')
        self.bindings[str(p)]=dict(path=str(p),role=role,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=digest)
        return digest

    def guard(self,deadline=None):
        now=time.time()
        if self.continuation is None:
            hard = self.c['hard_deadline_epoch']
        else:
            hard = authorized_deadline(self.continuation, action=self._continuation_action, now=now)
        if now>=min(deadline or hard,hard): raise TimeoutError('Original campaign/run deadline reached')
        if now-self.last_sample<5: return
        if self.continuation is not None:
            check_cached_identity(self.c, self.continuation)
        proc=psutil.Process(); rss=proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:rss+=child.memory_info().rss
            except psutil.NoSuchProcess:pass
        available=psutil.virtual_memory().available
        self.peak_rss=max(self.peak_rss,rss); self.minimum_available=min(self.minimum_available,available)
        if rss>self.c['resources']['tree_rss_bytes']:raise MemoryError('Process tree memory guard')
        if available<self.c['resources']['minimum_available_bytes']:raise MemoryError('Available system RAM guard')
        if shutil.disk_usage(str(self.root)).free<5*1024**3:raise OSError('Output disk free-space guard')
        self.last_sample=now
        write_json(self.root/'resources'/f'{self.step_id}_{os.getpid()}.json',dict(pid=os.getpid(),time=now,peak_rss_bytes=self.peak_rss,minimum_available_bytes=self.minimum_available,elapsed_seconds=now-self.started))

    def flush_bindings(self):
        write_json(self.root/'manifests'/f'inputs_{self.step_id}_{time.time_ns()}_{os.getpid()}.json',list(self.bindings.values()))
        write_json(self.root/'manifests'/f'inputs_{self.step_id}.json',list(self.bindings.values()))

def bootstrap():
    if CONFIG.exists():
        c=read_json(CONFIG)
        if c['original_start_epoch']!=1790304598:raise ValueError('Existing campaign clock differs')
        print('Existing original campaign configuration: '+str(CONFIG));return
    start=1790304598
    out=REPO.parent/(REPO.name+'_outputs')/'PACNHN_V1/run_20260925T024958Z'
    if out.exists():raise FileExistsError('Unindexed output root already exists')
    contract=REPO/'partC_doc_icnn/pacnhn_v1/APPROVED_PLAN.md'
    tasks=['intake','topology','preflight','reference_open','reference_i95','support','theory','training','assignments','publication','handoff']
    c=dict(schema='pacnhn.campaign.v1',phase_id='PACNHN_V1',step_id='BUILD_TRAIN_COMPARE_PUBLISH',run_id='run_20260925T024958Z',
        original_start_epoch=start,hard_deadline_epoch=start+52*3600,stage_hours=[12,24,12,4],
        maximum_cumulative_stage_deadlines=[start+h*3600 for h in [12,36,48,52]],
        expected_branch='phase1/icnn-rev',initial_head='7eaa793478137ccd979c9d7c969a03deb07f2202',
        output_root=str(out),contract_path=str(contract),contract_sha256=sha(contract),
        tasks=tasks,suite_ids=['s02','s03','s04','s05','s06'],excluded_suite_ids=['s01'],
        resolutions={'s02':0.3107232505953859,'s03':1.,'s04':1.6,'s05':4.,'s06':12.},
        input_versions={'P4_physical_config':str(REPO/'partC_doc_icnn/phase_4_work/phase4_dual_accuracy_benchmark_execution_config.json'),
          'P5_frozen_runtime':str(REPO.parent/(REPO.name+'_outputs')/'phase_05/campaign_v1_0905/run_001/suites/s03'),
          'P3_suite_root':str(REPO.parent/(REPO.name+'_outputs')/'phase_03/all_suite_acnhn/phase3_all_suites_geo_gc_1000m_20260831_v1_run_001'),
          'prior_publication':str(REPO.parent/(REPO.name+'_outputs')/'phase_N8/N8R4_V5_PUBLICATION/run_20260916T191025Z')},
        expected_outputs=['common_P_graph','five_fixed_backbone_partitions','common_OD','open_closed_P_references','complete_OD_witness_audit','frozen_active_pairs','model_registries','training_coverage','14_primary_comparisons','table2','I95_analysis','theory_TeX_PDF','numerical_TeX_PDF','standalone_CSV_assets','verified_handoff'],
        acceptance_tests=['source_identity','fixed_B_OD_masks','directed_witness_projection_lifting','conservation','cost_ownership','ICNN_cross_monotonicity','XGB_cost_only_positivity','native_KPI','fresh_certificates','portable_asset_regeneration','PDF_visual_QA','handoff_integrity'],
        resources={'tree_rss_bytes':24*1024**3,'minimum_available_bytes':3*1024**3,'added_disk_bytes':250*10**9,'national_assignments':1,'regional_workers':4,'torch_threads':1,'xgboost_threads':1,'gpu_memory_bytes':8*1024**3},
        reference={'solver':'CFW','per_scenario_seconds':10800,'RG_SPTT':1e-4},
        assignment={'per_query_seconds':2700,'RG_SPTT':1e-3,'solvers':{'FFTT':'FW','POWER4':'FW','ICNN':'FW','XGBOOST':'MSA'}},
        training={'seconds':86400,'target_train_cost_wmape_percent':1.,'primary_suite':'s03','additional_suite_order':['s04','s02','s05','s06'],'families':['ICNN','XGBOOST'],'widths':[32,64,128],'initial_seed':20260925,'continue_after_target':True},
        authority={'kind':'user_approved_conversation_plan','integrated_automatic_technical_gates':True,'single_final_handoff':True,'next_campaign_authorized':False})
    git=['git','-C',str(REPO),'-c','core.longpaths=true']
    branch=subprocess.check_output(git+['branch','--show-current'],text=True).strip()
    if branch!=c['expected_branch']:raise ValueError('Unexpected branch')
    status=subprocess.check_output(git+['status','--porcelain=v1'],text=True)
    if any(line and not line.startswith('??') for line in status.splitlines()):raise ValueError('New tracked edits need inventory before bootstrapping')
    initial=[line for line in status.splitlines() if 'pacnhn_v1' not in line]
    out.mkdir(parents=True)
    write_json(CONFIG,c)
    write_json(out/'campaign.json',c)
    write_json(out/'INITIAL_STATE.json',dict(branch=branch,head=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip(),preserved_untracked=initial,original_start_epoch=start,recorded_epoch=time.time()))
    write_json(out/'STATUS.json',dict(status='EXECUTING',stage='TOPOLOGY_REFERENCE_THEORY',original_start_epoch=start,hard_deadline_epoch=c['hard_deadline_epoch'],next_campaign_authorized=False))
    write_json(out/'APPROVAL.json',dict(authority='User active Goal: Execute the approved plan above end-to-end.',contract_sha256=c['contract_sha256'],approval_applies_to='entire52hplan_with_automatic_technical_gates',reviewed_zip=None,historical_24h_and_intermediate_stop_superseded_within_scope=True))
    print(json.dumps(dict(config=str(CONFIG),output=str(out),deadline_utc=datetime.fromtimestamp(c['hard_deadline_epoch'],timezone.utc).isoformat())))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--bootstrap',action='store_true',required=True)
    parser.parse_args();bootstrap()
