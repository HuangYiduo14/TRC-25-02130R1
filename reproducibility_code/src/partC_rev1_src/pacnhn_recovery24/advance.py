"""Explicitly admitted sequential successors; never launched by importing this module.

The root operator must honor ADVANCE_CONTROLLER_LOCK.json: manual execute stages
are forbidden while that lock or its identified controller remains active.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys
import time

from .execute import ACTIONS, source_paths, stage_limits

SCHEMA = 'RECOVERY24_ADVANCE_ADMISSION_V1'
STAGES = ('other', 'ablation', 'selection', 'holdout', 'evaluation', 'assignments', 'metrics')
REVISION_FIELDS = {'config_revision', 'supersedes_config_sha256', 'revision_reason', 'revision_created_epoch'}
RECOVERY_REVISION_FIELDS = {'xgb_region_quotas', 'xgb_quota_manifest', 'training_overlays'}


def read(path):
    return json.loads(Path(path).read_bytes())


def binding(path):
    p = Path(path).resolve(strict=True); h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return dict(path=str(p), sha256=h.hexdigest())


def authenticate(record):
    actual = binding(record['path'])
    if actual['sha256'] != record['sha256']: raise ValueError('Changed binding: ' + actual['path'])
    return Path(actual['path'])


def new(path, record):
    encoded = json.dumps(record, indent=2, allow_nan=False)
    with Path(path).open('x', encoding='utf-8') as f: f.write(encoded)


def contained(path, root):
    p = Path(path).resolve(strict=False); root = Path(root).resolve(strict=True)
    if not p.is_relative_to(root) or p == root: raise ValueError('Path outside explicit new-output subtree')
    return p


def check_sources(admission):
    for path, digest in admission['source_bindings'].items(): authenticate(dict(path=path, sha256=digest))


def invariant_config(config):
    result = copy.deepcopy(config)
    for name in REVISION_FIELDS: result.pop(name, None)
    for name in RECOVERY_REVISION_FIELDS: result.get('recovery', {}).pop(name, None)
    return json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)


def active_config(root, anchor, *, fixed=None):
    """Follow bounded, hash-linked revision receipts; never trust just a filename."""
    root = Path(root).resolve(strict=True); anchor_path = authenticate(anchor); base = read(anchor_path)
    current = read(root / 'ACTIVE_CONFIG.json'); path = authenticate(current)
    if not path.is_relative_to(root) or path.parent != root: raise ValueError('Foreign active configuration')
    if fixed is not None and current != fixed: raise ValueError('Configuration changed after additional-suite closure')
    seen = set(); cursor = current
    for _ in range(32):
        if cursor == anchor: break
        p = authenticate(cursor); c = read(p)
        if cursor['sha256'] in seen: raise ValueError('Cyclic configuration lineage')
        seen.add(cursor['sha256'])
        if invariant_config(c) != invariant_config(base): raise ValueError('Unauthorized configuration amendment')
        revision = c.get('config_revision')
        if type(revision) is not int or revision < 1: raise ValueError('Invalid configuration revision')
        rec = read(root / 'config' / f'REVISION_{revision:03d}.json')
        if rec['new'] != cursor or rec['parent']['sha256'] != c.get('supersedes_config_sha256'):
            raise ValueError('Configuration revision receipt mismatch')
        parent = authenticate(rec['parent'])
        if parent.parent != root: raise ValueError('Foreign configuration ancestor')
        if read(parent).get('config_revision', -1) >= revision: raise ValueError('Revision order does not advance')
        cursor = rec['parent']
    else: raise ValueError('Unbounded configuration lineage')
    return current, read(path)


def validate_wrapper(folder, action, config_binding, admission, *, intent_binding=None, earliest=None):
    folder = contained(folder, Path(admission['output_root']) / 'execution')
    ip = folder / 'INTENT.json'; rp = folder / 'RESULT.json'
    if intent_binding is not None and authenticate(intent_binding) != ip: raise ValueError('Wrong execution intent')
    intent = read(ip); receipt = read(rp)
    if (intent.get('step_id') != action or intent.get('config') != config_binding
            or intent.get('phase_id') != admission['phase_id'] or intent.get('run_id') != admission['run_id']):
        raise ValueError('Execution intent identity mismatch')
    if receipt.get('action') != action or receipt.get('config') != config_binding or receipt.get('secondary_errors'):
        raise RuntimeError('Execution wrapper failed or lacks terminal disposition')
    ordinary = receipt.get('status') == 'RETURNED_WITH_EXPLICIT_OUTCOMES' and receipt.get('error') is None
    if not ordinary: validate_postdispatch_timeout(receipt, action, admission)
    start, end = receipt.get('started_epoch'), receipt.get('ended_epoch')
    if not all(type(x) in (int, float) and math.isfinite(x) for x in (start, end)) or end < start:
        raise ValueError('Invalid wrapper interval')
    if intent.get('started_epoch') != start or earliest is not None and start < earliest - 1:
        raise ValueError('Wrapper does not belong to this launch')
    if receipt.get('sources') != intent.get('sources'): raise ValueError('Wrapper source evidence changed')
    for p, digest in intent['sources'].items():
        if admission['source_bindings'].get(p) != digest: raise ValueError('Unadmitted wrapper source')
    expected = {str(p.resolve()) for p in source_paths(action)}
    if set(intent['sources']) != expected: raise ValueError('Wrapper source closure incomplete')
    result=binding(rp)
    result['administrative_disposition']='ORDINARY_WRAPPER_RETURN' if ordinary else 'VERIFIED_TERMINAL_ACCOUNTING_AFTER_WORK_DEADLINE'
    return result


def validate_postdispatch_timeout(receipt, action, admission):
    """Only the exact source-bound post-return guard may receive metadata grace."""
    from . import execute
    source=Path(execute.__file__).resolve(); expected=admission['source_bindings'].get(str(source))
    if expected is None or binding(source)['sha256']!=expected: raise RuntimeError('No admitted wrapper source for timeout proof')
    tree=ast.parse(source.read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute_stage')
    guards=[n.lineno for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and isinstance(n.func.value,ast.Name) and n.func.value.id=='ctx' and n.func.attr=='guard']
    line=max(guards)  # Current admitted wrapper's last ctx.guard is immediately after dispatch/action check.
    error=receipt.get('error') or {};samples=receipt.get('resource_samples') or {}
    frames=re.findall(r'File "([^"]+)", line (\d+), in ([^\r\n]+)',error.get('traceback',''))
    proven=any(Path(p).resolve()==source and int(n)==line and name=='execute_stage' for p,n,name in frames)
    end=receipt.get('limits',{}).get('stage_deadline_epoch');finished=receipt.get('ended_epoch')
    if (receipt.get('status')!='FAILED_PRESERVED_OUTPUTS' or error.get('type')!='TimeoutError'
            or error.get('message')!='Approved recovery action/campaign deadline reached' or not proven
            or receipt.get('secondary_errors') or receipt.get('action_at_return')!=ACTIONS[action]
            or type(end) not in (int,float) or type(finished) not in (int,float)
            or not end<=finished<=min(end+60.,admission['hard_deadline_epoch'])
            or samples.get('monitor_joined') is not True or samples.get('samples',0)<1
            or samples.get('child_access_denied',0)!=0 or samples.get('sampled_rss_limit_exceeded') is not False
            or samples.get('sampled_available_floor_breached') is not False):
        raise RuntimeError('Wrapper failure is not proven post-dispatch metadata-only timeout')
    if action not in {'ablation','assignments'} and not (receipt.get('result_status') or receipt.get('nested_result_statuses')):
        raise RuntimeError('No recorded producer-return disposition')


def terminal_metadata(root, action, config_binding):
    """Check compact producer accounting/freeze metadata; never open scientific arrays."""
    root = Path(root); evidence = []
    def get(relative, statuses):
        p = root / relative; row = read(p)
        if row.get('status') not in statuses: raise ValueError('Nonterminal prerequisite: ' + relative)
        evidence.append(binding(p)); return row
    def key(row):
        if row.get('region_key') or row.get('job_key'):return row.get('region_key') or row['job_key']
        item=row.get('job',row);r=item.get('record',item)
        return f"{r['suite_id']}:{r['region_index']:06d}"
    if action == 'primary':
        r = get('training/PRIMARY_RESULT.json', {'POLICY_EXHAUSTED_OR_TARGETS_MET', 'PRIMARY_TIME_LIMIT'})
        if r.get('TEST_used') is not False or r.get('config') != config_binding: raise ValueError('Primary terminal identity')
    elif action == 'other':
        r = get('training/OTHER_SUITES_RESULT.json', {'ALL_FOUR_SUITE_ALLOCATIONS_ACCOUNTED'})
        if (r.get('TEST_used') is not False or r.get('config') != config_binding
                or [s['suite_id'] for s in r.get('suites', [])] != ['s04', 's02', 's05', 's06']):
            raise ValueError('Additional-suite accounting/configuration mismatch')
        frozen=read(root/'holdout/INPUT_FREEZE.json');evidence.append(binding(root/'holdout/INPUT_FREEZE.json'))
        for suite in r['suites']:
            expected=[key(x) for x in frozen['rows'] if x['record']['suite_id']==suite['suite_id']]
            actual=[key(x) for x in suite['results']]
            if len(expected)!=32 or len(actual)!=32 or len(set(actual))!=32 or set(actual)!=set(expected):
                raise ValueError('Unaccounted additional-suite input regions')
    elif action == 'ablation':
        r = get('ablation/FIT_FREEZE.json', {'ALL_160_DISPOSITIONS_FROZEN'})
        if r.get('total') != 160 or len(r.get('outcomes', [])) != 160 or r.get('TEST_used') is not False:
            raise ValueError('Incomplete ablation disposition freeze')
        keys = [x['task']['task_id'] for x in r['outcomes']]
        if len(set(keys)) != 160 or any(x.get('status') in (None, 'UNATTEMPTED') for x in r['outcomes']):
            raise ValueError('Ambiguous/nonterminal ablation tasks')
        plan_path=root/'ablation/PLAN.json';plan=read(plan_path);evidence.append(binding(plan_path))
        if r.get('plan_sha256')!=binding(plan_path)['sha256'] or {x['task_id'] for x in plan['tasks']}!=set(keys):
            raise ValueError('Ablation dispositions differ from frozen plan')
    elif action in {'selection', 'holdout', 'evaluation', 'assignments', 'metrics'}:
        f = get('training/MODEL_SELECTION_FREEZE.json', {'FROZEN'})
        if (f.get('config_sha256') != config_binding['sha256'] or Path(f['config_path']).resolve() != Path(config_binding['path']).resolve()
                or f.get('test_targets_opened') is not False): raise ValueError('Global selection/configuration mismatch')
        for binding_key, rel in [('holdout_input_freeze', 'holdout/INPUT_FREEZE.json'), ('ablation_fit_freeze', 'ablation/FIT_FREEZE.json'),
                         ('portfolio', 'portfolios/PORTFOLIOS.json'), ('model_records', 'training/NORMALIZED_RECORDS.json')]:
            if authenticate(f[binding_key]) != (root / rel).resolve(): raise ValueError('Foreign freeze dependency')
            evidence.append(f[binding_key])
        evidence.extend(terminal_metadata(root, 'ablation', config_binding))
        disposition_path=authenticate(f['region_dispositions'])
        if not disposition_path.is_relative_to(root/'selection'):raise ValueError('Foreign selection accounting')
        selection_path=disposition_path.parent/'RESULT.json';selection=read(selection_path)
        if (selection.get('status')!='FROZEN_WITH_EXPLICIT_FALLBACK_PORTFOLIOS'
                or selection.get('selection')!=binding(root/'training/MODEL_SELECTION_FREEZE.json')):
            raise ValueError('Selection terminal receipt does not bind actual freeze')
        evidence.extend([f['region_dispositions'],binding(selection_path)])
        if action != 'selection':
            opened = read(root / 'holdout/TEST_OPENED.json'); evidence.append(binding(root / 'holdout/TEST_OPENED.json'))
            freeze_sha = binding(root / 'training/MODEL_SELECTION_FREEZE.json')['sha256']
            if (opened.get('selection_freeze_sha256') != freeze_sha or opened.get('epoch', 0) < f['created_epoch']):
                raise ValueError('TEST opening chronology/binding mismatch')
            h = get('holdout/RESULT.json', {'HOLDOUT_STAGE_FINISHED'})
            if h.get('selection_freeze_sha256') != freeze_sha or h.get('targets_used_for_selection') is not False:
                raise ValueError('Holdout terminal selection binding mismatch')
            inputs=read(f['holdout_input_freeze']['path'])
            expected=[key(x) for x in inputs['rows']];actual=[key(x) for x in h['results']]
            if len(actual)!=len(expected) or set(actual)!=set(expected) or len(set(actual))!=len(actual) or h.get('requested_inputs')!=inputs['requested_inputs']:
                raise ValueError('Unaccounted holdout requests/regions')
        if action in {'evaluation', 'assignments', 'metrics'}:
            regional=get('evaluation/regional_v1/RESULT.json', {'COMPLETE_ACCOUNTING_WITH_SCOPE_LIMITATIONS', 'TIME_LIMIT_COMPLETE_DISPOSITION_ACCOUNTING'})
            if regional.get('regions')!=10442 or regional.get('row_count')!=10442*3*4*3:raise ValueError('Regional evaluation ledger incomplete')
            a = get('ablation/evaluation/RESULT.json', {'COMPLETE_WITH_EXPLICIT_LABEL_COVERAGE'})
            if a.get('fit_cases') != 160 or a.get('TEST_used_for_selection') is not False: raise ValueError('Ablation evaluation incomplete')
        if action in {'assignments', 'metrics'}:
            a = get('ASSIGNMENT_STAGE_RESULT.json', {'COMPLETE_WITH_RECORDED_OUTCOMES'})
            if a.get('rows') != 46 or a.get('reporting_rows') != 48: raise ValueError('Assignment matrix incomplete')
            cases=read(root/'ASSIGNMENT_RESULTS.json');reporting=read(root/'REPORTING_RESULTS.json')
            matrix=read(root/'ASSIGNMENT_MATRIX.json');expected={x['case_id'] for x in matrix}
            ids=[x['case']['case_id'] for x in cases];rids=[x['case']['case_id'] for x in reporting]
            if len(ids)!=46 or len(set(ids))!=46 or len(rids)!=48 or len(set(rids))!=48 or not set(ids)<set(rids):raise ValueError('Assignment rows are missing or duplicated')
            if len(matrix)!=46 or expected!=set(ids) or set(rids)-set(ids)!={'P_REFERENCE_open','P_REFERENCE_i95'}:raise ValueError('Assignment matrix identities mismatch')
            if any(not x.get('status') for x in cases+reporting):raise ValueError('Assignment disposition missing')
            evidence.extend([binding(root/'ASSIGNMENT_MATRIX.json'),binding(root/'ASSIGNMENT_RESULTS.json'),binding(root / 'REPORTING_RESULTS.json')])
        if action == 'metrics': get('report_data/MANIFEST.json', {'VERIFIED_REPORT_DATA'})
    else: raise ValueError('Unknown terminal action')
    return evidence


def check_cleanup_chronology(root,action,wrapper,launch_epoch):
    if wrapper['administrative_disposition']!='VERIFIED_TERMINAL_ACCOUNTING_AFTER_WORK_DEADLINE':return
    canonical={'primary':'training/PRIMARY_RESULT.json','other':'training/OTHER_SUITES_RESULT.json',
        'ablation':'ablation/FIT_FREEZE.json','selection':'training/MODEL_SELECTION_FREEZE.json',
        'holdout':'holdout/RESULT.json','evaluation':'evaluation/regional_v1/RESULT.json',
        'assignments':'ASSIGNMENT_STAGE_RESULT.json','metrics':'report_data/MANIFEST.json'}
    p=Path(root)/canonical[action];r=read(p);receipt=read(wrapper['path'])
    if not launch_epoch-1<=p.stat().st_mtime<=receipt['ended_epoch']+1:raise ValueError('Terminal accounting lacks current-stage chronology')
    if action=='ablation' and not launch_epoch<=r.get('frozen_epoch',0)<=receipt['ended_epoch']:raise ValueError('Ablation freeze predates launch')
    if action=='assignments' and not launch_epoch<=r.get('started_epoch',0)<=receipt['ended_epoch']:raise ValueError('Assignment accounting predates launch')
    if action in {'ablation','assignments'}:
        if receipt.get('result_status') is not None or receipt.get('nested_result_statuses'):
            raise ValueError('Statusless producer return metadata differs from exact wrapper schema')
    elif action=='evaluation':
        expected={'regional_evaluation':r['status'],'ablation_evaluation':read(Path(root)/'ablation/evaluation/RESULT.json')['status']}
        if receipt.get('result_status') is not None or receipt.get('nested_result_statuses')!=expected:
            raise ValueError('Wrapper nested producer dispositions mismatch')
    else:
        expected='FROZEN_WITH_EXPLICIT_FALLBACK_PORTFOLIOS' if action=='selection' else r['status']
        if receipt.get('result_status')!=expected:raise ValueError('Wrapper producer disposition differs from native terminal metadata')


class Runtime:
    """All operating-system process control is restricted to owned children."""
    def __init__(self):
        import psutil
        self.psutil = psutil
    def now(self): return time.time()
    def sleep(self, seconds): time.sleep(seconds)
    def describe(self, pid):
        try:
            p = self.psutil.Process(pid)
            return dict(pid=pid, create_time=p.create_time(), command=p.cmdline(), cwd=p.cwd())
        except self.psutil.NoSuchProcess: return None
    def stages(self, root):
        result = []
        for p in self.psutil.process_iter(['pid', 'cmdline', 'create_time', 'cwd']):
            try:
                info = p.info; args = info.get('cmdline') or []
                if 'partC_rev1_src.pacnhn_recovery24.execute' not in args: continue
                config = args[args.index('--config') + 1] if '--config' in args else next((x.split('=', 1)[1] for x in args if x.startswith('--config=')), None)
                if config and (Path(info['cwd']) / config).resolve().parent == Path(root).resolve():
                    result.append(dict(pid=p.pid, create_time=info['create_time'], command=args))
            except self.psutil.NoSuchProcess: pass
        return result
    def children(self, pid):
        try:
            result=[]
            for p in self.psutil.Process(pid).children(recursive=True):
                item=self.describe(p.pid)
                if item is not None:result.append(item)
            return result
        except self.psutil.NoSuchProcess:return []
    def guard_metadata(self,ctx):
        from .disk_accounting import check_output_disk
        p=self.psutil.Process();members=[p]+p.children(recursive=True);rss=0
        for member in members:
            try:rss+=member.memory_info().rss
            except self.psutil.NoSuchProcess:pass
        resource=ctx.c['resources']
        if rss>resource['tree_rss_bytes'] or self.psutil.virtual_memory().available<resource['minimum_available_bytes']:
            raise MemoryError('Successor metadata cleanup resource limit')
        if shutil.disk_usage(ctx.root).free<resource['minimum_output_free_bytes']:raise OSError('Metadata cleanup free disk limit')
        check_output_disk(ctx.root,source_root=ctx.source_root,resources=resource,now=self.now())
    def launch(self, command, cwd, log):
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        return subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    def terminate_owned(self, process, identity):
        current = self.describe(process.pid)
        if current is None: return
        if current['create_time'] != identity['create_time']: raise RuntimeError('Owned child PID was reused; no termination')
        owner = self.psutil.Process(process.pid); children = owner.children(recursive=True)
        for p in reversed(children + [owner]):
            try: p.terminate()
            except self.psutil.NoSuchProcess: pass
        _, alive = self.psutil.wait_procs(children + [owner], timeout=5)
        for p in alive:
            try: p.kill()
            except self.psutil.NoSuchProcess: pass
        _, survivors = self.psutil.wait_procs(alive, timeout=5)
        return not survivors


def identity_running(runtime, expected):
    actual = runtime.describe(expected['pid'])
    if actual is None: return False
    if (actual['create_time'] != expected['create_time'] or actual['command'] != expected['command']
            or 'cwd' in expected and Path(actual.get('cwd','')).resolve()!=Path(expected['cwd']).resolve()):
        raise RuntimeError('PID identity/command differs from admitted process')
    return True


def validate_admission(path, digest):
    p = authenticate(dict(path=str(path), sha256=digest)); a = read(p)
    if (a.get('schema') != SCHEMA or a.get('phase_id') != 'PACNHN_RECOVERY24'
            or a.get('stages') != list(STAGES) or a.get('root_honors_exclusive_controller_lock') is not True):
        raise ValueError('Explicit complete successor admission and root lock agreement required')
    if a.get('helper_sha256') != binding(__file__)['sha256']: raise ValueError('Unreviewed controller helper')
    check_sources(a)
    required = {str(p.resolve()) for action in ('primary', *STAGES) for p in source_paths(action)} | {str(Path(__file__).resolve())}
    if not required.issubset(a['source_bindings']): raise ValueError('Incomplete successor source inventory')
    config = read(authenticate(a['initial_config'])); root = Path(config['output_root']).resolve(strict=True)
    if root != Path(a['output_root']).resolve(strict=True) or config['run_id'] != a['run_id']: raise ValueError('Foreign campaign')
    if a['hard_deadline_epoch'] != config['hard_deadline_epoch']: raise ValueError('Controller cannot extend original clock')
    command = a['primary_process']['command']
    if (Path(a['repository']).resolve()!=Path(config['repository']).resolve()
            or Path(a['primary_process']['cwd']).resolve()!=Path(a['repository']).resolve()
            or Path(a['python_path']).resolve()!=Path(command[0]).resolve()):
        raise ValueError('Repository/interpreter differs from admitted primary campaign')
    if ('partC_rev1_src.pacnhn_recovery24.execute' not in command or '--action' not in command
            or command[command.index('--action')+1] != 'primary' or '--config' not in command
            or Path(command[command.index('--config')+1]).resolve() != Path(a['initial_config']['path']).resolve()):
        raise ValueError('Admitted primary command/config is not exact')
    ip = authenticate(a['primary_intent']); intent = read(ip)
    if intent.get('step_id') != 'primary' or intent.get('config') != a['initial_config']: raise ValueError('Primary intent identity')
    if ip.parent.parent != root/'execution': raise ValueError('Foreign primary execution directory')
    return a


def run(admission, *, runtime=None, context_factory=None):
    """No automatic restart: any unexpected exception preserves one stopped receipt."""
    runtime = runtime or Runtime()
    if context_factory is None:
        from .context import RecoveryContext
        context_factory = RecoveryContext
    a = admission; root = Path(a['output_root']); lock = root/'ADVANCE_CONTROLLER_LOCK.json'
    folder = contained(root/a['output_relative'], root/'controller'); folder.mkdir(parents=True, exist_ok=False)
    own = runtime.describe(os.getpid())
    token = dict(pid=os.getpid(),create_time=own['create_time'],command=own['command'],run_id=a['run_id'],created_epoch=runtime.now(),output=str(folder))
    new(lock, token)
    steps = []; child = None; child_identity = None; error = None; owned_closed = True; descendant_history={}
    def journal(status, **values):
        new(folder/f'STATUS_{len(list(folder.glob("STATUS_*.json"))):03d}.json', dict(status=status,epoch=runtime.now(),**values))
    def guard(ctx, end):
        if runtime.now() >= min(end, a['hard_deadline_epoch']): raise TimeoutError('Original successor/action deadline exhausted')
        ctx.guard(deadline=end)
    try:
        new(folder/'INTENT.json',dict(admission=a,controller=token,scientific_success_claim=False,clock_reset=False))
        ctx = context_factory(a['initial_config']['path'], step_id='training')
        journal('WAITING_FOR_EXACT_PRIMARY',primary=a['primary_process'])
        observed_children={}
        while identity_running(runtime, a['primary_process']):
            for child_record in runtime.children(a['primary_process']['pid']):
                observed_children[(child_record['pid'],child_record['create_time'])]=child_record
            guard(ctx,ctx.deadline()); runtime.sleep(2)
        if any(identity_running(runtime,p) for p in observed_children.values()):raise RuntimeError('Observed primary descendant remains live')
        if runtime.stages(root): raise RuntimeError('Another recovery stage/descendant is still live')
        check_sources(a)
        primary_wrapper = validate_wrapper(Path(a['primary_intent']['path']).parent,'primary',a['initial_config'],a,intent_binding=a['primary_intent'])
        current, _ = active_config(root,a['initial_config'])
        steps.append(dict(action='primary',wrapper=primary_wrapper,terminal=terminal_metadata(root,'primary',current),launched_by_controller=False))
        check_cleanup_chronology(root,'primary',primary_wrapper,read(a['primary_intent']['path'])['started_epoch'])
        fixed = None
        for action in STAGES:
            for completed in steps:
                authenticate(completed['wrapper'])
                for record in completed['terminal']: authenticate(record)
            current, cfg = active_config(root,a['initial_config'],fixed=fixed)
            ctx = context_factory(current['path'], step_id=ACTIONS[action]);check_sources(a)
            limits = stage_limits(action,ctx.deadline(),runtime.now(),cfg);end=limits['stage_deadline_epoch'];guard(ctx,end)
            if runtime.stages(root): raise RuntimeError('Another execute stage is live; no overlapping launch')
            before = {str(p.resolve()) for p in (root/'execution').iterdir() if p.is_dir()}
            command = [a['python_path'],'-B','-m','partC_rev1_src.pacnhn_recovery24.execute','--config',current['path'],'--action',action]
            started=runtime.now();journal('LAUNCHING',action=action,config=current,command=command,deadline_epoch=end)
            with (folder/(action+'.log')).open('x',encoding='utf-8') as log:
                child=runtime.launch(command,a['repository'],log);child_identity=runtime.describe(child.pid)
                if child_identity is None: raise RuntimeError('Child exited before identity capture; preserve for diagnosis')
                journal('CHILD_IDENTIFIED',action=action,child=child_identity)
                descendant_history={}
                cleanup_announced=False
                while child.poll() is None:
                    for descendant in runtime.children(child.pid):
                        descendant_history[(descendant['pid'],descendant['create_time'])]=descendant
                    if runtime.now()<end:guard(ctx,end)
                    else:
                        if runtime.now()>=min(end+60.,a['hard_deadline_epoch']):raise TimeoutError('Original metadata cleanup grace exhausted')
                        runtime.guard_metadata(ctx)
                        if not cleanup_announced:journal('BOUNDED_METADATA_CLEANUP_ONLY',action=action,work_deadline=end,cleanup_deadline=min(end+60.,a['hard_deadline_epoch']))
                        cleanup_announced=True
                    if not identity_running(runtime,child_identity): raise RuntimeError('Live child handle lost its identity')
                    runtime.sleep(2)
                code=child.wait();child=None;child_identity=None
            live_descendants=[p for p in descendant_history.values() if identity_running(runtime,p)]
            if live_descendants:
                owned_closed=False;raise RuntimeError('Observed successor descendants remain live: '+str(live_descendants))
            if runtime.stages(root): raise RuntimeError('Child left an overlapping execute stage')
            check_sources(a)
            after = [p for p in (root/'execution').iterdir() if p.is_dir() and str(p.resolve()) not in before]
            if len(after)!=1: raise ValueError('Exactly one new execution receipt directory required')
            wrapper=validate_wrapper(after[0],action,current,a,earliest=started)
            if code!=0 and wrapper['administrative_disposition']!='VERIFIED_TERMINAL_ACCOUNTING_AFTER_WORK_DEADLINE':
                raise RuntimeError(f'{action} wrapper exited {code}; no automatic retry')
            next_config,_=active_config(root,a['initial_config'],fixed=fixed)
            if action!='other' and next_config!=current: raise ValueError('Unexpected stage configuration revision')
            if action=='other': fixed=next_config
            terminal=terminal_metadata(root,action,next_config)
            check_cleanup_chronology(root,action,wrapper,started)
            steps.append(dict(action=action,wrapper=wrapper,terminal=terminal,config=next_config,exit_code=code,launched_by_controller=True))
            journal('STAGE_ACCOUNTED',action=action,wrapper=wrapper,config=next_config)
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc),epoch=runtime.now())
        if child is not None and child.poll() is None:
            try:
                if child_identity is None: raise RuntimeError('No authenticated child identity; root intervention required')
                owned_closed=runtime.terminate_owned(child,child_identity) is True and child.poll() is not None
            except Exception as cleanup:
                owned_closed=False;error['cleanup_error']=dict(type=type(cleanup).__name__,message=str(cleanup))
        try:
            if any(identity_running(runtime,p) for p in descendant_history.values()):owned_closed=False
        except Exception as cleanup:
            owned_closed=False;error['descendant_check_error']=str(cleanup)
    result=dict(status='ALL_SUCCESSOR_STAGES_ACCOUNTED' if error is None else 'STOPPED_PRESERVED_EVIDENCE',
                steps=steps,error=error,epoch=runtime.now(),controller=token,owned_child_closed=owned_closed,
                scientific_success_claim=False,clock_reset=False,manual_restart_allowed=False)
    new(folder/'RESULT.json',result)
    # A failed cleanup deliberately leaves the visible lock for root diagnosis.
    if owned_closed and read(lock)==token: lock.rename(folder/'CONTROLLER_LOCK_CLOSED.json')
    elif owned_closed: raise RuntimeError('Controller lock changed; do not remove another owner lock')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--admission',required=True);parser.add_argument('--admission-sha256',required=True)
    args=parser.parse_args(argv);a=validate_admission(args.admission,args.admission_sha256)
    result=run(a);print(json.dumps(dict(status=result['status'],steps=len(result['steps']),error=result['error'])))
    if result['error']: raise SystemExit(1)


if __name__=='__main__': main()
