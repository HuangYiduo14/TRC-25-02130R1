"""One explicit stage, with source identity, bounded clocks and sampled resources."""
from __future__ import annotations
import argparse, copy, hashlib, importlib, json, math, os, sys, threading, time, traceback
from pathlib import Path

ACTIONS={'primary':'training','other':'training','ablation':'training','selection':'portfolio',
         'holdout':'holdout','evaluation':'holdout','assignments':'assignments','metrics':'publication'}
MODULES={'primary':('campaign',),'other':('campaign',),
         'ablation':('ablation','training_recovery','labels'),
         'selection':('selection','numerical_admission','portfolio','labels','ablation'),
         'holdout':('labels','regional_solver'),
         'evaluation':('regional_evaluation','training_recovery','ablation','labels'),
         'assignments':('assignments','portfolio'),'metrics':('metrics_publication_data',)}


def binding(path):
    path=Path(path).resolve(strict=True);h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return dict(path=str(path),sha256=h.hexdigest())


def configure_environment(config):
    # Environment affects children; the interpreter flag affects this process.
    os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    os.environ['NUMBA_CACHE_DIR']=str(Path(config['output_root'])/'cache')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'


def stage_limits(action, action_deadline, now, config):
    """Original reserves, calculated once; no action or campaign clock reset."""
    if action not in ACTIONS or not all(math.isfinite(float(v)) for v in (action_deadline,now)):
        raise ValueError('Explicit action and finite original deadline required')
    end=float(action_deadline)
    result=dict(action_deadline_epoch=end,stage_deadline_epoch=end)
    if action=='ablation':
        ablation_end=float(config['ablation']['deadline_epoch'])
        if not math.isfinite(ablation_end):raise ValueError('Nonfinite ablation deadline')
        end=min(end,ablation_end)
    elif action=='selection':end=min(end-14400.,float(now)+1800.)
    elif action=='holdout':end-=3600.
    elif action=='evaluation':
        result.update(regional_evaluation_deadline_epoch=min(end-600.,float(now)+3600.),
                      ablation_evaluation_deadline_epoch=end)
    result['stage_deadline_epoch']=end
    if not all(math.isfinite(v) for v in result.values()):raise ValueError('Nonfinite stage deadline')
    return result


def source_paths(action):
    root=Path(__file__).resolve().parent
    return [root/(name+'.py') for name in ('execute','context','disk_accounting',*MODULES[action])]


class ResourceSampler:
    """Sampled extrema; Context/stage guards remain the active runtime guards."""
    def __init__(self,psutil_module,limits,interval=.5):
        if not math.isfinite(interval) or interval<=0:raise ValueError('Positive sample interval required')
        self.psutil,self.limits,self.interval=psutil_module,limits,interval
        self.stop_event=threading.Event();self.lock=threading.Lock();self.thread=None
        self.data=dict(peak_process_tree_rss_bytes=0,minimum_available_bytes=None,samples=0,
                       sample_errors=0,child_access_denied=0,first_sample_epoch=None,last_sample_epoch=None)

    def sample(self):
        p=self.psutil
        try:
            proc=p.Process();rss=proc.memory_info().rss;denied=0
            try:children=proc.children(recursive=True)
            except p.AccessDenied:children=[];denied+=1
            for child in children:
                try:rss+=child.memory_info().rss
                except p.NoSuchProcess:pass
                except p.AccessDenied:denied+=1
            available=p.virtual_memory().available;epoch=time.time()
            with self.lock:
                d=self.data;d['peak_process_tree_rss_bytes']=max(d['peak_process_tree_rss_bytes'],rss)
                old=d['minimum_available_bytes']
                d['minimum_available_bytes']=available if old is None else min(old,available)
                d['samples']+=1;d['child_access_denied']+=denied
                if d['first_sample_epoch'] is None:d['first_sample_epoch']=epoch
                d['last_sample_epoch']=epoch
        except (p.NoSuchProcess,p.AccessDenied):
            with self.lock:self.data['sample_errors']+=1

    def _loop(self):
        while not self.stop_event.wait(self.interval):self.sample()

    def start(self):
        self.sample();self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()

    def snapshot(self):
        with self.lock:result=copy.deepcopy(self.data)
        result['sampled_rss_limit_exceeded']=result['peak_process_tree_rss_bytes']>self.limits['tree_rss_bytes']
        result['sampled_available_floor_breached']=(result['minimum_available_bytes'] is not None
            and result['minimum_available_bytes']<self.limits['minimum_available_bytes'])
        return result

    def finish(self):
        self.stop_event.set()
        if self.thread is not None:self.thread.join(timeout=2.)
        self.sample();result=self.snapshot()
        result['monitor_joined']=self.thread is None or not self.thread.is_alive()
        return result

    def check(self):
        d=self.snapshot()
        if not d['samples'] or d['child_access_denied']:
            raise RuntimeError('Wrapper resource sampling cannot establish owned process-tree coverage')
        if d['sampled_rss_limit_exceeded'] or d['sampled_available_floor_breached']:
            raise MemoryError('Wrapper observed process-tree RSS or available-RAM breach')


def _module(name):return importlib.import_module('.'+name,__package__)


def dispatch(ctx,action,limits,folder,new_json):
    """Public stage APIs and their numerical guards remain unchanged."""
    if action!='assignments' and (ctx.root/'NATIONAL_TIMING_ACTIVE.json').exists():
        raise PermissionError('National timing active; no concurrent regional/reporting stage')
    end=limits['stage_deadline_epoch']
    if action=='primary':return _module('campaign').run_primary(ctx)
    if action=='other':return _module('campaign').run_other_suites(ctx)
    if action=='ablation':
        module=_module('ablation')
        if not (ctx.root/'ablation/PLAN.json').exists():module.prepare(ctx)
        return module.run(ctx,deadline=end)
    if action=='selection':return _module('selection').run(ctx,deadline=end)
    if action in ('holdout','evaluation'):
        freeze=binding(ctx.root/'training/MODEL_SELECTION_FREEZE.json')
        if action=='holdout':
            return _module('labels').run_holdouts(ctx,freeze_path=freeze['path'],freeze_sha256=freeze['sha256'],
                                                deadline=end,max_workers=4)
        result=_module('regional_evaluation').run(ctx,freeze_path=freeze['path'],freeze_sha256=freeze['sha256'],
            deadline=limits['regional_evaluation_deadline_epoch'],national_timing_active=False,max_workers=4)
        ctx.require_action('holdout');ctx.guard(deadline=limits['ablation_evaluation_deadline_epoch'])
        return dict(regional_evaluation=result,ablation_evaluation=_module('ablation').evaluate(ctx,freeze))
    if action=='assignments':
        active=ctx.root/'NATIONAL_TIMING_ACTIVE.json'
        lock=dict(pid=os.getpid(),start_epoch=time.time(),configuration=ctx.config_sha256,
                  phase_id=ctx.c['phase_id'],run_id=ctx.c['run_id'])
        new_json(active,lock)  # Existing/foreign locks are never overwritten or moved.
        succeeded=False
        try:
            result=_module('assignments').run(ctx,ctx.root/'portfolios/PORTFOLIOS.json',deadline=end)
            succeeded=True;return result
        finally:
            if json.loads(active.read_text(encoding='utf-8'))!=lock:
                raise ValueError('Owned national timing lock changed; preserving it in place')
            active.rename(folder/('NATIONAL_TIMING_COMPLETED.json' if succeeded else 'NATIONAL_TIMING_FAILED.json'))
    if action=='metrics':
        return _module('metrics_publication_data').evaluate_assignments(ctx,ctx.root/'REPORTING_RESULTS.json',deadline=end)
    raise ValueError('Unknown explicit stage')


def _error(exc):return dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())


def execute_stage(ctx,action,*,new_json,psutil_module,startup=None):
    """Wrapper disposition is not a scientific-convergence assertion."""
    preparation_epoch=time.time();preparation_tick=time.perf_counter()
    if action not in ACTIONS:raise ValueError('Unknown explicit stage')
    ctx.require_action(ACTIONS[action]);ctx.guard()
    c=ctx.c
    config_binding=dict(path=str(ctx.path),sha256=ctx.config_sha256)
    sources={str(p):ctx.bind(p,'EXPLICIT_STAGE_WRAPPER_SOURCE') for p in source_paths(action)}
    start=time.time();tick=time.perf_counter();limits=stage_limits(action,ctx.deadline(),start,c)
    preparation=dict(started_epoch=preparation_epoch,ended_epoch=start,elapsed_seconds=tick-preparation_tick,
                     scope='Wrapper action/guard/source authentication before stage interval')
    folder=ctx.root/'execution'/f'{action}_{time.time_ns()}';folder.mkdir(parents=True,exist_ok=False)
    sampler=ResourceSampler(psutil_module,c['resources'])
    result=None;error=None;secondary_errors=[];samples=None;manifest=None;observed_action=ctx.action
    try:
        new_json(folder/'INTENT.json',dict(phase_id=c['phase_id'],run_id=c['run_id'],step_id=action,
            config=config_binding,started_epoch=start,deadline_epoch=limits['stage_deadline_epoch'],
            limits=limits,tasks=[action],resources=c['resources'],expected_outputs=c['expected_outputs'],
            acceptance_tests=c['acceptance_tests'],startup=startup,preparation=preparation,
            timing_boundary='After Context/shared-import setup and source authentication; includes wrapper intent, sampling setup, stage imports/action through return/exception. Excludes monitor shutdown and terminal receipt.',
            accounting_role='STAGE_WALL_PARENT_OF_NESTED_WORKER_AND_QUERY_TIMES',sources=sources))
        sampler.start();sampler.check();ctx.guard(deadline=limits['stage_deadline_epoch'])
        result=dispatch(ctx,action,limits,folder,new_json)
        observed_action=ctx.action
        if observed_action!=ACTIONS[action]:raise ValueError('Stage returned under a different action/deadline')
        ctx.guard(deadline=limits['stage_deadline_epoch'])
    except Exception as exc:error=_error(exc)
    finally:
        ended=time.time();elapsed=time.perf_counter()-tick
        try:
            samples=sampler.finish();sampler.check()
            if not samples['monitor_joined']:raise RuntimeError('Resource monitor did not join')
        except Exception as exc:
            if error is None:error=_error(exc)
            else:secondary_errors.append(_error(exc))
        try:
            if binding(ctx.path)['sha256']!=ctx.config_sha256:raise ValueError('Running configuration changed')
            ctx.bind(ctx.path,'UNCHANGED_WRAPPER_CONFIGURATION',ctx.config_sha256)
            for path,digest in sources.items():
                if binding(path)['sha256']!=digest:raise ValueError('Action source changed: '+path)
                ctx.bind(path,'UNCHANGED_STAGE_WRAPPER_SOURCE',digest)
        except Exception as exc:
            if error is None:error=_error(exc)
            else:secondary_errors.append(_error(exc))
        try:
            manifest_path=ctx.flush_bindings()
            if manifest_path is not None:manifest=binding(manifest_path)
        except Exception as exc:
            if error is None:error=_error(exc)
            else:secondary_errors.append(_error(exc))
        status='RETURNED_WITH_EXPLICIT_OUTCOMES' if error is None else 'FAILED_PRESERVED_OUTPUTS'
        receipt=dict(status=status,action=action,started_epoch=start,ended_epoch=ended,
            elapsed_wall_seconds=elapsed,limits=limits,action_at_return=observed_action,
            terminal_receipt_started_epoch=time.time(),resource_samples=samples,config=config_binding,
            sources=sources,binding_manifest=manifest,error=error,secondary_errors=secondary_errors,
            result_status=result.get('status') if isinstance(result,dict) else None,
            nested_result_statuses={k:v.get('status') for k,v in result.items() if isinstance(v,dict)} if isinstance(result,dict) else {},
            scientific_success_claim=False,startup=startup,preparation=preparation,
            resource_scope='Samples of this Python process and descendants; unrelated processes excluded. Context/stage guards remain authoritative; samples are not exact peaks.',
            timing_boundary='Same start/end as INTENT. Source rechecks, monitor shutdown, binding flush and terminal receipt are outside this interval and remain in campaign wall.')
        new_json(folder/'RESULT.json',receipt)
    return receipt,folder/'RESULT.json'


def main(argv=None):
    entry=time.time();tick=time.perf_counter()
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--action',required=True,choices=list(ACTIONS));args=parser.parse_args(argv)
    path=Path(args.config).resolve(strict=True)
    configure_environment(json.loads(path.read_text(encoding='utf-8-sig')))
    from .context import RecoveryContext
    from .labels import new_json
    import psutil
    ctx=RecoveryContext(path,step_id=ACTIONS[args.action])
    configure_environment(ctx.c)
    startup=dict(started_epoch=entry,ended_epoch=time.time(),elapsed_seconds=time.perf_counter()-tick,
                 scope='CLI/config/shared-import/Context initialization; separate from stage interval')
    receipt,result_path=execute_stage(ctx,args.action,new_json=new_json,psutil_module=psutil,startup=startup)
    print(json.dumps(dict(status=receipt['status'],receipt=str(result_path),
                         elapsed_wall_seconds=receipt['elapsed_wall_seconds'],error=receipt['error'])))
    if receipt['error']:raise RuntimeError(receipt['error']['type']+': '+receipt['error']['message'])


if __name__=='__main__':main()
