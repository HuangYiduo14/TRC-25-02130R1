"""Read-only timing collection and explicitly conditional runtime forecasts.

No solver/model imports or numerical jobs. Complete-query clocks are the actual
caller-to-RESULT clocks, not process lifetime. Nested counters never contribute
to the additive job total. An elapsed clock is not a convergence forecast.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time


FIELDS = ('task','category','seconds','status','timing_boundary','start_epoch',
          'end_epoch','accounting_role','parent_task','time_kind','cap_seconds',
          'network','suite_id','scenario','method','full_coverage','source_path',
          'source_sha256','note','worker_scope','visit_count','region_count',
          'measured_receipt_count','missing_receipt_count')
QUERY_BOUNDARY = ('Caller complete_query_start_epoch before graph/model load to '
    'RESULT elapsed_seconds after final flow/traces/support exports; excludes '
    'CLI preflight/imports, final receipt writes, and caller post-run admission.')


@dataclass(frozen=True)
class JobSpec:
    task: str
    path: str | Path
    intent: str | Path
    network: str
    scenario: str
    method: str
    full_coverage: bool
    suite_id: str | None = None
    action: str | None = None
    module: str = 'partC_rev1_src.pacnhn_v1.cli'
    declared_status: str | None = None


class Snapshot:
    """Hash only the small metadata actually consumed, never native arrays."""
    def __init__(self):
        self.bindings = {}

    def raw(self, path):
        p = Path(path).resolve()
        if not p.exists():
            return None
        data = p.read_bytes()
        if len(data) > 32*1024**2:
            raise ValueError('Timing metadata exceeds bounded reader: '+str(p))
        digest = hashlib.sha256(data).hexdigest()
        self.bindings[str(p)] = dict(path=str(p),bytes=len(data),sha256=digest)
        return data

    def json(self, path, default=None):
        data = self.raw(path)
        return default if data is None else json.loads(data.decode('utf-8-sig'))

    def identity(self, path):
        p = str(Path(path).resolve())
        return dict(source_path=p,source_sha256=self.bindings.get(p,{}).get('sha256'))


def _number(value, name, *, optional=False):
    if value is None and optional:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError('Invalid nonnegative timer '+name)
    return result


def _flag(args, flag):
    for i, arg in enumerate(args):
        if arg == flag and i+1 < len(args):
            return args[i+1]
        if arg.startswith(flag+'='):
            return arg.split('=',1)[1]
    return None


def verify_live_process(record, spec, config_path, process_factory=None):
    """PID reuse and stale RUNNING receipts are not evidence of a live query."""
    if record.get('state') != 'RUNNING' or not spec.action:
        return False
    if record.get('action') != spec.action or record.get('scenario') != spec.scenario:
        return False
    if process_factory is None:
        import psutil
        process_factory = psutil.Process
    try:
        proc = process_factory(int(record['pid']))
        if not proc.is_running():
            return False
        args = proc.cmdline()
        if spec.module not in args or _flag(args,'--action') != spec.action:
            return False
        if _flag(args,'--scenario') != spec.scenario:
            return False
        arg = _flag(args,'--config')
        if not arg:
            return False
        actual = Path(arg)
        if not actual.is_absolute():
            actual = Path(proc.cwd())/actual
        if os.path.normcase(str(actual.resolve())) != os.path.normcase(str(Path(config_path).resolve())):
            return False
        return abs(float(proc.create_time())-float(record['start_epoch'])) <= 120
    except (OSError,ValueError,KeyError,TypeError):
        return False
    except Exception as exc:
        # psutil exceptions differ by platform; absence/access denial cannot
        # establish liveness. Do not classify arbitrary failures as RUNNING.
        if exc.__class__.__module__.startswith('psutil'):
            return False
        raise


def validate_disjoint(rows):
    intervals = []
    for row in rows:
        if row['accounting_role'] != 'DISJOINT_INTERVAL' or row['seconds'] is None:
            continue
        if row['time_kind'] != 'MEASURED_WALL':
            raise ValueError('Only measured wall intervals are additive')
        start = _number(row['start_epoch'],'interval start')
        end = _number(row['end_epoch'],'interval end')
        seconds = _number(row['seconds'],'interval seconds')
        if end < start or abs((end-start)-seconds) > max(1e-5,seconds*1e-10):
            raise ValueError('Interval clock mismatch: '+row['task'])
        intervals.append((start,end,row['task']))
    intervals.sort()
    for left,right in zip(intervals,intervals[1:]):
        if right[0] < left[1]-1e-6:
            raise ValueError('Overlapping additive intervals: '+left[2]+' / '+right[2])
    return dict(interval_count=len(intervals),seconds=sum(b-a for a,b,_ in intervals),
                excludes_nested_worker_and_estimate_rows=True)


def conditional_forecasts(spec, history, elapsed, status, cap_seconds):
    """Observed full refresh cycles; never extrapolate partial origin pilots."""
    base = dict(task=spec.task,network=spec.network,suite_id=spec.suite_id,
        scenario=spec.scenario,method=spec.method,cap_seconds=cap_seconds,
        timing_boundary=QUERY_BOUNDARY,source_status=status,
        time_kind='ESTIMATE',time_to_gap_claim=False)
    if not spec.full_coverage:
        return [dict(base,status='EXCLUDED_PARTIAL_ORIGIN_PILOT',updates=None,
                     seconds_low=None,seconds_high=None)]
    fresh = [r for r in history if r.get('fresh') is True]
    if len(fresh)<2:
        return [dict(base,status='INSUFFICIENT_FULL_OD_CYCLES',updates=None,
                     seconds_low=None,seconds_high=None)]
    if fresh[0]['iteration'] != 0:
        raise ValueError('Missing cold initialization certificate in '+spec.task)
    cycles = []
    for left,right in zip(fresh,fresh[1:]):
        if right['iteration'] != left['iteration']+1:
            raise ValueError('Nonadjacent certificate iterations in '+spec.task)
        delta = float(right['elapsed_seconds'])-float(left['elapsed_seconds'])
        if delta <= 0:
            raise ValueError('Nonincreasing certificate wall clock in '+spec.task)
        cycles.append(delta)
    prefix = _number(fresh[0]['elapsed_seconds'],'first fresh prefix')
    tail = None if elapsed is None else max(0.,elapsed-float(fresh[-1]['elapsed_seconds']))
    completed = status not in ('IN_PROGRESS','INCOMPLETE_UNVERIFIED','FAILED')
    if not completed:
        tail = None
    result=[]
    for updates in (50,100,300):
        low=prefix+updates*min(cycles)+(tail or 0.)
        high=prefix+updates*max(cycles)+(tail or 0.)
        result.append(dict(base,status='CONDITIONAL_OBSERVED_CYCLE_EXTRAPOLATION',
            updates=updates,seconds_low=low,seconds_high=high,
            observed_cycles=len(cycles),cycle_seconds_min=min(cycles),cycle_seconds_max=max(cycles),
            measured_initial_prefix_seconds=prefix,measured_final_tail_seconds=tail,
            finalization_tail_included=tail is not None,
            within_cap_low=cap_seconds is not None and low<=cap_seconds,
            within_cap_high=cap_seconds is not None and high<=cap_seconds,
            assumption='Future full-OD iteration wall times stay within observed range. '
              'Changing flow, line search, traces, model/device and system load can invalidate it. '
              'No guarantee of convergence or target-gap time; unmeasured finalization is excluded.'))
    return result


def _row(spec, category, seconds, status, boundary, *, parent=None, **extra):
    row = {k:None for k in FIELDS}
    row.update(task=spec.task,category=category,seconds=seconds,status=status,
        timing_boundary=boundary,accounting_role='NESTED_DETAIL',parent_task=parent or spec.task,
        time_kind='MEASURED_WALL',network=spec.network,suite_id=spec.suite_id,
        scenario=spec.scenario,method=spec.method,full_coverage=spec.full_coverage)
    if 'notes' in extra:extra['note']=extra.pop('notes')
    row.update(extra)
    return row


def collect_job(spec, snapshot, *, config_path, process_records=(), now_epoch=None):
    root=Path(spec.path);now=time.time() if now_epoch is None else now_epoch
    intent=snapshot.json(spec.intent,{})
    result=snapshot.json(root/'RESULT.json',{})
    history=snapshot.json(root/'HISTORY.json',[])
    stages=snapshot.json(root/'STAGES.json',[])
    progress=snapshot.json(root/'PROGRESS.json',{})
    failed=snapshot.json(root/'FAILED.json',{})
    pilot_receipt=snapshot.json(root/'TIMING_PILOT.json',{})
    if pilot_receipt:
        if (pilot_receipt.get('full_coverage') is not True or not spec.full_coverage or
            pilot_receipt.get('exact_served_OD_membership') is not True or
            pilot_receipt.get('result_sha256') != snapshot.identity(root/'RESULT.json')['source_sha256']):
            raise ValueError('Full-OD timing pilot admission/RESULT binding differs')
    if 'full_coverage' in intent and bool(intent['full_coverage']) != spec.full_coverage:
        raise ValueError('Job population declaration differs: '+spec.task)
    start=result.get('complete_query_start_epoch',intent.get('complete_query_start_epoch',intent.get('run_start_epoch')))
    deadline=result.get('deadline_epoch',intent.get('deadline_epoch'))
    start=_number(start,'query start',optional=True)
    deadline=_number(deadline,'deadline',optional=True)
    cap=None if start is None or deadline is None else deadline-start
    if cap is not None and cap<=0:
        raise ValueError('Nonpositive query cap')
    if result:
        status=result['status'];elapsed=_number(result['elapsed_seconds'],'elapsed')
        if start is None:
            raise ValueError('Completed query lacks start epoch')
        intended_start=intent.get('complete_query_start_epoch',intent.get('run_start_epoch'))
        if intended_start is not None and abs(start-float(intended_start))>1e-6:
            raise ValueError('Query result/intent start differs')
    elif failed:
        status='FAILED';elapsed=None
    elif intent:
        live=any(verify_live_process(r,spec,config_path) for r in process_records)
        status='IN_PROGRESS' if live else 'INCOMPLETE_UNVERIFIED'
        elapsed=max(0.,now-start) if live and start is not None else None
    else:
        status=spec.declared_status or 'NOT_STARTED';elapsed=None
    category='COMPLETE_QUERY' if result else 'QUERY_PROGRESS'
    if root.parent.name=='assignments':category='ONLINE_ASSIGNMENT_'+category
    rows=[_row(spec,category,elapsed,status,QUERY_BOUNDARY,
        start_epoch=start,end_epoch=None if elapsed is None else start+elapsed,
        accounting_role='DISJOINT_INTERVAL' if elapsed is not None else 'NESTED_DETAIL',
        parent_task='CURRENT_SERIAL_ASSIGNMENTS',cap_seconds=cap,
        **snapshot.identity(root/'RESULT.json' if result else spec.intent),
        notes='Current process verified live; unfinished wall clock.' if status=='IN_PROGRESS' else
            'Query clock complete; scientific convergence and caller admission are separate statuses.' if result else
            'No complete wall measurement; persisted status alone does not establish liveness.')]
    if history:
        last=max(float(r['elapsed_seconds']) for r in history)
        if elapsed is not None and last>elapsed+1e-4:
            raise ValueError('History exceeds query clock: '+spec.task)
        rows.append(_row(spec,'FIRST_FRESH_CERTIFICATE_PREFIX',float(history[0]['elapsed_seconds']),status,
            'Start through first persisted fresh certificate, before its checkpoint IO; overlaps all earlier detail.',
            **snapshot.identity(root/'HISTORY.json')))
    subtotal=0.
    detail = [('INITIALIZATION',sum(_number(r['seconds'],'stage') for r in stages),root/'STAGES.json',
        'Persisted solver STAGES: feasible AON, reachable OD membership, witness/coverage generation.'),
        ('VALUE_COST',sum(_number(r.get('value_cost_seconds',0),'evaluate') for r in history),root/'HISTORY.json',
         'Certificate value/cost evaluations only; excludes line-search and accepted-step evaluations.'),
        ('AON_REFRESH',sum(_number(r.get('aon_seconds',0),'AON') for r in history),root/'HISTORY.json',
         'Certificate AON wall time including parallel loader and trace assembly; not sum of worker CPU.'),
        ('LINE_SEARCH',sum(_number(r.get('line_search_seconds',0),'line search') for r in history),root/'HISTORY.json',
         'Recorded accepted-step line-search wall timers; missing/failed attempts remain unattributed.')]
    for category,seconds,path,boundary in detail:
        if not path.exists():
            rows.append(_row(spec,category,None,'NOT_RECORDED',boundary));continue
        subtotal+=seconds
        rows.append(_row(spec,category,seconds,status,boundary,**snapshot.identity(path)))
    if elapsed is not None:
        if subtotal>elapsed+1e-4:
            raise ValueError('Nested disjoint subclocks exceed query wall: '+spec.task)
        rows.append(_row(spec,'UNATTRIBUTED_WITHIN_QUERY',max(0.,elapsed-subtotal),status,
            'Remainder includes input load, certificate/conservation, trace mixing, checkpoint/export IO, '
            'accepted-step evaluations and unrecorded partial work; not an isolated load/IO measurement.'))
    for category in ('INPUT_MODEL_LOAD','CERTIFICATE_CONSERVATION','CHECKPOINT_IO','CPU_TIME','GPU_TIME'):
        rows.append(_row(spec,category,None,'NOT_ISOLATED','No independent timer in current source.'))
    return rows,conditional_forecasts(spec,history,elapsed,status,cap),dict(task=spec.task,status=status,
        graph_identity=result.get('graph_identity',intent.get('graph_identity')),
        OD_identity=result.get('OD_identity',intent.get('OD_identity')),
        final_certificate_fresh=result.get('final_certificate_fresh'),
        reported_state_iteration=result.get('reported_state_iteration'),
        accepted_updates=result.get('accepted_updates'),latest_persisted_progress=progress)


def _jobs(root, snapshot, suites):
    jobs=[]
    for scenario in ('open','i95'):
        for action in ('reference','pilot'):
            if action=='pilot' and not (root/'intents'/f'{action}_{scenario}.json').exists() and not (root/'pilots'/scenario).exists():
                continue
            jobs.append(JobSpec(f'{action}_{scenario}',root/('references' if action=='reference' else 'pilots')/scenario,
                root/'intents'/f'{action}_{scenario}.json','P',scenario,'CFW',action=='reference',action=action))
    for sid in suites:
        if sid=='s01':raise ValueError('s01 is excluded')
        for scenario in ('open','i95'):
            intent=root/'intents'/f'pilot_AC_{sid}_{scenario}.json'
            path=root/'pilots'/f'AC_{sid}_FFTT_{scenario}_fullOD'
            if intent.exists() or path.exists():
                jobs.append(JobSpec(f'pilot_AC_{sid}_FFTT_{scenario}_fullOD',path,intent,
                    'AC',scenario,'FW',True,sid,module='partC_rev1_src.pacnhn_v1.pilots'))
    dispositions={r['case']['case_id']:r['status'] for r in snapshot.json(root/'ASSIGNMENT_RESULTS.json',[]) if 'case' in r}
    for row in snapshot.json(root/'ASSIGNMENT_MATRIX.json',[]):
        name=row['case_id']
        jobs.append(JobSpec(name,root/'assignments'/name,root/'intents'/f'{name}.json',row['network'],
            row['scenario'],row['solver'],True,row.get('suite_id'),declared_status=dispositions.get(name)))
    return jobs


def historical_rows(path, snapshot):
    data=snapshot.raw(path)
    if data is None:
        raise FileNotFoundError(path)
    rows=[]
    for record in csv.DictReader(io.StringIO(data.decode('utf-8-sig'))):
        spec=JobSpec(record['benchmark_id'],Path(path).parent,path,'ORIGINAL_PHYSICAL','historical','CFW',True)
        for key,category in [('update_wall_seconds','HISTORICAL_CUMULATIVE_UPDATE_PREFIX'),
                             ('broader_campaign_elapsed_seconds','HISTORICAL_BROADER_CAMPAIGN'),
                             ('later_refresh_only_seconds','HISTORICAL_REFRESH_ONLY')]:
            if record.get(key):
                rows.append(_row(spec,category,_number(record[key],key),'HISTORICAL_CONTEXT_ONLY',
                    'Accepted original physical audit field '+key+'; standalone query clock and matched hardware unknown.',
                    parent='HISTORICAL_PHYSICAL_CAMPAIGN',**snapshot.identity(path),
                    notes='Cumulative prefixes overlap: never add. Refresh-only is not assignment time. '
                    'Worker count='+record.get('worker_count','')+'; source field='+record.get('source_field','')+
                    '; RG_SPTT='+record.get('RG_SPTT','')+'; checkpoint_available='+record.get('checkpoint_available','')+
                    '; no matched-hardware speedup supported.'))
    return rows


def worker_rows(path, snapshot, root):
    """Explicit visit receipts only: independent workers may overlap in time."""
    p=Path(path).resolve();r=snapshot.json(p)
    if not p.is_relative_to(Path(root).resolve()) or p.parent.name!='timing':
        raise ValueError('Explicit regional timing receipt must belong to this output root')
    total=_number(r['complete_slice_seconds'],'worker complete slice')
    name='training/'+str(p.relative_to(root)).replace('\\','/')
    spec=JobSpec(name,p,p,'REGIONAL','training','MIXED',False)
    common=dict(parent='TRAINING_STAGE',time_kind='WORKER_WALL',**snapshot.identity(p),
        notes='Independent worker wall; overlapping workers cannot be summed as campaign wall. '
              'Includes graph load; excludes final timing/STATE receipt write and directory-size accounting.')
    rows=[_row(spec,'WORKER_COMPLETE_SLICE',total,'RECORDED_SLICE',
        'region_slice entry through complete_slice_seconds capture in finally; receipt alone does not prove successful training.',**common)]
    subtotal=0.
    for key in ('input_load','labels','dataset_load','power4','training_icnn','training_xgboost','evaluation'):
        value=_number(r.get(key),key,optional=True)
        if value is not None:subtotal+=value
        rows.append(_row(spec,'WORKER_'+key.upper(),value,'RECORDED' if value is not None else 'NOT_RECORDED',
            'Isolated regional timer '+key+' within complete slice; POWER4 includes its local evaluation.',**common))
    if subtotal>total+1e-4:
        raise ValueError('Regional nested subclocks exceed complete worker slice')
    rows.append(_row(spec,'WORKER_UNATTRIBUTED',max(0.,total-subtotal),'RECORDED_SLICE',
        'Complete slice minus known disjoint detail timers; includes untimed audits/IO and partial failed work.',**common))
    return rows


def _manifest_bytes(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')


def aggregate_workers(paths,snapshot,root, *,retain_individual=False):
    """Bounded publication groups; receipt identity, not equal bytes, is the unit."""
    groups={};receipts=[];raw=[];seen=set();root=Path(root).resolve()
    for path in paths:
        p=Path(path).resolve()
        if p in seen:raise ValueError('Duplicate worker receipt would double-count work')
        seen.add(p);parts=p.relative_to(root).parts
        if 'regions' in parts:
            at=parts.index('regions');sid=parts[at+1];region=str(p.parent.parent)
            if sid not in {'s02','s03','s04','s05','s06'}:raise ValueError('Unapproved worker suite')
            scope=('CONDITIONING_PROBE' if 'conditioning_probes' in parts[:at] else
                   'REGIONAL_PILOT' if 'pilots' in parts[:at] else 'PRODUCTION')
        elif 'cpu_concurrency_gate' in parts:
            sid='SYNTHETIC';scope='SYNTHETIC_CONCURRENCY_PILOT';region=str(p.parent.parent)
        else:raise ValueError('Worker receipt lacks a recognized owned suite/pilot layout')
        rows=worker_rows(p,snapshot,root)
        identity=snapshot.identity(p)
        receipts.append(dict(**identity,bytes=snapshot.bindings[str(p)]['bytes'],suite_id=sid,
            worker_scope=scope,region_path=region))
        if retain_individual:raw.extend(rows)
        for row in rows:
            key=(sid,scope,row['category'])
            group=groups.setdefault(key,dict(seconds=[],visits=set(),regions=set(),missing=0,
                timing_boundary=row['timing_boundary']))
            group['visits'].add(str(p));group['regions'].add(region)
            if row['seconds'] is None:group['missing']+=1
            else:group['seconds'].append(row['seconds'])
    manifest=dict(schema='PACNHN_WORKER_TIMING_RECEIPTS_V1',
        identity_policy='Each unique resolved receipt path is one visit. Equal content at distinct paths is not deduplicated.',
        receipts=sorted(receipts,key=lambda r:r['source_path']))
    digest=hashlib.sha256(_manifest_bytes(manifest)).hexdigest()
    manifest_path=root/'analysis/WORKER_RECEIPT_MANIFEST.json';rows=[]
    for (sid,scope,category),group in sorted(groups.items()):
        spec=JobSpec('workers_'+sid+'_'+scope,manifest_path,manifest_path,'REGIONAL','training','MIXED',False,sid)
        measured=len(group['seconds']);missing=group['missing']
        rows.append(_row(spec,category,math.fsum(group['seconds']) if measured else None,
            'AGGREGATED_PARTIAL_RECEIPTS' if missing and measured else 'MISSING_TIMER' if missing else 'AGGREGATED_RECORDED_WORKER_WALL',
            group['timing_boundary'],parent='TRAINING_STAGE',time_kind='WORKER_WALL',
            worker_scope=scope,visit_count=len(group['visits']),region_count=len(group['regions']),
            measured_receipt_count=measured,missing_receipt_count=missing,
            source_path=str(manifest_path),source_sha256=digest,
            notes='Sum of independent worker wall counters, not campaign elapsed or CPU time. '
                  'Workers can overlap; category subtotals are nested inside WORKER_COMPLETE_SLICE. '
                  'Source SHA binds the canonical receipt manifest emitted with export.'))
    return rows,manifest,digest,raw


def _owned_metadata(path, root):
    path=Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Training timing evidence must belong to this campaign output')
    return path


def _context_prefix(path, snapshot, root):
    path=_owned_metadata(path,root);record=snapshot.json(path)
    if not record or int(record['pid'])<=0:raise ValueError('Missing controller resource receipt')
    end=_number(record['time'],'resource sample epoch')
    seconds=_number(record['elapsed_seconds'],'resource sampled elapsed')
    start=end-seconds
    if start<0:raise ValueError('Invalid Context resource clock')
    return record,start,end,seconds


def conditioning_rows(c, root, snapshot, *, resource_receipts=None):
    """Small, bounded probe summaries; every counter remains inside Stage2 wall.

    Resource associations are explicit probe-id -> Context resource paths. A
    resource's timestamp alone cannot identify which invocation produced it.
    """
    root=Path(root);rows=[];states=[];resources=resource_receipts or {}
    clock_path=root/'training/TRAINING_CLOCK.json';clock=snapshot.json(clock_path,{})
    folders=sorted((root/'training/conditioning_probes').glob('*/INTENT.json'))
    seen=set()
    for ipath in folders:
        intent=snapshot.json(ipath);probe=ipath.parent.name;seen.add(probe)
        if (intent.get('phase_id')!=c['phase_id'] or intent.get('run_id')!=c['run_id']
                or intent.get('action')!='conditioning-probe' or intent.get('probe_id')!=probe
                or not clock or intent.get('original_training_clock')!=clock):
            raise ValueError('Conditioning probe run/original clock identity mismatch')
        result_path=ipath.parent/'RESULT.json';result=snapshot.json(result_path,{})
        start=_number(intent['started_epoch'],'probe intent start')
        deadline=_number(intent['deadline_epoch'],'probe deadline')
        if start<clock['stage_start_epoch'] or deadline<=start:
            raise ValueError('Invalid conditioning probe clock')
        end=seconds=None;status='INCOMPLETE_NO_TERMINAL_TIME'
        if result:
            if (result.get('intent_sha256')!=snapshot.identity(ipath)['source_sha256']
                    or result.get('original_training_clock_sha256')!=snapshot.identity(clock_path)['source_sha256']
                    or result.get('started_epoch')!=start or result.get('deadline_epoch')!=deadline):
                raise ValueError('Conditioning result source binding/clock mismatch')
            end=_number(result['ended_epoch'],'probe result epoch')
            if end<start:raise ValueError('Conditioning result precedes intent')
            seconds=end-start;status='RECORDED_PROBE_'+str(result['status'])
        spec=JobSpec('conditioning_'+probe,result_path,ipath,'REGIONAL','training','MIXED',False)
        rows.append(_row(spec,'CONDITIONING_PROBE_WALL',seconds,status,
            'INTENT.started_epoch through RESULT.ended_epoch after source checks and bindings; '
            'excludes earlier Context/gate/input checks and final RESULT write.',
            parent='TRAINING_STAGE',start_epoch=start,end_epoch=end,cap_seconds=deadline-start,
            **snapshot.identity(result_path if result else ipath),
            notes='Nested within the original Stage2 clock. PASS is probe validation, not an accuracy-target claim.'))
        prefix=None;resource=resources.get(probe)
        if resource is not None:
            _,cs,ce,_=_context_prefix(resource,snapshot,root)
            if cs>start or ce<start or end is not None and ce>end:
                raise ValueError('Explicit probe Context resource falls outside its invocation')
            prefix=start-cs
        rows.append(_row(spec,'CONDITIONING_CONTEXT_TO_INTENT',prefix,
            'MEASURED_FROM_EXPLICIT_CONTEXT_RESOURCE' if prefix is not None else 'MISSING_CONTEXT_RESOURCE_ASSOCIATION',
            'Explicitly associated Context start (resource.time - resource.elapsed_seconds) through INTENT.started_epoch; '
            'excludes Python process startup/imports before Context construction.',parent='TRAINING_STAGE',
            start_epoch=cs if prefix is not None else None,end_epoch=start if prefix is not None else None,
            **snapshot.identity(resource if resource is not None else ipath),
            notes='Caller explicitly associates resource with probe; no association inferred from similar timestamps.'))
        groups={};region_total=0.;seen_regions=set()
        for record in result.get('rows',[]):
            key=record['key'];sid=key.split(':')[0]
            if key in seen_regions or sid not in c.get('suite_ids',[]):
                raise ValueError('Duplicate or unapproved conditioning region')
            seen_regions.add(key)
            elapsed=_number(record['elapsed_seconds'],'probe region elapsed');region_total+=elapsed
            groups.setdefault((sid,'CONDITIONING_REGION_WALL'),[]).append(elapsed)
            subtotal=0.
            for timer in record.get('timings',[]):
                if timer['family'] not in ('ICNN','XGBOOST'):raise ValueError('Unknown conditioning timer family')
                value=_number(timer['seconds'],'conditioning fit block');subtotal+=value
                groups.setdefault((sid,'CONDITIONING_'+timer['family']+'_FIT'),[]).append(value)
            if subtotal>elapsed+1e-4:raise ValueError('Conditioning fit timers exceed regional wall')
        if seconds is not None and region_total>seconds+1e-4:
            raise ValueError('Sequential conditioning regions exceed complete probe wall')
        for (sid,category),values in sorted(groups.items()):
            rspec=JobSpec(spec.task,result_path,ipath,'REGIONAL','training','MIXED',False,sid)
            rows.append(_row(rspec,category,math.fsum(values),'RECORDED_NESTED_COUNTERS',
                'Sum of sequential per-region elapsed counters before STATE/region RESULT writes.' if category=='CONDITIONING_REGION_WALL' else
                'Sum of measured '+category.removeprefix('CONDITIONING_').removesuffix('_FIT')+' fitting blocks; nested in regional elapsed. '
                'Actual callback overrun remains included, never truncated to a requested block cap.',
                parent=spec.task,worker_scope='CONDITIONING_PROBE',
                region_count=sum(k.startswith(sid+':') for k in seen_regions),measured_receipt_count=len(values),
                **snapshot.identity(result_path),notes='Sequential wall, not CPU/GPU utilization. No late numerical work is run by this collector.'))
        states.append(dict(task=spec.task,status=status,start_epoch=start,end_epoch=end,
            result_status=result.get('status'),region_rows=len(seen_regions),context_prefix_seconds=prefix,
            source_intent=snapshot.identity(ipath),source_result=snapshot.identity(result_path)))
    if set(resources)-seen:raise ValueError('Context resource names an unknown conditioning probe')
    return rows,states


def verify_training_process(resource, config_path, process_factory=None):
    """Accept only the actual parent training command, not a reused PID/worker."""
    if process_factory is None:
        import psutil
        process_factory=psutil.Process
    try:
        proc=process_factory(int(resource['pid']));args=proc.cmdline()
        if (not proc.is_running() or 'partC_rev1_src.pacnhn_v1.training' not in args
                or _flag(args,'--action')!='train'):return False
        arg=_flag(args,'--config')
        if not arg:return False
        actual=Path(arg)
        if not actual.is_absolute():actual=Path(proc.cwd())/actual
        start=float(resource['time'])-float(resource['elapsed_seconds'])
        return (os.path.normcase(str(actual.resolve()))==os.path.normcase(str(Path(config_path).resolve()))
                and 0<=start-float(proc.create_time())<=120)
    except (OSError,ValueError,KeyError,TypeError):return False
    except Exception as exc:
        if exc.__class__.__module__.startswith('psutil'):return False
        raise


def controller_rows(c, root, snapshot, episodes, *, config_path):
    """Explicit parent episodes, never workers. Missing terminal clocks stay missing.

    Each descriptor has resource_path and optional disposition_path. Supported
    dispositions are an authenticated intentional-stop audit, owned pause or
    resource-pause receipt. A resource sample is a prefix, not an exit timestamp.
    """
    rows=[];states=[];seen=set();root=Path(root)
    clock=snapshot.json(root/'training/TRAINING_CLOCK.json',{})
    for episode in episodes:
        rp=_owned_metadata(episode['resource_path'],root)
        resource,start,sampled,prefix=_context_prefix(rp,snapshot,root)
        key=(int(resource['pid']),start)
        if key in seen:raise ValueError('Duplicate controller invocation would repeat timing evidence')
        seen.add(key)
        if not clock or start<clock['stage_start_epoch']:raise ValueError('Controller precedes original Stage2 clock')
        dp=episode.get('disposition_path');disposition={};end=None;absence=None
        source_intent=None;intent_evidence=None
        status='RECORDED_PREFIX_RUNNING_VERIFIED' if verify_training_process(resource,config_path) else 'RECORDED_PREFIX_LIVENESS_UNVERIFIED'
        if dp is not None:
            dp=_owned_metadata(dp,root);disposition=snapshot.json(dp)
            if disposition.get('status')=='INTENTIONAL_PAUSE_FOR_SCHEDULER_REPAIR':
                if (disposition.get('phase_id')!=c['phase_id'] or disposition.get('run_id')!=c['run_id']
                        or disposition.get('original_clock')!=clock or disposition.get('stopped_parent_pid')!=resource['pid']
                        or disposition.get('all_stopped_processes_verified_absent') is not True):
                    raise ValueError('Interrupted controller run/clock/PID evidence mismatch')
                bound=[b for b in disposition.get('bindings',[]) if Path(b['snapshot']).resolve()==rp]
                if len(bound)!=1 or bound[0]['sha256']!=snapshot.identity(rp)['source_sha256']:
                    raise ValueError('Interrupted controller resource snapshot binding mismatch')
                absence=_number(disposition['observed_epoch'],'observed process absence')
                if absence<sampled:raise ValueError('Process absence precedes last resource sample')
                status='INTERRUPTED_TERMINAL_TIME_NOT_RECORDED'
            elif disposition.get('status')=='PAUSED_FOR_REPAIR':
                request=disposition.get('pause_request',{}).get('request',{})
                if (disposition.get('clock')!=clock or request.get('phase_id')!=c['phase_id']
                        or request.get('run_id')!=c['run_id'] or request.get('target_pid')!=resource['pid']):
                    raise ValueError('Owned pause run/clock/PID identity mismatch')
                end=_number(disposition['completed_epoch'],'pause receipt completion')
                if end<sampled:raise ValueError('Pause completion precedes last resource sample')
                status='PAUSED_RECORDED_CONTEXT_WALL'
            elif disposition.get('status')=='RESOURCE_PAUSED':
                cp=_owned_metadata(disposition['training_clock_path'],root)
                if (disposition.get('schema')!='PACNHN_RESOURCE_PAUSE_V1'
                        or disposition.get('phase_id')!=c['phase_id'] or disposition.get('run_id')!=c['run_id']
                        or disposition.get('pid')!=resource['pid'] or disposition.get('clock')!=clock
                        or cp!=(root/'training/TRAINING_CLOCK.json').resolve()
                        or disposition.get('training_clock_sha256')!=snapshot.identity(cp)['source_sha256']
                        or disposition.get('error_type')!='MemoryError'
                        or disposition.get('no_new_registry_or_TEST_during_pause') is not True):
                    raise ValueError('Resource pause run/PID/clock/schema identity mismatch')
                cleanup=disposition.get('owned_worker_cleanup',{})
                if cleanup.get('completed') is not True or cleanup.get('mode') not in (
                        'TERMINATED_OWNED_POOL','JOINED_OWNED_POOL','NO_ACTIVE_POOL_AT_EXCEPTION_BOUNDARY'):
                    raise ValueError('Resource pause lacks completed owned-worker cleanup')
                end=_number(disposition['completed_epoch'],'resource pause completion')
                cleaned=_number(cleanup.get('completed_epoch'),'resource pause cleanup completion')
                if end<sampled or not start<=cleaned<=end:
                    raise ValueError('Resource pause completion/cleanup precedes its measured context')
                context_start=disposition.get('context_start_epoch')
                if context_start is not None and abs(_number(context_start,'resource pause Context start')-start)>1e-5:
                    raise ValueError('Resource pause Context/resource start differs')
                ip=disposition.get('source_intent');intent_sha=disposition.get('source_intent_sha256')
                if ip is None and intent_sha is None:
                    intent_evidence='ABSENT_RECORDED'
                elif ip is None or not intent_sha:
                    raise ValueError('Resource pause source intent path/hash must be paired')
                else:
                    ip=_owned_metadata(ip,root);intent=snapshot.json(ip)
                    if (snapshot.identity(ip)['source_sha256']!=intent_sha or not intent
                            or intent.get('phase_id')!=c['phase_id'] or intent.get('run_id')!=c['run_id']
                            or intent.get('clock')!=clock or intent.get('pilot') is not False):
                        raise ValueError('Resource pause source intent binding/run/original clock mismatch')
                    source_intent=snapshot.identity(ip);intent_evidence='VERIFIED_IMMUTABLE_INTENT'
                status=('RESOURCE_PAUSED_RECORDED_CONTEXT_WALL' if source_intent is not None else
                        'RESOURCE_PAUSED_RECORDED_CONTEXT_WALL_INTENT_UNAVAILABLE')
            else:raise ValueError('Unsupported controller terminal disposition')
        name='training_controller_'+str(resource['pid'])+'_'+str(int(start*1e6))
        spec=JobSpec(name,rp,rp,'REGIONAL','training','MIXED',False)
        rows.append(_row(spec,'TRAINING_CONTROLLER_SAMPLED_PREFIX',prefix,status,
            'Context construction start through last resource sample; includes parent/worker waits and excludes process startup/imports. '
            'A sampled prefix is not terminal duration.',parent='TRAINING_STAGE',start_epoch=start,end_epoch=sampled,
            **snapshot.identity(rp),notes='Nested wall. Concurrent worker timers are contained and never added to this prefix.'))
        rows.append(_row(spec,'TRAINING_CONTROLLER_EPISODE_WALL',None if end is None else end-start,status,
            'Context construction start through owned administrative/resource pause receipt completed_epoch, when recorded; '
            'excludes pause receipt write and process teardown. No terminal epoch inferred from snapshot observation or resource sample.',
            parent='TRAINING_STAGE',start_epoch=start,end_epoch=end,
            **snapshot.identity(dp if dp is not None else rp),
            notes='All resumed episodes share the original Stage2 clock. Pauses/gaps remain inside that stage, not a new additive subtotal. '
                +('Resource pause source-intent evidence: '+intent_evidence+'. Existing TEST/registry state may predate this pause.' if intent_evidence else '')))
        states.append(dict(task=name,status=status,pid=resource['pid'],context_start_epoch=start,
            sampled_epoch=sampled,measured_prefix_seconds=prefix,terminal_epoch=end,
            observed_absence_epoch=absence,terminal_duration_lower_bound_seconds=prefix,
            terminal_duration_upper_bound_seconds=None if absence is None else absence-start,
            source_resource=snapshot.identity(rp),source_disposition=snapshot.identity(dp) if dp is not None else None,
            source_intent=source_intent,source_intent_evidence=intent_evidence))
    return rows,states


def stage_rows(c,root,snapshot,existing_rows, *,include_campaign_spans=True):
    """Terminal wall boundaries must come from exact recorded clocks, never mtimes."""
    root=Path(root);rows=[];states=[]
    clock_path=root/'training/TRAINING_CLOCK.json';result_path=root/'training/RESULT.json'
    clock=snapshot.json(clock_path,{});result=snapshot.json(result_path,{})
    spec=JobSpec('TRAINING_STAGE',result_path,clock_path,'REGIONAL','training','MIXED',False)
    start=_number(clock.get('stage_start_epoch'),'training start',optional=True)
    deadline=_number(clock.get('stage_deadline_epoch'),'training deadline',optional=True)
    status='MISSING_TRAINING_CLOCK' if start is None else 'INCOMPLETE_NO_TERMINAL_TIME'
    end=elapsed=None;overlaps=[]
    if start is not None and result:
        if result.get('clock')!=clock:raise ValueError('Training result/original clock differs')
        if result.get('status')=='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE' and all(
                k in result for k in ('finalization_started_epoch','finalization_elapsed_seconds')):
            final_start=_number(result['finalization_started_epoch'],'training finalization start')
            final_elapsed=_number(result['finalization_elapsed_seconds'],'training finalization elapsed')
            if final_start<start:raise ValueError('Finalization precedes original training clock')
            end=final_start+final_elapsed;elapsed=end-start;status='COMPLETED_RECORDED_WALL'
            if result.get('finalization_ended_epoch') is not None and abs(float(result['finalization_ended_epoch'])-end)>1e-5:
                raise ValueError('Training finalization end clock mismatch')
        else:status='MISSING_TERMINAL_FINALIZATION_CLOCK'
    if deadline is not None and start is not None and deadline<=start:raise ValueError('Invalid training cap')
    if elapsed is not None:
        for r in existing_rows:
            if r['accounting_role']=='DISJOINT_INTERVAL' and r['seconds'] is not None and (
                    start<r['end_epoch']-1e-6 and r['start_epoch']<end-1e-6):
                overlaps.append(r['task'])
        if overlaps:status='COMPLETED_OVERLAP_RETAINED_NESTED'
    rows.append(_row(spec,'OFFLINE_TRAINING_STAGE_WALL',elapsed,status,
        'TRAINING_CLOCK.stage_start_epoch through RESULT.finalization_started_epoch + finalization_elapsed_seconds; '
        'excludes final RESULT write. Includes regional pilots/retries, controller, worker startup, idle and finalization within that span.',
        parent='CAMPAIGN',start_epoch=start,end_epoch=end,cap_seconds=None if start is None or deadline is None else deadline-start,
        accounting_role='DISJOINT_INTERVAL' if elapsed is not None and not overlaps else 'NESTED_DETAIL',
        **snapshot.identity(result_path if result else clock_path),
        notes='All worker rows are nested; no CPU/GPU utilization inferred. '
          +('Overlaps recorded query tasks: '+', '.join(overlaps) if overlaps else
            'No terminal time inferred from current time, status/lock files or file modification times.')))
    states.append(dict(task=spec.task,status=status,overlap_tasks=overlaps,
        source_clock=snapshot.identity(clock_path),source_result=snapshot.identity(result_path),
        finalization_provenance={k:result.get(k) for k in ('finalization_start_basis',
            'pre_freeze_finalization_interval_known','finalization_resumed',
            'finalization_wall_includes_restart_gaps','completed_after_stage_deadline')}))
    topo_path=root/'topology/RESULT.json';topo=snapshot.json(topo_path,{})
    duration=_number(topo.get('construction_complete_elapsed_seconds'),'topology construction',optional=True)
    tspec=JobSpec('TOPOLOGY_CONSTRUCTION',topo_path,topo_path,'P','preparation','BUILD',False)
    rows.append(_row(tspec,'OFFLINE_TOPOLOGY_CONSTRUCTION',duration,
        'MEASURED_DURATION_WITHOUT_EPOCH_BOUNDS' if duration is not None else 'MISSING_SOURCE_TIMER',
        'Context construction start through topology RESULT construction_complete_elapsed_seconds capture; '
        'no recorded epoch bounds are inferred from this duration.',parent='STAGE1_BROAD_SPAN',**snapshot.identity(topo_path)))
    if include_campaign_spans:
        original=_number(c.get('original_start_epoch'),'original campaign start',optional=True)
        if start is not None and original is not None and start<original:raise ValueError('Stage2 precedes original campaign start')
        rows.append(_row(JobSpec('STAGE1_BROAD_SPAN',clock_path,clock_path,'CAMPAIGN','preparation','MIXED',False),
            'OFFLINE_STAGE1_BROAD_WALL',None if start is None or original is None else start-original,
            'RECORDED_BOUNDARY_SPAN' if start is not None and original is not None else 'MISSING_END_BOUNDARY',
            'Original campaign Goal start through original Stage2 clock start; includes topology/reference queries, '
            'preparation, retries, supervision and idle. Overlaps nested constituent queries; never add to them.',
            parent='CAMPAIGN',start_epoch=original,end_epoch=start,**snapshot.identity(clock_path)))
        sc_path=root/'STAGE3_CLOCK.json';sr_path=root/'STAGE3_RESULT.json'
        sc=snapshot.json(sc_path,{});sr=snapshot.json(sr_path,{})
        st=_number(sc.get('original_stage_start_epoch'),'Stage3 start',optional=True);se=sd=None
        ss='MISSING_STAGE3_CLOCK' if st is None else 'INCOMPLETE_NO_TERMINAL_TIME'
        if st is not None and sr:
            if sr.get('clock')!=sc:raise ValueError('Stage3 result/original clock differs')
            if sr.get('status')=='COMPLETE_WITH_RECORDED_OUTCOMES' and 'elapsed_seconds' in sr:
                sd=_number(sr['elapsed_seconds'],'Stage3 elapsed');se=st+sd;ss='RECORDED_BOUNDARY_SPAN'
        rows.append(_row(JobSpec('STAGE3_BROAD_SPAN',sr_path,sc_path,'CAMPAIGN','assignment','MIXED',False),
            'ONLINE_ASSIGNMENT_STAGE3_BROAD_WALL',sd,ss,
            'Original Stage3 clock start through STAGE3_RESULT elapsed_seconds capture; excludes its final JSON write. '
            'Includes inference pilots, individual queries, inter-query admission, supervision and idle; overlaps constituent queries.',
            parent='CAMPAIGN',start_epoch=st,end_epoch=se,
            cap_seconds=None if st is None else float(sc['deadline_epoch'])-st,**snapshot.identity(sr_path if sr else sc_path)))
        states.append(dict(task='STAGE3_BROAD_SPAN',status=ss,source_clock=snapshot.identity(sc_path),source_result=snapshot.identity(sr_path)))
    return rows,states


def collect_timing(config, *, now_epoch=None, extra_jobs=(), inference_receipts=(),
                   worker_receipts=(), historical_csv=None,include_campaign_spans=True,
                   retain_individual_worker_rows=False,conditioning_resources=None,controller_episodes=()):
    """Return (ledger, forecasts, summary), without creating any files.

    Explicit whole-model inference receipts use schema PACNHN_INFERENCE_WALL_V1,
    full_model=true, seconds, repetitions, family, suite_id, device, measurement,
    registry_sha256. Measurement is batch or sequential_calls; amortized batch
    latency is never represented as independently measured single-query latency.
    They are evidence only, not automatically multiplied into solver forecasts.
    """
    snapshot=Snapshot();config_path=Path(config).resolve();c=snapshot.json(config_path)
    if c.get('phase_id')!='PACNHN_V1':raise ValueError('Wrong explicit campaign')
    root=Path(c['output_root']);now=time.time() if now_epoch is None else now_epoch
    jobs=_jobs(root,snapshot,c.get('suite_ids',[]))+list(extra_jobs)
    if len({j.task for j in jobs})!=len(jobs):raise ValueError('Duplicate timing job identities')
    processes=[snapshot.json(p) for p in sorted((root/'processes').glob('*.json'))]
    ledger=[];forecasts=[];states=[]
    for spec in jobs:
        rows,estimate,state=collect_job(spec,snapshot,config_path=config_path,process_records=processes,now_epoch=now)
        ledger.extend(rows);forecasts.extend(estimate);states.append(state)
    inference_ids=set()
    for path in inference_receipts:
        r=snapshot.json(path)
        if r.get('schema')!='PACNHN_INFERENCE_WALL_V1' or r.get('full_model') is not True or not r.get('registry_sha256'):
            raise ValueError('Whole-model bound inference receipt required')
        if r.get('measurement') not in ('batch','sequential_calls') or int(r['repetitions'])<=0:
            raise ValueError('Invalid inference measurement design')
        scenario=r.get('scenario')
        if scenario not in ('open','i95'):raise ValueError('Inference receipt requires explicit open/i95 scenario')
        key=(r['suite_id'],r['family'],scenario)
        if key in inference_ids:raise ValueError('Duplicate suite/family/scenario inference receipt')
        inference_ids.add(key)
        spec=JobSpec('inference_'+r['suite_id']+'_'+r['family']+'_'+scenario,path,path,'AC',scenario,r['family'],True,r['suite_id'])
        ledger.append(_row(spec,'WHOLE_MODEL_INFERENCE',_number(r['seconds'],'inference'),'MEASURED',
            'Explicit whole-model '+r['measurement']+' wall time for '+str(r['repetitions'])+' calls/rows. '+r.get('call_boundary',''),
            parent='INFERENCE_MEASUREMENTS',**snapshot.identity(path),
            notes='Device='+r['device']+'; registry='+r['registry_sha256']+
                '; no CPU/GPU utilization inferred; batch amortization is not single-call latency.'))
        for key,category,boundary in (
            ('load_seconds','INFERENCE_MODEL_AND_GRAPH_LOAD','Complete load_ac graph/registry/model load wall; outside timed cost-field calls.'),
            ('baseline_validation_seconds','INFERENCE_BASELINE_PROJECTION_VALIDATION','Certified P baseline load, exact trace projection and served-OD conservation wall; outside timed cost-field calls.'),
            ('complete_pilot_seconds','INFERENCE_COMPLETE_PILOT','Pilot start through pre-receipt elapsed capture; includes load, projection, calls and guards. Nested in Stage3 broad wall.')):
            value=_number(r.get(key),key,optional=True)
            ledger.append(_row(spec,category,value,'MEASURED' if value is not None else 'MISSING_SOURCE_TIMER',
                boundary,parent='STAGE3_BROAD_SPAN',**snapshot.identity(path),
                notes='Nested scope; do not add complete pilot and its component counters.'))
    workers,worker_manifest,worker_digest,raw=aggregate_workers(worker_receipts,snapshot,root,
        retain_individual=retain_individual_worker_rows)
    ledger.extend(workers)
    probes,probe_states=conditioning_rows(c,root,snapshot,resource_receipts=conditioning_resources)
    episodes,episode_states=controller_rows(c,root,snapshot,controller_episodes,config_path=config_path)
    ledger.extend(probes);ledger.extend(episodes)
    spans,stage_states=stage_rows(c,root,snapshot,ledger,include_campaign_spans=include_campaign_spans)
    ledger.extend(spans)
    if historical_csv is not None:ledger.extend(historical_rows(historical_csv,snapshot))
    additive=validate_disjoint(ledger)
    summary=dict(schema='PACNHN_TIMING_AUDIT_V1',phase_id=c['phase_id'],run_id=c['run_id'],
        status='PROVISIONAL' if any(s['status'] in ('IN_PROGRESS','INCOMPLETE_UNVERIFIED','NOT_STARTED') for s in states) or
            any(s['status'].startswith(('MISSING','INCOMPLETE')) for s in stage_states) else 'COLLECTED',
        captured_epoch=now,jobs=states,additive_job_wall=additive,additive_disjoint_wall=additive,
        input_bindings=list(snapshot.bindings.values()),
        query_boundary=QUERY_BOUNDARY,full_national_inference_receipts=len(inference_receipts),
        regional_worker_receipts=len(worker_receipts),
        conditioning_probes=probe_states,controller_episodes=episode_states,
        stages=stage_states,worker_receipt_manifest=worker_manifest,worker_receipt_manifest_sha256=worker_digest,
        policy='Only DISJOINT_INTERVAL measured wall rows add, including the completed nonoverlapping Stage2 wall. '
          'The additive_job_wall field is retained as an alias of additive_disjoint_wall. Nested counters, worker walls, '
          'historical prefixes and forecasts never add to complete job wall. '
          'No measured CPU/GPU utilization or convergence guarantee is inferred.')
    if retain_individual_worker_rows:summary['_raw_worker_rows']=raw
    return ledger,forecasts,summary


def export_timing(ctx, **kwargs):
    """Explicit caller-owned final/provisional export; never launches work."""
    from .control import write_json
    ledger,forecasts,summary=collect_timing(ctx.path,**kwargs)
    root=ctx.root/'analysis';root.mkdir(parents=True,exist_ok=True)
    raw=summary.pop('_raw_worker_rows',None)
    exports=[('timing.csv',ledger,FIELDS),('conditional_forecasts.csv',forecasts,
        sorted({k for row in forecasts for k in row}))]
    if raw is not None:exports.append(('timing_workers_raw.csv',raw,FIELDS))
    for name,rows,fields in exports:
        target=root/name;tmp=target.with_name(target.name+'.tmp.'+str(os.getpid()))
        with tmp.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        os.replace(tmp,target)
    target=root/'WORKER_RECEIPT_MANIFEST.json';tmp=target.with_name(target.name+'.tmp.'+str(os.getpid()))
    tmp.write_bytes(_manifest_bytes(summary['worker_receipt_manifest']));os.replace(tmp,target)
    write_json(root/'TIMING_AUDIT.json',summary)
    return summary
