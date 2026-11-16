"""Metadata-only recovery timing crosswalk; no fitting, arrays or inference.

Canonical regional visits are counted once. Stage wall, family worker time,
shared label work, ablation and historical measurements remain distinct roles.
"""
from pathlib import Path
import csv,json,math,os,time
from collections import defaultdict
from ..pacnhn_v1.control import read_json,sha
from .ablation import _new
from . import assignment_reconciliation as repair

SCHEMA='RECOVERY24_OFFLINE_TIMING_V1'
FAMILIES={'ICNN':'icnn_fit_and_audit','XGBOOST':'xgboost_fit_and_audit'}
SCOPE='RECOVERY_FAMILY_FIT_AND_AUDIT_WORKER_SECONDS'
FIELDS=['task','category','suite_id','region_key','family','seconds','status','accounting_role',
        'time_kind','timing_boundary','start_epoch','end_epoch','peak_process_tree_bytes',
        'source_path','source_sha256','source_fields','note']


def seconds(value,field):
    if value is None or value=='':return None
    if isinstance(value,bool):raise ValueError('Boolean timing field: '+field)
    number=float(value)
    if not math.isfinite(number) or number<0:raise ValueError('Invalid timing field: '+field)
    return number


def total(values):
    try:value=math.fsum(values)
    except OverflowError as exc:raise ValueError('Timing sum overflow') from exc
    if not math.isfinite(value):raise ValueError('Nonfinite timing sum')
    return value


def visit_rows(record,source):
    key=record['region_key'];sid,rid=key.split(':');timings=record.get('timings',{})
    if sid not in {'s02','s03','s04','s05','s06'} or not rid.isdigit():raise ValueError('Invalid regional visit identity')
    rows=[]
    for family,field in FAMILIES.items():
        value=seconds(timings.get(field),field)
        no_fit=record.get('status')=='PRESERVED_BOTH_TARGETS_MET_NO_FITTING'
        if value is None and no_fit:value=0.
        rows.append(dict(task=f'{key}/visit{record["visit"]}/{family}',category='RECOVERY_FAMILY_FIT_AND_AUDIT',
            suite_id=sid,region_key=key,family=family,seconds=value,
            status='MEASURED' if field in timings else 'EXPLICIT_NO_FIT' if no_fit else 'UNKNOWN_MISSING_FAMILY_TIMER',
            accounting_role='WORKER_DETAIL_NOT_ADDITIVE_TO_STAGE_WALL',time_kind='RECOVERY_WORKER_SECONDS',
            timing_boundary='This complete family fit/audit call, including failed and unqualified candidates; excludes shared load/baseline audit.',
            source_fields='timings.'+field,note='No filtering by training qualification or candidate success.',**source))
    for field in ('dataset_load','initial_audit_and_baselines'):
        value=seconds(timings.get(field),field)
        rows.append(dict(task=f'{key}/visit{record["visit"]}/{field}',category='SHARED_'+field.upper(),
            suite_id=sid,region_key=key,family='',seconds=value,status='MEASURED' if value is not None else 'UNKNOWN_MISSING_TIMER',
            accounting_role='SHARED_WORKER_DETAIL_NOT_ADDITIVE_TO_STAGE_WALL',time_kind='RECOVERY_WORKER_SECONDS',
            timing_boundary='Shared across model families; initial audit includes baseline/Power4 work and cannot be split retrospectively.',
            source_fields='timings.'+field,note='Not allocated to ICNN/XGB and not a standalone Power4 timer.',**source))
    elapsed=seconds(record.get('elapsed_seconds'),'elapsed_seconds')
    rows.append(dict(task=f'{key}/visit{record["visit"]}',category='REGION_VISIT_TOTAL',suite_id=sid,region_key=key,
        family='',seconds=elapsed,status=record.get('status','UNKNOWN'),accounting_role='PARENT_WORKER_SCOPE_NOT_ADDITIVE_TO_DETAILS',
        time_kind='RECOVERY_WORKER_SECONDS',timing_boundary='Complete region visit, including shared and family duties.',
        start_epoch=record.get('start_epoch'),end_epoch=record.get('end_epoch'),source_fields='elapsed_seconds',note='',**source))
    known=[r['seconds'] for r in rows[:-1] if r['seconds'] is not None]
    if elapsed is not None and total(known)>elapsed+max(.01,elapsed*1e-6):raise ValueError('Nested visit timing exceeds its parent wall')
    return rows


def training_identity(record):
    key=record.get('region_key','');parts=key.split(':');visit=record.get('visit')
    if len(parts)!=2 or parts[0] not in {'s02','s03','s04','s05','s06'} or len(parts[1])!=6 or not parts[1].isdigit():
        raise ValueError('Invalid canonical training region identity')
    if type(visit) is not int or visit<1:raise ValueError('Invalid training visit ordinal')
    return key,visit


def unreturned_task_rows(task,key,source,disposition,start=None):
    # None is intentional: dispatch or intent does not prove either family's work.
    rows=[]
    for family in FAMILIES:
        rows.append(dict(task=task+'/'+family,category='RECOVERY_FAMILY_FIT_AND_AUDIT',
            suite_id=key.split(':')[0],region_key=key,family=family,seconds=None,
            status=disposition,accounting_role='WORKER_DETAIL_NOT_ADDITIVE_TO_STAGE_WALL',
            time_kind='RECOVERY_WORKER_SECONDS',timing_boundary='Unreturned task: family work and duration are unobserved.',
            start_epoch=start,source_fields='canonical intent / authenticated pass disposition',
            note='Unknown is not zero; no time is inferred from deadline, file mtime or dispatch.',**source))
    return rows


def primary_pass_records(root,read):
    """Bind the exact primary pass universe; never scan unrelated pass wrappers."""
    path=root/'training/PRIMARY_RESULT.json'
    if not path.exists():raise ValueError('Terminal primary receipt required for complete timing reconciliation')
    primary,primary_source=read(path,'TERMINAL_PRIMARY_TIMING_UNIVERSE')
    if primary.get('status') not in {'PRIMARY_TIME_LIMIT','POLICY_EXHAUSTED_OR_TARGETS_MET'} or primary.get('TEST_used') is not False:
        raise ValueError('Invalid terminal primary universe')
    passes=[];seen=set();last_start=-1.
    for binding in primary['passes']:
        path=Path(binding['path']).resolve(strict=True)
        if path.name!='RESULT.json' or path.parent.parent!=(root/'training/passes').resolve() or not path.parent.name.isdigit():
            raise ValueError('Foreign or noncanonical primary pass path')
        if str(path) in seen:raise ValueError('Duplicate primary pass')
        seen.add(str(path));record,source=read(path,'AUTHENTICATED_PRIMARY_PASS_TIMING',binding['sha256'])
        intent,intent_source=read(path.with_name('INTENT.json'),'PRIMARY_PASS_TIMING_INTENT')
        # Producer run() creates this directory with str(time.time_ns()). This is
        # invocation identity only, never an elapsed-work estimate or file mtime.
        start=int(path.parent.name)/1_000_000_000;end=seconds(intent.get('deadline_epoch'),'pass deadline')
        if end is None or not last_start<start<end:raise ValueError('Invalid primary pass invocation chronology')
        last_start=start
        if record.get('status')!='PASS_COMPLETE_TASK_ACCOUNTING' or record.get('TEST_opened') is not False or intent.get('TEST_opened') is not False:
            raise ValueError('Primary pass is not terminal task accounting')
        if Path(record['output_directory']).resolve()!=path.parent or record.get('deadline_epoch')!=end:
            raise ValueError('Pass directory/deadline mismatch')
        keys=intent.get('region_keys',[]);regions=record.get('regions',[])
        if not keys or len(set(keys))!=len(keys) or {r['region_key'] for r in regions}!=set(keys) or len(regions)!=len(keys) or record.get('requested_regions')!=len(keys):
            raise ValueError('Pass task population mismatch')
        missing=sum(r['status'] in {'INTERRUPTED','UNATTEMPTED'} for r in regions)
        if record.get('completed_visits')!=len(keys)-missing:raise ValueError('Pass completed count mismatch')
        passes.append(dict(start=start,end=end,record=record,source=source,intent_source=intent_source,pass_id=path.parent.name))
    if not passes:raise ValueError('Empty primary timing universe')
    for index,entry in enumerate(passes):
        entry['window_end']=min(entry['end'],passes[index+1]['start']) if index+1<len(passes) else entry['end']
    return passes


def reconcile_training_tasks(visits,intents,passes):
    """Metadata join. Canonical visits win; every omitted task stays explicit."""
    vmap={};imap={};extra=[];ledger=[];accounted_intents=set();by_region=defaultdict(list)
    for record,source in visits:
        key=training_identity(record)
        if key in vmap:raise ValueError('Duplicate canonical visit identity')
        vmap[key]=(record,source)
    for record,source in intents:
        key=training_identity(record)
        if key in imap:raise ValueError('Duplicate canonical intent identity')
        if record.get('TEST_opened') is not False:raise ValueError('Training intent TEST boundary')
        start=seconds(record.get('start_epoch'),'intent start');end=seconds(record.get('deadline_epoch'),'intent deadline')
        if start is None or end is None or end<=start:raise ValueError('Invalid intent chronology')
        imap[key]=(record,source)
        by_region[key[0]].append((key,record))
    for key,(record,source) in vmap.items():
        if key not in imap:raise ValueError('Canonical visit missing its durable intent')
        intent=imap[key][0]
        if record.get('start_epoch')!=intent['start_epoch'] or record.get('deadline_epoch')!=intent['deadline_epoch']:
            raise ValueError('Canonical intent/visit chronology mismatch')
    for key,(intent,source) in imap.items():
        if key not in vmap:
            extra.extend(unreturned_task_rows(f'{key[0]}/visit{key[1]}',key[0],source,'UNKNOWN_STARTED_NO_TERMINAL_RECEIPT',intent['start_epoch']))
    for entry in passes:
        start=entry['start'];end=entry['end'];window_end=entry['window_end'];pass_id=entry['pass_id']
        for item in entry['record']['regions']:
            key=item['region_key'];status=item['status']
            matching=[identity for identity,record in by_region[key] if start<=record['start_epoch']<window_end and record['deadline_epoch']<=end]
            if len(matching)>1:raise ValueError('Ambiguous pass-to-visit chronology')
            identity=matching[0] if matching else None
            if identity is not None:
                if identity in accounted_intents:raise ValueError('One visit assigned to multiple passes')
                accounted_intents.add(identity)
            row=dict(pass_id=pass_id,region_key=key,parent_status=status,visit=identity[1] if identity else None,
                pass_source=entry['source'],pass_intent_source=entry['intent_source'])
            if status=='UNATTEMPTED':
                if identity is not None:raise ValueError('Unattempted pass task has a recorded visit intent')
                row['disposition']='EXPLICIT_UNATTEMPTED_NOT_PERFORMED_WORK'
            elif identity in vmap:
                record,source=vmap[identity]
                if status not in {'INTERRUPTED','WORKER_FAILED'} and record['status']!=status:raise ValueError('Canonical/parent visit status mismatch')
                row.update(disposition='CANONICAL_RECEIPT_COUNTED_ONCE',canonical_source=source,canonical_status=record['status'])
            elif identity is not None:
                if status not in {'INTERRUPTED','WORKER_FAILED'}:raise ValueError('Returned task lacks canonical terminal receipt')
                row.update(disposition='STARTED_NO_TERMINAL_RECEIPT',intent_source=imap[identity][1])
            else:
                if status not in {'INTERRUPTED','WORKER_FAILED'}:raise ValueError('Returned task lacks canonical identity')
                row['disposition']='DISPATCHED_START_AND_WORK_UNOBSERVED'
                extra.extend(unreturned_task_rows(f'pass{pass_id}/{key}',key,entry['source'],'UNKNOWN_DISPATCHED_START_AND_WORK_UNOBSERVED'))
            ledger.append(row)
    return extra,ledger


def family_summaries(rows,suites):
    out=[]
    for sid in suites:
        for family in FAMILIES:
            group=[r for r in rows if r['category']=='RECOVERY_FAMILY_FIT_AND_AUDIT' and r['suite_id']==sid and r['family']==family]
            known=[r['seconds'] for r in group if r['seconds'] is not None];missing=len(group)-len(known)
            out.append(dict(suite_id=sid,family=family,visit_rows=len(group),known_timers=len(known),missing_timers=missing,
                unreturned_task_rows=sum(r['status'].startswith('UNKNOWN_STARTED_') or r['status'].startswith('UNKNOWN_DISPATCHED_') for r in group),
                recorded_training_seconds=total(known) if group and not missing else None,
                observed_known_worker_seconds=total(known),recorded_training_scope=SCOPE,
                status='COMPLETE_RECORDED_VISIT_COVERAGE' if group and not missing else 'PARTIAL_MISSING_TIMER' if group else 'NO_RECORDED_VISITS_UNKNOWN',
                note='Canonical visits plus authenticated unreturned task gaps; explicit unattempted tasks are not performed work. Known subtotal excludes unknown work; repeated across scenarios, never summed across them.'))
    return out


def attach_training(methods,summaries):
    if len(methods)!=48 or len({r['case_id'] for r in methods})!=48:raise ValueError('Exact48 unique method rows required')
    lookup={(r['suite_id'],r['family']):r for r in summaries};result=[]
    for old in methods:
        row=dict(old);family=row['family'];network=row['network']
        if 'recorded_training_seconds' in row or 'recorded_training_scope' in row:raise ValueError('Do not overwrite prior timing columns')
        if network=='P' or family=='FFTT':value=0.;disposition='NO_INCREMENTAL_REGIONAL_FIT_BY_METHOD'
        elif family=='POWER4':value=None;disposition='INSEPARABLE_SHARED_INITIAL_AUDIT_AND_BASELINES'
        elif family in FAMILIES:
            entry=lookup[(row['suite_id'],family)];value=entry['recorded_training_seconds'];disposition=entry['status']
        else:raise ValueError('Unknown assignment family: '+family)
        row.update(recorded_training_seconds=value,recorded_training_scope=SCOPE,
                   recorded_training_disposition=disposition)
        result.append(row)
    return result


def stage_row(result,intent,source,task):
    start=result.get('started_epoch',result.get('start_epoch',intent.get('started_epoch')))
    end=result.get('ended_epoch',result.get('end_epoch'))
    # campaign.run_primary explicitly writes epoch at its terminal result boundary.
    if end is None and result.get('status') in {'POLICY_EXHAUSTED_OR_TARGETS_MET','PRIMARY_TIME_LIMIT'}:end=result.get('epoch')
    elapsed=seconds(result.get('elapsed_seconds'),'elapsed_seconds')
    if start is not None:start=seconds(start,'stage start')
    if end is not None:end=seconds(end,'stage end')
    if start is not None and end is not None:
        if end<start:raise ValueError('Stage terminal precedes start')
        if elapsed is None:elapsed=end-start
    return dict(task=task,category='RECOVERY_STAGE_WALL',suite_id='',region_key='',family='',seconds=elapsed,
        status='MEASURED' if elapsed is not None else 'UNAVAILABLE_TERMINAL_WALL_TIME',
        accounting_role='STAGE_SCOPE_DO_NOT_ADD_TO_NESTED_WORKER_OR_OTHER_OVERLAPPING_STAGES',time_kind='RECOVERY_MEASURED_WALL',
        timing_boundary='Producer-declared complete stage interval; nested workers, validation and pauses remain inside this scope.',
        start_epoch=start,end_epoch=end,source_fields='explicit elapsed_seconds or terminal/start epoch fields',
        note='File modification time and deadline are never substituted for actual completion.',**source)


def export(ctx,*,deadline,assignment_revision=None):
    ctx.require_action('publication');ctx.guard(deadline);start=time.time();own_sha=sha(__file__)
    root=Path(ctx.root);settings=ctx.c.get('offline_timing',{});layout=repair.resolve(ctx,assignment_revision)
    output=layout['timing_root'] if assignment_revision is not None else ctx.output_path(settings.get('output_relative','report_timing/v1'))
    if not output.is_relative_to(root/'report_timing'):raise ValueError('Dedicated new timing namespace required')
    if output.exists():raise FileExistsError('Timing export is append-only; choose a new version')
    bindings={};rows=[]
    def read(path,role,expected=None):
        ctx.guard(deadline);path=Path(path).resolve(strict=True)
        digest=ctx.bind(path,role,expected);bindings[str(path)]=dict(path=str(path),sha256=digest,bytes=path.stat().st_size)
        return read_json(path),dict(source_path=str(path),source_sha256=digest)
    ctx.bind(repair.__file__,'ASSIGNMENT_REVISION_RESOLVER_SOURCE',sha(repair.__file__))
    if assignment_revision is not None:
        read(assignment_revision['path'],'EXPLICIT_ASSIGNMENT_REVISION',assignment_revision['sha256'])
    manifest,_=read(layout['metrics_root']/'MANIFEST.json','TERMINAL_REPORT_DATA_MANIFEST')
    if manifest.get('assignment_revision')!=assignment_revision:raise ValueError('Timing/metrics revision differs')
    if manifest.get('status')!='VERIFIED_REPORT_DATA':raise ValueError('Terminal source-bound scientific export required')
    method_path=(layout['metrics_root']/'table2.csv').resolve();matched=[r for r in manifest['files'] if Path(r['path']).resolve()==method_path]
    if len(matched)!=1:raise ValueError('Exact method CSV binding required')
    digest=ctx.bind(method_path,'IMMUTABLE_METHOD_CSV',matched[0]['sha256']);bindings[str(method_path)]=dict(path=str(method_path),sha256=digest,bytes=method_path.stat().st_size)
    with method_path.open(newline='',encoding='utf-8') as stream:methods=list(csv.DictReader(stream))
    # Canonical visits are counted once. Authenticated pass metadata only reconciles gaps.
    visit_paths=sorted((root/'training/regions').glob('*/*/visits/*.json'));identities=set();visit_records=[];intent_records=[]
    for path in visit_paths:
        record,source=read(path,'RECOVERY_CANONICAL_VISIT')
        expected_key=path.parents[2].name+':'+path.parents[1].name
        if record.get('region_key')!=expected_key or int(path.stem)!=record.get('visit'):
            raise ValueError('Canonical visit path and region/ordinal disagree')
        identity=(record['region_key'],record['visit'])
        if identity in identities:raise ValueError('Duplicate canonical regional visit')
        identities.add(identity);visit_records.append((record,source));rows.extend(visit_rows(record,source))
    for path in sorted((root/'training/regions').glob('*/*/intents/*.json')):
        record,source=read(path,'RECOVERY_CANONICAL_VISIT_INTENT')
        expected_key=path.parents[2].name+':'+path.parents[1].name
        if record.get('region_key')!=expected_key or int(path.stem)!=record.get('visit'):
            raise ValueError('Canonical intent path and identity disagree')
        intent_records.append((record,source))
    pass_records=primary_pass_records(root,read)
    gap_rows,task_accounting=reconcile_training_tasks(visit_records,intent_records,pass_records)
    rows.extend(gap_rows)
    families=family_summaries(rows,ctx.c['suite_ids']);augmented=attach_training(methods,families)
    for prefix in ('execution','controller'):
        for path in sorted((root/prefix).glob('*/RESULT.json')):
            record,source=read(path,'RECOVERY_STAGE_RESULT');intent_path=path.with_name('INTENT.json')
            intent=read(intent_path,'RECOVERY_STAGE_INTENT')[0] if intent_path.exists() else {}
            rows.append(stage_row(record,intent,source,prefix+'/'+path.parent.name))
    # Parse each compact current-producer JSON, retaining scalar fields only.
    # labels._save_attempt stores solver history in label.npz; no NPZ/history is read here.
    patterns=['labels/overlays/*/*/labels/*/*/attempt_*/START.json',
              'labels/initial/*/*/labels/*/*/attempt_*/START.json','holdout/results/*/*/attempts/*/START.json']
    seen=set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path in seen:raise ValueError('Duplicate label attempt in accounting patterns')
            seen.add(path);begin,source=read(path,'NEW_LABEL_ATTEMPT_START');receipt=path.with_name('RESULT.json')
            record,source=read(receipt,'NEW_LABEL_PRODUCER_SCALAR_RECEIPT') if receipt.exists() else ({},source)
            if record and record.get('role')!=begin.get('role'):raise ValueError('Label timing role changed between start and receipt')
            value=seconds(record.get('complete_label_elapsed_seconds'),'complete_label_elapsed_seconds')
            role=record.get('role',begin.get('role','UNKNOWN'));relative=path.relative_to(root).parts
            sid=relative[2];rid=relative[3]
            rows.append(dict(task=str(path.parent.relative_to(root)),category='NEW_LABEL_PRODUCER_'+role,suite_id=sid,
                region_key=sid+':'+rid,family='',seconds=value,status=record.get('status','MISSING_COMPLETE_PRODUCER_TIMER'),
                accounting_role='SHARED_LABEL_WORKER_NOT_ADDITIVE_TO_STAGE_WALL',time_kind='RECOVERY_WORKER_SECONDS',
                timing_boundary='Regional solver complete-label timer; excludes surrounding file serialization and parent orchestration.',
                start_epoch=record.get('complete_label_start_epoch'),source_fields='complete_label_elapsed_seconds',
                note='Every started attempt retained, successful or failed; missing timer is unknown, not zero.',**source))
    for path in sorted((root/'ablation/fits').glob('*/RESULT.json')):
        record,source=read(path,'ABLATION_FIT_TIMER');task=record['task']
        rows.append(dict(task=task['task_id'],category='ABLATION_FIT_WORKER',suite_id='s03',region_key=task['region_key'],family=task['family'],
            seconds=seconds(record.get('elapsed_seconds'),'ablation elapsed'),status=record['status'],
            accounting_role='SEPARATE_EXPERIMENT_WORKER_NOT_PRODUCTION_FAMILY_TIME',time_kind='RECOVERY_WORKER_SECONDS',
            timing_boundary='Complete ablation fit task including its calibration/audit/export; never folded into production family timing.',
            start_epoch=record.get('started_epoch'),source_fields='elapsed_seconds',note='',**source))
    for seal_binding in settings.get('historical_seals',[]):
        seal,source=read(seal_binding['path'],'HISTORICAL_TIMING_SEAL',seal_binding['sha256'])
        if seal.get('status')!='VERIFIED':raise ValueError('Historical timing seal is not verified')
        old,old_source=read(seal['result']['path'],'HISTORICAL_SEALED_TIMING_RESULT',seal['result']['sha256'])
        if old.get('status')!='PASS_WITH_LIMITATIONS':raise ValueError('Unrecognized historical timing evidence')
        rows.append(dict(task='Prior campaign disjoint wall scopes',category='HISTORICAL_STAGE_WALL',suite_id='',region_key='',family='',
            seconds=seconds(old.get('additive_disjoint_seconds'),'historical disjoint wall'),status='HISTORICAL_REPORTED_WITH_LIMITATIONS',
            accounting_role='HISTORICAL_CONTEXT_NOT_RECOVERY_WORK',time_kind='HISTORICAL_MEASURED_WALL',
            timing_boundary='Previously sealed sum of declared disjoint historical intervals; not total historical offline time.',
            source_fields='sealed RESULT.additive_disjoint_seconds',note='No archival source-tree replay or complete offline amortization claim.',**old_source))
        timing_path=(Path(ctx.source_root)/'analysis/timing.csv').resolve()
        timing_matches=[item for item in old.get('inputs',[]) if Path(item['path']).resolve()==timing_path]
        if len(timing_matches)>1:raise ValueError('Duplicate historical timing input binding')
        if timing_matches:
            digest=ctx.bind(timing_path,'SEALED_HISTORICAL_TIMING_CSV',timing_matches[0]['sha256'])
            bindings[str(timing_path)]=dict(path=str(timing_path),sha256=digest,bytes=timing_path.stat().st_size)
            with timing_path.open(newline='',encoding='utf-8') as stream:old_rows=list(csv.DictReader(stream))
            old_source=dict(source_path=str(timing_path),source_sha256=digest)
        else:old_rows=old.get('historical_rows',[])
        for item in old_rows:
            rows.append(dict(task=item.get('task',''),category='HISTORICAL_'+item.get('category',''),suite_id=item.get('suite_id',''),region_key='',
                family=item.get('method',''),seconds=seconds(item.get('seconds'),'historical seconds'),status=item.get('status','UNKNOWN'),
                accounting_role='HISTORICAL_CONTEXT_NOT_RECOVERY_WORK',time_kind='HISTORICAL_'+item.get('time_kind',''),
                timing_boundary=item.get('timing_boundary',''),source_fields='sealed canonical timing.csv' if timing_matches else 'sealed RESULT.historical_rows',
                note=str(item.get('note',''))+' Original accounting role: '+str(item.get('accounting_role','')),**old_source))
    if not settings.get('historical_seals'):
        rows.append(dict(task='Historical offline contribution',category='HISTORICAL_OFFLINE_UNKNOWN',seconds=None,status='NO_EXPLICIT_SEALED_BINDING',
            accounting_role='UNKNOWN_NOT_ZERO',time_kind='UNAVAILABLE',timing_boundary='Historical work remains required context; no archive scan was authorized.'))
    for path in sorted((root/'resources').glob('*.json')):
        if path.name==f'{ctx.step_id}_{os.getpid()}.json':continue
        record,source=read(path,'OBSERVED_PROCESS_RESOURCE_SCOPE')
        peak=record.get('peak_rss_bytes')
        if peak is None:continue
        peak=seconds(peak,'peak RSS')
        rows.append(dict(task=path.stem,category='OBSERVED_PROCESS_TREE_PEAK',seconds=None,status='MEASURED_PROCESS_SCOPE',
            accounting_role='RESOURCE_METADATA_NEVER_SUM_OVERLAPPING_PROCESS_TREES',time_kind='BYTES',peak_process_tree_bytes=int(peak),
            timing_boundary='Maximum recorded process-tree RSS in this producer scope; not whole-system utilization.',
            source_fields='peak_rss_bytes',note='No CPU/GPU-hours or utilization inferred.',**source))
    ctx.guard(deadline);output.mkdir(parents=True,exist_ok=False)
    def write(name,values,fields=None):
        if fields is None:fields=list(dict.fromkeys(k for row in values for k in row))
        with (output/name).open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(values)
    _new(output/'TRAINING_TASK_RECONCILIATION.json',dict(canonical_visits=len(visit_records),canonical_intents=len(intent_records),unreturned_family_rows=len(gap_rows),tasks=task_accounting,unattempted_is_performed_work=False))
    write('timing_observations.csv',rows,FIELDS);write('family_training_timing.csv',families);write('methods_with_training_timing.csv',augmented)
    for item in bindings.values():ctx.bind(item['path'],'FINAL_TIMING_SOURCE_RECHECK',item['sha256'])
    if sha(__file__)!=own_sha:raise ValueError('Timing source changed during export')
    _new(output/'PROVENANCE.json',dict(schema=SCHEMA,source_sha256=own_sha,config_sha256=ctx.config_sha256,inputs=list(bindings.values()),assignment_revision=assignment_revision,
        original_payload_writes=False,model_or_label_array_reads=False,stage_and_worker_times_additive=False,
        historical_tree_scan=False,CPU_GPU_utilization_measured=False,
        label_receipt_read_scope='Whole compact current-producer JSON parsed transiently; scalar timing fields retained; compressed solver history and arrays never opened.'))
    result=dict(schema=SCHEMA,status='EXPORTED_WITH_EXPLICIT_TIMING_SCOPE_AND_MISSINGNESS',started_epoch=start,elapsed_seconds=time.time()-start,
        canonical_visits=len(visit_paths),canonical_intents=len(intent_records),unreturned_task_family_rows=len(gap_rows),primary_task_dispositions=len(task_accounting),started_label_attempts=len(seen),method_rows=48,observation_rows=len(rows),
        family_rows=families,missing_seconds_rows=sum(r.get('seconds') is None for r in rows),
        overall_offline_seconds=None,overall_offline_reason='Overlapping stage and nested worker scopes are deliberately not summed.')
    _new(output/'RESULT.json',result)
    _new(output/'MANIFEST.json',dict(schema=SCHEMA,status='VERIFIED_TIMING_EXPORT',assignment_revision=assignment_revision,files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(output.iterdir()) if p.is_file()]))
    ctx.flush_bindings();return result
