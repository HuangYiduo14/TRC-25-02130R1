"""Selected-only, hash-only inheritance before recovery portfolio admission.

No model/label arrays are decoded, fitted, or evaluated here. The final OLD
registry, not regional STATE or eligible candidates, supplies immutable audits.
Every inherited record still requires the ordinary numerical admission step.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time

from ..pacnhn_v1.registry_projection import _Lines, _decode, _signature, _FIELD, _value, MAX_VALUE_BYTES

SCHEMA = 'RECOVERY_SELECTED_ONLY_INHERITANCE_V1'
FAMILIES = ('FFTT', 'POWER4', 'ICNN', 'XGBOOST')
KINDS = {'FFTT':'DIRECT_COST_POWER4', 'POWER4':'DIRECT_COST_POWER4',
         'ICNN':'MONOTONE_ICNN_POWER4', 'XGBOOST':'XGBOOST_LOG_COST'}


def key(row):
    return f"{row['suite_id']}:{row['region_index']:06d}"


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _block(lines, first, indent):
    """Read one bounded pretty-JSON value at a known structural depth."""
    if first not in ('{', '['):
        return _decode(first.removesuffix(',')), first.endswith(',')
    closer = ' '*indent + ('}' if first == '{' else ']')
    block = [first]; size = 1
    while True:
        line = lines.next()
        if line in (closer, closer+','):
            block.append(line.removesuffix(','))
            return _decode('\n'.join(block)), line.endswith(',')
        if not line.startswith(' '*(indent+2)):
            raise ValueError('Unsupported committed-value indentation')
        size += len(line)+1
        if size > MAX_VALUE_BYTES:raise ValueError('Committed value exceeds bounded parser')
        block.append(line)


def _object(lines, first, indent, select=None):
    if first in ('{}','{},'):return {}, first.endswith(',')
    if first != '{':raise ValueError('Expected frozen object')
    pattern = re.compile('^'+' '*(indent+2)+r'("(?:[^"\\]|\\.)*"): (.*)$')
    out={}; seen=set(); more=True
    while more:
        match=pattern.fullmatch(lines.next())
        if match is None:raise ValueError('Malformed frozen object member')
        name, start=_decode(match[1]),match[2]
        if name in seen:raise ValueError('Duplicate frozen object member')
        seen.add(name)
        if indent==2 and name=='regions':
            value,more=_object(lines,start,4,select)
        else:
            value,more=_block(lines,start,indent+2)
        if select is None or indent==2 or name in select:out[name]=value
    close=lines.next();expected=' '*indent+'}'
    if close not in (expected,expected+','):raise ValueError('Malformed frozen object close')
    return out,close.endswith(',')


def project_registry(path, expected, wanted, *, guard=lambda:None):
    """Stream/hash the entire registry; retain only requested regional blocks."""
    if not _sha(expected):raise ValueError('Registry expected SHA is required')
    path=Path(path);before=_signature(path.stat());out={};seen=set();identities=set()
    with path.open('rb') as stream:
        if _signature(os.fstat(stream.fileno()))!=before:raise ValueError('Registry changed at open')
        lines=_Lines(stream,guard)
        if lines.next()!='{':raise ValueError('Unsupported registry serialization')
        more=True
        while more:
            field=_FIELD.fullmatch(lines.next())
            if field is None:raise ValueError('Malformed registry property')
            name,first=_decode(field[1]),field[2]
            if name in seen:raise ValueError('Duplicate registry property')
            seen.add(name)
            if name=='models':
                if first!='[':raise ValueError('Expected nonempty model inventory')
                models=[];next_record=True
                while True:
                    line=lines.next()
                    if line in ('  ]','  ],'):
                        if next_record:raise ValueError('Empty/trailing model array')
                        more=line.endswith(',');break
                    if not next_record or line!='    {':raise ValueError('Malformed model block')
                    row,next_record=_block(lines,'{',4)
                    identity=(key(row),row['family'])
                    if identity in identities:raise ValueError('Duplicate selected model identity')
                    identities.add(identity)
                    if identity[0] in wanted:models.append(row)
                out[name]=models
            elif name=='current_dataset_commitment':
                out[name],more=_object(lines,first,2,wanted)
            else:
                value,more=_value(lines,first,keep=name in ('schema','support_freeze_sha256','selection_frozen_epoch','test_opened_before_selection'))
                if value is not None:out[name]=value
        if lines.next()!='}' or stream.read(1):raise ValueError('Trailing registry bytes')
        if _signature(os.fstat(stream.fileno()))!=before:raise ValueError('Registry changed during read')
    if (_signature(path.stat())!=before or lines.bytes!=before[2]
            or lines.digest.hexdigest()!=expected):raise ValueError('Registry bytes differ from terminal commitment')
    commitment=out.get('current_dataset_commitment',{})
    if (out.get('schema')!='PACNHN_REGISTRY_V1' or out.get('test_opened_before_selection') is not False
            or commitment.get('schema')!='PACNHN_CURRENT_DATASET_COMMITMENT_V1'
            or commitment.get('TEST_arrays_opened') is not False):
        raise ValueError('Unsupported or unsealed final registry commitment')
    if not wanted.issubset(commitment['regions']):raise ValueError('Final commitment omits requested regions')
    guard();return out


def _path(path, root):
    p=Path(path).resolve(strict=True)
    if not p.is_file() or not p.is_relative_to(Path(root).resolve(strict=True)):
        raise ValueError('Inherited source escapes immutable source root')
    return p


def hash_file(path, expected, *, stop, deadline):
    if not _sha(expected):raise ValueError('Missing source digest')
    p=Path(path);before=_signature(p.stat());h=hashlib.sha256()
    with p.open('rb') as stream:
        if _signature(os.fstat(stream.fileno()))!=before:raise ValueError('Source changed before read')
        while True:
            if stop.is_set() or time.time()>=deadline:raise TimeoutError('Inheritance hash deadline')
            block=stream.read(1<<20)
            if not block:break
            h.update(block)
        if _signature(os.fstat(stream.fileno()))!=before:raise ValueError('Source changed during read')
    if _signature(p.stat())!=before or h.hexdigest()!=expected:raise ValueError('Inherited source SHA differs: '+str(p))
    return dict(path=str(p),sha256=expected,bytes=before[2],mtime_ns=before[3],role='SELECTED_ONLY_INHERITANCE')


class HashPool:
    """At most four source readers; Context/resource and binding writes stay on owner."""
    def __init__(self,ctx,deadline,workers):
        self.ctx=ctx;self.deadline=deadline;self.stop=threading.Event()
        self.pool=ThreadPoolExecutor(max_workers=workers);self.workers=workers;self.verified={}

    def __enter__(self):return self

    def __exit__(self,*exc):
        self.stop.set();self.pool.shutdown(wait=True,cancel_futures=True)

    def authenticate(self,files):
        pending={};items=iter(files.items());done=False
        try:
            while pending or not done:
                self.ctx.guard(deadline=self.deadline)
                while not done and len(pending)<2*self.workers:
                    try:p,digest=next(items)
                    except StopIteration:done=True;break
                    p=_path(p,self.ctx.source_root);name=str(p)
                    pin=self.ctx.source_expected.get(os.path.normcase(name))
                    if pin is not None and pin!=digest:raise ValueError('Checkpoint pin differs from selected commitment')
                    prior=self.verified.get(name)
                    if prior:
                        st=p.stat()
                        if prior['sha256']!=digest or (st.st_size,st.st_mtime_ns)!=(prior['bytes'],prior['mtime_ns']):
                            raise ValueError('Previously verified inherited source changed')
                        continue
                    pending[self.pool.submit(hash_file,p,digest,stop=self.stop,deadline=self.deadline)]=name
                completed,_=wait(pending,timeout=.2,return_when=FIRST_COMPLETED) if pending else ((),())
                for future in completed:
                    name=pending.pop(future);row=future.result();prior=self.ctx.bindings.get(name)
                    if prior and prior['sha256']!=row['sha256']:raise ValueError('Prior Context binding differs')
                    self.verified[name]=row;self.ctx.bindings[name]=row
        except BaseException:
            for future in pending:future.cancel()
            # A regional mismatch must retire its readers before the next task.
            wait(pending)
            raise


def authenticate_population(ctx, frozen, support, pool):
    region_key=key(support);base=Path(ctx.source_root)/'training/regions'/support['suite_id']/f"{support['region_index']:06d}"/'labels'
    overlay=ctx.c.get('recovery',{}).get('training_overlays',{}).get(region_key)
    if overlay and Path(overlay['path']).resolve()!=base.resolve():
        raise ValueError('Changed recovery label overlay: OLD cohort cannot be inherited')
    files={}
    for name,pfield,hfield in [('REQUEST_INDEX.json','request_index_path','request_index_sha256'),('REGION_IDENTITY.json','region_identity_path','region_identity_sha256')]:
        path=_path(frozen[pfield],ctx.source_root)
        if path!=(base/name).resolve():raise ValueError('Committed population path differs')
        files[str(path)]=frozen[hfield]
    if frozen.get('TEST_arrays_opened') is not False:raise ValueError('Population does not seal TEST')
    pool.authenticate(files)
    index=_decode((base/'REQUEST_INDEX.json').read_text(encoding='utf-8'))
    identity=_decode((base/'REGION_IDENTITY.json').read_text(encoding='utf-8'))
    if identity.get('region_id')!=region_key or identity.get('graph_sha256')!=support['sha256']:
        raise ValueError('Region graph identity differs from frozen support')
    sources={};counts={role:0 for role in ('TRAIN','VALID','TEST')}
    for input_id,row in index.items():
        role=row.get('role');selected=row.get('selected_label')
        if role not in counts or not _sha(input_id):raise ValueError('Invalid request role/input identity')
        sample=hashlib.sha256(f"{identity['graph_identity']}|{role}|{input_id}".encode()).hexdigest()
        if row.get('sample_id')!=sample:raise ValueError('Invalid canonical role/sample identity')
        if selected is None:continue
        counts[role]+=1
        if role=='TEST':continue  # Never open TEST receipt or target.
        for pfield,hfield in [('path','sha256'),('receipt','receipt_sha256')]:
            path=_path(selected[pfield],ctx.source_root)
            allowed=(base/'labels'/role/sample).resolve()
            if not path.is_relative_to(allowed):raise ValueError('Selected TRAIN/VALID source has foreign role/path')
            sources[str(path)]=selected[hfield]
    committed={str(_path(p,ctx.source_root)):h for p,h in frozen['source_files'].items()}
    if sources!=committed or counts!=frozen['admitted_counts']:raise ValueError('Committed selected population differs from current request index')
    for role in ('TRAIN','VALID'):
        if counts[role]<=0 or frozen[role]['unique_rows']!=counts[role] or not _sha(frozen[role]['dataset_identity']):
            raise ValueError('Missing exact nonempty current TRAIN/VALID cohort')
    pool.authenticate(sources)
    pool.authenticate(files)  # Metadata still matches after payload checksum reads.
    return identity


def normalize(row,population,support,provenance):
    family=row['family'];train=population['TRAIN']['dataset_identity'];valid=population['VALID']['dataset_identity']
    if (row.get('status')!='AVAILABLE' or row.get('model_kind')!=KINDS.get(family)
            or row.get('graph_sha256')!=support['sha256'] or row.get('pairs')!=support['pairs']
            or not _sha(row.get('pair_keys_identity')) or not _sha(row.get('model_sha256'))):
        raise ValueError('Selected model graph/order/status metadata is invalid')
    if row.get('current_train_dataset_identity')!=train or row.get('current_valid_dataset_identity')!=valid:
        raise ValueError('Current selected cohort differs')
    if row.get('train_rows')!=population['TRAIN']['unique_rows'] or row.get('valid_rows')!=population['VALID']['unique_rows']:
        raise ValueError('Selected row counts differ from committed cohort')
    scores=[row.get('TRAIN_WMAPE_percent'),row.get('VALID_WMAPE_percent')]
    if family!='FFTT':
        if (row.get('train_dataset_identity')!=train or row.get('valid_dataset_identity')!=valid
                or not all(_finite(v) for v in scores)):
            raise ValueError('Selected audit is stale or nonfinite')
        if family in ('ICNN','XGBOOST') and scores[0]>=1:raise ValueError('Inherited learned TRAIN target is unmet')
    out=copy.deepcopy(row);params=row.get('parameters') or {}
    out.update(train_wmape_percent=scores[0],valid_wmape_percent=scores[1],
        audit_train_dataset_identity=train if family!='FFTT' else None,
        audit_valid_dataset_identity=valid if family!='FFTT' else None,
        audit_model_sha256=row['model_sha256'],property_status='PENDING',property_model_sha256=None,
        property_scope='MANDATORY_NEW_NUMERICAL_ADMISSION',c0_identity=None,
        c0_identity_scope='NOT_DECODED_HERE_REQUIRES_EXACT_MODEL_GRAPH_ADMISSION',
        fit_sanity_status='PASS' if family!='FFTT' else 'NOT_MEASURED',
        fit_sanity_scope='FINAL_HASH_BOUND_CURRENT_TRAIN_VALID_AUDIT' if family!='FFTT' else 'NO_INVENTED_FFTT_VALID_SCORE',
        width=params.get('width'),total_updates=params.get('total_updates'),
        inherited_selected=True,inheritance_provenance=copy.deepcopy(provenance),TEST_opened=False)
    if family=='XGBOOST':
        raw=row.get('uncompressed_model_bytes');native=row.get('native_parameter_bytes');booster=row.get('uncompressed_booster_bytes')
        if any(type(v) is not int or v<0 for v in (raw,native,booster)) or raw<native+booster:
            raise ValueError('XGB complete raw-memory forecast is missing/inconsistent')
        out['forecast_resident_booster_bytes']=3*raw
        out['memory_forecast_basis']='THREE_TIMES_FROZEN_RAW_UBJ_PLUS_NATIVE_ARRAY_BYTES'
    return out


def _pinned(ctx,path,role):
    p=Path(path).resolve(strict=True);expected=ctx.source_expected.get(os.path.normcase(str(p)))
    if not _sha(expected):raise ValueError('Independent checkpoint pin is required: '+str(p))
    ctx.bind(p,role,expected);return _decode(p.read_text(encoding='utf-8')),expected


def inherit_selected(ctx,new_records,*,deadline,max_workers=4):
    """Add only missing region/family selections; invalid proof remains fallback."""
    if type(max_workers) is not int or not 1<=max_workers<=4:raise ValueError('One to four hash readers required')
    ctx.require_action('portfolio');ctx.guard(deadline=deadline)
    for rel in ('training/MODEL_SELECTION_FREEZE.json','training/REGISTRY_FREEZE.json','holdout/TEST_OPENED.json','NATIONAL_TIMING_ACTIVE.json'):
        if (ctx.root/rel).exists():raise PermissionError('Inheritance is pre-freeze, without TEST or national work')
    primary=ctx.root/'training/PRIMARY_RESULT.json'
    if not primary.exists():raise PermissionError('Primary training must terminate before inheritance')
    terminal=_decode(primary.read_text(encoding='utf-8'))
    if terminal.get('status') not in ('POLICY_EXHAUSTED_OR_TARGETS_MET','PRIMARY_TIME_LIMIT'):
        raise PermissionError('Primary lacks a valid terminal disposition')
    ctx.bind(primary,'TERMINAL_PRIMARY_BEFORE_INHERITANCE')
    started=time.time();seen=set();records=copy.deepcopy(new_records);new_cohorts={}
    for row in records:
        k=(key(row),row['family'])
        if k in seen:raise ValueError('Duplicate NEW region/family record')
        seen.add(k)
        current=(row.get('current_train_dataset_identity'),row.get('current_valid_dataset_identity'))
        if k[0] in new_cohorts and new_cohorts[k[0]]!=current:raise ValueError('NEW families disagree on current cohort')
        new_cohorts[k[0]]=current
    support,support_sha=_pinned(ctx,ctx.source_root/'SUPPORT_FREEZE.json','PINNED_SUPPORT')
    if support.get('status')!='PASS':raise ValueError('Support is not admitted')
    regions={key(r):r for s in support['suites'] for r in s['region_records']}
    if not regions or any(k not in regions or f not in FAMILIES for k,f in seen):raise ValueError('Records outside frozen support')
    wanted={k for k in regions if any((k,f) not in seen for f in FAMILIES)}
    result,result_sha=_pinned(ctx,ctx.source_root/'training/RESULT.json','PINNED_TERMINAL_TRAINING')
    if result.get('status')!='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE':raise ValueError('OLD training is not terminal')
    registry_path=ctx.source_root/'training/REGISTRY_FREEZE.json'
    if Path(result['registry']).resolve()!=registry_path.resolve():raise ValueError('Foreign final registry')
    projected=project_registry(registry_path,result['registry_sha256'],wanted,guard=lambda:ctx.guard(deadline=deadline))
    if projected['support_freeze_sha256']!=support_sha:raise ValueError('Registry support differs')
    csv=ctx.source_root/'training/LEARNING_RESULTS.csv'
    ctx.bind(csv,'SEALED_LEARNING_CSV_HASH_ONLY_NO_TEST_COLUMNS_PARSED',result['learning_csv_sha256'])
    provenance=dict(terminal_result=dict(path=str(ctx.source_root/'training/RESULT.json'),sha256=result_sha),
        registry=dict(path=str(registry_path),sha256=result['registry_sha256']),
        learning_csv=dict(path=str(csv),sha256=result['learning_csv_sha256']),support_sha256=support_sha,
        source='FINAL_SELECTED_ONLY_CURRENT_DATASET_COMMITMENT',fresh_fit=False,TEST_values_read=False)
    by_region={k:{} for k in wanted}
    for row in projected['models']:by_region[key(row)][row['family']]=row
    dispositions=[]
    with HashPool(ctx,deadline,max_workers) as pool:
        for k in sorted(wanted):
            candidates=by_region[k];missing=[f for f in FAMILIES if (k,f) not in seen]
            available=[f for f in missing if candidates.get(f,{}).get('status')=='AVAILABLE']
            # No trained current population: implicit analytic FFTT remains available in portfolio.
            population=projected['current_dataset_commitment']['regions'][k]
            if not population.get('TRAIN',{}).get('dataset_identity') or not population.get('VALID',{}).get('dataset_identity'):
                dispositions.extend(dict(region_key=k,family=f,status='NO_CURRENT_TRAIN_VALID_PROOF_IMPLICIT_FFTT') for f in missing);continue
            try:
                if k in new_cohorts and new_cohorts[k]!=(population['TRAIN']['dataset_identity'],population['VALID']['dataset_identity']):
                    raise ValueError('NEW family proves a different current cohort')
                authenticate_population(ctx,population,regions[k],pool)
            except (ValueError,OSError,KeyError,TimeoutError) as exc:
                dispositions.extend(dict(region_key=k,family=f,status='NOT_INHERITED',error_type=type(exc).__name__,reason=str(exc)) for f in missing);continue
            for family in missing:
                try:
                    ctx.guard(deadline=deadline)
                    if family not in available:raise ValueError('No available final selected model')
                    row=candidates[family];normalized=normalize(row,population,regions[k],provenance)
                    artifacts=row.get('artifact_hashes',{})
                    if artifacts!={row['model_path']:row['model_sha256']}:raise ValueError('Selected artifact inventory differs')
                    pool.authenticate(artifacts)
                    if row.get('reassessment_receipt_path'):
                        pool.authenticate({row['reassessment_receipt_path']:row['reassessment_receipt_sha256']})
                    records.append(normalized)
                    dispositions.append(dict(region_key=k,family=family,status='INHERITED_PENDING_NUMERICAL_ADMISSION',model_sha256=row['model_sha256']))
                except (ValueError,OSError,KeyError,TimeoutError) as exc:
                    dispositions.append(dict(region_key=k,family=family,status='NOT_INHERITED',error_type=type(exc).__name__,reason=str(exc)))
        verified_count=len(pool.verified);verified_bytes=sum(r['bytes'] for r in pool.verified.values())
    evidence=dict(schema=SCHEMA,status='COMPLETE_WITH_EXPLICIT_INHERITANCE_DISPOSITIONS',started_epoch=started,
        ended_epoch=time.time(),deadline_epoch=deadline,new_records_preserved=len(new_records),inherited_records=len(records)-len(new_records),
        dispositions=dispositions,source=provenance,source_bindings={str(Path(__file__).resolve()):ctx.bind(__file__,'INHERITANCE_SOURCE')},
        verified_source_files=verified_count,verified_source_bytes=verified_bytes,max_hash_workers=max_workers,
        fitting_performed=False,eligible_pool_read=False,model_arrays_decoded=False,label_arrays_decoded=False,TEST_values_read=False,
        limitations=['No fresh accuracy evaluation; current final audits reused only with exact source-byte commitments.',
            'Property/c0/order checks remain mandatory in subsequent numerical admission.',
            'FFTT VALID is unknown in OLD registry; no baseline comparison score is fabricated.'])
    return records,evidence
