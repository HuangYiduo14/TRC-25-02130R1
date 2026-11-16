"""Explicit post-Stage3 M02 validation. Importing this module never loads models.

Analytic guarantees follow from authenticated parameter constraints, not sampling.
The finite checks below concern the production evaluator on declared synthetic
points only. They do not measure TRAIN/VALID/TEST fit or physical accuracy.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

SCHEMA = 'PACNHN_FINAL_ICNN_GUARANTEES_V1'
TASKS = ['AUTHENTICATE_SELECTED_MODELS', 'SYNTHETIC_PILOT', 'ALL_SELECTED_PROPERTIES', 'SEAL_EVIDENCE']
OUTPUTS = ['INTENT.json', 'PILOT_RESULT.json', 'MODEL_PROPERTY_RESULTS.csv', 'RESULT.json', 'REPORT.md', 'ARTIFACT_MANIFEST.json', 'GUARANTEES_SYNC.zip']
PARAMETERS = dict(schema='PACNHN_M02_NUMERICS_V1', fd_steps=[1e-4, 1e-5, 1e-6],
    gradient_rtol=2e-6, hvp_rtol=2e-5, roundoff_multiplier=64,
    dot_roundoff_multiplier=32, convergence_slack=1.5,max_roundoff_fraction=.01,
    constant_x=[0., 1e-12, 1e-6, 1., 10.], mixed_x=[0., .25, 1., 4.],
    signed_direction_formula='alternating +/-1; cos((i+1)*sqrt(2)), normalized by max(abs)',
    coordinate_columns='unique(first,middle,last)', pilot_count=8,
    fd_failure_policy='RESOLVED_MISMATCH_FAIL_UNRESOLVED_ONLY_INCONCLUSIVE',
    pilot_policy='CONTINUE_INCONCLUSIVE_WITHIN_ORIGINAL_RESOURCE_AND_TIME_BUDGET',
    numerical_scope='FINITE_SYNTHETIC_SAMPLING_NOT_GLOBAL_OR_ACCURACY_PROOF')
BUDGET = dict(seconds=600, sealing_reserve_seconds=30, process_rss_bytes=1024**3,
    tree_rss_bytes=24*1024**3, minimum_available_bytes=3*1024**3,
    max_npz_uncompressed_bytes=128*1024**2, processes=1, threads=1)
REQUIRED_SOURCES = ['final_icnn_guarantees.py', 'models.py', 'registry_projection.py',
                    'continuation.py', 'handoff.py', 'tests/test_final_icnn_guarantees.py']
TRAINING_TERMINAL = 'STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def file_sha(path, guard=lambda: None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            guard(); b = f.read(1024**2)
            if not b: break
            h.update(b)
    return h.hexdigest()


def read_json(path, maximum=4*1024**2):
    p = Path(path)
    if p.stat().st_size > maximum: raise ValueError('Oversize metadata: '+str(p))
    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d: raise ValueError('Duplicate JSON property: '+k)
            d[k] = v
        return d
    return json.loads(p.read_text(encoding='utf-8'), object_pairs_hook=unique,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def new_json(path, value):
    def scalar(item):
        import numpy as np
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError('Unsupported audit JSON value: '+type(item).__name__)
    # Serialize before opening: invalid metadata must not create an empty result.
    content=json.dumps(value, indent=2, allow_nan=False, default=scalar)+'\n'
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(content)


def bound(binding, *, guard=lambda: None, maximum=None):
    if not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}:
        raise ValueError('Binding requires exactly path and sha256')
    p = Path(binding['path']).resolve(strict=True)
    expected = binding['sha256']
    if not isinstance(expected, str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Invalid SHA256')
    if not p.is_file() or (maximum is not None and p.stat().st_size > maximum):
        raise ValueError('Invalid bound file')
    if file_sha(p, guard) != expected: raise ValueError('Changed bound file: '+str(p))
    return p


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_stage3(result, matrix, outcomes):
    if result.get('status')!='COMPLETE_WITH_RECORDED_OUTCOMES' or result.get('primary_rows')!=14:
        raise ValueError('Terminal Stage3 disposition evidence required')
    if not isinstance(matrix, list) or not isinstance(outcomes, list) or result.get('rows')!=len(matrix):
        raise ValueError('Stage3 complete matrix mismatch')
    ids = [r['case_id'] for r in matrix]
    if len(set(ids))!=len(ids) or len(outcomes)!=len(ids): raise ValueError('Duplicate/missing Stage3 cases')
    expected = {f'P_{solver}_{s}' for solver in ('CFW','FW','MSA') for s in ('open','i95')}
    expected |= {f'AC_s03_{family}_{s}' for family in ('FFTT','POWER4','XGBOOST','ICNN') for s in ('open','i95')}
    if {r['case_id'] for r in matrix if r.get('priority')=='PRIMARY'}!=expected:
        raise ValueError('Required primary matrix differs')
    found = {}
    for row in outcomes:
        case = row.get('case', {}); key = case.get('case_id')
        if key in found or key not in ids or not isinstance(row.get('status'),str) or not row['status']:
            raise ValueError('Missing terminal Stage3 outcome')
        if row['status'] in ('RUNNING','PENDING','EXECUTING'): raise ValueError('Stage3 still live')
        if case != matrix[ids.index(key)]: raise ValueError('Stage3 case identity changed')
        found[key] = row


def _preflight_has(record, path, digest):
    if isinstance(record, dict):
        if record.get('sha256')==digest and str(record.get('path','')).replace('\\','/').lower()==str(path).replace('\\','/').lower(): return True
        for k,v in record.items():
            if isinstance(v,str) and v==digest and k.replace('\\','/').lower()==str(path).replace('\\','/').lower(): return True
            if _preflight_has(v,path,digest): return True
    elif isinstance(record,list):
        return any(_preflight_has(x,path,digest) for x in record)
    return False


def admit(config_path, config_sha256, admission_path, *, now=None):
    """Metadata-only authority gate. No model read and no output mutation."""
    observed = time.time() if now is None else now
    cp = bound({'path':str(config_path),'sha256':config_sha256}, maximum=1024**2)
    c = read_json(cp); a = read_json(admission_path)
    if c.get('schema')!=SCHEMA or c.get('phase_id')!='PACNHN_V1' or c.get('step_id')!='M02_FINAL_GUARANTEES':
        raise ValueError('Explicit M02 phase/step contract required')
    if c.get('tasks')!=TASKS or c.get('expected_outputs')!=OUTPUTS or not c.get('acceptance_tests'):
        raise ValueError('Tasks, outputs and acceptance tests must be explicit')
    if canonical(c.get('numerics'))!=canonical(PARAMETERS) or canonical(c.get('budget'))!=canonical(BUDGET):
        raise ValueError('Frozen numerical/resource policy differs')
    if c.get('expected_models')!=1499 or c.get('suite_id')!='s03' or c.get('family')!='ICNN':
        raise ValueError('Exact final 1499-model s03 ICNN population required')
    if (a.get('schema')!='PACNHN_M02_ADMISSION_V1' or a.get('decision')!='ADMIT_POST_STAGE3_PROPERTY_AUDIT'
        or a.get('config_sha256')!=config_sha256 or a.get('phase_id')!=c['phase_id'] or a.get('run_id')!=c.get('run_id')):
        raise ValueError('Root operational admission must bind exact configuration')
    campaign_path = bound(c['campaign_config'], maximum=1024**2); campaign = read_json(campaign_path)
    if c['run_id']!=campaign['run_id']: raise ValueError('Campaign run differs')
    from .continuation import validate_continuation
    from .handoff import publication_window
    auth = validate_continuation(campaign, action='analysis', now=observed)
    if auth is None: raise ValueError('Authenticated continuation required')
    for key in ('training_result','registry'):
        pinned=campaign['continuation']
        if (Path(c[key]['path']).resolve()!=Path(pinned[key+'_path']).resolve()
            or c[key]['sha256']!=pinned[key+'_sha256']):
            raise ValueError('Audit must use the continuation-pinned '+key)
    window = publication_window(campaign, campaign_config_sha256=c['campaign_config']['sha256'],
        stage4_clock=c['stage4_clock'], clock_amendment=c['clock_amendment'], deadline_epoch=c['deadline_epoch'],now=observed)
    root=Path(campaign['output_root']).resolve(strict=True)
    shared_path=bound(c['shared_audit_clock'],maximum=1024**2)
    shared=read_json(shared_path)
    canonical_shared=root/'final_audits/AUDIT_CLOCK.json'
    if (Path(c['shared_audit_clock']['path']).absolute()!=canonical_shared or shared_path!=canonical_shared
        or Path(c['shared_audit_clock']['path']).is_symlink()
        or (hasattr(Path(c['shared_audit_clock']['path']),'is_junction') and Path(c['shared_audit_clock']['path']).is_junction())):
        raise ValueError('Canonical shared audit clock required')
    if (shared.get('schema')!='PACNHN_FINAL_AUDIT_CLOCK_V1' or shared.get('phase_id')!=c['phase_id']
        or shared.get('run_id')!=c['run_id'] or shared.get('stage4_clock_sha256')!=c['stage4_clock']['sha256']
        or not finite(shared.get('start_epoch')) or not finite(shared.get('deadline_epoch'))
        or not window['stage_start_epoch']<=shared['start_epoch']<=observed
        or shared['deadline_epoch']!=min(shared['start_epoch']+1800,window['stage_deadline_epoch'])):
        raise ValueError('Shared audit clock identity/window differs')
    audit_deadline=min(shared['start_epoch']+BUDGET['seconds'],shared['deadline_epoch'],window['deadline_epoch'])
    if observed>=audit_deadline-BUDGET['sealing_reserve_seconds']:raise TimeoutError('Original shared M02 audit allowance exhausted')
    output=Path(c['output_root']).absolute()
    if output.resolve()!=output or not output.is_relative_to(root/'audits') or output==root/'audits':
        raise ValueError('Output must be an unaliased new audit directory')
    for p in [output,*output.parents]:
        if p.exists() and (p.is_symlink() or (hasattr(p,'is_junction') and p.is_junction())):
            raise ValueError('Aliased output path')
    stage3 = read_json(bound(c['stage3_result'],maximum=1024**2))
    matrix=read_json(bound(c['stage3_matrix'])); outcomes=read_json(bound(c['stage3_outcomes'],maximum=32*1024**2))
    validate_stage3(stage3,matrix,outcomes)
    stage3_clock=read_json(bound(c['stage3_clock'],maximum=1024**2))
    if Path(c['stage3_clock']['path']).resolve()!=root/'STAGE3_CLOCK.json' or stage3.get('clock')!=stage3_clock:
        raise ValueError('Original Stage3 clock binding differs')
    sources=c.get('source_bindings'); base=Path(__file__).resolve().parent
    if not isinstance(sources,dict) or set(sources)!=set(REQUIRED_SOURCES): raise ValueError('Exact source closure required')
    preflight=read_json(bound(c['preflight'],maximum=4*1024**2))
    if preflight.get('status')!='PASS' or preflight.get('exit_code')!=0 or preflight.get('inputs_stable_during_tests') is not True:
        raise ValueError('Passed stable preflight required')
    if preflight.get('config_sha256')!=c['campaign_config']['sha256']:
        raise ValueError('Preflight belongs to another campaign configuration')
    if a.get('preflight_sha256')!=c['preflight']['sha256']: raise ValueError('Admission preflight differs')
    for rel,binding in sources.items():
        p=bound(binding,maximum=1024**2)
        if p!=(base/rel).resolve() or not _preflight_has(preflight,p,binding['sha256']):
            raise ValueError('Source missing from exact preflight closure: '+rel)
    proofs=c.get('proof_bindings')
    if not isinstance(proofs,list) or len(proofs)<2: raise ValueError('Proof and proof receipt bindings required')
    for b in proofs: bound(b,maximum=2*1024**2)
    if not isinstance(c.get('other_owned_processes'),list): raise ValueError('Explicit additional owned process inventory required')
    return c, {'root':str(root),'output':str(output),'config_sha256':config_sha256,'admission_sha256':file_sha(admission_path),
        'authority':auth,'stage4_window':window,'shared_audit_clock':shared,
        'audit_deadline_epoch':audit_deadline,'admitted_epoch':observed}


def reconcile_inventory(manifest_rows, registry_rows, inference_bindings, selected_rows, *, expected_count=1499):
    """Compare exact IDs and hashes; never infer model identity from filenames alone."""
    reg={}
    for row in registry_rows:
        if row.get('suite_id')=='s03' and row.get('family')=='ICNN':
            rid=row['region_index']
            if type(rid) is not int or rid in reg or row.get('status')!='AVAILABLE': raise ValueError('Registry ICNN coverage differs')
            if row.get('model_kind')!='MONOTONE_ICNN_POWER4': raise ValueError('Wrong selected model kind')
            reg[rid]=row
    infer={}
    for row in inference_bindings:
        if row.get('role')=='FROZEN_ICNN_COST_MODEL':
            p=str(Path(row['path']).resolve())
            if p in infer: raise ValueError('Duplicate ICNN inference binding')
            infer[p]=row['sha256']
    selected={}
    for row in selected_rows:
        if row.get('suite_id')=='s03' and row.get('family')=='ICNN':
            rid=int(row['region_index'])
            if rid in selected or row.get('status')!='AVAILABLE': raise ValueError('Selected metadata coverage differs')
            selected[rid]=row['model_sha256']
    result=[];seen=set()
    for row in manifest_rows:
        if row.get('suite_id')!='s03': raise ValueError('Unexpected manifest suite')
        rid=int(row['region_index']);p=str(Path(row['model_path']).resolve())
        if rid in seen or rid not in reg: raise ValueError('Manifest duplicate/missing model')
        seen.add(rid);rr=reg[rid];digest=row['model_sha256']
        if (p!=str(Path(rr['model_path']).resolve()) or digest!=rr['model_sha256']
            or infer.get(p)!=digest or selected.get(rid)!=digest): raise ValueError('Selected model authentication mismatch')
        artifacts=rr.get('artifact_hashes',{})
        if not isinstance(artifacts,dict) or len(artifacts)!=1 or {str(Path(k).resolve()):v for k,v in artifacts.items()}!={p:digest}:
            raise ValueError('Only the selected self-contained NPZ artifact is permitted')
        result.append(dict(region_index=rid,model_path=p,model_sha256=digest,
            recorded_compressed_bytes=int(row['recorded_compressed_bytes']),artifact_hashes=artifacts))
    if len(result)!=expected_count or seen!=set(reg) or seen!=set(selected) or len(infer)!=expected_count:
        raise ValueError('Not the exact selected model population')
    return sorted(result,key=lambda r:r['region_index'])


def _array_digest(a):
    import numpy as np
    v=np.ascontiguousarray(a,dtype=np.float64)
    return hashlib.sha256(str(v.shape).encode()+v.tobytes()).hexdigest()


def _fd_pair_accept(rows, which):
    """Two adjacent accurate steps, with refinement no worse than roundoff permits."""
    for a,b in zip(rows,rows[1:]):
        x,y=a[which],b[which]
        if (x['pass'] and y['pass'] and y['absolute_error'] <= max(
                PARAMETERS['convergence_slack']*x['absolute_error'],4*x['roundoff_bound'],4*y['roundoff_bound'])):
            return True
    return False


def fd_disposition(rows,which):
    """Inconclusive means insufficient resolution, never a resolved contradiction."""
    if _fd_pair_accept(rows,which):return 'PASS'
    if any(r[which]['signal_resolved'] and not r[which]['pass'] for r in rows):return 'FAIL'
    if any(a[which]['signal_resolved'] and b[which]['signal_resolved'] for a,b in zip(rows,rows[1:])):
        return 'FAIL'  # Resolved steps failed the predeclared refinement criterion.
    return 'NUMERIC_INCONCLUSIVE'


def audit_model(model, *, guard=lambda:None):
    """Finite synthetic property checks; caller supplies one authenticated model.

    All tolerances scale with native magnitudes; no fixed one-cost-unit floor.
    No dense Jacobian/Hessian or eigenvalue calculation is performed.
    """
    import numpy as np
    from .models import PortableICNN
    if not isinstance(model,PortableICNN): raise TypeError('Only production PortableICNN')
    s=model.scale;m=len(s);eps=np.finfo(np.float64).eps
    round_factor=PARAMETERS['roundoff_multiplier']*eps
    dot_factor=PARAMETERS['dot_roundoff_multiplier']*(m+2*model.width+8)*eps
    checks=[];fd=[];vector_hashes={}
    def check(name,error,tolerance,**extra):
        if not all(math.isfinite(float(v)) for v in (error,tolerance)): raise FloatingPointError('Nonfinite diagnostic')
        checks.append(dict(name=name,absolute_error=float(error),tolerance=float(tolerance),passed=bool(error<=tolerance),**extra))
    def vc(x):
        guard();v,c=model.value_cost(s*x);guard()
        if np.asarray(v).dtype!=np.float64 or c.dtype!=np.float64: raise TypeError('Native float64 outputs required')
        return float(v),c
    def hx(x,z):
        guard();h=s*model.hess_vec(s*x,s*z);guard()
        if not np.isfinite(h).all(): raise FloatingPointError('Nonfinite scaled HVP')
        return h
    zeros=np.zeros(m);value,cost=vc(zeros)
    check('origin_value',abs(value),0.)
    check('origin_cost',float(np.max(np.abs(cost-model.c0))),0.)
    xs=[np.full(m,x) for x in PARAMETERS['constant_x']]
    xs.append(np.resize(np.asarray(PARAMETERS['mixed_x']),m))
    for index,x in enumerate(xs):
        value,cost=vc(x);vector_hashes[f'x{index}']=_array_digest(x)
        check(f'value_floor_{index}',max(0.,-value),round_factor*abs(value))
        floor_excess=np.maximum(model.c0-cost-round_factor*(np.abs(cost)+np.abs(model.c0)),0.)
        check(f'cost_floor_{index}',float(np.max(floor_excess)),0.,minimum_increment=float(np.min(cost-model.c0)))
    i=np.arange(m,dtype=np.float64)
    z1=np.where((i.astype(np.int64)%2)==0,1.,-1.)
    z2=np.cos((i+1)*np.sqrt(2.));z2/=np.max(np.abs(z2))
    positive=np.ones(m)
    for name,z in [('signed1',z1),('signed2',z2),('positive',positive)]:vector_hashes[name]=_array_digest(z)
    for point,x in [('interior',np.ones(m)),('mixed',xs[-1])]:
        h1,h2,hp=hx(x,z1),hx(x,z2),hx(x,positive)
        for name,z,h in [('signed1',z1,h1),('signed2',z2,h2)]:
            q=float(z@h);scale=float(np.abs(z*h).sum())
            check(f'quadratic_{point}_{name}',max(0.,-q),dot_factor*scale,quadratic=q)
        left,right=float(z1@h2),float(z2@h1)
        check(f'symmetry_{point}',abs(left-right),dot_factor*float(np.abs(z1*h2).sum()+np.abs(z2*h1).sum()))
        check(f'positive_HVP_{point}',float(np.max(np.maximum(-hp,0.))),0.)
    columns={}
    for j in sorted({0,m//2,m-1}):
        basis=np.zeros(m);basis[j]=1.;h=hx(np.ones(m),basis)
        check(f'column_nonnegative_{j}',float(np.max(np.maximum(-h,0.))),0.)
        columns[j]=h
    for j in columns:
        for k in columns:
            if j<k:check(f'column_symmetry_{j}_{k}',abs(float(columns[j][k]-columns[k][j])),
                dot_factor*float(abs(columns[j][k])+abs(columns[k][j])))
    del columns
    x=np.ones(m);value,cost=vc(x)
    for name,z in [('signed1',z1),('signed2',z2),('positive',positive)]:
        expected_g=float(cost@(s*z));g_scale=float(np.abs(cost*s*z).sum());expected_h=hx(x,z)
        h_scale=float(np.max(np.abs(expected_h)));steps=[]
        for step in PARAMETERS['fd_steps']:
            vp,cp=vc(x+step*z);vm,cm=vc(x-step*z)
            numerical_g=(vp-vm)/(2*step);numerical_h=s*(cp-cm)/(2*step)
            gr=round_factor*(abs(vp)+abs(vm))/(2*step)+dot_factor*g_scale
            hr=round_factor*float(np.max(s*(np.abs(cp)+np.abs(cm))))/(2*step)+dot_factor*h_scale
            ge=abs(numerical_g-expected_g);he=float(np.max(np.abs(numerical_h-expected_h)))
            gt=PARAMETERS['gradient_rtol']*g_scale+gr;ht=PARAMETERS['hvp_rtol']*h_scale+hr
            if not all(math.isfinite(v) for v in (ge,he,gr,hr,gt,ht)):raise FloatingPointError('Nonfinite finite difference')
            # A large cancellation allowance is not evidence of consistency.
            # Exact zero/zero comparisons are retained for linear/inactive cases.
            # If the expected derivative is zero, a resolved observed nonzero
            # derivative must fail; this changes resolution, not its tolerance.
            g_signal=g_scale if g_scale>0 else abs(numerical_g)
            h_signal=h_scale if h_scale>0 else float(np.max(np.abs(numerical_h)))
            gres=(g_scale==0 and ge==0) or (g_signal>0 and gr<=PARAMETERS['max_roundoff_fraction']*g_signal)
            hres=(h_scale==0 and he==0) or (h_signal>0 and hr<=PARAMETERS['max_roundoff_fraction']*h_signal)
            steps.append(dict(step=step,gradient=dict(absolute_error=ge,roundoff_bound=gr,tolerance=gt,signal_resolved=gres,pass_=gres and ge<=gt),
                hvp=dict(absolute_error=he,roundoff_bound=hr,tolerance=ht,signal_resolved=hres,pass_=hres and he<=ht)))
            for field in ('gradient','hvp'):steps[-1][field]['pass']=steps[-1][field].pop('pass_')
        gs,hs=fd_disposition(steps,'gradient'),fd_disposition(steps,'hvp');fg,fh=gs=='PASS',hs=='PASS'
        checks.append(dict(name='FD_gradient_'+name,passed=fg,disposition=gs,criterion='two_adjacent_steps_and_refinement'))
        checks.append(dict(name='FD_HVP_'+name,passed=fh,disposition=hs,criterion='two_adjacent_steps_and_refinement'))
        fd.append(dict(direction=name,steps=steps,gradient_pass=fg,hvp_pass=fh,gradient_disposition=gs,hvp_disposition=hs))
    exported=model.export_arrays();rebuilt=PortableICNN(exported)
    if any(not np.array_equal(exported[k],v) for k,v in rebuilt.export_arrays().items()):raise ArithmeticError('Export reconstruction arrays differ')
    original_v,original_c=model.value_cost(s);new_v,new_c=rebuilt.value_cost(s)
    check('export_value',abs(float(original_v-new_v)),0.)
    check('export_cost',float(np.max(np.abs(original_c-new_c))),0.)
    check('export_HVP',float(np.max(np.abs(model.hess_vec(s,s*z1)-rebuilt.hess_vec(s,s*z1)))),0.)
    del rebuilt,exported
    failures=[x['name'] for x in checks if not x['passed']]
    status=('PASS' if not failures else ('NUMERIC_INCONCLUSIVE' if all(
        x.get('disposition')=='NUMERIC_INCONCLUSIVE' for x in checks if not x['passed']) else 'FAIL'))
    return dict(status=status,dimension=m,width=model.width,
        amplitude=model.amplitude,inactive_amplitude=model.amplitude==0,
        structural_constraints=dict(c0_min=float(model.c0.min()),scale_min=float(s.min()),eta_min=float(model.eta.min()),
            nonnegative_weight_min={k:float(model.arrays[k].min()) for k in ('A1','A2','W2','v','amplitude')},
            finite_and_shape_checked_by='PortableICNN production constructor'),
        checks=checks,finite_differences=fd,failed_checks=failures,vector_hashes=vector_hashes,
        analytic_scope='Intended smooth formula under bound finite nonnegative parameter constraints',
        numerical_scope=PARAMETERS['numerical_scope'],accuracy_evaluated=False)


def resource_observation(process, other_owned_processes=()):
    """Only inspect explicitly owned trees; never stop a different process."""
    import psutil
    memory=process.memory_info();rss=memory.rss
    # Windows peak_wset catches transient allocations between supervisor samples.
    process_peak=max(rss,getattr(memory,'peak_wset',rss))
    children=process.children(recursive=True)
    tree={process.pid:rss}
    for child in children:
        try:tree[child.pid]=child.memory_info().rss
        except psutil.NoSuchProcess:pass
    audit_rss=sum(tree.values())
    for identity in other_owned_processes:
        if set(identity)!={'pid','create_time'}:raise ValueError('Explicit owned PID/birth required')
        try:
            p=psutil.Process(identity['pid'])
            if abs(p.create_time()-identity['create_time'])>1e-6:raise ValueError('Owned process identity changed')
            for member in [p,*p.children(recursive=True)]:
                try:tree[member.pid]=member.memory_info().rss
                except psutil.NoSuchProcess:pass
        except psutil.NoSuchProcess:pass
    available=psutil.virtual_memory().available
    result=dict(audit_tree_rss_bytes=audit_rss,process_peak_rss_bytes=process_peak,aggregate_owned_rss_bytes=sum(tree.values()),available_bytes=available,
        sampled_epoch=time.time())
    if max(audit_rss,process_peak)>BUDGET['process_rss_bytes']:raise MemoryError('M02 one-process RSS cap')
    if result['aggregate_owned_rss_bytes']>BUDGET['tree_rss_bytes']:raise MemoryError('Aggregate owned RSS cap')
    if available<BUDGET['minimum_available_bytes']:raise MemoryError('Minimum available RAM guard')
    return result


def _csv_rows(path):
    if Path(path).stat().st_size>4*1024**2:raise ValueError('Oversize compact model metadata')
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def _inventory(c,guard):
    from .registry_projection import read_stage3_registry
    finished=read_json(bound(c['training_result'],guard=guard,maximum=1024**2))
    if finished.get('status')!=TRAINING_TERMINAL:raise ValueError('Frozen terminal Stage2 required')
    registry_path=Path(c['registry']['path']).resolve(strict=True)
    if c['registry']['sha256']!=finished.get('registry_sha256') or registry_path!=Path(finished['registry']).resolve():
        raise ValueError('Terminal selected registry differs')
    registry=read_stage3_registry(registry_path,c['registry']['sha256'],guard=guard)
    receipt_path=bound(c['inference_receipt'],guard=guard,maximum=4*1024**2);receipt=read_json(receipt_path)
    result=read_json(bound(c['inference_result'],guard=guard,maximum=1024**2))
    intent_path=bound(c['inference_intent'],guard=guard,maximum=4*1024**2)
    if (result.get('status')!='PASS' or receipt.get('status')!='PASS' or receipt.get('full_model') is not True
        or receipt.get('suite_id')!='s03' or receipt.get('family')!='ICNN'
        or receipt.get('registry_sha256')!=c['registry']['sha256']
        or result.get('registry_sha256')!=c['registry']['sha256']
        or result.get('inference_receipt_sha256')!=c['inference_receipt']['sha256']
        or Path(result['inference_receipt']).resolve()!=receipt_path
        or result.get('intent_sha256')!=c['inference_intent']['sha256']
        or receipt.get('intent_sha256')!=c['inference_intent']['sha256']):
        raise ValueError('Complete ICNN inference binding differs')
    intent=read_json(intent_path)
    if not _preflight_has(intent,Path(__file__).resolve().parent/'models.py',c['source_bindings']['models.py']['sha256']):
        raise ValueError('Inference used another model implementation')
    manifest=_csv_rows(bound(c['model_manifest'],guard=guard,maximum=4*1024**2))
    selected=_csv_rows(bound(c['selected_metadata'],guard=guard,maximum=4*1024**2))
    rows=reconcile_inventory(manifest,registry['models'],receipt['source_bindings'],selected)
    root=Path(read_json(c['campaign_config']['path'])['output_root']).resolve()
    for r in rows:
        if not Path(r['model_path']).is_relative_to(root/'training/regions/s03'):
            raise ValueError('Selected model outside frozen s03 training output')
    del registry,receipt,manifest,selected;gc.collect()
    return rows


def pilot_order(rows):
    by_id={r['region_index']:r for r in rows}
    largest=[r['region_index'] for r in sorted(rows,key=lambda r:(-r['recorded_compressed_bytes'],r['region_index']))[:3]]
    selected=list(dict.fromkeys(largest+[rid for rid in (0,96) if rid in by_id]))
    for rid in sorted(by_id):
        if len(selected)>=min(PARAMETERS['pilot_count'],len(rows)):break
        if rid not in selected:selected.append(rid)
    return [by_id[rid] for rid in selected]+[by_id[rid] for rid in sorted(by_id) if rid not in selected]


def pilot_continuation(completed,forecast_seconds,remaining_seconds):
    states=[r['status'] for r in completed]
    if not states or any(x not in ('PASS','NUMERIC_INCONCLUSIVE') for x in states):
        return dict(status='FAIL',continue_main=False,reason='HARD_FAILURE_OR_RESOURCE_LIMIT')
    if not finite(forecast_seconds) or not finite(remaining_seconds) or forecast_seconds>remaining_seconds or remaining_seconds<=0:
        return dict(status='BUDGET_INSUFFICIENT',continue_main=False,reason='ORIGINAL_ALLOWANCE_CANNOT_SUPPORT_FORECAST')
    return dict(status='PASS' if all(x=='PASS' for x in states) else 'COMPLETE_WITH_NUMERICAL_INCONCLUSIVES',
        continue_main=True,reason='INDEPENDENT_REMAINING_MODELS_RETAIN_ORIGINAL_TOLERANCES_AND_BUDGET')


def _load_one(row, guard):
    from .models import load_model,PortableICNN
    p=Path(row['model_path']);before=p.stat()
    if before.st_size!=row['recorded_compressed_bytes']:raise ValueError('Selected model size differs')
    if file_sha(p,guard)!=row['model_sha256']:raise ValueError('Selected model bytes differ')
    for path,digest in row['artifact_hashes'].items():
        if Path(path).resolve()!=p.resolve() or digest!=row['model_sha256']:raise ValueError('No companion data files admitted')
    with zipfile.ZipFile(p) as archive:
        if sum(x.file_size for x in archive.infolist())>BUDGET['max_npz_uncompressed_bytes']:
            raise MemoryError('Selected NPZ decompression bound')
    guard();model=load_model(p);guard();after=p.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('Selected model changed during load')
    if not isinstance(model,PortableICNN) or model.width not in (32,64,128):raise ValueError('Unexpected selected architecture')
    return model


def worker(config_path,config_sha256,admission_path,output):
    """Internal owned child only. This function is never called on import."""
    import psutil
    out=Path(output);intent=read_json(out/'INTENT.json')
    parent=psutil.Process(os.getppid())
    if (parent.pid!=intent['supervisor_pid'] or abs(parent.create_time()-intent['supervisor_create_time'])>1e-6
        or intent['config_sha256']!=config_sha256):raise ValueError('Audit child is not owned by admitted supervisor')
    c,admitted=admit(config_path,config_sha256,admission_path)
    if Path(admitted['output'])!=out or intent['admission_sha256']!=admitted['admission_sha256']:
        raise ValueError('Worker output/admission differs')
    if intent['deadline_epoch']!=admitted['audit_deadline_epoch']:raise ValueError('Worker cannot reset shared audit allowance')
    limit=admitted['audit_deadline_epoch']-BUDGET['sealing_reserve_seconds']
    process=psutil.Process();peak=0
    def guard():
        nonlocal peak
        if time.time()>=limit:raise TimeoutError('M02 finite audit deadline')
        obs=resource_observation(process,c['other_owned_processes']);peak=max(peak,obs['audit_tree_rss_bytes'],obs['process_peak_rss_bytes'])
    rows=[];completed=[];begin=time.time()
    source_signatures={b['path']:(Path(b['path']).stat().st_size,Path(b['path']).stat().st_mtime_ns) for b in c['source_bindings'].values()}
    try:
        guard();rows=_inventory(c,guard)
        new_json(out/'AUTHENTICATED_INVENTORY.json',rows)
        import torch
        torch.set_num_threads(1)
        # Thread configuration is an audit runtime bound, not a training change.
        guard();ordered=pilot_order(rows);pilot_begin=time.time()
        for index,row in enumerate(ordered):
            guard();started=time.time();rid=row['region_index']
            for path,signature in source_signatures.items():
                stat=Path(path).stat()
                if (stat.st_size,stat.st_mtime_ns)!=signature:raise ValueError('Admitted source changed during audit')
            entry=dict(region_index=rid,model_sha256=row['model_sha256'],started_epoch=started,
                phase='PILOT' if index<PARAMETERS['pilot_count'] else 'MAIN')
            model=None
            try:
                model=_load_one(row,guard)
                entry.update(audit_model(model,guard=guard));del model;gc.collect();guard()
            except Exception as exc:
                entry.update(status='RESOURCE_LIMIT' if isinstance(exc,MemoryError) else 'TIME_LIMIT' if isinstance(exc,TimeoutError)
                    else 'FAIL',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
            finally:
                model=None;gc.collect()
            entry.update(ended_epoch=time.time(),elapsed_seconds=time.time()-started,peak_process_rss_bytes=peak)
            new_json(out/'model_results'/f'{rid:06d}.json',entry);completed.append(entry)
            if entry['status'] in ('RESOURCE_LIMIT','TIME_LIMIT'):break
            if index==PARAMETERS['pilot_count']-1:
                pilot_seconds=time.time()-pilot_begin
                forecast=pilot_seconds/PARAMETERS['pilot_count']*(len(rows)-PARAMETERS['pilot_count'])
                pilot=dict(**pilot_continuation(completed,forecast,max(0.,limit-time.time())),
                    completed=len(completed),region_indices=[r['region_index'] for r in completed],
                    elapsed_seconds=pilot_seconds,estimated_remaining_seconds=forecast,
                    estimator='Mean of deliberately largest/active/inactive pilot models; runtime forecast only',
                    remaining_budget_seconds=max(0.,limit-time.time()),peak_process_rss_bytes=peak)
                new_json(out/'PILOT_RESULT.json',pilot)
                if not pilot['continue_main']:break
        for b in c['source_bindings'].values():bound(b,guard=guard,maximum=1024**2)
        new_json(out/'WORKER_RESULT.json',dict(status='FINISHED_WITH_RECORDED_OUTCOMES',expected=len(rows),
            completed=len(completed),elapsed_seconds=time.time()-begin,peak_process_rss_bytes=peak))
    except Exception as exc:
        new_json(out/'WORKER_RESULT.json',dict(status='RESOURCE_LIMIT' if isinstance(exc,MemoryError) else 'TIME_LIMIT' if isinstance(exc,TimeoutError) else 'FAIL',
            error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),expected=1499,
            completed=len(completed),elapsed_seconds=time.time()-begin,peak_process_rss_bytes=peak))


def _retire_owned(child):
    if child.poll() is None:
        child.terminate()
        try:child.wait(timeout=3)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=3)


def seal_outputs(out, c, *, supervisor):
    """Preserve every completed result; include explicit unattempted model rows."""
    out=Path(out)
    def seal_guard():
        if time.time()>=supervisor['deadline_epoch']:raise TimeoutError('Sealing exceeded the original shared audit deadline; preserve partial evidence')
    seal_guard()
    inventory=_csv_rows(bound(c['model_manifest'],maximum=4*1024**2))
    rows=[];counts={};failures=[]
    for original in inventory:
        seal_guard()
        rid=int(original['region_index']);p=out/'model_results'/f'{rid:06d}.json'
        try:
            data=read_json(p,maximum=1024**2) if p.exists() else {}
        except (ValueError, UnicodeError) as exc:
            # Retain the corrupt native evidence and count it as an explicit
            # failed metadata disposition, never PASS or a missing observation.
            data=dict(model_sha256=original['model_sha256'],status='INVALID_RESULT_METADATA',
                error_type=type(exc).__name__,error=str(exc))
        if data and data.get('model_sha256')!=original['model_sha256']:raise ValueError('Output model identity differs')
        status=data.get('status','UNATTEMPTED')
        counts[status]=counts.get(status,0)+1
        if status not in ('PASS','UNATTEMPTED'):failures.append(rid)
        rows.append(dict(region_index=rid,model_sha256=original['model_sha256'],status=status,
            dimension=data.get('dimension'),width=data.get('width'),amplitude=data.get('amplitude'),
            checks=len(data.get('checks',[])),elapsed_seconds=data.get('elapsed_seconds'),result_path=str(p) if data else ''))
    if len(rows)!=1499 or len({r['region_index'] for r in rows})!=1499:raise ValueError('Final manifest must contain1499 IDs')
    with (out/'MODEL_PROPERTY_RESULTS.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    worker_result=read_json(out/'WORKER_RESULT.json') if (out/'WORKER_RESULT.json').exists() else {}
    pass_all=(counts=={'PASS':1499} and supervisor['worker_exit_code']==0 and supervisor['status']=='FINISHED'
        and worker_result.get('status')=='FINISHED_WITH_RECORDED_OUTCOMES'
        and worker_result.get('expected')==1499 and worker_result.get('completed')==1499)
    summary=dict(schema='PACNHN_FINAL_ICNN_GUARANTEES_RESULT_V1',status='PASS_WITH_LIMITATIONS' if pass_all else 'PARTIAL_WITH_LIMITATIONS',
        expected=1499,dispositions=counts,failed_or_inconclusive_region_indices=failures,
        all_selected_constraints_and_declared_checks_pass=pass_all,
        analytic_global_claim='Conditional on authenticated nonnegative parameters and intended smooth model formula; see bound proof',
        finite_numerical_scope=PARAMETERS['numerical_scope'],accuracy_evaluated=False,training_target_changed=False,
        model_selection_changed=False,TEST_opened=False,supervisor=supervisor,worker_result=worker_result,source_bindings=c['source_bindings'],
        proof_bindings=c['proof_bindings'],registry_binding=c['registry'],numerics=PARAMETERS)
    new_json(out/'RESULT.json',summary)
    if not (out/'PILOT_RESULT.json').exists():new_json(out/'PILOT_RESULT.json',dict(status='INCOMPLETE',reason='Audit ended before the eight-model pilot finished'))
    text=('# Final ICNN guarantees\n\n'+summary['status']+'\n\n'+
        'Expected: 1,499 selected s03 ICNNs. Dispositions: '+canonical(counts)+'.\n\n'+
        'Analytic convexity, anchoring, cost floor, symmetric PSD Jacobian and nonnegative cross-partials follow from the bound parameter constraints and proof. Numerical results are finite synthetic sampling, not a global numerical guarantee, TRAIN/TEST error measurement or physical accuracy claim. The stable potential uses a documented local sixth-order evaluation. All unavailable, unattempted, failed and inconclusive outcomes remain explicit. No selection or model was changed.\n')
    with (out/'REPORT.md').open('x',encoding='utf-8') as f:f.write(text)
    files=[]
    for p in sorted(out.rglob('*')):
        seal_guard()
        if p.is_file():
            if p.stat().st_size>25_000_000:raise ValueError('Evidence member exceeds25MB; preserve and index explicitly')
            files.append(dict(relative_path=p.relative_to(out).as_posix(),sha256=file_sha(p),bytes=p.stat().st_size))
    new_json(out/'ARTIFACT_MANIFEST.json',dict(members=files,excludes=['ARTIFACT_MANIFEST.json','GUARANTEES_SYNC.zip','SEAL_RECEIPT.json']))
    p=out/'ARTIFACT_MANIFEST.json';files.append(dict(relative_path=p.name,sha256=file_sha(p),bytes=p.stat().st_size))
    archive=out/'GUARANTEES_SYNC.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in files:
            seal_guard();z.write(out/row['relative_path'],row['relative_path'])
    if archive.stat().st_size>200_000_000:raise ValueError('Evidence archive exceeds200MB; preserved, not silently pruned')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or len(z.namelist())!=len(files):raise ValueError('Archive verification failed')
        for row in files:
            seal_guard()
            content=z.read(row['relative_path'])
            if len(content)!=row['bytes'] or hashlib.sha256(content).hexdigest()!=row['sha256']:raise ValueError('Member hash differs')
    archive_sha=file_sha(archive,seal_guard);seal_guard()
    new_json(out/'SEAL_RECEIPT.json',dict(status='VERIFIED',zip_sha256=archive_sha,zip_bytes=archive.stat().st_size,
        members=len(files),result_sha256=file_sha(out/'RESULT.json'),final_epoch=time.time(),
        complete_shared_elapsed_seconds=time.time()-supervisor['shared_start_epoch']))
    return summary


def run(config_path,config_sha256,admission_path):
    """The only production entry point: explicit authority, supervised child."""
    import psutil
    started=time.time();c,admitted=admit(config_path,config_sha256,admission_path)
    out=Path(admitted['output']);out.mkdir(parents=True,exist_ok=False)
    limit=admitted['audit_deadline_epoch']
    owner=psutil.Process()
    new_json(out/'INTENT.json',dict(schema=SCHEMA,config_path=str(Path(config_path).resolve()),config_sha256=config_sha256,
        admission_path=str(Path(admission_path).resolve()),admission_sha256=admitted['admission_sha256'],
        supervisor_pid=owner.pid,supervisor_create_time=owner.create_time(),started_epoch=started,deadline_epoch=limit,
        tasks=TASKS,expected_outputs=OUTPUTS,budget=BUDGET,numerics=PARAMETERS,admitted=admitted))
    for label,path in [('CONFIG',config_path),('ADMISSION',admission_path)]:
        content=Path(path).read_bytes()
        with (out/(label+'.json')).open('xb') as f:f.write(content)
    for rel,b in c['source_bindings'].items():
        p=bound(b,maximum=1024**2);q=out/'code_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True)
        with q.open('xb') as f:f.write(p.read_bytes())
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[sys.executable,'-X','utf8','-B','-m','partC_rev1_src.pacnhn_v1.final_icnn_guarantees',
        '--config',str(Path(config_path).resolve()),'--config-sha256',config_sha256,'--admission',str(Path(admission_path).resolve()),
        '--internal-worker',str(out)]
    status='FINISHED';error=None;peak=0;child=None
    try:
        resource_observation(owner,c['other_owned_processes'])
        with (out/'worker_stdout.txt').open('xb') as stdout,(out/'worker_stderr.txt').open('xb') as stderr:
            child=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[2],env=env,stdout=stdout,stderr=stderr,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            proc=psutil.Process(child.pid)
            while child.poll() is None:
                if time.time()>=limit-BUDGET['sealing_reserve_seconds']:raise TimeoutError('Supervisor audit deadline')
                try:obs=resource_observation(proc,c['other_owned_processes']);peak=max(peak,obs['audit_tree_rss_bytes'],obs['process_peak_rss_bytes'])
                except psutil.NoSuchProcess:break
                time.sleep(.1)
    except Exception as exc:
        status='RESOURCE_LIMIT' if isinstance(exc,MemoryError) else 'TIME_LIMIT' if isinstance(exc,TimeoutError) else 'FAIL'
        error=dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        if child is not None:_retire_owned(child)
    supervisor=dict(status=status,error=error,worker_exit_code=child.returncode if child else None,
        invocation_elapsed_seconds=time.time()-started,shared_start_epoch=admitted['shared_audit_clock']['start_epoch'],
        shared_elapsed_seconds=time.time()-admitted['shared_audit_clock']['start_epoch'],peak_worker_tree_rss_bytes=peak,deadline_epoch=limit,
        scientific_scope='Synthetic model properties only; no data-fit or TEST evaluation')
    new_json(out/'SUPERVISOR_RESULT.json',supervisor)
    return seal_outputs(out,c,supervisor=supervisor)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True);p.add_argument('--admission',required=True)
    p.add_argument('--internal-worker',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.internal_worker:worker(args.config,args.config_sha256,args.admission,args.internal_worker)
    else:run(args.config,args.config_sha256,args.admission)


if __name__=='__main__':main()
