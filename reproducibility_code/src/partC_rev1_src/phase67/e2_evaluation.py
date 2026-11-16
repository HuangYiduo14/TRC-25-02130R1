"""Immutable BEFORE-access receipts and candidate-isolated E2 final evaluation."""
import os
import time
import traceback
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from .common import root,rd,read_json,write_json,sha,digest
from .data import Union,batch_path
from .metrics import predict_arrays,score_metric
from .sampling import operating_mask
from .final_evaluation import portable_math


def issue_access(c,source,rid):
    global_path=root(c)/'control/ALL_CHAMPIONS_FROZEN.json';global_freeze=read_json(global_path)
    assert global_freeze['episode_id']==c['episode_id'] and global_freeze['training_and_generation_now_prohibited']
    path=rd(c,rid)/'training/cycles/final/manifest.json';frozen=read_json(path)
    match=[b for b in global_freeze['manifests'] if b['region_id']==rid]
    assert len(match)==1 and match[0]['sha256']==sha(path)
    roles={};rel='regions/'+rid.removeprefix('partC_p2_region_')
    rows=pd.read_parquet(source.path7(rel+'/pack/rows.parquet'),columns=['input_hash','split'])
    for role,group in rows.groupby('split'):roles[role]=list(group.input_hash)
    for binding in frozen['revision']['batches']:
        p=batch_path(c,source,rid,binding)
        frame=pd.read_parquet(p.parent/'rows.parquet',columns=['input_hash'])
        roles.setdefault(binding['role'],[]).extend(frame.input_hash)
    body=dict(episode_id=c['episode_id'],run_id=c['run_id'],region_id=rid,attempt_id=uuid.uuid4().hex,
        pid=os.getpid(),unix=time.time(),action='BEFORE_PROTECTED_TRUTH_OPEN',
        global_freeze_sha256=sha(global_path),regional_freeze_sha256=sha(path),
        model_bindings=frozen['files'],role_input_hashes={k:sorted(v) for k,v in roles.items()},
        fresh_roles=['E2_AUDIT_OPERATING','E2_AUDIT_STRESS'],reused_roles=['LOW_TEST','HIGH_PROJECTED','PATCH_AUDIT_OPERATING','PATCH_AUDIT_STRESS'])
    receipt=dict(**body,payload_sha256=digest(body));p=rd(c,rid)/'final_evaluation/access_attempts'/(body['attempt_id']+'.json')
    if p.exists():raise FileExistsError('Audit attempt ID already exists')
    write_json(p,receipt)
    if read_json(p)!=receipt:raise ValueError('BEFORE-access receipt readback failed')
    verify_access(c,rid,p)
    return p


def verify_access(c,rid,path):
    if path is None:raise PermissionError('E2 protected truth requires BEFORE-access receipt')
    path=Path(path);expected=rd(c,rid)/'final_evaluation/access_attempts'
    if path.parent!=expected:raise PermissionError('Audit receipt is outside the region attempt namespace')
    value=read_json(path);body={k:v for k,v in value.items() if k!='payload_sha256'}
    assert value['payload_sha256']==digest(body)
    assert value['region_id']==rid and value['run_id']==c['run_id'] and value['episode_id']==c['episode_id']
    assert value['pid']==os.getpid() and value['action']=='BEFORE_PROTECTED_TRUTH_OPEN'
    global_path=root(c)/'control/ALL_CHAMPIONS_FROZEN.json'
    assert value['global_freeze_sha256']==sha(global_path)
    assert read_json(global_path)['frozen_unix']<=value['unix']
    assert value['regional_freeze_sha256']==sha(rd(c,rid)/'training/cycles/final/manifest.json')
    assert value['unix']<=time.time()
    return value


def evaluate_region(c,source,rid):
    dest=rd(c,rid)/'final_evaluation';done=dest/'RESULT.json'
    if done.exists():return read_json(done)
    started=time.perf_counter();receipt_path=issue_access(c,source,rid)
    # This constructor is the first protected-truth opening in this attempt.
    union=Union(c,source,rid,final=True,access_receipt=receipt_path)
    frozen=read_json(rd(c,rid)/'training/cycles/final/manifest.json');roster={};mathrows=[];failures=dict(frozen['candidate_failures'])
    for kind,b in frozen['files'].items():
        if kind in failures:continue
        p=rd(c,rid)/b['path'];assert sha(p)==b['sha256']
        with np.load(p,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        try:
            mathrows.append(portable_math(kind,arrays,union.s['low']));roster[kind]=arrays
        except (AssertionError,ValueError,FloatingPointError,RuntimeError) as error:
            failures[kind]=dict(error_type=type(error).__name__,error=repr(error),traceback=traceback.format_exc(),
                disposition='FROZEN_CANDIDATE_REOPEN_FAILURE_NO_RESELECTION')
    rows=[];metric_failures=[]
    splits=['TRAIN_old','TRAIN_new','TRAIN_v1','TRAIN_e2','VALID_old','PATCH_VALID','LOW_TEST','HIGH_PROJECTED',
            'PATCH_AUDIT_OPERATING','PATCH_AUDIT_STRESS','E2_AUDIT_OPERATING','E2_AUDIT_STRESS']
    for split in splits:
        fields=union.fields(split);u,t,L,U,eps,rr=fields
        if not len(u):continue
        operating=operating_mask(u,union.s['low'])
        masks=[('ALL','ALL',np.arange(len(u))),('OPERATING','ALL',np.flatnonzero(operating)),('STRESS','ALL',np.flatnonzero(~operating))]
        for regime,mask in [('OPERATING',operating),('STRESS',~operating)]:
            for family in rr.family.unique():masks.append((regime,str(family),np.flatnonzero(mask&rr.family.eq(family))))
        for kind,a in roster.items():
            fn=lambda q,k=kind,aa=a:predict_arrays(k,aa,q)
            try:
                for regime,family,ix in masks:
                    if not len(ix):continue
                    subset=tuple(x[ix] for x in fields[:5])+(rr.iloc[ix].reset_index(drop=True),)
                    metric=score_metric(fn,subset,union.s,full=True,batch=64 if union.meta['m_active']>=1024 else 128)
                    rows.append(dict(region_id=rid,m_active=union.meta['m_active'],LOW_exposure=float(union.s['low'].sum()),split=split,regime=regime,family=family,
                        model_kind=kind,selected_champion=kind==frozen['champion_kind'],fresh_E2_audit=split.startswith('E2_AUDIT'),
                        reused_historical_evaluation=split in ['LOW_TEST','HIGH_PROJECTED','PATCH_AUDIT_OPERATING','PATCH_AUDIT_STRESS'],national_readiness=False,**metric))
            except (AssertionError,ValueError,FloatingPointError,RuntimeError) as error:
                metric_failures.append(dict(kind=kind,split=split,error=repr(error),traceback=traceback.format_exc(),reselection=False))
    from partC_rev1_src.phase6.storage import Store
    d,a,pairs,s=source.local(rid);reopened=[];physical_failures=[];reopen_start=time.perf_counter()
    for namespace in ['train_labels','valid_labels','audit2_labels']:
        path=rd(c,rid)/'data_extension'/namespace
        if not path.exists():continue
        store=Store(rd(c,rid)/'data_extension',d,a,namespace=namespace)
        for receipt in (path/'receipts').glob('*.json'):
            try:
                result,rc=store.get(receipt.stem,recertify=True)
                assert np.allclose(np.bincount(store.oracle.pool_arrays()[0][result.path_ids],weights=result.weights,minlength=d['m_active']),result.demand,rtol=2e-11,atol=1e-12)
                reopened.append(dict(input_hash=receipt.stem,quality=result.quality,role=rc['metadata']['split'],
                    source_run_id=rc['metadata']['run_id'],receipt_sha256=sha(receipt),payload_sha256=rc['payload_sha256']))
            except Exception as error:
                physical_failures.append(dict(input_hash=receipt.stem,namespace=namespace,error=repr(error),traceback=traceback.format_exc()))
    frame=pd.DataFrame(rows);frame.to_parquet(dest/'metrics.parquet',index=False)
    write_json(dest/'math.json',dict(passed=mathrows,failed=failures,thresholds_unchanged=True))
    write_json(dest/'physical_reopen.json',dict(reopened=reopened,failed=physical_failures))
    from .e2_setup import checkpoint_evidence
    checkpoint_rows=[];checkpoint_failures=[]
    for path in (rd(c,rid)/'training').glob('*/last.pt'):
        try:checkpoint_rows.append(dict(kind=path.parent.name,path=str(path),**checkpoint_evidence(path)))
        except Exception as error:checkpoint_failures.append(dict(kind=path.parent.name,path=str(path),sha256=sha(path),error=repr(error),traceback=traceback.format_exc()))
    write_json(dest/'native_checkpoint_reopen.json',dict(reopened=checkpoint_rows,failed=checkpoint_failures))
    result=dict(status='PASS' if not metric_failures and not physical_failures and not checkpoint_failures else 'WITH_REPORTED_EXCEPTIONS',region_id=rid,m_active=d['m_active'],
        candidate_count=len(roster),invalid_candidates=len(failures),metric_failures=metric_failures,metric_rows=len(frame),
        champion_kind=frozen['champion_kind'],champion_is_analytic=frozen['champion_is_analytic'],hidden_audit_causes_reselection=False,
        physical_labels_freshly_reopened=len(reopened),physical_reopen_failures=physical_failures,
        native_checkpoints_reopened=len(checkpoint_rows),checkpoint_reopen_failures=checkpoint_failures,
        reopen_seconds=time.perf_counter()-reopen_start,seconds=time.perf_counter()-started,
        access_receipt_path=str(receipt_path),access_receipt_sha256=sha(receipt_path),
        metrics_sha256=sha(dest/'metrics.parquet'),math_sha256=sha(dest/'math.json'),completed_unix=time.time())
    write_json(done,result);return result
