"""Read-only regional sources and deterministic nested TRAIN-only study preparation."""
from pathlib import Path
import hashlib
import time
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,write_json,read_npz,save_npz,sha,digest,array_hash
from partC_rev1_src.phase67.source import Sources
from partC_rev1_src.phase67.data import Union,batch_path,FIELDS
from partC_rev1_src.phase67.model import weighted_power4_fit
from .learning_artifacts import persist_preparation


class RegionalSources:
    def __init__(self,ctx):
        self.ctx=ctx
        path=ctx.repo/'partC_doc_icnn/phase_7_work/p67_postaudit_e2_0908/effective_config.json'
        self.c=read_json(path);ctx.bind(path,'READ_ONLY_E2_SOURCE_CONFIGURATION')
        self.e2=Path(self.c['execution']['output_root']);self.source=Sources(self.c)
        for name in ('p7_source_manifest.parquet','parent_source_manifest.parquet'):
            ctx.bind(self.e2/'control'/name,'SEALED_REGIONAL_SOURCE_INDEX')

    def union(self,rid):
        self.ctx.guard()
        result=Union(self.c,self.source,rid,final=False)
        rp=self.e2/'regions'/rid.removeprefix('partC_p2_region_')/'data_extension/LATEST.json'
        self.ctx.bind(rp,'SEALED_REGION_DATA_REVISION_POINTER')
        pointer=read_json(rp);self.ctx.bind(rp.parent.parent/pointer['path'],'SEALED_REGION_DATA_REVISION',pointer['sha256'])
        return result

    def evaluation(self,union):
        """R4 expressly authorizes now-seen audit reuse; old episode execution gates remain untouched."""
        out={}
        for split in ('LOW_TEST','HIGH_PROJECTED'):
            idx=union.old.indices(split)
            if len(idx):out['REUSED_'+split]=(*union.old.fields(idx),union.old.rows.iloc[idx].copy().reset_index(drop=True))
        for binding in union.rev['batches']:
            role=binding['role']
            if not role.startswith(('E2_AUDIT','PATCH_AUDIT')):continue
            path=batch_path(self.c,self.source,union.rid,binding);self.ctx.bind(path,'NOW_SEEN_CERTIFIED_AUDIT_MANIFEST',binding['sha256'])
            m=read_json(path);d=path.parent
            self.ctx.bind(d/'native.npz','NOW_SEEN_CERTIFIED_AUDIT_ARRAYS',m['native_sha256'])
            self.ctx.bind(d/'rows.parquet','NOW_SEEN_CERTIFIED_AUDIT_ROWS',m['rows_sha256'])
            if m['model_hash']!=union.meta['source_descriptor']['model_hash'] or m['support_hash']!=union.meta['source_descriptor']['support_hash']:
                raise ValueError('Now-seen audit identity mismatch')
            a=read_npz(d/'native.npz');r=pd.read_parquet(d/'rows.parquet')
            take=np.flatnonzero(r.quality.isin(['A','B']).to_numpy())
            if len(take):out.setdefault('REUSED_'+role,[]).append((*[a[k][take] for k in FIELDS],r.iloc[take].reset_index(drop=True)))
        for key,value in list(out.items()):
            if isinstance(value,list):out[key]=(*[np.concatenate([item[k] for item in value]) for k in range(5)],pd.concat([item[5] for item in value],ignore_index=True))
        return out

    def flush_bindings(self):
        for r in self.source.records:self.ctx.bind(r['path'],'AUTHENTICATED_NATIVE_REGIONAL_SOURCE',r['sha256'])
        self.source.records=[]

    def close(self):self.flush_bindings();self.source.close()


def seed(*pieces):
    return int(hashlib.sha256('|'.join(map(str,pieces)).encode()).hexdigest()[:15],16)%(2**31-1)


def select_regions(ctx,source):
    p=source.e2/'reports/all_region_status.csv';ctx.bind(p,'FROZEN_E2_TRAINING_COUNTS')
    status=pd.read_csv(p);status=status.loc[status.m_active>0].copy()
    if len(status)!=1566:raise ValueError('Active region denominator changed')
    p=source.e2/'reports/candidate_metrics.parquet';ctx.bind(p,'FROZEN_TRAIN_VALID_CRITERIA')
    columns=['region_id','split','model_kind','regime','family','MARE_equal_pairs','MARE_LOW_pair_weighted','score']
    metrics=pd.read_parquet(p,columns=columns,filters=[('regime','==','ALL'),('family','==','ALL'),
        ('model_kind','==','POWER4_SKIP_ICNN32'),('split','in',['TRAIN_old','TRAIN_v1','TRAIN_e2','VALID_old','PATCH_VALID'])])
    valid=metrics.loc[metrics.split.isin(['VALID_old','PATCH_VALID'])].groupby('region_id').MARE_equal_pairs.mean()
    difficult=metrics.groupby('region_id').MARE_equal_pairs.max()
    status['frozen_VALID_MARE']=status.region_id.map(valid);status['frozen_TRAIN_VALID_max_MARE']=status.region_id.map(difficult)
    status['selection_tie_sha256']=status.region_id.map(lambda x:digest(['N8R4',20260915,x]))
    selected=[];used=set()
    for stratum,column,ascending in [('large','m_active',False),('small','m_active',True),
                                    ('good_VALID','frozen_VALID_MARE',True),('difficult_TRAIN_VALID','frozen_TRAIN_VALID_max_MARE',False)]:
        ranked=status.loc[~status.region_id.isin(used)&status[column].notna()].sort_values([column,'selection_tie_sha256'],ascending=[ascending,True],kind='stable')
        chosen=ranked.head(4).copy();chosen['stratum']=stratum;chosen['rank_within_stratum']=np.arange(1,len(chosen)+1)
        if len(chosen)!=4:raise ValueError('Insufficient distinct stratum members')
        selected.append(chosen);used.update(chosen.region_id)
    frame=pd.concat(selected,ignore_index=True)
    return frame,status,metrics


def prepare_region(ctx,source,row,destination):
    rid=row['region_id'];union=source.union(rid);u,t,L,U,eps,rows=union.fields('TRAIN')
    if len(u)<20 or not rows.input_hash.is_unique:raise ValueError('Too few independent TRAIN rows or duplicated input identities')
    # Explicitly verify unique native inputs, independently of source IDs.
    native_hashes=[hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in u]
    if len(set(native_hashes))!=len(u):raise ValueError('Duplicate native TRAIN vectors')
    order=np.argsort([digest(['N8R4_NESTED',20260915,rid,h]) for h in rows.input_hash],kind='stable')
    u,t,L,U,eps=[x[order] for x in (u,t,L,U,eps)];rows=rows.iloc[order].reset_index(drop=True)
    cal_n=max(1,int(np.ceil(.1*len(u))));cal=np.arange(cal_n)
    c0=union.s['c0'];low=union.s['low'];scale=np.maximum(np.maximum(np.quantile(u[cal],.95,axis=0),low),1e-12)
    eta=weighted_power4_fit(u[cal],t[cal],scale,c0);eta_unit=np.maximum(np.maximum(eta,.01*c0),1.)
    w=.5/len(c0)+.5*low/low.sum()
    destination.mkdir(parents=True,exist_ok=False)
    save_npz(destination/'train.npz',dict(u=u,t=t,L=L,U=U,eps=eps,c0=c0,low=low,scale=scale,
        eta_calibration=eta,eta_unit=eta_unit,pair_weight=w,calibration_indices=cal))
    rows.to_csv(destination/'train_rows.csv',index=False,float_format='%.17g')
    validation=[]
    for split in ('VALID_old','PATCH_VALID'):
        fields=union.fields(split)
        if not len(fields[0]):continue
        if set(fields[5].input_hash)&set(rows.input_hash):raise ValueError('TRAIN/VALID collision')
        name=split.lower();save_npz(destination/(name+'.npz'),dict(zip(['u','t','L','U','eps'],fields[:5])))
        fields[5].to_csv(destination/(name+'_rows.csv'),index=False,float_format='%.17g');validation.append(name)
    if not validation:raise ValueError('Missing fixed validation')
    subsets=[]
    for fraction in (.1,.25,.5,.7,1.):
        n=max(cal_n,int(np.ceil(fraction*len(u))))
        subsets.append(dict(fraction=fraction,unique_n=n,subset_input_hash=digest(rows.input_hash.iloc[:n].tolist())))
    record=dict(row)
    record.update(status='PREPARED',train_unique_n=len(u),calibration_n=cal_n,calibration_input_hash=digest(rows.input_hash.iloc[:cal_n].tolist()),
        train_file_sha256=sha(destination/'train.npz'),train_rows_sha256=sha(destination/'train_rows.csv'),
        union_version_hash=union.version_hash,validation_splits=validation,subsets=subsets,
        validation_artifacts={split:dict(arrays_sha256=sha(destination/(split+'.npz')),rows_sha256=sha(destination/(split+'_rows.csv'))) for split in validation},
        normalization='q95 of common first10% TRAIN calibration plus frozen physical LOW input; positive floor1e-12; no larger-subset or heldout-label fit',
        initialization='All NN parameters and power4 coefficients use the common calibration subset, included in every fraction; neural seed fixed per region/seed',
        analytic_baseline='Relative NNLS refit on every fraction separately',pair_weights='0.5/m+0.5*LOW/sum(LOW), frozen physical input only',
        protected_tests_opened=False,source_model_unchanged=True)
    write_json(destination/'PREPARED.json',record);source.flush_bindings()
    return record


def prepare(ctx):
    out=ctx.root/'learning'
    if out.exists():raise FileExistsError('Learning preparation already started; preserve its original shared clock and outputs')
    source=RegionalSources(ctx);out.mkdir(exist_ok=False)
    stage_deadline=min(ctx.started+14400,ctx.c['compute_cutoff_epoch'])
    write_json(out/'LEARNING_CLOCK.json',dict(stage_start_epoch=ctx.started,stage_deadline_epoch=stage_deadline,
        cap_seconds=14400,includes='preparation, runtime pilot, training, labels and study evaluation; never reset'))
    work_deadline=min(stage_deadline,ctx.deadline or stage_deadline)-60
    try:
        selected,all_active,criteria=select_regions(ctx,source)
        selected.to_csv(out/'selected_regions_16.csv',index=False,float_format='%.17g')
        all_active.to_csv(out/'active_selection_population.csv',index=False,float_format='%.17g')
        criteria.to_csv(out/'frozen_TRAIN_VALID_selection_criteria.csv',index=False,float_format='%.17g')
        large_zero=all_active.loc[all_active.primary_E2_additional_updates.eq(0)].sort_values('m_active',ascending=False).head(8)
        large_zero.to_csv(out/'large_zero_E2_docket.csv',index=False,float_format='%.17g')
        all_active.loc[all_active.m_active.eq(28)].to_csv(out/'m28_docket.csv',index=False,float_format='%.17g')
        rows=[];failures=[];docket=selected.to_dict('records');exhausted=False
        persist_preparation(out,docket,rows,failures,work_deadline)
        for row in docket:
            try:ctx.guard(deadline=work_deadline)
            except TimeoutError:
                exhausted=True;break
            try:
                rows.append(prepare_region(ctx,source,row,out/'regions'/row['region_id']))
            except Exception as exc:
                failures.append(dict(region_id=row['region_id'],stratum=row['stratum'],status='FAILED_PREPARE',
                    error_type=type(exc).__name__,error=str(exc),substitution=False))
                if isinstance(exc,TimeoutError):exhausted=True
            persist_preparation(out,docket,rows,failures,work_deadline,budget_exhausted=exhausted)
            if exhausted:break
        result=persist_preparation(out,docket,rows,failures,work_deadline,final=True,budget_exhausted=exhausted)
        return dict(status=result['status'],regions=len(rows),failed=len(failures),
            unavailable_regions=len(result['preparation_failures']),next='BOUNDED_RUNTIME_PILOT_THEN_FROZEN_CURVE_PLAN')
    finally:source.close()
