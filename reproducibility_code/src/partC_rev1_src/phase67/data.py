"""Append-only certified batch unions; hidden audit is inaccessible before freeze."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .common import root,rd,read_json,write_json,save_npz,sha,digest,input_hash

FIELDS=['inputs_native','costs_native','lower_native','upper_native','allowance_native']

def audit_role(role):
    return role.startswith(('PATCH_AUDIT','E2_AUDIT'))

def batch_path(c,source,rid,binding):
    if c.get('episode_id')=='p67_postaudit_e2_0908':
        inherited=read_json(rd(c,rid)/'provenance/INHERITED_BATCHES.json')['batches']
        matches=[b for b in inherited if b['path']==binding['path']]
        if matches:
            assert matches==[binding],'Inherited batch identity changed'
            return source.path_v1('regions/'+rid.removeprefix('partC_p2_region_')+'/'+binding['path'])
    return rd(c,rid)/binding['path']

def revision(c,rid):
    p=rd(c,rid)/'data_extension/LATEST.json'
    if not p.exists():return dict(version=0,batches=[],TRAIN_AB=0,VALID_AB=0)
    pointer=read_json(p);path=rd(c,rid)/pointer['path']
    assert sha(path)==pointer['sha256'];return read_json(path)

def attach_batch(c,rid,dest,manifest):
    """Finish either side of the immutable-revision/atomic-pointer crash boundary."""
    assert sha(dest/'native.npz')==manifest['native_sha256']
    assert sha(dest/'rows.parquet')==manifest['rows_sha256']
    prev=revision(c,rid);rel=(dest/'manifest.json').relative_to(rd(c,rid)).as_posix()
    binding=dict(path=rel,sha256=sha(dest/'manifest.json'),role=manifest['role'],rows=manifest['rows'],AB=manifest['AB'])
    existing=[b for b in prev['batches'] if b['path']==rel]
    if existing:
        assert existing==[binding],'Published batch binding changed'
        return manifest
    new={**prev,'version':prev['version']+1,'batches':prev['batches']+[binding],
        'previous_revision_hash':digest(prev),'published_unix':time.time()}
    new['TRAIN_AB']=sum(b['AB'] for b in new['batches'] if b['role']=='TRAIN')
    new['VALID_AB']=sum(b['AB'] for b in new['batches'] if b['role']=='PATCH_VALID')
    rp=rd(c,rid)/'data_extension/revisions'/f'v{new["version"]:06d}.json'
    if rp.exists():
        saved=read_json(rp)
        assert {k:v for k,v in saved.items() if k!='published_unix'}=={k:v for k,v in new.items() if k!='published_unix'},'Unlinked immutable revision differs'
    else:write_json(rp,new)
    write_json(rd(c,rid)/'data_extension/LATEST.json',dict(path=rp.relative_to(rd(c,rid)).as_posix(),sha256=sha(rp),version=new['version']))
    return manifest

def publish_batch(c,rid,store,results,role,jobid):
    """The solver's full physical receipts remain the authority for these compact arrays."""
    dest=rd(c,rid)/'data_extension/batches'/jobid
    if (dest/'manifest.json').exists():
        manifest=read_json(dest/'manifest.json')
        assert manifest['model_hash']==store.desc['model_hash'] and manifest['support_hash']==store.desc['support_hash']
        return attach_batch(c,rid,dest,manifest)
    if not results:return dict(rows=0,AB=0,C=0,role=role,empty=True)
    rows=[];arrays={k:[] for k in FIELDS};receipts=[]
    for result,receipt in results:
        meta=receipt['metadata'];ih=receipt['input_hash']
        assert receipt['model_hash']==store.desc['model_hash'] and receipt['support_hash']==store.desc['support_hash']
        assert receipt['finalized_full_LMO'] and result.quality in {'A','B','C'}
        for k,v in zip(FIELDS,[result.demand,result.tau,result.value_lower,result.value_upper,result.roundoff_allowance]):arrays[k].append(v)
        rows.append(dict(input_hash=ih,split=meta['split'],family=meta['family'],quality=result.quality,cycle=meta.get('cycle',0),regime=meta['regime'],phase_id='06_incremental_patch',
            source_run_id=meta.get('run_id'),
            relative_gap=result.relative_gap,source_payload_sha256=receipt['payload_sha256']))
        receipts.append(dict(path=str((store.root/'receipts'/(ih+'.json')).relative_to(rd(c,rid))).replace('\\','/'),sha256=sha(store.root/'receipts'/(ih+'.json')),
            payload_path=str((store.root/receipt['payload_path']).relative_to(rd(c,rid))).replace('\\','/'),payload_sha256=receipt['payload_sha256']))
    save_npz(dest/'native.npz',**{k:np.asarray(v,dtype=np.float64) for k,v in arrays.items()})
    pd.DataFrame(rows).to_parquet(dest/'rows.parquet',index=False)
    manifest=dict(status='CERTIFIED_IMMUTABLE_BATCH',jobid=jobid,region_id=rid,role=role,rows=len(rows),AB=sum(r['quality'] in {'A','B'} for r in rows),C=sum(r['quality']=='C' for r in rows),
        native_sha256=sha(dest/'native.npz'),rows_sha256=sha(dest/'rows.parquet'),receipts=receipts,model_hash=store.desc['model_hash'],support_hash=store.desc['support_hash'],
        namespace=str(store.root.relative_to(rd(c,rid))).replace('\\','/'),published_unix=time.time())
    write_json(dest/'manifest.json',manifest)
    return attach_batch(c,rid,dest,manifest)

class Union:
    def __init__(self,c,source,rid,*,old_only=False,final=False,revision_override=None,access_receipt=None):
        if final and c.get('episode_id')=='p67_postaudit_e2_0908':
            from .e2_evaluation import verify_access
            verify_access(c,rid,access_receipt)
        self.c=c;self.source=source;self.rid=rid;self.old=source.old_pack(rid);self.s=self.old.s;self.meta=self.old.meta
        self.rev=revision_override or revision(c,rid);self.old_only=old_only;self.final=final
        self.extensions={};self.loaded=[]
        if final:
            frozen=root(c)/'control/ALL_CHAMPIONS_FROZEN.json'
            if not frozen.exists():raise PermissionError('Hidden audit forbidden before ALL champions freeze')
        if not old_only:
            for binding in self.rev['batches']:
                role=binding['role']
                if audit_role(role) and not final:continue
                p=batch_path(c,source,rid,binding);assert sha(p)==binding['sha256'];b=read_json(p);directory=p.parent
                assert sha(directory/'native.npz')==b['native_sha256'] and sha(directory/'rows.parquet')==b['rows_sha256']
                assert b['model_hash']==self.meta['source_descriptor']['model_hash'] and b['support_hash']==self.meta['source_descriptor']['support_hash']
                rr=pd.read_parquet(directory/'rows.parquet')
                if c.get('episode_id')=='p67_postaudit_e2_0908':
                    rr['origin_episode']=np.where(rr.source_run_id.eq(c['run_id']),'E2','V1') if 'source_run_id' in rr else 'V1'
                with np.load(directory/'native.npz',allow_pickle=False) as z:aa={k:z[k] for k in FIELDS}
                for i,row in enumerate(rr.itertuples()):
                    assert input_hash(b['model_hash'],b['support_hash'],aa['inputs_native'][i])==row.input_hash
                self.extensions.setdefault(role,[]).append((rr,aa));self.loaded.append(binding)
        ids=list(self.old.rows.input_hash)
        for batches in self.extensions.values():
            for rr,aa in batches:ids.extend(rr.input_hash)
        assert len(ids)==len(set(ids)),'Cross-source/extension input collision'
        self.version_hash=digest(dict(old_manifest=sha(self.old.path/'manifest.json'),batches=self.loaded,old_only=old_only))
    def fields(self,split):
        if split in {'LOW_TEST','HIGH_PROJECTED'} and not self.final:raise PermissionError('Reused TEST/HIGH access only at final freeze')
        if audit_role(split) and not self.final:raise PermissionError('Hidden audit access only at final freeze')
        rows=[];arrs=[[] for _ in FIELDS]
        oldsplit='TRAIN' if split in {'TRAIN','TRAIN_old'} else 'VALID' if split in {'VALID','VALID_old'} else split
        if split!='TRAIN_new' and oldsplit in {'TRAIN','VALID','LOW_TEST','HIGH_PROJECTED'}:
            idx=self.old.indices(oldsplit);rows.append(self.old.rows.iloc[idx].copy())
            for vv,x in zip(arrs,self.old.fields(idx)):vv.append(x)
        roles=['TRAIN'] if split in {'TRAIN','TRAIN_new','TRAIN_v1','TRAIN_e2'} else ['PATCH_VALID'] if split=='PATCH_VALID' else [split] if audit_role(split) else []
        for role in roles:
            for rr,aa in self.extensions.get(role,[]):
                mask=rr.quality.isin(['A','B'])
                if split in ['TRAIN_v1','TRAIN_e2']:mask &= rr.origin_episode.eq('V1' if split=='TRAIN_v1' else 'E2')
                take=np.flatnonzero(mask);rows.append(rr.iloc[take].copy())
                for key,vv in zip(FIELDS,arrs):vv.append(aa[key][take])
        m=self.meta['m_active']
        merged=tuple(np.concatenate(v,axis=0) if v else np.empty((0,m) if i<2 else (0,),dtype=np.float64) for i,v in enumerate(arrs))
        frame=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=['input_hash','split','family','quality'])
        return (*merged,frame)
    def ownership_hashes(self):
        ids=set(self.old.rows.input_hash)
        for b in self.rev['batches']:
            # Input identities are public; hidden audit truths are never opened here.
            p=batch_path(self.c,self.source,self.rid,b);meta=read_json(p);rr=pd.read_parquet(p.parent/'rows.parquet',columns=['input_hash']);ids.update(rr.input_hash)
        ids.update(read_json(rd(self.c,self.rid)/'protected/FROZEN.json')['input_hashes'])
        if self.c.get('episode_id')=='p67_postaudit_e2_0908':
            ids.update(read_json(rd(self.c,self.rid)/'protected_e2/FROZEN.json')['input_hashes'])
        for p in (rd(self.c,self.rid)/'data_extension/plans').glob('*/plan.parquet'):
            planned=pd.read_parquet(p,columns=['input_hash']);ids.update(planned.input_hash.dropna())
        return ids
