"""One-time hash-verified compact native packs, with frozen canonical ownership.

Navigation reads the already sealed Parquet inventories, not the source directory
tree. Only scientific NPZ fields are decompressed; no path flows or oracle imports.
"""
from __future__ import annotations
import io, json, sqlite3, time, zipfile, hashlib, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .common import sha,read_json,write_json,root,region_out,input_hash,arrays_hash,digest,assert_training_time
from .model import train_scales,separable_power4_fit
from .policy import region_seed,assert_split_ownership

class SourceReader:
    def __init__(self,c):
        self.source=Path(c['execution']['source_root']).resolve()
        self.db=sqlite3.connect(f'file:{(root(c)/"source_index.sqlite").as_posix()}?mode=ro',uri=True)
        self.rows=[]
    def read(self,relative):
        rel=Path(relative).as_posix();p=safe_source_path(self.source,rel)
        row=self.db.execute('SELECT sha256,size_bytes FROM files WHERE relative_path=?',(rel,)).fetchone()
        if row is None: raise ValueError('Source file not bound by reviewed manifest: '+rel)
        before=p.stat();b=p.read_bytes();h=hashlib.sha256(b).hexdigest();after=p.stat()
        if h!=row[0] or len(b)!=row[1] or before.st_mtime_ns!=after.st_mtime_ns: raise ValueError('Source binding changed: '+rel)
        self.rows.append(dict(relative_path=rel,absolute_path=str(p),size_bytes=len(b),sha256=h,modified_unix_ns=after.st_mtime_ns))
        return b
    def json(self,p): return json.loads(self.read(p))
    def frame(self,p): return pd.read_parquet(io.BytesIO(self.read(p)))
    def npz(self,p): return np.load(io.BytesIO(self.read(p)),allow_pickle=False)
    def close(self): self.db.close()

def safe_source_path(source,relative):
    """The root is resolved once; reject traversal before any native path lookup.

    Repeated Windows realpath calls triggered an observed CPython access violation.
    Exact sealed path/hash membership remains mandatory immediately after this check.
    """
    rel=Path(relative)
    if rel.is_absolute() or rel.drive or '..' in rel.parts or ':' in str(relative) or '\x00' in str(relative):
        raise ValueError('Source path escapes root')
    p=source/rel
    if not p.is_relative_to(source):raise ValueError('Source path escapes root')
    return p

def build_index(c):
    out=root(c);dest=out/'source_index.sqlite'
    if (out/'validation/source_index_PASS.json').exists(): return
    if dest.exists(): raise ValueError('Unfinished source index must be preserved and repaired explicitly')
    auth=Path(c['execution']['repository_root'])/c['execution']['authority_install_relative']/'authority'
    zp=auth/'input'/c['source_bindings']['source_zip']
    assert sha(zp)==c['source_handoff_sha256']
    source=Path(c['execution']['source_root'])
    evidence=[]
    with zipfile.ZipFile(zp) as z:
        binding=json.loads(z.read('external_artifact_index/FULL_LOCAL_SHARD_BINDINGS.json'))
        shards=[r for r in binding['shards'] if r['group']=='full_file_manifests' and r['relative_path'].startswith('manifests/resume_output_')]
        con=sqlite3.connect(dest)
        con.execute('CREATE TABLE files(relative_path TEXT PRIMARY KEY,sha256 TEXT NOT NULL,size_bytes INTEGER NOT NULL,modified_unix_ns INTEGER)')
        total=0
        for r in shards:
            assert_training_time(c)
            p=source/r['relative_path'];assert sha(p)==r['sha256'] and p.stat().st_size==r['size_bytes']
            f=pd.read_parquet(p,columns=['relative_path','sha256','size_bytes','modified_unix_ns'])
            total+=len(f)
            # Keep all authoritative metadata; this is a one-time compact index read.
            f.to_sql('files',con,if_exists='append',index=False,chunksize=2000)
            evidence.append(r)
        # Population shards are explicitly bound directly by the reviewed handoff.
        for r in binding['shards']:
            if r['group']=='published_label_population':
                con.execute('INSERT OR IGNORE INTO files VALUES(?,?,?,?)',(r['relative_path'],r['sha256'],r['size_bytes'],r['modified_unix_ns']))
        con.commit();count=con.execute('SELECT COUNT(*) FROM files').fetchone()[0];con.close()
        for name in ['summary_data/validation/region_status_FINAL.csv','summary_data/reports/POPULATION.json','summary_data/registry/REGISTRY_COMPLETE.json']:
            target=source/name.removeprefix('summary_data/')
            if sha(target)!=hashlib.sha256(z.read(name)).hexdigest(): raise ValueError('Live reviewed control changed: '+str(target))
    write_json(out/'validation/source_index_PASS.json',dict(status='PASS',indexed_source_files=total,indexed_paths=count,shards=evidence,
        method='Read pre-existing sealed metadata shards once; no physical source tree inventory or path repricing'))

def initialize_registry(c):
    p=root(c)/'all_region_registry.csv'
    if p.exists():return pd.read_csv(p)
    reader=SourceReader(c)
    reg=reader.frame('registry/regions.parquet')
    final=pd.read_csv(Path(c['execution']['source_root'])/'validation/region_status_FINAL.csv')
    assert len(final)==1847 and final.region_id.is_unique and final.m_active.sum()==80316
    assert final.state.value_counts().to_dict()=={'HEALTHY_BASELINE_READY':1566,'NO_ACTIVE_LOW_PAIRS':281}
    assert np.array_equal(reg.region_id,final.region_id)
    contract=reader.json('registry/runtime_adjacency.json');closure=reader.json('registry/REGISTRY_COMPLETE.json')
    assert closure['source_registry_hash']==c['source_bindings']['source_registry_hash']
    assert contract['online_nodes']==72724 and contract['regional_logical_links']==80316 and contract['static_links']==143603
    edges=reader.frame('registry/'+contract['active_pairs_path'])
    assert len(edges)==80316 and edges.pair_id.is_unique and edges.LOW_CURRENT.gt(0).all() and edges.runtime_connected.all()
    group=edges.groupby('region_id',sort=False)
    pairs={k:digest(list(v.pair_id)) for k,v in group}
    exposure=group.LOW_CURRENT.sum()
    active=final.m_active.gt(0)
    assert np.allclose(final.loc[active,'low_exposure'],final.loc[active,'region_id'].map(exposure),rtol=1e-12,atol=1e-8)
    f=final.rename(columns={'state':'source_final_state'}).copy()
    f['pair_order_hash']=f.region_id.map(pairs).fillna('NO_ACTIVE_PAIRS')
    f['seed']=[region_seed(r) for r in f.region_id]
    f['stratum']=np.select([f.m_active.eq(0),f.m_active.eq(1),f.m_active.le(16),f.m_active.le(64),f.m_active.le(256)],['inactive','1','2_16','17_64','65_256'],default='257plus')
    f['status']=np.where(active,'PENDING_PACK','SKIPPED_NO_ACTIVE_LOW_PAIRS')
    for s in ['TRAIN','VALID','LOW_TEST','HIGH_PROJECTED']: f['n_AB_'+s]=0
    f['counts_verified']=False;f['eligible']=False;f['attempts']=0;f['steps']=0
    f.to_csv(p,index=False);f.to_parquet(p.with_suffix('.parquet'),index=False)
    pd.DataFrame(reader.rows).to_parquet(root(c)/'validation/source_registry_reads.parquet',index=False)
    reader.close()
    return f

def fair_queue(f):
    buckets=[list(g.sort_values(['seed','region_id']).region_id) for s in ['1','2_16','17_64','65_256','257plus'] if len(g:=f.loc[f.stratum.eq(s)])]
    return [b[i] for i in range(max(map(len,buckets))) for b in buckets if i<len(b)]

def pack_region(c,row):
    started=time.perf_counter();rid=row['region_id'];out=region_out(c,rid);pack=out/'pack';meta_path=pack/'manifest.json'
    if meta_path.exists(): return read_json(meta_path)
    assert int(row['m_active'])>0 and row['source_final_state']=='HEALTHY_BASELINE_READY'
    if pack.exists() and any(pack.iterdir()):
        # A dead worker may leave an incomplete NPY pack. Preserve it write-new;
        # it has no PASS manifest and is never an eligible training input.
        archive=root(c)/'repairs/incomplete_packs'/rid/(str(time.time_ns()))
        base=root(c).resolve();resolved_pack=pack.resolve();resolved_archive=archive.resolve()
        assert resolved_pack.is_relative_to(base) and resolved_archive.is_relative_to(base)
        archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(resolved_pack),str(resolved_archive))
        write_json(archive.parent/(archive.name+'_receipt.json'),dict(status='PRESERVED_INCOMPLETE_WORKER_PACK',original_path=str(pack),archive_path=str(archive),no_training_use=True))
    pack.mkdir(parents=True,exist_ok=True);reader=SourceReader(c);rel='regions/'+rid.removeprefix('partC_p2_region_')
    try:
        desc=reader.json(rel+'/descriptor.json')
        assert desc['model_hash']==row['model_hash'] and desc['support_hash']==row['support_hash'] and desc['m_active']==row['m_active']
        pairs=reader.frame(rel+'/active_pairs.parquet');m=len(pairs)
        assert m==row['m_active'] and digest(list(pairs.pair_id))==row['pair_order_hash'] and pairs.LOW_CURRENT.gt(0).all()
        with reader.npz(rel+'/scales.npz') as z: scales={k:z[k] for k in z.files}
        s=scales['input_scale'];c0=scales['c0'];V=float(scales['Vscale']);low=pairs.LOW_CURRENT.to_numpy(float)
        assert s.shape==c0.shape==(m,) and np.all(s>0) and V>0 and np.all(c0>=0)
        pop=reader.frame('reports/label_population/'+rid.removeprefix('partC_p2_region_')+'.parquet').sort_values('input_hash').reset_index(drop=True)
        assert pop.input_hash.is_unique and pop.model_hash.eq(desc['model_hash']).all() and pop.support_hash.eq(desc['support_hash']).all()
        assert_split_ownership(pop.to_dict('records'));n=len(pop)
        arrays={k:np.lib.format.open_memmap(pack/(k+'.npy'),mode='w+',dtype='<f8',shape=(n,m) if k in ['inputs_native','costs_native'] else (n,)) for k in ['inputs_native','costs_native','lower_native','upper_native','allowance_native']}
        raw_ids=set();rowdata=[];norm_max=0.
        for j,r in enumerate(pop.itertuples(index=False)):
            if j%100==0: assert_training_time(c,forecast=1.)
            rc=reader.json(rel+'/labels/receipts/'+r.input_hash+'.json')
            assert rc['input_hash']==r.input_hash and rc['region_id']==rid and rc['model_hash']==desc['model_hash'] and rc['support_hash']==desc['support_hash'] and rc['m_active']==m
            assert rc['scalars']['quality']==r.quality and rc['metadata']['split']==r.split and rc['metadata']['family']==r.family
            assert rc['payload_sha256']==r.payload_sha256 and rc['payload_bytes']==r.payload_bytes and rc['native_float64'] and rc['finalized_full_LMO']
            with reader.npz(rel+'/labels/'+rc['payload_path']) as z:
                u=z['demand'];tau=z['tau'];xn=z['normalized_input'];gn=z['normalized_gradient']
            assert reader.rows[-1]['sha256']==rc['payload_sha256']
            assert u.dtype==tau.dtype==np.dtype('float64') and u.shape==tau.shape==(m,) and np.isfinite(tau).all()
            assert input_hash(desc['model_hash'],desc['support_hash'],u)==r.input_hash
            native=np.asarray(u,dtype='<f8').copy();native[native==0]=0.;nh=hashlib.sha256(native.tobytes()).hexdigest()
            if nh in raw_ids:raise ValueError('Duplicate canonical native demand bytes')
            raw_ids.add(nh)
            sc=rc['scalars'];lo=float(sc['value_lower']);hi=float(sc['value_upper']);eps=float(sc['roundoff_allowance'])
            assert np.isfinite([lo,hi,eps]).all() and eps>=0 and lo<=hi+eps
            expected_g=s*(tau-c0)/V
            if not np.allclose(xn,u/s,rtol=1e-12,atol=1e-12) or not np.allclose(gn,expected_g,rtol=1e-12,atol=1e-12):raise ValueError('Native normalized chain rule mismatch')
            for key,v in [('value_lower',lo),('value_upper',hi)]:
                if not np.isclose(rc['normalized_scalars'][key],(v-float(c0@u))/V,rtol=1e-12,atol=1e-12):raise ValueError('Normalized interval mismatch')
            norm_max=max(norm_max,float(np.max(abs(gn-expected_g))))
            for key,v in [('inputs_native',u),('costs_native',tau),('lower_native',lo),('upper_native',hi),('allowance_native',eps)]:arrays[key][j]=v
            rowdata.append(dict(input_hash=r.input_hash,native_bytes_hash=nh,split=r.split,family=r.family,quality=r.quality,relative_gap=float(sc['relative_gap']),baseline=bool(r.baseline)))
        for a in arrays.values():a.flush()
        rows=pd.DataFrame(rowdata);rows.to_parquet(pack/'rows.parquet',index=False)
        plan=reader.frame(rel+'/baseline_plan.parquet');owners=rows.set_index('input_hash').split.to_dict()
        if not all(owners.get(r.input_hash)==r.owner_split for r in plan.itertuples()):raise ValueError('Alias/canonical owner mismatch')
        aliases=plan.loc[plan.alias].copy();aliases.to_parquet(pack/'aliases.parquet',index=False)
        counts=rows.groupby(['split','quality']).size().unstack(fill_value=0)
        ab=rows.quality.isin(['A','B']);idx=np.flatnonzero(ab & rows.split.eq('TRAIN'));valid=np.flatnonzero(ab & rows.split.eq('VALID'))
        eligible=len(idx)>=16 and len(valid)>=2
        scale_data={}
        if eligible:
            ts=train_scales(arrays['inputs_native'][idx],arrays['costs_native'][idx],arrays['upper_native'][idx],s,c0,V,low)
            scale_data={**ts,'baseline_eta':separable_power4_fit(arrays['inputs_native'][idx],arrays['costs_native'][idx],s,c0),
                'train_min':arrays['inputs_native'][idx].min(axis=0),'train_max':arrays['inputs_native'][idx].max(axis=0)}
        np.savez(pack/'scales.npz',scale=s,c0=c0,Vscale=np.array(V),low=low,**scale_data)
        pairs.to_parquet(pack/'active_pairs.parquet',index=False)
        pd.DataFrame(reader.rows).to_parquet(pack/'source_read_manifest.parquet',index=False)
        files={p.name:dict(sha256=sha(p),size_bytes=p.stat().st_size) for p in pack.iterdir() if p.is_file()}
        result=dict(status='PASS',region_id=rid,m_active=m,parameter_count=64*m+1120,eligible=eligible,
            training_status='PENDING_TRAIN' if eligible else 'INSUFFICIENT_INDEPENDENT_AB_DATA',
            seed=int(row['seed']),source_descriptor=desc,source_pair_order_hash=row['pair_order_hash'],
            n_labels=n,counts_AB={k:int((ab & rows.split.eq(k)).sum()) for k in ['TRAIN','VALID','LOW_TEST','HIGH_PROJECTED']},
            split_quality_counts={k:{str(g):int(v) for g,v in vals.items()} for k,vals in counts.to_dict('index').items()},
            alias_count=len(aliases),train_input_hash_binding=digest(rows.iloc[idx].input_hash.tolist()),train_only_scaling=True,
            normalization_max_gradient_error=norm_max,files=files,pack_seconds=time.perf_counter()-started)
        write_json(meta_path,result);return result
    finally:reader.close()

class Pack:
    def __init__(self,path,verify=True):
        self.path=Path(path);self.meta=read_json(self.path/'manifest.json')
        if verify:
            for n,r in self.meta['files'].items():
                if sha(self.path/n)!=r['sha256']:raise ValueError('Derived pack hash mismatch: '+n)
        self.rows=pd.read_parquet(self.path/'rows.parquet')
        self.a={n:np.load(self.path/(n+'.npy'),mmap_mode='r',allow_pickle=False) for n in ['inputs_native','costs_native','lower_native','upper_native','allowance_native']}
        with np.load(self.path/'scales.npz',allow_pickle=False) as z:self.s={k:z[k] for k in z.files}
    def indices(self,split):return np.flatnonzero(self.rows.split.eq(split)&self.rows.quality.isin(['A','B']))
    def fields(self,idx):return tuple(np.asarray(self.a[n][idx]) for n in ['inputs_native','costs_native','lower_native','upper_native','allowance_native'])
