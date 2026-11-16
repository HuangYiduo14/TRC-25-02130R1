"""Authenticated reuse of sealed native packs and exact Phase 6 local snapshots."""
from pathlib import Path
import io,json,sqlite3,time,zipfile,hashlib
import numpy as np
import pandas as pd
from partC_rev1_src.phase7.data import Pack,safe_source_path
from .common import root,rd,authority,sha,read_json,write_json,append_json,digest

class Sources:
    def __init__(self,c):
        self.c=c;self.p7=Path(c['source_bindings']['p7_result']['root']);self.p6=Path(c['source_bindings']['p6_result']['root'])
        self.bindings=pd.read_parquet(root(c)/'control/p7_source_manifest.parquet').set_index('relative_path')
        self.cache={};self.records=[]
        self.parent_bindings=None
        if c.get('episode_id')=='p67_postaudit_e2_0908':
            self.pv1=Path(c['parent_output_root_readonly'])
            self.parent_bindings=pd.read_parquet(root(c)/'control/parent_source_manifest.parquet').set_index('relative_path')
        self.path7('source_index.sqlite')
        self.db=sqlite3.connect(f'file:{(self.p7/"source_index.sqlite").as_posix()}?mode=ro',uri=True)
    def _path(self,base,rel,h,size):
        rel=Path(rel).as_posix();p=safe_source_path(base,rel);s=p.stat();sig=(s.st_size,s.st_mtime_ns)
        key=str(p)
        if self.cache.get(key)!=sig:
            if size!=s.st_size or sha(p)!=h:raise ValueError('Source binding changed: '+key)
            after=p.stat()
            if (after.st_size,after.st_mtime_ns)!=sig:raise ValueError('Source changed during read: '+key)
            self.cache[key]=sig
            self.records.append(dict(path=key,sha256=h,size_bytes=size,modified_unix_ns=s.st_mtime_ns))
        return p
    def path7(self,rel):
        r=self.bindings.loc[Path(rel).as_posix()]
        return self._path(self.p7,rel,str(r.sha256),int(r.size_bytes))
    def path6(self,rel):
        row=self.db.execute('SELECT sha256,size_bytes FROM files WHERE relative_path=?',(Path(rel).as_posix(),)).fetchone()
        if row is None:raise ValueError('Unbound source6 file '+str(rel))
        return self._path(self.p6,rel,row[0],row[1])
    def path_v1(self,rel):
        if self.parent_bindings is None:raise ValueError('No E2 parent bindings')
        rel=Path(rel).as_posix()
        if rel not in self.parent_bindings.index:raise ValueError('Unbound source v1 '+rel)
        row=self.parent_bindings.loc[rel]
        return self._path(self.pv1,rel,str(row.sha256),int(row.size_bytes))
    def npz6(self,rel):
        with np.load(self.path6(rel),allow_pickle=False) as z:return {k:z[k] for k in z.files}
    def frame6(self,rel):return pd.read_parquet(self.path6(rel))
    def json6(self,rel):return read_json(self.path6(rel))
    def old_pack(self,rid):
        rel='regions/'+rid.removeprefix('partC_p2_region_')+'/pack'
        meta=read_json(self.path7(rel+'/manifest.json'))
        for n,b in meta['files'].items():
            p=self.path7(rel+'/'+n)
            if self.bindings.loc[rel+'/'+n,'sha256']!=b['sha256']:raise ValueError('Pack manifest chain differs')
        p=Pack(self.p7/rel,verify=False)
        assert p.meta['region_id']==rid and p.meta['m_active']>0
        assert p.rows.input_hash.is_unique and p.rows.native_bytes_hash.is_unique
        pairs=pd.read_parquet(self.path7(rel+'/active_pairs.parquet'))
        assert len(pairs)==p.meta['m_active'] and pairs.LOW_CURRENT.gt(0).all()
        assert digest(list(pairs.pair_id))==p.meta['source_pair_order_hash']
        assert np.array_equal(p.s['low'],pairs.LOW_CURRENT.to_numpy(float))
        return p
    def old_arrays(self,rid):
        p=self.path7('regions/'+rid.removeprefix('partC_p2_region_')+'/evaluation/portable.npz')
        with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    def local(self,rid):
        rel='regions/'+rid.removeprefix('partC_p2_region_')
        d=self.json6(rel+'/descriptor.json');a=self.npz6(rel+'/local_graph.npz')
        pairs=self.frame6(rel+'/active_pairs.parquet');s=self.npz6(rel+'/scales.npz')
        assert d['model_class']==self.c['source_bindings']['model_class'] and d['m_active']==len(pairs)>0
        assert len(a['choice_offsets'])==len(pairs)+1 and np.array_equal(pairs.active_coordinate,np.arange(len(pairs)))
        assert pairs.model_hash.eq(d['model_hash']).all() and pairs.support_hash.eq(d['support_hash']).all()
        return d,a,pairs,s
    def flush(self,tag):
        if self.records:
            import os
            p=root(self.c)/'validation/source_reads'/f'{tag}_{os.getpid()}_{time.time_ns()}.parquet'
            p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(self.records).to_parquet(p,index=False);self.records=[]
    def close(self):self.db.close()

def prepare(c):
    out=root(c);done=out/'validation/source_prepare_PASS.json'
    if done.exists():return read_json(done)
    with zipfile.ZipFile(authority(c)/'input/p7_result.zip') as z:
        raw=z.read('external_artifact_index/phase7_artifact_manifest.parquet')
        index=json.loads(z.read('external_artifact_index/phase7_INDEX.json'))
        assert hashlib.sha256(raw).hexdigest()==index['manifest_sha256']
        from .common import atomic
        atomic(out/'control/p7_source_manifest.parquet',raw)
        reg=pd.read_csv(io.BytesIO(z.read('summary_data/reports/all_region_registry.csv')))
        metrics=pd.read_csv(io.BytesIO(z.read('summary_data/reports/metrics_by_region_split.csv')))
        # Deliberately retain only TRAIN/VALID for discovery and scheduling.
        metrics=metrics.loc[metrics.split.isin(['TRAIN','VALID'])].copy()
        metrics.to_parquet(out/'control/old_train_valid_metrics.parquet',index=False)
        # Byte-check direct current scientific dependencies against reviewed snapshot.
        code=[]
        for n in z.namelist():
            if n.startswith('code_snapshot/partC_rev1_src/phase6/') or n.startswith('code_snapshot/partC_rev1_src/phase7/'):
                rel=n.removeprefix('code_snapshot/');p=Path(c['execution']['repository_root'])/rel
                assert p.exists() and sha(p)==hashlib.sha256(z.read(n)).hexdigest(),rel
                code.append(dict(path=rel,sha256=sha(p),size_bytes=p.stat().st_size))
    assert len(reg)==1847 and reg.region_id.is_unique and reg.m_active.sum()==80316
    assert int(reg.m_active.gt(0).sum())==1566 and int(reg.m_active.eq(0).sum())==281
    reg.to_parquet(out/'control/source_region_registry.parquet',index=False)
    reg.to_csv(out/'control/source_region_registry.csv',index=False)
    source=Sources(c)
    # Source7 registry is independently bound to the exact handoff.
    source.path7('reports/all_region_registry.parquet')
    complete=source.json6('registry/REGISTRY_COMPLETE.json');adj=source.json6('registry/runtime_adjacency.json')
    assert complete['source_registry_hash']==c['source_bindings']['source_registry_hash']
    assert adj['online_nodes']==72724 and adj['static_links']==143603 and adj['regional_logical_links']==80316
    source.flush('J0_prepare');source.close()
    result=dict(status='PASS',source7_files=index['files'],source7_bytes=index['bytes'],
        source7_manifest_sha256=index['manifest_sha256'],code_bindings=code,active_regions=1566,inactive_regions=281,active_pairs=80316,
        historical_tests_reused_diagnostics=True,source_overwrite=False)
    write_json(done,result);return result
