"""Explicit E2 migration, input-only audit freeze, preflight and bounded reentry."""
import argparse
import copy
import io
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
import pandas as pd

from .common import root,rd,read_json,write_json,atomic,sha,digest,input_hash,native_threads,load_config,elapsed
from .source import Sources
from .data import revision
from .e2_policy import seed_for,validate_roots

EPISODE='p67_postaudit_e2_0908'
ORPHAN_REGION='partC_p2_region_0655878f5c4fb39339cb'
ORPHANS={
 '89ac5c740bbecf536192bbed158b1a331acdfcfe946b6c712dc9a37eb044f986':('e6ea23602e735ec86790f81a17249517ceb81949c8de3cedcb9b656e8f32f2aa','85e51171a155637c718e8788016fa9847a9eaa7bbe9740a3f2dc39537852ebbe'),
 'ba629f5162b51f5c775ad3cf345b9e2bd062cb704d1acd57a47a38d2b0960408':('48f59c3c6915183d3a4c32badd5738753d08f694addf56980f9387baf08e6724','7ba71dc8c3e19ae11581aaab5c6d70900c57c85c2bd86ff11c4437028bfddfd0')}


def copy_independent(src,dst,expected):
    src,dst=Path(src),Path(dst)
    if sha(src)!=expected:raise ValueError('Migration source changed: '+str(src))
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():
        with src.open('rb') as inp,dst.open('xb') as out:
            shutil.copyfileobj(inp,out,1024*1024);out.flush();os.fsync(out.fileno())
    a,b=src.stat(),dst.stat()
    if dst.is_symlink() or (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino) or sha(dst)!=expected:
        raise ValueError('Migration must be independent and byte-identical: '+str(dst))
    return dict(source_path=str(src),destination_path=str(dst),sha256=expected,size_bytes=a.st_size,
                source_file_id=str(a.st_ino),destination_file_id=str(b.st_ino),independent=True)


def configure(repo):
    repo=Path(repo);auth=repo/'partC_doc_icnn/review_gates/phase_07'/EPISODE/'authority'
    overlay=read_json(auth/'config/continuation.json');c=read_json(auth/'config/inherited_science.json')
    out=Path(overlay['output_root']);identity=read_json(out/'validation/E0_IDENTITY.json')
    validate_roots(overlay['parent_output_root_readonly'],out)
    parent=Path(overlay['parent_output_root_readonly']);sealed=read_json(parent/'FINAL_HANDOFF.json')
    frozen=read_json(parent/'control/ALL_CHAMPIONS_FROZEN.json');forecast=read_json(parent/'control/finalization_forecast.json')
    measured=sealed['postseal_verification']['verified_unix']-frozen['frozen_unix']
    initial_forecast=max(forecast['remaining_seconds'],measured)+600.
    guards=[parent/'FINAL_HANDOFF.json',parent/'control/ALL_CHAMPIONS_FROZEN.json',parent/'control/STOP_SCIENCE.json',parent/'control/SCIENCE_COMPLETE.json',
            Path(sealed['zip_path']),Path(sealed['zip_path']).with_suffix('.sha256'),Path(sealed['zip_path']).with_suffix('.companion.json')]
    c.update(episode_id=EPISODE,evaluation_episode='E2',step_id=EPISODE,run_id=overlay['run_id'],
        parent_output_root_readonly=overlay['parent_output_root_readonly'],parent_zip_sha256=overlay['parent_zip_sha256'],
        parent_frozen_manifest_sha256=overlay['parent_frozen_manifest_sha256'],parent_added_bytes_at_seal=identity['parent_added_bytes_at_seal'],
        clock_facts=identity,authority_sha256=identity['authority_sha256'],initial_head=identity['initial_head'],
        initial_finalization_forecast_seconds=initial_forecast,
        initial_forecast_basis=dict(parent_persisted_forecast_seconds=forecast['remaining_seconds'],
            parent_freeze_through_verified_seal_seconds=measured,additional_E2_protocol_tail_seconds=600.,
            parent_forecast_sha256=sha(parent/'control/finalization_forecast.json')),
        sealed_parent_guard_bindings=[dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size) for p in guards],
        task_list=['E1_SCOPED_IO_AND_CONTINUATION','E2_MIGRATION_AUDIT_PLAN','E3_REENTRY','E4_RECURRING_J23','E5_FREEZE_AUDIT_REOPEN_SEAL'],
        expected_outputs=['E2_WINDOWS_3PASS','NATIVE_MIGRATION','AUDIT2_INPUTS_FROZEN','E3_REENTRY_PASS','FRESH_CERTIFIED_LABELS','CONTINUED_MODELS','ONE_EXTENSION_ZIP_SHA_COMPANION'])
    c['execution'].update(output_root=str(out),authority_install_relative='partC_doc_icnn/review_gates/phase_07/'+EPISODE,
        run_id=c['run_id'],campaign_id=overlay['overall_campaign_id'],engineering_repairs_max=8,
        new_clock_explicitly_authorized=False,required_ancestor=identity['initial_head'])
    c['acceptance_tests']+=['windows_actual_3_consecutive_spawn_pass','E2_independent_state_copy','fresh_audit2_collision_and_access_barrier','finite_identity_bound_forecast']
    path=repo/'partC_doc_icnn/phase_7_work'/EPISODE/'effective_config.json'
    if path.exists():
        if read_json(path)!=c:raise ValueError('Existing E2 config differs; preserve and version explicitly')
    else:write_json(path,c)
    write_json(out/'run_identity.json',dict(**identity,effective_config_path=str(path),effective_config_sha256=sha(path),
        task_list=c['task_list'],expected_outputs=c['expected_outputs'],acceptance_tests=c['acceptance_tests'],
        resource_budget=dict(original_hard_deadline=c['hard_deadline_unix'],latest_science_cutoff=c['latest_science_cutoff_unix'],
                             rss_bytes=c['execution']['rss_bytes'],cumulative_added_disk_bytes=c['execution']['added_disk_bytes'])))
    return path


def checkpoint_evidence(path):
    from partC_rev1_src.phase7.training import torch_load
    import torch
    state=torch_load(path)
    for key in ['model','optimizer','rng_state','order','cursor','steps','dataset_hash','cycle']:
        if key not in state:raise ValueError('Missing native resume field '+key)
    assert state['optimizer'] is not None and state['steps']>0
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    assert state['rng_state'].dtype==torch.uint8 and 0<=state['cursor']<=len(state['order'])
    assert len(state['unique_seen'])==state['n_train']
    def tensors(value):
        if isinstance(value,torch.Tensor):yield value
        elif isinstance(value,dict):
            for x in value.values():yield from tensors(x)
        elif isinstance(value,(list,tuple)):
            for x in value:yield from tensors(x)
    optimizer=list(tensors(state['optimizer']))
    assert optimizer and all(torch.isfinite(x).all() for x in optimizer)
    return dict(steps=state['steps'],dataset_hash=state['dataset_hash'],cycle=state['cycle'],n_train=state['n_train'],
        cursor=state['cursor'],order_length=len(state['order']),rows_seen=state['rows_seen'],
        optimizer_tensor_count=len(optimizer),rng_sha256=__import__('hashlib').sha256(state['rng_state'].numpy().tobytes()).hexdigest(),
        native_state_sha256=sha(path),optimizer_rng_sampler_preserved=True)


def migrate(c):
    out=root(c);parent=Path(c['parent_output_root_readonly']);done=out/'validation/E2_MIGRATION_PASS.json'
    if done.exists():return read_json(done)
    proof=read_json(out/'validation/WINDOWS_PROTOCOL_3PASS.json')
    assert proof['status']=='PASS' and len(proof['trials'])==3
    from . import shared_io
    assert proof['helper_sha256']==sha(shared_io.__file__)
    auth=Path(c['execution']['repository_root'])/c['execution']['authority_install_relative']/'authority'
    with zipfile.ZipFile(auth/'input/sealed_v1.zip') as z:
        raw=z.read('external_artifact_index/manifest.parquet')
    assert __import__('hashlib').sha256(raw).hexdigest()=='6eadbb9cbd08aaa38f8e4e38ab3426b0059f53cc13d29ca5875681c029514348'
    atomic(out/'control/parent_source_manifest.parquet',raw)
    index=pd.read_parquet(io.BytesIO(raw));assert index.relative_path.is_unique
    table=index.set_index('relative_path');copies=[]
    required_controls=['p7_source_manifest.parquet','source_region_registry.parquet','source_region_registry.csv','old_train_valid_metrics.parquet','audit_plan.json']
    for name in required_controls:
        rel='control/'+name;copies.append(copy_independent(parent/rel,out/rel,table.loc[rel,'sha256']))
    source=Sources(c)
    # Select only pre-final paths. Sealed controls, evaluation and final candidates
    # are provenance, never active E2 state or a model-selection input.
    keep=index.relative_path.str.match(r'regions/[^/]+/(training/|protected/|data_extension/plans/|data_extension/jobs/)')
    keep &= ~index.relative_path.str.contains('/training/cycles/final/',regex=False)
    keep &= ~index.relative_path.str.contains('/protected/audit_solver_progress',regex=False)
    selected=index.loc[keep]
    for i,row in enumerate(selected.itertuples(index=False)):
        copies.append(copy_independent(source.path_v1(row.relative_path),out/row.relative_path,row.sha256))
        if i%1000==0:print(json.dumps(dict(stage='MIGRATION_COPY',files=i,total=len(selected))),flush=True)
    reg=pd.read_parquet(out/'control/source_region_registry.parquet');regions=list(reg.loc[reg.m_active.gt(0),'region_id'])
    assert len(reg)==1847 and len(regions)==1566 and int(reg.m_active.sum())==80316
    batches=[];states=[]
    for i,rid in enumerate(regions):
        rel='regions/'+rid.removeprefix('partC_p2_region_');lp=rel+'/data_extension/LATEST.json'
        rev=dict(version=0,batches=[],TRAIN_AB=0,VALID_AB=0)
        if lp in table.index:
            pointer=read_json(source.path_v1(lp));rp=rel+'/'+pointer['path'];assert table.loc[rp,'sha256']==pointer['sha256']
            rev=read_json(source.path_v1(rp))
            copies.append(copy_independent(parent/rp,out/rp,pointer['sha256']))
            copies.append(copy_independent(parent/lp,out/lp,table.loc[lp,'sha256']))
        write_json(rd(c,rid)/'provenance/INHERITED_BATCHES.json',dict(parent_episode='V1',batches=rev['batches'],
            revision_sha256=digest(rev),read_only_layer=True))
        for binding in rev['batches']:
            p=source.path_v1(rel+'/'+binding['path']);assert sha(p)==binding['sha256'];manifest=read_json(p)
            rows_path=source.path_v1((p.parent/'rows.parquet').relative_to(parent));source.path_v1((p.parent/'native.npz').relative_to(parent))
            assert table.loc[(p.parent/'native.npz').relative_to(parent).as_posix(),'sha256']==manifest['native_sha256']
            rows=pd.read_parquet(rows_path,columns=['input_hash','split','quality'])
            assert len(rows)==binding['rows'] and rows.input_hash.is_unique and rows.split.eq(binding['role']).all()
            batches.append(dict(region_id=rid,**binding,source_manifest_path=str(p),role_preserved=True))
        for p in (rd(c,rid)/'training').glob('*/last.pt'):
            evidence=checkpoint_evidence(p);evidence.update(region_id=rid,kind=p.parent.name,path=str(p));states.append(evidence)
        if i%100==0:print(json.dumps(dict(stage='MIGRATION_VALIDATE',regions=i,total=len(regions))),flush=True)
    # Fork only the unfinished job's catalog and two exact certified receipts.
    orphan_prefix='regions/'+ORPHAN_REGION.removeprefix('partC_p2_region_')+'/data_extension/train_labels/'
    for row in index.loc[index.relative_path.str.startswith(orphan_prefix)].itertuples(index=False):
        copies.append(copy_independent(source.path_v1(row.relative_path),out/row.relative_path,row.sha256))
    from partC_rev1_src.phase6.storage import Store
    d,a,pairs,s=source.local(ORPHAN_REGION);store=Store(rd(c,ORPHAN_REGION)/'data_extension',d,a,namespace='train_labels');recovered=[]
    reserved=pd.read_parquet(rd(c,ORPHAN_REGION)/'data_extension/plans/cycle_0001_pack_000000/plan.parquet')
    for ih,(receipt_hash,payload_hash) in ORPHANS.items():
        rp=store.root/'receipts'/(ih+'.json');assert sha(rp)==receipt_hash
        result,receipt=store.get(ih,recertify=True)
        assert receipt['payload_sha256']==payload_hash and result.quality=='A' and receipt['metadata']['split']=='TRAIN'
        assert len(reserved.loc[reserved.input_hash.eq(ih)])==1
        recovered.append(dict(input_hash=ih,receipt_sha256=receipt_hash,payload_sha256=payload_hash,quality='A',role='TRAIN',
            disposition='RECONCILED_PENDING_ORIGINAL_JOB_PUBLICATION',new_solve=False,credited_as_fresh_E2=False))
    write_json(out/'validation/ORPHAN_RECONCILIATION.json',dict(status='PASS',region_id=ORPHAN_REGION,receipts=recovered,jobid='c0001_TRAIN_p000000_o000'))
    ledger=read_json(source.path_v1('control/ENGINEERING_REPAIRS.json'));assert len(ledger['epochs'])==7
    ledger=copy.deepcopy(ledger);ledger['epochs'].append(dict(epoch=8,scope='Windows shared I/O and E2 continuation protocol',
        authority_sha256=c['authority_sha256'],parent_ledger_sha256=table.loc['control/ENGINEERING_REPAIRS.json','sha256'],
        windows_probe_sha256=sha(out/'validation/WINDOWS_PROTOCOL_3PASS.json'),unix=time.time()))
    write_json(out/'control/ENGINEERING_REPAIRS.json',ledger)
    pd.DataFrame(copies).to_parquet(out/'validation/migration_file_bindings.parquet',index=False)
    pd.DataFrame(states).to_parquet(out/'validation/inherited_checkpoint_states.parquet',index=False)
    pd.DataFrame(batches).to_parquet(out/'validation/inherited_batch_bindings.parquet',index=False)
    assert sum(r['kind']=='POWER4_SKIP_ICNN32' for r in states)==1566
    source.flush('E2_migration');source.close()
    result=dict(status='PASS',copied_files=len(copies),copied_bytes=sum(r['size_bytes'] for r in copies),
        checkpoint_states=len(states),primary_regions=1566,inherited_batches=len(batches),
        inherited_primary_updates=sum(r['steps'] for r in states if r['kind']=='POWER4_SKIP_ICNN32'),
        orphan_receipts_reconciled=2,J1_repeated=False,v1_writes=False,mutable_hardlinks=False,
        evidence_sha256=sha(out/'validation/migration_file_bindings.parquet'),finished_unix=time.time())
    write_json(done,result);return result


def freeze_audit(c):
    from .sampling import bounded,operating_mask
    from partC_rev1_src.phase6.sampling import draw,PACK
    from .common import save_npz
    out=root(c);done=out/'validation/AUDIT2_INPUTS_FROZEN.json'
    if done.exists():return read_json(done)
    assert read_json(out/'validation/E2_MIGRATION_PASS.json')['status']=='PASS'
    source=Sources(c);reg=pd.read_parquet(out/'control/source_region_registry.parquet');summaries=[]
    prior_plans={}
    query="SELECT relative_path FROM files WHERE relative_path GLOB 'regions/*/*plan*.parquet' OR relative_path GLOB 'regions/*/*_plan.json'"
    for (relative,) in source.db.execute(query):
        prior_plans.setdefault(relative.split('/')[1],[]).append(relative)
    for i,rid in enumerate(reg.loc[reg.m_active.gt(0),'region_id']):
        dest=rd(c,rid)/'protected_e2';rp=dest/'FROZEN.json'
        if rp.exists():
            value=read_json(rp);assert sha(dest/'inputs.npz')==value['input_npz_sha256'] and sha(dest/'plan.parquet')==value['plan_sha256']
            summaries.append(dict(region_id=rid,sha256=sha(rp),**{k:value[k] for k in ['rows','counts','no_unique_inputs']}));continue
        rel='regions/'+rid.removeprefix('partC_p2_region_')
        oldrows=pd.read_parquet(source.path7(rel+'/pack/rows.parquet'),columns=['input_hash','split','quality'])
        existing=set(oldrows.input_hash);existing.update(read_json(rd(c,rid)/'protected/FROZEN.json')['input_hashes'])
        old_reserved_paths=prior_plans.get(rid.removeprefix('partC_p2_region_'),[])
        for relative in old_reserved_paths:
            if relative.endswith('.parquet'):
                existing.update(source.frame6(relative).input_hash.dropna())
            else:
                plan=source.json6(relative)
                for key in ['rows','redraw_ledger']:
                    existing.update(row['input_hash'] for row in plan.get(key,[]) if row.get('input_hash'))
        # All reserved parent inputs are excluded, including unselected adaptive proposals.
        for p in (rd(c,rid)/'data_extension/plans').glob('*/FROZEN.json'):
            plan=read_json(p);existing.update(a['input_hash'] for a in plan.get('attempts',[]) if a.get('input_hash'))
        for binding in revision(c,rid)['batches']:
            p=source.path_v1(rel+'/'+binding['path']);existing.update(pd.read_parquet(p.parent/'rows.parquet',columns=['input_hash']).input_hash)
        if rid==ORPHAN_REGION:existing.update(ORPHANS)
        exclusion_count=len(existing);exclusion_hash=digest(sorted(existing))
        d,a,pairs,s=source.local(rid);low=pairs.LOW_CURRENT.to_numpy(float);src=pairs.source_terminal_index.to_numpy();tgt=pairs.target_terminal_index.to_numpy()
        xs=[];rows=[];attempts=[];unresolved=[]
        for role in ['E2_AUDIT_OPERATING','E2_AUDIT_STRESS']:
            for slot in range(8):
                family=PACK[slot*4%32] if role.endswith('STRESS') else ['RADIAL','PAIR','SOURCE','TARGET','SPARSE'][slot%5]
                for redraw in range(9):
                    seed=seed_for(rid,role,slot,redraw);rng=np.random.default_rng(seed)
                    u,grammar=draw(low,src,tgt,family,rng,s['bottleneck_masks']) if role.endswith('STRESS') else (bounded(low,src,tgt,rng,family),'LOW_BOUNDED')
                    ih=input_hash(d['model_hash'],d['support_hash'],u)
                    attempts.append(dict(role=role,slot=slot,redraw=redraw,input_hash=ih,collision=ih in existing))
                    if ih not in existing:
                        existing.add(ih);rows.append(dict(region_id=rid,input_index=len(xs),split=role,family=family,slot=slot,seed=str(seed),redraw=redraw,
                            input_hash=ih,model_hash=d['model_hash'],support_hash=d['support_hash'],grammar=grammar,
                            regime='OPERATING' if operating_mask(u[None,:],low)[0] else 'STRESS'));xs.append(u);break
                else:unresolved.append(dict(region_id=rid,role=role,slot=slot,reason='NO_UNIQUE_INPUT_AFTER_EIGHT_REDRAWS'))
        save_npz(dest/'inputs.npz',u=np.asarray(xs,dtype=np.float64).reshape(-1,len(low)))
        frame=pd.DataFrame(rows);frame.to_parquet(dest/'plan.parquet',index=False)
        value=dict(status='FROZEN_INPUTS_ONLY',episode_id=EPISODE,region_id=rid,rows=len(rows),counts=frame.split.value_counts().to_dict(),
            input_npz_sha256=sha(dest/'inputs.npz'),plan_sha256=sha(dest/'plan.parquet'),input_hashes=[r['input_hash'] for r in rows],
            model_hash=d['model_hash'],support_hash=d['support_hash'],scale_sha256=sha(source.path6(rel+'/scales.npz')),
            generator_sha256=sha(Path(__file__)),seed_namespace=EPISODE+'/AUDIT2_FIXED_20260908',
            excluded_unique_inputs=exclusion_count,excluded_input_hashes_sha256=exclusion_hash,old_reserved_plan_paths=old_reserved_paths,
            collision_attempts=attempts,no_unique_inputs=unresolved,no_labels_or_predictions_read=True,frozen_unix=time.time())
        write_json(rp,value);summaries.append(dict(region_id=rid,sha256=sha(rp),rows=len(rows),counts=value['counts'],no_unique_inputs=unresolved))
        if i%100==0:print(json.dumps(dict(stage='AUDIT2_INPUT_FREEZE',regions=i,labels_generated=0)),flush=True)
    source.flush('E2_audit_inputs');source.close()
    assert len(summaries)==1566
    result=dict(status='PASS',episode_id=EPISODE,regions=1566,planned_rows=sum(r['rows'] for r in summaries),
        target_rows=1566*16,region_manifests=summaries,original_cutoff_unix=c['latest_science_cutoff_unix'],
        no_unique_count=sum(len(r['no_unique_inputs']) for r in summaries),frozen_unix=time.time(),adaptation_started=False)
    write_json(done,result);return result


def verify_dataset_lineage(c):
    out=root(c);source=Sources(c);states=pd.read_parquet(out/'validation/inherited_checkpoint_states.parquet');records=[]
    for rid,group in states.groupby('region_id',sort=False):
        relative='regions/'+rid.removeprefix('partC_p2_region_')
        old_manifest_hash=sha(source.path7(relative+'/pack/manifest.json'))
        old_rows=pd.read_parquet(source.path7(relative+'/pack/rows.parquet'),columns=['split','quality'])
        old_n=int((old_rows.split.eq('TRAIN')&old_rows.quality.isin(['A','B'])).sum())
        versions={digest(dict(old_manifest=old_manifest_hash,batches=[],old_only=True)):('OLD_ONLY',old_n),
                  digest(dict(old_manifest=old_manifest_hash,batches=[],old_only=False)):('EMPTY_EXTENSION',old_n)}
        prefix=relative+'/data_extension/revisions/'
        for path in source.parent_bindings.index[source.parent_bindings.index.str.startswith(prefix)]:
            rev=read_json(source.path_v1(path));batches=[b for b in rev['batches'] if not b['role'].startswith('PATCH_AUDIT')]
            versions[digest(dict(old_manifest=old_manifest_hash,batches=batches,old_only=False))]=(path,old_n+sum(b['AB'] for b in batches if b['role']=='TRAIN'))
        for row in group.itertuples(index=False):
            assert row.dataset_hash in versions,(rid,row.kind,'Unbound saved dataset revision')
            revision_path,count=versions[row.dataset_hash];assert count==row.n_train,(rid,row.kind,count,row.n_train)
            records.append(dict(region_id=rid,kind=row.kind,dataset_hash=row.dataset_hash,source_revision=revision_path,n_train=count,status='PASS'))
    source.flush('E2_dataset_lineage');source.close()
    pd.DataFrame(records).to_parquet(out/'validation/dataset_lineage.parquet',index=False)
    result=dict(status='PASS',states=len(records),manifest_sha256=sha(out/'validation/dataset_lineage.parquet'),
        input_order_and_revision_semantics_preserved=True,source_arrays_rewritten=False,verified_unix=time.time())
    write_json(out/'validation/DATASET_LINEAGE_PASS.json',result);return result


def verify_ready(c,config_path,*,require_reentry=False):
    out=root(c);gate=read_json(out/'validation/PRE_RUN_GATE.json')
    assert gate['status']=='PASS' and gate['config_sha256']==sha(config_path)
    for b in gate['code_bindings']:assert sha(Path(c['execution']['repository_root'])/b['path'])==b['sha256']
    for name in ['WINDOWS_PROTOCOL_3PASS.json','E2_MIGRATION_PASS.json','DATASET_LINEAGE_PASS.json','AUDIT2_INPUTS_FROZEN.json']:
        assert read_json(out/'validation'/name)['status']=='PASS' and sha(out/'validation'/name)==gate['stage_receipts'][name]
    if require_reentry:assert read_json(out/'validation/E3_REENTRY_PASS.json')['status']=='PASS'
    assert not (out/'control/ALL_CHAMPIONS_FROZEN.json').exists()
    from .e2_forecast import read
    read(c)
    c['_e2_admission_verified']=True;c['execution_head']=gate['head'];c['effective_config_sha256']=gate['config_sha256']
    return gate


def preflight(c,config_path):
    import subprocess
    from .review_artifacts import snapshot_paths,snapshot_tests
    from partC_rev1_src.phase7.finalize import git
    from .e2_forecast import publish,read
    repo=Path(c['execution']['repository_root']);out=root(c)
    assert not git(c,'diff','--name-only').strip(),'Commit current E2 source and config before preflight'
    head=git(c,'rev-parse','HEAD').strip()
    git(c,'merge-base','--is-ancestor',c['initial_head'],head)
    receipts={}
    for name in ['WINDOWS_PROTOCOL_3PASS.json','E2_MIGRATION_PASS.json','DATASET_LINEAGE_PASS.json','AUDIT2_INPUTS_FROZEN.json']:
        assert read_json(out/'validation'/name)['status']=='PASS';receipts[name]=sha(out/'validation'/name)
    from . import shared_io
    protocol=read_json(out/'validation/WINDOWS_PROTOCOL_3PASS.json');assert protocol['helper_sha256']==sha(shared_io.__file__)
    for trial in protocol['trials']:
        assert sha(trial['path'])==trial['sha256'] and read_json(trial['path'])['status']=='PASS'
    for b in c['sealed_parent_guard_bindings']:
        assert sha(b['path'])==b['sha256'] and Path(b['path']).stat().st_size==b['size_bytes']
    evidence=out/'validation'/('preflight_'+str(time.time_ns()));evidence.mkdir(parents=True)
    env=os.environ.copy();env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    tests=subprocess.run([c['execution']['python_executable'],'-m','pytest','-q','-p','no:cacheprovider','tests/test_phase67_patch.py','tests/test_phase67_e2.py',
        '--basetemp='+str(out/'handoff/live_preflight_fixtures'), '--junitxml='+str(evidence/'live_tests.xml')],cwd=repo,env=env,capture_output=True,text=True,timeout=180)
    atomic(evidence/'live_tests.log',(tests.stdout+tests.stderr).encode());assert tests.returncode==0,(tests.stdout,tests.stderr)
    snapshot=out/'handoff'/('preflight_snapshot_'+str(time.time_ns()));paths=snapshot_paths(repo);bindings=[]
    for p in paths:
        rel=p.relative_to(repo);copy_independent(p,snapshot/rel,sha(p));bindings.append(dict(path=rel.as_posix(),sha256=sha(p),size_bytes=p.stat().st_size))
    isolated=snapshot_tests(c,snapshot,evidence/'isolated_snapshot')
    if not (out/'control/finalization_forecast.json').exists():
        publish(c,c['initial_finalization_forecast_seconds'],c['initial_forecast_basis'],initialize=True)
        write_json(out/'control/finalization_measurements.json',dict(added_labels=0,reopen_seconds=0.,completed_generation_jobs=0,
            inherited_measurements_are_bound_in_initial_forecast=True))
    else:read(c)
    result=dict(status='PASS',head=head,config_sha256=sha(config_path),code_bindings=bindings,stage_receipts=receipts,
        live_test_log_sha256=sha(evidence/'live_tests.log'),isolated_snapshot=isolated,windows_helper_sha256=sha(shared_io.__file__),
        original_deadline_unix=c['hard_deadline_unix'],finalization_forecast_sha256=sha(out/'control/finalization_forecast.json'),
        verified_unix=time.time(),automatic_E3_E4_authorized=True,new_approval_required=False,total_repair_epochs=8)
    write_json(out/'validation/PRE_RUN_GATE.json',result);return result


def reentry(c,config_path):
    from .campaign import Runner
    from .resources import Monitor,worker_setup
    from .training import PRIMARY
    from .e2_forecast import publish
    out=root(c);done=out/'validation/E3_REENTRY_PASS.json'
    if done.exists():return read_json(done)
    verify_ready(c,config_path);worker_setup(c,'E3_supervisor');started=time.time()
    checkpoint=rd(c,ORPHAN_REGION)/'training'/PRIMARY/'last.pt';before=checkpoint_evidence(checkpoint)
    monitor=Monitor(c,'E3_REENTRY').start();runner=Runner(c,2)
    try:
        job=dict(action='GENERATE',region_id=ORPHAN_REGION,cycle=1,role='TRAIN',pack_index=0,offset=0,jobid='c0001_TRAIN_p000000_o000')
        generation=runner.batch([job]);assert len(generation)==1 and generation[0]['ok'],generation
        generated=generation[0]['result'];assert generated['published']>0 and generated.get('new_certified',0)>0
        assert generated.get('recovered_existing')==2
        training=runner.batch([dict(action='TRAIN',region_id=ORPHAN_REGION,cycle=1,kind=PRIMARY,max_updates=200)])
        assert len(training)==1 and training[0]['ok'],training
        trained=training[0]['result'];after=checkpoint_evidence(checkpoint)
        assert trained['last_slice_updates']>0 and after['steps']>before['steps']
        assert after['n_train']>=before['n_train'] and after['dataset_hash']!=before['dataset_hash']
    finally:runner.close();monitor.close()
    measurements=read_json(out/'control/finalization_measurements.json')
    measurements.update(added_labels=generated['new_certified'],reopen_seconds=generated['fresh_reopen_seconds'],completed_generation_jobs=1)
    write_json(out/'control/finalization_measurements.json',measurements)
    publish(c,max(c['initial_finalization_forecast_seconds'],1800.+measurements['reopen_seconds']),measurements)
    result=dict(status='PASS',region_id=ORPHAN_REGION,started_unix=started,completed_unix=time.time(),
        generation=generated,before_checkpoint=before,after_checkpoint=after,optimizer_step_delta=after['steps']-before['steps'],
        original_checkpoint_state_restored=True,J1_repeated=False,protected_truth_read=False,
        audit2_global_freeze_sha256=sha(out/'validation/AUDIT2_INPUTS_FROZEN.json'),next='AUTOMATIC_FULL_J23_UNTIL_ORIGINAL_CUTOFF')
    write_json(done,result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--repository')
    parser.add_argument('--task',required=True,choices=['E1_CONFIGURE','E2_MIGRATE','E2_DATASET_LINEAGE','E2_FREEZE_AUDIT','E2_PREFLIGHT','E3_REENTRY'])
    args=parser.parse_args();native_threads()
    if args.task=='E1_CONFIGURE':
        if not args.repository:parser.error('Explicit repository required')
        print(configure(args.repository),flush=True);return
    if not args.config:parser.error('Explicit config required')
    c=load_config(args.config);assert elapsed(c)<86400 and time.time()<c['latest_science_cutoff_unix']
    functions={'E2_MIGRATE':lambda:migrate(c),'E2_DATASET_LINEAGE':lambda:verify_dataset_lineage(c),'E2_FREEZE_AUDIT':lambda:freeze_audit(c),'E2_PREFLIGHT':lambda:preflight(c,args.config),'E3_REENTRY':lambda:reentry(c,args.config)}
    result=functions[args.task]()
    print(json.dumps({k:v for k,v in result.items() if k not in ['region_manifests','code_bindings']}),flush=True)


if __name__=='__main__':main()
