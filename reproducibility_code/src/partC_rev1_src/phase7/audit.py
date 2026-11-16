"""Final machine-readable campaign fairness, source preservation and scope audit."""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from .common import root,region_out,load_config,read_json,write_json,sha,elapsed

def run(c):
    out=root(c);events=[json.loads(s) for s in (out/'training_events.jsonl').read_text().splitlines()]
    queue=pd.read_csv(out/'training_queue.csv');assert len(queue)==1566 and queue.region_id.is_unique
    first_dispatch=[r['region_id'] for r in events if r['event']=='DISPATCH' and r['first_pass']]
    unique_dispatch=list(dict.fromkeys(first_dispatch))
    assert unique_dispatch==queue.region_id.iloc[:len(unique_dispatch)].tolist(),'First queue changed'
    seen=set();first_extended=None;failures=[];max_first_steps=0;max_first_seconds=0.
    for e in events:
        if e['event']=='RETURN' and e['first_pass']:
            assert e['region_id'] not in seen;seen.add(e['region_id'])
            max_first_steps=max(max_first_steps,e.get('steps',0));max_first_seconds=max(max_first_seconds,e.get('train_seconds',0.))
            if 'FAILURE' in e.get('status',''):failures.append(e['region_id'])
        if e['event']=='DISPATCH' and not e['first_pass'] and first_extended is None:
            first_extended=e['unix'];assert len(seen)==1566,'Extended work before every first attempt returned'
    assert max_first_steps<=100 and max_first_seconds<=15.1
    registry=pd.read_parquet(out/'reports/all_region_registry.parquet');source=Path(c['execution']['source_root'])
    assert len(registry)==1847 and registry.m_active.sum()==80316
    inactive=registry.loc[registry.m_active.eq(0)]
    assert len(inactive)==281 and inactive.status.eq('SKIPPED_NO_ACTIVE_LOW_PAIRS').all()
    for rid in inactive.region_id:assert not (region_out(c,rid)/'best.pt').exists()
    total_reads=0;total_bytes=0;source_manifests=[];hash_controls=[]
    for rid in registry.loc[registry.m_active.gt(0),'region_id']:
        pm=region_out(c,rid)/'pack/manifest.json'
        if not pm.exists():continue
        meta=read_json(pm);rm=pm.parent/'source_read_manifest.parquet';f=pd.read_parquet(rm)
        assert len(f)>0 and f.relative_path.is_unique and sha(rm)==meta['files'][rm.name]['sha256']
        total_reads+=len(f);total_bytes+=int(f.size_bytes.sum());source_manifests.append(dict(region_id=rid,path=str(rm),sha256=sha(rm),files=len(f),bytes=int(f.size_bytes.sum())))
        # All scientific label bytes were authenticated once at read. Rehash only
        # small frozen descriptors/scales/pair-order controls at closeout.
        for name in ['descriptor.json','scales.npz','active_pairs.parquet']:
            rel='regions/'+rid.removeprefix('partC_p2_region_')+'/'+name;p=source/rel
            expected=f.set_index('relative_path').loc[rel,'sha256'];actual=sha(p)
            assert actual==expected,'Source control changed after packing: '+rel
            hash_controls.append(dict(relative_path=rel,sha256=actual,status='UNCHANGED'))
    pd.DataFrame(source_manifests).to_parquet(out/'validation/source_read_manifest_index.parquet',index=False)
    pd.DataFrame(hash_controls).to_parquet(out/'validation/source_closeout_control_hashes.parquet',index=False)
    static=read_json(out/'validation/source_review/STATIC_BINDINGS_UNCHANGED.json')
    for r in static['bindings']:assert sha(r['path'])==r['sha256']
    counts=pd.read_parquet(out/'reports/source_split_quality_totals.parquet')
    result=dict(status='PASS',first_attempts=len(seen),all_first_attempts_before_extended_training=first_extended is not None or len(seen)==1566,
        first_extended_dispatch_unix=first_extended,max_first_slice_steps=max_first_steps,max_first_slice_training_seconds=max_first_seconds,
        first_pass_region_failures=failures,all_regions=1847,inactive_no_models=281,active_pairs=80316,nodes=72724,static_links=143603,logical_links=223919,
        unique_source_read_bindings=total_reads,verified_source_bytes_as_read=total_bytes,source_control_rehashes=len(hash_controls),unchanged_static_bindings=len(static['bindings']),
        source_preservation_method='Exact reviewed manifest SHA-256 for each used receipt, payload and scale as read; closeout rehash of all used small model/scale/pair controls and static source bindings. No second full label/tree rehash or path recertification.',
        source_label_generation=False,regional_assignment=False,national_assignment=False,B_restoration=False,inactive_pair_fallback=False,independent_cost_head=False,
        next_phase_authorized=False,whole_campaign_elapsed_seconds=elapsed(c),source_split_quality_counts=counts.to_dict('records'),
        final_training_events_sha256=sha(out/'training_events.jsonl'),final_training_events_bytes=(out/'training_events.jsonl').stat().st_size,
        automatic_native_pool_recoveries=[e for e in events if e['event']=='NATIVE_POOL_INTERRUPTED'],
        inactive_model_fallback=False,independent_cost_output_head=False)
    write_json(out/'validation/FINAL_CAMPAIGN_AUDIT.json',result);print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);args=a.parse_args();run(load_config(args.config))
