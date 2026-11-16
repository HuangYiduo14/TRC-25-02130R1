"""All-region integrity and protected input freezing, before model-dependent draws."""
import argparse,time,os
import pandas as pd
from .common import root,read_json,write_json,sha,native_threads
from .source import Sources,prepare
from .audit import panel_plan
from .sampling import freeze_heldouts
from .resources import Monitor,worker_setup

SOURCE=None
def init(c):
    global SOURCE
    worker_setup(c,'J0_prepare');SOURCE=Sources(c)
def one(c,rid):
    result=freeze_heldouts(c,SOURCE,rid);SOURCE.flush('J0_protected');return result
def main():
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    from .common import load_config
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=load_config(a.config);native_threads()
    prepare(c);panel_plan(c);monitor=Monitor(c,'J0_PREPARE').start();results=[]
    active=pd.read_parquet(root(c)/'control/source_region_registry.parquet').query('m_active>0')
    with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn'),initializer=init,initargs=(c,)) as pool:
        for f in as_completed([pool.submit(one,c,r) for r in active.region_id]):
            results.append(f.result())
            if len(results)%100==0:print('Protected regions',len(results),'/',len(active),flush=True)
    counts={k:sum(r['counts'][k] for r in results) for k in ['PATCH_VALID','PATCH_AUDIT_OPERATING','PATCH_AUDIT_STRESS']}
    manifests=[dict(region_id=r['region_id'],plan_sha256=r['plan_sha256'],input_npz_sha256=r['input_npz_sha256']) for r in results]
    pd.DataFrame(manifests).to_parquet(root(c)/'control/protected_manifest.parquet',index=False)
    write_json(root(c)/'validation/PROTECTED_INPUTS_FROZEN.json',dict(status='PASS',active_regions=len(results),counts=counts,
        manifest_sha256=sha(root(c)/'control/protected_manifest.parquet'),frozen_unix=time.time(),candidate_dependent_generation_started=False,
        old_native_packs_authenticated_on_first_read=True,hidden_audit_labels_unread=True))
    monitor.close();print(counts,flush=True)
if __name__=='__main__':main()
