"""TRAIN/VALID comparison evidence and independently runnable code snapshots."""
from pathlib import Path
import os,sys,json,time,subprocess
import numpy as np
import pandas as pd
from .common import root,rd,read_json,write_json,sha,atomic


def snapshot_paths(repo):
    # Reuse the existing source handoff's AST import closure, including package
    # initializers. None of the historical scientific entrypoints is executed.
    from partC_rev1_src.phase6.handoff import dependency_snapshot
    repo=Path(repo)
    paths=dependency_snapshot(repo)
    for name in ['phase7','phase67']:
        paths.extend((repo/'partC_rev1_src'/name).glob('*.py'))
    paths.extend((repo/'tests').glob('test_phase67_*.py'))
    paths.extend((repo/'partC_doc_icnn/phase_7_work/p67_patch24h_0908').glob('*.py'))
    paths.extend((repo/'partC_doc_icnn/phase_7_work/p67_postaudit_e2_0908').glob('*.py'))
    return sorted(set(paths))


def snapshot_tests(c,snapshot,evidence):
    snapshot=Path(snapshot);evidence=Path(evidence);evidence.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(PYTHONPATH=str(snapshot),PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    runtime=root(c)/'handoff'/('snapshot_test_runtime_'+str(time.time_ns()));runtime.mkdir(parents=True)
    env['NUMBA_CACHE_DIR']=str(runtime/'numba_cache')
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
    # Test discovery and all local imports must originate in the snapshot.
    probe='import pathlib,sys; import partC_rev1_src.phase67.campaign; base=pathlib.Path.cwd(); bad=[(n,m.__file__) for n,m in sys.modules.items() if n.startswith("partC_rev1_src") and getattr(m,"__file__",None) and not pathlib.Path(m.__file__).is_relative_to(base)]; assert not bad,bad; print("SNAPSHOT_IMPORT_ISOLATION_PASS")'
    start=time.time();python=c['execution']['python_executable']
    imported=subprocess.run([python,'-c',probe],cwd=snapshot,env=env,capture_output=True,text=True,timeout=120)
    test=subprocess.run([python,'-m','pytest','-q','-p','no:cacheprovider','tests/test_phase67_patch.py','tests/test_phase67_e2.py',
        '--basetemp='+str(runtime/'fixtures'),'--junitxml='+str(evidence/'tests.xml')],cwd=snapshot,env=env,capture_output=True,text=True,timeout=180)
    atomic(evidence/'tests.log',(imported.stdout+imported.stderr+test.stdout+test.stderr).encode())
    result=dict(status='PASS' if imported.returncode==test.returncode==0 else 'FAIL',snapshot_root=str(snapshot),
        import_returncode=imported.returncode,test_returncode=test.returncode,seconds=time.time()-start,
        tests_xml_sha256=sha(evidence/'tests.xml') if (evidence/'tests.xml').exists() else None,
        tests_log_sha256=sha(evidence/'tests.log'),repository_import_fallback=False,packages_installed=False)
    write_json(evidence/'RESULT.json',result)
    assert result['status']=='PASS',result
    return result


def metric_rows(rid,stage,kind,metric,steps,best_steps):
    common=dict(region_id=rid,stage=stage,model_kind=kind,optimizer_updates=steps,selected_checkpoint_updates=best_steps,
        candidate_eligible=metric['eligible'],VALID_pool_score=metric['score'])
    rows=[dict(**common,split=split,**v) for split,v in metric.get('valid',{}).items() if v]
    if metric.get('train'):rows.append(dict(**common,split='TRAIN_old',**metric['train']))
    return rows


def existing_data_comparison(c,regions):
    from .training import PRIMARY,ESCALATION,LEGACY
    rows=[];coverage=[]
    for rid in regions:
        p=rd(c,rid)/'training/cycles/cycle_0000/manifest.json'
        if not p.exists():coverage.append(dict(region_id=rid,status='COMPARISON_FREEZE_UNRESOLVED'));continue
        m=read_json(p);assert m['old_data_only'] and m['new_TRAIN_AB']==0 and not m['TEST_HIGH_AUDIT_used']
        for kind,metric in m['metrics'].items():
            st=rd(c,rid)/'training'/kind/'status.json';s=read_json(st) if st.exists() else {}
            rows.extend(metric_rows(rid,'EXISTING_DATA_ONLY_CYCLE0',kind,metric,s.get('steps'),s.get('best_steps')))
            if kind in [PRIMARY,ESCALATION,LEGACY]:
                curve=rd(c,rid)/'training'/kind/'learning_curves.jsonl'
                if curve.exists():
                    with curve.open() as f:first=json.loads(f.readline())
                    assert first['steps']==0
                    rows.extend(metric_rows(rid,'INITIALIZATION_STEP_ZERO',kind,first['metrics'],0,0))
        coverage.append(dict(region_id=rid,status='FROZEN',champion_kind=m['champion_kind'],manifest_sha256=sha(p)))
    dest=root(c)/'reports/existing_data_comparison';dest.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_parquet(dest/'candidate_metrics.parquet',index=False);frame.to_csv(dest/'candidate_metrics.csv',index=False)
    pd.DataFrame(coverage).to_csv(dest/'coverage.csv',index=False)
    valid=frame.loc[frame.split.eq('VALID_old') & ~frame.nonfinite]
    summary=valid.groupby(['stage','model_kind']).agg(regions=('region_id','nunique'),MARE_equal_region=('MARE_equal_pairs','mean'),
        MARE_LOW_pair_equal_region=('MARE_LOW_pair_weighted','mean'),mean_interval_violation=('mean_interval_violation_relative','mean'),
        eligible_regions=('candidate_eligible','sum'),minimum_updates=('optimizer_updates','min'),maximum_updates=('optimizer_updates','max')).reset_index()
    summary.to_csv(dest/'VALID_equal_region_summary.csv',index=False)
    result=dict(status='FROZEN',regions=sum(r['status']=='FROZEN' for r in coverage),active_denominator=len(regions),
        new_TRAIN_labels=0,TEST_HIGH_AUDIT_used=False,rows=len(frame),metrics_sha256=sha(dest/'candidate_metrics.parquet'),path=str(dest))
    write_json(dest/'RESULT.json',result);return result
