"""Synthetic metadata-only timing boundary and duplicate-accounting tests."""
import csv,json
from pathlib import Path
import pytest
from partC_rev1_src.pacnhn_recovery24 import offline_timing as t


def visit(status='FAILED_PRESERVED_PRIOR_CHECKPOINTS'):
    return dict(region_key='s03:000012',visit=1,status=status,start_epoch=10.,end_epoch=20.,elapsed_seconds=10.,
                timings=dict(dataset_load=1.,initial_audit_and_baselines=2.,icnn_fit_and_audit=3.,xgboost_fit_and_audit=4.))


def methods():
    rows=[]
    for scenario in ['OPEN','REPRESENTED_I95']:
        for family in ['REFERENCE','CFW','FW','MSA']:
            rows.append(dict(case_id='P_'+family+'_'+scenario,network='P',suite_id='',family=family,scenario=scenario,status='CONVERGED'))
        for sid in ['s02','s03','s04','s05','s06']:
            for family in ['ICNN','XGBOOST','POWER4','FFTT']:
                rows.append(dict(case_id=sid+'_'+family+'_'+scenario,network='AC',suite_id=sid,family=family,scenario=scenario,status='TIME_LIMIT'))
    return rows


def test_failed_unqualified_candidate_worker_time_remains_counted():
    rows=t.visit_rows(visit(),{})
    summary=t.family_summaries(rows,['s03'])
    assert [r['recorded_training_seconds'] for r in summary]==[3.,4.]
    assert sum(r['seconds'] for r in rows if r['category']=='SHARED_INITIAL_AUDIT_AND_BASELINES')==2.
    assert all(r['accounting_role']!='ADDITIVE_OFFLINE_TOTAL' for r in rows)


def test_missing_family_timer_is_unknown_not_zero_or_proportional_allocation():
    v=visit();del v['timings']['icnn_fit_and_audit']
    summary=t.family_summaries(t.visit_rows(v,{}),['s03'])
    assert summary[0]['recorded_training_seconds'] is None
    assert summary[0]['missing_timers']==1 and summary[1]['recorded_training_seconds']==4.


def test_explicit_preserved_both_targets_means_zero_incremental_fit():
    v=visit('PRESERVED_BOTH_TARGETS_MET_NO_FITTING');v['timings']={'dataset_load':1.,'initial_audit_and_baselines':2.}
    summary=t.family_summaries(t.visit_rows(v,{}),['s03'])
    assert [r['recorded_training_seconds'] for r in summary]==[0.,0.]
    assert t.family_summaries([],['s02'])[0]['recorded_training_seconds'] is None


def test_multiple_visits_add_workers_not_parent_visit_total():
    first=t.visit_rows(visit(),{});v=visit();v['visit']=2;second=t.visit_rows(v,{})
    summary=t.family_summaries(first+second,['s03'])
    assert [r['recorded_training_seconds'] for r in summary]==[6.,8.]


@pytest.mark.parametrize('bad',[-1,float('nan'),float('inf'),True,'bad'])
def test_invalid_timing_rejected(bad):
    v=visit();v['timings']['icnn_fit_and_audit']=bad
    with pytest.raises((ValueError,TypeError)):t.visit_rows(v,{})


def test_nested_timing_cannot_exceed_parent():
    v=visit();v['elapsed_seconds']=5.
    with pytest.raises(ValueError):t.visit_rows(v,{})


def test_method_augmentation_preserves_all_rows_and_repeats_not_sums_scenario_time():
    source=methods();before=[dict(r) for r in source]
    summaries=t.family_summaries(t.visit_rows(visit(),{}),['s02','s03','s04','s05','s06'])
    result=t.attach_training(source,summaries)
    assert len(result)==48 and source==before
    for row in result:
        if row['network']=='P' or row['family']=='FFTT':assert row['recorded_training_seconds']==0.
        elif row['family']=='POWER4':assert row['recorded_training_seconds'] is None
        elif row['suite_id']=='s03':assert row['recorded_training_seconds']==(3. if row['family']=='ICNN' else 4.)
        else:assert row['recorded_training_seconds'] is None
    with pytest.raises(ValueError):t.attach_training(result,summaries)


def test_stage_deadline_or_file_timestamp_is_not_completion():
    row=t.stage_row({'status':'TIME_LIMIT','deadline_epoch':100.},{'started_epoch':5.},{},'stage')
    assert row['seconds'] is None and row['end_epoch'] is None
    row=t.stage_row({'status':'PRIMARY_TIME_LIMIT','epoch':10.},{'started_epoch':5.},{},'stage')
    assert row['seconds']==5.
    with pytest.raises(ValueError):t.stage_row({'start_epoch':10.,'end_epoch':9.},{},{},'stage')


def test_sum_overflow_rejects():
    with pytest.raises(ValueError):t.total([1e308,1e308])


@pytest.mark.parametrize('historical',['absent','valid','changed','wrong_visit','wrong_role'])
def test_tiny_export_reads_metadata_only_separates_scopes_and_preserves_sources(tmp_path,historical):
    root=tmp_path/'run';root.mkdir();sources=[]
    def save(relative,record):
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record));sources.append(path);return path
    method_path=root/'report_data/table2.csv';method_path.parent.mkdir()
    with method_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(methods()[0]));writer.writeheader();writer.writerows(methods())
    sources.append(method_path)
    save('report_data/MANIFEST.json',dict(status='VERIFIED_REPORT_DATA',files=[dict(path=str(method_path),sha256=t.sha(method_path))]))
    save('training/regions/s03/000012/visits/000001.json',visit())
    # Duplicate wrapper is intentionally not another canonical visit.
    save('training/passes/p1/s03_000012.json',visit())
    save('execution/stage/INTENT.json',dict(started_epoch=100.))
    save('execution/stage/RESULT.json',dict(started_epoch=100.,ended_epoch=120.,status='DONE'))
    save('labels/overlays/s03/000012/labels/TRAIN/one/attempt_000005/START.json',dict(role='TRAIN',epoch=1.))
    save('labels/overlays/s03/000012/labels/TRAIN/one/attempt_000005/RESULT.json',dict(role='TRAIN',status='TIME_LIMIT',complete_label_elapsed_seconds=2.))
    save('holdout/results/s03/000012/attempts/test1/START.json',dict(role='TEST',epoch=1.))
    # This invalid NPZ must never be opened by the metadata-only exporter.
    payload=root/'holdout/results/s03/000012/attempts/test1/label.npz';payload.write_bytes(b'NOT_AN_ARRAY');sources.append(payload)
    save('ablation/fits/task1/RESULT.json',dict(task=dict(task_id='task1',region_key='s03:000012',family='ICNN'),status='FAILED',elapsed_seconds=6.))
    save('resources/old_scope.json',dict(peak_rss_bytes=123456))
    historical_root=tmp_path/'old';settings={}
    if historical in {'valid','changed'}:
        old_csv=historical_root/'analysis/timing.csv';old_csv.parent.mkdir(parents=True)
        old_csv.write_text('task,category,seconds,status,time_kind,accounting_role,timing_boundary\nOld fit,WORKER_FIT,7,MEASURED,WORKER_SECONDS,NESTED_DETAIL,Old family call\n')
        old_result=historical_root/'RESULT.json';old_result.write_text(json.dumps(dict(status='PASS_WITH_LIMITATIONS',
            additive_disjoint_seconds=99.,inputs=[dict(path=str(old_csv),sha256=t.sha(old_csv))],historical_rows=[])))
        seal=historical_root/'SEAL.json';seal.write_text(json.dumps(dict(status='VERIFIED',result=dict(path=str(old_result),sha256=t.sha(old_result)))))
        settings={'historical_seals':[dict(path=str(seal),sha256=t.sha(seal))]}
        sources.extend([old_csv,old_result,seal])
        if historical=='changed':old_csv.write_text('changed')
    if historical=='wrong_visit':
        wrong=visit();wrong['region_key']='s02:000012'
        (root/'training/regions/s03/000012/visits/000001.json').write_text(json.dumps(wrong))
    if historical=='wrong_role':
        (root/'labels/overlays/s03/000012/labels/TRAIN/one/attempt_000005/RESULT.json').write_text(json.dumps(dict(role='TEST',status='DONE',complete_label_elapsed_seconds=2.)))
    class Ctx:
        c={'suite_ids':['s02','s03','s04','s05','s06']};config_sha256='config';step_id='publication'
        def require_action(self,action):assert action=='publication'
        def guard(self,*args):pass
        def output_path(self,relative):return self.root/relative
        def bind(self,path,role,expected=None):
            digest=t.sha(path)
            if expected is not None:assert digest==expected
            return digest
        def flush_bindings(self):pass
    ctx=Ctx();ctx.root=root;ctx.source_root=historical_root;ctx.c=dict(ctx.c,offline_timing=settings)
    before={str(p):t.sha(p) for p in sources}
    if historical=='changed':
        with pytest.raises(AssertionError):t.export(ctx,deadline=1e12)
        assert not (root/'report_timing/v1').exists()
        return
    if historical in {'wrong_visit','wrong_role'}:
        with pytest.raises(ValueError):t.export(ctx,deadline=1e12)
        assert not (root/'report_timing/v1').exists()
        return
    result=t.export(ctx,deadline=1e12)
    assert result['canonical_visits']==1 and result['started_label_attempts']==2
    assert result['overall_offline_seconds'] is None
    with (root/'report_timing/v1/timing_observations.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert next(r for r in rows if r['category']=='RECOVERY_STAGE_WALL')['seconds']=='20.0'
    assert next(r for r in rows if r['category']=='NEW_LABEL_PRODUCER_TRAIN')['seconds']=='2.0'
    assert next(r for r in rows if r['category']=='NEW_LABEL_PRODUCER_TEST')['seconds']==''
    assert next(r for r in rows if r['category']=='ABLATION_FIT_WORKER')['seconds']=='6.0'
    if historical=='valid':
        old=next(r for r in rows if r['category']=='HISTORICAL_WORKER_FIT')
        assert old['seconds']=='7.0' and old['accounting_role']=='HISTORICAL_CONTEXT_NOT_RECOVERY_WORK'
    assert before=={str(p):t.sha(p) for p in sources}
    with pytest.raises(FileExistsError):t.export(ctx,deadline=1e12)
