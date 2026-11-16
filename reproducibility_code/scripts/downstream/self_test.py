"""Small numerical/interface verification, requiring no FAF or historical files."""
from pathlib import Path
import argparse
import sys
import time
import json
import tempfile
import os
import numpy as np

PACKAGE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PACKAGE/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    os.environ['NUMBA_CACHE_DIR']=str(out/'numba_cache')
    from partC_rev1_src.pacnhn_v1.control import save_npz,write_json,read_json,sha,array_hash
    from partC_rev1_src.pacnhn_v1.regional import load_region_graph,solve_label,sample_requests,generate_labels,load_admitted_dataset
    from partC_rev1_src.pacnhn_v1.models import Power4Model,fit_power4_cost,save_model,load_model
    from partC_rev1_src.pacnhn_v1.training import _structural_zeros
    from partC_rev1_src.pacnhn_recovery24.portfolio import choose_region,FrozenComposite
    from partC_rev1_src.pacnhn_recovery24.ablation import nested_indices,subset_train
    import frozen_fig5_training as fig5
    passed=[];started=time.time()
    graphpath=out/'synthetic_graph.npz'
    save_npz(graphpath,tail=np.array([0,0,1,1],np.int64),head=np.array([1,1,2,2],np.int64),
        node_ids=np.array(['a','b','c']),edge_ids=np.array(['ab1','ab2','bc1','bc2']),
        base=np.array([1.,2.,1.,2.]),k4=np.array([.01,0.,.01,0.]),
        pair_source=np.array([0,1],np.int64),pair_target=np.array([2,2],np.int64),pair_keys=np.array([2,5],np.uint64),
        baseline_open=np.array([5.,3.]),baseline_i95=np.array([3.,2.]),c0=np.array([2.,1.]))
    region=load_region_graph(graphpath,expected_sha256=sha(graphpath))
    solved=solve_label(region,np.array([4.,2.]),max_seconds=20,max_iterations=1000,method='CFW',cfw_reset_on_bound=True,
        iterate_selection='BEST_CERTIFIED_RG',relative_gap_tolerance=1e-6)
    assert solved['record']['admitted'] and solved['record']['certificate']['fresh']
    assert np.all(solved['tau']>=region.c0)
    passed.append('fresh_CFW_label_conservation_and_admission')
    requests=[];existing=set()
    for role,count in [('TRAIN',12),('VALID',6),('TEST',6)]:
        rows=sample_requests(region,role=role,count=count,seed=20260925,existing_input_ids=existing,reference_scale=np.array([5.,3.]))
        requests.extend(rows);existing.update(r['input_id'] for r in rows)
    assert len(existing)==24
    generated=generate_labels(dict(phase_id='PACNHN_V1',run_id='synthetic_interface_v1',region_id='synthetic',graph_path=str(graphpath),
        graph_sha256=sha(graphpath),deadline_epoch=time.time()+60,per_label_seconds=2,max_iterations=1000,relative_gap_tolerance=1e-6,
        solver_method='CFW',cfw_reset_on_bound=True,iterate_selection='BEST_CERTIFIED_RG'),out/'labels',requests=requests)
    assert generated['admitted_by_role']=={'TRAIN':12,'VALID':6,'TEST':6}
    train=load_admitted_dataset(out/'labels',role='TRAIN');valid=load_admitted_dataset(out/'labels',role='VALID')
    try:load_admitted_dataset(out/'labels',role='TEST')
    except PermissionError:pass
    else:raise AssertionError('TEST training loader leaked')
    passed.append('immutable_unique_roles_and_blind_TEST_loader')
    indices=nested_indices([r['input_identity'] for r in train['records']],20260928,'synthetic')
    assert all(set(indices[a])<=set(indices[b]) for a,b in zip((.1,.25,.5,.7),(.25,.5,.7,1.)))
    subset=subset_train(train,indices[.5])
    assert len(subset['u'])==6
    passed.append('nested_subset_identity_and_TRAIN_only_scale')
    # Fixed Figure5 pure functions receive these freshly certified tiny cohorts.
    icnn=fig5.fit_icnn(train,valid,20260928,lambda:None,time.time()+60,out/'icnn')
    xgb=fig5.fit_xgb(train,valid,20260928,lambda:None,time.time()+60,out/'xgb',_structural_zeros(region),
        'FROZEN_GRAPH_ZERO_COST_DIRECTED_REACHABILITY',threads=1)
    assert icnn['actual_updates']==1000 and xgb['actual_boosting_rounds']==32
    assert icnn['TRAIN']['true_cost_sum']>0 and xgb['VALID']['true_cost_sum']>0
    for result in (icnn,xgb):
        m=load_model(result['model']['path']);assert m.cost(train['u']).shape==train['tau'].shape
    passed.append('latest_ICNN1000_and_XGB32_pure_fits_portable_roundtrip')
    power=fit_power4_cost(train['u'],train['tau'],train['c0'],np.array([5.,3.]))
    receipt=save_model(power,out/'power4.npz');rebuilt=load_model(receipt['path'])
    np.testing.assert_array_equal(power.cost(train['u']),rebuilt.cost(train['u']))
    objective=FrozenComposite(np.array([1.,2.,1.]),np.zeros(3),np.array([1,2]),[(np.array([1,2]),power)])
    v,c=objective.value_cost(np.array([1.,4.,2.]));assert np.isfinite(v) and np.all(c>=0)
    assert choose_region('ICNN',{})[0]=='FFTT'
    passed.append('Power4_FrozenComposite_and_fail_closed_whole_region_fallback')
    # Verify portable path/config/context integration through P traces and all-five support.
    from run_downstream import Run,reference,support,labels,retry_labels,training,freeze,assignments,evaluate,load_test
    from partC_rev1_src.pacnhn_v1.topology import runtime_identities
    fixture=out/'integration';topology=fixture/'topology';topology.mkdir(parents=True)
    a=dict(tail=np.array([0,1,1,3,2],np.int64),head=np.array([1,2,2,0,4],np.int64),
        blocked=np.array([False,False,False,True,True]),node_ids=np.array(['a','b','c','o','d']),
        edge_ids=np.array(['B','R1','R2','O','D']),base=np.array([1.,1.,2.,0.,0.]),
        k4=np.array([0.,.01,0.,0.,0.]),closed_mask=np.zeros(5,bool),edge_role=np.array([0,1,1,2,2]),capacity=np.full(5,10.))
    o=dict(origins=np.array([3],np.int64),offsets=np.array([0,1],np.int64),destinations=np.array([4],np.int64),
        demand=np.array([6.]),serve=np.array([True]))
    p=dict(edge_region=np.tile(np.array([-1,0,0,-1,-1],np.int64),(5,1)))
    gi,oi=runtime_identities(a,o)
    save_npz(topology/'common_graph.npz',**a);save_npz(topology/'common_od.npz',**o);save_npz(topology/'partitions.npz',**p)
    write_json(topology/'RESULT.json',dict(status='PASS',graph_array_identity=array_hash(a),OD_array_identity=array_hash(o),
        partitions_identity=array_hash(p),graph_identity=gi,OD_identity=oi))
    config=out/'integration_config.json'
    write_json(config,dict(phase_id='REPRODUCIBILITY_SYNTHETIC',run_id='tiny_v1',run_root=str(fixture),
        tasks=['references','support','labels','retry-labels','train','freeze','assign','evaluate'],input_versions={'graph':'synthetic_nonempty_directed_fixture_v1'},
        expected_outputs=['fresh certified P traces','all-five active movement freeze'],acceptance_tests=['all OD witness segments lift'],
        resources=dict(total_wall_seconds=300,max_rss_bytes=24*2**30,min_available_bytes=512*2**20,min_free_disk_bytes=10**8,max_added_disk_bytes=10**9),
        expected_backbone_count=1,workers=1,action_seconds={a:120 for a in ['references','support','labels','retry-labels','train','freeze','assign','evaluate']},
        reference={'per_scenario_seconds':60,'relative_gap':1e-4},labels=dict(seed=20260925,counts={'TRAIN':12,'VALID':6,'TEST':6},attempt_seconds=[2,3]),
        training=dict(seed=20260925,region_seconds=60,icnn_widths=[2],icnn_learning_rates=[.001],updates_per_block=250,initial_blocks=1,extra_blocks=0,
            batch_rows=12,audit_batch_rows=16,xgb_round_increments=[8],xgb_region_raw_bytes=2**20),
        assignment=dict(per_query_seconds=5,relative_gap=.001)))
    ctx=Run(config,'references');report=reference(ctx);ctx.finish(report)
    ctx=Run(config,'support');report=support(ctx);ctx.finish(report)
    assert len(report['suites'])==5 and all(r['regions']==1 and r['movements']==1 for r in report['suites'])
    assert report['witness']['status']=='PASS'
    passed.append('portable_reference_support_context_and_complete_five_suite_witness_integration')
    for action,function in [('labels',labels),('retry-labels',retry_labels),('train',training),('freeze',freeze),('assign',assignments),('evaluate',evaluate)]:
        ctx=Run(config,action);report=function(ctx);ctx.finish(report)
    assert read_json(fixture/'assignments/RESULTS.json') and read_json(fixture/'evaluation/REGIONAL_TEST.json')
    passed.append('portable_fresh_labels_train_two_policies_assignment_and_postfreeze_TEST_integration')
    records_path=fixture/'training/NORMALIZED_RECORDS.json'
    original=read_json(records_path)
    changed=list(original);changed[0]=dict(changed[0],train_wmape_percent=0.)
    write_json(records_path,changed)
    frozen_support=read_json(fixture/'SUPPORT_FREEZE.json')
    record=next(r for s in frozen_support['suites'] if s['suite_id']=='s03' for r in s['region_records'])
    try:load_test(ctx,record)
    except ValueError as exc:assert 'hash differs' in str(exc) or 'changed during' in str(exc)
    else:raise AssertionError('Mutated model-selection metadata was allowed before TEST')
    write_json(records_path,original)
    passed.append('postfreeze_selection_metadata_mutation_rejected_before_TEST')
    write_json(out/'RESULT.json',dict(status='PASS',checks=passed,check_count=len(passed),elapsed_seconds=time.time()-started,
        no_FAF_files_read=True,no_historical_models_read=True,no_national_assignment=True,
        synthetic_graph_sha256=sha(graphpath),model_receipts=[icnn['model'],xgb['model'],receipt]))
    print(json.dumps(dict(status='PASS',check_count=len(passed))))


if __name__=='__main__':main()
