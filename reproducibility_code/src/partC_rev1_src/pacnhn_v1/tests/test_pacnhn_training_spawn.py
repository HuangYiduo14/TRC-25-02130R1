"""Two isolated synthetic regional workers; no campaign/national inputs.

This test intentionally performs tiny CPU fits and should run only outside a
timed national measurement. Spawn startup has a separate generous test deadline.
"""
from concurrent.futures import ProcessPoolExecutor, wait
import multiprocessing
from pathlib import Path
import time

import numpy as np

from partC_rev1_src.pacnhn_v1.control import read_json,save_npz,sha,write_json
from partC_rev1_src.pacnhn_v1.models import load_model
from partC_rev1_src.pacnhn_v1.regional import load_region_graph
from partC_rev1_src.pacnhn_v1.training import read_training_config,regional_worker


def test_two_spawn_workers_keep_models_graph_and_sample_roles_isolated(tmp_path):
    started=time.time();deadline=started+180.
    contract=tmp_path/'CONTRACT.md';contract.write_text('Synthetic two-worker regression only.\n')
    config_path=tmp_path/'config.json'
    write_json(config_path,dict(phase_id='PACNHN_V1',tasks=['training'],run_id='synthetic_spawn',
        input_versions={'graph':'synthetic_one_arc'},expected_outputs=['two_region_states'],
        acceptance_tests=['distinct_files','graph_identity','role_identity','matching_predictions'],
        output_root=str(tmp_path/'output'),contract_path=str(contract),contract_sha256=sha(contract),
        hard_deadline_epoch=deadline,resources=dict(tree_rss_bytes=24*1024**3,
            minimum_available_bytes=3*1024**3,added_disk_bytes=250*10**9)))
    graph_path=tmp_path/'graph.npz'
    save_npz(graph_path,tail=np.array([0],np.int64),head=np.array([1],np.int64),
        node_ids=np.array(['P0','P1']),edge_ids=np.array(['toy']),base=np.array([2.]),k4=np.array([0.]),
        pair_source=np.array([0],np.int64),pair_target=np.array([1],np.int64),pair_keys=np.array([1],np.uint64),
        baseline_open=np.array([1.]),baseline_i95=np.array([2.]),c0=np.array([2.]))
    graph=load_region_graph(graph_path,expected_sha256=sha(graph_path))
    source=Path(__file__).resolve().parents[3]/'partC_doc_icnn/pacnhn_v1/training_config.json'
    settings=read_training_config(source)
    settings.update(initial_train=4,fixed_valid=2,blind_test=2,train_growth=2,icnn_updates_per_block=2,
        icnn_widths=[8],xgb_rounds_per_block=2,xgb_max_depth=1,xgb_depth_candidates=[1],
        per_label_seconds=5.,region_slice_seconds=60.)
    payloads=[]
    for index in range(2):
        payloads.append(dict(config_path=str(config_path),
            record=dict(path=str(graph_path),sha256=sha(graph_path),key=f's03:{index:06d}',suite_id='s03',region_index=index),
            region_root=str(tmp_path/'output/regions/s03'/f'{index:06d}'),settings=settings,
            device='cpu',science_deadline=deadline-10,slice_seconds=60.))
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(regional_worker,payload) for payload in payloads]
        done,pending=wait(futures,timeout=170.)
        # Worker guard provides a second bound if startup consumed the test wait.
        assert not pending,'Bounded two-worker synthetic fit did not finish'
        states=[future.result() for future in futures]
    assert time.time()<deadline
    all_paths=[];predictions=[];role_sets=[]
    for index,(state,payload) in enumerate(zip(states,payloads)):
        root=Path(payload['region_root'])
        assert state['region_state_path']==str(root/'STATE.json')
        assert state['visits']==1 and state['failure_count']==0
        assert state['training_rows']==4 and state['valid_rows']==2
        assert set(state['selected'])=={'ICNN','XGBOOST'}
        identity=read_json(root/'labels/REGION_IDENTITY.json')
        assert identity['graph_identity']==graph.identity and identity['graph_sha256']==graph.graph_sha256
        assert identity['region_id']==f's03:{index:06d}'
        records=read_json(root/'labels/REQUEST_INDEX.json')
        assert len(records)==8
        roles={role:{key for key,r in records.items() if r['role']==role} for role in ('TRAIN','VALID','TEST')}
        assert [len(roles[role]) for role in ('TRAIN','VALID','TEST')]==[4,2,2]
        assert not roles['TRAIN']&roles['VALID'] and not roles['TRAIN']&roles['TEST'] and not roles['VALID']&roles['TEST']
        role_sets.append(roles)
        assert not (root/'TEST_OPENING_AUTHORIZATION.json').exists()
        selected_predictions={}
        for family,selected in state['selected'].items():
            model_path=Path(selected['model_path'])
            assert model_path.is_relative_to(root)
            before=sha(model_path);assert before==selected['model_sha256']
            assert selected['audit']['train_dataset_identity']==state['dataset_identity']
            assert selected['audit']['valid_dataset_identity']==state['valid_dataset_identity']
            model=load_model(model_path)
            prediction=model.cost(np.array([[0.],[1.],[3.]]))
            assert prediction.shape==(3,1) and np.isfinite(prediction).all() and (prediction>=0).all()
            np.testing.assert_allclose(prediction,load_model(model_path).cost(np.array([[0.],[1.],[3.]])),rtol=0,atol=0)
            assert sha(model_path)==before
            selected_predictions[family]=prediction;all_paths.append(model_path.resolve())
        predictions.append(selected_predictions)
        receipt=read_json(root/'timing/visit_000001.json')
        assert 0<receipt['complete_slice_seconds']<170.
    assert len(set(all_paths))==4 and role_sets[0]==role_sets[1]
    for family in ('ICNN','XGBOOST'):
        np.testing.assert_allclose(predictions[0][family],predictions[1][family],rtol=1e-12,atol=1e-12)
