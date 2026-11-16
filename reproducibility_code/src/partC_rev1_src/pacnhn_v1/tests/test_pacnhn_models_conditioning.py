"""Synthetic TRAIN-only conditioning gates; never open campaign labels."""
import copy
import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from partC_rev1_src.pacnhn_v1.models import (
    MonotoneICNN, Power4Model, calibrate_icnn_train, cost_wmape, fit_power4_cost,
    fit_xgboost_cost, load_model, save_model, update_icnn, ICNN_SCALED, XGB_CENTERED,
)
from partC_rev1_src.pacnhn_v1.training import _icnn_block, _state, read_training_config


def v2_config():
    return read_training_config(Path(__file__).resolve().parents[3]/'partC_doc_icnn/pacnhn_v1/training_config_v2.json')


def conditioned(cost_units=1.):
    scale=np.array([1e-5, 1e4, 3., 2., 1.])
    x=np.array([[0.,0.,0.,0.,0.],[.1,.5,0.,.5,0.],[1.,.2,0.,1.,0.],[3.,2.,0.,2.,0.]])
    u=x*scale
    c0=np.array([2.,1e5,4.,0.,0.])*cost_units
    eta=np.array([3.,200.,0.,0.,0.])*cost_units
    tau=c0+eta*x**4
    power=fit_power4_cost(u,tau,c0,scale)
    model=MonotoneICNN(c0,scale,eta=power.eta,width=5,seed=17,amplitude=123.)
    evidence=calibrate_icnn_train(model,u,tau,delta=.001,train_dataset_identity='toy_TRAIN')
    return model,u,tau,power,evidence


def test_conditioning_native_share_bound_and_positive_zero_coordinate_units(tmp_path):
    model,u,tau,power,evidence=conditioned()
    assert evidence['eta_positive_fallback_coordinates']==[2,3,4]
    assert evidence['VALID_or_TEST_used'] is False
    assert np.all(model.eta_units.numpy()>0)
    native=model.export_model()
    neural=native.amplitude*native.unit_neural_cost(u)
    np.testing.assert_allclose(neural.sum()/tau.sum(),.001,rtol=2e-14)
    assert cost_wmape(native.cost(u),tau)<=cost_wmape(power.cost(u),tau)+.100000001
    np.testing.assert_array_equal(native.cost(np.zeros(5)),power.c0)
    assert native.value_cost(np.zeros(5))[0]==0.
    q=u[-1];h=np.column_stack([native.hess_vec(q,d) for d in np.eye(5)])
    assert h.min()>=0
    np.testing.assert_allclose(h,h.T,rtol=2e-14,atol=1e-12)
    assert np.linalg.eigvalsh(h).min()>=-1e-8
    p=tmp_path/'model.npz';save_model(model,p)
    np.testing.assert_array_equal(load_model(p).cost(u),native.cost(u))
    assert load_model(p).metadata['coefficient_parameterization']==ICNN_SCALED
    with pytest.raises(ValueError,match='cannot be recalibrated'):
        calibrate_icnn_train(model,u,tau)


def test_conditioning_cost_unit_invariance_and_update_responsiveness():
    torch.set_num_threads(1)
    left,u,tau,_,_=conditioned();right,_,scaled,_,_=conditioned(1e8)
    np.testing.assert_allclose(right.eta_units.numpy()/1e8,left.eta_units.numpy(),rtol=2e-14)
    np.testing.assert_allclose(right.amplitude_unit.numpy()/1e8,left.amplitude_unit.numpy(),rtol=2e-14)
    np.testing.assert_allclose(left.eta_coordinate.detach(),right.eta_coordinate.detach(),rtol=2e-14)
    optimizers=[torch.optim.Adam(m.parameters(),lr=.001) for m in (left,right)]
    # Demand a larger neural correction so its normalized amplitude responds.
    target=tau+left.export_model().unit_neural_cost(u)*float(left.amplitude_unit)*.1
    eta_before=left.eta_coordinate.detach().clone();gamma_before=float(left.amplitude_coordinate.detach())
    for model,opt,y in zip((left,right),optimizers,(target,target*1e8)):
        update_icnn(model,opt,u,y,train_cost_mean=float(y.mean()))
    assert float(left.amplitude_coordinate.detach())>gamma_before+.0009
    assert torch.max(torch.abs(left.eta_coordinate.detach()-eta_before))>.0009
    np.testing.assert_allclose(left.cost(u),right.cost(u)/1e8,rtol=1e-11,atol=1e-8)
    with torch.no_grad():left.amplitude_coordinate.fill_(-1.)
    left.project_parameters()
    assert float(left.amplitude.detach())==0.


def test_conditioning_zero_excitation_is_explicit_not_nan():
    model=MonotoneICNN([2.,0.],[1.,1.],width=3)
    result=calibrate_icnn_train(model,np.zeros((3,2)),np.tile([2.,0.],(3,1)))
    assert result['zero_neural_excitation'] and float(model.amplitude.detach())==0.
    assert (model.eta_units.numpy()>0).all()
    with pytest.raises(ValueError,match='positive cost sum'):
        calibrate_icnn_train(MonotoneICNN([0.],[1.]),[[0.]],[[0.]])


def test_conditioning_checkpoint_resume_is_exact_and_units_stay_frozen(tmp_path):
    torch.set_num_threads(1)
    model,u,tau,_,_=conditioned()
    data=dict(u=u,tau=tau,c0=model.c0.numpy(),train_cost_mean=float(tau.mean()),dataset_identity='original_TRAIN')
    c=v2_config();c.update(icnn_updates_per_block=2,icnn_widths=[5],icnn_batch_rows=4)
    split=tmp_path/'split';split.mkdir();state=_state(split/'STATE.json')
    _icnn_block(split,state,data,model.scale.numpy(),c,'cpu',time.time()+60,lambda:None)
    first=torch.load(state['optimizers']['5']['path'],weights_only=True)
    _icnn_block(split,state,data,model.scale.numpy(),c,'cpu',time.time()+60,lambda:None)
    last=torch.load(state['optimizers']['5']['path'],weights_only=True)
    direct=tmp_path/'direct';direct.mkdir();state2=_state(direct/'STATE.json');c['icnn_updates_per_block']=4
    _icnn_block(direct,state2,data,model.scale.numpy(),c,'cpu',time.time()+60,lambda:None)
    expected=torch.load(state2['optimizers']['5']['path'],weights_only=True)
    for name,value in expected['model'].items():torch.testing.assert_close(last['model'][name],value,rtol=0,atol=0)
    grown=dict(data,tau=tau*2,dataset_identity='grown_TRAIN',train_cost_mean=float(tau.mean()*2))
    _icnn_block(split,state,grown,model.scale.numpy(),c,'cpu',time.time()+60,lambda:None)
    grown_state=torch.load(state['optimizers']['5']['path'],weights_only=True)
    for key in ('eta_units','amplitude_unit'):
        torch.testing.assert_close(grown_state['model'][key],first['model'][key],rtol=0,atol=0)
    assert grown_state['TRAIN_calibration']['train_dataset_identity']=='original_TRAIN'
    wrong=copy.deepcopy(c);wrong['icnn_coefficient_parameterization']='NATIVE_PROJECTED_V1'
    with pytest.raises(ValueError,match='migration prohibited'):
        _icnn_block(split,state,data,model.scale.numpy(),wrong,'cpu',time.time()+60,lambda:None)


def test_centered_xgb_constants_continuation_and_legacy_roundtrip(tmp_path):
    pytest.importorskip('xgboost')
    u=np.array([[0.,0.,0.],[1.,2.,3.],[2.,4.,6.],[3.,6.,9.]])
    tau=np.tile([1e-8,1e8,0.],(4,1));c0=np.array([1e-8,1e8,0.])
    kw=dict(n_estimators=2,max_depth=1,structural_zero_mask=[False,False,True],
        structural_zero_evidence='analytic_toy',log_center_policy=XGB_CENTERED,train_dataset_identity='first')
    first=fit_xgboost_cost(u,tau,c0,np.ones(3),**kw)
    np.testing.assert_allclose(first.cost(u),tau,rtol=2e-14,atol=0.)
    assert float(json.loads(first.booster.save_config())['learner']['learner_model_param']['base_score'])==0.
    centered=np.array(first.log_center)
    kw['train_dataset_identity']='grown'
    continued=fit_xgboost_cost(u,tau*2,c0,np.ones(3),initial_model=first,**kw)
    np.testing.assert_array_equal(continued.log_center,centered)
    assert continued.metadata['log_center_train_dataset_identity']=='first'
    p=tmp_path/'centered.npz';save_model(continued,p)
    np.testing.assert_array_equal(load_model(p).cost(u),continued.cost(u))
    with pytest.raises(ValueError,match='different native/model contract'):
        fit_xgboost_cost(u,tau,c0,np.ones(3),initial_model=first,n_estimators=1,
            structural_zero_mask=[False,False,True],structural_zero_evidence='analytic_toy')
    # Old inference archives had no center array or center metadata at all.
    legacy=fit_xgboost_cost(u[:,:2],tau[:,:2],c0[:2],np.ones(2),n_estimators=1)
    old=tmp_path/'old.npz';save_model(legacy,old)
    with np.load(old,allow_pickle=False) as z:arrays={k:z[k] for k in z.files if k!='log_center'}
    header=json.loads(str(arrays['header'].item()))
    for key in list(header['metadata']):
        if key.startswith('log_center'):header['metadata'].pop(key)
    arrays['header']=np.array(json.dumps(header))
    old2=tmp_path/'actual_legacy.npz';np.savez_compressed(old2,**arrays)
    np.testing.assert_array_equal(load_model(old2).cost(u[:,:2]),legacy.cost(u[:,:2]))


def test_centered_xgb_cost_unit_invariance():
    u=np.arange(12,dtype=float).reshape(6,2)/10
    tau=np.column_stack([2.+u[:,0]**2,1e6*(3.+u[:,1]**2)])
    kw=dict(n_estimators=4,max_depth=2,log_center_policy=XGB_CENTERED)
    left=fit_xgboost_cost(u,tau,[2.,3e6],[1.,1.],**kw)
    right=fit_xgboost_cost(u,tau*1e7,[2e7,3e13],[1.,1.],**kw)
    np.testing.assert_allclose(left.cost(u),right.cost(u)/1e7,rtol=1e-7)
