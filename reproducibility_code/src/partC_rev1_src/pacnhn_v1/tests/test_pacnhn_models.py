"""Small analytic/synthetic gates only; no national files or label jobs."""
import numpy as np
import pytest
import torch

from partC_rev1_src.pacnhn_v1.models import (
    MonotoneICNN, PortableICNN, Power4Model, cost_wmape, fit_power4_cost,
    fit_xgboost_cost, load_model, save_model, update_icnn,
)


def neural():
    return MonotoneICNN([2., 5., .4], [3., .5, 7.], eta=[.3, .1, 0.],
                        width=5, seed=7, amplitude=11.)


def test_pacnhn_models_native_derivatives_and_entire_hessian():
    model = neural(); portable = model.export_model()
    q = np.array([2., .7, 4.]); d = np.array([.2, -.1, .4]); eps = 1e-5
    value, cost = portable.value_cost(q)
    finite_gradient = np.array([(portable.value_cost(q+eps*e)[0]-portable.value_cost(q-eps*e)[0])/(2*eps)
                               for e in np.eye(3)])
    np.testing.assert_allclose(cost, finite_gradient, rtol=2e-8, atol=2e-8)
    finite_hvp = (portable.cost(q+eps*d)-portable.cost(q-eps*d))/(2*eps)
    np.testing.assert_allclose(portable.hess_vec(q, d), finite_hvp, rtol=2e-7, atol=2e-8)
    hessian = np.column_stack([portable.hess_vec(q, e) for e in np.eye(3)])
    np.testing.assert_allclose(hessian, hessian.T, rtol=1e-13, atol=1e-13)
    assert hessian.min() >= -1e-14
    assert np.linalg.eigvalsh(hessian).min() >= -1e-12
    tv, tc = model.value_cost(q)
    np.testing.assert_allclose(tv, value, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(tc, cost, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(model.hess_vec(q, d), portable.hess_vec(q, d), rtol=2e-13, atol=2e-13)
    independent = torch.tensor(q, dtype=torch.float64, requires_grad=True)
    true_gradient = torch.autograd.grad(model(independent), independent)[0]
    np.testing.assert_allclose(tc, true_gradient.detach().numpy(), rtol=2e-13, atol=2e-13)


def test_pacnhn_models_anchor_floor_batch_stress_and_dtype():
    model = neural().export_model()
    zero = np.zeros(3)
    value, cost = model.value_cost(zero)
    assert value == 0.
    np.testing.assert_array_equal(cost, model.c0)
    q = np.array([[0, 0, 0], [1e-9, 2e-9, 3e-9], [1e3, 2e3, 3e3]], dtype=np.float32)
    values, costs = model.value_cost(q)
    assert values.dtype == costs.dtype == np.float64
    assert np.isfinite(values).all() and np.isfinite(costs).all()
    assert values.min() >= -1e-12
    assert np.min(costs-model.c0) >= -1e-12
    for j in range(3):
        larger = q.astype(np.float64); larger[:, j] += .5
        assert np.min(model.cost(larger)-costs) >= -1e-12
    with pytest.raises(ValueError):
        model.cost([-1, 0, 0])
    with pytest.raises(ValueError):
        model.cost([1+1j, 0, 0])
    with pytest.raises(FloatingPointError):
        model.cost([1e100, 1e100, 1e100])


def test_pacnhn_models_zero_floor_survives_batch_anchor_roundoff():
    trainable = MonotoneICNN(np.zeros(8), np.ones(8), width=7, seed=1, amplitude=100.)
    model = trainable.export_model()
    q = np.vstack([np.zeros((32, 8)), np.full((32, 8), 1e-20), np.full((32, 8), 1e-12)])
    value, cost = model.value_cost(q)
    np.testing.assert_array_equal(value[:32], np.zeros(32))
    np.testing.assert_array_equal(cost[:32], np.zeros((32, 8)))
    assert cost.min() >= 0.
    assert value.min() >= 0.
    torch_value, torch_cost = trainable.value_cost(torch.tensor(q, dtype=torch.float64))
    np.testing.assert_array_equal(torch_value[:32], np.zeros(32))
    np.testing.assert_array_equal(torch_cost[:32], np.zeros((32, 8)))
    assert float(torch_cost.min()) >= 0.
    assert float(torch_value.min()) >= 0.
    tiny=np.full(8,1e-20)
    assert model.value_cost(tiny)[0]>0.
    # Extreme signed biases remain permitted and cannot produce negative increments.
    a = model.export_arrays(); a['b1'][:] = np.linspace(-1000., 1000., model.width)
    a['b2'][:] = np.linspace(1000., -1000., model.width)
    stress = PortableICNN(a)
    assert stress.cost(q).min() >= 0.
    assert np.isfinite(stress.cost(np.full((4, 8), 1e4))).all()


def test_pacnhn_models_power4_direct_l1_fit_and_integral():
    source = Power4Model([2., 3.], [.5, 2.], [.7, .2])
    q = np.array([[0, 0], [.1, .2], [.3, 1.], [.5, 3.], [1., 4.]])
    for loss in ("l1", "l2"):
        fit = fit_power4_cost(q, source.cost(q), source.c0, source.scale, loss=loss)
        np.testing.assert_allclose(fit.eta, source.eta, rtol=1e-14)
        assert cost_wmape(fit.cost(q), source.cost(q)) < 1e-12
    # Native absolute-error optimum is a basis-weighted median, not relative NNLS.
    fit = fit_power4_cost([[1.], [1.], [1.]], [[2.], [3.], [100.]], [1.], [1.])
    np.testing.assert_array_equal(fit.eta, [2.])
    fit = fit_power4_cost([[0.], [0.]], [[3.], [4.]], [2.], [1.])
    np.testing.assert_array_equal(fit.eta, [0.])
    point = np.array([.4, .8]); eps = 1e-6
    derivative = np.array([(source.value_cost(point+eps*e)[0]-source.value_cost(point-eps*e)[0])/(2*eps)
                           for e in np.eye(2)])
    np.testing.assert_allclose(derivative, source.cost(point), rtol=1e-9)


def test_pacnhn_models_updates_train_direct_cost_and_preserve_constraints():
    torch.set_num_threads(1)
    model = neural()
    q = np.array([[.2, .3, .5], [.4, .7, 1.], [1., .2, .3], [.8, .6, 2.]])
    target = model.export_model().cost(q)*1.05
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    before = cost_wmape(model.cost(q), target)
    for _ in range(12):
        evidence = update_icnn(model, optimizer, q, target, train_cost_mean=float(target.mean()))
    assert evidence["batch_rows"] == len(q)
    assert cost_wmape(model.cost(q), target) < before
    portable = model.export_model()
    assert (portable.eta >= 0).all() and portable.amplitude >= 0
    assert all((portable.arrays[k] >= 0).all() for k in ("A1", "A2", "W2", "v"))
    assert all(p.dtype == torch.float64 for p in model.parameters())


def test_pacnhn_models_save_load_restore_and_reject_overwrite(tmp_path):
    model = neural(); q = np.array([[.2, .4, .6], [1., 2., 3.]])
    path = tmp_path / "neural.npz"
    receipt = save_model(model, path, metadata={"role": "synthetic_test"})
    assert len(receipt["sha256"]) == 64
    loaded = load_model(path)
    assert isinstance(loaded, PortableICNN)
    assert loaded.metadata["role"] == "synthetic_test"
    np.testing.assert_allclose(loaded.cost(q), model.cost(q), rtol=1e-13)
    restored = MonotoneICNN.from_portable(loaded)
    np.testing.assert_allclose(restored.cost(q), loaded.cost(q), rtol=1e-13)
    with pytest.raises(FileExistsError):
        save_model(model, path)
    arrays = loaded.export_arrays(); arrays["A2"][0, 0] = -.1
    with pytest.raises(ValueError, match="nonnegative"):
        PortableICNN(arrays)
    with pytest.raises(ValueError):
        cost_wmape([0.], [0.])


def test_pacnhn_models_xgb_cost_only_roundtrip_and_monotonicity(tmp_path):
    pytest.importorskip("xgboost")
    axis = np.linspace(0., 2., 6)
    q = np.array([(x, y) for x in axis for y in axis], dtype=np.float64)
    target = np.column_stack([2.+q[:, 0]+.2*q[:, 1], 4.+.5*q[:, 0]+q[:, 1]])
    model = fit_xgboost_cost(q, target, [2., 4.], [1., 1.], n_estimators=8,
                             max_depth=2, learning_rate=.2, n_jobs=1, monotone=True)
    costs = model.cost(q)
    assert costs.shape == target.shape and costs.dtype == np.float64
    assert np.isfinite(costs).all() and (costs > 0).all()
    assert not model.has_potential and not hasattr(model, "value_cost") and not hasattr(model, "hess_vec")
    for j in range(2):
        larger = q.copy(); larger[:, j] += .2
        assert np.min(model.cost(larger)-costs) >= -1e-12
    path = tmp_path / "xgb.npz"; save_model(model, path)
    loaded = load_model(path)
    np.testing.assert_array_equal(loaded.cost(q), costs)
    assert loaded.cost(q[0]).shape == (2,)
    with pytest.raises(ValueError, match="positive"):
        fit_xgboost_cost(q, np.zeros_like(target), [2., 4.], [1., 1.], n_estimators=1)


def test_pacnhn_models_xgb_verified_structural_zero_and_continuation(tmp_path):
    pytest.importorskip("xgboost")
    q = np.array([[0., 0.], [1., 2.], [3., 4.]])
    target = np.column_stack([np.zeros(len(q)), 2.+q[:, 0]+q[:, 1]])
    kw = dict(structural_zero_mask=[True, False], structural_zero_evidence="synthetic_zero_arc_proof",
              n_estimators=2, max_depth=1)
    first = fit_xgboost_cost(q, target, [0., 2.], [1., 1.], **kw)
    continued = fit_xgboost_cost(q, target, [0., 2.], [1., 1.], initial_model=first, **kw)
    assert first.booster.num_boosted_rounds() == 2
    assert continued.booster.num_boosted_rounds() == 4
    np.testing.assert_array_equal(continued.cost(q)[:, 0], np.zeros(len(q)))
    path = tmp_path / "partial_zero.npz"; save_model(continued, path)
    np.testing.assert_array_equal(load_model(path).cost(q), continued.cost(q))
    all_zero = fit_xgboost_cost(q, np.zeros_like(q), [0., 0.], [1., 1.],
        structural_zero_mask=[True, True], structural_zero_evidence="synthetic_all_zero_proof", n_estimators=1)
    assert all_zero.booster is None
    path = tmp_path / "all_zero.npz"; save_model(all_zero, path)
    np.testing.assert_array_equal(load_model(path).cost(q), np.zeros_like(q))
    with pytest.raises(ValueError, match="evidence"):
        fit_xgboost_cost(q, target, [0., 2.], [1., 1.], structural_zero_mask=[True, False], n_estimators=1)
