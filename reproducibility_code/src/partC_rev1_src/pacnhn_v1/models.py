"""P-ACNHN model primitives; no data discovery, campaign loop or implicit run.

The neural model is a scalar potential.  Its two input maps, hidden map and
linear head are nonnegative, so its Hessian is PSD *and* entrywise nonnegative.
Subtracting the tangent at zero preserves this Hessian.  On nonnegative native
inputs the resulting costs are at least c0 and every cross-effect is nonnegative.
This deliberately excludes substitution.  XGBoost is a cost-only predictor and
must not be passed to a solver that requires a potential or its Hessian.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from scipy.special import expit
from torch import nn
from torch.nn import functional as F

FORMAT = "PACNHN_MODEL_V1"
ICNN_KIND = "MONOTONE_ICNN_POWER4"
POWER_KIND = "DIRECT_COST_POWER4"
XGB_KIND = "XGBOOST_LOG_COST"
ICNN_NATIVE = "NATIVE_PROJECTED_V1"
ICNN_SCALED = "TRAIN_BASIS_NORMALIZED_V2"
XGB_UNCENTERED = "UNCENTERED_LOG_V1"
XGB_CENTERED = "TRAIN_OUTPUT_LOG_CENTER_V2"


def log_center_hash(center):
    """Full ordered float64 output-vector identity, including structural zeros."""
    a = np.ascontiguousarray(center, dtype=np.float64)
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()


def _array(value, name):
    raw = np.asarray(value)
    if not np.isrealobj(raw):
        raise ValueError(f"{name}: complex values are not native real data")
    out = np.asarray(value, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError(f"{name}: nonfinite value")
    return out


def _vectors(c0, scale, eta=None):
    c0, scale = _array(c0, "c0"), _array(scale, "scale")
    if c0.ndim != 1 or not c0.size or scale.shape != c0.shape:
        raise ValueError("Nonempty c0 and scale vectors must have the same shape")
    if (c0 < 0).any() or (scale <= 0).any():
        raise ValueError("Require c0 >= 0 and scale > 0")
    eta = np.zeros_like(c0) if eta is None else _array(eta, "eta")
    if eta.shape != c0.shape or (eta < 0).any():
        raise ValueError("Require eta >= 0 with the native coordinate shape")
    return c0.copy(), scale.copy(), eta.copy()


def _inputs(u, m, name="u", *, nonnegative=True):
    q = _array(u, name)
    if q.ndim not in (1, 2) or q.shape[-1] != m or (q.ndim == 2 and not len(q)):
        raise ValueError(f"{name}: expected ({m},) or a nonempty (n,{m}) matrix")
    if nonnegative and (q < 0).any():
        raise ValueError(f"{name}: negative native demand")
    return q


def _finite(*values):
    if not all(np.isfinite(v).all() for v in values):
        raise FloatingPointError("Nonfinite model output; no clipping/fallback was applied")


def cost_wmape(predicted, observed):
    """Percent native-cost ratio of sums, not a mean of coordinate percentages."""
    p, y = _array(predicted, "predicted"), _array(observed, "observed")
    if p.shape != y.shape or y.size == 0 or (y < 0).any():
        raise ValueError("Equal nonempty shapes and nonnegative observed costs required")
    denominator = float(y.sum(dtype=np.float64))
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Cost WMAPE denominator is undefined")
    numerator = float(np.abs(p-y).sum(dtype=np.float64))
    if not math.isfinite(numerator):
        raise FloatingPointError("Nonfinite aggregate absolute cost error")
    return float(100.0 * (numerator / denominator))


class Power4Model:
    """Direct cost fit c0 + eta*(u/scale)^4, with its exact analytic integral."""
    kind = POWER_KIND
    has_potential = True

    def __init__(self, c0, scale, eta, *, metadata=None):
        self.c0, self.scale, self.eta = _vectors(c0, scale, eta)
        self.metadata = dict(metadata or {})
        for a in (self.c0, self.scale, self.eta):
            a.setflags(write=False)

    def value_cost(self, u):
        q = _inputs(u, len(self.c0))
        with np.errstate(over="ignore", invalid="ignore"):
            x = q/self.scale
            value = q @ self.c0 + (self.eta*self.scale*x**5/5).sum(axis=-1)
            cost = self.c0+self.eta*x**4
        _finite(value, cost)
        return value, cost

    def cost(self, u):
        return self.value_cost(u)[1]

    def hess_vec(self, u, v):
        q = _inputs(u, len(self.c0))
        d = _inputs(v, len(self.c0), "direction", nonnegative=False)
        if q.shape != d.shape:
            raise ValueError("Hessian direction must match the input shape")
        with np.errstate(over="ignore", invalid="ignore"):
            h = 4*self.eta*(q/self.scale)**3*d/self.scale
        _finite(h)
        return h

    def export_arrays(self):
        return {k: getattr(self, k).copy() for k in ("c0", "scale", "eta")}


def fit_power4_cost(u, tau, c0, scale, *, loss="l1", row_weight=None):
    """Fit native cost directly. L1 is the global per-coordinate WMAPE optimum.

    L1 uses the weighted median of (tau-c0)/x^4 with weights row_weight*x^4.
    The positive scaling of each column cancels, avoiding needless x^8 overflow.
    Zero-demand rows have parameter-independent error and remain in WMAPE audits.
    """
    c0, scale, _ = _vectors(c0, scale)
    q = _inputs(u, len(c0)); y = _inputs(tau, len(c0), "tau")
    if q.ndim != 2 or q.shape != y.shape:
        raise ValueError("Training inputs and costs must be equal nonempty matrices")
    if loss not in {"l1", "l2"}:
        raise ValueError("Explicit cost fitting loss must be l1 or l2")
    weight = np.ones(len(q)) if row_weight is None else _array(row_weight, "row_weight")
    if weight.shape != (len(q),) or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError("Invalid row weights")
    with np.errstate(over="ignore", invalid="ignore"):
        basis = (q/scale)**4
    _finite(basis)
    eta = np.zeros(len(c0), dtype=np.float64)
    # This is one small optimization per native output, not a numerical row loop.
    for j in range(len(c0)):
        positive = (basis[:, j] > 0) & (weight > 0)
        if not positive.any():
            continue
        b = basis[positive, j]; target = y[positive, j]-c0[j]
        column_scale = float(b.max()); b = b/column_scale
        w = weight[positive]
        if loss == "l2":
            coefficient = np.dot(w*b, target)/np.dot(w*b, b)
        else:
            ratios = target/b
            order = np.argsort(ratios, kind="stable")
            cumulative = np.cumsum((w*b)[order], dtype=np.float64)
            index = min(int(np.searchsorted(cumulative, cumulative[-1]/2, side="left")), len(order)-1)
            coefficient = ratios[order[index]]
        eta[j] = max(0.0, coefficient/column_scale)
    _finite(eta)
    return Power4Model(c0, scale, eta, metadata={"fit_loss": loss, "training_rows": len(q),
        "label_target": "native_cost", "zero_excitation_coordinates": int(np.sum(~np.any(basis > 0, axis=0)))})


def _raw_numpy(a, x, direction=None):
    l1 = x @ a["A1"].T+a["b1"]
    h1 = np.logaddexp(0.0, l1); s1 = expit(l1)
    l2 = h1 @ a["W2"].T+x @ a["A2"].T+a["b2"]
    h2 = np.logaddexp(0.0, l2); s2 = expit(l2)
    k = s2*a["v"]; j = k @ a["W2"]
    value = h2 @ a["v"]
    gradient = (s1*j) @ a["A1"]+k @ a["A2"]
    if direction is None:
        return value, gradient
    dl1 = direction @ a["A1"].T
    dl2 = (s1*dl1) @ a["W2"].T+direction @ a["A2"].T
    dk = a["v"]*s2*(1-s2)*dl2
    hvp = (s1*(1-s1)*dl1*j+s1*(dk @ a["W2"])) @ a["A1"]+dk @ a["A2"]
    return value, gradient, hvp


def _softplus_increment(base, increment):
    """softplus(base+d)-softplus(base), d>=0, without tiny-negative cancellation."""
    b, d = np.broadcast_arrays(base, increment)
    result = np.empty_like(d)
    positive = b >= 0
    result[positive] = d[positive]+np.log1p(expit(-b[positive])*np.expm1(-d[positive]))
    negative = (~positive) & (b+d <= 0)
    result[negative] = np.log1p(np.exp(b[negative]+d[negative])*(-np.expm1(-d[negative]))/(1+np.exp(b[negative])))
    crossing = (~positive) & (~negative)
    result[crossing] = np.logaddexp(0., b[crossing]+d[crossing])-np.logaddexp(0., b[crossing])
    return result


def _gradient_increment_numpy(a, x):
    """Exact algebra for grad f(x)-grad f(0), using only nonnegative increments.

    This is not cost clipping. Direct subtraction of nearly equal gradients can
    manufacture negative costs for zero-c0 coordinates, even at the exact origin.
    sigmoid(b+d)-sigmoid(b)=sigmoid(b+d)*sigmoid(-b)*(1-exp(-d)).
    """
    d1 = x @ a["A1"].T
    s10 = expit(a["b1"])
    ds1 = expit(a["b1"]+d1)*expit(-a["b1"])*(-np.expm1(-d1))
    dh1 = _softplus_increment(a["b1"], d1)
    base2 = np.logaddexp(0., a["b1"]) @ a["W2"].T+a["b2"]
    d2 = dh1 @ a["W2"].T+x @ a["A2"].T
    s2 = expit(base2+d2)
    dk = a["v"]*s2*expit(-base2)*(-np.expm1(-d2))
    j = (s2*a["v"]) @ a["W2"]
    return (ds1*j+s10*(dk @ a["W2"])) @ a["A1"]+dk @ a["A2"]


def _softplus_increment_torch(base, increment):
    b, d = torch.broadcast_tensors(base, increment)
    result = torch.empty_like(d)
    positive = b >= 0
    result[positive] = d[positive]+torch.log1p(torch.sigmoid(-b[positive])*torch.expm1(-d[positive]))
    negative = (~positive) & (b+d <= 0)
    result[negative] = torch.log1p(torch.exp(b[negative]+d[negative])*(-torch.expm1(-d[negative]))/(1+torch.exp(b[negative])))
    crossing = (~positive) & (~negative)
    zeros = torch.zeros((), dtype=b.dtype, device=b.device)
    result[crossing] = torch.logaddexp(zeros, b[crossing]+d[crossing])-torch.logaddexp(zeros, b[crossing])
    return result


def _small_bernoulli_bregman(s, t, d):
    # Sixth-order log-mgf series. For |d|<=1e-3 the omitted term is below
    # float64 working precision relative to the quadratic term. This evaluates
    # the same softplus Bregman remainder, without subtracting linear O(d) terms.
    return s*t*d*d*(.5+(1-2*s)*d/6+(1-6*s+6*s*s)*d*d/24
        +(1-14*s+36*s*s-24*s*s*s)*d**3/120
        +(1-30*s+150*s*s-240*s*s*s+120*s**4)*d**4/720)


def _bregman_numpy(base, increment):
    b,d=np.broadcast_arrays(base,increment);out=np.empty_like(d)
    small=d<=1e-3
    out[small]=_small_bernoulli_bregman(expit(b[small]),expit(-b[small]),d[small])
    positive=(~small)&(b>=0);t=expit(-b[positive])
    out[positive]=t*d[positive]+np.log1p(t*np.expm1(-d[positive]))
    negative=(~small)&(~positive)
    out[negative]=_softplus_increment(b[negative],d[negative])-expit(b[negative])*d[negative]
    return out


def _bregman_torch(base, increment):
    b,d=torch.broadcast_tensors(base,increment);out=torch.empty_like(d)
    small=d<=1e-3
    out[small]=_small_bernoulli_bregman(torch.sigmoid(b[small]),torch.sigmoid(-b[small]),d[small])
    positive=(~small)&(b>=0);t=torch.sigmoid(-b[positive])
    out[positive]=t*d[positive]+torch.log1p(t*torch.expm1(-d[positive]))
    negative=(~small)&(~positive)
    out[negative]=_softplus_increment_torch(b[negative],d[negative])-torch.sigmoid(b[negative])*d[negative]
    return out


def _residual_numpy(a,x):
    d1=x @ a['A1'].T;dh1=_softplus_increment(a['b1'],d1)
    base2=np.logaddexp(0.,a['b1']) @ a['W2'].T+a['b2']
    d2=dh1 @ a['W2'].T+x @ a['A2'].T
    r1=_bregman_numpy(a['b1'],d1);r2=_bregman_numpy(base2,d2)
    return (r2+expit(base2)*(r1 @ a['W2'].T)) @ a['v']


class PortableICNN(Power4Model):
    """Immutable, float64 NumPy evaluator including exact full coupled HVP."""
    kind = ICNN_KIND

    def __init__(self, arrays: Mapping, *, metadata=None):
        super().__init__(arrays["c0"], arrays["scale"], arrays["eta"], metadata=metadata)
        a = {k: _array(arrays[k], k).copy() for k in ("A1", "A2", "W2", "v", "b1", "b2", "amplitude")}
        m = len(self.c0)
        if a["A1"].ndim != 2:
            raise ValueError("Invalid first-layer dimensions")
        width = a["A1"].shape[0]
        shapes = {"A1": (width, m), "A2": (width, m), "W2": (width, width),
                  "v": (width,), "b1": (width,), "b2": (width,), "amplitude": ()}
        if not width or any(a[k].shape != shape for k, shape in shapes.items()):
            raise ValueError("Invalid neural layer dimensions")
        if any((a[k] < 0).any() for k in ("A1", "A2", "W2", "v", "amplitude")):
            raise ValueError("Every input, skip, hidden and head weight must be nonnegative")
        self.amplitude = float(a["amplitude"])
        self.width = width
        self.anchor_value, self.anchor_gradient = _raw_numpy(a, np.zeros(m, dtype=np.float64))
        _finite(self.anchor_value, self.anchor_gradient)
        for value in a.values():
            value.setflags(write=False)
        self.arrays = a

    def value_cost(self, u):
        q = _inputs(u, len(self.c0))
        value, cost = super().value_cost(q)
        with np.errstate(over="ignore", invalid="ignore"):
            x = q/self.scale
            value = value+self.amplitude*_residual_numpy(self.arrays,x)
            cost = cost+self.amplitude/self.scale*_gradient_increment_numpy(self.arrays, x)
            # Evaluate the exact zero-anchor identity explicitly, including batches.
            value = np.where(np.all(q == 0, axis=-1), 0.0, value)
        _finite(value, cost)
        return value, cost

    def unit_neural_cost(self, u):
        """Native cost increment at Gamma=1, without subtracting large costs."""
        q = _inputs(u, len(self.c0))
        result = _gradient_increment_numpy(self.arrays, q/self.scale)/self.scale
        _finite(result)
        if (result < 0).any():
            raise FloatingPointError("Negative analytic neural cost increment")
        return result

    def hess_vec(self, u, v):
        q = _inputs(u, len(self.c0)); d = _inputs(v, len(self.c0), "direction", nonnegative=False)
        if q.shape != d.shape:
            raise ValueError("Hessian direction must match the input shape")
        with np.errstate(over="ignore", invalid="ignore"):
            hvp = super().hess_vec(q, d)+self.amplitude/self.scale*_raw_numpy(self.arrays, q/self.scale, d/self.scale)[2]
        _finite(hvp)
        return hvp

    def export_arrays(self):
        return {**super().export_arrays(), **{k: v.copy() for k, v in self.arrays.items()}}


class MonotoneICNN(nn.Module):
    """Trainable float64 potential; effective neural weights use softplus.

    eta/amplitude use nonnegative projected parameters; call update_icnn or
    project_parameters after an external optimizer step. Negative coefficients
    are rejected at evaluation, never repaired by clipping predicted costs.
    """
    kind = ICNN_KIND
    has_potential = True

    def __init__(self, c0, scale, *, eta=None, width=32, seed=0, amplitude=1.0, device="cpu"):
        super().__init__()
        c0, scale, eta = _vectors(c0, scale, eta)
        if not isinstance(width, int) or width < 1 or not math.isfinite(amplitude) or amplitude < 0:
            raise ValueError("Positive width and finite nonnegative amplitude required")
        self.width, self.m, self.seed = width, len(c0), int(seed)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        dtype = torch.float64
        self.register_buffer("c0", torch.tensor(c0, dtype=dtype))
        self.register_buffer("scale", torch.tensor(scale, dtype=dtype))
        # Unit buffers default to one, preserving the native constructor's
        # function. Explicit TRAIN calibration changes optimizer coordinates,
        # never the mathematical model family or portable coefficient semantics.
        self.register_buffer("eta_units", torch.ones(len(c0), dtype=dtype))
        self.register_buffer("amplitude_unit", torch.ones((), dtype=dtype))
        self.eta_coordinate = nn.Parameter(torch.tensor(eta, dtype=dtype))
        self.amplitude_coordinate = nn.Parameter(torch.tensor(float(amplitude), dtype=dtype))
        self.coefficient_parameterization = ICNN_NATIVE
        self.calibration = None
        # Row sums begin O(1), including high-dimensional regions.
        for name, shape in (("raw_A1", (width, self.m)), ("raw_A2", (width, self.m)),
                            ("raw_W2", (width, width)), ("raw_v", (width,))):
            value = math.log(math.expm1(1.0))+.1*torch.randn(shape, generator=generator, dtype=dtype)
            setattr(self, name, nn.Parameter(value))
        self.b1 = nn.Parameter(torch.zeros(width, dtype=dtype))
        self.b2 = nn.Parameter(torch.zeros(width, dtype=dtype))
        self.to(device=device, dtype=dtype)

    @property
    def eta(self):
        return self.eta_units*self.eta_coordinate

    @property
    def amplitude(self):
        return self.amplitude_unit*self.amplitude_coordinate

    def _weights(self):
        return (F.softplus(self.raw_A1)/self.m, F.softplus(self.raw_A2)/self.m,
                F.softplus(self.raw_W2)/self.width, F.softplus(self.raw_v)/self.width)

    def _raw(self, x):
        a1, a2, w, v = self._weights()
        # logaddexp provides the same smooth stable softplus as the portable API.
        h1 = torch.logaddexp(torch.zeros((), dtype=x.dtype, device=x.device), F.linear(x, a1, self.b1))
        h2 = torch.logaddexp(torch.zeros((), dtype=x.dtype, device=x.device), F.linear(h1, w)+F.linear(x, a2, self.b2))
        return h2 @ v

    def _tensor(self, u, *, nonnegative=True):
        if isinstance(u, torch.Tensor):
            if u.is_complex():
                raise ValueError("Complex native inputs are invalid")
            q = u.to(dtype=torch.float64, device=self.scale.device)
        else:
            q = torch.as_tensor(_inputs(u, self.m, nonnegative=nonnegative), dtype=torch.float64, device=self.scale.device)
        if q.ndim not in (1, 2) or q.shape[-1] != self.m or (q.ndim == 2 and not len(q)):
            raise ValueError("Invalid native input dimensions")
        if not torch.isfinite(q).all() or (nonnegative and torch.any(q < 0)):
            raise ValueError("Invalid native input values")
        return q

    def forward(self, u):
        q = self._tensor(u)
        if not torch.isfinite(self.eta).all() or not torch.isfinite(self.amplitude) or torch.any(self.eta < 0) or self.amplitude < 0:
            raise ValueError("Unprojected/invalid nonnegative coefficients")
        with torch.enable_grad():
            x = q/self.scale
            a1,a2,w,v=self._weights();d1=F.linear(x,a1)
            dh1=_softplus_increment_torch(self.b1,d1)
            zeros=torch.zeros((),dtype=q.dtype,device=q.device)
            base2=F.linear(torch.logaddexp(zeros,self.b1),w,self.b2)
            d2=F.linear(dh1,w)+F.linear(x,a2)
            r1=_bregman_torch(self.b1,d1);r2=_bregman_torch(base2,d2)
            residual=(r2+torch.sigmoid(base2)*F.linear(r1,w)) @ v
            result = q @ self.c0+(self.eta*self.scale*x.pow(5)/5).sum(-1)+self.amplitude*residual
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite neural potential")
        return result

    def _cost_tensor(self, q):
        # Same exact gradient as autograd of forward, expressed without subtracting
        # nearly equal gradients. This also keeps zero-c0 TRAIN rows nonnegative.
        x = q/self.scale
        a1, a2, w, v = self._weights()
        d1 = F.linear(x, a1)
        s10 = torch.sigmoid(self.b1)
        ds1 = torch.sigmoid(self.b1+d1)*torch.sigmoid(-self.b1)*(-torch.expm1(-d1))
        dh1 = _softplus_increment_torch(self.b1, d1)
        zeros = torch.zeros((), dtype=x.dtype, device=x.device)
        base2 = F.linear(torch.logaddexp(zeros, self.b1), w, self.b2)
        d2 = F.linear(dh1, w)+F.linear(x, a2)
        s2 = torch.sigmoid(base2+d2)
        dk = v*s2*torch.sigmoid(-base2)*(-torch.expm1(-d2))
        j = (s2*v) @ w
        increment = (ds1*j+s10*(dk @ w)) @ a1+dk @ a2
        return self.c0+self.eta*x.pow(4)+self.amplitude/self.scale*increment

    def value_cost(self, u, *, create_graph=False, training=None):
        if training is not None:
            create_graph = bool(training)
        numpy_input = not isinstance(u, torch.Tensor)
        with torch.enable_grad():
            q = self._tensor(u)
            if not q.requires_grad:
                q = q.detach().requires_grad_(True)
            value = self(q)
            cost = self._cost_tensor(q)
        if not torch.isfinite(cost).all():
            raise FloatingPointError("Nonfinite neural cost")
        if numpy_input and not create_graph:
            return value.detach().cpu().numpy(), cost.detach().cpu().numpy()
        return (value, cost) if create_graph else (value.detach(), cost.detach())

    def cost(self, u, *, create_graph=False):
        # Cost-only training need not construct an unused anchored-value graph.
        numpy_input = not isinstance(u, torch.Tensor)
        if not torch.isfinite(self.eta).all() or not torch.isfinite(self.amplitude) or torch.any(self.eta < 0) or self.amplitude < 0:
            raise ValueError("Unprojected/invalid nonnegative coefficients")
        with torch.enable_grad():
            result = self._cost_tensor(self._tensor(u))
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite neural cost")
        if create_graph:
            return result
        return result.detach().cpu().numpy() if numpy_input else result.detach()

    def hess_vec(self, u, v):
        numpy_input = not isinstance(u, torch.Tensor)
        with torch.enable_grad():
            q = self._tensor(u).detach().requires_grad_(True)
            d = self._tensor(v, nonnegative=False)
            if d.shape != q.shape:
                raise ValueError("Hessian direction must match the input shape")
            _, c = self.value_cost(q, create_graph=True)
            hvp = torch.autograd.grad((c*d).sum(), q)[0]
        if not torch.isfinite(hvp).all():
            raise FloatingPointError("Nonfinite neural HVP")
        return hvp.detach().cpu().numpy() if numpy_input else hvp.detach()

    @torch.no_grad()
    def project_parameters(self):
        self.eta_coordinate.clamp_(min=0.0); self.amplitude_coordinate.clamp_(min=0.0)

    def export_arrays(self):
        with torch.no_grad():
            a1, a2, w, v = self._weights()
            pairs = {"c0": self.c0, "scale": self.scale, "eta": self.eta, "amplitude": self.amplitude,
                     "A1": a1, "A2": a2, "W2": w, "v": v, "b1": self.b1, "b2": self.b2}
            return {k: t.detach().cpu().numpy().copy() for k, t in pairs.items()}

    def export_model(self, *, metadata=None):
        return PortableICNN(self.export_arrays(), metadata={
            "coefficient_parameterization": self.coefficient_parameterization,
            "TRAIN_calibration": self.calibration, **dict(metadata or {})})

    @classmethod
    def from_portable(cls, model, *, device="cpu"):
        if not isinstance(model, PortableICNN):
            raise TypeError("Expected PortableICNN")
        a = model.export_arrays()
        result = cls(a["c0"], a["scale"], eta=a["eta"], width=model.width,
                     amplitude=float(a["amplitude"]), device=device)
        with torch.no_grad():
            for raw, name, factor in (("raw_A1", "A1", result.m), ("raw_A2", "A2", result.m),
                                      ("raw_W2", "W2", result.width), ("raw_v", "v", result.width)):
                value = a[name]*factor
                if (value <= 0).any():
                    raise ValueError("Exact training restoration needs strictly positive softplus weights")
                inverse = value+np.log(-np.expm1(-value))
                getattr(result, raw).copy_(torch.as_tensor(inverse, device=device))
            result.b1.copy_(torch.as_tensor(a["b1"], device=device))
            result.b2.copy_(torch.as_tensor(a["b2"], device=device))
        return result


@torch.no_grad()
def calibrate_icnn_train(model, u, tau, *, delta=.001, train_dataset_identity=None):
    """Freeze positive TRAIN-only optimizer units before creating an optimizer.

    eta=d*theta with d=max(eta_P4, mean(tau)/mean(x^4)); Gamma=G*gamma,
    G=sum(tau)/sum(unit-neural-cost), gamma initially delta. The initial
    nonnegative neural increment is exactly delta of total TRAIN true cost.
    Consequently hybrid WMAPE <= initial power4 WMAPE +100*delta percentage
    points. A zero coefficient remains permitted: no positive branch floor.
    """
    if not isinstance(model, MonotoneICNN) or not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("Explicit fresh ICNN and 0<delta<1 required")
    if model.coefficient_parameterization != ICNN_NATIVE:
        raise ValueError("Frozen coefficient units cannot be recalibrated")
    q = _inputs(u, model.m); y = _inputs(tau, model.m, "TRAIN tau")
    if q.ndim != 2 or q.shape != y.shape or not float(y.sum()) > 0:
        raise ValueError("Equal nonempty TRAIN matrices and positive cost sum required")
    portable = model.export_model()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        basis_mean = np.mean((q/portable.scale)**4, axis=0)
        mean_cost = np.mean(y, axis=0)
        proposed = np.divide(mean_cost, basis_mean, out=np.zeros_like(mean_cost), where=basis_mean > 0)
    _finite(basis_mean, mean_cost, proposed)
    units = np.maximum(portable.eta, proposed)
    fallback = units <= 0
    # This covers zero excitation AND excited structural-zero cost coordinates.
    # It is a coefficient unit only: no label or predicted cost is altered.
    positive_train_mean = float(y.mean())
    units[fallback] = positive_train_mean
    neural = portable.unit_neural_cost(q)
    total = float(y.sum(dtype=np.float64)); basis_total = float(neural.sum(dtype=np.float64))
    amplitude_unit = total/basis_total if basis_total > 0 else positive_train_mean*float(portable.scale.mean())
    _finite(units, amplitude_unit)
    if np.any(units <= 0) or amplitude_unit <= 0:
        raise ValueError("TRAIN-derived coefficient units must be finite and positive")
    initial_gamma = delta if basis_total > 0 else 0.
    model.eta_units.copy_(torch.as_tensor(units, device=model.scale.device))
    model.amplitude_unit.fill_(amplitude_unit)
    model.eta_coordinate.copy_(torch.as_tensor(portable.eta/units, device=model.scale.device))
    model.amplitude_coordinate.fill_(initial_gamma)
    model.coefficient_parameterization = ICNN_SCALED
    model.calibration = dict(policy=ICNN_SCALED, train_dataset_identity=train_dataset_identity,
        training_rows=len(q), native_cost_sum=total, unit_neural_cost_sum=basis_total,
        delta=delta, initial_gamma_coordinate=initial_gamma,
        initial_neural_cost_share=initial_gamma*amplitude_unit*basis_total/total,
        eta_units_min=float(units.min()), eta_units_max=float(units.max()),
        eta_positive_fallback_coordinates=np.flatnonzero(fallback).tolist(),
        eta_fallback_native_unit=positive_train_mean, amplitude_unit=amplitude_unit,
        zero_neural_excitation=basis_total == 0,
        units_frozen_across_TRAIN_growth=True, VALID_or_TEST_used=False)
    return dict(model.calibration)


def update_icnn(model, optimizer, u, tau, *, train_cost_mean, row_importance=None,
                gradient_clip=10.0):
    """One explicit update of native MAE / frozen full-TRAIN mean cost.

    Uniform rows/all coordinates need no importance weights. If rows are sampled
    with replacement at p_i, pass 1/(N*p_i), not weights renormalized in the batch.
    This estimates the fixed full-TRAIN WMAPE objective without changing its
    denominator or substituting coordinate-relative error. No data are selected.
    """
    if not math.isfinite(train_cost_mean) or train_cost_mean <= 0:
        raise ValueError("Positive frozen full-TRAIN mean native cost required")
    if not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("Positive finite gradient clip required")
    q, y = model._tensor(u), model._tensor(tau)
    if q.ndim != 2 or y.shape != q.shape:
        raise ValueError("Equal nonempty training matrices required")
    optimizer.zero_grad(set_to_none=True)
    prediction = model.cost(q, create_graph=True)
    error = (prediction-y).abs().mean(dim=1)
    if row_importance is not None:
        weight = torch.as_tensor(row_importance, dtype=torch.float64, device=q.device)
        if weight.shape != error.shape or not torch.isfinite(weight).all() or torch.any(weight < 0) or weight.sum() <= 0:
            raise ValueError("Invalid row importance factors")
        error = error*weight
    loss = error.mean()/train_cost_mean
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite direct cost loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
    optimizer.step(); model.project_parameters()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Optimizer produced nonfinite model parameters")
    return {"loss_wmape_fraction_estimate": float(loss.detach().cpu()),
            "gradient_norm_before_clip": float(norm.detach().cpu()), "batch_rows": len(q)}


class XGBoostCostModel:
    """Full-vector native cost prediction; deliberately has no potential/HVP API."""
    kind = XGB_KIND
    has_potential = False

    def __init__(self, booster, c0, scale, *, structural_zero_mask=None, log_center=None, metadata=None):
        self.c0, self.scale, _ = _vectors(c0, scale)
        self.booster = booster
        self.metadata = dict(metadata or {})
        zero = np.zeros(len(self.c0), dtype=bool) if structural_zero_mask is None else np.asarray(structural_zero_mask)
        if zero.shape != self.c0.shape or not np.isin(zero, [0, 1]).all():
            raise ValueError("Invalid structural-zero coordinate mask")
        self.structural_zero_mask = zero.astype(bool, copy=True)
        self.log_center = np.zeros(len(self.c0)) if log_center is None else _array(log_center, "log_center").copy()
        if self.log_center.shape != self.c0.shape or np.any(self.log_center[self.structural_zero_mask] != 0):
            raise ValueError("Invalid native-output log centers")
        if self.metadata.get('log_center_policy') == XGB_CENTERED:
            if log_center is None or self.metadata.get('log_center_sha256') != log_center_hash(self.log_center):
                raise ValueError('Centered model output-vector identity mismatch')
            if booster is not None:
                base_score=json.loads(booster.save_config())['learner']['learner_model_param']['base_score']
                if float(base_score) != 0.:
                    raise ValueError('Centered booster must retain base_score=0')
        if np.any(self.c0[self.structural_zero_mask] != 0):
            raise ValueError("An analytic structural zero must have zero free-flow cost")
        if self.structural_zero_mask.any() and not self.metadata.get("structural_zero_evidence"):
            raise ValueError("Analytic structural-zero evidence must be explicitly supplied")
        if (booster is None) != bool(self.structural_zero_mask.all()):
            raise ValueError("A booster is required exactly when positive-cost outputs exist")
        for a in (self.c0, self.scale, self.structural_zero_mask, self.log_center):
            a.setflags(write=False)

    def cost(self, u):
        q = _inputs(u, len(self.c0)); one = q.ndim == 1
        shape = np.atleast_2d(q).shape
        active = ~self.structural_zero_mask
        result = np.zeros(shape, dtype=np.float64)
        if active.any():
            prediction = np.asarray(self.booster.inplace_predict(np.atleast_2d(q/self.scale)), dtype=np.float64)
            prediction = prediction.reshape((shape[0], int(active.sum())))
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                result[:, active] = np.exp(self.log_center[active]+prediction)
        _finite(result)
        if (result[:, active] <= 0).any():
            raise FloatingPointError("Log-cost inverse underflow; no floor was silently applied")
        return result[0] if one else result


def fit_xgboost_cost(u, tau, c0, scale, *, seed=0, n_estimators=200, max_depth=3,
                     learning_rate=.05, device="cpu", n_jobs=1, monotone=False,
                     structural_zero_mask=None, structural_zero_evidence=None,
                     initial_model=None, deadline_epoch=None, guard=None,
                     log_center_policy=XGB_UNCENTERED, train_dataset_identity=None, **kwargs):
    """Explicit bounded fit of log(native cost), returning native exp predictions.

    Positive labels are required outside analytically certified zero coordinates;
    unexplained zeros are rejected instead of inventing an epsilon. TRAIN WMAPE
    remains a separate native-cost audit. XGBoost 2.1.x uses
    float32 tree internals even though this API accepts/returns native float64.
    """
    from xgboost import XGBRegressor
    c0, scale, _ = _vectors(c0, scale)
    q = _inputs(u, len(c0)); y = _inputs(tau, len(c0), "tau")
    if q.ndim != 2 or q.shape != y.shape:
        raise ValueError("Equal TRAIN matrices required")
    zero = np.zeros(len(c0), dtype=bool) if structural_zero_mask is None else np.asarray(structural_zero_mask)
    if zero.shape != c0.shape or not np.isin(zero, [0, 1]).all():
        raise ValueError("Invalid structural-zero coordinate mask")
    zero = zero.astype(bool)
    if zero.any() and (not structural_zero_evidence or np.any(c0[zero] != 0) or np.any(y[:, zero] != 0)):
        raise ValueError("Structural zeros require explicit analytic evidence and zero c0/labels")
    if np.any(y[:, ~zero] <= 0):
        raise ValueError("Strictly positive cost labels required outside verified structural zeros")
    if (not isinstance(n_estimators, int) or n_estimators < 1 or not isinstance(max_depth, int) or max_depth < 1
            or not isinstance(n_jobs, int) or n_jobs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Explicit positive tree/learning budgets required")
    if log_center_policy not in {XGB_UNCENTERED, XGB_CENTERED}:
        raise ValueError("Unknown explicit log-center policy")
    forbidden = {"objective", "multi_strategy", "monotone_constraints", "tree_method", "random_state", "base_score"} & kwargs.keys()
    if forbidden:
        raise ValueError(f"Frozen model semantics cannot be overridden: {sorted(forbidden)}")
    params = dict(objective="reg:squarederror", tree_method="hist", multi_strategy="one_output_per_tree",
        n_estimators=int(n_estimators), max_depth=int(max_depth), learning_rate=float(learning_rate),
        random_state=int(seed), device=device, n_jobs=int(n_jobs), **kwargs)
    if monotone:
        params["monotone_constraints"] = tuple(1 for _ in c0)
    if log_center_policy == XGB_CENTERED:
        params["base_score"] = 0.0
    if initial_model is not None:
        if (not isinstance(initial_model, XGBoostCostModel) or not np.array_equal(initial_model.c0, c0)
                or not np.array_equal(initial_model.scale, scale)
                or not np.array_equal(initial_model.structural_zero_mask, zero)
                or initial_model.metadata.get("all_input_cost_monotonicity") != bool(monotone)
                or initial_model.metadata.get("log_center_policy", XGB_UNCENTERED) != log_center_policy):
            raise ValueError("Continued XGBoost model has a different native/model contract")
    center = np.zeros(len(c0), dtype=np.float64)
    if initial_model is not None:
        center[:] = initial_model.log_center
        center_dataset = initial_model.metadata.get("log_center_train_dataset_identity")
    else:
        if log_center_policy == XGB_CENTERED and (~zero).any():
            center[~zero] = np.mean(np.log(y[:, ~zero]), axis=0)
        center_dataset = train_dataset_identity
    booster = None
    if (~zero).any():
        fit_params = dict(params)
        if deadline_epoch is not None or guard is not None:
            from xgboost.callback import TrainingCallback
            class _Budget(TrainingCallback):
                def before_iteration(self, model, epoch, evals_log):
                    if guard is not None:
                        guard()
                    return deadline_epoch is not None and time.time() >= deadline_epoch
                def after_iteration(self, model, epoch, evals_log):
                    if guard is not None:
                        guard()
                    return deadline_epoch is not None and time.time() >= deadline_epoch
            fit_params["callbacks"] = [_Budget()]
        regressor = XGBRegressor(**fit_params)
        target = np.log(y[:, ~zero])-center[~zero]
        regressor.fit(q/scale, target[:, 0] if target.shape[1] == 1 else target,
                      xgb_model=None if initial_model is None else initial_model.booster)
        booster = regressor.get_booster()
    return XGBoostCostModel(booster, c0, scale, structural_zero_mask=zero, log_center=center, metadata={
        "label_target": "log_native_cost_minus_frozen_TRAIN_output_center" if log_center_policy==XGB_CENTERED else "log_native_cost",
        "inverse_transform": "exp(frozen_output_center + tree_margin)" if log_center_policy==XGB_CENTERED else "exp", "training_rows": len(q),
        "all_input_cost_monotonicity": bool(monotone), "parameters": params,
        "structural_zero_evidence": structural_zero_evidence, "structural_zero_coordinates": np.flatnonzero(zero).tolist(),
        "boosting_rounds_total": 0 if booster is None else booster.num_boosted_rounds(),
        "continued_from_existing_model": initial_model is not None,
        "log_center_policy": log_center_policy, "log_center_train_dataset_identity": center_dataset,
        "log_center_sha256": log_center_hash(center), "log_center_bytes": int(center.nbytes),
        "log_center_frozen_across_TRAIN_growth": True,
        "native_api_dtype": "float64", "tree_internal_dtype": "float32", "has_potential": False})


def save_model(model, path, *, metadata=None, overwrite=False):
    """Save one self-contained, non-pickle NPZ; source files are never overwritten by default."""
    path = Path(path)
    if isinstance(model, MonotoneICNN):
        model = model.export_model()
    if not isinstance(model, (Power4Model, XGBoostCostModel)):
        raise TypeError("Unknown P-ACNHN model type")
    meta = {**model.metadata, **dict(metadata or {})}
    header = {"format": FORMAT, "kind": model.kind, "metadata": meta}
    header_text = json.dumps(header, sort_keys=True, allow_nan=False)
    arrays = model.export_arrays() if model.has_potential else {
        "c0": model.c0, "scale": model.scale,
        "structural_zero_mask": model.structural_zero_mask,
        "log_center": model.log_center,
        "booster_ubj": np.zeros(0, dtype=np.uint8) if model.booster is None else
            np.frombuffer(model.booster.save_raw(raw_format="ubj"), dtype=np.uint8).copy()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb" if overwrite else "xb") as handle:
        np.savez_compressed(handle, header=np.array(header_text), **arrays)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "kind": model.kind, "bytes": path.stat().st_size}


def load_model(path, *, device="cpu"):
    """Load portable inference only; never unpickle executable Python objects."""
    with np.load(Path(path), allow_pickle=False) as archive:
        header = json.loads(str(archive["header"].item()))
        if header.get("format") != FORMAT:
            raise ValueError("Unknown model serialization format")
        arrays = {k: archive[k].copy() for k in archive.files if k != "header"}
    kind, metadata = header["kind"], header["metadata"]
    if kind == ICNN_KIND:
        return PortableICNN(arrays, metadata=metadata)
    if kind == POWER_KIND:
        return Power4Model(arrays["c0"], arrays["scale"], arrays["eta"], metadata=metadata)
    if kind == XGB_KIND:
        from xgboost import Booster
        booster = None
        if arrays["booster_ubj"].size:
            booster = Booster(model_file=bytearray(arrays["booster_ubj"].tobytes()))
            booster.set_param({"device": device, "nthread": 1})
        return XGBoostCostModel(booster, arrays["c0"], arrays["scale"],
            structural_zero_mask=arrays["structural_zero_mask"], log_center=arrays.get("log_center"), metadata=metadata)
    raise ValueError(f"Unknown model kind: {kind}")
