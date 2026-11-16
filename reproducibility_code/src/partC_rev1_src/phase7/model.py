"""Phase 7 reference: smooth convex residual potential, native-unit derivatives.
Not a full campaign runner. Adapt to the live source loader without changing math.
No regional optimization, hidden fallback or independent cost-output network.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def positive(raw: torch.Tensor, divisor: int) -> torch.Tensor:
    return F.softplus(raw, beta=1.0, threshold=40.0) / divisor


class AnchoredICNN(nn.Module):
    """Two softplus hidden layers; input weights signed, hidden/output weights >=0.

    R(x)=f(x)-f(0)-grad f(0).x is convex, nonnegative, zero with zero gradient at 0.
    This does NOT guarantee coordinatewise positive gradients everywhere.
    """
    def __init__(self, m: int, width: int = 32, seed: int = 20260908):
        super().__init__()
        if m < 1 or width < 1:
            raise ValueError('Inactive regions must not construct an ICNN')
        self.m, self.width = m, width
        gen = torch.Generator(device='cpu').manual_seed(seed)
        self.A1 = nn.Parameter(torch.randn(width, m, generator=gen, dtype=torch.float64)/math.sqrt(m))
        self.A2 = nn.Parameter(torch.randn(width, m, generator=gen, dtype=torch.float64)/math.sqrt(m))
        self.b1 = nn.Parameter(torch.zeros(width, dtype=torch.float64))
        self.b2 = nn.Parameter(torch.zeros(width, dtype=torch.float64))
        # Effective hidden row sum/output sum initially approx 1; avoid huge softplus sums.
        inverse_softplus_one = math.log(math.expm1(1.0))
        self.raw_W = nn.Parameter(torch.full((width,width), inverse_softplus_one, dtype=torch.float64))
        self.raw_v = nn.Parameter(torch.full((width,), inverse_softplus_one, dtype=torch.float64))

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        h1=F.softplus(F.linear(x,self.A1,self.b1), threshold=40.0)
        h2=F.softplus(F.linear(h1,positive(self.raw_W,self.width))+
                      F.linear(x,self.A2,self.b2), threshold=40.0)
        return h2 @ positive(self.raw_v,self.width)

    def anchor(self, *, differentiable: bool) -> tuple[torch.Tensor,torch.Tensor]:
        with torch.enable_grad():
            zero=torch.zeros((1,self.m),dtype=self.A1.dtype,device=self.A1.device,requires_grad=True)
            f0=self.raw(zero)
            g0=torch.autograd.grad(f0.sum(),zero,create_graph=differentiable,retain_graph=differentiable)[0][0]
            if differentiable:
                return f0[0],g0
            return f0[0].detach(),g0.detach()

    def forward(self, x: torch.Tensor, *, differentiable_anchor: bool=True) -> torch.Tensor:
        if x.shape[-1]!=self.m: raise ValueError('Wrong active coordinate dimension')
        f0,g0=self.anchor(differentiable=differentiable_anchor)
        return self.raw(x)-f0-x@g0

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class RegionalPotential(nn.Module):
    """W_hat(u)=c0.u+Vscale*gain*R(u/s); grad gives native seconds."""
    def __init__(self, scale, c0, Vscale: float, gain: float, *, seed=20260908, width=32):
        super().__init__()
        s=torch.as_tensor(scale,dtype=torch.float64); b=torch.as_tensor(c0,dtype=torch.float64)
        if s.ndim!=1 or s.numel()==0 or b.shape!=s.shape or not torch.isfinite(s).all() or not torch.isfinite(b).all():
            raise ValueError('Invalid scales or inactive model')
        if torch.any(s<=0) or torch.any(b<0) or not np.isfinite([Vscale,gain]).all() or min(Vscale,gain)<=0:
            raise ValueError('Invalid scales')
        self.register_buffer('scale',s);self.register_buffer('c0',b)
        self.register_buffer('Vscale',torch.tensor(Vscale,dtype=torch.float64))
        self.register_buffer('gain',torch.tensor(gain,dtype=torch.float64))
        self.net=AnchoredICNN(len(s),width,seed)

    def forward(self,u,*,differentiable_anchor=True):
        return u@self.c0+self.Vscale*self.gain*self.net(u/self.scale,differentiable_anchor=differentiable_anchor)

    def value_cost(self,u,*,training=False):
        """Works under an outer no_grad context; input derivatives still require grad.
        Cost loss needs create_graph=True when training; never detach anchor there.
        """
        with torch.enable_grad():
            q=u.detach().clone().requires_grad_(True)
            v=self(q,differentiable_anchor=training)
            t=torch.autograd.grad(v.sum(),q,create_graph=training,retain_graph=training)[0]
        if training:return v,t
        return v.detach(),t.detach()


def interval_distance(pred, lower, upper):
    if torch.any(lower>upper):raise ValueError('Inverted certified interval')
    return torch.relu(lower-pred)+torch.relu(pred-upper)


def objective(model,u,tau,lower,upper,cost_scale,pair_weight,value_scale):
    """All loss denominators and gain are frozen from TRAIN A/B + LOW metadata only."""
    value,cost=model.value_cost(u,training=True)
    lg=(((cost-tau)/cost_scale)**2 * pair_weight).sum(dim=-1).mean()
    lv=(interval_distance(value,lower,upper)/value_scale).square().mean()
    lm=((torch.relu(model.c0-cost)/cost_scale).square()*pair_weight).sum(dim=-1).mean()
    return lg+0.1*lv+0.1*lm, {'cost':lg,'value_interval':lv,'freeflow_floor':lm}


def train_scales(U,TAU,upper,s,c0,Vscale,low):
    """Fit ONLY to independent TRAIN A/B. Returns reproducible numerical scales."""
    U=np.asarray(U,float); TAU=np.asarray(TAU,float);up=np.asarray(upper,float)
    s=np.asarray(s,float);c0=np.asarray(c0,float);low=np.asarray(low,float)
    if U.ndim!=2 or TAU.shape!=U.shape or U.shape[1]!=len(s) or U.shape[0]<1:raise ValueError('Bad training arrays')
    if not all(np.isfinite(x).all() for x in [U,TAU,up,s,c0,low]):raise ValueError('Nonfinite training data')
    if np.any(low<=0):raise ValueError('All active LOW coordinates must be positive')
    tscale=np.maximum(np.maximum(c0,np.median(abs(TAU),axis=0)),1.)
    dscale=np.maximum(np.quantile(abs(TAU-c0),.9,axis=0),.01*tscale)
    residual=np.maximum(up-U@c0,0.)
    vscale=max(float(np.quantile(residual,.9)),1.)
    gain=max(float(np.quantile(residual/Vscale,.9)),
             float(np.quantile(abs((TAU-c0)*s/Vscale),.9)),1e-8)
    pairweight=.5/len(s)+.5*low/low.sum()
    return {'cost_scale':dscale,'value_scale':vscale,'gain':gain,'pair_weight':pairweight,'total_cost_scale':tscale}


def separable_power4_fit(U, TAU, s, c0):
    """Training-only nonnegative least-squares coefficients, one per active coordinate.
    Normalizes X^4 columns before squaring to avoid a spurious overflow.
    """
    x=np.asarray(U,float)/s; y=np.asarray(TAU,float)-c0
    b=x**4;scale=np.maximum(np.max(abs(b),axis=0),1e-100);bn=b/scale
    denom=(bn*bn).sum(axis=0)
    coef=np.divide((bn*y).sum(axis=0),denom,out=np.zeros_like(denom),where=denom>0)/scale
    return np.maximum(coef,0.)


def separable_power4_eval(U,s,c0,coef):
    x=np.asarray(U,float)/s
    return U@c0+np.sum(coef*s*x**5/5,axis=-1), c0+coef*x**4


def export_arrays(model: RegionalPotential) -> dict[str,np.ndarray]:
    """Portable deterministic, root-free inference representation. No optimizer included."""
    def n(t):return t.detach().cpu().numpy().copy()
    f0,g0=model.net.anchor(differentiable=False)
    return {'scale':n(model.scale),'c0':n(model.c0),'Vscale':n(model.Vscale),
            'gain':n(model.gain),'A1':n(model.net.A1),'A2':n(model.net.A2),
            'b1':n(model.net.b1),'b2':n(model.net.b2),
            'W2':n(positive(model.net.raw_W,model.net.width)),
            'v':n(positive(model.net.raw_v,model.net.width)),
            'anchor_value':n(f0),'anchor_gradient':n(g0)}


def frozen_eval(a: dict[str,np.ndarray],u: np.ndarray,direction: np.ndarray|None=None):
    """Analytic value, gradient and optional Hessian-vector product. Never clips costs."""
    from scipy.special import expit
    u=np.asarray(u,dtype=np.float64)
    if u.shape[-1]!=len(a['scale']) or not np.isfinite(u).all() or np.any(u<0):
        raise ValueError('Invalid active demand input')
    x=u/a['scale'];l1=x@a['A1'].T+a['b1'];h1=np.where(l1>40,l1,np.logaddexp(0,l1))
    l2=h1@a['W2'].T+x@a['A2'].T+a['b2'];h2=np.where(l2>40,l2,np.logaddexp(0,l2))
    sig1=np.where(l1>40,1.,expit(l1));sig2=np.where(l2>40,1.,expit(l2));k2=sig2*a['v'];j1=k2@a['W2']
    grad=(sig1*j1)@a['A1']+k2@a['A2']
    f=h2@a['v']-a['anchor_value']-x@a['anchor_gradient']
    gain=float(a['Vscale']*a['gain'])
    value=u@a['c0']+gain*f
    cost=a['c0']+gain/a['scale']*(grad-a['anchor_gradient'])
    if direction is None:return value,cost
    d=np.asarray(direction,dtype=np.float64)/a['scale']
    dl1=d@a['A1'].T;dh1=sig1*dl1;dl2=dh1@a['W2'].T+d@a['A2'].T
    dk2=a['v']*sig2*(1-sig2)*dl2
    dg=(sig1*(1-sig1)*dl1*j1+sig1*(dk2@a['W2']))@a['A1']+dk2@a['A2']
    return value,cost,gain/a['scale']*dg
