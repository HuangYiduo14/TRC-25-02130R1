"""Auditable reference, not a full campaign runner.

Input dimension is the frozen active-pair count. Nonnegative diagonal power terms
plus a tangent-anchored ICNN are convex; costs are derivatives in native units.
No claim of globally componentwise-positive gradients for signed ICNN skips.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ConvexCore(nn.Module):
    def __init__(self, m: int, width: int = 32, seed: int = 20260908, head_power: int = 1):
        super().__init__()
        if m < 1 or width < 1 or head_power not in (1,5):
            raise ValueError('Inactive regions have no neural model')
        g = torch.Generator().manual_seed(seed)
        self.m, self.width, self.head_power = m, width, head_power
        self.A1 = nn.Parameter(torch.randn(width, m, generator=g, dtype=torch.float64)/math.sqrt(m))
        self.A2 = nn.Parameter(torch.randn(width, m, generator=g, dtype=torch.float64)/math.sqrt(m))
        self.b1 = nn.Parameter(torch.zeros(width, dtype=torch.float64))
        self.b2 = nn.Parameter(torch.zeros(width, dtype=torch.float64))
        raw = math.log(math.expm1(1.0))
        self.raw_W = nn.Parameter(torch.full((width,width),raw,dtype=torch.float64))
        self.raw_v = nn.Parameter(torch.full((width,),raw,dtype=torch.float64))

    def raw(self, x):
        w = F.softplus(self.raw_W, threshold=40.0)/self.width
        v = F.softplus(self.raw_v, threshold=40.0)/self.width
        h1 = F.softplus(F.linear(x,self.A1,self.b1),threshold=40.0)
        h2 = F.softplus(F.linear(h1,w)+F.linear(x,self.A2,self.b2),threshold=40.0)
        return (h2.pow(self.head_power)/self.head_power)@v

    def forward(self, x, differentiable_anchor=True):
        with torch.enable_grad():
            zero=torch.zeros((1,self.m),dtype=x.dtype,device=x.device,requires_grad=True)
            f0=self.raw(zero)
            g0=torch.autograd.grad(f0.sum(),zero,create_graph=differentiable_anchor,
                                   retain_graph=differentiable_anchor)[0][0]
            if not differentiable_anchor:
                f0,g0=f0.detach(),g0.detach()
            return self.raw(x)-f0[0]-x@g0


class PowerResidualPotential(nn.Module):
    def __init__(self, scale, c0, eta0, eta_unit=None, residual_unit=1.0,
                 width=32, seed=20260908, gate=1.0, head_power=1):
        super().__init__()
        s,b,e=[torch.as_tensor(x,dtype=torch.float64) for x in (scale,c0,eta0)]
        if s.ndim!=1 or len(s)==0 or s.shape!=b.shape or s.shape!=e.shape:
            raise ValueError('Invalid active vector dimensions')
        if not all(torch.isfinite(x).all() for x in (s,b,e)) or (s<=0).any() or (b<0).any() or (e<0).any():
            raise ValueError('Invalid finite scales or coefficients')
        unit=torch.as_tensor(eta_unit,dtype=torch.float64) if eta_unit is not None else torch.maximum(e,torch.maximum(.01*b,torch.ones_like(b)))
        if unit.shape!=s.shape or not torch.isfinite(unit).all() or (unit<=0).any() or not math.isfinite(residual_unit) or residual_unit<=0 or gate<0:
            raise ValueError('Invalid parameter scales')
        self.register_buffer('scale',s.clone());self.register_buffer('c0',b.clone())
        self.register_buffer('eta_unit',unit.clone())
        self.register_buffer('residual_unit',torch.tensor(residual_unit,dtype=torch.float64))
        self.coef=nn.Parameter((e/unit).clone())
        self.gate=nn.Parameter(torch.tensor(gate,dtype=torch.float64))
        self.net=ConvexCore(len(s),width,seed,head_power)

    def forward(self,u,differentiable_anchor=True):
        x=u/self.scale
        power=((self.coef*self.eta_unit)*self.scale*x.pow(5)/5).sum(-1)
        return u@self.c0+power+self.residual_unit*self.gate*self.net(x,differentiable_anchor)

    def value_cost(self,u,training=False):
        with torch.enable_grad():
            q=u.detach().clone().requires_grad_(True)
            v=self(q,differentiable_anchor=training)
            t=torch.autograd.grad(v.sum(),q,create_graph=training,retain_graph=training)[0]
        return (v,t) if training else (v.detach(),t.detach())

    @torch.no_grad()
    def project_parameters(self):
        # Parameter projection preserves the potential; this is NOT cost clipping.
        self.coef.clamp_(min=0);self.gate.clamp_(min=0)


def huber(e,delta=.1):
    a=e.abs()
    return torch.where(a<=delta,.5*e.square(),delta*(a-.5*delta))


def robust_loss(model,u,tau,lower,upper,pair_weight,row_weight=None):
    """Target-relative total/residual losses and native potential interval loss.
    Target quantities are labels/constants; no log transform of network inputs.
    """
    value,cost=model.value_cost(u,training=True)
    c0=model.c0
    total_scale=torch.maximum(torch.maximum(tau.abs(),c0),torch.ones_like(tau))
    residual_scale=torch.maximum((tau-c0).abs(),.05*total_scale)
    error=cost-tau
    cg=(huber(error/total_scale)*pair_weight).sum(-1)
    cr=(huber(error/residual_scale)*pair_weight).sum(-1)
    interval=torch.relu(lower-value)+torch.relu(value-upper)
    pv=torch.maximum(torch.maximum(upper.abs(),(u@c0).abs()),torch.ones_like(upper))
    cv=huber(interval/pv)
    cf=((torch.relu(c0-cost)/torch.maximum(c0,torch.ones_like(c0))).square()*pair_weight).sum(-1)
    if row_weight is None:row_weight=torch.ones_like(cv)
    rw=row_weight/row_weight.sum()
    pieces={k:(v*rw).sum() for k,v in [('total_cost',cg),('congestion',cr),('value_interval',cv),('floor',cf)]}
    loss=.75*pieces['total_cost']+.25*pieces['congestion']+.1*pieces['value_interval']+pieces['floor']
    return loss,pieces


def weighted_power4_fit(u,tau,s,c0):
    """New explicit baseline: closed-form weighted NNLS for relative total cost.
    Keep the OLD unweighted baseline separately for apples-to-apples reporting.
    """
    u,tau,s,c0=map(lambda x:np.asarray(x,dtype=np.float64),(u,tau,s,c0))
    if u.ndim!=2 or tau.shape!=u.shape or s.shape!=(u.shape[1],) or c0.shape!=s.shape:
        raise ValueError('Bad shapes')
    if not all(np.isfinite(x).all() for x in (u,tau,s,c0)) or (u<0).any() or (s<=0).any():raise ValueError('Bad native input')
    denom=np.maximum(np.maximum(abs(tau),c0),1.)
    b=(u/s)**4/denom;y=(tau-c0)/denom
    col=np.maximum(np.max(abs(b),axis=0),1e-100);bn=b/col
    eta=np.divide(np.sum(bn*y,axis=0),np.sum(bn*bn,axis=0),out=np.zeros(u.shape[1]),where=np.sum(bn*bn,axis=0)>0)/col
    return np.maximum(eta,0.)


def export_arrays(model):
    def n(x):return x.detach().cpu().numpy().copy()
    with torch.enable_grad():
        z=torch.zeros((1,model.net.m),dtype=model.scale.dtype,device=model.scale.device,requires_grad=True)
        f0=model.net.raw(z);g0=torch.autograd.grad(f0.sum(),z)[0][0]
    return {**{k:n(getattr(model,k)) for k in ['scale','c0','eta_unit','residual_unit','coef','gate']},
            **{k:n(getattr(model.net,k)) for k in ['A1','A2','b1','b2']},
            'W2':n(F.softplus(model.net.raw_W,threshold=40.)/model.net.width),
            'v':n(F.softplus(model.net.raw_v,threshold=40.)/model.net.width),
            'anchor_value':n(f0[0]),'anchor_gradient':n(g0),'head_power':np.array(model.net.head_power,dtype=np.int64)}


def portable_eval(a,u,direction=None):
    """Independent vectorized native value/cost/HVP, no Torch at evaluation time."""
    from scipy.special import expit
    u=np.asarray(u,dtype=np.float64)
    if u.shape[-1]!=len(a['scale']) or (u<0).any() or not np.isfinite(u).all():raise ValueError('Bad active input')
    x=u/a['scale'];l1=x@a['A1'].T+a['b1'];h1=np.where(l1>40,l1,np.logaddexp(0,l1))
    l2=h1@a['W2'].T+x@a['A2'].T+a['b2'];h2=np.where(l2>40,l2,np.logaddexp(0,l2))
    s1=np.where(l1>40,1.,expit(l1));s2=np.where(l2>40,1.,expit(l2));power=int(a.get('head_power',1));k=s2*a['v']*h2**(power-1);j=k@a['W2']
    grad=(s1*j)@a['A1']+k@a['A2'];res=(h2**power/power)@a['v']-a['anchor_value']-x@a['anchor_gradient']
    eta=a['coef']*a['eta_unit'];amp=float(a['residual_unit']*a['gate'])
    value=u@a['c0']+np.sum(eta*a['scale']*x**5/5,axis=-1)+amp*res
    cost=a['c0']+eta*x**4+amp/a['scale']*(grad-a['anchor_gradient'])
    if direction is None:return value,cost
    d=np.asarray(direction,dtype=np.float64)/a['scale'];dl1=d@a['A1'].T;dl2=(s1*dl1)@a['W2'].T+d@a['A2'].T
    dk=a['v']*(h2**(power-1)*s2*(1-s2)+(0. if power==1 else (power-1)*h2**(power-2)*s2**2))*dl2
    dg=(s1*(1-s1)*dl1*j+s1*(dk@a['W2']))@a['A1']+dk@a['A2']
    hvp=4*eta*x**3*d+amp/a['scale']*dg
    return value,cost,hvp


def calibrate_residual_unit(model, u, tau, upper, pair_weight, target=.001, batch=128):
    """TRAIN-only initialization. Computes unit residual directly to avoid cancellation
    against a huge power/free-flow base. Frozen Gamma bounds specified observed probes.
    """
    q=np.asarray(u,dtype=np.float64);t=np.asarray(tau,dtype=np.float64);up=np.asarray(upper,dtype=np.float64)
    if q.ndim!=2 or q.shape!=t.shape or q.shape[1]!=len(model.scale) or len(q)<1 or up.shape!=(len(q),):
        raise ValueError('Invalid calibration TRAIN arrays')
    weight=torch.as_tensor(pair_weight,dtype=torch.float64,device=model.scale.device)
    if weight.shape!=model.scale.shape or (weight<0).any() or not torch.isclose(weight.sum(),torch.tensor(1.,dtype=weight.dtype,device=weight.device)):
        raise ValueError('Invalid weights')
    maximum=0.
    for i in range(0,len(q),batch):
        uu=torch.as_tensor(q[i:i+batch],dtype=torch.float64,device=model.scale.device)
        yy=torch.as_tensor(t[i:i+batch],dtype=torch.float64,device=model.scale.device)
        hi=torch.as_tensor(up[i:i+batch],dtype=torch.float64,device=model.scale.device)
        with torch.enable_grad():
            x=(uu/model.scale).detach().requires_grad_(True)
            rr=model.net(x,differentiable_anchor=False)
            gg=torch.autograd.grad(rr.sum(),x)[0]/model.scale
        den=torch.maximum(torch.maximum(yy.abs(),model.c0),torch.ones_like(yy))
        cm=torch.sqrt(((gg/den)**2*weight).sum(-1)).max().item()
        vm=(rr.detach().abs()/torch.maximum(torch.maximum(hi.abs(),(uu@model.c0).abs()),torch.ones_like(hi))).max().item()
        maximum=max(maximum,cm,vm)
    if not math.isfinite(maximum) or maximum<=1e-30:
        raise ValueError('Numerically zero/invalid residual: explicitly reinitialize, no fabricated Gamma')
    gamma=target/maximum
    if not math.isfinite(gamma) or gamma<=0:raise ValueError('Nonfinite residual amplitude')
    with torch.no_grad():model.residual_unit.fill_(gamma);model.gate.fill_(1.)
    return {'unit_max':maximum,'Gamma':gamma,'target_normalized_perturbation':target,'TRAIN_rows':len(q)}
