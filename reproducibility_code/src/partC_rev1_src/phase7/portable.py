"""Torch-free inference, extracted exactly from the validated portable kernel."""
from __future__ import annotations
import numpy as np


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
