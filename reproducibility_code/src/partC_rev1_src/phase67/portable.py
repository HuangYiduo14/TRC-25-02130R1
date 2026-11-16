"""NumPy/SciPy-only native potentials, derivatives and HVPs; no Torch/oracle import."""
import numpy as np
from scipy.special import expit
from partC_rev1_src.phase7.portable import frozen_eval as legacy_eval

def portable_eval(a,u,direction=None):
    u=np.asarray(u,dtype=np.float64)
    if u.shape[-1]!=len(a['scale']) or (u<0).any() or not np.isfinite(u).all():raise ValueError('Bad active input')
    x=u/a['scale'];l1=x@a['A1'].T+a['b1'];h1=np.where(l1>40,l1,np.logaddexp(0,l1))
    l2=h1@a['W2'].T+x@a['A2'].T+a['b2'];h2=np.where(l2>40,l2,np.logaddexp(0,l2))
    s1=np.where(l1>40,1.,expit(l1));s2=np.where(l2>40,1.,expit(l2));power=int(a.get('head_power',1))
    k=s2*a['v']*h2**(power-1);j=k@a['W2'];grad=(s1*j)@a['A1']+k@a['A2']
    residual=(h2**power/power)@a['v']-a['anchor_value']-x@a['anchor_gradient']
    eta=a['coef']*a['eta_unit'];amplitude=float(a['residual_unit']*a['gate'])
    value=u@a['c0']+np.sum(eta*a['scale']*x**5/5,axis=-1)+amplitude*residual
    cost=a['c0']+eta*x**4+amplitude/a['scale']*(grad-a['anchor_gradient'])
    if direction is None:return value,cost
    d=np.asarray(direction,dtype=np.float64)/a['scale'];dl1=d@a['A1'].T;dl2=(s1*dl1)@a['W2'].T+d@a['A2'].T
    dk=a['v']*(h2**(power-1)*s2*(1-s2)+(0. if power==1 else (power-1)*h2**(power-2)*s2**2))*dl2
    dg=(s1*(1-s1)*dl1*j+s1*(dk@a['W2']))@a['A1']+dk@a['A2']
    return value,cost,4*eta*x**3*d+amplitude/a['scale']*dg

def predict_arrays(kind,a,u,direction=None):
    u=np.asarray(u,dtype=np.float64)
    if u.shape[-1]!=len(a['c0']) or not np.isfinite(u).all() or (u<0).any():raise ValueError('Invalid ordered active input')
    if kind in {'OLD_ICNN_FROZEN','LEGACY_ICNN_CONTINUED'}:return legacy_eval(a,u,direction)
    if 'SKIP_ICNN' in kind:return portable_eval(a,u,direction)
    if kind=='FREEFLOW_LINEAR':
        result=(u@a['c0'],np.broadcast_to(a['c0'],u.shape))
        return result if direction is None else (*result,np.zeros_like(u))
    if kind in {'OLD_POWER4_NNLS','NEW_POWER4_RELATIVE_NNLS'}:
        x=u/a['scale'];result=(u@a['c0']+np.sum(a['eta']*a['scale']*x**5/5,axis=-1),a['c0']+a['eta']*x**4)
        return result if direction is None else (*result,4*a['eta']*x**3*np.asarray(direction)/a['scale'])
    raise ValueError('Unknown explicit model kind: '+kind)
