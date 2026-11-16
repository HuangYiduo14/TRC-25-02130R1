"""Weight-only attenuation for the actual two-layer, linear-head softplus ICNN.

This reference helper never changes a source model. It returns a conservative
fixed coefficient and native lower bounds. It does not perform model selection,
read test labels, or assert predictive accuracy. NumPy plus standard library only.
The analytic bound is global for the checked architecture; arithmetic margins are
engineering safeguards, not an interval-arithmetic proof of floating-point code.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np

@dataclass(frozen=True)
class Attenuation:
    rho: float
    margin_fraction: float
    raw_gradient_lower: np.ndarray
    native_negative_increment_bound: np.ndarray
    cost_lower_bound: np.ndarray
    source_amplitude: float
    status: str

def _arr(a: Mapping, key: str) -> np.ndarray:
    x=np.asarray(a[key],dtype=np.longdouble)
    if not np.isfinite(x).all():
        raise ValueError(f'{key}: nonfinite source')
    return x

def plan(a: Mapping, kind: str, margin: float=.01) -> Attenuation:
    """Return fixed rho, without mutating a. Only checked source families allowed."""
    if not 0 < margin < 1:
        raise ValueError('margin must be strictly between 0 and 1')
    p=float(np.asarray(a.get('head_power',1)))
    if p!=1:
        raise ValueError('Unsupported nonlinear head: derivative bound does not apply')
    A1,A2,W,v=(_arr(a,k) for k in ('A1','A2','W2','v'))
    s,c0,g0=(_arr(a,k) for k in ('scale','c0','anchor_gradient'))
    if A1.ndim!=2 or A2.ndim!=2 or W.ndim!=2 or v.ndim!=1:
        raise ValueError('Unexpected layer dimensions')
    m=len(c0)
    b1,b2=_arr(a,'b1'),_arr(a,'b2')
    _arr(a,'anchor_value')
    if b1.shape!=(A1.shape[0],) or b2.shape!=(len(v),):
        raise ValueError('Bias shape mismatch')
    if (s.shape!=(m,) or g0.shape!=(m,) or A1.shape[1]!=m or A2.shape!=(len(v),m)
        or W.shape!=(len(v),A1.shape[0])):
        raise ValueError('Native coordinate/layer shape mismatch')
    if np.any(s<=0) or np.any(c0<0) or np.any(W<0) or np.any(v<0):
        raise ValueError('Convexity, c0 or native-scale precondition violated')
    if 'SKIP_ICNN' in kind:
        for key in ('coef','eta_unit','residual_unit','gate'):
            if np.any(_arr(a,key)<0):
                raise ValueError(f'{key}: negative coefficient')
        amp=float(_arr(a,'residual_unit')*_arr(a,'gate'))
    elif kind in {'OLD_ICNN_FROZEN','LEGACY_ICNN_CONTINUED'}:
        if np.any(_arr(a,'Vscale')<0) or np.any(_arr(a,'gain')<0):
            raise ValueError('Negative legacy gain')
        amp=float(_arr(a,'Vscale')*_arr(a,'gain'))
    else:
        raise ValueError('Unsupported model family')
    if not np.isfinite(amp):
        raise ValueError('Nonfinite amplitude')
    # Conservative roundoff cushions, evaluated in extended precision where available.
    inner=A2+W@np.minimum(A1,0)
    scale_inner=np.abs(A2)+W@np.abs(A1)
    inner-=np.longdouble('1e-12')*(1+scale_inner)
    ell=v@np.minimum(inner,0)
    ell-=np.longdouble('1e-12')*(1+v@(1+np.abs(inner)))
    if amp==0:
        b=np.zeros(m,dtype=np.longdouble)
    else:
        b=np.longdouble(amp)/s*np.maximum(g0-ell,0)
        b*=1+np.longdouble('1e-12')
    positive=b>0
    rho_ld=np.longdouble(1)
    if positive.any():
        rho_ld=min(rho_ld,np.min((1-np.longdouble(margin))*c0[positive]/b[positive]))
    rho_ld=max(np.longdouble(0),rho_ld)
    rho=float(rho_ld)
    if 0 < rho < 1:
        rho=np.nextafter(rho,0.)
    lower=c0-np.longdouble(rho)*b
    if not np.isfinite(lower).all():
        raise ValueError('Nonfinite derived bound')
    return Attenuation(rho,margin,np.asarray(ell,dtype=float),np.asarray(b,dtype=float),
                       np.asarray(lower,dtype=float),amp,'UNCHANGED' if rho==1 else 'FIXED_ATTENUATION')

def transformed_copy(a: Mapping, kind: str, rho: float) -> dict:
    """Reference coefficient copy for parity tests; never writes source files."""
    if not np.isfinite(rho) or not 0<=rho<=1:
        raise ValueError('Invalid fixed attenuation')
    out={k:np.asarray(v).copy() for k,v in a.items()}
    key='gate' if 'SKIP_ICNN' in kind else 'gain'
    if key not in out:
        raise ValueError('Unknown coefficient mapping')
    out[key]=out[key]*rho
    return out
