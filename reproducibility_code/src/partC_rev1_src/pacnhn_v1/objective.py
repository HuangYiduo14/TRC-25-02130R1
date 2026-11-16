"""One fixed regional registry per query; no fallback or model mutation."""
from __future__ import annotations
import numpy as np
from .solver import PolynomialObjective

class CompositeObjective:
    """Regional blocks replace their FFTT values in the common B/port objective.

    blocks are (edge_indices, model), cover each regional edge exactly once,
    and preserve the model's exact input/output order. XGBoost is cost only.
    """
    def __init__(self,base,k4,regional_indices,blocks, *,family):
        if family not in {'FFTT','POWER4','ICNN','XGBOOST'}:raise ValueError('Unknown regional family')
        self.family=family;self.has_potential=family!='XGBOOST'
        self.base=np.asarray(base,np.float64).copy();self.k4=np.asarray(k4,np.float64).copy()
        self.regional_indices=np.asarray(regional_indices,np.int64);self.blocks=[]
        if family=='FFTT':
            if blocks:raise ValueError('FFTT has no trained models')
        else:
            covered=[]
            for indices,model in blocks:
                indices=np.asarray(indices,np.int64)
                if not len(indices) or len(np.unique(indices))!=len(indices):raise ValueError('Invalid regional block')
                if bool(model.has_potential)!=self.has_potential:raise ValueError('Cost/potential family mismatch')
                covered.append(indices);self.blocks.append((indices,model))
            if not covered or not np.array_equal(np.sort(np.concatenate(covered)),np.sort(self.regional_indices)):
                raise ValueError('Every regional edge requires exactly one model; fallback prohibited')
            self.base[self.regional_indices]=0.;self.k4[self.regional_indices]=0.
        self.polynomial=PolynomialObjective(self.base,self.k4)

    def cost(self,x):
        cost=self.polynomial.cost(x)
        for indices,model in self.blocks:cost[indices]=model.cost(x[indices])
        if not np.isfinite(cost).all() or (cost<0).any():raise ValueError('Invalid actual composite cost; no clipping')
        return cost

    def value_cost(self,x):
        if not self.has_potential:raise TypeError('XGBoost predicts costs; no potential is defined')
        value,cost=self.polynomial.value_cost(x)
        for indices,model in self.blocks:
            v,c=model.value_cost(x[indices]);value+=float(v);cost[indices]=c
        if not np.isfinite(value) or not np.isfinite(cost).all() or (cost<0).any():raise ValueError('Invalid actual composite value/cost')
        return value,cost

    def hess_vec(self,x,direction):
        if not self.has_potential:raise TypeError('XGBoost has no potential Hessian')
        result=self.polynomial.hessian_diag(x)*direction
        for indices,model in self.blocks:result[indices]=model.hess_vec(x[indices],direction[indices])
        return result
