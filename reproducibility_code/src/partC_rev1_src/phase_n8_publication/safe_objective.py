"""Batch whole-region failures without ever changing a live scalar potential."""
from __future__ import annotations
import numpy as np
from partC_rev1_src.phase_n8.objective import Composite, UnsafeCost
from partC_rev1_src.phase_n8.common import array_hash
from partC_rev1_src.phase67.portable import predict_arrays


def region_failures(entries, regional_flow, direction=None, guard=None):
    failures = []
    for i, entry in enumerate(entries):
        if guard is not None and i % 32 == 0:
            guard()
        sl = slice(entry['start'], entry['stop'])
        u = regional_flow[sl]
        # HVP is a diagnostic of the same potential; its components need not be nonnegative.
        d = np.ones_like(u) if direction is None else direction[sl]
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                value, cost, hvp = predict_arrays(entry['model_kind'], entry['arrays'], u, d)
            finite = np.isfinite(value) and np.isfinite(cost).all() and np.isfinite(hvp).all()
            negative = np.flatnonzero(cost < 0)
            if finite and not len(negative):
                continue
            detail = dict(reason='NONFINITE' if not finite else 'NEGATIVE_COST',
                          minimum_cost_seconds=float(np.min(cost)) if np.isfinite(cost).all() else None,
                          negative_coordinates=negative.tolist(),
                          negative_at_zero_flow_coordinates=np.flatnonzero((cost < 0) & (u == 0)).tolist())
        except (FloatingPointError, OverflowError) as exc:
            detail = dict(reason='NONFINITE', error=str(exc))
        failures.append(dict(region_id=entry['region_id'], model_kind=entry['model_kind'],
                             portable_sha256=entry['portable_sha256'], **detail))
    return failures


class FixedSafeComposite(Composite):
    def evaluate(self, flow, direction=None, *, enforce_safety=True):
        mask=getattr(self.suite['graph'],'unavailable',None)
        if mask is not None and np.any(np.asarray(flow)[mask]!=0):
            raise ArithmeticError('Unavailable coordinate carries trial flow')
        checked_direction=np.ones_like(flow) if direction is None else direction
        try:
            result = super().evaluate(flow, checked_direction, enforce_safety=False)
        except UnsafeCost:
            failures = region_failures(self.entries, np.asarray(flow)[self.np:self.np+self.m],
                                      None if direction is None else np.asarray(direction)[self.np:self.np+self.m], self.guard)
            if not failures:
                raise ArithmeticError('Nonfinite packed cost/value is a nonregional scenario defect')
            raise UnsafeCost('STOP_NONFINITE', dict(failing_regions=failures, input_state_hash=array_hash({'flow': flow})), flow)
        value, cost, hvp, diag = result
        if (cost[:self.np] < 0).any():
            raise ArithmeticError('Negative packed cost cannot trigger a regional fallback')
        if enforce_safety and (cost < 0).any():
            failures = region_failures(self.entries, np.asarray(flow)[self.np:self.np+self.m], guard=self.guard)
            if not failures:
                raise ArithmeticError('Negative component outside mapped regional blocks')
            raise UnsafeCost('STOP_NEGATIVE_COST', dict(diag, failing_regions=failures,
                input_state_hash=array_hash({'flow': flow})), flow)
        return (value,cost,None if direction is None else hvp,diag)
