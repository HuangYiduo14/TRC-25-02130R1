"""Explicit typed frozen candidates; inactive/unavailable regions raise, never zero-fill."""
from pathlib import Path
import hashlib,json
import numpy as np
from .portable import predict_arrays

class UnavailableCandidateError(RuntimeError):pass
class FrozenCandidate:
    def __init__(self,region_directory,model_kind):
        root=Path(region_directory);p=root/'training/cycles/final/manifest.json'
        if not p.exists():raise UnavailableCandidateError('No final candidate for this active region')
        meta=json.loads(p.read_text());binding=meta['files'].get(model_kind)
        if not binding or not meta['metrics'][model_kind]['eligible']:raise UnavailableCandidateError('Requested candidate is absent or ineligible')
        audit=root/'final_evaluation/RESULT.json'
        if audit.exists() and model_kind==meta['champion_kind'] and json.loads(audit.read_text()).get('champion_invalid_on_hidden_audit'):
            raise UnavailableCandidateError('Frozen champion failed hidden audit; no automatic replacement')
        rel=Path(binding['path'])
        if rel.is_absolute() or '..' in rel.parts or ':' in str(rel):raise ValueError('Invalid candidate path')
        path=root/rel
        if hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']:raise UnavailableCandidateError('Candidate bytes changed')
        with np.load(path,allow_pickle=False) as z:self.a={k:z[k] for k in z.files}
        self.kind=model_kind;self.meta=meta
    def query(self,u,direction=None):
        result=predict_arrays(self.kind,self.a,u,direction)
        return dict(model_kind=self.kind,potential=result[0],cost=result[1],hvp=result[2] if direction is not None else None,
            model_hash=self.meta['model_hash'],support_hash=self.meta['support_hash'],pair_order_hash=self.meta['pair_order_hash'],national_readiness=False)
