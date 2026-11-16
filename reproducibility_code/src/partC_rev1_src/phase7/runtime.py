"""Safe numeric frozen inference; unavailable models have no cost fallback."""
from pathlib import Path
import numpy as np
from .common import read_json,sha
from .portable import frozen_eval

class UnavailableRegionError(RuntimeError):pass
class FrozenRegion:
    def __init__(self,directory):
        self.directory=Path(directory)
        p=self.directory/'evaluation/manifest.json'
        if not p.exists():raise UnavailableRegionError('Region has no validated frozen ICNN: '+str(self.directory))
        meta=read_json(p)
        if meta['status']!='PASS' or meta['steps']<1:raise UnavailableRegionError('Untrained or invalid regional model')
        npz=self.directory/'evaluation/portable.npz'
        if sha(npz)!=meta['files']['portable.npz']['sha256']:raise UnavailableRegionError('Portable model hash mismatch')
        with np.load(npz,allow_pickle=False) as z:self.a={k:z[k] for k in z.files}
        with np.load(self.directory/'pack/scales.npz',allow_pickle=False) as z:self.lo=z['train_min'];self.hi=z['train_max']
    def query(self,u,direction=None):
        u=np.asarray(u,dtype=np.float64)
        values=frozen_eval(self.a,u,direction)
        return dict(potential=values[0],cost=values[1],hvp=values[2] if direction is not None else None,
            outside_observed_train_box=np.any((u<self.lo)|(u>self.hi),axis=-1),national_readiness=False)
