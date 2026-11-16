import numpy as np
import pytest
from partC_rev1_src.pacnhn_v1.witnesses import audit_forest

def test_independent_witness_replay_detects_missing_pair_and_mask():
    args=[np.array([[-1,0,1,2]],np.int32),np.array([0]),np.array([0]),np.array([0,1]),
        np.array([3]),np.array([True]),np.array([0,1,2]),np.array([1,2,3]),
        np.ones(3,bool),np.array([[0,0,-1]],np.int32),np.array([2],np.uint64),np.array([0,1]),4]
    assert audit_forest(*args)[0]==1
    args[10]=np.array([3],np.uint64)
    with pytest.raises(ValueError,match='segment absent'):audit_forest(*args)
    args[10]=np.array([2],np.uint64);args[8][1]=False
    with pytest.raises(ValueError,match='Illegal OD'):audit_forest(*args)
