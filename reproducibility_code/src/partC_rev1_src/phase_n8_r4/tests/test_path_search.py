import numpy as np
from partC_rev1_src.phase_n8.oracle import Network,Demand,aon
from partC_rev1_src.phase_n8_r4.path_search import query


def test_native_ties_zero_edges_and_blocked_centroid():
    g=Network([0,0,1,2,1,3],[1,2,2,1,3,2],[False,False,False,True],
        np.array(['z','a','b','c','d','e']),np.array(['n0','n1','n2','n3']))
    costs=np.array([0.,0.,0.,0.,2.,0.])
    for source,target in [(0,2),(0,3),(3,1)]:
        od=Demand([source],[0,1],[target],[1.],[True])
        original=aon(g,od,costs,return_distances=True)
        distance,path=query(g,costs,source,target)
        assert distance==original['distances'][0]
        np.testing.assert_array_equal(np.flatnonzero(original['flow']),np.sort(path))
