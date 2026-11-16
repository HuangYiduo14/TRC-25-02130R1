import numpy as np
from partC_rev1_src.pacnhn_v1.oracle import Network,Demand,aon,decode_keys,combine_traces,conservation

def test_parallel_zero_roles_trace_mixture_and_closed_witness(tmp_path):
    # O->a; two parallel a->b regional arcs; b->c B; c->d R; d->D.
    g=Network(np.array([0,1,1,2,3,4]),np.array([1,2,2,3,4,5]),
        np.array([True,False,False,False,False,True]),np.array(['o','r0','r1','b','r2','d']),np.array(list('OabcdD')))
    od=Demand(np.array([0]),np.array([0,1]),np.array([5]),np.array([7.]),np.array([True]))
    regions=np.array([[-1,0,0,-1,1,-1],[-1,0,0,-1,0,-1]],np.int32)
    costs=np.array([0.,1.,2.,3.,4.,0.]);r=aon(g,od,costs,edge_regions=regions,witness_root=tmp_path/'forest')
    np.testing.assert_array_equal(r['flow'],[7,7,0,7,7,7]);assert r['sptt']==56
    for trace in r['traces']:
        rid,src,dst=decode_keys(trace.keys,6)
        np.testing.assert_array_equal(src,[1,3]);np.testing.assert_array_equal(dst,[2,4]);np.testing.assert_array_equal(trace.values,[7,7])
    with np.load(tmp_path/'forest/forest_00000_00001.npz') as z:
        np.testing.assert_array_equal(z['predecessor'],[[-1,0,1,3,4,5]])
    mask=np.array([1,0,1,1,1,1],bool)
    closed=aon(g,od,costs,edge_regions=regions,enabled=mask)
    np.testing.assert_array_equal(closed['flow'],[7,0,7,7,7,7]);assert closed['sptt']==63
    mix=.3*r['flow']+.7*closed['flow'];conservation(g,mix,od.balance(6),od.total)
    tr=combine_traces(.7,r['traces'],closed['traces']);np.testing.assert_allclose(tr[0].values,[7,7])

def test_through_centroid_vertex_and_unreachable_no_renormalization():
    g=Network(np.array([0,1,2,1]),np.array([1,2,3,4]),np.array([1,0,0,1,1],bool),
        np.array(['o','road','d','dead']),np.array(['O','centroid_vertex','road_vertex','D','D2']))
    od=Demand(np.array([0]),np.array([0,2]),np.array([3,4]),np.array([9.,2.]),np.array([1,1],bool))
    r=aon(g,od,np.array([0.,1.,0.,0.]),enabled=np.array([1,1,1,0],bool))
    assert r['reachable_demand']==9.;assert od.total==11.;assert np.isinf(r['distances'][1])
