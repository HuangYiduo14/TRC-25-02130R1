import numpy as np
import pandas as pd
import pytest
from partC_rev1_src.phase_n8.oracle import Network,Demand,aon
from partC_rev1_src.phase_n8_publication.overlay import apply_overlay,closed_initialization
from partC_rev1_src.phase_n8_publication.reference import project_backbone,select_keys
from partC_rev1_src.phase_n8_publication.safe_objective import region_failures


def test_overlay_feasible_rerouting_and_unreachable_row_retained():
    g=Network(np.array([0,0,2,1]),np.array([1,2,1,3]),np.zeros(4,bool),np.array(['a','b','c','d']),np.array(['0','1','2','3']))
    od=Demand(np.array([0]),np.array([0,2]),np.array([1,3]),np.array([7.,5.]),np.ones(2,bool))
    closed=apply_overlay(g,np.array([True,False,False,True]))
    new,balance,flow,ledger=closed_initialization(closed,od,np.ones(4))
    assert np.array_equal(flow,[0,7,7,0])
    assert np.array_equal(new.serve,[True,False]) and np.array_equal(new.demand,od.demand)
    assert ledger['additional_unreachable_demand']==5
    assert np.array_equal(g.order,[0,1,3,2])
    assert closed.identity!=g.identity
    assert np.array_equal(closed.edge_ids,g.edge_ids)


def test_series_min_of_bundle_sums_and_all_zero_member():
    links=pd.DataFrame(dict(link_id=['p','m','d'],role=['EXACT_PRIMITIVE','MACRO','PHYSICAL_MEMBER']))
    inter=pd.DataFrame(dict(final_link_id=['p','p','m','m','m','d'],original_owner_id=['p','p','x','x','y','z'],
        physical_arc_idx=np.arange(6),native_index=[0,0,1,1,1,2],role=['EXACT_PRIMITIVE']*2+['MACRO']*3+['PHYSICAL_MEMBER']))
    projected,spread=project_backbone(inter,links,np.array([2.,3.,4.,5.,7.,0.]))
    assert np.array_equal(projected,[5.,7.,0.])
    assert spread.series_spread.iloc[0]==2 and not spread.source_consistent.iloc[0]


def test_batch_detects_tiny_negative_at_zero_and_retains_below_fftt_positive():
    entries=[dict(region_id=r,model_kind='FREEFLOW_LINEAR',arrays={'c0':np.array([v])},start=i,stop=i+1,portable_sha256=r)
             for i,(r,v) in enumerate([('a',-1e-12),('b',-2.),('c',.1)])]
    failures=region_failures(entries,np.zeros(3))
    assert [x['region_id'] for x in failures]==['a','b']
    assert all(x['negative_at_zero_flow_coordinates']==[0] for x in failures)


def test_key_selection_is_reference_only_and_order_invariant():
    n=1200
    f=pd.DataFrame(dict(abstract_link_id=[f'e{i:04}' for i in range(n)],lambda_reference=np.arange(n,0,-1),
        longitude=np.full(n,-96.),latitude=np.full(n,39.),x=(np.arange(n)%15)*250000.,y=(np.arange(n)%7)*250000.))
    a,meta=select_keys(f)
    f['prediction_error']=np.arange(n)*999
    b,_=select_keys(f.iloc[::-1])
    assert a.abstract_link_id.tolist()==b.abstract_link_id.tolist()
    assert len(a)==200 and meta['candidate_fraction']==.2
