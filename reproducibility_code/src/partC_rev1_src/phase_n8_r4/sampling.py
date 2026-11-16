"""Frozen represented-demand PPS draw units; duplicates retained, no second weighting."""
import time
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,read_npz,write_json,sha,array_hash
from .context import sources


def freeze(ctx):
    safe_path=ctx.root/'SAFE_SCENARIO_FREEZE.json';safe=read_json(safe_path)
    if safe['status']!='PASS':raise ValueError('Safe registries must freeze before probability sample')
    src=sources(ctx);a,uids,reachable,meta=src.original_od()
    graph_root=ctx.prior/'graphs/s03'
    gm=read_json(graph_root/'IDENTITY_AND_COVERAGE_SELF_OD_V2.json')
    mp=graph_root/gm['mapping_file'];ctx.bind(mp,'EXACT_REPRESENTED_SAMPLING_POPULATION',gm['mapping_sha256'])
    serve=read_npz(mp)['classification']==2
    if not np.array_equal(serve,reachable):raise ValueError('Unexpected represented/physical membership difference')
    weights=np.where(serve,a['demand'],0.);total=float(weights.sum());cdf=np.cumsum(weights,dtype=np.float64)
    if total<=0 or not serve.any():raise ValueError('Empty represented sample population')
    rng=np.random.default_rng(20260915);draws=2048
    row=np.searchsorted(cdf,rng.random(draws)*cdf[-1],side='right')
    group=np.searchsorted(a['origin_offsets'][1:],row,side='right')
    o=a['origin_group_node_idx'][group];d=a['destination_node_idx'][row]
    unique,counts=np.unique(row,return_counts=True);multiplicity=counts[np.searchsorted(unique,row)]
    frame=pd.DataFrame(dict(draw_id=np.arange(1,draws+1),accepted_od_row=row,origin_idx=o,destination_idx=d,
        origin_uid=uids[o].astype(str),destination_uid=uids[d].astype(str),demand=a['demand'][row],
        draw_probability=a['demand'][row]/total,draw_multiplicity=multiplicity,
        original_row_multiplicity=a['original_row_multiplicity'][row]))
    frame['stable_od_key']=frame.origin_uid+'|'+frame.destination_uid
    out=ctx.root/'sampling';out.mkdir(exist_ok=False)
    frame.to_csv(out/'PPS_draws.csv',index=False,float_format='%.17g')
    result=dict(status='PASS',design='REPRESENTED_DEMAND_PPS_WITH_REPLACEMENT',draw_units=draws,unique_od_rows=len(unique),
        unique_origins=int(frame.origin_idx.nunique()),seed=20260915,population_aggregated_rows=int(serve.sum()),
        represented_demand=total,population_mask_hash=array_hash({'serve':serve}),
        design_weights='UNWEIGHTED_DRAW_UNITS; p=demand/represented_total; no additional demand multiplier',
        WMEANREL='mean(abs(pred-ref)/max(ref,1 second)) over all draws including duplicates',
        WREL='sum(abs(pred-ref))/sum(ref) over all draws including duplicates',
        bootstrap='resample draw units with replacement; paired predictions and reference',
        sample_sha256=sha(out/'PPS_draws.csv'),safe_freeze_sha256=sha(safe_path),
        frozen_epoch=time.time(),before_fresh_path_outcomes=True,existing_456_cohorts_unchanged=True)
    write_json(out/'DESIGN_FREEZE.json',result)
    return result
