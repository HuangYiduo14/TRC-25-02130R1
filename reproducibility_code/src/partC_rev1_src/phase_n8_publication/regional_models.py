"""Evaluation-only all-region frozen standalone ICNN family; no selection by audit."""
from pathlib import Path
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,digest
from partC_rev1_src.phase_n8.objective import validate_arrays


def standalone_neural(ctx,src,suite):
    p=src.r1/'tables/candidate_inventory_refined.csv';src.bind(p,'FROZEN_STANDALONE_NEURAL_INVENTORY')
    frame=pd.read_csv(p);frame=frame.loc[frame.suite_id.eq('s03')&frame.model_kind.eq('POWER4_SKIP_ICNN32')]
    groups=suite['pairs'].groupby('region_id',sort=False).indices
    if len(frame)!=1566 or frame.region_id.duplicated().any() or set(frame.region_id)!=set(groups):
        raise ValueError('Standalone neural roster differs')
    result={}
    for i,row in enumerate(frame.itertuples(index=False)):
        if i%32==0:ctx.guard()
        manifest=Path(row.evidence);src.bind(manifest,'FROZEN_STANDALONE_NEURAL_MANIFEST');m=read_json(manifest)
        binding=m['files']['POWER4_SKIP_ICNN32'];path=Path(row.portable_path)
        if binding['sha256']!=row.portable_sha256 or (manifest.parents[3]/binding['path']).resolve()!=path.resolve():
            raise ValueError('Standalone neural file binding mismatch')
        native=suite['pairs'].iloc[groups[row.region_id]]
        if m['pair_order_hash']!=digest(native.pair_id.tolist()) or set(native.support_hash)!={m['support_hash']}:
            raise ValueError('Standalone neural native order/support mismatch')
        a=src.arrays(path,'FROZEN_STANDALONE_ICNN_DIAGNOSTIC',row.portable_sha256);validate_arrays(row.model_kind,a)
        if not np.array_equal(a['c0'],native.c0_seconds.to_numpy()):raise ValueError('Standalone neural c0 mismatch')
        result[row.region_id]=dict(region_id=row.region_id,model_kind=row.model_kind,arrays=a,
            portable_sha256=row.portable_sha256,portable_path=str(path))
    return result
