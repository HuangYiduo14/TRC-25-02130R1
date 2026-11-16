"""Direct native cost WMAPE; MARE remains separately named."""
import numpy as np
import pandas as pd
from partC_rev1_src.phase67.portable import predict_arrays


def error_rows(kind,arr,a,rows,rid,cohort,**extra):
    frames=[]
    for begin in range(0,len(a['u']),128):
        stop=min(begin+128,len(a['u']));u=a['u'][begin:stop];truth=a['t'][begin:stop]
        if not np.isfinite(truth).all() or (truth<0).any():raise ValueError('Invalid native true marginal costs')
        value,pred=predict_arrays(kind,arr,u)
        if not np.isfinite(pred).all() or not np.isfinite(value).all():raise FloatingPointError('Nonfinite diagnostic prediction')
        diff=pred-truth;ae=abs(diff);relative=ae/np.maximum(abs(truth),1.)
        numerator=ae.sum(axis=1);denominator=truth.sum(axis=1)
        gap=np.maximum(a['L'][begin:stop]-a['eps'][begin:stop]-value,0.)+np.maximum(value-a['U'][begin:stop]-a['eps'][begin:stop],0.)
        w=a.get('low',np.ones(u.shape[1]));w=w/w.sum()
        frame=pd.DataFrame(dict(region_id=rid,cohort=cohort,input_hash=rows.input_hash.iloc[begin:stop].to_numpy(),
            n_coordinates=u.shape[1],MARE_equal_pairs=relative.mean(axis=1),MARE_LOW_pair_weighted=relative@w,
            MAE_seconds=ae.mean(axis=1),MSE_seconds2=(diff**2).mean(axis=1),bias_seconds=diff.mean(axis=1),
            maximum_relative_error=relative.max(axis=1),minimum_cost_seconds=pred.min(axis=1),
            negative_coordinates=(pred<0).sum(axis=1),negative_zero_flow_coordinates=((pred<0)&(u==0)).sum(axis=1),
            below_fftt_coordinates=(pred<arr['c0']-1e-8).sum(axis=1),
            interval_violation_relative=gap/np.maximum(abs(a['U'][begin:stop]),1.),total_query_flow=u.sum(axis=1),
            cost_abs_error_sum=numerator,true_cost_sum=denominator,
            cost_wmape_query_percent=np.divide(100*numerator,denominator,out=np.full(len(u),np.nan),where=denominator>0),
            cost_wmape_status=np.where(denominator>0,'DEFINED','UNDEFINED_ZERO_TRUE_COST_TOTAL')))
        for k,v in extra.items():frame[k]=v
        frames.append(frame)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def regional_aggregate(frame,keys):
    if frame.empty:raise ValueError('Empty cost evidence is not a result')
    summary=frame.groupby(keys,dropna=False).agg(cost_abs_error_sum=('cost_abs_error_sum','sum'),
        true_cost_sum=('true_cost_sum','sum'),n_records=('input_hash','size'),n_coordinates=('n_coordinates','sum'),
        MARE_equal_pairs=('MARE_equal_pairs','mean'),MARE_LOW_pair_weighted=('MARE_LOW_pair_weighted','mean'),
        negative_coordinates=('negative_coordinates','sum'),minimum_cost_seconds=('minimum_cost_seconds','min')).reset_index()
    summary['cost_wmape_percent']=np.divide(100*summary.cost_abs_error_sum,summary.true_cost_sum,
        out=np.full(len(summary),np.nan),where=summary.true_cost_sum>0)
    summary['cost_wmape_status']=np.where(summary.true_cost_sum>0,'DEFINED','UNDEFINED_ZERO_TRUE_COST_TOTAL')
    return summary
