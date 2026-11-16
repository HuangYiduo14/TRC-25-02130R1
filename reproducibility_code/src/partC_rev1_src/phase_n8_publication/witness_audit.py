"""Late, explicit replay of every recorded negative-cost trial under frozen models."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,read_npz,write_json,array_hash,sha
from .context import sources,reopen_suite
from .registry import SOURCE,load_portfolio
from .safe_objective import region_failures


def run(ctx,args):
    src=sources(ctx);suite,od,_,_=reopen_suite(ctx,src,'s03')
    from partC_rev1_src.phase_n8.native import load_registry
    original,source=load_registry(src,suite,SOURCE)
    current=read_json(ctx.root/'registries/M1_CURRENT.json')
    final,registry=load_portfolio(ctx,src,suite,current['name'])
    folders=[ctx.prior/'runs'/n for n in ('P03_S03_PRIMARY','P08_S03_BEST_ANY','P09_S03_NONCONSTANT_VALID')]
    folders += [ctx.r4/'runs'/n for n in ('B05_ORIGINAL_PRIMARY_WARM','B06_ORIGINAL_BEST_ANY_WARM','B07_ORIGINAL_NONCONSTANT_WARM')]
    folders += sorted((ctx.root/'runs/B02_OPEN_FALLBACK').glob('epoch_00[0-5]'))
    if len(folders)!=12:raise ValueError('Expected twelve actual recorded witnesses')
    records=[];unique={};out=ctx.root/'registries/all_recorded_witnesses';out.mkdir(exist_ok=False)
    for folder in folders:
        query=folder/'offending_query.npz';stop_path=folder/'SAFETY_STOP.json';result_path=folder/'RESULT.json'
        ctx.bind(stop_path,'RECORDED_UNSAFE_QUERY_METADATA');ctx.bind(result_path,'RECORDED_UNSAFE_RUN_IDENTITY')
        stop=read_json(stop_path);result=read_json(result_path);identity=result['identity']
        for key,expected in [('graph',suite['graph'].identity),('support',suite['support_identity']),('demand',od.identity)]:
            if identity[key]!=expected:raise ValueError('Unsafe witness '+key+' mismatch: '+str(folder))
        file_hash=ctx.bind(query,'ACTUAL_OFFENDING_TRIAL_NOT_ACCEPTED_STATE',stop['offending_query_sha256'])
        flow=read_npz(query)['flow'];state=array_hash({'flow':flow})
        if state!=stop['input_state_hash'] or flow.shape!=(len(suite['graph'].tail),):raise ValueError('Unsafe query state/order mismatch')
        if not np.isfinite(flow).all() or np.any(flow<0):raise ValueError('Recorded query is not finite nonnegative native flow')
        freeze=folder/'PRE_RUN_FREEZE.json'
        if freeze.exists():ctx.bind(freeze,'RECORDED_WITNESS_PRE_RUN_FREEZE')
        key=(identity['graph'],identity['support'],state)
        if key not in unique:unique[key]=flow
        records.append(dict(source_run=str(folder),query_sha256=file_hash,input_state_hash=state,
            graph_identity=identity['graph'],support_identity=identity['support'],demand_identity=identity['demand'],
            source_registry_identity=identity['registry'],metadata_sha256=sha(stop_path),result_sha256=sha(result_path),
            audit_timing='SUPPLEMENTAL_AFTER_OPEN_DISCOVERY_AND_CLOSED_ASSIGNMENTS',
            original_preflight_included=folder.parent==ctx.prior/'runs'))
    failures=[];summaries=[]
    for (_,_,state),flow in unique.items():
        native=flow[suite['packed_count']:suite['packed_count']+len(suite['pairs'])]
        for label,entries,identity in [('ORIGINAL_NONCONSTANT',original,source['identity']),('FINAL_EFFECTIVE',final,registry['identity'])]:
            found=region_failures(entries,native,guard=ctx.guard)
            failures.extend(dict(f,input_state_hash=state,portfolio=label,registry_identity=identity) for f in found)
            summaries.append(dict(input_state_hash=state,portfolio=label,registry_identity=identity,
                regions_screened=len(entries),native_coordinates=len(native),unsafe_regions=len(found),
                zero_flow_coordinates_included=True,value_cost_HVP_checked=True))
            print('RECORDED_WITNESS',state[:12],label,'unsafe_regions',len(found),flush=True)
    pd.DataFrame(records).to_csv(out/'WITNESS_PROVENANCE.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'SCREEN_SUMMARY.csv',index=False)
    write_json(out/'FAILURES.json',failures)
    final_failures=[f for f in failures if f['portfolio']=='FINAL_EFFECTIVE']
    result=dict(status='PASS' if not final_failures else 'FAIL',recorded_witnesses=len(records),unique_query_vectors=len(unique),
        original_registry_identity=source['identity'],final_registry_identity=registry['identity'],
        final_unsafe_regions=len({f['region_id'] for f in final_failures}),final_registry_unchanged=True,
        late_supplement=True,omission='Initial preflight screened the three cold records but omitted the three warm records; this explicit replay repairs audit coverage without backdating screening.',
        model_selection_or_training_performed=False,elapsed_seconds=time.time()-args.job_start)
    write_json(out/'RESULT.json',result)
    if final_failures:raise ArithmeticError('Final portfolio failed recorded-witness audit')
    return result
