"""Authenticated immutable regional portfolios and permanent whole-region fallback."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,read_npz,write_json,save_npz,sha,digest,array_hash
from partC_rev1_src.phase_n8.native import load_registry
from partC_rev1_src.phase_n8.objective import Composite,validate_arrays
from partC_rev1_src.phase_n8.oracle import aon,conservation
from .context import sources,reopen_suite
from .overlay import apply_overlay,closed_initialization
from .safe_objective import region_failures

SOURCE='N8_PROPOSED_NONCONSTANT_VALID_V1'
POLY='NEW_POWER4_RELATIVE_NNLS'
FREE='FREEFLOW_LINEAR'


def stripped(entry):
    return {k:v for k,v in entry.items() if k!='arrays'}


def portfolio_identity(rows):
    # Mathematical identity, independent of display label and epoch number.
    return digest([dict(region_id=r['region_id'],model_kind=r['model_kind'],portable_sha256=r['portable_sha256'],
                        pair_order_hash=r['pair_order_hash'],support_hash=r['support_hash'],start=r['start'],stop=r['stop'])
                   for r in sorted(rows,key=lambda r:r['start'])])


def load_candidates(ctx,src,suite):
    original,source_registry=load_registry(src,suite,SOURCE)
    inventory_path=src.r1/'tables/candidate_inventory_refined.csv'
    src.bind(inventory_path,'ALL_CURRENT_RELATIVE_NNLS_CANDIDATES')
    inventory=pd.read_csv(inventory_path);inventory=inventory.loc[inventory.model_kind.eq(POLY)].set_index('region_id')
    if len(inventory)!=1566 or inventory.index.duplicated().any():raise ValueError('Incomplete M2 inventory')
    candidates={};admission=[]
    for i,e in enumerate(original):
        if i%32==0:ctx.guard()
        manifest_path=Path(e['portable_path']).parent/'manifest.json'
        src.bind(manifest_path,'ORIGINAL_CANDIDATE_ADMISSION',e['source_manifest_sha256'])
        m=read_json(manifest_path);rid=e['region_id'];candidates[rid]={e['model_kind']:e}
        for kind in (POLY,FREE):
            binding=m['files'][kind];path=(manifest_path.parents[3]/binding['path']).resolve()
            if m['candidate_math'][kind]['status']!='PASS' or not m['metrics'][kind]['eligible']:
                raise ValueError('Current fallback source lacks mathematical/admission proof')
            if kind==POLY:
                iv=inventory.loc[rid]
                if Path(iv.portable_path).resolve()!=path or iv.portable_sha256!=binding['sha256']:
                    raise ValueError('M2 inventory and frozen candidate manifest disagree')
            a=src.arrays(path,'FROZEN_FALLBACK_'+kind,binding['sha256']);validate_arrays(kind,a)
            if not np.array_equal(a['c0'],e['arrays']['c0']):raise ValueError('Fallback c0 changed')
            candidates[rid][kind]=dict(e,model_kind=kind,portable_path=str(path),portable_sha256=binding['sha256'],arrays=a)
        metrics=m['metrics'][e['model_kind']]
        checks=[('TRAIN',metrics.get('train',{}))]+list(metrics.get('valid',{}).items())
        minima=[float(d['minimum_cost_seconds']) for _,d in checks if d.get('minimum_cost_seconds') is not None]
        if metrics.get('guard_minimum_cost') is not None:minima.append(float(metrics['guard_minimum_cost']))
        unsafe=any(not np.isfinite(v) or v<0 for v in minima) or any(d.get('nonfinite',False) for _,d in checks)
        exposure=float(suite['pairs'].LOW_CURRENT.iloc[e['start']:e['stop']].sum())
        admission.append(dict(region_id=rid,source_kind=e['model_kind'],source_sha256=e['portable_sha256'],
            source_minimum_observed_cost_seconds=min(minima) if minima and np.isfinite(minima).all() else None,
            saved_TRAIN_VALID_unsafe=unsafe,support_hash=e['support_hash'],pair_order_hash=e['pair_order_hash'],
            baseline_regional_traversal_flow=exposure,baseline_regional_traversal_share=exposure/float(suite['pairs'].LOW_CURRENT.sum())))
    return original,candidates,admission,source_registry


def freeze_portfolio(ctx,name,entries,*,epoch,events):
    rows=[stripped(e) for e in sorted(entries,key=lambda e:e['start'])]
    registry=dict(registry_id=name,identity=portfolio_identity(rows),entries=len(rows),rows=rows,
                  epoch=epoch,events=events,model_weights_changed=False,whole_region_potential=True,
                  composition=pd.Series([r['model_kind'] for r in rows]).value_counts().to_dict(),frozen_epoch=time.time())
    path=ctx.root/'registries'/(name+'.json')
    if path.exists():raise FileExistsError('Frozen portfolio is immutable: '+str(path))
    write_json(path,registry)
    return registry


def load_portfolio(ctx,src,suite,name):
    path=ctx.root/'registries'/(name+'.json');ctx.bind(path,'FIXED_EFFECTIVE_REGISTRY')
    registry=read_json(path);rows=registry['rows']
    if portfolio_identity(rows)!=registry['identity']:raise ValueError('Registry identity differs')
    groups=suite['pairs'].groupby('region_id',sort=False).indices;entries=[]
    if {r['region_id'] for r in rows}!=set(groups):raise ValueError('Incomplete effective regional coverage')
    for i,row in enumerate(rows):
        if i%32==0:ctx.guard()
        native=suite['pairs'].iloc[groups[row['region_id']]]
        if digest(native.pair_id.tolist())!=row['pair_order_hash'] or set(native.support_hash)!={row['support_hash']}:
            raise ValueError('Effective registry native order/support differs')
        if (row['start'],row['stop'])!=(int(groups[row['region_id']][0]),int(groups[row['region_id']][-1])+1):
            raise ValueError('Effective regional native offsets changed')
        a=src.arrays(row['portable_path'],'EFFECTIVE_FROZEN_MODEL',row['portable_sha256']);validate_arrays(row['model_kind'],a)
        if not np.array_equal(a['c0'],native.c0_seconds.to_numpy()):raise ValueError('Effective model c0 differs')
        entries.append(dict(row,arrays=a))
    return entries,registry


def preflight(ctx,args):
    if read_json(ctx.root/'reference/REFERENCE_FREEZE.json')['status']!='PASS':raise ValueError('Reference gate incomplete')
    src=sources(ctx);suite,od,balance,coverage=reopen_suite(ctx,src,'s03')
    original,candidates,admission,source_registry=load_candidates(ctx,src,suite)
    pd.DataFrame(admission).to_csv(ctx.root/'registries/source_TRAIN_VALID_admission.csv',index=False)
    poly=[candidates[e['region_id']][POLY] for e in original]
    free=[candidates[e['region_id']][FREE] for e in original]
    freeze_portfolio(ctx,'M3_FFTT',free,epoch=0,events=[])
    chains={e['region_id']:([e['model_kind'],POLY,FREE] if 'ICNN' in e['model_kind'] else [e['model_kind'],FREE]) for e in original}
    write_json(ctx.root/'registries/FALLBACK_CHAINS.json',dict(source_registry=source_registry,chains=chains,
        candidates={rid:{kind:stripped(e) for kind,e in options.items()} for rid,options in candidates.items()},
        distinct_analytic_next='FFTT',restart_epoch_cap=6,cumulative_recovery_cap_seconds=7200,
        final_rescue='ALL_REMAINING_NEURAL_TO_VERIFIED_CURRENT_POWER4',no_switch_back=True))
    fftt=Composite(suite,free,dict(identity=portfolio_identity(free)),guard=ctx.guard)
    _,cost,_,_=fftt.evaluate(np.zeros(len(suite['graph'].tail)))
    opened=aon(suite['graph'],od,cost,guard=ctx.guard)
    conservation(suite['graph'],opened['flow'],balance,od.total)
    save_npz(ctx.root/'closure/open_fftt_initial.npz',dict(flow=opened['flow'],serve=od.serve,balance=balance))
    mask_record=read_json(ctx.root/'closure/MASK_FREEZE.json')
    ctx.bind(ctx.root/'closure/availability.npz','FROZEN_REPRESENTED_I95_MASK',mask_record['mask_sha256'])
    mask=read_npz(ctx.root/'closure/availability.npz')['unavailable']
    closed_graph=apply_overlay(suite['graph'],mask)
    closed_od,closed_balance,closed_flow,ledger=closed_initialization(closed_graph,od,cost,ctx.guard)
    # Raw original-row multiplicities are retained in addition to aggregate rows.
    raw,_,_,_=src.original_od();additional=od.serve&~closed_od.serve
    ledger.update(additional_unreachable_original_rows=int(raw['original_row_multiplicity'][additional].sum()),
        closed_served_original_rows=int(raw['original_row_multiplicity'][closed_od.serve].sum()),
        original_rows=int(raw['original_row_multiplicity'].sum()),graph_identity=closed_graph.identity,
        demand_identity=closed_od.identity,base_demand_identity=od.identity)
    save_npz(ctx.root/'closure/closed_fftt_initial.npz',dict(flow=closed_flow,serve=closed_od.serve,balance=closed_balance))
    write_json(ctx.root/'closure/CLOSED_REACHABILITY.json',ledger)
    queries=[('ZERO',np.zeros_like(opened['flow'])),('BASELINE_CURRENT_REFERENCE',suite['fixed_x45']),
             ('OPEN_FEASIBLE_FFTT_AON',opened['flow']),('CLOSED_FEASIBLE_FFTT_AON',closed_flow)]
    for p in sorted((ctx.prior/'runs').glob('*/offending_query.npz')):
        ctx.bind(p,'KNOWN_ACTUAL_UNSAFE_QUERY');a=read_npz(p)['flow']
        if a.shape==opened['flow'].shape:queries.append((p.parent.name+'_KNOWN_WITNESS',a))
    failures=[]
    for row in admission:
        if row['saved_TRAIN_VALID_unsafe']:
            failures.append(dict(region_id=row['region_id'],model_kind=row['source_kind'],witness='SAVED_TRAIN_VALID',reason='NEGATIVE_OR_NONFINITE'))
    docket_path=src.r1/'tables/negative_stress_docket.csv';src.bind(docket_path,'KNOWN_HELDOUT_SAFETY_WITNESSES_NOT_ERROR_SELECTION')
    docket=pd.read_csv(docket_path)
    source_by={e['region_id']:e for e in original}
    for row in docket.itertuples(index=False):
        if row.model_kind==source_by[row.region_id]['model_kind'] and row.minimum_cost_seconds<0:
            failures.append(dict(region_id=row.region_id,model_kind=row.model_kind,witness=row.role,
                reason='KNOWN_SAVED_NEGATIVE',minimum_cost_seconds=float(row.minimum_cost_seconds)))
    for label,flow in queries:
        found=region_failures(original,flow[suite['packed_count']:suite['packed_count']+len(suite['pairs'])],guard=ctx.guard)
        failures.extend(dict(f,witness=label) for f in found)
        print('preflight query',label,'unsafe regions',len(found),flush=True)
    failing={f['region_id'] for f in failures};effective=[];events=[]
    exposures={a['region_id']:{k:a[k] for k in ('baseline_regional_traversal_flow','baseline_regional_traversal_share')} for a in admission}
    for e in original:
        rid=e['region_id'];chosen=e
        if rid in failing:
            next_kind=chains[rid][1];chosen=candidates[rid][next_kind]
            events.append(dict(region_id=rid,from_kind=e['model_kind'],to_kind=next_kind,
                               from_sha256=e['portable_sha256'],to_sha256=chosen['portable_sha256'],**exposures[rid]))
        effective.append(chosen)
    # Permanently exhaust unsafe evaluated polynomials before ever entering a live objective.
    for _ in range(3):
        remaining=[]
        for label,flow in queries:
            found=region_failures(effective,flow[suite['packed_count']:suite['packed_count']+len(suite['pairs'])],guard=ctx.guard)
            remaining.extend(dict(f,witness=label) for f in found)
        if not remaining:break
        failures.extend(remaining);bad={r['region_id'] for r in remaining};failing|=bad
        for i,e in enumerate(effective):
            if e['region_id'] not in bad:continue
            chain=chains[e['region_id']];at=chain.index(e['model_kind'])
            if at+1>=len(chain):raise ArithmeticError('Terminal FFTT candidate is unsafe')
            chosen=candidates[e['region_id']][chain[at+1]]
            events.append(dict(region_id=e['region_id'],from_kind=e['model_kind'],to_kind=chosen['model_kind'],
                from_sha256=e['portable_sha256'],to_sha256=chosen['portable_sha256'],**exposures[e['region_id']]))
            effective[i]=chosen
    else:raise ArithmeticError('Fallback screening did not terminate')
    poly_failures=[]
    for label,flow in queries:
        poly_failures.extend(dict(f,witness=label) for f in region_failures(poly,
            flow[suite['packed_count']:suite['packed_count']+len(suite['pairs'])],guard=ctx.guard))
    poly_bad={f['region_id'] for f in poly_failures}
    if poly_bad:
        poly=[candidates[e['region_id']][FREE] if e['region_id'] in poly_bad else e for e in poly]
    freeze_portfolio(ctx,'M2_POWER4',poly,epoch=0,events=poly_failures)
    registry=freeze_portfolio(ctx,'M1_EPOCH_000',effective,epoch=0,events=events)
    write_json(ctx.root/'registries/PREFLIGHT_FAILURES.json',failures)
    write_json(ctx.root/'registries/M1_CURRENT.json',dict(name=registry['registry_id'],identity=registry['identity'],epoch=0,
        discovery_epochs=0,cumulative_recovery_seconds=time.time()-args.job_start,preflight_started_epoch=args.job_start,
        permanently_downgraded_regions=sorted(failing),discarded_epochs=[],rescue_used=False))
    result=dict(status='PASS',original_composition=pd.Series([e['model_kind'] for e in original]).value_counts().to_dict(),
        effective_composition=registry['composition'],M1_identity=registry['identity'],whole_regions_downgraded=len(failing),
        screening_queries=[q[0] for q in queries],screening_includes_zero_flow_components=True,closed_coverage=ledger,
        open_initial_aon_seconds=opened['elapsed_seconds'],power4_regions=sum(e['model_kind']==POLY for e in poly),
        flat_power4_regions=sum(e['model_kind']==POLY and not np.any(e['arrays']['eta']>0) for e in poly),
        power4_fftt_exceptions=len(poly_bad),
        source_weights_unchanged=True,elapsed_seconds=time.time()-args.job_start)
    write_json(ctx.root/'registries/PREFLIGHT.json',result)
    return result
