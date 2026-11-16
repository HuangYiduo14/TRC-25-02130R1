"""Complete forward backbone projection, reference-only keys and typed I95 mask."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer
from partC_rev1_src.phase_n8.common import read_json, write_json, save_npz, array_hash, digest, sha
from .context import sources, reopen_suite

B_ROLES = ('EXACT_PRIMITIVE', 'MACRO', 'PHYSICAL_MEMBER')
EXPECTED_B = dict(s02=87460, s03=87582, s04=87637, s05=87606, s06=88048)


def provenance(src, sid, suite, physical_graph):
    root = src.p5/'suites'/sid
    ledger = src.frame(root/'physical_link_ownership.parquet', sid+'_FORWARD_OWNERSHIP')
    ledger = ledger.sort_values('physical_arc_idx', kind='stable').reset_index(drop=True)
    if not np.array_equal(ledger.physical_arc_idx, np.arange(len(physical_graph.tail))):
        raise ValueError('Physical ownership index is incomplete or duplicated')
    if not np.array_equal(ledger.physical_arc_id.astype(str), physical_graph.arc_idx_to_directed_arc_id.astype(str)):
        raise ValueError('Physical source stable IDs disagree')
    if not ledger.owner_count.eq(1).all():
        raise ValueError('Nonunique physical ownership')
    links = suite['links']
    backbone = links.role.isin(B_ROLES)
    inter = ledger.loc[ledger.ownership_role.eq('INTER_B')].copy()
    if set(inter.final_link_id) != set(links.loc[backbone, 'link_id']):
        raise ValueError('Typed ownership does not cover exactly complete backbone')
    if int(backbone.sum()) != EXPECTED_B[sid]:
        raise ValueError('Backbone registry count changed')
    inter['native_index'] = pd.Index(links.link_id).get_indexer(inter.final_link_id)
    inter['role'] = links.role.to_numpy()[inter.native_index]
    direct = inter.role.eq('PHYSICAL_MEMBER')
    ix = inter.loc[direct, 'native_index'].to_numpy()
    if not np.array_equal(links.member_arc_idx.to_numpy()[ix], inter.loc[direct, 'physical_arc_idx']):
        raise ValueError('Direct-member indices disagree')
    if not np.array_equal(links.member_arc_id.to_numpy()[ix].astype(str), inter.loc[direct, 'physical_arc_id'].astype(str)):
        raise ValueError('Direct-member IDs disagree')
    seq = src.frame(root/'macro_sequences.parquet', sid+'_TYPED_SERIES_ORDER')
    actual = inter.loc[inter.role.eq('MACRO'), ['final_link_id', 'original_owner_id', 'sequence_position']].drop_duplicates()
    expected = seq[['corridor_id', 'primitive_id', 'sequence_position']].drop_duplicates()
    if set(map(tuple, actual.to_numpy())) != set(map(tuple, expected.to_numpy())):
        raise ValueError('Macro source sequence differs from complete ownership')
    count_path = root/'counts.json'
    src.bind(count_path, sid+'_INCIDENCE_CERTIFICATE')
    inc = read_json(count_path)['incidence']
    if inc['status'] != 'PASS' or not inc['exactly_one_owner'] or any(inc[k] != 0 for k in
            ('dangling_surviving_incidence', 'removed_od_nodes', 'replacement_endpoint_conflicts')):
        raise ValueError('Accepted contraction incidence proof failed')
    return ledger, inter, np.flatnonzero(backbone)


def project_backbone(inter, links, physical_flow):
    """ARC=self; PARALLEL=sum; SERIES=min(bundle sums), exactly compiler CURRENT."""
    flow = np.asarray(physical_flow, float)
    if not np.isfinite(flow).all() or (flow < 0).any():
        raise ValueError('Invalid physical CURRENT vector')
    work = inter[['final_link_id','original_owner_id','physical_arc_idx','native_index','role']].copy()
    work['physical_flow'] = flow[work.physical_arc_idx.to_numpy(np.int64)]
    out = np.full(len(links), np.nan)
    direct = work.role.eq('PHYSICAL_MEMBER')
    if work.loc[direct, 'native_index'].duplicated().any():
        raise ValueError('Direct coordinate has multiple source arcs')
    out[work.loc[direct, 'native_index']] = work.loc[direct, 'physical_flow']
    bundles = work.loc[~direct].groupby(['native_index','original_owner_id'], sort=True).physical_flow.sum().reset_index()
    throughput = bundles.groupby('native_index').physical_flow.agg(['min','max','count'])
    out[throughput.index] = throughput['min']
    primitive = links.role.eq('EXACT_PRIMITIVE').to_numpy()
    if not (throughput.loc[throughput.index.isin(np.flatnonzero(primitive)), 'count'] == 1).all():
        raise ValueError('Exact primitive has multiple sequential source bundles')
    macro = throughput.loc[throughput.index.isin(np.flatnonzero(links.role.eq('MACRO')))].copy()
    macro['series_spread'] = macro['max'] - macro['min']
    macro['source_consistency_allowance'] = 1e-8 + 1e-10*np.maximum(1., np.maximum(abs(macro['max']), abs(macro['min'])))
    macro['source_consistent'] = macro.series_spread <= macro.source_consistency_allowance
    macro['abstract_link_id'] = links.link_id.to_numpy()[macro.index]
    b = links.role.isin(B_ROLES).to_numpy()
    if not np.isfinite(out[b]).all() or (out[b] < 0).any():
        raise ValueError('Forward projection lacks backbone coordinates')
    return out, macro.reset_index()


def locations(suite, index):
    links = suite['links'].iloc[index]
    nodes = suite['nodes'].set_index('node_id')
    result = pd.DataFrame(dict(abstract_link_id=links.link_id.to_numpy(),native_index=index,role=links.role.to_numpy()))
    transform = Transformer.from_crs('EPSG:4269','EPSG:5070',always_xy=True)
    for prefix, key in [('tail','tail_node_id'),('head','head_node_id')]:
        result[prefix+'_longitude'] = links[key].map(nodes.longitude).to_numpy()
        result[prefix+'_latitude'] = links[key].map(nodes.latitude).to_numpy()
        result[prefix+'_x'], result[prefix+'_y'] = transform.transform(result[prefix+'_longitude'].to_numpy(),result[prefix+'_latitude'].to_numpy())
    for name in ('longitude','latitude','x','y'):
        result[name] = (result['tail_'+name]+result['head_'+name])/2
    if not np.isfinite(result[['longitude','latitude','x','y']].to_numpy()).all():
        raise ValueError('Incomplete abstract representative geometry')
    result['geometry_source'] = 'NATIVE_ABSTRACT_NODE_REPRESENTATIVE_ENDPOINTS_CHORD'
    return result


def select_keys(frame, target=200):
    if frame.abstract_link_id.duplicated().any():
        raise ValueError('Duplicate abstract selection IDs')
    eligible = frame.lambda_reference.gt(0) & frame.longitude.between(-125,-66) & frame.latitude.between(24,50)
    pool = frame.loc[eligible].sort_values(['lambda_reference','abstract_link_id'],ascending=[False,True],kind='stable').copy()
    if len(pool) < target:
        raise ValueError('Insufficient positive geographically located reference links')
    fraction = next(f for f in (.2,.25,.5,1.) if int(np.ceil(len(pool)*f)) >= target)
    candidates = pool.head(int(np.ceil(len(pool)*fraction))).copy()
    candidates['cell_id'] = np.floor(candidates.x/250000).astype(int).astype(str)+':'+np.floor(candidates.y/250000).astype(int).astype(str)
    cells = {key:g for key,g in candidates.groupby('cell_id',sort=False)}
    order = sorted(cells,key=lambda k:(-float(cells[k].lambda_reference.iloc[0]),k))
    selected = []
    for rank in range(len(candidates)):
        for cell in order:
            if rank < len(cells[cell]):
                selected.append(cells[cell].iloc[[rank]])
                if len(selected) == target:break
        if len(selected) == target:break
    keys = pd.concat(selected,ignore_index=True)
    keys['sample_rank'] = np.arange(1,target+1)
    return keys, dict(target=target, candidate_fraction=fraction, candidate_count=len(candidates),
        positive_CONUS_count=len(pool), geographic_or_zero_exclusions=int((~eligible).sum()),
        projection='EPSG:4269_TO_EPSG:5070',grid_metres=250000,
        candidate_minimum_reference_flow=float(candidates.lambda_reference.min()),
        selected_reference_flow_share=float(keys.lambda_reference.sum()/frame.lambda_reference.sum()),
        occupied_candidate_cells=len(cells),occupied_selected_cells=int(keys.cell_id.nunique()),
        ordering='CELL_BEST_REFERENCE_DESC_THEN_CELL_ID; WITHIN_CELL_REFERENCE_DESC_THEN_LINK_ID; ROUND_ROBIN',
        random_seed='NONE_DETERMINISTIC',prediction_errors_used=False)


def closure_mask(ctx, suite, ledger, inter):
    source = ctx.r4/'geography/I95_intended_closed_physical_ids.csv'
    ctx.bind(source,'EXACT_I95_ROUTE_SOURCE_METADATA')
    intended = pd.read_csv(source,dtype={'physical_arc_id':str,'route_number':str,'partC_rev1_faf_stfips':str})
    if not (intended.class_code.eq(11)&intended.route_type.eq('I')&intended.route_number.eq('95')&
            intended.sign_route.eq('I 95')&intended.partC_rev1_faf_stfips.isin(['51','37','45','13','12'])).all():
        raise ValueError('I95 route/state exact-match source failed')
    ids = set(intended.physical_arc_id)
    if len(ids) != len(intended) or not ids.issubset(set(ledger.physical_arc_id.astype(str))):
        raise ValueError('I95 physical identities do not join exactly')
    work = inter.copy(); work['selected'] = work.physical_arc_id.astype(str).isin(ids)
    bundles = work.groupby(['final_link_id','original_owner_id','role'],sort=True).selected.agg(['all','any','size','sum']).reset_index()
    links=suite['links'];disabled=np.zeros(len(suite['graph'].tail),bool)
    direct=work.role.eq('PHYSICAL_MEMBER')
    disabled[work.loc[direct&work.selected,'native_index']]=True
    typed=bundles.loc[~bundles.role.eq('PHYSICAL_MEMBER')].groupby('final_link_id')['all'].any()
    disabled[pd.Index(links.link_id).get_indexer(typed.index[typed])]=True
    selected=work.loc[work.selected].copy()
    unresolved=selected.loc[~disabled[selected.native_index.to_numpy()]]
    if len(unresolved):
        raise ValueError('Selected arcs alter unblocked coefficients; explicit narrower scenario required')
    matched=selected[['physical_arc_id','physical_arc_idx','final_link_id','original_owner_id','role']].copy()
    closed_physical=np.zeros(len(ledger),bool);closed_physical[matched.physical_arc_idx]=True
    # Recompute truth values from exactly D_match, independently of the intended list.
    work['matched']=closed_physical[work.physical_arc_idx]
    replay_bundles=work.groupby(['native_index','original_owner_id','role']).matched.all().reset_index()
    replay=np.zeros_like(disabled)
    rb=replay_bundles.loc[~replay_bundles.role.eq('PHYSICAL_MEMBER')].groupby('native_index').matched.any()
    replay[rb.index[rb]]=True
    replay[work.loc[direct&work.matched,'native_index']]=True
    if not np.array_equal(replay,disabled):raise ValueError('Physical and logical mask truth tables disagree')
    affected=ledger.loc[ledger.physical_arc_id.astype(str).isin(ids)].copy()
    omitted=affected.loc[~affected.ownership_role.eq('INTER_B')].copy()
    if len(intended)!=1713 or len(matched)!=1105 or int(disabled.sum())!=869 or len(omitted)!=608:
        raise ValueError('Audited I95 scope changed; investigate before execution')
    dst=ctx.root/'closure'
    intended.to_csv(dst/'intended_I95_route_metadata.csv',index=False,float_format='%.17g')
    matched.to_csv(dst/'matched_physical_arcs.csv',index=False)
    omitted.to_csv(dst/'omitted_suppressed_interior_arcs.csv',index=False)
    bundles.loc[bundles['any']].to_csv(dst/'typed_bundle_truth_table.csv',index=False)
    save_npz(dst/'availability.npz',dict(unavailable=disabled,physical_unavailable=closed_physical,
                                      abstract_link_ids=suite['graph'].edge_ids))
    mask_rows=locations(suite,np.flatnonzero(disabled[:len(links)]));mask_rows.to_csv(dst/'unavailable_abstract_links.csv',index=False,float_format='%.17g')
    result=dict(status='EXACT_REPRESENTED_MASK_EQUIVALENCE_PASS',scenario='REPRESENTED_I95_ABSTRACT_BACKBONE_CLOSURE',
        intended_physical_arcs=len(intended),matched_physical_arcs=len(matched),omitted_suppressed_B_arcs=len(omitted),
        unavailable_logical_links=int(disabled.sum()),unavailable_by_role=mask_rows.role.value_counts().to_dict(),
        omitted_suppressed_B_nodes=int(omitted.original_owner_id.nunique()),regional_source_arcs_closed=0,
        partially_selected_unblocked_coordinates=0,unblocked_coefficients_changed=0,
        healthy_physical_members_inside_blocked_macros_retained=int((~work.selected&disabled[work.native_index]).sum()),
        mask_sha256=sha(dst/'availability.npz'),base_graph_identity=suite['graph'].identity,
        closed_assignment_executed=False, physical_matched_control_eligible=True,
        interpretation='608 suppressed B-interior arcs are omitted; this is not a full physical I95 closure')
    write_json(dst/'MASK_FREEZE.json',result)
    return result


def run(ctx,args):
    from partC_rev1_src.phase_n8_r4.path_metrics import bindings
    src=sources(ctx); physical=bindings(ctx); records=[]
    for tier in ('low_benchmark','high_benchmark'):
        src.bind(physical.sources[tier]['physical_arrays'],'ACCEPTED_PHYSICAL_REFERENCE_'+tier)
    for sid in ctx.c['active_suites']:
        ctx.guard();suite,od,balance,coverage=reopen_suite(ctx,src,sid)
        ledger,inter,bi=provenance(src,sid,suite,physical.graph)
        projected,spread=project_backbone(inter,suite['links'],physical.low_physical.current_flow)
        ref=suite['fixed_x45'][bi];difference=projected[bi]-ref
        if not np.allclose(projected[bi],ref,rtol=1e-10,atol=1e-8):
            raise ValueError('Compiler forward reference disagreement: '+sid)
        if not spread.source_consistent.all():raise ValueError('Baseline series common-throughput source inconsistent')
        packed_reference=src.arrays(src.p5/'suites'/sid/'reference_low_current.npz',sid+'_COMPILER_REFERENCE_ARRAY')
        if not np.array_equal(packed_reference['flow'],suite['links'].low_current_flow.to_numpy()):
            raise ValueError('Complete compiler reference array differs')
        lm=src.frame(src.p5/'suites'/sid/'reference_mapping_ledger.parquet',sid+'_COMPILER_REFERENCE_LEDGER')
        lm=lm.loc[lm.source_id.eq('physical_mu030_low_rg1e3')&lm.state_role.eq('CURRENT')]
        if lm.link_id.duplicated().any() or set(lm.link_id)!=set(suite['links'].link_id):
            raise ValueError('Complete compiler reference ledger differs')
        if not np.array_equal(lm.set_index('link_id').loc[suite['links'].link_id,'flow'].to_numpy(),packed_reference['flow']):
            raise ValueError('Ledger CURRENT flow disagrees with saved compiler reference')
        trace_path=Path(str(physical.sources['low_benchmark']['suite_trace_template']).format(suite=sid))
        trace_sha=src.bind(trace_path,sid+'_SAVED_CURRENT_REGIONAL_PATH_MIXTURE')
        pairs=suite['pairs']
        if set(pairs.source_trace_identity)!={trace_sha}:
            raise ValueError('Regional reference source role/hash mismatch')
        if sid=='s03' and (set(pairs.source_flow_role)!={'CURRENT_X45'} or set(pairs.source_benchmark)!={'physical_mu030_low_rg1e3'}):
            raise ValueError('s03 explicitly recorded regional source role mismatch')
        with np.load(trace_path,allow_pickle=False) as z:
            keys=z['current__coarse_regional_pair_flow__keys'];values=z['current__coarse_regional_pair_flow__values']
        encoded=(pairs.source_terminal_index.to_numpy(np.uint64)<<np.uint64(32))|pairs.target_terminal_index.to_numpy(np.uint64)
        if not np.array_equal(encoded,pairs.coarse_key.to_numpy(np.uint64)) or not np.all(keys[1:]>keys[:-1]):
            raise ValueError('Regional trace pair encoding/order invalid')
        at=np.searchsorted(keys,encoded)
        if np.any(at>=len(keys)) or not np.array_equal(keys[at],encoded) or not np.array_equal(values[at],pairs.LOW_CURRENT.to_numpy()):
            raise ValueError('Regional CURRENT path-mixture mapping differs')
        dst=ctx.root/'reference'/sid;dst.mkdir(exist_ok=False)
        inter.to_parquet(dst/'backbone_source_incidence.parquet',index=False)
        spread.to_csv(dst/'baseline_series_projection_audit.csv',index=False,float_format='%.17g')
        mapping_id=digest(dict(ownership=sha(src.p5/'suites'/sid/'physical_link_ownership.parquet'),
            sequence=sha(src.p5/'suites'/sid/'macro_sequences.parquet'),projection='ARC_SELF_PARALLEL_SUM_SERIES_MIN_CURRENT'))
        frame=locations(suite,bi);frame.insert(0,'suite_id',sid);frame['lambda_reference']=ref
        frame['reference_id']=array_hash({'flow':physical.low_physical.current_flow})
        frame['graph_identity']=suite['graph'].identity;frame['mapping_identity']=mapping_id
        frame['units']='ANNUAL_KTON';frame.to_csv(dst/'backbone_reference.csv',index=False,float_format='%.17g')
        keys,keymeta=select_keys(frame);keys.to_csv(dst/'key_links_200.csv',index=False,float_format='%.17g')
        write_json(dst/'KEY_SELECTION_FREEZE.json',dict(**keymeta,epoch=time.time(),sha256=sha(dst/'key_links_200.csv')))
        save_npz(dst/'native_reference.npz',dict(flow=suite['fixed_x45'],edge_ids=suite['graph'].edge_ids,
                backbone_indices=bi,regional_pair_ids=suite['pairs'].pair_id.to_numpy(dtype=str)))
        # Reference is a benchmark projection; measure but do not repair its node imbalance.
        f=suite['fixed_x45'];g=suite['graph']
        net=np.bincount(g.tail,weights=f,minlength=len(g.node_ids))-np.bincount(g.head,weights=f,minlength=len(g.node_ids))
        rec=dict(suite_id=sid,status='FULL_FORWARD_REFERENCE_PASS',backbone_count=len(bi),key_count=len(keys),
            backbone_roles=frame.role.value_counts().to_dict(),regional_coordinates=len(suite['pairs']),
            maximum_reference_projection_difference=float(abs(difference).max()),
            reference_zero_backbone_count=int((ref==0).sum()),reference_flow_sum=float(ref.sum()),
            series_projection='MIN_OF_PARALLEL_BUNDLE_SUMS',maximum_series_spread=float(spread.series_spread.max()),
            reference_with_zero_OD_attachments_node_imbalance=float(abs(net-balance).max()),
            reference_is_optimizer_initialization=False,coverage=coverage,mapping_identity=mapping_id,
            reference_id=frame.reference_id.iloc[0],graph_identity=g.identity,
            regional_trace_sha256=trace_sha,regional_reference_exact_join=True,
            regional_role_binding='Explicit saved CURRENT trace array keys and accepted low benchmark source; source role columns additionally checked for s03')
        if sid=='s03':rec['closure']=closure_mask(ctx,suite,ledger,inter)
        records.append(rec);write_json(dst/'REFERENCE_AUDIT.json',rec)
        print(sid,rec['backbone_count'],'backbone projection and fixed200 PASS',flush=True)
    write_json(ctx.root/'reference/REFERENCE_FREEZE.json',dict(status='PASS',suites=records,
        no_predictions_loaded=True,reference_space='ACNHN',source_physical_reference_hash=array_hash({'flow':physical.low_physical.current_flow})))
    return dict(status='PASS',suites=records)
