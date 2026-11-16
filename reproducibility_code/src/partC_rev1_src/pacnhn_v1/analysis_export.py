"""Audited native-backbone publication inputs; no solver or training runs."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .control import Context, read_json, write_json, sha
from .metrics import (admit_assignment, backbone_frame, extract_backbone,
                      freeze_keys, load_frozen_keys, compare_native)

SCENARIOS = {'open': 'OPEN', 'i95': 'REPRESENTED_I95'}
TIMING_BOUNDARY = ('Before graph/model load through final flow/trace/support exports; '
                   'excludes final RESULT/STAGES/HISTORY writes and post-run admission')


def _csv(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    frame.to_csv(temporary, index=False, float_format='%.17g')
    os.replace(temporary, path)


def _served_identity(od, serve):
    from .oracle import Demand
    if serve.dtype != bool or serve.shape != od.serve.shape:
        raise ValueError('Served-OD membership has another shape/type')
    return Demand(od.origins, od.offsets, od.destinations, od.demand, serve).identity


def _admit(ctx, directory, graph, od):
    flow, result, serve, bindings = admit_assignment(directory,
        expected_graph_identity=graph.identity, expected_edge_ids=graph.edge_ids)
    if result['OD_identity'] != _served_identity(od, serve):
        raise ValueError('Certificate served OD is not the common demand')
    for binding in bindings:
        ctx.bind(binding['path'], 'CERTIFIED_ANALYSIS_SOURCE', binding['sha256'])
    return flow, result, serve, bindings


def freeze_reference_keys(ctx):
    """Freeze before candidate-error inspection, or verify the existing freeze."""
    if ctx.c.get('continuation'):
        ctx.require_action('freeze-keys')
    from .topology import load_common
    graph, od, arrays, _ = load_common(ctx.root)
    path = ctx.root/'topology/backbone.parquet'
    backbone = backbone_frame(pd.read_parquet(path))
    flow, result, serve, bindings = _admit(ctx, ctx.root/'references/open', graph, od)
    bflow = extract_backbone(flow, graph.edge_ids, backbone)
    bindings += [dict(path=str(path), sha256=sha(path))]
    target = ctx.root/'metrics/keys'
    if not target.exists():
        freeze_keys(target, backbone, bflow, reference_id='P_REFERENCE_open',
                    reference_scenario='OPEN', source_bindings=bindings)
    keys, meta = load_frozen_keys(target, backbone, bflow, reference_id='P_REFERENCE_open')
    ctx.bind(target/'KEY_SELECTION_FREEZE.json', 'OPEN_REFERENCE_ONLY_KEYS')
    ctx.flush_bindings()
    return keys, meta


def method_row(case, result, reference_result, comparisons):
    """Keep independent solver rows distinct from the reference self-comparison."""
    costs = {r['population']: r for r in comparisons}
    cert = result['certificate']
    return dict(case_id=case['case_id'], method_id=case['method_id'], network=case['network'],
        suite_id=case.get('suite_id'), scenario=SCENARIOS[case['scenario']],
        status=result['status'], solver=result['method'], RG_SPTT=cert['RG_SPTT'],
        elapsed_seconds=result['elapsed_seconds'],
        wmape_backbone_percent=costs['BACKBONE']['wmape_percent'],
        wmape_key_percent=costs['KEY_LINKS']['wmape_percent'],
        reference_case_id='P_REFERENCE_'+case['scenario'], assignment_role=case['assignment_role'],
        certificate_fresh=result['final_certificate_fresh'],
        residual_kind='FW_GAP' if result['objective_has_potential'] else 'VI_RESIDUAL',
        nominal_tolerance=result['nominal_tolerance'],
        query_cap_seconds=result['deadline_epoch']-result['complete_query_start_epoch'],
        timing_boundary=TIMING_BOUNDARY, reference_status=reference_result['status'],
        reference_RG_SPTT=reference_result['certificate']['RG_SPTT'],
        served_OD_membership_equal=all(r['served_OD_membership_equal'] for r in comparisons),
        reported_state_iteration=result['reported_state_iteration'],
        reference_zero_link_count=costs['BACKBONE']['reference_zero_link_count'],
        reference_zero_positive_prediction_count=costs['BACKBONE']['reference_zero_positive_prediction_count'])


def validate_outcome_matrix(planned,outcomes):
    planned_by_id={r['case_id']:r for r in planned}
    actual_by_id={r['case']['case_id']:r['case'] for r in outcomes}
    if not planned or len(planned_by_id)!=len(planned) or len(actual_by_id)!=len(outcomes) or actual_by_id!=planned_by_id:
        raise ValueError('Assignment outcome ledger omits or changes planned cases')


def enrich_test_aggregates(learning, summary, *, registry_sha256):
    """Attach sealed error/denominator sums without inference or changing selection.

    Regional medians cannot reconstruct pooled WMAPE. Missing TEST outcomes remain
    missing; zero denominators remain undefined rather than acquiring an epsilon.
    """
    if (summary.get('registry_sha256') != registry_sha256
            or summary.get('test_results_used_for_selection') is not False):
        raise ValueError('TEST aggregate summary is not bound to the frozen selection')
    keys = ['suite_id', 'region_index', 'family']
    if learning.duplicated(keys).any():
        raise ValueError('Duplicate learning model coordinates')
    columns = ['TEST_absolute_error_sum', 'TEST_true_cost_sum', 'TEST_coordinates', 'TEST_inputs']
    if any(name in learning for name in columns):
        raise ValueError('TEST aggregate enrichment may not overwrite existing values')
    records = []
    regions = set()
    for region in summary['results']:
        key = (region['suite_id'], region['region_index'])
        if key in regions or region['registry_sha256'] != registry_sha256:
            raise ValueError('Duplicate region or changed TEST registry identity')
        regions.add(key)
        for model in region['models']:
            values = [model['absolute_error_sum'], model['true_cost_sum'],
                      model['coordinates'], model['unique_inputs']]
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       and np.isfinite(v) and v >= 0 for v in values):
                raise ValueError('Invalid TEST aggregate value')
            if (values[2] != int(values[2]) or values[3] != int(values[3])
                    or values[3] != region['admitted_test_inputs']):
                raise ValueError('TEST aggregate counts differ from admitted inputs')
            records.append(dict(suite_id=key[0], region_index=key[1], family=model['family'],
                _TEST_model_sha256=model['model_sha256'],
                _TEST_recorded_WMAPE=model['WMAPE_percent'],
                **dict(zip(columns, values))))
    if not records:
        return learning.assign(**{name: np.nan for name in columns})
    metrics = pd.DataFrame(records)
    if metrics.duplicated(keys).any():
        raise ValueError('Duplicate family TEST outcomes')
    joined = learning.merge(metrics, on=keys, how='outer', validate='one_to_one',
                            indicator=True, sort=False)
    if joined['_merge'].eq('right_only').any():
        raise ValueError('TEST outcome has no frozen learning model row')
    measured = joined.loc[joined['_merge'].eq('both')]
    if (not measured['status'].eq('AVAILABLE').all()
            or not measured['model_sha256'].eq(measured['_TEST_model_sha256']).all()
            or not measured['test_opened'].astype(str).str.lower().isin(['true', '1']).all()):
        raise ValueError('TEST outcome differs from the available frozen model/opening')
    if (not np.array_equal(measured['test_rows'].to_numpy(float), measured['TEST_inputs'].to_numpy(float))
            or not np.array_equal(measured['pairs'].to_numpy(float) * measured['TEST_inputs'].to_numpy(float),
                                  measured['TEST_coordinates'].to_numpy(float))):
        raise ValueError('TEST input/coordinate denominators differ from learning metadata')
    numerator = measured['TEST_absolute_error_sum'].to_numpy(float)
    denominator = measured['TEST_true_cost_sum'].to_numpy(float)
    expected = np.divide(100 * numerator, denominator, out=np.full(len(measured), np.nan), where=denominator > 0)
    for name in ['TEST_WMAPE_percent', '_TEST_recorded_WMAPE']:
        actual = pd.to_numeric(measured[name], errors='raise').to_numpy(float)
        if not np.allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True):
            raise ValueError('TEST native-cost WMAPE differs from its sealed sums')
    return joined.drop(columns=['_merge', '_TEST_model_sha256', '_TEST_recorded_WMAPE'])


def _load_ac_graph(ctx, sid, common, od_identity, frozen_report):
    from .oracle import Network
    report_path = ctx.root/'acnhn'/sid/'REPORT.json'
    report = read_json(report_path)
    if report!=frozen_report or report['OD_identity']!=od_identity:
        raise ValueError('AC report/OD differs from the witness-bound support freeze')
    ctx.bind(report_path, 'AC_GRAPH_REPORT')
    path = ctx.root/'acnhn'/sid/'graph.npz'
    ctx.bind(path, 'FROZEN_AC_GRAPH', report['graph_sha256'])
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    graph = Network(a['tail'], a['head'], a['blocked'], a['edge_ids'], a['node_ids'])
    n = int(a['backbone_count'])
    if n!=87582 or report['backbone_count']!=n or graph.identity != report['graph_identity']:
        raise ValueError('Frozen AC graph identity differs')
    for key in ('tail', 'head', 'edge_ids', 'base', 'k4', 'closed_mask'):
        if not np.array_equal(a[key][:n], common[key][:n]):
            raise ValueError('AC/P native B differs: '+sid+'/'+key)
    return graph, a, report


def export(ctx):
    """Export only admitted completed states; preserve unrun/unavailable rows."""
    if ctx.c.get('continuation'):
        ctx.require_action('analysis')
        from .gates import require_component
        require_component(ctx, 'analysis')
    from .topology import load_common
    started = time.time(); out = ctx.root/'analysis'; out.mkdir(parents=True, exist_ok=True)
    timing_path=out/'timing.csv'
    if not timing_path.exists():
        raise ValueError('Run timing.export_timing first so physical and query timing are included')
    ctx.bind(timing_path,'MEASURED_AND_CONDITIONAL_TIMING_LEDGER')
    support_path = ctx.root/'SUPPORT_FREEZE.json'
    witness_path = ctx.root/'COMPLETE_WITNESS_AUDIT.json'
    support = read_json(support_path); witness = read_json(witness_path)
    if support['status'] != 'PASS' or witness['status'] != 'PASS' or witness['support_freeze_sha256'] != sha(support_path):
        raise ValueError('Frozen support and complete OD witness audit required')
    ctx.bind(support_path, 'COMMON_SUPPORT'); ctx.bind(witness_path, 'EXACT_OD_WITNESSES')
    graph, od, common, _ = load_common(ctx.root)
    topo = read_json(ctx.root/'topology/RESULT.json')
    ctx.bind(ctx.root/'topology/RESULT.json', 'COMMON_TOPOLOGY')
    backbone_path = ctx.root/'topology/backbone.parquet'
    backbone = backbone_frame(pd.read_parquet(backbone_path))
    ctx.bind(backbone_path, 'COMMON_BACKBONE_GEOGRAPHY')
    bcount = len(backbone); references = {}
    for scenario in SCENARIOS:
        flow, result, serve, bindings = _admit(ctx, ctx.root/'references'/scenario, graph, od)
        references[scenario] = (extract_backbone(flow, graph.edge_ids, backbone), result, serve)
    keys, key_meta = load_frozen_keys(ctx.root/'metrics/keys', backbone, references['open'][0],
                                    reference_id='P_REFERENCE_open')
    ctx.bind(ctx.root/'metrics/keys/KEY_SELECTION_FREEZE.json', 'OPEN_REFERENCE_ONLY_KEYS')
    _csv(keys, out/'keys.csv')
    result_path = ctx.root/'ASSIGNMENT_RESULTS.json'
    outcomes = read_json(result_path); ctx.bind(result_path, 'FINAL_ASSIGNMENT_OUTCOMES')
    if not outcomes:
        raise ValueError('Empty assignment outcome matrix')
    matrix_path=ctx.root/'ASSIGNMENT_MATRIX.json';planned=read_json(matrix_path)
    ctx.bind(matrix_path,'EXPLICIT_ASSIGNMENT_MATRIX')
    validate_outcome_matrix(planned,outcomes)
    cases = [dict(case_id='P_REFERENCE_'+s, method_id='P/REFERENCE', network='P', suite_id=None,
                  scenario=s, assignment_role='REFERENCE_SELF', directory=ctx.root/'references'/s)
             for s in SCENARIOS]
    unavailable = []
    for outcome in outcomes:
        case = dict(outcome['case']); case['assignment_role'] = 'INDEPENDENT_CONTROL' if case['network']=='P' else 'SURROGATE'
        case['method_id'] = ('P/'+case['solver']) if case['network']=='P' else case['suite_id']+'/'+case['family']
        case['directory'] = ctx.root/'assignments'/case['case_id']
        if outcome['status'] in {'CONVERGED','TIME_LIMIT','STAGNATION'} and outcome.get('final_certificate_fresh'):
            if read_json(case['directory']/'RESULT.json') != outcome:
                raise ValueError('Assignment outcome ledger differs from case result')
            cases.append(case)
        else:
            unavailable.append(dict(case_id=case['case_id'],method_id=case['method_id'],network=case['network'],
                suite_id=case.get('suite_id'),scenario=SCENARIOS[case['scenario']],status=outcome['status'],
                solver=case['solver'],assignment_role=case['assignment_role'],
                reference_case_id='P_REFERENCE_'+case['scenario'],certificate_fresh=False,
                elapsed_seconds=outcome.get('elapsed_seconds'),
                note=outcome.get('error','No admitted final-state comparison'),
                query_cap_seconds=(outcome['deadline_epoch']-outcome['complete_query_start_epoch']) if 'deadline_epoch' in outcome else None))
    if len({c['case_id'] for c in cases+unavailable}) != len(cases)+len(unavailable):
        raise ValueError('Duplicate analysis case')
    graphs = {}; network = [dict(suite_id='P', resolution=None, nodes=len(graph.node_ids), edges=len(graph.tail),
        backbone_links=bcount, regional_links=topo['residual_physical_arc_count'],
        regions=None, active_regions=None, active_pairs=None,
        served_rows_open=int(references['open'][2].sum()), served_rows_closed=int(references['i95'][2].sum()))]
    for sid in ctx.c['suite_ids']:
        frozen=[r for r in support['suites'] if r['suite_id']==sid]
        if len(frozen)!=1:raise ValueError('Frozen suite inventory incomplete/duplicated')
        g, a, report = _load_ac_graph(ctx, sid, common,od.identity,frozen[0]); graphs[sid] = g
        reach = {r['scenario']:r for r in report['coverage']}
        partition = next(p for p in topo['partitions'] if p['suite_id']==sid)
        network.append(dict(suite_id=sid,resolution=ctx.c['resolutions'][sid],nodes=len(g.node_ids),edges=len(g.tail),
            backbone_links=bcount,regional_links=report['active_pairs'],regions=partition['region_count'],
            active_regions=report['active_regions'],active_pairs=report['active_pairs'],
            served_rows_open=reach['open']['served_rows'],served_rows_closed=reach['i95']['served_rows']))
    _csv(pd.DataFrame(network),out/'network.csv')
    methods=[]; populations=[]; convergence=[]; by_method={}; by_case={}
    links_path=out/'backbone_links.csv'; temporary=links_path.with_name(links_path.name+'.tmp.'+str(os.getpid()))
    for i,case in enumerate(cases):
        ctx.guard(); g=graph if case['network']=='P' else graphs[case['suite_id']]
        flow,result,serve,_=_admit(ctx,case['directory'],g,od)
        bflow=extract_backbone(flow,g.edge_ids,backbone)
        ref,ref_result,ref_serve=references[case['scenario']]
        rows,per=compare_native(backbone,ref,bflow,keys,scenario=SCENARIOS[case['scenario']],
            reference_case_id='P_REFERENCE_'+case['scenario'],case_id=case['case_id'],method_id=case['method_id'],
            assignment_role=case['assignment_role'],reference_serve=ref_serve,predicted_serve=serve,
            closed_mask=common['closed_mask'][:bcount],base=common['base'][:bcount],k4=common['k4'][:bcount])
        methods.append(method_row(case,result,ref_result,rows));populations.extend(rows)
        per=per.rename(columns={'lambda_reference':'reference_flow','lambda_estimate':'predicted_flow','is_key_link':'is_key'})
        for col in ('tail_longitude','tail_latitude','head_longitude','head_latitude'):
            per[col]=backbone[col].to_numpy()
        per.to_csv(temporary,mode='w' if i==0 else 'a',header=i==0,index=False,float_format='%.17g')
        by_method.setdefault(case['method_id'],{})[case['scenario']]=bflow
        by_case[case['case_id']]=bflow
        history_path=case['directory']/'HISTORY.json';ctx.bind(history_path,'FRESH_CERTIFICATE_HISTORY')
        for r in read_json(history_path):
            convergence.append(dict(case_id=case['case_id'],iteration=r['iteration'],elapsed_seconds=r['elapsed_seconds'],RG_SPTT=r['RG_SPTT']))
    os.replace(temporary,links_path)
    # Compact complete native vectors for a portable handoff, avoiding repeated
    # 64-character link IDs and geography once per method in the long CSV.
    wide=backbone[['link_id','native_index','tail_longitude','tail_latitude','head_longitude','head_latitude']].copy()
    wide['is_key']=wide.link_id.isin(keys.link_id)
    wide['represented_I95_closed']=common['closed_mask'][:bcount]
    for case_id,values in by_case.items():wide['flow__'+case_id]=values
    _csv(wide,out/'native_backbone_vectors.csv')
    _csv(pd.DataFrame(methods+unavailable),out/'methods.csv')
    _csv(pd.DataFrame(populations),out/'metric_populations.csv')
    _csv(pd.DataFrame(convergence),out/'convergence.csv')
    closure=[]
    open_serve,closed_serve=references['open'][2],references['i95'][2]
    lost=open_serve&~closed_serve;gained=~open_serve&closed_serve
    coverage=dict(open_closed_OD_membership_equal=bool(np.array_equal(open_serve,closed_serve)),
        open_served_OD_rows=int(open_serve.sum()),closed_served_OD_rows=int(closed_serve.sum()),
        lost_OD_rows=int(lost.sum()),gained_OD_rows=int(gained.sum()),
        lost_OD_demand=float(od.demand[lost].sum()),gained_OD_demand=float(od.demand[gained].sum()),
        demand_units='ANNUAL_KTON',demand_renormalized=False,
        distinction='Within-scenario candidate/P membership is tested separately from open/closed scenario demand changes')
    write_json(out/'I95_OD_COVERAGE.json',coverage)
    for method,pair in by_method.items():
        if set(pair)==set(SCENARIOS):
            row=backbone[['link_id','tail_longitude','tail_latitude','head_longitude','head_latitude']].copy()
            row['method_id']=method;row['open_flow']=pair['open'];row['closed_flow']=pair['i95']
            row['delta']=pair['i95']-pair['open'];row['is_closed']=common['closed_mask'][:bcount];closure.append(row)
            for key in ['open_closed_OD_membership_equal','lost_OD_rows','gained_OD_rows','lost_OD_demand','gained_OD_demand']:
                row[key]=coverage[key]
    if not closure:raise ValueError('No paired open/I95 results')
    _csv(pd.concat(closure,ignore_index=True),out/'I95.csv')
    for source_name,destination in [('LEARNING_RESULTS.csv','learning.csv'),('TRAINING_CURVES.csv','training_curves.csv')]:
        source=ctx.root/'training'/source_name
        ctx.bind(source,'FROZEN_TRAINING_ANALYSIS')
        frame = pd.read_csv(source, low_memory=False)
        if source_name == 'LEARNING_RESULTS.csv':
            summary_path = ctx.root/'training/FINAL_TEST_SUMMARY.json'
            if summary_path.exists():
                finished_path = ctx.root/'training/RESULT.json'
                ctx.bind(finished_path, 'FROZEN_TRAINING_COMPLETION')
                finished = read_json(finished_path)
                matches = [x for x in finished.get('test_dispositions', [])
                           if Path(x['path']).resolve() == summary_path.resolve()]
                if len(matches) != 1:
                    raise ValueError('Final TEST aggregate lacks an exact terminal binding')
                ctx.bind(summary_path, 'SEALED_TEST_AGGREGATE_SUMS', matches[0]['sha256'])
                frame = enrich_test_aggregates(frame, read_json(summary_path),
                    registry_sha256=finished['registry_sha256'])
        _csv(frame,out/destination)
    ctx.bind(ctx.root/'training/REGISTRY_FREEZE.json','FROZEN_MODEL_SELECTION')
    fields={p.name:pd.read_csv(p,nrows=0).columns.tolist() for p in out.glob('*.csv')}
    write_json(out/'DATA_DICTIONARY.json',dict(schema='PACNHN_ANALYSIS_V1',columns=fields,
        units={'flow':'ANNUAL_KTON','cost':'SECONDS','WMAPE':'PERCENT','RG_SPTT':'FRACTION',
               'longitude':'DEGREES','latitude':'DEGREES','elapsed_seconds':'MEASURED_WALL_SECONDS'},
        definitions={'wmape':'100*sum(abs(predicted-reference))/sum(reference), all native coordinates including zero references',
          'signed_bounded_error':'(predicted-reference)/max(predicted,reference); both zero gives zero',
          'delta':'closed_flow-open_flow; same frozen B coordinate',
          'missing_metrics':'Unavailable/unrun/uncertified cases have no invented metrics',
          'training_curves':'Each point reports its current full TRAIN and fixed VALID dataset; dataset growth can change TRAIN error',
          'TEST_aggregate_sums':'Sealed post-selection native-cost error and truth sums permit pooled WMAPE; missing outcomes stay empty and zero denominators stay undefined. Regional medians have different weighting.',
          'reference_self':'Identity check only; independent P-CFW remains a separate cold query',
          'geography':'Native longitude/latitude; display may clip to continental US, full-B metrics are unfiltered'}))
    manifest=dict(status='PASS',schema='PACNHN_ANALYSIS_V1',generated_epoch=time.time(),elapsed_seconds=time.time()-started,
        admitted_cases=len(cases),unavailable_cases=len(unavailable),key_selection=key_meta,
        inputs=list(ctx.bindings.values()),outputs=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(out.iterdir()) if p.suffix in {'.csv','.json'} and p.name!='PROVENANCE.json'])
    write_json(out/'PROVENANCE.json',manifest);ctx.flush_bindings();return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--action',required=True,choices=['freeze-keys','export'])
    args=parser.parse_args();ctx=Context(args.config,step_id='publication')
    (freeze_reference_keys if args.action=='freeze-keys' else export)(ctx)
