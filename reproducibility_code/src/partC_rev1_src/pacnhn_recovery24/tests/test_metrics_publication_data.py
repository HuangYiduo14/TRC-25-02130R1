import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from partC_rev1_src.pacnhn_recovery24 import metrics_publication_data as m
from partC_rev1_src.pacnhn_recovery24.assignments import matrix
from partC_rev1_src.pacnhn_recovery24.portfolio import SCHEMA, _digest, make_portfolio
from partC_rev1_src.pacnhn_v1.control import array_hash, sha, write_json
from partC_rev1_src.pacnhn_v1.metrics import compare_vectors


def ledger():
    rows = []
    for s in m.SCENARIOS:
        rows.append(dict(case=dict(case_id='P_REFERENCE_'+s, network='P', suite_id=None, family=None, solver='CFW', scenario=s),
            status='CONVERGED', final_certificate_fresh=True, assignment_role='REFERENCE_SELF',
            execution_status='REUSED_FROZEN_REFERENCE', new_scientific_execution=False))
    for c in matrix():
        rows.append(dict(case=c, status='CONVERGED', final_certificate_fresh=True,
            execution_status='REUSED_VERIFIED_IMMUTABLE_RESULT' if c['reuse_candidate'] else 'EXECUTED_RECOVERY_PORTFOLIO',
            new_scientific_execution=not c['reuse_candidate']))
    terminal = dict(status='COMPLETE_WITH_RECORDED_OUTCOMES', phase_id='PACNHN_RECOVERY24', run_id='toy',
        rows=46, reporting_rows=48, statuses={'CONVERGED':46}, all_converged=True)
    return rows, terminal


def validate(rows, terminal):
    return m.validate_ledger(rows, matrix(), rows[2:], terminal, phase_id='PACNHN_RECOVERY24', run_id='toy')


def test_exact48_ledger_has_distinct_reference_and_independent_cfw():
    rows, terminal = ledger()
    assert len(validate(rows, terminal)) == 48
    assert m.case_metadata(rows[0]['case'])['assignment_role'] == 'REFERENCE_SELF'
    assert m.case_metadata(rows[2]['case'])['assignment_role'] == 'INDEPENDENT_CONTROL'
    assert m.case_metadata(next(r['case'] for r in rows if r['case'].get('family') == 'ICNN'))['effective_method'] == 'ICNN_HYBRID'


@pytest.mark.parametrize('mode', ['missing', 'duplicate', 'renamed', 'bad_solver', 'fake_reuse', 'terminal_claim', 'bad_reference'])
def test_ledger_corruptions_rejected(mode):
    rows, terminal = ledger()
    if mode == 'missing': rows.pop()
    if mode == 'duplicate': rows[-1] = rows[-2]
    if mode == 'renamed': rows[-1]['case']['case_id'] = 'foreign'
    if mode == 'bad_solver': rows[-1]['case']['solver'] = 'CFW'
    if mode == 'fake_reuse': rows[-1]['new_scientific_execution'] = False
    if mode == 'terminal_claim': terminal['all_converged'] = False
    if mode == 'bad_reference': rows[0]['assignment_role'] = 'INDEPENDENT_CONTROL'
    with pytest.raises(ValueError): validate(rows, terminal)


def test_fresh_timed_state_retained_without_convergence_claim():
    rows, terminal = ledger(); rows[-1]['status'] = 'TIME_LIMIT'
    terminal.update(statuses={'CONVERGED':45, 'TIME_LIMIT':1}, all_converged=False)
    assert len(validate(rows, terminal)) == 48 and m.admitted(rows[-1])
    rows[-1]['final_certificate_fresh'] = False
    assert not m.admitted(rows[-1])


def test_full_population_keeps_zero_reference_and_undefined_denominator():
    ref = np.array([0., 2., 0.]); pred = np.array([3., 1., 4.]); key = np.array([True, False, True])
    stats, _ = m.independent_metrics(ref, pred, key, np.ones(3), np.zeros(3))
    assert stats['BACKBONE']['wmape_percent'] == 400.
    assert stats['BACKBONE']['reference_zero_positive_prediction_count'] == 2
    assert stats['KEY_LINKS']['wmape_percent'] is None
    assert stats['KEY_LINKS']['backbone_cost_error_percent'] is None
    old, _ = compare_vectors(ref, pred, key, np.ones(3), np.zeros(3))
    assert old[0]['wmape_percent'] == stats['BACKBONE']['wmape_percent']
    assert old[1]['wmape_status'] == 'UNDEFINED_ZERO_REFERENCE_TOTAL'


def test_independent_control_is_not_forced_to_zero():
    stats, _ = m.independent_metrics(np.array([1., 2.]), np.array([1.5, 2.]),
        np.array([True, False]), np.ones(2), np.zeros(2))
    assert stats['BACKBONE']['wmape_percent'] == pytest.approx(100/6)


def test_measured_new_load_failure_is_not_historical():
    row = next(r for r in ledger()[0] if r['case'].get('family') == 'ICNN')
    row.update(status='TIME_LIMIT_DURING_INPUT_LOAD', new_scientific_execution=False,
        complete_query_start_epoch=100., deadline_epoch=110., elapsed_seconds=12.)
    t = m.timing_row(row)
    assert t['time_kind'] == 'MEASURED_WALL' and t['accounting_role'] == 'DISJOINT_INTERVAL'
    assert t['end_epoch'] == 112. and t['observed_cap_overrun_seconds'] == 2.
    historical = ledger()[0][2]; historical.update(complete_query_start_epoch=100., deadline_epoch=110., elapsed_seconds=4.)
    assert m.timing_row(historical)['accounting_role'] == 'HISTORICAL_CONTEXT_ONLY'
    row = next(r for r in ledger()[0] if r['case'].get('family') == 'ICNN')
    row.update(status='UNRUN_STAGE_TIME_LIMIT', elapsed_seconds=0., new_scientific_execution=False)
    assert m.timing_row(row)['time_kind'] == 'NO_MEASURED_QUERY'


def test_only_new_disjoint_query_intervals_are_additive():
    rows = [dict(accounting_role='DISJOINT_INTERVAL',start_epoch=1.,end_epoch=3.),
        dict(accounting_role='DISJOINT_INTERVAL',start_epoch=3.,end_epoch=4.),
        dict(accounting_role='HISTORICAL_CONTEXT_ONLY',start_epoch=0.,end_epoch=10.)]
    m.validate_time_intervals(rows)
    rows[1]['start_epoch'] = 2.
    with pytest.raises(ValueError): m.validate_time_intervals(rows)


def proof_fixture():
    oo, cc = np.array([True, True, False]), np.array([True, False, False])
    populations = []
    for name, mask, demand, count in [('COMMON_SERVED',oo&cc,5.,2), ('LOST_CLOSED_VS_OPEN',oo&~cc,3.,1),
        ('GAINED_CLOSED_VS_OPEN',~oo&cc,0.,0), ('NEITHER_SERVED',~oo&~cc,7.,4)]:
        populations.append(dict(population=name,aggregate_OD_rows=int(mask.sum()), original_OD_row_multiplicity=count,
            demand=demand, demand_units='ANNUAL_KTON', membership_mask_identity=array_hash({'membership':mask})))
    return dict(status='PASS',schema='PACNHN_FROZEN_REFERENCE_I95_MEMBERSHIP_V1',demand_units='ANNUAL_KTON',
        demand_renormalized=False,aggregate_OD_rows=3,original_OD_row_multiplicity=7,total_original_demand=15.,
        populations=populations,closed_served_subset_of_open=True,
        scenarios={'open':{'served_rows':2},'i95':{'served_rows':1}}), oo, cc


def test_exact_membership_units_and_source_multiplicity():
    proof, oo, cc = proof_fixture()
    rows = m.validate_membership(proof, oo, cc)
    assert rows[1]['aggregate_OD_rows'] == 1 and rows[1]['demand'] == 3.
    assert sum(r['original_OD_row_multiplicity'] for r in rows) == 7


@pytest.mark.parametrize('corruption',['order','integer_mask','proof_hash','units','renormalized','multiplicity','demand','gained'])
def test_membership_corruptions_rejected(corruption):
    proof, oo, cc = proof_fixture()
    if corruption == 'order': cc = np.array([False, True, False])
    if corruption == 'integer_mask': oo = oo.astype(int)
    if corruption == 'proof_hash': proof['populations'][0]['membership_mask_identity'] = 'x'
    if corruption == 'units': proof['demand_units'] = 'scaled'
    if corruption == 'renormalized': proof['demand_renormalized'] = True
    if corruption == 'multiplicity': proof['original_OD_row_multiplicity'] += 1
    if corruption == 'demand': proof['total_original_demand'] += 1
    if corruption == 'gained': proof['closed_served_subset_of_open'] = False
    with pytest.raises(ValueError): m.validate_membership(proof, oo, cc)


def result_fixture(case, flow, start=1000.):
    potential = case.get('family') != 'XGBOOST'
    cert = dict(fresh=True,state_hash=array_hash({'flow':flow}),RG_SPTT=0.,TSTT=1.,SPTT=1.,
        raw_signed_gap=0.,arithmetic_allowance=1e-10,value=2. if potential else None,
        interpretation='CONVEX_POTENTIAL_FW_GAP' if potential else 'POINTWISE_VI_RESIDUAL_NO_POTENTIAL')
    reference = case['case_id'].startswith('P_REFERENCE')
    return dict(status='CONVERGED',method=case['solver'],accepted_updates=1,reported_state_iteration=1,
        reported_state='LAST_FRESH_CERTIFIED_STATE',certificate=cert,elapsed_seconds=1.,
        complete_query_start_epoch=start,deadline_epoch=start+10,nominal_tolerance=1e-4 if reference else 1e-3,
        graph_identity='graph',OD_identity='od_'+case['scenario'],objective_has_potential=potential,
        final_certificate_fresh=True)


def test_certificate_semantics_cannot_relabel_xgb_or_widen_new_budget():
    row = next(r for r in ledger()[0] if r['case'].get('family') == 'XGBOOST')
    result = result_fixture(row['case'],np.ones(2));result['intent'] = dict(complete_query_start_epoch=1000.,
        deadline_epoch=1010.,initialization='COLD_FEASIBLE_AON',dynamic_fallback=False)
    history = [dict(iteration=1,state_hash=result['certificate']['state_hash'])]
    m.validate_semantics(row,result,history,original_start=900.,assignment_deadline=1100.)
    with pytest.raises(ValueError):m.validate_semantics(row,result,history,original_start=1001.,assignment_deadline=1100.)
    result['objective_has_potential'] = True
    with pytest.raises(ValueError):m.validate_semantics(row,result,history,original_start=900.,assignment_deadline=1100.)


class FakeContext:
    def __init__(self, tmp_path):
        self.root=tmp_path/'new';self.source_root=tmp_path/'old';self.repo=Path(m.__file__).resolve().parents[2]
        self.root.mkdir();self.source_root.mkdir();self.path=self.root/'campaign.json'
        self.c=dict(phase_id='PACNHN_RECOVERY24',run_id='toy',resources={},source_bindings={},
            original_start_epoch=1000.,action_deadlines={'assignments':10000.})
        write_json(self.path,self.c);self.config_sha256=sha(self.path);self.source_expected={};self.bindings={}
    def require_action(self, action):
        assert action=='publication';return dict(science_deadline_epoch=time.time()+120)
    def guard(self,deadline=None):
        if deadline is not None and time.time()>=deadline:raise TimeoutError('deadline')
    def output_path(self,relative):return self.root/relative
    def bind(self,path,role,expected=None):
        digest=sha(path)
        if expected is not None and digest!=expected:raise ValueError('Hash mismatch')
        self.bindings[str(Path(path).resolve())]=digest;return digest
    def flush_bindings(self):return None


def test_reader_refuses_foreign_and_unpinned_metadata_and_detects_mutation(tmp_path):
    ctx=FakeContext(tmp_path);reader=m.Reader(ctx,time.time()+30)
    p=ctx.source_root/'one.json';write_json(p,{'a':1})
    with pytest.raises(ValueError):reader.json(p,'SOURCE',pinned=True)
    ctx.source_expected[os.path.normcase(str(p.resolve()))]=sha(p)
    assert reader.json(p,'SOURCE',pinned=True)=={'a':1}
    p.write_text('{}')
    with pytest.raises(ValueError):reader.unchanged()
    outside=tmp_path/'foreign';outside.write_bytes(b'no')
    with pytest.raises(ValueError):reader.path(outside,'FOREIGN')


def build_tiny_campaign(tmp_path, monkeypatch, *, unavailable=False):
    ctx=FakeContext(tmp_path);source=ctx.source_root;root=ctx.root
    rows,terminal=ledger();proof,oo,cc=proof_fixture()
    proof.update(common_OD_identity='original_od',graph_identity='graph',scenario_OD_identities={'open':'od_open','i95':'od_i95'})
    reports=[dict(suite_id=s,active_regions=1,active_pairs=1,graph_identity='graph',graph_sha256='a'*64,OD_identity='original_od',
        region_records=[dict(region_index=0,graph_identity='g',sha256='a'*64,pairs=1)]) for s in m.SUITES]
    support={'status':'PASS','suites':reports};write_json(source/'SUPPORT_FREEZE.json',support)
    portfolio=dict(schema=SCHEMA,phase_id='PACNHN_RECOVERY24',run_id='toy',source_root=str(source.resolve()),
        support_freeze_sha256=sha(source/'SUPPORT_FREEZE.json'),portfolios=make_portfolio(support,[]))
    portfolio['identity']=_digest(portfolio);write_json(root/'portfolios/PORTFOLIOS.json',portfolio)
    portfolio_sha=sha(root/'portfolios/PORTFOLIOS.json');terminal['portfolio_sha256']=portfolio_sha
    write_json(source/'topology/RESULT.json',dict(status='PASS',backbone_count=2,OD_identity='original_od',graph_identity='graph'))
    write_json(source/'COMPLETE_WITNESS_AUDIT.json',dict(status='PASS',support_freeze_sha256=portfolio['support_freeze_sha256']))
    (source/'topology/common_od.npz').write_bytes(b'opaque source OD proof binding')
    backbone=pd.DataFrame(dict(link_id=['a','b'],tail_longitude=[-100.,-99.],tail_latitude=[35.,36.],head_longitude=[-99.,-98.],head_latitude=[36.,37.]))
    backbone.to_parquet(source/'topology/backbone.parquet',index=False)
    keyroot=source/'metrics/keys';keyroot.mkdir(parents=True)
    keyfile=keyroot/'key_links_200.csv';pd.DataFrame({'link_id':['a']}).to_csv(keyfile,index=False)
    reffile=keyroot/'open_backbone_reference.csv';pd.DataFrame({'link_id':['a','b'],'lambda_reference':[1.,2.]}).to_csv(reffile,index=False)
    keymeta=dict(source_bindings=[],key_file_sha256=sha(keyfile),reference_file_sha256=sha(reffile),prediction_errors_used=False,
        key_selection_role='REFERENCE_ONLY',full_backbone_population_filtered=False)
    write_json(keyroot/'KEY_SELECTION_FREEZE.json',keymeta)
    historical=[]
    for index,row in enumerate(rows):
        case=row['case'];reference=case['case_id'].startswith('P_REFERENCE');reused=reference or case.get('reuse_candidate')
        folder=(source/'references'/case['scenario'] if reference else source/'assignments'/case['case_id'] if reused else root/'assignments'/case['case_id'])
        flow=np.array([1.,2.]) if case['scenario']=='open' else np.array([0.,2.])
        if case['case_id']=='P_CFW_open':flow[1]+=.1
        stored=result_fixture(case,flow,start=100. if reused else 1000.+index*20)
        if not reference:stored['case']=copy.deepcopy(case)
        if not reused:
            intent=dict(case=case,phase_id='PACNHN_RECOVERY24',run_id='toy',portfolio_sha256=portfolio_sha,
                complete_query_start_epoch=stored['complete_query_start_epoch'],deadline_epoch=stored['deadline_epoch'],
                initialization='COLD_FEASIBLE_AON',dynamic_fallback=False,original_OD_identity='original_od',graph_identity='graph')
            stored.update(intent=intent,portfolio_sha256=portfolio_sha,execution_status=row['execution_status'],new_scientific_execution=True)
            write_json(root/'intents'/(case['case_id']+'.json'),intent)
        if unavailable and index==len(rows)-1:
            row.update(status='TIME_LIMIT',final_certificate_fresh=False,new_scientific_execution=True)
            write_json(folder/'RESULT.json',row)
            terminal.update(statuses={'CONVERGED':45,'TIME_LIMIT':1},all_converged=False)
            continue
        write_json(folder/'RESULT.json',stored)
        np.savez(folder/'final_flow.npz',flow=flow)
        np.savez(folder/'coverage.npz',serve=oo if case['scenario']=='open' else cc)
        write_json(folder/'HISTORY.json',[dict(iteration=1,elapsed_seconds=1.,**stored['certificate'])])
        bindings={n:dict(path=str((folder/n).resolve()),sha256=sha(folder/n),bytes=(folder/n).stat().st_size)
            for n in ('RESULT.json','final_flow.npz','coverage.npz','HISTORY.json')}
        if reference:
            row.update(stored,native_artifacts=bindings)
        elif reused:
            historical.append(copy.deepcopy(stored))
            row.update(stored,execution_status='REUSED_VERIFIED_IMMUTABLE_RESULT',new_scientific_execution=False,
                native_artifacts=bindings,reused_from=str(folder),historical_query_elapsed_seconds=1.,
                reuse_basis='exact',reuse_admission_elapsed_seconds=.1)
            write_json(root/'assignments'/case['case_id']/'RESULT.json',row)
        else:row.update(stored,native_artifacts=bindings)
    proofroot=source/'audits/reference_i95_membership_v1'
    write_json(proofroot/'REPORT.json',proof)
    proofpaths=[source/'topology/common_od.npz']+[source/'references'/s/n for s in m.SCENARIOS for n in ('RESULT.json','coverage.npz','final_flow.npz')]
    write_json(proofroot/'INPUT_MANIFEST.json',{'bindings':[dict(path=str(p.resolve()),sha256=sha(p)) for p in proofpaths]})
    write_json(root/'REPORTING_RESULTS.json',rows);write_json(root/'ASSIGNMENT_RESULTS.json',rows[2:])
    write_json(root/'ASSIGNMENT_MATRIX.json',matrix());write_json(root/'ASSIGNMENT_STAGE_RESULT.json',terminal)
    write_json(source/'ASSIGNMENT_RESULTS.json',historical)
    for p in [source/'ASSIGNMENT_RESULTS.json',source/'topology/RESULT.json',source/'COMPLETE_WITNESS_AUDIT.json',proofroot/'REPORT.json',proofroot/'INPUT_MANIFEST.json',keyroot/'KEY_SELECTION_FREEZE.json']:
        ctx.source_expected[os.path.normcase(str(p.resolve()))]=sha(p)
    fixed=dict(edge_ids=np.array(['a','b']),base=np.ones(2),k4=np.zeros(2),closed_mask=np.array([True,False]))
    graphs={s:dict(identity='graph',edge_ids=np.array(['a','b']),closed_mask=fixed['closed_mask']) for s in ('P',)+m.SUITES}
    monkeypatch.setattr(m,'B_COUNT',2);monkeypatch.setattr(m,'KEY_COUNT',1)
    monkeypatch.setattr(m,'_graphs',lambda *args:(graphs,[{'suite_id':s} for s in graphs],fixed))
    oldframe=m.backbone_frame;oldcompare=m.compare_native
    monkeypatch.setattr(m,'backbone_frame',lambda f:oldframe(f,expected_count=2))
    monkeypatch.setattr(m,'compare_native',lambda *a,**kw:oldcompare(*a,**kw,expected_count=2,expected_key_count=1))
    monkeypatch.setattr(m,'load_frozen_keys',lambda *a,**kw:(pd.read_csv(keyfile),keymeta))
    return ctx


@pytest.mark.parametrize('unavailable',[False,True])
def test_complete_tiny48_export_authenticates_actual_native_files_without_science(tmp_path,monkeypatch,unavailable):
    ctx=build_tiny_campaign(tmp_path,monkeypatch,unavailable=unavailable)
    before={str(p):sha(p) for p in ctx.source_root.rglob('*') if p.is_file()}
    result=m.evaluate_assignments(ctx,ctx.root/'REPORTING_RESULTS.json',deadline=time.time()+90)
    assert result['status']=='VERIFIED_REPORT_DATA' and result['reporting_rows']==48
    table=pd.read_csv(ctx.root/'report_data/table2.csv')
    assert len(table)==48 and len(pd.read_csv(ctx.root/'report_data/i95_paired.csv'))==24
    assert table.loc[table.case_id=='P_CFW_open','wmape_backbone_percent'].iloc[0]>0
    assert table.loc[table.case_id=='P_REFERENCE_open','wmape_backbone_percent'].iloc[0]==0
    if unavailable:
        last=table.loc[table.case_id=='AC_s06_XGBOOST_i95'].iloc[0]
        # Family iteration order is explicit, inspect the actual final case.
        last=table.iloc[-1]
        assert not last.comparison_available and pd.isna(last.wmape_backbone_percent)
    assert before=={str(p):sha(p) for p in ctx.source_root.rglob('*') if p.is_file()}
    for item in result['files']:assert sha(item['path'])==item['sha256']
    with pytest.raises(ValueError,match='overwritten'):
        m.evaluate_assignments(ctx,ctx.root/'REPORTING_RESULTS.json',deadline=time.time()+90)


def test_changed_native_artifact_preserves_failure_without_final_manifest(tmp_path,monkeypatch):
    ctx=build_tiny_campaign(tmp_path,monkeypatch)
    p=ctx.source_root/'references/open/final_flow.npz';np.savez(p,flow=np.array([9.,2.]))
    with pytest.raises(ValueError,match='Hash'):
        m.evaluate_assignments(ctx,ctx.root/'REPORTING_RESULTS.json',deadline=time.time()+90)
    assert (ctx.root/'report_data/FAILURE.json').exists()
    assert not (ctx.root/'report_data/MANIFEST.json').exists()


def test_native_json_writer_does_not_create_nan_receipt(tmp_path):
    p=tmp_path/'bad.json'
    with pytest.raises(ValueError):m._new(p,{'value':float('nan')})
    assert not p.exists()
