"""Small isolated synthetic recovery checks; no campaign inputs or TEST targets."""
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from partC_rev1_src.pacnhn_recovery24 import training_recovery as tr


class FakeContext:
    def __init__(self, base):
        self.root = base/'new'; self.source_root = base/'old'
        self.root.mkdir(); self.source_root.mkdir()
        self.path = self.root/'config.json'; self.path.write_text('{}')
        self.c = dict(phase_id='PACNHN_RECOVERY24', run_id='synthetic', recovery={})
        self.bindings = {}; self.actions = []; self.flushed = False

    def require_action(self, action):
        assert action == 'training'; self.actions.append(action)

    def bind(self, path, role, expected=None):
        digest = tr.sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Input SHA256 mismatch')
        self.bindings[str(path)] = digest
        return digest

    def guard(self, deadline=None):
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError('Synthetic deadline')

    def flush_bindings(self):
        self.flushed = True


@pytest.fixture
def ctx(tmp_path):
    return FakeContext(tmp_path)


def config():
    root = Path(__file__).resolve().parents[3]
    c = tr.read_json(root/'partC_doc_icnn/pacnhn_v1/training_config_v4.json')
    c.update(icnn_updates_per_block=2, recovery_initial_blocks=1, recovery_extra_blocks=0,
        recovery_min_gain_pp=.01, recovery_min_gain_relative=.01,
        recovery_valid_degradation_pp=.02, recovery_valid_degradation_relative=.02,
        recovery_family_seconds=1200, recovery_xgb_round_increments=(8, 16, 32),
        recovery_xgb_blocks=1, recovery_region_seconds=300)
    return c


def data():
    u = np.array([[0.], [.5], [1.], [1.5]])
    tau = 2.+3.*u**4
    c0 = np.array([2.])
    return dict(u=u, tau=tau, c0=c0, records=[], role='TRAIN', unique_rows=len(u),
        dataset_identity=tr.array_hash(dict(u=u, tau=tau, c0=c0)), train_cost_mean=float(tau.mean()))


def empty_state():
    return dict(visits=0, eligible={f: [] for f in tr.FAMILIES}, selected={}, optimizers={},
        blocks_by_width={}, xgb_latest=None, candidate_counter=0, retired=[], failures=[],
        recovery_failures=[], recovery_history=[], recovery_blocks=0, recovery_visits=0)


def candidate(train=2., valid=3., tid='T', vid='V'):
    return dict(candidate_id='c', model_path='x', model_sha256='a'*64, model_kind='test', width=32,
        audit=dict(train_dataset_identity=tid, valid_dataset_identity=vid, target_met=train < 1,
            TRAIN=dict(WMAPE_percent=train), VALID=dict(WMAPE_percent=valid)))


def label_fixture(ctx):
    root = ctx.source_root/'labels'; root.mkdir()
    graph = ctx.source_root/'graph.npz'; graph.write_bytes(b'synthetic bound graph; loader supplied explicitly')
    region = SimpleNamespace(graph_sha256=tr.sha(graph), identity='graph', support_identity='support',
        m=1, c0=np.array([2.]), pair_keys=np.array([17], np.int64))
    record = dict(suite_id='s03', region_index=17, path=str(graph), sha256=tr.sha(graph))
    identity = dict(graph_sha256=region.graph_sha256, graph_identity='graph', support_identity='support')
    tr.write_json(root/'REGION_IDENTITY.json', identity)
    index = {}
    for role, q in [('TRAIN', np.array([1.])), ('VALID', np.array([2.]))]:
        input_id = tr.array_hash({'u': q})
        sample = hashlib.sha256(f'graph|{role}|{input_id}'.encode()).hexdigest()
        folder = root/role; folder.mkdir()
        inp = folder/'input.npz'; np.savez(inp, u=q)
        req = dict(sample_id=sample, role=role, input_id=input_id, graph_identity='graph',
            support_identity='support', input_file_sha256=tr.sha(inp))
        request = folder/'REQUEST.json'; tr.write_json(request, req)
        label = folder/'label.npz'; np.savez(label, u=q, tau=2.+q**4, c0=region.c0)
        receipt = folder/'RESULT.json'; tr.write_json(receipt, dict(sample_id=sample, role=role,
            input_identity=input_id, graph_identity='graph', graph_sha256=region.graph_sha256,
            support_identity='support', input_file_sha256=tr.sha(inp), label_sha256=tr.sha(label),
            admitted=True, fresh_certificate=True, quality='A', history=[{'discard_this': True}]))
        index[input_id] = dict(sample_id=sample, role=role, input_file=str(inp), input_file_sha256=tr.sha(inp),
            request_metadata=str(request), request_metadata_sha256=tr.sha(request),
            selected_label=dict(path=str(label), sha256=tr.sha(label), receipt=str(receipt), receipt_sha256=tr.sha(receipt)))
    # Merely encountering this role in the index must never open its paths.
    index['blind'] = dict(role='TEST', selected_label={'path': 'DO_NOT_OPEN', 'receipt': 'DO_NOT_OPEN'})
    tr.write_json(root/'REQUEST_INDEX.json', index)
    return root, record, region, index


def test_disjoint_roots_required(ctx):
    ctx.root = ctx.source_root/'inside'
    with pytest.raises(ValueError, match='disjoint'):
        tr._roots(ctx)


def test_read_path_cannot_escape_source_or_new(ctx, tmp_path):
    external = tmp_path/'unbound'; external.write_text('x')
    with pytest.raises(ValueError, match='escapes'):
        tr._read_path(ctx, external)


@pytest.mark.parametrize('value,expected', [(0., True), (.999999, True), (1., False),
    (1.1, False), (float('inf'), False), (float('nan'), False), (-.01, False), (False, False)])
def test_target_strict_finite_current(value, expected):
    row = candidate(value)
    assert tr.target_met(row, 'T', 'V') is expected
    assert not tr.target_met(row, 'OTHER', 'V')


def test_loader_authenticates_requests_and_never_opens_test(ctx):
    root, record, region, _ = label_fixture(ctx)
    train, valid, same = tr.load_training_cohorts(ctx, record, labels_root=root, region=region)
    assert train['unique_rows'] == valid['unique_rows'] == 1
    assert same is region
    assert 'history' not in train['records'][0]
    assert all('DO_NOT_OPEN' not in path for path in ctx.bindings)


@pytest.mark.parametrize('field', ['role', 'sample_id', 'input_id', 'input_file_sha256', 'support_identity'])
def test_loader_rejects_authentic_but_wrong_request_metadata(ctx, field):
    root, record, region, index = label_fixture(ctx)
    key, row = next(iter(index.items()))
    request = tr.read_json(row['request_metadata']); request[field] = 'changed'
    tr.write_json(row['request_metadata'], request)
    row['request_metadata_sha256'] = tr.sha(row['request_metadata'])
    tr.write_json(root/'REQUEST_INDEX.json', index)
    with pytest.raises(ValueError, match='request metadata'):
        tr.load_training_cohorts(ctx, record, labels_root=root, region=region)


def test_loader_rejects_changed_label_even_if_index_hash_updated(ctx):
    root, record, region, index = label_fixture(ctx)
    row = next(iter(index.values()))
    Path(row['selected_label']['path']).write_bytes(b'changed')
    row['selected_label']['sha256'] = tr.sha(row['selected_label']['path'])
    tr.write_json(root/'REQUEST_INDEX.json', index)
    with pytest.raises(ValueError, match='admission/identity'):
        tr.load_training_cohorts(ctx, record, labels_root=root, region=region)


def test_test_role_denied_before_reads(ctx):
    with pytest.raises(PermissionError):
        list(tr._selected_entries(ctx, Path('absent'), None, 'TEST', lambda: None))


def test_tranche_requires_gain_and_valid_guard():
    c = config()
    assert tr.tranche_progress((5., 4.), (4.9, 4.05), c)['productive']
    assert not tr.tranche_progress((5., 4.), (4.9, 4.2), c)['productive']
    assert not tr.tranche_progress((5., 4.), (4.99, 3.), c)['productive']
    assert not tr.tranche_progress(None, (10., 10.), c)['productive']


def test_transactional_audit_failure_retains_selected_and_optimizer(ctx, monkeypatch):
    state = empty_state(); state['selected']['ICNN'] = candidate()
    prior = copy.deepcopy(state)
    def operation():
        state['selected']['ICNN'] = candidate(100.)
        state['optimizers']['32'] = {'path': 'bad'}
        state['candidate_counter'] += 1
        return {'updates': 2}
    def reject(*args):
        raise FloatingPointError('bad prediction')
    monkeypatch.setattr(tr, '_audit_select', reject)
    with pytest.raises(FloatingPointError):
        tr._transactional_block(ctx, ctx.root, 'ICNN', state, None, None, {}, lambda: None, operation)
    assert state['selected'] == prior['selected']
    assert state['optimizers'] == prior['optimizers']
    assert state['candidate_counter'] == 1


def test_real_synthetic_optimizer_resume_applies_new_lr_and_preserves_old(ctx):
    torch.set_num_threads(1)
    state = empty_state(); train = data(); c = config(); scale = np.ones(1)
    first = tr.icnn_block(ctx, ctx.root, state, train, scale, c, width=32, learning_rate=.001, deadline=time.time()+30)
    previous = copy.deepcopy(state['optimizers']['32']); old_sha = tr.sha(previous['path'])
    second = tr.icnn_block(ctx, ctx.root, state, train, scale, c, width=32, learning_rate=.0003, deadline=time.time()+30)
    saved = torch.load(state['optimizers']['32']['path'], weights_only=True)
    assert first['updates'] == second['updates'] == 2
    assert second['resume'] is True and second['total_updates'] == 4
    assert saved['optimizer']['param_groups'][0]['lr'] == .0003
    assert tr.sha(previous['path']) == old_sha
    assert torch.equal(saved['generator'], torch.load(previous['path'], weights_only=True)['generator']) is False


def test_resume_rejects_changed_scale_without_new_checkpoint(ctx):
    state = empty_state(); train = data(); c = config()
    tr.icnn_block(ctx, ctx.root, state, train, np.ones(1), c, width=32, learning_rate=.001, deadline=time.time()+30)
    before = state['candidate_counter']
    with pytest.raises(ValueError, match='incompatible'):
        tr.icnn_block(ctx, ctx.root, state, train, np.array([2.]), c, width=32, learning_rate=.001, deadline=time.time()+30)
    assert state['candidate_counter'] == before


def test_no_update_on_expired_budget(ctx):
    state = empty_state()
    with pytest.raises(TimeoutError):
        tr.icnn_block(ctx, ctx.root, state, data(), np.ones(1), config(), width=32, learning_rate=.001, deadline=time.time()-1)
    assert state['candidate_counter'] == 0


def test_zero_amplitude_diagnostic_does_not_mutate_or_impose_floor():
    train = data()
    model = tr.MonotoneICNN(train['c0'], np.ones(1), eta=np.zeros(1), width=32, amplitude=0.).export_model()
    arrays = model.export_arrays()
    receipt = tr.amplitude_boundary_diagnostic(model, train, 2, lambda: None)
    assert receipt['applicable'] and receipt['fresh_start_permitted']
    assert receipt['positive_amplitude_is_local_descent']
    assert all(np.array_equal(a, model.export_arrays()[k]) for k, a in arrays.items())
    assert model.amplitude == 0


@pytest.mark.parametrize('reason', ['budget', 'plateau', 'failure'])
def test_exhausted_family_never_dispatches_fit(ctx, reason):
    state = empty_state()
    policy = dict(elapsed_seconds=1200. if reason == 'budget' else 0.,
        unproductive_tranches=2 if reason == 'plateau' else 0,
        evaluated_tranches=0, failures=2 if reason == 'failure' else 0)
    state['recovery_family_policy'] = {'ICNN': policy}
    result = tr._fit_family(ctx, ctx.root, {}, None, state, data(), data(), np.ones(1), config(), 'ICNN', time.time()+30, None)
    assert result[0]['updates'] == 0
    assert result[0]['status'].startswith('STOPPED')


def test_property_audit_not_invented_from_finite_fit():
    train = data(); valid = dict(train)
    row = candidate(.5, 2., train['dataset_identity'], valid['dataset_identity'])
    region = SimpleNamespace(pair_keys=np.array([1]), graph_sha256='graph')
    result = tr._normal_record(dict(suite_id='s03', region_index=0), region, row, train, valid, 'ICNN')
    assert result['fit_sanity_status'] == 'PASS'
    assert result['property_status'] == 'PENDING' and result['property_model_sha256'] is None
    assert result['audit_model_sha256'] == result['model_sha256']


def test_explicit_empty_task_list_is_not_success(ctx):
    tr.write_json(ctx.source_root/'SUPPORT_FREEZE.json', dict(status='PASS', suites=[]))
    with pytest.raises(ValueError, match='Nonempty'):
        tr.run(ctx, deadline=time.time()+30, region_keys=[], max_workers=1)


@pytest.mark.parametrize('marker', ['training/MODEL_SELECTION_FREEZE.json',
    'training/REGISTRY_FREEZE.json', 'holdout/TEST_OPENED.json'])
def test_training_refused_after_selection_or_test_opening(ctx, marker):
    path = ctx.root/marker; path.parent.mkdir(parents=True); path.write_text('{}')
    with pytest.raises(PermissionError, match='sealed'):
        tr.run_region(ctx, dict(suite_id='s03', region_index=0), deadline=time.time()+30)


def test_source_control_requires_checkpoint_pin(ctx):
    path = ctx.source_root/'STATE.json'; path.write_text('{}')
    ctx.source_expected = {}
    with pytest.raises(ValueError, match='hash pin'):
        tr._bound(ctx, path, 'state')


def test_synthetic_xgb_block_rounds_and_raw_memory_record(ctx):
    train = data(); valid = dict(train); state = empty_state(); c = config()
    record = dict(suite_id='s03', region_index=0, xgb_raw_budget_bytes=4*1024**2)
    region = SimpleNamespace(base=np.array([2.]), k4=np.array([1.]), pair_source=np.array([0]),
        pair_target=np.array([1]), identity='graph', pair_keys=np.array([17]), graph_sha256='hash')
    outcomes = tr._fit_family(ctx, ctx.root, record, region, state, train, valid, np.ones(1), c,
        'XGBOOST', time.time()+30, None)
    assert any(row.get('rounds') == 8 for row in outcomes), outcomes
    selected = state['selected']['XGBOOST']
    row = tr._normal_record(record, region, selected, train, valid, 'XGBOOST')
    assert row['uncompressed_model_bytes'] >= row['uncompressed_booster_bytes'] > 0
    assert row['forecast_resident_booster_bytes'] == 3*row['uncompressed_model_bytes']


def test_full_synthetic_region_emits_four_current_cohort_records(ctx, monkeypatch):
    train = data(); valid = dict(train)
    region = SimpleNamespace(m=1, c0=train['c0'], base=np.array([2.]), k4=np.array([1.]),
        pair_source=np.array([0]), pair_target=np.array([1]), identity='graph',
        pair_keys=np.array([17]), graph_sha256='hash')
    record = dict(suite_id='s03', region_index=0, xgb_raw_budget_bytes=4*1024**2)
    state = empty_state()
    monkeypatch.setattr(tr, 'settings', lambda context: config())
    monkeypatch.setattr(tr, '_initial_state', lambda *args: state)
    monkeypatch.setattr(tr, 'load_training_cohorts', lambda *args, **kwargs: (train, valid, region))
    monkeypatch.setattr(tr.legacy, 'native_scale', lambda *args: (np.ones(1), {'graph_sha256': 'hash'}))
    result = tr.run_region(ctx, record, deadline=time.time()+30)
    assert result['status'] == 'RECOVERY_VISIT_COMPLETE', result
    assert {r['family'] for r in result['records']} == {'ICNN', 'XGBOOST', 'POWER4', 'FFTT'}
    assert result['trained_updates'] > 0 and result['new_xgb_rounds'] == 8
    assert ctx.flushed
    assert all(r['current_train_dataset_identity'] == train['dataset_identity'] for r in result['records'])
    assert all(not r['TEST_opened'] for r in result['records'])
    saved = tr.read_json(tr.region_root(ctx, record)/'STATE.json')
    assert saved['recovery_visits'] == 1


def test_new_labels_path_and_digest_are_paired(ctx, monkeypatch):
    monkeypatch.setattr(tr, 'settings', lambda context: config())
    with pytest.raises(ValueError, match='supplied together'):
        tr.run_region(ctx, dict(suite_id='s03', region_index=0), deadline=time.time()+30, labels_root=ctx.root)


def test_changed_cohort_resets_only_comparison_streak_and_phase():
    state = empty_state()
    tr.record_cohort_epoch(state, 'T1', 'V', 'index1')
    policy = dict(elapsed_seconds=901., failures=1, evaluated_tranches=7,
        unproductive_tranches=2, cohort_tranches=3, base_width=64,
        fresh_initialization_used=True, inactive_diagnostic={'old': True})
    state['recovery_family_policy'] = {'ICNN': policy}
    assert tr.record_cohort_epoch(state, 'T2', 'V', 'index2')
    assert policy['unproductive_tranches'] == policy['cohort_tranches'] == 0
    assert policy['elapsed_seconds'] == 901 and policy['failures'] == 1
    assert policy['evaluated_tranches'] == 7 and policy['fresh_initialization_used']
    assert 'base_width' not in policy and 'inactive_diagnostic' not in policy
    assert state['recovery_cohort_epochs'][-1]['prior_family_policies']['ICNN']['unproductive_tranches'] == 2


def test_same_population_index_change_never_refunds_allowances():
    state = empty_state(); tr.record_cohort_epoch(state, 'T', 'V', 'index1')
    policy = dict(elapsed_seconds=1200., unproductive_tranches=2, cohort_tranches=4)
    state['recovery_family_policy'] = {'ICNN': policy}
    before = copy.deepcopy(policy)
    assert not tr.record_cohort_epoch(state, 'T', 'V', 'index2')
    assert policy == before and len(state['recovery_cohort_epochs']) == 1


def test_saved_c0_identity_preserved_with_producer_graph_tolerance(ctx):
    root, record, region, index = label_fixture(ctx)
    row = next(iter(index.values())); path = Path(row['selected_label']['path'])
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays['c0'] += 5e-11
    np.savez(path, **arrays)
    receipt_path = Path(row['selected_label']['receipt'])
    receipt = tr.read_json(receipt_path); receipt['label_sha256'] = tr.sha(path)
    tr.write_json(receipt_path, receipt)
    row['selected_label'].update(sha256=tr.sha(path), receipt_sha256=tr.sha(receipt_path))
    tr.write_json(root/'REQUEST_INDEX.json', index)
    train, _, _ = tr.load_training_cohorts(ctx, record, labels_root=root, region=region)
    assert np.array_equal(train['c0'], arrays['c0'])
    assert not np.array_equal(train['c0'], region.c0)
    assert train['dataset_identity'] == tr.array_hash({'u': arrays['u'][None, :],
        'tau': arrays['tau'][None, :], 'c0': arrays['c0']})


def postfreeze_context(ctx,monkeypatch):
    """Real Context action/deadline/binding methods, only tiny fixture inputs."""
    from partC_rev1_src.pacnhn_recovery24.context import RecoveryContext
    from partC_rev1_src.pacnhn_recovery24 import labels
    root,record,region,_=label_fixture(ctx)
    actual=RecoveryContext.__new__(RecoveryContext);actual.__dict__.update(ctx.__dict__)
    now=time.time();actual.c=copy.deepcopy(ctx.c)
    actual.c.update(tasks=['training','holdout'],action_deadlines={'training':now-10,'holdout':now+3600},hard_deadline_epoch=now+7200)
    actual.config_sha256=tr.sha(actual.path);actual.action='holdout';actual.last_sample=now
    actual.source_expected={os.path.normcase(str(p.resolve())):tr.sha(p) for p in actual.source_root.rglob('*') if p.is_file()}
    freeze=actual.root/'training/MODEL_SELECTION_FREEZE.json';tr.write_json(freeze,{'created_epoch':now-1})
    digest=tr.sha(freeze);opening=actual.root/'holdout/TEST_OPENED.json'
    tr.write_json(opening,{'selection_freeze_sha256':digest,'epoch':now})
    def authenticate(c,path,expected):
        assert Path(path)==freeze and expected==digest
        c.bind(path,'SYNTHETIC_SELECTION_FREEZE',expected)
        return tr.read_json(path),{}
    monkeypatch.setattr(labels,'require_selection_freeze',authenticate)
    return actual,root,record,region,freeze,digest


def test_actual_context_reads_frozen_cohort_after_training_deadline_only_in_holdout_mode(ctx,monkeypatch):
    actual,root,record,region,freeze,digest=postfreeze_context(ctx,monkeypatch)
    train,valid,_=tr.load_training_cohorts(actual,record,labels_root=root,region=region,deadline=time.time()+60,
        read_only=True,selection_freeze_path=freeze,selection_freeze_sha256=digest)
    assert actual.action=='holdout' and train['unique_rows']==valid['unique_rows']==1
    with pytest.raises(TimeoutError):
        tr.load_training_cohorts(actual,record,labels_root=root,region=region,deadline=time.time()+60)
    with pytest.raises(PermissionError):tr.require_training_open(actual)


@pytest.mark.parametrize('bad',['missing_opening','wrong_opening_hash','early_opening','wrong_freeze','expired_holdout'])
def test_readonly_mode_rejects_unsealed_or_expired_authority(ctx,monkeypatch,bad):
    actual,root,record,region,freeze,digest=postfreeze_context(ctx,monkeypatch)
    opening=actual.root/'holdout/TEST_OPENED.json'
    if bad=='missing_opening':opening.unlink()
    elif bad=='wrong_opening_hash':tr.write_json(opening,{'selection_freeze_sha256':'e'*64,'epoch':time.time()})
    elif bad=='early_opening':tr.write_json(opening,{'selection_freeze_sha256':digest,'epoch':0})
    elif bad=='wrong_freeze':freeze=actual.root/'foreign.json'
    elif bad=='expired_holdout':actual.c['action_deadlines']['holdout']=time.time()-1
    monkeypatch.setattr(tr,'_load_role',lambda *args:pytest.fail('No payload read after failed authority'))
    with pytest.raises((PermissionError,FileNotFoundError,TimeoutError)):
        tr.load_training_cohorts(actual,record,labels_root=root,region=region,deadline=time.time()+60,
            read_only=True,selection_freeze_path=freeze,selection_freeze_sha256=digest)
