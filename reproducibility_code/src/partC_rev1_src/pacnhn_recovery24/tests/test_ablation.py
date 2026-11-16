"""Synthetic-only nested-fraction, calibration and selection-freeze regressions."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from partC_rev1_src.pacnhn_recovery24 import ablation as a


def cohort(role='TRAIN', n=20):
    u = np.arange(1, n * 2 + 1, dtype=float).reshape(n, 2) / 10
    tau = np.array([1., 2.]) + np.array([.03, .02]) * u ** 4
    c0 = np.array([1., 2.])
    return dict(role=role, u=u, tau=tau, c0=c0,
                records=[{'input_identity': f'{role}-{i}'} for i in range(n)],
                dataset_identity=a.array_hash({'u': u, 'tau': tau, 'c0': c0}),
                unique_rows=n, train_cost_mean=float(tau.mean()) if role == 'TRAIN' else None)


def metadata():
    return [dict(suite_id='s03', family='ICNN', status='AVAILABLE',
                 region_index=i, pairs=(i % 5) + 1, VALID_WMAPE_percent=float(i % 7),
                 TEST_WMAPE_percent='MUST_NOT_BE_READ') for i in range(32)]


def test_strata_disjoint_and_stable_independent_of_input_order_or_test():
    rows = metadata()
    expected = a.select_regions(rows)
    other = copy.deepcopy(rows[::-1])
    for r in other:
        r['TEST_WMAPE_percent'] = float('nan')
    assert a.select_regions(other) == expected
    assert len({r['region_index'] for r in expected}) == 16
    assert [r['stratum'] for r in expected] == [s for s in a.STRATA for _ in range(4)]
    assert [r['region_index'] for r in expected[:4]] == [4, 9, 14, 19]


@pytest.mark.parametrize('rows', [metadata()[:15], metadata() + [metadata()[0]]])
def test_bad_selection_population_rejected(rows):
    with pytest.raises(ValueError):
        a.select_regions(rows)


def test_nested_prefixes_exact_membership_and_order_invariant():
    ids = [f'i-{i}' for i in range(64)]
    first = a.nested_indices(ids, 11, 's03:000001')
    reverse = a.nested_indices(ids[::-1], 11, 's03:000001')
    assert [len(first[f]) for f in a.FRACTIONS] == [6, 16, 32, 44, 64]
    previous = []
    for f in a.FRACTIONS:
        values = [ids[i] for i in first[f]]
        assert values[:len(previous)] == previous
        assert values == [ids[::-1][i] for i in reverse[f]]
        previous = values
    assert len(set(previous)) == 64


@pytest.mark.parametrize('ids', [[], ['a', 'a'], ['a', '']])
def test_invalid_train_ids_rejected(ids):
    with pytest.raises(ValueError):
        a.nested_indices(ids, 1, 's03:000001')


def test_subset_copy_identity_and_scale_are_subset_only():
    train = cohort()
    subset = a.subset_train(train, np.array([0, 1]))
    scale, info = a.subset_scale(subset)
    changed = copy.deepcopy(train)
    changed['u'][2:] *= 1e6
    changed['tau'][2:] *= 1e6
    again = a.subset_train(changed, np.array([0, 1]))
    assert subset['dataset_identity'] == again['dataset_identity']
    np.testing.assert_array_equal(scale, a.subset_scale(again)[0])
    assert subset['dataset_identity'] != train['dataset_identity']
    assert info['train_dataset_identity'] == subset['dataset_identity']
    assert info['VALID_or_TEST_used'] is False
    subset['u'][0] = 999
    assert np.max(train['u']) < 999


@pytest.mark.parametrize('role', ['VALID', 'TEST'])
def test_calibration_helpers_reject_nontrain(role):
    data = cohort(role)
    with pytest.raises(PermissionError):
        a.subset_train(data, np.array([0]))
    with pytest.raises(PermissionError):
        a.subset_scale(data)


def test_all_zero_flow_scale_is_explicit_positive_convention():
    data = cohort(); data['u'][:] = 0
    scale, note = a.subset_scale(data)
    np.testing.assert_array_equal(scale, [1., 1.])
    assert note['all_zero_flow_unit_convention'] is True


@pytest.mark.parametrize('ix', [[0, 0], [], [-1], [99], [0.5]])
def test_bad_subset_rejected(ix):
    with pytest.raises(ValueError):
        a.subset_train(cohort(), np.asarray(ix))


def test_fresh_icnn_repeat_and_subset_calibration(monkeypatch):
    # Short synthetic optimization only; production's exact1000 is source-bound.
    monkeypatch.setitem(a.FIT_POLICY, 'icnn_updates', 4)
    monkeypatch.setitem(a.FIT_POLICY, 'icnn_audit_interval', 2)
    train = a.subset_train(cohort(), np.arange(4))
    valid = cohort('VALID', 4)
    kwargs = dict(family='ICNN', seed=7, check=lambda: None,
                  structural_zero_mask=np.zeros(2, bool), structural_zero_evidence='synthetic')
    m1, r1 = a.fit_case(train, valid, **kwargs)
    m2, r2 = a.fit_case(train, valid, **kwargs)
    assert a.array_hash(m1.export_arrays()) == a.array_hash(m2.export_arrays())
    assert r1['initial_model_array_sha256'] == r2['initial_model_array_sha256']
    assert r1['actual_updates'] == 4 and len(r1['history']) == 2
    assert r1['TRAIN_calibration']['train_dataset_identity'] == train['dataset_identity']
    assert r1['TRAIN_calibration']['training_rows'] == 4
    assert r1['TRAIN_calibration']['VALID_or_TEST_used'] is False
    np.testing.assert_allclose(m1.scale, a.subset_scale(train)[0])
    altered = copy.deepcopy(valid); altered['tau'] *= 100
    _, r3 = a.fit_case(train, altered, **kwargs)
    assert r3['initial_model_array_sha256'] == r1['initial_model_array_sha256']


def test_xgb_fresh_center_uses_only_fraction_train(monkeypatch):
    monkeypatch.setitem(a.FIT_POLICY, 'xgb_rounds', 2)
    train = a.subset_train(cohort(), np.arange(4))
    model, result = a.fit_case(train, cohort('VALID', 4), family='XGBOOST', seed=7,
        check=lambda: None, structural_zero_mask=np.zeros(2, bool), structural_zero_evidence='synthetic')
    np.testing.assert_allclose(model.log_center, np.log(train['tau']).mean(axis=0))
    assert not np.allclose(model.log_center, np.log(cohort()['tau']).mean(axis=0))
    assert model.metadata['continued_from_existing_model'] is False
    assert model.metadata['log_center_train_dataset_identity'] == train['dataset_identity']
    assert result['actual_boosting_rounds'] == 2
    assert model.has_potential is False


def test_leaked_role_identity_rejected_before_fit():
    train = cohort(); valid = cohort('VALID')
    valid['records'][0]['input_identity'] = train['records'][0]['input_identity']
    with pytest.raises(ValueError, match='leakage'):
        a.fit_case(train, valid, family='ICNN', seed=7, check=lambda: None,
                   structural_zero_mask=np.zeros(2, bool), structural_zero_evidence='synthetic')


def test_deadline_guard_reached_before_fitting():
    def check():
        raise TimeoutError('synthetic deadline')
    with pytest.raises(TimeoutError):
        a.fit_case(cohort(), cohort('VALID'), family='ICNN', seed=7, check=check,
                   structural_zero_mask=np.zeros(2, bool), structural_zero_evidence='synthetic')


def test_summary_fraction_units_quantiles_and_missingness():
    rows = [dict(stratum='large', family='ICNN', fraction=.1,
                 TEST_WMAPE_percent=v, admitted_TEST_inputs=47) for v in (1., 2., 3., None)]
    summary = a.summarize_test(rows)
    row = summary[0]
    assert row['median_WMAPE'] == .02
    assert row['q25_WMAPE'] == pytest.approx(.015)
    assert row['q75_WMAPE'] == pytest.approx(.025)
    assert row['region_seed_rows'] == 3 and not row['complete_region_coverage']
    assert row['admitted_TEST_inputs'] == 141 and row['requested_TEST_inputs'] == 192
    assert summary[1]['median_WMAPE'] is None


def test_evaluation_denied_before_loading_targets(tmp_path, monkeypatch):
    calls = []
    ctx = SimpleNamespace(require_action=lambda x: calls.append(x), guard=lambda: None)
    def deny(*args):
        raise PermissionError('missing authenticated freezes')
    monkeypatch.setattr(a, '_evaluation_authority', deny)
    with pytest.raises(PermissionError):
        a.evaluate(ctx, {'path': 'unused', 'sha256': '0' * 64})
    assert calls == ['holdout']


def test_settings_requires_explicit_exact_fit_budget():
    cfg = dict(enabled=True, seconds=1800, regions_per_stratum=4, fractions=list(a.FRACTIONS),
               fit_policy=dict(a.FIT_POLICY), seed=11, selection_metadata_sha256='a' * 64)
    assert a._settings(SimpleNamespace(c={'ablation': cfg})) == cfg
    wrong = copy.deepcopy(cfg); wrong['fit_policy']['icnn_updates'] = 999
    with pytest.raises(ValueError):
        a._settings(SimpleNamespace(c={'ablation': wrong}))


def test_new_receipts_never_overwrite(tmp_path):
    p = tmp_path / 'a.json'
    a._new(p, {'version': 1})
    with pytest.raises(FileExistsError):
        a._new(p, {'version': 2})
    assert json.loads(p.read_text()) == {'version': 1}


def context(tmp_path):
    def bind(path, role, expected=None):
        digest = a.sha(path)
        if expected is not None and digest != expected:
            raise ValueError('binding mismatch')
        return digest
    return SimpleNamespace(root=tmp_path, source_root=tmp_path.parent / 'immutable',
        output_path=lambda r: tmp_path / r, bind=bind,
        c={'recovery': {'s03_deadline_epoch': 5000.}, 'phase_id': 'P', 'run_id': 'R'})


def test_fit_clock_cannot_reset_across_pilot_and_main(tmp_path, monkeypatch):
    ctx = context(tmp_path); root = tmp_path / 'ablation'; root.mkdir()
    plan = root / 'PLAN.json'; a._new(plan, {'immutable': True})
    monkeypatch.setattr(a.time, 'time', lambda: 100.)
    path, first, stop = a._fit_clock(ctx, root, plan, 4000.)
    original = path.read_bytes()
    assert stop == 1900.
    monkeypatch.setattr(a.time, 'time', lambda: 1000.)
    _, second, stop2 = a._fit_clock(ctx, root, plan, 4500.)
    assert second == first and stop2 == stop and path.read_bytes() == original
    path.write_text(json.dumps(dict(first, start_epoch=1000., deadline_epoch=2800.)))
    with pytest.raises(ValueError, match='binding'):
        a._fit_clock(ctx, root, plan, 4500.)


def input_fixture(ctx, tmp_path):
    path = tmp_path / 'holdout' / 'INPUT_FREEZE.json'
    a._new(path, {'status': 'FROZEN_BEFORE_FITTING', 'created_epoch': 100., 'targets_opened': False})
    binding = dict(path=str(path), sha256=a.sha(path))
    ctx.c['ablation'] = {'holdout_input_freeze': binding}
    return binding


def test_input_freeze_is_required_and_posttest_fits_denied(tmp_path, monkeypatch):
    ctx = context(tmp_path); ctx.c['ablation'] = {}
    with pytest.raises(PermissionError):
        a._input_freeze(ctx)
    monkeypatch.setattr(a.time, 'time', lambda: 200.)
    expected = input_fixture(ctx, tmp_path)
    assert a._input_freeze(ctx) == expected
    a._new(tmp_path / 'holdout' / 'TEST_OPENED.json', {'epoch': 150.})
    with pytest.raises(PermissionError, match='after TEST'):
        a._input_freeze(ctx)


def authority_fixture(tmp_path, monkeypatch):
    ctx = context(tmp_path); binding = input_fixture(ctx, tmp_path)
    root = tmp_path / 'ablation'; root.mkdir()
    plan = {'tasks': [{'task_id': str(i)} for i in range(160)]}
    plan_path = root / 'PLAN.json'; a._new(plan_path, plan)
    freeze_path = root / 'FIT_FREEZE.json'
    a._new(freeze_path, dict(status='ALL_160_DISPOSITIONS_FROZEN', completed=160, total=160,
        ablation_complete=True, plan_sha256=a.sha(plan_path),
        outcomes=[dict(task=t, status='AVAILABLE') for t in plan['tasks']], frozen_epoch=200.))
    global_data = dict(status='FROZEN', phase_id='P', run_id='R', created_epoch=201.,
        test_targets_opened=False, holdout_input_freeze=binding,
        ablation_fit_freeze=dict(path=str(freeze_path), sha256=a.sha(freeze_path)))
    global_path = tmp_path / 'MODEL_SELECTION_FREEZE.json'; a._new(global_path, global_data)
    monkeypatch.setattr(a, '_read_plan', lambda c: (plan, plan_path))
    return ctx, global_path, global_data


def test_exact_global_and_ablation_freezes_accept(tmp_path, monkeypatch):
    ctx, path, data = authority_fixture(tmp_path, monkeypatch)
    assert a._evaluation_authority(ctx, dict(path=str(path), sha256=a.sha(path)))[2] == path


@pytest.mark.parametrize('field,value', [('phase_id', 'OTHER'), ('created_epoch', 199.),
    ('test_targets_opened', True), ('holdout_input_freeze', {'path': 'different', 'sha256': '0' * 64}),
    ('ablation_fit_freeze', {'path': 'different', 'sha256': '0' * 64})])
def test_foreign_or_early_global_freeze_rejected(tmp_path, monkeypatch, field, value):
    ctx, path, data = authority_fixture(tmp_path, monkeypatch)
    data[field] = value; path.write_text(json.dumps(data))
    with pytest.raises(PermissionError):
        a._evaluation_authority(ctx, dict(path=str(path), sha256=a.sha(path)))


def run_fixture(tmp_path, monkeypatch):
    from partC_rev1_src.pacnhn_recovery24 import training_recovery
    from partC_rev1_src.pacnhn_v1 import models, training
    ctx = context(tmp_path)
    ctx.require_action = lambda action: None
    ctx.guard = lambda **kwargs: None
    ctx.flush_bindings = lambda: None
    monkeypatch.setattr(a.time, 'time', lambda: 200.)
    input_fixture(ctx, tmp_path)
    regions = [dict(region_key=f's03:{i:06d}', stratum=a.STRATA[i//4], record={'i': i}) for i in range(16)]
    tasks = [dict(task_id=f"{r['region_key'].replace(':','_')}_{int(f*100):03d}_{fam}",
                  region_key=r['region_key'], stratum=r['stratum'], fraction=f, family=fam)
             for r in regions for f in a.FRACTIONS for fam in a.FAMILIES]
    plan = dict(regions=regions, tasks=tasks, seed=7)
    path = tmp_path/'ablation'/'PLAN.json'; a._new(path, plan)
    monkeypatch.setattr(a, '_read_plan', lambda ignored: (plan, path))
    train, valid = cohort(), cohort('VALID')
    train['request_index_sha256'] = 'a'*64
    region = SimpleNamespace(graph_sha256='b'*64, support_identity='c'*64)
    monkeypatch.setattr(training_recovery, 'load_training_cohorts', lambda *args, **kwargs: (train, valid, region))
    monkeypatch.setattr(training, '_structural_zeros', lambda ignored: np.zeros(2, bool))
    calls = []
    def fitting(subset, fixed_valid, **kwargs):
        assert fixed_valid is valid
        calls.append((kwargs['family'], subset['unique_rows'], subset['dataset_identity']))
        return object(), {'status': 'AVAILABLE', 'TRAIN_identity': subset['dataset_identity']}
    monkeypatch.setattr(a, 'fit_case', fitting)
    def save(model, path, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'SYNTHETIC_NO_MODEL')
        return dict(path=str(path), sha256=a.sha(path))
    monkeypatch.setattr(models, 'save_model', save)
    return ctx, plan, calls


def test_160_fit_freeze_pilot_reuse_and_original_clock(tmp_path, monkeypatch):
    ctx, plan, calls = run_fixture(tmp_path, monkeypatch)
    pilot = a.run(ctx, 4000., pilot_only=True)
    assert pilot['completed'] == 2 and not pilot['all_fits_frozen']
    assert not (tmp_path/'ablation'/'FIT_FREEZE.json').exists()
    assert [c[:2] for c in calls] == [('ICNN',20), ('XGBOOST',20)]
    first_clock = (tmp_path/'ablation'/'RUN_CLOCK.json').read_bytes()
    final = a.run(ctx, 4500.)
    assert len(calls) == 160 and final['completed'] == 160 and final['all_fits_frozen']
    assert (tmp_path/'ablation'/'RUN_CLOCK.json').read_bytes() == first_clock
    frozen = a.read_json(tmp_path/'ablation'/'FIT_FREEZE.json')
    assert [r['task'] for r in frozen['outcomes']] == plan['tasks']
    assert all(a.sha(r['result_path']) == r['result_sha256'] for r in frozen['outcomes'])
    with pytest.raises(FileExistsError):
        a.run(ctx, 4500.)


def test_timeout_is_preserved_and_all_remaining_terminally_unattempted(tmp_path, monkeypatch):
    ctx, plan, calls = run_fixture(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        raise TimeoutError('synthetic budget')
    monkeypatch.setattr(a, 'fit_case', fail)
    result = a.run(ctx, 4000.)
    assert result['completed'] == 0 and result['all_dispositions_frozen'] and not result['ablation_complete']
    statuses = [r['status'] for r in result['outcomes']]
    assert statuses.count('TIME_LIMIT') == 1 and statuses.count('UNATTEMPTED_TIME_LIMIT') == 159
    assert (tmp_path/'ablation'/'FIT_FREEZE.json').exists()
    timeout = next(r for r in result['outcomes'] if r['status']=='TIME_LIMIT')
    assert 'synthetic budget' in a.read_json(timeout['result_path'])['error']


def test_one_failed_fit_does_not_block_other_159_or_global_freeze(tmp_path, monkeypatch):
    ctx, plan, calls = run_fixture(tmp_path, monkeypatch)
    original = a.fit_case
    failed = []
    def one_failure(*args, **kwargs):
        if not failed:
            failed.append(True)
            raise ArithmeticError('synthetic isolated case failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(a, 'fit_case', one_failure)
    result = a.run(ctx, 4000.)
    assert len(calls) == 159 and result['completed'] == 159
    assert result['all_dispositions_frozen'] and not result['ablation_complete']
    freeze_path = tmp_path/'ablation'/'FIT_FREEZE.json'
    global_path = tmp_path/'MODEL_SELECTION_FREEZE.json'
    a._new(global_path, dict(status='FROZEN', phase_id='P', run_id='R', created_epoch=201.,
        test_targets_opened=False, holdout_input_freeze=ctx.c['ablation']['holdout_input_freeze'],
        ablation_fit_freeze=dict(path=str(freeze_path), sha256=a.sha(freeze_path))))
    authorized = a._evaluation_authority(ctx, dict(path=str(global_path), sha256=a.sha(global_path)))
    assert authorized[1]['completed'] == 159
    assert sum(r['status']=='FAILED' for r in authorized[1]['outcomes']) == 1
    with pytest.raises(FileExistsError):
        a.run(ctx, 4000.)


def test_terminal_missing_fits_evaluate_without_loading_any_targets(tmp_path, monkeypatch):
    from partC_rev1_src.pacnhn_recovery24 import labels
    ctx, plan, calls = run_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(a, 'fit_case', lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError('stop')))
    a.run(ctx, 4000.)
    frozen = a.read_json(tmp_path/'ablation'/'FIT_FREEZE.json')
    monkeypatch.setattr(a, '_evaluation_authority', lambda *args: (plan, frozen, tmp_path/'GLOBAL.json', 'a'*64))
    def forbidden(*args, **kwargs):
        raise AssertionError('Missing fits must not load targets')
    monkeypatch.setattr(labels, 'load_holdout_dataset', forbidden)
    result = a.evaluate(ctx, {'path':str(tmp_path/'GLOBAL.json'),'sha256':'a'*64})
    assert result['measured_cases'] == 0 and result['unavailable_fits'] == 160
    assert not result['ablation_complete']
    with (tmp_path/'ablation'/'evaluation'/'FIGURE5_SUMMARIES.csv').open() as stream:
        import csv
        summaries = list(csv.DictReader(stream))
    assert len(summaries) == 40
    assert all(r['median_WMAPE']=='' and r['unavailable_fit_regions']=='4' for r in summaries)
