"""Pure synthetic metadata; no production files, model imports or fitting."""
import copy
import json

import pytest

from partC_rev1_src.pacnhn_recovery24 import quota_allocation as q

D = 'd'*64
T = 'a'*64
V = 'b'*64


@pytest.fixture
def population():
    # Full production-size key accounting, but every byte count/model is synthetic.
    counts = [2089, 2089, 2088, 2088, 2088]
    rows = [dict(region_key=f'{sid}:{i:06d}', pairs=1,
                 raw_quota_bytes=q.SUITE_CAP//count)
            for sid, count in zip(q.SUITES, counts) for i in range(count)]
    manifest = dict(schema='RECOVERY_XGB_COMPLETE_SUITE_QUOTAS_V1',
        source_budget_path='synthetic/budget.json', source_budget_sha256=D,
        per_suite_cap_bytes=q.SUITE_CAP, resident_multiplier=3., rows=rows,
        suite_sums={sid: sum(r['raw_quota_bytes'] for r in rows if r['region_key'].startswith(sid+':'))
                    for sid in q.SUITES})
    return manifest, [r['region_key'] for r in rows]


def candidate(*, size=1_400_000, rounds=32, depth=3, sha=D, train=T):
    return dict(model_sha256=sha, uncompressed_model_bytes=size,
                native_parameter_bytes=25, uncompressed_booster_bytes=size-25,
                boosting_rounds_total=rounds, max_depth=depth,
                trained_dataset_identity=train)


def obs(key='s03:000000', *, train=2., size=1_400_000, **overrides):
    value = dict(region_key=key, source_binding=dict(path='synthetic/'+key, sha256=D),
        current_train_dataset_identity=T, current_valid_dataset_identity=V,
        requires_current_dataset_refresh=False, selected=candidate(size=size),
        latest=candidate(size=size), audit_model_sha256=D,
        audit_train_dataset_identity=T, audit_valid_dataset_identity=V,
        train_wmape_percent=train, valid_wmape_percent=3., recorded_target_met=train < 1.)
    value.update(overrides)
    return value


def allocate(population, observations, **kwargs):
    previous, keys = population
    return q.allocate_quotas(previous, observations,
        prior_binding=dict(path='synthetic/quotas.json', sha256=D), expected_keys=keys, **kwargs)


def donors():
    return [obs(f's03:{i:06d}', train=.5, size=100_000) for i in range(10, 15)]


def decision(manifest, key='s03:000000'):
    return next(d for d in manifest['decisions'] if d['region_key'] == key)


def test_complete_reallocation_is_pure_and_preserves_other_suites(population):
    before = copy.deepcopy(population)
    measurements = [obs()] + donors(); original = copy.deepcopy(measurements)
    new, mapping = allocate(population, measurements)
    assert population == before and measurements == original
    assert len(new['rows']) == len(mapping) == 10442
    assert set(mapping) == set(population[1])
    assert all(total <= q.SUITE_CAP for total in new['suite_sums'].values())
    assert decision(new)['reason'] == 'GRANTED'
    original_quota = next(r['raw_quota_bytes'] for r in population[0]['rows'] if r['region_key']=='s03:000000')
    assert mapping['s03:000000'] == 2*original_quota
    for row in population[0]['rows']:
        if not row['region_key'].startswith('s03:'):
            assert mapping[row['region_key']] == row['raw_quota_bytes']
    assert not new['actual_models_loaded'] and not new['production_config_applied']


def test_no_unmeasured_capacity_reclaimed(population):
    new, mapping = allocate(population, [])
    assert all(mapping[r['region_key']] == r['raw_quota_bytes'] for r in population[0]['rows'])
    assert all(d['reason']=='NO_AUTHENTICATED_OBSERVATION' for d in new['decisions'])


def test_qualified_donor_preserves_larger_latest_and_native_bytes(population):
    value = obs(train=.1, size=100_000)
    value['latest'] = candidate(size=900_001, sha='e'*64)
    new, mapping = allocate(population, [value])
    assert mapping[value['region_key']] == (900_001*11+9)//10
    assert decision(new)['latest_model_sha256'] == 'e'*64
    assert decision(new)['selected_model_sha256'] == D


@pytest.mark.parametrize('change,expected', [
    ({'requires_current_dataset_refresh': True}, 'STALE'),
    ({'audit_train_dataset_identity':'f'*64}, 'STALE'),
    ({'audit_valid_dataset_identity':'f'*64}, 'STALE'),
    ({'audit_model_sha256':'f'*64}, 'STALE'),
    ({'recorded_target_met':False}, 'CONTRADICTORY'),
    ({'train_wmape_percent':None}, 'UNDEFINED'),
    ({'valid_wmape_percent':-1.}, 'UNDEFINED'),
    ({'latest':None}, 'MISSING_SELECTED'),
    ({'selected':{}}, 'INVALID_OR_MISSING'),
])
def test_stale_missing_or_contradictory_donor_never_releases_bytes(population, change, expected):
    value = obs(train=.5, size=100_000, **change)
    new, mapping = allocate(population, [value])
    old = next(r['raw_quota_bytes'] for r in population[0]['rows'] if r['region_key']==value['region_key'])
    assert mapping[value['region_key']] == old
    assert decision(new)['reason'].startswith(expected)


def test_no_compressed_file_size_proxy(population):
    value = obs(train=.5)
    value['selected'].pop('uncompressed_model_bytes')
    value['selected']['size_bytes'] = 1
    new, _ = allocate(population, [value])
    assert decision(new)['reason'].startswith('INVALID_OR_MISSING')


def test_wrong_native_sum_not_used(population):
    value = obs(train=.5); value['latest']['native_parameter_bytes'] += 1
    new, _ = allocate(population, [value])
    assert decision(new)['reason'].startswith('INVALID_OR_MISSING')


def test_strict_one_percent_is_unresolved(population):
    new, _ = allocate(population, [obs(train=1.)] + donors())
    assert not decision(new)['qualified']
    assert decision(new)['reason']=='GRANTED'


def test_no_cross_suite_capacity_transfer(population):
    # Retained 10% safety is below quota here, but the new 2x request cannot fit.
    receiver = obs(size=1_200_000)
    other_donors = [obs(f's02:{i:06d}', train=.5, size=100_000) for i in range(10)]
    new, _ = allocate(population, [receiver]+other_donors)
    assert decision(new)['reason']=='INSUFFICIENT_UNUSED_SAME_SUITE_CAPACITY'


def test_near_target_first_and_explicit_diagnostic_reserved(population):
    # One donor funds one request; each receiving model has no old-quota append.
    measurements = [obs(train=1.01, size=1_200_000), obs('s03:000001',train=99.,size=1_200_000),
                    obs('s03:000010',train=.1,size=100_000), obs('s03:000011',train=.1,size=100_000)]
    new, _ = allocate(population, measurements)
    assert decision(new)['reason']=='GRANTED'
    assert decision(new,'s03:000001')['reason']=='INSUFFICIENT_UNUSED_SAME_SUITE_CAPACITY'
    reserved, _ = allocate(population, measurements, diagnostic_keys=['s03:000001'])
    assert decision(reserved,'s03:000001')['reason']=='GRANTED'
    assert decision(reserved)['reason']=='INSUFFICIENT_UNUSED_SAME_SUITE_CAPACITY'


def test_existing_headroom_does_not_consume_extra_quota(population):
    new, _ = allocate(population, [obs(size=500_000)] + donors())
    assert decision(new)['reason']=='EXISTING_QUOTA_HAS_REQUESTED_HEADROOM'


def test_round_depth_ceiling_is_not_silently_increased(population):
    value = obs(size=1_200_000)
    value['latest'] = candidate(size=1_200_000, rounds=128, depth=7)
    new, _ = allocate(population, [value] + donors())
    assert decision(new)['reason']=='NO_NEW_TREE_AT_REQUESTED_QUOTA'
    assert decision(new)['latest_headroom']['status']=='NO_NEW_TREE'


def test_selected_latest_headroom_distinct(population):
    value = obs(size=1_200_000)
    value['selected'] = candidate(size=100_000, rounds=128, depth=7)
    new, _ = allocate(population, [value] + donors())
    assert decision(new)['selected_headroom']['status']=='NO_NEW_TREE'
    assert decision(new)['latest_headroom']['status']=='APPEND'


def test_fourfold_requires_prior_twofold_trial_and_uses_original_quota(population):
    initial, _ = allocate(population, [obs()] + donors())
    denied, _ = allocate((initial,population[1]), [obs()] + donors(), multiplier=4)
    assert decision(denied)['reason']=='FOURFOLD_REQUIRES_BOUND_TWOFOLD_TRIAL'
    value = obs(size=2_600_000, requested_round_increment=32,
        twofold_trial=dict(allocation_sha256=D, status='FITTED', new_boosting_rounds=8,
                           result_binding=dict(path='synthetic/trial.json',sha256=D)))
    result, mapping = allocate((initial,population[1]), [value] + donors(), multiplier=4)
    baseline = next(r['baseline_raw_quota_bytes'] for r in initial['rows'] if r['region_key']==value['region_key'])
    assert mapping[value['region_key']] == 4*baseline
    assert decision(result)['reason']=='GRANTED'


def test_zero_tree_trial_is_not_fitting(population):
    initial, _ = allocate(population, [obs()] + donors())
    value = obs(twofold_trial=dict(allocation_sha256=D, status='FITTED',new_boosting_rounds=0,
                                  result_binding=dict(path='synthetic/trial',sha256=D)))
    with pytest.raises(ValueError,match='zero-tree'):
        allocate((initial,population[1]), [value], multiplier=4)


@pytest.mark.parametrize('mutation', ['missing','duplicate','unknown','sum','cap','factor','bool','depth','population'])
def test_invalid_complete_manifests_rejected(population, mutation):
    previous, keys = copy.deepcopy(population)
    if mutation=='missing': previous['rows'].pop()
    if mutation=='duplicate': previous['rows'][-1]=copy.deepcopy(previous['rows'][0])
    if mutation=='unknown': previous['rows'][0]['region_key']='s03:999999'
    if mutation=='sum': previous['suite_sums']['s03']-=1
    if mutation=='cap': previous['per_suite_cap_bytes']+=1
    if mutation=='factor': previous['resident_multiplier']=2.
    if mutation=='bool': previous['rows'][0]['raw_quota_bytes']=True
    if mutation=='depth': previous['rows'][0]['raw_quota_bytes']=q.REGION_CAP+1
    if mutation=='population': keys.pop()
    with pytest.raises(ValueError): allocate((previous,keys), [])


@pytest.mark.parametrize('multiplier',[1,3,8,True,2.0])
def test_only_two_or_four_factor(population,multiplier):
    with pytest.raises(ValueError): allocate(population,[],multiplier=multiplier)


def test_source_projection_is_bound_and_no_payload_access():
    selected=candidate(size=100_000)
    selected['audit']=dict(model_sha256=D,train_dataset_identity=T,valid_dataset_identity=V,
                          TRAIN={'WMAPE_percent':.2},VALID={'WMAPE_percent':.3},target_met=True)
    state=dict(selected={'XGBOOST':selected},xgb_latest=candidate(),dataset_identity=T,
               valid_dataset_identity=V,requires_current_dataset_refresh=False)
    result=q.observation_from_state('s03:000000',state,source_binding=dict(path='not-a-real-file',sha256=D))
    assert result['train_wmape_percent']==.2 and result['selected'] is not selected
    assert result['source_binding']['path']=='not-a-real-file'


def test_headroom_matches_explicit_legacy_cases():
    a=candidate(size=100,rounds=32)
    assert q.headroom(a,200,T)['append_rounds']==8
    assert q.headroom(a,100,T)['status']=='NO_NEW_TREE'
    assert q.headroom(a,100,'c'*64)['status']=='CHANGED_TRAIN_RESTART'
    assert q.headroom(candidate(size=100,rounds=128),100,T)['restart_depth']==5


def test_observation_duplicates_and_missing_digest_rejected(population):
    with pytest.raises(ValueError,match='duplicate'):
        allocate(population,[obs(),obs()])
    bad=obs(); bad['source_binding']['sha256']='missing'
    with pytest.raises(ValueError,match='SHA'):
        allocate(population,[bad])


def test_metadata_manifest_is_strict_json(population):
    new,_=allocate(population,[obs()] + donors())
    assert json.loads(json.dumps(new,allow_nan=False))==new
