import copy
import numpy as np
import pandas as pd
import pytest
from partC_rev1_src.pacnhn_v1.analysis_export import enrich_test_aggregates


def fixture():
    rows = []
    results = []
    for region, numerator, truth in [(0, .1, 10.), (1, 900., 1000.)]:
        error = 100 * numerator / truth
        rows.append(dict(suite_id='s03', region_index=region, family='ICNN', status='AVAILABLE',
                         model_sha256=str(region), test_opened=True, test_rows=2, pairs=1,
                         TEST_WMAPE_percent=error))
        results.append(dict(suite_id='s03', region_index=region, registry_sha256='frozen',
                            admitted_test_inputs=2, models=[dict(family='ICNN', model_sha256=str(region),
                            absolute_error_sum=numerator, true_cost_sum=truth, coordinates=2,
                            unique_inputs=2, WMAPE_percent=error)]))
    rows.append(dict(suite_id='s04', region_index=0, family='ICNN', status='MISSING',
                     model_sha256=None, test_opened=False, test_rows=0, pairs=1,
                     TEST_WMAPE_percent=np.nan))
    return pd.DataFrame(rows), dict(registry_sha256='frozen', test_results_used_for_selection=False, results=results)


def test_unequal_denominators_preserve_pooled_error_and_unavailable_rows():
    frame, summary = fixture()
    original = frame.copy(deep=True)
    enriched = enrich_test_aggregates(frame, summary, registry_sha256='frozen')
    assert len(enriched) == 3
    assert enriched.TEST_absolute_error_sum.sum() / enriched.TEST_true_cost_sum.sum() * 100 == pytest.approx(89.11881188118812)
    assert frame.TEST_WMAPE_percent.median() == 45.5
    assert pd.isna(enriched.loc[enriched.suite_id.eq('s04'), 'TEST_true_cost_sum']).all()
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize('mutation', ['registry', 'selection', 'model', 'count', 'wmape', 'negative', 'duplicate'])
def test_reject_inconsistent_sealed_aggregate(mutation):
    frame, summary = fixture()
    model = summary['results'][0]['models'][0]
    if mutation == 'registry': summary['registry_sha256'] = 'wrong'
    elif mutation == 'selection': summary['test_results_used_for_selection'] = True
    elif mutation == 'model': model['model_sha256'] = 'wrong'
    elif mutation == 'count': model['coordinates'] = 3
    elif mutation == 'wmape': model['WMAPE_percent'] = 0
    elif mutation == 'negative': model['absolute_error_sum'] = -1
    else: summary['results'].append(copy.deepcopy(summary['results'][0]))
    with pytest.raises(ValueError):
        enrich_test_aggregates(frame, summary, registry_sha256='frozen')


def test_zero_denominator_remains_undefined_without_epsilon():
    frame, summary = fixture()
    model = summary['results'][0]['models'][0]
    model.update(absolute_error_sum=0, true_cost_sum=0, WMAPE_percent=None)
    frame.loc[0, 'TEST_WMAPE_percent'] = np.nan
    result = enrich_test_aggregates(frame, summary, registry_sha256='frozen')
    assert result.loc[0, 'TEST_true_cost_sum'] == 0
    assert pd.isna(result.loc[0, 'TEST_WMAPE_percent'])
