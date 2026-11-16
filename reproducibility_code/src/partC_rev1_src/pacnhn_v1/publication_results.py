"""Scientific prose from already validated publication CSV frames only."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

FAMILIES = ('FFTT', 'POWER4', 'XGBOOST', 'ICNN')
LABELS = {'FFTT': 'FFTT', 'POWER4': 'power-4', 'XGBOOST': 'XGBoost', 'ICNN': 'ICNN'}
PHYSICAL = {'P', 'P-ACNHN', 'P_ACNHN', 'PACNHN'}


def number(value):
    try:
        n = float(value)
    except (ValueError, TypeError):
        return None
    return n if math.isfinite(n) else None


def true(value):
    return str(value).strip().lower() in {'true', '1', '1.0'}


def is_measured(row):
    return (true(row.get('certificate_fresh', False))
            and true(row.get('served_OD_membership_equal', False))
            and number(row.get('wmape_backbone_percent')) is not None
            and number(row.get('wmape_key_percent')) is not None)


def f(value):
    return format(float(value), '.5g')


def scientific_results(methods: pd.DataFrame, i95: pd.DataFrame):
    """Keep reference-self, controls and candidate families separate.

    This summarizes existing approved flow-error and link-change quantities;
    it performs no assignment, model evaluation, input discovery or new KPI.
    Caller remains responsible for full native CSV validation and provenance.
    """
    required = {'case_id', 'method_id', 'network', 'suite_id', 'scenario',
                'assignment_role', 'status', 'wmape_backbone_percent',
                'wmape_key_percent', 'certificate_fresh',
                'served_OD_membership_equal', 'elapsed_seconds'}
    if not required <= set(methods.columns):
        raise ValueError('Native method/result columns required')
    if methods.case_id.duplicated().any() or methods.duplicated(['method_id', 'scenario']).any():
        raise ValueError('Duplicate scientific case or method/scenario')
    paragraphs = {'assignments': [], 'i95': [], 'timing': []}
    measured_by_scenario = {}
    all_primary = []
    for scenario, label, section in [('OPEN', 'open network', 'assignments'),
            ('REPRESENTED_I95', 'represented I-95 closure', 'i95')]:
        rows = methods.loc[methods.scenario.eq(scenario)]
        physical = rows.network.astype(str).str.upper().isin(PHYSICAL)
        controls = rows.loc[physical & ~rows.assignment_role.eq('REFERENCE_SELF')]
        # Only the three independent physical controls form a control range.
        controls = controls.loc[controls.method_id.isin(['P/CFW', 'P/FW', 'P/MSA'])]
        candidates = rows.loc[rows.suite_id.eq('s03') & rows.method_id.isin(['s03/'+x for x in FAMILIES])]
        observed = {}; descriptions = []
        for family in FAMILIES:
            selected = candidates.loc[candidates.method_id.eq('s03/'+family)]
            if selected.empty:
                descriptions.append(LABELS[family]+': unavailable')
                continue
            row = selected.iloc[0]
            all_primary.append(row)
            if not is_measured(row):
                descriptions.append(LABELS[family]+': '+str(row.status)+' (no admitted flow-error comparison)')
                continue
            b, k = number(row.wmape_backbone_percent), number(row.wmape_key_percent)
            observed[family] = dict(backbone=b, key=k, status=str(row.status))
            descriptions.append(f'{LABELS[family]}: {f(b)}% / {f(k)}% ({row.status})')
        measured_by_scenario[scenario] = observed
        paragraphs[section].append('For the '+label+', the s03 full-backbone / fixed-200-link WMAPE values were '
                                  +'; '.join(descriptions)+'.')
        control_values = []
        for _, row in controls.iterrows():
            all_primary.append(row)
            if is_measured(row):
                control_values.append((number(row.wmape_backbone_percent), number(row.wmape_key_percent)))
        if control_values:
            b, k = zip(*control_values)
            paragraphs[section].append(
                f'The {len(control_values)} independently solved P controls in this scenario had full-backbone '
                f'WMAPE from {f(min(b))}% to {f(max(b))}% and fixed-key WMAPE from {f(min(k))}% to {f(max(k))}%. '
                'These differences measure numerical agreement with the frozen P reference, which used a tighter nominal stopping tolerance; its self-comparison is excluded.')
        if observed:
            best_b = min(row['backbone'] for row in observed.values())
            best_k = min(row['key'] for row in observed.values())
            names_b = ', '.join(LABELS[key] for key, row in observed.items() if row['backbone'] == best_b)
            names_k = ', '.join(LABELS[key] for key, row in observed.items() if row['key'] == best_k)
            paragraphs[section].append(
                f'Among the {len(observed)} reported s03 surrogate states, {names_b} had the smallest full-backbone error '
                f'and {names_k} the smallest fixed-key error. This ranking is conditional on the common backbone, '
                'represented route support and each reported solver state.')
    changes = []
    for family in FAMILIES:
        open_row = measured_by_scenario['OPEN'].get(family)
        closed_row = measured_by_scenario['REPRESENTED_I95'].get(family)
        if open_row and closed_row:
            delta = closed_row['backbone']-open_row['backbone']
            changes.append(f'{LABELS[family]} {delta:+.5g} percentage points')
    if changes:
        paragraphs['i95'].append('From open to closed, full-backbone WMAPE changed by '+
            '; '.join(changes)+'. Each error uses its own scenario-specific P reference and the unchanged backbone population.')
    if len(i95):
        if not {'method_id', 'link_id', 'delta'} <= set(i95.columns):
            raise ValueError('Native I95 link-change columns required')
        reference = i95.loc[i95.method_id.eq('P/REFERENCE')]
        if len(reference):
            if reference.link_id.duplicated().any():
                raise ValueError('Duplicate P-reference I95 native link')
            delta = pd.to_numeric(reference.delta, errors='raise').to_numpy(dtype=np.float64)
            if not np.isfinite(delta).all():
                raise ValueError('Nonfinite P-reference I95 changes')
            gain = float(np.maximum(delta, 0).sum())
            loss = float(-np.minimum(delta, 0).sum())
            if not math.isfinite(gain) or not math.isfinite(loss):
                raise ValueError('Overflow in P-reference link changes')
            paragraphs['i95'].append(
                f'For the P benchmark, positive backbone traversal changes summed to {f(gain)} annual thousand tons '
                f'and negative changes had magnitude {f(loss)} annual thousand tons. These sums count link traversals; '
                'they are distinct from the newly unserved OD demand and combine rerouting with the change in served demand.')
    timings = [number(r.elapsed_seconds) for r in all_primary if str(r.status) == 'CONVERGED']
    timings = [x for x in timings if x is not None]
    if timings:
        paragraphs['timing'].append(
            f'The {len(timings)} converged primary comparison queries took between {f(min(timings))} and '
            f'{f(max(timings))} seconds under the recorded complete-query boundary. These are measured query times; '
            'the tighter, path-tracing reference runs and historical full-physical timings have different duties.')
    return paragraphs
