"""Presentation-only checkpoint chronology; never reads files or evaluates models."""
import math

import pandas as pd


KEY = ['suite_id', 'region_index', 'family', 'candidate_id']
GROUP = KEY[:3]
REQUIRED = KEY + ['block', 'elapsed_seconds', 'train_wmape', 'valid_wmape',
                  'train_dataset_identity', 'valid_dataset_identity',
                  'evaluation_kind']


def _scalar(value):
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, 'item') else value


def select_first_candidate_evaluations(frame):
    """Return (plot_frame, ledger) without changing any input observation.

    Only NEW_CHECKPOINT rows compete. The earliest saved elapsed_seconds for
    each candidate is plotted. Identical ties choose the first source row;
    contradictory same-candidate/time records fail. Different candidates at
    one block remain separate, ordered by time then candidate ID. The new
    ordinal is presentation chronology, never an optimizer/update count.
    """
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError('Missing training-curve fields: '+', '.join(sorted(missing)))
    if not frame.columns.is_unique or {'source_observation_index', 'checkpoint_ordinal'} & set(frame.columns):
        raise ValueError('Ambiguous or reserved training-curve columns')
    raw = frame.copy(deep=True)
    work = raw.reset_index(drop=True).copy()
    work['source_observation_index'] = range(len(work))
    new = work.loc[work.evaluation_kind.eq('NEW_CHECKPOINT')].copy()
    identity_fields = ['suite_id', 'family', 'candidate_id', 'train_dataset_identity', 'valid_dataset_identity']
    if 'audit_path' in new:
        identity_fields.append('audit_path')
    for name in identity_fields:
        if not new[name].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError('Missing candidate identity: '+name)
    for name in ['region_index', 'elapsed_seconds']:
        values = pd.to_numeric(new[name], errors='raise')
        if not values.map(lambda x: math.isfinite(x) and x >= 0).all():
            raise ValueError('Invalid candidate chronology: '+name)
        if name == 'region_index' and not values.map(lambda x: int(x) == x).all():
            raise ValueError('Noninteger region index')
        new[name] = values
    tied = new.loc[new.duplicated(KEY+['elapsed_seconds'], keep=False)]
    for _, group in tied.groupby(KEY+['elapsed_seconds'], dropna=False, sort=False):
        # All original fields, including metrics and audit path, must agree.
        if len(group.drop(columns=['source_observation_index']).drop_duplicates()) != 1:
            raise ValueError('Contradictory same-candidate/time evaluation records')
    ordered = new.sort_values(GROUP+['elapsed_seconds', 'candidate_id', 'source_observation_index'], kind='stable')
    selected = ordered.drop_duplicates(KEY, keep='first').copy()
    selected['checkpoint_ordinal'] = selected.groupby(GROUP, sort=False).cumcount()+1
    chosen = {int(r.source_observation_index): int(r.checkpoint_ordinal)
              for r in selected[['source_observation_index', 'checkpoint_ordinal']].itertuples(index=False)}
    chosen_candidates = {tuple(getattr(r, k) for k in KEY): int(r.source_observation_index)
                         for r in selected[KEY+['source_observation_index']].itertuples(index=False)}
    observations = []
    for row in work.to_dict('records'):
        record = {k: _scalar(v) for k, v in row.items()}
        position = record['source_observation_index']
        is_new = record['evaluation_kind'] == 'NEW_CHECKPOINT'
        record['plotted'] = position in chosen
        record['checkpoint_ordinal'] = chosen.get(position)
        record['selected_source_observation_index'] = chosen_candidates.get(tuple(record[k] for k in KEY)) if is_new else None
        record['selection_reason'] = ('FIRST_RECORDED_CANDIDATE_EVALUATION' if position in chosen
                                      else 'LATER_OR_IDENTICAL_CANDIDATE_EVALUATION' if is_new
                                      else 'NON_NEW_OBSERVATION_RETAINED_ONLY')
        observations.append(record)
    ledger = dict(schema='PACNHN_PUBLICATION_TRAINING_CURVE_SELECTION_V1',
        selection_policy='FIRST_RECORDED_CANDIDATE_EVALUATION',
        counts=dict(raw_observations=len(work), new_checkpoint_observations=len(new),
                    plotted_candidates=len(selected), repeated_new_observations=len(new)-len(selected),
                    retained_non_new_observations=len(work)-len(new)),
        raw_observations_modified=False, observations=observations,
        chronology='elapsed_seconds; candidate ID breaks equal-time ties across distinct candidates',
        x_axis='checkpoint_ordinal: first recorded candidate evaluation within region/family',
        original_block_preserved=True,
        limitations=['This ordinal is not an optimizer-step count.',
                     'Historical TRAIN and admitted VALID identities may both change.',
                     'This is not a fixed-dataset comparison or a training-effect estimate.'])
    return selected.reset_index(drop=True), ledger
