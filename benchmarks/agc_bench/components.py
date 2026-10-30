"""Recover native Connections checks and keep one execution per generation."""

import json
import re

import pandas as pd

from measurement_db.build_base import ExactMatcher
from measurement_db.scripts.curate_benchmarks.testlet_tables import component_tables


def _token(value):
    return re.sub(r'\s+', ' ', value.strip().lower()).strip('.,;:!?"\'`()[]{}')


def groups(text):
    """The pinned GroupMatchScoreMetric's four-line, four-token normalization."""
    lines = [line.strip() for line in text.splitlines() if line.strip()][:4]
    result = []
    for line in lines + [''] * (4 - len(lines)):
        line = re.sub(r'^\s*group\s*\d+\s*:?\s*', '', line, flags=re.I)
        line = re.sub(r'\.\s*connection\s*:.*$', '', line, flags=re.I)
        tokens = [_token(token) for token in line.split(',')]
        # The native matcher normalizes the parsed tokens a second time.
        result.append(tuple(sorted(_token(token) for token in [token for token in tokens if token][:4]
            if _token(token))))
    return result


def group_outcomes(completion, reference):
    """Consume each reference group once, including duplicate reference groups."""
    targets = groups(reference)
    remaining = list(enumerate(targets))
    outcomes = [0.] * len(targets)
    for prediction in groups(completion):
        for index, (position, target) in enumerate(remaining):
            if prediction == target:
                outcomes[position] = 1.
                remaining.pop(index)
                break
    return outcomes


def convert(tables, observations, protocol):
    selected = observations.protocol.eq('canonical_nyt_connections')
    runs = observations.loc[selected]
    definitions, outcomes = {}, []
    parents = tables['items'].set_index('item_key', drop=False)
    for row in runs.itertuples():
        grades = group_outcomes(row.completion or '', row.reference_answer)
        if pd.isna(row.response) or float(all(grades)) != row.response:
            raise ValueError('Connections checks do not reproduce the released full-wall grade')
        for position, (target, grade) in enumerate(zip(groups(row.reference_answer), grades), 1):
            component = 'group_' + str(position)
            definitions[row.item_key, component] = dict(parent_item_key=row.item_key, component_id=component,
                grading_criterion=dict(reference_answer=json.dumps(target),
                    rule=json.dumps(dict(group_position=position, source_reference=row.reference_answer,
                        rule='Recover this reference group with the native four-line group matcher; each predicted group can match one reference group.'), sort_keys=True),
                    response_scale=dict(kind='discrete', values=[0, 1], direction='higher_is_better')),
                verifier=ExactMatcher(spec=json.dumps(dict(kind='native_group_match', original_protocol=protocol), sort_keys=True)))
            outcomes.append(dict(parent_response_key=row.response_key, component_id=component, response=grade))
    chosen = set(runs.response_key)
    expanded = component_tables(dict(subjects=tables['subjects'],
        items=parents.loc[runs.item_key.unique()].reset_index(drop=True),
        responses=tables['responses'].loc[tables['responses'].response_key.isin(chosen)].assign(trial=1),
        traces=tables['traces'].loc[tables['traces'].response_key.isin(chosen)]),
        pd.DataFrame(definitions.values()), pd.DataFrame(outcomes))

    # Panel judgments are measurements of these same generations, not new runs.
    execution_keys = observations.apply(lambda row: json.dumps([row.model, row.benchmark, row.instance_id]), axis=1)
    links = dict(zip(observations.response_key, execution_keys))
    expanded['responses']['attempt_key'] = expanded['responses'].attempt_key.map(links)
    remaining = tables['responses'].loc[~tables['responses'].response_key.isin(chosen)].copy()
    remaining['attempt_key'] = remaining.response_key.map(links)
    remaining['trial'] = 1
    attempts = []
    for key, group in observations.assign(execution_key=execution_keys).groupby('execution_key', sort=False):
        first = group.iloc[0]
        measurements = [dict(rater=row.rater, grade=None if pd.isna(row.response) else row.response,
            grade_status=row.grade_status, panel_rating=row.rating_record if isinstance(row.rating_record, dict) else None)
            for row in group.itertuples()]
        attempts.append(dict(attempt_key=key, raw_attempt_id=key, subject_key=first.model,
            trace=json.dumps(dict(source_file=first.source_file, source_row=int(first.source_row),
                generation=first.native_record, measurements=measurements), ensure_ascii=False, allow_nan=False)))
    return dict(subjects=tables['subjects'],
        items=pd.concat([tables['items'].loc[~tables['items'].item_key.isin(runs.item_key)], expanded['items']], ignore_index=True),
        responses=pd.concat([remaining, expanded['responses']], ignore_index=True), attempts=pd.DataFrame(attempts))
