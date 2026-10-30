"""Recover QATCH's reference-cell and row-multiplicity predicates safely."""

import ast
from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd

from measurement_db.build_base import ExactMatcher
from measurement_db.scripts.curate_benchmarks.testlet_tables import component_tables


def prediction_table(text):
    """Use the native shape rules with literal parsing instead of executing eval.

    Export-only markers such as EQUAL do not contain a returned table. They stay
    aggregate observations; this importer neither runs SQL nor infers cells.
    """
    if not isinstance(text, str) or not text.startswith('['):
        return None
    while text.startswith('['):
        text = text[1:]
    while text and not text.endswith(']'):
        text = text[:-1]
    while text.endswith(']'):
        text = text[:-1]
    if not text:
        return None
    try:
        value = ast.literal_eval('[[' + text + ']]')
        if isinstance(value, tuple):
            value = [row for part in value for row in part]
        array = np.array(value)
        if array.ndim > 2:
            while 1 in array.shape:
                array = np.squeeze(array, axis=list(array.shape).index(1))
        if array.ndim == 0:
            return [[array.tolist()]]
        if array.ndim == 1:
            return [[cell] for cell in array.tolist()]
        if array.ndim == 2:
            return array.tolist()
    except (ValueError, SyntaxError, TypeError):
        return None
    return None


def predicates(target_text, prediction_text, metric, released_grade):
    """Return distinct native predicates only when they reproduce the saved score.

    Cell recall repeats the same membership predicate for equal target values.
    Retain that multiplicity as its weight, not as duplicate test items. Tuple
    constraint already operates on distinct rows and checks their multiplicity.
    """
    if metric not in {'cell_recall', 'tuple_constraint'} or pd.isna(released_grade):
        return []
    prediction = prediction_table(prediction_text)
    if not prediction:
        return []
    target = ast.literal_eval(target_text)
    if not target:
        return []
    target = [[str(cell).replace('\n', '').strip().lower() for cell in row] for row in target]
    prediction = [[str(cell).strip().lower() for cell in row] for row in prediction]
    if metric == 'cell_recall':
        values = Counter(cell for row in target for cell in row)
        actual = {cell for row in prediction for cell in row}
        checks = [(dict(value=value, target_occurrences=count), float(value in actual), count)
            for value, count in values.items()]
    else:
        values = Counter(tuple(row) for row in target)
        actual = Counter(tuple(row) for row in prediction)
        checks = [(dict(row=list(row), target_occurrences=count), float(actual[row] == count), 1)
            for row, count in values.items()]
    # NumPy's rounding is part of the released metric, including half-way cases.
    total = sum(weight for _, _, weight in checks)
    score = float(round(np.sum([int(grade) * weight for _, grade, weight in checks]) / total, 3))
    return checks if score == released_grade else []


def convert(tables, observations, native_attempts, protocols):
    definitions, outcomes = {}, []
    records = native_attempts.set_index(['source_file', 'source_row', 'source_model']).to_dict('index')
    for row in observations.itertuples():
        original = records[row.source_file, row.source_row, row.source_model]
        prediction = original['record'].get('predictions_query_result', original['predictions'])
        for reference, grade, weight in predicates(original['record']['query_result'], prediction, row.metric, row.response):
            text = json.dumps(reference, ensure_ascii=False, sort_keys=True)
            component = hashlib.sha256(text.encode()).hexdigest()
            definitions[row.item_key, component] = dict(parent_item_key=row.item_key, component_id=component,
                grading_criterion=dict(reference_answer=text,
                    rule=json.dumps(dict(metric=row.metric, aggregation_weight=weight,
                        rule='Target-value membership in the normalized returned table.' if row.metric == 'cell_recall'
                            else 'The normalized reference row occurs exactly its target number of times.'), sort_keys=True),
                    response_scale=dict(kind='discrete', values=[0, 1], direction='higher_is_better')),
                verifier=ExactMatcher(spec=json.dumps(dict(kind='reconciled_native_table_predicate',
                    original_protocol=protocols[row.metric]), sort_keys=True)))
            outcomes.append(dict(parent_response_key=row.response_key, component_id=component, response=grade))
    result = component_tables(tables, pd.DataFrame(definitions.values()), pd.DataFrame(outcomes))
    links = {row.response_key: json.dumps([row.source_file, int(row.source_row), row.source_model])
        for row in observations.itertuples()}
    result['responses']['attempt_key'] = result['responses'].attempt_key.map(links)
    measurements = {}
    for row in observations.itertuples():
        measurements.setdefault(links[row.response_key], []).append(dict(metric=row.metric,
            grade=None if pd.isna(row.response) else row.response))
    attempts = []
    for row in native_attempts.itertuples():
        key = json.dumps([row.source_file, int(row.source_row), row.source_model])
        attempts.append(dict(attempt_key=key, raw_attempt_id=key, subject_key=row.subject_key,
            trace=json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
                source_model=row.source_model, record=row.record, measurements=measurements[key]),
                ensure_ascii=False, allow_nan=False)))
    result['attempts'] = pd.DataFrame(attempts)
    return result
