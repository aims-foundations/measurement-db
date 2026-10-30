"""Expose saved executions to native audits without losing child associations."""

import json

import pandas as pd


def attempt_view(tables):
    """Return one original aggregate and full trace per observed execution.

    Callers must separately check every component definition and grade. This
    view checks that all children retain the execution's subject and occasion;
    it does not turn aggregate scores into component outcomes.
    """
    responses = tables['responses']
    attempts = tables['attempts']
    occasion = ['subject_id', 'trial', 'test_condition', 'interactors']
    contexts = responses[['attempt_id', *occasion]].drop_duplicates()
    if contexts.attempt_id.isna().any() or contexts.attempt_id.duplicated().any():
        raise ValueError('Component observations disagree about their execution or occasion')
    if attempts.attempt_id.duplicated().any() or set(attempts.attempt_id) != set(contexts.attempt_id):
        raise ValueError('Native audit requires every execution exactly once with observations')
    rows = responses.drop_duplicates('attempt_id').merge(
        attempts[['attempt_id', 'subject_id', 'trace']], on='attempt_id',
        validate='one_to_one', suffixes=('', '_attempt'))
    if not rows.subject_id.eq(rows.subject_id_attempt).all():
        raise ValueError('Component subject differs from the original execution')
    evidence = rows.trace.map(json.loads)
    rows['response'] = evidence.map(lambda record: record['recorded_task_score'])
    rows['trace'] = evidence.map(lambda record: record['source_trace'])
    if not rows.trace.map(lambda value: isinstance(value, str)).all():
        raise ValueError('Missing full native execution trace')
    traces = rows.set_index('response_id').trace.to_dict()
    return rows[list(responses)], traces


def parent_view(tables, parents, item_parents=None, *, execution_parents=None):
    """Project validated child definitions back to the original task audit view."""
    responses, traces = attempt_view(tables)
    if execution_parents is not None:
        responses['item_id'] = responses.attempt_id.map(execution_parents)
    else:
        responses['item_id'] = responses.item_id.map(item_parents)
    if responses.item_id.isna().any():
        raise ValueError('A component has no validated original task')
    responses['attempt_id'] = None
    linked = responses[['response_id', 'subject_id', 'item_id', 'benchmark_id', 'trial', 'test_condition', 'interactors']].copy()
    linked['trace'] = linked.response_id.map(traces)
    items = pd.DataFrame(parents)
    benchmarks = tables['benchmarks'].copy()
    denominator = len(tables['subjects']) * len(items)
    observed = responses.loc[responses.response.notna(), ['subject_id', 'item_id']].drop_duplicates()
    statistics = dict(n_subjects=len(tables['subjects']), n_items=len(items), n_responses=len(responses),
        n_response_values=int(responses.response.nunique(dropna=True)),
        max_trial=int(responses.trial.max()) if len(responses) else 0,
        coverage=len(observed) / denominator if denominator else 0.,
        has_reference_answer=any(json.loads(value).get('reference_answer') is not None for value in items.grading_criterion))
    for name, value in statistics.items():
        benchmarks[name] = value
    return {**{name: frame for name, frame in tables.items() if name != 'attempts'},
        'items': items, 'benchmarks': benchmarks, 'responses': responses, 'traces': linked}
