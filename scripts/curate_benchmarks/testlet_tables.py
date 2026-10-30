"""Link explicitly identified component grades to their original task and execution.

Source-specific code supplies the component definitions and outcomes. This module
does not infer checks from aggregate scores, treat missing checks as failures, or
decide how a benchmark grades a component.
"""

import json

import pandas as pd


def component_tables(tables, components, outcomes):
    """Expand tabular builder inputs using source-declared component identities.

    ``components`` contains parent_item_key, component_id, grading_criterion and
    verifier. ``outcomes`` contains parent_response_key, component_id and response.
    Parent items must explicitly supply testlet_id. Existing task grades and full
    traces are retained once per attempt. An execution with no component outcomes
    remains a task-level observation, including an explicitly unavailable grade.
    Optional grading_condition distinguishes recorded repetitions of a grader
    within one execution. It extends test_condition without creating new attempts.
    """
    parents, runs = tables['items'].copy(), tables['responses'].copy()
    if 'attempts' in tables or 'attempt_key' in runs:
        raise ValueError('Component conversion expects one original response per execution')
    if 'testlet_id' not in parents or parents.testlet_id.isna().any():
        raise ValueError('Every parent task needs explicit testlet membership')
    if parents.item_key.duplicated().any() or runs.response_key.duplicated().any():
        raise ValueError('Parent task and execution keys must be unique')
    if not runs.item_key.isin(parents.item_key).all():
        raise ValueError('An execution refers to an unknown parent task')
    required = {'parent_item_key', 'component_id', 'grading_criterion', 'verifier'}
    outcome_columns = {'parent_response_key', 'component_id', 'response'}
    if components.empty and len(components.columns) == 0:
        components = pd.DataFrame(columns=sorted(required))
    if outcomes.empty and len(outcomes.columns) == 0:
        outcomes = pd.DataFrame(columns=sorted(outcome_columns))
    if set(components) != required or not outcome_columns <= set(outcomes) <= outcome_columns | {'grading_condition'}:
        raise ValueError('Unexpected component definition or outcome columns')
    if 'grading_condition' in outcomes and not outcomes.grading_condition.map(lambda v: isinstance(v, str) and bool(v)).all():
        raise ValueError('Grading conditions must be explicit nonempty strings')
    for frame, parent in ((components, 'parent_item_key'), (outcomes, 'parent_response_key')):
        if not frame.component_id.map(lambda v: isinstance(v, str) and bool(v)).all():
            raise ValueError('Component IDs must be nonempty source identifiers')
        identity = [parent, 'component_id'] + (['grading_condition'] if 'grading_condition' in frame else [])
        if frame.duplicated(identity).any():
            raise ValueError('Duplicate source component definition or outcome')
    if not components.parent_item_key.isin(parents.item_key).all():
        raise ValueError('A component refers to an unknown parent task')
    if not outcomes.parent_response_key.isin(runs.response_key).all():
        raise ValueError('A component outcome refers to an unknown execution')

    # Assign repetition numbers before expansion so missing checks cannot shift trials.
    if 'trial' not in runs:
        context = [c for c in ('subject_key', 'item_key', 'test_condition', 'interactors') if c in runs]
        runs['trial'] = runs.groupby(context, sort=False, dropna=False).cumcount() + 1
    runs['attempt_key'] = runs.response_key
    trace = tables.get('traces', pd.DataFrame(columns=['response_key', 'trace']))
    if trace.response_key.duplicated().any() or not trace.response_key.isin(runs.response_key).all():
        raise ValueError('Source traces must identify unique original executions')
    attempts = runs.merge(trace, on='response_key', how='left', validate='one_to_one')
    attempts['raw_attempt_id'] = attempts.response_key.astype(str)
    attempts['trace'] = [json.dumps(dict(recorded_task_score=None if pd.isna(row.response) else row.response,
        source_trace=row.trace if isinstance(row.trace, str) else None), ensure_ascii=False, allow_nan=False)
        for row in attempts.itertuples()]

    definitions = components.merge(parents.drop(columns=['grading_criterion', 'verifier']),
        left_on='parent_item_key', right_on='item_key', validate='many_to_one')
    definitions['item_key'] = [json.dumps([key, component], ensure_ascii=False)
        for key, component in zip(definitions.parent_item_key, definitions.component_id)]
    definitions['raw_item_id'] = definitions.raw_item_id.astype(str) + '::' + definitions.component_id.astype(str)
    records = outcomes.merge(runs.drop(columns='response'),
        left_on='parent_response_key', right_on='response_key', validate='many_to_one')
    records = records.rename(columns={'item_key': 'parent_item_key'}).merge(
        definitions[['parent_item_key', 'component_id', 'item_key']],
        on=['parent_item_key', 'component_id'], how='left', validate='many_to_one', indicator=True)
    if not records._merge.eq('both').all():
        raise ValueError('A released component outcome lacks its grading definition')
    if 'grading_condition' in outcomes:
        records['response_key'] = [json.dumps([key, component, condition], ensure_ascii=False)
            for key, component, condition in zip(records.parent_response_key, records.component_id, records.grading_condition)]
        if 'test_condition' not in runs:
            runs['test_condition'] = None
            records['test_condition'] = None
        records['test_condition'] = [str(context) + ';' + condition if pd.notna(context) and context else condition
            for context, condition in zip(records.test_condition, records.grading_condition)]
    else:
        records['response_key'] = [json.dumps([key, component], ensure_ascii=False)
            for key, component in zip(records.parent_response_key, records.component_id)]

    # Retain aggregate-only evidence; never invent which constituent checks passed.
    aggregate_only = runs.loc[~runs.response_key.isin(outcomes.parent_response_key)]
    retained = parents.loc[parents.item_key.isin(aggregate_only.item_key)]
    return {
        'subjects': tables['subjects'],
        # An empty merge can turn integer keys into floats. Keep the original
        # key types when the source supplies only aggregate observations.
        'items': retained.copy() if definitions.empty else pd.concat([definitions[list(parents)], retained], ignore_index=True),
        'responses': aggregate_only.copy() if records.empty else pd.concat([records[list(runs)], aggregate_only], ignore_index=True),
        'attempts': attempts[['attempt_key', 'subject_key', 'raw_attempt_id', 'trace']],
    }
