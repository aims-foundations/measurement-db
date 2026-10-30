"""Component imports preserve execution identity, missingness and source evidence."""

import copy
import json

import pandas as pd
import pytest

from scripts.curate_benchmarks.testlet_tables import component_tables


def source():
    tables = dict(
        subjects=pd.DataFrame([dict(subject_key='model', raw_label='model')]),
        items=pd.DataFrame([dict(item_key='task', raw_item_id='task', testlet_id='task-v1',
            content='Complete task', grading_criterion={'rule': 'Whole task'}, verifier='original',
            features={'difficulty': 'hard'})]),
        responses=pd.DataFrame([
            dict(response_key='first', subject_key='model', item_key='task', response=0.5),
            dict(response_key='second', subject_key='model', item_key='task', response=1.0),
            dict(response_key='ungraded', subject_key='model', item_key='task', response=None),
            dict(response_key='aggregate-only', subject_key='model', item_key='task', response=0.75)]),
        traces=pd.DataFrame([dict(response_key=key, trace=f'Full answer {key}')
            for key in ['first', 'second', 'ungraded', 'aggregate-only']]))
    components = pd.DataFrame([dict(parent_item_key='task', component_id=key,
        grading_criterion={'rule': 'Check ' + key}, verifier='verifier-' + key) for key in ['a', 'b']])
    outcomes = pd.DataFrame([
        dict(parent_response_key='first', component_id='a', response=0.0),
        dict(parent_response_key='first', component_id='b', response=None),
        dict(parent_response_key='second', component_id='b', response=1.0)])
    return tables, components, outcomes


def test_source_outcomes_and_missingness_survive_without_inventing_checks():
    tables, definitions, outcomes = source()
    before = copy.deepcopy(tables)
    result = component_tables(tables, definitions, outcomes)
    responses = result['responses'].set_index('response_key')
    assert len(responses) == 5
    assert responses.loc['["first", "a"]', 'response'] == 0
    assert pd.isna(responses.loc['["first", "b"]', 'response'])
    assert '["second", "a"]' not in responses.index
    assert responses.loc['["second", "b"]', 'trial'] == 2
    assert responses.loc['aggregate-only', 'response'] == .75
    assert responses.loc['aggregate-only', 'trial'] == 4
    assert result['items'].testlet_id.tolist() == ['task-v1'] * 3
    assert result['items'].content.tolist() == ['Complete task'] * 3
    assert result['items'].features.tolist() == [{'difficulty': 'hard'}] * 3
    attempts = result['attempts'].set_index('attempt_key')
    assert len(attempts) == 4  # Includes the execution with no released outcomes.
    assert json.loads(attempts.loc['ungraded', 'trace']) == dict(
        recorded_task_score=None, source_trace='Full answer ungraded')
    assert json.loads(attempts.loc['first', 'trace'])['recorded_task_score'] == .5
    assert set(responses.attempt_key) == {'first', 'second', 'ungraded', 'aggregate-only'}
    assert pd.isna(responses.loc['ungraded', 'response'])
    assert responses.loc['ungraded', 'trial'] == 3
    assert responses.loc['ungraded', 'item_key'] == 'task'
    assert 'traces' not in result  # Full answers are stored once per execution.
    for name in before:
        pd.testing.assert_frame_equal(tables[name], before[name])


def test_release_without_any_component_records_retains_its_evidence():
    tables, _, _ = source()
    result = component_tables(tables, pd.DataFrame(), pd.DataFrame())
    assert len(result['attempts']) == 4
    assert len(result['items']) == 1
    assert result['responses'].response.dropna().tolist() == [.5, 1., .75]
    assert pd.isna(result['responses'].iloc[2].response)
    assert result['responses'].attempt_key.tolist() == ['first', 'second', 'ungraded', 'aggregate-only']


@pytest.mark.parametrize('mutation', ['unknown-task', 'unknown-attempt', 'unknown-check',
    'duplicate-definition', 'duplicate-outcome', 'missing-membership', 'duplicate-trace'])
def test_ambiguous_or_unlinked_evidence_is_rejected(mutation):
    tables, definitions, outcomes = source()
    if mutation == 'unknown-task':
        definitions.loc[0, 'parent_item_key'] = 'absent'
    elif mutation == 'unknown-attempt':
        outcomes.loc[0, 'parent_response_key'] = 'absent'
    elif mutation == 'unknown-check':
        outcomes.loc[0, 'component_id'] = 'absent'
    elif mutation == 'duplicate-definition':
        definitions = pd.concat([definitions, definitions.iloc[:1]])
    elif mutation == 'duplicate-outcome':
        outcomes = pd.concat([outcomes, outcomes.iloc[:1]])
    elif mutation == 'missing-membership':
        tables['items'].loc[0, 'testlet_id'] = None
    else:
        tables['traces'] = pd.concat([tables['traces'], tables['traces'].iloc[:1]])
    with pytest.raises(ValueError):
        component_tables(tables, definitions, outcomes)


def test_grader_repetitions_share_the_execution_and_keep_their_conditions():
    tables, definitions, _ = source()
    outcomes = pd.DataFrame([
        dict(parent_response_key='first', component_id='a', response=grade, grading_condition=f'grader_repeat={i}')
        for i, grade in [(1, 0.), (2, 1.), (3, 0.)]])
    result = component_tables(tables, definitions, outcomes)
    rows = result['responses'].loc[lambda frame: frame.attempt_key.eq('first')]
    assert rows.response.tolist() == [0., 1., 0.]
    assert rows.test_condition.tolist() == ['grader_repeat=1', 'grader_repeat=2', 'grader_repeat=3']
    assert rows.trial.tolist() == [1, 1, 1]
    assert len(result['attempts']) == 4
    assert rows.item_key.nunique() == 1 and rows.response_key.nunique() == 3
