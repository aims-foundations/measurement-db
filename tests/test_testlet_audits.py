"""Independent source expectations must detect lost, moved, and changed outcomes."""

import json

import pandas as pd
import pytest

from scripts.curate_benchmarks import testlet_audits


@pytest.fixture
def release(tmp_path, monkeypatch):
    directory = tmp_path / 'example'
    directory.mkdir()
    (directory / 'metadata.yaml').write_text('build: {}\n')
    tables = directory / 'formatted_tables'
    tables.mkdir()
    items = pd.DataFrame([
        dict(item_id='a', raw_item_id='task::a', testlet_id='task'),
        dict(item_id='b', raw_item_id='task::b', testlet_id='task'),
        dict(item_id='task', raw_item_id='task', testlet_id='task')])
    attempts = pd.DataFrame([
        dict(attempt_id='run', raw_attempt_id='original-run', subject_id='model',
            trace=json.dumps(dict(recorded_task_score=.5, source_trace='Complete source answer'))),
        dict(attempt_id='ungraded', raw_attempt_id='original-ungraded', subject_id='model',
            trace=json.dumps(dict(recorded_task_score=None, source_trace='Error-only attempt')))])
    responses = pd.DataFrame([
        dict(attempt_id='run', subject_id='model', item_id='a', response=1.),
        dict(attempt_id='run', subject_id='model', item_id='b', response=None),
        dict(attempt_id='ungraded', subject_id='model', item_id='task', response=None)])
    for name, frame in dict(items=items, attempts=attempts, responses=responses).items():
        frame.to_parquet(tables / (name + '.parquet'), index=False)
    monkeypatch.setattr(testlet_audits, 'source_attempts', lambda *_: iter([
        ('original-run', .5, [('task::a', 'task', 1.), ('task::b', 'task', None)]),
        ('original-ungraded', None, [('task', 'task', None)])]))
    return directory, tables


def test_every_source_execution_and_null_grade_survives(release):
    directory, _ = release
    assert testlet_audits.verify_component_results(directory) == dict(
        source_attempts=2, source_component_observations=3, imported_items=3, source_observed_testlets=1)


@pytest.mark.parametrize('damage', ['invent-failure', 'lose-check', 'duplicate-check',
    'wrong-subject', 'wrong-testlet', 'lose-ungraded-run', 'lose-ungraded-cell', 'change-aggregate', 'lose-trace'])
def test_corrupt_component_import_is_rejected(release, damage):
    directory, tables = release
    table = ('items' if damage == 'wrong-testlet' else 'attempts'
        if damage in {'lose-ungraded-run', 'change-aggregate', 'lose-trace'} else 'responses')
    path = tables / (table + '.parquet')
    frame = pd.read_parquet(path)
    if damage == 'invent-failure':
        frame.loc[1, 'response'] = 0.
    elif damage == 'lose-check':
        frame = frame.iloc[:1]
    elif damage == 'lose-ungraded-cell':
        frame = frame.iloc[:2]
    elif damage == 'duplicate-check':
        frame = pd.concat([frame, frame.iloc[:1]])
    elif damage == 'wrong-subject':
        frame.loc[0, 'subject_id'] = 'different-model'
    elif damage == 'wrong-testlet':
        frame.loc[0, 'testlet_id'] = 'different-task'
    elif damage == 'lose-ungraded-run':
        frame = frame.iloc[:1]
    elif damage == 'change-aggregate':
        frame.loc[0, 'trace'] = json.dumps(dict(recorded_task_score=1., source_trace='Complete source answer'))
    elif damage == 'lose-trace':
        frame.loc[0, 'trace'] = json.dumps(dict(recorded_task_score=.5, source_trace=None))
    frame.to_parquet(path, index=False)
    with pytest.raises(AssertionError):
        testlet_audits.verify_component_results(directory)
