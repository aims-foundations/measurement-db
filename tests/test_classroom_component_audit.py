"""Readability indices retain exact values, shared executions and repeated questions."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.curate_benchmarks.testlet_classroom_audit import verify


@pytest.fixture
def release(tmp_path):
    root = Path(__file__).resolve().parents[1]
    build_file = root / 'benchmarks/classroom_ai/build.py'
    spec = importlib.util.spec_from_file_location('classroom_fixture', build_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / 'classroom_ai'
    raw = directory / 'raw'
    raw.mkdir(parents=True)
    metadata = yaml.safe_load(build_file.with_name('metadata.yaml').read_text())
    (directory / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    metrics = metadata['grading']['verifiers']['recorded_metric']['metrics']
    values = [14.139922907488991, 110.85336363636364, 8.2, -1.9672000000000018, 5., -2.8, 7.5, 6.]
    labels = {metric: "'8'" for metric in metrics}
    labels.update(Flesch="['8', '9']", **{'Dale Chall': "['7', '8']"})
    metric_text = ''.join(f'{metric}: score: {value!r}, '
        f'{"grade_levels" if metric in {"Flesch", "Dale Chall"} else "grade_level"}: {labels[metric]}\n'
        for metric, value in zip(metrics, values))
    native, blocks = {}, []
    questions = ['Explain rain.', 'Explain clouds.', 'Explain rain.']
    for position, question in enumerate(questions):
        answer = 'A short answer.' if position == 1 else 'Complete explanation ' * 15
        block = f'Q: {question}\nModel Response: {answer}\n\nMetric Analysis\n{metric_text}'
        blocks.append(f'{position}th question\n' + block)
        native['gpt4o_mini.txt', position] = dict(question=question, answer=answer,
            metrics=metric_text, block=block, printed_number=position,
            subject='gpt4o_mini', grade=None if position == 1 else 4.)
    (raw / 'gpt4o_mini.txt').write_text(''.join(blocks))
    source = dict(native=native, question_order=questions, numbering_restarts=0)
    from scripts.build_measurement_tables import reload
    reload()
    try:
        output = tmp_path / 'converted'
        module.ClassroomAI(str(directory / 'build.py')).main_from_args(['--source', str(raw), '--output', str(output)])
        tables = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}
        yield directory, tables, metadata, source
    finally:
        reload()


def test_exact_scores_and_short_answer_aggregate_policy(release):
    stats = verify(*release)
    assert stats['source_attempts'] == 3
    assert stats['source_component_observations'] == 24
    assert stats['imported_items'] == 16
    assert stats['source_observed_testlets'] == 2
    assert stats['source_ungraded'] == 1
    assert stats['source_repeated_attempts'] == 1
    responses = release[1]['responses']
    assert responses.response.isna().sum() == 0
    assert responses.response.min() < 0
    assert responses.response.max() > 100


@pytest.mark.parametrize('damage', ['grade', 'answer', 'aggregate', 'testlet', 'metric', 'trial'])
def test_corrupt_component_or_execution_is_rejected(release, damage):
    directory, tables, metadata, source = release
    if damage == 'grade':
        tables['responses'].loc[0, 'response'] += 0.001
    elif damage == 'trial':
        tables['responses'].loc[0, 'trial'] += 1
    elif damage == 'testlet':
        tables['items'].loc[0, 'testlet_id'] = 'different question'
    elif damage == 'metric':
        tables['items'].loc[0, 'raw_item_id'] += 'altered'
    else:
        trace = json.loads(tables['attempts'].loc[0, 'trace'])
        if damage == 'aggregate':
            trace['recorded_task_score'] = 6.
        else:
            record = json.loads(trace['source_trace'])
            record['model_answer'] = 'Changed output'
            trace['source_trace'] = json.dumps(record)
        tables['attempts'].loc[0, 'trace'] = json.dumps(trace)
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        verify(directory, tables, metadata, source)
