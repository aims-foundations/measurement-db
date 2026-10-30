"""Reference checks must preserve complete search passes and native error counts."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.curate_benchmarks.testlet_autoresearch_audit import verify


@pytest.fixture
def release(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'benchmarks/autoresearchbench/build.py'
    spec = importlib.util.spec_from_file_location('autoresearch_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = yaml.safe_load(path.with_name('metadata.yaml').read_text())
    directory = tmp_path / 'autoresearchbench'
    raw = directory / 'raw'
    output = raw / metadata['build']['parameters']['layout']['release'] / 'output_data'
    output.mkdir(parents=True)

    def attempt(question, number, ids):
        return dict(pass_id=number, messages=[dict(role='system', content='Search for papers.'),
            dict(role='user', content=question), dict(role='assistant', content='Complete answer ' * 1000)],
            final_candidates=[dict(arxiv_id=paper) for paper in ids])

    deep = dict(input_data=dict(question='Find one paper.', answer=['Original title']),
        inference_results=[attempt('Find one paper.', i, []) for i in range(2)])
    deep_evaluation = dict(deep, evaluation=dict(pass_scores=[1, 0], metrics={}))
    wide, judgments = [], []
    for number, (truth, prediction) in enumerate([
            (['2601.00001', '2601.00002'], ['2601.00001', '2601.00003', '2601.00004']), ([], [])]):
        question = 'Find this collection ' + str(number)
        wide.append(dict(input_data=dict(question=question, arxiv_id=truth),
            inference_results=[attempt(question, 0, prediction)]))
        t, p = set(truth), set(prediction)
        judgments.append(dict(line_num=number + 1, question=question, pass_results=[dict(pass_id=0,
            gt_arxiv_ids=truth, predicted_arxiv_ids=prediction, hit_ids=sorted(t & p), missed_ids=sorted(t - p),
            extra_ids=sorted(p - t), gt_count=len(t), predicted_count=len(p), hit_count=len(t & p),
            iou=round(len(t & p) / len(t | p), 6) if t | p else 1)]))
    parameters = metadata['build']['parameters']
    for family, records, evaluation in [('deep', [deep], {'detailed_results': [deep_evaluation]}),
            ('wide', wide, {'per_record_results': judgments})]:
        stem = parameters['runs'][family]
        (output / (stem + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in records))
        (output / (stem + parameters['evaluation_suffixes'][family])).write_text(json.dumps(evaluation))
    (directory / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    from scripts.build_measurement_tables import reload
    reload()
    try:
        target = tmp_path / 'converted'
        module.AutoResearchBench(str(directory / 'build.py')).main_from_args(['--source', str(raw), '--output', str(target)])
        yield directory, {p.stem: pd.read_parquet(p) for p in target.glob('*.parquet')}, metadata
    finally:
        reload()


def test_hits_misses_empty_rosters_and_error_counts(release):
    stats = verify(*release)
    assert stats['source_attempts'] == 4
    assert stats['source_component_observations'] == 6
    assert stats['imported_items'] == 5
    assert stats['source_observed_testlets'] == 3
    tables = release[1]
    assert 2 in tables['responses'].response.values
    assert tables['attempts'].trace.str.len().min() > 16000


@pytest.mark.parametrize('damage', ['grade', 'testlet', 'reference', 'scale', 'answer', 'aggregate', 'occasion', 'raw_attempt'])
def test_corrupted_native_associations_are_rejected(release, damage):
    directory, tables, metadata = release
    item_index = tables['items'].index[tables['items'].raw_item_id.str.contains('::reference:')][0]
    if damage == 'grade':
        row = tables['responses'].index[tables['responses'].item_id.eq(tables['items'].loc[item_index, 'item_id'])][0]
        tables['responses'].loc[row, 'response'] = 1 - tables['responses'].loc[row, 'response']
    elif damage == 'testlet':
        tables['items'].loc[item_index, 'testlet_id'] = 'other-task'
    elif damage in {'reference', 'scale'}:
        criterion = json.loads(tables['items'].loc[item_index, 'grading_criterion'])
        if damage == 'reference':
            criterion['reference_answer'] = '2601.99999'
        else:
            criterion['response_scale']['direction'] = 'lower_is_better'
        tables['items'].loc[item_index, 'grading_criterion'] = json.dumps(criterion)
    elif damage in {'answer', 'aggregate'}:
        trace = json.loads(tables['attempts'].loc[0, 'trace'])
        if damage == 'aggregate':
            trace['recorded_task_score'] = 0.75
        else:
            native = json.loads(trace['source_trace'])
            native['native_pass']['messages'][-1]['content'] = 'Truncated.'
            trace['source_trace'] = json.dumps(native)
        tables['attempts'].loc[0, 'trace'] = json.dumps(trace)
    elif damage == 'occasion':
        tables['responses'].loc[0, 'trial'] = 99
    else:
        tables['attempts'].loc[0, 'raw_attempt_id'] = 'invented'
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        verify(directory, tables, metadata)
