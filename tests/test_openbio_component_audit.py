"""Rubric conversion must retain full evidence, weights and missing executions."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.curate_benchmarks.testlet_openbio_audit import verify


@pytest.fixture
def release(tmp_path):
    root = Path(__file__).resolve().parents[1]
    build_file = root / 'benchmarks/openbiorq/build.py'
    spec = importlib.util.spec_from_file_location('openbio_fixture', build_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / 'openbiorq'
    raw = directory / 'raw'
    raw.mkdir(parents=True)
    metadata = yaml.safe_load(build_file.with_name('metadata.yaml').read_text())
    parameters = metadata['build']['parameters']
    parameters['models'] = {'fixture': 'GLM-5.1'}
    parameters['tool_access'] = {'fixture': 'enabled'}
    layout = parameters['layout']
    def jsonl(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    questions = [dict(task_id=key, self_contained_question='Question ' + key,
        gold_answer=dict(reference='Full reference ' + key)) for key in ['scored', 'ungraded']]
    criteria = [dict(id=1, text='First requirement', weight=3), dict(id=2, text='Second requirement', weight=1)]
    jsonl(raw / layout['tasks'], questions)
    jsonl(raw / layout['rubrics'], [dict(task_id=q['task_id'], question=q['self_contained_question'],
        criteria=criteria) for q in questions])
    folder = raw / layout['results'] / 'fixture'
    jsonl(folder / 'predictions.jsonl', [dict(task_id=q['task_id'], model_answer='Original answer ' + q['task_id'],
        trace=[dict(tool='original evidence')]) for q in questions])
    jsonl(folder / 'checklist.jsonl', [dict(task_id='scored', checklist_score=.625,
        verdicts=[dict(id=1, v='partial', why='Partial evidence'), dict(id=2, v='met', why='Complete evidence')]),
        dict(task_id='ungraded', checklist_score=None, verdicts=[])])
    (folder / 'summary.json').write_text(json.dumps(dict(tools=True, n_predictions=2, n_judged=1,
        full_core_657={'solve@0.5_pct': 100})))
    (directory / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    builder = module.OpenBioRQ(str(directory / 'build.py'))
    output = tmp_path / 'converted'
    from scripts.build_measurement_tables import reload
    reload()
    try:
        builder.main_from_args(['--source', str(raw), '--output', str(output)])
        tables = {p.stem: pd.read_parquet(p) for p in output.glob('*.parquet')}
        yield directory, tables, metadata
    finally:
        reload()


def test_native_weights_full_answers_and_ungraded_execution(release):
    stats = verify(*release)
    assert stats['source_attempts'] == 2
    assert stats['source_component_observations'] == 4
    assert stats['source_ungraded_observations'] == 1
    assert stats['imported_items'] == 4


@pytest.mark.parametrize('damage', ['answer', 'rationale', 'aggregate', 'weight', 'question', 'occasion', 'ungraded_execution'])
def test_changed_evidence_is_rejected(release, damage):
    directory, tables, metadata = release
    if damage in {'answer', 'rationale', 'aggregate'}:
        attempts = tables['attempts']
        index = attempts.index[attempts.raw_attempt_id.eq('fixture::scored')][0]
        evidence = json.loads(attempts.loc[index, 'trace'])
        native = json.loads(evidence['source_trace'])
        if damage == 'answer':
            native['prediction']['model_answer'] = 'Truncated answer'
        elif damage == 'rationale':
            native['verdicts'][0]['why'] = 'Changed rationale'
        else:
            evidence['recorded_task_score'] = 0.
        evidence['source_trace'] = json.dumps(native)
        attempts.loc[index, 'trace'] = json.dumps(evidence)
    elif damage == 'weight':
        criterion = json.loads(tables['items'].loc[0, 'grading_criterion'])
        rule = json.loads(criterion['rule'])
        rule['criterion']['weight'] = 100
        criterion['rule'] = json.dumps(rule)
        tables['items'].loc[0, 'grading_criterion'] = json.dumps(criterion)
    elif damage == 'question':
        tables['items'].loc[0, 'content'] = 'Another question'
    elif damage == 'occasion':
        tables['responses'].loc[0, 'trial'] = 2
    else:
        tables['attempts'] = tables['attempts'].loc[~tables['attempts'].raw_attempt_id.eq('fixture::ungraded')]
    with pytest.raises((AssertionError, ValueError, KeyError, RuntimeError)):
        verify(directory, tables, metadata)
