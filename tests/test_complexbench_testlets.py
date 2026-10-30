"""Known rubric structure must not turn unavailable judgments into failures."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.curate_benchmarks.testlet_complex_audit import verify


@pytest.fixture
def release(tmp_path):
    root = Path(__file__).resolve().parents[1]
    build_file = root / 'benchmarks/complexbench/build.py'
    spec = importlib.util.spec_from_file_location('complex_fixture', build_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / 'complexbench'
    raw = directory / 'raw'
    metadata = yaml.safe_load(build_file.with_name('metadata.yaml').read_text())
    layout = metadata['build']['parameters']['paths']
    tasks = [dict(main_id=key, instruction='Instruction ' + str(key), task_types='composition', category='language',
        scoring_questions=[dict(point_id=point, question='Criterion ' + str(point),
            rule=None, dep=[0] if point else []) for point in range(key + 2)]) for key in range(2)]
    task_file = raw / layout['tasks']
    task_file.parent.mkdir(parents=True)
    task_file.write_text(json.dumps(tasks))
    records = [dict(main_id=task['main_id'], model='Recorded-Model', instruction=task['instruction'],
        generated='Complete original answer' if task['main_id'] == 0 else None) for task in tasks]
    path = raw / 'release/llm_generations/Recorded-Model.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text('\n'.join(json.dumps(row) for row in records) + '\n')
    (directory / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    from scripts.build_measurement_tables import reload
    reload()
    try:
        output = tmp_path / 'converted'
        module.ComplexBench(str(directory / 'build.py')).main_from_args(['--source', str(raw), '--output', str(output)])
        tables = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}
        yield directory, tables, metadata
    finally:
        reload()


def test_known_points_unknown_grades_and_missing_generation(release):
    stats = verify(*release)
    assert stats['source_attempts'] == 2
    assert stats['source_component_observations'] == 5
    assert stats['source_observed_testlets'] == 2
    assert stats['source_missing_outputs'] == 1
    assert release[1]['responses'].response.isna().all()


@pytest.mark.parametrize('damage', ['grade', 'dependency', 'answer', 'testlet', 'aggregate'])
def test_invented_or_lost_evidence_is_rejected(release, damage):
    directory, tables, metadata = release
    if damage == 'grade':
        tables['responses'].loc[0, 'response'] = 0.
    elif damage == 'dependency':
        criterion = json.loads(tables['items'].loc[0, 'grading_criterion'])
        rule = json.loads(criterion['rule'])
        rule['scoring_question']['dep'] = [999]
        criterion['rule'] = json.dumps(rule)
        tables['items'].loc[0, 'grading_criterion'] = json.dumps(criterion)
    elif damage == 'testlet':
        tables['items'].loc[0, 'testlet_id'] = '999'
    else:
        trace = json.loads(tables['attempts'].loc[0, 'trace'])
        if damage == 'aggregate':
            trace['recorded_task_score'] = 0.
        else:
            record = json.loads(trace['source_trace'])
            record['source_record']['generated'] = 'Changed answer'
            trace['source_trace'] = json.dumps(record)
        tables['attempts'].loc[0, 'trace'] = json.dumps(trace)
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        verify(directory, tables, metadata)
