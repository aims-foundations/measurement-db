"""Conditional goal alternatives stay explicit without fabricated verdicts."""

import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest
import yaml

from scripts.curate_benchmarks.testlet_embodied_audit import verify


@pytest.fixture
def release(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'benchmarks/embodied_agent_interface/build.py'
    spec = importlib.util.spec_from_file_location('embodied_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / 'embodied_agent_interface'
    raw = directory / 'raw'
    raw.mkdir(parents=True)
    metadata = yaml.safe_load(path.with_name('metadata.yaml').read_text())
    paths = metadata['build']['parameters']['paths']
    goals = {'1': [[['open', 'drawer'], ['open', 'drawer']], [['clean', 'drawer'], ['open', 'drawer']]],
        '2': [[['not', ['wet', 'floor']]]]}
    for key, data in [('prompts', {'task': 'Original task instruction'}), ('conditions',
            {'task': dict(initial_conditions=[['closed', 'drawer']], goal_conditions=goals)})]:
        destination = raw / paths[key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(data))
    archive = raw / paths['archive']
    archive.parent.mkdir(parents=True)
    with ZipFile(archive, 'w') as bundle:
        for model, output in [('Recorded-Model', 'Full unexecuted model output'), ('Missing-Model', None)]:
            bundle.writestr('helm_output/behavior/goal_interpretation/' + model + '_outputs.json',
                json.dumps([dict(identifier='task', llm_output=output)]))
    (directory / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    from scripts.build_measurement_tables import reload
    reload()
    try:
        output = tmp_path / 'converted'
        module.EmbodiedAgentInterface(str(directory / 'build.py')).main_from_args(['--source', str(raw), '--output', str(output)])
        yield directory, {p.stem: pd.read_parquet(p) for p in output.glob('*.parquet')}, metadata
    finally:
        reload()


def test_alternatives_multiplicity_and_missing_answers(release):
    stats = verify(*release)
    assert stats['source_attempts'] == 2
    assert stats['imported_items'] == 3
    assert stats['source_component_observations'] == 6
    assert stats['source_missing_outputs'] == 1
    rules = [json.loads(json.loads(value)['rule']) for value in release[1]['items'].grading_criterion]
    shared = next(rule for rule in rules if rule['condition'] == ['open', 'drawer'])
    assert shared['positions'] == [dict(alternative=0, position=0), dict(alternative=0, position=1),
        dict(alternative=1, position=1)]


@pytest.mark.parametrize('damage', ['grade', 'alternative', 'multiplicity', 'answer', 'aggregate'])
def test_invented_judgments_or_lost_source_evidence_fail(release, damage):
    directory, tables, metadata = release
    if damage == 'grade':
        tables['responses'].loc[0, 'response'] = 0.
    elif damage in {'alternative', 'multiplicity'}:
        criterion = json.loads(tables['items'].loc[0, 'grading_criterion'])
        rule = json.loads(criterion['rule'])
        rule['goal_alternatives' if damage == 'alternative' else 'positions'] = []
        criterion['rule'] = json.dumps(rule)
        tables['items'].loc[0, 'grading_criterion'] = json.dumps(criterion)
    else:
        record = json.loads(tables['attempts'].loc[0, 'trace'])
        if damage == 'aggregate':
            record['recorded_task_score'] = 1.
        else:
            trace = json.loads(record['source_trace'])
            trace['source_record']['llm_output'] = 'Changed answer'
            record['source_trace'] = json.dumps(trace)
        tables['attempts'].loc[0, 'trace'] = json.dumps(record)
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        verify(directory, tables, metadata)
