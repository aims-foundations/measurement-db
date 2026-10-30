"""SAT and witness checks retain native failures, aliases and source occasions."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.curate_benchmarks.testlet_mathconstraint_audit import verify


@pytest.fixture
def release(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'benchmarks/mathconstraint/build.py'
    spec = importlib.util.spec_from_file_location('mathconstraint_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = yaml.safe_load(path.with_name('metadata.yaml').read_text())
    directory = tmp_path / 'mathconstraint'
    raw = directory / 'raw'
    (raw / 'easy_instances').mkdir(parents=True)
    for name, sat in [('sat', True), ('sat_alias', True), ('unsat', False)]:
        problem = dict(name=name, prompt='Find a solution.' if sat else 'Is this impossible?',
            solution=[1] if sat else None, satisfiable=sat)
        (raw / 'easy_instances' / (name + '.json')).write_text(json.dumps(problem))
    records = []
    for name, sat, decision, witness, error in [('sat', True, True, True, None),
            ('sat_alias', True, True, False, None), ('sat', True, False, False, 'parse_failure: empty'),
            ('unsat', False, True, True, None)]:
        records.append(dict(problem_name=name, is_satisfiable=sat, correct=decision and witness,
            satisfiability_correct=decision, solution_correct=witness, error=error,
            cot_trace='Complete reasoning ' * 1000, raw_response='Original submitted response',
            tool_call_chain=[dict(name='execute_python', output='native tool output')],
            validation_details='Correct UNSAT' if not sat else 'Native solver verdict'))
    result = raw / 'results/model/easy/no_tools.json'
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps(dict(model='gpt-4o', condition='no_tools', dataset='easy',
        results=records, metrics={'preserved': True})))
    (directory / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    from scripts.build_measurement_tables import reload
    reload()
    try:
        target = tmp_path / 'converted'
        module.MathConstraintBuild(str(directory / 'build.py')).main_from_args(['--source', str(raw), '--output', str(target)])
        yield directory, {p.stem: pd.read_parquet(p) for p in target.glob('*.parquet')}, metadata
    finally:
        reload()


def test_decision_witness_aliases_and_native_failure(release):
    stats = verify(*release)
    assert stats['source_attempts'] == 4
    assert stats['source_component_observations'] == 7
    assert stats['imported_items'] == 3
    assert stats['source_observed_testlets'] == 2
    assert stats['source_task_aliases'] == 3
    tables = release[1]
    assert tables['attempts'].trace.str.len().min() > 18000
    assert not tables['items'].raw_item_id.eq('unsat::solution_correct').any()
    assert tables['responses'].response.sum() == 4


@pytest.mark.parametrize('damage', ['grade', 'testlet', 'reference', 'trace', 'aggregate', 'occasion', 'raw_attempt', 'subject'])
def test_corrupted_source_details_are_rejected(release, damage):
    directory, tables, metadata = release
    if damage == 'grade':
        tables['responses'].loc[0, 'response'] = 1 - tables['responses'].loc[0, 'response']
    elif damage == 'testlet':
        tables['items'].loc[0, 'testlet_id'] = 'invented'
    elif damage == 'reference':
        criterion = json.loads(tables['items'].loc[0, 'grading_criterion'])
        criterion['reference_answer'] = 'wrong'
        tables['items'].loc[0, 'grading_criterion'] = json.dumps(criterion)
    elif damage in {'trace', 'aggregate'}:
        trace = json.loads(tables['attempts'].loc[0, 'trace'])
        if damage == 'trace':
            source = json.loads(trace['source_trace'])
            source['source_record']['cot_trace'] = 'Truncated.'
            trace['source_trace'] = json.dumps(source)
        else:
            trace['recorded_task_score'] = 1 - trace['recorded_task_score']
        tables['attempts'].loc[0, 'trace'] = json.dumps(trace)
    elif damage == 'occasion':
        tables['responses'].loc[0, 'trial'] = 99
    elif damage == 'raw_attempt':
        tables['attempts'].loc[0, 'raw_attempt_id'] = 'invented'
    else:
        tables['subjects'].loc[0, 'harness'] = 'invented'
    with pytest.raises((AssertionError, ValueError, RuntimeError, KeyError)):
        verify(directory, tables, metadata)
