"""Company checkpoints retain full provenance and ungraded run associations."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.curate_benchmarks.testlet_audits import verify_component_results


@pytest.fixture
def release(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'benchmarks/theagentcompany/build.py'
    spec = importlib.util.spec_from_file_location('company_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / 'theagentcompany'
    raw = directory / 'raw'
    metadata = yaml.safe_load(path.with_name('metadata.yaml').read_text())
    parameters = metadata['build']['parameters']
    chosen = list(parameters['models'])[:2]
    parameters['models'] = {key: parameters['models'][key] for key in chosen}
    parameters['harnesses'] = {key: parameters['harnesses'][key] for key in chosen}
    directory.mkdir()
    (directory / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    task = raw / parameters['layout']['tasks'] / 'example'
    task.mkdir(parents=True)
    (task / 'task.md').write_text('Original company task')
    (task / 'evaluator.py').write_text('# Original checkpoint evaluator\n')
    for index, model in enumerate(chosen):
        run = raw / parameters['layout']['runs'] / model
        (run / 'trajectories').mkdir(parents=True)
        (run / 'trajectories/traj_example.json').write_text(json.dumps({'actions': ['complete action ' + str(index)]}))
        if index == 0:
            (run / 'results').mkdir()
            (run / 'results/eval_example.json').write_text(json.dumps(dict(final_score=dict(result=2, total=3),
                checkpoints=[dict(result=1, total=2), dict(result=True, total=1)])))
    from scripts.build_measurement_tables import reload
    reload()
    try:
        output = tmp_path / 'converted'
        module.TheAgentCompany(str(directory / 'build.py')).main_from_args(['--source', str(raw), '--output', str(output)])
        yield directory, output, {p.stem: pd.read_parquet(p) for p in output.glob('*.parquet')}
    finally:
        reload()


def test_partial_credit_and_unknown_run_remain_linked(release):
    directory, output, tables = release
    stats = verify_component_results(directory, output, tables=tables)
    assert stats['source_attempts'] == 2
    assert stats['source_component_observations'] == 3
    assert stats['source_ungraded_observations'] == 1
    assert sorted(tables['responses'].response.dropna()) == [0.5, 1.]
    assert set(tables['attempts'].attempt_id) == set(tables['responses'].attempt_id)


@pytest.mark.parametrize('damage', ['instruction', 'checker', 'trace', 'unknown_grade', 'lost_observation'])
def test_missing_context_or_invented_grade_fails(release, damage):
    directory, output, tables = release
    if damage == 'instruction':
        tables['items'].loc[0, 'content'] = 'Changed task'
    elif damage == 'checker':
        verifier = json.loads(tables['items'].loc[0, 'verifier'])
        spec = json.loads(verifier['spec'])
        spec['grading_code'] = '# Different evaluator'
        verifier['spec'] = json.dumps(spec)
        tables['items'].loc[0, 'verifier'] = json.dumps(verifier)
    elif damage == 'trace':
        evidence = json.loads(tables['attempts'].loc[0, 'trace'])
        native = json.loads(evidence['source_trace'])
        native['trajectory'] = '{}'
        evidence['source_trace'] = json.dumps(native)
        tables['attempts'].loc[0, 'trace'] = json.dumps(evidence)
    elif damage == 'unknown_grade':
        tables['responses'].loc[tables['responses'].response.isna(), 'response'] = 0.
    else:
        tables['responses'] = tables['responses'].loc[tables['responses'].response.notna()].copy()
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        verify_component_results(directory, output, tables=tables)


def test_original_execution_view_keeps_valid_trace_links_and_counts(release):
    from scripts.curate_benchmarks.testlet_audit_views import parent_view
    from scripts.build_measurement_tables.validate_measurement_tables import validate_dataset

    _, _, tables = release
    parents = tables['items'].to_dict('records')
    projection = parent_view(tables, parents, {row['item_id']: row['item_id'] for row in parents})
    validate_dataset(projection, context='Original execution audit view')
    assert len(projection['responses']) == 2
    assert projection['benchmarks'].iloc[0].n_responses == 2
    assert projection['traces'].benchmark_id.eq('theagentcompany').all()
    assert projection['traces'].interactors.isna().all()
