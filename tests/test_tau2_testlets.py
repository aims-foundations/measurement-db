"""Task context, trial order and ungraded simulations survive tau2 expansion."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.curate_benchmarks.testlet_audits import verify_component_results


@pytest.fixture
def release(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'benchmarks/tau2_bench/build.py'
    spec = importlib.util.spec_from_file_location('tau2_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / 'tau2_bench'
    raw = directory / 'raw'
    (raw / 'repo_results').mkdir(parents=True)
    metadata = yaml.safe_load(path.with_name('metadata.yaml').read_text())
    (directory / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    info = dict(environment_info=dict(domain_name='airline', policy='Original policy', tools=['lookup']),
        agent_info=dict(llm='Recorded-Model', implementation='recorded-agent'),
        user_info=dict(llm='Recorded-User', global_simulation_guidelines='Original user guidelines'),
        git_commit='recorded-revision', seed=73)
    criteria = dict(reward_basis=['DB', 'NL_ASSERTION'], actions=[], nl_assertions=['Explain the policy'],
        env_assertions=None, communicate_info=None)
    simulations = [dict(id='run-' + str(trial), task_id='task', trial=trial,
        messages=[dict(role='assistant', content='Complete answer ' + str(trial))],
        reward_info=None if trial == 2 else dict(reward=int(trial == 0), reward_basis=['DB', 'NL_ASSERTION'],
            db_check=dict(db_match=trial == 0), nl_assertions=[dict(nl_assertion='Explain the policy', met=True)]))
        for trial in range(3)]
    run = dict(timestamp='2025-01-01', info=info,
        tasks=[dict(id='task', user_scenario='Original task', evaluation_criteria=criteria)], simulations=simulations)
    (raw / 'repo_results/fixture.json').write_text(json.dumps(run))
    from scripts.build_measurement_tables import reload
    reload()
    try:
        output = tmp_path / 'converted'
        module.Tau2Bench(str(directory / 'build.py')).main_from_args(['--source', str(raw), '--output', str(output)])
        tables = {p.stem: pd.read_parquet(p) for p in output.glob('*.parquet')}
        yield directory, output, tables
    finally:
        reload()


def test_full_context_and_unknown_outcome_remain_linked(release):
    directory, output, tables = release
    stats = verify_component_results(directory, output, tables=tables)
    assert stats['source_attempts'] == 3
    assert stats['source_component_observations'] == 5
    assert stats['source_ungraded'] == 1
    assert tables['responses'].trial.tolist().count(3) == 1
    assert tables['responses'].response.isna().sum() == 1
    assert set(tables['responses'].attempt_id) == set(tables['attempts'].attempt_id)


@pytest.mark.parametrize('damage', ['task', 'criterion', 'trace', 'trial', 'unknown_grade'])
def test_lost_context_or_invented_results_fail_the_audit(release, damage):
    directory, output, tables = release
    if damage == 'task':
        tables['items'].loc[0, 'content'] = 'Changed task'
    elif damage == 'criterion':
        verifier = json.loads(tables['items'].loc[0, 'verifier'])
        spec = json.loads(verifier['spec'])
        spec['evaluation_criteria']['actions'] = [{'name': 'invented_action'}]
        verifier['spec'] = json.dumps(spec)
        tables['items'].loc[0, 'verifier'] = json.dumps(verifier)
    elif damage == 'trace':
        record = json.loads(tables['attempts'].loc[0, 'trace'])
        trace = json.loads(record['source_trace'])
        trace['simulation']['messages'] = []
        record['source_trace'] = json.dumps(trace)
        tables['attempts'].loc[0, 'trace'] = json.dumps(record)
    elif damage == 'trial':
        tables['responses'].loc[0, 'trial'] = 99
    else:
        tables['responses'].loc[tables['responses'].response.isna(), 'response'] = 0.
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        verify_component_results(directory, output, tables=tables)
