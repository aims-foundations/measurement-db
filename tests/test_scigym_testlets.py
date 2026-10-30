"""Reaction testlets preserve native set semantics and one physical execution."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.curate_benchmarks.testlet_audits import verify_component_results


def _xml(reactions):
    bodies = []
    for index, (source, target, modifier) in enumerate(reactions):
        bodies.append(f'<reaction id="r{index}"><listOfReactants><speciesReference species="{source}"/></listOfReactants>'
            f'<listOfProducts><speciesReference species="{target}"/></listOfProducts>'
            f'<listOfModifiers><modifierSpeciesReference species="{modifier}"/></listOfModifiers></reaction>')
    return '<sbml><model><listOfReactions>' + ''.join(bodies) + '</listOfReactions></model></sbml>'


@pytest.fixture
def release(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'benchmarks/scigym/build.py'
    spec = importlib.util.spec_from_file_location('scigym_fixture', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / 'scigym'
    raw = directory / 'raw'
    raw.mkdir(parents=True)
    metadata = yaml.safe_load(source.with_name('metadata.yaml').read_text())
    (directory / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    initial = [('A', 'B', 'M')]
    partial = _xml(initial)
    truth = _xml(initial + [('B', 'C', 'M'), ('C', 'D', 'N')])
    predicted = _xml(initial + [('B', 'C', 'other'), ('X', 'Y', 'N'), ('X', 'Y', 'N')])
    prompt = 'Original task\n## Max iterations\n20\n' + partial + metadata['build']['parameters']['transcript']['initial_prompt_end']
    record = dict(biomodel_id='example', model_name='gpt-4o', chat_history=prompt + '\nComplete answer', final_model=predicted)
    pd.DataFrame([record, record, record]).to_parquet(raw / 'results.parquet', index=False)
    pd.DataFrame([dict(folder_name='example', truth_xml=truth, partial=partial, truth_sedml='original simulation')]).to_parquet(raw / 'tasks.parquet', index=False)
    from scripts.build_measurement_tables import reload
    reload()
    try:
        output = tmp_path / 'converted'
        module.SciGym(str(directory / 'build.py')).main_from_args(['--source', str(raw), '--output', str(output)])
        yield directory, output, {p.stem: pd.read_parquet(p) for p in output.glob('*.parquet')}
    finally:
        reload()


def test_reactions_counts_and_all_original_metrics(release):
    directory, output, tables = release
    stats = verify_component_results(directory, output, tables=tables)
    assert stats['source_attempts'] == 1
    assert stats['source_duplicate_copies'] == 2
    assert stats['source_component_observations'] == 7
    cells = tables['responses'].merge(tables['items'][['item_id', 'raw_item_id']], on='item_id')
    assert sorted(cells.loc[cells.raw_item_id.str.contains('with_modifiers:reaction:'), 'response']) == [0., 0.]
    assert sorted(cells.loc[cells.raw_item_id.str.contains('without_modifiers:reaction:'), 'response']) == [0., 1.]
    assert cells.loc[cells.raw_item_id.str.endswith('with_modifiers:extra'), 'response'].item() == 2.
    assert cells.loc[cells.raw_item_id.str.endswith('without_modifiers:extra'), 'response'].item() == 1.
    assert cells.loc[cells.raw_item_id.str.endswith('::ste'), 'response'].isna().all()


@pytest.mark.parametrize('damage', ['missing_reaction', 'invented_grade', 'changed_answer'])
def test_corrupted_components_or_original_answer_fail(release, damage):
    directory, output, tables = release
    if damage == 'missing_reaction':
        tables['responses'] = tables['responses'].iloc[1:].copy()
    elif damage == 'invented_grade':
        tables['responses'].loc[tables['responses'].response.isna(), 'response'] = 0.
    else:
        value = json.loads(tables['attempts'].loc[0, 'trace'])
        trace = json.loads(value['source_trace'])
        trace['chat_history'] += 'changed'
        value['source_trace'] = json.dumps(trace)
        tables['attempts'].loc[0, 'trace'] = json.dumps(value)
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        verify_component_results(directory, output, tables=tables)
