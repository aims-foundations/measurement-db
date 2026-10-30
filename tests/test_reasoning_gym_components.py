"""Native position checks remain distinct from format bonuses and whole answers."""

from benchmarks.reasoning_gym.component_checks import comparison_outcomes, reference_checks


def test_cryptarithm_keeps_wrong_mappings_but_does_not_guess_skipped_checks():
    assert comparison_outcomes('cryptarithm', 'A=1,B=7', 'A=1,B=2') == {'mapping:A': 1., 'mapping:B': 0.}
    assert comparison_outcomes('cryptarithm', 'A=1', 'A=1,B=2') is None
    assert comparison_outcomes('cryptarithm', 'A=1,A=2', 'A=1,B=2') is None


def test_missing_game_cells_follow_native_false_verdicts():
    assert comparison_outcomes('game_of_life', '[[1]]', '[[1,0],[0,1]]') == {
        'cell:0:0': 1., 'cell:0:1': 0., 'cell:1:0': 0., 'cell:1:1': 0.}
    assert comparison_outcomes('game_of_life', 'not json', '[[1]]') is None


def test_grid_parser_skips_non_rows_without_shifting_native_coordinates():
    reference = '1 2\n2 1'
    assert comparison_outcomes('mini_sudoku', 'Solution:\n1 2\ninvalid 9\n1 2', reference) == {
        'cell:0:0': 1., 'cell:0:1': 1., 'cell:1:0': 0., 'cell:1:1': 0.}


def test_word_metrics_have_different_native_length_rules():
    assert comparison_outcomes('letter_jumble', 'APPLE', 'apple pear') == {'word:0': 1., 'word:1': 0.}
    assert comparison_outcomes('sentence_reordering', 'APPLE', 'apple pear') is None
    assert comparison_outcomes('sentence_reordering', 'APPLE PEAR', 'apple pear') == {'word:0': 1., 'word:1': 1.}


def test_character_metrics_preserve_spaces_and_normalization_rules():
    assert comparison_outcomes('figlet_font', 'C A B', 'cat') == {'character:0': 1., 'character:1': 1., 'character:2': 0.}
    assert comparison_outcomes('emoji_mystery', 'C A T', 'CAT') is None
    assert len(reference_checks('emoji_mystery', 'A B')) == 3


def test_unknown_and_scored_completions_share_the_original_item(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    import pandas as pd
    import yaml
    from scripts.build_measurement_tables import reload
    from scripts.curate_benchmarks.testlet_reasoning_audit import verify

    build_file = Path(__file__).resolve().parents[1] / 'benchmarks/reasoning_gym/build.py'
    spec = importlib.util.spec_from_file_location('reasoning_fixture', build_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / 'reasoning_gym'
    directory.mkdir()
    metadata = yaml.safe_load(build_file.with_name('metadata.yaml').read_text())
    (directory / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    raw = directory / 'raw'
    root = raw / metadata['build']['parameters']['layout']['results']
    summary = dict(model='gpt-4o', provider='OpenAI', max_tokens=128, temperature=0., top_p=1.,
        completions_per_prompt=2, git_hash='fixture', system_prompt='Solve the question')
    excluded = root / metadata['build']['parameters']['exclusions']['ambiguous_identity_run']
    excluded.mkdir(parents=True)
    (excluded / 'summary.json').write_text(json.dumps(dict(summary, model='openai/o3-mini')))
    folder = root / 'easy/gpt-4o_20250413_235101'
    (folder / 'arithmetic').mkdir(parents=True)
    (folder / 'summary.json').write_text(json.dumps(summary))
    failed = dict(model_answer='4', full_model_response='Original output', score=0., error='Native grading error')
    succeeded = dict(model_answer='4', full_model_response='Original output', score=1.)
    record = dict(question='Two plus two?', expected_answer='4', mean_score=.5, best_score=1., completions=[failed, succeeded])
    (folder / 'arithmetic/basic_arithmetic.json').write_text(json.dumps(dict(name='basic_arithmetic', category='arithmetic',
        system_prompt=summary['system_prompt'], completions_per_prompt=2, config={}, total_examples=1, results=[record])))
    output = tmp_path / 'output'
    reload()
    try:
        module.ReasoningGym(str(directory / 'build.py')).main_from_args(['--source', str(raw), '--output', str(output)])
        tables = {p.stem: pd.read_parquet(p) for p in output.glob('*.parquet')}
        stats = verify(directory, tables, metadata)
        assert stats['source_attempts'] == 2 and stats['imported_items'] == 1
        assert sorted(tables['responses'].trial) == [1, 2]
        assert tables['responses'].response.isna().sum() == 1
        assert not tables['items'].raw_item_id.str.contains('ungraded_task_outcome').any()
    finally:
        reload()


def test_polynomial_roots_keep_continuous_rewards_and_missing_extra_counts():
    import math
    from benchmarks.reasoning_gym.component_checks import root_outcomes
    config = dict(exp_distance_factor=-10., penalty_missing_factor=.5, penalty_extra_factor=.5)
    observed = root_outcomes('1.01, -1, 9', '-1, 1', config)
    assert observed['root:0'] == 1.
    assert math.isclose(observed['root:1'], math.exp(-.1))
    assert observed['extra_roots'] == 1.
    assert root_outcomes('not numeric, 1', '-1, 1', config) == {'root:0': 0., 'root:1': 1., 'extra_roots': 0.}
    assert root_outcomes('', '', config) == {'extra_roots': 0.}
    assert root_outcomes('2', '2, 2', config) == {'root:0': 1., 'root:1': 0., 'extra_roots': 0.}
