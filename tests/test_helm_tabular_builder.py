"""Semantic regression checks shared by the native HELM table importers."""
import gzip
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_base_helm import HelmTabularBuild


@pytest.fixture
def builder(tmp_path):
    builder = object.__new__(HelmTabularBuild)
    builder.raw_dir = tmp_path
    builder.INFO = {'response_scale': {'kind': 'discrete', 'values': [0, 1]}}
    builder.build_parameters = {'helm': {'scenario': 'fixture', 'release': 'v1', 'metric': 'exact_match'}}
    builder.grading = {'rule': 'Match the correct option.', 'verifiers': {'published': {'metric': 'exact_match'}}}
    # Same source item IDs across runs, but a different prompt/reference. The
    # first-model shortcut would corrupt the second run without changing counts.
    for run, reference, grade in [('first', 'left', 1), ('second', 'right', None)]:
        request = {'model': 'model', 'model_deployment': run, 'prompt': f'{run}: A. left B. right',
                   'temperature': 0, 'max_tokens': 100}
        documents = {
            'run_spec': {'name': run, 'adapter_spec': {'model': 'model', 'method': 'multiple_choice_joint', 'max_train_instances': 0}},
            'instances': [{'id': 'same-id', 'input': {'text': 'shared stem'}, 'references': [
                {'output': {'text': reference}, 'tags': ['correct']}], 'split': 'test'}],
            'display_requests': [{'instance_id': 'same-id', 'train_trial_index': 0, 'request': request}],
            'display_predictions': [{'instance_id': 'same-id', 'train_trial_index': 0, 'predicted_text': 'z' * 20000,
                                     'annotations': {'details': ['preserve', 'all']},
                                     'stats': {} if grade is None else {'exact_match': grade}}],
            'per_instance_stats': [{'instance_id': 'same-id', 'train_trial_index': 0, 'stats':
                [{'name': {'name': 'tokens'}, 'mean': 99}] if grade is None else
                [{'name': {'name': 'exact_match'}, 'mean': grade}]}],
        }
        folder = tmp_path / 'runs' / run
        folder.mkdir(parents=True)
        for name, value in documents.items():
            with gzip.open(folder / f'{name}.json.gz', 'wt') as handle:
                json.dump(value, handle)
    return builder


def change(builder, kind, update, run='first'):
    path = builder.raw_dir / 'runs' / run / f'{kind}.json.gz'
    with gzip.open(path, 'rt') as handle:
        value = json.load(handle)
    update(value)
    with gzip.open(path, 'wt') as handle:
        json.dump(value, handle)


def test_each_attempt_keeps_its_own_prompt_reference_settings_and_full_trace(builder):
    tables = builder.build_tables()
    assert len(tables['subjects']) == 2
    assert tables['items'].content.tolist() == ['first: A. left B. right', 'second: A. left B. right']
    assert [row['reference_answer'] for row in tables['items'].grading_criterion] == ['left', 'right']
    assert tables['responses'].response.iloc[0] == 1
    assert tables['responses'].response.isna().sum() == 1
    assert 'trial' not in tables['responses']  # Shared writer resolves repeated canonical items.
    for text in tables['traces'].trace:
        trace = json.loads(text)
        assert trace['prediction']['predicted_text'] == 'z' * 20000
        assert trace['prediction']['annotations'] == {'details': ['preserve', 'all']}
        assert trace['request']['model_deployment'] == trace['source_run']


def test_retained_legacy_exports_do_not_enter_the_selected_native_runs(builder):
    # A source snapshot can also contain a previous curator's run_spec file.
    # It must not be imported alongside the newly selected native run tree.
    legacy = builder.raw_dir / 'legacy/run_spec.json'
    legacy.parent.mkdir()
    legacy.write_text('{"name": "obsolete"}')
    builder.source_files = [str(path.relative_to(builder.raw_dir))
                            for path in builder.raw_dir.rglob('*') if path.is_file()]
    tables = builder.build_tables()
    assert len(tables['responses']) == 2
    assert {json.loads(text)['source_run'] for text in tables['traces'].trace} == {'first', 'second'}


def test_duplicate_metric_records_do_not_multiply_observations(builder):
    change(builder, 'per_instance_stats', lambda records: records.extend(records[:1]))
    with pytest.raises(ValueError, match='not a one-to-one merge'):
        builder.build_tables()


def test_conflicting_grade_exports_fail_before_writing(builder):
    change(builder, 'display_predictions', lambda records: records[0]['stats'].update(exact_match=0))
    with pytest.raises(ValueError, match='grades disagree'):
        builder.build_tables()


def test_fractional_grades_are_not_silently_binarized(builder):
    change(builder, 'display_predictions', lambda records: records[0]['stats'].update(exact_match=0.5))
    change(builder, 'per_instance_stats', lambda records: records[0]['stats'][0].update(mean=0.5))
    with pytest.raises(ValueError, match='response_scale'):
        builder.build_tables()


def test_missing_item_does_not_reuse_a_different_runs_definition(builder):
    change(builder, 'instances', lambda records: records[0].update(id='different-id'))
    with pytest.raises(ValueError, match='item definition'):
        builder.build_tables()


def test_display_export_can_omit_a_metric_without_erasing_its_grade(builder):
    change(builder, 'display_predictions', lambda records: records[0].update(stats={}))
    assert builder.build_tables()['responses'].response.iloc[0] == 1


def test_safety_grades_and_category_metadata_are_not_binary_gold_answers(builder):
    builder.INFO['response_scale'] = {'kind': 'interval', 'min': 0, 'max': 1}
    builder.grading['verifiers']['published'].update(
        **{'class': 'judge', 'reference_kind': 'metadata', 'annotation_key': 'fixture',
           'model_response_start': '<response>', 'model_response_end': '</response>'})
    for folder in (builder.raw_dir / 'runs').iterdir():
        path = folder / 'display_predictions.json.gz'
        with gzip.open(path, 'rt') as handle:
            records = json.load(handle)
        records[0]['annotations'] = {'fixture': {
            'prompt_text': 'Rubric. <response>' + records[0]['predicted_text'] + '</response> Grade the <response>.'}}
        with gzip.open(path, 'wt') as handle:
            json.dump(records, handle)
    change(builder, 'display_predictions', lambda records: records[0]['stats'].update(exact_match=0.625))
    change(builder, 'per_instance_stats', lambda records: records[0]['stats'][0].update(mean=0.625))
    tables = builder.build_tables()
    assert tables['responses'].response.iloc[0] == 0.625
    assert all(c['reference_answer'] is None for c in tables['items'].grading_criterion)
    for verifier in tables['items'].verifier:
        assert verifier.judged_by == 'llm'
        assert json.loads(verifier.spec)['rubric'] == 'Rubric. <response>{{model_response}}</response> Grade the <response>.'
    assert tables['items'].features.iloc[0]['upstream_reference_metadata'][0]['output']['text'] == 'left'
    assert json.loads(tables['traces'].trace.iloc[0])['prediction']['predicted_text'] == 'z' * 20000


def test_all_native_completions_survive_a_request_level_metric(builder):
    builder.build_parameters['helm']['result_file'] = 'scenario_state.json'
    for folder in (builder.raw_dir / 'runs').iterdir():
        with gzip.open(folder / 'scenario_state.json.gz', 'wt') as handle:
            json.dump({'request_states': [{'instance': {'id': 'same-id'}, 'train_trial_index': 0,
                       'result': {'success': True, 'completions': [{'text': 'first'}, {'text': 'second'}]}}]}, handle)
    tables = builder.build_tables()
    assert len(tables['responses']) == 2
    for trace in tables['traces'].trace:
        assert json.loads(trace)['native_result']['completions'] == [{'text': 'first'}, {'text': 'second'}]


def test_all_accepted_references_are_preserved_when_array_encoding_is_declared(builder):
    builder.grading['verifiers']['published'].update(reference_kind='gold', reference_encoding='json_array')
    change(builder, 'instances', lambda rows: rows[0]['references'].extend([
        {'output': {'text': 'right'}, 'tags': ['correct']},
        {'output': {'text': 'left'}, 'tags': ['correct']},
    ]))
    items = builder.build_tables()['items']
    assert json.loads(items.grading_criterion.iloc[0]['reference_answer']) == ['left', 'right', 'left']
    assert json.loads(items.grading_criterion.iloc[1]['reference_answer']) == ['right']


def test_native_task_metrics_preserve_fractional_and_ungraded_attempts(builder):
    builder.INFO['response_scale'] = {'kind': 'mixed'}
    builder.build_parameters['helm'].pop('metric')
    builder.build_parameters['helm']['item_features'] = 'scenario_spec'
    builder.build_parameters['primary_metrics'] = {'CorpusMetric': 'corpus_bleu'}
    builder.grading['verifiers'] = {
        name: {'class': 'exact_matcher', 'reference_kind': 'gold', 'reference_encoding': 'json_array',
               'metric': name, 'criterion': f'Native {name}',
               'response_scale': {'kind': 'interval', 'min': 0, 'max': maximum}}
        for name, maximum in [('sentence_bleu', 1), ('corpus_bleu', 100)]}
    for run, metric_spec in [('first', {'class_name': 'BasicMetric', 'args': {'names': ['sentence_bleu']}}),
                             ('second', {'class_name': 'CorpusMetric', 'args': {}})]:
        change(builder, 'run_spec', lambda spec: spec.update(
            metric_specs=[metric_spec], scenario_spec={'class_name': 'Task', 'args': {'language': run}}), run)
    change(builder, 'display_predictions', lambda rows: rows[0].update(stats={'sentence_bleu': 0.375}))
    change(builder, 'per_instance_stats', lambda rows: rows[0].update(
        stats=[{'name': {'name': 'sentence_bleu'}, 'mean': 0.375}]))
    tables = builder.build_tables()
    assert tables['responses'].response.iloc[0] == 0.375
    assert tables['responses'].response.isna().tolist() == [False, True]
    assert [c['response_scale']['max'] for c in tables['items'].grading_criterion] == [1, 100]
    assert tables['items'].features.iloc[1]['scenario']['args'] == {'language': 'second'}
    assert json.loads(tables['items'].verifier.iloc[1].spec)['metric'] == 'corpus_bleu'
    assert len(tables['traces']) == 2
