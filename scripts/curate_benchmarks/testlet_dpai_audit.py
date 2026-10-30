"""Retain DPAI's complete native input, configuration and execution audit."""

import json

import pandas as pd


def verify(directory, tables, metadata):
    """Called after the independent component-grade reconciliation."""
    from .native_result_audits import _check, _dpai, _dpai_source_records, _features
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='DPAI complete native component evidence')
    source = _dpai_source_records(directory)
    bank, _ = source
    items = tables['items'].set_index('item_id').to_dict('index')
    parents, parent_ids, item_parents = {}, {}, {}
    for item_id, item in items.items():
        parent, _, component = item['raw_item_id'].partition('::')
        task, phase = parent.rsplit(':', 1)
        native, protocol = bank[task], metadata['grading']['verifiers'][phase]
        _check(item['testlet_id'], task, 'DPAI original task testlet across information conditions')
        visible = {key: native[key] for key in ('FAIL_TO_PASS', 'PASS_TO_PASS')} if phase == 'informed' else None
        _check(json.loads(item['content']), dict(problem_statement=native['problem_statement'],
            repo=native['repo'], base_commit=native['base_commit'], phase=phase, test_information=visible),
            'DPAI every test retains the complete original task and information condition')
        _check(_features(item['item_features']), dict(input_scope=metadata['build']['parameters']['tasks']['scope']),
            'DPAI original input scope on every check')
        _check(pd.isna(item['asset_manifest']), True, 'DPAI no invented task attachments')
        base_rule = metadata['grading']['rule'] + '\n' + json.dumps(dict(task_id=task, phase=phase,
            test_patch=native['test_patch'], FAIL_TO_PASS=native['FAIL_TO_PASS'], PASS_TO_PASS=native['PASS_TO_PASS']), ensure_ascii=False)
        base_criterion = dict(reference_answer=native['patch'], rule=base_rule, response_scale=protocol['response_scale'])
        base_verifier = {'class': 'exact_matcher', 'spec': json.dumps(protocol['implementation'], sort_keys=True)}
        if component:
            stage, test = component.split(':', 1)
            _check(stage in {'all_tests_validation', 'all_tests_after_prediction'}, True, 'DPAI post-patch test stage')
            _check(json.loads(item['grading_criterion']), dict(reference_answer=None,
                rule=json.dumps(dict(test_name=test, evaluation_stage=stage, phase=phase,
                    test_patch=native['test_patch']), ensure_ascii=False, sort_keys=True),
                response_scale=dict(kind='discrete', values=[0, 1], direction='higher_is_better')),
                'DPAI exact test identifier, patch, condition and scale')
            _check(json.loads(item['verifier']), {'class': 'exact_matcher', 'spec': json.dumps(dict(protocol['implementation'],
                field='evaluations.' + stage + '.data.test_result.tests[...].status', test_name=test), sort_keys=True)},
                'DPAI original grading provenance for every named test')
        else:
            _check(json.loads(item['grading_criterion']), base_criterion, 'DPAI retained aggregate-only criterion')
            _check(json.loads(item['verifier']), base_verifier, 'DPAI retained aggregate-only verifier')
        parent_ids.setdefault(parent, item_id)
        parents.setdefault(parent, dict(item, item_id=parent_ids[parent], raw_item_id=parent,
            grading_criterion=json.dumps(base_criterion), verifier=json.dumps(base_verifier)))
        item_parents[item_id] = parent
    _check(set(tables['responses'].item_id), set(items), 'DPAI no invented or orphaned test definitions')

    originals, traces = attempt_view(tables)
    attempts = tables['attempts'].set_index('attempt_id').to_dict('index')
    for row in originals.itertuples():
        trace = json.loads(traces[row.response_id])
        _check(attempts[row.attempt_id]['raw_attempt_id'], ':'.join(trace[key] for key in ('source_file', 'task_id', 'phase')),
            'DPAI original report, task and phase identify the execution')
    originals['item_id'] = originals.item_id.map(lambda key: parent_ids[item_parents[key]])
    projected = {name: frame for name, frame in tables.items() if name != 'attempts'}
    projected.update(items=pd.DataFrame(parents.values()), responses=originals,
        traces=pd.DataFrame(dict(response_id=list(traces), trace=list(traces.values()))))
    return _dpai(directory, projected, metadata, source=source)
