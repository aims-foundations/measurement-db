"""Validate tau-bench checks while retaining its full Sierra and HAL source audit."""

import hashlib
import json
import unicodedata

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _taubench
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.normalize_evaluation_settings import canonicalize_features, features_string
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='tau-bench complete native component evidence')
    originals, traces = attempt_view(tables)
    executions = tables['attempts'].set_index('attempt_id').to_dict('index')
    inputs = {row.attempt_id: json.loads(traces[row.response_id]) for row in originals.itertuples()}
    bank = {}
    for trace in inputs.values():
        native = trace['result']
        if isinstance(native, dict) and native.get('task'):
            if trace['task_id'] in bank:
                _check(bank[trace['task_id']], native['task'], 'tau-bench shared HAL task definition')
            bank[trace['task_id']] = native['task']
    items = tables['items'].set_index('item_id').to_dict('index')
    parents, attempt_parents = {}, {}
    protocol = metadata['grading']['verifiers']['native']
    labels = metadata['build']['parameters']['labels']
    for row in tables['responses'].itertuples():
        trace = inputs[row.attempt_id]
        native = trace['result']
        sierra = trace['configuration'].get('collection') == 'sierra'
        domain = metadata['build']['parameters']['sierra_domains'][trace['source_file'].rsplit('/', 1)[-1]] if sierra else 'airline'
        task = native['info']['task'] if sierra else native['task'] if isinstance(native, dict) else bank[trace['task_id']]
        _check(executions[row.attempt_id]['raw_attempt_id'], trace['source_file'] + '::' + str(trace['source_row']),
            'tau-bench original file and row identify the execution')
        content = json.dumps(dict(user_id=task['user_id'], instruction=task['instruction']), sort_keys=True, ensure_ascii=False)
        group = domain + '/' + hashlib.sha256(unicodedata.normalize('NFC', content).encode()).hexdigest()
        item = items[row.item_id]
        raw_parent, _, component = item['raw_item_id'].partition('::')
        _check((raw_parent, item['testlet_id'], item['content']), (group, group, unicodedata.normalize('NFC', content)),
            'tau-bench complete input and shared-stimulus identity')
        _check(item['item_features'], features_string(canonicalize_features(dict(domain=domain, input_scope=labels['input_scope']))),
            'tau-bench source input attributes')
        _check(pd.isna(item['asset_manifest']), True, 'tau-bench no invented assets')
        base_criterion = dict(reference_answer=None, rule=json.dumps(dict(rule=metadata['grading']['rule'],
            actions=task['actions'], outputs=task['outputs']), sort_keys=True, ensure_ascii=False))
        expected = base_criterion
        spec = protocol
        if component and component != 'ungraded_task_outcome':
            reward = native['info']['reward_info']['info']
            if component == 'database':
                rule = dict(reference_actions=task['actions'], expected_database_hash=reward.get('gt_data_hash'),
                    rule='Recorded final-database match; individual action success is not reported')
            else:
                _check(component.startswith('output:'), True, 'tau-bench supported native check type')
                rule = dict(required_output=component.removeprefix('output:'), rule='Recorded presence of the required textual output')
            expected = dict(reference_answer=None, rule=json.dumps(rule, ensure_ascii=False, sort_keys=True))
            spec = dict(protocol, component=component)
        _check(json.loads(item['grading_criterion']), expected, 'tau-bench full native check definition')
        _check(json.loads(item['verifier']), {'class': 'exact_matcher', 'spec': json.dumps(spec, sort_keys=True)}, 'tau-bench original grader provenance')
        identity = json.dumps(dict(domain=domain, task=task), sort_keys=True, ensure_ascii=False)
        parents.setdefault(identity, dict(item, item_id=identity, raw_item_id=domain + '/' + trace['task_id'],
            grading_criterion=json.dumps(base_criterion), verifier=json.dumps({'class': 'exact_matcher', 'spec': json.dumps(protocol, sort_keys=True)})))
        if row.attempt_id in attempt_parents:
            _check(attempt_parents[row.attempt_id], identity, 'tau-bench checks from the same complete task definition')
        attempt_parents[row.attempt_id] = identity
    _check(set(items), set(tables['responses'].item_id), 'tau-bench no orphan component definitions')
    originals['item_id'] = originals.attempt_id.map(attempt_parents)
    originals['attempt_id'] = None
    projected = {name: frame for name, frame in tables.items() if name != 'attempts'}
    projected.update(items=pd.DataFrame(parents.values()), responses=originals,
        traces=pd.DataFrame(dict(response_id=list(traces), trace=list(traces.values()))))
    return _taubench(directory, projected, metadata)
