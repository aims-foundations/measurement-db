"""Check AppWorld's original task data and executions alongside its test verdicts."""

import hashlib
import json

import pandas as pd


def verify(directory, tables, metadata):
    from .appworld_bundle import open_bundle
    from .batch3_audits import _appworld, _check, _features
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset
    from ..build_measurement_tables.normalize_evaluation_settings import canonicalize_features, features_string

    validate_dataset(tables, context='AppWorld complete native component evidence')
    parents, parent_ids, item_parents = {}, {}, {}
    with open_bundle(directory / 'raw/data-0.1.0.bundle') as bank:
        for item in tables['items'].to_dict('records'):
            task, _, component = item['raw_item_id'].partition('::')
            prefix = f'data/tasks/{task}/'
            spec = json.loads(bank.read(prefix + 'specs.json'))
            code = bank.read(prefix + 'ground_truth/evaluation.py').decode('utf-8')
            test_data = json.loads(bank.read(prefix + 'ground_truth/test_data.json'))
            criterion = json.loads(item['grading_criterion'])
            _check(item['testlet_id'], task, 'AppWorld original task testlet')
            _check(item['content'], spec['instruction'], 'AppWorld complete instruction on every check')
            features = _features(item['item_features'])
            _check(features.get('split') in {'test_normal', 'test_challenge'}, True, 'AppWorld released evaluation split')
            expected_features = canonicalize_features(dict(appworld_task_id=task, split=features['split'],
                db_version=spec['db_version'], datetime=spec['datetime']))
            _check(item['item_features'], features_string(expected_features), 'AppWorld original database version and task time')
            _check(pd.isna(item['asset_manifest']), True, 'AppWorld no invented task attachments')
            base_criterion = dict(reference_answer=None, rule=json.dumps(dict(
                criterion=metadata['grading']['rule'], test_data=test_data), sort_keys=True))
            if component:
                requirement = criterion['rule']
                _check(component, hashlib.sha256(requirement.encode()).hexdigest(), 'AppWorld exact requirement identity')
                _check(criterion, dict(reference_answer=None, rule=requirement), 'AppWorld native binary requirement criterion')
                expected_spec = json.dumps(dict(requirement=requirement, grading_code=code, test_data=test_data), sort_keys=True)
            else:
                _check(criterion, base_criterion, 'AppWorld unchanged aggregate fallback')
                expected_spec = code
            _check(json.loads(item['verifier']), {'class': 'exact_matcher', 'spec': expected_spec},
                'AppWorld complete grading code and original test data')
            parent_ids.setdefault(task, item['item_id'])
            parents.setdefault(task, dict(item, item_id=parent_ids[task], raw_item_id=task,
                grading_criterion=json.dumps(base_criterion), verifier=json.dumps({'class': 'exact_matcher', 'spec': code})))
            item_parents[item['item_id']] = task

    originals, traces = attempt_view(tables)
    attempts = tables['attempts'].set_index('raw_attempt_id').to_dict('index')
    seen = set()
    for path in sorted((directory / 'raw/bundles').glob('*.bundle')):
        with open_bundle(path) as bundle:
            name = path.stem
            run = json.loads(bundle.read(f'{name}/metadata.json'))
            records = json.loads(bundle.read(f'{name}/evaluations/{run["dataset"]}.json'))['individual']
            members = set(bundle.namelist())
            for task, record in records.items():
                key = path.name + '/' + task
                seen.add(key)
                location = f'{name}/tasks/{task}/logs/environment_io.md'
                log = bundle.read(location).decode('utf-8', 'replace') if location in members else None
                actual = json.loads(json.loads(attempts[key]['trace'])['source_trace'])
                _check(actual, dict(success=record['success'], num_tests=record['num_tests'],
                    passes=record['passes'], failures=record['failures'], environment_io=log),
                    'AppWorld complete original result and execution log')
    _check(seen, set(attempts), 'AppWorld all original executions retained exactly once')
    originals['item_id'] = originals.item_id.map(lambda key: parent_ids[item_parents[key]])
    logs = {key: json.loads(value)['environment_io'] for key, value in traces.items()}
    logs = {key: value for key, value in logs.items() if isinstance(value, str) and value.strip()}
    projected = {name: frame for name, frame in tables.items() if name != 'attempts'}
    projected.update(items=pd.DataFrame(parents.values()), responses=originals,
        traces=pd.DataFrame(dict(response_id=list(logs), trace=list(logs.values()))))
    return _appworld(directory, projected)
