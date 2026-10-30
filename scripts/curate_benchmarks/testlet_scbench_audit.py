"""Check named scientific endpoint tests and retain the complete native audit."""

import hashlib
import json

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _scbench_long, _scbench_long_sources
    from .testlet_audit_views import parent_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='scBench-Long complete named check evidence')
    source = _scbench_long_sources(directory, metadata)
    parents, mapping = {}, {}
    for item in tables['items'].to_dict('records'):
        task, _, component = item['raw_item_id'].partition('::')
        native = source['tasks'][task]
        _check(item['testlet_id'], task, 'scBench original scientific task testlet')
        base = dict(reference_answer=None, rule=json.dumps(native['criterion'], sort_keys=True))
        criterion = json.loads(item['grading_criterion'])
        if component:
            rule = json.loads(criterion['rule'])
            _check(component, hashlib.sha256(json.dumps([rule['kind'], rule['check']]).encode()).hexdigest(),
                'scBench original check kind and name')
            _check(criterion, dict(reference_answer=None, rule=json.dumps(dict(check=rule['check'], kind=rule['kind'],
                rule='Released passed flag for this named grader check', task_grader=native['criterion']), sort_keys=True)),
                'scBench named check and complete task grader')
        else:
            _check(criterion, base, 'scBench retained task-level criterion')
        parent = dict(item, item_id=task, raw_item_id=task, grading_criterion=json.dumps(base))
        if task in parents:
            _check(parent, parents[task], 'scBench every check retains complete instructions, input links and source metadata')
        parents[task] = parent
        mapping[item['item_id']] = task
    return _scbench_long(directory, parent_view(tables, list(parents.values()), mapping), metadata, source=source)
