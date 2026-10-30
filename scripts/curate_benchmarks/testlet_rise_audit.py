"""Validate RISE dimensions with original images, outputs, judge text and settings."""

from collections import Counter
import json

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _risebench
    from .testlet_audit_views import parent_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='RISE complete native dimension evidence')
    bank = {row['index']: row for row in json.loads((directory / 'raw/tasks/datav2_total_w_subtask.json').read_text())}
    protocols = metadata['grading']['verifiers']
    specs = [protocols['published_judgments'], protocols['current_judgments']]
    parents, mapping, roster = {}, {}, Counter()
    for item in tables['items'].to_dict('records'):
        task, _, dimension = item['raw_item_id'].partition('::')
        _check(item['testlet_id'], task, 'RISE original shared image-editing task')
        _check(dimension in {'Reasoning', 'ApprConsistency', 'VisualPlausibility'}, True, 'RISE named native rating dimension')
        spec = json.loads(json.loads(item['verifier'])['spec'])
        _check(spec in specs, True, 'RISE captured historical grading protocol')
        fields = {name: bank[task].get(name) for name in metadata['build']['parameters']['grading_fields']}
        fields['completion_rule'] = metadata['grading']['rule'] if spec == specs[0] else specs[1]['rule']
        reference = bank[task].get('reference') or bank[task].get('reference_txt')
        criterion = json.loads(item['grading_criterion'])
        _check(set(criterion), {'reference_answer', 'rule'}, 'RISE grading fields')
        _check(criterion['reference_answer'], reference, 'RISE original reference')
        _check(json.loads(criterion['rule']), dict(dimension=dimension,
            rule='Native 1-5 dimension rating from a successfully parsed judge record', task_rubric=fields),
            'RISE complete original rubric and dimension')
        identity = json.dumps([task, item['content'], spec], sort_keys=True)
        parent = dict(item, item_id=identity, raw_item_id=task, grading_criterion=json.dumps(dict(reference_answer=reference,
            rule=json.dumps(fields, ensure_ascii=False, allow_nan=False))))
        if identity in parents:
            _check(parent, parents[identity], 'RISE all ratings retain the same full task and image links')
        parents[identity] = parent
        mapping[item['item_id']] = identity
        roster[identity, dimension] += 1
    _check(roster, Counter({(key, dimension): 1 for key in parents for dimension in ('Reasoning', 'ApprConsistency', 'VisualPlausibility')}),
        'RISE complete dimension roster under each original input and grader setting')
    return _risebench(directory, parent_view(tables, list(parents.values()), mapping), metadata)
