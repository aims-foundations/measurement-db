"""Preserve complete clinical cases, weighted rubrics and original judge evidence."""

from collections import Counter
import json

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _csedb, _csedb_source_records, _features
    from .testlet_audit_views import parent_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='CSEDB complete native component evidence')
    source = _csedb_source_records(directory, metadata)
    tasks, records = source
    protocol = metadata['grading']['verifiers']['released_judge']
    parents, mapping, roster = {}, {}, Counter()
    for item in tables['items'].to_dict('records'):
        task, _, component = item['raw_item_id'].partition('::')
        native = tasks[task]
        _check((item['testlet_id'], item['content'], _features(item['item_features'])),
            (task, native['content'], native['features']), 'CSEDB complete clinical case and attributes')
        _check(pd.isna(item['asset_manifest']), True, 'CSEDB no invented attachments')
        rubric = dict(protocol=metadata['grading']['rule'], criterion=native['criterion'],
            design_principles=native['design_principles'], rules=native['rules'])
        base = dict(reference_answer=None, rule=json.dumps(rubric, ensure_ascii=False, sort_keys=True))
        expected, spec = base, protocol
        if component and component != 'ungraded_task_outcome':
            kind, index = component.split(':')
            _check(kind, 'criterion', 'CSEDB native rubric criterion')
            index = int(index)
            expected = dict(reference_answer=None, rule=json.dumps(dict(parent_rubric=rubric,
                selected_criterion_index=index, selected_criterion=native['rules']['规则判断列表'][index - 1]),
                ensure_ascii=False, sort_keys=True))
            spec = dict(protocol, selected_criterion_index=index)
            roster[task, index] += 1
        _check(json.loads(item['grading_criterion']), expected, 'CSEDB criterion content, weights and design context')
        verifier = {'class': 'judge', 'judged_by': 'llm', 'spec': json.dumps(spec, sort_keys=True)}
        _check(json.loads(item['verifier']), verifier, 'CSEDB original grading protocol')
        parents.setdefault(task, dict(item, item_id=task, raw_item_id=task,
            grading_criterion=json.dumps(base), verifier=json.dumps(dict(verifier, spec=json.dumps(protocol, sort_keys=True)))))
        mapping[item['item_id']] = task
    _check(roster, Counter({(task, i): 1 for task, native in tasks.items()
        for i in range(1, len(native['rules'].get('规则判断列表', [])) + 1)}), 'CSEDB complete native rubric roster')
    projected = parent_view(tables, list(parents.values()), mapping)
    # Reconstruct the historical aggregate-scale view for the unchanged audit;
    # the imported child observations remain binary (or explicitly unknown).
    projected['benchmarks'] = projected['benchmarks'].copy()
    projected['benchmarks']['response_type'] = 'fraction'
    projected['benchmarks']['response_scale'] = json.dumps(dict(kind='interval', min=0, max=1, direction='higher_is_better'))
    return _csedb(directory, projected, metadata, source_records=source)
