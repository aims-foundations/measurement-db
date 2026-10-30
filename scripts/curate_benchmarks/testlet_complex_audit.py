"""Validate complete native rubric structure without fabricating any judgments."""

from collections import Counter
import json

import pandas as pd


def verify(directory, tables, metadata, source=None):
    from .native_result_audits import _check, _complex, _complex_source_records, _features
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='ComplexBench rubric structure audit')
    tasks, records = source or _complex_source_records(directory, metadata)
    protocol = metadata['grading']['verifiers']['official_pipeline']
    items = tables['items'].set_index('item_id').to_dict('index')
    definitions = Counter()
    for item in items.values():
        parent, point_id = item['raw_item_id'].split('::')
        task = tasks[int(parent)]
        point = task['scoring_questions'][int(point_id)]
        _check(str(point['point_id']), point_id, 'ComplexBench original rubric coordinate')
        _check(item['testlet_id'], parent, 'ComplexBench original instruction membership')
        _check(item['content'], task['instruction'], 'ComplexBench complete input for each rubric point')
        _check(_features(item['item_features']), dict(task_type=task['task_types'], category=task['category']),
            'ComplexBench original attributes')
        _check(json.loads(item['grading_criterion']), dict(reference_answer=None,
            rule=json.dumps(dict(protocol=protocol['component_rule'], scoring_question=point), ensure_ascii=False, sort_keys=True)),
            'ComplexBench full criterion, rule and dependency IDs')
        _check(json.loads(item['verifier']), {'class': 'judge', 'judged_by': 'llm',
            'spec': json.dumps(dict(protocol, selected_point_id=point['point_id']), sort_keys=True)},
            'ComplexBench unavailable historical judge explicitly documented')
        _check(pd.isna(item['asset_manifest']), True, 'ComplexBench no invented resources')
        definitions[int(parent), point['point_id']] += 1
    _check(definitions, Counter({(key, point['point_id']): 1 for key, task in tasks.items() for point in task['scoring_questions']}),
        'ComplexBench every released rubric point exactly once')
    _check(tables['responses'].response.isna().all(), True, 'ComplexBench no fabricated child verdicts')
    observations, traces = attempt_view(tables)
    attempts = tables['attempts'].set_index('attempt_id')
    expected = Counter()
    for row in observations.itertuples():
        trace = json.loads(traces[row.response_id])
        key = trace['source_file'], trace['source_line']
        _check(attempts.loc[row.attempt_id, 'raw_attempt_id'], key[0] + ':' + str(key[1]),
            'ComplexBench original execution identity')
        task = tasks[records[key]['main_id']]
        for point in task['scoring_questions']:
            expected[row.attempt_id, str(task['main_id']) + '::' + str(point['point_id'])] += 1
    _check(Counter((row.attempt_id, items[row.item_id]['raw_item_id']) for row in tables['responses'].itertuples()),
        expected, 'ComplexBench exact rubric-to-execution associations')

    parents = tables['items'].drop_duplicates('testlet_id').copy()
    parents['raw_item_id'] = parents.testlet_id
    parent_ids = dict(zip(parents.testlet_id, parents.item_id))
    observations['item_id'] = observations.item_id.map(lambda key: parent_ids[items[key]['testlet_id']])
    parents['grading_criterion'] = [json.dumps(dict(reference_answer=None, rule=json.dumps(dict(
        protocol=metadata['grading']['rule'], scoring_questions=tasks[int(key)]['scoring_questions']),
        ensure_ascii=False, sort_keys=True))) for key in parents.raw_item_id]
    parents['verifier'] = json.dumps({'class': 'judge', 'judged_by': 'llm', 'spec': json.dumps(protocol, sort_keys=True)})
    projected = {name: table for name, table in tables.items() if name != 'attempts'}
    projected.update(items=parents, responses=observations,
        traces=pd.DataFrame(dict(response_id=list(traces), trace=list(traces.values()))),
        benchmarks=tables['benchmarks'].assign(response_type='fraction',
            response_scale=json.dumps(dict(kind='interval', min=0, max=1, direction='higher_is_better'))))
    stats = _complex(directory, projected, metadata, source_records=(tasks, records))
    return dict(stats, source_attempts=len(attempts), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=len(parents))
