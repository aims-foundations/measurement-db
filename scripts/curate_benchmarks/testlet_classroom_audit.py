"""Reconcile recorded readability indices and preserve the complete source audit."""

from collections import Counter
from copy import deepcopy
import json

import pandas as pd


def verify(directory, tables, metadata, source=None):
    from .native_result_audits import _check, _classroom_ai, _classroom_source_records
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.response_scales import canonical_response_scale
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='Classroom AI component source audit')
    source = source or _classroom_source_records(directory, metadata)
    metric_protocol = metadata['grading']['verifiers']['recorded_metric']
    metrics = metric_protocol['metrics']
    first_position = {}
    for position, question in enumerate(source['question_order']):
        first_position.setdefault(question, position)
    items = tables['items'].set_index('item_id').to_dict('index')
    definitions = Counter()
    for item in items.values():
        parent, metric = item['raw_item_id'].rsplit('::', 1)
        _check(metric in metrics, True, 'Classroom AI named source measurement')
        question = item['content']
        _check(parent, 'dgpt::' + str(first_position[question]), 'Classroom AI original question alias')
        _check(item['testlet_id'], parent, 'Classroom AI shared original question')
        criterion = json.loads(item['grading_criterion'])
        _check(criterion, dict(reference_answer=None,
            rule=json.dumps(dict(metric=metric, rule=metadata['grading']['rule']))),
            'Classroom AI exact metric criterion')
        _check(json.loads(item['verifier']), {'class': 'exact_matcher',
            'spec': json.dumps(dict(metric_protocol, selected_metric=metric), sort_keys=True)},
            'Classroom AI exact metric provenance')
        _check(pd.isna(item['asset_manifest']) and pd.isna(item['item_features']), True,
            'Classroom AI no invented assets or item attributes')
        definitions[question, metric] += 1
    _check(definitions, Counter({(question, metric): 1 for question in first_position for metric in metrics}),
        'Classroom AI every question and metric exactly once')

    observations, traces = attempt_view(tables)
    attempts = tables['attempts'].set_index('attempt_id')
    native_order = {key: str(index) for index, key in enumerate(source['native'])}
    trial_counts, native_trials = Counter(), {}
    for key, original in source['native'].items():
        context = original['subject'], original['question']
        trial_counts[context] += 1
        native_trials[key] = trial_counts[context]
    expected = Counter()
    for row in observations.itertuples():
        trace = json.loads(traces[row.response_id])
        key = trace['source_file'], trace['source_row']
        original = source['native'][key]
        _check(attempts.loc[row.attempt_id, 'raw_attempt_id'], native_order[key], 'Classroom AI original execution position')
        _check(row.trial, native_trials[key], 'Classroom AI repeated questions retain trials')
        # Parse the native lines independently of the builder's dataframe regex.
        readings = {}
        for line in original['metrics'].splitlines():
            if line.strip():
                metric, value = line.split(': score: ', 1)
                _check(metric not in readings, True, 'Classroom AI no duplicated native metric')
                readings[metric] = float(value.split(',', 1)[0])
        _check(set(readings), set(metrics), 'Classroom AI complete released measurement roster')
        parent = 'dgpt::' + str(first_position[original['question']])
        for metric, value in readings.items():
            expected[row.attempt_id, parent + '::' + metric, value] += 1
    actual = Counter((row.attempt_id, items[row.item_id]['raw_item_id'],
        None if pd.isna(row.response) else row.response) for row in tables['responses'].itertuples())
    _check(actual, expected, 'Classroom AI unmodified native numeric measurements and associations')

    # Reuse all prior answer, model, prompt, configuration and aggregate checks.
    protocol = metadata['grading']['verifiers']['integrated_readability']
    original_metadata = deepcopy(metadata)
    original_metadata['grading']['rule'] = protocol['rule']
    original_metadata['benchmark'].update(response_type='ordinal', response_scale=protocol['response_scale'], categorical=True)
    parents = tables['items'].drop_duplicates('testlet_id').copy()
    parents['raw_item_id'] = parents.testlet_id
    parent_ids = dict(zip(parents.testlet_id, parents.item_id))
    observations['item_id'] = observations.item_id.map(lambda key: parent_ids[items[key]['testlet_id']])
    parents['grading_criterion'] = json.dumps(dict(reference_answer=None, rule=protocol['rule']))
    parents['verifier'] = json.dumps({'class': 'exact_matcher', 'spec': json.dumps(protocol)})
    projected = {name: table for name, table in tables.items() if name != 'attempts'}
    projected.update(items=parents, responses=observations,
        traces=pd.DataFrame(dict(response_id=list(traces), trace=list(traces.values()))),
        benchmarks=tables['benchmarks'].assign(response_type='ordinal', categorical=True,
            response_scale=canonical_response_scale(protocol['response_scale'])))
    stats = _classroom_ai(directory, projected, original_metadata, source_records=source)
    return dict(stats, source_attempts=len(attempts), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=len(parents))
