"""Check the native selection scores, then retain the complete StatQA source audit."""

from collections import Counter
import json

import pandas as pd


def target_score(record, target):
    """Independent native count-and-score rule, including its parser failure policy."""
    try:
        prediction = json.loads(record['extracted_answer']).get(target, [])
        reference = json.loads(record['ground_truth'].replace("'", '"')).get(target, [])
        selected = {word.lower().strip() for word in prediction}
        gold = {word.lower().strip() for word in reference}
        correct, wrong, missed = len(selected & gold), len(selected - gold), len(gold - selected)
        return float(correct > 0 and wrong == 0 and missed == 0)
    except (ValueError, TypeError, AttributeError):
        return 0.


def verify(directory, tables, metadata, source=None):
    from .native_result_audits import _check, _features, _statqa, _statqa_sources
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='StatQA component source audit')
    source = source or _statqa_sources(directory, metadata)
    labels = metadata['build']['parameters']['labels']
    original_items = {item['raw_item_id']: item for item in source['definitions'].values()}
    items = tables['items'].set_index('item_id').to_dict('index')
    _check(len(items), 2 * len(original_items), 'StatQA exactly two named selection checks per question')
    item_keys = Counter()
    for item in items.values():
        parent, target = item['raw_item_id'].rsplit('::', 1)
        _check(target in {'columns', 'methods'}, True, 'StatQA native named selection target')
        original = original_items[parent]
        _check(item['testlet_id'], parent, 'StatQA original question membership')
        _check(item['content'], original['content'], 'StatQA complete task input for every check')
        _check(_features(item['item_features']), dict(dataset=original['dataset'], task=original['task'],
            difficulty=original['difficulty'], input_scope=labels['input_scope']), 'StatQA every component feature')
        criterion = json.loads(item['grading_criterion'])
        _check(criterion['reference_answer'], original['reference'], 'StatQA original complete reference')
        _check(json.loads(criterion['rule']), dict(target=target, rule=metadata['grading']['verifiers']['selection']['component_rule']), 'StatQA precise selection rule')
        verifier = json.loads(item['verifier'])
        _check(verifier['class'], 'exact_matcher', 'StatQA native scorer class')
        _check(json.loads(verifier['spec']), dict(metadata['grading']['verifiers']['selection'], selected_target=target), 'StatQA full scoring protocol')
        _check(pd.isna(item['asset_manifest']), True, 'StatQA no invented input assets')
        item_keys[parent, target] += 1
    _check(item_keys, Counter({(key, target): 1 for key in original_items for target in ['columns', 'methods']}), 'StatQA exact definition census')

    observations, traces = attempt_view(tables)
    attempts = tables['attempts'].set_index('attempt_id')
    expected = Counter()
    for row in observations.itertuples():
        trace = json.loads(traces[row.response_id])
        key = trace['source_member'], trace['source_row']
        original = source['native'][key]
        _check(attempts.loc[row.attempt_id, 'raw_attempt_id'], key[0] + '#' + str(key[1]), 'StatQA source execution identity')
        for target in ['columns', 'methods']:
            grade = None if original['grade'] is None else target_score(original['trace']['source_record'], target)
            expected[row.attempt_id, original['item']['raw_item_id'] + '::' + target, grade] += 1
    actual = Counter((row.attempt_id, items[row.item_id]['raw_item_id'],
        None if pd.isna(row.response) else row.response) for row in tables['responses'].itertuples())
    _check(actual, expected, 'StatQA exact native target scores and missingness')

    # The existing audit still checks all source configurations, input templates,
    # full answers, parser defects, original aggregates and repeated-run numbers.
    parents = tables['items'].copy()
    parents['raw_item_id'] = parents.testlet_id
    parents = parents.drop_duplicates('raw_item_id')
    parent_ids = dict(zip(parents.raw_item_id, parents.item_id))
    observations['item_id'] = observations.item_id.map(lambda key: parent_ids[items[key]['testlet_id']])
    parents['grading_criterion'] = [json.dumps(dict(reference_answer=original_items[key]['reference'],
        rule=metadata['grading']['rule'])) for key in parents.raw_item_id]
    parents['verifier'] = json.dumps({'class': 'exact_matcher', 'spec': json.dumps(metadata['grading']['verifiers']['selection'])})
    projected = {name: table for name, table in tables.items() if name != 'attempts'}
    projected.update(items=parents, responses=observations,
        traces=pd.DataFrame(dict(response_id=list(traces), trace=list(traces.values()))))
    stats = _statqa(directory, projected, metadata, source=source)
    return dict(stats, source_attempts=len(attempts), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=len(original_items))
