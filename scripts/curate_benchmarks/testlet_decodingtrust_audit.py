"""Validate exact-output attribute ratings and all original DecodingTrust evidence."""

from collections import Counter
import hashlib
import json
import unicodedata

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _decodingtrust, _features
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='DecodingTrust released attribute ratings')
    items = tables['items'].set_index('item_id').to_dict('index')
    _check(set(tables['responses'].item_id), set(items), 'DecodingTrust no orphaned measurement definitions')
    parent_metrics = {}
    for row in tables['responses'].itertuples():
        parent = items[row.item_id]['raw_item_id'].split('::')[0]
        metric = json.loads(row.test_condition)['metric']
        _check(parent_metrics.setdefault(parent, metric), metric, 'DecodingTrust original task-level grading protocol')
    parents, parent_ids, item_parents = {}, {}, {}
    toxicity = metadata['grading']['verifiers']['toxicity_score']
    _check(toxicity['attributes'], ['toxicity', 'severe_toxicity', 'sexually_explicit', 'threat', 'profanity', 'identity_attack'],
        'DecodingTrust exact released Perspective attribute roster')
    for item_id, item in items.items():
        parent, attribute = item['raw_item_id'].split('::')
        metric = parent_metrics[parent]
        protocol = metadata['grading']['verifiers'][metric]
        features = _features(item['item_features'])
        _check(set(features), {'perspective', 'input_scope'}, 'DecodingTrust original task attributes')
        stimulus = [features['perspective'], features['input_scope'], unicodedata.normalize('NFC', item['content']).strip()]
        _check(item['testlet_id'], hashlib.sha256(json.dumps(stimulus, ensure_ascii=False).encode()).hexdigest(),
            'DecodingTrust shared exact stimulus within its documented perspective and input scope')
        parent_ids.setdefault(parent, item_id)
        first = parents.setdefault(parent, dict(item))
        for field in ('content', 'testlet_id', 'item_features'):
            _check(item[field], first[field], 'DecodingTrust every attribute retains the complete original stimulus')
        _check(pd.isna(item['asset_manifest']), True, 'DecodingTrust no invented input assets')
        criterion = json.loads(item['grading_criterion'])
        if attribute == 'aggregate':
            _check(criterion, dict(reference_answer=criterion['reference_answer'], rule=protocol['rule'], response_scale=protocol['scale']),
                'DecodingTrust unchanged fallback criterion and scale')
            _check(json.loads(item['verifier']), {'class': 'exact_matcher', 'spec': json.dumps(protocol['verifier'], sort_keys=True)},
                'DecodingTrust unchanged fallback verifier')
            reference = criterion['reference_answer']
        else:
            _check(metric, 'toxicity_score', 'DecodingTrust only released Perspective ratings are expanded')
            _check(attribute in toxicity['attributes'], True, 'DecodingTrust known released attribute')
            _check(criterion, dict(reference_answer=None, rule=json.dumps(dict(
                interpretation=toxicity['component_rule'], attribute=attribute), sort_keys=True), response_scale=toxicity['scale']),
                'DecodingTrust attribute-specific criterion and native numeric scale')
            _check(json.loads(item['verifier']), {'class': 'exact_matcher', 'spec': json.dumps(dict(
                toxicity['verifier'], attribute=attribute), sort_keys=True)}, 'DecodingTrust original attribute provenance')
            reference = None
        first.update(item_id=parent_ids[parent], raw_item_id=parent,
            grading_criterion=json.dumps(dict(reference_answer=reference, rule=protocol['rule'], response_scale=protocol['scale'])),
            verifier=json.dumps({'class': 'exact_matcher', 'spec': json.dumps(protocol['verifier'], sort_keys=True)}))
        item_parents[item_id] = parent

    originals, traces = attempt_view(tables)
    expected = Counter()
    attributed = 0
    for row in originals.itertuples():
        parent = item_parents[row.item_id]
        trace = json.loads(traces[row.response_id])
        linked = trace['toxicity_annotations'] if _features(items[row.item_id]['item_features'])['perspective'] == 'toxicity' else None
        if linked:
            attributed += 1
            for attribute in toxicity['attributes']:
                values = {annotation['record']['prompt']['score'][attribute] for annotation in linked}
                _check(all(type(value) in (int, float) and 0 <= value <= 1 for value in values), True,
                    'DecodingTrust released attribute values satisfy the native domain')
                expected[row.attempt_id, parent + '::' + attribute, next(iter(values)) if len(values) == 1 else None] += 1
        else:
            expected[row.attempt_id, parent + '::aggregate', None if pd.isna(row.response) else row.response] += 1
    actual = Counter((r.attempt_id, items[r.item_id]['raw_item_id'], None if pd.isna(r.response) else r.response)
        for r in tables['responses'].itertuples())
    _check(actual, expected, 'DecodingTrust exact attribute ratings, unavailable values and retained original observations')
    for row in tables['subjects'].itertuples():
        _check(_features(row.subject_features_extra), dict(reported_model_id=row.display_name,
            inference_settings='not consistently recorded in the released logs'), 'DecodingTrust unchanged model evidence')
        _check(row.harness, 'DecodingTrust', 'DecodingTrust original harness')

    # The original audit independently rereads all native outputs and annotations,
    # including exact full-output matching and duplicate API-record provenance.
    originals['item_id'] = originals.item_id.map(lambda key: parent_ids[item_parents[key]])
    projected = {name: frame for name, frame in tables.items() if name != 'attempts'}
    projected.update(items=pd.DataFrame(parents.values()), responses=originals,
        traces=pd.DataFrame(dict(response_id=list(traces), trace=list(traces.values()))))
    stats = _decodingtrust(directory, projected, metadata)
    return dict(stats, source_attempts=len(originals), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=tables['items'].testlet_id.nunique(),
        source_complete_output_attribute_assessments=attributed)
