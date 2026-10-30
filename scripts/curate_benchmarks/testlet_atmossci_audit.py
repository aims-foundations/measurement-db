"""Validate subanswers and every original AtmosSci input, answer and grading context."""

import hashlib
import json

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _atmossci, _atmossci_source_records, _check, _features
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='AtmosSci complete subanswer evidence')
    source = _atmossci_source_records(directory, metadata)
    native, original_items, _, _ = source
    originals, traces = attempt_view(tables)
    attempts = tables['attempts'].set_index('attempt_id').to_dict('index')
    items = tables['items'].set_index('item_id').to_dict('index')
    parents, mapping = {}, {}
    for row in tables['responses'].itertuples():
        key = attempts[row.attempt_id]['raw_attempt_id']
        original = native[key]
        parent = original['item']
        definition = original_items[parent]
        item = items[row.item_id]
        group, _, component = item['raw_item_id'].partition('::')
        _check((group, item['testlet_id']), (hashlib.sha256(definition['content'].encode()).hexdigest(),) * 2,
            'AtmosSci complete stimulus identifies the testlet')
        _check((item['content'], _features(item['item_features'])),
            (definition['content'], dict(question_type=definition['kind'])), 'AtmosSci full historical task content')
        _check(pd.isna(item['asset_manifest']), True, 'AtmosSci no invented attachments')
        verifier = {'class': 'exact_matcher' if definition['kind'] == 'MCQ' else 'judge',
            'spec': json.dumps(definition['spec'], sort_keys=True)}
        base = dict(reference_answer=json.dumps(definition['reference'], ensure_ascii=False, sort_keys=True),
            rule=metadata['grading']['rule'])
        if component == 'ungraded_task_outcome':
            expected = base
        else:
            judgment = json.loads(original['trace']['evaluation_json'])
            check = next(entry for entry in judgment['evaluation'] if entry['id'] == component)
            expected = dict(reference_answer=json.dumps(check['expected_answer'], ensure_ascii=False),
                rule=json.dumps(dict(subquestion_id=component, rule='Released subanswer judgment'), sort_keys=True))
        _check(json.loads(item['grading_criterion']), expected, 'AtmosSci actual subanswer and historical reference')
        _check(json.loads(item['verifier']), verifier, 'AtmosSci recorded judge configuration and reference origin')
        parents.setdefault(parent, dict(item, item_id=parent, raw_item_id=key,
            grading_criterion=json.dumps(base), verifier=json.dumps(verifier)))
        if row.attempt_id in mapping:
            _check(mapping[row.attempt_id], parent, 'AtmosSci consistent complete task for all subanswers')
        mapping[row.attempt_id] = parent
    _check(set(items), set(tables['responses'].item_id), 'AtmosSci no orphaned child definitions')
    originals['item_id'] = originals.attempt_id.map(mapping)
    originals['attempt_id'] = None
    projected = {name: frame for name, frame in tables.items() if name != 'attempts'}
    projected.update(items=pd.DataFrame(parents.values()), responses=originals,
        traces=pd.DataFrame(dict(response_id=list(traces), trace=list(traces.values()))))
    return _atmossci(directory, projected, metadata, source_records=source)
