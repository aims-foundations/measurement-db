"""Reconcile all specialist labels and retain the original DAS provenance audit."""

import ast
from collections import Counter
import json

import pandas as pd


def verify(directory, tables, metadata, source_records=None):
    from .native_result_audits import _check, _das_med_hallucination, _das_source_records
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='DAS subtype source audit')
    source = source_records or _das_source_records(directory, metadata)
    protocol = metadata['grading']['verifiers']['detector']
    tree = ast.parse((directory / 'raw/legacy_protocol/hallucination/agent_outputs_o1pro.py').read_text())
    vocabulary = {}
    for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        for enum in (node for node in cls.body if isinstance(node, ast.ClassDef) and node.name == 'SubCode'):
            vocabulary.update({ast.literal_eval(node.value): node.targets[0].id
                for node in enum.body if isinstance(node, ast.Assign)})
    _check(protocol['subtypes'], vocabulary, 'DAS exact native subtype vocabulary')
    _check(metadata['benchmark']['response_scale']['values'], [0, 1], 'DAS subtype detection scale')
    items = tables['items'].set_index('item_id').to_dict('index')
    expected_items = Counter()
    for item in items.values():
        parent, code = item['raw_item_id'].rsplit('::', 1)
        _check(item['testlet_id'], parent, 'DAS original prompt membership')
        criterion = json.loads(item['grading_criterion'])
        _check(criterion.get('reference_answer'), None, 'DAS no invented correctness reference')
        _check(json.loads(criterion['rule']), dict(subtype=code, name=vocabulary[code], rule=protocol['subtype_rule']), 'DAS named subtype and missingness policy')
        verifier = json.loads(item['verifier'])
        _check((verifier['class'], verifier['judged_by']), ('judge', 'llm'), 'DAS native specialist judge')
        _check(json.loads(verifier['spec']), dict(protocol, selected_subtype=code), 'DAS complete grading protocol')
        _check(pd.isna(item['asset_manifest']), True, 'DAS no invented input resources')
        expected_items[parent, code] += 1
    parents = tables['items'].copy()
    parents['raw_item_id'] = parents.testlet_id
    parents = parents.drop_duplicates('raw_item_id')
    _check(expected_items, Counter({(parent, code): 1 for parent in parents.raw_item_id for code in vocabulary}), 'DAS every subtype exactly once per prompt')
    parent_ids = dict(zip(parents.raw_item_id, parents.item_id))
    parent_content = parents.set_index('raw_item_id')[['content', 'item_features']].to_dict('index')
    for item in items.values():
        _check({key: item[key] for key in ['content', 'item_features']}, parent_content[item['testlet_id']], 'DAS identical complete stimulus on every subtype')

    observations, traces = attempt_view(tables)
    attempts = tables['attempts'].set_index('attempt_id')
    expected, counts = Counter(), Counter()
    for row in observations.itertuples():
        trace = json.loads(traces[row.response_id])
        key = trace['generation_file'], trace['generation_row']
        _check(attempts.loc[row.attempt_id, 'raw_attempt_id'], key[0] + '/' + str(key[1]), 'DAS original execution identity')
        source_judgment = source[1].get(key)
        decisions = {} if source_judgment is None else {d['code']: d for d in source_judgment[2]['agent_decisions']}
        if source_judgment:
            _check(len(decisions), len(source_judgment[2]['agent_decisions']), 'DAS unique specialist per judgment')
        parent = items[row.item_id]['testlet_id']
        for code in vocabulary:
            decision = decisions.get(int(code[0]))
            label = decision.get('classification') if decision and decision['called'] else None
            if label is None or label == '0.5':
                grade = None
            elif label == '0':
                grade = 0.
            else:
                _check(isinstance(label, list) and all(value in vocabulary and value[0] == code[0] for value in label), True, 'DAS original specialist classification vocabulary')
                grade = float(code in label)
            expected[row.attempt_id, parent + '::' + code, grade] += 1
            counts['source_unknown_subtypes' if grade is None else 'source_detected_subtypes' if grade else 'source_undetected_subtypes'] += 1
    actual = Counter((row.attempt_id, items[row.item_id]['raw_item_id'],
        None if pd.isna(row.response) else row.response) for row in tables['responses'].itertuples())
    _check(actual, expected, 'DAS every native subtype detection and unknown label')
    observations['item_id'] = observations.item_id.map(lambda key: parent_ids[items[key]['testlet_id']])
    parents['grading_criterion'] = json.dumps(dict(reference_answer=None, rule=metadata['grading']['rule']))
    parents['verifier'] = json.dumps({'class': 'judge', 'judged_by': 'llm', 'spec': json.dumps(protocol)})
    projected = {name: table for name, table in tables.items() if name != 'attempts'}
    projected.update(items=parents, responses=observations,
        traces=pd.DataFrame(dict(response_id=list(traces), trace=list(traces.values()))))
    stats = _das_med_hallucination(directory, projected, metadata, source_records=source)
    return dict(stats, **counts, source_attempts=len(attempts), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=len(parents))
