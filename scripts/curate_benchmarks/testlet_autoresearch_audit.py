"""Verify released paper-level verdicts and the complete original search runs."""

from collections import Counter
import json

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _autoresearch, _autoresearch_source_records, _check
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='AutoResearchBench reference checks')
    source = _autoresearch_source_records(directory, metadata)
    native, inputs, _ = source
    items = tables['items'].set_index('item_id').to_dict('index')
    parents = tables['items'].drop_duplicates('testlet_id').copy()
    parent_ids = dict(zip(parents.testlet_id, parents.item_id))
    expected_items = Counter()
    for item in items.values():
        parent, _, check = item['raw_item_id'].partition('::')
        family, position = parent.split(':')
        original = inputs[family, int(position)]
        _check(item['testlet_id'], parent, 'AutoResearch original task testlet')
        _check(item['content'], original['content'], 'AutoResearch every check retains the exact initial messages')
        _check(item['item_features'], 'research_task=' + family, 'AutoResearch unchanged task attributes')
        _check(pd.isna(item['asset_manifest']), True, 'AutoResearch no invented task assets')
        protocol = metadata['grading']['verifiers'][family]
        if family == 'deep':
            _check(check, '', 'AutoResearch no unreleased Deep sub-judgments')
            _check(json.loads(item['grading_criterion']), dict(reference_answer=json.dumps(original['reference'], ensure_ascii=False),
                rule=protocol['logic'], response_scale=protocol['aggregate_scale']), 'AutoResearch unchanged Deep judgment and native scale')
            _check(json.loads(item['verifier']), {'class': 'judge', 'spec': json.dumps(protocol, sort_keys=True)},
                'AutoResearch retained Deep judgment provenance')
            expected_items[parent, ''] += 1
            continue
        kind = 'reference' if check.startswith('reference:') else 'extra_reference_count'
        paper = check.removeprefix('reference:') if kind == 'reference' else None
        _check(paper in original['reference'] if paper is not None else check == kind,
            True, 'AutoResearch source-declared reference or error count')
        spec = dict(protocol, selected_check=check, component_rule=protocol['components'][kind]['rule'])
        _check(json.loads(item['grading_criterion']), dict(reference_answer=paper,
            rule=spec['component_rule'], response_scale=protocol['components'][kind]['response_scale']),
            'AutoResearch exact reference-check criterion and native scale')
        _check(json.loads(item['verifier']), {'class': 'exact_matcher', 'spec': json.dumps(spec, sort_keys=True)},
            'AutoResearch original recorded evaluator and selected check')
        expected_items[parent, check] += 1
    roster = Counter()
    for (family, position), original in inputs.items():
        parent = family + ':' + str(position)
        checks = ['reference:' + paper for paper in original['reference']] + ['extra_reference_count'] if family == 'wide' else ['']
        roster.update((parent, check) for check in checks)
    _check(expected_items, roster, 'AutoResearch every source-defined check exactly once')

    originals, traces = attempt_view(tables)
    attempts = tables['attempts'].set_index('attempt_id').to_dict('index')
    expected = Counter()
    for row in originals.itertuples():
        trace = json.loads(traces[row.response_id])
        key = trace['source_file'], trace['source_row'], trace['native_pass']['pass_id']
        record = native[key]
        family, position = record['item']
        parent = family + ':' + str(position)
        _check(items[row.item_id]['testlet_id'], parent, 'AutoResearch execution belongs to its original task')
        _check(attempts[row.attempt_id]['raw_attempt_id'], ':'.join(map(str, key)),
            'AutoResearch source file, row and pass identify the execution')
        if family == 'deep':
            expected[row.attempt_id, parent, record['response']] += 1
            continue
        result, = [p for p in trace['evaluation_record']['pass_results'] if p['pass_id'] == key[2]]
        truth, prediction = set(result['gt_arxiv_ids']), set(result['predicted_arxiv_ids'])
        _check((set(result['hit_ids']), set(result['missed_ids']), set(result['extra_ids'])),
            (truth & prediction, truth - prediction, prediction - truth), 'AutoResearch native hit, miss and extra-ID partitions')
        for paper in truth:
            expected[row.attempt_id, parent + '::reference:' + paper, int(paper in prediction)] += 1
        expected[row.attempt_id, parent + '::extra_reference_count', len(prediction - truth)] += 1
    actual = Counter((r.attempt_id, items[r.item_id]['raw_item_id'], r.response) for r in tables['responses'].itertuples())
    _check(actual, expected, 'AutoResearch all reference verdicts, error counts and Deep judgments')

    # Reconstruct the original task-level view solely to run its unchanged audit.
    # Every child definition and outcome has already been verified above.
    parents['raw_item_id'] = parents.testlet_id
    criteria, verifiers = [], []
    for parent in parents.testlet_id:
        family, position = parent.split(':')
        protocol = metadata['grading']['verifiers'][family]
        criteria.append(json.dumps(dict(reference_answer=json.dumps(inputs[family, int(position)]['reference'], ensure_ascii=False),
            rule=protocol['logic'])))
        verifiers.append(json.dumps({'class': 'judge' if family == 'deep' else 'exact_matcher',
            'spec': json.dumps(protocol, sort_keys=True)}))
    parents['grading_criterion'], parents['verifier'] = criteria, verifiers
    originals['item_id'] = originals.item_id.map(lambda key: parent_ids[items[key]['testlet_id']])
    projected = {name: frame for name, frame in tables.items() if name != 'attempts'}
    projected.update(items=parents, responses=originals,
        traces=pd.DataFrame(dict(response_id=list(traces), trace=list(traces.values()))))
    stats = _autoresearch(directory, projected, metadata, source_records=source)
    return dict(stats, source_attempts=len(originals), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=len(parents))
