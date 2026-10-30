"""Preserve complete original task audits for native exact-match components."""

import hashlib
import json
import unicodedata

from .native_result_audits import _check
from .testlet_audit_views import parent_view


def verify(directory, tables, metadata):
    if directory.name == 'care_enzymes':
        return _care(directory, tables, metadata)
    return _arc(directory, tables, metadata)


def _care(directory, tables, metadata):
    from .native_result_audits import _care as native_audit, _care_source_records

    source = _care_source_records(directory, metadata)
    items = tables['items'].set_index('item_id').to_dict('index')
    executions = tables['attempts'].set_index('attempt_id').trace.map(json.loads)
    parents, mapping = {}, {}
    for attempt, children in tables['responses'].groupby('attempt_id', sort=False):
        trace = json.loads(executions[attempt]['source_trace'])
        record = source[0][trace['source_file'], trace['source_row']]
        content = json.dumps(record['inputs'], ensure_ascii=False, sort_keys=True)
        testlet = hashlib.sha256(unicodedata.normalize('NFC', content).encode()).hexdigest()
        criterion = dict(reference_answer=record['reference'], rule=metadata['grading']['rule'])
        identity = json.dumps([content, criterion, record['task'], record['mode']], sort_keys=True)
        for child in children.itertuples():
            item = items[child.item_id]
            _check(item['testlet_id'], testlet, 'CARE complete input testlet identity')
            index = int(item['raw_item_id'].rsplit('::', 1)[1])
            reference = record['reference'].split(';')[index]
            _check(json.loads(item['grading_criterion']), dict(reference_answer=json.dumps(reference),
                rule=json.dumps(dict(reference_index=index,
                    rule='Exact released EC-level-4 match against the source-decoded rank-zero candidate list'))),
                'CARE original reference component')
            parent = dict(item, item_id=identity, raw_item_id=testlet, grading_criterion=json.dumps(criterion))
            if identity in parents:
                _check(parent, parents[identity], 'CARE complete input and metadata on every reference')
            parents[identity] = parent
        mapping[attempt] = identity
    projected = parent_view(tables, list(parents.values()), execution_parents=mapping)
    return native_audit(directory, projected, metadata, source_records=source)


def _arc(directory, tables, metadata):
    from .native_result_audits import _arcagi, _arcagi_source_records

    source = _arcagi_source_records(directory, metadata)
    native = source[0]
    by_file = {name: record for record in native.values() for name in record['files']}
    items = tables['items'].set_index('item_id').to_dict('index')
    executions = tables['attempts'].set_index('attempt_id').trace.map(json.loads)
    parents, mapping = {}, {}
    for attempt, children in tables['responses'].groupby('attempt_id', sort=False):
        trace = json.loads(executions[attempt]['source_trace'])
        record = by_file[trace['source_files'][0]]
        version, puzzle = json.loads(record['item'])
        criterion = dict(reference_answer=json.dumps([test['output'] for test in puzzle['test']]),
            rule=metadata['grading']['rule'])
        for child in children.itertuples():
            item = items[child.item_id]
            raw, _, index = item['raw_item_id'].rpartition('::')
            _check(item['testlet_id'], raw.split('::')[1], 'ARC original puzzle testlet')
            index = int(index)
            _check(json.loads(item['grading_criterion']), dict(reference_answer=json.dumps(puzzle['test'][index]['output']),
                rule=json.dumps(dict(test_grid_index=index,
                    rule='Either of the two released candidate grids must exactly match this test-grid reference'))),
                'ARC dated native reference grid')
            parent = dict(item, item_id=record['item'], raw_item_id=raw, grading_criterion=json.dumps(criterion))
            if record['item'] in parents:
                _check(parent, parents[record['item']], 'ARC full original puzzle on every test-grid item')
            parents[record['item']] = parent
        mapping[attempt] = record['item']
    projected = parent_view(tables, list(parents.values()), execution_parents=mapping)
    projected['benchmarks']['response_scale'] = json.dumps(dict(kind='interval', min=0, max=1, direction='higher_is_better'))
    return _arcagi(directory, projected, metadata, source_records=source)
