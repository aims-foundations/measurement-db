"""Check Cybench subtask items and both measurements of each guided run."""

import json

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _cybench, _cybench_source_records, _features
    from .testlet_audit_views import attempt_view

    source = _cybench_source_records(directory, metadata)
    expected, native, published, _ = source
    items = tables['items'].set_index('item_id').to_dict('index')
    subjects = tables['subjects'].set_index('subject_id').to_dict('index')
    observations, traces = attempt_view(tables)
    children = tables['responses'].groupby('attempt_id', sort=False)
    parents, rows, restored_traces = {}, [], []
    protocols = metadata['grading']['verifiers']
    for row in observations.itertuples():
        trace = json.loads(traces[row.response_id])
        prototype = items[row.item_id]
        features = _features(prototype['item_features'])
        model = _features(subjects[row.subject_id]['subject_features_extra'])['deployment_name']
        key = model, features['task'], features['mode']
        entry = expected[key]
        _check(prototype['testlet_id'], key[1], 'Cybench original challenge testlet')
        fraction = expected.get((model, key[1], 'subtask_fractional')) if key[2] == 'subtask_guided' else None
        cell = published.get((model, key[1], 'subtask_fractional'))
        expected_fraction = None if fraction is None else dict(score=fraction['grade'],
            paper_table=cell['table'] if cell else None, paper_value=cell['value'] if cell else None)
        _check(trace, dict(source_file=entry.get('source_file'), record=native.get(entry.get('source_file')),
            measurement=key[2], subtask_fraction=expected_fraction, paper_cell=published.get(key)),
            'Cybench full native run and both original measurements')
        for child in children.get_group(row.attempt_id).itertuples():
            item = items[child.item_id]
            for field in ['content', 'item_features', 'asset_manifest', 'testlet_id']:
                _check(None if pd.isna(item[field]) else item[field],
                    None if pd.isna(prototype[field]) else prototype[field], 'Cybench complete common task ' + field)
            component = item['raw_item_id'].rsplit('::', 1)[1]
            if component == 'flag_recovery':
                criterion = dict(reference_answer=entry['reference'],
                    rule=json.dumps(dict(rule=protocols[key[2]]['rule'], subtasks=None), sort_keys=True))
                protocol = protocols[key[2]]
            else:
                check = trace['record']['subtask_completions'][component]
                criterion = dict(reference_answer=check['subtask']['answer'], rule=json.dumps(dict(
                    rule='Recorded binary subtask verdict; unanswered subtasks retain their released score',
                    subtask=component, definition=check['subtask']), ensure_ascii=False, sort_keys=True))
                protocol = dict(kind='recorded_subtask_verdict', protocol=protocols['subtask_fractional'])
            _check(json.loads(item['grading_criterion']), criterion, 'Cybench exact original check definition')
            _check(json.loads(item['verifier']), dict(**{'class': 'exact_matcher'}, spec=json.dumps(protocol, sort_keys=True)),
                'Cybench original checker protocol')
        for mode in [key[2], *(['subtask_fractional'] if fraction is not None else [])]:
            original = expected[model, key[1], mode]
            criterion = dict(reference_answer=original['reference'] if mode != 'subtask_fractional' else None,
                rule=json.dumps(dict(rule=protocols[mode]['rule'],
                    subtasks=original['subtasks'] if mode == 'subtask_fractional' else None), sort_keys=True))
            verifier = dict(**{'class': 'exact_matcher'}, spec=json.dumps(protocols[mode], sort_keys=True))
            identity = json.dumps([prototype['content'], criterion, verifier], sort_keys=True)
            parent = dict(prototype, item_id=identity, raw_item_id=key[1] + ':' + mode,
                item_features=';'.join(name + '=' + value for name, value in dict(features, mode=mode).items()),
                grading_criterion=json.dumps(criterion), verifier=json.dumps(verifier))
            if identity in parents:
                _check(parent, parents[identity], 'Cybench common original task definition')
            parents[identity] = parent
            response = row._asdict()
            response.pop('Index')
            response.update(response_id=row.response_id + ':' + mode, item_id=identity, attempt_id=None,
                response=row.response if mode == key[2] else trace['subtask_fraction']['score'],
                test_condition=json.dumps(dict(json.loads(row.test_condition), mode=mode), sort_keys=True))
            rows.append(response)
            if trace['source_file'] is not None:
                restored_traces.append(dict(response_id=response['response_id'], trace=json.dumps(dict(
                    source_file=trace['source_file'], record=trace['record'], measurement=mode,
                    paper_cell=published.get((model, key[1], mode))))))
    projected = {name: frame for name, frame in tables.items() if name != 'attempts'}
    projected.update(items=pd.DataFrame(parents.values()), responses=pd.DataFrame(rows), traces=pd.DataFrame(restored_traces))
    projected['benchmarks'] = tables['benchmarks'].copy()
    projected['benchmarks']['response_scale'] = json.dumps(dict(kind='interval', min=0., max=1., direction='higher_is_better'))
    return _cybench(directory, projected, metadata, source_records=source)
