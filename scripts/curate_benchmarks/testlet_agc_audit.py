"""Reconcile Connections checks, generation identities and all original AGC evidence."""

import ast
import json
import re
import typing
from collections import Counter

import pandas as pd


def native_matcher(path):
    """Load only the pinned pure normalization helpers, without the HELM runtime."""
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GroupMatchScoreMetric']
    tree.body[0].body = [node for node in tree.body[0].body
        if not (isinstance(node, ast.FunctionDef) and node.name == 'evaluate_generation')]
    namespace = dict(vars(typing), re=re, Metric=object)
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['GroupMatchScoreMetric']


def verify(directory, tables, metadata):
    from .native_result_audits import _agc_bench, _agc_source_records, _check, _features
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='AGC original generations and group judgments')
    parameters = metadata['build']['parameters']
    root = directory / 'raw'
    matcher = native_matcher(root / parameters['layout']['release'] / 'metrics/group_match_score_metric.py')
    source = _agc_source_records(directory, metadata)
    native, prompts = source[:2]
    bank = json.loads((root / parameters['layout']['nyt_connections'] / 'ConnectionsFinalDataset.json').read_text())
    references = {parameters['templates']['nyt_connections'].format(words=', '.join(row['words'])).strip():
        '\n'.join(sorted(', '.join(sorted(group['words'])) for group in row['answers'])) for row in bank}
    item_table = tables['items'].astype(object).where(tables['items'].notna(), None)
    items = item_table.set_index('item_id').to_dict('index')
    observations = tables['responses'].groupby('attempt_id', sort=False)
    _check(set(observations.groups), set(tables['attempts'].attempt_id), 'AGC every physical execution is observed')
    parents, responses, traces, used, seen = {}, [], [], set(), set()
    group_count = partial = 0
    for attempt in tables['attempts'].itertuples():
        trace = json.loads(attempt.trace)
        key = trace['source_file'], trace['source_row']
        original = native[key]
        _check(key not in seen, True, 'AGC one physical execution per source generation')
        seen.add(key)
        _check(trace['generation'], original, 'AGC full original generation')
        _check(attempt.raw_attempt_id, json.dumps([original['model'], original['benchmark'], original['instance_id']]),
            'AGC stable original execution identity')
        children = observations.get_group(attempt.attempt_id)
        by_protocol = {}
        for row in children.itertuples():
            item = items[row.item_id]
            spec = json.loads(item['verifier'])
            channel = _features(item['item_features'])['grading_channel']
            by_protocol.setdefault((channel, spec.get('judge')), []).append(row)
            _check(row.subject_id, attempt.subject_id, 'AGC unchanged model attribution')
            _check((row.trial, row.test_condition, pd.isna(row.interactors)),
                (1, 'source_run=' + original['source_run_dir'], True), 'AGC original occasion')
        visited = set()
        for measurement in trace['measurements']:
            rater = measurement['rater']
            protocol = ('panel_' if rater else 'canonical_') + original['benchmark']
            selected = by_protocol[protocol, parameters['panel_raters'].get(rater)]
            _check(rater not in visited, True, 'AGC one original measurement per judge')
            visited.add(rater)
            first = selected[0]
            prototype = items[first.item_id]
            raw = original['benchmark'] + '::' + original['instance_id']
            expected_testlet = re.sub(r'_ttcw_[0-9]+$', '', raw)
            for row in selected:
                item = items[row.item_id]
                for field in ['content', 'item_features', 'asset_manifest']:
                    _check(item[field], prototype[field], 'AGC common complete task field ' + field)
                _check(item['testlet_id'], expected_testlet, 'AGC original shared task')
                used.add(row.item_id)
            if protocol == 'canonical_nyt_connections':
                prompt = prompts['nyt_connections', original['instance_id']].strip().removeprefix(
                    parameters['prefixes'].get('nyt_connections', ''))
                reference = references[prompt]
                target = matcher._parse_groups(reference)
                predicted = matcher._parse_groups(original['completion'] or '')
                remaining = [(i, matcher._normalize_group(group)) for i, group in enumerate(target)]
                grades = [0.] * 4
                for group in predicted:
                    normalized = matcher._normalize_group(group)
                    for index, (position, value) in enumerate(remaining):
                        if normalized == value:
                            grades[position] = 1.
                            remaining.pop(index)
                            break
                _check(float(all(grades)), original['canonical_score'], 'AGC exact native full-wall replay')
                _check(len(selected), 4, 'AGC all four native groups')
                actual = Counter()
                for row in selected:
                    item = items[row.item_id]
                    criterion = json.loads(item['grading_criterion'])
                    rule = json.loads(criterion['rule'])
                    position = rule['group_position']
                    _check(position in range(1, 5), True, 'AGC source group position')
                    expected = dict(reference_answer=json.dumps(matcher._normalize_group(target[position - 1])),
                        rule=json.dumps(dict(group_position=position, source_reference=reference,
                            rule='Recover this reference group with the native four-line group matcher; each predicted group can match one reference group.'), sort_keys=True),
                        response_scale=dict(kind='discrete', values=[0, 1], direction='higher_is_better'))
                    _check(criterion, expected, 'AGC exact source group and checker definition')
                    _check(json.loads(item['verifier']), dict(**{'class': 'exact_matcher'},
                        spec=json.dumps(dict(kind='native_group_match', original_protocol=metadata['grading']['verifiers'][protocol]), sort_keys=True)),
                        'AGC pinned group-matching protocol')
                    _check(item['raw_item_id'], raw + '::group_' + str(position), 'AGC source check identity')
                    _check(row.response, grades[position - 1], 'AGC native group verdict')
                    actual[position] += 1
                _check(actual, Counter({i: 1 for i in range(1, 5)}), 'AGC exact group coverage')
                spec = metadata['grading']['verifiers'][protocol]
                parent = dict(prototype, raw_item_id=raw, grading_criterion=json.dumps(dict(reference_answer=reference,
                    rule=spec['rule'], response_scale=spec['response_scale'])),
                    verifier=json.dumps(dict(**{'class': 'exact_matcher'}, spec=json.dumps(spec['implementation'], sort_keys=True))))
                group_count += 4
                partial += 0 < sum(grades) < 4
            else:
                _check(len(selected), 1, 'AGC retained original scalar measurement')
                _check(None if pd.isna(first.response) else first.response, measurement['grade'], 'AGC unmodified original scalar')
                _check(prototype['raw_item_id'], raw, 'AGC original item label')
                parent = dict(prototype)
            identity = json.dumps([raw, protocol, rater])
            parent['item_id'] = identity
            if identity in parents:
                _check(parent, parents[identity], 'AGC original task and grading identity across models')
            parents[identity] = parent
            response = first._asdict()
            response.pop('Index', None)
            response.update(response_id=attempt.attempt_id + ':' + rater, item_id=identity, attempt_id=None, response=measurement['grade'])
            responses.append(response)
            traces.append(dict(response_id=response['response_id'], trace=json.dumps(dict(
                source_file=trace['source_file'], source_row=trace['source_row'], generation=original,
                grade_status=measurement['grade_status'], panel_rating=measurement['panel_rating']), ensure_ascii=False)))
        _check(len(visited), len(by_protocol), 'AGC no unaccounted grading channels')
    _check((seen, used), (set(native), set(items)), 'AGC complete source execution and item coverage')
    projected = {key: frame for key, frame in tables.items() if key != 'attempts'}
    projected.update(items=pd.DataFrame(parents.values()), responses=pd.DataFrame(responses), traces=pd.DataFrame(traces))
    details = _agc_bench(directory, projected, metadata, source_records=source)
    return dict(details, source_attempts=len(tables['attempts']), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=len({item['testlet_id'] for item in items.values()}),
        source_connection_group_observations=group_count, source_partial_connection_runs=partial)
