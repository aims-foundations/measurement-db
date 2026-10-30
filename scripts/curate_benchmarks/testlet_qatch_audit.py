"""Reconcile native QATCH predicates and preserve every original metric and context."""

import ast
from collections import Counter
import hashlib
import json
import unicodedata

import pandas as pd


def native_metrics(root):
    """Load the pinned metric code, replacing unsafe expression evaluation only."""
    namespace = dict(eval=ast.literal_eval)
    exec(compile((root / 'abstract_metric.py').read_text(), str(root / 'abstract_metric.py'), 'exec'), namespace)
    metrics = {}
    for name, cls in [('cell_recall', 'CellRecallTag'), ('tuple_constraint', 'TupleConstraintTag')]:
        tree = ast.parse((root / (name + '_tag.py')).read_text())
        tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.level)]
        exec(compile(tree, str(root / (name + '_tag.py')), 'exec'), namespace)
        metrics[name] = namespace[cls]()
    return metrics


def native_predicates(metrics, metric, target_text, prediction, released):
    if metric not in metrics or released is None or not isinstance(prediction, str) or not prediction.startswith('['):
        return []
    target = ast.literal_eval(target_text)
    try:
        parsed = metrics[metric].check_chatgpt_result(prediction)
        if not target or not parsed:
            return []
        if any(isinstance(cell, (list, tuple)) for row in parsed for cell in row):
            return []
        score = metrics[metric].evaluate_single_test_metric(target, prediction)
    except (ValueError, SyntaxError, TypeError, NameError):
        return []
    if score != released:
        return []
    target = [[str(value).replace('\n', '').strip().lower() for value in row] for row in target]
    parsed = [[str(value).strip().lower() for value in row] for row in parsed]
    if metric == 'cell_recall':
        counts = Counter(value for row in target for value in row)
        observed = {value for row in parsed for value in row}
        return [(dict(value=value, target_occurrences=count), float(value in observed), count)
            for value, count in counts.items()]
    counts = Counter(map(tuple, target))
    observed = Counter(map(tuple, parsed))
    return [(dict(row=list(row), target_occurrences=count), float(observed[row] == count), 1)
        for row, count in counts.items()]


def verify(directory, tables, metadata, source=None):
    from .native_result_audits import _check, _qatch, _qatch_sources
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='QATCH native predicates and original publication records')
    source = _qatch_sources(directory, metadata) if source is None else source
    metrics = None
    item_table = tables['items'].astype(object).where(tables['items'].notna(), None)
    items = item_table.set_index('item_id').to_dict('index')
    groups = tables['responses'].groupby('attempt_id', sort=False)
    _check(set(groups.groups), set(tables['attempts'].attempt_id), 'QATCH every recorded attempt remains observed')
    parents, responses, traces, used, seen = {}, [], [], set(), set()
    counters = Counter(source_expanded_metric_measurements=0, source_expanded_predicates=0, source_weighted_cell_comparisons=0)
    for attempt in tables['attempts'].itertuples():
        trace = json.loads(attempt.trace)
        name, index, alias = trace['source_file'], trace['source_row'], trace['source_model']
        key = name, index, alias
        _check(key not in seen, True, 'QATCH unique publication row/model association')
        seen.add(key)
        _check(attempt.raw_attempt_id, json.dumps(list(key)), 'QATCH original publication-row attempt identity')
        original = source['records'][name, index]
        _check(trace['record'], original, 'QATCH full unmodified original CSV fields')
        children = groups.get_group(attempt.attempt_id)
        channels = {}
        for row in children.itertuples():
            spec = json.loads(json.loads(items[row.item_id]['verifier'])['spec'])
            metric = spec.get('original_protocol', spec)['metric']
            channels.setdefault(metric, []).append(row)
            _check(row.subject_id, attempt.subject_id, 'QATCH same model on every metric/check')
            _check((row.trial, row.test_condition, pd.isna(row.interactors)),
                (1, name + ':' + str(index) + ';model=' + alias, True), 'QATCH source occasion without invented repetitions')
        measurements = {m['metric']: m['grade'] for m in trace['measurements']}
        _check(len(measurements), len(trace['measurements']), 'QATCH unique original metric evidence')
        _check(set(measurements), set(channels), 'QATCH every original metric has an observation')
        prediction = original.get('predictions_query_result', original.get(alias + '_predictions', original.get('predictions')))
        if metrics is None and isinstance(prediction, str) and prediction.startswith('['):
            metrics = native_metrics(directory / 'raw/protocol/src/metrics')
        reference = json.dumps(dict(sql=original['query'], released_target=original['query_result']), ensure_ascii=False, sort_keys=True)
        for metric, grade in measurements.items():
            protocol = metadata['grading']['verifiers'][metric]
            _check(grade, source['measurements'][name, index, alias, metric]['score'], 'QATCH exact original metric')
            selected = channels[metric]
            first = selected[0]
            prototype = items[first.item_id]
            _check(prototype['testlet_id'], hashlib.sha256(unicodedata.normalize('NFC', prototype['content']).encode()).hexdigest(),
                'QATCH testlet follows the complete original question')
            checks = native_predicates(metrics or {}, metric, original['query_result'], prediction, grade)
            expected = {}
            for target, value, weight in checks:
                text = json.dumps(target, ensure_ascii=False, sort_keys=True)
                identity = hashlib.sha256(text.encode()).hexdigest()
                criterion = dict(reference_answer=text, rule=json.dumps(dict(metric=metric, aggregation_weight=weight,
                    rule='Target-value membership in the normalized returned table.' if metric == 'cell_recall'
                        else 'The normalized reference row occurs exactly its target number of times.'), sort_keys=True),
                    response_scale=dict(kind='discrete', values=[0, 1], direction='higher_is_better'))
                expected[identity] = criterion, value
                if metric == 'cell_recall':
                    counters['source_weighted_cell_comparisons'] += weight
            actual = Counter()
            for row in selected:
                item = items[row.item_id]
                for field in ['content', 'item_features', 'asset_manifest', 'testlet_id']:
                    _check(item[field], prototype[field], 'QATCH common complete task context ' + field)
                if checks:
                    component = item['raw_item_id'].rsplit('::', 1)[1]
                    criterion, value = expected[component]
                    _check(json.loads(item['grading_criterion']), criterion, 'QATCH native predicate, weight and reference')
                    _check(json.loads(item['verifier']), dict(**{'class': 'exact_matcher'}, spec=json.dumps(dict(
                        kind='reconciled_native_table_predicate', original_protocol=protocol), sort_keys=True)), 'QATCH pinned native predicate protocol')
                    _check(row.response, value, 'QATCH independently replayed predicate')
                    actual[component] += 1
                else:
                    _check(len(selected), 1, 'QATCH retained scalar when finer evidence is unavailable or inconsistent')
                    _check(None if pd.isna(row.response) else row.response, grade, 'QATCH unchanged fallback grade')
                used.add(row.item_id)
            if checks:
                _check(actual, Counter({name: 1 for name in expected}), 'QATCH distinct predicate coverage without repeated equal-value items')
                counters['source_expanded_metric_measurements'] += 1
                counters['source_expanded_predicates'] += len(checks)
            identity = json.dumps([name, index, metric])
            criterion = dict(reference_answer=reference, rule=protocol['rule'], response_scale=protocol['response_scale'])
            verifier = dict(**{'class': 'exact_matcher'}, spec=json.dumps(protocol, sort_keys=True))
            if not checks:
                _check((json.loads(prototype['grading_criterion']), json.loads(prototype['verifier'])),
                    (criterion, verifier), 'QATCH exact retained original grading definition')
            parent = dict(prototype, item_id=identity, raw_item_id=name + ':' + str(index) + ':' + metric,
                grading_criterion=json.dumps(criterion), verifier=json.dumps(verifier))
            if identity in parents:
                _check(parent, parents[identity], 'QATCH same original item across released models')
            parents[identity] = parent
            response = first._asdict(); response.pop('Index', None)
            response.update(response_id=attempt.attempt_id + ':' + metric, item_id=identity, attempt_id=None, response=grade)
            responses.append(response)
            traces.append(dict(response_id=response['response_id'], trace=json.dumps(dict(source_file=name,
                source_row=index, source_model=alias, record=original), ensure_ascii=False)))
    _check(used, set(items), 'QATCH no unused or invented child definitions')
    projected = {key: frame for key, frame in tables.items() if key != 'attempts'}
    projected.update(items=pd.DataFrame(parents.values()), responses=pd.DataFrame(responses), traces=pd.DataFrame(traces))
    details = _qatch(directory, projected, metadata, source=source)
    return dict(details, **counters, source_attempts=len(tables['attempts']), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=int(tables['items'].testlet_id.nunique()))
